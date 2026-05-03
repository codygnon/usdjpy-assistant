"""v1 -> v2 dispatch bridge — Phase 9 Step 9.

Called by `api/autonomous_fillmore.py::tick_autonomous_fillmore` when the
engine flag is "v2". Builds a v2 Snapshot from the v1 tick inputs (best
effort), invokes the v2 orchestrator, and returns.

Stage 1 wiring is intentionally conservative. The bridge fills the blocking
telemetry that can be derived safely from v1's adapter, DB, tick, and bar cache.
When any required live input is absent, fields stay None and v2's strike/halt
policy fails closed before an LLM call.

This module is the trust boundary between v1's pile-of-globals world and
v2's strict dataclass contract. It logs loudly on dispatch so operators
can see when v2 is handling ticks.
"""
from __future__ import annotations

import logging
import math
import uuid
from pathlib import Path
from typing import Any, Iterable, Optional

from .llm_client import OpenAILlmClient
from .orchestrator import OrchestrationResult, run_decision
from .persistence import init_v2_schema
from .snapshot import Snapshot, new_snapshot_id, now_utc_iso
from .telemetry import (
    OpenPositionRow,
    build_level_age_metadata,
    build_level_packet,
    classify_volatility_regime,
    fetch_closed_trades_for_rolling,
    open_lots_by_side,
    pip_value_per_lot,
    rolling_pnl,
    unrealized_pnl_by_side,
)

logger = logging.getLogger(__name__)


def _as_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        f = float(value)
        return f if math.isfinite(f) else None
    except Exception:
        return None


def _get(obj: Any, *names: str) -> Any:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj.get(name)
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _extract_equity(adapter: Any) -> Optional[float]:
    if adapter is None:
        return None
    try:
        info = adapter.get_account_info()
    except Exception:
        logger.exception("v2 bridge: account info fetch failed")
        return None
    return _as_float(_get(info, "equity", "nav", "NAV", "balance"))


def _position_row(raw: Any) -> Optional[OpenPositionRow]:
    units = _as_float(_get(raw, "currentUnits", "current_units", "units", "initialUnits"))
    if units is None:
        volume = _as_float(_get(raw, "volume", "lots"))
        if volume is not None:
            direction = str(_get(raw, "side", "type", "direction") or "").lower()
            sign = -1 if "sell" in direction or direction in ("1", "short") else 1
            units = sign * volume * 100_000.0
    if units is None or abs(units) <= 0:
        return None
    side = "buy" if units > 0 else "sell"
    pnl = _as_float(_get(raw, "unrealizedPL", "unrealized_pl", "unrealized_pnl", "profit")) or 0.0
    return OpenPositionRow(side=side, units=units, unrealized_pnl_usd=pnl)


def _extract_positions(
    *,
    adapter: Any,
    profile: Any,
    open_positions: Optional[Iterable[Any]] = None,
) -> Optional[list[OpenPositionRow]]:
    raw_positions = open_positions
    if raw_positions is None and adapter is not None:
        try:
            raw_positions = adapter.get_open_positions(getattr(profile, "symbol", "USD_JPY"))
        except Exception:
            logger.exception("v2 bridge: open positions fetch failed")
            return None
    if raw_positions is None:
        return None
    rows = [_position_row(p) for p in raw_positions]
    return [r for r in rows if r is not None]


def _iter_recent_rows(df: Any, n: int = 120) -> list[Any]:
    if df is None:
        return []
    try:
        tail = df.tail(n)
    except Exception:
        tail = df[-n:] if isinstance(df, list) else df
    try:
        return [row for _, row in tail.iterrows()]
    except Exception:
        return list(tail) if isinstance(tail, (list, tuple)) else []


def _row_value(row: Any, key: str) -> Optional[float]:
    if isinstance(row, dict):
        return _as_float(row.get(key))
    try:
        return _as_float(row[key])
    except Exception:
        return _as_float(getattr(row, key, None))


def _row_time(row: Any) -> Optional[str]:
    for key in ("time", "timestamp", "datetime", "created_utc"):
        if isinstance(row, dict) and row.get(key):
            return str(row.get(key))
        try:
            val = row[key]
            if val is not None:
                return str(val)
        except Exception:
            pass
    return None


def _bars(data_by_tf: Optional[dict[str, Any]], *tfs: str) -> list[Any]:
    if not data_by_tf:
        return []
    for tf in tfs:
        rows = _iter_recent_rows(data_by_tf.get(tf), 160)
        if rows:
            return rows
    return []


def _derive_side(data_by_tf: Optional[dict[str, Any]], mid: Optional[float]) -> Optional[str]:
    rows = _bars(data_by_tf, "M5", "M1", "M15")
    closes = [_row_value(r, "close") for r in rows[-12:]]
    closes = [c for c in closes if c is not None]
    if len(closes) >= 2:
        return "buy" if closes[-1] >= closes[0] else "sell"
    return "buy" if mid is not None else None


def _derive_sl_tp(data_by_tf: Optional[dict[str, Any]], pip: float) -> tuple[Optional[float], Optional[float]]:
    rows = _bars(data_by_tf, "M5", "M1", "M15")
    ranges: list[float] = []
    for row in rows[-14:]:
        high = _row_value(row, "high")
        low = _row_value(row, "low")
        if high is not None and low is not None and high >= low:
            ranges.append((high - low) / pip)
    if not ranges:
        return 8.0, 12.0
    atr_proxy = sum(ranges) / len(ranges)
    sl = round(max(5.0, min(14.0, atr_proxy * 1.25)), 1)
    tp = round(max(sl * 1.2, sl + 3.0), 1)
    return sl, tp


def _derive_timeframe_alignment(
    data_by_tf: Optional[dict[str, Any]],
    proposed_side: Optional[str],
) -> Optional[str]:
    if proposed_side not in ("buy", "sell"):
        return None
    signs: list[int] = []
    for tf, lookback in (("M1", 12), ("M5", 12), ("H1", 6)):
        rows = _bars(data_by_tf, tf)
        closes = [_row_value(r, "close") for r in rows[-lookback:]]
        closes = [c for c in closes if c is not None]
        if len(closes) >= 2:
            signs.append(1 if closes[-1] >= closes[0] else -1)
    if not signs:
        return None
    if all(s > 0 for s in signs):
        direction = "buy"
    elif all(s < 0 for s in signs):
        direction = "sell"
    else:
        return "mixed"
    return f"aligned_{direction}" if direction == proposed_side else f"aligned_{direction}"


def _derive_atr_m5_pips(data_by_tf: Optional[dict[str, Any]], pip: float) -> Optional[float]:
    rows = _bars(data_by_tf, "M5")
    ranges: list[float] = []
    for row in rows[-14:]:
        high = _row_value(row, "high")
        low = _row_value(row, "low")
        if high is not None and low is not None and high >= low:
            ranges.append((high - low) / pip)
    return (sum(ranges) / len(ranges)) if ranges else None


def _sessions(created_utc: str) -> tuple[list[str], Optional[str]]:
    try:
        from .legacy_rationale_parser import derive_sessions
        return derive_sessions(created_utc)
    except Exception:
        return [], None


def _nearest_levels(
    data_by_tf: Optional[dict[str, Any]],
    *,
    mid: Optional[float],
    pip: float,
    proposed_side: Optional[str],
) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]], Optional[float], Optional[int], Optional[str]]:
    if mid is None:
        return None, None, None, None, None
    rows = _bars(data_by_tf, "M5", "M1", "M15")
    highs: list[tuple[float, Optional[str]]] = []
    lows: list[tuple[float, Optional[str]]] = []
    for row in rows:
        high = _row_value(row, "high")
        low = _row_value(row, "low")
        ts = _row_time(row)
        if high is not None:
            highs.append((high, ts))
        if low is not None:
            lows.append((low, ts))
    supports = [(p, ts) for p, ts in lows if p <= mid]
    resistances = [(p, ts) for p, ts in highs if p >= mid]
    support = max(supports, key=lambda x: x[0], default=(None, None))
    resistance = min(resistances, key=lambda x: x[0], default=(None, None))

    def level(price: Optional[float], origin: str) -> Optional[dict[str, Any]]:
        if price is None:
            return None
        return {"price": price, "quality_score": 65.0, "structural_origin": origin}

    nearest_support = level(support[0], "stage1_recent_bar_support")
    nearest_resistance = level(resistance[0], "stage1_recent_bar_resistance")
    blocker: Optional[float] = None
    if proposed_side == "buy" and resistance[0] is not None:
        blocker = round((resistance[0] - mid) / pip, 1)
    elif proposed_side == "sell" and support[0] is not None:
        blocker = round((mid - support[0]) / pip, 1)
    touch_price = support[0] if proposed_side == "buy" else resistance[0]
    touches = None
    last_touch = support[1] if proposed_side == "buy" else resistance[1]
    if touch_price is not None:
        all_prices = lows if proposed_side == "buy" else highs
        touches = sum(1 for p, _ in all_prices if abs(p - touch_price) <= 2.0 * pip)
    return nearest_support, nearest_resistance, blocker, touches, last_touch


def _resolve_db_path(profile_name: str, db_path: Optional[Path]) -> Optional[Path]:
    if db_path is not None:
        return Path(db_path)
    try:
        from api.main import _suggestions_db_path
        return Path(_suggestions_db_path(profile_name))
    except Exception:
        logger.exception("v2 bridge: could not resolve suggestions DB path")
        return None


def build_snapshot_from_v1_inputs(
    *,
    profile: Any,
    tick: Any,
    profile_name: str = "default",
    adapter: Any = None,
    db_path: Optional[Path] = None,
    data_by_tf: Optional[dict[str, Any]] = None,
    open_positions: Optional[Iterable[Any]] = None,
    proposed_side: Optional[str] = None,
    spread_pips: Optional[float] = None,
    sl_pips: Optional[float] = None,
    tp_pips: Optional[float] = None,
) -> Snapshot:
    """Best-effort projection of v1's live inputs into a v2 Snapshot."""
    pip = float(getattr(profile, "pip_size", 0.01) or 0.01)
    bid = float(getattr(tick, "bid", 0.0))
    ask = float(getattr(tick, "ask", 0.0))
    mid = (bid + ask) / 2.0 if (bid and ask) else None
    spread = ((ask - bid) / pip) if (pip and bid and ask) else spread_pips
    created = now_utc_iso()
    side = proposed_side or _derive_side(data_by_tf, mid)
    sl, tp = (sl_pips, tp_pips)
    if sl is None or tp is None:
        sl, tp = _derive_sl_tp(data_by_tf, pip)

    positions = _extract_positions(adapter=adapter, profile=profile, open_positions=open_positions)
    open_buy = open_sell = unrealized_buy = unrealized_sell = None
    if positions is not None:
        open_buy, open_sell = open_lots_by_side(positions)
        unrealized_buy, unrealized_sell = unrealized_pnl_by_side(positions)

    rolling_trade = rolling_weighted = None
    if db_path is not None:
        try:
            rolling_trade, rolling_weighted = rolling_pnl(
                fetch_closed_trades_for_rolling(Path(db_path), profile_name)
            )
        except Exception:
            logger.exception("v2 bridge: rolling P&L fetch failed")

    pip_value = None
    if mid is not None:
        try:
            pip_value = pip_value_per_lot(mid)
        except Exception:
            logger.exception("v2 bridge: pip value calculation failed")

    nearest_support, nearest_resistance, blocker, touches, last_touch = _nearest_levels(
        data_by_tf, mid=mid, pip=pip, proposed_side=side,
    )
    level_packet = None
    level_age = None
    if mid is not None and side in ("buy", "sell"):
        level_packet = build_level_packet(
            proposed_side=side,
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
            tick_mid=mid,
            profit_path_blocker_pips=blocker,
        )
        if level_packet is not None:
            level_age = build_level_age_metadata(
                touch_count=touches,
                broken_then_reclaimed=False,
                last_touch_utc=last_touch,
            )
    active_sessions, session_overlap = _sessions(created)

    return Snapshot(
        snapshot_id=new_snapshot_id(),
        created_utc=created,
        tick_mid=mid, tick_bid=bid or None, tick_ask=ask or None,
        spread_pips=spread,
        proposed_side=side,
        sl_pips=sl, tp_pips=tp,
        account_equity=_extract_equity(adapter),
        open_lots_buy=open_buy, open_lots_sell=open_sell,
        unrealized_pnl_buy=unrealized_buy, unrealized_pnl_sell=unrealized_sell,
        pip_value_per_lot=pip_value,
        risk_after_fill_usd=None,
        rolling_20_trade_pnl=rolling_trade,
        rolling_20_lot_weighted_pnl=rolling_weighted,
        level_packet=level_packet, level_age_metadata=level_age,
        timeframe_alignment=_derive_timeframe_alignment(data_by_tf, side),
        macro_bias="neutral",
        catalyst_category="structure_only",
        active_sessions=active_sessions,
        session_overlap=session_overlap,
        volatility_regime=classify_volatility_regime(_derive_atr_m5_pips(data_by_tf, pip)),
    )


def dispatch_v2_tick(
    *,
    profile: Any,
    profile_name: str,
    state_path: Path,
    tick: Any,
    adapter: Any = None,
    store: Any = None,
    data_by_tf: Optional[dict[str, Any]] = None,
    open_positions: Optional[Iterable[Any]] = None,
    db_path: Optional[Path] = None,
    llm_client: Any = None,
    model: str = "gpt-5.4-mini",
    stage: str = "paper",
    allow_non_paper: bool = False,
) -> OrchestrationResult:
    """Build a Snapshot from v1's inputs and run the v2 orchestrator.

    Returns the OrchestrationResult so callers (including v1's tick hook)
    can decide what to log. v1's tick is fire-and-forget — the dispatch
    return value is mostly for tests + operator visibility.
    """
    db_path = _resolve_db_path(profile_name, db_path)
    if db_path is not None:
        try:
            from api import suggestion_tracker
            suggestion_tracker.init_db(db_path)
            init_v2_schema(db_path)
        except Exception:
            logger.exception("v2 bridge: v2 persistence schema init failed")
    snap = build_snapshot_from_v1_inputs(
        profile=profile,
        profile_name=profile_name,
        tick=tick,
        adapter=adapter,
        db_path=db_path,
        data_by_tf=data_by_tf,
        open_positions=open_positions,
    )
    profile_dir = Path(state_path).parent
    if stage != "paper" and not allow_non_paper:
        return OrchestrationResult(
            suggestion_id=f"v2-{uuid.uuid4().hex[:16]}",
            final_decision="halt",
            reason=f"paper_mode_guard: refusing v2 dispatch with stage={stage!r}",
            snapshot=snap,
            halt_active=True,
        )
    client = llm_client or OpenAILlmClient()
    logger.info(
        "v2 dispatch: profile=%s side=%s mid=%s spread=%s",
        profile_name, snap.proposed_side, snap.tick_mid, snap.spread_pips,
    )
    return run_decision(
        snap, llm_client=client, profile_dir=profile_dir,
        db_path=db_path, profile=profile_name, model=model, stage=stage,  # type: ignore[arg-type]
    )
