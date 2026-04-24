"""Legacy **in-app** simulated fills for Autonomous Fillmore (older builds).

Autonomous **PAPER** mode now places real orders via the OANDA adapter when the
profile uses ``oanda_environment='practice'``. This module remains for any
``trades`` rows still carrying ``config_json.paper == true`` from prior versions.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

PAPER_ORDER_PREFIX = "paper:"


def paper_order_id(suggestion_id: str) -> str:
    sid = (suggestion_id or "").strip() or uuid.uuid4().hex
    return f"{PAPER_ORDER_PREFIX}{sid[:32]}"


def _cfg(trade_row: dict[str, Any]) -> dict[str, Any]:
    raw = trade_row.get("config_json")
    if not raw:
        return {}
    try:
        d = json.loads(str(raw))
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def is_paper_trade(trade_row: dict[str, Any]) -> bool:
    return bool(_cfg(trade_row).get("paper"))


def _next_paper_position_id(state_path: Path) -> int:
    from api.autonomous_fillmore import _load_state, _save_state

    state = _load_state(state_path)
    auto = state.setdefault("autonomous_fillmore", {})
    rt = auto.setdefault("runtime", {})
    cur = int(rt.get("paper_next_position_id") or 9_000_000_000)
    nxt = cur + 1
    rt["paper_next_position_id"] = nxt
    state["autonomous_fillmore"] = auto
    _save_state(state_path, state)
    return nxt


def _usd_profit_usdjpy(*, pips: float, lots: float, mid: float, pip_size: float) -> float:
    if mid <= 0 or lots <= 0:
        return 0.0
    return float(pips * pip_size * 100_000.0 * lots / mid)


def _merge_cfg_json(trade_row: dict[str, Any], patch: dict[str, Any]) -> str:
    base = _cfg(trade_row)
    base.update(patch)
    return json.dumps(base)


def _ema_series(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


def _last_completed_ema(df: pd.DataFrame, period: int) -> Optional[float]:
    if df is None or df.empty or len(df) < period + 2:
        return None
    try:
        closes = df["close"].astype(float).iloc[:-1]
        return float(_ema_series(closes, period).iloc[-1])
    except Exception:
        return None


def _last_completed_close(df: pd.DataFrame) -> Optional[float]:
    if df is None or df.empty or len(df) < 2:
        return None
    try:
        return float(df["close"].astype(float).iloc[-2])
    except Exception:
        return None


def finalize_paper_close(
    *,
    profile_name: str,
    state_path: Path,
    store: Any,
    trade_row: dict[str, Any],
    exit_price: float,
    exit_reason: str,
    pip_size: float,
    mid: float,
) -> None:
    from api import autonomous_fillmore, suggestion_tracker
    from api.main import _suggestions_db_path

    trade_id = str(trade_row["trade_id"])
    cfgj = _cfg(trade_row)
    paper_oid = str(cfgj.get("paper_order_id") or "")
    suggestion_id = str(cfgj.get("suggestion_id") or "")
    side = str(trade_row.get("side") or "buy").lower()
    entry = float(trade_row.get("entry_price") or 0.0)
    lots = float(trade_row.get("size_lots") or 0.0)
    partial = float(cfgj.get("paper_partial_pnl_usd") or 0.0)

    if entry <= 0 or lots <= 0:
        pips = 0.0
        leg = 0.0
    else:
        pips = ((exit_price - entry) / pip_size) if side == "buy" else ((entry - exit_price) / pip_size)
        leg = _usd_profit_usdjpy(pips=pips, lots=lots, mid=mid, pip_size=pip_size)
    total_profit = partial + leg

    now = datetime.now(timezone.utc).isoformat()
    store.close_trade(
        trade_id=trade_id,
        updates={
            "exit_price": float(exit_price),
            "exit_timestamp_utc": now,
            "exit_reason": exit_reason,
            "pips": round(float(pips), 2),
            "profit": round(float(total_profit), 4),
        },
    )

    db_path = _suggestions_db_path(profile_name)
    closed = False
    if paper_oid:
        closed = suggestion_tracker.mark_closed(
            db_path,
            oanda_order_id=paper_oid,
            exit_price=float(exit_price),
            pnl=float(total_profit),
            pips=float(pips),
            closed_at=now,
        )
    if not closed and suggestion_id:
        suggestion_tracker.mark_closed_by_suggestion_id(
            db_path,
            suggestion_id=suggestion_id,
            exit_price=float(exit_price),
            pnl=float(total_profit),
            pips=float(pips),
            closed_at=now,
        )

    try:
        cfg = autonomous_fillmore.get_config(state_path)
        autonomous_fillmore.record_trade_outcome(state_path, float(total_profit), cfg)
    except Exception:
        pass

    try:
        from api.fillmore_learning import maybe_generate_trade_reflection

        closed_row = dict(trade_row)
        closed_row.update(
            {
                "exit_price": float(exit_price),
                "exit_timestamp_utc": now,
                "exit_reason": exit_reason,
                "pips": round(float(pips), 2),
                "profit": round(float(total_profit), 4),
            }
        )
        maybe_generate_trade_reflection(
            profile_name=profile_name,
            trade_row=closed_row,
            db_path=db_path,
        )
    except Exception:
        pass


def place_paper_market_fill(
    *,
    profile: Any,
    profile_name: str,
    state_path: Path,
    store: Any,
    suggestion: dict[str, Any],
    tick: Any,
    pip_size: float,
    stamp_tracker: bool = True,
) -> dict[str, Any]:
    from api import suggestion_tracker
    from api.ai_exit_strategies import merge_exit_params, normalize_exit_strategy, trail_mode_for_strategy
    from api.main import _suggestions_db_path

    side = str(suggestion.get("side") or "buy").lower()
    lots = float(suggestion.get("lots") or 0)
    sl = suggestion.get("sl")
    tp = suggestion.get("tp")
    fill_price = float(tick.ask) if side == "buy" else float(tick.bid)
    position_id = _next_paper_position_id(state_path)
    sid = str(suggestion.get("suggestion_id") or "")
    paper_oid = paper_order_id(sid) if sid else f"{PAPER_ORDER_PREFIX}{uuid.uuid4().hex}"

    exit_strat = (suggestion.get("exit_strategy") or "").strip().lower()
    managed_strategy: Optional[str] = None
    managed_params: dict[str, Any] = {}
    if exit_strat and exit_strat != "none":
        managed_strategy = normalize_exit_strategy(exit_strat)
        managed_params = merge_exit_params(managed_strategy, suggestion.get("exit_params"))

    trail_mode = trail_mode_for_strategy(managed_strategy) if managed_strategy else "none"

    trade_id = f"{suggestion_tracker.ENTRY_TYPE_FILLMORE_AUTONOMOUS}:paper:{position_id}:{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d%H%M%S')}"
    cfg_payload: dict[str, Any] = {
        "paper": True,
        "paper_order_id": paper_oid,
        "suggestion_id": suggestion.get("suggestion_id"),
        "source": "autonomous_fillmore",
        "order_type": "market",
        "exit_strategy": managed_strategy,
        "exit_params": managed_params,
    }
    row: dict[str, Any] = {
        "trade_id": trade_id,
        "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "profile": profile.profile_name,
        "symbol": profile.symbol,
        "side": side,
        "policy_type": suggestion_tracker.ENTRY_TYPE_FILLMORE_AUTONOMOUS,
        "config_json": json.dumps(cfg_payload),
        "entry_price": fill_price,
        "stop_price": float(sl) if sl is not None else None,
        "target_price": float(tp) if tp is not None else None,
        "size_lots": lots,
        "notes": f"autonomous_fillmore_paper:{managed_strategy or 'none'}:paper_{position_id}",
        "snapshot_id": None,
        "mt5_order_id": position_id,
        "mt5_deal_id": None,
        "mt5_retcode": 0,
        "mt5_position_id": int(position_id),
        "opened_by": suggestion_tracker.ENTRY_TYPE_FILLMORE_AUTONOMOUS,
        "preset_name": getattr(profile, "active_preset_name", None) or "Autonomous Fillmore (Paper)",
        "entry_type": suggestion_tracker.ENTRY_TYPE_FILLMORE_AUTONOMOUS,
        "breakeven_applied": 0,
        "tp1_partial_done": 0,
        "tp1_triggered": 0,
        "managed_trail_mode": trail_mode or "none",
    }
    if managed_params.get("tp1_pips") is not None:
        row["managed_tp1_pips"] = float(managed_params["tp1_pips"])
    if managed_params.get("tp1_close_pct") is not None:
        row["managed_tp1_close_pct"] = float(managed_params["tp1_close_pct"])
    if managed_params.get("be_plus_pips") is not None:
        row["managed_be_plus_pips"] = float(managed_params["be_plus_pips"])
    store.insert_trade(row)

    if stamp_tracker and sid:
        try:
            db_path = _suggestions_db_path(profile_name)
            suggestion_tracker.log_action(
                db_path,
                suggestion_id=sid,
                action="placed",
                edited_fields={},
                placed_order={
                    "side": side,
                    "price": fill_price,
                    "lots": lots,
                    "sl": float(sl) if sl is not None else None,
                    "tp": float(tp) if tp is not None else None,
                    "time_in_force": "MARKET",
                    "gtd_time_utc": None,
                    "exit_strategy": managed_strategy or "none",
                    "exit_params": managed_params if managed_strategy else {},
                    "autonomous": True,
                    "paper": True,
                    "order_type": "market",
                },
                oanda_order_id=paper_oid,
            )
            suggestion_tracker.mark_filled(
                db_path,
                oanda_order_id=paper_oid,
                fill_price=fill_price,
                filled_at=pd.Timestamp.now(tz="UTC").isoformat(),
                trade_id=trade_id,
            )
        except Exception:
            pass
    return {"order_id": position_id, "status": "filled", "fill_price": fill_price, "paper_order_id": paper_oid}


def log_paper_limit_placed(
    *,
    profile_name: str,
    suggestion: dict[str, Any],
    paper_oid: str,
) -> None:
    from api import suggestion_tracker
    from api.main import _suggestions_db_path

    sid = suggestion.get("suggestion_id")
    if not sid:
        return
    try:
        db_path = _suggestions_db_path(profile_name)
        suggestion_tracker.log_action(
            db_path,
            suggestion_id=str(sid),
            action="placed",
            edited_fields={},
            placed_order={
                "side": suggestion.get("side"),
                "price": suggestion.get("price"),
                "lots": suggestion.get("lots"),
                "sl": suggestion.get("sl"),
                "tp": suggestion.get("tp"),
                "time_in_force": suggestion.get("time_in_force"),
                "gtd_time_utc": suggestion.get("gtd_time_utc"),
                "exit_strategy": suggestion.get("exit_strategy"),
                "exit_params": suggestion.get("exit_params") or {},
                "autonomous": True,
                "paper": True,
                "order_type": "limit",
            },
            oanda_order_id=paper_oid,
        )
    except Exception:
        pass


def register_paper_limit_pending(
    state_path: Path,
    *,
    profile_name: str,
    suggestion: dict[str, Any],
    paper_oid: str,
) -> None:
    from api.autonomous_fillmore import _load_state, _save_state

    state = _load_state(state_path)
    auto = state.setdefault("autonomous_fillmore", {})
    pending = list(auto.get("paper_pending_limits") or [])
    pending.append({
        "paper_order_id": paper_oid,
        "suggestion_id": suggestion.get("suggestion_id"),
        "profile_name": profile_name,
        "side": str(suggestion.get("side") or "buy").lower(),
        "price": float(suggestion.get("price") or 0),
        "lots": float(suggestion.get("lots") or 0),
        "sl": suggestion.get("sl"),
        "tp": suggestion.get("tp"),
        "time_in_force": str(suggestion.get("time_in_force") or "GTD").upper(),
        "gtd_time_utc": suggestion.get("gtd_time_utc"),
        "exit_strategy": suggestion.get("exit_strategy"),
        "exit_params": suggestion.get("exit_params") or {},
        "created_utc": datetime.now(timezone.utc).isoformat(),
    })
    auto["paper_pending_limits"] = pending
    state["autonomous_fillmore"] = auto
    _save_state(state_path, state)


def process_paper_pending_limits(
    *,
    profile: Any,
    profile_name: str,
    state_path: Path,
    store: Any,
    tick: Any,
) -> None:
    from api import suggestion_tracker
    from api.autonomous_fillmore import _load_state, _save_state
    from api.main import _suggestions_db_path

    state = _load_state(state_path)
    auto = state.setdefault("autonomous_fillmore", {})
    pending = list(auto.get("paper_pending_limits") or [])
    if not pending:
        return

    pip = float(getattr(profile, "pip_size", 0.01) or 0.01)
    bid = float(tick.bid)
    ask = float(tick.ask)
    now = datetime.now(timezone.utc)
    db_path = _suggestions_db_path(profile_name)

    kept: list[dict[str, Any]] = []
    for p in pending:
        if str(p.get("profile_name") or "") != profile_name:
            kept.append(p)
            continue
        side = str(p.get("side") or "buy").lower()
        lim = float(p.get("price") or 0)
        gtd = p.get("gtd_time_utc")
        tif = str(p.get("time_in_force") or "GTD").upper()
        paper_oid = str(p.get("paper_order_id") or "")

        expired = False
        if tif == "GTD" and gtd:
            try:
                gt = pd.Timestamp(gtd).to_pydatetime()
                if gt.tzinfo is None:
                    gt = gt.replace(tzinfo=timezone.utc)
                expired = now > gt
            except Exception:
                pass

        filled = (side == "buy" and ask <= lim) or (side == "sell" and bid >= lim)
        if expired and not filled:
            try:
                suggestion_tracker.mark_cancelled_or_expired(
                    db_path,
                    oanda_order_id=paper_oid,
                    status="expired",
                    at=now.isoformat(),
                )
            except Exception:
                pass
            continue
        if not filled:
            kept.append(p)
            continue

        fake_suggestion = {
            "suggestion_id": p.get("suggestion_id"),
            "side": side,
            "lots": float(p.get("lots") or 0),
            "sl": p.get("sl"),
            "tp": p.get("tp"),
            "exit_strategy": p.get("exit_strategy"),
            "exit_params": p.get("exit_params") or {},
        }

        class _T:
            bid = float(lim - pip * 0.02) if side == "buy" else float(lim + pip * 0.02)
            ask = float(lim + pip * 0.02) if side == "buy" else float(lim - pip * 0.02)

        place_paper_market_fill(
            profile=profile,
            profile_name=profile_name,
            state_path=state_path,
            store=store,
            suggestion=fake_suggestion,
            tick=_T(),
            pip_size=pip,
            stamp_tracker=False,
        )
        try:
            suggestion_tracker.mark_filled(
                db_path,
                oanda_order_id=paper_oid,
                fill_price=float(lim),
                filled_at=now.isoformat(),
                trade_id=None,
            )
        except Exception:
            pass

    auto["paper_pending_limits"] = kept
    state["autonomous_fillmore"] = auto
    _save_state(state_path, state)


def tick_paper_open_trades(
    *,
    profile: Any,
    profile_name: str,
    state_path: Path,
    store: Any,
    tick: Any,
    data_by_tf: dict[str, pd.DataFrame],
) -> None:
    open_rows = [dict(r) for r in store.list_open_trades(profile_name)]
    paper = [r for r in open_rows if is_paper_trade(r)]
    if not paper:
        return

    pip = float(getattr(profile, "pip_size", 0.01) or 0.01)
    bid = float(tick.bid)
    ask = float(tick.ask)
    mid = (bid + ask) / 2.0
    spread = ask - bid
    m1_df = data_by_tf.get("M1") if data_by_tf else None
    m5_df = data_by_tf.get("M5") if data_by_tf else None

    for trade_row in paper:
        try:
            _tick_one_paper_trade(
                profile=profile,
                profile_name=profile_name,
                state_path=state_path,
                store=store,
                trade_row=trade_row,
                bid=bid,
                ask=ask,
                mid=mid,
                spread=spread,
                pip=pip,
                m1_df=m1_df,
                m5_df=m5_df,
            )
        except Exception as e:
            print(f"[{profile_name}] paper_fillmore tick error {trade_row.get('trade_id')}: {e}")


def _tick_one_paper_trade(
    *,
    profile: Any,
    profile_name: str,
    state_path: Path,
    store: Any,
    trade_row: dict[str, Any],
    bid: float,
    ask: float,
    mid: float,
    spread: float,
    pip: float,
    m1_df: Optional[pd.DataFrame],
    m5_df: Optional[pd.DataFrame],
) -> None:
    trade_id = str(trade_row["trade_id"])
    side = str(trade_row.get("side") or "buy").lower()
    entry = float(trade_row.get("entry_price") or 0.0)
    lots = float(trade_row.get("size_lots") or 0.0)
    if entry <= 0 or lots <= 0:
        return

    sl0 = trade_row.get("stop_price")
    tp0 = trade_row.get("target_price")
    sl = float(sl0) if sl0 is not None else None
    tp = float(tp0) if tp0 is not None else None

    trail_mode = str(trade_row.get("managed_trail_mode") or "none").lower()
    tp1_done = bool(trade_row.get("tp1_partial_done") or 0)
    tp1_triggered = bool(trade_row.get("tp1_triggered") or 0)
    tp1_pips = float(trade_row.get("managed_tp1_pips") or 6.0)
    tp1_pct = float(trade_row.get("managed_tp1_close_pct") or 70.0)
    be_plus = float(trade_row.get("managed_be_plus_pips") or 0.5)

    cfgj = _cfg(trade_row)
    exit_params = cfgj.get("exit_params") if isinstance(cfgj.get("exit_params"), dict) else {}

    def _cfg_exit(key: str, default: float) -> float:
        try:
            v = exit_params.get(key)
            return float(v) if v is not None else default
        except (TypeError, ValueError):
            return default

    # --- TP1 + BE (two-phase: trigger bar, then partial on next ticks) ---
    if trail_mode != "none" and not tp1_done:
        reached = (side == "buy" and mid >= entry + tp1_pips * pip) or (
            side == "sell" and mid <= entry - tp1_pips * pip
        )
        if reached and not tp1_triggered:
            store.update_trade(trade_id, {"tp1_triggered": 1})
            return

        if reached and tp1_triggered:
            close_frac = max(0.0, min(1.0, tp1_pct / 100.0))
            close_lots = round(lots * close_frac, 4)
            if close_lots >= lots - 1e-6:
                px = ask if side == "buy" else bid
                finalize_paper_close(
                    profile_name=profile_name,
                    state_path=state_path,
                    store=store,
                    trade_row=trade_row,
                    exit_price=px,
                    exit_reason="paper_tp1_full",
                    pip_size=pip,
                    mid=mid,
                )
                return
            if close_lots > 0 and close_lots < lots:
                px = ask if side == "buy" else bid
                pips_part = ((px - entry) / pip) if side == "buy" else ((entry - px) / pip)
                part_pnl = _usd_profit_usdjpy(pips=pips_part, lots=close_lots, mid=mid, pip_size=pip)
                new_lots = max(0.0, round(lots - close_lots, 4))
                new_partial = float(cfgj.get("paper_partial_pnl_usd") or 0.0) + part_pnl
                from api.ai_exit_strategies import compute_post_tp1_stop

                be_sl = compute_post_tp1_stop(
                    trade_side=side,
                    entry_price=entry,
                    pip_size=pip,
                    tp1_pips=tp1_pips,
                    spread_pips=spread / pip if pip > 0 else 0.0,
                    be_plus_pips=be_plus,
                    lock_in_fraction=float(_cfg_exit("tp1_lock_in_fraction", 0.2)),
                )
                if side == "buy":
                    be_sl = min(be_sl, bid - pip * 0.5)
                else:
                    be_sl = max(be_sl, ask + pip * 0.5)
                store.update_trade(
                    trade_id,
                    {
                        "tp1_partial_done": 1,
                        "size_lots": new_lots,
                        "config_json": _merge_cfg_json(trade_row, {"paper_partial_pnl_usd": new_partial}),
                        "breakeven_applied": 1,
                        "breakeven_sl_price": round(be_sl, 3),
                        "stop_price": round(be_sl, 3),
                    },
                )
            return

    # Reload row fields that may have changed
    open_rows = [dict(r) for r in store.list_open_trades(profile_name)]
    cur = next((r for r in open_rows if str(r.get("trade_id")) == trade_id), None)
    if not cur:
        return
    trade_row = cur
    lots = float(trade_row.get("size_lots") or 0.0)
    tp1_done = bool(trade_row.get("tp1_partial_done") or 0)
    sl0 = trade_row.get("stop_price")
    tp0 = trade_row.get("target_price")
    sl = float(sl0) if sl0 is not None else None
    tp = float(tp0) if tp0 is not None else None
    prev_be = trade_row.get("breakeven_sl_price")
    working_sl = sl
    if prev_be is not None:
        try:
            prev_be_f = float(prev_be)
            if working_sl is None:
                working_sl = prev_be_f
            else:
                working_sl = max(working_sl, prev_be_f) if side == "buy" else min(working_sl, prev_be_f)
        except (TypeError, ValueError):
            pass

    if trail_mode == "hwm" and tp1_done:
        peak = trade_row.get("peak_price")
        from api.ai_exit_strategies import compute_post_tp1_trail_sl

        if side == "buy":
            cur_peak = max(float(peak) if peak is not None else entry, mid)
            lock_sl = working_sl if working_sl is not None else entry - 10 * pip
            new_sl = compute_post_tp1_trail_sl(
                trade_side=side,
                entry_price=entry,
                tp1_pips=tp1_pips,
                pip_size=pip,
                high_water_mark=cur_peak,
                lock_sl=lock_sl,
            )
            working_sl = new_sl
            store.update_trade(trade_id, {"peak_price": round(cur_peak, 5), "stop_price": round(new_sl, 3)})
        else:
            cur_peak = min(float(peak) if peak is not None else entry, mid)
            lock_sl = working_sl if working_sl is not None else entry + 10 * pip
            new_sl = compute_post_tp1_trail_sl(
                trade_side=side,
                entry_price=entry,
                tp1_pips=tp1_pips,
                pip_size=pip,
                high_water_mark=cur_peak,
                lock_sl=lock_sl,
            )
            working_sl = new_sl
            store.update_trade(trade_id, {"peak_price": round(cur_peak, 5), "stop_price": round(new_sl, 3)})

    elif trail_mode == "m1" and tp1_done:
        ema_n = max(2, int(round(_cfg_exit("trail_ema_period", 21.0))))
        ema_val = _last_completed_ema(m1_df, ema_n) if m1_df is not None else None
        last_close = _last_completed_close(m1_df) if m1_df is not None else None
        if ema_val is not None:
            new_sl = (ema_val - pip) if side == "buy" else (ema_val + pip)
            if working_sl is not None:
                new_sl = max(new_sl, working_sl) if side == "buy" else min(new_sl, working_sl)
            working_sl = new_sl
            store.update_trade(trade_id, {"stop_price": round(new_sl, 3)})
        if last_close is not None and ema_val is not None and lots > 0:
            cross_exit = (side == "buy" and last_close < ema_val) or (side == "sell" and last_close > ema_val)
            if cross_exit:
                px = bid if side == "buy" else ask
                finalize_paper_close(
                    profile_name=profile_name,
                    state_path=state_path,
                    store=store,
                    trade_row=trade_row,
                    exit_price=px,
                    exit_reason="paper_m1_trail_exit",
                    pip_size=pip,
                    mid=mid,
                )
                return

    elif trail_mode == "m5" and tp1_done:
        ema_n = max(2, int(round(_cfg_exit("trail_ema_period", 20.0))))
        ema_val = _last_completed_ema(m5_df, ema_n) if m5_df is not None else None
        last_close = _last_completed_close(m5_df) if m5_df is not None else None
        if ema_val is not None:
            new_sl = (ema_val - pip) if side == "buy" else (ema_val + pip)
            if working_sl is not None:
                new_sl = max(new_sl, working_sl) if side == "buy" else min(new_sl, working_sl)
            working_sl = new_sl
            store.update_trade(trade_id, {"stop_price": round(new_sl, 3)})
        if last_close is not None and ema_val is not None and lots > 0:
            cross_exit = (side == "buy" and last_close < ema_val) or (side == "sell" and last_close > ema_val)
            if cross_exit:
                px = bid if side == "buy" else ask
                finalize_paper_close(
                    profile_name=profile_name,
                    state_path=state_path,
                    store=store,
                    trade_row=trade_row,
                    exit_price=px,
                    exit_reason="paper_m5_trail_exit",
                    pip_size=pip,
                    mid=mid,
                )
                return

    sl_row = trade_row.get("stop_price")
    working_sl = float(sl_row) if sl_row is not None else working_sl

    if lots <= 0:
        return

    if tp is not None:
        if side == "buy" and bid >= tp:
            finalize_paper_close(
                profile_name=profile_name,
                state_path=state_path,
                store=store,
                trade_row=trade_row,
                exit_price=float(tp),
                exit_reason="paper_tp",
                pip_size=pip,
                mid=mid,
            )
            return
        if side == "sell" and ask <= tp:
            finalize_paper_close(
                profile_name=profile_name,
                state_path=state_path,
                store=store,
                trade_row=trade_row,
                exit_price=float(tp),
                exit_reason="paper_tp",
                pip_size=pip,
                mid=mid,
            )
            return

    if working_sl is not None:
        if side == "buy" and bid <= working_sl:
            finalize_paper_close(
                profile_name=profile_name,
                state_path=state_path,
                store=store,
                trade_row=trade_row,
                exit_price=float(working_sl),
                exit_reason="paper_sl",
                pip_size=pip,
                mid=mid,
            )
            return
        if side == "sell" and ask >= working_sl:
            finalize_paper_close(
                profile_name=profile_name,
                state_path=state_path,
                store=store,
                trade_row=trade_row,
                exit_price=float(working_sl),
                exit_reason="paper_sl",
                pip_size=pip,
                mid=mid,
            )
            return


def tick_paper_fillmore_engine(
    *,
    profile: Any,
    profile_name: str,
    state_path: Path,
    store: Any,
    tick: Any,
    data_by_tf: dict[str, pd.DataFrame],
) -> None:
    process_paper_pending_limits(
        profile=profile,
        profile_name=profile_name,
        state_path=state_path,
        store=store,
        tick=tick,
    )
    tick_paper_open_trades(
        profile=profile,
        profile_name=profile_name,
        state_path=state_path,
        store=store,
        tick=tick,
        data_by_tf=data_by_tf or {},
    )
