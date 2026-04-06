from __future__ import annotations

import copy
import logging
import math
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pandas as pd

from core.phase3_additive_contract import (
    classify_defended_slice_label,
    defended_baseline_slice_labels,
    defended_slice_source,
)
from core.phase3_package_spec import PHASE3_DEFENDED_PRESET_ID, load_phase3_package_spec
from core.phase3_spike_fade_v4 import default_v4_runtime_config, detect_latest_v4_event
from core.phase3_session_support import build_phase3_session_support_from_mapping
from core.phase3_variant_k_baseline import apply_variant_k_baseline_admission

logger = logging.getLogger(__name__)

# OANDA margin closeout triggers when margin_used >= this fraction of equity (NAV).
# Positions are forcibly liquidated at this level.
# Default 0.50 = OANDA standard. Override via strict_policy keys on the package spec.
MARGIN_CLOSEOUT_FRACTION = 0.50

# Safety buffer below closeout to avoid brushing the limit during price movement
# between placement and next poll. 0.80 => effective max margin =
# equity * margin_closeout_fraction * margin_safety_buffer
MARGIN_SAFETY_BUFFER = 0.80


def _margin_used_from_account(acct: Any) -> float:
    """Read margin-in-use from broker account objects or dicts (adapter-specific field names)."""
    if acct is None:
        return 0.0
    if isinstance(acct, dict):
        v = acct.get("margin_used") or acct.get("margin") or acct.get("marginUsed")
        try:
            return float(v or 0.0)
        except (TypeError, ValueError):
            return 0.0
    v = getattr(acct, "margin_used", None) or getattr(acct, "margin", None) or getattr(acct, "marginUsed", None)
    if v is None:
        return 0.0
    try:
        return float(v or 0.0)
    except (TypeError, ValueError):
        return 0.0


@dataclass(frozen=True)
class Phase3BaselineIntent:
    strategy_tag: str
    strategy_family: str
    side: str
    entry_time_utc: str
    ownership_cell: str | None
    units: int
    entry_price: float
    sl_price: float | None
    tp1_price: float | None
    reason: str
    state_updates: dict[str, Any] = field(default_factory=dict)
    raw_result: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Phase3OffensiveSliceIntent:
    slice_id: str
    strategy_tag: str
    strategy_family: str
    side: str
    entry_time_utc: str
    ownership_cell: str | None
    units: int
    entry_price: float
    sl_price: float | None
    tp1_price: float | None
    reason: str
    size_scale: float = 1.0
    state_updates: dict[str, Any] = field(default_factory=dict)
    raw_result: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Phase3AdditiveCandidate:
    identity: str
    intent_source: str
    slice_id: str | None
    strategy_tag: str
    strategy_family: str
    side: str
    ownership_cell: str | None
    entry_time_utc: str
    units: int
    lots: float
    entry_price: float
    sl_price: float | None
    tp1_price: float | None
    reason: str
    state_updates: dict[str, Any] = field(default_factory=dict)
    raw_result: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Phase3ConflictPolicyState:
    open_book_count_before: int
    same_side_open_count_before: int
    opposite_side_open_count_before: int
    allow_internal_overlap: bool
    allow_opposite_side_overlap: bool
    max_open_offensive: int | None
    max_entries_per_day: int | None
    entries_today_before: int


@dataclass(frozen=True)
class Phase3MarginState:
    equity: float
    margin_used: float
    free_margin: float
    required_margin: float
    leverage: float
    max_lot_per_trade: float | None
    blocked: bool
    reason: str | None = None
    margin_closeout_fraction: float = MARGIN_CLOSEOUT_FRACTION
    margin_safety_buffer: float = MARGIN_SAFETY_BUFFER
    margin_ceiling: float = 0.0
    margin_utilization_pct: float = 0.0


@dataclass(frozen=True)
class Phase3AdditiveDecisionEnvelope:
    generated_at_utc: str
    package_id: str
    preset_id: str | None
    baseline_intents: list[dict[str, Any]] = field(default_factory=list)
    offensive_intents: list[dict[str, Any]] = field(default_factory=list)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    accepted: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    placements: list[dict[str, Any]] = field(default_factory=list)
    conflict_state: dict[str, Any] = field(default_factory=dict)
    margin_state: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Phase3AdditiveReplayResult:
    envelope: Phase3AdditiveDecisionEnvelope
    phase3_state_updates: dict[str, Any] = field(default_factory=dict)


class CandidateCaptureAdapter:
    def __init__(self, real_adapter) -> None:
        self.real_adapter = real_adapter
        self.captured_orders: list[dict[str, Any]] = []
        self._next_id = 1

    def is_demo(self) -> bool:
        is_demo = getattr(self.real_adapter, "is_demo", True)
        return bool(is_demo() if callable(is_demo) else is_demo)

    def get_account_info(self):
        try:
            return self.real_adapter.get_account_info()
        except Exception:
            return SimpleNamespace(balance=100000.0, equity=100000.0, margin_used=0.0)

    def place_order(self, *, symbol, side, lots, stop_price, target_price, comment):
        order_id = self._next_id
        self._next_id += 1
        payload = {
            "order_id": order_id,
            "deal_id": order_id,
            "symbol": symbol,
            "side": side,
            "lots": float(lots),
            "stop_price": stop_price,
            "target_price": target_price,
            "comment": comment,
        }
        self.captured_orders.append(payload)
        return SimpleNamespace(order_id=order_id, deal_id=order_id, order_retcode=0, fill_price=None)

    def get_position_id_from_order(self, order_id: int) -> int:
        return int(order_id)

    def get_position_id_from_deal(self, deal_id: int) -> int:
        return int(deal_id)



def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _copy_state(state: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(state or {})


def _parse_strategy_family(strategy_tag: str | None) -> str:
    tag = str(strategy_tag or "")
    if tag.startswith("phase3:v14"):
        return "v14"
    if tag.startswith("phase3:london_v2"):
        return "london_v2"
    if tag.startswith("phase3:v44_ny"):
        return "v44_ny"
    if tag.startswith("phase3:spike_fade_v4"):
        return "spike_fade_v4"
    return "unknown"


def _open_phase3_trades(store, profile_name: str) -> list[dict[str, Any]]:
    if store is None:
        return []
    try:
        out = []
        for row in store.list_open_trades(profile_name):
            r = dict(row)
            entry_type = str(r.get("entry_type") or "")
            if entry_type.startswith("phase3:"):
                out.append(r)
        return out
    except Exception:
        return []


def _entries_today(phase3_state: dict[str, Any], now_utc: datetime) -> int:
    key = now_utc.date().isoformat()
    additive = dict(phase3_state.get("additive_runtime") or {})
    day_counts = dict(additive.get("entries_by_day") or {})
    try:
        return int(day_counts.get(key, 0) or 0)
    except Exception:
        return 0


def _strict_policy_from_spec(spec) -> dict[str, Any]:
    return dict(spec.strict_policy or {})


def _margin_closeout_fraction_from_strict(strict: dict[str, Any]) -> float:
    v = (
        strict.get("margin_closeout_fraction")
        or strict.get("margin_closeout_pct")
        or strict.get("closeout_fraction")
        or strict.get("margin_limit_pct")
    )
    if v is None:
        return float(MARGIN_CLOSEOUT_FRACTION)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(MARGIN_CLOSEOUT_FRACTION)


def _margin_safety_buffer_from_strict(strict: dict[str, Any]) -> float:
    v = strict.get("margin_safety_buffer")
    if v is None:
        return float(MARGIN_SAFETY_BUFFER)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(MARGIN_SAFETY_BUFFER)


def _conflict_state(spec, phase3_state: dict[str, Any], open_trades: list[dict[str, Any]], candidate_side: str, now_utc: datetime) -> Phase3ConflictPolicyState:
    strict = _strict_policy_from_spec(spec)
    allow_internal_overlap = bool(strict.get("allow_internal_overlap", True))
    allow_opposite_side_overlap = bool(strict.get("allow_opposite_side_overlap", True))
    max_open = strict.get("max_open_offensive")
    max_entries = strict.get("max_entries_per_day")
    same_side = 0
    opposite_side = 0
    side_norm = str(candidate_side or "").lower()
    for row in open_trades:
        row_side = str(row.get("side") or "").lower()
        if row_side == side_norm:
            same_side += 1
        elif row_side in {"buy", "sell"}:
            opposite_side += 1
    return Phase3ConflictPolicyState(
        open_book_count_before=len(open_trades),
        same_side_open_count_before=same_side,
        opposite_side_open_count_before=opposite_side,
        allow_internal_overlap=allow_internal_overlap,
        allow_opposite_side_overlap=allow_opposite_side_overlap,
        max_open_offensive=None if max_open is None else int(max_open),
        max_entries_per_day=None if max_entries is None else int(max_entries),
        entries_today_before=_entries_today(phase3_state, now_utc),
    )


def _margin_state(real_adapter, spec, lots: float, *, open_trades_count: int = -1) -> Phase3MarginState:
    # Adapter field mapping for margin in use:
    #   OandaAdapter: acct.margin (from OANDA v20 API "marginUsed" → mapped to .margin)
    #   Some stubs/tests: margin_used on SimpleNamespace
    #   Verify with: print(vars(acct)) or dir(acct) if field names change
    strict = _strict_policy_from_spec(spec)
    closeout_frac = _margin_closeout_fraction_from_strict(strict)
    safety_buf = _margin_safety_buffer_from_strict(strict)
    leverage = float(strict.get("margin_leverage") or 33.3)
    max_lot_per_trade = strict.get("max_lot_per_trade")
    blocked = False
    reason = None
    equity = 100000.0
    margin_used = 0.0
    try:
        acct = real_adapter.get_account_info()
        if isinstance(acct, dict):
            equity = float(acct.get("equity") or acct.get("balance") or equity)
        else:
            equity = float(getattr(acct, "equity", getattr(acct, "balance", equity)) or equity)
        margin_used = _margin_used_from_account(acct)
    except Exception:
        pass
    margin_ceiling = float(equity) * float(closeout_frac)
    free_margin = float(equity) - float(margin_used)
    margin_util_pct = (float(margin_used) / float(equity) * 100.0) if equity > 0 else 0.0
    if (
        margin_used == 0.0
        and equity > 100.0
        and open_trades_count >= 1
    ):
        logger.warning(
            "_margin_state: margin_used resolved to 0.0 with equity=%.2f — "
            "possible field mismatch with adapter. Margin gate may not be effective.",
            equity,
        )
    if max_lot_per_trade is not None and float(lots) > float(max_lot_per_trade):
        blocked = True
        reason = f"lot_cap>{float(max_lot_per_trade):.2f}"
    # NOTE: This is a simplified margin estimate assuming standard forex lot size
    # (100,000 units) at account-level leverage. Actual OANDA margin requirements
    # vary by:
    #   - Instrument (JPY pairs, metals, exotics have different margin rates)
    #   - Regulatory jurisdiction (ESMA 30:1 max, ASIC, US CFTC rules)
    #   - Intraday margin changes (news events, weekend close)
    # This approximation is acceptable for demo/paper trading on major pairs.
    # For live trading on non-majors, replace with instrument-specific margin lookup.
    required_margin = float(lots) * 100000.0 / max(1.0, leverage)
    if equity > 0 and margin_used > 0:
        utilization = float(margin_used) / float(equity)
        if utilization > 0.60:
            logger.warning(
                "_margin_state: margin utilization at %.1f%% (used=%.2f, equity=%.2f) — "
                "approaching OANDA 50%% closeout territory if equity drops",
                utilization * 100,
                margin_used,
                equity,
            )
    if not blocked and required_margin > free_margin:
        blocked = True
        reason = "margin_rejected"
    return Phase3MarginState(
        equity=float(equity),
        margin_used=float(margin_used),
        free_margin=float(free_margin),
        required_margin=float(required_margin),
        leverage=float(leverage),
        max_lot_per_trade=float(max_lot_per_trade) if max_lot_per_trade is not None else None,
        blocked=blocked,
        reason=reason,
        margin_closeout_fraction=float(closeout_frac),
        margin_safety_buffer=float(safety_buf),
        margin_ceiling=float(margin_ceiling),
        margin_utilization_pct=float(margin_util_pct),
    )


def _conflict_block_reason(
    conflict: Phase3ConflictPolicyState,
    *,
    accepted_count: int,
    apply_max_open_cap: bool,
) -> str | None:
    if conflict.max_entries_per_day is not None and conflict.max_entries_per_day > 0:
        current = conflict.entries_today_before + max(0, int(accepted_count))
        if current >= conflict.max_entries_per_day:
            return f"max_entries_day_{current}/{conflict.max_entries_per_day}"
    if apply_max_open_cap and conflict.max_open_offensive is not None and conflict.max_open_offensive > 0:
        current_open = conflict.open_book_count_before + max(0, int(accepted_count))
        if current_open >= conflict.max_open_offensive:
            return f"max_open_{current_open}/{conflict.max_open_offensive}"
    if not conflict.allow_internal_overlap and conflict.same_side_open_count_before > 0:
        return "internal_overlap_blocked"
    if not conflict.allow_opposite_side_overlap and conflict.opposite_side_open_count_before > 0:
        return "opposite_side_overlap_blocked"
    return None


def _candidate_identity(strategy_tag: str, side: str, now_utc: datetime) -> str:
    return f"{strategy_tag}|{side}|{pd.Timestamp(now_utc).floor('min').isoformat()}"


def _capture_candidate(now_utc: datetime, session_name: str, capture_adapter: CandidateCaptureAdapter, result: dict[str, Any]) -> Phase3AdditiveCandidate | None:
    decision = result.get("decision")
    if not getattr(decision, "placed", False):
        return None
    strategy_tag = str(result.get("strategy_tag") or "")
    side = str(getattr(decision, "side", None) or "")
    if not strategy_tag or not side:
        return None
    order = capture_adapter.captured_orders[-1] if capture_adapter.captured_orders else {}
    ownership_cell = None
    if "@" in strategy_tag:
        _, _, ownership_cell = strategy_tag.partition("@")
    units = int(result.get("units") or max(0, math.floor(float(order.get("lots") or 0.0) * 100000.0)))
    entry_price = float(result.get("entry_price") or 0.0)
    return Phase3AdditiveCandidate(
        identity=_candidate_identity(strategy_tag, side, now_utc),
        intent_source="reconstructed_family_candidate",
        slice_id=strategy_tag,
        strategy_tag=strategy_tag,
        strategy_family=_parse_strategy_family(strategy_tag),
        side=side,
        ownership_cell=ownership_cell,
        entry_time_utc=pd.Timestamp(now_utc).isoformat(),
        units=units,
        lots=float(order.get("lots") or (units / 100000.0)),
        entry_price=entry_price,
        sl_price=result.get("sl_price"),
        tp1_price=result.get("tp1_price"),
        reason=str(getattr(decision, "reason", "") or f"{session_name}: candidate"),
        state_updates=dict(result.get("phase3_state_updates") or {}),
        raw_result=dict(result),
    )


def _candidate_offline_slice_label(candidate: Phase3AdditiveCandidate, active_slice_scales: dict[str, float]) -> str | None:
    return classify_defended_slice_label(
        strategy_family=candidate.strategy_family,
        side=candidate.side,
        ownership_cell=candidate.ownership_cell,
        strategy_tag=candidate.strategy_tag,
        reason=candidate.reason,
        active_slice_scales=active_slice_scales,
    )


def _split_candidate_intents(
    candidates: list[Phase3AdditiveCandidate],
    spec,
) -> tuple[list[Phase3BaselineIntent], list[Phase3OffensiveSliceIntent], list[Phase3AdditiveCandidate], list[Phase3AdditiveCandidate]]:
    active_slice_scales = {
        str(label): float(scale)
        for label, scale in dict(spec.base_cell_scales or {}).items()
        if float(scale) > 0.0
    }
    baseline_labels = defended_baseline_slice_labels()
    baseline_intents: list[Phase3BaselineIntent] = []
    offensive_intents: list[Phase3OffensiveSliceIntent] = []
    baseline_candidates: list[Phase3AdditiveCandidate] = []
    offensive_candidates: list[Phase3AdditiveCandidate] = []

    for candidate in candidates:
        slice_label = _candidate_offline_slice_label(candidate, active_slice_scales)
        if slice_label and defended_slice_source(slice_label) == "offensive":
            offensive_intents.append(
                Phase3OffensiveSliceIntent(
                    slice_id=slice_label,
                    strategy_tag=candidate.strategy_tag,
                    strategy_family=candidate.strategy_family,
                    side=candidate.side,
                    entry_time_utc=candidate.entry_time_utc,
                    ownership_cell=candidate.ownership_cell,
                    units=candidate.units,
                    entry_price=candidate.entry_price,
                    sl_price=candidate.sl_price,
                    tp1_price=candidate.tp1_price,
                    reason=candidate.reason,
                    size_scale=float(active_slice_scales.get(slice_label, 1.0)),
                    state_updates=dict(candidate.state_updates or {}),
                    raw_result=dict(candidate.raw_result or {}),
                )
            )
            offensive_candidates.append(
                Phase3AdditiveCandidate(
                    identity=candidate.identity,
                    intent_source="offensive",
                    slice_id=slice_label,
                    strategy_tag=candidate.strategy_tag,
                    strategy_family=candidate.strategy_family,
                    side=candidate.side,
                    ownership_cell=candidate.ownership_cell,
                    entry_time_utc=candidate.entry_time_utc,
                    units=candidate.units,
                    lots=candidate.lots,
                    entry_price=candidate.entry_price,
                    sl_price=candidate.sl_price,
                    tp1_price=candidate.tp1_price,
                    reason=candidate.reason,
                    state_updates=dict(candidate.state_updates or {}),
                    raw_result=dict(candidate.raw_result or {}),
                )
            )
            continue

        baseline_slice_id = slice_label if slice_label in baseline_labels else f"baseline:{candidate.strategy_family}:{candidate.side}@{str(candidate.ownership_cell or 'uncategorized')}"
        baseline_intents.append(
            Phase3BaselineIntent(
                strategy_tag=candidate.strategy_tag,
                strategy_family=candidate.strategy_family,
                side=candidate.side,
                entry_time_utc=candidate.entry_time_utc,
                ownership_cell=candidate.ownership_cell,
                units=candidate.units,
                entry_price=candidate.entry_price,
                sl_price=candidate.sl_price,
                tp1_price=candidate.tp1_price,
                reason=candidate.reason,
                state_updates=dict(candidate.state_updates or {}),
                raw_result=dict(candidate.raw_result or {}),
            )
        )
        baseline_candidates.append(
            Phase3AdditiveCandidate(
                identity=candidate.identity,
                intent_source="baseline",
                slice_id=baseline_slice_id,
                strategy_tag=candidate.strategy_tag,
                strategy_family=candidate.strategy_family,
                side=candidate.side,
                ownership_cell=candidate.ownership_cell,
                entry_time_utc=candidate.entry_time_utc,
                units=candidate.units,
                lots=candidate.lots,
                entry_price=candidate.entry_price,
                sl_price=candidate.sl_price,
                tp1_price=candidate.tp1_price,
                reason=candidate.reason,
                state_updates=dict(candidate.state_updates or {}),
                raw_result=dict(candidate.raw_result or {}),
            )
        )

    baseline_candidates.sort(key=lambda c: (c.entry_time_utc, c.strategy_tag, c.side))
    offensive_candidates.sort(key=lambda c: (c.entry_time_utc, c.strategy_tag, c.side))
    return baseline_intents, offensive_intents, baseline_candidates, offensive_candidates


def _evaluate_family_candidates(*, adapter, profile, policy, data_by_tf, tick, mode: str, phase3_state: dict[str, Any], store, sizing_config: dict[str, Any], now_utc: datetime, ownership_audit: dict[str, Any], overlay_state: dict[str, Any]) -> tuple[list[Phase3AdditiveCandidate], list[dict[str, Any]]]:
    from core.phase3_integrated_engine import execute_tokyo_v14_entry, execute_london_v2_entry, execute_v44_ny_entry

    candidates: list[Phase3AdditiveCandidate] = []
    diagnostics: list[dict[str, Any]] = []
    for session_name, runner in (
        ("tokyo", lambda a, s: execute_tokyo_v14_entry(adapter=a, profile=profile, policy=policy, data_by_tf=data_by_tf, tick=tick, mode=mode, phase3_state=s, store=store, sizing_config=sizing_config, now_utc=now_utc, ownership_audit=ownership_audit, overlay_state=overlay_state, base_state_updates={})),
        ("london", lambda a, s: execute_london_v2_entry(adapter=a, profile=profile, policy=policy, data_by_tf=data_by_tf, tick=tick, phase3_state=s, store=store, sizing_config=sizing_config, now_utc=now_utc, ownership_audit=ownership_audit, overlay_state=overlay_state)),
        ("ny", lambda a, s: execute_v44_ny_entry(adapter=a, profile=profile, policy=policy, data_by_tf=data_by_tf, tick=tick, phase3_state=s, store=store, sizing_config=sizing_config, now_utc=now_utc, ownership_audit=ownership_audit, overlay_state=overlay_state)),
    ):
        capture = CandidateCaptureAdapter(adapter)
        local_state = _copy_state(phase3_state)
        result = runner(capture, local_state)
        decision = result.get("decision")
        diagnostics.append({
            "session": session_name,
            "strategy_tag": result.get("strategy_tag"),
            "attempted": bool(getattr(decision, "attempted", False)),
            "placed": bool(getattr(decision, "placed", False)),
            "reason": str(getattr(decision, "reason", "") or ""),
        })
        candidate = _capture_candidate(now_utc, session_name, capture, result)
        if candidate is not None:
            candidates.append(candidate)
    candidates.sort(key=lambda c: (c.entry_time_utc, c.strategy_tag, c.side))
    return candidates, diagnostics


def _merge_state_updates(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _store_execution(store, *, profile_name: str, symbol: str, signal_id: str, mode: str, attempted: bool, placed: bool, reason: str, order_id: int | None = None, deal_id: int | None = None) -> None:
    if store is None:
        return
    try:
        store.insert_execution(
            {
                "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "profile": profile_name,
                "symbol": symbol,
                "signal_id": signal_id,
                "rule_id": "phase3:spike_fade_v4",
                "mode": mode,
                "attempted": 1 if attempted else 0,
                "placed": 1 if placed else 0,
                "reason": str(reason),
                "mt5_retcode": None,
                "mt5_order_id": order_id,
                "mt5_deal_id": deal_id,
            }
        )
    except Exception:
        return


def _v4_runtime_cfg(sizing_config: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(default_v4_runtime_config())
    patch = dict((sizing_config or {}).get("spike_fade_v4") or {})
    cfg.update(patch)
    return cfg


def _trade_comment_value(row: dict[str, Any]) -> str:
    for key in ("tradeClientExtensions", "clientExtensions"):
        ext = row.get(key)
        if isinstance(ext, dict):
            comment = str(ext.get("comment") or "").strip()
            if comment:
                return comment
    return ""


def _v4_comment(profile) -> str:
    return f"phase3_integrated:{getattr(profile, 'active_preset_name', '')}:spike_fade_v4"


def _broker_open_positions(adapter, profile) -> list[dict[str, Any]]:
    try:
        return list(adapter.get_open_positions(profile.symbol) or [])
    except Exception:
        return []


def _broker_pending_orders(adapter, profile) -> list[dict[str, Any]]:
    try:
        if hasattr(adapter, "list_pending_orders"):
            return list(adapter.list_pending_orders(profile.symbol) or [])
    except Exception:
        return []
    return []


def _find_v4_live_trade(open_positions: list[dict[str, Any]], comment_value: str) -> dict[str, Any] | None:
    for row in open_positions:
        if _trade_comment_value(row) == comment_value:
            return dict(row)
    return None


def _find_v4_pending_order(pending_orders: list[dict[str, Any]], comment_value: str, order_id: Any | None = None) -> dict[str, Any] | None:
    if order_id is not None:
        for row in pending_orders:
            if str(row.get("id")) == str(order_id):
                return dict(row)
    for row in pending_orders:
        if _trade_comment_value(row) == comment_value:
            return dict(row)
    return None


def _v4_units_and_lots(live_row: dict[str, Any]) -> tuple[int, float]:
    units = int(float(live_row.get("currentUnits") or live_row.get("initialUnits") or 0.0))
    return units, abs(float(units)) / 100000.0


def _v4_side_from_units(units: int) -> str:
    return "buy" if int(units) > 0 else "sell"


def _v4_margin_gate(adapter, spec, lots: float, cap_pct: float) -> tuple[bool, dict[str, float]]:
    margin = _margin_state(adapter, spec, lots)
    required_after = float(margin.margin_used) + float(margin.required_margin)
    equity = float(margin.equity)
    projected_pct = (required_after / equity * 100.0) if equity > 0 else 0.0
    return projected_pct <= float(cap_pct), {
        "equity": equity,
        "margin_used": float(margin.margin_used),
        "required_margin": float(margin.required_margin),
        "projected_margin_pct": projected_pct,
        "free_margin": float(margin.free_margin),
    }


def _execute_v4_runtime(
    *,
    adapter,
    profile,
    spec,
    data_by_tf: dict[str, Any],
    tick,
    phase3_state: dict[str, Any],
    store,
    sizing_config: dict[str, Any],
    now_utc: datetime,
    mode: str,
    is_new_m1: bool,
) -> dict[str, Any]:
    from core.phase3_integrated_engine import ExecutionDecision

    cfg = _v4_runtime_cfg(sizing_config)
    runtime = dict(phase3_state.get("v4_runtime") or {})
    runtime.setdefault("lifecycle_state", "IDLE_WARMUP")
    runtime.setdefault("diagnostics", [])
    diagnostics: list[dict[str, Any]] = list(runtime.get("diagnostics") or [])
    placements: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    comment_value = _v4_comment(profile)
    profile_name = getattr(profile, "profile_name", "") or getattr(profile, "name", "") or str(profile)

    if not bool(cfg.get("enabled")):
        runtime["lifecycle_state"] = "DISABLED"
        runtime["diagnostics"] = diagnostics[-25:]
        return {"state_updates": {"v4_runtime": runtime}, "placements": placements, "rejected": rejected, "attempted": False}

    open_positions = _broker_open_positions(adapter, profile)
    pending_orders = _broker_pending_orders(adapter, profile)
    live_trade = _find_v4_live_trade(open_positions, comment_value)
    live_pending = _find_v4_pending_order(pending_orders, comment_value, runtime.get("pending_order_id"))

    if runtime.get("pending_order_id") and live_pending is None and live_trade is None:
        expired_order_id = runtime.get("pending_order_id")
        runtime["last_event"] = "broker_expired_gtd"
        runtime["last_expired_at"] = pd.Timestamp(now_utc).isoformat()
        runtime["lifecycle_state"] = "READY"
        runtime["pending_order_id"] = None
        _store_execution(
            store,
            profile_name=profile_name,
            symbol=profile.symbol,
            signal_id=f"phase3:v4:expired:{pd.Timestamp(now_utc).isoformat()}",
            mode=mode,
            attempted=True,
            placed=False,
            reason="broker_expired_gtd",
            order_id=int(expired_order_id) if str(expired_order_id or "").isdigit() else None,
        )

    if live_trade is not None:
        units, lots = _v4_units_and_lots(live_trade)
        fill_time = pd.Timestamp(str(live_trade.get("openTime") or runtime.get("active_fill_time") or now_utc))
        if fill_time.tzinfo is None:
            fill_time = fill_time.tz_localize("UTC")
        else:
            fill_time = fill_time.tz_convert("UTC")
        runtime["active_trade_id"] = str(live_trade.get("id"))
        runtime["active_fill_time"] = fill_time.isoformat()
        runtime["active_entry_price"] = float(live_trade.get("price") or runtime.get("active_entry_price") or 0.0)
        runtime["active_side"] = _v4_side_from_units(units)
        runtime["active_lots"] = lots
        runtime["pending_order_id"] = None
        trade_id_text = str(live_trade.get("id") or "")
        first_fill_seen = str(runtime.get("reported_trade_id") or "") != trade_id_text
        if first_fill_seen:
            runtime["reported_trade_id"] = trade_id_text
            runtime["last_event"] = "broker_filled"
            runtime["cluster_block_until"] = (
                fill_time + pd.Timedelta(minutes=int(cfg.get("cluster_block_minutes", 120)))
            ).isoformat()
            placements.append(
                {
                    "strategy_tag": "phase3:spike_fade_v4",
                    "entry_price": float(runtime["active_entry_price"]),
                    "sl_price": float(runtime.get("stop_price")) if runtime.get("stop_price") is not None else None,
                    "tp1_price": float(runtime.get("tp_price")) if runtime.get("tp_price") is not None else None,
                    "units": abs(int(units)),
                    "decision": ExecutionDecision(
                        attempted=True,
                        placed=True,
                        reason="broker_filled",
                        side=str(runtime["active_side"]),
                        order_retcode=0,
                        order_id=int(live_trade.get("id")) if trade_id_text.isdigit() else None,
                        deal_id=int(live_trade.get("id")) if trade_id_text.isdigit() else None,
                        fill_price=float(runtime["active_entry_price"]),
                    ),
                }
            )
            _store_execution(
                store,
                profile_name=profile_name,
                symbol=profile.symbol,
                signal_id=f"phase3:v4:filled:{fill_time.isoformat()}",
                mode=mode,
                attempted=True,
                placed=True,
                reason="broker_filled",
                deal_id=int(live_trade.get("id")) if trade_id_text.isdigit() else None,
            )
        runtime["lifecycle_state"] = "POSITION_OPEN"

        if not bool(is_new_m1):
            runtime["diagnostics"] = diagnostics[-25:]
            return {"state_updates": {"v4_runtime": runtime}, "placements": placements, "rejected": rejected, "attempted": True}

        side = str(runtime.get("active_side") or "buy")
        entry_price = float(runtime.get("active_entry_price") or 0.0)
        if side == "buy":
            check_price = float(tick.bid)
            favorable_price = float(tick.bid)
            current_pips = (check_price - entry_price) / 0.01
        else:
            check_price = float(tick.ask)
            favorable_price = float(tick.ask)
            current_pips = (entry_price - check_price) / 0.01
        tp_price = runtime.get("tp_price")
        exit_reason = None
        if tp_price is not None:
            if side == "buy" and check_price >= float(tp_price):
                exit_reason = "software_take_profit"
            elif side == "sell" and check_price <= float(tp_price):
                exit_reason = "software_take_profit"
        now_ts = pd.Timestamp(now_utc)
        if now_ts.tzinfo is None:
            now_ts = now_ts.tz_localize("UTC")
        else:
            now_ts = now_ts.tz_convert("UTC")
        held_minutes = (now_ts - fill_time).total_seconds() / 60.0
        if exit_reason is None and held_minutes >= float(cfg.get("prove_it_fast_minutes", 15)):
            if current_pips < float(cfg.get("prove_it_fast_threshold_pips", -5.0)):
                exit_reason = "software_prove_it_fast"
        if exit_reason is None and bool(cfg.get("trailing_enabled", True)):
            trail_stop = runtime.get("active_trail_stop")
            trigger_pips = float(cfg.get("trail_trigger_pips", 10.0))
            trail_distance_pips = float(cfg.get("trail_distance_pips", 5.0))
            if current_pips >= trigger_pips:
                if side == "buy":
                    new_stop = favorable_price - trail_distance_pips * 0.01
                    if trail_stop is None or float(new_stop) > float(trail_stop):
                        try:
                            adapter.update_position_stop_loss(int(live_trade.get("id")), profile.symbol, round(float(new_stop), 3))
                            runtime["active_trail_stop"] = float(new_stop)
                            runtime["last_event"] = "software_trailing_update"
                        except Exception:
                            pass
                else:
                    new_stop = favorable_price + trail_distance_pips * 0.01
                    if trail_stop is None or float(new_stop) < float(trail_stop):
                        try:
                            adapter.update_position_stop_loss(int(live_trade.get("id")), profile.symbol, round(float(new_stop), 3))
                            runtime["active_trail_stop"] = float(new_stop)
                            runtime["last_event"] = "software_trailing_update"
                        except Exception:
                            pass
        if exit_reason is not None:
            try:
                position_type = 0 if side == "buy" else 1
                adapter.close_position(
                    ticket=int(live_trade.get("id")),
                    symbol=profile.symbol,
                    volume=float(lots),
                    position_type=position_type,
                )
                runtime["last_event"] = exit_reason
                cooldown_until = runtime.get("cluster_block_until")
                runtime["lifecycle_state"] = (
                    "COOLDOWN_CLUSTER_BLOCK"
                    if cooldown_until and pd.Timestamp(cooldown_until) > pd.Timestamp(now_utc)
                    else "READY"
                )
                _store_execution(
                    store,
                    profile_name=profile_name,
                    symbol=profile.symbol,
                    signal_id=f"phase3:v4:exit:{pd.Timestamp(now_utc).isoformat()}",
                    mode=mode,
                    attempted=True,
                    placed=False,
                    reason=exit_reason,
                    deal_id=int(live_trade.get("id")) if str(live_trade.get("id") or "").isdigit() else None,
                )
            except Exception as exc:
                diagnostics.append({"event": "v4_exit_error", "reason": str(exc), "time": pd.Timestamp(now_utc).isoformat()})
        runtime["diagnostics"] = diagnostics[-25:]
        return {"state_updates": {"v4_runtime": runtime}, "placements": placements, "rejected": rejected, "attempted": True}

    if runtime.get("active_trade_id") and live_trade is None:
        runtime["active_trade_id"] = None
        runtime["active_fill_time"] = None
        runtime["active_entry_price"] = None
        runtime["active_side"] = None
        runtime["active_trail_stop"] = None
        runtime["reported_trade_id"] = None
        runtime["last_event"] = "broker_trade_closed"
        cooldown_until = runtime.get("cluster_block_until")
        runtime["lifecycle_state"] = (
            "COOLDOWN_CLUSTER_BLOCK"
            if cooldown_until and pd.Timestamp(cooldown_until) > pd.Timestamp(now_utc)
            else "READY"
        )

    if live_pending is not None:
        runtime["pending_order_id"] = str(live_pending.get("id"))
        runtime["lifecycle_state"] = "ORDER_ARMED"
        runtime["diagnostics"] = diagnostics[-25:]
        return {"state_updates": {"v4_runtime": runtime}, "placements": placements, "rejected": rejected, "attempted": True}

    if runtime.get("pending_order_id"):
        runtime["pending_order_id"] = None

    if not bool(is_new_m1):
        runtime["diagnostics"] = diagnostics[-25:]
        return {"state_updates": {"v4_runtime": runtime}, "placements": placements, "rejected": rejected, "attempted": False}

    detected_event, detected_state = detect_latest_v4_event(
        data_by_tf=data_by_tf,
        phase3_state={"v4_runtime": runtime},
        runtime_cfg=cfg,
        now_utc=now_utc,
    )
    runtime.update(detected_state)
    if detected_event is None:
        runtime["diagnostics"] = diagnostics[-25:]
        return {"state_updates": {"v4_runtime": runtime}, "placements": placements, "rejected": rejected, "attempted": False}

    if runtime.get("active_trade_id") or runtime.get("pending_order_id"):
        runtime["last_reject_reason"] = "v4_active_or_pending"
        rejected.append({"strategy_tag": "phase3:spike_fade_v4", "reason": "v4_active_or_pending"})
        runtime["diagnostics"] = diagnostics[-25:]
        return {"state_updates": {"v4_runtime": runtime}, "placements": placements, "rejected": rejected, "attempted": True}

    cap_ok, margin_diag = _v4_margin_gate(adapter, spec, float(cfg.get("lots", 20.0)), float(cfg.get("shared_margin_cap_pct", 75.0)))
    runtime["last_margin_diag"] = margin_diag
    if not cap_ok:
        runtime["last_reject_reason"] = "v4_shared_margin_cap"
        runtime["last_event"] = "rejected_margin"
        rejected.append({"strategy_tag": "phase3:spike_fade_v4", "reason": "v4_shared_margin_cap", "margin_state": margin_diag})
        _store_execution(
            store,
            profile_name=profile_name,
            symbol=profile.symbol,
            signal_id=f"phase3:v4:reject:{pd.Timestamp(now_utc).isoformat()}",
            mode=mode,
            attempted=True,
            placed=False,
            reason="v4_shared_margin_cap",
        )
        runtime["diagnostics"] = diagnostics[-25:]
        return {"state_updates": {"v4_runtime": runtime}, "placements": placements, "rejected": rejected, "attempted": True}

    if not hasattr(adapter, "order_send_pending_limit"):
        runtime["last_reject_reason"] = "broker_missing_pending_limit_support"
        rejected.append({"strategy_tag": "phase3:spike_fade_v4", "reason": "broker_missing_pending_limit_support"})
        runtime["diagnostics"] = diagnostics[-25:]
        return {"state_updates": {"v4_runtime": runtime}, "placements": placements, "rejected": rejected, "attempted": True}

    try:
        expiry_ts = pd.Timestamp(detected_event.expiry_time)
        if expiry_ts.tzinfo is None:
            expiry_ts = expiry_ts.tz_localize("UTC")
        else:
            expiry_ts = expiry_ts.tz_convert("UTC")
        result = adapter.order_send_pending_limit(
            symbol=profile.symbol,
            side=str(detected_event.side),
            price=float(detected_event.trigger_level),
            volume_lots=float(cfg.get("lots", 20.0)),
            sl=float(detected_event.stop_price),
            tp=None,
            time_in_force="GTD",
            gtd_time_utc=expiry_ts.isoformat().replace("+00:00", "Z"),
            comment=comment_value,
        )
        if getattr(result, "retcode", None) in (0, 10008, 10009) and getattr(result, "order", None) is not None:
            runtime["pending_order_id"] = str(result.order)
            runtime["pending_armed_at"] = detected_event.armed_time
            runtime["pending_expiry_time"] = detected_event.expiry_time
            runtime["active_side"] = str(detected_event.side)
            runtime["stop_price"] = float(detected_event.stop_price)
            runtime["tp_price"] = float(detected_event.tp_price) if detected_event.tp_price is not None else None
            runtime["confirmation_level"] = float(detected_event.confirmation_level)
            runtime["last_event"] = "broker_pending"
            runtime["lifecycle_state"] = "ORDER_ARMED"
            runtime["last_detected_event"] = detected_event.__dict__
            placements.append(
                {
                    "strategy_tag": "phase3:spike_fade_v4",
                    "entry_price": float(detected_event.trigger_level),
                    "sl_price": float(detected_event.stop_price),
                    "tp1_price": float(detected_event.tp_price) if detected_event.tp_price is not None else None,
                    "decision": SimpleNamespace(
                        attempted=True,
                        placed=False,
                        reason="broker_pending",
                        side=str(detected_event.side),
                        order_id=getattr(result, "order", None),
                        deal_id=getattr(result, "deal", None),
                        fill_price=None,
                        order_retcode=getattr(result, "retcode", None),
                    ),
                }
            )
            _store_execution(
                store,
                profile_name=profile_name,
                symbol=profile.symbol,
                signal_id=f"phase3:v4:arm:{detected_event.armed_time}",
                mode=mode,
                attempted=True,
                placed=True,
                reason="broker_pending",
                order_id=int(result.order) if str(result.order).isdigit() else None,
            )
        else:
            runtime["last_reject_reason"] = f"order_rejected:{getattr(result, 'comment', 'unknown')}"
            rejected.append({"strategy_tag": "phase3:spike_fade_v4", "reason": runtime["last_reject_reason"]})
    except Exception as exc:
        runtime["last_reject_reason"] = f"pending_order_error:{exc}"
        rejected.append({"strategy_tag": "phase3:spike_fade_v4", "reason": runtime["last_reject_reason"]})

    runtime["diagnostics"] = diagnostics[-25:]
    return {"state_updates": {"v4_runtime": runtime}, "placements": placements, "rejected": rejected, "attempted": True}


def _apply_candidate_adjustments(
    candidates: list[Phase3AdditiveCandidate],
    adjustments: dict[str, dict[str, Any]],
) -> list[Phase3AdditiveCandidate]:
    out: list[Phase3AdditiveCandidate] = []
    for candidate in candidates:
        patch = dict(adjustments.get(str(candidate.identity), {}) or {})
        if not patch:
            out.append(candidate)
            continue
        raw_updates = dict(patch.get("raw_result_updates") or {})
        reason_suffix = str(patch.get("reason_suffix") or "")
        out.append(
            replace(
                candidate,
                reason=f"{candidate.reason}{reason_suffix}" if reason_suffix else candidate.reason,
                raw_result=_deep_merge(dict(candidate.raw_result or {}), raw_updates),
            )
        )
    return out


def _place_candidate(real_adapter, profile, candidate: Phase3AdditiveCandidate) -> dict[str, Any]:
    from core.phase3_integrated_engine import ExecutionDecision, _phase3_order_confirmed

    raw = dict(candidate.raw_result or {})
    order_result = real_adapter.place_order(
        symbol=profile.symbol,
        side=candidate.side,
        lots=candidate.lots,
        stop_price=round(float(candidate.sl_price), 3) if candidate.sl_price is not None else None,
        target_price=(round(float(candidate.tp1_price), 3) if candidate.tp1_price is not None else None) if candidate.strategy_family != "v44_ny" else None,
        comment=f"phase3_integrated:{getattr(profile, 'active_preset_name', '')}:{candidate.strategy_family}",
    )
    confirmed_fill, confirmed_deal_id = _phase3_order_confirmed(real_adapter, profile, order_result)
    if not confirmed_fill:
        return {
            "decision": ExecutionDecision(
                attempted=True,
                placed=False,
                reason=f"additive: broker order pending/unfilled ({candidate.strategy_tag})",
                side=candidate.side,
                order_retcode=getattr(order_result, "order_retcode", None),
                order_id=getattr(order_result, "order_id", None),
                deal_id=getattr(order_result, "deal_id", None),
                fill_price=getattr(order_result, "fill_price", None),
            ),
            "strategy_tag": candidate.strategy_tag,
            "entry_price": candidate.entry_price,
            "sl_price": candidate.sl_price,
            "tp1_price": candidate.tp1_price,
            "units": candidate.units,
            "risk_usd_planned": raw.get("risk_usd_planned"),
            "phase3_state_updates": candidate.state_updates,
            "v44_exit_plan": raw.get("v44_exit_plan") or {},
            "v44_parity_context": raw.get("v44_parity_context") or {},
            "phase3_additive_candidate": asdict(candidate),
        }
    return {
        **raw,
        "decision": ExecutionDecision(
            attempted=True,
            placed=True,
            reason=f"additive accepted: {candidate.strategy_tag}",
            side=candidate.side,
            order_retcode=getattr(order_result, "order_retcode", None),
            order_id=getattr(order_result, "order_id", None),
            deal_id=confirmed_deal_id if confirmed_deal_id is not None else getattr(order_result, "deal_id", None),
            fill_price=getattr(order_result, "fill_price", None),
        ),
        "strategy_tag": candidate.strategy_tag,
        "entry_price": candidate.entry_price,
        "sl_price": candidate.sl_price,
        "tp1_price": candidate.tp1_price,
        "units": candidate.units,
        "phase3_state_updates": candidate.state_updates,
        "phase3_additive_candidate": asdict(candidate),
    }


def execute_phase3_defended_additive_policy(
    *,
    adapter,
    profile,
    log_dir,
    policy,
    context,
    data_by_tf: dict[str, Any],
    tick,
    mode: str,
    phase3_state: dict[str, Any],
    store=None,
    sizing_config: dict[str, Any] | None = None,
    ownership_audit: dict[str, Any] | None = None,
    overlay_state: dict[str, Any] | None = None,
    now_utc: datetime | None = None,
    preset_id: str | None = None,
    is_new_m1: bool = True,
) -> dict[str, Any]:
    from core.phase3_integrated_engine import ExecutionDecision

    from core.phase3_package_spec import uses_defended_phase3_package as _uses_defended
    now_utc = now_utc or datetime.now(timezone.utc)
    profile_preset = str(getattr(profile, "active_preset_name", "") or "").strip().lower()
    policy_id = str(getattr(policy, "id", "") or "").strip().lower()
    requested_preset = str(preset_id or "").strip().lower()
    defended_active = (
        any(
            value == PHASE3_DEFENDED_PRESET_ID
            for value in (profile_preset, policy_id, requested_preset)
        )
        or _uses_defended(requested_preset)
    )
    if not defended_active:
        return {
            "decision": ExecutionDecision(attempted=False, placed=False, reason="phase3_additive: generic phase3 preset disabled during defended rebuild", side=None),
            "phase3_state_updates": {},
            "strategy_tag": None,
            "placements": [],
            "additive_runtime": {"mode": "disabled_non_defended"},
        }

    preset_id = PHASE3_DEFENDED_PRESET_ID
    spec = load_phase3_package_spec(preset_id=preset_id)
    profile_name = getattr(profile, "profile_name", "") or getattr(profile, "name", "") or str(profile)
    open_trades = _open_phase3_trades(store, profile_name)
    v4_result = _execute_v4_runtime(
        adapter=adapter,
        profile=profile,
        spec=spec,
        data_by_tf=data_by_tf,
        tick=tick,
        phase3_state=phase3_state,
        store=store,
        sizing_config=sizing_config or {},
        now_utc=now_utc,
        mode=mode,
        is_new_m1=is_new_m1,
    )
    captured_candidates, diagnostics = _evaluate_family_candidates(
        adapter=adapter,
        profile=profile,
        policy=policy,
        data_by_tf=data_by_tf,
        tick=tick,
        mode=mode,
        phase3_state=phase3_state,
        store=store,
        sizing_config=sizing_config or {},
        now_utc=now_utc,
        ownership_audit=ownership_audit or {},
        overlay_state=overlay_state or {},
    )

    baseline_intents, offensive_intents, baseline_candidates, offensive_candidates = _split_candidate_intents(captured_candidates, spec)
    baseline_outcome = apply_variant_k_baseline_admission(baseline_candidates, data_by_tf=data_by_tf)
    baseline_candidates = _apply_candidate_adjustments(list(baseline_outcome.accepted), baseline_outcome.adjustments)
    baseline_intents = [
        Phase3BaselineIntent(
            strategy_tag=c.strategy_tag,
            strategy_family=c.strategy_family,
            side=c.side,
            entry_time_utc=c.entry_time_utc,
            ownership_cell=c.ownership_cell,
            units=c.units,
            entry_price=c.entry_price,
            sl_price=c.sl_price,
            tp1_price=c.tp1_price,
            reason=c.reason,
            state_updates=dict(c.state_updates or {}),
            raw_result=dict(c.raw_result or {}),
        )
        for c in baseline_candidates
    ]

    accepted: list[Phase3AdditiveCandidate] = []
    rejected: list[dict[str, Any]] = list(baseline_outcome.rejected) + list(v4_result.get("rejected") or [])
    placements: list[dict[str, Any]] = list(v4_result.get("placements") or [])
    state_updates: dict[str, Any] = dict(v4_result.get("state_updates") or {})
    simulated_open = list(open_trades)
    last_margin: Phase3MarginState | None = None

    for candidate in baseline_candidates:
        conflict = _conflict_state(spec, phase3_state, simulated_open, candidate.side, now_utc)
        margin = _margin_state(adapter, spec, candidate.lots, open_trades_count=len(open_trades))
        last_margin = margin
        block_reason = _conflict_block_reason(
            conflict,
            accepted_count=len(accepted),
            apply_max_open_cap=False,
        )
        if block_reason is None and margin.blocked:
            block_reason = margin.reason or "margin_blocked"
        if block_reason:
            rejected.append(
                {
                    "identity": candidate.identity,
                    "strategy_tag": candidate.strategy_tag,
                    "intent_source": candidate.intent_source,
                    "slice_id": candidate.slice_id,
                    "reason": block_reason,
                    "conflict_state": asdict(conflict),
                    "margin_state": asdict(margin),
                }
            )
            continue
        accepted.append(candidate)
        simulated_open.append({"side": candidate.side, "entry_type": candidate.strategy_tag, "intent_source": "baseline"})
        placement = _place_candidate(adapter, profile, candidate)
        placements.append(placement)
        state_updates = _merge_state_updates(state_updates, dict(candidate.state_updates or {}))

    for candidate in offensive_candidates:
        conflict = _conflict_state(spec, phase3_state, simulated_open, candidate.side, now_utc)
        margin = _margin_state(adapter, spec, candidate.lots, open_trades_count=len(open_trades))
        last_margin = margin
        block_reason = _conflict_block_reason(
            conflict,
            accepted_count=len(accepted),
            apply_max_open_cap=True,
        )
        if block_reason is None and margin.blocked:
            block_reason = margin.reason or "margin_blocked"

        if block_reason:
            rejected.append({
                "identity": candidate.identity,
                "strategy_tag": candidate.strategy_tag,
                "intent_source": candidate.intent_source,
                "slice_id": candidate.slice_id,
                "reason": block_reason,
                "conflict_state": asdict(conflict),
                "margin_state": asdict(margin),
            })
            continue

        accepted.append(candidate)
        simulated_open.append({"side": candidate.side, "entry_type": candidate.strategy_tag, "intent_source": "offensive"})
        placement = _place_candidate(adapter, profile, candidate)
        placements.append(placement)
        state_updates = _merge_state_updates(state_updates, dict(candidate.state_updates or {}))

    additive_state = dict(phase3_state.get("additive_runtime") or {})
    entries_by_day = dict(additive_state.get("entries_by_day") or {})
    day_key = now_utc.date().isoformat()
    entries_by_day[day_key] = int(entries_by_day.get(day_key, 0) or 0) + len([p for p in placements if getattr(p.get("decision"), "placed", False)])
    additive_state.update(
        {
            "mode": "defended_additive_runtime_v1",
            "last_bar_utc": pd.Timestamp(now_utc).isoformat(),
            "entries_by_day": entries_by_day,
            "last_candidate_count": len(captured_candidates),
            "last_baseline_count": len(baseline_candidates),
            "last_offensive_count": len(offensive_candidates),
            "last_accepted_count": len(accepted),
            "last_rejected_count": len(rejected),
            "last_open_book_count": len(open_trades),
            "last_candidate_diagnostics": diagnostics[-12:],
            "last_variant_k_diagnostics": list(baseline_outcome.diagnostics[-12:]),
            "last_v4_lifecycle_state": dict((state_updates.get("v4_runtime") or {})).get("lifecycle_state"),
            "last_v4_event": dict((state_updates.get("v4_runtime") or {})).get("last_event"),
            "last_v4_reject_reason": dict((state_updates.get("v4_runtime") or {})).get("last_reject_reason"),
            "last_v4_margin_diag": dict((state_updates.get("v4_runtime") or {})).get("last_margin_diag"),
        }
    )
    state_updates["additive_runtime"] = additive_state

    first = placements[0] if placements else {}
    first_decision = first.get("decision")
    attempted = bool(captured_candidates) or bool(v4_result.get("attempted"))
    placed = any(bool(getattr(p.get("decision"), "placed", False)) for p in placements)
    summary_reason = (
        f"phase3_additive: baseline={len(baseline_candidates)} offensive={len(offensive_candidates)} accepted={len(accepted)} rejected={len(rejected)}"
        if captured_candidates else
        (
            f"phase3_additive: v4_runtime={dict((state_updates.get('v4_runtime') or {})).get('lifecycle_state')}"
            if v4_result.get("attempted")
            else "phase3_additive: no defended candidates"
        )
    )
    summary_decision = ExecutionDecision(
        attempted=attempted,
        placed=placed,
        reason=summary_reason,
        side=getattr(first_decision, "side", None) if first_decision is not None else None,
        order_retcode=getattr(first_decision, "order_retcode", None) if first_decision is not None else None,
        order_id=getattr(first_decision, "order_id", None) if first_decision is not None else None,
        deal_id=getattr(first_decision, "deal_id", None) if first_decision is not None else None,
        fill_price=getattr(first_decision, "fill_price", None) if first_decision is not None else None,
    )
    margin_diag_truth: dict[str, Any] = {}
    if last_margin is not None:
        margin_diag_truth = {
            "margin_closeout_fraction": last_margin.margin_closeout_fraction,
            "margin_ceiling": round(last_margin.margin_ceiling, 2),
            "margin_effective_free": round(last_margin.free_margin, 2),
            "margin_utilization_pct": round(last_margin.margin_utilization_pct, 1),
        }
    envelope = Phase3AdditiveDecisionEnvelope(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        package_id=spec.package_id,
        preset_id=preset_id,
        baseline_intents=[asdict(intent) for intent in baseline_intents],
        offensive_intents=[asdict(intent) for intent in offensive_intents],
        candidates=[asdict(c) for c in (baseline_candidates + offensive_candidates)],
        accepted=[asdict(c) for c in accepted],
        rejected=rejected,
        placements=[{
            "strategy_tag": p.get("strategy_tag"),
            "side": getattr(p.get("decision"), "side", None),
            "placed": bool(getattr(p.get("decision"), "placed", False)),
            "intent_source": (p.get("phase3_additive_candidate") or {}).get("intent_source"),
            "slice_id": (p.get("phase3_additive_candidate") or {}).get("slice_id"),
        } for p in placements],
        conflict_state={
            "open_book_before": len(open_trades),
            "entries_today_before": _entries_today(phase3_state, now_utc),
            "baseline_candidates": len(baseline_candidates),
            "offensive_candidates": len(offensive_candidates),
            "variant_k_rejections": len(baseline_outcome.rejected),
        },
        margin_state={
            "placements": len(placements),
            **margin_diag_truth,
        },
    )
    return {
        "decision": summary_decision,
        "strategy_tag": first.get("strategy_tag"),
        "phase3_state_updates": state_updates,
        "sl_price": first.get("sl_price"),
        "tp1_price": first.get("tp1_price"),
        "entry_price": first.get("entry_price"),
        "units": first.get("units"),
        "risk_usd_planned": first.get("risk_usd_planned"),
        "placements": placements,
        "phase3_additive_envelope": asdict(envelope),
        "phase3_additive_candidates": [asdict(c) for c in (baseline_candidates + offensive_candidates)],
        "phase3_additive_rejected": rejected,
        "phase3_additive_mode": "defended_additive_runtime_v1",
        "v44_exit_plan": first.get("v44_exit_plan") or {},
        "v44_parity_context": first.get("v44_parity_context") or {},
        "phase3_additive_truth": {
            "open_book_count_before": len(open_trades),
            "candidate_count": len(captured_candidates),
            "baseline_candidate_count": len(baseline_candidates),
            "offensive_candidate_count": len(offensive_candidates),
            "variant_k_rejected_count": len(baseline_outcome.rejected),
            "accepted_baseline_count": len([c for c in accepted if c.intent_source == "baseline"]),
            "accepted_offensive_count": len([c for c in accepted if c.intent_source == "offensive"]),
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
            "v4_runtime_state": dict((state_updates.get("v4_runtime") or {})).get("lifecycle_state"),
            "v4_runtime_event": dict((state_updates.get("v4_runtime") or {})).get("last_event"),
            **margin_diag_truth,
        },
    }
