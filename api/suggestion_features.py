from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def compute_suggestion_features(
    suggestion: dict[str, Any],
    ctx: dict[str, Any],
    *,
    generation_time: datetime | None = None,
) -> dict[str, Any]:
    now = generation_time or datetime.now(timezone.utc)
    side = str(suggestion.get("side") or "").strip().lower()
    entry = _safe_float(suggestion.get("price"))
    sl = _safe_float(suggestion.get("sl"))
    tp = _safe_float(suggestion.get("tp"))

    sl_dist = abs(entry - sl) if entry is not None and sl is not None else None
    tp_dist = abs(tp - entry) if entry is not None and tp is not None else None
    planned_rr = (tp_dist / sl_dist) if sl_dist not in (None, 0.0) and tp_dist is not None else None

    session = ctx.get("session") or {}
    active_sessions = [str(s).strip().lower() for s in (session.get("active_sessions") or []) if s]
    overlap = str(session.get("overlap") or "").strip().lower()
    session_label = overlap or "+".join(sorted(active_sessions)) or "none"

    volatility = ctx.get("volatility") or {}
    ta = ctx.get("ta_snapshot") or {}
    h1 = ta.get("H1") or {}
    m5 = ta.get("M5") or {}
    m1 = ta.get("M1") or {}
    macro = (ctx.get("cross_asset_bias") or {})
    order_book = ctx.get("order_book") or {}
    spot = ctx.get("spot_price") or {}

    return {
        "planned_rr": _round_or_none(planned_rr, 3),
        "sl_pips": _distance_to_pips(sl_dist),
        "tp_pips": _distance_to_pips(tp_dist),
        "side": side or None,
        "session": session_label,
        "hour_utc": now.hour,
        "day_of_week": now.strftime("%A"),
        "spread_at_entry": _safe_float(spot.get("spread_pips")),
        "vol_regime": str(volatility.get("label") or "unknown"),
        "atr_m15": _safe_float(ctx.get("atr_m15")),
        "h1_regime": str(h1.get("regime") or "unknown"),
        "m5_regime": str(m5.get("regime") or "unknown"),
        "m1_regime": str(m1.get("regime") or "unknown"),
        "nearest_structure_pips": _nearest_structure_distance(order_book),
        "macro_bias": str(macro.get("combined_bias") or "unknown"),
        "dxy_direction": str((macro.get("dxy") or {}).get("direction") or "unknown"),
        "exit_strategy": str(suggestion.get("exit_strategy") or ""),
        "confidence": str(suggestion.get("quality") or suggestion.get("confidence") or ""),
        "trigger_family": str(suggestion.get("trigger_family") or ""),
        "trigger_reason": str(suggestion.get("trigger_reason") or ""),
        "named_catalyst": str(suggestion.get("named_catalyst") or ""),
        "side_bias_check": str(suggestion.get("side_bias_check") or ""),
        "setup_location": str(suggestion.get("setup_location") or ""),
        "edge_reason": str(suggestion.get("edge_reason") or ""),
        "adverse_context": str(suggestion.get("adverse_context") or ""),
        "caveat_resolution": str(suggestion.get("caveat_resolution") or ""),
        "micro_confirmation_event": str(suggestion.get("micro_confirmation_event") or ""),
        "reasoning_quality_gate": str(suggestion.get("reasoning_quality_gate") or ""),
        "selectivity_adjustments": suggestion.get("selectivity_adjustments") or [],
        "max_allowed_lots": _safe_float(suggestion.get("max_allowed_lots")),
        "original_model_lots": _safe_float(suggestion.get("original_model_lots")),
        "phase4_catalyst_score": _safe_float(suggestion.get("phase4_catalyst_score")),
        "phase4_green_matches": suggestion.get("phase4_green_matches") or [],
        "phase4_weakness_signals": suggestion.get("phase4_weakness_signals") or [],
        "phase5_reasoning_flags": suggestion.get("phase5_reasoning_flags") or [],
        "phase5_material_resolution_score": _safe_float(suggestion.get("phase5_material_resolution_score")),
    }


def _safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _distance_to_pips(distance: float | None) -> float | None:
    if distance is None:
        return None
    return round(float(distance) * 100.0, 1)


def _round_or_none(value: float | None, digits: int) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _nearest_structure_distance(order_book: dict[str, Any]) -> float | None:
    distances: list[float] = []
    for key in ("nearest_support_distance_pips", "nearest_resistance_distance_pips"):
        value = _safe_float(order_book.get(key))
        if value is not None:
            distances.append(value)
    if not distances:
        return None
    return round(min(distances), 1)
