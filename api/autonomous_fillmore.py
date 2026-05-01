"""Autonomous Fillmore — lets the AI place trades on its own, gated by a code-side
signal filter so we don't spam the LLM.

Architecture (three layers):

    tick →  L1 hard filters  → L2 signal gate (mode-dependent) → L3 adaptive throttle
                 │                       │                           │
                 └── block early ────────┴── decides whether to wake up the LLM ──┘

Only when all three layers pass do we call OpenAI via the existing
`ai_suggest_trade` endpoint logic, and if the model returns lots>0 we
auto-place through `place_limit_order_endpoint`.

All config and the rolling decisions log live in `runtime_state.json` under the
`autonomous_fillmore` key so the run loop and the HTTP API share one source of
truth without needing a new DB migration.

See MEMORY.md ("Autonomous Fillmore" planning notes) for the design discussion.
"""

from __future__ import annotations

import json
import math
import os
import re
import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd

from api import autonomous_performance
from api.fillmore_llm_guard import FillmoreLLMCircuitOpenError, get_fillmore_llm_health, run_guarded_fillmore_llm_call
from core.execution_state import load_state as load_execution_state
from core.json_state import load_json_state, save_json_state
from core.indicators import bollinger_bands

# -----------------------------------------------------------------------------
# Config defaults + types
# -----------------------------------------------------------------------------

AutonomousMode = Literal["off", "shadow", "paper", "live"]
Aggressiveness = Literal["conservative", "balanced", "aggressive", "very_aggressive"]
OrderType = Literal["market", "limit"]  # kept for _place_from_suggestion; LLM chooses per-suggestion

# Rough cost per 1M tokens (input/output) for the default model. Used to estimate
# budget spend from tokens — not authoritative, just directional.
# gpt-5.4-mini is priced similarly to gpt-5-mini / gpt-4o-mini tier.
_MODEL_COST_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-5.4-mini": (0.15, 0.60),
    "gpt-5.4":      (2.50, 10.00),
    "gpt-5-mini":   (0.15, 0.60),
    "gpt-4o-mini":  (0.15, 0.60),
    "gpt-4o":       (2.50, 10.00),
}

# Estimated prompt/response sizes for the autonomous call. These are
# pessimistic-ish so the budget stats trend slightly conservative.
_ASSUMED_INPUT_TOKENS = 4000
_ASSUMED_OUTPUT_TOKENS = 350
AUTONOMOUS_PROMPT_VERSION = "autonomous_phase5_reasoning_quality_v1"

DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": False,
    "mode": "off",                      # off|shadow|paper|live — paper = real OANDA practice orders (not in-app sim)
    "aggressiveness": "balanced",
    "limit_gtd_minutes": 15,            # autonomous limits are short-lived; stale scalp ideas should die quickly
    "daily_budget_usd": 2.00,
    "min_llm_cooldown_sec": 60,
    "trading_hours": {
        "tokyo": True,
        "london": True,
        "ny": True,
    },
    "max_lots_per_trade": 15.0,          # hard ceiling — LLM cannot exceed this
    "base_lot_size": 5.0,                # anchor for LLM lot sizing — "normal" trade size
    "lot_deviation": 4.0,                # LLM sizes from (base - dev) to (base + dev), clamped to [1, max_lots]
    "max_open_ai_trades": 6,
    "max_daily_loss_usd": 50.0,
    "max_consecutive_errors": 5,
    "model": "gpt-5.4-mini",
    # adaptive throttle
    "throttle_no_trade_streak": 5,       # after N "no trade" LLM replies, pause
    "throttle_no_trade_cooldown_sec": 300,
    "throttle_loss_streak": 2,
    "throttle_loss_cooldown_sec": 900,
    # Stage 3: book-correlation veto + same-setup dedupe
    "correlation_veto_enabled": False,   # discretionary mode: allow stacking/hedging decisions by the model
    "correlation_distance_pips": 15.0,
    "repeat_setup_dedupe_enabled": False,# discretionary mode: allow repeat-firing when model still sees edge
    "repeat_setup_window_min": 30,
    "repeat_setup_bucket_pips": 25.0,
    # Stage 6: event blackout + multi-trade planning
    "event_blackout_enabled": False,     # discretionary mode: let model trade event windows if edge is present
    "event_blackout_minutes": 30,
    "multi_trade_enabled": True,         # LLM can propose 0-2 setups per call instead of exactly 1
    "max_suggestions_per_call": 2,
    "streak_scratch_threshold_pips": 1.0,
    "min_lot_size": 0.01,
}

_PHASE4_GENERIC_TEXT = {
    "n/a", "na", "none", "null", "default", "-", "see thesis", "see analysis",
}
_PHASE4_GENERIC_CATALYSTS = {
    "level reject", "level reaction", "support reclaim", "reclaimed support",
    "resistance reject", "rejected resistance", "pullback", "pullback fade",
    "fade", "fade in chop", "trend continuation", "continuation", "support",
    "resistance", "structure", "technical setup", "price action",
    *_PHASE4_GENERIC_TEXT,
}
_PHASE4_STRUCTURE_TOKENS = (
    "support", "resistance", "reclaim", "reject", "half_yen", "half yen",
    "whole_yen", "whole yen", "session high", "session low", "oanda cluster",
    "liquidity cluster", "range high", "range low", "trendline", "ema", "vwap",
)
_PHASE4_MICRO_TOKENS = (
    "micro", "m1", "m3", "m5", "confirmation", "acceptance", "break",
    "retest", "impulse", "sweep", "hold", "close above", "close below",
    "higher low", "lower high", "failed break", "failed breakdown",
)
_PHASE4_MATERIAL_PHRASES = (
    "boj", "mof", "finance minister", "intervention", "rate check",
    "hawkish surprise", "dovish surprise", "policy shift", "policy divergence",
    "macro catalyst", "macro release", "macro surprise", "flow shift",
    "safe-haven", "safe haven", "safe-haven flow", "safe haven flow",
    "geopolitical risk", "war premium", "war-premium", "us-japan",
    "treasury yield", "yield spike", "yield compression", "cpi", "nfp",
    "fomc", "volatility regime", "liquidity shift", "fixing flow",
    "option expiry", "real money", "material change", "prior failure",
    "failed prior", "break of structure", "regime shift",
)
_PHASE4_MATERIAL_NEGATIONS = (
    "no", "not", "without", "absent", "contradict", "contradicts",
    "contradicted", "mixed", "unclear", "generic", "does not confirm",
    "doesn't confirm", "fails to confirm", "not confirmed", "not material",
)
_PHASE4_ADVERSE_CONTEXT_PATTERNS = (
    r"\bdespite\b",
    r"\beven though\b",
    r"\balthough\b",
    r"\bhowever\b",
    r"\bbut\b.{0,80}\b(h1|m15|m5|m1|policy|macro|cross|jpy|yen|tape|trend|structure|session|intraday|broader)\b",
    r"\bmixed (alignment|tape|structure|backdrop|context|trend|signal|signals)\b",
    r"\b(tape|macro|policy|backdrop|context|trend|structure).{0,30}\bmixed\b",
    r"\b(contradict|contradicts|contradicted|contradiction|conflict|conflicts|conflicting)\b",
    r"\bdoes not confirm\b",
    r"\bdoesn't confirm\b",
    r"\bfails to confirm\b",
    r"\b(jpy|yen)[ -]?weak\b",
    r"\b(jpy|yen).{0,20}\bweak\b",
    r"\bweak (jpy|yen)\b",
    r"\bbroader .{0,40}(pressure|weakness|headwind)\b",
    r"\bnot enough to cancel\b",
    r"\bstructurally weaker\b",
)
_PHASE4_WEAK_PERMISSION_PATTERNS = (
    r"\bprobe\b",
    r"\btactical\b",
    r"\bthin\b",
    r"\bmarginal\b",
    r"\breduced[- ]conviction\b",
    r"\blow[- ]conviction\b",
    r"\bsmall(?:est)? size\b",
    r"\bsize (?:stays|is|kept) (?:small|minimal|reduced)\b",
    r"\bmust be quick\b",
    r"\bquick .*?(?:because|given|despite|while)\b",
)
_PHASE4_MICRO_CONFIRMATION_PATTERNS = (
    r"\bmicro[- ]?confirm",
    r"\bm[135] confirm",
    r"\bm[135] confirmation",
    r"\breclaimed_support\b",
    r"\brejected_resistance\b",
)
_PHASE4_SPECIFIC_MICRO_EVENT_TOKENS = (
    "m1 close", "m3 close", "m5 close", "closed above", "closed below",
    "close above", "close below", "acceptance above", "acceptance below",
    "hold above", "hold below", "held above", "held below", "retest",
    "sweep", "wick", "higher low", "lower high", "higher high", "lower low",
    "failed break", "failed breakdown", "breakout retest", "breakdown retest",
    "ema reclaim", "ema rejection", "vwap reclaim", "vwap rejection",
)
_PHASE4_REASONING_TEXT_KEYS = (
    "trade_thesis", "named_catalyst", "side_bias_check", "edge_reason",
    "adverse_context", "caveat_resolution", "micro_confirmation_event",
    "countertrend_edge", "why_trade_despite_weakness", "why_not_stop",
    "low_rr_edge", "rationale",
)


def _phase4_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _phase4_contains_any(text: str, tokens: tuple[str, ...]) -> bool:
    return any(token in text for token in tokens)


def _phase4_phrase_regex(phrase: str) -> re.Pattern[str]:
    parts = [re.escape(part) for part in re.split(r"[\s_-]+", phrase.strip()) if part]
    body = r"[\s_-]+".join(parts)
    return re.compile(rf"(?<![a-z0-9]){body}(?![a-z0-9])")


def _phase4_regex_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def _phase4_has_clean_material_phrase(text: str) -> bool:
    for phrase in _PHASE4_MATERIAL_PHRASES:
        for match in _phase4_phrase_regex(phrase).finditer(text):
            window = text[max(0, match.start() - 40): min(len(text), match.end() + 40)]
            if any(_phase4_phrase_regex(neg).search(window) for neg in _PHASE4_MATERIAL_NEGATIONS):
                continue
            return True
    return False


def _phase4_has_meaningful_text(value: Any) -> bool:
    text = _phase4_text(value)
    return bool(text and text not in _PHASE4_GENERIC_TEXT)


def _phase4_catalyst_score(value: Any) -> int:
    """0 generic, 1 structure-only, 2 micro-confirmed, 3 material catalyst."""
    text = _phase4_text(value)
    stripped = text.strip(" .:-")
    if not stripped or stripped in _PHASE4_GENERIC_CATALYSTS or len(stripped) < 12:
        return 0
    if _phase4_has_clean_material_phrase(stripped):
        return 3
    has_structure = _phase4_contains_any(stripped, _PHASE4_STRUCTURE_TOKENS)
    has_micro = _phase4_contains_any(stripped, _PHASE4_MICRO_TOKENS)
    has_level = bool(re.search(r"\d", stripped))
    if has_structure and has_micro:
        return 2
    if has_micro and has_level:
        return 2
    if has_structure or has_level:
        return 1
    return 1 if len(stripped.split()) >= 4 else 0


def _phase4_reasoning_text(suggestion: dict[str, Any]) -> str:
    pieces: list[str] = []
    for key in _PHASE4_REASONING_TEXT_KEYS:
        value = suggestion.get(key)
        if value not in (None, ""):
            pieces.append(str(value))
    return _phase4_text(" ".join(pieces))


def _phase4_material_resolution_score(suggestion: dict[str, Any]) -> int:
    return max(
        _phase4_catalyst_score(suggestion.get("caveat_resolution")),
        _phase4_catalyst_score(suggestion.get("edge_reason")),
        _phase4_catalyst_score(suggestion.get("named_catalyst")),
        _phase4_catalyst_score(suggestion.get("trade_thesis")),
    )


def _phase4_has_adverse_context(suggestion: dict[str, Any]) -> bool:
    text = _phase4_reasoning_text(suggestion)
    if not text:
        return False
    return (
        _phase4_regex_any(text, _PHASE4_ADVERSE_CONTEXT_PATTERNS)
        or _phase4_regex_any(text, _PHASE4_WEAK_PERMISSION_PATTERNS)
    )


def _phase4_has_specific_micro_event(suggestion: dict[str, Any]) -> bool:
    explicit_event = _phase4_text(suggestion.get("micro_confirmation_event"))
    if explicit_event and _phase4_contains_any(explicit_event, _PHASE4_SPECIFIC_MICRO_EVENT_TOKENS):
        return True
    return _phase4_contains_any(_phase4_reasoning_text(suggestion), _PHASE4_SPECIFIC_MICRO_EVENT_TOKENS)


def _phase4_has_vague_micro_confirmation(suggestion: dict[str, Any]) -> bool:
    text = _phase4_reasoning_text(suggestion)
    if not text or not _phase4_regex_any(text, _PHASE4_MICRO_CONFIRMATION_PATTERNS):
        return False
    return not _phase4_has_specific_micro_event(suggestion)


def _phase4_session_is_london_ny(ctx: dict[str, Any]) -> bool:
    session = (ctx.get("session") or {}) if isinstance(ctx, dict) else {}
    labels = [
        _phase4_text(session.get("overlap")),
        *[_phase4_text(item) for item in (session.get("active_sessions") or [])],
    ]
    joined = " ".join(labels)
    return "london" in joined or "new york" in joined or any(label == "ny" for label in labels)


def _phase4_h1_is_bull(ctx: dict[str, Any]) -> bool:
    ta = (ctx.get("ta_snapshot") or {}) if isinstance(ctx, dict) else {}
    h1 = ta.get("H1") or {}
    text = _phase4_text(" ".join(str(v) for v in h1.values())) if isinstance(h1, dict) else _phase4_text(h1)
    return any(token in text for token in ("bull", "uptrend", "higher high", "higher low"))


def _phase4_weakness_signals(suggestion: dict[str, Any]) -> list[str]:
    signals: list[str] = []
    tf = _phase4_text(suggestion.get("timeframe_alignment"))
    if tf in {"mixed", "countertrend"}:
        signals.append(f"timeframe_alignment:{tf}")
    repeat_case = _phase4_text(suggestion.get("repeat_trade_case"))
    if repeat_case == "blind_retry":
        signals.append("blind_retry")
    zone = _phase4_text(suggestion.get("zone_memory_read"))
    if zone in {"failing_zone", "unresolved_chop"}:
        signals.append(f"zone_memory:{zone}")
    try:
        rr = float(suggestion.get("planned_rr_estimate"))
    except (TypeError, ValueError):
        rr = None
    if rr is not None and rr < 1.0:
        signals.append("planned_rr_below_1")
    if _phase4_has_meaningful_text(suggestion.get("why_trade_despite_weakness")):
        signals.append("weakness_admitted")
    return signals


def _phase4_green_pattern_matches(suggestion: dict[str, Any], ctx: dict[str, Any]) -> list[str]:
    side = _phase4_text(suggestion.get("side"))
    family = _phase4_text(suggestion.get("trigger_family"))
    fit = _phase4_text(suggestion.get("trigger_fit"))
    tf = _phase4_text(suggestion.get("timeframe_alignment"))
    catalyst_text = _phase4_text(suggestion.get("named_catalyst"))
    thesis_text = _phase4_text(suggestion.get("trade_thesis"))
    trigger_reason = _phase4_text(suggestion.get("trigger_reason"))
    combined = " ".join([catalyst_text, thesis_text, trigger_reason])
    matches: list[str] = []
    is_buy_critical_level = (
        side == "buy"
        and family == "critical_level_reaction"
        and fit == "level_reaction"
        and tf in {"aligned", "mixed"}
        and _phase4_contains_any(combined, _PHASE4_STRUCTURE_TOKENS)
    )
    if is_buy_critical_level and _phase4_session_is_london_ny(ctx) and _phase4_h1_is_bull(ctx):
        matches.append("buy_london_ny_h1_bull_critical_level")
    return matches


def _phase4_reasoning_flags(suggestion: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    family = _phase4_text(suggestion.get("trigger_family"))
    tf = _phase4_text(suggestion.get("timeframe_alignment"))
    if _phase4_has_adverse_context(suggestion):
        flags.append("contradiction_admitted")
    if family == "critical_level_reaction" and tf == "mixed":
        flags.append("critical_level_mixed")
    if _phase4_has_vague_micro_confirmation(suggestion):
        flags.append("vague_micro_confirmation")
    return flags


def _phase4_reasoning_veto_reason(suggestion: dict[str, Any]) -> str | None:
    flags = set(_phase4_reasoning_flags(suggestion))
    material_resolution = _phase4_material_resolution_score(suggestion) >= 3
    if "critical_level_mixed" in flags and not material_resolution:
        return "server_veto:critical_level_mixed_reasoning_risk"
    if "vague_micro_confirmation" in flags and not material_resolution:
        return "server_veto:vague_micro_confirmation"
    if "contradiction_admitted" in flags and not material_resolution:
        return "server_veto:contradiction_admitted_no_material_resolution"
    return None


def _phase4_apply_selectivity_sizing(suggestion: dict[str, Any], ctx: dict[str, Any]) -> None:
    if _phase4_text(suggestion.get("decision")) != "trade" or float(suggestion.get("lots") or 0.0) <= 0:
        return
    original_lots = float(suggestion.get("lots") or 0.0)
    catalyst_score = _phase4_catalyst_score(suggestion.get("named_catalyst"))
    green_matches = _phase4_green_pattern_matches(suggestion, ctx)
    weakness_signals = _phase4_weakness_signals(suggestion)
    reasoning_flags = _phase4_reasoning_flags(suggestion)
    material_resolution_score = _phase4_material_resolution_score(suggestion)
    side = _phase4_text(suggestion.get("side"))
    family = _phase4_text(suggestion.get("trigger_family"))
    tf = _phase4_text(suggestion.get("timeframe_alignment"))
    adjustments: list[str] = []

    suggestion["phase4_catalyst_score"] = catalyst_score
    suggestion["phase4_green_matches"] = green_matches
    suggestion["phase4_weakness_signals"] = weakness_signals
    suggestion["phase5_reasoning_flags"] = reasoning_flags
    suggestion["phase5_material_resolution_score"] = material_resolution_score

    def _cap_lots(max_lots: float, reason: str) -> None:
        current = float(suggestion.get("lots") or 0.0)
        if current <= max_lots and original_lots <= max_lots:
            return
        if reason not in adjustments:
            adjustments.append(reason)
        if current <= max_lots:
            return
        suggestion["lots"] = max_lots
        existing_cap = suggestion.get("max_allowed_lots")
        try:
            existing_cap_f = float(existing_cap) if existing_cap is not None else max_lots
        except (TypeError, ValueError):
            existing_cap_f = max_lots
        suggestion["max_allowed_lots"] = min(existing_cap_f, max_lots)

    clean_large_lot = (
        tf == "aligned"
        and not weakness_signals
        and bool(green_matches)
        and catalyst_score >= 2
    )
    if original_lots >= 8.0 and not clean_large_lot:
        _cap_lots(1.0, "phase4_large_lot_clean_setup_only")
    if side == "sell" and weakness_signals and catalyst_score < 3:
        _cap_lots(1.0, "phase4_weak_sell_max_1_lot_unless_material")
    if family == "critical_level_reaction" and tf == "mixed":
        _cap_lots(1.0, "phase4_critical_level_mixed_max_1_lot")
    if "contradiction_admitted" in reasoning_flags:
        _cap_lots(1.0, "phase5_contradiction_admitted_max_1_lot")
    if "vague_micro_confirmation" in reasoning_flags:
        _cap_lots(1.0, "phase5_vague_micro_confirmation_max_1_lot")

    if adjustments:
        existing = suggestion.get("selectivity_adjustments")
        if not isinstance(existing, list):
            existing = []
        suggestion["selectivity_adjustments"] = [*existing, *adjustments]
        suggestion["original_model_lots"] = original_lots

# Per-mode gate thresholds. Lower = easier to pass = more LLM calls.
# These are the *signal* filters applied after hard filters pass.
GATE_THRESHOLDS: dict[str, dict[str, Any]] = {
    "conservative": {
        "description": "Hybrid trigger: strict critical-level reaction, Tokyo tight-range mean reversion, or strict trend expansion near actionable USDJPY structure.",
        "critical_level_max_pips": 5.0,
        "critical_micro_window_bars": 3,
        "critical_touch_tolerance_pips": 0.6,
        "tokyo_meanrev_window_bars": 24,
        "tokyo_meanrev_min_session_bars": 45,
        "tokyo_meanrev_max_range_pips": 16.0,
        "tokyo_meanrev_range_atr_mult": 3.2,
        "tokyo_meanrev_edge_fraction": 0.22,
        "tokyo_meanrev_touch_tolerance_pips": 0.6,
        "tokyo_meanrev_adx_ceiling": 20.0,
        "tokyo_meanrev_bb_width_max": 0.0007,
        "tokyo_meanrev_min_reward_pips": 3.0,
        "trend_adx_min": 25.0,
        "trend_extension_atr_mult": 0.9,
        "require_min_m5_atr_pips": 3.0,
        "require_m3_trend": True,
        "require_m1_stack": True,
        "expected_pass_rate_pct": 1.5,
    },
    "balanced": {
        "description": "Hybrid trigger: critical-level reaction, Tokyo tight-range mean reversion, or trend expansion for ripe-now USDJPY setups.",
        "critical_level_max_pips": 6.0,
        "critical_micro_window_bars": 3,
        "critical_touch_tolerance_pips": 0.8,
        "tokyo_meanrev_window_bars": 20,
        "tokyo_meanrev_min_session_bars": 45,
        "tokyo_meanrev_max_range_pips": 18.0,
        "tokyo_meanrev_range_atr_mult": 3.6,
        "tokyo_meanrev_edge_fraction": 0.24,
        "tokyo_meanrev_touch_tolerance_pips": 0.8,
        "tokyo_meanrev_adx_ceiling": 22.0,
        "tokyo_meanrev_bb_width_max": 0.0008,
        "tokyo_meanrev_min_reward_pips": 2.5,
        "compression_level_max_pips": 4.0,
        "compression_window_bars": 12,
        "compression_range_atr_mult": 0.9,
        "compression_edge_fraction": 0.2,
        "compression_adx_floor": 14.0,
        "compression_adx_ceiling": 24.0,
        "momentum_adx_min": 18.0,
        "momentum_clear_path_pips": 8.0,
        "momentum_lookback_bars": 10,
        "momentum_pullback_zone_pips": 3.0,
        "momentum_extension_atr_mult": 1.25,
        "trend_adx_min": 23.0,
        "trend_extension_atr_mult": 1.0,
        "require_min_m5_atr_pips": 3.0,
        "require_m3_trend": True,
        "require_m1_stack": True,
        "expected_pass_rate_pct": 3.5,
    },
    "aggressive": {
        "description": "Hybrid trigger: looser critical-level reaction, Tokyo tight-range mean reversion, or trend expansion, still structure-aware.",
        "critical_level_max_pips": 8.0,
        "critical_micro_window_bars": 2,
        "critical_touch_tolerance_pips": 1.0,
        "tokyo_meanrev_window_bars": 16,
        "tokyo_meanrev_min_session_bars": 36,
        "tokyo_meanrev_max_range_pips": 20.0,
        "tokyo_meanrev_range_atr_mult": 4.0,
        "tokyo_meanrev_edge_fraction": 0.28,
        "tokyo_meanrev_touch_tolerance_pips": 1.0,
        "tokyo_meanrev_adx_ceiling": 24.0,
        "tokyo_meanrev_bb_width_max": 0.001,
        "tokyo_meanrev_min_reward_pips": 2.0,
        "compression_level_max_pips": 5.0,
        "compression_window_bars": 10,
        "compression_range_atr_mult": 1.0,
        "compression_edge_fraction": 0.25,
        "compression_adx_floor": 12.0,
        "compression_adx_ceiling": 22.0,
        "momentum_adx_min": 16.0,
        "momentum_clear_path_pips": 7.0,
        "momentum_lookback_bars": 8,
        "momentum_pullback_zone_pips": 3.5,
        "momentum_extension_atr_mult": 1.4,
        "trend_adx_min": 20.0,
        "trend_extension_atr_mult": 1.25,
        "require_m3_trend": False,
        "require_m1_stack": False,
        "require_any_trend_signal": True,
        "reject_m3_m1_mismatch_if_both_present": True,
        "require_min_m5_atr_pips": 3.0,
        "expected_pass_rate_pct": 2.5,
    },
    "very_aggressive": {
        "description": "Legacy alias for balanced hybrid trigger.",
        "critical_level_max_pips": 6.0,
        "critical_micro_window_bars": 3,
        "critical_touch_tolerance_pips": 0.8,
        "tokyo_meanrev_window_bars": 20,
        "tokyo_meanrev_min_session_bars": 45,
        "tokyo_meanrev_max_range_pips": 18.0,
        "tokyo_meanrev_range_atr_mult": 3.6,
        "tokyo_meanrev_edge_fraction": 0.24,
        "tokyo_meanrev_touch_tolerance_pips": 0.8,
        "tokyo_meanrev_adx_ceiling": 22.0,
        "tokyo_meanrev_bb_width_max": 0.0008,
        "tokyo_meanrev_min_reward_pips": 2.5,
        "compression_level_max_pips": 4.0,
        "compression_window_bars": 12,
        "compression_range_atr_mult": 0.9,
        "compression_edge_fraction": 0.2,
        "compression_adx_floor": 14.0,
        "compression_adx_ceiling": 24.0,
        "momentum_adx_min": 18.0,
        "momentum_clear_path_pips": 8.0,
        "momentum_lookback_bars": 10,
        "momentum_pullback_zone_pips": 3.0,
        "momentum_extension_atr_mult": 1.25,
        "trend_adx_min": 23.0,
        "trend_extension_atr_mult": 1.0,
        "require_m3_trend": False,
        "require_m1_stack": False,
        "require_any_trend_signal": True,
        "reject_m3_m1_mismatch_if_both_present": True,
        "require_min_m5_atr_pips": 3.0,
        "expected_pass_rate_pct": 3.5,
    },
}

# Session-specific calibrated overrides (train/test validated).
# Keys: (aggressiveness, session_label). Values: partial dicts merged onto the
# base GATE_THRESHOLDS for that mode.  Only "safe non-regressive" candidates
# are adopted here — see research_out/autonomous_gate_calibration_1000k_sessions.md.
SESSION_GATE_OVERRIDES: dict[tuple[str, str], dict[str, Any]] = {
    ("balanced", "tokyo"): {
        "require_min_m5_atr_pips": 3.0,
        "critical_level_max_pips": 5.0,
        "trend_adx_min": 25.0,
        "trend_extension_atr_mult": 0.85,
    },
    ("balanced", "ny"): {
        "require_min_m5_atr_pips": 3.5,
        "critical_level_max_pips": 7.0,
        "trend_adx_min": 24.0,
        "trend_extension_atr_mult": 1.1,
    },
    ("balanced", "london/ny"): {
        "require_min_m5_atr_pips": 3.0,
        "critical_level_max_pips": 6.0,
        "trend_adx_min": 23.0,
        "trend_extension_atr_mult": 1.0,
    },
    ("balanced", "london"): {
        "require_min_m5_atr_pips": 3.0,
        "critical_level_max_pips": 6.0,
        "trend_adx_min": 22.0,
        "trend_extension_atr_mult": 1.05,
    },
    ("aggressive", "ny"): {
        "require_min_m5_atr_pips": 3.0,
        "critical_level_max_pips": 8.0,
        "trend_adx_min": 22.0,
        "trend_extension_atr_mult": 1.3,
    },
    ("aggressive", "london"): {
        "require_min_m5_atr_pips": 3.0,
        "critical_level_max_pips": 7.0,
        "trend_adx_min": 20.0,
        "trend_extension_atr_mult": 1.2,
    },
    ("aggressive", "tokyo"): {
        "require_min_m5_atr_pips": 3.0,
        "critical_level_max_pips": 6.0,
        "trend_adx_min": 23.0,
        "trend_extension_atr_mult": 1.0,
    },
    ("aggressive", "london/ny"): {
        "require_min_m5_atr_pips": 3.0,
        "critical_level_max_pips": 7.0,
        "trend_adx_min": 21.0,
        "trend_extension_atr_mult": 1.2,
    },
}


def _resolve_gate_thresholds(agg: str, session_label: str) -> dict[str, Any]:
    base = dict(GATE_THRESHOLDS.get(agg) or GATE_THRESHOLDS["balanced"])
    overrides = SESSION_GATE_OVERRIDES.get((agg, session_label))
    if overrides:
        base.update(overrides)
    return base


_AUTONOMOUS_PROMPT_SKELETON_ID = "autonomous_decision_request_v2_zone_memory_custom_exit"
_AUTONOMOUS_AUX_MEMORY_BUDGET_WORDS = 2400
_AUTONOMOUS_LIMIT_MIN_OFFSET_PIPS = 0.1
_AUTONOMOUS_LIMIT_MAX_OFFSET_PIPS = 0.5
_AUTONOMOUS_MAX_DAILY_LOSS_USD = 50.0
_AUTONOMOUS_MAX_LOTS = 15.0
_AUTONOMOUS_MAX_CONSECUTIVE_ERRORS = 5
_AUTONOMOUS_MAX_LIMIT_GTD_MINUTES = 15
_GATE_SETUP_COOLDOWN_MINUTES = {
    "critical_level_reaction": 30,
    "tight_range_mean_reversion": 20,
    "compression_breakout": 20,
    "momentum_continuation": 20,
    "trend_expansion": 30,
    "failed_breakout_reversal_overlap_v1": 20,
    "post_spike_retracement_ny_overlap_v1": 20,
}
_CRITICAL_LEVEL_SETUP_BUCKET_PIPS = 25.0
_ENABLE_CRITICAL_RESISTANCE_REJECT = True
_ENABLE_COMPRESSION_BREAKOUT = False
_ENABLE_FAILED_BREAKOUT_REVERSAL = False
_ENABLE_POST_SPIKE_RETRACEMENT = False
_DISABLED_SUPPORT_RECLAIM_LEVEL_LABELS = {
    "LOCAL_RANGE_HIGH",
    "LOCAL_RANGE_LOW",
    "TOKYO_SESSION_LOW",
    "TODAY_LOW",
    "PWH",
}
_TREND_EXPANSION_SETUP_BUCKET_PIPS = 20.0
_TREND_EXPANSION_ALLOWED_SESSIONS = {"london/ny"}
_EQUAL_PRIORITY_GATE_FAMILIES = {"critical_level_reaction", "momentum_continuation"}
_TOKYO_CONTROLLED_EXPERIMENT_FAMILIES = {"tight_range_mean_reversion"}
_FAILED_BREAKOUT_ALLOWED_SESSIONS = {"london/ny"}
_FAILED_BREAKOUT_MIN_BREAK_PIPS = 2.0
_FAILED_BREAKOUT_MAX_BREAK_PIPS = 5.0
_FAILED_BREAKOUT_MAX_HOLD_BARS = 2
_FAILED_BREAKOUT_MIN_SESSION_BARS = 9
_FAILED_BREAKOUT_RECAPTURE_BODY_RATIO = 0.5
_FAILED_BREAKOUT_CONTINUATION_INVALIDATION_PIPS = 8.0
_POST_SPIKE_ALLOWED_SESSIONS = {"ny", "london/ny"}
_POST_SPIKE_WINDOW_BARS = 5
_POST_SPIKE_MIN_MOVE_PIPS = 12.0
_POST_SPIKE_MIN_DIRECTIONAL_CONSISTENCY = 0.60
_POST_SPIKE_MAX_CONFIRMATION_BARS = 3
_POST_SPIKE_ALLOWED_CONFIRMATION_BARS = {2, 3}
_POST_SPIKE_STALL_BODY_FRACTION = 0.60
_POST_SPIKE_MAX_EXTENSION_AFTER_SPIKE_PIPS = 4.0
_USDJPY_SPREAD_LIMITS_PIPS = {
    "tokyo": 3.0,
    "london": 3.0,
    "ny": 3.0,
    "london/ny": 3.0,
    "off-hours": 3.0,
}


def _support_reclaim_level_allowed(level_label: Any) -> bool:
    label = str(level_label or "").strip().upper()
    if not label:
        return False
    return label not in _DISABLED_SUPPORT_RECLAIM_LEVEL_LABELS


def _autonomous_prompt_hash() -> str:
    return hashlib.sha256(_AUTONOMOUS_PROMPT_SKELETON_ID.encode("utf-8")).hexdigest()[:12]


def _word_count(text: str) -> int:
    return len(str(text or "").split())


def _sanitize_autonomous_config(cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(cfg)

    agg = str(out.get("aggressiveness") or "balanced").lower()
    if agg == "very_aggressive":
        agg = "balanced"
    if agg not in GATE_THRESHOLDS:
        agg = "balanced"
    out["aggressiveness"] = agg

    try:
        out["max_daily_loss_usd"] = min(float(out.get("max_daily_loss_usd") or _AUTONOMOUS_MAX_DAILY_LOSS_USD), _AUTONOMOUS_MAX_DAILY_LOSS_USD)
    except (TypeError, ValueError):
        out["max_daily_loss_usd"] = _AUTONOMOUS_MAX_DAILY_LOSS_USD
    try:
        out["max_lots_per_trade"] = min(float(out.get("max_lots_per_trade") or _AUTONOMOUS_MAX_LOTS), _AUTONOMOUS_MAX_LOTS)
    except (TypeError, ValueError):
        out["max_lots_per_trade"] = _AUTONOMOUS_MAX_LOTS
    try:
        out["max_consecutive_errors"] = min(int(out.get("max_consecutive_errors") or _AUTONOMOUS_MAX_CONSECUTIVE_ERRORS), _AUTONOMOUS_MAX_CONSECUTIVE_ERRORS)
    except (TypeError, ValueError):
        out["max_consecutive_errors"] = _AUTONOMOUS_MAX_CONSECUTIVE_ERRORS
    try:
        out["limit_gtd_minutes"] = min(int(out.get("limit_gtd_minutes") or _AUTONOMOUS_MAX_LIMIT_GTD_MINUTES), _AUTONOMOUS_MAX_LIMIT_GTD_MINUTES)
    except (TypeError, ValueError):
        out["limit_gtd_minutes"] = _AUTONOMOUS_MAX_LIMIT_GTD_MINUTES
    try:
        saved_max_open = int(out.get("max_open_ai_trades") or DEFAULT_CONFIG.get("max_open_ai_trades") or 6)
    except (TypeError, ValueError):
        saved_max_open = int(DEFAULT_CONFIG.get("max_open_ai_trades") or 6)
    # Normalize stale saved profile configs back to the shared validated autonomous envelope.
    out["max_open_ai_trades"] = max(saved_max_open, int(DEFAULT_CONFIG.get("max_open_ai_trades") or 6))

    try:
        out["base_lot_size"] = max(1.0, min(float(out.get("base_lot_size") or 5.0), float(out["max_lots_per_trade"])))
    except (TypeError, ValueError):
        out["base_lot_size"] = min(5.0, float(out["max_lots_per_trade"]))
    try:
        out["lot_deviation"] = max(0.0, min(float(out.get("lot_deviation") or 4.0), float(out["max_lots_per_trade"])))
    except (TypeError, ValueError):
        out["lot_deviation"] = min(4.0, float(out["max_lots_per_trade"]))

    trading_hours = dict(DEFAULT_CONFIG.get("trading_hours") or {})
    saved_hours = out.get("trading_hours")
    if isinstance(saved_hours, dict):
        trading_hours.update(saved_hours)
    out["trading_hours"] = trading_hours
    # Discretion-first policy: AI should decide stacking/hedging/event behavior.
    out["correlation_veto_enabled"] = False
    out["repeat_setup_dedupe_enabled"] = False
    out["event_blackout_enabled"] = False
    return out


def _apply_autonomous_limit_price_strategy(
    suggestion: dict[str, Any],
    ctx: dict[str, Any],
) -> dict[str, Any]:
    """Clamp autonomous limit orders into a near-touch band around live bid/ask.

    This keeps autonomous fills actionable for scalp-style execution while
    preserving the model's originally requested structural level for audit.
    """
    if str(suggestion.get("order_type") or "").lower() != "limit":
        suggestion.pop("requested_price", None)
        suggestion["snap_distance_pips"] = 0.0
        return suggestion

    spot = (ctx.get("spot_price") or {}) if isinstance(ctx, dict) else {}
    try:
        bid = float(spot.get("bid") or 0.0)
        ask = float(spot.get("ask") or 0.0)
        requested = float(suggestion.get("price") or 0.0)
    except (TypeError, ValueError):
        return suggestion

    side = str(suggestion.get("side") or "").lower()
    pip_size = 0.01
    min_offset = _AUTONOMOUS_LIMIT_MIN_OFFSET_PIPS * pip_size
    max_offset = _AUTONOMOUS_LIMIT_MAX_OFFSET_PIPS * pip_size
    snapped = requested

    if side == "buy" and bid > 0.0:
        low = bid - max_offset
        high = bid - min_offset
        snapped = min(max(requested, low), high)
    elif side == "sell" and ask > 0.0:
        low = ask + min_offset
        high = ask + max_offset
        snapped = min(max(requested, low), high)

    snapped = round(float(snapped), 3)
    suggestion["requested_price"] = requested
    suggestion["price"] = snapped
    suggestion["snap_distance_pips"] = round(abs(snapped - requested) / pip_size, 2)
    return suggestion


def _fit_aux_memory_blocks(
    blocks: list[tuple[str, str, bool]],
    *,
    budget_words: int = _AUTONOMOUS_AUX_MEMORY_BUDGET_WORDS,
) -> str:
    """Fit overlapping memory/history blocks into a bounded auxiliary budget.

    Keep the higher-signal quantitative / cohort blocks first, then add optional
    narrative blocks only if they fit. This prevents recent-history views from
    silently crowding out current-context reasoning as the databases grow.
    """
    budget_words = max(100, int(budget_words))
    selected: list[str] = []
    used = 0
    optional: list[tuple[str, str]] = []
    omitted: list[str] = []

    for name, text, required in blocks:
        block = str(text or "").strip()
        if not block:
            continue
        words = _word_count(block)
        if required:
            selected.append(block)
            used += words
        else:
            optional.append((name, block))

    remaining = max(0, budget_words - used)
    for name, block in optional:
        words = _word_count(block)
        if words <= remaining:
            selected.append(block)
            remaining -= words
        else:
            omitted.append(name)

    if omitted:
        selected.append(
            "=== PROMPT MEMORY BUDGET NOTE ===\n"
            "Some lower-priority history blocks were omitted to keep the prompt focused: "
            + ", ".join(omitted)
            + "."
        )
    return "\n\n".join(selected)


@dataclass
class GateDecision:
    """One gate evaluation. Logged to the rolling decisions list."""
    timestamp_utc: str
    result: Literal["pass", "block"]
    layer: Literal["hard", "signal", "throttle", "pass"]
    reason: str
    mode: str
    aggressiveness: str
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "t": self.timestamp_utc,
            "r": self.result,
            "l": self.layer,
            "why": self.reason,
            "mode": self.mode,
            "agg": self.aggressiveness,
            **({"x": self.extras} if self.extras else {}),
        }


# -----------------------------------------------------------------------------
# Config load / save (merged with runtime_state.json)
# -----------------------------------------------------------------------------

def _load_state(state_path: Path) -> dict[str, Any]:
    return load_json_state(state_path, default={})


def _save_state(state_path: Path, data: dict[str, Any]) -> None:
    save_json_state(state_path, data, indent=2, trailing_newline=True)


def get_config(state_path: Path) -> dict[str, Any]:
    state = _load_state(state_path)
    cfg = dict(DEFAULT_CONFIG)
    saved = state.get("autonomous_fillmore") or {}
    saved_cfg = saved.get("config") or {}
    for k, v in saved_cfg.items():
        if k in cfg:
            # Shallow-merge trading_hours so partial updates keep siblings.
            if k == "trading_hours" and isinstance(v, dict) and isinstance(cfg[k], dict):
                merged = dict(cfg[k])
                merged.update(v)
                cfg[k] = merged
            else:
                cfg[k] = v
    return _sanitize_autonomous_config(cfg)


def set_config(state_path: Path, patch: dict[str, Any]) -> dict[str, Any]:
    """Merge-update the autonomous config. Returns the new effective config."""
    state = _load_state(state_path)
    auto = state.get("autonomous_fillmore") or {}
    cur = auto.get("config") or {}
    # Validate the patch against known keys (silently drop unknowns).
    valid_keys = set(DEFAULT_CONFIG.keys())
    clean: dict[str, Any] = {}
    for k, v in patch.items():
        if k not in valid_keys:
            continue
        clean[k] = v
    # Type coerce a few fields
    if "aggressiveness" in clean:
        agg = str(clean["aggressiveness"]).lower()
        if agg not in GATE_THRESHOLDS:
            agg = "balanced"
        clean["aggressiveness"] = agg
    if "mode" in clean:
        m = str(clean["mode"]).lower()
        if m not in ("off", "shadow", "paper", "live"):
            m = "off"
        clean["mode"] = m
    clean.pop("order_type", None)  # LLM chooses order type per-suggestion; no config knob
    clean.pop("min_confidence", None)  # replaced by lot-sizing — LLM expresses conviction via lots
    for numkey in (
        "daily_budget_usd", "min_llm_cooldown_sec", "max_lots_per_trade",
        "base_lot_size", "lot_deviation",
        "max_open_ai_trades", "max_daily_loss_usd", "max_consecutive_errors",
        "throttle_no_trade_streak", "throttle_no_trade_cooldown_sec",
        "throttle_loss_streak", "throttle_loss_cooldown_sec",
        "limit_gtd_minutes",
        "correlation_distance_pips", "repeat_setup_window_min", "repeat_setup_bucket_pips",
        "event_blackout_minutes", "max_suggestions_per_call",
        "streak_scratch_threshold_pips", "min_lot_size",
    ):
        if numkey in clean:
            try:
                clean[numkey] = float(clean[numkey]) if "." in str(clean[numkey]) or numkey.endswith("usd") or numkey.endswith("lots_per_trade") else int(clean[numkey])
            except (TypeError, ValueError):
                clean.pop(numkey, None)
    merged = _sanitize_autonomous_config({**cur, **clean})
    auto["config"] = merged
    if not bool(merged.get("enabled")) or str(merged.get("mode") or "off").lower() == "off":
        runtime = _runtime_block(state)
        runtime["throttle_until_utc"] = None
        runtime["throttle_reason"] = None
        _clear_llm_error_alert_state(runtime)
    state["autonomous_fillmore"] = auto
    _save_state(state_path, state)
    return get_config(state_path)


def clear_throttle(state_path: Path) -> dict[str, Any]:
    """Clear active throttle/cooldown state while preserving config and stats."""
    state = _load_state(state_path)
    rt = _runtime_block(state)
    rt["throttle_until_utc"] = None
    rt["throttle_reason"] = None
    rt["consecutive_no_trade_replies"] = 0
    rt["consecutive_errors"] = 0
    _clear_loss_streak_marker(rt)
    state["autonomous_fillmore"]["runtime"] = rt
    _save_state(state_path, state)
    return build_stats(state_path, cfg=get_config(state_path))


def refresh_runtime_from_history(state_path: Path, cfg: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Best-effort refresh of autonomous runtime/performance state from DB history."""
    state = _load_state(state_path)
    rt = _runtime_block(state)
    cfg = cfg or get_config(state_path)
    _rollover_daily_counters(rt)
    try:
        _refresh_autonomous_runtime_from_history(
            state_path=state_path,
            rt=rt,
            cfg=cfg,
        )
    except Exception:
        return rt
    state["autonomous_fillmore"]["runtime"] = rt
    _save_state(state_path, state)
    return rt


# -----------------------------------------------------------------------------
# Decisions log + runtime counters
# -----------------------------------------------------------------------------

_DECISIONS_MAX = 500  # rolling cap


def _runtime_block(state: dict[str, Any]) -> dict[str, Any]:
    auto = state.setdefault("autonomous_fillmore", {})
    runtime = auto.setdefault("runtime", {
        "decisions": [],             # rolling GateDecision dicts
        "last_llm_call_utc": None,
        "llm_calls_today": 0,
        "llm_spend_today_usd": 0.0,
        "spend_day_utc": None,
        "trades_placed_today": 0,
        "consecutive_errors": 0,
        "consecutive_no_trade_replies": 0,
        "consecutive_losses": 0,
        "consecutive_wins": 0,
        "throttle_until_utc": None,
        "throttle_reason": None,
        "loss_streak_last_throttled_day_utc": None,
        "loss_streak_last_throttled_value": 0,
        "last_suggestion_id": None,
        "last_placed_order_id": None,
        "daily_pnl_usd": 0.0,
        "pnl_day_utc": None,
        "previous_regime_label": "normal",
        "previous_streak_regime_label": "normal",
        "regime_entered_trade_id": None,
        "risk_regime_override": None,
        "risk_regime_override_until_utc": None,
        "recent_gate_blocks": {},
        "last_gate_pass_utc": None,
        "last_tick_utc": None,
        "last_stats_recompute_utc": None,
        "last_terminal_event_utc": None,
        "consecutive_llm_errors": 0,
        "consecutive_broker_rejects": 0,
        "active_gate_setups": {},
        "recent_fired_gate_setups": [],
    })
    if "previous_streak_regime_label" not in runtime:
        runtime["previous_streak_regime_label"] = str(runtime.get("previous_regime_label") or "normal")
    if not isinstance(runtime.get("active_gate_setups"), dict):
        runtime["active_gate_setups"] = {}
    if not isinstance(runtime.get("recent_fired_gate_setups"), list):
        runtime["recent_fired_gate_setups"] = []
    return runtime


def log_decision(state_path: Path, decision: GateDecision) -> None:
    state = _load_state(state_path)
    rt = _runtime_block(state)
    lst = list(rt.get("decisions") or [])
    lst.append(decision.to_dict())
    if len(lst) > _DECISIONS_MAX:
        lst = lst[-_DECISIONS_MAX:]
    rt["decisions"] = lst
    if decision.result == "pass":
        rt["last_gate_pass_utc"] = decision.timestamp_utc
    recent_blocks: dict[str, int] = {}
    for item in lst[-200:]:
        if item.get("r") != "block":
            continue
        reason = str(item.get("why") or "?")
        recent_blocks[reason] = recent_blocks.get(reason, 0) + 1
    rt["recent_gate_blocks"] = recent_blocks
    _save_state(state_path, state)


def _today_utc_key() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _parse_iso(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _rollover_daily_counters(rt: dict[str, Any]) -> None:
    today = _today_utc_key()
    if rt.get("spend_day_utc") != today:
        rt["spend_day_utc"] = today
        rt["llm_spend_today_usd"] = 0.0
        rt["llm_calls_today"] = 0
        rt["trades_placed_today"] = 0
    if rt.get("pnl_day_utc") != today:
        rt["pnl_day_utc"] = today
        rt["daily_pnl_usd"] = 0.0
        rt["loss_streak_last_throttled_day_utc"] = None
        rt["loss_streak_last_throttled_value"] = 0


def _clear_loss_streak_marker(rt: dict[str, Any]) -> None:
    rt["loss_streak_last_throttled_day_utc"] = None
    rt["loss_streak_last_throttled_value"] = 0


def _clear_llm_error_alert_state(rt: dict[str, Any]) -> None:
    """Clear persisted LLM error streak state so stale health alerts disappear."""
    rt["consecutive_llm_errors"] = 0
    rt["last_error_msg"] = None
    rt["last_error_utc"] = None


def _maybe_arm_loss_streak_throttle(rt: dict[str, Any], cfg: dict[str, Any], streak: int) -> bool:
    threshold = int(cfg.get("throttle_loss_streak") or 2)
    if streak < threshold:
        _clear_loss_streak_marker(rt)
        return False

    today = _today_utc_key()
    last_day = str(rt.get("loss_streak_last_throttled_day_utc") or "")
    try:
        last_value = int(rt.get("loss_streak_last_throttled_value") or 0)
    except (TypeError, ValueError):
        last_value = 0

    # Only arm once per streak level per day. This prevents an expired cooldown
    # from being re-armed forever from the same historical losses on every tick.
    if last_day == today and streak <= last_value:
        return False

    until = datetime.now(timezone.utc) + timedelta(seconds=int(cfg.get("throttle_loss_cooldown_sec") or 900))
    rt["throttle_until_utc"] = until.isoformat()
    rt["throttle_reason"] = f"loss_streak={streak}"
    rt["loss_streak_last_throttled_day_utc"] = today
    rt["loss_streak_last_throttled_value"] = streak
    return True


def _estimated_call_cost_usd(model: str) -> float:
    inp_per_1m, out_per_1m = _MODEL_COST_PER_1M.get(model, (0.15, 0.60))
    return (inp_per_1m * _ASSUMED_INPUT_TOKENS + out_per_1m * _ASSUMED_OUTPUT_TOKENS) / 1_000_000.0


def record_llm_invocation(state_path: Path, model: str) -> None:
    state = _load_state(state_path)
    rt = _runtime_block(state)
    _rollover_daily_counters(rt)
    rt["last_llm_call_utc"] = datetime.now(timezone.utc).isoformat()
    rt["llm_calls_today"] = int(rt.get("llm_calls_today") or 0) + 1
    rt["llm_spend_today_usd"] = float(rt.get("llm_spend_today_usd") or 0.0) + _estimated_call_cost_usd(model)
    rt["consecutive_llm_errors"] = 0
    _save_state(state_path, state)


def record_trade_placed(state_path: Path, order_id: Any, suggestion_id: Optional[str]) -> None:
    state = _load_state(state_path)
    rt = _runtime_block(state)
    _rollover_daily_counters(rt)
    rt["trades_placed_today"] = int(rt.get("trades_placed_today") or 0) + 1
    rt["last_placed_order_id"] = str(order_id) if order_id is not None else None
    rt["last_suggestion_id"] = suggestion_id
    rt["consecutive_errors"] = 0
    rt["consecutive_no_trade_replies"] = 0
    rt["consecutive_broker_rejects"] = 0
    _save_state(state_path, state)


def _clear_expired_no_trade_throttle(rt: dict[str, Any]) -> None:
    """Reset no-trade streak when its throttle cooldown has expired."""
    reason = str(rt.get("throttle_reason") or "")
    if not reason.startswith("no_trade_streak="):
        return
    throttle_until = rt.get("throttle_until_utc")
    if not throttle_until:
        return
    try:
        tu = datetime.fromisoformat(throttle_until)
        if tu.tzinfo is None:
            tu = tu.replace(tzinfo=timezone.utc)
    except Exception:
        return
    if tu <= datetime.now(timezone.utc):
        rt["consecutive_no_trade_replies"] = 0
        rt["throttle_until_utc"] = None
        rt["throttle_reason"] = None


def record_no_trade_reply(state_path: Path, cfg: dict[str, Any]) -> None:
    state = _load_state(state_path)
    rt = _runtime_block(state)
    rt["consecutive_no_trade_replies"] = int(rt.get("consecutive_no_trade_replies") or 0) + 1
    streak = int(rt["consecutive_no_trade_replies"])
    if streak >= int(cfg.get("throttle_no_trade_streak") or 5):
        until = datetime.now(timezone.utc) + timedelta(seconds=int(cfg.get("throttle_no_trade_cooldown_sec") or 300))
        rt["throttle_until_utc"] = until.isoformat()
        rt["throttle_reason"] = f"no_trade_streak={streak}"
    _save_state(state_path, state)


def record_error(state_path: Path, cfg: dict[str, Any], err_msg: str) -> None:
    state = _load_state(state_path)
    rt = _runtime_block(state)
    rt["consecutive_errors"] = int(rt.get("consecutive_errors") or 0) + 1
    rt["consecutive_llm_errors"] = int(rt.get("consecutive_llm_errors") or 0) + 1
    rt["last_error_msg"] = str(err_msg)[:300]
    rt["last_error_utc"] = datetime.now(timezone.utc).isoformat()
    if rt["consecutive_errors"] >= int(cfg.get("max_consecutive_errors") or 5):
        auto = state.setdefault("autonomous_fillmore", {})
        cur_cfg = auto.setdefault("config", {})
        cur_cfg["mode"] = "off"
        cur_cfg["enabled"] = False
        rt["throttle_reason"] = f"error_kill_switch: {err_msg[:120]}"
    _save_state(state_path, state)


def record_broker_reject(state_path: Path) -> None:
    state = _load_state(state_path)
    rt = _runtime_block(state)
    rt["consecutive_broker_rejects"] = int(rt.get("consecutive_broker_rejects") or 0) + 1
    _save_state(state_path, state)


def record_trade_outcome(state_path: Path, pnl_usd: float, cfg: dict[str, Any]) -> None:
    """Called from trade-close path when an autonomous Fillmore trade closes."""
    state = _load_state(state_path)
    rt = _runtime_block(state)
    _rollover_daily_counters(rt)
    rt["daily_pnl_usd"] = float(rt.get("daily_pnl_usd") or 0.0) + float(pnl_usd or 0.0)
    if pnl_usd is not None and float(pnl_usd) < 0:
        rt["consecutive_losses"] = int(rt.get("consecutive_losses") or 0) + 1
        rt["consecutive_wins"] = 0
    elif pnl_usd is not None and float(pnl_usd) > 0:
        rt["consecutive_losses"] = 0
        rt["consecutive_wins"] = int(rt.get("consecutive_wins") or 0) + 1
    _save_state(state_path, state)


def _regime_rank(label: str) -> int:
    return {"normal": 0, "defensive_soft": 1, "defensive_hard": 2}.get(str(label or "normal"), 0)


# -----------------------------------------------------------------------------
# Gate logic
# -----------------------------------------------------------------------------

def _session_flag_now(trading_hours: dict[str, Any]) -> tuple[bool, str]:
    """Return (in_allowed_session, label). Uses UTC hour bands.

    Tokyo: 23-09 UTC (wraps midnight).  London: 07-16 UTC.  NY: 12-21 UTC.
    If *any* enabled session overlaps the current hour, we allow.
    """
    now_h = datetime.now(timezone.utc).hour
    in_tokyo = now_h >= 23 or now_h < 9
    in_london = 7 <= now_h < 16
    in_ny = 12 <= now_h < 21
    allowed = False
    labels: list[str] = []
    if in_tokyo:
        labels.append("tokyo")
        if trading_hours.get("tokyo", True):
            allowed = True
    if in_london:
        labels.append("london")
        if trading_hours.get("london", True):
            allowed = True
    if in_ny:
        labels.append("ny")
        if trading_hours.get("ny", True):
            allowed = True
    return allowed, "/".join(labels) if labels else "off-hours"


def _session_spread_limit_pips(session_label: str) -> float:
    label = str(session_label or "off-hours").strip().lower()
    if "london" in label and "ny" in label:
        return float(_USDJPY_SPREAD_LIMITS_PIPS["london/ny"])
    if "london" in label:
        return float(_USDJPY_SPREAD_LIMITS_PIPS["london"])
    if "ny" in label:
        return float(_USDJPY_SPREAD_LIMITS_PIPS["ny"])
    if "tokyo" in label:
        return float(_USDJPY_SPREAD_LIMITS_PIPS["tokyo"])
    return float(_USDJPY_SPREAD_LIMITS_PIPS["off-hours"])


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _atr_pips(data_by_tf: dict[str, pd.DataFrame], timeframe: str, period: int = 14, pip_size: float = 0.01) -> Optional[float]:
    df = data_by_tf.get(timeframe) if data_by_tf else None
    if df is None or len(df) < max(period + 1, 20):
        return None
    try:
        highs = df["high"].astype(float)
        lows = df["low"].astype(float)
        closes = df["close"].astype(float)
        prev_close = closes.shift(1)
        tr = pd.concat([
            highs - lows,
            (highs - prev_close).abs(),
            (lows - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = float(tr.rolling(period).mean().iloc[-1])
        if not math.isfinite(atr):
            return None
        pip = pip_size or 0.01
        if pip <= 0:
            pip = 0.01
        return atr / pip
    except Exception:
        return None


def _sufficient_volatility(
    data_by_tf: dict[str, pd.DataFrame],
    *,
    min_atr_pips: float,
    timeframe: str = "M5",
    pip_size: float = 0.01,
) -> bool:
    atr_pips = _atr_pips(data_by_tf, timeframe, period=14, pip_size=pip_size)
    if atr_pips is None:
        return True
    return atr_pips >= float(min_atr_pips)


def _m3_trend(data_by_tf: dict[str, pd.DataFrame]) -> Optional[str]:
    """Return 'bull'|'bear'|None based on M3 EMA stack (close > EMA9 > EMA21)."""
    try:
        df = data_by_tf.get("M3") if data_by_tf else None
        if df is None or len(df) < 25:
            return None
        close = df["close"].astype(float)
        e9 = float(_ema(close, 9).iloc[-1])
        e21 = float(_ema(close, 21).iloc[-1])
        c = float(close.iloc[-1])
        if not all(math.isfinite(v) for v in (c, e9, e21)):
            return None
        if c > e9 > e21:
            return "bull"
        if c < e9 < e21:
            return "bear"
        return None
    except Exception:
        return None


def _m1_stack(data_by_tf: dict[str, pd.DataFrame]) -> Optional[str]:
    """M1 EMA5/9/21 stack alignment."""
    try:
        df = data_by_tf.get("M1") if data_by_tf else None
        if df is None or len(df) < 25:
            return None
        close = df["close"].astype(float)
        e5 = float(_ema(close, 5).iloc[-1])
        e9 = float(_ema(close, 9).iloc[-1])
        e21 = float(_ema(close, 21).iloc[-1])
        if not all(math.isfinite(v) for v in (e5, e9, e21)):
            return None
        if e5 > e9 > e21:
            return "bull"
        if e5 < e9 < e21:
            return "bear"
        return None
    except Exception:
        return None


def _m5_stack(data_by_tf: dict[str, pd.DataFrame]) -> Optional[str]:
    """M5 EMA9/21 directional stack."""
    try:
        df = data_by_tf.get("M5") if data_by_tf else None
        if df is None or len(df) < 25:
            return None
        close = df["close"].astype(float)
        e9 = float(_ema(close, 9).iloc[-1])
        e21 = float(_ema(close, 21).iloc[-1])
        c = float(close.iloc[-1])
        if not all(math.isfinite(v) for v in (e9, e21, c)):
            return None
        if c > e9 > e21:
            return "bull"
        if c < e9 < e21:
            return "bear"
        return None
    except Exception:
        return None


def _m1_pullback_or_zone(
    data_by_tf: dict[str, pd.DataFrame],
    trend: Optional[str],
    *,
    zone_min_pips: float = 1.5,
    lookback_bars: int = 3,
) -> bool:
    """True if price is at/near EMA9 (zone) or touched EMA13-17 (pullback) aligned with trend."""
    try:
        if trend is None:
            return False
        df = data_by_tf.get("M1") if data_by_tf else None
        if df is None or len(df) < 30:
            return False
        close = df["close"].astype(float)
        last = float(close.iloc[-1])
        e9 = float(_ema(close, 9).iloc[-1])
        e13 = float(_ema(close, 13).iloc[-1])
        e17 = float(_ema(close, 17).iloc[-1])
        if not all(math.isfinite(v) for v in (last, e9, e13, e17)):
            return False
        m1_atr_pips = _atr_pips(data_by_tf, "M1", period=14, pip_size=0.01)
        if m1_atr_pips is not None:
            zone_width_pips = max(float(zone_min_pips), min(4.0, m1_atr_pips * 0.5))
        else:
            zone_width_pips = max(float(zone_min_pips), 2.5)
        zone_width = zone_width_pips * 0.01

        # Zone: dynamic EMA9 proximity scaled to recent M1 volatility.
        if abs(last - e9) < zone_width:
            return True

        lows = df["low"].astype(float)
        highs = df["high"].astype(float)
        lookback = max(1, min(int(lookback_bars or 1), len(df)))
        for i in range(-lookback, 0):
            bar_low = float(lows.iloc[i])
            bar_high = float(highs.iloc[i])
            if not all(math.isfinite(v) for v in (bar_low, bar_high)):
                continue
            if trend == "bull" and (bar_low <= e13 or bar_low <= e17):
                return True
            if trend == "bear" and (bar_high >= e13 or bar_high >= e17):
                return True
        return False
    except Exception:
        return False


def _adx_value(
    data_by_tf: dict[str, pd.DataFrame],
    timeframe: str,
    period: int = 14,
) -> Optional[float]:
    """Compute a simple ADX value from OHLC bars."""
    df = data_by_tf.get(timeframe) if data_by_tf else None
    if df is None or len(df) < max(period * 2, 40):
        return None
    try:
        highs = df["high"].astype(float)
        lows = df["low"].astype(float)
        closes = df["close"].astype(float)
        up_move = highs.diff()
        down_move = -lows.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        prev_close = closes.shift(1)
        tr = pd.concat([
            highs - lows,
            (highs - prev_close).abs(),
            (lows - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        plus_di = 100.0 * (pd.Series(plus_dm, index=df.index).rolling(period).mean() / atr)
        minus_di = 100.0 * (pd.Series(minus_dm, index=df.index).rolling(period).mean() / atr)
        dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
        adx = float(dx.rolling(period).mean().iloc[-1])
        if not math.isfinite(adx):
            return None
        return adx
    except Exception:
        return None


def _classify_session_label_from_hour(hour_utc: int) -> str:
    in_tokyo = 0 <= hour_utc < 9
    in_london = 7 <= hour_utc < 16
    in_ny = 12 <= hour_utc < 21
    labels: list[str] = []
    if in_tokyo:
        labels.append("tokyo")
    if in_london:
        labels.append("london")
    if in_ny:
        labels.append("ny")
    return "/".join(labels) if labels else "off-hours"


def _extract_time_index(df: pd.DataFrame) -> Optional[pd.Series]:
    if "time" in df.columns:
        ts = pd.to_datetime(df["time"], utc=True, errors="coerce")
        return ts
    if isinstance(df.index, pd.DatetimeIndex):
        return pd.Series(df.index.tz_convert("UTC") if df.index.tz else df.index.tz_localize("UTC"))
    return None


def _current_session_range_levels(
    data_by_tf: dict[str, pd.DataFrame],
    session_label: str,
) -> list[tuple[str, float]]:
    df = data_by_tf.get("M1") if data_by_tf else None
    if df is None or len(df) < 30:
        return []
    ts = _extract_time_index(df)
    if ts is None:
        return []
    try:
        labels = ts.dt.hour.map(_classify_session_label_from_hour)
        if session_label == "london/ny":
            mask = labels == "london/ny"
        else:
            mask = labels.str.contains(session_label, na=False)
        recent_mask = ts >= (ts.iloc[-1] - pd.Timedelta(hours=12))
        mask = mask & recent_mask
        if not mask.any():
            return []
        highs = df.loc[mask, "high"].astype(float)
        lows = df.loc[mask, "low"].astype(float)
        out: list[tuple[str, float]] = []
        if len(highs):
            out.append((f"{session_label.upper()}_SESSION_HIGH", float(highs.max())))
        if len(lows):
            out.append((f"{session_label.upper()}_SESSION_LOW", float(lows.min())))
        return out
    except Exception:
        return []


def _local_range_levels(data_by_tf: dict[str, pd.DataFrame]) -> list[tuple[str, float]]:
    df = data_by_tf.get("M15") if data_by_tf else None
    if df is None or len(df) < 24:
        return []
    try:
        tail = df.tail(24)
        return [
            ("LOCAL_RANGE_HIGH", float(tail["high"].astype(float).max())),
            ("LOCAL_RANGE_LOW", float(tail["low"].astype(float).min())),
        ]
    except Exception:
        return []


def _order_book_cluster_levels(order_book: Optional[dict[str, Any]]) -> list[tuple[str, float]]:
    """Convert OANDA order-book cluster summaries into advisory structure levels."""
    if not isinstance(order_book, dict):
        return []
    levels: list[tuple[str, float]] = []

    def _append_clusters(key: str, label_prefix: str) -> None:
        clusters = order_book.get(key)
        if not isinstance(clusters, list):
            return
        for cluster in clusters[:5]:
            if not isinstance(cluster, dict):
                continue
            try:
                price = float(cluster.get("price"))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(price):
                continue
            pct = cluster.get("pct")
            try:
                pct_val = float(pct)
            except (TypeError, ValueError):
                pct_val = 0.0
            levels.append((f"{label_prefix}:{price:.3f}:{pct_val:.3f}", price))

    _append_clusters("buy_clusters", "OANDA_BUY_CLUSTER")
    _append_clusters("sell_clusters", "OANDA_SELL_CLUSTER")
    return levels


def _structure_levels(
    tick_mid: float,
    data_by_tf: dict[str, pd.DataFrame],
    session_label: str,
    order_book: Optional[dict[str, Any]] = None,
) -> list[tuple[str, float]]:
    levels: list[tuple[str, float]] = []

    d_df = data_by_tf.get("D") if data_by_tf else None
    if d_df is not None and len(d_df) >= 1:
        try:
            today = d_df.iloc[-1]
            levels.extend([
                ("TODAY_HIGH", float(today["high"])),
                ("TODAY_LOW", float(today["low"])),
            ])
        except Exception:
            pass
    if d_df is not None and len(d_df) >= 2:
        try:
            prev = d_df.iloc[-2]
            levels.extend([
                ("PDH", float(prev["high"])),
                ("PDL", float(prev["low"])),
            ])
        except Exception:
            pass

    w_df = data_by_tf.get("W") if data_by_tf else None
    if w_df is not None and len(w_df) >= 2:
        try:
            prev_w = w_df.iloc[-2]
            levels.extend([
                ("PWH", float(prev_w["high"])),
                ("PWL", float(prev_w["low"])),
            ])
        except Exception:
            pass

    levels.extend(_current_session_range_levels(data_by_tf, session_label))
    levels.extend(_local_range_levels(data_by_tf))
    levels.extend(_order_book_cluster_levels(order_book))

    try:
        base = int(tick_mid * 2) / 2.0
        for step in range(-3, 4):
            lvl = round(base + 0.5 * step, 2)
            whole = abs(lvl - round(lvl)) < 1e-9
            label = "WHOLE_YEN" if whole else "HALF_YEN"
            levels.append((f"{label}:{lvl:.2f}", lvl))
    except Exception:
        pass

    deduped: dict[float, str] = {}
    for label, price in levels:
        if not math.isfinite(price):
            continue
        key = round(float(price), 3)
        deduped.setdefault(key, label)
    return [(label, price) for price, label in deduped.items()]


def _nearest_structure_pips(
    tick_mid: float,
    data_by_tf: dict[str, pd.DataFrame],
    session_label: str,
    pip_size: float = 0.01,
    order_book: Optional[dict[str, Any]] = None,
) -> dict[str, Optional[float]]:
    """Distance in pips to nearest key structure level above and below current price."""
    pip = pip_size or 0.01
    levels_above: list[tuple[str, float]] = []
    levels_below: list[tuple[str, float]] = []
    for label, price in _structure_levels(tick_mid, data_by_tf, session_label, order_book=order_book):
        if price > tick_mid:
            levels_above.append((label, price))
        elif price < tick_mid:
            levels_below.append((label, price))

    def _closest(levels: list[tuple[str, float]]) -> tuple[Optional[float], Optional[str], Optional[float]]:
        if not levels:
            return None, None, None
        label, price = min(levels, key=lambda item: abs(item[1] - tick_mid))
        return abs(price - tick_mid) / pip, label, price

    up, up_label, up_price = _closest(levels_above)
    dn, dn_label, dn_price = _closest(levels_below)
    candidates = [v for v in (up, dn) if v is not None]
    return {
        "overhead_pips": up,
        "underfoot_pips": dn,
        "nearest_pips": min(candidates) if candidates else None,
        "overhead_label": up_label,
        "underfoot_label": dn_label,
        "overhead_price": up_price,
        "underfoot_price": dn_price,
    }


def _critical_level_reaction_trigger(
    tick_mid: float,
    data_by_tf: dict[str, pd.DataFrame],
    session_label: str,
    *,
    max_level_pips: float,
    micro_window_bars: int,
    touch_tolerance_pips: float,
    pip_size: float = 0.01,
    order_book: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    m1 = data_by_tf.get("M1") if data_by_tf else None
    if m1 is None or len(m1) < max(10, micro_window_bars + 2):
        return None
    prox = _nearest_structure_pips(tick_mid, data_by_tf, session_label, pip_size=pip_size, order_book=order_book)
    pip = pip_size or 0.01
    tol = max(0.1, float(touch_tolerance_pips)) * pip
    lookback = max(2, int(micro_window_bars))
    recent = m1.tail(lookback)
    closes = recent["close"].astype(float)
    highs = recent["high"].astype(float)
    lows = recent["low"].astype(float)
    if len(closes) < 2:
        return None
    last_close = float(closes.iloc[-1])
    prev_close = float(closes.iloc[-2])

    def _former_low_has_acceptance_below(label: Any, price: float) -> bool:
        label_s = str(label or "").upper()
        if "LOW" not in label_s:
            return True
        prior_closes = closes.iloc[:-1]
        if len(prior_closes) < 3:
            return False
        acceptance_buffer = max(0.2 * pip, tol * 0.25)
        below = prior_closes < (float(price) - acceptance_buffer)
        return bool((below & below.shift(1, fill_value=False)).any())

    underfoot_pips = prox.get("underfoot_pips")
    underfoot_price = prox.get("underfoot_price")
    underfoot_label = prox.get("underfoot_label")
    if isinstance(underfoot_pips, (int, float)) and underfoot_pips <= float(max_level_pips) and isinstance(underfoot_price, (int, float)):
        touched_support = float(lows.min()) <= float(underfoot_price) + tol
        reclaimed = last_close > float(underfoot_price) and last_close >= prev_close
        if touched_support and reclaimed and _support_reclaim_level_allowed(underfoot_label):
            return {
                "family": "critical_level_reaction",
                "reason": f"support_reclaim:{underfoot_label}",
                "bias": "buy",
                "level_label": underfoot_label,
                "level_price": float(underfoot_price),
                "nearest_level_pips": round(float(underfoot_pips), 2),
                "micro_confirmation": "reclaimed_support",
                "trigger_score": round(62.0 + max(0.0, float(max_level_pips) - float(underfoot_pips)), 2),
            }

    # Resistance rejects are forward-tested with LLM discretion enabled. The
    # mechanical backtest is only a raw wakeup-quality check, not a substitute
    # for the model's trade/skip reasoning.
    if not _ENABLE_CRITICAL_RESISTANCE_REJECT:
        return None

    overhead_pips = prox.get("overhead_pips")
    overhead_price = prox.get("overhead_price")
    overhead_label = prox.get("overhead_label")
    if isinstance(overhead_pips, (int, float)) and overhead_pips <= float(max_level_pips) and isinstance(overhead_price, (int, float)):
        touched_resistance = float(highs.max()) >= float(overhead_price) - tol
        rejected = last_close < float(overhead_price) and last_close <= prev_close
        accepted_below_former_low = _former_low_has_acceptance_below(overhead_label, float(overhead_price))
        if touched_resistance and rejected and accepted_below_former_low:
            return {
                "family": "critical_level_reaction",
                "reason": f"resistance_reject:{overhead_label}",
                "bias": "sell",
                "level_label": overhead_label,
                "level_price": float(overhead_price),
                "nearest_level_pips": round(float(overhead_pips), 2),
                "micro_confirmation": "rejected_resistance",
                "trigger_score": round(62.0 + max(0.0, float(max_level_pips) - float(overhead_pips)), 2),
            }

    return None


def _tokyo_tight_range_mean_reversion_trigger(
    tick_mid: float,
    data_by_tf: dict[str, pd.DataFrame],
    session_label: str,
    *,
    range_window_bars: int,
    min_session_bars: int,
    max_range_pips: float,
    range_atr_mult: float,
    edge_fraction: float,
    touch_tolerance_pips: float,
    adx_ceiling: float,
    bb_width_max: float,
    min_reward_pips: float,
    pip_size: float = 0.01,
) -> Optional[dict[str, Any]]:
    if session_label != "tokyo":
        return None
    m1_df = data_by_tf.get("M1") if data_by_tf else None
    m5_df = data_by_tf.get("M5") if data_by_tf else None
    if m1_df is None or m5_df is None or len(m1_df) < max(40, int(range_window_bars) + 5) or len(m5_df) < 25:
        return None

    ts = _extract_time_index(m1_df)
    if ts is None or len(ts) != len(m1_df):
        return None
    session_mask = ts.dt.hour.map(_classify_session_label_from_hour) == "tokyo"
    if int(session_mask.sum()) < int(min_session_bars):
        return None
    session_df = m1_df.loc[session_mask].copy()
    if len(session_df) < int(min_session_bars):
        return None

    recent = session_df.tail(max(6, int(range_window_bars))).copy()
    highs = recent["high"].astype(float)
    lows = recent["low"].astype(float)
    closes = recent["close"].astype(float)
    if len(closes) < 6:
        return None

    session_high = float(highs.max())
    session_low = float(lows.min())
    session_mid = (session_high + session_low) / 2.0
    pip = pip_size or 0.01
    range_pips = (session_high - session_low) / pip
    if not math.isfinite(range_pips) or range_pips <= 0:
        return None

    m5_atr_pips = _atr_pips(data_by_tf, "M5", period=14, pip_size=pip)
    if m5_atr_pips is None or not math.isfinite(m5_atr_pips) or m5_atr_pips <= 0:
        return None
    range_cap_pips = min(float(max_range_pips), float(m5_atr_pips) * float(range_atr_mult))
    if range_pips > range_cap_pips:
        return None

    adx_m5 = _adx_value(data_by_tf, "M5", period=14)
    if adx_m5 is None or not math.isfinite(adx_m5) or adx_m5 > float(adx_ceiling):
        return None

    bb_upper, bb_mid, bb_lower = bollinger_bands(closes, period=min(20, len(closes)), std_dev=2.0)
    try:
        bb_middle = float(bb_mid.iloc[-1])
        bb_width = float((bb_upper.iloc[-1] - bb_lower.iloc[-1]) / max(bb_middle, pip))
    except Exception:
        return None
    if not math.isfinite(bb_width) or bb_width > float(bb_width_max):
        return None

    last_close = float(closes.iloc[-1])
    prev_close = float(closes.iloc[-2])
    last_high = float(highs.iloc[-1])
    last_low = float(lows.iloc[-1])
    close_pos = (last_close - session_low) / max(session_high - session_low, pip)
    tol = max(0.1, float(touch_tolerance_pips)) * pip
    edge_fraction = min(max(float(edge_fraction), 0.1), 0.4)

    buy_touch = float(lows.tail(3).min()) <= session_low + tol
    buy_reclaim = last_close > prev_close and last_close > session_low and close_pos <= edge_fraction + 0.12
    buy_reward_pips = (session_mid - last_close) / pip

    sell_touch = float(highs.tail(3).max()) >= session_high - tol
    sell_reject = last_close < prev_close and last_close < session_high and close_pos >= 1.0 - edge_fraction - 0.12
    sell_reward_pips = (last_close - session_mid) / pip

    if buy_touch and buy_reclaim and buy_reward_pips >= float(min_reward_pips):
        return {
            "family": "tight_range_mean_reversion",
            "reason": "tokyo_range_reclaim:session_low",
            "bias": "buy",
            "level_label": "TOKYO_SESSION_LOW",
            "level_price": session_low,
            "nearest_level_pips": round((last_close - session_low) / pip, 2),
            "micro_confirmation": "tokyo_range_low_reclaim",
            "session_range_pips": round(range_pips, 2),
            "session_mid_price": round(session_mid, 3),
            "reward_to_mid_pips": round(buy_reward_pips, 2),
            "bb_width": round(bb_width, 6),
            "adx": round(float(adx_m5), 2),
        }
    if sell_touch and sell_reject and sell_reward_pips >= float(min_reward_pips):
        return {
            "family": "tight_range_mean_reversion",
            "reason": "tokyo_range_reject:session_high",
            "bias": "sell",
            "level_label": "TOKYO_SESSION_HIGH",
            "level_price": session_high,
            "nearest_level_pips": round((session_high - last_close) / pip, 2),
            "micro_confirmation": "tokyo_range_high_reject",
            "session_range_pips": round(range_pips, 2),
            "session_mid_price": round(session_mid, 3),
            "reward_to_mid_pips": round(sell_reward_pips, 2),
            "bb_width": round(bb_width, 6),
            "adx": round(float(adx_m5), 2),
        }
    return None


def _compression_breakout_trigger(
    tick_mid: float,
    data_by_tf: dict[str, pd.DataFrame],
    session_label: str,
    *,
    max_level_pips: float,
    compression_window_bars: int,
    compression_range_atr_mult: float,
    edge_fraction: float,
    adx_floor: float,
    adx_ceiling: float,
    require_m3_trend: bool,
    require_m1_stack: bool,
    require_any_trend_signal: bool,
    reject_mismatch: bool,
    pip_size: float = 0.01,
) -> Optional[dict[str, Any]]:
    if not _ENABLE_COMPRESSION_BREAKOUT:
        return None

    m1_df = data_by_tf.get("M1") if data_by_tf else None
    m5_df = data_by_tf.get("M5") if data_by_tf else None
    if m1_df is None or m5_df is None or len(m1_df) < max(12, compression_window_bars + 2) or len(m5_df) < 25:
        return None

    m3 = _m3_trend(data_by_tf)
    m1 = _m1_stack(data_by_tf)
    m5 = _m5_stack(data_by_tf)
    if require_m3_trend and m3 is None:
        return None
    if require_m1_stack and m1 is None:
        return None
    if require_any_trend_signal and m3 is None and m1 is None:
        return None
    if reject_mismatch and m3 is not None and m1 is not None and m3 != m1:
        return None

    directional = [x for x in (m3, m1, m5) if x in ("bull", "bear")]
    if not directional:
        return None
    if directional.count("bull") >= 2:
        bias = "buy"
    elif directional.count("bear") >= 2:
        bias = "sell"
    elif m3 == "bull" or m1 == "bull":
        bias = "buy"
    elif m3 == "bear" or m1 == "bear":
        bias = "sell"
    else:
        return None

    m5_atr_pips = _atr_pips(data_by_tf, "M5", period=14, pip_size=pip_size)
    if m5_atr_pips is None or not math.isfinite(m5_atr_pips) or m5_atr_pips <= 0:
        return None
    adx_m5 = _adx_value(data_by_tf, "M5", period=14)
    adx_m15 = _adx_value(data_by_tf, "M15", period=14)
    adx_candidates = [v for v in (adx_m5, adx_m15) if isinstance(v, (int, float)) and math.isfinite(v)]
    adx = max(adx_candidates) if adx_candidates else None
    if adx is None or adx < float(adx_floor) or adx > float(adx_ceiling):
        return None

    prox = _nearest_structure_pips(tick_mid, data_by_tf, session_label, pip_size=pip_size)
    recent = m1_df.tail(max(4, int(compression_window_bars)))
    highs = recent["high"].astype(float)
    lows = recent["low"].astype(float)
    closes = recent["close"].astype(float)
    if len(closes) < 4:
        return None
    box_high = float(highs.max())
    box_low = float(lows.min())
    box_range_pips = (box_high - box_low) / (pip_size or 0.01)
    compression_cap_pips = max(1.5, float(m5_atr_pips) * float(compression_range_atr_mult))
    if not math.isfinite(box_range_pips) or box_range_pips <= 0 or box_range_pips > compression_cap_pips:
        return None

    last_close = float(closes.iloc[-1])
    prev_close = float(closes.iloc[-2])
    edge_fraction = min(max(float(edge_fraction), 0.05), 0.4)
    box_span = max(box_high - box_low, pip_size or 0.01)
    close_pos = (last_close - box_low) / box_span

    if bias == "buy":
        level_pips = prox.get("overhead_pips")
        level_price = prox.get("overhead_price")
        level_label = prox.get("overhead_label")
        edge_ok = close_pos >= 1.0 - edge_fraction
        pressure_ok = last_close >= prev_close and float(highs.tail(3).max()) >= box_high - ((pip_size or 0.01) * 0.2)
    else:
        level_pips = prox.get("underfoot_pips")
        level_price = prox.get("underfoot_price")
        level_label = prox.get("underfoot_label")
        edge_ok = close_pos <= edge_fraction
        pressure_ok = last_close <= prev_close and float(lows.tail(3).min()) <= box_low + ((pip_size or 0.01) * 0.2)

    if not isinstance(level_pips, (int, float)) or not isinstance(level_price, (int, float)) or level_pips > float(max_level_pips):
        return None
    if not edge_ok or not pressure_ok:
        return None

    return {
        "family": "compression_breakout",
        "reason": f"compression_press:{level_label}",
        "bias": bias,
        "level_label": level_label,
        "level_price": float(level_price),
        "nearest_level_pips": round(float(level_pips), 2),
        "micro_confirmation": "compressed_range_pressing_boundary",
        "compression_range_pips": round(float(box_range_pips), 2),
        "compression_cap_pips": round(float(compression_cap_pips), 2),
        "adx": round(float(adx), 2),
        "m5_atr_pips": round(float(m5_atr_pips), 2),
    }


def _failed_breakout_reversal_trigger(
    tick_mid: float,
    data_by_tf: dict[str, pd.DataFrame],
    session_label: str,
    *,
    min_break_pips: float = _FAILED_BREAKOUT_MIN_BREAK_PIPS,
    max_break_pips: float = _FAILED_BREAKOUT_MAX_BREAK_PIPS,
    max_hold_bars: int = _FAILED_BREAKOUT_MAX_HOLD_BARS,
    min_session_bars: int = _FAILED_BREAKOUT_MIN_SESSION_BARS,
    recapture_body_ratio: float = _FAILED_BREAKOUT_RECAPTURE_BODY_RATIO,
    continuation_invalidation_pips: float = _FAILED_BREAKOUT_CONTINUATION_INVALIDATION_PIPS,
    pip_size: float = 0.01,
) -> Optional[dict[str, Any]]:
    if session_label not in _FAILED_BREAKOUT_ALLOWED_SESSIONS:
        return None
    m5_df = data_by_tf.get("M5") if data_by_tf else None
    if m5_df is None or len(m5_df) < max(20, int(min_session_bars) + int(max_hold_bars) + 3):
        return None
    ts = _extract_time_index(m5_df)
    if ts is None or len(ts) != len(m5_df):
        return None
    try:
        work = m5_df.copy()
        work["time"] = pd.to_datetime(ts, utc=True, errors="coerce")
        work = work.dropna(subset=["time"]).copy()
        if work.empty:
            return None
        work["session_label"] = work["time"].dt.hour.map(_classify_session_label_from_hour)
        session_df = work.loc[work["session_label"] == session_label].copy()
        if len(session_df) < int(min_session_bars):
            return None
        session_day = session_df["time"].iloc[-1].floor("D")
        session_df = session_df.loc[session_df["time"].dt.floor("D") == session_day].copy()
        if len(session_df) < int(min_session_bars):
            return None

        session_df["session_bar_index"] = range(1, len(session_df) + 1)
        session_df["prior_session_high"] = session_df["high"].astype(float).cummax().shift(1)
        session_df["prior_session_low"] = session_df["low"].astype(float).cummin().shift(1)
        session_df["bar_body_ratio"] = session_df.apply(
            lambda row: abs(float(row["close"]) - float(row["open"])) / max(float(row["high"]) - float(row["low"]), 1e-9),
            axis=1,
        )

        invalidation_px = float(continuation_invalidation_pips) * (pip_size or 0.01)
        active: list[dict[str, Any]] = []
        for row in session_df.itertuples(index=False):
            next_active: list[dict[str, Any]] = []
            for candidate in active:
                age = int(row.session_bar_index) - int(candidate["session_bar_index"])
                if age < 1:
                    next_active.append(candidate)
                    continue
                if age > int(max_hold_bars):
                    continue
                level = float(candidate["reference_level"])
                if candidate["breakout_side"] == "up":
                    if float(row.high) >= level + invalidation_px:
                        continue
                    if float(row.close) < level and float(row.bar_body_ratio) >= float(recapture_body_ratio):
                        return {
                            "family": "failed_breakout_reversal_overlap_v1",
                            "reason": f"failed_breakout_recapture:{candidate['level_label']}",
                            "bias": "sell",
                            "level_label": candidate["level_label"],
                            "level_price": round(level, 3),
                            "nearest_level_pips": round(abs(float(tick_mid) - level) / (pip_size or 0.01), 2),
                            "micro_confirmation": "failed_breakout_recapture",
                            "breakout_side": "up",
                            "breakout_excursion_pips": round(float(candidate["breakout_excursion_pips"]), 2),
                            "hold_bars": age,
                        }
                else:
                    if float(row.low) <= level - invalidation_px:
                        continue
                    if float(row.close) > level and float(row.bar_body_ratio) >= float(recapture_body_ratio):
                        return {
                            "family": "failed_breakout_reversal_overlap_v1",
                            "reason": f"failed_breakout_recapture:{candidate['level_label']}",
                            "bias": "buy",
                            "level_label": candidate["level_label"],
                            "level_price": round(level, 3),
                            "nearest_level_pips": round(abs(float(tick_mid) - level) / (pip_size or 0.01), 2),
                            "micro_confirmation": "failed_breakout_recapture",
                            "breakout_side": "down",
                            "breakout_excursion_pips": round(float(candidate["breakout_excursion_pips"]), 2),
                            "hold_bars": age,
                        }
                next_active.append(candidate)
            active = next_active

            if int(row.session_bar_index) < int(min_session_bars):
                continue
            prior_high = float(row.prior_session_high) if pd.notna(row.prior_session_high) else math.nan
            prior_low = float(row.prior_session_low) if pd.notna(row.prior_session_low) else math.nan
            if math.isfinite(prior_high):
                excursion_up = (float(row.high) - prior_high) / (pip_size or 0.01)
                if float(min_break_pips) <= excursion_up <= float(max_break_pips) and float(row.close) > prior_high:
                    active.append(
                        {
                            "session_bar_index": int(row.session_bar_index),
                            "breakout_side": "up",
                            "reference_level": prior_high,
                            "breakout_excursion_pips": excursion_up,
                            "level_label": "LONDON_NY_SESSION_HIGH",
                        }
                    )
            if math.isfinite(prior_low):
                excursion_down = (prior_low - float(row.low)) / (pip_size or 0.01)
                if float(min_break_pips) <= excursion_down <= float(max_break_pips) and float(row.close) < prior_low:
                    active.append(
                        {
                            "session_bar_index": int(row.session_bar_index),
                            "breakout_side": "down",
                            "reference_level": prior_low,
                            "breakout_excursion_pips": excursion_down,
                            "level_label": "LONDON_NY_SESSION_LOW",
                        }
                    )
    except Exception:
        return None
    return None


def _post_spike_retracement_trigger(
    tick_mid: float,
    data_by_tf: dict[str, pd.DataFrame],
    session_label: str,
    *,
    spike_window_bars: int = _POST_SPIKE_WINDOW_BARS,
    min_spike_pips: float = _POST_SPIKE_MIN_MOVE_PIPS,
    min_directional_consistency: float = _POST_SPIKE_MIN_DIRECTIONAL_CONSISTENCY,
    max_confirmation_bars: int = _POST_SPIKE_MAX_CONFIRMATION_BARS,
    allowed_confirmation_bars: set[int] = _POST_SPIKE_ALLOWED_CONFIRMATION_BARS,
    stall_body_fraction: float = _POST_SPIKE_STALL_BODY_FRACTION,
    max_extension_after_spike_pips: float = _POST_SPIKE_MAX_EXTENSION_AFTER_SPIKE_PIPS,
    pip_size: float = 0.01,
) -> Optional[dict[str, Any]]:
    if session_label not in _POST_SPIKE_ALLOWED_SESSIONS:
        return None
    m1_df = data_by_tf.get("M1") if data_by_tf else None
    min_rows = int(spike_window_bars) + int(max_confirmation_bars) + 2
    if m1_df is None or len(m1_df) < max(20, min_rows):
        return None
    ts = _extract_time_index(m1_df)
    if ts is None or len(ts) != len(m1_df):
        return None
    try:
        work = m1_df.copy()
        work["time"] = pd.to_datetime(ts, utc=True, errors="coerce")
        work = work.dropna(subset=["time"]).copy().reset_index(drop=True)
        if len(work) < min_rows:
            return None

        work["session_label"] = work["time"].dt.hour.map(_classify_session_label_from_hour)
        if str(work["session_label"].iloc[-1]) != session_label:
            return None

        closes = work["close"].astype(float).to_numpy()
        opens = work["open"].astype(float).to_numpy()
        highs = work["high"].astype(float).to_numpy()
        lows = work["low"].astype(float).to_numpy()
        signal_idx = len(work) - 1

        for confirmation_bars in sorted(set(int(x) for x in allowed_confirmation_bars if int(x) > 0)):
            if confirmation_bars > int(max_confirmation_bars) or signal_idx - confirmation_bars < int(spike_window_bars):
                continue
            spike_end_idx = signal_idx - confirmation_bars
            spike_start_idx = spike_end_idx - int(spike_window_bars)
            if spike_start_idx < 0:
                continue

            net_move_pips = (closes[spike_end_idx] - closes[spike_start_idx]) / (pip_size or 0.01)
            abs_move_pips = abs(float(net_move_pips))
            if abs_move_pips < float(min_spike_pips):
                continue

            spike_direction = 1 if net_move_pips > 0 else -1
            deltas = np.diff(closes[spike_start_idx : spike_end_idx + 1]) / (pip_size or 0.01)
            directional_hits = int(np.sum(deltas * spike_direction > 0))
            consistency = float(directional_hits) / max(len(deltas), 1)
            if consistency < float(min_directional_consistency):
                continue

            spike_bodies = np.abs(closes[spike_start_idx + 1 : spike_end_idx + 1] - opens[spike_start_idx + 1 : spike_end_idx + 1]) / (pip_size or 0.01)
            avg_spike_body = max(float(np.mean(spike_bodies)), 1e-6)
            spike_close = float(closes[spike_end_idx])
            saw_stall = False
            valid = True

            for j in range(spike_end_idx + 1, signal_idx + 1):
                if spike_direction > 0:
                    if ((highs[j] - spike_close) / (pip_size or 0.01)) > float(max_extension_after_spike_pips):
                        valid = False
                        break
                else:
                    if ((spike_close - lows[j]) / (pip_size or 0.01)) > float(max_extension_after_spike_pips):
                        valid = False
                        break
                body_pips = abs(closes[j] - opens[j]) / (pip_size or 0.01)
                if body_pips <= avg_spike_body * float(stall_body_fraction):
                    saw_stall = True

            if not valid or not saw_stall:
                continue

            confirm_direction = np.sign(closes[signal_idx] - opens[signal_idx])
            if confirm_direction == 0 or int(confirm_direction) == int(spike_direction):
                continue

            bias = "sell" if spike_direction > 0 else "buy"
            return {
                "family": "post_spike_retracement_ny_overlap_v1",
                "reason": f"post_spike_retrace:{session_label}:{'up' if spike_direction > 0 else 'down'}_spike",
                "bias": bias,
                "level_label": f"{session_label.upper()}_SPIKE_RETRACE",
                "level_price": round(float(tick_mid), 3),
                "nearest_level_pips": 0.0,
                "micro_confirmation": "stalled_spike_first_opposite_close",
                "spike_direction": "up" if spike_direction > 0 else "down",
                "spike_move_pips": round(abs_move_pips, 2),
                "confirmation_bars": int(confirmation_bars),
                "directional_consistency": round(float(consistency), 3),
            }
    except Exception:
        return None
    return None


def _trend_expansion_trigger(
    tick_mid: float,
    data_by_tf: dict[str, pd.DataFrame],
    *,
    adx_min: float,
    min_m5_atr_pips: float,
    extension_atr_mult: float,
    require_m3_trend: bool,
    require_m1_stack: bool,
    require_any_trend_signal: bool,
    reject_mismatch: bool,
    pip_size: float = 0.01,
) -> Optional[dict[str, Any]]:
    m3 = _m3_trend(data_by_tf)
    m1 = _m1_stack(data_by_tf)
    m5 = _m5_stack(data_by_tf)

    if require_m3_trend and m3 is None:
        return None
    if require_m1_stack and m1 is None:
        return None
    if require_any_trend_signal and m3 is None and m1 is None:
        return None
    if reject_mismatch and m3 is not None and m1 is not None and m3 != m1:
        return None

    directional = [x for x in (m3, m1, m5) if x in ("bull", "bear")]
    if not directional:
        return None
    if directional.count("bull") >= 2:
        bias = "buy"
    elif directional.count("bear") >= 2:
        bias = "sell"
    elif m3 == "bull" or m1 == "bull":
        bias = "buy"
    elif m3 == "bear" or m1 == "bear":
        bias = "sell"
    else:
        return None

    expected_stack = "bull" if bias == "buy" else "bear"
    if m1 in ("bull", "bear") and m1 != expected_stack:
        return None
    if m5 in ("bull", "bear") and m5 != expected_stack:
        return None

    adx_m5 = _adx_value(data_by_tf, "M5", period=14)
    adx_m15 = _adx_value(data_by_tf, "M15", period=14)
    adx = max(v for v in (adx_m5, adx_m15) if isinstance(v, (int, float)) and math.isfinite(v)) if any(
        isinstance(v, (int, float)) and math.isfinite(v) for v in (adx_m5, adx_m15)
    ) else None
    if adx is None or adx < float(adx_min):
        return None

    m5_atr_pips = _atr_pips(data_by_tf, "M5", period=14, pip_size=pip_size)
    if m5_atr_pips is None or m5_atr_pips < float(min_m5_atr_pips):
        return None

    m5_df = data_by_tf.get("M5") if data_by_tf else None
    if m5_df is None or len(m5_df) < 25:
        return None
    close = m5_df["close"].astype(float)
    e9 = float(_ema(close, 9).iloc[-1])
    last = float(close.iloc[-1])
    if not all(math.isfinite(v) for v in (e9, last)):
        return None
    extension_pips = abs(last - e9) / (pip_size or 0.01)
    extension_limit = max(float(min_m5_atr_pips), float(m5_atr_pips) * float(extension_atr_mult))
    if extension_pips > extension_limit:
        return None

    return {
        "family": "trend_expansion",
        "reason": f"adx_trend_expansion:{bias}",
        "bias": bias,
        "adx": round(float(adx), 2),
        "m5_atr_pips": round(float(m5_atr_pips), 2),
        "extension_pips": round(float(extension_pips), 2),
        "extension_limit_pips": round(float(extension_limit), 2),
        "trend_alignment": {"m3": m3, "m1": m1, "m5": m5},
    }


def _momentum_continuation_trigger(
    tick_mid: float,
    data_by_tf: dict[str, pd.DataFrame],
    session_label: str,
    *,
    min_m5_atr_pips: float,
    adx_min: float,
    clear_path_pips: float,
    lookback_bars: int,
    pullback_zone_pips: float,
    extension_atr_mult: float,
    pip_size: float = 0.01,
) -> Optional[dict[str, Any]]:
    """Detect a clean directional run that is offering a pullback/retest entry."""
    try:
        m1_df = data_by_tf.get("M1") if data_by_tf else None
        m5_df = data_by_tf.get("M5") if data_by_tf else None
        if m1_df is None or m5_df is None or len(m1_df) < 30 or len(m5_df) < 25:
            return None

        m1 = _m1_stack(data_by_tf)
        m5 = _m5_stack(data_by_tf)
        if m1 is None or m5 is None or m1 != m5:
            return None
        bias = "buy" if m1 == "bull" else "sell"

        m5_atr_pips = _atr_pips(data_by_tf, "M5", period=14, pip_size=pip_size)
        if m5_atr_pips is None or m5_atr_pips < float(min_m5_atr_pips):
            return None
        adx_m5 = _adx_value(data_by_tf, "M5", period=14)
        adx_m15 = _adx_value(data_by_tf, "M15", period=14)
        adx_candidates = [v for v in (adx_m5, adx_m15) if isinstance(v, (int, float)) and math.isfinite(v)]
        adx = max(adx_candidates) if adx_candidates else None
        if adx is None or adx < float(adx_min):
            return None

        pip = pip_size or 0.01
        m1_close = m1_df["close"].astype(float)
        m1_high = m1_df["high"].astype(float)
        m1_low = m1_df["low"].astype(float)
        m5_close = m5_df["close"].astype(float)
        m5_e9 = _ema(m5_close, 9)
        m5_e21 = _ema(m5_close, 21)
        m1_e9 = _ema(m1_close, 9)
        m1_e21 = _ema(m1_close, 21)
        last = float(m1_close.iloc[-1])
        last_m5 = float(m5_close.iloc[-1])
        last_m1_e9 = float(m1_e9.iloc[-1])
        last_m1_e21 = float(m1_e21.iloc[-1])
        last_m5_e9 = float(m5_e9.iloc[-1])
        last_m5_e21 = float(m5_e21.iloc[-1])
        if not all(math.isfinite(v) for v in (last, last_m5, last_m1_e9, last_m1_e21, last_m5_e9, last_m5_e21)):
            return None

        if bias == "buy":
            if not (last > last_m1_e9 > last_m1_e21 and last_m5 > last_m5_e9 > last_m5_e21):
                return None
        else:
            if not (last < last_m1_e9 < last_m1_e21 and last_m5 < last_m5_e9 < last_m5_e21):
                return None

        lookback = max(6, min(int(lookback_bars or 10), len(m1_df) - 1))
        recent = m1_df.tail(lookback)
        half = max(3, lookback // 2)
        prior = recent.iloc[: lookback - half]
        current = recent.iloc[lookback - half :]
        if len(prior) < 3 or len(current) < 3:
            return None
        prior_high = float(prior["high"].astype(float).max())
        prior_low = float(prior["low"].astype(float).min())
        current_high = float(current["high"].astype(float).max())
        current_low = float(current["low"].astype(float).min())
        if bias == "buy":
            structure_ok = current_high > prior_high and current_low > prior_low
        else:
            structure_ok = current_low < prior_low and current_high < prior_high
        if not structure_ok:
            return None

        prox = _nearest_structure_pips(tick_mid, data_by_tf, session_label, pip_size=pip_size)
        if bias == "buy":
            path_pips = prox.get("overhead_pips")
            path_label = prox.get("overhead_label")
            path_price = prox.get("overhead_price")
        else:
            path_pips = prox.get("underfoot_pips")
            path_label = prox.get("underfoot_label")
            path_price = prox.get("underfoot_price")
        if isinstance(path_pips, (int, float)) and path_pips < float(clear_path_pips):
            return None

        m1_atr_pips = _atr_pips(data_by_tf, "M1", period=14, pip_size=pip_size)
        zone_pips = max(float(pullback_zone_pips), min(5.0, (float(m1_atr_pips) * 0.8) if m1_atr_pips else float(pullback_zone_pips)))
        zone = zone_pips * pip
        tail_lows = m1_low.tail(4)
        tail_highs = m1_high.tail(4)
        if bias == "buy":
            touched_ema = bool((tail_lows <= last_m1_e9 + zone).any() or (tail_lows <= last_m1_e21 + zone).any())
            retested_break = prior_high > 0 and abs(last - prior_high) <= max(zone, 2.0 * pip) and last >= prior_high - zone
            recovered = last >= last_m1_e9
        else:
            touched_ema = bool((tail_highs >= last_m1_e9 - zone).any() or (tail_highs >= last_m1_e21 - zone).any())
            retested_break = prior_low > 0 and abs(last - prior_low) <= max(zone, 2.0 * pip) and last <= prior_low + zone
            recovered = last <= last_m1_e9
        if not recovered or not (touched_ema or retested_break):
            return None

        m5_extension_pips = abs(last_m5 - last_m5_e9) / pip
        extension_limit = max(float(min_m5_atr_pips), float(m5_atr_pips) * float(extension_atr_mult))
        if m5_extension_pips > extension_limit:
            return None

        score = 70.0
        score += min(12.0, max(0.0, float(adx) - float(adx_min)) * 0.6)
        score += min(8.0, max(0.0, (float(path_pips) if isinstance(path_pips, (int, float)) else float(clear_path_pips)) - float(clear_path_pips)) * 0.25)
        score += 5.0 if retested_break else 0.0
        score -= min(10.0, m5_extension_pips)

        return {
            "family": "momentum_continuation",
            "reason": f"clean_impulse_continuation:{bias}",
            "bias": bias,
            "level_label": "M1_EMA_RETEST" if touched_ema else "POST_BREAK_RETEST",
            "level_price": round(float(last_m1_e9 if touched_ema else (prior_high if bias == "buy" else prior_low)), 3),
            "nearest_level_pips": round(float(path_pips), 2) if isinstance(path_pips, (int, float)) else None,
            "micro_confirmation": "m1_m5_aligned_hh_hl_pullback" if bias == "buy" else "m1_m5_aligned_ll_lh_pullback",
            "m1_structure": "higher_highs_higher_lows" if bias == "buy" else "lower_lows_lower_highs",
            "m5_atr_pips": round(float(m5_atr_pips), 2),
            "adx": round(float(adx), 2),
            "clear_path_pips": round(float(path_pips), 2) if isinstance(path_pips, (int, float)) else None,
            "clear_path_label": path_label,
            "clear_path_price": path_price,
            "entry_pattern": "pullback_to_ema" if touched_ema else "post_break_retest",
            "m5_extension_pips": round(float(m5_extension_pips), 2),
            "extension_limit_pips": round(float(extension_limit), 2),
            "trigger_score": round(float(score), 2),
        }
    except Exception:
        return None


def _correlated_open_position(
    db_path: Optional[Path],
    side: str,
    proposed_price: float,
    distance_pips: float,
    pip_size: float = 0.01,
) -> Optional[dict[str, Any]]:
    """Return an open autonomous Fillmore position that conflicts with a new proposed entry.

    Conflict = same side AND fill_price within ``distance_pips`` of the
    proposed entry. The autonomous loop calls this AFTER the LLM responds, so
    we can act on the proposed side+price (which the gate doesn't know).
    """
    if db_path is None or distance_pips <= 0 or proposed_price <= 0:
        return None
    side_norm = str(side or "").lower()
    if side_norm not in ("buy", "sell"):
        return None
    try:
        from api import suggestion_tracker
        open_rows = suggestion_tracker.get_open_filled_positions(db_path)
    except Exception:
        return None
    pip = pip_size or 0.01
    distance_price = distance_pips * pip
    for r in open_rows:
        try:
            from api import suggestion_tracker
            if not suggestion_tracker.is_autonomous_suggestion_row(r):
                continue
        except Exception:
            continue
        if str(r.get("side") or "").lower() != side_norm:
            continue
        try:
            fill = float(r.get("fill_price") or r.get("limit_price") or 0)
        except (TypeError, ValueError):
            continue
        if fill <= 0:
            continue
        if abs(fill - proposed_price) <= distance_price:
            return r
    return None


def _recent_setup_already_fired(
    db_path: Optional[Path],
    tick_mid: float,
    window_min: int,
    bucket_pips: float,
    pip_size: float = 0.01,
) -> Optional[dict[str, Any]]:
    """Same-setup dedupe lookup.

    Returns the matching prior suggestion row if a *placed* suggestion fired
    within ``window_min`` minutes whose ``limit_price`` falls in the same
    ``bucket_pips`` price bucket as ``tick_mid``. The bucket is anchored on
    ``tick_mid`` (so a 25p bucket means anything within ±12.5p counts as same
    bucket — captures repeat fades at the same level regardless of side).
    Returns None if no match or the lookup fails.

    Skipping rejected/non-placed suggestions is intentional: an LLM skip
    (lots=0) doesn't burn the level the way a real placement does.
    """
    if db_path is None or window_min <= 0 or bucket_pips <= 0:
        return None
    try:
        from api import suggestion_tracker
        cutoff_iso = (datetime.now(timezone.utc) - timedelta(minutes=int(window_min))).isoformat()
        rows = suggestion_tracker.get_recent_suggestions_since(db_path, cutoff_iso)
    except Exception:
        return None
    pip = pip_size or 0.01
    half_bucket = (bucket_pips / 2.0) * pip
    for r in rows:
        action = str(r.get("action") or "")
        if action != "placed":
            continue
        try:
            prior_price = float(r.get("limit_price") or 0)
        except (TypeError, ValueError):
            continue
        if prior_price <= 0:
            continue
        if abs(prior_price - tick_mid) <= half_bucket:
            return r
    return None


def _near_daily_hl(tick_mid: float, data_by_tf: dict[str, pd.DataFrame], buffer_pips: float) -> bool:
    """True if current price is within buffer_pips of today's H or L."""
    df = data_by_tf.get("D") if data_by_tf else None
    if df is None or len(df) == 0 or buffer_pips <= 0:
        return False
    try:
        today = df.iloc[-1]
        hi = float(today["high"])
        lo = float(today["low"])
    except Exception:
        return False
    pip = 0.01
    if abs(tick_mid - hi) / pip <= buffer_pips:
        return True
    if abs(tick_mid - lo) / pip <= buffer_pips:
        return True
    return False


def _gate_setup_key(
    trigger: dict[str, Any],
    *,
    tick_mid: float,
    session_label: str,
    pip_size: float = 0.01,
) -> str:
    family = str(trigger.get("family") or "unknown").strip().lower()
    bias = str(trigger.get("bias") or "flat").strip().lower()
    if family == "critical_level_reaction":
        try:
            price = float(trigger.get("level_price") or tick_mid)
        except (TypeError, ValueError):
            price = float(tick_mid)
        bucket_size = max((pip_size or 0.01) * float(_CRITICAL_LEVEL_SETUP_BUCKET_PIPS), pip_size or 0.01)
        bucket = round(round(price / bucket_size) * bucket_size, 3)
        return f"{family}:{bias}:{session_label}:{bucket:.3f}"
    if family == "tight_range_mean_reversion":
        label = str(trigger.get("level_label") or "tokyo_range")
        try:
            price = float(trigger.get("level_price") or tick_mid)
        except (TypeError, ValueError):
            price = float(tick_mid)
        bucket_size = max((pip_size or 0.01) * 20.0, pip_size or 0.01)
        bucket = round(round(price / bucket_size) * bucket_size, 3)
        return f"{family}:{bias}:{label}:{bucket:.3f}"
    if family == "compression_breakout":
        label = str(trigger.get("level_label") or "level")
        try:
            price = round(float(trigger.get("level_price") or tick_mid), 3)
        except (TypeError, ValueError):
            price = round(float(tick_mid), 3)
        return f"{family}:{bias}:{label}:{price:.3f}"
    if family == "failed_breakout_reversal_overlap_v1":
        label = str(trigger.get("level_label") or "session_level")
        hold_bars = int(trigger.get("hold_bars") or 0)
        try:
            price = round(float(trigger.get("level_price") or tick_mid), 3)
        except (TypeError, ValueError):
            price = round(float(tick_mid), 3)
        return f"{family}:{bias}:{label}:{hold_bars}:{price:.3f}"
    if family == "post_spike_retracement_ny_overlap_v1":
        spike_direction = str(trigger.get("spike_direction") or "spike")
        confirmation_bars = int(trigger.get("confirmation_bars") or 0)
        bucket_size = max((pip_size or 0.01) * 20.0, pip_size or 0.01)
        bucket = round(round(float(tick_mid) / bucket_size) * bucket_size, 3)
        return f"{family}:{bias}:{session_label}:{spike_direction}:{confirmation_bars}:{bucket:.3f}"
    bucket_size = max((pip_size or 0.01) * float(_TREND_EXPANSION_SETUP_BUCKET_PIPS), pip_size or 0.01)
    bucket = round(round(float(tick_mid) / bucket_size) * bucket_size, 3)
    return f"{family}:{bias}:{session_label}:{bucket:.3f}"


def _prune_gate_setup_cooldowns(rt: dict[str, Any], now_utc: datetime) -> None:
    rows = list(rt.get("recent_fired_gate_setups") or [])
    kept: list[dict[str, Any]] = []
    for row in rows:
        until = _parse_iso(row.get("until_utc"))
        if until is None or until <= now_utc:
            continue
        kept.append(row)
    rt["recent_fired_gate_setups"] = kept


def _gate_setup_is_suppressed(rt: dict[str, Any], family: str, setup_key: str, now_utc: datetime) -> bool:
    _prune_gate_setup_cooldowns(rt, now_utc)
    for row in list(rt.get("recent_fired_gate_setups") or []):
        if str(row.get("family") or "") != str(family):
            continue
        if str(row.get("setup_key") or "") != str(setup_key):
            continue
        until = _parse_iso(row.get("until_utc"))
        if until is not None and until > now_utc:
            return True
    return False


def _mark_gate_setup_fired(rt: dict[str, Any], family: str, setup_key: str, now_utc: datetime) -> None:
    _prune_gate_setup_cooldowns(rt, now_utc)
    minutes = int(_GATE_SETUP_COOLDOWN_MINUTES.get(str(family), 20))
    until = (now_utc + timedelta(minutes=max(1, minutes))).isoformat()
    rows = [
        row
        for row in list(rt.get("recent_fired_gate_setups") or [])
        if not (str(row.get("family") or "") == str(family) and str(row.get("setup_key") or "") == str(setup_key))
    ]
    rows.append({
        "family": str(family),
        "setup_key": str(setup_key),
        "fired_utc": now_utc.isoformat(),
        "until_utc": until,
    })
    rt["recent_fired_gate_setups"] = rows[-50:]


def _update_active_gate_setup(
    rt: dict[str, Any],
    family: str,
    trigger: Optional[dict[str, Any]],
    *,
    tick_mid: float,
    session_label: str,
    now_utc: datetime,
    pip_size: float = 0.01,
) -> tuple[bool, Optional[str]]:
    active = dict(rt.get("active_gate_setups") or {})
    current = active.get(family) if isinstance(active.get(family), dict) else None
    if trigger is None:
        if family in active:
            active.pop(family, None)
            rt["active_gate_setups"] = active
        return False, None
    setup_key = _gate_setup_key(trigger, tick_mid=tick_mid, session_label=session_label, pip_size=pip_size)
    if current is not None and str(current.get("setup_key") or "") == setup_key:
        current["last_seen_utc"] = now_utc.isoformat()
        active[family] = current
        rt["active_gate_setups"] = active
        return False, setup_key
    active[family] = {
        "setup_key": setup_key,
        "family": family,
        "first_seen_utc": now_utc.isoformat(),
        "last_seen_utc": now_utc.isoformat(),
    }
    rt["active_gate_setups"] = active
    return True, setup_key


@dataclass
class GateInputs:
    spread_pips: float
    tick_mid: float
    open_ai_trade_count: int
    data_by_tf: dict[str, pd.DataFrame]
    ntz_active: bool = False
    suggestions_db_path: Optional[Path] = None  # for same-setup dedupe lookup; None disables the check
    upcoming_events: Optional[list[dict[str, Any]]] = None  # from get_economic_calendar_events(); None skips check
    order_book: Optional[dict[str, Any]] = None  # OANDA order-book cluster summary, when available


# -----------------------------------------------------------------------------
# Phase 1 guided-reasoning helpers: thesis fingerprint + recent-fire stats
# -----------------------------------------------------------------------------

def compute_thesis_fingerprint(
    side: Optional[str],
    trigger_family: Optional[str],
    level_label: Optional[str],
    level_price: Optional[float],
    htf_bias: Optional[str] = None,
) -> str:
    """Canonical thesis fingerprint for repetition visibility.

    Buckets level_price to nearest 0.10 (≈10 pips on USDJPY) so 159.48 / 159.50
    collapse to the same fingerprint while 159.50 / 159.62 do not. Side and
    trigger family are normalized; htf_bias is optional.
    """
    s = (side or "").strip().lower() or "unknown"
    fam = (trigger_family or "unknown").strip().lower()
    label = (level_label or "unknown").strip().upper()
    try:
        if level_price is None:
            bucket_s = "na"
        else:
            bucket = round(float(level_price) * 10.0) / 10.0
            bucket_s = f"{bucket:.2f}"
    except (TypeError, ValueError):
        bucket_s = "na"
    htf = (htf_bias or "na").strip().lower()
    return f"{s}:{fam}:{label}@{bucket_s}:{htf}"


def _compute_recent_fingerprint_stats(
    store,
    profile_name: str,
    fingerprint: str,
    window_minutes: int = 120,
) -> dict[str, Any]:
    """Look up trades with matching thesis_fingerprint within the window.

    Returns dict with: count, wins, losses, net_pips, last_outcome, last_pips,
    minutes_since_last. Best-effort — any failure returns zeros.
    """
    out: dict[str, Any] = {
        "count": 0,
        "wins": 0,
        "losses": 0,
        "net_pips": 0.0,
        "last_outcome": None,
        "last_pips": None,
        "minutes_since_last": None,
    }
    if not fingerprint or store is None:
        return out
    try:
        df = store.get_trades_df(profile_name) if hasattr(store, "get_trades_df") else None
    except Exception:
        return out
    if df is None or getattr(df, "empty", True):
        return out
    if "thesis_fingerprint" not in df.columns:
        return out
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
        cutoff_iso = cutoff.isoformat()
        sub = df[df["thesis_fingerprint"] == fingerprint]
        if sub.empty:
            return out
        sub = sub[sub["timestamp_utc"] >= cutoff_iso]
        if sub.empty:
            return out
        sub = sub.sort_values("timestamp_utc")
        out["count"] = int(len(sub))
        for _, row in sub.iterrows():
            pips = row.get("pips")
            if pips is None or (isinstance(pips, float) and pd.isna(pips)):
                continue
            try:
                p = float(pips)
            except (TypeError, ValueError):
                continue
            out["net_pips"] += p
            if p > 0:
                out["wins"] += 1
            elif p < 0:
                out["losses"] += 1
        last = sub.iloc[-1]
        last_pips = last.get("pips")
        if last_pips is not None and not (isinstance(last_pips, float) and pd.isna(last_pips)):
            try:
                lp = float(last_pips)
                out["last_pips"] = round(lp, 1)
                if lp > 0:
                    out["last_outcome"] = "win"
                elif lp < 0:
                    out["last_outcome"] = "loss"
                else:
                    out["last_outcome"] = "scratch"
            except (TypeError, ValueError):
                pass
        try:
            last_ts = datetime.fromisoformat(str(last.get("timestamp_utc")).replace("Z", "+00:00"))
            if last_ts.tzinfo is None:
                last_ts = last_ts.replace(tzinfo=timezone.utc)
            out["minutes_since_last"] = round(
                (datetime.now(timezone.utc) - last_ts).total_seconds() / 60.0,
                1,
            )
        except Exception:
            pass
        out["net_pips"] = round(out["net_pips"], 1)
    except Exception:
        return out
    return out


def evaluate_gate(
    cfg: dict[str, Any],
    rt: dict[str, Any],
    inputs: GateInputs,
    risk_regime: Optional[dict[str, Any]] = None,
    now_utc: Optional[datetime] = None,
) -> GateDecision:
    """Run all three gate layers. Returns the decision (does NOT mutate state)."""
    now_utc = now_utc or datetime.now(timezone.utc)
    mode = str(cfg.get("mode") or "off")
    agg = str(cfg.get("aggressiveness") or "balanced")
    risk_regime = risk_regime or {
        "label": "normal",
        "effective_min_llm_cooldown_sec": int(cfg.get("min_llm_cooldown_sec") or 60),
        "effective_max_open_ai_trades": int(cfg.get("max_open_ai_trades") or 2),
    }

    def _block(layer: str, reason: str, extras: Optional[dict[str, Any]] = None) -> GateDecision:
        return GateDecision(
            timestamp_utc=now_utc.isoformat(),
            result="block", layer=layer, reason=reason,  # type: ignore[arg-type]
            mode=mode, aggressiveness=agg,
            extras=extras or {},
        )

    # ---- Layer 1: hard filters ----
    if not cfg.get("enabled"):
        return _block("hard", "autonomous_disabled")
    if mode == "off":
        return _block("hard", "mode_off")
    allowed, session_label = _session_flag_now(cfg.get("trading_hours") or {})
    if not allowed:
        return _block("hard", f"out_of_session:{session_label}")
    if inputs.ntz_active:
        return _block("hard", "ntz_active")
    # Event blackout: block when a high-impact USD/JPY event is imminent.
    if cfg.get("event_blackout_enabled") and inputs.upcoming_events:
        blackout_min = int(cfg.get("event_blackout_minutes") or 30)
        for ev in inputs.upcoming_events:
            mins = ev.get("minutes_to_event")
            impact = str(ev.get("impact") or "").lower()
            currency = str(ev.get("currency") or "").upper()
            if currency not in ("USD", "JPY", "ALL"):
                continue
            if impact not in ("high", "red"):
                continue
            if isinstance(mins, (int, float)) and 0 <= mins <= blackout_min:
                return _block("hard", f"event_blackout:{ev.get('event', '?')}_{mins}m")
    # Budget cap
    if float(rt.get("llm_spend_today_usd") or 0.0) >= float(cfg.get("daily_budget_usd") or 2.0):
        return _block("hard", f"budget_cap:${rt.get('llm_spend_today_usd'):.3f}")
    # Min LLM cooldown
    last_call = rt.get("last_llm_call_utc")
    if last_call:
        try:
            last = datetime.fromisoformat(last_call)
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            elapsed = (now_utc - last).total_seconds()
            effective_cooldown = float(risk_regime.get("effective_min_llm_cooldown_sec") or cfg.get("min_llm_cooldown_sec") or 60)
            if elapsed < effective_cooldown:
                return _block("hard", f"llm_cooldown:{elapsed:.0f}s", {"effective_cooldown_sec": effective_cooldown})
        except Exception:
            return _block("hard", "llm_cooldown:malformed_timestamp")

    # ---- Layer 2: adaptive throttle (checked before signal gate so we
    #     don't pay the signal-compute cost during a cooldown) ----
    throttle_until = rt.get("throttle_until_utc")
    throttle_reason = str(rt.get("throttle_reason") or "")
    if throttle_until and not throttle_reason.startswith("loss_streak="):
        try:
            tu = datetime.fromisoformat(throttle_until)
            if tu.tzinfo is None:
                tu = tu.replace(tzinfo=timezone.utc)
            if tu > now_utc:
                return _block("throttle", f"{throttle_reason or 'adaptive'}_until_{tu.isoformat()}")
        except Exception:
            pass

    # ---- Layer 3: hybrid opportunity gate (session-calibrated) ----
    thresholds = _resolve_gate_thresholds(agg, session_label)

    m3 = _m3_trend(inputs.data_by_tf)
    m1 = _m1_stack(inputs.data_by_tf)
    m5 = _m5_stack(inputs.data_by_tf)
    extras: dict[str, Any] = {
        "m3": m3,
        "m1": m1,
        "m5": m5,
        "spread": inputs.spread_pips,
        "session": session_label,
    }

    min_m5_atr_pips = float(thresholds.get("require_min_m5_atr_pips") or 0.0)
    m5_atr_pips = _atr_pips(inputs.data_by_tf, "M5", period=14, pip_size=0.01)
    extras["m5_atr_pips"] = round(m5_atr_pips, 2) if isinstance(m5_atr_pips, (int, float)) else None
    extras["adx_m5"] = round(_adx_value(inputs.data_by_tf, "M5", period=14) or 0.0, 2) or None
    extras["adx_m15"] = round(_adx_value(inputs.data_by_tf, "M15", period=14) or 0.0, 2) or None

    if min_m5_atr_pips > 0 and not _sufficient_volatility(
        inputs.data_by_tf,
        min_atr_pips=min_m5_atr_pips,
        timeframe="M5",
        pip_size=0.01,
    ):
        return _block("signal", "low_volatility", extras)

    prox = _nearest_structure_pips(
        inputs.tick_mid,
        inputs.data_by_tf,
        session_label,
        pip_size=0.01,
        order_book=inputs.order_book,
    )
    nearest = prox.get("nearest_pips")
    extras["nearest_level_pips"] = round(nearest, 1) if isinstance(nearest, (int, float)) else None
    extras["overhead_level_pips"] = round(prox["overhead_pips"], 1) if isinstance(prox.get("overhead_pips"), (int, float)) else None
    extras["underfoot_level_pips"] = round(prox["underfoot_pips"], 1) if isinstance(prox.get("underfoot_pips"), (int, float)) else None
    extras["overhead_level_label"] = prox.get("overhead_label")
    extras["underfoot_level_label"] = prox.get("underfoot_label")

    critical = _critical_level_reaction_trigger(
        inputs.tick_mid,
        inputs.data_by_tf,
        session_label,
        max_level_pips=float(thresholds.get("critical_level_max_pips") or 0.0),
        micro_window_bars=int(thresholds.get("critical_micro_window_bars") or 3),
        touch_tolerance_pips=float(thresholds.get("critical_touch_tolerance_pips") or 0.8),
        pip_size=0.01,
        order_book=inputs.order_book,
    )
    tokyo_meanrev = _tokyo_tight_range_mean_reversion_trigger(
        inputs.tick_mid,
        inputs.data_by_tf,
        session_label,
        range_window_bars=int(thresholds.get("tokyo_meanrev_window_bars") or 20),
        min_session_bars=int(thresholds.get("tokyo_meanrev_min_session_bars") or 45),
        max_range_pips=float(thresholds.get("tokyo_meanrev_max_range_pips") or 18.0),
        range_atr_mult=float(thresholds.get("tokyo_meanrev_range_atr_mult") or 3.6),
        edge_fraction=float(thresholds.get("tokyo_meanrev_edge_fraction") or 0.24),
        touch_tolerance_pips=float(thresholds.get("tokyo_meanrev_touch_tolerance_pips") or 0.8),
        adx_ceiling=float(thresholds.get("tokyo_meanrev_adx_ceiling") or 22.0),
        bb_width_max=float(thresholds.get("tokyo_meanrev_bb_width_max") or 0.0008),
        min_reward_pips=float(thresholds.get("tokyo_meanrev_min_reward_pips") or 2.5),
        pip_size=0.01,
    )
    compression = _compression_breakout_trigger(
        inputs.tick_mid,
        inputs.data_by_tf,
        session_label,
        max_level_pips=float(thresholds.get("compression_level_max_pips") or thresholds.get("critical_level_max_pips") or 0.0),
        compression_window_bars=int(thresholds.get("compression_window_bars") or 12),
        compression_range_atr_mult=float(thresholds.get("compression_range_atr_mult") or 0.9),
        edge_fraction=float(thresholds.get("compression_edge_fraction") or 0.2),
        adx_floor=float(thresholds.get("compression_adx_floor") or 12.0),
        adx_ceiling=float(thresholds.get("compression_adx_ceiling") or thresholds.get("trend_adx_min") or 24.0),
        require_m3_trend=bool(thresholds.get("require_m3_trend")),
        require_m1_stack=bool(thresholds.get("require_m1_stack")),
        require_any_trend_signal=bool(thresholds.get("require_any_trend_signal")),
        reject_mismatch=bool(thresholds.get("reject_m3_m1_mismatch_if_both_present")),
        pip_size=0.01,
    )
    if not _ENABLE_COMPRESSION_BREAKOUT:
        compression = None
    failed_breakout = (
        _failed_breakout_reversal_trigger(
            inputs.tick_mid,
            inputs.data_by_tf,
            session_label,
            pip_size=0.01,
        )
        if _ENABLE_FAILED_BREAKOUT_REVERSAL
        else None
    )
    post_spike = (
        _post_spike_retracement_trigger(
            inputs.tick_mid,
            inputs.data_by_tf,
            session_label,
            pip_size=0.01,
        )
        if _ENABLE_POST_SPIKE_RETRACEMENT
        else None
    )
    trend = _trend_expansion_trigger(
        inputs.tick_mid,
        inputs.data_by_tf,
        adx_min=float(thresholds.get("trend_adx_min") or 0.0),
        min_m5_atr_pips=min_m5_atr_pips,
        extension_atr_mult=float(thresholds.get("trend_extension_atr_mult") or 1.0),
        require_m3_trend=bool(thresholds.get("require_m3_trend")),
        require_m1_stack=bool(thresholds.get("require_m1_stack")),
        require_any_trend_signal=bool(thresholds.get("require_any_trend_signal")),
        reject_mismatch=bool(thresholds.get("reject_m3_m1_mismatch_if_both_present")),
        pip_size=0.01,
    )
    momentum = _momentum_continuation_trigger(
        inputs.tick_mid,
        inputs.data_by_tf,
        session_label,
        min_m5_atr_pips=min_m5_atr_pips,
        adx_min=float(thresholds.get("momentum_adx_min") or thresholds.get("trend_adx_min") or 18.0),
        clear_path_pips=float(thresholds.get("momentum_clear_path_pips") or 8.0),
        lookback_bars=int(thresholds.get("momentum_lookback_bars") or 10),
        pullback_zone_pips=float(thresholds.get("momentum_pullback_zone_pips") or 3.0),
        extension_atr_mult=float(thresholds.get("momentum_extension_atr_mult") or 1.25),
        pip_size=0.01,
    )
    if trend is not None and session_label not in _TREND_EXPANSION_ALLOWED_SESSIONS:
        extras["trend_session_veto"] = session_label
        trend = None
    if session_label == "tokyo":
        if "critical_level_reaction" not in _TOKYO_CONTROLLED_EXPERIMENT_FAMILIES:
            critical = None
        if "compression_breakout" not in _TOKYO_CONTROLLED_EXPERIMENT_FAMILIES:
            compression = None
        if "trend_expansion" not in _TOKYO_CONTROLLED_EXPERIMENT_FAMILIES:
            trend = None

    family_candidates = [
        ("critical_level_reaction", critical),
        ("momentum_continuation", momentum),
        ("tight_range_mean_reversion", tokyo_meanrev),
        ("compression_breakout", compression),
        ("failed_breakout_reversal_overlap_v1", failed_breakout),
        ("post_spike_retracement_ny_overlap_v1", post_spike),
        ("trend_expansion", trend),
    ]
    armed_candidates: list[tuple[str, dict[str, Any], str]] = []
    active_families: list[str] = []
    suppressed_families: list[str] = []
    for family_name, trig in family_candidates:
        transitioned, setup_key = _update_active_gate_setup(
            rt,
            family_name,
            trig,
            tick_mid=inputs.tick_mid,
            session_label=session_label,
            now_utc=now_utc,
            pip_size=0.01,
        )
        if trig is None or setup_key is None:
            continue
        active_families.append(family_name)
        if not transitioned:
            continue
        if _gate_setup_is_suppressed(rt, family_name, setup_key, now_utc):
            suppressed_families.append(family_name)
            continue
        trig = dict(trig)
        trig["setup_key"] = setup_key
        armed_candidates.append((family_name, trig, setup_key))

    if armed_candidates:
        equal_priority = [
            item for item in armed_candidates
            if item[0] in _EQUAL_PRIORITY_GATE_FAMILIES
        ]
        pool = equal_priority or armed_candidates
        chosen_family, chosen_trigger, _chosen_setup_key = max(
            pool,
            key=lambda item: float(item[1].get("trigger_score") or 50.0),
        )
        extras["chosen_trigger_family"] = chosen_family
        extras["candidate_trigger_scores"] = {
            family: trig.get("trigger_score")
            for family, trig, _setup_key in armed_candidates
            if trig.get("trigger_score") is not None
        }
    else:
        chosen_trigger = None
    extras["active_trigger_families"] = list(active_families)
    extras["suppressed_trigger_families"] = list(suppressed_families)
    if chosen_trigger is None:
        if active_families:
            if suppressed_families:
                return _block("signal", "trigger_setup_cooldown", extras)
            return _block("signal", "trigger_still_active", extras)
        if m3 is None and m1 is None:
            return _block("signal", "no_hybrid_trigger:no_trend_signal", extras)
        return _block("signal", "no_hybrid_trigger", extras)

    # Same-setup dedupe: skip the LLM call entirely if a placement already fired
    # in this price bucket recently. Avoids paying tokens to retry a tape-rejected
    # idea. The today-block in the prompt also surfaces it visually for the LLM,
    # but a hard gate-side block is cheaper and more reliable.
    if cfg.get("repeat_setup_dedupe_enabled"):
        window_min = int(cfg.get("repeat_setup_window_min") or 30)
        bucket_pips = float(cfg.get("repeat_setup_bucket_pips") or 25.0)
        prior = _recent_setup_already_fired(
            inputs.suggestions_db_path, inputs.tick_mid, window_min, bucket_pips,
        )
        if prior is not None:
            extras["dedupe_prior_side"] = prior.get("side")
            extras["dedupe_prior_price"] = prior.get("limit_price")
            extras["dedupe_prior_at_utc"] = prior.get("created_utc")
            return _block(
                "signal",
                f"repeat_setup_within_{bucket_pips:.0f}p_in_{window_min}m",
                extras,
            )

    _mark_gate_setup_fired(
        rt,
        str(chosen_trigger.get("family") or "unknown"),
        str(chosen_trigger.get("setup_key") or ""),
        now_utc,
    )

    thesis_fingerprint = compute_thesis_fingerprint(
        side=chosen_trigger.get("bias"),
        trigger_family=chosen_trigger.get("family"),
        level_label=chosen_trigger.get("level_label"),
        level_price=chosen_trigger.get("level_price"),
    )

    return GateDecision(
        timestamp_utc=now_utc.isoformat(),
        result="pass", layer="pass", reason="ok",
        mode=mode,
        aggressiveness=agg,
        extras={
            **extras,
            "risk_regime": risk_regime.get("label"),
            "thesis_fingerprint": thesis_fingerprint,
            "trigger_family": chosen_trigger.get("family"),
            "trigger_reason": chosen_trigger.get("reason"),
            "trigger_bias": chosen_trigger.get("bias"),
            "trigger_level_label": chosen_trigger.get("level_label"),
            "trigger_level_price": chosen_trigger.get("level_price"),
            "trigger_setup_key": chosen_trigger.get("setup_key"),
            "trigger_micro_confirmation": chosen_trigger.get("micro_confirmation"),
            "session_range_pips": chosen_trigger.get("session_range_pips"),
            "session_mid_price": chosen_trigger.get("session_mid_price"),
            "reward_to_mid_pips": chosen_trigger.get("reward_to_mid_pips"),
            "bb_width": chosen_trigger.get("bb_width"),
            "compression_range_pips": chosen_trigger.get("compression_range_pips"),
            "compression_cap_pips": chosen_trigger.get("compression_cap_pips"),
            "extension_pips": chosen_trigger.get("extension_pips"),
            "extension_limit_pips": chosen_trigger.get("extension_limit_pips"),
            "trigger_breakout_side": chosen_trigger.get("breakout_side"),
            "trigger_breakout_excursion_pips": chosen_trigger.get("breakout_excursion_pips"),
            "trigger_hold_bars": chosen_trigger.get("hold_bars"),
            "trigger_spike_direction": chosen_trigger.get("spike_direction"),
            "trigger_spike_move_pips": chosen_trigger.get("spike_move_pips"),
            "trigger_confirmation_bars": chosen_trigger.get("confirmation_bars"),
        },
    )


# -----------------------------------------------------------------------------
# Stats for the UI
# -----------------------------------------------------------------------------

def _default_performance_row() -> dict[str, Any]:
    return {
        "trade_count": 0,
        "closed_count": 0,
        "net_pnl": None,
        "win_rate": None,
        "avg_win_pips": None,
        "avg_loss_pips": None,
        "profit_factor": None,
        "avg_hold_minutes": None,
        "fill_rate_limits": None,
        "avg_time_to_fill_sec": None,
        "avg_fill_vs_requested_pips": None,
        "thesis_intervention_rate": None,
        "avg_mae_pips": None,
        "avg_mfe_pips": None,
        "win_rate_by_confidence_json": {},
        "win_rate_by_side_json": {},
        "win_rate_by_session_json": {},
        "prompt_version_breakdown_json": {},
        "updated_utc": None,
    }


def build_stats(state_path: Path, cfg: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    cfg = cfg or get_config(state_path)
    rt = refresh_runtime_from_history(state_path, cfg=cfg)
    risk_regime = _compute_risk_regime(rt, cfg)

    decisions = list(rt.get("decisions") or [])
    # Break down by result + layer over last N.
    total = len(decisions)
    passes = sum(1 for d in decisions if d.get("r") == "pass")
    blocks = total - passes
    layer_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    trigger_family_counts: dict[str, int] = {}
    trigger_reason_counts: dict[str, int] = {}
    for d in decisions:
        if d.get("r") == "block":
            layer_counts[str(d.get("l") or "?")] = layer_counts.get(str(d.get("l") or "?"), 0) + 1
            reason_counts[str(d.get("why") or "?")] = reason_counts.get(str(d.get("why") or "?"), 0) + 1
        trigger_family = ((d.get("x") or {}).get("trigger_family")) if isinstance(d.get("x"), dict) else None
        trigger_reason = ((d.get("x") or {}).get("trigger_reason")) if isinstance(d.get("x"), dict) else None
        if trigger_family:
            trigger_family_counts[str(trigger_family)] = trigger_family_counts.get(str(trigger_family), 0) + 1
        if trigger_reason:
            trigger_reason_counts[str(trigger_reason)] = trigger_reason_counts.get(str(trigger_reason), 0) + 1

    thresholds = GATE_THRESHOLDS.get(cfg.get("aggressiveness") or "balanced") or {}
    est_cost_per_call = _estimated_call_cost_usd(cfg.get("model") or "gpt-5.4-mini")

    throttle_until = rt.get("throttle_until_utc")
    throttled = False
    if throttle_until:
        try:
            tu = datetime.fromisoformat(throttle_until)
            if tu.tzinfo is None:
                tu = tu.replace(tzinfo=timezone.utc)
            throttled = tu > datetime.now(timezone.utc)
        except Exception:
            return _block("hard", "llm_cooldown:malformed_timestamp")

    suggestions_db, _assistant_db = _db_paths_from_state(state_path)
    try:
        perf_rows = autonomous_performance.get_materialized_stats(suggestions_db)
    except Exception:
        perf_rows = {}
    if not perf_rows or "rolling_20" not in perf_rows:
        try:
            autonomous_performance.recompute_performance_stats(
                profile=state_path.parent.name,
                suggestions_db_path=suggestions_db,
                assistant_db_path=_assistant_db,
            )
            perf_rows = autonomous_performance.get_materialized_stats(suggestions_db)
        except Exception:
            perf_rows = perf_rows or {}
    for key in ("rolling_20", "rolling_50", "today", "week"):
        if key not in perf_rows:
            perf_rows[key] = _default_performance_row()
    for row in perf_rows.values():
        for key in (
            "win_rate_by_confidence_json",
            "win_rate_by_side_json",
            "win_rate_by_session_json",
            "prompt_version_breakdown_json",
        ):
            if key in row:
                try:
                    row[key] = json.loads(row[key]) if row[key] else {}
                except Exception:
                    row[key] = {}

    order_metrics: dict[str, Any] = {
        "suggested": {"market": 0, "limit": 0},
        "placed": {"market": 0, "limit": 0},
        "filled": {"market": 0, "limit": 0},
        "cancelled": {"market": 0, "limit": 0},
        "expired": {"market": 0, "limit": 0},
        "by_trigger_family": {},
    }
    selection_metrics: dict[str, Any] = {
        "generated": 0,
        "placed": 0,
        "skips": 0,
        "skip_rate_pct": 0.0,
        "llm_custom_exit_suggestions": 0,
        "server_veto_total": 0,
        "server_veto_by_reason": {},
        "server_veto_by_day": [],
        "autonomous_pnl_by_day": [],
    }
    try:
        auto_rows = autonomous_performance.load_autonomous_suggestions(suggestions_db)
        selection_metrics["generated"] = len(auto_rows)
        selection_metrics["placed"] = sum(1 for row in auto_rows if str(row.get("action") or "") == "placed")
        selection_metrics["skips"] = sum(
            1
            for row in auto_rows
            if str(row.get("decision") or "").lower() == "skip" or float(row.get("lots") or 0.0) <= 0
        )
        selection_metrics["skip_rate_pct"] = round(
            float(selection_metrics["skips"]) / max(int(selection_metrics["generated"]), 1) * 100.0,
            1,
        )
        selection_metrics["llm_custom_exit_suggestions"] = sum(
            1 for row in auto_rows if str(row.get("exit_strategy") or "").lower() == "llm_custom_exit"
        )
        server_veto_by_reason: dict[str, int] = {}
        server_veto_by_day: dict[str, dict[str, Any]] = {}
        for row in auto_rows:
            skip_reason = str(row.get("skip_reason") or "").strip()
            if not skip_reason.startswith("server_veto:"):
                continue
            server_veto_by_reason[skip_reason] = server_veto_by_reason.get(skip_reason, 0) + 1
            created_utc = str(row.get("created_utc") or "")
            day_key = created_utc[:10] if len(created_utc) >= 10 else "unknown"
            day_bucket = server_veto_by_day.setdefault(
                day_key,
                {
                    "date": day_key,
                    "total": 0,
                    "by_reason": {},
                    "by_session": {},
                },
            )
            day_bucket["total"] = int(day_bucket.get("total") or 0) + 1
            by_reason = day_bucket.get("by_reason") if isinstance(day_bucket.get("by_reason"), dict) else {}
            by_reason[skip_reason] = int(by_reason.get(skip_reason) or 0) + 1
            day_bucket["by_reason"] = by_reason

            features = row.get("features") if isinstance(row.get("features"), dict) else {}
            session_label = str(features.get("session") or "unknown")
            by_session = day_bucket.get("by_session") if isinstance(day_bucket.get("by_session"), dict) else {}
            by_session[session_label] = int(by_session.get(session_label) or 0) + 1
            day_bucket["by_session"] = by_session

        selection_metrics["server_veto_total"] = int(sum(server_veto_by_reason.values()))
        selection_metrics["server_veto_by_reason"] = dict(
            sorted(server_veto_by_reason.items(), key=lambda kv: -kv[1])
        )
        selection_metrics["server_veto_by_day"] = sorted(
            server_veto_by_day.values(),
            key=lambda row: str(row.get("date") or ""),
            reverse=True,
        )[:30]

        daily_pnl_by_day: dict[str, dict[str, Any]] = {}
        for row in auto_rows:
            closed_at = str(row.get("closed_at") or "")
            if not closed_at:
                continue
            pnl_raw = row.get("pnl")
            try:
                pnl_val = float(pnl_raw) if pnl_raw is not None else None
            except (TypeError, ValueError):
                pnl_val = None
            if pnl_val is None:
                continue
            day_key = closed_at[:10] if len(closed_at) >= 10 else "unknown"
            bucket = daily_pnl_by_day.setdefault(
                day_key,
                {
                    "date": day_key,
                    "closed_count": 0,
                    "wins": 0,
                    "losses": 0,
                    "net_pnl": 0.0,
                    "net_pips": 0.0,
                },
            )
            bucket["closed_count"] = int(bucket.get("closed_count") or 0) + 1
            bucket["net_pnl"] = float(bucket.get("net_pnl") or 0.0) + pnl_val
            try:
                bucket["net_pips"] = float(bucket.get("net_pips") or 0.0) + float(row.get("pips") or 0.0)
            except (TypeError, ValueError):
                pass
            if pnl_val > 0:
                bucket["wins"] = int(bucket.get("wins") or 0) + 1
            elif pnl_val < 0:
                bucket["losses"] = int(bucket.get("losses") or 0) + 1
        daily_rows = sorted(
            daily_pnl_by_day.values(),
            key=lambda row: str(row.get("date") or ""),
            reverse=True,
        )[:30]
        for row in daily_rows:
            row["net_pnl"] = round(float(row.get("net_pnl") or 0.0), 2)
            row["net_pips"] = round(float(row.get("net_pips") or 0.0), 1)
        selection_metrics["autonomous_pnl_by_day"] = daily_rows

        def _inc(bucket: dict[str, int], order_type: str) -> None:
            key = "limit" if str(order_type).lower() == "limit" else "market"
            bucket[key] = int(bucket.get(key) or 0) + 1

        def _family_bucket(family: str) -> dict[str, Any]:
            fam = str(family or "unknown")
            bucket = order_metrics["by_trigger_family"].get(fam)
            if isinstance(bucket, dict):
                return bucket
            bucket = {
                "suggested": {"market": 0, "limit": 0},
                "placed": {"market": 0, "limit": 0},
                "filled": {"market": 0, "limit": 0},
                "cancelled": {"market": 0, "limit": 0},
                "expired": {"market": 0, "limit": 0},
                "fill_rate": {},
                "avg_time_to_fill_sec": {},
            }
            order_metrics["by_trigger_family"][fam] = bucket
            return bucket

        time_to_fill: dict[tuple[str, str], list[float]] = {}
        for row in auto_rows:
            suggested_type = str(row.get("order_type") or "market").lower()
            placed_order = row.get("placed_order") or {}
            placed_type = str(placed_order.get("order_type") or suggested_type or "market").lower()
            family = str(row.get("trigger_family") or placed_order.get("trigger_family") or "unknown")
            status = str(row.get("outcome_status") or "").lower()
            fam_bucket = _family_bucket(family)

            _inc(order_metrics["suggested"], suggested_type)
            _inc(fam_bucket["suggested"], suggested_type)

            if str(row.get("action") or "") == "placed":
                _inc(order_metrics["placed"], placed_type)
                _inc(fam_bucket["placed"], placed_type)
                if status == "filled":
                    _inc(order_metrics["filled"], placed_type)
                    _inc(fam_bucket["filled"], placed_type)
                    secs = autonomous_performance._time_to_fill_seconds(row)
                    if isinstance(secs, (int, float)):
                        time_to_fill.setdefault((family, placed_type), []).append(float(secs))
                elif status == "cancelled":
                    _inc(order_metrics["cancelled"], placed_type)
                    _inc(fam_bucket["cancelled"], placed_type)
                elif status == "expired":
                    _inc(order_metrics["expired"], placed_type)
                    _inc(fam_bucket["expired"], placed_type)

        for family, fam_bucket in list(order_metrics["by_trigger_family"].items()):
            for order_type in ("market", "limit"):
                placed_n = int((fam_bucket.get("placed") or {}).get(order_type) or 0)
                filled_n = int((fam_bucket.get("filled") or {}).get(order_type) or 0)
                fam_bucket["fill_rate"][order_type] = round((filled_n / placed_n), 4) if placed_n > 0 else None
                times = time_to_fill.get((family, order_type), [])
                fam_bucket["avg_time_to_fill_sec"][order_type] = (
                    round(sum(times) / len(times), 2) if times else None
                )
    except Exception:
        pass

    health_alerts: list[dict[str, Any]] = []
    last_tick = _parse_iso(rt.get("last_tick_utc"))
    if cfg.get("enabled") and last_tick is not None:
        age = (datetime.now(timezone.utc) - last_tick).total_seconds()
        if age > 120:
            health_alerts.append({
                "level": "error",
                "code": "loop_stale",
                "msg": f"Last autonomous tick was {age:.0f}s ago.",
            })
    if thr := dict(rt.get("recent_gate_blocks") or {}):
        top_reason = max(thr.items(), key=lambda kv: kv[1])[0] if thr else None
        if top_reason and passes == 0 and total >= 20:
            health_alerts.append({
                "level": "warning",
                "code": "gate_stagnation",
                "msg": f"Recent gate blocks are dominated by {top_reason}.",
            })
    if int(rt.get("consecutive_llm_errors") or 0) >= 3:
        health_alerts.append({
            "level": "error",
            "code": "llm_errors",
            "msg": f"{int(rt.get('consecutive_llm_errors') or 0)} consecutive LLM errors. Last: {rt.get('last_error_msg', '?')[:100]}",
        })
    llm_circuit = get_fillmore_llm_health()
    if str(llm_circuit.get("state") or "closed") != "closed":
        health_alerts.append({
            "level": "error" if llm_circuit.get("state") == "open" else "warning",
            "code": "llm_circuit_breaker",
            "msg": (
                f"Fillmore LLM circuit is {llm_circuit.get('state')}. "
                f"Last error: {str(llm_circuit.get('last_error') or 'n/a')[:100]}"
            ),
        })
    if int(rt.get("consecutive_broker_rejects") or 0) >= 3:
        health_alerts.append({
            "level": "error",
            "code": "broker_errors",
            "msg": f"{int(rt.get('consecutive_broker_rejects') or 0)} consecutive broker placement errors.",
        })

    return {
        "config": cfg,
        "gate_description": thresholds.get("description"),
        "expected_pass_rate_pct": thresholds.get("expected_pass_rate_pct"),
        "est_cost_per_llm_call_usd": round(est_cost_per_call, 6),
        "window": {
            "total": total,
            "passes": passes,
            "blocks": blocks,
            "pass_rate_pct": round((passes / total * 100.0), 2) if total > 0 else 0.0,
            "top_block_layers": dict(sorted(layer_counts.items(), key=lambda kv: -kv[1])[:5]),
            "top_block_reasons": dict(sorted(reason_counts.items(), key=lambda kv: -kv[1])[:8]),
            "trigger_families": dict(sorted(trigger_family_counts.items(), key=lambda kv: -kv[1])[:5]),
            "trigger_reasons": dict(sorted(trigger_reason_counts.items(), key=lambda kv: -kv[1])[:8]),
        },
        "today": {
            "llm_calls": int(rt.get("llm_calls_today") or 0),
            "spend_usd": round(float(rt.get("llm_spend_today_usd") or 0.0), 4),
            "budget_usd": float(cfg.get("daily_budget_usd") or 0.0),
            "budget_used_pct": round(
                (float(rt.get("llm_spend_today_usd") or 0.0) / max(float(cfg.get("daily_budget_usd") or 0.0001), 1e-6)) * 100.0, 1
            ),
            "trades_placed": int(rt.get("trades_placed_today") or 0),
            "pnl_usd": round(float(rt.get("daily_pnl_usd") or 0.0), 2),
        },
        "throttle": {
            "active": throttled,
            "until_utc": throttle_until if throttled else None,
            "reason": rt.get("throttle_reason") if throttled else None,
            "consecutive_no_trade_replies": int(rt.get("consecutive_no_trade_replies") or 0),
            "consecutive_losses": int(rt.get("consecutive_losses") or 0),
            "consecutive_wins": int(rt.get("consecutive_wins") or 0),
            "consecutive_errors": int(rt.get("consecutive_errors") or 0),
        },
        "risk_regime": {
            **risk_regime,
            "override_until_utc": rt.get("risk_regime_override_until_utc"),
        },
        "llm_circuit_breaker": llm_circuit,
        "recent_gate_blocks": dict(sorted((rt.get("recent_gate_blocks") or {}).items(), key=lambda kv: -kv[1])[:8]),
        "health_alerts": health_alerts,
        "performance": perf_rows,
        "order_metrics": order_metrics,
        "selection_metrics": selection_metrics,
        "last_tick_utc": rt.get("last_tick_utc"),
        "last_llm_call_utc": rt.get("last_llm_call_utc"),
        "last_placed_order_id": rt.get("last_placed_order_id"),
        "last_suggestion_id": rt.get("last_suggestion_id"),
        "recent_decisions": decisions[-30:][::-1],  # most recent first
    }


# -----------------------------------------------------------------------------
# Engine — called from run_loop each tick
# -----------------------------------------------------------------------------

def _count_open_ai_trades(store, profile_name: str) -> int:
    from api import suggestion_tracker

    try:
        rows = store.list_open_trades(profile_name)
    except Exception:
        return 0
    return sum(1 for r in rows if suggestion_tracker.is_autonomous_trade_row(dict(r)))


def _refresh_daily_pnl(store, profile_name: str, rt: dict[str, Any], cfg: Optional[dict[str, Any]] = None) -> None:
    """Query closed autonomous Fillmore trades for today and update rt['daily_pnl_usd'].

    Runs best-effort — any error leaves rt['daily_pnl_usd'] unchanged. Called
    each tick so the daily loss cap and loss-streak throttle see real outcomes
    without us needing to hook into every close path.
    """
    today = _today_utc_key()
    try:
        rows = store.get_trades_df(profile_name) if hasattr(store, "get_trades_df") else None
        if rows is None or getattr(rows, "empty", True):
            return
    except Exception:
        return


def _db_paths_from_state(state_path: Path) -> tuple[Path, Path]:
    base = state_path.parent
    return base / "ai_suggestions.sqlite", base / "assistant.db"


def _refresh_autonomous_runtime_from_history(
    *,
    state_path: Path,
    rt: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    suggestions_db, assistant_db = _db_paths_from_state(state_path)
    try:
        autonomous_performance.reconcile_closed_outcomes(
            suggestions_db_path=suggestions_db,
            assistant_db_path=assistant_db,
        )
    except Exception:
        pass
    try:
        today_activity = autonomous_performance.get_today_activity_counters(suggestions_db)
        rt["llm_calls_today"] = int(today_activity.get("llm_calls_today") or 0)
        rt["trades_placed_today"] = int(today_activity.get("trades_placed_today") or 0)
        rt["llm_spend_today_usd"] = (
            float(rt["llm_calls_today"]) * _estimated_call_cost_usd(str(cfg.get("model") or "gpt-5.4-mini"))
        )
    except Exception:
        pass
    try:
        latest_terminal = autonomous_performance.get_last_terminal_event_utc(suggestions_db)
    except Exception:
        latest_terminal = None
    rt["last_terminal_event_utc"] = latest_terminal

    stale = False
    if latest_terminal:
        last_stats = _parse_iso(rt.get("last_stats_recompute_utc"))
        term_dt = _parse_iso(latest_terminal)
        stale = term_dt is not None and (last_stats is None or term_dt >= last_stats)
    elif rt.get("last_stats_recompute_utc") is None:
        stale = True

    if stale:
        try:
            autonomous_performance.recompute_performance_stats(
                profile=state_path.parent.name,
                suggestions_db_path=suggestions_db,
                assistant_db_path=assistant_db,
            )
            rt["last_stats_recompute_utc"] = datetime.now(timezone.utc).isoformat()
        except Exception as _recompute_err:
            print(f"[{state_path.parent.name}] autonomous performance recompute error: {_recompute_err}")

    scratch_thr = float(cfg.get("streak_scratch_threshold_pips") or 1.0)
    try:
        outcomes = autonomous_performance.load_autonomous_closed_outcomes(
            suggestions_db,
            scratch_threshold_pips=scratch_thr,
        )
    except Exception:
        outcomes = []

    today = _today_utc_key()
    daily_pnl = 0.0
    consecutive_losses = 0
    consecutive_wins = 0
    latest_trade_id: str | None = None
    for row in outcomes:
        latest_trade_id = str(row.get("trade_id") or latest_trade_id or "")
        if str(row.get("closed_at") or "").startswith(today):
            daily_pnl += float(row.get("pnl") or 0.0)
        streak_outcome = str(row.get("streak_outcome") or "")
        if streak_outcome == "loss":
            consecutive_losses += 1
            consecutive_wins = 0
        elif streak_outcome == "win":
            consecutive_wins += 1
            consecutive_losses = 0
        elif streak_outcome == "scratch":
            consecutive_losses = 0
            consecutive_wins = 0
    rt["daily_pnl_usd"] = float(daily_pnl)
    rt["consecutive_losses"] = consecutive_losses
    rt["consecutive_wins"] = consecutive_wins
    if latest_trade_id:
        rt["latest_closed_trade_id"] = latest_trade_id

    # Phase 1: track session peak realized P&L for drawdown-from-peak visibility
    # in the LLM prompt. Resets each UTC day via the today-key check.
    peak_day = str(rt.get("session_peak_day") or "")
    if peak_day != today:
        rt["session_peak_day"] = today
        rt["session_peak_pnl_usd"] = float(daily_pnl)
        rt["session_peak_pnl_time_utc"] = datetime.now(timezone.utc).isoformat()
    else:
        prev_peak = float(rt.get("session_peak_pnl_usd") or 0.0)
        if float(daily_pnl) > prev_peak:
            rt["session_peak_pnl_usd"] = float(daily_pnl)
            rt["session_peak_pnl_time_utc"] = datetime.now(timezone.utc).isoformat()
    return {
        "outcomes": outcomes,
        "stats_stale": stale,
    }


def _compute_risk_regime(rt: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    label = "normal"
    prev_streak = str(
        rt.get("previous_streak_regime_label")
        or rt.get("previous_regime_label")
        or "normal"
    )
    losses = int(rt.get("consecutive_losses") or 0)
    wins = int(rt.get("consecutive_wins") or 0)
    latest_trade_id = rt.get("latest_closed_trade_id")

    if losses >= 3:
        streak_label = "defensive_hard"
    elif losses >= 2:
        streak_label = "defensive_soft"
    elif prev_streak == "defensive_hard":
        if wins >= 2:
            streak_label = "normal"
        elif wins >= 1:
            streak_label = "defensive_soft"
        else:
            streak_label = "defensive_hard"
    elif prev_streak == "defensive_soft":
        streak_label = "normal" if wins >= 1 else "defensive_soft"
    else:
        streak_label = "normal"

    label = streak_label

    override_label = None
    override_until = _parse_iso(rt.get("risk_regime_override_until_utc"))
    if rt.get("risk_regime_override") and override_until and override_until > datetime.now(timezone.utc):
        override_label = str(rt.get("risk_regime_override"))
        label = override_label

    if streak_label != prev_streak:
        rt["previous_streak_regime_label"] = streak_label
        rt["regime_entered_trade_id"] = latest_trade_id
    rt["previous_regime_label"] = label

    if label == "defensive_hard":
        risk_multiplier = 0.5
        effective_cooldown = int(cfg.get("min_llm_cooldown_sec") or 60) * 2
        effective_max_open = int(cfg.get("max_open_ai_trades") or 2)
    elif label == "defensive_soft":
        risk_multiplier = 0.75
        effective_cooldown = int(math.ceil(float(cfg.get("min_llm_cooldown_sec") or 60) * 1.5))
        effective_max_open = int(cfg.get("max_open_ai_trades") or 2)
    else:
        risk_multiplier = 1.0
        effective_cooldown = int(cfg.get("min_llm_cooldown_sec") or 60)
        effective_max_open = int(cfg.get("max_open_ai_trades") or 2)

    return {
        "label": label,
        "streak_label": streak_label,
        # Retained for API compatibility; no longer used to influence behavior.
        "daily_drawdown_active": False,
        "risk_multiplier": risk_multiplier,
        "effective_min_llm_cooldown_sec": effective_cooldown,
        "effective_max_open_ai_trades": effective_max_open,
        "override_label": override_label,
        "previous_regime_label": label,
        "previous_streak_regime_label": prev_streak,
        "recovery_wins": wins,
        "regime_entered_trade_id": rt.get("regime_entered_trade_id"),
    }
    try:
        df = rows.copy()
        # Filter to autonomous Fillmore entries that closed today.
        if not df.empty:
            mask = df.apply(lambda row: suggestion_tracker.is_autonomous_trade_row(row.to_dict()), axis=1)
            df = df[mask]
        if "close_time" not in df.columns and "exit_time_utc" in df.columns:
            df = df.rename(columns={"exit_time_utc": "close_time"})
        if "close_time" not in df.columns and "exit_timestamp_utc" in df.columns:
            df = df.rename(columns={"exit_timestamp_utc": "close_time"})
        if "close_time" in df.columns:
            df = df[df["close_time"].astype(str).str.startswith(today)]
        if df.empty:
            return
        pnl_col = None
        for cand in ("pnl_usd", "pnl", "net_pnl", "profit_usd", "profit"):
            if cand in df.columns:
                pnl_col = cand
                break
        if pnl_col is None:
            return
        daily = float(df[pnl_col].fillna(0).astype(float).sum())
        # Compute loss streak from closed autonomous Fillmore trades today (chronological).
        if "close_time" in df.columns:
            df = df.sort_values("close_time")
        losses_in_streak = 0
        for v in df[pnl_col].fillna(0).astype(float).tolist():
            if v < 0:
                losses_in_streak += 1
            elif v > 0:
                losses_in_streak = 0
            # v == 0: leave streak as-is
        rt["daily_pnl_usd"] = daily
        rt["consecutive_losses"] = losses_in_streak
        loss_thr = int((cfg or {}).get("throttle_loss_streak") or DEFAULT_CONFIG.get("throttle_loss_streak") or 2)
        if losses_in_streak < loss_thr:
            _clear_loss_streak_marker(rt)
    except Exception:
        return


def _summarize_oanda_order_book(raw_book: Optional[dict[str, Any]], *, range_pips: float = 100.0, top_n: int = 5) -> Optional[dict[str, Any]]:
    if not isinstance(raw_book, dict):
        return None
    ob = raw_book.get("orderBook") if isinstance(raw_book.get("orderBook"), dict) else raw_book
    if not isinstance(ob, dict):
        return None
    buckets = ob.get("buckets")
    if not isinstance(buckets, list):
        return None
    try:
        book_price = float(ob.get("price"))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(book_price) or book_price <= 0:
        return None
    pip = 0.01
    lo = book_price - float(range_pips) * pip
    hi = book_price + float(range_pips) * pip
    nearby: list[dict[str, float]] = []
    for bucket in buckets:
        if not isinstance(bucket, dict):
            continue
        try:
            price = float(bucket.get("price"))
            long_pct = float(bucket.get("longCountPercent", 0.0))
            short_pct = float(bucket.get("shortCountPercent", 0.0))
        except (TypeError, ValueError):
            continue
        if lo <= price <= hi:
            nearby.append({"price": price, "long_pct": long_pct, "short_pct": short_pct})
    if not nearby:
        return None
    buy_clusters = sorted(nearby, key=lambda x: x["long_pct"], reverse=True)[:top_n]
    sell_clusters = sorted(nearby, key=lambda x: x["short_pct"], reverse=True)[:top_n]
    below_buys = [b for b in buy_clusters if b["price"] < book_price]
    above_sells = [s for s in sell_clusters if s["price"] > book_price]
    nearest_support = below_buys[0]["price"] if below_buys else None
    nearest_resistance = above_sells[0]["price"] if above_sells else None
    return {
        "current_price": book_price,
        "buy_clusters": [{"price": b["price"], "pct": b["long_pct"]} for b in buy_clusters],
        "sell_clusters": [{"price": s["price"], "pct": s["short_pct"]} for s in sell_clusters],
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
        "nearest_support_distance_pips": round((book_price - nearest_support) / pip, 1) if nearest_support else None,
        "nearest_resistance_distance_pips": round((nearest_resistance - book_price) / pip, 1) if nearest_resistance else None,
    }


def tick_autonomous_fillmore(
    profile,
    profile_name: str,
    state_path: Path,
    store,
    tick,
    data_by_tf: dict[str, pd.DataFrame],
    ntz_active: bool = False,
    adapter: Any = None,
    **_unused: Any,
) -> None:
    """Called from the run loop each iteration after trade management.

    Fast-paths out when autonomous is disabled. When the gate passes and mode is
    paper/live, it invokes the suggest endpoint and places the limit.
    """
    cfg = get_config(state_path)
    if not cfg.get("enabled"):
        return  # fast path — don't even log when fully off.

    # Global kill/exit-only must override autonomous placement. The top-level
    # preset mode can be DISARMED while autonomous Fillmore is intentionally
    # running in its own paper/shadow mode, so do not treat DISARMED as an
    # autonomous hard stop.
    runtime_state = load_execution_state(state_path)
    if runtime_state.kill_switch or runtime_state.exit_system_only:
        return

    # Gather inputs.
    pip = float(getattr(profile, "pip_size", 0.01) or 0.01)
    mid = (float(tick.bid) + float(tick.ask)) / 2.0
    spread_pips = (float(tick.ask) - float(tick.bid)) / pip if pip > 0 else None

    # Resolve the suggestions DB path once so the gate can run dedupe lookups
    # and the post-LLM correlation veto can find open positions.
    db_path: Optional[Path] = None
    try:
        from api.main import _suggestions_db_path
        db_path = _suggestions_db_path(profile_name)
    except Exception:
        db_path = None

    # Fetch upcoming economic events for the event blackout gate (cached 1 hour).
    upcoming: Optional[list[dict[str, Any]]] = None
    if cfg.get("event_blackout_enabled"):
        try:
            from api.ai_trading_chat import get_economic_calendar_events
            upcoming = get_economic_calendar_events(days_ahead=1, limit=5)
        except Exception:
            upcoming = None

    order_book: Optional[dict[str, Any]] = None
    if adapter is not None:
        try:
            from core.book_cache import get_book_cache
            book_cache = get_book_cache()
            book_cache.poll_books(adapter, getattr(profile, "symbol", "USD/JPY"))
            snaps = book_cache.get_order_books(getattr(profile, "symbol", "USD/JPY"))
            if snaps:
                order_book = _summarize_oanda_order_book(snaps[-1].data)
        except Exception:
            order_book = None

    inputs = GateInputs(
        spread_pips=float(spread_pips) if spread_pips is not None else 999.0,
        tick_mid=mid,
        open_ai_trade_count=_count_open_ai_trades(store, profile_name),
        data_by_tf=data_by_tf or {},
        ntz_active=bool(ntz_active),
        suggestions_db_path=db_path,
        upcoming_events=upcoming,
        order_book=order_book,
    )

    state = _load_state(state_path)
    rt = _runtime_block(state)
    _rollover_daily_counters(rt)
    rt["last_tick_utc"] = datetime.now(timezone.utc).isoformat()
    _refresh_autonomous_runtime_from_history(state_path=state_path, rt=rt, cfg=cfg)
    risk_regime = _compute_risk_regime(rt, cfg)

    _clear_expired_no_trade_throttle(rt)

    state["autonomous_fillmore"]["runtime"] = rt
    _save_state(state_path, state)

    decision = evaluate_gate(cfg, rt, inputs, risk_regime=risk_regime)
    state = _load_state(state_path)
    _runtime_block(state).update(rt)
    state["autonomous_fillmore"]["runtime"] = _runtime_block(state)
    _save_state(state_path, state)
    if decision.result != "pass":
        log_decision(state_path, decision)
        return

    # ---- Gate passed: invoke LLM via existing suggest plumbing ----
    mode = str(cfg.get("mode") or "off")
    print(f"[{profile_name}] autonomous Fillmore: gate passed ({cfg.get('aggressiveness')}, {mode}); invoking LLM")

    # Phase 1 guided reasoning: surface recent-fingerprint stats so the LLM
    # can see when this exact thesis has just fired and how it ended. Logged
    # to decision.extras only — never overrides the model's choice.
    fingerprint = (decision.extras or {}).get("thesis_fingerprint")
    if fingerprint:
        try:
            fp_stats = _compute_recent_fingerprint_stats(
                store, profile_name, str(fingerprint), window_minutes=120,
            )
            if decision.extras is None:
                decision.extras = {}
            decision.extras["recent_fires_2h_count"] = fp_stats["count"]
            decision.extras["recent_fires_2h_record"] = (
                f"{fp_stats['wins']}W/{fp_stats['losses']}L"
            )
            decision.extras["recent_fires_2h_net_pips"] = fp_stats["net_pips"]
            decision.extras["last_outcome"] = fp_stats["last_outcome"]
            decision.extras["last_outcome_pips"] = fp_stats["last_pips"]
            decision.extras["minutes_since_last_fire"] = fp_stats["minutes_since_last"]
        except Exception as _e:
            print(f"[{profile_name}] autonomous Fillmore: fingerprint stats error: {_e}")

    # Log pass decisions after Phase 1 enrichment so diagnostics show the same
    # fingerprint/repetition context that the LLM sees.
    log_decision(state_path, decision)

    try:
        suggestions = _invoke_suggest(
            profile,
            profile_name,
            cfg,
            risk_regime=risk_regime,
            gate_decision=decision,
            runtime_snapshot={
                "daily_pnl_usd": rt.get("daily_pnl_usd"),
                "open_ai_trade_count": inputs.open_ai_trade_count,
                "session_peak_pnl_usd": rt.get("session_peak_pnl_usd"),
                "session_peak_pnl_time_utc": rt.get("session_peak_pnl_time_utc"),
                "consecutive_losses": rt.get("consecutive_losses"),
                "consecutive_wins": rt.get("consecutive_wins"),
            },
        )
    except FillmoreLLMCircuitOpenError as e:
        print(f"[{profile_name}] autonomous Fillmore: {e}")
        return
    except Exception as e:
        print(f"[{profile_name}] autonomous Fillmore: LLM error: {e}")
        record_error(state_path, cfg, str(e))
        return

    # Record the call & cost regardless of trade-or-not.
    record_llm_invocation(state_path, str(cfg.get("model") or "gpt-5.4-mini"))

    # Process each suggestion from the LLM (1 in single mode, up to max_suggestions in multi).
    max_lots = float(cfg.get("max_lots_per_trade") or 15.0)
    min_lot_size = float(cfg.get("min_lot_size") or 0.01)
    placed_any = False
    agg_label = str(cfg.get("aggressiveness") or "balanced")

    for suggestion in suggestions:
        # Re-check runtime disarm before each potential placement in case the
        # operator toggled mode/exit-only while an LLM call was in-flight.
        runtime_state = load_execution_state(state_path)
        if runtime_state.kill_switch or runtime_state.exit_system_only:
            print(f"[{profile_name}] autonomous Fillmore: runtime disarmed/exit-only; skipping new placement")
            return
        print(
            f"[{profile_name}] autonomous Fillmore: processing suggestion — "
            f"side={suggestion.get('side')} lots={suggestion.get('lots')} "
            f"quality={suggestion.get('quality')} mode={mode}"
        )

        sug_lots = float(suggestion.get("lots") or 0)
        sug_decision = str(suggestion.get("decision") or "").lower()
        if sug_lots <= 0 or sug_decision == "skip":
            quality = str(suggestion.get("quality") or "C")
            rung = str(suggestion.get("conviction_rung") or "?")
            skip_rsn = str(suggestion.get("skip_reason") or "").strip() or "(no reason given)"
            print(
                f"[{profile_name}] autonomous Fillmore: SKIP "
                f"(decision={sug_decision or 'implicit'} rung={rung} quality={quality}) — {skip_rsn}"
            )
            try:
                log_decision(
                    state_path,
                    GateDecision(
                        timestamp_utc=datetime.now(timezone.utc).isoformat(),
                        result="block", layer="signal",
                        reason=f"llm_skip:{rung}",
                        mode=mode, aggressiveness=agg_label,
                        extras={
                            "decision": sug_decision or "implicit_zero_lots",
                            "conviction_rung": rung,
                            "quality": quality,
                            "skip_reason": skip_rsn,
                            "thesis_fingerprint": suggestion.get("thesis_fingerprint"),
                            "trigger_family": suggestion.get("trigger_family"),
                            "trigger_reason": suggestion.get("trigger_reason"),
                        },
                    ),
                )
            except Exception:
                pass
            continue

        # Book-correlation veto: don't stack a same-side trade near an existing fill.
        if cfg.get("correlation_veto_enabled"):
            veto_dist = float(cfg.get("correlation_distance_pips") or 15.0)
            try:
                sug_side = str(suggestion.get("side") or "").lower()
                sug_price = float(suggestion.get("price") or 0)
            except (TypeError, ValueError):
                sug_side, sug_price = "", 0.0
            prior_open = _correlated_open_position(db_path, sug_side, sug_price, veto_dist)
            if prior_open is not None:
                try:
                    prior_fill = float(prior_open.get("fill_price") or prior_open.get("limit_price") or 0)
                except (TypeError, ValueError):
                    prior_fill = 0.0
                print(
                    f"[{profile_name}] autonomous Fillmore: correlation veto — "
                    f"{sug_side.upper()} @ {sug_price:.3f} blocked, open same-side at {prior_fill:.3f} "
                    f"(within {veto_dist:.1f}p)"
                )
                log_decision(
                    state_path,
                    GateDecision(
                        timestamp_utc=datetime.now(timezone.utc).isoformat(),
                        result="block", layer="signal",
                        reason=f"correlation_veto_within_{veto_dist:.0f}p",
                        mode=mode, aggressiveness=agg_label,
                        extras={
                            "proposed_side": sug_side,
                            "proposed_price": sug_price,
                            "open_fill_price": prior_fill,
                            "open_suggestion_id": prior_open.get("suggestion_id"),
                        },
                    ),
                )
                continue

        # Shadow mode logs but doesn't place.
        if mode == "shadow":
            print(f"[{profile_name}] autonomous Fillmore: SHADOW — would have placed {suggestion.get('side')} @ {suggestion.get('price')}")
            placed_any = True
            continue

        requested_lots = float(suggestion.get("lots") or 0.0)
        scaled_lots = requested_lots * float(risk_regime.get("risk_multiplier") or 1.0)
        scaled_lots = min(max_lots, scaled_lots)
        max_allowed_lots = suggestion.get("max_allowed_lots")
        if max_allowed_lots is not None:
            try:
                scaled_lots = min(scaled_lots, float(max_allowed_lots))
            except (TypeError, ValueError):
                pass
        if scaled_lots < min_lot_size:
            print(
                f"[{profile_name}] autonomous Fillmore: risk regime veto — scaled lots "
                f"{scaled_lots:.4f} below minimum {min_lot_size:.2f}"
            )
            log_decision(
                state_path,
                GateDecision(
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    result="block",
                    layer="hard",
                    reason="risk_regime_lot_veto",
                    mode=mode,
                    aggressiveness=agg_label,
                    extras={
                        "requested_lots": requested_lots,
                        "scaled_lots": scaled_lots,
                        "min_lot_size": min_lot_size,
                        "risk_regime": risk_regime.get("label"),
                    },
                ),
            )
            continue
        suggestion["lots"] = max(min_lot_size, scaled_lots)

        # Safety cap on lots.
        if float(suggestion.get("lots") or 0) > max_lots:
            print(f"[{profile_name}] autonomous Fillmore: clipping lots {suggestion.get('lots')} -> {max_lots}")
            suggestion["lots"] = max_lots
        if float(suggestion.get("lots") or 0) <= 0:
            print(f"[{profile_name}] autonomous Fillmore: invalid lots {suggestion.get('lots')}; skipping")
            continue

        # Paper mode safety check.
        if mode == "paper":
            bt = str(getattr(profile, "broker_type", None) or "mt5").lower()
            if bt == "oanda":
                env = str(getattr(profile, "oanda_environment", None) or "practice").lower()
                if env != "practice":
                    print(
                        f"[{profile_name}] autonomous Fillmore: PAPER mode requires "
                        f"profile oanda_environment='practice' (found {env!r}); skipping place."
                    )
                    continue
        try:
            sug_order_type = str(suggestion.get("order_type") or "market").lower()
            placed = _place_from_suggestion(
                profile, profile_name, state_path, suggestion, sug_order_type, store,
            )
            record_trade_placed(state_path, placed.get("order_id"), suggestion.get("suggestion_id"))
            kind = "MKT" if sug_order_type == "market" else "LMT"
            print(
                f"[{profile_name}] autonomous Fillmore: placed {kind} {suggestion.get('side')} "
                f"{suggestion.get('lots')} @ {suggestion.get('price')} (order {placed.get('order_id')})"
            )
            placed_any = True
        except Exception as e:
            import traceback
            print(f"[{profile_name}] autonomous Fillmore: place error: {e}")
            traceback.print_exc()
            record_broker_reject(state_path)
            record_error(state_path, cfg, str(e))

    if not placed_any and suggestions:
        all_zero = all(float(s.get("lots") or 0) <= 0 for s in suggestions)
        if all_zero:
            record_no_trade_reply(state_path, cfg)
    elif placed_any and mode == "shadow":
        state = _load_state(state_path)
        rt = _runtime_block(state)
        rt["consecutive_no_trade_replies"] = 0
        _save_state(state_path, state)


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\[\{].*?[\]\}])\s*```", re.DOTALL | re.IGNORECASE)


def _extract_json_object(text: str) -> tuple[str, Optional[str]]:
    """Return (json_str, analysis_text).

    The autonomous prompt asks the model to reply with free-text ANALYSIS
    followed by a fenced ```json``` block containing either a single object
    or an array of objects (for multi-trade planning).
    We prefer the last fenced JSON block.
    Fallback: scan for the last balanced ``{ ... }`` or ``[ ... ]`` in the response.
    ``analysis_text`` is whatever preceded the JSON (stripped), for logging.
    """
    if not text:
        raise ValueError("empty LLM response")
    stripped = text.strip()

    matches = list(_JSON_FENCE_RE.finditer(stripped))
    if matches:
        m = matches[-1]
        analysis = stripped[: m.start()].strip() or None
        return m.group(1).strip(), analysis

    # Fallback: find the last balanced top-level JSON object or array by brace scan.
    last_close = max(stripped.rfind("}"), stripped.rfind("]"))
    if last_close < 0:
        raise ValueError("no JSON object found in LLM response")
    close_char = stripped[last_close]
    open_char = "{" if close_char == "}" else "["
    depth = 0
    start = -1
    for i in range(last_close, -1, -1):
        ch = stripped[i]
        if ch == close_char:
            depth += 1
        elif ch == open_char:
            depth -= 1
            if depth == 0:
                start = i
                break
    if start < 0:
        raise ValueError("unbalanced JSON braces in LLM response")
    analysis = stripped[:start].strip() or None
    return stripped[start : last_close + 1].strip(), analysis


def _autonomous_session_profile(ctx: dict[str, Any] | None) -> str:
    session = ((ctx or {}).get("session") or {}) if isinstance(ctx, dict) else {}
    active = [str(x).strip().lower() for x in (session.get("active_sessions") or []) if str(x).strip()]
    overlap = str(session.get("overlap") or "").strip().lower()
    labels = set(active)
    if overlap:
        if "tokyo" in overlap:
            labels.add("tokyo")
        if "london" in overlap:
            labels.add("london")
        if "new york" in overlap or "ny" in overlap:
            labels.add("ny")

    if "london" in labels or "ny" in labels:
        return "trend"
    if labels == {"tokyo"} or "tokyo" in labels:
        return "tokyo"
    return "trend"


def _apply_autonomous_exit_calibration(
    strategy_id: str,
    exit_params: dict[str, Any],
    ctx: dict[str, Any] | None,
) -> dict[str, Any]:
    """Autonomous-only exit calibration.

    Preserve wider runners than the manual suggestion path. Tokyo gets a slightly
    tighter first take-profit; London/NY keep more size on for runner capture.
    """
    if str(strategy_id or "").strip().lower() == "none":
        return dict(exit_params or {})

    calibrated = dict(exit_params or {})
    profile = _autonomous_session_profile(ctx)
    target_tp1 = 5.0 if profile == "tokyo" else 6.0
    target_tp1_close_pct = 50.0 if profile == "tokyo" else 33.0

    if "tp1_pips" in calibrated:
        try:
            calibrated["tp1_pips"] = max(float(calibrated["tp1_pips"]), target_tp1)
        except (TypeError, ValueError):
            calibrated["tp1_pips"] = target_tp1
    if "tp1_close_pct" in calibrated:
        try:
            calibrated["tp1_close_pct"] = min(float(calibrated["tp1_close_pct"]), target_tp1_close_pct)
        except (TypeError, ValueError):
            calibrated["tp1_close_pct"] = target_tp1_close_pct
    return calibrated


def _format_gate_opportunity_block(gate_decision: GateDecision | None) -> str:
    if gate_decision is None:
        return ""
    extras = gate_decision.extras or {}
    lines = [
        "=== GATE-QUALIFIED OPPORTUNITY ===",
        f"Trigger family: {extras.get('trigger_family') or 'unknown'}",
        f"Trigger reason: {extras.get('trigger_reason') or gate_decision.reason}",
    ]
    if extras.get("trigger_bias"):
        lines.append(f"Gate bias: {extras.get('trigger_bias')}")
    if extras.get("trigger_level_label") and extras.get("trigger_level_price") is not None:
        lines.append(
            f"Qualified structure: {extras.get('trigger_level_label')} @ {float(extras.get('trigger_level_price')):.3f}"
        )
    if extras.get("nearest_level_pips") is not None:
        lines.append(f"Nearest structure distance: {extras.get('nearest_level_pips')} pips")
    if extras.get("m5_atr_pips") is not None:
        lines.append(f"M5 ATR: {extras.get('m5_atr_pips')} pips")
    if extras.get("adx_m5") is not None or extras.get("adx_m15") is not None:
        lines.append(f"ADX M5/M15: {extras.get('adx_m5')} / {extras.get('adx_m15')}")
    if extras.get("compression_range_pips") is not None and extras.get("compression_cap_pips") is not None:
        lines.append(
            f"Compression range: {extras.get('compression_range_pips')}p vs cap {extras.get('compression_cap_pips')}p"
        )
    if extras.get("extension_pips") is not None and extras.get("extension_limit_pips") is not None:
        lines.append(
            f"Trend extension: {extras.get('extension_pips')}p vs cap {extras.get('extension_limit_pips')}p"
        )
    if extras.get("trigger_micro_confirmation"):
        lines.append(f"Micro confirmation: {extras.get('trigger_micro_confirmation')}")
    if extras.get("session_range_pips") is not None and extras.get("reward_to_mid_pips") is not None:
        lines.append(
            f"Tokyo range: {extras.get('session_range_pips')}p, reward to mid: {extras.get('reward_to_mid_pips')}p"
        )
    if extras.get("bb_width") is not None:
        lines.append(f"Bollinger width: {extras.get('bb_width')}")

    # Phase 1: thesis fingerprint and recent-fire history.
    if extras.get("thesis_fingerprint"):
        lines.append("")
        lines.append(f"THESIS FINGERPRINT: {extras.get('thesis_fingerprint')}")
        count = extras.get("recent_fires_2h_count")
        if count is not None:
            record = extras.get("recent_fires_2h_record") or "0W/0L"
            net_pips = extras.get("recent_fires_2h_net_pips")
            net_str = f"{net_pips:+.1f}p" if isinstance(net_pips, (int, float)) else "0.0p"
            lines.append(
                f"ZONE MEMORY (this fingerprint, last 2h): {count} attempt(s), {record}, net {net_str}"
            )
            last = extras.get("last_outcome")
            if last:
                last_pips = extras.get("last_outcome_pips")
                mins = extras.get("minutes_since_last_fire")
                pip_str = f"{last_pips:+.1f}p" if isinstance(last_pips, (int, float)) else "?"
                ago_str = f"{mins:.0f}m ago" if isinstance(mins, (int, float)) else "?"
                lines.append(f"LAST OUTCOME: {last} ({pip_str}, {ago_str})")
            try:
                wins = int(str(record).split("W")[0] or 0)
                losses = int(str(record).split("L")[0].split("/")[1] or 0)
            except Exception:
                wins = 0
                losses = 0
            if int(count) >= 1 and wins > losses and isinstance(net_pips, (int, float)) and net_pips > 0:
                lines.append(
                    "WORKING-ZONE NOTE: repetition is not automatically bad. If price is still respecting "
                    "this zone and structure remains clean, another same-zone trade can be valid."
                )
            elif int(count) >= 1 and losses >= 1:
                lines.append(
                    f"FAILING-ZONE NOTE: this thesis has fired {count} time(s) in the last 2h with {losses} loss(es). "
                    "Do not retry it unless price/structure/context has materially changed."
                )
    return "\n".join(lines)


def _apply_autonomous_order_policy(
    suggestion: dict[str, Any],
    ctx: dict[str, Any] | None,
    gate_decision: GateDecision | None,
) -> dict[str, Any]:
    """Constrain market-vs-limit behavior by trigger family."""
    out = dict(suggestion or {})
    extras = (gate_decision.extras or {}) if gate_decision else {}
    family = str(extras.get("trigger_family") or "").strip().lower()
    side = str(out.get("side") or "").strip().lower()
    order_type = str(out.get("order_type") or "market").strip().lower()
    out["order_policy_reason"] = None

    spot = ((ctx or {}).get("spot_price") or {}) if isinstance(ctx, dict) else {}
    try:
        mid = float(spot.get("mid") or 0.0)
        bid = float(spot.get("bid") or 0.0)
        ask = float(spot.get("ask") or 0.0)
    except (TypeError, ValueError):
        mid = bid = ask = 0.0

    if family in {"trend_expansion", "compression_breakout", "momentum_continuation"}:
        if order_type != "market":
            out["requested_order_type"] = order_type
            out["order_type"] = "market"
            out["order_policy_reason"] = (
                "compression_breakout_market_only"
                if family == "compression_breakout"
                else "momentum_continuation_market_only"
                if family == "momentum_continuation"
                else "trend_expansion_market_only"
            )
        if mid > 0:
            out["price"] = mid
        return out

    if family in {"critical_level_reaction", "tight_range_mean_reversion"} and order_type == "limit":
        requested_price = float(out.get("price") or 0.0)
        if requested_price > 0:
            out["requested_price"] = requested_price
        out = _apply_autonomous_limit_price_strategy(out, ctx)
        snap_distance = out.get("snap_distance_pips")
        if isinstance(snap_distance, (int, float)) and snap_distance > 1.5:
            out["requested_order_type"] = "limit"
            out["order_type"] = "market"
            out["order_policy_reason"] = (
                "tokyo_mean_reversion_far_limit_normalized_to_market"
                if family == "tight_range_mean_reversion"
                else "critical_reaction_far_limit_normalized_to_market"
            )
            if mid > 0:
                out["price"] = mid
            return out
        out["order_type"] = "limit"
        out["order_policy_reason"] = (
            "tokyo_mean_reversion_near_touch_limit_ok"
            if family == "tight_range_mean_reversion"
            else "critical_reaction_near_touch_limit_ok"
        )
        return out

    out["order_type"] = "market" if order_type not in ("market", "limit") else order_type
    if out["order_type"] == "limit":
        return _apply_autonomous_limit_price_strategy(out, ctx)
    return out


def _invoke_suggest(
    profile,
    profile_name: str,
    cfg: dict[str, Any],
    risk_regime: Optional[dict[str, Any]] = None,
    gate_decision: GateDecision | None = None,
    runtime_snapshot: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Invoke the LLM for autonomous Fillmore.

    Reason-then-commit flow: the user prompt asks for a plain-text ANALYSIS
    section first, then a fenced ```json``` DECISION block. When multi-trade
    planning is enabled the model may return an array of 0-2 trade objects;
    otherwise a single object. Returns a list of normalized suggestion dicts.

    Lazy imports avoid circular dependency with api.main.
    """
    from api.ai_trading_chat import (
        build_trading_context,
        resolve_ai_suggest_model,
    )
    from api.prompt_builder import PromptBuilder
    from api import suggestion_tracker
    from api.ai_exit_strategies import (
        AI_EXIT_STRATEGIES, DEFAULT_AI_EXIT_STRATEGY,
        exit_strategies_prompt_block, merge_exit_params, normalize_exit_strategy,
    )
    from api.suggestion_schema import ValidationError as SuggestionValidationError
    from api.suggestion_schema import validate_autonomous_suggestion
    import openai
    import json as _json

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not configured")

    ctx = build_trading_context(profile, profile_name)
    model = resolve_ai_suggest_model(cfg.get("model"))
    risk_regime = risk_regime or {
        "label": "normal",
        "risk_multiplier": 1.0,
    }
    runtime_snapshot = runtime_snapshot or {}
    prompt_builder = PromptBuilder.for_autonomous_suggest(
        profile=profile,
        profile_name=profile_name,
        ctx=ctx,
        model=model,
        autonomous_config=cfg,
        risk_regime=risk_regime,
    )
    # Best-effort news enrichment; swallow errors.
    try:
        prompt_builder.append_news_block(
            symbol=getattr(profile, "symbol", "USD_JPY"),
            rss_headline_count=8,
            web_result_count=3,
            parallel_timeout_sec=10.0,
        )
    except Exception:
        pass
    # Learning memory block + today's autonomous run history.
    try:
        from api.main import _suggestions_db_path
        db_path = _suggestions_db_path(profile_name)
        prompt_builder.append_autonomous_memory(
            db_path=db_path,
            risk_regime=risk_regime,
            max_recent_examples=6,
            reflection_limit=8,
            today_limit=10,
        )
    except Exception:
        pass
    gate_block = _format_gate_opportunity_block(gate_decision)

    agg = cfg.get("aggressiveness") or "balanced"
    max_lots = float(cfg.get("max_lots_per_trade") or 15.0)
    mode = str(cfg.get("mode") or "off")
    limit_gtd_min = int(cfg.get("limit_gtd_minutes") or 15)
    exec_note = (
        "ORDER TYPE: You choose — 'market' or 'limit'.\n"
        "  MARKET: fills instantly at current bid/ask. Use when the tape is moving and you want in now.\n"
        f"  LIMIT: use only for near-touch passive entries. The server clamps your limit into a near-market band: "
        f"BUY LIMIT {_AUTONOMOUS_LIMIT_MAX_OFFSET_PIPS:.1f} to {_AUTONOMOUS_LIMIT_MIN_OFFSET_PIPS:.1f} pips below bid, "
        f"SELL LIMIT {_AUTONOMOUS_LIMIT_MIN_OFFSET_PIPS:.1f} to {_AUTONOMOUS_LIMIT_MAX_OFFSET_PIPS:.1f} pips above ask. "
        f"Default expiry {limit_gtd_min} min (GTD). If your structural level is far from current price, prefer MARKET or 0 lots."
    )
    price_instr = (
        "For MARKET: 'price' is informational (current mid is fine). "
        "For LIMIT: give the structural level you care about, but only when price is already near that level. "
        "If your intended level is several pips away, that is not a good autonomous limit candidate."
    )

    paper_note = (
        " Paper mode sends real orders to OANDA practice (real fills on demo funds)."
        if mode == "paper" else ""
    )

    _strategy_ids = ", ".join(f'"{sid}"' for sid in AI_EXIT_STRATEGIES.keys())
    _exit_catalog_text = exit_strategies_prompt_block()
    _exit_profile = _autonomous_session_profile(ctx)
    if _exit_profile == "tokyo":
        exit_calibration_text = (
            "AUTONOMOUS EXIT CALIBRATION: Tokyo/chop profile. Keep TP1 around +5p and do not scale more than 50% "
            "at TP1 unless you explicitly justify it. The point is to leave a runner, not cash most of it at the first pop."
        )
    else:
        exit_calibration_text = (
            "AUTONOMOUS EXIT CALIBRATION: London/NY trend profile. TP1 should usually be around +6p and you should "
            "scale no more than 33% at TP1 unless you explicitly justify a tighter or heavier first take."
        )

    multi_enabled = bool(cfg.get("multi_trade_enabled"))
    max_suggestions = int(cfg.get("max_suggestions_per_call") or 2) if multi_enabled else 1

    if multi_enabled:
        multi_note = (
            f"You may propose 0 to {max_suggestions} trade ideas. If the tape has two distinct, uncorrelated setups "
            f"at different levels/sides you may return an array of {max_suggestions} objects. If only one setup is clear, "
            "return a single object. If nothing is worth taking, return a single object with lots=0."
        )
        decision_format = (
            f"2. DECISION (single fenced JSON code block containing one object OR an array of up to {max_suggestions} objects).\n"
        )
    else:
        multi_note = ""
        decision_format = "2. DECISION (single fenced JSON code block). Use this exact schema:\n"

    daily_pnl_usd = float(runtime_snapshot.get("daily_pnl_usd") or 0.0)
    open_ai_trade_count = int(runtime_snapshot.get("open_ai_trade_count") or 0)
    session_peak_pnl = runtime_snapshot.get("session_peak_pnl_usd")
    session_peak_time = runtime_snapshot.get("session_peak_pnl_time_utc")
    consecutive_losses = int(runtime_snapshot.get("consecutive_losses") or 0)
    live_spread_pips = float((ctx.get("spot_price") or {}).get("spread_pips") or 0.0)

    # Phase 1: structured SESSION STATE block exposes drawdown-from-peak so the
    # model can reason about cumulative session pain. Pure context — never
    # used by code to gate or alter the LLM's output.
    peak_value = float(session_peak_pnl) if session_peak_pnl is not None else daily_pnl_usd
    drawdown_from_peak = max(0.0, peak_value - daily_pnl_usd)
    peak_time_str = "n/a"
    if session_peak_time:
        try:
            peak_dt = datetime.fromisoformat(str(session_peak_time).replace("Z", "+00:00"))
            peak_time_str = peak_dt.strftime("%H:%M UTC")
        except Exception:
            peak_time_str = str(session_peak_time)
    context_note = (
        "SESSION STATE:\n"
        f"  realized P&L today: ${daily_pnl_usd:+.2f}\n"
        f"  peak P&L today: ${peak_value:+.2f} at {peak_time_str}\n"
        f"  drawdown from peak: ${drawdown_from_peak:.2f}\n"
        f"  open AI trades: {open_ai_trade_count}\n"
        f"  consecutive losses: {consecutive_losses}\n"
        "There is no server-side daily-loss veto. The bar to trade naturally rises with session pain — "
        "as realized P&L falls and drawdown from peak grows, the evidence required to justify a new trade "
        "should grow with it. Use your judgment about how much the day's path should raise your selectivity.\n"
        "PHASE 4 SELECTIVITY: the last Phase 3 sample placed 76.8% of calls and still lost because losers "
        "were too large relative to winners. If the setup is not clearly above the base rate, skip."
    )
    spread_note = (
        f"SPREAD CONTEXT: live OANDA spread is {live_spread_pips:.1f} pips. "
        "A good tradable range is roughly 1.6-2.9 pips. "
        "Inside that range, spread is normal context. Much tighter is unusually favorable. "
        "Much wider should reduce conviction, worsen entry quality, or push you to pass."
    )

    base_lots = float(cfg.get("base_lot_size") or 5.0)
    lot_dev = float(cfg.get("lot_deviation") or 4.0)
    lot_lo = max(1, int(base_lots - lot_dev))
    lot_hi = min(int(max_lots), int(base_lots + lot_dev))
    lot_mid = int(base_lots)

    # Phase 1: conditional required fields. whats_different is required when this
    # exact thesis fingerprint has fired in the last 2h; why_not_stop is required
    # when realized P&L today is negative. Both are forcing-function fields —
    # logged either way, never used by code to override the model's decision.
    fp_extras = (gate_decision.extras or {}) if gate_decision else {}
    recent_fires = int(fp_extras.get("recent_fires_2h_count") or 0)
    needs_whats_different = recent_fires >= 1
    needs_why_not_stop = daily_pnl_usd < 0.0

    conditional_field_lines = []
    if needs_whats_different:
        conditional_field_lines.append(
            '  "whats_different": "<one specific sentence on what is materially '
            'different about price/structure/context vs. recent fires of this same fingerprint>",'
        )
    if needs_why_not_stop:
        conditional_field_lines.append(
            '  "why_not_stop": "<one specific sentence on why this trade is better '
            'than stopping for the day, given the current session drawdown>",'
        )
    conditional_block = ("\n" + "\n".join(conditional_field_lines)) if conditional_field_lines else ""

    forcing_function_notes = []
    if needs_whats_different:
        forcing_function_notes.append(
            "WHAT'S DIFFERENT: this thesis fingerprint has fired recently. If you decide to trade, "
            "classify whether this is a working zone, failing zone, unresolved chop, or fresh setup. "
            "Repeating a working zone can be valid; retrying a failing/unresolved zone needs a material change "
            "(new structural break, new HTF shift, new catalyst, fresh momentum leg). Generic 'conditions look "
            "better' answers tell you the trade is a skip."
        )
    if needs_why_not_stop:
        forcing_function_notes.append(
            "WHY NOT STOP: realized P&L today is negative. If you decide to trade, answer specifically "
            "why this trade beats stopping for the day. Generic optimism tells you the trade is a skip."
        )
    forcing_function_block = ("\n\n" + "\n\n".join(forcing_function_notes)) if forcing_function_notes else ""

    suggest_prompt = (
        f"=== AUTONOMOUS DECISION REQUEST ===\n"
        f"Gate mode: {agg}. Execution mode: {mode}.{paper_note}\n"
        + (f"\n{multi_note}\n" if multi_note else "") +
        "\nGATE OPPORTUNITY\n"
        + (f"{gate_block}\n" if gate_block else "(no gate metadata)\n") +
        f"\n{context_note}\n"
        f"{spread_note}\n"
        f"\nEXECUTION\n{exec_note}\n"
        f"LOT RANGE: {lot_lo}-{lot_hi} (base {lot_mid}). Hard ceiling: {int(max_lots)}.\n"
        f"{exit_calibration_text}\n"
        f"{forcing_function_block}\n"
        "\n"
        "=== ANALYSIS CHECKLIST ===\n"
        "Write 6-12 lines. Cover each in order. If a check does not apply, say so.\n"
        "  1. GATE FIT — Is the gated opportunity actually present right now or has it degraded? "
        "Name the level you would anchor on.\n"
        "  2. TREND CONSENSUS — H1/M15/M5/M1. Classify as aligned, mixed, or countertrend. "
        "If mixed: name the specific catalyst that makes this still tradeable, or skip.\n"
        "  3. SIDE BIAS CHECK — If side=sell: state explicitly why this beats the buy-side base "
        "rate (29% of wins, 53% of losses). 'Resistance reject' alone is not enough. If side=sell "
        "AND any weakness signal is present, the named_catalyst must be specific or skip.\n"
        "  4. POLICY + GEOPOLITICAL — MOF/FM signaling, intervention risk, coordination, "
        "war-premium. CONFIRMS / CONTRADICTS / MIXED.\n"
        "  5. JPY CROSS BIAS — confirm or contradict.\n"
        "  6. PRICE STRUCTURE — nearest level(s), order-book clusters.\n"
        "  7. ATR + SPREAD — does ATR justify your SL? Is spread fine, favorable, or expensive?\n"
        "  8. ZONE MEMORY + REPEAT — fresh_setup / working_zone / failing_zone / unresolved_chop. "
        "Compare against your last same-fingerprint fires.\n"
        "  9. OPEN BOOK — stacking/hedging — additive alpha or duplicate?\n"
        " 10. PLANNED GEOMETRY — entry/SL/TP, R:R. If R:R lands in [1.3, 2.0] because you widened "
        "TP after a marginal entry, the entry is wrong; either re-anchor or skip.\n"
        " 11. WEAKNESS WORDS — if you used probe/hedge/tactical/additive/reduced-conviction "
        "anywhere in your reasoning, state why this trade is worth taking despite that. If you cannot, skip.\n"
        " 12. REASONING QUALITY GATE — name the setup_location, edge_reason, adverse_context, "
        "caveat_resolution, and micro_confirmation_event. A level is location, not edge. "
        "If you admit contradiction/mixed tape/weak JPY/probe/thin/tactical logic, the caveat_resolution "
        "must be material or you skip. Micro confirmation must name the actual M1/M3/M5 event.\n"
        " 13. PHASE 5 SIZE CHECK — large lots require aligned, no weakness, a green-pattern match, and "
        "catalyst_score >= 2. Sell+weakness is max 1 lot unless catalyst is material. "
        "critical_level_reaction+mixed is skip by default; only a material caveat can allow max 1 lot.\n"
        " 14. CALL — trade or skip. If trade: side, size, exit strategy.\n"
        "\n"
        + decision_format +
        "```json\n"
        "{\n"
        '  "decision": "trade" | "skip",\n'
        '  "skip_reason": "<one specific sentence — required if decision is skip>",\n'
        '  "trade_thesis": "<one specific sentence naming the catalyst, not the level — required if decision is trade>",\n'
        '  "named_catalyst": "<the specific event/structure/flow that makes this trade beat the base rate; not \'level reject\' alone — required if decision is trade>",\n'
        '  "side_bias_check": "<required if side=\'sell\': one sentence on why this beats the buy-side base rate, else null>",\n'
        '  "setup_location": "<the level/zone being traded; location only, not the edge>",\n'
        '  "edge_reason": "<the specific reason this setup should beat the base rate; required if decision is trade>",\n'
        '  "adverse_context": "<contradicting evidence you noticed, or null if none>",\n'
        '  "caveat_resolution": "<material reason adverse_context is overcome, else null; generic structure is not enough>",\n'
        '  "micro_confirmation_event": "<specific M1/M3/M5 event such as close/retest/sweep/HL/LH, else null>",\n'
        '  "reasoning_quality_gate": "pass" | "cap_to_1_lot" | "skip",\n'
        '  "conviction_rung": "A" | "B" | "C" | "D",'
        + conditional_block + "\n"
        '  "zone_memory_read": "fresh_setup" | "working_zone" | "failing_zone" | "unresolved_chop",\n'
        '  "repeat_trade_case": "none" | "same_zone_continuation" | "retest_after_success" | "material_change_after_failure" | "blind_retry",\n'
        '  "planned_rr_estimate": <number or null>,\n'
        '  "low_rr_edge": "<required if planned_rr_estimate < 1.0, else null>",\n'
        '  "timeframe_alignment": "aligned" | "mixed" | "countertrend",\n'
        '  "countertrend_edge": "<required if timeframe_alignment is countertrend or mixed, else null>",\n'
        '  "trigger_fit": "true_escape" | "momentum_continuation" | "micro_expansion_inside_chop" | "level_reaction" | "mean_reversion" | "unclear",\n'
        '  "why_trade_despite_weakness": "<required if using hedge/probe/tactical/additive/reduced-conviction logic, else null>",\n'
        '  "order_type": "market" | "limit",\n'
        '  "side": "buy" | "sell",\n'
        '  "price": <entry price as number>,\n'
        '  "sl": <stop loss price as number>,\n'
        '  "tp": <take profit price as number>,\n'
        f'  "lots": <0 to skip, or {lot_lo}-{lot_hi} based on conviction>,\n'
        '  "time_in_force": "GTC" | "GTD",\n'
        '  "gtd_time_utc": <ISO datetime if GTD, else null>,\n'
        '  "exit_strategy": one of [' + _strategy_ids + '],\n'
        '  "exit_params": {<optional numeric overrides>} or null,\n'
        '  "exit_plan": "<1-2 sentence custom exit logic for the thesis monitor>",\n'
        '  "custom_exit_plan": {\n'
        '    "first_profit_objective_pips": <number or null>,\n'
        '    "partial_close_pct": <number or null>,\n'
        '    "breakeven_trigger_pips": <number or null>,\n'
        '    "invalidation_conditions": "<specific conditions that kill the thesis>",\n'
        '    "trail_preference": "<hwm | m1_ema | m5_ema | price_action | none plus details>",\n'
        '    "time_stop_minutes": <number or null>,\n'
        '    "early_exit_if": "<what would make you exit early>",\n'
        '    "runner_mode": <true if momentum/trend runner, else false>,\n'
        '    "runner_hold_rule": "<what must stay true to keep trailing the runner>"\n'
        '  },\n'
        '  "rationale": "<1-2 sentence concise summary>",\n'
        '  "quality": "A" | "B" | "C"\n'
        "}\n"
        "```\n"
        "\n"
        "=== CONVICTION RUNG (self-assessment, logged) ===\n"
        "  A — fresh structural event or clearly working zone, multi-TF alignment, clean invalidation, "
        "R:R 1.0-1.3, no recent same-fingerprint failure, matches a green pattern.\n"
        "  B — two-TF alignment, R:R 1.0-1.3, no recent same-fingerprint failure, OR working-zone "
        "continuation with one soft element.\n"
        "  C — marginal but defensible. Smallest size, often skip.\n"
        "  D — probe/hedge/tactical/additive/reduced conviction. Pairs with skip unless the "
        "named_catalyst is unusually strong.\n"
        f"Lots and rung are independent: a B setup can size to {lot_lo} or {lot_hi} based on green/red "
        "pattern fit. A rung A setup with a red-pattern match (e.g., sell + weakness signal without a "
        f"named catalyst) downgrades to C and either sizes to {lot_lo} or skips.\n"
        "\n"
        "ENTRY PRICE RULE: " + price_instr + "\n"
        f'EXIT STRATEGY: default "{DEFAULT_AI_EXIT_STRATEGY}" unless analysis favors another, "llm_custom_exit", or "none".\n'
        "EXIT PLAN: 1-2 sentences for the thesis monitor (runs every ~3 min for default strategies; "
        "event-driven and budgeted for llm_custom_exit). It can only hold, tighten SL, scale out, or exit. "
        "Give crisp conditions: 'exit if M3 flips bear', 'tighten SL to BE after +5p', "
        "'scale 50% at TP1 then trail on M1 21 EMA'. Otherwise write \"default\".\n"
        "RUNNER MANAGEMENT: If trigger_fit or gate family is momentum_continuation, prefer "
        "exit_strategy='llm_custom_exit' with runner_mode=true: partial at first objective, then trail "
        "while M1/M5 structure and EMA9/21 hold. Do not cap a clean runner with a tiny fixed TP.\n"
        "\n"
        "LOT SIZING — selectivity, not size, is the lever:\n"
        f"  {lot_hi}: cleanest aligned setup matching a green pattern.\n"
        f"  {lot_mid}: solid setup with one soft element.\n"
        f"  {lot_lo}: thin but defensible.\n"
        "  0:    marginal — pair with decision='skip' and a specific skip_reason.\n"
        "Do not size up to compensate for low conviction. The gate says 'look here', not 'trade now'.\n"
        "SERVER SIZE / REASONING BACKSTOPS:\n"
        "  - lots >= 8 are allowed only for aligned, no-weakness setups with a green-pattern match "
        "and catalyst_score >= 2.\n"
        f"  - sell + any weakness without a material catalyst is capped to {lot_lo} lot.\n"
        "  - contradiction admitted without a material caveat_resolution is skipped; with a material "
        f"resolution, it is capped to {lot_lo} lot.\n"
        "  - critical_level_reaction + mixed alignment is skipped unless caveat_resolution is material; "
        f"material exceptions are capped to {lot_lo} lot.\n"
        "  - vague micro confirmation without a specific M1/M3/M5 event is skipped unless a material "
        f"edge exists; material exceptions are capped to {lot_lo} lot.\n"
        "  - if recent placement rate is high, prefer skip over lot_hi unless the edge is obvious.\n"
        "\n"
        "QUALITY TAG (logged, does NOT affect placement):\n"
        '  A = textbook. B = solid with one soft element. C = thin or skip.\n'
        "\n"
        + _exit_catalog_text
    )

    client = openai.OpenAI()
    assembly = prompt_builder.build(
        user=suggest_prompt,
        prompt_version=AUTONOMOUS_PROMPT_VERSION,
    )
    resp = run_guarded_fillmore_llm_call(
        "autonomous_suggest",
        lambda: client.chat.completions.create(
            model=assembly.model,
            messages=[
                {"role": "system", "content": assembly.system},
                {"role": "user", "content": assembly.user},
            ],
        ),
    )
    raw = (resp.choices[0].message.content or "").strip()
    try:
        json_str, analysis_text = _extract_json_object(raw)
        parsed = _json.loads(json_str)
    except Exception as e:
        cleaned = raw
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[: cleaned.rfind("```")]
        try:
            parsed = _json.loads(cleaned.strip())
            analysis_text = None
        except Exception:
            raise ValueError(f"failed to parse suggestion JSON: {e}; raw[:300]={raw[:300]!r}") from e

    # Normalize to a list of suggestions (single object → list of one).
    raw_suggestions: list[dict[str, Any]] = []
    if isinstance(parsed, list):
        raw_suggestions = [s for s in parsed if isinstance(s, dict)]
    elif isinstance(parsed, dict):
        raw_suggestions = [parsed]

    if not raw_suggestions:
        raw_suggestions = [{"lots": 0, "quality": "C", "rationale": "empty response", "side": "buy", "price": 0}]

    suggestions_out: list[dict[str, Any]] = []
    for suggestion in raw_suggestions[:max_suggestions]:
        # Preserve the analysis text alongside the rationale.
        if analysis_text:
            existing_rat = str(suggestion.get("rationale") or "").strip()
            if existing_rat:
                suggestion["rationale"] = f"{existing_rat}\n\nANALYSIS:\n{analysis_text}"
            else:
                suggestion["rationale"] = f"ANALYSIS:\n{analysis_text}"

        # Normalize fields.
        suggestion["side"] = str(suggestion.get("side") or "buy").lower()
        sug_order_type = str(suggestion.get("order_type") or "market").lower()
        if sug_order_type not in ("market", "limit"):
            sug_order_type = "market"
        suggestion["order_type"] = sug_order_type
        suggestion["price"] = float(suggestion.get("price") or 0)
        suggestion["sl"] = float(suggestion.get("sl") or 0)
        suggestion["tp"] = float(suggestion.get("tp") or 0)
        raw_lots = suggestion.get("lots")
        suggestion["lots"] = float(raw_lots) if raw_lots is not None and raw_lots != "" else 0.0
        suggestion.pop("confidence", None)
        quality = str(suggestion.get("quality") or "C").upper()
        if quality not in ("A", "B", "C"):
            quality = "C"
        suggestion["quality"] = quality

        # Phase 1 guided-reasoning fields. Decision is first-class trade/skip;
        # conviction_rung is self-assessment; whats_different / why_not_stop are
        # forcing-function fields conditionally required by prompt context.
        # We generally preserve the model's choice, but apply targeted server
        # vetoes for known weak-pattern regressions.
        decision_field = str(suggestion.get("decision") or "").strip().lower()
        if decision_field not in ("trade", "skip"):
            decision_field = "trade" if float(suggestion.get("lots") or 0) > 0 else "skip"
        suggestion["decision"] = decision_field

        rung = str(suggestion.get("conviction_rung") or "").strip().upper()
        if rung not in ("A", "B", "C", "D"):
            rung = "D" if decision_field == "skip" else "C"
        suggestion["conviction_rung"] = rung

        suggestion["skip_reason"] = str(suggestion.get("skip_reason") or "").strip() or None
        suggestion["trade_thesis"] = str(suggestion.get("trade_thesis") or "").strip() or None
        suggestion["whats_different"] = str(suggestion.get("whats_different") or "").strip() or None
        suggestion["why_not_stop"] = str(suggestion.get("why_not_stop") or "").strip() or None
        suggestion["zone_memory_read"] = str(suggestion.get("zone_memory_read") or "fresh_setup").strip().lower()
        if suggestion["zone_memory_read"] not in {"fresh_setup", "working_zone", "failing_zone", "unresolved_chop"}:
            suggestion["zone_memory_read"] = "fresh_setup"
        suggestion["repeat_trade_case"] = str(suggestion.get("repeat_trade_case") or "none").strip().lower()
        if suggestion["repeat_trade_case"] not in {
            "none",
            "same_zone_continuation",
            "retest_after_success",
            "material_change_after_failure",
            "blind_retry",
        }:
            suggestion["repeat_trade_case"] = "none"

        def _optional_text(key: str) -> None:
            suggestion[key] = str(suggestion.get(key) or "").strip() or None

        for _k in (
            "low_rr_edge",
            "timeframe_alignment",
            "countertrend_edge",
            "trigger_fit",
            "why_trade_despite_weakness",
            "named_catalyst",
            "side_bias_check",
            "setup_location",
            "edge_reason",
            "adverse_context",
            "caveat_resolution",
            "micro_confirmation_event",
            "reasoning_quality_gate",
        ):
            _optional_text(_k)
        if suggestion["timeframe_alignment"] not in {"aligned", "mixed", "countertrend", None}:
            suggestion["timeframe_alignment"] = "mixed"
        if suggestion["trigger_fit"] not in {
            "true_escape",
            "momentum_continuation",
            "micro_expansion_inside_chop",
            "level_reaction",
            "mean_reversion",
            "unclear",
            None,
        }:
            suggestion["trigger_fit"] = "unclear"
        try:
            suggestion["planned_rr_estimate"] = float(suggestion.get("planned_rr_estimate"))
        except (TypeError, ValueError):
            risk = abs(float(suggestion.get("price") or 0.0) - float(suggestion.get("sl") or 0.0))
            reward = abs(float(suggestion.get("tp") or 0.0) - float(suggestion.get("price") or 0.0))
            suggestion["planned_rr_estimate"] = round(reward / risk, 3) if risk > 0 else None

        suggestion["trigger_family"] = fp_extras.get("trigger_family") or suggestion.get("trigger_family")
        suggestion["trigger_reason"] = fp_extras.get("trigger_reason") or suggestion.get("trigger_reason")
        suggestion["trigger_bias"] = fp_extras.get("trigger_bias") or suggestion.get("trigger_bias")
        suggestion["thesis_fingerprint"] = fp_extras.get("thesis_fingerprint") or suggestion.get("thesis_fingerprint")
        suggestion["phase4_catalyst_score"] = _phase4_catalyst_score(suggestion.get("named_catalyst"))
        suggestion["phase4_green_matches"] = _phase4_green_pattern_matches(suggestion, ctx)
        suggestion["phase4_weakness_signals"] = _phase4_weakness_signals(suggestion)
        suggestion["phase5_reasoning_flags"] = _phase4_reasoning_flags(suggestion)
        suggestion["phase5_material_resolution_score"] = _phase4_material_resolution_score(suggestion)

        # Binding skip discipline: when the model self-identifies a known weak
        # setup pattern, convert trade -> skip server-side so it cannot execute.
        veto_reason: Optional[str] = None
        planned_rr = suggestion.get("planned_rr_estimate")
        recent_fires_2h = 0
        if gate_decision and isinstance(gate_decision.extras, dict):
            try:
                recent_fires_2h = int(gate_decision.extras.get("recent_fires_2h_count") or 0)
            except (TypeError, ValueError):
                recent_fires_2h = 0
        if decision_field == "trade":
            if (
                suggestion.get("zone_memory_read") == "failing_zone"
                and suggestion.get("repeat_trade_case") == "blind_retry"
            ):
                veto_reason = "server_veto:failing_zone_blind_retry"
            elif (
                suggestion.get("conviction_rung") == "D"
                and planned_rr is not None
                and float(planned_rr) < 1.0
            ):
                veto_reason = "server_veto:rung_d_sub_1_rr"
            elif (
                suggestion.get("zone_memory_read") == "unresolved_chop"
                and planned_rr is not None
                and float(planned_rr) < 1.1
            ):
                veto_reason = "server_veto:unresolved_chop_low_rr"
            elif (
                suggestion.get("trigger_fit") in {"unclear", "micro_expansion_inside_chop"}
                and planned_rr is not None
                and float(planned_rr) < 1.0
            ):
                veto_reason = "server_veto:unclear_trigger_low_rr"
            elif (
                suggestion.get("repeat_trade_case") == "blind_retry"
                and recent_fires_2h >= 1
                and (suggestion.get("zone_memory_read") in {"working_zone", "failing_zone", "unresolved_chop"})
            ):
                veto_reason = "server_veto:repeat_fire_without_fresh_edge"
            else:
                # House-edge rules from the prompt rewrite. A named_catalyst is
                # "generic" when missing, blank, shorter than 12 chars, or made
                # entirely of common filler words. The bar is deliberately low:
                # the model only needs one specific phrase.
                named_catalyst = (suggestion.get("named_catalyst") or "").strip()
                catalyst_is_generic = (
                    not named_catalyst
                    or len(named_catalyst) < 12
                    or named_catalyst.lower() in _PHASE4_GENERIC_CATALYSTS
                )
                if (
                    suggestion.get("timeframe_alignment") == "mixed"
                    and catalyst_is_generic
                ):
                    veto_reason = "server_veto:mixed_alignment_no_catalyst"
                else:
                    side = str(suggestion.get("side") or "").lower()
                    rr = suggestion.get("planned_rr_estimate")
                    rr_low = isinstance(rr, (int, float)) and float(rr) < 1.0
                    has_weakness = (
                        suggestion.get("timeframe_alignment") in {"mixed", "countertrend"}
                        or suggestion.get("repeat_trade_case") == "blind_retry"
                        or suggestion.get("zone_memory_read") in {"failing_zone", "unresolved_chop"}
                        or rr_low
                        or _phase4_has_meaningful_text(suggestion.get("why_trade_despite_weakness"))
                    )
                    if side == "sell" and has_weakness and catalyst_is_generic:
                        veto_reason = "server_veto:sell_with_weakness_no_catalyst"
                    else:
                        veto_reason = _phase4_reasoning_veto_reason(suggestion)
        if veto_reason:
            decision_field = "skip"
            suggestion["decision"] = "skip"
            suggestion["skip_reason"] = veto_reason

        _phase4_apply_selectivity_sizing(suggestion, ctx)

        custom_plan = suggestion.get("custom_exit_plan")
        if not isinstance(custom_plan, dict):
            custom_plan = {}
        suggestion["custom_exit_plan"] = custom_plan

        # If the model chose skip, force lots to 0 so the existing skip path
        # at the placement loop catches it cleanly.
        if decision_field == "skip":
            suggestion["lots"] = 0.0
        if sug_order_type == "limit":
            suggestion["time_in_force"] = "GTD"
            suggestion["gtd_time_utc"] = (
                (datetime.now(timezone.utc) + timedelta(minutes=limit_gtd_min))
                .replace(microsecond=0).isoformat().replace("+00:00", "Z")
            )
        else:
            tif = str(suggestion.get("time_in_force") or "GTC").upper()
            if tif not in ("GTC", "GTD"):
                tif = "GTC"
            suggestion["time_in_force"] = tif
            if tif == "GTD" and not suggestion.get("gtd_time_utc"):
                suggestion["gtd_time_utc"] = (
                    (datetime.now(timezone.utc) + timedelta(minutes=60))
                    .replace(microsecond=0).isoformat().replace("+00:00", "Z")
                )
        suggestion = _apply_autonomous_order_policy(suggestion, ctx, gate_decision)
        if suggestion.get("order_type") == "limit":
            suggestion["time_in_force"] = "GTD"
            suggestion["gtd_time_utc"] = (
                (datetime.now(timezone.utc) + timedelta(minutes=limit_gtd_min))
                .replace(microsecond=0).isoformat().replace("+00:00", "Z")
            )
        suggestion["trigger_family"] = fp_extras.get("trigger_family") or suggestion.get("trigger_family")
        suggestion["trigger_reason"] = fp_extras.get("trigger_reason") or suggestion.get("trigger_reason")
        suggestion["trigger_bias"] = fp_extras.get("trigger_bias") or suggestion.get("trigger_bias")
        suggestion["thesis_fingerprint"] = fp_extras.get("thesis_fingerprint") or suggestion.get("thesis_fingerprint")
        # exit_plan: free-text field for the thesis monitor to reason about mid-trade.
        suggestion["exit_plan"] = str(suggestion.get("exit_plan") or "default").strip()
        suggestion["prompt_version"] = assembly.prompt_version
        suggestion["prompt_hash"] = assembly.prompt_hash

        is_runner_trigger = (
            suggestion.get("trigger_family") == "momentum_continuation"
            or suggestion.get("trigger_fit") == "momentum_continuation"
        )
        raw_exit = suggestion.get("exit_strategy")
        raw_exit_s = str(raw_exit).strip().lower() if raw_exit is not None else ""
        if raw_exit_s == "none":
            suggestion["exit_strategy"] = "none"
            suggestion["exit_params"] = {}
        else:
            chosen = normalize_exit_strategy(raw_exit_s) if raw_exit_s in AI_EXIT_STRATEGIES else DEFAULT_AI_EXIT_STRATEGY
            if is_runner_trigger:
                chosen = "llm_custom_exit"
            suggestion["exit_strategy"] = chosen
            raw_params = suggestion.get("exit_params") if isinstance(suggestion.get("exit_params"), dict) else None
            merged_params = merge_exit_params(chosen, raw_params)
            suggestion["exit_params"] = _apply_autonomous_exit_calibration(chosen, merged_params, ctx)
            if chosen == "llm_custom_exit":
                plan = suggestion.get("custom_exit_plan") if isinstance(suggestion.get("custom_exit_plan"), dict) else {}
                if is_runner_trigger:
                    plan = dict(plan)
                    plan["runner_mode"] = True
                    plan.setdefault("first_profit_objective_pips", suggestion["exit_params"].get("tp1_pips"))
                    plan.setdefault("partial_close_pct", suggestion["exit_params"].get("tp1_close_pct"))
                    plan.setdefault("breakeven_trigger_pips", suggestion["exit_params"].get("tp1_pips"))
                    plan.setdefault(
                        "trail_preference",
                        "scale partial at the first objective, then trail the runner behind M1/M5 swing structure or EMA21 hold while trying to capture a 20+ pip move when the path stays open",
                    )
                    plan.setdefault(
                        "runner_hold_rule",
                        "hold the remainder for a possible 20+ pip runner while M1/M5 remain aligned, price respects EMA9/21, and no major level blocks the path",
                    )
                    plan.setdefault(
                        "invalidation_conditions",
                        "exit or tighten aggressively if the run loses M1/M5 alignment, breaks EMA21 acceptance, or prints a lower high/lower low against the trade",
                    )
                    plan.setdefault(
                        "early_exit_if",
                        "momentum acceptance fails after entry or the pullback becomes a full structure break instead of a retest",
                    )
                    plan.setdefault("time_stop_minutes", 20)
                if not plan:
                    plan = {
                        "first_profit_objective_pips": suggestion["exit_params"].get("tp1_pips"),
                        "partial_close_pct": suggestion["exit_params"].get("tp1_close_pct"),
                        "breakeven_trigger_pips": suggestion["exit_params"].get("tp1_pips"),
                        "invalidation_conditions": suggestion.get("exit_plan") or "exit if the original thesis is materially invalidated",
                        "trail_preference": "discretionary price action",
                        "time_stop_minutes": None,
                        "early_exit_if": suggestion.get("exit_plan") or "momentum or zone behavior flips against the thesis",
                        "runner_mode": False,
                        "runner_hold_rule": None,
                    }
                suggestion["custom_exit_plan"] = plan
        suggestion["model_used"] = model
        suggestion["entry_type"] = suggestion_tracker.ENTRY_TYPE_FILLMORE_AUTONOMOUS
        try:
            suggestion = validate_autonomous_suggestion(suggestion)
        except SuggestionValidationError as e:
            print(f"[{profile_name}] autonomous Fillmore: suggestion schema validation failed: {e}")
            continue

        # Persist the suggestion row.
        try:
            from api.main import _suggestions_db_path
            sid = suggestion_tracker.log_generated(
                _suggestions_db_path(profile_name),
                profile=profile_name,
                model=model,
                suggestion=suggestion,
                ctx=ctx,
            )
            suggestion["suggestion_id"] = sid
        except Exception as e:
            import traceback
            print(f"[{profile_name}] autonomous Fillmore: log_generated failed: {e}")
            traceback.print_exc()
        suggestions_out.append(suggestion)

    return suggestions_out


def _place_from_suggestion(
    profile, profile_name: str, state_path: Path, suggestion: dict[str, Any],
    order_type: str, store,
) -> dict[str, Any]:
    """Place the order (market or limit) + register managed exit for the run loop.

    Used for both Autonomous ``live`` and ``paper`` modes: orders always go through
    the broker adapter (OANDA practice when profile ``oanda_environment`` is
    ``practice``). ``paper`` mode is a product toggle only; it does not bypass the broker.
    """
    from adapters.broker import get_adapter
    from api.ai_exit_strategies import (
        merge_exit_params, normalize_exit_strategy, trail_mode_for_strategy,
    )
    from api import suggestion_tracker
    import json as _json

    side = str(suggestion.get("side") or "buy").lower()
    price = float(suggestion.get("price") or 0)
    lots = float(suggestion.get("lots") or 0)
    sl = suggestion.get("sl")
    tp = suggestion.get("tp")
    tif = str(suggestion.get("time_in_force") or "GTD").upper()
    gtd = suggestion.get("gtd_time_utc")

    exit_strat = (suggestion.get("exit_strategy") or "").strip().lower()
    managed_strategy: Optional[str] = None
    managed_params: dict[str, Any] = {}
    if exit_strat and exit_strat != "none":
        managed_strategy = normalize_exit_strategy(exit_strat)
        managed_params = merge_exit_params(managed_strategy, suggestion.get("exit_params"))

    def _resolve_position_id() -> Any:
        position_id = None
        deal_id = getattr(result, "deal", None)
        order_id = getattr(result, "order", None)
        try:
            if deal_id:
                position_id = adapter.get_position_id_from_deal(deal_id)
        except Exception:
            position_id = None
        if position_id is None and order_id:
            try:
                position_id = adapter.get_position_id_from_order(order_id)
            except Exception:
                position_id = None
        if position_id is None and order_id and getattr(profile, "broker_type", None) == "oanda":
            for _ in range(3):
                try:
                    time.sleep(0.5)
                    position_id = adapter.get_position_id_from_order(order_id)
                    if position_id is not None:
                        break
                except Exception:
                    position_id = None
        return position_id

    adapter = get_adapter(profile)
    adapter.initialize()
    try:
        if order_type == "market":
            # ----- Market path -----
            result = adapter.order_send_market(
                symbol=profile.symbol,
                side=side,
                volume_lots=lots,
                sl=float(sl) if sl is not None else None,
                tp=float(tp) if tp is not None else None,
                comment=f"autonomous_fillmore:{profile_name}",
            )
            if result.retcode != 0:
                raise RuntimeError(f"market order rejected: {result.comment}")

            fill_price = float(getattr(result, "fill_price", None) or price or 0.0)
            position_id = _resolve_position_id()
            order_id = getattr(result, "order", None)
            trade_id: str | None = None

            # Insert trade row directly so local accounting + later sync can
            # see every autonomous market order, even when exit_strategy='none'.
            if position_id is not None:
                try:
                    trail_mode = trail_mode_for_strategy(managed_strategy) if managed_strategy else "none"
                    trade_id = f"{suggestion_tracker.ENTRY_TYPE_FILLMORE_AUTONOMOUS}:{order_id or position_id}:{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d%H%M%S')}"
                    row: dict[str, Any] = {
                        "trade_id": trade_id,
                        "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                        "profile": profile.profile_name,
                        "symbol": profile.symbol,
                        "side": side,
                        "policy_type": suggestion_tracker.ENTRY_TYPE_FILLMORE_AUTONOMOUS,
                        "config_json": _json.dumps({
                            "source": "autonomous_fillmore",
                            "order_id": order_id,
                            "exit_strategy": managed_strategy or "none",
                            "exit_params": managed_params,
                            "order_type": "market",
                            "trigger_family": suggestion.get("trigger_family"),
                            "order_policy_reason": suggestion.get("order_policy_reason"),
                            "thesis_fingerprint": suggestion.get("thesis_fingerprint"),
                            "decision": suggestion.get("decision"),
                            "conviction_rung": suggestion.get("conviction_rung"),
                            "trade_thesis": suggestion.get("trade_thesis"),
                            "whats_different": suggestion.get("whats_different"),
                            "why_not_stop": suggestion.get("why_not_stop"),
                            "zone_memory_read": suggestion.get("zone_memory_read"),
                            "repeat_trade_case": suggestion.get("repeat_trade_case"),
                            "planned_rr_estimate": suggestion.get("planned_rr_estimate"),
                            "low_rr_edge": suggestion.get("low_rr_edge"),
                            "timeframe_alignment": suggestion.get("timeframe_alignment"),
                            "countertrend_edge": suggestion.get("countertrend_edge"),
                            "trigger_fit": suggestion.get("trigger_fit"),
                            "why_trade_despite_weakness": suggestion.get("why_trade_despite_weakness"),
                            "named_catalyst": suggestion.get("named_catalyst"),
                            "side_bias_check": suggestion.get("side_bias_check"),
                            "setup_location": suggestion.get("setup_location"),
                            "edge_reason": suggestion.get("edge_reason"),
                            "adverse_context": suggestion.get("adverse_context"),
                            "caveat_resolution": suggestion.get("caveat_resolution"),
                            "micro_confirmation_event": suggestion.get("micro_confirmation_event"),
                            "reasoning_quality_gate": suggestion.get("reasoning_quality_gate"),
                            "selectivity_adjustments": suggestion.get("selectivity_adjustments") or [],
                            "max_allowed_lots": suggestion.get("max_allowed_lots"),
                            "original_model_lots": suggestion.get("original_model_lots"),
                            "phase4_catalyst_score": suggestion.get("phase4_catalyst_score"),
                            "phase4_green_matches": suggestion.get("phase4_green_matches") or [],
                            "phase4_weakness_signals": suggestion.get("phase4_weakness_signals") or [],
                            "phase5_reasoning_flags": suggestion.get("phase5_reasoning_flags") or [],
                            "phase5_material_resolution_score": suggestion.get("phase5_material_resolution_score"),
                            "custom_exit_plan": suggestion.get("custom_exit_plan") or {},
                        }),
                        "entry_price": fill_price,
                        "stop_price": float(sl) if sl is not None else None,
                        "target_price": float(tp) if tp is not None else None,
                        "size_lots": lots,
                        "notes": f"autonomous_fillmore:{managed_strategy or 'none'}:order_{order_id}",
                        "snapshot_id": None,
                        "mt5_order_id": int(order_id) if order_id is not None else None,
                        "mt5_deal_id": None,
                        "mt5_retcode": 0,
                        "mt5_position_id": int(position_id),
                        "opened_by": suggestion_tracker.ENTRY_TYPE_FILLMORE_AUTONOMOUS,
                        "preset_name": getattr(profile, "active_preset_name", None) or "Autonomous Fillmore",
                        "entry_type": suggestion_tracker.ENTRY_TYPE_FILLMORE_AUTONOMOUS,
                        "breakeven_applied": 0,
                        "tp1_partial_done": 0,
                        "managed_trail_mode": trail_mode or "none",
                        "custom_exit_plan_json": _json.dumps(suggestion.get("custom_exit_plan") or {}),
                        "llm_exit_check_count": 0,
                    }
                    if managed_strategy == "llm_custom_exit":
                        row["llm_exit_max_checks"] = int(float(managed_params.get("custom_exit_max_checks") or 3))
                        row["llm_exit_runner_max_checks"] = int(float(managed_params.get("custom_exit_runner_max_checks") or 5))
                    if managed_params.get("tp1_pips") is not None:
                        row["managed_tp1_pips"] = float(managed_params["tp1_pips"])
                    if managed_params.get("tp1_close_pct") is not None:
                        row["managed_tp1_close_pct"] = float(managed_params["tp1_close_pct"])
                    if managed_params.get("be_plus_pips") is not None:
                        row["managed_be_plus_pips"] = float(managed_params["be_plus_pips"])
                    fp = suggestion.get("thesis_fingerprint")
                    if fp:
                        row["thesis_fingerprint"] = str(fp)
                    store.insert_trade(row)
                    print(
                        f"[{profile_name}] autonomous Fillmore: inserted market trade {trade_id} "
                        f"(pos {position_id}, strategy={managed_strategy or 'none'}, trail={trail_mode})"
                    )
                except Exception as e:
                    print(f"[{profile_name}] autonomous Fillmore: insert_trade error: {e}")

            # Stamp the suggestion as placed + filled.
            sid = suggestion.get("suggestion_id")
            if sid:
                try:
                    from api.main import _suggestions_db_path
                    suggestion_tracker.log_action(
                        _suggestions_db_path(profile_name),
                        suggestion_id=sid,
                        action="placed",
                        edited_fields={},
                        placed_order={
                            "side": side, "price": fill_price, "lots": lots,
                            "sl": float(sl) if sl is not None else None,
                            "tp": float(tp) if tp is not None else None,
                            "time_in_force": "MARKET", "gtd_time_utc": None,
                            "exit_strategy": managed_strategy or "none",
                            "exit_params": managed_params if managed_strategy else {},
                            "autonomous": True,
                            "entry_type": suggestion_tracker.ENTRY_TYPE_FILLMORE_AUTONOMOUS,
                            "order_type": "market",
                            "trigger_family": suggestion.get("trigger_family"),
                            "order_policy_reason": suggestion.get("order_policy_reason"),
                            "thesis_fingerprint": suggestion.get("thesis_fingerprint"),
                            "decision": suggestion.get("decision"),
                            "conviction_rung": suggestion.get("conviction_rung"),
                            "trade_thesis": suggestion.get("trade_thesis"),
                            "whats_different": suggestion.get("whats_different"),
                            "why_not_stop": suggestion.get("why_not_stop"),
                            "zone_memory_read": suggestion.get("zone_memory_read"),
                            "repeat_trade_case": suggestion.get("repeat_trade_case"),
                            "planned_rr_estimate": suggestion.get("planned_rr_estimate"),
                            "low_rr_edge": suggestion.get("low_rr_edge"),
                            "timeframe_alignment": suggestion.get("timeframe_alignment"),
                            "countertrend_edge": suggestion.get("countertrend_edge"),
                            "trigger_fit": suggestion.get("trigger_fit"),
                            "why_trade_despite_weakness": suggestion.get("why_trade_despite_weakness"),
                            "named_catalyst": suggestion.get("named_catalyst"),
                            "side_bias_check": suggestion.get("side_bias_check"),
                            "setup_location": suggestion.get("setup_location"),
                            "edge_reason": suggestion.get("edge_reason"),
                            "adverse_context": suggestion.get("adverse_context"),
                            "caveat_resolution": suggestion.get("caveat_resolution"),
                            "micro_confirmation_event": suggestion.get("micro_confirmation_event"),
                            "reasoning_quality_gate": suggestion.get("reasoning_quality_gate"),
                            "selectivity_adjustments": suggestion.get("selectivity_adjustments") or [],
                            "max_allowed_lots": suggestion.get("max_allowed_lots"),
                            "original_model_lots": suggestion.get("original_model_lots"),
                            "phase4_catalyst_score": suggestion.get("phase4_catalyst_score"),
                            "phase4_green_matches": suggestion.get("phase4_green_matches") or [],
                            "phase4_weakness_signals": suggestion.get("phase4_weakness_signals") or [],
                            "phase5_reasoning_flags": suggestion.get("phase5_reasoning_flags") or [],
                            "phase5_material_resolution_score": suggestion.get("phase5_material_resolution_score"),
                            "custom_exit_plan": suggestion.get("custom_exit_plan") or {},
                        },
                        oanda_order_id=str(order_id) if order_id is not None else None,
                    )
                    suggestion_tracker.mark_filled(
                        _suggestions_db_path(profile_name),
                        oanda_order_id=str(order_id) if order_id is not None else None,
                        fill_price=fill_price,
                        filled_at=pd.Timestamp.now(tz="UTC").isoformat(),
                        trade_id=trade_id,
                    )
                except Exception as e:
                    print(f"[{profile_name}] autonomous Fillmore: tracker stamp failed: {e}")
            return {"order_id": order_id, "status": "filled", "fill_price": fill_price}

        # ----- Limit path (existing behavior) -----
        result = adapter.order_send_pending_limit(
            symbol=profile.symbol,
            side=side,
            price=price,
            volume_lots=lots,
            sl=float(sl) if sl is not None else None,
            tp=float(tp) if tp is not None else None,
            time_in_force=tif,
            gtd_time_utc=gtd,
            comment=f"autonomous_fillmore:{profile_name}",
        )
        if result.retcode != 0:
            raise RuntimeError(f"limit order rejected: {result.comment}")

        if result.order is not None:
            state = _load_state(state_path)
            pending = list(state.get("managed_pending_orders") or [])
            pending = [p for p in pending if str(p.get("order_id")) != str(result.order)]
            pending.append({
                "order_id": int(result.order),
                "side": side,
                "price": price,
                "lots": lots,
                "sl": float(sl) if sl is not None else None,
                "tp": float(tp) if tp is not None else None,
                "exit_strategy": managed_strategy or "none",
                "trail_mode": trail_mode_for_strategy(managed_strategy) if managed_strategy else "none",
                "exit_params": managed_params,
                "source": "autonomous_fillmore",
                "suggestion_id": suggestion.get("suggestion_id"),
                "thesis_fingerprint": suggestion.get("thesis_fingerprint"),
                "decision": suggestion.get("decision"),
                "conviction_rung": suggestion.get("conviction_rung"),
                "skip_reason": suggestion.get("skip_reason"),
                "trade_thesis": suggestion.get("trade_thesis"),
                "whats_different": suggestion.get("whats_different"),
                "why_not_stop": suggestion.get("why_not_stop"),
                "zone_memory_read": suggestion.get("zone_memory_read"),
                "repeat_trade_case": suggestion.get("repeat_trade_case"),
                "planned_rr_estimate": suggestion.get("planned_rr_estimate"),
                "low_rr_edge": suggestion.get("low_rr_edge"),
                "timeframe_alignment": suggestion.get("timeframe_alignment"),
                "countertrend_edge": suggestion.get("countertrend_edge"),
                "trigger_fit": suggestion.get("trigger_fit"),
                "why_trade_despite_weakness": suggestion.get("why_trade_despite_weakness"),
                "named_catalyst": suggestion.get("named_catalyst"),
                "side_bias_check": suggestion.get("side_bias_check"),
                "setup_location": suggestion.get("setup_location"),
                "edge_reason": suggestion.get("edge_reason"),
                "adverse_context": suggestion.get("adverse_context"),
                "caveat_resolution": suggestion.get("caveat_resolution"),
                "micro_confirmation_event": suggestion.get("micro_confirmation_event"),
                "reasoning_quality_gate": suggestion.get("reasoning_quality_gate"),
                "selectivity_adjustments": suggestion.get("selectivity_adjustments") or [],
                "max_allowed_lots": suggestion.get("max_allowed_lots"),
                "original_model_lots": suggestion.get("original_model_lots"),
                "phase4_catalyst_score": suggestion.get("phase4_catalyst_score"),
                "phase4_green_matches": suggestion.get("phase4_green_matches") or [],
                "phase4_weakness_signals": suggestion.get("phase4_weakness_signals") or [],
                "phase5_reasoning_flags": suggestion.get("phase5_reasoning_flags") or [],
                "phase5_material_resolution_score": suggestion.get("phase5_material_resolution_score"),
                "custom_exit_plan": suggestion.get("custom_exit_plan") or {},
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "autonomous": True,
            })
            state["managed_pending_orders"] = pending
            _save_state(state_path, state)

        sid = suggestion.get("suggestion_id")
        if sid:
            try:
                from api.main import _suggestions_db_path
                suggestion_tracker.log_action(
                    _suggestions_db_path(profile_name),
                    suggestion_id=sid,
                    action="placed",
                    edited_fields={},
                    placed_order={
                        "side": side, "price": price, "lots": lots,
                        "sl": float(sl) if sl is not None else None,
                        "tp": float(tp) if tp is not None else None,
                        "time_in_force": tif, "gtd_time_utc": gtd,
                        "exit_strategy": managed_strategy or "none",
                        "exit_params": managed_params if managed_strategy else {},
                        "autonomous": True,
                        "entry_type": suggestion_tracker.ENTRY_TYPE_FILLMORE_AUTONOMOUS,
                        "order_type": "limit",
                        "trigger_family": suggestion.get("trigger_family"),
                        "order_policy_reason": suggestion.get("order_policy_reason"),
                        "thesis_fingerprint": suggestion.get("thesis_fingerprint"),
                        "decision": suggestion.get("decision"),
                        "conviction_rung": suggestion.get("conviction_rung"),
                        "trade_thesis": suggestion.get("trade_thesis"),
                        "whats_different": suggestion.get("whats_different"),
                        "why_not_stop": suggestion.get("why_not_stop"),
                        "zone_memory_read": suggestion.get("zone_memory_read"),
                        "repeat_trade_case": suggestion.get("repeat_trade_case"),
                        "planned_rr_estimate": suggestion.get("planned_rr_estimate"),
                        "low_rr_edge": suggestion.get("low_rr_edge"),
                        "timeframe_alignment": suggestion.get("timeframe_alignment"),
                        "countertrend_edge": suggestion.get("countertrend_edge"),
                        "trigger_fit": suggestion.get("trigger_fit"),
                        "why_trade_despite_weakness": suggestion.get("why_trade_despite_weakness"),
                        "named_catalyst": suggestion.get("named_catalyst"),
                        "side_bias_check": suggestion.get("side_bias_check"),
                        "setup_location": suggestion.get("setup_location"),
                        "edge_reason": suggestion.get("edge_reason"),
                        "adverse_context": suggestion.get("adverse_context"),
                        "caveat_resolution": suggestion.get("caveat_resolution"),
                        "micro_confirmation_event": suggestion.get("micro_confirmation_event"),
                        "reasoning_quality_gate": suggestion.get("reasoning_quality_gate"),
                        "selectivity_adjustments": suggestion.get("selectivity_adjustments") or [],
                        "max_allowed_lots": suggestion.get("max_allowed_lots"),
                        "original_model_lots": suggestion.get("original_model_lots"),
                        "phase4_catalyst_score": suggestion.get("phase4_catalyst_score"),
                        "phase4_green_matches": suggestion.get("phase4_green_matches") or [],
                        "phase4_weakness_signals": suggestion.get("phase4_weakness_signals") or [],
                        "phase5_reasoning_flags": suggestion.get("phase5_reasoning_flags") or [],
                        "phase5_material_resolution_score": suggestion.get("phase5_material_resolution_score"),
                        "custom_exit_plan": suggestion.get("custom_exit_plan") or {},
                    },
                    oanda_order_id=str(result.order) if result.order is not None else None,
                )
            except Exception as e:
                print(f"[{profile_name}] autonomous Fillmore: log_action failed: {e}")
        return {"order_id": result.order, "status": "placed"}
    finally:
        try:
            adapter.shutdown()
        except Exception:
            pass
