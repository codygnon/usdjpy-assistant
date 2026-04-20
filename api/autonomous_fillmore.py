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

import pandas as pd

from api import autonomous_performance

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
AUTONOMOUS_PROMPT_VERSION = "autonomous_phase_a_v1"

DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": False,
    "mode": "off",                      # off|shadow|paper|live — paper = real OANDA practice orders (not in-app sim)
    "aggressiveness": "balanced",
    "limit_gtd_minutes": 45,            # autonomous limit orders expire if not filled; keep stale ideas from resting forever
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
    "max_open_ai_trades": 2,
    "max_daily_loss_usd": 50.0,
    "max_consecutive_errors": 5,
    "model": "gpt-5.4-mini",
    # adaptive throttle
    "throttle_no_trade_streak": 5,       # after N "no trade" LLM replies, pause
    "throttle_no_trade_cooldown_sec": 300,
    "throttle_loss_streak": 2,
    "throttle_loss_cooldown_sec": 900,
    # Stage 3: book-correlation veto + same-setup dedupe
    "correlation_veto_enabled": True,    # post-LLM: block placement if same-side position open within distance
    "correlation_distance_pips": 15.0,
    "repeat_setup_dedupe_enabled": True, # pre-LLM: block tick if a placed suggestion in same price bucket fired within window
    "repeat_setup_window_min": 30,
    "repeat_setup_bucket_pips": 25.0,
    # Stage 6: event blackout + multi-trade planning
    "event_blackout_enabled": True,      # gate blocks when high-impact USD/JPY event within N minutes
    "event_blackout_minutes": 30,
    "multi_trade_enabled": True,         # LLM can propose 0-2 setups per call instead of exactly 1
    "max_suggestions_per_call": 2,
    "streak_scratch_threshold_pips": 1.0,
    "min_lot_size": 0.01,
}

# Per-mode gate thresholds. Lower = easier to pass = more LLM calls.
# These are the *signal* filters applied after hard filters pass.
GATE_THRESHOLDS: dict[str, dict[str, Any]] = {
    "conservative": {
        "description": "M3 trend + M1 EMA stack + (pullback OR zone entry) + daily H/L buffer + within 5p of structure",
        "require_m3_trend": True,
        "require_m1_stack": True,
        "require_pullback_or_zone": True,
        "pullback_zone_min_pips": 1.5,
        "pullback_lookback_bars": 3,
        "require_daily_hl_buffer_pips": 5.0,
        "require_level_proximity_pips": 5.0,
        "require_min_m5_atr_pips": 3.0,
        "expected_pass_rate_pct": 1.5,
    },
    "balanced": {
        "description": "M3 trend aligned + M1 EMA stack + within 8p of a structure level (PDH/PDL/round/today's H-L)",
        "require_m3_trend": True,
        "require_m1_stack": True,
        "require_pullback_or_zone": False,
        "require_daily_hl_buffer_pips": 0.0,
        "require_level_proximity_pips": 8.0,
        "require_min_m5_atr_pips": 3.0,
        "expected_pass_rate_pct": 3.5,
    },
    "aggressive": {
        "description": "M3 or M1 trend signal (veto on disagreement) + pullback/zone + daily H/L buffer + within 12p of structure",
        "require_m3_trend": False,
        "require_m1_stack": False,
        "require_pullback_or_zone": True,
        "pullback_zone_min_pips": 2.0,
        "pullback_lookback_bars": 2,
        "require_daily_hl_buffer_pips": 3.0,
        "require_any_trend_signal": True,
        "reject_m3_m1_mismatch_if_both_present": True,
        "require_level_proximity_pips": 12.0,
        "require_min_m5_atr_pips": 3.0,
        "expected_pass_rate_pct": 2.5,
    },
    "very_aggressive": {
        "description": "M3 or M1 trend signal (veto on disagreement). No structure proximity check.",
        "require_m3_trend": False,
        "require_m1_stack": False,
        "require_pullback_or_zone": False,
        "require_daily_hl_buffer_pips": 0.0,
        "require_any_trend_signal": True,
        "reject_m3_m1_mismatch_if_both_present": True,
        "require_level_proximity_pips": 0.0,
        "require_min_m5_atr_pips": 2.5,
        "expected_pass_rate_pct": 8.0,
    },
}

# Session-specific calibrated overrides (train/test validated).
# Keys: (aggressiveness, session_label). Values: partial dicts merged onto the
# base GATE_THRESHOLDS for that mode.  Only "safe non-regressive" candidates
# are adopted here — see research_out/autonomous_gate_calibration_1000k_sessions.md.
SESSION_GATE_OVERRIDES: dict[tuple[str, str], dict[str, Any]] = {
    # ── balanced ──
    ("balanced", "tokyo"): {
        "require_min_m5_atr_pips": 3.0,
        "require_level_proximity_pips": 8.0,
    },
    ("balanced", "ny"): {
        "require_min_m5_atr_pips": 3.5,
        "require_level_proximity_pips": 12.0,
    },
    ("balanced", "london/ny"): {
        "require_min_m5_atr_pips": 3.0,
        "require_level_proximity_pips": 8.0,
    },
    # balanced london: no safe candidate — uses base thresholds
    # ── aggressive ──
    ("aggressive", "ny"): {
        "require_min_m5_atr_pips": 3.0,
        "require_level_proximity_pips": 12.0,
        "pullback_zone_min_pips": 1.5,
        "pullback_lookback_bars": 1,
    },
    ("aggressive", "london"): {
        "require_min_m5_atr_pips": 3.0,
        "require_level_proximity_pips": 10.0,
        "pullback_zone_min_pips": 2.0,
        "pullback_lookback_bars": 1,
    },
    # aggressive tokyo: no safe candidate — uses base thresholds
    # aggressive london/ny: no safe candidate — uses base thresholds
}


def _resolve_gate_thresholds(agg: str, session_label: str) -> dict[str, Any]:
    base = dict(GATE_THRESHOLDS.get(agg) or GATE_THRESHOLDS["balanced"])
    overrides = SESSION_GATE_OVERRIDES.get((agg, session_label))
    if overrides:
        base.update(overrides)
    return base


_AUTONOMOUS_PROMPT_SKELETON_ID = "autonomous_decision_request_v1"
_AUTONOMOUS_AUX_MEMORY_BUDGET_WORDS = 650
_USDJPY_SPREAD_LIMITS_PIPS = {
    "tokyo": 3.0,
    "london": 3.0,
    "ny": 3.0,
    "london/ny": 3.0,
    "off-hours": 3.0,
}


def _autonomous_prompt_hash() -> str:
    return hashlib.sha256(_AUTONOMOUS_PROMPT_SKELETON_ID.encode("utf-8")).hexdigest()[:12]


def _word_count(text: str) -> int:
    return len(str(text or "").split())


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
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _save_state(state_path: Path, data: dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


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
    return cfg


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
    merged = {**cur, **clean}
    auto["config"] = merged
    if not bool(merged.get("enabled")) or str(merged.get("mode") or "off").lower() == "off":
        runtime = _runtime_block(state)
        runtime["throttle_until_utc"] = None
        runtime["throttle_reason"] = None
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
    })
    if "previous_streak_regime_label" not in runtime:
        runtime["previous_streak_regime_label"] = str(runtime.get("previous_regime_label") or "normal")
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
    """Called from trade-close path when an ai_manual trade closes."""
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


def _nearest_structure_pips(
    tick_mid: float,
    data_by_tf: dict[str, pd.DataFrame],
    pip_size: float = 0.01,
) -> dict[str, Optional[float]]:
    """Distance in pips to nearest key structure level above and below current price.

    Considers: today's H/L, previous day's H/L, and round levels at x.00/x.50 cadence
    within ±3 round-level steps. Returns a dict with keys:
        overhead_pips: distance to nearest level strictly above price (or None)
        underfoot_pips: distance to nearest level strictly below price (or None)
        nearest_pips: min of the two, or None if neither side has a level.

    Used by the gate to ensure the LLM is invoked near actionable structure,
    not in mid-range chop where fades/breaks are premature.
    """
    levels_above: list[float] = []
    levels_below: list[float] = []

    d_df = data_by_tf.get("D") if data_by_tf else None
    if d_df is not None and len(d_df) >= 1:
        try:
            today = d_df.iloc[-1]
            th = float(today["high"]); tl = float(today["low"])
            if th > tick_mid:
                levels_above.append(th)
            if tl < tick_mid:
                levels_below.append(tl)
        except Exception:
            pass
    if d_df is not None and len(d_df) >= 2:
        try:
            prev = d_df.iloc[-2]
            pdh = float(prev["high"]); pdl = float(prev["low"])
            if pdh > tick_mid:
                levels_above.append(pdh)
            if pdl < tick_mid:
                levels_below.append(pdl)
        except Exception:
            pass

    # Round levels: x.00 and x.50 cadence (50-pip spacing at pip_size 0.01).
    try:
        base = int(tick_mid * 2) / 2.0  # floor to nearest 0.50
        for step in range(-3, 4):
            lvl = round(base + 0.5 * step, 2)
            if lvl > tick_mid:
                levels_above.append(lvl)
            elif lvl < tick_mid:
                levels_below.append(lvl)
    except Exception:
        pass

    pip = pip_size or 0.01

    def _closest(levels: list[float]) -> Optional[float]:
        if not levels:
            return None
        return min(abs(l - tick_mid) / pip for l in levels)

    up = _closest(levels_above)
    dn = _closest(levels_below)
    candidates = [v for v in (up, dn) if v is not None]
    return {
        "overhead_pips": up,
        "underfoot_pips": dn,
        "nearest_pips": min(candidates) if candidates else None,
    }


def _correlated_open_position(
    db_path: Optional[Path],
    side: str,
    proposed_price: float,
    distance_pips: float,
    pip_size: float = 0.01,
) -> Optional[dict[str, Any]]:
    """Return an open ai_manual position that conflicts with a new proposed entry.

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


@dataclass
class GateInputs:
    spread_pips: float
    tick_mid: float
    open_ai_trade_count: int
    data_by_tf: dict[str, pd.DataFrame]
    ntz_active: bool = False
    suggestions_db_path: Optional[Path] = None  # for same-setup dedupe lookup; None disables the check
    upcoming_events: Optional[list[dict[str, Any]]] = None  # from get_economic_calendar_events(); None skips check


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
    max_spread_pips = _session_spread_limit_pips(session_label)
    if inputs.spread_pips is None or inputs.spread_pips > max_spread_pips:
        return _block(
            "hard",
            f"spread_too_wide:{inputs.spread_pips}",
            {"max_spread_pips": max_spread_pips, "session": session_label},
        )
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
    if inputs.open_ai_trade_count >= int(risk_regime.get("effective_max_open_ai_trades") or cfg.get("max_open_ai_trades") or 2):
        return _block(
            "hard",
            f"max_open_ai_trades:{inputs.open_ai_trade_count}",
            {"effective_limit": int(risk_regime.get("effective_max_open_ai_trades") or 0)},
        )
    # Daily loss cap
    if float(rt.get("daily_pnl_usd") or 0.0) <= -float(cfg.get("max_daily_loss_usd") or 50.0):
        return _block("hard", f"daily_loss_cap:{rt.get('daily_pnl_usd')}")
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

    # ---- Layer 3: signal gate (mode-dependent, session-calibrated) ----
    thresholds = _resolve_gate_thresholds(agg, session_label)

    m3 = _m3_trend(inputs.data_by_tf)
    m1 = _m1_stack(inputs.data_by_tf)
    extras: dict[str, Any] = {"m3": m3, "m1": m1, "spread": inputs.spread_pips, "session": session_label}

    min_m5_atr_pips = float(thresholds.get("require_min_m5_atr_pips") or 0.0)
    if min_m5_atr_pips > 0:
        m5_atr_pips = _atr_pips(inputs.data_by_tf, "M5", period=14, pip_size=0.01)
        extras["m5_atr_pips"] = round(m5_atr_pips, 2) if isinstance(m5_atr_pips, (int, float)) else None
        if not _sufficient_volatility(inputs.data_by_tf, min_atr_pips=min_m5_atr_pips, timeframe="M5", pip_size=0.01):
            return _block("signal", "low_volatility", extras)

    if thresholds.get("require_m3_trend") and m3 is None:
        return _block("signal", "no_m3_trend", extras)
    if thresholds.get("require_m1_stack") and m1 is None:
        return _block("signal", "no_m1_stack", extras)
    mismatch_veto = bool(
        thresholds.get("reject_m3_m1_mismatch_if_both_present")
        or (thresholds.get("require_m3_trend") and thresholds.get("require_m1_stack"))
    )
    if mismatch_veto and m3 is not None and m1 is not None and m3 != m1:
        return _block("signal", f"m3_m1_mismatch:{m3}/{m1}", extras)

    if thresholds.get("require_pullback_or_zone"):
        trend = m1 or m3
        if not _m1_pullback_or_zone(
            inputs.data_by_tf,
            trend,
            zone_min_pips=float(thresholds.get("pullback_zone_min_pips") or 1.5),
            lookback_bars=int(thresholds.get("pullback_lookback_bars") or 3),
        ):
            return _block("signal", "no_pullback_or_zone", extras)

    buf = float(thresholds.get("require_daily_hl_buffer_pips") or 0.0)
    if buf > 0 and _near_daily_hl(inputs.tick_mid, inputs.data_by_tf, buf):
        return _block("signal", f"near_daily_hl_within_{buf:.1f}p", extras)

    # "aggressive" mode only requires *some* trend signal — either M3 or M1.
    if thresholds.get("require_any_trend_signal"):
        if m3 is None and m1 is None:
            return _block("signal", "no_trend_signal", extras)

    # Level-proximity: only fire when price is actually near actionable structure.
    # Prevents the LLM from being woken mid-range where its level-based rationale
    # can't actually be expressed as a clean entry.
    prox_thr = float(thresholds.get("require_level_proximity_pips") or 0.0)
    if prox_thr > 0:
        prox = _nearest_structure_pips(inputs.tick_mid, inputs.data_by_tf)
        nearest = prox.get("nearest_pips")
        extras["nearest_level_pips"] = round(nearest, 1) if nearest is not None else None
        extras["overhead_level_pips"] = round(prox["overhead_pips"], 1) if prox.get("overhead_pips") is not None else None
        extras["underfoot_level_pips"] = round(prox["underfoot_pips"], 1) if prox.get("underfoot_pips") is not None else None
        if nearest is None or nearest > prox_thr:
            return _block("signal", f"no_structure_within_{prox_thr:.0f}p", extras)

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

    return GateDecision(
        timestamp_utc=now_utc.isoformat(),
        result="pass", layer="pass", reason="ok",
        mode=mode, aggressiveness=agg, extras={**extras, "risk_regime": risk_regime.get("label")},
    )


# -----------------------------------------------------------------------------
# Stats for the UI
# -----------------------------------------------------------------------------

def build_stats(state_path: Path, cfg: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    cfg = cfg or get_config(state_path)
    state = _load_state(state_path)
    rt = (state.get("autonomous_fillmore") or {}).get("runtime") or {}
    _rollover_daily_counters(rt)  # in-memory only, we don't persist here
    risk_regime = _compute_risk_regime(rt, cfg)

    decisions = list(rt.get("decisions") or [])
    # Break down by result + layer over last N.
    total = len(decisions)
    passes = sum(1 for d in decisions if d.get("r") == "pass")
    blocks = total - passes
    layer_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for d in decisions:
        if d.get("r") == "block":
            layer_counts[str(d.get("l") or "?")] = layer_counts.get(str(d.get("l") or "?"), 0) + 1
            reason_counts[str(d.get("why") or "?")] = reason_counts.get(str(d.get("why") or "?"), 0) + 1

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
        "recent_gate_blocks": dict(sorted((rt.get("recent_gate_blocks") or {}).items(), key=lambda kv: -kv[1])[:8]),
        "health_alerts": health_alerts,
        "performance": perf_rows,
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
    try:
        rows = store.list_open_trades(profile_name)
    except Exception:
        return 0
    n = 0
    for r in rows:
        rd = dict(r)
        if str(rd.get("entry_type") or "").lower() == "ai_manual":
            n += 1
    return n


def _refresh_daily_pnl(store, profile_name: str, rt: dict[str, Any], cfg: Optional[dict[str, Any]] = None) -> None:
    """Query closed ai_manual trades for today and update rt['daily_pnl_usd'].

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

    daily_drawdown_active = float(rt.get("daily_pnl_usd") or 0.0) <= -(
        float(cfg.get("max_daily_loss_usd") or 50.0) * 0.6
    )
    if daily_drawdown_active:
        label = "defensive_hard"
    else:
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
        effective_max_open = 1
    elif label == "defensive_soft":
        risk_multiplier = 0.75
        effective_cooldown = int(math.ceil(float(cfg.get("min_llm_cooldown_sec") or 60) * 1.5))
        effective_max_open = min(int(cfg.get("max_open_ai_trades") or 2), 1)
    else:
        risk_multiplier = 1.0
        effective_cooldown = int(cfg.get("min_llm_cooldown_sec") or 60)
        effective_max_open = int(cfg.get("max_open_ai_trades") or 2)

    return {
        "label": label,
        "streak_label": streak_label,
        "daily_drawdown_active": daily_drawdown_active,
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
        # Filter to ai_manual entries that closed today.
        if "entry_type" in df.columns:
            df = df[df["entry_type"].astype(str).str.lower() == "ai_manual"]
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
        # Compute loss streak from closed ai_manual trades today (chronological).
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


def tick_autonomous_fillmore(
    profile,
    profile_name: str,
    state_path: Path,
    store,
    tick,
    data_by_tf: dict[str, pd.DataFrame],
    ntz_active: bool = False,
    **_unused: Any,
) -> None:
    """Called from the run loop each iteration after trade management.

    Fast-paths out when autonomous is disabled. When the gate passes and mode is
    paper/live, it invokes the suggest endpoint and places the limit.
    """
    cfg = get_config(state_path)
    if not cfg.get("enabled"):
        return  # fast path — don't even log when fully off.

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

    inputs = GateInputs(
        spread_pips=float(spread_pips) if spread_pips is not None else 999.0,
        tick_mid=mid,
        open_ai_trade_count=_count_open_ai_trades(store, profile_name),
        data_by_tf=data_by_tf or {},
        ntz_active=bool(ntz_active),
        suggestions_db_path=db_path,
        upcoming_events=upcoming,
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
    log_decision(state_path, decision)

    if decision.result != "pass":
        return

    # ---- Gate passed: invoke LLM via existing suggest plumbing ----
    mode = str(cfg.get("mode") or "off")
    print(f"[{profile_name}] autonomous Fillmore: gate passed ({cfg.get('aggressiveness')}, {mode}); invoking LLM")

    try:
        suggestions = _invoke_suggest(profile, profile_name, cfg, risk_regime=risk_regime)
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
    max_open = int(risk_regime.get("effective_max_open_ai_trades") or cfg.get("max_open_ai_trades") or 2)

    for suggestion in suggestions:
        print(
            f"[{profile_name}] autonomous Fillmore: processing suggestion — "
            f"side={suggestion.get('side')} lots={suggestion.get('lots')} "
            f"quality={suggestion.get('quality')} mode={mode}"
        )
        # Respect max_open_ai_trades across the batch — count may change after each placement.
        current_open = _count_open_ai_trades(store, profile_name)
        if current_open >= max_open:
            print(f"[{profile_name}] autonomous Fillmore: max open trades reached ({current_open}/{max_open}); skipping remaining suggestions")
            break

        sug_lots = float(suggestion.get("lots") or 0)
        if sug_lots <= 0:
            quality = str(suggestion.get("quality") or "C")
            print(f"[{profile_name}] autonomous Fillmore: LLM returned 0 lots (quality={quality}); skipping")
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
            print(f"[{profile_name}] autonomous Fillmore: place error: {e}")
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


def _invoke_suggest(
    profile,
    profile_name: str,
    cfg: dict[str, Any],
    risk_regime: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Invoke the LLM for autonomous Fillmore.

    Reason-then-commit flow: the user prompt asks for a plain-text ANALYSIS
    section first, then a fenced ```json``` DECISION block. When multi-trade
    planning is enabled the model may return an array of 0-2 trade objects;
    otherwise a single object. Returns a list of normalized suggestion dicts.

    Lazy imports avoid circular dependency with api.main.
    """
    from api.ai_trading_chat import (
        autonomous_system_prompt_from_context,
        build_trade_suggestion_news_block,
        build_trading_context,
        resolve_ai_suggest_model,
    )
    from api import suggestion_tracker
    from api.ai_exit_strategies import (
        AI_EXIT_STRATEGIES, DEFAULT_AI_EXIT_STRATEGY,
        exit_strategies_prompt_block, merge_exit_params, normalize_exit_strategy,
    )
    import openai
    import json as _json

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not configured")

    ctx = build_trading_context(profile, profile_name)
    model = resolve_ai_suggest_model(cfg.get("model"))
    system = autonomous_system_prompt_from_context(
        ctx,
        model,
        autonomous_config=cfg,
        risk_regime=risk_regime,
    )
    risk_regime = risk_regime or {
        "label": "normal",
        "risk_multiplier": 1.0,
    }
    # Best-effort news enrichment; swallow errors.
    try:
        news_block = build_trade_suggestion_news_block(
            symbol=getattr(profile, "symbol", "USD_JPY"),
            rss_headline_count=8,
            web_result_count=3,
            parallel_timeout_sec=10.0,
        )
        system = f"{system}\n\n{news_block}"
    except Exception:
        pass
    # Learning memory block + today's autonomous run history.
    try:
        from api.main import _suggestions_db_path
        db_path = _suggestions_db_path(profile_name)
        learning = suggestion_tracker.build_learning_prompt_block(
            db_path, days_back=180, max_recent_examples=6, current_ctx=ctx,
        )
        reflection_block = suggestion_tracker.build_autonomous_reflection_prompt_block(
            db_path,
            limit=8,
            autonomous_only=True,
        )
        perf_block = autonomous_performance.build_performance_memory_block(
            db_path,
            risk_regime=risk_regime,
        )
        today_block = suggestion_tracker.build_autonomous_today_block(db_path, max_items=10)
        aux_memory = _fit_aux_memory_blocks([
            ("learning", learning, True),
            ("performance", perf_block, True),
            ("reflections", reflection_block, False),
            ("today", today_block, False),
        ])
        if aux_memory:
            system = f"{system}\n\n{aux_memory}"
    except Exception:
        pass

    agg = cfg.get("aggressiveness") or "balanced"
    max_lots = float(cfg.get("max_lots_per_trade") or 15.0)
    mode = str(cfg.get("mode") or "off")
    limit_gtd_min = int(cfg.get("limit_gtd_minutes") or 45)
    exec_note = (
        "ORDER TYPE: You choose — 'market' or 'limit'.\n"
        "  MARKET: fills instantly at current bid/ask. Use when the tape is moving and you want in now.\n"
        f"  LIMIT: rests at the 'price' you set; fills only if the market reaches it. Default expiry {limit_gtd_min} min (GTD). "
        "BUY LIMIT must be BELOW current bid; SELL LIMIT must be ABOVE current ask. "
        "Use limit when you see a structural level worth waiting for — the order dies harmlessly if it doesn't fill."
    )
    price_instr = (
        "For MARKET: 'price' is informational (current mid is fine). "
        "For LIMIT: 'price' is the exact level where the order rests. Place it where your thesis says the edge is."
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

    base_lots = float(cfg.get("base_lot_size") or 5.0)
    lot_dev = float(cfg.get("lot_deviation") or 4.0)
    lot_lo = max(1, int(base_lots - lot_dev))
    lot_hi = min(int(max_lots), int(base_lots + lot_dev))
    lot_mid = int(base_lots)

    suggest_prompt = (
        f"=== AUTONOMOUS DECISION REQUEST ===\n"
        f"Gate mode: {agg}. Execution mode: {mode}.{paper_note}\n"
        f"{exec_note}\n"
        f"LOT RANGE: {lot_lo}-{lot_hi} lots (base {lot_mid}). Hard ceiling: {int(max_lots)}.\n"
        + (f"\n{multi_note}\n" if multi_note else "") +
        "\n"
        "RESPONSE FORMAT (two parts, in this exact order):\n"
        "\n"
        "1. ANALYSIS (plain text, 6-12 lines). Think through the setup. Cover each of:\n"
        "   - H1/M15/M5/M1 trend consensus (note any divergence).\n"
        "   - JPY CROSS BIAS — does it confirm or contradict direction?\n"
        "   - Nearest PRICE STRUCTURE level(s) + order-book clusters. Name the level you'd anchor the entry on.\n"
        "   - Relevant bar patterns / recent candle streak.\n"
        "   - M5/M15 ATR — does it justify a wider or tighter SL?\n"
        "   - OPEN POSITIONS + YOUR MOST RECENT SUGGESTION — are you stacking? "
        "Same side, same level = size down or skip. Different level = new trade.\n"
        "   - Imminent events / session risk — factor into size, not into whether to trade.\n"
        "   - EXIT STRATEGY + EXIT PLAN: Choose a strategy and briefly describe your conditional exit logic "
        "(e.g., 'trail on M1 21 EMA after TP1' or 'exit if M3 trend flips bearish within 10 min'). "
        "The thesis monitor will use your exit_plan to judge mid-trade.\n"
        "   - Final call: what direction, what size? Be direct.\n"
        "\n"
        + decision_format +
        "```json\n"
        "{\n"
        '  "order_type": "market" | "limit",\n'
        '  "side": "buy" | "sell",\n'
        '  "price": <entry price as number>,\n'
        '  "sl": <stop loss price as number>,\n'
        '  "tp": <take profit price as number>,\n'
        f'  "lots": <0 to skip, or {lot_lo}-{lot_hi} — this IS your conviction>,\n'
        '  "time_in_force": "GTC" | "GTD",\n'
        '  "gtd_time_utc": <ISO datetime string if GTD, else null>,\n'
        '  "exit_strategy": one of [' + _strategy_ids + '],\n'
        '  "exit_params": {<optional numeric overrides>} or null,\n'
        '  "exit_plan": "<1-2 sentence custom exit logic for the thesis monitor>",\n'
        '  "rationale": "<1-2 sentence concise summary of the decision>",\n'
        '  "quality": "A" | "B" | "C"\n'
        "}\n"
        "```\n"
        "\n"
        "ENTRY PRICE RULE: " + price_instr + "\n"
        "SL/TP: Baseline SL 10-15p, TP 4-10p; flex with ATR. Never glue SL 1-2p past a known level.\n"
        f'EXIT STRATEGY: default "{DEFAULT_AI_EXIT_STRATEGY}" unless analysis favors another (or "none").\n'
        + exit_calibration_text + "\n"
        "EXIT PLAN: Describe how you want to manage the trade mid-flight. The thesis monitor runs every ~3 min "
        "and can hold, tighten SL, scale out, or exit. Give it clear conditions: 'exit if M3 flips bear', "
        "'tighten SL to BE after +5p', 'scale 50% at TP1 then trail on M1 21 EMA'. If you have no custom plan, "
        'write "default" and the template strategy handles it.\n'
        "LOT SIZING — YOUR LOTS ARE YOUR CONVICTION:\n"
        "You are a trader. You see a setup, you size it, you take it. Do not second-guess yourself. "
        "Do not talk yourself out of trades. The gate already confirmed conditions are worth looking at — "
        "your job is to find the trade and size it.\n"
        f"- {lot_hi} lots: everything lines up. Trend, structure, invalidation, session — put it on.\n"
        f"- {lot_mid} lots: the trade is there but one thing is not perfect. Take it at base size.\n"
        f"- {lot_lo} lots: edge is thin but real. Small size, tight stop, take it.\n"
        "- 0 lots: there is no trade. Price is mid-range with no structure, or you are stacking into an "
        "existing position at the same level. 0 is for 'there is literally nothing here' — not for 'I am unsure'.\n"
        "If you can identify a direction and a level, you have a trade. Size it and go.\n"
        "\n"
        "QUALITY TAG (logged for analytics, does NOT affect placement):\n"
        '- "A" = textbook. You sized it up.\n'
        '- "B" = solid. One soft element, you sized it down.\n'
        '- "C" = thin or skip.\n'
        "\n"
        + _exit_catalog_text
    )

    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": suggest_prompt},
        ],
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
        default_tif = "GTD" if sug_order_type == "limit" else "GTC"
        tif = str(suggestion.get("time_in_force") or default_tif).upper()
        if tif not in ("GTC", "GTD"):
            tif = default_tif
        suggestion["time_in_force"] = tif
        if tif == "GTD" and not suggestion.get("gtd_time_utc"):
            gtd_min = limit_gtd_min if sug_order_type == "limit" else 60
            suggestion["gtd_time_utc"] = (
                (datetime.now(timezone.utc) + timedelta(minutes=gtd_min))
                .replace(microsecond=0).isoformat().replace("+00:00", "Z")
            )
        # exit_plan: free-text field for the thesis monitor to reason about mid-trade.
        suggestion["exit_plan"] = str(suggestion.get("exit_plan") or "default").strip()
        suggestion["prompt_version"] = AUTONOMOUS_PROMPT_VERSION
        suggestion["prompt_hash"] = _autonomous_prompt_hash()

        raw_exit = suggestion.get("exit_strategy")
        raw_exit_s = str(raw_exit).strip().lower() if raw_exit is not None else ""
        if raw_exit_s == "none":
            suggestion["exit_strategy"] = "none"
            suggestion["exit_params"] = {}
        else:
            chosen = normalize_exit_strategy(raw_exit_s) if raw_exit_s in AI_EXIT_STRATEGIES else DEFAULT_AI_EXIT_STRATEGY
            suggestion["exit_strategy"] = chosen
            raw_params = suggestion.get("exit_params") if isinstance(suggestion.get("exit_params"), dict) else None
            merged_params = merge_exit_params(chosen, raw_params)
            suggestion["exit_params"] = _apply_autonomous_exit_calibration(chosen, merged_params, ctx)
        suggestion["model_used"] = model

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
            print(f"[{profile_name}] autonomous Fillmore: log_generated failed: {e}")
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
                    trade_id = f"ai_manual:{order_id or position_id}:{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d%H%M%S')}"
                    row: dict[str, Any] = {
                        "trade_id": trade_id,
                        "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                        "profile": profile.profile_name,
                        "symbol": profile.symbol,
                        "side": side,
                        "policy_type": "ai_manual",
                        "config_json": _json.dumps({
                            "source": "autonomous_fillmore",
                            "order_id": order_id,
                            "exit_strategy": managed_strategy or "none",
                            "exit_params": managed_params,
                            "order_type": "market",
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
                        "opened_by": "ai_manual",
                        "preset_name": getattr(profile, "active_preset_name", None) or "Autonomous Fillmore",
                        "entry_type": "ai_manual",
                        "breakeven_applied": 0,
                        "tp1_partial_done": 0,
                        "managed_trail_mode": trail_mode or "none",
                    }
                    if managed_params.get("tp1_pips") is not None:
                        row["managed_tp1_pips"] = float(managed_params["tp1_pips"])
                    if managed_params.get("tp1_close_pct") is not None:
                        row["managed_tp1_close_pct"] = float(managed_params["tp1_close_pct"])
                    if managed_params.get("be_plus_pips") is not None:
                        row["managed_be_plus_pips"] = float(managed_params["be_plus_pips"])
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
                            "order_type": "market",
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
                "source": "ai_manual",  # reuse watchdog path
                "suggestion_id": suggestion.get("suggestion_id"),
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
                        "order_type": "limit",
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
