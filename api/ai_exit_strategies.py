"""Exit strategy catalog for AI-suggested manual limit orders.

These mirror the managed-exit behaviors already implemented in run_loop.py for the
KT/CG Trial presets. The AI trade suggestion panel lets the model pick (or reject)
one of these strategies; the run loop's `_manage_ai_manual_trades` watchdog applies
the chosen exit logic once the pending limit order fills.
"""
from __future__ import annotations

from typing import Any

# Strategy IDs used in trade rows (managed_trail_mode) and in the AI response.
# "hwm"    -> tp1_be_hwm_trail (tick-driven HWM trail)
# "m5"     -> tp1_be_m5_trail  (M5 EMA bar-close-only trail)
# "m1"     -> tp1_be_trail     (M1 EMA bar-close-only trail)
# "be"     -> tp1_be_only      (TP1 + BE, no trail on runner)
# "none"   -> none             (broker-side SL/TP only, no runtime management)

AI_EXIT_STRATEGIES: dict[str, dict[str, Any]] = {
    "tp1_be_hwm_trail": {
        "id": "tp1_be_hwm_trail",
        "trail_mode": "hwm",
        "label": "TP1 + BE + HWM tick trail",
        "description": (
            "Close 70% at +6p, move SL to BE (spread + 0.5p), then trail the runner "
            "3 pips behind the high-water mark on every tick. Best for clean trends "
            "with room to run — the HWM trail captures the bulk of extended moves."
        ),
        "defaults": {
            "tp1_pips": 6.0,
            "tp1_close_pct": 70.0,
            "be_plus_pips": 0.5,
            "tp1_lock_in_fraction": 0.2,
            "hwm_trail_pips": 3.0,
        },
    },
    "tp1_be_m5_trail": {
        "id": "tp1_be_m5_trail",
        "trail_mode": "m5",
        "label": "TP1 + BE + M5 EMA bar-close trail",
        "description": (
            "Close 80% at +6p, move SL to BE (spread + 0.5p), then trail the runner "
            "on M5 EMA-20 bar closes. Slower, more forgiving trail than HWM — better "
            "for choppy trends where wicks would stop you out on the HWM version."
        ),
        "defaults": {
            "tp1_pips": 6.0,
            "tp1_close_pct": 80.0,
            "be_plus_pips": 0.5,
            "tp1_lock_in_fraction": 0.2,
            "trail_ema_period": 20,
        },
    },
    "tp1_be_trail": {
        "id": "tp1_be_trail",
        "trail_mode": "m1",
        "label": "TP1 + BE + M1 EMA bar-close trail",
        "description": (
            "Close 50% at +4p, move SL to BE (spread + 0.5p), then trail the runner "
            "on M1 EMA-21 bar closes. Tighter than M5 — locks in more profit but "
            "gets stopped out faster in retracements."
        ),
        "defaults": {
            "tp1_pips": 4.0,
            "tp1_close_pct": 50.0,
            "be_plus_pips": 0.5,
            "tp1_lock_in_fraction": 0.2,
            "trail_ema_period": 21,
        },
    },
    "tp1_be_only": {
        "id": "tp1_be_only",
        "trail_mode": "be",
        "label": "TP1 + BE only (no trail)",
        "description": (
            "Close 50% at +4p, move SL to BE (spread + 0.5p), let the runner hit "
            "the original TP or get stopped at BE. Safest runner management — no "
            "risk of the trail closing the trade prematurely."
        ),
        "defaults": {
            "tp1_pips": 4.0,
            "tp1_close_pct": 50.0,
            "be_plus_pips": 0.5,
            "tp1_lock_in_fraction": 0.2,
        },
    },
    "none": {
        "id": "none",
        "trail_mode": "none",
        "label": "No managed exit (broker SL/TP only)",
        "description": (
            "Rely entirely on the broker-side SL and TP attached to the order. "
            "Use when you want a pure set-and-forget trade with no runtime "
            "management overhead."
        ),
        "defaults": {},
    },
}

DEFAULT_AI_EXIT_STRATEGY = "tp1_be_hwm_trail"


def exit_strategies_prompt_block() -> str:
    """Human-readable catalog for inclusion in the AI suggestion prompt."""
    lines = [
        "Available managed-exit strategies (the bot runtime will apply these automatically once the limit fills):",
    ]
    for sid, cfg in AI_EXIT_STRATEGIES.items():
        defaults = cfg.get("defaults") or {}
        default_str = ", ".join(f"{k}={v}" for k, v in defaults.items()) if defaults else "—"
        lines.append(f'  - "{sid}" — {cfg["label"]}. {cfg["description"]} Defaults: {default_str}.')
    lines.append("")
    lines.append(
        f'Default choice: "{DEFAULT_AI_EXIT_STRATEGY}" (TP1 + BE + HWM tick trail). '
        "Use it UNLESS you determine another strategy is a better fit for the current "
        "setup (e.g., choppy trend -> tp1_be_m5_trail, tight scalp -> tp1_be_only, "
        'pure set-and-forget -> "none"). You may override any default parameter values '
        'via the exit_params object. If you choose "none" or believe no managed exit is '
        "appropriate, explain why in the rationale."
    )
    return "\n".join(lines)


def normalize_exit_strategy(sid: str | None) -> str:
    """Return a valid strategy ID, falling back to the default."""
    if not sid:
        return DEFAULT_AI_EXIT_STRATEGY
    sid = str(sid).strip().lower()
    return sid if sid in AI_EXIT_STRATEGIES else DEFAULT_AI_EXIT_STRATEGY


def merge_exit_params(sid: str, overrides: dict[str, Any] | None) -> dict[str, Any]:
    """Merge user/AI overrides onto the strategy defaults."""
    cfg = AI_EXIT_STRATEGIES.get(sid) or AI_EXIT_STRATEGIES[DEFAULT_AI_EXIT_STRATEGY]
    merged = dict(cfg.get("defaults") or {})
    if overrides:
        for k, v in overrides.items():
            if v is None:
                continue
            try:
                merged[k] = float(v)
            except (TypeError, ValueError):
                continue
    return merged


def trail_mode_for_strategy(sid: str) -> str:
    cfg = AI_EXIT_STRATEGIES.get(sid) or AI_EXIT_STRATEGIES[DEFAULT_AI_EXIT_STRATEGY]
    return str(cfg.get("trail_mode") or "none")


def compute_post_tp1_stop(
    *,
    trade_side: str,
    entry_price: float,
    pip_size: float,
    tp1_pips: float,
    spread_pips: float = 0.0,
    be_plus_pips: float = 0.5,
    lock_in_fraction: float = 0.2,
) -> float:
    tp1_distance = max(0.0, float(tp1_pips)) * float(pip_size)
    fractional_lock = max(0.0, float(lock_in_fraction)) * tp1_distance
    lock_distance = fractional_lock + max(0.0, float(be_plus_pips)) * float(pip_size)
    if str(trade_side).lower() == "sell":
        return round(float(entry_price) - lock_distance, 3)
    return round(float(entry_price) + lock_distance, 3)


def compute_post_tp1_trail_sl(
    *,
    trade_side: str,
    entry_price: float,
    tp1_pips: float,
    pip_size: float,
    high_water_mark: float,
    lock_sl: float,
    trail_fraction: float = 0.40,
    extended_threshold: float = 1.5,
    extended_trail_fraction: float = 0.30,
) -> float:
    """Compute a proportional post-TP1 runner trail, floored/capped at the lock SL."""
    tp1_distance = max(0.0, float(tp1_pips)) * float(pip_size)
    if tp1_distance <= 0:
        return round(float(lock_sl), 3)

    side = str(trade_side).lower()
    if side == "sell":
        hwm_distance = float(entry_price) - float(high_water_mark)
    else:
        hwm_distance = float(high_water_mark) - float(entry_price)

    effective_fraction = float(trail_fraction)
    if hwm_distance > tp1_distance * float(extended_threshold):
        effective_fraction = float(extended_trail_fraction)

    trail_distance = tp1_distance * max(0.0, effective_fraction)

    if side == "sell":
        trail_sl = float(high_water_mark) + trail_distance
        return round(min(trail_sl, float(lock_sl)), 3)

    trail_sl = float(high_water_mark) - trail_distance
    return round(max(trail_sl, float(lock_sl)), 3)
