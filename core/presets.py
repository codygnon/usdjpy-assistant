"""Trading presets (templates) that can be applied to a ProfileV1.

Each preset is a partial patch of ProfileV1 fields. When applied, only the fields
defined in the preset are overwritten; the rest of the profile stays intact.
"""
from __future__ import annotations

from copy import deepcopy
from enum import Enum
from typing import Any

from .profile import (
    ExecutionPolicyConfirmedCross,
    ExecutionPolicyIndicator,
    ProfileV1,
    save_profile_v1,
)


class PresetId(str, Enum):
    """Available preset identifiers."""

    AGGRESSIVE_SCALPING = "aggressive_scalping"
    CONSERVATIVE_SCALPING = "conservative_scalping"
    AGGRESSIVE_SWING = "aggressive_swing"
    CONSERVATIVE_SWING = "conservative_swing"
    MEAN_REVERSION_DIP = "mean_reversion_dip"
    VWAP_TREND = "vwap_trend"
    TREND_CONTINUATION = "trend_continuation"


# ---------------------------------------------------------------------------
# Preset definitions
# ---------------------------------------------------------------------------
# Each preset is a nested dict that mirrors the ProfileV1 structure.
# Only include fields you want to change; others remain untouched.
# ---------------------------------------------------------------------------

PRESETS: dict[PresetId, dict[str, Any]] = {
    # -----------------------------------------------------------------------
    # AGGRESSIVE SCALPING
    # Fast M1 trades, looser filters, higher trade count, tighter TP/SL
    # -----------------------------------------------------------------------
    PresetId.AGGRESSIVE_SCALPING: {
        "description": "Fast M1 scalping with loose filters and tight targets.",
        "pros": [
            "High opportunity count (up to 20 trades/day)",
            "Very responsive to short-term moves",
            "Tight TP (5 pips) gives quick wins in active sessions",
            "Can catch small momentum bursts",
        ],
        "cons": [
            "No higher-timeframe alignment - more trades in choppy markets",
            "Sensitive to spread and execution quality (tiny TP/SL)",
            "Intraday P&L will be more volatile",
            "Requires active market hours for best results",
        ],
        "risk": {
            "max_lots": 0.1,
            "require_stop": True,
            "min_stop_pips": 5.0,
            "max_spread_pips": 3.0,
            "max_trades_per_day": 20,
            "max_open_trades": 2,
        },
        "strategy": {
            "filters": {
                "alignment": {"enabled": False},
                "ema_stack_filter": {"enabled": False},
                "atr_filter": {"enabled": False},
            },
            "setups": {
                "m1_cross_entry": {
                    "enabled": True,
                    "timeframe": "M1",
                    "confirmation": {
                        "confirm_bars": 1,
                        "max_wait_bars": 3,
                    },
                },
            },
        },
        "trade_management": {
            "target": {
                "mode": "fixed_pips",
                "pips_default": 5.0,
            },
        },
        "execution": {
            "loop_poll_seconds": 3.0,
            "loop_poll_seconds_fast": 1.0,
            "policies": [
                {"type": "confirmed_cross", "id": "scalp_cross", "enabled": True, "setup_id": "m1_cross_entry"},
            ],
        },
    },
    # -----------------------------------------------------------------------
    # CONSERVATIVE SCALPING
    # M1 trades but with stricter filters, lower trade count, require stop
    # -----------------------------------------------------------------------
    PresetId.CONSERVATIVE_SCALPING: {
        "description": "M1 scalping with strict spread/alignment filters.",
        "pros": [
            "Higher quality trades due to strict filters",
            "Lower risk per trade (0.05 lots, tight spread gate)",
            "Requires multi-timeframe alignment for entry",
            "Better suited for newer traders",
        ],
        "cons": [
            "Fewer trading opportunities (max 5/day)",
            "May miss moves when spread widens slightly",
            "Stricter confirmation can cause late entries",
            "Only 1 trade open at a time limits scaling",
        ],
        "risk": {
            "max_lots": 0.05,
            "require_stop": True,
            "min_stop_pips": 8.0,
            "max_spread_pips": 1.0,
            "max_trades_per_day": 5,
            "max_open_trades": 1,
        },
        "strategy": {
            "filters": {
                "alignment": {"enabled": True, "method": "score", "min_score_to_trade": 1},
                "ema_stack_filter": {"enabled": True, "timeframe": "M1", "min_separation_pips": 0.5},
                "atr_filter": {"enabled": False},
            },
            "setups": {
                "m1_cross_entry": {
                    "enabled": True,
                    "timeframe": "M1",
                    "confirmation": {
                        "confirm_bars": 2,
                        "max_wait_bars": 5,
                        "require_close_on_correct_side": True,
                    },
                },
            },
        },
        "trade_management": {
            "target": {
                "mode": "fixed_pips",
                "pips_default": 8.0,
            },
        },
        "execution": {
            "loop_poll_seconds": 5.0,
            "loop_poll_seconds_fast": 2.0,
            "policies": [
                {"type": "confirmed_cross", "id": "scalp_cross_cons", "enabled": True, "setup_id": "m1_cross_entry"},
            ],
        },
    },
    # -----------------------------------------------------------------------
    # AGGRESSIVE SWING (Bollinger mean-reversion, bi-directional)
    # Buy at lower band in uptrend, sell at upper band in downtrend
    # -----------------------------------------------------------------------
    PresetId.AGGRESSIVE_SWING: {
        "description": "Bi-directional Bollinger mean reversion: buys when price touches or is at/below the lower band in an uptrend, sells when price touches or is at/above the upper band in a downtrend (M15, 20-period, 2 std).",
        "pros": [
            "Mean reversion at lower band catches bounces in uptrends",
            "Mean reversion at upper band catches pullbacks in downtrends",
            "More entry possibilities in both trends",
            "Moderate targets (15 pips TP, 10 pips SL)",
        ],
        "cons": [
            "In strong trends price can stay outside the band",
            "Wider stops mean larger risk per trade",
            "Fewer signals than cross-based presets",
            "Requires patience for band touches",
        ],
        "risk": {
            "max_lots": 0.1,
            "require_stop": True,
            "min_stop_pips": 10.0,
            "max_spread_pips": 3.0,
            "max_trades_per_day": 5,
            "max_open_trades": 2,
            "cooldown_minutes_after_loss": 0,
        },
        "strategy": {
            "filters": {
                "alignment": {"enabled": False},
                "ema_stack_filter": {"enabled": False},
                "atr_filter": {"enabled": False},
            },
            "setups": {
                "m1_cross_entry": {"enabled": False},
            },
        },
        "trade_management": {
            "target": {
                "mode": "fixed_pips",
                "pips_default": 15.0,
            },
        },
        "execution": {
            "loop_poll_seconds": 10.0,
            "loop_poll_seconds_fast": 5.0,
            "policies": [
                {
                    "type": "bollinger_bands",
                    "id": "bb_lower_buy",
                    "enabled": True,
                    "timeframe": "M15",
                    "period": 20,
                    "std_dev": 2.0,
                    "trigger": "lower_band_buy",
                    "regime": "bull",
                    "side": "buy",
                    "tp_pips": 15.0,
                    "sl_pips": 10.0,
                },
                {
                    "type": "bollinger_bands",
                    "id": "bb_upper_sell",
                    "enabled": True,
                    "timeframe": "M15",
                    "period": 20,
                    "std_dev": 2.0,
                    "trigger": "upper_band_sell",
                    "regime": "bear",
                    "side": "sell",
                    "tp_pips": 15.0,
                    "sl_pips": 10.0,
                },
            ],
        },
    },
    # -----------------------------------------------------------------------
    # CONSERVATIVE SWING (Bollinger mean-reversion, bi-directional)
    # Buy at lower band in uptrend, sell at upper band in downtrend
    # -----------------------------------------------------------------------
    PresetId.CONSERVATIVE_SWING: {
        "description": "Bi-directional Bollinger mean reversion: buys when price touches or is at/below the lower band in an uptrend, sells when price touches or is at/above the upper band in a downtrend (M15, 20-period, 2 std). Conservative risk: lower size, fewer trades, wider stops, cooldown after loss.",
        "pros": [
            "Mean reversion at lower band catches bounces in uptrends",
            "Mean reversion at upper band catches pullbacks in downtrends",
            "More entry possibilities in both trends",
            "Conservative risk: 0.05 lots, 1 open trade, cooldown after loss",
        ],
        "cons": [
            "In strong trends price can stay outside the band",
            "Very few trades (max 3/day) - patience required",
            "Wider stops (12 pips) mean larger risk per trade",
            "Smaller position size limits profit potential",
        ],
        "risk": {
            "max_lots": 0.05,
            "require_stop": True,
            "min_stop_pips": 12.0,
            "max_spread_pips": 1.5,
            "max_trades_per_day": 3,
            "max_open_trades": 1,
            "cooldown_minutes_after_loss": 5,
        },
        "strategy": {
            "filters": {
                "alignment": {"enabled": False},
                "ema_stack_filter": {"enabled": False},
                "atr_filter": {"enabled": False},
            },
            "setups": {
                "m1_cross_entry": {"enabled": False},
            },
        },
        "trade_management": {
            "target": {
                "mode": "fixed_pips",
                "pips_default": 12.0,
            },
        },
        "execution": {
            "loop_poll_seconds": 15.0,
            "loop_poll_seconds_fast": 5.0,
            "policies": [
                {
                    "type": "bollinger_bands",
                    "id": "bb_upper_sell",
                    "enabled": True,
                    "timeframe": "M15",
                    "period": 20,
                    "std_dev": 2.0,
                    "trigger": "upper_band_sell",
                    "regime": "bear",
                    "side": "sell",
                    "tp_pips": 12.0,
                    "sl_pips": 10.0,
                },
                {
                    "type": "bollinger_bands",
                    "id": "bb_lower_buy",
                    "enabled": True,
                    "timeframe": "M15",
                    "period": 20,
                    "std_dev": 2.0,
                    "trigger": "lower_band_buy",
                    "regime": "bull",
                    "side": "buy",
                    "tp_pips": 12.0,
                    "sl_pips": 10.0,
                },
            ],
        },
    },
    # -----------------------------------------------------------------------
    # MEAN REVERSION (RSI)
    # Single preset: buy when RSI oversold (≤30) in bull regime; sell when RSI overbought (≥70) in bear regime.
    # -----------------------------------------------------------------------
    PresetId.MEAN_REVERSION_DIP: {
        "description": "Mean reversion using RSI: buys when RSI is oversold (≤30) in bull regime, sells when RSI is overbought (≥70) in bear regime. Trades only at these extremes; opportunities are rare since most of the time RSI is between 30 and 70.",
        "pros": [
            "Clear rules: only trades at RSI ≤30 (buy) or ≥70 (sell)",
            "Buys at discount in uptrends, sells at premium in downtrends",
            "Good risk/reward (12 pips TP vs 10 pips SL)",
            "Works well in ranging markets",
        ],
        "cons": [
            "Opportunities are less frequent than cross-based presets",
            "Requires patience for RSI to reach oversold or overbought",
            "May hold through deeper pullbacks or bounces",
            "No trade when RSI stays in the 30–70 zone",
        ],
        "risk": {
            "max_lots": 0.1,
            "require_stop": True,
            "min_stop_pips": 10.0,
            "max_spread_pips": 2.0,
            "max_trades_per_day": 8,
            "max_open_trades": 2,
        },
        "strategy": {
            "filters": {
                "alignment": {"enabled": False},
                "ema_stack_filter": {"enabled": False},
                "atr_filter": {"enabled": False},
            },
            "setups": {
                "m1_cross_entry": {"enabled": False},
            },
        },
        "trade_management": {
            "target": {
                "mode": "fixed_pips",
                "pips_default": 12.0,
            },
        },
        "execution": {
            "loop_poll_seconds": 5.0,
            "loop_poll_seconds_fast": 2.0,
            "policies": [
                {
                    "type": "indicator_based",
                    "id": "dip_buy",
                    "enabled": True,
                    "timeframe": "M15",
                    "regime": "bull",
                    "side": "buy",
                    "rsi_period": 14,
                    "rsi_oversold": 30.0,
                    "rsi_overbought": 70.0,
                    "rsi_zone": "oversold",
                    "use_macd_cross": False,
                    "tp_pips": 12.0,
                    "sl_pips": 10.0,
                },
                {
                    "type": "indicator_based",
                    "id": "dip_sell",
                    "enabled": True,
                    "timeframe": "M15",
                    "regime": "bear",
                    "side": "sell",
                    "rsi_period": 14,
                    "rsi_oversold": 30.0,
                    "rsi_overbought": 70.0,
                    "rsi_zone": "overbought",
                    "use_macd_cross": False,
                    "tp_pips": 12.0,
                    "sl_pips": 10.0,
                },
            ],
        },
    },
    # -----------------------------------------------------------------------
    # VWAP TREND
    # Entries based on price vs VWAP (cross above = buy, cross below = sell)
    # -----------------------------------------------------------------------
    PresetId.VWAP_TREND: {
        "description": "VWAP-based entries: buys when price crosses above VWAP, sells when price crosses below VWAP. Uses volume-weighted average price as a trend/reference level.",
        "pros": [
            "VWAP is a common institutional reference; trades with price vs VWAP",
            "Cross triggers give clear entry signals",
            "Good for intraday trend following",
            "Single policy can generate both buy and sell signals (via cross_above / cross_below)",
        ],
        "cons": [
            "VWAP is cumulative per session - best used within session context",
            "In choppy markets crosses can be frequent and whipsaw",
            "Requires volume data (tick volume used when available)",
            "No higher-timeframe filter in this preset",
        ],
        "risk": {
            "max_lots": 0.1,
            "require_stop": True,
            "min_stop_pips": 10.0,
            "max_spread_pips": 2.0,
            "max_trades_per_day": 8,
            "max_open_trades": 2,
        },
        "strategy": {
            "filters": {
                "alignment": {"enabled": False},
                "ema_stack_filter": {"enabled": False},
                "atr_filter": {"enabled": False},
            },
            "setups": {
                "m1_cross_entry": {"enabled": False},
            },
        },
        "trade_management": {
            "target": {
                "mode": "fixed_pips",
                "pips_default": 12.0,
            },
        },
        "execution": {
            "loop_poll_seconds": 5.0,
            "loop_poll_seconds_fast": 2.0,
            "policies": [
                {
                    "type": "vwap",
                    "id": "vwap_cross",
                    "enabled": True,
                    "timeframe": "M15",
                    "trigger": "cross_above",
                    "side": "buy",
                    "tp_pips": 12.0,
                    "sl_pips": 10.0,
                },
                {
                    "type": "vwap",
                    "id": "vwap_cross_sell",
                    "enabled": True,
                    "timeframe": "M15",
                    "trigger": "cross_below",
                    "side": "sell",
                    "tp_pips": 12.0,
                    "sl_pips": 10.0,
                },
            ],
        },
    },
    # -----------------------------------------------------------------------
    # TREND CONTINUATION
    # Confirmed cross with stricter confirmation settings
    # -----------------------------------------------------------------------
    PresetId.TREND_CONTINUATION: {
        "description": "Trade with trend: confirmed M1 cross + alignment filter.",
        "pros": [
            "Trades with the trend using alignment filter",
            "Confirmed cross reduces false signals",
            "Balanced settings - moderate risk and reward",
            "Good all-around strategy for trending markets",
        ],
        "cons": [
            "Requires 2 confirmation bars - may miss fast moves",
            "Alignment filter may block trades in choppy markets",
            "Medium TP (10 pips) in between scalp and swing",
            "Needs clear trend to perform well",
        ],
        "risk": {
            "max_lots": 0.1,
            "require_stop": True,
            "min_stop_pips": 10.0,
            "max_spread_pips": 2.0,
            "max_trades_per_day": 10,
            "max_open_trades": 2,
        },
        "strategy": {
            "filters": {
                "alignment": {"enabled": True, "method": "score", "min_score_to_trade": 1},
                "ema_stack_filter": {"enabled": False},
                "atr_filter": {"enabled": False},
            },
            "setups": {
                "m1_cross_entry": {
                    "enabled": True,
                    "timeframe": "M1",
                    "confirmation": {
                        "confirm_bars": 2,
                        "max_wait_bars": 6,
                        "require_close_on_correct_side": True,
                        "min_distance_pips": 0.5,
                    },
                },
            },
        },
        "trade_management": {
            "target": {
                "mode": "fixed_pips",
                "pips_default": 10.0,
            },
        },
        "execution": {
            "loop_poll_seconds": 5.0,
            "loop_poll_seconds_fast": 2.0,
            "policies": [
                {"type": "confirmed_cross", "id": "trend_cross", "enabled": True, "setup_id": "m1_cross_entry"},
            ],
        },
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def list_presets() -> list[dict[str, Any]]:
    """Return a list of preset metadata (id, description, pros, cons)."""
    result = []
    for preset_id, patch in PRESETS.items():
        result.append({
            "id": preset_id.value,
            "name": preset_id.name.replace("_", " ").title(),
            "description": patch.get("description", ""),
            "pros": patch.get("pros", []),
            "cons": patch.get("cons", []),
        })
    return result


def get_preset_patch(preset_id: PresetId | str) -> dict[str, Any]:
    """Return the raw patch dict for a preset (excluding metadata like description, pros, cons)."""
    if isinstance(preset_id, str):
        preset_id = PresetId(preset_id)
    patch = deepcopy(PRESETS[preset_id])
    # Remove metadata fields that aren't part of profile config
    patch.pop("description", None)
    patch.pop("pros", None)
    patch.pop("cons", None)
    return patch


def _deep_merge(base: dict, patch: dict) -> dict:
    """Recursively merge patch into base (patch wins on conflicts)."""
    result = deepcopy(base)
    for key, value in patch.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _effective_risk_capped_by_limits(patch_risk: dict[str, Any], limits: dict[str, Any]) -> dict[str, Any]:
    """Build effective_risk from preset risk, capped by profile limits (Profile Editor)."""
    eff = deepcopy(patch_risk)
    cap_fields = ["max_lots", "max_spread_pips", "max_trades_per_day", "max_open_trades"]
    for k in cap_fields:
        if k in limits and k in eff:
            try:
                eff[k] = min(float(eff[k]), float(limits[k]))
            except (TypeError, ValueError):
                pass
    return eff


def apply_preset(profile: ProfileV1, preset_id: PresetId | str) -> ProfileV1:
    """Apply a preset patch on top of an existing profile (returns new ProfileV1).

    Profile Editor risk (profile.risk) is never overwritten. Preset risk is stored as
    effective_risk (capped by profile.risk) and used at runtime.
    """
    if isinstance(preset_id, str):
        preset_id = PresetId(preset_id)
    patch = get_preset_patch(preset_id)
    base_dict = profile.model_dump()
    limits = base_dict.get("risk", {})
    merged = _deep_merge(base_dict, patch)
    # Keep profile editor risk limits unchanged (only editable in Profile Editor)
    merged["risk"] = deepcopy(limits)
    # Set effective_risk from preset, capped by profile limits
    if "risk" in patch and patch["risk"]:
        merged["effective_risk"] = _effective_risk_capped_by_limits(patch["risk"], limits)
    else:
        merged["effective_risk"] = None
    merged["active_preset_name"] = preset_id.value
    return ProfileV1.model_validate(merged)


def apply_and_save_preset(
    profile: ProfileV1,
    preset_id: PresetId | str,
    out_path: str,
) -> ProfileV1:
    """Apply preset and save to disk; returns the new profile."""
    new_profile = apply_preset(profile, preset_id)
    save_profile_v1(new_profile, out_path)
    return new_profile
