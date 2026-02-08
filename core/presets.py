"""Trading presets (templates) that can be applied to a ProfileV1.

Each preset is a partial patch of ProfileV1 fields. When applied, only the fields
defined in the preset are overwritten; the rest of the profile stays intact.
"""
from __future__ import annotations

from copy import deepcopy
from enum import Enum
from typing import Any, Optional

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
    M5_M15_MOMENTUM_PULLBACK = "m5_m15_momentum_pullback"
    DAY_TRADING_USD_JPY = "day_trading_usd_jpy"
    DAY_TRADING_USD_JPY_9_21 = "day_trading_usd_jpy_9_21"
    M5_M15_MOMENTUM_PULLBACK_9_21 = "m5_m15_momentum_pullback_9_21"
    CODY_WIZARD_AGGR_OANDA_M5_CROSS = "cody_wizard_aggressive_oanda_m5_cross"
    KUMA_TORA_BB_EMA_9_21_SCALPER = "kuma_tora_bb_ema_9_21_scalper"
    KT_CG_COUNTER_TREND_PULLBACK = "kt_cg_counter_trend_pullback"


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
            "min_stop_pips": 12.0,
            "max_spread_pips": 5.0,
            "max_trades_per_day": 20,
            "max_open_trades": 2,
        },
        "strategy": {
            "filters": {
                "alignment": {"enabled": False},
                "ema_stack_filter": {"enabled": False},
                "atr_filter": {"enabled": True, "timeframe": "M1", "min_atr_pips": 5.0},
                "session_filter": {"enabled": True, "sessions": ["London", "NewYork"]},
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
                "pips_default": 18.0,
            },
            "breakeven": {"enabled": True, "after_pips": 8.0},
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
            "min_stop_pips": 12.0,
            "max_spread_pips": 5.0,
            "max_trades_per_day": 5,
            "max_open_trades": 1,
        },
        "strategy": {
            "filters": {
                "alignment": {"enabled": True, "method": "score", "min_score_to_trade": 1},
                "ema_stack_filter": {"enabled": True, "timeframe": "M1", "min_separation_pips": 1.0},
                "atr_filter": {"enabled": True, "timeframe": "M1", "min_atr_pips": 5.0},
                "session_filter": {"enabled": True, "sessions": ["London", "NewYork"]},
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
                "pips_default": 18.0,
            },
            "breakeven": {"enabled": True, "after_pips": 8.0},
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
            "min_stop_pips": 12.0,
            "max_spread_pips": 5.0,
            "max_trades_per_day": 5,
            "max_open_trades": 2,
            "cooldown_minutes_after_loss": 0,
        },
        "strategy": {
            "filters": {
                "alignment": {"enabled": False},
                "ema_stack_filter": {"enabled": False},
                "atr_filter": {"enabled": True, "timeframe": "M15", "min_atr_pips": 8.0},
            },
            "setups": {
                "m1_cross_entry": {"enabled": False},
            },
        },
        "trade_management": {
            "target": {
                "mode": "fixed_pips",
                "pips_default": 18.0,
            },
            "breakeven": {"enabled": True, "after_pips": 8.0},
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
                    "tp_pips": 18.0,
                    "sl_pips": 12.0,
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
                    "tp_pips": 18.0,
                    "sl_pips": 12.0,
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
            "max_spread_pips": 5.0,
            "max_trades_per_day": 5,
            "max_open_trades": 1,
            "cooldown_minutes_after_loss": 5,
        },
        "strategy": {
            "filters": {
                "alignment": {"enabled": False},
                "ema_stack_filter": {"enabled": False},
                "atr_filter": {"enabled": True, "timeframe": "M15", "min_atr_pips": 8.0},
            },
            "setups": {
                "m1_cross_entry": {"enabled": False},
            },
        },
        "trade_management": {
            "target": {
                "mode": "fixed_pips",
                "pips_default": 18.0,
            },
            "breakeven": {"enabled": True, "after_pips": 8.0},
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
                    "tp_pips": 18.0,
                    "sl_pips": 12.0,
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
                    "tp_pips": 18.0,
                    "sl_pips": 12.0,
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
            "min_stop_pips": 12.0,
            "max_spread_pips": 5.0,
            "max_trades_per_day": 8,
            "max_open_trades": 2,
        },
        "strategy": {
            "filters": {
                "alignment": {"enabled": False},
                "ema_stack_filter": {"enabled": False},
                "atr_filter": {"enabled": True, "timeframe": "M15", "min_atr_pips": 8.0},
                "session_filter": {"enabled": True, "sessions": ["London", "NewYork"]},
            },
            "setups": {
                "m1_cross_entry": {"enabled": False},
            },
        },
        "trade_management": {
            "target": {
                "mode": "fixed_pips",
                "pips_default": 18.0,
            },
            "breakeven": {"enabled": True, "after_pips": 8.0},
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
                    "tp_pips": 18.0,
                    "sl_pips": 12.0,
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
                    "tp_pips": 18.0,
                    "sl_pips": 12.0,
                },
            ],
        },
    },
    # -----------------------------------------------------------------------
    # VWAP TREND (Slope + Session + Zone)
    # Entries based on price vs VWAP with slope filter, optional session filter, no-trade zone
    # -----------------------------------------------------------------------
    PresetId.VWAP_TREND: {
        "name": "VWAP Trend (Slope + Session + Zone)",
        "description": "VWAP-based entries with slope filter (buy only when VWAP slope positive, sell when negative), optional London+NY session filter, and a small no-trade zone around VWAP so entries are outside the band.",
        "pros": [
            "VWAP slope filter aligns trades with VWAP direction; fewer counter-trend entries",
            "Optional session filter (London + NY) limits trading to liquid hours",
            "No-trade zone avoids entries right at VWAP, reducing chop",
            "Cross triggers give clear entry signals; good for intraday trend following",
        ],
        "cons": [
            "VWAP is cumulative per session - best used within session context",
            "Session filter reduces opportunity count outside London/NY",
            "Requires volume data (tick volume used when available)",
        ],
        "risk": {
            "max_lots": 0.1,
            "require_stop": True,
            "min_stop_pips": 12.0,
            "max_spread_pips": 5.0,
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
                "pips_default": 18.0,
            },
            "breakeven": {"enabled": True, "after_pips": 8.0},
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
                    "tp_pips": 18.0,
                    "sl_pips": 12.0,
                    "use_slope_filter": True,
                    "vwap_slope_lookback_bars": 20,
                    "session_filter_enabled": True,
                    "no_trade_zone_pips": 1.5,
                },
                {
                    "type": "vwap",
                    "id": "vwap_cross_sell",
                    "enabled": True,
                    "timeframe": "M15",
                    "trigger": "cross_below",
                    "side": "sell",
                    "tp_pips": 18.0,
                    "sl_pips": 12.0,
                    "use_slope_filter": True,
                    "vwap_slope_lookback_bars": 20,
                    "session_filter_enabled": True,
                    "no_trade_zone_pips": 1.5,
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
            "min_stop_pips": 12.0,
            "max_spread_pips": 5.0,
            "max_trades_per_day": 10,
            "max_open_trades": 2,
        },
        "strategy": {
            "filters": {
                "alignment": {"enabled": True, "method": "score", "min_score_to_trade": 1},
                "ema_stack_filter": {"enabled": False},
                "atr_filter": {"enabled": True, "timeframe": "M1", "min_atr_pips": 5.0},
                "session_filter": {"enabled": True, "sessions": ["London", "NewYork"]},
            },
            "setups": {
                "m1_cross_entry": {
                    "enabled": True,
                    "timeframe": "M1",
                    "confirmation": {
                        "confirm_bars": 2,
                        "max_wait_bars": 6,
                        "require_close_on_correct_side": True,
                        "min_distance_pips": 1.5,
                    },
                },
            },
        },
        "trade_management": {
            "target": {
                "mode": "fixed_pips",
                "pips_default": 18.0,
            },
            "breakeven": {"enabled": True, "after_pips": 8.0},
        },
        "execution": {
            "loop_poll_seconds": 5.0,
            "loop_poll_seconds_fast": 2.0,
            "policies": [
                {"type": "confirmed_cross", "id": "trend_cross", "enabled": True, "setup_id": "m1_cross_entry"},
            ],
        },
    },
    # -----------------------------------------------------------------------
    # M5 to M15 Momentum Pullback (~4 pip spread)
    # Trend from EMA 50/200, entry on pullback to EMA 20-50 zone
    # -----------------------------------------------------------------------
    PresetId.M5_M15_MOMENTUM_PULLBACK: {
        "description": "M5–M15 momentum pullback for ~4 pip spread: trend from EMA 50/200, entry on pullback to EMA 20–50 zone.",
        "pros": [
            "Suited to wider spread (up to 4 pips); TP 20–40, SL 12–20",
            "Trend from M15 EMA 50/200; entries on M5 pullback to EMA 20–50 zone",
            "Up to 20 trades/day; good for momentum pullback style",
        ],
        "cons": [
            "Requires clear trend; chop can produce false zone touches",
            "M5/M15 only – no M1; fewer signals than scalping presets",
        ],
        "risk": {
            "max_lots": 0.1,
            "require_stop": True,
            "min_stop_pips": 12.0,
            "max_spread_pips": 5.0,
            "max_trades_per_day": 20,
            "max_open_trades": 2,
        },
        "strategy": {
            "filters": {
                "alignment": {"enabled": False},
                "ema_stack_filter": {"enabled": False},
                "atr_filter": {"enabled": True, "timeframe": "M15", "min_atr_pips": 10.0},
                "session_filter": {"enabled": True, "sessions": ["London", "NewYork"]},
            },
            "setups": {
                "m1_cross_entry": {"enabled": False},
            },
        },
        "trade_management": {
            "target": {
                "mode": "fixed_pips",
                "pips_default": 30.0,
            },
            "breakeven": {"enabled": True, "after_pips": 8.0},
        },
        "execution": {
            "loop_poll_seconds": 5.0,
            "loop_poll_seconds_fast": 2.0,
            "policies": [
                {
                    "type": "ema_pullback",
                    "id": "m5_m15_pullback",
                    "enabled": True,
                    "trend_timeframe": "M15",
                    "entry_timeframe": "M5",
                    "ema_trend_fast": 50,
                    "ema_trend_slow": 200,
                    "ema_zone_low": 20,
                    "ema_zone_high": 50,
                    "tp_pips": 30.0,
                    "sl_pips": 16.0,
                },
            ],
        },
    },
    # -----------------------------------------------------------------------
    # M5 to M15 Momentum Pullback (EMA 9–21 Zone) - simpler, more frequent entries
    # Fork of M5_M15_MOMENTUM_PULLBACK with 9/21 pullback zone + OANDA spread.
    # -----------------------------------------------------------------------
    PresetId.M5_M15_MOMENTUM_PULLBACK_9_21: {
        "description": "M5–M15 momentum pullback for OANDA-wide spread (~4.5 pips): trend from EMA 50/200, entry on pullback to the faster EMA 9–21 zone (more frequent entries than 20–50).",
        "pros": [
            "EMA 9–21 zone tends to produce more entries than 20–50",
            "Simple fixed target sizing (30 pips default) with defined SL",
            "Optional filters (session + ATR) enabled to reduce chop and protect win-rate",
        ],
        "cons": [
            "Faster zone can overtrade in choppy conditions; relies on filters",
            "Requires clear trend; spread still matters (OANDA)",
        ],
        "risk": {
            "max_lots": 0.1,
            "require_stop": True,
            "min_stop_pips": 12.0,
            "max_spread_pips": 5.0,
            "max_trades_per_day": 20,
            "max_open_trades": 2,
        },
        "strategy": {
            "filters": {
                "alignment": {"enabled": False},
                "ema_stack_filter": {"enabled": False},
                "atr_filter": {"enabled": True, "timeframe": "M15", "min_atr_pips": 10.0},
                "session_filter": {"enabled": True, "sessions": ["London", "NewYork"]},
            },
            "setups": {
                "m1_cross_entry": {"enabled": False},
            },
        },
        "trade_management": {
            "target": {
                "mode": "fixed_pips",
                "pips_default": 30.0,
            },
            "breakeven": {"enabled": True, "after_pips": 8.0},
        },
        "execution": {
            "loop_poll_seconds": 5.0,
            "loop_poll_seconds_fast": 2.0,
            "policies": [
                {
                    "type": "ema_pullback",
                    "id": "m5_m15_pullback_9_21",
                    "enabled": True,
                    "trend_timeframe": "M15",
                    "entry_timeframe": "M5",
                    "ema_trend_fast": 50,
                    "ema_trend_slow": 200,
                    "ema_zone_low": 9,
                    "ema_zone_high": 21,
                    "tp_pips": 30.0,
                    "sl_pips": 16.0,
                },
            ],
        },
    },
    # -----------------------------------------------------------------------
    # Cody Wizard Aggressive (OANDA) M5 EMA Cross
    # Aggressive day trading preset tuned for OANDA ~4 pip spread.
    # Uses M5 confirmed-cross (EMA13/SMA30) with M15 trend/filters.
    # -----------------------------------------------------------------------
    PresetId.CODY_WIZARD_AGGR_OANDA_M5_CROSS: {
        "name": "Cody Wizard Aggressive (OANDA) M5 EMA Cross",
        "description": "Aggressive day trading preset for USD/JPY on OANDA (~4 pip spread): entries from M5 EMA13/SMA30 confirmed crosses, trading only during London+NY sessions with M15 trend/ATR filters. Uses 15 pip SL/TP and up to 30 trades/day, max 3 open.",
        "pros": [
            "M5 confirmed-cross entries provide frequent intraday opportunities",
            "M15 trend + ATR + session filters help maintain win-rate despite wide spread",
            "Fixed 15 pip SL/TP keeps risk and targets simple",
        ],
        "cons": [
            "Aggressive trade count (up to 30/day) can increase drawdowns if filters are disabled",
            "Wide spread means small moves can be eaten by costs; best in active London/NY hours",
        ],
        "risk": {
            # Max lots is set very high so effective max is capped by Profile Editor limits.
            "max_lots": 999.0,
            "require_stop": True,
            "min_stop_pips": 15.0,
            "max_spread_pips": 5.0,
            "max_trades_per_day": 30,
            "max_open_trades": 3,
            "cooldown_minutes_after_loss": 0,
        },
        "strategy": {
            "filters": {
                "alignment": {"enabled": True, "trend_timeframe": "M15"},
                "ema_stack_filter": {"enabled": False},
                "atr_filter": {"enabled": True, "timeframe": "M15", "min_atr_pips": 8.0},
                "session_filter": {"enabled": True, "sessions": ["London", "NewYork"]},
            },
            "setups": {
                "m1_cross_entry": {
                    "enabled": False,
                },
                "m5_cross_entry": {
                    "enabled": True,
                    "timeframe": "M5",
                    "ema": 13,
                    "sma": 30,
                    "confirmation": {
                        "confirm_bars": 2,
                        "max_wait_bars": 5,
                        "require_close_on_correct_side": True,
                        "min_distance_pips": 1.0,
                    },
                },
            },
        },
        "trade_management": {
            "target": {
                "mode": "fixed_pips",
                "pips_default": 18.0,
            },
            "breakeven": {"enabled": True, "after_pips": 8.0},
        },
        "execution": {
            "loop_poll_seconds": 5.0,
            "loop_poll_seconds_fast": 2.0,
            "policies": [
                {
                    "type": "confirmed_cross",
                    "id": "wizard_m5_cross_oanda",
                    "enabled": True,
                    "setup_id": "m5_cross_entry",
                },
            ],
        },
    },
    # -----------------------------------------------------------------------
    # KumaTora Bollinger Expansion Scalper (EMA 9/21)
    # OANDA-tuned M1 scalper: EMA 9/21 + Bollinger expansion + ATR filter.
    # -----------------------------------------------------------------------
    PresetId.KUMA_TORA_BB_EMA_9_21_SCALPER: {
        "name": "KumaTora Bollinger Expansion Scalper (EMA 9/21)",
        "description": "M1 USD/JPY scalper for OANDA (~4 pip spread): EMA 9/21 trend with Bollinger Band expansion and ATR filters, 10 pip SL and 15 pip TP, tuned for London+NY sessions.",
        "pros": [
            "EMA 9/21 trend + pullback entries capture short-term momentum moves on M1",
            "Bollinger expansion and ATR filters help avoid dead/choppy periods",
            "Risk is simple and consistent: 10 pip SL, 15 pip TP",
        ],
        "cons": [
            "Aggressive trade limits (many trades/day) can increase drawdown in difficult markets",
            "Requires sufficient volatility; filters may skip trades during quiet hours",
        ],
        "risk": {
            "max_lots": 0.1,
            "require_stop": True,
            "min_stop_pips": 12.0,
            "max_spread_pips": 5.0,
            "max_trades_per_day": 30,
            "max_open_trades": 2,
        },
        "strategy": {
            "filters": {
                "alignment": {"enabled": False},
                "ema_stack_filter": {"enabled": True, "timeframe": "M1", "periods": [9, 21]},
                "atr_filter": {"enabled": True, "timeframe": "M1", "min_atr_pips": 5.0},
                "session_filter": {"enabled": True, "sessions": ["London", "NewYork"]},
            },
        },
        "trade_management": {
            # Use policy.sl_pips (12) and risk.min_stop_pips (12) for SL sizing.
            # Target uses fixed_pips with pips_default.
            "target": {"mode": "fixed_pips", "pips_default": 18.0},
            "breakeven": {"enabled": True, "after_pips": 8.0},
        },
        "execution": {
            "loop_poll_seconds": 1.0,
            "loop_poll_seconds_fast": 0.5,
            "policies": [
                {
                    "type": "ema_bb_scalp",
                    "id": "usd_jpy_m1_kumatora",
                    "enabled": True,
                    "timeframe": "M1",
                    "ema_fast": 9,
                    "ema_slow": 21,
                    "bollinger_period": 20,
                    "bollinger_deviation": 2.0,
                    "tp_pips": 18.0,
                    "sl_pips": 12.0,
                    "confirm_bars": 2,
                    "min_distance_pips": 1.0,
                },
            ],
        },
    },
    # -----------------------------------------------------------------------
    # KT/CG Counter-Trend Pullback (Trial #3)
    # M5 trend from EMA 9/21, M1 zone entry, M1 pullback cross.
    # -----------------------------------------------------------------------
    PresetId.KT_CG_COUNTER_TREND_PULLBACK: {
        "name": "KT/CG Counter-Trend Pullback",
        "description": "Counter-trend pullback strategy: M5 trend from EMA 9/21, M1 zone entry when EMA9 vs EMA13 agrees with trend, M1 pullback cross when EMA9 crosses EMA15 in counter-trend direction. Auto-closes opposite trades before new entries.",
        "pros": [
            "Multi-timeframe confirmation: M5 trend + M1 zone + M1 pullback cross",
            "Counter-trend pullback entries often catch reversions to mean",
            "Auto-close opposite trades ensures clean direction switches",
            "Temporary EMA customization via UI for real-time tuning",
        ],
        "cons": [
            "Counter-trend entries can be risky in strong trends",
            "Requires M5 and M1 data alignment",
            "May miss entries if zone or cross conditions are too strict",
        ],
        "risk": {
            "max_lots": 0.1,
            "require_stop": True,
            "min_stop_pips": 10.0,
            "max_spread_pips": 5.0,
            "max_trades_per_day": 20,
            "max_open_trades": 2,
        },
        "strategy": {
            "filters": {
                "alignment": {"enabled": False},
                "ema_stack_filter": {"enabled": False},
                "atr_filter": {"enabled": True, "timeframe": "M1", "min_atr_pips": 5.0},
                "session_filter": {"enabled": True, "sessions": ["London", "NewYork"]},
            },
        },
        "trade_management": {
            "target": {"mode": "fixed_pips", "pips_default": 15.0},
            "breakeven": {"enabled": True, "after_pips": 8.0},
        },
        "execution": {
            "loop_poll_seconds": 3.0,
            "loop_poll_seconds_fast": 1.0,
            "policies": [
                {
                    "type": "kt_cg_counter_trend_pullback",
                    "id": "kt_cg_ctp_default",
                    "enabled": True,
                    "m5_trend_ema_fast": 9,
                    "m5_trend_ema_slow": 21,
                    "m1_zone_entry_ema_slow": 13,
                    "m1_pullback_cross_ema_slow": 15,
                    "close_opposite_on_trade": True,
                    "tp_pips": 15.0,
                    "sl_pips": 10.0,
                    "confirm_bars": 1,
                },
            ],
        },
    },
    # -----------------------------------------------------------------------
    # Day Trading USD/JPY (uncle preset: OANDA ~4.5 pip spread, scaled target, breakeven)
    # -----------------------------------------------------------------------
    PresetId.DAY_TRADING_USD_JPY: {
        "name": "Day Trading USD/JPY",
        "description": "Day trading USD/JPY with M15 trend, M5 pullback, strict filters, ATR-based SL, scaled target (TP1 partial + runner), and breakeven. Suited to OANDA spread (~4.5 pips).",
        "pros": [
            "Alignment and EMA stack (20/50/200) on M15; ATR and session filters reduce bad entries",
            "ATR-based stop (1.3x, max 20 pips); breakeven at 14 pips; TP1 at 18 pips (50% close), remainder runner",
            "Rejection candle and engulfing confirmation; min RR 2.3; avoid round numbers",
        ],
        "cons": [
            "Wider spread (4.5 pips) requires larger targets; fewer trades (max 6/day, 1 open)",
        ],
        "risk": {
            "max_lots": 0.03,
            "require_stop": True,
            "min_stop_pips": 12.0,
            "max_spread_pips": 5.0,
            "max_trades_per_day": 10,
            "max_open_trades": 1,
            "risk_per_trade_pct": 0.4,
        },
        "strategy": {
            "filters": {
                "alignment": {"enabled": True, "trend_timeframe": "M15"},
                "ema_stack_filter": {"enabled": True, "timeframe": "M15", "periods": [20, 50]},
                "atr_filter": {"enabled": True, "timeframe": "M15", "min_atr_pips": 10.0},
                "session_filter": {"enabled": True, "sessions": ["Tokyo", "London", "NewYork"]},
            },
        },
        "trade_management": {
            "stop_loss": {"mode": "atr", "atr_multiplier": 1.3, "max_sl_pips": 20.0},
            "breakeven": {"enabled": True, "after_pips": 8.0},
            "target": {
                "mode": "scaled",
                "tp1_pips": 18.0,
                "tp1_close_percent": 50.0,
                "tp2_mode": "runner",
                "trail_after_tp1": True,
                "trail_type": "ema",
                "trail_ema": 20,
            },
        },
        "execution": {
            "loop_poll_seconds": 5.0,
            "loop_poll_seconds_fast": 1.0,
            "policies": [
                {
                    "type": "ema_pullback",
                    "id": "usdjpy_m5_m15_pullback",
                    "enabled": True,
                    "trend_timeframe": "M15",
                    "entry_timeframe": "M5",
                    "ema_trend_fast": 50,
                    "ema_trend_slow": 200,
                    "ema_zone_low": 20,
                    "ema_zone_high": 50,
                    "require_rejection_candle": True,
                    "require_engulfing_confirmation": False,
                    "min_rr": 1.5,
                    "avoid_round_numbers": True,
                    "round_number_buffer_pips": 5.0,
                    "tp_pips": 18.0,
                },
            ],
        },
    },
    # -----------------------------------------------------------------------
    # Day Trading USD/JPY (EMA 9–21 Pullback) - more signals, win-rate focused
    # Fork of DAY_TRADING_USD_JPY with 9/21 pullback zone.
    # -----------------------------------------------------------------------
    PresetId.DAY_TRADING_USD_JPY_9_21: {
        "name": "Day Trading USD/JPY (EMA 9–21 Pullback)",
        "description": "Day trading USD/JPY with M15 trend and M5 pullback entries in the EMA 9–21 zone (more frequent entries than 20–50). Strict filters + ATR-based SL + scaled target + breakeven. Tuned for OANDA spread (~4.5 pips).",
        "pros": [
            "EMA 9–21 pullback zone tends to trigger earlier/more often than 20–50",
            "Keeps session + ATR + EMA-stack + alignment-by-trend filters to protect win-rate",
            "Scaled target with earlier breakeven aims to lock in wins in choppier pullbacks",
        ],
        "cons": [
            "Shallower pullback zone can overtrade chop if filters are disabled",
            "Wider spread (4.5 pips) still requires meaningful targets; fewer trades than tight-spread brokers",
        ],
        "risk": {
            "max_lots": 0.03,
            "require_stop": True,
            "min_stop_pips": 12.0,
            "max_spread_pips": 5.0,
            "max_trades_per_day": 12,
            "max_open_trades": 1,
            "risk_per_trade_pct": 0.4,
        },
        "strategy": {
            "filters": {
                "alignment": {"enabled": True, "trend_timeframe": "M15"},
                "ema_stack_filter": {"enabled": True, "timeframe": "M15", "periods": [20, 50]},
                "atr_filter": {"enabled": True, "timeframe": "M15", "min_atr_pips": 10.0},
                "session_filter": {"enabled": True, "sessions": ["Tokyo", "London", "NewYork"]},
            },
        },
        "trade_management": {
            "stop_loss": {"mode": "atr", "atr_multiplier": 1.3, "max_sl_pips": 20.0},
            "breakeven": {"enabled": True, "after_pips": 8.0},
            "target": {
                "mode": "scaled",
                "tp1_pips": 18.0,
                "tp1_close_percent": 50.0,
                "tp2_mode": "runner",
                "trail_after_tp1": True,
                "trail_type": "ema",
                "trail_ema": 20,
            },
        },
        "execution": {
            "loop_poll_seconds": 5.0,
            "loop_poll_seconds_fast": 1.0,
            "policies": [
                {
                    "type": "ema_pullback",
                    "id": "usdjpy_m5_m15_pullback_9_21",
                    "enabled": True,
                    "trend_timeframe": "M15",
                    "entry_timeframe": "M5",
                    "ema_trend_fast": 50,
                    "ema_trend_slow": 200,
                    "ema_zone_low": 9,
                    "ema_zone_high": 21,
                    "require_rejection_candle": True,
                    # Balanced default: keep rejection candle, loosen engulfing for more trades
                    "require_engulfing_confirmation": False,
                    # Win-rate focus: min_rr uses tp1_pips when scaled; keep realistic given ATR SL/min_stop_pips
                    "min_rr": 1.5,
                    "avoid_round_numbers": True,
                    "round_number_buffer_pips": 5.0,
                    # Used for min_rr calculations (scaled mode sets TP=None on initial order)
                    "tp_pips": 18.0,
                },
            ],
        },
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def list_presets() -> list[dict[str, Any]]:
    """Return a list of preset metadata (id, name, description, pros, cons)."""
    result = []
    for preset_id, patch in PRESETS.items():
        result.append({
            "id": preset_id.value,
            "name": patch.get("name") or preset_id.name.replace("_", " ").title(),
            "description": patch.get("description", ""),
            "pros": patch.get("pros", []),
            "cons": patch.get("cons", []),
        })
    return result


def get_preset_patch(preset_id: PresetId | str) -> dict[str, Any]:
    """Return the raw patch dict for a preset (excluding metadata like name, description, pros, cons)."""
    if isinstance(preset_id, str):
        preset_id = PresetId(preset_id)
    patch = deepcopy(PRESETS[preset_id])
    # Remove metadata fields that aren't part of profile config
    patch.pop("name", None)
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
    """Build effective_risk from preset risk, capped by profile limits (Profile Editor).
    Max-style fields are capped by profile; min_stop_pips uses at least the profile value
    so the user can set a higher min stop (e.g. >10 for scalping) in the Profile Editor.
    """
    eff = deepcopy(patch_risk)
    cap_fields = ["max_lots", "max_spread_pips", "max_trades_per_day", "max_open_trades"]
    for k in cap_fields:
        if k in limits and k in eff:
            try:
                eff[k] = min(float(eff[k]), float(limits[k]))
            except (TypeError, ValueError):
                pass
    # Profile Editor min_stop_pips is the floor: use at least that so user can set >10 for scalping
    if "min_stop_pips" in limits and "min_stop_pips" in eff:
        try:
            profile_min = float(limits["min_stop_pips"])
            preset_min = float(eff["min_stop_pips"])
            eff["min_stop_pips"] = max(preset_min, profile_min)
        except (TypeError, ValueError):
            pass
    return eff


def apply_preset(profile: ProfileV1, preset_id: PresetId | str, patch_overrides: Optional[dict[str, Any]] = None) -> ProfileV1:
    """Apply a preset patch on top of an existing profile (returns new ProfileV1).

    Profile Editor risk (profile.risk) is never overwritten. Preset risk is stored as
    effective_risk (capped by profile.risk) and used at runtime.

    If patch_overrides is provided, it is merged into the preset patch before applying.
    For vwap_trend, options may include execution.policies_overlay_for_vwap: dict of
    fields to apply to each policy with type "vwap" (e.g. session_filter_enabled).
    """
    if isinstance(preset_id, str):
        preset_id = PresetId(preset_id)
    patch = get_preset_patch(preset_id)
    # Apply patch_overrides: merge into patch; handle policies_overlay_for_vwap for vwap policies
    if patch_overrides:
        exec_opts = patch_overrides.get("execution")
        if isinstance(exec_opts, dict):
            overlay = exec_opts.get("policies_overlay_for_vwap")
            if overlay and isinstance(overlay, dict):
                for p in patch.get("execution", {}).get("policies", []):
                    if isinstance(p, dict) and p.get("type") == "vwap":
                        for k, v in overlay.items():
                            p[k] = v
        for key, value in patch_overrides.items():
            if key == "execution":
                if not isinstance(value, dict):
                    continue
                rest = {k: deepcopy(v) for k, v in value.items() if k != "policies_overlay_for_vwap"}
                if rest:
                    _deep_merge(patch.setdefault("execution", {}), rest)
            elif key in patch and isinstance(patch.get(key), dict) and isinstance(value, dict):
                patch[key] = _deep_merge(patch[key], value)
            else:
                patch[key] = deepcopy(value)
    base_dict = profile.model_dump()
    limits = base_dict.get("risk", {})
    merged = _deep_merge(base_dict, patch)
    # Keep profile editor risk limits unchanged (only editable in Profile Editor)
    merged["risk"] = deepcopy(limits)
    # Broker connection: never overwrite with preset (only editable in Profile Editor)
    for key in ("broker_type", "oanda_token", "oanda_account_id", "oanda_environment"):
        if key in base_dict:
            merged[key] = deepcopy(base_dict[key])
    # Display/account fields: keep from profile
    for key in ("display_currency", "deposit_amount", "leverage_ratio", "created_utc"):
        if key in base_dict:
            merged[key] = deepcopy(base_dict[key])
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
