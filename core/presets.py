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
    KT_CG_TRIAL_BLOW_ACCOUNT = "kt_cg_trial_blow_account"
    KT_CG_TRIAL_2 = "kt_cg_trial_2"
    KT_CG_TRIAL_3 = "kt_cg_trial_3"
    KT_CG_TRIAL_4 = "kt_cg_trial_4"
    KT_CG_TRIAL_5 = "kt_cg_trial_5"
    KT_CG_TRIAL_6 = "kt_cg_trial_6"
    KT_CG_TRIAL_7 = "kt_cg_trial_7"
    M5_M1_EMA_CROSS_9_21 = "m5_m1_ema_cross_9_21"


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
                        "confirm_bars": 2,
                        "max_wait_bars": 4,
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
            "loop_poll_seconds": 8.0,
            "loop_poll_seconds_fast": 3.0,
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
                    "use_macd_cross": True,
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
                    "use_macd_cross": True,
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
            "loop_poll_seconds": 3.0,
            "loop_poll_seconds_fast": 1.0,
            "policies": [
                {
                    "type": "vwap",
                    "id": "vwap_cross",
                    "enabled": True,
                    "timeframe": "M5",
                    "trigger": "cross_above",
                    "side": "buy",
                    "tp_pips": 18.0,
                    "sl_pips": 12.0,
                    "use_slope_filter": True,
                    "vwap_slope_lookback_bars": 12,
                    "session_filter_enabled": True,
                    "no_trade_zone_pips": 1.5,
                },
                {
                    "type": "vwap",
                    "id": "vwap_cross_sell",
                    "enabled": True,
                    "timeframe": "M5",
                    "trigger": "cross_below",
                    "side": "sell",
                    "tp_pips": 18.0,
                    "sl_pips": 12.0,
                    "use_slope_filter": True,
                    "vwap_slope_lookback_bars": 12,
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
                "alignment": {"enabled": True, "trend_timeframe": "M15"},
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
            "min_stop_pips": 12.0,
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
    # -----------------------------------------------------------------------
    # KT/CG Trial #1 "Blow Account" - Aggressive hybrid entry testing preset
    # Bull: zone entry (price in EMA 9-21). Bear: cross entry (EMA 9 < EMA 21).
    # -----------------------------------------------------------------------
    PresetId.KT_CG_TRIAL_BLOW_ACCOUNT: {
        "name": "KT/CG Trial #1 Blow Account",
        "description": "Aggressive M1 scalper with M15 trend filter. Bull: buy when EMA 9 is above EMA 21. Bear: sell when EMA 9 is below EMA 21. High frequency, no filters, testing preset.",
        "pros": [
            "Very fast entries on M1 with immediate execution",
            "High trade frequency (up to 100/day)",
            "No filters = maximum opportunity count",
            "24/5 trading across all sessions",
        ],
        "cons": [
            "Poor R:R (0.25:1) requires very high win rate",
            "Wide SL (20 pips) vs tight TP (5 pips)",
            "No session filter = trades during low liquidity",
            "Testing preset - expect significant drawdowns",
        ],
        "risk": {
            "max_lots": 0.1,
            "require_stop": True,
            "min_stop_pips": 20.0,
            "max_spread_pips": 6.0,
            "max_trades_per_day": 100,
            "max_open_trades": 20,
            "cooldown_minutes_after_loss": 0,
        },
        "strategy": {
            "filters": {
                "alignment": {"enabled": False},
                "ema_stack_filter": {"enabled": False},
                "atr_filter": {"enabled": False},
                "session_filter": {"enabled": False},
            },
            "setups": {
                "m1_cross_entry": {"enabled": False},
            },
        },
        "trade_management": {
            "target": {
                "mode": "fixed_pips",
                "pips_default": 0.5,
            },
            "breakeven": {"enabled": False},
        },
        "execution": {
            "loop_poll_seconds": 1.0,
            "loop_poll_seconds_fast": 0.5,
            "policies": [
                {
                    "type": "kt_cg_hybrid",
                    "id": "kt_cg_trial_1",
                    "enabled": True,
                    "trend_timeframe": "M15",
                    "entry_timeframe": "M1",
                    "ema_fast": 9,
                    "ema_slow": 21,
                    "tp_pips": 0.5,
                    "sl_pips": 20.0,
                    "cooldown_minutes": 2.0,
                },
            ],
        },
    },
    # -----------------------------------------------------------------------
    # KT/CG Trial #2 (Zone Entry + Pullback Cross)
    # Two independent entry triggers:
    # 1. Zone Entry (continuous, respects cooldown): M1 EMA9 vs EMA21
    # 2. Pullback Cross (discrete, overrides cooldown): M1 EMA9 crosses EMA13
    # Direction switch close: closes opposite positions before new trade.
    # -----------------------------------------------------------------------
    PresetId.KT_CG_TRIAL_2: {
        "name": "KT/CG Trial #2 (Zone Entry + Pullback Cross)",
        "description": "Two independent triggers: Zone Entry (M1 EMA 9 vs 21, respects cooldown) OR Pullback Cross (M1 EMA 9 crosses 13, overrides cooldown). Direction switch close enabled.",
        "pros": [
            "Zone Entry trades with trend when EMAs aligned",
            "Pullback Cross catches counter-trend pullbacks",
            "Pullback Cross overrides cooldown for faster re-entry",
            "Direction switch close prevents hedged positions",
            "Swing filter avoids entries near key reversal zones",
        ],
        "cons": [
            "May enter too early if pullback continues",
            "Testing preset - expect significant drawdowns",
        ],
        "risk": {
            "max_lots": 0.1,
            "require_stop": True,
            "min_stop_pips": 20.0,
            "max_spread_pips": 6.0,
            "max_trades_per_day": 100,
            "max_open_trades": 20,
            "cooldown_minutes_after_loss": 0,
        },
        "strategy": {
            "filters": {
                "alignment": {"enabled": False},
                "ema_stack_filter": {"enabled": False},
                "atr_filter": {"enabled": False},
                "session_filter": {"enabled": False},
            },
            "setups": {},
        },
        "trade_management": {
            "target": {
                "mode": "fixed_pips",
                "pips_default": 0.5,
            },
            "stop_loss": None,
            "breakeven": {"enabled": False},
        },
        "execution": {
            "loop_poll_seconds": 1.0,
            "loop_poll_seconds_fast": 0.5,
            "policies": [
                {
                    "type": "kt_cg_hybrid",
                    "id": "kt_cg_trial_2",
                    "enabled": True,
                    # M5 Trend EMAs
                    "m5_trend_ema_fast": 9,
                    "m5_trend_ema_slow": 21,
                    # M1 Zone Entry - EMA 9 vs EMA 21 (respects cooldown)
                    "m1_zone_entry_ema_slow": 21,
                    # M1 Pullback Cross - EMA 9 crosses EMA 13 (overrides cooldown)
                    "m1_pullback_cross_ema_slow": 13,
                    # Close opposite trades before placing new trade
                    "close_opposite_on_trade": True,
                    # Cooldown (Zone Entry respects, Pullback Cross overrides)
                    "cooldown_minutes": 3.0,
                    "tp_pips": 0.5,
                    "sl_pips": 20.0,
                    "confirm_bars": 1,
                    # Swing level proximity filter
                    "swing_level_filter_enabled": True,
                    "swing_lookback_bars": 100,
                    "swing_confirmation_bars": 5,
                    "swing_danger_zone_pct": 0.15,
                },
            ],
        },
    },
    # -----------------------------------------------------------------------
    # KT/CG Trial #3 (Zone Entry + Pullback Cross)
    # Two independent entry triggers:
    # 1. Zone Entry (continuous, respects cooldown): M1 EMA9 vs EMA13
    # 2. Pullback Cross (discrete, overrides cooldown): M1 EMA9 crosses EMA15
    # Direction switch close: closes opposite positions before new trade.
    # -----------------------------------------------------------------------
    PresetId.KT_CG_TRIAL_3: {
        "name": "KT/CG Trial #3 (Counter-Trend Pullback)",
        "description": "Two independent triggers: Zone Entry (M1 EMA 9 vs 13, respects cooldown) OR Pullback Cross (M1 EMA 9 crosses 15, overrides cooldown). Direction switch close enabled.",
        "pros": [
            "Zone Entry trades with trend when EMAs aligned",
            "Pullback Cross catches counter-trend pullbacks",
            "Pullback Cross overrides cooldown for faster re-entry",
            "Direction switch close prevents hedged positions",
            "Swing filter avoids entries near key reversal zones",
        ],
        "cons": [
            "May enter too early if pullback continues",
            "Testing preset - expect significant drawdowns",
        ],
        "risk": {
            "max_lots": 0.1,
            "require_stop": True,
            "min_stop_pips": 20.0,
            "max_spread_pips": 6.0,
            "max_trades_per_day": 100,
            "max_open_trades": 20,
            "cooldown_minutes_after_loss": 0,
        },
        "strategy": {
            "filters": {
                "alignment": {"enabled": False},
                "ema_stack_filter": {"enabled": False},
                "atr_filter": {"enabled": False},
                "session_filter": {"enabled": False},
            },
            "setups": {},
        },
        "trade_management": {
            "target": {
                "mode": "fixed_pips",
                "pips_default": 0.5,
            },
            "stop_loss": None,
            "breakeven": {"enabled": False},
        },
        "execution": {
            "loop_poll_seconds": 1.0,
            "loop_poll_seconds_fast": 0.5,
            "policies": [
                {
                    "type": "kt_cg_counter_trend_pullback",
                    "id": "kt_cg_trial_3",
                    "enabled": True,
                    # M5 Trend EMAs (default 9/21)
                    "m5_trend_ema_fast": 9,
                    "m5_trend_ema_slow": 21,
                    # M1 Zone Entry - EMA 9 vs EMA 13 (respects cooldown)
                    "m1_zone_entry_ema_slow": 13,
                    # M1 Pullback Cross - EMA 9 crosses EMA 15 (overrides cooldown)
                    "m1_pullback_cross_ema_slow": 15,
                    # Close opposite trades before placing new trade
                    "close_opposite_on_trade": True,
                    # Cooldown (Zone Entry respects, Pullback Cross overrides)
                    "cooldown_minutes": 3.0,
                    "tp_pips": 0.5,
                    "sl_pips": 20.0,
                    "confirm_bars": 1,
                    # Swing level proximity filter
                    "swing_level_filter_enabled": True,
                    "swing_lookback_bars": 100,
                    "swing_confirmation_bars": 5,
                    "swing_danger_zone_pct": 0.15,
                },
            ],
        },
    },
    # -----------------------------------------------------------------------
    # KT/CG Trial #4 (M3 Trend + Tiered Pullback System)
    # Uses M3 for trend detection (faster updates) and tiered pullback entries.
    # Tiered Pullback: price touches M1 EMA 9/11/13/15/17 triggers entry.
    # -----------------------------------------------------------------------
    PresetId.KT_CG_TRIAL_4: {
        "name": "KT/CG Trial #4 (M3 Trend + Tiered Pullback)",
        "description": "Tiered Pullback: When M3 BULL and live price touches M1 EMA 9/11/12/13/14/15/16/17, triggers BUY. When M3 BEAR and price touches tier EMAs, triggers SELL. Each tier fires once per touch and resets when price moves away.",
        "pros": [
            "M3 updates every 3 minutes (faster than M5's 5 minutes)",
            "8 tiered pullback levels = multiple entry opportunities per pullback",
            "Each tier fires independently (no cooldown for tiered entries)",
            "Tier resets when price moves away - allows re-entry on deep pullbacks",
        ],
        "cons": [
            "Higher trade frequency = more spread costs",
            "More whipsaw risk in choppy markets",
            "False signals in consolidation zones",
            "Testing preset - expect significant drawdowns",
        ],
        "risk": {
            "max_lots": 0.1,
            "require_stop": True,
            "min_stop_pips": 20.0,
            "max_spread_pips": 6.0,
            "max_trades_per_day": 100,
            "max_open_trades": 20,
            "cooldown_minutes_after_loss": 0,
        },
        "strategy": {
            "filters": {
                "alignment": {"enabled": False},
                "ema_stack_filter": {"enabled": False},
                "atr_filter": {"enabled": False},
                "session_filter": {"enabled": False},
            },
            "setups": {},
        },
        "trade_management": {
            "target": {
                "mode": "fixed_pips",
                "pips_default": 0.5,
            },
            "stop_loss": None,
            "breakeven": {"enabled": False},
        },
        "execution": {
            "loop_poll_seconds": 0.25,
            "loop_poll_seconds_fast": 0.25,
            "policies": [
                {
                    "type": "kt_cg_trial_4",
                    "id": "kt_cg_trial_4",
                    "enabled": True,
                    # M3 Trend EMAs (default 5/9)
                    "m3_trend_ema_fast": 5,
                    "m3_trend_ema_slow": 9,
                    # M1 Zone Entry - EMA5 vs EMA9 (respects cooldown)
                    "zone_entry_enabled": True,
                    "m1_zone_entry_ema_fast": 5,
                    "m1_zone_entry_ema_slow": 9,
                    # Tiered Pullback Configuration
                    "tiered_pullback_enabled": True,
                    "tier_ema_periods": [9, 11, 12, 13, 14, 15, 16, 17],
                    "tier_reset_buffer_pips": 1.0,
                    # Close opposite trades before placing new trade
                    "close_opposite_on_trade": True,
                    # Cooldown (Zone Entry respects this, Tiered Pullback has NO cooldown)
                    "cooldown_minutes": 3.0,
                    "tp_pips": 0.5,
                    "sl_pips": 20.0,
                    "confirm_bars": 1,
                    # Rolling Danger Zone Filter (M1-based)
                    "rolling_danger_zone_enabled": True,
                    "rolling_danger_lookback_bars": 100,
                    "rolling_danger_zone_pct": 0.15,
                    # RSI Divergence Detection (M5-based, rolling window) - disabled by default
                    "rsi_divergence_enabled": False,
                    "rsi_divergence_period": 14,
                    "rsi_divergence_lookback_bars": 50,
                    "rsi_divergence_block_minutes": 5.0,
                    # Tiered ATR(14) Filter (replaces generic ATR for Trial #4)
                    "tiered_atr_filter_enabled": True,
                    "tiered_atr_block_below_pips": 4.0,
                    "tiered_atr_allow_all_max_pips": 12.0,
                    "tiered_atr_pullback_only_max_pips": 15.0,
                    # Daily High/Low Filter
                    "daily_hl_filter_enabled": False,
                    "daily_hl_buffer_pips": 5.0,
                    # Spread-Aware Breakeven
                    "spread_aware_be_enabled": False,
                    "spread_aware_be_trigger_mode": "fixed_pips",
                    "spread_aware_be_fixed_trigger_pips": 5.0,
                    "spread_aware_be_spread_buffer_pips": 1.0,
                    "spread_aware_be_apply_to_zone_entry": True,
                    "spread_aware_be_apply_to_tiered_pullback": True,
                    # EMA Zone Entry Filter (blocks zone entries during EMA compression)
                    "ema_zone_filter_enabled": True,
                    "ema_zone_filter_lookback_bars": 3,
                    "ema_zone_filter_block_threshold": 0.35,
                },
            ],
        },
    },
    # -----------------------------------------------------------------------
    # KT/CG Trial #5 (Dual ATR Filter with Session-Dynamic Thresholds)
    # Copy of Trial #4 with upgraded ATR filter system:
    # - M1 ATR(7) with per-session dynamic thresholds
    # - M3 ATR(14) with simple configurable range
    # -----------------------------------------------------------------------
    PresetId.KT_CG_TRIAL_5: {
        "name": "KT/CG Trial #5 (Fresh Cross + Exhaustion + Extended Tiers)",
        "description": "Overhauled Trial #5: Fresh Cross replaces cooldown, Trend Extension Exhaustion prevents chasing, tiers extended to EMA 34, dead zone 21:00-02:00 UTC, expanded EMA Zone Filter.",
        "pros": [
            "Fresh Cross prevents re-entry during unbroken runs",
            "Trend Exhaustion blocks entries during extended moves",
            "Deep tiers (EMA 18-34) catch meaningful pullbacks",
            "Fully configurable EMA Zone Filter weights",
        ],
        "cons": [
            "More filters = fewer trade signals",
            "Fresh Cross may delay entry after choppy crosses",
            "Testing preset - expect tuning needed for threshold values",
        ],
        "risk": {
            "max_lots": 0.1,
            "require_stop": True,
            "min_stop_pips": 20.0,
            "max_spread_pips": 6.0,
            "max_trades_per_day": 100,
            "max_open_trades": 20,
            "cooldown_minutes_after_loss": 0,
        },
        "strategy": {
            "filters": {
                "alignment": {"enabled": False},
                "ema_stack_filter": {"enabled": False},
                "atr_filter": {"enabled": False},
                "session_filter": {"enabled": False},
            },
            "setups": {},
        },
        "trade_management": {
            "target": {
                "mode": "fixed_pips",
                "pips_default": 0.5,
            },
            "stop_loss": None,
            "breakeven": {"enabled": False},
        },
        "execution": {
            "loop_poll_seconds": 0.25,
            "loop_poll_seconds_fast": 0.25,
            "policies": [
                {
                    "type": "kt_cg_trial_5",
                    "id": "kt_cg_trial_5",
                    "enabled": True,
                    # M3 Trend EMAs (default 5/9)
                    "m3_trend_ema_fast": 5,
                    "m3_trend_ema_slow": 9,
                    # M1 Zone Entry - EMA5 vs EMA9 (hardcoded)
                    "zone_entry_enabled": True,
                    "m1_zone_entry_ema_fast": 5,
                    "m1_zone_entry_ema_slow": 9,
                    # Tiered Pullback Configuration (default: 18, 21, 25, 29, 34)
                    "tiered_pullback_enabled": True,
                    "tier_ema_periods": [18, 21, 25, 29, 34],
                    "tier_reset_buffer_pips": 3.0,
                    # Close opposite trades before placing new trade
                    "close_opposite_on_trade": True,
                    # Cooldown REMOVED (replaced by Fresh Cross)
                    "cooldown_minutes": 0.0,
                    "tp_pips": 0.5,
                    "sl_pips": 20.0,
                    "confirm_bars": 1,
                    # --- Trial #5 Dual ATR Filter ---
                    # M1 ATR(7) - Session-Dynamic (master on/off)
                    "m1_atr_filter_enabled": True,
                    "m1_atr_period": 7,
                    "m1_atr_min_pips": 2.5,
                    "session_dynamic_atr_enabled": True,
                    "auto_session_detection_enabled": True,
                    "m1_atr_tokyo_min_pips": 3.0,
                    "m1_atr_london_min_pips": 3.0,
                    "m1_atr_ny_min_pips": 3.5,
                    # M3 ATR(14) - Simple Range
                    "m3_atr_filter_enabled": True,
                    "m3_atr_period": 14,
                    "m3_atr_min_pips": 5.0,
                    "m3_atr_max_pips": 16.0,
                    # M1 ATR(7) Session-Dynamic MAX thresholds
                    "m1_atr_max_pips": 11.0,
                    "m1_atr_tokyo_max_pips": 12.0,
                    "m1_atr_london_max_pips": 14.0,
                    "m1_atr_ny_max_pips": 16.0,
                    # Dead Zone (21:00-02:00 UTC)
                    "daily_reset_block_enabled": True,
                    # Daily High/Low Filter (blocks BOTH zone entry AND pullback)
                    "daily_hl_filter_enabled": True,
                    "daily_hl_buffer_pips": 15.0,
                    # Spread-Aware Breakeven (Trial #5: spread+buffer)
                    "spread_aware_be_enabled": False,
                    "spread_aware_be_trigger_mode": "spread_relative",
                    "spread_aware_be_fixed_trigger_pips": 5.0,
                    "spread_aware_be_spread_buffer_pips": 1.0,
                    "spread_aware_be_apply_to_zone_entry": True,
                    "spread_aware_be_apply_to_tiered_pullback": True,
                    # EMA Zone Entry Filter (fully configurable)
                    "ema_zone_filter_enabled": True,
                    "ema_zone_filter_lookback_bars": 3,
                    "ema_zone_filter_block_threshold": 0.35,
                    "ema_zone_filter_spread_weight": 0.45,
                    "ema_zone_filter_slope_weight": 0.40,
                    "ema_zone_filter_direction_weight": 0.15,
                    "ema_zone_filter_spread_min_pips": 0.0,
                    "ema_zone_filter_spread_max_pips": 4.0,
                    "ema_zone_filter_slope_min_pips": -1.0,
                    "ema_zone_filter_slope_max_pips": 3.0,
                    "ema_zone_filter_dir_min_pips": -3.0,
                    "ema_zone_filter_dir_max_pips": 3.0,
                    # Trend Extension Exhaustion
                    "trend_exhaustion_enabled": True,
                    "trend_exhaustion_fresh_max": 2.0,
                    "trend_exhaustion_mature_max": 3.5,
                    "trend_exhaustion_extended_max": 5.0,
                    "trend_exhaustion_ramp_minutes": 12.0,
                    # Per-direction open trade cap
                    "max_open_trades_per_side": 5,
                },
            ],
        },
    },
    # -----------------------------------------------------------------------
    # KT/CG Trial #6 (BB Slope Trend + EMA Tier Pullback + BB Reversal)
    # Clean rebuild: M3 slope-based trend gated by BB expansion,
    # EMA tier pullback with M1 BB gating, BB reversal counter-trend.
    # -----------------------------------------------------------------------
    PresetId.KT_CG_TRIAL_6: {
        "name": "KT/CG Trial #6 (BB Slope Trend + EMA Tier + BB Reversal)",
        "description": "Trial #6: M3 slope trend (EMA 5>9>21 + BB expanding), EMA tier pullback with M1 BB gating (System A), and BB reversal counter-trend entries (System B). Spread-aware BE with 7p trigger.",
        "pros": [
            "M3 slope + BB expansion = strong trend confirmation before entries",
            "12 EMA tiers (9-21) with BB gating prevents overtrading in extended moves",
            "BB reversal catches mean-reversion at Bollinger extremes",
            "Configurable dead zone blocks low-liquidity hours",
        ],
        "cons": [
            "More complex system = more parameters to tune",
            "NONE trend blocks all entries; may miss early moves",
            "BB reversal counter-trend carries inherent risk",
            "Testing preset - expect tuning needed",
        ],
        "risk": {
            "max_lots": 0.1,
            "require_stop": True,
            "min_stop_pips": 10.0,
            "max_spread_pips": 6.0,
            "max_trades_per_day": 60,
            "max_open_trades": 10,
            "cooldown_minutes_after_loss": 3,
        },
        "strategy": {
            "filters": {
                "alignment": {"enabled": False},
                "ema_stack_filter": {"enabled": False},
                "atr_filter": {"enabled": False},
                "session_filter": {"enabled": False},
            },
            "setups": {},
        },
        "trade_management": {
            "target": {
                "mode": "fixed_pips",
                "pips_default": 7.0,
            },
            "stop_loss": None,
            "breakeven": {"enabled": False},
        },
        "execution": {
            "loop_poll_seconds": 0.25,
            "loop_poll_seconds_fast": 0.25,
            "policies": [
                {
                    "type": "kt_cg_trial_6",
                    "id": "kt_cg_trial_6",
                    "enabled": True,
                    # M3 Slope Trend Engine
                    "m3_trend_ema_fast": 5,
                    "m3_trend_ema_slow": 9,
                    "m3_trend_ema_extra": 21,
                    "m3_slope_lookback": 2,
                    "m3_bb_period": 20,
                    "m3_bb_std": 2.0,
                    # M1 Bollinger Bands
                    "m1_bb_period": 20,
                    "m1_bb_std": 2.0,
                    # System A: EMA Tier Pullback
                    "ema_tier_enabled": True,
                    "tier_ema_periods": [9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
                    "tier_reset_buffer_pips": 1.0,
                    "ema_tier_tp_pips": 7.0,
                    "sl_pips": 10.0,
                    "bb_gating_deep_tier_min_period": 21,
                    # System B: BB Reversal
                    "bb_reversal_enabled": True,
                    "bb_reversal_start_offset_pips": 0.5,
                    "bb_reversal_increment_pips": 0.5,
                    "bb_reversal_num_tiers": 10,
                    "max_bb_reversal_positions": 3,
                    "bb_reversal_tp_mode": "middle_bb_entry",
                    "bb_reversal_tp_fixed_pips": 8.0,
                    "bb_reversal_tp_min_pips": 4.0,
                    "bb_reversal_tp_max_pips": 15.0,
                    "bb_reversal_sl_pips": 10.0,
                    # Dead Zone
                    "dead_zone_enabled": True,
                    "dead_zone_start_hour_utc": 21,
                    "dead_zone_end_hour_utc": 2,
                    # Risk
                    "cooldown_after_loss_seconds": 180.0,
                    "close_opposite_on_trade": False,
                    "max_open_trades_per_side": 15,
                    # Spread-Aware Breakeven (Spread + Buffer only; no fixed trigger)
                    "spread_aware_be_enabled": True,
                    "spread_aware_be_trigger_mode": "spread_relative",
                    "spread_aware_be_fixed_trigger_pips": 7.0,
                    "spread_aware_be_spread_buffer_pips": 1.5,
                    "spread_aware_be_apply_to_ema_tier": True,
                    "spread_aware_be_apply_to_bb_reversal": True,
                },
            ],
        },
    },
    # -----------------------------------------------------------------------
    # KT/CG Trial #7 Recover Account (M5 Trend + Tiered Pullback + Slope Gate)
    # Clone of Trial #4 with:
    # - M5 EMA 9/21 trend basis
    # - M1 tiered pullback levels configurable across EMA 9-34
    # - Slope-only EMA zone filter (disabled by default)
    # - Per-side + per-entry-type open trade caps
    # -----------------------------------------------------------------------
    PresetId.KT_CG_TRIAL_7: {
        "name": "KT/CG Trial #7 Recover Account (M5 Trend + Tiered Pullback)",
        "description": "Trial #7 recover variant: M5 EMA9/21 trend with M1 zone entry + tiered pullback across EMA 9-34. Slope-only EMA zone filter for anti-chop gating on zone entries.",
        "pros": [
            "M5 EMA9/21 trend gives smoother directional bias than M3",
            "Tiered pullback supports the full EMA 9-34 ladder",
            "Simple slope gate can block choppy/sideways zone entries",
            "Independent caps for side, zone entries, and tiered pullbacks",
        ],
        "cons": [
            "Higher TP target can reduce hit-rate vs scalp settings",
            "No ATR/danger/divergence/daily-HL filters in this variant",
            "Many tiers can increase frequency if caps are too loose",
            "Testing preset - tune slope thresholds per session volatility",
        ],
        "risk": {
            "max_lots": 0.1,
            "require_stop": True,
            "min_stop_pips": 20.0,
            "max_spread_pips": 6.0,
            "max_trades_per_day": 100,
            "max_open_trades": 20,
            "cooldown_minutes_after_loss": 0,
        },
        "strategy": {
            "filters": {
                "alignment": {"enabled": False},
                "ema_stack_filter": {"enabled": False},
                "atr_filter": {"enabled": False},
                "session_filter": {"enabled": False},
            },
            "setups": {},
        },
        "trade_management": {
            "target": {
                "mode": "fixed_pips",
                "pips_default": 4.0,
            },
            "stop_loss": None,
            "breakeven": {"enabled": False},
        },
        "execution": {
            "loop_poll_seconds": 0.25,
            "loop_poll_seconds_fast": 0.25,
            "policies": [
                {
                    "type": "kt_cg_trial_7",
                    "id": "kt_cg_trial_7",
                    "enabled": True,
                    # M5 Trend EMAs
                    "m5_trend_ema_fast": 9,
                    "m5_trend_ema_slow": 21,
                    # M1 Zone Entry - EMA5 vs EMA9
                    "zone_entry_enabled": True,
                    "m1_zone_entry_ema_fast": 5,
                    "m1_zone_entry_ema_slow": 9,
                    # Tiered Pullback Configuration (EMA 9-34)
                    "tiered_pullback_enabled": True,
                    "tier_ema_periods": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
                    "tier_reset_buffer_pips": 1.0,
                    # Core controls from Trial #4
                    "close_opposite_on_trade": True,
                    "cooldown_minutes": 3.0,
                    "tp_pips": 4.0,
                    "sl_pips": 20.0,
                    "confirm_bars": 1,
                    # Spread-Aware Breakeven (kept for compatibility)
                    "spread_aware_be_enabled": False,
                    "spread_aware_be_trigger_mode": "fixed_pips",
                    "spread_aware_be_fixed_trigger_pips": 5.0,
                    "spread_aware_be_spread_buffer_pips": 1.0,
                    "spread_aware_be_apply_to_zone_entry": True,
                    "spread_aware_be_apply_to_tiered_pullback": True,
                    # EMA Zone Entry Filter (slope-only, disabled by default)
                    "ema_zone_filter_enabled": False,
                    "ema_zone_filter_lookback_bars": 3,
                    "ema_zone_filter_ema5_min_slope_pips_per_bar": 0.10,
                    "ema_zone_filter_ema9_min_slope_pips_per_bar": 0.08,
                    "ema_zone_filter_ema21_min_slope_pips_per_bar": 0.05,
                    # Open trade caps
                    "max_open_trades_per_side": 5,
                    "max_zone_entry_open": 3,
                    "max_tiered_pullback_open": 8,
                },
            ],
        },
    },
    # -----------------------------------------------------------------------
    # M5/M1 EMA Cross 9/21 (Experimental)
    # Every M5 EMA 9/21 cross triggers a trade. M1 EMA state determines direction/TP.
    # Fixed 20 pip SL, hedging allowed.
    # -----------------------------------------------------------------------
    PresetId.M5_M1_EMA_CROSS_9_21: {
        "name": "M5/M1 EMA Cross 9/21 (Experimental)",
        "description": "Experimental scalping: every M5 EMA 9/21 cross triggers a trade. Direction from M1 EMA position, TP scaled by M1 momentum + M5 BB width. Fixed 20 pip SL, hedging allowed.",
        "pros": [
            "High frequency - trades every M5 cross",
            "Dynamic TP based on momentum",
            "Hedging allowed",
        ],
        "cons": [
            "Experimental - not backtested",
            "Many small trades",
            "No trend filtering",
        ],
        "risk": {
            "max_lots": 0.05,
            "require_stop": True,
            "min_stop_pips": 10.0,
            "max_spread_pips": 8.0,
            "max_trades_per_day": 100,
            "max_open_trades": 20,
            "cooldown_minutes_after_loss": 0,
        },
        "strategy": {
            "filters": {
                "alignment": {"enabled": False},
                "ema_stack_filter": {"enabled": False},
                "atr_filter": {"enabled": False},
                "session_filter": {"enabled": False},
            },
            "setups": {},
        },
        "trade_management": {
            "target": {"mode": "fixed_pips", "pips_default": 2.0},
            "stop_loss": None,  # SL is set by the policy (sl_pips: 20.0)
            "breakeven": {"enabled": False},
        },
        "execution": {
            "loop_poll_seconds": 3.0,
            "loop_poll_seconds_fast": 1.0,
            "policies": [
                {
                    "type": "m5_m1_ema_cross",
                    "id": "m5_m1_cross_main",
                    "enabled": True,
                    "m5_ema_fast": 9,
                    "m5_ema_slow": 21,
                    "m1_ema_fast": 9,
                    "m1_ema_slow": 21,
                    "m1_slope_lookback": 5,
                    "strong_slope_threshold": 0.3,
                    "moderate_slope_threshold": 0.15,
                    "weak_slope_threshold": 0.05,
                    "cross_history_count": 5,
                    "use_history_for_tp": True,
                    # Cross quality scoring - sharp crosses get higher TP targets
                    "use_cross_quality": True,
                    "cross_quality_lookback": 3,
                    "sharp_cross_threshold": 0.5,
                    # Bollinger Bands
                    "bb_period": 20,
                    "bb_std_dev": 2.0,
                    "bb_thin_threshold": 12.0,
                    "bb_wide_threshold": 35.0,
                    # TP targets (strong = 6-10 pips to let winners run)
                    "tp_strong": 8.0,
                    "tp_moderate": 4.0,
                    "tp_weak": 2.0,
                    "tp_flat": 1.0,
                    "tp_min": 0.5,
                    "tp_max": 12.0,
                    "tp_spread_buffer": 0.5,
                    # Risk management
                    "sl_pips": 20.0,
                    "lots": 0.02,  # base lot size
                    # Momentum-based position sizing
                    "use_momentum_sizing": True,
                    "lots_multiplier_strong": 1.0,  # full size for strong momentum
                    "lots_multiplier_moderate": 0.75,  # 75% for moderate
                    "lots_multiplier_weak": 0.5,  # 50% for weak/flat
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
