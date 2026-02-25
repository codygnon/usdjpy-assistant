from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, Field, ValidationError
from pydantic.config import ConfigDict


SchemaVersion = Literal[1]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RiskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_lots: float = 0.2
    require_stop: bool = True
    min_stop_pips: float = 10.0
    max_spread_pips: float = 5.0
    max_trades_per_day: int = 10

    # Optional advanced controls (v1)
    cooldown_minutes_after_loss: int = 0
    max_open_trades: int = 1
    risk_per_trade_pct: Optional[float] = None  # if set, sizing can use stop distance + equity
    max_daily_loss_pct: Optional[float] = None  # if set, can stop trading after reaching limit


class TimeframeIndicators(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ema_fast: int = 13
    sma_slow: int = 30
    ema_stack: Optional[list[int]] = None  # e.g. [8, 13, 21]


class ConfirmationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirm_bars: int = 1
    require_close_on_correct_side: bool = True
    min_distance_pips: float = 0.0
    max_wait_bars: int = 5


class CrossSetup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    timeframe: Literal["M1", "M5", "M15", "H4"] = "M1"
    ema: int = 13
    sma: int = 30
    confirmation: ConfirmationConfig = Field(default_factory=ConfirmationConfig)


class AlignmentFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    method: Literal["score", "strict"] = "score"
    weights: dict[Literal["H4", "M15", "M1"], int] = Field(default_factory=lambda: {"H4": 1, "M15": 1, "M1": 1})
    min_score_to_trade: int = -3
    trend_timeframe: Optional[Literal["M15", "H4", "M1"]] = None  # when set, alignment for non-cross policies = trend on this TF agrees with side


class EmaStackFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    timeframe: Literal["M1", "M15"] = "M1"
    periods: list[int] = Field(default_factory=lambda: [8, 13, 21])
    min_separation_pips: float = 0.0


class AtrFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    timeframe: Literal["M1", "M15"] = "M1"
    atr_period: int = 14
    min_atr_pips: float = 0.0
    max_atr_pips: Optional[float] = None


class SessionFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    sessions: list[Literal["Tokyo", "London", "NewYork"]] = Field(default_factory=lambda: ["Tokyo", "London", "NewYork"])


class SessionBoundaryBlockFilter(BaseModel):
    """Optional: block all entries 15 min before until 15 min after NY/London/Tokyo open and close."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    buffer_minutes: int = 15  # minutes before/after each open and close


class StrategyFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alignment: AlignmentFilter = Field(default_factory=AlignmentFilter)
    ema_stack_filter: EmaStackFilter = Field(default_factory=EmaStackFilter)
    atr_filter: AtrFilter = Field(default_factory=AtrFilter)
    session_filter: SessionFilter = Field(default_factory=SessionFilter)
    session_boundary_block: SessionBoundaryBlockFilter = Field(default_factory=SessionBoundaryBlockFilter)


class StrategyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timeframes: dict[Literal["M1", "M15", "H4"], TimeframeIndicators] = Field(
        default_factory=lambda: {
            "M1": TimeframeIndicators(ema_fast=13, sma_slow=30, ema_stack=[8, 13, 21]),
            "M15": TimeframeIndicators(ema_fast=13, sma_slow=30),
            "H4": TimeframeIndicators(ema_fast=13, sma_slow=30),
        }
    )

    setups: dict[str, CrossSetup] = Field(default_factory=lambda: {"m1_cross_entry": CrossSetup()})
    filters: StrategyFilters = Field(default_factory=StrategyFilters)


class TargetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["fixed_pips", "rr", "scaled"] = "fixed_pips"
    pips_default: float = 10.0
    rr_default: float = 1.0
    # scaled target: no TP on initial order; loop does partial close at TP1
    tp1_pips: Optional[float] = None
    tp1_close_percent: Optional[float] = None
    tp2_mode: Optional[str] = None  # e.g. "runner"
    trail_after_tp1: bool = False
    trail_type: Optional[str] = None  # e.g. "ema"
    trail_ema: Optional[int] = None


class StopLossConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["fixed_pips", "atr"] = "fixed_pips"
    atr_multiplier: float = 1.5
    max_sl_pips: float = 20.0


class BreakevenConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    after_pips: float = 10.0


class TradeManagementConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: TargetConfig = Field(default_factory=TargetConfig)
    stop_loss: Optional[StopLossConfig] = None  # when None, use policy sl_pips or min_stop_pips
    breakeven: Optional[BreakevenConfig] = None


# --- V1.1 Execution policies ---


class ExecutionPolicyConfirmedCross(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["confirmed_cross"] = "confirmed_cross"
    id: str = "confirmed_cross_default"
    enabled: bool = True
    setup_id: str = "m1_cross_entry"


class ExecutionPolicyPriceLevelTrend(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["price_level_trend"] = "price_level_trend"
    id: str = "price_level_trend_default"
    enabled: bool = False
    price_level: float = 0.0
    side: Literal["buy", "sell"] = "buy"
    tp_pips: float = 10.0
    sl_pips: Optional[float] = None
    trend_timeframes: list[Literal["M1", "M15", "H4"]] = Field(default_factory=lambda: ["M15", "M1"])
    trend_direction: Literal["bearish", "bullish"] = "bearish"
    max_wait_minutes: Optional[int] = None
    use_pending_order: bool = True


class ExecutionPolicyIndicator(BaseModel):
    """Indicator-based policy: RSI zone + regime (and optional MACD cross). Emits candidate when conditions match."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["indicator_based"] = "indicator_based"
    id: str = "indicator_based_default"
    enabled: bool = False
    timeframe: Literal["M1", "M15", "H4"] = "M15"
    regime: Literal["bull", "bear"] = "bull"
    side: Literal["buy", "sell"] = "buy"
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    rsi_zone: Literal["oversold", "overbought", "neutral"] = "oversold"
    use_macd_cross: bool = False
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    tp_pips: float = 10.0
    sl_pips: Optional[float] = None


class ExecutionPolicyBreakout(BaseModel):
    """Breakout policy: Detects consolidation via ATR and enters on range breakout."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["breakout_range"] = "breakout_range"
    id: str = "breakout_range_default"
    enabled: bool = False
    timeframe: Literal["M1", "M15", "H4"] = "M15"
    lookback_bars: int = 20  # Number of bars to calculate range
    atr_period: int = 14
    atr_threshold_ratio: float = 0.7  # ATR below mean * ratio = consolidation
    breakout_buffer_pips: float = 2.0  # Buffer above/below range for breakout confirmation
    tp_pips: float = 15.0
    sl_pips: Optional[float] = 10.0
    require_volume_increase: bool = False  # If true, also check for volume spike


class ExecutionPolicySessionMomentum(BaseModel):
    """Session momentum policy: Trades in direction of first N-minute move after session open."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["session_momentum"] = "session_momentum"
    id: str = "session_momentum_default"
    enabled: bool = False
    session: Literal["tokyo", "london", "newyork"] = "london"
    setup_minutes: int = 30  # How long after session open to measure initial move
    momentum_threshold_pips: float = 10.0  # Minimum move to consider momentum established
    tp_pips: float = 15.0
    sl_pips: Optional[float] = 10.0
    max_trades_per_session: int = 1  # Only one trade per session
    use_session_high_low_stops: bool = False  # Use session high/low as SL instead of fixed pips


class ExecutionPolicyBollingerBands(BaseModel):
    """Bollinger Bands policy: Mean reversion at lower/upper band with optional regime filter."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["bollinger_bands"] = "bollinger_bands"
    id: str = "bollinger_bands_default"
    enabled: bool = False
    timeframe: Literal["M1", "M15", "H4"] = "M15"
    period: int = 20
    std_dev: float = 2.0
    trigger: Literal["lower_band_buy", "upper_band_sell"] = "lower_band_buy"
    regime: Literal["bull", "bear"] = "bull"
    side: Literal["buy", "sell"] = "buy"
    tp_pips: float = 15.0
    sl_pips: Optional[float] = 10.0


class ExecutionPolicyVWAP(BaseModel):
    """VWAP policy: Enter on price vs VWAP (cross or trend filter)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["vwap"] = "vwap"
    id: str = "vwap_default"
    enabled: bool = False
    timeframe: Literal["M1", "M15", "H4"] = "M15"
    trigger: Literal["cross_above", "cross_below", "above_buy", "below_sell"] = "cross_above"
    side: Literal["buy", "sell"] = "buy"
    tp_pips: float = 15.0
    sl_pips: Optional[float] = 10.0
    use_slope_filter: bool = False
    vwap_slope_lookback_bars: int = 20
    session_filter_enabled: bool = False
    no_trade_zone_pips: float = 0.0


class ExecutionPolicyEmaPullback(BaseModel):
    """Momentum pullback: trend from EMA 50/200, entry when price is in EMA 20-50 zone."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["ema_pullback"] = "ema_pullback"
    id: str = "ema_pullback_default"
    enabled: bool = False
    trend_timeframe: Literal["M5", "M15"] = "M15"
    entry_timeframe: Literal["M5", "M15"] = "M5"
    ema_trend_fast: int = 50
    ema_trend_slow: int = 200
    ema_zone_low: int = 20
    ema_zone_high: int = 50
    tp_pips: float = 30.0
    sl_pips: Optional[float] = 16.0
    require_rejection_candle: bool = False
    require_engulfing_confirmation: bool = False
    min_rr: float = 1.0
    avoid_round_numbers: bool = False
    round_number_buffer_pips: float = 5.0


class ExecutionPolicyEmaBbScalp(BaseModel):
    """EMA 9/21 + Bollinger Band expansion scalper on a single timeframe (typically M1)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["ema_bb_scalp"] = "ema_bb_scalp"
    id: str = "ema_bb_scalp_default"
    enabled: bool = False
    timeframe: Literal["M1", "M5"] = "M1"
    ema_fast: int = 9
    ema_slow: int = 21
    bollinger_period: int = 20
    bollinger_deviation: float = 2.0
    tp_pips: float = 15.0
    sl_pips: Optional[float] = 10.0
    confirm_bars: int = 2
    min_distance_pips: float = 1.0


class ExecutionPolicyKtCgHybrid(BaseModel):
    """KT/CG Hybrid strategy (Trial #2).

    Two INDEPENDENT entry triggers:

    1. Zone Entry (continuous state, respects cooldown):
       - If M5 BULL AND M1 EMA9 > M1 EMA(zone_slow) AND cooldown elapsed -> BUY
       - If M5 BEAR AND M1 EMA9 < M1 EMA(zone_slow) AND cooldown elapsed -> SELL

    2. Pullback Cross (discrete event, OVERRIDES cooldown):
       - If M5 BULL AND M1 EMA9 crosses BELOW EMA(pullback_slow) -> BUY (ignore cooldown)
       - If M5 BEAR AND M1 EMA9 crosses ABOVE EMA(pullback_slow) -> SELL (ignore cooldown)

    Either condition can trigger a trade independently.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["kt_cg_hybrid"] = "kt_cg_hybrid"
    id: str = "kt_cg_hybrid_default"
    enabled: bool = False

    # M5 Trend EMAs
    m5_trend_ema_fast: int = 9
    m5_trend_ema_slow: int = 21

    # M1 Zone Entry - EMA 9 vs slow EMA (default 21 for Trial #2)
    m1_zone_entry_ema_slow: int = 21

    # M1 Pullback Cross - EMA 9 crosses slow EMA (default 13 for Trial #2)
    m1_pullback_cross_ema_slow: int = 13

    # Close opposite trades before placing new trade
    close_opposite_on_trade: bool = True

    # Cooldown after trade (Zone Entry respects this, Pullback Cross overrides it)
    cooldown_minutes: float = 3.0

    tp_pips: float = 15.0
    sl_pips: Optional[float] = 10.0
    confirm_bars: int = 1

    # Swing level proximity filter
    swing_level_filter_enabled: bool = False
    swing_lookback_bars: int = 100
    swing_confirmation_bars: int = 5
    swing_danger_zone_pct: float = 0.15


class ExecutionPolicyKtCgCounterTrendPullback(BaseModel):
    """KT/CG Counter-Trend Pullback strategy (Trial #3).

    Two INDEPENDENT entry triggers:

    1. Zone Entry (continuous state, respects cooldown):
       - If M5 BULL AND M1 EMA9 > M1 EMA(zone_slow) AND cooldown elapsed -> BUY
       - If M5 BEAR AND M1 EMA9 < M1 EMA(zone_slow) AND cooldown elapsed -> SELL

    2. Pullback Cross (discrete event, OVERRIDES cooldown):
       - If M5 BULL AND M1 EMA9 crosses BELOW EMA(pullback_slow) -> BUY (ignore cooldown)
       - If M5 BEAR AND M1 EMA9 crosses ABOVE EMA(pullback_slow) -> SELL (ignore cooldown)

    Either condition can trigger a trade independently.
    Direction switch close: auto-close opposite direction trades before placing new trade.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["kt_cg_counter_trend_pullback"] = "kt_cg_counter_trend_pullback"
    id: str = "kt_cg_ctp_default"
    enabled: bool = False

    # M5 Trend EMAs
    m5_trend_ema_fast: int = 9
    m5_trend_ema_slow: int = 21

    # M1 Zone Entry - EMA 9 vs slow EMA
    m1_zone_entry_ema_slow: int = 13

    # M1 Pullback Cross - EMA 9 crosses slow EMA
    m1_pullback_cross_ema_slow: int = 15

    # Close opposite trades before placing new trade
    close_opposite_on_trade: bool = True

    # Cooldown after trade (Zone Entry respects this, Pullback Cross overrides it)
    cooldown_minutes: float = 3.0

    tp_pips: float = 15.0
    sl_pips: Optional[float] = 10.0
    confirm_bars: int = 1

    # Swing level proximity filter (from Trial #2)
    swing_level_filter_enabled: bool = False
    swing_lookback_bars: int = 100
    swing_confirmation_bars: int = 5
    swing_danger_zone_pct: float = 0.15


class ExecutionPolicyKtCgTrial5(BaseModel):
    """KT/CG Trial #5 (Overhauled: Fresh Cross, Trend Exhaustion, Extended Tiers).

    Two INDEPENDENT entry triggers:
    1. Zone Entry (requires fresh EMA5/EMA9 cross within 10 bars):
       - M3 BULL + M1 EMA5 > EMA9 + fresh cross -> BUY
    2. Tiered Pullback (NO cooldown): live price touches M1 EMA tiers

    Features: Dual ATR filter, Dead Zone 21:00-02:00 UTC, Daily H/L filter,
    Trend Extension Exhaustion, Expanded EMA Zone Filter, Spread-Aware BE.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["kt_cg_trial_5"] = "kt_cg_trial_5"
    id: str = "kt_cg_trial_5_default"
    enabled: bool = False

    # M3 Trend EMAs
    m3_trend_ema_fast: int = 5
    m3_trend_ema_slow: int = 9

    # M1 Zone Entry - EMA5 vs EMA9 (hardcoded, kept for getattr compat)
    zone_entry_enabled: bool = True
    m1_zone_entry_ema_fast: int = 5
    m1_zone_entry_ema_slow: int = 9

    # Tiered Pullback Configuration (default tiers: 18, 21, 25, 29, 34)
    tiered_pullback_enabled: bool = True
    tier_ema_periods: tuple[int, ...] = (18, 21, 25, 29, 34)
    tier_reset_buffer_pips: float = 1.0  # hardcoded

    # Close opposite trades before placing new trade
    close_opposite_on_trade: bool = True

    # Cooldown REMOVED from Trial #5 (replaced by Fresh Cross)
    # Field kept for backward compat during migration
    cooldown_minutes: float = 0.0

    tp_pips: float = 15.0
    sl_pips: Optional[float] = 10.0
    confirm_bars: int = 1  # hardcoded

    # --- Trial #5 Dual ATR Filter ---
    # M1 ATR(7) - Session-Dynamic (master on/off for entire M1 ATR filter)
    m1_atr_filter_enabled: bool = True
    m1_atr_period: int = 7
    m1_atr_min_pips: float = 2.5  # hardcoded fallback
    session_dynamic_atr_enabled: bool = True
    auto_session_detection_enabled: bool = True
    m1_atr_tokyo_min_pips: float = 3.0
    m1_atr_london_min_pips: float = 3.0
    m1_atr_ny_min_pips: float = 3.5

    # M3 ATR(14) - Simple Range
    m3_atr_filter_enabled: bool = True
    m3_atr_period: int = 14  # hardcoded
    m3_atr_min_pips: float = 5.0
    m3_atr_max_pips: float = 16.0

    # M1 ATR(7) Session-Dynamic MAX thresholds
    m1_atr_max_pips: float = 11.0  # hardcoded fallback
    m1_atr_tokyo_max_pips: float = 12.0
    m1_atr_london_max_pips: float = 14.0
    m1_atr_ny_max_pips: float = 16.0

    # Dead Zone (21:00-02:00 UTC) — hours hardcoded, only toggle exposed
    daily_reset_block_enabled: bool = True

    # Daily High/Low Filter (blocks BOTH zone entry AND pullback near daily extremes)
    daily_hl_filter_enabled: bool = True
    daily_hl_buffer_pips: float = 15.0

    # Spread-Aware Breakeven Stop Loss (Trial #5: default and hardcoded to spread+buffer)
    spread_aware_be_enabled: bool = False
    spread_aware_be_trigger_mode: Literal["fixed_pips", "spread_relative"] = "spread_relative"
    spread_aware_be_fixed_trigger_pips: float = 5.0
    spread_aware_be_spread_buffer_pips: float = 1.0
    spread_aware_be_apply_to_zone_entry: bool = True
    spread_aware_be_apply_to_tiered_pullback: bool = True

    # EMA Zone Entry Filter (fully configurable weights and interpolation ranges)
    ema_zone_filter_enabled: bool = True
    ema_zone_filter_lookback_bars: int = 3
    ema_zone_filter_block_threshold: float = 0.35
    ema_zone_filter_spread_weight: float = 0.45
    ema_zone_filter_slope_weight: float = 0.40
    ema_zone_filter_direction_weight: float = 0.15
    ema_zone_filter_spread_min_pips: float = 0.0
    ema_zone_filter_spread_max_pips: float = 4.0
    ema_zone_filter_slope_min_pips: float = -1.0
    ema_zone_filter_slope_max_pips: float = 3.0
    ema_zone_filter_dir_min_pips: float = -3.0
    ema_zone_filter_dir_max_pips: float = 3.0

    # Trend Extension Exhaustion
    trend_exhaustion_enabled: bool = True
    trend_exhaustion_fresh_max: float = 2.0
    trend_exhaustion_mature_max: float = 3.5
    trend_exhaustion_extended_max: float = 5.0
    trend_exhaustion_ramp_minutes: float = 12.0

    # Per-direction open trade cap (None = disabled)
    max_open_trades_per_side: Optional[int] = None


class ExecutionPolicyKtCgTrial4(BaseModel):
    """KT/CG Trial #4 (M3 Trend + Tiered Pullback System).

    Two INDEPENDENT entry triggers:

    1. Zone Entry (continuous state, respects cooldown):
       - If M3 BULL AND M1 EMA5 > M1 EMA9 AND cooldown elapsed -> BUY
       - If M3 BEAR AND M1 EMA5 < M1 EMA9 AND cooldown elapsed -> SELL

    2. Tiered Pullback (discrete event, NO cooldown):
       - 8 tiers: M1 EMA 9, 11, 12, 13, 14, 15, 16, 17
       - If M3 BULL AND live bid touches/goes below any tier EMA -> BUY
       - If M3 BEAR AND live ask touches/goes above any tier EMA -> SELL
       - Each tier fires only once per touch (state tracking)
       - Tier resets when price moves away from the EMA by reset_buffer

    Either condition can trigger a trade independently.
    Direction switch close: auto-close opposite direction trades before placing new trade.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["kt_cg_trial_4"] = "kt_cg_trial_4"
    id: str = "kt_cg_trial_4_default"
    enabled: bool = False

    # M3 Trend EMAs
    m3_trend_ema_fast: int = 5
    m3_trend_ema_slow: int = 9

    # M1 Zone Entry - EMA5 vs EMA9
    zone_entry_enabled: bool = True
    m1_zone_entry_ema_fast: int = 5
    m1_zone_entry_ema_slow: int = 9

    # Tiered Pullback Configuration
    tiered_pullback_enabled: bool = True
    tier_ema_periods: tuple[int, ...] = (9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30)
    tier_reset_buffer_pips: float = 1.0  # Distance from EMA to reset tier

    # Close opposite trades before placing new trade
    close_opposite_on_trade: bool = True

    # Cooldown after trade (Zone Entry respects this, Tiered Pullback has NO cooldown)
    cooldown_minutes: float = 3.0

    tp_pips: float = 15.0
    sl_pips: Optional[float] = 10.0
    confirm_bars: int = 1

    # Rolling Danger Zone Filter (M1-based)
    # Calculates rolling high/low over X M1 bars, blocks entries near extremes
    rolling_danger_zone_enabled: bool = True
    rolling_danger_lookback_bars: int = 100  # X bars to calculate high/low
    rolling_danger_zone_pct: float = 0.15  # Y% of range at top/bottom

    # RSI Divergence Detection (M5-based, rolling window comparison)
    # Detects price-RSI divergence and blocks entries against the divergence
    # BULL trend + bearish divergence -> block BUY for X minutes
    # BEAR trend + bullish divergence -> block SELL for X minutes
    rsi_divergence_enabled: bool = False
    rsi_divergence_period: int = 14  # RSI calculation period
    rsi_divergence_lookback_bars: int = 50  # Bars to analyze (split into reference/recent halves)
    rsi_divergence_block_minutes: float = 5.0  # How long to block entries after divergence

    # Tiered ATR(14) Filter (replaces generic ATR filter for Trial #4)
    # < block_below: block ALL (too quiet)
    # block_below to allow_all_max: allow ALL
    # allow_all_max to pullback_only_max: block zone entry, allow pullback only
    # > pullback_only_max: block ALL (too volatile)
    tiered_atr_filter_enabled: bool = True
    tiered_atr_block_below_pips: float = 4.0
    tiered_atr_allow_all_max_pips: float = 12.0
    tiered_atr_pullback_only_max_pips: float = 15.0

    # Daily High/Low Filter (blocks zone entry near daily extremes)
    daily_hl_filter_enabled: bool = False
    daily_hl_buffer_pips: float = 5.0

    # Spread-Aware Breakeven Stop Loss
    spread_aware_be_enabled: bool = False
    spread_aware_be_trigger_mode: Literal["fixed_pips", "spread_relative"] = "fixed_pips"
    spread_aware_be_fixed_trigger_pips: float = 5.0
    spread_aware_be_spread_buffer_pips: float = 1.0
    spread_aware_be_apply_to_zone_entry: bool = True
    spread_aware_be_apply_to_tiered_pullback: bool = True

    # EMA Zone Entry Filter (scoring system: blocks zone entries during EMA compression)
    # Uses weighted score of M1 EMA 9 vs EMA 17 spread, slope, and spread direction
    # Only affects zone entries; tiered pullback entries are NOT filtered
    ema_zone_filter_enabled: bool = True
    ema_zone_filter_lookback_bars: int = 3      # Bars back for slope/direction calculation
    ema_zone_filter_block_threshold: float = 0.35  # Block if weighted score below this


class ExecutionPolicyKtCgTrial7(BaseModel):
    """KT/CG Trial #7 Recover Account (M5 Trend + Tiered Pullback + Slope Gate)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["kt_cg_trial_7"] = "kt_cg_trial_7"
    id: str = "kt_cg_trial_7_default"
    enabled: bool = False

    # M5 Trend EMAs
    m5_trend_ema_fast: int = 9
    m5_trend_ema_slow: int = 21
    # Block all entries when M5 EMA9/EMA21 are too close (chop guard)
    m5_min_ema_distance_pips: float = 1.0

    # M1 Zone Entry - EMA5 vs EMA9
    zone_entry_enabled: bool = True
    zone_entry_mode: Literal["ema_cross", "price_vs_ema5"] = "ema_cross"
    m1_zone_entry_ema_fast: int = 5
    m1_zone_entry_ema_slow: int = 9

    # Tiered Pullback Configuration (9-34 subset, configurable)
    tiered_pullback_enabled: bool = True
    tier_ema_periods: tuple[int, ...] = (
        9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29, 30, 31, 32, 33, 34
    )
    tier_reset_buffer_pips: float = 1.0

    # Close opposite trades before placing new trade
    close_opposite_on_trade: bool = True

    # Cooldown after trade (Zone Entry respects this, Tiered Pullback has NO cooldown)
    cooldown_minutes: float = 3.0

    tp_pips: float = 4.0
    sl_pips: Optional[float] = 20.0
    confirm_bars: int = 1

    # Spread-Aware Breakeven Stop Loss
    spread_aware_be_enabled: bool = False
    spread_aware_be_trigger_mode: Literal["fixed_pips", "spread_relative"] = "fixed_pips"
    spread_aware_be_fixed_trigger_pips: float = 5.0
    spread_aware_be_spread_buffer_pips: float = 1.0
    spread_aware_be_apply_to_zone_entry: bool = True
    spread_aware_be_apply_to_tiered_pullback: bool = True

    # EMA Zone Entry Filter (slope-only gate; zone entries only)
    ema_zone_filter_enabled: bool = False
    ema_zone_filter_lookback_bars: int = 3
    ema_zone_filter_ema5_min_slope_pips_per_bar: float = 0.10
    ema_zone_filter_ema9_min_slope_pips_per_bar: float = 0.08
    ema_zone_filter_ema21_min_slope_pips_per_bar: float = 0.05

    # Trend exhaustion (M5 stretch from EMA21), default OFF.
    # Stretch = abs(current_price - EMA21_M5) / pip_size.
    trend_exhaustion_enabled: bool = False
    trend_exhaustion_mode: Literal["global", "session", "session_and_side"] = "session_and_side"
    trend_exhaustion_use_current_price: bool = True
    trend_exhaustion_hysteresis_pips: float = 0.5

    # Global thresholds (fallback / global mode)
    trend_exhaustion_p80_global: float = 12.03
    trend_exhaustion_p90_global: float = 17.02

    # Session thresholds (session mode)
    trend_exhaustion_p80_tokyo: float = 12.67
    trend_exhaustion_p90_tokyo: float = 17.63
    trend_exhaustion_p80_london: float = 11.06
    trend_exhaustion_p90_london: float = 14.41
    trend_exhaustion_p80_ny: float = 12.66
    trend_exhaustion_p90_ny: float = 18.83

    # Session+side thresholds (session_and_side mode)
    trend_exhaustion_p80_bull_tokyo: float = 11.85
    trend_exhaustion_p90_bull_tokyo: float = 15.52
    trend_exhaustion_p80_bull_london: float = 10.21
    trend_exhaustion_p90_bull_london: float = 12.97
    trend_exhaustion_p80_bull_ny: float = 11.21
    trend_exhaustion_p90_bull_ny: float = 15.84
    trend_exhaustion_p80_bear_tokyo: float = 13.44
    trend_exhaustion_p90_bear_tokyo: float = 19.73
    trend_exhaustion_p80_bear_london: float = 12.01
    trend_exhaustion_p90_bear_london: float = 17.44
    trend_exhaustion_p80_bear_ny: float = 13.97
    trend_exhaustion_p90_bear_ny: float = 21.51

    # Behavior controls by zone
    trend_exhaustion_extended_disable_zone_entry: bool = True
    trend_exhaustion_very_extended_disable_zone_entry: bool = True
    trend_exhaustion_extended_min_tier_period: int = 21
    trend_exhaustion_very_extended_min_tier_period: int = 29
    trend_exhaustion_very_extended_tighten_caps: bool = True
    trend_exhaustion_very_extended_cap_multiplier: float = 0.5
    trend_exhaustion_very_extended_cap_min: int = 1
    # Adaptive TP based on exhaustion zone (offsets from base tp_pips)
    trend_exhaustion_adaptive_tp_enabled: bool = False
    trend_exhaustion_tp_extended_offset_pips: float = 1.0
    trend_exhaustion_tp_very_extended_offset_pips: float = 2.0
    trend_exhaustion_tp_min_pips: float = 0.5

    # Open trade caps (all caps enforced together)
    max_open_trades_per_side: Optional[int] = 5
    max_zone_entry_open: Optional[int] = 3
    max_tiered_pullback_open: Optional[int] = 8


class ExecutionPolicyKtCgTrial6(BaseModel):
    """KT/CG Trial #6 (BB Slope Trend + EMA Tier Pullback + BB Reversal).

    Two INDEPENDENT entry systems:

    System A: EMA Tier Pullback
      - M3 slope trend (EMA 5>9>21, all slopes>0, BB expanding) determines direction
      - Live price touches M1 EMA tiers -> entry
      - BB gating: when price outside M1 BB, only deep tiers allowed

    System B: BB Reversal (counter-trend)
      - Price exceeds M1 BB upper/lower by tiered offsets -> counter-trend entry
      - TP targets middle BB (snapshot or dynamic)
      - Resets when price returns inside BB
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["kt_cg_trial_6"] = "kt_cg_trial_6"
    id: str = "kt_cg_trial_6_default"
    enabled: bool = False

    # M3 Slope Trend Engine
    m3_trend_ema_fast: int = 5
    m3_trend_ema_slow: int = 9
    m3_trend_ema_extra: int = 21
    m3_slope_lookback: int = 2
    m3_bb_period: int = 20
    m3_bb_std: float = 2.0

    # M1 Bollinger Bands
    m1_bb_period: int = 20
    m1_bb_std: float = 2.0

    # System A: EMA Tier Pullback
    ema_tier_enabled: bool = True
    tier_ema_periods: tuple[int, ...] = (9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34)
    tier_reset_buffer_pips: float = 1.0
    ema_tier_tp_pips: float = 7.0
    sl_pips: float = 10.0
    bb_gating_deep_tier_min_period: int = 21

    # System B: BB Reversal
    bb_reversal_enabled: bool = True
    bb_reversal_start_offset_pips: float = 0.5
    bb_reversal_increment_pips: float = 0.5
    bb_reversal_num_tiers: int = 10
    max_bb_reversal_positions: int = 3
    bb_reversal_tp_mode: Literal["middle_bb_entry", "middle_bb_dynamic", "fixed_pips"] = "middle_bb_entry"
    bb_reversal_tp_fixed_pips: float = 8.0
    bb_reversal_tp_min_pips: float = 4.0
    bb_reversal_tp_max_pips: float = 15.0
    bb_reversal_sl_pips: float = 10.0

    # Dead Zone (configurable hours)
    dead_zone_enabled: bool = True
    dead_zone_start_hour_utc: int = 21
    dead_zone_end_hour_utc: int = 2

    # Risk
    cooldown_after_loss_seconds: float = 180.0
    close_opposite_on_trade: bool = False
    max_open_trades_per_side: int = 15

    # Spread-Aware Breakeven (Spread + Buffer only)
    spread_aware_be_enabled: bool = True
    spread_aware_be_trigger_mode: Literal["fixed_pips", "spread_relative"] = "spread_relative"
    spread_aware_be_fixed_trigger_pips: float = 7.0
    spread_aware_be_spread_buffer_pips: float = 1.5
    spread_aware_be_apply_to_ema_tier: bool = True
    spread_aware_be_apply_to_bb_reversal: bool = True


ExecutionPolicy = Annotated[
    Union[
        ExecutionPolicyConfirmedCross,
        ExecutionPolicyPriceLevelTrend,
        ExecutionPolicyIndicator,
        ExecutionPolicyBreakout,
        ExecutionPolicySessionMomentum,
        ExecutionPolicyBollingerBands,
        ExecutionPolicyVWAP,
        ExecutionPolicyEmaPullback,
        ExecutionPolicyEmaBbScalp,
        ExecutionPolicyKtCgHybrid,
        ExecutionPolicyKtCgCounterTrendPullback,
        ExecutionPolicyKtCgTrial4,
        ExecutionPolicyKtCgTrial5,
        ExecutionPolicyKtCgTrial7,
        ExecutionPolicyKtCgTrial6,
    ],
    Field(discriminator="type"),
]


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policies: list[ExecutionPolicy] = Field(default_factory=list)
    loop_poll_seconds: float = 5.0
    loop_poll_seconds_fast: float = 2.0


class ProfileV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: SchemaVersion = 1
    created_utc: str = Field(default_factory=utc_now_iso)

    profile_name: str
    symbol: str
    pip_size: float = 0.01

    # Broker: MT5 (Windows) or OANDA (Japan/Canada, works on PaaS)
    broker_type: Literal["mt5", "oanda"] = "mt5"
    oanda_token: Optional[str] = None
    oanda_account_id: Optional[str] = None
    oanda_environment: Literal["practice", "live"] = "practice"

    # Display currency for stats and logs (USD or JPY)
    display_currency: Optional[Literal["USD", "JPY"]] = "USD"

    # Account settings for risk calculation
    deposit_amount: Optional[float] = None  # e.g., 100000.0
    leverage_ratio: Optional[int] = None  # e.g., 100 for 1:100
    
    # Active preset tracking
    active_preset_name: Optional[str] = None  # e.g., "aggressive_scalping" or "Custom: Scalp/Agg/MS"

    # Profile Editor limits (final; only editable in Profile Editor). Never overwritten by presets or wizard.
    risk: RiskConfig = Field(default_factory=RiskConfig)
    # Effective risk for the active preset (capped by risk). Set when applying a preset or saving from wizard.
    # Run loop and execution use this when set; otherwise use risk.
    effective_risk: Optional[RiskConfig] = None

    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    trade_management: TradeManagementConfig = Field(default_factory=TradeManagementConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)


class ProfileV1AllowExtra(ProfileV1):
    """Same as ProfileV1 but allows extra keys (e.g. broker fields on old deploys). Used for save/load when strict validation fails."""
    model_config = ConfigDict(extra="ignore")
    # Redeclare broker fields so they are always accepted and persisted even if an old deploy's ProfileV1 lacks them
    broker_type: Literal["mt5", "oanda"] = "mt5"
    oanda_token: Optional[str] = None
    oanda_account_id: Optional[str] = None
    oanda_environment: Literal["practice", "live"] = "practice"


def _looks_like_v1(d: dict[str, Any]) -> bool:
    return d.get("schema_version") == 1 and "risk" in d and "strategy" in d


def get_effective_risk(profile: ProfileV1) -> RiskConfig:
    """Return the risk config to use at runtime (effective preset risk capped by profile limits).

    When effective_risk is set, use it but cap max_lots, max_spread_pips, max_trades_per_day,
    max_open_trades by profile.risk so the preset never exceeds Profile Editor limits.
    When effective_risk is None, use profile.risk (legacy behavior).
    """
    limits = profile.risk
    if profile.effective_risk is None:
        return limits
    eff = profile.effective_risk
    return RiskConfig(
        max_lots=min(eff.max_lots, limits.max_lots),
        require_stop=eff.require_stop,
        min_stop_pips=eff.min_stop_pips,
        max_spread_pips=min(eff.max_spread_pips, limits.max_spread_pips),
        max_trades_per_day=min(eff.max_trades_per_day, limits.max_trades_per_day),
        cooldown_minutes_after_loss=eff.cooldown_minutes_after_loss,
        max_open_trades=min(eff.max_open_trades, limits.max_open_trades),
        risk_per_trade_pct=eff.risk_per_trade_pct,
        max_daily_loss_pct=eff.max_daily_loss_pct,
    )


def _default_execution() -> dict[str, Any]:
    return {
        "policies": [
            {"type": "confirmed_cross", "id": "confirmed_cross_default", "enabled": True, "setup_id": "m1_cross_entry"}
        ],
        "loop_poll_seconds": 5.0,
        "loop_poll_seconds_fast": 2.0,
    }


def migrate_profile_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Migrate older flat profiles to ProfileV1-compatible dict.

    This is intentionally conservative: it only maps known fields and fills defaults.
    Unknown fields are ignored to avoid accidental schema bloat.
    """
    if _looks_like_v1(d):
        if "execution" not in d:
            d = {**d, "execution": _default_execution()}
        if "display_currency" not in d:
            d = {**d, "display_currency": "USD"}
        # Ensure broker fields exist (OANDA support)
        if "broker_type" not in d:
            d = {**d, "broker_type": "mt5"}
        if "oanda_token" not in d:
            d = {**d, "oanda_token": None}
        if "oanda_account_id" not in d:
            d = {**d, "oanda_account_id": None}
        if "oanda_environment" not in d:
            d = {**d, "oanda_environment": "practice"}
        # Migrate legacy preset names to single RSI preset
        ap = d.get("active_preset_name")
        if ap in ("mean_reversion_dip_buy", "mean_reversion_dip_sell"):
            d = {**d, "active_preset_name": "mean_reversion_dip"}
        # Migrate Trial #4 policies: remove old fields, convert swing filter to rolling danger zone
        execution = d.get("execution")
        if isinstance(execution, dict):
            policies = execution.get("policies")
            if isinstance(policies, list):
                for pol in policies:
                    if isinstance(pol, dict) and pol.get("type") == "kt_cg_trial_4":
                        # Remove old pullback cross EMA fields (now uses tiered pullback)
                        pol.pop("m1_pullback_cross_ema_fast", None)
                        pol.pop("m1_pullback_cross_ema_slow", None)
                        # Migrate swing filter to rolling danger zone
                        if "swing_level_filter_enabled" in pol:
                            pol["rolling_danger_zone_enabled"] = pol.pop("swing_level_filter_enabled")
                        if "swing_lookback_bars" in pol:
                            pol["rolling_danger_lookback_bars"] = pol.pop("swing_lookback_bars")
                        if "swing_danger_zone_pct" in pol:
                            pol["rolling_danger_zone_pct"] = pol.pop("swing_danger_zone_pct")
                        pol.pop("swing_confirmation_bars", None)  # No longer needed
                        # Remove old RSI divergence swing window (now uses rolling window comparison)
                        pol.pop("rsi_divergence_swing_window", None)
                    if isinstance(pol, dict) and pol.get("type") == "kt_cg_trial_5":
                        # Remove tiered ATR fields (replaced by M1 ATR max in Trial #5)
                        pol.pop("tiered_atr_filter_enabled", None)
                        pol.pop("tiered_atr_block_below_pips", None)
                        pol.pop("tiered_atr_allow_all_max_pips", None)
                        pol.pop("tiered_atr_pullback_only_max_pips", None)
                        # Remove rolling danger zone (removed from Trial #5)
                        pol.pop("rolling_danger_zone_enabled", None)
                        pol.pop("rolling_danger_lookback_bars", None)
                        pol.pop("rolling_danger_zone_pct", None)
                        # Remove RSI divergence (removed from Trial #5)
                        pol.pop("rsi_divergence_enabled", None)
                        pol.pop("rsi_divergence_period", None)
                        pol.pop("rsi_divergence_lookback_bars", None)
                        pol.pop("rsi_divergence_block_minutes", None)
                        # Migrate cooldown_minutes -> 0 (replaced by Fresh Cross)
                        if "cooldown_minutes" in pol:
                            pol["cooldown_minutes"] = 0.0
                    if isinstance(pol, dict) and pol.get("type") == "kt_cg_trial_7":
                        # Remove filters not used by Trial #7
                        pol.pop("tiered_atr_filter_enabled", None)
                        pol.pop("tiered_atr_block_below_pips", None)
                        pol.pop("tiered_atr_allow_all_max_pips", None)
                        pol.pop("tiered_atr_pullback_only_max_pips", None)
                        pol.pop("rolling_danger_zone_enabled", None)
                        pol.pop("rolling_danger_lookback_bars", None)
                        pol.pop("rolling_danger_zone_pct", None)
                        pol.pop("rsi_divergence_enabled", None)
                        pol.pop("rsi_divergence_period", None)
                        pol.pop("rsi_divergence_lookback_bars", None)
                        pol.pop("rsi_divergence_block_minutes", None)
                        pol.pop("daily_hl_filter_enabled", None)
                        pol.pop("daily_hl_buffer_pips", None)
                        # Remove legacy scoring fields; Trial #7 uses slope-only gate
                        pol.pop("ema_zone_filter_block_threshold", None)
                        pol.pop("ema_zone_filter_spread_weight", None)
                        pol.pop("ema_zone_filter_slope_weight", None)
                        pol.pop("ema_zone_filter_direction_weight", None)
                        pol.pop("ema_zone_filter_spread_min_pips", None)
                        pol.pop("ema_zone_filter_spread_max_pips", None)
                        pol.pop("ema_zone_filter_slope_min_pips", None)
                        pol.pop("ema_zone_filter_slope_max_pips", None)
                        pol.pop("ema_zone_filter_dir_min_pips", None)
                        pol.pop("ema_zone_filter_dir_max_pips", None)
                        # Ensure required Trial #7 defaults
                        if "m5_min_ema_distance_pips" not in pol:
                            pol["m5_min_ema_distance_pips"] = 1.0
                        if "zone_entry_mode" not in pol:
                            pol["zone_entry_mode"] = "ema_cross"
                        elif pol.get("zone_entry_mode") not in ("ema_cross", "price_vs_ema5"):
                            pol["zone_entry_mode"] = "ema_cross"
                        if "trend_exhaustion_adaptive_tp_enabled" not in pol:
                            pol["trend_exhaustion_adaptive_tp_enabled"] = False
                        if "trend_exhaustion_tp_extended_offset_pips" not in pol:
                            pol["trend_exhaustion_tp_extended_offset_pips"] = 1.0
                        if "trend_exhaustion_tp_very_extended_offset_pips" not in pol:
                            pol["trend_exhaustion_tp_very_extended_offset_pips"] = 2.0
                        if "trend_exhaustion_tp_min_pips" not in pol:
                            pol["trend_exhaustion_tp_min_pips"] = 0.5
                        try:
                            pol["trend_exhaustion_tp_extended_offset_pips"] = max(
                                0.0, float(pol.get("trend_exhaustion_tp_extended_offset_pips", 1.0))
                            )
                            pol["trend_exhaustion_tp_very_extended_offset_pips"] = max(
                                0.0, float(pol.get("trend_exhaustion_tp_very_extended_offset_pips", 2.0))
                            )
                            pol["trend_exhaustion_tp_min_pips"] = max(
                                0.1, float(pol.get("trend_exhaustion_tp_min_pips", 0.5))
                            )
                        except Exception:
                            pol["trend_exhaustion_tp_extended_offset_pips"] = 1.0
                            pol["trend_exhaustion_tp_very_extended_offset_pips"] = 2.0
                            pol["trend_exhaustion_tp_min_pips"] = 0.5
                        # Canonical tier set for Trial #7: EMA 9 and 11..34 (no EMA10)
                        raw_tiers = pol.get("tier_ema_periods")
                        allowed = {9, *range(11, 35)}
                        canonical_default = [9, *list(range(11, 35))]
                        if isinstance(raw_tiers, (list, tuple)):
                            cleaned = sorted({int(x) for x in raw_tiers if isinstance(x, (int, float)) and int(x) in allowed})
                            pol["tier_ema_periods"] = cleaned if cleaned else canonical_default
                        else:
                            pol["tier_ema_periods"] = canonical_default
        return d

    profile_name = d.get("profile_name") or d.get("name") or "default"
    symbol = d.get("symbol") or "USDJPY"
    pip_size = float(d.get("pip_size", 0.01))

    max_lots = d.get("max_lots", 0.2)
    require_stop = d.get("require_stop", True)
    min_stop_pips = d.get("min_stop_pips", 10)
    max_spread_pips = d.get("max_spread_pips", 2.0)
    max_trades_per_day = d.get("max_trades_per_day", 10)
    target_profit_pips_default = d.get("target_profit_pips_default", 10)

    migrated: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": d.get("created_utc", utc_now_iso()),
        "profile_name": profile_name,
        "symbol": symbol,
        "pip_size": pip_size,
        "broker_type": d.get("broker_type", "mt5"),
        "oanda_token": d.get("oanda_token"),
        "oanda_account_id": d.get("oanda_account_id"),
        "oanda_environment": d.get("oanda_environment", "practice"),
        "risk": {
            "max_lots": max_lots,
            "require_stop": require_stop,
            "min_stop_pips": min_stop_pips,
            "max_spread_pips": max_spread_pips,
            "max_trades_per_day": max_trades_per_day,
        },
        "strategy": {
            "timeframes": {
                "M1": {"ema_fast": 13, "sma_slow": 30, "ema_stack": [8, 13, 21]},
                "M15": {"ema_fast": 13, "sma_slow": 30},
                "H4": {"ema_fast": 13, "sma_slow": 30},
            },
            "setups": {
                "m1_cross_entry": {
                    "enabled": True,
                    "timeframe": "M1",
                    "ema": 13,
                    "sma": 30,
                    "confirmation": {
                        "confirm_bars": 1,
                        "require_close_on_correct_side": True,
                        "min_distance_pips": 0.0,
                        "max_wait_bars": 5,
                    },
                }
            },
            "filters": {
                "alignment": {"enabled": False, "method": "score", "weights": {"H4": 1, "M15": 1, "M1": 1}, "min_score_to_trade": -3, "trend_timeframe": None},
                "ema_stack_filter": {"enabled": False, "timeframe": "M1", "periods": [8, 13, 21], "min_separation_pips": 0.0},
                "atr_filter": {"enabled": False, "timeframe": "M1", "atr_period": 14, "min_atr_pips": 0.0, "max_atr_pips": None},
                "session_filter": {"enabled": False, "sessions": ["Tokyo", "London", "NewYork"]},
                "session_boundary_block": {"enabled": False, "buffer_minutes": 15},
            },
        },
        "trade_management": {
            "target": {
                "mode": "fixed_pips",
                "pips_default": float(target_profit_pips_default),
                "rr_default": 1.0,
            },
            "stop_loss": None,
            "breakeven": None,
        },
        "execution": _default_execution(),
    }

    return migrated


def load_profile_v1(profile_path: str | Path) -> ProfileV1 | ProfileV1AllowExtra:
    path = Path(profile_path)
    raw = path.read_text(encoding="utf-8")
    data = json_loads(raw)
    migrated = migrate_profile_dict(data)
    try:
        return ProfileV1.model_validate(migrated)
    except ValidationError as e:
        if "extra_forbidden" in str(e) or "Extra inputs" in str(e):
            try:
                return ProfileV1AllowExtra.model_validate(migrated)
            except ValidationError:
                pass
        raise ValueError(f"Invalid profile after migration: {path}\n{e}") from e


def json_loads(s: str) -> dict[str, Any]:
    import json

    v = json.loads(s)
    if not isinstance(v, dict):
        raise ValueError("Profile JSON must be an object")
    return v


def save_profile_v1(profile: ProfileV1 | ProfileV1AllowExtra, out_path: str | Path) -> None:
    import json

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile.model_dump(), indent=2, sort_keys=False) + "\n", encoding="utf-8")


def default_profile_for_name(profile_name: str) -> ProfileV1:
    """Build a ProfileV1 with default settings for a new account."""
    return ProfileV1(profile_name=profile_name, symbol="USDJPY.PRO")
