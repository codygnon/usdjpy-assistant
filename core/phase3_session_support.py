from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Phase3SessionSupport:
    ExecutionDecision: Any
    pd: Any
    np: Any
    PIP_SIZE: float
    LEVERAGE: float
    MAX_UNITS: int
    MAX_ENTRY_SPREAD_PIPS: float
    BB_WIDTH_LOOKBACK: int
    BB_WIDTH_RANGING_PCT: float
    BB_PERIOD: int
    RSI_PERIOD: int
    ATR_PERIOD: int
    ADX_PERIOD: int
    ATR_MAX: float
    ADX_MAX: float
    MIN_CONFLUENCE: int
    BLOCKED_COMBOS: Any
    ZONE_TOLERANCE_PIPS: float
    RSI_LONG_ENTRY: float
    RSI_SHORT_ENTRY: float
    SL_BUFFER_PIPS: float
    SL_MIN_PIPS: float
    SL_MAX_PIPS: float
    TP1_ATR_MULT: float
    TP1_MIN_PIPS: float
    TP1_MAX_PIPS: float
    SESSION_LOSS_STOP_PCT: float
    COOLDOWN_MINUTES: int
    MAX_TRADES_PER_SESSION: int
    STOP_AFTER_CONSECUTIVE_LOSSES: int
    MAX_CONCURRENT: int
    TOKYO_START_UTC: float
    LDN_MAX_OPEN: int
    LDN_RISK_PCT: float
    LDN_MAX_TRADES_PER_DAY_TOTAL: int
    LDN_ARB_RANGE_MIN_PIPS: float
    LDN_ARB_RANGE_MAX_PIPS: float
    LDN_D_LOR_MIN_PIPS: float
    LDN_D_LOR_MAX_PIPS: float
    LDN_D_MAX_TRADES: int
    LDN_MAX_SPREAD_PIPS: float
    LDN_ARB_SL_BUFFER_PIPS: float
    LDN_ARB_SL_MIN_PIPS: float
    LDN_ARB_SL_MAX_PIPS: float
    LDN_ARB_TP1_R: float
    LDN_D_TP1_R: float
    LDN_D_BREAKOUT_BUFFER_PIPS: float
    LDN_D_SL_BUFFER_PIPS: float
    LDN_D_SL_MIN_PIPS: float
    LDN_D_SL_MAX_PIPS: float
    V44_MAX_ENTRY_SPREAD: float
    V44_MAX_OPEN: int
    V44_MAX_ENTRIES_DAY: int
    V44_SESSION_STOP_LOSSES: int
    V44_STRONG_TP1_PIPS: float
    V44_H1_EMA_FAST: int
    V44_H1_EMA_SLOW: int
    V44_M5_EMA_FAST: int
    V44_M5_EMA_SLOW: int
    V44_SLOPE_BARS: int
    V44_STRONG_SLOPE: float
    V44_WEAK_SLOPE: float
    V44_MIN_BODY_PIPS: float
    V44_ATR_PCT_CAP: float
    V44_ATR_PCT_LOOKBACK: int
    _phase3_order_confirmed: Any
    _drop_incomplete_tf: Any
    _compute_bb: Any
    _compute_rsi: Any
    _compute_atr: Any
    _compute_adx: Any
    _compute_ema: Any
    _compute_session_windows: Any
    _compute_risk_units: Any
    _account_sizing_value: Any
    _compute_v44_atr_rank: Any
    _determine_v44_session_mode: Any
    _load_news_events_cached: Any
    _v44_most_recent_news_event: Any
    _v44_news_trend_active_event: Any
    _as_risk_fraction: Any
    _compute_regime_snapshot: Any
    _london_v2_ownership_cluster_block: Any
    _v44_regime_block: Any
    compute_daily_fib_pivots: Any
    compute_bb_width_regime: Any
    detect_sar_flip: Any
    parse_hhmm_to_hour: Any
    compute_efficiency_ratio: Any
    compute_delta_efficiency_ratio: Any
    compute_asian_range: Any
    evaluate_london_v2_arb: Any
    london_setup_d_weekday_block_from_state: Any
    resolve_v14_cell_scale_override_from_state: Any
    compute_v14_lot_size: Any
    compute_v14_units_from_config: Any
    evaluate_v14_confluence: Any
    compute_v14_signal_strength: Any
    compute_v14_sl: Any
    compute_v14_tp1: Any
    resolve_ny_window_hours: Any
    evaluate_v44_entry: Any
    compute_v44_h1_trend: Any
    compute_v44_sl: Any
    is_in_news_window: Any
    v44_defensive_veto_block_from_state: Any


def build_phase3_session_support(**kwargs: Any) -> Phase3SessionSupport:
    return Phase3SessionSupport(**kwargs)


SESSION_SUPPORT_FIELDS = tuple(Phase3SessionSupport.__dataclass_fields__.keys())


def build_phase3_session_support_from_mapping(mapping: dict[str, Any]) -> Phase3SessionSupport:
    return build_phase3_session_support(**{name: mapping[name] for name in SESSION_SUPPORT_FIELDS})
