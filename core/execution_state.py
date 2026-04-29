from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from core.json_state import load_json_state, save_json_state


Mode = Literal["DISARMED", "ARMED_MANUAL_CONFIRM", "ARMED_AUTO_DEMO"]


@dataclass(frozen=True)
class RuntimeState:
    mode: Mode = "DISARMED"
    kill_switch: bool = False
    exit_system_only: bool = False

    # Used for idempotency / loop progress tracking
    last_processed_bar_time_utc: Optional[str] = None

    # Temporary EMA overrides for Trial #3 (Apply Temporary Settings)
    temp_m5_trend_ema_fast: Optional[int] = None
    temp_m5_trend_ema_slow: Optional[int] = None
    temp_m5_trend_source: Optional[str] = None
    temp_m1_zone_entry_ema_slow: Optional[int] = None
    temp_m1_pullback_cross_ema_slow: Optional[int] = None

    # Temporary EMA overrides for Trial #4 (Apply Temporary Settings)
    temp_m3_trend_ema_fast: Optional[int] = None
    temp_m3_trend_ema_slow: Optional[int] = None
    temp_m1_t4_zone_entry_ema_fast: Optional[int] = None
    temp_m1_t4_zone_entry_ema_slow: Optional[int] = None

    # Trial #4 Tiered Pullback State (dynamic dict: EMA period -> fired bool)
    tier_fired: dict = field(default_factory=dict)

    # RSI Divergence Block State (Trial #4)
    # ISO timestamps indicating when the block expires
    divergence_block_buy_until: Optional[str] = None
    divergence_block_sell_until: Optional[str] = None

    # Daily Reset Block (Trial #5)
    daily_reset_date: Optional[str] = None        # "YYYY-MM-DD" of current tracking day
    daily_reset_high: Optional[float] = None       # Tracked daily high from ticks
    daily_reset_low: Optional[float] = None        # Tracked daily low from ticks
    daily_reset_block_active: bool = False          # True during dead zone (21:00-02:00 UTC)
    daily_reset_settled: bool = False               # True when outside dead zone (H/L is usable)

    # Trend Extension Exhaustion (Trial #5)
    trend_flip_price: Optional[float] = None       # Price at last M3 EMA9/EMA21 trend flip
    trend_flip_direction: Optional[str] = None     # "bull" or "bear"
    trend_flip_time: Optional[str] = None          # ISO UTC timestamp of last M3 EMA9/21 flip

    # Trial #6 BB Reversal Tier State (offset_index -> fired bool)
    bb_tier_fired: dict = field(default_factory=dict)

    # Uncle Parsh H1 Breakout temp overrides
    temp_up_m5_ema_fast: Optional[int] = None
    temp_up_m5_ema_slow: Optional[int] = None

    # Uncle Parsh H1 Breakout: H1 Detection overrides
    temp_up_major_extremes_only: Optional[bool] = None
    temp_up_h1_lookback_hours: Optional[int] = None
    temp_up_h1_swing_strength: Optional[int] = None
    temp_up_h1_cluster_tolerance_pips: Optional[float] = None
    temp_up_h1_min_touches_for_major: Optional[int] = None

    # Uncle Parsh H1 Breakout: M5 Catalyst overrides
    temp_up_power_close_body_pct: Optional[float] = None
    temp_up_velocity_pips: Optional[float] = None

    # Uncle Parsh H1 Breakout: Exit Strategy overrides
    temp_up_initial_sl_spread_plus_pips: Optional[float] = None
    temp_up_tp1_pips: Optional[float] = None
    temp_up_tp1_close_pct: Optional[float] = None
    temp_up_be_spread_plus_pips: Optional[float] = None
    temp_up_trail_ema_period: Optional[int] = None

    # Uncle Parsh H1 Breakout: Discipline overrides
    temp_up_max_spread_pips: Optional[float] = None

    # Trial #8 exit strategy temp overrides
    temp_t8_exit_strategy: Optional[str] = None
    temp_t8_tp1_pips: Optional[float] = None
    temp_t8_tp1_close_pct: Optional[float] = None
    temp_t8_be_spread_plus_pips: Optional[float] = None
    temp_t8_trail_ema_period: Optional[int] = None
    temp_t8_m1_exit_ema_fast: Optional[int] = None
    temp_t8_m1_exit_ema_slow: Optional[int] = None
    temp_t8_scale_out_pct: Optional[float] = None
    temp_t8_initial_sl_spread_plus_pips: Optional[float] = None

    # Trial #9 exit strategy temp overrides
    temp_t9_exit_strategy: Optional[str] = None
    temp_t9_hwm_trail_pips: Optional[float] = None
    temp_t9_tp1_pips: Optional[float] = None
    temp_t9_tp1_close_pct: Optional[float] = None
    temp_t9_be_spread_plus_pips: Optional[float] = None
    temp_t9_trail_ema_period: Optional[int] = None
    temp_t9_trail_m5_ema_period: Optional[int] = None

    # Trial #10 proof / regime temp overrides
    temp_t10_regime_gate_enabled: Optional[bool] = None
    temp_t10_regime_london_sell_veto: Optional[bool] = None
    temp_t10_regime_london_start_hour_et: Optional[int] = None
    temp_t10_regime_london_end_hour_et: Optional[int] = None
    temp_t10_regime_boost_multiplier: Optional[float] = None
    temp_t10_regime_buy_base_multiplier: Optional[float] = None
    temp_t10_regime_sell_base_multiplier: Optional[float] = None
    temp_t10_regime_chop_pause_enabled: Optional[bool] = None
    temp_t10_regime_chop_pause_minutes: Optional[int] = None
    temp_t10_regime_chop_pause_lookback_trades: Optional[int] = None
    temp_t10_regime_chop_pause_stop_rate: Optional[float] = None
    temp_t10_tier17_nonboost_multiplier: Optional[float] = None
    temp_t10_bucketed_exit_enabled: Optional[bool] = None
    temp_t10_quick_tp1_pips: Optional[float] = None
    temp_t10_quick_tp1_close_pct: Optional[float] = None
    temp_t10_quick_be_spread_plus_pips: Optional[float] = None
    temp_t10_runner_tp1_pips: Optional[float] = None
    temp_t10_runner_tp1_close_pct: Optional[float] = None
    temp_t10_runner_be_spread_plus_pips: Optional[float] = None
    temp_t10_trail_escalation_enabled: Optional[bool] = None
    temp_t10_trail_escalation_tier1_pips: Optional[float] = None
    temp_t10_trail_escalation_tier2_pips: Optional[float] = None
    temp_t10_trail_escalation_m15_ema_period: Optional[int] = None
    temp_t10_trail_escalation_m15_buffer_pips: Optional[float] = None
    temp_t10_runner_score_sizing_enabled: Optional[bool] = None
    temp_t10_runner_base_lots: Optional[float] = None
    temp_t10_runner_min_lots: Optional[float] = None
    temp_t10_runner_max_lots: Optional[float] = None
    temp_t10_atr_stop_enabled: Optional[bool] = None
    temp_t10_atr_stop_multiplier: Optional[float] = None
    temp_t10_atr_stop_max_pips: Optional[float] = None

    # Trial #10 live chop-pause state
    chop_pause_buy_start_utc: Optional[str] = None
    chop_pause_buy_reason: Optional[str] = None
    chop_pause_sell_start_utc: Optional[str] = None
    chop_pause_sell_reason: Optional[str] = None


def _load_tier_fired(data: dict) -> dict:
    """Load tier_fired from JSON data with backward compat for old tier_X_fired keys."""
    # New format: single dict
    if "tier_fired" in data and isinstance(data["tier_fired"], dict):
        return {int(k): bool(v) for k, v in data["tier_fired"].items()}
    # Legacy format: individual tier_X_fired keys
    result = {}
    for key, val in data.items():
        if key.startswith("tier_") and key.endswith("_fired") and key != "tier_fired":
            try:
                period = int(key.replace("tier_", "").replace("_fired", ""))
                result[period] = bool(val)
            except ValueError:
                pass
    return result


def load_state(path: str | Path) -> RuntimeState:
    p = Path(path)
    data = load_json_state(p, default={})
    if not data:
        return RuntimeState()
    return RuntimeState(
        mode=data.get("mode", "DISARMED"),
        kill_switch=bool(data.get("kill_switch", False)),
        exit_system_only=bool(data.get("exit_system_only", False)),
        last_processed_bar_time_utc=data.get("last_processed_bar_time_utc"),
        temp_m5_trend_ema_fast=data.get("temp_m5_trend_ema_fast"),
        temp_m5_trend_ema_slow=data.get("temp_m5_trend_ema_slow"),
        temp_m5_trend_source=data.get("temp_m5_trend_source"),
        temp_m1_zone_entry_ema_slow=data.get("temp_m1_zone_entry_ema_slow"),
        temp_m1_pullback_cross_ema_slow=data.get("temp_m1_pullback_cross_ema_slow"),
        temp_m3_trend_ema_fast=data.get("temp_m3_trend_ema_fast"),
        temp_m3_trend_ema_slow=data.get("temp_m3_trend_ema_slow"),
        temp_m1_t4_zone_entry_ema_fast=data.get("temp_m1_t4_zone_entry_ema_fast"),
        temp_m1_t4_zone_entry_ema_slow=data.get("temp_m1_t4_zone_entry_ema_slow"),
        tier_fired=_load_tier_fired(data),
        divergence_block_buy_until=data.get("divergence_block_buy_until"),
        divergence_block_sell_until=data.get("divergence_block_sell_until"),
        daily_reset_date=data.get("daily_reset_date"),
        daily_reset_high=data.get("daily_reset_high"),
        daily_reset_low=data.get("daily_reset_low"),
        daily_reset_block_active=bool(data.get("daily_reset_block_active", False)),
        daily_reset_settled=bool(data.get("daily_reset_settled", False)),
        trend_flip_price=data.get("trend_flip_price"),
        trend_flip_direction=data.get("trend_flip_direction"),
        trend_flip_time=data.get("trend_flip_time"),
        bb_tier_fired={int(k): bool(v) for k, v in data.get("bb_tier_fired", {}).items()},
        temp_up_m5_ema_fast=data.get("temp_up_m5_ema_fast"),
        temp_up_m5_ema_slow=data.get("temp_up_m5_ema_slow"),
        temp_up_major_extremes_only=data.get("temp_up_major_extremes_only"),
        temp_up_h1_lookback_hours=data.get("temp_up_h1_lookback_hours"),
        temp_up_h1_swing_strength=data.get("temp_up_h1_swing_strength"),
        temp_up_h1_cluster_tolerance_pips=data.get("temp_up_h1_cluster_tolerance_pips"),
        temp_up_h1_min_touches_for_major=data.get("temp_up_h1_min_touches_for_major"),
        temp_up_power_close_body_pct=data.get("temp_up_power_close_body_pct"),
        temp_up_velocity_pips=data.get("temp_up_velocity_pips"),
        temp_up_initial_sl_spread_plus_pips=data.get("temp_up_initial_sl_spread_plus_pips"),
        temp_up_tp1_pips=data.get("temp_up_tp1_pips"),
        temp_up_tp1_close_pct=data.get("temp_up_tp1_close_pct"),
        temp_up_be_spread_plus_pips=data.get("temp_up_be_spread_plus_pips"),
        temp_up_trail_ema_period=data.get("temp_up_trail_ema_period"),
        temp_up_max_spread_pips=data.get("temp_up_max_spread_pips"),
        temp_t8_exit_strategy=data.get("temp_t8_exit_strategy"),
        temp_t8_tp1_pips=data.get("temp_t8_tp1_pips"),
        temp_t8_tp1_close_pct=data.get("temp_t8_tp1_close_pct"),
        temp_t8_be_spread_plus_pips=data.get("temp_t8_be_spread_plus_pips"),
        temp_t8_trail_ema_period=data.get("temp_t8_trail_ema_period"),
        temp_t8_m1_exit_ema_fast=data.get("temp_t8_m1_exit_ema_fast"),
        temp_t8_m1_exit_ema_slow=data.get("temp_t8_m1_exit_ema_slow"),
        temp_t8_scale_out_pct=data.get("temp_t8_scale_out_pct"),
        temp_t8_initial_sl_spread_plus_pips=data.get("temp_t8_initial_sl_spread_plus_pips"),
        temp_t9_exit_strategy=data.get("temp_t9_exit_strategy"),
        temp_t9_hwm_trail_pips=data.get("temp_t9_hwm_trail_pips"),
        temp_t9_tp1_pips=data.get("temp_t9_tp1_pips"),
        temp_t9_tp1_close_pct=data.get("temp_t9_tp1_close_pct"),
        temp_t9_be_spread_plus_pips=data.get("temp_t9_be_spread_plus_pips"),
        temp_t9_trail_ema_period=data.get("temp_t9_trail_ema_period"),
        temp_t9_trail_m5_ema_period=data.get("temp_t9_trail_m5_ema_period"),
        temp_t10_regime_gate_enabled=data.get("temp_t10_regime_gate_enabled"),
        temp_t10_regime_london_sell_veto=data.get("temp_t10_regime_london_sell_veto"),
        temp_t10_regime_london_start_hour_et=data.get("temp_t10_regime_london_start_hour_et"),
        temp_t10_regime_london_end_hour_et=data.get("temp_t10_regime_london_end_hour_et"),
        temp_t10_regime_boost_multiplier=data.get("temp_t10_regime_boost_multiplier"),
        temp_t10_regime_buy_base_multiplier=data.get("temp_t10_regime_buy_base_multiplier"),
        temp_t10_regime_sell_base_multiplier=data.get("temp_t10_regime_sell_base_multiplier"),
        temp_t10_regime_chop_pause_enabled=data.get("temp_t10_regime_chop_pause_enabled"),
        temp_t10_regime_chop_pause_minutes=data.get("temp_t10_regime_chop_pause_minutes"),
        temp_t10_regime_chop_pause_lookback_trades=data.get("temp_t10_regime_chop_pause_lookback_trades"),
        temp_t10_regime_chop_pause_stop_rate=data.get("temp_t10_regime_chop_pause_stop_rate"),
        temp_t10_tier17_nonboost_multiplier=data.get("temp_t10_tier17_nonboost_multiplier"),
        temp_t10_bucketed_exit_enabled=data.get("temp_t10_bucketed_exit_enabled"),
        temp_t10_quick_tp1_pips=data.get("temp_t10_quick_tp1_pips"),
        temp_t10_quick_tp1_close_pct=data.get("temp_t10_quick_tp1_close_pct"),
        temp_t10_quick_be_spread_plus_pips=data.get("temp_t10_quick_be_spread_plus_pips"),
        temp_t10_runner_tp1_pips=data.get("temp_t10_runner_tp1_pips"),
        temp_t10_runner_tp1_close_pct=data.get("temp_t10_runner_tp1_close_pct"),
        temp_t10_runner_be_spread_plus_pips=data.get("temp_t10_runner_be_spread_plus_pips"),
        temp_t10_trail_escalation_enabled=data.get("temp_t10_trail_escalation_enabled"),
        temp_t10_trail_escalation_tier1_pips=data.get("temp_t10_trail_escalation_tier1_pips"),
        temp_t10_trail_escalation_tier2_pips=data.get("temp_t10_trail_escalation_tier2_pips"),
        temp_t10_trail_escalation_m15_ema_period=data.get("temp_t10_trail_escalation_m15_ema_period"),
        temp_t10_trail_escalation_m15_buffer_pips=data.get("temp_t10_trail_escalation_m15_buffer_pips"),
        temp_t10_runner_score_sizing_enabled=data.get("temp_t10_runner_score_sizing_enabled"),
        temp_t10_runner_base_lots=data.get("temp_t10_runner_base_lots"),
        temp_t10_runner_min_lots=data.get("temp_t10_runner_min_lots"),
        temp_t10_runner_max_lots=data.get("temp_t10_runner_max_lots"),
        temp_t10_atr_stop_enabled=data.get("temp_t10_atr_stop_enabled"),
        temp_t10_atr_stop_multiplier=data.get("temp_t10_atr_stop_multiplier"),
        temp_t10_atr_stop_max_pips=data.get("temp_t10_atr_stop_max_pips"),
        chop_pause_buy_start_utc=data.get("chop_pause_buy_start_utc"),
        chop_pause_buy_reason=data.get("chop_pause_buy_reason"),
        chop_pause_sell_start_utc=data.get("chop_pause_sell_start_utc"),
        chop_pause_sell_reason=data.get("chop_pause_sell_reason"),
    )


def save_state(path: str | Path, state: RuntimeState) -> None:
    p = Path(path)
    existing = load_json_state(p, default={})
    payload = dict(existing) if isinstance(existing, dict) else {}
    payload.update(
        {
            "mode": state.mode,
            "kill_switch": state.kill_switch,
            "exit_system_only": state.exit_system_only,
            "last_processed_bar_time_utc": state.last_processed_bar_time_utc,
            "temp_m5_trend_ema_fast": state.temp_m5_trend_ema_fast,
            "temp_m5_trend_ema_slow": state.temp_m5_trend_ema_slow,
            "temp_m5_trend_source": state.temp_m5_trend_source,
            "temp_m1_zone_entry_ema_slow": state.temp_m1_zone_entry_ema_slow,
            "temp_m1_pullback_cross_ema_slow": state.temp_m1_pullback_cross_ema_slow,
            "temp_m3_trend_ema_fast": state.temp_m3_trend_ema_fast,
            "temp_m3_trend_ema_slow": state.temp_m3_trend_ema_slow,
            "temp_m1_t4_zone_entry_ema_fast": state.temp_m1_t4_zone_entry_ema_fast,
            "temp_m1_t4_zone_entry_ema_slow": state.temp_m1_t4_zone_entry_ema_slow,
            "tier_fired": state.tier_fired,
            "divergence_block_buy_until": state.divergence_block_buy_until,
            "divergence_block_sell_until": state.divergence_block_sell_until,
            "daily_reset_date": state.daily_reset_date,
            "daily_reset_high": state.daily_reset_high,
            "daily_reset_low": state.daily_reset_low,
            "daily_reset_block_active": state.daily_reset_block_active,
            "daily_reset_settled": state.daily_reset_settled,
            "trend_flip_price": state.trend_flip_price,
            "trend_flip_direction": state.trend_flip_direction,
            "trend_flip_time": state.trend_flip_time,
            "bb_tier_fired": state.bb_tier_fired,
            "temp_up_m5_ema_fast": state.temp_up_m5_ema_fast,
            "temp_up_m5_ema_slow": state.temp_up_m5_ema_slow,
            "temp_up_major_extremes_only": state.temp_up_major_extremes_only,
            "temp_up_h1_lookback_hours": state.temp_up_h1_lookback_hours,
            "temp_up_h1_swing_strength": state.temp_up_h1_swing_strength,
            "temp_up_h1_cluster_tolerance_pips": state.temp_up_h1_cluster_tolerance_pips,
            "temp_up_h1_min_touches_for_major": state.temp_up_h1_min_touches_for_major,
            "temp_up_power_close_body_pct": state.temp_up_power_close_body_pct,
            "temp_up_velocity_pips": state.temp_up_velocity_pips,
            "temp_up_initial_sl_spread_plus_pips": state.temp_up_initial_sl_spread_plus_pips,
            "temp_up_tp1_pips": state.temp_up_tp1_pips,
            "temp_up_tp1_close_pct": state.temp_up_tp1_close_pct,
            "temp_up_be_spread_plus_pips": state.temp_up_be_spread_plus_pips,
            "temp_up_trail_ema_period": state.temp_up_trail_ema_period,
            "temp_up_max_spread_pips": state.temp_up_max_spread_pips,
            "temp_t8_exit_strategy": state.temp_t8_exit_strategy,
            "temp_t8_tp1_pips": state.temp_t8_tp1_pips,
            "temp_t8_tp1_close_pct": state.temp_t8_tp1_close_pct,
            "temp_t8_be_spread_plus_pips": state.temp_t8_be_spread_plus_pips,
            "temp_t8_trail_ema_period": state.temp_t8_trail_ema_period,
            "temp_t8_m1_exit_ema_fast": state.temp_t8_m1_exit_ema_fast,
            "temp_t8_m1_exit_ema_slow": state.temp_t8_m1_exit_ema_slow,
            "temp_t8_scale_out_pct": state.temp_t8_scale_out_pct,
            "temp_t8_initial_sl_spread_plus_pips": state.temp_t8_initial_sl_spread_plus_pips,
            "temp_t9_exit_strategy": state.temp_t9_exit_strategy,
            "temp_t9_hwm_trail_pips": state.temp_t9_hwm_trail_pips,
            "temp_t9_tp1_pips": state.temp_t9_tp1_pips,
            "temp_t9_tp1_close_pct": state.temp_t9_tp1_close_pct,
            "temp_t9_be_spread_plus_pips": state.temp_t9_be_spread_plus_pips,
            "temp_t9_trail_ema_period": state.temp_t9_trail_ema_period,
            "temp_t9_trail_m5_ema_period": state.temp_t9_trail_m5_ema_period,
            "temp_t10_regime_gate_enabled": state.temp_t10_regime_gate_enabled,
            "temp_t10_regime_london_sell_veto": state.temp_t10_regime_london_sell_veto,
            "temp_t10_regime_london_start_hour_et": state.temp_t10_regime_london_start_hour_et,
            "temp_t10_regime_london_end_hour_et": state.temp_t10_regime_london_end_hour_et,
            "temp_t10_regime_boost_multiplier": state.temp_t10_regime_boost_multiplier,
            "temp_t10_regime_buy_base_multiplier": state.temp_t10_regime_buy_base_multiplier,
            "temp_t10_regime_sell_base_multiplier": state.temp_t10_regime_sell_base_multiplier,
            "temp_t10_regime_chop_pause_enabled": state.temp_t10_regime_chop_pause_enabled,
            "temp_t10_regime_chop_pause_minutes": state.temp_t10_regime_chop_pause_minutes,
            "temp_t10_regime_chop_pause_lookback_trades": state.temp_t10_regime_chop_pause_lookback_trades,
            "temp_t10_regime_chop_pause_stop_rate": state.temp_t10_regime_chop_pause_stop_rate,
            "temp_t10_tier17_nonboost_multiplier": state.temp_t10_tier17_nonboost_multiplier,
            "temp_t10_bucketed_exit_enabled": state.temp_t10_bucketed_exit_enabled,
            "temp_t10_quick_tp1_pips": state.temp_t10_quick_tp1_pips,
            "temp_t10_quick_tp1_close_pct": state.temp_t10_quick_tp1_close_pct,
            "temp_t10_quick_be_spread_plus_pips": state.temp_t10_quick_be_spread_plus_pips,
            "temp_t10_runner_tp1_pips": state.temp_t10_runner_tp1_pips,
            "temp_t10_runner_tp1_close_pct": state.temp_t10_runner_tp1_close_pct,
            "temp_t10_runner_be_spread_plus_pips": state.temp_t10_runner_be_spread_plus_pips,
            "temp_t10_trail_escalation_enabled": state.temp_t10_trail_escalation_enabled,
            "temp_t10_trail_escalation_tier1_pips": state.temp_t10_trail_escalation_tier1_pips,
            "temp_t10_trail_escalation_tier2_pips": state.temp_t10_trail_escalation_tier2_pips,
            "temp_t10_trail_escalation_m15_ema_period": state.temp_t10_trail_escalation_m15_ema_period,
            "temp_t10_trail_escalation_m15_buffer_pips": state.temp_t10_trail_escalation_m15_buffer_pips,
            "temp_t10_runner_score_sizing_enabled": state.temp_t10_runner_score_sizing_enabled,
            "temp_t10_runner_base_lots": state.temp_t10_runner_base_lots,
            "temp_t10_runner_min_lots": state.temp_t10_runner_min_lots,
            "temp_t10_runner_max_lots": state.temp_t10_runner_max_lots,
            "temp_t10_atr_stop_enabled": state.temp_t10_atr_stop_enabled,
            "temp_t10_atr_stop_multiplier": state.temp_t10_atr_stop_multiplier,
            "temp_t10_atr_stop_max_pips": state.temp_t10_atr_stop_max_pips,
            "chop_pause_buy_start_utc": state.chop_pause_buy_start_utc,
            "chop_pause_buy_reason": state.chop_pause_buy_reason,
            "chop_pause_sell_start_utc": state.chop_pause_sell_start_utc,
            "chop_pause_sell_reason": state.chop_pause_sell_reason,
        }
    )
    save_json_state(
        p,
        payload,
        indent=2,
        trailing_newline=True,
    )
