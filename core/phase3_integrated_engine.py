"""
Phase 3 Integrated Engine.

Single policy routes by UTC session:
- Tokyo: V14 mean reversion
- London: London V2 (Setup A + Setup D)
- NY: V44 session momentum
"""
from __future__ import annotations

import json
import math
import os
import re
import time
import hashlib
from functools import lru_cache
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from core.signal_engine import drop_incomplete_last_bar
from core.fib_pivots import compute_daily_fib_pivots

try:
    from core.execution_engine import ExecutionDecision
except Exception:  # pragma: no cover - runtime fallback
    @dataclass(frozen=True)
    class ExecutionDecision:  # type: ignore[no-redef]
        attempted: bool
        placed: bool
        reason: str
        order_retcode: Optional[int] = None
        order_id: Optional[int] = None
        deal_id: Optional[int] = None
        side: Optional[str] = None
        fill_price: Optional[float] = None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _phase3_order_confirmed(adapter, profile, order_result) -> tuple[bool, Optional[int]]:
    """
    Confirm a market order actually opened on the broker before Phase 3 treats it
    as placed. For OANDA, a market order can return an order id with no immediate
    fill/deal; those must not be counted as open trades.
    """
    deal_id = getattr(order_result, "deal_id", None)
    if deal_id is not None:
        return True, int(deal_id)
    if getattr(profile, "broker_type", None) != "oanda":
        return True, None
    order_id = getattr(order_result, "order_id", None)
    if order_id is None:
        return False, None
    for _ in range(3):
        try:
            position_id = adapter.get_position_id_from_order(int(order_id))
            if position_id is not None:
                try:
                    setattr(order_result, "deal_id", int(position_id))
                except Exception:
                    pass
                return True, int(position_id)
        except Exception:
            position_id = None
        time.sleep(0.5)
    return False, None


def _normalize_v14_source(v14_src: dict[str, Any]) -> dict[str, Any]:
    ps = v14_src.get("position_sizing", {}) if isinstance(v14_src.get("position_sizing"), dict) else {}
    tm = v14_src.get("trade_management", {}) if isinstance(v14_src.get("trade_management"), dict) else {}
    exec_model = v14_src.get("execution_model", {}) if isinstance(v14_src.get("execution_model"), dict) else {}
    ind = v14_src.get("indicators", {}) if isinstance(v14_src.get("indicators"), dict) else {}
    atr_i = ind.get("atr", {}) if isinstance(ind.get("atr"), dict) else {}
    adx_i = v14_src.get("adx_filter", {}) if isinstance(v14_src.get("adx_filter"), dict) else {}
    ent = v14_src.get("entry_rules", {}) if isinstance(v14_src.get("entry_rules"), dict) else {}
    long_e = ent.get("long", {}) if isinstance(ent.get("long"), dict) else {}
    short_e = ent.get("short", {}) if isinstance(ent.get("short"), dict) else {}
    core_gate = ent.get("core_gate", {}) if isinstance(ent.get("core_gate"), dict) else {}
    ex = v14_src.get("exit_rules", {}) if isinstance(v14_src.get("exit_rules"), dict) else {}
    ex_tp = ex.get("take_profit", {}) if isinstance(ex.get("take_profit"), dict) else {}
    ex_sl = ex.get("stop_loss", {}) if isinstance(ex.get("stop_loss"), dict) else {}
    ex_tr = ex.get("trailing_stop", {}) if isinstance(ex.get("trailing_stop"), dict) else {}
    ex_tm = ex.get("time_exit", {}) if isinstance(ex.get("time_exit"), dict) else {}
    ccf = v14_src.get("confluence_combo_filter", {}) if isinstance(v14_src.get("confluence_combo_filter"), dict) else {}
    ss = v14_src.get("signal_strength_tracking", {}) if isinstance(v14_src.get("signal_strength_tracking"), dict) else {}
    conf = v14_src.get("entry_confirmation", {}) if isinstance(v14_src.get("entry_confirmation"), dict) else {}
    sf = v14_src.get("session_filter", {}) if isinstance(v14_src.get("session_filter"), dict) else {}
    long_zone = long_e.get("price_zone", {}) if isinstance(long_e.get("price_zone"), dict) else {}
    long_rsi = long_e.get("rsi_soft_filter", {}) if isinstance(long_e.get("rsi_soft_filter"), dict) else {}
    short_rsi = short_e.get("rsi_soft_filter", {}) if isinstance(short_e.get("rsi_soft_filter"), dict) else {}
    long_conf = long_e.get("confluence_scoring", {}) if isinstance(long_e.get("confluence_scoring"), dict) else {}
    short_conf = short_e.get("confluence_scoring", {}) if isinstance(short_e.get("confluence_scoring"), dict) else {}
    bb_regime = ind.get("bb_width_regime_filter", {}) if isinstance(ind.get("bb_width_regime_filter"), dict) else {}
    allowed_days = sf.get("allowed_trading_days", ["Tuesday", "Wednesday", "Friday"])
    return {
        "risk_per_trade_pct": float(ps.get("risk_per_trade_pct", 2.0)),
        "max_units": int(ps.get("max_units", MAX_UNITS)),
        "max_concurrent_positions": int(ps.get("max_concurrent_positions", 2)),
        "leverage": float(ps.get("leverage", LEVERAGE)),
        "day_risk_multipliers": ps.get("day_risk_multipliers", {}) if isinstance(ps.get("day_risk_multipliers"), dict) else {},
        "max_trades_per_session": int(tm.get("max_trades_per_session", 4)),
        "stop_after_consecutive_losses": int(tm.get("stop_after_consecutive_losses", 3)),
        "session_loss_stop_pct": float(tm.get("session_loss_stop_pct", 1.5)),
        "cooldown_minutes": int(tm.get("cooldown_minutes", tm.get("min_time_between_entries_minutes", 10))),
        "min_time_between_entries_minutes": int(tm.get("min_time_between_entries_minutes", 10)),
        "no_reentry_same_direction_after_stop_minutes": int(tm.get("no_reentry_same_direction_after_stop_minutes", 30)),
        "disable_entries_if_move_from_tokyo_open_range_exceeds_pips": float(tm.get("disable_entries_if_move_from_tokyo_open_range_exceeds_pips", 0.0)),
        "breakout_detection_mode": str(tm.get("breakout_detection_mode", "rolling")).lower(),
        "rolling_window_minutes": int(tm.get("rolling_window_minutes", 60)),
        "rolling_range_threshold_pips": float(tm.get("rolling_range_threshold_pips", 40.0)),
        "atr_max_threshold_price_units": float(atr_i.get("max_threshold_price_units", ATR_MAX)),
        "adx_max_for_entry": float(adx_i.get("max_adx_for_entry", ADX_MAX)),
        "adx_filter_enabled": bool(adx_i.get("enabled", True)),
        "bb_regime_mode": str(bb_regime.get("mode", "percentile")).strip().lower(),
        "bb_width_lookback": int(bb_regime.get("percentile_lookback_m5_bars", BB_WIDTH_LOOKBACK)),
        "bb_width_ranging_pct": float(bb_regime.get("ranging_percentile", BB_WIDTH_RANGING_PCT)),
        "zone_tolerance_pips": float(long_zone.get("tolerance_pips", ZONE_TOLERANCE_PIPS)),
        "rsi_long_entry": float(long_rsi.get("entry_soft_threshold", RSI_LONG_ENTRY)),
        "rsi_short_entry": float(short_rsi.get("entry_soft_threshold", RSI_SHORT_ENTRY)),
        "min_confluence": int(min(long_conf.get("minimum_score", MIN_CONFLUENCE), short_conf.get("minimum_score", MIN_CONFLUENCE))),
        "confluence_min_long": int(long_conf.get("minimum_score", MIN_CONFLUENCE)),
        "confluence_min_short": int(short_conf.get("minimum_score", MIN_CONFLUENCE)),
        "blocked_combos": ccf.get("blocked_combos", list(BLOCKED_COMBOS)) if bool(ccf.get("enabled", True)) else [],
        "core_gate_required": int(core_gate.get("required_count", 2)),
        "core_gate_use_zone": bool(core_gate.get("use_zone", True)),
        "core_gate_use_bb": bool(core_gate.get("use_bb_touch", True)),
        "core_gate_use_sar": bool(core_gate.get("use_sar_flip", True)),
        "core_gate_use_rsi": bool(core_gate.get("use_rsi_soft", True)),
        "sl_buffer_pips": float(ex_sl.get("buffer_pips", SL_BUFFER_PIPS)),
        "min_sl_pips": float(ex_sl.get("minimum_sl_pips", SL_MIN_PIPS)),
        "max_sl_pips": float(ex_sl.get("hard_max_sl_pips", SL_MAX_PIPS)),
        "partial_tp_atr_mult": float(ex_tp.get("partial_tp_atr_mult", TP1_ATR_MULT)),
        "partial_tp_min_pips": float(ex_tp.get("partial_tp_min_pips", TP1_MIN_PIPS)),
        "partial_tp_max_pips": float(ex_tp.get("partial_tp_max_pips", TP1_MAX_PIPS)),
        "tp1_close_pct": float(ex_tp.get("partial_close_pct", TP1_CLOSE_PCT)),
        "breakeven_offset_pips": float(ex_tp.get("breakeven_offset_pips", BE_OFFSET_PIPS)),
        "trail_activate_profit_pips": float(ex_tr.get("activate_after_profit_pips", TRAIL_ACTIVATE_PROFIT_PIPS)),
        "trail_distance_pips": float(ex_tr.get("trail_distance_pips", TRAIL_DISTANCE_PIPS)),
        "time_decay_minutes": int(ex_tm.get("time_decay_minutes", TIME_DECAY_MINUTES)),
        "time_decay_profit_cap_pips": float(ex_tm.get("time_decay_profit_cap_pips", TIME_DECAY_CAP_PIPS)),
        "min_tp_distance_pips": float(ex_tp.get("minimum_tp_distance_pips", MIN_TP_DISTANCE_PIPS)),
        "confirmation_enabled": bool(conf.get("enabled", False)),
        "confirmation_type": str(conf.get("type", "m1")).lower(),
        "confirmation_window_bars": int(conf.get("window_bars", 0)),
        "signal_strength_tracking": ss if isinstance(ss, dict) else {},
        "session_start_utc": str(sf.get("session_start_utc", "16:00")),
        "session_end_utc": str(sf.get("session_end_utc", "22:00")),
        "allowed_trading_days": list(allowed_days) if isinstance(allowed_days, list) else ["Tuesday", "Wednesday", "Friday"],
        # Backtest parity: Tokyo backtest defaults to 10.0 when this key is absent.
        "max_entry_spread_pips": float(exec_model.get("max_entry_spread_pips", 10.0)),
    }


def _normalize_london_source(v2_src: dict[str, Any]) -> dict[str, Any]:
    account = v2_src.get("account", {}) if isinstance(v2_src.get("account"), dict) else {}
    session = v2_src.get("session", {}) if isinstance(v2_src.get("session"), dict) else {}
    execution_model = v2_src.get("execution_model", {}) if isinstance(v2_src.get("execution_model"), dict) else {}
    risk = v2_src.get("risk", {}) if isinstance(v2_src.get("risk"), dict) else {}
    limits = v2_src.get("entry_limits", {}) if isinstance(v2_src.get("entry_limits"), dict) else {}
    levels = v2_src.get("levels", {}) if isinstance(v2_src.get("levels"), dict) else {}
    setups = v2_src.get("setups", {}) if isinstance(v2_src.get("setups"), dict) else {}
    sa = setups.get("A", {}) if isinstance(setups.get("A"), dict) else {}
    sd = setups.get("D", {}) if isinstance(setups.get("D"), dict) else {}
    return {
        "risk_per_trade_pct": float(risk.get("risk_per_trade_pct", 0.01)),
        "arb_risk_per_trade_pct": float(sa.get("risk_per_trade_pct", risk.get("risk_per_trade_pct", 0.01))),
        "d_risk_per_trade_pct": float(sd.get("risk_per_trade_pct", risk.get("risk_per_trade_pct", 0.01))),
        "max_total_open_risk_pct": float(risk.get("max_total_open_risk_pct", 0.05)),
        "max_trades_per_day_total": limits.get("max_trades_per_day_total", 1),
        "max_trades_per_setup_per_day": limits.get("max_trades_per_setup_per_day", None),
        "max_trades_per_setup_direction_per_day": limits.get("max_trades_per_setup_direction_per_day", None),
        "disable_channel_reset_after_exit": bool(limits.get("disable_channel_reset_after_exit", False)),
        "active_days_utc": session.get("active_days_utc", ["Tuesday", "Wednesday"]),
        "arb_range_min_pips": float(levels.get("asian_range_min_pips", 30.0)),
        "arb_range_max_pips": float(levels.get("asian_range_max_pips", 60.0)),
        "lor_range_min_pips": float(levels.get("lor_range_min_pips", 4.0)),
        "lor_range_max_pips": float(levels.get("lor_range_max_pips", 20.0)),
        "d_entry_start_min_after_london": int(sd.get("entry_start_min_after_london", 15)),
        "d_entry_end_min_after_london": int(sd.get("entry_end_min_after_london", 120)),
        "d_allow_long": bool(sd.get("allow_long", True)),
        "d_allow_short": bool(sd.get("allow_short", False)),
        "hard_close_at_ny_open": bool(session.get("hard_close_at_ny_open", True)),
        "arb_tp1_r": float(sa.get("tp1_r_multiple", 1.0)),
        "arb_tp2_r": float(sa.get("tp2_r_multiple", 2.0)),
        "arb_tp1_close_pct": float(sa.get("tp1_close_fraction", 0.5)),
        "arb_be_offset_pips": float(sa.get("be_offset_pips", 1.0)),
        "d_tp1_r": float(sd.get("tp1_r_multiple", 1.0)),
        "d_tp2_r": float(sd.get("tp2_r_multiple", 2.0)),
        "d_tp1_close_pct": float(sd.get("tp1_close_fraction", 0.5)),
        "d_be_offset_pips": float(sd.get("be_offset_pips", 1.0)),
        "leverage": float(account.get("leverage", 33.0)),
        "max_margin_usage_fraction_per_trade": float(account.get("max_margin_usage_fraction_per_trade", 0.5)),
        "max_open_positions": int(account.get("max_open_positions", 5)),
        "a_allow_long": bool(sa.get("allow_long", True)),
        "a_allow_short": bool(sa.get("allow_short", True)),
        "a_entry_start_min_after_london": int(sa.get("entry_start_min_after_london", 0)),
        "a_entry_end_min_after_london": int(sa.get("entry_end_min_after_london", 90)),
        "a_reset_mode": str(sa.get("reset_mode", "touch_level")),
        "d_reset_mode": str(sd.get("reset_mode", "touch_level")),
        "max_entry_spread_pips": float(execution_model.get("spread_max_pips", LDN_MAX_SPREAD_PIPS)),
    }


def _normalize_v44_source(v44_src: dict[str, Any]) -> dict[str, Any]:
    ny_start_raw = v44_src.get("ny_start", v44_src.get("v5_ny_start"))
    ny_end_raw = v44_src.get("ny_end", v44_src.get("v5_ny_end"))
    # The benchmark session_momentum backtest uses fixed UTC ny_start/ny_end values.
    # Default live normalization to fixed_utc unless the source config explicitly opts
    # into DST-aware routing.
    ny_window_mode = str(v44_src.get("ny_window_mode", "fixed_utc")).lower()
    ny_daily_loss_usd = v44_src.get("v5_ny_daily_loss_usd", v44_src.get("v5_daily_loss_limit_usd", 0.0))
    return {
        "risk_per_trade_pct": float(v44_src.get("v5_risk_per_trade_pct", V44_RISK_PCT) or V44_RISK_PCT),
        "rp_min_lot": float(v44_src.get("v5_rp_min_lot", 1.0)),
        "rp_max_lot": float(v44_src.get("v5_rp_max_lot", 20.0)),
        "max_open_positions": int(v44_src.get("v5_max_open", V44_MAX_OPEN)),
        "max_entries_per_day": int(v44_src.get("v5_max_entries_day", V44_MAX_ENTRIES_DAY)),
        "session_stop_losses": int(v44_src.get("v5_session_stop_losses", V44_SESSION_STOP_LOSSES)),
        "session_entry_cutoff_minutes": int(v44_src.get("v5_session_entry_cutoff_minutes", 60)),
        "session_range_cap_pips": float(v44_src.get("v5_session_range_cap_pips", 0.0)),
        "cooldown_win_bars": int(v44_src.get("v5_cooldown_win", V44_COOLDOWN_WIN)),
        "cooldown_loss_bars": int(v44_src.get("v5_cooldown_loss", V44_COOLDOWN_LOSS)),
        "cooldown_scratch_bars": int(v44_src.get("v5_cooldown_scratch", 2)),
        "start_delay_minutes": int(v44_src.get("v5_ny_start_delay_minutes", 5)),
        "max_entry_spread_pips": float(v44_src.get("v5_max_entry_spread_pips", v44_src.get("max_entry_spread_pips", V44_MAX_ENTRY_SPREAD))),
        "ny_strength_allow": str(v44_src.get("v5_ny_strength_allow", v44_src.get("v5_strength_allow", "strong_normal"))),
        "skip_weak": bool(v44_src.get("v5_skip_weak", True)),
        "skip_normal": bool(v44_src.get("v5_skip_normal", False)),
        "allow_normal_plus": bool(v44_src.get("v5_allow_normal_plus", False)),
        "normalplus_atr_min_pips": float(v44_src.get("v5_normalplus_atr_min_pips", 6.0)),
        "normalplus_slope_min": float(v44_src.get("v5_normalplus_slope_min", 0.45)),
        "news_filter_enabled": bool(v44_src.get("v5_news_filter_enabled", False)),
        "news_window_minutes_before": int(v44_src.get("v5_news_window_minutes_before", 60)),
        "news_window_minutes_after": int(v44_src.get("v5_news_window_minutes_after", 30)),
        "news_impact_min": str(v44_src.get("v5_news_impact_min", "high")),
        "news_calendar_path": str(v44_src.get("v5_news_calendar_path", "research_out/v5_scheduled_events_utc.csv")),
        "news_trend_enabled": bool(v44_src.get("v5_news_trend_enabled", False)),
        "news_trend_delay_minutes": int(v44_src.get("v5_news_trend_delay_minutes", 45)),
        "news_trend_window_minutes": int(v44_src.get("v5_news_trend_window_minutes", 90)),
        "news_trend_confirm_bars": int(v44_src.get("v5_news_trend_confirm_bars", 3)),
        "news_trend_min_body_pips": float(v44_src.get("v5_news_trend_min_body_pips", 1.5)),
        "news_trend_require_ema_align": bool(v44_src.get("v5_news_trend_require_ema_align", True)),
        "news_trend_risk_pct": float(v44_src.get("v5_news_trend_risk_pct", 0.5)),
        "news_trend_tp_rr": float(v44_src.get("v5_news_trend_tp_rr", 1.5)),
        "news_trend_sl_pips": float(v44_src.get("v5_news_trend_sl_pips", 8.0)),
        "atr_pct_filter_enabled": bool(v44_src.get("v5_atr_pct_filter_enabled", False)),
        "atr_pct_cap": float(v44_src.get("v5_atr_pct_cap", V44_ATR_PCT_CAP)),
        "atr_pct_lookback": int(v44_src.get("v5_atr_pct_lookback", V44_ATR_PCT_LOOKBACK)),
        "skip_days": str(v44_src.get("v5_skip_days", "")),
        "skip_months": str(v44_src.get("v5_skip_months", "")),
        "strong_tp1_pips": float(v44_src.get("v5_strong_tp1", V44_STRONG_TP1_PIPS)),
        "normal_tp1_pips": float(v44_src.get("v5_normal_tp1", 1.75)),
        "weak_tp1_pips": float(v44_src.get("v5_weak_tp1", 1.2)),
        "strong_tp2_pips": float(v44_src.get("v5_strong_tp2", V44_STRONG_TP2_PIPS)),
        "normal_tp2_pips": float(v44_src.get("v5_normal_tp2", 3.0)),
        "weak_tp2_pips": float(v44_src.get("v5_weak_tp2", 2.0)),
        "strong_tp1_close_pct": float(v44_src.get("v5_strong_tp1_close_pct", V44_STRONG_TP1_CLOSE_PCT)),
        "normal_tp1_close_pct": float(v44_src.get("v5_normal_tp1_close_pct", 0.5)),
        "weak_tp1_close_pct": float(v44_src.get("v5_weak_tp1_close_pct", 0.6)),
        "be_offset_pips": float(v44_src.get("v5_be_offset", V44_BE_OFFSET_PIPS)),
        "strong_trail_buffer": float(v44_src.get("v5_strong_trail_buffer_pips", v44_src.get("v5_strong_trail_buffer", V44_STRONG_TRAIL_BUFFER))),
        "normal_trail_buffer": float(v44_src.get("v5_normal_trail_buffer", 3.0)),
        "weak_trail_buffer": float(v44_src.get("v5_weak_trail_buffer", 2.0)),
        "strong_trail_ema": int(v44_src.get("v5_strong_trail_ema", 21)),
        "normal_trail_ema": int(v44_src.get("v5_normal_trail_ema", 9)),
        "trail_start_after_tp1_mult": float(v44_src.get("v5_trail_start_after_tp1_mult", 0.5)),
        "h1_ema_fast": int(v44_src.get("h1_ema_fast", V44_H1_EMA_FAST)),
        "h1_ema_slow": int(v44_src.get("h1_ema_slow", V44_H1_EMA_SLOW)),
        "h1_slope_min": float(v44_src.get("v5_h1_slope_min", 0.0)),
        "h1_slope_consistent_bars": int(v44_src.get("v5_h1_slope_consistent_bars", 0)),
        "m5_ema_fast": int(v44_src.get("v5_m5_ema_fast", V44_M5_EMA_FAST)),
        "m5_ema_slow": int(v44_src.get("v5_m5_ema_slow", V44_M5_EMA_SLOW)),
        "slope_bars": int(v44_src.get("v5_slope_bars", V44_SLOPE_BARS)),
        "strong_slope": float(v44_src.get("v5_strong_slope", V44_STRONG_SLOPE)),
        "weak_slope": float(v44_src.get("v5_weak_slope", 0.2)),
        "entry_min_body_pips": float(v44_src.get("v5_entry_min_body_pips", V44_MIN_BODY_PIPS)),
        "dual_mode_enabled": bool(v44_src.get("v5_dual_mode_enabled", False)),
        "trend_mode_efficiency_min": float(v44_src.get("v5_trend_mode_efficiency_min", 0.4)),
        "range_mode_efficiency_max": float(v44_src.get("v5_range_mode_efficiency_max", 0.3)),
        "range_fade_enabled": bool(v44_src.get("v5_range_fade_enabled", False)),
        "queued_confirm_bars": int(v44_src.get("v5_queued_confirm_bars", 0)),
        "london_confirm_bars": int(v44_src.get("v5_london_confirm_bars", 1)),
        "sizing_mode": str(v44_src.get("v5_sizing_mode", "risk_parity")).lower(),
        "base_lot": float(v44_src.get("v5_base_lot", 2.0)),
        "strong_size_mult": float(v44_src.get("v5_strong_size_mult", 3.0)),
        "normal_size_mult": float(v44_src.get("v5_normal_size_mult", 2.0)),
        "weak_size_mult": float(v44_src.get("v5_weak_size_mult", 0.0)),
        "win_bonus_per_step": float(v44_src.get("v5_win_bonus_per_step", 0.0)),
        "win_streak_scope": str(v44_src.get("v5_win_streak_scope", "session")),
        "max_lot": float(v44_src.get("v5_max_lot", 10.0)),
        "rp_strong_mult": float(v44_src.get("v5_rp_strong_mult", 1.0)),
        "rp_normal_mult": float(v44_src.get("v5_rp_normal_mult", 1.0)),
        "rp_weak_mult": float(v44_src.get("v5_rp_weak_mult", 0.0)),
        "rp_london_mult": float(v44_src.get("v5_rp_london_mult", 1.0)),
        "rp_ny_mult": float(v44_src.get("v5_rp_ny_mult", 1.0)),
        "rp_win_bonus_pct": float(v44_src.get("v5_rp_win_bonus_pct", 0.0)),
        "rp_max_lot_mult": float(v44_src.get("v5_rp_max_lot_mult", 2.0)),
        "hybrid_strong_boost": float(v44_src.get("v5_hybrid_strong_boost", 1.3)),
        "hybrid_normal_boost": float(v44_src.get("v5_hybrid_normal_boost", 0.8)),
        "hybrid_weak_boost": float(v44_src.get("v5_hybrid_weak_boost", 0.0)),
        "hybrid_ny_boost": float(v44_src.get("v5_hybrid_ny_boost", 1.0)),
        "h4_adx_min": float(v44_src.get("v5_h4_adx_min", 0.0)),
        "exhaustion_gate_enabled": bool(v44_src.get("v5_exhaustion_gate_enabled", False)),
        "exhaustion_gate_window_minutes": int(v44_src.get("v5_exhaustion_gate_window_minutes", 60)),
        "exhaustion_gate_max_range_pips": float(v44_src.get("v5_exhaustion_gate_max_range_pips", 40.0)),
        "exhaustion_gate_cooldown_minutes": int(v44_src.get("v5_exhaustion_gate_cooldown_minutes", 15)),
        "daily_loss_limit_pips": float(v44_src.get("v5_daily_loss_limit_pips", 0.0)),
        "daily_loss_limit_usd": float(ny_daily_loss_usd or 0.0),
        "weekly_loss_limit_pips": float(v44_src.get("v5_weekly_loss_limit_pips", 0.0)),
        "gl_enabled": bool(v44_src.get("v5_gl_enabled", False)),
        "gl_min_wins": int(v44_src.get("v5_gl_min_wins", 1)),
        "gl_extra_entries": int(v44_src.get("v5_gl_extra_entries", 2)),
        "gl_allow_normal": bool(v44_src.get("v5_gl_allow_normal", True)),
        "gl_normal_atr_cap": float(v44_src.get("v5_gl_normal_atr_cap", 0.67)),
        "gl_slope_relax": float(v44_src.get("v5_gl_slope_relax", 0.0)),
        "gl_max_open": int(v44_src.get("v5_gl_max_open", 0)),
        "ny_window_mode": ny_window_mode,
        "ny_start_hour": float(ny_start_raw) if ny_start_raw is not None else None,
        "ny_end_hour": float(ny_end_raw) if ny_end_raw is not None else None,
        "ny_duration_hours": float(v44_src.get("ny_duration_hours", 3.0)),
    }


def load_phase3_sizing_config(
    config_path: Optional[Path] = None,
    source_paths: Optional[dict[str, Path]] = None,
) -> dict[str, Any]:
    """
    Load effective Phase 3 config with precedence:
      source-of-truth configs -> optional runtime override -> code defaults.
    """
    root = Path(__file__).resolve().parent.parent
    src_paths = {
        "v14": root / "research_out" / "tokyo_optimized_v14_config.json",
        "london_v2": root / "research_out" / "v2_exp4_winner_baseline_config.json",
        "v44_ny": root / "research_out" / "session_momentum_v44_base_config.json",
    }
    if isinstance(source_paths, dict):
        for k in ("v14", "london_v2", "v44_ny"):
            p = source_paths.get(k)
            if isinstance(p, Path):
                src_paths[k] = p

    v14_src = _read_json(src_paths["v14"])
    ldn_src = _read_json(src_paths["london_v2"])
    v44_src = _read_json(src_paths["v44_ny"])

    normalized = {
        "v14": _normalize_v14_source(v14_src),
        "london_v2": _normalize_london_source(ldn_src),
        "v44_ny": _normalize_v44_source(v44_src),
    }

    if config_path is None:
        config_path = root / "research_out" / "phase3_integrated_sizing_config.json"
    overrides = _read_json(config_path) if config_path.exists() else {}
    effective = _deep_merge(normalized, overrides if isinstance(overrides, dict) else {})

    hash_input = json.dumps(effective, sort_keys=True, separators=(",", ":"))
    effective_hash = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()[:16]
    effective["_meta"] = {
        "effective_hash": effective_hash,
        "source_paths": {k: str(v) for k, v in src_paths.items()},
        "override_path": str(config_path),
        "loaded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return effective


# ---------------------------------------------------------------------------
# Session constants (single source of truth; aligned with backtest)
# ---------------------------------------------------------------------------
# London V2: UK-DST aware (07:00-11:00 UTC summer, 08:00-12:00 UTC winter)
#   ARB: London open -> +90m
#   Setup D: per config minutes after London open
# V44 NY: US-DST aware by default (12:00-15:00 UTC summer, 13:00-16:00 UTC winter)
#   plus configurable start delay (default +5m)
# Tokyo V14: fixed 16:00-22:00 UTC
TOKYO_START_UTC = 16  # 16:00 UTC
TOKYO_END_UTC = 22    # 22:00 UTC
TOKYO_ALLOWED_DAYS = {1, 2, 4}  # Tuesday=1, Wednesday=2, Friday=4  (weekday())

LONDON_START_UTC = 8
LONDON_END_UTC = 12
LONDON_ALLOWED_DAYS = {1, 2}  # Tue/Wed

NY_START_UTC = 13
NY_END_UTC = 16
NY_ALLOWED_DAYS = {0, 1, 2, 3, 4}  # Mon-Fri

# ---------------------------------------------------------------------------
# Indicator constants
# ---------------------------------------------------------------------------
BB_TF = "M5"
BB_PERIOD = 25
BB_STD = 2.2

BB_WIDTH_LOOKBACK = 100
BB_WIDTH_RANGING_PCT = 0.80  # below 80th percentile = ranging

PSAR_AF_START = 0.02
PSAR_AF_STEP = 0.02
PSAR_AF_MAX = 0.20
PSAR_FLIP_LOOKBACK = 12  # M1 bars

RSI_TF = "M5"
RSI_PERIOD = 14
RSI_LONG_ENTRY = 35
RSI_SHORT_ENTRY = 65

ATR_TF = "M15"
ATR_PERIOD = 14
ATR_MAX = 0.30  # price units (= 30 pips for USDJPY)

ADX_TF = "M15"
ADX_PERIOD = 14
ADX_MAX = 35

# ---------------------------------------------------------------------------
# Entry constants
# ---------------------------------------------------------------------------
MIN_CONFLUENCE = 2
ZONE_TOLERANCE_PIPS = 20
BLOCKED_COMBOS = {"ABCD"}

# ---------------------------------------------------------------------------
# SL / TP / Exit constants
# ---------------------------------------------------------------------------
SL_BUFFER_PIPS = 8
SL_MIN_PIPS = 12
SL_MAX_PIPS = 35

TP1_ATR_MULT = 0.5
TP1_MIN_PIPS = 6
TP1_MAX_PIPS = 12
TP1_CLOSE_PCT = 0.50  # 50 %

BE_OFFSET_PIPS = 2

TRAIL_ACTIVATE_PROFIT_PIPS = 8
TRAIL_DISTANCE_PIPS = 5
TRAIL_REQUIRES_TP1 = True
TRAIL_NEVER_WIDEN = True

TIME_DECAY_MINUTES = 120
TIME_DECAY_CAP_PIPS = 3.0

PRIMARY_TP_TARGET = "pivot_P"
MIN_TP_DISTANCE_PIPS = 8

# ---------------------------------------------------------------------------
# Sizing / risk
# ---------------------------------------------------------------------------
RISK_PCT = 0.02  # 2 %
MAX_UNITS = 500_000
MAX_CONCURRENT = 3
LEVERAGE = 33.3

# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------
MAX_TRADES_PER_SESSION = 4
COOLDOWN_MINUTES = 15
STOP_AFTER_CONSECUTIVE_LOSSES = 3
SESSION_LOSS_STOP_PCT = 0.015  # 1.5 %
MAX_ENTRY_SPREAD_PIPS = 3.0

PIP_SIZE = 0.01  # USDJPY

# ---------------------------------------------------------------------------
# London V2 constants (ARB + LMP)
# ---------------------------------------------------------------------------
LDN_ARB_BREAKOUT_BUFFER_PIPS = 7.0
LDN_ARB_SL_BUFFER_PIPS = 3.0
LDN_ARB_RANGE_MIN_PIPS = 30.0
LDN_ARB_RANGE_MAX_PIPS = 60.0
LDN_ARB_SL_MIN_PIPS = 15.0
LDN_ARB_SL_MAX_PIPS = 40.0
LDN_ARB_TP1_R = 1.0
LDN_ARB_TP2_R = 2.0
LDN_ARB_TP1_CLOSE_PCT = 0.5
LDN_ARB_BE_OFFSET_PIPS = 1.0
LDN_ARB_MAX_TRADES = 1

LDN_LMP_IMPULSE_MINUTES = 90
LDN_LMP_IMPULSE_MIN_PIPS = 20.0
LDN_LMP_ZONE_FIB = 0.5
LDN_LMP_SL_BUFFER_PIPS = 5.0
LDN_LMP_SL_MIN_PIPS = 12.0
LDN_LMP_SL_MAX_PIPS = 30.0
LDN_LMP_TP2_EXTENSION = 0.618
LDN_LMP_TP1_CLOSE_PCT = 0.5
LDN_LMP_BE_OFFSET_PIPS = 1.0
LDN_LMP_EMA_M15_PERIOD = 20
LDN_LMP_MAX_TRADES = 1

LDN_RISK_PCT = 0.0075
LDN_MAX_OPEN = 2
LDN_MAX_SPREAD_PIPS = 3.5
LDN_FORCE_CLOSE_AT_NY_OPEN = True

# London V2 Setup D (LOR breakout, long-only in exp4 baseline winner)
LDN_D_LOR_MIN_PIPS = 4.0
LDN_D_LOR_MAX_PIPS = 20.0
LDN_D_BREAKOUT_BUFFER_PIPS = 3.0
LDN_D_SL_BUFFER_PIPS = 3.0
LDN_D_SL_MIN_PIPS = 5.0
LDN_D_SL_MAX_PIPS = 20.0
LDN_D_TP1_R = 1.0
LDN_D_TP2_R = 2.0
LDN_D_MAX_TRADES = 1
LDN_MAX_TRADES_PER_DAY_TOTAL = 1

# ---------------------------------------------------------------------------
# V44 NY constants
# ---------------------------------------------------------------------------
V44_H1_EMA_FAST = 20
V44_H1_EMA_SLOW = 50

# Unused for Phase 3 session routing (we use LONDON_* / NY_* above)
# V44_LONDON_START = 8.5
# V44_LONDON_END = 11.0
V44_NY_START = 13.0
V44_NY_END = 16.0

V44_M5_EMA_FAST = 9
V44_M5_EMA_SLOW = 21
V44_SLOPE_BARS = 4
V44_STRONG_SLOPE = 0.5
V44_WEAK_SLOPE = 0.2
V44_LONDON_STRONG_SLOPE = 0.6
V44_MIN_BODY_PIPS = 1.5

V44_SL_LOOKBACK = 6
V44_SL_BUFFER_PIPS = 1.5
V44_SL_FLOOR_PIPS = 7.0
V44_SL_CAP_PIPS = 9.0

V44_STRONG_TP1_PIPS = 2.0
V44_STRONG_TP2_PIPS = 5.0
V44_STRONG_TP1_CLOSE_PCT = 0.3
V44_STRONG_TRAIL_BUFFER = 4.0
V44_LONDON_TP1_PIPS = 1.2
V44_LONDON_TRAIL_BUFFER = 2.0

V44_RISK_PCT = 0.005
V44_MAX_OPEN = 3
V44_MAX_ENTRIES_DAY = 7
V44_COOLDOWN_WIN = 1
V44_COOLDOWN_LOSS = 1
V44_SESSION_STOP_LOSSES = 3
V44_BE_OFFSET_PIPS = 0.5
V44_MAX_ENTRY_SPREAD = 3.0

V44_ATR_PCT_CAP = 0.67
V44_ATR_PCT_LOOKBACK = 200


# ===================================================================
#  Session classifier
# ===================================================================

_DAY_NAME_TO_WEEKDAY = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def _parse_hhmm_to_hour(value: Any, default_hour: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value or "").strip()
    m = re.match(r"^(\d{1,2}):(\d{2})$", s)
    if not m:
        return float(default_hour)
    hh = int(m.group(1))
    mm = int(m.group(2))
    return float(hh) + float(mm) / 60.0


def _weekday_set_from_names(values: Any, default_set: set[int]) -> set[int]:
    if not isinstance(values, (list, tuple, set)):
        return set(default_set)
    out: set[int] = set()
    for v in values:
        wd = _DAY_NAME_TO_WEEKDAY.get(str(v).strip().lower())
        if wd is not None:
            out.add(wd)
    return out if out else set(default_set)


def _in_hour_window(hour_frac: float, start_h: float, end_h: float) -> bool:
    if start_h <= end_h:
        return start_h <= hour_frac < end_h
    # Cross-midnight window.
    return hour_frac >= start_h or hour_frac < end_h


def _resolve_ny_window_hours(now_utc: datetime, v44_cfg: Optional[dict[str, Any]] = None) -> tuple[float, float]:
    """Resolve NY start/end hours in UTC. Default is DST-aware unless fixed_utc is requested."""
    cfg = v44_cfg if isinstance(v44_cfg, dict) else {}
    mode = str(cfg.get("ny_window_mode", "dst_auto")).strip().lower()
    if mode == "fixed_utc":
        raw_start = cfg.get("ny_start_hour", NY_START_UTC)
        raw_end = cfg.get("ny_end_hour", NY_END_UTC)
        ny_start = float(NY_START_UTC if raw_start is None else raw_start)
        ny_end = float(NY_END_UTC if raw_end is None else raw_end)
        if ny_end <= ny_start:
            ny_end = ny_start + max(1.0, float(cfg.get("ny_duration_hours", 3.0)))
        return ny_start, ny_end
    ny_start = float(us_ny_open_utc(pd.Timestamp(now_utc)))
    ny_duration = max(1.0, float(cfg.get("ny_duration_hours", 3.0)))
    ny_end = ny_start + ny_duration
    return ny_start, ny_end


def classify_session(now_utc: datetime, effective_config: Optional[dict[str, Any]] = None) -> Optional[str]:
    """Return 'tokyo', 'london', 'ny', or None using effective config if provided."""
    hour_frac = now_utc.hour + now_utc.minute / 60.0
    weekday = now_utc.weekday()

    if not isinstance(effective_config, dict):
        london_open_utc = float(uk_london_open_utc(pd.Timestamp(now_utc)))
        london_end_utc = london_open_utc + 4.0
        ny_start_utc, ny_end_utc = _resolve_ny_window_hours(now_utc, {})
        if _in_hour_window(hour_frac, float(TOKYO_START_UTC), float(TOKYO_END_UTC)) and weekday in TOKYO_ALLOWED_DAYS:
            return "tokyo"
        if _in_hour_window(hour_frac, london_open_utc, london_end_utc) and weekday in LONDON_ALLOWED_DAYS:
            return "london"
        if _in_hour_window(hour_frac, ny_start_utc, ny_end_utc) and weekday in NY_ALLOWED_DAYS:
            return "ny"
        return None

    v14_cfg = effective_config.get("v14", {}) if isinstance(effective_config.get("v14"), dict) else {}
    ldn_cfg = effective_config.get("london_v2", {}) if isinstance(effective_config.get("london_v2"), dict) else {}
    v44_cfg = effective_config.get("v44_ny", {}) if isinstance(effective_config.get("v44_ny"), dict) else {}

    tokyo_start = _parse_hhmm_to_hour(v14_cfg.get("session_start_utc", "16:00"), float(TOKYO_START_UTC))
    tokyo_end = _parse_hhmm_to_hour(v14_cfg.get("session_end_utc", "22:00"), float(TOKYO_END_UTC))
    tokyo_days = _weekday_set_from_names(v14_cfg.get("allowed_trading_days"), TOKYO_ALLOWED_DAYS)

    london_open = float(uk_london_open_utc(pd.Timestamp(now_utc)))
    a_end = int(ldn_cfg.get("a_entry_end_min_after_london", 90))
    d_end = int(ldn_cfg.get("d_entry_end_min_after_london", 120))
    london_end = london_open + (max(240, a_end, d_end) / 60.0)
    london_days = _weekday_set_from_names(ldn_cfg.get("active_days_utc"), LONDON_ALLOWED_DAYS)

    ny_start, ny_end = _resolve_ny_window_hours(now_utc, v44_cfg)
    ny_days = NY_ALLOWED_DAYS

    if _in_hour_window(hour_frac, tokyo_start, tokyo_end) and weekday in tokyo_days:
        return "tokyo"
    if _in_hour_window(hour_frac, london_open, london_end) and weekday in london_days:
        return "london"
    if _in_hour_window(hour_frac, ny_start, ny_end) and weekday in ny_days:
        return "ny"
    return None



# ===================================================================
#  BB Width Regime
# ===================================================================

def compute_bb_width_regime(
    m5_df: pd.DataFrame,
    *,
    mode: str = "percentile",
    lookback: int = BB_WIDTH_LOOKBACK,
    ranging_pct: float = BB_WIDTH_RANGING_PCT,
) -> str:
    """Return 'ranging' or 'trending' using the configured backtest regime mode."""
    close = m5_df["close"].astype(float)
    sma = close.rolling(BB_PERIOD).mean()
    std = close.rolling(BB_PERIOD, min_periods=BB_PERIOD).std(ddof=0)
    upper = sma + BB_STD * std
    lower = sma - BB_STD * std
    width = (upper - lower) / sma
    mode_norm = str(mode).strip().lower()
    if mode_norm in {"expanding3", "bb_width_expanding3"}:
        valid = width.dropna()
        if len(valid) < 4:
            return "ranging"
        expanding3 = bool(valid.iloc[-2] > valid.iloc[-3] and valid.iloc[-3] > valid.iloc[-4])
        return "trending" if expanding3 else "ranging"
    lookback_n = max(1, int(lookback))
    if len(width.dropna()) < lookback_n:
        return "ranging"
    recent = width.iloc[-lookback_n:]
    cutoff = recent.quantile(float(ranging_pct))
    current = width.iloc[-1]
    if pd.isna(current) or pd.isna(cutoff):
        return "ranging"
    return "trending" if current >= cutoff else "ranging"


def _compute_bb(m5_df: pd.DataFrame) -> tuple[float, float, float]:
    """Return (bb_upper, bb_middle, bb_lower) for latest M5 bar."""
    close = m5_df["close"].astype(float)
    sma = close.rolling(BB_PERIOD).mean()
    std = close.rolling(BB_PERIOD, min_periods=BB_PERIOD).std(ddof=0)
    bb_upper = float(sma.iloc[-1] + BB_STD * std.iloc[-1])
    bb_lower = float(sma.iloc[-1] - BB_STD * std.iloc[-1])
    bb_mid = float(sma.iloc[-1])
    return bb_upper, bb_mid, bb_lower


# ===================================================================
#  RSI
# ===================================================================

def _compute_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Compute RSI on a price series using simple rolling averages (backtest parity)."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


# ===================================================================
#  ATR
# ===================================================================

def _compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """True Range -> simple rolling ATR (backtest parity)."""
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


# ===================================================================
#  ADX
# ===================================================================

def _compute_adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> float:
    """Return latest ADX value using simple rolling smoothing (backtest parity)."""
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).rolling(window=period, min_periods=period).mean() / atr.replace(0.0, np.nan)
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).rolling(window=period, min_periods=period).mean() / atr.replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx = dx.rolling(window=period, min_periods=period).mean()
    val = adx.iloc[-1]
    return float(val) if pd.notna(val) else 0.0


# ===================================================================
#  Parabolic SAR
# ===================================================================

def compute_parabolic_sar(m1_df: pd.DataFrame) -> pd.Series:
    """Return PSAR series for M1 data."""
    high = m1_df["high"].astype(float).values
    low = m1_df["low"].astype(float).values
    n = len(high)
    psar = np.full(n, np.nan)
    af = PSAR_AF_START
    trend = 1  # 1 = up, -1 = down
    if n < 2:
        return pd.Series(psar, index=m1_df.index)
    psar[0] = low[0]
    ep = high[0]
    for i in range(1, n):
        prev_psar = psar[i - 1]
        if trend == 1:
            psar[i] = prev_psar + af * (ep - prev_psar)
            psar[i] = min(psar[i], low[i - 1])
            if i >= 2:
                psar[i] = min(psar[i], low[i - 2])
            if high[i] > ep:
                ep = high[i]
                af = min(af + PSAR_AF_STEP, PSAR_AF_MAX)
            if low[i] < psar[i]:
                trend = -1
                psar[i] = ep
                ep = low[i]
                af = PSAR_AF_START
        else:
            psar[i] = prev_psar + af * (ep - prev_psar)
            psar[i] = max(psar[i], high[i - 1])
            if i >= 2:
                psar[i] = max(psar[i], high[i - 2])
            if low[i] < ep:
                ep = low[i]
                af = min(af + PSAR_AF_STEP, PSAR_AF_MAX)
            if high[i] > psar[i]:
                trend = 1
                psar[i] = ep
                ep = high[i]
                af = PSAR_AF_START
    return pd.Series(psar, index=m1_df.index)


def detect_sar_flip(m1_df: pd.DataFrame, lookback: int = PSAR_FLIP_LOOKBACK) -> tuple[bool, bool]:
    """Return (bullish_flip, bearish_flip) within last N M1 bars."""
    psar = compute_parabolic_sar(m1_df)
    close = m1_df["close"].astype(float)
    if len(psar) < lookback + 1:
        return False, False
    bullish = False
    bearish = False
    for i in range(-lookback, 0):
        prev_idx = i - 1
        if abs(prev_idx) > len(psar):
            continue
        prev_above = psar.iloc[prev_idx] > close.iloc[prev_idx]
        curr_below = psar.iloc[i] < close.iloc[i]
        if prev_above and curr_below:
            bullish = True
        prev_below = psar.iloc[prev_idx] < close.iloc[prev_idx]
        curr_above = psar.iloc[i] > close.iloc[i]
        if prev_below and curr_above:
            bearish = True
    return bullish, bearish


# ===================================================================
#  Confluence scoring
# ===================================================================

def evaluate_v14_confluence(
    side: str,
    close: float,
    high: float,
    low: float,
    pivots: dict,
    bb_upper: float,
    bb_lower: float,
    sar_bullish_flip: bool,
    sar_bearish_flip: bool,
    rsi: float,
    pip_size: float,
    *,
    zone_tolerance_pips: float = ZONE_TOLERANCE_PIPS,
    rsi_long_entry: float = RSI_LONG_ENTRY,
    rsi_short_entry: float = RSI_SHORT_ENTRY,
    core_gate_use_zone: bool = True,
    core_gate_use_bb: bool = True,
    core_gate_use_sar: bool = True,
    core_gate_use_rsi: bool = True,
) -> tuple[int, str]:
    """
    Score confluence for V14 entry.

    Components (labelled A-D):
      A = price in pivot zone (close <= S1+tol for long, close >= R1-tol for short)
      B = BB touch/penetration (close/low <= bb_lower for long, close/high >= bb_upper for short)
      C = SAR flip (bullish for long, bearish for short)
      D = RSI soft filter (< 35 for long, > 65 for short)

    Returns (score, combo_string) e.g. (3, "ABC").
    """
    score = 0
    combo = ""
    tol = float(zone_tolerance_pips) * pip_size

    if side == "buy":
        # A: pivot zone
        if core_gate_use_zone and close <= pivots["S1"] + tol:
            score += 1
            combo += "A"
        # B: BB lower touch
        if core_gate_use_bb and (close <= bb_lower or low <= bb_lower):
            score += 1
            combo += "B"
        # C: SAR bullish flip
        if core_gate_use_sar and sar_bullish_flip:
            score += 1
            combo += "C"
        # D: RSI
        if core_gate_use_rsi and rsi < float(rsi_long_entry):
            score += 1
            combo += "D"
    else:  # sell
        if core_gate_use_zone and close >= pivots["R1"] - tol:
            score += 1
            combo += "A"
        if core_gate_use_bb and (close >= bb_upper or high >= bb_upper):
            score += 1
            combo += "B"
        if core_gate_use_sar and sar_bearish_flip:
            score += 1
            combo += "C"
        if core_gate_use_rsi and rsi > float(rsi_short_entry):
            score += 1
            combo += "D"

    return score, combo


def compute_v14_signal_strength(
    *,
    side: str,
    confluence_score: int,
    close: float,
    high: float,
    low: float,
    pivots: dict[str, Any],
    bb_upper: float,
    bb_lower: float,
    rsi: float,
    sar_bullish_flip: bool,
    sar_bearish_flip: bool,
    now_utc: datetime,
    pip_size: float,
    sst_cfg: dict[str, Any],
) -> dict[str, Any]:
    """
    Backtest-aligned signal-strength scoring hooks.
    Defaults are no-op unless enabled/configured.
    """
    components = sst_cfg.get("components", {}) if isinstance(sst_cfg.get("components"), dict) else {}
    cc_map = components.get("confluence_count", {}) if isinstance(components.get("confluence_count"), dict) else {}
    score = int(cc_map.get(str(int(confluence_score)), 0))

    bb_pen_threshold = float(components.get("bb_penetration_bonus_pips", 0.0))
    if bb_pen_threshold > 0:
        if side == "buy":
            pen = (bb_lower - low) / pip_size
        else:
            pen = (high - bb_upper) / pip_size
        if pen >= bb_pen_threshold:
            score += 1

    if bool(components.get("deep_pivot_zone", False)):
        if side == "buy" and close <= float(pivots.get("S2", np.nan)):
            score += 1
        if side == "sell" and close >= float(pivots.get("R2", np.nan)):
            score += 1

    if bool(components.get("same_candle_sar_flip", False)):
        if side == "buy" and sar_bullish_flip:
            score += 1
        if side == "sell" and sar_bearish_flip:
            score += 1

    rsi_ext = float(components.get("rsi_extreme_bonus_threshold", 0.0))
    if rsi_ext > 0:
        if side == "buy" and rsi <= rsi_ext:
            score += 1
        if side == "sell" and rsi >= (100.0 - rsi_ext):
            score += 1

    favorable = components.get("favorable_hour", [])
    if isinstance(favorable, (list, tuple, set)):
        fav_set = set()
        for h in favorable:
            try:
                fav_set.add(int(h))
            except Exception:
                continue
        if now_utc.hour in fav_set:
            score += 1

    weak_max = int(sst_cfg.get("weak_max_score", 2))
    strong_min = int(sst_cfg.get("strong_min_score", 5))
    if score >= strong_min:
        bucket = "strong"
    elif score <= weak_max:
        bucket = "weak"
    else:
        bucket = "moderate"

    size_mults = sst_cfg.get("size_multipliers", {}) if isinstance(sst_cfg.get("size_multipliers"), dict) else {}
    default_mults = {"weak": 1.0, "moderate": 1.0, "strong": 1.0}
    for k in ("weak", "moderate", "strong"):
        try:
            if k in size_mults:
                default_mults[k] = float(size_mults[k])
        except Exception:
            pass
    size_mult = float(default_mults.get(bucket, 1.0))
    return {"score": int(score), "bucket": bucket, "size_mult": size_mult}


# ===================================================================
#  SL / TP computation
# ===================================================================

def compute_v14_sl(
    side: str,
    entry_price: float,
    pivots: dict,
    pip_size: float,
    *,
    sl_buffer_pips: float = SL_BUFFER_PIPS,
    min_sl_pips: float = SL_MIN_PIPS,
    max_sl_pips: float = SL_MAX_PIPS,
) -> float:
    """Compute SL price based on pivot + buffer, clamped to min/max."""
    buffer = float(sl_buffer_pips) * pip_size
    if side == "buy":
        # SL below nearest support
        support_levels = sorted([pivots["S1"], pivots["S2"], pivots["S3"]])
        nearest = None
        for lvl in support_levels:
            if lvl < entry_price:
                nearest = lvl
                break
        if nearest is None:
            nearest = pivots["S1"]
        raw_sl = nearest - buffer
        sl_dist = (entry_price - raw_sl) / pip_size
    else:
        resist_levels = sorted([pivots["R1"], pivots["R2"], pivots["R3"]], reverse=True)
        nearest = None
        for lvl in resist_levels:
            if lvl > entry_price:
                nearest = lvl
                break
        if nearest is None:
            nearest = pivots["R1"]
        raw_sl = nearest + buffer
        sl_dist = (raw_sl - entry_price) / pip_size

    # Clamp
    sl_dist = max(float(min_sl_pips), min(float(max_sl_pips), sl_dist))
    if side == "buy":
        return entry_price - sl_dist * pip_size
    else:
        return entry_price + sl_dist * pip_size


def compute_v14_tp1(
    side: str,
    entry_price: float,
    atr_value: float,
    pip_size: float,
    *,
    partial_tp_atr_mult: float = TP1_ATR_MULT,
    partial_tp_min_pips: float = TP1_MIN_PIPS,
    partial_tp_max_pips: float = TP1_MAX_PIPS,
) -> float:
    """Compute TP1 = ATR * 0.5, clamped to 6-12 pips."""
    tp_dist = atr_value * float(partial_tp_atr_mult)
    tp_pips = tp_dist / pip_size
    tp_pips = max(float(partial_tp_min_pips), min(float(partial_tp_max_pips), tp_pips))
    if side == "buy":
        return entry_price + tp_pips * pip_size
    else:
        return entry_price - tp_pips * pip_size


# ===================================================================
#  Lot sizing (config-driven when sizing_config provided, else fallback constants)
# ===================================================================

def compute_v14_lot_size(
    equity: float,
    sl_pips: float,
    current_price: float,
    pip_size: float,
    leverage: float = LEVERAGE,
) -> int:
    """Risk-based units, margin-aware.  Returns integer units. Legacy fallback."""
    if sl_pips <= 0 or current_price <= 0:
        return 0
    risk_amount = equity * RISK_PCT
    pip_value_per_unit = pip_size / current_price
    units = risk_amount / (sl_pips * pip_value_per_unit)
    max_margin_units = (equity * leverage) / current_price
    units = min(units, max_margin_units)
    units = min(units, MAX_UNITS)
    return int(math.floor(units))


def compute_v14_units_from_config(
    equity: float,
    sl_pips: float,
    current_price: float,
    pip_size: float,
    now_utc: datetime,
    v14_config: dict[str, Any],
) -> int:
    """Config-driven V14 sizing: risk_pct, day_risk_multipliers, max_units, leverage."""
    if sl_pips <= 0 or current_price <= 0:
        return 0
    risk_pct = float(v14_config.get("risk_per_trade_pct", 2.0)) / 100.0
    max_units = int(v14_config.get("max_units", MAX_UNITS))
    leverage = float(v14_config.get("leverage", LEVERAGE))
    day_multipliers = v14_config.get("day_risk_multipliers") or {}
    day_name = now_utc.strftime("%A")
    day_mult = float(day_multipliers.get(day_name, 1.0)) if isinstance(day_multipliers.get(day_name), (int, float)) else 1.0
    risk_amount = equity * risk_pct * day_mult
    pip_value_per_unit = pip_size / current_price
    units = risk_amount / (sl_pips * pip_value_per_unit)
    # USDJPY on OANDA uses USD as the base currency, so position size is already in
    # USD-denominated units. Margin capacity is therefore units/leverage in USD,
    # not quote-price-adjusted.
    max_margin_units = equity * leverage
    units = min(units, max_margin_units)
    units = min(units, float(max_units))
    return int(math.floor(max(0.0, units)))


def _compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.astype(float).ewm(span=period, adjust=False).mean()


def _as_risk_fraction(value: Any, default_fraction: float) -> float:
    """Accept either percent-style (2.0) or fraction-style (0.02)."""
    try:
        v = float(value)
    except Exception:
        return float(default_fraction)
    if v <= 0:
        return 0.0
    return v / 100.0 if v > 1.0 else v


def _drop_incomplete_tf(df: Optional[pd.DataFrame], tf: str) -> pd.DataFrame:
    """Return only completed bars for the timeframe."""
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    if "time" in d.columns:
        d["time"] = pd.to_datetime(d["time"], utc=True, errors="coerce")
        d = d.dropna(subset=["time"]).sort_values("time")
    if d.empty:
        return d
    try:
        return drop_incomplete_last_bar(d, tf)  # type: ignore[arg-type]
    except Exception:
        return d


def _compute_session_windows(now_utc: datetime) -> dict[str, datetime]:
    d0 = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    london_open_hour = uk_london_open_utc(pd.Timestamp(now_utc))
    ny_open_hour = us_ny_open_utc(pd.Timestamp(now_utc))
    london_open_dt = d0.replace(hour=int(london_open_hour), minute=0)
    ny_open_dt = d0.replace(hour=int(ny_open_hour), minute=0)
    return {
        "day_start": d0,
        "london_open": london_open_dt,
        "london_arb_end": london_open_dt + pd.Timedelta(minutes=90),
        "lmp_impulse_end": london_open_dt + pd.Timedelta(minutes=LDN_LMP_IMPULSE_MINUTES),
        "london_end": london_open_dt + pd.Timedelta(hours=4),
        "ny_open": ny_open_dt,
        "ny_end": ny_open_dt + pd.Timedelta(hours=3),
    }


def last_sunday(year: int, month: int) -> pd.Timestamp:
    d = pd.Timestamp(year=year, month=month, day=1, tz="UTC") + pd.offsets.MonthEnd(0)
    while d.weekday() != 6:
        d -= pd.Timedelta(days=1)
    return d


def nth_sunday(year: int, month: int, n: int) -> pd.Timestamp:
    d = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    while d.weekday() != 6:
        d += pd.Timedelta(days=1)
    d += pd.Timedelta(days=(n - 1) * 7)
    return d


def uk_london_open_utc(ts_day: pd.Timestamp) -> int:
    ts_day = pd.Timestamp(ts_day)
    if ts_day.tzinfo is None:
        ts_day = ts_day.tz_localize("UTC")
    else:
        ts_day = ts_day.tz_convert("UTC")
    y = ts_day.year
    summer_start = last_sunday(y, 3).normalize()
    summer_end = last_sunday(y, 10).normalize()
    d = ts_day.normalize()
    return 7 if summer_start <= d < summer_end else 8


def us_ny_open_utc(ts_day: pd.Timestamp) -> int:
    ts_day = pd.Timestamp(ts_day)
    if ts_day.tzinfo is None:
        ts_day = ts_day.tz_localize("UTC")
    else:
        ts_day = ts_day.tz_convert("UTC")
    y = ts_day.year
    summer_start = nth_sunday(y, 3, 2).normalize()
    summer_end = nth_sunday(y, 11, 1).normalize()
    d = ts_day.normalize()
    return 12 if summer_start <= d < summer_end else 13


def impact_rank(v: object) -> int:
    s = str(v).strip().lower()
    if s in {"high", "h"}:
        return 3
    if s in {"medium", "med", "m"}:
        return 2
    if s in {"low", "l"}:
        return 1
    return 0


@lru_cache(maxsize=16)
def _load_news_events_cached_inner(calendar_path: str, impact_floor: str, mtime_ns: int) -> tuple[pd.Timestamp, ...]:
    events: list[pd.Timestamp] = []
    try:
        news_df = pd.read_csv(calendar_path)
    except Exception:
        return tuple()
    floor = impact_rank(impact_floor)
    for _, row in news_df.iterrows():
        try:
            imp = impact_rank(row.get("impact", "high"))
            if imp < floor:
                continue
            d = pd.to_datetime(row.get("date"), utc=True, errors="coerce")
            if pd.isna(d):
                continue
            ts_event = pd.Timestamp(d).normalize()
            if "time_utc" in news_df.columns and pd.notna(row.get("time_utc")):
                t_raw = str(row.get("time_utc")).strip()
                hm = re.match(r"^(\d{1,2}):(\d{2})$", t_raw)
                if hm:
                    hh = int(hm.group(1))
                    mm = int(hm.group(2))
                    ts_event = ts_event + pd.Timedelta(hours=hh, minutes=mm)
                else:
                    continue
            else:
                mins = pd.to_timedelta(row.get("minutes"), unit="m", errors="coerce")
                if pd.isna(mins):
                    continue
                ts_event = ts_event + pd.Timedelta(minutes=int(mins / pd.Timedelta(minutes=1)))
            events.append(ts_event)
        except Exception:
            continue
    events.sort()
    return tuple(events)


def _load_news_events_cached(calendar_path: str, impact_floor: str) -> tuple[pd.Timestamp, ...]:
    try:
        mtime_ns = int(os.stat(calendar_path).st_mtime_ns)
    except Exception:
        mtime_ns = -1
    return _load_news_events_cached_inner(calendar_path, impact_floor, mtime_ns)


def is_in_news_window(ts_now: datetime, events: tuple[pd.Timestamp, ...], before_min: int, after_min: int) -> bool:
    if not events:
        return False
    now = pd.Timestamp(ts_now).tz_convert("UTC") if pd.Timestamp(ts_now).tzinfo is not None else pd.Timestamp(ts_now, tz="UTC")
    w_start = now - pd.Timedelta(minutes=max(0, int(before_min)))
    w_end = now + pd.Timedelta(minutes=max(0, int(after_min)))
    for ev in events:
        if w_start <= ev <= w_end:
            return True
    return False


def compute_asian_range(
    m1_df: pd.DataFrame,
    london_open_utc_hour: int,
    range_min_pips: float = LDN_ARB_RANGE_MIN_PIPS,
    range_max_pips: float = LDN_ARB_RANGE_MAX_PIPS,
) -> tuple[float, float, float, bool]:
    """Return (high, low, range_pips, is_valid)."""
    if m1_df is None or m1_df.empty:
        return np.nan, np.nan, 0.0, False
    d = m1_df.copy()
    d["time"] = pd.to_datetime(d["time"], utc=True, errors="coerce")
    d = d.dropna(subset=["time"])
    if d.empty:
        return np.nan, np.nan, 0.0, False
    now = pd.Timestamp(d["time"].iloc[-1])
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")
    day_start = now.floor("D")
    london_open = day_start + pd.Timedelta(hours=london_open_utc_hour)
    w = d[(d["time"] >= day_start) & (d["time"] < london_open)]
    if w.empty:
        return np.nan, np.nan, 0.0, False
    high = float(w["high"].max())
    low = float(w["low"].min())
    pips = (high - low) / PIP_SIZE
    is_valid = float(range_min_pips) <= pips <= float(range_max_pips)
    return high, low, pips, bool(is_valid)


def compute_lmp_impulse(
    m1_df: pd.DataFrame,
    session_start_utc: datetime,
    impulse_minutes: int,
) -> tuple[float, float, Optional[str]]:
    """Return (impulse_high, impulse_low, direction). direction in {'up','down',None}."""
    if m1_df is None or m1_df.empty:
        return np.nan, np.nan, None
    d = m1_df.copy()
    d["time"] = pd.to_datetime(d["time"], utc=True, errors="coerce")
    d = d.dropna(subset=["time"]).sort_values("time")
    sess_start = pd.Timestamp(session_start_utc)
    if sess_start.tzinfo is None:
        sess_start = sess_start.tz_localize("UTC")
    else:
        sess_start = sess_start.tz_convert("UTC")
    end = sess_start + pd.Timedelta(minutes=impulse_minutes)
    w = d[(d["time"] >= sess_start) & (d["time"] < end)]
    if w.empty:
        return np.nan, np.nan, None
    impulse_high = float(w["high"].max())
    impulse_low = float(w["low"].min())
    impulse_pips = (impulse_high - impulse_low) / PIP_SIZE
    if impulse_pips < LDN_LMP_IMPULSE_MIN_PIPS:
        return impulse_high, impulse_low, None
    sub_open = float(w["open"].iloc[0])
    sub_close = float(w["close"].iloc[-1])
    t_h = pd.Timestamp(w.loc[w["high"].idxmax(), "time"])
    t_l = pd.Timestamp(w.loc[w["low"].idxmin(), "time"])
    bull = sub_close > sub_open and t_l < t_h
    bear = sub_close < sub_open and t_h < t_l
    direction = "up" if bull else "down" if bear else None
    return impulse_high, impulse_low, direction


def compute_lmp_zone(impulse_high: float, impulse_low: float, direction: str | None, fib_ratio: float) -> tuple[float, float]:
    if direction == "up":
        fib50 = impulse_low + (impulse_high - impulse_low) * fib_ratio
    elif direction == "down":
        fib50 = impulse_high - (impulse_high - impulse_low) * fib_ratio
    else:
        return np.nan, np.nan
    zone_top = max(fib50, fib50)
    zone_bottom = min(fib50, fib50)
    return float(zone_top), float(zone_bottom)


def _compute_risk_units(equity: float, risk_pct: float, sl_pips: float, entry_price: float, pip_size: float, max_units: int = MAX_UNITS) -> int:
    if equity <= 0 or risk_pct <= 0 or sl_pips <= 0 or entry_price <= 0:
        return 0
    pip_value_per_unit = pip_size / entry_price
    units = (equity * risk_pct) / (sl_pips * pip_value_per_unit)
    units = min(units, float(max_units))
    return int(math.floor(max(0.0, units)))


def _account_sizing_value(adapter, fallback: float = 100000.0) -> float:
    """Use account balance when available (realized-equity parity), fallback to equity."""
    try:
        acct = adapter.get_account_info()
    except Exception:
        return float(fallback)
    bal = getattr(acct, "balance", None)
    eq = getattr(acct, "equity", None)
    try:
        if bal is not None and float(bal) > 0:
            return float(bal)
    except Exception:
        pass
    try:
        if eq is not None and float(eq) > 0:
            return float(eq)
    except Exception:
        pass
    return float(fallback)


def evaluate_london_v2_arb(
    m1_df: pd.DataFrame,
    tick,
    asian_high: float,
    asian_low: float,
    pip_size: float,
    session_state: dict,
) -> tuple[Optional[str], str]:
    if m1_df is None or len(m1_df) < 2:
        return None, "london_arb: insufficient M1"
    if int(session_state.get("arb_trades", 0)) >= LDN_ARB_MAX_TRADES:
        return None, "london_arb: max trades reached"
    row = m1_df.iloc[-1]
    close = float(row["close"])
    if close > asian_high + LDN_ARB_BREAKOUT_BUFFER_PIPS * pip_size:
        return "buy", "london_arb: breakout above asian high"
    if close < asian_low - LDN_ARB_BREAKOUT_BUFFER_PIPS * pip_size:
        return "sell", "london_arb: breakout below asian low"
    return None, "london_arb: no breakout"


def evaluate_london_v2_lmp(
    m1_df: pd.DataFrame,
    m15_df: pd.DataFrame,
    tick,
    impulse_direction: str | None,
    zone_top: float,
    zone_bottom: float,
    pip_size: float,
    session_state: dict,
) -> tuple[Optional[str], str]:
    if m1_df is None or len(m1_df) < 2 or m15_df is None or len(m15_df) < LDN_LMP_EMA_M15_PERIOD + 2:
        return None, "london_lmp: insufficient data"
    if impulse_direction is None:
        return None, "london_lmp: no valid impulse"
    if int(session_state.get("lmp_trades", 0)) >= LDN_LMP_MAX_TRADES:
        return None, "london_lmp: max trades reached"

    m15_close = m15_df["close"].astype(float)
    ema20 = float(_compute_ema(m15_close, LDN_LMP_EMA_M15_PERIOD).iloc[-1])

    row = m1_df.iloc[-1]
    hi = float(row["high"])
    lo = float(row["low"])
    close = float(row["close"])
    if impulse_direction == "up":
        touched = lo <= zone_top + 0.1 * pip_size
        if touched and close > zone_top and close > ema20:
            return "buy", "london_lmp: bullish pullback confirmation"
    elif impulse_direction == "down":
        touched = hi >= zone_bottom - 0.1 * pip_size
        if touched and close < zone_bottom and close < ema20:
            return "sell", "london_lmp: bearish pullback confirmation"
    return None, "london_lmp: no pullback signal"


def compute_v44_h1_trend(h1_df: pd.DataFrame, ema_fast_period: int = V44_H1_EMA_FAST, ema_slow_period: int = V44_H1_EMA_SLOW) -> str | None:
    if h1_df is None or len(h1_df) < max(ema_fast_period, ema_slow_period) + 2:
        return None
    close = h1_df["close"].astype(float)
    ema_fast = _compute_ema(close, ema_fast_period)
    ema_slow = _compute_ema(close, ema_slow_period)
    if float(ema_fast.iloc[-1]) > float(ema_slow.iloc[-1]):
        return "up"
    if float(ema_fast.iloc[-1]) < float(ema_slow.iloc[-1]):
        return "down"
    return None


def compute_v44_m5_slope(
    m5_df: pd.DataFrame,
    slope_bars: int,
    ema_fast_period: int = V44_M5_EMA_FAST,
    ema_slow_period: int = V44_M5_EMA_SLOW,
) -> float:
    if m5_df is None or len(m5_df) < max(ema_fast_period, ema_slow_period) + slope_bars + 2:
        return 0.0
    ema_fast = _compute_ema(m5_df["close"].astype(float), ema_fast_period)
    now = float(ema_fast.iloc[-1])
    prev = float(ema_fast.iloc[-1 - slope_bars])
    return (now - prev) / PIP_SIZE / max(1, slope_bars)


def classify_v44_strength(slope: float, is_london: bool) -> str:
    threshold = V44_LONDON_STRONG_SLOPE if is_london else V44_STRONG_SLOPE
    return "strong" if abs(slope) >= threshold else "normal"


def compute_v44_atr_pct_filter(
    m5_df: pd.DataFrame,
    *,
    enabled: bool = True,
    cap: float = V44_ATR_PCT_CAP,
    lookback: int = V44_ATR_PCT_LOOKBACK,
) -> bool:
    if not enabled:
        return True
    if m5_df is None or len(m5_df) < int(lookback):
        return True
    atr = _compute_atr(m5_df, ATR_PERIOD).dropna()
    if len(atr) < 20:
        return True
    current = float(atr.iloc[-1])
    look = atr.iloc[-int(lookback):]
    cutoff = float(np.quantile(look, float(cap)))
    return current <= cutoff


def compute_v44_sl(side: str, m5_df: pd.DataFrame, entry_price: float, pip_size: float) -> float:
    if m5_df is None or len(m5_df) < V44_SL_LOOKBACK + 1:
        raw_pips = V44_SL_FLOOR_PIPS
    else:
        w = m5_df.tail(V44_SL_LOOKBACK)
        if side == "buy":
            raw_sl = float(w["low"].min()) - V44_SL_BUFFER_PIPS * pip_size
            raw_pips = (entry_price - raw_sl) / pip_size
        else:
            raw_sl = float(w["high"].max()) + V44_SL_BUFFER_PIPS * pip_size
            raw_pips = (raw_sl - entry_price) / pip_size
    sl_pips = max(V44_SL_FLOOR_PIPS, min(V44_SL_CAP_PIPS, raw_pips))
    return entry_price - sl_pips * pip_size if side == "buy" else entry_price + sl_pips * pip_size


def evaluate_v44_entry(
    h1_df: pd.DataFrame,
    m5_df: pd.DataFrame,
    tick,
    pip_size: float,
    session: str,
    session_state: dict,
    *,
    now_utc: Optional[datetime] = None,
    max_entries_per_day: int = V44_MAX_ENTRIES_DAY,
    session_stop_losses: int = V44_SESSION_STOP_LOSSES,
    h1_ema_fast_period: int = V44_H1_EMA_FAST,
    h1_ema_slow_period: int = V44_H1_EMA_SLOW,
    m5_ema_fast_period: int = V44_M5_EMA_FAST,
    m5_ema_slow_period: int = V44_M5_EMA_SLOW,
    slope_bars: int = V44_SLOPE_BARS,
    strong_slope_threshold: float = V44_STRONG_SLOPE,
    weak_slope_threshold: float = V44_WEAK_SLOPE,
    min_body_pips: float = V44_MIN_BODY_PIPS,
    atr_pct_filter_enabled: bool = True,
    atr_pct_cap: float = V44_ATR_PCT_CAP,
    atr_pct_lookback: int = V44_ATR_PCT_LOOKBACK,
) -> tuple[Optional[str], str, str]:
    if int(session_state.get("trade_count", 0)) >= int(max_entries_per_day):
        return None, "normal", "v44: max entries/day reached"
    if int(session_state.get("consecutive_losses", 0)) >= int(session_stop_losses):
        return None, "normal", "v44: consecutive loss stop"

    cooldown_until = session_state.get("cooldown_until")
    if cooldown_until:
        try:
            ts_now = now_utc or datetime.now(timezone.utc)
            if ts_now < datetime.fromisoformat(str(cooldown_until)):
                return None, "normal", "v44: cooldown active"
        except Exception:
            pass

    trend = compute_v44_h1_trend(h1_df, h1_ema_fast_period, h1_ema_slow_period)
    if trend is None:
        return None, "normal", "v44: no H1 trend"
    if not compute_v44_atr_pct_filter(
        m5_df,
        enabled=atr_pct_filter_enabled,
        cap=atr_pct_cap,
        lookback=atr_pct_lookback,
    ):
        return None, "normal", "v44: ATR percentile block"
    if m5_df is None or len(m5_df) < max(m5_ema_fast_period, m5_ema_slow_period) + 4:
        return None, "normal", "v44: insufficient M5"

    close = m5_df["close"].astype(float)
    open_ = m5_df["open"].astype(float)
    ema_fast = _compute_ema(close, m5_ema_fast_period)
    ema_slow = _compute_ema(close, m5_ema_slow_period)
    body_pips = abs(float(close.iloc[-1]) - float(open_.iloc[-1])) / pip_size
    slope = compute_v44_m5_slope(m5_df, slope_bars, m5_ema_fast_period, m5_ema_slow_period)
    abs_slope = abs(slope)
    if abs_slope > float(strong_slope_threshold):
        strength = "strong"
    elif abs_slope > float(weak_slope_threshold):
        strength = "normal"
    else:
        strength = "weak"
    bullish_bar = float(close.iloc[-1]) > float(open_.iloc[-1]) and body_pips >= min_body_pips
    bearish_bar = float(close.iloc[-1]) < float(open_.iloc[-1]) and body_pips >= min_body_pips

    if trend == "up" and float(ema_fast.iloc[-1]) > float(ema_slow.iloc[-1]) and bullish_bar and slope > 0:
        return "buy", strength, "v44: H1 up + M5 strong bullish momentum"
    if trend == "down" and float(ema_fast.iloc[-1]) < float(ema_slow.iloc[-1]) and bearish_bar and slope < 0:
        return "sell", strength, "v44: H1 down + M5 strong bearish momentum"
    return None, strength, "v44: directional conditions not met"


def _compute_v44_atr_rank(
    m5_df: pd.DataFrame,
    lookback: int,
) -> Optional[float]:
    if m5_df is None or len(m5_df) < max(20, int(lookback)):
        return None
    tr = pd.concat(
        [
            m5_df["high"].astype(float) - m5_df["low"].astype(float),
            (m5_df["high"].astype(float) - m5_df["close"].astype(float).shift(1)).abs(),
            (m5_df["low"].astype(float) - m5_df["close"].astype(float).shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(14).mean()
    if len(atr.dropna()) < max(20, int(lookback)):
        return None
    now = float(atr.iloc[-1])
    window = atr.iloc[-int(lookback):].to_numpy(dtype=float)
    if len(window) == 0:
        return None
    return float(np.count_nonzero(window <= now)) / float(len(window))


def _determine_v44_session_mode(
    sdat: dict[str, Any],
    *,
    dual_mode_enabled: bool,
    trend_mode_efficiency_min: float,
    range_mode_efficiency_max: float,
) -> str:
    if not dual_mode_enabled:
        return "trend"
    hist = sdat.get("session_efficiency_history", [])
    if not isinstance(hist, list) or len(hist) < 3:
        return "trend"
    vals = [float(x) for x in hist if isinstance(x, (int, float))]
    if len(vals) < 3:
        return "trend"
    avg_eff = float(sum(vals) / len(vals))
    if avg_eff >= float(trend_mode_efficiency_min):
        return "trend"
    if avg_eff <= float(range_mode_efficiency_max):
        return "range_fade"
    return "neutral"


def _v44_news_trend_active_event(
    now_utc: datetime,
    events: tuple[pd.Timestamp, ...],
    delay_minutes: int,
    window_minutes: int,
) -> Optional[pd.Timestamp]:
    if not events:
        return None
    now_ts = pd.Timestamp(now_utc)
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize("UTC")
    else:
        now_ts = now_ts.tz_convert("UTC")
    for ev in events:
        ev_ts = pd.Timestamp(ev)
        if ev_ts.tzinfo is None:
            ev_ts = ev_ts.tz_localize("UTC")
        else:
            ev_ts = ev_ts.tz_convert("UTC")
        start = ev_ts + pd.Timedelta(minutes=max(0, int(delay_minutes)))
        end = start + pd.Timedelta(minutes=max(1, int(window_minutes)))
        if start <= now_ts <= end:
            return ev_ts
    return None


def execute_london_v2_entry(
    *,
    adapter,
    profile,
    policy,
    data_by_tf: dict,
    tick,
    phase3_state: dict,
    sizing_config: Optional[dict[str, Any]] = None,
    now_utc: Optional[datetime] = None,
    store=None,
) -> dict:
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    pip = float(profile.pip_size) if hasattr(profile, "pip_size") else PIP_SIZE
    no_trade = {
        "decision": ExecutionDecision(attempted=False, placed=False, reason="", side=None),
        "phase3_state_updates": {},
        "strategy_tag": None,
    }

    ldn_config = (sizing_config or {}).get("london_v2", {})
    ldn_max_open = int(ldn_config.get("max_open_positions", LDN_MAX_OPEN))
    ldn_default_risk = _as_risk_fraction(ldn_config.get("risk_per_trade_pct", LDN_RISK_PCT), LDN_RISK_PCT)
    ldn_arb_risk_pct = _as_risk_fraction(ldn_config.get("arb_risk_per_trade_pct", ldn_default_risk), ldn_default_risk)
    ldn_d_risk_pct = _as_risk_fraction(ldn_config.get("d_risk_per_trade_pct", ldn_config.get("lmp_risk_per_trade_pct", ldn_default_risk)), ldn_default_risk)
    ldn_max_total_risk_pct = _as_risk_fraction(ldn_config.get("max_total_open_risk_pct", 0.05), 0.05)
    _ldn_cap_raw = ldn_config.get("max_trades_per_day_total", LDN_MAX_TRADES_PER_DAY_TOTAL)
    try:
        ldn_max_trades_per_day_total = int(_ldn_cap_raw) if _ldn_cap_raw is not None else 0
    except Exception:
        ldn_max_trades_per_day_total = 0
    ldn_range_min_pips = float(ldn_config.get("arb_range_min_pips", LDN_ARB_RANGE_MIN_PIPS))
    ldn_range_max_pips = float(ldn_config.get("arb_range_max_pips", LDN_ARB_RANGE_MAX_PIPS))
    ldn_lor_min_pips = float(ldn_config.get("lor_range_min_pips", LDN_D_LOR_MIN_PIPS))
    ldn_lor_max_pips = float(ldn_config.get("lor_range_max_pips", LDN_D_LOR_MAX_PIPS))
    ldn_d_max_trades = int(ldn_config.get("d_max_trades", LDN_D_MAX_TRADES))
    ldn_d_allow_long = bool(ldn_config.get("d_allow_long", True))
    ldn_d_allow_short = bool(ldn_config.get("d_allow_short", False))
    ldn_a_allow_long = bool(ldn_config.get("a_allow_long", True))
    ldn_a_allow_short = bool(ldn_config.get("a_allow_short", True))
    ldn_leverage = float(ldn_config.get("leverage", 33.0))
    ldn_max_margin_frac = float(ldn_config.get("max_margin_usage_fraction_per_trade", 0.5))
    ldn_active_days = ldn_config.get("active_days_utc", ["Tuesday", "Wednesday"])
    ldn_active_days = set(str(d) for d in ldn_active_days) if isinstance(ldn_active_days, (list, tuple, set)) else {"Tuesday", "Wednesday"}
    ldn_disable_channel_reset = bool(ldn_config.get("disable_channel_reset_after_exit", False))

    ldn_max_entry_spread = float(ldn_config.get("max_entry_spread_pips", LDN_MAX_SPREAD_PIPS))
    if (tick.ask - tick.bid) / pip > ldn_max_entry_spread:
        no_trade["decision"] = ExecutionDecision(
            attempted=False,
            placed=False,
            reason=f"london_v2: spread veto ({(tick.ask - tick.bid) / pip:.1f}p > {ldn_max_entry_spread:.1f}p)",
            side=None,
        )
        return no_trade

    m1_df = _drop_incomplete_tf(data_by_tf.get("M1"), "M1")
    if m1_df is None or m1_df.empty:
        no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason="london_v2: missing M1", side=None)
        return no_trade

    windows = _compute_session_windows(now_utc)
    if now_utc.strftime("%A") not in ldn_active_days:
        no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason="london_v2: day not active", side=None)
        return no_trade
    today = now_utc.date().isoformat()
    session_key = f"session_london_{today}"
    sdat = dict(phase3_state.get(session_key, {}))
    sdat.setdefault("arb_trades", 0)
    sdat.setdefault("d_trades", 0)
    sdat.setdefault("total_trades", 0)
    sdat.setdefault("consecutive_losses", 0)
    sdat.setdefault("last_entry_time", None)
    channels = dict(sdat.get("channels", {}))
    for _ck in ("A_long", "A_short", "D_long", "D_short"):
        channels.setdefault(_ck, "ARMED")

    if ldn_max_trades_per_day_total > 0 and int(sdat.get("total_trades", 0)) >= ldn_max_trades_per_day_total:
        no_trade["decision"] = ExecutionDecision(
            attempted=False,
            placed=False,
            reason=f"london_v2: daily cap {sdat.get('total_trades', 0)}/{ldn_max_trades_per_day_total}",
            side=None,
        )
        state_updates = {}
        no_trade["phase3_state_updates"] = state_updates
        return no_trade

    high, low, range_pips, range_ok = compute_asian_range(
        m1_df,
        int(windows["london_open"].hour),
        range_min_pips=ldn_range_min_pips,
        range_max_pips=ldn_range_max_pips,
    )
    state_updates = {
        "london_asian_range": {"date": today, "high": high, "low": low, "pips": range_pips, "is_valid": range_ok},
    }
    open_count = int(phase3_state.get("open_trade_count", 0))
    if open_count >= ldn_max_open:
        no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason=f"london_v2: max open {open_count}/{ldn_max_open}", side=None)
        no_trade["phase3_state_updates"] = state_updates
        return no_trade

    strategy_tag = None
    side = None
    reason = ""
    entry_price = 0.0
    sl_price = None
    tp1_price = None
    lor_high = np.nan
    lor_low = np.nan

    # Setup A window can overlap Setup D window. Evaluate A first, then D if A did not signal.
    a_start = windows["london_open"] + pd.Timedelta(minutes=int(ldn_config.get("a_entry_start_min_after_london", 0)))
    a_end = windows["london_open"] + pd.Timedelta(minutes=int(ldn_config.get("a_entry_end_min_after_london", 90)))
    in_arb_window = a_start <= now_utc < a_end
    d_start = windows["london_open"] + pd.Timedelta(minutes=int(ldn_config.get("d_entry_start_min_after_london", 15)))
    d_end = windows["london_open"] + pd.Timedelta(minutes=int(ldn_config.get("d_entry_end_min_after_london", 120)))
    in_d_window = d_start <= now_utc < d_end

    # Channel reset logic (backtest parity): WAITING_RESET -> ARMED when level is retouched.
    m1_last_high = float(m1_df["high"].astype(float).iloc[-1])
    m1_last_low = float(m1_df["low"].astype(float).iloc[-1])
    if not ldn_disable_channel_reset:
        if str(channels.get("A_long", "ARMED")) == "WAITING_RESET" and m1_last_low <= float(high):
            channels["A_long"] = "ARMED"
        if str(channels.get("A_short", "ARMED")) == "WAITING_RESET" and m1_last_high >= float(low):
            channels["A_short"] = "ARMED"

    if in_arb_window:
        if range_ok:
            side, reason = evaluate_london_v2_arb(m1_df, tick, high, low, pip, sdat)
            if side:
                ch_key = "A_long" if side == "buy" else "A_short"
                if str(channels.get(ch_key, "ARMED")) != "ARMED":
                    side = None
                    reason = f"london_arb: {ch_key} waiting reset"
            if side:
                if side == "buy" and not ldn_a_allow_long:
                    side = None
                    reason = "london_arb: long disabled"
                elif side == "sell" and not ldn_a_allow_short:
                    side = None
                    reason = "london_arb: short disabled"
            if side:
                entry_price = tick.ask if side == "buy" else tick.bid
                raw_sl = low - LDN_ARB_SL_BUFFER_PIPS * pip if side == "buy" else high + LDN_ARB_SL_BUFFER_PIPS * pip
                sl_pips = abs(entry_price - raw_sl) / pip
                sl_pips = max(LDN_ARB_SL_MIN_PIPS, min(LDN_ARB_SL_MAX_PIPS, sl_pips))
                sl_price = entry_price - sl_pips * pip if side == "buy" else entry_price + sl_pips * pip
                tp1_price = entry_price + (LDN_ARB_TP1_R * sl_pips * pip if side == "buy" else -LDN_ARB_TP1_R * sl_pips * pip)
                strategy_tag = "phase3:london_v2_arb"
        else:
            reason = f"london_v2: asian range invalid ({range_pips:.1f}p)"

    if side is None and in_d_window:
        # Setup D (LOR breakout, long-only), independent from Asian-range validity.
        lor_form_end = windows["london_open"] + pd.Timedelta(minutes=15)
        lor_w = m1_df[(m1_df["time"] >= windows["london_open"]) & (m1_df["time"] < lor_form_end)]
        if not lor_w.empty:
            lor_high = float(lor_w["high"].max())
            lor_low = float(lor_w["low"].min())
            if not ldn_disable_channel_reset:
                if str(channels.get("D_long", "ARMED")) == "WAITING_RESET" and m1_last_low <= float(lor_high):
                    channels["D_long"] = "ARMED"
                if str(channels.get("D_short", "ARMED")) == "WAITING_RESET" and m1_last_high >= float(lor_low):
                    channels["D_short"] = "ARMED"
            lor_pips = (lor_high - lor_low) / pip
            lor_valid = ldn_lor_min_pips <= lor_pips <= ldn_lor_max_pips
            state_updates["london_lor"] = {"date": today, "high": lor_high, "low": lor_low, "pips": lor_pips, "is_valid": lor_valid}
            if lor_valid and int(sdat.get("d_trades", 0)) < ldn_d_max_trades:
                close_px = float(m1_df.iloc[-1]["close"])
                if ldn_d_allow_long and close_px > lor_high + LDN_D_BREAKOUT_BUFFER_PIPS * pip:
                    if str(channels.get("D_long", "ARMED")) == "ARMED":
                        side = "buy"
                        reason = "london_v2_d: LOR long breakout"
                        entry_price = tick.ask
                        raw_sl = lor_low - LDN_D_SL_BUFFER_PIPS * pip
                        sl_pips = abs(entry_price - raw_sl) / pip
                        sl_pips = max(LDN_D_SL_MIN_PIPS, min(LDN_D_SL_MAX_PIPS, sl_pips))
                        sl_price = entry_price - sl_pips * pip
                        tp1_price = entry_price + (LDN_D_TP1_R * sl_pips * pip)
                        strategy_tag = "phase3:london_v2_d"
                    else:
                        reason = "london_v2_d: D_long waiting reset"
                elif ldn_d_allow_short and close_px < lor_low - LDN_D_BREAKOUT_BUFFER_PIPS * pip:
                    if str(channels.get("D_short", "ARMED")) == "ARMED":
                        side = "sell"
                        reason = "london_v2_d: LOR short breakout"
                        entry_price = tick.bid
                        raw_sl = lor_high + LDN_D_SL_BUFFER_PIPS * pip
                        sl_pips = abs(raw_sl - entry_price) / pip
                        sl_pips = max(LDN_D_SL_MIN_PIPS, min(LDN_D_SL_MAX_PIPS, sl_pips))
                        sl_price = entry_price + sl_pips * pip
                        tp1_price = entry_price - (LDN_D_TP1_R * sl_pips * pip)
                        strategy_tag = "phase3:london_v2_d"
                    else:
                        reason = "london_v2_d: D_short waiting reset"
                else:
                    reason = "london_v2_d: no breakout"
            elif int(sdat.get("d_trades", 0)) >= ldn_d_max_trades:
                reason = f"london_v2_d: max trades reached ({sdat.get('d_trades', 0)}/{ldn_d_max_trades})"
            else:
                reason = f"london_v2_d: invalid LOR range ({lor_pips:.1f}p)"
        elif not reason:
            reason = "london_v2_d: missing LOR window data"

    if side is None or sl_price is None or tp1_price is None:
        no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason=reason or "london_v2: no setup", side=None)
        no_trade["phase3_state_updates"] = state_updates
        return no_trade

    equity = _account_sizing_value(adapter, fallback=100000.0)
    acct_equity = float(equity)
    margin_used = 0.0
    try:
        acct = adapter.get_account_info()
        _eq = getattr(acct, "equity", None)
        if _eq is not None:
            acct_equity = float(_eq)
        _mu = getattr(acct, "margin_used", None)
        if _mu is not None:
            margin_used = float(_mu)
    except Exception:
        pass

    current_setup_risk_pct = ldn_arb_risk_pct if (strategy_tag or "").endswith("_arb") else ldn_d_risk_pct
    new_risk_usd = equity * current_setup_risk_pct
    # Planned-risk ledger (backtest parity): sum actual initial SL risk from open London trades.
    open_risk_usd = 0.0
    if store is not None:
        try:
            profile_name = getattr(profile, "profile_name", str(profile))
            for ot in store.list_open_trades(profile_name):
                if not str(ot["entry_type"] or "").startswith("phase3:london_v2"):
                    continue
                rup = ot["risk_usd_planned"] if "risk_usd_planned" in ot.keys() else None
                if rup is not None:
                    try:
                        open_risk_usd += float(rup)
                        continue
                    except Exception:
                        pass
                ep = ot["entry_price"]
                sp = ot["stop_price"]
                lots = ot["size_lots"]
                if ep and sp and lots:
                    sl_pips = abs(float(ep) - float(sp)) / pip
                    open_risk_usd += sl_pips * (pip / float(ep)) * float(lots) * 100000.0
        except Exception:
            open_risk_usd = open_count * equity * current_setup_risk_pct  # fallback
    else:
        open_risk_usd = open_count * equity * current_setup_risk_pct  # fallback (no store)
    if open_risk_usd + new_risk_usd > equity * ldn_max_total_risk_pct:
        no_trade["decision"] = ExecutionDecision(
            attempted=False, placed=False,
            reason=(
                f"london_v2: open risk cap "
                f"(open={open_risk_usd:.0f}+new={new_risk_usd:.0f} > cap={equity * ldn_max_total_risk_pct:.0f} USD)"
            ),
            side=None,
        )
        no_trade["phase3_state_updates"] = state_updates
        return no_trade

    sl_pips = abs(entry_price - sl_price) / pip
    units = _compute_risk_units(equity, current_setup_risk_pct, sl_pips, entry_price, pip, max_units=MAX_UNITS)
    if units <= 0:
        no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason="london_v2: size=0", side=None)
        no_trade["phase3_state_updates"] = state_updates
        return no_trade
    risk_usd_planned = sl_pips * (pip / entry_price) * float(units)

    # USDJPY on OANDA: position units are base USD units, so required margin is close to units/leverage in USD.
    req_margin = abs(float(units)) / max(1.0, ldn_leverage)
    try:
        free_margin = max(0.0, acct_equity - float(margin_used))
    except Exception:
        free_margin = acct_equity * (1.0 - ldn_max_margin_frac)
    if req_margin > ldn_max_margin_frac * free_margin:
        no_trade["decision"] = ExecutionDecision(
            attempted=False, placed=False, reason="london_v2: margin constraint", side=None,
        )
        no_trade["phase3_state_updates"] = state_updates
        return no_trade

    comment = f"phase3_integrated:{policy.id}:{strategy_tag.replace('phase3:','')}"
    try:
        dec = adapter.place_order(
            symbol=profile.symbol,
            side=side,
            lots=units / 100000.0,
            stop_price=round(float(sl_price), 3),
            target_price=round(float(tp1_price), 3),
            comment=comment,
        )
    except Exception as e:
        return {
            "decision": ExecutionDecision(attempted=True, placed=False, reason=f"london_v2 order error: {e}", side=side),
            "phase3_state_updates": state_updates,
            "strategy_tag": strategy_tag,
        }

    confirmed_fill, confirmed_deal_id = _phase3_order_confirmed(adapter, profile, dec)
    if not confirmed_fill:
        return {
            "decision": ExecutionDecision(
                attempted=True,
                placed=False,
                reason="london_v2: broker order pending/unfilled",
                side=side,
                order_retcode=getattr(dec, "order_retcode", None),
                order_id=getattr(dec, "order_id", None),
                deal_id=getattr(dec, "deal_id", None),
                fill_price=getattr(dec, "fill_price", None),
            ),
            "phase3_state_updates": state_updates,
            "strategy_tag": strategy_tag,
        }

    if strategy_tag.endswith("_arb"):
        sdat["arb_trades"] = int(sdat.get("arb_trades", 0)) + 1
        channels["A_long" if side == "buy" else "A_short"] = "FIRED"
    else:
        sdat["d_trades"] = int(sdat.get("d_trades", 0)) + 1
        channels["D_long" if side == "buy" else "D_short"] = "FIRED"
    sdat["total_trades"] = int(sdat.get("total_trades", 0)) + 1
    sdat["last_entry_time"] = now_utc.isoformat()
    sdat["channels"] = channels
    state_updates[session_key] = sdat

    return {
        "decision": ExecutionDecision(
            attempted=True,
            placed=True,
            reason=f"{reason} | SL={sl_pips:.1f}p units={units}",
            side=side,
            order_retcode=getattr(dec, "order_retcode", None),
            order_id=getattr(dec, "order_id", None),
            deal_id=confirmed_deal_id if confirmed_deal_id is not None else getattr(dec, "deal_id", None),
            fill_price=getattr(dec, "fill_price", None),
        ),
        "phase3_state_updates": state_updates,
        "strategy_tag": strategy_tag,
        "sl_price": float(sl_price),
        "tp1_price": float(tp1_price),
        "units": int(units),
        "entry_price": float(entry_price),
        "sl_pips": float(sl_pips),
        "risk_usd_planned": float(risk_usd_planned),
    }


def execute_v44_ny_entry(
    *,
    adapter,
    profile,
    policy,
    data_by_tf: dict,
    tick,
    phase3_state: dict,
    sizing_config: Optional[dict[str, Any]] = None,
    now_utc: Optional[datetime] = None,
    store=None,
) -> dict:
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    pip = float(profile.pip_size) if hasattr(profile, "pip_size") else PIP_SIZE
    no_trade = {
        "decision": ExecutionDecision(attempted=False, placed=False, reason="", side=None),
        "phase3_state_updates": {},
        "strategy_tag": None,
    }

    v44_config = (sizing_config or {}).get("v44_ny", {})
    v44_start_delay_min = int(v44_config.get("start_delay_minutes", 5))
    v44_max_entry_spread = float(v44_config.get("max_entry_spread_pips", V44_MAX_ENTRY_SPREAD))
    v44_ny_start_hour, v44_ny_end_hour = _resolve_ny_window_hours(now_utc, v44_config)

    _ts_now = pd.Timestamp(now_utc)
    if _ts_now.tzinfo is None:
        _ts_now = _ts_now.tz_localize("UTC")
    else:
        _ts_now = _ts_now.tz_convert("UTC")
    _day0 = _ts_now.normalize()
    ny_start_ts = _day0 + pd.Timedelta(hours=v44_ny_start_hour) + pd.Timedelta(minutes=max(0, v44_start_delay_min))
    ny_end_ts = _day0 + pd.Timedelta(hours=v44_ny_end_hour)

    if _ts_now < ny_start_ts:
        no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason="v44_ny: start delay active", side=None)
        return no_trade
    if (tick.ask - tick.bid) / pip > v44_max_entry_spread:
        no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason="v44_ny: spread veto", side=None)
        return no_trade

    m1_df = _drop_incomplete_tf(data_by_tf.get("M1"), "M1")
    h1_df = _drop_incomplete_tf(data_by_tf.get("H1"), "H1")
    m5_df = _drop_incomplete_tf(data_by_tf.get("M5"), "M5")
    h4_df = _drop_incomplete_tf(data_by_tf.get("H4"), "H4")
    if h1_df is None or h1_df.empty or m5_df is None or m5_df.empty or m1_df is None or m1_df.empty:
        no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason="v44_ny: missing M1/H1/M5", side=None)
        return no_trade

    v44_max_open = int(v44_config.get("max_open_positions", V44_MAX_OPEN))
    v44_risk_pct = _as_risk_fraction(v44_config.get("risk_per_trade_pct", 0.5), 0.005)
    v44_rp_min_lot = float(v44_config.get("rp_min_lot", 1.0))
    v44_rp_max_lot = float(v44_config.get("rp_max_lot", 20.0))
    v44_max_entries_day = int(v44_config.get("max_entries_per_day", V44_MAX_ENTRIES_DAY))
    v44_session_stop_losses = int(v44_config.get("session_stop_losses", V44_SESSION_STOP_LOSSES))
    v44_ny_strength_allow = str(v44_config.get("ny_strength_allow", v44_config.get("strength_allow", "strong_normal"))).lower()
    v44_strong_tp1_pips = float(v44_config.get("strong_tp1_pips", V44_STRONG_TP1_PIPS))
    v44_normal_tp1_pips = float(v44_config.get("normal_tp1_pips", 1.75))
    v44_weak_tp1_pips = float(v44_config.get("weak_tp1_pips", 1.2))
    v44_h1_ema_fast = int(v44_config.get("h1_ema_fast", V44_H1_EMA_FAST))
    v44_h1_ema_slow = int(v44_config.get("h1_ema_slow", V44_H1_EMA_SLOW))
    v44_m5_ema_fast = int(v44_config.get("m5_ema_fast", V44_M5_EMA_FAST))
    v44_m5_ema_slow = int(v44_config.get("m5_ema_slow", V44_M5_EMA_SLOW))
    v44_slope_bars = int(v44_config.get("slope_bars", V44_SLOPE_BARS))
    v44_strong_slope = float(v44_config.get("strong_slope", V44_STRONG_SLOPE))
    v44_weak_slope = float(v44_config.get("weak_slope", V44_WEAK_SLOPE))
    v44_min_body_pips = float(v44_config.get("entry_min_body_pips", V44_MIN_BODY_PIPS))
    v44_atr_pct_filter_enabled = bool(v44_config.get("atr_pct_filter_enabled", True))
    v44_atr_pct_cap = float(v44_config.get("atr_pct_cap", V44_ATR_PCT_CAP))
    v44_atr_pct_lookback = int(v44_config.get("atr_pct_lookback", V44_ATR_PCT_LOOKBACK))
    v44_entry_cutoff_minutes = int(v44_config.get("session_entry_cutoff_minutes", 60))
    v44_news_filter_enabled = bool(v44_config.get("news_filter_enabled", False))
    v44_news_before = int(v44_config.get("news_window_minutes_before", 60))
    v44_news_after = int(v44_config.get("news_window_minutes_after", 30))
    v44_news_impact_min = str(v44_config.get("news_impact_min", "high"))
    v44_news_calendar_path = str(v44_config.get("news_calendar_path", "research_out/v5_scheduled_events_utc.csv"))
    v44_skip_days_raw = str(v44_config.get("skip_days", "")).strip()
    v44_skip_months_raw = str(v44_config.get("skip_months", "")).strip()
    v44_skip_weak = bool(v44_config.get("skip_weak", True))
    v44_skip_normal = bool(v44_config.get("skip_normal", False))
    v44_allow_normal_plus = bool(v44_config.get("allow_normal_plus", False))
    v44_normalplus_atr_min_pips = float(v44_config.get("normalplus_atr_min_pips", 6.0))
    v44_normalplus_slope_min = float(v44_config.get("normalplus_slope_min", 0.45))
    v44_session_range_cap_pips = float(v44_config.get("session_range_cap_pips", 0.0))
    v44_h1_slope_min = float(v44_config.get("h1_slope_min", 0.0))
    v44_h1_slope_consistent_bars = int(v44_config.get("h1_slope_consistent_bars", 0))
    v44_dual_mode_enabled = bool(v44_config.get("dual_mode_enabled", False))
    v44_trend_mode_efficiency_min = float(v44_config.get("trend_mode_efficiency_min", 0.4))
    v44_range_mode_efficiency_max = float(v44_config.get("range_mode_efficiency_max", 0.3))
    v44_range_fade_enabled = bool(v44_config.get("range_fade_enabled", False))
    v44_queued_confirm_bars = max(0, int(v44_config.get("queued_confirm_bars", 0)))
    v44_gl_enabled = bool(v44_config.get("gl_enabled", False))
    v44_gl_min_wins = max(1, int(v44_config.get("gl_min_wins", 1)))
    v44_gl_extra_entries = max(0, int(v44_config.get("gl_extra_entries", 0)))
    v44_gl_allow_normal = bool(v44_config.get("gl_allow_normal", True))
    v44_gl_normal_atr_cap = float(v44_config.get("gl_normal_atr_cap", 0.67))
    v44_gl_slope_relax = float(v44_config.get("gl_slope_relax", 0.0))
    v44_gl_max_open = int(v44_config.get("gl_max_open", 0))
    v44_news_trend_enabled = bool(v44_config.get("news_trend_enabled", False))
    v44_news_trend_delay_minutes = int(v44_config.get("news_trend_delay_minutes", 45))
    v44_news_trend_window_minutes = int(v44_config.get("news_trend_window_minutes", 90))
    v44_news_trend_confirm_bars = max(1, int(v44_config.get("news_trend_confirm_bars", 3)))
    v44_news_trend_min_body_pips = float(v44_config.get("news_trend_min_body_pips", 1.5))
    v44_news_trend_require_ema_align = bool(v44_config.get("news_trend_require_ema_align", True))
    v44_news_trend_risk_pct = _as_risk_fraction(v44_config.get("news_trend_risk_pct", 0.5), 0.005)
    v44_news_trend_tp_rr = float(v44_config.get("news_trend_tp_rr", 1.5))
    v44_news_trend_sl_pips = float(v44_config.get("news_trend_sl_pips", 8.0))
    v44_h4_adx_min = float(v44_config.get("h4_adx_min", 0.0))
    v44_exhaustion_gate_enabled = bool(v44_config.get("exhaustion_gate_enabled", False))
    v44_exhaustion_window_minutes = max(5, int(v44_config.get("exhaustion_gate_window_minutes", 60)))
    v44_exhaustion_max_range_pips = float(v44_config.get("exhaustion_gate_max_range_pips", 40.0))
    v44_exhaustion_cooldown_minutes = max(1, int(v44_config.get("exhaustion_gate_cooldown_minutes", 15)))
    v44_daily_loss_limit_pips = float(v44_config.get("daily_loss_limit_pips", 0.0))
    v44_weekly_loss_limit_pips = float(v44_config.get("weekly_loss_limit_pips", 0.0))
    v44_daily_loss_limit_usd = float(v44_config.get("daily_loss_limit_usd", 0.0))
    v44_sizing_mode = str(v44_config.get("sizing_mode", "risk_parity")).lower()
    v44_base_lot = float(v44_config.get("base_lot", 2.0))
    v44_str_size_mults = {
        "strong": float(v44_config.get("strong_size_mult", 3.0)),
        "normal": float(v44_config.get("normal_size_mult", 2.0)),
        "weak": float(v44_config.get("weak_size_mult", 0.0)),
    }
    v44_win_bonus_step = float(v44_config.get("win_bonus_per_step", 0.0))
    v44_win_streak_scope = str(v44_config.get("win_streak_scope", "session")).lower()
    v44_max_lot = float(v44_config.get("max_lot", 10.0))
    v44_rp_str_mults = {
        "strong": float(v44_config.get("rp_strong_mult", 1.0)),
        "normal": float(v44_config.get("rp_normal_mult", 1.0)),
        "weak": float(v44_config.get("rp_weak_mult", 0.0)),
    }
    v44_rp_win_bonus_pct = float(v44_config.get("rp_win_bonus_pct", 0.0))
    v44_rp_max_lot_mult = float(v44_config.get("rp_max_lot_mult", 2.0))
    v44_rp_ny_mult = float(v44_config.get("rp_ny_mult", 1.0))
    v44_hyb_str_mults = {
        "strong": float(v44_config.get("hybrid_strong_boost", 1.3)),
        "normal": float(v44_config.get("hybrid_normal_boost", 0.8)),
        "weak": float(v44_config.get("hybrid_weak_boost", 0.0)),
    }
    v44_hybrid_ny_boost = float(v44_config.get("hybrid_ny_boost", 1.0))

    today = now_utc.date().isoformat()
    prev_day = str(phase3_state.get("ny_last_day") or "")
    if prev_day and prev_day != today:
        prev_key = f"session_ny_{prev_day}"
        prev_sdat = phase3_state.get(prev_key, {})
        if isinstance(prev_sdat, dict):
            try:
                o = float(prev_sdat.get("session_open_price"))
                c = float(prev_sdat.get("session_close_price"))
                hi = float(prev_sdat.get("session_high"))
                lo = float(prev_sdat.get("session_low"))
                rng = hi - lo
                if rng > 0:
                    eff = abs(c - o) / rng
                    hist = list(phase3_state.get("ny_session_efficiency_history", []))
                    hist.append(float(eff))
                    phase3_state["ny_session_efficiency_history"] = hist[-50:]
            except Exception:
                pass
    phase3_state["ny_last_day"] = today

    session_key = f"session_ny_{today}"
    sdat = dict(phase3_state.get(session_key, {}))
    sdat.setdefault("trade_count", 0)
    sdat.setdefault("consecutive_losses", 0)
    sdat.setdefault("wins_closed", 0)
    sdat.setdefault("cooldown_until", None)
    sdat.setdefault("last_entry_time", None)
    sdat.setdefault("exhaustion_cooldown_until", None)
    sdat.setdefault("win_streak", 0)
    sdat.setdefault("session_open_price", None)
    sdat.setdefault("session_high", None)
    sdat.setdefault("session_low", None)
    sdat.setdefault("session_close_price", None)
    sdat.setdefault("session_efficiency_history", list(phase3_state.get("ny_session_efficiency_history", [])))
    sdat.setdefault("queued_pending", None)
    sdat.setdefault("queued_stats", {"generated": 0, "confirmed": 0, "expired": 0, "replaced": 0})
    sdat.setdefault("news_trend_state", {})

    # Maintain session OHLC state used by dual-mode/session-range gates.
    m1_last = m1_df.iloc[-1]
    c_last = float(m1_last["close"])
    h_last = float(m1_last["high"])
    l_last = float(m1_last["low"])
    if sdat.get("session_open_price") is None:
        sdat["session_open_price"] = c_last
        sdat["session_high"] = h_last
        sdat["session_low"] = l_last
        sdat["session_close_price"] = c_last
    else:
        try:
            sdat["session_high"] = max(float(sdat.get("session_high", h_last)), h_last)
            sdat["session_low"] = min(float(sdat.get("session_low", l_last)), l_last)
            sdat["session_close_price"] = c_last
        except Exception:
            sdat["session_high"] = h_last
            sdat["session_low"] = l_last
            sdat["session_close_price"] = c_last

    if float(v44_session_range_cap_pips) > 0:
        try:
            sess_range_pips = (float(sdat.get("session_high")) - float(sdat.get("session_low"))) / pip
            if sess_range_pips > float(v44_session_range_cap_pips):
                no_trade["decision"] = ExecutionDecision(
                    attempted=False, placed=False,
                    reason=f"v44_ny: session range cap ({sess_range_pips:.1f}p > {v44_session_range_cap_pips:.1f}p)",
                    side=None,
                )
                no_trade["phase3_state_updates"] = {session_key: sdat}
                return no_trade
        except Exception:
            pass

    session_mode = _determine_v44_session_mode(
        sdat,
        dual_mode_enabled=v44_dual_mode_enabled,
        trend_mode_efficiency_min=v44_trend_mode_efficiency_min,
        range_mode_efficiency_max=v44_range_mode_efficiency_max,
    )
    _sess_hist_vals = [float(x) for x in sdat.get("session_efficiency_history", []) if isinstance(x, (int, float))]
    _sess_eff_avg = (sum(_sess_hist_vals) / len(_sess_hist_vals)) if _sess_hist_vals else None
    sdat["session_mode"] = session_mode
    sdat["session_efficiency_avg"] = float(_sess_eff_avg) if _sess_eff_avg is not None else None
    sdat["session_efficiency_hist_len"] = int(len(_sess_hist_vals))
    sdat["range_fade_enabled"] = bool(v44_range_fade_enabled)
    sdat["trend_mode_efficiency_min"] = float(v44_trend_mode_efficiency_min)
    sdat["range_mode_efficiency_max"] = float(v44_range_mode_efficiency_max)
    if v44_dual_mode_enabled and session_mode == "neutral":
        no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason="v44_ny: dual mode neutral", side=None)
        no_trade["phase3_state_updates"] = {session_key: sdat}
        return no_trade
    if v44_dual_mode_enabled and session_mode == "range_fade" and not v44_range_fade_enabled:
        no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason="v44_ny: range mode disabled", side=None)
        no_trade["phase3_state_updates"] = {session_key: sdat}
        return no_trade

    open_count = int(phase3_state.get("open_trade_count", 0))
    green_light_active = bool(v44_gl_enabled and int(sdat.get("wins_closed", 0)) >= int(v44_gl_min_wins))
    effective_max_open = int(v44_max_open)
    if green_light_active and int(v44_gl_max_open) > effective_max_open:
        effective_max_open = int(v44_gl_max_open)
    if open_count >= effective_max_open:
        no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason=f"v44_ny: max open {open_count}/{effective_max_open}", side=None)
        no_trade["phase3_state_updates"] = {session_key: sdat}
        return no_trade

    if v44_skip_days_raw:
        try:
            skip_days = {int(x.strip()) for x in v44_skip_days_raw.split(",") if str(x).strip() != ""}
        except Exception:
            skip_days = set()
        if now_utc.weekday() in skip_days:
            no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason="v44_ny: skip day", side=None)
            return no_trade
    if v44_skip_months_raw:
        try:
            skip_months = {int(x.strip()) for x in v44_skip_months_raw.split(",") if str(x).strip() != ""}
        except Exception:
            skip_months = set()
        if now_utc.month in skip_months:
            no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason="v44_ny: skip month", side=None)
            return no_trade

    # Daily / weekly loss limits (backtest parity: v5_daily_loss_limit_pips / v5_weekly_loss_limit_pips)
    # Use EXIT date semantics for realized-loss accounting.
    if store is not None and (v44_daily_loss_limit_pips > 0 or v44_weekly_loss_limit_pips > 0 or v44_daily_loss_limit_usd > 0):
        try:
            profile_name = getattr(profile, "profile_name", str(profile))
            get_by_exit_date = getattr(store, "get_closed_trades_for_exit_date", None)
            day_trades = None
            if v44_daily_loss_limit_pips > 0 or v44_daily_loss_limit_usd > 0:
                if callable(get_by_exit_date):
                    day_trades = get_by_exit_date(profile_name, today)
                else:
                    day_trades = store.get_trades_for_date(profile_name, today)
            if day_trades is not None:
                day_pips = sum(
                    float(t["pips"] or 0)
                    for t in day_trades
                    if str(t["entry_type"] or "").startswith("phase3:v44") and t["pips"] is not None
                )
                day_usd = sum(
                    float(t["profit"] or 0)
                    for t in day_trades
                    if str(t["entry_type"] or "").startswith("phase3:v44") and t["profit"] is not None
                )
                if v44_daily_loss_limit_pips > 0 and day_pips <= -v44_daily_loss_limit_pips:
                    no_trade["decision"] = ExecutionDecision(
                        attempted=False, placed=False,
                        reason=f"v44_ny: daily loss limit ({day_pips:.1f}p <= -{v44_daily_loss_limit_pips:.0f}p)",
                        side=None,
                    )
                    return no_trade
                if v44_daily_loss_limit_usd > 0 and day_usd <= -v44_daily_loss_limit_usd:
                    no_trade["decision"] = ExecutionDecision(
                        attempted=False, placed=False,
                        reason=f"v44_ny: daily loss USD ({day_usd:.2f} <= -{v44_daily_loss_limit_usd:.2f})",
                        side=None,
                    )
                    return no_trade
            if v44_weekly_loss_limit_pips > 0:
                wk_start = now_utc.date() - pd.Timedelta(days=now_utc.weekday())
                week_pips = 0.0
                for d_offset in range(7):
                    d_str = (wk_start + pd.Timedelta(days=d_offset)).isoformat()
                    if d_str > today:
                        break
                    if callable(get_by_exit_date):
                        wk_trades = get_by_exit_date(profile_name, d_str)
                    else:
                        wk_trades = store.get_trades_for_date(profile_name, d_str)
                    week_pips += sum(
                        float(t["pips"] or 0)
                        for t in wk_trades
                        if str(t["entry_type"] or "").startswith("phase3:v44") and t["pips"] is not None
                    )
                if week_pips <= -v44_weekly_loss_limit_pips:
                    no_trade["decision"] = ExecutionDecision(
                        attempted=False, placed=False,
                        reason=f"v44_ny: weekly loss limit ({week_pips:.1f}p <= -{v44_weekly_loss_limit_pips:.0f}p)",
                        side=None,
                    )
                    return no_trade
        except Exception:
            pass

    # H4 ADX gate (backtest parity: v5_h4_adx_min)
    if v44_h4_adx_min > 0 and h4_df is not None and len(h4_df) >= 14:
        h4_adx_val = _compute_adx(h4_df, 14)
        if h4_adx_val < v44_h4_adx_min:
            no_trade["decision"] = ExecutionDecision(
                attempted=False, placed=False,
                reason=f"v44_ny: H4 ADX {h4_adx_val:.1f} < min {v44_h4_adx_min:.1f}",
                side=None,
            )
            return no_trade

    # H1 slope magnitude + consistency gates.
    if v44_h1_slope_min > 0 and h1_df is not None and len(h1_df) >= max(2, v44_slope_bars + 1):
        _h1_close = h1_df["close"].astype(float)
        _h1_ema = _compute_ema(_h1_close, v44_h1_ema_fast)
        _sb = max(1, int(v44_slope_bars))
        if len(_h1_ema) > _sb:
            _h1_slope_mag = abs((float(_h1_ema.iloc[-1]) - float(_h1_ema.iloc[-1 - _sb])) / (float(_sb) * pip))
            if _h1_slope_mag < float(v44_h1_slope_min):
                no_trade["decision"] = ExecutionDecision(
                    attempted=False, placed=False,
                    reason=f"v44_ny: H1 slope weak ({_h1_slope_mag:.2f} < {v44_h1_slope_min:.2f})",
                    side=None,
                )
                no_trade["phase3_state_updates"] = {session_key: sdat}
                return no_trade
    if v44_h1_slope_consistent_bars > 0 and h1_df is not None and len(h1_df) >= int(v44_h1_slope_consistent_bars) + 2:
        _h1_close = h1_df["close"].astype(float)
        _h1_fast = _compute_ema(_h1_close, v44_h1_ema_fast)
        _h1_slow = _compute_ema(_h1_close, v44_h1_ema_slow)
        trend_side = None
        if float(_h1_fast.iloc[-1]) > float(_h1_slow.iloc[-1]):
            trend_side = "buy"
        elif float(_h1_fast.iloc[-1]) < float(_h1_slow.iloc[-1]):
            trend_side = "sell"
        if trend_side is not None:
            consistent = True
            n = int(v44_h1_slope_consistent_bars)
            for k in range(1, n + 1):
                e_now = float(_h1_fast.iloc[-k])
                e_prev = float(_h1_fast.iloc[-k - 1])
                bar_slope = e_now - e_prev
                if (trend_side == "buy" and bar_slope <= 0) or (trend_side == "sell" and bar_slope >= 0):
                    consistent = False
                    break
            if not consistent:
                no_trade["decision"] = ExecutionDecision(
                    attempted=False, placed=False,
                    reason="v44_ny: H1 slope inconsistent",
                    side=None,
                )
                no_trade["phase3_state_updates"] = {session_key: sdat}
                return no_trade

    # Exhaustion gate (backtest parity: v5_exhaustion_gate_*)
    if v44_exhaustion_gate_enabled:
        exh_cooldown = sdat.get("exhaustion_cooldown_until")
        if exh_cooldown is not None:
            try:
                exh_cd_dt = pd.Timestamp(exh_cooldown).tz_localize("UTC") if pd.Timestamp(exh_cooldown).tzinfo is None else pd.Timestamp(exh_cooldown).tz_convert("UTC")
                if pd.Timestamp(now_utc) < exh_cd_dt:
                    no_trade["decision"] = ExecutionDecision(
                        attempted=False, placed=False,
                        reason=f"v44_ny: exhaustion gate cooldown until {exh_cooldown}",
                        side=None,
                    )
                    return no_trade
            except Exception:
                pass
        if m5_df is not None and not m5_df.empty and "time" in m5_df.columns:
            try:
                m5_times = pd.to_datetime(m5_df["time"])
                if m5_times.dt.tz is None:
                    m5_times = m5_times.dt.tz_localize("UTC")
                cutoff_ts = pd.Timestamp(now_utc).tz_convert("UTC") - pd.Timedelta(minutes=v44_exhaustion_window_minutes)
                m5_win = m5_df[m5_times >= cutoff_ts]
            except Exception:
                m5_win = m5_df.tail(max(1, v44_exhaustion_window_minutes // 5))
            if not m5_win.empty:
                ex_range_pips = (float(m5_win["high"].max()) - float(m5_win["low"].min())) / pip
                if ex_range_pips > v44_exhaustion_max_range_pips:
                    cd_until = (now_utc + pd.Timedelta(minutes=v44_exhaustion_cooldown_minutes)).isoformat()
                    sdat["exhaustion_cooldown_until"] = cd_until
                    no_trade["decision"] = ExecutionDecision(
                        attempted=False, placed=False,
                        reason=f"v44_ny: exhaustion gate ({ex_range_pips:.1f}p > {v44_exhaustion_max_range_pips:.0f}p)",
                        side=None,
                    )
                    no_trade["phase3_state_updates"] = {session_key: sdat}
                    return no_trade

    if _ts_now >= (ny_end_ts - pd.Timedelta(minutes=max(0, v44_entry_cutoff_minutes))):
        no_trade["decision"] = ExecutionDecision(
            attempted=False,
            placed=False,
            reason=f"v44_ny: entry cutoff ({v44_entry_cutoff_minutes}m before session end)",
            side=None,
        )
        no_trade["phase3_state_updates"] = {session_key: sdat}
        return no_trade

    news_events: tuple[pd.Timestamp, ...] = tuple()
    in_news_window = False
    if v44_news_filter_enabled:
        news_events = _load_news_events_cached(v44_news_calendar_path, v44_news_impact_min)
        in_news_window = bool(is_in_news_window(now_utc, news_events, v44_news_before, v44_news_after))
        if in_news_window and not v44_news_trend_enabled:
            no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason="v44_ny: news window block", side=None)
            no_trade["phase3_state_updates"] = {session_key: sdat}
            return no_trade

    effective_max_entries_day = int(v44_max_entries_day) + (int(v44_gl_extra_entries) if green_light_active else 0)
    eff_strong_slope = float(v44_strong_slope)
    if green_light_active and v44_gl_slope_relax > 0:
        eff_strong_slope = max(0.0, float(v44_strong_slope) * (1.0 - float(v44_gl_slope_relax)))

    side: Optional[str] = None
    strength: str = "normal"
    reason: str = "v44_ny: news window block"
    queued = sdat.get("queued_pending")
    queued_confirmed = False
    latest_m5_ts = pd.Timestamp(m5_df["time"].iloc[-1]) if ("time" in m5_df.columns and len(m5_df) > 0) else None
    if latest_m5_ts is not None:
        if latest_m5_ts.tzinfo is None:
            latest_m5_ts = latest_m5_ts.tz_localize("UTC")
        else:
            latest_m5_ts = latest_m5_ts.tz_convert("UTC")

    if not in_news_window:
        side, strength, reason = evaluate_v44_entry(
            h1_df,
            m5_df,
            tick,
            pip,
            "ny",
            sdat,
            now_utc=now_utc,
            max_entries_per_day=effective_max_entries_day,
            session_stop_losses=v44_session_stop_losses,
            h1_ema_fast_period=v44_h1_ema_fast,
            h1_ema_slow_period=v44_h1_ema_slow,
            m5_ema_fast_period=v44_m5_ema_fast,
            m5_ema_slow_period=v44_m5_ema_slow,
            slope_bars=v44_slope_bars,
            strong_slope_threshold=eff_strong_slope,
            weak_slope_threshold=v44_weak_slope,
            min_body_pips=v44_min_body_pips,
            atr_pct_filter_enabled=v44_atr_pct_filter_enabled,
            atr_pct_cap=v44_atr_pct_cap,
            atr_pct_lookback=v44_atr_pct_lookback,
        )
        # Optional queued confirmation parity (v5_queued_confirm_bars) on closed M5 bars.
        if isinstance(queued, dict) and latest_m5_ts is not None:
            exp_ts = pd.Timestamp(queued.get("expiry_time")) if queued.get("expiry_time") else None
            if exp_ts is not None:
                if exp_ts.tzinfo is None:
                    exp_ts = exp_ts.tz_localize("UTC")
                else:
                    exp_ts = exp_ts.tz_convert("UTC")
            if exp_ts is not None and latest_m5_ts > exp_ts:
                sdat["queued_pending"] = None
                qs = dict(sdat.get("queued_stats", {}))
                qs["expired"] = int(qs.get("expired", 0)) + 1
                sdat["queued_stats"] = qs
                queued = None
            elif str(queued.get("last_checked_m5_time", "")) != latest_m5_ts.isoformat():
                m5_open = float(m5_df["open"].astype(float).iloc[-1])
                m5_close = float(m5_df["close"].astype(float).iloc[-1])
                m5_body_pips = abs(m5_close - m5_open) / pip
                q_side = str(queued.get("side", "")).lower()
                is_confirm = (
                    ((q_side == "buy" and m5_close > m5_open) or (q_side == "sell" and m5_close < m5_open))
                    and m5_body_pips >= float(v44_min_body_pips)
                )
                if is_confirm:
                    rem = int(queued.get("remaining", 1)) - 1
                    queued["remaining"] = rem
                    queued["last_checked_m5_time"] = latest_m5_ts.isoformat()
                    if rem <= 0:
                        side = q_side
                        strength = str(queued.get("strength", strength))
                        reason = f"v44: queued confirm ({queued.get('confirm_bars', 0)} bars)"
                        sdat["queued_pending"] = None
                        queued_confirmed = True
                        qs = dict(sdat.get("queued_stats", {}))
                        qs["confirmed"] = int(qs.get("confirmed", 0)) + 1
                        sdat["queued_stats"] = qs
                    else:
                        sdat["queued_pending"] = queued
                        no_trade["decision"] = ExecutionDecision(
                            attempted=False,
                            placed=False,
                            reason=f"v44: queued confirmation progress {max(0, rem)} remaining",
                            side=None,
                        )
                        no_trade["phase3_state_updates"] = {session_key: sdat}
                        return no_trade
                else:
                    # Keep waiting until expiry; do not consume bar on failed direction.
                    queued["last_checked_m5_time"] = latest_m5_ts.isoformat()
                    sdat["queued_pending"] = queued

    if side is None:
        # Optional post-news trend workflow parity.
        if v44_news_filter_enabled and v44_news_trend_enabled and news_events:
            ev = _v44_news_trend_active_event(
                now_utc,
                news_events,
                delay_minutes=v44_news_trend_delay_minutes,
                window_minutes=v44_news_trend_window_minutes,
            )
            if ev is not None and len(m1_df) >= int(v44_news_trend_confirm_bars):
                h1_tr = compute_v44_h1_trend(h1_df, v44_h1_ema_fast, v44_h1_ema_slow)
                if h1_tr in {"up", "down"}:
                    nt_side = "buy" if h1_tr == "up" else "sell"
                    confirm_ok = True
                    conf_n = int(v44_news_trend_confirm_bars)
                    for j in range(len(m1_df) - conf_n, len(m1_df)):
                        oj = float(m1_df.iloc[j]["open"])
                        cj = float(m1_df.iloc[j]["close"])
                        body = abs(cj - oj) / pip
                        dir_ok = (cj > oj) if nt_side == "buy" else (cj < oj)
                        if (not dir_ok) or (body < float(v44_news_trend_min_body_pips)):
                            confirm_ok = False
                            break
                    if confirm_ok:
                        if v44_news_trend_require_ema_align:
                            cser = m5_df["close"].astype(float)
                            ef = _compute_ema(cser, v44_m5_ema_fast)
                            es = _compute_ema(cser, v44_m5_ema_slow)
                            ema_ok = float(ef.iloc[-1]) > float(es.iloc[-1]) if nt_side == "buy" else float(ef.iloc[-1]) < float(es.iloc[-1])
                            if not ema_ok:
                                no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason="v44_ny: news-trend EMA misaligned", side=None)
                                no_trade["phase3_state_updates"] = {session_key: sdat}
                                return no_trade
                        side = nt_side
                        strength = "strong"
                        reason = "v44: news trend confirmation"
        if side is None:
            no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason=reason, side=None)
            no_trade["phase3_state_updates"] = {session_key: sdat}
            return no_trade

    # Strength gating + green-light normal override parity.
    atr_rank = _compute_v44_atr_rank(m5_df, lookback=v44_atr_pct_lookback)
    allow_map = {
        "strong_only": {"strong"},
        "strong_normal": {"strong", "normal"},
        "all": {"strong", "normal", "weak"},
    }
    allowed_strengths = allow_map.get(v44_ny_strength_allow, {"strong", "normal"})
    strength_allowed = strength in allowed_strengths
    if strength == "weak" and v44_skip_weak:
        strength_allowed = False
    if strength == "normal" and v44_skip_normal:
        strength_allowed = False
    if (not strength_allowed) and strength == "normal" and v44_allow_normal_plus and atr_rank is not None:
        atr_now = float(m5_df["high"].astype(float).iloc[-1] - m5_df["low"].astype(float).iloc[-1]) / pip
        if atr_now >= float(v44_normalplus_atr_min_pips):
            strength_allowed = True
    if (not strength_allowed) and strength == "normal" and green_light_active and v44_gl_allow_normal and atr_rank is not None:
        if float(atr_rank) < float(v44_gl_normal_atr_cap):
            strength_allowed = True
    if not strength_allowed:
        no_trade["decision"] = ExecutionDecision(
            attempted=False,
            placed=False,
            reason=f"v44_ny: strength {strength} blocked by {v44_ny_strength_allow}",
            side=None,
        )
        no_trade["phase3_state_updates"] = {session_key: sdat}
        return no_trade

    if v44_queued_confirm_bars > 0 and (not queued_confirmed):
        cur_pending = sdat.get("queued_pending")
        if isinstance(cur_pending, dict):
            cur_side = str(cur_pending.get("side", "")).lower()
            if cur_side != str(side).lower():
                exp_ts = _ts_now + pd.Timedelta(minutes=max(5, int(v44_queued_confirm_bars) * 5))
                sdat["queued_pending"] = {
                    "side": side,
                    "strength": strength,
                    "confirm_bars": int(v44_queued_confirm_bars),
                    "remaining": int(v44_queued_confirm_bars),
                    "expiry_time": exp_ts.isoformat(),
                    "last_checked_m5_time": latest_m5_ts.isoformat() if latest_m5_ts is not None else "",
                }
                qs = dict(sdat.get("queued_stats", {}))
                qs["generated"] = int(qs.get("generated", 0)) + 1
                qs["replaced"] = int(qs.get("replaced", 0)) + 1
                sdat["queued_stats"] = qs
                no_trade["decision"] = ExecutionDecision(
                    attempted=False,
                    placed=False,
                    reason=f"v44: queued signal replaced with {side}",
                    side=None,
                )
                no_trade["phase3_state_updates"] = {session_key: sdat}
                return no_trade
            no_trade["decision"] = ExecutionDecision(
                attempted=False,
                placed=False,
                reason=f"v44: queued signal already pending ({cur_side})",
                side=None,
            )
            no_trade["phase3_state_updates"] = {session_key: sdat}
            return no_trade
        exp_ts = _ts_now + pd.Timedelta(minutes=max(5, int(v44_queued_confirm_bars) * 5))
        sdat["queued_pending"] = {
            "side": side,
            "strength": strength,
            "confirm_bars": int(v44_queued_confirm_bars),
            "remaining": int(v44_queued_confirm_bars),
            "expiry_time": exp_ts.isoformat(),
            "last_checked_m5_time": latest_m5_ts.isoformat() if latest_m5_ts is not None else "",
        }
        qs = dict(sdat.get("queued_stats", {}))
        qs["generated"] = int(qs.get("generated", 0)) + 1
        sdat["queued_stats"] = qs
        no_trade["decision"] = ExecutionDecision(
            attempted=False,
            placed=False,
            reason=f"v44: signal queued for {v44_queued_confirm_bars} confirmations ({strength})",
            side=None,
        )
        no_trade["phase3_state_updates"] = {session_key: sdat}
        return no_trade

    entry_price = tick.ask if side == "buy" else tick.bid
    is_news_trend_entry = "news trend" in str(reason).lower()
    risk_pct_for_trade = float(v44_news_trend_risk_pct) if is_news_trend_entry else float(v44_risk_pct)
    if is_news_trend_entry:
        sl_pips = max(0.1, float(v44_news_trend_sl_pips))
        sl_price = entry_price - sl_pips * pip if side == "buy" else entry_price + sl_pips * pip
        tp1_pips = max(0.1, float(v44_news_trend_tp_rr) * sl_pips)
    else:
        sl_price = compute_v44_sl(side, m5_df, entry_price, pip)
        sl_pips = abs(entry_price - sl_price) / pip
        # Config values (v5_strong_tp1 etc.) are R-multiples, not absolute pips.
        if strength == "strong":
            tp1_pips = v44_strong_tp1_pips * sl_pips
        elif strength == "normal":
            tp1_pips = v44_normal_tp1_pips * sl_pips
        else:
            tp1_pips = v44_weak_tp1_pips * sl_pips
    tp1_price = entry_price + (tp1_pips * pip if side == "buy" else -tp1_pips * pip)

    equity = _account_sizing_value(adapter, fallback=100000.0)

    pip_value_per_lot = (pip / entry_price) * 100000.0
    if v44_win_streak_scope == "session":
        bonus_steps = max(0, int(sdat.get("win_streak", 0)))
    else:
        bonus_steps = max(0, int(phase3_state.get("v44_win_streak_global", 0)))
    units = 0
    if v44_sizing_mode == "multiplier":
        regime_lot = v44_base_lot * float(v44_str_size_mults.get(strength, 1.0))
        lot = min(v44_max_lot, max(0.01, regime_lot + (v44_win_bonus_step * bonus_steps)))
        units = int(lot * 100000.0)
    elif v44_sizing_mode == "hybrid":
        if sl_pips > 0 and pip_value_per_lot > 0:
            risk_usd = equity * risk_pct_for_trade
            raw_lot = risk_usd / (sl_pips * pip_value_per_lot)
            raw_lot *= float(v44_hyb_str_mults.get(strength, 1.0))
            raw_lot *= float(v44_hybrid_ny_boost)
            bonus_factor = min(v44_rp_max_lot_mult, 1.0 + (v44_rp_win_bonus_pct / 100.0) * bonus_steps)
            lot = max(v44_rp_min_lot, min(v44_rp_max_lot, raw_lot * bonus_factor))
            units = int(lot * 100000.0)
    else:  # risk_parity
        if sl_pips > 0 and pip_value_per_lot > 0:
            risk_usd = equity * risk_pct_for_trade
            raw_lot = risk_usd / (sl_pips * pip_value_per_lot)
            raw_lot *= float(v44_rp_str_mults.get(strength, 1.0))
            raw_lot *= float(v44_rp_ny_mult)
            bonus_factor = min(v44_rp_max_lot_mult, 1.0 + (v44_rp_win_bonus_pct / 100.0) * bonus_steps)
            lot = max(v44_rp_min_lot, min(v44_rp_max_lot, raw_lot * bonus_factor))
            units = int(lot * 100000.0)
    if units <= 0:
        no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason="v44_ny: size=0", side=None)
        return no_trade

    strategy_tag = "phase3:v44_ny:news" if is_news_trend_entry else f"phase3:v44_ny:{strength}"
    comment = f"phase3_integrated:{policy.id}:v44_ny"
    try:
        dec = adapter.place_order(
            symbol=profile.symbol,
            side=side,
            lots=units / 100000.0,
            stop_price=round(float(sl_price), 3),
            target_price=round(float(tp1_price), 3),
            comment=comment,
        )
    except Exception as e:
        return {
            "decision": ExecutionDecision(attempted=True, placed=False, reason=f"v44_ny order error: {e}", side=side),
            "phase3_state_updates": {},
            "strategy_tag": strategy_tag,
        }

    confirmed_fill, confirmed_deal_id = _phase3_order_confirmed(adapter, profile, dec)
    if not confirmed_fill:
        return {
            "decision": ExecutionDecision(
                attempted=True,
                placed=False,
                reason="v44_ny: broker order pending/unfilled",
                side=side,
                order_retcode=getattr(dec, "order_retcode", None),
                order_id=getattr(dec, "order_id", None),
                deal_id=getattr(dec, "deal_id", None),
                fill_price=getattr(dec, "fill_price", None),
            ),
            "phase3_state_updates": {},
            "strategy_tag": strategy_tag,
        }

    sdat["trade_count"] = int(sdat.get("trade_count", 0)) + 1
    sdat["last_entry_time"] = now_utc.isoformat()

    risk_usd_planned = float(sl_pips) * (pip / entry_price) * float(units) if units > 0 else 0.0

    return {
        "decision": ExecutionDecision(
            attempted=True,
            placed=True,
            reason=f"{reason} | strength={strength} TP1={tp1_pips:.2f}p SL={sl_pips:.1f}p units={units}",
            side=side,
            order_retcode=getattr(dec, "order_retcode", None),
            order_id=getattr(dec, "order_id", None),
            deal_id=confirmed_deal_id if confirmed_deal_id is not None else getattr(dec, "deal_id", None),
            fill_price=getattr(dec, "fill_price", None),
        ),
        "phase3_state_updates": {session_key: sdat},
        "strategy_tag": strategy_tag,
        "sl_price": float(sl_price),
        "tp1_price": float(tp1_price),
        "units": int(units),
        "entry_price": float(entry_price),
        "sl_pips": float(sl_pips),
        "risk_usd_planned": float(risk_usd_planned),
    }


# ===================================================================
#  Main execute function
# ===================================================================

def execute_phase3_integrated_policy_demo_only(
    *,
    adapter,
    profile,
    log_dir,
    policy,
    context,
    data_by_tf: dict,
    tick,
    mode: str,
    phase3_state: dict,
    store=None,
    sizing_config: Optional[dict[str, Any]] = None,
    is_new_m1: bool = True,
) -> dict:
    """
    Main Phase 3 entry point.  Returns dict with:
      decision: ExecutionDecision-like dict
      phase3_state_updates: dict to merge into phase3_state
      strategy_tag: str | None  (e.g. "phase3:v14")
    """
    now_utc = datetime.now(timezone.utc)
    effective_cfg = sizing_config if isinstance(sizing_config, dict) and ("v14" in sizing_config or "london_v2" in sizing_config or "v44_ny" in sizing_config) else load_phase3_sizing_config()
    pip = float(profile.pip_size) if hasattr(profile, "pip_size") else PIP_SIZE
    no_trade = {
        "decision": ExecutionDecision(attempted=False, placed=False, reason="", side=None),
        "phase3_state_updates": {},
        "strategy_tag": None,
    }

    # 1) Session routing
    # Use latest fully closed M1 bar time for session/window decisions (bar-close parity).
    _m1_ref = _drop_incomplete_tf(data_by_tf.get("M1"), "M1")
    if _m1_ref is not None and not _m1_ref.empty and "time" in _m1_ref.columns:
        try:
            _t = pd.Timestamp(_m1_ref["time"].iloc[-1])
            if _t.tzinfo is None:
                _t = _t.tz_localize("UTC")
            else:
                _t = _t.tz_convert("UTC")
            now_utc = _t.to_pydatetime()
        except Exception:
            pass

    session = classify_session(now_utc, effective_cfg)
    base_state_updates: dict[str, Any] = {}
    meta = effective_cfg.get("_meta", {}) if isinstance(effective_cfg, dict) else {}
    eff_hash = str(meta.get("effective_hash", "") or "")
    today_str_for_cfg = now_utc.date().isoformat()
    if eff_hash:
        prev_hash = str(phase3_state.get("effective_phase3_config_hash", "") or "")
        prev_day = str(phase3_state.get("effective_phase3_config_date", "") or "")
        if prev_hash != eff_hash or prev_day != today_str_for_cfg:
            base_state_updates["effective_phase3_config_hash"] = eff_hash
            base_state_updates["effective_phase3_config_date"] = today_str_for_cfg
            base_state_updates["effective_phase3_config"] = {
                "hash": eff_hash,
                "loaded_at_utc": meta.get("loaded_at_utc"),
                "source_paths": meta.get("source_paths", {}),
                "v14": effective_cfg.get("v14", {}),
                "london_v2": effective_cfg.get("london_v2", {}),
                "v44_ny": effective_cfg.get("v44_ny", {}),
            }
    no_trade["phase3_state_updates"] = dict(base_state_updates)

    # Backtest/live parity: evaluate entries only once per newly closed M1 bar for all Phase 3 sessions.
    if session in {"tokyo", "london", "ny"} and not is_new_m1:
        no_trade["decision"] = ExecutionDecision(
            attempted=False, placed=False,
            reason="phase3: waiting for new closed M1 bar", side=None,
        )
        no_trade["phase3_state_updates"] = base_state_updates
        return no_trade
    if session == "london":
        ldn_res = execute_london_v2_entry(
            adapter=adapter,
            profile=profile,
            policy=policy,
            data_by_tf=data_by_tf,
            tick=tick,
            phase3_state=phase3_state,
            sizing_config=effective_cfg,
            now_utc=now_utc,
            store=store,
        )
        merged = dict(base_state_updates)
        merged.update(ldn_res.get("phase3_state_updates") or {})
        ldn_res["phase3_state_updates"] = merged
        return ldn_res
    if session == "ny":
        ny_res = execute_v44_ny_entry(
            adapter=adapter,
            profile=profile,
            policy=policy,
            data_by_tf=data_by_tf,
            tick=tick,
            phase3_state=phase3_state,
            sizing_config=effective_cfg,
            now_utc=now_utc,
            store=store,
        )
        merged = dict(base_state_updates)
        merged.update(ny_res.get("phase3_state_updates") or {})
        ny_res["phase3_state_updates"] = merged
        return ny_res
    if session != "tokyo":
        no_trade["decision"] = ExecutionDecision(
            attempted=False, placed=False,
            reason=f"phase3: no active session (current={session})", side=None,
        )
        no_trade["phase3_state_updates"] = base_state_updates
        return no_trade

    # 2) Demo guard
    if mode != "ARMED_AUTO_DEMO":
        no_trade["decision"] = ExecutionDecision(
            attempted=False, placed=False,
            reason=f"phase3: mode={mode} (need ARMED_AUTO_DEMO)", side=None,
        )
        return no_trade

    is_demo = getattr(adapter, "is_demo", True)
    if callable(is_demo):
        is_demo = is_demo()
    if not is_demo:
        no_trade["decision"] = ExecutionDecision(
            attempted=False, placed=False,
            reason="phase3: adapter is not demo", side=None,
        )
        return no_trade

    # 3) Spread gate
    spread_pips = (tick.ask - tick.bid) / pip
    v14_cfg_for_spread = (effective_cfg or {}).get("v14", {})
    v14_max_entry_spread = float(v14_cfg_for_spread.get("max_entry_spread_pips", MAX_ENTRY_SPREAD_PIPS))
    if spread_pips > v14_max_entry_spread:
        no_trade["decision"] = ExecutionDecision(
            attempted=False, placed=False,
            reason=f"phase3: spread {spread_pips:.1f}p > max {v14_max_entry_spread:.1f}p", side=None,
        )
        return no_trade

    # 4) Pivots (cache by NY-close trading day, matching the Tokyo backtest)
    current_ny_day = (pd.Timestamp(now_utc) - pd.Timedelta(hours=22)).date()
    prev_ny_day = current_ny_day - pd.Timedelta(days=1)
    today_str = current_ny_day.isoformat()
    pivots = phase3_state.get("pivots")
    pivots_date = phase3_state.get("pivots_date")
    state_updates: dict = dict(base_state_updates)
    if pivots is None or pivots_date != today_str:
        m1_for_pivots = _drop_incomplete_tf(data_by_tf.get("M1"), "M1")
        if m1_for_pivots is None or m1_for_pivots.empty:
            no_trade["decision"] = ExecutionDecision(
                attempted=False, placed=False,
                reason="phase3: no M1 data for pivots", side=None,
            )
            return no_trade
        pivot_src = m1_for_pivots.copy()
        pivot_src["time"] = pd.to_datetime(pivot_src["time"], utc=True, errors="coerce")
        pivot_src = pivot_src.dropna(subset=["time"]).sort_values("time")
        pivot_src["ny_day"] = (pivot_src["time"] - pd.Timedelta(hours=22)).dt.date
        prev_day_rows = pivot_src[pivot_src["ny_day"] == prev_ny_day]
        if prev_day_rows.empty:
            no_trade["decision"] = ExecutionDecision(
                attempted=False, placed=False,
                reason="phase3: no prior NY-close day data for pivots", side=None,
            )
            return no_trade
        pivots = compute_daily_fib_pivots(
            float(prev_day_rows["high"].max()),
            float(prev_day_rows["low"].min()),
            float(prev_day_rows["close"].iloc[-1]),
        )
        state_updates["pivots"] = pivots
        state_updates["pivots_date"] = today_str

    # 5) Indicators
    m5_df = _drop_incomplete_tf(data_by_tf.get("M5"), "M5")
    m15_df = _drop_incomplete_tf(data_by_tf.get("M15"), "M15")
    m1_df = _drop_incomplete_tf(data_by_tf.get("M1"), "M1")
    if m5_df is None or m5_df.empty or m15_df is None or m15_df.empty or m1_df is None or m1_df.empty:
        no_trade["decision"] = ExecutionDecision(
            attempted=False, placed=False,
            reason="phase3: missing M1/M5/M15 data", side=None,
        )
        no_trade["phase3_state_updates"] = state_updates
        return no_trade

    v14_config = (effective_cfg or {}).get("v14", {})
    bb_regime_mode = str(v14_config.get("bb_regime_mode", "percentile")).strip().lower()
    bb_regime_lookback = max(1, int(v14_config.get("bb_width_lookback", BB_WIDTH_LOOKBACK)))
    bb_regime_ranging_pct = float(v14_config.get("bb_width_ranging_pct", BB_WIDTH_RANGING_PCT))
    bb_required_bars = BB_PERIOD + (4 if bb_regime_mode in {"expanding3", "bb_width_expanding3"} else bb_regime_lookback)

    # BB + regime
    if len(m5_df) < bb_required_bars:
        no_trade["decision"] = ExecutionDecision(
            attempted=False, placed=False,
            reason="phase3: insufficient M5 data for BB", side=None,
        )
        no_trade["phase3_state_updates"] = state_updates
        return no_trade

    regime = compute_bb_width_regime(
        m5_df,
        mode=bb_regime_mode,
        lookback=bb_regime_lookback,
        ranging_pct=bb_regime_ranging_pct,
    )
    bb_upper, bb_mid, bb_lower = _compute_bb(m5_df)

    # RSI
    m5_close = m5_df["close"].astype(float)
    rsi_series = _compute_rsi(m5_close, RSI_PERIOD)
    rsi_val = float(rsi_series.iloc[-1]) if pd.notna(rsi_series.iloc[-1]) else 50.0

    # ATR (M15)
    if len(m15_df) < ATR_PERIOD + 2:
        no_trade["decision"] = ExecutionDecision(
            attempted=False, placed=False,
            reason="phase3: insufficient M15 data for ATR", side=None,
        )
        no_trade["phase3_state_updates"] = state_updates
        return no_trade
    atr_series = _compute_atr(m15_df, ATR_PERIOD)
    atr_val = float(atr_series.iloc[-1]) if pd.notna(atr_series.iloc[-1]) else 0.0

    # ADX (M15)
    adx_val = _compute_adx(m15_df, ADX_PERIOD)

    # PSAR + flip
    sar_bull, sar_bear = detect_sar_flip(m1_df, PSAR_FLIP_LOOKBACK)

    day_name = now_utc.strftime("%A")
    atr_max_entry = float(v14_config.get("atr_max_threshold_price_units", ATR_MAX))
    adx_max_entry = float(v14_config.get("adx_max_for_entry", ADX_MAX))
    adx_filter_enabled = bool(v14_config.get("adx_filter_enabled", True))
    adx_day_overrides = v14_config.get("adx_max_by_day", {})
    if isinstance(adx_day_overrides, dict):
        try:
            adx_max_entry = float(adx_day_overrides.get(day_name, adx_max_entry))
        except Exception:
            pass
    core_gate_required = int(v14_config.get("core_gate_required", v14_config.get("min_confluence", MIN_CONFLUENCE)))
    min_conf_long = int(v14_config.get("confluence_min_long", core_gate_required))
    min_conf_short = int(v14_config.get("confluence_min_short", core_gate_required))
    blocked_combos_cfg = v14_config.get("blocked_combos", BLOCKED_COMBOS)
    if isinstance(blocked_combos_cfg, (list, tuple, set)):
        blocked_combos = {str(x) for x in blocked_combos_cfg}
    elif isinstance(blocked_combos_cfg, str):
        blocked_combos = {x.strip() for x in blocked_combos_cfg.split(",") if x.strip()}
    else:
        blocked_combos = set(BLOCKED_COMBOS)
    zone_tolerance_pips = float(v14_config.get("zone_tolerance_pips", ZONE_TOLERANCE_PIPS))
    rsi_long_entry = float(v14_config.get("rsi_long_entry", RSI_LONG_ENTRY))
    rsi_short_entry = float(v14_config.get("rsi_short_entry", RSI_SHORT_ENTRY))
    sl_buffer_pips = float(v14_config.get("sl_buffer_pips", SL_BUFFER_PIPS))
    min_sl_pips = float(v14_config.get("min_sl_pips", SL_MIN_PIPS))
    max_sl_pips = float(v14_config.get("max_sl_pips", SL_MAX_PIPS))
    partial_tp_atr_mult = float(v14_config.get("partial_tp_atr_mult", TP1_ATR_MULT))
    partial_tp_min_pips = float(v14_config.get("partial_tp_min_pips", TP1_MIN_PIPS))
    partial_tp_max_pips = float(v14_config.get("partial_tp_max_pips", TP1_MAX_PIPS))
    core_gate_use_zone       = bool(v14_config.get("core_gate_use_zone", True))
    core_gate_use_bb         = bool(v14_config.get("core_gate_use_bb", True))
    core_gate_use_sar        = bool(v14_config.get("core_gate_use_sar", True))
    core_gate_use_rsi        = bool(v14_config.get("core_gate_use_rsi", True))
    confirmation_enabled     = bool(v14_config.get("confirmation_enabled", False))
    confirmation_type        = str(v14_config.get("confirmation_type", "m1")).lower()
    confirmation_window_bars = max(0, int(v14_config.get("confirmation_window_bars", 0)))
    if not confirmation_enabled:
        confirmation_window_bars = 0
    sst_cfg = v14_config.get("signal_strength_tracking", {}) if isinstance(v14_config.get("signal_strength_tracking"), dict) else {}
    sst_enabled = bool(sst_cfg.get("enabled", False))
    sst_filter_on = bool(sst_cfg.get("filter_on_it", False))
    sst_min_score = int(sst_cfg.get("min_strength_score", 0))
    session_loss_stop_frac = _as_risk_fraction(v14_config.get("session_loss_stop_pct", SESSION_LOSS_STOP_PCT), SESSION_LOSS_STOP_PCT)
    min_time_between_entries = int(v14_config.get("min_time_between_entries_minutes", v14_config.get("cooldown_minutes", COOLDOWN_MINUTES)))
    no_reentry_after_stop_min = int(v14_config.get("no_reentry_same_direction_after_stop_minutes", 0))
    breakout_disable_pips = float(v14_config.get("disable_entries_if_move_from_tokyo_open_range_exceeds_pips", 0.0))
    breakout_mode = str(v14_config.get("breakout_detection_mode", "rolling")).lower()
    breakout_window_min = max(1, int(v14_config.get("rolling_window_minutes", 60)))
    breakout_window_pips = float(v14_config.get("rolling_range_threshold_pips", 40.0))

    # 6) Regime gate
    if regime != "ranging":
        no_trade["decision"] = ExecutionDecision(
            attempted=False, placed=False,
            reason=f"phase3: regime={regime} (need ranging)", side=None,
        )
        no_trade["phase3_state_updates"] = state_updates
        return no_trade

    # 7) ADX gate
    if adx_filter_enabled and adx_val >= adx_max_entry:
        no_trade["decision"] = ExecutionDecision(
            attempted=False, placed=False,
            reason=f"phase3: ADX={adx_val:.1f} >= {adx_max_entry:.1f}", side=None,
        )
        no_trade["phase3_state_updates"] = state_updates
        return no_trade

    # 8) ATR gate
    if atr_val >= atr_max_entry:
        no_trade["decision"] = ExecutionDecision(
            attempted=False, placed=False,
            reason=f"phase3: ATR={atr_val:.4f} >= {atr_max_entry:.4f}", side=None,
        )
        no_trade["phase3_state_updates"] = state_updates
        return no_trade

    # 9) Session management: concurrent positions, trade count, cooldown, consecutive losses (config-driven)
    max_trades_session = int(v14_config.get("max_trades_per_session", MAX_TRADES_PER_SESSION))
    stop_consec_losses = int(v14_config.get("stop_after_consecutive_losses", STOP_AFTER_CONSECUTIVE_LOSSES))
    cooldown_min = int(min_time_between_entries)
    max_concurrent = int(v14_config.get("max_concurrent_positions", MAX_CONCURRENT))

    session_key = f"session_tokyo_{today_str}"
    session_data = phase3_state.get(session_key, {})
    trade_count = session_data.get("trade_count", 0)
    consecutive_losses = session_data.get("consecutive_losses", 0)
    last_entry_time = session_data.get("last_entry_time")

    if trade_count >= max_trades_session:
        no_trade["decision"] = ExecutionDecision(
            attempted=False, placed=False,
            reason=f"phase3: max trades/session ({trade_count}/{max_trades_session})", side=None,
        )
        no_trade["phase3_state_updates"] = state_updates
        return no_trade

    if consecutive_losses >= stop_consec_losses:
        no_trade["decision"] = ExecutionDecision(
            attempted=False, placed=False,
            reason=f"phase3: stopped after {consecutive_losses} consecutive losses", side=None,
        )
        no_trade["phase3_state_updates"] = state_updates
        return no_trade

    if last_entry_time is not None:
        try:
            _lt = datetime.fromisoformat(str(last_entry_time))
            if _lt.tzinfo is None:
                _lt = _lt.replace(tzinfo=timezone.utc)
            elapsed = (now_utc - _lt).total_seconds() / 60.0
            if elapsed < cooldown_min:
                no_trade["decision"] = ExecutionDecision(
                    attempted=False, placed=False,
                    reason=f"phase3: cooldown ({elapsed:.1f}/{cooldown_min}min)", side=None,
                )
                no_trade["phase3_state_updates"] = state_updates
                return no_trade
        except Exception:
            pass

    if store is not None and session_loss_stop_frac > 0:
        try:
            profile_name = getattr(profile, "profile_name", str(profile))
            get_by_exit_date = getattr(store, "get_closed_trades_for_exit_date", None)
            if callable(get_by_exit_date):
                day_trades = get_by_exit_date(profile_name, today_str)
            else:
                day_trades = store.get_trades_for_date(profile_name, today_str)
            session_loss_usd = 0.0
            for t in day_trades:
                et = str(t["entry_type"] or "")
                if not et.startswith("phase3:v14"):
                    continue
                p = t["profit"] if "profit" in t.keys() else None
                if p is not None:
                    try:
                        session_loss_usd += float(p)
                    except Exception:
                        pass
            sizing_basis = _account_sizing_value(adapter, fallback=100000.0)
            loss_limit = -abs(float(sizing_basis) * float(session_loss_stop_frac))
            if session_loss_usd <= loss_limit:
                no_trade["decision"] = ExecutionDecision(
                    attempted=False, placed=False,
                    reason=f"phase3: session loss stop ({session_loss_usd:.2f} <= {loss_limit:.2f})",
                    side=None,
                )
                no_trade["phase3_state_updates"] = state_updates
                return no_trade
        except Exception:
            pass

    concurrent = phase3_state.get("open_trade_count", 0)
    if concurrent >= max_concurrent:
        no_trade["decision"] = ExecutionDecision(
            attempted=False, placed=False,
            reason=f"phase3: max concurrent ({concurrent}/{max_concurrent})", side=None,
        )
        no_trade["phase3_state_updates"] = state_updates
        return no_trade

    if breakout_disable_pips > 0:
        if bool(session_data.get("breakout_blocked", False)):
            no_trade["decision"] = ExecutionDecision(
                attempted=False, placed=False,
                reason="phase3: breakout block active for session", side=None,
            )
            no_trade["phase3_state_updates"] = state_updates
            return no_trade
        try:
            m1t = m1_df.copy()
            m1t["time"] = pd.to_datetime(m1t["time"], utc=True, errors="coerce")
            m1t = m1t.dropna(subset=["time"]).sort_values("time")
            if not m1t.empty:
                ts_now = pd.Timestamp(now_utc)
                if ts_now.tzinfo is None:
                    ts_now = ts_now.tz_localize("UTC")
                else:
                    ts_now = ts_now.tz_convert("UTC")
                if breakout_mode == "rolling":
                    w_start = ts_now - pd.Timedelta(minutes=breakout_window_min)
                    w = m1t[m1t["time"] >= w_start]
                    if not w.empty:
                        rng_pips = (float(w["high"].max()) - float(w["low"].min())) / pip
                        if rng_pips >= breakout_window_pips:
                            session_data["breakout_blocked"] = True
                            session_data["breakout_blocked_reason"] = f"rolling {rng_pips:.1f}p >= {breakout_window_pips:.1f}p"
                            state_updates[session_key] = session_data
                            no_trade["decision"] = ExecutionDecision(
                                attempted=False, placed=False,
                                reason=f"phase3: breakout rolling block ({rng_pips:.1f}p)", side=None,
                            )
                            no_trade["phase3_state_updates"] = state_updates
                            return no_trade
                else:
                    tokyo_start_h = _parse_hhmm_to_hour(v14_config.get("session_start_utc", "16:00"), float(TOKYO_START_UTC))
                    tokyo_start = ts_now.replace(
                        hour=int(tokyo_start_h),
                        minute=int(round((tokyo_start_h - int(tokyo_start_h)) * 60.0)),
                        second=0,
                        microsecond=0,
                    )
                    if tokyo_start > ts_now:
                        tokyo_start -= pd.Timedelta(days=1)
                    sess = m1t[m1t["time"] >= tokyo_start]
                    if not sess.empty:
                        first = sess.iloc[0]
                        open_mid = (float(first["high"]) + float(first["low"])) / 2.0
                        cur_mid = (float(tick.ask) + float(tick.bid)) / 2.0
                        move_pips = abs(cur_mid - open_mid) / pip
                        if move_pips >= breakout_disable_pips:
                            session_data["breakout_blocked"] = True
                            session_data["breakout_blocked_reason"] = f"anchor {move_pips:.1f}p >= {breakout_disable_pips:.1f}p"
                            state_updates[session_key] = session_data
                            no_trade["decision"] = ExecutionDecision(
                                attempted=False, placed=False,
                                reason=f"phase3: breakout anchor block ({move_pips:.1f}p)", side=None,
                            )
                            no_trade["phase3_state_updates"] = state_updates
                            return no_trade
        except Exception:
            pass

    # 9.5) Pending V14 confirmation queue (optional)
    pending_v14_raw = phase3_state.get("pending_v14_signals", [])
    pending_v14: list[dict[str, Any]] = []
    now_ts = pd.Timestamp(now_utc).tz_convert("UTC") if pd.Timestamp(now_utc).tzinfo is not None else pd.Timestamp(now_utc, tz="UTC")
    for s in pending_v14_raw if isinstance(pending_v14_raw, list) else []:
        if not isinstance(s, dict):
            continue
        if str(s.get("session_date", "")) != today_str:
            continue
        exp = s.get("expiry_time")
        if not exp:
            continue
        try:
            exp_ts = pd.Timestamp(exp)
            if exp_ts.tzinfo is None:
                exp_ts = exp_ts.tz_localize("UTC")
            else:
                exp_ts = exp_ts.tz_convert("UTC")
        except Exception:
            continue
        if exp_ts >= now_ts:
            pending_v14.append(s)

    if confirmation_window_bars > 0 and pending_v14:
        ref_df = m5_df if confirmation_type == "m5" else m1_df
        if ref_df is not None and not ref_df.empty and "time" in ref_df.columns:
            ref_bar_ts = pd.Timestamp(ref_df["time"].iloc[-1])
            if ref_bar_ts.tzinfo is None:
                ref_bar_ts = ref_bar_ts.tz_localize("UTC")
            else:
                ref_bar_ts = ref_bar_ts.tz_convert("UTC")
            bar_open = float(ref_df["open"].astype(float).iloc[-1])
            bar_close = float(ref_df["close"].astype(float).iloc[-1])
            for sig in list(pending_v14):
                try:
                    sig_side = str(sig.get("side", "")).lower()
                    if sig_side not in {"buy", "sell"}:
                        continue
                    sig_bar_ts = pd.Timestamp(sig.get("bar_time"))
                    if sig_bar_ts.tzinfo is None:
                        sig_bar_ts = sig_bar_ts.tz_localize("UTC")
                    else:
                        sig_bar_ts = sig_bar_ts.tz_convert("UTC")
                    # Require a closed confirmation bar strictly after signal bar.
                    # Prevents m5 confirmations from reusing the pre-signal bar.
                    if ref_bar_ts <= sig_bar_ts:
                        continue
                    confirmed = (bar_close > bar_open) if sig_side == "buy" else (bar_close < bar_open)
                    if not confirmed:
                        continue
                    sig_sl = float(sig.get("sl_price"))
                    sig_tp1 = float(sig.get("tp1_price"))
                    sig_units = int(sig.get("units"))
                    if sig_units <= 0:
                        continue
                    strategy_tag = "phase3:v14_mean_reversion"
                    comment = f"phase3_integrated:{policy.id}:v14_mean_reversion"
                    try:
                        adapter.place_order(
                            symbol=profile.symbol,
                            side=sig_side,
                            lots=sig_units / 100_000.0,
                            stop_price=round(sig_sl, 3),
                            target_price=round(sig_tp1, 3),
                            comment=comment,
                        )
                    except Exception as e:
                        return {
                            "decision": ExecutionDecision(
                                attempted=True, placed=False,
                                reason=f"phase3: queued order error: {e}", side=sig_side,
                            ),
                            "phase3_state_updates": state_updates,
                            "strategy_tag": strategy_tag,
                        }
                    pending_v14.remove(sig)
                    session_data["trade_count"] = trade_count + 1
                    session_data["last_entry_time"] = now_utc.isoformat()
                    state_updates[session_key] = session_data
                    state_updates["pending_v14_signals"] = pending_v14
                    return {
                        "decision": ExecutionDecision(
                            attempted=True, placed=True,
                            reason=(
                                f"phase3:v14 queued {sig_side.upper()} confirmed "
                                f"confluence={sig.get('score')} combo={sig.get('combo')}"
                            ),
                            side=sig_side,
                        ),
                        "phase3_state_updates": state_updates,
                        "strategy_tag": strategy_tag,
                        "sl_price": sig_sl,
                        "tp1_price": sig_tp1,
                        "units": sig_units,
                        "entry_price": tick.ask if sig_side == "buy" else tick.bid,
                        "sl_pips": abs((tick.ask if sig_side == "buy" else tick.bid) - sig_sl) / pip,
                    }
                except Exception:
                    continue

    # 10) Evaluate confluence for both sides
    close_price = float(m5_close.iloc[-1])
    high_price = float(m5_df["high"].astype(float).iloc[-1])
    low_price = float(m5_df["low"].astype(float).iloc[-1])
    best_side = None
    best_score = 0
    best_combo = ""

    for side in ("buy", "sell"):
        score, combo = evaluate_v14_confluence(
            side, close_price, high_price, low_price, pivots, bb_upper, bb_lower,
            sar_bull, sar_bear, rsi_val, pip,
            zone_tolerance_pips=zone_tolerance_pips,
            rsi_long_entry=rsi_long_entry,
            rsi_short_entry=rsi_short_entry,
            core_gate_use_zone=core_gate_use_zone,
            core_gate_use_bb=core_gate_use_bb,
            core_gate_use_sar=core_gate_use_sar,
            core_gate_use_rsi=core_gate_use_rsi,
        )
        min_conf = min_conf_long if side == "buy" else min_conf_short
        if score >= min_conf and combo not in blocked_combos and score > best_score:
            best_side = side
            best_score = score
            best_combo = combo

    if best_side is None:
        no_trade["decision"] = ExecutionDecision(
            attempted=False, placed=False,
            reason=f"phase3: no qualifying confluence (RSI={rsi_val:.1f})", side=None,
        )
        if confirmation_window_bars > 0:
            state_updates["pending_v14_signals"] = pending_v14
        no_trade["phase3_state_updates"] = state_updates
        return no_trade

    if no_reentry_after_stop_min > 0:
        stop_key = f"last_stopout_time_{best_side}"
        last_stop = session_data.get(stop_key)
        if last_stop:
            try:
                stop_dt = datetime.fromisoformat(str(last_stop))
                if stop_dt.tzinfo is None:
                    stop_dt = stop_dt.replace(tzinfo=timezone.utc)
                elapsed_stop = (now_utc - stop_dt).total_seconds() / 60.0
                if elapsed_stop < float(no_reentry_after_stop_min):
                    no_trade["decision"] = ExecutionDecision(
                        attempted=False,
                        placed=False,
                        reason=f"phase3: no-reentry {best_side} after stop ({elapsed_stop:.1f}/{no_reentry_after_stop_min}m)",
                        side=None,
                    )
                    no_trade["phase3_state_updates"] = state_updates
                    return no_trade
            except Exception:
                pass

    strength_info = compute_v14_signal_strength(
        side=best_side,
        confluence_score=int(best_score),
        close=close_price,
        high=high_price,
        low=low_price,
        pivots=pivots,
        bb_upper=bb_upper,
        bb_lower=bb_lower,
        rsi=rsi_val,
        sar_bullish_flip=sar_bull,
        sar_bearish_flip=sar_bear,
        now_utc=now_utc,
        pip_size=pip,
        sst_cfg=sst_cfg if sst_enabled else {},
    ) if sst_enabled else {"score": 0, "bucket": "moderate", "size_mult": 1.0}
    if sst_enabled and sst_filter_on and int(strength_info.get("score", 0)) < int(sst_min_score):
        no_trade["decision"] = ExecutionDecision(
            attempted=False, placed=False,
            reason=f"phase3: signal-strength below min ({strength_info.get('score', 0)} < {sst_min_score})",
            side=None,
        )
        no_trade["phase3_state_updates"] = state_updates
        return no_trade

    # 11) Compute SL, TP1, lot size
    entry_price = tick.ask if best_side == "buy" else tick.bid
    sl_price = compute_v14_sl(
        best_side,
        entry_price,
        pivots,
        pip,
        sl_buffer_pips=sl_buffer_pips,
        min_sl_pips=min_sl_pips,
        max_sl_pips=max_sl_pips,
    )
    sl_pips = abs(entry_price - sl_price) / pip
    tp1_price = compute_v14_tp1(
        best_side,
        entry_price,
        atr_val,
        pip,
        partial_tp_atr_mult=partial_tp_atr_mult,
        partial_tp_min_pips=partial_tp_min_pips,
        partial_tp_max_pips=partial_tp_max_pips,
    )

    equity = _account_sizing_value(adapter, fallback=100_000.0)

    if v14_config:
        units = compute_v14_units_from_config(
            equity, sl_pips, entry_price, pip, now_utc, v14_config,
        )
    else:
        units = compute_v14_lot_size(equity, sl_pips, entry_price, pip, LEVERAGE)
    if units <= 0:
        no_trade["decision"] = ExecutionDecision(
            attempted=False, placed=False,
            reason=f"phase3: lot size 0 (equity={equity:.0f}, sl_pips={sl_pips:.1f})", side=None,
        )
        no_trade["phase3_state_updates"] = state_updates
        return no_trade

    size_mult = float(strength_info.get("size_mult", 1.0))
    if size_mult > 0 and size_mult != 1.0:
        units = int(max(0, math.floor(float(units) * size_mult)))
    if units <= 0:
        no_trade["decision"] = ExecutionDecision(
            attempted=False, placed=False,
            reason="phase3: lot size 0 after strength multiplier", side=None,
        )
        no_trade["phase3_state_updates"] = state_updates
        return no_trade
    state_updates["last_v14_signal_strength"] = {
        "time": now_utc.isoformat(),
        "side": best_side,
        "confluence_score": int(best_score),
        "confluence_combo": str(best_combo),
        "signal_strength_score": int(strength_info.get("score", 0)),
        "signal_strength_bucket": str(strength_info.get("bucket", "moderate")),
        "signal_strength_mult": float(strength_info.get("size_mult", 1.0)),
    }

    if confirmation_window_bars > 0:
        same_side_pending = any(str(s.get("side", "")).lower() == best_side for s in pending_v14)
        if not same_side_pending:
            window_minutes = confirmation_window_bars if confirmation_type == "m1" else confirmation_window_bars * 5
            pending_v14.append(
                {
                    "side": best_side,
                    "bar_time": now_utc.isoformat(),
                    "expiry_time": (now_utc + pd.Timedelta(minutes=window_minutes)).isoformat(),
                    "session_date": today_str,
                    "sl_price": float(sl_price),
                    "tp1_price": float(tp1_price),
                    "units": int(units),
                    "score": int(best_score),
                    "combo": str(best_combo),
                    "signal_strength_score": int(strength_info.get("score", 0)),
                    "signal_strength_bucket": str(strength_info.get("bucket", "moderate")),
                    "signal_strength_mult": float(strength_info.get("size_mult", 1.0)),
                }
            )
        state_updates["pending_v14_signals"] = pending_v14
        return {
            "decision": ExecutionDecision(
                attempted=False,
                placed=False,
                reason=(
                    f"v14 signal queued {best_side} conf={best_score} combo={best_combo} "
                    f"sst={strength_info.get('score', 0)}({strength_info.get('bucket', 'moderate')})"
                ),
                side=None,
            ),
            "phase3_state_updates": state_updates,
            "strategy_tag": "phase3:v14_pending",
        }

    # 12) Place order
    strategy_tag = "phase3:v14_mean_reversion"
    comment = f"phase3_integrated:{policy.id}:v14_mean_reversion"

    try:
        order_result = adapter.place_order(
            symbol=profile.symbol,
            side=best_side,
            lots=units / 100_000.0,
            stop_price=round(sl_price, 3),
            target_price=round(tp1_price, 3),
            comment=comment,
        )
        placed = True
    except Exception as e:
        return {
            "decision": ExecutionDecision(
                attempted=True, placed=False,
                reason=f"phase3: order error: {e}", side=best_side,
            ),
            "phase3_state_updates": state_updates,
            "strategy_tag": strategy_tag,
        }

    confirmed_fill, confirmed_deal_id = _phase3_order_confirmed(adapter, profile, order_result)
    if not confirmed_fill:
        return {
            "decision": ExecutionDecision(
                attempted=True,
                placed=False,
                reason="phase3: broker order pending/unfilled",
                side=best_side,
                order_retcode=getattr(order_result, "order_retcode", None),
                order_id=getattr(order_result, "order_id", None),
                deal_id=getattr(order_result, "deal_id", None),
                fill_price=getattr(order_result, "fill_price", None),
            ),
            "phase3_state_updates": state_updates,
            "strategy_tag": strategy_tag,
        }

    # Update session state
    session_data["trade_count"] = trade_count + 1
    session_data["last_entry_time"] = now_utc.isoformat()
    state_updates[session_key] = session_data

    return {
        "decision": ExecutionDecision(
            attempted=True, placed=True,
            reason=f"phase3:v14 {best_side.upper()} confluence={best_score} combo={best_combo} "
                   f"SL={sl_price:.3f}({sl_pips:.1f}p) TP1={tp1_price:.3f} units={units} "
                   f"sst={strength_info.get('score', 0)}({strength_info.get('bucket', 'moderate')})",
            side=best_side,
            order_retcode=getattr(order_result, "order_retcode", None),
            order_id=getattr(order_result, "order_id", None),
            deal_id=confirmed_deal_id if confirmed_deal_id is not None else getattr(order_result, "deal_id", None),
            fill_price=getattr(order_result, "fill_price", None),
        ),
        "phase3_state_updates": state_updates,
        "strategy_tag": strategy_tag,
        "sl_price": sl_price,
        "tp1_price": tp1_price,
        "units": units,
        "entry_price": entry_price,
        "sl_pips": sl_pips,
        "pivots": pivots,
        "atr_val": atr_val,
    }


# ===================================================================
#  Exit management
# ===================================================================

def _phase3_position_meta(position, side: str) -> tuple[Optional[int], float, int]:
    if isinstance(position, dict):
        position_id = position.get("id")
        current_units = abs(int(position.get("currentUnits") or 0))
        current_lots = current_units / 100_000.0
    else:
        position_id = getattr(position, "ticket", None)
        current_lots = float(getattr(position, "volume", 0) or 0)
        current_units = int(current_lots * 100_000)
    return position_id, current_lots, current_units


def _close_full(adapter, profile, position_id, current_lots: float, side: str) -> None:
    position_type = 1 if side == "sell" else 0
    adapter.close_position(
        ticket=position_id,
        symbol=profile.symbol,
        volume=current_lots,
        position_type=position_type,
    )


def _manage_london_v2_exit(
    *,
    adapter,
    profile,
    store,
    tick,
    trade_row: dict,
    position,
    london_config: Optional[dict[str, Any]] = None,
) -> dict:
    pip = float(profile.pip_size) if hasattr(profile, "pip_size") else PIP_SIZE
    side = str(trade_row["side"]).lower()
    entry = float(trade_row["entry_price"])
    trade_id = str(trade_row["trade_id"])
    now_utc = datetime.now(timezone.utc)
    position_id, current_lots, _ = _phase3_position_meta(position, side)
    if current_lots <= 0:
        return {"action": "none", "reason": "no units"}

    day_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    ny_open_hour = us_ny_open_utc(pd.Timestamp(now_utc))
    ny_open = day_start + pd.Timedelta(hours=int(ny_open_hour))
    tp_check_price = tick.bid if side == "buy" else tick.ask
    cfg = london_config or {}
    entry_type = str(trade_row.get("entry_type") or "")
    is_setup_d = entry_type.endswith("_d")
    force_close_at_ny_open = bool(cfg.get("hard_close_at_ny_open", LDN_FORCE_CLOSE_AT_NY_OPEN))
    tp1_r = float(cfg.get("d_tp1_r", LDN_D_TP1_R)) if is_setup_d else float(cfg.get("arb_tp1_r", LDN_ARB_TP1_R))
    tp2_r = float(cfg.get("d_tp2_r", LDN_D_TP2_R)) if is_setup_d else float(cfg.get("arb_tp2_r", LDN_ARB_TP2_R))
    tp1_close_pct = float(cfg.get("d_tp1_close_pct", LDN_D_TP1_CLOSE_PCT)) if is_setup_d else float(cfg.get("arb_tp1_close_pct", LDN_ARB_TP1_CLOSE_PCT))
    be_offset_pips = float(cfg.get("d_be_offset_pips", LDN_D_BE_OFFSET_PIPS)) if is_setup_d else float(cfg.get("arb_be_offset_pips", LDN_ARB_BE_OFFSET_PIPS))

    if force_close_at_ny_open and now_utc >= ny_open:
        try:
            _close_full(adapter, profile, position_id, current_lots, side)
            close_pips = ((tp_check_price - entry) / pip) if side == "buy" else ((entry - tp_check_price) / pip)
            return {"action": "session_end_close", "reason": "london_v2 hard close at NY open", "closed_pips_est": float(close_pips)}
        except Exception as e:
            return {"action": "error", "reason": f"london_v2 hard close error: {e}"}

    stop_price = trade_row.get("stop_price")
    if stop_price is None:
        return {"action": "none", "reason": "no stop in trade row"}
    stop_price = float(stop_price)
    r_pips = abs(entry - stop_price) / pip
    tp1_price = entry + (tp1_r * r_pips * pip if side == "buy" else -tp1_r * r_pips * pip)
    tp2_price = entry + (tp2_r * r_pips * pip if side == "buy" else -tp2_r * r_pips * pip)

    tp1_done = int(trade_row.get("tp1_partial_done") or 0) == 1
    reached_tp1 = tp_check_price >= tp1_price if side == "buy" else tp_check_price <= tp1_price
    reached_tp2 = tp_check_price >= tp2_price if side == "buy" else tp_check_price <= tp2_price

    if (not tp1_done) and reached_tp1:
        try:
            position_type = 1 if side == "sell" else 0
            close_lots = current_lots * tp1_close_pct
            adapter.close_position(ticket=position_id, symbol=profile.symbol, volume=close_lots, position_type=position_type)
            be_sl = entry + (be_offset_pips * pip if side == "buy" else -be_offset_pips * pip)
            adapter.update_position_stop_loss(position_id, profile.symbol, round(be_sl, 3))
            store.update_trade(trade_id, {"tp1_partial_done": 1, "breakeven_applied": 1, "breakeven_sl_price": round(be_sl, 5)})
            return {"action": "tp1_partial", "reason": f"london_v2 TP1 partial + BE ({be_sl:.3f})"}
        except Exception as e:
            return {"action": "error", "reason": f"london_v2 TP1 error: {e}"}

    if tp1_done and reached_tp2:
        try:
            _close_full(adapter, profile, position_id, current_lots, side)
            close_pips = float(tp2_r * r_pips)
            return {"action": "tp2_full", "reason": "london_v2 TP2 runner close", "closed_pips_est": close_pips}
        except Exception as e:
            return {"action": "error", "reason": f"london_v2 TP2 close error: {e}"}

    return {"action": "none", "reason": "no london_v2 exit condition met"}


def _manage_v44_exit(
    *,
    adapter,
    profile,
    store,
    tick,
    trade_row: dict,
    position,
    v44_config: Optional[dict[str, Any]] = None,
) -> dict:
    pip = float(profile.pip_size) if hasattr(profile, "pip_size") else PIP_SIZE
    side = str(trade_row["side"]).lower()
    entry = float(trade_row["entry_price"])
    trade_id = str(trade_row["trade_id"])
    now_utc = datetime.now(timezone.utc)
    position_id, current_lots, _ = _phase3_position_meta(position, side)
    if current_lots <= 0:
        return {"action": "none", "reason": "no units"}

    day_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    cfg = v44_config or {}
    _ny_start_hour, ny_end_hour = _resolve_ny_window_hours(now_utc, cfg)
    ny_end = day_start + pd.Timedelta(hours=ny_end_hour)
    tp_check_price = tick.bid if side == "buy" else tick.ask
    trail_mark_price = tick.bid if side == "buy" else tick.ask

    if now_utc >= ny_end:
        try:
            _close_full(adapter, profile, position_id, current_lots, side)
            close_pips = ((tp_check_price - entry) / pip) if side == "buy" else ((entry - tp_check_price) / pip)
            return {"action": "session_end_close", "reason": "v44_ny hard close at session end", "closed_pips_est": float(close_pips)}
        except Exception as e:
            return {"action": "error", "reason": f"v44_ny hard close error: {e}"}

    be_offset_pips = float(cfg.get("be_offset_pips", V44_BE_OFFSET_PIPS))
    strong_tp2_pips = float(cfg.get("strong_tp2_pips", V44_STRONG_TP2_PIPS))
    normal_tp2_pips = float(cfg.get("normal_tp2_pips", 3.0))
    weak_tp2_pips = float(cfg.get("weak_tp2_pips", 2.0))
    strong_tp1_close_pct = float(cfg.get("strong_tp1_close_pct", V44_STRONG_TP1_CLOSE_PCT))
    normal_tp1_close_pct = float(cfg.get("normal_tp1_close_pct", 0.5))
    weak_tp1_close_pct = float(cfg.get("weak_tp1_close_pct", 0.6))
    strong_trail_buffer = float(cfg.get("strong_trail_buffer", V44_STRONG_TRAIL_BUFFER))
    normal_trail_buffer = float(cfg.get("normal_trail_buffer", 3.0))
    weak_trail_buffer = float(cfg.get("weak_trail_buffer", 2.0))
    trail_start_after_tp1_mult = float(cfg.get("trail_start_after_tp1_mult", 0.5))

    tp1_done = int(trade_row.get("tp1_partial_done") or 0) == 1
    entry_type = str(trade_row.get("entry_type") or "")
    is_news = ":news" in entry_type
    if ":weak" in entry_type:
        tp1_close_pct = weak_tp1_close_pct
    elif ":normal" in entry_type:
        tp1_close_pct = normal_tp1_close_pct
    else:
        tp1_close_pct = strong_tp1_close_pct
    tp1_price = trade_row.get("target_price")
    if tp1_price is None:
        tp1_price = entry + (V44_STRONG_TP1_PIPS * pip if side == "buy" else -V44_STRONG_TP1_PIPS * pip)
    tp1_price = float(tp1_price)
    reached_tp1 = tp_check_price >= tp1_price if side == "buy" else tp_check_price <= tp1_price

    if (not tp1_done) and reached_tp1:
        if is_news:
            try:
                _close_full(adapter, profile, position_id, current_lots, side)
                close_pips = ((tp1_price - entry) / pip) if side == "buy" else ((entry - tp1_price) / pip)
                return {"action": "tp1_full", "reason": "v44_ny news-trend TP1 full close", "closed_pips_est": float(close_pips)}
            except Exception as e:
                return {"action": "error", "reason": f"v44_ny news TP1 error: {e}"}
        try:
            position_type = 1 if side == "sell" else 0
            close_lots = current_lots * tp1_close_pct
            adapter.close_position(ticket=position_id, symbol=profile.symbol, volume=close_lots, position_type=position_type)
            be_sl = entry + (be_offset_pips * pip if side == "buy" else -be_offset_pips * pip)
            adapter.update_position_stop_loss(position_id, profile.symbol, round(be_sl, 3))
            store.update_trade(trade_id, {"tp1_partial_done": 1, "breakeven_applied": 1, "breakeven_sl_price": round(be_sl, 5)})
            return {"action": "tp1_partial", "reason": f"v44_ny TP1 partial + BE ({be_sl:.3f})"}
        except Exception as e:
            return {"action": "error", "reason": f"v44_ny TP1 error: {e}"}

    if tp1_done:
        if ":weak" in entry_type:
            tp2_pips = weak_tp2_pips
        elif ":normal" in entry_type:
            tp2_pips = normal_tp2_pips
        else:
            tp2_pips = strong_tp2_pips
        # TP2 fixed target then trail fallback.
        tp2_price = entry + (tp2_pips * pip if side == "buy" else -tp2_pips * pip)
        reached_tp2 = tp_check_price >= tp2_price if side == "buy" else tp_check_price <= tp2_price
        if reached_tp2:
            try:
                _close_full(adapter, profile, position_id, current_lots, side)
                close_pips = float(tp2_pips)
                return {"action": "tp2_full", "reason": f"v44_ny TP2 hit ({tp2_pips:.2f}p)", "closed_pips_est": close_pips}
            except Exception as e:
                return {"action": "error", "reason": f"v44_ny TP2 error: {e}"}

        # Trailing stop fallback (only reaches here if TP2 not yet hit)
        stop_price = trade_row.get("stop_price")
        if stop_price is not None and trail_start_after_tp1_mult > 0:
            try:
                sl_pips = abs(float(entry) - float(stop_price)) / pip
                profit_pips = ((tp_check_price - entry) / pip) if side == "buy" else ((entry - tp_check_price) / pip)
                tp1_pips = abs(float(tp1_price) - float(entry)) / pip
                arm_pips = tp1_pips + trail_start_after_tp1_mult * sl_pips
                if profit_pips < arm_pips:
                    return {"action": "none", "reason": "v44_ny trail not armed yet"}
            except Exception:
                pass
        if ":normal" in entry_type:
            trail_buffer_pips = normal_trail_buffer
        elif ":weak" in entry_type:
            trail_buffer_pips = weak_trail_buffer
        else:
            trail_buffer_pips = strong_trail_buffer
        prev_trail = trade_row.get("breakeven_sl_price")
        prev_trail = float(prev_trail) if prev_trail is not None else None
        if side == "buy":
            new_trail = trail_mark_price - trail_buffer_pips * pip
            if prev_trail is not None:
                new_trail = max(new_trail, prev_trail)
            should_update = prev_trail is None or new_trail > prev_trail
        else:
            new_trail = trail_mark_price + trail_buffer_pips * pip
            if prev_trail is not None:
                new_trail = min(new_trail, prev_trail)
            should_update = prev_trail is None or new_trail < prev_trail
        if should_update:
            try:
                adapter.update_position_stop_loss(position_id, profile.symbol, round(new_trail, 3))
                store.update_trade(trade_id, {"breakeven_sl_price": round(new_trail, 5)})
                return {"action": "trail_update", "reason": f"v44_ny trail -> {new_trail:.3f}"}
            except Exception as e:
                return {"action": "error", "reason": f"v44_ny trail error: {e}"}

    return {"action": "none", "reason": "no v44_ny exit condition met"}

def manage_phase3_exit(
    *,
    adapter,
    profile,
    store,
    tick,
    trade_row: dict,
    position,
    data_by_tf: dict,
    phase3_state: dict,
    sizing_config: Optional[dict[str, Any]] = None,
) -> dict:
    """
    Exit management for Phase 3 trades.

    Routes by strategy tag in entry_type/comment:
    - phase3:v14*
    - phase3:london_v2*
    - phase3:v44*
    Returns dict with action taken info.
    """
    pip = float(profile.pip_size) if hasattr(profile, "pip_size") else PIP_SIZE
    entry = float(trade_row["entry_price"])
    side = str(trade_row["side"]).lower()
    trade_id = str(trade_row["trade_id"])
    entry_type = str(trade_row.get("entry_type", "") or "")
    now_utc = datetime.now(timezone.utc)
    exit_check_price = tick.bid if side == "buy" else tick.ask
    current_spread = tick.ask - tick.bid

    if entry_type.startswith("phase3:london_v2"):
        return _manage_london_v2_exit(
            adapter=adapter,
            profile=profile,
            store=store,
            tick=tick,
            trade_row=trade_row,
            position=position,
            london_config=(sizing_config or {}).get("london_v2", {}),
        )
    if entry_type.startswith("phase3:v44"):
        return _manage_v44_exit(
            adapter=adapter,
            profile=profile,
            store=store,
            tick=tick,
            trade_row=trade_row,
            position=position,
            v44_config=(sizing_config or {}).get("v44_ny", {}),
        )

    v14_cfg = (sizing_config or {}).get("v14", {})
    tp1_close_pct = float(v14_cfg.get("tp1_close_pct", TP1_CLOSE_PCT))
    be_offset_pips = float(v14_cfg.get("breakeven_offset_pips", BE_OFFSET_PIPS))
    trail_activate_profit_pips = float(v14_cfg.get("trail_activate_profit_pips", TRAIL_ACTIVATE_PROFIT_PIPS))
    trail_distance_pips = float(v14_cfg.get("trail_distance_pips", TRAIL_DISTANCE_PIPS))
    time_decay_minutes = int(v14_cfg.get("time_decay_minutes", TIME_DECAY_MINUTES))
    time_decay_profit_cap_pips = float(v14_cfg.get("time_decay_profit_cap_pips", TIME_DECAY_CAP_PIPS))
    min_tp_distance_pips = float(v14_cfg.get("min_tp_distance_pips", MIN_TP_DISTANCE_PIPS))

    if isinstance(position, dict):
        position_id = position.get("id")
        current_units = abs(int(position.get("currentUnits") or 0))
        current_lots = current_units / 100_000.0
    else:
        position_id = getattr(position, "ticket", None)
        current_lots = float(getattr(position, "volume", 0) or 0)
        current_units = int(current_lots * 100_000)

    if current_lots <= 0:
        return {"action": "none", "reason": "no units"}

    position_type = 1 if side == "sell" else 0

    # 1) Session end force close
    session = classify_session(now_utc, sizing_config if isinstance(sizing_config, dict) else None)
    if session != "tokyo":
        try:
            adapter.close_position(
                ticket=position_id,
                symbol=profile.symbol,
                volume=current_lots,
                position_type=position_type,
            )
            close_pips = ((exit_check_price - entry) / pip) if side == "buy" else ((entry - exit_check_price) / pip)
            return {"action": "session_end_close", "reason": f"session ended (now={session})", "closed_pips_est": float(close_pips)}
        except Exception as e:
            return {"action": "error", "reason": f"session end close error: {e}"}

    tp1_done = int(trade_row.get("tp1_partial_done") or 0)

    # 2) Time decay check (backtest parity: only after TP1 and only for small positive profit)
    opened_at = trade_row.get("opened_at") or trade_row.get("timestamp_utc")
    if tp1_done and opened_at is not None:
        try:
            if isinstance(opened_at, str):
                open_time = datetime.fromisoformat(opened_at)
                if open_time.tzinfo is None:
                    open_time = open_time.replace(tzinfo=timezone.utc)
            else:
                open_time = opened_at
            elapsed_min = (now_utc - open_time).total_seconds() / 60.0
            if elapsed_min >= time_decay_minutes:
                profit_pips = ((exit_check_price - entry) / pip) if side == "buy" else ((entry - exit_check_price) / pip)
                if 0.0 <= profit_pips < time_decay_profit_cap_pips:
                    try:
                        adapter.close_position(
                            ticket=position_id,
                            symbol=profile.symbol,
                            volume=current_lots,
                            position_type=position_type,
                        )
                        return {
                            "action": "time_decay_close",
                            "reason": f"time decay {elapsed_min:.0f}min, profit={profit_pips:.1f}p < {time_decay_profit_cap_pips}p",
                            "closed_pips_est": float(profit_pips),
                        }
                    except Exception as e:
                        return {"action": "error", "reason": f"time decay close error: {e}"}
        except Exception:
            pass

    # 3) Hard SL check (software SL in case broker SL didn't trigger)
    check_price = tick.bid if side == "buy" else tick.ask
    stop_price = trade_row.get("stop_price")
    if stop_price is not None:
        stop_price = float(stop_price)
        if side == "buy" and check_price <= stop_price:
            try:
                adapter.close_position(ticket=position_id, symbol=profile.symbol, volume=current_lots, position_type=position_type)
                close_pips = (stop_price - entry) / pip
                return {"action": "hard_sl", "reason": f"hard SL hit at {check_price:.3f}", "closed_pips_est": float(close_pips)}
            except Exception as e:
                return {"action": "error", "reason": f"hard SL close error: {e}"}
        elif side == "sell" and check_price >= stop_price:
            try:
                adapter.close_position(ticket=position_id, symbol=profile.symbol, volume=current_lots, position_type=position_type)
                close_pips = (entry - stop_price) / pip
                return {"action": "hard_sl", "reason": f"hard SL hit at {check_price:.3f}", "closed_pips_est": float(close_pips)}
            except Exception as e:
                return {"action": "error", "reason": f"hard SL close error: {e}"}

    # 4) TP1 partial close
    if not tp1_done:
        target_price = trade_row.get("target_price")
        if target_price is not None:
            target_price = float(target_price)
            reached_buy = exit_check_price >= target_price and side == "buy"
            reached_sell = exit_check_price <= target_price and side == "sell"
            if reached_buy or reached_sell:
                close_lots = current_lots * tp1_close_pct
                try:
                    adapter.close_position(
                        ticket=position_id,
                        symbol=profile.symbol,
                        volume=close_lots,
                        position_type=position_type,
                    )
                    # Move SL to BE + offset
                    be_offset = current_spread + be_offset_pips * pip
                    if side == "buy":
                        be_sl = entry + be_offset
                    else:
                        be_sl = entry - be_offset
                    adapter.update_position_stop_loss(position_id, profile.symbol, round(be_sl, 3))
                    store.update_trade(trade_id, {
                        "tp1_partial_done": 1,
                        "breakeven_applied": 1,
                        "breakeven_sl_price": round(be_sl, 5),
                    })
                    return {
                        "action": "tp1_partial",
                        "reason": f"TP1 {tp1_close_pct*100:.0f}% closed, BE SL={be_sl:.3f}",
                    }
                except Exception as e:
                    return {"action": "error", "reason": f"TP1 close error: {e}"}

    # 5) Trailing stop (after TP1)
    elif tp1_done:
        # TP2 runner target at daily pivot P (with minimum TP floor), then trail as fallback.
        piv = phase3_state.get("pivots") if isinstance(phase3_state, dict) else None
        pivot_p = None
        if isinstance(piv, dict):
            try:
                pivot_p = float(piv.get("P"))
            except Exception:
                pivot_p = None
        if pivot_p is not None and np.isfinite(pivot_p):
            min_tp = float(min_tp_distance_pips) * pip
            if side == "buy":
                tp2_price = max(pivot_p, entry + min_tp)
                reached_tp2 = exit_check_price >= tp2_price
            else:
                tp2_price = min(pivot_p, entry - min_tp)
                reached_tp2 = exit_check_price <= tp2_price
            if reached_tp2:
                try:
                    adapter.close_position(
                        ticket=position_id,
                        symbol=profile.symbol,
                        volume=current_lots,
                        position_type=position_type,
                    )
                    close_pips = ((tp2_price - entry) / pip) if side == "buy" else ((entry - tp2_price) / pip)
                    return {"action": "tp2_full", "reason": f"pivot TP2 hit at {tp2_price:.3f}", "closed_pips_est": float(close_pips)}
                except Exception as e:
                    return {"action": "error", "reason": f"TP2 close error: {e}"}

        profit_pips = ((exit_check_price - entry) / pip) if side == "buy" else ((entry - exit_check_price) / pip)
        if profit_pips >= trail_activate_profit_pips:
            prev_trail = trade_row.get("breakeven_sl_price")
            if prev_trail is not None:
                prev_trail = float(prev_trail)
            if side == "buy":
                new_trail = exit_check_price - trail_distance_pips * pip
                if prev_trail is not None and TRAIL_NEVER_WIDEN:
                    new_trail = max(new_trail, prev_trail)
            else:
                new_trail = exit_check_price + trail_distance_pips * pip
                if prev_trail is not None and TRAIL_NEVER_WIDEN:
                    new_trail = min(new_trail, prev_trail)
            should_update = False
            if side == "buy" and (prev_trail is None or new_trail > prev_trail):
                should_update = True
            elif side == "sell" and (prev_trail is None or new_trail < prev_trail):
                should_update = True
            if should_update:
                try:
                    adapter.update_position_stop_loss(position_id, profile.symbol, round(new_trail, 3))
                    store.update_trade(trade_id, {"breakeven_sl_price": round(new_trail, 5)})
                    return {
                        "action": "trail_update",
                        "reason": f"trail SL -> {new_trail:.3f} (profit={profit_pips:.1f}p)",
                    }
                except Exception as e:
                    return {"action": "error", "reason": f"trail update error: {e}"}

    return {"action": "none", "reason": "no exit condition met"}


# ===================================================================
#  Dashboard helpers
# ===================================================================

def report_phase3_session(now_utc: datetime | None = None, effective_config: Optional[dict[str, Any]] = None) -> dict:
    """Report current session classification."""
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    cfg = effective_config if isinstance(effective_config, dict) else load_phase3_sizing_config()
    session = classify_session(now_utc, cfg)
    return {
        "name": "Active Session",
        "value": session or "none",
        "ok": session is not None,
        "detail": f"UTC {now_utc.strftime('%H:%M')} | day={now_utc.strftime('%A')}",
    }


def report_phase3_strategy(session: str | None) -> dict:
    """Report active strategy for this session."""
    if session == "tokyo":
        return {"name": "Active Strategy", "value": "V14 Mean Reversion", "ok": True, "detail": "Tokyo session"}
    elif session == "london":
        return {"name": "Active Strategy", "value": "London V2 (A+D)", "ok": True, "detail": "London session"}
    elif session == "ny":
        return {"name": "Active Strategy", "value": "V44 NY Momentum", "ok": True, "detail": "NY session"}
    return {"name": "Active Strategy", "value": "none", "ok": False, "detail": "No session"}


def report_phase3_regime(regime: str) -> dict:
    """Report BB width regime."""
    return {
        "name": "V14 Regime",
        "value": regime,
        "ok": regime == "ranging",
        "detail": f"BB({BB_PERIOD}, {BB_STD}) width P{int(BB_WIDTH_RANGING_PCT*100)}",
    }


def report_phase3_adx(adx_val: float, max_adx: float | None = None) -> dict:
    """Report ADX status."""
    limit = float(ADX_MAX if max_adx is None else max_adx)
    return {
        "name": "V14 ADX",
        "value": f"{adx_val:.1f}",
        "ok": adx_val < limit,
        "detail": f"max {limit:.1f}",
    }


def report_phase3_atr(atr_val: float, atr_max: float | None = None) -> dict:
    """Report ATR status."""
    limit = float(ATR_MAX if atr_max is None else atr_max)
    atr_pips = atr_val / PIP_SIZE
    return {
        "name": "V14 ATR",
        "value": f"{atr_pips:.1f}p",
        "ok": atr_val < limit,
        "detail": f"max {limit/PIP_SIZE:.0f}p",
    }


def report_phase3_london_range(
    asian_pips: float,
    is_valid: bool,
    min_pips: float | None = None,
    max_pips: float | None = None,
) -> dict:
    min_limit = float(LDN_ARB_RANGE_MIN_PIPS if min_pips is None else min_pips)
    max_limit = float(LDN_ARB_RANGE_MAX_PIPS if max_pips is None else max_pips)
    return {
        "name": "London Asian Range",
        "value": f"{asian_pips:.1f}p",
        "ok": bool(is_valid),
        "detail": f"valid={min_limit:.0f}-{max_limit:.0f}p",
    }


def report_phase3_london_levels(asian_high: float, asian_low: float) -> dict:
    return {
        "name": "London ARB Levels",
        "value": f"H:{asian_high:.3f} / L:{asian_low:.3f}" if np.isfinite(asian_high) and np.isfinite(asian_low) else "n/a",
        "ok": np.isfinite(asian_high) and np.isfinite(asian_low),
        "detail": f"buffers ±{LDN_ARB_BREAKOUT_BUFFER_PIPS:.1f}p",
    }


def report_phase3_ny_trend(trend: str | None) -> dict:
    return {
        "name": "NY H1 Trend",
        "value": trend or "none",
        "ok": trend in {"up", "down"},
        "detail": f"EMA{V44_H1_EMA_FAST}/{V44_H1_EMA_SLOW}",
    }


def report_phase3_ny_slope(slope_pips_per_bar: float, threshold: float | None = None) -> dict:
    limit = float(V44_STRONG_SLOPE if threshold is None else threshold)
    return {
        "name": "NY M5 Slope",
        "value": f"{slope_pips_per_bar:.2f} p/bar",
        "ok": abs(slope_pips_per_bar) >= limit,
        "detail": f"strong>={limit:.2f}",
    }


def report_phase3_ny_atr_filter(ok: bool, cap: float | None = None, lookback: int | None = None) -> dict:
    cap_v = float(V44_ATR_PCT_CAP if cap is None else cap)
    lookback_v = int(V44_ATR_PCT_LOOKBACK if lookback is None else lookback)
    return {
        "name": "NY ATR PCT Filter",
        "value": "pass" if ok else "blocked",
        "ok": bool(ok),
        "detail": f"cap P{int(cap_v*100)} lookback={lookback_v}",
    }


def report_phase3_last_decision(eval_result: Optional[dict]) -> dict:
    """Report last decision reason so user sees why trade was/wasn't taken."""
    reason = ""
    if eval_result and isinstance(eval_result, dict):
        dec = eval_result.get("decision")
        if dec is not None and getattr(dec, "reason", None):
            reason = str(dec.reason)
    return {
        "name": "Last decision",
        "value": reason or "—",
        "ok": True,
        "detail": "Reason from this poll",
    }


def report_phase3_tokyo_caps(phase3_state: dict, now_utc: datetime) -> list[dict]:
    """Tokyo session caps: trade count, cooldown, consecutive losses, max concurrent."""
    reports = []
    today_str = now_utc.strftime("%Y-%m-%d")
    session_key = f"session_tokyo_{today_str}"
    session_data = phase3_state.get(session_key, {})
    trade_count = session_data.get("trade_count", 0)
    consecutive_losses = session_data.get("consecutive_losses", 0)
    last_entry = session_data.get("last_entry_time")
    open_count = int(phase3_state.get("open_trade_count", 0))

    reports.append({
        "name": "V14 Trades this session",
        "value": f"{trade_count}/{MAX_TRADES_PER_SESSION}",
        "ok": trade_count < MAX_TRADES_PER_SESSION,
        "detail": f"max {MAX_TRADES_PER_SESSION}",
    })
    reports.append({
        "name": "V14 Consecutive losses",
        "value": f"{consecutive_losses}/{STOP_AFTER_CONSECUTIVE_LOSSES}",
        "ok": consecutive_losses < STOP_AFTER_CONSECUTIVE_LOSSES,
        "detail": f"stop after {STOP_AFTER_CONSECUTIVE_LOSSES}",
    })
    reports.append({
        "name": "V14 Max concurrent",
        "value": f"{open_count}/{MAX_CONCURRENT}",
        "ok": open_count < MAX_CONCURRENT,
        "detail": "Phase 3 open positions",
    })
    cooldown_ok = True
    if last_entry:
        try:
            from datetime import datetime as dt
            t = dt.fromisoformat(last_entry)
            if t.tzinfo is None:
                import pandas as pd
                t = pd.Timestamp(t).tz_localize("UTC").to_pydatetime()
            elapsed = (now_utc - t).total_seconds() / 60.0
            cooldown_ok = elapsed >= COOLDOWN_MINUTES
        except Exception:
            pass
    reports.append({
        "name": "V14 Cooldown",
        "value": "clear" if cooldown_ok else "active",
        "ok": cooldown_ok,
        "detail": f"{COOLDOWN_MINUTES} min between entries",
    })
    return reports


def report_phase3_london_window(now_utc: datetime) -> dict:
    """Report which London sub-window we're in (A vs D)."""
    windows = _compute_session_windows(now_utc)
    fmt = lambda t: t.strftime("%H:%M")
    if now_utc < windows["london_open"]:
        return {"name": "London Window", "value": "pre-open", "ok": False, "detail": f"Before {fmt(windows['london_open'])} UTC"}
    if now_utc < windows["london_arb_end"]:
        return {
            "name": "London Window",
            "value": f"Setup A ({fmt(windows['london_open'])}–{fmt(windows['london_arb_end'])} UTC)",
            "ok": True,
            "detail": "Asian range breakout",
        }
    if now_utc < windows["london_end"]:
        d_start = windows["london_open"] + pd.Timedelta(minutes=15)
        d_end = windows["london_open"] + pd.Timedelta(minutes=120)
        return {
            "name": "London Window",
            "value": f"Setup D ({fmt(d_start)}–{fmt(d_end)} UTC active)",
            "ok": True,
            "detail": "LOR breakout",
        }
    return {"name": "London Window", "value": "closed", "ok": False, "detail": f"After {fmt(windows['london_end'])} UTC"}


def report_phase3_london_caps(phase3_state: dict) -> list[dict]:
    """London session caps: max open, A/D trade counts."""
    reports = []
    today = datetime.now(timezone.utc).date().isoformat()
    session_key = f"session_london_{today}"
    sdat = phase3_state.get(session_key, {})
    arb = int(sdat.get("arb_trades", 0))
    d_trades = int(sdat.get("d_trades", 0))
    total_trades = int(sdat.get("total_trades", 0))
    open_count = int(phase3_state.get("open_trade_count", 0))

    reports.append({
        "name": "London Max open",
        "value": f"{open_count}/{LDN_MAX_OPEN}",
        "ok": open_count < LDN_MAX_OPEN,
        "detail": "Phase 3 open positions",
    })
    reports.append({
        "name": "London ARB trades",
        "value": f"{arb}/{LDN_ARB_MAX_TRADES}",
        "ok": arb < LDN_ARB_MAX_TRADES,
        "detail": "Asian range breakout",
    })
    reports.append({
        "name": "London D trades",
        "value": f"{d_trades}/{LDN_D_MAX_TRADES}",
        "ok": d_trades < LDN_D_MAX_TRADES,
        "detail": "LOR breakout (long-only)",
    })
    reports.append({
        "name": "London total trades",
        "value": f"{total_trades}/{LDN_MAX_TRADES_PER_DAY_TOTAL}",
        "ok": total_trades < LDN_MAX_TRADES_PER_DAY_TOTAL,
        "detail": "Daily cap",
    })
    return reports


def report_phase3_ny_caps(phase3_state: dict, now_utc: datetime) -> list[dict]:
    """NY session caps: max open, session stop losses, cooldown."""
    reports = []
    today = now_utc.date().isoformat()
    session_key = f"session_ny_{today}"
    sdat = phase3_state.get(session_key, {})
    trade_count = int(sdat.get("trade_count", 0))
    consecutive_losses = int(sdat.get("consecutive_losses", 0))
    open_count = int(phase3_state.get("open_trade_count", 0))

    reports.append({
        "name": "NY Max open",
        "value": f"{open_count}/{V44_MAX_OPEN}",
        "ok": open_count < V44_MAX_OPEN,
        "detail": "Phase 3 open positions",
    })
    reports.append({
        "name": "NY Session stop losses",
        "value": f"{consecutive_losses}/{V44_SESSION_STOP_LOSSES}",
        "ok": consecutive_losses < V44_SESSION_STOP_LOSSES,
        "detail": f"stop after {V44_SESSION_STOP_LOSSES}",
    })
    return reports
