#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


PIP_SIZE = 0.01


DEFAULTS: dict[str, object] = {
    "inputs": [],
    "out": "research_out/session_momentum_50k.json",
    "mode": "session",
    "version": "v1",
    "base_lot": 0.1,
    "spread_mode": "fixed",
    "spread_pips": 2.0,
    "spread_min_pips": 1.0,
    "spread_max_pips": 3.0,
    "max_entry_spread_pips": 10.0,
    "h1_ema_fast": 20,
    "h1_ema_slow": 50,
    "h1_allow_slope_direction": False,
    "h1_slope_bars": 6,
    "london_start": 7.0,
    "london_end": 11.0,
    "ny_start": 13.0,
    "ny_end": 16.0,
    "sessions": "both",
    "mode1_only": False,
    "max_open_positions": 1,
    "max_entries_per_day": 3,
    "loss_after_first_full_sl_lot_mult": 1.0,
    "mode1_range_minutes": 30,
    "range_min_pips": 5.0,
    "range_max_pips": 20.0,
    "breakout_margin_pips": 1.0,
    "sl_floor_pips": 4.0,
    "sl_buffer_pips": 1.5,
    "sl_mode1_max_pips": 15.0,
    "mode2_impulse_lookback_bars": 10,
    "mode2_impulse_min_body_pips": 3.0,
    "mode2_pullback_lookback_bars": 10,
    "mode2_cooldown_bars": 6,
    "sl_mode2_lookback_bars": 5,
    "sl_mode2_max_pips": 12.0,
    "tp1_multiplier": 1.5,
    "tp2_multiplier": 2.5,
    "tp1_close_fraction": 0.5,
    "be_offset_pips": 1.0,
    "trail_buffer_pips": 3.0,
    # v2 defaults
    "v2_max_entries_per_day": 4,
    "v2_m5_ema_fast": 9,
    "v2_m5_ema_slow": 21,
    "v2_impulse_lookback_bars": 8,
    "v2_impulse_min_body_pips": 3.0,
    "v2_pullback_lookback_bars": 8,
    "v2_m1_ema_fast": 13,
    "v2_m1_ema_slow": 34,
    "v2_cooldown_win_bars": 2,
    "v2_cooldown_loss_bars": 8,
    "v2_cooldown_scratch_bars": 4,
    "v2_scratch_threshold_pips": 1.0,
    "v2_sl_source": "m5",
    "v2_sl_lookback_bars": 10,
    "v2_sl_lookback_m5_bars": 5,
    "v2_sl_buffer_pips": 1.5,
    "v2_sl_max_pips": 12.0,
    "v2_sl_floor_pips": 4.0,
    "v2_tp1_multiplier": 1.5,
    "v2_tp2_multiplier": 3.0,
    "v2_tp1_close_pct": 0.5,
    "v2_be_offset_pips": 1.0,
    "v2_trail_buffer_pips": 3.0,
    "v2_trail_source": "m5",
    "v2_no_reentry_same_m5_after_loss": True,
    # v3 defaults
    "v3_atr_period": 14,
    "v3_atr_sma_period": 50,
    "v3_atr_high_ratio": 1.2,
    "v3_atr_low_ratio": 0.8,
    "v3_slope_bars": 6,
    "v3_slope_trending_threshold": 0.5,
    "v3_momentum_cooldown_win": 1,
    "v3_grind_cooldown_win": 3,
    "v3_momentum_cooldown_loss": 4,
    "v3_grind_cooldown_loss": 8,
    "v3_momentum_cooldown_scratch": 2,
    "v3_grind_cooldown_scratch": 4,
    "v3_scratch_threshold_pips": 1.0,
    "v3_momentum_tp1_mult": 2.0,
    "v3_grind_tp1_mult": 1.5,
    "v3_momentum_tp2_mult": 4.0,
    "v3_grind_tp2_mult": 3.0,
    "v3_momentum_trail_buffer": 4.0,
    "v3_grind_trail_buffer": 3.0,
    "v3_momentum_sl_max": 12.0,
    "v3_grind_sl_max": 10.0,
    "v3_sl_floor": 4.0,
    "v3_session_feedback": True,
    "v3_burst_lookback": 8,
    "v3_burst_min_bars": 3,
    "v3_doji_pips": 0.3,
    "v3_momentum_allow_simple_entry": True,
    "v3_m5_ema_fast": 9,
    "v3_m5_ema_slow": 21,
    "v3_impulse_lookback_bars": 8,
    "v3_impulse_min_body_pips": 3.0,
    "v3_pullback_lookback_bars": 8,
    "v3_m1_ema_fast": 13,
    "v3_m1_ema_slow": 34,
    "v3_sl_lookback_m5_bars": 5,
    "v3_sl_buffer_pips": 1.5,
    "v3_tp1_close_pct": 0.5,
    "v3_be_offset_pips": 1.0,
    "v3_adaptive_lots": False,
    "v3_base_lot": 0.1,
    "v3_lot_momentum_mult": 1.5,
    "v3_lot_grind_mult": 1.0,
    "v3_lot_hot_mult": 2.0,
    "v3_max_entries_per_day": 5,
    "v3_london_in_momentum": False,
    # v4 defaults
    "v4_sessions": "both",
    "v4_max_open": 2,
    "v4_max_entries_day": 7,
    "v4_m5_ema_fast": 9,
    "v4_m5_ema_slow": 21,
    "v4_m1_ema_fast": 8,
    "v4_m1_ema_slow": 21,
    "v4_probe_lot": 0.1,
    "v4_press_lot": 0.1,
    "v4_recovery_lot": 0.05,
    "v4_sl_lookback": 6,
    "v4_sl_buffer": 1.0,
    "v4_sl_max": 10.0,
    "v4_sl_floor": 3.0,
    "v4_tp1_mult": 1.5,
    "v4_tp2_mult": 3.0,
    "v4_tp1_close_pct": 0.5,
    "v4_be_offset": 0.5,
    "v4_trail_buffer": 2.0,
    "v4_cooldown_win": 1,
    "v4_cooldown_loss": 3,
    "v4_cooldown_scratch": 2,
    "v4_scratch_threshold": 1.0,
    "v4_session_stop_losses": 3,
    "v4_close_full_risk_at_session_end": True,
    # v5 defaults
    "v5_sessions": "both",
    "v5_max_entries_day": 7,
    "v5_max_open": 1,
    "v5_m5_ema_fast": 9,
    "v5_m5_ema_slow": 21,
    "v5_h4_adx_min": 0.0,
    "v5_h1_slope_min": 0.0,
    "v5_h1_slope_consistent_bars": 0,
    "v5_session_range_cap_pips": 0.0,
    "v5_adx_min": 0.0,
    "v5_tp_atr_scale": 0.0,
    "v5_sl_atr_scale": 0.0,
    "v5_slope_bars": 6,
    "v5_strong_slope": 0.6,
    "v5_weak_slope": 0.2,
    "v5_skip_weak": True,
    "v5_strength_allow": "strong_normal",
    "v5_ny_strength_allow": "strong_normal",
    "v5_london_strength_allow": "strong_only",
    "v5_london_strong_slope": 0.8,
    "v5_london_max_entries": 2,
    "v5_london_confirm_bars": 2,
    "v5_london_min_body_pips": 1.5,
    "v5_tokyo_enabled": False,
    "v5_tokyo_start": 0.0,
    "v5_tokyo_end": 3.0,
    "v5_tokyo_strong_slope": 0.3,
    "v5_tokyo_sl_floor_pips": 3.5,
    "v5_tokyo_sl_cap_pips": 6.0,
    "v5_tokyo_tp1_mult": 1.5,
    "v5_tokyo_tp2_mult": 3.0,
    "v5_tokyo_tp1_close_pct": 0.5,
    "v5_tokyo_cutoff_minutes": 30,
    "v5_tokyo_max_entries": 2,
    "v5_tokyo_strength_allow": "strong_only",
    "v5_rp_tokyo_mult": 0.8,
    "v5_tokyo_trail_buffer": 2.0,
    "v5_tokyo_confirm_bars": 1,
    "v5_tokyo_require_h4_trend": False,
    "v5_allow_normal_plus": False,
    "v5_normalplus_atr_min_pips": 6.0,
    "v5_normalplus_slope_min": 0.45,
    "v5_skip_normal": False,
    "v5_entry_min_body_pips": 1.0,
    "v5_session_entry_cutoff_minutes": 45,
    "v5_trail_start_after_tp1_mult": 0.5,
    "v5_london_start_hour": 6.5,
    "v5_london_active_start": 7.5,
    "v5_london_active_end": 10.0,
    "v5_m1_ema_fast": 8,
    "v5_m1_ema_slow": 21,
    "v5_base_lot": 2.0,
    "v5_london_size_mult": 0.5,
    "v5_weak_size_mult": 0.0,
    "v5_normal_size_mult": 2.0,
    "v5_strong_size_mult": 3.0,
    "v5_win_bonus_per_step": 1.0,
    "v5_win_streak_scope": "session",
    "v5_max_lot": 10.0,
    "v5_daily_loss_limit_pips": 20.0,
    "v5_weekly_loss_limit_pips": 40.0,
    "v5_ny_start_delay_minutes": 5,
    "v5_strong_tp1": 2.0,
    "v5_strong_tp2": 4.0,
    "v5_london_strong_tp1": 1.5,
    "v5_london_strong_tp2": 2.5,
    "v5_london_tp1_close_pct": 0.5,
    "v5_normal_tp1": 1.5,
    "v5_normal_tp2": 2.5,
    "v5_weak_tp1": 1.2,
    "v5_weak_tp2": 2.0,
    "v5_strong_trail_buffer": 6.0,
    "v5_strong_trail_buffer_pips": 6.0,
    "v5_london_trail_buffer": 4.0,
    "v5_strong_trail_start_threshold": 0.5,
    "v5_normal_trail_buffer": 3.0,
    "v5_weak_trail_buffer": 2.0,
    "v5_strong_tp1_close_pct": 0.3,
    "v5_normal_tp1_close_pct": 0.5,
    "v5_weak_tp1_close_pct": 0.6,
    "v5_strong_trail_ema": 21,
    "v5_normal_trail_ema": 9,
    "v5_be_offset": 0.5,
    "v5_sl_lookback": 6,
    "v5_sl_buffer": 1.5,
    "v5_sl_max": 10.0,
    "v5_sl_floor": 3.0,
    "v5_cooldown_win": 1,
    "v5_cooldown_loss": 3,
    "v5_cooldown_scratch": 2,
    "v5_scratch_threshold": 1.0,
    "v5_session_stop_losses": 3,
    "v5_close_full_risk_at_session_end": True,
    "v5_sizing_mode": "risk_parity",
    "v5_risk_per_trade_pct": 0.5,
    "v5_account_size": 100000.0,
    "v5_rp_london_mult": 0.7,
    "v5_rp_ny_mult": 1.0,
    "v5_rp_strong_mult": 1.0,
    "v5_rp_normal_mult": 0.6,
    "v5_rp_win_bonus_pct": 25.0,
    "v5_rp_max_lot_mult": 2.0,
    "v5_rp_min_lot": 1.0,
    "v5_rp_max_lot": 20.0,
    "v5_daily_loss_limit_usd": 1500.0,
    "v5_stale_exit_enabled": True,
    "v5_stale_exit_bars": 25,
    "v5_stale_exit_underwater_pct": 0.5,
    "v5_breakout_entry_enabled": False,
    "v5_breakout_min_body_pips": 2.0,
    "v5_breakout_slope_min": 0.3,
    "v5_breakout_sl_lookback": 1,
    "v5_atr_min_entry_pips": 0.0,
    "v5_rolling_wr_lookback": 10,
    "v5_rolling_wr_min": 0.0,
    "v5_rolling_wr_pause_bars": 48,
    "v5_wr_scale_lookback": 20,
    "v5_wr_scale_min_trades": 10,
    "v5_wr_scale_high": 0.40,
    "v5_wr_scale_mid": 0.33,
    "v5_wr_scale_low": 0.25,
    "v5_wr_scale_full": 1.0,
    "v5_wr_scale_reduced": 0.6,
    "v5_wr_scale_minimal": 0.35,
    "v5_wr_scale_survival": 0.2,
    "v5_wr_scale_boost": 1.0,
    "v5_wr_scale_boost_min": 0.50,
    "v5_max_open_boost": 2,
    "v5_wr_open_boost_min": 0.50,
    "v5_conviction_h1_sep_pips": 0.0,
    "v5_conviction_size_mult": 1.0,
    "v5_conviction_wr_gate": 1.0,
    "v5_ny_normal_conviction_override": False,
    "v5_london_orb_enabled": False,
    "v5_london_orb_fallback_pullback": False,
    "v5_orb_min_range_pips": 5.0,
    "v5_orb_max_range_pips": 25.0,
    "v5_orb_tp1_mult": 1.0,
    "v5_orb_tp2_mult": 2.0,
    "v5_orb_sl_buffer_pips": 2.0,
    "v5_orb_sl_cap_pips": 999.0,
    "v5_orb_min_body_pips": 1.5,
    "v5_dual_mode_enabled": False,
    "v5_range_fade_enabled": True,
    "v5_trend_mode_efficiency_min": 0.40,
    "v5_range_mode_efficiency_max": 0.30,
    "v5_range_fade_pips": 15.0,
    "v5_range_fade_sl_pips": 8.0,
    "v5_range_fade_tp_pips": 10.0,
    "v5_skip_months": "",
    "v5_thin_month": 0,
    "v5_thin_start_day": 20,
    "v5_thin_scale": 0.25,
    "v5_hybrid_strong_boost": 1.3,
    "v5_hybrid_normal_boost": 0.8,
    "v5_hybrid_london_boost": 0.7,
    "v5_hybrid_ny_boost": 1.0,
    "v5_sl_floor_pips": 5.0,
    "v5_sl_cap_pips": 9.0,
    # swing defaults
    "swing_mode_enabled": False,
    "swing_weekly_ema": 10,
    "swing_daily_ema_fast": 20,
    "swing_daily_ema_slow": 50,
    "swing_h4_ema": 21,
    "swing_h4_ema_slow": 50,
    "swing_atr_expansion_ratio": 0.9,
    "swing_sl_h4_lookback": 3,
    "swing_sl_buffer_pips": 5.0,
    "swing_sl_cap_pips": 50.0,
    "swing_tp1_rr": 2.0,
    "swing_tp2_rr": 4.0,
    "swing_tp1_close_pct": 0.4,
    "swing_max_duration_days": 5,
    "swing_risk_pct": 0.75,
    "swing_account_size": 100000.0,
}


@dataclass
class OpenPosition:
    trade_id: int
    side: str
    entry_mode: int
    entry_session: str
    entry_time: pd.Timestamp
    entry_day: str
    entry_index_in_day: int
    entry_price: float
    lots_initial: float
    lots_remaining: float
    stop_price: float
    tp1_price: float
    tp2_price: float
    sl_pips: float
    tp1_pips: float
    tp2_pips: float
    tp1_filled: bool
    realized_pips: float
    realized_usd: float
    exit_price_last: Optional[float]
    entry_regime: Optional[str] = None
    entry_profile: Optional[str] = None
    position_type: Optional[str] = None
    trail_buffer_pips: float = 0.0
    trail_ema_period: int = 9
    tp1_close_fraction: float = 0.5
    trail_start_multiple: float = 0.0
    trail_armed: bool = True
    trail_delay_observed: bool = False
    entry_bar_index: int = 0
    entry_signal_mode: Optional[str] = None
    wr_size_scale: float = 1.0
    conviction_scale: float = 1.0


@dataclass
class ClosedTrade:
    trade_id: int
    side: str
    entry_mode: int
    entry_session: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    lots: float
    sl_pips: float
    tp1_pips: float
    tp2_pips: float
    pips: float
    usd: float
    exit_reason: str
    entry_day: str
    entry_regime: Optional[str] = None
    entry_profile: Optional[str] = None
    position_type: Optional[str] = None
    entry_signal_mode: Optional[str] = None
    wr_size_scale: float = 1.0
    conviction_scale: float = 1.0


def _preparse_config(argv: list[str]) -> Optional[str]:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", type=str)
    ns, _ = p.parse_known_args(argv)
    return ns.config


def parse_bool(v: object) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {v}")


def _full_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Session Momentum backtest for USDJPY on M1 CSV.")
    p.add_argument("--mode", choices=["session", "swing", "combined"], help="Backtest engine mode")
    p.add_argument("--config", type=str, help="Optional JSON config; CLI args override config values")
    p.add_argument("--in", dest="inputs", action="append", help="Input M1 CSV path (repeatable)")
    p.add_argument("--out", type=str, help="Output JSON path")
    p.add_argument("--output", dest="out", type=str, help="Output JSON path (alias)")
    p.add_argument("--version", choices=["v1", "v2", "v3", "v4", "v5"], help="Backtest version selector")

    p.add_argument("--base-lot", type=float)
    p.add_argument("--spread-mode", choices=["fixed", "variable", "realistic"])
    p.add_argument("--spread-pips", type=float)
    p.add_argument("--spread-min-pips", type=float)
    p.add_argument("--spread-max-pips", type=float)
    p.add_argument("--max-entry-spread-pips", type=float)

    p.add_argument("--h1-ema-fast", type=int)
    p.add_argument("--h1-ema-slow", type=int)
    p.add_argument("--h1-allow-slope-direction", type=parse_bool, nargs="?", const=True)
    p.add_argument("--h1-slope-bars", type=int)

    p.add_argument("--london-start", type=float)
    p.add_argument("--london-end", type=float)
    p.add_argument("--ny-start", type=float)
    p.add_argument("--ny-end", type=float)
    p.add_argument("--sessions", choices=["both", "ny_only", "london_only"], help="Limit where new entries are allowed")
    p.add_argument("--mode1-only", action="store_true", help="Only Mode 1 can open new trades")

    p.add_argument("--max-open-positions", type=int)
    p.add_argument("--max-entries-per-day", type=int)
    p.add_argument("--loss-after-first-full-sl-lot-mult", type=float)

    p.add_argument("--mode1-range-minutes", type=int)
    p.add_argument("--range-min-pips", type=float)
    p.add_argument("--range-max-pips", type=float)
    p.add_argument("--breakout-margin-pips", type=float)

    p.add_argument("--sl-floor-pips", type=float)
    p.add_argument("--sl-buffer-pips", type=float)
    p.add_argument("--sl-mode1-max-pips", type=float)

    p.add_argument("--mode2-impulse-lookback-bars", type=int)
    p.add_argument("--mode2-impulse-min-body-pips", type=float)
    p.add_argument("--mode2-pullback-lookback-bars", type=int)
    p.add_argument("--mode2-cooldown-bars", type=int)
    p.add_argument("--sl-mode2-lookback-bars", type=int)
    p.add_argument("--sl-mode2-max-pips", type=float)

    p.add_argument("--tp1-multiplier", type=float)
    p.add_argument("--tp2-multiplier", type=float)
    p.add_argument("--tp1-close-fraction", type=float)
    p.add_argument("--be-offset-pips", type=float)
    p.add_argument("--trail-buffer-pips", type=float)

    # v2 parameters
    p.add_argument("--v2-max-entries-per-day", type=int)
    p.add_argument("--v2-m5-ema-fast", type=int)
    p.add_argument("--v2-m5-ema-slow", type=int)
    p.add_argument("--v2-impulse-lookback-bars", type=int)
    p.add_argument("--v2-impulse-min-body-pips", type=float)
    p.add_argument("--v2-pullback-lookback-bars", type=int)
    p.add_argument("--v2-m1-ema-fast", type=int)
    p.add_argument("--v2-m1-ema-slow", type=int)
    p.add_argument("--v2-cooldown-win-bars", type=int)
    p.add_argument("--v2-cooldown-loss-bars", type=int)
    p.add_argument("--v2-cooldown-scratch-bars", type=int)
    p.add_argument("--v2-scratch-threshold-pips", type=float)
    p.add_argument("--v2-sl-source", choices=["m1", "m5"], type=str)
    p.add_argument("--v2-sl-lookback-bars", type=int)
    p.add_argument("--v2-sl-lookback-m5-bars", type=int)
    p.add_argument("--v2-sl-buffer-pips", type=float)
    p.add_argument("--v2-sl-max-pips", type=float)
    p.add_argument("--v2-sl-floor-pips", type=float)
    p.add_argument("--v2-tp1-multiplier", type=float)
    p.add_argument("--v2-tp2-multiplier", type=float)
    p.add_argument("--v2-tp1-close-pct", type=float)
    p.add_argument("--v2-be-offset-pips", type=float)
    p.add_argument("--v2-trail-buffer-pips", type=float)
    p.add_argument("--v2-trail-source", choices=["m1", "m5"], type=str)
    p.add_argument(
        "--v2-no-reentry-same-m5-after-loss",
        type=parse_bool,
        nargs="?",
        const=True,
        help="Block re-entry on same M5 bar after an SL loss (true/false)",
    )

    # v3 parameters
    p.add_argument("--v3-atr-period", type=int)
    p.add_argument("--v3-atr-sma-period", type=int)
    p.add_argument("--v3-atr-high-ratio", type=float)
    p.add_argument("--v3-atr-low-ratio", type=float)
    p.add_argument("--v3-slope-bars", type=int)
    p.add_argument("--v3-slope-trending-threshold", type=float)
    p.add_argument("--v3-momentum-cooldown-win", type=int)
    p.add_argument("--v3-grind-cooldown-win", type=int)
    p.add_argument("--v3-momentum-cooldown-loss", type=int)
    p.add_argument("--v3-grind-cooldown-loss", type=int)
    p.add_argument("--v3-momentum-cooldown-scratch", type=int)
    p.add_argument("--v3-grind-cooldown-scratch", type=int)
    p.add_argument("--v3-scratch-threshold-pips", type=float)
    p.add_argument("--v3-momentum-tp1-mult", type=float)
    p.add_argument("--v3-grind-tp1-mult", type=float)
    p.add_argument("--v3-momentum-tp2-mult", type=float)
    p.add_argument("--v3-grind-tp2-mult", type=float)
    p.add_argument("--v3-momentum-trail-buffer", type=float)
    p.add_argument("--v3-grind-trail-buffer", type=float)
    p.add_argument("--v3-momentum-sl-max", type=float)
    p.add_argument("--v3-grind-sl-max", type=float)
    p.add_argument("--v3-sl-floor", type=float)
    p.add_argument("--v3-session-feedback", type=parse_bool, nargs="?", const=True)
    p.add_argument("--v3-burst-lookback", type=int)
    p.add_argument("--v3-burst-min-bars", type=int)
    p.add_argument("--v3-doji-pips", type=float)
    p.add_argument("--v3-momentum-allow-simple-entry", type=parse_bool, nargs="?", const=True)
    p.add_argument("--v3-m5-ema-fast", type=int)
    p.add_argument("--v3-m5-ema-slow", type=int)
    p.add_argument("--v3-impulse-lookback-bars", type=int)
    p.add_argument("--v3-impulse-min-body-pips", type=float)
    p.add_argument("--v3-pullback-lookback-bars", type=int)
    p.add_argument("--v3-m1-ema-fast", type=int)
    p.add_argument("--v3-m1-ema-slow", type=int)
    p.add_argument("--v3-sl-lookback-m5-bars", type=int)
    p.add_argument("--v3-sl-buffer-pips", type=float)
    p.add_argument("--v3-tp1-close-pct", type=float)
    p.add_argument("--v3-be-offset-pips", type=float)
    p.add_argument("--v3-adaptive-lots", type=parse_bool, nargs="?", const=True)
    p.add_argument("--v3-base-lot", type=float)
    p.add_argument("--v3-lot-momentum-mult", type=float)
    p.add_argument("--v3-lot-grind-mult", type=float)
    p.add_argument("--v3-lot-hot-mult", type=float)
    p.add_argument("--v3-max-entries-per-day", type=int)
    p.add_argument("--v3-london-in-momentum", type=parse_bool, nargs="?", const=True)

    # v4 parameters
    p.add_argument("--v4-sessions", choices=["both", "ny_only", "london_only"], type=str)
    p.add_argument("--v4-max-open", type=int)
    p.add_argument("--v4-max-entries-day", type=int)
    p.add_argument("--v4-m5-ema-fast", type=int)
    p.add_argument("--v4-m5-ema-slow", type=int)
    p.add_argument("--v4-m1-ema-fast", type=int)
    p.add_argument("--v4-m1-ema-slow", type=int)
    p.add_argument("--v4-probe-lot", type=float)
    p.add_argument("--v4-press-lot", type=float)
    p.add_argument("--v4-recovery-lot", type=float)
    p.add_argument("--v4-sl-lookback", type=int)
    p.add_argument("--v4-sl-buffer", type=float)
    p.add_argument("--v4-sl-max", type=float)
    p.add_argument("--v4-sl-floor", type=float)
    p.add_argument("--v4-tp1-mult", type=float)
    p.add_argument("--v4-tp2-mult", type=float)
    p.add_argument("--v4-tp1-close-pct", type=float)
    p.add_argument("--v4-be-offset", type=float)
    p.add_argument("--v4-trail-buffer", type=float)
    p.add_argument("--v4-cooldown-win", type=int)
    p.add_argument("--v4-cooldown-loss", type=int)
    p.add_argument("--v4-cooldown-scratch", type=int)
    p.add_argument("--v4-scratch-threshold", type=float)
    p.add_argument("--v4-session-stop-losses", type=int)
    p.add_argument("--v4-close-full-risk-at-session-end", type=parse_bool, nargs="?", const=True)

    # v5 parameters
    p.add_argument("--v5-sessions", choices=["both", "ny_only"], type=str)
    p.add_argument("--v5-max-entries-day", type=int)
    p.add_argument("--v5-max-open", type=int)
    p.add_argument("--v5-m5-ema-fast", type=int)
    p.add_argument("--v5-m5-ema-slow", type=int)
    p.add_argument("--v5-h4-adx-min", type=float)
    p.add_argument("--v5-h1-slope-min", type=float)
    p.add_argument("--v5-h1-slope-consistent-bars", type=int)
    p.add_argument("--v5-session-range-cap-pips", type=float)
    p.add_argument("--v5-adx-min", type=float)
    p.add_argument("--v5-tp-atr-scale", type=float)
    p.add_argument("--v5-sl-atr-scale", type=float)
    p.add_argument("--v5-slope-bars", type=int)
    p.add_argument("--v5-strong-slope", type=float)
    p.add_argument("--v5-weak-slope", type=float)
    p.add_argument("--v5-skip-weak", type=parse_bool, nargs="?", const=True)
    p.add_argument("--v5-strength-allow", choices=["all", "strong_only", "strong_normal", "normal_and_strong"], type=str)
    p.add_argument("--v5-ny-strength-allow", choices=["all", "strong_only", "strong_normal", "normal_and_strong"], type=str)
    p.add_argument("--v5-london-strength-allow", choices=["all", "strong_only", "strong_normal", "normal_and_strong"], type=str)
    p.add_argument("--v5-london-strong-slope", type=float)
    p.add_argument("--v5-london-max-entries", type=int)
    p.add_argument("--v5-london-confirm-bars", type=int)
    p.add_argument("--v5-london-min-body-pips", type=float)
    p.add_argument("--v5-tokyo-enabled", type=parse_bool, nargs="?", const=True)
    p.add_argument("--v5-tokyo-start", type=float)
    p.add_argument("--v5-tokyo-end", type=float)
    p.add_argument("--v5-tokyo-strong-slope", type=float)
    p.add_argument("--v5-tokyo-sl-floor-pips", type=float)
    p.add_argument("--v5-tokyo-sl-cap-pips", type=float)
    p.add_argument("--v5-tokyo-tp1-mult", type=float)
    p.add_argument("--v5-tokyo-tp2-mult", type=float)
    p.add_argument("--v5-tokyo-tp1-close-pct", type=float)
    p.add_argument("--v5-tokyo-cutoff-minutes", type=int)
    p.add_argument("--v5-tokyo-max-entries", type=int)
    p.add_argument("--v5-tokyo-strength-allow", type=str, choices=["all", "strong_only", "strong_normal", "normal_and_strong"])
    p.add_argument("--v5-rp-tokyo-mult", type=float)
    p.add_argument("--v5-tokyo-trail-buffer", type=float)
    p.add_argument("--v5-tokyo-confirm-bars", type=int)
    p.add_argument("--v5-tokyo-require-h4-trend", type=parse_bool, nargs="?", const=True)
    p.add_argument("--v5-allow-normal-plus", type=parse_bool, nargs="?", const=True)
    p.add_argument("--v5-normalplus-atr-min-pips", type=float)
    p.add_argument("--v5-normalplus-slope-min", type=float)
    p.add_argument("--v5-skip-normal", type=parse_bool, nargs="?", const=True)
    p.add_argument("--v5-entry-min-body-pips", type=float)
    p.add_argument("--v5-session-entry-cutoff-minutes", type=int)
    p.add_argument("--v5-trail-start-after-tp1-mult", type=float)
    p.add_argument("--v5-london-start-hour", type=float)
    p.add_argument("--v5-london-active-start", type=float)
    p.add_argument("--v5-london-active-end", type=float)
    p.add_argument("--v5-london-allow-strength", choices=["all", "strong_normal", "strong_only"], type=str)
    p.add_argument("--v5-m1-ema-fast", type=int)
    p.add_argument("--v5-m1-ema-slow", type=int)
    p.add_argument("--v5-base-lot", type=float)
    p.add_argument("--v5-london-size-mult", type=float)
    p.add_argument("--v5-weak-size-mult", type=float)
    p.add_argument("--v5-normal-size-mult", type=float)
    p.add_argument("--v5-strong-size-mult", type=float)
    p.add_argument("--v5-win-bonus-per-step", type=float)
    p.add_argument("--v5-win-streak-scope", choices=["day", "session"], type=str)
    p.add_argument("--v5-max-lot", type=float)
    p.add_argument("--v5-daily-loss-limit-pips", type=float)
    p.add_argument("--v5-weekly-loss-limit-pips", type=float)
    p.add_argument("--v5-ny-start-delay-minutes", type=int)
    p.add_argument("--v5-strong-tp1", type=float)
    p.add_argument("--v5-strong-tp2", type=float)
    p.add_argument("--v5-london-strong-tp1", type=float)
    p.add_argument("--v5-london-strong-tp2", type=float)
    p.add_argument("--v5-london-tp1-close-pct", type=float)
    p.add_argument("--v5-normal-tp1", type=float)
    p.add_argument("--v5-normal-tp2", type=float)
    p.add_argument("--v5-weak-tp1", type=float)
    p.add_argument("--v5-weak-tp2", type=float)
    p.add_argument("--v5-strong-trail-buffer", type=float)
    p.add_argument("--v5-strong-trail-buffer-pips", type=float)
    p.add_argument("--v5-london-trail-buffer", type=float)
    p.add_argument("--v5-strong-trail-start-threshold", type=float)
    p.add_argument("--v5-normal-trail-buffer", type=float)
    p.add_argument("--v5-weak-trail-buffer", type=float)
    p.add_argument("--v5-strong-tp1-close-pct", type=float)
    p.add_argument("--v5-normal-tp1-close-pct", type=float)
    p.add_argument("--v5-weak-tp1-close-pct", type=float)
    p.add_argument("--v5-strong-trail-ema", type=int)
    p.add_argument("--v5-normal-trail-ema", type=int)
    p.add_argument("--v5-be-offset", type=float)
    p.add_argument("--v5-sl-lookback", type=int)
    p.add_argument("--v5-sl-buffer", type=float)
    p.add_argument("--v5-sl-floor-pips", type=float)
    p.add_argument("--v5-sl-cap-pips", type=float)
    p.add_argument("--v5-sl-max", type=float)
    p.add_argument("--v5-sl-floor", type=float)
    p.add_argument("--v5-cooldown-win", type=int)
    p.add_argument("--v5-cooldown-loss", type=int)
    p.add_argument("--v5-cooldown-scratch", type=int)
    p.add_argument("--v5-scratch-threshold", type=float)
    p.add_argument("--v5-session-stop-losses", type=int)
    p.add_argument("--v5-close-full-risk-at-session-end", type=parse_bool, nargs="?", const=True)
    p.add_argument("--v5-sizing-mode", choices=["multiplier", "risk_parity", "hybrid"], type=str)
    p.add_argument("--v5-risk-per-trade-pct", type=float)
    p.add_argument("--v5-account-size", type=float)
    p.add_argument("--v5-rp-london-mult", type=float)
    p.add_argument("--v5-rp-ny-mult", type=float)
    p.add_argument("--v5-rp-strong-mult", type=float)
    p.add_argument("--v5-rp-normal-mult", type=float)
    p.add_argument("--v5-rp-win-bonus-pct", type=float)
    p.add_argument("--v5-rp-max-lot-mult", type=float)
    p.add_argument("--v5-rp-min-lot", type=float)
    p.add_argument("--v5-rp-max-lot", type=float)
    p.add_argument("--v5-hybrid-strong-boost", type=float)
    p.add_argument("--v5-hybrid-normal-boost", type=float)
    p.add_argument("--v5-hybrid-london-boost", type=float)
    p.add_argument("--v5-hybrid-ny-boost", type=float)
    p.add_argument("--v5-daily-loss-limit-usd", type=float)
    p.add_argument("--v5-stale-exit-enabled", type=parse_bool, nargs="?", const=True)
    p.add_argument("--v5-stale-exit-bars", type=int)
    p.add_argument("--v5-stale-exit-underwater-pct", type=float)
    p.add_argument("--v5-breakout-entry-enabled", type=parse_bool, nargs="?", const=True)
    p.add_argument("--v5-breakout-min-body-pips", type=float)
    p.add_argument("--v5-breakout-slope-min", type=float)
    p.add_argument("--v5-breakout-sl-lookback", type=int)
    p.add_argument("--v5-atr-min-entry-pips", type=float)
    p.add_argument("--v5-rolling-wr-lookback", type=int)
    p.add_argument("--v5-rolling-wr-min", type=float)
    p.add_argument("--v5-rolling-wr-pause-bars", type=int)
    p.add_argument("--v5-wr-scale-lookback", type=int)
    p.add_argument("--v5-wr-scale-min-trades", type=int)
    p.add_argument("--v5-wr-scale-high", type=float)
    p.add_argument("--v5-wr-scale-mid", type=float)
    p.add_argument("--v5-wr-scale-low", type=float)
    p.add_argument("--v5-wr-scale-full", type=float)
    p.add_argument("--v5-wr-scale-reduced", type=float)
    p.add_argument("--v5-wr-scale-minimal", type=float)
    p.add_argument("--v5-wr-scale-survival", type=float)
    p.add_argument("--v5-wr-scale-boost", type=float)
    p.add_argument("--v5-wr-scale-boost-min", type=float)
    p.add_argument("--v5-max-open-boost", type=int)
    p.add_argument("--v5-wr-open-boost-min", type=float)
    p.add_argument("--v5-conviction-h1-sep-pips", type=float)
    p.add_argument("--v5-conviction-size-mult", type=float)
    p.add_argument("--v5-conviction-wr-gate", type=float)
    p.add_argument("--v5-ny-normal-conviction-override", type=parse_bool, nargs="?", const=True)
    p.add_argument("--v5-london-orb-enabled", type=parse_bool, nargs="?", const=True)
    p.add_argument("--v5-london-orb-fallback-pullback", type=parse_bool, nargs="?", const=True)
    p.add_argument("--v5-orb-min-range-pips", type=float)
    p.add_argument("--v5-orb-max-range-pips", type=float)
    p.add_argument("--v5-orb-tp1-mult", type=float)
    p.add_argument("--v5-orb-tp2-mult", type=float)
    p.add_argument("--v5-orb-sl-buffer-pips", type=float)
    p.add_argument("--v5-orb-sl-cap-pips", type=float)
    p.add_argument("--v5-orb-min-body-pips", type=float)
    p.add_argument("--v5-dual-mode-enabled", type=parse_bool, nargs="?", const=True)
    p.add_argument("--v5-range-fade-enabled", type=parse_bool, nargs="?", const=True)
    p.add_argument("--v5-trend-mode-efficiency-min", type=float)
    p.add_argument("--v5-range-mode-efficiency-max", type=float)
    p.add_argument("--v5-range-fade-pips", type=float)
    p.add_argument("--v5-range-fade-sl-pips", type=float)
    p.add_argument("--v5-range-fade-tp-pips", type=float)
    p.add_argument("--v5-skip-months", type=str)
    p.add_argument("--v5-thin-month", type=int)
    p.add_argument("--v5-thin-start-day", type=int)
    p.add_argument("--v5-thin-scale", type=float)
    # swing parameters
    p.add_argument("--swing-mode-enabled", type=parse_bool, nargs="?", const=True)
    p.add_argument("--swing-weekly-ema", type=int)
    p.add_argument("--swing-daily-ema-fast", type=int)
    p.add_argument("--swing-daily-ema-slow", type=int)
    p.add_argument("--swing-h4-ema", type=int)
    p.add_argument("--swing-h4-ema-slow", type=int)
    p.add_argument("--swing-atr-expansion-ratio", type=float)
    p.add_argument("--swing-sl-h4-lookback", type=int)
    p.add_argument("--swing-sl-buffer-pips", type=float)
    p.add_argument("--swing-sl-cap-pips", type=float)
    p.add_argument("--swing-tp1-rr", type=float)
    p.add_argument("--swing-tp2-rr", type=float)
    p.add_argument("--swing-tp1-close-pct", type=float)
    p.add_argument("--swing-max-duration-days", type=int)
    p.add_argument("--swing-risk-pct", type=float)
    p.add_argument("--swing-account-size", type=float)
    return p


def parse_args(argv: list[str]) -> argparse.Namespace:
    cfg_path = _preparse_config(argv)
    merged = dict(DEFAULTS)
    if cfg_path:
        cfg_obj = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
        if not isinstance(cfg_obj, dict):
            raise ValueError("Config file must contain a JSON object")
        merged.update(cfg_obj)

    parser = _full_parser()
    parser.set_defaults(**merged)
    args = parser.parse_args(argv)
    if not args.inputs:
        parser.error("at least one --in path is required")
    return args


def load_m1(paths: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for p in paths:
        f = Path(p)
        if not f.exists():
            raise FileNotFoundError(f"Missing CSV: {f}")
        df = pd.read_csv(f)
        need = {"time", "open", "high", "low", "close"}
        if not need.issubset(df.columns):
            raise ValueError(f"{f} missing required columns: {sorted(need)}")
        df = df[["time", "open", "high", "low", "close"]].copy()
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        for c in ("open", "high", "low", "close"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna().sort_values("time")
        frames.append(df)
    out = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["time"], keep="last")
        .sort_values("time")
        .reset_index(drop=True)
    )
    return out


def resample_ohlc(m1: pd.DataFrame, rule: str) -> pd.DataFrame:
    d = m1.set_index("time").sort_index()
    out = (
        d.resample(rule, label="right", closed="right")
        .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
        .dropna()
        .reset_index()
    )
    return out


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(int(period), min_periods=int(period)).mean() / PIP_SIZE


def pip_value_usd_per_lot(price: float) -> float:
    return 1000.0 / max(1e-6, float(price))


def hour_in_window(hour: float, start: float, end: float) -> bool:
    h = float(hour) % 24.0
    s = float(start) % 24.0
    e = float(end) % 24.0
    if s == e:
        return False
    if s < e:
        return s <= h < e
    return h >= s or h < e


def ts_hour(ts: pd.Timestamp) -> float:
    return float(ts.hour) + float(ts.minute) / 60.0 + float(ts.second) / 3600.0


def classify_session(
    ts: pd.Timestamp,
    london_start: float,
    london_end: float,
    ny_start: float,
    ny_end: float,
    tokyo_start: float = 0.0,
    tokyo_end: float = 3.0,
    tokyo_enabled: bool = False,
) -> str:
    h = ts_hour(ts)
    if hour_in_window(h, london_start, london_end):
        return "london"
    if hour_in_window(h, ny_start, ny_end):
        return "ny_overlap"
    if bool(tokyo_enabled) and hour_in_window(h, tokyo_start, tokyo_end):
        return "tokyo"
    return "dead"


def compute_spread_pips(i: int, ts: pd.Timestamp, mode: str, avg: float, mn: float, mx: float) -> float:
    if mode == "fixed":
        return max(1.0, float(avg))
    if mode == "realistic":
        h = ts_hour(pd.Timestamp(ts))
        # Deterministic per-session means from spread study.
        if 0.0 <= h < 7.0:
            x = 1.5510  # Tokyo
        elif 7.0 <= h < 11.0:
            x = 2.0007  # London core
        elif 13.0 <= h < 16.0:
            x = 2.3066  # NY overlap
        else:
            x = 1.5500  # all other hours
        return max(1.0, min(float(mx), max(float(mn), x)))
    x = float(avg) + 0.35 * math.sin(i * 0.017) + 0.15 * math.sin(i * 0.071)
    h = int(ts.hour)
    if h >= 16 or h < 7:
        x += 0.08
    return max(1.0, min(float(mx), max(float(mn), x)))


def _mode2_impulse_ok(m5: pd.DataFrame, p5: int, side: str, lookback_bars: int, min_body_pips: float) -> bool:
    lb = int(lookback_bars)
    if p5 < 0 or p5 < lb - 1:
        return False
    w = m5.iloc[p5 - lb + 1 : p5 + 1]
    min_body_px = float(min_body_pips) * PIP_SIZE
    if side == "buy":
        cond = (w["close"] > w["open"]) & ((w["close"] - w["open"]) >= min_body_px)
    else:
        cond = (w["close"] < w["open"]) & ((w["open"] - w["close"]) >= min_body_px)
    return bool(cond.any())


def _mode2_pullback_reentry_ok(
    m5: pd.DataFrame,
    p5: int,
    side: str,
    lookback_bars: int,
    ema_fast_col: str = "ema9",
    ema_slow_col: str = "ema21",
) -> bool:
    lb = int(lookback_bars)
    if p5 < 0 or p5 < lb - 1:
        return False
    w = m5.iloc[p5 - lb + 1 : p5 + 1]
    cur = w.iloc[-1]
    if side == "buy":
        had_wrong_side = bool((w["close"] < w[ema_fast_col]).any())
        return had_wrong_side and float(cur["close"]) > float(cur[ema_fast_col]) and float(cur["close"]) > float(cur[ema_slow_col])
    had_wrong_side = bool((w["close"] > w[ema_fast_col]).any())
    return had_wrong_side and float(cur["close"]) < float(cur[ema_fast_col]) and float(cur["close"]) < float(cur[ema_slow_col])


def _v3_regime_for_index(
    m5: pd.DataFrame,
    p5: int,
    slope_bars: int,
    slope_threshold: float,
    atr_high_ratio: float,
    ema_col: str = "ema9_v3",
    atr_ratio_col: str = "atr_ratio_v3",
) -> str:
    if p5 < 0 or p5 < int(slope_bars):
        return "CHOP"
    now = float(m5.iloc[p5][ema_col])
    ago = float(m5.iloc[p5 - int(slope_bars)][ema_col])
    slope = (now - ago) / (float(slope_bars) * PIP_SIZE)
    atr_ratio = float(m5.iloc[p5][atr_ratio_col]) if pd.notna(m5.iloc[p5][atr_ratio_col]) else 0.0
    if abs(slope) > float(slope_threshold) and atr_ratio > float(atr_high_ratio):
        return "MOMENTUM"
    if abs(slope) > float(slope_threshold):
        return "GRIND"
    return "CHOP"


def _v3_longest_burst_and_pullback(
    m1: pd.DataFrame,
    i: int,
    side: str,
    lookback: int,
    doji_pips: float,
) -> tuple[bool, bool, int]:
    start = max(0, int(i) - int(lookback))
    if int(i) <= start:
        return False, False, 0
    w = m1.iloc[start:int(i)].copy()
    if w.empty:
        return False, False, 0

    doji_px = float(doji_pips) * PIP_SIZE
    dir_flags: list[bool] = []
    counter_or_doji: list[bool] = []
    for _, r in w.iterrows():
        op = float(r["open"])
        cl = float(r["close"])
        body = abs(cl - op)
        if side == "buy":
            dir_bar = cl > op
            counter = cl < op or body < doji_px
        else:
            dir_bar = cl < op
            counter = cl > op or body < doji_px
        dir_flags.append(bool(dir_bar))
        counter_or_doji.append(bool(counter))

    best_len = 0
    best_end = -1
    cur_len = 0
    for j, flag in enumerate(dir_flags):
        if flag:
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_end = j
        else:
            cur_len = 0
    burst = best_len > 0
    if not burst:
        return False, False, 0
    pullback_after = any(counter_or_doji[j] for j in range(best_end + 1, len(counter_or_doji)))
    return True, bool(pullback_after), int(best_len)


def _v3_metrics_from_closed(closed: list[ClosedTrade], max_open_positions: int, blocked_reasons: dict[str, int]) -> dict:
    tdf = pd.DataFrame([x.__dict__ for x in closed])
    if tdf.empty:
        return {
            "summary": {
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": None,
                "net_pips": 0.0,
                "net_usd": 0.0,
                "avg_win_pips": None,
                "avg_loss_pips": None,
                "median_win_pips": None,
                "median_loss_pips": None,
                "breakeven_win_rate_est": None,
                "profit_factor": None,
                "max_drawdown_usd": 0.0,
                "max_open_positions": int(max_open_positions),
            },
            "by_mode": [],
            "by_entry_signal_mode": [],
            "by_session": [],
            "by_exit_reason": [],
            "exit_reason_counts": {},
            "closed_trades": [],
            "daily": [],
            "blocked_reasons": dict(sorted(blocked_reasons.items(), key=lambda kv: kv[1], reverse=True)),
        }

    tdf["is_win"] = tdf["pips"] > 0
    tdf = tdf.sort_values("exit_time").reset_index(drop=True)
    tdf["cum_usd"] = tdf["usd"].cumsum()
    tdf["cum_peak"] = tdf["cum_usd"].cummax()
    tdf["dd_usd"] = tdf["cum_peak"] - tdf["cum_usd"]

    wins = int(tdf["is_win"].sum())
    losses = int((~tdf["is_win"]).sum())
    win_rate = float(tdf["is_win"].mean()) * 100.0
    net_pips = float(tdf["pips"].sum())
    net_usd = float(tdf["usd"].sum())
    avg_win = float(tdf.loc[tdf["is_win"], "pips"].mean()) if wins else None
    avg_loss = float(tdf.loc[~tdf["is_win"], "pips"].mean()) if losses else None
    med_win = float(tdf.loc[tdf["is_win"], "pips"].median()) if wins else None
    med_loss = float(tdf.loc[~tdf["is_win"], "pips"].median()) if losses else None
    abs_avg_loss = abs(avg_loss) if avg_loss is not None else None
    be_wr = (abs_avg_loss / (float(avg_win) + abs_avg_loss) * 100.0) if (avg_win is not None and abs_avg_loss is not None and (float(avg_win) + abs_avg_loss) > 0.0) else None
    gross_win = float(tdf.loc[tdf["pips"] > 0, "pips"].sum())
    gross_loss = abs(float(tdf.loc[tdf["pips"] < 0, "pips"].sum()))
    pf = (gross_win / gross_loss) if gross_loss > 0 else None
    max_dd = float(tdf["dd_usd"].max()) if len(tdf) else 0.0

    by_mode = (
        tdf.groupby("entry_mode", dropna=False)
        .agg(trades=("trade_id", "size"), wins=("is_win", "sum"), losses=("is_win", lambda s: int((~s).sum())), win_rate=("is_win", "mean"), net_pips=("pips", "sum"), net_usd=("usd", "sum"))
        .reset_index()
        .sort_values("trades", ascending=False)
    )
    by_mode["win_rate"] = by_mode["win_rate"] * 100.0

    by_sess = (
        tdf.groupby("entry_session", dropna=False)
        .agg(trades=("trade_id", "size"), wins=("is_win", "sum"), losses=("is_win", lambda s: int((~s).sum())), win_rate=("is_win", "mean"), net_pips=("pips", "sum"), net_usd=("usd", "sum"))
        .reset_index()
        .sort_values("trades", ascending=False)
    )
    by_sess["win_rate"] = by_sess["win_rate"] * 100.0

    by_exit = (
        tdf.groupby("exit_reason", dropna=False)
        .agg(trades=("trade_id", "size"), wins=("is_win", "sum"), losses=("is_win", lambda s: int((~s).sum())), win_rate=("is_win", "mean"), net_pips=("pips", "sum"), net_usd=("usd", "sum"))
        .reset_index()
        .sort_values("trades", ascending=False)
    )
    by_exit["win_rate"] = by_exit["win_rate"] * 100.0

    by_day = (
        tdf.groupby("entry_day", dropna=False)
        .agg(trades=("trade_id", "size"), wins=("is_win", "sum"), net_pips=("pips", "sum"), net_usd=("usd", "sum"))
        .reset_index()
        .sort_values("entry_day")
    )
    exit_reason_counts = {str(k): int(v) for k, v in tdf["exit_reason"].value_counts(dropna=False).to_dict().items()}
    if "entry_signal_mode" in tdf.columns:
        by_sig = (
            tdf.groupby("entry_signal_mode", dropna=False)
            .agg(
                trades=("trade_id", "size"),
                wins=("is_win", "sum"),
                losses=("is_win", lambda s: int((~s).sum())),
                win_rate=("is_win", "mean"),
                net_pips=("pips", "sum"),
                net_usd=("usd", "sum"),
            )
            .reset_index()
            .sort_values("trades", ascending=False)
        )
        by_sig["win_rate"] = by_sig["win_rate"] * 100.0
        by_sig = by_sig.rename(columns={"entry_signal_mode": "signal_mode"})
        by_sig_records = by_sig.to_dict("records")
    else:
        by_sig_records = []

    out = {
        "summary": {
            "trades": int(len(tdf)),
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 3),
            "net_pips": round(net_pips, 3),
            "net_usd": round(net_usd, 3),
            "avg_win_pips": round(avg_win, 3) if avg_win is not None else None,
            "avg_loss_pips": round(avg_loss, 3) if avg_loss is not None else None,
            "median_win_pips": round(med_win, 3) if med_win is not None else None,
            "median_loss_pips": round(med_loss, 3) if med_loss is not None else None,
            "breakeven_win_rate_est": round(be_wr, 3) if be_wr is not None else None,
            "profit_factor": round(pf, 4) if pf is not None else None,
            "max_drawdown_usd": round(max_dd, 3),
            "max_open_positions": int(max_open_positions),
        },
        "by_mode": by_mode.to_dict("records"),
        "by_entry_signal_mode": by_sig_records,
        "by_session": by_sess.to_dict("records"),
        "by_exit_reason": by_exit.to_dict("records"),
        "exit_reason_counts": exit_reason_counts,
        "closed_trades": tdf[
            [
                "trade_id",
                "side",
                "entry_mode",
                "entry_session",
                "entry_time",
                "exit_time",
                "entry_price",
                "exit_price",
                "sl_pips",
                "tp1_pips",
                "tp2_pips",
                "pips",
                "usd",
                "exit_reason",
                "entry_day",
                "entry_regime",
                "entry_profile",
                "position_type",
                "entry_signal_mode",
                "wr_size_scale",
                "conviction_scale",
            ]
        ].to_dict("records"),
        "daily": by_day.to_dict("records"),
        "blocked_reasons": dict(sorted(blocked_reasons.items(), key=lambda kv: kv[1], reverse=True)),
    }
    return out


def run_backtest(args: argparse.Namespace) -> dict:
    m1 = load_m1(args.inputs)
    m5 = resample_ohlc(m1, "5min")
    h1 = resample_ohlc(m1, "1h")
    m5["ema9"] = m5["close"].ewm(span=9, adjust=False).mean()
    m5["ema21"] = m5["close"].ewm(span=21, adjust=False).mean()
    h1["ema_fast"] = h1["close"].ewm(span=int(args.h1_ema_fast), adjust=False).mean()
    h1["ema_slow"] = h1["close"].ewm(span=int(args.h1_ema_slow), adjust=False).mean()

    m5_times = m5["time"].tolist()
    h1_times = h1["time"].tolist()
    p5 = p1 = -1

    daily_state: dict[str, dict] = {}
    open_pos: Optional[OpenPosition] = None
    closed: list[ClosedTrade] = []
    blocked_reasons: dict[str, int] = {}
    max_open_positions = 0
    trade_id_seq = 0

    def add_block(reason: str) -> None:
        blocked_reasons[reason] = blocked_reasons.get(reason, 0) + 1

    def init_session_state() -> dict:
        return {
            "range_high": None,
            "range_low": None,
            "range_bar_count": 0,
            "range_finalized": False,
            "range_valid": False,
            "range_width_pips": None,
            "mode1_taken": False,
        }

    def day_bucket(day_key: str) -> dict:
        if day_key not in daily_state:
            daily_state[day_key] = {
                "entries_opened": 0,
                "first_trade_full_sl": None,
                "mode2_last_entry_p5": None,
                "sessions": {"london": init_session_state(), "ny_overlap": init_session_state()},
            }
        return daily_state[day_key]

    def close_leg(pos: OpenPosition, exit_price: float, leg_lots: float) -> None:
        if leg_lots <= 0:
            return
        pips_leg = ((float(exit_price) - float(pos.entry_price)) / PIP_SIZE) if pos.side == "buy" else ((float(pos.entry_price) - float(exit_price)) / PIP_SIZE)
        usd_leg = pips_leg * pip_value_usd_per_lot(float(exit_price)) * float(leg_lots)
        portion = float(leg_lots) / max(1e-12, float(pos.lots_initial))
        pos.realized_pips += pips_leg * portion
        pos.realized_usd += usd_leg
        pos.exit_price_last = float(exit_price)

    def finalize_trade(pos: OpenPosition, ts: pd.Timestamp, reason: str, full_sl: bool) -> None:
        nonlocal open_pos
        day_key = str(pos.entry_day)
        b = day_bucket(day_key)
        if pos.entry_index_in_day == 1 and b.get("first_trade_full_sl") is None:
            b["first_trade_full_sl"] = bool(full_sl)
        closed.append(
            ClosedTrade(
                trade_id=int(pos.trade_id),
                side=str(pos.side),
                entry_mode=int(pos.entry_mode),
                entry_session=str(pos.entry_session),
                entry_time=pd.Timestamp(pos.entry_time),
                exit_time=pd.Timestamp(ts),
                entry_price=float(pos.entry_price),
                exit_price=float(pos.exit_price_last if pos.exit_price_last is not None else pos.entry_price),
                lots=float(pos.lots_initial),
                sl_pips=float(pos.sl_pips),
                tp1_pips=float(pos.tp1_pips),
                tp2_pips=float(pos.tp2_pips),
                pips=float(pos.realized_pips),
                usd=float(pos.realized_usd),
                exit_reason=str(reason),
                entry_day=day_key,
            )
        )
        open_pos = None

    for i in range(len(m1)):
        row = m1.iloc[i]
        ts = pd.Timestamp(row["time"])
        o = float(row["open"])
        h = float(row["high"])
        l = float(row["low"])
        c = float(row["close"])
        day_key = str(ts.date())
        day_cfg = day_bucket(day_key)

        spread_pips = compute_spread_pips(i, ts, str(args.spread_mode), float(args.spread_pips), float(args.spread_min_pips), float(args.spread_max_pips))
        half_spread = spread_pips * PIP_SIZE / 2.0
        bid = c - half_spread
        ask = c + half_spread

        old5 = p5
        while p5 + 1 < len(m5_times) and m5_times[p5 + 1] <= ts:
            p5 += 1
        new_m5_bar = p5 != old5
        while p1 + 1 < len(h1_times) and h1_times[p1 + 1] <= ts:
            p1 += 1

        if open_pos is not None and open_pos.tp1_filled and p5 >= 0:
            ema9 = float(m5.iloc[p5]["ema9"])
            if open_pos.side == "buy":
                new_stop = ema9 - float(args.trail_buffer_pips) * PIP_SIZE
                if new_stop > float(open_pos.stop_price):
                    open_pos.stop_price = float(new_stop)
            else:
                new_stop = ema9 + float(args.trail_buffer_pips) * PIP_SIZE
                if new_stop < float(open_pos.stop_price):
                    open_pos.stop_price = float(new_stop)

        if open_pos is not None:
            if not open_pos.tp1_filled:
                sl_hit = (l <= float(open_pos.stop_price)) if open_pos.side == "buy" else (h >= float(open_pos.stop_price))
                if sl_hit:
                    close_leg(open_pos, float(open_pos.stop_price), float(open_pos.lots_remaining))
                    finalize_trade(open_pos, ts, "sl", full_sl=True)
                else:
                    tp1_hit = (h >= float(open_pos.tp1_price)) if open_pos.side == "buy" else (l <= float(open_pos.tp1_price))
                    if tp1_hit:
                        leg = float(open_pos.lots_initial) * float(args.tp1_close_fraction)
                        close_leg(open_pos, float(open_pos.tp1_price), leg)
                        open_pos.lots_remaining = max(0.0, float(open_pos.lots_initial) - leg)
                        open_pos.tp1_filled = True
                        open_pos.stop_price = float(open_pos.entry_price) + (float(args.be_offset_pips) * PIP_SIZE if open_pos.side == "buy" else -float(args.be_offset_pips) * PIP_SIZE)
            else:
                tp2_hit = (h >= float(open_pos.tp2_price)) if open_pos.side == "buy" else (l <= float(open_pos.tp2_price))
                if tp2_hit:
                    close_leg(open_pos, float(open_pos.tp2_price), float(open_pos.lots_remaining))
                    finalize_trade(open_pos, ts, "tp1_then_tp2", full_sl=False)
                else:
                    stop_hit = (l <= float(open_pos.stop_price)) if open_pos.side == "buy" else (h >= float(open_pos.stop_price))
                    if stop_hit:
                        close_leg(open_pos, float(open_pos.stop_price), float(open_pos.lots_remaining))
                        finalize_trade(open_pos, ts, "tp1_then_trail", full_sl=False)

        if open_pos is not None:
            max_open_positions = max(max_open_positions, 1)
            continue

        sess = classify_session(ts, float(args.london_start), float(args.london_end), float(args.ny_start), float(args.ny_end))
        if sess not in {"london", "ny_overlap"}:
            continue
        if str(args.sessions) == "ny_only" and sess != "ny_overlap":
            continue
        if str(args.sessions) == "london_only" and sess != "london":
            continue

        if int(day_cfg["entries_opened"]) >= int(args.max_entries_per_day):
            add_block("daily_entry_cap")
            continue
        if int(args.max_open_positions) <= 0:
            add_block("max_open_positions_disabled")
            continue

        if p1 < max(int(args.h1_ema_fast), int(args.h1_ema_slow)) - 1:
            add_block("h1_insufficient_bars")
            continue
        h1_row = h1.iloc[p1]
        ema_fast = float(h1_row["ema_fast"])
        ema_slow = float(h1_row["ema_slow"])
        if ema_fast > ema_slow:
            side = "buy"
        elif ema_fast < ema_slow:
            side = "sell"
        else:
            add_block("h1_no_trend")
            continue

        if not new_m5_bar or p5 < 0:
            continue

        m5_bar = m5.iloc[p5]
        t5 = pd.Timestamp(m5_bar["time"])
        h5 = ts_hour(t5)

        for sname, sstart, send in (
            ("london", float(args.london_start), float(args.london_end)),
            ("ny_overlap", float(args.ny_start), float(args.ny_end)),
        ):
            sstate = day_cfg["sessions"][sname]
            range_end = sstart + float(args.mode1_range_minutes) / 60.0
            in_range_window = hour_in_window(h5, sstart, range_end)
            if in_range_window:
                hi = float(m5_bar["high"])
                lo = float(m5_bar["low"])
                sstate["range_high"] = hi if sstate["range_high"] is None else max(float(sstate["range_high"]), hi)
                sstate["range_low"] = lo if sstate["range_low"] is None else min(float(sstate["range_low"]), lo)
                sstate["range_bar_count"] = int(sstate["range_bar_count"]) + 1
            if (not sstate["range_finalized"]) and (h5 >= range_end):
                sstate["range_finalized"] = True
                if int(sstate["range_bar_count"]) > 0 and sstate["range_high"] is not None and sstate["range_low"] is not None:
                    width_pips = (float(sstate["range_high"]) - float(sstate["range_low"])) / PIP_SIZE
                    sstate["range_width_pips"] = float(width_pips)
                    sstate["range_valid"] = (float(args.range_min_pips) <= width_pips <= float(args.range_max_pips))
                else:
                    sstate["range_valid"] = False
                    sstate["range_width_pips"] = None

        lot_mult = 1.0
        if int(day_cfg["entries_opened"]) >= 1 and bool(day_cfg.get("first_trade_full_sl")):
            lot_mult = float(args.loss_after_first_full_sl_lot_mult)

        def open_trade(entry_mode: int, raw_stop: float, sl_max_pips: float) -> bool:
            nonlocal open_pos, trade_id_seq, max_open_positions
            entry_price = float(ask if side == "buy" else bid)
            sl_dist = abs(entry_price - float(raw_stop)) / PIP_SIZE
            if sl_dist > float(sl_max_pips):
                add_block(f"sl_too_wide_mode_{entry_mode}")
                return False
            if sl_dist < float(args.sl_floor_pips):
                sl_dist = float(args.sl_floor_pips)
                raw_stop = entry_price - sl_dist * PIP_SIZE if side == "buy" else entry_price + sl_dist * PIP_SIZE
            tp1_pips = float(args.tp1_multiplier) * sl_dist
            tp2_pips = float(args.tp2_multiplier) * sl_dist
            tp1_price = entry_price + tp1_pips * PIP_SIZE if side == "buy" else entry_price - tp1_pips * PIP_SIZE
            tp2_price = entry_price + tp2_pips * PIP_SIZE if side == "buy" else entry_price - tp2_pips * PIP_SIZE

            trade_id_seq += 1
            day_cfg["entries_opened"] = int(day_cfg["entries_opened"]) + 1
            lots = max(0.01, float(args.base_lot) * float(lot_mult))
            open_pos = OpenPosition(
                trade_id=int(trade_id_seq),
                side=side,
                entry_mode=int(entry_mode),
                entry_session=sess,
                entry_time=ts,
                entry_day=day_key,
                entry_index_in_day=int(day_cfg["entries_opened"]),
                entry_price=float(entry_price),
                lots_initial=float(lots),
                lots_remaining=float(lots),
                stop_price=float(raw_stop),
                tp1_price=float(tp1_price),
                tp2_price=float(tp2_price),
                sl_pips=float(sl_dist),
                tp1_pips=float(tp1_pips),
                tp2_pips=float(tp2_pips),
                tp1_filled=False,
                realized_pips=0.0,
                realized_usd=0.0,
                exit_price_last=None,
            )
            max_open_positions = max(max_open_positions, 1)
            if int(entry_mode) == 2:
                day_cfg["mode2_last_entry_p5"] = int(p5)
            return True

        opened = False

        # Mode 1 priority
        sstate = day_cfg["sessions"][sess]
        sess_start = float(args.london_start) if sess == "london" else float(args.ny_start)
        range_end = sess_start + float(args.mode1_range_minutes) / 60.0
        if (
            not bool(sstate["mode1_taken"])
            and bool(sstate["range_finalized"])
            and bool(sstate["range_valid"])
            and h5 >= range_end
        ):
            close5 = float(m5_bar["close"])
            up_break = close5 >= float(sstate["range_high"]) + float(args.breakout_margin_pips) * PIP_SIZE
            dn_break = close5 <= float(sstate["range_low"]) - float(args.breakout_margin_pips) * PIP_SIZE
            if side == "buy" and up_break:
                raw_stop = float(sstate["range_low"]) - float(args.sl_buffer_pips) * PIP_SIZE
                opened = open_trade(1, raw_stop, float(args.sl_mode1_max_pips))
            elif side == "sell" and dn_break:
                raw_stop = float(sstate["range_high"]) + float(args.sl_buffer_pips) * PIP_SIZE
                opened = open_trade(1, raw_stop, float(args.sl_mode1_max_pips))
            if opened:
                sstate["mode1_taken"] = True

        # Mode 2
        if (not opened) and (not bool(args.mode1_only)):
            last_mode2 = day_cfg.get("mode2_last_entry_p5")
            if last_mode2 is not None and (int(p5) - int(last_mode2)) < int(args.mode2_cooldown_bars):
                add_block("mode2_cooldown")
            else:
                impulse_ok = _mode2_impulse_ok(m5, p5, side, int(args.mode2_impulse_lookback_bars), float(args.mode2_impulse_min_body_pips))
                pullback_ok = _mode2_pullback_reentry_ok(m5, p5, side, int(args.mode2_pullback_lookback_bars))
                if impulse_ok and pullback_ok:
                    lb = int(args.sl_mode2_lookback_bars)
                    if p5 < lb - 1:
                        add_block("mode2_sl_structure_insufficient")
                    else:
                        w = m5.iloc[p5 - lb + 1 : p5 + 1]
                        if side == "buy":
                            raw_stop = float(w["low"].min()) - float(args.sl_buffer_pips) * PIP_SIZE
                        else:
                            raw_stop = float(w["high"].max()) + float(args.sl_buffer_pips) * PIP_SIZE
                        opened = open_trade(2, raw_stop, float(args.sl_mode2_max_pips))
                else:
                    add_block("mode2_conditions")

    if open_pos is not None:
        last = m1.iloc[-1]
        ts_last = pd.Timestamp(last["time"])
        cl_last = float(last["close"])
        sp_last = compute_spread_pips(len(m1) - 1, ts_last, str(args.spread_mode), float(args.spread_pips), float(args.spread_min_pips), float(args.spread_max_pips))
        hs_last = sp_last * PIP_SIZE / 2.0
        bid_last = cl_last - hs_last
        ask_last = cl_last + hs_last
        exit_px = bid_last if open_pos.side == "buy" else ask_last
        close_leg(open_pos, float(exit_px), float(open_pos.lots_remaining))
        reason = "tp1_then_eod" if open_pos.tp1_filled else "eod"
        finalize_trade(open_pos, ts_last, reason, full_sl=False)

    tdf = pd.DataFrame([x.__dict__ for x in closed])
    if tdf.empty:
        return {
            "summary": {
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": None,
                "net_pips": 0.0,
                "net_usd": 0.0,
                "avg_win_pips": None,
                "avg_loss_pips": None,
                "median_win_pips": None,
                "median_loss_pips": None,
                "breakeven_win_rate_est": None,
                "profit_factor": None,
                "max_drawdown_usd": 0.0,
                "max_open_positions": int(max_open_positions),
            },
            "by_mode": [],
            "by_session": [],
            "by_exit_reason": [],
            "exit_reason_counts": {},
            "closed_trades": [],
            "daily": [],
            "blocked_reasons": dict(sorted(blocked_reasons.items(), key=lambda kv: kv[1], reverse=True)),
        }

    tdf["is_win"] = tdf["pips"] > 0
    tdf = tdf.sort_values("exit_time").reset_index(drop=True)
    tdf["cum_usd"] = tdf["usd"].cumsum()
    tdf["cum_peak"] = tdf["cum_usd"].cummax()
    tdf["dd_usd"] = tdf["cum_peak"] - tdf["cum_usd"]

    wins = int(tdf["is_win"].sum())
    losses = int((~tdf["is_win"]).sum())
    win_rate = float(tdf["is_win"].mean()) * 100.0
    net_pips = float(tdf["pips"].sum())
    net_usd = float(tdf["usd"].sum())
    avg_win = float(tdf.loc[tdf["is_win"], "pips"].mean()) if wins else None
    avg_loss = float(tdf.loc[~tdf["is_win"], "pips"].mean()) if losses else None
    med_win = float(tdf.loc[tdf["is_win"], "pips"].median()) if wins else None
    med_loss = float(tdf.loc[~tdf["is_win"], "pips"].median()) if losses else None
    abs_avg_loss = abs(avg_loss) if avg_loss is not None else None
    be_wr = (abs_avg_loss / (float(avg_win) + abs_avg_loss) * 100.0) if (avg_win is not None and abs_avg_loss is not None and (float(avg_win) + abs_avg_loss) > 0.0) else None
    gross_win = float(tdf.loc[tdf["pips"] > 0, "pips"].sum())
    gross_loss = abs(float(tdf.loc[tdf["pips"] < 0, "pips"].sum()))
    pf = (gross_win / gross_loss) if gross_loss > 0 else None
    max_dd = float(tdf["dd_usd"].max()) if len(tdf) else 0.0

    by_mode = (
        tdf.groupby("entry_mode", dropna=False)
        .agg(trades=("trade_id", "size"), wins=("is_win", "sum"), losses=("is_win", lambda s: int((~s).sum())), win_rate=("is_win", "mean"), net_pips=("pips", "sum"), net_usd=("usd", "sum"))
        .reset_index()
        .sort_values("trades", ascending=False)
    )
    by_mode["win_rate"] = by_mode["win_rate"] * 100.0

    by_sess = (
        tdf.groupby("entry_session", dropna=False)
        .agg(trades=("trade_id", "size"), wins=("is_win", "sum"), losses=("is_win", lambda s: int((~s).sum())), win_rate=("is_win", "mean"), net_pips=("pips", "sum"), net_usd=("usd", "sum"))
        .reset_index()
        .sort_values("trades", ascending=False)
    )
    by_sess["win_rate"] = by_sess["win_rate"] * 100.0

    by_exit = (
        tdf.groupby("exit_reason", dropna=False)
        .agg(trades=("trade_id", "size"), wins=("is_win", "sum"), losses=("is_win", lambda s: int((~s).sum())), win_rate=("is_win", "mean"), net_pips=("pips", "sum"), net_usd=("usd", "sum"))
        .reset_index()
        .sort_values("trades", ascending=False)
    )
    by_exit["win_rate"] = by_exit["win_rate"] * 100.0

    by_day = (
        tdf.groupby("entry_day", dropna=False)
        .agg(trades=("trade_id", "size"), wins=("is_win", "sum"), net_pips=("pips", "sum"), net_usd=("usd", "sum"))
        .reset_index()
        .sort_values("entry_day")
    )
    exit_reason_counts = {str(k): int(v) for k, v in tdf["exit_reason"].value_counts(dropna=False).to_dict().items()}

    return {
        "summary": {
            "trades": int(len(tdf)),
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 3),
            "net_pips": round(net_pips, 3),
            "net_usd": round(net_usd, 3),
            "avg_win_pips": round(avg_win, 3) if avg_win is not None else None,
            "avg_loss_pips": round(avg_loss, 3) if avg_loss is not None else None,
            "median_win_pips": round(med_win, 3) if med_win is not None else None,
            "median_loss_pips": round(med_loss, 3) if med_loss is not None else None,
            "breakeven_win_rate_est": round(be_wr, 3) if be_wr is not None else None,
            "profit_factor": round(pf, 4) if pf is not None else None,
            "max_drawdown_usd": round(max_dd, 3),
            "max_open_positions": int(max_open_positions),
        },
        "by_mode": by_mode.to_dict("records"),
        "by_session": by_sess.to_dict("records"),
        "by_exit_reason": by_exit.to_dict("records"),
        "exit_reason_counts": exit_reason_counts,
        "closed_trades": tdf[
            [
                "trade_id",
                "side",
                "entry_mode",
                "entry_session",
                "entry_time",
                "exit_time",
                "entry_price",
                "exit_price",
                "sl_pips",
                "tp1_pips",
                "tp2_pips",
                "pips",
                "usd",
                "exit_reason",
                "entry_day",
            ]
        ].to_dict("records"),
        "daily": by_day.to_dict("records"),
        "blocked_reasons": dict(sorted(blocked_reasons.items(), key=lambda kv: kv[1], reverse=True)),
    }


def run_backtest_v2(args: argparse.Namespace) -> dict:
    m1 = load_m1(args.inputs)
    m5 = resample_ohlc(m1, "5min")
    h1 = resample_ohlc(m1, "1h")

    m1["ema_fast_v2"] = m1["close"].ewm(span=int(args.v2_m1_ema_fast), adjust=False).mean()
    m1["ema_slow_v2"] = m1["close"].ewm(span=int(args.v2_m1_ema_slow), adjust=False).mean()
    m5["ema9_trail"] = m5["close"].ewm(span=9, adjust=False).mean()
    m5["ema_fast_v2"] = m5["close"].ewm(span=int(args.v2_m5_ema_fast), adjust=False).mean()
    m5["ema_slow_v2"] = m5["close"].ewm(span=int(args.v2_m5_ema_slow), adjust=False).mean()
    h1["ema_fast"] = h1["close"].ewm(span=int(args.h1_ema_fast), adjust=False).mean()
    h1["ema_slow"] = h1["close"].ewm(span=int(args.h1_ema_slow), adjust=False).mean()

    m5_times = m5["time"].tolist()
    h1_times = h1["time"].tolist()
    p5 = p1 = -1

    daily_state: dict[str, dict] = {}
    open_pos: Optional[OpenPosition] = None
    closed: list[ClosedTrade] = []
    blocked_reasons: dict[str, int] = {}
    max_open_positions = 0
    trade_id_seq = 0
    last_sl_loss_close_p5: Optional[int] = None

    setup_active = False
    setup_side: Optional[str] = None
    setup_m5_index: Optional[int] = None

    def add_block(reason: str) -> None:
        blocked_reasons[reason] = blocked_reasons.get(reason, 0) + 1

    def day_bucket(day_key: str) -> dict:
        if day_key not in daily_state:
            daily_state[day_key] = {
                "entries_opened": 0,
                "cooldown_until_p5": None,
            }
        return daily_state[day_key]

    def close_leg(pos: OpenPosition, exit_price: float, leg_lots: float) -> None:
        if leg_lots <= 0:
            return
        pips_leg = ((float(exit_price) - float(pos.entry_price)) / PIP_SIZE) if pos.side == "buy" else ((float(pos.entry_price) - float(exit_price)) / PIP_SIZE)
        usd_leg = pips_leg * pip_value_usd_per_lot(float(exit_price)) * float(leg_lots)
        portion = float(leg_lots) / max(1e-12, float(pos.lots_initial))
        pos.realized_pips += pips_leg * portion
        pos.realized_usd += usd_leg
        pos.exit_price_last = float(exit_price)

    def finalize_trade(pos: OpenPosition, ts: pd.Timestamp, reason: str, p5_now: int) -> None:
        nonlocal open_pos, last_sl_loss_close_p5
        closed.append(
            ClosedTrade(
                trade_id=int(pos.trade_id),
                side=str(pos.side),
                entry_mode=int(pos.entry_mode),
                entry_session=str(pos.entry_session),
                entry_time=pd.Timestamp(pos.entry_time),
                exit_time=pd.Timestamp(ts),
                entry_price=float(pos.entry_price),
                exit_price=float(pos.exit_price_last if pos.exit_price_last is not None else pos.entry_price),
                lots=float(pos.lots_initial),
                sl_pips=float(pos.sl_pips),
                tp1_pips=float(pos.tp1_pips),
                tp2_pips=float(pos.tp2_pips),
                pips=float(pos.realized_pips),
                usd=float(pos.realized_usd),
                exit_reason=str(reason),
                entry_day=str(pos.entry_day),
            )
        )
        close_day = day_bucket(str(ts.date()))
        pips = float(pos.realized_pips)
        if abs(pips) < float(args.v2_scratch_threshold_pips):
            cdb = int(args.v2_cooldown_scratch_bars)
        elif pips > 0:
            cdb = int(args.v2_cooldown_win_bars)
        else:
            cdb = int(args.v2_cooldown_loss_bars)
        close_day["cooldown_until_p5"] = int(p5_now + cdb)
        if str(reason) == "sl":
            last_sl_loss_close_p5 = int(p5_now)
        open_pos = None

    for i in range(len(m1)):
        row = m1.iloc[i]
        ts = pd.Timestamp(row["time"])
        o = float(row["open"])
        h = float(row["high"])
        l = float(row["low"])
        c = float(row["close"])
        day_key = str(ts.date())
        day_cfg = day_bucket(day_key)

        spread_pips = compute_spread_pips(i, ts, str(args.spread_mode), float(args.spread_pips), float(args.spread_min_pips), float(args.spread_max_pips))
        half_spread = spread_pips * PIP_SIZE / 2.0
        bid = c - half_spread
        ask = c + half_spread

        old5 = p5
        while p5 + 1 < len(m5_times) and m5_times[p5 + 1] <= ts:
            p5 += 1
        new_m5_bar = p5 != old5
        while p1 + 1 < len(h1_times) and h1_times[p1 + 1] <= ts:
            p1 += 1

        if open_pos is not None and open_pos.tp1_filled:
            if str(args.v2_trail_source) == "m5":
                if p5 < 0:
                    emaf = float(m1.iloc[i]["ema_fast_v2"])
                else:
                    emaf = float(m5.iloc[p5]["ema9_trail"])
            else:
                emaf = float(m1.iloc[i]["ema_fast_v2"])
            if open_pos.side == "buy":
                new_stop = emaf - float(args.v2_trail_buffer_pips) * PIP_SIZE
                if new_stop > float(open_pos.stop_price):
                    open_pos.stop_price = float(new_stop)
            else:
                new_stop = emaf + float(args.v2_trail_buffer_pips) * PIP_SIZE
                if new_stop < float(open_pos.stop_price):
                    open_pos.stop_price = float(new_stop)

        if open_pos is not None:
            if not open_pos.tp1_filled:
                sl_hit = (l <= float(open_pos.stop_price)) if open_pos.side == "buy" else (h >= float(open_pos.stop_price))
                if sl_hit:
                    close_leg(open_pos, float(open_pos.stop_price), float(open_pos.lots_remaining))
                    finalize_trade(open_pos, ts, "sl", p5_now=p5)
                else:
                    tp1_hit = (h >= float(open_pos.tp1_price)) if open_pos.side == "buy" else (l <= float(open_pos.tp1_price))
                    if tp1_hit:
                        leg = float(open_pos.lots_initial) * float(args.v2_tp1_close_pct)
                        close_leg(open_pos, float(open_pos.tp1_price), leg)
                        open_pos.lots_remaining = max(0.0, float(open_pos.lots_initial) - leg)
                        open_pos.tp1_filled = True
                        open_pos.stop_price = float(open_pos.entry_price) + (float(args.v2_be_offset_pips) * PIP_SIZE if open_pos.side == "buy" else -float(args.v2_be_offset_pips) * PIP_SIZE)
            else:
                tp2_hit = (h >= float(open_pos.tp2_price)) if open_pos.side == "buy" else (l <= float(open_pos.tp2_price))
                if tp2_hit:
                    close_leg(open_pos, float(open_pos.tp2_price), float(open_pos.lots_remaining))
                    finalize_trade(open_pos, ts, "tp1_then_tp2", p5_now=p5)
                else:
                    stop_hit = (l <= float(open_pos.stop_price)) if open_pos.side == "buy" else (h >= float(open_pos.stop_price))
                    if stop_hit:
                        close_leg(open_pos, float(open_pos.stop_price), float(open_pos.lots_remaining))
                        finalize_trade(open_pos, ts, "tp1_then_trail", p5_now=p5)

        trend_side: Optional[str] = None
        if p1 >= max(int(args.h1_ema_fast), int(args.h1_ema_slow)) - 1:
            h1_row = h1.iloc[p1]
            ef = float(h1_row["ema_fast"])
            es = float(h1_row["ema_slow"])
            if ef > es:
                trend_side = "buy"
            elif ef < es:
                trend_side = "sell"

        if new_m5_bar:
            setup_active = False
            setup_side = None
            setup_m5_index = None
            if p5 >= 0 and trend_side is not None:
                m5_row = m5.iloc[p5]
                ef5 = float(m5_row["ema_fast_v2"])
                es5 = float(m5_row["ema_slow_v2"])
                aligned = (trend_side == "buy" and ef5 > es5) or (trend_side == "sell" and ef5 < es5)
                impulse_ok = _mode2_impulse_ok(
                    m5,
                    p5,
                    trend_side,
                    int(args.v2_impulse_lookback_bars),
                    float(args.v2_impulse_min_body_pips),
                )
                pullback_ok = _mode2_pullback_reentry_ok(
                    m5,
                    p5,
                    trend_side,
                    int(args.v2_pullback_lookback_bars),
                    ema_fast_col="ema_fast_v2",
                    ema_slow_col="ema_slow_v2",
                )
                if aligned and impulse_ok and pullback_ok:
                    setup_active = True
                    setup_side = str(trend_side)
                    setup_m5_index = int(p5)

        if open_pos is not None:
            max_open_positions = max(max_open_positions, 1)
            continue

        sess = classify_session(ts, float(args.london_start), float(args.london_end), float(args.ny_start), float(args.ny_end))
        if sess != "ny_overlap":
            continue
        if int(args.max_open_positions) <= 0:
            add_block("max_open_positions_disabled")
            continue
        if int(day_cfg["entries_opened"]) >= int(args.v2_max_entries_per_day):
            add_block("daily_entry_cap")
            continue
        if trend_side is None:
            add_block("h1_no_trend")
            continue
        if not setup_active or setup_side != trend_side:
            add_block("v2_setup_inactive")
            continue
        if bool(args.v2_no_reentry_same_m5_after_loss) and last_sl_loss_close_p5 is not None and int(p5) == int(last_sl_loss_close_p5):
            add_block("v2_same_m5_reentry_block")
            continue
        cooldown_until = day_cfg.get("cooldown_until_p5")
        if cooldown_until is not None and p5 < int(cooldown_until):
            add_block("v2_cooldown")
            continue

        emaf1 = float(m1.iloc[i]["ema_fast_v2"])
        emas1 = float(m1.iloc[i]["ema_slow_v2"])
        if trend_side == "buy":
            trigger = (c > emaf1) and (emaf1 > emas1) and (c > o)
        else:
            trigger = (c < emaf1) and (emaf1 < emas1) and (c < o)
        if not trigger:
            continue

        if str(args.v2_sl_source) == "m5":
            lb5 = int(args.v2_sl_lookback_m5_bars)
            if p5 < lb5 - 1:
                add_block("v2_sl_structure_insufficient")
                continue
            w5 = m5.iloc[p5 - lb5 + 1 : p5 + 1]
            if trend_side == "buy":
                raw_stop = float(w5["low"].min()) - float(args.v2_sl_buffer_pips) * PIP_SIZE
            else:
                raw_stop = float(w5["high"].max()) + float(args.v2_sl_buffer_pips) * PIP_SIZE
        else:
            lb = int(args.v2_sl_lookback_bars)
            if i < lb - 1:
                add_block("v2_sl_structure_insufficient")
                continue
            w1 = m1.iloc[i - lb + 1 : i + 1]
            if trend_side == "buy":
                raw_stop = float(w1["low"].min()) - float(args.v2_sl_buffer_pips) * PIP_SIZE
            else:
                raw_stop = float(w1["high"].max()) + float(args.v2_sl_buffer_pips) * PIP_SIZE

        entry_price = float(ask if trend_side == "buy" else bid)
        sl_dist = abs(entry_price - float(raw_stop)) / PIP_SIZE
        if sl_dist > float(args.v2_sl_max_pips):
            add_block("v2_sl_too_wide")
            continue
        if sl_dist < float(args.v2_sl_floor_pips):
            sl_dist = float(args.v2_sl_floor_pips)
            raw_stop = entry_price - sl_dist * PIP_SIZE if trend_side == "buy" else entry_price + sl_dist * PIP_SIZE

        tp1_pips = float(args.v2_tp1_multiplier) * sl_dist
        tp2_pips = float(args.v2_tp2_multiplier) * sl_dist
        tp1_price = entry_price + tp1_pips * PIP_SIZE if trend_side == "buy" else entry_price - tp1_pips * PIP_SIZE
        tp2_price = entry_price + tp2_pips * PIP_SIZE if trend_side == "buy" else entry_price - tp2_pips * PIP_SIZE

        trade_id_seq += 1
        day_cfg["entries_opened"] = int(day_cfg["entries_opened"]) + 1
        lots = max(0.01, float(args.base_lot))
        open_pos = OpenPosition(
            trade_id=int(trade_id_seq),
            side=trend_side,
            entry_mode=2,
            entry_session=sess,
            entry_time=ts,
            entry_day=day_key,
            entry_index_in_day=int(day_cfg["entries_opened"]),
            entry_price=float(entry_price),
            lots_initial=float(lots),
            lots_remaining=float(lots),
            stop_price=float(raw_stop),
            tp1_price=float(tp1_price),
            tp2_price=float(tp2_price),
            sl_pips=float(sl_dist),
            tp1_pips=float(tp1_pips),
            tp2_pips=float(tp2_pips),
            tp1_filled=False,
            realized_pips=0.0,
            realized_usd=0.0,
            exit_price_last=None,
        )
        max_open_positions = max(max_open_positions, 1)

    if open_pos is not None:
        last = m1.iloc[-1]
        ts_last = pd.Timestamp(last["time"])
        cl_last = float(last["close"])
        sp_last = compute_spread_pips(len(m1) - 1, ts_last, str(args.spread_mode), float(args.spread_pips), float(args.spread_min_pips), float(args.spread_max_pips))
        hs_last = sp_last * PIP_SIZE / 2.0
        bid_last = cl_last - hs_last
        ask_last = cl_last + hs_last
        exit_px = bid_last if open_pos.side == "buy" else ask_last
        close_leg(open_pos, float(exit_px), float(open_pos.lots_remaining))
        reason = "tp1_then_eod" if open_pos.tp1_filled else "eod"
        finalize_trade(open_pos, ts_last, reason, p5_now=p5)

    tdf = pd.DataFrame([x.__dict__ for x in closed])
    if tdf.empty:
        return {
            "summary": {
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": None,
                "net_pips": 0.0,
                "net_usd": 0.0,
                "avg_win_pips": None,
                "avg_loss_pips": None,
                "median_win_pips": None,
                "median_loss_pips": None,
                "breakeven_win_rate_est": None,
                "profit_factor": None,
                "max_drawdown_usd": 0.0,
                "max_open_positions": int(max_open_positions),
            },
            "by_mode": [],
            "by_session": [],
            "by_exit_reason": [],
            "exit_reason_counts": {},
            "closed_trades": [],
            "daily": [],
            "blocked_reasons": dict(sorted(blocked_reasons.items(), key=lambda kv: kv[1], reverse=True)),
        }

    tdf["is_win"] = tdf["pips"] > 0
    tdf = tdf.sort_values("exit_time").reset_index(drop=True)
    tdf["cum_usd"] = tdf["usd"].cumsum()
    tdf["cum_peak"] = tdf["cum_usd"].cummax()
    tdf["dd_usd"] = tdf["cum_peak"] - tdf["cum_usd"]

    wins = int(tdf["is_win"].sum())
    losses = int((~tdf["is_win"]).sum())
    win_rate = float(tdf["is_win"].mean()) * 100.0
    net_pips = float(tdf["pips"].sum())
    net_usd = float(tdf["usd"].sum())
    avg_win = float(tdf.loc[tdf["is_win"], "pips"].mean()) if wins else None
    avg_loss = float(tdf.loc[~tdf["is_win"], "pips"].mean()) if losses else None
    med_win = float(tdf.loc[tdf["is_win"], "pips"].median()) if wins else None
    med_loss = float(tdf.loc[~tdf["is_win"], "pips"].median()) if losses else None
    abs_avg_loss = abs(avg_loss) if avg_loss is not None else None
    be_wr = (abs_avg_loss / (float(avg_win) + abs_avg_loss) * 100.0) if (avg_win is not None and abs_avg_loss is not None and (float(avg_win) + abs_avg_loss) > 0.0) else None
    gross_win = float(tdf.loc[tdf["pips"] > 0, "pips"].sum())
    gross_loss = abs(float(tdf.loc[tdf["pips"] < 0, "pips"].sum()))
    pf = (gross_win / gross_loss) if gross_loss > 0 else None
    max_dd = float(tdf["dd_usd"].max()) if len(tdf) else 0.0

    by_mode = (
        tdf.groupby("entry_mode", dropna=False)
        .agg(trades=("trade_id", "size"), wins=("is_win", "sum"), losses=("is_win", lambda s: int((~s).sum())), win_rate=("is_win", "mean"), net_pips=("pips", "sum"), net_usd=("usd", "sum"))
        .reset_index()
        .sort_values("trades", ascending=False)
    )
    by_mode["win_rate"] = by_mode["win_rate"] * 100.0

    by_sess = (
        tdf.groupby("entry_session", dropna=False)
        .agg(trades=("trade_id", "size"), wins=("is_win", "sum"), losses=("is_win", lambda s: int((~s).sum())), win_rate=("is_win", "mean"), net_pips=("pips", "sum"), net_usd=("usd", "sum"))
        .reset_index()
        .sort_values("trades", ascending=False)
    )
    by_sess["win_rate"] = by_sess["win_rate"] * 100.0

    by_exit = (
        tdf.groupby("exit_reason", dropna=False)
        .agg(trades=("trade_id", "size"), wins=("is_win", "sum"), losses=("is_win", lambda s: int((~s).sum())), win_rate=("is_win", "mean"), net_pips=("pips", "sum"), net_usd=("usd", "sum"))
        .reset_index()
        .sort_values("trades", ascending=False)
    )
    by_exit["win_rate"] = by_exit["win_rate"] * 100.0

    by_day = (
        tdf.groupby("entry_day", dropna=False)
        .agg(trades=("trade_id", "size"), wins=("is_win", "sum"), net_pips=("pips", "sum"), net_usd=("usd", "sum"))
        .reset_index()
        .sort_values("entry_day")
    )
    exit_reason_counts = {str(k): int(v) for k, v in tdf["exit_reason"].value_counts(dropna=False).to_dict().items()}

    return {
        "summary": {
            "trades": int(len(tdf)),
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 3),
            "net_pips": round(net_pips, 3),
            "net_usd": round(net_usd, 3),
            "avg_win_pips": round(avg_win, 3) if avg_win is not None else None,
            "avg_loss_pips": round(avg_loss, 3) if avg_loss is not None else None,
            "median_win_pips": round(med_win, 3) if med_win is not None else None,
            "median_loss_pips": round(med_loss, 3) if med_loss is not None else None,
            "breakeven_win_rate_est": round(be_wr, 3) if be_wr is not None else None,
            "profit_factor": round(pf, 4) if pf is not None else None,
            "max_drawdown_usd": round(max_dd, 3),
            "max_open_positions": int(max_open_positions),
        },
        "by_mode": by_mode.to_dict("records"),
        "by_session": by_sess.to_dict("records"),
        "by_exit_reason": by_exit.to_dict("records"),
        "exit_reason_counts": exit_reason_counts,
        "closed_trades": tdf[
            [
                "trade_id",
                "side",
                "entry_mode",
                "entry_session",
                "entry_time",
                "exit_time",
                "entry_price",
                "exit_price",
                "sl_pips",
                "tp1_pips",
                "tp2_pips",
                "pips",
                "usd",
                "exit_reason",
                "entry_day",
            ]
        ].to_dict("records"),
        "daily": by_day.to_dict("records"),
        "blocked_reasons": dict(sorted(blocked_reasons.items(), key=lambda kv: kv[1], reverse=True)),
    }


def run_backtest_v3(args: argparse.Namespace) -> dict:
    m1 = load_m1(args.inputs)
    m5 = resample_ohlc(m1, "5min")
    h1 = resample_ohlc(m1, "1h")

    m1["ema_fast_v3"] = m1["close"].ewm(span=int(args.v3_m1_ema_fast), adjust=False).mean()
    m1["ema_slow_v3"] = m1["close"].ewm(span=int(args.v3_m1_ema_slow), adjust=False).mean()
    m5["ema9_v3"] = m5["close"].ewm(span=int(args.v3_m5_ema_fast), adjust=False).mean()
    m5["ema21_v3"] = m5["close"].ewm(span=int(args.v3_m5_ema_slow), adjust=False).mean()

    prev_close = m5["close"].shift(1)
    tr = pd.concat(
        [
            (m5["high"] - m5["low"]).abs(),
            (m5["high"] - prev_close).abs(),
            (m5["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    m5["atr_v3"] = tr.rolling(int(args.v3_atr_period), min_periods=int(args.v3_atr_period)).mean()
    m5["atr_sma_v3"] = m5["atr_v3"].rolling(int(args.v3_atr_sma_period), min_periods=int(args.v3_atr_sma_period)).mean()
    m5["atr_ratio_v3"] = m5["atr_v3"] / m5["atr_sma_v3"]

    h1["ema_fast"] = h1["close"].ewm(span=int(args.h1_ema_fast), adjust=False).mean()
    h1["ema_slow"] = h1["close"].ewm(span=int(args.h1_ema_slow), adjust=False).mean()

    m5_times = m5["time"].tolist()
    h1_times = h1["time"].tolist()
    p5 = p1 = -1

    regime_by_p5: list[str] = []
    slope_by_p5: list[float] = []
    for idx in range(len(m5)):
        reg = _v3_regime_for_index(
            m5,
            idx,
            int(args.v3_slope_bars),
            float(args.v3_slope_trending_threshold),
            float(args.v3_atr_high_ratio),
            ema_col="ema9_v3",
            atr_ratio_col="atr_ratio_v3",
        )
        regime_by_p5.append(reg)
        if idx < int(args.v3_slope_bars):
            slope_by_p5.append(0.0)
        else:
            now = float(m5.iloc[idx]["ema9_v3"])
            ago = float(m5.iloc[idx - int(args.v3_slope_bars)]["ema9_v3"])
            slope_by_p5.append((now - ago) / (float(args.v3_slope_bars) * PIP_SIZE))

    daily_state: dict[str, dict] = {}
    open_pos: Optional[OpenPosition] = None
    closed: list[ClosedTrade] = []
    blocked_reasons: dict[str, int] = {}
    max_open_positions = 0
    trade_id_seq = 0
    session_feedback_stops = 0

    setup_active = False
    setup_side: Optional[str] = None
    setup_regime: Optional[str] = None

    regime_distribution = {"MOMENTUM": 0, "GRIND": 0, "CHOP": 0}

    def add_block(reason: str) -> None:
        blocked_reasons[reason] = blocked_reasons.get(reason, 0) + 1

    def day_bucket(day_key: str) -> dict:
        if day_key not in daily_state:
            daily_state[day_key] = {
                "entries_opened": 0,
                "cooldown_until_p5": None,
                "feedback": {
                    "consec_wins": 0,
                    "consec_losses": 0,
                    "stopped": False,
                    "next_profile_override": None,
                    "next_extra_slope": False,
                },
            }
        return daily_state[day_key]

    def close_leg(pos: OpenPosition, exit_price: float, leg_lots: float) -> None:
        if leg_lots <= 0:
            return
        pips_leg = ((float(exit_price) - float(pos.entry_price)) / PIP_SIZE) if pos.side == "buy" else ((float(pos.entry_price) - float(exit_price)) / PIP_SIZE)
        usd_leg = pips_leg * pip_value_usd_per_lot(float(exit_price)) * float(leg_lots)
        portion = float(leg_lots) / max(1e-12, float(pos.lots_initial))
        pos.realized_pips += pips_leg * portion
        pos.realized_usd += usd_leg
        pos.exit_price_last = float(exit_price)

    def finalize_trade(pos: OpenPosition, ts: pd.Timestamp, reason: str, p5_now: int) -> None:
        nonlocal open_pos, session_feedback_stops
        closed.append(
            ClosedTrade(
                trade_id=int(pos.trade_id),
                side=str(pos.side),
                entry_mode=int(pos.entry_mode),
                entry_session=str(pos.entry_session),
                entry_time=pd.Timestamp(pos.entry_time),
                exit_time=pd.Timestamp(ts),
                entry_price=float(pos.entry_price),
                exit_price=float(pos.exit_price_last if pos.exit_price_last is not None else pos.entry_price),
                lots=float(pos.lots_initial),
                sl_pips=float(pos.sl_pips),
                tp1_pips=float(pos.tp1_pips),
                tp2_pips=float(pos.tp2_pips),
                pips=float(pos.realized_pips),
                usd=float(pos.realized_usd),
                exit_reason=str(reason),
                entry_day=str(pos.entry_day),
                entry_regime=str(pos.entry_regime) if pos.entry_regime is not None else None,
                entry_profile=str(pos.entry_profile) if pos.entry_profile is not None else None,
            )
        )

        day_cfg = day_bucket(str(pos.entry_day))
        profile = str(pos.entry_profile or pos.entry_regime or "GRIND")
        pips = float(pos.realized_pips)
        if abs(pips) < float(args.v3_scratch_threshold_pips):
            cdb = int(args.v3_momentum_cooldown_scratch if profile == "MOMENTUM" else args.v3_grind_cooldown_scratch)
        elif pips > 0:
            cdb = int(args.v3_momentum_cooldown_win if profile == "MOMENTUM" else args.v3_grind_cooldown_win)
        else:
            cdb = int(args.v3_momentum_cooldown_loss if profile == "MOMENTUM" else args.v3_grind_cooldown_loss)
        day_cfg["cooldown_until_p5"] = int(p5_now + cdb)

        if bool(args.v3_session_feedback) and str(pos.entry_session) == "ny_overlap":
            fb = day_cfg["feedback"]
            if abs(pips) < float(args.v3_scratch_threshold_pips):
                fb["consec_wins"] = 0
                fb["consec_losses"] = 0
                fb["next_profile_override"] = None
                fb["next_extra_slope"] = False
            elif pips > 0:
                fb["consec_wins"] = int(fb["consec_wins"]) + 1
                fb["consec_losses"] = 0
                fb["next_extra_slope"] = False
                if str(pos.entry_regime) == "GRIND":
                    fb["next_profile_override"] = "MOMENTUM"
                else:
                    fb["next_profile_override"] = None
            else:
                fb["consec_losses"] = int(fb["consec_losses"]) + 1
                fb["consec_wins"] = 0
                if str(pos.entry_regime) == "MOMENTUM":
                    fb["next_profile_override"] = "GRIND"
                    fb["next_extra_slope"] = False
                else:
                    fb["next_profile_override"] = None
                    fb["next_extra_slope"] = True
                if int(fb["consec_losses"]) >= 2 and not bool(fb["stopped"]):
                    fb["stopped"] = True
                    session_feedback_stops += 1

        open_pos = None

    for i in range(len(m1)):
        row = m1.iloc[i]
        ts = pd.Timestamp(row["time"])
        o = float(row["open"])
        h = float(row["high"])
        l = float(row["low"])
        c = float(row["close"])
        day_key = str(ts.date())
        day_cfg = day_bucket(day_key)

        spread_pips = compute_spread_pips(i, ts, str(args.spread_mode), float(args.spread_pips), float(args.spread_min_pips), float(args.spread_max_pips))
        half_spread = spread_pips * PIP_SIZE / 2.0
        bid = c - half_spread
        ask = c + half_spread

        old5 = p5
        while p5 + 1 < len(m5_times) and m5_times[p5 + 1] <= ts:
            p5 += 1
        new_m5_bar = p5 != old5
        while p1 + 1 < len(h1_times) and h1_times[p1 + 1] <= ts:
            p1 += 1

        trend_side: Optional[str] = None
        if p1 >= max(int(args.h1_ema_fast), int(args.h1_ema_slow)) - 1:
            h1_row = h1.iloc[p1]
            ef = float(h1_row["ema_fast"])
            es = float(h1_row["ema_slow"])
            if ef > es:
                trend_side = "buy"
            elif ef < es:
                trend_side = "sell"

        regime = regime_by_p5[p5] if p5 >= 0 else "CHOP"
        slope_now = slope_by_p5[p5] if p5 >= 0 else 0.0

        if new_m5_bar and p5 >= 0:
            sess_now = classify_session(ts, float(args.london_start), float(args.london_end), float(args.ny_start), float(args.ny_end))
            if sess_now == "ny_overlap" or (bool(args.v3_london_in_momentum) and sess_now == "london"):
                regime_distribution[regime] = int(regime_distribution.get(regime, 0)) + 1

            setup_active = False
            setup_side = None
            setup_regime = None
            if trend_side is not None:
                m5_row = m5.iloc[p5]
                ef5 = float(m5_row["ema9_v3"])
                es5 = float(m5_row["ema21_v3"])
                aligned = (trend_side == "buy" and ef5 > es5) or (trend_side == "sell" and ef5 < es5)
                impulse_ok = _mode2_impulse_ok(
                    m5,
                    p5,
                    trend_side,
                    int(args.v3_impulse_lookback_bars),
                    float(args.v3_impulse_min_body_pips),
                )
                pullback_ok = _mode2_pullback_reentry_ok(
                    m5,
                    p5,
                    trend_side,
                    int(args.v3_pullback_lookback_bars),
                    ema_fast_col="ema9_v3",
                    ema_slow_col="ema21_v3",
                )
                if aligned and impulse_ok and pullback_ok:
                    setup_active = True
                    setup_side = str(trend_side)
                    setup_regime = str(regime)

        if open_pos is not None and open_pos.tp1_filled:
            if p5 >= 0:
                ema9 = float(m5.iloc[p5]["ema9_v3"])
            else:
                ema9 = c
            if open_pos.side == "buy":
                new_stop = ema9 - float(open_pos.trail_buffer_pips) * PIP_SIZE
                if new_stop > float(open_pos.stop_price):
                    open_pos.stop_price = float(new_stop)
            else:
                new_stop = ema9 + float(open_pos.trail_buffer_pips) * PIP_SIZE
                if new_stop < float(open_pos.stop_price):
                    open_pos.stop_price = float(new_stop)

        if open_pos is not None:
            if not open_pos.tp1_filled:
                sl_hit = (l <= float(open_pos.stop_price)) if open_pos.side == "buy" else (h >= float(open_pos.stop_price))
                if sl_hit:
                    close_leg(open_pos, float(open_pos.stop_price), float(open_pos.lots_remaining))
                    finalize_trade(open_pos, ts, "sl", p5_now=p5)
                else:
                    tp1_hit = (h >= float(open_pos.tp1_price)) if open_pos.side == "buy" else (l <= float(open_pos.tp1_price))
                    if tp1_hit:
                        leg = float(open_pos.lots_initial) * float(args.v3_tp1_close_pct)
                        close_leg(open_pos, float(open_pos.tp1_price), leg)
                        open_pos.lots_remaining = max(0.0, float(open_pos.lots_initial) - leg)
                        open_pos.tp1_filled = True
                        open_pos.stop_price = float(open_pos.entry_price) + (float(args.v3_be_offset_pips) * PIP_SIZE if open_pos.side == "buy" else -float(args.v3_be_offset_pips) * PIP_SIZE)
            else:
                tp2_hit = (h >= float(open_pos.tp2_price)) if open_pos.side == "buy" else (l <= float(open_pos.tp2_price))
                if tp2_hit:
                    close_leg(open_pos, float(open_pos.tp2_price), float(open_pos.lots_remaining))
                    finalize_trade(open_pos, ts, "tp1_then_tp2", p5_now=p5)
                else:
                    stop_hit = (l <= float(open_pos.stop_price)) if open_pos.side == "buy" else (h >= float(open_pos.stop_price))
                    if stop_hit:
                        close_leg(open_pos, float(open_pos.stop_price), float(open_pos.lots_remaining))
                        finalize_trade(open_pos, ts, "tp1_then_trail", p5_now=p5)

        if open_pos is not None:
            max_open_positions = max(max_open_positions, 1)
            continue

        sess = classify_session(ts, float(args.london_start), float(args.london_end), float(args.ny_start), float(args.ny_end))
        allowed_session = (sess == "ny_overlap") or (bool(args.v3_london_in_momentum) and sess == "london" and regime == "MOMENTUM")
        if not allowed_session:
            continue
        if int(args.max_open_positions) <= 0:
            add_block("max_open_positions_disabled")
            continue
        if int(day_cfg["entries_opened"]) >= int(args.v3_max_entries_per_day):
            add_block("daily_entry_cap")
            continue
        if trend_side is None:
            add_block("h1_no_trend")
            continue
        if regime not in {"MOMENTUM", "GRIND"}:
            add_block("v3_regime_chop")
            continue
        if not setup_active or setup_side != trend_side or setup_regime != regime:
            add_block("v3_setup_inactive")
            continue

        fb = day_cfg["feedback"]
        if bool(args.v3_session_feedback) and sess == "ny_overlap":
            if bool(fb.get("stopped")):
                add_block("v3_session_stopped")
                continue

        cooldown_until = day_cfg.get("cooldown_until_p5")
        if cooldown_until is not None and p5 < int(cooldown_until):
            add_block("v3_cooldown")
            continue

        if bool(args.v3_session_feedback) and sess == "ny_overlap" and bool(fb.get("next_extra_slope")):
            if abs(float(slope_now)) <= 2.0 * float(args.v3_slope_trending_threshold):
                add_block("v3_extra_slope_filter")
                continue

        simple_trigger = (
            ((c > o) and (c > float(m1.iloc[i]["ema_fast_v3"])) and (float(m1.iloc[i]["ema_fast_v3"]) > float(m1.iloc[i]["ema_slow_v3"])))
            if trend_side == "buy"
            else ((c < o) and (c < float(m1.iloc[i]["ema_fast_v3"])) and (float(m1.iloc[i]["ema_fast_v3"]) < float(m1.iloc[i]["ema_slow_v3"])))
        )
        burst_ok, pullback_from_burst, burst_len = _v3_longest_burst_and_pullback(
            m1,
            i,
            trend_side,
            int(args.v3_burst_lookback),
            float(args.v3_doji_pips),
        )
        burst_trigger = bool(burst_ok and int(burst_len) >= int(args.v3_burst_min_bars) and pullback_from_burst and simple_trigger)
        if regime == "MOMENTUM" and bool(args.v3_momentum_allow_simple_entry):
            entry_trigger = bool(simple_trigger or burst_trigger)
        else:
            entry_trigger = bool(burst_trigger)
        if not entry_trigger:
            continue

        profile = str(regime)
        if bool(args.v3_session_feedback) and sess == "ny_overlap":
            override = fb.get("next_profile_override")
            if override in {"MOMENTUM", "GRIND"}:
                profile = str(override)

        lb5 = int(args.v3_sl_lookback_m5_bars)
        if p5 < lb5 - 1:
            add_block("v3_sl_structure_insufficient")
            continue
        w5 = m5.iloc[p5 - lb5 + 1 : p5 + 1]
        if trend_side == "buy":
            raw_stop = float(w5["low"].min()) - float(args.v3_sl_buffer_pips) * PIP_SIZE
        else:
            raw_stop = float(w5["high"].max()) + float(args.v3_sl_buffer_pips) * PIP_SIZE

        sl_max = float(args.v3_momentum_sl_max if profile == "MOMENTUM" else args.v3_grind_sl_max)
        sl_floor = float(args.v3_sl_floor)
        tp1_mult = float(args.v3_momentum_tp1_mult if profile == "MOMENTUM" else args.v3_grind_tp1_mult)
        tp2_mult = float(args.v3_momentum_tp2_mult if profile == "MOMENTUM" else args.v3_grind_tp2_mult)
        trail_buf = float(args.v3_momentum_trail_buffer if profile == "MOMENTUM" else args.v3_grind_trail_buffer)

        entry_price = float(ask if trend_side == "buy" else bid)
        sl_dist = abs(entry_price - float(raw_stop)) / PIP_SIZE
        if sl_dist > sl_max:
            add_block("v3_sl_too_wide")
            continue
        if sl_dist < sl_floor:
            sl_dist = sl_floor
            raw_stop = entry_price - sl_dist * PIP_SIZE if trend_side == "buy" else entry_price + sl_dist * PIP_SIZE

        tp1_pips = tp1_mult * sl_dist
        tp2_pips = tp2_mult * sl_dist
        tp1_price = entry_price + tp1_pips * PIP_SIZE if trend_side == "buy" else entry_price - tp1_pips * PIP_SIZE
        tp2_price = entry_price + tp2_pips * PIP_SIZE if trend_side == "buy" else entry_price - tp2_pips * PIP_SIZE

        if bool(args.v3_adaptive_lots):
            base = float(args.v3_base_lot)
            regime_mult = float(args.v3_lot_momentum_mult if regime == "MOMENTUM" else args.v3_lot_grind_mult)
            lots = base * regime_mult
            if bool(args.v3_session_feedback) and sess == "ny_overlap" and int(fb.get("consec_wins", 0)) >= 2:
                lots = float(args.v3_lot_hot_mult) * base
        else:
            lots = float(args.base_lot)
        lots = max(0.01, float(lots))

        trade_id_seq += 1
        day_cfg["entries_opened"] = int(day_cfg["entries_opened"]) + 1
        open_pos = OpenPosition(
            trade_id=int(trade_id_seq),
            side=trend_side,
            entry_mode=2,
            entry_session=sess,
            entry_time=ts,
            entry_day=day_key,
            entry_index_in_day=int(day_cfg["entries_opened"]),
            entry_price=float(entry_price),
            lots_initial=float(lots),
            lots_remaining=float(lots),
            stop_price=float(raw_stop),
            tp1_price=float(tp1_price),
            tp2_price=float(tp2_price),
            sl_pips=float(sl_dist),
            tp1_pips=float(tp1_pips),
            tp2_pips=float(tp2_pips),
            tp1_filled=False,
            realized_pips=0.0,
            realized_usd=0.0,
            exit_price_last=None,
            entry_regime=str(regime),
            entry_profile=str(profile),
            trail_buffer_pips=float(trail_buf),
        )
        max_open_positions = max(max_open_positions, 1)

        if bool(args.v3_session_feedback) and sess == "ny_overlap":
            fb["next_profile_override"] = None
            fb["next_extra_slope"] = False

    if open_pos is not None:
        last = m1.iloc[-1]
        ts_last = pd.Timestamp(last["time"])
        cl_last = float(last["close"])
        sp_last = compute_spread_pips(len(m1) - 1, ts_last, str(args.spread_mode), float(args.spread_pips), float(args.spread_min_pips), float(args.spread_max_pips))
        hs_last = sp_last * PIP_SIZE / 2.0
        bid_last = cl_last - hs_last
        ask_last = cl_last + hs_last
        exit_px = bid_last if open_pos.side == "buy" else ask_last
        close_leg(open_pos, float(exit_px), float(open_pos.lots_remaining))
        reason = "tp1_then_eod" if open_pos.tp1_filled else "eod"
        finalize_trade(open_pos, ts_last, reason, p5_now=p5)

    results = _v3_metrics_from_closed(closed, max_open_positions, blocked_reasons)
    tdf = pd.DataFrame([x.__dict__ for x in closed])
    if not tdf.empty and "entry_regime" in tdf.columns:
        by_regime = (
            tdf.groupby("entry_regime", dropna=False)
            .agg(
                trades=("trade_id", "size"),
                wins=("pips", lambda s: int((s > 0).sum())),
                losses=("pips", lambda s: int((s <= 0).sum())),
                win_rate=("pips", lambda s: float((s > 0).mean()) * 100.0),
                net_pips=("pips", "sum"),
                net_usd=("usd", "sum"),
            )
            .reset_index()
            .rename(columns={"entry_regime": "regime"})
            .sort_values("trades", ascending=False)
        )
        results["by_regime"] = by_regime.to_dict("records")
    else:
        results["by_regime"] = []
    results["regime_distribution"] = dict(regime_distribution)
    results["session_feedback_stops"] = int(session_feedback_stops)
    return results


def run_backtest_v4(args: argparse.Namespace) -> dict:
    m1 = load_m1(args.inputs)
    m5 = resample_ohlc(m1, "5min")
    h1 = resample_ohlc(m1, "1h")

    m1["ema_fast_v4"] = m1["close"].ewm(span=int(args.v4_m1_ema_fast), adjust=False).mean()
    m1["ema_slow_v4"] = m1["close"].ewm(span=int(args.v4_m1_ema_slow), adjust=False).mean()
    m5["ema_fast_v4"] = m5["close"].ewm(span=int(args.v4_m5_ema_fast), adjust=False).mean()
    m5["ema_slow_v4"] = m5["close"].ewm(span=int(args.v4_m5_ema_slow), adjust=False).mean()
    h1["ema_fast"] = h1["close"].ewm(span=int(args.h1_ema_fast), adjust=False).mean()
    h1["ema_slow"] = h1["close"].ewm(span=int(args.h1_ema_slow), adjust=False).mean()

    m5_times = m5["time"].tolist()
    h1_times = h1["time"].tolist()
    p5 = p1 = -1

    open_positions: list[OpenPosition] = []
    closed: list[ClosedTrade] = []
    blocked_reasons: dict[str, int] = {}
    daily_state: dict[str, dict] = {}
    trade_id_seq = 0
    cooldown_until_p5: Optional[int] = None
    last_trade_outcome: Optional[str] = None
    max_open_positions = 0

    regime_distribution = {"Trending": 0, "Flat": 0}
    sessions_stopped_early = 0
    session_stop_marked: set[str] = set()
    press_sessions: set[str] = set()

    def add_block(reason: str) -> None:
        blocked_reasons[reason] = blocked_reasons.get(reason, 0) + 1

    def day_bucket(day_key: str) -> dict:
        if day_key not in daily_state:
            daily_state[day_key] = {
                "entries_opened": 0,
                "sessions": {
                    "london": {"consec_losses": 0, "stopped": False, "press_override_used": False},
                    "ny_overlap": {"consec_losses": 0, "stopped": False, "press_override_used": False},
                },
            }
        return daily_state[day_key]

    def _session_allowed(sess: str) -> bool:
        mode = str(args.v4_sessions)
        if mode == "both":
            return sess in {"london", "ny_overlap"}
        if mode == "ny_only":
            return sess == "ny_overlap"
        if mode == "london_only":
            return sess == "london"
        return False

    def _session_end_ts(day_key: str, sess: str) -> pd.Timestamp:
        d = pd.Timestamp(day_key, tz="UTC")
        if sess == "london":
            return d + pd.Timedelta(hours=float(args.london_end))
        return d + pd.Timedelta(hours=float(args.ny_end))

    def _close_leg(pos: OpenPosition, exit_price: float, leg_lots: float) -> None:
        if leg_lots <= 0:
            return
        pips_leg = ((float(exit_price) - float(pos.entry_price)) / PIP_SIZE) if pos.side == "buy" else ((float(pos.entry_price) - float(exit_price)) / PIP_SIZE)
        usd_leg = pips_leg * pip_value_usd_per_lot(float(exit_price)) * float(leg_lots)
        portion = float(leg_lots) / max(1e-12, float(pos.lots_initial))
        pos.realized_pips += pips_leg * portion
        pos.realized_usd += usd_leg
        pos.exit_price_last = float(exit_price)

    def _finalize_position(pos: OpenPosition, ts: pd.Timestamp, reason: str, p5_now: int) -> None:
        nonlocal cooldown_until_p5, last_trade_outcome, sessions_stopped_early
        closed.append(
            ClosedTrade(
                trade_id=int(pos.trade_id),
                side=str(pos.side),
                entry_mode=int(pos.entry_mode),
                entry_session=str(pos.entry_session),
                entry_time=pd.Timestamp(pos.entry_time),
                exit_time=pd.Timestamp(ts),
                entry_price=float(pos.entry_price),
                exit_price=float(pos.exit_price_last if pos.exit_price_last is not None else pos.entry_price),
                lots=float(pos.lots_initial),
                sl_pips=float(pos.sl_pips),
                tp1_pips=float(pos.tp1_pips),
                tp2_pips=float(pos.tp2_pips),
                pips=float(pos.realized_pips),
                usd=float(pos.realized_usd),
                exit_reason=str(reason),
                entry_day=str(pos.entry_day),
                entry_regime=str(pos.entry_regime) if pos.entry_regime is not None else None,
                entry_profile=str(pos.entry_profile) if pos.entry_profile is not None else None,
                position_type=str(pos.position_type) if pos.position_type is not None else None,
                entry_signal_mode=str(pos.entry_signal_mode) if pos.entry_signal_mode is not None else None,
            )
        )

        pips = float(pos.realized_pips)
        if abs(pips) < float(args.v4_scratch_threshold):
            cdb = int(args.v4_cooldown_scratch)
            last_trade_outcome = "scratch"
        elif pips > 0:
            cdb = int(args.v4_cooldown_win)
            last_trade_outcome = "win"
        else:
            cdb = int(args.v4_cooldown_loss)
            last_trade_outcome = "loss"
        until = int(p5_now + cdb)
        cooldown_until_p5 = until if cooldown_until_p5 is None else max(int(cooldown_until_p5), until)

        day_cfg = day_bucket(str(pos.entry_day))
        s = day_cfg["sessions"][str(pos.entry_session)]
        if pips < 0:
            s["consec_losses"] = int(s["consec_losses"]) + 1
            if int(s["consec_losses"]) >= int(args.v4_session_stop_losses):
                if not bool(s["stopped"]):
                    s["stopped"] = True
                    sess_key = f"{pos.entry_day}:{pos.entry_session}"
                    if sess_key not in session_stop_marked:
                        session_stop_marked.add(sess_key)
                        sessions_stopped_early += 1
        else:
            s["consec_losses"] = 0

    for i in range(len(m1)):
        row = m1.iloc[i]
        ts = pd.Timestamp(row["time"])
        day_key = str(ts.date())
        day_cfg = day_bucket(day_key)
        o = float(row["open"])
        h = float(row["high"])
        l = float(row["low"])
        c = float(row["close"])

        spread_pips = compute_spread_pips(i, ts, str(args.spread_mode), float(args.spread_pips), float(args.spread_min_pips), float(args.spread_max_pips))
        half_spread = spread_pips * PIP_SIZE / 2.0
        bid = c - half_spread
        ask = c + half_spread

        old5 = p5
        while p5 + 1 < len(m5_times) and m5_times[p5 + 1] <= ts:
            p5 += 1
        new_m5_bar = p5 != old5
        while p1 + 1 < len(h1_times) and h1_times[p1 + 1] <= ts:
            p1 += 1

        sess = classify_session(ts, float(args.london_start), float(args.london_end), float(args.ny_start), float(args.ny_end))

        trend_side: Optional[str] = None
        h1_trend_weak = False
        h1_ema_sep_pips = 0.0
        if p1 >= max(int(args.h1_ema_fast), int(args.h1_ema_slow)) - 1:
            hr = h1.iloc[p1]
            ef = float(hr["ema_fast"])
            es = float(hr["ema_slow"])
            h1_ema_sep_pips = abs(ef - es) / PIP_SIZE
            if ef > es:
                trend_side = "buy"
            elif ef < es:
                trend_side = "sell"
        if trend_side is not None and float(getattr(args, "v5_h1_slope_min", 0.0)) > 0 and p1 >= int(getattr(args, "h1_slope_bars", 6)):
            h1_sb = int(getattr(args, "h1_slope_bars", 6))
            ema_now_h1 = float(h1.iloc[p1]["ema_fast"])
            ema_ago_h1 = float(h1.iloc[p1 - h1_sb]["ema_fast"])
            h1_slope_mag = abs((ema_now_h1 - ema_ago_h1) / (float(h1_sb) * PIP_SIZE))
            if h1_slope_mag < float(args.v5_h1_slope_min):
                h1_trend_weak = True
        # Optional fallback: if EMA crossover is neutral, infer direction from H1 EMA-fast slope.
        if trend_side is None and bool(getattr(args, "h1_allow_slope_direction", False)):
            h1_slope_bars = int(getattr(args, "h1_slope_bars", 6))
            if p1 >= h1_slope_bars:
                ema_now_h1 = float(h1.iloc[p1]["ema_fast"])
                ema_ago_h1 = float(h1.iloc[p1 - h1_slope_bars]["ema_fast"])
                h1_slope = (ema_now_h1 - ema_ago_h1) / (float(h1_slope_bars) * PIP_SIZE)
                if h1_slope > 0.5:
                    trend_side = "buy"
                elif h1_slope < -0.5:
                    trend_side = "sell"

        regime = "Flat"
        setup_active = False
        if p5 >= 2 and trend_side is not None:
            m5r = m5.iloc[p5]
            df = float(m5r["ema_fast_v4"]) - float(m5r["ema_slow_v4"])
            aligned = (trend_side == "buy" and df > 0) or (trend_side == "sell" and df < 0)

            def sign(x: float) -> int:
                return 1 if x > 0 else -1 if x < 0 else 0

            d0 = sign(float(m5.iloc[p5]["ema_fast_v4"]) - float(m5.iloc[p5]["ema_slow_v4"]))
            d1 = sign(float(m5.iloc[p5 - 1]["ema_fast_v4"]) - float(m5.iloc[p5 - 1]["ema_slow_v4"]))
            d2 = sign(float(m5.iloc[p5 - 2]["ema_fast_v4"]) - float(m5.iloc[p5 - 2]["ema_slow_v4"]))
            crossed_recent = (d0 != d1) or (d1 != d2) or d0 == 0 or d1 == 0

            if aligned and not crossed_recent:
                regime = "Trending"
                setup_active = True

        if new_m5_bar and _session_allowed(sess):
            regime_distribution[regime] = int(regime_distribution.get(regime, 0)) + 1

        remaining_positions: list[OpenPosition] = []
        for pos in open_positions:
            closed_now = False

            # Session-end handling for full-risk positions.
            if bool(args.v4_close_full_risk_at_session_end) and not bool(pos.tp1_filled):
                end_ts = _session_end_ts(str(pos.entry_day), str(pos.entry_session))
                if ts >= end_ts:
                    exit_px = bid if pos.side == "buy" else ask
                    _close_leg(pos, float(exit_px), float(pos.lots_remaining))
                    _finalize_position(pos, ts, "session_end_full_risk", p5_now=p5)
                    closed_now = True

            if not closed_now:
                if bool(pos.tp1_filled) and p5 >= 0:
                    ema9 = float(m5.iloc[p5]["ema_fast_v4"])
                    if pos.side == "buy":
                        new_stop = ema9 - float(pos.trail_buffer_pips) * PIP_SIZE
                        if new_stop > float(pos.stop_price):
                            pos.stop_price = float(new_stop)
                    else:
                        new_stop = ema9 + float(pos.trail_buffer_pips) * PIP_SIZE
                        if new_stop < float(pos.stop_price):
                            pos.stop_price = float(new_stop)

                if not bool(pos.tp1_filled):
                    sl_hit = (l <= float(pos.stop_price)) if pos.side == "buy" else (h >= float(pos.stop_price))
                    if sl_hit:
                        _close_leg(pos, float(pos.stop_price), float(pos.lots_remaining))
                        _finalize_position(pos, ts, "sl", p5_now=p5)
                        closed_now = True
                    else:
                        tp1_hit = (h >= float(pos.tp1_price)) if pos.side == "buy" else (l <= float(pos.tp1_price))
                        if tp1_hit:
                            leg = float(pos.lots_initial) * float(args.v4_tp1_close_pct)
                            _close_leg(pos, float(pos.tp1_price), leg)
                            pos.lots_remaining = max(0.0, float(pos.lots_initial) - leg)
                            pos.tp1_filled = True
                            pos.stop_price = float(pos.entry_price) + (float(args.v4_be_offset) * PIP_SIZE if pos.side == "buy" else -float(args.v4_be_offset) * PIP_SIZE)
                else:
                    tp2_hit = (h >= float(pos.tp2_price)) if pos.side == "buy" else (l <= float(pos.tp2_price))
                    if tp2_hit:
                        _close_leg(pos, float(pos.tp2_price), float(pos.lots_remaining))
                        _finalize_position(pos, ts, "tp1_then_tp2", p5_now=p5)
                        closed_now = True
                    else:
                        stop_hit = (l <= float(pos.stop_price)) if pos.side == "buy" else (h >= float(pos.stop_price))
                        if stop_hit:
                            _close_leg(pos, float(pos.stop_price), float(pos.lots_remaining))
                            _finalize_position(pos, ts, "tp1_then_trail", p5_now=p5)
                            closed_now = True

            if not closed_now:
                remaining_positions.append(pos)
        open_positions = remaining_positions
        max_open_positions = max(max_open_positions, len(open_positions))

        if not _session_allowed(sess):
            continue
        if trend_side is None:
            add_block("h1_no_trend")
            continue
        if not setup_active:
            add_block("v4_flat_regime")
            continue
        if cooldown_until_p5 is not None and p5 < int(cooldown_until_p5):
            add_block("v4_cooldown")
            continue
        if int(day_cfg["entries_opened"]) >= int(args.v4_max_entries_day):
            add_block("daily_entry_cap")
            continue
        if len(open_positions) >= int(args.v4_max_open):
            add_block("max_open_cap")
            continue

        sess_state = day_cfg["sessions"][sess]

        position_type: Optional[str] = None
        lot = float(args.v4_probe_lot)
        if len(open_positions) == 0:
            if str(last_trade_outcome) == "loss":
                position_type = "Recovery"
                lot = float(args.v4_recovery_lot)
            else:
                position_type = "Probe"
                lot = float(args.v4_probe_lot)
        elif len(open_positions) == 1:
            p0 = open_positions[0]
            at_be_or_better = (p0.side == "buy" and float(p0.stop_price) >= float(p0.entry_price)) or (p0.side == "sell" and float(p0.stop_price) <= float(p0.entry_price))
            if bool(p0.tp1_filled) and at_be_or_better:
                position_type = "Press"
                lot = float(args.v4_press_lot)

        if position_type is None:
            add_block("position_type_unavailable")
            continue

        if bool(sess_state["stopped"]) and position_type in {"Probe", "Recovery"}:
            add_block("session_stopped")
            continue
        if bool(sess_state["stopped"]) and position_type == "Press":
            if bool(sess_state["press_override_used"]):
                add_block("press_override_used")
                continue
            sess_state["press_override_used"] = True

        # M1 trigger
        ef1 = float(m1.iloc[i]["ema_fast_v4"])
        es1 = float(m1.iloc[i]["ema_slow_v4"])
        if trend_side == "buy":
            trigger = (ef1 > es1) and (c > ef1) and (c > o)
        else:
            trigger = (ef1 < es1) and (c < ef1) and (c < o)
        if not trigger:
            continue

        if i < int(args.v4_sl_lookback) - 1 or p5 < 0:
            add_block("sl_structure_insufficient")
            continue
        w1 = m1.iloc[i - int(args.v4_sl_lookback) + 1 : i + 1]
        m5_last = m5.iloc[p5]
        if trend_side == "buy":
            sl_m1 = float(w1["low"].min()) - float(args.v4_sl_buffer) * PIP_SIZE
            sl_m5_floor = float(m5_last["low"]) - 1.0 * PIP_SIZE
            raw_stop = min(sl_m1, sl_m5_floor)
        else:
            sl_m1 = float(w1["high"].max()) + float(args.v4_sl_buffer) * PIP_SIZE
            sl_m5_floor = float(m5_last["high"]) + 1.0 * PIP_SIZE
            raw_stop = max(sl_m1, sl_m5_floor)

        entry_price = float(ask if trend_side == "buy" else bid)
        sl_dist = abs(entry_price - float(raw_stop)) / PIP_SIZE
        if sl_dist > float(args.v4_sl_max):
            add_block("sl_too_wide")
            continue
        if sl_dist < float(args.v4_sl_floor):
            sl_dist = float(args.v4_sl_floor)
            raw_stop = entry_price - sl_dist * PIP_SIZE if trend_side == "buy" else entry_price + sl_dist * PIP_SIZE

        tp1_pips = float(args.v4_tp1_mult) * sl_dist
        tp2_pips = float(args.v4_tp2_mult) * sl_dist
        tp1_price = entry_price + tp1_pips * PIP_SIZE if trend_side == "buy" else entry_price - tp1_pips * PIP_SIZE
        tp2_price = entry_price + tp2_pips * PIP_SIZE if trend_side == "buy" else entry_price - tp2_pips * PIP_SIZE

        trade_id_seq += 1
        day_cfg["entries_opened"] = int(day_cfg["entries_opened"]) + 1
        pos = OpenPosition(
            trade_id=int(trade_id_seq),
            side=trend_side,
            entry_mode=4,
            entry_session=sess,
            entry_time=ts,
            entry_day=day_key,
            entry_index_in_day=int(day_cfg["entries_opened"]),
            entry_price=float(entry_price),
            lots_initial=max(0.01, float(lot)),
            lots_remaining=max(0.01, float(lot)),
            stop_price=float(raw_stop),
            tp1_price=float(tp1_price),
            tp2_price=float(tp2_price),
            sl_pips=float(sl_dist),
            tp1_pips=float(tp1_pips),
            tp2_pips=float(tp2_pips),
            tp1_filled=False,
            realized_pips=0.0,
            realized_usd=0.0,
            exit_price_last=None,
            entry_regime="Trending",
            entry_profile=str(position_type),
            position_type=str(position_type),
            trail_buffer_pips=float(args.v4_trail_buffer),
        )
        open_positions.append(pos)
        max_open_positions = max(max_open_positions, len(open_positions))
        if position_type == "Press":
            press_sessions.add(f"{day_key}:{sess}")

    # Close remaining at end-of-data.
    if session_tracker_key is not None:
        _finalize_session_efficiency()

    if open_positions:
        last = m1.iloc[-1]
        ts_last = pd.Timestamp(last["time"])
        cl_last = float(last["close"])
        sp_last = compute_spread_pips(len(m1) - 1, ts_last, str(args.spread_mode), float(args.spread_pips), float(args.spread_min_pips), float(args.spread_max_pips))
        hs_last = sp_last * PIP_SIZE / 2.0
        bid_last = cl_last - hs_last
        ask_last = cl_last + hs_last
        for pos in list(open_positions):
            exit_px = bid_last if pos.side == "buy" else ask_last
            _close_leg(pos, float(exit_px), float(pos.lots_remaining))
            reason = "tp1_then_eod" if bool(pos.tp1_filled) else "eod"
            _finalize_position(pos, ts_last, reason, p5_now=p5)
        open_positions = []

    results = _v3_metrics_from_closed(closed, max_open_positions, blocked_reasons)
    tdf = pd.DataFrame([x.__dict__ for x in closed])
    if tdf.empty:
        results["by_position_type"] = []
        results["pyramid_stats"] = {
            "avg_trades_per_day": 0.0,
            "max_trades_in_one_day": 0,
            "sessions_with_press_entries": 0,
            "sessions_stopped_early": int(sessions_stopped_early),
        }
        results["regime_distribution"] = dict(regime_distribution)
        return results

    if "position_type" in tdf.columns:
        by_pt = (
            tdf.groupby("position_type", dropna=False)
            .agg(
                trades=("trade_id", "size"),
                wins=("pips", lambda s: int((s > 0).sum())),
                losses=("pips", lambda s: int((s <= 0).sum())),
                win_rate=("pips", lambda s: float((s > 0).mean()) * 100.0),
                net_pips=("pips", "sum"),
                net_usd=("usd", "sum"),
            )
            .reset_index()
            .sort_values("trades", ascending=False)
        )
        results["by_position_type"] = by_pt.to_dict("records")
    else:
        results["by_position_type"] = []

    day_counts = tdf.groupby("entry_day")["trade_id"].size()
    avg_trades_per_day = float(day_counts.mean()) if len(day_counts) else 0.0
    max_trades_in_one_day = int(day_counts.max()) if len(day_counts) else 0
    results["pyramid_stats"] = {
        "avg_trades_per_day": round(avg_trades_per_day, 3),
        "max_trades_in_one_day": max_trades_in_one_day,
        "sessions_with_press_entries": int(len(press_sessions)),
        "sessions_stopped_early": int(sessions_stopped_early),
    }
    results["regime_distribution"] = dict(regime_distribution)
    return results


def run_backtest_v5(args: argparse.Namespace) -> dict:
    m1 = load_m1(args.inputs)
    m5 = resample_ohlc(m1, "5min")
    h4 = resample_ohlc(m1, "4h")
    h1 = resample_ohlc(m1, "1h")

    m1["ema_fast_v5"] = m1["close"].ewm(span=int(args.v5_m1_ema_fast), adjust=False).mean()
    m1["ema_slow_v5"] = m1["close"].ewm(span=int(args.v5_m1_ema_slow), adjust=False).mean()
    m5["ema_fast_v5"] = m5["close"].ewm(span=int(args.v5_m5_ema_fast), adjust=False).mean()
    m5["ema_slow_v5"] = m5["close"].ewm(span=int(args.v5_m5_ema_slow), adjust=False).mean()
    m5["ema_fast_v5_prev"] = m5["ema_fast_v5"].shift(1)
    m5["ema_slow_v5_prev"] = m5["ema_slow_v5"].shift(1)
    m5["ema9_v5"] = m5["close"].ewm(span=9, adjust=False).mean()
    prev_close = m5["close"].shift(1)
    tr = pd.concat(
        [
            (m5["high"] - m5["low"]).abs(),
            (m5["high"] - prev_close).abs(),
            (m5["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    m5["atr14_v5_pips"] = tr.rolling(14, min_periods=14).mean() / PIP_SIZE

    # ADX (14) on M5 for trend-strength regime filtering.
    def compute_adx(df: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        plus_dm = high.diff()
        minus_dm = low.diff().abs()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        mask = plus_dm > minus_dm
        minus_dm[mask] = 0
        plus_dm[~mask] = 0
        tr_adx = pd.concat(
            [
                (high - low).abs(),
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr_adx = tr_adx.ewm(span=period, adjust=False).mean().replace(0, pd.NA)
        plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr_adx
        minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr_adx
        den = (plus_di + minus_di).replace(0, pd.NA)
        dx = (100 * (plus_di - minus_di).abs() / den).fillna(0.0)
        adx = dx.ewm(span=period, adjust=False).mean().fillna(0.0)
        return adx, plus_di.fillna(0.0), minus_di.fillna(0.0)

    m5["adx14_v5"], m5["plus_di14_v5"], m5["minus_di14_v5"] = compute_adx(m5, 14)
    h4["ema_fast_h4"] = h4["close"].ewm(span=20, adjust=False).mean()
    h4["ema_slow_h4"] = h4["close"].ewm(span=50, adjust=False).mean()
    h4_adx, h4_plus_di, h4_minus_di = compute_adx(h4, 14)
    h4["adx14_h4"] = h4_adx
    h4["plus_di14_h4"] = h4_plus_di
    h4["minus_di14_h4"] = h4_minus_di
    trail_periods = sorted({int(args.v5_strong_trail_ema), int(args.v5_normal_trail_ema)})
    for p in trail_periods:
        m5[f"ema_trail_{p}_v5"] = m5["close"].ewm(span=int(p), adjust=False).mean()
    h1["ema_fast"] = h1["close"].ewm(span=int(args.h1_ema_fast), adjust=False).mean()
    h1["ema_slow"] = h1["close"].ewm(span=int(args.h1_ema_slow), adjust=False).mean()

    m5_times = m5["time"].tolist()
    h4_times = h4["time"].tolist()
    h1_times = h1["time"].tolist()
    p5 = p4 = p1 = -1

    open_positions: list[OpenPosition] = []
    closed: list[ClosedTrade] = []
    blocked_reasons: dict[str, int] = {}
    daily_state: dict[str, dict] = {}
    trade_id_seq = 0
    cooldown_until_p5: Optional[int] = None
    wr_pause_until_p5: Optional[int] = None
    max_open_positions = 0
    sessions_stopped_early = 0
    session_stop_marked: set[str] = set()
    win_bonus_applied_count = 0
    daily_stop_days: set[str] = set()
    daily_stop_usd_days: set[str] = set()
    weekly_stop_keys: set[str] = set()
    realized_day_pips: dict[str, float] = {}
    realized_day_usd: dict[str, float] = {}
    realized_week_pips: dict[str, float] = {}
    max_consecutive_wins = 0
    trail_delayed_count = 0
    sl_cap_skipped_count = 0
    rolling_wr_lookback = max(1, int(getattr(args, "v5_rolling_wr_lookback", 10)))
    recent_trades_outcomes: deque[int] = deque(maxlen=rolling_wr_lookback)
    wr_scale_lookback = max(1, int(getattr(args, "v5_wr_scale_lookback", 20)))
    wr_deque_london: deque[int] = deque(maxlen=wr_scale_lookback)
    wr_deque_ny: deque[int] = deque(maxlen=wr_scale_lookback)
    wr_deque_tokyo: deque[int] = deque(maxlen=wr_scale_lookback)
    session_tracker_key: Optional[str] = None
    session_high: Optional[float] = None
    session_low: Optional[float] = None
    session_open_price: Optional[float] = None
    session_close_price: Optional[float] = None
    session_efficiency_history: deque[float] = deque(maxlen=5)
    session_mode_active: str = "trend"
    session_mode_counts = {"trend": 0, "range_fade": 0, "neutral": 0}
    dual_mode_enabled = bool(getattr(args, "v5_dual_mode_enabled", False))
    # London ORB state (07:00-07:30 range, single breakout trade per London day).
    orb_range_high: Optional[float] = None
    orb_range_low: Optional[float] = None
    orb_range_built = False
    orb_fired_today = False
    orb_last_session_date: Optional[object] = None

    regime_distribution = {"Trending": 0, "Flat": 0}

    def add_block(reason: str) -> None:
        blocked_reasons[reason] = blocked_reasons.get(reason, 0) + 1

    def day_bucket(day_key: str) -> dict:
        if day_key not in daily_state:
            daily_state[day_key] = {
                "entries_opened": 0,
                "win_streak_day": 0,
                "sessions": {
                    "london": {"consec_losses": 0, "stopped": False, "win_streak": 0, "entries_opened": 0},
                    "ny_overlap": {"consec_losses": 0, "stopped": False, "win_streak": 0, "entries_opened": 0},
                    "tokyo": {"consec_losses": 0, "stopped": False, "win_streak": 0, "entries_opened": 0},
                },
            }
        return daily_state[day_key]

    def _week_key(ts: pd.Timestamp) -> str:
        d = pd.Timestamp(ts).tz_convert("UTC")
        monday = (d - pd.Timedelta(days=int(d.dayofweek))).normalize()
        return str(monday.date())

    def _session_allowed(ts: pd.Timestamp, sess: str) -> bool:
        if str(args.v5_sessions) == "ny_only" and sess != "ny_overlap":
            return False
        allowed = {"london", "ny_overlap"}
        if bool(getattr(args, "v5_tokyo_enabled", False)):
            allowed.add("tokyo")
        if sess not in allowed:
            return False
        if sess == "ny_overlap":
            ny_delay = float(args.v5_ny_start_delay_minutes) / 60.0
            if ts_hour(ts) < float(args.ny_start) + ny_delay:
                return False
        return True

    def _in_session_entry_cutoff(ts: pd.Timestamp, sess: str) -> bool:
        cutoff_h = max(0.0, float(args.v5_session_entry_cutoff_minutes) / 60.0)
        if cutoff_h <= 0.0:
            return False
        h = ts_hour(ts)
        if sess == "london":
            london_cutoff_end = min(float(args.london_end), float(args.v5_london_active_end))
            return h >= (london_cutoff_end - cutoff_h)
        if sess == "ny_overlap":
            return h >= (float(args.ny_end) - cutoff_h)
        if sess == "tokyo":
            tokyo_cutoff_h = max(0.0, float(getattr(args, "v5_tokyo_cutoff_minutes", 30)) / 60.0)
            return h >= (float(getattr(args, "v5_tokyo_end", 3.0)) - tokyo_cutoff_h)
        return False

    def _in_london_active_window(ts: pd.Timestamp) -> bool:
        h = ts_hour(ts)
        return float(args.v5_london_active_start) <= h < float(args.v5_london_active_end)

    def _session_end_ts(day_key: str, sess: str) -> pd.Timestamp:
        d = pd.Timestamp(day_key, tz="UTC")
        if sess == "london":
            return d + pd.Timedelta(hours=float(args.london_end))
        if sess == "tokyo":
            return d + pd.Timedelta(hours=float(getattr(args, "v5_tokyo_end", 3.0)))
        return d + pd.Timedelta(hours=float(args.ny_end))

    def _determine_dual_mode() -> str:
        if not dual_mode_enabled:
            return "trend"
        if len(session_efficiency_history) >= 3:
            avg_efficiency = float(sum(session_efficiency_history) / max(1, len(session_efficiency_history)))
            if avg_efficiency >= float(args.v5_trend_mode_efficiency_min):
                return "trend"
            if avg_efficiency <= float(args.v5_range_mode_efficiency_max):
                return "range_fade"
            return "neutral"
        return "trend"

    def _finalize_session_efficiency() -> None:
        nonlocal session_open_price, session_close_price, session_high, session_low
        if session_open_price is None or session_close_price is None or session_high is None or session_low is None:
            return
        session_range_pips = (float(session_high) - float(session_low)) / PIP_SIZE
        if session_range_pips > 0:
            sess_efficiency = abs(float(session_close_price) - float(session_open_price)) / (float(session_range_pips) * PIP_SIZE)
            session_efficiency_history.append(float(sess_efficiency))

    def _wr_deque_for_session(sess: str) -> deque[int]:
        if sess == "london":
            return wr_deque_london
        if sess == "tokyo":
            return wr_deque_tokyo
        return wr_deque_ny

    def _close_leg(pos: OpenPosition, exit_price: float, leg_lots: float) -> None:
        if leg_lots <= 0:
            return
        pips_leg = ((float(exit_price) - float(pos.entry_price)) / PIP_SIZE) if pos.side == "buy" else ((float(pos.entry_price) - float(exit_price)) / PIP_SIZE)
        usd_leg = pips_leg * pip_value_usd_per_lot(float(exit_price)) * float(leg_lots)
        portion = float(leg_lots) / max(1e-12, float(pos.lots_initial))
        pos.realized_pips += pips_leg * portion
        pos.realized_usd += usd_leg
        pos.exit_price_last = float(exit_price)

    def _trend_strength_for_p5(p5_now: int, sess: str) -> tuple[str, float]:
        sb = int(args.v5_slope_bars)
        if p5_now < sb:
            return "Weak", 0.0
        ema_now = float(m5.iloc[p5_now]["ema_fast_v5"])
        ema_ago = float(m5.iloc[p5_now - sb]["ema_fast_v5"])
        slope = (ema_now - ema_ago) / (float(sb) * PIP_SIZE)
        abs_slope = abs(slope)
        if sess == "tokyo":
            strong_threshold = float(getattr(args, "v5_tokyo_strong_slope", 0.3))
        elif sess == "london":
            strong_threshold = float(args.v5_london_strong_slope)
        else:
            strong_threshold = float(args.v5_strong_slope)
        if abs_slope > strong_threshold:
            return "Strong", float(abs_slope)
        if abs_slope > float(args.v5_weak_slope):
            return "Normal", float(abs_slope)
        return "Weak", float(abs_slope)

    def _profile_params(profile: str) -> tuple[float, float, float, float, int, float]:
        if profile == "Strong":
            return (
                float(args.v5_strong_tp1),
                float(args.v5_strong_tp2),
                float(args.v5_strong_tp1_close_pct),
                float(args.v5_strong_trail_buffer_pips),
                int(args.v5_strong_trail_ema),
                float(args.v5_strong_size_mult),
            )
        if profile == "Normal":
            return (
                float(args.v5_normal_tp1),
                float(args.v5_normal_tp2),
                float(args.v5_normal_tp1_close_pct),
                float(args.v5_normal_trail_buffer),
                int(args.v5_normal_trail_ema),
                float(args.v5_normal_size_mult),
            )
        return (
            float(args.v5_weak_tp1),
            float(args.v5_weak_tp2),
            float(args.v5_weak_tp1_close_pct),
            float(args.v5_weak_trail_buffer),
            int(args.v5_normal_trail_ema),
            float(args.v5_weak_size_mult),
        )

    def _strength_allowed(sess: str, strength: str, p5_now: int, slope_abs: float) -> bool:
        def _norm_allow(v: object) -> str:
            s = str(v)
            return "strong_normal" if s == "normal_and_strong" else s

        legacy_london = str(args.v5_london_allow_strength) if getattr(args, "v5_london_allow_strength", None) is not None else None
        london_allow = _norm_allow(args.v5_london_strength_allow) if getattr(args, "v5_london_strength_allow", None) is not None else _norm_allow(legacy_london or args.v5_strength_allow)
        ny_allow = _norm_allow(args.v5_ny_strength_allow) if getattr(args, "v5_ny_strength_allow", None) is not None else _norm_allow(args.v5_strength_allow)
        if sess == "london":
            allow = london_allow
        elif sess == "tokyo":
            allow = _norm_allow(getattr(args, "v5_tokyo_strength_allow", "strong_only"))
        elif sess == "ny_overlap":
            allow = ny_allow
        else:
            allow = _norm_allow(args.v5_strength_allow)
        if allow not in {"all", "strong_only", "strong_normal"}:
            allow = _norm_allow(args.v5_strength_allow)
        if strength == "Weak":
            if bool(args.v5_skip_weak):
                return False
            return allow == "all"
        if strength == "Strong":
            return True
        if strength == "Normal":
            if bool(args.v5_allow_normal_plus):
                if p5_now < 0:
                    return False
                atr_pips = float(m5.iloc[p5_now]["atr14_v5_pips"]) if "atr14_v5_pips" in m5.columns else 0.0
                return (atr_pips >= float(args.v5_normalplus_atr_min_pips)) and (float(slope_abs) >= float(args.v5_normalplus_slope_min))
            return allow in {"all", "strong_normal"}
        return allow == "all"

    def _finalize_position(pos: OpenPosition, ts: pd.Timestamp, reason: str, p5_now: int) -> None:
        nonlocal open_positions, cooldown_until_p5, sessions_stopped_early, max_consecutive_wins, trail_delayed_count, recent_trades_outcomes
        closed.append(
            ClosedTrade(
                trade_id=int(pos.trade_id),
                side=str(pos.side),
                entry_mode=int(pos.entry_mode),
                entry_session=str(pos.entry_session),
                entry_time=pd.Timestamp(pos.entry_time),
                exit_time=pd.Timestamp(ts),
                entry_price=float(pos.entry_price),
                exit_price=float(pos.exit_price_last if pos.exit_price_last is not None else pos.entry_price),
                lots=float(pos.lots_initial),
                sl_pips=float(pos.sl_pips),
                tp1_pips=float(pos.tp1_pips),
                tp2_pips=float(pos.tp2_pips),
                pips=float(pos.realized_pips),
                usd=float(pos.realized_usd),
                exit_reason=str(reason),
                entry_day=str(pos.entry_day),
                entry_regime=str(pos.entry_regime) if pos.entry_regime is not None else None,
                entry_profile=str(pos.entry_profile) if pos.entry_profile is not None else None,
                position_type=str(pos.position_type) if pos.position_type is not None else None,
                entry_signal_mode=str(pos.entry_signal_mode) if pos.entry_signal_mode is not None else None,
                wr_size_scale=float(pos.wr_size_scale),
                conviction_scale=float(pos.conviction_scale),
            )
        )

        pips = float(pos.realized_pips)
        if bool(pos.tp1_filled) and bool(pos.trail_delay_observed):
            trail_delayed_count += 1
        entry_day_cfg = day_bucket(str(pos.entry_day))
        sess_cfg = entry_day_cfg["sessions"][str(pos.entry_session)]
        exit_day_key = str(pd.Timestamp(ts).date())
        exit_day_cfg = day_bucket(exit_day_key)

        if abs(pips) < 1e-12:
            outcome = "scratch"
        elif pips > 0:
            outcome = "win"
        else:
            outcome = "loss"
        recent_trades_outcomes.append(1 if outcome == "win" else 0)
        _wr_deque_for_session(str(pos.entry_session)).append(1 if outcome == "win" else 0)

        if abs(pips) < float(args.v5_scratch_threshold):
            cdb = int(args.v5_cooldown_scratch)
        elif pips > 0:
            cdb = int(args.v5_cooldown_win)
        else:
            cdb = int(args.v5_cooldown_loss)
        until = int(p5_now + cdb)
        cooldown_until_p5 = until if cooldown_until_p5 is None else max(int(cooldown_until_p5), until)

        # Realized pips for day/week loss breakers.
        realized_day_pips[exit_day_key] = float(realized_day_pips.get(exit_day_key, 0.0) + pips)
        realized_day_usd[exit_day_key] = float(realized_day_usd.get(exit_day_key, 0.0) + float(pos.realized_usd))
        wk_key = _week_key(ts)
        realized_week_pips[wk_key] = float(realized_week_pips.get(wk_key, 0.0) + pips)
        if float(args.v5_daily_loss_limit_pips) > 0 and float(realized_day_pips[exit_day_key]) <= -float(args.v5_daily_loss_limit_pips):
            daily_stop_days.add(exit_day_key)
        if float(args.v5_daily_loss_limit_usd) > 0 and float(realized_day_usd[exit_day_key]) <= -float(args.v5_daily_loss_limit_usd):
            daily_stop_usd_days.add(exit_day_key)
        if float(args.v5_weekly_loss_limit_pips) > 0 and float(realized_week_pips[wk_key]) <= -float(args.v5_weekly_loss_limit_pips):
            weekly_stop_keys.add(wk_key)

        if outcome == "win":
            if str(args.v5_win_streak_scope) == "day":
                exit_day_cfg["win_streak_day"] = int(exit_day_cfg.get("win_streak_day", 0)) + 1
                max_consecutive_wins = max(max_consecutive_wins, int(exit_day_cfg["win_streak_day"]))
            else:
                sess_cfg["win_streak"] = int(sess_cfg.get("win_streak", 0)) + 1
                max_consecutive_wins = max(max_consecutive_wins, int(sess_cfg["win_streak"]))
            sess_cfg["consec_losses"] = 0
        elif outcome == "loss":
            if str(args.v5_win_streak_scope) == "day":
                exit_day_cfg["win_streak_day"] = 0
            else:
                sess_cfg["win_streak"] = 0
            sess_cfg["consec_losses"] = int(sess_cfg["consec_losses"]) + 1
            if int(sess_cfg["consec_losses"]) >= int(args.v5_session_stop_losses):
                if not bool(sess_cfg["stopped"]):
                    sess_cfg["stopped"] = True
                    skey = f"{pos.entry_day}:{pos.entry_session}"
                    if skey not in session_stop_marked:
                        session_stop_marked.add(skey)
                        sessions_stopped_early += 1
        else:
            sess_cfg["consec_losses"] = 0

        open_positions = [p for p in open_positions if int(p.trade_id) != int(pos.trade_id)]

    for i in range(len(m1)):
        row = m1.iloc[i]
        ts = pd.Timestamp(row["time"])
        day_key = str(ts.date())
        day_cfg = day_bucket(day_key)
        o = float(row["open"])
        h = float(row["high"])
        l = float(row["low"])
        c = float(row["close"])

        spread_pips = compute_spread_pips(i, ts, str(args.spread_mode), float(args.spread_pips), float(args.spread_min_pips), float(args.spread_max_pips))
        half_spread = spread_pips * PIP_SIZE / 2.0
        bid = c - half_spread
        ask = c + half_spread

        old5 = p5
        while p5 + 1 < len(m5_times) and m5_times[p5 + 1] <= ts:
            p5 += 1
        new_m5_bar = p5 != old5
        while p4 + 1 < len(h4_times) and h4_times[p4 + 1] <= ts:
            p4 += 1
        while p1 + 1 < len(h1_times) and h1_times[p1 + 1] <= ts:
            p1 += 1

        london_start_hour_v5 = float(args.v5_london_start_hour) if getattr(args, "v5_london_start_hour", None) is not None else float(args.london_start)
        tokyo_enabled = bool(getattr(args, "v5_tokyo_enabled", False))
        tokyo_start = float(getattr(args, "v5_tokyo_start", 0.0))
        tokyo_end = float(getattr(args, "v5_tokyo_end", 3.0))
        sess = classify_session(
            ts,
            london_start_hour_v5,
            float(args.london_end),
            float(args.ny_start),
            float(args.ny_end),
            tokyo_start,
            tokyo_end,
            tokyo_enabled,
        )
        ts_dt = pd.Timestamp(ts)
        is_london_open_window = (sess == "london" and int(ts_dt.hour) == 7 and int(ts_dt.minute) < 30)
        is_new_london_day = (sess == "london" and orb_last_session_date != ts_dt.date())
        if is_new_london_day and int(ts_dt.hour) == 7 and int(ts_dt.minute) == 0:
            orb_range_high = float(h)
            orb_range_low = float(l)
            orb_range_built = False
            orb_fired_today = False
            orb_last_session_date = ts_dt.date()
        elif is_london_open_window:
            if orb_last_session_date != ts_dt.date() or orb_range_high is None or orb_range_low is None:
                orb_range_high = float(h)
                orb_range_low = float(l)
                orb_fired_today = False
                orb_last_session_date = ts_dt.date()
            else:
                orb_range_high = max(float(orb_range_high), float(h))
                orb_range_low = min(float(orb_range_low), float(l))
        if sess == "london" and int(ts_dt.hour) == 7 and int(ts_dt.minute) >= 30 and not orb_range_built and orb_range_high is not None and orb_range_low is not None and orb_last_session_date == ts_dt.date():
            orb_range_built = True

        current_session_key = f"{day_key}:{sess}" if sess in {"london", "ny_overlap", "tokyo"} else None
        if current_session_key != session_tracker_key:
            if session_tracker_key is not None:
                _finalize_session_efficiency()
            if current_session_key is None:
                session_tracker_key = None
                session_high = None
                session_low = None
                session_open_price = None
                session_close_price = None
                session_mode_active = "trend"
            else:
                session_tracker_key = current_session_key
                session_high = float(h)
                session_low = float(l)
                session_open_price = float(o)
                session_close_price = float(c)
                session_mode_active = _determine_dual_mode()
                session_mode_counts[str(session_mode_active)] = int(session_mode_counts.get(str(session_mode_active), 0)) + 1
        elif current_session_key is not None:
            session_high = max(float(session_high), float(h)) if session_high is not None else float(h)
            session_low = min(float(session_low), float(l)) if session_low is not None else float(l)
            session_close_price = float(c)

        trend_side: Optional[str] = None
        h1_trend_weak = False
        h1_ema_sep_pips = 0.0
        if p1 >= max(int(args.h1_ema_fast), int(args.h1_ema_slow)) - 1:
            hr = h1.iloc[p1]
            ef = float(hr["ema_fast"])
            es = float(hr["ema_slow"])
            h1_ema_sep_pips = abs(ef - es) / PIP_SIZE
            if ef > es:
                trend_side = "buy"
            elif ef < es:
                trend_side = "sell"
        if trend_side is not None and float(getattr(args, "v5_h1_slope_min", 0.0)) > 0 and p1 >= int(getattr(args, "h1_slope_bars", 6)):
            h1_sb = int(getattr(args, "h1_slope_bars", 6))
            ema_now_h1 = float(h1.iloc[p1]["ema_fast"])
            ema_ago_h1 = float(h1.iloc[p1 - h1_sb]["ema_fast"])
            h1_slope_mag = abs((ema_now_h1 - ema_ago_h1) / (float(h1_sb) * PIP_SIZE))
            if h1_slope_mag < float(args.v5_h1_slope_min):
                h1_trend_weak = True

        setup_active = False
        regime = "Flat"
        if p5 >= 0 and trend_side is not None:
            m5r = m5.iloc[p5]
            m5diff = float(m5r["ema_fast_v5"]) - float(m5r["ema_slow_v5"])
            setup_active = (trend_side == "buy" and m5diff > 0) or (trend_side == "sell" and m5diff < 0)
            if setup_active:
                regime = "Trending"
        if new_m5_bar and sess in {"london", "ny_overlap", "tokyo"} and _session_allowed(ts, sess):
            regime_distribution[regime] = int(regime_distribution.get(regime, 0)) + 1

        if open_positions:
            for pos in list(open_positions):
                closed_now = False
                if bool(args.v5_close_full_risk_at_session_end) and not bool(pos.tp1_filled):
                    end_ts = _session_end_ts(str(pos.entry_day), str(pos.entry_session))
                    if ts >= end_ts:
                        exit_px = bid if pos.side == "buy" else ask
                        _close_leg(pos, float(exit_px), float(pos.lots_remaining))
                        _finalize_position(pos, ts, "session_end_full_risk", p5_now=p5)
                        closed_now = True

                if closed_now:
                    continue

                if bool(pos.tp1_filled) and p5 >= 0:
                    can_trail = True
                    if not bool(pos.trail_armed):
                        activation_px = float(pos.entry_price) + float(pos.trail_start_multiple) * float(pos.sl_pips) * PIP_SIZE if pos.side == "buy" else float(pos.entry_price) - float(pos.trail_start_multiple) * float(pos.sl_pips) * PIP_SIZE
                        threshold_reached = (h >= activation_px) if pos.side == "buy" else (l <= activation_px)
                        if threshold_reached:
                            pos.trail_armed = True
                        else:
                            can_trail = False
                            pos.trail_delay_observed = True
                    if can_trail:
                        trail_col = f"ema_trail_{int(pos.trail_ema_period)}_v5"
                        ema_fast = float(m5.iloc[p5][trail_col]) if trail_col in m5.columns else float(m5.iloc[p5]["ema_fast_v5"])
                        if pos.side == "buy":
                            new_stop = ema_fast - float(pos.trail_buffer_pips) * PIP_SIZE
                            if new_stop > float(pos.stop_price):
                                pos.stop_price = float(new_stop)
                        else:
                            new_stop = ema_fast + float(pos.trail_buffer_pips) * PIP_SIZE
                            if new_stop < float(pos.stop_price):
                                pos.stop_price = float(new_stop)

                if not bool(pos.tp1_filled):
                    sl_hit = (l <= float(pos.stop_price)) if pos.side == "buy" else (h >= float(pos.stop_price))
                    if sl_hit:
                        _close_leg(pos, float(pos.stop_price), float(pos.lots_remaining))
                        _finalize_position(pos, ts, "sl", p5_now=p5)
                        continue
                    tp1_hit = (h >= float(pos.tp1_price)) if pos.side == "buy" else (l <= float(pos.tp1_price))
                    if tp1_hit:
                        leg = float(pos.lots_initial) * float(pos.tp1_close_fraction)
                        _close_leg(pos, float(pos.tp1_price), leg)
                        pos.lots_remaining = max(0.0, float(pos.lots_initial) - leg)
                        pos.tp1_filled = True
                        pos.stop_price = float(pos.entry_price) + (float(args.v5_be_offset) * PIP_SIZE if pos.side == "buy" else -float(args.v5_be_offset) * PIP_SIZE)
                    elif bool(args.v5_stale_exit_enabled):
                        bars_since_entry = int(i - int(pos.entry_bar_index))
                        stale_bars = int(args.v5_stale_exit_bars)
                        if bars_since_entry >= stale_bars:
                            current_pips = ((float(bid) - float(pos.entry_price)) / PIP_SIZE) if pos.side == "buy" else ((float(pos.entry_price) - float(ask)) / PIP_SIZE)
                            stale_limit = -float(pos.sl_pips) * float(args.v5_stale_exit_underwater_pct)
                            if current_pips < stale_limit:
                                exit_px = float(bid if pos.side == "buy" else ask)
                                _close_leg(pos, exit_px, float(pos.lots_remaining))
                                _finalize_position(pos, ts, "stale_exit", p5_now=p5)
                                continue
                else:
                    tp2_hit = (h >= float(pos.tp2_price)) if pos.side == "buy" else (l <= float(pos.tp2_price))
                    if tp2_hit:
                        _close_leg(pos, float(pos.tp2_price), float(pos.lots_remaining))
                        _finalize_position(pos, ts, "tp1_then_tp2", p5_now=p5)
                        continue
                    stop_hit = (l <= float(pos.stop_price)) if pos.side == "buy" else (h >= float(pos.stop_price))
                    if stop_hit:
                        _close_leg(pos, float(pos.stop_price), float(pos.lots_remaining))
                        _finalize_position(pos, ts, "tp1_then_trail", p5_now=p5)
                        continue

            max_open_positions = max(max_open_positions, len(open_positions))
            # Dynamic max-open cap: allow higher concurrency in strong rolling-WR regimes.
            base_max_open = int(getattr(args, "v5_max_open", 2))
            boost_max_open = int(getattr(args, "v5_max_open_boost", base_max_open))
            wr_open_boost_min = float(getattr(args, "v5_wr_open_boost_min", 0.50))
            min_wr_trades_for_open = max(0, int(getattr(args, "v5_wr_scale_min_trades", 10)))
            wr_scale_deque_open = _wr_deque_for_session(sess)
            if len(wr_scale_deque_open) >= min_wr_trades_for_open and len(wr_scale_deque_open) > 0:
                current_rolling_wr = float(sum(wr_scale_deque_open)) / float(len(wr_scale_deque_open))
                effective_max_open = boost_max_open if current_rolling_wr >= wr_open_boost_min else base_max_open
            else:
                effective_max_open = base_max_open
            if len(open_positions) >= int(effective_max_open):
                add_block("v5_max_open_cap")
                continue

        if not _session_allowed(ts, sess):
            continue
        ts_dt = pd.Timestamp(ts)
        cal_month = int(ts_dt.month)
        cal_day = int(ts_dt.day)

        # Structural calendar filter: skip full months, optionally scale late-month exposure.
        skip_months_str = str(getattr(args, "v5_skip_months", ""))
        if skip_months_str.strip():
            skip_months = [int(m.strip()) for m in skip_months_str.split(",") if m.strip().isdigit()]
            if cal_month in skip_months:
                add_block("calendar_month_skip")
                continue

        thin_month = int(getattr(args, "v5_thin_month", 0))
        thin_start_day = int(getattr(args, "v5_thin_start_day", 20))
        thin_scale = float(getattr(args, "v5_thin_scale", 0.25))
        holiday_scale = 1.0
        if thin_month > 0 and cal_month == thin_month and cal_day >= thin_start_day:
            holiday_scale = thin_scale
            if holiday_scale == 0.0:
                add_block("calendar_thin_skip")
                continue
        active_mode = str(session_mode_active) if dual_mode_enabled else "trend"
        if dual_mode_enabled and active_mode == "neutral":
            add_block("v5_dual_mode_neutral")
            continue

        entry_side: Optional[str] = trend_side
        strength = "Strong"
        slope_abs = 0.0
        pullback_trigger = False
        breakout_trigger = False
        range_fade_trigger = False
        orb_trigger = False
        entry_signal_mode = "pullback"
        entry_regime = "Trending"
        entry_profile_label = str(strength)
        orb_enabled = bool(getattr(args, "v5_london_orb_enabled", False))
        orb_allow_pullback_fallback = bool(getattr(args, "v5_london_orb_fallback_pullback", False))
        london_orb_mode = (sess == "london" and orb_enabled and active_mode == "trend")
        orb_tp1_mult = float(getattr(args, "v5_orb_tp1_mult", 1.0))
        orb_tp2_mult = float(getattr(args, "v5_orb_tp2_mult", 2.0))
        orb_size_pips = 0.0

        if active_mode == "trend":
            if trend_side is None:
                add_block("h1_no_trend")
                continue
            if h1_trend_weak:
                add_block("v5_h1_trend_weak")
                continue
            if sess == "tokyo" and bool(getattr(args, "v5_tokyo_require_h4_trend", False)):
                if p4 >= 1:
                    h4_ef = float(h4.iloc[p4]["ema_fast_h4"])
                    h4_es = float(h4.iloc[p4]["ema_slow_h4"])
                    h4_bullish = h4_ef > h4_es
                    h4_bearish = h4_ef < h4_es
                    if trend_side == "buy" and not h4_bullish:
                        add_block("tokyo_h4_disagree")
                        continue
                    if trend_side == "sell" and not h4_bearish:
                        add_block("tokyo_h4_disagree")
                        continue
                else:
                    add_block("tokyo_h4_insufficient")
                    continue
            if float(getattr(args, "v5_h4_adx_min", 0.0)) > 0 and p4 >= 14:
                h4_adx_val = float(h4.iloc[p4]["adx14_h4"])
                if h4_adx_val < float(args.v5_h4_adx_min):
                    add_block("v5_h4_adx_too_low")
                    continue
            if int(getattr(args, "v5_h1_slope_consistent_bars", 0)) > 0 and trend_side is not None:
                n = int(args.v5_h1_slope_consistent_bars)
                if p1 >= n:
                    consistent = True
                    for k in range(1, n + 1):
                        e_now = float(h1.iloc[p1 - k + 1]["ema_fast"])
                        e_ago = float(h1.iloc[p1 - k]["ema_fast"])
                        bar_slope = e_now - e_ago
                        if (trend_side == "buy" and bar_slope <= 0) or (trend_side == "sell" and bar_slope >= 0):
                            consistent = False
                            break
                    if not consistent:
                        add_block("v5_h1_slope_inconsistent")
                        continue
            if not london_orb_mode:
                if not setup_active:
                    add_block("v5_flat_regime")
                    continue
                strength, slope_abs = _trend_strength_for_p5(p5, sess)
                strength_allowed = _strength_allowed(sess, strength, p5, slope_abs)
                if not strength_allowed:
                    strength_override = False
                    if (
                        sess == "ny_overlap"
                        and strength == "Normal"
                        and bool(getattr(args, "v5_ny_normal_conviction_override", False))
                        and float(h1_ema_sep_pips) >= float(getattr(args, "v5_conviction_h1_sep_pips", 999.0))
                    ):
                        strength_override = True
                    if not strength_override:
                        if strength == "Weak" and bool(args.v5_skip_weak):
                            add_block("v5_skip_weak")
                        elif strength == "Normal" and bool(args.v5_allow_normal_plus):
                            add_block("v5_normalplus_filter")
                        else:
                            add_block("v5_strength_filter")
                        continue
                if bool(args.v5_skip_normal) and strength == "Normal":
                    add_block("v5_skip_normal")
                    continue
                if float(getattr(args, "v5_atr_min_entry_pips", 0.0)) > 0 and p5 >= 0:
                    m5_atr = float(m5.iloc[p5].get("atr14_v5_pips", 0.0))
                    if m5_atr < float(args.v5_atr_min_entry_pips):
                        add_block("v5_atr_too_low")
                        continue
                if float(getattr(args, "v5_adx_min", 0.0)) > 0 and p5 >= 0:
                    adx_val = float(m5.iloc[p5].get("adx14_v5", 0.0))
                    if adx_val < float(args.v5_adx_min):
                        add_block("v5_adx_too_low")
                        continue
                entry_profile_label = str(strength)
            else:
                strength = "Strong"
                slope_abs = 0.0
                entry_profile_label = "london_orb"
        else:
            entry_regime = "RangeFade"
            strength = "Normal"
            entry_profile_label = "range_fade"
        if float(getattr(args, "v5_rolling_wr_min", 0.0)) > 0:
            if wr_pause_until_p5 is not None and p5 < int(wr_pause_until_p5):
                add_block("v5_wr_circuit_breaker")
                continue
            if len(recent_trades_outcomes) >= int(args.v5_rolling_wr_lookback):
                rolling_wr = sum(recent_trades_outcomes) / float(len(recent_trades_outcomes))
                if rolling_wr < float(args.v5_rolling_wr_min):
                    wr_pause_until_p5 = int(p5 + int(args.v5_rolling_wr_pause_bars))
                    add_block("v5_wr_circuit_breaker")
                    continue
        if cooldown_until_p5 is not None and p5 < int(cooldown_until_p5):
            add_block("v5_cooldown")
            continue
        wk_key_now = _week_key(ts)
        if float(args.v5_daily_loss_limit_pips) > 0 and day_key in daily_stop_days:
            add_block("v5_daily_loss_stop")
            continue
        if float(args.v5_daily_loss_limit_usd) > 0 and day_key in daily_stop_usd_days:
            add_block("v5_daily_loss_usd_stop")
            continue
        if float(args.v5_weekly_loss_limit_pips) > 0 and wk_key_now in weekly_stop_keys:
            add_block("v5_weekly_loss_stop")
            continue
        if float(spread_pips) > float(args.max_entry_spread_pips):
            add_block("v5_entry_spread_too_high")
            continue
        if sess == "london" and not _in_london_active_window(ts):
            add_block("v5_london_active_window")
            continue
        if _in_session_entry_cutoff(ts, sess):
            add_block("v5_session_entry_cutoff")
            continue
        if int(day_cfg["entries_opened"]) >= int(args.v5_max_entries_day):
            add_block("daily_entry_cap")
            continue
        sess_cfg = day_cfg["sessions"][sess]
        if sess == "london" and int(sess_cfg.get("entries_opened", 0)) >= int(args.v5_london_max_entries):
            add_block("v5_london_entry_cap")
            continue
        if sess == "tokyo" and int(sess_cfg.get("entries_opened", 0)) >= int(getattr(args, "v5_tokyo_max_entries", 3)):
            add_block("v5_tokyo_entry_cap")
            continue
        if bool(sess_cfg["stopped"]):
            add_block("session_stopped")
            continue
        if float(getattr(args, "v5_session_range_cap_pips", 0.0)) > 0 and session_high is not None and session_low is not None:
            session_range = (float(session_high) - float(session_low)) / PIP_SIZE
            if session_range > float(args.v5_session_range_cap_pips):
                add_block("v5_session_range_cap")
                continue

        # London ORB entry path (replaces London pullback unless fallback is enabled).
        if london_orb_mode:
            if orb_range_built and not orb_fired_today and orb_range_high is not None and orb_range_low is not None:
                orb_size_pips = (float(orb_range_high) - float(orb_range_low)) / PIP_SIZE
                orb_min_size = float(getattr(args, "v5_orb_min_range_pips", 5.0))
                orb_max_size = float(getattr(args, "v5_orb_max_range_pips", 25.0))
                if orb_size_pips < orb_min_size:
                    add_block("orb_range_too_small")
                    orb_fired_today = True
                elif orb_size_pips > orb_max_size:
                    add_block("orb_range_too_large")
                    orb_fired_today = True
                else:
                    orb_sl_buffer = float(getattr(args, "v5_orb_sl_buffer_pips", 2.0))
                    orb_body_min = float(getattr(args, "v5_orb_min_body_pips", 1.5))
                    orb_bar_body = abs(float(c) - float(o)) / PIP_SIZE
                    if trend_side == "buy" and float(c) > float(orb_range_high) and orb_bar_body >= orb_body_min:
                        entry_side = "buy"
                        entry_signal_mode = "orb_breakout"
                        entry_regime = "ORB"
                        orb_trigger = True
                        orb_fired_today = True
                        add_block("orb_entry_buy")
                    elif trend_side == "sell" and float(c) < float(orb_range_low) and orb_bar_body >= orb_body_min:
                        entry_side = "sell"
                        entry_signal_mode = "orb_breakout"
                        entry_regime = "ORB"
                        orb_trigger = True
                        orb_fired_today = True
                        add_block("orb_entry_sell")
                    else:
                        add_block("orb_no_breakout")
            if not orb_allow_pullback_fallback and not orb_trigger:
                continue

        if not orb_trigger and active_mode == "range_fade" and dual_mode_enabled:
            if not bool(getattr(args, "v5_range_fade_enabled", True)):
                add_block("range_fade_disabled")
                continue
            if session_open_price is None:
                add_block("v5_range_fade_no_session_open")
                continue
            fade_threshold = float(args.v5_range_fade_pips) * PIP_SIZE
            move_from_open = float(c) - float(session_open_price)
            if move_from_open > fade_threshold and trend_side != "sell":
                entry_side = "sell"
                range_fade_trigger = True
            elif move_from_open < -fade_threshold and trend_side != "buy":
                entry_side = "buy"
                range_fade_trigger = True
            else:
                add_block("v5_range_fade_no_signal")
                continue
            entry_signal_mode = "range_fade"
        elif not orb_trigger:
            ef1 = float(m1.iloc[i]["ema_fast_v5"])
            es1 = float(m1.iloc[i]["ema_slow_v5"])
            if trend_side == "buy":
                pullback_trigger = (ef1 > es1) and (c > ef1) and (c > o)
            else:
                pullback_trigger = (ef1 < es1) and (c < ef1) and (c < o)

            # --- BREAKOUT ENTRY MODE ---
            # Fires on new M5 EMA cross bar with strong slope/body, no pullback requirement.
            if bool(getattr(args, "v5_breakout_entry_enabled", False)) and new_m5_bar and p5 >= 1:
                m5r_now = m5.iloc[p5]
                ef_now = float(m5r_now["ema_fast_v5"])
                es_now = float(m5r_now["ema_slow_v5"])
                ef_prev = float(m5r_now["ema_fast_v5_prev"])
                es_prev = float(m5r_now["ema_slow_v5_prev"])
                m5_body = abs(float(m5r_now["close"]) - float(m5r_now["open"])) / PIP_SIZE
                body_ok = m5_body >= float(getattr(args, "v5_breakout_min_body_pips", 2.0))
                if p5 >= int(args.v5_slope_bars) + 1:
                    ema_prev_now = float(m5.iloc[p5 - 1]["ema_fast_v5"])
                    ema_prev_ago = float(m5.iloc[p5 - 1 - int(args.v5_slope_bars)]["ema_fast_v5"])
                    prev_slope_abs = abs((ema_prev_now - ema_prev_ago) / (float(args.v5_slope_bars) * PIP_SIZE))
                    slope_ok = float(slope_abs) >= float(getattr(args, "v5_breakout_slope_min", 0.3)) and float(slope_abs) >= float(prev_slope_abs) * 0.8
                else:
                    slope_ok = float(slope_abs) >= float(getattr(args, "v5_breakout_slope_min", 0.3))
                if trend_side == "buy":
                    cross_ok = (ef_prev <= es_prev) and (ef_now > es_now)
                    bar_dir = float(m5r_now["close"]) > float(m5r_now["open"])
                else:
                    cross_ok = (ef_prev >= es_prev) and (ef_now < es_now)
                    bar_dir = float(m5r_now["close"]) < float(m5r_now["open"])
                if cross_ok and body_ok and slope_ok and bar_dir:
                    breakout_trigger = True
            if not pullback_trigger and not breakout_trigger:
                continue
            entry_side = str(trend_side)
            entry_signal_mode = "breakout" if (breakout_trigger and not pullback_trigger) else "pullback"

        if pullback_trigger:
            if sess == "london":
                confirm_bars = int(args.v5_london_confirm_bars)
                min_body_pips = float(args.v5_london_min_body_pips)
            elif sess == "tokyo":
                confirm_bars = int(getattr(args, "v5_tokyo_confirm_bars", 1))
                min_body_pips = float(args.v5_entry_min_body_pips)
            else:
                confirm_bars = 1
                min_body_pips = float(args.v5_entry_min_body_pips)
            if i < (confirm_bars - 1):
                add_block("v5_confirm_insufficient_history")
                continue
            confirm_ok = True
            for j in range(i - confirm_bars + 1, i + 1):
                oj = float(m1.iloc[j]["open"])
                cj = float(m1.iloc[j]["close"])
                body_pips = abs(cj - oj) / PIP_SIZE
                dir_ok = (cj > oj) if trend_side == "buy" else (cj < oj)
                if (not dir_ok) or (body_pips < min_body_pips):
                    confirm_ok = False
                    break
            if not confirm_ok:
                if sess == "london":
                    add_block("v5_london_confirm_fail")
                elif sess == "tokyo":
                    add_block("v5_tokyo_confirm_fail")
                else:
                    add_block("v5_entry_body_too_small")
                continue

        entry_price = float(ask if entry_side == "buy" else bid)
        if orb_trigger:
            orb_sl_buffer = float(getattr(args, "v5_orb_sl_buffer_pips", 2.0))
            if entry_side == "buy":
                raw_stop = float(orb_range_low) - orb_sl_buffer * PIP_SIZE
                sl_dist = (entry_price - float(raw_stop)) / PIP_SIZE
            else:
                raw_stop = float(orb_range_high) + orb_sl_buffer * PIP_SIZE
                sl_dist = (float(raw_stop) - entry_price) / PIP_SIZE
            if sl_dist <= 0:
                add_block("orb_invalid_sl")
                continue
            orb_sl_cap = float(getattr(args, "v5_orb_sl_cap_pips", 999.0))
            if float(orb_sl_cap) > 0 and sl_dist > float(orb_sl_cap):
                sl_dist = float(orb_sl_cap)
                if entry_side == "buy":
                    raw_stop = float(entry_price) - float(sl_dist) * PIP_SIZE
                else:
                    raw_stop = float(entry_price) + float(sl_dist) * PIP_SIZE
        elif float(getattr(args, "v5_sl_atr_scale", 0.0)) > 0 and p5 >= 0:
            atr_val = float(m5.iloc[p5].get("atr14_v5_pips", 7.0))
            sl_dist = max(float(atr_val) * float(args.v5_sl_atr_scale), float(args.v5_sl_floor_pips))
            if float(args.v5_sl_cap_pips) > 0:
                sl_dist = min(float(sl_dist), float(args.v5_sl_cap_pips))
            raw_stop = entry_price - float(sl_dist) * PIP_SIZE if entry_side == "buy" else entry_price + float(sl_dist) * PIP_SIZE
        elif range_fade_trigger:
            sl_dist = float(args.v5_range_fade_sl_pips)
            raw_stop = entry_price - float(sl_dist) * PIP_SIZE if entry_side == "buy" else entry_price + float(sl_dist) * PIP_SIZE
        else:
            if p5 < 0:
                add_block("sl_structure_insufficient")
                continue
            if breakout_trigger and not pullback_trigger:
                brk_lb = max(1, int(getattr(args, "v5_breakout_sl_lookback", 1)))
                start = max(0, p5 - brk_lb + 1)
                m5w = m5.iloc[start : p5 + 1]
                if entry_side == "buy":
                    raw_stop = float(m5w["low"].min()) - float(args.v5_sl_buffer) * PIP_SIZE
                else:
                    raw_stop = float(m5w["high"].max()) + float(args.v5_sl_buffer) * PIP_SIZE
            else:
                if i < int(args.v5_sl_lookback) - 1:
                    add_block("sl_structure_insufficient")
                    continue
                w1 = m1.iloc[i - int(args.v5_sl_lookback) + 1 : i + 1]
                m5_last = m5.iloc[p5]
                if entry_side == "buy":
                    raw_stop = min(float(w1["low"].min()), float(m5_last["low"])) - float(args.v5_sl_buffer) * PIP_SIZE
                else:
                    raw_stop = max(float(w1["high"].max()), float(m5_last["high"])) + float(args.v5_sl_buffer) * PIP_SIZE

            sl_dist = abs(entry_price - float(raw_stop)) / PIP_SIZE
            sl_floor = float(getattr(args, "v5_tokyo_sl_floor_pips", 3.5)) if sess == "tokyo" else float(args.v5_sl_floor_pips)
            sl_cap = float(getattr(args, "v5_tokyo_sl_cap_pips", 6.0)) if sess == "tokyo" else float(args.v5_sl_cap_pips)
            if float(sl_cap) > 0 and sl_dist > float(sl_cap):
                sl_cap_skipped_count += 1
                add_block("v5_sl_cap_skip")
                continue
            if sl_dist < float(sl_floor):
                sl_dist = float(sl_floor)
                raw_stop = entry_price - sl_dist * PIP_SIZE if trend_side == "buy" else entry_price + sl_dist * PIP_SIZE

        tp1_mult, tp2_mult, tp1_close_pct, trail_buf, trail_ema_period, ny_mult = _profile_params(strength)
        if sess == "tokyo":
            tp1_mult = float(getattr(args, "v5_tokyo_tp1_mult", 1.5))
            tp2_mult = float(getattr(args, "v5_tokyo_tp2_mult", 3.0))
            tp1_close_pct = float(getattr(args, "v5_tokyo_tp1_close_pct", 0.5))
            trail_buf = float(getattr(args, "v5_tokyo_trail_buffer", 2.0))
        elif sess == "london" and strength == "Strong":
            tp1_mult = float(args.v5_london_strong_tp1)
            tp2_mult = float(args.v5_london_strong_tp2)
            tp1_close_pct = float(args.v5_london_tp1_close_pct)
            trail_buf = float(args.v5_london_trail_buffer)
        if str(args.v5_win_streak_scope) == "day":
            bonus_steps = int(day_cfg.get("win_streak_day", 0))
        else:
            bonus_steps = int(sess_cfg.get("win_streak", 0))
        bonus_steps = max(0, int(bonus_steps))

        wr_scale = float(getattr(args, "v5_wr_scale_full", 1.0))
        min_wr_trades = max(0, int(getattr(args, "v5_wr_scale_min_trades", 10)))
        wr_scale_deque = _wr_deque_for_session(sess)
        if len(wr_scale_deque) >= min_wr_trades and len(wr_scale_deque) > 0:
            rolling_wr = float(sum(wr_scale_deque)) / float(len(wr_scale_deque))
            v5_wr_scale_boost = float(getattr(args, "v5_wr_scale_boost", 1.0))
            v5_wr_scale_boost_min = float(getattr(args, "v5_wr_scale_boost_min", 0.50))
            if rolling_wr >= v5_wr_scale_boost_min:
                wr_scale = v5_wr_scale_boost
            elif rolling_wr >= float(getattr(args, "v5_wr_scale_high", 0.40)):
                wr_scale = float(getattr(args, "v5_wr_scale_full", 1.0))
            elif rolling_wr >= float(getattr(args, "v5_wr_scale_mid", 0.33)):
                wr_scale = float(getattr(args, "v5_wr_scale_reduced", 0.6))
            elif rolling_wr >= float(getattr(args, "v5_wr_scale_low", 0.25)):
                wr_scale = float(getattr(args, "v5_wr_scale_minimal", 0.35))
            else:
                wr_scale = float(getattr(args, "v5_wr_scale_survival", 0.2))
            if wr_scale == v5_wr_scale_boost and v5_wr_scale_boost > 1.0:
                add_block("v5_wr_scale_boosted")
        if wr_scale <= 0.35:
            add_block("v5_wr_scale_reduced")
        if wr_scale <= 0.2:
            add_block("v5_wr_scale_survival")

        conviction_sep_min = float(getattr(args, "v5_conviction_h1_sep_pips", 0.0))
        conviction_mult = float(getattr(args, "v5_conviction_size_mult", 1.0))
        conviction_wr_gate = float(getattr(args, "v5_conviction_wr_gate", 1.0))
        wr_gate_ok = float(wr_scale) >= float(conviction_wr_gate)
        conviction_active = (
            conviction_sep_min > 0.0
            and float(h1_ema_sep_pips) >= conviction_sep_min
            and str(strength) == "Strong"
            and bool(wr_gate_ok)
            and (not orb_trigger)
        )
        conviction_scale = float(conviction_mult) if conviction_active else 1.0
        if conviction_active:
            add_block("v5_conviction_fired")

        if str(args.v5_sizing_mode) in {"risk_parity", "hybrid"}:
            base_risk_usd = float(args.v5_account_size) * (float(args.v5_risk_per_trade_pct) / 100.0)
            pip_value_per_lot = (100000.0 * PIP_SIZE) / max(1e-6, float(entry_price))
            raw_lot = base_risk_usd / max(1e-9, float(sl_dist) * float(pip_value_per_lot))
            if str(args.v5_sizing_mode) == "hybrid":
                if sess == "london":
                    sess_mult = float(args.v5_hybrid_london_boost)
                elif sess == "tokyo":
                    sess_mult = float(getattr(args, "v5_rp_tokyo_mult", 0.8))
                else:
                    sess_mult = float(args.v5_hybrid_ny_boost)
                if strength == "Strong":
                    strength_mult = float(args.v5_hybrid_strong_boost)
                else:
                    strength_mult = float(args.v5_hybrid_normal_boost)
            else:
                if sess == "london":
                    sess_mult = float(args.v5_rp_london_mult)
                elif sess == "tokyo":
                    sess_mult = float(getattr(args, "v5_rp_tokyo_mult", 0.8))
                else:
                    sess_mult = float(args.v5_rp_ny_mult)
                if strength == "Strong":
                    strength_mult = float(args.v5_rp_strong_mult)
                else:
                    strength_mult = float(args.v5_rp_normal_mult)
            rp_lot = float(raw_lot) * sess_mult * strength_mult
            bonus_factor = 1.0 + (float(args.v5_rp_win_bonus_pct) / 100.0) * float(bonus_steps)
            bonus_factor = min(float(args.v5_rp_max_lot_mult), bonus_factor)
            if bonus_steps > 0:
                win_bonus_applied_count += 1
            final_lot = float(rp_lot) * float(bonus_factor) * float(wr_scale) * float(conviction_scale) * float(holiday_scale)
            final_lot = max(float(args.v5_rp_min_lot), min(float(args.v5_rp_max_lot), final_lot))
        else:
            if sess == "london":
                regime_lot = float(args.v5_base_lot) * float(args.v5_london_size_mult)
            else:
                regime_lot = float(args.v5_base_lot) * ny_mult
            bonus_lot = float(args.v5_win_bonus_per_step) * float(bonus_steps)
            if bonus_lot > 0:
                win_bonus_applied_count += 1
            final_lot = min(float(args.v5_max_lot), (regime_lot + bonus_lot) * float(conviction_scale) * float(holiday_scale))
            final_lot = max(0.01, float(final_lot))

        if range_fade_trigger:
            tp1_pips = float(args.v5_range_fade_tp_pips) * 0.5
            tp2_pips = float(args.v5_range_fade_tp_pips)
            tp1_close_pct = 0.5
            trail_buf = float(args.v5_normal_trail_buffer)
            trail_ema_period = int(args.v5_normal_trail_ema)
            tp1_mult = tp1_pips / max(1e-9, float(sl_dist))
        elif orb_trigger:
            tp1_pips = float(orb_size_pips) * float(orb_tp1_mult)
            tp2_pips = float(orb_size_pips) * float(orb_tp2_mult)
            tp1_close_pct = 0.5
            trail_buf = float(args.v5_london_trail_buffer)
            trail_ema_period = int(args.v5_normal_trail_ema)
            tp1_mult = tp1_pips / max(1e-9, float(sl_dist))
        elif float(getattr(args, "v5_tp_atr_scale", 0.0)) > 0 and p5 >= 0:
            tp1_pips = float(sl_dist) * 2.0
            tp2_pips = float(sl_dist) * 4.0
        else:
            tp1_pips = tp1_mult * float(sl_dist)
            tp2_pips = tp2_mult * float(sl_dist)
        tp1_price = entry_price + tp1_pips * PIP_SIZE if entry_side == "buy" else entry_price - tp1_pips * PIP_SIZE
        tp2_price = entry_price + tp2_pips * PIP_SIZE if entry_side == "buy" else entry_price - tp2_pips * PIP_SIZE

        trade_id_seq += 1
        day_cfg["entries_opened"] = int(day_cfg["entries_opened"]) + 1
        sess_cfg["entries_opened"] = int(sess_cfg.get("entries_opened", 0)) + 1
        new_pos = OpenPosition(
            trade_id=int(trade_id_seq),
            side=str(entry_side),
            entry_mode=5,
            entry_session=str(sess),
            entry_time=pd.Timestamp(ts),
            entry_day=str(day_key),
            entry_index_in_day=int(day_cfg["entries_opened"]),
            entry_price=float(entry_price),
            lots_initial=float(final_lot),
            lots_remaining=float(final_lot),
            stop_price=float(raw_stop),
            tp1_price=float(tp1_price),
            tp2_price=float(tp2_price),
            sl_pips=float(sl_dist),
            tp1_pips=float(tp1_pips),
            tp2_pips=float(tp2_pips),
            tp1_filled=False,
            realized_pips=0.0,
            realized_usd=0.0,
            exit_price_last=None,
            entry_regime=str(entry_regime),
            entry_profile=str(entry_profile_label),
            position_type="Single",
            trail_buffer_pips=float(trail_buf),
            trail_ema_period=int(trail_ema_period),
            tp1_close_fraction=float(tp1_close_pct),
            trail_start_multiple=float(tp1_mult + float(args.v5_trail_start_after_tp1_mult)),
            trail_armed=False if float(args.v5_trail_start_after_tp1_mult) > 0 else True,
            trail_delay_observed=False,
            entry_bar_index=int(i),
            entry_signal_mode=str(entry_signal_mode),
            wr_size_scale=float(wr_scale),
            conviction_scale=float(conviction_scale),
        )
        open_positions.append(new_pos)
        max_open_positions = max(max_open_positions, len(open_positions))

    if open_positions:
        last = m1.iloc[-1]
        ts_last = pd.Timestamp(last["time"])
        cl_last = float(last["close"])
        sp_last = compute_spread_pips(len(m1) - 1, ts_last, str(args.spread_mode), float(args.spread_pips), float(args.spread_min_pips), float(args.spread_max_pips))
        hs_last = sp_last * PIP_SIZE / 2.0
        bid_last = cl_last - hs_last
        ask_last = cl_last + hs_last
        for pos in list(open_positions):
            exit_px = bid_last if pos.side == "buy" else ask_last
            _close_leg(pos, float(exit_px), float(pos.lots_remaining))
            reason = "tp1_then_eod" if bool(pos.tp1_filled) else "eod"
            _finalize_position(pos, ts_last, reason, p5_now=p5)

    results = _v3_metrics_from_closed(closed, max_open_positions, blocked_reasons)
    tdf = pd.DataFrame([x.__dict__ for x in closed])
    if tdf.empty:
        results["by_trend_strength"] = []
        results["by_mode"] = [
            {"mode": "trend", "sessions": int(session_mode_counts.get("trend", 0)), "trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "net_pips": 0.0, "net_usd": 0.0, "avg_win_pips": 0.0, "avg_loss_pips": 0.0},
            {"mode": "range_fade", "sessions": int(session_mode_counts.get("range_fade", 0)), "trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "net_pips": 0.0, "net_usd": 0.0, "avg_win_pips": 0.0, "avg_loss_pips": 0.0},
            {"mode": "neutral", "sessions": int(session_mode_counts.get("neutral", 0)), "trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "net_pips": 0.0, "net_usd": 0.0, "avg_win_pips": 0.0, "avg_loss_pips": 0.0},
        ]
        results["session_mode_counts"] = dict(session_mode_counts)
        results["by_sl_bucket"] = []
        results["sizing_stats"] = {
            "avg_lot_size": 0.0,
            "max_lot_used": 0.0,
            "total_lot_volume": 0.0,
            "win_bonus_applied_count": int(win_bonus_applied_count),
            "max_consecutive_wins": int(max_consecutive_wins),
            "avg_lot_on_wins": 0.0,
            "avg_lot_on_losses": 0.0,
            "avg_risk_per_trade_usd": 0.0,
            "max_risk_per_trade_usd": 0.0,
            "risk_consistency": 0.0,
        }
        results["regime_distribution"] = dict(regime_distribution)
        results["session_feedback_stops"] = int(sessions_stopped_early)
        results["daily_stop_counts"] = int(len(daily_stop_days))
        results["weekly_stop_counts"] = int(len(weekly_stop_keys))
        results["trail_delayed_count"] = int(trail_delayed_count)
        results["sl_cap_skipped_count"] = int(sl_cap_skipped_count)
        results["exit_reason_counts"] = results.get("exit_reason_counts", {})
        results["exit_reason_counts"]["stale_exit"] = int(results["exit_reason_counts"].get("stale_exit", 0))
        results["exit_reason_counts"]["session_end_full_risk_london"] = 0
        results["exit_reason_counts"]["session_end_full_risk_ny"] = 0
        return results

    tdf["risk_per_trade_usd"] = tdf.apply(
        lambda r: float(r["sl_pips"]) * pip_value_usd_per_lot(float(r["entry_price"])) * float(r["lots"]),
        axis=1,
    )
    bins = [3.0, 5.0, 7.0, 10.000001]
    labels = ["3-5", "5-7", "7-10"]
    tdf["sl_bucket"] = pd.cut(tdf["sl_pips"], bins=bins, labels=labels, right=False)

    by_strength = (
        tdf.groupby("entry_profile", dropna=False)
        .agg(
            trades=("trade_id", "size"),
            wins=("pips", lambda s: int((s > 0).sum())),
            losses=("pips", lambda s: int((s <= 0).sum())),
            win_rate=("pips", lambda s: float((s > 0).mean()) * 100.0),
            net_pips=("pips", "sum"),
            net_usd=("usd", "sum"),
            avg_lot=("lots", "mean"),
        )
        .reset_index()
        .rename(columns={"entry_profile": "trend_strength"})
        .sort_values("trades", ascending=False)
    )
    results["by_trend_strength"] = by_strength.to_dict("records")
    tdf["dual_mode"] = tdf["entry_signal_mode"].apply(lambda v: "range_fade" if str(v) == "range_fade" else "trend")
    mode_rows = []
    for mode_name in ("trend", "range_fade"):
        w = tdf[tdf["dual_mode"] == mode_name]
        trades = int(len(w))
        wins = int((w["pips"] > 0).sum()) if trades else 0
        losses = int((w["pips"] <= 0).sum()) if trades else 0
        win_rate = float((wins / trades) * 100.0) if trades else 0.0
        avg_win = float(w.loc[w["pips"] > 0, "pips"].mean()) if wins else 0.0
        avg_loss = float(w.loc[w["pips"] <= 0, "pips"].mean()) if losses else 0.0
        mode_rows.append(
            {
                "mode": mode_name,
                "sessions": int(session_mode_counts.get(mode_name, 0)),
                "trades": trades,
                "wins": wins,
                "losses": losses,
                "win_rate": win_rate,
                "net_pips": float(w["pips"].sum()) if trades else 0.0,
                "net_usd": float(w["usd"].sum()) if trades else 0.0,
                "avg_win_pips": avg_win,
                "avg_loss_pips": avg_loss,
            }
        )
    mode_rows.append(
        {
            "mode": "neutral",
            "sessions": int(session_mode_counts.get("neutral", 0)),
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "net_pips": 0.0,
            "net_usd": 0.0,
            "avg_win_pips": 0.0,
            "avg_loss_pips": 0.0,
        }
    )
    results["by_mode"] = mode_rows
    results["session_mode_counts"] = dict(session_mode_counts)
    by_sl_bucket = (
        tdf.dropna(subset=["sl_bucket"])
        .groupby("sl_bucket", dropna=False)
        .agg(
            trades=("trade_id", "size"),
            wins=("pips", lambda s: int((s > 0).sum())),
            losses=("pips", lambda s: int((s <= 0).sum())),
            net_pips=("pips", "sum"),
            net_usd=("usd", "sum"),
            avg_lot=("lots", "mean"),
        )
        .reset_index()
        .rename(columns={"sl_bucket": "bucket"})
    )
    results["by_sl_bucket"] = by_sl_bucket.to_dict("records")
    results["sizing_stats"] = {
        "avg_lot_size": round(float(tdf["lots"].mean()), 6),
        "max_lot_used": round(float(tdf["lots"].max()), 6),
        "total_lot_volume": round(float(tdf["lots"].sum()), 6),
        "win_bonus_applied_count": int(win_bonus_applied_count),
        "max_consecutive_wins": int(max_consecutive_wins),
        "avg_lot_on_wins": round(float(tdf.loc[tdf["pips"] > 0, "lots"].mean()), 6) if int((tdf["pips"] > 0).sum()) else 0.0,
        "avg_lot_on_losses": round(float(tdf.loc[tdf["pips"] <= 0, "lots"].mean()), 6) if int((tdf["pips"] <= 0).sum()) else 0.0,
        "avg_risk_per_trade_usd": round(float(tdf["risk_per_trade_usd"].mean()), 6),
        "max_risk_per_trade_usd": round(float(tdf["risk_per_trade_usd"].max()), 6),
        "risk_consistency": round(float(tdf["risk_per_trade_usd"].std(ddof=0)), 6),
    }
    exit_counts = results.get("exit_reason_counts", {})
    exit_counts["stale_exit"] = int(exit_counts.get("stale_exit", 0))
    session_end_mask = tdf["exit_reason"] == "session_end_full_risk"
    exit_counts["session_end_full_risk_london"] = int(((session_end_mask) & (tdf["entry_session"] == "london")).sum())
    exit_counts["session_end_full_risk_ny"] = int(((session_end_mask) & (tdf["entry_session"] == "ny_overlap")).sum())
    results["exit_reason_counts"] = exit_counts
    results["regime_distribution"] = dict(regime_distribution)
    results["session_feedback_stops"] = int(sessions_stopped_early)
    results["daily_stop_counts"] = int(len(daily_stop_days))
    results["weekly_stop_counts"] = int(len(weekly_stop_keys))
    results["trail_delayed_count"] = int(trail_delayed_count)
    results["sl_cap_skipped_count"] = int(sl_cap_skipped_count)
    return results


def run_backtest_swing(args: argparse.Namespace) -> dict:
    m1 = load_m1(args.inputs)
    h4 = resample_ohlc(m1, "4h")
    daily = resample_ohlc(m1, "1D")
    weekly = resample_ohlc(m1, "1W")

    weekly["ema10_w"] = weekly["close"].ewm(span=int(args.swing_weekly_ema), adjust=False).mean()
    weekly["atr14_w"] = compute_atr(weekly, 14)
    weekly["atr14_w_sma20"] = weekly["atr14_w"].rolling(20, min_periods=20).mean()

    daily["ema20_d"] = daily["close"].ewm(span=int(args.swing_daily_ema_fast), adjust=False).mean()
    daily["ema50_d"] = daily["close"].ewm(span=int(args.swing_daily_ema_slow), adjust=False).mean()
    daily["atr14_d"] = compute_atr(daily, 14)
    daily["atr14_d_sma20"] = daily["atr14_d"].rolling(20, min_periods=20).mean()

    h4["ema21_h4"] = h4["close"].ewm(span=int(args.swing_h4_ema), adjust=False).mean()
    h4["ema50_h4"] = h4["close"].ewm(span=int(args.swing_h4_ema_slow), adjust=False).mean()
    h4["atr14_h4"] = compute_atr(h4, 14)

    h4_times = h4["time"].tolist()
    d_times = daily["time"].tolist()
    w_times = weekly["time"].tolist()
    p4 = pd_idx = pw = -1

    open_positions: list[dict] = []
    closed: list[ClosedTrade] = []
    blocked_reasons: dict[str, int] = {}
    trade_id_seq = 0
    max_open_positions = 0

    def add_block(reason: str) -> None:
        blocked_reasons[reason] = blocked_reasons.get(reason, 0) + 1

    def _in_swing_entry_h4_window(h4_ts: pd.Timestamp) -> tuple[bool, str]:
        h = ts_hour(pd.Timestamp(h4_ts))
        if 8.0 <= h < 9.0:
            return True, "london"
        if 14.0 <= h < 15.0:
            return True, "ny_overlap"
        return False, "dead"

    def _weekly_tp2_price(side: str, entry_price: float, pw_now: int) -> float:
        wb = 8
        if pw_now <= 0:
            rr = float(args.swing_tp2_rr)
            return entry_price + rr * PIP_SIZE if side == "buy" else entry_price - rr * PIP_SIZE
        start = max(0, pw_now - wb)
        wwin = weekly.iloc[start:pw_now]
        if wwin.empty:
            rr = float(args.swing_tp2_rr)
            return entry_price + rr * PIP_SIZE if side == "buy" else entry_price - rr * PIP_SIZE
        if side == "buy":
            tgt = float(wwin["high"].max())
            return tgt if tgt > entry_price else entry_price + float(args.swing_tp2_rr) * PIP_SIZE
        tgt = float(wwin["low"].min())
        return tgt if tgt < entry_price else entry_price - float(args.swing_tp2_rr) * PIP_SIZE

    for i in range(len(m1)):
        row = m1.iloc[i]
        ts = pd.Timestamp(row["time"])
        o = float(row["open"])
        h = float(row["high"])
        l = float(row["low"])
        c = float(row["close"])

        spread_pips = compute_spread_pips(i, ts, str(args.spread_mode), float(args.spread_pips), float(args.spread_min_pips), float(args.spread_max_pips))
        half_spread = spread_pips * PIP_SIZE / 2.0
        bid = c - half_spread
        ask = c + half_spread

        old4 = p4
        while p4 + 1 < len(h4_times) and h4_times[p4 + 1] <= ts:
            p4 += 1
        new_h4_bar = p4 != old4
        while pd_idx + 1 < len(d_times) and d_times[pd_idx + 1] <= ts:
            pd_idx += 1
        while pw + 1 < len(w_times) and w_times[pw + 1] <= ts:
            pw += 1

        # Manage open positions on every M1 bar.
        for pos in list(open_positions):
            side = str(pos["side"])
            closed_now = False
            if not bool(pos["tp1_filled"]):
                sl_hit = (l <= float(pos["stop_price"])) if side == "buy" else (h >= float(pos["stop_price"]))
                if sl_hit:
                    exit_px = float(pos["stop_price"])
                    pips = ((exit_px - float(pos["entry_price"])) / PIP_SIZE) if side == "buy" else ((float(pos["entry_price"]) - exit_px) / PIP_SIZE)
                    usd = pips * pip_value_usd_per_lot(exit_px) * float(pos["lots_remaining"])
                    closed.append(
                        ClosedTrade(
                            trade_id=int(pos["trade_id"]),
                            side=side,
                            entry_mode=18,
                            entry_session=str(pos["entry_session"]),
                            entry_time=pd.Timestamp(pos["entry_time"]),
                            exit_time=pd.Timestamp(ts),
                            entry_price=float(pos["entry_price"]),
                            exit_price=exit_px,
                            lots=float(pos["lots_initial"]),
                            sl_pips=float(pos["sl_pips"]),
                            tp1_pips=float(pos["tp1_pips"]),
                            tp2_pips=float(pos["tp2_pips"]),
                            pips=float(pos["realized_pips"] + pips * (float(pos["lots_remaining"]) / max(1e-12, float(pos["lots_initial"])))),
                            usd=float(pos["realized_usd"] + usd),
                            exit_reason="sl",
                            entry_day=str(pd.Timestamp(pos["entry_time"]).date()),
                            entry_regime="swing",
                            entry_profile=str(pos["weekly_bias"]),
                            position_type="Single",
                            entry_signal_mode="swing_trend",
                        )
                    )
                    open_positions.remove(pos)
                    closed_now = True
                else:
                    tp1_hit = (h >= float(pos["tp1_price"])) if side == "buy" else (l <= float(pos["tp1_price"]))
                    if tp1_hit:
                        leg_lots = float(pos["lots_initial"]) * float(args.swing_tp1_close_pct)
                        exit_px = float(pos["tp1_price"])
                        pips = ((exit_px - float(pos["entry_price"])) / PIP_SIZE) if side == "buy" else ((float(pos["entry_price"]) - exit_px) / PIP_SIZE)
                        usd = pips * pip_value_usd_per_lot(exit_px) * leg_lots
                        pos["realized_pips"] += pips * (leg_lots / max(1e-12, float(pos["lots_initial"])))
                        pos["realized_usd"] += usd
                        pos["lots_remaining"] = max(0.0, float(pos["lots_initial"]) - leg_lots)
                        pos["tp1_filled"] = True
                        pos["stop_price"] = float(pos["entry_price"])  # breakeven
            else:
                tp2_hit = (h >= float(pos["tp2_price"])) if side == "buy" else (l <= float(pos["tp2_price"]))
                if tp2_hit:
                    exit_px = float(pos["tp2_price"])
                    pips = ((exit_px - float(pos["entry_price"])) / PIP_SIZE) if side == "buy" else ((float(pos["entry_price"]) - exit_px) / PIP_SIZE)
                    usd = pips * pip_value_usd_per_lot(exit_px) * float(pos["lots_remaining"])
                    closed.append(
                        ClosedTrade(
                            trade_id=int(pos["trade_id"]),
                            side=side,
                            entry_mode=18,
                            entry_session=str(pos["entry_session"]),
                            entry_time=pd.Timestamp(pos["entry_time"]),
                            exit_time=pd.Timestamp(ts),
                            entry_price=float(pos["entry_price"]),
                            exit_price=exit_px,
                            lots=float(pos["lots_initial"]),
                            sl_pips=float(pos["sl_pips"]),
                            tp1_pips=float(pos["tp1_pips"]),
                            tp2_pips=float(pos["tp2_pips"]),
                            pips=float(pos["realized_pips"] + pips * (float(pos["lots_remaining"]) / max(1e-12, float(pos["lots_initial"])))),
                            usd=float(pos["realized_usd"] + usd),
                            exit_reason="tp2_hit",
                            entry_day=str(pd.Timestamp(pos["entry_time"]).date()),
                            entry_regime="swing",
                            entry_profile=str(pos["weekly_bias"]),
                            position_type="Single",
                            entry_signal_mode="swing_trend",
                        )
                    )
                    open_positions.remove(pos)
                    closed_now = True
                else:
                    stop_hit = (l <= float(pos["stop_price"])) if side == "buy" else (h >= float(pos["stop_price"]))
                    if stop_hit:
                        exit_px = float(pos["stop_price"])
                        pips = ((exit_px - float(pos["entry_price"])) / PIP_SIZE) if side == "buy" else ((float(pos["entry_price"]) - exit_px) / PIP_SIZE)
                        usd = pips * pip_value_usd_per_lot(exit_px) * float(pos["lots_remaining"])
                        closed.append(
                            ClosedTrade(
                                trade_id=int(pos["trade_id"]),
                                side=side,
                                entry_mode=18,
                                entry_session=str(pos["entry_session"]),
                                entry_time=pd.Timestamp(pos["entry_time"]),
                                exit_time=pd.Timestamp(ts),
                                entry_price=float(pos["entry_price"]),
                                exit_price=exit_px,
                                lots=float(pos["lots_initial"]),
                                sl_pips=float(pos["sl_pips"]),
                                tp1_pips=float(pos["tp1_pips"]),
                                tp2_pips=float(pos["tp2_pips"]),
                                pips=float(pos["realized_pips"] + pips * (float(pos["lots_remaining"]) / max(1e-12, float(pos["lots_initial"])))),
                                usd=float(pos["realized_usd"] + usd),
                                exit_reason="tp1_then_tp2",
                                entry_day=str(pd.Timestamp(pos["entry_time"]).date()),
                                entry_regime="swing",
                                entry_profile=str(pos["weekly_bias"]),
                                position_type="Single",
                                entry_signal_mode="swing_trend",
                            )
                        )
                        open_positions.remove(pos)
                        closed_now = True

            if closed_now:
                continue

            held_days = (pd.Timestamp(ts) - pd.Timestamp(pos["entry_time"])).total_seconds() / 86400.0
            if held_days >= float(args.swing_max_duration_days):
                exit_px = float(bid if side == "buy" else ask)
                pips = ((exit_px - float(pos["entry_price"])) / PIP_SIZE) if side == "buy" else ((float(pos["entry_price"]) - exit_px) / PIP_SIZE)
                usd = pips * pip_value_usd_per_lot(exit_px) * float(pos["lots_remaining"])
                closed.append(
                    ClosedTrade(
                        trade_id=int(pos["trade_id"]),
                        side=side,
                        entry_mode=18,
                        entry_session=str(pos["entry_session"]),
                        entry_time=pd.Timestamp(pos["entry_time"]),
                        exit_time=pd.Timestamp(ts),
                        entry_price=float(pos["entry_price"]),
                        exit_price=exit_px,
                        lots=float(pos["lots_initial"]),
                        sl_pips=float(pos["sl_pips"]),
                        tp1_pips=float(pos["tp1_pips"]),
                        tp2_pips=float(pos["tp2_pips"]),
                        pips=float(pos["realized_pips"] + pips * (float(pos["lots_remaining"]) / max(1e-12, float(pos["lots_initial"])))),
                        usd=float(pos["realized_usd"] + usd),
                        exit_reason="max_duration_close",
                        entry_day=str(pd.Timestamp(pos["entry_time"]).date()),
                        entry_regime="swing",
                        entry_profile=str(pos["weekly_bias"]),
                        position_type="Single",
                        entry_signal_mode="swing_trend",
                    )
                )
                open_positions.remove(pos)

        max_open_positions = max(max_open_positions, len(open_positions))

        if not bool(getattr(args, "swing_mode_enabled", False)):
            continue
        if not new_h4_bar or p4 < 1 or pd_idx < 3 or pw < 1:
            continue
        if float(spread_pips) > float(args.max_entry_spread_pips):
            add_block("swing_entry_spread_too_high")
            continue
        if len(open_positions) >= int(getattr(args, "v5_max_open", 1)):
            add_block("swing_max_open_cap")
            continue

        h4_ts = pd.Timestamp(h4.iloc[p4]["time"])
        allowed_h4, entry_sess = _in_swing_entry_h4_window(h4_ts)
        if not allowed_h4:
            add_block("swing_session_window")
            continue

        wr = weekly.iloc[pw]
        weekly_close = float(wr["close"])
        weekly_ema = float(wr["ema10_w"])
        if weekly_close > weekly_ema:
            weekly_bias = "buy"
        elif weekly_close < weekly_ema:
            weekly_bias = "sell"
        else:
            add_block("swing_weekly_no_bias")
            continue

        dr_now = daily.iloc[pd_idx]
        dr_ago = daily.iloc[pd_idx - 3]
        d_ema_now = float(dr_now["ema20_d"])
        d_ema_ago = float(dr_ago["ema20_d"])
        daily_slope_ok = d_ema_now > d_ema_ago if weekly_bias == "buy" else d_ema_now < d_ema_ago
        if not daily_slope_ok:
            add_block("swing_daily_slope_mismatch")
            continue
        atr_d = float(dr_now["atr14_d"]) if pd.notna(dr_now["atr14_d"]) else 0.0
        atr_d_sma = float(dr_now["atr14_d_sma20"]) if pd.notna(dr_now["atr14_d_sma20"]) else 0.0
        if atr_d <= 0 or atr_d_sma <= 0 or atr_d <= atr_d_sma * float(args.swing_atr_expansion_ratio):
            add_block("swing_daily_atr_not_expanding")
            continue

        h4_now = h4.iloc[p4]
        h4_prev = h4.iloc[p4 - 1]
        h4_body_pips = abs(float(h4_now["close"]) - float(h4_now["open"])) / PIP_SIZE
        if weekly_bias == "buy":
            h4_entry_ok = (
                float(h4_prev["close"]) <= float(h4_prev["ema21_h4"])
                and float(h4_now["close"]) > float(h4_now["ema21_h4"])
                and float(h4_now["close"]) > float(h4_now["ema50_h4"])
                and float(h4_now["close"]) > float(h4_now["open"])
                and h4_body_pips >= 3.0
            )
        else:
            h4_entry_ok = (
                float(h4_prev["close"]) >= float(h4_prev["ema21_h4"])
                and float(h4_now["close"]) < float(h4_now["ema21_h4"])
                and float(h4_now["close"]) < float(h4_now["ema50_h4"])
                and float(h4_now["close"]) < float(h4_now["open"])
                and h4_body_pips >= 3.0
            )
        if not h4_entry_ok:
            add_block("swing_h4_entry_not_met")
            continue

        entry_price = float(ask if weekly_bias == "buy" else bid)
        lb = int(args.swing_sl_h4_lookback)
        if p4 < lb:
            add_block("swing_sl_lookback_insufficient")
            continue
        h4_sl_win = h4.iloc[p4 - lb : p4]
        if weekly_bias == "buy":
            raw_stop = float(h4_sl_win["low"].min()) - float(args.swing_sl_buffer_pips) * PIP_SIZE
            sl_pips = (entry_price - raw_stop) / PIP_SIZE
        else:
            raw_stop = float(h4_sl_win["high"].max()) + float(args.swing_sl_buffer_pips) * PIP_SIZE
            sl_pips = (raw_stop - entry_price) / PIP_SIZE
        if sl_pips <= 0:
            add_block("swing_invalid_sl")
            continue
        if sl_pips > float(args.swing_sl_cap_pips):
            sl_pips = float(args.swing_sl_cap_pips)
            raw_stop = entry_price - sl_pips * PIP_SIZE if weekly_bias == "buy" else entry_price + sl_pips * PIP_SIZE

        tp1_pips = float(args.swing_tp1_rr) * float(sl_pips)
        tp1_price = entry_price + tp1_pips * PIP_SIZE if weekly_bias == "buy" else entry_price - tp1_pips * PIP_SIZE
        weekly_tp2 = _weekly_tp2_price(weekly_bias, entry_price, pw)
        fallback_tp2 = entry_price + float(args.swing_tp2_rr) * float(sl_pips) * PIP_SIZE if weekly_bias == "buy" else entry_price - float(args.swing_tp2_rr) * float(sl_pips) * PIP_SIZE
        tp2_price = float(weekly_tp2) if ((weekly_bias == "buy" and weekly_tp2 > entry_price) or (weekly_bias == "sell" and weekly_tp2 < entry_price)) else float(fallback_tp2)
        tp2_pips = ((tp2_price - entry_price) / PIP_SIZE) if weekly_bias == "buy" else ((entry_price - tp2_price) / PIP_SIZE)
        if tp2_pips <= 0:
            tp2_pips = float(args.swing_tp2_rr) * float(sl_pips)
            tp2_price = entry_price + tp2_pips * PIP_SIZE if weekly_bias == "buy" else entry_price - tp2_pips * PIP_SIZE

        risk_usd = float(args.swing_account_size) * (float(args.swing_risk_pct) / 100.0)
        lots = risk_usd / max(1e-9, float(sl_pips) * pip_value_usd_per_lot(entry_price))
        lots = max(0.01, float(lots))

        trade_id_seq += 1
        open_positions.append(
            {
                "trade_id": int(trade_id_seq),
                "side": str(weekly_bias),
                "entry_time": pd.Timestamp(ts),
                "entry_price": float(entry_price),
                "entry_session": str(entry_sess),
                "weekly_bias": str(weekly_bias),
                "lots_initial": float(lots),
                "lots_remaining": float(lots),
                "stop_price": float(raw_stop),
                "tp1_price": float(tp1_price),
                "tp2_price": float(tp2_price),
                "sl_pips": float(sl_pips),
                "tp1_pips": float(tp1_pips),
                "tp2_pips": float(tp2_pips),
                "tp1_filled": False,
                "realized_pips": 0.0,
                "realized_usd": 0.0,
            }
        )
        max_open_positions = max(max_open_positions, len(open_positions))

    # Close remaining at last price.
    if open_positions:
        last = m1.iloc[-1]
        ts_last = pd.Timestamp(last["time"])
        cl_last = float(last["close"])
        sp_last = compute_spread_pips(len(m1) - 1, ts_last, str(args.spread_mode), float(args.spread_pips), float(args.spread_min_pips), float(args.spread_max_pips))
        hs_last = sp_last * PIP_SIZE / 2.0
        bid_last = cl_last - hs_last
        ask_last = cl_last + hs_last
        for pos in list(open_positions):
            side = str(pos["side"])
            exit_px = float(bid_last if side == "buy" else ask_last)
            pips = ((exit_px - float(pos["entry_price"])) / PIP_SIZE) if side == "buy" else ((float(pos["entry_price"]) - exit_px) / PIP_SIZE)
            usd = pips * pip_value_usd_per_lot(exit_px) * float(pos["lots_remaining"])
            closed.append(
                ClosedTrade(
                    trade_id=int(pos["trade_id"]),
                    side=side,
                    entry_mode=18,
                    entry_session=str(pos["entry_session"]),
                    entry_time=pd.Timestamp(pos["entry_time"]),
                    exit_time=pd.Timestamp(ts_last),
                    entry_price=float(pos["entry_price"]),
                    exit_price=exit_px,
                    lots=float(pos["lots_initial"]),
                    sl_pips=float(pos["sl_pips"]),
                    tp1_pips=float(pos["tp1_pips"]),
                    tp2_pips=float(pos["tp2_pips"]),
                    pips=float(pos["realized_pips"] + pips * (float(pos["lots_remaining"]) / max(1e-12, float(pos["lots_initial"])))),
                    usd=float(pos["realized_usd"] + usd),
                    exit_reason="eod",
                    entry_day=str(pd.Timestamp(pos["entry_time"]).date()),
                    entry_regime="swing",
                    entry_profile=str(pos["weekly_bias"]),
                    position_type="Single",
                    entry_signal_mode="swing_trend",
                )
            )

    results = _v3_metrics_from_closed(closed, max_open_positions, blocked_reasons)
    tdf = pd.DataFrame([x.__dict__ for x in closed])
    if tdf.empty:
        results["monthly"] = []
        results["swing_stats"] = {
            "avg_hold_hours": 0.0,
            "avg_hold_hours_wins": 0.0,
            "avg_hold_hours_losses": 0.0,
            "avg_win_loss_ratio_abs": 0.0,
            "max_simultaneous_open_positions": int(max_open_positions),
            "monthly_bias_distribution": [],
        }
        return results

    tdf["entry_time"] = pd.to_datetime(tdf["entry_time"], utc=True)
    tdf["exit_time"] = pd.to_datetime(tdf["exit_time"], utc=True)
    tdf["month"] = tdf["entry_time"].dt.strftime("%Y-%m")
    tdf["hold_hours"] = (tdf["exit_time"] - tdf["entry_time"]).dt.total_seconds() / 3600.0
    avg_win = float(tdf.loc[tdf["pips"] > 0, "pips"].mean()) if int((tdf["pips"] > 0).sum()) else 0.0
    avg_loss_abs = abs(float(tdf.loc[tdf["pips"] < 0, "pips"].mean())) if int((tdf["pips"] < 0).sum()) else 0.0
    ratio = (avg_win / avg_loss_abs) if avg_loss_abs > 0 else 0.0

    monthly_rows = []
    for m, g in tdf.groupby("month"):
        wins = int((g["pips"] > 0).sum())
        losses = int((g["pips"] <= 0).sum())
        wr = (wins / max(1, len(g))) * 100.0
        gp = float(g.loc[g["pips"] > 0, "pips"].sum())
        gl = abs(float(g.loc[g["pips"] < 0, "pips"].sum()))
        pf = (gp / gl) if gl > 0 else None
        monthly_rows.append(
            {
                "month": str(m),
                "trades": int(len(g)),
                "wins": wins,
                "losses": losses,
                "win_rate": round(float(wr), 3),
                "net_pips": round(float(g["pips"].sum()), 3),
                "net_usd": round(float(g["usd"].sum()), 3),
                "profit_factor": round(float(pf), 4) if pf is not None else None,
            }
        )
    monthly_rows = sorted(monthly_rows, key=lambda x: x["month"])

    bias_dist = (
        tdf.groupby(["month", "side"], dropna=False)["trade_id"]
        .count()
        .reset_index()
        .rename(columns={"trade_id": "trades", "side": "bias_side"})
        .to_dict("records")
    )
    results["monthly"] = monthly_rows
    results["swing_stats"] = {
        "avg_hold_hours": round(float(tdf["hold_hours"].mean()), 3),
        "avg_hold_hours_wins": round(float(tdf.loc[tdf["pips"] > 0, "hold_hours"].mean()), 3) if int((tdf["pips"] > 0).sum()) else 0.0,
        "avg_hold_hours_losses": round(float(tdf.loc[tdf["pips"] <= 0, "hold_hours"].mean()), 3) if int((tdf["pips"] <= 0).sum()) else 0.0,
        "avg_win_loss_ratio_abs": round(float(ratio), 3),
        "max_simultaneous_open_positions": int(max_open_positions),
        "monthly_bias_distribution": bias_dist,
    }
    return results


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if float(args.spread_min_pips) < 1.0:
        args.spread_min_pips = 1.0
    if float(args.spread_pips) < 1.0:
        args.spread_pips = 1.0
    if float(args.spread_max_pips) < float(args.spread_min_pips):
        args.spread_max_pips = float(args.spread_min_pips)

    m1 = load_m1(args.inputs)
    spreads = [
        compute_spread_pips(i, pd.Timestamp(m1.iloc[i]["time"]), str(args.spread_mode), float(args.spread_pips), float(args.spread_min_pips), float(args.spread_max_pips))
        for i in range(len(m1))
    ]
    if str(args.mode) == "swing":
        results = run_backtest_swing(args)
    elif str(args.mode) == "combined":
        res_session = run_backtest_v5(args) if str(args.version) == "v5" else run_backtest(args)
        res_swing = run_backtest_swing(args)
        c1 = pd.DataFrame(res_session.get("closed_trades", []))
        c2 = pd.DataFrame(res_swing.get("closed_trades", []))
        c = pd.concat([c1, c2], ignore_index=True) if not c1.empty or not c2.empty else pd.DataFrame()
        if c.empty:
            combined = _v3_metrics_from_closed([], 0, {})
        else:
            combined = _v3_metrics_from_closed(
                [
                    ClosedTrade(
                        trade_id=int(r["trade_id"]),
                        side=str(r["side"]),
                        entry_mode=int(r["entry_mode"]),
                        entry_session=str(r["entry_session"]),
                        entry_time=pd.Timestamp(r["entry_time"]),
                        exit_time=pd.Timestamp(r["exit_time"]),
                        entry_price=float(r["entry_price"]),
                        exit_price=float(r["exit_price"]),
                        lots=float(r.get("lots", 0.0)),
                        sl_pips=float(r.get("sl_pips", 0.0)),
                        tp1_pips=float(r.get("tp1_pips", 0.0)),
                        tp2_pips=float(r.get("tp2_pips", 0.0)),
                        pips=float(r["pips"]),
                        usd=float(r["usd"]),
                        exit_reason=str(r.get("exit_reason", "unknown")),
                        entry_day=str(r.get("entry_day", "")),
                        entry_regime=str(r.get("entry_regime")) if pd.notna(r.get("entry_regime")) else None,
                        entry_profile=str(r.get("entry_profile")) if pd.notna(r.get("entry_profile")) else None,
                        position_type=str(r.get("position_type")) if pd.notna(r.get("position_type")) else None,
                        entry_signal_mode=str(r.get("entry_signal_mode")) if pd.notna(r.get("entry_signal_mode")) else None,
                    )
                    for _, r in c.iterrows()
                ],
                max(
                    int(res_session.get("summary", {}).get("max_open_positions", 0)),
                    int(res_swing.get("summary", {}).get("max_open_positions", 0)),
                ),
                {},
            )
        combined["components"] = {"session": res_session.get("summary", {}), "swing": res_swing.get("summary", {})}
        results = combined
    elif str(args.version) == "v2":
        results = run_backtest_v2(args)
    elif str(args.version) == "v3":
        results = run_backtest_v3(args)
    elif str(args.version) == "v4":
        results = run_backtest_v4(args)
    elif str(args.version) == "v5":
        results = run_backtest_v5(args)
    else:
        results = run_backtest(args)

    report = {
        "config": {
            "version": str(args.version),
            "mode": str(args.mode),
            "inputs": args.inputs,
            "bars_m1": int(len(m1)),
            "start_utc": str(m1["time"].min()),
            "end_utc": str(m1["time"].max()),
            "pip_size": float(PIP_SIZE),
            "spread_mode": str(args.spread_mode),
            "spread_avg_target_pips": float(args.spread_pips),
            "spread_avg_actual_pips": round(float(sum(spreads) / max(1, len(spreads))), 6),
            "spread_min_pips": float(args.spread_min_pips),
            "spread_max_pips": float(args.spread_max_pips),
            "max_entry_spread_pips": float(args.max_entry_spread_pips),
            "base_lot": float(args.base_lot),
            "trend": {
                "h1_ema_fast": int(args.h1_ema_fast),
                "h1_ema_slow": int(args.h1_ema_slow),
                "h1_allow_slope_direction": bool(args.h1_allow_slope_direction),
                "h1_slope_bars": int(args.h1_slope_bars),
            },
            "sessions_utc": {
                "london": [float(args.london_start), float(args.london_end)],
                "ny_overlap": [float(args.ny_start), float(args.ny_end)],
                "entry_sessions_mode": str(args.sessions),
            },
            "mode1": {
                "range_minutes": int(args.mode1_range_minutes),
                "range_min_pips": float(args.range_min_pips),
                "range_max_pips": float(args.range_max_pips),
                "breakout_margin_pips": float(args.breakout_margin_pips),
                "mode1_only": bool(args.mode1_only),
            },
            "mode2": {
                "impulse_lookback_bars": int(args.mode2_impulse_lookback_bars),
                "impulse_min_body_pips": float(args.mode2_impulse_min_body_pips),
                "pullback_lookback_bars": int(args.mode2_pullback_lookback_bars),
                "cooldown_bars": int(args.mode2_cooldown_bars),
            },
            "risk": {
                "sl_floor_pips": float(args.sl_floor_pips),
                "sl_buffer_pips": float(args.sl_buffer_pips),
                "sl_mode1_max_pips": float(args.sl_mode1_max_pips),
                "sl_mode2_lookback_bars": int(args.sl_mode2_lookback_bars),
                "sl_mode2_max_pips": float(args.sl_mode2_max_pips),
                "tp1_multiplier": float(args.tp1_multiplier),
                "tp2_multiplier": float(args.tp2_multiplier),
                "tp1_close_fraction": float(args.tp1_close_fraction),
                "be_offset_pips": float(args.be_offset_pips),
                "trail_buffer_pips": float(args.trail_buffer_pips),
            },
            "caps": {
                "max_open_positions": int(args.max_open_positions),
                "max_entries_per_day": int(args.max_entries_per_day),
                "loss_after_first_full_sl_lot_mult": float(args.loss_after_first_full_sl_lot_mult),
            },
            "v2": {
                "ny_only_entries": True,
                "max_entries_per_day": int(args.v2_max_entries_per_day),
                "m5_ema_fast": int(args.v2_m5_ema_fast),
                "m5_ema_slow": int(args.v2_m5_ema_slow),
                "impulse_lookback_bars": int(args.v2_impulse_lookback_bars),
                "impulse_min_body_pips": float(args.v2_impulse_min_body_pips),
                "pullback_lookback_bars": int(args.v2_pullback_lookback_bars),
                "m1_ema_fast": int(args.v2_m1_ema_fast),
                "m1_ema_slow": int(args.v2_m1_ema_slow),
                "cooldown_win_bars": int(args.v2_cooldown_win_bars),
                "cooldown_loss_bars": int(args.v2_cooldown_loss_bars),
                "cooldown_scratch_bars": int(args.v2_cooldown_scratch_bars),
                "scratch_threshold_pips": float(args.v2_scratch_threshold_pips),
                "sl_source": str(args.v2_sl_source),
                "sl_lookback_bars": int(args.v2_sl_lookback_bars),
                "sl_lookback_m5_bars": int(args.v2_sl_lookback_m5_bars),
                "sl_buffer_pips": float(args.v2_sl_buffer_pips),
                "sl_max_pips": float(args.v2_sl_max_pips),
                "sl_floor_pips": float(args.v2_sl_floor_pips),
                "tp1_multiplier": float(args.v2_tp1_multiplier),
                "tp2_multiplier": float(args.v2_tp2_multiplier),
                "tp1_close_pct": float(args.v2_tp1_close_pct),
                "be_offset_pips": float(args.v2_be_offset_pips),
                "trail_buffer_pips": float(args.v2_trail_buffer_pips),
                "trail_source": str(args.v2_trail_source),
                "no_reentry_same_m5_after_loss": bool(args.v2_no_reentry_same_m5_after_loss),
            },
            "v3": {
                "atr_period": int(args.v3_atr_period),
                "atr_sma_period": int(args.v3_atr_sma_period),
                "atr_high_ratio": float(args.v3_atr_high_ratio),
                "atr_low_ratio": float(args.v3_atr_low_ratio),
                "slope_bars": int(args.v3_slope_bars),
                "slope_trending_threshold": float(args.v3_slope_trending_threshold),
                "momentum_cooldown_win": int(args.v3_momentum_cooldown_win),
                "grind_cooldown_win": int(args.v3_grind_cooldown_win),
                "momentum_cooldown_loss": int(args.v3_momentum_cooldown_loss),
                "grind_cooldown_loss": int(args.v3_grind_cooldown_loss),
                "momentum_cooldown_scratch": int(args.v3_momentum_cooldown_scratch),
                "grind_cooldown_scratch": int(args.v3_grind_cooldown_scratch),
                "scratch_threshold_pips": float(args.v3_scratch_threshold_pips),
                "momentum_tp1_mult": float(args.v3_momentum_tp1_mult),
                "grind_tp1_mult": float(args.v3_grind_tp1_mult),
                "momentum_tp2_mult": float(args.v3_momentum_tp2_mult),
                "grind_tp2_mult": float(args.v3_grind_tp2_mult),
                "momentum_trail_buffer": float(args.v3_momentum_trail_buffer),
                "grind_trail_buffer": float(args.v3_grind_trail_buffer),
                "momentum_sl_max": float(args.v3_momentum_sl_max),
                "grind_sl_max": float(args.v3_grind_sl_max),
                "sl_floor": float(args.v3_sl_floor),
                "session_feedback": bool(args.v3_session_feedback),
                "burst_lookback": int(args.v3_burst_lookback),
                "burst_min_bars": int(args.v3_burst_min_bars),
                "doji_pips": float(args.v3_doji_pips),
                "momentum_allow_simple_entry": bool(args.v3_momentum_allow_simple_entry),
                "m5_ema_fast": int(args.v3_m5_ema_fast),
                "m5_ema_slow": int(args.v3_m5_ema_slow),
                "impulse_lookback_bars": int(args.v3_impulse_lookback_bars),
                "impulse_min_body_pips": float(args.v3_impulse_min_body_pips),
                "pullback_lookback_bars": int(args.v3_pullback_lookback_bars),
                "m1_ema_fast": int(args.v3_m1_ema_fast),
                "m1_ema_slow": int(args.v3_m1_ema_slow),
                "sl_lookback_m5_bars": int(args.v3_sl_lookback_m5_bars),
                "sl_buffer_pips": float(args.v3_sl_buffer_pips),
                "tp1_close_pct": float(args.v3_tp1_close_pct),
                "be_offset_pips": float(args.v3_be_offset_pips),
                "adaptive_lots": bool(args.v3_adaptive_lots),
                "base_lot": float(args.v3_base_lot),
                "lot_momentum_mult": float(args.v3_lot_momentum_mult),
                "lot_grind_mult": float(args.v3_lot_grind_mult),
                "lot_hot_mult": float(args.v3_lot_hot_mult),
                "max_entries_per_day": int(args.v3_max_entries_per_day),
                "london_in_momentum": bool(args.v3_london_in_momentum),
            },
            "v4": {
                "sessions": str(args.v4_sessions),
                "max_open": int(args.v4_max_open),
                "max_entries_day": int(args.v4_max_entries_day),
                "m5_ema_fast": int(args.v4_m5_ema_fast),
                "m5_ema_slow": int(args.v4_m5_ema_slow),
                "m1_ema_fast": int(args.v4_m1_ema_fast),
                "m1_ema_slow": int(args.v4_m1_ema_slow),
                "probe_lot": float(args.v4_probe_lot),
                "press_lot": float(args.v4_press_lot),
                "recovery_lot": float(args.v4_recovery_lot),
                "sl_lookback": int(args.v4_sl_lookback),
                "sl_buffer": float(args.v4_sl_buffer),
                "sl_max": float(args.v4_sl_max),
                "sl_floor": float(args.v4_sl_floor),
                "tp1_mult": float(args.v4_tp1_mult),
                "tp2_mult": float(args.v4_tp2_mult),
                "tp1_close_pct": float(args.v4_tp1_close_pct),
                "be_offset": float(args.v4_be_offset),
                "trail_buffer": float(args.v4_trail_buffer),
                "cooldown_win": int(args.v4_cooldown_win),
                "cooldown_loss": int(args.v4_cooldown_loss),
                "cooldown_scratch": int(args.v4_cooldown_scratch),
                "scratch_threshold": float(args.v4_scratch_threshold),
                "session_stop_losses": int(args.v4_session_stop_losses),
                "close_full_risk_at_session_end": bool(args.v4_close_full_risk_at_session_end),
            },
            "v5": {
                "sessions": str(args.v5_sessions),
                "max_entries_day": int(args.v5_max_entries_day),
                "max_open": int(args.v5_max_open),
                "m5_ema_fast": int(args.v5_m5_ema_fast),
                "m5_ema_slow": int(args.v5_m5_ema_slow),
                "h4_adx_min": float(args.v5_h4_adx_min),
                "h1_slope_min": float(args.v5_h1_slope_min),
                "h1_slope_consistent_bars": int(args.v5_h1_slope_consistent_bars),
                "session_range_cap_pips": float(args.v5_session_range_cap_pips),
                "adx_min": float(args.v5_adx_min),
                "tp_atr_scale": float(args.v5_tp_atr_scale),
                "sl_atr_scale": float(args.v5_sl_atr_scale),
                "slope_bars": int(args.v5_slope_bars),
                "strong_slope": float(args.v5_strong_slope),
                "weak_slope": float(args.v5_weak_slope),
                "skip_weak": bool(args.v5_skip_weak),
                "skip_normal": bool(args.v5_skip_normal),
                "strength_allow": str(args.v5_strength_allow),
                "ny_strength_allow": str(args.v5_ny_strength_allow),
                "london_strength_allow": str(args.v5_london_strength_allow),
                "london_strong_slope": float(args.v5_london_strong_slope),
                "london_max_entries": int(args.v5_london_max_entries),
                "london_confirm_bars": int(args.v5_london_confirm_bars),
                "london_min_body_pips": float(args.v5_london_min_body_pips),
                "tokyo_enabled": bool(args.v5_tokyo_enabled),
                "tokyo_start": float(args.v5_tokyo_start),
                "tokyo_end": float(args.v5_tokyo_end),
                "tokyo_strong_slope": float(args.v5_tokyo_strong_slope),
                "tokyo_strength_allow": str(args.v5_tokyo_strength_allow),
                "tokyo_max_entries": int(args.v5_tokyo_max_entries),
                "tokyo_confirm_bars": int(args.v5_tokyo_confirm_bars),
                "tokyo_sl_floor_pips": float(args.v5_tokyo_sl_floor_pips),
                "tokyo_sl_cap_pips": float(args.v5_tokyo_sl_cap_pips),
                "tokyo_tp1_mult": float(args.v5_tokyo_tp1_mult),
                "tokyo_tp2_mult": float(args.v5_tokyo_tp2_mult),
                "tokyo_tp1_close_pct": float(args.v5_tokyo_tp1_close_pct),
                "tokyo_cutoff_minutes": int(args.v5_tokyo_cutoff_minutes),
                "tokyo_trail_buffer": float(args.v5_tokyo_trail_buffer),
                "tokyo_require_h4_trend": bool(args.v5_tokyo_require_h4_trend),
                "london_allow_strength": str(args.v5_london_allow_strength) if getattr(args, "v5_london_allow_strength", None) is not None else None,
                "allow_normal_plus": bool(args.v5_allow_normal_plus),
                "normalplus_atr_min_pips": float(args.v5_normalplus_atr_min_pips),
                "normalplus_slope_min": float(args.v5_normalplus_slope_min),
                "entry_min_body_pips": float(args.v5_entry_min_body_pips),
                "session_entry_cutoff_minutes": int(args.v5_session_entry_cutoff_minutes),
                "session_range_cap_pips": float(args.v5_session_range_cap_pips),
                "trail_start_after_tp1_mult": float(args.v5_trail_start_after_tp1_mult),
                "max_entry_spread_pips": float(args.max_entry_spread_pips),
                "london_start_hour": float(args.v5_london_start_hour) if getattr(args, "v5_london_start_hour", None) is not None else float(args.london_start),
                "london_active_start": float(args.v5_london_active_start),
                "london_active_end": float(args.v5_london_active_end),
                "sizing_mode": str(args.v5_sizing_mode),
                "risk_per_trade_pct": float(args.v5_risk_per_trade_pct),
                "account_size": float(args.v5_account_size),
                "rp_london_mult": float(args.v5_rp_london_mult),
                "rp_ny_mult": float(args.v5_rp_ny_mult),
                "rp_tokyo_mult": float(args.v5_rp_tokyo_mult),
                "rp_strong_mult": float(args.v5_rp_strong_mult),
                "rp_normal_mult": float(args.v5_rp_normal_mult),
                "rp_win_bonus_pct": float(args.v5_rp_win_bonus_pct),
                "rp_max_lot_mult": float(args.v5_rp_max_lot_mult),
                "rp_min_lot": float(args.v5_rp_min_lot),
                "rp_max_lot": float(args.v5_rp_max_lot),
                "hybrid_strong_boost": float(args.v5_hybrid_strong_boost),
                "hybrid_normal_boost": float(args.v5_hybrid_normal_boost),
                "hybrid_london_boost": float(args.v5_hybrid_london_boost),
                "hybrid_ny_boost": float(args.v5_hybrid_ny_boost),
                "m1_ema_fast": int(args.v5_m1_ema_fast),
                "m1_ema_slow": int(args.v5_m1_ema_slow),
                "base_lot": float(args.v5_base_lot),
                "london_size_mult": float(args.v5_london_size_mult),
                "weak_size_mult": float(args.v5_weak_size_mult),
                "normal_size_mult": float(args.v5_normal_size_mult),
                "strong_size_mult": float(args.v5_strong_size_mult),
                "win_bonus_per_step": float(args.v5_win_bonus_per_step),
                "win_streak_scope": str(args.v5_win_streak_scope),
                "max_lot": float(args.v5_max_lot),
                "daily_loss_limit_pips": float(args.v5_daily_loss_limit_pips),
                "daily_loss_limit_usd": float(args.v5_daily_loss_limit_usd),
                "weekly_loss_limit_pips": float(args.v5_weekly_loss_limit_pips),
                "ny_start_delay_minutes": int(args.v5_ny_start_delay_minutes),
                "strong_tp1": float(args.v5_strong_tp1),
                "strong_tp2": float(args.v5_strong_tp2),
                "london_strong_tp1": float(args.v5_london_strong_tp1),
                "london_strong_tp2": float(args.v5_london_strong_tp2),
                "london_tp1_close_pct": float(args.v5_london_tp1_close_pct),
                "normal_tp1": float(args.v5_normal_tp1),
                "normal_tp2": float(args.v5_normal_tp2),
                "weak_tp1": float(args.v5_weak_tp1),
                "weak_tp2": float(args.v5_weak_tp2),
                "strong_trail_buffer": float(args.v5_strong_trail_buffer),
                "strong_trail_buffer_pips": float(args.v5_strong_trail_buffer_pips),
                "london_trail_buffer": float(args.v5_london_trail_buffer),
                "strong_trail_start_threshold": float(args.v5_strong_trail_start_threshold),
                "normal_trail_buffer": float(args.v5_normal_trail_buffer),
                "weak_trail_buffer": float(args.v5_weak_trail_buffer),
                "strong_trail_ema": int(args.v5_strong_trail_ema),
                "normal_trail_ema": int(args.v5_normal_trail_ema),
                "strong_tp1_close_pct": float(args.v5_strong_tp1_close_pct),
                "normal_tp1_close_pct": float(args.v5_normal_tp1_close_pct),
                "weak_tp1_close_pct": float(args.v5_weak_tp1_close_pct),
                "be_offset": float(args.v5_be_offset),
                "sl_lookback": int(args.v5_sl_lookback),
                "sl_buffer": float(args.v5_sl_buffer),
                "sl_floor_pips": float(args.v5_sl_floor_pips),
                "sl_cap_pips": float(args.v5_sl_cap_pips),
                "sl_max": float(args.v5_sl_max),
                "sl_floor": float(args.v5_sl_floor),
                "cooldown_win": int(args.v5_cooldown_win),
                "cooldown_loss": int(args.v5_cooldown_loss),
                "cooldown_scratch": int(args.v5_cooldown_scratch),
                "scratch_threshold": float(args.v5_scratch_threshold),
                "session_stop_losses": int(args.v5_session_stop_losses),
                "close_full_risk_at_session_end": bool(args.v5_close_full_risk_at_session_end),
                "stale_exit_enabled": bool(args.v5_stale_exit_enabled),
                "stale_exit_bars": int(args.v5_stale_exit_bars),
                "stale_exit_underwater_pct": float(args.v5_stale_exit_underwater_pct),
                "breakout_entry_enabled": bool(args.v5_breakout_entry_enabled),
                "breakout_min_body_pips": float(args.v5_breakout_min_body_pips),
                "breakout_slope_min": float(args.v5_breakout_slope_min),
                "breakout_sl_lookback": int(args.v5_breakout_sl_lookback),
                "atr_min_entry_pips": float(args.v5_atr_min_entry_pips),
                "rolling_wr_lookback": int(args.v5_rolling_wr_lookback),
                "rolling_wr_min": float(args.v5_rolling_wr_min),
                "rolling_wr_pause_bars": int(args.v5_rolling_wr_pause_bars),
                "wr_scale_lookback": int(args.v5_wr_scale_lookback),
                "wr_scale_min_trades": int(args.v5_wr_scale_min_trades),
                "wr_scale_high": float(args.v5_wr_scale_high),
                "wr_scale_mid": float(args.v5_wr_scale_mid),
                "wr_scale_low": float(args.v5_wr_scale_low),
                "wr_scale_full": float(args.v5_wr_scale_full),
                "wr_scale_reduced": float(args.v5_wr_scale_reduced),
                "wr_scale_minimal": float(args.v5_wr_scale_minimal),
                "wr_scale_survival": float(args.v5_wr_scale_survival),
                "wr_scale_boost": float(args.v5_wr_scale_boost),
                "wr_scale_boost_min": float(args.v5_wr_scale_boost_min),
                "max_open_boost": int(args.v5_max_open_boost),
                "wr_open_boost_min": float(args.v5_wr_open_boost_min),
                "conviction_h1_sep_pips": float(args.v5_conviction_h1_sep_pips),
                "conviction_size_mult": float(args.v5_conviction_size_mult),
                "conviction_wr_gate": float(args.v5_conviction_wr_gate),
                "ny_normal_conviction_override": bool(args.v5_ny_normal_conviction_override),
                "london_orb_enabled": bool(args.v5_london_orb_enabled),
                "london_orb_fallback_pullback": bool(args.v5_london_orb_fallback_pullback),
                "orb_min_range_pips": float(args.v5_orb_min_range_pips),
                "orb_max_range_pips": float(args.v5_orb_max_range_pips),
                "orb_tp1_mult": float(args.v5_orb_tp1_mult),
                "orb_tp2_mult": float(args.v5_orb_tp2_mult),
                "orb_sl_buffer_pips": float(args.v5_orb_sl_buffer_pips),
                "orb_sl_cap_pips": float(args.v5_orb_sl_cap_pips),
                "orb_min_body_pips": float(args.v5_orb_min_body_pips),
                "dual_mode_enabled": bool(args.v5_dual_mode_enabled),
                "range_fade_enabled": bool(args.v5_range_fade_enabled),
                "trend_mode_efficiency_min": float(args.v5_trend_mode_efficiency_min),
                "range_mode_efficiency_max": float(args.v5_range_mode_efficiency_max),
                "range_fade_pips": float(args.v5_range_fade_pips),
                "range_fade_sl_pips": float(args.v5_range_fade_sl_pips),
                "range_fade_tp_pips": float(args.v5_range_fade_tp_pips),
                "skip_months": str(args.v5_skip_months),
                "thin_month": int(args.v5_thin_month),
                "thin_start_day": int(args.v5_thin_start_day),
                "thin_scale": float(args.v5_thin_scale),
            },
            "swing": {
                "enabled": bool(args.swing_mode_enabled),
                "weekly_ema": int(args.swing_weekly_ema),
                "daily_ema_fast": int(args.swing_daily_ema_fast),
                "daily_ema_slow": int(args.swing_daily_ema_slow),
                "h4_ema": int(args.swing_h4_ema),
                "h4_ema_slow": int(args.swing_h4_ema_slow),
                "atr_expansion_ratio": float(args.swing_atr_expansion_ratio),
                "sl_h4_lookback": int(args.swing_sl_h4_lookback),
                "sl_buffer_pips": float(args.swing_sl_buffer_pips),
                "sl_cap_pips": float(args.swing_sl_cap_pips),
                "tp1_rr": float(args.swing_tp1_rr),
                "tp2_rr": float(args.swing_tp2_rr),
                "tp1_close_pct": float(args.swing_tp1_close_pct),
                "max_duration_days": int(args.swing_max_duration_days),
                "risk_pct": float(args.swing_risk_pct),
                "account_size": float(args.swing_account_size),
            },
        },
        "results": results,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")

    s = report["results"]["summary"]
    print(
        f"Session Momentum complete | trades={s['trades']} win_rate={s['win_rate']}% "
        f"net_pips={s['net_pips']} net_usd={s['net_usd']} maxDD_usd={s['max_drawdown_usd']}"
    )
    print(f"Wrote report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
