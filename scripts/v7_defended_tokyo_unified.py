"""
v7_defended_tokyo_unified.py

Per-bar Tokyo V14 evaluator for the bar-by-bar backtest.
Ports session logic from scripts/backtest_tokyo_meanrev.py into a callable-per-bar
interface aligned with scripts/v7_defended_london_unified.py.

Source references are cited in docstrings (backtest_tokyo_meanrev.py line ranges).
"""
from __future__ import annotations

import json
import math
import sys
from bisect import bisect_left, bisect_right, insort
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.v14_entry_evaluator import evaluate_v14_entry_signal

from scripts import backtest_tokyo_meanrev as tokyo_bt
from scripts.backtest_tokyo_meanrev import Position, calc_leg_usd_pips

PIP_SIZE = tokyo_bt.PIP_SIZE
TOKYO_TZ = tokyo_bt.TOKYO_TZ

# Match unified backtest margin convention (scripts/backtest_v7_defended_bar_by_bar.py margin_lev=33.3).
UNIFIED_MARGIN_LEVERAGE = 33.3

DEFAULT_TOKYO_CFG = ROOT / "research_out" / "tokyo_optimized_v14_config.json"


def init_tokyo_config(path: Optional[Path] = None) -> dict:
    """
    Return Tokyo V14 config used by the shadow backtest (JSON on disk).

    Runtime values come from research_out/tokyo_optimized_v14_config.json.
    Shadow defaults when keys are missing are documented in
    scripts/backtest_tokyo_meanrev.py ``run_one`` (e.g. session_start_utc default
    "15:00" / session_end_utc "00:00" at lines 525-526; execution spread defaults
    lines 622-627; trade_management / exit_rules defaults lines 577-621).
    """
    p = Path(path) if path is not None else DEFAULT_TOKYO_CFG
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def _apply_atr_percentile_rank(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """scripts/backtest_tokyo_meanrev.py run_one lines 591-597, 772-791."""
    out = df.copy()
    atr_pct_cfg = cfg["indicators"]["atr"].get("percentile_filter", {})
    atr_pct_enabled = bool(atr_pct_cfg.get("enabled", False))
    atr_pct_lookback = max(10, int(atr_pct_cfg.get("lookback_bars", 150)))
    atr_pct_max = float(atr_pct_cfg.get("max_percentile", 0.67))
    atr_pct_min_obs = max(20, int(atr_pct_cfg.get("min_observations", min(50, atr_pct_lookback))))
    if atr_pct_enabled:
        atr_vals = pd.to_numeric(out["atr_m15"], errors="coerce").to_numpy(dtype=float)
        atr_pct_rank = np.full(len(atr_vals), np.nan, dtype=float)
        atr_window: deque[float] = deque()
        atr_sorted: list[float] = []
        for idx, aval in enumerate(atr_vals):
            if np.isfinite(aval):
                aval_f = float(aval)
                atr_window.append(aval_f)
                insort(atr_sorted, aval_f)
                if len(atr_window) > atr_pct_lookback:
                    old = float(atr_window.popleft())
                    rm_idx = bisect_left(atr_sorted, old)
                    if 0 <= rm_idx < len(atr_sorted):
                        atr_sorted.pop(rm_idx)
                if len(atr_sorted) >= atr_pct_min_obs:
                    atr_pct_rank[idx] = float(bisect_right(atr_sorted, aval_f) / len(atr_sorted))
        out["atr_m15_percentile_rank"] = atr_pct_rank
    else:
        out["atr_m15_percentile_rank"] = np.nan
    return out


def _build_m5_indicator_frame(m5: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Port M5 block from scripts/backtest_tokyo_meanrev.py add_indicators lines 243-339."""
    m5 = m5.copy()
    bb_p = int(cfg["indicators"]["bollinger_bands"]["period"])
    bb_std = float(cfg["indicators"]["bollinger_bands"]["std_dev"])
    rsi_p = int(cfg["indicators"]["rsi"]["period"])
    bb_pct_window = int(cfg["indicators"]["bb_width_regime_filter"].get("percentile_lookback_m5_bars", 100))
    bb_pct_cutoff = float(cfg["indicators"]["bb_width_regime_filter"].get("ranging_percentile", 0.80))
    rg = cfg.get("regime_gate", {})
    bb_rank_lookback = int(rg.get("bb_width_lookback", 200))
    bb_high_q = float(rg.get("bb_width_high_pctile", 70)) / 100.0
    bb_low_q = float(rg.get("bb_width_low_pctile", 40)) / 100.0
    mid = m5["close"].rolling(bb_p, min_periods=bb_p).mean()
    std = m5["close"].rolling(bb_p, min_periods=bb_p).std(ddof=0)
    m5["bb_mid"] = mid
    m5["bb_upper"] = mid + bb_std * std
    m5["bb_lower"] = mid - bb_std * std
    m5["rsi_m5"] = tokyo_bt.rolling_rsi(m5["close"], rsi_p)
    m5["bb_width"] = (m5["bb_upper"] - m5["bb_lower"]) / m5["bb_mid"].replace(0.0, np.nan)
    m5["bb_width_cutoff"] = m5["bb_width"].rolling(bb_pct_window, min_periods=bb_pct_window).quantile(bb_pct_cutoff)
    m5["bb_regime"] = np.where(m5["bb_width"] < m5["bb_width_cutoff"], "ranging", "trending")
    m5["bb_width_expanding3"] = (m5["bb_width"].shift(1) > m5["bb_width"].shift(2)) & (
        m5["bb_width"].shift(2) > m5["bb_width"].shift(3)
    )
    m5["bb_regime_expanding3"] = np.where(m5["bb_width_expanding3"].fillna(False), "trending", "ranging")
    m5["bb_width_q_high"] = m5["bb_width"].rolling(bb_rank_lookback, min_periods=bb_rank_lookback).quantile(bb_high_q)
    m5["bb_width_q_low"] = m5["bb_width"].rolling(bb_rank_lookback, min_periods=bb_rank_lookback).quantile(bb_low_q)
    rb = cfg.get("rejection_bonus", {})
    rb_ratio = float(rb.get("wick_to_body_ratio", 1.5))
    rb_close_pct = float(rb.get("close_position_pct", 0.4))
    rb_min_range_pips = float(rb.get("min_candle_range_pips", 3.0))
    rb_doji_wick_pct = float(rb.get("doji_wick_pct", 0.60))
    rb_lookback = int(max(1, rb.get("lookback_m5_bars", 2)))
    body_pips = (m5["close"] - m5["open"]).abs() / PIP_SIZE
    body_eff = body_pips.clip(lower=0.5)
    range_pips = (m5["high"] - m5["low"]) / PIP_SIZE
    upper_wick_pips = (m5["high"] - m5[["open", "close"]].max(axis=1)) / PIP_SIZE
    lower_wick_pips = (m5[["open", "close"]].min(axis=1) - m5["low"]) / PIP_SIZE
    close_upper_frac = (m5["close"] - m5["low"]) / (m5["high"] - m5["low"]).replace(0.0, np.nan)
    close_lower_frac = (m5["high"] - m5["close"]) / (m5["high"] - m5["low"]).replace(0.0, np.nan)
    doji = (body_pips <= 1e-9) & (range_pips > rb_min_range_pips)
    bull_main = (lower_wick_pips / body_eff >= rb_ratio) & (close_upper_frac >= (1.0 - rb_close_pct))
    bear_main = (upper_wick_pips / body_eff >= rb_ratio) & (close_lower_frac >= (1.0 - rb_close_pct))
    bull_doji = doji & ((lower_wick_pips / range_pips.replace(0.0, np.nan)) >= rb_doji_wick_pct)
    bear_doji = doji & ((upper_wick_pips / range_pips.replace(0.0, np.nan)) >= rb_doji_wick_pct)
    m5["rej_bull_m5"] = (bull_main | bull_doji).fillna(False)
    m5["rej_bear_m5"] = (bear_main | bear_doji).fillna(False)
    m5["rej_bull_recent"] = m5["rej_bull_m5"].rolling(rb_lookback, min_periods=1).max().astype(bool)
    m5["rej_bear_recent"] = m5["rej_bear_m5"].rolling(rb_lookback, min_periods=1).max().astype(bool)
    m5["rej_bull_low_recent"] = np.where(m5["rej_bull_m5"], m5["low"], np.nan)
    m5["rej_bear_high_recent"] = np.where(m5["rej_bear_m5"], m5["high"], np.nan)
    m5["rej_bull_low_recent"] = pd.Series(m5["rej_bull_low_recent"]).rolling(rb_lookback, min_periods=1).min().to_numpy()
    m5["rej_bear_high_recent"] = pd.Series(m5["rej_bear_high_recent"]).rolling(rb_lookback, min_periods=1).max().to_numpy()
    m5["rej_wick_ratio_bull"] = (lower_wick_pips / body_eff).replace([np.inf, -np.inf], np.nan)
    m5["rej_wick_ratio_bear"] = (upper_wick_pips / body_eff).replace([np.inf, -np.inf], np.nan)
    div_cfg = cfg.get("rsi_divergence_tracking", {})
    div_enabled = bool(div_cfg.get("enabled", False))
    div_rsi_period = int(div_cfg.get("rsi_period", 7))
    div_lb_min = int(div_cfg.get("lookback_min", 3))
    div_lb_max = int(div_cfg.get("lookback_max", 10))
    div_bull_rsi_max = float(div_cfg.get("bullish_rsi_max", 45.0))
    div_bear_rsi_min = float(div_cfg.get("bearish_rsi_min", 55.0))
    m5["rsi7_m5"] = tokyo_bt.rolling_rsi(m5["close"], div_rsi_period)
    bull_div = np.zeros(len(m5), dtype=bool)
    bear_div = np.zeros(len(m5), dtype=bool)
    if div_enabled and len(m5):
        for ii in range(len(m5)):
            j0 = max(0, ii - div_lb_max)
            j1 = max(0, ii - div_lb_min + 1)
            if j1 <= j0:
                continue
            prev = m5.iloc[j0:j1]
            if prev.empty:
                continue
            cur_low = float(m5.iloc[ii]["low"])
            cur_high = float(m5.iloc[ii]["high"])
            cur_rsi = float(m5.iloc[ii]["rsi7_m5"]) if not pd.isna(m5.iloc[ii]["rsi7_m5"]) else np.nan
            if pd.isna(cur_rsi):
                continue
            low_idx = prev["low"].idxmin()
            high_idx = prev["high"].idxmax()
            prev_low = float(m5.loc[low_idx, "low"])
            prev_high = float(m5.loc[high_idx, "high"])
            prev_low_rsi = float(m5.loc[low_idx, "rsi7_m5"]) if not pd.isna(m5.loc[low_idx, "rsi7_m5"]) else np.nan
            prev_high_rsi = float(m5.loc[high_idx, "rsi7_m5"]) if not pd.isna(m5.loc[high_idx, "rsi7_m5"]) else np.nan
            if (cur_low < prev_low) and (not pd.isna(prev_low_rsi)) and (cur_rsi > prev_low_rsi) and (cur_rsi < div_bull_rsi_max):
                bull_div[ii] = True
            if (cur_high > prev_high) and (not pd.isna(prev_high_rsi)) and (cur_rsi < prev_high_rsi) and (cur_rsi > div_bear_rsi_min):
                bear_div[ii] = True
    m5["rsi_div_bull"] = bull_div
    m5["rsi_div_bear"] = bear_div
    m5["rsi_div_bull_recent"] = m5["rsi_div_bull"].rolling(3, min_periods=1).max().astype(bool)
    m5["rsi_div_bear_recent"] = m5["rsi_div_bear"].rolling(3, min_periods=1).max().astype(bool)
    return m5


def _build_m15_indicator_frame(m15: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Port M15 block from scripts/backtest_tokyo_meanrev.py add_indicators lines 341-347."""
    m15 = m15.copy()
    atr_p = int(cfg["indicators"]["atr"]["period"])
    rg = cfg.get("regime_gate", {})
    atr_slow_p = int(rg.get("atr_slow_period", 50))
    adx_p = int(rg.get("adx_period", 14))
    m15["atr_m15"] = tokyo_bt.rolling_atr_price(m15, atr_p)
    m15["atr_m15_slow"] = tokyo_bt.rolling_atr_price(m15, atr_slow_p)
    m15["adx_m15"], m15["plus_di_m15"], m15["minus_di_m15"] = tokyo_bt.rolling_adx(m15, adx_p)
    return m15


def compute_tokyo_indicators(
    m1_df: pd.DataFrame,
    m5_df: Optional[pd.DataFrame],
    m15_df: Optional[pd.DataFrame],
    cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build M1 frame with Tokyo indicators (merge + pivots + SAR) and standalone M5/M15
    indicator frames for causal slicing.

    Source: scripts/backtest_tokyo_meanrev.py add_indicators lines 237-414,
    plus ATR percentile block run_one lines 772-791.
    """
    m5_src = m5_df if m5_df is not None else tokyo_bt.resample_ohlc_continuous(m1_df, "5min")
    m15_src = m15_df if m15_df is not None else tokyo_bt.resample_ohlc_continuous(m1_df, "15min")
    m5_ind = _build_m5_indicator_frame(m5_src, cfg)
    m15_ind = _build_m15_indicator_frame(m15_src, cfg)
    m1_enriched = tokyo_bt.add_indicators(m1_df.copy(), cfg)
    m1_enriched = _apply_atr_percentile_rank(m1_enriched, cfg)
    return m1_enriched, m5_ind, m15_ind


# TokyoPosition: native shadow type (scripts/backtest_tokyo_meanrev.py lines 33-84).
TokyoPosition = Position


@dataclass
class TokyoState:
    open_positions: list = field(default_factory=list)
    pending_confirmation: list = field(default_factory=list)
    pending_limit_orders: list = field(default_factory=list)
    session_active: bool = False
    daily_trades: int = 0
    current_day: str = ""
    last_bb_regime: str = ""
    last_breakout_bar: int = -1
    closed_trades: list = field(default_factory=list)
    total_realized_pnl: float = 0.0
    session_state: dict = field(default_factory=dict)
    diag: Counter = field(default_factory=Counter)
    engine: Optional[Any] = field(default=None, repr=False)
    # Inclusive last row index with TF close time <= current M1 ts (monotone; avoids O(len(tf)) scan per bar).
    m5_asof_idx: int = -1
    m15_asof_idx: int = -1


@dataclass
class TokyoBarResult:
    new_entries: list = field(default_factory=list)
    exits: list = field(default_factory=list)
    equity_delta: float = 0.0
    margin_delta: float = 0.0
    margin_used: float = 0.0
    diagnostics: dict = field(default_factory=dict)


def _get_tokyo_spread(_ts: Any, cfg: dict) -> float:
    """Shadow default spread; scripts/backtest_tokyo_meanrev.py run_one lines 622-624.

    Unified backtest uses fixed spread_pips from config (not --spread-mode).
    """
    return float(cfg.get("execution_model", {}).get("spread_pips", 1.5))


def _is_in_tokyo_session(minute_of_day_utc: int, start_min: int, end_min: int) -> bool:
    """scripts/backtest_tokyo_meanrev.py run_one in_window + session flags lines 549-567."""
    def in_window(m: int, ws: int, we: int) -> bool:
        if ws < we:
            return ws <= m < we
        return m >= ws or m < we

    return in_window(int(minute_of_day_utc), start_min, end_min)


def _minutes_to_session_end(minute_of_day_utc: int, start_min: int, end_min: int) -> int:
    """scripts/backtest_tokyo_meanrev.py run_one lines 555-562."""
    m = int(minute_of_day_utc)
    if start_min < end_min:
        return max(0, end_min - m)
    if m >= start_min:
        return (1440 - m) + end_min
    return max(0, end_min - m)


def tokyo_margin_used_open_positions(positions: list[Position], leverage: float = UNIFIED_MARGIN_LEVERAGE) -> float:
    return float(sum(max(0, int(p.units_remaining)) / max(1e-9, leverage) for p in positions))


class TokyoUnifiedEngine:
    """Holds extracted run_one parameters and per-bar advancement."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        sf = cfg.get("session_filter", {})
        self.session_start_utc = str(sf.get("session_start_utc", "15:00"))
        self.session_end_utc = str(sf.get("session_end_utc", "00:00"))
        self.entry_start_utc = str(sf.get("entry_start_utc", self.session_start_utc))
        self.entry_end_utc = str(sf.get("entry_end_utc", self.session_end_utc))
        self.block_new_entries_minutes_before_end = int(
            sf.get("block_new_entries_minutes_before_end", sf.get("session_entry_cutoff_minutes", 0))
        )
        self.lunch_block_enabled = bool(sf.get("lunch_block_enabled", False))
        self.lunch_block_start_utc = str(sf.get("lunch_block_start_utc", "02:30"))
        self.lunch_block_end_utc = str(sf.get("lunch_block_end_utc", "03:30"))
        self.allowed_days = set(sf.get("allowed_trading_days", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]))

        def hhmm_to_minutes(s: str) -> int:
            hh, mm = s.strip().split(":")
            return int(hh) * 60 + int(mm)

        self.start_min = hhmm_to_minutes(self.session_start_utc)
        self.end_min = hhmm_to_minutes(self.session_end_utc)
        self.entry_start_min = hhmm_to_minutes(self.entry_start_utc)
        self.entry_end_min = hhmm_to_minutes(self.entry_end_utc)
        self.lunch_start_min = hhmm_to_minutes(self.lunch_block_start_utc)
        self.lunch_end_min = hhmm_to_minutes(self.lunch_block_end_utc)

        self.risk_pct = float(cfg["position_sizing"]["risk_per_trade_pct"]) / 100.0
        self.day_risk_multipliers = {str(k): float(v) for k, v in cfg.get("position_sizing", {}).get("day_risk_multipliers", {}).items()}
        self.max_units = int(cfg["position_sizing"]["max_units"])
        self.max_open = int(cfg["position_sizing"]["max_concurrent_positions"])
        self.max_trades_session = int(cfg["trade_management"]["max_trades_per_session"])
        self.min_entry_gap_min = int(cfg["trade_management"]["min_time_between_entries_minutes"])
        self.no_reentry_stop_min = int(cfg["trade_management"]["no_reentry_same_direction_after_stop_minutes"])
        self.session_loss_stop_pct = float(cfg["trade_management"].get("session_loss_stop_pct", 0.0)) / 100.0
        self.stop_after_consecutive_losses = int(cfg["trade_management"].get("stop_after_consecutive_losses", 3))
        self.breakout_disable_pips = float(cfg["trade_management"]["disable_entries_if_move_from_tokyo_open_range_exceeds_pips"])
        self.breakout_mode = str(cfg["trade_management"].get("breakout_detection_mode", "rolling")).strip().lower()
        self.rolling_window_minutes = int(cfg["trade_management"].get("rolling_window_minutes", 60))
        self.rolling_range_threshold_pips = float(cfg["trade_management"].get("rolling_range_threshold_pips", 40.0))
        self.breakout_cooldown_minutes = int(cfg["trade_management"].get("cooldown_minutes", 15))
        consec_pause_cfg = cfg["trade_management"].get("consecutive_loss_pause", {})
        self.consec_pause_enabled = bool(consec_pause_cfg.get("enabled", False))
        self.consec_pause_losses = int(consec_pause_cfg.get("consecutive_losses", 2))
        self.consec_pause_minutes = int(consec_pause_cfg.get("pause_minutes", 30))

        self.atr_max = float(cfg["indicators"]["atr"]["max_threshold_price_units"])
        self.atr_gate_enabled = bool(cfg["indicators"]["atr"].get("use_as_hard_gate", True))
        atr_pct_cfg = cfg["indicators"]["atr"].get("percentile_filter", {})
        self.atr_pct_enabled = bool(atr_pct_cfg.get("enabled", False))
        self.regime_filter_mode = str(cfg["indicators"].get("bb_width_regime_filter", {}).get("mode", "percentile")).strip().lower()
        scoring_model = str(cfg.get("entry_rules", {}).get("scoring_model", "")).strip().lower()
        self.tokyo_v2_scoring = scoring_model in {"tokyo_v2", "v2", "tokyo_actual_v2"}
        self.tol_pips = float(cfg["entry_rules"]["long"].get("price_zone", {}).get("tolerance_pips", 10.0))
        self.tol = self.tol_pips * PIP_SIZE
        self.min_tp_pips = 8.0
        self.min_sl_pips = float(cfg["exit_rules"]["stop_loss"].get("minimum_sl_pips", 10.0))
        self.max_sl_pips = float(cfg["exit_rules"]["stop_loss"].get("hard_max_sl_pips", 28.0))
        self.sl_buf = float(cfg["exit_rules"]["stop_loss"].get("buffer_pips", 8.0)) * PIP_SIZE
        self.trail_activate_pips = float(cfg["exit_rules"]["trailing_stop"].get("activate_after_profit_pips", 10.0))
        self.trail_dist_pips = float(cfg["exit_rules"]["trailing_stop"].get("trail_distance_pips", 8.0))
        self.trail_enabled = bool(cfg["exit_rules"]["trailing_stop"].get("enabled", True))
        self.trail_requires_tp1 = bool(cfg["exit_rules"]["trailing_stop"].get("requires_tp1_hit", True))
        self.partial_close_pct = float(cfg["exit_rules"]["take_profit"].get("partial_close_pct", 0.5))
        self.partial_tp_min_pips = float(cfg["exit_rules"]["take_profit"].get("partial_tp_min_pips", 6.0))
        self.partial_tp_max_pips = float(cfg["exit_rules"]["take_profit"].get("partial_tp_max_pips", 12.0))
        self.partial_tp_atr_mult = float(cfg["exit_rules"]["take_profit"].get("partial_tp_atr_mult", 0.5))
        self.tp_mode = str(cfg["exit_rules"]["take_profit"].get("mode", "partial")).strip().lower()
        self.single_tp_atr_mult = float(cfg["exit_rules"]["take_profit"].get("single_tp_atr_mult", 1.0))
        self.single_tp_min_pips = float(cfg["exit_rules"]["take_profit"].get("single_tp_min_pips", self.min_tp_pips))
        self.single_tp_max_pips = float(cfg["exit_rules"]["take_profit"].get("single_tp_max_pips", 40.0))
        self.breakeven_offset_pips = float(cfg["exit_rules"]["take_profit"].get("breakeven_offset_pips", 1.0))
        self.time_decay_minutes = int(cfg["exit_rules"]["time_exit"].get("time_decay_minutes", 120))
        self.time_decay_profit_cap_pips = float(cfg["exit_rules"]["time_exit"].get("time_decay_profit_cap_pips", 3.0))

        self.max_entry_spread_pips = float(cfg.get("execution_model", {}).get("max_entry_spread_pips", 10.0))

        self.confluence_min_long = int(cfg["entry_rules"]["long"].get("confluence_scoring", {}).get("minimum_score", 2))
        self.confluence_min_short = int(cfg["entry_rules"]["short"].get("confluence_scoring", {}).get("minimum_score", 2))
        self.long_rsi_soft_entry = float(cfg["entry_rules"]["long"].get("rsi_soft_filter", {}).get("entry_soft_threshold", 35.0))
        self.long_rsi_bonus = float(cfg["entry_rules"]["long"].get("rsi_soft_filter", {}).get("bonus_threshold", 30.0))
        self.short_rsi_soft_entry = float(cfg["entry_rules"]["short"].get("rsi_soft_filter", {}).get("entry_soft_threshold", 65.0))
        self.short_rsi_bonus = float(cfg["entry_rules"]["short"].get("rsi_soft_filter", {}).get("bonus_threshold", 70.0))
        core_gate_cfg = cfg.get("entry_rules", {}).get("core_gate", {})
        self.core_gate_required = int(core_gate_cfg.get("required_count", 4))
        self.core_gate_use_zone = bool(core_gate_cfg.get("use_zone", True))
        self.core_gate_use_bb = bool(core_gate_cfg.get("use_bb_touch", True))
        self.core_gate_use_sar = bool(core_gate_cfg.get("use_sar_flip", True))
        self.core_gate_use_rsi = bool(core_gate_cfg.get("use_rsi_soft", True))

        confirm_cfg = cfg.get("entry_confirmation", {})
        self.confirmation_enabled = bool(confirm_cfg.get("enabled", True))
        self.confirmation_type = str(confirm_cfg.get("type", "m1")).strip().lower()
        if self.confirmation_type not in {"m1", "m5"}:
            self.confirmation_type = "m1"
        self.confirmation_window_bars = max(0, int(confirm_cfg.get("window_bars", 5)))
        if not self.confirmation_enabled:
            self.confirmation_window_bars = 0

        adx_filter_cfg = cfg.get("adx_filter", {})
        self.adx_filter_enabled = bool(adx_filter_cfg.get("enabled", False))
        self.adx_max_for_entry = float(adx_filter_cfg.get("max_adx_for_entry", 30.0))
        self.adx_day_max_by_day = {str(k): float(v) for k, v in adx_filter_cfg.get("day_max_by_day", {}).items()}
        self.adx_day_max_by_day.update({str(k): float(v) for k, v in adx_filter_cfg.get("day_overrides", {}).items()})

        combo_filter_cfg = cfg.get("confluence_combo_filter", {})
        self.combo_filter_enabled = bool(combo_filter_cfg.get("enabled", False))
        self.combo_filter_mode = str(combo_filter_cfg.get("mode", "allowlist")).strip().lower()
        self.combo_allow = set(combo_filter_cfg.get("allowed_combos", []))
        self.combo_block = set(combo_filter_cfg.get("blocked_combos", []))

        self.daily_range_enabled = bool(cfg.get("daily_range_filter", {}).get("enabled", False))
        self.daily_range_min_pips = float(cfg.get("daily_range_filter", {}).get("min_prior_day_range_pips", 15.0))
        self.daily_range_max_pips = float(cfg.get("daily_range_filter", {}).get("max_prior_day_range_pips", 80.0))

        self.entry_imp_enabled = bool(cfg.get("entry_improvement", {}).get("enabled", False))
        self.chase_enabled = bool(cfg.get("entry_chase_filter", {}).get("enabled", False))
        self.chase_max_pips = float(cfg.get("entry_chase_filter", {}).get("max_chase_pips", 2.0))

        ss_cfg = cfg.get("signal_strength_tracking", {})
        self.ss_enabled = bool(ss_cfg.get("enabled", False))
        self.ss_comp = ss_cfg.get("components", {})
        ss_filter_cfg = cfg.get("signal_strength_filter", {})
        self.ss_filter_enabled = bool(ss_filter_cfg.get("enabled", False))
        self.ss_filter_min_score = int(ss_filter_cfg.get("min_score", 0))
        ss_sizing_cfg = cfg.get("signal_strength_sizing", {})
        self.ss_sizing_enabled = bool(ss_sizing_cfg.get("enabled", False))
        self.ss_sizing_weak_mult = float(ss_sizing_cfg.get("weak_mult", 1.0))
        self.ss_sizing_moderate_mult = float(ss_sizing_cfg.get("moderate_mult", 1.0))
        self.ss_sizing_strong_mult = float(ss_sizing_cfg.get("strong_mult", 1.0))

        dd_cfg = cfg.get("drawdown_adaptive_sizing", {})
        self.dd_enabled = bool(dd_cfg.get("enabled", False))
        self.dd_t1 = float(dd_cfg.get("tier_1_dd_pct", 2.0))
        self.dd_t1_red = float(dd_cfg.get("tier_1_size_reduction", 0.25))
        self.dd_t2 = float(dd_cfg.get("tier_2_dd_pct", 4.0))
        self.dd_t2_red = float(dd_cfg.get("tier_2_size_reduction", 0.50))

        hp = cfg.get("hour_preference", {})
        self.hp_enabled = bool(hp.get("enabled", False))
        self.hp_mults = {str(k): float(v) for k, v in hp.get("multipliers", {}).items()}

        regime_gate = cfg.get("regime_gate", {})
        self.regime_enabled = bool(regime_gate.get("enabled", False))
        self.atr_ratio_trend = float(regime_gate.get("atr_ratio_trending_threshold", 1.3))
        self.atr_ratio_calm = float(regime_gate.get("atr_ratio_calm_threshold", 0.8))
        self.adx_trend = float(regime_gate.get("adx_trending_threshold", 25.0))
        self.adx_range = float(regime_gate.get("adx_ranging_threshold", 20.0))
        self.favorable_min_score = int(regime_gate.get("favorable_min_score", 1))
        self.neutral_min_score = int(regime_gate.get("neutral_min_score", 0))
        self.neutral_size_mult = float(regime_gate.get("neutral_size_multiplier", 0.5))

        cq = cfg.get("confluence_quality", {})
        self.cq_enabled = bool(cq.get("enabled", False))
        self.top_combos = set(cq.get("top_combos", []))
        self.bottom_combos = set(cq.get("bottom_combos", []))
        self.high_quality_mult = float(cq.get("high_quality_size_mult", 1.0))
        self.medium_quality_mult = float(cq.get("medium_quality_size_mult", 0.75))
        self.low_quality_skip = bool(cq.get("low_quality_skip", True))

        rb = cfg.get("rejection_bonus", {})
        self.rejection_bonus_enabled = bool(rb.get("enabled", False))
        self.rejection_sl_improvement = bool(rb.get("sl_improvement", False))
        self.rejection_sl_buffer_pips = float(rb.get("sl_buffer_pips", 2.0))

        div_cfg = cfg.get("rsi_divergence_tracking", {})
        self.div_track_enabled = bool(div_cfg.get("enabled", False))

        se = cfg.get("session_envelope", {})
        self.session_env_enabled = bool(se.get("enabled", False))
        self.session_env_log_ir_pos = bool(se.get("log_ir_position", True))
        self.session_env_use_for_tp = bool(se.get("use_for_tp", True))
        self.session_env_tp_mode = str(se.get("tp_mode", "nearest_of_pivot_or_midpoint")).strip().lower()

        momentum_cfg = cfg.get("momentum_check", {})
        self.momentum_enabled = bool(momentum_cfg.get("enabled", False))
        self.momentum_lookback = int(momentum_cfg.get("lookback_candles", 10))
        self.momentum_slope_th = float(momentum_cfg.get("slope_threshold_pips_per_candle", 0.3))
        self.momentum_delay_candles = int(momentum_cfg.get("delay_candles", 5))
        self.momentum_max_delays = int(momentum_cfg.get("max_delays", 1))

        early_exit_cfg = cfg.get("early_exit", {})
        self.early_exit_enabled = bool(early_exit_cfg.get("enabled", False))
        self.early_exit_time_min = int(early_exit_cfg.get("time_threshold_minutes", 30))
        self.early_exit_loss_pips = float(early_exit_cfg.get("loss_threshold_pips", 5.0))
        self.early_exit_max_profit_seen = float(early_exit_cfg.get("max_profit_seen_pips", 2.0))

        late_session_cfg = cfg.get("late_session_management", {})
        self.late_session_enabled = bool(late_session_cfg.get("enabled", False))
        self.late_session_minutes_before_end = int(late_session_cfg.get("minutes_before_end", 45))
        self.late_session_close_if_no_tp1_and_pips_below = float(late_session_cfg.get("close_if_no_tp1_and_pips_below", -2.0))
        self.late_session_tp1_hit_tighten_trail_pips = float(late_session_cfg.get("tp1_hit_tighten_trail_pips", 3.0))
        self.late_session_hard_close_all_minutes_before_end = int(late_session_cfg.get("hard_close_all_minutes_before_end", 0))
        self.late_session_be_or_close_minutes_before_end = int(late_session_cfg.get("be_or_close_minutes_before_end", 0))
        self.late_session_be_min_profit_pips = float(late_session_cfg.get("be_min_profit_pips", 1.0))
        self.late_session_be_offset_pips = float(late_session_cfg.get("be_offset_pips", 0.0))
        self.late_session_profit_tighten_minutes_before_end = int(late_session_cfg.get("profit_tighten_minutes_before_end", 0))
        self.late_session_profit_tighten_trail_mult = float(late_session_cfg.get("profit_tighten_trail_mult", 0.5))

        self.news_enabled = bool(cfg.get("news_filter", {}).get("enabled", False))

        self.trend_skip_enabled = bool(cfg.get("trend_regime_skip", {}).get("enabled", False))
        self.regime_switch_enabled = bool(cfg.get("regime_switch_filter", {}).get("enabled", False))

        self._v14_cfg_params = {
            "tokyo_v2_scoring": self.tokyo_v2_scoring,
            "confluence_min_long": self.confluence_min_long,
            "confluence_min_short": self.confluence_min_short,
            "long_rsi_soft_entry": self.long_rsi_soft_entry,
            "long_rsi_bonus": self.long_rsi_bonus,
            "short_rsi_soft_entry": self.short_rsi_soft_entry,
            "short_rsi_bonus": self.short_rsi_bonus,
            "tol": self.tol,
            "atr_max": self.atr_max,
            "core_gate_use_zone": self.core_gate_use_zone,
            "core_gate_use_bb": self.core_gate_use_bb,
            "core_gate_use_sar": self.core_gate_use_sar,
            "core_gate_use_rsi": self.core_gate_use_rsi,
            "core_gate_required": self.core_gate_required,
            "regime_enabled": self.regime_enabled,
            "atr_ratio_trend": self.atr_ratio_trend,
            "atr_ratio_calm": self.atr_ratio_calm,
            "adx_trend": self.adx_trend,
            "adx_range": self.adx_range,
            "favorable_min_score": self.favorable_min_score,
            "neutral_min_score": self.neutral_min_score,
            "neutral_size_mult": self.neutral_size_mult,
            "ss_enabled": self.ss_enabled,
            "ss_comp": self.ss_comp,
            "combo_filter_enabled": self.combo_filter_enabled,
            "combo_filter_mode": self.combo_filter_mode,
            "combo_allow": self.combo_allow,
            "combo_block": self.combo_block,
            "ss_filter_enabled": self.ss_filter_enabled,
            "ss_filter_min_score": self.ss_filter_min_score,
            "cq_enabled": self.cq_enabled,
            "top_combos": self.top_combos,
            "bottom_combos": self.bottom_combos,
            "high_quality_mult": self.high_quality_mult,
            "medium_quality_mult": self.medium_quality_mult,
            "low_quality_skip": self.low_quality_skip,
            "rejection_bonus_enabled": self.rejection_bonus_enabled,
            "div_track_enabled": self.div_track_enabled,
            "session_env_enabled": self.session_env_enabled,
            "session_env_log_ir_pos": self.session_env_log_ir_pos,
        }

    def in_window(self, minute_of_day_utc: int, win_start: int, win_end: int) -> bool:
        m = int(minute_of_day_utc)
        if win_start < win_end:
            return (m >= win_start) and (m < win_end)
        return (m >= win_start) or (m < win_end)

    def ensure_session(self, state: TokyoState, sday: str, row: pd.Series, equity: float) -> dict:
        """scripts/backtest_tokyo_meanrev.py run_one ensure_session lines 1002-1073 (subset)."""
        if sday not in state.session_state:
            state.session_state[sday] = {
                "trades": 0,
                "consec_losses": 0,
                "stopped": False,
                "last_entry_time": None,
                "last_stop_time_long": None,
                "last_stop_time_short": None,
                "session_open_price": float(row["open"]),
                "session_high": float(row["high"]),
                "session_low": float(row["low"]),
                "ir_ready": False,
                "ir_high": float(row["high"]),
                "ir_low": float(row["low"]),
                "warmup_end_ts": pd.Timestamp(row["time"])
                + pd.Timedelta(minutes=int(self.cfg.get("session_envelope", {}).get("warmup_minutes", 30))),
                "session_pnl_usd": 0.0,
                "session_start_equity": float(equity),
                "rolling_window": [],
                "breakout_cooldown_until": None,
                "daily_range_allowed": True,
                "daily_range_block_reason": None,
                "regime_switch_allowed": True,
                "trend_skip_allowed": True,
                "loss_pause_until": None,
                "signals_generated": 0,
                "wins": 0,
                "losses": 0,
                "entry_confluence_scores": [],
            }
            if self.daily_range_enabled:
                pdr = row.get("prior_day_range_pips", np.nan)
                if pd.isna(pdr):
                    state.session_state[sday]["daily_range_allowed"] = False
                    state.session_state[sday]["daily_range_block_reason"] = "missing"
                elif float(pdr) < self.daily_range_min_pips:
                    state.session_state[sday]["daily_range_allowed"] = False
                    state.session_state[sday]["daily_range_block_reason"] = "low"
                elif float(pdr) > self.daily_range_max_pips:
                    state.session_state[sday]["daily_range_allowed"] = False
                    state.session_state[sday]["daily_range_block_reason"] = "high"
        return state.session_state[sday]

    def close_position_record(
        self,
        pos: Position,
        ts: pd.Timestamp,
        exit_price: float,
        reason: str,
        bar_idx: int,
        *,
        partial_close_1_time: str = "",
        partial_close_1_pct: Any = "",
        be_triggered: Any = "",
    ) -> dict:
        """Finalize remaining units + CSV row; scripts/backtest_tokyo_meanrev.py close_position lines 1075-1084."""
        if pos.units_remaining > 0:
            pips, usd = calc_leg_usd_pips(pos.direction, pos.entry_price, exit_price, pos.units_remaining)
            pos.realized_pip_units += pips * pos.units_remaining
            pos.realized_usd += usd
            pos.units_remaining = 0
        pos.exit_price = float(exit_price)
        pos.exit_reason = reason
        pos.exit_time = ts
        total_pips = pos.realized_pip_units / max(1, pos.units_initial)
        side = "buy" if pos.direction == "long" else "sell"
        oc = getattr(pos, "ownership_cell", "") or ""
        reg = getattr(pos, "regime_label_at_entry", "") or pos.regime_label
        return {
            "trade_id": pos.trade_id,
            "session": "tokyo_v14",
            "setup_type": "v14",
            "entry_bar_index": getattr(pos, "entry_bar_idx", -1),
            "entry_time": str(pos.entry_time),
            "entry_price": pos.entry_price,
            "side": side,
            "lots": pos.units_initial / 100_000.0,
            "sl_price": pos.sl_price,
            "sl_pips": pos.initial_sl_pips,
            "tp1_price": pos.tp1_price,
            "tp2_price": pos.tp2_price,
            "exit_bar_index": bar_idx,
            "exit_time": str(ts),
            "exit_price": exit_price,
            "exit_reason": reason,
            "pnl_pips": float(total_pips),
            "pnl_usd": float(pos.realized_usd),
            "ownership_cell": oc,
            "regime_label": reg,
            "er": "",
            "delta_er": "",
            "admission_filters_applied": "variant_i",
            "admission_filters_passed": True,
            "was_v44_oracle": False,
            "partial_close_1_time": partial_close_1_time,
            "partial_close_1_price": "",
            "partial_close_1_pct": partial_close_1_pct,
            "be_triggered": be_triggered,
            "peak_favorable_pips": pos.max_profit_seen_pips,
            "peak_adverse_pips": pos.max_adverse_seen_pips,
            "equity_at_entry": getattr(pos, "equity_at_entry", ""),
            "margin_used_at_entry": getattr(pos, "margin_used_at_entry", ""),
            "mfe_pips": pos.max_profit_seen_pips,
            "mae_pips": pos.max_adverse_seen_pips,
            "oracle_pnl_usd": "",
            "sizing_scale_vs_oracle": "",
        }

    def update_sst_after_close(self, state: TokyoState, pos: Position, ts: pd.Timestamp) -> None:
        """Session stats after a fully closed trade (pos.realized_usd = total trade USD)."""
        sst = state.session_state.get(pos.entry_session_day, {})
        total = float(pos.realized_usd)
        win = total > 0
        sst["session_pnl_usd"] = float(sst.get("session_pnl_usd", 0.0)) + total
        sess_start_eq = float(sst.get("session_start_equity", 0.0))
        if self.session_loss_stop_pct > 0 and sess_start_eq > 0 and float(sst["session_pnl_usd"]) <= (-self.session_loss_stop_pct * sess_start_eq):
            sst["stopped"] = True
        if win:
            sst["consec_losses"] = 0
            sst["wins"] = int(sst.get("wins", 0)) + 1
        else:
            sst["consec_losses"] = int(sst.get("consec_losses", 0)) + 1
            sst["losses"] = int(sst.get("losses", 0)) + 1
            if self.stop_after_consecutive_losses > 0 and int(sst["consec_losses"]) >= self.stop_after_consecutive_losses:
                sst["stopped"] = True
            if self.consec_pause_enabled and self.consec_pause_losses > 0 and int(sst["consec_losses"]) >= self.consec_pause_losses:
                sst["loss_pause_until"] = ts + pd.Timedelta(minutes=self.consec_pause_minutes)
                sst["consec_losses"] = 0
        er = str(pos.exit_reason or "")
        if "sl" in er:
            if pos.direction == "long":
                sst["last_stop_time_long"] = ts
            else:
                sst["last_stop_time_short"] = ts

    def _full_close(
        self,
        state: TokyoState,
        pos: Position,
        ts: pd.Timestamp,
        exit_price: float,
        reason: str,
        bar_idx: int,
        **kw: Any,
    ) -> tuple[float, dict[str, Any]]:
        """Close remaining units, update session stats, remove from open list; return (total trade USD, CSV row)."""
        rec = self.close_position_record(pos, ts, exit_price, reason, bar_idx, **kw)
        tot = float(pos.realized_usd)
        self.update_sst_after_close(state, pos, ts)
        state.open_positions.remove(pos)
        return tot, rec

    def manage_positions(
        self,
        state: TokyoState,
        *,
        i: int,
        ts: pd.Timestamp,
        row: pd.Series,
        in_session: bool,
        remaining_session_minutes: Optional[int],
        bid_close: float,
        ask_close: float,
        bid_high: float,
        bid_low: float,
        ask_high: float,
        ask_low: float,
        equity: float,
    ) -> tuple[float, list[dict[str, Any]]]:
        """
        Manage open Tokyo positions for one bar.
        Source: scripts/backtest_tokyo_meanrev.py lines 1727-1978 (long + short branches).
        """
        exit_pnl = 0.0
        exits_out: list[dict[str, Any]] = []
        for pos in list(state.open_positions):
            if pos.direction == "long":
                favorable = (bid_high - pos.entry_price) / PIP_SIZE
            else:
                favorable = (pos.entry_price - ask_low) / PIP_SIZE
            can_trail = (not self.trail_requires_tp1) or pos.tp1_hit
            if self.trail_enabled and can_trail and favorable >= self.trail_activate_pips:
                pos.trail_active = True
            if pos.trail_active and pos.units_remaining > 0:
                if pos.direction == "long":
                    new_trail = bid_close - self.trail_dist_pips * PIP_SIZE
                    pos.trail_stop_price = new_trail if pos.trail_stop_price is None else max(pos.trail_stop_price, new_trail)
                else:
                    new_trail = ask_close + self.trail_dist_pips * PIP_SIZE
                    pos.trail_stop_price = new_trail if pos.trail_stop_price is None else min(pos.trail_stop_price, new_trail)

            if pos.units_remaining <= 0:
                continue
            held_minutes = (ts - pd.Timestamp(pos.entry_time)).total_seconds() / 60.0
            if pos.direction == "long":
                current_pips = (bid_close - pos.entry_price) / PIP_SIZE
                bar_fav = (bid_high - pos.entry_price) / PIP_SIZE
                bar_adv = (pos.entry_price - bid_low) / PIP_SIZE
                pos.max_profit_seen_pips = max(float(pos.max_profit_seen_pips), float(bar_fav))
                pos.max_adverse_seen_pips = max(float(pos.max_adverse_seen_pips), float(bar_adv))
                if (
                    in_session
                    and remaining_session_minutes is not None
                    and int(self.late_session_hard_close_all_minutes_before_end) > 0
                    and int(remaining_session_minutes) <= int(self.late_session_hard_close_all_minutes_before_end)
                ):
                    tot, rec = self._full_close(
                        state,
                        pos,
                        ts,
                        bid_close,
                        "tp1_then_late_session_hard_close" if pos.tp1_hit else "late_session_hard_close",
                        i,
                    )
                    exit_pnl += tot
                    exits_out.append(rec)
                    continue
                if (
                    in_session
                    and remaining_session_minutes is not None
                    and int(self.late_session_be_or_close_minutes_before_end) > 0
                    and int(remaining_session_minutes) <= int(self.late_session_be_or_close_minutes_before_end)
                ):
                    if current_pips >= float(self.late_session_be_min_profit_pips):
                        be_px = pos.entry_price + float(self.late_session_be_offset_pips) * PIP_SIZE
                        if be_px > float(pos.sl_price):
                            pos.sl_price = be_px
                            pos.moved_to_breakeven = True
                    elif current_pips < 0:
                        tot, rec = self._full_close(state, pos, ts, bid_close, "late_session_be_or_close_loss", i)
                        exit_pnl += tot
                        exits_out.append(rec)
                        continue
                if (
                    in_session
                    and remaining_session_minutes is not None
                    and int(self.late_session_profit_tighten_minutes_before_end) > 0
                    and int(remaining_session_minutes) <= int(self.late_session_profit_tighten_minutes_before_end)
                    and current_pips > 0
                    and self.trail_enabled
                ):
                    tight_dist = max(0.1, float(self.trail_dist_pips) * float(self.late_session_profit_tighten_trail_mult))
                    tight_trail = bid_close - tight_dist * PIP_SIZE
                    pos.trail_active = True
                    pos.trail_stop_price = tight_trail if pos.trail_stop_price is None else max(pos.trail_stop_price, tight_trail)
                if (
                    self.late_session_enabled
                    and in_session
                    and remaining_session_minutes is not None
                    and int(remaining_session_minutes) <= int(self.late_session_minutes_before_end)
                ):
                    if (not pos.tp1_hit) and (current_pips < float(self.late_session_close_if_no_tp1_and_pips_below)):
                        tot, rec = self._full_close(state, pos, ts, bid_close, "late_session_no_tp1_cut", i)
                        exit_pnl += tot
                        exits_out.append(rec)
                        continue
                    if pos.tp1_hit and float(self.late_session_tp1_hit_tighten_trail_pips) > 0:
                        tight_trail = bid_close - float(self.late_session_tp1_hit_tighten_trail_pips) * PIP_SIZE
                        pos.trail_active = True
                        pos.trail_stop_price = tight_trail if pos.trail_stop_price is None else max(pos.trail_stop_price, tight_trail)
                hit_sl = bid_low <= pos.sl_price
                hit_tp1 = (not pos.tp1_hit) and (bid_high >= pos.tp1_price)
                hit_tp2 = bid_high >= pos.tp2_price
                hit_trail = can_trail and pos.trail_active and pos.trail_stop_price is not None and (bid_low <= float(pos.trail_stop_price))
                hit_time_decay = pos.tp1_hit and held_minutes >= self.time_decay_minutes and (0.0 <= current_pips < self.time_decay_profit_cap_pips)
                hit_early_exit = (
                    self.early_exit_enabled
                    and (not pos.tp1_hit)
                    and held_minutes >= self.early_exit_time_min
                    and current_pips <= (-self.early_exit_loss_pips)
                    and float(pos.max_profit_seen_pips) <= self.early_exit_max_profit_seen
                )
                if hit_early_exit:
                    tot, rec = self._full_close(state, pos, ts, bid_close, "early_exit_dead_wrong", i)
                    exit_pnl += tot
                    exits_out.append(rec)
                    continue
                if hit_sl:
                    be_px = pos.entry_price + self.breakeven_offset_pips * PIP_SIZE
                    if pos.tp1_hit and pos.moved_to_breakeven and abs(pos.sl_price - be_px) <= 1e-9:
                        reason = "tp1_then_be_stop"
                    else:
                        reason = "tp1_then_sl" if pos.tp1_hit else "sl"
                    tot, rec = self._full_close(state, pos, ts, float(pos.sl_price), reason, i)
                    exit_pnl += tot
                    exits_out.append(rec)
                    continue
                if hit_tp1:
                    close_units = int(math.floor(pos.units_initial * pos.tp1_close_pct))
                    close_units = max(1, min(close_units, pos.units_remaining))
                    pips, usd = calc_leg_usd_pips("long", pos.entry_price, pos.tp1_price, close_units)
                    pos.realized_pip_units += pips * close_units
                    pos.realized_usd += usd
                    pos.units_remaining -= close_units
                    pos.tp1_hit = True
                    pos.sl_price = pos.entry_price + self.breakeven_offset_pips * PIP_SIZE
                    pos.moved_to_breakeven = True
                    if pos.units_remaining <= 0:
                        tot, rec = self._full_close(
                            state,
                            pos,
                            ts,
                            float(pos.tp1_price),
                            "tp",
                            i,
                            partial_close_1_time=str(ts),
                            partial_close_1_pct=pos.tp1_close_pct,
                            be_triggered=True,
                        )
                        exit_pnl += tot
                        exits_out.append(rec)
                        continue
                if pos.tp1_hit and hit_tp2:
                    tot, rec = self._full_close(state, pos, ts, float(pos.tp2_price), "tp1_then_tp2", i)
                    exit_pnl += tot
                    exits_out.append(rec)
                    continue
                if hit_trail:
                    tot, rec = self._full_close(state, pos, ts, float(pos.trail_stop_price), "tp1_then_trailing_stop", i)
                    exit_pnl += tot
                    exits_out.append(rec)
                    continue
                if hit_time_decay:
                    tot, rec = self._full_close(state, pos, ts, bid_close, "tp1_then_time_decay", i)
                    exit_pnl += tot
                    exits_out.append(rec)
                    continue
            else:
                current_pips = (pos.entry_price - ask_close) / PIP_SIZE
                bar_fav = (pos.entry_price - ask_low) / PIP_SIZE
                bar_adv = (ask_high - pos.entry_price) / PIP_SIZE
                pos.max_profit_seen_pips = max(float(pos.max_profit_seen_pips), float(bar_fav))
                pos.max_adverse_seen_pips = max(float(pos.max_adverse_seen_pips), float(bar_adv))
                if (
                    in_session
                    and remaining_session_minutes is not None
                    and int(self.late_session_hard_close_all_minutes_before_end) > 0
                    and int(remaining_session_minutes) <= int(self.late_session_hard_close_all_minutes_before_end)
                ):
                    tot, rec = self._full_close(
                        state,
                        pos,
                        ts,
                        ask_close,
                        "tp1_then_late_session_hard_close" if pos.tp1_hit else "late_session_hard_close",
                        i,
                    )
                    exit_pnl += tot
                    exits_out.append(rec)
                    continue
                if (
                    in_session
                    and remaining_session_minutes is not None
                    and int(self.late_session_be_or_close_minutes_before_end) > 0
                    and int(remaining_session_minutes) <= int(self.late_session_be_or_close_minutes_before_end)
                ):
                    if current_pips >= float(self.late_session_be_min_profit_pips):
                        be_px = pos.entry_price - float(self.late_session_be_offset_pips) * PIP_SIZE
                        if be_px < float(pos.sl_price):
                            pos.sl_price = be_px
                            pos.moved_to_breakeven = True
                    elif current_pips < 0:
                        tot, rec = self._full_close(state, pos, ts, ask_close, "late_session_be_or_close_loss", i)
                        exit_pnl += tot
                        exits_out.append(rec)
                        continue
                if (
                    in_session
                    and remaining_session_minutes is not None
                    and int(self.late_session_profit_tighten_minutes_before_end) > 0
                    and int(remaining_session_minutes) <= int(self.late_session_profit_tighten_minutes_before_end)
                    and current_pips > 0
                    and self.trail_enabled
                ):
                    tight_dist = max(0.1, float(self.trail_dist_pips) * float(self.late_session_profit_tighten_trail_mult))
                    tight_trail = ask_close + tight_dist * PIP_SIZE
                    pos.trail_active = True
                    pos.trail_stop_price = tight_trail if pos.trail_stop_price is None else min(pos.trail_stop_price, tight_trail)
                if (
                    self.late_session_enabled
                    and in_session
                    and remaining_session_minutes is not None
                    and int(remaining_session_minutes) <= int(self.late_session_minutes_before_end)
                ):
                    if (not pos.tp1_hit) and (current_pips < float(self.late_session_close_if_no_tp1_and_pips_below)):
                        tot, rec = self._full_close(state, pos, ts, ask_close, "late_session_no_tp1_cut", i)
                        exit_pnl += tot
                        exits_out.append(rec)
                        continue
                    if pos.tp1_hit and float(self.late_session_tp1_hit_tighten_trail_pips) > 0:
                        tight_trail = ask_close + float(self.late_session_tp1_hit_tighten_trail_pips) * PIP_SIZE
                        pos.trail_active = True
                        pos.trail_stop_price = tight_trail if pos.trail_stop_price is None else min(pos.trail_stop_price, tight_trail)
                hit_sl = ask_high >= pos.sl_price
                hit_tp1 = (not pos.tp1_hit) and (ask_low <= pos.tp1_price)
                hit_tp2 = ask_low <= pos.tp2_price
                hit_trail = can_trail and pos.trail_active and pos.trail_stop_price is not None and (ask_high >= float(pos.trail_stop_price))
                hit_time_decay = pos.tp1_hit and held_minutes >= self.time_decay_minutes and (0.0 <= current_pips < self.time_decay_profit_cap_pips)
                hit_early_exit = (
                    self.early_exit_enabled
                    and (not pos.tp1_hit)
                    and held_minutes >= self.early_exit_time_min
                    and current_pips <= (-self.early_exit_loss_pips)
                    and float(pos.max_profit_seen_pips) <= self.early_exit_max_profit_seen
                )
                if hit_early_exit:
                    tot, rec = self._full_close(state, pos, ts, ask_close, "early_exit_dead_wrong", i)
                    exit_pnl += tot
                    exits_out.append(rec)
                    continue
                if hit_sl:
                    be_px = pos.entry_price - self.breakeven_offset_pips * PIP_SIZE
                    if pos.tp1_hit and pos.moved_to_breakeven and abs(pos.sl_price - be_px) <= 1e-9:
                        reason = "tp1_then_be_stop"
                    else:
                        reason = "tp1_then_sl" if pos.tp1_hit else "sl"
                    tot, rec = self._full_close(state, pos, ts, float(pos.sl_price), reason, i)
                    exit_pnl += tot
                    exits_out.append(rec)
                    continue
                if hit_tp1:
                    close_units = int(math.floor(pos.units_initial * pos.tp1_close_pct))
                    close_units = max(1, min(close_units, pos.units_remaining))
                    pips, usd = calc_leg_usd_pips("short", pos.entry_price, pos.tp1_price, close_units)
                    pos.realized_pip_units += pips * close_units
                    pos.realized_usd += usd
                    pos.units_remaining -= close_units
                    pos.tp1_hit = True
                    pos.sl_price = pos.entry_price - self.breakeven_offset_pips * PIP_SIZE
                    pos.moved_to_breakeven = True
                    if pos.units_remaining <= 0:
                        tot, rec = self._full_close(
                            state,
                            pos,
                            ts,
                            float(pos.tp1_price),
                            "tp",
                            i,
                            partial_close_1_time=str(ts),
                            partial_close_1_pct=pos.tp1_close_pct,
                            be_triggered=True,
                        )
                        exit_pnl += tot
                        exits_out.append(rec)
                        continue
                if pos.tp1_hit and hit_tp2:
                    tot, rec = self._full_close(state, pos, ts, float(pos.tp2_price), "tp1_then_tp2", i)
                    exit_pnl += tot
                    exits_out.append(rec)
                    continue
                if hit_trail:
                    tot, rec = self._full_close(state, pos, ts, float(pos.trail_stop_price), "tp1_then_trailing_stop", i)
                    exit_pnl += tot
                    exits_out.append(rec)
                    continue
                if hit_time_decay:
                    tot, rec = self._full_close(state, pos, ts, ask_close, "tp1_then_time_decay", i)
                    exit_pnl += tot
                    exits_out.append(rec)
                    continue

        return exit_pnl, exits_out

    def _compute_entry_sl_tp(
        self, sdir: str, sig: dict, entry_price: float
    ) -> Optional[tuple[float, float, float, float, float, float, bool, str, str]]:
        """scripts/backtest_tokyo_meanrev.py try_open_position lines 1407-1528."""
        sl_source = "pivot"
        tp_source = "pivot"
        min_tp_pips = self.min_tp_pips
        if sdir == "long":
            from_zone = bool(sig["from_zone"])
            sl_raw = (float(sig["S3"]) - self.sl_buf) if from_zone else (float(sig["S2"]) - self.sl_buf)
            if self.rejection_bonus_enabled and self.rejection_sl_improvement and bool(sig.get("rejection_confirmed", False)):
                rej_low = float(sig.get("rejection_low", np.nan))
                if np.isfinite(rej_low):
                    rej_sl_raw = rej_low - self.rejection_sl_buffer_pips * PIP_SIZE
                    base_sl_pips = (entry_price - sl_raw) / PIP_SIZE
                    rej_sl_pips = (entry_price - rej_sl_raw) / PIP_SIZE
                    if rej_sl_pips > 0 and rej_sl_pips < base_sl_pips:
                        sl_raw = rej_sl_raw
                        sl_source = "rejection_bonus"
            sl_pips = (entry_price - sl_raw) / PIP_SIZE
            sl_pips = max(self.min_sl_pips, min(self.max_sl_pips, sl_pips))
            sl_price = entry_price - sl_pips * PIP_SIZE
            tp_pivot = float(sig["P"])
            tp_mid = float(sig.get("session_midpoint", np.nan))
            if (
                self.session_env_enabled
                and self.session_env_use_for_tp
                and self.session_env_tp_mode == "nearest_of_pivot_or_midpoint"
                and np.isfinite(tp_mid)
            ):
                d_p = abs(tp_pivot - entry_price)
                d_m = abs(tp_mid - entry_price)
                tp2 = tp_mid if d_m <= d_p else tp_pivot
                tp_source = "midpoint" if d_m <= d_p else "pivot"
            else:
                tp2 = tp_pivot
                tp_source = "pivot"
            tp2_pips = (tp2 - entry_price) / PIP_SIZE
            if tp2_pips < min_tp_pips:
                tp2 = max(float(sig["R1"]), entry_price + min_tp_pips * PIP_SIZE)
                tp_source = "pivot_fallback"
            atr_pips = float(sig["atr_m15"]) / PIP_SIZE
            if self.tp_mode == "trail_only":
                tp1_pips = 9999.0
                tp1 = entry_price + tp1_pips * PIP_SIZE
                tp2 = float(tp1)
                local_tp_close_pct = 0.0
                tp_source = "trail_only"
            elif self.tp_mode == "pivot_v2":
                if from_zone:
                    tp1 = float(sig["S1"])
                    tp1_pips = max(0.0, (tp1 - entry_price) / PIP_SIZE)
                    local_tp_close_pct = self.partial_close_pct
                else:
                    tp1 = float(tp2)
                    tp1_pips = max(0.0, (tp1 - entry_price) / PIP_SIZE)
                    local_tp_close_pct = 1.0
            elif self.tp_mode == "single_pivot":
                tp1 = float(tp2)
                tp1_pips = max(0.0, (tp1 - entry_price) / PIP_SIZE)
                local_tp_close_pct = 1.0
            elif self.tp_mode == "single_atr":
                tp1_pips = min(self.single_tp_max_pips, max(self.single_tp_min_pips, self.single_tp_atr_mult * atr_pips))
                tp1 = entry_price + tp1_pips * PIP_SIZE
                tp2 = float(tp1)
                local_tp_close_pct = 1.0
                tp_source = "single_atr"
            else:
                tp1_pips = min(self.partial_tp_max_pips, max(self.partial_tp_min_pips, self.partial_tp_atr_mult * atr_pips))
                tp1 = entry_price + tp1_pips * PIP_SIZE
                local_tp_close_pct = self.partial_close_pct
        else:
            from_zone = bool(sig["from_zone"])
            sl_raw = (float(sig["R3"]) + self.sl_buf) if from_zone else (float(sig["R2"]) + self.sl_buf)
            if self.rejection_bonus_enabled and self.rejection_sl_improvement and bool(sig.get("rejection_confirmed", False)):
                rej_high = float(sig.get("rejection_high", np.nan))
                if np.isfinite(rej_high):
                    rej_sl_raw = rej_high + self.rejection_sl_buffer_pips * PIP_SIZE
                    base_sl_pips = (sl_raw - entry_price) / PIP_SIZE
                    rej_sl_pips = (rej_sl_raw - entry_price) / PIP_SIZE
                    if rej_sl_pips > 0 and rej_sl_pips < base_sl_pips:
                        sl_raw = rej_sl_raw
                        sl_source = "rejection_bonus"
            sl_pips = (sl_raw - entry_price) / PIP_SIZE
            sl_pips = max(self.min_sl_pips, min(self.max_sl_pips, sl_pips))
            sl_price = entry_price + sl_pips * PIP_SIZE
            tp_pivot = float(sig["P"])
            tp_mid = float(sig.get("session_midpoint", np.nan))
            if (
                self.session_env_enabled
                and self.session_env_use_for_tp
                and self.session_env_tp_mode == "nearest_of_pivot_or_midpoint"
                and np.isfinite(tp_mid)
            ):
                d_p = abs(tp_pivot - entry_price)
                d_m = abs(tp_mid - entry_price)
                tp2 = tp_mid if d_m <= d_p else tp_pivot
                tp_source = "midpoint" if d_m <= d_p else "pivot"
            else:
                tp2 = tp_pivot
                tp_source = "pivot"
            tp2_pips = (entry_price - tp2) / PIP_SIZE
            if tp2_pips < min_tp_pips:
                tp2 = min(float(sig["S1"]), entry_price - min_tp_pips * PIP_SIZE)
                tp_source = "pivot_fallback"
            atr_pips = float(sig["atr_m15"]) / PIP_SIZE
            if self.tp_mode == "trail_only":
                tp1_pips = 9999.0
                tp1 = entry_price - tp1_pips * PIP_SIZE
                tp2 = float(tp1)
                local_tp_close_pct = 0.0
                tp_source = "trail_only"
            elif self.tp_mode == "pivot_v2":
                if from_zone:
                    tp1 = float(sig["R1"])
                    tp1_pips = max(0.0, (entry_price - tp1) / PIP_SIZE)
                    local_tp_close_pct = self.partial_close_pct
                else:
                    tp1 = float(tp2)
                    tp1_pips = max(0.0, (entry_price - tp1) / PIP_SIZE)
                    local_tp_close_pct = 1.0
            elif self.tp_mode == "single_pivot":
                tp1 = float(tp2)
                tp1_pips = max(0.0, (entry_price - tp1) / PIP_SIZE)
                local_tp_close_pct = 1.0
            elif self.tp_mode == "single_atr":
                tp1_pips = min(self.single_tp_max_pips, max(self.single_tp_min_pips, self.single_tp_atr_mult * atr_pips))
                tp1 = entry_price - tp1_pips * PIP_SIZE
                tp2 = float(tp1)
                local_tp_close_pct = 1.0
                tp_source = "single_atr"
            else:
                tp1_pips = min(self.partial_tp_max_pips, max(self.partial_tp_min_pips, self.partial_tp_atr_mult * atr_pips))
                tp1 = entry_price - tp1_pips * PIP_SIZE
                local_tp_close_pct = self.partial_close_pct
        if sl_pips <= 0:
            return None
        return (sl_price, sl_pips, tp1, tp2, tp1_pips, local_tp_close_pct, from_zone, sl_source, tp_source)

    def try_unified_open(
        self,
        state: TokyoState,
        sig: dict,
        sdir: str,
        entry_price: float,
        ts: pd.Timestamp,
        i: int,
        row: pd.Series,
        sst: dict,
        spread_pips_now: float,
        margin_avail: float,
        peak_equity: float,
        equity: float,
        trade_id_counter: list[int],
        bar_idx: int,
        ownership_cell: str,
        regime_at_entry: str,
        equity_at_entry: float,
        margin_used_at_entry: float,
    ) -> bool:
        """scripts/backtest_tokyo_meanrev.py try_open_position lines 1334-1652 (unified margin)."""
        if bool(sst.get("stopped", False)):
            state.diag["blocked_session_stopped"] += 1
            return False
        minute_now = int(row.get("minute_of_day_utc", 0))
        entry_window_ok = self.in_window(minute_now, self.entry_start_min, self.entry_end_min)
        if self.block_new_entries_minutes_before_end > 0 and self.in_window(minute_now, self.start_min, self.end_min):
            mins_to_end = _minutes_to_session_end(minute_now, self.start_min, self.end_min)
            if mins_to_end <= self.block_new_entries_minutes_before_end:
                entry_window_ok = False
        if not entry_window_ok:
            state.diag["blocked_entry_window"] += 1
            return False
        if self.consec_pause_enabled and sst.get("loss_pause_until") is not None and ts < pd.Timestamp(sst["loss_pause_until"]):
            state.diag["blocked_consecutive_loss_pause"] += 1
            return False
        if self.stop_after_consecutive_losses > 0 and int(sst.get("consec_losses", 0)) >= self.stop_after_consecutive_losses:
            state.diag["blocked_consecutive_loss_stop"] += 1
            sst["stopped"] = True
            return False
        if self.session_loss_stop_pct > 0 and float(sst.get("session_pnl_usd", 0.0)) <= (
            -self.session_loss_stop_pct * float(sst.get("session_start_equity", equity))
        ):
            state.diag["blocked_session_pnl_stop"] += 1
            sst["stopped"] = True
            return False
        if self.max_trades_session > 0 and int(sst["trades"]) >= self.max_trades_session:
            state.diag["blocked_max_trades_per_session"] += 1
            return False
        if len(state.open_positions) >= self.max_open:
            state.diag["blocked_max_open_cap"] += 1
            return False
        if float(spread_pips_now) > self.max_entry_spread_pips:
            state.diag["blocked_max_entry_spread"] += 1
            return False
        if sst["last_entry_time"] is not None and (ts - pd.Timestamp(sst["last_entry_time"])).total_seconds() < self.min_entry_gap_min * 60.0:
            state.diag["blocked_min_entry_gap"] += 1
            return False
        last_stop_key = "last_stop_time_long" if sdir == "long" else "last_stop_time_short"
        if sst.get(last_stop_key) is not None and (ts - pd.Timestamp(sst[last_stop_key])).total_seconds() < self.no_reentry_stop_min * 60.0:
            state.diag["blocked_reentry_after_stop"] += 1
            return False
        if self.adx_filter_enabled:
            adx_now = float(row.get("adx_m15", np.nan))
            if pd.isna(adx_now):
                state.diag["blocked_missing_adx"] += 1
                return False
            adx_cap_here = float(self.adx_day_max_by_day.get(str(row.get("utc_day_name", "")), self.adx_max_for_entry))
            if adx_now > adx_cap_here:
                state.diag["blocked_adx_filter"] += 1
                return False
        if self.chase_enabled:
            bid_now, ask_now = tokyo_bt.get_bid_ask(float(row["close"]), float(spread_pips_now))
            conf_close = float(sig.get("confirmation_close", float(row["close"])))
            if sdir == "long" and ask_now > (conf_close + self.chase_max_pips * PIP_SIZE):
                state.diag["blocked_chase_filter"] += 1
                return False
            if sdir == "short" and bid_now < (conf_close - self.chase_max_pips * PIP_SIZE):
                state.diag["blocked_chase_filter"] += 1
                return False
        stp = self._compute_entry_sl_tp(sdir, sig, entry_price)
        if stp is None:
            state.diag["blocked_invalid_sl_distance"] += 1
            return False
        sl_price, sl_pips, tp1, tp2, tp1_pips, local_tp_close_pct, from_zone, sl_source, tp_source = stp
        dd_mult = 1.0
        dd_tier = "full"
        if self.dd_enabled:
            dd_now = ((peak_equity - equity) / peak_equity * 100.0) if peak_equity > 0 else 0.0
            if dd_now > self.dd_t2:
                dd_mult = max(0.0, 1.0 - self.dd_t2_red)
                dd_tier = "tier2"
            elif dd_now > self.dd_t1:
                dd_mult = max(0.0, 1.0 - self.dd_t1_red)
                dd_tier = "tier1"
        hour_mult = 1.0
        if self.hp_enabled:
            hour_mult = float(self.hp_mults.get(str(int(row["hour_utc"])), 1.0))
        ss_size_mult = 1.0
        if self.ss_sizing_enabled:
            ss_tier = str(sig.get("signal_strength_tier", "weak")).strip().lower()
            if ss_tier == "strong":
                ss_size_mult = self.ss_sizing_strong_mult
            elif ss_tier == "moderate":
                ss_size_mult = self.ss_sizing_moderate_mult
            else:
                ss_size_mult = self.ss_sizing_weak_mult
        total_size_mult = float(sig.get("regime_mult", 1.0)) * float(sig.get("quality_mult", 1.0)) * hour_mult * dd_mult
        total_size_mult *= float(ss_size_mult)
        day_name_here = str(row.get("utc_day_name", ""))
        risk_pct_local = float(self.risk_pct) * float(self.day_risk_multipliers.get(day_name_here, 1.0))
        units = math.floor((equity * risk_pct_local) / (sl_pips * (PIP_SIZE / max(1e-9, entry_price))))
        units = int(math.floor(units * max(0.0, total_size_mult)))
        units = int(max(0, min(self.max_units, units)))
        if units < 1:
            state.diag["blocked_units_lt_1"] += 1
            return False
        req_m = float(units) / UNIFIED_MARGIN_LEVERAGE
        if req_m > float(margin_avail) + 1e-9:
            state.diag["blocked_margin_cap"] += 1
            return False
        trade_id_counter[0] += 1
        tid = trade_id_counter[0]
        pos = Position(
            trade_id=tid,
            direction=sdir,
            entry_time=ts,
            entry_session_day=str(sig["session_day"]),
            entry_price=float(entry_price),
            sl_price=float(sl_price),
            tp1_price=float(tp1),
            tp2_price=float(tp2),
            tp1_close_pct=float(local_tp_close_pct),
            units_initial=int(units),
            units_remaining=int(units),
            confluence_score=int(sig["confluence_score"]),
            entry_indicators={
                "pivot_P": float(sig["P"]),
                "pivot_R1": float(sig["R1"]),
                "pivot_R2": float(sig["R2"]),
                "pivot_S1": float(sig["S1"]),
                "pivot_S2": float(sig["S2"]),
                "bb_upper": float(sig["bb_upper"]),
                "bb_mid": float(sig["bb_mid"]),
                "bb_lower": float(sig["bb_lower"]),
                "sar_value": float(sig["sar_value"]),
                "sar_direction": str(sig["sar_direction"]),
                "rsi_m5": float(sig["rsi_m5"]),
                "atr_m15": float(sig["atr_m15"]),
            },
            from_s2_or_r2_zone=bool(from_zone),
            signal_time=pd.Timestamp(sig["signal_time"]),
            confirmation_delay_candles=int(sig.get("confirmation_delay_candles", 0)),
            partial_tp_pips=float(tp1_pips),
            initial_sl_pips=float(sl_pips),
            regime_label=str(sig.get("regime_label", "neutral")),
            confluence_combo=str(sig.get("confluence_combo", "")),
            quality_label=str(sig.get("quality_label", "medium")),
            size_mult_total=float(total_size_mult),
            size_mult_regime=float(sig.get("regime_mult", 1.0)),
            size_mult_quality=float(sig.get("quality_mult", 1.0)),
            size_mult_hour=float(hour_mult),
            size_mult_dd=float(dd_mult),
            dd_tier=dd_tier,
            signal_strength_score=int(sig.get("signal_strength_score", 0)),
            signal_strength_tier=str(sig.get("signal_strength_tier", "weak")),
            entry_delay_type=str(sig.get("entry_delay_type", "immediate")),
            rejection_confirmed=bool(sig.get("rejection_confirmed", False)),
            divergence_present=bool(sig.get("divergence_present", False)),
            inside_ir=bool(sig.get("inside_ir", False)),
            quality_markers=str(sig.get("quality_markers", "")),
            sl_source=str(sl_source),
            tp_source=str(tp_source),
            distance_to_ir_boundary_pips=float(sig.get("distance_to_ir_boundary_pips", 0.0) if not pd.isna(sig.get("distance_to_ir_boundary_pips", np.nan)) else 0.0),
            distance_to_midpoint_pips=float(sig.get("distance_to_midpoint_pips", 0.0) if not pd.isna(sig.get("distance_to_midpoint_pips", np.nan)) else 0.0),
            distance_to_pivot_pips=float(sig.get("distance_to_pivot_pips", 0.0) if not pd.isna(sig.get("distance_to_pivot_pips", np.nan)) else 0.0),
        )
        setattr(pos, "ownership_cell", ownership_cell)
        setattr(pos, "regime_label_at_entry", regime_at_entry)
        setattr(pos, "entry_bar_idx", bar_idx)
        setattr(pos, "equity_at_entry", equity_at_entry)
        setattr(pos, "margin_used_at_entry", margin_used_at_entry)
        state.open_positions.append(pos)
        sst["trades"] = int(sst["trades"]) + 1
        sst["last_entry_time"] = ts
        sst.setdefault("entry_confluence_scores", []).append(int(sig["confluence_score"]))
        state.diag["entries_total"] += 1
        state.diag[f"entries_{sdir}"] += 1
        state.diag["bars_with_entry_triggered"] += 1
        return True

    def advance_bar(
        self,
        state: TokyoState,
        *,
        bar_idx: int,
        m1_full: pd.DataFrame,
        m5_full: pd.DataFrame,
        m15_full: pd.DataFrame,
        equity: float,
        peak_equity: float,
        margin_used_other: float,
        exec_gate: Optional[Callable[[dict[str, Any]], tuple[bool, str]]],
        trade_id_counter: list[int],
        ownership_cell: str,
        regime_label: str,
    ) -> TokyoBarResult:
        """
        One M1 bar: exits, session rules, gates, V14 eval, confirmation, opens.
        Order mirrors scripts/backtest_tokyo_meanrev.py for i, row loop (lines 1654+).
        """
        result = TokyoBarResult()
        row = m1_full.iloc[bar_idx]
        ts = pd.Timestamp(row["time"])
        i = bar_idx
        spread_pips_now = _get_tokyo_spread(ts, self.cfg)

        m5_t = m5_full["time"]
        m15_t = m15_full["time"]
        while state.m5_asof_idx + 1 < len(m5_full) and pd.Timestamp(m5_t.iloc[state.m5_asof_idx + 1]) <= ts:
            state.m5_asof_idx += 1
        while state.m15_asof_idx + 1 < len(m15_full) and pd.Timestamp(m15_t.iloc[state.m15_asof_idx + 1]) <= ts:
            state.m15_asof_idx += 1
        if state.m5_asof_idx >= 0:
            assert pd.Timestamp(m5_t.iloc[state.m5_asof_idx]) <= ts, (
                f"Tokyo lookahead: M5 has {m5_t.iloc[state.m5_asof_idx]} but current bar is {ts}"
            )
        if state.m15_asof_idx >= 0:
            assert pd.Timestamp(m15_t.iloc[state.m15_asof_idx]) <= ts, (
                f"Tokyo lookahead: M15 has {m15_t.iloc[state.m15_asof_idx]} but current bar is {ts}"
            )

        mid_open = float(row["open"])
        mid_high = float(row["high"])
        mid_low = float(row["low"])
        mid_close = float(row["close"])
        _, _ = tokyo_bt.get_bid_ask(mid_open, spread_pips_now)
        bid_high, ask_high = tokyo_bt.get_bid_ask(mid_high, spread_pips_now)
        bid_low, ask_low = tokyo_bt.get_bid_ask(mid_low, spread_pips_now)
        bid_close, ask_close = tokyo_bt.get_bid_ask(mid_close, spread_pips_now)

        in_session = bool(row["in_tokyo_session"]) and bool(row["allowed_trading_day"])
        session_day = str(row["session_day_jst"])
        minute_utc = int(row["minute_of_day_utc"])
        remaining_session_minutes = (
            _minutes_to_session_end(minute_utc, self.start_min, self.end_min) if in_session else None
        )

        for sig in list(state.pending_confirmation):
            if i > int(sig["expiry_index"]):
                state.pending_confirmation.remove(sig)
                state.diag["signals_expired"] += 1

        exit_pnl, exits = self.manage_positions(
            state,
            i=i,
            ts=ts,
            row=row,
            in_session=in_session,
            remaining_session_minutes=remaining_session_minutes,
            bid_close=bid_close,
            ask_close=ask_close,
            bid_high=bid_high,
            bid_low=bid_low,
            ask_high=ask_high,
            ask_low=ask_low,
            equity=equity,
        )
        result.exits.extend(exits)
        result.equity_delta += exit_pnl
        equity2 = equity + exit_pnl

        if not in_session:
            for pos in list(state.open_positions):
                px = bid_close if pos.direction == "long" else ask_close
                tot, rec = self._full_close(
                    state,
                    pos,
                    ts,
                    px,
                    "tp1_then_session_close" if pos.tp1_hit else "session_close",
                    bar_idx,
                )
                result.equity_delta += tot
                result.exits.append(rec)
            result.margin_used = tokyo_margin_used_open_positions(state.open_positions)
            result.diagnostics = dict(state.diag)
            return result

        sst = self.ensure_session(state, session_day, row, equity2)
        sst["session_high"] = max(float(sst.get("session_high", mid_high)), float(mid_high))
        sst["session_low"] = min(float(sst.get("session_low", mid_low)), float(mid_low))
        if self.session_env_enabled and not bool(sst.get("ir_ready", False)):
            if ts <= pd.Timestamp(sst.get("warmup_end_ts")):
                sst["ir_high"] = max(float(sst.get("ir_high", mid_high)), float(mid_high))
                sst["ir_low"] = min(float(sst.get("ir_low", mid_low)), float(mid_low))
            else:
                sst["ir_ready"] = True

        if self.lunch_block_enabled and self.in_window(
            minute_utc, self.lunch_start_min, self.lunch_end_min
        ):
            result.margin_used = tokyo_margin_used_open_positions(state.open_positions)
            result.diagnostics = dict(state.diag)
            return result

        if self.daily_range_enabled and not bool(sst.get("daily_range_allowed", True)):
            result.margin_used = tokyo_margin_used_open_positions(state.open_positions)
            result.diagnostics = dict(state.diag)
            return result
        if self.trend_skip_enabled and not bool(sst.get("trend_skip_allowed", True)):
            result.margin_used = tokyo_margin_used_open_positions(state.open_positions)
            result.diagnostics = dict(state.diag)
            return result
        if self.regime_switch_enabled and not bool(sst.get("regime_switch_allowed", True)):
            result.margin_used = tokyo_margin_used_open_positions(state.open_positions)
            result.diagnostics = dict(state.diag)
            return result

        if self.stop_after_consecutive_losses > 0 and int(sst.get("consec_losses", 0)) >= self.stop_after_consecutive_losses:
            sst["stopped"] = True
            result.margin_used = tokyo_margin_used_open_positions(state.open_positions)
            result.diagnostics = dict(state.diag)
            return result

        piv_cols = ["pivot_P", "pivot_R1", "pivot_R2", "pivot_R3", "pivot_S1", "pivot_S2", "pivot_S3"]
        if any(pd.isna(row[c]) for c in piv_cols):
            result.margin_used = tokyo_margin_used_open_positions(state.open_positions)
            result.diagnostics = dict(state.diag)
            return result
        if (
            pd.isna(row["bb_upper"])
            or pd.isna(row["bb_lower"])
            or pd.isna(row["bb_mid"])
            or pd.isna(row["rsi_m5"])
            or pd.isna(row["atr_m15"])
        ):
            result.margin_used = tokyo_margin_used_open_positions(state.open_positions)
            result.diagnostics = dict(state.diag)
            return result

        breakout_blocked = False
        if self.breakout_mode == "from_open":
            moved_up_pips = (float(sst["session_high"]) - float(sst["session_open_price"])) / PIP_SIZE
            moved_dn_pips = (float(sst["session_open_price"]) - float(sst["session_low"])) / PIP_SIZE
            moved_from_open_pips = max(moved_up_pips, moved_dn_pips)
            if moved_from_open_pips > self.breakout_disable_pips:
                sst["stopped"] = True
                breakout_blocked = True
            if bool(sst["stopped"]):
                breakout_blocked = True
        elif self.breakout_mode == "rolling":
            rw = sst["rolling_window"]
            rw.append((ts, mid_high, mid_low))
            cutoff = ts - pd.Timedelta(minutes=self.rolling_window_minutes)
            while rw and pd.Timestamp(rw[0][0]) < cutoff:
                rw.pop(0)
            if sst.get("breakout_cooldown_until") is not None and ts < pd.Timestamp(sst["breakout_cooldown_until"]):
                breakout_blocked = True
            else:
                highs = [x[1] for x in rw]
                lows = [x[2] for x in rw]
                rolling_range_pips = (max(highs) - min(lows)) / PIP_SIZE if highs and lows else 0.0
                if rolling_range_pips > self.rolling_range_threshold_pips:
                    sst["breakout_cooldown_until"] = ts + pd.Timedelta(minutes=self.breakout_cooldown_minutes)
                    breakout_blocked = True
                else:
                    sst["breakout_cooldown_until"] = None
        if breakout_blocked:
            result.margin_used = tokyo_margin_used_open_positions(state.open_positions)
            result.diagnostics = dict(state.diag)
            return result

        regime_col = "bb_regime_expanding3" if self.regime_filter_mode in {"expanding3", "bb_width_expanding3"} else "bb_regime"
        if str(row.get(regime_col, "trending")) != "ranging":
            result.margin_used = tokyo_margin_used_open_positions(state.open_positions)
            result.diagnostics = dict(state.diag)
            return result

        if self.atr_gate_enabled:
            if self.atr_pct_enabled:
                atr_pct_rank_val = float(row.get("atr_m15_percentile_rank", np.nan))
                if not np.isfinite(atr_pct_rank_val):
                    result.margin_used = tokyo_margin_used_open_positions(state.open_positions)
                    result.diagnostics = dict(state.diag)
                    return result
                atr_pct_max = float(self.cfg["indicators"]["atr"]["percentile_filter"].get("max_percentile", 0.67))
                if atr_pct_rank_val > atr_pct_max:
                    result.margin_used = tokyo_margin_used_open_positions(state.open_positions)
                    result.diagnostics = dict(state.diag)
                    return result
            elif float(row["atr_m15"]) > self.atr_max:
                result.margin_used = tokyo_margin_used_open_positions(state.open_positions)
                result.diagnostics = dict(state.diag)
                return result

        if self.news_enabled and bool(row.get("news_blocked", False)):
            result.margin_used = tokyo_margin_used_open_positions(state.open_positions)
            result.diagnostics = dict(state.diag)
            return result

        P = float(row["pivot_P"])
        R1, R2, R3 = float(row["pivot_R1"]), float(row["pivot_R2"]), float(row["pivot_R3"])
        S1, S2, S3 = float(row["pivot_S1"]), float(row["pivot_S2"]), float(row["pivot_S3"])
        cand = evaluate_v14_entry_signal(
            row=row,
            mid_close=mid_close,
            mid_open=mid_open,
            mid_high=mid_high,
            mid_low=mid_low,
            pivot_levels={"P": P, "R1": R1, "R2": R2, "R3": R3, "S1": S1, "S2": S2, "S3": S3},
            cfg_params=self._v14_cfg_params,
            sst=sst,
        )

        if cand is None:
            result.margin_used = tokyo_margin_used_open_positions(state.open_positions)
            result.diagnostics = dict(state.diag)
            return result

        br = cand.get("_blocked_reason")
        if br == "combo_filter":
            state.diag["blocked_combo_filter"] += 1
            result.margin_used = tokyo_margin_used_open_positions(state.open_positions)
            result.diagnostics = dict(state.diag)
            return result
        if br == "signal_strength_filter":
            state.diag["blocked_signal_strength_filter"] += 1
            result.margin_used = tokyo_margin_used_open_positions(state.open_positions)
            result.diagnostics = dict(state.diag)
            return result
        if br == "low_quality_combo":
            state.diag["blocked_low_quality_combo"] += 1
            result.margin_used = tokyo_margin_used_open_positions(state.open_positions)
            result.diagnostics = dict(state.diag)
            return result

        direction = str(cand["direction"])
        combo = str(cand["confluence_combo"])
        sig_obj = {
            "direction": direction,
            "session_day": session_day,
            "signal_index": int(i),
            "signal_time": ts,
            "expiry_index": int(
                i
                + (
                    self.confirmation_window_bars
                    if self.confirmation_type == "m1"
                    else self.confirmation_window_bars * 5
                )
            ),
            "confluence_score": int(cand["confluence_score"]),
            "confluence_combo": combo,
            "signal_strength_score": int(cand["signal_strength_score"]),
            "signal_strength_tier": str(cand["signal_strength_tier"]),
            "quality_label": str(cand["quality_label"]),
            "quality_mult": float(cand["quality_mult"]),
            "regime_label": str(cand["regime_label"]),
            "regime_mult": float(cand["regime_mult"]),
            "from_zone": bool(cand["from_zone"]),
            "P": P,
            "R1": R1,
            "R2": R2,
            "R3": R3,
            "S1": S1,
            "S2": S2,
            "S3": S3,
            "bb_upper": float(cand["bb_upper"]),
            "bb_lower": float(cand["bb_lower"]),
            "sar_value": float(cand["sar_value"]),
            "sar_direction": str(cand["sar_direction"]),
            "rsi_m5": float(cand["rsi_m5"]),
            "atr_m15": float(cand["atr_m15"]),
            "bb_mid": float(cand["bb_mid"]),
            "rejection_confirmed": bool(cand["rejection_confirmed"]),
            "rejection_low": cand["rejection_low"],
            "rejection_high": cand["rejection_high"],
            "rejection_wick_ratio": cand["rejection_wick_ratio"],
            "divergence_present": bool(cand["divergence_present"]),
            "inside_ir": bool(cand["inside_ir"]),
            "quality_markers": str(cand["quality_markers"]),
            "session_midpoint": cand["session_midpoint"],
            "distance_to_ir_boundary_pips": cand["distance_to_ir_boundary_pips"],
            "distance_to_midpoint_pips": cand["distance_to_midpoint_pips"],
            "distance_to_pivot_pips": float(cand["distance_to_pivot_pips"]),
            "entry_delay_type": "immediate",
            "momentum_delays": 0,
        }
        state.diag["signals_generated"] += 1

        def _margin_avail_for_new() -> float:
            eq3 = equity + result.equity_delta
            tm = tokyo_margin_used_open_positions(state.open_positions)
            return float(eq3 - float(margin_used_other) - tm)

        if self.confirmation_enabled and self.confirmation_window_bars > 0:
            has_pending_same = any(
                ps["session_day"] == session_day
                and ps["direction"] == direction
                and i <= int(ps["expiry_index"])
                for ps in state.pending_confirmation
            )
            if not has_pending_same:
                state.pending_confirmation.append(sig_obj)
        else:
            sig_obj["confirmation_delay_candles"] = 0
            sig_obj["confirmation_close"] = float(mid_close)
            entry_px_now = ask_close if direction == "long" else bid_close
            side = "buy" if direction == "long" else "sell"
            if exec_gate is not None:
                ok_g, _rs = exec_gate(
                    {
                        "session": "tokyo_v14",
                        "side": side,
                        "ownership_cell": ownership_cell,
                        "regime_label": regime_label,
                        "strategy_tag": "tokyo_v14",
                    }
                )
                if not ok_g:
                    state.diag["blocked_exec_gate"] += 1
                    result.margin_used = tokyo_margin_used_open_positions(state.open_positions)
                    result.diagnostics = dict(state.diag)
                    return result
            opened = self.try_unified_open(
                state,
                sig_obj,
                direction,
                float(entry_px_now),
                ts,
                i,
                row,
                sst,
                spread_pips_now,
                _margin_avail_for_new(),
                peak_equity,
                equity + result.equity_delta,
                trade_id_counter,
                bar_idx,
                ownership_cell,
                regime_label,
                equity + result.equity_delta,
                float(margin_used_other) + tokyo_margin_used_open_positions(state.open_positions),
            )
            if opened:
                result.new_entries.append({"trade_id": trade_id_counter[0], "session": "tokyo_v14"})
                state.diag["tokyo_entries_placed"] += 1
            else:
                state.diag["tokyo_entries_blocked_post_eval"] += 1

        for sig in list(state.pending_confirmation):
            if sig["session_day"] != session_day:
                continue
            if i <= int(sig["signal_index"]):
                continue
            if i > int(sig["expiry_index"]):
                continue
            sdir = str(sig["direction"])
            delay_until = sig.get("momentum_delay_until", None)
            in_delayed_recheck = delay_until is not None and i >= int(delay_until)
            if delay_until is not None and i < int(delay_until):
                continue
            if self.confirmation_type == "m5":
                if int(row["minute_utc"]) % 5 != 0:
                    continue
                m5_open = float(row.get("m5_open", np.nan))
                m5_close = float(row.get("m5_close", np.nan))
                if pd.isna(m5_open) or pd.isna(m5_close):
                    continue
                confirmed = (m5_close > m5_open) if sdir == "long" else (m5_close < m5_open)
            else:
                confirmed = (mid_close > mid_open) if sdir == "long" else (mid_close < mid_open)
            if not in_delayed_recheck and not confirmed:
                continue
            if self.momentum_enabled:
                j0 = max(0, i - self.momentum_lookback + 1)
                y = m1_full.iloc[j0 : i + 1]["close"].to_numpy(dtype=float)
                if len(y) >= 2:
                    x = np.arange(len(y), dtype=float)
                    xv = x - x.mean()
                    yv = y - y.mean()
                    den = float((xv * xv).sum())
                    slope_price = float((xv * yv).sum() / den) if den > 0 else 0.0
                    slope_pips = slope_price / PIP_SIZE
                else:
                    slope_pips = 0.0
                adverse = (slope_pips < -self.momentum_slope_th) if sdir == "long" else (slope_pips > self.momentum_slope_th)
                if adverse:
                    if int(sig.get("momentum_delays", 0)) < self.momentum_max_delays and not in_delayed_recheck:
                        sig["momentum_delays"] = int(sig.get("momentum_delays", 0)) + 1
                        sig["momentum_delay_until"] = int(i + self.momentum_delay_candles)
                        sig["entry_delay_type"] = "delayed"
                        continue
                    state.pending_confirmation.remove(sig)
                    state.diag["signals_expired"] += 1
                    continue
            sig["confirmation_delay_candles"] = int(i - int(sig["signal_index"]))
            sig["confirmation_close"] = float(mid_close)
            if self.entry_imp_enabled:
                state.pending_confirmation.remove(sig)
                state.diag["entry_improvement_skipped_unified"] += 1
                continue
            entry_px = ask_close if sdir == "long" else bid_close
            side = "buy" if sdir == "long" else "sell"
            if exec_gate is not None:
                ok_g, _rs = exec_gate(
                    {
                        "session": "tokyo_v14",
                        "side": side,
                        "ownership_cell": ownership_cell,
                        "regime_label": regime_label,
                        "strategy_tag": "tokyo_v14",
                    }
                )
                if not ok_g:
                    state.pending_confirmation.remove(sig)
                    state.diag["blocked_exec_gate"] += 1
                    continue
            opened = self.try_unified_open(
                state,
                sig,
                sdir,
                float(entry_px),
                ts,
                i,
                row,
                sst,
                spread_pips_now,
                _margin_avail_for_new(),
                peak_equity,
                equity + result.equity_delta,
                trade_id_counter,
                bar_idx,
                ownership_cell,
                regime_label,
                equity + result.equity_delta,
                float(margin_used_other) + tokyo_margin_used_open_positions(state.open_positions),
            )
            state.pending_confirmation.remove(sig)
            if opened:
                result.new_entries.append({"trade_id": trade_id_counter[0], "session": "tokyo_v14"})
                state.diag["tokyo_entries_placed"] += 1

        result.margin_used = tokyo_margin_used_open_positions(state.open_positions)
        result.diagnostics = dict(state.diag)
        return result


def advance_tokyo_bar(
    bar_idx: int,
    m1_row: Any,
    m5_indicators: pd.DataFrame,
    m15_indicators: pd.DataFrame,
    pivots: dict,
    cfg: dict,
    state: TokyoState,
    equity: float,
    exec_gate: Optional[Callable[[dict[str, Any]], tuple[bool, str]]] = None,
    margin_avail: Optional[float] = None,
    spread_pips: float = 1.5,
    pip_size: float = 0.01,
    pip_value: float = 9.30,
    *,
    m1_full: pd.DataFrame,
    peak_equity: float,
    margin_used_other: float,
    trade_id_counter: list[int],
    ownership_cell: str,
    regime_label: str,
) -> TokyoBarResult:
    """
    Per-bar Tokyo API (pivots/m1_row/pip_value are accepted for compatibility; row must match m1_full.iloc[bar_idx]).
    """
    if state.engine is None:
        state.engine = TokyoUnifiedEngine(cfg)
    eng: TokyoUnifiedEngine = state.engine
    _ = (m1_row, pivots, spread_pips, pip_size, pip_value, margin_avail)
    return eng.advance_bar(
        state,
        bar_idx=bar_idx,
        m1_full=m1_full,
        m5_full=m5_indicators,
        m15_full=m15_indicators,
        equity=equity,
        peak_equity=peak_equity,
        margin_used_other=margin_used_other,
        exec_gate=exec_gate,
        trade_id_counter=trade_id_counter,
        ownership_cell=ownership_cell,
        regime_label=regime_label,
    )
