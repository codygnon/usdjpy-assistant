#!/usr/bin/env python3
"""
Tokyo session mean-reversion backtest runner driven by JSON config.

Usage:
  python3 scripts/backtest_tokyo_meanrev.py \
    --config research_out/tokyo_mean_reversion_v1_config.json
"""

from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right, insort
import json
import math
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PIP_SIZE = 0.01
TOKYO_TZ = "Asia/Tokyo"


@dataclass
class Position:
    trade_id: int
    direction: str  # long | short
    entry_time: pd.Timestamp
    entry_session_day: str
    entry_price: float  # ask for long, bid for short
    sl_price: float
    tp1_price: float
    tp2_price: float
    tp1_close_pct: float
    units_initial: int
    units_remaining: int
    confluence_score: int
    entry_indicators: dict
    from_s2_or_r2_zone: bool
    signal_time: Optional[pd.Timestamp] = None
    confirmation_delay_candles: int = 0
    partial_tp_pips: float = 0.0
    initial_sl_pips: float = 0.0
    moved_to_breakeven: bool = False
    regime_label: str = "neutral"
    confluence_combo: str = ""
    quality_label: str = "medium"
    size_mult_total: float = 1.0
    size_mult_regime: float = 1.0
    size_mult_quality: float = 1.0
    size_mult_hour: float = 1.0
    size_mult_dd: float = 1.0
    dd_tier: str = "full"
    signal_strength_score: int = 0
    signal_strength_tier: str = "weak"
    entry_delay_type: str = "immediate"
    max_profit_seen_pips: float = 0.0
    max_adverse_seen_pips: float = 0.0
    rejection_confirmed: bool = False
    divergence_present: bool = False
    inside_ir: bool = False
    quality_markers: str = ""
    sl_source: str = "pivot"
    tp_source: str = "pivot"
    distance_to_ir_boundary_pips: float = 0.0
    distance_to_midpoint_pips: float = 0.0
    distance_to_pivot_pips: float = 0.0
    tp1_hit: bool = False
    trail_active: bool = False
    trail_stop_price: Optional[float] = None
    realized_usd: float = 0.0
    realized_pip_units: float = 0.0
    exit_reason: Optional[str] = None
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tokyo mean-reversion backtest")
    p.add_argument("--config", required=True, help="Path to strategy config JSON")
    return p.parse_args()


def load_m1(path: str) -> pd.DataFrame:
    f = Path(path)
    if not f.exists():
        raise FileNotFoundError(f"Missing CSV: {f}")
    df = pd.read_csv(f)
    need = {"time", "open", "high", "low", "close"}
    if not need.issubset(df.columns):
        raise ValueError(f"{f} missing columns: {sorted(need)}")
    cols = ["time", "open", "high", "low", "close"] + ([c for c in ["volume"] if c in df.columns])
    out = df[cols].copy()
    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
    for c in ("open", "high", "low", "close"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if "volume" in out.columns:
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    out = out.dropna(subset=["time", "open", "high", "low", "close"]).sort_values("time").reset_index(drop=True)
    return out


def resample_ohlc_continuous(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    r = (
        df.set_index("time")
        .resample(rule, label="right", closed="right")
        .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
        .dropna()
        .reset_index()
    )
    return r.sort_values("time").reset_index(drop=True)


def rolling_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_up = up.rolling(period, min_periods=period).mean()
    avg_down = down.rolling(period, min_periods=period).mean()
    rs = avg_up / avg_down.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def rolling_atr_price(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def rolling_adx(df: pd.DataFrame, period: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period, min_periods=period).mean()
    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).rolling(period, min_periods=period).mean() / atr.replace(0.0, np.nan)
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).rolling(period, min_periods=period).mean() / atr.replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx = dx.rolling(period, min_periods=period).mean()
    return adx, plus_di, minus_di


def compute_psar(df: pd.DataFrame, af_start: float, af_step: float, af_max: float) -> tuple[pd.Series, pd.Series]:
    n = len(df)
    if n == 0:
        return pd.Series(dtype=float), pd.Series(dtype=object)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    psar = np.zeros(n, dtype=float)
    dirn = np.empty(n, dtype=object)

    bull = True if n < 2 else close[1] >= close[0]
    ep = high[0] if bull else low[0]
    psar[0] = low[0] if bull else high[0]
    af = af_start
    dirn[0] = "bullish" if bull else "bearish"

    for i in range(1, n):
        prev_psar = psar[i - 1]
        cand = prev_psar + af * (ep - prev_psar)
        if bull:
            cand = min(cand, low[i - 1], low[i - 2] if i > 1 else low[i - 1])
            if low[i] < cand:
                bull = False
                psar[i] = ep
                ep = low[i]
                af = af_start
            else:
                psar[i] = cand
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_step, af_max)
        else:
            cand = max(cand, high[i - 1], high[i - 2] if i > 1 else high[i - 1])
            if high[i] > cand:
                bull = True
                psar[i] = ep
                ep = high[i]
                af = af_start
            else:
                psar[i] = cand
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_step, af_max)
        dirn[i] = "bullish" if bull else "bearish"
    return pd.Series(psar, index=df.index), pd.Series(dirn, index=df.index)


def load_news_events(cfg: dict) -> pd.DataFrame:
    nf = cfg.get("news_filter", {})
    if not bool(nf.get("enabled", False)):
        return pd.DataFrame(columns=["event_ts", "impact", "event_id", "source"])
    p = Path(str(nf.get("calendar_path", "research_out/v5_scheduled_events_utc.csv")))
    if not p.exists():
        return pd.DataFrame(columns=["event_ts", "impact", "event_id", "source"])
    ev = pd.read_csv(p)
    if not {"date", "time_utc", "event_id", "impact"}.issubset(ev.columns):
        return pd.DataFrame(columns=["event_ts", "impact", "event_id", "source"])
    ev["event_ts"] = pd.to_datetime(ev["date"].astype(str) + " " + ev["time_utc"].astype(str), utc=True, errors="coerce")
    ev = ev.dropna(subset=["event_ts"]).copy()
    if "source" not in ev.columns:
        ev["source"] = ""
    ev["impact"] = ev["impact"].astype(str).str.lower()
    return ev.sort_values("event_ts").reset_index(drop=True)


def add_indicators(m1: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = m1.copy()
    # Keep indicators continuous over the full dataset (no session-boundary resets).
    m5 = resample_ohlc_continuous(out, "5min")
    m15 = resample_ohlc_continuous(out, "15min")

    # M5 BB + RSI + width regime
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
    m5["rsi_m5"] = rolling_rsi(m5["close"], rsi_p)
    m5["bb_width"] = (m5["bb_upper"] - m5["bb_lower"]) / m5["bb_mid"].replace(0.0, np.nan)
    m5["bb_width_cutoff"] = m5["bb_width"].rolling(bb_pct_window, min_periods=bb_pct_window).quantile(bb_pct_cutoff)
    m5["bb_regime"] = np.where(m5["bb_width"] < m5["bb_width_cutoff"], "ranging", "trending")
    # V2 regime mode: trending only when prior 3 BB-width values are strictly increasing.
    m5["bb_width_expanding3"] = (
        (m5["bb_width"].shift(1) > m5["bb_width"].shift(2))
        & (m5["bb_width"].shift(2) > m5["bb_width"].shift(3))
    )
    m5["bb_regime_expanding3"] = np.where(m5["bb_width_expanding3"].fillna(False), "trending", "ranging")
    m5["bb_width_q_high"] = m5["bb_width"].rolling(bb_rank_lookback, min_periods=bb_rank_lookback).quantile(bb_high_q)
    m5["bb_width_q_low"] = m5["bb_width"].rolling(bb_rank_lookback, min_periods=bb_rank_lookback).quantile(bb_low_q)

    # V10: rejection-candle tracking on M5 (bonus/quality layer).
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

    # V10: RSI(7) divergence tracking on M5 (log-only by default).
    div_cfg = cfg.get("rsi_divergence_tracking", {})
    div_enabled = bool(div_cfg.get("enabled", False))
    div_rsi_period = int(div_cfg.get("rsi_period", 7))
    div_lb_min = int(div_cfg.get("lookback_min", 3))
    div_lb_max = int(div_cfg.get("lookback_max", 10))
    div_bull_rsi_max = float(div_cfg.get("bullish_rsi_max", 45.0))
    div_bear_rsi_min = float(div_cfg.get("bearish_rsi_min", 55.0))
    m5["rsi7_m5"] = rolling_rsi(m5["close"], div_rsi_period)
    bull_div = np.zeros(len(m5), dtype=bool)
    bear_div = np.zeros(len(m5), dtype=bool)
    if div_enabled and len(m5):
        for i in range(len(m5)):
            j0 = max(0, i - div_lb_max)
            j1 = max(0, i - div_lb_min + 1)
            if j1 <= j0:
                continue
            prev = m5.iloc[j0:j1]
            if prev.empty:
                continue
            cur_low = float(m5.iloc[i]["low"])
            cur_high = float(m5.iloc[i]["high"])
            cur_rsi = float(m5.iloc[i]["rsi7_m5"]) if not pd.isna(m5.iloc[i]["rsi7_m5"]) else np.nan
            if pd.isna(cur_rsi):
                continue
            low_idx = prev["low"].idxmin()
            high_idx = prev["high"].idxmax()
            prev_low = float(m5.loc[low_idx, "low"])
            prev_high = float(m5.loc[high_idx, "high"])
            prev_low_rsi = float(m5.loc[low_idx, "rsi7_m5"]) if not pd.isna(m5.loc[low_idx, "rsi7_m5"]) else np.nan
            prev_high_rsi = float(m5.loc[high_idx, "rsi7_m5"]) if not pd.isna(m5.loc[high_idx, "rsi7_m5"]) else np.nan
            if (cur_low < prev_low) and (not pd.isna(prev_low_rsi)) and (cur_rsi > prev_low_rsi) and (cur_rsi < div_bull_rsi_max):
                bull_div[i] = True
            if (cur_high > prev_high) and (not pd.isna(prev_high_rsi)) and (cur_rsi < prev_high_rsi) and (cur_rsi > div_bear_rsi_min):
                bear_div[i] = True
    m5["rsi_div_bull"] = bull_div
    m5["rsi_div_bear"] = bear_div
    m5["rsi_div_bull_recent"] = m5["rsi_div_bull"].rolling(3, min_periods=1).max().astype(bool)
    m5["rsi_div_bear_recent"] = m5["rsi_div_bear"].rolling(3, min_periods=1).max().astype(bool)

    # M15 ATR (price units)
    atr_p = int(cfg["indicators"]["atr"]["period"])
    atr_slow_p = int(rg.get("atr_slow_period", 50))
    m15["atr_m15"] = rolling_atr_price(m15, atr_p)
    m15["atr_m15_slow"] = rolling_atr_price(m15, atr_slow_p)
    adx_p = int(rg.get("adx_period", 14))
    m15["adx_m15"], m15["plus_di_m15"], m15["minus_di_m15"] = rolling_adx(m15, adx_p)

    # M1 PSAR
    af_start = float(cfg["indicators"]["parabolic_sar"]["af_start"])
    af_inc = float(cfg["indicators"]["parabolic_sar"]["af_increment"])
    af_max = float(cfg["indicators"]["parabolic_sar"]["af_max"])
    sar_lookback = int(
        max(
            cfg["entry_rules"]["long"].get("sar_flip_lookback_bars", 12),
            cfg["entry_rules"]["short"].get("sar_flip_lookback_bars", 12),
        )
    )
    out["sar_value"], out["sar_direction"] = compute_psar(out, af_start, af_inc, af_max)
    out["sar_flip_bullish"] = (out["sar_direction"] == "bullish") & (out["sar_direction"].shift(1) == "bearish")
    out["sar_flip_bearish"] = (out["sar_direction"] == "bearish") & (out["sar_direction"].shift(1) == "bullish")
    out["sar_flip_bullish_recent"] = out["sar_flip_bullish"].rolling(sar_lookback, min_periods=1).max().astype(bool)
    out["sar_flip_bearish_recent"] = out["sar_flip_bearish"].rolling(sar_lookback, min_periods=1).max().astype(bool)

    # Merge M5 + M15 indicators to M1
    m5_merge_cols = [
        "time",
        "bb_upper",
        "bb_mid",
        "bb_lower",
        "rsi_m5",
        "bb_width",
        "bb_regime",
        "bb_regime_expanding3",
        "bb_width_cutoff",
        "bb_width_expanding3",
        "bb_width_q_high",
        "bb_width_q_low",
        "rej_bull_m5",
        "rej_bear_m5",
        "rej_bull_recent",
        "rej_bear_recent",
        "rej_bull_low_recent",
        "rej_bear_high_recent",
        "rej_wick_ratio_bull",
        "rej_wick_ratio_bear",
        "rsi7_m5",
        "rsi_div_bull",
        "rsi_div_bear",
        "rsi_div_bull_recent",
        "rsi_div_bear_recent",
    ]
    m15_merge_cols = ["time", "atr_m15", "atr_m15_slow", "adx_m15", "plus_di_m15", "minus_di_m15"]
    out = pd.merge_asof(out.sort_values("time"), m5[m5_merge_cols].sort_values("time"), on="time", direction="backward")
    out = pd.merge_asof(out.sort_values("time"), m15[m15_merge_cols].sort_values("time"), on="time", direction="backward")

    # Daily pivots by NY close boundary (22:00 UTC day rollover).
    out["ny_day"] = (out["time"] - pd.Timedelta(hours=22)).dt.date
    day = out.groupby("ny_day").agg(day_high=("high", "max"), day_low=("low", "min"), day_close=("close", "last")).reset_index()
    day["prev_high"] = day["day_high"].shift(1)
    day["prev_low"] = day["day_low"].shift(1)
    day["prev_close"] = day["day_close"].shift(1)
    day["pivot_P"] = (day["prev_high"] + day["prev_low"] + day["prev_close"]) / 3.0
    rng = day["prev_high"] - day["prev_low"]
    day["pivot_R1"] = day["pivot_P"] + 0.382 * rng
    day["pivot_R2"] = day["pivot_P"] + 0.618 * rng
    day["pivot_R3"] = day["pivot_P"] + 1.000 * rng
    day["pivot_S1"] = day["pivot_P"] - 0.382 * rng
    day["pivot_S2"] = day["pivot_P"] - 0.618 * rng
    day["pivot_S3"] = day["pivot_P"] - 1.000 * rng
    piv = day[["ny_day", "pivot_P", "pivot_R1", "pivot_R2", "pivot_R3", "pivot_S1", "pivot_S2", "pivot_S3"]]
    out = out.merge(piv, on="ny_day", how="left")

    return out


def get_bid_ask(mid_price: float, spread_pips: float) -> tuple[float, float]:
    hs = spread_pips * PIP_SIZE / 2.0
    bid = float(mid_price) - hs
    ask = float(mid_price) + hs
    return bid, ask


def ts_hour(ts: pd.Timestamp) -> float:
    ts = pd.Timestamp(ts)
    return float(ts.hour) + float(ts.minute) / 60.0 + float(ts.second) / 3600.0


def compute_spread_pips(i: int, ts: pd.Timestamp, mode: str, avg: float, mn: float, mx: float) -> float:
    mode = str(mode).strip().lower()
    avg = float(avg)
    mn = float(mn)
    mx = float(mx)
    if mode == "fixed":
        return max(mn, min(mx, avg))
    if mode == "realistic":
        h = ts_hour(ts)
        # Deterministic session/rollover profile approximating live spread behavior.
        if 21.0 <= h or h < 0.5:
            base = 2.6  # rollover / low-liquidity widen
        elif 0.5 <= h < 2.0:
            base = 1.9  # Tokyo open
        elif 2.0 <= h < 6.0:
            base = 1.5  # Tokyo core
        elif 6.0 <= h < 9.0:
            base = 1.7
        elif 13.0 <= h < 16.0:
            base = 2.3  # NY overlap
        else:
            base = 1.6
        wiggle = 0.18 * math.sin(i * 0.017) + 0.07 * math.sin(i * 0.071)
        return max(mn, min(mx, base + wiggle))
    # variable mode: deterministic oscillation around avg
    x = avg + 0.35 * math.sin(i * 0.017) + 0.15 * math.sin(i * 0.071)
    return max(mn, min(mx, x))


def calc_leg_usd_pips(direction: str, entry_price: float, exit_price: float, units: int) -> tuple[float, float]:
    if direction == "long":
        pips = (float(exit_price) - float(entry_price)) / PIP_SIZE
        usd = pips * float(units) * (PIP_SIZE / max(1e-9, float(exit_price)))
    else:
        pips = (float(entry_price) - float(exit_price)) / PIP_SIZE
        usd = pips * float(units) * (PIP_SIZE / max(1e-9, float(exit_price)))
    return float(pips), float(usd)


def build_from_open_breakout_analysis(df: pd.DataFrame, threshold_pips: float) -> dict:
    rows = []
    for sday, g in df[df["in_tokyo_session"] & (df["weekday_jst"] < 5)].groupby("session_day_jst"):
        gg = g.sort_values("time")
        if gg.empty:
            continue
        session_open = float(gg.iloc[0]["open"])
        running_high = session_open
        running_low = session_open
        breach_minute_offset = None
        breach_side = None
        for _, r in gg.iterrows():
            running_high = max(running_high, float(r["high"]))
            running_low = min(running_low, float(r["low"]))
            up = (running_high - session_open) / PIP_SIZE
            dn = (session_open - running_low) / PIP_SIZE
            if breach_minute_offset is None and (up > threshold_pips or dn > threshold_pips):
                breach_minute_offset = int((pd.Timestamp(r["time"]) - pd.Timestamp(gg.iloc[0]["time"])).total_seconds() / 60.0)
                breach_side = "high_side" if up >= dn else "low_side"
        rows.append(
            {
                "session_day_jst": str(sday),
                "session_open_price": session_open,
                "session_high": float(gg["high"].max()),
                "session_low": float(gg["low"].min()),
                "threshold_pips": float(threshold_pips),
                "breached": breach_minute_offset is not None,
                "breach_minute_offset": breach_minute_offset,
                "breach_side": breach_side,
            }
        )
    rdf = pd.DataFrame(rows)
    breach_rows = rdf[rdf["breached"]] if not rdf.empty else pd.DataFrame()
    return {
        "summary": {
            "sessions": int(len(rdf)),
            "breached_sessions": int(len(breach_rows)),
            "breach_pct": float((len(breach_rows) / len(rdf) * 100.0) if len(rdf) else 0.0),
            "median_minutes_to_breach": float(breach_rows["breach_minute_offset"].median()) if not breach_rows.empty else None,
            "threshold_pips": float(threshold_pips),
        },
        "sessions": rows,
    }


def run_one(cfg: dict, run_cfg: dict) -> dict:
    df = add_indicators(load_m1(run_cfg["input_csv"]), cfg)
    df["time_utc"] = df["time"].dt.tz_convert("UTC")
    df["time_jst"] = df["time"].dt.tz_convert(TOKYO_TZ)
    df["session_day_jst"] = df["time_jst"].dt.date.astype(str)
    df["weekday_jst"] = df["time_jst"].dt.dayofweek
    df["utc_day_name"] = df["time_utc"].dt.day_name()
    df["hour_utc"] = df["time_utc"].dt.hour
    df["minute_utc"] = df["time_utc"].dt.minute
    df["minute_of_day_utc"] = df["hour_utc"] * 60 + df["minute_utc"]

    sf = cfg.get("session_filter", {})
    session_start_utc = str(sf.get("session_start_utc", "15:00"))
    session_end_utc = str(sf.get("session_end_utc", "00:00"))
    entry_start_utc = str(sf.get("entry_start_utc", session_start_utc))
    entry_end_utc = str(sf.get("entry_end_utc", session_end_utc))
    block_new_entries_minutes_before_end = int(
        sf.get("block_new_entries_minutes_before_end", sf.get("session_entry_cutoff_minutes", 0))
    )
    lunch_block_enabled = bool(sf.get("lunch_block_enabled", False))
    lunch_block_start_utc = str(sf.get("lunch_block_start_utc", "02:30"))
    lunch_block_end_utc = str(sf.get("lunch_block_end_utc", "03:30"))
    allowed_days = list(sf.get("allowed_trading_days", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]))
    allowed_days_set = set(allowed_days)

    def hhmm_to_minutes(s: str) -> int:
        hh, mm = s.strip().split(":")
        return int(hh) * 60 + int(mm)

    start_min = hhmm_to_minutes(session_start_utc)
    end_min = hhmm_to_minutes(session_end_utc)
    entry_start_min = hhmm_to_minutes(entry_start_utc)
    entry_end_min = hhmm_to_minutes(entry_end_utc)
    lunch_start_min = hhmm_to_minutes(lunch_block_start_utc)
    lunch_end_min = hhmm_to_minutes(lunch_block_end_utc)

    def in_window(minute_of_day_utc: int, win_start: int, win_end: int) -> bool:
        m = int(minute_of_day_utc)
        if win_start < win_end:
            return (m >= win_start) and (m < win_end)
        return (m >= win_start) or (m < win_end)

    def minutes_to_session_end(minute_of_day_utc: int) -> int:
        m = int(minute_of_day_utc)
        if start_min < end_min:
            return max(0, end_min - m)
        # Cross-midnight session.
        if m >= start_min:
            return (1440 - m) + end_min
        return max(0, end_min - m)

    if start_min < end_min:
        df["in_tokyo_session"] = (df["minute_of_day_utc"] >= start_min) & (df["minute_of_day_utc"] < end_min)
    else:
        df["in_tokyo_session"] = (df["minute_of_day_utc"] >= start_min) | (df["minute_of_day_utc"] < end_min)
    df["allowed_trading_day"] = df["utc_day_name"].isin(allowed_days_set)

    starting_equity = float(cfg.get("starting_equity_usd", 50000.0))
    equity = starting_equity
    risk_pct = float(cfg["position_sizing"]["risk_per_trade_pct"]) / 100.0
    day_risk_multipliers_cfg = cfg.get("position_sizing", {}).get("day_risk_multipliers", {})
    day_risk_multipliers = {str(k): float(v) for k, v in day_risk_multipliers_cfg.items()}
    max_units = int(cfg["position_sizing"]["max_units"])
    max_open = int(cfg["position_sizing"]["max_concurrent_positions"])
    max_trades_session = int(cfg["trade_management"]["max_trades_per_session"])
    min_entry_gap_min = int(cfg["trade_management"]["min_time_between_entries_minutes"])
    no_reentry_stop_min = int(cfg["trade_management"]["no_reentry_same_direction_after_stop_minutes"])
    session_loss_stop_pct = float(cfg["trade_management"].get("session_loss_stop_pct", 0.0)) / 100.0
    stop_after_consecutive_losses = int(cfg["trade_management"].get("stop_after_consecutive_losses", 3))
    breakout_disable_pips = float(cfg["trade_management"]["disable_entries_if_move_from_tokyo_open_range_exceeds_pips"])
    breakout_mode = str(cfg["trade_management"].get("breakout_detection_mode", "rolling")).strip().lower()
    rolling_window_minutes = int(cfg["trade_management"].get("rolling_window_minutes", 60))
    rolling_range_threshold_pips = float(cfg["trade_management"].get("rolling_range_threshold_pips", 40.0))
    breakout_cooldown_minutes = int(cfg["trade_management"].get("cooldown_minutes", 15))
    consec_pause_cfg = cfg["trade_management"].get("consecutive_loss_pause", {})
    consec_pause_enabled = bool(consec_pause_cfg.get("enabled", False))
    consec_pause_losses = int(consec_pause_cfg.get("consecutive_losses", 2))
    consec_pause_minutes = int(consec_pause_cfg.get("pause_minutes", 30))
    atr_max = float(cfg["indicators"]["atr"]["max_threshold_price_units"])
    atr_gate_enabled = bool(cfg["indicators"]["atr"].get("use_as_hard_gate", True))
    atr_pct_cfg = cfg["indicators"]["atr"].get("percentile_filter", {})
    atr_pct_enabled = bool(atr_pct_cfg.get("enabled", False))
    atr_pct_lookback = max(10, int(atr_pct_cfg.get("lookback_bars", 150)))
    atr_pct_max = float(atr_pct_cfg.get("max_percentile", 0.67))
    atr_pct_min_obs = max(20, int(atr_pct_cfg.get("min_observations", min(50, atr_pct_lookback))))
    regime_filter_mode = str(cfg["indicators"].get("bb_width_regime_filter", {}).get("mode", "percentile")).strip().lower()
    scoring_model = str(cfg.get("entry_rules", {}).get("scoring_model", "")).strip().lower()
    tokyo_v2_scoring = scoring_model in {"tokyo_v2", "v2", "tokyo_actual_v2"}
    tol_pips = float(cfg["entry_rules"]["long"].get("price_zone", {}).get("tolerance_pips", 10.0))
    tol = tol_pips * PIP_SIZE
    min_tp_pips = 8.0
    min_sl_pips = float(cfg["exit_rules"]["stop_loss"].get("minimum_sl_pips", 10.0))
    max_sl_pips = float(cfg["exit_rules"]["stop_loss"].get("hard_max_sl_pips", 28.0))
    sl_buf = float(cfg["exit_rules"]["stop_loss"].get("buffer_pips", 8.0)) * PIP_SIZE
    trail_activate_pips = float(cfg["exit_rules"]["trailing_stop"].get("activate_after_profit_pips", 10.0))
    trail_dist_pips = float(cfg["exit_rules"]["trailing_stop"].get("trail_distance_pips", 8.0))
    trail_enabled = bool(cfg["exit_rules"]["trailing_stop"].get("enabled", True))
    trail_requires_tp1 = bool(cfg["exit_rules"]["trailing_stop"].get("requires_tp1_hit", True))
    partial_close_pct = float(cfg["exit_rules"]["take_profit"].get("partial_close_pct", 0.5))
    partial_tp_min_pips = float(cfg["exit_rules"]["take_profit"].get("partial_tp_min_pips", 6.0))
    partial_tp_max_pips = float(cfg["exit_rules"]["take_profit"].get("partial_tp_max_pips", 12.0))
    partial_tp_atr_mult = float(cfg["exit_rules"]["take_profit"].get("partial_tp_atr_mult", 0.5))
    tp_mode = str(cfg["exit_rules"]["take_profit"].get("mode", "partial")).strip().lower()
    single_tp_atr_mult = float(cfg["exit_rules"]["take_profit"].get("single_tp_atr_mult", 1.0))
    single_tp_min_pips = float(cfg["exit_rules"]["take_profit"].get("single_tp_min_pips", min_tp_pips))
    single_tp_max_pips = float(cfg["exit_rules"]["take_profit"].get("single_tp_max_pips", 40.0))
    breakeven_offset_pips = float(cfg["exit_rules"]["take_profit"].get("breakeven_offset_pips", 1.0))
    time_decay_minutes = int(cfg["exit_rules"]["time_exit"].get("time_decay_minutes", 120))
    time_decay_profit_cap_pips = float(cfg["exit_rules"]["time_exit"].get("time_decay_profit_cap_pips", 3.0))
    exec_cfg = cfg.get("execution_model", {})
    spread_mode = str(exec_cfg.get("spread_mode", "fixed"))
    spread_pips = float(exec_cfg.get("spread_pips", 1.5))
    spread_min_pips = float(exec_cfg.get("spread_min_pips", 1.0))
    spread_max_pips = float(exec_cfg.get("spread_max_pips", 3.0))
    max_entry_spread_pips = float(exec_cfg.get("max_entry_spread_pips", 10.0))
    margin_cfg = cfg.get("margin_model", {})
    margin_model_enabled = bool(margin_cfg.get("enabled", False))
    margin_leverage = max(1.0, float(margin_cfg.get("leverage", 50.0)))
    margin_buffer_pct = max(0.0, float(margin_cfg.get("buffer_pct", 0.0)))
    confluence_min_long = int(cfg["entry_rules"]["long"].get("confluence_scoring", {}).get("minimum_score", 2))
    confluence_min_short = int(cfg["entry_rules"]["short"].get("confluence_scoring", {}).get("minimum_score", 2))
    long_rsi_soft_entry = float(cfg["entry_rules"]["long"].get("rsi_soft_filter", {}).get("entry_soft_threshold", 35.0))
    long_rsi_bonus = float(cfg["entry_rules"]["long"].get("rsi_soft_filter", {}).get("bonus_threshold", 30.0))
    short_rsi_soft_entry = float(cfg["entry_rules"]["short"].get("rsi_soft_filter", {}).get("entry_soft_threshold", 65.0))
    short_rsi_bonus = float(cfg["entry_rules"]["short"].get("rsi_soft_filter", {}).get("bonus_threshold", 70.0))
    core_gate_cfg = cfg.get("entry_rules", {}).get("core_gate", {})
    core_gate_required = int(core_gate_cfg.get("required_count", 4))
    core_gate_use_zone = bool(core_gate_cfg.get("use_zone", True))
    core_gate_use_bb = bool(core_gate_cfg.get("use_bb_touch", True))
    core_gate_use_sar = bool(core_gate_cfg.get("use_sar_flip", True))
    core_gate_use_rsi = bool(core_gate_cfg.get("use_rsi_soft", True))
    confirm_cfg = cfg.get("entry_confirmation", {})
    confirmation_enabled = bool(confirm_cfg.get("enabled", True))
    confirmation_type = str(confirm_cfg.get("type", "m1")).strip().lower()
    if confirmation_type not in {"m1", "m5"}:
        confirmation_type = "m1"
    confirmation_window_bars = max(0, int(confirm_cfg.get("window_bars", 5)))
    if not confirmation_enabled:
        confirmation_window_bars = 0
    adx_filter_cfg = cfg.get("adx_filter", {})
    adx_filter_enabled = bool(adx_filter_cfg.get("enabled", False))
    adx_max_for_entry = float(adx_filter_cfg.get("max_adx_for_entry", 30.0))
    adx_day_max_by_day = {str(k): float(v) for k, v in adx_filter_cfg.get("day_max_by_day", {}).items()}
    # Backward/forward-compatible alias for per-day ADX caps.
    adx_day_overrides = {str(k): float(v) for k, v in adx_filter_cfg.get("day_overrides", {}).items()}
    adx_day_max_by_day.update(adx_day_overrides)

    combo_filter_cfg = cfg.get("confluence_combo_filter", {})
    combo_filter_enabled = bool(combo_filter_cfg.get("enabled", False))
    combo_filter_mode = str(combo_filter_cfg.get("mode", "allowlist")).strip().lower()
    combo_allow = set(combo_filter_cfg.get("allowed_combos", []))
    combo_block = set(combo_filter_cfg.get("blocked_combos", []))

    daily_range_cfg = cfg.get("daily_range_filter", {})
    daily_range_enabled = bool(daily_range_cfg.get("enabled", False))
    daily_range_min_pips = float(daily_range_cfg.get("min_prior_day_range_pips", 15.0))
    daily_range_max_pips = float(daily_range_cfg.get("max_prior_day_range_pips", 80.0))

    entry_imp_cfg = cfg.get("entry_improvement", {})
    entry_imp_enabled = bool(entry_imp_cfg.get("enabled", False))
    entry_imp_long_offset = float(entry_imp_cfg.get("long_offset_from_confirm_low_pips", 1.0))
    entry_imp_short_offset = float(entry_imp_cfg.get("short_offset_from_confirm_high_pips", 1.0))
    entry_imp_fill_window = int(entry_imp_cfg.get("fill_window_candles", 3))
    chase_cfg = cfg.get("entry_chase_filter", {})
    chase_enabled = bool(chase_cfg.get("enabled", False))
    chase_max_pips = float(chase_cfg.get("max_chase_pips", 2.0))
    ss_cfg = cfg.get("signal_strength_tracking", {})
    ss_enabled = bool(ss_cfg.get("enabled", False))
    ss_comp = ss_cfg.get("components", {})
    ss_filter_cfg = cfg.get("signal_strength_filter", {})
    ss_filter_enabled = bool(ss_filter_cfg.get("enabled", False))
    ss_filter_min_score = int(ss_filter_cfg.get("min_score", 0))
    ss_sizing_cfg = cfg.get("signal_strength_sizing", {})
    ss_sizing_enabled = bool(ss_sizing_cfg.get("enabled", False))
    ss_sizing_weak_mult = float(ss_sizing_cfg.get("weak_mult", 1.0))
    ss_sizing_moderate_mult = float(ss_sizing_cfg.get("moderate_mult", 1.0))
    ss_sizing_strong_mult = float(ss_sizing_cfg.get("strong_mult", 1.0))

    regime_switch_cfg = cfg.get("regime_switch", {})
    regime_switch_enabled = bool(regime_switch_cfg.get("enabled", False))
    regime_switch_metric = str(regime_switch_cfg.get("metric", "avg_atr_m15"))
    regime_metric_used = regime_switch_metric
    regime_switch_threshold = float(regime_switch_cfg.get("threshold", 0.0))
    regime_switch_lookback_days = int(regime_switch_cfg.get("lookback_days", 5))

    # V10 additions (light-touch overlays on V7 baseline).
    rejection_bonus_cfg = cfg.get("rejection_bonus", {})
    rejection_bonus_enabled = bool(rejection_bonus_cfg.get("enabled", False))
    rejection_sl_improvement = bool(rejection_bonus_cfg.get("sl_improvement", True))
    rejection_sl_buffer_pips = float(rejection_bonus_cfg.get("sl_buffer_pips", 2.0))
    rejection_lookback_m5 = int(max(1, rejection_bonus_cfg.get("lookback_m5_bars", 2)))

    session_env_cfg = cfg.get("session_envelope", {})
    session_env_enabled = bool(session_env_cfg.get("enabled", False))
    session_env_warmup_minutes = int(session_env_cfg.get("warmup_minutes", 30))
    session_env_use_for_tp = bool(session_env_cfg.get("use_for_tp", True))
    session_env_tp_mode = str(session_env_cfg.get("tp_mode", "nearest_of_pivot_or_midpoint")).strip().lower()
    session_env_log_ir_pos = bool(session_env_cfg.get("log_ir_position", True))

    div_track_cfg = cfg.get("rsi_divergence_tracking", {})
    div_track_enabled = bool(div_track_cfg.get("enabled", False))

    trend_skip_cfg = cfg.get("trend_regime_skip", {})
    trend_skip_enabled = bool(trend_skip_cfg.get("enabled", False))
    trend_skip_lookback_days = int(trend_skip_cfg.get("lookback_days", 3))
    trend_skip_max_move_pips = float(trend_skip_cfg.get("max_directional_move_pips", 80.0))
    trend_skip_selection_method = str(trend_skip_cfg.get("selection_method", "configured"))
    news_cfg = cfg.get("news_filter", {})
    news_enabled = bool(news_cfg.get("enabled", False))
    news_mode = str(news_cfg.get("mode", "tiered")).strip().lower()
    news_all_pre = int(news_cfg.get("all_pre_block_minutes", 15))
    news_all_post = int(news_cfg.get("all_post_block_minutes", 15))
    news_high_pre = int(news_cfg.get("high_impact_pre_block_minutes", 60))
    news_high_post = int(news_cfg.get("high_impact_post_block_minutes", 60))
    news_med_pre = int(news_cfg.get("medium_impact_pre_block_minutes", 15))
    news_med_post = int(news_cfg.get("medium_impact_post_block_minutes", 15))
    news_low_pre = int(news_cfg.get("low_impact_pre_block_minutes", news_med_pre))
    news_low_post = int(news_cfg.get("low_impact_post_block_minutes", news_med_post))
    news_block_entire_session = bool(news_cfg.get("block_entire_session", False))
    news_session_proximity_hours = float(news_cfg.get("session_proximity_hours", 4.0))
    news_block_impacts = {str(x).lower() for x in news_cfg.get("block_impacts", ["high", "medium", "low"])}
    if not news_block_impacts:
        news_block_impacts = {"high", "medium", "low"}

    momentum_cfg = cfg.get("momentum_check", {})
    momentum_enabled = bool(momentum_cfg.get("enabled", False))
    momentum_lookback = int(momentum_cfg.get("lookback_candles", 10))
    momentum_slope_th = float(momentum_cfg.get("slope_threshold_pips_per_candle", 0.3))
    momentum_delay_candles = int(momentum_cfg.get("delay_candles", 5))
    momentum_max_delays = int(momentum_cfg.get("max_delays", 1))

    early_exit_cfg = cfg.get("early_exit", {})
    early_exit_enabled = bool(early_exit_cfg.get("enabled", False))
    early_exit_time_min = int(early_exit_cfg.get("time_threshold_minutes", 30))
    early_exit_loss_pips = float(early_exit_cfg.get("loss_threshold_pips", 5.0))
    early_exit_max_profit_seen = float(early_exit_cfg.get("max_profit_seen_pips", 2.0))
    late_session_cfg = cfg.get("late_session_management", {})
    late_session_enabled = bool(late_session_cfg.get("enabled", False))
    late_session_minutes_before_end = int(late_session_cfg.get("minutes_before_end", 45))
    late_session_close_if_no_tp1_and_pips_below = float(late_session_cfg.get("close_if_no_tp1_and_pips_below", -2.0))
    late_session_tp1_hit_tighten_trail_pips = float(late_session_cfg.get("tp1_hit_tighten_trail_pips", 3.0))
    late_session_hard_close_all_minutes_before_end = int(late_session_cfg.get("hard_close_all_minutes_before_end", 0))
    late_session_be_or_close_minutes_before_end = int(late_session_cfg.get("be_or_close_minutes_before_end", 0))
    late_session_be_min_profit_pips = float(late_session_cfg.get("be_min_profit_pips", 1.0))
    late_session_be_offset_pips = float(late_session_cfg.get("be_offset_pips", 0.0))
    late_session_profit_tighten_minutes_before_end = int(late_session_cfg.get("profit_tighten_minutes_before_end", 0))
    late_session_profit_tighten_trail_mult = float(late_session_cfg.get("profit_tighten_trail_mult", 0.5))

    regime_gate = cfg.get("regime_gate", {})
    regime_enabled = bool(regime_gate.get("enabled", False))
    atr_ratio_trend = float(regime_gate.get("atr_ratio_trending_threshold", 1.3))
    atr_ratio_calm = float(regime_gate.get("atr_ratio_calm_threshold", 0.8))
    adx_trend = float(regime_gate.get("adx_trending_threshold", 25.0))
    adx_range = float(regime_gate.get("adx_ranging_threshold", 20.0))
    favorable_min_score = int(regime_gate.get("favorable_min_score", 1))
    neutral_min_score = int(regime_gate.get("neutral_min_score", 0))
    neutral_size_mult = float(regime_gate.get("neutral_size_multiplier", 0.5))

    # Optional adaptive ATR percentile gate (default OFF). Compute once for speed.
    if atr_pct_enabled:
        atr_vals = pd.to_numeric(df["atr_m15"], errors="coerce").to_numpy(dtype=float)
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
        df["atr_m15_percentile_rank"] = atr_pct_rank
    else:
        df["atr_m15_percentile_rank"] = np.nan

    cq = cfg.get("confluence_quality", {})
    cq_enabled = bool(cq.get("enabled", False))
    top_combos = set(cq.get("top_combos", []))
    bottom_combos = set(cq.get("bottom_combos", []))
    high_quality_mult = float(cq.get("high_quality_size_mult", 1.0))
    medium_quality_mult = float(cq.get("medium_quality_size_mult", 0.75))
    low_quality_skip = bool(cq.get("low_quality_skip", True))

    hp = cfg.get("hour_preference", {})
    hp_enabled = bool(hp.get("enabled", False))
    hp_mults = {str(k): float(v) for k, v in hp.get("multipliers", {}).items()}

    dd_cfg = cfg.get("drawdown_adaptive_sizing", {})
    dd_enabled = bool(dd_cfg.get("enabled", False))
    dd_t1 = float(dd_cfg.get("tier_1_dd_pct", 2.0))
    dd_t1_red = float(dd_cfg.get("tier_1_size_reduction", 0.25))
    dd_t2 = float(dd_cfg.get("tier_2_dd_pct", 4.0))
    dd_t2_red = float(dd_cfg.get("tier_2_size_reduction", 0.50))
    dd_t3 = float(dd_cfg.get("tier_3_dd_pct", 6.0))
    dd_resume = float(dd_cfg.get("resume_dd_pct", 4.0))

    trade_id = 0
    open_positions: list[Position] = []
    closed_rows: list[dict] = []
    equity_curve: list[dict] = []
    diag = Counter()

    session_state: dict[str, dict] = {}
    session_days = sorted(
        {
            d
            for d, ins, okd in zip(df["session_day_jst"], df["in_tokyo_session"], df["allowed_trading_day"])
            if bool(ins) and bool(okd)
        }
    )
    df["news_blocked"] = False
    if news_enabled:
        ev = load_news_events(cfg)
        if not ev.empty:
            block_mask = np.zeros(len(df), dtype=bool)
            tcol = df["time"]
            for _, er in ev.iterrows():
                impact = str(er.get("impact", "")).lower()
                if impact not in news_block_impacts:
                    continue
                if news_mode == "all":
                    pre_min = news_all_pre
                    post_min = news_all_post
                else:
                    if impact == "high":
                        pre_min, post_min = news_high_pre, news_high_post
                    elif impact == "medium":
                        pre_min, post_min = news_med_pre, news_med_post
                    else:
                        pre_min, post_min = news_low_pre, news_low_post
                evt = pd.Timestamp(er["event_ts"])
                if pd.isna(evt):
                    continue
                ws = evt - pd.Timedelta(minutes=max(0, int(pre_min)))
                we = evt + pd.Timedelta(minutes=max(0, int(post_min)))
                block_mask |= ((tcol >= ws) & (tcol <= we)).to_numpy()
            if news_block_entire_session:
                session_rows = df["in_tokyo_session"].to_numpy()
                for day in sorted(df.loc[df["in_tokyo_session"], "session_day_jst"].unique()):
                    day_mask = (df["session_day_jst"] == day).to_numpy() & session_rows
                    if not day_mask.any():
                        continue
                    day_times = tcol[day_mask]
                    day_start = day_times.min() - pd.Timedelta(hours=max(0.0, news_session_proximity_hours))
                    day_end = day_times.max() + pd.Timedelta(hours=max(0.0, news_session_proximity_hours))
                    if ((ev["event_ts"] >= day_start) & (ev["event_ts"] <= day_end)).any():
                        block_mask |= day_mask
            df["news_blocked"] = block_mask

    last_trade_was_win = False
    consec_wins = 0
    consec_losses = 0
    max_consec_wins = 0
    max_consec_losses_seen = 0
    peak_equity = equity
    trading_paused_dd = False

    regime_gate_stats = {
        "bars_favorable": 0,
        "bars_neutral": 0,
        "bars_unfavorable": 0,
        "trades_at_full_size": 0,
        "trades_at_reduced_size": 0,
        "trades_blocked_by_regime": 0,
        "pnl_favorable_trades": 0.0,
        "pnl_neutral_trades": 0.0,
    }
    signal_quality_stats = {
        "high_quality_trades": 0,
        "medium_quality_trades": 0,
        "low_quality_skipped": 0,
        "pnl_high_quality": 0.0,
        "pnl_medium_quality": 0.0,
    }
    drawdown_sizing_stats = {
        "trades_at_full_size": 0,
        "trades_at_tier1_reduction": 0,
        "trades_at_tier2_reduction": 0,
        "times_trading_paused": 0,
        "total_bars_paused": 0,
    }
    adx_filter_stats = {
        "trades_allowed": 0,
        "trades_blocked_by_adx": 0,
        "adx_allowed_values": [],
        "adx_blocked_values": [],
    }
    combo_filter_stats = {
        "mode_used": combo_filter_mode,
        "trades_allowed": 0,
        "trades_blocked_by_combo": 0,
        "combo_distribution": {},
    }
    daily_range_filter_stats = {
        "sessions_allowed": 0,
        "sessions_blocked_high_range": 0,
        "sessions_blocked_low_range": 0,
    }
    entry_improvement_stats = {
        "signals_with_limit_placed": 0,
        "limits_filled": 0,
        "limits_expired": 0,
        "improvement_values_pips": [],
    }
    momentum_check_stats = {
        "entries_immediate": 0,
        "entries_delayed_then_filled": 0,
        "entries_delayed_then_expired": 0,
        "immediate_usd": [],
        "delayed_usd": [],
    }
    early_exit_stats = {
        "early_exits_triggered": 0,
        "loss_saved_pips": [],
    }
    ss_filter_stats = {
        "min_score_used": int(ss_filter_min_score),
        "trades_above_threshold": 0,
        "trades_below_threshold_skipped": 0,
    }
    regime_switch_stats = {
        "metric_used": regime_metric_used,
        "threshold_used": regime_switch_threshold,
        "sessions_traded": 0,
        "sessions_skipped": 0,
        "pnl_traded_sessions": 0.0,
        "estimated_pnl_skipped_sessions": 0.0,
    }
    trend_regime_stats = {
        "sessions_traded": 0,
        "sessions_skipped_by_trend": 0,
        "estimated_pnl_skipped": 0.0,
    }

    # Prior day range (UTC day) used as simple session pre-check.
    day_rng = (
        df.assign(utc_date=df["time_utc"].dt.date)
        .groupby("utc_date")
        .agg(day_high=("high", "max"), day_low=("low", "min"))
        .reset_index()
    )
    day_rng["range_pips"] = (day_rng["day_high"] - day_rng["day_low"]) / PIP_SIZE
    day_rng["prior_day_range_pips"] = day_rng["range_pips"].shift(1)
    prior_range_map = dict(zip(day_rng["utc_date"], day_rng["prior_day_range_pips"]))
    df["prior_day_range_pips"] = df["time_utc"].dt.date.map(prior_range_map)

    # Precompute daily regime metric series for optional regime switch.
    daily_market = (
        df.assign(utc_date=df["time_utc"].dt.date)
        .groupby("utc_date", as_index=False)
        .agg(
            avg_atr_m15=("atr_m15", "mean"),
            avg_adx_m15=("adx_m15", "mean"),
            avg_bb_width_m5=("bb_width", "mean"),
            avg_daily_range_pips=("high", lambda s: np.nan),  # placeholder
            day_high=("high", "max"),
            day_low=("low", "min"),
        )
    )
    daily_market["avg_daily_range_pips"] = (daily_market["day_high"] - daily_market["day_low"]) / PIP_SIZE
    daily_market = daily_market.sort_values("utc_date").reset_index(drop=True)
    regime_metric_used = regime_switch_metric
    if regime_metric_used not in daily_market.columns:
        regime_metric_used = "avg_atr_m15"
    daily_market["regime_metric_roll"] = (
        daily_market[regime_metric_used]
        .rolling(regime_switch_lookback_days, min_periods=1)
        .mean()
    )
    regime_metric_map = dict(zip(daily_market["utc_date"], daily_market["regime_metric_roll"]))

    # V10: simple 3-day directional move map for trend regime skip.
    daily_close = (
        df.assign(utc_date=df["time_utc"].dt.date)
        .groupby("utc_date", as_index=False)
        .agg(day_close=("close", "last"))
        .sort_values("utc_date")
        .reset_index(drop=True)
    )
    daily_close["directional_move_pips"] = (
        (daily_close["day_close"] - daily_close["day_close"].shift(trend_skip_lookback_days)).abs() / PIP_SIZE
    )
    trend_move_map = dict(zip(daily_close["utc_date"], daily_close["directional_move_pips"]))

    def ensure_session(sday: str, row: pd.Series) -> dict:
        if sday not in session_state:
            session_state[sday] = {
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
                "warmup_end_ts": pd.Timestamp(row["time"]) + pd.Timedelta(minutes=session_env_warmup_minutes),
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
            if daily_range_enabled:
                pdr = row.get("prior_day_range_pips", np.nan)
                if pd.isna(pdr):
                    session_state[sday]["daily_range_allowed"] = False
                    session_state[sday]["daily_range_block_reason"] = "missing"
                elif float(pdr) < daily_range_min_pips:
                    session_state[sday]["daily_range_allowed"] = False
                    session_state[sday]["daily_range_block_reason"] = "low"
                    daily_range_filter_stats["sessions_blocked_low_range"] += 1
                elif float(pdr) > daily_range_max_pips:
                    session_state[sday]["daily_range_allowed"] = False
                    session_state[sday]["daily_range_block_reason"] = "high"
                    daily_range_filter_stats["sessions_blocked_high_range"] += 1
                else:
                    daily_range_filter_stats["sessions_allowed"] += 1
            if regime_switch_enabled:
                cur_date = pd.Timestamp(row["time_utc"]).date()
                metric_val = regime_metric_map.get(cur_date, np.nan)
                session_state[sday]["regime_metric_value"] = float(metric_val) if not pd.isna(metric_val) else np.nan
                if pd.isna(metric_val):
                    session_state[sday]["regime_switch_allowed"] = True
                else:
                    ok = float(metric_val) <= float(regime_switch_threshold)
                    session_state[sday]["regime_switch_allowed"] = bool(ok)
                    if ok:
                        regime_switch_stats["sessions_traded"] += 1
                    else:
                        regime_switch_stats["sessions_skipped"] += 1
            if trend_skip_enabled:
                cur_date = pd.Timestamp(row["time_utc"]).date()
                mv = trend_move_map.get(cur_date, np.nan)
                session_state[sday]["trend_move_pips"] = float(mv) if not pd.isna(mv) else np.nan
                if pd.isna(mv):
                    session_state[sday]["trend_skip_allowed"] = True
                    trend_regime_stats["sessions_traded"] += 1
                elif float(mv) > trend_skip_max_move_pips:
                    session_state[sday]["trend_skip_allowed"] = False
                    trend_regime_stats["sessions_skipped_by_trend"] += 1
                else:
                    session_state[sday]["trend_skip_allowed"] = True
                    trend_regime_stats["sessions_traded"] += 1
        return session_state[sday]

    def close_position(pos: Position, ts: pd.Timestamp, exit_price: float, reason: str) -> None:
        nonlocal equity, consec_wins, consec_losses, max_consec_wins, max_consec_losses_seen, last_trade_was_win, peak_equity
        if pos.units_remaining > 0:
            pips, usd = calc_leg_usd_pips(pos.direction, pos.entry_price, exit_price, pos.units_remaining)
            pos.realized_pip_units += pips * pos.units_remaining
            pos.realized_usd += usd
            pos.units_remaining = 0
            pos.exit_price = float(exit_price)
        pos.exit_reason = reason
        pos.exit_time = ts

        total_pips = pos.realized_pip_units / max(1, pos.units_initial)
        equity_before = equity
        equity += pos.realized_usd
        peak_equity = max(peak_equity, equity)
        win = pos.realized_usd > 0
        if win:
            consec_wins = consec_wins + 1 if last_trade_was_win else 1
            consec_losses = 0
            last_trade_was_win = True
            max_consec_wins = max(max_consec_wins, consec_wins)
        else:
            consec_losses = consec_losses + 1 if not last_trade_was_win else 1
            consec_wins = 0
            last_trade_was_win = False
            max_consec_losses_seen = max(max_consec_losses_seen, consec_losses)

        sst = session_state.get(pos.entry_session_day, {})
        if pos.regime_label == "favorable":
            regime_gate_stats["pnl_favorable_trades"] += float(pos.realized_usd)
        elif pos.regime_label == "neutral":
            regime_gate_stats["pnl_neutral_trades"] += float(pos.realized_usd)
        if pos.quality_label == "high":
            signal_quality_stats["pnl_high_quality"] += float(pos.realized_usd)
        elif pos.quality_label == "medium":
            signal_quality_stats["pnl_medium_quality"] += float(pos.realized_usd)
        sst["session_pnl_usd"] = float(sst.get("session_pnl_usd", 0.0)) + float(pos.realized_usd)
        sess_start_eq = float(sst.get("session_start_equity", equity_before))
        if session_loss_stop_pct > 0 and sess_start_eq > 0 and float(sst["session_pnl_usd"]) <= (-session_loss_stop_pct * sess_start_eq):
            sst["stopped"] = True
        if win:
            sst["consec_losses"] = 0
            sst["wins"] = int(sst.get("wins", 0)) + 1
        else:
            sst["consec_losses"] = int(sst.get("consec_losses", 0)) + 1
            sst["losses"] = int(sst.get("losses", 0)) + 1
            if stop_after_consecutive_losses > 0 and int(sst["consec_losses"]) >= stop_after_consecutive_losses:
                sst["stopped"] = True
            if consec_pause_enabled and consec_pause_losses > 0 and int(sst["consec_losses"]) >= consec_pause_losses:
                sst["loss_pause_until"] = ts + pd.Timedelta(minutes=consec_pause_minutes)
                sst["consec_losses"] = 0
            if "sl" in reason:
                if pos.direction == "long":
                    sst["last_stop_time_long"] = ts
                else:
                    sst["last_stop_time_short"] = ts

        closed_rows.append(
            {
                "trade_id": pos.trade_id,
                "entry_datetime": str(pos.entry_time),
                "signal_datetime": str(pos.signal_time) if pos.signal_time is not None else str(pos.entry_time),
                "confirmation_delay_candles": int(pos.confirmation_delay_candles),
                "exit_datetime": str(pos.exit_time),
                "direction": pos.direction,
                "entry_price": pos.entry_price,
                "exit_price": pos.exit_price,
                "sl_price": pos.sl_price,
                "tp_price": pos.tp2_price,
                "exit_reason": pos.exit_reason,
                "pips": total_pips,
                "usd": pos.realized_usd,
                "pnl_usd": pos.realized_usd,
                "confluence_score": pos.confluence_score,
                "position_size_units": pos.units_initial,
                "position_units": pos.units_initial,
                "partial_tp_pips": float(pos.partial_tp_pips),
                "initial_sl_pips": float(pos.initial_sl_pips),
                "regime_label": pos.regime_label,
                "regime": pos.regime_label,
                "confluence_combo": pos.confluence_combo,
                "quality_label": pos.quality_label,
                "size_mult_total": float(pos.size_mult_total),
                "size_mult_regime": float(pos.size_mult_regime),
                "size_mult_quality": float(pos.size_mult_quality),
                "size_mult_hour": float(pos.size_mult_hour),
                "size_mult_dd": float(pos.size_mult_dd),
                "dd_tier": pos.dd_tier,
                "signal_strength_score": int(pos.signal_strength_score),
                "signal_strength_tier": str(pos.signal_strength_tier),
                "entry_delay_type": str(pos.entry_delay_type),
                "mfe_pips": float(pos.max_profit_seen_pips),
                "mae_pips": float(pos.max_adverse_seen_pips),
                "rejection_confirmed": bool(pos.rejection_confirmed),
                "divergence_present": bool(pos.divergence_present),
                "inside_ir": bool(pos.inside_ir),
                "quality_markers": str(pos.quality_markers),
                "sl_source": str(pos.sl_source),
                "tp_source": str(pos.tp_source),
                "distance_to_ir_boundary_pips": float(pos.distance_to_ir_boundary_pips),
                "distance_to_midpoint_pips": float(pos.distance_to_midpoint_pips),
                "distance_to_pivot_pips": float(pos.distance_to_pivot_pips),
                "entry_session_day": pos.entry_session_day,
                "duration_minutes": (pos.exit_time - pos.entry_time).total_seconds() / 60.0 if pos.exit_time is not None else np.nan,
                "equity_before": equity_before,
                "equity_after": equity,
                "entry_regime": "TOKYO_MEANREV",
                "entry_signal_mode": "tokyo_meanrev",
                "entry_session": "tokyo",
                "pivot_P": pos.entry_indicators.get("pivot_P"),
                "pivot_R1": pos.entry_indicators.get("pivot_R1"),
                "pivot_R2": pos.entry_indicators.get("pivot_R2"),
                "pivot_S1": pos.entry_indicators.get("pivot_S1"),
                "pivot_S2": pos.entry_indicators.get("pivot_S2"),
                "bb_upper": pos.entry_indicators.get("bb_upper"),
                "bb_mid": pos.entry_indicators.get("bb_mid"),
                "bb_middle": pos.entry_indicators.get("bb_mid"),
                "bb_lower": pos.entry_indicators.get("bb_lower"),
                "sar_value": pos.entry_indicators.get("sar_value"),
                "sar_direction": pos.entry_indicators.get("sar_direction"),
                "rsi_m5": pos.entry_indicators.get("rsi_m5"),
                "rsi_value": pos.entry_indicators.get("rsi_m5"),
                "atr_m15": pos.entry_indicators.get("atr_m15"),
                "atr_value": pos.entry_indicators.get("atr_m15"),
                "hour_utc": pd.Timestamp(pos.entry_time).tz_convert("UTC").hour if pd.Timestamp(pos.entry_time).tzinfo is not None else pd.Timestamp(pos.entry_time).hour,
                "day_of_week": pd.Timestamp(pos.entry_time).tz_convert("UTC").day_name() if pd.Timestamp(pos.entry_time).tzinfo is not None else pd.Timestamp(pos.entry_time).day_name(),
            }
        )
        equity_curve.append({"trade_number": len(closed_rows), "time": str(ts), "equity": equity})

    # Verification checklist before trade loop.
    piv_cols = ["pivot_P", "pivot_R1", "pivot_R2", "pivot_R3", "pivot_S1", "pivot_S2", "pivot_S3"]
    ind_cols = piv_cols + ["bb_upper", "bb_lower", "bb_mid", "rsi_m5", "atr_m15", "sar_value", "sar_direction"]
    valid_ind_mask = (~df[ind_cols].isna().any(axis=1))
    in_session_mask = (df["in_tokyo_session"]) & (df["allowed_trading_day"])
    first_valid_idx = int(valid_ind_mask.idxmax() + 1) if valid_ind_mask.any() else -1
    total_in_session = int(in_session_mask.sum())
    in_session_valid = int((in_session_mask & valid_ind_mask).sum())
    in_session_valid_pct = (100.0 * in_session_valid / total_in_session) if total_in_session else 0.0

    def preview_breakout_pass_count() -> int:
        if breakout_mode == "disabled":
            return total_in_session
        pass_count = 0
        preview_state: dict[str, dict] = {}
        for _, row in df.loc[in_session_mask].iterrows():
            sday = str(row["session_day_jst"])
            ts = pd.Timestamp(row["time"])
            if sday not in preview_state:
                preview_state[sday] = {
                    "session_open_price": float(row["open"]),
                    "rolling_window": [],
                    "cooldown_until": None,
                }
            st = preview_state[sday]
            blocked = False
            if breakout_mode == "from_open":
                up = (float(row["high"]) - float(st["session_open_price"])) / PIP_SIZE
                dn = (float(st["session_open_price"]) - float(row["low"])) / PIP_SIZE
                moved_from_open_pips = max(up, dn)
                blocked = moved_from_open_pips > breakout_disable_pips
            elif breakout_mode == "rolling":
                rw = st["rolling_window"]
                rw.append((ts, float(row["high"]), float(row["low"])))
                cutoff = ts - pd.Timedelta(minutes=rolling_window_minutes)
                while rw and rw[0][0] < cutoff:
                    rw.pop(0)
                if st["cooldown_until"] is not None and ts < pd.Timestamp(st["cooldown_until"]):
                    blocked = True
                else:
                    highs = [x[1] for x in rw]
                    lows = [x[2] for x in rw]
                    range_pips = (max(highs) - min(lows)) / PIP_SIZE if highs and lows else 0.0
                    if range_pips > rolling_range_threshold_pips:
                        st["cooldown_until"] = ts + pd.Timedelta(minutes=breakout_cooldown_minutes)
                        blocked = True
                    else:
                        st["cooldown_until"] = None
            if not blocked:
                pass_count += 1
        return pass_count

    breakout_pass_preview = preview_breakout_pass_count()
    breakout_pass_preview_pct = (100.0 * breakout_pass_preview / total_in_session) if total_in_session else 0.0

    first_valid_in_session = df.loc[in_session_mask & valid_ind_mask].head(1)
    pivot_samples = (
        df.loc[in_session_mask & valid_ind_mask]
        .drop_duplicates(subset=["session_day_jst"])
        .head(3)[["session_day_jst", "pivot_P", "pivot_S1", "pivot_S2", "pivot_R1", "pivot_R2"]]
        .to_dict(orient="records")
    )
    print(f"First valid indicator bar: candle #{first_valid_idx}")
    print(
        "Session window UTC mapping: "
        f"start={session_start_utc} end={session_end_utc} "
        f"cross_midnight={'yes' if start_min >= end_min else 'no'}"
    )
    excluded_mon = int(((df["utc_day_name"] == "Monday") & (df["in_tokyo_session"])).sum())
    excluded_sun = int(((df["utc_day_name"] == "Sunday") & (df["in_tokyo_session"])).sum())
    print(
        "Monday/Sunday exclusion check (in-session candles excluded from trading): "
        f"Monday={excluded_mon} Sunday={excluded_sun}"
    )
    shm = cfg.get("second_half_management", {})
    if shm:
        print(
            "Second-half management option selected: "
            f"{shm.get('selected_option','n/a')} — {shm.get('reason','')}"
        )
    print(f"Total in-session bars: {total_in_session}")
    print(f"In-session bars with valid indicators: {in_session_valid} ({in_session_valid_pct:.2f}%)")
    print(
        f"In-session bars NOT blocked by breakout gate: {breakout_pass_preview} "
        f"({breakout_pass_preview_pct:.2f}%)"
    )
    print("Sample pivot levels for first 3 sessions:")
    for p in pivot_samples:
        print(
            f"  {p['session_day_jst']} P={p['pivot_P']:.5f} S1={p['pivot_S1']:.5f} "
            f"S2={p['pivot_S2']:.5f} R1={p['pivot_R1']:.5f} R2={p['pivot_R2']:.5f}"
        )
    if not first_valid_in_session.empty:
        v = first_valid_in_session.iloc[0]
        print(
            "Sample BB values for first in-session bar with valid indicators: "
            f"upper={float(v['bb_upper']):.5f} mid={float(v['bb_mid']):.5f} lower={float(v['bb_lower']):.5f}"
        )
        print(
            "Sample ATR value for first in-session bar with valid indicators: "
            f"{float(v['atr_m15']):.5f}"
        )

    pending_signals: list[dict] = []
    pending_limit_orders: list[dict] = []
    entry_confirmation_stats = {
        "signals_generated": 0,
        "signals_confirmed": 0,
        "signals_expired": 0,
        "confirmation_delays": [],
    }
    near_miss_candidates: list[dict] = []
    condition_hit_counts = Counter()
    condition_check_counts = Counter()

    def used_margin_usd() -> float:
        if not margin_model_enabled:
            return 0.0
        # USDJPY notional in USD is units; required margin ~= units/leverage.
        return float(sum(max(0.0, float(p.units_remaining)) / margin_leverage for p in open_positions))

    def margin_gate_allows(new_units: int) -> bool:
        if not margin_model_enabled:
            return True
        required_margin = max(0.0, float(new_units)) / margin_leverage
        buffer_usd = equity * (margin_buffer_pct / 100.0)
        free_margin = equity - used_margin_usd() - buffer_usd
        return free_margin >= required_margin

    def try_open_position(
        sig: dict,
        sdir: str,
        entry_price: float,
        ts: pd.Timestamp,
        i: int,
        row: pd.Series,
        sst: dict,
        spread_pips_now: float,
    ) -> bool:
        nonlocal trade_id
        if bool(sst.get("stopped", False)):
            diag["blocked_session_stopped"] += 1
            return False
        minute_now = int(row.get("minute_of_day_utc", 0))
        entry_window_ok = in_window(minute_now, entry_start_min, entry_end_min)
        if block_new_entries_minutes_before_end > 0 and in_window(minute_now, start_min, end_min):
            mins_to_end = minutes_to_session_end(minute_now)
            if mins_to_end <= block_new_entries_minutes_before_end:
                entry_window_ok = False
        if not entry_window_ok:
            diag["blocked_entry_window"] += 1
            return False
        if consec_pause_enabled and sst.get("loss_pause_until") is not None and ts < pd.Timestamp(sst["loss_pause_until"]):
            diag["blocked_consecutive_loss_pause"] += 1
            return False
        if stop_after_consecutive_losses > 0 and int(sst.get("consec_losses", 0)) >= stop_after_consecutive_losses:
            diag["blocked_consecutive_loss_stop"] += 1
            sst["stopped"] = True
            return False
        if session_loss_stop_pct > 0 and float(sst.get("session_pnl_usd", 0.0)) <= (-session_loss_stop_pct * float(sst.get("session_start_equity", equity))):
            diag["blocked_session_pnl_stop"] += 1
            sst["stopped"] = True
            return False
        if max_trades_session > 0 and int(sst["trades"]) >= max_trades_session:
            diag["blocked_max_trades_per_session"] += 1
            return False
        if len(open_positions) >= max_open:
            diag["blocked_max_open_cap"] += 1
            return False
        if float(spread_pips_now) > max_entry_spread_pips:
            diag["blocked_max_entry_spread"] += 1
            return False
        if sst["last_entry_time"] is not None and (ts - pd.Timestamp(sst["last_entry_time"])).total_seconds() < min_entry_gap_min * 60.0:
            diag["blocked_min_entry_gap"] += 1
            return False
        last_stop_key = "last_stop_time_long" if sdir == "long" else "last_stop_time_short"
        if sst.get(last_stop_key) is not None and (ts - pd.Timestamp(sst[last_stop_key])).total_seconds() < no_reentry_stop_min * 60.0:
            diag["blocked_reentry_after_stop"] += 1
            return False
        if adx_filter_enabled:
            adx_now = float(row.get("adx_m15", np.nan))
            if pd.isna(adx_now):
                diag["blocked_missing_adx"] += 1
                return False
            adx_cap_here = float(adx_day_max_by_day.get(str(row.get("utc_day_name", "")), adx_max_for_entry))
            if adx_now > adx_cap_here:
                adx_filter_stats["trades_blocked_by_adx"] += 1
                adx_filter_stats["adx_blocked_values"].append(float(adx_now))
                diag["blocked_adx_filter"] += 1
                return False
            adx_filter_stats["trades_allowed"] += 1
            adx_filter_stats["adx_allowed_values"].append(float(adx_now))
        if chase_enabled:
            bid_now, ask_now = get_bid_ask(float(row["close"]), float(spread_pips_now))
            conf_close = float(sig.get("confirmation_close", float(row["close"])))
            if sdir == "long" and ask_now > (conf_close + chase_max_pips * PIP_SIZE):
                diag["blocked_chase_filter"] += 1
                return False
            if sdir == "short" and bid_now < (conf_close - chase_max_pips * PIP_SIZE):
                diag["blocked_chase_filter"] += 1
                return False

        sl_source = "pivot"
        tp_source = "pivot"
        if sdir == "long":
            from_zone = bool(sig["from_zone"])
            sl_raw = (float(sig["S3"]) - sl_buf) if from_zone else (float(sig["S2"]) - sl_buf)
            if rejection_bonus_enabled and rejection_sl_improvement and bool(sig.get("rejection_confirmed", False)):
                rej_low = float(sig.get("rejection_low", np.nan))
                if np.isfinite(rej_low):
                    rej_sl_raw = rej_low - rejection_sl_buffer_pips * PIP_SIZE
                    base_sl_pips = (entry_price - sl_raw) / PIP_SIZE
                    rej_sl_pips = (entry_price - rej_sl_raw) / PIP_SIZE
                    if rej_sl_pips > 0 and rej_sl_pips < base_sl_pips:
                        sl_raw = rej_sl_raw
                        sl_source = "rejection_bonus"
            sl_pips = (entry_price - sl_raw) / PIP_SIZE
            sl_pips = max(min_sl_pips, min(max_sl_pips, sl_pips))
            sl_price = entry_price - sl_pips * PIP_SIZE

            tp_pivot = float(sig["P"])
            tp_mid = float(sig.get("session_midpoint", np.nan))
            if session_env_enabled and session_env_use_for_tp and session_env_tp_mode == "nearest_of_pivot_or_midpoint" and np.isfinite(tp_mid):
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
            if tp_mode == "trail_only":
                tp1_pips = 9999.0
                tp1 = entry_price + tp1_pips * PIP_SIZE
                tp2 = float(tp1)
                local_tp_close_pct = 0.0
                tp_source = "trail_only"
            elif tp_mode == "pivot_v2":
                if from_zone:
                    tp1 = float(sig["S1"])
                    tp1_pips = max(0.0, (tp1 - entry_price) / PIP_SIZE)
                    local_tp_close_pct = partial_close_pct
                else:
                    tp1 = float(tp2)
                    tp1_pips = max(0.0, (tp1 - entry_price) / PIP_SIZE)
                    local_tp_close_pct = 1.0
            elif tp_mode == "single_pivot":
                tp1 = float(tp2)
                tp1_pips = max(0.0, (tp1 - entry_price) / PIP_SIZE)
                local_tp_close_pct = 1.0
            elif tp_mode == "single_atr":
                tp1_pips = min(single_tp_max_pips, max(single_tp_min_pips, single_tp_atr_mult * atr_pips))
                tp1 = entry_price + tp1_pips * PIP_SIZE
                tp2 = float(tp1)
                local_tp_close_pct = 1.0
                tp_source = "single_atr"
            else:
                tp1_pips = min(partial_tp_max_pips, max(partial_tp_min_pips, partial_tp_atr_mult * atr_pips))
                tp1 = entry_price + tp1_pips * PIP_SIZE
                local_tp_close_pct = partial_close_pct
        else:
            from_zone = bool(sig["from_zone"])
            sl_raw = (float(sig["R3"]) + sl_buf) if from_zone else (float(sig["R2"]) + sl_buf)
            if rejection_bonus_enabled and rejection_sl_improvement and bool(sig.get("rejection_confirmed", False)):
                rej_high = float(sig.get("rejection_high", np.nan))
                if np.isfinite(rej_high):
                    rej_sl_raw = rej_high + rejection_sl_buffer_pips * PIP_SIZE
                    base_sl_pips = (sl_raw - entry_price) / PIP_SIZE
                    rej_sl_pips = (rej_sl_raw - entry_price) / PIP_SIZE
                    if rej_sl_pips > 0 and rej_sl_pips < base_sl_pips:
                        sl_raw = rej_sl_raw
                        sl_source = "rejection_bonus"
            sl_pips = (sl_raw - entry_price) / PIP_SIZE
            sl_pips = max(min_sl_pips, min(max_sl_pips, sl_pips))
            sl_price = entry_price + sl_pips * PIP_SIZE

            tp_pivot = float(sig["P"])
            tp_mid = float(sig.get("session_midpoint", np.nan))
            if session_env_enabled and session_env_use_for_tp and session_env_tp_mode == "nearest_of_pivot_or_midpoint" and np.isfinite(tp_mid):
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
            if tp_mode == "trail_only":
                tp1_pips = 9999.0
                tp1 = entry_price - tp1_pips * PIP_SIZE
                tp2 = float(tp1)
                local_tp_close_pct = 0.0
                tp_source = "trail_only"
            elif tp_mode == "pivot_v2":
                if from_zone:
                    tp1 = float(sig["R1"])
                    tp1_pips = max(0.0, (entry_price - tp1) / PIP_SIZE)
                    local_tp_close_pct = partial_close_pct
                else:
                    tp1 = float(tp2)
                    tp1_pips = max(0.0, (entry_price - tp1) / PIP_SIZE)
                    local_tp_close_pct = 1.0
            elif tp_mode == "single_pivot":
                tp1 = float(tp2)
                tp1_pips = max(0.0, (entry_price - tp1) / PIP_SIZE)
                local_tp_close_pct = 1.0
            elif tp_mode == "single_atr":
                tp1_pips = min(single_tp_max_pips, max(single_tp_min_pips, single_tp_atr_mult * atr_pips))
                tp1 = entry_price - tp1_pips * PIP_SIZE
                tp2 = float(tp1)
                local_tp_close_pct = 1.0
                tp_source = "single_atr"
            else:
                tp1_pips = min(partial_tp_max_pips, max(partial_tp_min_pips, partial_tp_atr_mult * atr_pips))
                tp1 = entry_price - tp1_pips * PIP_SIZE
                local_tp_close_pct = partial_close_pct

        if sl_pips <= 0:
            diag["blocked_invalid_sl_distance"] += 1
            return False
        dd_mult = 1.0
        dd_tier = "full"
        if dd_enabled:
            dd_now = ((peak_equity - equity) / peak_equity * 100.0) if peak_equity > 0 else 0.0
            if dd_now > dd_t2:
                dd_mult = max(0.0, 1.0 - dd_t2_red)
                dd_tier = "tier2"
            elif dd_now > dd_t1:
                dd_mult = max(0.0, 1.0 - dd_t1_red)
                dd_tier = "tier1"
        hour_mult = 1.0
        if hp_enabled:
            hour_mult = float(hp_mults.get(str(int(row["hour_utc"])), 1.0))
        ss_size_mult = 1.0
        if ss_sizing_enabled:
            ss_tier = str(sig.get("signal_strength_tier", "weak")).strip().lower()
            if ss_tier == "strong":
                ss_size_mult = ss_sizing_strong_mult
            elif ss_tier == "moderate":
                ss_size_mult = ss_sizing_moderate_mult
            else:
                ss_size_mult = ss_sizing_weak_mult
        total_size_mult = float(sig.get("regime_mult", 1.0)) * float(sig.get("quality_mult", 1.0)) * hour_mult * dd_mult
        total_size_mult *= float(ss_size_mult)
        day_name_here = str(row.get("utc_day_name", ""))
        risk_pct_local = float(risk_pct) * float(day_risk_multipliers.get(day_name_here, 1.0))
        units = math.floor((equity * risk_pct_local) / (sl_pips * (PIP_SIZE / max(1e-9, entry_price))))
        units = int(math.floor(units * max(0.0, total_size_mult)))
        units = int(max(0, min(max_units, units)))
        if units < 1:
            diag["blocked_units_lt_1"] += 1
            return False
        if not margin_gate_allows(units):
            diag["blocked_margin_cap"] += 1
            return False

        trade_id += 1
        pos = Position(
            trade_id=trade_id,
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
        open_positions.append(pos)
        sst["trades"] = int(sst["trades"]) + 1
        sst["last_entry_time"] = ts
        sst.setdefault("entry_confluence_scores", []).append(int(sig["confluence_score"]))
        diag["entries_total"] += 1
        diag[f"entries_{sdir}"] += 1
        diag["bars_with_entry_triggered"] += 1
        if str(sig.get("quality_label", "medium")) == "high":
            signal_quality_stats["high_quality_trades"] += 1
        else:
            signal_quality_stats["medium_quality_trades"] += 1
        if dd_tier == "tier2":
            drawdown_sizing_stats["trades_at_tier2_reduction"] += 1
        elif dd_tier == "tier1":
            drawdown_sizing_stats["trades_at_tier1_reduction"] += 1
        else:
            drawdown_sizing_stats["trades_at_full_size"] += 1
        if total_size_mult < 0.999:
            regime_gate_stats["trades_at_reduced_size"] += 1
        else:
            regime_gate_stats["trades_at_full_size"] += 1
        if str(sig.get("entry_delay_type", "immediate")) == "delayed":
            momentum_check_stats["entries_delayed_then_filled"] += 1
        else:
            momentum_check_stats["entries_immediate"] += 1
        if ss_filter_enabled:
            ss_filter_stats["trades_above_threshold"] += 1
        combo_filter_stats["trades_allowed"] += 1
        return True

    for i, row in df.iterrows():
        diag["bars_total"] += 1
        ts = pd.Timestamp(row["time"])
        spread_pips_now = compute_spread_pips(
            i=i,
            ts=ts,
            mode=spread_mode,
            avg=spread_pips,
            mn=spread_min_pips,
            mx=spread_max_pips,
        )
        in_session = bool(row["in_tokyo_session"]) and bool(row["allowed_trading_day"])
        session_day = str(row["session_day_jst"])
        mid_open = float(row["open"])
        mid_high = float(row["high"])
        mid_low = float(row["low"])
        mid_close = float(row["close"])

        _, _ = get_bid_ask(mid_open, spread_pips_now)
        bid_high, ask_high = get_bid_ask(mid_high, spread_pips_now)
        bid_low, ask_low = get_bid_ask(mid_low, spread_pips_now)
        bid_close, ask_close = get_bid_ask(mid_close, spread_pips_now)
        remaining_session_minutes = minutes_to_session_end(int(row["minute_of_day_utc"])) if in_session else None

        # Expire stale pending signals.
        for sig in list(pending_signals):
            if i > int(sig["expiry_index"]):
                pending_signals.remove(sig)
                entry_confirmation_stats["signals_expired"] += 1

        # Expire unfilled limit entries.
        for lo in list(pending_limit_orders):
            if i > int(lo["expiry_index"]):
                pending_limit_orders.remove(lo)
                entry_improvement_stats["limits_expired"] += 1

        # Fill pending limit entries.
        if pending_limit_orders:
            for lo in list(pending_limit_orders):
                if i < int(lo["fill_start_index"]) or i > int(lo["expiry_index"]):
                    continue
                if str(lo["session_day"]) != session_day:
                    continue
                if not in_session:
                    continue
                sdir = str(lo["direction"])
                lp = float(lo["limit_price"])
                if sdir == "long":
                    fillable = (ask_low <= lp <= ask_high)
                else:
                    fillable = (bid_low <= lp <= bid_high)
                if not fillable:
                    continue
                sst_fill = ensure_session(session_day, row)
                sig = dict(lo["sig"])
                sig["confirmation_delay_candles"] = int(lo.get("confirmation_delay_candles", 0))
                opened = try_open_position(
                    sig=sig,
                    sdir=sdir,
                    entry_price=lp,
                    ts=ts,
                    i=i,
                    row=row,
                    sst=sst_fill,
                    spread_pips_now=spread_pips_now,
                )
                if opened:
                    entry_improvement_stats["limits_filled"] += 1
                    entry_improvement_stats["improvement_values_pips"].append(float(lo.get("improvement_pips", 0.0)))
                    entry_confirmation_stats["signals_confirmed"] += 1
                    entry_confirmation_stats["confirmation_delays"].append(int(lo.get("confirmation_delay_candles", 0)))
                pending_limit_orders.remove(lo)

        # Manage open positions each bar.
        for pos in list(open_positions):
            if pos.direction == "long":
                favorable = (bid_high - pos.entry_price) / PIP_SIZE
            else:
                favorable = (pos.entry_price - ask_low) / PIP_SIZE
            can_trail = (not trail_requires_tp1) or pos.tp1_hit
            if trail_enabled and can_trail and favorable >= trail_activate_pips:
                pos.trail_active = True
            if pos.trail_active and pos.units_remaining > 0:
                if pos.direction == "long":
                    new_trail = bid_close - trail_dist_pips * PIP_SIZE
                    pos.trail_stop_price = new_trail if pos.trail_stop_price is None else max(pos.trail_stop_price, new_trail)
                else:
                    new_trail = ask_close + trail_dist_pips * PIP_SIZE
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
                    and int(late_session_hard_close_all_minutes_before_end) > 0
                    and int(remaining_session_minutes) <= int(late_session_hard_close_all_minutes_before_end)
                ):
                    diag["late_session_hard_close_all"] += 1
                    close_position(pos, ts, bid_close, "late_session_hard_close" if not pos.tp1_hit else "tp1_then_late_session_hard_close")
                    open_positions.remove(pos)
                    continue
                if (
                    in_session
                    and remaining_session_minutes is not None
                    and int(late_session_be_or_close_minutes_before_end) > 0
                    and int(remaining_session_minutes) <= int(late_session_be_or_close_minutes_before_end)
                ):
                    if current_pips >= float(late_session_be_min_profit_pips):
                        be_px = pos.entry_price + float(late_session_be_offset_pips) * PIP_SIZE
                        if be_px > float(pos.sl_price):
                            pos.sl_price = be_px
                            pos.moved_to_breakeven = True
                            diag["late_session_be_set"] += 1
                    elif current_pips < 0:
                        diag["late_session_be_or_close_loss"] += 1
                        close_position(pos, ts, bid_close, "late_session_be_or_close_loss")
                        open_positions.remove(pos)
                        continue
                if (
                    in_session
                    and remaining_session_minutes is not None
                    and int(late_session_profit_tighten_minutes_before_end) > 0
                    and int(remaining_session_minutes) <= int(late_session_profit_tighten_minutes_before_end)
                    and current_pips > 0
                    and trail_enabled
                ):
                    tight_dist = max(0.1, float(trail_dist_pips) * float(late_session_profit_tighten_trail_mult))
                    tight_trail = bid_close - tight_dist * PIP_SIZE
                    pos.trail_active = True
                    pos.trail_stop_price = tight_trail if pos.trail_stop_price is None else max(pos.trail_stop_price, tight_trail)
                    diag["late_session_profit_trail_tighten"] += 1
                if (
                    late_session_enabled
                    and in_session
                    and remaining_session_minutes is not None
                    and int(remaining_session_minutes) <= int(late_session_minutes_before_end)
                ):
                    if (not pos.tp1_hit) and (current_pips < float(late_session_close_if_no_tp1_and_pips_below)):
                        diag["late_session_no_tp1_cut"] += 1
                        close_position(pos, ts, bid_close, "late_session_no_tp1_cut")
                        open_positions.remove(pos)
                        continue
                    if pos.tp1_hit and float(late_session_tp1_hit_tighten_trail_pips) > 0:
                        tight_trail = bid_close - float(late_session_tp1_hit_tighten_trail_pips) * PIP_SIZE
                        pos.trail_active = True
                        pos.trail_stop_price = tight_trail if pos.trail_stop_price is None else max(pos.trail_stop_price, tight_trail)
                        diag["late_session_tp1_trail_tighten"] += 1
                hit_sl = bid_low <= pos.sl_price
                hit_tp1 = (not pos.tp1_hit) and (bid_high >= pos.tp1_price)
                hit_tp2 = bid_high >= pos.tp2_price
                hit_trail = can_trail and pos.trail_active and pos.trail_stop_price is not None and (bid_low <= float(pos.trail_stop_price))
                hit_time_decay = pos.tp1_hit and held_minutes >= time_decay_minutes and (0.0 <= current_pips < time_decay_profit_cap_pips)
                hit_early_exit = (
                    early_exit_enabled
                    and (not pos.tp1_hit)
                    and held_minutes >= early_exit_time_min
                    and current_pips <= (-early_exit_loss_pips)
                    and float(pos.max_profit_seen_pips) <= early_exit_max_profit_seen
                )
                if hit_early_exit:
                    early_exit_stats["early_exits_triggered"] += 1
                    sl_pips = (pos.entry_price - pos.sl_price) / PIP_SIZE
                    early_exit_stats["loss_saved_pips"].append(max(0.0, float(sl_pips - abs(current_pips))))
                    close_position(pos, ts, bid_close, "early_exit_dead_wrong")
                    open_positions.remove(pos)
                    continue
                if hit_sl:
                    be_px = pos.entry_price + breakeven_offset_pips * PIP_SIZE
                    if pos.tp1_hit and pos.moved_to_breakeven and abs(pos.sl_price - be_px) <= 1e-9:
                        reason = "tp1_then_be_stop"
                    else:
                        reason = "tp1_then_sl" if pos.tp1_hit else "sl"
                    close_position(pos, ts, float(pos.sl_price), reason)
                    open_positions.remove(pos)
                    continue
                if hit_tp1:
                    close_units = int(math.floor(pos.units_initial * pos.tp1_close_pct))
                    close_units = max(1, min(close_units, pos.units_remaining))
                    pips, usd = calc_leg_usd_pips("long", pos.entry_price, pos.tp1_price, close_units)
                    pos.realized_pip_units += pips * close_units
                    pos.realized_usd += usd
                    pos.units_remaining -= close_units
                    pos.tp1_hit = True
                    pos.sl_price = pos.entry_price + breakeven_offset_pips * PIP_SIZE
                    pos.moved_to_breakeven = True
                    if pos.units_remaining <= 0:
                        close_position(pos, ts, float(pos.tp1_price), "tp")
                        open_positions.remove(pos)
                        continue
                if pos.tp1_hit and hit_tp2:
                    close_position(pos, ts, float(pos.tp2_price), "tp1_then_tp2")
                    open_positions.remove(pos)
                    continue
                if hit_trail:
                    close_position(pos, ts, float(pos.trail_stop_price), "tp1_then_trailing_stop")
                    open_positions.remove(pos)
                    continue
                if hit_time_decay:
                    close_position(pos, ts, bid_close, "tp1_then_time_decay")
                    open_positions.remove(pos)
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
                    and int(late_session_hard_close_all_minutes_before_end) > 0
                    and int(remaining_session_minutes) <= int(late_session_hard_close_all_minutes_before_end)
                ):
                    diag["late_session_hard_close_all"] += 1
                    close_position(pos, ts, ask_close, "late_session_hard_close" if not pos.tp1_hit else "tp1_then_late_session_hard_close")
                    open_positions.remove(pos)
                    continue
                if (
                    in_session
                    and remaining_session_minutes is not None
                    and int(late_session_be_or_close_minutes_before_end) > 0
                    and int(remaining_session_minutes) <= int(late_session_be_or_close_minutes_before_end)
                ):
                    if current_pips >= float(late_session_be_min_profit_pips):
                        be_px = pos.entry_price - float(late_session_be_offset_pips) * PIP_SIZE
                        if be_px < float(pos.sl_price):
                            pos.sl_price = be_px
                            pos.moved_to_breakeven = True
                            diag["late_session_be_set"] += 1
                    elif current_pips < 0:
                        diag["late_session_be_or_close_loss"] += 1
                        close_position(pos, ts, ask_close, "late_session_be_or_close_loss")
                        open_positions.remove(pos)
                        continue
                if (
                    in_session
                    and remaining_session_minutes is not None
                    and int(late_session_profit_tighten_minutes_before_end) > 0
                    and int(remaining_session_minutes) <= int(late_session_profit_tighten_minutes_before_end)
                    and current_pips > 0
                    and trail_enabled
                ):
                    tight_dist = max(0.1, float(trail_dist_pips) * float(late_session_profit_tighten_trail_mult))
                    tight_trail = ask_close + tight_dist * PIP_SIZE
                    pos.trail_active = True
                    pos.trail_stop_price = tight_trail if pos.trail_stop_price is None else min(pos.trail_stop_price, tight_trail)
                    diag["late_session_profit_trail_tighten"] += 1
                if (
                    late_session_enabled
                    and in_session
                    and remaining_session_minutes is not None
                    and int(remaining_session_minutes) <= int(late_session_minutes_before_end)
                ):
                    if (not pos.tp1_hit) and (current_pips < float(late_session_close_if_no_tp1_and_pips_below)):
                        diag["late_session_no_tp1_cut"] += 1
                        close_position(pos, ts, ask_close, "late_session_no_tp1_cut")
                        open_positions.remove(pos)
                        continue
                    if pos.tp1_hit and float(late_session_tp1_hit_tighten_trail_pips) > 0:
                        tight_trail = ask_close + float(late_session_tp1_hit_tighten_trail_pips) * PIP_SIZE
                        pos.trail_active = True
                        pos.trail_stop_price = tight_trail if pos.trail_stop_price is None else min(pos.trail_stop_price, tight_trail)
                        diag["late_session_tp1_trail_tighten"] += 1
                hit_sl = ask_high >= pos.sl_price
                hit_tp1 = (not pos.tp1_hit) and (ask_low <= pos.tp1_price)
                hit_tp2 = ask_low <= pos.tp2_price
                hit_trail = can_trail and pos.trail_active and pos.trail_stop_price is not None and (ask_high >= float(pos.trail_stop_price))
                hit_time_decay = pos.tp1_hit and held_minutes >= time_decay_minutes and (0.0 <= current_pips < time_decay_profit_cap_pips)
                hit_early_exit = (
                    early_exit_enabled
                    and (not pos.tp1_hit)
                    and held_minutes >= early_exit_time_min
                    and current_pips <= (-early_exit_loss_pips)
                    and float(pos.max_profit_seen_pips) <= early_exit_max_profit_seen
                )
                if hit_early_exit:
                    early_exit_stats["early_exits_triggered"] += 1
                    sl_pips = (pos.sl_price - pos.entry_price) / PIP_SIZE
                    early_exit_stats["loss_saved_pips"].append(max(0.0, float(sl_pips - abs(current_pips))))
                    close_position(pos, ts, ask_close, "early_exit_dead_wrong")
                    open_positions.remove(pos)
                    continue
                if hit_sl:
                    be_px = pos.entry_price - breakeven_offset_pips * PIP_SIZE
                    if pos.tp1_hit and pos.moved_to_breakeven and abs(pos.sl_price - be_px) <= 1e-9:
                        reason = "tp1_then_be_stop"
                    else:
                        reason = "tp1_then_sl" if pos.tp1_hit else "sl"
                    close_position(pos, ts, float(pos.sl_price), reason)
                    open_positions.remove(pos)
                    continue
                if hit_tp1:
                    close_units = int(math.floor(pos.units_initial * pos.tp1_close_pct))
                    close_units = max(1, min(close_units, pos.units_remaining))
                    pips, usd = calc_leg_usd_pips("short", pos.entry_price, pos.tp1_price, close_units)
                    pos.realized_pip_units += pips * close_units
                    pos.realized_usd += usd
                    pos.units_remaining -= close_units
                    pos.tp1_hit = True
                    pos.sl_price = pos.entry_price - breakeven_offset_pips * PIP_SIZE
                    pos.moved_to_breakeven = True
                    if pos.units_remaining <= 0:
                        close_position(pos, ts, float(pos.tp1_price), "tp")
                        open_positions.remove(pos)
                        continue
                if pos.tp1_hit and hit_tp2:
                    close_position(pos, ts, float(pos.tp2_price), "tp1_then_tp2")
                    open_positions.remove(pos)
                    continue
                if hit_trail:
                    close_position(pos, ts, float(pos.trail_stop_price), "tp1_then_trailing_stop")
                    open_positions.remove(pos)
                    continue
                if hit_time_decay:
                    close_position(pos, ts, ask_close, "tp1_then_time_decay")
                    open_positions.remove(pos)
                    continue

        # Force close if outside session.
        if not in_session:
            if str(row["utc_day_name"]) in {"Saturday"}:
                diag["bars_weekend"] += 1
            elif not bool(row["allowed_trading_day"]):
                diag["bars_disallowed_day"] += 1
            elif not bool(row["in_tokyo_session"]):
                diag["bars_outside_tokyo_hours"] += 1
            else:
                diag["bars_outside_session_other"] += 1
            if open_positions:
                for pos in list(open_positions):
                    px = bid_close if pos.direction == "long" else ask_close
                    reason = "tp1_then_session_close" if pos.tp1_hit else "session_close"
                    close_position(pos, ts, px, reason)
                    open_positions.remove(pos)
            continue

        diag["bars_in_session"] += 1
        diag["bars_passed_session_filter"] += 1
        sst = ensure_session(session_day, row)
        in_lunch_block = bool(lunch_block_enabled and in_window(int(row["minute_of_day_utc"]), lunch_start_min, lunch_end_min))
        if in_lunch_block:
            diag["blocked_lunch_deadzone"] += 1
            continue
        diag["bars_passed_lunch_filter"] += 1
        # V10: maintain developing session envelope (IR + running session range).
        sst["session_high"] = max(float(sst.get("session_high", mid_high)), float(mid_high))
        sst["session_low"] = min(float(sst.get("session_low", mid_low)), float(mid_low))
        if session_env_enabled and not bool(sst.get("ir_ready", False)):
            if ts <= pd.Timestamp(sst.get("warmup_end_ts")):
                sst["ir_high"] = max(float(sst.get("ir_high", mid_high)), float(mid_high))
                sst["ir_low"] = min(float(sst.get("ir_low", mid_low)), float(mid_low))
            else:
                sst["ir_ready"] = True
        if daily_range_enabled and not bool(sst.get("daily_range_allowed", True)):
            reason = str(sst.get("daily_range_block_reason", "unknown"))
            if reason == "high":
                diag["blocked_daily_range_high"] += 1
            elif reason == "low":
                diag["blocked_daily_range_low"] += 1
            else:
                diag["blocked_daily_range_missing"] += 1
            continue
        if trend_skip_enabled and not bool(sst.get("trend_skip_allowed", True)):
            diag["blocked_trend_regime_skip"] += 1
            continue
        if stop_after_consecutive_losses > 0 and int(sst.get("consec_losses", 0)) >= stop_after_consecutive_losses:
            sst["stopped"] = True
            diag["blocked_consecutive_loss_stop"] += 1
            continue
        if regime_switch_enabled and not bool(sst.get("regime_switch_allowed", True)):
            diag["blocked_regime_switch"] += 1
            continue
        current_dd_pct = ((peak_equity - equity) / peak_equity * 100.0) if peak_equity > 0 else 0.0
        if dd_enabled:
            if (not trading_paused_dd) and current_dd_pct > dd_t3:
                trading_paused_dd = True
                drawdown_sizing_stats["times_trading_paused"] += 1
            if trading_paused_dd and current_dd_pct <= dd_resume:
                trading_paused_dd = False
        if trading_paused_dd:
            drawdown_sizing_stats["total_bars_paused"] += 1
            diag["blocked_drawdown_pause"] += 1
            continue

        # Indicator availability check.
        if any(pd.isna(row[c]) for c in piv_cols):
            diag["blocked_missing_pivots"] += 1
            continue
        if pd.isna(row["bb_upper"]) or pd.isna(row["bb_lower"]) or pd.isna(row["bb_mid"]) or pd.isna(row["rsi_m5"]) or pd.isna(row["atr_m15"]):
            diag["blocked_missing_indicators"] += 1
            continue
        diag["bars_passed_indicator_check"] += 1

        # Breakout detection gate.
        breakout_blocked = False
        if breakout_mode == "from_open":
            moved_up_pips = (float(sst["session_high"]) - float(sst["session_open_price"])) / PIP_SIZE
            moved_dn_pips = (float(sst["session_open_price"]) - float(sst["session_low"])) / PIP_SIZE
            moved_from_open_pips = max(moved_up_pips, moved_dn_pips)
            if moved_from_open_pips > breakout_disable_pips:
                sst["stopped"] = True
                diag["gate_breakout_disable_triggered"] += 1
                diag["blocked_breakout_from_open"] += 1
                breakout_blocked = True
            if bool(sst["stopped"]):
                diag["blocked_session_stopped"] += 1
                breakout_blocked = True
        elif breakout_mode == "rolling":
            rw = sst["rolling_window"]
            rw.append((ts, mid_high, mid_low))
            cutoff = ts - pd.Timedelta(minutes=rolling_window_minutes)
            while rw and pd.Timestamp(rw[0][0]) < cutoff:
                rw.pop(0)
            if sst.get("breakout_cooldown_until") is not None and ts < pd.Timestamp(sst["breakout_cooldown_until"]):
                breakout_blocked = True
                diag["blocked_breakout_cooldown_active"] += 1
            else:
                highs = [x[1] for x in rw]
                lows = [x[2] for x in rw]
                rolling_range_pips = (max(highs) - min(lows)) / PIP_SIZE if highs and lows else 0.0
                if rolling_range_pips > rolling_range_threshold_pips:
                    sst["breakout_cooldown_until"] = ts + pd.Timedelta(minutes=breakout_cooldown_minutes)
                    breakout_blocked = True
                    diag["gate_breakout_disable_triggered"] += 1
                    diag["blocked_breakout_rolling_range"] += 1
                else:
                    sst["breakout_cooldown_until"] = None
        elif breakout_mode == "disabled":
            pass
        else:
            raise ValueError(f"Unsupported breakout_detection_mode: {breakout_mode}")
        if breakout_blocked:
            continue
        diag["bars_passed_breakout_check"] += 1

        regime_col = "bb_regime_expanding3" if regime_filter_mode in {"expanding3", "bb_width_expanding3"} else "bb_regime"
        if str(row.get(regime_col, "trending")) != "ranging":
            diag["blocked_regime_not_ranging"] += 1
            continue
        diag["bars_passed_regime_filter"] += 1
        if atr_gate_enabled:
            if atr_pct_enabled:
                atr_pct_rank_val = float(row.get("atr_m15_percentile_rank", np.nan))
                if not np.isfinite(atr_pct_rank_val):
                    diag["blocked_atr_percentile_insufficient"] += 1
                    continue
                if atr_pct_rank_val > atr_pct_max:
                    diag["blocked_atr_above_percentile"] += 1
                    continue
            else:
                if float(row["atr_m15"]) > atr_max:
                    diag["blocked_atr_above_max"] += 1
                    continue
        diag["bars_passed_atr_filter"] += 1
        diag["bars_after_global_filters"] += 1
        if news_enabled and bool(row.get("news_blocked", False)):
            diag["blocked_news_window"] += 1
            continue
        diag["bars_reached_signal_scoring"] += 1

        P = float(row["pivot_P"])
        R1 = float(row["pivot_R1"])
        R2 = float(row["pivot_R2"])
        R3 = float(row["pivot_R3"])
        S1 = float(row["pivot_S1"])
        S2 = float(row["pivot_S2"])
        S3 = float(row["pivot_S3"])
        bb_u = float(row["bb_upper"])
        bb_l = float(row["bb_lower"])
        rsi = float(row["rsi_m5"])
        sar = float(row["sar_value"])
        sar_dir = str(row["sar_direction"])

        # Regime gate score/classification.
        regime_label = "neutral"
        regime_size_mult = 1.0
        if regime_enabled:
            score = 0
            atr_fast = float(row.get("atr_m15", np.nan))
            atr_slow = float(row.get("atr_m15_slow", np.nan))
            atr_ratio = (atr_fast / atr_slow) if (not pd.isna(atr_fast) and not pd.isna(atr_slow) and atr_slow > 0) else np.nan
            if not pd.isna(atr_ratio):
                if atr_ratio > atr_ratio_trend:
                    score -= 1
                elif atr_ratio < atr_ratio_calm:
                    score += 1
            bbw = float(row.get("bb_width", np.nan))
            bbw_hi = float(row.get("bb_width_q_high", np.nan))
            bbw_lo = float(row.get("bb_width_q_low", np.nan))
            if not pd.isna(bbw) and not pd.isna(bbw_hi) and bbw > bbw_hi:
                score -= 1
            elif not pd.isna(bbw) and not pd.isna(bbw_lo) and bbw < bbw_lo:
                score += 1
            adx = float(row.get("adx_m15", np.nan))
            if not pd.isna(adx):
                if adx > adx_trend:
                    score -= 1
                elif adx < adx_range:
                    score += 1
            if S1 <= mid_close <= R1:
                score += 1
            if (mid_close <= S2) or (mid_close >= R2):
                score -= 1
            if score >= favorable_min_score:
                regime_label = "favorable"
                regime_gate_stats["bars_favorable"] += 1
            elif score == neutral_min_score:
                regime_label = "neutral"
                regime_size_mult = neutral_size_mult
                regime_gate_stats["bars_neutral"] += 1
            else:
                regime_label = "unfavorable"
                regime_gate_stats["bars_unfavorable"] += 1
                regime_gate_stats["trades_blocked_by_regime"] += 1
                diag["blocked_regime_gate_unfavorable"] += 1
                continue

        long_signal = False
        short_signal = False
        long_score = 0
        short_score = 0
        cond_long_zone = mid_close <= (S1 + tol)
        cond_short_zone = mid_close >= (R1 - tol)
        cond_long_bb = (mid_close <= bb_l) or (mid_low <= bb_l)
        cond_short_bb = (mid_close >= bb_u) or (mid_high >= bb_u)
        cond_long_sar = bool(row.get("sar_flip_bullish_recent", False))
        cond_short_sar = bool(row.get("sar_flip_bearish_recent", False))
        cond_long_rsi_soft = rsi < long_rsi_soft_entry
        cond_short_rsi_soft = rsi > short_rsi_soft_entry
        cond_long_rsi_ext = rsi < long_rsi_bonus
        cond_short_rsi_ext = rsi > short_rsi_bonus
        cond_long_s2 = mid_close <= (S2 + tol)
        cond_short_r2 = mid_close >= (R2 - tol)
        cond_atr_low = float(row["atr_m15"]) <= atr_max

        if cond_long_zone:
            diag["bars_with_long_signal_zone"] += 1
        if cond_short_zone:
            diag["bars_with_short_signal_zone"] += 1
        if cond_long_bb or cond_short_bb:
            diag["bars_with_bb_touch"] += 1
        if cond_long_sar or cond_short_sar:
            diag["bars_with_sar_flip"] += 1
        diag["long_cond_zone_true"] += int(cond_long_zone)
        diag["long_cond_bb_true"] += int(cond_long_bb)
        diag["long_cond_sar_true"] += int(cond_long_sar)
        diag["long_cond_rsi_soft_true"] += int(cond_long_rsi_soft)
        diag["short_cond_zone_true"] += int(cond_short_zone)
        diag["short_cond_bb_true"] += int(cond_short_bb)
        diag["short_cond_sar_true"] += int(cond_short_sar)
        diag["short_cond_rsi_soft_true"] += int(cond_short_rsi_soft)

        if tokyo_v2_scoring:
            long_conditions = {
                "pivot_S1": bool(cond_long_zone),
                "pivot_S2": bool(cond_long_s2),
                "bb_lower": bool(cond_long_bb),
                "sar_flip": bool(cond_long_sar),
                "rsi_oversold": bool(cond_long_rsi_soft),
                "rsi_extreme": bool(cond_long_rsi_ext),
                "atr_low": bool(cond_atr_low),
            }
            short_conditions = {
                "pivot_R1": bool(cond_short_zone),
                "pivot_R2": bool(cond_short_r2),
                "bb_upper": bool(cond_short_bb),
                "sar_flip": bool(cond_short_sar),
                "rsi_overbought": bool(cond_short_rsi_soft),
                "rsi_extreme": bool(cond_short_rsi_ext),
                "atr_low": bool(cond_atr_low),
            }

            for k, v in long_conditions.items():
                condition_check_counts[f"long_{k}"] += 1
                condition_hit_counts[f"long_{k}"] += int(v)
            for k, v in short_conditions.items():
                condition_check_counts[f"short_{k}"] += 1
                condition_hit_counts[f"short_{k}"] += int(v)

            long_score = int(sum(1 for v in long_conditions.values() if v))
            short_score = int(sum(1 for v in short_conditions.values() if v))
            if long_score >= 1:
                diag["signals_score_ge_1"] += 1
                sst["signals_generated"] = int(sst.get("signals_generated", 0)) + 1
            if short_score >= 1:
                diag["signals_score_ge_1"] += 1
                sst["signals_generated"] = int(sst.get("signals_generated", 0)) + 1
            if long_score >= 2:
                diag["signals_score_ge_2"] += 1
            if short_score >= 2:
                diag["signals_score_ge_2"] += 1
            if long_score >= 3:
                diag["signals_score_ge_3"] += 1
            if short_score >= 3:
                diag["signals_score_ge_3"] += 1

            long_signal = long_score >= confluence_min_long
            short_signal = short_score >= confluence_min_short
            diag["long_confluence_min_met"] += int(long_signal)
            diag["short_confluence_min_met"] += int(short_signal)

            if long_score == max(0, confluence_min_long - 1):
                near_miss_candidates.append(
                    {
                        "index": int(i),
                        "datetime": str(ts),
                        "direction": "long",
                        "price": float(mid_close),
                        "score": int(long_score),
                        "met_conditions": [k for k, v in long_conditions.items() if v],
                        "failed_conditions": [k for k, v in long_conditions.items() if not v],
                    }
                )
            if short_score == max(0, confluence_min_short - 1):
                near_miss_candidates.append(
                    {
                        "index": int(i),
                        "datetime": str(ts),
                        "direction": "short",
                        "price": float(mid_close),
                        "score": int(short_score),
                        "met_conditions": [k for k, v in short_conditions.items() if v],
                        "failed_conditions": [k for k, v in short_conditions.items() if not v],
                    }
                )
        else:
            long_core_flags = []
            if core_gate_use_zone:
                long_core_flags.append(bool(cond_long_zone))
            if core_gate_use_bb:
                long_core_flags.append(bool(cond_long_bb))
            if core_gate_use_sar:
                long_core_flags.append(bool(cond_long_sar))
            if core_gate_use_rsi:
                long_core_flags.append(bool(cond_long_rsi_soft))
            long_core_ok = (sum(1 for x in long_core_flags if x) >= max(1, min(core_gate_required, len(long_core_flags)))) if long_core_flags else True
            if long_core_ok:
                diag["long_all_core_conditions_true"] += 1
                long_score += 1 if cond_long_zone else 0
                long_score += 1 if cond_long_bb else 0
                long_score += 1 if cond_long_sar else 0
                long_score += 1 if cond_long_rsi_ext else 0
                long_score += 1 if cond_long_s2 else 0
                long_signal = long_score >= confluence_min_long
                diag["long_confluence_min_met"] += int(long_signal)

            short_core_flags = []
            if core_gate_use_zone:
                short_core_flags.append(bool(cond_short_zone))
            if core_gate_use_bb:
                short_core_flags.append(bool(cond_short_bb))
            if core_gate_use_sar:
                short_core_flags.append(bool(cond_short_sar))
            if core_gate_use_rsi:
                short_core_flags.append(bool(cond_short_rsi_soft))
            short_core_ok = (sum(1 for x in short_core_flags if x) >= max(1, min(core_gate_required, len(short_core_flags)))) if short_core_flags else True
            if short_core_ok:
                diag["short_all_core_conditions_true"] += 1
                short_score += 1 if cond_short_zone else 0
                short_score += 1 if cond_short_bb else 0
                short_score += 1 if cond_short_sar else 0
                short_score += 1 if cond_short_rsi_ext else 0
                short_score += 1 if cond_short_r2 else 0
                short_signal = short_score >= confluence_min_short
                diag["short_confluence_min_met"] += int(short_signal)
            if long_score >= 1:
                diag["signals_score_ge_1"] += 1
                sst["signals_generated"] = int(sst.get("signals_generated", 0)) + 1
            if short_score >= 1:
                diag["signals_score_ge_1"] += 1
                sst["signals_generated"] = int(sst.get("signals_generated", 0)) + 1
            if long_score >= 2:
                diag["signals_score_ge_2"] += 1
            if short_score >= 2:
                diag["signals_score_ge_2"] += 1
            if long_score >= 3:
                diag["signals_score_ge_3"] += 1
            if short_score >= 3:
                diag["signals_score_ge_3"] += 1
        if long_signal or short_signal:
            diag["bars_with_confluence_met"] += 1
        if long_signal and short_signal:
            diag["blocked_ambiguous_both_signals"] += 1
            continue
        if not long_signal and not short_signal:
            diag["blocked_no_signal"] += 1
        else:
            direction = "long" if long_signal else "short"
            diag["signals_pre_v7_filters"] += 1
            A = bool(cond_long_zone if direction == "long" else cond_short_zone)
            B = bool(cond_long_bb if direction == "long" else cond_short_bb)
            C = bool(cond_long_sar if direction == "long" else cond_short_sar)
            D = bool((rsi < long_rsi_bonus) if direction == "long" else (rsi > short_rsi_bonus))
            E = bool((mid_close <= (S2 + tol)) if direction == "long" else (mid_close >= (R2 - tol)))
            combo = "".join(x for x,ok in [("A",A),("B",B),("C",C),("D",D),("E",E)] if ok)
            signal_strength_score = 0
            signal_strength_tier = "weak"
            if ss_enabled:
                cc_map = ss_comp.get("confluence_count", {"2": 1, "3": 2, "4": 3, "5": 4})
                cscore = int(long_score if direction == "long" else short_score)
                signal_strength_score += int(cc_map.get(str(cscore), 0))
                bb_pen = ((bb_l - mid_close) / PIP_SIZE) if direction == "long" else ((mid_close - bb_u) / PIP_SIZE)
                if float(bb_pen) > float(ss_comp.get("bb_penetration_bonus_pips", 2)):
                    signal_strength_score += 1
                rsi_ext = float(ss_comp.get("rsi_extreme_bonus_threshold", 25))
                if (direction == "long" and rsi < rsi_ext) or (direction == "short" and rsi > (100.0 - rsi_ext)):
                    signal_strength_score += 1
                if E:
                    signal_strength_score += 1
                same_flip = bool(row.get("sar_flip_bullish", False)) if direction == "long" else bool(row.get("sar_flip_bearish", False))
                if same_flip:
                    signal_strength_score += 1
                fav_hours = set(int(h) for h in ss_comp.get("favorable_hour", [17, 18, 21]))
                if int(row["hour_utc"]) in fav_hours:
                    signal_strength_score += 1
                if signal_strength_score <= 3:
                    signal_strength_tier = "weak"
                elif signal_strength_score <= 6:
                    signal_strength_tier = "moderate"
                else:
                    signal_strength_tier = "strong"
            if combo_filter_enabled:
                allow = True
                if combo_filter_mode == "allowlist":
                    allow = combo in combo_allow
                elif combo_filter_mode == "blocklist":
                    allow = combo not in combo_block
                if not allow:
                    combo_filter_stats["trades_blocked_by_combo"] += 1
                    diag["blocked_combo_filter"] += 1
                    continue
            if ss_filter_enabled and int(signal_strength_score) < ss_filter_min_score:
                ss_filter_stats["trades_below_threshold_skipped"] += 1
                diag["blocked_signal_strength_filter"] += 1
                continue
            quality_label = "medium"
            quality_mult = medium_quality_mult
            if cq_enabled:
                if combo in top_combos:
                    quality_label = "high"
                    quality_mult = high_quality_mult
                elif combo in bottom_combos:
                    quality_label = "low"
                    quality_mult = 0.0
                    if low_quality_skip:
                        signal_quality_stats["low_quality_skipped"] += 1
                        diag["blocked_low_quality_combo"] += 1
                        continue
            diag["signal_long"] += int(direction == "long")
            diag["signal_short"] += int(direction == "short")

            # V10 quality markers (log-only except rejection bonus used during sizing/SL).
            rejection_confirmed = False
            rejection_low = np.nan
            rejection_high = np.nan
            rejection_wick_ratio = np.nan
            if rejection_bonus_enabled:
                if direction == "long":
                    rejection_confirmed = bool(row.get("rej_bull_recent", False))
                    rejection_low = float(row.get("rej_bull_low_recent", np.nan))
                    rejection_wick_ratio = float(row.get("rej_wick_ratio_bull", np.nan))
                else:
                    rejection_confirmed = bool(row.get("rej_bear_recent", False))
                    rejection_high = float(row.get("rej_bear_high_recent", np.nan))
                    rejection_wick_ratio = float(row.get("rej_wick_ratio_bear", np.nan))
            divergence_present = False
            if div_track_enabled:
                divergence_present = bool(row.get("rsi_div_bull_recent", False)) if direction == "long" else bool(row.get("rsi_div_bear_recent", False))
            session_midpoint = np.nan
            inside_ir = False
            dist_ir_boundary = np.nan
            if session_env_enabled:
                s_hi = float(sst.get("session_high", np.nan))
                s_lo = float(sst.get("session_low", np.nan))
                if np.isfinite(s_hi) and np.isfinite(s_lo):
                    session_midpoint = (s_hi + s_lo) / 2.0
                if session_env_log_ir_pos and bool(sst.get("ir_ready", False)):
                    ir_hi = float(sst.get("ir_high", np.nan))
                    ir_lo = float(sst.get("ir_low", np.nan))
                    if np.isfinite(ir_hi) and np.isfinite(ir_lo):
                        inside_ir = bool(ir_lo <= mid_close <= ir_hi)
                        dist_ir_boundary = min(abs(mid_close - ir_lo), abs(mid_close - ir_hi)) / PIP_SIZE
            dist_to_midpoint = abs(mid_close - session_midpoint) / PIP_SIZE if np.isfinite(session_midpoint) else np.nan
            dist_to_pivot = abs(mid_close - P) / PIP_SIZE
            marker_list = []
            if rejection_confirmed:
                marker_list.append("rejection")
            if divergence_present:
                marker_list.append("divergence")
            if inside_ir:
                marker_list.append("inside_ir")
            quality_markers = ",".join(marker_list)

            sig_obj = {
                "direction": direction,
                "session_day": session_day,
                "signal_index": int(i),
                "signal_time": ts,
                "expiry_index": int(i + (confirmation_window_bars if confirmation_type == "m1" else confirmation_window_bars * 5)),
                "confluence_score": int(long_score if long_signal else short_score),
                "confluence_combo": combo,
                "signal_strength_score": int(signal_strength_score),
                "signal_strength_tier": str(signal_strength_tier),
                "quality_label": quality_label,
                "quality_mult": float(quality_mult),
                "regime_label": regime_label,
                "regime_mult": float(regime_size_mult),
                "from_zone": bool(mid_close <= (S2 + tol) if direction == "long" else mid_close >= (R2 - tol)),
                "P": P,
                "R1": R1,
                "R2": R2,
                "R3": R3,
                "S1": S1,
                "S2": S2,
                "S3": S3,
                "bb_upper": bb_u,
                "bb_lower": bb_l,
                "sar_value": sar,
                "sar_direction": sar_dir,
                "rsi_m5": rsi,
                "atr_m15": float(row["atr_m15"]),
                "bb_mid": float(row["bb_mid"]),
                "rejection_confirmed": bool(rejection_confirmed),
                "rejection_low": float(rejection_low) if np.isfinite(rejection_low) else np.nan,
                "rejection_high": float(rejection_high) if np.isfinite(rejection_high) else np.nan,
                "rejection_wick_ratio": float(rejection_wick_ratio) if np.isfinite(rejection_wick_ratio) else np.nan,
                "divergence_present": bool(divergence_present),
                "inside_ir": bool(inside_ir),
                "quality_markers": quality_markers,
                "session_midpoint": float(session_midpoint) if np.isfinite(session_midpoint) else np.nan,
                "distance_to_ir_boundary_pips": float(dist_ir_boundary) if np.isfinite(dist_ir_boundary) else np.nan,
                "distance_to_midpoint_pips": float(dist_to_midpoint) if np.isfinite(dist_to_midpoint) else np.nan,
                "distance_to_pivot_pips": float(dist_to_pivot),
                "entry_delay_type": "immediate",
                "momentum_delays": 0,
            }
            entry_confirmation_stats["signals_generated"] += 1
            if confirmation_enabled and confirmation_window_bars > 0:
                has_pending_same = any(
                    ps["session_day"] == session_day and ps["direction"] == direction and i <= int(ps["expiry_index"])
                    for ps in pending_signals
                )
                if not has_pending_same:
                    pending_signals.append(sig_obj)
            else:
                sig_obj["confirmation_delay_candles"] = 0
                sig_obj["confirmation_close"] = float(mid_close)
                entry_px_now = ask_close if direction == "long" else bid_close
                opened_now = try_open_position(
                    sig=sig_obj,
                    sdir=direction,
                    entry_price=float(entry_px_now),
                    ts=ts,
                    i=i,
                    row=row,
                    sst=sst,
                    spread_pips_now=spread_pips_now,
                )
                if opened_now:
                    entry_confirmation_stats["signals_confirmed"] += 1
                    entry_confirmation_stats["confirmation_delays"].append(0)

        # Confirmation entries (must be on a later bar than signal bar).
        for sig in list(pending_signals):
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
            if confirmation_type == "m5":
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
            # Momentum check (delay once if immediate momentum is against entry).
            if momentum_enabled:
                j0 = max(0, i - momentum_lookback + 1)
                y = df.iloc[j0 : i + 1]["close"].to_numpy(dtype=float)
                if len(y) >= 2:
                    x = np.arange(len(y), dtype=float)
                    xv = x - x.mean()
                    yv = y - y.mean()
                    den = float((xv * xv).sum())
                    slope_price = float((xv * yv).sum() / den) if den > 0 else 0.0
                    slope_pips = slope_price / PIP_SIZE
                else:
                    slope_pips = 0.0
                adverse = (slope_pips < -momentum_slope_th) if sdir == "long" else (slope_pips > momentum_slope_th)
                if adverse:
                    if int(sig.get("momentum_delays", 0)) < momentum_max_delays and not in_delayed_recheck:
                        sig["momentum_delays"] = int(sig.get("momentum_delays", 0)) + 1
                        sig["momentum_delay_until"] = int(i + momentum_delay_candles)
                        sig["entry_delay_type"] = "delayed"
                        continue
                    momentum_check_stats["entries_delayed_then_expired"] += 1
                    entry_confirmation_stats["signals_expired"] += 1
                    pending_signals.remove(sig)
                    continue
            sig["confirmation_delay_candles"] = int(i - int(sig["signal_index"]))
            sig["confirmation_close"] = float(mid_close)
            if entry_imp_enabled:
                if sdir == "long":
                    limit_price = mid_low + entry_imp_long_offset * PIP_SIZE
                    market_ref = ask_close
                else:
                    limit_price = mid_high - entry_imp_short_offset * PIP_SIZE
                    market_ref = bid_close
                entry_improvement_stats["signals_with_limit_placed"] += 1
                best_fill_idx = None
                for j in range(i + 1, min(len(df), i + 1 + entry_imp_fill_window)):
                    rj = df.iloc[j]
                    sp_j = compute_spread_pips(
                        i=int(j),
                        ts=pd.Timestamp(rj["time"]),
                        mode=spread_mode,
                        avg=spread_pips,
                        mn=spread_min_pips,
                        mx=spread_max_pips,
                    )
                    bj_low, aj_low = get_bid_ask(float(rj["low"]), sp_j)
                    bj_high, aj_high = get_bid_ask(float(rj["high"]), sp_j)
                    if sdir == "long":
                        ok = (aj_low <= limit_price <= aj_high)
                    else:
                        ok = (bj_low <= limit_price <= bj_high)
                    if ok:
                        best_fill_idx = int(j)
                        break
                if best_fill_idx is None:
                    entry_improvement_stats["limits_expired"] += 1
                else:
                    improvement = ((market_ref - limit_price) / PIP_SIZE) if sdir == "long" else ((limit_price - market_ref) / PIP_SIZE)
                    pending_limit_orders.append(
                        {
                            "sig": dict(sig),
                            "direction": sdir,
                            "session_day": session_day,
                            "fill_start_index": int(i + 1),
                            "expiry_index": int(best_fill_idx),
                            "limit_price": float(limit_price),
                            "improvement_pips": float(improvement),
                            "confirmation_delay_candles": int(sig["confirmation_delay_candles"]),
                        }
                    )
            else:
                entry_px = ask_close if sdir == "long" else bid_close
                opened = try_open_position(
                    sig=sig,
                    sdir=sdir,
                    entry_price=float(entry_px),
                    ts=ts,
                    i=i,
                    row=row,
                    sst=sst,
                    spread_pips_now=spread_pips_now,
                )
                if opened:
                    entry_confirmation_stats["signals_confirmed"] += 1
                    entry_confirmation_stats["confirmation_delays"].append(int(sig["confirmation_delay_candles"]))
            pending_signals.remove(sig)

    if pending_signals:
        entry_confirmation_stats["signals_expired"] += len(pending_signals)
        pending_signals = []
    if pending_limit_orders:
        entry_improvement_stats["limits_expired"] += len(pending_limit_orders)
        pending_limit_orders = []

    required_diag_keys = [
        "bars_passed_session_filter",
        "bars_passed_lunch_filter",
        "bars_passed_indicator_check",
        "bars_passed_breakout_check",
        "bars_passed_atr_filter",
        "bars_passed_regime_filter",
        "bars_reached_signal_scoring",
        "bars_with_long_signal_zone",
        "bars_with_short_signal_zone",
        "bars_with_bb_touch",
        "bars_with_sar_flip",
        "bars_with_confluence_met",
        "bars_with_entry_triggered",
        "blocked_consecutive_loss_stop",
        "blocked_consecutive_loss_pause",
        "blocked_session_pnl_stop",
    ]
    for k in required_diag_keys:
        _ = diag[k]

    if diag["bars_reached_signal_scoring"] == 0:
        blocker_fields = [
            "blocked_missing_indicators",
            "blocked_missing_pivots",
            "blocked_breakout_from_open",
            "blocked_breakout_rolling_range",
            "blocked_breakout_cooldown_active",
            "blocked_regime_not_ranging",
            "blocked_atr_above_max",
            "blocked_session_stopped",
        ]
        blocker_counts = {k: int(diag[k]) for k in blocker_fields}
        top_blocker = max(blocker_counts.items(), key=lambda kv: kv[1])[0]
        print(
            "bars_reached_signal_scoring is 0; top blocker is "
            f"{top_blocker}={blocker_counts[top_blocker]}. "
            f"Thresholds: atr_max={atr_max}, tol_pips={tol_pips}, "
            f"breakout_mode={breakout_mode}, rolling_range_threshold_pips={rolling_range_threshold_pips}, "
            f"breakout_from_open_pips={breakout_disable_pips}"
        )

    # Near-miss diagnostics (score exactly one below threshold, capped to 200 evenly sampled).
    near_miss_log: list[dict] = []
    if near_miss_candidates:
        max_nm = min(200, len(near_miss_candidates))
        if len(near_miss_candidates) > max_nm:
            idxs = np.linspace(0, len(near_miss_candidates) - 1, max_nm).astype(int).tolist()
            sampled_nm = [near_miss_candidates[j] for j in idxs]
        else:
            sampled_nm = near_miss_candidates
        for nm in sampled_nm:
            i0 = int(nm.get("index", -1))
            px = float(nm.get("price", np.nan))
            future = df.iloc[i0 + 1 : min(len(df), i0 + 31)] if i0 >= 0 else pd.DataFrame()
            if future.empty or not np.isfinite(px):
                nm["next_30m_outcome"] = "insufficient_data"
                nm["next_30m_favorable_move_pips"] = None
                nm["next_30m_adverse_move_pips"] = None
            else:
                if str(nm.get("direction")) == "long":
                    fav = (float(future["high"].max()) - px) / PIP_SIZE
                    adv = (px - float(future["low"].min())) / PIP_SIZE
                else:
                    fav = (px - float(future["low"].min())) / PIP_SIZE
                    adv = (float(future["high"].max()) - px) / PIP_SIZE
                if fav > adv:
                    outcome = "moved_in_would_be_direction"
                elif adv > fav:
                    outcome = "moved_against_would_be_direction"
                else:
                    outcome = "mixed_or_flat"
                nm["next_30m_outcome"] = outcome
                nm["next_30m_favorable_move_pips"] = float(fav)
                nm["next_30m_adverse_move_pips"] = float(adv)
            nm.pop("index", None)
            near_miss_log.append(nm)

    filter_hit_rates = {
        "pivot_S1_or_R1": {
            "true": int(condition_hit_counts["long_pivot_S1"] + condition_hit_counts["short_pivot_R1"]),
            "checks": int(condition_check_counts["long_pivot_S1"] + condition_check_counts["short_pivot_R1"]),
        },
        "pivot_S2_or_R2": {
            "true": int(condition_hit_counts["long_pivot_S2"] + condition_hit_counts["short_pivot_R2"]),
            "checks": int(condition_check_counts["long_pivot_S2"] + condition_check_counts["short_pivot_R2"]),
        },
        "bb_lower_or_upper": {
            "true": int(condition_hit_counts["long_bb_lower"] + condition_hit_counts["short_bb_upper"]),
            "checks": int(condition_check_counts["long_bb_lower"] + condition_check_counts["short_bb_upper"]),
        },
        "sar_flip": {
            "true": int(condition_hit_counts["long_sar_flip"] + condition_hit_counts["short_sar_flip"]),
            "checks": int(condition_check_counts["long_sar_flip"] + condition_check_counts["short_sar_flip"]),
        },
        "rsi_oversold_or_overbought": {
            "true": int(condition_hit_counts["long_rsi_oversold"] + condition_hit_counts["short_rsi_overbought"]),
            "checks": int(condition_check_counts["long_rsi_oversold"] + condition_check_counts["short_rsi_overbought"]),
        },
        "rsi_extreme": {
            "true": int(condition_hit_counts["long_rsi_extreme"] + condition_hit_counts["short_rsi_extreme"]),
            "checks": int(condition_check_counts["long_rsi_extreme"] + condition_check_counts["short_rsi_extreme"]),
        },
        "atr_low": {
            "true": int(condition_hit_counts["long_atr_low"] + condition_hit_counts["short_atr_low"]),
            "checks": int(condition_check_counts["long_atr_low"] + condition_check_counts["short_atr_low"]),
        },
    }

    diagnostics = {
        "counts": dict(sorted(diag.items(), key=lambda kv: (-kv[1], kv[0]))),
        "top_15": dict(sorted(diag.items(), key=lambda kv: (-kv[1], kv[0]))[:15]),
        "signal_funnel": {
            "total_m1_candles": int(len(df)),
            "candles_inside_session_window": int(diag["bars_passed_session_filter"]),
            "candles_passing_regime_filter": int(diag["bars_passed_regime_filter"]),
            "candles_passing_lunch_block_filter": int(diag["bars_passed_lunch_filter"]),
            "total_signals_confluence_ge_1": int(diag["signals_score_ge_1"]),
            "signals_confluence_ge_2": int(diag["signals_score_ge_2"]),
            "signals_confluence_ge_3": int(diag["signals_score_ge_3"]),
            "actual_entries_taken": int(diag["entries_total"]),
            "entries_blocked_by_max_concurrent_positions": int(diag["blocked_max_open_cap"]),
            "entries_blocked_by_min_time_between_entries": int(diag["blocked_min_entry_gap"]),
            "entries_blocked_by_consecutive_loss_pause": int(diag["blocked_consecutive_loss_pause"]),
            "entries_blocked_by_breakout_detection": int(
                diag["blocked_breakout_from_open"] + diag["blocked_breakout_rolling_range"] + diag["blocked_breakout_cooldown_active"]
            ),
        },
        "filter_hit_rates": filter_hit_rates,
        "near_miss_log": near_miss_log,
        "verification": {
            "first_valid_indicator_bar": int(first_valid_idx),
            "total_in_session_bars": int(total_in_session),
            "in_session_valid_indicator_bars": int(in_session_valid),
            "in_session_valid_indicator_pct": float(in_session_valid_pct),
            "in_session_not_blocked_by_breakout_preview": int(breakout_pass_preview),
            "in_session_not_blocked_by_breakout_preview_pct": float(breakout_pass_preview_pct),
            "breakout_mode": breakout_mode,
            "rolling_window_minutes": int(rolling_window_minutes),
            "rolling_range_threshold_pips": float(rolling_range_threshold_pips),
            "cooldown_minutes": int(breakout_cooldown_minutes),
        },
    }

    # End of file: close any remainder.
    if open_positions:
        last = df.iloc[-1]
        ts = pd.Timestamp(last["time"])
        sp_last = compute_spread_pips(
            i=len(df) - 1,
            ts=pd.Timestamp(last["time"]),
            mode=spread_mode,
            avg=spread_pips,
            mn=spread_min_pips,
            mx=spread_max_pips,
        )
        bid, ask = get_bid_ask(float(last["close"]), sp_last)
        for pos in list(open_positions):
            px = bid if pos.direction == "long" else ask
            close_position(pos, ts, px, "session_close")
            open_positions.remove(pos)

    tdf = pd.DataFrame(closed_rows)
    if tdf.empty:
        delays = entry_confirmation_stats.get("confirmation_delays", [])
        session_pnls = [float(v.get("session_pnl_usd", 0.0)) for v in session_state.values()]
        avg_sess = float(np.mean(session_pnls)) if session_pnls else 0.0
        session_hours_empty = sorted({int(h) for h in df.loc[df["in_tokyo_session"], "hour_utc"].dropna().astype(int).tolist()})
        if not session_hours_empty:
            session_hours_empty = list(range(0, 24))
        return {
            "strategy_id": cfg.get("strategy_id", "tokyo_mean_reversion_v1"),
            "run_label": run_cfg["label"],
            "input_csv": run_cfg["input_csv"],
            "summary": {
                "total_trades": 0,
                "win_rate_pct": 0.0,
                "average_win_pips": 0.0,
                "average_win_usd": 0.0,
                "average_loss_pips": 0.0,
                "average_loss_usd": 0.0,
                "largest_win_pips": 0.0,
                "largest_loss_pips": 0.0,
                "largest_win_usd": 0.0,
                "largest_loss_usd": 0.0,
                "profit_factor": 0.0,
                "net_profit_usd": 0.0,
                "net_profit_pips": 0.0,
                "expectancy_per_trade_usd": 0.0,
                "return_on_starting_equity_pct": 0.0,
                "max_drawdown_usd": 0.0,
                "max_drawdown_pct": 0.0,
                "max_consecutive_wins": 0,
                "max_consecutive_losses": 0,
                "average_trade_duration_minutes": 0.0,
                "sharpe_ratio": 0.0,
                "calmar_ratio": 0.0,
                "starting_equity_usd": float(starting_equity),
                "ending_equity_usd": float(starting_equity),
            },
            "breakdown": {
                "long_performance": {"trades": 0, "win_rate_pct": 0.0, "avg_pips": 0.0, "profit_factor": 0.0, "net_usd": 0.0},
                "short_performance": {"trades": 0, "win_rate_pct": 0.0, "avg_pips": 0.0, "profit_factor": 0.0, "net_usd": 0.0},
                "day_of_week": [],
                "monthly": [],
                "average_trades_per_session": 0.0,
                "pct_sessions_with_zero_trades": 100.0,
                "exit_distribution": [],
                "avg_confluence_score_winners": 0.0,
                "avg_confluence_score_losers": 0.0,
            },
            "partial_tp_stats": {
                "times_partial_tp_hit": 0,
                "avg_pips_at_partial_tp": 0.0,
                "remaining_50pct_outcomes": {
                    "hit_full_tp": 0,
                    "hit_trailing_stop": 0,
                    "hit_breakeven_stop": 0,
                    "closed_time_decay": 0,
                    "closed_session_end": 0,
                },
            },
            "entry_confirmation_stats": {
                "signals_generated": int(entry_confirmation_stats.get("signals_generated", 0)),
                "signals_confirmed": int(entry_confirmation_stats.get("signals_confirmed", 0)),
                "signals_expired": int(entry_confirmation_stats.get("signals_expired", 0)),
                "avg_confirmation_delay_candles": float(np.mean(delays)) if delays else 0.0,
            },
            "session_pnl_distribution": {
                "sessions_positive": int(sum(1 for x in session_pnls if x > 0)),
                "sessions_negative": int(sum(1 for x in session_pnls if x < 0)),
                "sessions_zero_trades": int(sum(1 for v in session_state.values() if int(v.get("trades", 0)) == 0)),
                "avg_session_pnl": float(avg_sess),
                "best_session_pnl": float(max(session_pnls)) if session_pnls else 0.0,
                "worst_session_pnl": float(min(session_pnls)) if session_pnls else 0.0,
            },
            "day_of_week_detailed": {k: {"trades": 0, "wr": 0.0, "net_pnl": 0.0, "pf": 0.0} for k in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]},
            "hourly_detailed": {str(h): {"trades": 0, "wr": 0.0, "net_pnl": 0.0} for h in session_hours_empty},
            "regime_gate_stats": regime_gate_stats,
            "signal_quality_stats": signal_quality_stats,
            "drawdown_sizing_stats": drawdown_sizing_stats,
            "hour_preference_stats": {str(h): {"trades": 0, "avg_size_mult": 0.0, "net_pnl": 0.0} for h in session_hours_empty},
            "adx_filter_stats": {
                "trades_allowed": int(adx_filter_stats["trades_allowed"]),
                "trades_blocked_by_adx": int(adx_filter_stats["trades_blocked_by_adx"]),
                "avg_adx_at_entry": 0.0,
                "avg_adx_at_blocked": 0.0,
            },
            "combo_filter_stats": {
                "mode_used": str(combo_filter_mode),
                "trades_allowed": int(combo_filter_stats["trades_allowed"]),
                "trades_blocked_by_combo": int(combo_filter_stats["trades_blocked_by_combo"]),
                "combo_distribution": {},
            },
            "daily_range_filter_stats": daily_range_filter_stats,
            "entry_improvement_stats": {
                "signals_with_limit_placed": int(entry_improvement_stats["signals_with_limit_placed"]),
                "limits_filled": int(entry_improvement_stats["limits_filled"]),
                "limits_expired": int(entry_improvement_stats["limits_expired"]),
                "fill_rate": 0.0,
                "avg_improvement_pips": 0.0,
            },
            "filter_impact_summary": {
                "adx_filter_used": bool(adx_filter_enabled),
                "adx_trades_blocked": int(adx_filter_stats["trades_blocked_by_adx"]),
                "combo_filter_used": bool(combo_filter_enabled),
                "combo_trades_blocked": int(combo_filter_stats["trades_blocked_by_combo"]),
                "daily_range_filter_used": bool(daily_range_enabled),
                "sessions_blocked": int(daily_range_filter_stats["sessions_blocked_high_range"] + daily_range_filter_stats["sessions_blocked_low_range"]),
                "chase_filter_used": bool(chase_enabled),
                "entries_blocked_by_chase": int(diag["blocked_chase_filter"]),
                "total_v4_equivalent_signals": int(entry_confirmation_stats.get("signals_generated", 0)),
                "total_v7_entries": 0,
                "filter_pass_rate": 0.0,
            },
            "signal_strength_analysis": {
                "avg_score_winners": 0.0,
                "avg_score_losers": 0.0,
                "score_distribution": {
                    "weak_1_3": {"trades": 0, "wr": 0.0, "pf": 0.0, "net": 0.0},
                    "moderate_4_6": {"trades": 0, "wr": 0.0, "pf": 0.0, "net": 0.0},
                    "strong_7_10": {"trades": 0, "wr": 0.0, "pf": 0.0, "net": 0.0},
                },
            },
            "vs_v4_comparison": {
                "v4_trades": 0,
                "v4_pf": 0.0,
                "v4_net": 0.0,
                "v7_trades": 0,
                "v7_pf": 0.0,
                "v7_net": 0.0,
                "trade_reduction_pct": 0.0,
                "pf_improvement": 0.0,
            },
            "session_envelope_tp_stats": {
                "times_midpoint_used_as_tp": 0,
                "times_pivot_used_as_tp": 0,
                "avg_tp_distance_midpoint": 0.0,
                "avg_tp_distance_pivot": 0.0,
                "wr_midpoint_tp": 0.0,
                "wr_pivot_tp": 0.0,
            },
            "trend_regime_stats": trend_regime_stats,
            "trend_skip_calibration": {
                "threshold_used": float(trend_skip_max_move_pips),
                "selection_method": trend_skip_selection_method,
                "sessions_blocked_pct": float((100.0 * trend_regime_stats["sessions_skipped_by_trend"] / max(1, (trend_regime_stats["sessions_traded"] + trend_regime_stats["sessions_skipped_by_trend"])))),
            },
            "quality_marker_matrix": {
                "none": {"trades": 0, "wr": 0.0, "pf": 0.0},
                "rejection_only": {"trades": 0, "wr": 0.0, "pf": 0.0},
                "divergence_only": {"trades": 0, "wr": 0.0, "pf": 0.0},
                "rejection_and_divergence": {"trades": 0, "wr": 0.0, "pf": 0.0},
                "inside_ir": {"trades": 0, "wr": 0.0, "pf": 0.0},
                "outside_ir": {"trades": 0, "wr": 0.0, "pf": 0.0},
            },
            "vs_v7_comparison": {
                "v7_trades": 0,
                "v7_wr": 0.0,
                "v7_pf": 0.0,
                "v7_net": 0.0,
                "v7_maxdd": 0.0,
                "v10_trades": 0,
                "v10_wr": 0.0,
                "v10_pf": 0.0,
                "v10_net": 0.0,
                "v10_maxdd": 0.0,
            },
            "diagnostics": diagnostics,
            "equity_curve": [],
            "drawdown_curve": [],
            "trades": [],
        }

    wins = tdf[tdf["usd"] > 0.0]
    losses = tdf[tdf["usd"] < 0.0]
    gross_pos = wins["usd"].sum()
    gross_neg = abs(losses["usd"].sum())
    profit_factor = float(gross_pos / gross_neg) if gross_neg > 0 else float("inf")
    net_usd = float(tdf["usd"].sum())
    net_pips = float(tdf["pips"].sum())
    expectancy_per_trade_usd = float(net_usd / len(tdf)) if len(tdf) else 0.0
    win_rate = float((tdf["usd"] > 0).mean() * 100.0)
    avg_win_pips = float(wins["pips"].mean()) if len(wins) else 0.0
    avg_loss_pips = float(losses["pips"].mean()) if len(losses) else 0.0
    avg_win_usd = float(wins["usd"].mean()) if len(wins) else 0.0
    avg_loss_usd = float(losses["usd"].mean()) if len(losses) else 0.0
    largest_win_pips = float(tdf["pips"].max()) if len(tdf) else 0.0
    largest_loss_pips = float(tdf["pips"].min()) if len(tdf) else 0.0
    largest_win = float(tdf["usd"].max())
    largest_loss = float(tdf["usd"].min())
    ret_pct = (equity - starting_equity) / starting_equity * 100.0

    # Equity + drawdown
    eq = tdf["equity_after"].to_numpy(dtype=float)
    peak = -1e18
    dd = []
    dd_pct = []
    max_dd = 0.0
    max_dd_pct = 0.0
    for v in eq:
        peak = max(peak, float(v))
        d = max(0.0, peak - float(v))
        dp = (d / peak * 100.0) if peak > 0 else 0.0
        dd.append(d)
        dd_pct.append(dp)
        max_dd = max(max_dd, d)
        max_dd_pct = max(max_dd_pct, dp)

    equity_curve_df = pd.DataFrame(
        {
            "trade_number": np.arange(1, len(tdf) + 1, dtype=int),
            "entry_datetime": tdf["entry_datetime"],
            "exit_datetime": tdf["exit_datetime"],
            "equity_after": eq,
            "drawdown_usd": dd,
            "drawdown_pct": dd_pct,
        }
    )

    # Ratios
    returns = (tdf["usd"] / tdf["equity_before"].replace(0.0, np.nan)).fillna(0.0)
    sharpe = float((returns.mean() / returns.std(ddof=0)) * np.sqrt(len(returns))) if returns.std(ddof=0) > 0 else 0.0
    calmar = float(ret_pct / max_dd_pct) if max_dd_pct > 0 else 0.0

    # Splits
    def split_metrics(mask: pd.Series) -> dict:
        g = tdf[mask]
        if g.empty:
            return {"trades": 0, "win_rate_pct": 0.0, "avg_pips": 0.0, "profit_factor": 0.0, "net_usd": 0.0}
        gp = g[g["usd"] > 0]["usd"].sum()
        gl = abs(g[g["usd"] < 0]["usd"].sum())
        pf = float(gp / gl) if gl > 0 else float("inf")
        return {
            "trades": int(len(g)),
            "win_rate_pct": float((g["usd"] > 0).mean() * 100.0),
            "avg_pips": float(g["pips"].mean()),
            "profit_factor": float(pf),
            "net_usd": float(g["usd"].sum()),
        }

    long_metrics = split_metrics(tdf["direction"] == "long")
    short_metrics = split_metrics(tdf["direction"] == "short")
    dow = (
        tdf.assign(entry_ts=pd.to_datetime(tdf["entry_datetime"], utc=True))
        .assign(day=lambda x: x["entry_ts"].dt.day_name().str[:3])
        .groupby("day", dropna=False)
        .agg(trades=("trade_id", "count"), win_rate_pct=("usd", lambda s: float((s > 0).mean() * 100.0)), net_usd=("usd", "sum"), avg_pips=("pips", "mean"))
        .reset_index()
        .to_dict(orient="records")
    )
    monthly = (
        tdf.assign(entry_ts=pd.to_datetime(tdf["entry_datetime"], utc=True))
        .assign(month=lambda x: x["entry_ts"].dt.to_period("M").astype(str))
        .groupby("month", dropna=False)
        .agg(trades=("trade_id", "count"), win_rate_pct=("usd", lambda s: float((s > 0).mean() * 100.0)), net_usd=("usd", "sum"), net_pips=("pips", "sum"))
        .reset_index()
        .to_dict(orient="records")
    )
    session_trade_counts = [v["trades"] for v in session_state.values()]
    avg_trades_per_session = float(np.mean(session_trade_counts)) if session_trade_counts else 0.0
    sessions_with_zero = max(0, len(session_days) - sum(1 for v in session_trade_counts if v > 0))
    pct_zero_sessions = float(100.0 * sessions_with_zero / len(session_days)) if session_days else 0.0

    exit_dist = (
        tdf.groupby("exit_reason", dropna=False)
        .agg(count=("trade_id", "count"), net_usd=("usd", "sum"))
        .reset_index()
        .to_dict(orient="records")
    )
    avg_conf_win = float(tdf.loc[tdf["usd"] > 0, "confluence_score"].mean()) if (tdf["usd"] > 0).any() else 0.0
    avg_conf_loss = float(tdf.loc[tdf["usd"] <= 0, "confluence_score"].mean()) if (tdf["usd"] <= 0).any() else 0.0

    # V4 advanced execution analytics.
    tp1_hit_mask = tdf["exit_reason"].isin(
        ["tp1_then_tp2", "tp1_then_trailing_stop", "tp1_then_be_stop", "tp1_then_time_decay", "tp1_then_session_close", "tp1_then_sl"]
    )
    tp1_hit_trades = tdf[tp1_hit_mask]
    partial_tp_stats = {
        "times_partial_tp_hit": int(len(tp1_hit_trades)),
        "avg_pips_at_partial_tp": float(tp1_hit_trades["partial_tp_pips"].mean()) if len(tp1_hit_trades) else 0.0,
        "remaining_50pct_outcomes": {
            "hit_full_tp": int((tdf["exit_reason"] == "tp1_then_tp2").sum()),
            "hit_trailing_stop": int((tdf["exit_reason"] == "tp1_then_trailing_stop").sum()),
            "hit_breakeven_stop": int((tdf["exit_reason"] == "tp1_then_be_stop").sum()),
            "closed_time_decay": int((tdf["exit_reason"] == "tp1_then_time_decay").sum()),
            "closed_session_end": int((tdf["exit_reason"] == "tp1_then_session_close").sum()),
        },
    }

    delays = entry_confirmation_stats.get("confirmation_delays", [])
    entry_confirmation_block = {
        "signals_generated": int(entry_confirmation_stats.get("signals_generated", 0)),
        "signals_confirmed": int(entry_confirmation_stats.get("signals_confirmed", 0)),
        "signals_expired": int(entry_confirmation_stats.get("signals_expired", 0)),
        "avg_confirmation_delay_candles": float(np.mean(delays)) if delays else 0.0,
    }

    session_pnls = [float(v.get("session_pnl_usd", 0.0)) for v in session_state.values()]
    session_pnl_distribution = {
        "sessions_positive": int(sum(1 for x in session_pnls if x > 0)),
        "sessions_negative": int(sum(1 for x in session_pnls if x < 0)),
        "sessions_zero_trades": int(sum(1 for v in session_state.values() if int(v.get("trades", 0)) == 0)),
        "avg_session_pnl": float(np.mean(session_pnls)) if session_pnls else 0.0,
        "best_session_pnl": float(max(session_pnls)) if session_pnls else 0.0,
        "worst_session_pnl": float(min(session_pnls)) if session_pnls else 0.0,
    }

    tdf["entry_ts_utc"] = pd.to_datetime(tdf["entry_datetime"], utc=True, errors="coerce")
    tdf["entry_day_name"] = tdf["entry_ts_utc"].dt.day_name()
    day_of_week_detailed = {}
    for day_name in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
        g = tdf[tdf["entry_day_name"] == day_name]
        gp = g[g["usd"] > 0]["usd"].sum()
        gl = abs(g[g["usd"] < 0]["usd"].sum())
        pf = float(gp / gl) if gl > 0 else (float("inf") if gp > 0 else 0.0)
        day_of_week_detailed[day_name] = {
            "trades": int(len(g)),
            "wr": float((g["usd"] > 0).mean() * 100.0) if len(g) else 0.0,
            "net_pnl": float(g["usd"].sum()) if len(g) else 0.0,
            "pf": float(pf),
        }

    tdf["entry_hour_utc"] = tdf["entry_ts_utc"].dt.hour
    session_hours = sorted({int(h) for h in df.loc[df["in_tokyo_session"], "hour_utc"].dropna().astype(int).tolist()})
    if not session_hours:
        session_hours = list(range(0, 24))
    hourly_detailed = {}
    for h in session_hours:
        g = tdf[tdf["entry_hour_utc"] == h]
        hourly_detailed[str(h)] = {
            "trades": int(len(g)),
            "wr": float((g["usd"] > 0).mean() * 100.0) if len(g) else 0.0,
            "net_pnl": float(g["usd"].sum()) if len(g) else 0.0,
        }
    hour_preference_stats = {}
    for h in session_hours:
        g = tdf[tdf["entry_hour_utc"] == h]
        hour_preference_stats[str(h)] = {
            "trades": int(len(g)),
            "avg_size_mult": float(g["size_mult_hour"].mean()) if len(g) and "size_mult_hour" in g.columns else 0.0,
            "net_pnl": float(g["usd"].sum()) if len(g) else 0.0,
        }

    session_summary_table = []
    for sday in sorted(session_state.keys()):
        sst = session_state[sday]
        session_summary_table.append(
            {
                "date": str(sday),
                "signals_generated": int(sst.get("signals_generated", 0)),
                "entries_taken": int(sst.get("trades", 0)),
                "trades_won": int(sst.get("wins", 0)),
                "trades_lost": int(sst.get("losses", 0)),
                "session_pnl_usd": float(sst.get("session_pnl_usd", 0.0)),
                "entry_confluence_scores": list(sst.get("entry_confluence_scores", [])),
            }
        )

    hour_of_session_breakdown = []
    for h in range(0, 9):
        g = tdf[tdf["entry_hour_utc"] == h]
        hour_of_session_breakdown.append(
            {
                "hour_utc": int(h),
                "entries": int(len(g)),
                "wins": int((g["usd"] > 0).sum()) if len(g) else 0,
                "losses": int((g["usd"] <= 0).sum()) if len(g) else 0,
                "win_rate_pct": float((g["usd"] > 0).mean() * 100.0) if len(g) else 0.0,
                "avg_pips": float(g["pips"].mean()) if len(g) else 0.0,
                "net_pnl_usd": float(g["usd"].sum()) if len(g) else 0.0,
                "avg_confluence_score": float(g["confluence_score"].mean()) if len(g) else 0.0,
            }
        )
    diagnostics["session_summary_table"] = session_summary_table
    diagnostics["hour_of_session_breakdown"] = hour_of_session_breakdown

    combo_distribution = {}
    for combo, g in tdf.groupby("confluence_combo", dropna=False):
        gp = g[g["usd"] > 0]["usd"].sum()
        gl = abs(g[g["usd"] < 0]["usd"].sum())
        combo_distribution[str(combo)] = {
            "trades": int(len(g)),
            "wr": float((g["usd"] > 0).mean() * 100.0) if len(g) else 0.0,
            "pf": float((gp / gl) if gl > 0 else (float("inf") if gp > 0 else 0.0)),
            "net": float(g["usd"].sum()),
        }
    combo_filter_stats["combo_distribution"] = combo_distribution
    adx_stats_block = {
        "trades_allowed": int(adx_filter_stats["trades_allowed"]),
        "trades_blocked_by_adx": int(adx_filter_stats["trades_blocked_by_adx"]),
        "avg_adx_at_entry": float(np.mean(adx_filter_stats["adx_allowed_values"])) if adx_filter_stats["adx_allowed_values"] else 0.0,
        "avg_adx_at_blocked": float(np.mean(adx_filter_stats["adx_blocked_values"])) if adx_filter_stats["adx_blocked_values"] else 0.0,
    }
    entry_imp_block = {
        "signals_with_limit_placed": int(entry_improvement_stats["signals_with_limit_placed"]),
        "limits_filled": int(entry_improvement_stats["limits_filled"]),
        "limits_expired": int(entry_improvement_stats["limits_expired"]),
        "fill_rate": float(
            100.0 * entry_improvement_stats["limits_filled"] / entry_improvement_stats["signals_with_limit_placed"]
        ) if entry_improvement_stats["signals_with_limit_placed"] else 0.0,
        "avg_improvement_pips": float(np.mean(entry_improvement_stats["improvement_values_pips"]))
        if entry_improvement_stats["improvement_values_pips"] else 0.0,
    }
    filter_impact_summary = {
        "adx_filter_used": bool(adx_filter_enabled),
        "adx_trades_blocked": int(adx_filter_stats["trades_blocked_by_adx"]),
        "combo_filter_used": bool(combo_filter_enabled),
        "combo_trades_blocked": int(combo_filter_stats["trades_blocked_by_combo"]),
        "daily_range_filter_used": bool(daily_range_enabled),
        "sessions_blocked": int(daily_range_filter_stats["sessions_blocked_high_range"] + daily_range_filter_stats["sessions_blocked_low_range"]),
        "chase_filter_used": bool(chase_enabled),
        "entries_blocked_by_chase": int(diag["blocked_chase_filter"]),
        "total_v4_equivalent_signals": int(diag["signals_pre_v7_filters"]),
        "total_v7_entries": int(diag["entries_total"]),
        "filter_pass_rate": float((100.0 * int(diag["entries_total"]) / int(diag["signals_pre_v7_filters"])) if int(diag["signals_pre_v7_filters"]) > 0 else 0.0),
    }

    def split_pf_net(g: pd.DataFrame) -> dict:
        gp = g[g["usd"] > 0]["usd"].sum()
        gl = abs(g[g["usd"] < 0]["usd"].sum())
        pf = float(gp / gl) if gl > 0 else (float("inf") if gp > 0 else 0.0)
        return {
            "trades": int(len(g)),
            "wr": float((g["usd"] > 0).mean() * 100.0) if len(g) else 0.0,
            "pf": float(pf),
            "net": float(g["usd"].sum()) if len(g) else 0.0,
        }

    weak = tdf[(tdf["signal_strength_score"] >= 1) & (tdf["signal_strength_score"] <= 3)] if "signal_strength_score" in tdf.columns else tdf.iloc[0:0]
    mod = tdf[(tdf["signal_strength_score"] >= 4) & (tdf["signal_strength_score"] <= 6)] if "signal_strength_score" in tdf.columns else tdf.iloc[0:0]
    strong = tdf[tdf["signal_strength_score"] >= 7] if "signal_strength_score" in tdf.columns else tdf.iloc[0:0]
    signal_strength_analysis = {
        "avg_score_winners": float(tdf.loc[tdf["usd"] > 0, "signal_strength_score"].mean()) if "signal_strength_score" in tdf.columns and (tdf["usd"] > 0).any() else 0.0,
        "avg_score_losers": float(tdf.loc[tdf["usd"] <= 0, "signal_strength_score"].mean()) if "signal_strength_score" in tdf.columns and (tdf["usd"] <= 0).any() else 0.0,
        "score_distribution": {
            "weak_1_3": split_pf_net(weak),
            "moderate_4_6": split_pf_net(mod),
            "strong_7_10": split_pf_net(strong),
        },
    }
    ss_filter_block = {
        "min_score_used": int(ss_filter_min_score),
        "trades_above_threshold": int(ss_filter_stats["trades_above_threshold"]),
        "trades_below_threshold_skipped": int(ss_filter_stats["trades_below_threshold_skipped"]),
        "pf_above": float(split_pf_net(tdf[tdf["signal_strength_score"] >= ss_filter_min_score])["pf"]) if "signal_strength_score" in tdf.columns and ss_filter_min_score > 0 else float(split_pf_net(tdf)["pf"]),
        "pf_below": float(split_pf_net(tdf[tdf["signal_strength_score"] < ss_filter_min_score])["pf"]) if "signal_strength_score" in tdf.columns and ss_filter_min_score > 0 else 0.0,
    }

    if "entry_delay_type" in tdf.columns:
        g_im = tdf[tdf["entry_delay_type"] != "delayed"]
        g_de = tdf[tdf["entry_delay_type"] == "delayed"]
        momentum_check_stats_block = {
            "entries_immediate": int(momentum_check_stats["entries_immediate"]),
            "entries_delayed_then_filled": int(momentum_check_stats["entries_delayed_then_filled"]),
            "entries_delayed_then_expired": int(momentum_check_stats["entries_delayed_then_expired"]),
            "pf_immediate": float(split_pf_net(g_im)["pf"]),
            "pf_delayed": float(split_pf_net(g_de)["pf"]),
        }
    else:
        momentum_check_stats_block = {
            "entries_immediate": 0,
            "entries_delayed_then_filled": 0,
            "entries_delayed_then_expired": 0,
            "pf_immediate": 0.0,
            "pf_delayed": 0.0,
        }

    regime_switch_stats["pnl_traded_sessions"] = float(sum(float(v.get("session_pnl_usd", 0.0)) for v in session_state.values() if bool(v.get("regime_switch_allowed", True))))
    regime_switch_stats["estimated_pnl_skipped_sessions"] = float(sum(float(v.get("session_pnl_usd", 0.0)) for v in session_state.values() if not bool(v.get("regime_switch_allowed", True))))
    early_exit_stats_block = {
        "early_exits_triggered": int(early_exit_stats["early_exits_triggered"]),
        "avg_loss_saved_pips": float(np.mean(early_exit_stats["loss_saved_pips"])) if early_exit_stats["loss_saved_pips"] else 0.0,
    }
    mfe_mae_tracking = {
        "all_trades_avg_mfe": float(tdf["mfe_pips"].mean()) if "mfe_pips" in tdf.columns else 0.0,
        "all_trades_avg_mae": float(tdf["mae_pips"].mean()) if "mae_pips" in tdf.columns else 0.0,
        "winners_avg_mfe": float(tdf.loc[tdf["usd"] > 0, "mfe_pips"].mean()) if "mfe_pips" in tdf.columns and (tdf["usd"] > 0).any() else 0.0,
        "winners_avg_mae": float(tdf.loc[tdf["usd"] > 0, "mae_pips"].mean()) if "mae_pips" in tdf.columns and (tdf["usd"] > 0).any() else 0.0,
        "losers_avg_mfe": float(tdf.loc[tdf["usd"] <= 0, "mfe_pips"].mean()) if "mfe_pips" in tdf.columns and (tdf["usd"] <= 0).any() else 0.0,
        "losers_avg_mae": float(tdf.loc[tdf["usd"] <= 0, "mae_pips"].mean()) if "mae_pips" in tdf.columns and (tdf["usd"] <= 0).any() else 0.0,
    }

    v4_comp = {
        "v4_trades": 0,
        "v4_pf": 0.0,
        "v4_net": 0.0,
        "v7_trades": int(len(tdf)),
        "v7_pf": float(profit_factor),
        "v7_net": float(net_usd),
        "trade_reduction_pct": 0.0,
        "pf_improvement": 0.0,
    }
    v4_path = Path(
        f"/Users/codygnon/Documents/usdjpy_assistant/research_out/"
        f"tokyo_meanrev_v4_{str(run_cfg['label']).lower()}_report.json"
    )
    if v4_path.exists():
        try:
            v4r = json.loads(v4_path.read_text(encoding="utf-8"))
            v4t = int(v4r.get("summary", {}).get("total_trades", 0))
            v4pf = float(v4r.get("summary", {}).get("profit_factor", 0.0))
            v4net = float(v4r.get("summary", {}).get("net_profit_usd", 0.0))
            v4_comp = {
                "v4_trades": v4t,
                "v4_pf": v4pf,
                "v4_net": v4net,
                "v7_trades": int(len(tdf)),
                "v7_pf": float(profit_factor),
                "v7_net": float(net_usd),
                "trade_reduction_pct": float((100.0 * (v4t - len(tdf)) / v4t) if v4t > 0 else 0.0),
                "pf_improvement": float(profit_factor - v4pf),
            }
        except Exception:
            pass

    # V10 overlays analytics.
    with_rej = tdf[tdf.get("rejection_confirmed", False) == True] if "rejection_confirmed" in tdf.columns else tdf.iloc[0:0]
    without_rej = tdf[tdf.get("rejection_confirmed", False) != True] if "rejection_confirmed" in tdf.columns else tdf
    tp_mid = tdf[tdf.get("tp_source", "") == "midpoint"] if "tp_source" in tdf.columns else tdf.iloc[0:0]
    tp_piv = tdf[tdf.get("tp_source", "").astype(str).str.startswith("pivot")] if "tp_source" in tdf.columns else tdf.iloc[0:0]
    marker_none = tdf[(tdf.get("rejection_confirmed", False) != True) & (tdf.get("divergence_present", False) != True)] if {"rejection_confirmed", "divergence_present"}.issubset(tdf.columns) else tdf.iloc[0:0]
    marker_rej_only = tdf[(tdf.get("rejection_confirmed", False) == True) & (tdf.get("divergence_present", False) != True)] if {"rejection_confirmed", "divergence_present"}.issubset(tdf.columns) else tdf.iloc[0:0]
    marker_div_only = tdf[(tdf.get("rejection_confirmed", False) != True) & (tdf.get("divergence_present", False) == True)] if {"rejection_confirmed", "divergence_present"}.issubset(tdf.columns) else tdf.iloc[0:0]
    marker_both = tdf[(tdf.get("rejection_confirmed", False) == True) & (tdf.get("divergence_present", False) == True)] if {"rejection_confirmed", "divergence_present"}.issubset(tdf.columns) else tdf.iloc[0:0]
    inside_ir_df = tdf[tdf.get("inside_ir", False) == True] if "inside_ir" in tdf.columns else tdf.iloc[0:0]
    outside_ir_df = tdf[tdf.get("inside_ir", False) != True] if "inside_ir" in tdf.columns else tdf

    def wr_of(g: pd.DataFrame) -> float:
        return float((g["usd"] > 0).mean() * 100.0) if len(g) else 0.0

    session_envelope_tp_stats = {
        "times_midpoint_used_as_tp": int(len(tp_mid)),
        "times_pivot_used_as_tp": int(len(tp_piv)),
        "avg_tp_distance_midpoint": float(tdf["distance_to_midpoint_pips"].mean()) if "distance_to_midpoint_pips" in tdf.columns and len(tdf) else 0.0,
        "avg_tp_distance_pivot": float(tdf["distance_to_pivot_pips"].mean()) if "distance_to_pivot_pips" in tdf.columns and len(tdf) else 0.0,
        "wr_midpoint_tp": wr_of(tp_mid),
        "wr_pivot_tp": wr_of(tp_piv),
    }

    if trend_skip_enabled:
        traded_vals = [float(v.get("session_pnl_usd", 0.0)) for v in session_state.values() if bool(v.get("trend_skip_allowed", True))]
        avg_traded = float(np.mean(traded_vals)) if traded_vals else 0.0
        trend_regime_stats["estimated_pnl_skipped"] = float(avg_traded * trend_regime_stats["sessions_skipped_by_trend"])
    else:
        trend_regime_stats["sessions_traded"] = len(session_state)
        trend_regime_stats["sessions_skipped_by_trend"] = 0
        trend_regime_stats["estimated_pnl_skipped"] = 0.0

    quality_marker_matrix = {
        "none": split_pf_net(marker_none),
        "rejection_only": split_pf_net(marker_rej_only),
        "divergence_only": split_pf_net(marker_div_only),
        "rejection_and_divergence": split_pf_net(marker_both),
        "inside_ir": split_pf_net(inside_ir_df),
        "outside_ir": split_pf_net(outside_ir_df),
    }

    v7_comp = {
        "v7_trades": 0,
        "v7_wr": 0.0,
        "v7_pf": 0.0,
        "v7_net": 0.0,
        "v7_maxdd": 0.0,
        "v10_trades": int(len(tdf)),
        "v10_wr": float(win_rate),
        "v10_pf": float(profit_factor),
        "v10_net": float(net_usd),
        "v10_maxdd": float(max_dd),
    }
    v7_path = Path(
        f"/Users/codygnon/Documents/usdjpy_assistant/research_out/"
        f"tokyo_meanrev_v7_{str(run_cfg['label']).lower()}_report.json"
    )
    if v7_path.exists():
        try:
            v7r = json.loads(v7_path.read_text(encoding="utf-8"))
            v7s = v7r.get("summary", {})
            v7_comp.update(
                {
                    "v7_trades": int(v7s.get("total_trades", 0)),
                    "v7_wr": float(v7s.get("win_rate_pct", 0.0)),
                    "v7_pf": float(v7s.get("profit_factor", 0.0)),
                    "v7_net": float(v7s.get("net_profit_usd", 0.0)),
                    "v7_maxdd": float(v7s.get("max_drawdown_usd", 0.0)),
                }
            )
        except Exception:
            pass

    # Required verification print: first 3 trades with signal/confirmation/levels.
    print("First 3 trade entries (signal bar, confirmation delay, partial TP, initial SL, full TP):")
    for _, tr in tdf.head(3).iterrows():
        print(
            f"  signal={tr.get('signal_datetime')} entry={tr.get('entry_datetime')} "
            f"delay={int(tr.get('confirmation_delay_candles', 0))} "
            f"partial_tp_pips={float(tr.get('partial_tp_pips', 0.0)):.2f} "
            f"initial_sl_pips={float(tr.get('initial_sl_pips', 0.0)):.2f} "
            f"full_tp_price={float(tr.get('tp_price', 0.0)):.5f}"
        )
    print(
        "Signal confirmation totals: "
        f"generated={entry_confirmation_block['signals_generated']} "
        f"confirmed={entry_confirmation_block['signals_confirmed']} "
        f"expired={entry_confirmation_block['signals_expired']}"
    )

    tdf_out = tdf.drop(columns=["entry_ts_utc", "entry_day_name", "entry_hour_utc"], errors="ignore")

    report = {
        "strategy_id": cfg.get("strategy_id", "tokyo_mean_reversion_v1"),
        "run_label": run_cfg["label"],
        "input_csv": run_cfg["input_csv"],
        "summary": {
            "total_trades": int(len(tdf)),
            "win_rate_pct": float(win_rate),
            "average_win_pips": float(avg_win_pips),
            "average_win_usd": float(avg_win_usd),
            "average_loss_pips": float(avg_loss_pips),
            "average_loss_usd": float(avg_loss_usd),
            "largest_win_pips": float(largest_win_pips),
            "largest_loss_pips": float(largest_loss_pips),
            "largest_win_usd": float(largest_win),
            "largest_loss_usd": float(largest_loss),
            "profit_factor": float(profit_factor),
            "net_profit_usd": float(net_usd),
            "net_profit_pips": float(net_pips),
            "expectancy_per_trade_usd": float(expectancy_per_trade_usd),
            "return_on_starting_equity_pct": float(ret_pct),
            "max_drawdown_usd": float(max_dd),
            "max_drawdown_pct": float(max_dd_pct),
            "max_consecutive_wins": int(max_consec_wins),
            "max_consecutive_losses": int(max_consec_losses_seen),
            "average_trade_duration_minutes": float(tdf["duration_minutes"].mean()),
            "sharpe_ratio": float(sharpe),
            "calmar_ratio": float(calmar),
            "starting_equity_usd": float(starting_equity),
            "ending_equity_usd": float(equity),
        },
        "breakdown": {
            "long_performance": long_metrics,
            "short_performance": short_metrics,
            "day_of_week": dow,
            "monthly": monthly,
            "average_trades_per_session": float(avg_trades_per_session),
            "pct_sessions_with_zero_trades": float(pct_zero_sessions),
            "exit_distribution": exit_dist,
            "avg_confluence_score_winners": float(avg_conf_win),
            "avg_confluence_score_losers": float(avg_conf_loss),
        },
        "partial_tp_stats": partial_tp_stats,
        "entry_confirmation_stats": entry_confirmation_block,
        "session_pnl_distribution": session_pnl_distribution,
        "day_of_week_detailed": day_of_week_detailed,
        "hourly_detailed": hourly_detailed,
        "regime_gate_stats": regime_gate_stats,
        "signal_quality_stats": signal_quality_stats,
        "drawdown_sizing_stats": drawdown_sizing_stats,
        "hour_preference_stats": hour_preference_stats,
        "adx_filter_stats": adx_stats_block,
        "combo_filter_stats": {
            "mode_used": str(combo_filter_mode),
            "trades_allowed": int(combo_filter_stats["trades_allowed"]),
            "trades_blocked_by_combo": int(combo_filter_stats["trades_blocked_by_combo"]),
            "combo_distribution": combo_distribution,
        },
        "daily_range_filter_stats": daily_range_filter_stats,
        "entry_improvement_stats": entry_imp_block,
        "filter_impact_summary": filter_impact_summary,
        "signal_strength_analysis": signal_strength_analysis,
        "signal_strength_filter_stats": ss_filter_block,
        "regime_switch_stats": regime_switch_stats,
        "momentum_check_stats": momentum_check_stats_block,
        "early_exit_stats": early_exit_stats_block,
        "mfe_mae_tracking": mfe_mae_tracking,
        "vs_v4_comparison": v4_comp,
        "session_envelope_tp_stats": session_envelope_tp_stats,
        "trend_regime_stats": trend_regime_stats,
        "trend_skip_calibration": {
            "threshold_used": float(trend_skip_max_move_pips),
            "selection_method": trend_skip_selection_method,
            "sessions_blocked_pct": float((100.0 * trend_regime_stats["sessions_skipped_by_trend"] / max(1, (trend_regime_stats["sessions_traded"] + trend_regime_stats["sessions_skipped_by_trend"])))),
        },
        "quality_marker_matrix": quality_marker_matrix,
        "vs_v7_comparison": v7_comp,
        "diagnostics": diagnostics,
        "equity_curve": equity_curve_df.to_dict(orient="records"),
        "drawdown_curve": equity_curve_df[["trade_number", "drawdown_usd", "drawdown_pct"]].to_dict(orient="records"),
        "trades": tdf_out.to_dict(orient="records"),
    }
    return report


def write_outputs(report: dict, run_cfg: dict) -> None:
    Path(run_cfg["output_json"]).write_text(json.dumps(report, indent=2), encoding="utf-8")
    tdf = pd.DataFrame(report["trades"])
    edf = pd.DataFrame(report["equity_curve"])
    tdf.to_csv(run_cfg["output_trades_csv"], index=False)
    edf.to_csv(run_cfg["output_equity_csv"], index=False)


def main() -> int:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config: {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    runs = list(cfg.get("run_sequence", []))
    if not runs:
        raise RuntimeError("Config missing run_sequence")

    breakout_analysis_path = cfg.get("breakout_analysis_output")
    if breakout_analysis_path:
        run_for_analysis = runs[0]
        dfa = add_indicators(load_m1(run_for_analysis["input_csv"]), cfg)
        dfa["time_jst"] = dfa["time"].dt.tz_convert(TOKYO_TZ)
        dfa["session_day_jst"] = dfa["time_jst"].dt.date.astype(str)
        dfa["weekday_jst"] = dfa["time_jst"].dt.dayofweek
        dfa["hour_jst"] = dfa["time_jst"].dt.hour
        dfa["in_tokyo_session"] = (dfa["hour_jst"] >= 0) & (dfa["hour_jst"] < 9)
        threshold = float(cfg["trade_management"]["disable_entries_if_move_from_tokyo_open_range_exceeds_pips"])
        breakout_analysis = build_from_open_breakout_analysis(dfa, threshold_pips=threshold)
        Path(str(breakout_analysis_path)).write_text(json.dumps(breakout_analysis, indent=2), encoding="utf-8")
        s = breakout_analysis["summary"]
        print(
            "Breakout analysis (from_open) "
            f"sessions={s['sessions']} breached={s['breached_sessions']} "
            f"breach_pct={s['breach_pct']:.2f}% median_minutes_to_breach={s['median_minutes_to_breach']}"
        )

    run_summaries: list[dict] = []
    for r in runs:
        report = run_one(cfg, r)
        ccf = cfg.get("confluence_combo_filter", {})
        if (
            bool(ccf.get("enabled", False))
            and str(ccf.get("mode", "allowlist")).strip().lower() == "allowlist"
            and str(r.get("label", "")).lower() == "250k"
            and int(report.get("summary", {}).get("total_trades", 0)) < 30
        ):
            print(
                "Confluence combo allowlist produced <30 trades on 250k; "
                "switching to blocklist mode and re-running 250k."
            )
            cfg.setdefault("confluence_combo_filter", {})["mode"] = "blocklist"
            report = run_one(cfg, r)
        write_outputs(report, r)
        s = report["summary"]
        run_summaries.append({"label": str(r["label"]).lower(), "summary": s})
        print(
            f"[{r['label']}] trades={s['total_trades']} wr={s['win_rate_pct']:.2f}% "
            f"net_usd={s['net_profit_usd']:.2f} pf={s['profit_factor']:.3f} "
            f"maxdd={s['max_drawdown_usd']:.2f} -> {r['output_json']}"
        )
        if str(r.get("label", "")).lower() == "50k" and int(s.get("total_trades", 0)) < 20:
            print("50k trades < 20; funnel diagnostics:")
            fkeys = [
                "bars_passed_session_filter",
                "bars_passed_indicator_check",
                "bars_passed_breakout_check",
                "bars_passed_regime_filter",
                "bars_passed_atr_filter",
                "bars_reached_signal_scoring",
                "bars_with_confluence_met",
                "bars_with_entry_triggered",
            ]
            dc = report.get("diagnostics", {}).get("counts", {})
            for k in fkeys:
                print(f"  {k}: {dc.get(k, 0)}")
    if run_summaries:
        pfs = [float(x["summary"].get("profit_factor", 0.0)) for x in run_summaries]
        pf_std = float(np.std(pfs, ddof=0)) if pfs else 0.0
        print("\nSCALING CHECK:")
        print("| Dataset | Trades | WR | PF | Net | MaxDD | Net/MaxDD |")
        for x in run_summaries:
            s = x["summary"]
            dd = float(s.get("max_drawdown_usd", 0.0))
            ndd = float(s.get("net_profit_usd", 0.0)) / dd if dd > 0 else 0.0
            print(
                f"| {x['label']} | {int(s.get('total_trades',0))} | {float(s.get('win_rate_pct',0.0)):.2f}% | "
                f"{float(s.get('profit_factor',0.0)):.3f} | {float(s.get('net_profit_usd',0.0)):.2f} | "
                f"{dd:.2f} | {ndd:.3f} |"
            )
        print(f"PF StdDev across datasets: {pf_std:.4f} ({'consistent' if pf_std < 0.2 else 'inconsistent'})")

        s1000 = next((x["summary"] for x in run_summaries if x["label"] == "1000k"), None)
        if s1000 is not None:
            t = int(s1000.get("total_trades", 0))
            pf = float(s1000.get("profit_factor", 0.0))
            print("\nV7 VERDICT:")
            print(f"Trades on 1000k: {t}")
            print(f"PF on 1000k: {pf:.3f}")
            if pf > 1.25 and t > 80:
                print("STRATEGY IS VIABLE — proceed to V8 optimization")
            elif pf > 1.25 and t < 80:
                print("STRATEGY HAS EDGE but needs more signal generation")
            elif pf < 1.25 and t > 80:
                print("STRATEGY NEEDS better signal quality")
            else:
                print("FUNDAMENTAL RETHINK NEEDED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
