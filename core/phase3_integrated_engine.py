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
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from core.signal_engine import drop_incomplete_last_bar

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


def load_phase3_sizing_config(config_path: Optional[Path] = None) -> dict[str, Any]:
    """Load Phase 3 sizing config from JSON. Returns {} if file missing or invalid."""
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "research_out" / "phase3_integrated_sizing_config.json"
    if not config_path.exists():
        return {}
    try:
        with open(config_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Session constants (single source of truth; aligned with backtest)
# ---------------------------------------------------------------------------
# London V2: 08:00-12:00 UTC = 3:00-7:00 AM EST
#   ARB: 08:00-09:30 UTC (3:00-4:30 AM EST)
#   LMP: 09:30-12:00 UTC (4:30-7:00 AM EST)
# V44 NY: 13:00-16:00 UTC with 5-min start delay = 8:05-11:00 AM EST
# Tokyo V14: 16:00-22:00 UTC = 11:00 AM-5:00 PM EST
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

def classify_session(now_utc: datetime) -> Optional[str]:
    """Return 'tokyo', 'london', 'ny', or None."""
    hour = now_utc.hour
    minute = now_utc.minute
    weekday = now_utc.weekday()
    hour_frac = hour + minute / 60.0

    if TOKYO_START_UTC <= hour < TOKYO_END_UTC and weekday in TOKYO_ALLOWED_DAYS:
        return "tokyo"
    if LONDON_START_UTC <= hour_frac < LONDON_END_UTC and weekday in LONDON_ALLOWED_DAYS:
        return "london"
    if NY_START_UTC <= hour_frac < NY_END_UTC and weekday in NY_ALLOWED_DAYS:
        return "ny"
    return None


# ===================================================================
#  Fibonacci Pivots
# ===================================================================

def compute_daily_fib_pivots(prev_day_high: float, prev_day_low: float, prev_day_close: float) -> dict:
    """Compute daily Fibonacci pivot levels {P, R1, R2, R3, S1, S2, S3}."""
    p = (prev_day_high + prev_day_low + prev_day_close) / 3.0
    r = prev_day_high - prev_day_low
    return {
        "P": p,
        "R1": p + 0.382 * r,
        "R2": p + 0.618 * r,
        "R3": p + 1.000 * r,
        "S1": p - 0.382 * r,
        "S2": p - 0.618 * r,
        "S3": p - 1.000 * r,
    }


# ===================================================================
#  BB Width Regime
# ===================================================================

def compute_bb_width_regime(m5_df: pd.DataFrame) -> str:
    """Return 'ranging' or 'trending' based on BB width percentile."""
    close = m5_df["close"].astype(float)
    sma = close.rolling(BB_PERIOD).mean()
    std = close.rolling(BB_PERIOD).std()
    upper = sma + BB_STD * std
    lower = sma - BB_STD * std
    width = (upper - lower) / sma
    if len(width.dropna()) < BB_WIDTH_LOOKBACK:
        return "ranging"  # default safe
    recent = width.iloc[-BB_WIDTH_LOOKBACK:]
    cutoff = recent.quantile(BB_WIDTH_RANGING_PCT)
    current = width.iloc[-1]
    if pd.isna(current) or pd.isna(cutoff):
        return "ranging"
    return "trending" if current >= cutoff else "ranging"


def _compute_bb(m5_df: pd.DataFrame) -> tuple[float, float, float]:
    """Return (bb_upper, bb_middle, bb_lower) for latest M5 bar."""
    close = m5_df["close"].astype(float)
    sma = close.rolling(BB_PERIOD).mean()
    std = close.rolling(BB_PERIOD).std()
    bb_upper = float(sma.iloc[-1] + BB_STD * std.iloc[-1])
    bb_lower = float(sma.iloc[-1] - BB_STD * std.iloc[-1])
    bb_mid = float(sma.iloc[-1])
    return bb_upper, bb_mid, bb_lower


# ===================================================================
#  RSI
# ===================================================================

def _compute_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Compute RSI on a price series."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


# ===================================================================
#  ATR
# ===================================================================

def _compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """True Range -> EWM ATR."""
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


# ===================================================================
#  ADX
# ===================================================================

def _compute_adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> float:
    """Return latest ADX value."""
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)
    plus_dm = (high - prev_high).clip(lower=0)
    minus_dm = (prev_low - low).clip(lower=0)
    # Zero out when the other is larger
    plus_dm = plus_dm.where(plus_dm > minus_dm, 0.0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0.0)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100.0 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
    minus_di = 100.0 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)
    dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    adx = dx.ewm(span=period, adjust=False).mean()
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
    pivots: dict,
    bb_upper: float,
    bb_lower: float,
    sar_bullish_flip: bool,
    sar_bearish_flip: bool,
    rsi: float,
    pip_size: float,
) -> tuple[int, str]:
    """
    Score confluence for V14 entry.

    Components (labelled A-D):
      A = price in pivot zone (close <= S1+tol for long, close >= R1-tol for short)
      B = BB touch/penetration (close <= bb_lower for long, close >= bb_upper for short)
      C = SAR flip (bullish for long, bearish for short)
      D = RSI soft filter (< 35 for long, > 65 for short)

    Returns (score, combo_string) e.g. (3, "ABC").
    """
    score = 0
    combo = ""
    tol = ZONE_TOLERANCE_PIPS * pip_size

    if side == "buy":
        # A: pivot zone
        if close <= pivots["S1"] + tol:
            score += 1
            combo += "A"
        # B: BB lower touch
        if close <= bb_lower:
            score += 1
            combo += "B"
        # C: SAR bullish flip
        if sar_bullish_flip:
            score += 1
            combo += "C"
        # D: RSI
        if rsi < RSI_LONG_ENTRY:
            score += 1
            combo += "D"
    else:  # sell
        if close >= pivots["R1"] - tol:
            score += 1
            combo += "A"
        if close >= bb_upper:
            score += 1
            combo += "B"
        if sar_bearish_flip:
            score += 1
            combo += "C"
        if rsi > RSI_SHORT_ENTRY:
            score += 1
            combo += "D"

    return score, combo


# ===================================================================
#  SL / TP computation
# ===================================================================

def compute_v14_sl(side: str, entry_price: float, pivots: dict, pip_size: float) -> float:
    """Compute SL price based on pivot + buffer, clamped to min/max."""
    buffer = SL_BUFFER_PIPS * pip_size
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
    sl_dist = max(SL_MIN_PIPS, min(SL_MAX_PIPS, sl_dist))
    if side == "buy":
        return entry_price - sl_dist * pip_size
    else:
        return entry_price + sl_dist * pip_size


def compute_v14_tp1(side: str, entry_price: float, atr_value: float, pip_size: float) -> float:
    """Compute TP1 = ATR * 0.5, clamped to 6-12 pips."""
    tp_dist = atr_value * TP1_ATR_MULT
    tp_pips = tp_dist / pip_size
    tp_pips = max(TP1_MIN_PIPS, min(TP1_MAX_PIPS, tp_pips))
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
    max_margin_units = (equity * leverage) / current_price
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
    return {
        "day_start": d0,
        "london_open": d0.replace(hour=LONDON_START_UTC, minute=0),
        "london_arb_end": d0.replace(hour=LONDON_START_UTC, minute=0) + pd.Timedelta(minutes=90),
        "lmp_impulse_end": d0.replace(hour=LONDON_START_UTC, minute=0) + pd.Timedelta(minutes=LDN_LMP_IMPULSE_MINUTES),
        "london_end": d0.replace(hour=LONDON_END_UTC, minute=0),
        "ny_open": d0.replace(hour=NY_START_UTC, minute=0),
        "ny_end": d0.replace(hour=NY_END_UTC, minute=0),
    }


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


def compute_v44_m5_slope(m5_df: pd.DataFrame, slope_bars: int) -> float:
    if m5_df is None or len(m5_df) < max(V44_M5_EMA_FAST, V44_M5_EMA_SLOW) + slope_bars + 2:
        return 0.0
    ema_fast = _compute_ema(m5_df["close"].astype(float), V44_M5_EMA_FAST)
    now = float(ema_fast.iloc[-1])
    prev = float(ema_fast.iloc[-1 - slope_bars])
    return (now - prev) / PIP_SIZE / max(1, slope_bars)


def classify_v44_strength(slope: float, is_london: bool) -> str:
    threshold = V44_LONDON_STRONG_SLOPE if is_london else V44_STRONG_SLOPE
    return "strong" if abs(slope) >= threshold else "normal"


def compute_v44_atr_pct_filter(m5_df: pd.DataFrame) -> bool:
    if m5_df is None or len(m5_df) < V44_ATR_PCT_LOOKBACK:
        return True
    atr = _compute_atr(m5_df, ATR_PERIOD).dropna()
    if len(atr) < 20:
        return True
    current = float(atr.iloc[-1])
    look = atr.iloc[-V44_ATR_PCT_LOOKBACK:]
    cutoff = float(np.quantile(look, V44_ATR_PCT_CAP))
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
    min_body_pips: float = V44_MIN_BODY_PIPS,
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
    if not compute_v44_atr_pct_filter(m5_df):
        return None, "normal", "v44: ATR percentile block"
    if m5_df is None or len(m5_df) < max(m5_ema_fast_period, m5_ema_slow_period) + 4:
        return None, "normal", "v44: insufficient M5"

    close = m5_df["close"].astype(float)
    open_ = m5_df["open"].astype(float)
    ema_fast = _compute_ema(close, m5_ema_fast_period)
    ema_slow = _compute_ema(close, m5_ema_slow_period)
    body_pips = abs(float(close.iloc[-1]) - float(open_.iloc[-1])) / pip_size
    slope = compute_v44_m5_slope(m5_df, slope_bars)
    strength = "strong" if abs(slope) >= strong_slope_threshold else "normal"
    bullish_bar = float(close.iloc[-1]) > float(open_.iloc[-1]) and body_pips >= min_body_pips
    bearish_bar = float(close.iloc[-1]) < float(open_.iloc[-1]) and body_pips >= min_body_pips

    if trend == "up" and float(ema_fast.iloc[-1]) > float(ema_slow.iloc[-1]) and bullish_bar and slope > 0:
        return "buy", strength, "v44: H1 up + M5 strong bullish momentum"
    if trend == "down" and float(ema_fast.iloc[-1]) < float(ema_slow.iloc[-1]) and bearish_bar and slope < 0:
        return "sell", strength, "v44: H1 down + M5 strong bearish momentum"
    return None, strength, "v44: directional conditions not met"


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
    ldn_max_trades_per_day_total = int(ldn_config.get("max_trades_per_day_total", LDN_MAX_TRADES_PER_DAY_TOTAL))
    ldn_range_min_pips = float(ldn_config.get("arb_range_min_pips", LDN_ARB_RANGE_MIN_PIPS))
    ldn_range_max_pips = float(ldn_config.get("arb_range_max_pips", LDN_ARB_RANGE_MAX_PIPS))
    ldn_lor_min_pips = float(ldn_config.get("lor_range_min_pips", LDN_D_LOR_MIN_PIPS))
    ldn_lor_max_pips = float(ldn_config.get("lor_range_max_pips", LDN_D_LOR_MAX_PIPS))
    ldn_d_max_trades = int(ldn_config.get("d_max_trades", LDN_D_MAX_TRADES))
    ldn_leverage = float(ldn_config.get("leverage", 33.0))
    ldn_max_margin_frac = float(ldn_config.get("max_margin_usage_fraction_per_trade", 0.5))
    ldn_active_days = ldn_config.get("active_days_utc", ["Tuesday", "Wednesday"])
    ldn_active_days = set(str(d) for d in ldn_active_days) if isinstance(ldn_active_days, (list, tuple, set)) else {"Tuesday", "Wednesday"}

    if (tick.ask - tick.bid) / pip > LDN_MAX_SPREAD_PIPS:
        no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason="london_v2: spread veto", side=None)
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

    if int(sdat.get("total_trades", 0)) >= ldn_max_trades_per_day_total:
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
        LONDON_START_UTC,
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

    # Setup A window can overlap Setup D window. Evaluate A first, then D if A did not signal.
    in_arb_window = windows["london_open"] <= now_utc < windows["london_arb_end"]
    d_start = windows["london_open"] + pd.Timedelta(minutes=int(ldn_config.get("d_entry_start_min_after_london", 15)))
    d_end = windows["london_open"] + pd.Timedelta(minutes=int(ldn_config.get("d_entry_end_min_after_london", 120)))
    in_d_window = d_start <= now_utc < d_end

    if in_arb_window:
        if range_ok:
            side, reason = evaluate_london_v2_arb(m1_df, tick, high, low, pip, sdat)
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
            lor_pips = (lor_high - lor_low) / pip
            lor_valid = ldn_lor_min_pips <= lor_pips <= ldn_lor_max_pips
            state_updates["london_lor"] = {"date": today, "high": lor_high, "low": lor_low, "pips": lor_pips, "is_valid": lor_valid}
            if lor_valid and int(sdat.get("d_trades", 0)) < ldn_d_max_trades:
                close_px = float(m1_df.iloc[-1]["close"])
                if close_px > lor_high + LDN_D_BREAKOUT_BUFFER_PIPS * pip:
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

    try:
        acct = adapter.get_account_info()
        equity = float(acct.equity)
    except Exception:
        equity = 100000.0

    current_setup_risk_pct = ldn_arb_risk_pct if (strategy_tag or "").endswith("_arb") else ldn_d_risk_pct
    if (open_count * current_setup_risk_pct) + current_setup_risk_pct > ldn_max_total_risk_pct:
        no_trade["decision"] = ExecutionDecision(
            attempted=False, placed=False,
            reason=(
                f"london_v2: open risk cap "
                f"({((open_count * current_setup_risk_pct) + current_setup_risk_pct):.2%} > {ldn_max_total_risk_pct:.0%})"
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

    # USDJPY on OANDA: position units are base USD units, so required margin is close to units/leverage in USD.
    req_margin = abs(float(units)) / max(1.0, ldn_leverage)
    try:
        margin_used = getattr(adapter.get_account_info(), "margin_used", 0.0) or 0.0
        free_margin = max(0.0, equity - float(margin_used))
    except Exception:
        free_margin = equity * (1.0 - ldn_max_margin_frac)
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

    if strategy_tag.endswith("_arb"):
        sdat["arb_trades"] = int(sdat.get("arb_trades", 0)) + 1
    else:
        sdat["d_trades"] = int(sdat.get("d_trades", 0)) + 1
    sdat["total_trades"] = int(sdat.get("total_trades", 0)) + 1
    sdat["last_entry_time"] = now_utc.isoformat()
    state_updates[session_key] = sdat

    return {
        "decision": ExecutionDecision(
            attempted=True,
            placed=True,
            reason=f"{reason} | SL={sl_pips:.1f}p units={units}",
            side=side,
            order_retcode=getattr(dec, "order_retcode", None),
            order_id=getattr(dec, "order_id", None),
            deal_id=getattr(dec, "deal_id", None),
            fill_price=getattr(dec, "fill_price", None),
        ),
        "phase3_state_updates": state_updates,
        "strategy_tag": strategy_tag,
        "sl_price": float(sl_price),
        "tp1_price": float(tp1_price),
        "units": int(units),
        "entry_price": float(entry_price),
        "sl_pips": float(sl_pips),
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

    if now_utc < now_utc.replace(hour=NY_START_UTC, minute=v44_start_delay_min, second=0, microsecond=0):
        no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason="v44_ny: start delay active", side=None)
        return no_trade
    if (tick.ask - tick.bid) / pip > v44_max_entry_spread:
        no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason="v44_ny: spread veto", side=None)
        return no_trade

    h1_df = _drop_incomplete_tf(data_by_tf.get("H1"), "H1")
    m5_df = _drop_incomplete_tf(data_by_tf.get("M5"), "M5")
    if h1_df is None or h1_df.empty or m5_df is None or m5_df.empty:
        no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason="v44_ny: missing H1/M5", side=None)
        return no_trade

    v44_max_open = int(v44_config.get("max_open_positions", V44_MAX_OPEN))
    v44_risk_pct = float(v44_config.get("risk_per_trade_pct", 0.5)) / 100.0
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
    v44_min_body_pips = float(v44_config.get("entry_min_body_pips", V44_MIN_BODY_PIPS))

    today = now_utc.date().isoformat()
    session_key = f"session_ny_{today}"
    sdat = dict(phase3_state.get(session_key, {}))
    sdat.setdefault("trade_count", 0)
    sdat.setdefault("consecutive_losses", 0)
    sdat.setdefault("cooldown_until", None)
    sdat.setdefault("last_entry_time", None)

    open_count = int(phase3_state.get("open_trade_count", 0))
    if open_count >= v44_max_open:
        no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason=f"v44_ny: max open {open_count}/{v44_max_open}", side=None)
        return no_trade

    side, strength, reason = evaluate_v44_entry(
        h1_df,
        m5_df,
        tick,
        pip,
        "ny",
        sdat,
        now_utc=now_utc,
        max_entries_per_day=v44_max_entries_day,
        session_stop_losses=v44_session_stop_losses,
        h1_ema_fast_period=v44_h1_ema_fast,
        h1_ema_slow_period=v44_h1_ema_slow,
        m5_ema_fast_period=v44_m5_ema_fast,
        m5_ema_slow_period=v44_m5_ema_slow,
        slope_bars=v44_slope_bars,
        strong_slope_threshold=v44_strong_slope,
        min_body_pips=v44_min_body_pips,
    )
    if side is None:
        no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason=reason, side=None)
        return no_trade
    allow_map = {
        "strong_only": {"strong"},
        "strong_normal": {"strong", "normal"},
        "all": {"strong", "normal", "weak"},
    }
    allowed_strengths = allow_map.get(v44_ny_strength_allow, {"strong", "normal"})
    if strength not in allowed_strengths:
        no_trade["decision"] = ExecutionDecision(
            attempted=False,
            placed=False,
            reason=f"v44_ny: strength {strength} blocked by {v44_ny_strength_allow}",
            side=None,
        )
        return no_trade

    entry_price = tick.ask if side == "buy" else tick.bid
    sl_price = compute_v44_sl(side, m5_df, entry_price, pip)
    sl_pips = abs(entry_price - sl_price) / pip
    if strength == "strong":
        tp1_pips = v44_strong_tp1_pips
    elif strength == "normal":
        tp1_pips = v44_normal_tp1_pips
    else:
        tp1_pips = v44_weak_tp1_pips
    tp1_price = entry_price + (tp1_pips * pip if side == "buy" else -tp1_pips * pip)

    try:
        acct = adapter.get_account_info()
        equity = float(acct.equity)
    except Exception:
        equity = 100000.0

    risk_usd = equity * v44_risk_pct
    pip_value_per_lot = (pip / entry_price) * 100000.0
    if sl_pips <= 0 or pip_value_per_lot <= 0:
        units = 0
    else:
        lot = risk_usd / (sl_pips * pip_value_per_lot)
        lot = max(v44_rp_min_lot, min(v44_rp_max_lot, lot))
        units = int(lot * 100000.0)
    if units <= 0:
        no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason="v44_ny: size=0", side=None)
        return no_trade

    strategy_tag = f"phase3:v44_ny:{strength}"
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

    sdat["trade_count"] = int(sdat.get("trade_count", 0)) + 1
    sdat["last_entry_time"] = now_utc.isoformat()

    return {
        "decision": ExecutionDecision(
            attempted=True,
            placed=True,
            reason=f"{reason} | strength={strength} TP1={tp1_pips:.2f}p SL={sl_pips:.1f}p units={units}",
            side=side,
            order_retcode=getattr(dec, "order_retcode", None),
            order_id=getattr(dec, "order_id", None),
            deal_id=getattr(dec, "deal_id", None),
            fill_price=getattr(dec, "fill_price", None),
        ),
        "phase3_state_updates": {session_key: sdat},
        "strategy_tag": strategy_tag,
        "sl_price": float(sl_price),
        "tp1_price": float(tp1_price),
        "units": int(units),
        "entry_price": float(entry_price),
        "sl_pips": float(sl_pips),
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

    session = classify_session(now_utc)
    # Backtest/live parity: evaluate entries only once per newly closed M1 bar for all Phase 3 sessions.
    if session in {"tokyo", "london", "ny"} and not is_new_m1:
        no_trade["decision"] = ExecutionDecision(
            attempted=False, placed=False,
            reason="phase3: waiting for new closed M1 bar", side=None,
        )
        return no_trade
    if session == "london":
        return execute_london_v2_entry(
            adapter=adapter,
            profile=profile,
            policy=policy,
            data_by_tf=data_by_tf,
            tick=tick,
            phase3_state=phase3_state,
            sizing_config=sizing_config,
            now_utc=now_utc,
        )
    if session == "ny":
        return execute_v44_ny_entry(
            adapter=adapter,
            profile=profile,
            policy=policy,
            data_by_tf=data_by_tf,
            tick=tick,
            phase3_state=phase3_state,
            sizing_config=sizing_config,
            now_utc=now_utc,
        )
    if session != "tokyo":
        no_trade["decision"] = ExecutionDecision(
            attempted=False, placed=False,
            reason=f"phase3: no active session (current={session})", side=None,
        )
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
    if spread_pips > MAX_ENTRY_SPREAD_PIPS:
        no_trade["decision"] = ExecutionDecision(
            attempted=False, placed=False,
            reason=f"phase3: spread {spread_pips:.1f}p > max {MAX_ENTRY_SPREAD_PIPS}p", side=None,
        )
        return no_trade

    # 4) Pivots (cache by date)
    today_str = now_utc.date().isoformat()
    pivots = phase3_state.get("pivots")
    pivots_date = phase3_state.get("pivots_date")
    state_updates: dict = {}
    if pivots is None or pivots_date != today_str:
        d_df = _drop_incomplete_tf(data_by_tf.get("D"), "D")
        if d_df is None or d_df.empty or len(d_df) < 2:
            no_trade["decision"] = ExecutionDecision(
                attempted=False, placed=False,
                reason="phase3: no daily data for pivots", side=None,
            )
            return no_trade
        d_sorted = d_df.sort_values("time")
        prev = d_sorted.iloc[-2]
        pivots = compute_daily_fib_pivots(
            float(prev["high"]), float(prev["low"]), float(prev["close"]),
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

    # BB + regime
    if len(m5_df) < BB_PERIOD + BB_WIDTH_LOOKBACK:
        no_trade["decision"] = ExecutionDecision(
            attempted=False, placed=False,
            reason="phase3: insufficient M5 data for BB", side=None,
        )
        no_trade["phase3_state_updates"] = state_updates
        return no_trade

    regime = compute_bb_width_regime(m5_df)
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

    # 6) Regime gate
    if regime != "ranging":
        no_trade["decision"] = ExecutionDecision(
            attempted=False, placed=False,
            reason=f"phase3: regime={regime} (need ranging)", side=None,
        )
        no_trade["phase3_state_updates"] = state_updates
        return no_trade

    # 7) ADX gate
    if adx_val >= ADX_MAX:
        no_trade["decision"] = ExecutionDecision(
            attempted=False, placed=False,
            reason=f"phase3: ADX={adx_val:.1f} >= {ADX_MAX}", side=None,
        )
        no_trade["phase3_state_updates"] = state_updates
        return no_trade

    # 8) ATR gate
    if atr_val >= ATR_MAX:
        no_trade["decision"] = ExecutionDecision(
            attempted=False, placed=False,
            reason=f"phase3: ATR={atr_val:.4f} >= {ATR_MAX}", side=None,
        )
        no_trade["phase3_state_updates"] = state_updates
        return no_trade

    # 9) Session management: concurrent positions, trade count, cooldown, consecutive losses (config-driven)
    v14_config = (sizing_config or {}).get("v14", {})
    max_trades_session = int(v14_config.get("max_trades_per_session", MAX_TRADES_PER_SESSION))
    stop_consec_losses = int(v14_config.get("stop_after_consecutive_losses", STOP_AFTER_CONSECUTIVE_LOSSES))
    cooldown_min = int(v14_config.get("cooldown_minutes", COOLDOWN_MINUTES))
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
        elapsed = (now_utc - datetime.fromisoformat(last_entry_time)).total_seconds() / 60.0
        if elapsed < cooldown_min:
            no_trade["decision"] = ExecutionDecision(
                attempted=False, placed=False,
                reason=f"phase3: cooldown ({elapsed:.1f}/{cooldown_min}min)", side=None,
            )
            no_trade["phase3_state_updates"] = state_updates
            return no_trade

    concurrent = phase3_state.get("open_trade_count", 0)
    if concurrent >= max_concurrent:
        no_trade["decision"] = ExecutionDecision(
            attempted=False, placed=False,
            reason=f"phase3: max concurrent ({concurrent}/{max_concurrent})", side=None,
        )
        no_trade["phase3_state_updates"] = state_updates
        return no_trade

    # 10) Evaluate confluence for both sides
    close_price = float(m5_close.iloc[-1])
    best_side = None
    best_score = 0
    best_combo = ""

    for side in ("buy", "sell"):
        score, combo = evaluate_v14_confluence(
            side, close_price, pivots, bb_upper, bb_lower,
            sar_bull, sar_bear, rsi_val, pip,
        )
        if score >= MIN_CONFLUENCE and combo not in BLOCKED_COMBOS and score > best_score:
            best_side = side
            best_score = score
            best_combo = combo

    if best_side is None:
        no_trade["decision"] = ExecutionDecision(
            attempted=False, placed=False,
            reason=f"phase3: no qualifying confluence (RSI={rsi_val:.1f})", side=None,
        )
        no_trade["phase3_state_updates"] = state_updates
        return no_trade

    # 11) Compute SL, TP1, lot size
    entry_price = tick.ask if best_side == "buy" else tick.bid
    sl_price = compute_v14_sl(best_side, entry_price, pivots, pip)
    sl_pips = abs(entry_price - sl_price) / pip
    tp1_price = compute_v14_tp1(best_side, entry_price, atr_val, pip)

    try:
        acct = adapter.get_account_info()
        equity = float(acct.equity)
    except Exception:
        equity = 100_000.0  # fallback

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

    # Update session state
    session_data["trade_count"] = trade_count + 1
    session_data["last_entry_time"] = now_utc.isoformat()
    state_updates[session_key] = session_data

    return {
        "decision": ExecutionDecision(
            attempted=True, placed=True,
            reason=f"phase3:v14 {best_side.upper()} confluence={best_score} combo={best_combo} "
                   f"SL={sl_price:.3f}({sl_pips:.1f}p) TP1={tp1_price:.3f} units={units}",
            side=best_side,
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


def _manage_london_v2_exit(*, adapter, profile, store, tick, trade_row: dict, position) -> dict:
    pip = float(profile.pip_size) if hasattr(profile, "pip_size") else PIP_SIZE
    side = str(trade_row["side"]).lower()
    entry = float(trade_row["entry_price"])
    trade_id = str(trade_row["trade_id"])
    now_utc = datetime.now(timezone.utc)
    position_id, current_lots, _ = _phase3_position_meta(position, side)
    if current_lots <= 0:
        return {"action": "none", "reason": "no units"}

    day_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    ny_open = day_start + pd.Timedelta(hours=NY_START_UTC)
    tp_check_price = tick.bid if side == "buy" else tick.ask

    if LDN_FORCE_CLOSE_AT_NY_OPEN and now_utc >= ny_open:
        try:
            _close_full(adapter, profile, position_id, current_lots, side)
            return {"action": "session_end_close", "reason": "london_v2 hard close at NY open"}
        except Exception as e:
            return {"action": "error", "reason": f"london_v2 hard close error: {e}"}

    stop_price = trade_row.get("stop_price")
    if stop_price is None:
        return {"action": "none", "reason": "no stop in trade row"}
    stop_price = float(stop_price)
    r_pips = abs(entry - stop_price) / pip
    tp1_price = entry + (LDN_ARB_TP1_R * r_pips * pip if side == "buy" else -LDN_ARB_TP1_R * r_pips * pip)
    tp2_price = entry + (LDN_ARB_TP2_R * r_pips * pip if side == "buy" else -LDN_ARB_TP2_R * r_pips * pip)

    tp1_done = int(trade_row.get("tp1_partial_done") or 0) == 1
    reached_tp1 = tp_check_price >= tp1_price if side == "buy" else tp_check_price <= tp1_price
    reached_tp2 = tp_check_price >= tp2_price if side == "buy" else tp_check_price <= tp2_price

    if (not tp1_done) and reached_tp1:
        try:
            position_type = 1 if side == "sell" else 0
            close_lots = current_lots * LDN_ARB_TP1_CLOSE_PCT
            adapter.close_position(ticket=position_id, symbol=profile.symbol, volume=close_lots, position_type=position_type)
            be_sl = entry + (LDN_ARB_BE_OFFSET_PIPS * pip if side == "buy" else -LDN_ARB_BE_OFFSET_PIPS * pip)
            adapter.update_position_stop_loss(position_id, profile.symbol, round(be_sl, 3))
            store.update_trade(trade_id, {"tp1_partial_done": 1, "breakeven_applied": 1, "breakeven_sl_price": round(be_sl, 5)})
            return {"action": "tp1_partial", "reason": f"london_v2 TP1 partial + BE ({be_sl:.3f})"}
        except Exception as e:
            return {"action": "error", "reason": f"london_v2 TP1 error: {e}"}

    if tp1_done and reached_tp2:
        try:
            _close_full(adapter, profile, position_id, current_lots, side)
            return {"action": "tp2_full", "reason": "london_v2 TP2 runner close"}
        except Exception as e:
            return {"action": "error", "reason": f"london_v2 TP2 close error: {e}"}

    return {"action": "none", "reason": "no london_v2 exit condition met"}


def _manage_v44_exit(*, adapter, profile, store, tick, trade_row: dict, position) -> dict:
    pip = float(profile.pip_size) if hasattr(profile, "pip_size") else PIP_SIZE
    side = str(trade_row["side"]).lower()
    entry = float(trade_row["entry_price"])
    trade_id = str(trade_row["trade_id"])
    now_utc = datetime.now(timezone.utc)
    position_id, current_lots, _ = _phase3_position_meta(position, side)
    if current_lots <= 0:
        return {"action": "none", "reason": "no units"}

    day_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    ny_end = day_start + pd.Timedelta(hours=NY_END_UTC)
    tp_check_price = tick.bid if side == "buy" else tick.ask
    trail_mark_price = tick.bid if side == "buy" else tick.ask

    if now_utc >= ny_end:
        try:
            _close_full(adapter, profile, position_id, current_lots, side)
            return {"action": "session_end_close", "reason": "v44_ny hard close at session end"}
        except Exception as e:
            return {"action": "error", "reason": f"v44_ny hard close error: {e}"}

    tp1_done = int(trade_row.get("tp1_partial_done") or 0) == 1
    tp1_price = trade_row.get("target_price")
    if tp1_price is None:
        tp1_price = entry + (V44_STRONG_TP1_PIPS * pip if side == "buy" else -V44_STRONG_TP1_PIPS * pip)
    tp1_price = float(tp1_price)
    reached_tp1 = tp_check_price >= tp1_price if side == "buy" else tp_check_price <= tp1_price

    if (not tp1_done) and reached_tp1:
        try:
            position_type = 1 if side == "sell" else 0
            close_lots = current_lots * V44_STRONG_TP1_CLOSE_PCT
            adapter.close_position(ticket=position_id, symbol=profile.symbol, volume=close_lots, position_type=position_type)
            be_sl = entry + (V44_BE_OFFSET_PIPS * pip if side == "buy" else -V44_BE_OFFSET_PIPS * pip)
            adapter.update_position_stop_loss(position_id, profile.symbol, round(be_sl, 3))
            store.update_trade(trade_id, {"tp1_partial_done": 1, "breakeven_applied": 1, "breakeven_sl_price": round(be_sl, 5)})
            return {"action": "tp1_partial", "reason": f"v44_ny TP1 partial + BE ({be_sl:.3f})"}
        except Exception as e:
            return {"action": "error", "reason": f"v44_ny TP1 error: {e}"}

    if tp1_done:
        entry_type = str(trade_row.get("entry_type") or "")
        if ":normal" in entry_type:
            trail_buffer_pips = 3.0
        elif ":weak" in entry_type:
            trail_buffer_pips = 2.0
        else:
            trail_buffer_pips = V44_STRONG_TRAIL_BUFFER
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
        )
    if entry_type.startswith("phase3:v44"):
        return _manage_v44_exit(
            adapter=adapter,
            profile=profile,
            store=store,
            tick=tick,
            trade_row=trade_row,
            position=position,
        )

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
    session = classify_session(now_utc)
    if session != "tokyo":
        try:
            adapter.close_position(
                ticket=position_id,
                symbol=profile.symbol,
                volume=current_lots,
                position_type=position_type,
            )
            return {"action": "session_end_close", "reason": f"session ended (now={session})"}
        except Exception as e:
            return {"action": "error", "reason": f"session end close error: {e}"}

    # 2) Time decay check
    opened_at = trade_row.get("opened_at")
    if opened_at is not None:
        try:
            if isinstance(opened_at, str):
                open_time = datetime.fromisoformat(opened_at)
                if open_time.tzinfo is None:
                    open_time = open_time.replace(tzinfo=timezone.utc)
            else:
                open_time = opened_at
            elapsed_min = (now_utc - open_time).total_seconds() / 60.0
            if elapsed_min >= TIME_DECAY_MINUTES:
                profit_pips = ((exit_check_price - entry) / pip) if side == "buy" else ((entry - exit_check_price) / pip)
                if profit_pips < TIME_DECAY_CAP_PIPS:
                    try:
                        adapter.close_position(
                            ticket=position_id,
                            symbol=profile.symbol,
                            volume=current_lots,
                            position_type=position_type,
                        )
                        return {
                            "action": "time_decay_close",
                            "reason": f"time decay {elapsed_min:.0f}min, profit={profit_pips:.1f}p < {TIME_DECAY_CAP_PIPS}p",
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
                return {"action": "hard_sl", "reason": f"hard SL hit at {check_price:.3f}"}
            except Exception as e:
                return {"action": "error", "reason": f"hard SL close error: {e}"}
        elif side == "sell" and check_price >= stop_price:
            try:
                adapter.close_position(ticket=position_id, symbol=profile.symbol, volume=current_lots, position_type=position_type)
                return {"action": "hard_sl", "reason": f"hard SL hit at {check_price:.3f}"}
            except Exception as e:
                return {"action": "error", "reason": f"hard SL close error: {e}"}

    # 4) TP1 partial close
    tp1_done = trade_row.get("tp1_partial_done") or 0
    if not tp1_done:
        target_price = trade_row.get("target_price")
        if target_price is not None:
            target_price = float(target_price)
            reached_buy = exit_check_price >= target_price and side == "buy"
            reached_sell = exit_check_price <= target_price and side == "sell"
            if reached_buy or reached_sell:
                close_lots = current_lots * TP1_CLOSE_PCT
                try:
                    adapter.close_position(
                        ticket=position_id,
                        symbol=profile.symbol,
                        volume=close_lots,
                        position_type=position_type,
                    )
                    # Move SL to BE + offset
                    be_offset = current_spread + BE_OFFSET_PIPS * pip
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
                        "reason": f"TP1 {TP1_CLOSE_PCT*100:.0f}% closed, BE SL={be_sl:.3f}",
                    }
                except Exception as e:
                    return {"action": "error", "reason": f"TP1 close error: {e}"}

    # 5) Trailing stop (after TP1)
    elif tp1_done:
        profit_pips = ((exit_check_price - entry) / pip) if side == "buy" else ((entry - exit_check_price) / pip)
        if profit_pips >= TRAIL_ACTIVATE_PROFIT_PIPS:
            prev_trail = trade_row.get("breakeven_sl_price")
            if prev_trail is not None:
                prev_trail = float(prev_trail)
            if side == "buy":
                new_trail = exit_check_price - TRAIL_DISTANCE_PIPS * pip
                if prev_trail is not None and TRAIL_NEVER_WIDEN:
                    new_trail = max(new_trail, prev_trail)
            else:
                new_trail = exit_check_price + TRAIL_DISTANCE_PIPS * pip
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

def report_phase3_session(now_utc: datetime | None = None) -> dict:
    """Report current session classification."""
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    session = classify_session(now_utc)
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


def report_phase3_adx(adx_val: float) -> dict:
    """Report ADX status."""
    return {
        "name": "V14 ADX",
        "value": f"{adx_val:.1f}",
        "ok": adx_val < ADX_MAX,
        "detail": f"max {ADX_MAX}",
    }


def report_phase3_atr(atr_val: float) -> dict:
    """Report ATR status."""
    atr_pips = atr_val / PIP_SIZE
    return {
        "name": "V14 ATR",
        "value": f"{atr_pips:.1f}p",
        "ok": atr_val < ATR_MAX,
        "detail": f"max {ATR_MAX/PIP_SIZE:.0f}p",
    }


def report_phase3_london_range(asian_pips: float, is_valid: bool) -> dict:
    return {
        "name": "London Asian Range",
        "value": f"{asian_pips:.1f}p",
        "ok": bool(is_valid),
        "detail": f"valid={LDN_ARB_RANGE_MIN_PIPS:.0f}-{LDN_ARB_RANGE_MAX_PIPS:.0f}p",
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


def report_phase3_ny_slope(slope_pips_per_bar: float) -> dict:
    return {
        "name": "NY M5 Slope",
        "value": f"{slope_pips_per_bar:.2f} p/bar",
        "ok": abs(slope_pips_per_bar) >= V44_STRONG_SLOPE,
        "detail": f"strong>={V44_STRONG_SLOPE:.2f}",
    }


def report_phase3_ny_atr_filter(ok: bool) -> dict:
    return {
        "name": "NY ATR PCT Filter",
        "value": "pass" if ok else "blocked",
        "ok": bool(ok),
        "detail": f"cap P{int(V44_ATR_PCT_CAP*100)} lookback={V44_ATR_PCT_LOOKBACK}",
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
    if now_utc < windows["london_open"]:
        return {"name": "London Window", "value": "pre-open", "ok": False, "detail": "Before 08:00 UTC"}
    if now_utc < windows["london_arb_end"]:
        return {"name": "London Window", "value": "Setup A (08:00–09:30 UTC)", "ok": True, "detail": "Asian range breakout"}
    if now_utc < windows["london_end"]:
        return {"name": "London Window", "value": "Setup D (08:15–10:00 UTC active)", "ok": True, "detail": "LOR breakout"}
    return {"name": "London Window", "value": "closed", "ok": False, "detail": "After 12:00 UTC"}


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
