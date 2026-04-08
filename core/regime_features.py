"""
Advanced regime features for the regime classifier.

These features capture structural market transitions that lagging indicators
(ADX, BB width) miss — specifically, the moment when directional movement
stops being accepted and starts being rejected at a meaningful level.

Core features:
    efficiency_ratio  – net displacement / total path (0=chop, 1=one-way)
    rejection_intensity – directional wick pressure against the trend
    level_touch_density – repeated interaction with a narrow price zone
    trend_decay_rate   – how much slope has degraded vs recent peak

Dynamic transition features:
    delta_efficiency_ratio – ER change vs a few bars ago (negative = deterioration)
    delta_touch_density    – touch-density change vs a few bars ago (positive = sticking)
    failed_continuation    – trend stalls near a recent swing extreme without continuation

All functions are pure, stateless, and operate on M5 OHLC DataFrames.
They return 0.0–1.0 normalized values suitable for direct use as
regime scoring components.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


PIP_SIZE = 0.01


@dataclass
class RegimeFeatures:
    """Container for advanced features fed into the regime classifier."""
    efficiency_ratio: float = 0.5       # 0=pure chop, 1=pure trend
    rejection_intensity: float = 0.0    # 0=no rejection, 1=heavy rejection
    level_touch_density: float = 0.0    # 0=no clustering, 1=stuck at level
    trend_decay_rate: float = 0.0       # 0=trend stable/growing, 1=trend collapsing
    delta_efficiency_ratio: float = 0.0 # signed change, negative = deterioration
    delta_touch_density: float = 0.0    # signed change, positive = clustering increasing
    failed_continuation: float = 0.0    # 0=no stall, 1=stalled near swing extreme

    @property
    def as_dict(self) -> dict[str, float]:
        return {
            "efficiency_ratio": self.efficiency_ratio,
            "rejection_intensity": self.rejection_intensity,
            "level_touch_density": self.level_touch_density,
            "trend_decay_rate": self.trend_decay_rate,
            "delta_efficiency_ratio": self.delta_efficiency_ratio,
            "delta_touch_density": self.delta_touch_density,
            "failed_continuation": self.failed_continuation,
        }


# ── Feature 1: Efficiency Ratio ────────────────────────────────────

def compute_efficiency_ratio(
    m5_df: pd.DataFrame,
    lookback: int = 12,
) -> float:
    """Kaufman-style efficiency ratio over the last `lookback` M5 bars.

    ER = |close[-1] - close[-lookback]| / sum(|close[i] - close[i-1]|)

    Returns 0.0–1.0.  Near 1.0 = clean one-way move, near 0.0 = churning.
    A trending market that runs into resistance will see ER drop sharply
    even while ADX remains elevated.
    """
    if len(m5_df) < lookback + 1:
        return 0.5  # not enough data, neutral

    closes = m5_df["close"].astype(float).values[-(lookback + 1):]
    net_displacement = abs(closes[-1] - closes[0])
    total_path = np.sum(np.abs(np.diff(closes)))

    if total_path < PIP_SIZE * 0.1:
        return 0.0  # effectively zero movement
    return min(1.0, net_displacement / total_path)


# ── Feature 2: Rejection Intensity ─────────────────────────────────

def compute_rejection_intensity(
    m5_df: pd.DataFrame,
    lookback: int = 8,
    trend_sign: int = 0,
) -> float:
    """Measure directional wick rejection pressure.

    In an uptrend (trend_sign > 0): upper wicks signal rejection at highs.
    In a downtrend (trend_sign < 0): lower wicks signal rejection at lows.
    If trend_sign == 0: uses max(upper, lower) wick ratio.

    Returns 0.0–1.0.  Higher = more rejection behavior.

    Computed as: mean(rejection_wick / bar_range) over lookback bars.
    """
    if len(m5_df) < lookback:
        return 0.0

    df = m5_df.iloc[-lookback:]
    o = df["open"].astype(float).values
    h = df["high"].astype(float).values
    l = df["low"].astype(float).values
    c = df["close"].astype(float).values

    bar_range = h - l
    # Avoid division by zero on doji bars
    bar_range = np.where(bar_range < PIP_SIZE * 0.1, PIP_SIZE * 0.1, bar_range)

    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l

    if trend_sign > 0:
        # Uptrend: upper wicks are rejection
        rejection_wick = upper_wick
    elif trend_sign < 0:
        # Downtrend: lower wicks are rejection
        rejection_wick = lower_wick
    else:
        # No clear trend: take whichever wick is larger per bar
        rejection_wick = np.maximum(upper_wick, lower_wick)

    wick_ratios = rejection_wick / bar_range
    return float(np.clip(np.mean(wick_ratios), 0.0, 1.0))


# ── Feature 3: Level Touch Density ─────────────────────────────────

def compute_level_touch_density(
    m5_df: pd.DataFrame,
    lookback: int = 20,
    zone_pips: float = 5.0,
) -> float:
    """Detect repeated interaction with a narrow price zone.

    Algorithm:
    1. Take the last `lookback` M5 closes.
    2. Find the most-visited zone (rounded to zone_pips).
    3. Count what fraction of bars had a high/low that entered that zone.

    Returns 0.0–1.0.  Higher = price is stuck at a level.

    This captures the "tried to break, came back, tried again" pattern
    that precedes regime transitions.
    """
    if len(m5_df) < lookback:
        return 0.0

    df = m5_df.iloc[-lookback:]
    h = df["high"].astype(float).values
    l = df["low"].astype(float).values
    c = df["close"].astype(float).values

    zone_size = zone_pips * PIP_SIZE

    # Round closes to zone grid to find the most-visited zone
    zone_centers = np.round(c / zone_size) * zone_size
    # Find mode (most common zone center)
    unique, counts = np.unique(zone_centers, return_counts=True)
    mode_center = unique[np.argmax(counts)]

    # Count bars whose high-low range overlaps with the mode zone
    zone_lo = mode_center - zone_size / 2
    zone_hi = mode_center + zone_size / 2
    touches = np.sum((h >= zone_lo) & (l <= zone_hi))

    # Normalize: if every bar touches the zone, density = 1.0
    density = touches / lookback
    return float(np.clip(density, 0.0, 1.0))


# ── Feature 4: Trend Decay Rate ────────────────────────────────────

def compute_trend_decay_rate(
    m5_df: pd.DataFrame,
    ema_period: int = 9,
    slope_bars: int = 4,
    lookback: int = 12,
) -> float:
    """Detect trend quality degradation: slope decaying from recent peak.

    1. Compute EMA slope at each bar over the last `lookback` bars.
    2. Find peak |slope| in that window.
    3. Return 1 - (current |slope| / peak |slope|).

    Returns 0.0–1.0.  0.0 = slope is at or near its recent peak (healthy).
    1.0 = slope has collapsed to zero from a meaningful peak.

    This catches the subtle case where ADX is still high (lagging) but the
    actual rate of price change has already decayed.
    """
    needed = lookback + slope_bars + ema_period + 5  # safety margin
    if len(m5_df) < needed:
        return 0.0

    closes = m5_df["close"].astype(float)
    ema = closes.ewm(span=ema_period, adjust=False).mean()

    # Compute slope at each bar (pips per bar)
    slopes = (ema.diff(slope_bars) / PIP_SIZE / max(1, slope_bars)).values

    # Take the last `lookback` slope values
    recent_slopes = slopes[-lookback:]
    recent_abs = np.abs(recent_slopes)

    peak = np.nanmax(recent_abs)
    current = recent_abs[-1] if not np.isnan(recent_abs[-1]) else 0.0

    if peak < 0.05:
        # No meaningful trend existed in this window
        return 0.0

    decay = 1.0 - (current / peak)
    return float(np.clip(decay, 0.0, 1.0))


def compute_trend_decay_rate_from_ema(
    m5_df: pd.DataFrame,
    ema_series: pd.Series,
    *,
    slope_bars: int = 4,
    lookback: int = 12,
) -> float:
    """Same as ``compute_trend_decay_rate`` but uses a precomputed EMA (full-series warm).

    Lets callers pass only a short OHLC tail while keeping EMA values identical to a
    full-history ``ewm`` on M5 closes.
    """
    needed = lookback + slope_bars + 5
    if len(m5_df) < needed or len(ema_series) != len(m5_df):
        return 0.0

    ema = ema_series.astype(float)
    slopes = (ema.diff(slope_bars) / PIP_SIZE / max(1, slope_bars)).values
    recent_slopes = slopes[-lookback:]
    recent_abs = np.abs(recent_slopes)

    peak = np.nanmax(recent_abs)
    current = recent_abs[-1] if not np.isnan(recent_abs[-1]) else 0.0

    if peak < 0.05:
        return 0.0

    decay = 1.0 - (current / peak)
    return float(np.clip(decay, 0.0, 1.0))


# ── Dynamic Feature 5: Delta Efficiency Ratio ───────────────────────

def compute_delta_efficiency_ratio(
    m5_df: pd.DataFrame,
    lookback: int = 12,
    delta_bars: int = 3,
) -> float:
    """Signed ER change versus `delta_bars` bars ago.

    Negative values mean directional efficiency is deteriorating.
    Positive values mean efficiency is improving.
    """
    if len(m5_df) < lookback + delta_bars + 1:
        return 0.0

    current_er = compute_efficiency_ratio(m5_df, lookback=lookback)
    prior_er = compute_efficiency_ratio(m5_df.iloc[:-delta_bars], lookback=lookback)
    return float(np.clip(current_er - prior_er, -1.0, 1.0))


# ── Dynamic Feature 6: Delta Touch Density ──────────────────────────

def compute_delta_touch_density(
    m5_df: pd.DataFrame,
    lookback: int = 20,
    delta_bars: int = 3,
    zone_pips: float = 5.0,
) -> float:
    """Signed touch-density change versus `delta_bars` bars ago.

    Positive values mean price is becoming more stuck at a level.
    """
    if len(m5_df) < lookback + delta_bars:
        return 0.0

    current_touch = compute_level_touch_density(m5_df, lookback=lookback, zone_pips=zone_pips)
    prior_touch = compute_level_touch_density(m5_df.iloc[:-delta_bars], lookback=lookback, zone_pips=zone_pips)
    return float(np.clip(current_touch - prior_touch, -1.0, 1.0))


# ── Dynamic Feature 7: Failed Continuation ──────────────────────────

def compute_failed_continuation(
    m5_df: pd.DataFrame,
    trend_sign: int = 0,
    *,
    lookback: int = 12,
    stall_bars: int = 4,
    proximity_pips: float = 5.0,
) -> float:
    """Detect stalled continuation near a recent swing extreme.

    Uptrend case:
      - a recent swing high exists
      - price is still within `proximity_pips` of that high
      - no fresh high has been made for at least `stall_bars`

    Downtrend case is symmetric at the swing low.
    Returns 1.0 when the pattern is present, else 0.0.
    """
    if len(m5_df) < max(lookback, stall_bars + 2):
        return 0.0

    df = m5_df.iloc[-lookback:]
    closes = df["close"].astype(float).values
    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values

    if trend_sign == 0:
        ema9 = pd.Series(closes).ewm(span=9, adjust=False).mean().iloc[-1]
        ema21 = pd.Series(closes).ewm(span=21, adjust=False).mean().iloc[-1]
        if ema9 > ema21:
            trend_sign = 1
        elif ema9 < ema21:
            trend_sign = -1
        else:
            return 0.0

    proximity = proximity_pips * PIP_SIZE

    if trend_sign > 0:
        swing_idx = int(np.argmax(highs))
        bars_since_swing = len(highs) - 1 - swing_idx
        if bars_since_swing < stall_bars:
            return 0.0
        swing_high = highs[swing_idx]
        near_extreme = (swing_high - closes[-1]) <= proximity
        return 1.0 if near_extreme else 0.0

    swing_idx = int(np.argmin(lows))
    bars_since_swing = len(lows) - 1 - swing_idx
    if bars_since_swing < stall_bars:
        return 0.0
    swing_low = lows[swing_idx]
    near_extreme = (closes[-1] - swing_low) <= proximity
    return 1.0 if near_extreme else 0.0


# ── Convenience: compute all features at once ──────────────────────

def compute_all_features(
    m5_df: pd.DataFrame,
    trend_sign: int = 0,
    *,
    er_lookback: int = 12,
    rej_lookback: int = 8,
    touch_lookback: int = 20,
    touch_zone_pips: float = 5.0,
    decay_lookback: int = 12,
    decay_ema_period: int = 9,
    decay_slope_bars: int = 4,
    delta_er_bars: int = 3,
    delta_touch_bars: int = 3,
    failed_cont_lookback: int = 12,
    failed_cont_stall_bars: int = 4,
    failed_cont_proximity_pips: float = 5.0,
) -> RegimeFeatures:
    """Compute all four advanced features from M5 OHLC data.

    Args:
        m5_df: M5 OHLC DataFrame with open/high/low/close columns.
        trend_sign: +1 for bullish, -1 for bearish, 0 for unknown.
            Used by rejection_intensity to know which wicks matter.
    """
    return RegimeFeatures(
        efficiency_ratio=compute_efficiency_ratio(m5_df, lookback=er_lookback),
        rejection_intensity=compute_rejection_intensity(m5_df, lookback=rej_lookback, trend_sign=trend_sign),
        level_touch_density=compute_level_touch_density(m5_df, lookback=touch_lookback, zone_pips=touch_zone_pips),
        trend_decay_rate=compute_trend_decay_rate(m5_df, ema_period=decay_ema_period, slope_bars=decay_slope_bars, lookback=decay_lookback),
        delta_efficiency_ratio=compute_delta_efficiency_ratio(m5_df, lookback=er_lookback, delta_bars=delta_er_bars),
        delta_touch_density=compute_delta_touch_density(m5_df, lookback=touch_lookback, delta_bars=delta_touch_bars, zone_pips=touch_zone_pips),
        failed_continuation=compute_failed_continuation(
            m5_df,
            trend_sign=trend_sign,
            lookback=failed_cont_lookback,
            stall_bars=failed_cont_stall_bars,
            proximity_pips=failed_cont_proximity_pips,
        ),
    )
