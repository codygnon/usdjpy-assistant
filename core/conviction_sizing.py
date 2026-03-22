"""Conviction-based lot sizing for Trial #9.

Uses M5 EMA 9/21 spread + slope as the primary regime signal,
and M1 EMA 5/9 spread + compression as a confirming/dampening factor.
Outputs a single lot multiplier applied to a configurable base_lots value.

Thresholds are derived from USDJPY candle-close quantile analysis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Thresholds (data-backed from USDJPY quantile study)
# ---------------------------------------------------------------------------

# M5 EMA 9/21 spread thresholds (pips)
M5_WEAK_SPREAD_PIPS = 1.1       # 25th percentile
M5_STRONG_SPREAD_PIPS = 3.0     # aligned-trend median

# M5 EMA 9 slope thresholds (pips per bar, 3-bar lookback)
M5_WEAK_SLOPE_PIPS_PER_BAR = 0.30   # 25th percentile
M5_STRONG_SLOPE_PIPS_PER_BAR = 0.80  # aligned-trend median
M5_SLOPE_LOOKBACK_BARS = 3

# M1 EMA 5/9 spread thresholds (pips)
M1_TIGHT_SPREAD_PIPS = 0.23     # 25th percentile
M1_CONFIRM_SPREAD_PIPS = 0.53   # median

# M1 compression detection
M1_COMPRESSION_BARS = 3          # consecutive decreasing spread bars

# ---------------------------------------------------------------------------
# Multiplier matrix: (m5_bucket, m1_bucket) -> multiplier
# ---------------------------------------------------------------------------

MULTIPLIER_MATRIX: dict[tuple[str, str], float] = {
    ("strong", "confirm"):  1.5,
    ("strong", "neutral"):  1.2,
    ("strong", "dampen"):   0.8,
    ("normal", "confirm"):  1.0,
    ("normal", "neutral"):  0.8,
    ("normal", "dampen"):   0.6,
    ("weak",   "confirm"):  0.7,
    ("weak",   "neutral"):  0.5,
    ("weak",   "dampen"):   0.3,
}


@dataclass
class ConvictionResult:
    """Output of conviction sizing calculation."""

    enabled: bool = False
    m5_bucket: str = "normal"       # strong / normal / weak
    m1_bucket: str = "neutral"      # confirm / neutral / dampen
    multiplier: float = 1.0
    conviction_lots: float = 0.0    # final lot size after clamping
    base_lots: float = 0.0
    min_lots: float = 0.0
    max_lots: float = 0.0
    # Diagnostic metadata for dashboard
    m5_spread_pips: float = 0.0
    m5_slope_pips_per_bar: float = 0.0
    m5_slope_aligned: bool = False
    m5_slope_disagrees_with_ema21: bool = False
    m1_spread_pips: float = 0.0
    m1_aligned: bool = False
    m1_compressing: bool = False
    m1_compression_bars_count: int = 0
    m5_trend: str = ""              # "bull" or "bear"


def _ema(series: pd.Series, span: int) -> pd.Series:
    """Compute EMA using pandas ewm (same as execution engine)."""
    return series.ewm(span=span, adjust=False).mean()


def _compute_m5_bucket(
    m5_close: pd.Series,
    pip_size: float,
    is_bull: bool,
) -> tuple[str, float, float, bool, bool]:
    """Classify M5 regime into strong/normal/weak.

    Returns (bucket, spread_pips, slope_pips_per_bar, slope_aligned, slope_disagrees_ema21).
    """
    ema9 = _ema(m5_close, 9)
    ema21 = _ema(m5_close, 21)

    spread_pips = abs(float(ema9.iloc[-1]) - float(ema21.iloc[-1])) / pip_size

    # EMA 9 slope: (ema9[-1] - ema9[-1 - lookback]) / lookback
    if len(ema9) > M5_SLOPE_LOOKBACK_BARS:
        ema9_current = float(ema9.iloc[-1])
        ema9_past = float(ema9.iloc[-1 - M5_SLOPE_LOOKBACK_BARS])
        raw_slope = (ema9_current - ema9_past) / M5_SLOPE_LOOKBACK_BARS
        slope_pips_per_bar = abs(raw_slope) / pip_size
        # Slope aligned with trend direction
        slope_aligned = (raw_slope > 0 and is_bull) or (raw_slope < 0 and not is_bull)
    else:
        slope_pips_per_bar = 0.0
        slope_aligned = False

    # EMA 21 slope for disagreement check
    slope_disagrees_ema21 = False
    if len(ema21) > M5_SLOPE_LOOKBACK_BARS:
        ema21_current = float(ema21.iloc[-1])
        ema21_past = float(ema21.iloc[-1 - M5_SLOPE_LOOKBACK_BARS])
        ema21_slope = ema21_current - ema21_past
        ema9_slope_raw = float(ema9.iloc[-1]) - float(ema9.iloc[-1 - M5_SLOPE_LOOKBACK_BARS])
        # Disagree = one rising and the other falling
        if (ema9_slope_raw > 0 and ema21_slope < 0) or (ema9_slope_raw < 0 and ema21_slope > 0):
            slope_disagrees_ema21 = True

    # Bucket classification
    # Strong: spread wide AND slope strong AND aligned
    if spread_pips > M5_STRONG_SPREAD_PIPS and slope_pips_per_bar > M5_STRONG_SLOPE_PIPS_PER_BAR and slope_aligned:
        bucket = "strong"
    # Weak: spread narrow OR slope flat OR slopes disagree
    elif spread_pips < M5_WEAK_SPREAD_PIPS or slope_pips_per_bar < M5_WEAK_SLOPE_PIPS_PER_BAR or slope_disagrees_ema21:
        bucket = "weak"
    else:
        bucket = "normal"

    return bucket, spread_pips, slope_pips_per_bar, slope_aligned, slope_disagrees_ema21


def _compute_m1_bucket(
    m1_close: pd.Series,
    pip_size: float,
    is_bull: bool,
) -> tuple[str, float, bool, bool, int]:
    """Classify M1 health into confirm/neutral/dampen.

    Returns (bucket, spread_pips, aligned, compressing, compression_bars_count).
    """
    ema5 = _ema(m1_close, 5)
    ema9 = _ema(m1_close, 9)

    ema5_val = float(ema5.iloc[-1])
    ema9_val = float(ema9.iloc[-1])
    spread_pips = abs(ema5_val - ema9_val) / pip_size

    # Alignment: EMA5 > EMA9 for bull, EMA5 < EMA9 for bear
    if is_bull:
        aligned = ema5_val > ema9_val
    else:
        aligned = ema5_val < ema9_val

    # Compression: spread has decreased for N consecutive closed M1 bars
    compressing = False
    compression_count = 0
    if len(ema5) >= M1_COMPRESSION_BARS + 1:
        spreads = (ema5 - ema9).abs() / pip_size
        # Check last N bars (completed bars, so iloc[-N-1:-1] for the N bars before current)
        # We use the last N+1 spread values to check N consecutive decreases
        recent = spreads.iloc[-(M1_COMPRESSION_BARS + 1):].values
        consecutive_decrease = 0
        for i in range(1, len(recent)):
            if recent[i] < recent[i - 1]:
                consecutive_decrease += 1
            else:
                consecutive_decrease = 0
        compression_count = consecutive_decrease
        if consecutive_decrease >= M1_COMPRESSION_BARS:
            compressing = True

    # Also check absolute floor
    if spread_pips < M1_TIGHT_SPREAD_PIPS:
        compressing = True

    # Bucket classification
    # Dampen: crossing against M5 OR compressing
    if not aligned or compressing:
        if not aligned:
            bucket = "dampen"
        elif compressing:
            bucket = "dampen"
        else:
            bucket = "dampen"
    # Confirm: aligned AND spread above median AND not compressing
    elif aligned and spread_pips > M1_CONFIRM_SPREAD_PIPS and not compressing:
        bucket = "confirm"
    # Neutral: aligned but tight or inconclusive
    else:
        bucket = "neutral"

    return bucket, spread_pips, aligned, compressing, compression_count


def compute_conviction(
    *,
    m5_df: Optional[pd.DataFrame],
    m1_df: Optional[pd.DataFrame],
    pip_size: float,
    is_bull: bool,
    enabled: bool = False,
    base_lots: float = 0.05,
    min_lots: float = 0.01,
    max_lots: float = 1.0,
) -> ConvictionResult:
    """Compute conviction-based lot size from M5/M1 EMA data.

    Parameters
    ----------
    m5_df : DataFrame with 'close' column (completed M5 bars)
    m1_df : DataFrame with 'close' column (completed M1 bars)
    pip_size : instrument pip size (0.01 for USDJPY)
    is_bull : True if M5 trend is bullish
    enabled : master toggle
    base_lots : starting lot size before conviction scaling
    min_lots : minimum lot size floor
    max_lots : maximum lot size ceiling (from risk config)

    Returns
    -------
    ConvictionResult with all diagnostic data.
    """
    safe_max_lots = max(0.0, float(max_lots))
    safe_min_lots = max(0.0, float(min_lots))
    safe_base_lots = max(0.0, float(base_lots))
    if safe_max_lots > 0.0:
        safe_min_lots = min(safe_min_lots, safe_max_lots)
        safe_base_lots = min(safe_base_lots, safe_max_lots)
        safe_base_lots = max(safe_base_lots, safe_min_lots)
    else:
        safe_min_lots = 0.0
        safe_base_lots = 0.0

    result = ConvictionResult(
        enabled=enabled,
        base_lots=safe_base_lots,
        min_lots=safe_min_lots,
        max_lots=safe_max_lots,
        m5_trend="bull" if is_bull else "bear",
    )

    if not enabled:
        result.conviction_lots = safe_max_lots  # fallback to runtime ceiling when disabled
        return result

    # Need sufficient data
    if m5_df is None or m5_df.empty or len(m5_df) < 22:
        result.conviction_lots = safe_base_lots
        return result
    if m1_df is None or m1_df.empty or len(m1_df) < 10:
        result.conviction_lots = safe_base_lots
        return result

    m5_close = m5_df["close"].astype(float)
    m1_close = m1_df["close"].astype(float)

    # Compute buckets
    m5_bucket, m5_spread, m5_slope, m5_slope_aligned, m5_disagrees = _compute_m5_bucket(
        m5_close, pip_size, is_bull
    )
    m1_bucket, m1_spread, m1_aligned, m1_compressing, m1_compress_count = _compute_m1_bucket(
        m1_close, pip_size, is_bull
    )

    # Look up multiplier
    multiplier = MULTIPLIER_MATRIX.get((m5_bucket, m1_bucket), 1.0)

    # Calculate lots
    raw_lots = safe_base_lots * multiplier
    conviction_lots = max(safe_min_lots, min(safe_max_lots, raw_lots))

    # Round to 2 decimal places (standard lot precision)
    conviction_lots = round(conviction_lots, 2)

    result.m5_bucket = m5_bucket
    result.m1_bucket = m1_bucket
    result.multiplier = multiplier
    result.conviction_lots = conviction_lots
    result.m5_spread_pips = round(m5_spread, 3)
    result.m5_slope_pips_per_bar = round(m5_slope, 3)
    result.m5_slope_aligned = m5_slope_aligned
    result.m5_slope_disagrees_with_ema21 = m5_disagrees
    result.m1_spread_pips = round(m1_spread, 3)
    result.m1_aligned = m1_aligned
    result.m1_compressing = m1_compressing
    result.m1_compression_bars_count = m1_compress_count

    return result


def conviction_snapshot(result: ConvictionResult) -> dict:
    """Convert ConvictionResult to a JSON-serializable dict for dashboard/API."""
    return {
        "enabled": result.enabled,
        "m5_bucket": result.m5_bucket,
        "m1_bucket": result.m1_bucket,
        "multiplier": result.multiplier,
        "conviction_lots": result.conviction_lots,
        "base_lots": result.base_lots,
        "min_lots": result.min_lots,
        "max_lots": result.max_lots,
        "m5_spread_pips": result.m5_spread_pips,
        "m5_slope_pips_per_bar": result.m5_slope_pips_per_bar,
        "m5_slope_aligned": result.m5_slope_aligned,
        "m5_slope_disagrees_with_ema21": result.m5_slope_disagrees_with_ema21,
        "m1_spread_pips": result.m1_spread_pips,
        "m1_aligned": result.m1_aligned,
        "m1_compressing": result.m1_compressing,
        "m1_compression_bars_count": result.m1_compression_bars_count,
        "m5_trend": result.m5_trend,
    }
