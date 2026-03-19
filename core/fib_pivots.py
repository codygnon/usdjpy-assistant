"""Shared Fibonacci pivot-point calculation."""
from __future__ import annotations

from typing import Optional

import pandas as pd


# Ordered list of all fib level names from lowest to highest
FIB_LEVEL_NAMES = ("S3", "S2", "S1", "PP", "R1", "R2", "R3")


def compute_rolling_intraday_fib_levels(
    df: pd.DataFrame,
    lookback_bars: int = 16,
) -> Optional[dict[str, float]]:
    """Compute Fibonacci pivot levels from a rolling intraday high/low window.

    Uses the highest high and lowest low over the last *lookback_bars* completed
    bars (excludes the current in-progress bar).  The midpoint of that range
    serves as the pivot (PP) and S/R levels are placed at standard fib ratios.

    Parameters
    ----------
    df : pd.DataFrame
        OHLC DataFrame for an intraday timeframe (e.g. M15, M5).
        Must contain ``high`` and ``low`` columns.
    lookback_bars : int
        Number of completed bars to use for the rolling window.

    Returns
    -------
    dict or None
        Keys: PP, R1, R2, R3, S1, S2, S3.  None if insufficient data.

    Level convention (consistent with ``compute_daily_fib_pivots``):
        PP = midpoint of rolling range
        R1 = PP + 0.382 * range,  S1 = PP - 0.382 * range
        R2 = PP + 0.618 * range,  S2 = PP - 0.618 * range
        R3 = PP + 1.000 * range,  S3 = PP - 1.000 * range
    """
    if df is None or df.empty:
        return None
    # Use only completed bars (drop the last row which may be in-progress)
    completed = df.iloc[:-1] if len(df) > 1 else df
    if len(completed) < max(1, lookback_bars):
        return None

    window = completed.iloc[-lookback_bars:]
    rolling_high = float(window["high"].max())
    rolling_low = float(window["low"].min())

    rng = rolling_high - rolling_low
    if rng <= 0:
        return None

    pp = (rolling_high + rolling_low) / 2.0
    return {
        "PP": pp,
        "R1": pp + 0.382 * rng,
        "R2": pp + 0.618 * rng,
        "R3": pp + 1.000 * rng,
        "S1": pp - 0.382 * rng,
        "S2": pp - 0.618 * rng,
        "S3": pp - 1.000 * rng,
    }


def resolve_fib_level(levels: dict[str, float], name: str) -> Optional[float]:
    """Look up a fib level by name (PP, S1, R1, etc.). Returns None if missing."""
    return levels.get(name)


def compute_daily_fib_pivots(
    prev_day_high: float, prev_day_low: float, prev_day_close: float
) -> dict:
    """Compute daily Fibonacci pivot levels {P, R1, R2, R3, S1, S2, S3}.

    Parameters
    ----------
    prev_day_high : float
        Previous trading day high.
    prev_day_low : float
        Previous trading day low.
    prev_day_close : float
        Previous trading day close.

    Returns
    -------
    dict
        Keys: P, R1, R2, R3, S1, S2, S3.
    """
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
