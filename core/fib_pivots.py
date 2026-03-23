"""Shared Fibonacci pivot-point calculation."""
from __future__ import annotations

from typing import Optional

import pandas as pd


# Ordered list of all fib level names from lowest to highest
FIB_LEVEL_NAMES = ("S3", "S2", "S1", "PP", "R1", "R2", "R3")
ROLLING_INTRADAY_FIB_TIMEFRAMES = ("M15", "M5")
FIXED_INTRADAY_FIB_TIMEFRAMES = ("H1", "H2", "H3")
SUPPORTED_INTRADAY_FIB_TIMEFRAMES = ROLLING_INTRADAY_FIB_TIMEFRAMES + FIXED_INTRADAY_FIB_TIMEFRAMES


def is_fixed_intraday_fib_timeframe(timeframe: str) -> bool:
    """Return True when the timeframe uses previous completed candle pivots."""
    return str(timeframe).upper() in FIXED_INTRADAY_FIB_TIMEFRAMES


def _standard_fib_levels_from_hlc(high: float, low: float, close: float) -> Optional[dict[str, float]]:
    """Compute standard Fibonacci pivot levels using PP/R1..R3/S1..S3 keys."""
    rng = float(high) - float(low)
    if rng <= 0:
        return None
    pp = (float(high) + float(low) + float(close)) / 3.0
    return {
        "PP": pp,
        "R1": pp + 0.382 * rng,
        "R2": pp + 0.618 * rng,
        "R3": pp + 1.000 * rng,
        "S1": pp - 0.382 * rng,
        "S2": pp - 0.618 * rng,
        "S3": pp - 1.000 * rng,
    }


def _prepare_time_index(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Normalize OHLC data to a UTC DatetimeIndex."""
    if df is None or df.empty:
        return None
    out = df.copy()
    if "time" in out.columns:
        idx = pd.to_datetime(out["time"], utc=True, errors="coerce")
        mask = idx.notna()
        out = out.loc[mask].copy()
        if out.empty:
            return None
        out.index = idx[mask]
    elif isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
        out = out.loc[out.index.notna()].copy()
        if out.empty:
            return None
    else:
        return None
    return out.sort_index()


def resample_intraday_ohlc(df: pd.DataFrame, timeframe: str) -> Optional[pd.DataFrame]:
    """Resample intraday OHLC data to the requested timeframe."""
    timeframe = str(timeframe).upper()
    rule_map = {
        "H1": "1h",
        "H2": "2h",
        "H3": "3h",
    }
    rule = rule_map.get(timeframe)
    if rule is None:
        return None

    prepared = _prepare_time_index(df)
    if prepared is None or prepared.empty:
        return None

    # Drop the potentially forming base bar before resampling.
    completed_base = prepared.iloc[:-1] if len(prepared) > 1 else prepared
    if completed_base.empty:
        return None

    resampled = (
        completed_base[["open", "high", "low", "close"]]
        .resample(rule)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna(subset=["open", "high", "low", "close"])
    )
    return resampled if not resampled.empty else None


def compute_previous_candle_fib_levels(df: pd.DataFrame) -> Optional[dict[str, float]]:
    """Compute Fibonacci pivots from the previous completed candle in the DataFrame."""
    if df is None or df.empty:
        return None
    completed = df.iloc[:-1] if len(df) > 1 else df
    if completed.empty:
        return None
    candle = completed.iloc[-1]
    return _standard_fib_levels_from_hlc(
        float(candle["high"]),
        float(candle["low"]),
        float(candle["close"]),
    )


def compute_rolling_intraday_fib_levels(
    df: pd.DataFrame,
    lookback_bars: int = 16,
) -> Optional[dict[str, float]]:
    """Compute Fibonacci pivot levels from a rolling intraday window.

    Uses the highest high and lowest low over the last *lookback_bars* completed
    bars (excludes the current in-progress bar). The pivot (PP) uses the
    standard close-based formula from that rolling source window:

    ``PP = (rolling_high + rolling_low + latest_close) / 3``

    Parameters
    ----------
    df : pd.DataFrame
        OHLC DataFrame for an intraday timeframe (e.g. M15, M5).
        Must contain ``high``, ``low``, and ``close`` columns.
    lookback_bars : int
        Number of completed bars to use for the rolling window.

    Returns
    -------
    dict or None
        Keys: PP, R1, R2, R3, S1, S2, S3.  None if insufficient data.

    Level convention (consistent with ``compute_daily_fib_pivots``):
        PP = (rolling_high + rolling_low + latest_close) / 3
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
    latest_close = float(window["close"].iloc[-1])

    rng = rolling_high - rolling_low
    if rng <= 0:
        return None

    return _standard_fib_levels_from_hlc(rolling_high, rolling_low, latest_close)


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
