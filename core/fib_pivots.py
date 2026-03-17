"""Shared Fibonacci pivot-point calculation."""
from __future__ import annotations


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
