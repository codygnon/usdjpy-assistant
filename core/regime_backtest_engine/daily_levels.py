"""
Daily structural level computation.

Computes PDH/PDL, PWH/PWL, PMH/PML, round numbers, and level clusters
from daily bar history. Used by multiple strategy adapters.

Rules:
- All lookbacks use COMPLETED bars only
- Current day's data never leaks into "prior day" computations
- Levels recomputed once per day, not on every bar
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any, Optional

from .synthetic_bars import BarDaily


def compute_pdh_pdl(daily_bars: list[BarDaily], current_day_index: int) -> tuple[Optional[float], Optional[float]]:
    if current_day_index < 1 or not daily_bars:
        return None, None
    if current_day_index >= len(daily_bars):
        return None, None
    prev = daily_bars[current_day_index - 1]
    return float(prev.high), float(prev.low)


def _monday_prev_week(ref: date) -> date:
    monday_this = ref - timedelta(days=ref.weekday())
    return monday_this - timedelta(days=7)


def compute_pwh_pwl(daily_bars: list[BarDaily], current_day_index: int) -> tuple[Optional[float], Optional[float]]:
    if current_day_index < 1 or not daily_bars or current_day_index >= len(daily_bars):
        return None, None
    current_td = daily_bars[current_day_index].trading_day
    mon0 = _monday_prev_week(current_td)
    week_dates = {mon0 + timedelta(days=i) for i in range(5)}
    highs: list[float] = []
    lows: list[float] = []
    for i in range(current_day_index):
        b = daily_bars[i]
        if b.trading_day in week_dates:
            highs.append(float(b.high))
            lows.append(float(b.low))
    if not highs:
        return None, None
    return max(highs), min(lows)


def _prior_calendar_month(y: int, m: int) -> tuple[int, int]:
    if m > 1:
        return y, m - 1
    return y - 1, 12


def compute_pmh_pml(daily_bars: list[BarDaily], current_day_index: int) -> tuple[Optional[float], Optional[float]]:
    if current_day_index < 1 or not daily_bars or current_day_index >= len(daily_bars):
        return None, None
    current_td = daily_bars[current_day_index].trading_day
    py, pm = _prior_calendar_month(current_td.year, current_td.month)
    highs: list[float] = []
    lows: list[float] = []
    for i in range(current_day_index):
        b = daily_bars[i]
        if b.trading_day.year == py and b.trading_day.month == pm:
            highs.append(float(b.high))
            lows.append(float(b.low))
    if not highs:
        return None, None
    return max(highs), min(lows)


def compute_round_levels(price: float, radius_pips: float = 50, pip_size: float = 0.01) -> list[float]:
    if not math.isfinite(price) or not math.isfinite(radius_pips) or not math.isfinite(pip_size):
        return []
    if pip_size <= 0:
        return []
    radius = float(radius_pips) * float(pip_size)
    lo = float(price) - radius
    hi = float(price) + radius
    start = int(math.floor(lo))
    end = int(math.ceil(hi)) + 1
    out: list[float] = []
    for k in range(start, end + 1):
        for frac in (0.0, 0.5):
            lvl = float(k) + frac
            if lo - 1e-12 <= lvl <= hi + 1e-12:
                out.append(lvl)
    return sorted(set(out))


def find_nearest_level(
    price: float, levels: list[float], pip_size: float = 0.01
) -> tuple[Optional[float], Optional[float]]:
    if not levels or not math.isfinite(price) or pip_size <= 0:
        return None, None
    best: Optional[float] = None
    best_d = float("inf")
    for lv in levels:
        if not math.isfinite(lv):
            continue
        d = abs(float(price) - float(lv)) / float(pip_size)
        if d < best_d:
            best_d = d
            best = float(lv)
    if best is None:
        return None, None
    return best, best_d


def is_level_cluster(levels: list[float], cluster_radius_pips: float = 20, pip_size: float = 0.01) -> bool:
    if len(levels) < 2 or pip_size <= 0 or cluster_radius_pips < 0:
        return False
    thr = float(cluster_radius_pips) * float(pip_size)
    n = len(levels)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = float(levels[i]), float(levels[j])
            if not math.isfinite(a) or not math.isfinite(b):
                continue
            if abs(a - b) <= thr + 1e-12:
                return True
    return False


def get_all_levels(
    daily_bars: list[BarDaily],
    current_day_index: int,
    current_price: float,
    *,
    round_radius_pips: float = 50,
    cluster_radius_pips: float = 20,
    pip_size: float = 0.01,
) -> dict[str, Any]:
    pdh, pdl = compute_pdh_pdl(daily_bars, current_day_index)
    pwh, pwl = compute_pwh_pwl(daily_bars, current_day_index)
    pmh, pml = compute_pmh_pml(daily_bars, current_day_index)
    round_levels = compute_round_levels(current_price, radius_pips=round_radius_pips, pip_size=pip_size)
    struct = [x for x in (pdh, pdl, pwh, pwl, pmh, pml) if x is not None]
    combined = struct + list(round_levels)
    nearest_level, nearest_distance_pips = find_nearest_level(current_price, combined, pip_size=pip_size)
    has_cluster = is_level_cluster(combined, cluster_radius_pips=cluster_radius_pips, pip_size=pip_size)
    return {
        "pdh": pdh,
        "pdl": pdl,
        "pwh": pwh,
        "pwl": pwl,
        "pmh": pmh,
        "pml": pml,
        "round_levels": round_levels if round_levels else None,
        "nearest_level": nearest_level,
        "nearest_distance_pips": nearest_distance_pips,
        "has_cluster": has_cluster,
    }
