"""
Weekly macro signals for the Swing-Macro strategy.

Uses 5-day returns on oil and DXY (inverted EUR/USD) to determine
macro alignment. Gold and silver are dropped — they added noise in
prior testing.

Causal enforcement: only uses bars that are fully complete before
the query timestamp.

Components:
  - MacroDirection: UP / DOWN / NEUTRAL enum
  - MacroBias: Combined reading from oil + DXY
  - WeeklyMacroSignal: Computes bias from cross-asset daily/H1 data
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional, Tuple
import bisect


class MacroDirection(Enum):
    UP = "UP"
    DOWN = "DOWN"
    NEUTRAL = "NEUTRAL"


class MacroBias(Enum):
    """Combined oil + DXY reading."""

    LONG = "LONG"  # Both support long USDJPY
    LEAN_LONG = "LEAN_LONG"  # One supports, one neutral
    NEUTRAL = "NEUTRAL"  # Conflicting or both neutral
    LEAN_SHORT = "LEAN_SHORT"
    SHORT = "SHORT"  # Both support short USDJPY


@dataclass(frozen=True)
class MacroReading:
    """Complete macro state at a point in time."""

    timestamp: datetime
    oil_direction: MacroDirection
    oil_return_5d: Optional[float]  # 5-day return as decimal (0.02 = 2%)
    dxy_direction: MacroDirection
    dxy_return_5d: Optional[float]
    bias: MacroBias
    tradeable: bool  # True if bias is not NEUTRAL


def compute_5d_return(closes: List[float]) -> Optional[float]:
    """Compute 5-day return from a list of daily closes.

    Expects at least 6 closes (today + 5 prior).
    Returns (close_today - close_5_days_ago) / close_5_days_ago
    """
    if len(closes) < 6:
        return None
    return (closes[-1] - closes[-5]) / closes[-5]


def direction_from_return(ret: Optional[float], threshold: float = 0.002) -> MacroDirection:
    """Convert a 5-day return into a direction.

    threshold=0.002 means moves < 0.2% in either direction are NEUTRAL.
    This filters out noise — a 0.1% move in oil over a week is meaningless.
    """
    if ret is None:
        return MacroDirection.NEUTRAL
    if ret > threshold:
        return MacroDirection.UP
    if ret < -threshold:
        return MacroDirection.DOWN
    return MacroDirection.NEUTRAL


def combine_bias(oil: MacroDirection, dxy: MacroDirection) -> MacroBias:
    """Combine oil and DXY signals into a USDJPY bias.

    Oil UP → JPY weakens → USDJPY rises → LONG signal
    DXY UP → USD strengthens → USDJPY rises → LONG signal

    Agreement matrix:
        Oil UP   + DXY UP   → LONG
        Oil UP   + DXY NEU  → LEAN_LONG
        Oil UP   + DXY DOWN → NEUTRAL (conflict)
        Oil NEU  + DXY UP   → LEAN_LONG
        Oil NEU  + DXY NEU  → NEUTRAL
        Oil NEU  + DXY DOWN → LEAN_SHORT
        Oil DOWN + DXY UP   → NEUTRAL (conflict)
        Oil DOWN + DXY NEU  → LEAN_SHORT
        Oil DOWN + DXY DOWN → SHORT
    """
    if oil == MacroDirection.UP and dxy == MacroDirection.UP:
        return MacroBias.LONG
    if oil == MacroDirection.DOWN and dxy == MacroDirection.DOWN:
        return MacroBias.SHORT

    if oil == MacroDirection.UP and dxy == MacroDirection.NEUTRAL:
        return MacroBias.LEAN_LONG
    if oil == MacroDirection.NEUTRAL and dxy == MacroDirection.UP:
        return MacroBias.LEAN_LONG

    if oil == MacroDirection.DOWN and dxy == MacroDirection.NEUTRAL:
        return MacroBias.LEAN_SHORT
    if oil == MacroDirection.NEUTRAL and dxy == MacroDirection.DOWN:
        return MacroBias.LEAN_SHORT

    # Conflicts (UP+DOWN or DOWN+UP) and NEUTRAL+NEUTRAL
    return MacroBias.NEUTRAL


def bias_supports_long(bias: MacroBias) -> bool:
    return bias in (MacroBias.LONG, MacroBias.LEAN_LONG)


def bias_supports_short(bias: MacroBias) -> bool:
    return bias in (MacroBias.SHORT, MacroBias.LEAN_SHORT)


class WeeklyMacroSignal:
    """Computes macro bias from pre-loaded cross-asset data.

    Data format: list of (datetime, close) tuples, sorted by time.
    Uses bisect for efficient causal lookups.

    Causal rule for daily bars: a daily bar dated Jan 15 is only
    available after 22:00 UTC on Jan 15 (market close).
    """

    def __init__(
        self,
        oil_daily: List[Tuple[datetime, float]],
        eurusd_daily: List[Tuple[datetime, float]],
        return_threshold: float = 0.002,
    ):
        """
        Args:
            oil_daily: List of (timestamp, close) for Brent crude, daily.
            eurusd_daily: List of (timestamp, close) for EUR/USD, daily.
                          Will be inverted to approximate DXY.
            return_threshold: Minimum 5-day return to count as directional.
        """
        self._oil_times = [t for t, _ in oil_daily]
        self._oil_closes = [c for _, c in oil_daily]
        self._eurusd_times = [t for t, _ in eurusd_daily]
        self._eurusd_closes = [c for _, c in eurusd_daily]
        self._threshold = return_threshold

        # Cache: avoid recomputing if timestamp hasn't changed
        self._cache_time: Optional[datetime] = None
        self._cache_reading: Optional[MacroReading] = None

    def _available_index(self, times: List[datetime], query_time: datetime) -> int:
        """Return the number of bars available (completed) at query_time.

        A daily bar with timestamp T is complete at T + 22 hours (22:00 UTC).
        We use bisect to find how many bars have completion time <= query_time.
        """
        query_date = query_time.date()
        query_hour = query_time.hour

        idx = bisect.bisect_right(times, query_time)

        available = 0
        for i in range(idx):
            bar_date = times[i].date()
            if bar_date < query_date:
                available = i + 1
            elif bar_date == query_date and query_hour >= 22:
                available = i + 1

        return available

    def _get_recent_closes(
        self,
        times: List[datetime],
        closes: List[float],
        query_time: datetime,
        n: int = 6,
    ) -> List[float]:
        """Get the last n completed daily closes as of query_time."""
        avail = self._available_index(times, query_time)
        if avail < n:
            return []
        return closes[avail - n : avail]

    def compute(self, query_time: datetime) -> MacroReading:
        """Compute the macro reading at a given time.

        Returns cached result if query_time hasn't changed.
        """
        if self._cache_time == query_time and self._cache_reading is not None:
            return self._cache_reading

        # Oil: 5-day return on Brent daily closes
        oil_closes = self._get_recent_closes(
            self._oil_times, self._oil_closes, query_time, n=6
        )
        oil_ret = compute_5d_return(oil_closes) if len(oil_closes) >= 6 else None
        oil_dir = direction_from_return(oil_ret, self._threshold)

        # DXY proxy: invert EUR/USD
        # EUR/USD up → USD weak → DXY down. So DXY return ≈ -EURUSD return.
        eurusd_closes = self._get_recent_closes(
            self._eurusd_times, self._eurusd_closes, query_time, n=6
        )
        if len(eurusd_closes) >= 6:
            eurusd_ret = compute_5d_return(eurusd_closes)
            dxy_ret = -eurusd_ret if eurusd_ret is not None else None
        else:
            eurusd_ret = None
            dxy_ret = None
        dxy_dir = direction_from_return(dxy_ret, self._threshold)

        bias = combine_bias(oil_dir, dxy_dir)
        tradeable = bias != MacroBias.NEUTRAL

        reading = MacroReading(
            timestamp=query_time,
            oil_direction=oil_dir,
            oil_return_5d=oil_ret,
            dxy_direction=dxy_dir,
            dxy_return_5d=dxy_ret,
            bias=bias,
            tradeable=tradeable,
        )

        self._cache_time = query_time
        self._cache_reading = reading
        return reading
