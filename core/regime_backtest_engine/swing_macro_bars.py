"""
4H bar aggregation and technical computations for the Swing-Macro strategy.

Components:
  - Bar4H: Completed 4-hour bar dataclass
  - IncrementalBar4HBuilder: Builds 4H bars from 1M bars one at a time
  - IncrementalEMA: Streaming EMA with slope tracking
  - IncrementalATR: Streaming Wilder ATR
  - find_recent_swing_low/high: For trailing stops
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

UTC = timezone.utc


@dataclass(frozen=True)
class Bar4H:
    """A completed 4-hour OHLC bar."""

    timestamp: datetime  # Bar open time (UTC), aligned to 00/04/08/12/16/20
    open: float
    high: float
    low: float
    close: float
    bar_count: int  # Number of 1M bars composing this bar


def get_4h_boundary(dt: datetime) -> datetime:
    """Return the 4H bar open time for any datetime.

    Boundaries: 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC.
    Example: 14:37 UTC -> 12:00 UTC
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    else:
        dt = dt.astimezone(UTC)
    hour_block = (dt.hour // 4) * 4
    return dt.replace(hour=hour_block, minute=0, second=0, microsecond=0)


class IncrementalBar4HBuilder:
    """Builds 4H bars incrementally from 1M bars.

    Usage:
        builder = IncrementalBar4HBuilder()
        for bar in minute_bars:
            completed = builder.update(bar.timestamp, bar.open, bar.high, bar.low, bar.close)
            if completed is not None:
                process(completed)  # A 4H bar just closed
    """

    def __init__(self) -> None:
        self._current_boundary: Optional[datetime] = None
        self._open: float = 0.0
        self._high: float = 0.0
        self._low: float = 0.0
        self._close: float = 0.0
        self._bar_count: int = 0

    def update(self, timestamp: datetime, o: float, h: float, l: float, c: float) -> Optional[Bar4H]:
        """Feed one 1M bar. Returns completed Bar4H when a 4H boundary is crossed."""
        boundary = get_4h_boundary(timestamp)

        if self._current_boundary is None:
            self._current_boundary = boundary
            self._open = o
            self._high = h
            self._low = l
            self._close = c
            self._bar_count = 1
            return None

        if boundary != self._current_boundary:
            completed = Bar4H(
                timestamp=self._current_boundary,
                open=self._open,
                high=self._high,
                low=self._low,
                close=self._close,
                bar_count=self._bar_count,
            )
            self._current_boundary = boundary
            self._open = o
            self._high = h
            self._low = l
            self._close = c
            self._bar_count = 1
            return completed

        self._high = max(self._high, h)
        self._low = min(self._low, l)
        self._close = c
        self._bar_count += 1
        return None


class IncrementalEMA:
    """Streaming EMA computation with slope tracking.

    Seeds with SMA over the first `period` values, then applies:
        EMA = (value - prev_ema) * k + prev_ema
    where k = 2 / (period + 1).
    """

    def __init__(self, period: int) -> None:
        if period < 1:
            raise ValueError("period must be >= 1")
        self.period = period
        self._multiplier = 2.0 / (period + 1)
        self._seed_values: list[float] = []
        self._ema: Optional[float] = None
        self._prev_ema: Optional[float] = None
        self._count: int = 0

    def update(self, value: float) -> Optional[float]:
        """Feed one value. Returns current EMA or None if insufficient data."""
        self._count += 1
        self._prev_ema = self._ema

        if self._ema is None:
            self._seed_values.append(value)
            if len(self._seed_values) == self.period:
                self._ema = sum(self._seed_values) / self.period
                self._seed_values.clear()
                return self._ema
            return None

        assert self._ema is not None
        self._ema = (value - self._ema) * self._multiplier + self._ema
        return self._ema

    @property
    def value(self) -> Optional[float]:
        """Current EMA value, or None if not yet initialized."""
        return self._ema

    @property
    def prev_value(self) -> Optional[float]:
        """EMA value before the last update. For slope computation."""
        return self._prev_ema

    @property
    def slope(self) -> Optional[float]:
        """Current minus previous EMA. Positive = rising."""
        if self._ema is not None and self._prev_ema is not None:
            return self._ema - self._prev_ema
        return None


class IncrementalATR:
    """Streaming Wilder ATR computation.

    First bar establishes prev_close. After `period` true ranges,
    initializes ATR with simple average. Subsequent updates use
    Wilder smoothing: ATR = (prev_ATR * (period-1) + TR) / period
    """

    def __init__(self, period: int = 14) -> None:
        if period < 1:
            raise ValueError("period must be >= 1")
        self.period = period
        self._prev_close: Optional[float] = None
        self._true_ranges: list[float] = []
        self._atr: Optional[float] = None
        self._initialized: bool = False

    def update(self, bar: Bar4H) -> Optional[float]:
        """Feed one completed 4H bar. Returns ATR or None if insufficient data."""
        if self._prev_close is None:
            self._prev_close = bar.close
            return None

        tr = max(
            bar.high - bar.low,
            abs(bar.high - self._prev_close),
            abs(bar.low - self._prev_close),
        )
        self._prev_close = bar.close

        if not self._initialized:
            self._true_ranges.append(tr)
            if len(self._true_ranges) == self.period:
                self._atr = sum(self._true_ranges) / self.period
                self._initialized = True
                self._true_ranges.clear()
                return self._atr
            return None

        assert self._atr is not None
        self._atr = (self._atr * (self.period - 1) + tr) / self.period
        return self._atr

    @property
    def value(self) -> Optional[float]:
        """Current ATR in price units, or None."""
        return self._atr


def find_recent_swing_low(bars: list[Bar4H], lookback: int = 5) -> Optional[float]:
    """Lowest low in the last `lookback` bars. For trailing stop on longs."""
    if len(bars) < lookback:
        return None
    return min(bar.low for bar in bars[-lookback:])


def find_recent_swing_high(bars: list[Bar4H], lookback: int = 5) -> Optional[float]:
    """Highest high in the last `lookback` bars. For trailing stop on shorts."""
    if len(bars) < lookback:
        return None
    return max(bar.high for bar in bars[-lookback:])
