"""
Daily bar aggregation for the daily_trend strategy (22:00 UTC forex day).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Any, Optional

UTC = timezone.utc


@dataclass(frozen=True)
class DailyBar:
    """One completed forex daily bar (22:00 UTC open to 22:00 UTC close)."""

    timestamp: datetime  # Start of the daily period (22:00 UTC)
    open: float
    high: float
    low: float
    close: float
    bar_index: int


def get_daily_boundary(timestamp: datetime) -> datetime:
    """Return the START of the forex daily bar containing this timestamp.

    Forex daily bars run 22:00 UTC to 22:00 UTC.
    - 2024-01-15 03:00 UTC → 2024-01-14 22:00 UTC
    - 2024-01-15 21:59 UTC → 2024-01-14 22:00 UTC
    - 2024-01-15 22:00 UTC → 2024-01-15 22:00 UTC
    - 2024-01-15 22:01 UTC → 2024-01-15 22:00 UTC
    """
    if timestamp.tzinfo is None:
        dt = timestamp.replace(tzinfo=UTC)
    else:
        dt = timestamp.astimezone(UTC)
    if dt.hour >= 22:
        return dt.replace(hour=22, minute=0, second=0, microsecond=0)
    d = dt.date() - timedelta(days=1)
    return datetime.combine(d, time(22, 0, tzinfo=UTC))


class IncrementalDailyBarBuilder:
    """Aggregates 1-minute bars into daily bars (22:00 UTC boundaries).

    Same completion semantics as IncrementalBar4HBuilder: when the boundary
    changes, returns the completed bar for the *previous* period and starts
    the new period with the current bar's OHLC.
    """

    def __init__(self) -> None:
        self._current_boundary: Optional[datetime] = None
        self._open: float = 0.0
        self._high: float = 0.0
        self._low: float = 0.0
        self._close: float = 0.0
        self._bar_count: int = 0
        self._next_bar_index: int = 0

    def update(self, bar: Any) -> Optional[DailyBar]:
        """Feed one 1M bar (BarView-like: timestamp, mid_open/high/low/close)."""
        ts = bar.timestamp
        if isinstance(ts, datetime):
            boundary = get_daily_boundary(ts)
        else:
            boundary = get_daily_boundary(datetime.fromisoformat(str(ts).replace("Z", "+00:00")))

        o = float(bar.mid_open)
        h = float(bar.mid_high)
        l = float(bar.mid_low)
        c = float(bar.mid_close)

        if self._current_boundary is None:
            self._current_boundary = boundary
            self._open = o
            self._high = h
            self._low = l
            self._close = c
            self._bar_count = 1
            return None

        if boundary != self._current_boundary:
            idx = self._next_bar_index
            self._next_bar_index += 1
            completed = DailyBar(
                timestamp=self._current_boundary,
                open=self._open,
                high=self._high,
                low=self._low,
                close=self._close,
                bar_index=idx,
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
