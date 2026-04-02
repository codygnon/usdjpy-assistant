"""
Synthetic bar aggregation from 1M bars.

Aggregates 1-minute bars into 5M, 15M, and daily bars on fixed clock
boundaries. Used by multiple strategy adapters.

Rules:
- No lookahead: a synthetic bar is only available AFTER its last
  constituent 1M bar completes
- Only COMPLETE bars are emitted (all constituent 1M bars present)
- Gaps are not interpolated
- Each synthetic bar tracks the index range of its constituent 1M bars
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional, Sequence


@dataclass
class Bar5M:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float]
    bar_index_start: int
    bar_index_end: int
    complete: bool


@dataclass
class Bar15M:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float]
    bar_index_start: int
    bar_index_end: int
    complete: bool


@dataclass
class BarDaily:
    trading_day: date
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float]
    bar_index_start: int
    bar_index_end: int
    bar_count: int


def _to_utc_datetime(ts: Any) -> datetime:
    if ts is None:
        raise TypeError("bar timestamp is required")
    if isinstance(ts, datetime):
        dt = ts
    else:
        try:
            import pandas as pd

            if isinstance(ts, pd.Timestamp):
                dt = ts.to_pydatetime()
            else:
                dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except ImportError:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.replace(second=0, microsecond=0)


def _bar_ohlc(bar: Any) -> tuple[float, float, float, float]:
    if hasattr(bar, "mid_open"):
        return (
            float(bar.mid_open),
            float(bar.mid_high),
            float(bar.mid_low),
            float(bar.mid_close),
        )
    return (float(bar.open), float(bar.high), float(bar.low), float(bar.close))


def _bar_volume(bar: Any) -> Any:
    if hasattr(bar, "tick_volume"):
        return getattr(bar, "tick_volume")
    if hasattr(bar, "volume"):
        return getattr(bar, "volume")
    return None


def _volume_sum(bars: Sequence[Any]) -> Optional[float]:
    total = 0.0
    for b in bars:
        v = _bar_volume(b)
        if v is None:
            return None
        try:
            total += float(v)
        except (TypeError, ValueError):
            return None
    return total


def _minute_key(dt: datetime) -> datetime:
    return dt.replace(second=0, microsecond=0)


def _five_minute_anchor_close_label(ts: datetime) -> datetime:
    """Anchor for the 5M window that this 1M close-timestamp belongs to."""
    m = ts.minute
    h = ts.hour
    d = ts.date()
    tz = timezone.utc
    base_midnight = datetime(d.year, d.month, d.day, tzinfo=tz)
    hour_base = base_midnight + timedelta(hours=h)

    if m == 0:
        prev = ts - timedelta(hours=1)
        return datetime(prev.year, prev.month, prev.day, prev.hour, 55, tzinfo=tz)
    if 1 <= m <= 5:
        return hour_base.replace(minute=0)
    if 6 <= m <= 10:
        return hour_base.replace(minute=5)
    if 11 <= m <= 15:
        return hour_base.replace(minute=10)
    if 16 <= m <= 20:
        return hour_base.replace(minute=15)
    if 21 <= m <= 25:
        return hour_base.replace(minute=20)
    if 26 <= m <= 30:
        return hour_base.replace(minute=25)
    if 31 <= m <= 35:
        return hour_base.replace(minute=30)
    if 36 <= m <= 40:
        return hour_base.replace(minute=35)
    if 41 <= m <= 45:
        return hour_base.replace(minute=40)
    if 46 <= m <= 50:
        return hour_base.replace(minute=45)
    if 51 <= m <= 55:
        return hour_base.replace(minute=50)
    # 56-59
    return hour_base.replace(minute=55)


def _required_1m_keys_5m(anchor: datetime) -> list[datetime]:
    start = anchor + timedelta(minutes=1)
    return [start + timedelta(minutes=i) for i in range(5)]


def _fifteen_minute_anchor_close_label(ts: datetime) -> datetime:
    m = ts.minute
    h = ts.hour
    d = ts.date()
    tz = timezone.utc
    base_midnight = datetime(d.year, d.month, d.day, tzinfo=tz)
    hour_base = base_midnight + timedelta(hours=h)

    if m == 0:
        prev = ts - timedelta(hours=1)
        return datetime(prev.year, prev.month, prev.day, prev.hour, 45, tzinfo=tz)
    if 1 <= m <= 15:
        return hour_base.replace(minute=0)
    if 16 <= m <= 30:
        return hour_base.replace(minute=15)
    if 31 <= m <= 45:
        return hour_base.replace(minute=30)
    return hour_base.replace(minute=45)


def _required_1m_keys_15m(anchor: datetime) -> list[datetime]:
    start = anchor + timedelta(minutes=1)
    return [start + timedelta(minutes=i) for i in range(15)]


def _aggregate_window(
    key_to_bar_idx: dict[datetime, tuple[Any, int]], required_keys: list[datetime]
) -> Optional[tuple[list[tuple[Any, int]], datetime]]:
    seq: list[tuple[Any, int]] = []
    for k in required_keys:
        mk = _minute_key(k)
        if mk not in key_to_bar_idx:
            return None
        seq.append(key_to_bar_idx[mk])
    first_ts = _to_utc_datetime(seq[0][0].timestamp)
    return seq, first_ts


def _emit_from_sequence(seq: list[tuple[Any, int]], first_ts: datetime, cls: type) -> Bar5M | Bar15M:
    bars = [b for b, _ in seq]
    idxs = [i for _, i in seq]
    o, _, _, _ = _bar_ohlc(bars[0])
    _, _, _, c = _bar_ohlc(bars[-1])
    highs = [_bar_ohlc(b)[1] for b in bars]
    lows = [_bar_ohlc(b)[2] for b in bars]
    vol = _volume_sum(bars)
    start_i, end_i = min(idxs), max(idxs)
    if cls is Bar5M:
        return Bar5M(
            timestamp=first_ts,
            open=o,
            high=max(highs),
            low=min(lows),
            close=c,
            volume=vol,
            bar_index_start=start_i,
            bar_index_end=end_i,
            complete=True,
        )
    return Bar15M(
        timestamp=first_ts,
        open=o,
        high=max(highs),
        low=min(lows),
        close=c,
        volume=vol,
        bar_index_start=start_i,
        bar_index_end=end_i,
        complete=True,
    )


def aggregate_to_5m(bars_1m: Sequence[Any]) -> list[Bar5M]:
    if not bars_1m:
        return []
    by_anchor: dict[datetime, dict[datetime, tuple[Any, int]]] = {}
    for i, bar in enumerate(bars_1m):
        ts = _to_utc_datetime(bar.timestamp)
        anchor = _five_minute_anchor_close_label(ts)
        mk = _minute_key(ts)
        by_anchor.setdefault(anchor, {})[mk] = (bar, i)

    out: list[Bar5M] = []
    for anchor in sorted(by_anchor.keys()):
        req = _required_1m_keys_5m(anchor)
        flat = by_anchor[anchor]
        got = _aggregate_window(flat, req)
        if got is None:
            continue
        seq, first_ts = got
        out.append(_emit_from_sequence(seq, first_ts, Bar5M))
    return out


def aggregate_to_15m(bars_1m: Sequence[Any]) -> list[Bar15M]:
    if not bars_1m:
        return []
    by_anchor: dict[datetime, dict[datetime, tuple[Any, int]]] = {}
    for i, bar in enumerate(bars_1m):
        ts = _to_utc_datetime(bar.timestamp)
        anchor = _fifteen_minute_anchor_close_label(ts)
        mk = _minute_key(ts)
        by_anchor.setdefault(anchor, {})[mk] = (bar, i)

    out: list[Bar15M] = []
    for anchor in sorted(by_anchor.keys()):
        req = _required_1m_keys_15m(anchor)
        flat = by_anchor[anchor]
        got = _aggregate_window(flat, req)
        if got is None:
            continue
        seq, first_ts = got
        out.append(_emit_from_sequence(seq, first_ts, Bar15M))
    return out


def _session_date_utc(ts: datetime, day_boundary_utc_hour: int) -> date:
    shifted = ts - timedelta(hours=int(day_boundary_utc_hour))
    return shifted.date()


def aggregate_to_daily(bars_1m: Sequence[Any], day_boundary_utc_hour: int = 22) -> list[BarDaily]:
    if not bars_1m:
        return []
    h = int(day_boundary_utc_hour)
    out: list[BarDaily] = []

    current_day: Optional[date] = None
    bucket: list[tuple[Any, int]] = []

    def flush() -> None:
        nonlocal bucket, current_day
        if current_day is None or not bucket:
            bucket = []
            return
        bars = [b for b, _ in bucket]
        idxs = [i for _, i in bucket]
        o, _, _, _ = _bar_ohlc(bars[0])
        _, _, _, c = _bar_ohlc(bars[-1])
        highs = [_bar_ohlc(b)[1] for b in bars]
        lows = [_bar_ohlc(b)[2] for b in bars]
        vol = _volume_sum(bars)
        first_ts = _to_utc_datetime(bars[0].timestamp)
        start_i, end_i = min(idxs), max(idxs)
        out.append(
            BarDaily(
                trading_day=current_day,
                timestamp=first_ts,
                open=o,
                high=max(highs),
                low=min(lows),
                close=c,
                volume=vol,
                bar_index_start=start_i,
                bar_index_end=end_i,
                bar_count=len(bucket),
            )
        )
        bucket = []

    for i, bar in enumerate(bars_1m):
        ts = _to_utc_datetime(bar.timestamp)
        d = _session_date_utc(ts, h)
        if current_day is None:
            current_day = d
        elif d != current_day:
            flush()
            current_day = d
        bucket.append((bar, i))

    flush()
    return out


def compute_ema(values: list[float], period: int) -> list[float]:
    if period < 1:
        raise ValueError("period must be >= 1")
    if not values:
        return []
    n = len(values)
    out = [math.nan] * n
    if n < period:
        return out
    k = 2.0 / (period + 1)
    sma = sum(values[:period]) / period
    out[period - 1] = sma
    prev = sma
    for i in range(period, n):
        prev = (values[i] - prev) * k + prev
        out[i] = prev
    return out
