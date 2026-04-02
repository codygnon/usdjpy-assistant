from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from core.regime_backtest_engine.synthetic_bars import (
    BarDaily,
    aggregate_to_15m,
    aggregate_to_5m,
    aggregate_to_daily,
    compute_ema,
)


def _utc(y: int, mo: int, d: int, h: int, m: int = 0) -> datetime:
    return datetime(y, mo, d, h, m, tzinfo=timezone.utc)


def _bar(ts: datetime, o: float, h: float, l: float, c: float, **kwargs) -> SimpleNamespace:
    return SimpleNamespace(timestamp=ts, open=o, high=h, low=l, close=c, **kwargs)


def test_five_minute_complete_window_starts_at_boundary_clock() -> None:
    base = _utc(2025, 3, 10, 10, 0)
    bars = [
        _bar(base.replace(minute=1), 1, 2, 0.5, 1.5),
        _bar(base.replace(minute=2), 1.5, 2.5, 1.4, 2.0),
        _bar(base.replace(minute=3), 2.0, 3.0, 1.9, 2.5),
        _bar(base.replace(minute=4), 2.5, 3.5, 2.4, 3.0),
        _bar(base.replace(minute=5), 3.0, 4.0, 2.9, 3.5),
    ]
    out = aggregate_to_5m(bars)
    assert len(out) == 1
    b5 = out[0]
    assert b5.timestamp == bars[0].timestamp
    assert b5.open == 1.0
    assert b5.high == 4.0
    assert b5.low == 0.5
    assert b5.close == 3.5
    assert b5.bar_index_start == 0
    assert b5.bar_index_end == 4
    assert b5.complete is True


def test_five_minute_gap_suppresses_bar() -> None:
    base = _utc(2025, 3, 10, 10, 0)
    bars = [
        _bar(base.replace(minute=1), 1, 1, 1, 1),
        _bar(base.replace(minute=2), 1, 1, 1, 1),
        _bar(base.replace(minute=4), 1, 1, 1, 1),
        _bar(base.replace(minute=5), 1, 1, 1, 1),
    ]
    assert aggregate_to_5m(bars) == []


def test_fifteen_minute_splits_on_boundary() -> None:
    base = _utc(2025, 6, 1, 14, 0)
    first = [_bar(base.replace(minute=m), float(m), float(m) + 1, float(m) - 1, float(m) + 0.5) for m in range(1, 16)]
    second = [_bar(base.replace(minute=m), float(m), float(m) + 1, float(m) - 1, float(m) + 0.5) for m in range(16, 31)]
    bars = first + second
    out = aggregate_to_15m(bars)
    assert len(out) == 2
    assert out[0].open == first[0].open
    assert out[0].close == first[-1].close
    assert out[1].open == second[0].open
    assert out[1].close == second[-1].close
    assert out[0].bar_index_start == 0
    assert out[0].bar_index_end == 14
    assert out[1].bar_index_start == 15
    assert out[1].bar_index_end == 29


def test_daily_aggregation_22utc_boundary() -> None:
    bars = [
        _bar(_utc(2025, 1, 5, 22, 0), 100, 101, 99, 100.5),
        _bar(_utc(2025, 1, 6, 10, 0), 100.5, 102, 100, 101),
        _bar(_utc(2025, 1, 6, 22, 0), 101, 103, 100, 102),
    ]
    dly = aggregate_to_daily(bars, day_boundary_utc_hour=22)
    assert len(dly) == 2
    assert dly[0].trading_day.isoformat() == "2025-01-05"
    assert dly[0].open == 100.0
    assert dly[0].close == 101.0
    assert dly[0].high == 102.0
    assert dly[0].low == 99.0
    assert dly[0].bar_count == 2
    assert dly[1].trading_day.isoformat() == "2025-01-06"
    assert dly[1].bar_count == 1


def test_ohlcv_mid_columns_and_volume_rules() -> None:
    @dataclass
    class MidBar:
        timestamp: datetime
        mid_open: float
        mid_high: float
        mid_low: float
        mid_close: float
        tick_volume: float | None = None

    base = _utc(2025, 4, 2, 9, 0)
    bars = [
        MidBar(base.replace(minute=1), 1, 2, 0.5, 1.2, 10.0),
        MidBar(base.replace(minute=2), 1.2, 2.2, 1.0, 1.8, 5.0),
        MidBar(base.replace(minute=3), 1.8, 2.8, 1.5, 2.0, 3.0),
        MidBar(base.replace(minute=4), 2.0, 3.0, 1.9, 2.5, 2.0),
        MidBar(base.replace(minute=5), 2.5, 3.5, 2.4, 3.0, 1.0),
    ]
    b5 = aggregate_to_5m(bars)[0]
    assert b5.volume == 21.0

    bars_nv = [
        MidBar(base.replace(minute=m), 1, 1, 1, 1) for m in range(1, 6)
    ]
    assert aggregate_to_5m(bars_nv)[0].volume is None


def test_ema_matches_manual_period3() -> None:
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    out = compute_ema(vals, period=3)
    assert math.isnan(out[0]) and math.isnan(out[1])
    k = 2.0 / (3 + 1)
    sma = (1 + 2 + 3) / 3
    assert out[2] == pytest.approx(sma)
    assert out[3] == pytest.approx((vals[3] - sma) * k + sma)
    assert out[4] == pytest.approx((vals[4] - out[3]) * k + out[3])


def test_ema_warmup_and_empty() -> None:
    assert compute_ema([], period=3) == []
    out = compute_ema([1.0], period=3)
    assert len(out) == 1 and math.isnan(out[0])
    out2 = compute_ema([1.0, 2.0], period=3)
    assert all(math.isnan(x) for x in out2)


def test_single_bar_no_synthetic() -> None:
    b = [_bar(_utc(2025, 1, 1, 12, 1), 1, 1, 1, 1)]
    assert aggregate_to_5m(b) == []
    assert aggregate_to_15m(b) == []


def test_bar_index_range_non_contiguous_indices() -> None:
    base = _utc(2025, 8, 1, 11, 0)
    prefix = [_bar(base.replace(minute=0), 0, 0, 0, 0)]
    window = [_bar(base.replace(minute=m), 1, 1, 1, 1) for m in range(1, 6)]
    bars = prefix + window
    b5 = aggregate_to_5m(bars)[0]
    assert b5.bar_index_start == 1
    assert b5.bar_index_end == 5


def test_bar_daily_dataclass_volume_optional() -> None:
    bd = BarDaily(
        trading_day=_utc(2025, 1, 1, 0, 0).date(),
        timestamp=_utc(2025, 1, 1, 0, 0),
        open=1.0,
        high=2.0,
        low=0.5,
        close=1.5,
        volume=None,
        bar_index_start=0,
        bar_index_end=9,
        bar_count=10,
    )
    assert bd.bar_count == 10
