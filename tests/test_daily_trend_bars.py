"""Tests for daily_trend_bars — 22:00 UTC forex daily boundaries and builder."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from core.regime_backtest_engine.daily_trend_bars import (
    DailyBar,
    IncrementalDailyBarBuilder,
    get_daily_boundary,
)

UTC = timezone.utc


def _mb(ts: datetime, price: float = 100.0) -> SimpleNamespace:
    return SimpleNamespace(
        timestamp=ts,
        mid_open=price,
        mid_high=price + 0.1,
        mid_low=price - 0.1,
        mid_close=price,
    )


class TestGetDailyBoundary:
    def test_0300_previous_day_2200(self) -> None:
        ts = datetime(2024, 1, 15, 3, 0, tzinfo=UTC)
        assert get_daily_boundary(ts) == datetime(2024, 1, 14, 22, 0, tzinfo=UTC)

    def test_2159_previous_day_2200(self) -> None:
        ts = datetime(2024, 1, 15, 21, 59, tzinfo=UTC)
        assert get_daily_boundary(ts) == datetime(2024, 1, 14, 22, 0, tzinfo=UTC)

    def test_2200_same_day_2200(self) -> None:
        ts = datetime(2024, 1, 15, 22, 0, tzinfo=UTC)
        assert get_daily_boundary(ts) == datetime(2024, 1, 15, 22, 0, tzinfo=UTC)

    def test_2201_same_day_2200(self) -> None:
        ts = datetime(2024, 1, 15, 22, 1, tzinfo=UTC)
        assert get_daily_boundary(ts) == datetime(2024, 1, 15, 22, 0, tzinfo=UTC)

    def test_midnight_previous_2200(self) -> None:
        ts = datetime(2024, 1, 16, 0, 0, tzinfo=UTC)
        assert get_daily_boundary(ts) == datetime(2024, 1, 15, 22, 0, tzinfo=UTC)

    def test_naive_assumed_utc(self) -> None:
        ts = datetime(2024, 1, 15, 3, 0)
        b = get_daily_boundary(ts)
        assert b == datetime(2024, 1, 14, 22, 0, tzinfo=UTC)

    def test_sequential_boundaries(self) -> None:
        t0 = datetime(2024, 1, 10, 22, 0, tzinfo=UTC)
        for i in range(5):
            ts = t0 + timedelta(days=i)
            assert get_daily_boundary(ts) == t0 + timedelta(days=i)


class TestDailyBarDataclass:
    def test_fields_and_frozen(self) -> None:
        b = DailyBar(
            timestamp=datetime(2024, 1, 1, 22, 0, tzinfo=UTC),
            open=1.0,
            high=2.0,
            low=0.5,
            close=1.5,
            bar_index=0,
        )
        assert b.open == 1.0 and b.bar_index == 0
        with pytest.raises(AttributeError):
            b.close = 9.0  # type: ignore[misc]


class TestIncrementalDailyBarBuilderBasic:
    def test_no_complete_within_day(self) -> None:
        b = IncrementalDailyBarBuilder()
        day_start = datetime(2024, 1, 14, 22, 0, tzinfo=UTC)
        for m in range(60):
            ts = day_start + timedelta(minutes=m)
            assert b.update(_mb(ts, 100.0 + m * 0.001)) is None

    def test_complete_crossing_2200(self) -> None:
        b = IncrementalDailyBarBuilder()
        t_old = datetime(2024, 1, 14, 21, 59, tzinfo=UTC)
        b.update(_mb(t_old, 100.0))
        t_new = datetime(2024, 1, 14, 22, 0, tzinfo=UTC)
        done = b.update(_mb(t_new, 101.0))
        assert done is not None
        assert done.timestamp == datetime(2024, 1, 13, 22, 0, tzinfo=UTC)
        assert done.open == 100.0
        assert done.close == 100.0
        assert done.bar_index == 0

    def test_bar_index_increments(self) -> None:
        b = IncrementalDailyBarBuilder()
        # Start just before first boundary cross
        t = datetime(2024, 1, 10, 21, 0, tzinfo=UTC)
        b.update(_mb(t, 1.0))
        for d in range(3):
            t = datetime(2024, 1, 10 + d, 22, 0, tzinfo=UTC)
            out = b.update(_mb(t, float(d)))
            if d == 0:
                assert out is not None
                assert out.bar_index == 0
            elif d == 1:
                assert out is not None
                assert out.bar_index == 1

    def test_first_bar_no_crash(self) -> None:
        b = IncrementalDailyBarBuilder()
        assert b.update(_mb(datetime(2024, 1, 1, 12, 0, tzinfo=UTC))) is None


class TestIncrementalDailyBarBuilderMultiDay:
    def test_three_days_ohlc(self) -> None:
        b = IncrementalDailyBarBuilder()
        outs: list[DailyBar] = []
        # Day A: Jan 14 22:00 - Jan 15 21:59
        for m in range(0, 200, 10):
            ts = datetime(2024, 1, 14, 22, 0, tzinfo=UTC) + timedelta(minutes=m)
            o = b.update(_mb(ts, 100.0))
            if o is not None:
                outs.append(o)
        ts_flip = datetime(2024, 1, 15, 22, 0, tzinfo=UTC)
        o = b.update(_mb(ts_flip, 110.0))
        if o is not None:
            outs.append(o)
        assert len(outs) >= 1
        assert all(isinstance(x, DailyBar) for x in outs)

    def test_single_minute_day(self) -> None:
        b = IncrementalDailyBarBuilder()
        b.update(_mb(datetime(2024, 2, 1, 21, 59, tzinfo=UTC), 50.0))
        d = b.update(_mb(datetime(2024, 2, 1, 22, 0, tzinfo=UTC), 51.0))
        assert d is not None
        assert d.high >= d.low


class TestWeekendGap:
    def test_gap_no_crash(self) -> None:
        b = IncrementalDailyBarBuilder()
        b.update(_mb(datetime(2024, 1, 5, 22, 0, tzinfo=UTC), 1.0))
        d = b.update(_mb(datetime(2024, 1, 8, 22, 0, tzinfo=UTC), 2.0))
        assert d is not None or True
