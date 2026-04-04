"""Tests for swing_macro_bars.py — 4H bar infrastructure."""

import pytest
from datetime import datetime, timezone

from core.regime_backtest_engine.swing_macro_bars import (
    Bar4H,
    get_4h_boundary,
    IncrementalBar4HBuilder,
    IncrementalEMA,
    IncrementalATR,
    find_recent_swing_low,
    find_recent_swing_high,
)


# ─────────────────────────────────────────────
#  get_4h_boundary
# ─────────────────────────────────────────────


class TestGet4HBoundary:
    def test_at_midnight(self):
        dt = datetime(2024, 1, 15, 0, 0, tzinfo=timezone.utc)
        assert get_4h_boundary(dt) == datetime(2024, 1, 15, 0, 0, tzinfo=timezone.utc)

    def test_within_first_block(self):
        dt = datetime(2024, 1, 15, 2, 30, tzinfo=timezone.utc)
        assert get_4h_boundary(dt) == datetime(2024, 1, 15, 0, 0, tzinfo=timezone.utc)

    def test_at_second_boundary(self):
        dt = datetime(2024, 1, 15, 4, 0, tzinfo=timezone.utc)
        assert get_4h_boundary(dt) == datetime(2024, 1, 15, 4, 0, tzinfo=timezone.utc)

    def test_mid_afternoon(self):
        dt = datetime(2024, 1, 15, 14, 37, tzinfo=timezone.utc)
        assert get_4h_boundary(dt) == datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)

    def test_last_block_of_day(self):
        dt = datetime(2024, 1, 15, 23, 59, tzinfo=timezone.utc)
        assert get_4h_boundary(dt) == datetime(2024, 1, 15, 20, 0, tzinfo=timezone.utc)

    def test_all_six_boundaries(self):
        for expected_hour in [0, 4, 8, 12, 16, 20]:
            dt = datetime(2024, 1, 15, expected_hour, 0, tzinfo=timezone.utc)
            assert get_4h_boundary(dt).hour == expected_hour

    def test_preserves_date(self):
        dt = datetime(2025, 7, 22, 15, 45, tzinfo=timezone.utc)
        result = get_4h_boundary(dt)
        assert result.year == 2025
        assert result.month == 7
        assert result.day == 22
        assert result.hour == 12


# ─────────────────────────────────────────────
#  IncrementalBar4HBuilder
# ─────────────────────────────────────────────


class TestBar4HBuilder:
    def _ts(self, hour, minute=0, day=15):
        return datetime(2024, 1, day, hour, minute, tzinfo=timezone.utc)

    def test_first_bar_returns_none(self):
        b = IncrementalBar4HBuilder()
        assert b.update(self._ts(0, 0), 150.0, 150.5, 149.5, 150.2) is None

    def test_same_period_returns_none(self):
        b = IncrementalBar4HBuilder()
        b.update(self._ts(0, 0), 150.0, 150.5, 149.5, 150.2)
        assert b.update(self._ts(0, 1), 150.2, 150.8, 149.3, 150.6) is None
        assert b.update(self._ts(3, 59), 150.0, 150.3, 149.8, 150.1) is None

    def test_boundary_crossing_emits_completed_bar(self):
        b = IncrementalBar4HBuilder()
        b.update(self._ts(0, 0), 150.0, 150.5, 149.5, 150.2)
        b.update(self._ts(1, 0), 150.2, 151.0, 149.0, 150.8)

        result = b.update(self._ts(4, 0), 150.8, 151.5, 150.5, 151.0)

        assert result is not None
        assert result.timestamp == self._ts(0, 0)
        assert result.open == 150.0
        assert result.high == 151.0
        assert result.low == 149.0
        assert result.close == 150.8
        assert result.bar_count == 2

    def test_ohlc_aggregation_three_bars(self):
        b = IncrementalBar4HBuilder()
        b.update(self._ts(8, 0), 145.0, 145.5, 144.8, 145.3)
        b.update(self._ts(8, 1), 145.3, 146.0, 145.1, 145.8)
        b.update(self._ts(8, 2), 145.8, 145.9, 144.5, 144.7)

        result = b.update(self._ts(12, 0), 144.7, 145.0, 144.5, 144.8)

        assert result.open == 145.0
        assert result.high == 146.0
        assert result.low == 144.5
        assert result.close == 144.7
        assert result.bar_count == 3

    def test_multiple_completed_periods(self):
        b = IncrementalBar4HBuilder()
        completed = []

        # Feed one bar per hour for 12 hours
        for hour in range(12):
            r = b.update(self._ts(hour, 0), 150.0, 150.5, 149.5, 150.0)
            if r is not None:
                completed.append(r)

        # 00-03 emitted at 04:00, 04-07 emitted at 08:00
        assert len(completed) == 2
        assert completed[0].timestamp.hour == 0
        assert completed[1].timestamp.hour == 4

    def test_day_boundary_crossing(self):
        b = IncrementalBar4HBuilder()
        b.update(datetime(2024, 1, 15, 23, 59, tzinfo=timezone.utc), 150.0, 150.5, 149.5, 150.2)

        result = b.update(datetime(2024, 1, 16, 0, 0, tzinfo=timezone.utc), 150.2, 150.3, 150.0, 150.1)

        assert result is not None
        assert result.timestamp == datetime(2024, 1, 15, 20, 0, tzinfo=timezone.utc)

    def test_bar_count_full_4h_period(self):
        """240 one-minute bars in one 4H period produces bar_count=240."""
        b = IncrementalBar4HBuilder()

        # Feed 240 bars across hours 0-3 (one full 4H period)
        for i in range(240):
            hour = i // 60  # 0, 1, 2, 3
            minute = i % 60  # 0-59
            b.update(self._ts(hour, minute), 150.0, 150.5, 149.5, 150.0)

        # Trigger emission with the next period
        result = b.update(self._ts(4, 0), 150.0, 150.5, 149.5, 150.0)

        assert result is not None
        assert result.bar_count == 240

    def test_open_is_first_bar_close_is_last(self):
        b = IncrementalBar4HBuilder()
        b.update(self._ts(0, 0), 100.0, 101.0, 99.0, 100.5)  # open = 100.0
        b.update(self._ts(1, 0), 100.5, 102.0, 98.0, 101.5)
        b.update(self._ts(3, 0), 101.5, 101.8, 100.0, 101.2)  # close = 101.2

        result = b.update(self._ts(4, 0), 101.2, 102.0, 101.0, 101.5)

        assert result.open == 100.0
        assert result.close == 101.2

    def test_high_is_max_of_all_bars(self):
        b = IncrementalBar4HBuilder()
        b.update(self._ts(0, 0), 150.0, 150.5, 149.5, 150.2)
        b.update(self._ts(1, 0), 150.2, 155.0, 149.0, 150.8)  # highest
        b.update(self._ts(2, 0), 150.8, 151.0, 149.8, 150.5)

        result = b.update(self._ts(4, 0), 150.5, 151.0, 150.0, 150.8)

        assert result.high == 155.0

    def test_low_is_min_of_all_bars(self):
        b = IncrementalBar4HBuilder()
        b.update(self._ts(0, 0), 150.0, 150.5, 149.5, 150.2)
        b.update(self._ts(1, 0), 150.2, 151.0, 144.0, 150.8)  # lowest
        b.update(self._ts(2, 0), 150.8, 151.0, 149.8, 150.5)

        result = b.update(self._ts(4, 0), 150.5, 151.0, 150.0, 150.8)

        assert result.low == 144.0


# ─────────────────────────────────────────────
#  IncrementalEMA
# ─────────────────────────────────────────────


class TestIncrementalEMA:
    def test_returns_none_until_seeded(self):
        ema = IncrementalEMA(period=3)
        assert ema.update(10.0) is None
        assert ema.update(11.0) is None
        result = ema.update(12.0)  # Third value — should seed
        assert result is not None

    def test_seed_is_sma(self):
        ema = IncrementalEMA(period=3)
        ema.update(10.0)
        ema.update(11.0)
        result = ema.update(12.0)
        assert result == pytest.approx(11.0)  # (10+11+12) / 3

    def test_ema_after_seed(self):
        ema = IncrementalEMA(period=3)
        ema.update(10.0)
        ema.update(11.0)
        ema.update(12.0)  # seed = 11.0

        # k = 2/(3+1) = 0.5
        # EMA = (13 - 11) * 0.5 + 11 = 12.0
        result = ema.update(13.0)
        assert result == pytest.approx(12.0)

    def test_value_property(self):
        ema = IncrementalEMA(period=3)
        assert ema.value is None
        ema.update(10.0)
        assert ema.value is None
        ema.update(11.0)
        assert ema.value is None
        ema.update(12.0)
        assert ema.value == pytest.approx(11.0)

    def test_slope_none_after_seed(self):
        """Slope needs two EMA values — None right after seeding."""
        ema = IncrementalEMA(period=3)
        ema.update(10.0)
        ema.update(11.0)
        ema.update(12.0)
        # Only one EMA value so far
        assert ema.slope is None

    def test_slope_positive_uptrend(self):
        ema = IncrementalEMA(period=3)
        ema.update(10.0)
        ema.update(11.0)
        ema.update(12.0)  # seed = 11.0
        ema.update(14.0)  # EMA = 12.5
        assert ema.slope > 0

    def test_slope_negative_downtrend(self):
        ema = IncrementalEMA(period=3)
        ema.update(12.0)
        ema.update(11.0)
        ema.update(10.0)  # seed = 11.0
        ema.update(7.0)  # EMA = 9.0
        assert ema.slope < 0

    def test_prev_value_tracks(self):
        ema = IncrementalEMA(period=3)
        ema.update(10.0)
        ema.update(11.0)
        ema.update(12.0)  # seed = 11.0
        ema.update(14.0)  # EMA moves from 11.0 to 12.5
        assert ema.prev_value == pytest.approx(11.0)
        assert ema.value == pytest.approx(12.5)

    def test_long_sequence_converges(self):
        """EMA of a constant sequence should converge to that constant."""
        ema = IncrementalEMA(period=20)
        for _ in range(200):
            ema.update(100.0)
        assert ema.value == pytest.approx(100.0, abs=0.001)

    def test_ema50_requires_50_values(self):
        ema = IncrementalEMA(period=50)
        for i in range(49):
            assert ema.update(float(i)) is None
        result = ema.update(49.0)
        assert result is not None


# ─────────────────────────────────────────────
#  IncrementalATR
# ─────────────────────────────────────────────


class TestIncrementalATR:
    def _bar(self, h, l, c, hour=0):
        return Bar4H(
            timestamp=datetime(2024, 1, 15, hour, 0, tzinfo=timezone.utc),
            open=c,  # open doesn't matter for ATR
            high=h,
            low=l,
            close=c,
            bar_count=1,
        )

    def test_first_bar_returns_none(self):
        atr = IncrementalATR(period=3)
        assert atr.update(self._bar(151.0, 149.0, 150.0)) is None

    def test_needs_period_plus_one_bars(self):
        """ATR(3) needs 1 bar for prev_close + 3 bars for TRs = 4 bars total."""
        atr = IncrementalATR(period=3)
        atr.update(self._bar(151.0, 149.0, 150.0))  # establishes prev_close
        assert atr.update(self._bar(152.0, 149.5, 151.0)) is None  # TR 1
        assert atr.update(self._bar(153.0, 150.0, 152.0)) is None  # TR 2
        result = atr.update(self._bar(154.0, 151.0, 153.0))  # TR 3
        assert result is not None

    def test_initial_atr_is_average(self):
        """First ATR value is simple average of first `period` true ranges."""
        atr = IncrementalATR(period=3)

        # prev_close = 150.0
        atr.update(self._bar(151.0, 149.0, 150.0))

        # TR1: max(152-149.5, |152-150|, |149.5-150|) = max(2.5, 2.0, 0.5) = 2.5
        atr.update(self._bar(152.0, 149.5, 151.0))

        # TR2: max(153-150, |153-151|, |150-151|) = max(3.0, 2.0, 1.0) = 3.0
        atr.update(self._bar(153.0, 150.0, 152.0))

        # TR3: max(154-151, |154-152|, |151-152|) = max(3.0, 2.0, 1.0) = 3.0
        result = atr.update(self._bar(154.0, 151.0, 153.0))

        expected = (2.5 + 3.0 + 3.0) / 3
        assert result == pytest.approx(expected)

    def test_wilder_smoothing_after_init(self):
        atr = IncrementalATR(period=3)
        atr.update(self._bar(151.0, 149.0, 150.0))
        atr.update(self._bar(152.0, 149.5, 151.0))  # TR = 2.5
        atr.update(self._bar(153.0, 150.0, 152.0))  # TR = 3.0
        initial = atr.update(self._bar(154.0, 151.0, 153.0))  # TR = 3.0, ATR init

        # Next bar: TR4
        # max(155-152, |155-153|, |152-153|) = max(3.0, 2.0, 1.0) = 3.0
        result = atr.update(self._bar(155.0, 152.0, 154.0))

        # Wilder: (prev * (3-1) + 3.0) / 3
        expected = (initial * 2 + 3.0) / 3
        assert result == pytest.approx(expected)

    def test_value_property(self):
        atr = IncrementalATR(period=3)
        assert atr.value is None
        atr.update(self._bar(151.0, 149.0, 150.0))
        atr.update(self._bar(152.0, 149.5, 151.0))
        atr.update(self._bar(153.0, 150.0, 152.0))
        atr.update(self._bar(154.0, 151.0, 153.0))
        assert atr.value is not None

    def test_constant_bars_produce_stable_atr(self):
        """Bars with identical range should produce ATR equal to that range."""
        atr = IncrementalATR(period=14)
        # First bar sets prev_close
        atr.update(self._bar(151.0, 149.0, 150.0))
        # Feed 100 bars with range = 2.0, close always at midpoint
        for _ in range(100):
            atr.update(self._bar(151.0, 149.0, 150.0))
        # TR each time: max(2.0, |151-150|, |149-150|) = 2.0
        assert atr.value == pytest.approx(2.0, abs=0.01)

    def test_gap_up_increases_atr(self):
        """A gap should increase true range and thus ATR."""
        atr = IncrementalATR(period=3)
        atr.update(self._bar(151.0, 149.0, 150.0))
        atr.update(self._bar(151.0, 149.0, 150.0))  # TR = 2.0
        atr.update(self._bar(151.0, 149.0, 150.0))  # TR = 2.0
        initial = atr.update(self._bar(151.0, 149.0, 150.0))  # TR = 2.0, ATR = 2.0

        # Gap up: prev_close = 150, new bar high=160, low=158
        # TR = max(2.0, |160-150|, |158-150|) = 10.0
        result = atr.update(self._bar(160.0, 158.0, 159.0))
        assert result > initial


# ─────────────────────────────────────────────
#  find_recent_swing_low / high
# ─────────────────────────────────────────────


class TestSwingFinders:
    def _bar(self, low, high, ts_hour=0):
        return Bar4H(
            timestamp=datetime(2024, 1, 15, ts_hour, 0, tzinfo=timezone.utc),
            open=low + 0.5,
            high=high,
            low=low,
            close=high - 0.5,
            bar_count=1,
        )

    def test_swing_low_returns_none_if_insufficient_bars(self):
        bars = [self._bar(149.0, 151.0)]
        assert find_recent_swing_low(bars, lookback=5) is None

    def test_swing_low_basic(self):
        bars = [
            self._bar(150.0, 152.0),
            self._bar(149.0, 151.0),
            self._bar(148.0, 150.0),  # lowest
            self._bar(149.5, 151.5),
            self._bar(150.5, 152.5),
        ]
        assert find_recent_swing_low(bars, lookback=5) == 148.0

    def test_swing_high_basic(self):
        bars = [
            self._bar(150.0, 152.0),
            self._bar(149.0, 155.0),  # highest
            self._bar(148.0, 150.0),
            self._bar(149.5, 151.5),
            self._bar(150.5, 152.5),
        ]
        assert find_recent_swing_high(bars, lookback=5) == 155.0

    def test_swing_low_respects_lookback(self):
        bars = [
            self._bar(140.0, 145.0),  # very low but outside lookback=3
            self._bar(150.0, 152.0),
            self._bar(149.0, 151.0),
            self._bar(148.0, 150.0),  # lowest within lookback=3
        ]
        assert find_recent_swing_low(bars, lookback=3) == 148.0

    def test_swing_high_respects_lookback(self):
        bars = [
            self._bar(150.0, 200.0),  # very high but outside lookback=3
            self._bar(150.0, 152.0),
            self._bar(149.0, 155.0),  # highest within lookback=3
            self._bar(148.0, 150.0),
        ]
        assert find_recent_swing_high(bars, lookback=3) == 155.0

    def test_swing_low_with_more_bars_than_lookback(self):
        bars = [self._bar(145.0 + i, 147.0 + i) for i in range(20)]
        result = find_recent_swing_low(bars, lookback=5)
        # Last 5 bars: i=15..19, lows = 160..164
        assert result == 160.0

    def test_swing_high_with_more_bars_than_lookback(self):
        bars = [self._bar(145.0 + i, 147.0 + i) for i in range(20)]
        result = find_recent_swing_high(bars, lookback=5)
        # Last 5 bars: i=15..19, highs = 162..166
        assert result == 166.0


# ─────────────────────────────────────────────
#  Integration: Builder → EMA → ATR pipeline
# ─────────────────────────────────────────────


class TestBuilderToIndicatorPipeline:
    """Verify the components chain together correctly."""

    def test_builder_feeds_ema(self):
        builder = IncrementalBar4HBuilder()
        ema = IncrementalEMA(period=3)

        completed_bars = []
        ema_values = []

        # Hours 0–15 are four 4H buckets; first emission needs hour 4, last needs hour 16
        # to close the 12:00 block (range(16) stops at 15 and leaves that block open).
        for hour in range(17):
            ts = datetime(2024, 1, 15, hour, 0, tzinfo=timezone.utc)
            bar4h = builder.update(ts, 150.0 + hour, 151.0 + hour, 149.0 + hour, 150.5 + hour)
            if bar4h is not None:
                completed_bars.append(bar4h)
                result = ema.update(bar4h.close)
                if result is not None:
                    ema_values.append(result)

        # 4 completed bars (00, 04, 08, 12), EMA(3) needs 3 → one EMA value after bar 3
        assert len(completed_bars) == 4
        assert len(ema_values) >= 1

    def test_builder_feeds_atr(self):
        builder = IncrementalBar4HBuilder()
        atr = IncrementalATR(period=3)

        completed_bars = []
        atr_values = []

        # Need 4+ completed 4H bars → 5 periods of data
        for hour in range(24):
            ts = datetime(2024, 1, 15, hour, 0, tzinfo=timezone.utc)
            bar4h = builder.update(ts, 150.0, 152.0, 148.0, 151.0)
            if bar4h is not None:
                completed_bars.append(bar4h)
                result = atr.update(bar4h)
                if result is not None:
                    atr_values.append(result)

        # 6 4H periods in 24 hours → 5 completed (last is still building)
        # Wait — 0,4,8,12,16,20 boundaries. Feeding 0-23, we get emissions
        # at 4,8,12,16,20 = 5 completed bars.
        # ATR(3) needs 1 prev_close + 3 TRs = 4 bars → first ATR after bar 4
        assert len(completed_bars) == 5
        assert len(atr_values) >= 1

    def test_full_pipeline_bar_count_accumulates(self):
        """Build bars, track them, use for swing detection."""
        builder = IncrementalBar4HBuilder()
        bar_history = []

        # Build 20+ bars across multiple days
        for day in range(15, 22):  # 7 days
            for hour in range(24):
                ts = datetime(2024, 1, day, hour, 0, tzinfo=timezone.utc)
                price = 150.0 + (day - 15) * 0.5 + hour * 0.01
                bar4h = builder.update(ts, price, price + 1.0, price - 1.0, price + 0.3)
                if bar4h is not None:
                    bar_history.append(bar4h)

        # 7 days × 6 periods/day = 42, minus incomplete first/last
        assert len(bar_history) >= 35

        # Swing detection should work
        swing_low = find_recent_swing_low(bar_history, lookback=5)
        swing_high = find_recent_swing_high(bar_history, lookback=5)
        assert swing_low is not None
        assert swing_high is not None
        assert swing_high > swing_low
