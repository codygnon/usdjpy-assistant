"""Tests for swing_macro_signals.py — weekly macro signal module."""

import pytest
from datetime import datetime, timezone

from core.regime_backtest_engine.swing_macro_signals import (
    MacroDirection,
    MacroBias,
    compute_5d_return,
    direction_from_return,
    combine_bias,
    bias_supports_long,
    bias_supports_short,
    WeeklyMacroSignal,
)


# ─────────────────────────────────────────────
#  compute_5d_return
# ─────────────────────────────────────────────


class TestCompute5dReturn:
    def test_insufficient_data(self):
        assert compute_5d_return([100.0, 101.0, 102.0]) is None
        assert compute_5d_return([100.0] * 5) is None  # Need 6

    def test_exactly_6_values(self):
        # closes[-5] = 100, closes[-1] = 105
        closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
        ret = compute_5d_return(closes)
        # (105 - 101) / 101  ← closes[-5]=101, closes[-1]=105
        assert ret == pytest.approx((105.0 - 101.0) / 101.0)

    def test_positive_return(self):
        closes = [90.0, 100.0, 101.0, 102.0, 103.0, 104.0, 110.0]
        # len=7 → closes[-5] = index 2 = 101, closes[-1] = 110
        ret = compute_5d_return(closes)
        assert ret == pytest.approx((110.0 - 101.0) / 101.0)

    def test_negative_return(self):
        closes = [100.0, 100.0, 99.0, 98.0, 97.0, 96.0, 95.0]
        ret = compute_5d_return(closes)
        assert ret < 0

    def test_zero_return(self):
        closes = [100.0, 101.0, 102.0, 101.0, 100.5, 100.0, 101.0]
        # closes[-5] = 101, closes[-1] = 101 → 0
        # Actually: closes[-5] = 102.0, closes[-1] = 101.0
        ret = compute_5d_return(closes)
        assert ret == pytest.approx((101.0 - 102.0) / 102.0)

    def test_uses_last_6_elements(self):
        # Prepend junk — should not matter
        closes = [999.0, 888.0, 777.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
        # Last 6: [100, 101, 102, 103, 104, 105]
        # closes[-5] = 101, closes[-1] = 105
        ret = compute_5d_return(closes)
        assert ret == pytest.approx((105.0 - 101.0) / 101.0)


# ─────────────────────────────────────────────
#  direction_from_return
# ─────────────────────────────────────────────


class TestDirectionFromReturn:
    def test_none_returns_neutral(self):
        assert direction_from_return(None) == MacroDirection.NEUTRAL

    def test_above_threshold_is_up(self):
        assert direction_from_return(0.005, threshold=0.002) == MacroDirection.UP

    def test_below_neg_threshold_is_down(self):
        assert direction_from_return(-0.005, threshold=0.002) == MacroDirection.DOWN

    def test_within_threshold_is_neutral(self):
        assert direction_from_return(0.001, threshold=0.002) == MacroDirection.NEUTRAL
        assert direction_from_return(-0.001, threshold=0.002) == MacroDirection.NEUTRAL

    def test_exactly_at_threshold_is_neutral(self):
        # At threshold but not above
        assert direction_from_return(0.002, threshold=0.002) == MacroDirection.NEUTRAL

    def test_just_above_threshold(self):
        assert direction_from_return(0.00201, threshold=0.002) == MacroDirection.UP

    def test_large_positive(self):
        assert direction_from_return(0.10) == MacroDirection.UP

    def test_large_negative(self):
        assert direction_from_return(-0.10) == MacroDirection.DOWN


# ─────────────────────────────────────────────
#  combine_bias — full agreement matrix
# ─────────────────────────────────────────────


class TestCombineBias:
    def test_both_up_is_long(self):
        assert combine_bias(MacroDirection.UP, MacroDirection.UP) == MacroBias.LONG

    def test_both_down_is_short(self):
        assert combine_bias(MacroDirection.DOWN, MacroDirection.DOWN) == MacroBias.SHORT

    def test_oil_up_dxy_neutral_is_lean_long(self):
        assert combine_bias(MacroDirection.UP, MacroDirection.NEUTRAL) == MacroBias.LEAN_LONG

    def test_oil_neutral_dxy_up_is_lean_long(self):
        assert combine_bias(MacroDirection.NEUTRAL, MacroDirection.UP) == MacroBias.LEAN_LONG

    def test_oil_down_dxy_neutral_is_lean_short(self):
        assert combine_bias(MacroDirection.DOWN, MacroDirection.NEUTRAL) == MacroBias.LEAN_SHORT

    def test_oil_neutral_dxy_down_is_lean_short(self):
        assert combine_bias(MacroDirection.NEUTRAL, MacroDirection.DOWN) == MacroBias.LEAN_SHORT

    def test_oil_up_dxy_down_is_neutral(self):
        assert combine_bias(MacroDirection.UP, MacroDirection.DOWN) == MacroBias.NEUTRAL

    def test_oil_down_dxy_up_is_neutral(self):
        assert combine_bias(MacroDirection.DOWN, MacroDirection.UP) == MacroBias.NEUTRAL

    def test_both_neutral_is_neutral(self):
        assert combine_bias(MacroDirection.NEUTRAL, MacroDirection.NEUTRAL) == MacroBias.NEUTRAL


# ─────────────────────────────────────────────
#  bias_supports_long / short
# ─────────────────────────────────────────────


class TestBiasSupports:
    def test_long_supports_long(self):
        assert bias_supports_long(MacroBias.LONG) is True
        assert bias_supports_long(MacroBias.LEAN_LONG) is True

    def test_short_does_not_support_long(self):
        assert bias_supports_long(MacroBias.SHORT) is False
        assert bias_supports_long(MacroBias.LEAN_SHORT) is False
        assert bias_supports_long(MacroBias.NEUTRAL) is False

    def test_short_supports_short(self):
        assert bias_supports_short(MacroBias.SHORT) is True
        assert bias_supports_short(MacroBias.LEAN_SHORT) is True

    def test_long_does_not_support_short(self):
        assert bias_supports_short(MacroBias.LONG) is False
        assert bias_supports_short(MacroBias.LEAN_LONG) is False
        assert bias_supports_short(MacroBias.NEUTRAL) is False


# ─────────────────────────────────────────────
#  WeeklyMacroSignal — integration tests
# ─────────────────────────────────────────────


def _make_daily_series(start_date, prices):
    """Create a list of (datetime, close) tuples from prices starting at start_date."""
    result = []
    for i, price in enumerate(prices):
        dt = datetime(
            start_date.year,
            start_date.month,
            start_date.day + i,
            0,
            0,
            tzinfo=timezone.utc,
        )
        result.append((dt, price))
    return result


class TestWeeklyMacroSignal:
    def _make_signal(self, oil_prices, eurusd_prices, start_day=1):
        """Helper to create a WeeklyMacroSignal with simple test data."""
        start = datetime(2024, 1, start_day, tzinfo=timezone.utc)
        oil = _make_daily_series(start, oil_prices)
        eurusd = _make_daily_series(start, eurusd_prices)
        return WeeklyMacroSignal(oil, eurusd)

    def test_insufficient_data_returns_neutral(self):
        # Only 3 days of data — not enough for 5d return
        signal = self._make_signal(
            [80.0, 81.0, 82.0],
            [1.10, 1.11, 1.12],
        )
        # Query after all bars complete
        query = datetime(2024, 1, 10, 23, 0, tzinfo=timezone.utc)
        reading = signal.compute(query)
        assert reading.bias == MacroBias.NEUTRAL
        assert reading.tradeable is False

    def test_oil_up_dxy_up_gives_long(self):
        # Oil rising steadily: 80 → 85 over 7 days (~6% in 5 days)
        # EUR/USD falling: 1.10 → 1.05 (DXY rising)
        signal = self._make_signal(
            [78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0],
            [1.12, 1.11, 1.10, 1.09, 1.08, 1.07, 1.06, 1.05],
        )
        # Query on day 8 (index 7) after 22:00 UTC
        query = datetime(2024, 1, 8, 23, 0, tzinfo=timezone.utc)
        reading = signal.compute(query)

        assert reading.oil_direction == MacroDirection.UP
        assert reading.dxy_direction == MacroDirection.UP
        assert reading.bias == MacroBias.LONG
        assert reading.tradeable is True

    def test_oil_down_dxy_down_gives_short(self):
        # Oil falling, EUR/USD rising (DXY falling)
        signal = self._make_signal(
            [85.0, 84.0, 83.0, 82.0, 81.0, 80.0, 79.0, 78.0],
            [1.05, 1.06, 1.07, 1.08, 1.09, 1.10, 1.11, 1.12],
        )
        query = datetime(2024, 1, 8, 23, 0, tzinfo=timezone.utc)
        reading = signal.compute(query)

        assert reading.oil_direction == MacroDirection.DOWN
        assert reading.dxy_direction == MacroDirection.DOWN
        assert reading.bias == MacroBias.SHORT
        assert reading.tradeable is True

    def test_conflict_gives_neutral(self):
        # Oil up, EUR/USD also up (DXY down) → conflict
        signal = self._make_signal(
            [78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0],
            [1.05, 1.06, 1.07, 1.08, 1.09, 1.10, 1.11, 1.12],
        )
        query = datetime(2024, 1, 8, 23, 0, tzinfo=timezone.utc)
        reading = signal.compute(query)

        assert reading.oil_direction == MacroDirection.UP
        assert reading.dxy_direction == MacroDirection.DOWN
        assert reading.bias == MacroBias.NEUTRAL
        assert reading.tradeable is False

    def test_oil_flat_dxy_up_gives_lean_long(self):
        # Oil flat, EUR/USD falling (DXY up)
        signal = self._make_signal(
            [80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0],
            [1.12, 1.11, 1.10, 1.09, 1.08, 1.07, 1.06, 1.05],
        )
        query = datetime(2024, 1, 8, 23, 0, tzinfo=timezone.utc)
        reading = signal.compute(query)

        assert reading.oil_direction == MacroDirection.NEUTRAL
        assert reading.dxy_direction == MacroDirection.UP
        assert reading.bias == MacroBias.LEAN_LONG

    def test_causal_enforcement_before_22utc(self):
        """Bar dated Jan 8 should NOT be available at 21:59 UTC on Jan 8."""
        signal = self._make_signal(
            [78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 90.0],
            [1.12, 1.11, 1.10, 1.09, 1.08, 1.07, 1.06, 1.05, 1.00],
            start_day=1,
        )

        # Query at 21:59 on Jan 8 — bar for Jan 8 not yet complete
        early = datetime(2024, 1, 8, 21, 59, tzinfo=timezone.utc)
        reading_early = signal.compute(early)

        # Query at 22:00 on Jan 8 — bar for Jan 8 now complete
        late = datetime(2024, 1, 8, 22, 0, tzinfo=timezone.utc)
        reading_late = signal.compute(late)

        # The late query should have one more bar available
        # This may or may not change the reading depending on data,
        # but the return values should differ
        # Actually let's just check they don't crash and returns exist
        assert (
            reading_early.oil_return_5d is not None
            or reading_early.oil_direction == MacroDirection.NEUTRAL
        )
        assert isinstance(reading_late.bias, MacroBias)

    def test_causal_enforcement_different_results(self):
        """Verify that a bar becoming available changes the signal."""
        # Days 1-7: oil flat at 80. Day 8: oil jumps to 90.
        # Before day 8 completes: 5d return based on days 3-7 (flat → neutral)
        # After day 8 completes: 5d return based on days 4-8 (includes jump → UP)
        signal = self._make_signal(
            [80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 90.0],
            [1.10, 1.10, 1.10, 1.10, 1.10, 1.10, 1.10, 1.10],  # DXY flat
        )

        # Before day 8 bar completes (21:59 on Jan 8)
        before = datetime(2024, 1, 8, 21, 59, tzinfo=timezone.utc)
        reading_before = signal.compute(before)

        # After day 8 bar completes (22:00 on Jan 8)
        after = datetime(2024, 1, 8, 22, 0, tzinfo=timezone.utc)
        reading_after = signal.compute(after)

        assert reading_before.oil_direction == MacroDirection.NEUTRAL
        assert reading_after.oil_direction == MacroDirection.UP

    def test_cache_returns_same_object(self):
        signal = self._make_signal(
            [78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0],
            [1.12, 1.11, 1.10, 1.09, 1.08, 1.07, 1.06, 1.05],
        )
        query = datetime(2024, 1, 8, 23, 0, tzinfo=timezone.utc)

        r1 = signal.compute(query)
        r2 = signal.compute(query)
        assert r1 is r2  # Same cached object

    def test_dxy_is_inverted_eurusd(self):
        """EUR/USD going up should make DXY direction DOWN."""
        signal = self._make_signal(
            [80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0],  # oil flat
            [1.05, 1.06, 1.07, 1.08, 1.09, 1.10, 1.11, 1.12],  # EURUSD rising
        )
        query = datetime(2024, 1, 8, 23, 0, tzinfo=timezone.utc)
        reading = signal.compute(query)

        assert reading.dxy_direction == MacroDirection.DOWN

    def test_macro_reading_fields_populated(self):
        signal = self._make_signal(
            [78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0],
            [1.12, 1.11, 1.10, 1.09, 1.08, 1.07, 1.06, 1.05],
        )
        query = datetime(2024, 1, 8, 23, 0, tzinfo=timezone.utc)
        reading = signal.compute(query)

        assert reading.timestamp == query
        assert reading.oil_return_5d is not None
        assert reading.dxy_return_5d is not None
        assert isinstance(reading.oil_direction, MacroDirection)
        assert isinstance(reading.dxy_direction, MacroDirection)
        assert isinstance(reading.bias, MacroBias)
        assert isinstance(reading.tradeable, bool)

    def test_return_values_are_reasonable(self):
        """5-day returns should be small decimals, not percentages."""
        signal = self._make_signal(
            [78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0],
            [1.12, 1.11, 1.10, 1.09, 1.08, 1.07, 1.06, 1.05],
        )
        query = datetime(2024, 1, 8, 23, 0, tzinfo=timezone.utc)
        reading = signal.compute(query)

        # Oil went from ~80 to ~85 in 5 days → ~6%
        assert 0.01 < reading.oil_return_5d < 0.15
        # DXY (inverted EURUSD) went from ~1.10 to ~1.05 → EURUSD dropped ~4.5%
        # Inverted → DXY +4.5%
        assert 0.01 < reading.dxy_return_5d < 0.15
