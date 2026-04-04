"""Unit tests for bullish/bearish 4H reversal bar helpers (Swing-Macro)."""

from datetime import datetime, timezone

from core.regime_backtest_engine.swing_macro_bars import Bar4H
from core.regime_backtest_engine.swing_macro_trend import (
    DEFAULT_REVERSAL_WICK_RATIO,
    is_bearish_reversal_bar,
    is_bullish_reversal_bar,
)

UTC = timezone.utc


def _b(
    o: float,
    h: float,
    l: float,
    c: float,
    *,
    day: int = 1,
) -> Bar4H:
    return Bar4H(
        timestamp=datetime(2024, 1, day, 0, 0, tzinfo=UTC),
        open=o,
        high=h,
        low=l,
        close=c,
        bar_count=240,
    )


class TestBullishReversalBar:
    def test_classic_hammer_at_ema20(self) -> None:
        prev = _b(102.0, 102.5, 100.5, 101.0, day=1)
        # Probe below prev low 100.5, long lower wick, close above mid and EMA(20)
        cur = _b(101.5, 104.0, 98.0, 103.5, day=2)
        ema20 = 101.0
        assert is_bullish_reversal_bar(cur, prev, ema20) is True

    def test_long_lower_wick_above_ema(self) -> None:
        prev = _b(100.0, 101.0, 99.0, 100.5, day=1)
        cur = _b(100.2, 101.5, 97.5, 101.2, day=2)
        ema20 = 100.8
        assert is_bullish_reversal_bar(cur, prev, ema20) is True

    def test_doji_probe_below_prior_low_close_above_ema(self) -> None:
        prev = _b(100.0, 101.0, 100.0, 100.5, day=1)
        o = c = 102.0
        cur = _b(o, 102.5, 99.5, c, day=2)
        ema20 = 101.0
        assert is_bullish_reversal_bar(cur, prev, ema20) is True

    def test_fail_no_probe_lower(self) -> None:
        prev = _b(100.0, 102.0, 98.0, 101.0, day=1)
        cur = _b(101.0, 104.0, 98.5, 103.0, day=2)  # low 98.5 >= prev low 98.0 is False — actually 98.5 >= 98
        # prev low 98, cur low 98.5 — did not probe lower
        assert is_bullish_reversal_bar(cur, prev, 100.0) is False

    def test_fail_close_below_midrange(self) -> None:
        prev = _b(100.0, 101.0, 99.0, 100.0, day=1)
        # Mid = (102+98)/2 = 100, close must be > 100
        cur = _b(101.0, 102.0, 98.0, 100.0, day=2)
        assert is_bullish_reversal_bar(cur, prev, 99.0) is False

    def test_fail_close_below_ema20(self) -> None:
        prev = _b(100.0, 101.0, 99.0, 100.0, day=1)
        cur = _b(100.5, 102.0, 98.5, 100.8, day=2)
        ema20 = 101.0
        assert is_bullish_reversal_bar(cur, prev, ema20) is False

    def test_fail_wick_too_short(self) -> None:
        prev = _b(100.0, 101.0, 99.5, 100.0, day=1)
        # Body 2.0, need lower wick >= 3.0 at 1.5×
        cur = _b(101.0, 103.0, 99.0, 103.0, day=2)
        assert is_bullish_reversal_bar(cur, prev, 100.0) is False

    def test_fail_all_but_probe(self) -> None:
        prev = _b(100.0, 101.0, 99.0, 100.0, day=1)
        cur = _b(100.5, 101.0, 98.5, 100.2, day=2)  # probes, but close not above mid/ema
        assert is_bullish_reversal_bar(cur, prev, 100.4) is False

    def test_fail_all_but_wick_ratio(self) -> None:
        prev = _b(100.0, 101.0, 99.0, 100.0, day=1)
        body = 1.0
        # Lower wick 1.0 < 1.5 * body
        cur = _b(101.0, 102.0, 100.0, 102.0, day=2)
        assert is_bullish_reversal_bar(cur, prev, 100.5) is False

    def test_perfect_doji_body_zero_passes_wick(self) -> None:
        prev = _b(100.0, 101.0, 99.0, 100.0, day=1)
        cur = _b(102.0, 102.5, 98.5, 102.0, day=2)
        assert is_bullish_reversal_bar(cur, prev, 101.0) is True

    def test_very_small_body_ratio_applies(self) -> None:
        prev = _b(100.0, 101.0, 99.0, 100.0, day=1)
        cur = _b(100.0, 101.0, 99.0, 100.001, day=2)
        # body 0.001, wick min(o,c)-l = 99.0-99.0=0 if l=99... use l=98.999
        cur2 = _b(100.0, 101.0, 98.999, 100.001, day=3)
        assert is_bullish_reversal_bar(cur2, prev, 99.5) is True

    def test_open_close_equal_low_no_lower_wick_bullish_false(self) -> None:
        prev = _b(100.0, 101.0, 99.0, 100.0, day=1)
        cur = _b(102.0, 103.0, 102.0, 102.0, day=2)
        assert is_bullish_reversal_bar(cur, prev, 101.0) is False


class TestBearishReversalBar:
    def test_classic_shooting_star_at_ema20(self) -> None:
        prev = _b(100.0, 101.0, 99.5, 100.5, day=1)
        cur = _b(100.5, 103.0, 99.0, 99.2, day=2)
        ema20 = 100.0
        assert is_bearish_reversal_bar(cur, prev, ema20) is True

    def test_probe_higher_long_upper_wick(self) -> None:
        prev = _b(100.0, 101.5, 99.5, 101.0, day=1)
        cur = _b(101.0, 104.0, 100.0, 99.5, day=2)
        ema20 = 100.5
        assert is_bearish_reversal_bar(cur, prev, ema20) is True

    def test_doji_probe_above_prior_high_close_below_ema(self) -> None:
        prev = _b(100.0, 101.0, 99.5, 100.5, day=1)
        o = c = 99.0
        cur = _b(o, 101.5, 98.5, c, day=2)
        ema20 = 99.5
        assert is_bearish_reversal_bar(cur, prev, ema20) is True

    def test_fail_no_probe_higher(self) -> None:
        prev = _b(100.0, 104.0, 99.0, 103.0, day=1)
        cur = _b(102.0, 103.5, 101.0, 101.5, day=2)
        assert is_bearish_reversal_bar(cur, prev, 102.0) is False

    def test_fail_close_above_midrange(self) -> None:
        prev = _b(100.0, 102.0, 99.0, 101.0, day=1)
        cur = _b(100.0, 103.0, 99.0, 101.0, day=2)
        assert is_bearish_reversal_bar(cur, prev, 100.5) is False

    def test_fail_close_above_ema20(self) -> None:
        prev = _b(100.0, 102.0, 99.0, 101.0, day=1)
        cur = _b(99.0, 103.0, 98.5, 100.0, day=2)
        ema20 = 99.5
        assert is_bearish_reversal_bar(cur, prev, ema20) is False

    def test_fail_upper_wick_too_short(self) -> None:
        prev = _b(100.0, 101.0, 99.0, 100.5, day=1)
        # Body = 1.0; upper wick 1.49 < 1.5 × 1.0
        cur = _b(100.0, 101.49, 99.5, 99.0, day=2)
        assert is_bearish_reversal_bar(cur, prev, 99.8) is False

    def test_open_close_equal_high_no_upper_wick_bearish_false(self) -> None:
        prev = _b(100.0, 101.0, 99.0, 100.5, day=1)
        cur = _b(98.0, 98.0, 97.0, 98.0, day=2)
        assert is_bearish_reversal_bar(cur, prev, 99.0) is False

    def test_body_zero_upper_wick_zero_other_rules_fail(self) -> None:
        prev = _b(100.0, 101.0, 99.0, 100.0, day=1)
        cur = _b(99.0, 99.0, 99.0, 99.0, day=2)
        assert is_bearish_reversal_bar(cur, prev, 100.0) is False


def test_wick_ratio_parameter_respected() -> None:
    prev = _b(100.0, 101.0, 99.0, 100.0, day=1)
    # Body = 2.0; lower wick 2.9 fails at 1.5× but passes at 1.0×
    cur = _b(101.0, 103.0, 98.1, 103.0, day=2)
    assert (
        is_bullish_reversal_bar(cur, prev, 100.0, wick_ratio=DEFAULT_REVERSAL_WICK_RATIO)
        is False
    )
    assert is_bullish_reversal_bar(cur, prev, 100.0, wick_ratio=1.0) is True
