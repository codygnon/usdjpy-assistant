"""Unit tests for DailyTrendDetector."""

from datetime import datetime, timedelta, timezone

import pytest

from core.regime_backtest_engine.daily_trend_bars import DailyBar
from core.regime_backtest_engine.daily_trend_detector import (
    DailyTrend,
    DailyTrendDetector,
    PullbackState,
)
from core.regime_backtest_engine.swing_macro_bars import Bar4H

UTC = timezone.utc


def _d(
    seq_day: int,
    close: float,
    *,
    h_off: float = 0.5,
    l_off: float = 0.5,
    idx: int = 0,
) -> DailyBar:
    """seq_day: 0-based day offset from 2024-01-01 22:00 UTC."""
    ts = datetime(2024, 1, 1, 22, 0, tzinfo=UTC) + timedelta(days=seq_day)
    return DailyBar(
        timestamp=ts,
        open=close,
        high=close + h_off,
        low=close - l_off,
        close=close,
        bar_index=idx,
    )


def _h4(day: int, hour: int, o: float, h: float, l: float, c: float) -> Bar4H:
    ts = datetime(2024, 2, day, hour, 0, tzinfo=UTC)
    return Bar4H(timestamp=ts, open=o, high=h, low=l, close=c, bar_count=60)


def _warmup_uptrend(det: DailyTrendDetector, n: int = 55) -> None:
    base = 100.0
    for i in range(n):
        det.update_daily(_d(i, base + i * 0.5, idx=i))


class TestWarmupAndTrend:
    def test_not_warmed_before_slow_period(self) -> None:
        det = DailyTrendDetector(daily_ema_slow=50)
        for i in range(49):
            det.update_daily(_d(i, 100.0 + i * 0.01, idx=i))
        assert det.is_warmed_up is False
        det.update_daily(_d(49, 101.0, idx=49))
        assert det.is_warmed_up is True

    def test_up_trend_monotonic(self) -> None:
        det = DailyTrendDetector(daily_ema_slow=5, daily_ema_fast=3)
        for i in range(30):
            det.update_daily(_d(i, 100.0 + i * 1.0, idx=i))
        assert det.trend == DailyTrend.UP

    def test_down_trend_monotonic(self) -> None:
        det = DailyTrendDetector(daily_ema_slow=5, daily_ema_fast=3)
        for i in range(30):
            det.update_daily(_d(i, 200.0 - i * 1.0, idx=i))
        assert det.trend == DailyTrend.DOWN

    def test_mixed_close_above_falling_ema_none(self) -> None:
        det = DailyTrendDetector(daily_ema_slow=5, daily_ema_fast=3)
        # Perfect flat: EMA50 flat (not rising/falling), close == EMA → NONE
        for i in range(40):
            det.update_daily(_d(i, 100.0, idx=i))
        assert det.trend == DailyTrend.NONE

    def test_trend_transitions(self) -> None:
        det = DailyTrendDetector(daily_ema_slow=5, daily_ema_fast=3)
        for i in range(20):
            det.update_daily(_d(i, 100.0 + i * 2.0, idx=i))
        assert det.trend == DailyTrend.UP
        for j in range(15):
            det.update_daily(_d(20 + j, 200.0 - j * 5.0, idx=20 + j))
        assert det.trend in (DailyTrend.DOWN, DailyTrend.NONE)


class TestLongPullback:
    def test_touch_zone_activates(self) -> None:
        det = DailyTrendDetector(daily_ema_slow=5, daily_ema_fast=3, proximity_atr=1.0)
        _warmup_uptrend(det)
        assert det.trend == DailyTrend.UP
        ema20 = det.ema20
        assert ema20 is not None
        atr = det.daily_atr
        assert atr is not None
        det.update_4h(_h4(1, 0, ema20 + 2, ema20 + 2, ema20 + 2, ema20 + 2))
        assert det._pullback == PullbackState.NONE  # type: ignore[attr-defined]
        # Touch EMA zone but close still below EMA(20) — ACTIVE, no same-bar entry
        det.update_4h(_h4(1, 4, ema20 + 1, ema20 + 1, ema20 - 0.01, ema20 - 0.2))
        assert det._pullback == PullbackState.ACTIVE  # type: ignore[attr-defined]

    def test_active_close_below_ema20_no_entry(self) -> None:
        det = DailyTrendDetector(daily_ema_slow=5, daily_ema_fast=3, proximity_atr=2.0)
        _warmup_uptrend(det)
        ema20 = float(det.ema20)
        det.update_4h(_h4(1, 0, ema20 + 1, ema20 + 1, ema20 - 1, ema20 - 0.5))
        r = det.update_4h(_h4(1, 4, ema20, ema20, ema20 - 0.5, ema20 - 0.2))
        assert r is None

    def test_active_reclaim_entry(self) -> None:
        det = DailyTrendDetector(daily_ema_slow=5, daily_ema_fast=3, proximity_atr=2.0)
        _warmup_uptrend(det)
        ema20 = float(det.ema20)
        for k in range(4):
            det.update_4h(_h4(2, k * 4, ema20 + 2, ema20 + 2, ema20 + 1.5, ema20 + 1.8))
        det.update_4h(_h4(2, 16, ema20 + 1, ema20 + 1, ema20 - 2, ema20 - 0.5))
        sig = det.update_4h(_h4(2, 20, ema20, ema20 + 3, ema20 - 0.5, ema20 + 1.0))
        assert sig is not None
        assert sig.direction == "long"
        assert sig.stop_loss < sig.entry_price

    def test_expiry_no_entry(self) -> None:
        det = DailyTrendDetector(daily_ema_slow=5, daily_ema_fast=3, proximity_atr=5.0)
        _warmup_uptrend(det)
        ema20 = float(det.ema20)
        atr = float(det.daily_atr)
        det.update_4h(_h4(3, 0, ema20 + 1, ema20 + 1, ema20 - 1, ema20))
        assert det._pullback == PullbackState.ACTIVE  # type: ignore[attr-defined]
        plunge = ema20 - 2.5 * atr
        r = det.update_4h(_h4(3, 4, ema20, ema20, plunge, plunge))
        assert r is None
        assert det._pullback == PullbackState.EXPIRED  # type: ignore[attr-defined]

    def test_entry_resets_pullback(self) -> None:
        det = DailyTrendDetector(daily_ema_slow=5, daily_ema_fast=3, proximity_atr=2.0)
        _warmup_uptrend(det)
        ema20 = float(det.ema20)
        det.update_4h(_h4(4, 0, ema20 + 1, ema20 + 1, ema20 - 2, ema20 - 0.5))
        det.update_4h(_h4(4, 4, ema20, ema20 + 2, ema20 - 0.5, ema20 + 0.5))
        assert det._pullback == PullbackState.NONE  # type: ignore[attr-defined]

    def test_none_trend_no_touch_entry(self) -> None:
        det = DailyTrendDetector(daily_ema_slow=5, daily_ema_fast=3)
        for i in range(10):
            det.update_daily(_d(i, 100.0, idx=i))
        ema20 = det.ema20
        if ema20 is None:
            pytest.skip("ema not ready")
        det.update_4h(_h4(5, 0, float(ema20), float(ema20), float(ema20) - 1, float(ema20)))
        assert det.update_4h(_h4(5, 4, float(ema20), float(ema20) + 1, float(ema20), float(ema20) + 0.5)) is None


class TestShortPullback:
    def test_short_touch_and_entry(self) -> None:
        det = DailyTrendDetector(daily_ema_slow=5, daily_ema_fast=3, proximity_atr=2.0)
        for i in range(30):
            det.update_daily(_d(i, 200.0 - i * 1.0, idx=i))
        assert det.trend == DailyTrend.DOWN
        ema20 = float(det.ema20)
        det.update_4h(_h4(6, 0, ema20 - 1, ema20 + 1, ema20 - 2, ema20 - 1))
        sig = det.update_4h(_h4(6, 4, ema20 + 0.5, ema20 + 0.5, ema20 - 1, ema20 - 0.5))
        if sig is None:
            sig = det.update_4h(_h4(6, 8, ema20, ema20 + 0.3, ema20 - 2, ema20 - 0.8))
        assert sig is not None
        assert sig.direction == "short"
        assert sig.stop_loss > sig.entry_price

    def test_short_expiry(self) -> None:
        det = DailyTrendDetector(daily_ema_slow=5, daily_ema_fast=3, proximity_atr=5.0)
        for i in range(30):
            det.update_daily(_d(i, 200.0 - i * 1.0, idx=i))
        ema20 = float(det.ema20)
        atr = float(det.daily_atr)
        det.update_4h(_h4(7, 0, ema20 - 1, ema20 + 2, ema20 - 2, ema20))
        top = ema20 + 2.5 * atr
        r = det.update_4h(_h4(7, 4, ema20, top, ema20 - 0.5, top))
        assert r is None


class TestStopsAndTrailing:
    def test_long_stop_uses_swing_and_factor(self) -> None:
        det = DailyTrendDetector(
            daily_ema_slow=5,
            daily_ema_fast=3,
            proximity_atr=2.0,
            stop_atr_factor=2.0,
        )
        _warmup_uptrend(det)
        ema20 = float(det.ema20)
        atr = float(det.daily_atr)
        lows = [120.0, 119.0, 118.0, 117.0, 116.0]
        for i, lo in enumerate(lows):
            det.update_4h(
                _h4(10, i * 4, ema20 + 2, ema20 + 2, lo, ema20 + 1.5)
            )
        det.update_4h(_h4(10, 20, ema20 + 1, ema20 + 1, ema20 - 2, ema20 - 0.5))
        sig = det.update_4h(_h4(11, 0, ema20, ema20 + 2, ema20 - 0.5, ema20 + 1))
        assert sig is not None
        expect = min(lows) - 2.0 * atr
        assert abs(sig.stop_loss - expect) < 1e-6

    def test_trailing_long(self) -> None:
        det = DailyTrendDetector(daily_ema_slow=5, daily_ema_fast=3, trail_buffer_atr=0.25)
        for i in range(25):
            det.update_daily(_d(i, 100.0 + i * 0.2, l_off=0.2, idx=i))
        t = det.compute_trailing_stop_long()
        assert t is not None
        atr = float(det.daily_atr)
        recent_low = min(b.low for b in det._recent_daily[-5:])  # type: ignore[attr-defined]
        assert abs(t - (recent_low - 0.25 * atr)) < 1e-5

    def test_trailing_none_if_not_enough_dailies(self) -> None:
        det = DailyTrendDetector(trail_lookback=99)
        det.update_daily(_d(0, 100.0, idx=0))
        assert det.compute_trailing_stop_long() is None

    def test_trailing_respects_buffer(self) -> None:
        det = DailyTrendDetector(daily_ema_slow=5, daily_ema_fast=3, trail_buffer_atr=1.0)
        for i in range(25):
            det.update_daily(_d(i, 100.0 + i, idx=i))
        t = det.compute_trailing_stop_short()
        assert t is not None


class TestTrendResetPullback:
    def test_pullback_cleared_on_trend_change(self) -> None:
        det = DailyTrendDetector(daily_ema_slow=5, daily_ema_fast=3, proximity_atr=5.0)
        _warmup_uptrend(det)
        ema20 = float(det.ema20)
        det.update_4h(_h4(11, 0, ema20 + 1, ema20 + 1, ema20 - 1, ema20))
        assert det._pullback == PullbackState.ACTIVE  # type: ignore[attr-defined]
        det.update_daily(_d(200, 50.0, idx=999))
        for j in range(10):
            det.update_daily(_d(201 + j, 40.0 - j, idx=1000 + j))
        assert det._pullback == PullbackState.NONE  # type: ignore[attr-defined]


class TestEntryMetadata:
    def test_metadata_fields(self) -> None:
        det = DailyTrendDetector(daily_ema_slow=5, daily_ema_fast=3, proximity_atr=2.0)
        _warmup_uptrend(det)
        ema20 = float(det.ema20)
        det.update_4h(_h4(12, 0, ema20 + 1, ema20 + 1, ema20 - 2, ema20 - 0.5))
        sig = det.update_4h(_h4(12, 4, ema20, ema20 + 2, ema20 - 0.5, ema20 + 0.5))
        assert sig is not None
        assert "swing_low" in sig.metadata
        assert sig.metadata["trend"] == "UP"
        assert sig.daily_atr > 0
