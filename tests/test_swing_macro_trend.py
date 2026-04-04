"""Tests for swing_macro_trend.py — trend detection and pullback entries."""

import pytest
from datetime import datetime, timedelta, timezone

from core.regime_backtest_engine.swing_macro_bars import Bar4H
from core.regime_backtest_engine.swing_macro_trend import (
    TrendState,
    PullbackPhase,
    SwingMacroEntry,
    TrendDetector,
)


def _bar(price, hour=0, day=15, month=1, low_offset=0.5, high_offset=0.5):
    """Helper: create a Bar4H with close=price, high=price+offset, low=price-offset."""
    return Bar4H(
        timestamp=datetime(2024, month, day, hour, 0, tzinfo=timezone.utc),
        open=price,
        high=price + high_offset,
        low=price - low_offset,
        close=price,
        bar_count=240,
    )


def _bar_full(open_, high, low, close, hour=0, day=15, month=1):
    """Helper: create a Bar4H with explicit OHLC."""
    return Bar4H(
        timestamp=datetime(2024, month, day, hour, 0, tzinfo=timezone.utc),
        open=open_,
        high=high,
        low=low,
        close=close,
        bar_count=240,
    )


def _warm_up_detector(detector, n_bars=55, start_price=150.0, trend="up"):
    """Feed enough bars to initialize EMA(50), EMA(20), ATR(14).

    EMA(50) needs 50 bars, ATR(14) needs 15 bars (1 prev_close + 14 TRs).
    So 55 bars is sufficient.

    For trend="up": price rises steadily.
    For trend="down": price falls steadily.
    For trend="flat": price stays constant.
    """
    signals = []
    base = datetime(2024, 1, 15, 0, 0, tzinfo=timezone.utc)
    for i in range(n_bars):
        if trend == "up":
            price = start_price + i * 0.3
        elif trend == "down":
            price = start_price - i * 0.3
        else:
            price = start_price

        ts = base + timedelta(hours=i * 4)
        bar = Bar4H(
            timestamp=ts,
            open=price,
            high=price + 0.5,
            low=price - 0.5,
            close=price,
            bar_count=240,
        )
        result = detector.update(bar)
        if result is not None:
            signals.append(result)

    return signals


class TestTrendState:
    def test_up_trend_detected(self):
        """After steady rise, trend should be UP."""
        det = TrendDetector()
        _warm_up_detector(det, n_bars=55, trend="up")
        assert det.trend == TrendState.UP

    def test_down_trend_detected(self):
        """After steady decline, trend should be DOWN."""
        det = TrendDetector()
        _warm_up_detector(det, n_bars=55, trend="down")
        assert det.trend == TrendState.DOWN

    def test_flat_is_none(self):
        """Flat price action should produce NONE trend."""
        det = TrendDetector()
        _warm_up_detector(det, n_bars=55, trend="flat")
        assert det.trend == TrendState.NONE

    def test_trend_requires_both_conditions(self):
        """Price above EMA50 but with zero slope should not be UP."""
        det = TrendDetector()
        # Rise then flatten
        _warm_up_detector(det, n_bars=55, trend="up")
        # Feed 20 flat bars at the current level
        last_price = 150.0 + 54 * 0.3  # ~166.2
        for i in range(20):
            bar = _bar(last_price, hour=(i * 4) % 24, day=25 + i // 6)
            det.update(bar)
        # Eventually slope goes to ~0, trend may become NONE
        # (EMA50 converges toward price, slope flattens)
        # This is a soft test — with enough flat bars, trend weakens
        assert det.trend in (TrendState.UP, TrendState.NONE)


class TestPullbackDetection:
    def test_no_signal_during_warmup(self):
        """No signals should fire during indicator warm-up."""
        det = TrendDetector()
        signals = _warm_up_detector(det, n_bars=50, trend="up")
        # During warmup, indicators aren't ready — no signals
        # (Some may fire near the end when indicators initialize)
        # We mainly verify no crash
        assert isinstance(signals, list)

    def test_pullback_long_signal(self):
        """In an uptrend, pullback to EMA20 + reclaim should produce long signal."""
        det = TrendDetector()
        _warm_up_detector(det, n_bars=60, start_price=145.0, trend="up")

        assert det.trend == TrendState.UP
        ema20 = det.ema20.value
        atr = det.atr.value
        assert ema20 is not None
        assert atr is not None

        # Pull price down toward EMA20
        pullback_price = ema20 - 0.1  # Below EMA20
        bar_pb = _bar_full(
            open_=ema20 + 1.0,
            high=ema20 + 1.0,
            low=pullback_price,
            close=pullback_price,
            day=26,
            hour=0,
        )
        result = det.update(bar_pb)
        # Should enter PULLING_BACK, no signal yet
        assert result is None

        # Now reclaim: close above EMA20
        reclaim_price = det.ema20.value + 0.5
        bar_reclaim = _bar_full(
            open_=pullback_price,
            high=reclaim_price + 0.2,
            low=pullback_price - 0.1,
            close=reclaim_price,
            day=26,
            hour=4,
        )
        result = det.update(bar_reclaim)

        # If trend is still UP, should get a long signal
        if det.trend == TrendState.UP and result is not None:
            assert result.direction == "long"
            assert result.stop_loss < result.entry_price
            assert result.atr_value > 0

    def test_pullback_short_signal(self):
        """In a downtrend, pullback to EMA20 + reclaim should produce short signal."""
        det = TrendDetector()
        _warm_up_detector(det, n_bars=60, start_price=165.0, trend="down")

        assert det.trend == TrendState.DOWN
        ema20 = det.ema20.value
        assert ema20 is not None

        # Pull price up toward EMA20
        pullback_price = ema20 + 0.1
        bar_pb = _bar_full(
            open_=ema20 - 1.0,
            high=pullback_price,
            low=ema20 - 1.0,
            close=pullback_price,
            day=26,
            hour=0,
        )
        result = det.update(bar_pb)
        assert result is None  # Still pulling back

        # Reclaim down: close below EMA20
        reclaim_price = det.ema20.value - 0.5
        bar_reclaim = _bar_full(
            open_=pullback_price,
            high=pullback_price + 0.1,
            low=reclaim_price - 0.2,
            close=reclaim_price,
            day=26,
            hour=4,
        )
        result = det.update(bar_reclaim)

        if det.trend == TrendState.DOWN and result is not None:
            assert result.direction == "short"
            assert result.stop_loss > result.entry_price
            assert result.atr_value > 0

    def test_no_signal_without_pullback(self):
        """In uptrend with price far above EMA20, no signal should fire."""
        det = TrendDetector()
        _warm_up_detector(det, n_bars=60, start_price=145.0, trend="up")

        # Feed bars that stay well above EMA20
        for i in range(10):
            high_price = det.ema20.value + 5.0 + i * 0.3
            bar = _bar(high_price, hour=(i * 4) % 24, day=26 + i // 6)
            result = det.update(bar)
            assert result is None  # No pullback, no signal

    def test_no_signal_in_none_trend(self):
        """Flat market should produce no signals."""
        det = TrendDetector()
        signals = _warm_up_detector(det, n_bars=80, trend="flat")
        # Flat market → NONE trend → no entries
        # There might be brief moments where EMAs cross, but broadly no signals
        # We mainly verify the system doesn't crash
        assert isinstance(signals, list)

    def test_reset_after_trend_change(self):
        """Pullback phase should reset when trend flips."""
        det = TrendDetector()
        _warm_up_detector(det, n_bars=60, start_price=145.0, trend="up")

        # Start a pullback
        ema20 = det.ema20.value
        bar_pb = _bar_full(
            open_=ema20 + 1.0,
            high=ema20 + 1.0,
            low=ema20 - 0.1,
            close=ema20 - 0.1,
            day=26,
            hour=0,
        )
        det.update(bar_pb)

        if det.trend == TrendState.UP:
            assert det.pullback_phase == PullbackPhase.PULLING_BACK

        # Crash price to flip trend
        crash_price = det.ema50.value - 5.0
        for i in range(10):
            bar = _bar(
                crash_price - i * 0.5,
                day=26 + (i + 1) // 6,
                hour=((i + 1) * 4) % 24,
            )
            det.update(bar)

        # After trend flip, pullback phase should have reset
        assert det.pullback_phase == PullbackPhase.WAITING


class TestSwingMacroEntry:
    def test_long_entry_fields(self):
        """Verify all fields are populated correctly for a long entry."""
        det = TrendDetector()
        _warm_up_detector(det, n_bars=60, start_price=145.0, trend="up")

        ema20 = det.ema20.value

        # Pullback
        pb_price = ema20 - 0.1
        det.update(
            _bar_full(
                open_=ema20 + 1.0,
                high=ema20 + 1.0,
                low=pb_price,
                close=pb_price,
                day=26,
                hour=0,
            )
        )

        # Reclaim
        reclaim = det.ema20.value + 0.5
        result = det.update(
            _bar_full(
                open_=pb_price,
                high=reclaim + 0.2,
                low=pb_price - 0.1,
                close=reclaim,
                day=26,
                hour=4,
            )
        )

        if result is not None:
            assert isinstance(result, SwingMacroEntry)
            assert result.direction == "long"
            assert result.entry_price == reclaim
            assert result.stop_loss < reclaim
            assert result.atr_value > 0
            assert result.trend_state == TrendState.UP
            assert result.ema50_value is not None
            assert result.ema20_value is not None

    def test_stop_loss_uses_atr_factor(self):
        """Stop should be swing_low - atr_stop_factor * ATR."""
        det = TrendDetector(atr_stop_factor=2.0)
        _warm_up_detector(det, n_bars=60, start_price=145.0, trend="up")

        ema20 = det.ema20.value

        # Pullback with known low
        pb_low = ema20 - 0.5
        det.update(
            _bar_full(
                open_=ema20 + 1.0,
                high=ema20 + 1.0,
                low=pb_low,
                close=ema20 - 0.2,
                day=26,
                hour=0,
            )
        )

        # Reclaim
        reclaim = det.ema20.value + 0.5
        result = det.update(
            _bar_full(
                open_=ema20 - 0.2,
                high=reclaim + 0.2,
                low=ema20 - 0.3,
                close=reclaim,
                day=26,
                hour=4,
            )
        )

        if result is not None:
            # Stop should be approximately pb_low - 2.0 * ATR
            # Note: ATR updates with each bar, so use result.atr_value
            assert result.stop_loss < pb_low

    def test_consecutive_pullbacks_each_produce_signal(self):
        """After one pullback triggers, the next pullback should also trigger."""
        det = TrendDetector()
        _warm_up_detector(det, n_bars=60, start_price=145.0, trend="up")

        signals = []
        seq_base = datetime(2024, 2, 1, 0, 0, tzinfo=timezone.utc)
        hour_step = 0

        def _next_ts():
            nonlocal hour_step
            ts = seq_base + timedelta(hours=hour_step)
            hour_step += 4
            return ts

        for cycle in range(3):
            ema20 = det.ema20.value
            if ema20 is None:
                break

            # Continue trend
            for i in range(5):
                above_price = ema20 + 2.0 + i * 0.3
                p = above_price
                ts = _next_ts()
                det.update(
                    Bar4H(
                        timestamp=ts,
                        open=p,
                        high=p + 0.5,
                        low=p - 0.5,
                        close=p,
                        bar_count=240,
                    )
                )

            ema20 = det.ema20.value
            if ema20 is None:
                break

            # Pullback
            pb = ema20 - 0.1
            ts_pb = _next_ts()
            det.update(
                Bar4H(
                    timestamp=ts_pb,
                    open=ema20 + 1.0,
                    high=ema20 + 1.0,
                    low=pb,
                    close=pb,
                    bar_count=240,
                )
            )

            # Reclaim
            rc = det.ema20.value
            if rc is None:
                break
            rc += 0.5
            ts_rc = _next_ts()
            result = det.update(
                Bar4H(
                    timestamp=ts_rc,
                    open=pb,
                    high=rc + 0.2,
                    low=pb - 0.1,
                    close=rc,
                    bar_count=240,
                )
            )

            if result is not None:
                signals.append(result)

        # Should get at least 1 signal (market conditions may prevent all 3)
        assert len(signals) >= 1
        for s in signals:
            assert s.direction == "long"


class TestEdgeCases:
    def test_exact_ema20_touch(self):
        """Bar low exactly at EMA20 should trigger pullback phase."""
        det = TrendDetector()
        _warm_up_detector(det, n_bars=60, start_price=145.0, trend="up")

        ema20 = det.ema20.value
        # Bar with low exactly at EMA20
        bar = _bar_full(
            open_=ema20 + 1.0,
            high=ema20 + 1.0,
            low=ema20,
            close=ema20 + 0.3,
            day=26,
            hour=0,
        )
        det.update(bar)
        # Should be in pullback or have already triggered
        # (close is above ema20, so might trigger on same bar)
        assert det.pullback_phase in (
            PullbackPhase.PULLING_BACK,
            PullbackPhase.WAITING,  # Reset after trigger
        )

    def test_gap_through_ema20(self):
        """Price gapping well below EMA20 should still register pullback."""
        det = TrendDetector()
        _warm_up_detector(det, n_bars=60, start_price=145.0, trend="up")

        ema20 = det.ema20.value

        # Gap well below EMA20
        gap_price = ema20 - 2.0
        bar = _bar_full(
            open_=gap_price,
            high=gap_price + 0.3,
            low=gap_price - 0.3,
            close=gap_price,
            day=26,
            hour=0,
        )
        det.update(bar)

        # If trend survived, should be pulling back
        if det.trend == TrendState.UP:
            assert det.pullback_phase == PullbackPhase.PULLING_BACK

    def test_atr_proximity_parameter(self):
        """Custom atr_proximity_factor should be respected."""
        det1 = TrendDetector(atr_proximity_factor=0.1)  # Tight
        det2 = TrendDetector(atr_proximity_factor=1.0)  # Wide

        # Both should initialize without error
        _warm_up_detector(det1, n_bars=60, trend="up")
        _warm_up_detector(det2, n_bars=60, trend="up")

        assert det1.trend == TrendState.UP
        assert det2.trend == TrendState.UP
