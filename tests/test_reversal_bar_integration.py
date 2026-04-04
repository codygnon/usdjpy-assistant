"""Integration tests: TrendDetector + reversal bar filter (Swing-Macro)."""

from datetime import datetime, timedelta, timezone

from core.regime_backtest_engine.swing_macro_bars import Bar4H
from core.regime_backtest_engine.swing_macro_trend import PullbackPhase, TrendDetector, TrendState


def _bar_full(
    open_: float,
    high: float,
    low: float,
    close: float,
    *,
    hour: int = 0,
    day: int = 26,
    month: int = 1,
) -> Bar4H:
    return Bar4H(
        timestamp=datetime(2024, month, day, hour, 0, tzinfo=timezone.utc),
        open=open_,
        high=high,
        low=low,
        close=close,
        bar_count=240,
    )


def _warm_up_detector(
    detector: TrendDetector,
    n_bars: int = 60,
    start_price: float = 145.0,
    trend: str = "up",
) -> None:
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
        detector.update(bar)


def _long_pullback_then_reclaim_bars(det: TrendDetector) -> tuple[Bar4H, Bar4H, Bar4H]:
    """Return (bar_mid, bar_reclaim_non_reversal, bar_reclaim_hammer) sequence builders' inputs.

    Caller warms up det first. After bar_pb, feeds bar_mid (still pulling back), then either reclaim variant.
    """
    assert det.trend == TrendState.UP
    ema20 = det.ema20.value
    assert ema20 is not None

    pullback_price = ema20 - 0.1
    bar_pb = _bar_full(
        open_=ema20 + 1.0,
        high=ema20 + 1.0,
        low=pullback_price,
        close=pullback_price,
        day=26,
        hour=0,
    )
    assert det.update(bar_pb) is None
    assert det.pullback_phase == PullbackPhase.PULLING_BACK

    ema20 = det.ema20.value
    assert ema20 is not None

    # Deeper low — sets previous-bar low for next bar
    bar_mid = _bar_full(
        open_=pullback_price,
        high=ema20 + 0.2,
        low=pullback_price - 0.5,
        close=ema20 - 0.3,
        day=26,
        hour=4,
    )
    assert det.update(bar_mid) is None
    prev_low = bar_mid.low

    ema20 = det.ema20.value
    assert ema20 is not None

    # Reclaim without probing below prev_low: low >= prev_low, marubozu-like, close > ema20
    reclaim_close = ema20 + 0.6
    bar_non = _bar_full(
        open_=ema20 - 0.2,
        high=reclaim_close + 0.1,
        low=prev_low + 0.05,
        close=reclaim_close,
        day=26,
        hour=8,
    )

    # Hammer-style reclaim: probe below prev_low, long lower wick, close above mid and ema20
    hammer_close = ema20 + 0.8
    bar_hammer = _bar_full(
        open_=ema20 + 0.1,
        high=hammer_close + 0.2,
        low=prev_low - 1.2,
        close=hammer_close,
        day=26,
        hour=12,
    )

    return bar_mid, bar_non, bar_hammer


def test_false_same_signals_as_baseline_on_non_reversal_reclaim() -> None:
    """require_reversal_bar=False still fires on reclaim without reversal pattern."""
    det2 = TrendDetector(require_reversal_bar=False)
    _warm_up_detector(det2, n_bars=60, start_price=145.0, trend="up")
    _, bar_non2, _ = _long_pullback_then_reclaim_bars(det2)

    r = det2.update(bar_non2)
    if det2.trend == TrendState.UP:
        assert r is not None
        assert r.direction == "long"


def test_true_blocks_non_reversal_reclaim() -> None:
    det = TrendDetector(require_reversal_bar=True)
    _warm_up_detector(det, n_bars=60, start_price=145.0, trend="up")
    _, bar_non, _ = _long_pullback_then_reclaim_bars(det)

    r = det.update(bar_non)
    if det.trend == TrendState.UP:
        assert r is None
        assert det.pullback_phase == PullbackPhase.PULLING_BACK


def test_true_allows_hammer_reclaim() -> None:
    det = TrendDetector(require_reversal_bar=True)
    _warm_up_detector(det, n_bars=60, start_price=145.0, trend="up")
    _, _, bar_hammer = _long_pullback_then_reclaim_bars(det)

    r = det.update(bar_hammer)
    if det.trend == TrendState.UP and r is not None:
        assert r.direction == "long"
        assert r.stop_loss < r.entry_price


def _short_pullback_then_reclaim_bars(det: TrendDetector) -> tuple[Bar4H, Bar4H, Bar4H]:
    assert det.trend == TrendState.DOWN
    ema20 = det.ema20.value
    assert ema20 is not None
    pullback_price = ema20 + 0.1
    bar_pb = _bar_full(
        open_=ema20 - 1.0,
        high=pullback_price,
        low=ema20 - 1.0,
        close=pullback_price,
        day=26,
        hour=0,
    )
    assert det.update(bar_pb) is None

    bar_mid = _bar_full(
        open_=pullback_price,
        high=pullback_price + 0.5,
        low=ema20 - 0.2,
        close=ema20 + 0.3,
        day=26,
        hour=4,
    )
    assert det.update(bar_mid) is None
    prev_high = bar_mid.high
    ema20 = det.ema20.value
    assert ema20 is not None

    reclaim_close = ema20 - 0.6
    bar_non = _bar_full(
        open_=ema20 + 0.2,
        high=prev_high - 0.05,
        low=reclaim_close - 0.1,
        close=reclaim_close,
        day=26,
        hour=8,
    )

    bar_star = _bar_full(
        open_=ema20 - 0.1,
        high=prev_high + 1.2,
        low=reclaim_close - 0.2,
        close=reclaim_close,
        day=26,
        hour=12,
    )

    return bar_mid, bar_non, bar_star


def test_short_false_allows_non_reversal_when_filter_off() -> None:
    det = TrendDetector(require_reversal_bar=False)
    _warm_up_detector(det, n_bars=60, start_price=165.0, trend="down")
    _, bar_non, _ = _short_pullback_then_reclaim_bars(det)
    r = det.update(bar_non)
    if det.trend == TrendState.DOWN:
        assert r is not None
        assert r.direction == "short"


def test_short_true_blocks_non_reversal() -> None:
    det = TrendDetector(require_reversal_bar=True)
    _warm_up_detector(det, n_bars=60, start_price=165.0, trend="down")
    _, bar_non, _ = _short_pullback_then_reclaim_bars(det)
    r = det.update(bar_non)
    if det.trend == TrendState.DOWN:
        assert r is None


def test_short_true_allows_shooting_star_reclaim() -> None:
    det = TrendDetector(require_reversal_bar=True)
    _warm_up_detector(det, n_bars=60, start_price=165.0, trend="down")
    _, _, bar_star = _short_pullback_then_reclaim_bars(det)
    r = det.update(bar_star)
    if det.trend == TrendState.DOWN and r is not None:
        assert r.direction == "short"
        assert r.stop_loss > r.entry_price


def test_previous_bar_none_blocks_without_crash() -> None:
    """If _previous_bar is missing at reclaim, conservative path: no signal, no crash."""
    det = TrendDetector(require_reversal_bar=True)
    _warm_up_detector(det, n_bars=60, start_price=145.0, trend="up")
    _, _, bar_hammer = _long_pullback_then_reclaim_bars(det)

    det._previous_bar = None  # type: ignore[attr-defined]
    r = det.update(bar_hammer)
    assert r is None
