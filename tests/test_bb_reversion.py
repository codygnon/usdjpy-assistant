from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.regime_backtest_engine.bb_reversion import (
    BBReversionStrategy,
    _IndicatorSnapshot,
    bb_reversion_session_open_utc,
    compute_bollinger_bands,
)
from core.regime_backtest_engine.models import ExitAction, PortfolioSnapshot, PositionSnapshot, Signal
from core.regime_backtest_engine.strategy import BarView, HistoricalDataView, MarketDataStore
from core.regime_backtest_engine.synthetic_bars import Bar5M


def _make_store(rows: list[dict[str, float | str]], spread_pips: float = 1.0) -> MarketDataStore:
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    pip_size = 0.01
    half = spread_pips * pip_size / 2.0
    df["mid_open"] = df["open"]
    df["mid_high"] = df["high"]
    df["mid_low"] = df["low"]
    df["mid_close"] = df["close"]
    df["bid_open"] = df["open"] - half
    df["bid_high"] = df["high"] - half
    df["bid_low"] = df["low"] - half
    df["bid_close"] = df["close"] - half
    df["ask_open"] = df["open"] + half
    df["ask_high"] = df["high"] + half
    df["ask_low"] = df["low"] + half
    df["ask_close"] = df["close"] + half
    df["spread_pips"] = spread_pips
    cols = [
        "timestamp",
        "bid_open",
        "bid_high",
        "bid_low",
        "bid_close",
        "ask_open",
        "ask_high",
        "ask_low",
        "ask_close",
        "mid_open",
        "mid_high",
        "mid_low",
        "mid_close",
        "spread_pips",
    ]
    return MarketDataStore({col: df[col].to_numpy() for col in cols})


def _portfolio(*, open_positions: tuple[PositionSnapshot, ...] = ()) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        balance=100000.0,
        equity=100000.0,
        unrealized_pnl=0.0,
        margin_used=0.0,
        available_margin=100000.0,
        open_positions=open_positions,
        closed_trade_count=0,
    )


def _row(timestamp: str, open_px: float, high_px: float, low_px: float, close_px: float) -> dict[str, float | str]:
    return {
        "timestamp": timestamp,
        "open": open_px,
        "high": high_px,
        "low": low_px,
        "close": close_px,
    }


def _series_rows(start: str, count: int, *, price: float = 150.0, drift: float = 0.0) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    ts = pd.Timestamp(start)
    current = price
    for idx in range(count):
        open_px = current
        close_px = current + drift
        high_px = max(open_px, close_px) + 0.01
        low_px = min(open_px, close_px) - 0.01
        rows.append(_row((ts + pd.Timedelta(minutes=idx)).isoformat(), open_px, high_px, low_px, close_px))
        current = close_px
    return rows


def _store_with_warmup(entry_row: dict[str, float | str], *, warmup_bars: int = 250) -> tuple[MarketDataStore, int]:
    entry_ts = pd.Timestamp(str(entry_row["timestamp"]))
    start_ts = entry_ts - pd.Timedelta(minutes=warmup_bars)
    rows = _series_rows(start_ts.isoformat(), warmup_bars, price=150.0, drift=0.0)
    rows.append(entry_row)
    return _make_store(rows), len(rows) - 1


def _dummy_5m_bars(count: int = 50) -> list[Bar5M]:
    out: list[Bar5M] = []
    ts = pd.Timestamp("2026-01-01T00:01:00Z")
    for idx in range(count):
        price = 150.0 + idx * 0.01
        out.append(
            Bar5M(
                timestamp=(ts + pd.Timedelta(minutes=5 * idx)).to_pydatetime(),
                open=price,
                high=price + 0.02,
                low=price - 0.02,
                close=price + 0.01,
                volume=None,
                bar_index_start=idx * 5,
                bar_index_end=idx * 5 + 4,
                complete=True,
            )
        )
    return out


def _snapshot(
    *,
    mid: float = 150.00,
    upper: float = 150.20,
    lower: float = 149.80,
    width: float = 0.002,
    ema: float = 149.90,
) -> _IndicatorSnapshot:
    return _IndicatorSnapshot(
        five_minute_timestamp=pd.Timestamp("2026-01-02T06:56:00Z").to_pydatetime(),
        bb_mid=mid,
        bb_upper=upper,
        bb_lower=lower,
        bb_width=width,
        ema_5m_50=ema,
    )


def _freeze_ready_state(
    monkeypatch: pytest.MonkeyPatch,
    strategy: BBReversionStrategy,
    *,
    indicator: _IndicatorSnapshot,
    pdh: float | None = 151.50,
    pdl: float | None = 148.50,
    daily_trades: int = 0,
    cooldown_until: int | None = None,
) -> None:
    strategy._completed_5m = _dummy_5m_bars()
    strategy._indicator_snapshot = indicator
    strategy._daily_trade_count = daily_trades
    strategy._cooldown_until_bar_index = cooldown_until
    monkeypatch.setattr(strategy, "_advance_state", lambda current_bar: None)
    monkeypatch.setattr(strategy, "current_indicator_snapshot", lambda: indicator)
    monkeypatch.setattr(strategy, "current_pdh_pdl", lambda: (pdh, pdl))


def _position(
    *,
    trade_id: int = 1,
    direction: str = "long",
    entry_price: float = 150.00,
    entry_bar: int = 0,
    size: int = 200000,
    stop_loss: float = 149.80,
    take_profit: float | None = None,
) -> PositionSnapshot:
    return PositionSnapshot(
        trade_id=trade_id,
        family="bb_reversion",
        direction=direction,
        entry_price=entry_price,
        entry_bar=entry_bar,
        size=size,
        margin_held=1000.0,
        stop_loss=stop_loss,
        take_profit=take_profit,
        unrealized_pnl=0.0,
    )


def test_adapter_initializes_without_error() -> None:
    strategy = BBReversionStrategy()

    assert strategy.family_name == "bb_reversion"


def test_evaluate_returns_signal_or_none_never_crashes() -> None:
    store = _make_store([_row("2026-01-02T07:00:00Z", 150.0, 150.1, 149.9, 150.0)])
    strategy = BBReversionStrategy()
    result = strategy.evaluate(BarView(store, 0), HistoricalDataView(store, 0), _portfolio())

    assert result is None or isinstance(result, Signal)


def test_session_gate_rejects_bars_outside_london_ny(monkeypatch: pytest.MonkeyPatch) -> None:
    store, idx = _store_with_warmup(_row("2026-01-02T06:59:00Z", 149.90, 149.95, 149.75, 149.85))
    current_bar = BarView(store, idx)
    strategy = BBReversionStrategy()
    _freeze_ready_state(monkeypatch, strategy, indicator=_snapshot())

    assert strategy.evaluate(current_bar, HistoricalDataView(store, idx), _portfolio()) is None
    assert bb_reversion_session_open_utc(pd.Timestamp(current_bar.timestamp)) is False


def test_session_gate_accepts_bars_inside_london_ny(monkeypatch: pytest.MonkeyPatch) -> None:
    store, idx = _store_with_warmup(_row("2026-01-02T07:30:00Z", 149.90, 149.95, 149.75, 149.85))
    current_bar = BarView(store, idx)
    strategy = BBReversionStrategy()
    _freeze_ready_state(monkeypatch, strategy, indicator=_snapshot())

    signal = strategy.evaluate(current_bar, HistoricalDataView(store, idx), _portfolio())

    assert bb_reversion_session_open_utc(pd.Timestamp(current_bar.timestamp)) is True
    assert signal is not None
    assert signal.direction == "long"


def test_bollinger_band_computation_known_price_sequence_matches_manual() -> None:
    closes = [float(value) for value in range(1, 21)]
    bb_mid, bb_upper, bb_lower = compute_bollinger_bands(closes, period=20, num_std=2.0)

    manual_mid = sum(closes) / len(closes)
    variance = sum((value - manual_mid) ** 2 for value in closes) / len(closes)
    std_dev = variance ** 0.5

    assert bb_mid == pytest.approx(manual_mid)
    assert bb_upper == pytest.approx(manual_mid + 2.0 * std_dev)
    assert bb_lower == pytest.approx(manual_mid - 2.0 * std_dev)


def test_bb_width_filter_width_00005_no_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    store, idx = _store_with_warmup(_row("2026-01-02T07:30:00Z", 149.90, 149.95, 149.75, 149.85))
    strategy = BBReversionStrategy()
    _freeze_ready_state(monkeypatch, strategy, indicator=_snapshot(width=0.0005))

    assert strategy.evaluate(BarView(store, idx), HistoricalDataView(store, idx), _portfolio()) is None


def test_bb_width_filter_width_0002_signal_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    store, idx = _store_with_warmup(_row("2026-01-02T07:30:00Z", 149.90, 149.95, 149.75, 149.85))
    strategy = BBReversionStrategy()
    _freeze_ready_state(monkeypatch, strategy, indicator=_snapshot(width=0.002))

    signal = strategy.evaluate(BarView(store, idx), HistoricalDataView(store, idx), _portfolio())

    assert signal is not None
    assert signal.direction == "long"


def test_lower_band_touch_and_close_above_fires_buy_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    store, idx = _store_with_warmup(_row("2026-01-02T07:30:00Z", 149.90, 149.95, 149.79, 149.85))
    strategy = BBReversionStrategy()
    _freeze_ready_state(monkeypatch, strategy, indicator=_snapshot(lower=149.80))

    signal = strategy.evaluate(BarView(store, idx), HistoricalDataView(store, idx), _portfolio())

    assert signal is not None
    assert signal.direction == "long"


def test_lower_band_touch_and_close_below_no_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    store, idx = _store_with_warmup(_row("2026-01-02T07:30:00Z", 149.90, 149.95, 149.79, 149.78))
    strategy = BBReversionStrategy()
    _freeze_ready_state(monkeypatch, strategy, indicator=_snapshot(lower=149.80))

    assert strategy.evaluate(BarView(store, idx), HistoricalDataView(store, idx), _portfolio()) is None


def test_trend_filter_price_20_pips_below_ema50_no_long_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    store, idx = _store_with_warmup(_row("2026-01-02T07:30:00Z", 149.90, 149.95, 149.75, 149.80))
    strategy = BBReversionStrategy()
    _freeze_ready_state(monkeypatch, strategy, indicator=_snapshot(ema=150.00))

    assert strategy.evaluate(BarView(store, idx), HistoricalDataView(store, idx), _portfolio()) is None


def test_trend_filter_price_5_pips_below_ema50_long_signal_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    store, idx = _store_with_warmup(_row("2026-01-02T07:30:00Z", 149.90, 149.96, 149.79, 149.95))
    strategy = BBReversionStrategy()
    _freeze_ready_state(monkeypatch, strategy, indicator=_snapshot(ema=150.00))

    signal = strategy.evaluate(BarView(store, idx), HistoricalDataView(store, idx), _portfolio())

    assert signal is not None
    assert signal.direction == "long"


def test_pdh_pdl_filter_long_within_15_pips_of_pdh_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    store, idx = _store_with_warmup(_row("2026-01-02T07:30:00Z", 149.90, 149.96, 149.79, 149.95))
    strategy = BBReversionStrategy()
    _freeze_ready_state(monkeypatch, strategy, indicator=_snapshot(ema=149.90), pdh=150.00)

    assert strategy.evaluate(BarView(store, idx), HistoricalDataView(store, idx), _portfolio()) is None


def test_cooldown_no_signal_within_10_bars_of_sl_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    store, idx = _store_with_warmup(_row("2026-01-02T07:30:00Z", 149.90, 149.96, 149.79, 149.95))
    strategy = BBReversionStrategy()
    _freeze_ready_state(monkeypatch, strategy, indicator=_snapshot(ema=149.90), cooldown_until=idx + 10)

    assert strategy.evaluate(BarView(store, idx), HistoricalDataView(store, idx), _portfolio()) is None


def test_max_daily_trades_ninth_trade_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    store, idx = _store_with_warmup(_row("2026-01-02T07:30:00Z", 149.90, 149.96, 149.79, 149.95))
    strategy = BBReversionStrategy()
    _freeze_ready_state(monkeypatch, strategy, indicator=_snapshot(ema=149.90), daily_trades=8)

    assert strategy.evaluate(BarView(store, idx), HistoricalDataView(store, idx), _portfolio()) is None


def test_tp1_at_bb_mid_computed_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        _row("2026-01-02T07:30:00Z", 150.00, 150.02, 149.98, 150.01),
        _row("2026-01-02T07:31:00Z", 150.01, 150.18, 150.00, 150.16),
    ]
    store = _make_store(rows)
    strategy = BBReversionStrategy()
    monkeypatch.setattr(strategy, "current_indicator_snapshot", lambda: _snapshot(mid=150.15, upper=150.35, lower=149.95, ema=150.00))
    monkeypatch.setattr(strategy, "_advance_state", lambda current_bar: None)

    signal = Signal(family="bb_reversion", direction="long", stop_loss=149.80, take_profit=None, size=200000)
    position = _position(entry_bar=0, direction="long", entry_price=150.00)

    strategy.on_position_opened(position, signal, BarView(store, 0))
    plan = strategy._trade_plans[position.trade_id]
    action = strategy.get_exit_conditions(position, BarView(store, 1), HistoricalDataView(store, 1))

    assert plan.tp1_price == pytest.approx(150.15)
    assert isinstance(action, ExitAction)
    assert action.reason == "tp1"
    assert action.exit_type == "partial"
    assert action.close_fraction == pytest.approx(0.5)
    assert action.price == pytest.approx(150.15)


def test_tp2_frozen_at_entry_time_opposite_band_value(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        _row("2026-01-02T07:30:00Z", 150.00, 150.02, 149.98, 150.01),
        _row("2026-01-02T07:31:00Z", 150.01, 150.37, 150.00, 150.20),
    ]
    store = _make_store(rows)
    strategy = BBReversionStrategy()
    monkeypatch.setattr(strategy, "_advance_state", lambda current_bar: None)
    monkeypatch.setattr(strategy, "current_indicator_snapshot", lambda: _snapshot(mid=150.15, upper=150.35, lower=149.95, ema=150.00))

    signal = Signal(family="bb_reversion", direction="long", stop_loss=149.80, take_profit=None, size=200000)
    position = _position(entry_bar=0, direction="long", entry_price=150.00)
    strategy.on_position_opened(position, signal, BarView(store, 0))
    strategy._trade_plans[position.trade_id].tp1_done = True

    monkeypatch.setattr(strategy, "current_indicator_snapshot", lambda: _snapshot(mid=150.25, upper=150.55, lower=149.85, ema=150.10))
    action = strategy.get_exit_conditions(position, BarView(store, 1), HistoricalDataView(store, 1))

    assert strategy._trade_plans[position.trade_id].tp2_price == pytest.approx(150.35)
    assert isinstance(action, ExitAction)
    assert action.reason == "tp2"
    assert action.price == pytest.approx(150.35)


def test_max_hold_at_120_bars(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = _series_rows("2026-01-02T07:30:00Z", 121, price=150.0, drift=0.001)
    store = _make_store(rows)
    strategy = BBReversionStrategy()
    monkeypatch.setattr(strategy, "_advance_state", lambda current_bar: None)
    monkeypatch.setattr(strategy, "current_indicator_snapshot", lambda: _snapshot())

    signal = Signal(family="bb_reversion", direction="long", stop_loss=149.80, take_profit=None, size=200000)
    position = _position(entry_bar=0, direction="long", entry_price=150.00)
    strategy.on_position_opened(position, signal, BarView(store, 0))

    action = strategy.get_exit_conditions(position, BarView(store, 120), HistoricalDataView(store, 120))

    assert isinstance(action, ExitAction)
    assert action.reason == "max_hold"
    assert action.exit_type == "full"


def test_no_future_data_access() -> None:
    base_rows = _series_rows("2026-01-02T07:00:00Z", 260, price=150.0, drift=0.001)
    alt_rows = list(base_rows)
    alt_rows[-1] = _row("2026-01-02T11:19:00Z", 160.0, 165.0, 140.0, 150.0)
    store_a = _make_store(base_rows)
    store_b = _make_store(alt_rows)
    strategy_a = BBReversionStrategy()
    strategy_b = BBReversionStrategy()

    out_a = None
    out_b = None
    for idx in range(255):
        out_a = strategy_a.evaluate(BarView(store_a, idx), HistoricalDataView(store_a, idx), _portfolio())
        out_b = strategy_b.evaluate(BarView(store_b, idx), HistoricalDataView(store_b, idx), _portfolio())

    if out_a is None or out_b is None:
        assert out_a is None and out_b is None
    else:
        assert out_a.direction == out_b.direction
        assert out_a.stop_loss == pytest.approx(out_b.stop_loss)


def test_warmup_no_signals_during_first_250_bars(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = _series_rows("2026-01-02T07:00:00Z", 250, price=150.0, drift=0.001)
    store = _make_store(rows)
    strategy = BBReversionStrategy()
    _freeze_ready_state(monkeypatch, strategy, indicator=_snapshot())

    assert strategy.evaluate(BarView(store, 249), HistoricalDataView(store, 249), _portfolio()) is None


def test_max_concurrent_one_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    store, idx = _store_with_warmup(_row("2026-01-02T07:30:00Z", 149.90, 149.96, 149.79, 149.95))
    strategy = BBReversionStrategy()
    _freeze_ready_state(monkeypatch, strategy, indicator=_snapshot())

    open_position = _position()

    assert strategy.evaluate(
        BarView(store, idx),
        HistoricalDataView(store, idx),
        _portfolio(open_positions=(open_position,)),
    ) is None
