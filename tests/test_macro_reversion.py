from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.regime_backtest_engine.macro_reversion import (
    MacroReversionStrategy,
    _LevelSnapshot,
    _TradePlan,
    _session_day_utc,
)
from core.regime_backtest_engine.models import PortfolioSnapshot, PositionSnapshot, Signal
from core.regime_backtest_engine.strategy import BarView, HistoricalDataView, MarketDataStore
from core.regime_backtest_engine.synthetic_bars import BarDaily


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


def _level_snapshot_for_bar(ts: pd.Timestamp, *, levels: tuple[float, ...], pdh: float | None = 151.20, pdl: float | None = 149.20, version: int = 1) -> _LevelSnapshot:
    return _LevelSnapshot(
        trading_day=_session_day_utc(ts, 22),
        version=version,
        recompute_bar_index=0,
        levels=tuple(levels),
        pdh=pdh,
        pdl=pdl,
        pwh=None,
        pwl=None,
        pmh=None,
        pml=None,
        round_levels=(),
        has_cluster=len(levels) >= 2,
    )


def _make_daily_bar(trading_day: date, price: float) -> BarDaily:
    return BarDaily(
        trading_day=trading_day,
        timestamp=pd.Timestamp(trading_day).tz_localize("UTC").to_pydatetime(),
        open=price,
        high=price + 0.05,
        low=price - 0.05,
        close=price,
        volume=None,
        bar_index_start=0,
        bar_index_end=0,
        bar_count=1,
    )


def _freeze_ready_state(
    monkeypatch: pytest.MonkeyPatch,
    strategy: MacroReversionStrategy,
    *,
    current_bar: BarView,
    range_position: float,
    levels: tuple[float, ...],
    pdh: float | None = 151.20,
    pdl: float | None = 149.20,
    version: int = 1,
) -> None:
    strategy._level_snapshot = _level_snapshot_for_bar(
        pd.Timestamp(current_bar.timestamp),
        levels=levels,
        pdh=pdh,
        pdl=pdl,
        version=version,
    )
    strategy._last_recompute_day = strategy._level_snapshot.trading_day
    monkeypatch.setattr(strategy, "_advance_state", lambda bar: None)
    monkeypatch.setattr(strategy, "compute_range_position", lambda current_price: range_position)


def test_adapter_initializes_without_error() -> None:
    strategy = MacroReversionStrategy()

    assert strategy.family_name == "macro_reversion"


def test_evaluate_returns_signal_or_none_never_crashes() -> None:
    store = _make_store([_row("2026-01-02T07:00:00Z", 150.0, 150.1, 149.9, 150.0)])
    strategy = MacroReversionStrategy()
    result = strategy.evaluate(BarView(store, 0), HistoricalDataView(store, 0), _portfolio())

    assert result is None or isinstance(result, Signal)


def test_bias_short_when_range_position_085() -> None:
    strategy = MacroReversionStrategy()

    assert strategy.bias_from_range_position(0.85) == "short"


def test_bias_long_when_range_position_015() -> None:
    strategy = MacroReversionStrategy()

    assert strategy.bias_from_range_position(0.15) == "long"


def test_bias_neutral_when_range_position_050_and_no_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _make_store([_row("2026-01-02T07:00:00Z", 150.0, 151.0, 149.9, 149.8)])
    current_bar = BarView(store, 0)
    history = HistoricalDataView(store, 0)
    strategy = MacroReversionStrategy()
    _freeze_ready_state(monkeypatch, strategy, current_bar=current_bar, range_position=0.50, levels=(151.0, 151.18))

    assert strategy.bias_from_range_position(0.50) == "neutral"
    assert strategy.evaluate(current_bar, history, _portfolio()) is None


def test_level_proximity_signal_only_when_within_10_pips(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _make_store([_row("2026-01-02T07:00:00Z", 150.90, 150.96, 150.70, 150.80)])
    current_bar = BarView(store, 0)
    history = HistoricalDataView(store, 0)
    strategy = MacroReversionStrategy()
    _freeze_ready_state(monkeypatch, strategy, current_bar=current_bar, range_position=0.85, levels=(151.00, 151.18))

    signal = strategy.evaluate(current_bar, history, _portfolio())

    assert signal is not None
    assert signal.direction == "short"


def test_level_proximity_bar_15_pips_from_nearest_level_no_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _make_store([_row("2026-01-02T07:00:00Z", 150.90, 150.85, 150.70, 150.80)])
    current_bar = BarView(store, 0)
    history = HistoricalDataView(store, 0)
    strategy = MacroReversionStrategy()
    _freeze_ready_state(monkeypatch, strategy, current_bar=current_bar, range_position=0.85, levels=(151.00, 151.18))

    assert strategy.evaluate(current_bar, history, _portfolio()) is None


def test_level_cluster_required_single_isolated_level_no_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _make_store([_row("2026-01-02T07:00:00Z", 150.90, 150.96, 150.70, 150.80)])
    current_bar = BarView(store, 0)
    history = HistoricalDataView(store, 0)
    strategy = MacroReversionStrategy()
    _freeze_ready_state(monkeypatch, strategy, current_bar=current_bar, range_position=0.85, levels=(151.00,))

    assert strategy.evaluate(current_bar, history, _portfolio()) is None


def test_level_cluster_present_with_levels_18_pips_apart_signal_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _make_store([_row("2026-01-02T07:00:00Z", 150.90, 150.96, 150.70, 150.80)])
    current_bar = BarView(store, 0)
    history = HistoricalDataView(store, 0)
    strategy = MacroReversionStrategy()
    _freeze_ready_state(monkeypatch, strategy, current_bar=current_bar, range_position=0.85, levels=(151.00, 151.18))

    assert strategy.evaluate(current_bar, history, _portfolio()) is not None


def test_bar_rejection_wrong_candle_type_for_sell_no_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _make_store([_row("2026-01-02T07:00:00Z", 150.80, 150.96, 150.70, 150.90)])
    current_bar = BarView(store, 0)
    history = HistoricalDataView(store, 0)
    strategy = MacroReversionStrategy()
    _freeze_ready_state(monkeypatch, strategy, current_bar=current_bar, range_position=0.85, levels=(151.00, 151.18))

    assert strategy.evaluate(current_bar, history, _portfolio()) is None


def test_bar_rejection_right_candle_type_for_sell_signal_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _make_store([_row("2026-01-02T07:00:00Z", 150.90, 150.96, 150.70, 150.80)])
    current_bar = BarView(store, 0)
    history = HistoricalDataView(store, 0)
    strategy = MacroReversionStrategy()
    _freeze_ready_state(monkeypatch, strategy, current_bar=current_bar, range_position=0.85, levels=(151.00, 151.18))

    assert strategy.evaluate(current_bar, history, _portfolio()) is not None


def test_warmup_no_signals_during_first_252_daily_bars_of_data(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _make_store([_row("2026-01-02T07:00:00Z", 150.90, 150.96, 150.70, 150.80)])
    current_bar = BarView(store, 0)
    history = HistoricalDataView(store, 0)
    strategy = MacroReversionStrategy()
    strategy._completed_daily_bars = [_make_daily_bar(date(2025, 1, 1) + pd.Timedelta(days=i), 150.0 + i * 0.01) for i in range(251)]  # type: ignore[list-item]
    strategy._level_snapshot = _level_snapshot_for_bar(pd.Timestamp(current_bar.timestamp), levels=(151.00, 151.18))
    strategy._last_recompute_day = strategy._level_snapshot.trading_day
    monkeypatch.setattr(strategy, "_advance_state", lambda bar: None)

    assert strategy.evaluate(current_bar, history, _portfolio()) is None


def test_stop_loss_at_30_pips_from_fill_computed_correctly() -> None:
    strategy = MacroReversionStrategy()

    assert strategy.stop_loss_price(150.00, "long") == pytest.approx(149.70)
    assert strategy.stop_loss_price(150.00, "short") == pytest.approx(150.30)


def test_tp1_at_15_pips_and_50pct_close_computed_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        _row("2026-01-02T07:00:00Z", 150.00, 150.02, 149.98, 150.01),
        _row("2026-01-02T07:01:00Z", 150.01, 150.20, 150.00, 150.18),
    ]
    store = _make_store(rows)
    strategy = MacroReversionStrategy()
    strategy._level_snapshot = _level_snapshot_for_bar(pd.Timestamp(rows[0]["timestamp"]), levels=(151.00, 151.18), pdh=150.50, pdl=149.40)
    signal = Signal(
        family="macro_reversion",
        direction="long",
        stop_loss=149.70,
        take_profit=150.15,
        size=200000,
        metadata={"tp1_pips": 15.0, "tp2_pips": 30.0},
    )
    position = PositionSnapshot(
        trade_id=1,
        family="macro_reversion",
        direction="long",
        entry_price=150.00,
        entry_bar=0,
        size=200000,
        margin_held=0.0,
        stop_loss=149.70,
        take_profit=150.15,
        unrealized_pnl=0.0,
    )
    strategy.on_position_opened(position, signal, BarView(store, 0))

    action = strategy.get_exit_conditions(position, BarView(store, 1), HistoricalDataView(store, 1))

    assert strategy._trade_plans[1].tp1_price == pytest.approx(150.15)
    assert action is not None
    assert action.reason == "tp1_partial"
    assert action.close_fraction == pytest.approx(0.5)
    assert action.price == pytest.approx(150.15)


def test_tp2_at_30_pips_and_50pct_of_remaining_computed_correctly() -> None:
    rows = [
        _row("2026-01-02T07:00:00Z", 150.00, 150.02, 149.98, 150.01),
        _row("2026-01-02T07:01:00Z", 150.01, 150.35, 150.00, 150.32),
    ]
    store = _make_store(rows)
    strategy = MacroReversionStrategy()
    strategy._level_snapshot = _level_snapshot_for_bar(pd.Timestamp(rows[0]["timestamp"]), levels=(151.00, 151.18), pdh=150.70, pdl=149.40)
    signal = Signal(
        family="macro_reversion",
        direction="long",
        stop_loss=149.70,
        take_profit=150.15,
        size=200000,
        metadata={"tp1_pips": 15.0, "tp2_pips": 30.0},
    )
    position = PositionSnapshot(
        trade_id=1,
        family="macro_reversion",
        direction="long",
        entry_price=150.00,
        entry_bar=0,
        size=100000,
        margin_held=0.0,
        stop_loss=149.70,
        take_profit=150.30,
        unrealized_pnl=0.0,
    )
    strategy.on_position_opened(position, signal, BarView(store, 0))
    strategy._trade_plans[1].tp1_done = True

    action = strategy.get_exit_conditions(position, BarView(store, 1), HistoricalDataView(store, 1))

    assert strategy._trade_plans[1].tp2_price == pytest.approx(150.30)
    assert action is not None
    assert action.reason == "tp2_partial"
    assert action.close_fraction == pytest.approx(0.5)
    assert action.price == pytest.approx(150.30)


def test_runner_trail_updates_only_when_daily_levels_recompute() -> None:
    rows = [_row("2026-01-02T12:30:00Z", 150.00, 150.02, 149.98, 150.00)]
    store = _make_store(rows)
    current_bar = BarView(store, 0)
    history = HistoricalDataView(store, 0)
    strategy = MacroReversionStrategy()
    ts = pd.Timestamp(current_bar.timestamp)
    strategy._level_snapshot = _level_snapshot_for_bar(ts, levels=(149.20, 149.35), pdh=151.20, pdl=149.20, version=1)
    strategy._trade_plans[1] = _TradePlan(
        direction="short",
        session_end_ts=ts.normalize() + pd.Timedelta(hours=17),
        max_hold_until_bar=999,
        tp1_price=149.85,
        tp2_price=149.70,
        tp1_done=True,
        tp2_done=True,
        runner_target_price=149.20,
        last_trail_version=1,
    )
    position = PositionSnapshot(
        trade_id=1,
        family="macro_reversion",
        direction="short",
        entry_price=150.00,
        entry_bar=0,
        size=50000,
        margin_held=0.0,
        stop_loss=150.30,
        take_profit=149.20,
        unrealized_pnl=0.0,
    )

    assert strategy.get_exit_conditions(position, current_bar, history) is None

    strategy._level_snapshot = _level_snapshot_for_bar(ts, levels=(149.05, 149.20), pdh=151.20, pdl=149.05, version=2)
    update = strategy.get_exit_conditions(position, current_bar, history)

    assert update is not None
    assert update.reason == "runner_trail_update"
    assert update.new_take_profit == pytest.approx(149.05)
    assert strategy.get_exit_conditions(position, current_bar, history) is None


def test_max_concurrent_one_second_signal_rejected_while_position_open(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _make_store([_row("2026-01-02T07:00:00Z", 150.90, 150.96, 150.70, 150.80)])
    current_bar = BarView(store, 0)
    history = HistoricalDataView(store, 0)
    strategy = MacroReversionStrategy()
    _freeze_ready_state(monkeypatch, strategy, current_bar=current_bar, range_position=0.85, levels=(151.00, 151.18))
    open_position = PositionSnapshot(
        trade_id=1,
        family="macro_reversion",
        direction="short",
        entry_price=150.80,
        entry_bar=0,
        size=200000,
        margin_held=0.0,
        stop_loss=151.10,
        take_profit=150.65,
        unrealized_pnl=0.0,
    )

    assert strategy.evaluate(current_bar, history, _portfolio(open_positions=(open_position,))) is None


def test_cooldown_no_signal_within_50_bars_of_any_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _make_store([_row("2026-01-02T07:40:00Z", 150.90, 150.96, 150.70, 150.80)])
    current_bar = BarView(store, 0)
    history = HistoricalDataView(store, 0)
    strategy = MacroReversionStrategy()
    strategy._cooldown_until_bar_index = 50
    _freeze_ready_state(monkeypatch, strategy, current_bar=current_bar, range_position=0.85, levels=(151.00, 151.18))

    assert strategy.evaluate(current_bar, history, _portfolio()) is None


def test_session_gate_rejects_bars_outside_london_or_ny(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _make_store([_row("2026-01-02T06:59:00Z", 150.90, 150.96, 150.70, 150.80)])
    current_bar = BarView(store, 0)
    history = HistoricalDataView(store, 0)
    strategy = MacroReversionStrategy()
    _freeze_ready_state(monkeypatch, strategy, current_bar=current_bar, range_position=0.85, levels=(151.00, 151.18))

    assert strategy.evaluate(current_bar, history, _portfolio()) is None


def test_no_future_data_evaluate_at_bar_n_only_sees_bars_0_to_n() -> None:
    rows: list[dict[str, float | str]] = []
    start = pd.Timestamp("2025-01-01T07:00:00Z")
    price = 149.00
    for i in range(252):
        px = price + i * 0.01
        rows.append(_row((start + pd.Timedelta(days=i)).isoformat(), px, px + 0.05, px - 0.05, px + 0.02))
    signal_day = start + pd.Timedelta(days=252)
    rows.append(_row(signal_day.isoformat(), 151.00, 151.02, 150.95, 151.01))
    rows.append(_row((signal_day + pd.Timedelta(minutes=1)).isoformat(), 151.05, 151.10, 150.95, 150.98))
    store = _make_store(rows)
    strategy = MacroReversionStrategy()
    seen_signal_at_bar_n = None
    seen_signal_at_bar_n_plus_1 = None

    for idx in range(len(rows)):
        current_bar = BarView(store, idx)
        history = HistoricalDataView(store, idx)
        signal = strategy.evaluate(current_bar, history, _portfolio())
        if idx == len(rows) - 2:
            seen_signal_at_bar_n = signal
        if idx == len(rows) - 1:
            seen_signal_at_bar_n_plus_1 = signal

    assert seen_signal_at_bar_n is None
    assert seen_signal_at_bar_n_plus_1 is None or isinstance(seen_signal_at_bar_n_plus_1, Signal)
