from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from core.regime_backtest_engine import (
    AdmissionConfig,
    BacktestEngine,
    CheatingStrategy,
    DummyStrategy,
    ExitAction,
    FixedSpreadConfig,
    InstrumentSpec,
    PortfolioSnapshot,
    RunConfig,
    Signal,
    SlippageConfig,
    SpreadConfig,
    StrategyValidationError,
)
from core.regime_backtest_engine.data import load_market_data
from core.regime_backtest_engine.strategy import BarView, HistoricalDataView


class OneShotLongStrategy:
    def __init__(self, family_name: str = "oneshot") -> None:
        self.family_name = family_name

    def evaluate(self, current_bar: BarView, history: HistoricalDataView, portfolio: PortfolioSnapshot) -> Signal | None:
        if current_bar.bar_index != 1:
            return None
        if any(pos.family == self.family_name for pos in portfolio.open_positions):
            return None
        return Signal(
            family=self.family_name,
            direction="long",
            stop_loss=float(current_bar.bid_close) - 0.10,
            take_profit=float(current_bar.ask_close) + 0.10,
            size=10_000,
            metadata={"test": "one_shot"},
        )

    def get_exit_conditions(self, position, current_bar: BarView, history: HistoricalDataView) -> ExitAction | None:
        return None


class PartialProfitStrategy:
    def __init__(self, family_name: str = "partial") -> None:
        self.family_name = family_name

    def evaluate(self, current_bar: BarView, history: HistoricalDataView, portfolio: PortfolioSnapshot) -> Signal | None:
        if current_bar.bar_index != 1:
            return None
        if any(pos.family == self.family_name for pos in portfolio.open_positions):
            return None
        return Signal(
            family=self.family_name,
            direction="long",
            stop_loss=float(current_bar.bid_close) - 0.20,
            take_profit=float(current_bar.ask_close) + 0.40,
            size=10_000,
        )

    def get_exit_conditions(self, position, current_bar: BarView, history: HistoricalDataView) -> ExitAction | None:
        if current_bar.bar_index == 2:
            return ExitAction(
                reason="tp1_partial",
                exit_type="partial",
                close_fraction=0.5,
                price=float(current_bar.bid_close),
                new_stop_loss=float(position.entry_price),
                new_take_profit=float(position.entry_price) + 0.40,
            )
        return None


class WrongFamilyStrategy:
    family_name = "expected"

    def evaluate(self, current_bar: BarView, history: HistoricalDataView, portfolio: PortfolioSnapshot) -> Signal | None:
        if current_bar.bar_index != 1:
            return None
        return Signal(
            family="unexpected",
            direction="long",
            stop_loss=float(current_bar.bid_close) - 0.10,
            take_profit=float(current_bar.ask_close) + 0.10,
            size=10_000,
        )

    def get_exit_conditions(self, position, current_bar: BarView, history: HistoricalDataView) -> ExitAction | None:
        return None


class ColumnMutationStrategy:
    family_name = "mutator"

    def evaluate(self, current_bar: BarView, history: HistoricalDataView, portfolio: PortfolioSnapshot) -> Signal | None:
        col = history.column("mid_close")
        col[0] = 999.0
        return None

    def get_exit_conditions(self, position, current_bar: BarView, history: HistoricalDataView) -> ExitAction | None:
        return None


class WindowPeekStrategy:
    family_name = "window_peeker"

    def evaluate(self, current_bar: BarView, history: HistoricalDataView, portfolio: PortfolioSnapshot) -> Signal | None:
        _ = history.window(0, history.max_index + 1)
        return None

    def get_exit_conditions(self, position, current_bar: BarView, history: HistoricalDataView) -> ExitAction | None:
        return None


def _write_mid_csv(path: Path, bars: int = 40) -> Path:
    rows = []
    price = 150.0
    ts = pd.Timestamp("2025-01-01T00:00:00Z")
    for i in range(bars):
        open_px = price
        close_px = price + (0.03 if i % 2 == 0 else -0.01)
        high_px = max(open_px, close_px) + 0.08
        low_px = min(open_px, close_px) - 0.08
        rows.append(
            {
                "timestamp": ts + pd.Timedelta(minutes=i),
                "open": open_px,
                "high": high_px,
                "low": low_px,
                "close": close_px,
            }
        )
        price = close_px
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _config(tmp_path: Path, data_path: Path, *, mode: str, families: tuple[str, ...], bar_log_format: str = "csv") -> RunConfig:
    return RunConfig(
        data_path=data_path,
        output_dir=tmp_path / f"out_{mode}_{'_'.join(families)}",
        mode=mode,
        active_families=families,
        instrument=InstrumentSpec(symbol="USDJPY"),
        spread=SpreadConfig(spread_source="fixed", fixed=FixedSpreadConfig(spread_pips=1.2)),
        slippage=SlippageConfig(fixed_slippage_pips=0.1),
        admission=AdmissionConfig(
            allow_opposing_exposure=False,
            max_total_open_positions=3,
            max_open_positions_per_family={family: 1 for family in families},
            max_total_units=50_000,
            max_units_per_family={family: 10_000 for family in families},
            family_priority=families,
        ),
        bar_log_format=bar_log_format,
    )


def _write_bid_ask_csv(path: Path) -> Path:
    rows = []
    ts = pd.Timestamp("2025-01-01T00:00:00Z")
    mids = [150.00, 150.05, 150.10, 149.95]
    spread = 0.012
    for i, mid in enumerate(mids):
        rows.append(
            {
                "timestamp": ts + pd.Timedelta(minutes=i),
                "bid_open": mid - spread / 2.0,
                "bid_high": mid + 0.02 - spread / 2.0,
                "bid_low": mid - 0.02 - spread / 2.0,
                "bid_close": mid - spread / 2.0,
                "ask_open": mid + spread / 2.0,
                "ask_high": mid + 0.02 + spread / 2.0,
                "ask_low": mid - 0.02 + spread / 2.0,
                "ask_close": mid + spread / 2.0,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_worst_case_csv(path: Path) -> Path:
    rows = [
        {"timestamp": "2025-01-01T00:00:00Z", "open": 150.00, "high": 150.02, "low": 149.98, "close": 150.00},
        {"timestamp": "2025-01-01T00:01:00Z", "open": 150.00, "high": 150.03, "low": 149.99, "close": 150.01},
        {"timestamp": "2025-01-01T00:02:00Z", "open": 150.01, "high": 150.20, "low": 149.80, "close": 150.00},
        {"timestamp": "2025-01-01T00:03:00Z", "open": 150.00, "high": 150.01, "low": 149.99, "close": 150.00},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_future_spike_csv(path: Path) -> Path:
    rows = [
        {"timestamp": "2025-01-01T00:00:00Z", "open": 150.00, "high": 150.01, "low": 149.99, "close": 150.00},
        {"timestamp": "2025-01-01T00:01:00Z", "open": 150.00, "high": 150.02, "low": 149.99, "close": 150.01},
        {"timestamp": "2025-01-01T00:02:00Z", "open": 150.01, "high": 150.03, "low": 150.00, "close": 150.02},
        {"timestamp": "2025-01-01T00:03:00Z", "open": 150.02, "high": 155.00, "low": 149.00, "close": 154.00},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_partial_exit_csv(path: Path) -> Path:
    rows = [
        {"timestamp": "2025-01-01T00:00:00Z", "open": 150.00, "high": 150.01, "low": 149.99, "close": 150.00},
        {"timestamp": "2025-01-01T00:01:00Z", "open": 150.00, "high": 150.03, "low": 149.99, "close": 150.01},
        {"timestamp": "2025-01-01T00:02:00Z", "open": 150.02, "high": 150.12, "low": 150.01, "close": 150.10},
        {"timestamp": "2025-01-01T00:03:00Z", "open": 150.10, "high": 150.12, "low": 149.96, "close": 149.98},
        {"timestamp": "2025-01-01T00:04:00Z", "open": 149.98, "high": 150.00, "low": 149.95, "close": 149.97},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_standalone_matches_integrated_single_family(tmp_path: Path) -> None:
    data_path = _write_mid_csv(tmp_path / "bars.csv")
    dummy = DummyStrategy(family_name="dummy", every_n_bars=5, direction="long")
    engine = BacktestEngine({"dummy": dummy})

    standalone = engine.run(_config(tmp_path, data_path, mode="standalone", families=("dummy",)))
    integrated = engine.run(_config(tmp_path, data_path, mode="integrated", families=("dummy",)))

    assert standalone.summary["net_pnl_usd"] == integrated.summary["net_pnl_usd"]
    assert standalone.summary["trade_count"] == integrated.summary["trade_count"]
    assert standalone.final_portfolio.model_dump() == integrated.final_portfolio.model_dump()


def test_opposing_exposure_is_rejected_and_arbitration_logged(tmp_path: Path) -> None:
    data_path = _write_mid_csv(tmp_path / "bars2.csv")
    engine = BacktestEngine(
        {
            "longer": DummyStrategy(family_name="longer", every_n_bars=4, direction="long"),
            "shorter": DummyStrategy(family_name="shorter", every_n_bars=4, direction="short"),
        }
    )
    result = engine.run(_config(tmp_path, data_path, mode="integrated", families=("longer", "shorter")))

    assert result.arbitration_log_path is not None
    arbitration = pd.read_csv(result.arbitration_log_path)
    assert not arbitration.empty
    assert any("opposing_exposure_block" in str(x) for x in arbitration["rejection_reasons"].tolist())


def test_cheating_strategy_cannot_see_future_bars(tmp_path: Path) -> None:
    data_path = _write_mid_csv(tmp_path / "bars3.csv", bars=10)
    engine = BacktestEngine({"cheater": CheatingStrategy()})
    with pytest.raises(IndexError):
        engine.run(_config(tmp_path, data_path, mode="standalone", families=("cheater",)))


def test_strategy_adapter_rejects_wrong_family_signal(tmp_path: Path) -> None:
    data_path = _write_mid_csv(tmp_path / "bars_wrong_family.csv", bars=10)
    engine = BacktestEngine({"expected": WrongFamilyStrategy()})
    with pytest.raises(StrategyValidationError):
        engine.run(_config(tmp_path, data_path, mode="standalone", families=("expected",)))


def test_history_column_is_read_only(tmp_path: Path) -> None:
    data_path = _write_mid_csv(tmp_path / "bars_mutation.csv", bars=10)
    engine = BacktestEngine({"mutator": ColumnMutationStrategy()})
    with pytest.raises(ValueError):
        engine.run(_config(tmp_path, data_path, mode="standalone", families=("mutator",)))


def test_history_window_helper_cannot_peek_forward(tmp_path: Path) -> None:
    data_path = _write_mid_csv(tmp_path / "bars_window.csv", bars=10)
    engine = BacktestEngine({"window_peeker": WindowPeekStrategy()})
    with pytest.raises(IndexError):
        engine.run(_config(tmp_path, data_path, mode="standalone", families=("window_peeker",)))


def test_start_end_subset_is_reflected_in_summary(tmp_path: Path) -> None:
    data_path = _write_mid_csv(tmp_path / "bars4.csv", bars=20)
    engine = BacktestEngine({"dummy": DummyStrategy(family_name="dummy", every_n_bars=3, direction="long")})
    cfg = _config(tmp_path, data_path, mode="standalone", families=("dummy",))
    cfg = cfg.model_copy(update={"start_index": 5, "end_index": 10})
    result = engine.run(cfg)
    assert result.summary["processed_bar_count"] == 6


def test_mid_only_input_is_normalized_to_synthetic_bid_ask(tmp_path: Path) -> None:
    data_path = _write_mid_csv(tmp_path / "bars5.csv", bars=12)
    engine = BacktestEngine({"dummy": DummyStrategy(family_name="dummy", every_n_bars=4, direction="long")})
    result = engine.run(_config(tmp_path, data_path, mode="standalone", families=("dummy",)))
    assert result.summary["synthetic_bid_ask"] is True


def test_bid_ask_input_passes_through_without_synthesis(tmp_path: Path) -> None:
    data_path = _write_bid_ask_csv(tmp_path / "bars6.csv")
    engine = BacktestEngine({"dummy": DummyStrategy(family_name="dummy", every_n_bars=2, direction="long")})
    cfg = RunConfig(
        data_path=data_path,
        output_dir=tmp_path / "out_bidask",
        mode="standalone",
        active_families=("dummy",),
        instrument=InstrumentSpec(symbol="USDJPY"),
        spread=SpreadConfig(spread_source="from_data"),
        slippage=SlippageConfig(fixed_slippage_pips=0.0),
        admission=AdmissionConfig(
            allow_opposing_exposure=False,
            max_total_open_positions=2,
            max_open_positions_per_family={"dummy": 1},
            max_total_units=50_000,
            max_units_per_family={"dummy": 10_000},
            family_priority=("dummy",),
        ),
        bar_log_format="csv",
    )
    result = engine.run(cfg)
    assert result.summary["synthetic_bid_ask"] is False


def test_margin_rejection_is_logged(tmp_path: Path) -> None:
    data_path = _write_mid_csv(tmp_path / "bars7.csv", bars=12)
    engine = BacktestEngine({"dummy": DummyStrategy(family_name="dummy", every_n_bars=3, direction="long", size_units=1_000_000)})
    cfg = _config(tmp_path, data_path, mode="standalone", families=("dummy",))
    cfg = cfg.model_copy(
        update={
            "initial_balance": 500.0,
            "admission": AdmissionConfig(
                allow_opposing_exposure=False,
                max_total_open_positions=5,
                max_open_positions_per_family={"dummy": 5},
                max_total_units=10_000_000,
                max_units_per_family={"dummy": 10_000_000},
                family_priority=("dummy",),
            ),
        }
    )
    result = engine.run(cfg)
    assert result.arbitration_log_path is not None
    arbitration = pd.read_csv(result.arbitration_log_path)
    assert any("margin_insufficient" in str(x) for x in arbitration["rejection_reasons"].tolist())


def test_simultaneous_stop_and_target_uses_worst_case(tmp_path: Path) -> None:
    data_path = _write_worst_case_csv(tmp_path / "bars8.csv")
    engine = BacktestEngine({"oneshot": OneShotLongStrategy()})
    cfg = _config(tmp_path, data_path, mode="standalone", families=("oneshot",))
    cfg = cfg.model_copy(update={"slippage": SlippageConfig(fixed_slippage_pips=0.0)})
    result = engine.run(cfg)
    trades = pd.read_csv(result.trade_log_path)
    assert len(trades) == 1
    assert trades.loc[0, "exit_reason"] == "worst_case_stop"


def test_partial_exit_closes_fraction_and_keeps_runner(tmp_path: Path) -> None:
    data_path = _write_partial_exit_csv(tmp_path / "bars_partial.csv")
    engine = BacktestEngine({"partial": PartialProfitStrategy()})
    cfg = _config(tmp_path, data_path, mode="standalone", families=("partial",))
    cfg = cfg.model_copy(update={"slippage": SlippageConfig(fixed_slippage_pips=0.0)})
    result = engine.run(cfg)
    trades = pd.read_csv(result.trade_log_path)

    assert len(trades) == 2
    assert trades.loc[0, "event_type"] == "partial"
    assert trades.loc[0, "closed_units"] == 5000
    assert trades.loc[0, "remaining_units"] == 5000
    assert trades.loc[0, "exit_reason"] == "tp1_partial"
    assert trades.loc[1, "event_type"] == "full"
    assert trades.loc[1, "closed_units"] == 5000
    assert trades.loc[1, "remaining_units"] == 0
    assert trades.loc[1, "exit_reason"] == "stop_loss"
    assert result.summary["trade_count"] == 1


def test_rolling_helpers_are_bounded_to_current_bar(tmp_path: Path) -> None:
    data_path = _write_future_spike_csv(tmp_path / "bars_future_spike.csv")
    cfg = _config(tmp_path, data_path, mode="standalone", families=("dummy",))
    loaded = load_market_data(cfg)
    history = HistoricalDataView(loaded.store, 2)

    bounded_values = history.column("mid_close")
    assert len(bounded_values) == 3
    expected_mean = float(bounded_values.mean())
    assert history.rolling_mean("mid_close", window=10, min_periods=1) == expected_mean
    assert history.rolling_max("mid_high", window=10, min_periods=1) < 155.0


def test_load_market_data_preserves_extra_causal_columns(tmp_path: Path) -> None:
    data_path = tmp_path / 'bars_extra.csv'
    pd.DataFrame(
        [
            {
                'timestamp': '2025-01-01T00:00:00Z',
                'open': 150.0,
                'high': 150.1,
                'low': 149.9,
                'close': 150.0,
                'custom_flag': 1,
                'custom_value': 3.14,
            },
            {
                'timestamp': '2025-01-01T00:01:00Z',
                'open': 150.0,
                'high': 150.1,
                'low': 149.9,
                'close': 150.0,
                'custom_flag': 0,
                'custom_value': 2.72,
            },
        ]
    ).to_csv(data_path, index=False)
    loaded = load_market_data(_config(tmp_path, data_path, mode='standalone', families=('dummy',)))
    assert 'custom_flag' in loaded.contract.columns
    assert 'custom_value' in loaded.contract.columns
    assert float(loaded.store.value('custom_value', 1)) == 2.72
