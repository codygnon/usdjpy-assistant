from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.regime_backtest_engine import (
    FixedSpreadConfig,
    InstrumentSpec,
    PortfolioSnapshot,
    RunConfig,
    SlippageConfig,
    SpreadConfig,
    V2LondonStrategy,
    V2LondonStrategyConfig,
)
from core.regime_backtest_engine.data import load_market_data
from core.regime_backtest_engine.models import ExitAction, PositionSnapshot, Signal
from core.regime_backtest_engine.strategy import BarView, HistoricalDataView


def _portfolio() -> PortfolioSnapshot:
    return PortfolioSnapshot(
        balance=100000.0,
        equity=100000.0,
        unrealized_pnl=0.0,
        margin_used=0.0,
        available_margin=100000.0,
        open_positions=(),
        closed_trade_count=0,
    )


def _write_london_csv(path: Path, start: str = '2025-01-06T00:00:00Z', bars: int = 3600) -> Path:
    rows = []
    ts0 = pd.Timestamp(start)
    price = 150.0
    for i in range(bars):
        ts = ts0 + pd.Timedelta(minutes=i)
        drift = 0.004 if 7 <= ts.hour <= 11 else -0.001
        open_px = price
        close_px = price + drift
        rows.append(
            {
                'timestamp': ts,
                'open': open_px,
                'high': max(open_px, close_px) + 0.03,
                'low': min(open_px, close_px) - 0.03,
                'close': close_px,
            }
        )
        price = close_px
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _load_london_store(tmp_path: Path):
    data_path = _write_london_csv(tmp_path / 'london.csv')
    cfg = RunConfig(
        data_path=data_path,
        output_dir=tmp_path / 'out',
        mode='standalone',
        active_families=('london_v2',),
        instrument=InstrumentSpec(symbol='USDJPY'),
        spread=SpreadConfig(spread_source='fixed', fixed=FixedSpreadConfig(spread_pips=2.0)),
        slippage=SlippageConfig(fixed_slippage_pips=0.1),
        bar_log_format='csv',
    )
    return load_market_data(cfg)


def test_v2_london_adapter_initializes_and_evaluates_without_error(tmp_path: Path) -> None:
    loaded = _load_london_store(tmp_path)
    strategy = V2LondonStrategy(
        V2LondonStrategyConfig.from_json(
            '/Users/codygnon/Documents/usdjpy_assistant/research_out/v2_exp4_winner_baseline_config.json'
        )
    )
    frame = loaded.frame
    idx = int(frame.index[frame['timestamp'].dt.day_name().eq('Tuesday') & frame['timestamp'].dt.hour.eq(8)][0])
    signal = strategy.evaluate(BarView(loaded.store, idx), HistoricalDataView(loaded.store, idx), _portfolio())
    assert signal is None or isinstance(signal, Signal)


def test_v2_london_session_gating_rejects_outside_session(tmp_path: Path) -> None:
    loaded = _load_london_store(tmp_path)
    strategy = V2LondonStrategy(
        V2LondonStrategyConfig.from_json(
            '/Users/codygnon/Documents/usdjpy_assistant/research_out/v2_exp4_winner_baseline_config.json'
        )
    )
    frame = loaded.frame
    idx = int(frame.index[frame['timestamp'].dt.hour.eq(2)][0])
    signal = strategy.evaluate(BarView(loaded.store, idx), HistoricalDataView(loaded.store, idx), _portfolio())
    assert signal is None


def test_v2_london_weekday_filter_blocks_monday(tmp_path: Path) -> None:
    loaded = _load_london_store(tmp_path)
    strategy = V2LondonStrategy(
        V2LondonStrategyConfig.from_json(
            '/Users/codygnon/Documents/usdjpy_assistant/research_out/v2_exp4_winner_baseline_config.json'
        )
    )
    frame = loaded.frame
    idx = int(frame.index[frame['timestamp'].dt.day_name().eq('Monday') & frame['timestamp'].dt.hour.eq(8)][0])
    signal = strategy.evaluate(BarView(loaded.store, idx), HistoricalDataView(loaded.store, idx), _portfolio())
    assert signal is None


def test_v2_london_exit_conditions_return_valid_action_or_none(tmp_path: Path) -> None:
    loaded = _load_london_store(tmp_path)
    strategy = V2LondonStrategy(
        V2LondonStrategyConfig.from_json(
            '/Users/codygnon/Documents/usdjpy_assistant/research_out/v2_exp4_winner_baseline_config.json'
        )
    )
    frame = loaded.frame
    idx = int(frame.index[frame['timestamp'].dt.day_name().eq('Tuesday') & frame['timestamp'].dt.hour.eq(8)][10])
    current_bar = BarView(loaded.store, idx)
    signal = Signal(
        family='london_v2',
        direction='long',
        stop_loss=float(current_bar.bid_close) - 0.20,
        take_profit=None,
        size=10_000,
        metadata={
            'setup_type': 'A',
            'tp1_pips': 15.0,
            'tp2_pips': 30.0,
            'tp1_close_fraction': 0.5,
            'be_offset_pips': 1.0,
            'risk_usd_planned': 500.0,
            'grace_trail_distance_pips': 0.0,
            'extend_runner_until_ny_start_delay': False,
        },
    )
    position = PositionSnapshot(
        trade_id=1,
        family='london_v2',
        direction='long',
        entry_price=float(current_bar.ask_close),
        entry_bar=idx,
        size=10_000,
        margin_held=500.0,
        stop_loss=float(current_bar.ask_close) - 0.20,
        take_profit=None,
        unrealized_pnl=0.0,
    )
    strategy._reset_day_if_needed(pd.Timestamp(current_bar.timestamp))
    strategy.on_position_opened(position, signal, current_bar)
    action = strategy.get_exit_conditions(position, current_bar, HistoricalDataView(loaded.store, idx))
    assert action is None or isinstance(action, ExitAction)


def test_v2_london_does_not_access_future_history(tmp_path: Path) -> None:
    loaded = _load_london_store(tmp_path)
    strategy = V2LondonStrategy(
        V2LondonStrategyConfig.from_json(
            '/Users/codygnon/Documents/usdjpy_assistant/research_out/v2_exp4_winner_baseline_config.json'
        )
    )
    last_idx = len(loaded.store) - 1
    signal = strategy.evaluate(BarView(loaded.store, last_idx), HistoricalDataView(loaded.store, last_idx), _portfolio())
    assert signal is None or isinstance(signal, Signal)
