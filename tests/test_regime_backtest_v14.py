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
    V14TokyoStrategy,
    V14TokyoStrategyConfig,
    prepare_v14_augmented_data,
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


def _prepare_augmented_subset(tmp_path: Path, rows: int = 3000):
    src = Path('/Users/codygnon/Documents/usdjpy_assistant/research_out/USDJPY_M1_OANDA_500k.csv')
    raw_subset = tmp_path / 'tokyo_subset.csv'
    pd.read_csv(src, nrows=rows).to_csv(raw_subset, index=False)
    augmented = tmp_path / 'tokyo_subset_augmented.csv'
    prepare_v14_augmented_data(
        raw_subset,
        '/Users/codygnon/Documents/usdjpy_assistant/research_out/tokyo_optimized_v14_config.json',
        augmented,
    )
    cfg = RunConfig(
        data_path=augmented,
        output_dir=tmp_path / 'out',
        mode='standalone',
        active_families=('v14',),
        instrument=InstrumentSpec(symbol='USDJPY'),
        spread=SpreadConfig(spread_source='fixed', fixed=FixedSpreadConfig(spread_pips=2.0)),
        slippage=SlippageConfig(fixed_slippage_pips=0.1),
        bar_log_format='csv',
    )
    loaded = load_market_data(cfg)
    return augmented, loaded


def test_v14_adapter_initializes_and_evaluates_without_error(tmp_path: Path) -> None:
    _augmented, loaded = _prepare_augmented_subset(tmp_path)
    strategy = V14TokyoStrategy(
        V14TokyoStrategyConfig.from_json(
            '/Users/codygnon/Documents/usdjpy_assistant/research_out/tokyo_optimized_v14_config.json'
        )
    )
    frame = loaded.frame
    idx = int(frame.index[(frame['in_tokyo_session'] == 1) & (frame['allowed_trading_day'] == 1)][50])
    signal = strategy.evaluate(BarView(loaded.store, idx), HistoricalDataView(loaded.store, idx), _portfolio())
    assert signal is None or isinstance(signal, Signal)


def test_v14_session_gating_rejects_outside_session(tmp_path: Path) -> None:
    _augmented, loaded = _prepare_augmented_subset(tmp_path)
    strategy = V14TokyoStrategy(
        V14TokyoStrategyConfig.from_json(
            '/Users/codygnon/Documents/usdjpy_assistant/research_out/tokyo_optimized_v14_config.json'
        )
    )
    frame = loaded.frame
    idx = int(frame.index[frame['in_tokyo_session'] == 0][0])
    signal = strategy.evaluate(BarView(loaded.store, idx), HistoricalDataView(loaded.store, idx), _portfolio())
    assert signal is None


def test_v14_exit_conditions_return_valid_action_or_none(tmp_path: Path) -> None:
    _augmented, loaded = _prepare_augmented_subset(tmp_path)
    strategy = V14TokyoStrategy(
        V14TokyoStrategyConfig.from_json(
            '/Users/codygnon/Documents/usdjpy_assistant/research_out/tokyo_optimized_v14_config.json'
        )
    )
    frame = loaded.frame
    idx = int(frame.index[(frame['in_tokyo_session'] == 1) & (frame['allowed_trading_day'] == 1)][80])
    current_bar = BarView(loaded.store, idx)
    signal = Signal(
        family='v14',
        direction='long',
        stop_loss=float(current_bar.bid_close) - 0.20,
        take_profit=None,
        size=10_000,
        metadata={
            'entry_session_day': str(current_bar.session_day_jst),
            'pivot_P': float(current_bar.pivot_P),
            'pivot_R1': float(current_bar.pivot_R1),
            'pivot_S1': float(current_bar.pivot_S1),
            'atr_m15': float(current_bar.atr_m15),
            'from_zone': True,
            'partial_close_pct': 0.5,
            'partial_tp_min_pips': 6.0,
            'partial_tp_max_pips': 12.0,
            'partial_tp_atr_mult': 0.5,
            'single_tp_atr_mult': 1.0,
            'single_tp_min_pips': 8.0,
            'single_tp_max_pips': 40.0,
            'breakeven_offset_pips': 1.0,
            'trail_activate_pips': 10.0,
            'trail_distance_pips': 8.0,
            'trail_requires_tp1': True,
            'time_decay_minutes': 120,
            'time_decay_profit_cap_pips': 3.0,
            'tp_mode': 'partial',
            'confluence_score': 2,
        },
    )
    position = PositionSnapshot(
        trade_id=1,
        family='v14',
        direction='long',
        entry_price=float(current_bar.ask_close),
        entry_bar=idx,
        size=10_000,
        margin_held=500.0,
        stop_loss=float(current_bar.ask_close) - 0.20,
        take_profit=None,
        unrealized_pnl=0.0,
    )
    strategy.on_position_opened(position, signal, current_bar)
    action = strategy.get_exit_conditions(position, current_bar, HistoricalDataView(loaded.store, idx))
    assert action is None or isinstance(action, ExitAction)


def test_v14_does_not_access_future_history(tmp_path: Path) -> None:
    _augmented, loaded = _prepare_augmented_subset(tmp_path)
    strategy = V14TokyoStrategy(
        V14TokyoStrategyConfig.from_json(
            '/Users/codygnon/Documents/usdjpy_assistant/research_out/tokyo_optimized_v14_config.json'
        )
    )
    last_idx = len(loaded.store) - 1
    signal = strategy.evaluate(BarView(loaded.store, last_idx), HistoricalDataView(loaded.store, last_idx), _portfolio())
    assert signal is None or isinstance(signal, Signal)
