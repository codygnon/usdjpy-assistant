from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from core.regime_backtest_engine import (
    AdmissionConfig,
    BacktestEngine,
    FixedSpreadConfig,
    InstrumentSpec,
    RunConfig,
    SlippageConfig,
    SpreadConfig,
    StrategyFamily,
    V14TokyoStrategy,
    V14TokyoStrategyConfig,
    V2LondonStrategy,
    V2LondonStrategyConfig,
    V44NYStrategy,
    V44StrategyConfig,
)
from core.regime_backtest_engine.models import ExitAction, PortfolioSnapshot, PositionSnapshot, Signal
from core.regime_backtest_engine.strategy import BarView, HistoricalDataView

NY_TZ = ZoneInfo('America/New_York')


@dataclass(frozen=True)
class TimeWindow:
    start_minute: int
    end_minute: int

    def contains(self, minute_of_day: int) -> bool:
        return self.start_minute <= minute_of_day <= self.end_minute


SCHEDULES: dict[str, dict[int, dict[str, tuple[TimeWindow, ...]]]] = {
    'v7': {
        0: {'v44_ny': (TimeWindow(7 * 60 + 55, 11 * 60 + 5),)},
        1: {
            'london_v2': (TimeWindow(2 * 60 + 55, 7 * 60 + 5),),
            'v44_ny': (TimeWindow(7 * 60 + 55, 11 * 60 + 5),),
            'v14': (TimeWindow(11 * 60 + 55, 18 * 60 + 5),),
        },
        2: {
            'london_v2': (TimeWindow(2 * 60 + 55, 7 * 60 + 5),),
            'v44_ny': (TimeWindow(7 * 60 + 55, 11 * 60 + 5),),
            'v14': (TimeWindow(11 * 60 + 55, 18 * 60 + 5),),
        },
        3: {'v44_ny': (TimeWindow(7 * 60 + 55, 11 * 60 + 5),)},
        4: {
            'v44_ny': (TimeWindow(7 * 60 + 55, 11 * 60 + 5),),
            'v14': (TimeWindow(11 * 60 + 55, 18 * 60 + 5),),
        },
    },
    'legacy_phase3': {
        0: {'v44_ny': (TimeWindow(8 * 60 + 55, 12 * 60 + 5),)},
        1: {
            'london_v2': (TimeWindow(3 * 60 + 55, 6 * 60 + 5),),
            'v44_ny': (TimeWindow(8 * 60 + 55, 12 * 60 + 5),),
            'v14': (TimeWindow(11 * 60 + 55, 18 * 60 + 5),),
        },
        2: {
            'london_v2': (TimeWindow(3 * 60 + 55, 6 * 60 + 5),),
            'v44_ny': (TimeWindow(8 * 60 + 55, 12 * 60 + 5),),
            'v14': (TimeWindow(11 * 60 + 55, 18 * 60 + 5),),
        },
        3: {'v44_ny': (TimeWindow(8 * 60 + 55, 12 * 60 + 5),)},
        4: {
            'v44_ny': (TimeWindow(8 * 60 + 55, 18 * 60 + 5),),
            'v14': (TimeWindow(8 * 60 + 55, 18 * 60 + 5),),
        },
    },
}


class ScheduledStrategy(StrategyFamily):
    def __init__(self, family_name: str, inner: StrategyFamily, schedule_name: str) -> None:
        self.family_name = family_name
        self.inner = inner
        self.schedule_name = schedule_name

    def _allowed(self, timestamp: Any) -> bool:
        ts = timestamp if getattr(timestamp, 'tzinfo', None) is not None else None
        if ts is None:
            from pandas import Timestamp
            ts = Timestamp(timestamp, tz='UTC')
        local_ts = ts.tz_convert(NY_TZ)
        weekday = int(local_ts.weekday())
        minute_of_day = int(local_ts.hour) * 60 + int(local_ts.minute)
        day_windows = SCHEDULES[self.schedule_name].get(weekday, {})
        windows = day_windows.get(self.family_name, ())
        return any(window.contains(minute_of_day) for window in windows)

    def evaluate(self, current_bar: BarView, history: HistoricalDataView, portfolio: PortfolioSnapshot) -> Signal | None:
        if not self._allowed(current_bar.timestamp):
            return None
        return self.inner.evaluate(current_bar, history, portfolio)

    def get_exit_conditions(self, position: PositionSnapshot, current_bar: BarView, history: HistoricalDataView) -> ExitAction | None:
        return self.inner.get_exit_conditions(position, current_bar, history)

    def fit(self, history: HistoricalDataView) -> None:
        hook = getattr(self.inner, 'fit', None)
        if callable(hook):
            hook(history)

    def on_position_opened(self, position: PositionSnapshot, signal: Signal, current_bar: BarView) -> None:
        hook = getattr(self.inner, 'on_position_opened', None)
        if callable(hook):
            hook(position, signal, current_bar)

    def on_position_closed(self, trade) -> None:
        hook = getattr(self.inner, 'on_position_closed', None)
        if callable(hook):
            hook(trade)


def build_engine(schedule_name: str) -> tuple[BacktestEngine, tuple[str, ...], int, int]:
    v14_cfg = V14TokyoStrategyConfig.from_json('/Users/codygnon/Documents/usdjpy_assistant/research_out/tokyo_optimized_v14_config.json')
    v2_cfg = V2LondonStrategyConfig.from_json('/Users/codygnon/Documents/usdjpy_assistant/research_out/v2_exp4_winner_baseline_config.json')
    v44_cfg = V44StrategyConfig.from_v44_json('/Users/codygnon/Documents/usdjpy_assistant/research_out/session_momentum_v44_base_config.json')

    v14 = ScheduledStrategy(v14_cfg.family_name, V14TokyoStrategy(v14_cfg), schedule_name)
    v2 = ScheduledStrategy(v2_cfg.family_name, V2LondonStrategy(v2_cfg), schedule_name)
    v44 = ScheduledStrategy(v44_cfg.family_name, V44NYStrategy(v44_cfg), schedule_name)

    max_total_open_positions = int(v14_cfg.config['position_sizing']['max_concurrent_positions']) + int(v2_cfg.config['account']['max_open_positions']) + int(v44_cfg.max_open_positions)
    max_total_units = int(v14_cfg.config['position_sizing']['max_units']) + 2_000_000 + int(v44_cfg.rp_max_lot * 100_000)
    engine = BacktestEngine({
        v2_cfg.family_name: v2,
        v44_cfg.family_name: v44,
        v14_cfg.family_name: v14,
    })
    return engine, (v2_cfg.family_name, v44_cfg.family_name, v14_cfg.family_name), max_total_open_positions, max_total_units


def main() -> None:
    parser = argparse.ArgumentParser(description='Run integrated clean-room replay under a top-level Phase 3 schedule gate.')
    parser.add_argument('--schedule', choices=tuple(SCHEDULES.keys()), required=True)
    parser.add_argument('--data', default='/Users/codygnon/Documents/usdjpy_assistant/research_out/regime_backtest_v14_standalone_500k/market_data_augmented.csv')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--initial-balance', type=float, default=100000.0)
    parser.add_argument('--fixed-spread-pips', type=float, default=2.0)
    parser.add_argument('--slippage-pips', type=float, default=0.1)
    parser.add_argument('--margin-leverage', type=float, default=33.3)
    parser.add_argument('--bar-log-format', choices=('csv', 'parquet'), default='csv')
    args = parser.parse_args()

    engine, families, max_total_open_positions, max_total_units = build_engine(args.schedule)
    config = RunConfig(
        hypothesis=f'integrated_schedule_replay_{args.schedule}',
        data_path=Path(args.data),
        output_dir=Path(args.output_dir),
        mode='integrated',
        active_families=families,
        instrument=InstrumentSpec(symbol='USDJPY', margin_rate=(1.0 / float(args.margin_leverage))),
        spread=SpreadConfig(spread_source='fixed', fixed=FixedSpreadConfig(spread_pips=float(args.fixed_spread_pips))),
        slippage=SlippageConfig(fixed_slippage_pips=float(args.slippage_pips)),
        admission=AdmissionConfig(
            allow_opposing_exposure=False,
            max_total_open_positions=max_total_open_positions,
            max_open_positions_per_family={'london_v2': 5, 'v44_ny': 3, 'v14': 1},
            max_total_units=max_total_units,
            max_units_per_family={'london_v2': 2_000_000, 'v44_ny': int(20 * 100_000), 'v14': 500_000},
            family_priority=families,
        ),
        initial_balance=float(args.initial_balance),
        bar_log_format=args.bar_log_format,
    )
    result = engine.run(config)
    print(json.dumps(result.summary, indent=2))
    print(result.trade_log_path)
    print(result.bar_log_path)
    print(result.config_snapshot_path)


if __name__ == '__main__':
    main()
