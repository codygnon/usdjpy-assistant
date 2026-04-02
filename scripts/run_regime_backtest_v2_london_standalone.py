from __future__ import annotations

import argparse
from pathlib import Path

from core.regime_backtest_engine import (
    AdmissionConfig,
    BacktestEngine,
    FixedSpreadConfig,
    InstrumentSpec,
    RunConfig,
    RunManifest,
    SlippageConfig,
    SpreadConfig,
    V2LondonStrategy,
    V2LondonStrategyConfig,
)


def build_run_config(args: argparse.Namespace, strategy_cfg: V2LondonStrategyConfig) -> RunConfig:
    cfg = strategy_cfg.config
    max_open = int(cfg["account"]["max_open_positions"])
    return RunConfig(
        hypothesis=args.hypothesis,
        data_path=Path(args.data),
        output_dir=Path(args.output_dir),
        mode="standalone",
        active_families=(strategy_cfg.family_name,),
        instrument=InstrumentSpec(symbol="USDJPY", margin_rate=(1.0 / float(args.margin_leverage))),
        spread=SpreadConfig(spread_source="fixed", fixed=FixedSpreadConfig(spread_pips=float(args.fixed_spread_pips))),
        slippage=SlippageConfig(fixed_slippage_pips=float(args.slippage_pips)),
        admission=AdmissionConfig(
            allow_opposing_exposure=False,
            max_total_open_positions=max(1, max_open),
            max_open_positions_per_family={strategy_cfg.family_name: max(1, max_open)},
            max_total_units=2_000_000,
            max_units_per_family={strategy_cfg.family_name: 2_000_000},
            family_priority=(strategy_cfg.family_name,),
        ),
        manifest=RunManifest(
            hypothesis=args.hypothesis,
            minimum_trade_count=int(args.minimum_trade_count),
            minimum_profit_factor=float(args.minimum_profit_factor),
            maximum_drawdown_usd=float(args.maximum_drawdown_usd),
            maximum_drawdown_pct=float(args.maximum_drawdown_pct),
            expected_win_rate_min=float(args.expected_win_rate_min),
            expected_win_rate_max=float(args.expected_win_rate_max),
        ),
        initial_balance=float(args.initial_balance),
        start_index=args.start_index,
        end_index=args.end_index,
        bar_log_format=args.bar_log_format,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run standalone V2 London through the clean-room regime backtest engine.")
    parser.add_argument("--data", default="/Users/codygnon/Documents/usdjpy_assistant/research_out/USDJPY_M1_OANDA_500k.csv")
    parser.add_argument("--v2-config", default="/Users/codygnon/Documents/usdjpy_assistant/research_out/v2_exp4_winner_baseline_config.json")
    parser.add_argument("--output-dir", default="/Users/codygnon/Documents/usdjpy_assistant/research_out/regime_backtest_v2_london_standalone_500k")
    parser.add_argument("--hypothesis", default="phase4_v2_london_standalone_validation")
    parser.add_argument("--initial-balance", type=float, default=100000.0)
    parser.add_argument("--fixed-spread-pips", type=float, default=2.0)
    parser.add_argument("--slippage-pips", type=float, default=0.1)
    parser.add_argument("--margin-leverage", type=float, default=33.3)
    parser.add_argument("--minimum-trade-count", type=int, default=30)
    parser.add_argument("--minimum-profit-factor", type=float, default=1.0)
    parser.add_argument("--maximum-drawdown-usd", type=float, default=25000.0)
    parser.add_argument("--maximum-drawdown-pct", type=float, default=25.0)
    parser.add_argument("--expected-win-rate-min", type=float, default=30.0)
    parser.add_argument("--expected-win-rate-max", type=float, default=80.0)
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--bar-log-format", choices=("csv", "parquet"), default="csv")
    args = parser.parse_args()

    strategy_cfg = V2LondonStrategyConfig.from_json(args.v2_config)
    strategy = V2LondonStrategy(strategy_cfg)
    config = build_run_config(args, strategy_cfg)
    result = BacktestEngine({strategy_cfg.family_name: strategy}).run(config)
    print(result.summary)
    print(result.trade_log_path)
    print(result.bar_log_path)
    print(result.config_snapshot_path)
    print(result.manifest_path)


if __name__ == "__main__":
    main()
