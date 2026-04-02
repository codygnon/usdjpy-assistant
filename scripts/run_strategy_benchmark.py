from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.regime_backtest_engine import (
    AdmissionConfig,
    BacktestEngine,
    FixedSpreadConfig,
    InstrumentSpec,
    RunConfig,
    SlippageConfig,
    SpreadConfig,
    V14TokyoStrategy,
    V14TokyoStrategyConfig,
    V2LondonStrategy,
    V2LondonStrategyConfig,
    V44NYStrategy,
    V44StrategyConfig,
    prepare_v14_augmented_data,
)


StrategyKind = Literal["v14", "v2", "v44"]


class BenchmarkStrategySpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str = Field(min_length=1)
    kind: StrategyKind
    config_path: Path
    family_name: str | None = None

    @model_validator(mode="after")
    def validate_label(self) -> "BenchmarkStrategySpec":
        if not self.label.strip():
            raise ValueError("strategy label must not be blank")
        return self


@dataclass(frozen=True)
class BenchmarkRun:
    spec: BenchmarkStrategySpec
    config: RunConfig
    strategy: Any
    run_dir: Path
    prepared_data_path: Path


@dataclass(frozen=True)
class BenchmarkOutcome:
    label: str
    kind: str
    family_name: str
    config_path: str
    status: str
    output_dir: str
    prepared_data_path: str
    trade_log_path: str | None = None
    bar_log_path: str | None = None
    config_snapshot_path: str | None = None
    manifest_path: str | None = None
    arbitration_log_path: str | None = None
    hypothesis: str | None = None
    mode: str | None = None
    processed_bar_count: int | None = None
    processed_start_time: str | None = None
    processed_end_time: str | None = None
    synthetic_bid_ask: bool | None = None
    initial_balance: float | None = None
    ending_balance: float | None = None
    ending_equity: float | None = None
    net_pnl_usd: float | None = None
    return_pct: float | None = None
    profit_factor: float | None = None
    trade_count: int | None = None
    win_rate: float | None = None
    max_drawdown_usd: float | None = None
    max_drawdown_pct: float | None = None
    pnl_to_max_drawdown: float | None = None
    max_concurrent_positions: int | None = None
    arbitration_event_count: int | None = None
    error: str | None = None


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").lower()
    return slug or "strategy"


def _parse_cli_strategy(value: str) -> BenchmarkStrategySpec:
    parts: dict[str, str] = {}
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(
                "strategy specs must use comma-separated key=value pairs, for example: "
                "kind=v44,label=v44_base,config=/abs/path/config.json"
            )
        key, raw_val = part.split("=", 1)
        parts[key.strip()] = raw_val.strip()
    if "config" in parts and "config_path" not in parts:
        parts["config_path"] = parts.pop("config")
    return BenchmarkStrategySpec.model_validate(parts)


def _load_specs_from_file(path: Path) -> list[BenchmarkStrategySpec]:
    payload = json.loads(path.read_text())
    strategies = payload["strategies"] if isinstance(payload, dict) else payload
    specs: list[BenchmarkStrategySpec] = []
    for item in strategies:
        normalized = dict(item)
        if "config" in normalized and "config_path" not in normalized:
            normalized["config_path"] = normalized.pop("config")
        if "config_path" in normalized:
            cfg_path = Path(normalized["config_path"])
            if not cfg_path.is_absolute():
                normalized["config_path"] = str((path.parent / cfg_path).resolve())
        specs.append(BenchmarkStrategySpec.model_validate(normalized))
    return specs


def _dedupe_and_validate_specs(specs: list[BenchmarkStrategySpec]) -> list[BenchmarkStrategySpec]:
    if not specs:
        raise ValueError("provide at least one strategy via --strategy or --spec-file")
    seen_labels: set[str] = set()
    seen_slugs: set[str] = set()
    for spec in specs:
        if spec.label in seen_labels:
            raise ValueError(f"duplicate strategy label: {spec.label}")
        seen_labels.add(spec.label)
        slug = _slugify(spec.label)
        if slug in seen_slugs:
            raise ValueError(
                f"strategy labels must remain unique after slugification; "
                f"label {spec.label!r} collides with another label"
            )
        seen_slugs.add(slug)
    return specs


def _common_spread(args: argparse.Namespace) -> SpreadConfig:
    if args.spread_mode == "from_data":
        return SpreadConfig(spread_source="from_data")
    return SpreadConfig(
        spread_source="fixed",
        fixed=FixedSpreadConfig(spread_pips=float(args.fixed_spread_pips)),
    )


def _common_slippage(args: argparse.Namespace) -> SlippageConfig:
    return SlippageConfig(fixed_slippage_pips=float(args.slippage_pips))


def _common_instrument(args: argparse.Namespace) -> InstrumentSpec:
    return InstrumentSpec(symbol=args.symbol, margin_rate=(1.0 / float(args.margin_leverage)))


def _build_v14_run(spec: BenchmarkStrategySpec, data_path: Path, run_dir: Path, args: argparse.Namespace) -> BenchmarkRun:
    strategy_cfg = V14TokyoStrategyConfig.from_json(spec.config_path, family_name=spec.family_name or "v14")
    cfg = strategy_cfg.config
    max_open = int(cfg["position_sizing"]["max_concurrent_positions"])
    max_units = int(cfg["position_sizing"]["max_units"])
    prepared_path = run_dir / "market_data_augmented.csv"
    prepare_v14_augmented_data(data_path, spec.config_path, prepared_path)
    run_config = RunConfig(
        hypothesis=args.hypothesis,
        data_path=prepared_path,
        output_dir=run_dir,
        mode="standalone",
        active_families=(strategy_cfg.family_name,),
        instrument=_common_instrument(args),
        spread=_common_spread(args),
        slippage=_common_slippage(args),
        admission=AdmissionConfig(
            allow_opposing_exposure=False,
            max_total_open_positions=max(1, max_open),
            max_open_positions_per_family={strategy_cfg.family_name: max(1, max_open)},
            max_total_units=max(1, max_units),
            max_units_per_family={strategy_cfg.family_name: max(1, max_units)},
            family_priority=(strategy_cfg.family_name,),
        ),
        initial_balance=float(args.initial_balance),
        start_index=args.start_index,
        end_index=args.end_index,
        bar_log_format=args.bar_log_format,
    )
    return BenchmarkRun(
        spec=spec,
        config=run_config,
        strategy=V14TokyoStrategy(strategy_cfg),
        run_dir=run_dir,
        prepared_data_path=prepared_path,
    )


def _build_v2_run(spec: BenchmarkStrategySpec, data_path: Path, run_dir: Path, args: argparse.Namespace) -> BenchmarkRun:
    strategy_cfg = V2LondonStrategyConfig.from_json(spec.config_path, family_name=spec.family_name or "london_v2")
    cfg = strategy_cfg.config
    max_open = int(cfg["account"]["max_open_positions"])
    run_config = RunConfig(
        hypothesis=args.hypothesis,
        data_path=data_path,
        output_dir=run_dir,
        mode="standalone",
        active_families=(strategy_cfg.family_name,),
        instrument=_common_instrument(args),
        spread=_common_spread(args),
        slippage=_common_slippage(args),
        admission=AdmissionConfig(
            allow_opposing_exposure=False,
            max_total_open_positions=max(1, max_open),
            max_open_positions_per_family={strategy_cfg.family_name: max(1, max_open)},
            max_total_units=2_000_000,
            max_units_per_family={strategy_cfg.family_name: 2_000_000},
            family_priority=(strategy_cfg.family_name,),
        ),
        initial_balance=float(args.initial_balance),
        start_index=args.start_index,
        end_index=args.end_index,
        bar_log_format=args.bar_log_format,
    )
    return BenchmarkRun(
        spec=spec,
        config=run_config,
        strategy=V2LondonStrategy(strategy_cfg),
        run_dir=run_dir,
        prepared_data_path=data_path,
    )


def _build_v44_run(spec: BenchmarkStrategySpec, data_path: Path, run_dir: Path, args: argparse.Namespace) -> BenchmarkRun:
    strategy_cfg = V44StrategyConfig.from_v44_json(spec.config_path, family_name=spec.family_name or "v44_ny")
    max_positions = max(1, int(strategy_cfg.max_open_positions))
    max_units = int(max(1, strategy_cfg.rp_max_lot * 100_000))
    run_config = RunConfig(
        hypothesis=args.hypothesis,
        data_path=data_path,
        output_dir=run_dir,
        mode="standalone",
        active_families=(strategy_cfg.family_name,),
        instrument=_common_instrument(args),
        spread=_common_spread(args),
        slippage=_common_slippage(args),
        admission=AdmissionConfig(
            allow_opposing_exposure=False,
            max_total_open_positions=max_positions,
            max_open_positions_per_family={strategy_cfg.family_name: max_positions},
            max_total_units=max_units,
            max_units_per_family={strategy_cfg.family_name: max_units},
            family_priority=(strategy_cfg.family_name,),
        ),
        initial_balance=float(args.initial_balance),
        start_index=args.start_index,
        end_index=args.end_index,
        bar_log_format=args.bar_log_format,
    )
    return BenchmarkRun(
        spec=spec,
        config=run_config,
        strategy=V44NYStrategy(strategy_cfg),
        run_dir=run_dir,
        prepared_data_path=data_path,
    )


def _build_run(spec: BenchmarkStrategySpec, data_path: Path, output_dir: Path, args: argparse.Namespace) -> BenchmarkRun:
    run_dir = output_dir / "runs" / _slugify(spec.label)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if spec.kind == "v14":
        return _build_v14_run(spec, data_path, run_dir, args)
    if spec.kind == "v2":
        return _build_v2_run(spec, data_path, run_dir, args)
    if spec.kind == "v44":
        return _build_v44_run(spec, data_path, run_dir, args)
    raise ValueError(f"unsupported strategy kind: {spec.kind}")


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def _outcome_from_result(run: BenchmarkRun, result: Any) -> BenchmarkOutcome:
    summary = result.summary
    initial_balance = _safe_float(summary.get("initial_balance"))
    net_pnl = _safe_float(summary.get("net_pnl_usd"))
    max_dd = _safe_float(summary.get("max_drawdown_usd"))
    return_pct = ((net_pnl / initial_balance) * 100.0) if initial_balance and net_pnl is not None else None
    pnl_to_dd = (net_pnl / max_dd) if net_pnl is not None and max_dd not in (None, 0.0) else None
    return BenchmarkOutcome(
        label=run.spec.label,
        kind=run.spec.kind,
        family_name=run.config.active_families[0],
        config_path=str(run.spec.config_path),
        status="ok",
        output_dir=str(run.run_dir),
        prepared_data_path=str(run.prepared_data_path),
        trade_log_path=str(result.trade_log_path),
        bar_log_path=str(result.bar_log_path),
        config_snapshot_path=str(result.config_snapshot_path),
        manifest_path=str(result.manifest_path) if result.manifest_path is not None else None,
        arbitration_log_path=str(result.arbitration_log_path) if result.arbitration_log_path is not None else None,
        hypothesis=str(summary.get("hypothesis")),
        mode=str(summary.get("mode")),
        processed_bar_count=int(summary["processed_bar_count"]),
        processed_start_time=str(summary.get("processed_start_time")),
        processed_end_time=str(summary.get("processed_end_time")),
        synthetic_bid_ask=bool(summary.get("synthetic_bid_ask")),
        initial_balance=initial_balance,
        ending_balance=_safe_float(summary.get("ending_balance")),
        ending_equity=_safe_float(summary.get("ending_equity")),
        net_pnl_usd=net_pnl,
        return_pct=return_pct,
        profit_factor=_safe_float(summary.get("profit_factor")),
        trade_count=int(summary["trade_count"]) if summary.get("trade_count") is not None else None,
        win_rate=_safe_float(summary.get("win_rate")),
        max_drawdown_usd=max_dd,
        max_drawdown_pct=_safe_float(summary.get("max_drawdown_pct")),
        pnl_to_max_drawdown=pnl_to_dd,
        max_concurrent_positions=int(summary["max_concurrent_positions"]) if summary.get("max_concurrent_positions") is not None else None,
        arbitration_event_count=int(summary["arbitration_event_count"]) if summary.get("arbitration_event_count") is not None else None,
    )


def _failure_outcome(run: BenchmarkRun, exc: Exception) -> BenchmarkOutcome:
    return BenchmarkOutcome(
        label=run.spec.label,
        kind=run.spec.kind,
        family_name=run.config.active_families[0],
        config_path=str(run.spec.config_path),
        status="error",
        output_dir=str(run.run_dir),
        prepared_data_path=str(run.prepared_data_path),
        error=str(exc),
    )


def _rank_sort_value(outcome: BenchmarkOutcome, rank_by: str) -> tuple[int, float]:
    raw = getattr(outcome, rank_by)
    if raw is None:
        return (1, 0.0)
    value = float(raw)
    if rank_by in {"max_drawdown_usd", "max_drawdown_pct"}:
        return (0, value)
    return (0, -value)


def _sorted_successes(outcomes: list[BenchmarkOutcome], rank_by: str) -> list[BenchmarkOutcome]:
    successes = [outcome for outcome in outcomes if outcome.status == "ok"]
    return sorted(successes, key=lambda outcome: _rank_sort_value(outcome, rank_by))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _write_csv(path: Path, rows: list[BenchmarkOutcome]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(asdict(rows[0]).keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _persist_run_outcome(run_dir: Path, outcome: BenchmarkOutcome, error_traceback: str | None = None) -> None:
    _write_json(run_dir / "benchmark_outcome.json", asdict(outcome))
    if error_traceback is not None:
        (run_dir / "error.txt").write_text(error_traceback)


def _print_summary(outcomes: list[BenchmarkOutcome], rank_by: str) -> None:
    ranked = _sorted_successes(outcomes, rank_by)
    if not ranked:
        print("No successful benchmark runs.")
        return
    print(f"Leaderboard (ranked by {rank_by})")
    for index, outcome in enumerate(ranked, start=1):
        print(
            f"{index}. {outcome.label} [{outcome.kind}] "
            f"net_pnl_usd={outcome.net_pnl_usd} "
            f"profit_factor={outcome.profit_factor} "
            f"win_rate={outcome.win_rate} "
            f"max_drawdown_usd={outcome.max_drawdown_usd} "
            f"trade_count={outcome.trade_count}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run multiple clean-room strategies independently under one shared benchmark harness "
            "for apples-to-apples comparison."
        )
    )
    parser.add_argument("--data", required=True, help="Shared input data file used for every strategy.")
    parser.add_argument("--output-dir", required=True, help="Benchmark output root.")
    parser.add_argument("--strategy", action="append", default=[], help="Strategy spec like kind=v44,label=v44_base,config=/abs/path/config.json")
    parser.add_argument("--spec-file", action="append", default=[], help="JSON file containing a strategies array.")
    parser.add_argument("--hypothesis", default="strategy_benchmark")
    parser.add_argument("--symbol", default="USDJPY")
    parser.add_argument("--initial-balance", type=float, default=100000.0)
    parser.add_argument("--spread-mode", choices=("fixed", "from_data"), default="fixed")
    parser.add_argument("--fixed-spread-pips", type=float, default=2.0)
    parser.add_argument("--slippage-pips", type=float, default=0.1)
    parser.add_argument("--margin-leverage", type=float, default=33.3)
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--bar-log-format", choices=("csv", "parquet"), default="csv")
    parser.add_argument(
        "--rank-by",
        choices=(
            "net_pnl_usd",
            "profit_factor",
            "win_rate",
            "ending_equity",
            "trade_count",
            "return_pct",
            "pnl_to_max_drawdown",
            "max_drawdown_usd",
            "max_drawdown_pct",
        ),
        default="net_pnl_usd",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def _collect_specs(args: argparse.Namespace) -> list[BenchmarkStrategySpec]:
    specs: list[BenchmarkStrategySpec] = []
    for raw in args.strategy:
        specs.append(_parse_cli_strategy(raw))
    for raw_path in args.spec_file:
        specs.extend(_load_specs_from_file(Path(raw_path).resolve()))
    return _dedupe_and_validate_specs(specs)


def main() -> None:
    args = build_arg_parser().parse_args()
    data_path = Path(args.data).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = _collect_specs(args)
    runs = [_build_run(spec, data_path, output_dir, args) for spec in specs]

    request_snapshot = {
        "data_path": str(data_path),
        "output_dir": str(output_dir),
        "hypothesis": args.hypothesis,
        "symbol": args.symbol,
        "initial_balance": float(args.initial_balance),
        "spread_mode": args.spread_mode,
        "fixed_spread_pips": float(args.fixed_spread_pips),
        "slippage_pips": float(args.slippage_pips),
        "margin_leverage": float(args.margin_leverage),
        "start_index": args.start_index,
        "end_index": args.end_index,
        "bar_log_format": args.bar_log_format,
        "rank_by": args.rank_by,
        "strategies": [spec.model_dump(mode="json") for spec in specs],
    }
    _write_json(output_dir / "benchmark_request.json", request_snapshot)

    outcomes: list[BenchmarkOutcome] = []
    for run in runs:
        try:
            result = BacktestEngine({run.config.active_families[0]: run.strategy}).run(run.config)
            outcome = _outcome_from_result(run, result)
            _persist_run_outcome(run.run_dir, outcome)
            outcomes.append(outcome)
        except Exception as exc:
            error_traceback = traceback.format_exc()
            outcome = _failure_outcome(run, exc)
            _persist_run_outcome(run.run_dir, outcome, error_traceback=error_traceback)
            outcomes.append(outcome)
            if not args.continue_on_error:
                _write_json(output_dir / "benchmark_results.json", [asdict(outcome) for outcome in outcomes])
                _write_csv(output_dir / "benchmark_results.csv", outcomes)
                raise

    _write_json(output_dir / "benchmark_results.json", [asdict(outcome) for outcome in outcomes])
    _write_csv(output_dir / "benchmark_results.csv", outcomes)
    _write_json(output_dir / "leaderboard.json", [asdict(outcome) for outcome in _sorted_successes(outcomes, args.rank_by)])
    _write_csv(output_dir / "leaderboard.csv", _sorted_successes(outcomes, args.rank_by))
    _print_summary(outcomes, args.rank_by)


if __name__ == "__main__":
    main()
