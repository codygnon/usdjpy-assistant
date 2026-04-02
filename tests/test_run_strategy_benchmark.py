from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_strategy_benchmark import (
    BenchmarkOutcome,
    BenchmarkStrategySpec,
    _dedupe_and_validate_specs,
    _parse_cli_strategy,
    _sorted_successes,
)


def test_parse_cli_strategy_accepts_config_alias() -> None:
    spec = _parse_cli_strategy("kind=v44,label=V44 Base,config=/tmp/v44.json")

    assert spec == BenchmarkStrategySpec(
        label="V44 Base",
        kind="v44",
        config_path=Path("/tmp/v44.json"),
    )


def test_dedupe_rejects_slug_collisions() -> None:
    specs = [
        BenchmarkStrategySpec(label="V44 Base", kind="v44", config_path=Path("/tmp/a.json")),
        BenchmarkStrategySpec(label="V44-Base", kind="v44", config_path=Path("/tmp/b.json")),
    ]

    with pytest.raises(ValueError, match="slugification"):
        _dedupe_and_validate_specs(specs)


def test_sorted_successes_ranks_descending_metrics() -> None:
    outcomes = [
        BenchmarkOutcome(
            label="lower",
            kind="v14",
            family_name="v14",
            config_path="/tmp/lower.json",
            status="ok",
            output_dir="/tmp/lower",
            prepared_data_path="/tmp/data.csv",
            net_pnl_usd=100.0,
        ),
        BenchmarkOutcome(
            label="higher",
            kind="v44",
            family_name="v44_ny",
            config_path="/tmp/higher.json",
            status="ok",
            output_dir="/tmp/higher",
            prepared_data_path="/tmp/data.csv",
            net_pnl_usd=250.0,
        ),
        BenchmarkOutcome(
            label="failed",
            kind="v2",
            family_name="london_v2",
            config_path="/tmp/failed.json",
            status="error",
            output_dir="/tmp/failed",
            prepared_data_path="/tmp/data.csv",
            error="boom",
        ),
    ]

    ranked = _sorted_successes(outcomes, "net_pnl_usd")

    assert [outcome.label for outcome in ranked] == ["higher", "lower"]


def test_sorted_successes_ranks_drawdown_metrics_ascending() -> None:
    outcomes = [
        BenchmarkOutcome(
            label="higher_dd",
            kind="v14",
            family_name="v14",
            config_path="/tmp/a.json",
            status="ok",
            output_dir="/tmp/a",
            prepared_data_path="/tmp/data.csv",
            max_drawdown_pct=12.0,
        ),
        BenchmarkOutcome(
            label="lower_dd",
            kind="v2",
            family_name="london_v2",
            config_path="/tmp/b.json",
            status="ok",
            output_dir="/tmp/b",
            prepared_data_path="/tmp/data.csv",
            max_drawdown_pct=4.0,
        ),
    ]

    ranked = _sorted_successes(outcomes, "max_drawdown_pct")

    assert [outcome.label for outcome in ranked] == ["lower_dd", "higher_dd"]
