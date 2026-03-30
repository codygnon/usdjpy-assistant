#!/usr/bin/env python3
"""
Canonical narrow-system backtest:

    Variant K defensive baseline
    + frozen London Setup D whitespace slice

The offensive slice is fixed to the first validated candidate:
  - strategy: london_v2
  - setup: D
  - direction: long-only
  - ownership_cell: mean_reversion/er_low/der_neg
  - timing gate: first_30min (15-45 minutes after London open)

This script is intentionally narrow. It is the canonical entry point for
getting final portfolio stats for the current "new system" candidate without
reopening broader router questions.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path("/Users/codygnon/Documents/usdjpy_assistant")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import backtest_offensive_setupd_additive as additive
from scripts import diagnostic_london_setupd_trade_outcomes as setupd_outcomes


OUT_DIR = ROOT / "research_out"
DEFAULT_OUTPUT = OUT_DIR / "system_variant_k_plus_setupd_whitespace.json"
DEFAULT_ADDITIVE_ARTIFACT = OUT_DIR / "offensive_setupd_additive_mean_reversion_er_low_der_neg_100pct_first30.json"
TARGET_CELLS = ["mean_reversion/er_low/der_neg"]
FILTER_NAME = "first_30min"
SIZE_SCALES = [1.0]
DATASETS = [
    str(OUT_DIR / "USDJPY_M1_OANDA_500k.csv"),
    str(OUT_DIR / "USDJPY_M1_OANDA_1000k.csv"),
]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return str(obj)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Canonical Variant K + Setup D whitespace system backtest")
    p.add_argument("--dataset", nargs="+", default=DATASETS, help="Dataset path(s) to backtest.")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output JSON path.")
    p.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore the validated additive artifact and recompute from raw research paths.",
    )
    return p.parse_args()


def _build_payload(results: dict[str, Any]) -> dict[str, Any]:
    datasets_out: dict[str, Any] = {}
    for dataset_key, result in results.items():
        variant = result["variants"]["100pct"]
        datasets_out[dataset_key] = {
            "baseline_summary": result["baseline_summary"],
            "system_summary": variant["variant_summary"],
            "delta_vs_baseline": variant["delta_vs_baseline"],
            "selection_counts": variant["selection_counts"],
            "samples": variant["samples"],
        }

    return {
        "title": "Variant K plus Setup D whitespace system backtest",
        "system_name": "variant_k_plus_setupd_whitespace_v1",
        "frozen_rule": {
            "strategy": "london_v2",
            "setup": "D",
            "direction": "long_only",
            "ownership_cell": "mean_reversion/er_low/der_neg",
            "timing_filter": FILTER_NAME,
            "timing_definition": "signal fires 15-45 minutes after London open",
            "size_scale": 1.0,
        },
        "datasets": datasets_out,
        "source_of_truth": {
            "baseline": str(ROOT / "scripts" / "backtest_variant_k_london_cluster.py"),
            "offensive_trade_path": str(ROOT / "scripts" / "diagnostic_london_setupd_trade_outcomes.py"),
            "coupling": str(ROOT / "scripts" / "backtest_offensive_setupd_additive.py"),
        },
    }


def _canonical_results_from_additive_artifact(path: Path) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, Any] = {}
    for dataset_key, result in artifact["results"].items():
        variant = result["variants"]["100pct"]
        out[dataset_key] = {
            "baseline_summary": result["baseline_summary"],
            "system_summary": variant["variant_summary"],
            "delta_vs_baseline": variant["delta_vs_baseline"],
            "selection_counts": variant["selection_counts"],
            "samples": variant["samples"],
        }
    return out


def main() -> int:
    args = _parse_args()
    output_path = Path(args.output)
    additive_artifact_usable = (
        not args.force_recompute
        and Path(DEFAULT_ADDITIVE_ARTIFACT).exists()
        and [str(Path(ds).resolve()) for ds in args.dataset] == [str(Path(ds).resolve()) for ds in DATASETS]
    )

    if additive_artifact_usable:
        payload = {
            "title": "Variant K plus Setup D whitespace system backtest",
            "system_name": "variant_k_plus_setupd_whitespace_v1",
            "frozen_rule": {
                "strategy": "london_v2",
                "setup": "D",
                "direction": "long_only",
                "ownership_cell": "mean_reversion/er_low/der_neg",
                "timing_filter": FILTER_NAME,
                "timing_definition": "signal fires 15-45 minutes after London open",
                "size_scale": 1.0,
            },
            "datasets": _canonical_results_from_additive_artifact(DEFAULT_ADDITIVE_ARTIFACT),
            "source_of_truth": {
                "baseline": str(ROOT / "scripts" / "backtest_variant_k_london_cluster.py"),
                "offensive_trade_path": str(ROOT / "scripts" / "diagnostic_london_setupd_trade_outcomes.py"),
                "coupling": str(ROOT / "scripts" / "backtest_offensive_setupd_additive.py"),
                "reused_validated_artifact": str(DEFAULT_ADDITIVE_ARTIFACT),
            },
        }
    else:
        reports = [setupd_outcomes.run_dataset(ds, additive.TRADE_OUTCOME_ARGS) for ds in args.dataset]
        reports_by_path = {str(Path(ds).resolve()): report for ds, report in zip(args.dataset, reports)}

        results = {
            result["dataset"]: result
            for result in [
                additive.run_dataset(
                    ds,
                    TARGET_CELLS,
                    reports_by_path[str(Path(ds).resolve())],
                    size_scales=SIZE_SCALES,
                    filter_name=FILTER_NAME,
                )
                for ds in args.dataset
            ]
        }
        payload = _build_payload(results)

    output_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(payload, indent=2, default=_json_default))
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
