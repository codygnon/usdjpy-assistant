#!/usr/bin/env python3
"""
Canonical offensive family system backtest:

    Variant K defensive baseline
    + refined V44 offensive family

Refinements:
  - ambiguous/er_high/der_pos short -> Strong-only
  - ambiguous/er_low/der_pos short -> base
  - momentum/er_high/der_pos short -> base
  - ambiguous/er_mid/der_neg long -> base (all trades already Strong/pullback)
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

from scripts import backtest_offensive_slice_additive as additive
from scripts import backtest_offensive_slice_family_combo as family_combo
from scripts import run_offensive_slice_discovery as discovery

OUT_DIR = ROOT / "research_out"
DEFAULT_OUTPUT = OUT_DIR / "system_variant_k_plus_v44_offensive_family_upgraded.json"
DEFAULT_FAMILY_ARTIFACT = OUT_DIR / "v44_offensive_family_upgraded_combo.json"
SLICE_IDS = [
    "v44_ny__short__cells_ambiguous_er_high_der_pos__entry_profiles_Strong",
    "v44_ny__short__cells_ambiguous_er_low_der_pos",
    "v44_ny__short__cells_momentum_er_high_der_pos",
    "v44_ny__long__cells_ambiguous_er_mid_der_neg",
]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return discovery._json_default(obj)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Canonical Variant K + refined V44 offensive family system backtest")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    p.add_argument("--force-recompute", action="store_true")
    return p.parse_args()


def _payload_from_family_artifact(path: Path) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    top4 = next(v for v in artifact["variants"] if v["name"] == "top4")
    datasets_out = {}
    for dataset_key, ds in top4["datasets"].items():
        variant = ds["size_sweep"]["100pct"]
        baseline_ctx = additive.build_baseline_context(discovery.DATASETS[dataset_key])
        datasets_out[dataset_key] = {
            "baseline_summary": baseline_ctx.baseline_summary,
            "system_summary": variant["variant_summary"],
            "delta_vs_baseline": variant["delta_vs_baseline"],
            "selection_counts": variant["selection_counts"],
            "standalone_summary": ds["standalone"],
            "coverage_by_cell": ds["coverage_by_cell"],
            "samples": variant["samples"],
        }
    return {
        "title": "Variant K plus refined V44 offensive family system backtest",
        "system_name": "variant_k_plus_v44_offensive_family_upgraded_v1",
        "family_rule": {
            "strategy": "v44_ny",
            "slice_ids": SLICE_IDS,
            "family_variant": "top4",
            "direction_mix": {
                "short": [
                    "ambiguous/er_high/der_pos__entry_profiles_Strong",
                    "ambiguous/er_low/der_pos",
                    "momentum/er_high/der_pos",
                ],
                "long": [
                    "ambiguous/er_mid/der_neg",
                ],
            },
            "size_scale": 1.0,
        },
        "datasets": datasets_out,
        "source_of_truth": {
            "baseline": str(ROOT / "scripts" / "backtest_variant_k_london_cluster.py"),
            "offensive_trade_path": str(ROOT / "research_out" / "phase1_v44_baseline_500k_report.json"),
            "coupling": str(ROOT / "scripts" / "backtest_offensive_slice_family_combo.py"),
            "reused_validated_artifact": str(path),
        },
    }


def _recompute_payload() -> dict[str, Any]:
    matrix = family_combo._load_matrix(family_combo.DEFAULT_MATRIX)
    specs = family_combo._specs_from_ids(matrix, SLICE_IDS)
    all_trades = discovery._load_all_normalized_trades({"v44_ny"})
    top4 = family_combo._summarize_variant(name="top4", specs=specs, all_trades=all_trades)
    datasets_out = {}
    for dataset_key, ds in top4["datasets"].items():
        variant = ds["size_sweep"]["100pct"]
        baseline_ctx = additive.build_baseline_context(discovery.DATASETS[dataset_key])
        datasets_out[dataset_key] = {
            "baseline_summary": baseline_ctx.baseline_summary,
            "system_summary": variant["variant_summary"],
            "delta_vs_baseline": variant["delta_vs_baseline"],
            "selection_counts": variant["selection_counts"],
            "standalone_summary": ds["standalone"],
            "coverage_by_cell": ds["coverage_by_cell"],
            "samples": variant["samples"],
        }
    return {
        "title": "Variant K plus refined V44 offensive family system backtest",
        "system_name": "variant_k_plus_v44_offensive_family_upgraded_v1",
        "family_rule": {
            "strategy": "v44_ny",
            "slice_ids": SLICE_IDS,
            "family_variant": "top4",
            "direction_mix": {
                "short": [
                    "ambiguous/er_high/der_pos__entry_profiles_Strong",
                    "ambiguous/er_low/der_pos",
                    "momentum/er_high/der_pos",
                ],
                "long": [
                    "ambiguous/er_mid/der_neg",
                ],
            },
            "size_scale": 1.0,
        },
        "datasets": datasets_out,
        "source_of_truth": {
            "baseline": str(ROOT / "scripts" / "backtest_variant_k_london_cluster.py"),
            "offensive_trade_path": str(ROOT / "research_out" / "phase1_v44_baseline_500k_report.json"),
            "coupling": str(ROOT / "scripts" / "backtest_offensive_slice_family_combo.py"),
        },
    }


def main() -> int:
    args = _parse_args()
    output_path = Path(args.output)
    if not args.force_recompute and DEFAULT_FAMILY_ARTIFACT.exists():
        payload = _payload_from_family_artifact(DEFAULT_FAMILY_ARTIFACT)
    else:
        payload = _recompute_payload()
    output_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(payload, indent=2, default=_json_default))
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
