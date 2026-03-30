#!/usr/bin/env python3
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
from scripts import backtest_variant_k_v6_search as v6
from scripts import backtest_offensive_slice_family_combo as family_combo
from scripts import run_offensive_slice_discovery as discovery

OUT_DIR = ROOT / "research_out"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return discovery._json_default(obj)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--cell-label", action="append", dest="cell_labels", required=True)
    ap.add_argument("--output-json", default="")
    ap.add_argument("--output-md", default="")
    return ap.parse_args()


def _policy() -> additive.ConflictPolicy:
    return additive.ConflictPolicy(
        name="native_v44_hedging_like",
        hedging_enabled=True,
        allow_internal_overlap=True,
        allow_opposite_side_overlap=True,
        max_open_offensive=None,
        max_entries_per_day=None,
        margin_model_enabled=True,
        margin_leverage=33.3,
        margin_buffer_pct=0.0,
        max_lot_per_trade=20.0,
    )


def _validate_labels(cell_labels: list[str]) -> None:
    unknown = [label for label in cell_labels if label not in v6.CELL_REGISTRY]
    if unknown:
        raise SystemExit(f"Unknown cell labels: {unknown}")


def _build_payload(name: str, cell_labels: list[str]) -> dict[str, Any]:
    matrix = family_combo._load_matrix(v6.DEFAULT_MATRIX)
    specs = v6._load_all_specs(matrix)
    _validate_labels(cell_labels)
    strategies = {specs[label].strategy for label in cell_labels if label in specs}
    all_trades = discovery._load_all_normalized_trades(strategies)
    trades_by_ds = v6._select_all_trades(specs, all_trades)
    policy = _policy()

    datasets: dict[str, Any] = {}
    for dataset_key in ["500k", "1000k"]:
        baseline_ctx = additive.build_baseline_context(discovery.DATASETS[dataset_key])
        combined = v6._collect_family_trades(cell_labels, trades_by_ds[dataset_key])
        result = additive.run_slice_additive_with_policy(
            baseline_ctx=baseline_ctx,
            slice_spec={"variant": name, "cells": cell_labels},
            selected_trades=combined,
            conflict_policy=policy,
            size_scale=1.0,
        )
        datasets[dataset_key] = {
            "summary": result["variant_summary"],
            "delta_vs_baseline": result["delta_vs_baseline"],
            "selection_counts": result["selection_counts"],
            "policy_stats": result["policy_stats"],
        }

    combined_usd = round(
        datasets["500k"]["delta_vs_baseline"]["net_usd"]
        + datasets["1000k"]["delta_vs_baseline"]["net_usd"],
        2,
    )
    combined_pf = round(
        datasets["500k"]["delta_vs_baseline"]["profit_factor"]
        + datasets["1000k"]["delta_vs_baseline"]["profit_factor"],
        4,
    )
    return {
        "title": "V6 Combo Candidate Validation",
        "name": name,
        "cells": cell_labels,
        "policy": {
            "name": policy.name,
            "hedging_enabled": policy.hedging_enabled,
            "allow_internal_overlap": policy.allow_internal_overlap,
            "allow_opposite_side_overlap": policy.allow_opposite_side_overlap,
            "margin_model_enabled": policy.margin_model_enabled,
            "margin_leverage": policy.margin_leverage,
            "margin_buffer_pct": policy.margin_buffer_pct,
            "max_lot_per_trade": policy.max_lot_per_trade,
        },
        "combined_delta_usd": combined_usd,
        "combined_delta_pf": combined_pf,
        "datasets": datasets,
    }


def _build_md(payload: dict[str, Any]) -> str:
    lines = [
        "# V6 Combo Candidate Validation",
        "",
        f"- name: `{payload['name']}`",
        f"- cells: `{', '.join(payload['cells'])}`",
        f"- policy: `{payload['policy']['name']}`",
        "",
        f"- combined delta USD: `{payload['combined_delta_usd']}`",
        f"- combined delta PF: `{payload['combined_delta_pf']}`",
        "",
    ]
    for dataset_key in ["500k", "1000k"]:
        ds = payload["datasets"][dataset_key]
        summary = ds["summary"]
        delta = ds["delta_vs_baseline"]
        selection = ds["selection_counts"]
        stats = ds["policy_stats"]
        lines += [
            f"## {dataset_key}",
            "",
            f"- total trades: `{summary['total_trades']}`",
            f"- net USD: `{round(summary['net_usd'], 2)}`",
            f"- PF: `{round(summary['profit_factor'], 4)}`",
            f"- max DD: `{round(summary['max_drawdown_usd'], 2)}`",
            f"- delta USD: `{delta['net_usd']}`",
            f"- delta PF: `{delta['profit_factor']}`",
            f"- delta DD: `{delta['max_drawdown_usd']}`",
            f"- raw selected: `{selection['raw_selected_trade_count']}`",
            f"- exact overlap: `{selection['exact_baseline_overlap_count']}`",
            f"- margin selected: `{selection['margin_selected_trade_count']}`",
            f"- additive trades: `{selection['new_additive_trades_count']}`",
            f"- internal overlap pairs: `{selection['internal_overlap_pairs']}`",
            f"- internal opposite-side pairs: `{selection['internal_opposite_side_overlap_pairs']}`",
            f"- policy stats: `{json.dumps(stats, sort_keys=True)}`",
            "",
        ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parse_args()
    payload = _build_payload(args.name, args.cell_labels)
    output_json = Path(args.output_json) if args.output_json else OUT_DIR / f"{args.name}_validation.json"
    output_md = Path(args.output_md) if args.output_md else OUT_DIR / f"{args.name}_validation.md"
    output_json.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    output_md.write_text(_build_md(payload), encoding="utf-8")
    print(output_json)
    print(output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
