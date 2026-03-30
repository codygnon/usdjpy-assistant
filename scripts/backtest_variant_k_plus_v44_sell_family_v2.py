#!/usr/bin/env python3
"""
Variant K + V44 sell family v2 — asymmetric per-cell sizing.

System definition:
  Variant K defensive baseline
  + V44 NY sell family (4 cells, per-cell size scales)
  + Cell 1 filtered to Strong-only entry profile

Changes from v1:
  - Cell 1 (ambiguous/er_high/der_pos) filtered to Strong entry profile only
  - Cell 4 added (ambiguous/er_mid/der_neg | buy, promotable near-miss)
  - Per-cell size scales swept independently

Approach:
  For each sizing variant, pre-scale each cell's trades by its own size_scale,
  then merge all into a single additive batch at scale=1.0 for coupling.
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path("/Users/codygnon/Documents/usdjpy_assistant")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.offensive_slice_spec import OffensiveSliceSpec
from scripts import backtest_offensive_slice_additive as additive
from scripts import backtest_offensive_slice_family_combo as family_combo
from scripts import run_offensive_slice_discovery as discovery

OUT_DIR = ROOT / "research_out"
DEFAULT_OUTPUT = OUT_DIR / "system_variant_k_plus_v44_sell_family_v2.json"
DEFAULT_MATRIX = OUT_DIR / "offensive_slice_discovery_matrix.json"

# Cell definitions: (slice_id, label, direction_for_display)
CELLS = [
    {
        "label": "ambig_er_high_der_pos_sell_strong",
        "slice_id": "v44_ny__short__cells_ambiguous_er_high_der_pos__entry_profiles_Strong",
        "cell": "ambiguous/er_high/der_pos",
        "note": "Strong-only (upgrade from v1 base)",
    },
    {
        "label": "ambig_er_low_der_pos_sell",
        "slice_id": "v44_ny__short__cells_ambiguous_er_low_der_pos",
        "cell": "ambiguous/er_low/der_pos",
        "note": "base (no sub-filter improvement possible)",
    },
    {
        "label": "mom_er_high_der_pos_sell",
        "slice_id": "v44_ny__short__cells_momentum_er_high_der_pos",
        "cell": "momentum/er_high/der_pos",
        "note": "base (all trades already Strong)",
    },
    {
        "label": "ambig_er_mid_der_neg_buy",
        "slice_id": "v44_ny__long__cells_ambiguous_er_mid_der_neg",
        "cell": "ambiguous/er_mid/der_neg",
        "note": "NEW — promotable near-miss, all Strong/pullback",
    },
]

# Sizing variants to sweep
# Each variant maps cell_label -> size_scale
SCALE_OPTIONS = [0.5, 0.75, 1.0]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return discovery._json_default(obj)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Variant K + V44 family v2 — asymmetric sizing sweep")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return p.parse_args()


def _load_specs(matrix: dict[str, Any]) -> list[OffensiveSliceSpec]:
    specs = []
    for cell_def in CELLS:
        row = (matrix.get("results") or {}).get(cell_def["slice_id"])
        if not row:
            raise SystemExit(f"Slice id not found in matrix: {cell_def['slice_id']}")
        specs.append(OffensiveSliceSpec(**row["slice_spec"]))
    return specs


def _select_trades_per_cell(
    specs: list[OffensiveSliceSpec],
    all_trades: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, list[list[dict[str, Any]]]]:
    """Per dataset, per cell index -> list of selected trades."""
    out: dict[str, list[list[dict[str, Any]]]] = {}
    for dataset_key in ["500k", "1000k"]:
        cell_trades: list[list[dict[str, Any]]] = []
        for spec in specs:
            selected = [
                t for t in all_trades[dataset_key].get(spec.strategy, [])
                if discovery._passes_filters(t, spec)
            ]
            cell_trades.append(selected)
        out[dataset_key] = cell_trades
    return out


def _run_asymmetric_variant(
    *,
    cell_scales: dict[int, float],
    trades_per_cell: list[list[dict[str, Any]]],
    baseline_ctx: additive.BaselineContext,
) -> dict[str, Any]:
    """Run a single asymmetric sizing variant.

    Pre-scales each cell's trades by its own size_scale, then combines
    into one batch and runs additive merge at scale=1.0.
    """
    # Pre-scale: modify each trade's usd by its cell's scale
    combined: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    per_cell_counts: list[int] = []

    for cell_idx, cell_trade_list in enumerate(trades_per_cell):
        scale = cell_scales.get(cell_idx, 1.0)
        count = 0
        for trade in cell_trade_list:
            key = (str(trade["strategy"]), str(trade["entry_time"]),
                   str(trade["exit_time"]), str(trade["side"]))
            if key in seen:
                continue
            seen.add(key)
            if scale != 1.0:
                t = deepcopy(trade)
                t["usd"] = float(t["usd"]) * scale
                t["size_scale"] = float(t.get("size_scale", 1.0)) * scale
                combined.append(t)
            else:
                combined.append(trade)
            count += 1
        per_cell_counts.append(count)

    combined.sort(key=lambda t: (t["entry_time"], t["exit_time"]))

    result = additive.run_slice_additive(
        baseline_ctx=baseline_ctx,
        slice_spec={"variant": "asymmetric", "cell_scales": {str(k): v for k, v in cell_scales.items()}},
        selected_trades=combined,
        size_scale=1.0,  # scaling already applied per-cell
    )
    result["per_cell_selected_counts"] = per_cell_counts
    return result


def _build_named_variants() -> list[dict[str, Any]]:
    """Build the set of sizing variants to sweep."""
    variants: list[dict[str, Any]] = []

    # 1. Uniform variants
    for scale in SCALE_OPTIONS:
        variants.append({
            "name": f"uniform_{int(scale*100)}pct",
            "scales": {i: scale for i in range(len(CELLS))},
        })

    # 2. Per-cell individual sweep (vary one cell, others at 100%)
    for cell_idx in range(len(CELLS)):
        for scale in SCALE_OPTIONS:
            if scale == 1.0:
                continue  # already covered by uniform_100pct
            name = f"{CELLS[cell_idx]['label']}_at_{int(scale*100)}pct"
            scales = {i: 1.0 for i in range(len(CELLS))}
            scales[cell_idx] = scale
            variants.append({"name": name, "scales": scales})

    # 3. Best-guess asymmetric combos
    # Cell 0 (strongest): 100%, Cell 1: 100%, Cell 2 (smallest): 75%, Cell 3 (new, asymmetric): 75%
    variants.append({
        "name": "asymmetric_best_guess_A",
        "scales": {0: 1.0, 1: 1.0, 2: 0.75, 3: 0.75},
    })
    # Cell 0: 100%, Cell 1: 75%, Cell 2: 50%, Cell 3: 100%
    variants.append({
        "name": "asymmetric_best_guess_B",
        "scales": {0: 1.0, 1: 0.75, 2: 0.50, 3: 1.0},
    })
    # All cells at 75% (conservative)
    # Already covered by uniform_75pct
    # Full send
    # Already covered by uniform_100pct

    return variants


def main() -> int:
    args = _parse_args()
    matrix = family_combo._load_matrix(DEFAULT_MATRIX)
    specs = _load_specs(matrix)
    strategies = {s.strategy for s in specs}
    all_trades = discovery._load_all_normalized_trades(strategies)
    trades_per_cell = _select_trades_per_cell(specs, all_trades)

    # Print cell trade counts
    print("Cell definitions:")
    for i, cell_def in enumerate(CELLS):
        for ds in ["500k", "1000k"]:
            print(f"  [{i}] {cell_def['label']}  {ds}: {len(trades_per_cell[ds][i])} trades  ({cell_def['note']})")
    print()

    variants = _build_named_variants()
    results_by_variant: list[dict[str, Any]] = []

    print(f"Running {len(variants)} sizing variants across 2 datasets...")
    for vi, variant in enumerate(variants):
        vname = variant["name"]
        scales = variant["scales"]
        ds_results: dict[str, Any] = {}
        for dataset_key in ["500k", "1000k"]:
            baseline_ctx = additive.build_baseline_context(discovery.DATASETS[dataset_key])
            result = _run_asymmetric_variant(
                cell_scales=scales,
                trades_per_cell=trades_per_cell[dataset_key],
                baseline_ctx=baseline_ctx,
            )
            ds_results[dataset_key] = result

        # Quick summary
        d5 = ds_results["500k"]["delta_vs_baseline"]
        d1 = ds_results["1000k"]["delta_vs_baseline"]
        passes = (d5["net_usd"] > 0 and d1["net_usd"] > 0
                  and d5["profit_factor"] >= -0.005 and d1["profit_factor"] >= -0.005)
        tag = "PASS" if passes else "    "
        print(f"  [{tag}] {vname:<45}  "
              f"500k: dUSD={d5['net_usd']:>+10,.0f} dPF={d5['profit_factor']:>+.4f} dDD={d5['max_drawdown_usd']:>+8,.0f}  "
              f"1000k: dUSD={d1['net_usd']:>+10,.0f} dPF={d1['profit_factor']:>+.4f} dDD={d1['max_drawdown_usd']:>+8,.0f}")

        results_by_variant.append({
            "name": vname,
            "cell_scales": {CELLS[k]["label"]: v for k, v in scales.items()},
            "datasets": ds_results,
            "passes_strict": passes,
        })

    print()

    # Find best by combined net USD among passing variants
    passing = [v for v in results_by_variant if v["passes_strict"]]
    if passing:
        best = max(passing, key=lambda v: (
            v["datasets"]["500k"]["delta_vs_baseline"]["net_usd"]
            + v["datasets"]["1000k"]["delta_vs_baseline"]["net_usd"]
        ))
        print(f"Best passing variant: {best['name']}")
        for ds in ["500k", "1000k"]:
            d = best["datasets"][ds]["delta_vs_baseline"]
            s = best["datasets"][ds]["variant_summary"]
            print(f"  {ds}: trades={s['total_trades']}, net_usd={s['net_usd']:,.2f}, "
                  f"PF={s['profit_factor']:.4f}, DD={s['max_drawdown_usd']:,.2f}")
            print(f"    delta: +{d['total_trades']}t, USD {d['net_usd']:+,.2f}, "
                  f"PF {d['profit_factor']:+.4f}, DD {d['max_drawdown_usd']:+,.2f}")
    else:
        print("No variants passed strict filter.")

    # Write output
    payload = {
        "title": "Variant K + V44 family v2 — asymmetric sizing sweep",
        "system_name": "variant_k_plus_v44_family_v2",
        "cells": CELLS,
        "variants_tested": len(results_by_variant),
        "variants_passing": len(passing),
        "best_variant": best["name"] if passing else None,
        "results": results_by_variant,
    }
    output = Path(args.output)
    output.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    print(f"\nWrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
