#!/usr/bin/env python3
"""
Variant K + V44 family v3 — displacement-aware, sub-filtered, expanded.

Changes from v2:
  - Cell 1: Strong-only (kept from v2)
  - Cell 2 (ambig/er_low/der_pos sell): test Normal-only variant (0 displacement)
  - Cell 3: unchanged (all Strong already)
  - Cell 4 (ambig/er_mid/der_neg buy): test at reduced scale (high displacement on 500k)
  - Cell 5 NEW: ambig/er_mid/der_pos sell Strong-only (adjacent expansion, reduces 1000k disp 13->3)
  - Cell 6+7: post_breakout_trend sells (tiny but zero-displacement)

Named variants explore displacement-reduction and adjacent-cell expansion.
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
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
DEFAULT_OUTPUT = OUT_DIR / "system_variant_k_plus_v44_family_v3.json"
DEFAULT_MATRIX = OUT_DIR / "offensive_slice_discovery_matrix.json"

# ═══════════════════════════════════════════════════════════════════════
# Cell definitions
# ═══════════════════════════════════════════════════════════════════════

CELLS = [
    # --- v2 cells (indices 0-3) ---
    {
        "idx": 0,
        "label": "C0_ambig_high_pos_sell_strong",
        "slice_id": "v44_ny__short__cells_ambiguous_er_high_der_pos__entry_profiles_Strong",
        "cell": "ambiguous/er_high/der_pos",
        "note": "v2 cell 1, Strong-only",
    },
    {
        "idx": 1,
        "label": "C1_ambig_low_pos_sell_base",
        "slice_id": "v44_ny__short__cells_ambiguous_er_low_der_pos",
        "cell": "ambiguous/er_low/der_pos",
        "note": "v2 cell 2, base",
    },
    {
        "idx": 2,
        "label": "C2_mom_high_pos_sell",
        "slice_id": "v44_ny__short__cells_momentum_er_high_der_pos",
        "cell": "momentum/er_high/der_pos",
        "note": "v2 cell 3, all Strong",
    },
    {
        "idx": 3,
        "label": "C3_ambig_mid_neg_buy",
        "slice_id": "v44_ny__long__cells_ambiguous_er_mid_der_neg",
        "cell": "ambiguous/er_mid/der_neg",
        "note": "v2 cell 4, high disp on 500k (6d/7a)",
    },
    # --- New cells ---
    {
        "idx": 4,
        "label": "C4_ambig_mid_pos_sell_strong",
        "slice_id": "v44_ny__short__cells_ambiguous_er_mid_der_pos__entry_profiles_Strong",
        "cell": "ambiguous/er_mid/der_pos",
        "note": "NEW adjacent, Strong-only reduces 1000k disp 13->3",
    },
    {
        "idx": 5,
        "label": "C5_pbt_high_pos_sell",
        "slice_id": "v44_ny__short__cells_post_breakout_trend_er_high_der_pos",
        "cell": "post_breakout_trend/er_high/der_pos",
        "note": "NEW adjacent, 1t each ds, zero displacement",
    },
    {
        "idx": 6,
        "label": "C6_pbt_mid_pos_sell",
        "slice_id": "v44_ny__short__cells_post_breakout_trend_er_mid_der_pos",
        "cell": "post_breakout_trend/er_mid/der_pos",
        "note": "NEW adjacent, 1t each ds, zero displacement",
    },
    # --- Alternative sub-filter for cell 1 ---
    {
        "idx": 7,
        "label": "C1alt_ambig_low_pos_sell_normal",
        "slice_id": "v44_ny__short__cells_ambiguous_er_low_der_pos__entry_profiles_Normal",
        "cell": "ambiguous/er_low/der_pos",
        "note": "ALT: Normal-only, zero displacement",
    },
]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return discovery._json_default(obj)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Variant K + V44 family v3 — displacement-aware expansion")
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


def _run_variant(
    *,
    active_cells: list[int],
    cell_scales: dict[int, float],
    trades_per_cell: list[list[dict[str, Any]]],
    baseline_ctx: additive.BaselineContext,
) -> dict[str, Any]:
    combined: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    per_cell_counts: list[int] = [0] * len(CELLS)

    for cell_idx in active_cells:
        scale = cell_scales.get(cell_idx, 1.0)
        for trade in trades_per_cell[cell_idx]:
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
            per_cell_counts[cell_idx] += 1

    combined.sort(key=lambda t: (t["entry_time"], t["exit_time"]))
    result = additive.run_slice_additive(
        baseline_ctx=baseline_ctx,
        slice_spec={
            "variant": "v3",
            "active_cells": active_cells,
            "cell_scales": {str(k): v for k, v in cell_scales.items()},
        },
        selected_trades=combined,
        size_scale=1.0,
    )
    result["per_cell_selected_counts"] = per_cell_counts
    return result


def _build_variants() -> list[dict[str, Any]]:
    variants = []

    def _add(name: str, active: list[int], scales: dict[int, float] | None = None):
        variants.append({
            "name": name,
            "active_cells": active,
            "scales": scales or {i: 1.0 for i in active},
        })

    # --- Reference: v2 uniform 100% (cells 0-3) ---
    _add("v2_reference", [0, 1, 2, 3])

    # --- Cell 1 refinement: Normal-only (zero displacement) vs base ---
    _add("C1_normal_only", [0, 7, 2, 3])  # swap cell 1 base for cell 7 (Normal-only)

    # --- Cell 3 refinement: reduce scale on high-displacement buy ---
    _add("C3_at_50pct", [0, 1, 2, 3], {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.5})
    _add("v2_minus_C3", [0, 1, 2])  # drop Cell 3 entirely

    # --- Adjacent expansion: add Cell 4 (ambig/er_mid/der_pos sell Strong) ---
    _add("v2_plus_C4", [0, 1, 2, 3, 4])
    _add("v2_plus_C4_strong", [0, 1, 2, 3, 4])  # C4 is already Strong-only

    # --- Adjacent expansion: add post_breakout_trend sells (zero displacement) ---
    _add("v2_plus_pbt", [0, 1, 2, 3, 5, 6])

    # --- Full expansion: all 7 new cells ---
    _add("full_v3", [0, 1, 2, 3, 4, 5, 6])

    # --- Full with C1 Normal-only (displacement reduction) ---
    _add("full_v3_C1_normal", [0, 7, 2, 3, 4, 5, 6])

    # --- Full minus C3 (drop high-displacement buy) ---
    _add("full_v3_no_C3", [0, 1, 2, 4, 5, 6])

    # --- Full with C3 at 50% ---
    _add("full_v3_C3_half", [0, 1, 2, 3, 4, 5, 6], {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.5, 4: 1.0, 5: 1.0, 6: 1.0})

    # --- Best displacement profile: C1 Normal + no C3 + C4 Strong + pbt ---
    _add("low_displacement", [0, 7, 2, 4, 5, 6])

    # --- Sells-only family (drop the lone buy cell) ---
    _add("sells_only_expanded", [0, 1, 2, 4, 5, 6])

    return variants


def main() -> int:
    args = _parse_args()
    matrix = family_combo._load_matrix(DEFAULT_MATRIX)
    specs = _load_specs(matrix)
    strategies = {s.strategy for s in specs}
    all_trades = discovery._load_all_normalized_trades(strategies)
    trades_per_cell = _select_trades_per_cell(specs, all_trades)

    print("Cell definitions:")
    for i, cell_def in enumerate(CELLS):
        for ds in ["500k", "1000k"]:
            n = len(trades_per_cell[ds][i])
            print(f"  [{i}] {cell_def['label']:<45} {ds}: {n:>2}t  ({cell_def['note']})")
    print()

    variants = _build_variants()
    results: list[dict[str, Any]] = []

    print(f"Running {len(variants)} variants...")
    print()
    print(f"{'Variant':<35} {'500k dUSD':>10} {'dPF':>7} {'dDD':>8} {'Add':>4} {'Dsp':>4}  "
          f"{'1000k dUSD':>10} {'dPF':>7} {'dDD':>8} {'Add':>4} {'Dsp':>4}  {'CombUSD':>10}")
    print("-" * 130)

    for variant in variants:
        vname = variant["name"]
        active = variant["active_cells"]
        scales = variant["scales"]
        ds_results: dict[str, Any] = {}

        for dataset_key in ["500k", "1000k"]:
            baseline_ctx = additive.build_baseline_context(discovery.DATASETS[dataset_key])
            ds_results[dataset_key] = _run_variant(
                active_cells=active,
                cell_scales=scales,
                trades_per_cell=trades_per_cell[dataset_key],
                baseline_ctx=baseline_ctx,
            )

        d5 = ds_results["500k"]["delta_vs_baseline"]
        d1 = ds_results["1000k"]["delta_vs_baseline"]
        s5 = ds_results["500k"]["selection_counts"]
        s1 = ds_results["1000k"]["selection_counts"]
        comb = d5["net_usd"] + d1["net_usd"]
        passes = (d5["net_usd"] > 0 and d1["net_usd"] > 0
                  and d5["profit_factor"] >= -0.005 and d1["profit_factor"] >= -0.005)
        tag = "+" if passes else " "

        print(f"{tag}{vname:<34} {d5['net_usd']:>+10,.0f} {d5['profit_factor']:>+.4f} {d5['max_drawdown_usd']:>+8,.0f} "
              f"{s5['new_additive_trades_count']:>4} {s5['displaced_trades_count']:>4}  "
              f"{d1['net_usd']:>+10,.0f} {d1['profit_factor']:>+.4f} {d1['max_drawdown_usd']:>+8,.0f} "
              f"{s1['new_additive_trades_count']:>4} {s1['displaced_trades_count']:>4}  {comb:>+10,.0f}")

        results.append({
            "name": vname,
            "active_cells": active,
            "cell_labels": [CELLS[i]["label"] for i in active],
            "cell_scales": {CELLS[k]["label"]: v for k, v in scales.items()},
            "passes_strict": passes,
            "datasets": ds_results,
        })

    print()
    passing = [v for v in results if v["passes_strict"]]
    if passing:
        best = max(passing, key=lambda v: (
            v["datasets"]["500k"]["delta_vs_baseline"]["net_usd"]
            + v["datasets"]["1000k"]["delta_vs_baseline"]["net_usd"]
        ))
        print(f"Best passing: {best['name']} (combined USD: "
              f"{best['datasets']['500k']['delta_vs_baseline']['net_usd'] + best['datasets']['1000k']['delta_vs_baseline']['net_usd']:+,.0f})")

        # Also find best PF-per-displacement
        best_eff = max(passing, key=lambda v: (
            (v["datasets"]["500k"]["delta_vs_baseline"]["profit_factor"]
             + v["datasets"]["1000k"]["delta_vs_baseline"]["profit_factor"])
            / max(1, v["datasets"]["500k"]["selection_counts"]["displaced_trades_count"]
                  + v["datasets"]["1000k"]["selection_counts"]["displaced_trades_count"])
        ))
        print(f"Best PF efficiency: {best_eff['name']}")

    payload = {
        "title": "Variant K + V44 family v3 — displacement-aware expansion",
        "system_name": "variant_k_plus_v44_family_v3",
        "cells": CELLS,
        "variants_tested": len(results),
        "variants_passing": len(passing),
        "best_variant": best["name"] if passing else None,
        "results": results,
    }
    output = Path(args.output)
    output.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    print(f"\nWrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
