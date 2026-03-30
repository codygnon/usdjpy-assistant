#!/usr/bin/env python3
"""
V5 family search — disciplined refinement of the v4 winner.

Search strategy:
  1. Internal refinement: profile swaps, cell removals, displacement reduction
  2. Adjacent expansion: only cells passing strict individually
  3. Combined: best internal + best adjacent

v4 baseline (10 cells):
  C0: ambiguous/er_high/der_pos | sell | Strong
  C1: ambiguous/er_low/der_pos | sell | base
  C2: momentum/er_high/der_pos | sell
  C3: ambiguous/er_mid/der_neg | buy
  C4: ambiguous/er_mid/der_pos | sell | base
  C5: post_breakout_trend/er_high/der_pos | sell
  C6: post_breakout_trend/er_mid/der_pos | sell
  O0: ambiguous/er_high/der_pos | buy | Strong
  O1: ambiguous/er_low/der_pos | buy | Strong
  O2: ambiguous/er_mid/der_pos | buy | Strong
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
DEFAULT_OUTPUT = OUT_DIR / "system_variant_k_v5_search.json"
DEFAULT_MATRIX = OUT_DIR / "offensive_slice_discovery_matrix.json"

# ═══════════════════════════════════════════════════════════════════════
# Cell registry — all cells available for v5 search
# ═══════════════════════════════════════════════════════════════════════

CELL_REGISTRY = {
    # --- v4 baseline cells ---
    "C0_sell_strong": "v44_ny__short__cells_ambiguous_er_high_der_pos__entry_profiles_Strong",
    "C1_sell_base": "v44_ny__short__cells_ambiguous_er_low_der_pos",
    "C2_sell": "v44_ny__short__cells_momentum_er_high_der_pos",
    "C3_buy": "v44_ny__long__cells_ambiguous_er_mid_der_neg",
    "C4_sell_base": "v44_ny__short__cells_ambiguous_er_mid_der_pos",
    "C5_pbt_sell": "v44_ny__short__cells_post_breakout_trend_er_high_der_pos",
    "C6_pbt_sell": "v44_ny__short__cells_post_breakout_trend_er_mid_der_pos",
    "O0_buy_strong": "v44_ny__long__cells_ambiguous_er_high_der_pos__entry_profiles_Strong",
    "O1_buy_strong": "v44_ny__long__cells_ambiguous_er_low_der_pos__entry_profiles_Strong",
    "O2_buy_strong": "v44_ny__long__cells_ambiguous_er_mid_der_pos__entry_profiles_Strong",

    # --- Alternative sub-filters for existing cells ---
    "C0_sell_base": "v44_ny__short__cells_ambiguous_er_high_der_pos",
    "C1_sell_strong": "v44_ny__short__cells_ambiguous_er_low_der_pos__entry_profiles_Strong",
    "C1_sell_normal": "v44_ny__short__cells_ambiguous_er_low_der_pos__entry_profiles_Normal",
    "C4_sell_strong": "v44_ny__short__cells_ambiguous_er_mid_der_pos__entry_profiles_Strong",
    "O1_buy_base": "v44_ny__long__cells_ambiguous_er_low_der_pos",

    # --- Adjacent expansion candidates (pass strict individually) ---
    "ADJ_meanrev_low_neg_buy": "v44_ny__long__cells_mean_reversion_er_low_der_neg",
    "ADJ_ambig_mid_neg_sell": "v44_ny__short__cells_ambiguous_er_mid_der_neg",
    "ADJ_mom_high_neg_sell": "v44_ny__short__cells_momentum_er_high_der_neg",
}

V4_BASELINE = [
    "C0_sell_strong", "C1_sell_base", "C2_sell", "C3_buy",
    "C4_sell_base", "C5_pbt_sell", "C6_pbt_sell",
    "O0_buy_strong", "O1_buy_strong", "O2_buy_strong",
]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return discovery._json_default(obj)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V5 family search")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return p.parse_args()


def _load_all_specs(matrix: dict[str, Any]) -> dict[str, OffensiveSliceSpec]:
    specs = {}
    for label, sid in CELL_REGISTRY.items():
        row = (matrix.get("results") or {}).get(sid)
        if not row:
            print(f"  WARNING: {sid} not in matrix, skipping {label}")
            continue
        specs[label] = OffensiveSliceSpec(**row["slice_spec"])
    return specs


def _select_all_trades(
    specs: dict[str, OffensiveSliceSpec],
    all_trades: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Per dataset, per cell label -> selected trades."""
    out: dict[str, dict[str, list[dict[str, Any]]]] = {"500k": {}, "1000k": {}}
    for dataset_key in ["500k", "1000k"]:
        for label, spec in specs.items():
            out[dataset_key][label] = [
                t for t in all_trades[dataset_key].get(spec.strategy, [])
                if discovery._passes_filters(t, spec)
            ]
    return out


def _run_family(
    *,
    cell_labels: list[str],
    cell_scales: dict[str, float] | None,
    trades_by_label: dict[str, list[dict[str, Any]]],
    baseline_ctx: additive.BaselineContext,
) -> dict[str, Any]:
    scales = cell_scales or {}
    combined: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    per_cell: dict[str, int] = {}

    for label in cell_labels:
        scale = scales.get(label, 1.0)
        count = 0
        for trade in trades_by_label.get(label, []):
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
        per_cell[label] = count

    combined.sort(key=lambda t: (t["entry_time"], t["exit_time"]))
    result = additive.run_slice_additive(
        baseline_ctx=baseline_ctx,
        slice_spec={"variant": "v5_search", "cells": cell_labels},
        selected_trades=combined,
        size_scale=1.0,
    )
    result["per_cell_counts"] = per_cell
    return result


def _build_variants() -> list[dict[str, Any]]:
    variants = []

    def _add(name: str, cells: list[str], scales: dict[str, float] | None = None):
        variants.append({"name": name, "cells": cells, "scales": scales})

    # === v4 baseline ===
    _add("v4_baseline", V4_BASELINE)

    # === 1. Internal refinement ===

    # C4 Strong (reduces 1000k displacement 13->3)
    _add("v4_C4_strong", [c if c != "C4_sell_base" else "C4_sell_strong" for c in V4_BASELINE])

    # C0 base (adds news_trend trades)
    _add("v4_C0_base", [c if c != "C0_sell_strong" else "C0_sell_base" for c in V4_BASELINE])

    # Drop C3 (high displacement buy: 6d/7a on 500k)
    _add("v4_no_C3", [c for c in V4_BASELINE if c != "C3_buy"])

    # Drop C2 (smallest contributor: +$740/+$737)
    _add("v4_no_C2", [c for c in V4_BASELINE if c != "C2_sell"])

    # C1 Strong (fewer trades, less DD)
    _add("v4_C1_strong", [c if c != "C1_sell_base" else "C1_sell_strong" for c in V4_BASELINE])

    # C1 Normal (zero displacement)
    _add("v4_C1_normal", [c if c != "C1_sell_base" else "C1_sell_normal" for c in V4_BASELINE])

    # O1 base (more trades but more displacement)
    _add("v4_O1_base", [c if c != "O1_buy_strong" else "O1_buy_base" for c in V4_BASELINE])

    # Combined: C4 Strong + no C3 (dual displacement reduction)
    _add("v4_C4strong_noC3", [c if c != "C4_sell_base" else "C4_sell_strong"
                               for c in V4_BASELINE if c != "C3_buy"])

    # Combined: C1 Strong + C4 Strong (both tighter)
    _add("v4_C1strong_C4strong", [
        "C4_sell_strong" if c == "C4_sell_base" else
        "C1_sell_strong" if c == "C1_sell_base" else c
        for c in V4_BASELINE
    ])

    # === 2. Adjacent expansion ===

    # Add each adjacent individually
    _add("v4_plus_meanrev", V4_BASELINE + ["ADJ_meanrev_low_neg_buy"])
    _add("v4_plus_ambig_mid_neg_sell", V4_BASELINE + ["ADJ_ambig_mid_neg_sell"])
    _add("v4_plus_mom_high_neg_sell", V4_BASELINE + ["ADJ_mom_high_neg_sell"])

    # All three adjacents
    _add("v4_plus_all_adj", V4_BASELINE + ["ADJ_meanrev_low_neg_buy", "ADJ_ambig_mid_neg_sell", "ADJ_mom_high_neg_sell"])

    # === 3. Combined best-guess ===

    # C4 Strong + all adjacents
    _add("v5_candidate_A", [c if c != "C4_sell_base" else "C4_sell_strong" for c in V4_BASELINE]
         + ["ADJ_meanrev_low_neg_buy", "ADJ_ambig_mid_neg_sell", "ADJ_mom_high_neg_sell"])

    # C4 Strong + no C3 + all adjacents
    no_c3 = [c if c != "C4_sell_base" else "C4_sell_strong" for c in V4_BASELINE if c != "C3_buy"]
    _add("v5_candidate_B", no_c3 + ["ADJ_meanrev_low_neg_buy", "ADJ_ambig_mid_neg_sell", "ADJ_mom_high_neg_sell"])

    # C1 Strong + C4 Strong + adjacents
    _add("v5_candidate_C", [
        "C4_sell_strong" if c == "C4_sell_base" else
        "C1_sell_strong" if c == "C1_sell_base" else c
        for c in V4_BASELINE
    ] + ["ADJ_ambig_mid_neg_sell", "ADJ_mom_high_neg_sell"])

    # C3 at 50% scale
    _add("v4_C3_half", V4_BASELINE, {"C3_buy": 0.5})

    # v4 + adj, C3 at 50%
    _add("v5_candidate_D", V4_BASELINE + ["ADJ_ambig_mid_neg_sell", "ADJ_mom_high_neg_sell"],
         {"C3_buy": 0.5})

    return variants


def main() -> int:
    args = _parse_args()
    matrix = family_combo._load_matrix(DEFAULT_MATRIX)
    specs = _load_all_specs(matrix)
    strategies = {s.strategy for s in specs.values()}
    all_trades = discovery._load_all_normalized_trades(strategies)
    trades_by_ds = _select_all_trades(specs, all_trades)

    # Print cell trade counts
    print("Cell registry:")
    for label in sorted(CELL_REGISTRY.keys()):
        if label not in specs:
            continue
        for ds in ["500k", "1000k"]:
            n = len(trades_by_ds[ds].get(label, []))
            if n > 0:
                print(f"  {label:<35} {ds}: {n:>2}t")
    print()

    variants = _build_variants()
    results: list[dict[str, Any]] = []

    print(f"Running {len(variants)} variants...")
    print()
    hdr = (f"{'Variant':<35} {'500k dUSD':>10} {'dPF':>7} {'dDD':>8} {'Add':>4} {'Dsp':>4}  "
           f"{'1000k dUSD':>10} {'dPF':>7} {'dDD':>8} {'Add':>4} {'Dsp':>4}  {'CombUSD':>10}")
    print(hdr)
    print("-" * len(hdr))

    for variant in variants:
        vname = variant["name"]
        cells = variant["cells"]
        scales = variant.get("scales")
        ds_results: dict[str, Any] = {}

        for dataset_key in ["500k", "1000k"]:
            baseline_ctx = additive.build_baseline_context(discovery.DATASETS[dataset_key])
            ds_results[dataset_key] = _run_family(
                cell_labels=cells,
                cell_scales=scales,
                trades_by_label=trades_by_ds[dataset_key],
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
            "cells": cells,
            "cell_scales": scales,
            "passes_strict": passes,
            "datasets": ds_results,
            "combined_usd": round(comb, 2),
        })

    print()

    # Rank passing variants
    passing = sorted([v for v in results if v["passes_strict"]],
                     key=lambda v: -v["combined_usd"])

    print("=== PASSING VARIANTS RANKED BY COMBINED USD ===")
    print()
    v4_comb = next((v["combined_usd"] for v in results if v["name"] == "v4_baseline"), 0)
    for i, v in enumerate(passing):
        delta_vs_v4 = v["combined_usd"] - v4_comb
        d5 = v["datasets"]["500k"]
        d1 = v["datasets"]["1000k"]
        tot_disp = (d5["selection_counts"]["displaced_trades_count"]
                    + d1["selection_counts"]["displaced_trades_count"])
        print(f"  #{i+1} {v['name']:<35} comb={v['combined_usd']:>+10,.0f}  "
              f"Δv4={delta_vs_v4:>+8,.0f}  disp={tot_disp:>3}")

    payload = {
        "title": "V5 family search — refinement of v4 winner",
        "v4_baseline_combined_usd": v4_comb,
        "variants_tested": len(results),
        "variants_passing": len(passing),
        "results": results,
    }
    output = Path(args.output)
    output.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    print(f"\nWrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
