#!/usr/bin/env python3
"""
V6 family search — expand beyond V44-only into multi-archetype offensive family.

v5 frontier (hedging policy):
  v5_best_current_framework (v4_plus_all_adj): combined delta +$128,472
  v4_best: combined delta +$127,109

v6 search vectors:
  1. New V44 cells: breakout/er_low/der_neg sell Strong (+$12,441 standalone)
  2. London V2 cells: momentum long (+$9,890), breakout long (+$6,616) — no V44 conflict
  3. V14 Tokyo cells: ambig long (+$2,614), breakout long (+$2,025), ambig sell (+$1,952)
  4. Small V44 breakout/pbt cells

Uses the hedging policy (internal overlap allowed, identity-based baseline match).
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
DEFAULT_OUTPUT = OUT_DIR / "system_variant_k_v6_search.json"
DEFAULT_MATRIX = OUT_DIR / "offensive_slice_discovery_matrix.json"

# ═══════════════════════════════════════════════════════════════════════
# Cell registry — v5 base + v6 expansion candidates
# ═══════════════════════════════════════════════════════════════════════

CELL_REGISTRY = {
    # --- v4/v5 baseline cells (13 cells) ---
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
    "ADJ_meanrev_low_neg_buy": "v44_ny__long__cells_mean_reversion_er_low_der_neg",
    "ADJ_ambig_mid_neg_sell": "v44_ny__short__cells_ambiguous_er_mid_der_neg",
    "ADJ_mom_high_neg_sell": "v44_ny__short__cells_momentum_er_high_der_neg",

    # --- v6 NEW: V44 breakout cells ---
    "N1_brkout_low_neg_sell_strong": "v44_ny__short__cells_breakout_er_low_der_neg__entry_profiles_Strong",
    "N2_brkout_low_pos_buy_strong": "v44_ny__long__cells_breakout_er_low_der_pos__entry_profiles_Strong",
    "N3_brkout_low_neg_buy_news": "v44_ny__long__cells_breakout_er_low_der_neg__entry_signal_modes_news_trend_confirm",
    "N4_pbt_low_neg_buy_news": "v44_ny__long__cells_post_breakout_trend_er_low_der_neg__entry_profiles_news_trend",

    # --- v6 NEW: London V2 cells (different strategy — no V44 conflict) ---
    "L1_mom_low_pos_buy": "london_v2__setup_D__long__first_30min__london__cells_momentum_er_low_der_pos__native_allowed_True",
    "L2_brkout_mid_neg_buy": "london_v2__setup_D__long__london__cells_breakout_er_mid_der_neg__native_allowed_True",

    # --- v6 NEW: V14 Tokyo cells (different strategy — no V44 conflict) ---
    "T1_ambig_high_pos_buy": "v14__long__tokyo__cells_ambiguous_er_high_der_pos",
    "T2_brkout_mid_pos_buy": "v14__long__tokyo__cells_breakout_er_mid_der_pos",
    "T3_ambig_mid_pos_sell": "v14__short__tokyo__cells_ambiguous_er_mid_der_pos",
}

V5_BEST = [
    "C0_sell_strong", "C1_sell_base", "C2_sell", "C3_buy",
    "C4_sell_base", "C5_pbt_sell", "C6_pbt_sell",
    "O0_buy_strong", "O1_buy_strong", "O2_buy_strong",
    "ADJ_meanrev_low_neg_buy", "ADJ_ambig_mid_neg_sell", "ADJ_mom_high_neg_sell",
]

V4_BEST = [
    "C0_sell_strong", "C1_sell_base", "C2_sell", "C3_buy",
    "C4_sell_base", "C5_pbt_sell", "C6_pbt_sell",
    "O0_buy_strong", "O1_buy_strong", "O2_buy_strong",
]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return discovery._json_default(obj)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V6 multi-archetype family search")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return p.parse_args()


def _load_all_specs(matrix: dict[str, Any]) -> dict[str, OffensiveSliceSpec]:
    specs = {}
    for label, sid in CELL_REGISTRY.items():
        row = matrix.get("results", {}).get(sid)
        if not row:
            # Try per_dataset structure
            pd_data = row if row else {}
            if not pd_data:
                # Search in top-level results
                for top_sid, top_row in matrix.get("results", {}).items():
                    if top_sid == sid:
                        pd_data = top_row
                        break
        if not row:
            print(f"  WARNING: {sid} not in matrix, skipping {label}")
            continue
        # Get slice_spec from either top-level or per_dataset
        slice_spec = row.get("slice_spec")
        if not slice_spec:
            slice_spec = row.get("per_dataset", {}).get("500k", {}).get("additive", {}).get("slice_spec")
        if not slice_spec:
            print(f"  WARNING: no slice_spec for {sid}, skipping {label}")
            continue
        specs[label] = OffensiveSliceSpec(**slice_spec)
    return specs


def _select_all_trades(
    specs: dict[str, OffensiveSliceSpec],
    all_trades: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    out: dict[str, dict[str, list[dict[str, Any]]]] = {"500k": {}, "1000k": {}}
    for dataset_key in ["500k", "1000k"]:
        for label, spec in specs.items():
            out[dataset_key][label] = [
                t for t in all_trades[dataset_key].get(spec.strategy, [])
                if discovery._passes_filters(t, spec)
            ]
    return out


def _collect_family_trades(
    cell_labels: list[str],
    trades_by_label: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    combined: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for label in cell_labels:
        for trade in trades_by_label.get(label, []):
            key = (str(trade["strategy"]), str(trade["entry_time"]),
                   str(trade["exit_time"]), str(trade["side"]))
            if key in seen:
                continue
            seen.add(key)
            combined.append(trade)
    combined.sort(key=lambda t: (t["entry_time"], t["exit_time"]))
    return combined


def _run_variant(
    name: str,
    cells: list[str],
    trades_by_ds: dict[str, dict[str, list[dict[str, Any]]]],
    policy: additive.ConflictPolicy,
) -> dict[str, Any]:
    ds_results: dict[str, Any] = {}
    for dataset_key in ["500k", "1000k"]:
        baseline_ctx = additive.build_baseline_context(discovery.DATASETS[dataset_key])
        combined = _collect_family_trades(cells, trades_by_ds[dataset_key])
        result = additive.run_slice_additive_with_policy(
            baseline_ctx=baseline_ctx,
            slice_spec={"variant": name, "cells": cells},
            selected_trades=combined,
            conflict_policy=policy,
            size_scale=1.0,
        )
        ds_results[dataset_key] = result
    return {"name": name, "cells": cells, "datasets": ds_results}


def _build_variants() -> list[dict[str, Any]]:
    variants = []

    def _add(name: str, cells: list[str]):
        variants.append({"name": name, "cells": cells})

    # === Reference baselines ===
    _add("v4_best", V4_BEST)
    _add("v5_best", V5_BEST)

    # === Vector 1: Add new V44 breakout cells individually ===
    _add("v5+N1_brkout_sell", V5_BEST + ["N1_brkout_low_neg_sell_strong"])
    _add("v5+N2_brkout_buy", V5_BEST + ["N2_brkout_low_pos_buy_strong"])
    _add("v5+N1+N2_brkout", V5_BEST + ["N1_brkout_low_neg_sell_strong", "N2_brkout_low_pos_buy_strong"])
    _add("v5+N3_news_buy", V5_BEST + ["N3_brkout_low_neg_buy_news"])
    _add("v5+N4_pbt_news", V5_BEST + ["N4_pbt_low_neg_buy_news"])

    # === Vector 2: Add London V2 cells (no V44 conflict) ===
    _add("v5+L1_london_mom", V5_BEST + ["L1_mom_low_pos_buy"])
    _add("v5+L2_london_brk", V5_BEST + ["L2_brkout_mid_neg_buy"])
    _add("v5+L1+L2_london", V5_BEST + ["L1_mom_low_pos_buy", "L2_brkout_mid_neg_buy"])

    # === Vector 3: Add V14 Tokyo cells (no V44 conflict) ===
    _add("v5+T1_v14_buy", V5_BEST + ["T1_ambig_high_pos_buy"])
    _add("v5+T2_v14_brk", V5_BEST + ["T2_brkout_mid_pos_buy"])
    _add("v5+T3_v14_sell", V5_BEST + ["T3_ambig_mid_pos_sell"])
    _add("v5+T1+T2+T3_v14", V5_BEST + ["T1_ambig_high_pos_buy", "T2_brkout_mid_pos_buy", "T3_ambig_mid_pos_sell"])

    # === Combined: best from each vector ===
    # V44 breakout + London
    _add("v5+N1+L1+L2", V5_BEST + ["N1_brkout_low_neg_sell_strong", "L1_mom_low_pos_buy", "L2_brkout_mid_neg_buy"])

    # V44 breakout + V14
    _add("v5+N1+T1+T2+T3", V5_BEST + ["N1_brkout_low_neg_sell_strong",
         "T1_ambig_high_pos_buy", "T2_brkout_mid_pos_buy", "T3_ambig_mid_pos_sell"])

    # London + V14
    _add("v5+L1+L2+T1+T2+T3", V5_BEST + ["L1_mom_low_pos_buy", "L2_brkout_mid_neg_buy",
         "T1_ambig_high_pos_buy", "T2_brkout_mid_pos_buy", "T3_ambig_mid_pos_sell"])

    # Full expansion: all three vectors
    _add("v6_full_expansion", V5_BEST + [
        "N1_brkout_low_neg_sell_strong", "N2_brkout_low_pos_buy_strong",
        "L1_mom_low_pos_buy", "L2_brkout_mid_neg_buy",
        "T1_ambig_high_pos_buy", "T2_brkout_mid_pos_buy", "T3_ambig_mid_pos_sell",
    ])

    # Kitchen sink: every new cell
    _add("v6_kitchen_sink", V5_BEST + [
        "N1_brkout_low_neg_sell_strong", "N2_brkout_low_pos_buy_strong",
        "N3_brkout_low_neg_buy_news", "N4_pbt_low_neg_buy_news",
        "L1_mom_low_pos_buy", "L2_brkout_mid_neg_buy",
        "T1_ambig_high_pos_buy", "T2_brkout_mid_pos_buy", "T3_ambig_mid_pos_sell",
    ])

    # === v4 base + best new cells (skip v5 adjacents, go straight to v6) ===
    _add("v4+N1+L1+L2", V4_BEST + ["N1_brkout_low_neg_sell_strong", "L1_mom_low_pos_buy", "L2_brkout_mid_neg_buy"])
    _add("v4+all_new", V4_BEST + [
        "N1_brkout_low_neg_sell_strong", "N2_brkout_low_pos_buy_strong",
        "L1_mom_low_pos_buy", "L2_brkout_mid_neg_buy",
        "T1_ambig_high_pos_buy", "T2_brkout_mid_pos_buy", "T3_ambig_mid_pos_sell",
    ])

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
        strat = specs[label].strategy
        for ds in ["500k", "1000k"]:
            n = len(trades_by_ds[ds].get(label, []))
            if n > 0:
                print(f"  {label:<40} {strat:<12} {ds}: {n:>2}t")
    print()

    policy = additive.ConflictPolicy(
        name="native_v44_hedging_like",
        hedging_enabled=True,
        allow_internal_overlap=True,
        allow_opposite_side_overlap=True,
        max_open_offensive=None,
        max_entries_per_day=None,
    )

    variants = _build_variants()
    results: list[dict[str, Any]] = []

    print(f"Running {len(variants)} variants under hedging policy...")
    print()
    hdr = (f"{'Variant':<30} {'500k dUSD':>10} {'dPF':>7} {'dDD':>8} {'Add':>4}  "
           f"{'1000k dUSD':>10} {'dPF':>7} {'dDD':>8} {'Add':>4}  {'CombUSD':>12} {'CombPF':>8}")
    print(hdr)
    print("-" * len(hdr))

    for variant in variants:
        vname = variant["name"]
        cells = variant["cells"]
        vresult = _run_variant(vname, cells, trades_by_ds, policy)

        d5 = vresult["datasets"]["500k"]["delta_vs_baseline"]
        d1 = vresult["datasets"]["1000k"]["delta_vs_baseline"]
        s5 = vresult["datasets"]["500k"]["selection_counts"]
        s1 = vresult["datasets"]["1000k"]["selection_counts"]
        comb_usd = d5["net_usd"] + d1["net_usd"]
        comb_pf = d5["profit_factor"] + d1["profit_factor"]
        passes = (d5["net_usd"] > 0 and d1["net_usd"] > 0
                  and d5["profit_factor"] >= -0.005 and d1["profit_factor"] >= -0.005)
        tag = "+" if passes else " "

        print(f"{tag}{vname:<29} {d5['net_usd']:>+10,.0f} {d5['profit_factor']:>+.4f} {d5['max_drawdown_usd']:>+8,.0f} {s5['new_additive_trades_count']:>4}  "
              f"{d1['net_usd']:>+10,.0f} {d1['profit_factor']:>+.4f} {d1['max_drawdown_usd']:>+8,.0f} {s1['new_additive_trades_count']:>4}  "
              f"{comb_usd:>+12,.0f} {comb_pf:>+8.4f}")

        results.append({
            "name": vname,
            "cells": cells,
            "cell_count": len(cells),
            "passes_strict": passes,
            "combined_usd": round(comb_usd, 2),
            "combined_pf": round(comb_pf, 4),
            "datasets": {
                ds: {
                    "delta": vresult["datasets"][ds]["delta_vs_baseline"],
                    "summary": vresult["datasets"][ds]["variant_summary"],
                    "selection": vresult["datasets"][ds]["selection_counts"],
                }
                for ds in ["500k", "1000k"]
            },
        })

    print()

    # Rank passing variants
    passing = sorted([v for v in results if v["passes_strict"]], key=lambda v: -v["combined_usd"])
    v5_comb = next((v["combined_usd"] for v in results if v["name"] == "v5_best"), 0)

    print("=== PASSING VARIANTS RANKED BY COMBINED USD ===")
    print()
    print(f"{'#':<4} {'Variant':<30} {'CombUSD':>12} {'Δ v5':>10} {'CombPF':>8} {'Cells':>5}")
    print("-" * 75)
    for i, v in enumerate(passing):
        delta_v5 = v["combined_usd"] - v5_comb
        print(f"  #{i+1:<2} {v['name']:<30} {v['combined_usd']:>+12,.0f} {delta_v5:>+10,.0f} {v['combined_pf']:>+8.4f} {v['cell_count']:>5}")

    payload = {
        "title": "V6 multi-archetype family search",
        "policy": "native_v44_hedging_like (internal overlap + opposite-side allowed)",
        "v5_reference_combined_usd": v5_comb,
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
