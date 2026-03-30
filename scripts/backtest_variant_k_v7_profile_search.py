#!/usr/bin/env python3
"""
V7 profile-swap search — mine alternative entry profiles for existing v6 cells.

V6 tested ONE profile per ownership cell. The discovery matrix has multiple
profile variants (base, Strong, pullback, Normal, news) per cell. V7 tests:
  1. Profile swaps: replace v6 cells with better-performing profile variants
  2. Profile additions: add alternative profiles alongside v6 cells (dedup catches exact overlap)
  3. New cells: genuinely new ownership cells not in v6

Uses strict validation policy (hedging + margin) matching v6 validation.
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

from core.offensive_slice_spec import OffensiveSliceSpec
from scripts import backtest_offensive_slice_additive as additive
from scripts import backtest_offensive_slice_family_combo as family_combo
from scripts import run_offensive_slice_discovery as discovery

OUT_DIR = ROOT / "research_out"
DEFAULT_OUTPUT = OUT_DIR / "system_variant_k_v7_profile_search.json"
DEFAULT_MATRIX = OUT_DIR / "offensive_slice_discovery_matrix.json"

# ═══════════════════════════════════════════════════════════════════════
# V6 validated baseline (v6_validated_no_singletons = 18 cells)
# ═══════════════════════════════════════════════════════════════════════

V6_REGISTRY = {
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
    "N1_brkout_low_neg_sell_strong": "v44_ny__short__cells_breakout_er_low_der_neg__entry_profiles_Strong",
    "N2_brkout_low_pos_buy_strong": "v44_ny__long__cells_breakout_er_low_der_pos__entry_profiles_Strong",
    "L2_brkout_mid_neg_buy": "london_v2__setup_D__long__london__cells_breakout_er_mid_der_neg__native_allowed_True",
    "T1_ambig_high_pos_buy": "v14__long__tokyo__cells_ambiguous_er_high_der_pos",
    "T2_brkout_mid_pos_buy": "v14__long__tokyo__cells_breakout_er_mid_der_pos",
}

V6_VALIDATED = list(V6_REGISTRY.keys())

# ═══════════════════════════════════════════════════════════════════════
# V7 candidate cells — profile alternatives + genuinely new cells
# ═══════════════════════════════════════════════════════════════════════

V7_CANDIDATES = {
    # --- C0 alternatives (v6 = Strong) ---
    "C0_base": "v44_ny__short__cells_ambiguous_er_high_der_pos",
    "C0_pullback": "v44_ny__short__cells_ambiguous_er_high_der_pos__entry_signal_modes_pullback",
    # --- C1 alternatives (v6 = base) ---
    "C1_pullback": "v44_ny__short__cells_ambiguous_er_low_der_pos__entry_signal_modes_pullback",
    "C1_strong": "v44_ny__short__cells_ambiguous_er_low_der_pos__entry_profiles_Strong",
    "C1_normal": "v44_ny__short__cells_ambiguous_er_low_der_pos__entry_profiles_Normal",
    # --- C3 alternatives (v6 = base) ---
    "C3_pullback": "v44_ny__long__cells_ambiguous_er_mid_der_neg__entry_signal_modes_pullback",
    "C3_strong": "v44_ny__long__cells_ambiguous_er_mid_der_neg__entry_profiles_Strong",
    # --- C4 alternatives (v6 = base) ---
    "C4_pullback": "v44_ny__short__cells_ambiguous_er_mid_der_pos__entry_signal_modes_pullback",
    "C4_strong": "v44_ny__short__cells_ambiguous_er_mid_der_pos__entry_profiles_Strong",
    # --- O0 alternatives (v6 = Strong) ---
    "O0_base": "v44_ny__long__cells_ambiguous_er_high_der_pos",
    "O0_pullback": "v44_ny__long__cells_ambiguous_er_high_der_pos__entry_signal_modes_pullback",
    # --- O1 alternatives (v6 = Strong) ---
    "O1_base": "v44_ny__long__cells_ambiguous_er_low_der_pos",
    "O1_pullback": "v44_ny__long__cells_ambiguous_er_low_der_pos__entry_signal_modes_pullback",
    # --- O2 alternatives (v6 = Strong) ---
    "O2_base": "v44_ny__long__cells_ambiguous_er_mid_der_pos",
    "O2_pullback": "v44_ny__long__cells_ambiguous_er_mid_der_pos__entry_signal_modes_pullback",
    # --- C2 alternatives (v6 = base) ---
    "C2_pullback": "v44_ny__short__cells_momentum_er_high_der_pos__entry_signal_modes_pullback",
    "C2_strong": "v44_ny__short__cells_momentum_er_high_der_pos__entry_profiles_Strong",
    # --- ADJ_mom alternatives (v6 = base) ---
    "ADJ_mom_pullback": "v44_ny__short__cells_momentum_er_high_der_neg__entry_signal_modes_pullback",
    "ADJ_mom_strong": "v44_ny__short__cells_momentum_er_high_der_neg__entry_profiles_Strong",

    # --- Genuinely new cells ---
    "NEW_v14_ambig_low_pos_sell": "v14__short__tokyo__cells_ambiguous_er_low_der_pos",
    "NEW_v14_ambig_mid_pos_buy": "v14__long__tokyo__cells_ambiguous_er_mid_der_pos",
    "NEW_v44_ambig_low_neg_sell_normal": "v44_ny__short__cells_ambiguous_er_low_der_neg__entry_profiles_Normal",
}

# Map each swap candidate to the v6 cell it replaces
SWAP_MAP = {
    "C0_base": "C0_sell_strong",
    "C0_pullback": "C0_sell_strong",
    "C1_pullback": "C1_sell_base",
    "C1_strong": "C1_sell_base",
    "C1_normal": "C1_sell_base",
    "C3_pullback": "C3_buy",
    "C3_strong": "C3_buy",
    "C4_pullback": "C4_sell_base",
    "C4_strong": "C4_sell_base",
    "O0_base": "O0_buy_strong",
    "O0_pullback": "O0_buy_strong",
    "O1_base": "O1_buy_strong",
    "O1_pullback": "O1_buy_strong",
    "O2_base": "O2_buy_strong",
    "O2_pullback": "O2_buy_strong",
    "C2_pullback": "C2_sell",
    "C2_strong": "C2_sell",
    "ADJ_mom_pullback": "ADJ_mom_high_neg_sell",
    "ADJ_mom_strong": "ADJ_mom_high_neg_sell",
}

ALL_REGISTRY = {**V6_REGISTRY, **V7_CANDIDATES}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return discovery._json_default(obj)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V7 profile-swap family search")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    p.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    return p.parse_args()


def _load_all_specs(matrix: dict[str, Any]) -> dict[str, OffensiveSliceSpec]:
    specs = {}
    for label, sid in ALL_REGISTRY.items():
        row = matrix.get("results", {}).get(sid)
        if not row:
            print(f"  WARNING: {sid} not in matrix, skipping {label}")
            continue
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
    baseline_ctx_by_ds: dict[str, additive.BaselineContext],
) -> dict[str, Any]:
    ds_results: dict[str, Any] = {}
    for dataset_key in ["500k", "1000k"]:
        combined = _collect_family_trades(cells, trades_by_ds[dataset_key])
        result = additive.run_slice_additive_with_policy(
            baseline_ctx=baseline_ctx_by_ds[dataset_key],
            slice_spec={"variant": name, "cells": cells},
            selected_trades=combined,
            conflict_policy=policy,
            size_scale=1.0,
        )
        ds_results[dataset_key] = result
    return {"name": name, "cells": cells, "datasets": ds_results}


def _build_variants(available_labels: set[str]) -> list[dict[str, Any]]:
    variants = []

    def _add(name: str, cells: list[str]):
        # Only add if all cells have specs
        if all(c in available_labels for c in cells):
            variants.append({"name": name, "cells": cells})

    # === Reference: v6 validated baseline ===
    _add("v6_validated", V6_VALIDATED)

    # === VECTOR 1: PROFILE SWAPS ===
    for v7_label, v6_target in SWAP_MAP.items():
        swapped = [c for c in V6_VALIDATED if c != v6_target] + [v7_label]
        _add(f"swap_{v7_label}", swapped)

    # === VECTOR 2: PROFILE ADDITIONS ===
    for v7_label, v6_target in SWAP_MAP.items():
        added = V6_VALIDATED + [v7_label]
        _add(f"add_{v7_label}", added)

    # === VECTOR 3: NEW CELLS ===
    new_cells = [l for l in V7_CANDIDATES if l.startswith("NEW_")]
    for nc in new_cells:
        if nc in available_labels:
            _add(f"add_{nc}", V6_VALIDATED + [nc])

    # === VECTOR 4: BEST COMBOS ===

    # All Strong→base swaps (widen trade coverage)
    strong_to_base = list(V6_VALIDATED)
    for v7, v6 in [("C0_base", "C0_sell_strong"), ("O0_base", "O0_buy_strong"),
                    ("O1_base", "O1_buy_strong"), ("O2_base", "O2_buy_strong")]:
        strong_to_base = [c for c in strong_to_base if c != v6] + [v7]
    _add("v7_strong_to_base", strong_to_base)

    # All Strong→pullback swaps
    strong_to_pb = list(V6_VALIDATED)
    for v7, v6 in [("C0_pullback", "C0_sell_strong"), ("O0_pullback", "O0_buy_strong"),
                    ("O1_pullback", "O1_buy_strong"), ("O2_pullback", "O2_buy_strong")]:
        strong_to_pb = [c for c in strong_to_pb if c != v6] + [v7]
    _add("v7_strong_to_pullback", strong_to_pb)

    # All base→pullback swaps
    base_to_pb = list(V6_VALIDATED)
    for v7, v6 in [("C1_pullback", "C1_sell_base"), ("C3_pullback", "C3_buy"),
                    ("C4_pullback", "C4_sell_base")]:
        base_to_pb = [c for c in base_to_pb if c != v6] + [v7]
    _add("v7_base_to_pullback", base_to_pb)

    # All base→Strong swaps
    base_to_strong = list(V6_VALIDATED)
    for v7, v6 in [("C1_strong", "C1_sell_base"), ("C3_strong", "C3_buy"),
                    ("C4_strong", "C4_sell_base")]:
        base_to_strong = [c for c in base_to_strong if c != v6] + [v7]
    _add("v7_base_to_strong", base_to_strong)

    # Full pullback conversion (every cell that has a pullback variant)
    full_pb = list(V6_VALIDATED)
    for v7, v6 in SWAP_MAP.items():
        if "pullback" in v7:
            full_pb = [c for c in full_pb if c != v6] + [v7]
    _add("v7_full_pullback", full_pb)

    # Add all alternatives (maximum trade coverage via dedup)
    all_added = list(V6_VALIDATED) + [l for l in V7_CANDIDATES if l in available_labels]
    _add("v7_kitchen_sink", all_added)

    # Kitchen sink + new cells
    _add("v7_everything", all_added)

    return variants


def main() -> int:
    args = _parse_args()
    matrix = family_combo._load_matrix(Path(args.matrix))
    specs = _load_all_specs(matrix)
    available = set(specs.keys())

    strategies = {s.strategy for s in specs.values()}
    all_trades = discovery._load_all_normalized_trades(strategies)
    trades_by_ds = _select_all_trades(specs, all_trades)

    # Print cell trade counts
    print("Cell registry:")
    for label in sorted(ALL_REGISTRY.keys()):
        if label not in specs:
            continue
        strat = specs[label].strategy
        counts = []
        for ds in ["500k", "1000k"]:
            n = len(trades_by_ds[ds].get(label, []))
            counts.append(f"{ds}: {n:>3}t")
        is_v6 = "V6" if label in V6_REGISTRY else "v7"
        print(f"  {label:<45} {strat:<12} {is_v6}  {', '.join(counts)}")
    print()

    # Strict policy matching v6 validation exactly
    policy = additive.ConflictPolicy(
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

    # Pre-build baseline contexts
    baseline_ctx_by_ds = {
        ds: additive.build_baseline_context(discovery.DATASETS[ds])
        for ds in ["500k", "1000k"]
    }

    variants = _build_variants(available)
    results: list[dict[str, Any]] = []

    print(f"Running {len(variants)} variants under strict policy...")
    print()
    hdr = (f"{'Variant':<45} {'500k dUSD':>10} {'dPF':>7} {'dDD':>8} {'Add':>4}  "
           f"{'1000k dUSD':>10} {'dPF':>7} {'dDD':>8} {'Add':>4}  {'CombUSD':>12} {'CombPF':>8}")
    print(hdr)
    print("-" * len(hdr))

    for i, variant in enumerate(variants):
        vname = variant["name"]
        cells = variant["cells"]
        vresult = _run_variant(vname, cells, trades_by_ds, policy, baseline_ctx_by_ds)

        d5 = vresult["datasets"]["500k"]["delta_vs_baseline"]
        d1 = vresult["datasets"]["1000k"]["delta_vs_baseline"]
        s5 = vresult["datasets"]["500k"]["selection_counts"]
        s1 = vresult["datasets"]["1000k"]["selection_counts"]
        comb_usd = d5["net_usd"] + d1["net_usd"]
        comb_pf = d5["profit_factor"] + d1["profit_factor"]
        passes = (d5["net_usd"] > 0 and d1["net_usd"] > 0
                  and d5["profit_factor"] >= -0.005 and d1["profit_factor"] >= -0.005)
        tag = "+" if passes else " "

        print(f"{tag}{vname:<44} {d5['net_usd']:>+10,.0f} {d5['profit_factor']:>+.4f} {d5['max_drawdown_usd']:>+8,.0f} {s5['new_additive_trades_count']:>4}  "
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
                    "policy_stats": vresult["datasets"][ds]["policy_stats"],
                }
                for ds in ["500k", "1000k"]
            },
        })

    print()

    # === ANALYSIS ===
    v6_ref = next((v for v in results if v["name"] == "v6_validated"), None)
    v6_comb = v6_ref["combined_usd"] if v6_ref else 0
    v6_pf = v6_ref["combined_pf"] if v6_ref else 0

    passing = sorted([v for v in results if v["passes_strict"]], key=lambda v: -v["combined_usd"])

    print("=" * 95)
    print("PASSING VARIANTS RANKED BY COMBINED USD")
    print("=" * 95)
    print()
    print(f"{'#':<4} {'Variant':<45} {'CombUSD':>12} {'dv6':>10} {'CombPF':>8} {'Cells':>5}")
    print("-" * 90)
    for i, v in enumerate(passing):
        delta_v6 = v["combined_usd"] - v6_comb
        marker = " ***" if delta_v6 > 500 else ""
        print(f"  #{i+1:<2} {v['name']:<45} {v['combined_usd']:>+12,.0f} {delta_v6:>+10,.0f} {v['combined_pf']:>+8.4f} {v['cell_count']:>5}{marker}")

    # Swap analysis
    print()
    print("=" * 95)
    print("SWAP IMPACT (replace v6 cell with alternative profile)")
    print("=" * 95)
    print()
    print(f"{'Swap':<45} {'dv6 USD':>10} {'dv6 PF':>8} {'Verdict':>10}")
    print("-" * 80)

    swaps = [(v, v["combined_usd"] - v6_comb) for v in results if v["name"].startswith("swap_")]
    swaps.sort(key=lambda x: -x[1])
    for v, delta_usd in swaps:
        delta_pf = v["combined_pf"] - v6_pf
        if delta_usd > 1000 and delta_pf >= -0.1:
            verdict = "UPGRADE"
        elif delta_usd > 0:
            verdict = "MARGINAL"
        elif delta_usd > -1000:
            verdict = "NEUTRAL"
        else:
            verdict = "DOWNGRADE"
        print(f"  {v['name']:<45} {delta_usd:>+10,.0f} {delta_pf:>+8.4f} {verdict:>10}")

    # Addition analysis
    print()
    print("=" * 95)
    print("ADDITION IMPACT (add alternative profile alongside v6 cell)")
    print("=" * 95)
    print()
    print(f"{'Addition':<45} {'dv6 USD':>10} {'dv6 PF':>8} {'Extra Trades':>12}")
    print("-" * 80)

    adds = [(v, v["combined_usd"] - v6_comb) for v in results if v["name"].startswith("add_")]
    adds.sort(key=lambda x: -x[1])
    for v, delta_usd in adds:
        delta_pf = v["combined_pf"] - v6_pf
        new_t = sum(v["datasets"][ds]["selection"]["new_additive_trades_count"] for ds in ["500k", "1000k"])
        v6_t = sum(v6_ref["datasets"][ds]["selection"]["new_additive_trades_count"] for ds in ["500k", "1000k"]) if v6_ref else 0
        extra = new_t - v6_t
        print(f"  {v['name']:<45} {delta_usd:>+10,.0f} {delta_pf:>+8.4f} {extra:>+12}")

    payload = {
        "title": "V7 profile-swap family search",
        "policy": "strict: hedging + margin (leverage=33.3, buffer=20%, lot_cap=0.10)",
        "v6_reference_combined_usd": v6_comb,
        "v6_reference_combined_pf": v6_pf,
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
