#!/usr/bin/env python3
"""
Conflict-aware V44 family rerun — v2.

Enforces realistic V44 runtime constraints:
  1. Offensive V44 trades cannot overlap baseline V44 trades (same engine)
  2. Offensive V44 trades cannot overlap each other
  3. Greedy acceptance: chronological order, first-in wins

This is the honest test: only offensive V44 trades that could actually fire
at runtime (when no other V44 trade is active) get counted.

Reruns: v4_baseline, v4_plus_all_adj, v5_candidate_A
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
from scripts import backtest_merged_integrated_tokyo_london_v2_ny as merged_engine
from scripts import backtest_offensive_slice_family_combo as family_combo
from scripts import run_offensive_slice_discovery as discovery

OUT_DIR = ROOT / "research_out"
DEFAULT_OUTPUT = OUT_DIR / "system_variant_k_v5_conflict_aware.json"
DEFAULT_MATRIX = OUT_DIR / "offensive_slice_discovery_matrix.json"

CELL_REGISTRY = {
    "C0_sell_strong": "v44_ny__short__cells_ambiguous_er_high_der_pos__entry_profiles_Strong",
    "C1_sell_base": "v44_ny__short__cells_ambiguous_er_low_der_pos",
    "C2_sell": "v44_ny__short__cells_momentum_er_high_der_pos",
    "C3_buy": "v44_ny__long__cells_ambiguous_er_mid_der_neg",
    "C4_sell_base": "v44_ny__short__cells_ambiguous_er_mid_der_pos",
    "C4_sell_strong": "v44_ny__short__cells_ambiguous_er_mid_der_pos__entry_profiles_Strong",
    "C5_pbt_sell": "v44_ny__short__cells_post_breakout_trend_er_high_der_pos",
    "C6_pbt_sell": "v44_ny__short__cells_post_breakout_trend_er_mid_der_pos",
    "O0_buy_strong": "v44_ny__long__cells_ambiguous_er_high_der_pos__entry_profiles_Strong",
    "O1_buy_strong": "v44_ny__long__cells_ambiguous_er_low_der_pos__entry_profiles_Strong",
    "O2_buy_strong": "v44_ny__long__cells_ambiguous_er_mid_der_pos__entry_profiles_Strong",
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
    p = argparse.ArgumentParser(description="Conflict-aware V44 family rerun")
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
    out: dict[str, dict[str, list[dict[str, Any]]]] = {"500k": {}, "1000k": {}}
    for dataset_key in ["500k", "1000k"]:
        for label, spec in specs.items():
            out[dataset_key][label] = [
                t for t in all_trades[dataset_key].get(spec.strategy, [])
                if discovery._passes_filters(t, spec)
            ]
    return out


def _ts(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _intervals_overlap(a0: pd.Timestamp, a1: pd.Timestamp,
                        b0: pd.Timestamp, b1: pd.Timestamp) -> bool:
    return max(a0, b0) < min(a1, b1)


def _collect_family_trades(
    *,
    cell_labels: list[str],
    cell_scales: dict[str, float] | None,
    trades_by_label: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Collect and dedupe trades across cells."""
    scales = cell_scales or {}
    combined: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()

    for label in cell_labels:
        scale = scales.get(label, 1.0)
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

    combined.sort(key=lambda t: (t["entry_time"], t["exit_time"]))
    return combined


def _conflict_filter_with_baseline(
    offensive_trades: list[dict[str, Any]],
    baseline_v44_slots: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> tuple[list[dict[str, Any]], int, int]:
    """Greedy conflict filter: no overlap with baseline V44 or accepted offensive.

    Returns (accepted, rejected_by_baseline, rejected_by_internal).
    """
    sorted_trades = sorted(offensive_trades, key=lambda t: (t["entry_time"], t["exit_time"]))
    accepted: list[dict[str, Any]] = []
    accepted_slots: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    rejected_baseline = 0
    rejected_internal = 0

    for trade in sorted_trades:
        t_entry = _ts(trade["entry_time"])
        t_exit = _ts(trade["exit_time"])

        # Check vs baseline V44 trades
        if any(_intervals_overlap(t_entry, t_exit, s[0], s[1]) for s in baseline_v44_slots):
            rejected_baseline += 1
            continue

        # Check vs already-accepted offensive trades
        if any(_intervals_overlap(t_entry, t_exit, s[0], s[1]) for s in accepted_slots):
            rejected_internal += 1
            continue

        accepted.append(trade)
        accepted_slots.append((t_entry, t_exit))

    return accepted, rejected_baseline, rejected_internal


def _count_overlaps(trades: list[dict[str, Any]]) -> dict[str, int]:
    overlap_pairs = 0
    opposite_pairs = 0
    for i in range(len(trades)):
        for j in range(i + 1, len(trades)):
            if _intervals_overlap(
                _ts(trades[i]["entry_time"]), _ts(trades[i]["exit_time"]),
                _ts(trades[j]["entry_time"]), _ts(trades[j]["exit_time"]),
            ):
                overlap_pairs += 1
                if trades[i]["side"] != trades[j]["side"]:
                    opposite_pairs += 1
    return {"overlap_pairs": overlap_pairs, "opposite_side_pairs": opposite_pairs}


def _run_additive(
    selected_trades: list[dict[str, Any]],
    baseline_ctx: additive.BaselineContext,
) -> dict[str, Any]:
    return additive.run_slice_additive(
        baseline_ctx=baseline_ctx,
        slice_spec={"variant": "conflict_aware_v2"},
        selected_trades=selected_trades,
        size_scale=1.0,
    )


def main() -> int:
    args = _parse_args()
    matrix = family_combo._load_matrix(DEFAULT_MATRIX)
    specs = _load_all_specs(matrix)
    strategies = {s.strategy for s in specs.values()}
    all_trades = discovery._load_all_normalized_trades(strategies)
    trades_by_ds = _select_all_trades(specs, all_trades)

    variants = [
        {"name": "v4_baseline", "cells": V4_BASELINE, "scales": None},
        {
            "name": "v4_plus_all_adj",
            "cells": V4_BASELINE + ["ADJ_meanrev_low_neg_buy", "ADJ_ambig_mid_neg_sell", "ADJ_mom_high_neg_sell"],
            "scales": None,
        },
        {
            "name": "v5_candidate_A",
            "cells": [c if c != "C4_sell_base" else "C4_sell_strong" for c in V4_BASELINE]
                  + ["ADJ_meanrev_low_neg_buy", "ADJ_ambig_mid_neg_sell", "ADJ_mom_high_neg_sell"],
            "scales": None,
        },
    ]

    results: list[dict[str, Any]] = []

    print("=" * 130)
    print("CONFLICT-AWARE V44 FAMILY RERUN — v2")
    print("Policy: no offensive V44 overlap with baseline V44 trades OR each other")
    print("=" * 130)
    print()

    for variant in variants:
        vname = variant["name"]
        cells = variant["cells"]
        scales = variant.get("scales")

        print(f"─── {vname} ({len(cells)} cells) ───")
        print()

        ds_results: dict[str, dict[str, Any]] = {}

        for dataset_key in ["500k", "1000k"]:
            baseline_ctx = additive.build_baseline_context(discovery.DATASETS[dataset_key])

            # Extract baseline V44 time slots
            baseline_v44_slots = [
                (_ts(t.entry_time), _ts(t.exit_time))
                for t in baseline_ctx.baseline_kept
                if t.strategy == "v44_ny"
            ]

            # Collect all offensive trades (pre-filter)
            all_offensive = _collect_family_trades(
                cell_labels=cells,
                cell_scales=scales,
                trades_by_label=trades_by_ds[dataset_key],
            )

            overlaps = _count_overlaps(all_offensive)

            # Run optimistic (no conflict filter — original behavior)
            opt_result = _run_additive(all_offensive, baseline_ctx)

            # Apply conflict filter: vs baseline V44 + vs internal
            conflict_accepted, rej_baseline, rej_internal = _conflict_filter_with_baseline(
                all_offensive, baseline_v44_slots,
            )

            # Run conflict-aware
            con_result = _run_additive(conflict_accepted, baseline_ctx)

            ds_results[dataset_key] = {
                "optimistic": {
                    "selected": len(all_offensive),
                    "overlap_pairs": overlaps["overlap_pairs"],
                    "opposite_pairs": overlaps["opposite_side_pairs"],
                    "additive": opt_result["selection_counts"]["new_additive_trades_count"],
                    "displaced": opt_result["selection_counts"]["displaced_trades_count"],
                    "delta": opt_result["delta_vs_baseline"],
                    "summary": opt_result["variant_summary"],
                },
                "conflict_aware": {
                    "accepted": len(conflict_accepted),
                    "rejected_vs_baseline_v44": rej_baseline,
                    "rejected_vs_internal": rej_internal,
                    "additive": con_result["selection_counts"]["new_additive_trades_count"],
                    "displaced": con_result["selection_counts"]["displaced_trades_count"],
                    "delta": con_result["delta_vs_baseline"],
                    "summary": con_result["variant_summary"],
                },
                "baseline_v44_count": len(baseline_v44_slots),
            }

            do = ds_results[dataset_key]["optimistic"]
            dc = ds_results[dataset_key]["conflict_aware"]
            print(f"  {dataset_key} (baseline V44: {len(baseline_v44_slots)} trades):")
            print(f"    Optimistic:      {do['selected']:>3} selected → {do['additive']:>2} additive, {do['displaced']:>2} displaced  "
                  f"(internal overlap: {do['overlap_pairs']} pairs, {do['opposite_pairs']} opposite)")
            print(f"                     dUSD={do['delta']['net_usd']:>+10,.0f}  dPF={do['delta']['profit_factor']:>+.4f}  dDD={do['delta']['max_drawdown_usd']:>+8,.0f}")
            print(f"    Conflict-aware:  {dc['accepted']:>3} accepted  (rejected: {dc['rejected_vs_baseline_v44']} vs baseline V44, {dc['rejected_vs_internal']} vs internal)")
            print(f"                     → {dc['additive']:>2} additive, {dc['displaced']:>2} displaced")
            print(f"                     dUSD={dc['delta']['net_usd']:>+10,.0f}  dPF={dc['delta']['profit_factor']:>+.4f}  dDD={dc['delta']['max_drawdown_usd']:>+8,.0f}")
            print(f"    System totals:   opt {do['summary']['total_trades']}t / ${do['summary']['net_usd']:>+12,.2f}  "
                  f"conflict {dc['summary']['total_trades']}t / ${dc['summary']['net_usd']:>+12,.2f}")
            print()

        opt_comb = (ds_results["500k"]["optimistic"]["delta"]["net_usd"]
                    + ds_results["1000k"]["optimistic"]["delta"]["net_usd"])
        con_comb = (ds_results["500k"]["conflict_aware"]["delta"]["net_usd"]
                    + ds_results["1000k"]["conflict_aware"]["delta"]["net_usd"])
        inflation = opt_comb - con_comb

        print(f"  COMBINED:  optimistic={opt_comb:>+12,.0f}  conflict-aware={con_comb:>+12,.0f}  inflation={inflation:>+10,.0f}")
        print()

        results.append({
            "name": vname,
            "cells": cells,
            "datasets": ds_results,
            "combined_optimistic": round(opt_comb, 2),
            "combined_conflict_aware": round(con_comb, 2),
            "inflation": round(inflation, 2),
        })

    # ═══════════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════════

    print("=" * 130)
    print("CONFLICT-AWARE RANKING")
    print("=" * 130)
    print()

    ranked = sorted(results, key=lambda v: -v["combined_conflict_aware"])
    v4_con = next((v["combined_conflict_aware"] for v in results if v["name"] == "v4_baseline"), 0)

    hdr = (f"{'Rank':<5} {'Variant':<25} {'Opt Comb':>12} {'Conflict Comb':>14} "
           f"{'Δ vs v4':>10} {'Inflation':>12} {'Infl %':>8}")
    print(hdr)
    print("-" * len(hdr))
    for i, v in enumerate(ranked):
        delta_v4 = v["combined_conflict_aware"] - v4_con
        pct = (v["inflation"] / v["combined_conflict_aware"] * 100) if v["combined_conflict_aware"] != 0 else 0
        print(f"  #{i+1:<3} {v['name']:<25} {v['combined_optimistic']:>+12,.0f} "
              f"{v['combined_conflict_aware']:>+14,.0f} {delta_v4:>+10,.0f} "
              f"{v['inflation']:>+12,.0f} {pct:>+7.0f}%")
    print()

    print("STRICT BAR CHECK (conflict-aware):")
    for v in ranked:
        d5 = v["datasets"]["500k"]["conflict_aware"]["delta"]
        d1 = v["datasets"]["1000k"]["conflict_aware"]["delta"]
        passes = (d5["net_usd"] > 0 and d1["net_usd"] > 0
                  and d5["profit_factor"] >= -0.005 and d1["profit_factor"] >= -0.005)
        tag = "PASS" if passes else "FAIL"
        print(f"  [{tag}] {v['name']:<25} "
              f"500k: dUSD={d5['net_usd']:>+10,.0f} dPF={d5['profit_factor']:>+.4f}  "
              f"1000k: dUSD={d1['net_usd']:>+10,.0f} dPF={d1['profit_factor']:>+.4f}")
    print()

    # Per-dataset detail table
    print("PER-DATASET CONFLICT-AWARE DETAIL:")
    print(f"{'Variant':<25} {'DS':>5} {'Accepted':>8} {'Additive':>8} {'Displaced':>9} "
          f"{'dUSD':>10} {'dPF':>8} {'dDD':>8} {'SysTrades':>9} {'SysUSD':>12}")
    print("-" * 115)
    for v in ranked:
        for ds in ["500k", "1000k"]:
            dc = v["datasets"][ds]["conflict_aware"]
            print(f"{v['name']:<25} {ds:>5} {dc['accepted']:>8} {dc['additive']:>8} {dc['displaced']:>9} "
                  f"{dc['delta']['net_usd']:>+10,.0f} {dc['delta']['profit_factor']:>+8.4f} "
                  f"{dc['delta']['max_drawdown_usd']:>+8,.0f} {dc['summary']['total_trades']:>9} "
                  f"{dc['summary']['net_usd']:>+12,.2f}")

    payload = {
        "title": "Conflict-aware V44 family rerun — v2",
        "policy": "No offensive V44 overlap with baseline V44 trades OR each other (greedy chronological)",
        "variants_tested": len(results),
        "results": results,
    }
    output = Path(args.output)
    output.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    print(f"\nWrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
