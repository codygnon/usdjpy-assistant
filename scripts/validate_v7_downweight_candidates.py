#!/usr/bin/env python3
"""
Validate top V7 down-weight candidates using the exact v7 search pipeline.

Reuses backtest_variant_k_v7_search infrastructure (scaled trades, hedging policy,
additive pipeline) and emits per-candidate JSON + markdown at the same fidelity
as validate_v6_combo_candidate.

Usage:
  python3 scripts/validate_v7_downweight_candidates.py
"""
from __future__ import annotations

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
from scripts import backtest_variant_k_v6_search as v6
from scripts import backtest_variant_k_v7_search as v7
from scripts import run_offensive_slice_discovery as discovery

OUT_DIR = ROOT / "research_out"

CANDIDATES = {
    "v7_dw_highest_usd": {
        **{label: 1.0 for label in v7.FIXED_CORE},
        "L2_brkout_mid_neg_buy": 1.0,
        "T1_ambig_high_pos_buy": 1.0,
        "T2_brkout_mid_pos_buy": 1.0,
        "N3_brkout_low_neg_buy_news": 1.0,
        "N4_pbt_low_neg_buy_news": 1.0,
        "L1_mom_low_pos_buy": 1.0,
        "T3_ambig_mid_pos_sell": 0.5,
    },
    "v7_dw_best_pf_dd": {
        **{label: 1.0 for label in v7.FIXED_CORE},
        "L2_brkout_mid_neg_buy": 1.0,
        "T1_ambig_high_pos_buy": 1.0,
        "T2_brkout_mid_pos_buy": 1.0,
        "N3_brkout_low_neg_buy_news": 1.0,
        "N4_pbt_low_neg_buy_news": 1.0,
        "L1_mom_low_pos_buy": 1.0,
        "T3_ambig_mid_pos_sell": 0.25,
    },
    "v7_dw_pruned_t1": {
        **{label: 1.0 for label in v7.FIXED_CORE},
        "L2_brkout_mid_neg_buy": 1.0,
        "T1_ambig_high_pos_buy": 0.5,
        "T2_brkout_mid_pos_buy": 1.0,
        "N3_brkout_low_neg_buy_news": 1.0,
        "N4_pbt_low_neg_buy_news": 1.0,
        "L1_mom_low_pos_buy": 1.0,
        "T3_ambig_mid_pos_sell": 1.0,
    },
}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return discovery._json_default(obj)


def _validate_one(
    name: str,
    scales: dict[str, float],
    trades_by_ds: dict[str, dict[str, list[dict[str, Any]]]],
    policy: additive.ConflictPolicy,
    baseline_ctx_by_ds: dict[str, additive.BaselineContext],
) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for ds in ["500k", "1000k"]:
        combined = v7._scaled_combined_trades(scales, trades_by_ds[ds])
        result = additive.run_slice_additive_with_policy(
            baseline_ctx=baseline_ctx_by_ds[ds],
            slice_spec={"variant": name, "cell_scales": scales},
            selected_trades=combined,
            conflict_policy=policy,
            size_scale=1.0,
        )
        datasets[ds] = {
            "summary": result["variant_summary"],
            "delta_vs_baseline": result["delta_vs_baseline"],
            "selection_counts": result["selection_counts"],
            "policy_stats": result.get("policy_stats", {}),
        }
    d5 = datasets["500k"]["delta_vs_baseline"]
    d1 = datasets["1000k"]["delta_vs_baseline"]
    return {
        "title": "V7 Down-Weight Candidate Validation",
        "name": name,
        "cell_scales": scales,
        "active_cells": sorted(k for k, v in scales.items() if v > 0),
        "downweighted_cells": sorted(k for k, v in scales.items() if 0 < v < 1.0),
        "pruned_cells": sorted(k for k, v in scales.items() if v == 0),
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
        "combined_delta_usd": round(d5["net_usd"] + d1["net_usd"], 2),
        "combined_delta_pf": round(d5["profit_factor"] + d1["profit_factor"], 4),
        "combined_delta_dd": round(d5["max_drawdown_usd"] + d1["max_drawdown_usd"], 2),
        "passes_strict": d5["net_usd"] > 0.0 and d1["net_usd"] > 0.0 and d5["profit_factor"] >= 0.0 and d1["profit_factor"] >= 0.0,
        "datasets": datasets,
    }


def _build_md(payload: dict[str, Any]) -> str:
    lines = [
        "# V7 Down-Weight Candidate Validation",
        "",
        f"- name: `{payload['name']}`",
        f"- active cells: {len(payload['active_cells'])}",
        f"- downweighted: `{', '.join(payload['downweighted_cells']) or 'none'}`",
        f"- pruned: `{', '.join(payload['pruned_cells']) or 'none'}`",
        f"- policy: `{payload['policy']['name']}`",
        "",
        f"- combined delta USD: `{payload['combined_delta_usd']}`",
        f"- combined delta PF: `{payload['combined_delta_pf']}`",
        f"- combined delta DD: `{payload['combined_delta_dd']}`",
        f"- passes strict: `{payload['passes_strict']}`",
        "",
    ]
    for ds_key in ["500k", "1000k"]:
        ds = payload["datasets"][ds_key]
        s = ds["summary"]
        delta = ds["delta_vs_baseline"]
        sel = ds["selection_counts"]
        stats = ds["policy_stats"]
        lines += [
            f"## {ds_key}",
            "",
            f"- total trades: `{s['total_trades']}`",
            f"- wins/losses: `{s['wins']}`/`{s['losses']}`",
            f"- win rate: `{round(s['win_rate_pct'], 2)}%`",
            f"- net USD: `{round(s['net_usd'], 2)}`",
            f"- PF: `{round(s['profit_factor'], 4)}`",
            f"- max DD: `{round(s['max_drawdown_usd'], 2)}`",
            f"- Sharpe: `{round(s.get('sharpe_ratio', 0), 3)}`",
            f"- Calmar: `{round(s.get('calmar_ratio', 0), 3)}`",
            f"- delta USD: `{delta['net_usd']}`",
            f"- delta PF: `{delta['profit_factor']}`",
            f"- delta DD: `{delta['max_drawdown_usd']}`",
            f"- additive trades: `{sel['new_additive_trades_count']}`",
            f"- internal overlap pairs: `{sel['internal_overlap_pairs']}`",
            f"- policy stats: `{json.dumps(stats, sort_keys=True)}`",
            "",
        ]
    return "\n".join(lines) + "\n"


def main() -> int:
    print("Loading matrix and trades ...", flush=True)
    matrix = family_combo._load_matrix(v6.DEFAULT_MATRIX)
    specs = v6._load_all_specs(matrix)
    wanted = set()
    for scales in CANDIDATES.values():
        wanted |= {k for k, v in scales.items() if v > 0}
    strategies = {specs[label].strategy for label in wanted if label in specs}
    all_trades = discovery._load_all_normalized_trades(strategies)
    trades_by_ds = v6._select_all_trades(specs, all_trades)
    policy = v7._policy()

    baseline_ctx_by_ds = {}
    for ds_key, ds_path in [("500k", discovery.DATASETS["500k"]), ("1000k", discovery.DATASETS["1000k"])]:
        baseline_ctx_by_ds[ds_key] = additive.build_baseline_context(ds_path)

    results = []
    for name, scales in CANDIDATES.items():
        print(f"\nValidating {name} ...", flush=True)
        payload = _validate_one(name, scales, trades_by_ds, policy, baseline_ctx_by_ds)
        results.append(payload)

        json_path = OUT_DIR / f"{name}_validation.json"
        md_path = OUT_DIR / f"{name}_validation.md"
        json_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
        md_path.write_text(_build_md(payload), encoding="utf-8")
        print(f"  Wrote {json_path}")
        print(f"  Wrote {md_path}")

    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Name':45s}  {'cUSD':>12s}  {'cPF':>8s}  {'cDD':>10s}  {'Strict':>6s}")
    print("-" * 90)
    for r in sorted(results, key=lambda x: -x["combined_delta_usd"]):
        print(f"{r['name']:45s}  {r['combined_delta_usd']:>12.2f}  {r['combined_delta_pf']:>8.4f}  {r['combined_delta_dd']:>10.2f}  {str(r['passes_strict']):>6s}")
    print()

    pick_usd = max(results, key=lambda x: x["combined_delta_usd"])
    pick_pfdd = max(results, key=lambda x: (x["combined_delta_pf"], -x["combined_delta_dd"]))

    print(f"PICK (highest USD):      {pick_usd['name']}  USD={pick_usd['combined_delta_usd']}")
    print(f"PICK (best PF/DD):       {pick_pfdd['name']}  PF={pick_pfdd['combined_delta_pf']}  DD={pick_pfdd['combined_delta_dd']}")

    winner_path = OUT_DIR / "v7_downweight_winner.json"
    winner_path.write_text(json.dumps({
        "pick_highest_usd": {"name": pick_usd["name"], "combined_delta_usd": pick_usd["combined_delta_usd"], "combined_delta_pf": pick_usd["combined_delta_pf"], "combined_delta_dd": pick_usd["combined_delta_dd"]},
        "pick_best_pf_dd": {"name": pick_pfdd["name"], "combined_delta_usd": pick_pfdd["combined_delta_usd"], "combined_delta_pf": pick_pfdd["combined_delta_pf"], "combined_delta_dd": pick_pfdd["combined_delta_dd"]},
        "all_candidates": [{"name": r["name"], "combined_delta_usd": r["combined_delta_usd"], "combined_delta_pf": r["combined_delta_pf"], "combined_delta_dd": r["combined_delta_dd"], "passes_strict": r["passes_strict"]} for r in results],
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {winner_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
