#!/usr/bin/env python3
"""
Whitespace census: map defensive stack (Variant K) coverage gaps vs Setup D candidates.

For each ownership cell, answers:
  - How many Variant K (defensive) trades exist? (whitespace / thin / covered)
  - How many Setup D candidates exist? (long-only, survived realistic state)
  - How many with first_30min filter?
  - What is their avg PnL?

Identifies promotion-candidate cells where Setup D fills whitespace.

Research/diagnostic only -- no live code changes.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path("/Users/codygnon/Documents/usdjpy_assistant")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.ownership_table import cell_key, der_bucket, er_bucket
from scripts import backtest_variant_k_london_cluster as variant_k
from scripts import backtest_variant_i_pbt_standdown as variant_i
from scripts import diagnostic_london_setupd_trade_outcomes as setupd_outcomes
from scripts import backtest_v2_multisetup_london as london_v2_engine

OUT_DIR = ROOT / "research_out"
OUTPUT_PATH = OUT_DIR / "diagnostic_whitespace_census.json"

TRADE_OUTCOME_ARGS = SimpleNamespace(
    spread_pips=0.3,
    sl_buffer_pips=3.0,
    sl_min_pips=5.0,
    sl_max_pips=20.0,
    tp1_r_multiple=1.0,
    tp2_r_multiple=2.0,
    tp1_close_fraction=0.5,
    be_offset_pips=1.0,
)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return str(obj)


def _dataset_key(path: str) -> str:
    name = Path(path).name
    if "500k" in name:
        return "500k"
    if "1000k" in name:
        return "1000k"
    return name


def _er_bucket(er: float) -> str:
    if er < 0.35:
        return "er_low"
    if er < 0.55:
        return "er_mid"
    return "er_high"


def _der_bucket(delta_er: float) -> str:
    return "der_neg" if delta_er < 0 else "der_pos"


# ── First 30min filter (same logic as additive backtest) ────────────────


def _passes_first_30min(trade: dict[str, Any]) -> bool:
    signal_time = pd.Timestamp(trade["signal_time"])
    if signal_time.tzinfo is None:
        signal_time = signal_time.tz_localize("UTC")
    day = signal_time.floor("D")
    london_open_hour = london_v2_engine.uk_london_open_utc(day)
    london_open = day + pd.Timedelta(hours=london_open_hour)
    minutes_since_open = (signal_time - london_open).total_seconds() / 60.0
    return 15.0 <= minutes_since_open <= 45.0


# ── Classify Variant K trades into cells ────────────────────────────────


def _classify_variant_k_trades(
    dataset: str,
) -> dict[str, list[dict[str, Any]]]:
    """
    Get all Variant K kept trades and classify each into an ownership cell.
    Returns {cell_key: [trade_info, ...]}.
    """
    kept, _baseline, classified_dynamic, dyn_time_idx, _blocked_cluster, _blocked_global = (
        variant_k.build_variant_k_pre_coupling_kept(dataset)
    )

    cell_trades: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for trade in kept:
        row = variant_i._lookup_regime_with_dynamic(
            classified_dynamic, dyn_time_idx, trade.entry_time
        )
        full_row = classified_dynamic.iloc[
            dyn_time_idx.get_indexer([pd.Timestamp(trade.entry_time)], method="ffill")[0]
        ]
        er = float(full_row.get("sf_er", 0.5))
        if np.isnan(er):
            er = 0.5
        cell = cell_key(
            row["regime_label"], _er_bucket(er), _der_bucket(row["delta_er"])
        )

        cell_trades[cell].append({
            "strategy": trade.strategy,
            "entry_time": str(trade.entry_time),
            "side": trade.side,
            "pips": float(trade.pips),
            "usd": float(trade.usd),
        })

    return dict(cell_trades)


# ── Classify Setup D trades by cell ─────────────────────────────────────


def _classify_setupd_trades(
    report: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """
    Extract long-only, native-allowed Setup D trades from the report,
    grouped by ownership cell.
    """
    cell_trades: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in report["_all_trades"]:
        if t["direction"] != "long" or not t["native_allowed"]:
            continue
        cell = t["ownership_cell"]
        cell_trades[cell].append(t)
    return dict(cell_trades)


# ── Coverage classification ─────────────────────────────────────────────


def _coverage_class(count: int) -> str:
    if count <= 2:
        return "whitespace"
    if count <= 5:
        return "thin"
    return "covered"


# ── Metrics helper ──────────────────────────────────────────────────────


def _cell_setupd_metrics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    if not trades:
        return {
            "count_all": 0,
            "count_first30": 0,
            "avg_pnl_all": 0.0,
            "total_pnl_all": 0.0,
            "avg_pnl_first30": 0.0,
            "total_pnl_first30": 0.0,
            "win_rate_all": 0.0,
            "win_rate_first30": 0.0,
        }

    pnls_all = [float(t["pnl_pips"]) for t in trades]
    first30 = [t for t in trades if _passes_first_30min(t)]
    pnls_f30 = [float(t["pnl_pips"]) for t in first30]

    return {
        "count_all": len(trades),
        "count_first30": len(first30),
        "avg_pnl_all": round(statistics.mean(pnls_all), 2) if pnls_all else 0.0,
        "total_pnl_all": round(sum(pnls_all), 2),
        "avg_pnl_first30": round(statistics.mean(pnls_f30), 2) if pnls_f30 else 0.0,
        "total_pnl_first30": round(sum(pnls_f30), 2),
        "win_rate_all": round(100.0 * sum(1 for p in pnls_all if p > 0) / len(pnls_all), 1),
        "win_rate_first30": round(100.0 * sum(1 for p in pnls_f30 if p > 0) / len(pnls_f30), 1) if pnls_f30 else 0.0,
    }


# ── Run one dataset ────────────────────────────────────────────────────


def run_dataset(dataset: str) -> dict[str, Any]:
    dk = _dataset_key(dataset)
    print(f"\n{'='*70}")
    print(f"Whitespace census: {dk}")
    print(f"{'='*70}")

    # Step 1: Classify Variant K trades by cell
    print(f"  [1/3] Classifying Variant K defensive trades by cell ...")
    vk_by_cell = _classify_variant_k_trades(dataset)

    # Step 2: Get Setup D trade outcomes with per-trade detail
    print(f"  [2/3] Running Setup D trade outcome simulation ...")
    setupd_report = setupd_outcomes.run_dataset(dataset, TRADE_OUTCOME_ARGS)
    setupd_by_cell = _classify_setupd_trades(setupd_report)

    # Step 3: Cross-reference
    print(f"  [3/3] Cross-referencing coverage ...")

    # Collect all observed cells
    all_cells = sorted(set(list(vk_by_cell.keys()) + list(setupd_by_cell.keys())))

    whitespace_cells: list[dict[str, Any]] = []
    thin_cells: list[dict[str, Any]] = []
    covered_cells: list[dict[str, Any]] = []

    for cell in all_cells:
        vk_trades = vk_by_cell.get(cell, [])
        vk_count = len(vk_trades)
        coverage = _coverage_class(vk_count)

        setupd_trades = setupd_by_cell.get(cell, [])
        setupd_metrics = _cell_setupd_metrics(setupd_trades)

        # Also break down VK by strategy
        vk_by_strategy: dict[str, int] = defaultdict(int)
        for t in vk_trades:
            vk_by_strategy[t["strategy"]] += 1

        entry = {
            "cell": cell,
            "coverage_class": coverage,
            "variant_k_trade_count": vk_count,
            "variant_k_by_strategy": dict(vk_by_strategy),
            "variant_k_net_pips": round(sum(t["pips"] for t in vk_trades), 2) if vk_trades else 0.0,
            "setupd_candidates_all": setupd_metrics["count_all"],
            "setupd_candidates_first30": setupd_metrics["count_first30"],
            "setupd_avg_pnl_all": setupd_metrics["avg_pnl_all"],
            "setupd_total_pnl_all": setupd_metrics["total_pnl_all"],
            "setupd_avg_pnl_first30": setupd_metrics["avg_pnl_first30"],
            "setupd_total_pnl_first30": setupd_metrics["total_pnl_first30"],
            "setupd_win_rate_all": setupd_metrics["win_rate_all"],
            "setupd_win_rate_first30": setupd_metrics["win_rate_first30"],
            "has_setupd_candidates": setupd_metrics["count_all"] > 0,
        }

        if coverage == "whitespace":
            whitespace_cells.append(entry)
        elif coverage == "thin":
            thin_cells.append(entry)
        else:
            covered_cells.append(entry)

    # Coverage summary
    coverage_summary = {
        "total_cells_observed": len(all_cells),
        "whitespace_cells": len(whitespace_cells),
        "thin_cells": len(thin_cells),
        "covered_cells": len(covered_cells),
        "whitespace_with_setupd": sum(1 for c in whitespace_cells if c["has_setupd_candidates"]),
        "thin_with_setupd": sum(1 for c in thin_cells if c["has_setupd_candidates"]),
    }

    # Promotion candidates: whitespace/thin with Setup D candidates
    promotion_candidates = []
    for entry in whitespace_cells + thin_cells:
        if entry["setupd_candidates_all"] > 0:
            promotion_candidates.append({
                **entry,
                "priority": "high" if entry["setupd_avg_pnl_first30"] > 0 and entry["setupd_candidates_first30"] > 0 else (
                    "medium" if entry["setupd_avg_pnl_all"] > 0 else "low"
                ),
            })

    # Sort by first30 candidate count, then by avg PnL
    promotion_candidates.sort(
        key=lambda x: (-x["setupd_candidates_first30"], -x["setupd_avg_pnl_first30"]),
    )

    # Aggregate opportunity
    ws_thin = whitespace_cells + thin_cells
    ws_thin_with_setupd = [c for c in ws_thin if c["has_setupd_candidates"]]
    total_candidates_all = sum(c["setupd_candidates_all"] for c in ws_thin_with_setupd)
    total_candidates_f30 = sum(c["setupd_candidates_first30"] for c in ws_thin_with_setupd)
    total_pnl_all = sum(c["setupd_total_pnl_all"] for c in ws_thin_with_setupd)
    total_pnl_f30 = sum(c["setupd_total_pnl_first30"] for c in ws_thin_with_setupd)

    aggregate_opportunity = {
        "total_setupd_candidates_in_whitespace_thin": total_candidates_all,
        "total_setupd_candidates_first30_in_whitespace_thin": total_candidates_f30,
        "total_pnl_pips_all": round(total_pnl_all, 2),
        "total_pnl_pips_first30": round(total_pnl_f30, 2),
        "avg_pnl_per_trade_all": round(total_pnl_all / total_candidates_all, 2) if total_candidates_all else 0.0,
        "avg_pnl_per_trade_first30": round(total_pnl_f30 / total_candidates_f30, 2) if total_candidates_f30 else 0.0,
        "cells_contributing": len(ws_thin_with_setupd),
    }

    # Also compute covered-cell setupd stats for comparison
    covered_with_setupd = [c for c in covered_cells if c["has_setupd_candidates"]]
    covered_candidates_all = sum(c["setupd_candidates_all"] for c in covered_with_setupd)
    covered_pnl_all = sum(c["setupd_total_pnl_all"] for c in covered_with_setupd)
    aggregate_opportunity["comparison_covered_candidates_all"] = covered_candidates_all
    aggregate_opportunity["comparison_covered_total_pnl_all"] = round(covered_pnl_all, 2)
    aggregate_opportunity["comparison_covered_avg_pnl_all"] = (
        round(covered_pnl_all / covered_candidates_all, 2) if covered_candidates_all else 0.0
    )

    return {
        "dataset": dk,
        "coverage_summary": coverage_summary,
        "whitespace_cells": whitespace_cells,
        "thin_cells": thin_cells,
        "covered_cells": covered_cells,
        "promotion_candidates": promotion_candidates,
        "aggregate_opportunity": aggregate_opportunity,
    }


def _build_cross_dataset_verdict(results: dict[str, dict[str, Any]]) -> str:
    """Build plain-language verdict from both datasets."""
    lines = []
    lines.append("=== WHITESPACE CENSUS VERDICT ===\n")

    for dk in ["500k", "1000k"]:
        r = results.get(dk)
        if not r:
            continue
        cs = r["coverage_summary"]
        ao = r["aggregate_opportunity"]
        lines.append(f"--- {dk} ---")
        lines.append(f"  Cells observed: {cs['total_cells_observed']}")
        lines.append(f"  Whitespace (0-2 VK trades): {cs['whitespace_cells']} cells, {cs['whitespace_with_setupd']} with Setup D candidates")
        lines.append(f"  Thin (3-5 VK trades): {cs['thin_cells']} cells, {cs['thin_with_setupd']} with Setup D candidates")
        lines.append(f"  Covered (6+ VK trades): {cs['covered_cells']} cells")
        lines.append(f"  Setup D trades in whitespace+thin cells: {ao['total_setupd_candidates_in_whitespace_thin']} all, {ao['total_setupd_candidates_first30_in_whitespace_thin']} first30")
        lines.append(f"  Total PnL (pips, all): {ao['total_pnl_pips_all']}, first30: {ao['total_pnl_pips_first30']}")
        lines.append(f"  Avg PnL per trade (all): {ao['avg_pnl_per_trade_all']}, first30: {ao['avg_pnl_per_trade_first30']}")
        lines.append(f"  For comparison, Setup D in covered cells: {ao['comparison_covered_candidates_all']} trades, avg PnL {ao['comparison_covered_avg_pnl_all']} pips")
        lines.append("")

    # Cross-dataset promotion candidates
    promo_500k = {c["cell"] for c in results.get("500k", {}).get("promotion_candidates", [])}
    promo_1000k = {c["cell"] for c in results.get("1000k", {}).get("promotion_candidates", [])}
    stable_promos = sorted(promo_500k & promo_1000k)
    unstable_promos = sorted((promo_500k | promo_1000k) - (promo_500k & promo_1000k))

    lines.append(f"STABLE promotion candidates (present in both datasets): {len(stable_promos)}")
    for cell in stable_promos:
        info_500 = next((c for c in results["500k"]["promotion_candidates"] if c["cell"] == cell), None)
        info_1000 = next((c for c in results["1000k"]["promotion_candidates"] if c["cell"] == cell), None)
        lines.append(f"  {cell}:")
        if info_500:
            lines.append(f"    500k: VK={info_500['variant_k_trade_count']}, SetupD all={info_500['setupd_candidates_all']}, first30={info_500['setupd_candidates_first30']}, avg_pnl_first30={info_500['setupd_avg_pnl_first30']}")
        if info_1000:
            lines.append(f"    1000k: VK={info_1000['variant_k_trade_count']}, SetupD all={info_1000['setupd_candidates_all']}, first30={info_1000['setupd_candidates_first30']}, avg_pnl_first30={info_1000['setupd_avg_pnl_first30']}")

    if unstable_promos:
        lines.append(f"\nUNSTABLE promotion candidates (one dataset only): {len(unstable_promos)}")
        for cell in unstable_promos:
            ds_label = "500k" if cell in promo_500k else "1000k"
            lines.append(f"  {cell} ({ds_label} only)")

    # Overall volume assessment
    total_stable_f30 = 0
    total_stable_pnl_f30 = 0.0
    for cell in stable_promos:
        for dk in ["500k", "1000k"]:
            info = next((c for c in results.get(dk, {}).get("promotion_candidates", []) if c["cell"] == cell), None)
            if info:
                total_stable_f30 += info["setupd_candidates_first30"]
                total_stable_pnl_f30 += info["setupd_total_pnl_first30"]

    lines.append(f"\n=== VOLUME ASSESSMENT ===")
    lines.append(f"Total stable-promo first30 trades across both datasets: {total_stable_f30}")
    lines.append(f"Total stable-promo first30 PnL across both datasets: {round(total_stable_pnl_f30, 2)} pips")

    if total_stable_f30 >= 40:
        lines.append("VERDICT: Substantial coverage expansion. The offensive whitespace layer adds meaningful trade volume.")
    elif total_stable_f30 >= 15:
        lines.append("VERDICT: Moderate coverage expansion. Worth pursuing if PnL is positive.")
    elif total_stable_f30 >= 5:
        lines.append("VERDICT: Marginal coverage expansion. A few trades in low-competition cells.")
    else:
        lines.append("VERDICT: Minimal opportunity. The whitespace cells have very few Setup D candidates.")

    return "\n".join(lines)


def main() -> int:
    datasets = [
        str(OUT_DIR / "USDJPY_M1_OANDA_500k.csv"),
        str(OUT_DIR / "USDJPY_M1_OANDA_1000k.csv"),
    ]

    results: dict[str, dict[str, Any]] = {}
    for ds in datasets:
        if not Path(ds).exists():
            print(f"Skipping {ds} (not found)")
            continue
        dk = _dataset_key(ds)
        results[dk] = run_dataset(ds)

    verdict = _build_cross_dataset_verdict(results)
    print(f"\n{verdict}")

    # Build cross-dataset stability info
    promo_500k = {c["cell"]: c for c in results.get("500k", {}).get("promotion_candidates", [])}
    promo_1000k = {c["cell"]: c for c in results.get("1000k", {}).get("promotion_candidates", [])}
    stable_cells = sorted(set(promo_500k.keys()) & set(promo_1000k.keys()))

    cross_dataset_promotion = []
    for cell in stable_cells:
        cross_dataset_promotion.append({
            "cell": cell,
            "500k": promo_500k[cell],
            "1000k": promo_1000k[cell],
            "stable": True,
        })

    output = {
        "run_info": {
            "datasets": [_dataset_key(ds) for ds in datasets if Path(ds).exists()],
            "trade_outcome_params": {
                "spread_pips": TRADE_OUTCOME_ARGS.spread_pips,
                "sl_buffer_pips": TRADE_OUTCOME_ARGS.sl_buffer_pips,
                "sl_min_pips": TRADE_OUTCOME_ARGS.sl_min_pips,
                "sl_max_pips": TRADE_OUTCOME_ARGS.sl_max_pips,
                "tp1_r_multiple": TRADE_OUTCOME_ARGS.tp1_r_multiple,
                "tp2_r_multiple": TRADE_OUTCOME_ARGS.tp2_r_multiple,
                "tp1_close_fraction": TRADE_OUTCOME_ARGS.tp1_close_fraction,
                "be_offset_pips": TRADE_OUTCOME_ARGS.be_offset_pips,
            },
            "coverage_thresholds": {
                "whitespace": "0-2 Variant K trades",
                "thin": "3-5 Variant K trades",
                "covered": "6+ Variant K trades",
            },
        },
        "per_dataset": results,
        "cross_dataset_promotion_candidates": cross_dataset_promotion,
        "aggregate_opportunity": {
            dk: results[dk]["aggregate_opportunity"]
            for dk in results
        },
        "verdict": verdict,
    }

    OUTPUT_PATH.write_text(
        json.dumps(output, indent=2, default=_json_default), encoding="utf-8"
    )
    print(f"\nWrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
