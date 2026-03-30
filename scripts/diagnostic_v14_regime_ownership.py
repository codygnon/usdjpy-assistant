#!/usr/bin/env python3
"""
V14 Regime-Cell Ownership Diagnostic.

Determines whether V14 losses concentrate in stable regime/ER/ΔER cells
strongly enough to justify a V14-specific veto rule.

Uses the current promoted stack (F + G + I + K) as baseline.
Isolates V14 trades and computes per-cell performance on both datasets.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path("/Users/codygnon/Documents/usdjpy_assistant")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import backtest_merged_integrated_tokyo_london_v2_ny as merged_engine
from scripts import backtest_variant_i_pbt_standdown as variant_i
from scripts import backtest_variant_k_london_cluster as variant_k

OUT_DIR = ROOT / "research_out"
STARTING_EQUITY = 100_000.0

DATASETS = [
    str(OUT_DIR / "USDJPY_M1_OANDA_500k.csv"),
    str(OUT_DIR / "USDJPY_M1_OANDA_1000k.csv"),
]


def _dataset_key(dataset_path: str) -> str:
    name = Path(dataset_path).name
    if "500k" in name:
        return "500k"
    if "1000k" in name:
        return "1000k"
    raise ValueError(f"Unknown dataset: {name}")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)


def _er_bucket(er: float) -> str:
    if er < 0.35:
        return "er_low"
    if er < 0.55:
        return "er_mid"
    return "er_high"


def _der_bucket(delta_er: float) -> str:
    return "der_neg" if delta_er < 0 else "der_pos"


def _cell_key(regime: str, er_b: str, der_b: str) -> str:
    return f"{regime}/{er_b}/{der_b}"


# ── Load ownership context from existing diagnostic ──

def _load_ownership_context() -> dict[str, dict[str, Any]]:
    """Load cell owners from the existing ownership diagnostic output."""
    path = OUT_DIR / "diagnostic_strategy_ownership.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    context: dict[str, dict[str, Any]] = {}
    for dk in ["500k", "1000k"]:
        if dk not in data:
            continue
        grid = data[dk].get("grid", {})
        for cell_name, cell_data in grid.items():
            if cell_name not in context:
                context[cell_name] = {}
            context[cell_name][dk] = {
                "owner": cell_data.get("owner", "unknown"),
                "no_trade": cell_data.get("no_trade", "False"),
            }
    # Determine stable owner
    for cell_name, ds_info in context.items():
        owners = [ds_info[dk]["owner"] for dk in ["500k", "1000k"] if dk in ds_info]
        if len(owners) == 2 and owners[0] == owners[1]:
            context[cell_name]["stable_owner"] = owners[0]
        elif len(owners) == 1:
            context[cell_name]["stable_owner"] = f"{owners[0]} (one-sided)"
        else:
            context[cell_name]["stable_owner"] = "unstable"
    return context


# ── Per-dataset analysis ──

def _analyze_dataset(dataset: str) -> dict[str, Any]:
    dk = _dataset_key(dataset)
    print(f"\n{'='*60}")
    print(f"  Dataset: {dk}")
    print(f"{'='*60}")

    # Build promoted baseline (F + G + I + K) with equity coupling
    kept, baseline, classified_dynamic, dyn_time_idx, _, _ = (
        variant_k.build_variant_k_pre_coupling_kept(dataset)
    )
    coupled = merged_engine._apply_shared_equity_coupling(
        sorted(kept, key=lambda t: (t.exit_time, t.entry_time)),
        STARTING_EQUITY,
        v14_max_units=baseline["v14_max_units"],
    )

    # Isolate V14 trades
    v14_trades = [t for t in coupled if t.strategy == "v14"]
    print(f"  V14 trades: {len(v14_trades)}")

    wins = [t for t in v14_trades if t.pips > 0]
    losses = [t for t in v14_trades if t.pips <= 0]
    total_pips = sum(t.pips for t in v14_trades)
    total_usd = sum(t.usd for t in v14_trades)
    print(f"  Wins: {len(wins)}, Losses: {len(losses)}, WR: {len(wins)/len(v14_trades)*100:.1f}%")
    print(f"  Net pips: {total_pips:.1f}, Net USD: ${total_usd:,.2f}")

    # Classify each V14 trade into regime/ER/ΔER cell
    cell_trades: dict[str, list[merged_engine.TradeRow]] = defaultdict(list)
    trade_cells: list[dict[str, Any]] = []

    for trade in v14_trades:
        regime_info = variant_i._lookup_regime_with_dynamic(
            classified_dynamic, dyn_time_idx, trade.entry_time,
        )
        idx = dyn_time_idx.get_indexer([pd.Timestamp(trade.entry_time)], method="ffill")[0]
        full_row = classified_dynamic.iloc[idx]
        er = float(full_row.get("sf_er", 0.5))
        if np.isnan(er):
            er = 0.5

        er_b = _er_bucket(er)
        der_b = _der_bucket(regime_info["delta_er"])
        cell = _cell_key(regime_info["regime_label"], er_b, der_b)

        cell_trades[cell].append(trade)
        trade_cells.append({
            "entry_time": trade.entry_time.isoformat(),
            "cell": cell,
            "pips": float(trade.pips),
            "usd": float(trade.usd),
            "er": round(er, 4),
            "delta_er": round(regime_info["delta_er"], 4),
        })

    # Compute per-cell metrics
    cell_metrics: dict[str, dict[str, Any]] = {}
    for cell, trades in sorted(cell_trades.items()):
        n = len(trades)
        w = sum(1 for t in trades if t.pips > 0)
        l = n - w
        avg_pips = sum(t.pips for t in trades) / n
        net_pips = sum(t.pips for t in trades)
        avg_usd = sum(t.usd for t in trades) / n
        net_usd = sum(t.usd for t in trades)
        cell_metrics[cell] = {
            "count": n,
            "wins": w,
            "losses": l,
            "win_rate": round(w / n * 100, 1),
            "avg_pips": round(avg_pips, 2),
            "net_pips": round(net_pips, 2),
            "avg_usd": round(avg_usd, 2),
            "net_usd": round(net_usd, 2),
        }

    # Print cell breakdown
    print(f"\n  Cell breakdown ({len(cell_metrics)} cells):")
    for cell, m in sorted(cell_metrics.items(), key=lambda x: x[1]["net_pips"]):
        marker = " *** NEGATIVE" if m["avg_pips"] < 0 else ""
        print(f"    {cell:40s}  n={m['count']:3d}  WR={m['win_rate']:5.1f}%  "
              f"avg={m['avg_pips']:+7.2f}p  net={m['net_pips']:+8.2f}p  "
              f"${m['net_usd']:+10,.2f}{marker}")

    return {
        "total_v14_trades": len(v14_trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(v14_trades) * 100, 1),
        "net_pips": round(total_pips, 2),
        "net_usd": round(total_usd, 2),
        "by_cell": cell_metrics,
        "trade_detail": trade_cells,
    }


# ── Cross-dataset comparison ──

def _cross_dataset(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    all_cells = sorted(set(
        list(results["500k"]["by_cell"].keys()) +
        list(results["1000k"]["by_cell"].keys())
    ))

    cross: dict[str, dict[str, Any]] = {}
    for cell in all_cells:
        m500 = results["500k"]["by_cell"].get(cell)
        m1000 = results["1000k"]["by_cell"].get(cell)

        entry: dict[str, Any] = {"cell": cell}
        if m500:
            entry["500k"] = {
                "count": m500["count"],
                "avg_pips": m500["avg_pips"],
                "net_pips": m500["net_pips"],
                "avg_usd": m500["avg_usd"],
                "net_usd": m500["net_usd"],
                "win_rate": m500["win_rate"],
            }
        else:
            entry["500k"] = None

        if m1000:
            entry["1000k"] = {
                "count": m1000["count"],
                "avg_pips": m1000["avg_pips"],
                "net_pips": m1000["net_pips"],
                "avg_usd": m1000["avg_usd"],
                "net_usd": m1000["net_usd"],
                "win_rate": m1000["win_rate"],
            }
        else:
            entry["1000k"] = None

        neg_500 = m500 is not None and m500["avg_pips"] < 0
        neg_1000 = m1000 is not None and m1000["avg_pips"] < 0
        pos_500 = m500 is not None and m500["avg_pips"] > 0
        pos_1000 = m1000 is not None and m1000["avg_pips"] > 0

        entry["stable_negative_both"] = neg_500 and neg_1000
        entry["stable_positive_both"] = pos_500 and pos_1000

        cross[cell] = entry

    return cross


def _find_candidates(
    cross: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    strict: list[dict[str, Any]] = []
    watchlist: list[dict[str, Any]] = []

    for cell, info in cross.items():
        if not info["stable_negative_both"]:
            continue
        d500 = info["500k"]
        d1000 = info["1000k"]
        if d500 is None or d1000 is None:
            continue

        candidate = {
            "cell": cell,
            "500k_count": d500["count"],
            "500k_avg_pips": d500["avg_pips"],
            "500k_net_pips": d500["net_pips"],
            "500k_net_usd": d500["net_usd"],
            "1000k_count": d1000["count"],
            "1000k_avg_pips": d1000["avg_pips"],
            "1000k_net_pips": d1000["net_pips"],
            "1000k_net_usd": d1000["net_usd"],
            "total_saveable_pips": round(d500["net_pips"] + d1000["net_pips"], 2),
            "total_saveable_usd": round(d500["net_usd"] + d1000["net_usd"], 2),
        }

        if d500["count"] >= 3 and d1000["count"] >= 3:
            strict.append(candidate)
        elif d500["count"] >= 2 and d1000["count"] >= 3:
            watchlist.append(candidate)

    strict.sort(key=lambda x: x["total_saveable_pips"])
    watchlist.sort(key=lambda x: x["total_saveable_pips"])
    return strict, watchlist


# ── Loss concentration ──

def _loss_concentration(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    concentration: dict[str, Any] = {}
    for dk in ["500k", "1000k"]:
        by_cell = results[dk]["by_cell"]
        total_losses = results[dk]["losses"]

        # Rank cells by number of V14 losses
        cell_losses = []
        for cell, m in by_cell.items():
            if m["losses"] > 0:
                cell_losses.append({
                    "cell": cell,
                    "losses": m["losses"],
                    "loss_pips": round(
                        sum(t["pips"] for t in results[dk]["trade_detail"]
                            if t["cell"] == cell and t["pips"] <= 0), 2),
                })
        cell_losses.sort(key=lambda x: x["losses"], reverse=True)

        top1 = sum(c["losses"] for c in cell_losses[:1])
        top2 = sum(c["losses"] for c in cell_losses[:2])
        top3 = sum(c["losses"] for c in cell_losses[:3])

        concentration[dk] = {
            "total_v14_losses": total_losses,
            "top_1_cell_losses": top1,
            "top_1_pct_of_total": round(top1 / total_losses * 100, 1) if total_losses > 0 else 0,
            "top_2_cells_losses": top2,
            "top_2_pct_of_total": round(top2 / total_losses * 100, 1) if total_losses > 0 else 0,
            "top_3_cells_losses": top3,
            "top_3_pct_of_total": round(top3 / total_losses * 100, 1) if total_losses > 0 else 0,
            "cells_with_losses": len(cell_losses),
            "ranked_loss_cells": cell_losses,
        }

    return concentration


# ── Ownership context ──

def _add_ownership_context(
    strict: list[dict[str, Any]],
    watchlist: list[dict[str, Any]],
    ownership: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    context: dict[str, Any] = {}
    for candidate in strict + watchlist:
        cell = candidate["cell"]
        if cell in ownership:
            ctx = ownership[cell]
            context[cell] = {
                "stable_owner": ctx.get("stable_owner", "unknown"),
                "500k_owner": ctx.get("500k", {}).get("owner", "no data"),
                "1000k_owner": ctx.get("1000k", {}).get("owner", "no data"),
            }
        else:
            context[cell] = {"stable_owner": "no data in ownership diagnostic"}
    return context


# ── Verdict ──

def _verdict(
    strict: list[dict[str, Any]],
    watchlist: list[dict[str, Any]],
    concentration: dict[str, Any],
    results: dict[str, dict[str, Any]],
) -> str:
    lines: list[str] = []

    if strict:
        lines.append(f"STRICT CANDIDATES FOUND: {len(strict)} cell(s) with V14 avg_pips < 0 "
                      f"on both datasets and ≥3 trades on each.")
        total_saveable = sum(c["total_saveable_pips"] for c in strict)
        lines.append(f"Total saveable pips across strict candidates: {total_saveable:.1f}")
        lines.append("")
        lines.append("V14 chart-gating branch is WORTH TESTING.")
        lines.append("Next step: build a Variant M backtest that blocks V14 in these cells "
                      "and measures integrated impact.")
    elif watchlist:
        lines.append(f"No strict candidates, but {len(watchlist)} WATCHLIST cell(s) found "
                      f"(≥2 trades on 500k, ≥3 on 1000k, negative avg on both).")
        total_saveable = sum(c["total_saveable_pips"] for c in watchlist)
        lines.append(f"Total saveable pips across watchlist: {total_saveable:.1f}")
        lines.append("")
        lines.append("V14 chart-gating is MARGINAL. A single watchlist cell may be worth "
                      "testing if the saveable pips are meaningful relative to V14's total net.")
    else:
        lines.append("NO CANDIDATES FOUND.")
        lines.append("")

    # Concentration assessment
    for dk in ["500k", "1000k"]:
        c = concentration[dk]
        lines.append(f"{dk}: V14 has {c['total_v14_losses']} losses across {c['cells_with_losses']} cells. "
                     f"Top 3 cells contain {c['top_3_pct_of_total']:.0f}% of losses.")

    lines.append("")

    # Final call
    if not strict and not watchlist:
        lines.append("VERDICT: V14 losses are diffuse across cells with no stable negative concentration. "
                      "V14 chart-gating should be CLOSED for now. "
                      "V14 is already the cleanest strategy (83%/76% WR) and its losses are noise, "
                      "not regime-addressable signal.")
    elif strict:
        total_v14_pips = results["500k"]["net_pips"] + results["1000k"]["net_pips"]
        total_saveable = sum(c["total_saveable_pips"] for c in strict)
        pct = abs(total_saveable) / abs(total_v14_pips) * 100 if total_v14_pips != 0 else 0
        lines.append(f"VERDICT: Strict candidate(s) represent {pct:.1f}% of total V14 net pips. "
                      f"Proceed to Variant M backtest.")
    else:
        total_v14_pips = results["500k"]["net_pips"] + results["1000k"]["net_pips"]
        total_saveable = sum(c["total_saveable_pips"] for c in watchlist)
        pct = abs(total_saveable) / abs(total_v14_pips) * 100 if total_v14_pips != 0 else 0
        lines.append(f"VERDICT: Watchlist candidate(s) represent {pct:.1f}% of total V14 net pips. "
                      f"Borderline — test only if concentration is strong enough.")

    return "\n".join(lines)


# ── Main ──

def main() -> None:
    results: dict[str, dict[str, Any]] = {}

    for dataset in DATASETS:
        dk = _dataset_key(dataset)
        results[dk] = _analyze_dataset(dataset)

    # Cross-dataset comparison
    cross = _cross_dataset(results)
    strict, watchlist = _find_candidates(cross)
    concentration = _loss_concentration(results)
    ownership = _load_ownership_context()
    ownership_ctx = _add_ownership_context(strict, watchlist, ownership)
    verdict_text = _verdict(strict, watchlist, concentration, results)

    # ── stdout summary ──
    print(f"\n{'='*60}")
    print("  CROSS-DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"  500k V14 trades: {results['500k']['total_v14_trades']} "
          f"({results['500k']['wins']}W/{results['500k']['losses']}L)")
    print(f"  1000k V14 trades: {results['1000k']['total_v14_trades']} "
          f"({results['1000k']['wins']}W/{results['1000k']['losses']}L)")
    print(f"\n  Strict candidates: {len(strict)}")
    for c in strict:
        print(f"    {c['cell']:40s}  "
              f"500k: n={c['500k_count']} avg={c['500k_avg_pips']:+.2f}p  "
              f"1000k: n={c['1000k_count']} avg={c['1000k_avg_pips']:+.2f}p  "
              f"saveable={c['total_saveable_pips']:+.1f}p")
    print(f"\n  Watchlist candidates: {len(watchlist)}")
    for c in watchlist:
        print(f"    {c['cell']:40s}  "
              f"500k: n={c['500k_count']} avg={c['500k_avg_pips']:+.2f}p  "
              f"1000k: n={c['1000k_count']} avg={c['1000k_avg_pips']:+.2f}p  "
              f"saveable={c['total_saveable_pips']:+.1f}p")

    print(f"\n  Loss concentration:")
    for dk in ["500k", "1000k"]:
        c = concentration[dk]
        print(f"    {dk}: {c['total_v14_losses']} losses in {c['cells_with_losses']} cells, "
              f"top 3 = {c['top_3_pct_of_total']:.0f}%")

    if strict or watchlist:
        print(f"\n  Top negative cells by saveable pips:")
        all_cands = sorted(strict + watchlist, key=lambda x: x["total_saveable_pips"])
        for c in all_cands[:5]:
            owner_info = ownership_ctx.get(c["cell"], {}).get("stable_owner", "?")
            print(f"    {c['cell']:40s}  saveable={c['total_saveable_pips']:+.1f}p  "
                  f"owner={owner_info}")

    print(f"\n  VERDICT:")
    for line in verdict_text.split("\n"):
        print(f"    {line}")

    # ── Build output JSON ──
    # Strip trade_detail from dataset_summaries (keep cell-level only)
    dataset_summaries = {}
    for dk in ["500k", "1000k"]:
        summary = dict(results[dk])
        del summary["trade_detail"]
        dataset_summaries[dk] = summary

    output = {
        "dataset_summaries": dataset_summaries,
        "cross_dataset_cells": cross,
        "strict_candidates": strict,
        "watchlist_candidates": watchlist,
        "loss_concentration": concentration,
        "ownership_context": ownership_ctx,
        "verdict": verdict_text,
    }

    out_path = OUT_DIR / "diagnostic_v14_regime_ownership.json"
    out_path.write_text(
        json.dumps(output, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(f"\n  Output: {out_path}")


if __name__ == "__main__":
    main()
