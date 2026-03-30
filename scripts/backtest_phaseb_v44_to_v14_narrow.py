#!/usr/bin/env python3
"""
Phase B: narrow V44 -> V14 chart-first override test.

This is a strict timestamp-level substitution test:
  - build the current promoted Variant K baseline
  - find V44 trades in stable V14-owned cells
  - remove those specific V44 trades
  - only add a replacement if existing V14 NY logic fires at the exact same
    entry timestamp
  - re-couple and compare to the baseline

The purpose is to answer one narrow question:
When the ownership table says V14 should own instead of V44, does real existing
V14 entry logic actually fire and improve the result at those exact moments?
"""
from __future__ import annotations

import json
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.ownership_table import cell_key
from scripts import backtest_merged_integrated_tokyo_london_v2_ny as merged_engine
from scripts import backtest_variant_i_pbt_standdown as variant_i
from scripts import backtest_variant_k_london_cluster as variant_k

OUT_DIR = ROOT / "research_out"
STARTING_EQUITY = 100_000.0
V14_CFG_PATH = OUT_DIR / "tokyo_optimized_v14_config.json"
PHASE_A_PATH = OUT_DIR / "diagnostic_chart_first_routing.json"
OWNERSHIP_STABILITY_PATH = OUT_DIR / "diagnostic_ownership_stability.json"
MAX_CANDIDATE_CELLS = 3

DATASETS = [
    str(OUT_DIR / "USDJPY_M1_OANDA_500k.csv"),
    str(OUT_DIR / "USDJPY_M1_OANDA_1000k.csv"),
]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return str(obj)


def _dataset_key(path: str) -> str:
    name = Path(path).name
    if "500k" in name:
        return "500k"
    if "1000k" in name:
        return "1000k"
    raise ValueError(f"Unknown dataset: {name}")


def _get_trade_cell(
    trade: merged_engine.TradeRow,
    classified_dynamic: pd.DataFrame,
    dyn_time_idx: pd.DatetimeIndex,
) -> tuple[str, str, str]:
    regime_info = variant_i._lookup_regime_with_dynamic(
        classified_dynamic, dyn_time_idx, trade.entry_time,
    )
    idx = dyn_time_idx.get_indexer([pd.Timestamp(trade.entry_time)], method="ffill")[0]
    full_row = classified_dynamic.iloc[idx]
    er = float(full_row.get("sf_er", 0.5))
    if pd.isna(er):
        er = 0.5
    return (
        regime_info["regime_label"],
        variant_k._er_bucket(er),
        variant_k._der_bucket(regime_info["delta_er"]),
    )


def _build_ny_v14_config(dataset: str) -> Path:
    cfg = json.loads(V14_CFG_PATH.read_text(encoding="utf-8"))
    cfg["session_filter"]["session_start_utc"] = "13:00"
    cfg["session_filter"]["session_end_utc"] = "16:00"
    cfg["session_filter"]["allowed_trading_days"] = [
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
    ]
    cfg["trade_management"]["disable_entries_if_move_from_tokyo_open_range_exceeds_pips"] = 0.0
    cfg["trade_management"]["breakout_detection_mode"] = "disabled"

    run0 = cfg.get("run_sequence", [{}])[0]
    run0["label"] = "phaseb_v14_ny_exact"
    run0["input_csv"] = dataset
    run0["output_json"] = ""
    run0["output_trades_csv"] = ""
    run0["output_equity_csv"] = ""
    cfg["run_sequence"] = [run0]

    td = tempfile.mkdtemp(prefix="phaseb_v14_ny_")
    out = Path(td) / "v14_ny_config.json"
    out.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return out


def _run_ny_v14(dataset: str) -> list[merged_engine.TradeRow]:
    cfg_path = _build_ny_v14_config(dataset)
    ny_report, _ = merged_engine._run_v14_in_process(cfg_path, dataset)
    trades = merged_engine._extract_v14_trades(ny_report, STARTING_EQUITY)
    out: list[merged_engine.TradeRow] = []
    for t in trades:
        out.append(
            merged_engine.TradeRow(
                strategy="v14_ny_override",
                entry_time=t.entry_time,
                exit_time=t.exit_time,
                entry_session="ny",
                side=t.side,
                pips=t.pips,
                usd=t.usd,
                exit_reason=t.exit_reason,
                standalone_entry_equity=t.standalone_entry_equity,
                raw=t.raw,
                size_scale=t.size_scale,
            )
        )
    return out


def _trade_stats(trades: list[merged_engine.TradeRow]) -> dict[str, Any]:
    if not trades:
        return {
            "count": 0,
            "wins": 0,
            "losses": 0,
            "win_rate_pct": 0.0,
            "avg_pips": 0.0,
            "net_pips": 0.0,
            "avg_usd": 0.0,
            "net_usd": 0.0,
        }
    wins = sum(1 for t in trades if t.pips > 0)
    n = len(trades)
    return {
        "count": n,
        "wins": wins,
        "losses": n - wins,
        "win_rate_pct": round(wins / n * 100.0, 1),
        "avg_pips": round(sum(t.pips for t in trades) / n, 2),
        "net_pips": round(sum(t.pips for t in trades), 2),
        "avg_usd": round(sum(t.usd for t in trades) / n, 2),
        "net_usd": round(sum(t.usd for t in trades), 2),
    }


def _load_candidate_cells() -> dict[str, Any]:
    phase_a = json.loads(PHASE_A_PATH.read_text(encoding="utf-8"))
    stability = json.loads(OWNERSHIP_STABILITY_PATH.read_text(encoding="utf-8"))

    v44_to_v14_total = 0
    for dataset_rows in phase_a["disagreement_pair_breakdown"].values():
        for row in dataset_rows:
            if row["pair"] == "v44_ny -> v14":
                v44_to_v14_total += int(row["count"])
    if v44_to_v14_total <= 0:
        raise RuntimeError("Phase A has no v44_ny -> v14 disagreement pairs.")

    ranked = []
    for row in stability.get("opportunities", []):
        if row.get("owner") != "v14":
            continue
        if row.get("blocked_strategy") != "v44_ny":
            continue
        ranked.append(
            {
                "cell": row["cell"],
                "recommended_owner": "v14",
                "count_500k": int(row.get("n_500k", 0)),
                "count_1000k": int(row.get("n_1000k", 0)),
                "count_total": int(row.get("n_500k", 0)) + int(row.get("n_1000k", 0)),
                "saveable_total_pips": round(float(row.get("total_net", 0.0)), 2),
            }
        )

    ranked.sort(
        key=lambda x: (-x["count_total"], -abs(x.get("saveable_total_pips", 0.0)), x["cell"])
    )

    selected = ranked[:MAX_CANDIDATE_CELLS]
    return {
        "selected_cells": [row["cell"] for row in selected],
        "selection_rows": selected,
        "selection_reason": (
            "Stable V14-owned cells selected from diagnostic_ownership_stability "
            "opportunities where blocked_strategy = v44_ny, ranked by combined "
            "v44->v14 disagreement count across 500k/1000k."
        ),
    }


def _analyze_dataset(dataset: str, candidate_cells: set[str]) -> dict[str, Any]:
    dk = _dataset_key(dataset)
    print(f"\n{'=' * 60}\nDataset: {dk}\n{'=' * 60}")

    kept_pre, baseline, classified_dynamic, dyn_time_idx, _, _ = variant_k.build_variant_k_pre_coupling_kept(dataset)
    baseline_coupled = merged_engine._apply_shared_equity_coupling(
        sorted(kept_pre, key=lambda t: (t.exit_time, t.entry_time)),
        STARTING_EQUITY,
        v14_max_units=baseline["v14_max_units"],
    )
    baseline_eq = merged_engine._build_equity_curve(baseline_coupled, STARTING_EQUITY)
    baseline_stats = merged_engine._stats(baseline_coupled, STARTING_EQUITY, baseline_eq)
    print(
        f"Baseline K: {baseline_stats['total_trades']} trades, "
        f"${baseline_stats['net_usd']:,.2f}, PF={baseline_stats['profit_factor']:.3f}"
    )

    removable_v44: list[tuple[merged_engine.TradeRow, str]] = []
    preserved_trades: list[merged_engine.TradeRow] = []
    cell_counter: Counter[str] = Counter()
    for trade in kept_pre:
        if trade.strategy != "v44_ny":
            preserved_trades.append(trade)
            continue
        cell_str = cell_key(*_get_trade_cell(trade, classified_dynamic, dyn_time_idx))
        if cell_str in candidate_cells:
            removable_v44.append((trade, cell_str))
            cell_counter[cell_str] += 1
        else:
            preserved_trades.append(trade)

    print(f"V44 candidate-cell trades to test: {len(removable_v44)}")
    for cell_str, count in sorted(cell_counter.items()):
        print(f"  {cell_str}: {count}")

    ny_v14_all = _run_ny_v14(dataset)
    ny_v14_by_ts: dict[pd.Timestamp, list[merged_engine.TradeRow]] = defaultdict(list)
    for trade in ny_v14_all:
        ny_v14_by_ts[pd.Timestamp(trade.entry_time)].append(trade)

    replacements: list[merged_engine.TradeRow] = []
    removed_v44_trades: list[merged_engine.TradeRow] = []
    replacement_v14_trades: list[merged_engine.TradeRow] = []
    sample_rows: list[dict[str, Any]] = []
    replacement_hits = 0
    no_trade_count = 0

    for original, cell_str in removable_v44:
        removed_v44_trades.append(original)
        exact_candidates = ny_v14_by_ts.get(pd.Timestamp(original.entry_time), [])
        matched_trade = None
        for candidate in exact_candidates:
            cand_cell = cell_key(*_get_trade_cell(candidate, classified_dynamic, dyn_time_idx))
            if cand_cell == cell_str:
                matched_trade = candidate
                break

        if matched_trade is not None:
            replacements.append(matched_trade)
            replacement_v14_trades.append(matched_trade)
            replacement_hits += 1
        else:
            no_trade_count += 1

        if len(sample_rows) < 30:
            sample_rows.append(
                {
                    "original_v44_entry_time": original.entry_time.isoformat(),
                    "cell": cell_str,
                    "original_v44_side": original.side,
                    "original_v44_pips": round(float(original.pips), 2),
                    "original_v44_usd": round(float(original.usd), 2),
                    "replacement_happened": matched_trade is not None,
                    "replacement_v14_entry_time": matched_trade.entry_time.isoformat() if matched_trade else None,
                    "replacement_v14_side": matched_trade.side if matched_trade else None,
                    "replacement_v14_pips": round(float(matched_trade.pips), 2) if matched_trade else None,
                    "replacement_v14_usd": round(float(matched_trade.usd), 2) if matched_trade else None,
                }
            )

    modified_pre = preserved_trades + replacements
    modified_coupled = merged_engine._apply_shared_equity_coupling(
        sorted(modified_pre, key=lambda t: (t.exit_time, t.entry_time)),
        STARTING_EQUITY,
        v14_max_units=baseline["v14_max_units"],
    )
    modified_eq = merged_engine._build_equity_curve(modified_coupled, STARTING_EQUITY)
    modified_stats = merged_engine._stats(modified_coupled, STARTING_EQUITY, modified_eq)

    delta = {
        "total_trades": int(modified_stats["total_trades"]) - int(baseline_stats["total_trades"]),
        "net_usd": round(modified_stats["net_usd"] - baseline_stats["net_usd"], 2),
        "profit_factor": round(modified_stats["profit_factor"] - baseline_stats["profit_factor"], 4),
        "max_drawdown_usd": round(modified_stats["max_drawdown_usd"] - baseline_stats["max_drawdown_usd"], 2),
    }

    print(
        f"Modified: {modified_stats['total_trades']} trades, "
        f"${modified_stats['net_usd']:,.2f}, PF={modified_stats['profit_factor']:.3f}"
    )
    print(
        f"Delta: trades {delta['total_trades']:+d}, "
        f"net ${delta['net_usd']:+,.2f}, PF {delta['profit_factor']:+.4f}, "
        f"DD ${delta['max_drawdown_usd']:+,.2f}"
    )
    print(f"Exact replacements: {replacement_hits}, no-trade outcomes: {no_trade_count}")

    return {
        "baseline_stats": baseline_stats,
        "modified_stats": modified_stats,
        "delta_vs_baseline": delta,
        "substitution_opportunities": {
            "v44_trades_in_candidate_cells": len(removable_v44),
            "v44_removed": len(removed_v44_trades),
            "ny_v14_total_raw_trades": len(ny_v14_all),
            "ny_v14_exact_replacements": replacement_hits,
            "became_no_trade": no_trade_count,
        },
        "removed_v44_stats": _trade_stats(removed_v44_trades),
        "replacement_v14_stats": _trade_stats(replacement_v14_trades),
        "samples": sample_rows,
    }


def _build_verdict(results: dict[str, dict[str, Any]]) -> str:
    lines = ["=" * 60, "PHASE B VERDICT", "=" * 60, ""]

    repl_1000k = results["1000k"]["substitution_opportunities"]["ny_v14_exact_replacements"]
    if repl_1000k < 5:
        lines.append(
            f"QUICK CLOSE: exact V14 replacements on 1000k = {repl_1000k} (< 5)."
        )
        lines.append("Real existing V14 entry logic does not fire often enough at the exact V44 moments.")
        lines.append("Close the V44 -> V14 offensive override branch.")
        return "\n".join(lines)

    net_1000k = results["1000k"]["delta_vs_baseline"]["net_usd"]
    if net_1000k < 0:
        lines.append(
            f"QUICK CLOSE: 1000k net delta = ${net_1000k:+,.2f}."
        )
        lines.append("Replacement V14 trades are not improving the targeted override.")
        lines.append("Close the V44 -> V14 offensive override branch.")
        return "\n".join(lines)

    net_500k = results["500k"]["delta_vs_baseline"]["net_usd"]
    if net_500k > 0 and net_1000k > 0:
        lines.append("PASS: narrow V44 -> V14 override improves both datasets.")
    elif net_500k > -500 and net_1000k > 0:
        lines.append("PARTIAL PASS: strong 1000k improvement with no material 500k harm.")
    else:
        lines.append("FAIL: results do not justify promotion.")

    for dk in ["500k", "1000k"]:
        r = results[dk]
        lines.append("")
        lines.append(
            f"{dk}: removed V44={r['substitution_opportunities']['v44_removed']}, "
            f"exact V14 replacements={r['substitution_opportunities']['ny_v14_exact_replacements']}, "
            f"net delta=${r['delta_vs_baseline']['net_usd']:+,.2f}"
        )

    return "\n".join(lines)


def main() -> None:
    cell_info = _load_candidate_cells()
    selected_cells = set(cell_info["selected_cells"])

    print("Phase B: Narrow V44 -> V14 exact-timestamp override")
    print("Selected candidate cells:")
    for row in cell_info["selection_rows"]:
        print(
            f"  {row['cell']:32s}  total={row['count_total']:>3d}  "
            f"500k={row['count_500k']:>2d}  1000k={row['count_1000k']:>2d}"
        )

    results: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        dk = _dataset_key(dataset)
        results[dk] = _analyze_dataset(dataset, selected_cells)

    verdict = _build_verdict(results)
    print("\n" + verdict)

    output = {
        "candidate_cells": {
            "cells": cell_info["selected_cells"],
            "selection_rows": cell_info["selection_rows"],
            "selection_reason": cell_info["selection_reason"],
        },
        "dataset_results": {
            dk: {
                "substitution_opportunities": data["substitution_opportunities"],
                "replacement_trade_stats": data["replacement_v14_stats"],
                "removed_v44_stats": data["removed_v44_stats"],
                "baseline_summary": {
                    "total_trades": data["baseline_stats"]["total_trades"],
                    "net_usd": data["baseline_stats"]["net_usd"],
                    "profit_factor": data["baseline_stats"]["profit_factor"],
                    "max_drawdown_usd": data["baseline_stats"]["max_drawdown_usd"],
                },
                "modified_summary": {
                    "total_trades": data["modified_stats"]["total_trades"],
                    "net_usd": data["modified_stats"]["net_usd"],
                    "profit_factor": data["modified_stats"]["profit_factor"],
                    "max_drawdown_usd": data["modified_stats"]["max_drawdown_usd"],
                },
                "delta_vs_baseline": data["delta_vs_baseline"],
                "samples": data["samples"],
            }
            for dk, data in results.items()
        },
        "verdict": verdict,
    }

    out_path = OUT_DIR / "phaseb_v44_to_v14_narrow.json"
    out_path.write_text(json.dumps(output, indent=2, default=_json_default), encoding="utf-8")
    print(f"\nOutput: {out_path}")


if __name__ == "__main__":
    main()
