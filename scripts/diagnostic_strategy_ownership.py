#!/usr/bin/env python3
"""
Strategy Ownership Diagnostic — Empirical Expected-Value Tables.

Inverts the question from "should this strategy be blocked?" to
"given the chart state right now, which strategy has positive expected value?"

For every trade across all three strategies:
  1. Capture the chart condition at entry (regime label, ER, ΔER, session hour)
  2. Bucket into coarse condition groups
  3. For each bucket × strategy: compute avg pips, PF, win rate
  4. Identify ownership: who has the best edge in each bucket?
  5. Identify no-trade buckets: where ALL strategies have negative expected pips

Coarse buckets (keep it simple, maximize trades per cell):
  - Regime context: momentum, mean_reversion, breakout, post_breakout_trend, ambiguous
  - ER level: low (<0.35), mid (0.35-0.55), high (>=0.55)
  - ΔER direction: deteriorating (<0), stable/improving (>=0)

This produces a 5 × 3 × 2 = 30-cell grid. Each cell shows expected pips
for V14, London V2, and V44. The cell's "owner" is the strategy with the
highest positive expected pips. If all are negative, it's a no-trade cell.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/codygnon/Documents/usdjpy_assistant")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.regime_features import (
    compute_delta_efficiency_ratio,
    compute_efficiency_ratio,
    compute_level_touch_density,
)
from core.regime_classifier import RegimeThresholds
from scripts import validate_regime_classifier as regime_validation
from scripts.backtest_v44_conservative_router import _lookup_regime


# ── Data loading (reuse patterns from no-trade diagnostic) ──

def _build_m5(input_csv: str) -> pd.DataFrame:
    from scripts.regime_threshold_analysis import _load_m1, _resample
    m1 = _load_m1(input_csv)
    return _resample(m1, "5min")


def _compute_dynamic_features_on_m5(m5: pd.DataFrame) -> pd.DataFrame:
    n = len(m5)
    er_vals = np.full(n, 0.5)
    delta_er_vals = np.full(n, 0.0)

    for i in range(n):
        window = m5.iloc[max(0, i - 60):i + 1]
        if len(window) < 5:
            continue
        er_vals[i] = compute_efficiency_ratio(window, lookback=12)
        delta_er_vals[i] = compute_delta_efficiency_ratio(window, lookback=12, delta_bars=3)

    out = m5.copy()
    out["sf_er"] = er_vals
    out["sf_delta_er"] = delta_er_vals
    return out


def _load_classified_with_dynamic(input_csv: str) -> pd.DataFrame:
    m1 = regime_validation._load_m1(input_csv)
    featured = regime_validation.compute_features(m1)
    classified = regime_validation.classify_all_bars(featured, RegimeThresholds())
    m5_dynamic = _compute_dynamic_features_on_m5(_build_m5(input_csv))
    dynamic_cols = ["time", "sf_er", "sf_delta_er"]
    return pd.merge_asof(
        classified.sort_values("time"),
        m5_dynamic[dynamic_cols].sort_values("time"),
        on="time",
        direction="backward",
    )


def _lookup_row(classified: pd.DataFrame, ts: pd.Timestamp, time_idx: pd.DatetimeIndex) -> pd.Series | None:
    idx = time_idx.get_indexer([pd.Timestamp(ts)], method="ffill")[0]
    if idx < 0:
        return None
    return classified.iloc[idx]


# ── Bucketing ──

def _er_bucket(er: float) -> str:
    if er < 0.35:
        return "er_low"
    elif er < 0.55:
        return "er_mid"
    else:
        return "er_high"


def _der_bucket(delta_er: float) -> str:
    return "der_neg" if delta_er < 0 else "der_pos"


REGIME_LABELS = ["momentum", "mean_reversion", "breakout", "post_breakout_trend", "ambiguous"]
ER_BUCKETS = ["er_low", "er_mid", "er_high"]
DER_BUCKETS = ["der_neg", "der_pos"]
STRATEGIES = ["v14", "london_v2", "v44_ny"]


def run_diagnostic(dataset_label: str, input_csv: str, baseline_json: str) -> dict:
    print(f"\n{'='*70}")
    print(f"  STRATEGY OWNERSHIP DIAGNOSTIC: {dataset_label}")
    print(f"{'='*70}")

    with open(baseline_json, encoding="utf-8") as f:
        baseline = json.load(f)

    trades = baseline["closed_trades"]
    print(f"Total trades: {len(trades)}")

    print("Building classified bars with dynamic features...")
    classified = _load_classified_with_dynamic(input_csv)
    time_idx = pd.DatetimeIndex(classified["time"])

    # ── Tag every trade with chart condition ──
    tagged_trades = []
    for t in trades:
        entry_ts = pd.Timestamp(t["entry_time"])
        regime = _lookup_regime(classified, entry_ts)
        row = _lookup_row(classified, entry_ts, time_idx)

        label = regime["regime_label"]
        scores = regime["regime_scores"]
        top_regime = max(scores, key=scores.get) if scores else "unknown"

        def _safe(val, default=0.0):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return default
            return float(val)

        er = _safe(row.get("sf_er") if row is not None else None, 0.5)
        delta_er = _safe(row.get("sf_delta_er") if row is not None else None, 0.0)

        tagged_trades.append({
            "strategy": t["strategy"],
            "pips": t["pips"],
            "usd": t["usd"],
            "side": t["side"],
            "exit_reason": t["exit_reason"],
            "regime_label": label,
            "top_regime": top_regime,
            "er": er,
            "delta_er": delta_er,
            "er_bucket": _er_bucket(er),
            "der_bucket": _der_bucket(delta_er),
            "entry_hour": entry_ts.hour,
        })

    print(f"Tagged: {len(tagged_trades)} trades")

    # ── Build the ownership grid ──
    # Key: (regime_label, er_bucket, der_bucket, strategy) → list of pips
    grid: dict[tuple, list[float]] = defaultdict(list)
    for t in tagged_trades:
        key = (t["regime_label"], t["er_bucket"], t["der_bucket"], t["strategy"])
        grid[key].append(t["pips"])

    # ── 1. Full grid table ──
    print(f"\n{'─'*90}")
    print(f"  OWNERSHIP GRID: Expected pips per trade by condition × strategy")
    print(f"{'─'*90}")

    header = f"  {'Condition':<40s}"
    for strat in STRATEGIES:
        header += f" {strat:>12s}"
    header += f" {'OWNER':>12s} {'NO-TRADE':>9s}"
    print(header)
    print(f"  {'─'*88}")

    grid_results = {}
    no_trade_cells = []
    ownership_counts = defaultdict(int)
    total_cells_with_data = 0

    for regime in REGIME_LABELS:
        for er_b in ER_BUCKETS:
            for der_b in DER_BUCKETS:
                condition = f"{regime}/{er_b}/{der_b}"
                cell_data = {}
                best_strat = None
                best_avg = -999
                any_data = False

                parts = []
                for strat in STRATEGIES:
                    key = (regime, er_b, der_b, strat)
                    pips_list = grid.get(key, [])
                    if pips_list:
                        any_data = True
                        avg = np.mean(pips_list)
                        n = len(pips_list)
                        wins = sum(1 for p in pips_list if p > 0)
                        wr = 100.0 * wins / n
                        net = sum(pips_list)
                        cell_data[strat] = {
                            "count": n,
                            "avg_pips": round(avg, 1),
                            "net_pips": round(net, 1),
                            "win_rate": round(wr, 1),
                        }
                        parts.append(f"{avg:>+7.1f}({n:>3d})")
                        if avg > best_avg:
                            best_avg = avg
                            best_strat = strat
                    else:
                        cell_data[strat] = {"count": 0}
                        parts.append(f"{'---':>12s}")

                if not any_data:
                    continue

                total_cells_with_data += 1
                is_no_trade = best_avg <= 0
                owner = "NO-TRADE" if is_no_trade else best_strat

                if is_no_trade:
                    no_trade_cells.append(condition)
                else:
                    ownership_counts[best_strat] += 1

                row_str = f"  {condition:<40s}"
                for strat in STRATEGIES:
                    key = (regime, er_b, der_b, strat)
                    pips_list = grid.get(key, [])
                    if pips_list:
                        avg = np.mean(pips_list)
                        n = len(pips_list)
                        marker = " *" if strat == best_strat and not is_no_trade else ""
                        row_str += f" {avg:>+6.1f}({n:>3d}){marker}"
                    else:
                        row_str += f" {'---':>12s}"
                row_str += f" {owner:>12s}"
                row_str += f" {'YES' if is_no_trade else '':>9s}"
                print(row_str)

                grid_results[condition] = {
                    "strategies": cell_data,
                    "owner": owner,
                    "no_trade": is_no_trade,
                }

    # ── 2. Ownership summary ──
    print(f"\n{'─'*70}")
    print(f"  OWNERSHIP SUMMARY")
    print(f"{'─'*70}")

    print(f"\n  Cells with data: {total_cells_with_data}")
    print(f"  No-trade cells:  {len(no_trade_cells)}")
    for strat in STRATEGIES:
        print(f"  {strat} owns:     {ownership_counts.get(strat, 0)} cells")

    # ── 3. No-trade cell detail ──
    if no_trade_cells:
        print(f"\n{'─'*70}")
        print(f"  NO-TRADE CELLS (all strategies negative)")
        print(f"{'─'*70}")

        no_trade_trades = 0
        no_trade_pips = 0.0
        for condition in no_trade_cells:
            parts = condition.split("/")
            regime, er_b, der_b = parts[0], parts[1], parts[2]
            cell_trades = []
            for strat in STRATEGIES:
                key = (regime, er_b, der_b, strat)
                pips_list = grid.get(key, [])
                cell_trades.extend(pips_list)
            n = len(cell_trades)
            net = sum(cell_trades)
            no_trade_trades += n
            no_trade_pips += net
            print(f"  {condition}: {n} trades, {net:+.1f} pips")

        print(f"\n  Total no-trade trades: {no_trade_trades}")
        print(f"  Total no-trade pips:   {no_trade_pips:+.1f}")
        print(f"  (Standing down would have saved {-no_trade_pips:+.1f} pips)")

    # ── 4. Simplified 2D view: regime × ER only ──
    print(f"\n{'─'*90}")
    print(f"  SIMPLIFIED VIEW: regime × ER (ignoring ΔER)")
    print(f"{'─'*90}")

    header2 = f"  {'Regime × ER':<30s}"
    for strat in STRATEGIES:
        header2 += f" {strat:>15s}"
    header2 += f" {'OWNER':>10s}"
    print(header2)
    print(f"  {'─'*85}")

    simple_grid_results = {}
    for regime in REGIME_LABELS:
        for er_b in ER_BUCKETS:
            condition = f"{regime}/{er_b}"
            parts = []
            best_strat = None
            best_avg = -999
            any_data = False

            for strat in STRATEGIES:
                all_pips = []
                for der_b in DER_BUCKETS:
                    key = (regime, er_b, der_b, strat)
                    all_pips.extend(grid.get(key, []))
                if all_pips:
                    any_data = True
                    avg = np.mean(all_pips)
                    n = len(all_pips)
                    parts.append((strat, avg, n))
                    if avg > best_avg:
                        best_avg = avg
                        best_strat = strat
                else:
                    parts.append((strat, None, 0))

            if not any_data:
                continue

            is_no_trade = best_avg <= 0
            owner = "NO-TRADE" if is_no_trade else best_strat

            row_str = f"  {condition:<30s}"
            for strat, avg, n in parts:
                if avg is not None:
                    marker = " *" if strat == best_strat and not is_no_trade else ""
                    row_str += f" {avg:>+6.1f}({n:>3d}){marker}"
                else:
                    row_str += f" {'---':>15s}"
            row_str += f" {owner:>10s}"
            print(row_str)

            simple_grid_results[condition] = {
                "owner": owner,
                "no_trade": is_no_trade,
            }

    # ── 5. Misallocation analysis ──
    # Trades taken by the WRONG strategy according to the ownership grid
    print(f"\n{'─'*70}")
    print(f"  MISALLOCATION ANALYSIS")
    print(f"{'─'*70}")

    correct_owner = 0
    wrong_owner = 0
    no_trade_violation = 0
    correct_pips = 0.0
    wrong_pips = 0.0
    no_trade_violation_pips = 0.0

    for t in tagged_trades:
        condition = f"{t['regime_label']}/{t['er_bucket']}/{t['der_bucket']}"
        cell = grid_results.get(condition)
        if cell is None:
            continue

        if cell["no_trade"]:
            no_trade_violation += 1
            no_trade_violation_pips += t["pips"]
        elif cell["owner"] == t["strategy"]:
            correct_owner += 1
            correct_pips += t["pips"]
        else:
            wrong_owner += 1
            wrong_pips += t["pips"]

    total_classified = correct_owner + wrong_owner + no_trade_violation
    print(f"\n  Trades in correct-owner cell: {correct_owner} ({correct_pips:+.1f} pips, avg {correct_pips/max(1,correct_owner):+.1f})")
    print(f"  Trades in wrong-owner cell:   {wrong_owner} ({wrong_pips:+.1f} pips, avg {wrong_pips/max(1,wrong_owner):+.1f})")
    print(f"  Trades in no-trade cell:      {no_trade_violation} ({no_trade_violation_pips:+.1f} pips, avg {no_trade_violation_pips/max(1,no_trade_violation):+.1f})")

    if no_trade_violation > 0:
        print(f"\n  If no-trade cells were enforced: {-no_trade_violation_pips:+.1f} pips saved, {no_trade_violation} trades avoided")

    # ── 6. Per-strategy ownership profile ──
    print(f"\n{'─'*70}")
    print(f"  PER-STRATEGY OWNERSHIP PROFILE")
    print(f"{'─'*70}")

    for strat in STRATEGIES:
        strat_trades = [t for t in tagged_trades if t["strategy"] == strat]
        in_owned = [t for t in strat_trades
                    if grid_results.get(f"{t['regime_label']}/{t['er_bucket']}/{t['der_bucket']}", {}).get("owner") == strat]
        in_other = [t for t in strat_trades
                    if grid_results.get(f"{t['regime_label']}/{t['er_bucket']}/{t['der_bucket']}", {}).get("owner") not in (strat, "NO-TRADE", None)]
        in_notrade = [t for t in strat_trades
                      if grid_results.get(f"{t['regime_label']}/{t['er_bucket']}/{t['der_bucket']}", {}).get("no_trade")]

        print(f"\n  {strat} ({len(strat_trades)} total trades):")
        if in_owned:
            avg_owned = np.mean([t["pips"] for t in in_owned])
            print(f"    In own cells:     {len(in_owned)} trades, {sum(t['pips'] for t in in_owned):+.1f} pips, avg {avg_owned:+.1f}")
        if in_other:
            avg_other = np.mean([t["pips"] for t in in_other])
            print(f"    In other's cells: {len(in_other)} trades, {sum(t['pips'] for t in in_other):+.1f} pips, avg {avg_other:+.1f}")
        if in_notrade:
            avg_nt = np.mean([t["pips"] for t in in_notrade])
            print(f"    In no-trade cells:{len(in_notrade)} trades, {sum(t['pips'] for t in in_notrade):+.1f} pips, avg {avg_nt:+.1f}")

    return {
        "total_trades": len(tagged_trades),
        "cells_with_data": total_cells_with_data,
        "no_trade_cells": len(no_trade_cells),
        "no_trade_cell_list": no_trade_cells,
        "ownership_counts": dict(ownership_counts),
        "grid": grid_results,
        "simple_grid": simple_grid_results,
        "misallocation": {
            "correct_owner": correct_owner,
            "correct_pips": round(correct_pips, 1),
            "wrong_owner": wrong_owner,
            "wrong_pips": round(wrong_pips, 1),
            "no_trade_violations": no_trade_violation,
            "no_trade_violation_pips": round(no_trade_violation_pips, 1),
        },
    }


def main() -> int:
    datasets = [
        (
            "500k",
            str(ROOT / "research_out" / "USDJPY_M1_OANDA_500k.csv"),
            str(ROOT / "research_out" / "v44_conservative_router_aligned_500k_baseline.json"),
        ),
        (
            "1000k",
            str(ROOT / "research_out" / "USDJPY_M1_OANDA_1000k.csv"),
            str(ROOT / "research_out" / "v44_conservative_router_aligned_1000k_baseline.json"),
        ),
    ]

    all_results = {}
    for label, csv_path, baseline_path in datasets:
        if not Path(csv_path).exists():
            print(f"Skipping {label}: {csv_path} not found")
            continue
        if not Path(baseline_path).exists():
            print(f"Skipping {label}: {baseline_path} not found")
            continue
        all_results[label] = run_diagnostic(label, csv_path, baseline_path)

    out_path = ROOT / "research_out" / "diagnostic_strategy_ownership.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults written to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
