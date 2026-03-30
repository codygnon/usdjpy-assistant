#!/usr/bin/env python3
"""
V6 experiment: sweep bb_width_exp_rate thresholds for the V44 momentum scorer.

Loads data and runs backtests ONCE per dataset, then re-classifies for each
threshold variant.  Produces per-variant JSON reports and a compact comparison
summary to stdout.

Usage:
    python scripts/sweep_v6_v44_momentum.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.regime_classifier import RegimeThresholds
from scripts.validate_regime_classifier import (
    compute_features,
    classify_all_bars,
    label_trades,
    build_report,
)
from scripts.regime_threshold_analysis import (
    _load_m1,
    _run_v14,
    _run_london_v2,
    _run_v44,
)


DATASETS = [
    ("250k", "research_out/USDJPY_M1_OANDA_200k_split.csv"),
    ("500k", "research_out/USDJPY_M1_OANDA_500k_train_from_1000k.csv"),
    ("1000k", "research_out/USDJPY_M1_OANDA_800k_split.csv"),
]

# Each entry: (label, exp_rate_min, exp_rate_max)  — None means v5 baseline
VARIANTS = [
    ("v5_baseline", None, None),
    ("band_0.93_1.10", 0.93, 1.10),
    ("band_0.95_1.08", 0.95, 1.08),
    ("band_0.95_1.10", 0.95, 1.10),
    ("band_0.95_1.12", 0.95, 1.12),
    ("band_0.95_1.15", 0.95, 1.15),
    ("band_0.97_1.10", 0.97, 1.10),
]


def _wr(trades: list[dict]) -> float:
    if not trades:
        return 0.0
    return round(100.0 * sum(1 for t in trades if t["profitable"]) / len(trades), 1)


def run_dataset(label: str, csv_path: str, out_dir: Path) -> list[dict]:
    csv_abs = str((ROOT / csv_path).resolve())
    print(f"\n{'='*70}")
    print(f"  Dataset: {label}  ({csv_path})")
    print(f"{'='*70}")

    t0 = time.time()
    print("  [1/4] Loading M1 data ...")
    m1 = _load_m1(csv_abs)
    print(f"         {len(m1)} bars")

    print("  [2/4] Computing regime features ...")
    featured = compute_features(m1)

    print("  [3/4] Running standalone backtests (once) ...")
    print("         V14 ...")
    v14_trades = _run_v14(csv_abs)
    print(f"         → {len(v14_trades)} trades")
    print("         London V2 ...")
    v2_trades = _run_london_v2(m1)
    print(f"         → {len(v2_trades)} trades")
    print("         V44 NY ...")
    v44_trades = _run_v44(csv_abs)
    print(f"         → {len(v44_trades)} trades")
    all_trades = v14_trades + v2_trades + v44_trades
    shared_time = time.time() - t0
    print(f"         shared work: {shared_time:.1f}s")

    results = []
    print(f"  [4/4] Sweeping {len(VARIANTS)} variants ...")
    for variant_label, rate_min, rate_max in VARIANTS:
        th = RegimeThresholds()
        if rate_max is not None:
            th.momentum_bb_exp_rate_max = rate_max
            th.momentum_bb_exp_rate_min = rate_min

        classified = classify_all_bars(featured, th)
        enriched = label_trades(classified, [dict(t) for t in all_trades])
        report = build_report(classified, enriched, csv_abs)
        report["_meta"]["variant"] = variant_label
        report["_meta"]["momentum_bb_exp_rate_min"] = rate_min
        report["_meta"]["momentum_bb_exp_rate_max"] = rate_max

        out_file = out_dir / f"regime_validation_v6_v44exp_{label}_{variant_label}.json"
        out_file.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

        v44 = report["strategy_by_regime"]["v44_ny"]
        v44_mom = v44["hysteresis"].get("momentum", {})
        v44_bo = v44["hysteresis"].get("breakout", {})
        v44_pbt = v44["hysteresis"].get("post_breakout_trend", {})
        v44_amb = v44["hysteresis"].get("ambiguous", {})
        dist = report["bar_distribution"]["hysteresis"]

        results.append({
            "dataset": label,
            "variant": variant_label,
            "rate_min": rate_min,
            "rate_max": rate_max,
            "v44_total": v44["total"],
            "v44_overall_wr": v44["win_rate"],
            "v44_mom_trades": v44_mom.get("trades", 0),
            "v44_mom_wr": v44_mom.get("win_rate", 0),
            "v44_bo_trades": v44_bo.get("trades", 0),
            "v44_bo_wr": v44_bo.get("win_rate", 0),
            "v44_pbt_trades": v44_pbt.get("trades", 0),
            "v44_pbt_wr": v44_pbt.get("win_rate", 0),
            "v44_amb_trades": v44_amb.get("trades", 0),
            "v44_amb_wr": v44_amb.get("win_rate", 0),
            "dist_momentum": dist.get("momentum", {}).get("pct", 0),
            "dist_mr": dist.get("mean_reversion", {}).get("pct", 0),
            "dist_bo": dist.get("breakout", {}).get("pct", 0),
            "dist_pbt": dist.get("post_breakout_trend", {}).get("pct", 0),
            "dist_amb": dist.get("ambiguous", {}).get("pct", 0),
        })
        print(f"         {variant_label:20s}  V44 mom: {v44_mom.get('trades',0):3d} tr {v44_mom.get('win_rate',0):5.1f}%  "
              f"bo: {v44_bo.get('trades',0):3d} tr {v44_bo.get('win_rate',0):5.1f}%  "
              f"dist_mom: {dist.get('momentum',{}).get('pct',0):5.1f}%")

    return results


def main() -> int:
    out_dir = ROOT / "research_out"
    out_dir.mkdir(exist_ok=True)

    all_results = []
    for label, csv_path in DATASETS:
        all_results.extend(run_dataset(label, csv_path, out_dir))

    summary_path = out_dir / "regime_v6_v44_sweep_band_summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    print(f"\n\n{'='*80}")
    print("V6 V44 MOMENTUM SWEEP — COMPARISON TABLE")
    print(f"{'='*80}")
    print(f"{'Variant':20s} {'Dataset':6s}  {'V44 mom':>10s}  {'V44 bo':>10s}  {'V44 pbt':>10s}  {'V44 amb':>10s}  {'dist_mom':>8s}  {'dist_amb':>8s}")
    print("-" * 100)
    for r in all_results:
        mom_str = f"{r['v44_mom_trades']:3d}/{r['v44_mom_wr']:5.1f}%"
        bo_str = f"{r['v44_bo_trades']:3d}/{r['v44_bo_wr']:5.1f}%"
        pbt_str = f"{r['v44_pbt_trades']:3d}/{r['v44_pbt_wr']:5.1f}%"
        amb_str = f"{r['v44_amb_trades']:3d}/{r['v44_amb_wr']:5.1f}%"
        print(f"{r['variant']:20s} {r['dataset']:6s}  {mom_str:>10s}  {bo_str:>10s}  {pbt_str:>10s}  {amb_str:>10s}  {r['dist_momentum']:7.1f}%  {r['dist_amb']:7.1f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
