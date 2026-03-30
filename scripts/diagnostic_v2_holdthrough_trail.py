#!/usr/bin/env python3
"""
London V2 Hold-Through with Trailing Stop — Parameter Calibration.

Tests several trailing stop configurations on the same hard-closed trades
to find the best balance between protecting profits and allowing runners.

Hold-through rules:
  - Trade at a loss at hard-close: SL at entry (BE)
  - Trade in profit < threshold: SL at entry + BE offset (1 pip)
  - Trade in profit >= threshold: trailing stop activates
    Trail ratchets favorably only, never widens.

Parameters to sweep:
  - trail_activate_pips: profit from entry needed to start trailing (8, 10, 15)
  - trail_distance_pips: how far behind best price the stop sits (8, 10, 12)
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/codygnon/Documents/usdjpy_assistant")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PIP = 0.01
BE_OFFSET_PIPS = 1.0
NY_END_HOUR = 16


def _load_m1(input_csv: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time")
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    return df


def _get_price_at(m1: pd.DataFrame, ts: pd.Timestamp, m1_time_idx: pd.DatetimeIndex) -> float | None:
    idx = m1_time_idx.get_indexer([ts], method="ffill")[0]
    if idx < 0:
        return None
    return float(m1.iloc[idx]["close"])


@dataclass
class TrailConfig:
    name: str
    activate_pips: float  # profit from entry to start trailing
    distance_pips: float  # trail distance behind best price


def _simulate_trailing_holdthrough(
    m1: pd.DataFrame,
    m1_time_idx: pd.DatetimeIndex,
    entry_price: float,
    hardclose_time: pd.Timestamp,
    side: str,
    pips_at_hardclose: float,
    cfg: TrailConfig,
) -> dict:
    """Simulate hold-through with trailing stop logic."""

    in_profit = pips_at_hardclose > 0
    profit_pips = pips_at_hardclose

    # Determine initial SL
    if side == "buy":
        if profit_pips >= cfg.activate_pips:
            # Trail is active — set stop at (best_price - trail_distance)
            hc_price = entry_price + pips_at_hardclose * PIP
            best_price = hc_price
            trail_sl = best_price - cfg.distance_pips * PIP
            # But never below entry + BE offset
            trail_sl = max(trail_sl, entry_price + BE_OFFSET_PIPS * PIP)
            trailing_active = True
        elif in_profit:
            trail_sl = entry_price + BE_OFFSET_PIPS * PIP
            best_price = entry_price + pips_at_hardclose * PIP
            trailing_active = False
        else:
            trail_sl = entry_price
            best_price = entry_price + pips_at_hardclose * PIP
            trailing_active = False
    else:  # sell
        if profit_pips >= cfg.activate_pips:
            hc_price = entry_price - pips_at_hardclose * PIP
            best_price = hc_price
            trail_sl = best_price + cfg.distance_pips * PIP
            trail_sl = min(trail_sl, entry_price - BE_OFFSET_PIPS * PIP)
            trailing_active = True
        elif in_profit:
            trail_sl = entry_price - BE_OFFSET_PIPS * PIP
            best_price = entry_price - pips_at_hardclose * PIP
            trailing_active = False
        else:
            trail_sl = entry_price
            best_price = entry_price - pips_at_hardclose * PIP
            trailing_active = False

    # Extended deadline
    deadline = hardclose_time.replace(hour=NY_END_HOUR, minute=0, second=0, microsecond=0)
    if deadline <= hardclose_time:
        return {"possible": False}

    mask = (m1_time_idx > hardclose_time) & (m1_time_idx <= deadline)
    forward_bars = m1.loc[mask]
    if forward_bars.empty:
        return {"possible": False}

    hc_price_val = _get_price_at(m1, hardclose_time, m1_time_idx)
    if hc_price_val is None:
        return {"possible": False}

    exit_price = None
    exit_time = None
    exit_type = None

    for _, bar in forward_bars.iterrows():
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])
        bar_close = float(bar["close"])
        bar_time = bar["time"]

        if side == "buy":
            # Update best price
            if bar_high > best_price:
                best_price = bar_high
                # Check if trailing should activate
                profit_from_entry = (best_price - entry_price) / PIP
                if not trailing_active and profit_from_entry >= cfg.activate_pips:
                    trailing_active = True
                # Update trail stop
                if trailing_active:
                    new_trail = best_price - cfg.distance_pips * PIP
                    new_trail = max(new_trail, entry_price + BE_OFFSET_PIPS * PIP)
                    trail_sl = max(trail_sl, new_trail)  # ratchet only

            # Check stop
            if bar_low <= trail_sl:
                exit_price = trail_sl
                exit_time = bar_time
                exit_type = "trail_stop" if trailing_active else "be_stop"
                break
        else:  # sell
            if bar_low < best_price:
                best_price = bar_low
                profit_from_entry = (entry_price - best_price) / PIP
                if not trailing_active and profit_from_entry >= cfg.activate_pips:
                    trailing_active = True
                if trailing_active:
                    new_trail = best_price + cfg.distance_pips * PIP
                    new_trail = min(new_trail, entry_price - BE_OFFSET_PIPS * PIP)
                    trail_sl = min(trail_sl, new_trail)

            if bar_high >= trail_sl:
                exit_price = trail_sl
                exit_time = bar_time
                exit_type = "trail_stop" if trailing_active else "be_stop"
                break

    if exit_price is None:
        last_bar = forward_bars.iloc[-1]
        exit_price = float(last_bar["close"])
        exit_time = last_bar["time"]
        exit_type = "ny_session_close"

    if side == "buy":
        total_pips = (exit_price - entry_price) / PIP
    else:
        total_pips = (entry_price - exit_price) / PIP

    additional_pips = total_pips - pips_at_hardclose

    return {
        "possible": True,
        "total_pips": round(total_pips, 1),
        "additional_pips": round(additional_pips, 1),
        "exit_type": exit_type,
        "pips_at_hc": round(pips_at_hardclose, 1),
    }


def run_sweep(dataset_label: str, input_csv: str, baseline_json: str) -> dict:
    print(f"\n{'='*70}")
    print(f"  TRAILING STOP CALIBRATION: {dataset_label}")
    print(f"{'='*70}")

    with open(baseline_json, encoding="utf-8") as f:
        baseline = json.load(f)

    hardclosed = [
        t for t in baseline["closed_trades"]
        if t["strategy"] == "london_v2"
        and t["exit_reason"] in ("HARD_CLOSE", "TP1_ONLY_HARD_CLOSE")
    ]

    print(f"Hard-closed trades: {len(hardclosed)}")
    print("Loading M1 data...")
    m1 = _load_m1(input_csv)
    m1_time_idx = pd.DatetimeIndex(m1["time"])

    # Parameter sweep
    configs = [
        TrailConfig("BE_only (baseline)", activate_pips=999, distance_pips=999),  # never trails, just BE
        TrailConfig("trail_8a_8d", activate_pips=8, distance_pips=8),
        TrailConfig("trail_8a_10d", activate_pips=8, distance_pips=10),
        TrailConfig("trail_8a_12d", activate_pips=8, distance_pips=12),
        TrailConfig("trail_10a_8d", activate_pips=10, distance_pips=8),
        TrailConfig("trail_10a_10d", activate_pips=10, distance_pips=10),
        TrailConfig("trail_10a_12d", activate_pips=10, distance_pips=12),
        TrailConfig("trail_15a_10d", activate_pips=15, distance_pips=10),
        TrailConfig("trail_15a_12d", activate_pips=15, distance_pips=12),
    ]

    all_config_results = {}

    for cfg in configs:
        sim_results = []
        for t in hardclosed:
            entry_ts = pd.Timestamp(t["entry_time"])
            exit_ts = pd.Timestamp(t["exit_time"])
            entry_price = _get_price_at(m1, entry_ts, m1_time_idx)
            if entry_price is None:
                continue

            sim = _simulate_trailing_holdthrough(
                m1, m1_time_idx,
                entry_price=entry_price,
                hardclose_time=exit_ts,
                side=t["side"],
                pips_at_hardclose=t["pips"],
                cfg=cfg,
            )
            if sim.get("possible"):
                sim["original_exit_reason"] = t["exit_reason"]
                sim["in_profit_at_hc"] = t["pips"] > 0
                sim_results.append(sim)

        if not sim_results:
            continue

        total_orig = sum(r["pips_at_hc"] for r in sim_results)
        total_ht = sum(r["total_pips"] for r in sim_results)
        total_add = sum(r["additional_pips"] for r in sim_results)
        improved = sum(1 for r in sim_results if r["additional_pips"] > 0)
        worsened = sum(1 for r in sim_results if r["additional_pips"] < 0)

        # Split by original exit reason
        hc_only = [r for r in sim_results if r["original_exit_reason"] == "HARD_CLOSE"]
        tp1_hc = [r for r in sim_results if r["original_exit_reason"] == "TP1_ONLY_HARD_CLOSE"]

        hc_add = sum(r["additional_pips"] for r in hc_only) if hc_only else 0
        tp1_add = sum(r["additional_pips"] for r in tp1_hc) if tp1_hc else 0

        # Exit type distribution
        exit_dist = defaultdict(int)
        for r in sim_results:
            exit_dist[r["exit_type"]] += 1

        # Worst single trade deterioration
        worst = min(r["additional_pips"] for r in sim_results)

        all_config_results[cfg.name] = {
            "total_additional_pips": round(total_add, 1),
            "hc_additional": round(hc_add, 1),
            "tp1_hc_additional": round(tp1_add, 1),
            "improved": improved,
            "worsened": worsened,
            "worst_single": round(worst, 1),
            "exit_dist": dict(exit_dist),
        }

    # Print comparison table
    print(f"\n  {'Config':<25s} {'Total Δ':>8s} {'HC Δ':>8s} {'TP1 Δ':>8s} {'Better':>7s} {'Worse':>7s} {'Worst':>7s}")
    print(f"  {'─'*75}")
    for name, r in all_config_results.items():
        print(f"  {name:<25s} {r['total_additional_pips']:>+8.1f} {r['hc_additional']:>+8.1f} "
              f"{r['tp1_hc_additional']:>+8.1f} {r['improved']:>7d} {r['worsened']:>7d} {r['worst_single']:>+7.1f}")

    # Exit distribution for each config
    print(f"\n  EXIT DISTRIBUTION:")
    print(f"  {'Config':<25s} {'be_stop':>8s} {'trail':>8s} {'ny_close':>8s}")
    print(f"  {'─'*55}")
    for name, r in all_config_results.items():
        ed = r["exit_dist"]
        print(f"  {name:<25s} {ed.get('be_stop',0):>8d} {ed.get('trail_stop',0):>8d} {ed.get('ny_session_close',0):>8d}")

    return all_config_results


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
            print(f"Skipping {label}: not found")
            continue
        if not Path(baseline_path).exists():
            print(f"Skipping {label}: not found")
            continue
        all_results[label] = run_sweep(label, csv_path, baseline_path)

    out_path = ROOT / "research_out" / "diagnostic_v2_holdthrough_trail_sweep.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
