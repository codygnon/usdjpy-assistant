#!/usr/bin/env python3
"""
London V2 Hold-Through Diagnostic.

For each London V2 trade that was hard-closed (HARD_CLOSE or TP1_ONLY_HARD_CLOSE),
simulate what would have happened if the trade continued running into NY session.

Hold-through rules:
  - Trade stays open past London session end
  - SL moves to breakeven (entry ± BE offset of 1 pip) if trade was in profit at hard-close
  - For losing trades at hard-close: keep original SL (reconstructed from config)
  - Extended deadline: close at 16:00 UTC (NY session end)
  - Track: did the trade hit BE stop, improve, or just bleed out?

For TP1_ONLY_HARD_CLOSE trades: TP1 already hit (50% closed), remainder was hard-closed.
  These are interesting because the trade was already profitable — the runner was cut.

Measures MFE (max favorable excursion) and MAE (max adverse excursion) after
the hard-close point to understand the opportunity cost.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/codygnon/Documents/usdjpy_assistant")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PIP = 0.01
BE_OFFSET_PIPS = 1.0
NY_END_HOUR = 16  # hard deadline for hold-through


def _load_m1(input_csv: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time")
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    return df


def _get_price_at(m1: pd.DataFrame, ts: pd.Timestamp, m1_time_idx: pd.DatetimeIndex) -> float | None:
    """Get M1 close price at or just before timestamp."""
    idx = m1_time_idx.get_indexer([ts], method="ffill")[0]
    if idx < 0:
        return None
    return float(m1.iloc[idx]["close"])


def _simulate_holdthrough(
    m1: pd.DataFrame,
    m1_time_idx: pd.DatetimeIndex,
    entry_price: float,
    hardclose_time: pd.Timestamp,
    side: str,
    pips_at_hardclose: float,
    exit_reason: str,
) -> dict:
    """Simulate holding a trade past the hard-close point.

    Returns dict with MFE, MAE, holdthrough outcome, and simulated exit.
    """
    # Determine BE stop level
    in_profit_at_hc = pips_at_hardclose > 0
    if in_profit_at_hc:
        # Move SL to breakeven + offset
        if side == "buy":
            be_sl = entry_price + BE_OFFSET_PIPS * PIP
        else:
            be_sl = entry_price - BE_OFFSET_PIPS * PIP
    else:
        # Keep trade open but with a tight stop at entry (true BE)
        be_sl = entry_price

    # Extended deadline: 16:00 UTC same day
    deadline = hardclose_time.replace(hour=NY_END_HOUR, minute=0, second=0, microsecond=0)
    if deadline <= hardclose_time:
        # Already past NY end
        return {
            "holdthrough_possible": False,
            "reason": "past_ny_end",
        }

    # Get M1 bars from hard-close to deadline
    mask = (m1_time_idx > hardclose_time) & (m1_time_idx <= deadline)
    forward_bars = m1.loc[mask]

    if forward_bars.empty:
        return {
            "holdthrough_possible": False,
            "reason": "no_forward_data",
        }

    # Simulate bar by bar
    mfe_pips = 0.0  # max favorable excursion from hard-close price
    mae_pips = 0.0  # max adverse excursion from hard-close price
    hc_price = _get_price_at(m1, hardclose_time, m1_time_idx)
    if hc_price is None:
        return {"holdthrough_possible": False, "reason": "no_hc_price"}

    exit_price = None
    exit_time = None
    exit_type = None

    for _, bar in forward_bars.iterrows():
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])
        bar_close = float(bar["close"])
        bar_time = bar["time"]

        if side == "buy":
            # Check if BE stop hit (price went below be_sl)
            if bar_low <= be_sl:
                exit_price = be_sl
                exit_time = bar_time
                exit_type = "be_stop"
                break
            # Track excursions from hard-close price
            fav = (bar_high - hc_price) / PIP
            adv = (hc_price - bar_low) / PIP
        else:  # sell
            if bar_high >= be_sl:
                exit_price = be_sl
                exit_time = bar_time
                exit_type = "be_stop"
                break
            fav = (hc_price - bar_low) / PIP
            adv = (bar_high - hc_price) / PIP

        mfe_pips = max(mfe_pips, fav)
        mae_pips = max(mae_pips, adv)

    # If no stop hit, close at deadline
    if exit_price is None:
        last_bar = forward_bars.iloc[-1]
        exit_price = float(last_bar["close"])
        exit_time = last_bar["time"]
        exit_type = "ny_session_close"

    # Calculate final pips
    if side == "buy":
        total_pips = (exit_price - entry_price) / PIP
        additional_pips = (exit_price - hc_price) / PIP
    else:
        total_pips = (entry_price - exit_price) / PIP
        additional_pips = (hc_price - exit_price) / PIP

    hold_duration_min = (exit_time - hardclose_time).total_seconds() / 60.0

    return {
        "holdthrough_possible": True,
        "entry_price": round(entry_price, 3),
        "hc_price": round(hc_price, 3),
        "pips_at_hc": round(pips_at_hardclose, 1),
        "exit_price": round(exit_price, 3),
        "exit_time": exit_time.isoformat() if hasattr(exit_time, 'isoformat') else str(exit_time),
        "exit_type": exit_type,
        "total_pips": round(total_pips, 1),
        "additional_pips": round(additional_pips, 1),
        "mfe_after_hc_pips": round(mfe_pips, 1),
        "mae_after_hc_pips": round(mae_pips, 1),
        "hold_duration_min": round(hold_duration_min, 0),
        "be_sl_price": round(be_sl, 3),
        "in_profit_at_hc": in_profit_at_hc,
    }


def run_diagnostic(dataset_label: str, input_csv: str, baseline_json: str) -> dict:
    print(f"\n{'='*70}")
    print(f"  LONDON V2 HOLD-THROUGH DIAGNOSTIC: {dataset_label}")
    print(f"{'='*70}")

    with open(baseline_json, encoding="utf-8") as f:
        baseline = json.load(f)

    # Get hard-closed London V2 trades (both HARD_CLOSE and TP1_ONLY_HARD_CLOSE)
    hardclosed = [
        t for t in baseline["closed_trades"]
        if t["strategy"] == "london_v2"
        and t["exit_reason"] in ("HARD_CLOSE", "TP1_ONLY_HARD_CLOSE")
    ]
    print(f"Hard-closed London V2 trades: {len(hardclosed)}")
    hc_count = sum(1 for t in hardclosed if t["exit_reason"] == "HARD_CLOSE")
    tp1_hc_count = sum(1 for t in hardclosed if t["exit_reason"] == "TP1_ONLY_HARD_CLOSE")
    print(f"  HARD_CLOSE: {hc_count}, TP1_ONLY_HARD_CLOSE: {tp1_hc_count}")

    print("Loading M1 data...")
    m1 = _load_m1(input_csv)
    m1_time_idx = pd.DatetimeIndex(m1["time"])

    results = []
    for t in hardclosed:
        entry_ts = pd.Timestamp(t["entry_time"])
        exit_ts = pd.Timestamp(t["exit_time"])

        # Reconstruct entry price from M1 data
        entry_price = _get_price_at(m1, entry_ts, m1_time_idx)
        if entry_price is None:
            continue

        sim = _simulate_holdthrough(
            m1, m1_time_idx,
            entry_price=entry_price,
            hardclose_time=exit_ts,
            side=t["side"],
            pips_at_hardclose=t["pips"],
            exit_reason=t["exit_reason"],
        )

        if not sim.get("holdthrough_possible", False):
            continue

        sim["original_exit_reason"] = t["exit_reason"]
        sim["side"] = t["side"]
        sim["original_pips"] = round(t["pips"], 1)
        sim["original_usd"] = round(t["usd"], 2)
        sim["entry_time"] = t["entry_time"]
        sim["hc_time"] = t["exit_time"]
        results.append(sim)

    # ── Analyze results ──────────────────────────────────────────────

    if not results:
        print("  No hold-through simulations completed.")
        return {"simulations": 0}

    print(f"\n  Simulations completed: {len(results)}")

    # Split by original exit reason
    for exit_type in ["HARD_CLOSE", "TP1_ONLY_HARD_CLOSE"]:
        subset = [r for r in results if r["original_exit_reason"] == exit_type]
        if not subset:
            continue

        print(f"\n  {'─'*60}")
        print(f"  {exit_type} trades ({len(subset)} total)")
        print(f"  {'─'*60}")

        # Original outcome
        orig_net = sum(r["original_pips"] for r in subset)
        print(f"  Original net pips (at hard-close): {orig_net:+.1f}")

        # Hold-through outcome
        ht_net = sum(r["total_pips"] for r in subset)
        additional_net = sum(r["additional_pips"] for r in subset)
        print(f"  Hold-through net pips:             {ht_net:+.1f}")
        print(f"  Additional pips from holding:      {additional_net:+.1f}")

        # By hold-through exit type
        by_exit = defaultdict(list)
        for r in subset:
            by_exit[r["exit_type"]].append(r)
        print(f"\n  Hold-through exit distribution:")
        for etype, trades in sorted(by_exit.items(), key=lambda x: len(x[1]), reverse=True):
            add_pips = sum(r["additional_pips"] for r in trades)
            avg_dur = np.mean([r["hold_duration_min"] for r in trades])
            print(f"    {etype:<25s} {len(trades):>3d} trades, additional: {add_pips:+.1f}p, avg hold: {avg_dur:.0f}min")

        # Winners at hard-close vs losers at hard-close
        hc_winners = [r for r in subset if r["in_profit_at_hc"]]
        hc_losers = [r for r in subset if not r["in_profit_at_hc"]]

        if hc_winners:
            add_w = sum(r["additional_pips"] for r in hc_winners)
            orig_w = sum(r["original_pips"] for r in hc_winners)
            ht_w = sum(r["total_pips"] for r in hc_winners)
            print(f"\n  Trades IN PROFIT at hard-close ({len(hc_winners)}):")
            print(f"    Original: {orig_w:+.1f}p → Hold-through: {ht_w:+.1f}p (additional: {add_w:+.1f}p)")
            # MFE/MAE
            mfes = [r["mfe_after_hc_pips"] for r in hc_winners]
            maes = [r["mae_after_hc_pips"] for r in hc_winners]
            print(f"    MFE after HC: avg={np.mean(mfes):.1f}p median={np.median(mfes):.1f}p max={max(mfes):.1f}p")
            print(f"    MAE after HC: avg={np.mean(maes):.1f}p median={np.median(maes):.1f}p max={max(maes):.1f}p")

        if hc_losers:
            add_l = sum(r["additional_pips"] for r in hc_losers)
            orig_l = sum(r["original_pips"] for r in hc_losers)
            ht_l = sum(r["total_pips"] for r in hc_losers)
            print(f"\n  Trades AT LOSS at hard-close ({len(hc_losers)}):")
            print(f"    Original: {orig_l:+.1f}p → Hold-through: {ht_l:+.1f}p (additional: {add_l:+.1f}p)")
            mfes = [r["mfe_after_hc_pips"] for r in hc_losers]
            maes = [r["mae_after_hc_pips"] for r in hc_losers]
            print(f"    MFE after HC: avg={np.mean(mfes):.1f}p median={np.median(mfes):.1f}p max={max(mfes):.1f}p")
            print(f"    MAE after HC: avg={np.mean(maes):.1f}p median={np.median(maes):.1f}p max={max(maes):.1f}p")

    # ── Overall summary ──────────────────────────────────────────────

    total_orig = sum(r["original_pips"] for r in results)
    total_ht = sum(r["total_pips"] for r in results)
    total_add = sum(r["additional_pips"] for r in results)

    print(f"\n  {'='*60}")
    print(f"  OVERALL HOLD-THROUGH IMPACT")
    print(f"  {'='*60}")
    print(f"  Original hard-close net pips: {total_orig:+.1f}")
    print(f"  Hold-through net pips:        {total_ht:+.1f}")
    print(f"  Additional pips from holding: {total_add:+.1f}")
    print(f"  Trades improved:  {sum(1 for r in results if r['additional_pips'] > 0)}")
    print(f"  Trades worsened:  {sum(1 for r in results if r['additional_pips'] < 0)}")
    print(f"  Trades unchanged: {sum(1 for r in results if r['additional_pips'] == 0)}")

    # Per-trade detail for top improvements and worst deteriorations
    sorted_by_additional = sorted(results, key=lambda r: r["additional_pips"], reverse=True)
    print(f"\n  TOP 5 IMPROVEMENTS from hold-through:")
    for r in sorted_by_additional[:5]:
        print(f"    {r['entry_time'][:10]} {r['side']:>4s} | HC:{r['original_pips']:+.1f}p → HT:{r['total_pips']:+.1f}p | +{r['additional_pips']:.1f}p | exit:{r['exit_type']} after {r['hold_duration_min']:.0f}min")

    print(f"\n  WORST 5 DETERIORATIONS from hold-through:")
    for r in sorted_by_additional[-5:]:
        print(f"    {r['entry_time'][:10]} {r['side']:>4s} | HC:{r['original_pips']:+.1f}p → HT:{r['total_pips']:+.1f}p | {r['additional_pips']:+.1f}p | exit:{r['exit_type']} after {r['hold_duration_min']:.0f}min")

    return {
        "simulations": len(results),
        "total_original_pips": round(total_orig, 1),
        "total_holdthrough_pips": round(total_ht, 1),
        "total_additional_pips": round(total_add, 1),
        "trades_improved": sum(1 for r in results if r["additional_pips"] > 0),
        "trades_worsened": sum(1 for r in results if r["additional_pips"] < 0),
        "detail": results,
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
            print(f"Skipping {label}: not found")
            continue
        if not Path(baseline_path).exists():
            print(f"Skipping {label}: not found")
            continue
        all_results[label] = run_diagnostic(label, csv_path, baseline_path)

    out_path = ROOT / "research_out" / "diagnostic_v2_holdthrough.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
