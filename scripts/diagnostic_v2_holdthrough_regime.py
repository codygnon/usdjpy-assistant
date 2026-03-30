#!/usr/bin/env python3
"""
Diagnostic: Split London V2 hold-through results by regime at hard-close time.

Variant G holds ALL hard-closed London V2 trades through with trail 8/8.
25 of 69 held trades on 1000k worsened. This diagnostic checks whether
the regime at hard-close time predicts which trades will improve vs worsen.

If a specific regime at hard-close consistently produces worsened trades,
we can conditionally hold through only in favorable regimes.

For each hard-closed London V2 trade:
  1. Look up regime at hard-close time (not entry time)
  2. Simulate hold-through with trail 8/8 (same as Variant G)
  3. Tag result as improved/worsened
  4. Group by regime at hard-close
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

from core.regime_classifier import RegimeThresholds
from scripts import validate_regime_classifier as regime_validation
from scripts.backtest_v44_conservative_router import _lookup_regime

PIP = 0.01
BE_OFFSET_PIPS = 1.0
NY_END_HOUR = 16
TRAIL_ACTIVATE_PIPS = 8.0
TRAIL_DISTANCE_PIPS = 8.0


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


def _load_classified_bars(input_csv: str) -> pd.DataFrame:
    m1 = regime_validation._load_m1(input_csv)
    featured = regime_validation.compute_features(m1)
    classified = regime_validation.classify_all_bars(featured, RegimeThresholds())
    return classified


def _simulate_holdthrough(
    m1: pd.DataFrame,
    m1_time_idx: pd.DatetimeIndex,
    entry_price: float,
    hardclose_time: pd.Timestamp,
    side: str,
    pips_at_hardclose: float,
) -> dict | None:
    """Simulate hold-through with trail 8/8. Returns None if not possible."""
    in_profit = pips_at_hardclose > 0
    profit_pips = pips_at_hardclose

    # Determine initial SL and trail state
    if side == "buy":
        if profit_pips >= TRAIL_ACTIVATE_PIPS:
            hc_price = entry_price + pips_at_hardclose * PIP
            best_price = hc_price
            trail_sl = best_price - TRAIL_DISTANCE_PIPS * PIP
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
        if profit_pips >= TRAIL_ACTIVATE_PIPS:
            hc_price = entry_price - pips_at_hardclose * PIP
            best_price = hc_price
            trail_sl = best_price + TRAIL_DISTANCE_PIPS * PIP
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

    deadline = hardclose_time.replace(hour=NY_END_HOUR, minute=0, second=0, microsecond=0)
    if deadline <= hardclose_time:
        return None

    mask = (m1_time_idx > hardclose_time) & (m1_time_idx <= deadline)
    forward_bars = m1.loc[mask]
    if forward_bars.empty:
        return None

    exit_price = None
    exit_time = None
    exit_type = None

    for _, bar in forward_bars.iterrows():
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])
        bar_time = bar["time"]

        if side == "buy":
            if bar_high > best_price:
                best_price = bar_high
                profit_from_entry = (best_price - entry_price) / PIP
                if not trailing_active and profit_from_entry >= TRAIL_ACTIVATE_PIPS:
                    trailing_active = True
                if trailing_active:
                    new_trail = best_price - TRAIL_DISTANCE_PIPS * PIP
                    new_trail = max(new_trail, entry_price + BE_OFFSET_PIPS * PIP)
                    trail_sl = max(trail_sl, new_trail)
            if bar_low <= trail_sl:
                exit_price = trail_sl
                exit_time = bar_time
                exit_type = "trail_stop" if trailing_active else "be_stop"
                break
        else:  # sell
            if bar_low < best_price:
                best_price = bar_low
                profit_from_entry = (entry_price - best_price) / PIP
                if not trailing_active and profit_from_entry >= TRAIL_ACTIVATE_PIPS:
                    trailing_active = True
                if trailing_active:
                    new_trail = best_price + TRAIL_DISTANCE_PIPS * PIP
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
        "total_pips": round(total_pips, 1),
        "additional_pips": round(additional_pips, 1),
        "exit_type": exit_type,
        "pips_at_hc": round(pips_at_hardclose, 1),
    }


def run_diagnostic(dataset_label: str, input_csv: str, baseline_json: str) -> dict:
    print(f"\n{'='*70}")
    print(f"  HOLD-THROUGH REGIME CONDITIONING: {dataset_label}")
    print(f"{'='*70}")

    with open(baseline_json, encoding="utf-8") as f:
        baseline = json.load(f)

    hardclosed = [
        t for t in baseline["closed_trades"]
        if t["strategy"] == "london_v2"
        and t["exit_reason"] in ("HARD_CLOSE", "TP1_ONLY_HARD_CLOSE")
    ]
    print(f"Hard-closed London V2 trades: {len(hardclosed)}")

    print("Loading M1 data...")
    m1 = _load_m1(input_csv)
    m1_time_idx = pd.DatetimeIndex(m1["time"])

    print("Classifying bars...")
    classified = _load_classified_bars(input_csv)

    # Simulate hold-through and tag with regime at hard-close
    results = []
    for t in hardclosed:
        entry_ts = pd.Timestamp(t["entry_time"])
        exit_ts = pd.Timestamp(t["exit_time"])

        entry_price = _get_price_at(m1, entry_ts, m1_time_idx)
        if entry_price is None:
            continue

        sim = _simulate_holdthrough(
            m1, m1_time_idx,
            entry_price=entry_price,
            hardclose_time=exit_ts,
            side=t["side"],
            pips_at_hardclose=t["pips"],
        )
        if sim is None:
            continue

        # Regime at HARD-CLOSE time (not entry)
        regime_at_hc = _lookup_regime(classified, exit_ts)
        scores_hc = regime_at_hc["regime_scores"]
        label_hc = regime_at_hc["regime_label"]
        top_regime_hc = max(scores_hc, key=scores_hc.get) if scores_hc else "unknown"
        margin_hc = regime_at_hc["regime_margin"]
        mom_score_hc = scores_hc.get("momentum", 0)

        # Also get regime at ENTRY time for comparison
        regime_at_entry = _lookup_regime(classified, entry_ts)
        label_entry = regime_at_entry["regime_label"]
        top_regime_entry = max(regime_at_entry["regime_scores"], key=regime_at_entry["regime_scores"].get) if regime_at_entry["regime_scores"] else "unknown"

        results.append({
            **sim,
            "side": t["side"],
            "exit_reason": t["exit_reason"],
            "in_profit_at_hc": t["pips"] > 0,
            # Regime at hard-close
            "hc_label": label_hc,
            "hc_top_regime": top_regime_hc,
            "hc_margin": margin_hc,
            "hc_mom_score": mom_score_hc,
            "hc_scores": scores_hc,
            # Regime at entry
            "entry_label": label_entry,
            "entry_top_regime": top_regime_entry,
            # Transition
            "regime_changed": label_entry != label_hc,
        })

    print(f"Simulated: {len(results)} trades")
    improved = [r for r in results if r["additional_pips"] > 0]
    worsened = [r for r in results if r["additional_pips"] < 0]
    flat = [r for r in results if r["additional_pips"] == 0]
    print(f"Improved: {len(improved)}, Worsened: {len(worsened)}, Flat: {len(flat)}")
    total_add = sum(r["additional_pips"] for r in results)
    print(f"Total additional pips: {total_add:+.1f}")

    def _summarize(group_name: str, trades: list) -> dict:
        if not trades:
            print(f"\n  {group_name}: 0 trades")
            return {"count": 0}
        imp = [t for t in trades if t["additional_pips"] > 0]
        wrs = [t for t in trades if t["additional_pips"] < 0]
        add_pips = sum(t["additional_pips"] for t in trades)
        avg_add = add_pips / len(trades)
        worst = min(t["additional_pips"] for t in trades)
        best = max(t["additional_pips"] for t in trades)

        # Exit type distribution
        exit_dist = defaultdict(int)
        for t in trades:
            exit_dist[t["exit_type"]] += 1

        # Profit at HC split
        in_profit = [t for t in trades if t["in_profit_at_hc"]]
        at_loss = [t for t in trades if not t["in_profit_at_hc"]]
        profit_add = sum(t["additional_pips"] for t in in_profit) if in_profit else 0
        loss_add = sum(t["additional_pips"] for t in at_loss) if at_loss else 0

        print(f"\n  {group_name}:")
        print(f"    Trades: {len(trades)}  |  Improved: {len(imp)}  Worsened: {len(wrs)}")
        print(f"    Additional pips: {add_pips:+.1f}  |  Avg: {avg_add:+.1f}  |  Best: {best:+.1f}  Worst: {worst:+.1f}")
        print(f"    Exits: {dict(exit_dist)}")
        print(f"    In-profit at HC: {len(in_profit)} ({profit_add:+.1f}p)  |  At-loss: {len(at_loss)} ({loss_add:+.1f}p)")

        # Side breakdown
        buys = [t for t in trades if t["side"] == "buy"]
        sells = [t for t in trades if t["side"] == "sell"]
        buy_add = sum(t["additional_pips"] for t in buys)
        sell_add = sum(t["additional_pips"] for t in sells)
        print(f"    Sides: buy={len(buys)}({buy_add:+.1f}p) sell={len(sells)}({sell_add:+.1f}p)")

        return {
            "count": len(trades),
            "improved": len(imp),
            "worsened": len(wrs),
            "additional_pips": round(add_pips, 1),
            "avg_additional": round(avg_add, 1),
            "worst": round(worst, 1),
            "best": round(best, 1),
            "exit_dist": dict(exit_dist),
        }

    output = {}

    # ── 1. By regime LABEL at hard-close ──
    print(f"\n{'─'*70}")
    print(f"  BY REGIME LABEL AT HARD-CLOSE")
    print(f"{'─'*70}")

    by_hc_label = defaultdict(list)
    for r in results:
        by_hc_label[r["hc_label"]].append(r)

    output["by_hc_label"] = {}
    for label in ["momentum", "mean_reversion", "breakout", "post_breakout_trend", "ambiguous"]:
        output["by_hc_label"][label] = _summarize(
            f"HC LABEL: {label.upper()}", by_hc_label.get(label, [])
        )

    # ── 2. By TOP REGIME at hard-close ──
    print(f"\n{'─'*70}")
    print(f"  BY TOP REGIME SCORE AT HARD-CLOSE")
    print(f"{'─'*70}")

    by_hc_top = defaultdict(list)
    for r in results:
        by_hc_top[r["hc_top_regime"]].append(r)

    output["by_hc_top"] = {}
    for regime in ["momentum", "mean_reversion", "breakout", "post_breakout_trend"]:
        output["by_hc_top"][regime] = _summarize(
            f"HC TOP: {regime.upper()}", by_hc_top.get(regime, [])
        )

    # ── 3. By momentum score at hard-close ──
    print(f"\n{'─'*70}")
    print(f"  BY MOMENTUM SCORE AT HARD-CLOSE")
    print(f"{'─'*70}")

    mom_buckets = {
        "mom<2": lambda r: r["hc_mom_score"] < 2.0,
        "mom_2-3": lambda r: 2.0 <= r["hc_mom_score"] < 3.0,
        "mom>=3": lambda r: r["hc_mom_score"] >= 3.0,
    }

    output["by_hc_mom_score"] = {}
    for name, pred in mom_buckets.items():
        trades = [r for r in results if pred(r)]
        output["by_hc_mom_score"][name] = _summarize(f"HC MOM SCORE: {name}", trades)

    # ── 4. By regime transition (entry → hard-close) ──
    print(f"\n{'─'*70}")
    print(f"  BY REGIME TRANSITION (entry → hard-close)")
    print(f"{'─'*70}")

    changed = [r for r in results if r["regime_changed"]]
    stable = [r for r in results if not r["regime_changed"]]

    output["by_transition"] = {
        "regime_changed": _summarize("REGIME CHANGED (entry→HC)", changed),
        "regime_stable": _summarize("REGIME STABLE (entry→HC)", stable),
    }

    # ── 5. Transition detail ──
    print(f"\n{'─'*70}")
    print(f"  TRANSITION DETAIL (entry_label → hc_label)")
    print(f"{'─'*70}")

    transitions = defaultdict(list)
    for r in results:
        key = f"{r['entry_label']} → {r['hc_label']}"
        transitions[key].append(r)

    output["transitions"] = {}
    for key in sorted(transitions.keys(), key=lambda k: len(transitions[k]), reverse=True):
        output["transitions"][key] = _summarize(f"TRANSITION: {key}", transitions[key])

    # ── 6. By profit state at HC × regime ──
    print(f"\n{'─'*70}")
    print(f"  IN-PROFIT vs AT-LOSS × REGIME AT HC")
    print(f"{'─'*70}")

    output["profit_x_regime"] = {}
    for profit_state, pred in [("in_profit", lambda r: r["in_profit_at_hc"]), ("at_loss", lambda r: not r["in_profit_at_hc"])]:
        state_trades = [r for r in results if pred(r)]
        by_top = defaultdict(list)
        for r in state_trades:
            by_top[r["hc_top_regime"]].append(r)
        output["profit_x_regime"][profit_state] = {}
        for regime in ["momentum", "mean_reversion", "breakout"]:
            output["profit_x_regime"][profit_state][regime] = _summarize(
                f"{profit_state.upper()} × HC TOP={regime.upper()}", by_top.get(regime, [])
            )

    # ── 7. Summary: best and worst regime conditions for hold-through ──
    print(f"\n{'─'*70}")
    print(f"  HOLD-THROUGH DECISION MATRIX")
    print(f"{'─'*70}")

    print(f"\n  {'HC Regime':<25s} {'Trades':>7s} {'Imp':>5s} {'Wrs':>5s} {'Add Pips':>10s} {'Avg':>7s} {'Worst':>7s}")
    print(f"  {'─'*70}")
    for label in ["momentum", "mean_reversion", "breakout", "post_breakout_trend", "ambiguous"]:
        trades = by_hc_label.get(label, [])
        if not trades:
            continue
        imp = sum(1 for t in trades if t["additional_pips"] > 0)
        wrs = sum(1 for t in trades if t["additional_pips"] < 0)
        add = sum(t["additional_pips"] for t in trades)
        avg = add / len(trades)
        worst = min(t["additional_pips"] for t in trades)
        print(f"  {label:<25s} {len(trades):>7d} {imp:>5d} {wrs:>5d} {add:>+10.1f} {avg:>+7.1f} {worst:>+7.1f}")

    return output


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

    out_path = ROOT / "research_out" / "diagnostic_v2_holdthrough_regime.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults written to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
