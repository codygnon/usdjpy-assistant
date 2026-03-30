#!/usr/bin/env python3
"""
Diagnostic: Split ambiguous V44 trades by whether momentum was the top-scoring regime.

Reads:
  - The aligned baseline trades (1000k and 500k)
  - The classified bar data (regime scores per M5 bar)

Outputs a summary showing, for ambiguous V44 trades:
  - Group A: momentum is the top score (but margin < min_margin) — "near-momentum"
  - Group B: momentum is NOT the top score — "true ambiguous"

With W/L counts, net pips, and average margin for each group.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/codygnon/Documents/usdjpy_assistant")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.regime_classifier import RegimeThresholds
from scripts import validate_regime_classifier as regime_validation
from scripts.backtest_v44_conservative_router import _lookup_regime


def _load_classified_bars(input_csv: str) -> pd.DataFrame:
    m1 = regime_validation._load_m1(input_csv)
    featured = regime_validation.compute_features(m1)
    classified = regime_validation.classify_all_bars(featured, RegimeThresholds())
    return classified


def run_diagnostic(dataset_label: str, input_csv: str, baseline_json: str) -> dict:
    print(f"\n{'='*70}")
    print(f"  {dataset_label}")
    print(f"{'='*70}")

    # Load baseline trades
    with open(baseline_json, encoding="utf-8") as f:
        baseline = json.load(f)

    v44_trades = [t for t in baseline["closed_trades"] if t["strategy"] == "v44_ny"]
    print(f"Total V44 trades: {len(v44_trades)}")

    # Load classified bars
    print("Classifying bars (this takes a moment)...")
    classified = _load_classified_bars(input_csv)

    # Look up regime for each V44 trade
    near_momentum = []  # ambiguous, momentum is top score
    true_ambiguous = []  # ambiguous, momentum is NOT top score
    momentum_trades = []
    other_regime = []

    for t in v44_trades:
        regime = _lookup_regime(classified, pd.Timestamp(t["entry_time"]))
        label = regime["regime_label"]
        scores = regime["regime_scores"]

        entry = {
            "entry_time": t["entry_time"],
            "side": t["side"],
            "pips": t["pips"],
            "usd": t["usd"],
            "exit_reason": t["exit_reason"],
            "regime_label": label,
            "scores": scores,
            "margin": regime["regime_margin"],
        }

        if label == "momentum":
            momentum_trades.append(entry)
        elif label == "ambiguous":
            # Check if momentum is the top score
            top_regime = max(scores, key=scores.get)
            if top_regime == "momentum":
                entry["top_score"] = "momentum"
                entry["momentum_score"] = scores["momentum"]
                entry["runner_up"] = sorted(scores.items(), key=lambda x: x[1], reverse=True)[1]
                near_momentum.append(entry)
            else:
                entry["top_score"] = top_regime
                entry["momentum_score"] = scores["momentum"]
                entry["top_score_value"] = scores[top_regime]
                true_ambiguous.append(entry)
        else:
            other_regime.append(entry)

    # Print results
    def _summarize(label: str, trades: list) -> dict:
        if not trades:
            print(f"\n  {label}: 0 trades")
            return {"count": 0}
        winners = [t for t in trades if t["pips"] > 0]
        losers = [t for t in trades if t["pips"] <= 0]
        net_pips = sum(t["pips"] for t in trades)
        net_usd = sum(t["usd"] for t in trades)
        avg_pips = net_pips / len(trades)
        win_rate = 100.0 * len(winners) / len(trades)

        print(f"\n  {label}:")
        print(f"    Trades: {len(trades)}  |  W: {len(winners)}  L: {len(losers)}  |  WR: {win_rate:.1f}%")
        print(f"    Net pips: {net_pips:+.1f}  |  Net USD: ${net_usd:+,.2f}  |  Avg pips: {avg_pips:+.1f}")
        if len(losers) > 0 and len(winners) > 0:
            print(f"    L:W ratio: {len(losers)/len(winners):.2f}:1")

        return {
            "count": len(trades),
            "winners": len(winners),
            "losers": len(losers),
            "win_rate_pct": round(win_rate, 1),
            "net_pips": round(net_pips, 1),
            "net_usd": round(net_usd, 2),
            "lw_ratio": round(len(losers) / max(1, len(winners)), 2),
        }

    results = {}
    results["momentum"] = _summarize("MOMENTUM (V44 keeps these)", momentum_trades)
    results["near_momentum_ambiguous"] = _summarize(
        "NEAR-MOMENTUM AMBIGUOUS (momentum is top score, margin < 0.5)", near_momentum
    )
    results["true_ambiguous"] = _summarize(
        "TRUE AMBIGUOUS (momentum is NOT top score)", true_ambiguous
    )
    results["other_regime"] = _summarize(
        "OTHER REGIME (breakout/post_breakout/mr — blocked by Variant A)", other_regime
    )

    # Score distribution within near-momentum
    if near_momentum:
        margins = [t["margin"] for t in near_momentum]
        mom_scores = [t["momentum_score"] for t in near_momentum]
        print(f"\n  Near-momentum score details:")
        print(f"    Momentum scores: min={min(mom_scores):.1f} median={sorted(mom_scores)[len(mom_scores)//2]:.1f} max={max(mom_scores):.1f}")
        print(f"    Margins: min={min(margins):.2f} median={sorted(margins)[len(margins)//2]:.2f} max={max(margins):.2f}")

    # Score distribution within true ambiguous
    if true_ambiguous:
        top_scores_dist = defaultdict(int)
        for t in true_ambiguous:
            top_scores_dist[t["top_score"]] += 1
        print(f"\n  True ambiguous — top-scoring regime distribution:")
        for regime, count in sorted(top_scores_dist.items(), key=lambda x: x[1], reverse=True):
            subset = [t for t in true_ambiguous if t["top_score"] == regime]
            w = sum(1 for t in subset if t["pips"] > 0)
            l = sum(1 for t in subset if t["pips"] <= 0)
            net = sum(t["pips"] for t in subset)
            print(f"    {regime}: {count} trades (W:{w} L:{l} net:{net:+.1f}p)")

    return results


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

    # Write results
    out_path = ROOT / "research_out" / "diagnostic_ambiguous_split.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults written to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
