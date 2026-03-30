#!/usr/bin/env python3
"""
Diagnostic: Split V14 (Tokyo) trades by top-scoring regime at entry time.

Same pattern as London V2 and V44 regime diagnostics — tag every V14 trade
with the regime classifier's scores at entry, then group by:
  - Which regime was the top score
  - Which hysteresis label was active
  - Ambiguous sub-split by top score

Looking for a "bleed bucket" where V14 takes trades in a regime
that's structurally bad for mean-reversion strategies.

V14 already has its own gates (BB "ranging", ADX < 35, ATR < 0.3),
so this checks whether the score-based classifier catches anything
those gates miss.
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
    print(f"  V14 (TOKYO) REGIME AUDIT: {dataset_label}")
    print(f"{'='*70}")

    with open(baseline_json, encoding="utf-8") as f:
        baseline = json.load(f)

    v14_trades = [t for t in baseline["closed_trades"] if t["strategy"] == "v14"]
    print(f"Total V14 trades: {len(v14_trades)}")

    print("Classifying bars...")
    classified = _load_classified_bars(input_csv)

    # Group trades by top regime score AND by hysteresis label
    by_top_regime: dict[str, list] = defaultdict(list)
    by_label: dict[str, list] = defaultdict(list)

    for t in v14_trades:
        regime = _lookup_regime(classified, pd.Timestamp(t["entry_time"]))
        scores = regime["regime_scores"]
        label = regime["regime_label"]
        top_regime = max(scores, key=scores.get) if scores else "unknown"
        margin = regime["regime_margin"]

        entry = {
            "entry_time": t["entry_time"],
            "exit_time": t["exit_time"],
            "side": t["side"],
            "pips": t["pips"],
            "usd": t["usd"],
            "exit_reason": t["exit_reason"],
            "regime_label": label,
            "top_regime": top_regime,
            "scores": scores,
            "margin": margin,
        }

        by_top_regime[top_regime].append(entry)
        by_label[label].append(entry)

    def _summarize(group_name: str, trades: list) -> dict:
        if not trades:
            print(f"\n  {group_name}: 0 trades")
            return {"count": 0}
        winners = [t for t in trades if t["pips"] > 0]
        losers = [t for t in trades if t["pips"] <= 0]
        net_pips = sum(t["pips"] for t in trades)
        net_usd = sum(t["usd"] for t in trades)
        avg_pips = net_pips / len(trades)
        win_rate = 100.0 * len(winners) / len(trades)
        gross_win = sum(t["pips"] for t in winners) if winners else 0
        gross_loss = abs(sum(t["pips"] for t in losers)) if losers else 0
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

        print(f"\n  {group_name}:")
        print(f"    Trades: {len(trades)}  |  W: {len(winners)}  L: {len(losers)}  |  WR: {win_rate:.1f}%")
        print(f"    Net pips: {net_pips:+.1f}  |  Net USD: ${net_usd:+,.2f}  |  Avg pips: {avg_pips:+.1f}")
        print(f"    PF: {pf:.3f}")
        if len(losers) > 0 and len(winners) > 0:
            print(f"    L:W ratio: {len(losers)/len(winners):.2f}:1")

        # Exit reason breakdown
        exit_dist = defaultdict(lambda: {"count": 0, "net_pips": 0.0, "winners": 0, "losers": 0})
        for t in trades:
            r = t["exit_reason"]
            exit_dist[r]["count"] += 1
            exit_dist[r]["net_pips"] += t["pips"]
            if t["pips"] > 0:
                exit_dist[r]["winners"] += 1
            else:
                exit_dist[r]["losers"] += 1
        print(f"    Exit reasons: ", end="")
        parts = []
        for reason, d in sorted(exit_dist.items(), key=lambda x: x[1]["count"], reverse=True):
            parts.append(f"{reason}={d['count']}({d['net_pips']:+.1f}p)")
        print(", ".join(parts))

        # Side breakdown
        buy_trades = [t for t in trades if t["side"] == "buy"]
        sell_trades = [t for t in trades if t["side"] == "sell"]
        buy_net = sum(t["pips"] for t in buy_trades)
        sell_net = sum(t["pips"] for t in sell_trades)
        print(f"    Sides: buy={len(buy_trades)}({buy_net:+.1f}p) sell={len(sell_trades)}({sell_net:+.1f}p)")

        # Regime margin distribution for this group
        margins = [t["margin"] for t in trades]
        avg_margin = sum(margins) / len(margins)
        print(f"    Avg regime margin: {avg_margin:.2f}")

        return {
            "count": len(trades),
            "winners": len(winners),
            "losers": len(losers),
            "win_rate_pct": round(win_rate, 1),
            "net_pips": round(net_pips, 1),
            "net_usd": round(net_usd, 2),
            "pf": round(pf, 3),
            "lw_ratio": round(len(losers) / max(1, len(winners)), 2),
            "avg_pips": round(avg_pips, 1),
            "avg_margin": round(avg_margin, 2),
        }

    # ── By top regime score ──
    print(f"\n{'─'*70}")
    print(f"  BY TOP REGIME SCORE (which regime scored highest at entry)")
    print(f"{'─'*70}")

    results_by_top = {}
    for regime in ["momentum", "mean_reversion", "breakout", "post_breakout_trend"]:
        trades = by_top_regime.get(regime, [])
        results_by_top[regime] = _summarize(
            f"TOP SCORE: {regime.upper()}", trades
        )

    # ── By hysteresis label ──
    print(f"\n{'─'*70}")
    print(f"  BY REGIME LABEL (after hysteresis/dwell)")
    print(f"{'─'*70}")

    results_by_label = {}
    for label in ["momentum", "mean_reversion", "breakout", "post_breakout_trend", "ambiguous"]:
        trades = by_label.get(label, [])
        results_by_label[label] = _summarize(
            f"LABEL: {label.upper()}", trades
        )

    # ── Ambiguous sub-split (same as V44 diagnostic) ──
    ambiguous_trades = by_label.get("ambiguous", [])
    results_ambig_sub = {}
    if ambiguous_trades:
        print(f"\n{'─'*70}")
        print(f"  AMBIGUOUS SUB-SPLIT (by top score within ambiguous)")
        print(f"{'─'*70}")

        ambig_by_top = defaultdict(list)
        for t in ambiguous_trades:
            ambig_by_top[t["top_regime"]].append(t)

        for regime in sorted(ambig_by_top.keys()):
            results_ambig_sub[regime] = _summarize(
                f"AMBIGUOUS, TOP={regime.upper()}", ambig_by_top[regime]
            )

    # ── Winners vs Losers regime profile ──
    print(f"\n{'─'*70}")
    print(f"  REGIME PROFILE: WINNERS vs LOSERS")
    print(f"{'─'*70}")

    all_entries = []
    for trades_list in by_top_regime.values():
        all_entries.extend(trades_list)

    winners = [t for t in all_entries if t["pips"] > 0]
    losers = [t for t in all_entries if t["pips"] <= 0]

    def _regime_dist(trades: list, label: str):
        dist = defaultdict(int)
        for t in trades:
            dist[t["top_regime"]] += 1
        print(f"\n  {label} ({len(trades)} trades):")
        for regime in ["momentum", "mean_reversion", "breakout", "post_breakout_trend"]:
            count = dist.get(regime, 0)
            pct = 100.0 * count / len(trades) if trades else 0
            print(f"    {regime:<25s}: {count:>4d} ({pct:>5.1f}%)")
        return dict(dist)

    winner_dist = _regime_dist(winners, "WINNERS")
    loser_dist = _regime_dist(losers, "LOSERS")

    return {
        "total_trades": len(v14_trades),
        "by_top_regime": results_by_top,
        "by_label": results_by_label,
        "ambiguous_sub": results_ambig_sub,
        "winner_regime_dist": winner_dist,
        "loser_regime_dist": loser_dist,
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

    out_path = ROOT / "research_out" / "diagnostic_v14_regime_split.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults written to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
