#!/usr/bin/env python3
"""
Diagnostic: Split V44 (NY) trades by regime margin at entry time.

V44 is currently binary — blocked in breakout/post_breakout_trend/non-momentum-ambiguous
(Variant F), allowed everywhere else. This diagnostic checks whether the *confidence*
of the regime classification matters for V44 performance.

Splits V44's ALLOWED trades (post Variant F filter) by:
  1. Regime margin buckets (how confidently the top regime won)
  2. Within momentum: high vs low margin
  3. Near-momentum ambiguous (margin=0, allowed by Variant F) performance

Looking for: low-margin momentum trades that bleed disproportionately,
which would support confidence-based sizing instead of binary block/allow.
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


def _is_variant_f_allowed(label: str, top_regime: str) -> bool:
    """Reproduce Variant F filter logic: what trades survive?"""
    # Blocked regimes
    if label in ("breakout", "post_breakout_trend"):
        return False
    # Ambiguous: only allowed if momentum is top score
    if label == "ambiguous" and top_regime != "momentum":
        return False
    return True


def run_diagnostic(dataset_label: str, input_csv: str, baseline_json: str) -> dict:
    print(f"\n{'='*70}")
    print(f"  V44 MARGIN CONFIDENCE AUDIT: {dataset_label}")
    print(f"{'='*70}")

    with open(baseline_json, encoding="utf-8") as f:
        baseline = json.load(f)

    v44_trades = [t for t in baseline["closed_trades"] if t["strategy"] == "v44_ny"]
    print(f"Total V44 trades (pre-filter): {len(v44_trades)}")

    print("Classifying bars...")
    classified = _load_classified_bars(input_csv)

    # Tag each trade with regime info and filter through Variant F
    allowed_trades = []
    blocked_trades = []

    for t in v44_trades:
        regime = _lookup_regime(classified, pd.Timestamp(t["entry_time"]))
        scores = regime["regime_scores"]
        label = regime["regime_label"]
        top_regime = max(scores, key=scores.get) if scores else "unknown"
        margin = regime["regime_margin"]
        momentum_score = scores.get("momentum", 0)

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
            "momentum_score": momentum_score,
        }

        if _is_variant_f_allowed(label, top_regime):
            allowed_trades.append(entry)
        else:
            blocked_trades.append(entry)

    print(f"Variant F allowed: {len(allowed_trades)}, blocked: {len(blocked_trades)}")

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

        # Side breakdown
        buy_trades = [t for t in trades if t["side"] == "buy"]
        sell_trades = [t for t in trades if t["side"] == "sell"]
        buy_net = sum(t["pips"] for t in buy_trades)
        sell_net = sum(t["pips"] for t in sell_trades)
        print(f"    Sides: buy={len(buy_trades)}({buy_net:+.1f}p) sell={len(sell_trades)}({sell_net:+.1f}p)")

        # Exit reason breakdown
        exit_dist = defaultdict(lambda: {"count": 0, "net_pips": 0.0})
        for t in trades:
            exit_dist[t["exit_reason"]]["count"] += 1
            exit_dist[t["exit_reason"]]["net_pips"] += t["pips"]
        parts = []
        for reason, d in sorted(exit_dist.items(), key=lambda x: x[1]["count"], reverse=True):
            parts.append(f"{reason}={d['count']}({d['net_pips']:+.1f}p)")
        print(f"    Exits: {', '.join(parts)}")

        # Margin distribution
        margins = [t["margin"] for t in trades]
        avg_margin = sum(margins) / len(margins)
        print(f"    Avg margin: {avg_margin:.2f}, min: {min(margins):.2f}, max: {max(margins):.2f}")

        return {
            "count": len(trades),
            "winners": len(winners),
            "losers": len(losers),
            "win_rate_pct": round(win_rate, 1),
            "net_pips": round(net_pips, 1),
            "net_usd": round(net_usd, 2),
            "pf": round(pf, 3),
            "avg_pips": round(avg_pips, 1),
            "avg_margin": round(avg_margin, 2),
        }

    results = {}

    # ── 1. Split by regime label (allowed trades only) ──
    print(f"\n{'─'*70}")
    print(f"  ALLOWED TRADES BY REGIME LABEL")
    print(f"{'─'*70}")

    by_label = defaultdict(list)
    for t in allowed_trades:
        by_label[t["regime_label"]].append(t)

    results["by_label"] = {}
    for label in ["momentum", "mean_reversion", "ambiguous"]:
        results["by_label"][label] = _summarize(
            f"LABEL: {label.upper()}", by_label.get(label, [])
        )

    # ── 2. Split by margin buckets (allowed trades only) ──
    print(f"\n{'─'*70}")
    print(f"  ALLOWED TRADES BY MARGIN BUCKET")
    print(f"{'─'*70}")

    margin_buckets = {
        "ambiguous (margin=0)": lambda t: t["margin"] == 0,
        "low_margin (0<m<1.0)": lambda t: 0 < t["margin"] < 1.0,
        "mid_margin (1.0<=m<1.5)": lambda t: 1.0 <= t["margin"] < 1.5,
        "high_margin (m>=1.5)": lambda t: t["margin"] >= 1.5,
    }

    results["by_margin"] = {}
    for bucket_name, pred in margin_buckets.items():
        bucket_trades = [t for t in allowed_trades if pred(t)]
        results["by_margin"][bucket_name] = _summarize(
            f"MARGIN: {bucket_name}", bucket_trades
        )

    # ── 3. Momentum-only margin split ──
    print(f"\n{'─'*70}")
    print(f"  MOMENTUM TRADES BY MARGIN (label=momentum only)")
    print(f"{'─'*70}")

    momentum_trades = [t for t in allowed_trades if t["regime_label"] == "momentum"]

    mom_margin_buckets = {
        "mom_low (0.5<=m<1.0)": lambda t: 0.5 <= t["margin"] < 1.0,
        "mom_mid (1.0<=m<1.5)": lambda t: 1.0 <= t["margin"] < 1.5,
        "mom_high (m>=1.5)": lambda t: t["margin"] >= 1.5,
    }

    results["momentum_by_margin"] = {}
    for bucket_name, pred in mom_margin_buckets.items():
        bucket_trades = [t for t in momentum_trades if pred(t)]
        results["momentum_by_margin"][bucket_name] = _summarize(
            f"{bucket_name}", bucket_trades
        )

    # ── 4. Momentum score split (all allowed trades) ──
    print(f"\n{'─'*70}")
    print(f"  ALL ALLOWED TRADES BY MOMENTUM SCORE")
    print(f"{'─'*70}")

    mom_score_buckets = {
        "mom_score_0-1": lambda t: t["momentum_score"] < 1.5,
        "mom_score_1.5-2": lambda t: 1.5 <= t["momentum_score"] < 2.5,
        "mom_score_2.5-3": lambda t: 2.5 <= t["momentum_score"] < 3.5,
        "mom_score_3.5-4": lambda t: t["momentum_score"] >= 3.5,
    }

    results["by_momentum_score"] = {}
    for bucket_name, pred in mom_score_buckets.items():
        bucket_trades = [t for t in allowed_trades if pred(t)]
        results["by_momentum_score"][bucket_name] = _summarize(
            f"MOMENTUM SCORE: {bucket_name}", bucket_trades
        )

    # ── 5. Near-momentum ambiguous deep dive ──
    print(f"\n{'─'*70}")
    print(f"  NEAR-MOMENTUM AMBIGUOUS DEEP DIVE")
    print(f"{'─'*70}")

    ambig_allowed = [t for t in allowed_trades if t["regime_label"] == "ambiguous"]
    if ambig_allowed:
        # Split by momentum score within ambiguous
        ambig_mom_buckets = {
            "ambig_mom_score<2": lambda t: t["momentum_score"] < 2.0,
            "ambig_mom_score>=2": lambda t: t["momentum_score"] >= 2.0,
        }

        results["ambiguous_by_mom_score"] = {}
        for bucket_name, pred in ambig_mom_buckets.items():
            bucket_trades = [t for t in ambig_allowed if pred(t)]
            results["ambiguous_by_mom_score"][bucket_name] = _summarize(
                f"{bucket_name}", bucket_trades
            )

    # ── 6. Summary table ──
    print(f"\n{'─'*70}")
    print(f"  SUMMARY: POTENTIAL SIZING TIERS")
    print(f"{'─'*70}")

    tiers = {
        "FULL SIZE (momentum, margin>=1.0)": [
            t for t in allowed_trades
            if t["regime_label"] == "momentum" and t["margin"] >= 1.0
        ],
        "STANDARD (momentum, margin<1.0)": [
            t for t in allowed_trades
            if t["regime_label"] == "momentum" and t["margin"] < 1.0
        ],
        "REDUCED (mean_reversion label)": [
            t for t in allowed_trades
            if t["regime_label"] == "mean_reversion"
        ],
        "MINIMUM (ambiguous, momentum-topped)": [
            t for t in allowed_trades
            if t["regime_label"] == "ambiguous"
        ],
    }

    results["sizing_tiers"] = {}
    for tier_name, trades in tiers.items():
        results["sizing_tiers"][tier_name] = _summarize(tier_name, trades)

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

    out_path = ROOT / "research_out" / "diagnostic_v44_margin_split.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults written to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
