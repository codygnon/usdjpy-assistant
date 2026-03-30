#!/usr/bin/env python3
"""
V44 Trade-Level ΔER Diagnostic.

The day-level no-trade diagnostic showed ΔER < 0 days are net-negative on 1000k.
This diagnostic checks whether ΔER at the moment of V44 ENTRY predicts V44
trade outcomes more sharply than the day-level average.

For each V44 trade that passes Variant F:
  1. Look up ΔER at entry time (from M5 dynamic features)
  2. Also look up ER level, touch density, and failed continuation
  3. Split by ΔER buckets and cross with win/loss
  4. Check if ΔER < 0 entries bleed disproportionately

If ΔER < 0 at V44 entry predicts losses, this becomes a Variant H candidate:
  Variant H = Variant G + block V44 when ΔER < 0 at entry
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

from core.regime_features import (
    compute_delta_efficiency_ratio,
    compute_delta_touch_density,
    compute_efficiency_ratio,
    compute_failed_continuation,
    compute_level_touch_density,
)
from core.regime_classifier import RegimeThresholds
from scripts import validate_regime_classifier as regime_validation
from scripts.backtest_v44_conservative_router import _lookup_regime


def _build_m5(input_csv: str) -> pd.DataFrame:
    from scripts.regime_threshold_analysis import _load_m1, _resample
    m1 = _load_m1(input_csv)
    return _resample(m1, "5min")


def _compute_dynamic_features_on_m5(m5: pd.DataFrame) -> pd.DataFrame:
    n = len(m5)
    er_vals = np.full(n, 0.5)
    touch_vals = np.full(n, 0.0)
    delta_er_vals = np.full(n, 0.0)
    delta_touch_vals = np.full(n, 0.0)
    failed_cont_vals = np.full(n, 0.0)

    ema9 = m5["close"].ewm(span=9, adjust=False).mean()
    ema21 = m5["close"].ewm(span=21, adjust=False).mean()
    trend_sign = np.sign((ema9 - ema21).values)

    for i in range(n):
        window = m5.iloc[max(0, i - 60):i + 1]
        if len(window) < 5:
            continue
        ts = int(trend_sign[i]) if not np.isnan(trend_sign[i]) else 0
        er_vals[i] = compute_efficiency_ratio(window, lookback=12)
        touch_vals[i] = compute_level_touch_density(window, lookback=20, zone_pips=5.0)
        delta_er_vals[i] = compute_delta_efficiency_ratio(window, lookback=12, delta_bars=3)
        delta_touch_vals[i] = compute_delta_touch_density(window, lookback=20, delta_bars=3, zone_pips=5.0)
        failed_cont_vals[i] = compute_failed_continuation(
            window,
            trend_sign=ts,
            lookback=12,
            stall_bars=4,
            proximity_pips=5.0,
        )

    out = m5.copy()
    out["sf_efficiency_ratio"] = er_vals
    out["sf_touch_density"] = touch_vals
    out["sf_delta_efficiency_ratio"] = delta_er_vals
    out["sf_delta_touch_density"] = delta_touch_vals
    out["sf_failed_continuation"] = failed_cont_vals
    return out


def _load_classified_with_dynamic(input_csv: str) -> pd.DataFrame:
    m1 = regime_validation._load_m1(input_csv)
    featured = regime_validation.compute_features(m1)
    classified = regime_validation.classify_all_bars(featured, RegimeThresholds())
    m5_dynamic = _compute_dynamic_features_on_m5(_build_m5(input_csv))
    dynamic_cols = [
        "time",
        "sf_efficiency_ratio",
        "sf_touch_density",
        "sf_delta_efficiency_ratio",
        "sf_delta_touch_density",
        "sf_failed_continuation",
    ]
    return pd.merge_asof(
        classified.sort_values("time"),
        m5_dynamic[dynamic_cols].sort_values("time"),
        on="time",
        direction="backward",
    )


def _lookup_row(classified: pd.DataFrame, ts: pd.Timestamp, time_idx: pd.DatetimeIndex) -> pd.Series | None:
    idx = time_idx.get_indexer([pd.Timestamp(ts)], method="ffill")[0]
    if idx < 0:
        idx = time_idx.get_indexer([pd.Timestamp(ts)], method="bfill")[0]
    if idx < 0:
        return None
    return classified.iloc[idx]


def _is_variant_f_allowed(label: str, top_regime: str) -> bool:
    """Reproduce Variant F filter logic."""
    if label in ("breakout", "post_breakout_trend"):
        return False
    if label == "ambiguous" and top_regime != "momentum":
        return False
    return True


def run_diagnostic(dataset_label: str, input_csv: str, baseline_json: str) -> dict:
    print(f"\n{'='*70}")
    print(f"  V44 TRADE-LEVEL ΔER DIAGNOSTIC: {dataset_label}")
    print(f"{'='*70}")

    with open(baseline_json, encoding="utf-8") as f:
        baseline = json.load(f)

    v44_trades = [t for t in baseline["closed_trades"] if t["strategy"] == "v44_ny"]
    print(f"Total V44 trades (pre-filter): {len(v44_trades)}")

    print("Building classified bars with dynamic features...")
    classified = _load_classified_with_dynamic(input_csv)
    time_idx = pd.DatetimeIndex(classified["time"])

    # Tag each trade with regime + dynamic features, filter through Variant F
    allowed_trades = []

    for t in v44_trades:
        entry_ts = pd.Timestamp(t["entry_time"])
        regime = _lookup_regime(classified, entry_ts)
        scores = regime["regime_scores"]
        label = regime["regime_label"]
        top_regime = max(scores, key=scores.get) if scores else "unknown"

        if not _is_variant_f_allowed(label, top_regime):
            continue

        row = _lookup_row(classified, entry_ts, time_idx)

        def _safe_float(val, default=0.0):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return default
            return float(val)

        delta_er = _safe_float(row.get("sf_delta_efficiency_ratio") if row is not None else None)
        er = _safe_float(row.get("sf_efficiency_ratio") if row is not None else None, 0.5)
        touch = _safe_float(row.get("sf_touch_density") if row is not None else None)
        delta_touch = _safe_float(row.get("sf_delta_touch_density") if row is not None else None)
        failed_cont = _safe_float(row.get("sf_failed_continuation") if row is not None else None)

        allowed_trades.append({
            "entry_time": t["entry_time"],
            "exit_time": t["exit_time"],
            "side": t["side"],
            "pips": t["pips"],
            "usd": t["usd"],
            "exit_reason": t["exit_reason"],
            "regime_label": label,
            "top_regime": top_regime,
            "delta_er": delta_er,
            "er": er,
            "touch_density": touch,
            "delta_touch": delta_touch,
            "failed_cont": failed_cont,
            "mom_score": scores.get("momentum", 0),
        })

    print(f"Variant F allowed V44 trades: {len(allowed_trades)}")

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

        # Side breakdown
        buy_trades = [t for t in trades if t["side"] == "buy"]
        sell_trades = [t for t in trades if t["side"] == "sell"]
        buy_net = sum(t["pips"] for t in buy_trades)
        sell_net = sum(t["pips"] for t in sell_trades)
        print(f"    Sides: buy={len(buy_trades)}({buy_net:+.1f}p) sell={len(sell_trades)}({sell_net:+.1f}p)")

        # Avg dynamic features
        avg_der = np.mean([t["delta_er"] for t in trades])
        avg_er = np.mean([t["er"] for t in trades])
        avg_fc = np.mean([t["failed_cont"] for t in trades])
        print(f"    Avg ΔER: {avg_der:+.3f}  |  Avg ER: {avg_er:.3f}  |  Avg failed_cont: {avg_fc:.3f}")

        return {
            "count": len(trades),
            "winners": len(winners),
            "losers": len(losers),
            "win_rate_pct": round(win_rate, 1),
            "net_pips": round(net_pips, 1),
            "net_usd": round(net_usd, 2),
            "pf": round(pf, 3),
            "avg_pips": round(avg_pips, 1),
            "avg_delta_er": round(float(avg_der), 3),
            "avg_er": round(float(avg_er), 3),
        }

    results = {}

    # ── 1. Primary split: ΔER at entry ──
    print(f"\n{'─'*70}")
    print(f"  PRIMARY: ΔER AT V44 ENTRY")
    print(f"{'─'*70}")

    der_neg = [t for t in allowed_trades if t["delta_er"] < 0]
    der_pos = [t for t in allowed_trades if t["delta_er"] >= 0]

    results["delta_er_split"] = {
        "negative": _summarize("ΔER < 0 (efficiency deteriorating)", der_neg),
        "positive": _summarize("ΔER >= 0 (efficiency stable/improving)", der_pos),
    }

    # ── 2. Finer ΔER buckets ──
    print(f"\n{'─'*70}")
    print(f"  FINE ΔER BUCKETS")
    print(f"{'─'*70}")

    der_buckets = {
        "ΔER < -0.15 (sharp collapse)": lambda t: t["delta_er"] < -0.15,
        "ΔER -0.15 to -0.05": lambda t: -0.15 <= t["delta_er"] < -0.05,
        "ΔER -0.05 to 0": lambda t: -0.05 <= t["delta_er"] < 0,
        "ΔER 0 to +0.10": lambda t: 0 <= t["delta_er"] < 0.10,
        "ΔER +0.10 to +0.25": lambda t: 0.10 <= t["delta_er"] < 0.25,
        "ΔER >= +0.25 (strong improvement)": lambda t: t["delta_er"] >= 0.25,
    }

    results["delta_er_buckets"] = {}
    for name, pred in der_buckets.items():
        trades = [t for t in allowed_trades if pred(t)]
        results["delta_er_buckets"][name] = _summarize(name, trades)

    # ── 3. ΔER × regime label ──
    print(f"\n{'─'*70}")
    print(f"  ΔER × REGIME LABEL")
    print(f"{'─'*70}")

    results["delta_er_x_regime"] = {}
    for regime in ["momentum", "ambiguous"]:
        regime_trades = [t for t in allowed_trades if t["regime_label"] == regime]
        neg = [t for t in regime_trades if t["delta_er"] < 0]
        pos = [t for t in regime_trades if t["delta_er"] >= 0]
        results["delta_er_x_regime"][f"{regime}_neg"] = _summarize(
            f"{regime.upper()} × ΔER < 0", neg
        )
        results["delta_er_x_regime"][f"{regime}_pos"] = _summarize(
            f"{regime.upper()} × ΔER >= 0", pos
        )

    # ── 4. ΔER × ER level (double condition) ──
    print(f"\n{'─'*70}")
    print(f"  ΔER × ER LEVEL (double condition)")
    print(f"{'─'*70}")

    double_conditions = {
        "ΔER<0 AND ER<0.4 (collapsing + already low)": lambda t: t["delta_er"] < 0 and t["er"] < 0.4,
        "ΔER<0 AND ER>=0.4 (collapsing from healthy)": lambda t: t["delta_er"] < 0 and t["er"] >= 0.4,
        "ΔER>=0 AND ER<0.4 (stable but choppy)": lambda t: t["delta_er"] >= 0 and t["er"] < 0.4,
        "ΔER>=0 AND ER>=0.4 (healthy trend)": lambda t: t["delta_er"] >= 0 and t["er"] >= 0.4,
    }

    results["double_conditions"] = {}
    for name, pred in double_conditions.items():
        trades = [t for t in allowed_trades if pred(t)]
        results["double_conditions"][name] = _summarize(name, trades)

    # ── 5. ΔER × momentum score ──
    print(f"\n{'─'*70}")
    print(f"  ΔER × MOMENTUM SCORE")
    print(f"{'─'*70}")

    der_x_mom = {
        "ΔER<0, mom>=3.5": lambda t: t["delta_er"] < 0 and t["mom_score"] >= 3.5,
        "ΔER<0, mom<3.5": lambda t: t["delta_er"] < 0 and t["mom_score"] < 3.5,
        "ΔER>=0, mom>=3.5": lambda t: t["delta_er"] >= 0 and t["mom_score"] >= 3.5,
        "ΔER>=0, mom<3.5": lambda t: t["delta_er"] >= 0 and t["mom_score"] < 3.5,
    }

    results["delta_er_x_mom"] = {}
    for name, pred in der_x_mom.items():
        trades = [t for t in allowed_trades if pred(t)]
        results["delta_er_x_mom"][name] = _summarize(name, trades)

    # ── 6. Winners vs losers feature profile ──
    print(f"\n{'─'*70}")
    print(f"  FEATURE PROFILE: V44 WINNERS vs LOSERS")
    print(f"{'─'*70}")

    winners = [t for t in allowed_trades if t["pips"] > 0]
    losers = [t for t in allowed_trades if t["pips"] <= 0]

    for label, group in [("WINNERS", winners), ("LOSERS", losers)]:
        if not group:
            continue
        avg_der = np.mean([t["delta_er"] for t in group])
        avg_er = np.mean([t["er"] for t in group])
        avg_touch = np.mean([t["touch_density"] for t in group])
        avg_dt = np.mean([t["delta_touch"] for t in group])
        avg_fc = np.mean([t["failed_cont"] for t in group])
        print(f"\n  {label} ({len(group)} trades):")
        print(f"    Avg ΔER:             {avg_der:+.3f}")
        print(f"    Avg ER:              {avg_er:.3f}")
        print(f"    Avg touch density:   {avg_touch:.3f}")
        print(f"    Avg Δtouch:          {avg_dt:+.3f}")
        print(f"    Avg failed_cont:     {avg_fc:.3f}")

    # ── 7. Summary: blocking impact ──
    print(f"\n{'─'*70}")
    print(f"  BLOCKING IMPACT SIMULATION")
    print(f"{'─'*70}")

    total_pips = sum(t["pips"] for t in allowed_trades)
    total_trades = len(allowed_trades)

    block_rules = {
        "Block ΔER < 0": lambda t: t["delta_er"] < 0,
        "Block ΔER < -0.05": lambda t: t["delta_er"] < -0.05,
        "Block ΔER < -0.10": lambda t: t["delta_er"] < -0.10,
        "Block ΔER < -0.15": lambda t: t["delta_er"] < -0.15,
        "Block ΔER<0 AND ER<0.4": lambda t: t["delta_er"] < 0 and t["er"] < 0.4,
        "Block ΔER<0 AND mom<3.5": lambda t: t["delta_er"] < 0 and t["mom_score"] < 3.5,
    }

    print(f"\n  Baseline: {total_pips:+.1f} pips, {total_trades} trades")
    print(f"\n  {'Rule':<35s} {'Blocked':>8s} {'B-W':>5s} {'B-L':>5s} {'Pips Cut':>10s} {'Remaining':>10s} {'Δ':>8s}")
    print(f"  {'─'*85}")

    results["blocking_impact"] = {}
    for rule_name, pred in block_rules.items():
        blocked = [t for t in allowed_trades if pred(t)]
        kept = [t for t in allowed_trades if not pred(t)]
        b_winners = sum(1 for t in blocked if t["pips"] > 0)
        b_losers = sum(1 for t in blocked if t["pips"] <= 0)
        blocked_pips = sum(t["pips"] for t in blocked)
        remaining_pips = sum(t["pips"] for t in kept)
        delta = -blocked_pips

        print(f"  {rule_name:<35s} {len(blocked):>8d} {b_winners:>5d} {b_losers:>5d} "
              f"{blocked_pips:>+10.1f} {remaining_pips:>+10.1f} {delta:>+8.1f}")

        results["blocking_impact"][rule_name] = {
            "blocked": len(blocked),
            "blocked_winners": b_winners,
            "blocked_losers": b_losers,
            "blocked_pips": round(blocked_pips, 1),
            "remaining_pips": round(remaining_pips, 1),
            "delta_pips": round(delta, 1),
        }

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

    out_path = ROOT / "research_out" / "diagnostic_v44_delta_er.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults written to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
