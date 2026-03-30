#!/usr/bin/env python3
"""
No-Trade Intelligence Diagnostic.

Identify days/windows where the system should stand down entirely.
Groups all trades by calendar day, finds "bad days" where multiple strategies
lost simultaneously, and checks the regime pattern on those days.

Looking for:
  1. Coordinated loss days — days where 2+ strategies lose
  2. Regime signatures on bad vs good days
  3. Whether a regime-based "stand down" signal could have avoided losses
  4. Time-of-day patterns within bad days

A "stand down" signal is only valuable if it:
  - Catches a meaningful portion of losses
  - Doesn't cut too many winning days
  - Has a clean, implementable trigger
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


def _load_classified_bars(input_csv: str) -> pd.DataFrame:
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


def _lookup_row(classified: pd.DataFrame, ts: pd.Timestamp) -> pd.Series | None:
    time_idx = pd.DatetimeIndex(classified["time"])
    idx = time_idx.get_indexer([pd.Timestamp(ts)], method="ffill")[0]
    if idx < 0:
        idx = time_idx.get_indexer([pd.Timestamp(ts)], method="bfill")[0]
    if idx < 0:
        return None
    return classified.iloc[idx]


def run_diagnostic(dataset_label: str, input_csv: str, baseline_json: str) -> dict:
    print(f"\n{'='*70}")
    print(f"  NO-TRADE INTELLIGENCE: {dataset_label}")
    print(f"{'='*70}")

    with open(baseline_json, encoding="utf-8") as f:
        baseline = json.load(f)

    trades = baseline["closed_trades"]
    print(f"Total trades: {len(trades)}")

    print("Classifying bars...")
    classified = _load_classified_bars(input_csv)

    # ── Group trades by calendar day ──
    days: dict[str, list] = defaultdict(list)
    for t in trades:
        entry_ts = pd.Timestamp(t["entry_time"])
        day_key = entry_ts.strftime("%Y-%m-%d")

        regime = _lookup_regime(classified, entry_ts)
        row = _lookup_row(classified, entry_ts)
        scores = regime["regime_scores"]
        label = regime["regime_label"]
        top_regime = max(scores, key=scores.get) if scores else "unknown"
        mom_score = scores.get("momentum", 0)

        days[day_key].append({
            "strategy": t["strategy"],
            "side": t["side"],
            "pips": t["pips"],
            "usd": t["usd"],
            "exit_reason": t["exit_reason"],
            "entry_hour": entry_ts.hour,
            "regime_label": label,
            "top_regime": top_regime,
            "mom_score": mom_score,
            "efficiency_ratio": float(row.get("sf_efficiency_ratio", 0.5)) if row is not None and pd.notna(row.get("sf_efficiency_ratio")) else 0.5,
            "touch_density": float(row.get("sf_touch_density", 0.0)) if row is not None and pd.notna(row.get("sf_touch_density")) else 0.0,
            "delta_efficiency_ratio": float(row.get("sf_delta_efficiency_ratio", 0.0)) if row is not None and pd.notna(row.get("sf_delta_efficiency_ratio")) else 0.0,
            "delta_touch_density": float(row.get("sf_delta_touch_density", 0.0)) if row is not None and pd.notna(row.get("sf_delta_touch_density")) else 0.0,
            "failed_continuation": float(row.get("sf_failed_continuation", 0.0)) if row is not None and pd.notna(row.get("sf_failed_continuation")) else 0.0,
            "day_of_week": entry_ts.day_name(),
        })

    print(f"Trading days: {len(days)}")

    # ── Compute daily summaries ──
    day_summaries = []
    for day_key, day_trades in sorted(days.items()):
        by_strat = defaultdict(lambda: {"pips": 0.0, "usd": 0.0, "count": 0, "wins": 0, "losses": 0})
        for t in day_trades:
            s = by_strat[t["strategy"]]
            s["pips"] += t["pips"]
            s["usd"] += t["usd"]
            s["count"] += 1
            if t["pips"] > 0:
                s["wins"] += 1
            else:
                s["losses"] += 1

        total_pips = sum(t["pips"] for t in day_trades)
        total_usd = sum(t["usd"] for t in day_trades)
        total_count = len(day_trades)

        # Strategy-level P&L
        strat_pips = {s: d["pips"] for s, d in by_strat.items()}
        strat_losing = [s for s, p in strat_pips.items() if p < 0]
        strat_winning = [s for s, p in strat_pips.items() if p > 0]

        # Dominant regime for the day (mode of entry regimes)
        regime_counts = defaultdict(int)
        top_regime_counts = defaultdict(int)
        mom_scores = []
        er_vals = []
        touch_vals = []
        delta_er_vals = []
        delta_touch_vals = []
        failed_cont_vals = []
        for t in day_trades:
            regime_counts[t["regime_label"]] += 1
            top_regime_counts[t["top_regime"]] += 1
            mom_scores.append(t["mom_score"])
            er_vals.append(t["efficiency_ratio"])
            touch_vals.append(t["touch_density"])
            delta_er_vals.append(t["delta_efficiency_ratio"])
            delta_touch_vals.append(t["delta_touch_density"])
            failed_cont_vals.append(t["failed_continuation"])

        dominant_label = max(regime_counts, key=regime_counts.get)
        dominant_top = max(top_regime_counts, key=top_regime_counts.get)
        avg_mom_score = np.mean(mom_scores)

        day_summaries.append({
            "day": day_key,
            "day_of_week": day_trades[0]["day_of_week"],
            "total_pips": round(total_pips, 1),
            "total_usd": round(total_usd, 2),
            "total_count": total_count,
            "strat_pips": {s: round(p, 1) for s, p in strat_pips.items()},
            "strat_losing": strat_losing,
            "strat_winning": strat_winning,
            "n_strats_losing": len(strat_losing),
            "n_strats_active": len(by_strat),
            "dominant_label": dominant_label,
            "dominant_top": dominant_top,
            "avg_mom_score": round(avg_mom_score, 2),
            "avg_efficiency_ratio": round(float(np.mean(er_vals)), 3),
            "avg_touch_density": round(float(np.mean(touch_vals)), 3),
            "avg_delta_efficiency_ratio": round(float(np.mean(delta_er_vals)), 3),
            "avg_delta_touch_density": round(float(np.mean(delta_touch_vals)), 3),
            "failed_continuation_ratio": round(float(np.mean(failed_cont_vals)), 3),
            "regime_mix": dict(regime_counts),
            "top_regime_mix": dict(top_regime_counts),
        })

    # ── 1. Day classification ──
    print(f"\n{'─'*70}")
    print(f"  DAY CLASSIFICATION")
    print(f"{'─'*70}")

    bad_days = [d for d in day_summaries if d["n_strats_losing"] >= 2]
    ugly_days = [d for d in day_summaries if d["n_strats_losing"] >= 2 and d["total_pips"] < -10]
    single_loss_days = [d for d in day_summaries if d["n_strats_losing"] == 1]
    green_days = [d for d in day_summaries if d["n_strats_losing"] == 0 and d["total_pips"] > 0]
    flat_days = [d for d in day_summaries if d["total_pips"] == 0 or d["n_strats_active"] == 0]

    print(f"\n  Green days (0 strats losing, net positive): {len(green_days)}")
    print(f"  Single-loss days (1 strat losing): {len(single_loss_days)}")
    print(f"  Bad days (2+ strats losing): {len(bad_days)}")
    print(f"  Ugly days (2+ losing AND net < -10p): {len(ugly_days)}")

    total_bad_pips = sum(d["total_pips"] for d in bad_days)
    total_ugly_pips = sum(d["total_pips"] for d in ugly_days)
    total_green_pips = sum(d["total_pips"] for d in green_days)
    total_single_pips = sum(d["total_pips"] for d in single_loss_days)

    print(f"\n  Green day pips:       {total_green_pips:+.1f} ({len(green_days)} days)")
    print(f"  Single-loss day pips: {total_single_pips:+.1f} ({len(single_loss_days)} days)")
    print(f"  Bad day pips:         {total_bad_pips:+.1f} ({len(bad_days)} days)")
    print(f"  Ugly day pips:        {total_ugly_pips:+.1f} ({len(ugly_days)} days)")

    # ── 2. Regime profile of bad vs good days ──
    print(f"\n{'─'*70}")
    print(f"  REGIME PROFILE: BAD DAYS vs GREEN DAYS")
    print(f"{'─'*70}")

    def _regime_profile(label: str, days_list: list):
        if not days_list:
            print(f"\n  {label}: no days")
            return {}
        label_counts = defaultdict(int)
        top_counts = defaultdict(int)
        mom_scores = []
        for d in days_list:
            label_counts[d["dominant_label"]] += 1
            top_counts[d["dominant_top"]] += 1
            mom_scores.append(d["avg_mom_score"])

        print(f"\n  {label} ({len(days_list)} days):")
        print(f"    Dominant label distribution:")
        for regime in ["momentum", "mean_reversion", "breakout", "post_breakout_trend", "ambiguous"]:
            count = label_counts.get(regime, 0)
            pct = 100.0 * count / len(days_list)
            print(f"      {regime:<25s}: {count:>4d} ({pct:>5.1f}%)")
        print(f"    Dominant top-score distribution:")
        for regime in ["momentum", "mean_reversion", "breakout", "post_breakout_trend"]:
            count = top_counts.get(regime, 0)
            pct = 100.0 * count / len(days_list)
            print(f"      {regime:<25s}: {count:>4d} ({pct:>5.1f}%)")
        print(f"    Avg momentum score: {np.mean(mom_scores):.2f}")

        return {
            "dominant_label_dist": dict(label_counts),
            "dominant_top_dist": dict(top_counts),
            "avg_mom_score": round(float(np.mean(mom_scores)), 2),
        }

    green_profile = _regime_profile("GREEN DAYS", green_days)
    bad_profile = _regime_profile("BAD DAYS (2+ losing)", bad_days)
    ugly_profile = _regime_profile("UGLY DAYS (2+ losing, < -10p)", ugly_days)

    # ── 2b. Dynamic feature profile of bad vs good days ──
    print(f"\n{'─'*70}")
    print(f"  DYNAMIC FEATURE PROFILE: BAD DAYS vs GREEN DAYS")
    print(f"{'─'*70}")

    def _dynamic_profile(label: str, days_list: list):
        if not days_list:
            print(f"\n  {label}: no days")
            return {}
        avg_er = np.mean([d["avg_efficiency_ratio"] for d in days_list])
        avg_touch = np.mean([d["avg_touch_density"] for d in days_list])
        avg_delta_er = np.mean([d["avg_delta_efficiency_ratio"] for d in days_list])
        avg_delta_touch = np.mean([d["avg_delta_touch_density"] for d in days_list])
        avg_failed = np.mean([d["failed_continuation_ratio"] for d in days_list])
        print(f"\n  {label} ({len(days_list)} days):")
        print(f"    Avg ER:              {avg_er:.3f}")
        print(f"    Avg touch density:   {avg_touch:.3f}")
        print(f"    Avg ΔER:             {avg_delta_er:+.3f}")
        print(f"    Avg Δtouch density:  {avg_delta_touch:+.3f}")
        print(f"    Failed continuation: {avg_failed:.3f}")
        return {
            "avg_efficiency_ratio": round(float(avg_er), 3),
            "avg_touch_density": round(float(avg_touch), 3),
            "avg_delta_efficiency_ratio": round(float(avg_delta_er), 3),
            "avg_delta_touch_density": round(float(avg_delta_touch), 3),
            "failed_continuation_ratio": round(float(avg_failed), 3),
        }

    green_dynamic = _dynamic_profile("GREEN DAYS", green_days)
    bad_dynamic = _dynamic_profile("BAD DAYS (2+ losing)", bad_days)
    ugly_dynamic = _dynamic_profile("UGLY DAYS (2+ losing, < -10p)", ugly_days)

    # ── 3. Day-of-week analysis ──
    print(f"\n{'─'*70}")
    print(f"  DAY OF WEEK ANALYSIS")
    print(f"{'─'*70}")

    dow_stats = defaultdict(lambda: {"count": 0, "pips": 0.0, "bad": 0, "green": 0})
    for d in day_summaries:
        dow = d["day_of_week"]
        dow_stats[dow]["count"] += 1
        dow_stats[dow]["pips"] += d["total_pips"]
        if d["n_strats_losing"] >= 2:
            dow_stats[dow]["bad"] += 1
        if d["n_strats_losing"] == 0 and d["total_pips"] > 0:
            dow_stats[dow]["green"] += 1

    print(f"\n  {'Day':<12s} {'Days':>6s} {'Net Pips':>10s} {'Avg':>8s} {'Bad':>5s} {'Green':>6s} {'Bad%':>6s}")
    print(f"  {'─'*55}")
    for dow in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
        s = dow_stats.get(dow, {"count": 0, "pips": 0, "bad": 0, "green": 0})
        if s["count"] == 0:
            continue
        avg = s["pips"] / s["count"]
        bad_pct = 100.0 * s["bad"] / s["count"]
        print(f"  {dow:<12s} {s['count']:>6d} {s['pips']:>+10.1f} {avg:>+8.1f} {s['bad']:>5d} {s['green']:>6d} {bad_pct:>5.1f}%")

    # ── 4. Worst days detail ──
    print(f"\n{'─'*70}")
    print(f"  WORST 15 DAYS (by total pips)")
    print(f"{'─'*70}")

    worst_days = sorted(day_summaries, key=lambda d: d["total_pips"])[:15]
    print(f"\n  {'Day':<12s} {'DoW':<10s} {'Pips':>8s} {'#T':>4s} {'#SL':>4s} {'Label':<15s} {'Top':>15s} {'Mom':>5s} {'Strat P&L'}")
    print(f"  {'─'*100}")
    for d in worst_days:
        strat_str = " | ".join(f"{s}:{p:+.1f}" for s, p in sorted(d["strat_pips"].items()))
        print(f"  {d['day']:<12s} {d['day_of_week']:<10s} {d['total_pips']:>+8.1f} {d['total_count']:>4d} "
              f"{d['n_strats_losing']:>4d} {d['dominant_label']:<15s} {d['dominant_top']:>15s} "
              f"{d['avg_mom_score']:>5.2f} {strat_str}")

    # ── 5. Potential stand-down signals ──
    print(f"\n{'─'*70}")
    print(f"  POTENTIAL STAND-DOWN SIGNALS")
    print(f"{'─'*70}")

    # Signal A: dominant label = ambiguous
    ambig_days = [d for d in day_summaries if d["dominant_label"] == "ambiguous"]
    non_ambig_days = [d for d in day_summaries if d["dominant_label"] != "ambiguous"]

    ambig_pips = sum(d["total_pips"] for d in ambig_days)
    non_ambig_pips = sum(d["total_pips"] for d in non_ambig_days)
    ambig_avg = ambig_pips / len(ambig_days) if ambig_days else 0
    non_ambig_avg = non_ambig_pips / len(non_ambig_days) if non_ambig_days else 0

    print(f"\n  Signal A: Dominant regime = ambiguous")
    print(f"    Ambiguous days: {len(ambig_days)}, net pips: {ambig_pips:+.1f}, avg: {ambig_avg:+.1f}")
    print(f"    Non-ambiguous days: {len(non_ambig_days)}, net pips: {non_ambig_pips:+.1f}, avg: {non_ambig_avg:+.1f}")

    # Signal B: low momentum score (avg < 2.0)
    low_mom_days = [d for d in day_summaries if d["avg_mom_score"] < 2.0]
    high_mom_days = [d for d in day_summaries if d["avg_mom_score"] >= 2.0]

    low_mom_pips = sum(d["total_pips"] for d in low_mom_days)
    high_mom_pips = sum(d["total_pips"] for d in high_mom_days)

    print(f"\n  Signal B: Avg momentum score < 2.0")
    print(f"    Low mom days: {len(low_mom_days)}, net pips: {low_mom_pips:+.1f}, avg: {low_mom_pips/max(1,len(low_mom_days)):+.1f}")
    print(f"    High mom days: {len(high_mom_days)}, net pips: {high_mom_pips:+.1f}, avg: {high_mom_pips/max(1,len(high_mom_days)):+.1f}")

    # Signal C: dominant top = mean_reversion
    mr_top_days = [d for d in day_summaries if d["dominant_top"] == "mean_reversion"]
    non_mr_top_days = [d for d in day_summaries if d["dominant_top"] != "mean_reversion"]

    mr_pips = sum(d["total_pips"] for d in mr_top_days)
    non_mr_pips = sum(d["total_pips"] for d in non_mr_top_days)

    print(f"\n  Signal C: Dominant top-score = mean_reversion")
    print(f"    MR-topped days: {len(mr_top_days)}, net pips: {mr_pips:+.1f}, avg: {mr_pips/max(1,len(mr_top_days)):+.1f}")
    print(f"    Non-MR days: {len(non_mr_top_days)}, net pips: {non_mr_pips:+.1f}, avg: {non_mr_pips/max(1,len(non_mr_top_days)):+.1f}")

    # Signal D: no regime has margin >= 0.5 (all entries ambiguous)
    all_ambig_days = [d for d in day_summaries
                      if d["regime_mix"].get("ambiguous", 0) == d["total_count"]]
    some_clear_days = [d for d in day_summaries
                       if d["regime_mix"].get("ambiguous", 0) < d["total_count"]]

    all_ambig_pips = sum(d["total_pips"] for d in all_ambig_days)
    some_clear_pips = sum(d["total_pips"] for d in some_clear_days)

    print(f"\n  Signal D: ALL trades on day entered in ambiguous regime")
    print(f"    All-ambiguous days: {len(all_ambig_days)}, net pips: {all_ambig_pips:+.1f}, avg: {all_ambig_pips/max(1,len(all_ambig_days)):+.1f}")
    print(f"    Some-clear days: {len(some_clear_days)}, net pips: {some_clear_pips:+.1f}, avg: {some_clear_pips/max(1,len(some_clear_days)):+.1f}")

    # Signal E: dominant label = mean_reversion
    mr_label_days = [d for d in day_summaries if d["dominant_label"] == "mean_reversion"]
    non_mr_label_days = [d for d in day_summaries if d["dominant_label"] != "mean_reversion"]

    mr_label_pips = sum(d["total_pips"] for d in mr_label_days)
    non_mr_label_pips = sum(d["total_pips"] for d in non_mr_label_days)

    print(f"\n  Signal E: Dominant label = mean_reversion")
    print(f"    MR-label days: {len(mr_label_days)}, net pips: {mr_label_pips:+.1f}, avg: {mr_label_pips/max(1,len(mr_label_days)):+.1f}")
    print(f"    Non-MR-label days: {len(non_mr_label_days)}, net pips: {non_mr_label_pips:+.1f}, avg: {non_mr_label_pips/max(1,len(non_mr_label_days)):+.1f}")

    # Signal F: multi-regime day (3+ different labels)
    multi_regime_days = [d for d in day_summaries if len(d["regime_mix"]) >= 3]
    few_regime_days = [d for d in day_summaries if len(d["regime_mix"]) < 3]

    multi_pips = sum(d["total_pips"] for d in multi_regime_days)
    few_pips = sum(d["total_pips"] for d in few_regime_days)

    print(f"\n  Signal F: 3+ different regime labels in one day")
    print(f"    Multi-regime days: {len(multi_regime_days)}, net pips: {multi_pips:+.1f}, avg: {multi_pips/max(1,len(multi_regime_days)):+.1f}")
    print(f"    Coherent days: {len(few_regime_days)}, net pips: {few_pips:+.1f}, avg: {few_pips/max(1,len(few_regime_days)):+.1f}")

    # Signal G: ER deteriorating on the day
    er_decay_days = [d for d in day_summaries if d["avg_delta_efficiency_ratio"] < 0.0]
    er_stable_days = [d for d in day_summaries if d["avg_delta_efficiency_ratio"] >= 0.0]
    print(f"\n  Signal G: Avg ΔER < 0 (efficiency deteriorating)")
    print(f"    ER-collapse days: {len(er_decay_days)}, net pips: {sum(d['total_pips'] for d in er_decay_days):+.1f}, avg: {sum(d['total_pips'] for d in er_decay_days)/max(1,len(er_decay_days)):+.1f}")
    print(f"    ER-stable days:   {len(er_stable_days)}, net pips: {sum(d['total_pips'] for d in er_stable_days):+.1f}, avg: {sum(d['total_pips'] for d in er_stable_days)/max(1,len(er_stable_days)):+.1f}")

    # Signal H: touch density increasing on the day
    touch_rising_days = [d for d in day_summaries if d["avg_delta_touch_density"] > 0.0]
    touch_falling_days = [d for d in day_summaries if d["avg_delta_touch_density"] <= 0.0]
    print(f"\n  Signal H: Avg Δtouch density > 0 (price sticking more)")
    print(f"    Touch-rising days: {len(touch_rising_days)}, net pips: {sum(d['total_pips'] for d in touch_rising_days):+.1f}, avg: {sum(d['total_pips'] for d in touch_rising_days)/max(1,len(touch_rising_days)):+.1f}")
    print(f"    Touch-falling days:{len(touch_falling_days)}, net pips: {sum(d['total_pips'] for d in touch_falling_days):+.1f}, avg: {sum(d['total_pips'] for d in touch_falling_days)/max(1,len(touch_falling_days)):+.1f}")

    # Signal I: failed continuation shows up across a meaningful fraction of entries
    failed_cont_days = [d for d in day_summaries if d["failed_continuation_ratio"] >= 0.25]
    clean_cont_days = [d for d in day_summaries if d["failed_continuation_ratio"] < 0.25]
    print(f"\n  Signal I: Failed continuation ratio >= 0.25")
    print(f"    Failed-cont days: {len(failed_cont_days)}, net pips: {sum(d['total_pips'] for d in failed_cont_days):+.1f}, avg: {sum(d['total_pips'] for d in failed_cont_days)/max(1,len(failed_cont_days)):+.1f}")
    print(f"    Clean-cont days:  {len(clean_cont_days)}, net pips: {sum(d['total_pips'] for d in clean_cont_days):+.1f}, avg: {sum(d['total_pips'] for d in clean_cont_days)/max(1,len(clean_cont_days)):+.1f}")

    # Signal J: combined dynamic deterioration
    combo_dynamic_days = [
        d for d in day_summaries
        if d["avg_delta_efficiency_ratio"] < 0.0
        and d["avg_delta_touch_density"] > 0.0
        and d["failed_continuation_ratio"] >= 0.25
    ]
    non_combo_dynamic_days = [d for d in day_summaries if d not in combo_dynamic_days]
    print(f"\n  Signal J: ΔER<0 AND Δtouch>0 AND failed continuation >= 0.25")
    print(f"    Combo-dynamic days: {len(combo_dynamic_days)}, net pips: {sum(d['total_pips'] for d in combo_dynamic_days):+.1f}, avg: {sum(d['total_pips'] for d in combo_dynamic_days)/max(1,len(combo_dynamic_days)):+.1f}")
    print(f"    Non-combo days:     {len(non_combo_dynamic_days)}, net pips: {sum(d['total_pips'] for d in non_combo_dynamic_days):+.1f}, avg: {sum(d['total_pips'] for d in non_combo_dynamic_days)/max(1,len(non_combo_dynamic_days)):+.1f}")

    # ── 6. Stand-down impact simulation ──
    print(f"\n{'─'*70}")
    print(f"  STAND-DOWN IMPACT SIMULATION")
    print(f"{'─'*70}")
    print(f"  (What if we skipped all trades on days matching each signal?)")

    total_system_pips = sum(d["total_pips"] for d in day_summaries)
    total_system_days = len(day_summaries)
    total_system_trades = sum(d["total_count"] for d in day_summaries)

    print(f"\n  Baseline: {total_system_pips:+.1f} pips across {total_system_days} days, {total_system_trades} trades")

    signals = {
        "A: ambiguous-dominant": ambig_days,
        "B: low momentum (<2.0)": low_mom_days,
        "C: MR top-score dominant": mr_top_days,
        "D: all-ambiguous entries": all_ambig_days,
        "E: MR label dominant": mr_label_days,
        "F: 3+ regime labels": multi_regime_days,
        "G: avg ΔER < 0": er_decay_days,
        "H: avg Δtouch > 0": touch_rising_days,
        "I: failed continuation >= 0.25": failed_cont_days,
        "J: dynamic combo": combo_dynamic_days,
    }

    print(f"\n  {'Signal':<30s} {'Skip':>5s} {'Pips Cut':>10s} {'Remaining':>10s} {'Δ':>8s} {'Trades Cut':>11s}")
    print(f"  {'─'*80}")
    for sig_name, sig_days in signals.items():
        skip_pips = sum(d["total_pips"] for d in sig_days)
        remaining = total_system_pips - skip_pips
        delta = remaining - total_system_pips
        skip_trades = sum(d["total_count"] for d in sig_days)
        print(f"  {sig_name:<30s} {len(sig_days):>5d} {skip_pips:>+10.1f} {remaining:>+10.1f} {-skip_pips:>+8.1f} {skip_trades:>11d}")

    # ── 7. Strategy-specific impact on bad days ──
    print(f"\n{'─'*70}")
    print(f"  STRATEGY BREAKDOWN ON BAD DAYS (2+ strats losing)")
    print(f"{'─'*70}")

    for strat in ["v14", "london_v2", "v44_ny"]:
        strat_on_bad = []
        strat_on_good = []
        for d in day_summaries:
            if strat in d["strat_pips"]:
                if d["n_strats_losing"] >= 2:
                    strat_on_bad.append(d["strat_pips"][strat])
                else:
                    strat_on_good.append(d["strat_pips"][strat])
        bad_total = sum(strat_on_bad)
        good_total = sum(strat_on_good)
        print(f"\n  {strat}:")
        print(f"    On bad days: {len(strat_on_bad)} days, {bad_total:+.1f} pips, avg {bad_total/max(1,len(strat_on_bad)):+.1f}")
        print(f"    On good days: {len(strat_on_good)} days, {good_total:+.1f} pips, avg {good_total/max(1,len(strat_on_good)):+.1f}")

    return {
        "total_days": len(day_summaries),
        "total_pips": round(total_system_pips, 1),
        "green_days": len(green_days),
        "single_loss_days": len(single_loss_days),
        "bad_days": len(bad_days),
        "ugly_days": len(ugly_days),
        "bad_day_pips": round(total_bad_pips, 1),
        "ugly_day_pips": round(total_ugly_pips, 1),
        "green_profile": green_profile,
        "bad_profile": bad_profile,
        "ugly_profile": ugly_profile,
        "green_dynamic_profile": green_dynamic,
        "bad_dynamic_profile": bad_dynamic,
        "ugly_dynamic_profile": ugly_dynamic,
        "worst_days": [{
            "day": d["day"],
            "pips": d["total_pips"],
            "strats_losing": d["n_strats_losing"],
            "dominant_label": d["dominant_label"],
            "strat_pips": d["strat_pips"],
            "avg_delta_er": d["avg_delta_efficiency_ratio"],
            "avg_delta_touch": d["avg_delta_touch_density"],
            "failed_cont_ratio": d["failed_continuation_ratio"],
        } for d in worst_days],
        "signals": {
            name: {
                "days_cut": len(sig_days),
                "pips_cut": round(sum(d["total_pips"] for d in sig_days), 1),
                "trades_cut": sum(d["total_count"] for d in sig_days),
            }
            for name, sig_days in signals.items()
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

    out_path = ROOT / "research_out" / "diagnostic_no_trade_intelligence.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults written to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
