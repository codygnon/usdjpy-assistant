#!/usr/bin/env python3
"""
Session Boundary Census: What happens at the London→NY transition?

For each trading day, captures:
1. London session summary:
   - Regime at London close (last M5 bar before session end)
   - Did London V2 have a trade? What happened to it? (exit reason, pips)
   - Regime trajectory during last 30 min of London (regime stability)

2. Gap period (London end → NY start, ~12:00-13:00 UTC):
   - Regime during the gap
   - Any regime transition during the gap?

3. NY session first 30 minutes:
   - Regime at NY open
   - Did V44 fire in first 30 min? In what regime? Was it profitable?
   - Regime transition from London close → NY open

4. Cross-boundary patterns:
   - When London ends in breakout/momentum, what does NY see?
   - When London ends in mean_reversion, what does NY see?
   - Days where V44 enters in first 30 min of NY: regime at London close vs NY open
   - Days where London V2 was hard-closed: what was happening on the chart?

Output: regime transition matrix + trade outcome cross-reference.
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
from core.phase3_integrated_engine import uk_london_open_utc
from scripts import validate_regime_classifier as regime_validation
from scripts.backtest_v44_conservative_router import _lookup_regime


def _load_classified_bars(input_csv: str) -> pd.DataFrame:
    m1 = regime_validation._load_m1(input_csv)
    featured = regime_validation.compute_features(m1)
    classified = regime_validation.classify_all_bars(featured, RegimeThresholds())
    return classified


def _get_regime_info(classified: pd.DataFrame, ts: pd.Timestamp,
                     time_idx: pd.DatetimeIndex) -> dict:
    idx = time_idx.get_indexer([ts], method="ffill")[0]
    if idx < 0:
        return {"label": "unknown", "top_regime": "unknown", "scores": {}, "margin": 0.0}
    row = classified.iloc[idx]
    scores = {
        "momentum": float(row.get("score_momentum", 0.0)),
        "mean_reversion": float(row.get("score_mean_reversion", 0.0)),
        "breakout": float(row.get("score_breakout", 0.0)),
        "post_breakout_trend": float(row.get("score_post_breakout_trend", 0.0)),
    }
    top_regime = max(scores, key=scores.get) if scores else "unknown"
    label = str(row.get("regime_hysteresis", "ambiguous"))
    return {"label": label, "top_regime": top_regime, "scores": scores,
            "margin": float(row.get("score_margin", 0.0))}


def run_census(dataset_label: str, input_csv: str, baseline_json: str) -> dict:
    print(f"\n{'='*70}")
    print(f"  SESSION BOUNDARY CENSUS: {dataset_label}")
    print(f"{'='*70}")

    # Load baseline trades
    with open(baseline_json, encoding="utf-8") as f:
        baseline = json.load(f)

    v2_trades = [t for t in baseline["closed_trades"] if t["strategy"] == "london_v2"]
    v44_trades = [t for t in baseline["closed_trades"] if t["strategy"] == "v44_ny"]

    # Index trades by date for quick lookup
    v2_by_date: dict[str, list] = defaultdict(list)
    for t in v2_trades:
        day = pd.Timestamp(t["entry_time"]).strftime("%Y-%m-%d")
        v2_by_date[day].append(t)

    v44_by_date: dict[str, list] = defaultdict(list)
    for t in v44_trades:
        day = pd.Timestamp(t["entry_time"]).strftime("%Y-%m-%d")
        v44_by_date[day].append(t)

    print("Classifying bars...")
    classified = _load_classified_bars(input_csv)
    classified_time_idx = pd.DatetimeIndex(
        pd.to_datetime(classified["time"], utc=True, errors="coerce")
    )

    # Get date range from classified data
    all_dates = pd.to_datetime(classified["time"], utc=True).dt.date.unique()
    all_dates = sorted(all_dates)

    # Track transition patterns
    transitions = []  # list of per-day boundary records
    regime_transition_matrix = defaultdict(lambda: defaultdict(int))  # london_close_top -> ny_open_top

    # V2 hard-close analysis
    v2_hardclose_records = []
    # V44 early-entry analysis (first 30 min of NY)
    v44_early_records = []

    for day in all_dates:
        ts_day = pd.Timestamp(day, tz="UTC")
        weekday = ts_day.weekday()
        # London V2 only active Tue/Wed (weekday 1, 2)
        # V44 active Mon-Fri
        # We care about days where at least one strategy was active
        if weekday > 4:  # skip weekends
            continue

        london_open_hour = uk_london_open_utc(ts_day)
        london_end_hour = london_open_hour + 4  # 4-hour London session
        ny_start_hour = 13  # fixed
        ny_end_hour = 16

        # Key timestamps
        london_close_ts = ts_day.replace(hour=london_end_hour, minute=0)
        london_last_30_start = london_close_ts - pd.Timedelta(minutes=30)
        ny_open_ts = ts_day.replace(hour=ny_start_hour, minute=0)
        ny_first_30_end = ny_open_ts + pd.Timedelta(minutes=30)

        # Get regime at key points
        regime_london_close = _get_regime_info(classified, london_close_ts, classified_time_idx)
        regime_london_last_30 = _get_regime_info(classified, london_last_30_start, classified_time_idx)
        regime_ny_open = _get_regime_info(classified, ny_open_ts, classified_time_idx)
        regime_ny_30 = _get_regime_info(classified, ny_first_30_end, classified_time_idx)

        # Gap regime (midpoint of London end → NY start)
        gap_mid_ts = london_close_ts + pd.Timedelta(minutes=30)
        regime_gap_mid = _get_regime_info(classified, gap_mid_ts, classified_time_idx)

        # Transition matrix
        lc_top = regime_london_close["top_regime"]
        no_top = regime_ny_open["top_regime"]
        regime_transition_matrix[lc_top][no_top] += 1

        day_str = day.isoformat()

        # London V2 trades on this day
        v2_day_trades = v2_by_date.get(day_str, [])
        v2_active = len(v2_day_trades) > 0
        v2_hardclosed = any(t["exit_reason"] == "HARD_CLOSE" for t in v2_day_trades)
        v2_net_pips = sum(t["pips"] for t in v2_day_trades)

        # V44 trades in first 30 min of NY
        v44_day_trades = v44_by_date.get(day_str, [])
        v44_early = [t for t in v44_day_trades
                     if pd.Timestamp(t["entry_time"]) < ny_first_30_end]
        v44_early_net = sum(t["pips"] for t in v44_early)

        record = {
            "date": day_str,
            "weekday": ts_day.strftime("%A"),
            "london_close_top": lc_top,
            "london_close_label": regime_london_close["label"],
            "london_close_scores": regime_london_close["scores"],
            "gap_mid_top": regime_gap_mid["top_regime"],
            "ny_open_top": no_top,
            "ny_open_label": regime_ny_open["label"],
            "ny_30_top": regime_ny_30["top_regime"],
            "regime_changed_london_to_ny": lc_top != no_top,
            "v2_active": v2_active,
            "v2_trades": len(v2_day_trades),
            "v2_hardclosed": v2_hardclosed,
            "v2_net_pips": round(v2_net_pips, 1),
            "v44_early_trades": len(v44_early),
            "v44_early_net_pips": round(v44_early_net, 1),
        }
        transitions.append(record)

        # V2 hard-close detail
        for t in v2_day_trades:
            if t["exit_reason"] == "HARD_CLOSE":
                v2_hardclose_records.append({
                    "date": day_str,
                    "side": t["side"],
                    "pips": round(t["pips"], 1),
                    "usd": round(t["usd"], 2),
                    "london_close_top": lc_top,
                    "ny_open_top": no_top,
                    "regime_at_exit": regime_london_close["label"],
                })

        # V44 early entry detail
        for t in v44_early:
            v44_entry_regime = _get_regime_info(
                classified, pd.Timestamp(t["entry_time"]), classified_time_idx
            )
            v44_early_records.append({
                "date": day_str,
                "entry_time": t["entry_time"],
                "side": t["side"],
                "pips": round(t["pips"], 1),
                "usd": round(t["usd"], 2),
                "exit_reason": t["exit_reason"],
                "london_close_top": lc_top,
                "ny_open_top": no_top,
                "entry_regime_top": v44_entry_regime["top_regime"],
                "entry_regime_label": v44_entry_regime["label"],
            })

    # ── Print results ──────────────────────────────────────────────

    total_days = len(transitions)
    regime_changed_days = sum(1 for t in transitions if t["regime_changed_london_to_ny"])

    print(f"\n  Total trading days analyzed: {total_days}")
    print(f"  Days where regime top-score changed London→NY: {regime_changed_days} ({100*regime_changed_days/max(1,total_days):.1f}%)")

    # Transition matrix
    print(f"\n  REGIME TRANSITION MATRIX (London close → NY open)")
    print(f"  {'London close →':<25s}", end="")
    all_regimes = ["momentum", "mean_reversion", "breakout", "post_breakout_trend"]
    for r in all_regimes:
        print(f" {r[:8]:>10s}", end="")
    print()
    print(f"  {'─'*70}")
    for from_r in all_regimes:
        row_total = sum(regime_transition_matrix[from_r].values())
        print(f"  {from_r:<25s}", end="")
        for to_r in all_regimes:
            count = regime_transition_matrix[from_r][to_r]
            pct = 100 * count / max(1, row_total)
            print(f" {count:>4d}({pct:>3.0f}%)", end="")
        print(f"  [n={row_total}]")

    # V2 hard-close analysis
    print(f"\n  LONDON V2 HARD-CLOSE ANALYSIS")
    print(f"  Total hard-closed trades: {len(v2_hardclose_records)}")
    if v2_hardclose_records:
        hc_winners = [r for r in v2_hardclose_records if r["pips"] > 0]
        hc_losers = [r for r in v2_hardclose_records if r["pips"] <= 0]
        hc_net = sum(r["pips"] for r in v2_hardclose_records)
        print(f"  Winners: {len(hc_winners)}  Losers: {len(hc_losers)}  Net pips: {hc_net:+.1f}")

        # By regime at London close
        hc_by_regime = defaultdict(list)
        for r in v2_hardclose_records:
            hc_by_regime[r["london_close_top"]].append(r)
        print(f"\n  Hard-closed by London-close regime:")
        for regime in sorted(hc_by_regime.keys()):
            trades = hc_by_regime[regime]
            w = sum(1 for t in trades if t["pips"] > 0)
            l = sum(1 for t in trades if t["pips"] <= 0)
            net = sum(t["pips"] for t in trades)
            avg = net / len(trades)
            print(f"    {regime:<25s} {len(trades):>3d} trades (W:{w} L:{l}) net:{net:+.1f}p avg:{avg:+.1f}p")

        # Winners that were hard-closed — these are the "cut short" trades
        if hc_winners:
            print(f"\n  Hard-closed WINNERS (potentially cut short):")
            print(f"    Count: {len(hc_winners)}")
            hc_win_pips = [r["pips"] for r in hc_winners]
            print(f"    Pips: min={min(hc_win_pips):.1f} median={sorted(hc_win_pips)[len(hc_win_pips)//2]:.1f} max={max(hc_win_pips):.1f} total={sum(hc_win_pips):.1f}")
            hc_win_by_regime = defaultdict(list)
            for r in hc_winners:
                hc_win_by_regime[r["london_close_top"]].append(r)
            for regime in sorted(hc_win_by_regime.keys()):
                trades = hc_win_by_regime[regime]
                net = sum(t["pips"] for t in trades)
                print(f"      {regime}: {len(trades)} trades, {net:+.1f}p")

    # V44 early entry analysis
    print(f"\n  V44 EARLY ENTRY ANALYSIS (first 30 min of NY)")
    print(f"  Total early V44 trades: {len(v44_early_records)}")
    if v44_early_records:
        early_w = sum(1 for r in v44_early_records if r["pips"] > 0)
        early_l = sum(1 for r in v44_early_records if r["pips"] <= 0)
        early_net = sum(r["pips"] for r in v44_early_records)
        print(f"  Winners: {early_w}  Losers: {early_l}  Net pips: {early_net:+.1f}")

        # By regime at entry
        early_by_regime = defaultdict(list)
        for r in v44_early_records:
            early_by_regime[r["entry_regime_top"]].append(r)
        print(f"\n  Early V44 by entry regime (top score):")
        for regime in sorted(early_by_regime.keys()):
            trades = early_by_regime[regime]
            w = sum(1 for t in trades if t["pips"] > 0)
            l = sum(1 for t in trades if t["pips"] <= 0)
            net = sum(t["pips"] for t in trades)
            print(f"    {regime:<25s} {len(trades):>3d} trades (W:{w} L:{l}) net:{net:+.1f}p")

        # By London close regime → V44 early outcome
        print(f"\n  Early V44 outcome by LONDON CLOSE regime:")
        early_by_lc = defaultdict(list)
        for r in v44_early_records:
            early_by_lc[r["london_close_top"]].append(r)
        for regime in sorted(early_by_lc.keys()):
            trades = early_by_lc[regime]
            w = sum(1 for t in trades if t["pips"] > 0)
            l = sum(1 for t in trades if t["pips"] <= 0)
            net = sum(t["pips"] for t in trades)
            print(f"    {regime:<25s} {len(trades):>3d} trades (W:{w} L:{l}) net:{net:+.1f}p")

    # Continuity analysis: when London ends in breakout, does NY continue or reverse?
    print(f"\n  REGIME CONTINUITY ANALYSIS")
    for from_regime in all_regimes:
        days_from = [t for t in transitions if t["london_close_top"] == from_regime]
        if not days_from:
            continue
        same = sum(1 for t in days_from if t["ny_open_top"] == from_regime)
        pct_same = 100 * same / len(days_from)
        # V44 outcomes on those days
        v44_on_days = [t for t in days_from if t["v44_early_trades"] > 0]
        v44_net = sum(t["v44_early_net_pips"] for t in v44_on_days)
        print(f"  London close={from_regime:<20s}: {len(days_from):>3d} days, "
              f"NY same regime: {same}/{len(days_from)} ({pct_same:.0f}%), "
              f"V44 early: {len(v44_on_days)} days net:{v44_net:+.1f}p")

    results = {
        "total_days": total_days,
        "regime_changed_days": regime_changed_days,
        "regime_changed_pct": round(100 * regime_changed_days / max(1, total_days), 1),
        "transition_matrix": {
            from_r: dict(regime_transition_matrix[from_r])
            for from_r in regime_transition_matrix
        },
        "v2_hardclose_count": len(v2_hardclose_records),
        "v44_early_count": len(v44_early_records),
        "v2_hardclose_records": v2_hardclose_records[:50],
        "v44_early_records": v44_early_records[:50],
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
            print(f"Skipping {label}: not found")
            continue
        if not Path(baseline_path).exists():
            print(f"Skipping {label}: not found")
            continue
        all_results[label] = run_census(label, csv_path, baseline_path)

    out_path = ROOT / "research_out" / "diagnostic_session_boundary_census.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
