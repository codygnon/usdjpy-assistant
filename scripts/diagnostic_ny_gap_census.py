#!/usr/bin/env python3
"""
Gap Census: During NY session bars where the regime classifier scores mean_reversion
as the top regime, would V14 entry conditions be met?

This is a funnel analysis:
  1. How many M5 bars fall in NY session?
  2. Of those, how many have mean_reversion as top regime score?
  3. Of those, how many pass V14's BB regime gate (ranging)?
  4. Of those, how many pass V14's ADX gate (<35)?
  5. Of those, how many pass V14's ATR gate (<0.3)?
  6. Of those, how many have confluence signals >= 2?

Also measures:
  - Duration of mean-reversion gaps (consecutive M5 bars)
  - What % of NY session time is spent in mean-reversion-topped regime
  - Distribution of confluence scores when gates pass
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
from core.phase3_integrated_engine import (
    BB_PERIOD, BB_STD, BB_WIDTH_LOOKBACK, BB_WIDTH_RANGING_PCT,
    ADX_PERIOD, ADX_MAX, ATR_MAX,
    PSAR_FLIP_LOOKBACK,
    RSI_PERIOD, RSI_LONG_ENTRY, RSI_SHORT_ENTRY,
    ZONE_TOLERANCE_PIPS,
    NY_START_UTC, NY_END_UTC,
    compute_bb_width_regime, evaluate_v14_confluence,
    _compute_bb, _compute_adx, _compute_atr,
    _compute_rsi, compute_parabolic_sar,
)
from core.fib_pivots import compute_daily_fib_pivots
from scripts import validate_regime_classifier as regime_validation

PIP = 0.01

# ── Helpers ──────────────────────────────────────────────────────────

def _resample_m1_to_m5(m1: pd.DataFrame) -> pd.DataFrame:
    df = m1.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time")
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df = df.set_index("time")
    m5 = df.resample("5min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last"
    }).dropna().reset_index()
    return m5


def _resample_m1_to_m15(m1: pd.DataFrame) -> pd.DataFrame:
    df = m1.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time")
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df = df.set_index("time")
    m15 = df.resample("15min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last"
    }).dropna().reset_index()
    return m15


def _is_ny_session(ts: pd.Timestamp) -> bool:
    """Check if timestamp falls within NY session (13:00-16:00 UTC)."""
    h = ts.hour + ts.minute / 60.0
    return NY_START_UTC <= h < NY_END_UTC


def _load_classified_bars(input_csv: str) -> pd.DataFrame:
    m1 = regime_validation._load_m1(input_csv)
    featured = regime_validation.compute_features(m1)
    classified = regime_validation.classify_all_bars(featured, RegimeThresholds())
    return classified


def _get_regime_at_m5_bar(classified: pd.DataFrame, ts: pd.Timestamp,
                          time_idx: pd.DatetimeIndex) -> dict:
    idx = time_idx.get_indexer([ts], method="ffill")[0]
    if idx < 0:
        idx = time_idx.get_indexer([ts], method="bfill")[0]
    if idx < 0:
        return {"top_regime": "unknown", "scores": {}, "margin": 0.0}
    row = classified.iloc[idx]
    scores = {
        "momentum": float(row.get("score_momentum", 0.0)),
        "mean_reversion": float(row.get("score_mean_reversion", 0.0)),
        "breakout": float(row.get("score_breakout", 0.0)),
        "post_breakout_trend": float(row.get("score_post_breakout_trend", 0.0)),
    }
    top_regime = max(scores, key=scores.get)
    label = str(row.get("regime_hysteresis", "ambiguous"))
    return {"top_regime": top_regime, "label": label, "scores": scores,
            "margin": float(row.get("score_margin", 0.0))}


def run_census(dataset_label: str, input_csv: str) -> dict:
    print(f"\n{'='*70}")
    print(f"  GAP CENSUS: {dataset_label}")
    print(f"{'='*70}")

    # Load and resample data
    print("Loading M1 data...")
    m1_raw = regime_validation._load_m1(input_csv)
    m1 = m1_raw.copy()
    m1["time"] = pd.to_datetime(m1["time"], utc=True, errors="coerce")
    m1 = m1.dropna(subset=["time"]).sort_values("time")

    print("Resampling to M5/M15...")
    m5 = _resample_m1_to_m5(m1)
    m15 = _resample_m1_to_m15(m1)

    print("Classifying bars...")
    classified = _load_classified_bars(input_csv)
    classified_time_idx = pd.DatetimeIndex(
        pd.to_datetime(classified["time"], utc=True, errors="coerce")
    )

    # Compute daily pivots (NY close boundary = 22 UTC)
    print("Computing daily pivots...")
    m1_for_pivots = m1.copy()
    for col in ["open", "high", "low", "close"]:
        m1_for_pivots[col] = m1_for_pivots[col].astype(float)

    # Build pivot cache by NY-close trading day
    pivot_cache = {}
    for _, row in m1_for_pivots.iterrows():
        ts = row["time"]
        ny_day = (ts - pd.Timedelta(hours=22)).date()
        if ny_day not in pivot_cache:
            # Get previous day's data for pivots
            prev_day = ny_day
            day_start = pd.Timestamp(prev_day, tz="UTC") + pd.Timedelta(hours=22)
            day_end = day_start + pd.Timedelta(hours=24)
            day_data = m1_for_pivots[
                (m1_for_pivots["time"] >= day_start - pd.Timedelta(hours=24)) &
                (m1_for_pivots["time"] < day_start)
            ]
            if not day_data.empty:
                H = day_data["high"].max()
                L = day_data["low"].min()
                C = day_data["close"].iloc[-1]
                P = (H + L + C) / 3.0
                pivot_cache[ny_day] = {
                    "P": P, "R1": P + 0.382*(H-L), "R2": P + 0.618*(H-L),
                    "R3": P + 1.0*(H-L), "S1": P - 0.382*(H-L),
                    "S2": P - 0.618*(H-L), "S3": P - 1.0*(H-L),
                }

    # Identify NY session M5 bars
    ny_bars = m5[m5["time"].apply(_is_ny_session)].copy()
    print(f"Total NY session M5 bars: {len(ny_bars)}")

    # Funnel analysis
    funnel = {
        "total_ny_m5_bars": len(ny_bars),
        "mr_topped": 0,
        "mr_topped_bb_ranging": 0,
        "mr_topped_bb_ranging_adx_ok": 0,
        "mr_topped_bb_ranging_adx_ok_atr_ok": 0,
        "mr_topped_all_gates_confluence_ge2": 0,
    }

    # Track gap durations
    gap_durations = []  # consecutive M5 bars in MR-topped regime
    current_gap = 0

    # Track confluence details
    confluence_details = []
    mr_bar_details = []

    # Pre-compute M15 ADX and ATR at each M5 bar time
    # We need at least ADX_PERIOD+1 M15 bars
    m15_close = m15["close"].astype(float)
    m15_high = m15["high"].astype(float)
    m15_low = m15["low"].astype(float)

    for i, (_, bar) in enumerate(ny_bars.iterrows()):
        ts = bar["time"]

        # Get regime
        regime_info = _get_regime_at_m5_bar(classified, ts, classified_time_idx)
        top = regime_info["top_regime"]

        if top != "mean_reversion":
            if current_gap > 0:
                gap_durations.append(current_gap)
                current_gap = 0
            continue

        funnel["mr_topped"] += 1
        current_gap += 1

        # Get M5 data up to this bar for BB regime and BB bands
        m5_up_to = m5[m5["time"] <= ts]
        if len(m5_up_to) < BB_PERIOD + BB_WIDTH_LOOKBACK:
            continue

        # V14 gate 1: BB regime (ranging?)
        bb_regime = compute_bb_width_regime(
            m5_up_to, mode="percentile",
            lookback=BB_WIDTH_LOOKBACK, ranging_pct=BB_WIDTH_RANGING_PCT,
        )
        if bb_regime != "ranging":
            mr_bar_details.append({
                "time": ts.isoformat(), "blocked_at": "bb_regime",
                "bb_regime": bb_regime, "scores": regime_info["scores"],
            })
            continue
        funnel["mr_topped_bb_ranging"] += 1

        # V14 gate 2: ADX < 35 (M15)
        m15_up_to = m15[m15["time"] <= ts]
        if len(m15_up_to) < ADX_PERIOD + 2:
            continue
        adx_val = float(_compute_adx(m15_up_to, period=ADX_PERIOD))
        if adx_val >= ADX_MAX:
            mr_bar_details.append({
                "time": ts.isoformat(), "blocked_at": "adx",
                "adx": round(adx_val, 1), "scores": regime_info["scores"],
            })
            continue
        funnel["mr_topped_bb_ranging_adx_ok"] += 1

        # V14 gate 3: ATR < 0.3 (M15)
        atr_series = _compute_atr(m15_up_to, period=14)
        atr_val = float(atr_series.iloc[-1]) if not atr_series.empty and pd.notna(atr_series.iloc[-1]) else 999.0
        if atr_val >= ATR_MAX:
            mr_bar_details.append({
                "time": ts.isoformat(), "blocked_at": "atr",
                "atr": round(atr_val, 4), "scores": regime_info["scores"],
            })
            continue
        funnel["mr_topped_bb_ranging_adx_ok_atr_ok"] += 1

        # V14 gate 4: Confluence >= 2
        close_price = float(m5_up_to["close"].iloc[-1])
        high_price = float(m5_up_to["high"].iloc[-1])
        low_price = float(m5_up_to["low"].iloc[-1])

        # BB bands
        m5_close_series = m5_up_to["close"].astype(float)
        sma = m5_close_series.rolling(BB_PERIOD).mean()
        std = m5_close_series.rolling(BB_PERIOD, min_periods=BB_PERIOD).std(ddof=0)
        bb_upper = float((sma + BB_STD * std).iloc[-1])
        bb_lower = float((sma - BB_STD * std).iloc[-1])

        # RSI
        rsi_series = _compute_rsi(m5_up_to["close"].astype(float), period=RSI_PERIOD)
        rsi_val = float(rsi_series.iloc[-1]) if not rsi_series.empty and pd.notna(rsi_series.iloc[-1]) else 50.0

        # SAR (needs M1 data)
        m1_up_to = m1[m1["time"] <= ts].tail(200)
        sar_bull = False
        sar_bear = False
        if len(m1_up_to) > 20:
            sar_series = compute_parabolic_sar(m1_up_to)
            if len(sar_series) >= PSAR_FLIP_LOOKBACK:
                recent_sar = sar_series.values[-PSAR_FLIP_LOOKBACK:]
                recent_close = m1_up_to["close"].astype(float).values[-PSAR_FLIP_LOOKBACK:]
                for j in range(1, len(recent_sar)):
                    if not (np.isnan(recent_sar[j-1]) or np.isnan(recent_sar[j])):
                        if recent_sar[j-1] > recent_close[j-1] and recent_sar[j] < recent_close[j]:
                            sar_bull = True
                        if recent_sar[j-1] < recent_close[j-1] and recent_sar[j] > recent_close[j]:
                            sar_bear = True

        # Pivots
        ny_day = (ts - pd.Timedelta(hours=22)).date()
        pivots = pivot_cache.get(ny_day)
        if pivots is None:
            mr_bar_details.append({
                "time": ts.isoformat(), "blocked_at": "no_pivots",
                "scores": regime_info["scores"],
            })
            continue

        # Evaluate confluence for both sides
        best_score = 0
        best_combo = ""
        best_side = None
        for side in ("buy", "sell"):
            score, combo = evaluate_v14_confluence(
                side, close_price, high_price, low_price,
                pivots, bb_upper, bb_lower,
                sar_bull, sar_bear, rsi_val, PIP,
                zone_tolerance_pips=ZONE_TOLERANCE_PIPS,
                rsi_long_entry=RSI_LONG_ENTRY,
                rsi_short_entry=RSI_SHORT_ENTRY,
            )
            if score > best_score:
                best_score = score
                best_combo = combo
                best_side = side

        confluence_details.append({
            "time": ts.isoformat(),
            "score": best_score,
            "combo": best_combo,
            "side": best_side,
            "close": round(close_price, 3),
            "rsi": round(rsi_val, 1),
            "adx": round(adx_val, 1),
            "atr": round(atr_val, 4),
            "bb_upper": round(bb_upper, 3),
            "bb_lower": round(bb_lower, 3),
            "regime_scores": regime_info["scores"],
        })

        if best_score >= 2:
            funnel["mr_topped_all_gates_confluence_ge2"] += 1

    # Final gap
    if current_gap > 0:
        gap_durations.append(current_gap)

    # ── Print results ──────────────────────────────────────────────

    print(f"\n  FUNNEL ANALYSIS:")
    print(f"  {'Total NY M5 bars:':<50s} {funnel['total_ny_m5_bars']:>6d}")
    print(f"  {'Mean-reversion topped:':<50s} {funnel['mr_topped']:>6d}  ({100*funnel['mr_topped']/max(1,funnel['total_ny_m5_bars']):.1f}%)")
    print(f"  {'  + BB regime ranging:':<50s} {funnel['mr_topped_bb_ranging']:>6d}  ({100*funnel['mr_topped_bb_ranging']/max(1,funnel['mr_topped']):.1f}% of MR)")
    print(f"  {'  + ADX < 35:':<50s} {funnel['mr_topped_bb_ranging_adx_ok']:>6d}  ({100*funnel['mr_topped_bb_ranging_adx_ok']/max(1,funnel['mr_topped']):.1f}% of MR)")
    print(f"  {'  + ATR < 0.3:':<50s} {funnel['mr_topped_bb_ranging_adx_ok_atr_ok']:>6d}  ({100*funnel['mr_topped_bb_ranging_adx_ok_atr_ok']/max(1,funnel['mr_topped']):.1f}% of MR)")
    print(f"  {'  + Confluence >= 2:':<50s} {funnel['mr_topped_all_gates_confluence_ge2']:>6d}  ({100*funnel['mr_topped_all_gates_confluence_ge2']/max(1,funnel['mr_topped']):.1f}% of MR)")

    # Gap duration stats
    if gap_durations:
        print(f"\n  MR GAP DURATION (consecutive M5 bars):")
        print(f"    Count: {len(gap_durations)} gaps")
        print(f"    Mean: {np.mean(gap_durations):.1f} bars ({np.mean(gap_durations)*5:.0f} min)")
        print(f"    Median: {np.median(gap_durations):.0f} bars ({np.median(gap_durations)*5:.0f} min)")
        print(f"    Max: {max(gap_durations)} bars ({max(gap_durations)*5} min)")
        print(f"    Min: {min(gap_durations)} bars ({min(gap_durations)*5} min)")
        # Distribution
        buckets = {"1 bar": 0, "2-3 bars": 0, "4-6 bars": 0, "7-12 bars": 0, "13+ bars": 0}
        for g in gap_durations:
            if g == 1: buckets["1 bar"] += 1
            elif g <= 3: buckets["2-3 bars"] += 1
            elif g <= 6: buckets["4-6 bars"] += 1
            elif g <= 12: buckets["7-12 bars"] += 1
            else: buckets["13+ bars"] += 1
        print(f"    Distribution: {dict(buckets)}")

    # Confluence score distribution
    if confluence_details:
        score_dist = defaultdict(int)
        for c in confluence_details:
            score_dist[c["score"]] += 1
        print(f"\n  CONFLUENCE SCORE DISTRIBUTION (bars passing all pre-gates):")
        for score in sorted(score_dist):
            pct = 100 * score_dist[score] / len(confluence_details)
            print(f"    Score {score}: {score_dist[score]} bars ({pct:.1f}%)")

        # Side distribution for confluence >= 2
        qualifying = [c for c in confluence_details if c["score"] >= 2]
        if qualifying:
            side_dist = defaultdict(int)
            combo_dist = defaultdict(int)
            for c in qualifying:
                side_dist[c["side"]] += 1
                combo_dist[c["combo"]] += 1
            print(f"\n  QUALIFYING SIGNALS (confluence >= 2): {len(qualifying)}")
            print(f"    By side: {dict(side_dist)}")
            print(f"    By combo: {dict(sorted(combo_dist.items(), key=lambda x: x[1], reverse=True))}")
            print(f"    Avg RSI: {np.mean([c['rsi'] for c in qualifying]):.1f}")
            print(f"    Avg ADX: {np.mean([c['adx'] for c in qualifying]):.1f}")

    # Blocked-at distribution
    if mr_bar_details:
        blocked_at_dist = defaultdict(int)
        for d in mr_bar_details:
            blocked_at_dist[d["blocked_at"]] += 1
        print(f"\n  MR BARS BLOCKED BEFORE CONFLUENCE:")
        for reason, count in sorted(blocked_at_dist.items(), key=lambda x: x[1], reverse=True):
            print(f"    {reason}: {count}")

    results = {
        "funnel": funnel,
        "gap_durations": {
            "count": len(gap_durations),
            "mean_bars": round(np.mean(gap_durations), 1) if gap_durations else 0,
            "median_bars": round(float(np.median(gap_durations)), 0) if gap_durations else 0,
            "max_bars": max(gap_durations) if gap_durations else 0,
        },
        "confluence_score_distribution": dict(sorted(
            {str(k): v for k, v in score_dist.items()}.items()
        )) if confluence_details else {},
        "qualifying_signals_count": len([c for c in confluence_details if c["score"] >= 2]),
        "qualifying_signals_sample": [c for c in confluence_details if c["score"] >= 2][:30],
    }
    return results


def main() -> int:
    datasets = [
        ("500k", str(ROOT / "research_out" / "USDJPY_M1_OANDA_500k.csv")),
        ("1000k", str(ROOT / "research_out" / "USDJPY_M1_OANDA_1000k.csv")),
    ]

    all_results = {}
    for label, csv_path in datasets:
        if not Path(csv_path).exists():
            print(f"Skipping {label}: not found")
            continue
        all_results[label] = run_census(label, csv_path)

    out_path = ROOT / "research_out" / "diagnostic_ny_gap_census.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
