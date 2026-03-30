#!/usr/bin/env python3
"""
Validate the score-based regime classifier against backtested trade data.

For each dataset:
  1. Compute regime indicators on every bar
  2. Run the classifier (both stateless and with hysteresis)
  3. Run standalone backtests to get trade entries
  4. Map each trade to its bar-level regime label
  5. Report:
     - Overall regime distribution (% of bars in each regime)
     - Per-strategy win rate by regime
     - Score distributions at wins vs losses
     - "Right regime" rate (V14 at mean_reversion, V44 at momentum, etc.)

Usage:
    python scripts/validate_regime_classifier.py \
        --input research_out/USDJPY_M1_OANDA_250k.csv \
        --output research_out/regime_validation_250k.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.regime_classifier import (
    RegimeClassifier,
    RegimeThresholds,
    classify_bar,
)
from scripts.regime_threshold_analysis import (
    _load_m1,
    _resample,
    _rolling_adx,
    _ema,
    _run_v14,
    _run_london_v2,
    _run_v44,
    BB_PERIOD,
    BB_STD,
    BB_WIDTH_LOOKBACK,
    M5_EMA_FAST,
    SLOPE_BARS,
    H1_EMA_FAST,
    H1_EMA_SLOW,
    PIP_SIZE,
)


STRATEGY_NATURAL_REGIME = {
    "v14": "mean_reversion",
    "london_v2": "breakout",
    "v44_ny": "momentum",
}


DIR_EFF_BARS = 12        # M5 bars for directional efficiency lookback
COMP_PCTILE_THRESH = 0.40  # bb_width_pctile threshold for compression
BB_EXP_RATE_BARS = 6     # M5 bars for BB width expansion rate


def compute_features(m1: pd.DataFrame) -> pd.DataFrame:
    """Compute regime indicator columns on M1 bars."""
    out = m1.copy()
    m5 = _resample(out, "5min")
    m15 = _resample(out, "15min")
    h1 = _resample(out, "1h")

    sma = m5["close"].rolling(BB_PERIOD, min_periods=BB_PERIOD).mean()
    std = m5["close"].rolling(BB_PERIOD, min_periods=BB_PERIOD).std(ddof=0)
    m5["bb_upper"] = sma + BB_STD * std
    m5["bb_lower"] = sma - BB_STD * std
    m5["bb_mid"] = sma
    m5["bb_width"] = (m5["bb_upper"] - m5["bb_lower"]) / m5["bb_mid"].replace(0, np.nan)
    m5["bb_width_pctile"] = m5["bb_width"].rolling(BB_WIDTH_LOOKBACK, min_periods=BB_WIDTH_LOOKBACK).rank(pct=True)
    cutoff = m5["bb_width"].rolling(BB_WIDTH_LOOKBACK, min_periods=BB_WIDTH_LOOKBACK).quantile(0.80)
    m5["bb_regime"] = np.where(m5["bb_width"] < cutoff, "ranging", "trending")

    m5_ema_fast = _ema(m5["close"], M5_EMA_FAST)
    m5["m5_slope"] = (m5_ema_fast - m5_ema_fast.shift(SLOPE_BARS)) / PIP_SIZE / max(1, SLOPE_BARS)
    m5["m5_slope_abs"] = m5["m5_slope"].abs()

    # ── New feature A1: directional efficiency ──
    close = m5["close"]
    net_disp = (close - close.shift(DIR_EFF_BARS)).abs()
    total_path = close.diff().abs().rolling(DIR_EFF_BARS, min_periods=DIR_EFF_BARS).sum()
    m5["dir_efficiency"] = net_disp / total_path.replace(0, np.nan)

    # ── New feature B2: compression duration ──
    is_compressed = (m5["bb_width_pctile"] < COMP_PCTILE_THRESH).astype(int)
    groups = (is_compressed != is_compressed.shift()).cumsum()
    m5["compression_duration"] = is_compressed.groupby(groups).cumsum()

    # ── New feature C1: BB width expansion rate ──
    m5["bb_width_exp_rate"] = m5["bb_width"] / m5["bb_width"].shift(BB_EXP_RATE_BARS).replace(0, np.nan)

    m15["adx"] = _rolling_adx(m15, 14)

    h1_fast = _ema(h1["close"], H1_EMA_FAST)
    h1_slow = _ema(h1["close"], H1_EMA_SLOW)
    h1["h1_trend_dir"] = np.where(h1_fast > h1_slow, "up", "down")

    m5_cols = [
        "time", "bb_regime", "bb_width_pctile", "m5_slope", "m5_slope_abs",
        "dir_efficiency", "compression_duration", "bb_width_exp_rate",
    ]
    m15_cols = ["time", "adx"]
    h1_cols = ["time", "h1_trend_dir"]

    out = pd.merge_asof(out.sort_values("time"), m5[m5_cols].sort_values("time"),
                        on="time", direction="backward")
    out = pd.merge_asof(out.sort_values("time"), m15[m15_cols].sort_values("time"),
                        on="time", direction="backward")
    out = pd.merge_asof(out.sort_values("time"), h1[h1_cols].sort_values("time"),
                        on="time", direction="backward")

    # Side-aware H1 alignment: H1 direction agrees with M5 slope direction
    out["h1_aligned"] = (
        ((out["m5_slope"] > 0) & (out["h1_trend_dir"] == "up"))
        | ((out["m5_slope"] < 0) & (out["h1_trend_dir"] == "down"))
    ).astype(int)

    return out


def classify_all_bars(df: pd.DataFrame, th: RegimeThresholds) -> pd.DataFrame:
    """Apply stateless and stateful classifier to every bar."""
    labels_stateless = []
    scores_momentum = []
    scores_mr = []
    scores_bo = []
    scores_pbt = []
    margins = []

    classifier = RegimeClassifier(th)
    labels_hysteresis = []
    held_flags = []

    for _, row in df.iterrows():
        adx = float(row.get("adx", 0)) if pd.notna(row.get("adx")) else 0.0
        slope = float(row.get("m5_slope_abs", 0)) if pd.notna(row.get("m5_slope_abs")) else 0.0
        aligned = bool(row.get("h1_aligned", 0)) if pd.notna(row.get("h1_aligned")) else False
        pctile = float(row.get("bb_width_pctile", 0.5)) if pd.notna(row.get("bb_width_pctile")) else 0.5
        regime = str(row.get("bb_regime", "ranging")) if pd.notna(row.get("bb_regime")) else "ranging"
        exp_rate = float(row.get("bb_width_exp_rate", 1.0)) if pd.notna(row.get("bb_width_exp_rate")) else None

        lbl, sc, mg = classify_bar(adx, slope, aligned, pctile, regime, th, bb_width_exp_rate=exp_rate)
        labels_stateless.append(lbl)
        scores_momentum.append(sc.momentum)
        scores_mr.append(sc.mean_reversion)
        scores_bo.append(sc.breakout)
        scores_pbt.append(sc.post_breakout_trend)
        margins.append(mg)

        res = classifier.update(adx, slope, aligned, pctile, regime, bb_width_exp_rate=exp_rate)
        labels_hysteresis.append(res.label)
        held_flags.append(res.held_by_dwell)

    df = df.copy()
    df["regime_stateless"] = labels_stateless
    df["regime_hysteresis"] = labels_hysteresis
    df["regime_held"] = held_flags
    df["score_momentum"] = scores_momentum
    df["score_mean_reversion"] = scores_mr
    df["score_breakout"] = scores_bo
    df["score_post_breakout_trend"] = scores_pbt
    df["score_margin"] = margins
    return df


def label_trades(df: pd.DataFrame, trades: list[dict]) -> list[dict]:
    """Attach regime labels and scores to each trade's entry bar."""
    time_idx = pd.DatetimeIndex(df["time"])
    enriched = []
    for t in trades:
        entry = pd.Timestamp(t["entry_time"])
        if entry.tzinfo is None:
            entry = entry.tz_localize("UTC")
        idx = time_idx.get_indexer([entry], method="ffill")[0]
        if idx < 0:
            idx = time_idx.get_indexer([entry], method="bfill")[0]
        if idx < 0:
            t["regime_stateless"] = "unknown"
            t["regime_hysteresis"] = "unknown"
            enriched.append(t)
            continue

        row = df.iloc[idx]
        t["regime_stateless"] = row.get("regime_stateless", "unknown")
        t["regime_hysteresis"] = row.get("regime_hysteresis", "unknown")
        t["score_momentum"] = float(row.get("score_momentum", 0))
        t["score_mean_reversion"] = float(row.get("score_mean_reversion", 0))
        t["score_breakout"] = float(row.get("score_breakout", 0))
        t["score_post_breakout_trend"] = float(row.get("score_post_breakout_trend", 0))
        t["score_margin"] = float(row.get("score_margin", 0))
        t["adx"] = float(row.get("adx", 0)) if pd.notna(row.get("adx")) else None
        t["m5_slope_abs"] = float(row.get("m5_slope_abs", 0)) if pd.notna(row.get("m5_slope_abs")) else None
        t["h1_aligned"] = bool(row.get("h1_aligned", 0)) if pd.notna(row.get("h1_aligned")) else None
        t["bb_width_pctile"] = float(row.get("bb_width_pctile", 0)) if pd.notna(row.get("bb_width_pctile")) else None
        t["dir_efficiency"] = float(row.get("dir_efficiency", 0)) if pd.notna(row.get("dir_efficiency")) else None
        t["compression_duration"] = int(row.get("compression_duration", 0)) if pd.notna(row.get("compression_duration")) else None
        t["bb_width_exp_rate"] = float(row.get("bb_width_exp_rate", 0)) if pd.notna(row.get("bb_width_exp_rate")) else None
        enriched.append(t)
    return enriched


def _pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 1) if d > 0 else 0.0


def _wr(trades: list[dict]) -> float:
    if not trades:
        return 0.0
    return round(100.0 * sum(1 for t in trades if t["profitable"]) / len(trades), 1)


def _avg_score(trades: list[dict], key: str) -> float:
    vals = [t.get(key, 0.0) for t in trades if t.get(key) is not None]
    return round(float(np.mean(vals)), 2) if vals else 0.0


def build_report(
    df: pd.DataFrame,
    all_trades: list[dict],
    input_csv: str,
    experiment_name: str,
    th: RegimeThresholds,
) -> dict[str, Any]:
    valid = df.dropna(subset=["adx", "m5_slope_abs", "bb_width_pctile"])
    total_bars = len(valid)

    regime_labels = ["momentum", "mean_reversion", "breakout", "post_breakout_trend", "ambiguous"]

    # Bar distribution
    bar_dist: dict[str, Any] = {}
    for mode in ["stateless", "hysteresis"]:
        col = f"regime_{mode}"
        dist = {}
        for r in regime_labels:
            n = int((valid[col] == r).sum())
            dist[r] = {"count": n, "pct": _pct(n, total_bars)}
        bar_dist[mode] = dist

    # Hysteresis hold rate
    held = int(valid["regime_held"].sum())
    bar_dist["hysteresis_held_pct"] = _pct(held, total_bars)

    # Per-strategy breakdown by regime
    strategies = ["v14", "london_v2", "v44_ny"]
    strat_by_regime: dict[str, Any] = {}

    for strat in strategies:
        strat_trades = [t for t in all_trades if t["strategy"] == strat]
        natural = STRATEGY_NATURAL_REGIME[strat]
        strat_result: dict[str, Any] = {
            "total": len(strat_trades),
            "win_rate": _wr(strat_trades),
            "natural_regime": natural,
        }

        for mode in ["stateless", "hysteresis"]:
            regime_key = f"regime_{mode}"
            by_regime: dict[str, Any] = {}
            for r in regime_labels:
                in_regime = [t for t in strat_trades if t.get(regime_key) == r]
                wins = [t for t in in_regime if t["profitable"]]
                losses = [t for t in in_regime if not t["profitable"]]
                by_regime[r] = {
                    "trades": len(in_regime),
                    "wins": len(wins),
                    "losses": len(losses),
                    "win_rate": _wr(in_regime),
                    "pct_of_strat": _pct(len(in_regime), len(strat_trades)),
                }
            strat_result[mode] = by_regime

            # "Right regime" rate
            in_natural = [t for t in strat_trades if t.get(regime_key) == natural]
            strat_result[f"{mode}_in_natural_regime_pct"] = _pct(len(in_natural), len(strat_trades))
            strat_result[f"{mode}_in_natural_regime_wr"] = _wr(in_natural)

        # Score distributions at wins vs losses
        wins = [t for t in strat_trades if t["profitable"]]
        losses = [t for t in strat_trades if not t["profitable"]]
        score_key = {
            "v14": "score_mean_reversion",
            "london_v2": "score_breakout",
            "v44_ny": "score_momentum",
        }[strat]
        strat_result["natural_score_at_wins"] = _avg_score(wins, score_key)
        strat_result["natural_score_at_losses"] = _avg_score(losses, score_key)
        strat_result["margin_at_wins"] = _avg_score(wins, "score_margin")
        strat_result["margin_at_losses"] = _avg_score(losses, "score_margin")

        strat_by_regime[strat] = strat_result

    # "Would the router have helped?" — trades in wrong regime
    misrouted: dict[str, Any] = {}
    for strat in strategies:
        strat_trades = [t for t in all_trades if t["strategy"] == strat]
        natural = STRATEGY_NATURAL_REGIME[strat]
        wrong = [t for t in strat_trades if t.get("regime_hysteresis") != natural and t.get("regime_hysteresis") != "ambiguous"]
        wrong_losses = [t for t in wrong if not t["profitable"]]
        misrouted[strat] = {
            "total_in_wrong_regime": len(wrong),
            "losses_in_wrong_regime": len(wrong_losses),
            "potential_loss_avoidance_pips": round(sum(abs(t.get("pips", 0)) for t in wrong_losses), 1),
            "pct_of_losses_misrouted": _pct(
                len(wrong_losses),
                sum(1 for t in strat_trades if not t["profitable"]),
            ),
        }

    # ── Feature signal analysis: new features at wins vs losses ──
    new_features = ["dir_efficiency", "compression_duration", "bb_width_exp_rate"]
    feature_signal: dict[str, Any] = {}
    for strat in strategies:
        strat_trades = [t for t in all_trades if t["strategy"] == strat]
        strat_signal: dict[str, Any] = {}
        for r in regime_labels:
            in_regime = [t for t in strat_trades if t.get("regime_hysteresis") == r]
            if not in_regime:
                continue
            wins = [t for t in in_regime if t["profitable"]]
            losses = [t for t in in_regime if not t["profitable"]]
            bucket: dict[str, Any] = {"trades": len(in_regime), "wins": len(wins), "losses": len(losses)}
            for feat in new_features:
                w_vals = [t[feat] for t in wins if t.get(feat) is not None]
                l_vals = [t[feat] for t in losses if t.get(feat) is not None]
                bucket[feat] = {
                    "win_median": round(float(np.median(w_vals)), 4) if w_vals else None,
                    "win_mean": round(float(np.mean(w_vals)), 4) if w_vals else None,
                    "loss_median": round(float(np.median(l_vals)), 4) if l_vals else None,
                    "loss_mean": round(float(np.mean(l_vals)), 4) if l_vals else None,
                    "win_n": len(w_vals),
                    "loss_n": len(l_vals),
                }
            strat_signal[r] = bucket
        # Also compute across all regimes
        all_wins = [t for t in strat_trades if t["profitable"]]
        all_losses = [t for t in strat_trades if not t["profitable"]]
        overall: dict[str, Any] = {"trades": len(strat_trades), "wins": len(all_wins), "losses": len(all_losses)}
        for feat in new_features:
            w_vals = [t[feat] for t in all_wins if t.get(feat) is not None]
            l_vals = [t[feat] for t in all_losses if t.get(feat) is not None]
            overall[feat] = {
                "win_median": round(float(np.median(w_vals)), 4) if w_vals else None,
                "win_mean": round(float(np.mean(w_vals)), 4) if w_vals else None,
                "loss_median": round(float(np.median(l_vals)), 4) if l_vals else None,
                "loss_mean": round(float(np.mean(l_vals)), 4) if l_vals else None,
                "win_n": len(w_vals),
                "loss_n": len(l_vals),
            }
        strat_signal["_overall"] = overall
        feature_signal[strat] = strat_signal

    return {
        "_meta": {
            "dataset": Path(input_csv).stem,
            "total_bars": total_bars,
            "total_trades": len(all_trades),
            "experiment": experiment_name,
            "thresholds": {
                "soft_scoring_enabled": th.soft_scoring_enabled,
                "soft_ramp_half_width_adx": th.soft_ramp_half_width_adx,
                "soft_ramp_half_width_slope": th.soft_ramp_half_width_slope,
                "soft_ramp_half_width_bb_pctile": th.soft_ramp_half_width_bb_pctile,
                "score_ema_enabled": th.score_ema_enabled,
                "score_ema_alpha": th.score_ema_alpha,
                "schmitt_enabled": th.schmitt_enabled,
                "schmitt_enter_threshold": th.schmitt_enter_threshold,
                "schmitt_exit_threshold": th.schmitt_exit_threshold,
            },
        },
        "bar_distribution": bar_dist,
        "strategy_by_regime": strat_by_regime,
        "misrouted_analysis": misrouted,
        "feature_signal_analysis": feature_signal,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate regime classifier against backtest trades")
    p.add_argument("--input", required=True, help="M1 OANDA CSV")
    p.add_argument("--output", required=True, help="Output JSON path")
    p.add_argument("--experiment-name", default="legacy_baseline", help="Short experiment label for report metadata")
    p.add_argument("--soft-scoring", action="store_true", help="Enable soft ramp scoring")
    p.add_argument("--score-ema", action="store_true", help="Enable EMA smoothing of regime scores")
    p.add_argument("--score-ema-alpha", type=float, default=0.3, help="EMA alpha when --score-ema is enabled")
    p.add_argument("--schmitt", action="store_true", help="Enable Schmitt-trigger hysteresis")
    p.add_argument("--schmitt-enter-threshold", type=float, default=2.8, help="Smoothed score needed to enter a regime")
    p.add_argument("--schmitt-exit-threshold", type=float, default=2.0, help="Smoothed score floor to keep current regime")
    p.add_argument("--soft-ramp-half-width-adx", type=float, default=5.0, help="Half-width for ADX soft ramps")
    p.add_argument("--soft-ramp-half-width-slope", type=float, default=0.4, help="Half-width for slope soft ramps")
    p.add_argument("--soft-ramp-half-width-bb-pctile", type=float, default=0.15, help="Half-width for BB-width percentile ramps")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_csv = str(Path(args.input).resolve())
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    th = RegimeThresholds(
        soft_scoring_enabled=args.soft_scoring,
        soft_ramp_half_width_adx=args.soft_ramp_half_width_adx,
        soft_ramp_half_width_slope=args.soft_ramp_half_width_slope,
        soft_ramp_half_width_bb_pctile=args.soft_ramp_half_width_bb_pctile,
        score_ema_enabled=args.score_ema,
        score_ema_alpha=args.score_ema_alpha,
        schmitt_enabled=args.schmitt,
        schmitt_enter_threshold=args.schmitt_enter_threshold,
        schmitt_exit_threshold=args.schmitt_exit_threshold,
    )

    print(f"[1/6] Loading M1 data from {input_csv} ...")
    m1 = _load_m1(input_csv)
    print(f"       {len(m1)} bars")

    print("[2/6] Computing regime features ...")
    featured = compute_features(m1)

    print("[3/6] Classifying every bar (stateless + hysteresis) ...")
    classified = classify_all_bars(featured, th)
    valid = classified.dropna(subset=["adx", "m5_slope_abs", "bb_width_pctile"])
    total = len(valid)
    for mode in ["stateless", "hysteresis"]:
        col = f"regime_{mode}"
        print(f"       {mode}:")
        for r in ["momentum", "mean_reversion", "breakout", "post_breakout_trend", "ambiguous"]:
            n = int((valid[col] == r).sum())
            print(f"         {r:25s} {n:>8d}  ({100*n/total:5.1f}%)")

    print("[4/6] Running standalone backtests ...")
    print("       V14 ...")
    v14_trades = _run_v14(input_csv)
    print(f"       → {len(v14_trades)} trades")
    print("       London V2 ...")
    v2_trades = _run_london_v2(m1)
    print(f"       → {len(v2_trades)} trades")
    print("       V44 NY ...")
    v44_trades = _run_v44(input_csv)
    print(f"       → {len(v44_trades)} trades")
    all_trades = v14_trades + v2_trades + v44_trades

    print("[5/6] Labeling trades with regime at entry ...")
    enriched = label_trades(classified, all_trades)

    print("[6/6] Building report ...")
    report = build_report(classified, enriched, input_csv, args.experiment_name, th)

    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nResults written to {out_path}")

    # Compact summary
    print("\n" + "=" * 75)
    print("REGIME VALIDATION SUMMARY")
    print("=" * 75)

    for strat in ["v14", "london_v2", "v44_ny"]:
        s = report["strategy_by_regime"][strat]
        natural = s["natural_regime"]
        print(f"\n{strat} ({s['total']} trades, {s['win_rate']}% WR)")
        print(f"  Natural regime: {natural}")
        print(f"  In natural regime (hysteresis): {s['hysteresis_in_natural_regime_pct']}% of trades, {s['hysteresis_in_natural_regime_wr']}% WR")
        print(f"  Natural score: wins={s['natural_score_at_wins']}, losses={s['natural_score_at_losses']}")
        print(f"  Margin: wins={s['margin_at_wins']}, losses={s['margin_at_losses']}")
        print(f"  By regime (hysteresis):")
        for r in ["momentum", "mean_reversion", "breakout", "post_breakout_trend", "ambiguous"]:
            d = s["hysteresis"][r]
            if d["trades"] > 0:
                marker = " ←" if r == natural else ""
                print(f"    {r:25s} {d['trades']:>4d} trades  WR={d['win_rate']:>5.1f}%  ({d['pct_of_strat']:>5.1f}% of strat){marker}")

    mr = report["misrouted_analysis"]
    print("\n" + "-" * 75)
    print("MISROUTED LOSS ANALYSIS (trades in wrong regime)")
    print("-" * 75)
    for strat in ["v14", "london_v2", "v44_ny"]:
        m = mr[strat]
        print(f"  {strat}: {m['total_in_wrong_regime']} in wrong regime, "
              f"{m['losses_in_wrong_regime']} losses ({m['pct_of_losses_misrouted']}% of all losses), "
              f"{m['potential_loss_avoidance_pips']} pips avoidable")

    # ── Feature signal analysis summary ──
    fsa = report.get("feature_signal_analysis", {})
    feat_names = ["dir_efficiency", "compression_duration", "bb_width_exp_rate"]
    feat_short = {"dir_efficiency": "dir_eff", "compression_duration": "comp_dur", "bb_width_exp_rate": "bb_exp_r"}
    print("\n" + "=" * 75)
    print("NEW FEATURE SIGNAL ANALYSIS (median at wins vs losses)")
    print("=" * 75)
    for strat in ["v14", "london_v2", "v44_ny"]:
        s = fsa.get(strat, {})
        print(f"\n  {strat}:")
        for regime_key in ["_overall", "momentum", "mean_reversion", "breakout", "post_breakout_trend", "ambiguous"]:
            bucket = s.get(regime_key)
            if not bucket or bucket.get("trades", 0) == 0:
                continue
            label = "ALL" if regime_key == "_overall" else regime_key
            w = bucket.get("wins", 0)
            lo = bucket.get("losses", 0)
            parts = []
            for feat in feat_names:
                fd = bucket.get(feat, {})
                wm = fd.get("win_median")
                lm = fd.get("loss_median")
                wn = fd.get("win_n", 0)
                ln = fd.get("loss_n", 0)
                ws = f"{wm:.3f}" if wm is not None else "  n/a"
                ls = f"{lm:.3f}" if lm is not None else "  n/a"
                parts.append(f"{feat_short[feat]}: W={ws}({wn}) L={ls}({ln})")
            print(f"    {label:25s} {w:>3d}W/{lo:>3d}L  {' | '.join(parts)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
