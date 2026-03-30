#!/usr/bin/env python3
"""
Validate a narrow V44 exhaustion penalty on real trade data.

The experiment keeps the base classifier intact and applies a targeted,
multiplicative attenuation to the momentum score only when BOTH:
  - efficiency_ratio is poor
  - trend_decay_rate is high

Rejection intensity and level touch density remain in the dataset for
diagnostics only; they do not affect classification in this experiment.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.regime_classifier import RegimeClassifier, RegimeThresholds
from core.regime_features import (
    RegimeFeatures,
    compute_efficiency_ratio,
    compute_level_touch_density,
    compute_rejection_intensity,
    compute_trend_decay_rate,
)
from scripts.regime_threshold_analysis import (
    _ema,
    _load_m1,
    _resample,
    _rolling_adx,
    _run_london_v2,
    _run_v14,
    _run_v44,
    BB_PERIOD,
    BB_STD,
    BB_WIDTH_LOOKBACK,
    H1_EMA_FAST,
    H1_EMA_SLOW,
    M5_EMA_FAST,
    PIP_SIZE,
    SLOPE_BARS,
)

REGIME_LABELS = ["momentum", "mean_reversion", "breakout", "post_breakout_trend", "ambiguous"]
V44_NATURAL = "momentum"
SWEEP_ER = [0.25, 0.30, 0.35]
SWEEP_DECAY = [0.50, 0.60, 0.70]
SWEEP_ATTENUATOR = [0.85, 0.75]
DATASETS = {
    "500k": ROOT / "research_out/USDJPY_M1_OANDA_500k_train_from_1000k.csv",
    "1000k": ROOT / "research_out/USDJPY_M1_OANDA_800k_split.csv",
}


def _pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 1) if d > 0 else 0.0


def _wr(trades: list[dict[str, Any]]) -> float:
    return round(100.0 * sum(1 for t in trades if t["profitable"]) / len(trades), 1) if trades else 0.0


def _avg(vals: list[float]) -> float | None:
    return round(float(np.mean(vals)), 4) if vals else None


def _median(vals: list[float]) -> float | None:
    return round(float(np.median(vals)), 4) if vals else None


def compute_indicator_features(m1: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    m15["adx"] = _rolling_adx(m15, 14)

    h1_fast = _ema(h1["close"], H1_EMA_FAST)
    h1_slow = _ema(h1["close"], H1_EMA_SLOW)
    h1["h1_trend_dir"] = np.where(h1_fast > h1_slow, "up", "down")

    out = pd.merge_asof(out.sort_values("time"),
                        m5[["time", "bb_regime", "bb_width_pctile", "m5_slope", "m5_slope_abs"]].sort_values("time"),
                        on="time", direction="backward")
    out = pd.merge_asof(out.sort_values("time"),
                        m15[["time", "adx"]].sort_values("time"),
                        on="time", direction="backward")
    out = pd.merge_asof(out.sort_values("time"),
                        h1[["time", "h1_trend_dir"]].sort_values("time"),
                        on="time", direction="backward")

    out["h1_aligned"] = (
        ((out["m5_slope"] > 0) & (out["h1_trend_dir"] == "up"))
        | ((out["m5_slope"] < 0) & (out["h1_trend_dir"] == "down"))
    ).astype(int)
    return out, m5


def compute_structural_features_on_m5(m5: pd.DataFrame) -> pd.DataFrame:
    n = len(m5)
    er_vals = np.full(n, 0.5)
    rej_vals = np.full(n, 0.0)
    touch_vals = np.full(n, 0.0)
    decay_vals = np.full(n, 0.0)

    ema9 = m5["close"].ewm(span=9, adjust=False).mean()
    slope_sign = np.sign((ema9 - ema9.shift(4)).values)

    for i in range(n):
        window = m5.iloc[max(0, i - 50):i + 1]
        if len(window) < 5:
            continue
        ts = int(slope_sign[i]) if not np.isnan(slope_sign[i]) else 0
        er_vals[i] = compute_efficiency_ratio(window, lookback=12)
        rej_vals[i] = compute_rejection_intensity(window, lookback=8, trend_sign=ts)
        touch_vals[i] = compute_level_touch_density(window, lookback=20, zone_pips=5.0)
        decay_vals[i] = compute_trend_decay_rate(window, ema_period=9, slope_bars=4, lookback=12)

    out = m5.copy()
    out["sf_efficiency_ratio"] = er_vals
    out["sf_rejection_intensity"] = rej_vals
    out["sf_level_touch_density"] = touch_vals
    out["sf_trend_decay_rate"] = decay_vals
    return out


def merge_structural_features_to_m1(m1: pd.DataFrame, m5_sf: pd.DataFrame) -> pd.DataFrame:
    sf_cols = ["time", "sf_efficiency_ratio", "sf_rejection_intensity", "sf_level_touch_density", "sf_trend_decay_rate"]
    return pd.merge_asof(
        m1.sort_values("time"),
        m5_sf[sf_cols].sort_values("time"),
        on="time",
        direction="backward",
    )


def classify_bars(df: pd.DataFrame, th: RegimeThresholds) -> pd.DataFrame:
    clf = RegimeClassifier(th)
    labels = []
    score_mom = []
    score_margin = []

    for _, row in df.iterrows():
        adx = float(row.get("adx", 0.0)) if pd.notna(row.get("adx")) else 0.0
        slope = float(row.get("m5_slope_abs", 0.0)) if pd.notna(row.get("m5_slope_abs")) else 0.0
        aligned = bool(row.get("h1_aligned", 0)) if pd.notna(row.get("h1_aligned")) else False
        pctile = float(row.get("bb_width_pctile", 0.5)) if pd.notna(row.get("bb_width_pctile")) else 0.5
        regime = str(row.get("bb_regime", "ranging")) if pd.notna(row.get("bb_regime")) else "ranging"

        features = None
        if th.features_enabled and th.feat_v44_exhaustion_enabled:
            features = RegimeFeatures(
                efficiency_ratio=float(row.get("sf_efficiency_ratio", 0.5)) if pd.notna(row.get("sf_efficiency_ratio")) else 0.5,
                rejection_intensity=0.0,
                level_touch_density=0.0,
                trend_decay_rate=float(row.get("sf_trend_decay_rate", 0.0)) if pd.notna(row.get("sf_trend_decay_rate")) else 0.0,
            )

        res = clf.update(adx, slope, aligned, pctile, regime, features=features)
        labels.append(res.label)
        score_mom.append(res.scores.momentum)
        score_margin.append(res.margin)

    out = df.copy()
    out["regime"] = labels
    out["score_momentum"] = score_mom
    out["score_margin"] = score_margin
    return out


def label_trades(df: pd.DataFrame, trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    time_idx = pd.DatetimeIndex(df["time"])
    enriched: list[dict[str, Any]] = []
    for trade in trades:
        entry = pd.Timestamp(trade["entry_time"])
        if entry.tzinfo is None:
            entry = entry.tz_localize("UTC")
        idx = time_idx.get_indexer([entry], method="ffill")[0]
        if idx < 0:
            idx = time_idx.get_indexer([entry], method="bfill")[0]
        row = df.iloc[idx]
        t = dict(trade)
        t["regime"] = row.get("regime", "unknown")
        t["score_momentum"] = float(row.get("score_momentum", 0.0))
        t["score_margin"] = float(row.get("score_margin", 0.0))
        for key in ["sf_efficiency_ratio", "sf_rejection_intensity", "sf_level_touch_density", "sf_trend_decay_rate"]:
            v = row.get(key)
            t[key] = float(v) if pd.notna(v) else None
        enriched.append(t)
    return enriched


def by_regime(trades: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for regime in REGIME_LABELS:
        bucket = [t for t in trades if t.get("regime") == regime]
        wins = [t for t in bucket if t["profitable"]]
        losses = [t for t in bucket if not t["profitable"]]
        out[regime] = {
            "trades": len(bucket),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": _wr(bucket),
        }
    return out


def misrouted(trades: list[dict[str, Any]]) -> dict[str, Any]:
    total_losses = sum(1 for t in trades if not t["profitable"])
    wrong = [t for t in trades if t.get("regime") not in {V44_NATURAL, "ambiguous"}]
    wrong_losses = [t for t in wrong if not t["profitable"]]
    return {
        "total_in_wrong_regime": len(wrong),
        "losses_in_wrong_regime": len(wrong_losses),
        "avoidable_pips": round(sum(abs(t.get("pips", 0)) for t in wrong_losses), 1),
        "pct_of_losses_misrouted": _pct(len(wrong_losses), total_losses),
    }


def feature_signal(trades: list[dict[str, Any]], regime: str) -> dict[str, Any]:
    scoped = [t for t in trades if t.get("regime") == regime]
    wins = [t for t in scoped if t["profitable"]]
    losses = [t for t in scoped if not t["profitable"]]
    out: dict[str, Any] = {"trades": len(scoped), "wins": len(wins), "losses": len(losses)}
    for feat in ["sf_efficiency_ratio", "sf_trend_decay_rate", "sf_rejection_intensity", "sf_level_touch_density"]:
        w_vals = [t[feat] for t in wins if t.get(feat) is not None]
        l_vals = [t[feat] for t in losses if t.get(feat) is not None]
        out[feat] = {
            "win_mean": _avg(w_vals),
            "win_median": _median(w_vals),
            "loss_mean": _avg(l_vals),
            "loss_median": _median(l_vals),
            "win_n": len(w_vals),
            "loss_n": len(l_vals),
        }
    return out


def classify_variant_name(er: float | None, decay: float | None, att: float | None) -> str:
    if er is None:
        return "baseline"
    return f"er_{er:.2f}_decay_{decay:.2f}_att_{att:.2f}".replace(".", "p")


def evaluate_variant(df: pd.DataFrame, v44_trades: list[dict[str, Any]], th: RegimeThresholds) -> dict[str, Any]:
    classified = classify_bars(df, th)
    labeled_v44 = label_trades(classified, v44_trades)
    return {
        "classified": classified,
        "labeled_v44": labeled_v44,
        "bar_distribution": {
            regime: {
                "count": int((classified["regime"] == regime).sum()),
                "pct": _pct(int((classified["regime"] == regime).sum()), len(classified.dropna(subset=["adx", "m5_slope_abs", "bb_width_pctile"]))),
            }
            for regime in REGIME_LABELS
        },
        "v44": {
            "total": len(labeled_v44),
            "overall_wr": _wr(labeled_v44),
            "by_regime": by_regime(labeled_v44),
            "misrouted": misrouted(labeled_v44),
            "signal": {
                "momentum": feature_signal(labeled_v44, "momentum"),
                "breakout": feature_signal(labeled_v44, "breakout"),
            },
        },
    }


def variant_passes_guardrails(base: dict[str, Any], cand: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons = []
    base_v44 = base["v44"]["by_regime"]
    cand_v44 = cand["v44"]["by_regime"]
    base_mom = base_v44["momentum"]["trades"]
    cand_mom = cand_v44["momentum"]["trades"]
    if cand_mom < max(5, int(round(base_mom * 0.65))):
        reasons.append("momentum_bucket_collapsed")
    if cand_v44["breakout"]["trades"] > base_v44["breakout"]["trades"] + max(5, int(round(base_v44["breakout"]["trades"] * 0.15))):
        reasons.append("breakout_assignment_increased")
    if cand_v44["post_breakout_trend"]["trades"] > base_v44["post_breakout_trend"]["trades"] + max(3, int(round(base_v44["post_breakout_trend"]["trades"] * 0.15))):
        reasons.append("pbt_assignment_increased")
    if abs(cand["bar_distribution"]["momentum"]["pct"] - base["bar_distribution"]["momentum"]["pct"]) > 10.0:
        reasons.append("bar_distribution_distorted")
    if cand["v44"]["misrouted"]["pct_of_losses_misrouted"] > base["v44"]["misrouted"]["pct_of_losses_misrouted"] and cand["v44"]["misrouted"]["avoidable_pips"] > base["v44"]["misrouted"]["avoidable_pips"]:
        reasons.append("wrong_regime_damage_worse")
    return (len(reasons) == 0, reasons)


def candidate_sort_key(base: dict[str, Any], cand: dict[str, Any]) -> tuple:
    base_m = base["v44"]["misrouted"]
    cand_m = cand["v44"]["misrouted"]
    base_v44 = base["v44"]["by_regime"]
    cand_v44 = cand["v44"]["by_regime"]
    return (
        cand_m["losses_in_wrong_regime"] - base_m["losses_in_wrong_regime"],
        cand_m["avoidable_pips"] - base_m["avoidable_pips"],
        cand_v44["breakout"]["trades"] - base_v44["breakout"]["trades"],
        -(cand_v44["momentum"]["win_rate"] - base_v44["momentum"]["win_rate"]),
        -(cand_v44["ambiguous"]["trades"] - base_v44["ambiguous"]["trades"]),
    )


def run_dataset(dataset_name: str, input_csv: Path) -> dict[str, Any]:
    print(f"\n=== {dataset_name} :: {input_csv.name} ===")
    m1 = _load_m1(str(input_csv))
    featured, m5 = compute_indicator_features(m1)
    m5_sf = compute_structural_features_on_m5(m5)
    featured = merge_structural_features_to_m1(featured, m5_sf)

    print("Running standalone backtests once ...")
    _run_v14(str(input_csv))
    _run_london_v2(m1)
    v44_trades = _run_v44(str(input_csv))
    print(f"V44 trades: {len(v44_trades)}")

    baseline_th = RegimeThresholds(features_enabled=False, feat_v44_exhaustion_enabled=False)
    baseline_eval = evaluate_variant(featured, v44_trades, baseline_th)

    variants = []
    best_pass = None
    best_any = None
    for er, decay, att in itertools.product(SWEEP_ER, SWEEP_DECAY, SWEEP_ATTENUATOR):
        name = classify_variant_name(er, decay, att)
        th = RegimeThresholds(
            features_enabled=True,
            feat_v44_exhaustion_enabled=True,
            feat_er_threshold=er,
            feat_decay_threshold=decay,
            feat_momentum_attenuator=att,
        )
        result = evaluate_variant(featured, v44_trades, th)
        passed, reasons = variant_passes_guardrails(baseline_eval, result)
        entry = {
            "name": name,
            "params": {
                "feat_er_threshold": er,
                "feat_decay_threshold": decay,
                "feat_momentum_attenuator": att,
            },
            "guardrail_pass": passed,
            "guardrail_fail_reasons": reasons,
            "bar_distribution": result["bar_distribution"],
            "v44": result["v44"],
            "deltas_vs_baseline": {
                "momentum_trades": result["v44"]["by_regime"]["momentum"]["trades"] - baseline_eval["v44"]["by_regime"]["momentum"]["trades"],
                "momentum_wr": round(result["v44"]["by_regime"]["momentum"]["win_rate"] - baseline_eval["v44"]["by_regime"]["momentum"]["win_rate"], 1),
                "breakout_trades": result["v44"]["by_regime"]["breakout"]["trades"] - baseline_eval["v44"]["by_regime"]["breakout"]["trades"],
                "breakout_wr": round(result["v44"]["by_regime"]["breakout"]["win_rate"] - baseline_eval["v44"]["by_regime"]["breakout"]["win_rate"], 1),
                "ambiguous_trades": result["v44"]["by_regime"]["ambiguous"]["trades"] - baseline_eval["v44"]["by_regime"]["ambiguous"]["trades"],
                "ambiguous_wr": round(result["v44"]["by_regime"]["ambiguous"]["win_rate"] - baseline_eval["v44"]["by_regime"]["ambiguous"]["win_rate"], 1),
                "misrouted_losses": result["v44"]["misrouted"]["losses_in_wrong_regime"] - baseline_eval["v44"]["misrouted"]["losses_in_wrong_regime"],
                "avoidable_pips": round(result["v44"]["misrouted"]["avoidable_pips"] - baseline_eval["v44"]["misrouted"]["avoidable_pips"], 1),
                "pct_of_losses_misrouted": round(result["v44"]["misrouted"]["pct_of_losses_misrouted"] - baseline_eval["v44"]["misrouted"]["pct_of_losses_misrouted"], 1),
            },
        }
        variants.append(entry)
        if best_any is None or candidate_sort_key(baseline_eval, result) < candidate_sort_key(baseline_eval, best_any["result"]):
            best_any = {"entry": entry, "result": result}
        if passed and (best_pass is None or candidate_sort_key(baseline_eval, result) < candidate_sort_key(baseline_eval, best_pass["result"])):
            best_pass = {"entry": entry, "result": result}

    chosen = best_pass if best_pass is not None else best_any
    assert chosen is not None
    chosen_name = chosen["entry"]["name"]

    report = {
        "_meta": {
            "dataset": dataset_name,
            "input_csv": str(input_csv),
            "experiment": "v44_exhaustion_penalty",
            "selection_rule": "minimize V44 wrong-regime damage first; then breakout trade growth; then preserve momentum quality",
            "chosen_variant": chosen_name,
            "chosen_variant_passed_guardrails": chosen["entry"]["guardrail_pass"],
        },
        "baseline": {
            "bar_distribution": baseline_eval["bar_distribution"],
            "v44": baseline_eval["v44"],
        },
        "chosen_variant": chosen["entry"],
        "chosen_result": {
            "bar_distribution": chosen["result"]["bar_distribution"],
            "v44": chosen["result"]["v44"],
        },
        "variants": sorted(variants, key=lambda x: (
            not x["guardrail_pass"],
            x["deltas_vs_baseline"]["misrouted_losses"],
            x["deltas_vs_baseline"]["avoidable_pips"],
            x["deltas_vs_baseline"]["breakout_trades"],
            -x["deltas_vs_baseline"]["momentum_wr"],
        )),
    }
    return report


def build_summary(reports: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"datasets": []}
    for report in reports:
        base = report["baseline"]["v44"]
        chosen = report["chosen_result"]["v44"]
        summary["datasets"].append({
            "dataset": report["_meta"]["dataset"],
            "chosen_variant": report["_meta"]["chosen_variant"],
            "guardrail_pass": report["_meta"]["chosen_variant_passed_guardrails"],
            "v44": {
                "baseline_momentum": base["by_regime"]["momentum"],
                "chosen_momentum": chosen["by_regime"]["momentum"],
                "baseline_breakout": base["by_regime"]["breakout"],
                "chosen_breakout": chosen["by_regime"]["breakout"],
                "baseline_pbt": base["by_regime"]["post_breakout_trend"],
                "chosen_pbt": chosen["by_regime"]["post_breakout_trend"],
                "baseline_ambiguous": base["by_regime"]["ambiguous"],
                "chosen_ambiguous": chosen["by_regime"]["ambiguous"],
            },
            "misrouted": {
                "baseline": base["misrouted"],
                "chosen": chosen["misrouted"],
                "delta_losses": chosen["misrouted"]["losses_in_wrong_regime"] - base["misrouted"]["losses_in_wrong_regime"],
                "delta_pips": round(chosen["misrouted"]["avoidable_pips"] - base["misrouted"]["avoidable_pips"], 1),
            },
            "bar_distribution_delta": {
                regime: round(report["chosen_result"]["bar_distribution"][regime]["pct"] - report["baseline"]["bar_distribution"][regime]["pct"], 1)
                for regime in REGIME_LABELS
            },
            "signal_sanity": {
                "baseline": base["signal"],
                "chosen": chosen["signal"],
            },
        })
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate V44 exhaustion penalty sweep")
    p.add_argument("--run-all", action="store_true", help="Run standard 500k and 1000k datasets")
    p.add_argument("--input", help="Single CSV input")
    p.add_argument("--output", help="Single JSON output")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    reports = []

    if args.run_all:
        for dataset_name, input_csv in DATASETS.items():
            report = run_dataset(dataset_name, input_csv)
            out_path = ROOT / f"research_out/regime_validation_v44_exhaustion_{dataset_name}.json"
            out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
            print(f"Wrote {out_path}")
            reports.append(report)
        summary = build_summary(reports)
        summary_path = ROOT / "research_out/regime_validation_v44_exhaustion_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        print(f"Wrote {summary_path}")
        return 0

    if not args.input or not args.output:
        raise SystemExit("Use --run-all or provide --input and --output")

    report = run_dataset(Path(args.input).stem, Path(args.input))
    Path(args.output).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
