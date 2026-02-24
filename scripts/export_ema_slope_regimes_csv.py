#!/usr/bin/env python3
"""
Usage:
  python scripts/export_ema_slope_regimes_csv.py \
    --in USDJPY_M1_50k.csv \
    --out USDJPY_M1_ema_slopes.csv \
    --summary-out USDJPY_M1_ema_slope_summary.csv \
    --lookback 3 \
    --pip-size 0.01
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Thresholds:
    run_ema5: float
    run_ema9: float
    run_ema21: float
    flat_ema5: float
    flat_ema9: float
    flat_ema21: float


def parse_triplet(name: str, value: str) -> tuple[float, float, float]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3:
        raise ValueError(f"{name} must be 3 comma-separated numbers (got: {value})")
    try:
        a, b, c = (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError as exc:
        raise ValueError(f"{name} contains a non-numeric value: {value}") from exc
    return a, b, c


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export EMA5/EMA9/EMA21 slope regimes from an M1 OHLC CSV."
    )
    p.add_argument("--in", dest="input_csv", required=True, help="Input CSV with columns: time,open,high,low,close")
    p.add_argument("--out", default="USDJPY_M1_ema_slopes.csv", help="Per-bar output CSV path")
    p.add_argument("--summary-out", default="USDJPY_M1_ema_slope_summary.csv", help="Summary CSV path")
    p.add_argument("--pip-size", type=float, default=0.01, help="Pip size (USDJPY=0.01)")
    p.add_argument("--lookback", type=int, default=3, help="Bars back used for slope per bar")
    p.add_argument(
        "--run-thresholds",
        default="0.05,0.04,0.03",
        help="Run thresholds in pips/bar for EMA5,EMA9,EMA21",
    )
    p.add_argument(
        "--flat-thresholds",
        default="0.02,0.016,0.012",
        help="Horizontal thresholds in pips/bar for EMA5,EMA9,EMA21",
    )
    return p.parse_args()


def classify_regime(row: pd.Series, th: Thresholds) -> str:
    s5 = float(row["slope_ema5_pips_per_bar"])
    s9 = float(row["slope_ema9_pips_per_bar"])
    s21 = float(row["slope_ema21_pips_per_bar"])

    all_pos = s5 > 0 and s9 > 0 and s21 > 0
    all_neg = s5 < 0 and s9 < 0 and s21 < 0
    run_ok = (
        abs(s5) >= th.run_ema5
        and abs(s9) >= th.run_ema9
        and abs(s21) >= th.run_ema21
    )
    if run_ok and all_pos:
        return "running_bull"
    if run_ok and all_neg:
        return "running_bear"

    flat_ok = (
        abs(s5) <= th.flat_ema5
        and abs(s9) <= th.flat_ema9
        and abs(s21) <= th.flat_ema21
    )
    if flat_ok:
        return "horizontal"
    return "transition"


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    quantiles = [0.5, 0.8, 0.9, 0.95]
    for regime, g in df.groupby("regime", dropna=False):
        rec: dict[str, object] = {
            "regime": str(regime),
            "count": int(len(g)),
            "pct_of_total": round((len(g) / len(df)) * 100.0, 2) if len(df) else 0.0,
        }
        for ema_col in (
            "slope_ema5_pips_per_bar",
            "slope_ema9_pips_per_bar",
            "slope_ema21_pips_per_bar",
            "abs_slope_ema5",
            "abs_slope_ema9",
            "abs_slope_ema21",
        ):
            series = g[ema_col].astype(float)
            rec[f"{ema_col}_min"] = round(float(series.min()), 6)
            rec[f"{ema_col}_max"] = round(float(series.max()), 6)
            rec[f"{ema_col}_mean"] = round(float(series.mean()), 6)
            for q in quantiles:
                rec[f"{ema_col}_q{int(q*100)}"] = round(float(series.quantile(q)), 6)
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(by="count", ascending=False)


def main() -> None:
    args = parse_args()
    if args.lookback < 1:
        print("Error: --lookback must be >= 1", file=sys.stderr)
        raise SystemExit(1)
    if args.pip_size <= 0:
        print("Error: --pip-size must be > 0", file=sys.stderr)
        raise SystemExit(1)

    try:
        run5, run9, run21 = parse_triplet("run-thresholds", args.run_thresholds)
        flat5, flat9, flat21 = parse_triplet("flat-thresholds", args.flat_thresholds)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)

    try:
        df = pd.read_csv(args.input_csv)
    except Exception as exc:
        print(f"Error: failed to read input CSV '{args.input_csv}': {exc}", file=sys.stderr)
        raise SystemExit(1)

    required = {"time", "open", "high", "low", "close"}
    if not required.issubset(set(df.columns)):
        print(f"Error: input CSV must contain columns: {','.join(sorted(required))}", file=sys.stderr)
        raise SystemExit(1)

    out = df[["time", "close"]].copy()
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["close"]).reset_index(drop=True)
    if out.empty:
        print("Error: no valid close prices found in input CSV", file=sys.stderr)
        raise SystemExit(1)

    out["ema5"] = out["close"].ewm(span=5, adjust=False).mean()
    out["ema9"] = out["close"].ewm(span=9, adjust=False).mean()
    out["ema21"] = out["close"].ewm(span=21, adjust=False).mean()

    lb = int(args.lookback)
    out["slope_ema5_pips_per_bar"] = ((out["ema5"] - out["ema5"].shift(lb)) / lb) / args.pip_size
    out["slope_ema9_pips_per_bar"] = ((out["ema9"] - out["ema9"].shift(lb)) / lb) / args.pip_size
    out["slope_ema21_pips_per_bar"] = ((out["ema21"] - out["ema21"].shift(lb)) / lb) / args.pip_size

    out = out.dropna(
        subset=[
            "slope_ema5_pips_per_bar",
            "slope_ema9_pips_per_bar",
            "slope_ema21_pips_per_bar",
        ]
    ).reset_index(drop=True)

    out["abs_slope_ema5"] = out["slope_ema5_pips_per_bar"].abs()
    out["abs_slope_ema9"] = out["slope_ema9_pips_per_bar"].abs()
    out["abs_slope_ema21"] = out["slope_ema21_pips_per_bar"].abs()

    thresholds = Thresholds(
        run_ema5=run5,
        run_ema9=run9,
        run_ema21=run21,
        flat_ema5=flat5,
        flat_ema9=flat9,
        flat_ema21=flat21,
    )
    out["regime"] = out.apply(lambda r: classify_regime(r, thresholds), axis=1)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "time",
                "close",
                "ema5",
                "ema9",
                "ema21",
                "slope_ema5_pips_per_bar",
                "slope_ema9_pips_per_bar",
                "slope_ema21_pips_per_bar",
                "abs_slope_ema5",
                "abs_slope_ema9",
                "abs_slope_ema21",
                "regime",
            ]
        )
        for row in out.itertuples(index=False):
            writer.writerow(
                [
                    row.time,
                    f"{row.close:.6f}",
                    f"{row.ema5:.6f}",
                    f"{row.ema9:.6f}",
                    f"{row.ema21:.6f}",
                    f"{row.slope_ema5_pips_per_bar:.6f}",
                    f"{row.slope_ema9_pips_per_bar:.6f}",
                    f"{row.slope_ema21_pips_per_bar:.6f}",
                    f"{row.abs_slope_ema5:.6f}",
                    f"{row.abs_slope_ema9:.6f}",
                    f"{row.abs_slope_ema21:.6f}",
                    row.regime,
                ]
            )

    summary = build_summary(out)
    summary.to_csv(args.summary_out, index=False)

    print(f"Wrote {len(out)} rows -> {args.out}")
    print(f"Wrote {len(summary)} rows -> {args.summary_out}")


if __name__ == "__main__":
    main()
