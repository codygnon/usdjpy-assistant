#!/usr/bin/env python3
"""
Chop-threshold study for Trial 10 on USDJPY.

Evaluates four candidate chop blockers using real historical Trial 10 trades:
1. M5 EMA gap compression
2. M5 ADX
3. M5 Bollinger Band width
4. Composite chop score built from the three indicators above

Outputs:
- JSON summary with full threshold sweeps and recommendations
- Markdown report
- CSV of threshold-level results
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CAL_PATH = ROOT / "research_out" / "trial10_indicator_calibration_250k.csv"
TRADES_PATH = ROOT / "research_out" / "trial10_250k_pause1_trades.csv"
M1_PATH = ROOT / "research_out" / "USDJPY_M1_OANDA_250k.csv"
OUT_JSON = ROOT / "research_out" / "trial10_chop_threshold_study.json"
OUT_MD = ROOT / "research_out" / "trial10_chop_threshold_study.md"
OUT_CSV = ROOT / "research_out" / "trial10_chop_threshold_candidates.csv"

PIP_SIZE = 0.01


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    if df is None or len(df) < period + 1:
        return pd.Series(dtype=float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = df["close"].astype(float).shift(1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    alpha = 1.0 / float(period)
    atr_series = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_series.replace(0, np.nan).ffill().fillna(1e-10)
    minus_di = 100.0 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_series.replace(0, np.nan).ffill().fillna(1e-10)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan).ffill().fillna(1e-10)
    return dx.ewm(alpha=alpha, adjust=False).mean()


def bollinger_width(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.Series:
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return (upper - lower) / PIP_SIZE


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cal = pd.read_csv(CAL_PATH, parse_dates=["time"], low_memory=False)
    trades = pd.read_csv(TRADES_PATH, parse_dates=["signal_time", "entry_time", "exit_time"])
    m1 = pd.read_csv(M1_PATH, parse_dates=["time"])
    for df in (cal, trades, m1):
        for col in ("time", "signal_time", "entry_time", "exit_time"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True)
    return cal, trades, m1


def prepare_m5_indicator_frame(m1: pd.DataFrame) -> pd.DataFrame:
    m1 = m1.sort_values("time").copy()
    m1 = m1.set_index("time")
    ohlc = (
        m1[["open", "high", "low", "close"]]
        .resample("5min", label="right", closed="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )
    ohlc["m5_adx_14"] = adx(ohlc, 14)
    ohlc["m5_bb_width_20_2"] = bollinger_width(ohlc["close"], 20, 2.0)
    return ohlc.reset_index()


def build_trade_feature_frame(cal: pd.DataFrame, trades: pd.DataFrame, m1: pd.DataFrame) -> pd.DataFrame:
    trades = trades.copy()
    trades["signal_time_rounded"] = trades["signal_time"].dt.floor("min")
    trades["signal_time_m5"] = trades["signal_time"].dt.floor("5min")

    cal_cols = [
        "time",
        "m5_bucket",
        "m5_gap_pips",
        "m5_atr_pips",
        "m5_ema9_slope_pips_per_bar",
        "m5_slope_aligned",
        "m1_compressing",
        "m1_bucket",
        "m1_aligned",
    ]
    merged = trades.merge(
        cal[cal_cols],
        left_on="signal_time_rounded",
        right_on="time",
        how="left",
        suffixes=("", "_cal"),
    )

    m5_ind = prepare_m5_indicator_frame(m1)
    merged = merged.merge(
        m5_ind[["time", "m5_adx_14", "m5_bb_width_20_2"]],
        left_on="signal_time_m5",
        right_on="time",
        how="left",
        suffixes=("", "_m5"),
    )

    merged = merged[merged["signal_time_rounded"].notna()].copy()
    merged["is_loser"] = merged["profit_usd"] < 0
    merged["is_winner"] = merged["profit_usd"] > 0
    merged["is_fast_stop"] = merged["exit_reason"].fillna("").eq("initial_stop")
    merged["is_runner_like"] = merged["tp1_done"].fillna(False).astype(bool)
    merged = merged.sort_values("entry_time").reset_index(drop=True)

    # Percentile-style ranks for composite chopiness.
    merged["gap_chop_rank"] = 1.0 - merged["m5_gap_pips"].rank(pct=True, na_option="keep")
    merged["adx_chop_rank"] = 1.0 - merged["m5_adx_14"].rank(pct=True, na_option="keep")
    merged["bbw_chop_rank"] = 1.0 - merged["m5_bb_width_20_2"].rank(pct=True, na_option="keep")
    merged["composite_chop_score"] = merged[["gap_chop_rank", "adx_chop_rank", "bbw_chop_rank"]].mean(axis=1)
    return merged


@dataclass
class EvalResult:
    feature: str
    rule: str
    threshold: float
    blocked_trades: int
    blocked_pct: float
    allowed_trades: int
    baseline_net_usd: float
    allowed_net_usd: float
    pnl_delta_usd: float
    avg_allowed_pips: float
    blocked_losers: int
    blocked_winners: int
    loser_per_winner: float | None
    blocked_fast_stops: int
    blocked_runner_like: int
    block_fast_stop_rate: float
    block_runner_like_rate: float
    early_delta_usd: float
    middle_delta_usd: float
    late_delta_usd: float
    positive_blocks: int


def _safe_ratio(n: float, d: float) -> float | None:
    if d == 0:
        return None
    return n / d


def evaluate_mask(df: pd.DataFrame, mask: pd.Series, feature: str, rule: str, threshold: float) -> EvalResult:
    mask = mask.fillna(False)
    total = len(df)
    blocked = df[mask]
    allowed = df[~mask]
    baseline = float(df["profit_usd"].sum())
    allowed_net = float(allowed["profit_usd"].sum())
    blocked_losers = int(blocked["is_loser"].sum())
    blocked_winners = int(blocked["is_winner"].sum())
    loser_per_winner = _safe_ratio(blocked_losers, blocked_winners)

    blocks = np.array_split(df.index.to_numpy(), 3)
    block_deltas: list[float] = []
    for idx in blocks:
        part = df.loc[idx]
        part_mask = mask.loc[idx]
        part_delta = float(part.loc[~part_mask, "profit_usd"].sum() - part["profit_usd"].sum())
        block_deltas.append(part_delta)

    return EvalResult(
        feature=feature,
        rule=rule,
        threshold=float(threshold),
        blocked_trades=int(mask.sum()),
        blocked_pct=round(float(mask.mean() * 100.0), 2),
        allowed_trades=int((~mask).sum()),
        baseline_net_usd=round(baseline, 2),
        allowed_net_usd=round(allowed_net, 2),
        pnl_delta_usd=round(allowed_net - baseline, 2),
        avg_allowed_pips=round(float(allowed["weighted_pips"].mean()) if len(allowed) else 0.0, 4),
        blocked_losers=blocked_losers,
        blocked_winners=blocked_winners,
        loser_per_winner=None if loser_per_winner is None else round(float(loser_per_winner), 3),
        blocked_fast_stops=int(blocked["is_fast_stop"].sum()),
        blocked_runner_like=int(blocked["is_runner_like"].sum()),
        block_fast_stop_rate=round(float(blocked["is_fast_stop"].mean() * 100.0) if len(blocked) else 0.0, 2),
        block_runner_like_rate=round(float(blocked["is_runner_like"].mean() * 100.0) if len(blocked) else 0.0, 2),
        early_delta_usd=round(block_deltas[0], 2),
        middle_delta_usd=round(block_deltas[1], 2),
        late_delta_usd=round(block_deltas[2], 2),
        positive_blocks=sum(1 for x in block_deltas if x > 0),
    )


def sweep_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    results: list[EvalResult] = []

    gap_thresholds = [1.5, 1.8, 2.0, 2.3, 2.5, 3.0]
    for threshold in gap_thresholds:
        mask = df["m5_gap_pips"].abs() < threshold
        results.append(evaluate_mask(df, mask, "m5_gap", f"block |gap| < {threshold:.2f}p", threshold))

    for low, high in [(1.8, 4.8), (2.0, 4.8), (2.0, 5.5), (2.3, 4.8), (2.3, 5.5), (2.3, 6.5)]:
        mask = ~df["m5_gap_pips"].between(low, high)
        results.append(evaluate_mask(df, mask, "m5_gap", f"keep gap {low:.1f}-{high:.1f}p", high))

    adx_thresholds = [10, 12, 14, 16, 18, 20, 22, 25, 28, 30]
    for threshold in adx_thresholds:
        mask = df["m5_adx_14"] < threshold
        results.append(evaluate_mask(df, mask, "m5_adx", f"block adx < {threshold:.0f}", threshold))

    for low, high in [(16, 31), (16, 36), (18, 31), (18, 36), (18, 41), (20, 31), (20, 36)]:
        mask = ~df["m5_adx_14"].between(low, high)
        results.append(evaluate_mask(df, mask, "m5_adx", f"keep adx {low:.0f}-{high:.0f}", high))

    bb_width_thresholds = [12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 25.0, 28.0, 30.0]
    for threshold in bb_width_thresholds:
        mask = df["m5_bb_width_20_2"] < threshold
        results.append(evaluate_mask(df, mask, "m5_bb_width", f"block bb_width < {threshold:.1f}p", threshold))

    for low, high in [(16, 28), (18, 28), (18, 32), (20, 28), (20, 32), (20, 38), (22, 32), (22, 38)]:
        mask = ~df["m5_bb_width_20_2"].between(low, high)
        results.append(evaluate_mask(df, mask, "m5_bb_width", f"keep bb_width {low:.0f}-{high:.0f}p", high))

    composite_thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    for threshold in composite_thresholds:
        mask = df["composite_chop_score"] >= threshold
        results.append(evaluate_mask(df, mask, "composite", f"block score >= {threshold:.2f}", threshold))

    for low, high in [(0.20, 0.60), (0.20, 0.70), (0.25, 0.60), (0.25, 0.65), (0.25, 0.70), (0.30, 0.65)]:
        mask = ~df["composite_chop_score"].between(low, high)
        results.append(evaluate_mask(df, mask, "composite", f"keep score {low:.2f}-{high:.2f}", high))

    out = pd.DataFrame(asdict(r) for r in results)
    out["quality_score"] = (
        out["pnl_delta_usd"]
        + (out["positive_blocks"] * 500.0)
        + out["loser_per_winner"].fillna(0.0) * 100.0
        + out["avg_allowed_pips"] * 100.0
        - out["blocked_pct"] * 6.0
        - (out["blocked_pct"] > 35.0).astype(float) * 150.0
    )
    return out.sort_values(["quality_score", "pnl_delta_usd"], ascending=False).reset_index(drop=True)


def choose_recommendations(results: pd.DataFrame) -> dict[str, dict]:
    recs: dict[str, dict] = {}
    for feature in ["m5_gap", "m5_adx", "m5_bb_width", "composite"]:
        subset = results[results["feature"] == feature].copy()
        subset = subset[
            (subset["blocked_trades"] >= 100)
            & (subset["blocked_pct"] <= 35.0)
            & (subset["positive_blocks"] == 3)
        ].copy()
        if subset.empty:
            subset = results[results["feature"] == feature].copy()
        subset = subset.sort_values(
            ["quality_score", "pnl_delta_usd", "loser_per_winner", "avg_allowed_pips"],
            ascending=[False, False, False, False],
        )
        row = subset.iloc[0].to_dict()
        recs[feature] = row
    return recs


def build_markdown(merged: pd.DataFrame, results: pd.DataFrame, recs: dict[str, dict]) -> str:
    baseline_net = float(merged["profit_usd"].sum())
    baseline_avg_pips = float(merged["weighted_pips"].mean())
    lines = [
        "# Trial 10 Chop Threshold Study",
        "",
        f"- Trades analyzed: `{len(merged)}`",
        f"- Baseline net: `${baseline_net:.2f}`",
        f"- Baseline avg pips/trade: `{baseline_avg_pips:.4f}`",
        f"- Data sources: `{TRADES_PATH}` + `{CAL_PATH}` + `{M1_PATH}`",
        "",
        "## Recommended Thresholds",
        "",
        "> Note: none of these single-feature rules cleanly block more losers than winners.",
        "> Treat them as light damage-control filters, not magic chop detectors.",
        "",
    ]
    labels = {
        "m5_gap": "M5 EMA Gap",
        "m5_adx": "M5 ADX(14)",
        "m5_bb_width": "M5 Bollinger Width(20,2)",
        "composite": "Composite Chop Score",
    }
    for key in ["m5_gap", "m5_adx", "m5_bb_width", "composite"]:
        row = recs[key]
        lines.extend(
            [
                f"### {labels[key]}",
                f"- Rule: `{row['rule']}`",
                f"- Blocks: `{int(row['blocked_trades'])}` trades (`{row['blocked_pct']}%`)",
                f"- Net improvement vs baseline: `${row['pnl_delta_usd']}`",
                f"- Avg allowed pips/trade: `{row['avg_allowed_pips']}`",
                f"- Blocked losers/winners: `{int(row['blocked_losers'])}/{int(row['blocked_winners'])}`",
                f"- Loser per winner blocked: `{row['loser_per_winner']}`",
                f"- Walk-forward blocks positive: `{int(row['positive_blocks'])}/3`",
                f"- Early / Middle / Late delta: `${row['early_delta_usd']}` / `${row['middle_delta_usd']}` / `${row['late_delta_usd']}`",
                "",
            ]
        )

    top = results[
        (results["blocked_pct"] <= 35.0) & (results["positive_blocks"] == 3)
    ].head(12)[
        [
            "feature",
            "rule",
            "blocked_pct",
            "pnl_delta_usd",
            "avg_allowed_pips",
            "loser_per_winner",
            "positive_blocks",
        ]
    ]
    lines.extend(
        [
            "## Top Candidates",
            "",
            "| feature | rule | blocked % | pnl delta usd | avg allowed pips | loser/winner | positive blocks |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            "",
            "## Takeaway",
            "",
        ]
    )
    for _, row in top.iterrows():
        lines.insert(
            -3,
            f"| {row['feature']} | {row['rule']} | {row['blocked_pct']} | {row['pnl_delta_usd']} | {row['avg_allowed_pips']} | {row['loser_per_winner']} | {row['positive_blocks']} |",
        )
    best_feature = max(recs.items(), key=lambda kv: (kv[1]["quality_score"], kv[1]["pnl_delta_usd"]))[0]
    lines.append(f"- Strongest overall *light* chop filter in this study: `{labels[best_feature]}`.")
    lines.append("- The data favors moderate filters and mid-range bands more than aggressive hard blocking.")
    lines.append("- If you want the simplest starting point, use the M5 gap or composite recommendation and keep the block rate under about one-third of trades.")
    return "\n".join(lines) + "\n"


def main() -> None:
    cal, trades, m1 = load_inputs()
    merged = build_trade_feature_frame(cal, trades, m1)
    merged = merged[
        merged["m5_gap_pips"].notna()
        & merged["m5_adx_14"].notna()
        & merged["m5_bb_width_20_2"].notna()
    ].copy()

    results = sweep_thresholds(merged)
    recs = choose_recommendations(results)

    OUT_JSON.write_text(
        json.dumps(
            {
                "metadata": {
                    "trades_analyzed": int(len(merged)),
                    "baseline_net_usd": round(float(merged["profit_usd"].sum()), 2),
                    "baseline_avg_pips": round(float(merged["weighted_pips"].mean()), 4),
                    "sources": {
                        "calibration_csv": str(CAL_PATH),
                        "trades_csv": str(TRADES_PATH),
                        "m1_csv": str(M1_PATH),
                    },
                },
                "recommendations": recs,
                "all_results": results.to_dict(orient="records"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    results.to_csv(OUT_CSV, index=False)
    OUT_MD.write_text(build_markdown(merged, results, recs), encoding="utf-8")

    print(f"Trades analyzed: {len(merged)}")
    print(f"Baseline net: ${merged['profit_usd'].sum():.2f}")
    print("Recommendations:")
    for feature, row in recs.items():
        print(
            f"  {feature}: {row['rule']} | delta=${row['pnl_delta_usd']} | "
            f"blocked={row['blocked_pct']}% | LPW={row['loser_per_winner']} | blocks={int(row['positive_blocks'])}/3"
        )
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
