#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(REPO_ROOT))

from core.runner_score import compute_runner_score


RUNNER_EXIT_REASONS = {"runner_stop", "m5_trail_close"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze Trial #10 runner score elite validation on pause1 baseline.")
    p.add_argument("--trades", default="research_out/trial10_250k_pause1_trades.csv")
    p.add_argument("--m1", default="research_out/USDJPY_M1_OANDA_250k.csv")
    p.add_argument("--spread-pips", type=float, default=2.0)
    p.add_argument("--json-out", default="research_out/trial10_pause1_runner_elite_validation.json")
    p.add_argument("--md-out", default="research_out/trial10_pause1_runner_elite_validation.md")
    p.add_argument("--csv-out", default="research_out/trial10_pause1_runner_elite_labeled.csv")
    return p.parse_args()


def load_trades(path: str, spread_pips: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["signal_time"] = pd.to_datetime(df["signal_time"], utc=True)
    for col in [
        "weighted_pips",
        "profit_usd",
        "hold_minutes",
        "atr_stop_pips",
        "pullback_quality_structure_ratio",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["pre_spread_pips_est"] = df["weighted_pips"] + float(spread_pips)
    df["is_win"] = df["profit_usd"] > 0
    df["is_runner"] = df["exit_reason"].astype(str).isin(RUNNER_EXIT_REASONS) | (df["hold_minutes"] >= 240)
    df["fast_stop"] = df["exit_reason"].astype(str).eq("initial_stop") & (df["hold_minutes"] <= 15)
    return df.sort_values("signal_time").reset_index(drop=True)


def load_m1(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["time", "close"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.dropna().sort_values("time").reset_index(drop=True)


def attach_freshness_inputs(trades: pd.DataFrame, m1: pd.DataFrame) -> pd.DataFrame:
    out = trades.copy()

    # M1 EMA 5/9 cross recency at each signal time.
    m1 = m1.copy()
    m1["ema5"] = m1["close"].ewm(span=5, adjust=False).mean()
    m1["ema9"] = m1["close"].ewm(span=9, adjust=False).mean()
    m1["m1_side"] = np.where(m1["ema5"] > m1["ema9"], "buy", "sell")
    last_buy_cross = None
    last_sell_cross = None
    bars_since_cross: list[Optional[int]] = []
    prev_side = None
    for i, side in enumerate(m1["m1_side"]):
        if prev_side is not None and side != prev_side:
            if side == "buy":
                last_buy_cross = i
            else:
                last_sell_cross = i
        if side == "buy":
            bars_since_cross.append(i - last_buy_cross if last_buy_cross is not None else np.nan)
        else:
            bars_since_cross.append(i - last_sell_cross if last_sell_cross is not None else np.nan)
        prev_side = side
    m1["bars_since_m1_cross"] = bars_since_cross
    m1_map = m1[["time", "m1_side", "bars_since_m1_cross"]].rename(columns={"time": "signal_time"})
    out = out.merge(m1_map, on="signal_time", how="left")
    out.loc[out["m1_side"] != out["side"], "bars_since_m1_cross"] = np.nan

    # M5 9/21 recross state aligned to signal time.
    m5 = (
        m1[["time", "close"]]
        .set_index("time")
        .resample("5min", label="right", closed="right")
        .agg({"close": "last"})
        .dropna()
        .reset_index()
    )
    m5["ema9"] = m5["close"].ewm(span=9, adjust=False).mean()
    m5["ema21"] = m5["close"].ewm(span=21, adjust=False).mean()
    m5["m5_side"] = np.where(m5["ema9"] > m5["ema21"], "buy", "sell")
    last_buy_recross = pd.NaT
    last_sell_recross = pd.NaT
    recross_times: list[pd.Timestamp] = []
    prev_side = None
    for _, row in m5.iterrows():
        side = row["m5_side"]
        ts = row["time"]
        if prev_side is not None and side != prev_side:
            if side == "buy":
                last_buy_recross = ts
            else:
                last_sell_recross = ts
        recross_times.append(last_buy_recross if side == "buy" else last_sell_recross)
        prev_side = side
    m5["last_m5_recross_time"] = recross_times
    m5_map = m5[["time", "m5_side", "last_m5_recross_time"]].rename(columns={"time": "m5_bar_time"})
    out = pd.merge_asof(
        out.sort_values("signal_time"),
        m5_map.sort_values("m5_bar_time"),
        left_on="signal_time",
        right_on="m5_bar_time",
        direction="backward",
    )
    out.loc[out["m5_side"] != out["side"], "last_m5_recross_time"] = pd.NaT

    # Count prior same-side entries since the last M5 recross, without future leakage.
    out = out.sort_values(["side", "signal_time"]).copy()
    out["signal_ns"] = out["signal_time"].astype("int64")
    out["cross_ns"] = out["last_m5_recross_time"].astype("int64")
    nat_int = pd.NaT.value

    def _prior_entries_since_cross(group: pd.DataFrame) -> pd.Series:
        sig = group["signal_ns"].to_numpy()
        cross = group["cross_ns"].to_numpy()
        result = np.zeros(len(group), dtype=int)
        j = 0
        for i in range(len(group)):
            ct = cross[i]
            if ct == nat_int:
                result[i] = 0
                continue
            while j < i and sig[j] < ct:
                j += 1
            result[i] = i - j
        return pd.Series(result, index=group.index)

    out["prior_entries"] = out.groupby("side", group_keys=False).apply(_prior_entries_since_cross)
    return out.sort_values("signal_time").reset_index(drop=True)


def assign_runner_buckets(df: pd.DataFrame, freshness_mode: str) -> pd.DataFrame:
    rows = []
    for row in df.itertuples(index=False):
        res = compute_runner_score(
            atr_stop_pips=float(row.atr_stop_pips or 0.0),
            regime_label=str(row.regime_label or ""),
            m5_bucket=str(row.m5_bucket or ""),
            structure_ratio=float(row.pullback_quality_structure_ratio) if pd.notna(row.pullback_quality_structure_ratio) else None,
            bars_since_cross=int(row.bars_since_m1_cross) if pd.notna(row.bars_since_m1_cross) else None,
            prior_entries=int(row.prior_entries) if pd.notna(row.prior_entries) else None,
            freshness_mode=freshness_mode,
        )
        rows.append(asdict(res))
    res_df = pd.DataFrame(rows)
    renamed = res_df.add_prefix(f"{freshness_mode}_")
    return pd.concat([df.reset_index(drop=True), renamed], axis=1)


def build_time_blocks(df: pd.DataFrame) -> pd.Series:
    start = df["signal_time"].min()
    end = df["signal_time"].max()
    span = end - start
    t1 = start + span / 3
    t2 = start + (span * 2) / 3

    def _label(ts: pd.Timestamp) -> str:
        if ts < t1:
            return "early"
        if ts < t2:
            return "middle"
        return "late"

    return df["signal_time"].map(_label)


def bucket_metrics(df: pd.DataFrame, bucket_col: str) -> dict[str, dict]:
    order = ["floor", "base", "elevated", "press", "elite"]
    out: dict[str, dict] = {}
    for bucket in order:
        g = df[df[bucket_col] == bucket]
        if g.empty:
            out[bucket] = {
                "n": 0,
                "win_rate_pct": None,
                "avg_pips_after": None,
                "avg_pips_before": None,
                "runner_rate_pct": None,
                "fast_stop_pct": None,
                "total_usd": None,
            }
            continue
        out[bucket] = {
            "n": int(len(g)),
            "win_rate_pct": round(float(g["is_win"].mean() * 100), 2),
            "avg_pips_after": round(float(g["weighted_pips"].mean()), 4),
            "avg_pips_before": round(float(g["pre_spread_pips_est"].mean()), 4),
            "runner_rate_pct": round(float(g["is_runner"].mean() * 100), 2),
            "fast_stop_pct": round(float(g["fast_stop"].mean() * 100), 2),
            "total_usd": round(float(g["profit_usd"].sum()), 4),
        }
    return out


def elite_verdict(full_metrics: dict[str, dict], block_metrics: dict[str, dict]) -> dict[str, object]:
    elite = full_metrics["elite"]
    press = full_metrics["press"]
    elite_avg = elite["avg_pips_after"] if elite["avg_pips_after"] is not None else -999.0
    press_avg = press["avg_pips_after"] if press["avg_pips_after"] is not None else -999.0
    strongest_each_block = True
    near_breakeven_blocks = 0
    degraded_blocks: list[str] = []
    for block_name, metrics in block_metrics.items():
        ranked = []
        for bucket_name, vals in metrics.items():
            if vals["avg_pips_after"] is not None:
                ranked.append((vals["avg_pips_after"], bucket_name))
        if ranked:
            ranked.sort(reverse=True)
            if ranked[0][1] != "elite":
                strongest_each_block = False
                degraded_blocks.append(block_name)
        elite_block = metrics["elite"]["avg_pips_after"]
        if elite_block is not None and elite_block >= -0.25:
            near_breakeven_blocks += 1

    promising = elite_avg > press_avg and strongest_each_block
    future_sizeable = promising and near_breakeven_blocks >= 2 and elite_avg >= -0.25
    return {
        "promising": promising,
        "future_sizeable": future_sizeable,
        "strongest_in_all_blocks": strongest_each_block,
        "near_breakeven_blocks": near_breakeven_blocks,
        "degraded_blocks": degraded_blocks,
    }


def render_markdown(results: dict) -> str:
    lines = [
        "# Trial 10 Runner Elite Validation",
        "",
        f"Baseline: `{results['baseline']['trades_path']}`",
        f"M1 source: `{results['baseline']['m1_path']}`",
        f"Spread assumption: `{results['baseline']['spread_pips']} pips`",
        "",
    ]
    for mode in ("strict", "relaxed"):
        section = results["modes"][mode]
        lines.append(f"## {mode.title()} Freshness")
        lines.append("")
        lines.append(
            f"Verdict: promising=`{section['verdict']['promising']}` | "
            f"future_sizeable=`{section['verdict']['future_sizeable']}` | "
            f"strongest_in_all_blocks=`{section['verdict']['strongest_in_all_blocks']}` | "
            f"near_breakeven_blocks=`{section['verdict']['near_breakeven_blocks']}`"
        )
        lines.append("")
        lines.append("| Bucket | N | WR% | Avg Pips After | Avg Pips Before | Runner % | Fast Stop % | Total USD |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for bucket in ("floor", "base", "elevated", "press", "elite"):
            row = section["full_sample"][bucket]
            lines.append(
                f"| {bucket} | {row['n']} | {row['win_rate_pct']} | {row['avg_pips_after']} | "
                f"{row['avg_pips_before']} | {row['runner_rate_pct']} | {row['fast_stop_pct']} | {row['total_usd']} |"
            )
        lines.append("")
        for block_name in ("early", "middle", "late"):
            lines.append(f"### {block_name.title()} Block")
            lines.append("")
            lines.append("| Bucket | N | Avg Pips After | Avg Pips Before | Runner % | Fast Stop % |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for bucket in ("floor", "base", "elevated", "press", "elite"):
                row = section["blocks"][block_name][bucket]
                lines.append(
                    f"| {bucket} | {row['n']} | {row['avg_pips_after']} | {row['avg_pips_before']} | "
                    f"{row['runner_rate_pct']} | {row['fast_stop_pct']} |"
                )
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    args = parse_args()
    trades = load_trades(args.trades, args.spread_pips)
    m1 = load_m1(args.m1)
    labeled = attach_freshness_inputs(trades, m1)
    labeled["wf_block"] = build_time_blocks(labeled)
    labeled = assign_runner_buckets(labeled, "strict")
    labeled = assign_runner_buckets(labeled, "relaxed")

    results = {
        "baseline": {
            "trades_path": args.trades,
            "m1_path": args.m1,
            "spread_pips": args.spread_pips,
            "trade_count": int(len(labeled)),
            "period_start": labeled["signal_time"].min().isoformat(),
            "period_end": labeled["signal_time"].max().isoformat(),
        },
        "modes": {},
    }

    for mode in ("strict", "relaxed"):
        bucket_col = f"{mode}_bucket"
        full = bucket_metrics(labeled, bucket_col)
        blocks = {
            block: bucket_metrics(labeled[labeled["wf_block"] == block], bucket_col)
            for block in ("early", "middle", "late")
        }
        results["modes"][mode] = {
            "full_sample": full,
            "blocks": blocks,
            "verdict": elite_verdict(full, blocks),
        }

    labeled_csv = labeled[
        [
            "trade_id",
            "signal_time",
            "side",
            "entry_type",
            "tier_period",
            "weighted_pips",
            "pre_spread_pips_est",
            "profit_usd",
            "exit_reason",
            "hold_minutes",
            "atr_stop_pips",
            "regime_label",
            "m5_bucket",
            "pullback_quality_structure_ratio",
            "bars_since_m1_cross",
            "prior_entries",
            "wf_block",
            "strict_bucket",
            "strict_points",
            "strict_fresh",
            "relaxed_bucket",
            "relaxed_points",
            "relaxed_fresh",
        ]
    ].copy()

    Path(args.json_out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    Path(args.md_out).write_text(render_markdown(results), encoding="utf-8")
    labeled_csv.to_csv(args.csv_out, index=False)

    print(f"Wrote {args.json_out}")
    print(f"Wrote {args.md_out}")
    print(f"Wrote {args.csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
