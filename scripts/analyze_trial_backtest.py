#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PIP_SIZE = 0.01
PIP_VALUE_UNITS = 100_000.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Advanced analytics for Trial backtest trade CSVs.")
    p.add_argument("--trades", required=True, help="Backtest trades CSV path")
    p.add_argument("--summary", help="Optional backtest summary JSON path")
    p.add_argument("--spread-pips", type=float, default=None, help="Override fixed spread pips")
    p.add_argument("--out-json", required=True, help="Analytics JSON output path")
    p.add_argument("--out-md", required=True, help="Analytics markdown output path")
    return p.parse_args()


def _agg(g: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {
            "trades": int(len(g)),
            "net_usd": float(g["profit_usd"].sum()),
            "avg_usd": float(g["profit_usd"].mean()) if len(g) else 0.0,
            "avg_pips": float(g["weighted_pips"].mean()) if len(g) else 0.0,
            "win_rate_pct": float((g["profit_usd"] > 0).mean() * 100.0) if len(g) else 0.0,
            "spread_usd_est": float(g["spread_cost_usd_est"].sum()),
            "net_before_spread_est": float(g["profit_before_spread_usd_est"].sum()),
        }
    )


def _table(df: pd.DataFrame, cols: list[str]) -> list[dict]:
    out: list[dict] = []
    for idx, row in df.iterrows():
        item = {"key": idx if not isinstance(idx, tuple) else " | ".join(str(x) for x in idx)}
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                item[col] = round(val, 4)
            else:
                item[col] = val
        out.append(item)
    return out


def main() -> None:
    args = parse_args()
    trades_path = Path(args.trades)
    summary_path = Path(args.summary) if args.summary else None
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)

    df = pd.read_csv(trades_path)
    if df.empty:
        raise ValueError("Trades CSV is empty")

    spread_pips = args.spread_pips
    summary = None
    if summary_path and summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if spread_pips is None:
            spread_pips = float(summary.get("spread_pips", 0.0))
    if spread_pips is None:
        raise ValueError("Provide --spread-pips or a summary JSON with spread_pips")

    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    df["entry_month"] = df["entry_time"].dt.to_period("M").astype(str)
    df["entry_hour_utc"] = df["entry_time"].dt.hour
    df["hold_bucket"] = pd.cut(
        df["hold_minutes"],
        bins=[-1, 5, 15, 60, 240, 10_000],
        labels=["0-5m", "5-15m", "15-60m", "1-4h", "4h+"],
    )

    df["spread_cost_usd_est"] = spread_pips * PIP_SIZE * PIP_VALUE_UNITS * df["initial_lots"] / df["entry_price"]
    df["profit_before_spread_usd_est"] = df["profit_usd"] + df["spread_cost_usd_est"]
    df["weighted_pips_before_spread_est"] = df["weighted_pips"] + spread_pips
    df["is_win"] = df["profit_usd"] > 0
    df["is_loss"] = df["profit_usd"] < 0

    overall = {
        "trades": int(len(df)),
        "net_usd": round(float(df["profit_usd"].sum()), 4),
        "gross_profit_usd": round(float(df.loc[df["profit_usd"] > 0, "profit_usd"].sum()), 4),
        "gross_loss_usd": round(float(df.loc[df["profit_usd"] < 0, "profit_usd"].sum()), 4),
        "win_rate_pct": round(float((df["profit_usd"] > 0).mean() * 100.0), 4),
        "avg_trade_usd": round(float(df["profit_usd"].mean()), 4),
        "avg_trade_pips": round(float(df["weighted_pips"].mean()), 4),
        "spread_usd_est": round(float(df["spread_cost_usd_est"].sum()), 4),
        "avg_spread_usd_per_trade": round(float(df["spread_cost_usd_est"].mean()), 4),
        "net_before_spread_est": round(float(df["profit_before_spread_usd_est"].sum()), 4),
    }
    if overall["gross_loss_usd"] < 0:
        overall["profit_factor"] = round(overall["gross_profit_usd"] / abs(overall["gross_loss_usd"]), 4)
    gross_profit_before = float(df.loc[df["profit_before_spread_usd_est"] > 0, "profit_before_spread_usd_est"].sum())
    gross_loss_before = float(df.loc[df["profit_before_spread_usd_est"] < 0, "profit_before_spread_usd_est"].sum())
    overall["profit_factor_before_spread_est"] = round(gross_profit_before / abs(gross_loss_before), 4) if gross_loss_before < 0 else None
    overall["spread_pct_of_net_loss"] = round((overall["spread_usd_est"] / abs(overall["net_usd"])) * 100.0, 4) if overall["net_usd"] < 0 else None

    exit_stats = df.groupby("exit_reason", dropna=False).apply(_agg).sort_values("net_usd")
    entry_stats = df.groupby("entry_type", dropna=False).apply(_agg).sort_values("net_usd")
    side_stats = df.groupby("side", dropna=False).apply(_agg).sort_values("net_usd")
    month_stats = df.groupby("entry_month", dropna=False).apply(_agg).sort_values("net_usd")
    hold_stats = df.groupby("hold_bucket", dropna=False).apply(_agg).sort_values("net_usd")
    bucket_stats = df.groupby(["m5_bucket", "m1_bucket"], dropna=False).apply(_agg).sort_values("net_usd")
    mult_stats = df.groupby("conviction_multiplier", dropna=False).apply(_agg).sort_values("net_usd")

    tier_df = df.dropna(subset=["tier_period"]).copy()
    tier_df["tier_period_int"] = tier_df["tier_period"].round().astype(int)
    tier_stats = tier_df.groupby("tier_period_int", dropna=False).apply(_agg).sort_values("net_usd")

    negative = df[df["profit_usd"] < 0].copy()
    loss_contrib_exit = negative.groupby("exit_reason")["profit_usd"].sum().sort_values()
    loss_contrib_entry = negative.groupby("entry_type")["profit_usd"].sum().sort_values()
    loss_contrib_side = negative.groupby("side")["profit_usd"].sum().sort_values()
    loss_contrib_tier = (
        negative.dropna(subset=["tier_period"])
        .assign(tier_period_int=lambda x: x["tier_period"].round().astype(int))
        .groupby("tier_period_int")["profit_usd"]
        .sum()
        .sort_values()
    )

    stop_family = df[df["exit_reason"].isin(["initial_stop", "runner_stop", "tp1_then_be_same_bar"])]
    kill_family = df[df["exit_reason"] == "kill_switch"]
    trail_family = df[df["exit_reason"] == "m5_trail_close"]
    families = {
        "stop_family": {
            "trades": int(len(stop_family)),
            "net_usd": round(float(stop_family["profit_usd"].sum()), 4),
            "avg_usd": round(float(stop_family["profit_usd"].mean()), 4),
            "spread_usd_est": round(float(stop_family["spread_cost_usd_est"].sum()), 4),
        },
        "kill_switch_family": {
            "trades": int(len(kill_family)),
            "net_usd": round(float(kill_family["profit_usd"].sum()), 4),
            "avg_usd": round(float(kill_family["profit_usd"].mean()), 4),
            "spread_usd_est": round(float(kill_family["spread_cost_usd_est"].sum()), 4),
        },
        "m5_trail_family": {
            "trades": int(len(trail_family)),
            "net_usd": round(float(trail_family["profit_usd"].sum()), 4),
            "avg_usd": round(float(trail_family["profit_usd"].mean()), 4),
            "spread_usd_est": round(float(trail_family["spread_cost_usd_est"].sum()), 4),
        },
    }

    analytics = {
        "overall": overall,
        "families": families,
        "tables": {
            "exit_stats": _table(exit_stats, ["trades", "net_usd", "avg_usd", "avg_pips", "win_rate_pct", "spread_usd_est", "net_before_spread_est"]),
            "entry_stats": _table(entry_stats, ["trades", "net_usd", "avg_usd", "avg_pips", "win_rate_pct", "spread_usd_est", "net_before_spread_est"]),
            "side_stats": _table(side_stats, ["trades", "net_usd", "avg_usd", "avg_pips", "win_rate_pct", "spread_usd_est", "net_before_spread_est"]),
            "tier_stats": _table(tier_stats, ["trades", "net_usd", "avg_usd", "avg_pips", "win_rate_pct", "spread_usd_est", "net_before_spread_est"]),
            "bucket_stats": _table(bucket_stats, ["trades", "net_usd", "avg_usd", "avg_pips", "win_rate_pct"]),
            "conviction_multiplier_stats": _table(mult_stats, ["trades", "net_usd", "avg_usd", "avg_pips", "win_rate_pct"]),
            "month_stats": _table(month_stats, ["trades", "net_usd", "avg_usd", "avg_pips", "win_rate_pct"]),
            "hold_bucket_stats": _table(hold_stats, ["trades", "net_usd", "avg_usd", "avg_pips", "win_rate_pct"]),
        },
        "loss_contribution": {
            "by_exit_reason": {str(k): round(float(v), 4) for k, v in loss_contrib_exit.items()},
            "by_entry_type": {str(k): round(float(v), 4) for k, v in loss_contrib_entry.items()},
            "by_side": {str(k): round(float(v), 4) for k, v in loss_contrib_side.items()},
            "by_tier": {str(k): round(float(v), 4) for k, v in loss_contrib_tier.items()},
        },
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(analytics, indent=2) + "\n", encoding="utf-8")

    md_lines = [
        "# Backtest Analytics",
        "",
        "## Overall",
        f"- Trades: {overall['trades']}",
        f"- Net P/L: ${overall['net_usd']:.2f}",
        f"- Spread drag estimate: ${overall['spread_usd_est']:.2f}",
        f"- Net before spread estimate: ${overall['net_before_spread_est']:.2f}",
        f"- Profit factor: {overall.get('profit_factor')}",
        f"- Profit factor before spread estimate: {overall.get('profit_factor_before_spread_est')}",
        "",
        "## Loss Contribution",
        "### By Exit Reason",
    ]
    for k, v in analytics["loss_contribution"]["by_exit_reason"].items():
        md_lines.append(f"- {k}: ${v:.2f}")
    md_lines.extend(["", "### By Entry Type"])
    for k, v in analytics["loss_contribution"]["by_entry_type"].items():
        md_lines.append(f"- {k}: ${v:.2f}")
    md_lines.extend(["", "### By Tier"])
    for k, v in analytics["loss_contribution"]["by_tier"].items():
        md_lines.append(f"- Tier {k}: ${v:.2f}")
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(json.dumps(analytics["overall"], indent=2))


if __name__ == "__main__":
    main()
