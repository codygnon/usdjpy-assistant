#!/usr/bin/env python3
"""Run a tight V4 lot-size ladder on the real combined bar-by-bar margin engine."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "scripts/run_v71_h1_v4_margin_bar_by_bar.py"
BASE_OUT = ROOT / "research_out/trade_analysis/spike_fade_v4/backtest/combined_v71_h1_v4_margin_bar_by_bar"
LADDER_OUT = ROOT / "research_out/trade_analysis/spike_fade_v4/backtest/combined_v71_h1_v4_margin_ladder"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run V7.1 + H1 + V4 real bar-by-bar lot-size ladder")
    p.add_argument("--lots", nargs="+", type=float, default=[10, 15, 18, 20])
    p.add_argument("--entry-margin-cap-pcts", nargs="+", type=float, default=[100.0])
    p.add_argument("--starting-equity", type=float, default=100000.0)
    p.add_argument("--leverage", type=float, default=33.0)
    p.add_argument("--data-path", default=str(ROOT / "research_out/USDJPY_M1_OANDA_extended.csv"))
    p.add_argument("--quiet-engine", action="store_true")
    return p.parse_args()


def read_metrics(run_dir: Path) -> dict[str, float]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    equity = pd.read_csv(run_dir / "equity_log.csv")
    trades = pd.read_csv(run_dir / "combined_trade_log.csv")
    v4 = pd.read_csv(run_dir / "v4_trade_log.csv")

    pnls = pd.to_numeric(trades["pnl_usd"], errors="coerce").dropna()
    v4_pnls = pd.to_numeric(v4["pnl_usd"], errors="coerce").dropna() if not v4.empty else pd.Series(dtype=float)
    max_dd_pct = float(equity["drawdown_pct_nav"].max()) if not equity.empty else 0.0
    peak_margin = float(equity["margin_used"].max()) if not equity.empty else 0.0
    finite_margin = equity["margin_level_pct"].replace([float("inf")], pd.NA).dropna()
    min_margin_level = float(finite_margin.min()) if not finite_margin.empty else float("inf")
    ending_nav = float(summary.get("ending_nav", summary.get("ending_balance", 0.0)))
    combined_net = ending_nav - float(summary.get("starting_equity", 0.0))

    def pf(series: pd.Series) -> float:
        wins = series[series > 0].sum()
        losses = abs(series[series < 0].sum())
        if losses == 0:
            return float("inf") if wins > 0 else 0.0
        return float(wins / losses)

    return {
        "entry_margin_cap_pct": float(summary.get("entry_margin_cap_pct", 100.0)),
        "combined_trades": int(len(trades)),
        "combined_net": combined_net,
        "combined_pf": pf(pnls),
        "v4_trades": int(len(v4)),
        "v4_net": float(v4_pnls.sum()) if not v4_pnls.empty else 0.0,
        "v4_pf": pf(v4_pnls),
        "max_dd_pct_nav": max_dd_pct,
        "peak_margin_used": peak_margin,
        "lowest_margin_level_pct": min_margin_level,
        "margin_calls": int(summary.get("margin_call_count", 0)),
        "v4_blocked_margin": int(summary.get("v4_state", {}).get("blocked_margin", 0)),
        "h1_v44_blocked": int(summary.get("h1_filter_stats", {}).get("h1_v44_blocked", 0)),
        "margin_v44_blocks": int(summary.get("blocked_counts", {}).get("margin_v44", 0)),
        "max_concurrent_positions": int(summary.get("max_concurrent_positions", 0)),
        "elapsed_seconds": float(summary.get("elapsed_seconds", 0.0)),
    }


def write_summary(rows: list[dict[str, float]]) -> None:
    LADDER_OUT.mkdir(parents=True, exist_ok=True)
    csv_path = LADDER_OUT / "ladder_results.csv"
    txt_path = LADDER_OUT / "ladder_summary.txt"

    fieldnames = [
        "label",
        "v4_lots",
        "entry_margin_cap_pct",
        "combined_trades",
        "combined_net",
        "combined_pf",
        "v4_trades",
        "v4_net",
        "v4_pf",
        "max_dd_pct_nav",
        "peak_margin_used",
        "lowest_margin_level_pct",
        "margin_calls",
        "v4_blocked_margin",
        "margin_v44_blocks",
        "max_concurrent_positions",
        "elapsed_seconds",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    lines = []
    lines.append("V7.1 + H1 + V4 REAL BAR-BY-BAR LOT LADDER")
    lines.append("=========================================")
    lines.append("")
    lines.append(
        f"{'Lot':>6} {'Cap%':>6} {'Net':>14} {'PF':>7} {'V4Net':>14} {'V4PF':>7} {'MaxDD%':>8} {'PeakMargin':>14} {'MinLevel%':>10} {'Calls':>6} {'V44Blk':>7}"
    )
    lines.append(
        f"{'-'*6} {'-'*6} {'-'*14} {'-'*7} {'-'*14} {'-'*7} {'-'*8} {'-'*14} {'-'*10} {'-'*6} {'-'*7}"
    )
    for row in rows:
        lines.append(
            f"{row['v4_lots']:6.1f} "
            f"{row['entry_margin_cap_pct']:6.0f} "
            f"{row['combined_net']:14,.2f} "
            f"{row['combined_pf']:7.2f} "
            f"{row['v4_net']:14,.2f} "
            f"{row['v4_pf']:7.2f} "
            f"{row['max_dd_pct_nav']:8.2f} "
            f"{row['peak_margin_used']:14,.2f} "
            f"{row['lowest_margin_level_pct']:10.2f} "
            f"{int(row['margin_calls']):6d} "
            f"{int(row['margin_v44_blocks']):7d}"
        )

    safe = [
        row for row in rows
        if int(row["margin_calls"]) == 0
        and int(row["v4_blocked_margin"]) == 0
        and int(row["margin_v44_blocks"]) == 0
        and float(row["lowest_margin_level_pct"]) >= 120.0
    ]
    lines.append("")
    if safe:
        best_safe = max(safe, key=lambda r: float(r["v4_lots"]))
        lines.append(
            f"Best clean point by current rules: {best_safe['v4_lots']:.1f} lots at "
            f"{best_safe['entry_margin_cap_pct']:.0f}% cap "
            f"(min margin level {best_safe['lowest_margin_level_pct']:.2f}%, no calls, no V44 margin blocks)"
        )
    else:
        lines.append("No ladder point met the stricter clean threshold (0 calls, 0 margin blocks, min level >= 120%).")

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    LADDER_OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float]] = []

    for cap_pct in args.entry_margin_cap_pcts:
        for lots in args.lots:
            label = f"v4_{str(lots).replace('.', 'p')}lot_cap{str(cap_pct).replace('.', 'p')}"
            run_dir = LADDER_OUT / label
            print(f"\n=== Running {label} ===", flush=True)
            cmd = [
                sys.executable,
                str(ENGINE),
                "--data-path",
                str(args.data_path),
                "--starting-equity",
                str(args.starting_equity),
                "--leverage",
                str(args.leverage),
                "--v4-lots",
                str(lots),
                "--entry-margin-cap-pct",
                str(cap_pct),
            ]
            if args.quiet_engine:
                cmd.append("--quiet")
            t0 = time.time()
            result = subprocess.run(cmd, cwd=str(ROOT))
            if result.returncode != 0:
                raise SystemExit(result.returncode)
            elapsed = time.time() - t0

            if run_dir.exists():
                shutil.rmtree(run_dir)
            shutil.copytree(BASE_OUT, run_dir)

            metrics = read_metrics(run_dir)
            metrics["label"] = label
            metrics["v4_lots"] = float(lots)
            metrics["entry_margin_cap_pct"] = float(cap_pct)
            metrics["elapsed_seconds"] = elapsed
            rows.append(metrics)
            print(
                f"  done {label}: net=${metrics['combined_net']:,.2f} PF={metrics['combined_pf']:.2f} "
                f"V4=${metrics['v4_net']:,.2f} min_margin_level={metrics['lowest_margin_level_pct']:.2f}% "
                f"calls={int(metrics['margin_calls'])} v44_margin_blocks={int(metrics['margin_v44_blocks'])}",
                flush=True,
            )

    write_summary(rows)
    print(f"\nWrote {LADDER_OUT / 'ladder_results.csv'}")
    print(f"Wrote {LADDER_OUT / 'ladder_summary.txt'}")


if __name__ == "__main__":
    main()
