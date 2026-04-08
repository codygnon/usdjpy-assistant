#!/usr/bin/env python3
"""Print side-by-side PnL / trade counts from two combined_trade_log.csv files.

Example:
  python3 scripts/compare_combined_trade_logs.py \\
    research_out/.../combined_trade_log.csv \\
    research_out/.../no_spike/combined_trade_log.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def _summarize(path: Path, label: str) -> None:
    if not path.is_file():
        print(f"{label}: missing {path}", file=sys.stderr)
        return
    df = pd.read_csv(path)
    if df.empty:
        print(f"{label}: 0 rows")
        return
    pnl = pd.to_numeric(df.get("pnl_usd"), errors="coerce").dropna()
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    pf = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf") if wins.sum() > 0 else 0.0
    print(f"=== {label} ({path.name}) ===")
    print(f"  rows: {len(df)} | closed PnL rows: {len(pnl)} | net ${pnl.sum():,.2f} | PF {pf:.2f}")
    if "strategy" in df.columns:
        for strat, g in df.groupby("strategy"):
            p = pd.to_numeric(g["pnl_usd"], errors="coerce").dropna()
            print(f"  - {strat}: n={len(g)} net=${p.sum():,.2f}")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description="Compare two combined trade logs")
    p.add_argument("path_a", type=Path, help="First combined_trade_log.csv")
    p.add_argument("path_b", type=Path, help="Second combined_trade_log.csv")
    p.add_argument("--label-a", default="A", help="Label for first file")
    p.add_argument("--label-b", default="B", help="Label for second file")
    args = p.parse_args()
    _summarize(args.path_a, args.label_a)
    _summarize(args.path_b, args.label_b)


if __name__ == "__main__":
    main()
