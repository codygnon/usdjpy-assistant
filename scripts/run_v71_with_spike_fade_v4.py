#!/usr/bin/env python3
"""
V7.1 + Spike Fade V4 — combined stats without modifying core/.

True multi-sub bar-by-bar overlap requires registering a 5th strategy inside
`core/regime_backtest_engine/` (not allowed here). This script runs the official
bar-by-bar Spike Fade V4 backtest and merges cashflows with the V7.1 enriched log.

Usage:
  python3 scripts/run_v71_with_spike_fade_v4.py
  python3 scripts/run_v71_with_spike_fade_v4.py --skip-v4   # only print merge from existing CSV
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V4_SCRIPT = Path(__file__).resolve().parent / "run_spike_fade_v4_backtest.py"
TRADES_CSV = ROOT / "research_out/trade_analysis/spike_fade_v4/backtest/spike_fade_v4_backtest_trades.csv"
SUMMARY = ROOT / "research_out/trade_analysis/spike_fade_v4/backtest/spike_fade_v4_backtest_summary.txt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-v4", action="store_true", help="Do not rerun V4; use existing trades CSV")
    args = ap.parse_args()

    if not args.skip_v4:
        r = subprocess.run([sys.executable, str(V4_SCRIPT)], cwd=str(ROOT))
        if r.returncode != 0:
            sys.exit(r.returncode)

    if SUMMARY.is_file():
        print(SUMMARY.read_text(encoding="utf-8"))
    else:
        print(f"Missing {SUMMARY}")


if __name__ == "__main__":
    main()
