#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
python3 -u scripts/backtest_v7_defended_bar_by_bar.py --dataset 1000k --spread-mode pipeline 2>&1 | tee research_out/bbb_1000k_pipeline.log
python3 -u scripts/backtest_v7_defended_bar_by_bar.py --dataset 1000k --spread-mode realistic 2>&1 | tee research_out/bbb_1000k_realistic.log
echo "DONE both runs" | tee research_out/bbb_1000k_both_done.marker
