#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path('/Users/codygnon/Documents/usdjpy_assistant')
SCRIPTS = ROOT / 'scripts'
OUT_DIR = ROOT / 'research_out'


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Run the finish-line package freeze research program end to end')
    ap.add_argument('--skip-package-frontier', action='store_true')
    ap.add_argument('--skip-exit-frontier', action='store_true')
    ap.add_argument('--skip-pruning-frontier', action='store_true')
    ap.add_argument('--skip-time-frontier', action='store_true')
    ap.add_argument('--skip-closeout-memo', action='store_true')
    ap.add_argument('--package-frontier-args', default='')
    ap.add_argument('--exit-frontier-args', default='')
    ap.add_argument('--pruning-frontier-args', default='')
    ap.add_argument('--time-frontier-args', default='')
    return ap.parse_args()


def _run(script: str, extra: str) -> None:
    cmd = [sys.executable, str(SCRIPTS / script)]
    if extra.strip():
        cmd.extend(extra.strip().split())
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> int:
    args = parse_args()
    if not args.skip_package_frontier:
        _run('run_package_freeze_frontier.py', args.package_frontier_args)
    if not args.skip_exit_frontier:
        _run('run_package_slice_exit_frontier.py', args.exit_frontier_args)
    if not args.skip_pruning_frontier:
        _run('run_package_pruning_frontier.py', args.pruning_frontier_args)
    if not args.skip_time_frontier:
        _run('run_package_time_regime_frontier.py', args.time_frontier_args)
    if not args.skip_closeout_memo:
        _run('build_package_freeze_closeout_memo.py', '')
    print('\nFinish-line research program complete.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
