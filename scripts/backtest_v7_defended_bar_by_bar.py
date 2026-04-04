#!/usr/bin/env python3
"""CLI wrapper for Phase3 v7_pfdd defended bar-by-bar backtest (implementation in core)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.phase3_v7_pfdd_defended_runner import parse_args, run

if __name__ == "__main__":
    raise SystemExit(run())
