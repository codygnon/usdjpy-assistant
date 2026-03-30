#!/usr/bin/env python3
"""
Agent 3 / Track 3: Archetype portability program — gate definitions and scaffold.

This module documents success/failure criteria for parallel portability work.
Executable as a script to print the gate matrix (no backtests here).

Sub-tracks:
  3A Momentum (V44-style) portability
  3B Generalized range-breakout archetype (London successor)
  3C Mean reversion (V14 validate vs rebuild decision)

Usage:
  python3 scripts/portability_program_gates.py
  python3 -c "from scripts.portability_program_gates import GATES; print(GATES)"
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

GATES: dict[str, Any] = {
    "3A_momentum_portability": {
        "goal": "V44-style momentum logic outside native NY window in validated chart states only.",
        "success": [
            "Net improvement on both 500k and 1000k OR strong 1000k with no material harm on 500k.",
            "Max DD does not increase materially vs promoted Variant K baseline.",
            "Effect from real fills, not authorization-only hypotheticals.",
        ],
        "failure": [
            "No/few fills in pilot contexts (signal null).",
            "Off-session momentum underperforms same regime cell in-session.",
        ],
        "code_touchpoints": [
            "scripts/backtest_session_momentum.py",
            "scripts/backtest_merged_integrated_tokyo_london_v2_ny.py",
        ],
    },
    "3B_generalized_range_archetype": {
        "goal": "Range-breakout archetype decoupled from fixed Asian/London UTC anchors.",
        "success": [
            "Defined range formation + breakout trigger reproducible on historical CSVs.",
            "Coupled backtest shows edge vs naive session-only London clone in target regimes.",
        ],
        "failure": [
            "Range detector fires too often (overtrading) or too rarely (no unlock).",
            "Cannot match or beat defensive-only baseline after costs.",
        ],
        "code_touchpoints": [
            "scripts/backtest_v2_multisetup_london.py",
        ],
    },
    "3C_mean_reversion_decision": {
        "goal": "Either validate Tokyo V14 as portable MR or freeze as session-native and plan new MR archetype.",
        "success": [
            "Written decision: VALIDATE_V14_PORTABLE | KEEP_TOKYO_NATIVE_ONLY.",
            "If validate: cross-session pilot passes same gate style as 3A.",
        ],
        "failure": [
            "Forcing V14 into non-Tokyo without independent edge proof.",
        ],
        "code_touchpoints": [
            "scripts/backtest_tokyo_meanrev.py",
        ],
    },
}


def main() -> int:
    out = ROOT / "research_out" / "portability_program_gates.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(GATES, indent=2), encoding="utf-8")
    print(json.dumps(GATES, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
