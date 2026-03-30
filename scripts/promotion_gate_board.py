#!/usr/bin/env python3
"""
Weekly cross-agent promotion board — summarizes artifact presence and gate reminders.

Reads optional JSON artifacts if present under research_out/. Prints a checklist
for defensive / offensive / portability promotion decisions.

Usage:
  python3 scripts/promotion_gate_board.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "research_out"


def _exists(name: str) -> bool:
    return (OUT / name).exists()


def _brief_json(path: Path, keys: list[str]) -> dict[str, Any]:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        out: dict[str, Any] = {"path": str(path)}
        for k in keys:
            if k in d:
                out[k] = d[k]
        return out
    except Exception as e:
        return {"path": str(path), "error": str(e)}


def main() -> int:
    print("=== Promotion gate board (hybrid regime-ownership) ===\n")

    print("Artifacts:")
    artifacts = [
        "diagnostic_strategy_ownership.json",
        "diagnostic_ownership_stability.json",
        "diagnostic_chart_first_routing.json",
        "phaseb_v44_to_v14_narrow.json",
        "defensive_ownership_v15_candidates.json",
        "offensive_shadow_ledger.json",
        "portability_program_gates.json",
    ]
    for a in artifacts:
        print(f"  [{'x' if _exists(a) else ' '}] {a}")

    print("\nDefensive promotion gate:")
    print("  - Stable on BOTH 500k and 1000k")
    print("  - Clear reduction in bad trades / improved PF or DD")
    print("  - Live parity: regime cell matches research (phase3_ownership_audit in minute diagnostics)")
    if _exists("defensive_ownership_v15_candidates.json"):
        print("  - Candidates file present; review cells before editing phase3 vetoes")

    print("\nOffensive promotion gate:")
    print("  - Phase A shadow shows disagreement signal; Phase B narrow test passes real-entry bar")
    print("  - Never promote from coverage-gap bar counts alone")
    if _exists("diagnostic_chart_first_routing.json"):
        s = _brief_json(OUT / "diagnostic_chart_first_routing.json", ["dataset_summaries", "verdict"])
        print(f"  - Phase A snapshot: {s}")

    print("\nPortability promotion gate:")
    print("  - Archetype works in non-native context without session crutches")
    print("  - See research_out/portability_program_gates.json (run portability_program_gates.py)")

    print("\nAgent roles:")
    print("  Agent1: defensive v1.5 + live observability")
    print("  Agent2: offensive shadow + narrow experiments")
    print("  Agent3: portability harness + archetype specs")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
