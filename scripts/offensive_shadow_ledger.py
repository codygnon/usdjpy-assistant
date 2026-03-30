#!/usr/bin/env python3
"""
Agent 2 / Track 2: Offensive shadow — portable vs non-portable disagreement ledger.

Reads research_out/diagnostic_chart_first_routing.json (from Phase A) and splits
disagreement_pair_breakdown rows into:
  - portable_candidates (momentum / V44 <-> V14 class)
  - non_portable_london_table (any flow involving london_v2 as table anchor)
  - other (manual review)

Does not claim edge; does not gate Phase B. Research coordination artifact only.

Usage:
  python3 scripts/offensive_shadow_ledger.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "research_out"
PHASE_A_PATH = OUT_DIR / "diagnostic_chart_first_routing.json"
LEDGER_OUT = OUT_DIR / "offensive_shadow_ledger.json"


def _classify_pair(pair: str) -> str:
    p = pair.replace(" ", "")
    if "london_v2" in p:
        return "non_portable_london_table"
    if p in {"v44_ny->v14", "v14->v44_ny"}:
        return "portable_candidate"
    if "v44_ny" in p and "v14" in p:
        return "portable_candidate"
    return "other_review"


def main() -> int:
    if not PHASE_A_PATH.exists():
        print(f"Missing {PHASE_A_PATH}; run diagnostic_chart_first_routing.py first.", file=sys.stderr)
        return 1

    phase_a = json.loads(PHASE_A_PATH.read_text(encoding="utf-8"))
    breakdown = phase_a.get("disagreement_pair_breakdown") or {}

    ledger: dict[str, Any] = {
        "source": str(PHASE_A_PATH),
        "note": "Exploratory ledger only. Not a success gate. Does not prove cross-session edge.",
        "by_dataset": {},
    }

    for dk, rows in breakdown.items():
        if not isinstance(rows, list):
            continue
        buckets: dict[str, list[dict[str, Any]]] = {
            "portable_candidates": [],
            "non_portable_london_table": [],
            "other_review": [],
        }
        for row in rows:
            pair = str(row.get("pair", ""))
            bucket = _classify_pair(pair)
            if bucket == "non_portable_london_table":
                buckets["non_portable_london_table"].append(row)
            elif bucket == "portable_candidate":
                buckets["portable_candidates"].append(row)
            else:
                buckets["other_review"].append(row)
        ledger["by_dataset"][dk] = buckets

    LEDGER_OUT.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    print(f"Wrote {LEDGER_OUT}")
    for dk, b in ledger["by_dataset"].items():
        print(
            f"  {dk}: portable={len(b['portable_candidates'])} "
            f"non_portable_london={len(b['non_portable_london_table'])} "
            f"other={len(b['other_review'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
