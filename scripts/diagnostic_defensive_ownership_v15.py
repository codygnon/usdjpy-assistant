#!/usr/bin/env python3
"""
Agent 1 / Track 1: Defensive Ownership v1.5 — candidate cell mining.

Reads diagnostic_strategy_ownership.json (500k + 1000k grids) and surfaces
cells where a strategy has negative avg pips on BOTH datasets, for:
  - v44_ny, london_v2, v14

Also builds `exhaust_candidates`: rows that pass min trade count per dataset
and are not in the stability `unstable` set (for automated pocket-grid runs).

Does NOT auto-promote vetoes. Human + stability review required before any
live change. Cross-check with diagnostic_ownership_stability.json for
stable_no_trade / unstable cells.

Usage:
  python3 scripts/diagnostic_defensive_ownership_v15.py [--min-count N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "research_out"
OWNERSHIP_PATH = OUT_DIR / "diagnostic_strategy_ownership.json"
STABILITY_PATH = OUT_DIR / "diagnostic_ownership_stability.json"


def _cells_negative_both(
    grid_500: dict[str, Any],
    grid_1000: dict[str, Any],
    strategy: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    all_cells = set(grid_500.keys()) | set(grid_1000.keys())
    for cell in sorted(all_cells):
        c5 = grid_500.get(cell, {}) or {}
        c1 = grid_1000.get(cell, {}) or {}
        s5 = c5.get("strategies", {}).get(strategy, {})
        s1 = c1.get("strategies", {}).get(strategy, {})
        n5 = int(s5.get("count", 0) or 0)
        n1 = int(s1.get("count", 0) or 0)
        if n5 < 1 or n1 < 1:
            continue
        a5 = float(s5.get("avg_pips", 0) or 0)
        a1 = float(s1.get("avg_pips", 0) or 0)
        if a5 < 0 and a1 < 0:
            out.append(
                {
                    "cell": cell,
                    "strategy": strategy,
                    "avg_pips_500k": round(a5, 2),
                    "avg_pips_1000k": round(a1, 2),
                    "count_500k": n5,
                    "count_1000k": n1,
                    "net_pips_500k": round(float(s5.get("net_pips", 0) or 0), 2),
                    "net_pips_1000k": round(float(s1.get("net_pips", 0) or 0), 2),
                }
            )
    out.sort(key=lambda x: (x["avg_pips_500k"] + x["avg_pips_1000k"]) / 2.0)
    return out


def _apply_exhaust_filters(
    rows: list[dict[str, Any]],
    unstable: set[str],
    min_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Returns (eligible, excluded_low_n, excluded_unstable)."""
    eligible: list[dict[str, Any]] = []
    excluded_low_n: list[dict[str, Any]] = []
    excluded_unstable: list[dict[str, Any]] = []
    for row in rows:
        cell = str(row.get("cell", ""))
        n5 = int(row.get("count_500k", 0) or 0)
        n1 = int(row.get("count_1000k", 0) or 0)
        if n5 < min_count or n1 < min_count:
            excluded_low_n.append({**row, "reason": "below_min_count_per_dataset"})
            continue
        if cell in unstable:
            excluded_unstable.append({**row, "reason": "unstable_cell"})
            continue
        eligible.append(dict(row))
    return eligible, excluded_low_n, excluded_unstable


def main() -> int:
    ap = argparse.ArgumentParser(description="Mine defensive v1.5 ownership candidates.")
    ap.add_argument(
        "--min-count",
        type=int,
        default=5,
        help="Minimum trades per dataset (500k and 1000k) for exhaust_candidates (default 5).",
    )
    args = ap.parse_args()
    min_count = max(1, int(args.min_count))

    if not OWNERSHIP_PATH.exists():
        print(f"Missing {OWNERSHIP_PATH}; run diagnostic_strategy_ownership.py first.", file=sys.stderr)
        return 1
    raw = json.loads(OWNERSHIP_PATH.read_text(encoding="utf-8"))
    g5 = raw.get("500k", {}).get("grid", {}) or {}
    g1 = raw.get("1000k", {}).get("grid", {}) or {}

    v44_bad = _cells_negative_both(g5, g1, "v44_ny")
    ldn_bad = _cells_negative_both(g5, g1, "london_v2")
    v14_bad = _cells_negative_both(g5, g1, "v14")

    stability_note = ""
    stable_nt: set[str] = set()
    unstable: set[str] = set()
    stable_same_owner: set[str] = set()
    if STABILITY_PATH.exists():
        st = json.loads(STABILITY_PATH.read_text(encoding="utf-8"))
        stable_nt = set(st.get("stable_no_trade", []) or [])
        unstable = set(st.get("unstable", []) or [])
        stable_same_owner = set(st.get("stable_same_owner", []) or [])
        stability_note = f"stable_no_trade_cells={len(stable_nt)} unstable_cells={len(unstable)}"

    v44_elig, v44_lo, v44_un = _apply_exhaust_filters(v44_bad, unstable, min_count)
    ldn_elig, ldn_lo, ldn_un = _apply_exhaust_filters(ldn_bad, unstable, min_count)
    v14_elig, v14_lo, v14_un = _apply_exhaust_filters(v14_bad, unstable, min_count)

    exhaust_candidates: list[dict[str, Any]] = []
    for row in v44_elig + ldn_elig + v14_elig:
        c = row["cell"]
        exhaust_candidates.append(
            {
                **row,
                "in_stable_same_owner": c in stable_same_owner,
            }
        )
    exhaust_candidates.sort(key=lambda x: ((x["avg_pips_500k"] + x["avg_pips_1000k"]) / 2.0, x["strategy"], x["cell"]))

    result: dict[str, Any] = {
        "purpose": "Defensive v1.5 candidate mining — negative avg pips on BOTH datasets",
        "stability_context": stability_note or "diagnostic_ownership_stability.json not found",
        "exhaust_filters": {
            "min_count_per_dataset": min_count,
            "exclude_unstable": True,
            "note": "exhaust_candidates excludes unstable cells and rows below min_count on either dataset.",
        },
        "v44_ny_negative_both_datasets": v44_bad,
        "london_v2_negative_both_datasets": ldn_bad,
        "v14_negative_both_datasets": v14_bad,
        "exhaust_candidates": exhaust_candidates,
        "exhaust_counts_by_strategy": {
            "v44_ny": len(v44_elig),
            "london_v2": len(ldn_elig),
            "v14": len(v14_elig),
        },
        "excluded_below_min_count": {
            "v44_ny": v44_lo,
            "london_v2": ldn_lo,
            "v14": v14_lo,
        },
        "excluded_unstable": {
            "v44_ny": v44_un,
            "london_v2": ldn_un,
            "v14": v14_un,
        },
        "promotion_checklist": [
            "Require min trade count threshold per dataset (e.g. n>=5 each).",
            "Cross-check cell is not unstable across datasets.",
            "Backtest a narrow Variant before live promotion.",
            "Map cell to live regime+ER+ΔER gates (see phase3_integrated_engine).",
            "Optional: prefer cells in stable_same_owner (see exhaust_candidates.in_stable_same_owner).",
        ],
    }

    out_path = OUT_DIR / "defensive_ownership_v15_candidates.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"  v44_ny negative-both cells: {len(v44_bad)} (exhaust-eligible: {len(v44_elig)})")
    print(f"  london_v2 negative-both cells: {len(ldn_bad)} (exhaust-eligible: {len(ldn_elig)})")
    print(f"  v14 negative-both cells: {len(v14_bad)} (exhaust-eligible: {len(v14_elig)})")
    print(f"  total exhaust_candidates: {len(exhaust_candidates)}")
    if v44_bad[:5]:
        print("  sample v44 worst:", v44_bad[:3])
    if ldn_bad[:5]:
        print("  sample ldn worst:", ldn_bad[:3])
    if v14_bad[:5]:
        print("  sample v14 worst:", v14_bad[:3])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
