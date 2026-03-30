#!/usr/bin/env python3
"""
Exhaust single-pocket defensive blocks from defensive_ownership_v15_candidates.json.

Runs backtest_defensive_v15_pocket_grid for each `exhaust_candidates` row (500k+1000k),
writes one JSON per candidate under research_out/pockets_defensive_v15/ and a master
index with strict PF+USD pass/fail (both datasets: Δnet_usd>0 and ΔPF>0).

Optional: --pairwise-strict re-tests the union of all strict-pass single pockets
(small M only; intended when exhaust set is tiny).
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

from scripts.backtest_defensive_v15_pocket_grid import (
    _json_default,
    cell_slug,
    run_multi_pocket_datasets,
    run_pocket_datasets,
    strict_pass_both,
)

OUT_DIR = ROOT / "research_out"
CANDIDATES_PATH = OUT_DIR / "defensive_ownership_v15_candidates.json"
DEFAULT_POCKET_DIR = OUT_DIR / "pockets_defensive_v15"
INDEX_PATH = OUT_DIR / "defensive_v15_strict_pf_usd_index.json"


def _rel_to_root(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(p)


def _index_row(
    strategy: str,
    cell: str,
    artifact: str | None,
    full: dict[str, Any],
) -> dict[str, Any]:
    d500 = (full.get("datasets") or {}).get("500k") or {}
    d1000 = (full.get("datasets") or {}).get("1000k") or {}
    def _pack(block: dict[str, Any]) -> dict[str, Any]:
        if not block:
            return {}
        sp = block.get("single_pocket_block") or {}
        dv = block.get("delta_vs_variant_k") or {}
        return {
            "blocked_count": sp.get("blocked_count"),
            "blocked_winners": sp.get("blocked_winners"),
            "blocked_losers": sp.get("blocked_losers"),
            "blocked_net_usd": sp.get("blocked_net_usd"),
            "delta_net_usd": dv.get("net_usd"),
            "delta_profit_factor": dv.get("profit_factor"),
            "delta_max_drawdown_usd": dv.get("max_drawdown_usd"),
            "delta_total_trades": dv.get("total_trades"),
        }

    row = {
        "strategy": strategy,
        "cell": cell,
        "artifact_relpath": artifact,
        "500k": _pack(d500),
        "1000k": _pack(d1000),
        "passes_strict_pf_usd": strict_pass_both(full),
    }
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description="Exhaust defensive v1.5 pocket grid vs Variant K.")
    ap.add_argument(
        "--candidates",
        type=Path,
        default=CANDIDATES_PATH,
        help="Path to defensive_ownership_v15_candidates.json",
    )
    ap.add_argument(
        "--pocket-dir",
        type=Path,
        default=DEFAULT_POCKET_DIR,
        help="Directory for per-candidate JSON artifacts",
    )
    ap.add_argument(
        "--index",
        type=Path,
        default=INDEX_PATH,
        help="Master index output path",
    )
    ap.add_argument(
        "--pairwise-strict",
        action="store_true",
        help="If multiple rows pass strict, also run one combined multi-pocket backtest",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print candidates only, no backtests",
    )
    args = ap.parse_args()

    if not args.candidates.exists():
        print(f"Missing {args.candidates}", file=sys.stderr)
        return 1
    raw = json.loads(args.candidates.read_text(encoding="utf-8"))
    candidates: list[dict[str, Any]] = list(raw.get("exhaust_candidates") or [])
    if not candidates:
        print("No exhaust_candidates in JSON; check mining filters.", file=sys.stderr)
        return 1

    args.pocket_dir.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, Any]] = []

    if args.dry_run:
        for row in candidates:
            print(f"{row.get('strategy')} {row.get('cell')}")
        return 0

    for row in candidates:
        strategy = str(row["strategy"])
        cell = str(row["cell"])
        slug = f"{strategy}_{cell_slug(cell)}"
        out_name = f"defensive_v15_pocket_{slug}.json"
        out_path = args.pocket_dir / out_name
        print(f"Running {strategy} @ {cell} ...")
        full = run_pocket_datasets(strategy, cell)
        out_path.write_text(json.dumps(full, indent=2, default=_json_default), encoding="utf-8")
        rel = _rel_to_root(out_path)
        index_rows.append(_index_row(strategy, cell, rel, full))
        print(f"  -> {rel} strict_pass={index_rows[-1]['passes_strict_pf_usd']}")

    strict_passers: list[tuple[str, str]] = []
    for r in index_rows:
        if r.get("passes_strict_pf_usd"):
            strict_passers.append((str(r["strategy"]), str(r["cell"])))

    multi_block: dict[str, Any] | None = None
    if args.pairwise_strict and len(strict_passers) >= 2:
        pockets = strict_passers
        print(f"Running multi-pocket union ({len(pockets)} rules) ...")
        multi_block = run_multi_pocket_datasets(pockets)
        multi_path = args.pocket_dir / "defensive_v15_pocket_multi_strict_union.json"
        multi_path.write_text(json.dumps(multi_block, indent=2, default=_json_default), encoding="utf-8")
        print(f"  -> {multi_path.relative_to(ROOT)} strict_pass={strict_pass_both(multi_block)}")

    summary = {
        "purpose": "Strict pass: delta_vs_variant_k net_usd > 0 AND profit_factor > 0 on both 500k and 1000k",
        "candidates_path": _rel_to_root(args.candidates),
        "pocket_dir": _rel_to_root(args.pocket_dir),
        "single_pocket_rows": index_rows,
        "strict_pass_count": sum(1 for r in index_rows if r.get("passes_strict_pf_usd")),
        "strict_passers": [{"strategy": s, "cell": c} for s, c in strict_passers],
        "multi_pocket_strict_union": (
            {
                "artifact": _rel_to_root(args.pocket_dir / "defensive_v15_pocket_multi_strict_union.json"),
                "passes_strict_pf_usd": strict_pass_both(multi_block) if multi_block else None,
            }
            if multi_block is not None
            else None
        ),
    }

    args.index.parent.mkdir(parents=True, exist_ok=True)
    args.index.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote index {args.index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
