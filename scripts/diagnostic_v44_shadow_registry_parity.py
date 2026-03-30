#!/usr/bin/env python3
"""
Track 3 / V44 shadow: formal registry parity protocol + integrity check.

V44 shadow invocation uses ``native_trade_entry_registry`` (standalone V44 run
trade list keyed by entry_time ISO), not a per-bar callable evaluator.

This script:
  1. Documents the parity protocol (what shadow can/cannot claim).
  2. Verifies every native V44 trade has a registry entry at its entry_time.
  3. Reports registry key cardinality vs trade count.

Usage:
  python3 scripts/diagnostic_v44_shadow_registry_parity.py --dataset research_out/USDJPY_M1_OANDA_500k.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.shadow_entry_invocation import build_native_invocation_registry
from scripts import backtest_merged_integrated_tokyo_london_v2_ny as merged_engine

OUT_DIR = ROOT / "research_out"
V44_CFG = OUT_DIR / "session_momentum_v44_base_config.json"
V14_CFG = OUT_DIR / "tokyo_optimized_v14_config.json"
LONDON_CFG = OUT_DIR / "v2_exp4_winner_baseline_config.json"
DEFAULT_OUT = OUT_DIR / "track3_v44_shadow_registry_parity.json"


PROTOCOL = {
    "mode": "native_trade_entry_registry",
    "source": "merged_engine._run_v44_in_process + _extract_v44_trades",
    "what_shadow_asserts": [
        "If shadow_invoke_authorized_bar(v44_ny, ts) returns candidate_found, then a native V44 trade exists with entry_time == ts (M1 bar alignment).",
    ],
    "what_shadow_does_not_assert": [
        "Per-bar absence of signal: registry has no entry on bars where native V44 did not trade.",
        "Equivalence to a hypothetical extracted per-bar evaluator (not yet implemented).",
    ],
    "path_to_extracted_evaluator": "Future: share one entry-evaluator with backtest_session_momentum loop; until then registry mode is audited stand-in.",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(OUT_DIR / "USDJPY_M1_OANDA_500k.csv"))
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    ap.add_argument(
        "--no-build",
        action="store_true",
        help="Write protocol + dataset path only (skip V44 registry build / integrity check).",
    )
    args = ap.parse_args()

    dataset = Path(args.dataset).resolve()
    out_path = Path(args.output)

    base_payload: dict[str, Any] = {
        "artifact": "track3_v44_shadow_registry_parity",
        "protocol": PROTOCOL,
        "dataset": str(dataset),
    }

    if args.no_build:
        base_payload["status"] = "no_build_requested"
        base_payload["integrity"] = None
        base_payload["note"] = "Re-run without --no-build when CSV and time are available to populate integrity."
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(base_payload, indent=2), encoding="utf-8")
        print(f"Wrote {out_path} (--no-build)")
        return 0

    if not dataset.is_file():
        base_payload["status"] = "dataset_missing"
        base_payload["integrity"] = None
        base_payload["note"] = "Re-run when USDJPY M1 CSV is available at --dataset."
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(base_payload, indent=2), encoding="utf-8")
        print(f"Dataset missing: {dataset}", file=sys.stderr)
        print(f"Wrote placeholder {out_path}")
        return 2

    if not V44_CFG.is_file():
        base_payload["status"] = "v44_config_missing"
        out_path.write_text(json.dumps(base_payload, indent=2), encoding="utf-8")
        print(f"Missing {V44_CFG}", file=sys.stderr)
        return 2

    # Build registry (uses pickle cache under research_out/shadow_cache/)
    registry = build_native_invocation_registry(
        dataset=str(dataset),
        merged_engine=merged_engine,
        v14_config_path=V14_CFG,
        london_v2_config_path=LONDON_CFG,
        v44_config_path=V44_CFG,
    )

    v44_map = registry.entries_by_strategy.get("v44_ny") or {}
    results, embedded = merged_engine._run_v44_in_process(V44_CFG, str(dataset))
    if isinstance(embedded, dict) and "v5_account_size" in embedded:
        v44_eq = float(embedded.get("v5_account_size", 100_000.0))
    else:
        v44_eq = float((embedded or {}).get("v5", {}).get("account_size", 100_000.0))
    trades = merged_engine._extract_v44_trades(results, default_entry_equity=v44_eq)

    mismatches: list[dict[str, Any]] = []
    for t in trades:
        k = pd.Timestamp(t.entry_time).isoformat()
        if k not in v44_map or not v44_map[k]:
            mismatches.append({"entry_time": k, "issue": "trade_missing_from_registry"})

    extra_keys = set(v44_map.keys()) - {pd.Timestamp(t.entry_time).isoformat() for t in trades}
    trade_times = {pd.Timestamp(t.entry_time).isoformat() for t in trades}

    base_payload["status"] = "ok" if not mismatches and not extra_keys else "integrity_warning"
    base_payload["integrity"] = {
        "trade_count": len(trades),
        "registry_bar_key_count": len(v44_map),
        "registry_candidate_entries": sum(len(v) for v in v44_map.values()),
        "trade_times_match_registry_keys": len(mismatches) == 0,
        "registry_keys_without_matching_trade_count": len(extra_keys),
        "mismatch_samples": mismatches[:20],
        "extra_key_samples": sorted(extra_keys)[:20],
    }
    base_payload["source_mode"] = registry.source_mode_by_strategy.get("v44_ny")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(base_payload, indent=2), encoding="utf-8")
    print(json.dumps(base_payload["integrity"], indent=2))
    print(f"Wrote {out_path}")
    return 0 if base_payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
