#!/usr/bin/env python3
"""
Track 3: spot-check that shadow invocation's V14/London candidate maps are
internally consistent (repeatable) and that the same extracted evaluators are
imported by native backtest scripts.

This is not a full backtest-vs-shadow PnL parity test; it catches accidental
drift in shadow_entry_invocation builders.
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

from core.shadow_entry_invocation import _build_london_candidate_map, _build_v14_candidate_map
from scripts import backtest_merged_integrated_tokyo_london_v2_ny as merged_engine

OUT_DIR = ROOT / "research_out"
V14_CFG = OUT_DIR / "tokyo_optimized_v14_config.json"
LONDON_CFG = OUT_DIR / "v2_exp4_winner_baseline_config.json"
DEFAULT_OUT = OUT_DIR / "track3_v14_london_evaluator_parity_spotcheck.json"


def _script_imports_core_evaluator(module_path: Path, core_submodule: str) -> bool:
    src = module_path.read_text(encoding="utf-8")
    return f"from core.{core_submodule} import" in src


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(OUT_DIR / "USDJPY_M1_OANDA_500k.csv"))
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    ap.add_argument(
        "--import-only",
        action="store_true",
        help="Skip repeat-build (fast); only verify scripts import core evaluators.",
    )
    args = ap.parse_args()

    dataset = Path(args.dataset).resolve()
    out_path = Path(args.output)

    payload: dict[str, Any] = {
        "artifact": "track3_v14_london_evaluator_parity_spotcheck",
        "dataset": str(dataset),
        "import_sanity": {
            "tokyo_meanrev_imports_v14_evaluator": _script_imports_core_evaluator(
                ROOT / "scripts" / "backtest_tokyo_meanrev.py",
                "v14_entry_evaluator",
            ),
            "v2_multisetup_imports_london_evaluator": _script_imports_core_evaluator(
                ROOT / "scripts" / "backtest_v2_multisetup_london.py",
                "london_v2_entry_evaluator",
            ),
        },
        "repeat_build": None,
    }

    if args.import_only:
        payload["status"] = "import_only"
        payload["note"] = "Re-run without --import-only to execute repeat-build parity on a full CSV."
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {out_path} (--import-only)")
        return 0

    if not dataset.is_file():
        payload["status"] = "dataset_missing"
        payload["note"] = "Repeat-build checks skipped; import_sanity still valid."
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {out_path} (dataset missing)")
        return 2

    if not V14_CFG.is_file() or not LONDON_CFG.is_file():
        payload["status"] = "config_missing"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return 2

    m1, c1 = _build_v14_candidate_map(dataset=str(dataset), config_path=V14_CFG)
    m2, c2 = _build_v14_candidate_map(dataset=str(dataset), config_path=V14_CFG)
    l1, lc1 = _build_london_candidate_map(
        dataset=str(dataset), config_path=LONDON_CFG, merged_engine=merged_engine
    )
    l2, lc2 = _build_london_candidate_map(
        dataset=str(dataset), config_path=LONDON_CFG, merged_engine=merged_engine
    )

    v14_ts_match = set(m1.keys()) == set(m2.keys()) and c1 == c2
    london_ts_match = set(l1.keys()) == set(l2.keys()) and lc1 == lc2

    payload["repeat_build"] = {
        "v14_counts_match": c1 == c2,
        "v14_timestamp_keys_match": set(m1.keys()) == set(m2.keys()),
        "london_counts_match": lc1 == lc2,
        "london_timestamp_keys_match": set(l1.keys()) == set(l2.keys()),
        "v14_candidate_count": c1,
        "london_candidate_count": lc1,
    }
    payload["status"] = "ok" if v14_ts_match and london_ts_match else "mismatch"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Wrote {out_path}")
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
