#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path("/Users/codygnon/Documents/usdjpy_assistant")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.phase3_package_spec import DEFENDED_PACKAGE_ID, PHASE3_DEFENDED_PRESET_ID
from scripts.run_phase3_parity_harness import compare_traces, run_trace
from scripts.run_offensive_slice_discovery import DATASETS

OUT_DIR = ROOT / "research_out"
SUMMARY_JSON = OUT_DIR / "phase3_defended_shared_replay_summary.json"
SUMMARY_MD = OUT_DIR / "phase3_defended_shared_replay_summary.md"
TRACE_SAMPLE_BARS = 360


def _dataset_outputs(name: str) -> dict[str, Path]:
    return {
        "golden": OUT_DIR / f"phase3_defended_{name}_golden_trace.json",
        "live_style": OUT_DIR / f"phase3_defended_{name}_live_style_trace.json",
        "diff": OUT_DIR / f"phase3_defended_{name}_parity_diff.json",
    }


def _build_md(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase 3 Defended Shared Replay",
        "",
        f"- package: `{payload['package_id']}`",
        f"- preset: `{payload['preset_id']}`",
        "",
    ]
    for dataset_key, row in payload["datasets"].items():
        lines.extend(
            [
                f"## {dataset_key}",
                "",
                f"- input: `{row['input_csv']}`",
                f"- dataset key: `{row.get('dataset_key', dataset_key)}`",
                f"- start index: `{row.get('start_index', 0)}`",
                f"- sample bars: `{row['sample_bars']}`",
                f"- golden rows: `{row['golden_rows']}`",
                f"- live-style rows: `{row['live_style_rows']}`",
                f"- parity diffs: `{row['diff_count']}`",
                f"- golden trace: `{row['golden_trace_path']}`",
                f"- live-style trace: `{row['live_style_trace_path']}`",
                f"- diff artifact: `{row['diff_path']}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate defended-package shared replay and live-style parity traces.")
    ap.add_argument("--dataset-key", default="500k")
    ap.add_argument("--start-index", type=int, default=120)
    ap.add_argument("--max-bars", type=int, default=TRACE_SAMPLE_BARS)
    ap.add_argument("--label", default=None, help="Optional artifact label override.")
    args = ap.parse_args()

    dataset_key = str(args.dataset_key)
    label = str(args.label or dataset_key)
    payload: dict[str, Any] = {
        "schema": "phase3_defended_shared_replay_v1",
        "package_id": DEFENDED_PACKAGE_ID,
        "preset_id": PHASE3_DEFENDED_PRESET_ID,
        "datasets": {},
    }
    csv_path = Path(DATASETS[dataset_key])
    outputs = _dataset_outputs(label)
    golden = run_trace(
        csv_path,
        outputs["golden"],
        symbol="USDJPY",
        pip_size=0.01,
        spread_pips=1.5,
        start_index=int(args.start_index),
        trace_mode="replay",
        max_bars=int(args.max_bars),
    )
    live_style = run_trace(
        csv_path,
        outputs["live_style"],
        symbol="USDJPY",
        pip_size=0.01,
        spread_pips=1.5,
        start_index=int(args.start_index),
        trace_mode="live_style",
        max_bars=int(args.max_bars),
    )
    diff = compare_traces(outputs["golden"], outputs["live_style"], outputs["diff"])
    payload["datasets"][label] = {
        "input_csv": str(csv_path),
        "dataset_key": dataset_key,
        "start_index": int(args.start_index),
        "sample_bars": int(args.max_bars),
        "golden_rows": len(golden.get("trace") or []),
        "live_style_rows": len(live_style.get("trace") or []),
        "diff_count": int(diff.get("diff_count", 0)),
        "golden_trace_path": str(outputs["golden"]),
        "live_style_trace_path": str(outputs["live_style"]),
        "diff_path": str(outputs["diff"]),
    }

    SUMMARY_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    SUMMARY_MD.write_text(_build_md(payload), encoding="utf-8")
    print(SUMMARY_JSON)
    print(SUMMARY_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
