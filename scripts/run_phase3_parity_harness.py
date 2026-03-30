#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

ROOT = Path("/Users/codygnon/Documents/usdjpy_assistant")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.phase3_package_spec import PHASE3_DEFENDED_PRESET_ID
from core.phase3_shared_engine import ReplayAdapter, ReplayStore, compare_phase3_envelopes, evaluate_phase3_bar, evaluate_phase3_bar_replay


def _load_m1(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"time", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")
    out = df[list(required)].copy()
    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
    out = out.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    return out


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    temp = df.set_index("time").sort_index()
    out = temp.resample(rule, label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    out = out.reset_index()
    return out


def _precompute_frames(m1: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "M1": m1,
        "M5": _resample(m1, "5min"),
        "M15": _resample(m1, "15min"),
        "H1": _resample(m1, "1h"),
    }


def _data_snapshot(frames: dict[str, pd.DataFrame], idx: int) -> dict[str, pd.DataFrame]:
    bar_time = frames["M1"].iloc[idx]["time"]
    m1_cut = frames["M1"].iloc[: idx + 1].copy()
    return {
        "M1": m1_cut,
        "M5": frames["M5"][frames["M5"]["time"] <= bar_time].copy(),
        "M15": frames["M15"][frames["M15"]["time"] <= bar_time].copy(),
        "H1": frames["H1"][frames["H1"]["time"] <= bar_time].copy(),
    }


def _tick_from_bar(bar: pd.Series, pip_size: float, spread_pips: float) -> Any:
    spread = float(spread_pips) * float(pip_size)
    mid = float(bar["close"])
    return SimpleNamespace(bid=mid - spread / 2.0, ask=mid + spread / 2.0)


def _profile(symbol: str, pip_size: float) -> Any:
    return SimpleNamespace(
        symbol=symbol,
        pip_size=pip_size,
        broker_type="simulated",
        profile_name="phase3_parity_harness",
        active_preset_name=PHASE3_DEFENDED_PRESET_ID,
    )


def _policy(policy_id: str) -> Any:
    return SimpleNamespace(id=policy_id, type="phase3_integrated", enabled=True)


def run_trace(
    csv_path: Path,
    out_path: Path,
    *,
    symbol: str,
    pip_size: float,
    spread_pips: float,
    start_index: int,
    trace_mode: str = "replay",
    max_bars: int | None = None,
    warmup_bars: int = 0,
) -> dict[str, Any]:
    m1 = _load_m1(csv_path)
    frames = _precompute_frames(m1)
    profile = _profile(symbol, pip_size)
    policy = _policy("phase3_integrated_v7_defended")
    phase3_state: dict[str, Any] = {}
    adapter = ReplayAdapter(equity=100000.0)
    store = ReplayStore()
    trace = []
    start = max(0, start_index)
    warmup_start = max(0, start - max(0, int(warmup_bars)))
    stop = len(m1) if max_bars is None else min(len(m1), start + max(0, int(max_bars)))
    for idx in range(warmup_start, stop):
        snapshot = _data_snapshot(frames, idx)
        if snapshot["M15"].empty or snapshot["H1"].empty:
            continue
        tick = _tick_from_bar(m1.iloc[idx], pip_size, spread_pips)
        if trace_mode == "live_style":
            result = evaluate_phase3_bar(
                adapter=adapter,
                profile=profile,
                log_dir=None,
                policy=policy,
                context={},
                data_by_tf=snapshot,
                tick=tick,
                mode="ARMED_AUTO_DEMO",
                phase3_state=phase3_state,
                store=store,
                sizing_config=None,
                is_new_m1=True,
                preset_id=PHASE3_DEFENDED_PRESET_ID,
            )
        else:
            result = evaluate_phase3_bar_replay(
                adapter=adapter,
                profile=profile,
                policy=policy,
                data_by_tf=snapshot,
                tick=tick,
                phase3_state=phase3_state,
                store=store,
                preset_id=PHASE3_DEFENDED_PRESET_ID,
            )
        phase3_state.update(result.get("phase3_state_updates") or {})
        if idx < start:
            continue
        envelope = dict(result.get("decision_envelope") or {})
        envelope["bar_time_utc"] = str(m1.iloc[idx]["time"])
        trace.append(envelope)
    payload = {
        "schema": "phase3_parity_trace_v1",
        "input_csv": str(csv_path),
        "preset_id": PHASE3_DEFENDED_PRESET_ID,
        "trace_mode": trace_mode,
        "start_index": start,
        "warmup_start_index": warmup_start,
        "warmup_bars": int(max(0, warmup_bars)),
        "max_bars": None if max_bars is None else int(max_bars),
        "trace": trace,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def compare_traces(left_path: Path, right_path: Path, out_path: Path) -> dict[str, Any]:
    left = json.loads(left_path.read_text(encoding="utf-8"))
    right = json.loads(right_path.read_text(encoding="utf-8"))
    left_rows = list(left.get("trace") or [])
    right_rows = list(right.get("trace") or [])
    count = min(len(left_rows), len(right_rows))
    diffs = []
    dimension_counts: dict[str, int] = {}
    for idx in range(count):
        diff = compare_phase3_envelopes(dict(left_rows[idx]), dict(right_rows[idx]))
        if not diff.matches:
            dimensions = _categorize_mismatches(diff.mismatches)
            for dimension in dimensions:
                dimension_counts[dimension] = int(dimension_counts.get(dimension, 0)) + 1
            diffs.append(
                {
                    "index": idx,
                    "left_bar_time_utc": left_rows[idx].get("bar_time_utc"),
                    "right_bar_time_utc": right_rows[idx].get("bar_time_utc"),
                    "mismatches": diff.mismatches,
                    "dimensions": dimensions,
                }
            )
    payload = {
        "schema": "phase3_parity_diff_v1",
        "left": str(left_path),
        "right": str(right_path),
        "shared_count": count,
        "diff_count": len(diffs),
        "dimension_counts": dimension_counts,
        "diffs": diffs,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def parse_phase3_diagnostics_log(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part for part in line.split("\t") if part]
        if not parts:
            continue
        row: dict[str, Any] = {"logged_at_utc": parts[0]}
        for token in parts[1:]:
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            row[key] = value.strip()
        rows.append(row)
    return rows


def compare_trace_to_diagnostics(trace_path: Path, diagnostics_path: Path, out_path: Path) -> dict[str, Any]:
    trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    trace_rows = list(trace_payload.get("trace") or [])
    diag_rows = parse_phase3_diagnostics_log(diagnostics_path)
    diag_by_bar = {str(row.get("bar") or ""): row for row in diag_rows if row.get("bar")}
    diffs: list[dict[str, Any]] = []
    matched = 0
    for idx, trace_row in enumerate(trace_rows):
        bar_time = str(trace_row.get("bar_time_utc") or "")
        diag_row = diag_by_bar.get(bar_time)
        if not diag_row:
            diffs.append(
                {
                    "index": idx,
                    "bar_time_utc": bar_time,
                    "dimensions": ["runtime_forensic"],
                    "mismatches": ["missing diagnostics row"],
                }
            )
            continue
        matched += 1
        mismatches: list[str] = []
        expected_session = str(trace_row.get("session") or "none")
        if str(diag_row.get("session") or "none") != expected_session:
            mismatches.append(f"session: {diag_row.get('session')!r} != {expected_session!r}")
        expected_strategy = _diagnostic_strategy_family(trace_row)
        if str(diag_row.get("strategy") or "") != expected_strategy:
            mismatches.append(f"strategy: {diag_row.get('strategy')!r} != {expected_strategy!r}")
        expected_placed = "1" if trace_row.get("placed") else "0"
        if str(diag_row.get("placed") or "") != expected_placed:
            mismatches.append(f"placed: {diag_row.get('placed')!r} != {expected_placed!r}")
        expected_reason = str(trace_row.get("reason") or "")
        actual_reason = str(diag_row.get("reason") or "").strip("'")
        if actual_reason != expected_reason:
            mismatches.append(f"reason: {actual_reason!r} != {expected_reason!r}")
        expected_cell = str(trace_row.get("ownership_cell") or "")
        if str(diag_row.get("ownership_cell") or "") != expected_cell:
            mismatches.append(f"ownership_cell: {diag_row.get('ownership_cell')!r} != {expected_cell!r}")
        if mismatches:
            diffs.append(
                {
                    "index": idx,
                    "bar_time_utc": bar_time,
                    "dimensions": ["runtime_forensic", "surface"],
                    "mismatches": mismatches,
                }
            )
    payload = {
        "schema": "phase3_runtime_forensic_diff_v1",
        "trace": str(trace_path),
        "diagnostics_log": str(diagnostics_path),
        "shared_count": matched,
        "diff_count": len(diffs),
        "diffs": diffs,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _diagnostic_strategy_family(trace_row: dict[str, Any]) -> str:
    strategy_tag = str(trace_row.get("strategy_tag") or "")
    if strategy_tag.startswith("phase3:v44_ny"):
        return "v44_ny"
    if strategy_tag.startswith("phase3:london_v2_d"):
        return "london_v2_d"
    if strategy_tag.startswith("phase3:london_v2_arb"):
        return "london_v2_arb"
    if strategy_tag.startswith("phase3:v14"):
        return "v14"
    return ""


def _categorize_mismatches(mismatches: list[str]) -> list[str]:
    categories: set[str] = set()
    for item in mismatches:
        if re.search(r"^(session|strategy_tag|strategy_family|blocking_filter_ids|reason):", item):
            categories.add("entry")
        if re.search(r"(ownership_cell|attribution)", item):
            categories.add("surface")
        if re.search(r"(size_units|entry_price|sl_price|tp1_price)", item):
            categories.add("sizing")
        if re.search(r"(exit_policy|managed_exit_plan)", item):
            categories.add("exit")
        if re.search(r"(parity_state)", item):
            categories.add("state")
    return sorted(categories or {"uncategorized"})


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the shared Phase 3 replay/parity harness.")
    ap.add_argument("--input-csv", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--compare-left", type=Path)
    ap.add_argument("--compare-right", type=Path)
    ap.add_argument("--symbol", default="USDJPY")
    ap.add_argument("--pip-size", type=float, default=0.01)
    ap.add_argument("--spread-pips", type=float, default=1.5)
    ap.add_argument("--start-index", type=int, default=120)
    ap.add_argument("--max-bars", type=int, default=None)
    ap.add_argument("--warmup-bars", type=int, default=0)
    ap.add_argument("--trace-mode", choices=["replay", "live_style"], default="replay")
    ap.add_argument("--compare-trace", type=Path)
    ap.add_argument("--compare-diagnostics", type=Path)
    args = ap.parse_args()

    if args.compare_left and args.compare_right and args.out:
        payload = compare_traces(args.compare_left, args.compare_right, args.out)
        print(args.out)
        print(f"diff_count={payload['diff_count']}")
        return 0
    if args.compare_trace and args.compare_diagnostics and args.out:
        payload = compare_trace_to_diagnostics(args.compare_trace, args.compare_diagnostics, args.out)
        print(args.out)
        print(f"diff_count={payload['diff_count']}")
        return 0

    if not args.input_csv or not args.out:
        ap.error("--input-csv and --out are required unless using a compare mode")
    payload = run_trace(
        args.input_csv,
        args.out,
        symbol=args.symbol,
        pip_size=args.pip_size,
        spread_pips=args.spread_pips,
        start_index=args.start_index,
        trace_mode=args.trace_mode,
        max_bars=args.max_bars,
        warmup_bars=args.warmup_bars,
    )
    print(args.out)
    print(f"rows={len(payload['trace'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
