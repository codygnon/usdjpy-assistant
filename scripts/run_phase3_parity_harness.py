#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
) -> dict[str, Any]:
    m1 = _load_m1(csv_path)
    frames = _precompute_frames(m1)
    profile = _profile(symbol, pip_size)
    policy = _policy("phase3_integrated_v7_defended")
    phase3_state: dict[str, Any] = {}
    trace = []
    start = max(0, start_index)
    stop = len(m1) if max_bars is None else min(len(m1), start + max(0, int(max_bars)))
    for idx in range(start, stop):
        snapshot = _data_snapshot(frames, idx)
        if snapshot["M15"].empty or snapshot["H1"].empty:
            continue
        tick = _tick_from_bar(m1.iloc[idx], pip_size, spread_pips)
        if trace_mode == "live_style":
            result = evaluate_phase3_bar(
                adapter=ReplayAdapter(equity=100000.0),
                profile=profile,
                log_dir=None,
                policy=policy,
                context={},
                data_by_tf=snapshot,
                tick=tick,
                mode="ARMED_AUTO_DEMO",
                phase3_state=phase3_state,
                store=ReplayStore(),
                sizing_config=None,
                is_new_m1=True,
                preset_id=PHASE3_DEFENDED_PRESET_ID,
            )
        else:
            result = evaluate_phase3_bar_replay(
                profile=profile,
                policy=policy,
                data_by_tf=snapshot,
                tick=tick,
                phase3_state=phase3_state,
                preset_id=PHASE3_DEFENDED_PRESET_ID,
            )
        phase3_state.update(result.get("phase3_state_updates") or {})
        envelope = dict(result.get("decision_envelope") or {})
        envelope["bar_time_utc"] = str(m1.iloc[idx]["time"])
        trace.append(envelope)
    payload = {
        "schema": "phase3_parity_trace_v1",
        "input_csv": str(csv_path),
        "preset_id": PHASE3_DEFENDED_PRESET_ID,
        "trace_mode": trace_mode,
        "start_index": start,
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
    for idx in range(count):
        diff = compare_phase3_envelopes(dict(left_rows[idx]), dict(right_rows[idx]))
        if not diff.matches:
            diffs.append(
                {
                    "index": idx,
                    "left_bar_time_utc": left_rows[idx].get("bar_time_utc"),
                    "right_bar_time_utc": right_rows[idx].get("bar_time_utc"),
                    "mismatches": diff.mismatches,
                }
            )
    payload = {
        "schema": "phase3_parity_diff_v1",
        "left": str(left_path),
        "right": str(right_path),
        "shared_count": count,
        "diff_count": len(diffs),
        "diffs": diffs,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


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
    ap.add_argument("--trace-mode", choices=["replay", "live_style"], default="replay")
    args = ap.parse_args()

    if args.compare_left and args.compare_right and args.out:
        payload = compare_traces(args.compare_left, args.compare_right, args.out)
        print(args.out)
        print(f"diff_count={payload['diff_count']}")
        return 0

    if not args.input_csv or not args.out:
        ap.error("--input-csv and --out are required unless using --compare-left/--compare-right")
    payload = run_trace(
        args.input_csv,
        args.out,
        symbol=args.symbol,
        pip_size=args.pip_size,
        spread_pips=args.spread_pips,
        start_index=args.start_index,
        trace_mode=args.trace_mode,
        max_bars=args.max_bars,
    )
    print(args.out)
    print(f"rows={len(payload['trace'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
