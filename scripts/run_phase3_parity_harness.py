#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from functools import lru_cache
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
        "additive_summary": _trace_additive_summary(trace),
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


def _trace_additive_summary(trace_rows: list[dict[str, Any]]) -> dict[str, Any]:
    package_ids: set[str] = set()
    slice_labels: set[str] = set()
    accepted_slice_labels: set[str] = set()
    baseline_intents = 0
    offensive_intents = 0
    accepted_baseline = 0
    accepted_offensive = 0
    slice_scales: dict[str, float] = {}
    strict_policy: dict[str, Any] = {}
    for row in trace_rows:
        if row.get("package_id"):
            package_ids.add(str(row.get("package_id")))
        attribution = dict(row.get("attribution") or {})
        package_spec = dict(attribution.get("package_spec") or {})
        if not strict_policy and isinstance(package_spec.get("strict_policy"), dict):
            strict_policy = dict(package_spec.get("strict_policy") or {})
        envelope = dict(attribution.get("additive_envelope") or {})
        for item in list(envelope.get("baseline_intents") or []):
            baseline_intents += 1
        for item in list(envelope.get("offensive_intents") or []):
            offensive_intents += 1
            label = str(item.get("slice_id") or "")
            if label:
                slice_labels.add(label)
                try:
                    slice_scales[label] = float(item.get("size_scale"))
                except Exception:
                    pass
        for item in list(envelope.get("accepted") or []):
            source = str(item.get("intent_source") or "")
            if source == "baseline":
                accepted_baseline += 1
            elif source == "offensive":
                accepted_offensive += 1
                label = str(item.get("slice_id") or "")
                if label:
                    accepted_slice_labels.add(label)
    return {
        "package_ids": sorted(package_ids),
        "offensive_slice_labels_observed": sorted(slice_labels),
        "accepted_offensive_slice_labels": sorted(accepted_slice_labels),
        "baseline_intent_count": baseline_intents,
        "offensive_intent_count": offensive_intents,
        "accepted_baseline_count": accepted_baseline,
        "accepted_offensive_count": accepted_offensive,
        "offensive_slice_scales": slice_scales,
        "strict_policy": strict_policy,
    }


def _extract_trace_additive_rows(trace_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for envelope_row in trace_rows:
        attribution = dict(envelope_row.get("attribution") or {})
        additive = dict(attribution.get("additive_envelope") or {})
        for item in list(additive.get("accepted") or []):
            rows.append(
                {
                    "entry_time_utc": str(item.get("entry_time_utc") or ""),
                    "source": str(item.get("intent_source") or ""),
                    "slice_id": str(item.get("slice_id") or ""),
                    "strategy_family": str(item.get("strategy_family") or ""),
                    "side": str(item.get("side") or ""),
                    "ownership_cell": str(item.get("ownership_cell") or ""),
                    "strategy_tag": str(item.get("strategy_tag") or ""),
                }
            )
    rows.sort(key=lambda row: (row["entry_time_utc"], row["source"], row["slice_id"], row["strategy_tag"], row["side"]))
    return rows


def _load_offline_additive_artifact(path: Path, package_name: str | None = None) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "candidate_name" in payload or "package_id" in payload:
        return {
            "package_id": str(payload.get("candidate_name") or payload.get("package_id") or ""),
            "base_cell_scales": dict(payload.get("base_cell_scales") or payload.get("base_slice_scales") or {}),
            "strict_policy": dict(payload.get("strict_policy") or {}),
            "source_path": str(path),
        }
    packages = list(payload.get("packages") or [])
    if packages:
        target_name = str(package_name or "").strip()
        for package in packages:
            for row in list(package.get("rows") or []):
                if target_name and str(row.get("name") or "") != target_name:
                    continue
                return {
                    "package_id": str(row.get("name") or ""),
                    "base_cell_scales": dict(row.get("cell_scales") or {}),
                    "strict_policy": dict(payload.get("policy") or {}),
                    "source_path": str(path),
                }
    raise ValueError(f"Unsupported offline additive artifact shape: {path}")


@lru_cache(maxsize=8)
def _offline_fixture_context(dataset_key: str, package_family: str) -> dict[str, Any]:
    from core.phase3_additive_contract import classify_defended_slice_label, defended_slice_source
    from scripts import backtest_offensive_slice_additive as additive
    from scripts import package_freeze_closeout_lib as freeze

    context = freeze.load_context()
    baseline_ctx = context["baseline_ctx_by_ds"][dataset_key]
    selected = freeze.scaled_combined_trades(freeze.package_scales(package_family), context["trades_by_ds"][dataset_key])
    policy = context["policy"]
    policy_selected, _policy_stats = additive._apply_conflict_policy_to_selected_trades(selected, policy)
    baseline_ids = {additive._trade_identity_from_trade_row(t) for t in baseline_ctx.baseline_coupled}
    additive_candidates = [t for t in policy_selected if not additive._has_exact_baseline_match_by_identity(t, baseline_ids)]
    margin_selected, _margin_stats = additive._apply_margin_policy_to_candidates(
        additive_candidates=additive_candidates,
        baseline_trades=baseline_ctx.baseline_coupled,
        starting_equity=100000.0,
        policy=policy,
    )
    baseline_rows: list[dict[str, Any]] = []
    offensive_rows: list[dict[str, Any]] = []
    active_scales = freeze.package_scales(package_family)
    for trade in baseline_ctx.baseline_kept:
        entry_ts = pd.Timestamp(trade.entry_time)
        if entry_ts.tzinfo is None:
            entry_ts = entry_ts.tz_localize("UTC")
        else:
            entry_ts = entry_ts.tz_convert("UTC")
        raw = dict(getattr(trade, "raw", {}) or {})
        strategy_family = str(trade.strategy)
        slice_label = classify_defended_slice_label(
            strategy_family=strategy_family,
            side=str(trade.side),
            ownership_cell=str(raw.get("ownership_cell") or ""),
            strategy_tag="",
            reason=str(raw.get("entry_reason") or raw.get("entry_profile") or ""),
            active_slice_scales=active_scales,
        )
        baseline_rows.append(
            {
                "entry_time_utc": entry_ts.isoformat(),
                "source": "baseline",
                "slice_id": str(slice_label or f"baseline:{trade.strategy}:{trade.side}@{str(raw.get('ownership_cell') or '')}"),
                "strategy_family": strategy_family,
                "side": str(trade.side),
                "ownership_cell": str(raw.get("ownership_cell") or ""),
                "strategy_tag": str(raw.get("strategy_tag") or ""),
            }
        )
    for trade in margin_selected:
        entry_ts = pd.Timestamp(trade["entry_time"])
        if entry_ts.tzinfo is None:
            entry_ts = entry_ts.tz_localize("UTC")
        else:
            entry_ts = entry_ts.tz_convert("UTC")
        raw = dict(trade.get("raw") or {})
        slice_label = str(raw.get("slice_id") or classify_defended_slice_label(
            strategy_family=str(trade.get("strategy") or ""),
            side=str(trade.get("side") or ""),
            ownership_cell=str(trade.get("ownership_cell") or ""),
            strategy_tag=str(raw.get("strategy_tag") or ""),
            reason=str(raw.get("entry_reason") or raw.get("entry_profile") or ""),
            active_slice_scales=active_scales,
        ) or "")
        offensive_rows.append(
            {
                "entry_time_utc": entry_ts.isoformat(),
                "source": defended_slice_source(slice_label),
                "slice_id": slice_label,
                "strategy_family": str(trade.get("strategy") or ""),
                "side": str(trade.get("side") or ""),
                "ownership_cell": str(trade.get("ownership_cell") or ""),
                "strategy_tag": str(raw.get("strategy_tag") or ""),
            }
        )
    return {
        "dataset_key": dataset_key,
        "package_family": package_family,
        "baseline_rows": baseline_rows,
        "offensive_rows": offensive_rows,
    }


def _offline_rows_for_fixture(dataset_key: str, *, start_utc: str, end_utc: str, package_family: str = "v7_pfdd") -> dict[str, Any]:
    cached = _offline_fixture_context(dataset_key, package_family)
    start_ts = pd.Timestamp(start_utc)
    end_ts = pd.Timestamp(end_utc)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    else:
        end_ts = end_ts.tz_convert("UTC")

    rows = [
        row
        for row in list(cached.get("baseline_rows") or []) + list(cached.get("offensive_rows") or [])
        if start_ts <= pd.Timestamp(row["entry_time_utc"]).tz_convert("UTC") <= end_ts
    ]
    rows.sort(key=lambda row: (row["entry_time_utc"], row["source"], row["slice_id"], row["strategy_family"], row["side"]))
    return {
        "dataset_key": dataset_key,
        "package_family": package_family,
        "start_utc": start_ts.isoformat(),
        "end_utc": end_ts.isoformat(),
        "rows": rows,
    }


def compare_trace_to_offline_additive_artifact(
    trace_path: Path,
    artifact_path: Path,
    out_path: Path,
    *,
    package_name: str | None = None,
) -> dict[str, Any]:
    trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    summary = dict(trace_payload.get("additive_summary") or _trace_additive_summary(list(trace_payload.get("trace") or [])))
    artifact = _load_offline_additive_artifact(artifact_path, package_name=package_name)
    mismatches: list[str] = []

    observed_package_ids = [str(v) for v in list(summary.get("package_ids") or []) if str(v).strip()]
    expected_package_id = str(artifact.get("package_id") or "")
    if expected_package_id and observed_package_ids and expected_package_id not in observed_package_ids:
        mismatches.append(f"package_id: observed={observed_package_ids!r} expected={expected_package_id!r}")

    expected_slices = {
        str(label): float(scale)
        for label, scale in dict(artifact.get("base_cell_scales") or {}).items()
        if float(scale) > 0.0
    }
    observed_slices = set(str(label) for label in list(summary.get("offensive_slice_labels_observed") or []) if str(label).strip())
    accepted_slices = set(str(label) for label in list(summary.get("accepted_offensive_slice_labels") or []) if str(label).strip())
    unexpected_observed = sorted(observed_slices - set(expected_slices))
    unexpected_accepted = sorted(accepted_slices - set(expected_slices))
    if unexpected_observed:
        mismatches.append(f"unexpected_offensive_slices: {unexpected_observed}")
    if unexpected_accepted:
        mismatches.append(f"unexpected_accepted_offensive_slices: {unexpected_accepted}")

    for label, observed_scale in dict(summary.get("offensive_slice_scales") or {}).items():
        if label not in expected_slices:
            continue
        expected_scale = float(expected_slices[label])
        if abs(float(observed_scale) - expected_scale) > 1e-9:
            mismatches.append(f"slice_scale[{label}]: observed={observed_scale!r} expected={expected_scale!r}")

    expected_policy = dict(artifact.get("strict_policy") or {})
    observed_policy = dict(summary.get("strict_policy") or {})
    for key in (
        "allow_internal_overlap",
        "allow_opposite_side_overlap",
        "max_open_offensive",
        "max_entries_per_day",
        "margin_model_enabled",
        "margin_leverage",
        "max_lot_per_trade",
    ):
        if key not in expected_policy:
            continue
        if observed_policy.get(key) != expected_policy.get(key):
            mismatches.append(f"strict_policy[{key}]: observed={observed_policy.get(key)!r} expected={expected_policy.get(key)!r}")

    payload = {
        "schema": "phase3_offline_additive_diff_v1",
        "trace": str(trace_path),
        "offline_artifact": str(artifact_path),
        "matches": len(mismatches) == 0,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "trace_additive_summary": summary,
        "offline_additive_artifact": artifact,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def compare_trace_to_offline_rows(
    trace_path: Path,
    out_path: Path,
    *,
    dataset_key: str,
    package_family: str = "v7_pfdd",
) -> dict[str, Any]:
    trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    trace_rows = list(trace_payload.get("trace") or [])
    accepted_rows = _extract_trace_additive_rows(trace_rows)
    if trace_rows:
        start_utc = str(trace_rows[0].get("bar_time_utc") or "")
        end_utc = str(trace_rows[-1].get("bar_time_utc") or "")
    else:
        start_utc = ""
        end_utc = ""
    offline = _offline_rows_for_fixture(dataset_key, start_utc=start_utc, end_utc=end_utc, package_family=package_family)
    observed = {
        (
            row["entry_time_utc"],
            row["source"],
            row["slice_id"],
            row["strategy_family"],
            row["side"],
            row["ownership_cell"],
        )
        for row in accepted_rows
    }
    expected = {
        (
            row["entry_time_utc"],
            row["source"],
            row["slice_id"],
            row["strategy_family"],
            row["side"],
            row["ownership_cell"],
        )
        for row in list(offline.get("rows") or [])
    }
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    payload = {
        "schema": "phase3_offline_row_diff_v1",
        "trace": str(trace_path),
        "dataset_key": dataset_key,
        "package_family": package_family,
        "matches": not missing and not extra,
        "missing_count": len(missing),
        "extra_count": len(extra),
        "missing_rows": [
            {
                "entry_time_utc": row[0],
                "source": row[1],
                "slice_id": row[2],
                "strategy_family": row[3],
                "side": row[4],
                "ownership_cell": row[5],
            }
            for row in missing
        ],
        "extra_rows": [
            {
                "entry_time_utc": row[0],
                "source": row[1],
                "slice_id": row[2],
                "strategy_family": row[3],
                "side": row[4],
                "ownership_cell": row[5],
            }
            for row in extra
        ],
        "observed_rows": accepted_rows,
        "offline_rows": offline,
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
    ap.add_argument("--compare-offline-artifact", type=Path)
    ap.add_argument("--offline-package-name", default="")
    ap.add_argument("--compare-offline-rows", action="store_true")
    ap.add_argument("--dataset-key", default="500k")
    ap.add_argument("--package-family", default="v7_pfdd")
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
    if args.compare_trace and args.compare_offline_artifact and args.out:
        payload = compare_trace_to_offline_additive_artifact(
            args.compare_trace,
            args.compare_offline_artifact,
            args.out,
            package_name=args.offline_package_name or None,
        )
        print(args.out)
        print(f"mismatch_count={payload['mismatch_count']}")
        return 0
    if args.compare_trace and args.compare_offline_rows and args.out:
        payload = compare_trace_to_offline_rows(
            args.compare_trace,
            args.out,
            dataset_key=args.dataset_key,
            package_family=args.package_family,
        )
        print(args.out)
        print(f"missing_count={payload['missing_count']} extra_count={payload['extra_count']}")
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
