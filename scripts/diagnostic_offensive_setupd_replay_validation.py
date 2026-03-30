#!/usr/bin/env python3
"""
Fast validation for the offensive Setup D pilot candidate.

This replays the frozen gate chain over the London candidate universe instead of
scanning every bar. That is sufficient for validating the exact additive trade
list, schema completeness, and defensive-state integrity for this narrow pilot.
"""
from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path("/Users/codygnon/Documents/usdjpy_assistant")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.chart_authorization import authorize_bar, load_authorization_context
from core.chart_entry_eligibility import EntryDataFlags, evaluate_shadow_entry_eligibility
from core.offensive_setupd_pilot import (
    default_offensive_setupd_pilot_rule,
    london_session_window_matches_timestamp,
    pilot_bar_matches_scope,
    pilot_candidate_matches_rule,
    pilot_window_matches_timestamp,
)
from core.ownership_table import cell_key, der_bucket, er_bucket
from core.phase3_integrated_engine import classify_session, load_phase3_sizing_config
from core.shadow_entry_invocation import _build_london_candidate_map
from scripts import backtest_merged_integrated_tokyo_london_v2_ny as merged_engine
from scripts import diagnostic_chart_authorization_loop as auth_loop
from scripts import diagnostic_shadow_entry_invocation as shadow_diag
from scripts.diagnostic_offensive_setupd_whitespace_pilot import _defensive_state_at, LONDON_CFG_PATH, OUT_DIR

OUTPUT_PATH = OUT_DIR / "offensive_pilot_setupd_replay_validation.json"
ADDITIVE_REFERENCE = OUT_DIR / "offensive_setupd_additive_mean_reversion_er_low_der_neg_100pct_first30.json"
DATASETS = {
    "500k": str(OUT_DIR / "USDJPY_M1_OANDA_500k.csv"),
    "1000k": str(OUT_DIR / "USDJPY_M1_OANDA_1000k.csv"),
}
VARIANT_K_REPORTS = {
    "500k": OUT_DIR / "phase3_integrated_variant_k_500k_report.json",
    "1000k": OUT_DIR / "phase3_integrated_variant_k_1000k_report.json",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate the Setup D pilot gate chain against additive artifacts.")
    p.add_argument(
        "--dataset-key",
        action="append",
        choices=sorted(DATASETS.keys()),
        help="Optional dataset key(s) to validate. Defaults to all.",
    )
    p.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_PATH),
        help="Output JSON path.",
    )
    return p.parse_args()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return str(obj)


def _required_event_fields() -> set[str]:
    return {
        "signal_time",
        "entry_time",
        "session",
        "ownership_cell",
        "authorized_strategy",
        "candidate_side",
        "candidate_trigger_type",
        "candidate_setup_type",
        "defensive_state",
        "outcome",
        "reason",
    }


def _required_defensive_state_fields() -> set[str]:
    return {
        "defensive_long_open",
        "defensive_short_open",
        "defensive_trade_count",
        "defensive_active_strategies",
    }


def _schema_check(events: list[dict[str, Any]]) -> dict[str, Any]:
    required = _required_event_fields()
    defensive_required = _required_defensive_state_fields()
    errors: list[dict[str, Any]] = []
    for idx, event in enumerate(events):
        missing = sorted(required - set(event.keys()))
        if missing:
            errors.append({
                "index": idx,
                "signal_time": event.get("signal_time"),
                "reason": "missing_event_fields",
                "missing_fields": missing,
            })
            continue
        defensive_state = event.get("defensive_state")
        if not isinstance(defensive_state, dict):
            errors.append({
                "index": idx,
                "signal_time": event.get("signal_time"),
                "reason": "defensive_state_not_dict",
            })
            continue
        missing_def = sorted(defensive_required - set(defensive_state.keys()))
        if missing_def:
            errors.append({
                "index": idx,
                "signal_time": event.get("signal_time"),
                "reason": "missing_defensive_state_fields",
                "missing_fields": missing_def,
            })
            continue
        if defensive_state["defensive_trade_count"] == 0 and (
            defensive_state["defensive_long_open"] or defensive_state["defensive_short_open"]
        ):
            errors.append({
                "index": idx,
                "signal_time": event.get("signal_time"),
                "reason": "open_flag_without_open_trade",
            })
    return {
        "event_count": len(events),
        "error_count": len(errors),
        "errors": errors[:20],
        "schema_complete_and_parseable": len(errors) == 0,
    }


def _load_additive_reference() -> dict[str, Any]:
    return json.loads(ADDITIVE_REFERENCE.read_text(encoding="utf-8"))


def _baseline_trade_maps_from_report(dataset_key: str) -> tuple[set[tuple[str, str]], set[str], list[dict[str, Any]]]:
    report = json.loads(VARIANT_K_REPORTS[dataset_key].read_text(encoding="utf-8"))
    exact_entries: set[tuple[str, str]] = set()
    any_entry_times: set[str] = set()
    intervals: list[dict[str, Any]] = []
    for trade in report.get("closed_trades", []):
        ts = pd.Timestamp(trade["entry_time"]).isoformat()
        side = str(trade["side"])
        exact_entries.add((ts, side))
        any_entry_times.add(ts)
        intervals.append({
            "entry_time": pd.Timestamp(trade["entry_time"]),
            "exit_time": pd.Timestamp(trade["exit_time"]),
            "side": side,
            "strategy": str(trade["strategy"]),
        })
    return exact_entries, any_entry_times, intervals


def _expected_lists(additive_variant: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    exact = additive_variant["samples"].get("exact_overlap", [])
    new_add = additive_variant["samples"].get("new_additive", [])
    return {
        "exact_overlap": sorted([
            {
                "signal_time": str(t["signal_time"]),
                "entry_time": str(t["entry_time"]),
                "side": "buy" if str(t["direction"]).lower() == "long" else "sell",
            }
            for t in exact
        ], key=lambda x: (x["signal_time"], x["entry_time"], x["side"])),
        "new_additive": sorted([
            {
                "signal_time": str(t["raw"]["signal_time"]),
                "entry_time": str(t["entry_time"]),
                "side": str(t["side"]),
            }
            for t in new_add
        ], key=lambda x: (x["signal_time"], x["entry_time"], x["side"])),
    }


def _observed_lists(events: list[dict[str, Any]]) -> dict[str, list[dict[str, str]]]:
    exact = []
    new_add = []
    for event in events:
        if event["outcome"] != "pilot_candidate":
            continue
        row = {
            "signal_time": str(event["signal_time"]),
            "entry_time": str(event["entry_time"]),
            "side": str(event["candidate_side"]),
        }
        if event.get("exact_baseline_entry_overlap", False):
            exact.append(row)
        else:
            new_add.append(row)
    exact.sort(key=lambda x: (x["signal_time"], x["entry_time"], x["side"]))
    new_add.sort(key=lambda x: (x["signal_time"], x["entry_time"], x["side"]))
    return {
        "exact_overlap": exact,
        "new_additive": new_add,
    }


def _run_dataset(dataset_key: str, dataset: str) -> dict[str, Any]:
    rule = default_offensive_setupd_pilot_rule()
    context = load_authorization_context(research_out=OUT_DIR)
    sizing_cfg = load_phase3_sizing_config()
    bars = auth_loop._load_bar_frame(dataset).copy()
    bars["time"] = pd.to_datetime(bars["time"], utc=True, errors="coerce")
    bars = bars.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    bars_by_time = {pd.Timestamp(row["time"]).isoformat(): row for _, row in bars.iterrows()}
    day_flags = shadow_diag._build_daily_entry_reference_flags(dataset)
    london_by_ts, _ = _build_london_candidate_map(dataset=dataset, config_path=LONDON_CFG_PATH, merged_engine=merged_engine)
    exact_baseline_entries, any_baseline_entries, defensive_intervals = _baseline_trade_maps_from_report(dataset_key)

    events: list[dict[str, Any]] = []
    defensive_state_errors: list[dict[str, Any]] = []

    fired_channel_days: dict[tuple[str, str], str] = {}

    for signal_time in sorted(london_by_ts.keys()):
        candidates = london_by_ts[signal_time]
        if not candidates:
            continue
        candidate = candidates[0]
        ts = pd.Timestamp(signal_time)
        candidate_entry_time = str(candidate.entry_time or "")
        channel_key = (pd.Timestamp(candidate_entry_time or signal_time).date().isoformat(), rule.direction)
        prior_signal = fired_channel_days.get(channel_key)
        general_setupd_long_first30 = (
            str(candidate.trigger_type or "") == f"setup_{rule.setup_type}"
            and str(candidate.side or "").lower() == "buy"
            and str(candidate.raw.get("setup_type") or "") == rule.setup_type
            and str(candidate.raw.get("direction") or "").lower() == rule.direction
            and london_session_window_matches_timestamp(ts)
            and pilot_window_matches_timestamp(ts, rule)
        )
        if not general_setupd_long_first30:
            continue
        if prior_signal is None:
            fired_channel_days[channel_key] = signal_time
        row = bars_by_time.get(signal_time)
        if row is None:
            continue
        session = classify_session(ts.to_pydatetime(), sizing_cfg)
        if session is None and london_session_window_matches_timestamp(ts):
            session = "london"
        regime_label = str(row.get("regime_hysteresis", "ambiguous"))
        ownership_cell = cell_key(
            regime_label,
            er_bucket(float(row.get("sf_er", 0.5))),
            der_bucket(float(row.get("delta_er", 0.0))),
        )
        if not pilot_bar_matches_scope(ts=ts, session=session, ownership_cell=ownership_cell, rule=rule):
            continue

        auth = authorize_bar(
            ts_iso=ts.isoformat(),
            session=session,
            ownership_cell=ownership_cell,
            context=context,
        )
        if auth.authorized_strategy != rule.strategy:
            continue

        day_key = ts.date().isoformat()
        flags = EntryDataFlags(
            has_m1_bar=True,
            has_m5_context=bool(pd.notna(row.get("m5_slope"))),
            has_m15_context=bool(pd.notna(row.get("adx"))),
            has_h1_context=bool(pd.notna(row.get("h1_trend_dir"))),
            pivot_levels_available=bool(day_flags.get(day_key, {}).get("pivot_levels_available", False)),
            asian_range_valid=bool(day_flags.get(day_key, {}).get("asian_range_valid", False)),
            lor_valid=bool(day_flags.get(day_key, {}).get("lor_valid", False)),
        )
        elig = evaluate_shadow_entry_eligibility(strategy=auth.authorized_strategy, flags=flags)
        if not elig.can_evaluate_entry_logic:
            continue

        defensive_state = _defensive_state_at(ts, defensive_intervals)
        missing_defensive_keys = _required_defensive_state_fields() - set(defensive_state.keys())
        if missing_defensive_keys:
            defensive_state_errors.append({
                "signal_time": signal_time,
                "reason": "missing_defensive_state_fields",
                "missing_fields": sorted(missing_defensive_keys),
            })
        if (
            defensive_state["defensive_trade_count"] == 0
            and (defensive_state["defensive_long_open"] or defensive_state["defensive_short_open"])
        ):
            defensive_state_errors.append({
                "signal_time": signal_time,
                "reason": "open_flag_without_open_trade",
            })

        event = {
            "signal_time": signal_time,
            "entry_time": candidate_entry_time,
            "session": session,
            "ownership_cell": ownership_cell,
            "authorized_strategy": auth.authorized_strategy,
            "candidate_side": str(candidate.side or ""),
            "candidate_trigger_type": str(candidate.trigger_type or ""),
            "candidate_setup_type": str(candidate.raw.get("setup_type") or ""),
            "defensive_state": defensive_state,
            "outcome": "",
            "reason": None,
        }
        if prior_signal is not None and prior_signal != signal_time:
            event["outcome"] = "state_blocked"
            event["reason"] = "setupd_channel_consumed_by_earlier_setupd_long"
            event["state_consumed_by_signal_time"] = prior_signal
            events.append(event)
            continue

        if defensive_state["defensive_long_open"]:
            event["outcome"] = "defensive_conflict"
            event["reason"] = "defensive_long_already_open"
            events.append(event)
            continue

        exact_overlap = (candidate_entry_time, str(candidate.side or "")) in exact_baseline_entries
        any_overlap = candidate_entry_time in any_baseline_entries
        event["outcome"] = "pilot_candidate"
        event["exact_baseline_entry_overlap"] = exact_overlap
        event["any_baseline_entry_time_overlap"] = any_overlap
        events.append(event)

    schema = _schema_check(events)
    observed = _observed_lists(events)
    return {
        "dataset": dataset_key,
        "events": events,
        "schema_validation": schema,
        "defensive_state_errors": defensive_state_errors,
        "observed_lists": observed,
    }


def main() -> int:
    args = _parse_args()
    additive = _load_additive_reference()
    results: dict[str, Any] = {}
    selected = args.dataset_key or list(DATASETS.keys())

    for dataset_key in selected:
        dataset = DATASETS[dataset_key]
        observed = _run_dataset(dataset_key, dataset)
        additive_variant = additive["results"][dataset_key]["variants"]["100pct"]
        expected = _expected_lists(additive_variant)
        results[dataset_key] = {
            "observed_lists": observed["observed_lists"],
            "expected_lists": expected,
            "schema_validation": observed["schema_validation"],
            "defensive_state_error_count": len(observed["defensive_state_errors"]),
            "defensive_state_errors": observed["defensive_state_errors"][:20],
            "events_preview": observed["events"][:20],
            "comparison": {
                "exact_trade_list_matches": observed["observed_lists"]["exact_overlap"] == expected["exact_overlap"],
                "additive_trade_list_matches": observed["observed_lists"]["new_additive"] == expected["new_additive"],
                "schema_complete_and_parseable": observed["schema_validation"]["schema_complete_and_parseable"],
                "zero_defensive_state_errors": len(observed["defensive_state_errors"]) == 0,
            },
        }

    payload = {
        "title": "Offensive Setup D replay validation",
        "candidate": "mean_reversion/er_low/der_neg + Setup D + long-only + first_30min",
        "method": "candidate-universe gate-chain replay",
        "artifacts": {
            "additive_reference": str(ADDITIVE_REFERENCE),
        },
        "results": results,
    }
    output_path = Path(args.output)
    output_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(payload, indent=2, default=_json_default))
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
