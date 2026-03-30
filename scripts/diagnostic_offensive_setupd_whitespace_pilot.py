#!/usr/bin/env python3
"""
Narrow shadow/paper pilot replay for the first offensive Setup D candidate.

This script does not route live trades. It replays the frozen pilot rule over
historical bars, logs authorization / eligibility / candidate decisions, and
cross-checks coexistence with the current defensive stack.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
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
from core.shadow_entry_invocation import (
    build_native_invocation_registry,
    shadow_invoke_authorized_bar,
    _build_london_candidate_map,
)
from scripts import backtest_variant_k_london_cluster as variant_k
from scripts import backtest_merged_integrated_tokyo_london_v2_ny as merged_engine
from scripts import diagnostic_chart_authorization_loop as auth_loop
from scripts import diagnostic_shadow_entry_invocation as shadow_diag

OUT_DIR = ROOT / "research_out"
DEFAULT_OUTPUT = OUT_DIR / "offensive_setupd_whitespace_shadow_pilot.json"
DEFAULT_DATASETS = [
    str(OUT_DIR / "USDJPY_M1_OANDA_500k.csv"),
    str(OUT_DIR / "USDJPY_M1_OANDA_1000k.csv"),
]
V14_CFG_PATH = OUT_DIR / "tokyo_optimized_v14_config.json"
LONDON_CFG_PATH = OUT_DIR / "v2_exp4_winner_baseline_config.json"
V44_CFG_PATH = OUT_DIR / "session_momentum_v44_base_config.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Shadow replay for the offensive Setup D pilot candidate")
    p.add_argument("--dataset", nargs="+", default=DEFAULT_DATASETS)
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return p.parse_args()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)


def _baseline_trade_maps(dataset: str) -> tuple[set[tuple[str, str]], set[str], list[dict[str, Any]]]:
    dataset_path = Path(dataset).resolve()
    cache_dir = dataset_path.parent / "shadow_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"offensive_setupd_baseline_maps_{dataset_path.stem}_v1.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as fh:
            cached = pickle.load(fh)
        return cached["exact_entries"], cached["any_entry_times"], cached["intervals"]

    kept, baseline_meta, _classified, _dyn_idx, _blocked_cluster, _blocked_global = variant_k.build_variant_k_pre_coupling_kept(dataset)
    coupled = merged_engine._apply_shared_equity_coupling(
        sorted(kept, key=lambda t: (t.exit_time, t.entry_time)),
        100000.0,
        v14_max_units=baseline_meta["v14_max_units"],
    )
    exact_entries: set[tuple[str, str]] = set()
    any_entry_times: set[str] = set()
    intervals: list[dict[str, Any]] = []
    for trade in coupled:
        ts = pd.Timestamp(trade.entry_time).isoformat()
        exact_entries.add((ts, str(trade.side)))
        any_entry_times.add(ts)
        intervals.append(
            {
                "entry_time": pd.Timestamp(trade.entry_time),
                "exit_time": pd.Timestamp(trade.exit_time),
                "side": str(trade.side),
                "strategy": str(trade.strategy),
            }
        )
    with cache_path.open("wb") as fh:
        pickle.dump(
            {
                "exact_entries": exact_entries,
                "any_entry_times": any_entry_times,
                "intervals": intervals,
            },
            fh,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return exact_entries, any_entry_times, intervals


def _build_consumed_channel_days(
    dataset: str,
    rule: "OffensiveSetupDPilotRule",
) -> dict[tuple[str, str], str]:
    """Pre-scan the full london candidate universe for Setup D long first_30min signals.

    Returns a dict mapping (day_iso, direction) -> first signal_time that consumed
    the channel. This mirrors the validator logic at
    diagnostic_offensive_setupd_replay_validation.py:242-253 — any Setup D long
    in the first_30min window consumes the channel for that day, regardless of
    which ownership cell it falls in.
    """
    london_by_ts, _ = _build_london_candidate_map(
        dataset=dataset,
        config_path=LONDON_CFG_PATH,
        merged_engine=merged_engine,
    )
    fired: dict[tuple[str, str], str] = {}
    for signal_time in sorted(london_by_ts.keys()):
        candidates = london_by_ts[signal_time]
        if not candidates:
            continue
        candidate = candidates[0]
        ts = pd.Timestamp(signal_time)
        is_setupd_long_first30 = (
            str(candidate.trigger_type or "") == f"setup_{rule.setup_type}"
            and str(candidate.side or "").lower() == "buy"
            and str(candidate.raw.get("setup_type") or "") == rule.setup_type
            and str(candidate.raw.get("direction") or "").lower() == rule.direction
            and london_session_window_matches_timestamp(ts)
            and pilot_window_matches_timestamp(ts, rule)
        )
        if not is_setupd_long_first30:
            continue
        candidate_entry_time = str(candidate.entry_time or signal_time)
        channel_key = (pd.Timestamp(candidate_entry_time).date().isoformat(), rule.direction)
        if channel_key not in fired:
            fired[channel_key] = signal_time
    return fired


def _defensive_state_at(ts: pd.Timestamp, defensive_intervals: list[dict[str, Any]]) -> dict[str, Any]:
    open_trades = [t for t in defensive_intervals if t["entry_time"] <= ts < t["exit_time"]]
    return {
        "defensive_long_open": any(t["side"] == "buy" for t in open_trades),
        "defensive_short_open": any(t["side"] == "sell" for t in open_trades),
        "defensive_trade_count": len(open_trades),
        "defensive_active_strategies": sorted({t["strategy"] for t in open_trades}),
    }


def _run_dataset(dataset: str) -> dict[str, Any]:
    dataset = str(Path(dataset).resolve())
    rule = default_offensive_setupd_pilot_rule()
    context = load_authorization_context(research_out=OUT_DIR)
    sizing_cfg = load_phase3_sizing_config()

    registry = build_native_invocation_registry(
        dataset=dataset,
        merged_engine=merged_engine,
        v14_config_path=V14_CFG_PATH,
        london_v2_config_path=LONDON_CFG_PATH,
        v44_config_path=V44_CFG_PATH,
    )
    bars = auth_loop._load_bar_frame(dataset)
    day_flags = shadow_diag._build_daily_entry_reference_flags(dataset)
    exact_baseline_entries, any_baseline_entries, defensive_intervals = _baseline_trade_maps(dataset)

    # Pre-compute consumed channel days from the FULL london candidate universe.
    # Any Setup D long in first_30min consumes the channel for that day, regardless
    # of which ownership cell the signal falls in. This matches the validator logic
    # at diagnostic_offensive_setupd_replay_validation.py:242-253.
    consumed_channel_days = _build_consumed_channel_days(dataset, rule)

    counts = Counter()
    blocked_auth_reasons = Counter()
    blocked_eligibility_reasons = Counter()
    blocked_invocation_reasons = Counter()
    overlap_counts = Counter()
    by_cell = defaultdict(Counter)
    by_session = defaultdict(Counter)
    candidate_samples: list[dict[str, Any]] = []
    blocked_samples: list[dict[str, Any]] = []
    matched_rule_events: list[dict[str, Any]] = []
    defensive_state_errors: list[dict[str, Any]] = []

    for _, row in bars.iterrows():
        ts = pd.Timestamp(row["time"])
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

        counts["pilot_scope_bars"] += 1
        defensive_state = _defensive_state_at(ts, defensive_intervals)
        missing_defensive_keys = {
            "defensive_long_open",
            "defensive_short_open",
            "defensive_trade_count",
            "defensive_active_strategies",
        } - set(defensive_state.keys())
        if missing_defensive_keys:
            defensive_state_errors.append({
                "time": ts.isoformat(),
                "missing_keys": sorted(missing_defensive_keys),
                "defensive_state": defensive_state,
            })
        if (
            defensive_state["defensive_trade_count"] == 0
            and (defensive_state["defensive_long_open"] or defensive_state["defensive_short_open"])
        ):
            defensive_state_errors.append({
                "time": ts.isoformat(),
                "reason": "open_flag_without_open_trade",
                "defensive_state": defensive_state,
            })

        auth = authorize_bar(
            ts_iso=ts.isoformat(),
            session=session,
            ownership_cell=ownership_cell,
            context=context,
        )
        if auth.authorized_strategy != rule.strategy:
            counts["blocked_before_authorization"] += 1
            blocked_auth_reasons[auth.no_owner_reason or "wrong_authorized_strategy"] += 1
            if len(blocked_samples) < 30:
                blocked_samples.append({
                    "time": ts.isoformat(),
                    "session": session,
                    "ownership_cell": ownership_cell,
                    "stage": "authorization",
                    "reason": auth.no_owner_reason or "wrong_authorized_strategy",
                    "authorized_strategy": auth.authorized_strategy,
                    "recommended_strategy": auth.recommended_strategy,
                    "defensive_state": defensive_state,
                })
            continue

        counts["authorized_scope_bars"] += 1

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
            counts["blocked_before_invocation"] += 1
            blocked_eligibility_reasons[elig.reason] += 1
            if len(blocked_samples) < 30:
                blocked_samples.append({
                    "time": ts.isoformat(),
                    "session": session,
                    "ownership_cell": ownership_cell,
                    "stage": "eligibility",
                    "reason": elig.reason,
                    "authorized_strategy": auth.authorized_strategy,
                    "defensive_state": defensive_state,
                })
            continue

        counts["entry_eligible_scope_bars"] += 1

        inv = shadow_invoke_authorized_bar(strategy=auth.authorized_strategy, ts=ts, registry=registry)
        if not inv.candidate_found:
            counts["invoked_no_candidate"] += 1
            blocked_invocation_reasons[inv.reason] += 1
            if len(blocked_samples) < 30:
                blocked_samples.append({
                    "time": ts.isoformat(),
                    "session": session,
                    "ownership_cell": ownership_cell,
                    "stage": "invocation",
                    "reason": inv.reason,
                    "authorized_strategy": auth.authorized_strategy,
                    "defensive_state": defensive_state,
                })
            continue

        candidate = inv.as_dict().get("candidate")
        if not pilot_candidate_matches_rule(candidate=candidate, rule=rule):
            counts["candidate_found_but_not_pilot_rule"] += 1
            blocked_invocation_reasons["candidate_not_matching_frozen_rule"] += 1
            if len(blocked_samples) < 30:
                blocked_samples.append({
                    "time": ts.isoformat(),
                    "session": session,
                    "ownership_cell": ownership_cell,
                    "stage": "candidate_filter",
                    "reason": "candidate_not_matching_frozen_rule",
                    "candidate": candidate,
                    "defensive_state": defensive_state,
                })
            continue

        candidate_entry_time = str(candidate.get("entry_time") or "")
        candidate_day = pd.Timestamp(candidate_entry_time or ts.isoformat()).date().isoformat()
        channel_key = (candidate_day, rule.direction)
        event_base = {
            "signal_time": ts.isoformat(),
            "entry_time": candidate_entry_time,
            "session": session,
            "ownership_cell": ownership_cell,
            "authorized_strategy": auth.authorized_strategy,
            "candidate_side": str(candidate.get("side") or ""),
            "candidate_trigger_type": str(candidate.get("trigger_type") or ""),
            "candidate_setup_type": str((candidate.get("raw") or {}).get("setup_type") or ""),
            "defensive_state": defensive_state,
        }
        # Check if the channel was already consumed by an earlier Setup D long
        # in first_30min (possibly in a different ownership cell).
        prior_signal = consumed_channel_days.get(channel_key)
        if prior_signal is not None and prior_signal != ts.isoformat():
            counts["candidate_found_but_state_blocked"] += 1
            blocked_invocation_reasons["setupd_channel_consumed_by_earlier_setupd_long"] += 1
            matched_rule_events.append({
                **event_base,
                "outcome": "state_blocked",
                "reason": "setupd_channel_consumed_by_earlier_setupd_long",
                "state_consumed_by_signal_time": prior_signal,
            })
            if len(blocked_samples) < 30:
                blocked_samples.append({
                    "time": ts.isoformat(),
                    "session": session,
                    "ownership_cell": ownership_cell,
                    "stage": "state_filter",
                    "reason": "setupd_channel_consumed_by_earlier_setupd_long",
                    "state_consumed_by_signal_time": prior_signal,
                    "candidate": candidate,
                    "defensive_state": defensive_state,
                })
            continue

        if defensive_state["defensive_long_open"]:
            counts["candidate_found_but_defensive_conflict"] += 1
            blocked_invocation_reasons["defensive_long_already_open"] += 1
            matched_rule_events.append({
                **event_base,
                "outcome": "defensive_conflict",
                "reason": "defensive_long_already_open",
            })
            if len(blocked_samples) < 30:
                blocked_samples.append({
                    "time": ts.isoformat(),
                    "session": session,
                    "ownership_cell": ownership_cell,
                    "stage": "non_conflict",
                    "reason": "defensive_long_already_open",
                    "candidate": candidate,
                    "defensive_state": defensive_state,
                })
            continue

        counts["pilot_candidates"] += 1
        entry_time = candidate_entry_time
        side = str(candidate.get("side") or "")
        exact_overlap = (entry_time, side) in exact_baseline_entries
        any_overlap = entry_time in any_baseline_entries
        overlap_counts["exact_baseline_entry_overlap" if exact_overlap else "exact_baseline_entry_no_overlap"] += 1
        overlap_counts["any_baseline_entry_time_overlap" if any_overlap else "any_baseline_entry_time_no_overlap"] += 1
        matched_rule_events.append({
            **event_base,
            "outcome": "pilot_candidate",
            "reason": None,
            "exact_baseline_entry_overlap": exact_overlap,
            "any_baseline_entry_time_overlap": any_overlap,
        })
        by_cell[ownership_cell]["pilot_candidates"] += 1
        by_session[str(session)]["pilot_candidates"] += 1
        if len(candidate_samples) < 30:
            candidate_samples.append({
                "time": ts.isoformat(),
                "session": session,
                "ownership_cell": ownership_cell,
                "authorized_strategy": auth.authorized_strategy,
                "candidate": candidate,
                "defensive_state": defensive_state,
                "exact_baseline_entry_overlap": exact_overlap,
                "any_baseline_entry_time_overlap": any_overlap,
            })

    verdict = []
    if counts["pilot_scope_bars"] == 0:
        verdict.append("Frozen pilot rule never appeared in replay scope.")
    else:
        verdict.append(f"Pilot scope appeared on {counts['pilot_scope_bars']} bars.")
        verdict.append(f"Authorized on {counts['authorized_scope_bars']} scope bars.")
        verdict.append(f"Entry-eligible on {counts['entry_eligible_scope_bars']} scope bars.")
        verdict.append(f"Produced {counts['pilot_candidates']} matching pilot candidates.")
        if counts["pilot_candidates"] == 0:
            verdict.append("Current blocker is candidate scarcity inside the frozen rule.")
        else:
            overlap = overlap_counts["exact_baseline_entry_overlap"]
            if overlap > 0:
                verdict.append(f"{overlap} candidates exactly overlapped existing defensive entries.")
            else:
                verdict.append("No pilot candidate exactly overlapped an existing defensive entry.")
            if counts.get("candidate_found_but_defensive_conflict", 0) > 0:
                verdict.append(
                    f"{counts['candidate_found_but_defensive_conflict']} matching candidates were blocked by an already-open defensive long."
                )

    return {
        "dataset": Path(dataset).name,
        "rule": {
            "name": rule.name,
            "strategy": rule.strategy,
            "setup_type": rule.setup_type,
            "direction": rule.direction,
            "ownership_cell": rule.ownership_cell,
            "required_session": rule.required_session,
            "min_minutes_after_london_open": rule.min_minutes_after_london_open,
            "max_minutes_after_london_open": rule.max_minutes_after_london_open,
        },
        "summary": dict(counts),
        "blocked_authorization_reasons": dict(blocked_auth_reasons),
        "blocked_eligibility_reasons": dict(blocked_eligibility_reasons),
        "blocked_invocation_reasons": dict(blocked_invocation_reasons),
        "overlap_counts": dict(overlap_counts),
        "by_cell": {k: dict(v) for k, v in by_cell.items()},
        "by_session": {k: dict(v) for k, v in by_session.items()},
        "samples": {
            "pilot_candidates": candidate_samples,
            "blocked": blocked_samples,
        },
        "matched_rule_events": matched_rule_events,
        "defensive_state_errors": defensive_state_errors,
        "verdict": " ".join(verdict),
    }


def main() -> int:
    args = parse_args()
    out = {
        "title": "Offensive Setup D whitespace shadow pilot replay",
        "results": {},
    }
    for dataset in args.dataset:
        print(f"Running pilot replay for {Path(dataset).name} ...")
        result = _run_dataset(dataset)
        out["results"][result["dataset"]] = result

    output = Path(args.output)
    output.write_text(json.dumps(out, indent=2, default=_json_default), encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
