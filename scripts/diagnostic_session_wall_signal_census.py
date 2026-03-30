#!/usr/bin/env python3
"""
Session wall signal census — how much signal is blocked by session gates?

For every bar in the dataset, this diagnostic asks:
  1. What ownership cell is this bar in?
  2. Which strategy is the stable owner?
  3. Is this bar inside the owner's validated session?
  4. If NOT — would the owner's entry logic have fired anyway?

The goal is to measure how much potential signal exists outside the current
session windows, without changing anything live.

Output:
  - Total bars by session classification (tokyo/london/ny/None)
  - Bars where a strategy owns the cell but is session-blocked
  - Of those, how many would have produced a candidate via shadow invocation
  - Breakdown by strategy, regime, session, and ownership cell
  - Sample trades with full metadata for quality inspection
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.chart_authorization import (
    AuthorizationDecision,
    default_archetype_policies,
    load_authorization_context,
)
from core.chart_entry_eligibility import EntryDataFlags, evaluate_shadow_entry_eligibility
from core.chart_shadow_contracts import (
    NO_OWNER_NOT_FEASIBLE,
    NO_OWNER_NOT_VALIDATED,
)
from core.ownership_table import cell_key, der_bucket, er_bucket, load_conservative_table
from core.phase3_integrated_engine import classify_session, load_phase3_sizing_config
from core.shadow_entry_invocation import (
    InvocationCandidate,
    build_native_invocation_registry,
    shadow_invoke_authorized_bar,
)
from scripts import backtest_merged_integrated_tokyo_london_v2_ny as merged_engine
from scripts import diagnostic_chart_authorization_loop as auth_loop
from scripts import diagnostic_shadow_entry_invocation as shadow_diag

OUT_DIR = ROOT / "research_out"
DEFAULT_OUTPUT = OUT_DIR / "diagnostic_session_wall_signal_census.json"
DEFAULT_DATASETS = [
    str(OUT_DIR / "USDJPY_M1_OANDA_500k.csv"),
    str(OUT_DIR / "USDJPY_M1_OANDA_1000k.csv"),
]
V14_CFG_PATH = OUT_DIR / "tokyo_optimized_v14_config.json"
LONDON_CFG_PATH = OUT_DIR / "v2_exp4_winner_baseline_config.json"
V44_CFG_PATH = OUT_DIR / "session_momentum_v44_base_config.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Census of signal blocked by session walls")
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


def _session_label(session: str | None) -> str:
    return session if session else "none"


def _authorize_bar_permissive(
    *,
    ts_iso: str,
    session: str | None,
    ownership_cell: str,
    table: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Determine the stable owner for a cell WITHOUT applying session validation.

    Returns the recommended_strategy from the ownership table directly,
    bypassing the session/validation/feasibility gates.
    """
    cell_info = table.get(ownership_cell)
    if cell_info is None:
        return {
            "recommended_strategy": None,
            "ownership_type": "unknown",
            "reason": "no_stable_owner",
        }
    rec = str(cell_info.get("recommended_strategy", "none"))
    if rec == "NO-TRADE":
        return {
            "recommended_strategy": None,
            "ownership_type": "stable_no_trade",
            "reason": "stable_no_trade",
        }
    return {
        "recommended_strategy": rec,
        "ownership_type": str(cell_info.get("type", "unknown")),
        "reason": "stable_owner",
    }


def _run_dataset(dataset: str) -> dict[str, Any]:
    dataset = str(Path(dataset).resolve())
    sizing_cfg = load_phase3_sizing_config()
    table = load_conservative_table(research_out=OUT_DIR)

    print(f"  Building invocation registry...")
    registry = build_native_invocation_registry(
        dataset=dataset,
        merged_engine=merged_engine,
        v14_config_path=V14_CFG_PATH,
        london_v2_config_path=LONDON_CFG_PATH,
        v44_config_path=V44_CFG_PATH,
    )

    print(f"  Loading bar frame...")
    bars = auth_loop._load_bar_frame(dataset)
    day_flags = shadow_diag._build_daily_entry_reference_flags(dataset)

    # Counters
    total_bars = 0
    bars_by_session = Counter()
    bars_with_stable_owner = 0
    bars_owner_in_session = 0
    bars_owner_blocked_by_session = 0

    # Blocked-by-session detail
    blocked_by_strategy = Counter()
    blocked_by_cell = Counter()
    blocked_by_session_label = Counter()
    blocked_by_regime = Counter()

    # Of the blocked bars, how many would have produced a candidate?
    blocked_candidate_found = 0
    blocked_candidate_by_strategy = Counter()
    blocked_candidate_by_cell = Counter()
    blocked_candidate_by_regime = Counter()
    blocked_candidate_by_session_label = Counter()

    # Candidate samples (capped)
    blocked_candidate_samples: list[dict[str, Any]] = []
    MAX_SAMPLES = 50

    # Hourly heatmap: strategy -> hour_utc -> {blocked, candidate_found}
    hourly_heatmap: dict[str, dict[int, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"blocked": 0, "candidate_found": 0})
    )

    # Day-of-week breakdown
    dow_blocked: dict[str, Counter] = defaultdict(Counter)
    dow_candidate: dict[str, Counter] = defaultdict(Counter)

    print(f"  Scanning {len(bars)} bars...")
    for _, row in bars.iterrows():
        total_bars += 1
        ts = pd.Timestamp(row["time"])
        session = classify_session(ts.to_pydatetime(), sizing_cfg)
        session_label = _session_label(session)
        bars_by_session[session_label] += 1

        regime_label = str(row.get("regime_hysteresis", "ambiguous"))
        ownership_cell = cell_key(
            regime_label,
            er_bucket(float(row.get("sf_er", 0.5))),
            der_bucket(float(row.get("delta_er", 0.0))),
        )

        # Who is the stable owner of this cell (ignoring session)?
        permissive = _authorize_bar_permissive(
            ts_iso=ts.isoformat(),
            session=session,
            ownership_cell=ownership_cell,
            table=table,
        )
        owner = permissive["recommended_strategy"]
        if owner is None:
            continue

        bars_with_stable_owner += 1

        # Is the owner validated in this session?
        policies = default_archetype_policies()
        policy = policies.get(owner)
        if policy is None:
            continue

        in_validated_session = policy.is_validated(session) and policy.is_feasible(session)

        if in_validated_session:
            bars_owner_in_session += 1
            continue

        # --- This bar has a stable owner that is BLOCKED by session walls ---
        bars_owner_blocked_by_session += 1
        blocked_by_strategy[owner] += 1
        blocked_by_cell[ownership_cell] += 1
        blocked_by_session_label[session_label] += 1
        blocked_by_regime[regime_label] += 1
        hour_utc = ts.hour
        hourly_heatmap[owner][hour_utc]["blocked"] += 1
        dow_blocked[owner][ts.day_name()] += 1

        # Would the owner's entry logic have fired?
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
        elig = evaluate_shadow_entry_eligibility(strategy=owner, flags=flags)
        if not elig.can_evaluate_entry_logic:
            continue

        inv = shadow_invoke_authorized_bar(strategy=owner, ts=ts, registry=registry)
        if not inv.candidate_found:
            continue

        # --- Candidate WOULD have fired outside its session window ---
        blocked_candidate_found += 1
        blocked_candidate_by_strategy[owner] += 1
        blocked_candidate_by_cell[ownership_cell] += 1
        blocked_candidate_by_regime[regime_label] += 1
        blocked_candidate_by_session_label[session_label] += 1
        hourly_heatmap[owner][hour_utc]["candidate_found"] += 1
        dow_candidate[owner][ts.day_name()] += 1

        candidate = inv.candidate
        if len(blocked_candidate_samples) < MAX_SAMPLES:
            raw = candidate.raw if candidate else {}
            blocked_candidate_samples.append({
                "time": ts.isoformat(),
                "session": session_label,
                "ownership_cell": ownership_cell,
                "regime": regime_label,
                "er": float(row.get("sf_er", 0.5)),
                "delta_er": float(row.get("delta_er", 0.0)),
                "owner": owner,
                "candidate_side": candidate.side if candidate else None,
                "candidate_trigger_type": candidate.trigger_type if candidate else None,
                "candidate_entry_time": candidate.entry_time if candidate else None,
                "candidate_pips": candidate.pips if candidate else None,
                "candidate_usd": candidate.usd if candidate else None,
                "candidate_exit_reason": candidate.exit_reason if candidate else None,
                "hour_utc": hour_utc,
                "day_of_week": ts.day_name(),
            })

    # Build hourly heatmap output
    heatmap_out: dict[str, list[dict[str, Any]]] = {}
    for strategy, hours in sorted(hourly_heatmap.items()):
        heatmap_out[strategy] = [
            {"hour_utc": h, **counts}
            for h, counts in sorted(hours.items())
        ]

    # Build day-of-week output
    dow_out: dict[str, dict[str, dict[str, int]]] = {}
    for strategy in set(list(dow_blocked.keys()) + list(dow_candidate.keys())):
        dow_out[strategy] = {
            "blocked": dict(dow_blocked.get(strategy, Counter())),
            "candidate_found": dict(dow_candidate.get(strategy, Counter())),
        }

    # Candidate quality summary (if we have pips data)
    candidate_quality: dict[str, dict[str, Any]] = {}
    for strategy in blocked_candidate_by_strategy:
        strat_samples = [s for s in blocked_candidate_samples if s["owner"] == strategy and s["candidate_pips"] is not None]
        if strat_samples:
            pips_list = [s["candidate_pips"] for s in strat_samples]
            wins = [p for p in pips_list if p > 0]
            losses = [p for p in pips_list if p <= 0]
            candidate_quality[strategy] = {
                "sample_count": len(strat_samples),
                "avg_pips": round(sum(pips_list) / len(pips_list), 2) if pips_list else 0,
                "total_pips": round(sum(pips_list), 2),
                "win_count": len(wins),
                "loss_count": len(losses),
                "win_rate_pct": round(100 * len(wins) / len(pips_list), 1) if pips_list else 0,
                "avg_win_pips": round(sum(wins) / len(wins), 2) if wins else 0,
                "avg_loss_pips": round(sum(losses) / len(losses), 2) if losses else 0,
            }

    # Top blocked cells by candidate count
    top_blocked_cells = sorted(
        blocked_candidate_by_cell.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:20]

    return {
        "dataset": Path(dataset).name,
        "total_bars": total_bars,
        "bars_by_session": dict(bars_by_session),
        "bars_with_stable_owner": bars_with_stable_owner,
        "bars_owner_in_session": bars_owner_in_session,
        "bars_owner_blocked_by_session": bars_owner_blocked_by_session,
        "blocked_candidate_found": blocked_candidate_found,
        "summary_rates": {
            "pct_bars_with_owner": round(100 * bars_with_stable_owner / total_bars, 2) if total_bars else 0,
            "pct_owner_bars_blocked": round(100 * bars_owner_blocked_by_session / bars_with_stable_owner, 2) if bars_with_stable_owner else 0,
            "pct_blocked_bars_with_candidate": round(100 * blocked_candidate_found / bars_owner_blocked_by_session, 2) if bars_owner_blocked_by_session else 0,
        },
        "blocked_by_strategy": dict(blocked_by_strategy),
        "blocked_by_session_label": dict(blocked_by_session_label),
        "blocked_by_regime": dict(blocked_by_regime),
        "blocked_candidate_by_strategy": dict(blocked_candidate_by_strategy),
        "blocked_candidate_by_session_label": dict(blocked_candidate_by_session_label),
        "blocked_candidate_by_regime": dict(blocked_candidate_by_regime),
        "top_blocked_cells_with_candidates": [
            {"cell": cell, "candidate_count": count}
            for cell, count in top_blocked_cells
        ],
        "candidate_quality_by_strategy": candidate_quality,
        "hourly_heatmap": heatmap_out,
        "day_of_week": dow_out,
        "samples": blocked_candidate_samples,
    }


def main() -> int:
    args = parse_args()
    out: dict[str, Any] = {
        "title": "Session wall signal census",
        "question": "How much signal is blocked by session gates? If every bar were open to its stable owner, how many candidates would fire?",
        "results": {},
    }
    for dataset in args.dataset:
        name = Path(dataset).name
        print(f"Running census for {name} ...")
        result = _run_dataset(dataset)
        out["results"][result["dataset"]] = result
        print(f"  Blocked bars: {result['bars_owner_blocked_by_session']}")
        print(f"  Candidates behind walls: {result['blocked_candidate_found']}")
        print()

    output = Path(args.output)
    output.write_text(json.dumps(out, indent=2, default=_json_default), encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
