#!/usr/bin/env python3
"""
Shadow entry-engine invocation replay.

For bars where an archetype is:
  - chart-authorized
  - validated
  - feasible
  - entry-eligible

this diagnostic asks whether the real native strategy engine emitted an entry
candidate on that same bar timestamp in its standalone run.
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

from core.chart_authorization import authorize_bar, load_authorization_context
from core.chart_entry_eligibility import EntryDataFlags, evaluate_shadow_entry_eligibility
from core.ownership_table import cell_key, der_bucket, er_bucket
from core.phase3_integrated_engine import classify_session, compute_asian_range, load_phase3_sizing_config
from core.shadow_entry_invocation import build_native_invocation_registry, shadow_invoke_authorized_bar
from scripts import backtest_merged_integrated_tokyo_london_v2_ny as merged_engine
from scripts import backtest_v2_multisetup_london as london_v2
from scripts import diagnostic_chart_authorization_loop as auth_loop

OUT_DIR = ROOT / "research_out"
DEFAULT_DATASET = str(OUT_DIR / "USDJPY_M1_OANDA_500k.csv")
DEFAULT_OUTPUT = str(OUT_DIR / "diagnostic_shadow_entry_invocation_500k.json")
V14_CFG_PATH = OUT_DIR / "tokyo_optimized_v14_config.json"
LONDON_CFG_PATH = OUT_DIR / "v2_exp4_winner_baseline_config.json"
V44_CFG_PATH = OUT_DIR / "session_momentum_v44_base_config.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Shadow entry-engine invocation diagnostic")
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    return p.parse_args()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)


def _build_daily_entry_reference_flags(dataset: str) -> dict[str, dict[str, bool]]:
    m1 = auth_loop.regime_validation._load_m1(dataset)
    m1["time"] = pd.to_datetime(m1["time"], utc=True, errors="coerce")
    m1 = m1.dropna(subset=["time"]).sort_values("time")
    if m1.empty:
        return {}
    dates = sorted({pd.Timestamp(ts).date().isoformat() for ts in m1["time"]})
    first_day = dates[0]
    out: dict[str, dict[str, bool]] = {}
    for day in dates:
        day_ts = pd.Timestamp(day, tz="UTC")
        day_df = m1[m1["time"].dt.date == day_ts.date()].copy()
        london_open = london_v2.uk_london_open_utc(day_ts)
        _hi, _lo, _pips, asian_valid = compute_asian_range(day_df, london_open)
        lor_start = day_ts + pd.Timedelta(hours=london_open)
        lor_end = lor_start + pd.Timedelta(minutes=15)
        lor_w = day_df[(day_df["time"] >= lor_start) & (day_df["time"] < lor_end)]
        lor_valid = False
        if not lor_w.empty:
            lor_high = float(lor_w["high"].max())
            lor_low = float(lor_w["low"].min())
            lor_pips = (lor_high - lor_low) / 0.01
            lor_valid = 4.0 <= lor_pips <= 20.0
        out[day] = {
            "pivot_levels_available": day != first_day,
            "asian_range_valid": bool(asian_valid),
            "lor_valid": bool(lor_valid),
        }
    return out


def main() -> int:
    args = parse_args()
    dataset = str(Path(args.dataset).resolve())
    output = Path(args.output)

    print("Building native engine invocation registry...")
    registry = build_native_invocation_registry(
        dataset=dataset,
        merged_engine=merged_engine,
        v14_config_path=V14_CFG_PATH,
        london_v2_config_path=LONDON_CFG_PATH,
        v44_config_path=V44_CFG_PATH,
    )

    print("Building bar-level authorization frame...")
    bars = auth_loop._load_bar_frame(dataset)
    day_flags = _build_daily_entry_reference_flags(dataset)
    sizing_cfg = load_phase3_sizing_config()
    context = load_authorization_context(research_out=OUT_DIR)

    total_bars = len(bars)
    auth_counts = Counter()
    eligible_counts = Counter()
    invoke_counts = Counter()
    by_strategy = defaultdict(Counter)
    by_session = defaultdict(Counter)
    by_cell = defaultdict(Counter)
    candidate_samples: list[dict[str, Any]] = []
    no_candidate_samples: list[dict[str, Any]] = []
    not_invoked_samples: list[dict[str, Any]] = []

    for _, row in bars.iterrows():
        ts = pd.Timestamp(row["time"])
        session = classify_session(ts.to_pydatetime(), sizing_cfg)
        regime_label = str(row.get("regime_hysteresis", "ambiguous"))
        cell = cell_key(
            regime_label,
            er_bucket(float(row.get("sf_er", 0.5))),
            der_bucket(float(row.get("delta_er", 0.0))),
        )

        decision = authorize_bar(
            ts_iso=ts.isoformat(),
            session=session,
            ownership_cell=cell,
            context=context,
        )
        auth_key = str(decision.authorized_strategy or "NO_OWNER")
        auth_counts[auth_key] += 1

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
        entry_eval = evaluate_shadow_entry_eligibility(strategy=decision.authorized_strategy, flags=flags)
        if decision.authorized_strategy:
            eligible_counts["authorized_total"] += 1
        if decision.authorized_strategy and entry_eval.can_evaluate_entry_logic:
            eligible_counts["authorized_and_entry_eligible"] += 1
            inv = shadow_invoke_authorized_bar(
                strategy=decision.authorized_strategy,
                ts=ts,
                registry=registry,
            )
        else:
            inv = shadow_invoke_authorized_bar(strategy=None, ts=ts, registry=registry)

        invoke_counts[inv.reason] += 1
        strategy_key = str(decision.authorized_strategy or "NO_OWNER")
        by_strategy[strategy_key][inv.reason] += 1
        by_session[str(session or "dead")][inv.reason] += 1
        by_cell[cell][inv.reason] += 1

        row_out = {
            "time": ts.isoformat(),
            "session": session,
            "ownership_cell": cell,
            "authorized_strategy": decision.authorized_strategy,
            "entry_can_evaluate": bool(entry_eval.can_evaluate_entry_logic),
            "entry_eligibility_reason": entry_eval.reason,
            "invocation_reason": inv.reason,
            "candidate": inv.as_dict().get("candidate"),
        }
        if inv.candidate_found:
            candidate_samples.append(row_out)
        elif inv.reason == "invoked_no_candidate" and len(no_candidate_samples) < 30:
            no_candidate_samples.append(row_out)
        elif inv.reason == "no_authorized_strategy" and len(not_invoked_samples) < 30:
            not_invoked_samples.append(row_out)

    summary = {
        "total_bars": total_bars,
        "authorized_counts": dict(auth_counts),
        "authorized_and_entry_eligible": int(eligible_counts["authorized_and_entry_eligible"]),
        "invocation_counts": dict(invoke_counts),
        "candidate_produced": int(invoke_counts.get("candidate_found", 0)),
        "candidate_rate_of_eligible_pct": round(
            100.0 * invoke_counts.get("candidate_found", 0) / max(1, eligible_counts["authorized_and_entry_eligible"]),
            2,
        ),
        "registry_counts_by_strategy": registry.counts_by_strategy,
        "source_mode_by_strategy": registry.source_mode_by_strategy,
    }

    top_cells = [
        {
            "cell": cell,
            "counts": dict(counter),
        }
        for cell, counter in sorted(
            by_cell.items(),
            key=lambda kv: kv[1].get("candidate_found", 0),
            reverse=True,
        )[:20]
    ]

    verdict_lines = []
    if summary["authorized_and_entry_eligible"] == 0:
        verdict_lines.append("No bars reached authorized + entry-eligible status.")
        verdict_lines.append("Current bottleneck remains validation/feasibility before invocation.")
    elif summary["candidate_produced"] == 0:
        verdict_lines.append("Invocation was attempted on eligible bars but produced zero candidates.")
        verdict_lines.append("Current bottleneck is real candidate scarcity, not authorization.")
    else:
        verdict_lines.append(
            f"Invocation produced {summary['candidate_produced']} candidates from "
            f"{summary['authorized_and_entry_eligible']} authorized+eligible bars."
        )
        if summary["candidate_rate_of_eligible_pct"] < 1.0:
            verdict_lines.append("Candidate scarcity is still the main bottleneck.")
        else:
            verdict_lines.append("Invocation layer is producing real candidates often enough to study deeper.")

    payload = {
        "dataset": dataset,
        "summary": summary,
        "by_strategy": {k: dict(v) for k, v in by_strategy.items()},
        "by_session": {k: dict(v) for k, v in by_session.items()},
        "top_cells": top_cells,
        "samples": {
            "candidate_found": candidate_samples,
            "invoked_no_candidate": no_candidate_samples,
            "not_invoked": not_invoked_samples,
        },
        "verdict": " ".join(verdict_lines),
    }

    output.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    print(f"Total bars: {summary['total_bars']}")
    print(f"Authorized+eligible: {summary['authorized_and_entry_eligible']}")
    print(f"Invocation counts: {summary['invocation_counts']}")
    print(f"Candidate produced: {summary['candidate_produced']}")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
