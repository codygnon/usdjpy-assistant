#!/usr/bin/env python3
"""
Replay the current chart-authorization scaffold on every bar of a dataset.

This is not a trade backtest. It is a bar-by-bar authorization audit:
  - classify chart state on every bar
  - compute ownership cell
  - ask which archetypes are chart-eligible, validated, and feasible
  - authorize one archetype or no owner

Outputs a JSON artifact summarizing how often the current evidence would
authorize each archetype under strict current-validation rules.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import pickle

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.chart_authorization import authorize_bar, load_authorization_context
from core.chart_entry_eligibility import EntryDataFlags, evaluate_shadow_entry_eligibility
from core.ownership_table import cell_key, der_bucket, er_bucket
from core.phase3_integrated_engine import classify_session, compute_asian_range, load_phase3_sizing_config
from core.regime_features import compute_delta_efficiency_ratio, compute_efficiency_ratio
from core.regime_classifier import RegimeThresholds
from scripts import backtest_v2_multisetup_london as london_v2
from scripts import validate_regime_classifier as regime_validation

OUT_DIR = ROOT / "research_out"
DEFAULT_DATASET = str(OUT_DIR / "USDJPY_M1_OANDA_500k.csv")
DEFAULT_OUTPUT = str(OUT_DIR / "diagnostic_chart_authorization_loop_500k.json")
BAR_FRAME_CACHE_VERSION = "v1"
DAY_FLAGS_CACHE_VERSION = "v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chart authorization loop diagnostic")
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    return p.parse_args()


def _dataset_key(path: str) -> str:
    name = Path(path).name
    if "500k" in name:
        return "500k"
    if "1000k" in name:
        return "1000k"
    return name


def _json_default(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)


def _dataset_cache_dir(dataset: str) -> Path:
    cache_dir = Path(dataset).resolve().parent / "shadow_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _load_bar_frame(dataset: str) -> pd.DataFrame:
    cache_path = _dataset_cache_dir(dataset) / f"bar_frame_{Path(dataset).stem}_{BAR_FRAME_CACHE_VERSION}.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as fh:
            cached = pickle.load(fh)
        if isinstance(cached, pd.DataFrame):
            return cached

    m1 = regime_validation._load_m1(dataset)
    featured = regime_validation.compute_features(m1)
    classified = regime_validation.classify_all_bars(featured, RegimeThresholds())

    m5 = (
        m1.set_index("time")
        .resample("5min", label="right", closed="right")
        .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
        .dropna()
        .reset_index()
    )

    er_vals: list[float] = []
    der_vals: list[float] = []
    for i in range(len(m5)):
        window = m5.iloc[: i + 1]
        er_vals.append(float(compute_efficiency_ratio(window, lookback=12)))
        der_vals.append(float(compute_delta_efficiency_ratio(window, lookback=12, delta_bars=3)))
    m5["sf_er"] = er_vals
    m5["delta_er"] = der_vals

    merged = pd.merge_asof(
        classified.sort_values("time"),
        m5[["time", "sf_er", "delta_er"]].sort_values("time"),
        on="time",
        direction="backward",
    )
    merged["sf_er"] = merged["sf_er"].fillna(0.5)
    merged["delta_er"] = merged["delta_er"].fillna(0.0)
    with cache_path.open("wb") as fh:
        pickle.dump(merged, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return merged


def _build_daily_entry_reference_flags(dataset: str) -> dict[str, dict[str, bool]]:
    cache_path = _dataset_cache_dir(dataset) / f"day_flags_{Path(dataset).stem}_{DAY_FLAGS_CACHE_VERSION}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    m1 = regime_validation._load_m1(dataset)
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
    cache_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    auth_counts = Counter(r["authorized_strategy"] or "NO_OWNER" for r in rows)
    session_counts = Counter(r["session"] or "dead" for r in rows)
    reason_counts = Counter(r["no_owner_reason"] or "authorized" for r in rows)
    entry_reason_counts = Counter(r.get("entry_eligibility_reason", "not_checked") for r in rows)
    cell_counts = Counter(r["ownership_cell"] for r in rows)
    entry_ok = sum(1 for r in rows if r.get("entry_can_evaluate") is True)
    entry_blocked = sum(
        1
        for r in rows
        if r.get("authorized_strategy") is not None and r.get("entry_can_evaluate") is False
    )

    return {
        "total_bars": total,
        "authorized_counts": dict(auth_counts),
        "authorized_pct": {
            k: round(v / total * 100.0, 2) if total else 0.0
            for k, v in auth_counts.items()
        },
        "session_counts": dict(session_counts),
        "no_owner_reasons": dict(reason_counts),
        "entry_shadow": {
            "authorized_and_entry_eligible": int(entry_ok),
            "authorized_but_entry_ineligible": int(entry_blocked),
            "entry_eligibility_reasons": dict(entry_reason_counts),
        },
        "top_cells": [
            {"cell": cell, "count": count}
            for cell, count in cell_counts.most_common(20)
        ],
    }


def main() -> int:
    args = parse_args()
    dataset = str(Path(args.dataset).resolve())
    output = Path(args.output)

    print(f"Building bar-level authorization replay for {_dataset_key(dataset)}...")
    bars = _load_bar_frame(dataset)
    day_flags = _build_daily_entry_reference_flags(dataset)
    sizing_cfg = load_phase3_sizing_config()
    context = load_authorization_context(research_out=OUT_DIR)

    rows: list[dict[str, Any]] = []
    by_session_auth: dict[str, Counter] = defaultdict(Counter)

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
        d = decision.as_dict()
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
        d["entry_can_evaluate"] = bool(entry_eval.can_evaluate_entry_logic)
        d["entry_eligibility_reason"] = str(entry_eval.reason)
        d["entry_data_flags"] = entry_eval.as_dict()["data_flags"]
        rows.append(d)
        by_session_auth[str(session or "dead")][str(decision.authorized_strategy or "NO_OWNER")] += 1

    payload = {
        "dataset": dataset,
        "mode": "strict_current_validation",
        "summary": _summarize(rows),
        "by_session_authorization": {
            sess: dict(counter) for sess, counter in by_session_auth.items()
        },
        "samples": {
            "authorized": [r for r in rows if r["authorized_strategy"] is not None][:30],
            "authorized_but_entry_ineligible": [
                r for r in rows
                if r["authorized_strategy"] is not None and not r["entry_can_evaluate"]
            ][:30],
            "blocked_by_validation": [r for r in rows if r["no_owner_reason"] == "owner_not_validated_in_session"][:30],
            "stable_no_trade": [r for r in rows if r["no_owner_reason"] == "stable_no_trade"][:30],
            "no_stable_owner": [r for r in rows if r["no_owner_reason"] == "no_stable_owner"][:30],
        },
    }

    output.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    s = payload["summary"]
    print(f"Total bars: {s['total_bars']}")
    print(f"Authorized counts: {s['authorized_counts']}")
    print(f"No-owner reasons: {s['no_owner_reasons']}")
    print(f"Entry shadow: {s['entry_shadow']}")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
