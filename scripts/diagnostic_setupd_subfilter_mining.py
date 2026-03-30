#!/usr/bin/env python3
"""
Subfilter mining for London Setup D top cells.

Target cells: ambiguous/er_low/der_pos, breakout/er_low/der_pos
Goal: find narrow observable subfilters that isolate higher-quality subsets.

Features captured per trade:
  - minutes_since_london_open: signal time minus London open
  - lor_range_pips: LOR range for that day
  - breakout_distance_pips: close - (lor_high + buffer) for longs
  - sf_er: raw efficiency ratio (continuous, not bucketed)
  - delta_er: raw delta ER (continuous, not bucketed)
  - day_of_week: 0=Mon .. 4=Fri
  - asian_range_pips: Asian session range
  - risk_pips: SL distance

This is research/diagnostic only -- no live code changes.
"""
from __future__ import annotations

import json
import pickle
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.london_v2_entry_evaluator import evaluate_london_setup_d
from core.ownership_table import cell_key, der_bucket, er_bucket
from core.phase3_integrated_engine import classify_session, load_phase3_sizing_config
from core.regime_features import compute_delta_efficiency_ratio, compute_efficiency_ratio
from core.regime_classifier import RegimeThresholds
from scripts import backtest_v2_multisetup_london as london_v2
from scripts import validate_regime_classifier as regime_validation
from scripts.diagnostic_london_setupd_trade_outcomes import (
    _load_bar_frame,
    _ny_open_utc_hour,
    simulate_trade,
)

PIP_SIZE = 0.01
OUT_DIR = ROOT / "research_out"
OUTPUT_MD = OUT_DIR / "setupd_top_cells_subfilter_memo.md"
OUTPUT_JSON = OUT_DIR / "setupd_top_cells_subfilter_memo.json"

TARGET_CELLS = ["ambiguous/er_low/der_pos", "breakout/er_low/der_pos"]

DATASETS = [
    str(OUT_DIR / "USDJPY_M1_OANDA_500k.csv"),
    str(OUT_DIR / "USDJPY_M1_OANDA_1000k.csv"),
]

TRADE_ARGS = SimpleNamespace(
    spread_pips=0.3,
    sl_buffer_pips=3.0,
    sl_min_pips=5.0,
    sl_max_pips=20.0,
    tp1_r_multiple=1.0,
    tp2_r_multiple=2.0,
    tp1_close_fraction=0.5,
    be_offset_pips=1.0,
)


def _dataset_key(path: str) -> str:
    name = Path(path).name
    if "500k" in name:
        return "500k"
    if "1000k" in name:
        return "1000k"
    return name


def _load_london_config() -> dict[str, Any]:
    cfg_path = OUT_DIR / "v2_exp4_winner_baseline_config.json"
    with cfg_path.open() as f:
        return json.load(f)


def _metrics(pnls: list[float]) -> dict[str, Any]:
    if not pnls:
        return {"count": 0, "avg_pips": 0.0, "total_pips": 0.0, "win_rate": 0.0, "pf": 0.0, "median_pips": 0.0}
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gp = sum(wins) if wins else 0.0
    gl = abs(sum(losses)) if losses else 0.0
    pf = round(gp / gl, 3) if gl > 0 else (float("inf") if gp > 0 else 0.0)
    return {
        "count": len(pnls),
        "avg_pips": round(statistics.mean(pnls), 3),
        "total_pips": round(sum(pnls), 2),
        "win_rate": round(100.0 * len(wins) / len(pnls), 2),
        "pf": pf,
        "median_pips": round(statistics.median(pnls), 2),
    }


def collect_trades(dataset: str) -> list[dict[str, Any]]:
    """
    Run trade simulation for the two target cells only, capturing extra features.
    Modeled on diagnostic_london_setupd_trade_outcomes.run_dataset.
    """
    print(f"\n{'='*60}")
    print(f"Collecting enriched trades: {Path(dataset).name}")
    print(f"{'='*60}")

    cfg = _load_london_config()
    cfg_d = cfg["setups"]["D"]

    native_allow_long = bool(cfg_d.get("allow_long", True))

    m1_raw = regime_validation._load_m1(dataset)
    m1_raw["time"] = pd.to_datetime(m1_raw["time"], utc=True, errors="coerce")
    m1_raw = m1_raw.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    bars = _load_bar_frame(dataset)
    bars["time"] = pd.to_datetime(bars["time"], utc=True, errors="coerce")
    bars = bars.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    asian_min = float(cfg["levels"]["asian_range_min_pips"])
    asian_max = float(cfg["levels"]["asian_range_max_pips"])
    lor_min = float(cfg["levels"]["lor_range_min_pips"])
    lor_max = float(cfg["levels"]["lor_range_max_pips"])

    m1_raw["day_utc"] = m1_raw["time"].dt.floor("D")
    bars["day_utc"] = bars["time"].dt.floor("D")

    m1_by_day = {k: g.copy().reset_index(drop=True) for k, g in m1_raw.groupby("day_utc")}
    bars_by_day = {k: g.copy().reset_index(drop=True) for k, g in bars.groupby("day_utc")}

    breakout_buffer = float(cfg_d["breakout_buffer_pips"]) * PIP_SIZE

    all_trades: list[dict[str, Any]] = []

    for day in sorted(m1_by_day.keys()):
        day_m1 = m1_by_day[day]
        if day_m1.empty:
            continue

        london_h = london_v2.uk_london_open_utc(day)
        ny_h = london_v2.us_ny_open_utc(day)
        london_open = day + pd.Timedelta(hours=london_h)
        ny_open = day + pd.Timedelta(hours=ny_h)
        ny_open_hard_close_h = _ny_open_utc_hour(day)
        ny_open_hard_close = day + pd.Timedelta(hours=ny_open_hard_close_h)

        # Asian range
        asian = day_m1[(day_m1["time"] >= day) & (day_m1["time"] < london_open)]
        if asian.empty:
            continue
        asian_high = float(asian["high"].max())
        asian_low = float(asian["low"].min())
        asian_range_pips = (asian_high - asian_low) / PIP_SIZE
        asian_valid = asian_min <= asian_range_pips <= asian_max

        # LOR range
        lor_end_time = london_open + pd.Timedelta(minutes=15)
        lor = day_m1[(day_m1["time"] >= london_open) & (day_m1["time"] < lor_end_time)]
        if lor.empty:
            continue
        lor_high = float(lor["high"].max())
        lor_low = float(lor["low"].min())
        lor_range_pips = (lor_high - lor_low) / PIP_SIZE
        lor_valid = lor_min <= lor_range_pips <= lor_max

        if not lor_valid:
            continue

        # Setup D window
        d_start = london_open + pd.Timedelta(minutes=int(cfg_d["entry_start_min_after_london"]))
        d_end_raw = london_open + pd.Timedelta(minutes=int(cfg_d["entry_end_min_after_london"]))
        d_end = min(d_end_raw, ny_open)

        # Channel state (realistic)
        channel_state = {"long": "ARMED", "short": "ARMED"}
        channel_entries = {"long": 0, "short": 0}
        pending_direction: str | None = None

        day_bars = bars_by_day.get(day)
        if day_bars is None or day_bars.empty:
            continue

        window_bars = day_m1[(day_m1["time"] >= d_start) & (day_m1["time"] < d_end)]
        if window_bars.empty:
            continue

        for idx in range(len(window_bars)):
            row = window_bars.iloc[idx]
            ts = pd.Timestamp(row["time"])

            # Process pending
            if pending_direction is not None:
                channel_state[pending_direction] = "FIRED"
                channel_entries[pending_direction] += 1
                pending_direction = None

            # Channel reset
            for direction in ["long", "short"]:
                if channel_state[direction] != "WAITING_RESET":
                    continue
                if direction == "long" and float(row["low"]) <= lor_high:
                    channel_state[direction] = "ARMED"
                if direction == "short" and float(row["high"]) >= lor_low:
                    channel_state[direction] = "ARMED"

            if idx + 1 >= len(window_bars):
                continue
            nxt_ts = pd.Timestamp(window_bars.iloc[idx + 1]["time"])

            # Regime cell + raw features
            bar_match = day_bars[day_bars["time"] == ts]
            if bar_match.empty:
                bar_match = day_bars.iloc[(day_bars["time"] - ts).abs().argsort()[:1]]

            regime_label = str(bar_match.iloc[0].get("regime_hysteresis", "ambiguous")) if not bar_match.empty else "ambiguous"
            sf_er = float(bar_match.iloc[0].get("sf_er", 0.5)) if not bar_match.empty else 0.5
            delta_er = float(bar_match.iloc[0].get("delta_er", 0.0)) if not bar_match.empty else 0.0
            cell = cell_key(regime_label, er_bucket(sf_er), der_bucket(delta_er))

            if cell not in TARGET_CELLS:
                continue

            # Evaluate with realistic channel state (long-only for native)
            cfg_d_eval = dict(cfg_d)
            cfg_d_eval["allow_long"] = True
            cfg_d_eval["allow_short"] = False  # only long per native config

            realistic_results = evaluate_london_setup_d(
                row=row,
                cfg_d=cfg_d_eval,
                lor_high=lor_high,
                lor_low=lor_low,
                asian_range_pips=asian_range_pips if asian_valid else None,
                lor_range_pips=lor_range_pips,
                lor_valid=lor_valid,
                asian_valid=asian_valid,
                ts=ts,
                nxt_ts=nxt_ts,
                d_start=d_start,
                d_end=d_end,
                channel_long_state=channel_state["long"],
                channel_short_state=channel_state["short"],
                channel_long_entries=channel_entries["long"],
                channel_short_entries=channel_entries["short"],
            )

            if not realistic_results:
                continue

            for entry_sig in realistic_results:
                direction = entry_sig["direction"]

                signal_abs_idx_matches = day_m1.index[day_m1["time"] == ts]
                if len(signal_abs_idx_matches) == 0:
                    continue
                signal_abs_idx = signal_abs_idx_matches[0]

                trade = simulate_trade(
                    direction=direction,
                    signal_bar_idx=signal_abs_idx,
                    m1_day=day_m1,
                    lor_high=lor_high,
                    lor_low=lor_low,
                    ny_open_time=ny_open_hard_close,
                    spread_pips=TRADE_ARGS.spread_pips,
                    sl_buffer_pips=TRADE_ARGS.sl_buffer_pips,
                    sl_min_pips=TRADE_ARGS.sl_min_pips,
                    sl_max_pips=TRADE_ARGS.sl_max_pips,
                    tp1_r_multiple=TRADE_ARGS.tp1_r_multiple,
                    tp2_r_multiple=TRADE_ARGS.tp2_r_multiple,
                    tp1_close_fraction=TRADE_ARGS.tp1_close_fraction,
                    be_offset_pips=TRADE_ARGS.be_offset_pips,
                )

                if trade is None:
                    continue

                # Enriched features
                close = float(row["close"])
                minutes_since_london = (ts - london_open).total_seconds() / 60.0
                breakout_dist_pips = (close - (lor_high + breakout_buffer)) / PIP_SIZE

                trade["ownership_cell"] = cell
                trade["regime_label"] = regime_label
                trade["sf_er"] = round(sf_er, 5)
                trade["delta_er"] = round(delta_er, 5)
                trade["minutes_since_london_open"] = round(minutes_since_london, 1)
                trade["lor_range_pips"] = round(lor_range_pips, 2)
                trade["breakout_distance_pips"] = round(breakout_dist_pips, 2)
                trade["asian_range_pips"] = round(asian_range_pips, 2)
                trade["day_of_week"] = ts.dayofweek  # 0=Mon .. 4=Fri
                trade["native_allowed"] = native_allow_long if direction == "long" else False

                all_trades.append(trade)

                channel_state[direction] = "PENDING"
                pending_direction = direction

            for direction in ["long", "short"]:
                if channel_state[direction] == "FIRED":
                    channel_state[direction] = "WAITING_RESET"

    print(f"  Target-cell trades collected: {len(all_trades)}")
    return all_trades


# ── Filter definitions ──────────────────────────────────────────────────────

def _define_filters() -> list[dict[str, Any]]:
    """
    Each filter is a dict with:
      name: human-readable label
      fn: callable(trade) -> bool  (True = passes filter)
      rationale: why we test this
    """
    filters = []

    # 1. Time-in-window filters
    filters.append({
        "name": "first_30min (15-45 after London)",
        "fn": lambda t: t["minutes_since_london_open"] <= 45,
        "rationale": "Early momentum is strongest right after LOR forms",
    })
    filters.append({
        "name": "first_45min (15-60 after London)",
        "fn": lambda t: t["minutes_since_london_open"] <= 60,
        "rationale": "First hour: before mean-reversion pressure builds",
    })
    filters.append({
        "name": "first_60min (15-75 after London)",
        "fn": lambda t: t["minutes_since_london_open"] <= 75,
        "rationale": "First 75 minutes from London open",
    })
    filters.append({
        "name": "after_45min (>45 after London)",
        "fn": lambda t: t["minutes_since_london_open"] > 45,
        "rationale": "Later entries: extended trend confirmation",
    })

    # 2. LOR range filters
    filters.append({
        "name": "lor_range_tight (4-8 pips)",
        "fn": lambda t: t["lor_range_pips"] <= 8,
        "rationale": "Tighter LOR = tighter SL = better R:R if breakout works",
    })
    filters.append({
        "name": "lor_range_mid (8-14 pips)",
        "fn": lambda t: 8 < t["lor_range_pips"] <= 14,
        "rationale": "Medium LOR range",
    })
    filters.append({
        "name": "lor_range_wide (>14 pips)",
        "fn": lambda t: t["lor_range_pips"] > 14,
        "rationale": "Wide LOR = bigger SL, harder to hit R:R",
    })

    # 3. Breakout distance filters
    filters.append({
        "name": "breakout_small (<5 pips past buffer)",
        "fn": lambda t: t["breakout_distance_pips"] < 5,
        "rationale": "Close to breakout level = less chasing",
    })
    filters.append({
        "name": "breakout_moderate (5-10 pips past buffer)",
        "fn": lambda t: 5 <= t["breakout_distance_pips"] < 10,
        "rationale": "Moderate breakout extension",
    })
    filters.append({
        "name": "breakout_large (>=10 pips past buffer)",
        "fn": lambda t: t["breakout_distance_pips"] >= 10,
        "rationale": "Big breakout bar: strong momentum or overextension?",
    })
    filters.append({
        "name": "breakout_max8 (<8 pips past buffer)",
        "fn": lambda t: t["breakout_distance_pips"] < 8,
        "rationale": "Excludes most overextended entries",
    })

    # 4. Raw ER filters (within er_low bucket: er < 0.35)
    filters.append({
        "name": "er_very_low (<0.20)",
        "fn": lambda t: t["sf_er"] < 0.20,
        "rationale": "Very low ER = choppy, maybe avoid",
    })
    filters.append({
        "name": "er_low_upper (0.20-0.35)",
        "fn": lambda t: 0.20 <= t["sf_er"] < 0.35,
        "rationale": "Upper half of er_low bucket: slightly more directional",
    })

    # 5. Raw delta_er filters (within der_pos: delta_er >= 0)
    filters.append({
        "name": "der_strong (delta_er > 0.05)",
        "fn": lambda t: t["delta_er"] > 0.05,
        "rationale": "Stronger positive momentum shift",
    })
    filters.append({
        "name": "der_moderate (0 <= delta_er <= 0.05)",
        "fn": lambda t: 0 <= t["delta_er"] <= 0.05,
        "rationale": "Barely positive delta ER",
    })
    filters.append({
        "name": "der_very_strong (delta_er > 0.10)",
        "fn": lambda t: t["delta_er"] > 0.10,
        "rationale": "Strong improving directional regime",
    })

    # 6. Day-of-week filters
    for dow, name in [(0, "Monday"), (1, "Tuesday"), (2, "Wednesday"), (3, "Thursday"), (4, "Friday")]:
        filters.append({
            "name": f"dow_{name}",
            "fn": (lambda d: lambda t: t["day_of_week"] == d)(dow),
            "rationale": f"Day-of-week: {name}",
        })
    filters.append({
        "name": "dow_tue_wed_thu",
        "fn": lambda t: t["day_of_week"] in (1, 2, 3),
        "rationale": "Mid-week: avoids Monday positioning / Friday unwind",
    })

    # 7. Asian range filters
    filters.append({
        "name": "asian_range_narrow (30-42 pips)",
        "fn": lambda t: t["asian_range_pips"] <= 42,
        "rationale": "Narrow Asian range: more room for London breakout",
    })
    filters.append({
        "name": "asian_range_wide (>42 pips)",
        "fn": lambda t: t["asian_range_pips"] > 42,
        "rationale": "Wide Asian: already-extended market",
    })

    # 8. Risk (SL distance) filters
    filters.append({
        "name": "risk_tight (<=8 pips SL)",
        "fn": lambda t: t["risk_pips"] <= 8,
        "rationale": "Tight SL = better R:R",
    })
    filters.append({
        "name": "risk_wide (>12 pips SL)",
        "fn": lambda t: t["risk_pips"] > 12,
        "rationale": "Wide SL: harder to reach 1R/2R",
    })

    # 9. Combination filters
    filters.append({
        "name": "combo: first_45min + breakout_max8",
        "fn": lambda t: t["minutes_since_london_open"] <= 60 and t["breakout_distance_pips"] < 8,
        "rationale": "Early + not overextended",
    })
    filters.append({
        "name": "combo: first_45min + lor_tight",
        "fn": lambda t: t["minutes_since_london_open"] <= 60 and t["lor_range_pips"] <= 10,
        "rationale": "Early + tight LOR = best R:R profile",
    })
    filters.append({
        "name": "combo: first_45min + der_strong",
        "fn": lambda t: t["minutes_since_london_open"] <= 60 and t["delta_er"] > 0.05,
        "rationale": "Early + strong momentum improvement",
    })
    filters.append({
        "name": "combo: lor_tight + breakout_small",
        "fn": lambda t: t["lor_range_pips"] <= 10 and t["breakout_distance_pips"] < 5,
        "rationale": "Tight range + close to level = cleanest breakout",
    })
    filters.append({
        "name": "combo: mid-week + first_45min",
        "fn": lambda t: t["day_of_week"] in (1, 2, 3) and t["minutes_since_london_open"] <= 60,
        "rationale": "Mid-week early: avoids Mon/Fri + late entries",
    })
    filters.append({
        "name": "combo: risk_tight + first_45min",
        "fn": lambda t: t["risk_pips"] <= 8 and t["minutes_since_london_open"] <= 60,
        "rationale": "Good R:R + early momentum",
    })

    # 10. Cell-specific filters
    filters.append({
        "name": "cell: ambiguous only",
        "fn": lambda t: t["ownership_cell"] == "ambiguous/er_low/der_pos",
        "rationale": "Isolate the ambiguous cell (larger count)",
    })
    filters.append({
        "name": "cell: breakout only",
        "fn": lambda t: t["ownership_cell"] == "breakout/er_low/der_pos",
        "rationale": "Isolate the breakout cell",
    })

    return filters


def run_filter_analysis(
    trades_by_ds: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Apply each filter to both datasets and collect metrics."""
    filters = _define_filters()
    results = []

    # Baseline (unfiltered) for comparison
    for ds_key, trades in trades_by_ds.items():
        pnls = [t["pnl_pips"] for t in trades]
        m = _metrics(pnls)
        print(f"  Baseline {ds_key}: {m['count']} trades, avg {m['avg_pips']} pips, total {m['total_pips']} pips")

    for f in filters:
        row: dict[str, Any] = {"filter_name": f["name"], "rationale": f["rationale"]}
        stable = True
        for ds_key in ["500k", "1000k"]:
            trades = trades_by_ds.get(ds_key, [])
            passing = [t for t in trades if f["fn"](t)]
            pnls = [t["pnl_pips"] for t in passing]
            m = _metrics(pnls)
            row[f"{ds_key}_count"] = m["count"]
            row[f"{ds_key}_avg_pips"] = m["avg_pips"]
            row[f"{ds_key}_total_pips"] = m["total_pips"]
            row[f"{ds_key}_win_rate"] = m["win_rate"]
            row[f"{ds_key}_pf"] = m["pf"]
            row[f"{ds_key}_median_pips"] = m["median_pips"]
            # Check stability: both datasets positive avg?
            if m["count"] < 5 or m["avg_pips"] <= 0:
                stable = False
        row["stable_positive"] = stable
        # Quality score: geometric mean of avg_pips * sqrt(min_count) * stability
        avg_500 = row.get("500k_avg_pips", 0)
        avg_1000 = row.get("1000k_avg_pips", 0)
        min_count = min(row.get("500k_count", 0), row.get("1000k_count", 0))
        if stable and avg_500 > 0 and avg_1000 > 0 and min_count >= 5:
            row["quality_score"] = round((avg_500 + avg_1000) / 2.0 * (min_count ** 0.5), 2)
        else:
            row["quality_score"] = 0.0
        results.append(row)

    results.sort(key=lambda r: -r["quality_score"])
    return results


def _json_default(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return str(obj)


def build_memo(
    filter_results: list[dict[str, Any]],
    baseline_by_ds: dict[str, dict[str, Any]],
) -> str:
    """Build the markdown memo."""
    lines = [
        "# Setup D Top-Cell Subfilter Mining Results",
        "",
        "## Target Cells",
        "- `ambiguous/er_low/der_pos`",
        "- `breakout/er_low/der_pos`",
        "",
        "## Baseline (unfiltered, both cells combined, long only, native-allowed)",
        "",
        "| Dataset | Count | Avg Pips | Total Pips | Win Rate | PF |",
        "|---------|-------|----------|------------|----------|----|",
    ]
    for ds_key in ["500k", "1000k"]:
        b = baseline_by_ds[ds_key]
        lines.append(f"| {ds_key} | {b['count']} | {b['avg_pips']} | {b['total_pips']} | {b['win_rate']}% | {b['pf']} |")

    lines += [
        "",
        "## Ranked Filters (by quality score)",
        "",
        "Quality score = (avg of avg_pips across datasets) * sqrt(min_count). Zero if unstable or negative on either dataset.",
        "",
        "| Rank | Filter | Score | 500k Count | 500k Avg | 500k Total | 1000k Count | 1000k Avg | 1000k Total | Stable |",
        "|------|--------|-------|------------|----------|------------|-------------|-----------|-------------|--------|",
    ]

    for i, f in enumerate(filter_results[:35], 1):
        stable_str = "YES" if f["stable_positive"] else "no"
        lines.append(
            f"| {i} | {f['filter_name']} | {f['quality_score']} | "
            f"{f['500k_count']} | {f['500k_avg_pips']} | {f['500k_total_pips']} | "
            f"{f['1000k_count']} | {f['1000k_avg_pips']} | {f['1000k_total_pips']} | {stable_str} |"
        )

    # Recommendation
    stable_filters = [f for f in filter_results if f["stable_positive"] and f["quality_score"] > 0]
    top3 = stable_filters[:3] if len(stable_filters) >= 3 else stable_filters

    lines += [
        "",
        "## Recommendation",
        "",
    ]

    if top3:
        lines.append("### Top filters worth promoting to additive backtest:")
        lines.append("")
        for i, f in enumerate(top3, 1):
            lines.append(f"**{i}. {f['filter_name']}** (score: {f['quality_score']})")
            lines.append(f"   - 500k: {f['500k_count']} trades, {f['500k_avg_pips']} avg pips, {f['500k_total_pips']} total")
            lines.append(f"   - 1000k: {f['1000k_count']} trades, {f['1000k_avg_pips']} avg pips, {f['1000k_total_pips']} total")
            lines.append(f"   - Rationale: {f['rationale']}")
            lines.append("")
    else:
        lines.append("No filters achieved stable positive edge on both datasets with sufficient count.")
        lines.append("")

    # Overfit warnings
    overfit_candidates = [
        f for f in filter_results
        if f.get("500k_count", 0) < 8 or f.get("1000k_count", 0) < 8
        or (f.get("500k_avg_pips", 0) > 0) != (f.get("1000k_avg_pips", 0) > 0)
    ]
    if overfit_candidates:
        lines.append("### Filters that look like overfit noise (skip these):")
        lines.append("")
        for f in overfit_candidates[:10]:
            reason = []
            if f.get("500k_count", 0) < 8 or f.get("1000k_count", 0) < 8:
                reason.append("sparse")
            if (f.get("500k_avg_pips", 0) > 0) != (f.get("1000k_avg_pips", 0) > 0):
                reason.append("sign flip between datasets")
            lines.append(f"- **{f['filter_name']}**: {', '.join(reason)}")
        lines.append("")

    lines.append("## Verdict")
    lines.append("")
    if top3 and top3[0]["quality_score"] > 15:
        # Check improvement over baseline
        baseline_avg = (baseline_by_ds["500k"]["avg_pips"] + baseline_by_ds["1000k"]["avg_pips"]) / 2.0
        top_avg = (top3[0]["500k_avg_pips"] + top3[0]["1000k_avg_pips"]) / 2.0
        if top_avg > baseline_avg * 1.3:
            lines.append(
                f"YES -- the top filter ({top3[0]['filter_name']}) shows a meaningfully higher average "
                f"({top_avg:.2f} pips vs baseline {baseline_avg:.2f} pips) with sufficient count on both datasets. "
                f"Worth one more additive backtest using this subfilter as the gate."
            )
        else:
            lines.append(
                f"MARGINAL -- the top filter ({top3[0]['filter_name']}) improves average pips modestly "
                f"({top_avg:.2f} vs baseline {baseline_avg:.2f}) but may not be enough to overcome the "
                f"portfolio drag seen in the broad cell additive test. Test if easy, but manage expectations."
            )
    else:
        lines.append(
            "NO -- no simple subfilter found that is both stable and meaningfully better than "
            "the unfiltered cell surface. The edge in these cells is real but diffuse; "
            "tighter gating just sheds volume without concentrating quality."
        )

    return "\n".join(lines)


def main() -> int:
    trades_by_ds: dict[str, list[dict[str, Any]]] = {}
    baseline_by_ds: dict[str, dict[str, Any]] = {}

    for dataset in DATASETS:
        if not Path(dataset).exists():
            print(f"WARN: {dataset} not found, skipping")
            continue
        ds_key = _dataset_key(dataset)
        trades = collect_trades(dataset)
        # Only native-allowed long trades (Setup D native is long-only)
        native_long = [t for t in trades if t.get("native_allowed", False) and t["direction"] == "long"]
        trades_by_ds[ds_key] = native_long
        baseline_by_ds[ds_key] = _metrics([t["pnl_pips"] for t in native_long])

    if not trades_by_ds:
        print("ERROR: no datasets processed")
        return 1

    print("\n--- Running filter analysis ---")
    filter_results = run_filter_analysis(trades_by_ds)

    # Print top results
    print("\n--- Top 10 Filters by Quality Score ---")
    for i, f in enumerate(filter_results[:10], 1):
        print(f"  {i}. {f['filter_name']:40s} score={f['quality_score']:6.1f}  "
              f"500k: {f['500k_count']:3d} @ {f['500k_avg_pips']:+6.2f}  "
              f"1000k: {f['1000k_count']:3d} @ {f['1000k_avg_pips']:+6.2f}  "
              f"{'STABLE' if f['stable_positive'] else 'unstable'}")

    # Write outputs
    memo = build_memo(filter_results, baseline_by_ds)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text(memo, encoding="utf-8")
    print(f"\nWrote {OUTPUT_MD}")

    payload = {
        "target_cells": TARGET_CELLS,
        "baseline": baseline_by_ds,
        "filters": filter_results,
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    print(f"Wrote {OUTPUT_JSON}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
