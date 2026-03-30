#!/usr/bin/env python3
"""
Offensive narrow-slice mining factory — all three archetypes.

Three-phase pipeline:
  Phase 1 — Harvest: run all engines, tag every trade with ownership cell
  Phase 2 — Discover: find whitespace cells with cross-dataset edge
  Phase 3 — Test: additive merge top candidates into Variant K baseline

Output: research_out/narrow_slice_mining_board.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import backtest_merged_integrated_tokyo_london_v2_ny as merged_engine
from scripts import backtest_variant_k_london_cluster as variant_k
from scripts import run_offensive_slice_discovery as discovery

OUT_DIR = ROOT / "research_out"
STARTING_EQUITY = 100000.0

DATASETS = {
    "500k": str(OUT_DIR / "USDJPY_M1_OANDA_500k.csv"),
    "1000k": str(OUT_DIR / "USDJPY_M1_OANDA_1000k.csv"),
}
DEFAULT_OUTPUT = OUT_DIR / "narrow_slice_mining_board.json"

# Per-archetype discovery thresholds
ARCHETYPE_THRESHOLDS = {
    "v14": {"min_avg_pips": 1.5, "min_count": 3, "daily_cap": 1},
    "london_v2": {"min_avg_pips": 2.0, "min_count": 5, "daily_cap": 1},
    "v44_ny": {"min_avg_pips": 1.0, "min_count": 5, "daily_cap": 2},
}
MAX_BASELINE_PRESENCE = 5  # whitespace = baseline has < this many trades in cell
SIZE_SCALES = [0.5, 0.75, 1.0]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return str(obj)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Narrow-slice mining factory for all archetypes")
    p.add_argument("--top-n", type=int, default=10, help="How many whitespace candidates to run additive tests for")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════
# Phase 1: Harvest — get all trades, tag with ownership cells
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TaggedTrade:
    trade: merged_engine.TradeRow
    cell: str
    regime_label: str
    er: float
    delta_er: float
    direction: str  # "buy" or "sell"
    in_defensive_baseline: bool


def _native_trade_to_tagged(
    t: dict[str, Any],
    defensive_keys: set[tuple[str, str, str]],
) -> TaggedTrade:
    """Convert a normalized native trade dict into a TaggedTrade."""
    cell = str(t["ownership_cell"])
    parts = cell.split("/")
    regime_label = parts[0] if parts else "ambiguous"
    er = float(t.get("sf_er", 0.5))
    delta_er = float(t.get("delta_er", 0.0))
    entry_ts = pd.Timestamp(t["entry_time"])
    exit_ts = pd.Timestamp(t["exit_time"])
    trade_row = merged_engine.TradeRow(
        strategy=t["strategy"],
        entry_time=entry_ts,
        exit_time=exit_ts,
        entry_session=t.get("entry_session", ""),
        side=t["side"],
        pips=float(t["pips"]),
        usd=float(t["usd"]),
        exit_reason=t.get("exit_reason", "unknown"),
        standalone_entry_equity=float(t.get("standalone_entry_equity", STARTING_EQUITY)),
        raw=t.get("raw", {}),
        size_scale=float(t.get("size_scale", 1.0)),
    )
    key = (entry_ts.isoformat(), t["strategy"], t["side"])
    return TaggedTrade(
        trade=trade_row,
        cell=cell,
        regime_label=regime_label,
        er=er,
        delta_er=delta_er,
        direction=t["side"],
        in_defensive_baseline=key in defensive_keys,
    )


def harvest_dataset(
    dataset: str,
    all_native_trades: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, Any]:
    """Tag native trades with ownership cells, split defensive/offensive."""
    ds_key = "500k" if "500k" in Path(dataset).name else "1000k"
    ds_trades = all_native_trades[ds_key]

    print(f"    Building Variant K defensive baseline...")
    kept, k_meta, classified_dynamic, dyn_time_idx, _blocked_cluster, _blocked_global = (
        variant_k.build_variant_k_pre_coupling_kept(dataset)
    )

    # Build defensive key set
    defensive_keys: set[tuple[str, str, str]] = set()
    for t in kept:
        defensive_keys.add((pd.Timestamp(t.entry_time).isoformat(), t.strategy, t.side))

    # Convert all native trades to TaggedTrade
    all_native: list[dict[str, Any]] = []
    for strategy_trades in ds_trades.values():
        all_native.extend(strategy_trades)
    print(f"    Tagging {len(all_native)} native trades with ownership cells...")

    tagged: list[TaggedTrade] = []
    for t in all_native:
        tagged.append(_native_trade_to_tagged(t, defensive_keys))

    # Split by strategy
    by_strategy: dict[str, list[TaggedTrade]] = defaultdict(list)
    for t in tagged:
        by_strategy[t.trade.strategy].append(t)

    return {
        "all_tagged": tagged,
        "by_strategy": dict(by_strategy),
        "defensive_kept": kept,
        "defensive_meta": k_meta,
        "classified_dynamic": classified_dynamic,
        "dyn_time_idx": dyn_time_idx,
        "v14_max_units": k_meta["v14_max_units"],
    }


# ═══════════════════════════════════════════════════════════════════════
# Phase 2: Discover — find whitespace cells with cross-dataset edge
# ═══════════════════════════════════════════════════════════════════════

def _cell_metrics(trades: list[TaggedTrade]) -> dict[str, Any]:
    if not trades:
        return {"count": 0}
    pips = [t.trade.pips for t in trades]
    wins = [p for p in pips if p > 0]
    losses = [p for p in pips if p <= 0]
    gross_win = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    return {
        "count": len(pips),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(100 * len(wins) / len(pips), 1),
        "avg_pips": round(sum(pips) / len(pips), 2),
        "net_pips": round(sum(pips), 2),
        "pf": round(gross_win / gross_loss, 4) if gross_loss > 0 else float("inf"),
    }


def compute_cell_grids(
    harvest_by_ds: dict[str, dict[str, Any]],
) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    """Per archetype, per dataset, per (cell|direction) → metrics."""
    grids: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for ds_key, harvest in harvest_by_ds.items():
        for strategy, trades in harvest["by_strategy"].items():
            # Group by (cell, direction)
            groups: dict[str, list[TaggedTrade]] = defaultdict(list)
            for t in trades:
                key = f"{t.cell}|{t.direction}"
                groups[key].append(t)
            for group_key, group_trades in groups.items():
                grids[strategy][ds_key][group_key] = _cell_metrics(group_trades)
    return dict(grids)


def compute_defensive_cell_counts(
    harvest_by_ds: dict[str, dict[str, Any]],
) -> dict[str, dict[str, dict[str, int]]]:
    """Per dataset, per cell → {strategy: count} for defensive-only trades."""
    out: dict[str, dict[str, dict[str, int]]] = {}
    for ds_key, harvest in harvest_by_ds.items():
        cell_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for t in harvest["all_tagged"]:
            if t.in_defensive_baseline:
                cell_counts[t.cell][t.trade.strategy] += 1
        out[ds_key] = {c: dict(s) for c, s in cell_counts.items()}
    return out


def discover_whitespace(
    cell_grids: dict[str, dict[str, dict[str, dict[str, Any]]]],
    defensive_counts: dict[str, dict[str, dict[str, int]]],
) -> list[dict[str, Any]]:
    """Find whitespace candidates that pass all gates on both datasets."""
    ds_keys = sorted(defensive_counts.keys())
    if len(ds_keys) < 2:
        return []

    candidates = []
    for strategy, ds_grids in cell_grids.items():
        thresholds = ARCHETYPE_THRESHOLDS.get(strategy)
        if not thresholds:
            continue

        # Collect all (cell|direction) keys across datasets
        all_keys = set()
        for ds_key in ds_keys:
            all_keys.update(ds_grids.get(ds_key, {}).keys())

        for group_key in sorted(all_keys):
            parts = group_key.rsplit("|", 1)
            if len(parts) != 2:
                continue
            cell, direction = parts

            # Must exist on both datasets
            metrics = {}
            for ds_key in ds_keys:
                m = ds_grids.get(ds_key, {}).get(group_key)
                if m is None or m["count"] == 0:
                    break
                metrics[ds_key] = m
            else:
                pass  # all datasets have data
            if len(metrics) != len(ds_keys):
                continue

            # Gate: minimum count on both datasets
            if any(m["count"] < thresholds["min_count"] for m in metrics.values()):
                continue

            # Gate: positive avg_pips above threshold on both datasets
            if any(m["avg_pips"] < thresholds["min_avg_pips"] for m in metrics.values()):
                continue

            # Gate: positive net_pips on both datasets
            if any(m["net_pips"] <= 0 for m in metrics.values()):
                continue

            # Gate: whitespace — defensive baseline has few trades in this cell
            is_whitespace = True
            for ds_key in ds_keys:
                baseline_count = sum(defensive_counts.get(ds_key, {}).get(cell, {}).values())
                if baseline_count >= MAX_BASELINE_PRESENCE:
                    is_whitespace = False
                    break

            # Even if not pure whitespace, still include as "contested" if edge is strong
            rank_score = sum(m["net_pips"] for m in metrics.values())

            candidates.append({
                "archetype": strategy,
                "cell": cell,
                "direction": direction,
                "whitespace": is_whitespace,
                "rank_score": round(rank_score, 2),
                **{ds_key: metrics[ds_key] for ds_key in ds_keys},
                "defensive_presence": {
                    ds_key: sum(defensive_counts.get(ds_key, {}).get(cell, {}).values())
                    for ds_key in ds_keys
                },
            })

    # Sort: whitespace first, then by rank_score descending
    candidates.sort(key=lambda c: (not c["whitespace"], -c["rank_score"]))
    return candidates


# ═══════════════════════════════════════════════════════════════════════
# Phase 3: Test — additive merge into Variant K baseline
# ═══════════════════════════════════════════════════════════════════════

def _daily_cap(trades: list[TaggedTrade], cap: int) -> list[TaggedTrade]:
    """Keep first `cap` trades per day (by entry_time date)."""
    by_day: dict[str, list[TaggedTrade]] = defaultdict(list)
    for t in trades:
        day = pd.Timestamp(t.trade.entry_time).date().isoformat()
        by_day[day].append(t)
    kept = []
    for day in sorted(by_day.keys()):
        day_trades = sorted(by_day[day], key=lambda t: t.trade.entry_time)
        kept.extend(day_trades[:cap])
    return kept


def run_additive_test(
    *,
    candidate: dict[str, Any],
    harvest_by_ds: dict[str, dict[str, Any]],
    size_scales: list[float],
) -> dict[str, dict[str, Any]]:
    """Run additive test for one whitespace candidate across all datasets and size scales."""
    archetype = candidate["archetype"]
    cell = candidate["cell"]
    direction = candidate["direction"]
    thresholds = ARCHETYPE_THRESHOLDS[archetype]
    results: dict[str, dict[str, Any]] = {}

    for ds_key, harvest in harvest_by_ds.items():
        # Get offensive trades: in the right cell+direction, NOT in defensive baseline
        offensive = [
            t for t in harvest["by_strategy"].get(archetype, [])
            if t.cell == cell
            and t.direction == direction
            and not t.in_defensive_baseline
        ]

        # Daily cap
        offensive = _daily_cap(offensive, thresholds["daily_cap"])

        if not offensive:
            results[ds_key] = {"skipped": True, "reason": "no_offensive_trades_after_filtering"}
            continue

        # Build defensive baseline stats
        defensive_kept = harvest["defensive_kept"]
        v14_max_units = harvest["v14_max_units"]
        baseline_coupled = merged_engine._apply_shared_equity_coupling(
            sorted(defensive_kept, key=lambda t: (t.exit_time, t.entry_time)),
            STARTING_EQUITY,
            v14_max_units=v14_max_units,
        )
        baseline_eq = merged_engine._build_equity_curve(baseline_coupled, STARTING_EQUITY)
        baseline_stats = merged_engine._stats(baseline_coupled, STARTING_EQUITY, baseline_eq)

        ds_results: dict[str, Any] = {
            "baseline_trades": baseline_stats["total_trades"],
            "baseline_net_usd": round(baseline_stats["net_usd"], 2),
            "baseline_pf": round(baseline_stats["profit_factor"], 4),
            "baseline_max_dd": round(baseline_stats["max_drawdown_usd"], 2),
            "offensive_candidate_count": len(offensive),
            "size_variants": {},
        }

        for scale in size_scales:
            scale_label = f"{int(scale * 100)}pct"

            # Build offensive TradeRows with scaled USD
            offensive_rows = []
            for t in offensive:
                scaled_usd = t.trade.usd * (scale / t.trade.size_scale) if t.trade.size_scale else t.trade.usd * scale
                offensive_rows.append(merged_engine.TradeRow(
                    strategy=t.trade.strategy,
                    entry_time=t.trade.entry_time,
                    exit_time=t.trade.exit_time,
                    entry_session=t.trade.entry_session,
                    side=t.trade.side,
                    pips=t.trade.pips,
                    usd=scaled_usd,
                    exit_reason=t.trade.exit_reason,
                    standalone_entry_equity=t.trade.standalone_entry_equity,
                    raw=t.trade.raw,
                    size_scale=scale,
                ))

            # Merge and recouple
            merged = sorted(
                list(defensive_kept) + offensive_rows,
                key=lambda t: (t.exit_time, t.entry_time),
            )
            coupled = merged_engine._apply_shared_equity_coupling(
                merged, STARTING_EQUITY, v14_max_units=v14_max_units,
            )
            eq = merged_engine._build_equity_curve(coupled, STARTING_EQUITY)
            stats = merged_engine._stats(coupled, STARTING_EQUITY, eq)

            ds_results["size_variants"][scale_label] = {
                "total_trades": stats["total_trades"],
                "net_usd": round(stats["net_usd"], 2),
                "pf": round(stats["profit_factor"], 4),
                "max_dd": round(stats["max_drawdown_usd"], 2),
                "delta_trades": stats["total_trades"] - baseline_stats["total_trades"],
                "delta_net_usd": round(stats["net_usd"] - baseline_stats["net_usd"], 2),
                "delta_pf": round(stats["profit_factor"] - baseline_stats["profit_factor"], 4),
                "delta_max_dd": round(stats["max_drawdown_usd"] - baseline_stats["max_drawdown_usd"], 2),
            }

        results[ds_key] = ds_results

    return results


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> int:
    args = parse_args()
    top_n = args.top_n

    # Phase 1: Harvest
    print("Phase 1: Harvest — loading native trades and tagging")
    print("  Loading native trade reports (all archetypes, both datasets)...")
    all_native_trades = discovery._load_all_normalized_trades({"v14", "london_v2", "v44_ny"})
    harvest_by_ds: dict[str, dict[str, Any]] = {}
    for ds_key, ds_path in DATASETS.items():
        print(f"  [{ds_key}]")
        harvest_by_ds[ds_key] = harvest_dataset(ds_path, all_native_trades)
        for strategy, trades in harvest_by_ds[ds_key]["by_strategy"].items():
            n_def = sum(1 for t in trades if t.in_defensive_baseline)
            n_off = sum(1 for t in trades if not t.in_defensive_baseline)
            print(f"    {strategy}: {len(trades)} total ({n_def} defensive, {n_off} offensive)")
    print()

    # Phase 2: Discover
    print("Phase 2: Discover — finding whitespace cells")
    cell_grids = compute_cell_grids(harvest_by_ds)
    defensive_counts = compute_defensive_cell_counts(harvest_by_ds)
    candidates = discover_whitespace(cell_grids, defensive_counts)
    n_whitespace = sum(1 for c in candidates if c["whitespace"])
    n_contested = sum(1 for c in candidates if not c["whitespace"])
    print(f"  Found {n_whitespace} whitespace + {n_contested} contested candidates")
    for i, c in enumerate(candidates[:top_n]):
        tag = "WS" if c["whitespace"] else "CT"
        print(f"  [{tag}] #{i+1}: {c['archetype']} | {c['cell']} | {c['direction']} | rank={c['rank_score']}")
    print()

    # Phase 3: Test
    print(f"Phase 3: Test — additive merge for top {top_n} candidates")
    additive_results: dict[str, dict[str, dict[str, Any]]] = {}
    for i, candidate in enumerate(candidates[:top_n]):
        key = f"{candidate['archetype']}|{candidate['cell']}|{candidate['direction']}"
        print(f"  [{i+1}/{min(top_n, len(candidates))}] {key}")
        additive_results[key] = run_additive_test(
            candidate=candidate,
            harvest_by_ds=harvest_by_ds,
            size_scales=SIZE_SCALES,
        )
        # Print quick summary
        for ds_key, ds_result in additive_results[key].items():
            if ds_result.get("skipped"):
                print(f"    {ds_key}: skipped ({ds_result['reason']})")
                continue
            full = ds_result["size_variants"].get("100pct", {})
            print(f"    {ds_key}: +{full.get('delta_trades', 0)} trades, "
                  f"Δ${full.get('delta_net_usd', 0):.0f}, "
                  f"ΔPF={full.get('delta_pf', 0):+.4f}, "
                  f"ΔDD={full.get('delta_max_dd', 0):+.0f}")
    print()

    # Build output
    # Serialize cell grids (convert defaultdicts)
    serializable_grids: dict[str, dict[str, dict[str, Any]]] = {}
    for strategy, ds_grids in cell_grids.items():
        serializable_grids[strategy] = {}
        for ds_key, grid in ds_grids.items():
            serializable_grids[strategy][ds_key] = dict(grid)

    payload = {
        "meta": {
            "title": "Narrow-slice mining board — all archetypes",
            "datasets": list(DATASETS.keys()),
            "run_timestamp": datetime.now(timezone.utc).isoformat(),
            "top_n": top_n,
            "size_scales": SIZE_SCALES,
            "archetype_thresholds": ARCHETYPE_THRESHOLDS,
            "max_baseline_presence": MAX_BASELINE_PRESENCE,
        },
        "per_archetype_cell_grids": serializable_grids,
        "defensive_baseline_cell_counts": defensive_counts,
        "whitespace_candidates": candidates,
        "additive_results": additive_results,
    }

    output = Path(args.output)
    output.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
