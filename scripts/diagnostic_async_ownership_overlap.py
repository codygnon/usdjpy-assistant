#!/usr/bin/env python3
"""
Asynchronous ownership overlap diagnostic (research only).

Uses Variant K pre-coupling kept trades (F+G+I+K). For trades that enter in
known V44- or V14-owned chart cells, scans forward 5/10/15/30 minutes for
entries from a *different* strategy. Does not modify entries, live code, or
promote a variant.

London's K cluster is reported only as context (blocked counts); London is not
used as an async owner here.
"""
from __future__ import annotations

import bisect
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

from scripts import backtest_merged_integrated_tokyo_london_v2_ny as merged_engine
from scripts import backtest_variant_i_pbt_standdown as variant_i
from scripts import backtest_variant_k_london_cluster as variant_k

OUT_DIR = ROOT / "research_out"
WINDOWS_MIN = (5, 10, 15, 30)

V44_OWNED_CELLS: frozenset[tuple[str, str, str]] = frozenset(
    {
        ("momentum", "er_mid", "der_neg"),
        ("momentum", "er_low", "der_pos"),
    }
)
V14_OWNED_CELLS: frozenset[tuple[str, str, str]] = frozenset(
    {("ambiguous", "er_low", "der_neg")}
)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)


def _dataset_key(dataset_path: str) -> str:
    name = Path(dataset_path).name
    if "500k" in name:
        return "500k"
    if "1000k" in name:
        return "1000k"
    raise ValueError(f"Unknown dataset: {name}")


def _cell_str(cell: tuple[str, str, str]) -> str:
    return f"{cell[0]}|{cell[1]}|{cell[2]}"


def cell_at_entry(
    classified_dynamic: pd.DataFrame,
    dyn_time_idx: pd.DatetimeIndex,
    ts: pd.Timestamp,
) -> tuple[str, str, str]:
    row = variant_i._lookup_regime_with_dynamic(classified_dynamic, dyn_time_idx, ts)
    idx = dyn_time_idx.get_indexer([pd.Timestamp(ts)], method="ffill")[0]
    if idx < 0:
        return ("ambiguous", "er_mid", "der_pos")
    full_row = classified_dynamic.iloc[idx]
    er = float(full_row.get("sf_er", 0.5))
    if np.isnan(er):
        er = 0.5
    return (
        row["regime_label"],
        variant_k._er_bucket(er),
        variant_k._der_bucket(row["delta_er"]),
    )


def _owner_strategy_for_cell(cell: tuple[str, str, str]) -> str | None:
    if cell in V44_OWNED_CELLS:
        return "v44_ny"
    if cell in V14_OWNED_CELLS:
        return "v14"
    return None


def _london_cluster_context(blocked_cluster: list[dict[str, Any]]) -> dict[str, Any]:
    by_cell: Counter[str] = Counter()
    for b in blocked_cluster:
        rg = b.get("regime_label", "")
        erb = b.get("er_bucket", "")
        der = b.get("der_bucket", "")
        by_cell[_cell_str((rg, erb, der))] += 1
    return {
        "description": (
            "London V2 trades blocked in Variant K non-owned ΔER-negative cluster; "
            "not used as async owner in this diagnostic."
        ),
        "cluster_cells": [list(c) for c in sorted(variant_k.LONDON_BLOCK_CLUSTER)],
        "blocked_trade_count": len(blocked_cluster),
        "blocked_by_cell": dict(by_cell.most_common()),
    }


def run_diagnostic(dataset: str) -> dict[str, Any]:
    kept, _baseline, classified_dynamic, dyn_time_idx, blocked_cluster, _blocked_global = (
        variant_k.build_variant_k_pre_coupling_kept(dataset)
    )

    trades_sorted = sorted(kept, key=lambda t: t.entry_time)

    enriched: list[dict[str, Any]] = []
    for t in trades_sorted:
        ts = pd.Timestamp(t.entry_time).tz_convert("UTC")
        cell = cell_at_entry(classified_dynamic, dyn_time_idx, ts)
        m5 = ts.floor("5min")
        enriched.append(
            {
                "trade": t,
                "entry_time": ts,
                "m5_floor": m5,
                "cell": cell,
                "cell_str": _cell_str(cell),
                "strategy": t.strategy,
                "pips": float(t.pips),
                "usd": float(t.usd),
            }
        )
    entry_times = [e["entry_time"] for e in enriched]

    test_cells = sorted(V44_OWNED_CELLS | V14_OWNED_CELLS, key=_cell_str)
    owner_trade_indices: dict[str, list[int]] = defaultdict(list)
    for i, e in enumerate(enriched):
        os_ = _owner_strategy_for_cell(e["cell"])
        if os_ is None:
            continue
        if e["strategy"] != os_:
            continue
        owner_trade_indices[_cell_str(e["cell"])].append(i)

    all_conflicts: list[dict[str, Any]] = []

    for cell in test_cells:
        cell_s = _cell_str(cell)
        owner_strat = _owner_strategy_for_cell(cell)
        assert owner_strat is not None
        for w in WINDOWS_MIN:
            delta = pd.Timedelta(minutes=w)
            for i in owner_trade_indices.get(cell_s, []):
                own = enriched[i]
                t0 = own["entry_time"]
                end = t0 + delta
                j = bisect.bisect_right(entry_times, end)
                for k in range(i + 1, j):
                    lat = enriched[k]
                    if lat["strategy"] == own["strategy"]:
                        continue
                    lag_m = (lat["entry_time"] - t0).total_seconds() / 60.0
                    all_conflicts.append(
                        {
                            "owner_strategy": own["strategy"],
                            "owner_cell": list(cell),
                            "owner_cell_str": cell_s,
                            "window_minutes": w,
                            "later_strategy": lat["strategy"],
                            "later_entry_time": lat["entry_time"],
                            "owner_entry_time": t0,
                            "lag_minutes": round(lag_m, 4),
                            "later_trade_pips": lat["pips"],
                            "later_trade_usd": lat["usd"],
                            "saveable_pips": max(0.0, -lat["pips"]),
                            "saveable_usd": max(0.0, -lat["usd"]),
                        }
                    )

    owner_to_later: list[dict[str, Any]] = []
    for cell in test_cells:
        cell_s = _cell_str(cell)
        owner_strat = _owner_strategy_for_cell(cell)
        assert owner_strat is not None
        n_owners = len(owner_trade_indices.get(cell_s, []))
        for w in WINDOWS_MIN:
            subset = [c for c in all_conflicts if c["owner_cell_str"] == cell_s and c["window_minutes"] == w]
            by_lat = Counter(c["later_strategy"] for c in subset)
            lags = [c["lag_minutes"] for c in subset]
            pips = [c["later_trade_pips"] for c in subset]
            usds = [c["later_trade_usd"] for c in subset]
            owner_to_later.append(
                {
                    "owner_strategy": owner_strat,
                    "owner_cell": list(cell),
                    "window_minutes": w,
                    "total_owner_trades": n_owners,
                    "later_conflicting_entries_count": len(subset),
                    "conflicting_strategy_breakdown": dict(by_lat.most_common()),
                    "avg_lag_minutes": round(sum(lags) / len(lags), 4) if lags else None,
                    "conflicting_trades_avg_pips": round(sum(pips) / len(pips), 4) if pips else None,
                    "conflicting_trades_total_pips": round(sum(pips), 2),
                    "conflicting_trades_total_usd": round(sum(usds), 2),
                    "conflicting_later_losers_count": sum(1 for u in usds if u < 0),
                    "saveable_pips_sum": round(sum(max(0.0, -p) for p in pips), 2),
                    "saveable_usd_sum": round(sum(max(0.0, -u) for u in usds), 2),
                }
            )

    group_key_counts: Counter[tuple[str, int, str]] = Counter()
    group_save_pips: defaultdict[tuple[str, int, str], float] = defaultdict(float)
    group_save_usd: defaultdict[tuple[str, int, str], float] = defaultdict(float)
    for c in all_conflicts:
        key = (c["owner_cell_str"], c["window_minutes"], c["later_strategy"])
        group_key_counts[key] += 1
        group_save_pips[key] += c["saveable_pips"]
        group_save_usd[key] += c["saveable_usd"]

    top_by_count = sorted(group_key_counts.items(), key=lambda x: -x[1])[:25]
    top_by_save_pips = sorted(group_save_pips.items(), key=lambda x: -x[1])[:25]
    top_by_save_usd = sorted(group_save_usd.items(), key=lambda x: -x[1])[:25]

    def _expand_top(ranked: list[tuple[tuple[str, int, str], float]]) -> list[dict[str, Any]]:
        out = []
        for key, metric in ranked:
            cs, w, ls = key
            out.append(
                {
                    "owner_cell_str": cs,
                    "window_minutes": w,
                    "later_strategy": ls,
                    "pair_count": group_key_counts[key],
                    "saveable_pips_sum": round(group_save_pips[key], 2),
                    "saveable_usd_sum": round(group_save_usd[key], 2),
                    "_sort_metric": round(metric, 4) if isinstance(metric, float) else metric,
                }
            )
        return out

    top_async_conflicts = {
        "by_conflict_count": _expand_top([(k, float(v)) for k, v in top_by_count]),
        "by_saveable_pips": _expand_top(top_by_save_pips),
        "by_saveable_usd": _expand_top(top_by_save_usd),
    }
    for bucket in top_async_conflicts.values():
        for row in bucket:
            row.pop("_sort_metric", None)

    sample_sorted = sorted(all_conflicts, key=lambda c: (-c["saveable_usd"], -c["saveable_pips"], c["owner_entry_time"]))
    sample_conflicts = [
        {
            "owner_strategy": c["owner_strategy"],
            "owner_entry_time": c["owner_entry_time"].isoformat(),
            "owner_cell": c["owner_cell"],
            "later_strategy": c["later_strategy"],
            "later_entry_time": c["later_entry_time"].isoformat(),
            "window_minutes": c["window_minutes"],
            "lag_minutes": c["lag_minutes"],
            "later_trade_pips": c["later_trade_pips"],
            "later_trade_usd": c["later_trade_usd"],
        }
        for c in sample_sorted[:30]
    ]

    no_overlap_findings: list[str] = []
    for row in owner_to_later:
        if row["total_owner_trades"] == 0:
            no_overlap_findings.append(
                f"No owner trades: cell {row['owner_cell']} ({row['owner_strategy']}) — "
                "no entries in this ownership cell for that strategy on this path."
            )
            continue
        if row["later_conflicting_entries_count"] == 0:
            no_overlap_findings.append(
                f"Zero later conflicting entries: {row['owner_strategy']} in cell "
                f"{row['owner_cell']} → {row['window_minutes']}m window "
                f"({row['total_owner_trades']} owner trades)."
            )

    total_conflicts = len(all_conflicts)
    max_cell_window_count = max(
        (r["later_conflicting_entries_count"] for r in owner_to_later),
        default=0,
    )
    best_row = max(owner_to_later, key=lambda r: r["later_conflicting_entries_count"], default=None)

    later_loss_rate = None
    if all_conflicts:
        later_loss_rate = sum(1 for c in all_conflicts if c["later_trade_usd"] < 0) / len(all_conflicts)

    worth = False
    rationale_parts: list[str] = []
    if total_conflicts == 0:
        rationale_parts.append(
            "No (owner cell, different strategy) pairs within any tested window; "
            "asynchronous overlap is absent on this stack."
        )
    else:
        rationale_parts.append(
            f"Total async conflict observations: {total_conflicts} "
            f"(same owner trade can contribute up to {len(WINDOWS_MIN)} windows if multiple horizons hit)."
        )
        if later_loss_rate is not None:
            rationale_parts.append(
                f"Share of later trades with negative USD: {round(100 * later_loss_rate, 1)}%."
            )
        if best_row and best_row["later_conflicting_entries_count"] > 0:
            rationale_parts.append(
                f"Densest bucket: {best_row['owner_cell']} @ {best_row['window_minutes']}m → "
                f"{best_row['later_conflicting_entries_count']} later entries "
                f"(owners={best_row['total_owner_trades']})."
            )
        strong_bucket = any(
            r["later_conflicting_entries_count"] >= 8
            and r.get("saveable_usd_sum", 0) > 500
            and r.get("conflicting_later_losers_count", 0) >= max(4, r["later_conflicting_entries_count"] // 2)
            for r in owner_to_later
        )
        if strong_bucket and (later_loss_rate or 0) >= 0.5:
            worth = True
            rationale_parts.append(
                "At least one (cell, window) bucket shows repeated later entries with meaningful "
                "saveable loss and a majority of later trades losing — a narrow backtested rule may be justified."
            )
        else:
            rationale_parts.append(
                "No bucket meets a conservative threshold for volume + saveable loss + loser share; "
                "async suppression is not an obvious next step from this diagnostic alone."
            )

    verdict = {
        "async_suppression_worth_pursuing": worth,
        "rationale": " ".join(rationale_parts),
        "total_conflict_observations": total_conflicts,
        "max_later_entries_single_cell_window": max_cell_window_count,
        "later_trade_usd_negative_rate": round(later_loss_rate, 4) if later_loss_rate is not None else None,
        "strongest_by_count": (
            {
                "owner_cell": best_row["owner_cell"],
                "window_minutes": best_row["window_minutes"],
                "later_conflicting_entries_count": best_row["later_conflicting_entries_count"],
                "conflicting_strategy_breakdown": best_row["conflicting_strategy_breakdown"],
            }
            if best_row and best_row["later_conflicting_entries_count"] > 0
            else None
        ),
    }

    return {
        "dataset": _dataset_key(dataset),
        "pre_coupling_trade_count": len(kept),
        "windows_minutes": list(WINDOWS_MIN),
        "v44_owned_cells": [list(c) for c in sorted(V44_OWNED_CELLS, key=_cell_str)],
        "v14_owned_cells": [list(c) for c in sorted(V14_OWNED_CELLS, key=_cell_str)],
        "london_cluster_context": _london_cluster_context(blocked_cluster),
        "owner_to_later_entries": owner_to_later,
        "top_async_conflicts": top_async_conflicts,
        "sample_conflicts": sample_conflicts,
        "no_overlap_findings": no_overlap_findings,
        "verdict": verdict,
    }


def main() -> int:
    results: dict[str, Any] = {}
    for dataset in [
        str(OUT_DIR / "USDJPY_M1_OANDA_500k.csv"),
        str(OUT_DIR / "USDJPY_M1_OANDA_1000k.csv"),
    ]:
        if not Path(dataset).exists():
            print(f"SKIP missing {dataset}", file=sys.stderr)
            continue
        dk = _dataset_key(dataset)
        print(f"Async ownership diagnostic: {dk} ...")
        results[dk] = run_diagnostic(dataset)
        v = results[dk]["verdict"]
        print(f"  pre_coupling_trades={results[dk]['pre_coupling_trade_count']}  "
              f"conflict_obs={v['total_conflict_observations']}  "
              f"worth_pursuing={v['async_suppression_worth_pursuing']}")
        if v["strongest_by_count"]:
            s = v["strongest_by_count"]
            print(
                f"  strongest: cell={s['owner_cell']} window={s['window_minutes']}m "
                f"later_n={s['later_conflicting_entries_count']} "
                f"breakdown={s['conflicting_strategy_breakdown']}"
            )
        print(f"  verdict: {v['rationale'][:220]}...")

    if not results:
        print("No datasets processed.", file=sys.stderr)
        return 1

    out_path = OUT_DIR / "diagnostic_async_ownership_overlap.json"
    out_path.write_text(json.dumps(results, indent=2, default=_json_default), encoding="utf-8")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
