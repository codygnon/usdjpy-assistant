#!/usr/bin/env python3
"""
Variant L′ (L-prime): M5-bar arbitration on top of Variant K.

Same ownership cells as Variant L, but group entries by ``entry_time.floor("5min")``
(UTC) instead of same minute.

Research / backtest only. Does not touch live code.

``collision_census`` includes cross-strategy pair counts only when a bucket has
2+ distinct strategies; same-strategy multi-fill buckets are counted in
``intra_strategy_multi_buckets`` and ``m5_buckets_same_strategy_only``.
Session-pair counts still apply when entry_session differs within a bucket.
``multi_trade_m5_bucket_log`` carries ``distinct_strategies`` and
``cross_strategy_bucket`` per bucket.

Usage:
  python scripts/backtest_variant_lprime_m5_handoff.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
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
STARTING_EQUITY = 100000.0

V44_WINS_OVER_LONDON: set[tuple[str, str, str]] = {
    ("momentum", "er_mid", "der_neg"),
    ("momentum", "er_low", "der_pos"),
}

V14_WINS_OVER_V44: set[tuple[str, str, str]] = {
    ("ambiguous", "er_low", "der_neg"),
}


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


def _m5_bucket(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(ts).tz_convert("UTC").floor("5min")


def _cell_at(
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


def census_m5_collisions(
    trades: list[merged_engine.TradeRow],
    classified_dynamic: pd.DataFrame,
    dyn_time_idx: pd.DatetimeIndex,
) -> dict[str, Any]:
    by_m5: dict[pd.Timestamp, list[merged_engine.TradeRow]] = defaultdict(list)
    for t in trades:
        by_m5[_m5_bucket(t.entry_time)].append(t)

    lon_v44_any = 0
    lon_v44_in_v44_cells = 0
    v14_v44_any = 0
    v14_v44_in_v14_cell = 0
    buckets_2plus = 0

    pair_bucket_counts: Counter[str] = Counter()
    set_signature_counts: Counter[str] = Counter()
    session_pair_bucket_counts: Counter[str] = Counter()
    buckets_multi_session = 0
    multi_trade_bucket_log: list[dict[str, Any]] = []
    intra_strategy_multi_buckets: Counter[str] = Counter()
    cross_strategy_m5_buckets = 0

    for bucket, group in by_m5.items():
        if len(group) < 2:
            continue
        buckets_2plus += 1
        strats = {t.strategy for t in group}
        cell = _cell_at(classified_dynamic, dyn_time_idx, bucket)
        if "london_v2" in strats and "v44_ny" in strats:
            lon_v44_any += 1
            if cell in V44_WINS_OVER_LONDON:
                lon_v44_in_v44_cells += 1
        if "v14" in strats and "v44_ny" in strats:
            v14_v44_any += 1
            if cell in V14_WINS_OVER_V44:
                v14_v44_in_v14_cell += 1

        u_strats = sorted(strats)
        if len(u_strats) == 1:
            intra_strategy_multi_buckets[u_strats[0]] += 1
            sig = f"{u_strats[0]} x{len(group)}"
        else:
            cross_strategy_m5_buckets += 1
            sig = "+".join(u_strats)
            for a, b in combinations(u_strats, 2):
                pair_bucket_counts[f"{a}+{b}"] += 1
        set_signature_counts[sig] += 1

        sessions = [str(t.entry_session) for t in group]
        u_sess = sorted(set(sessions))
        if len(u_sess) >= 2:
            buckets_multi_session += 1
            for sa, sb in combinations(u_sess, 2):
                session_pair_bucket_counts[f"{sa}+{sb}"] += 1

        cell_list = [cell[0], cell[1], cell[2]]
        multi_trade_bucket_log.append(
            {
                "m5_bucket": bucket.isoformat(),
                "trade_count": len(group),
                "distinct_strategies": len(u_strats),
                "cross_strategy_bucket": len(u_strats) >= 2,
                "unique_strategies": u_strats,
                "strategy_pairs_in_bucket": [f"{a}+{b}" for a, b in combinations(u_strats, 2)],
                "unique_sessions": u_sess,
                "session_pairs_in_bucket": (
                    [f"{a}+{b}" for a, b in combinations(u_sess, 2)] if len(u_sess) >= 2 else []
                ),
                "chart_cell_at_bucket": cell_list,
                "trades": [
                    {
                        "strategy": t.strategy,
                        "entry_session": str(t.entry_session),
                        "entry_time": t.entry_time.isoformat(),
                        "pips": float(t.pips),
                        "usd": float(t.usd),
                    }
                    for t in sorted(group, key=lambda x: (x.entry_time, x.strategy))
                ],
            }
        )

    multi_trade_bucket_log.sort(key=lambda x: x["m5_bucket"])

    return {
        "m5_buckets_with_2plus_trades": buckets_2plus,
        "m5_buckets_cross_strategy": cross_strategy_m5_buckets,
        "m5_buckets_same_strategy_only": buckets_2plus - cross_strategy_m5_buckets,
        "intra_strategy_multi_buckets": dict(intra_strategy_multi_buckets.most_common()),
        "london_and_v44_same_m5_any_cell": lon_v44_any,
        "london_and_v44_same_m5_in_v44_win_cells": lon_v44_in_v44_cells,
        "v14_and_v44_same_m5_any_cell": v14_v44_any,
        "v14_and_v44_same_m5_in_v14_win_cell": v14_v44_in_v14_cell,
        "strategy_pair_bucket_counts": dict(pair_bucket_counts.most_common()),
        "strategy_set_signature_counts": dict(set_signature_counts.most_common()),
        "m5_buckets_with_2plus_distinct_sessions": buckets_multi_session,
        "session_pair_bucket_counts": dict(session_pair_bucket_counts.most_common()),
        "multi_trade_m5_bucket_log": multi_trade_bucket_log,
    }


def apply_m5_handoff(
    trades: list[merged_engine.TradeRow],
    classified_dynamic: pd.DataFrame,
    dyn_time_idx: pd.DatetimeIndex,
) -> tuple[list[merged_engine.TradeRow], list[dict[str, Any]]]:
    by_m5: dict[pd.Timestamp, list[tuple[int, merged_engine.TradeRow]]] = defaultdict(list)
    for i, t in enumerate(trades):
        by_m5[_m5_bucket(t.entry_time)].append((i, t))

    drop_idx: set[int] = set()
    log: list[dict[str, Any]] = []

    for m5_bucket, items in by_m5.items():
        if len(items) < 2:
            continue
        strats = {t.strategy for _, t in items}
        cell = _cell_at(classified_dynamic, dyn_time_idx, m5_bucket)
        cell_list = [cell[0], cell[1], cell[2]]

        if "london_v2" in strats and "v44_ny" in strats and cell in V44_WINS_OVER_LONDON:
            for idx, t in items:
                if t.strategy == "london_v2":
                    drop_idx.add(idx)
                    log.append(
                        {
                            "handoff": "v44_wins_over_london",
                            "handoff_reason": "v44_wins_over_london",
                            "strategy": t.strategy,
                            "dropped_strategy": t.strategy,
                            "entry_time": t.entry_time.isoformat(),
                            "m5_bucket": m5_bucket.isoformat(),
                            "cell": cell_list,
                            "pips": float(t.pips),
                            "usd": float(t.usd),
                        }
                    )

        if "v14" in strats and "v44_ny" in strats and cell in V14_WINS_OVER_V44:
            for idx, t in items:
                if t.strategy == "v44_ny":
                    drop_idx.add(idx)
                    log.append(
                        {
                            "handoff": "v14_wins_over_v44",
                            "handoff_reason": "v14_wins_over_v44",
                            "strategy": t.strategy,
                            "dropped_strategy": t.strategy,
                            "entry_time": t.entry_time.isoformat(),
                            "m5_bucket": m5_bucket.isoformat(),
                            "cell": cell_list,
                            "pips": float(t.pips),
                            "usd": float(t.usd),
                        }
                    )

    kept = [t for i, t in enumerate(trades) if i not in drop_idx]
    return kept, log


def run_one(dataset: str) -> dict[str, Any]:
    dk = _dataset_key(dataset)
    variant_i_result = json.loads((OUT_DIR / "variant_i_pbt_der_standdown.json").read_text(encoding="utf-8"))[dk]
    variant_i_summary = variant_i_result["variant_i_summary"]

    k_path = OUT_DIR / "variant_k_london_derneg_cluster.json"
    if k_path.exists():
        variant_k_summary = json.loads(k_path.read_text(encoding="utf-8"))[dk]["variant_k_summary"]
    else:
        variant_k_summary = None

    kept, baseline, classified_dynamic, dyn_time_idx, _, _ = (
        variant_k.build_variant_k_pre_coupling_kept(dataset)
    )

    collision_census = census_m5_collisions(kept, classified_dynamic, dyn_time_idx)
    kept_lp, handoff_log = apply_m5_handoff(kept, classified_dynamic, dyn_time_idx)

    coupled = merged_engine._apply_shared_equity_coupling(
        sorted(kept_lp, key=lambda t: (t.exit_time, t.entry_time)),
        STARTING_EQUITY,
        v14_max_units=baseline["v14_max_units"],
    )
    eq_curve = merged_engine._build_equity_curve(coupled, STARTING_EQUITY)
    summary = merged_engine._stats(coupled, STARTING_EQUITY, eq_curve)

    by_strategy_detail = {}
    for strat_key in ["v14", "london_v2", "v44_ny"]:
        strat_trades = [t for t in coupled if t.strategy == strat_key]
        by_strategy_detail[strat_key] = {
            "trades": len(strat_trades),
            "net_usd": round(sum(t.usd for t in strat_trades), 2),
            "winners": sum(1 for t in strat_trades if t.usd > 0),
            "losers": sum(1 for t in strat_trades if t.usd <= 0),
            "win_rate_pct": round(
                100.0 * sum(1 for t in strat_trades if t.usd > 0) / max(1, len(strat_trades)),
                1,
            ),
        }

    dropped_pips = sum(h["pips"] for h in handoff_log)
    dropped_usd_standalone = sum(h["usd"] for h in handoff_log)

    out: dict[str, Any] = {
        "dataset": dk,
        "variant": "Lprime_m5_bar_chart_handoff",
        "rule": (
            "On top of Variant K: same M5 bucket (entry_time.floor('5min') UTC), "
            "drop London if V44 also fires and cell in V44_WINS_OVER_LONDON; "
            "drop V44 if V14 also fires and cell in V14_WINS_OVER_V44."
        ),
        "variant_i_summary": variant_i_summary,
        "variant_lprime_summary": summary,
        "delta_vs_variant_i": {
            "net_usd": round(summary["net_usd"] - variant_i_summary["net_usd"], 2),
            "profit_factor": round(summary["profit_factor"] - variant_i_summary["profit_factor"], 4),
            "max_drawdown_usd": round(summary["max_drawdown_usd"] - variant_i_summary["max_drawdown_usd"], 2),
            "total_trades": summary["total_trades"] - variant_i_summary["total_trades"],
        },
        "by_strategy_detail": by_strategy_detail,
        "handoff": {
            "events": len(handoff_log),
            "by_type": dict(Counter(h["handoff"] for h in handoff_log)),
            "dropped_standalone_pips": round(dropped_pips, 2),
            "dropped_standalone_usd_sum": round(dropped_usd_standalone, 2),
            "log": handoff_log,
        },
        "collision_census": collision_census,
    }
    if variant_k_summary is not None:
        out["variant_k_summary"] = variant_k_summary
        out["delta_vs_variant_k"] = {
            "net_usd": round(summary["net_usd"] - variant_k_summary["net_usd"], 2),
            "profit_factor": round(summary["profit_factor"] - variant_k_summary["profit_factor"], 4),
            "max_drawdown_usd": round(summary["max_drawdown_usd"] - variant_k_summary["max_drawdown_usd"], 2),
            "total_trades": summary["total_trades"] - variant_k_summary["total_trades"],
        }
    return out


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
        print(f"Running Variant L′ (M5) on {dk} ...")
        results[dk] = run_one(dataset)
        h = results[dk]["handoff"]
        c = results[dk]["collision_census"]
        s = results[dk]["variant_lprime_summary"]
        print(f"  handoff_events={h['events']}  trades={s['total_trades']}")
        print(
            f"  census: m5_buckets_2+={c['m5_buckets_with_2plus_trades']}  "
            f"Lon+V44 any={c['london_and_v44_same_m5_any_cell']}  "
            f"in_V44_cells={c['london_and_v44_same_m5_in_v44_win_cells']}  "
            f"V14+V44 any={c['v14_and_v44_same_m5_any_cell']}  "
            f"in_V14_cell={c['v14_and_v44_same_m5_in_v14_win_cell']}"
        )
        print(
            f"  cross_strategy_m5_buckets={c.get('m5_buckets_cross_strategy', 0)}  "
            f"same_strategy_only={c.get('m5_buckets_same_strategy_only', 0)}  "
            f"intra_multi={c.get('intra_strategy_multi_buckets', {})}"
        )
        pp = c.get("strategy_pair_bucket_counts") or {}
        if pp:
            top = ", ".join(f"{k}={v}" for k, v in list(pp.items())[:6])
            print(f"  strategy_pair_bucket_counts (top): {top}")
        sp = c.get("session_pair_bucket_counts") or {}
        if sp:
            st = ", ".join(f"{k}={v}" for k, v in list(sp.items())[:6])
            print(f"  session_pair_bucket_counts: {st}")
        if "delta_vs_variant_k" in results[dk]:
            d = results[dk]["delta_vs_variant_k"]
            print(
                f"  vs K: Δnet_usd={d['net_usd']}  ΔPF={d['profit_factor']}  "
                f"ΔDD={d['max_drawdown_usd']}  Δtrades={d['total_trades']}"
            )

    if not results:
        print("No datasets processed.", file=sys.stderr)
        return 1

    out_path = OUT_DIR / "variant_lprime_m5_handoff.json"
    out_path.write_text(json.dumps(results, indent=2, default=_json_default), encoding="utf-8")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
