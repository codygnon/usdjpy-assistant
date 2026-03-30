#!/usr/bin/env python3
"""
Variant L: Same-minute chart handoff on top of Variant K.

After the London V2 ownership-cluster gate (K) + F + G + I, if multiple strategies
share the same entry minute, drop the non-owner per stable cross-dataset cells:

  - V44 wins over London: (momentum, er_mid, der_neg), (momentum, er_low, der_pos)
  - V14 wins over V44: (ambiguous, er_low, der_neg)

Then shared-equity coupling as usual. Research / backtest only.

Note: On 500k/1000k merged baselines, entry minutes are unique per trade (no two
strategies share the same floored entry minute), so handoff events may be zero.
Use ``collision_census`` in the output to confirm. If so, arbitration needs a
wider window (e.g. same M5 bar) or live-style bar batching.

Usage:
  python scripts/backtest_variant_l_same_minute_handoff.py
"""
from __future__ import annotations

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
STARTING_EQUITY = 100000.0

# Stable V44-owned cells where London V2 bleeds (cross-dataset ownership map).
V44_WINS_OVER_LONDON: set[tuple[str, str, str]] = {
    ("momentum", "er_mid", "der_neg"),
    ("momentum", "er_low", "der_pos"),
}

# Stable V14-owned cell where V44 bleeds.
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


def _entry_minute(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(ts).tz_convert("UTC").floor("min")


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


def census_same_minute_collisions(
    trades: list[merged_engine.TradeRow],
    classified_dynamic: pd.DataFrame,
    dyn_time_idx: pd.DatetimeIndex,
) -> dict[str, Any]:
    """How often do strategies share an entry minute (any cell)?"""
    by_minute: dict[pd.Timestamp, list[merged_engine.TradeRow]] = defaultdict(list)
    for t in trades:
        by_minute[_entry_minute(t.entry_time)].append(t)

    lon_v44_any = 0
    lon_v44_in_v44_cells = 0
    v14_v44_any = 0
    v14_v44_in_v14_cell = 0
    multi_2plus = 0

    for minute, group in by_minute.items():
        if len(group) < 2:
            continue
        multi_2plus += 1
        strats = {t.strategy for t in group}
        cell = _cell_at(classified_dynamic, dyn_time_idx, minute)
        if "london_v2" in strats and "v44_ny" in strats:
            lon_v44_any += 1
            if cell in V44_WINS_OVER_LONDON:
                lon_v44_in_v44_cells += 1
        if "v14" in strats and "v44_ny" in strats:
            v14_v44_any += 1
            if cell in V14_WINS_OVER_V44:
                v14_v44_in_v14_cell += 1

    return {
        "minutes_with_2plus_trades": multi_2plus,
        "london_and_v44_same_minute_any_cell": lon_v44_any,
        "london_and_v44_same_minute_in_v44_win_cells": lon_v44_in_v44_cells,
        "v14_and_v44_same_minute_any_cell": v14_v44_any,
        "v14_and_v44_same_minute_in_v14_win_cell": v14_v44_in_v14_cell,
    }


def apply_same_minute_handoff(
    trades: list[merged_engine.TradeRow],
    classified_dynamic: pd.DataFrame,
    dyn_time_idx: pd.DatetimeIndex,
) -> tuple[list[merged_engine.TradeRow], list[dict[str, Any]]]:
    by_minute: dict[pd.Timestamp, list[tuple[int, merged_engine.TradeRow]]] = defaultdict(list)
    for i, t in enumerate(trades):
        by_minute[_entry_minute(t.entry_time)].append((i, t))

    drop_idx: set[int] = set()
    log: list[dict[str, Any]] = []

    for minute, items in by_minute.items():
        if len(items) < 2:
            continue
        strats = {t.strategy for _, t in items}
        cell = _cell_at(classified_dynamic, dyn_time_idx, minute)

        if "london_v2" in strats and "v44_ny" in strats and cell in V44_WINS_OVER_LONDON:
            for idx, t in items:
                if t.strategy == "london_v2":
                    drop_idx.add(idx)
                    log.append(
                        {
                            "handoff": "v44_wins_over_london",
                            "entry_minute": minute.isoformat(),
                            "cell": list(cell),
                            "dropped_strategy": t.strategy,
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
                            "entry_minute": minute.isoformat(),
                            "cell": list(cell),
                            "dropped_strategy": t.strategy,
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

    collision_census = census_same_minute_collisions(kept, classified_dynamic, dyn_time_idx)
    kept_l, handoff_log = apply_same_minute_handoff(kept, classified_dynamic, dyn_time_idx)

    coupled = merged_engine._apply_shared_equity_coupling(
        sorted(kept_l, key=lambda t: (t.exit_time, t.entry_time)),
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
        "variant": "L_same_minute_chart_handoff",
        "rule": (
            "On top of Variant K: same entry minute, drop London if V44 also fires and cell "
            "in V44_WINS_OVER_LONDON; drop V44 if V14 also fires and cell in V14_WINS_OVER_V44."
        ),
        "variant_i_summary": variant_i_summary,
        "variant_l_summary": summary,
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
            "sample": handoff_log[:40],
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
        print(f"Running Variant L on {dk} ...")
        results[dk] = run_one(dataset)
        h = results[dk]["handoff"]
        c = results[dk]["collision_census"]
        print(f"  handoff events={h['events']}  trades={results[dk]['variant_l_summary']['total_trades']}")
        print(
            f"  census: 2+trades/min={c['minutes_with_2plus_trades']}  "
            f"Lon+V44 any={c['london_and_v44_same_minute_any_cell']}  "
            f"in V44-win cells={c['london_and_v44_same_minute_in_v44_win_cells']}  "
            f"V14+V44 any={c['v14_and_v44_same_minute_any_cell']}  "
            f"in V14-win cell={c['v14_and_v44_same_minute_in_v14_win_cell']}"
        )
        if "delta_vs_variant_k" in results[dk]:
            print(f"  delta vs K net_usd: {results[dk]['delta_vs_variant_k']['net_usd']}")

    if not results:
        print("No datasets processed.", file=sys.stderr)
        return 1

    out_path = OUT_DIR / "variant_l_same_minute_handoff.json"
    out_path.write_text(json.dumps(results, indent=2, default=_json_default), encoding="utf-8")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
