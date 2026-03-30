#!/usr/bin/env python3
"""
Defensive Ownership v1.5 (Agent 1): single-pocket promotion test.

Pocket #1:
  Block v44_ny in cell ambiguous/er_low/der_neg
  (selected from defensive_ownership_v15_candidates + stability constraints).

Rules for this script:
  - One pocket only (narrow, testable)
  - Compare directly vs current promoted stack (Variant K)
  - No live code changes
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.ownership_table import cell_key
from scripts import backtest_merged_integrated_tokyo_london_v2_ny as merged_engine
from scripts import backtest_variant_k_london_cluster as variant_k
from scripts import backtest_variant_i_pbt_standdown as variant_i

OUT_DIR = ROOT / "research_out"
STARTING_EQUITY = 100_000.0
CANDIDATES_PATH = OUT_DIR / "defensive_ownership_v15_candidates.json"
STABILITY_PATH = OUT_DIR / "diagnostic_ownership_stability.json"
TARGET_CELL = "ambiguous/er_low/der_neg"
TARGET_STRATEGY = "v44_ny"


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


def _trade_cell(
    trade: merged_engine.TradeRow,
    classified_dynamic: pd.DataFrame,
    dyn_time_idx: pd.DatetimeIndex,
) -> str:
    regime_info = variant_i._lookup_regime_with_dynamic(classified_dynamic, dyn_time_idx, trade.entry_time)
    idx = dyn_time_idx.get_indexer([pd.Timestamp(trade.entry_time)], method="ffill")[0]
    full_row = classified_dynamic.iloc[idx]
    er = float(full_row.get("sf_er", 0.5))
    if np.isnan(er):
        er = 0.5
    return cell_key(
        regime_info["regime_label"],
        variant_k._er_bucket(er),
        variant_k._der_bucket(regime_info["delta_er"]),
    )


def _load_selection_reason() -> dict[str, Any]:
    info: dict[str, Any] = {
        "target_cell": TARGET_CELL,
        "target_strategy": TARGET_STRATEGY,
        "selection_constraints": [
            "negative avg pips on BOTH datasets",
            "stable cell across datasets",
            "avoid one-off (n=1) unless live pain; here require n>=5 on each dataset",
            "single-pocket test only",
        ],
    }
    if CANDIDATES_PATH.exists():
        cand = json.loads(CANDIDATES_PATH.read_text(encoding="utf-8"))
        rows = cand.get("v44_ny_negative_both_datasets", []) or []
        for row in rows:
            if row.get("cell") == TARGET_CELL:
                info["candidate_row"] = row
                break
    if STABILITY_PATH.exists():
        st = json.loads(STABILITY_PATH.read_text(encoding="utf-8"))
        info["stable_same_owner"] = TARGET_CELL in set(st.get("stable_same_owner", []) or [])
        info["stable_no_trade"] = TARGET_CELL in set(st.get("stable_no_trade", []) or [])
        info["unstable"] = TARGET_CELL in set(st.get("unstable", []) or [])
    return info


def run_one(dataset: str) -> dict[str, Any]:
    dk = _dataset_key(dataset)
    kept_k, baseline, classified_dynamic, dyn_time_idx, _, _ = variant_k.build_variant_k_pre_coupling_kept(dataset)

    # Baseline = promoted stack (Variant K)
    coupled_k = merged_engine._apply_shared_equity_coupling(
        sorted(kept_k, key=lambda t: (t.exit_time, t.entry_time)),
        STARTING_EQUITY,
        v14_max_units=baseline["v14_max_units"],
    )
    eq_k = merged_engine._build_equity_curve(coupled_k, STARTING_EQUITY)
    summary_k = merged_engine._stats(coupled_k, STARTING_EQUITY, eq_k)

    # Single-pocket extra veto
    kept_v15: list[merged_engine.TradeRow] = []
    blocked: list[dict[str, Any]] = []
    for t in kept_k:
        if t.strategy != TARGET_STRATEGY:
            kept_v15.append(t)
            continue
        c = _trade_cell(t, classified_dynamic, dyn_time_idx)
        if c == TARGET_CELL:
            blocked.append(
                {
                    "strategy": t.strategy,
                    "entry_time": t.entry_time.isoformat(),
                    "entry_session": str(t.entry_session),
                    "cell": c,
                    "pips": float(t.pips),
                    "usd": float(t.usd),
                    "exit_reason": str(t.exit_reason),
                    "reason": f"defensive_v15_block_{TARGET_STRATEGY}_{TARGET_CELL.replace('/', '_')}",
                }
            )
        else:
            kept_v15.append(t)

    coupled_v15 = merged_engine._apply_shared_equity_coupling(
        sorted(kept_v15, key=lambda t: (t.exit_time, t.entry_time)),
        STARTING_EQUITY,
        v14_max_units=baseline["v14_max_units"],
    )
    eq_v15 = merged_engine._build_equity_curve(coupled_v15, STARTING_EQUITY)
    summary_v15 = merged_engine._stats(coupled_v15, STARTING_EQUITY, eq_v15)

    delta_vs_k = {
        "net_usd": round(summary_v15["net_usd"] - summary_k["net_usd"], 2),
        "profit_factor": round(summary_v15["profit_factor"] - summary_k["profit_factor"], 4),
        "max_drawdown_usd": round(summary_v15["max_drawdown_usd"] - summary_k["max_drawdown_usd"], 2),
        "total_trades": int(summary_v15["total_trades"] - summary_k["total_trades"]),
    }
    blocked_w = sum(1 for b in blocked if b["pips"] > 0)
    blocked_l = sum(1 for b in blocked if b["pips"] <= 0)

    return {
        "dataset": dk,
        "baseline_variant_k_summary": summary_k,
        "defensive_v15_summary": summary_v15,
        "delta_vs_variant_k": delta_vs_k,
        "single_pocket_block": {
            "cell": TARGET_CELL,
            "strategy": TARGET_STRATEGY,
            "blocked_count": len(blocked),
            "blocked_winners": blocked_w,
            "blocked_losers": blocked_l,
            "blocked_net_pips": round(sum(b["pips"] for b in blocked), 2),
            "blocked_net_usd": round(sum(b["usd"] for b in blocked), 2),
            "sample": blocked[:30],
            "by_session": dict(Counter(b["entry_session"] for b in blocked)),
        },
    }


def main() -> int:
    selection = _load_selection_reason()
    results: dict[str, Any] = {
        "variant": "defensive_v15_single_pocket",
        "rule": f"On top of Variant K, block {TARGET_STRATEGY} when cell == {TARGET_CELL}",
        "selection": selection,
        "datasets": {},
    }

    for dataset in [
        str(OUT_DIR / "USDJPY_M1_OANDA_500k.csv"),
        str(OUT_DIR / "USDJPY_M1_OANDA_1000k.csv"),
    ]:
        if not Path(dataset).exists():
            print(f"SKIP missing {dataset}", file=sys.stderr)
            continue
        dk = _dataset_key(dataset)
        print(f"Running defensive v1.5 pocket #1 on {dk} ...")
        results["datasets"][dk] = run_one(dataset)
        d = results["datasets"][dk]["delta_vs_variant_k"]
        b = results["datasets"][dk]["single_pocket_block"]
        print(
            f"  blocked={b['blocked_count']} ({b['blocked_winners']}W/{b['blocked_losers']}L) "
            f"Δnet_usd={d['net_usd']} ΔPF={d['profit_factor']} ΔDD={d['max_drawdown_usd']} Δtrades={d['total_trades']}"
        )

    if not results["datasets"]:
        print("No datasets processed.", file=sys.stderr)
        return 1

    out_path = OUT_DIR / "defensive_v15_pocket1_v44_ambiguous_low_derneg.json"
    out_path.write_text(json.dumps(results, indent=2, default=_json_default), encoding="utf-8")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
