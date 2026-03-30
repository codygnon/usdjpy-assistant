#!/usr/bin/env python3
"""
Parameterized defensive v1.5 single-pocket block vs Variant K (500k + 1000k).

On top of the promoted Variant K stack, drop trades where
(strategy == target_strategy and ownership_cell == target_cell), then
re-apply shared equity coupling and compare to baseline.

CLI:
  python3 scripts/backtest_defensive_v15_pocket_grid.py \\
    --strategy v44_ny --cell ambiguous/er_low/der_neg \\
    --output research_out/defensive_v15_pocket_v44_ny_ambiguous_er_low_der_neg.json
"""
from __future__ import annotations

import argparse
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
DEFAULT_DATASETS = [
    str(OUT_DIR / "USDJPY_M1_OANDA_500k.csv"),
    str(OUT_DIR / "USDJPY_M1_OANDA_1000k.csv"),
]

VALID_STRATEGIES = frozenset({"v44_ny", "london_v2", "v14"})


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


def cell_slug(cell: str) -> str:
    return cell.replace("/", "_")


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


def _should_block(
    trade: merged_engine.TradeRow,
    cell: str,
    pockets: list[tuple[str, str]],
) -> tuple[bool, str | None]:
    """Return (blocked, pocket_id) for first matching (strategy, cell) rule."""
    for target_strategy, target_cell in pockets:
        if trade.strategy != target_strategy:
            continue
        if cell == target_cell:
            pid = f"{target_strategy}_{target_cell.replace('/', '_')}"
            return True, pid
    return False, None


def run_one(
    dataset: str,
    target_strategy: str,
    target_cell: str,
) -> dict[str, Any]:
    return run_one_multi(dataset, [(target_strategy, target_cell)])


def run_one_multi(
    dataset: str,
    pockets: list[tuple[str, str]],
) -> dict[str, Any]:
    if not pockets:
        raise ValueError("pockets must be non-empty")
    dk = _dataset_key(dataset)
    kept_k, baseline, classified_dynamic, dyn_time_idx, _, _ = variant_k.build_variant_k_pre_coupling_kept(dataset)

    coupled_k = merged_engine._apply_shared_equity_coupling(
        sorted(kept_k, key=lambda t: (t.exit_time, t.entry_time)),
        STARTING_EQUITY,
        v14_max_units=baseline["v14_max_units"],
    )
    eq_k = merged_engine._build_equity_curve(coupled_k, STARTING_EQUITY)
    summary_k = merged_engine._stats(coupled_k, STARTING_EQUITY, eq_k)

    kept_v15: list[merged_engine.TradeRow] = []
    blocked: list[dict[str, Any]] = []
    for t in kept_k:
        c = _trade_cell(t, classified_dynamic, dyn_time_idx)
        hit, pocket_id = _should_block(t, c, pockets)
        if hit:
            blocked.append(
                {
                    "strategy": t.strategy,
                    "entry_time": t.entry_time.isoformat(),
                    "entry_session": str(t.entry_session),
                    "cell": c,
                    "pips": float(t.pips),
                    "usd": float(t.usd),
                    "exit_reason": str(t.exit_reason),
                    "pocket_id": pocket_id,
                    "reason": f"defensive_v15_block_{pocket_id}",
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

    pocket_meta: dict[str, Any]
    if len(pockets) == 1:
        ts, tc = pockets[0]
        pocket_meta = {"cell": tc, "strategy": ts}
    else:
        pocket_meta = {"pockets": [{"strategy": s, "cell": c} for s, c in pockets]}

    return {
        "dataset": dk,
        "baseline_variant_k_summary": summary_k,
        "defensive_v15_summary": summary_v15,
        "delta_vs_variant_k": delta_vs_k,
        "single_pocket_block": {
            **pocket_meta,
            "blocked_count": len(blocked),
            "blocked_winners": blocked_w,
            "blocked_losers": blocked_l,
            "blocked_net_pips": round(sum(b["pips"] for b in blocked), 2),
            "blocked_net_usd": round(sum(b["usd"] for b in blocked), 2),
            "sample": blocked[:30],
            "by_session": dict(Counter(b["entry_session"] for b in blocked)),
            "by_pocket_id": dict(Counter(b.get("pocket_id") or "" for b in blocked)),
        },
    }


def run_pocket_datasets(
    target_strategy: str,
    target_cell: str,
    dataset_paths: list[str] | None = None,
) -> dict[str, Any]:
    return run_multi_pocket_datasets([(target_strategy, target_cell)], dataset_paths)


def run_multi_pocket_datasets(
    pockets: list[tuple[str, str]],
    dataset_paths: list[str] | None = None,
) -> dict[str, Any]:
    paths = dataset_paths if dataset_paths is not None else DEFAULT_DATASETS
    if len(pockets) == 1:
        s, c = pockets[0]
        rule = f"On top of Variant K, block {s} when cell == {c}"
        variant = "defensive_v15_single_pocket_grid"
    else:
        parts = [f"{s}@{c}" for s, c in pockets]
        rule = "On top of Variant K, block trades matching any of: " + "; ".join(parts)
        variant = "defensive_v15_multi_pocket_grid"
    results: dict[str, Any] = {
        "variant": variant,
        "rule": rule,
        "pockets": [{"strategy": s, "cell": c} for s, c in pockets],
        "datasets": {},
    }
    for dataset in paths:
        if not Path(dataset).exists():
            continue
        dk = _dataset_key(dataset)
        results["datasets"][dk] = run_one_multi(dataset, pockets)
    return results


def strict_pass_both(datasets_block: dict[str, Any]) -> bool:
    d = datasets_block.get("datasets") or {}
    for key in ("500k", "1000k"):
        if key not in d:
            return False
        delta = d[key].get("delta_vs_variant_k") or {}
        if float(delta.get("net_usd", 0)) <= 0:
            return False
        if float(delta.get("profit_factor", 0)) <= 0:
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Single-pocket defensive block vs Variant K.")
    ap.add_argument("--strategy", required=True, choices=sorted(VALID_STRATEGIES))
    ap.add_argument("--cell", required=True, help="ownership cell, e.g. ambiguous/er_low/der_neg")
    ap.add_argument("--output", type=Path, required=True, help="JSON output path")
    ap.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Optional CSV path (repeatable); default 500k+1000k in research_out",
    )
    args = ap.parse_args()
    strategy = str(args.strategy)
    cell = str(args.cell)
    paths = args.datasets if args.datasets else None
    results = run_pocket_datasets(strategy, cell, paths)
    if not results["datasets"]:
        print("No datasets processed.", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, default=_json_default), encoding="utf-8")
    print(f"Wrote {args.output}")
    for dk, block in results["datasets"].items():
        d = block["delta_vs_variant_k"]
        b = block["single_pocket_block"]
        print(
            f"  {dk}: blocked={b['blocked_count']} "
            f"Δnet_usd={d['net_usd']} ΔPF={d['profit_factor']} ΔDD={d['max_drawdown_usd']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
