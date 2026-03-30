#!/usr/bin/env python3
"""
Defensive analog to offensive down-weight: keep matching trades but scale pips/usd.

On Variant K pre-coupling kept list, for trades where (strategy, ownership_cell)
matches, replace TradeRow with dataclasses.replace(..., pips=pips*scale, usd=usd*scale),
then re-couple and compare to baseline.

Usage:
  python3 scripts/backtest_defensive_v15_pocket_downweight.py \\
    --strategy v44_ny --cell ambiguous/er_low/der_neg --scale 0.5 \\
    --output research_out/defensive_v15_downweight_v44_ambig_0p5.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
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


def run_one_downweight(
    dataset: str,
    target_strategy: str,
    target_cell: str,
    scale: float,
) -> dict[str, Any]:
    dk = _dataset_key(dataset)
    kept_k, baseline, classified_dynamic, dyn_time_idx, _, _ = variant_k.build_variant_k_pre_coupling_kept(dataset)

    coupled_k = merged_engine._apply_shared_equity_coupling(
        sorted(kept_k, key=lambda t: (t.exit_time, t.entry_time)),
        STARTING_EQUITY,
        v14_max_units=baseline["v14_max_units"],
    )
    eq_k = merged_engine._build_equity_curve(coupled_k, STARTING_EQUITY)
    summary_k = merged_engine._stats(coupled_k, STARTING_EQUITY, eq_k)

    adjusted: list[merged_engine.TradeRow] = []
    touched: list[dict[str, Any]] = []
    for t in kept_k:
        c = _trade_cell(t, classified_dynamic, dyn_time_idx)
        if t.strategy == target_strategy and c == target_cell:
            nt = replace(t, pips=float(t.pips) * scale, usd=float(t.usd) * scale)
            adjusted.append(nt)
            touched.append(
                {
                    "entry_time": t.entry_time.isoformat(),
                    "pips_before": float(t.pips),
                    "pips_after": float(nt.pips),
                    "usd_before": float(t.usd),
                    "usd_after": float(nt.usd),
                }
            )
        else:
            adjusted.append(t)

    coupled_dw = merged_engine._apply_shared_equity_coupling(
        sorted(adjusted, key=lambda t: (t.exit_time, t.entry_time)),
        STARTING_EQUITY,
        v14_max_units=baseline["v14_max_units"],
    )
    eq_dw = merged_engine._build_equity_curve(coupled_dw, STARTING_EQUITY)
    summary_dw = merged_engine._stats(coupled_dw, STARTING_EQUITY, eq_dw)

    delta_vs_k = {
        "net_usd": round(summary_dw["net_usd"] - summary_k["net_usd"], 2),
        "profit_factor": round(summary_dw["profit_factor"] - summary_k["profit_factor"], 4),
        "max_drawdown_usd": round(summary_dw["max_drawdown_usd"] - summary_k["max_drawdown_usd"], 2),
        "total_trades": int(summary_dw["total_trades"] - summary_k["total_trades"]),
    }
    return {
        "dataset": dk,
        "scale": scale,
        "baseline_variant_k_summary": summary_k,
        "defensive_downweight_summary": summary_dw,
        "delta_vs_variant_k": delta_vs_k,
        "downweight_slice": {
            "strategy": target_strategy,
            "cell": target_cell,
            "adjusted_trade_count": len(touched),
            "sample": touched[:25],
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Down-weight defensive slice vs Variant K.")
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--scale", type=float, required=True, help="Multiply pips and usd for matching trades (e.g. 0.5)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--dataset", action="append", dest="datasets")
    args = ap.parse_args()
    scale = float(args.scale)
    if not (0 < scale <= 1.0):
        print("--scale must be in (0, 1]", file=sys.stderr)
        return 1
    paths = args.datasets if args.datasets else DEFAULT_DATASETS
    results: dict[str, Any] = {
        "variant": "defensive_v15_downweight",
        "rule": f"Scale {args.strategy} in cell {args.cell} by {scale} (pips+usd)",
        "datasets": {},
    }
    for p in paths:
        if not Path(p).exists():
            continue
        dk = _dataset_key(p)
        results["datasets"][dk] = run_one_downweight(p, args.strategy, args.cell, scale)
    if not results["datasets"]:
        print("No datasets processed.", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, default=_json_default), encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
