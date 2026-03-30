#!/usr/bin/env python3
"""
London V2 day-of-week expansion test.

Runs the London V2 backtest engine three ways:
  1. Baseline (Tue/Wed only) — current production config
  2. Expansion days only (Mon/Thu/Fri) — isolated new-day signal quality
  3. All weekdays (Mon-Fri) — combined system if days are added

Same config, same setups, same risk — only active_days_utc changes.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import backtest_v2_multisetup_london as v2_engine

OUT_DIR = ROOT / "research_out"
BASELINE_CFG_PATH = OUT_DIR / "v2_exp4_winner_baseline_config.json"
DATASETS = {
    "500k": str(OUT_DIR / "USDJPY_M1_OANDA_500k.csv"),
    "1000k": str(OUT_DIR / "USDJPY_M1_OANDA_1000k.csv"),
}
OUTPUT_PATH = OUT_DIR / "backtest_london_v2_day_expansion.json"

VARIANTS = {
    "baseline_tue_wed": ["Tuesday", "Wednesday"],
    "expansion_mon_thu_fri": ["Monday", "Thursday", "Friday"],
    "all_weekdays": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
}


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


def _summarize_trades(trades_df: pd.DataFrame) -> dict[str, Any]:
    if trades_df.empty:
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate_pct": 0,
            "profit_factor": 0,
            "net_pips": 0,
            "net_usd": 0,
            "avg_pips": 0,
            "avg_win_pips": 0,
            "avg_loss_pips": 0,
            "max_win_pips": 0,
            "max_loss_pips": 0,
        }

    pips = trades_df["pnl_pips"].astype(float)
    usd = trades_df["pnl_usd"].astype(float) if "pnl_usd" in trades_df.columns else pips * 0
    wins = pips[pips > 0]
    losses = pips[pips <= 0]
    gross_win = float(wins.sum()) if len(wins) else 0
    gross_loss = float(abs(losses.sum())) if len(losses) else 0

    return {
        "total_trades": len(trades_df),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(100 * len(wins) / len(trades_df), 2) if len(trades_df) else 0,
        "profit_factor": round(gross_win / gross_loss, 4) if gross_loss > 0 else float("inf"),
        "net_pips": round(float(pips.sum()), 2),
        "net_usd": round(float(usd.sum()), 2),
        "avg_pips": round(float(pips.mean()), 2) if len(pips) else 0,
        "avg_win_pips": round(float(wins.mean()), 2) if len(wins) else 0,
        "avg_loss_pips": round(float(losses.mean()), 2) if len(losses) else 0,
        "max_win_pips": round(float(pips.max()), 2) if len(pips) else 0,
        "max_loss_pips": round(float(pips.min()), 2) if len(pips) else 0,
    }


def _by_setup(trades_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if trades_df.empty:
        return {}
    out = {}
    setup_col = "setup_type" if "setup_type" in trades_df.columns else None
    if setup_col is None:
        return {}
    for setup, group in trades_df.groupby(setup_col):
        out[str(setup)] = _summarize_trades(group)
    return out


def _by_direction(trades_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if trades_df.empty:
        return {}
    out = {}
    dir_col = "direction" if "direction" in trades_df.columns else None
    if dir_col is None:
        return {}
    for direction, group in trades_df.groupby(dir_col):
        out[str(direction)] = _summarize_trades(group)
    return out


def _by_day_of_week(trades_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if trades_df.empty:
        return {}
    out = {}
    if "entry_time" not in trades_df.columns:
        return {}
    df = trades_df.copy()
    df["_dow"] = pd.to_datetime(df["entry_time"]).dt.day_name()
    for day, group in df.groupby("_dow"):
        out[str(day)] = _summarize_trades(group)
    return out


def _run_variant(dataset: str, base_cfg: dict[str, Any], days: list[str]) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg["session"]["active_days_utc"] = days
    merged_cfg = v2_engine.merge_config(cfg)
    df = pd.read_csv(dataset)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    trades_df, equity_df, meta = v2_engine.run_backtest(df, merged_cfg)
    summary = _summarize_trades(trades_df)
    summary["by_setup"] = _by_setup(trades_df)
    summary["by_direction"] = _by_direction(trades_df)
    summary["by_day_of_week"] = _by_day_of_week(trades_df)

    # Equity curve stats
    if not equity_df.empty and "equity" in equity_df.columns:
        eq = equity_df["equity"].astype(float)
        peak = eq.cummax()
        dd = eq - peak
        summary["max_drawdown_usd"] = round(float(dd.min()), 2)
        summary["final_equity"] = round(float(eq.iloc[-1]), 2)
    else:
        summary["max_drawdown_usd"] = 0
        summary["final_equity"] = 100000.0

    return summary


def main() -> int:
    base_cfg = json.loads(BASELINE_CFG_PATH.read_text(encoding="utf-8"))

    results: dict[str, Any] = {}
    for ds_name, ds_path in DATASETS.items():
        print(f"=== {ds_name} ===")
        ds_results: dict[str, Any] = {}

        for variant_name, days in VARIANTS.items():
            print(f"  Running {variant_name} ({', '.join(days)}) ...")
            summary = _run_variant(ds_path, base_cfg, days)
            ds_results[variant_name] = {
                "active_days": days,
                **summary,
            }
            print(f"    trades={summary['total_trades']}, net_pips={summary['net_pips']}, "
                  f"PF={summary['profit_factor']}, WR={summary['win_rate_pct']}%, "
                  f"DD={summary['max_drawdown_usd']}")

        # Compute deltas
        baseline = ds_results["baseline_tue_wed"]
        for variant_name in ["expansion_mon_thu_fri", "all_weekdays"]:
            v = ds_results[variant_name]
            ds_results[f"delta_{variant_name}_vs_baseline"] = {
                "trades": v["total_trades"] - baseline["total_trades"],
                "net_pips": round(v["net_pips"] - baseline["net_pips"], 2),
                "net_usd": round(v["net_usd"] - baseline["net_usd"], 2),
                "profit_factor": round(v["profit_factor"] - baseline["profit_factor"], 4),
                "win_rate_pct": round(v["win_rate_pct"] - baseline["win_rate_pct"], 2),
                "max_drawdown_usd": round(v["max_drawdown_usd"] - baseline["max_drawdown_usd"], 2),
            }

        results[ds_name] = ds_results
        print()

    payload = {
        "title": "London V2 day-of-week expansion test",
        "question": "Does London V2 have edge on Monday/Thursday/Friday?",
        "baseline_config": str(BASELINE_CFG_PATH),
        "results": results,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
