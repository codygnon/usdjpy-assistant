#!/usr/bin/env python3
"""
Variant K: Block London V2 in a small stable cluster of ΔER-negative non-owned cells.

Builds on Variant I (= Variant F V44 filter + London V2 hold-through + global
stand-down in post_breakout_trend with negative ΔER).

Adds one small London V2-specific authorization cluster:
  - breakout / er_low / der_neg
  - momentum / er_mid / der_neg
  - ambiguous / er_high / der_neg

These were the stable cross-dataset London V2 non-owned cells with negative
average pips on both datasets and meaningful saveable loss.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path("/Users/codygnon/Documents/usdjpy_assistant")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import backtest_merged_integrated_tokyo_london_v2_ny as merged_engine
from scripts import backtest_v44_conservative_router as v44_router
from scripts import backtest_variant_i_pbt_standdown as variant_i

OUT_DIR = ROOT / "research_out"
STARTING_EQUITY = 100000.0

LONDON_BLOCK_CLUSTER = {
    ("breakout", "er_low", "der_neg"),
    ("momentum", "er_mid", "der_neg"),
    ("ambiguous", "er_high", "der_neg"),
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


def _er_bucket(er: float) -> str:
    if er < 0.35:
        return "er_low"
    if er < 0.55:
        return "er_mid"
    return "er_high"


def _der_bucket(delta_er: float) -> str:
    return "der_neg" if delta_er < 0 else "der_pos"


def build_variant_k_pre_coupling_kept(
    dataset: str,
) -> tuple[
    list[merged_engine.TradeRow],
    dict[str, Any],
    pd.DataFrame,
    pd.DatetimeIndex,
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """F + London cluster block + G hold-through + Variant I → pre-coupling trade list."""
    baseline = v44_router._build_baseline(
        dataset,
        Path(OUT_DIR / "tokyo_optimized_v14_config.json"),
        Path(OUT_DIR / "v2_exp4_winner_baseline_config.json"),
        Path(OUT_DIR / "session_momentum_v44_base_config.json"),
        STARTING_EQUITY,
    )

    classified_basic = v44_router._load_classified_bars(dataset)
    m5_basic = v44_router._build_m5(dataset)
    f_config = v44_router.VariantConfig(
        "F_block_plus_ambiguous_non_momentum",
        block_breakout=True,
        block_post_breakout=True,
        exhaustion_gate=False,
        block_ambiguous_non_momentum=True,
    )
    _, f_pre_coupling_trades = v44_router._run_variant(
        baseline["trades"], classified_basic, m5_basic, f_config,
        STARTING_EQUITY, baseline["v14_max_units"],
    )

    classified_dynamic = variant_i._load_classified_with_dynamic(dataset)
    dyn_time_idx = pd.DatetimeIndex(classified_dynamic["time"])

    kept_after_cluster: list[merged_engine.TradeRow] = []
    blocked_cluster: list[dict[str, Any]] = []
    for trade in f_pre_coupling_trades:
        if trade.strategy != "london_v2":
            kept_after_cluster.append(trade)
            continue

        row = variant_i._lookup_regime_with_dynamic(classified_dynamic, dyn_time_idx, trade.entry_time)
        full_row = classified_dynamic.iloc[dyn_time_idx.get_indexer([pd.Timestamp(trade.entry_time)], method="ffill")[0]]
        er = float(full_row.get("sf_er", 0.5))
        if np.isnan(er):
            er = 0.5
        cell = (row["regime_label"], _er_bucket(er), _der_bucket(row["delta_er"]))
        if cell in LONDON_BLOCK_CLUSTER:
            blocked_cluster.append(
                {
                    "strategy": trade.strategy,
                    "entry_time": trade.entry_time.isoformat(),
                    "side": trade.side,
                    "pips": float(trade.pips),
                    "usd": float(trade.usd),
                    "exit_reason": trade.exit_reason,
                    "regime_label": row["regime_label"],
                    "er_bucket": _er_bucket(er),
                    "der_bucket": _der_bucket(row["delta_er"]),
                    "delta_er": round(float(row["delta_er"]), 4),
                    "reason": f"blocked_london_cluster_{row['regime_label']}_{_er_bucket(er)}_{_der_bucket(row['delta_er'])}",
                }
            )
        else:
            kept_after_cluster.append(trade)

    m1 = merged_engine._load_m1(dataset)
    m1_idx = pd.DatetimeIndex(m1["time"])
    g_pre_coupling: list[merged_engine.TradeRow] = []
    for trade in kept_after_cluster:
        if trade.strategy != "london_v2":
            g_pre_coupling.append(trade)
            continue
        sim = v44_router._simulate_london_holdthrough(trade, m1, m1_idx)
        if sim is None:
            g_pre_coupling.append(trade)
            continue
        new_raw = dict(trade.raw or {})
        new_raw["exit_time_utc"] = sim["new_exit_time"]
        new_raw["exit_reason"] = sim["new_exit_reason"]
        new_raw["pnl_pips"] = sim["new_pips"]
        new_raw["pnl_usd"] = sim["new_usd"]
        g_pre_coupling.append(
            merged_engine.TradeRow(
                strategy=trade.strategy,
                entry_time=trade.entry_time,
                exit_time=pd.Timestamp(sim["new_exit_time"]).tz_convert("UTC"),
                entry_session=trade.entry_session,
                side=trade.side,
                pips=sim["new_pips"],
                usd=sim["new_usd"],
                exit_reason=sim["new_exit_reason"],
                standalone_entry_equity=trade.standalone_entry_equity,
                raw=new_raw,
                size_scale=trade.size_scale,
            )
        )

    kept: list[merged_engine.TradeRow] = []
    blocked_global: list[dict[str, Any]] = []
    for trade in g_pre_coupling:
        regime_info = variant_i._lookup_regime_with_dynamic(classified_dynamic, dyn_time_idx, trade.entry_time)
        if variant_i._is_variant_i_blocked(regime_info["regime_label"], regime_info["delta_er"]):
            blocked_global.append(
                {
                    "strategy": trade.strategy,
                    "entry_time": trade.entry_time.isoformat(),
                    "side": trade.side,
                    "pips": float(trade.pips),
                    "usd": float(trade.usd),
                    "exit_reason": trade.exit_reason,
                    "regime_label": regime_info["regime_label"],
                    "delta_er": round(regime_info["delta_er"], 4),
                    "reason": "blocked_global_pbt_der_neg",
                }
            )
        else:
            kept.append(trade)

    return kept, baseline, classified_dynamic, dyn_time_idx, blocked_cluster, blocked_global


def run_one(dataset: str) -> dict[str, Any]:
    dk = _dataset_key(dataset)
    variant_i_result = json.loads((OUT_DIR / "variant_i_pbt_der_standdown.json").read_text(encoding="utf-8"))[dk]
    variant_i_summary = variant_i_result["variant_i_summary"]

    kept, baseline, _, _, blocked_cluster, blocked_global = build_variant_k_pre_coupling_kept(dataset)

    coupled = merged_engine._apply_shared_equity_coupling(
        sorted(kept, key=lambda t: (t.exit_time, t.entry_time)),
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
            "win_rate_pct": round(100.0 * sum(1 for t in strat_trades if t.usd > 0) / max(1, len(strat_trades)), 1),
        }

    blocked_all = blocked_cluster + blocked_global
    delta_vs_i = {
        "net_usd": round(summary["net_usd"] - variant_i_summary["net_usd"], 2),
        "profit_factor": round(summary["profit_factor"] - variant_i_summary["profit_factor"], 4),
        "max_drawdown_usd": round(summary["max_drawdown_usd"] - variant_i_summary["max_drawdown_usd"], 2),
        "total_trades": summary["total_trades"] - variant_i_summary["total_trades"],
    }

    cluster_w = sum(1 for t in blocked_cluster if t["pips"] > 0)
    cluster_l = sum(1 for t in blocked_cluster if t["pips"] <= 0)

    return {
        "dataset": dk,
        "variant": "K_london_derneg_nonowned_cluster",
        "rule": "Block London V2 in the stable non-owned ΔER-negative cluster: breakout/er_low/der_neg, momentum/er_mid/der_neg, ambiguous/er_high/der_neg. Variant I still applies.",
        "variant_i_summary": variant_i_summary,
        "variant_k_summary": summary,
        "delta_vs_variant_i": delta_vs_i,
        "by_strategy_detail": by_strategy_detail,
        "blocked": {
            "total": len(blocked_all),
            "cluster_total": len(blocked_cluster),
            "cluster_winners_blocked": cluster_w,
            "cluster_losers_blocked": cluster_l,
            "cluster_net_pips_blocked": round(sum(t["pips"] for t in blocked_cluster), 1),
            "cluster_net_usd_blocked": round(sum(t["usd"] for t in blocked_cluster), 2),
            "by_reason": dict(Counter(t["reason"] for t in blocked_all)),
            "sample": blocked_all[:20],
        },
    }


def main() -> int:
    results = {}
    for dataset in [
        str(OUT_DIR / "USDJPY_M1_OANDA_500k.csv"),
        str(OUT_DIR / "USDJPY_M1_OANDA_1000k.csv"),
    ]:
        if not Path(dataset).exists():
            continue
        dk = _dataset_key(dataset)
        print(f"Running Variant K on {dk} ...")
        results[dk] = run_one(dataset)

    out_path = OUT_DIR / "variant_k_london_derneg_cluster.json"
    out_path.write_text(json.dumps(results, indent=2, default=_json_default), encoding="utf-8")
    print(f"\nWrote results to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
