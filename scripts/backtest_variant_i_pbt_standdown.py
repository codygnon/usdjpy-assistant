#!/usr/bin/env python3
"""
Variant I: Global stand-down when regime = post_breakout_trend AND ΔER < 0.

Builds on Variant G (= Variant F V44 filter + London V2 hold-through trail 8/8).
Adds one additional filter: block ALL strategies (V14, London V2, V44) when the
M5 bar at entry time is classified as post_breakout_trend AND the 3-bar ΔER is negative.

This is the first cross-strategy regime gate — it doesn't just filter V44, it
stands down the entire engine in a specific chart condition.
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

from core.regime_features import compute_delta_efficiency_ratio, compute_efficiency_ratio
from core.regime_classifier import RegimeThresholds
from scripts import backtest_merged_integrated_tokyo_london_v2_ny as merged_engine
from scripts import backtest_v44_conservative_router as v44_router
from scripts import validate_regime_classifier as regime_validation

OUT_DIR = ROOT / "research_out"


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


# ── Dynamic feature computation (reused from ownership diagnostic) ──

def _build_m5(input_csv: str) -> pd.DataFrame:
    from scripts.regime_threshold_analysis import _load_m1, _resample
    m1 = _load_m1(input_csv)
    return _resample(m1, "5min")


def _compute_dynamic_features_on_m5(m5: pd.DataFrame) -> pd.DataFrame:
    n = len(m5)
    er_vals = np.full(n, 0.5)
    delta_er_vals = np.full(n, 0.0)

    for i in range(n):
        window = m5.iloc[max(0, i - 60):i + 1]
        if len(window) < 5:
            continue
        er_vals[i] = compute_efficiency_ratio(window, lookback=12)
        delta_er_vals[i] = compute_delta_efficiency_ratio(window, lookback=12, delta_bars=3)

    out = m5.copy()
    out["sf_er"] = er_vals
    out["sf_delta_er"] = delta_er_vals
    return out


def _load_classified_with_dynamic(input_csv: str) -> pd.DataFrame:
    m1 = regime_validation._load_m1(input_csv)
    featured = regime_validation.compute_features(m1)
    classified = regime_validation.classify_all_bars(featured, RegimeThresholds())
    m5_dynamic = _compute_dynamic_features_on_m5(_build_m5(input_csv))
    dynamic_cols = ["time", "sf_er", "sf_delta_er"]
    return pd.merge_asof(
        classified.sort_values("time"),
        m5_dynamic[dynamic_cols].sort_values("time"),
        on="time",
        direction="backward",
    )


def _lookup_regime_with_dynamic(
    classified: pd.DataFrame,
    time_idx: pd.DatetimeIndex,
    ts: pd.Timestamp,
) -> dict[str, Any]:
    idx = time_idx.get_indexer([pd.Timestamp(ts)], method="ffill")[0]
    if idx < 0:
        return {"regime_label": "ambiguous", "delta_er": 0.0}
    row = classified.iloc[idx]
    label = str(row.get("regime_hysteresis", "ambiguous"))
    delta_er = float(row.get("sf_delta_er", 0.0))
    if np.isnan(delta_er):
        delta_er = 0.0
    return {"regime_label": label, "delta_er": delta_er}


# ── Variant I filter ──

def _is_variant_i_blocked(regime_label: str, delta_er: float) -> bool:
    """Block when post_breakout_trend AND ΔER < 0."""
    return regime_label == "post_breakout_trend" and delta_er < 0


# ── Main backtest logic ──

def run_one(dataset: str) -> dict[str, Any]:
    starting_equity = 100000.0
    dk = _dataset_key(dataset)

    # Load Variant G baseline result (for comparison numbers)
    g_path = OUT_DIR / f"v44_conservative_router_aligned_{dk}_G_london_holdthrough.json"
    g_data = json.loads(g_path.read_text(encoding="utf-8"))
    g_summary = g_data["summary"]

    # Rebuild the Variant G pre-coupling trade list:
    # 1. Build the unfiltered baseline
    print(f"  [1/5] Building baseline trades ...")
    baseline = v44_router._build_baseline(
        dataset,
        Path(OUT_DIR / "tokyo_optimized_v14_config.json"),
        Path(OUT_DIR / "v2_exp4_winner_baseline_config.json"),
        Path(OUT_DIR / "session_momentum_v44_base_config.json"),
        starting_equity,
    )

    # 2. Apply Variant F filter (V44 only)
    print(f"  [2/5] Applying Variant F V44 filter ...")
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
        starting_equity, baseline["v14_max_units"],
    )

    # 3. Apply London V2 hold-through (produces Variant G pre-coupling trades)
    print(f"  [3/5] Applying London V2 hold-through ...")
    m1 = merged_engine._load_m1(dataset)
    g_result = v44_router._run_holdthrough_variant(
        f_pre_coupling_trades, m1, starting_equity, baseline["v14_max_units"],
    )
    # We need the pre-coupling trades from Variant G. The holdthrough function
    # applies coupling internally, but we can reconstruct pre-coupling trades
    # by running the holdthrough modification without coupling.
    # Actually, _run_holdthrough_variant applies coupling internally.
    # We need to build the Variant G trades WITHOUT coupling first, then
    # apply Variant I filter, then couple.

    # Re-do holdthrough to get modified trades before coupling
    m1_idx = pd.DatetimeIndex(m1["time"])
    g_pre_coupling: list[merged_engine.TradeRow] = []
    for trade in f_pre_coupling_trades:
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

    # 4. Build classified bars with dynamic features for Variant I filter
    print(f"  [4/5] Building classified bars with dynamic features ...")
    classified_dynamic = _load_classified_with_dynamic(dataset)
    dyn_time_idx = pd.DatetimeIndex(classified_dynamic["time"])

    # 5. Apply Variant I stand-down
    print(f"  [5/5] Applying Variant I post_breakout_trend + ΔER<0 stand-down ...")
    kept: list[merged_engine.TradeRow] = []
    blocked_trades: list[dict[str, Any]] = []

    for trade in g_pre_coupling:
        regime_info = _lookup_regime_with_dynamic(classified_dynamic, dyn_time_idx, trade.entry_time)
        if _is_variant_i_blocked(regime_info["regime_label"], regime_info["delta_er"]):
            blocked_trades.append({
                "strategy": trade.strategy,
                "entry_time": trade.entry_time.isoformat(),
                "side": trade.side,
                "pips": float(trade.pips),
                "usd": float(trade.usd),
                "exit_reason": trade.exit_reason,
                "regime_label": regime_info["regime_label"],
                "delta_er": round(regime_info["delta_er"], 4),
            })
        else:
            kept.append(trade)

    # Apply equity coupling to kept trades
    kept_sorted = sorted(kept, key=lambda t: (t.exit_time, t.entry_time))
    coupled = merged_engine._apply_shared_equity_coupling(
        kept_sorted, starting_equity, v14_max_units=baseline["v14_max_units"],
    )
    eq_curve = merged_engine._build_equity_curve(coupled, starting_equity)
    i_summary = merged_engine._stats(coupled, starting_equity, eq_curve)

    # Per-strategy detail
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

    # Blocked trade stats
    blocked_by_strategy = Counter(t["strategy"] for t in blocked_trades)
    blocked_winners = sum(1 for t in blocked_trades if t["pips"] > 0)
    blocked_losers = sum(1 for t in blocked_trades if t["pips"] <= 0)
    blocked_net_pips = sum(t["pips"] for t in blocked_trades)
    blocked_net_usd = sum(t["usd"] for t in blocked_trades)

    # Delta vs Variant G
    delta_vs_g = {
        "net_usd": round(i_summary["net_usd"] - g_summary["net_usd"], 2),
        "profit_factor": round(i_summary["profit_factor"] - g_summary["profit_factor"], 4),
        "max_drawdown_usd": round(i_summary["max_drawdown_usd"] - g_summary["max_drawdown_usd"], 2),
        "total_trades": i_summary["total_trades"] - g_summary["total_trades"],
    }

    result = {
        "dataset": Path(dataset).name,
        "variant": "I_pbt_der_standdown",
        "rule": "Block ALL strategies when regime=post_breakout_trend AND delta_ER<0",
        "variant_g_summary": g_summary,
        "variant_i_summary": i_summary,
        "delta_vs_variant_g": delta_vs_g,
        "by_strategy_detail": by_strategy_detail,
        "blocked": {
            "total": len(blocked_trades),
            "by_strategy": {k: int(v) for k, v in sorted(blocked_by_strategy.items())},
            "winners_blocked": blocked_winners,
            "losers_blocked": blocked_losers,
            "net_pips_blocked": round(blocked_net_pips, 1),
            "net_usd_blocked": round(blocked_net_usd, 2),
            "trades": blocked_trades,
        },
        "by_strategy": merged_engine._subset_breakdown(coupled, lambda t: t.strategy),
        "by_month": merged_engine._group_monthly(coupled),
    }

    # Print summary
    print(f"\n{'='*90}")
    print(f"  VARIANT I: {dk}")
    print(f"{'='*90}")
    print(f"  Rule: Block all strategies when regime=post_breakout_trend AND ΔER<0")
    print(f"\n  {'Variant G baseline':<28s} trades={g_summary['total_trades']:>4d} "
          f"net=${g_summary['net_usd']:>10.2f} PF={g_summary['profit_factor']:.3f} "
          f"DD=${g_summary['max_drawdown_usd']:>8.2f}")
    print(f"  {'Variant I':<28s} trades={i_summary['total_trades']:>4d} "
          f"net=${i_summary['net_usd']:>10.2f} PF={i_summary['profit_factor']:.3f} "
          f"DD=${i_summary['max_drawdown_usd']:>8.2f}")
    print(f"  {'Delta':<28s} trades={delta_vs_g['total_trades']:>+4d} "
          f"net=${delta_vs_g['net_usd']:>+10.2f} PF={delta_vs_g['profit_factor']:>+.4f} "
          f"DD=${delta_vs_g['max_drawdown_usd']:>+8.2f}")

    print(f"\n  Blocked trades: {len(blocked_trades)} "
          f"({blocked_winners}W / {blocked_losers}L, "
          f"net {blocked_net_pips:+.1f} pips / ${blocked_net_usd:+.2f})")
    for strat, cnt in sorted(blocked_by_strategy.items()):
        strat_blocked = [t for t in blocked_trades if t["strategy"] == strat]
        strat_pips = sum(t["pips"] for t in strat_blocked)
        print(f"    {strat}: {cnt} blocked ({strat_pips:+.1f} pips)")

    print(f"\n  Per-strategy (Variant I):")
    for strat_key in ["v14", "london_v2", "v44_ny"]:
        sd = by_strategy_detail[strat_key]
        print(f"    {strat_key:<12s} trades={sd['trades']:>4d} net=${sd['net_usd']:>10.2f} "
              f"WR={sd['win_rate_pct']:.1f}%")

    return result


def main() -> int:
    datasets = [
        ("500k", str(ROOT / "research_out" / "USDJPY_M1_OANDA_500k.csv")),
        ("1000k", str(ROOT / "research_out" / "USDJPY_M1_OANDA_1000k.csv")),
    ]

    all_results = {}
    for label, csv_path in datasets:
        if not Path(csv_path).exists():
            print(f"Skipping {label}: {csv_path} not found")
            continue
        print(f"\n{'#'*90}")
        print(f"  DATASET: {label}")
        print(f"{'#'*90}")
        result = run_one(csv_path)
        all_results[label] = result

    out_path = OUT_DIR / "variant_i_pbt_der_standdown.json"
    out_path.write_text(json.dumps(all_results, indent=2, default=_json_default), encoding="utf-8")
    print(f"\nResults written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
