#!/usr/bin/env python3
"""
Variant J: Positive ownership rule for V44's owned ambiguous cell.

Builds on Variant I (= Variant F V44 filter + London V2 hold-through + global
stand-down in post_breakout_trend with negative ΔER).

Adds one positive authorization refinement for V44:
  - Keep the existing breakout / post_breakout_trend blocks
  - In ambiguous, allow V44 only when:
      * momentum is the top regime score
      * ER bucket is mid (0.35 <= ER < 0.55)
      * ΔER is negative

This turns the ownership diagnostic into one narrow, testable router rule.
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
STARTING_EQUITY = 100000.0


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


def _build_m5(input_csv: str) -> pd.DataFrame:
    from scripts.regime_threshold_analysis import _load_m1, _resample

    m1 = _load_m1(input_csv)
    return _resample(m1, "5min")


def _compute_dynamic_features_on_m5(m5: pd.DataFrame) -> pd.DataFrame:
    n = len(m5)
    er_vals = np.full(n, 0.5)
    delta_er_vals = np.full(n, 0.0)

    for i in range(n):
        window = m5.iloc[max(0, i - 60) : i + 1]
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


def _lookup_dynamic_row(
    classified: pd.DataFrame,
    time_idx: pd.DatetimeIndex,
    ts: pd.Timestamp,
) -> pd.Series | None:
    idx = time_idx.get_indexer([pd.Timestamp(ts)], method="ffill")[0]
    if idx < 0:
        return None
    return classified.iloc[idx]


def _is_er_mid(er: float) -> bool:
    return 0.35 <= er < 0.55


def _build_variant_j_pre_coupling(
    dataset: str,
) -> tuple[list[merged_engine.TradeRow], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    baseline = v44_router._build_baseline(
        dataset,
        Path(OUT_DIR / "tokyo_optimized_v14_config.json"),
        Path(OUT_DIR / "v2_exp4_winner_baseline_config.json"),
        Path(OUT_DIR / "session_momentum_v44_base_config.json"),
        STARTING_EQUITY,
    )

    classified = _load_classified_with_dynamic(dataset)
    time_idx = pd.DatetimeIndex(classified["time"])

    kept_pre: list[merged_engine.TradeRow] = []
    blocked_v44: list[dict[str, Any]] = []

    for trade in baseline["trades"]:
        if trade.strategy != "v44_ny":
            kept_pre.append(trade)
            continue

        regime = v44_router._lookup_regime(classified, trade.entry_time)
        row = _lookup_dynamic_row(classified, time_idx, trade.entry_time)
        er = float(row.get("sf_er", 0.5)) if row is not None else 0.5
        delta_er = float(row.get("sf_delta_er", 0.0)) if row is not None else 0.0
        if np.isnan(er):
            er = 0.5
        if np.isnan(delta_er):
            delta_er = 0.0
        label = regime["regime_label"]
        scores = regime["regime_scores"]
        top_regime = max(scores, key=scores.get) if scores else "unknown"

        reason = ""
        blocked = False
        if label == "breakout":
            blocked = True
            reason = "blocked_breakout"
        elif label == "post_breakout_trend":
            blocked = True
            reason = "blocked_post_breakout_trend"
        elif label == "ambiguous":
            allow_ambiguous = top_regime == "momentum" and _is_er_mid(er) and delta_er < 0
            if not allow_ambiguous:
                blocked = True
                reason = f"blocked_ambiguous_not_v44_owned_cell_top_{top_regime}"

        if blocked:
            blocked_v44.append(
                {
                    "strategy": trade.strategy,
                    "entry_time": trade.entry_time.isoformat(),
                    "side": trade.side,
                    "pips": float(trade.pips),
                    "usd": float(trade.usd),
                    "exit_reason": trade.exit_reason,
                    "regime_label": label,
                    "top_regime": top_regime,
                    "er": round(er, 4),
                    "delta_er": round(delta_er, 4),
                    "reason": reason,
                }
            )
        else:
            kept_pre.append(trade)

    # Apply London hold-through like Variant G before the global stand-down.
    m1 = merged_engine._load_m1(dataset)
    m1_idx = pd.DatetimeIndex(m1["time"])
    held_pre: list[merged_engine.TradeRow] = []
    for trade in kept_pre:
        if trade.strategy != "london_v2":
            held_pre.append(trade)
            continue
        sim = v44_router._simulate_london_holdthrough(trade, m1, m1_idx)
        if sim is None:
            held_pre.append(trade)
            continue
        new_raw = dict(trade.raw or {})
        new_raw["exit_time_utc"] = sim["new_exit_time"]
        new_raw["exit_reason"] = sim["new_exit_reason"]
        new_raw["pnl_pips"] = sim["new_pips"]
        new_raw["pnl_usd"] = sim["new_usd"]
        held_pre.append(
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

    # Apply Variant I stand-down before coupling.
    kept_after_i: list[merged_engine.TradeRow] = []
    blocked_global: list[dict[str, Any]] = []
    for trade in held_pre:
        row = _lookup_dynamic_row(classified, time_idx, trade.entry_time)
        label = str(row.get("regime_hysteresis", "ambiguous")) if row is not None else "ambiguous"
        delta_er = float(row.get("sf_delta_er", 0.0)) if row is not None else 0.0
        if np.isnan(delta_er):
            delta_er = 0.0
        if label == "post_breakout_trend" and delta_er < 0:
            blocked_global.append(
                {
                    "strategy": trade.strategy,
                    "entry_time": trade.entry_time.isoformat(),
                    "side": trade.side,
                    "pips": float(trade.pips),
                    "usd": float(trade.usd),
                    "exit_reason": trade.exit_reason,
                    "regime_label": label,
                    "delta_er": round(delta_er, 4),
                    "reason": "blocked_global_pbt_der_neg",
                }
            )
        else:
            kept_after_i.append(trade)

    return kept_after_i, baseline, classified, blocked_v44 + blocked_global


def run_one(dataset: str) -> dict[str, Any]:
    dk = _dataset_key(dataset)
    variant_i = json.loads((OUT_DIR / "variant_i_pbt_der_standdown.json").read_text(encoding="utf-8"))[dk]
    variant_i_summary = variant_i["variant_i_summary"]

    pre_coupling, baseline, _classified, blocked = _build_variant_j_pre_coupling(dataset)
    coupled = merged_engine._apply_shared_equity_coupling(
        sorted(pre_coupling, key=lambda t: (t.exit_time, t.entry_time)),
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

    blocked_by_reason = Counter(t["reason"] for t in blocked)
    blocked_winners = sum(1 for t in blocked if t["pips"] > 0)
    blocked_losers = sum(1 for t in blocked if t["pips"] <= 0)
    blocked_v44_only = [t for t in blocked if t["strategy"] == "v44_ny"]
    blocked_v44_ownedcell_filter = [
        t for t in blocked_v44_only if t["reason"].startswith("blocked_ambiguous_not_v44_owned_cell")
    ]

    delta_vs_i = {
        "net_usd": round(summary["net_usd"] - variant_i_summary["net_usd"], 2),
        "profit_factor": round(summary["profit_factor"] - variant_i_summary["profit_factor"], 4),
        "max_drawdown_usd": round(summary["max_drawdown_usd"] - variant_i_summary["max_drawdown_usd"], 2),
        "total_trades": summary["total_trades"] - variant_i_summary["total_trades"],
    }

    return {
        "dataset": dk,
        "variant": "J_v44_owned_ambiguous_cell",
        "rule": "V44 ambiguous entries are allowed only when momentum is top score, ER is mid, and ΔER < 0. Variant I global stand-down still applies.",
        "variant_i_summary": variant_i_summary,
        "variant_j_summary": summary,
        "delta_vs_variant_i": delta_vs_i,
        "by_strategy_detail": by_strategy_detail,
        "blocked": {
            "total": len(blocked),
            "winners_blocked": blocked_winners,
            "losers_blocked": blocked_losers,
            "net_pips_blocked": round(sum(t["pips"] for t in blocked), 1),
            "net_usd_blocked": round(sum(t["usd"] for t in blocked), 2),
            "by_reason": dict(blocked_by_reason),
            "v44_ownedcell_filter_count": len(blocked_v44_ownedcell_filter),
            "v44_ownedcell_filter_winners": sum(1 for t in blocked_v44_ownedcell_filter if t["pips"] > 0),
            "v44_ownedcell_filter_losers": sum(1 for t in blocked_v44_ownedcell_filter if t["pips"] <= 0),
            "v44_ownedcell_filter_net_pips": round(sum(t["pips"] for t in blocked_v44_ownedcell_filter), 1),
            "sample": blocked[:20],
        },
    }


def main() -> int:
    datasets = [
        str(OUT_DIR / "USDJPY_M1_OANDA_500k.csv"),
        str(OUT_DIR / "USDJPY_M1_OANDA_1000k.csv"),
    ]

    results = {}
    for dataset in datasets:
        if not Path(dataset).exists():
            print(f"Skipping missing dataset: {dataset}")
            continue
        dk = _dataset_key(dataset)
        print(f"\nRunning Variant J on {dk} ...")
        results[dk] = run_one(dataset)

    out_path = OUT_DIR / "variant_j_v44_owned_ambiguous.json"
    out_path.write_text(json.dumps(results, indent=2, default=_json_default), encoding="utf-8")
    print(f"\nWrote results to {out_path}")

    for dk, data in results.items():
        i_sum = data["variant_i_summary"]
        j_sum = data["variant_j_summary"]
        delta = data["delta_vs_variant_i"]
        print(f"\n=== {dk} Variant J vs Variant I ===")
        print(
            f"Variant I: trades={i_sum['total_trades']} net={i_sum['net_usd']:.2f} "
            f"PF={i_sum['profit_factor']:.4f} DD={i_sum['max_drawdown_usd']:.2f}"
        )
        print(
            f"Variant J: trades={j_sum['total_trades']} net={j_sum['net_usd']:.2f} "
            f"PF={j_sum['profit_factor']:.4f} DD={j_sum['max_drawdown_usd']:.2f}"
        )
        print(
            f"Delta: net={delta['net_usd']:+.2f} PF={delta['profit_factor']:+.4f} "
            f"DD={delta['max_drawdown_usd']:+.2f} trades={delta['total_trades']:+d}"
        )
        blocked = data["blocked"]
        print(
            f"Blocked by owned-cell filter: {blocked['v44_ownedcell_filter_count']} "
            f"({blocked['v44_ownedcell_filter_winners']}W/{blocked['v44_ownedcell_filter_losers']}L), "
            f"net {blocked['v44_ownedcell_filter_net_pips']:+.1f} pips"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
