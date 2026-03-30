#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
from scripts import backtest_v2_multisetup_london as v2_engine
from scripts import backtest_v44_conservative_router as v44_router

OUT_DIR = ROOT / "research_out"
V14_CFG_PATH = OUT_DIR / "tokyo_optimized_v14_config.json"
LONDON_CFG_PATH = OUT_DIR / "v2_exp4_winner_baseline_config.json"
V44_CFG_PATH = OUT_DIR / "session_momentum_v44_base_config.json"
PIP = v2_engine.PIP_SIZE
BE_OFFSET_PIPS = 1.0
TRAIL_ACTIVATE_PIPS = 8.0
TRAIL_DISTANCE_PIPS = 8.0
NY_END_HOUR = 16


def _json_default(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)


def _load_m1(input_csv: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df


def _stats_with_breakdown(trades: list[merged_engine.TradeRow], start_equity: float) -> dict[str, Any]:
    eq = merged_engine._build_equity_curve(trades, start_equity)
    return merged_engine._stats(trades, start_equity, eq)


def _build_variant_f_v44_raw(
    dataset: str,
    classified: pd.DataFrame,
    m5: pd.DataFrame,
    starting_equity: float,
) -> list[merged_engine.TradeRow]:
    v44_results, v44_embedded = v44_router._run_v44_ny_only(V44_CFG_PATH, dataset)
    if isinstance(v44_embedded, dict) and "v5_account_size" in v44_embedded:
        v44_base_eq = merged_engine._safe_float(v44_embedded.get("v5_account_size", starting_equity), starting_equity)
    else:
        v44_base_eq = merged_engine._safe_float(v44_embedded.get("v5", {}).get("account_size", starting_equity), starting_equity)
    raw_v44 = merged_engine._extract_v44_trades(v44_results, default_entry_equity=v44_base_eq)

    kept: list[merged_engine.TradeRow] = []
    variant_f = v44_router.VariantConfig(
        "F_block_plus_ambiguous_non_momentum",
        block_breakout=True,
        block_post_breakout=True,
        exhaustion_gate=False,
        block_ambiguous_non_momentum=True,
    )
    for trade in raw_v44:
        res = v44_router._filter_v44_trade(
            trade,
            classified,
            m5,
            block_breakout=variant_f.block_breakout,
            block_post_breakout=variant_f.block_post_breakout,
            block_ambiguous_non_momentum=variant_f.block_ambiguous_non_momentum,
            momentum_only=variant_f.momentum_only,
            exhaustion_gate=variant_f.exhaustion_gate,
            soft_exhaustion=variant_f.soft_exhaustion,
            er_threshold=variant_f.er_threshold,
            decay_threshold=variant_f.decay_threshold,
        )
        if not res.blocked:
            kept.append(trade)
    return kept


def _load_london_config() -> dict[str, Any]:
    return json.loads(LONDON_CFG_PATH.read_text(encoding="utf-8"))


def _simulate_holdthrough_trade(
    m1: pd.DataFrame,
    m1_idx: pd.DatetimeIndex,
    row: pd.Series,
    *,
    include_loss_group: bool,
    tp1_close_fraction: float,
) -> dict[str, Any] | None:
    exit_reason = str(row["exit_reason"])
    if exit_reason not in {"HARD_CLOSE", "TP1_ONLY_HARD_CLOSE"}:
        return None

    side = "buy" if str(row["direction"]).lower() == "long" else "sell"
    entry_time = pd.Timestamp(row["entry_time_utc"])
    hardclose_time = pd.Timestamp(row["exit_time_utc"])
    entry_price = float(row["entry_price"])
    hardclose_price = float(row["exit_price"])
    initial_units = int(row["position_units"])

    deadline = hardclose_time.replace(hour=NY_END_HOUR, minute=0, second=0, microsecond=0)
    if deadline <= hardclose_time:
        return None

    forward = m1.loc[(m1_idx > hardclose_time) & (m1_idx <= deadline)]
    if forward.empty:
        return None

    if side == "buy":
        runner_pips_at_hc = (hardclose_price - entry_price) / PIP
    else:
        runner_pips_at_hc = (entry_price - hardclose_price) / PIP

    if not include_loss_group and runner_pips_at_hc <= 0:
        return None

    tp1_units = 0
    runner_units = initial_units
    locked_pips_sum = 0.0
    locked_usd = 0.0
    if exit_reason == "TP1_ONLY_HARD_CLOSE":
        tp1_units = int(np.floor(initial_units * tp1_close_fraction / v2_engine.ROUND_UNITS) * v2_engine.ROUND_UNITS)
        tp1_units = max(0, min(initial_units, tp1_units))
        runner_units = initial_units - tp1_units
        tp1_price = float(row["tp1_price"])
        tp1_pips, tp1_usd = v2_engine.calc_leg_pnl(str(row["direction"]).lower(), entry_price, tp1_price, tp1_units)
        locked_pips_sum += tp1_pips * tp1_units
        locked_usd += tp1_usd

    if runner_units <= 0:
        return None

    best_price = hardclose_price
    if runner_pips_at_hc >= TRAIL_ACTIVATE_PIPS:
        trailing_active = True
        if side == "buy":
            stop_price = max(best_price - TRAIL_DISTANCE_PIPS * PIP, entry_price + BE_OFFSET_PIPS * PIP)
        else:
            stop_price = min(best_price + TRAIL_DISTANCE_PIPS * PIP, entry_price - BE_OFFSET_PIPS * PIP)
    elif runner_pips_at_hc > 0:
        trailing_active = False
        stop_price = entry_price + BE_OFFSET_PIPS * PIP if side == "buy" else entry_price - BE_OFFSET_PIPS * PIP
    else:
        trailing_active = False
        stop_price = entry_price

    sim_exit_price = None
    sim_exit_time = None
    sim_exit_type = None

    for _, bar in forward.iterrows():
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])
        bar_close = float(bar["close"])
        bar_time = pd.Timestamp(bar["time"])

        if side == "buy":
            if bar_high > best_price:
                best_price = bar_high
            profit_from_entry = (best_price - entry_price) / PIP
            if not trailing_active and profit_from_entry >= TRAIL_ACTIVATE_PIPS:
                trailing_active = True
            if trailing_active:
                candidate = max(best_price - TRAIL_DISTANCE_PIPS * PIP, entry_price + BE_OFFSET_PIPS * PIP)
                stop_price = max(stop_price, candidate)
            if bar_low <= stop_price:
                sim_exit_price = stop_price
                sim_exit_time = bar_time
                sim_exit_type = "trail_stop" if trailing_active else "be_stop"
                break
        else:
            if bar_low < best_price:
                best_price = bar_low
            profit_from_entry = (entry_price - best_price) / PIP
            if not trailing_active and profit_from_entry >= TRAIL_ACTIVATE_PIPS:
                trailing_active = True
            if trailing_active:
                candidate = min(best_price + TRAIL_DISTANCE_PIPS * PIP, entry_price - BE_OFFSET_PIPS * PIP)
                stop_price = min(stop_price, candidate)
            if bar_high >= stop_price:
                sim_exit_price = stop_price
                sim_exit_time = bar_time
                sim_exit_type = "trail_stop" if trailing_active else "be_stop"
                break

    if sim_exit_price is None:
        last_bar = forward.iloc[-1]
        sim_exit_price = float(last_bar["close"])
        sim_exit_time = pd.Timestamp(last_bar["time"])
        sim_exit_type = "ny_session_close"

    runner_pips_new, runner_usd_new = v2_engine.calc_leg_pnl(str(row["direction"]).lower(), entry_price, sim_exit_price, runner_units)
    runner_pips_old, runner_usd_old = v2_engine.calc_leg_pnl(str(row["direction"]).lower(), entry_price, hardclose_price, runner_units)

    total_pips_sum_new = locked_pips_sum + runner_pips_new * runner_units
    total_usd_new = locked_usd + runner_usd_new
    total_pips_new = total_pips_sum_new / initial_units if initial_units > 0 else 0.0

    total_pips_sum_old = locked_pips_sum + runner_pips_old * runner_units
    total_usd_old = locked_usd + runner_usd_old
    total_pips_old = total_pips_sum_old / initial_units if initial_units > 0 else 0.0

    return {
        "new_exit_time": sim_exit_time,
        "new_exit_price": float(sim_exit_price),
        "new_exit_reason": f"HT_{sim_exit_type.upper()}",
        "new_pnl_pips": float(total_pips_new),
        "new_pnl_usd": float(total_usd_new),
        "delta_pips": float(total_pips_new - total_pips_old),
        "delta_usd": float(total_usd_new - total_usd_old),
        "held_improved": total_usd_new > total_usd_old,
        "held_worsened": total_usd_new < total_usd_old,
        "hold_exit_type": sim_exit_type,
        "runner_units": int(runner_units),
        "tp1_units": int(tp1_units),
        "runner_pips_at_hc": float(runner_pips_at_hc),
    }


def _apply_holdthrough_variant(
    v2_trades_df: pd.DataFrame,
    m1: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    include_loss_group: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    m1_idx = pd.DatetimeIndex(m1["time"])
    out = v2_trades_df.copy()
    held = 0
    improved = 0
    worsened = 0
    add_pips = 0.0
    add_usd = 0.0
    exit_types = Counter()
    tp1_fraction_by_setup = {
        setup: float(cfg["setups"][setup]["tp1_close_fraction"])
        for setup in cfg.get("setups", {})
    }

    for idx, row in out.iterrows():
        reason = str(row["exit_reason"])
        if reason not in {"HARD_CLOSE", "TP1_ONLY_HARD_CLOSE"}:
            continue
        if not include_loss_group:
            entry = float(row["entry_price"])
            hc = float(row["exit_price"])
            side = "buy" if str(row["direction"]).lower() == "long" else "sell"
            pips_at_hc = (hc - entry) / PIP if side == "buy" else (entry - hc) / PIP
            if pips_at_hc <= 0:
                continue
        setup = str(row["setup_type"])
        sim = _simulate_holdthrough_trade(
            m1,
            m1_idx,
            row,
            include_loss_group=include_loss_group,
            tp1_close_fraction=tp1_fraction_by_setup[setup],
        )
        if sim is None:
            continue
        held += 1
        improved += int(sim["held_improved"])
        worsened += int(sim["held_worsened"])
        add_pips += sim["delta_pips"]
        add_usd += sim["delta_usd"]
        exit_types[sim["hold_exit_type"]] += 1

        out.at[idx, "exit_time_utc"] = sim["new_exit_time"]
        out.at[idx, "exit_price"] = sim["new_exit_price"]
        out.at[idx, "pnl_pips"] = sim["new_pnl_pips"]
        out.at[idx, "pnl_usd"] = sim["new_pnl_usd"]
        out.at[idx, "exit_reason"] = sim["new_exit_reason"]
        out.at[idx, "trade_duration_minutes"] = int((pd.Timestamp(sim["new_exit_time"]) - pd.Timestamp(row["entry_time_utc"])).total_seconds() / 60.0)

    stats = {
        "trades_held": int(held),
        "held_improved": int(improved),
        "held_worsened": int(worsened),
        "additional_pips": round(add_pips, 1),
        "additional_usd": round(add_usd, 2),
        "exit_distribution": {k: int(v) for k, v in sorted(exit_types.items())},
    }
    return out, stats


def _strategy_breakdown(trades: list[merged_engine.TradeRow], start_equity: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for strategy in ["v14", "london_v2", "v44_ny"]:
        subset = [t for t in trades if t.strategy == strategy]
        out[strategy] = _stats_with_breakdown(subset, start_equity)
    return out


def _build_variant_f_baseline(dataset: str) -> tuple[dict[str, Any], list[merged_engine.TradeRow], int]:
    starting_equity = 100000.0
    classified = v44_router._load_classified_bars(dataset)
    m5 = v44_router._build_m5(dataset)
    m1 = merged_engine._load_m1(dataset)
    v14_report, v14_cfg = merged_engine._run_v14_in_process(V14_CFG_PATH, dataset)
    v14_raw = merged_engine._extract_v14_trades(v14_report, default_entry_equity=starting_equity)
    v2_df, _, _ = merged_engine._run_v2_in_process(LONDON_CFG_PATH, m1)
    v2_raw = merged_engine._extract_v2_trades(v2_df, default_entry_equity=starting_equity)
    v44_raw = _build_variant_f_v44_raw(dataset, classified, m5, starting_equity)
    all_raw = sorted(v14_raw + v2_raw + v44_raw, key=lambda t: (t.exit_time, t.entry_time))
    v14_max_units = merged_engine._safe_int(v14_cfg.get("position_sizing", {}).get("max_units", 500000), 500000)
    coupled = merged_engine._apply_shared_equity_coupling(all_raw, starting_equity, v14_max_units=v14_max_units)
    summary = merged_engine._stats(coupled, starting_equity, merged_engine._build_equity_curve(coupled, starting_equity))
    return {"summary": summary, "classified": classified, "m5": m5, "m1": m1, "v2_df": v2_df, "v14_raw": v14_raw, "v44_raw": v44_raw}, coupled, v14_max_units


def _build_variant_from_context(
    baseline_ctx: dict[str, Any],
    v14_max_units: int,
    variant_name: str,
    include_loss_group: bool,
) -> dict[str, Any]:
    starting_equity = 100000.0
    london_cfg = _load_london_config()
    modified_v2_df, ht_stats = _apply_holdthrough_variant(
        baseline_ctx["v2_df"],
        baseline_ctx["m1"],
        london_cfg,
        include_loss_group=include_loss_group,
    )
    modified_v2_raw = merged_engine._extract_v2_trades(modified_v2_df, default_entry_equity=starting_equity)
    all_raw = sorted(baseline_ctx["v14_raw"] + modified_v2_raw + baseline_ctx["v44_raw"], key=lambda t: (t.exit_time, t.entry_time))
    coupled = merged_engine._apply_shared_equity_coupling(all_raw, starting_equity, v14_max_units=v14_max_units)
    summary = merged_engine._stats(coupled, starting_equity, merged_engine._build_equity_curve(coupled, starting_equity))
    baseline_summary = baseline_ctx["summary"]
    return {
        "baseline_summary": baseline_summary,
        "summary": summary,
        "by_strategy": _strategy_breakdown(coupled, starting_equity),
        "holdthrough_stats": ht_stats,
        "delta_vs_baseline": {
            "net_usd": round(summary["net_usd"] - baseline_summary["net_usd"], 2),
            "pf": round(summary["profit_factor"] - baseline_summary["profit_factor"], 4),
            "max_dd_usd": round(summary["max_drawdown_usd"] - baseline_summary["max_drawdown_usd"], 2),
            "total_trades": int(summary["total_trades"] - baseline_summary["total_trades"]),
        },
        "variant_name": variant_name,
    }


def run_one(dataset: str) -> dict[str, Any]:
    baseline_ctx, _, v14_max_units = _build_variant_f_baseline(dataset)
    ht_a = _build_variant_from_context(baseline_ctx, v14_max_units, "variant_ht_a", include_loss_group=True)
    ht_b = _build_variant_from_context(baseline_ctx, v14_max_units, "variant_ht_b", include_loss_group=False)
    return {
        "dataset": Path(dataset).name,
        "baseline": baseline_ctx["summary"],
        "variant_ht_a": {
            "summary": ht_a["summary"],
            "by_strategy": ht_a["by_strategy"],
            "holdthrough_stats": ht_a["holdthrough_stats"],
            "delta_vs_baseline": ht_a["delta_vs_baseline"],
        },
        "variant_ht_b": {
            "summary": ht_b["summary"],
            "by_strategy": ht_b["by_strategy"],
            "holdthrough_stats": ht_b["holdthrough_stats"],
            "delta_vs_baseline": ht_b["delta_vs_baseline"],
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Integrated Phase 3 backtest with London V2 hold-through variants")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output-prefix", required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    dataset = str(Path(args.dataset).resolve())
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    result = run_one(dataset)
    out_path = output_prefix.with_name(output_prefix.name + "_results.json")
    out_path.write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")

    base = result["baseline"]
    a = result["variant_ht_a"]
    b = result["variant_ht_b"]
    print("=" * 110)
    print(f"V2 HOLDTHROUGH: {Path(dataset).name}")
    print("=" * 110)
    print(f"{'Variant':<18s} {'Trades':>6s} {'Net USD':>12s} {'ΔNet':>10s} {'PF':>7s} {'ΔPF':>8s} {'DD':>10s} {'ΔDD':>10s}")
    print("-" * 110)
    print(f"{'BASELINE':<18s} {base['total_trades']:>6d} {base['net_usd']:>12.2f} {'':>10s} {base['profit_factor']:>7.3f} {'':>8s} {base['max_drawdown_usd']:>10.2f} {'':>10s}")
    for name, block in [('HT-A', a), ('HT-B', b)]:
        s = block["summary"]
        d = block["delta_vs_baseline"]
        print(f"{name:<18s} {s['total_trades']:>6d} {s['net_usd']:>12.2f} {d['net_usd']:>+10.2f} {s['profit_factor']:>7.3f} {d['pf']:>+8.4f} {s['max_drawdown_usd']:>10.2f} {d['max_dd_usd']:>+10.2f}")
        ht = block["holdthrough_stats"]
        print(f"{'':<18s} held={ht['trades_held']:>4d} improved={ht['held_improved']:>4d} worsened={ht['held_worsened']:>4d} add_pips={ht['additional_pips']:+.1f} add_usd={ht['additional_usd']:+.2f} exits={ht['exit_distribution']}")
    print(f"Output: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
