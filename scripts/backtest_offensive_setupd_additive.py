#!/usr/bin/env python3
"""
Narrow offensive additive backtest:

baseline:
    current promoted stack (Variant K coupled)

variant:
    baseline + London Setup D long-only shadow trades in the strongest
    stable positive cells, excluding exact matches to existing baseline
    London trades

Outputs overlap/additivity metrics and portfolio deltas.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

ROOT = Path("/Users/codygnon/Documents/usdjpy_assistant")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import backtest_merged_integrated_tokyo_london_v2_ny as merged_engine
from scripts import backtest_variant_k_london_cluster as variant_k
from scripts import backtest_v2_multisetup_london as london_v2_engine
from scripts import diagnostic_london_setupd_trade_outcomes as setupd_outcomes

OUT_DIR = ROOT / "research_out"
STARTING_EQUITY = 100000.0
OUTPUT_PATH = OUT_DIR / "offensive_setupd_additive_vs_variant_k.json"
SIZE_SWEEP_OUTPUT_PATH = OUT_DIR / "offensive_setupd_additive_vs_variant_k_size_sweep.json"
TRADE_OUTCOME_ARGS = SimpleNamespace(
    spread_pips=0.3,
    sl_buffer_pips=3.0,
    sl_min_pips=5.0,
    sl_max_pips=20.0,
    tp1_r_multiple=1.0,
    tp2_r_multiple=2.0,
    tp1_close_fraction=0.5,
    be_offset_pips=1.0,
)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return str(obj)


def _dataset_key(path: str) -> str:
    name = Path(path).name
    if "500k" in name:
        return "500k"
    if "1000k" in name:
        return "1000k"
    return name


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Additive backtest for offensive London Setup D overlays.")
    p.add_argument(
        "--target-cell",
        action="append",
        dest="target_cells",
        help="Ownership cell to test. Repeat for multiple cells. If omitted, auto-select top 2 stable positive cells.",
    )
    p.add_argument(
        "--size-scale",
        action="append",
        dest="size_scales",
        type=float,
        help="Setup D sizing scale(s), e.g. --size-scale 0.5 --size-scale 0.25. Defaults to 1.0,0.5,0.25.",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output path. Defaults to the standard additive output for auto mode, or a descriptive path for custom cells.",
    )
    p.add_argument(
        "--filter-name",
        type=str,
        default=None,
        help="Optional trade filter. Supported: first_30min",
    )
    return p.parse_args()


def _load_london_cfg() -> dict[str, Any]:
    return json.loads((OUT_DIR / "v2_exp4_winner_baseline_config.json").read_text(encoding="utf-8"))


def _choose_target_cells(reports: list[dict[str, Any]]) -> list[str]:
    """
    Choose the top 1-2 stable positive cells from long-only native-allowed Setup D trades.

    Criteria:
      - avg_pnl_pips > 2.0 on both datasets
      - count >= 15 on both datasets
      - rank by combined total pips saved / produced
    """
    per_dataset: dict[str, dict[str, dict[str, float]]] = {}
    for report in reports:
        ds = report["dataset_key"]
        cell_metrics: dict[str, dict[str, float]] = {}
        groups: dict[str, list[dict[str, Any]]] = {}
        for t in report["_all_trades"]:
            if not (t["native_allowed"] and t["direction"] == "long"):
                continue
            groups.setdefault(t["ownership_cell"], []).append(t)
        for cell, trades in groups.items():
            pnls = [float(t["pnl_pips"]) for t in trades]
            cell_metrics[cell] = {
                "count": len(trades),
                "avg_pnl_pips": sum(pnls) / len(pnls) if pnls else 0.0,
                "total_pnl_pips": sum(pnls),
            }
        per_dataset[ds] = cell_metrics

    all_cells = set()
    for metrics in per_dataset.values():
        all_cells.update(metrics.keys())

    candidates: list[tuple[str, float]] = []
    for cell in sorted(all_cells):
        ds500 = per_dataset.get("500k", {}).get(cell, {"count": 0, "avg_pnl_pips": -999.0, "total_pnl_pips": 0.0})
        ds1000 = per_dataset.get("1000k", {}).get(cell, {"count": 0, "avg_pnl_pips": -999.0, "total_pnl_pips": 0.0})
        if ds500["count"] >= 15 and ds1000["count"] >= 15 and ds500["avg_pnl_pips"] > 2.0 and ds1000["avg_pnl_pips"] > 2.0:
            score = float(ds500["total_pnl_pips"] + ds1000["total_pnl_pips"])
            candidates.append((cell, score))

    candidates.sort(key=lambda kv: (-kv[1], kv[0]))
    return [cell for cell, _ in candidates[:2]]


def _calc_setupd_usd(trade: dict[str, Any], cfg: dict[str, Any], size_scale: float = 1.0) -> tuple[float, int]:
    """
    Reconstruct native-style USD from the shadow trade record using London's sizing model.
    """
    setup_cfg = cfg["setups"]["D"]
    risk_pct_trade = float(setup_cfg.get("risk_per_trade_pct", cfg["risk"]["risk_per_trade_pct"]))
    leverage = float(cfg["account"]["leverage"])
    max_margin_frac = float(cfg["account"]["max_margin_usage_fraction_per_trade"])
    entry_price = float(trade["entry_price"])
    risk_pips = float(trade["risk_pips"])
    risk_usd = STARTING_EQUITY * risk_pct_trade * float(size_scale)
    units = int(
        math.floor(
            risk_usd / max(1e-9, risk_pips * london_v2_engine.pip_value_per_unit(entry_price)) / london_v2_engine.ROUND_UNITS
        ) * london_v2_engine.ROUND_UNITS
    )
    if units <= 0:
        return 0.0, 0

    req_margin = (units * entry_price * london_v2_engine.PIP_SIZE) / leverage
    free_margin = STARTING_EQUITY
    if req_margin > max_margin_frac * free_margin:
        return 0.0, 0

    direction = str(trade["direction"])
    exit_reason = str(trade["exit_reason"])
    tp1_fraction = float(cfg["setups"]["D"]["tp1_close_fraction"])
    tp1_units = int(math.floor(units * tp1_fraction / london_v2_engine.ROUND_UNITS) * london_v2_engine.ROUND_UNITS)
    tp1_units = max(0, min(units, tp1_units))
    runner_units = units - tp1_units

    entry = float(trade["entry_price"])
    exit_px = float(trade["exit_price"])
    tp1 = float(trade["tp1_price"])
    tp2 = float(trade["tp2_price"])
    be = float(trade["be_price"])

    if exit_reason == "TP1_PARTIAL_THEN_TP2":
        _p1, usd1 = london_v2_engine.calc_leg_pnl(direction, entry, tp1, tp1_units)
        _p2, usd2 = london_v2_engine.calc_leg_pnl(direction, entry, tp2, runner_units)
        return float(usd1 + usd2), units
    if exit_reason == "TP1_PARTIAL_THEN_BE":
        _p1, usd1 = london_v2_engine.calc_leg_pnl(direction, entry, tp1, tp1_units)
        _p2, usd2 = london_v2_engine.calc_leg_pnl(direction, entry, be, runner_units)
        return float(usd1 + usd2), units
    if exit_reason == "TP1_PARTIAL_THEN_HARD_CLOSE":
        _p1, usd1 = london_v2_engine.calc_leg_pnl(direction, entry, tp1, tp1_units)
        _p2, usd2 = london_v2_engine.calc_leg_pnl(direction, entry, exit_px, runner_units)
        return float(usd1 + usd2), units

    _p, usd = london_v2_engine.calc_leg_pnl(direction, entry, exit_px, units)
    return float(usd), units


def _daily_cap_first(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen_days: set[str] = set()
    for t in sorted(trades, key=lambda x: pd.Timestamp(x["entry_time"])):
        day = pd.Timestamp(t["entry_time"]).floor("D").isoformat()
        if day in seen_days:
            continue
        seen_days.add(day)
        out.append(t)
    return out


def _passes_named_filter(trade: dict[str, Any], filter_name: str | None) -> bool:
    if not filter_name:
        return True
    if filter_name == "first_30min":
        signal_time = pd.Timestamp(trade["signal_time"])
        if signal_time.tzinfo is None:
            signal_time = signal_time.tz_localize("UTC")
        day = signal_time.floor("D")
        london_open_hour = london_v2_engine.uk_london_open_utc(day)
        london_open = day + pd.Timedelta(hours=london_open_hour)
        minutes_since_open = (signal_time - london_open).total_seconds() / 60.0
        return 15.0 <= minutes_since_open <= 45.0
    raise ValueError(f"Unsupported filter: {filter_name}")


def _to_trade_row(trade: dict[str, Any], cfg: dict[str, Any], size_scale: float = 1.0) -> merged_engine.TradeRow | None:
    usd, units = _calc_setupd_usd(trade, cfg, size_scale=size_scale)
    if units <= 0:
        return None
    entry_time = pd.Timestamp(trade["entry_time"])
    exit_time = pd.Timestamp(trade["exit_time"])
    if entry_time.tzinfo is None:
        entry_time = entry_time.tz_localize("UTC")
    if exit_time.tzinfo is None:
        exit_time = exit_time.tz_localize("UTC")
    raw = dict(trade)
    raw["position_units"] = int(units)
    raw["setup_type"] = "D"
    raw["strategy_mode"] = "offensive_setupd_cell_gated"
    raw["size_scale"] = float(size_scale)
    return merged_engine.TradeRow(
        strategy="london_v2_setupd_off",
        entry_time=entry_time.tz_convert("UTC"),
        exit_time=exit_time.tz_convert("UTC"),
        entry_session="london",
        side="buy",
        pips=float(trade["pnl_pips"]),
        usd=float(usd),
        exit_reason=str(trade["exit_reason"]),
        standalone_entry_equity=float(STARTING_EQUITY),
        raw=raw,
    )


def _has_exact_baseline_match(trade: dict[str, Any], baseline_trades: list[merged_engine.TradeRow]) -> bool:
    entry_time = pd.Timestamp(trade["entry_time"])
    if entry_time.tzinfo is None:
        entry_time = entry_time.tz_localize("UTC")
    side = "buy" if str(trade["direction"]) == "long" else "sell"
    for t in baseline_trades:
        if t.strategy == "london_v2" and pd.Timestamp(t.entry_time) == entry_time.tz_convert("UTC") and t.side == side:
            return True
    return False


def _intervals_overlap(a0: pd.Timestamp, a1: pd.Timestamp, b0: pd.Timestamp, b1: pd.Timestamp) -> bool:
    return max(a0, b0) < min(a1, b1)


def _build_variant_result(
    *,
    baseline_kept: list[merged_engine.TradeRow],
    baseline_coupled: list[merged_engine.TradeRow],
    baseline_meta: dict[str, Any],
    baseline_summary: dict[str, Any],
    target_cells: list[str],
    report: dict[str, Any],
    cfg: dict[str, Any],
    size_scale: float,
    filter_name: str | None,
) -> dict[str, Any]:
    scale_label = f"{int(size_scale * 100)}pct"
    all_setupd = [
        t
        for t in report["_all_trades"]
        if t["native_allowed"]
        and t["direction"] == "long"
        and t["ownership_cell"] in target_cells
        and _passes_named_filter(t, filter_name)
    ]
    capped = _daily_cap_first(all_setupd)

    exact_overlap = [t for t in capped if _has_exact_baseline_match(t, baseline_coupled)]
    additive_candidates = [t for t in capped if not _has_exact_baseline_match(t, baseline_coupled)]

    offensive_rows = [r for r in (_to_trade_row(t, cfg, size_scale=size_scale) for t in additive_candidates) if r is not None]

    displaced_baseline: list[merged_engine.TradeRow] = []
    for bt in baseline_coupled:
        if any(_intervals_overlap(pd.Timestamp(bt.entry_time), pd.Timestamp(bt.exit_time), pd.Timestamp(ot.entry_time), pd.Timestamp(ot.exit_time)) for ot in offensive_rows):
            displaced_baseline.append(bt)

    variant_coupled = merged_engine._apply_shared_equity_coupling(
        sorted(baseline_kept + offensive_rows, key=lambda t: (t.exit_time, t.entry_time)),
        STARTING_EQUITY,
        v14_max_units=baseline_meta["v14_max_units"],
    )
    variant_eq = merged_engine._build_equity_curve(variant_coupled, STARTING_EQUITY)
    variant_summary = merged_engine._stats(variant_coupled, STARTING_EQUITY, variant_eq)

    delta = {
        "total_trades": int(variant_summary["total_trades"] - baseline_summary["total_trades"]),
        "net_usd": round(variant_summary["net_usd"] - baseline_summary["net_usd"], 2),
        "profit_factor": round(variant_summary["profit_factor"] - baseline_summary["profit_factor"], 4),
        "max_drawdown_usd": round(variant_summary["max_drawdown_usd"] - baseline_summary["max_drawdown_usd"], 2),
    }

    return {
        "size_scale": float(size_scale),
        "size_label": scale_label,
        "filter_name": filter_name,
        "selection_counts": {
            "raw_native_long_trades_in_cells": len(all_setupd),
            "after_daily_cap": len(capped),
            "exact_baseline_overlap_count": len(exact_overlap),
            "new_additive_trades_count": len(offensive_rows),
            "displaced_trades_count": len({(t.strategy, t.entry_time.isoformat(), t.exit_time.isoformat(), t.side) for t in displaced_baseline}),
        },
        "delta_vs_baseline": delta,
        "variant_summary": variant_summary,
        "samples": {
            "exact_overlap": exact_overlap[:20],
            "new_additive": [asdict(t) for t in offensive_rows[:20]],
            "displaced_baseline": [asdict(t) for t in displaced_baseline[:20]],
        },
    }


def run_dataset(
    dataset: str,
    target_cells: list[str],
    report: dict[str, Any],
    size_scales: list[float] | None = None,
    filter_name: str | None = None,
) -> dict[str, Any]:
    if size_scales is None:
        size_scales = [1.0]
    cfg = _load_london_cfg()
    baseline_kept, baseline_meta, _classified, _dyn_idx, _blocked_cluster, _blocked_global = variant_k.build_variant_k_pre_coupling_kept(dataset)
    baseline_coupled = merged_engine._apply_shared_equity_coupling(
        sorted(baseline_kept, key=lambda t: (t.exit_time, t.entry_time)),
        STARTING_EQUITY,
        v14_max_units=baseline_meta["v14_max_units"],
    )
    baseline_eq = merged_engine._build_equity_curve(baseline_coupled, STARTING_EQUITY)
    baseline_summary = merged_engine._stats(baseline_coupled, STARTING_EQUITY, baseline_eq)

    return {
        "dataset": _dataset_key(dataset),
        "selected_cells": target_cells,
        "baseline_summary": baseline_summary,
        "variants": {
            f"{int(scale * 100)}pct": _build_variant_result(
                baseline_kept=baseline_kept,
                baseline_coupled=baseline_coupled,
                baseline_meta=baseline_meta,
                baseline_summary=baseline_summary,
                target_cells=target_cells,
                report=report,
                cfg=cfg,
                size_scale=scale,
                filter_name=filter_name,
            )
            for scale in size_scales
        },
    }


def _default_output_path(target_cells: list[str], size_scales: list[float], custom_cells: bool, filter_name: str | None) -> Path:
    if not custom_cells and size_scales == [1.0, 0.5, 0.25]:
        return SIZE_SWEEP_OUTPUT_PATH
    if not custom_cells and size_scales == [1.0]:
        return OUTPUT_PATH
    cell_part = "__".join(c.replace("/", "_") for c in target_cells)
    scale_part = "__".join(f"{int(s * 100)}pct" for s in size_scales)
    filter_part = f"_{filter_name}" if filter_name else ""
    return OUT_DIR / f"offensive_setupd_additive_{cell_part}_{scale_part}{filter_part}.json"


def main() -> int:
    args = _parse_args()
    datasets = [
        str(OUT_DIR / "USDJPY_M1_OANDA_500k.csv"),
        str(OUT_DIR / "USDJPY_M1_OANDA_1000k.csv"),
    ]
    reports = [setupd_outcomes.run_dataset(ds, TRADE_OUTCOME_ARGS) for ds in datasets]
    reports_by_path = {str(Path(ds).resolve()): report for ds, report in zip(datasets, reports)}
    target_cells = list(args.target_cells) if args.target_cells else _choose_target_cells(reports)
    size_scales = list(args.size_scales) if args.size_scales else [1.0, 0.5, 0.25]
    output_path = Path(args.output) if args.output else _default_output_path(target_cells, size_scales, bool(args.target_cells), args.filter_name)
    out = {
        "target_cells": target_cells,
        "selection_rule": (
            "manual target cells"
            if args.target_cells
            else "stable positive long-only native-allowed Setup D cells with avg_pnl_pips > 2.0 and count >= 15 on both datasets; top 2 by combined total pnl"
        ),
        "size_scales": size_scales,
        "filter_name": args.filter_name,
        "results": {
            r["dataset"]: r
            for r in [run_dataset(ds, target_cells, reports_by_path[str(Path(ds).resolve())], size_scales=size_scales, filter_name=args.filter_name) for ds in datasets]
        },
    }
    output_path.write_text(json.dumps(out, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(out, indent=2, default=_json_default))
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
