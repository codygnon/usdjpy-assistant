#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import tempfile
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

OUT_DIR = ROOT / "research_out"
V14_CFG_PATH = OUT_DIR / "tokyo_optimized_v14_config.json"
LONDON_CFG_PATH = OUT_DIR / "v2_exp4_winner_baseline_config.json"
V44_CFG_PATH = OUT_DIR / "session_momentum_v44_base_config.json"


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
    raise ValueError(f"Unsupported dataset naming for aligned baseline lookup: {name}")


def _avg_duration_minutes(trades: list[merged_engine.TradeRow]) -> float:
    if not trades:
        return 0.0
    vals = []
    for t in trades:
        mins = (pd.Timestamp(t.exit_time) - pd.Timestamp(t.entry_time)).total_seconds() / 60.0
        vals.append(float(mins))
    return float(np.mean(vals)) if vals else 0.0


def _exit_reason_distribution(trades: list[merged_engine.TradeRow]) -> dict[str, int]:
    c = Counter(t.exit_reason for t in trades)
    return {k: int(v) for k, v in sorted(c.items())}


def _blocked_stats(trades: list[merged_engine.TradeRow]) -> dict[str, Any]:
    wins = [t for t in trades if t.usd > 0]
    losses = [t for t in trades if t.usd <= 0]
    return {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "net_pips": float(sum(t.pips for t in trades)),
        "net_usd": float(sum(t.usd for t in trades)),
    }


def _stats_with_duration(trades: list[merged_engine.TradeRow], start_equity: float) -> dict[str, Any]:
    eq = merged_engine._build_equity_curve(trades, start_equity)
    stats = merged_engine._stats(trades, start_equity, eq)
    stats["avg_duration_minutes"] = _avg_duration_minutes(trades)
    return stats


def _top_regime(scores: dict[str, float]) -> str:
    if not scores:
        return "unknown"
    return max(scores, key=scores.get)


def _build_ny_v14_config(dataset: str, output_prefix: Path) -> Path:
    cfg = json.loads(V14_CFG_PATH.read_text(encoding="utf-8"))
    cfg["session_filter"]["session_start_utc"] = "13:00"
    cfg["session_filter"]["session_end_utc"] = "16:00"
    cfg["session_filter"]["allowed_trading_days"] = [
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
    ]
    cfg["trade_management"]["disable_entries_if_move_from_tokyo_open_range_exceeds_pips"] = 0.0
    cfg["trade_management"]["breakout_detection_mode"] = "disabled"
    run0 = cfg.get("run_sequence", [{}])[0]
    run0["label"] = "ny_regime_gated"
    run0["input_csv"] = dataset
    run0["output_json"] = str(output_prefix.with_name(output_prefix.name + "_ny_v14_tmp_report.json"))
    run0["output_trades_csv"] = str(output_prefix.with_name(output_prefix.name + "_ny_v14_tmp_trades.csv"))
    run0["output_equity_csv"] = str(output_prefix.with_name(output_prefix.name + "_ny_v14_tmp_equity.csv"))
    cfg["run_sequence"] = [run0]

    td = tempfile.mkdtemp(prefix="v14_ny_regime_")
    path = Path(td) / "v14_ny_config.json"
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return path


def _extract_v14_with_label(report: dict[str, Any], default_entry_equity: float, strategy_name: str, session_name: str) -> list[merged_engine.TradeRow]:
    trades = merged_engine._extract_v14_trades(report, default_entry_equity)
    out = []
    for t in trades:
        out.append(
            merged_engine.TradeRow(
                strategy=strategy_name,
                entry_time=t.entry_time,
                exit_time=t.exit_time,
                entry_session=session_name,
                side=t.side,
                pips=t.pips,
                usd=t.usd,
                exit_reason=t.exit_reason,
                standalone_entry_equity=t.standalone_entry_equity,
                raw=t.raw,
                size_scale=t.size_scale,
            )
        )
    return out


def _apply_shared_equity_coupling_v14aware(
    trades: list[merged_engine.TradeRow],
    starting_equity: float,
    v14_max_units: int,
) -> list[merged_engine.TradeRow]:
    if not trades:
        return []
    sim = [merged_engine.TradeRow(**{**t.__dict__}) for t in trades]
    by_idx = {i: t for i, t in enumerate(sim)}
    events: list[tuple[pd.Timestamp, int, int]] = []
    for i, t in by_idx.items():
        events.append((pd.Timestamp(t.entry_time), 1, i))
        events.append((pd.Timestamp(t.exit_time), 0, i))
    events.sort(key=lambda x: (x[0], x[1]))

    equity = float(starting_equity)
    entry_scale: dict[int, float] = {}
    for _, evt_type, i in events:
        t = by_idx[i]
        if evt_type == 1:
            base_eq = float(t.standalone_entry_equity) if float(t.standalone_entry_equity) > 0 else float(starting_equity)
            scale = float(equity / base_eq) if base_eq > 0 else 1.0
            if t.strategy in {"v14", "v14_tokyo", "v14_ny"}:
                raw_units = merged_engine._safe_float(t.raw.get("position_size_units", t.raw.get("position_units", 0.0)))
                if raw_units >= float(v14_max_units) - 1 and scale > 1.0:
                    scale = 1.0
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0
            entry_scale[i] = float(scale)
            t.size_scale = float(scale)
        else:
            sc = float(entry_scale.get(i, 1.0))
            t.usd = float(t.usd * sc)
            equity += float(t.usd)
    return sim


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


def _by_strategy_dict(trades: list[merged_engine.TradeRow], start_equity: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for strategy in ["v14_tokyo", "v14_ny", "london_v2", "v44_ny"]:
        subset = [t for t in trades if t.strategy == strategy]
        out[strategy] = _stats_with_duration(subset, start_equity)
    return out


def _load_variant_f_baseline_summary(dataset_key: str) -> dict[str, Any]:
    comp_path = OUT_DIR / f"v44_conservative_router_aligned_{dataset_key}_comparison.json"
    data = json.loads(comp_path.read_text(encoding="utf-8"))
    return data["variants"]["F_block_plus_ambiguous_non_momentum"]["summary"]


def run_one(dataset: str, output_prefix: Path) -> dict[str, Any]:
    starting_equity = 100000.0
    dataset_key = _dataset_key(dataset)

    classified = v44_router._load_classified_bars(dataset)
    m5 = v44_router._build_m5(dataset)

    ny_cfg_path = _build_ny_v14_config(dataset, output_prefix)
    ny_report, ny_cfg = merged_engine._run_v14_in_process(ny_cfg_path, dataset)
    ny_raw_trades = _extract_v14_with_label(ny_report, starting_equity, "v14_ny", "ny")

    kept: list[merged_engine.TradeRow] = []
    blocked: list[merged_engine.TradeRow] = []
    kept_samples: list[dict[str, Any]] = []

    for trade in ny_raw_trades:
        regime = v44_router._lookup_regime(classified, trade.entry_time)
        top = _top_regime(regime["regime_scores"])
        if top == "mean_reversion":
            kept.append(trade)
            if len(kept_samples) < 30:
                dur = (pd.Timestamp(trade.exit_time) - pd.Timestamp(trade.entry_time)).total_seconds() / 60.0
                kept_samples.append(
                    {
                        "entry_time": trade.entry_time.isoformat(),
                        "exit_time": trade.exit_time.isoformat(),
                        "side": trade.side,
                        "pips": float(trade.pips),
                        "usd": float(trade.usd),
                        "exit_reason": trade.exit_reason,
                        "regime_scores": {k: float(v) for k, v in regime["regime_scores"].items()},
                        "duration_min": float(dur),
                    }
                )
        else:
            blocked.append(trade)

    kept_stats = _stats_with_duration(kept, starting_equity)
    blocked_stats = _blocked_stats(blocked)

    # Rebuild raw baseline legs for accurate coupling.
    m1 = merged_engine._load_m1(dataset)
    tokyo_report, tokyo_cfg = merged_engine._run_v14_in_process(V14_CFG_PATH, dataset)
    tokyo_raw = _extract_v14_with_label(tokyo_report, starting_equity, "v14_tokyo", "tokyo")
    london_df, _, _ = merged_engine._run_v2_in_process(LONDON_CFG_PATH, m1)
    london_raw = merged_engine._extract_v2_trades(london_df, default_entry_equity=starting_equity)
    v44_f_raw = _build_variant_f_v44_raw(dataset, classified, m5, starting_equity)

    combined_raw = sorted(tokyo_raw + kept + london_raw + v44_f_raw, key=lambda t: (t.exit_time, t.entry_time))
    v14_max_units = merged_engine._safe_int(tokyo_cfg.get("position_sizing", {}).get("max_units", 500000), 500000)
    combined = _apply_shared_equity_coupling_v14aware(combined_raw, starting_equity, v14_max_units)
    combined_eq = merged_engine._build_equity_curve(combined, starting_equity)
    combined_summary = merged_engine._stats(combined, starting_equity, combined_eq)

    variant_f_baseline = _load_variant_f_baseline_summary(dataset_key)
    delta_vs_f = {
        "net_usd": round(combined_summary["net_usd"] - variant_f_baseline["net_usd"], 2),
        "pf": round(combined_summary["profit_factor"] - variant_f_baseline["profit_factor"], 4),
        "max_dd_usd": round(combined_summary["max_drawdown_usd"] - variant_f_baseline["max_drawdown_usd"], 2),
        "total_trades": int(combined_summary["total_trades"] - variant_f_baseline["total_trades"]),
    }

    result = {
        "dataset": Path(dataset).name,
        "build_note": "NY V14 is run from a temporary config override. Portfolio coupling is rebuilt from raw trades so standalone_entry_equity is preserved.",
        "ny_v14_standalone": {
            "raw_trades": len(ny_raw_trades),
            "regime_gated_kept": len(kept),
            "regime_gated_blocked": len(blocked),
            "kept_stats": kept_stats,
            "blocked_stats": blocked_stats,
            "exit_reason_distribution": _exit_reason_distribution(kept),
            "kept_trades_sample": kept_samples,
        },
        "combined_portfolio": {
            "summary": combined_summary,
            "by_strategy": _by_strategy_dict(combined, starting_equity),
            "delta_vs_variant_f_baseline": delta_vs_f,
        },
        "variant_f_baseline": variant_f_baseline,
    }
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest V14 in NY, gated by mean_reversion top-score regime")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output-prefix", required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    dataset = str(Path(args.dataset).resolve())
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    result = run_one(dataset, output_prefix)

    out_path = output_prefix.with_name(output_prefix.name + "_results.json")
    out_path.write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")

    ny = result["ny_v14_standalone"]
    kept = ny["kept_stats"]
    blocked = ny["blocked_stats"]
    comb = result["combined_portfolio"]["summary"]
    base = result["variant_f_baseline"]
    delta = result["combined_portfolio"]["delta_vs_variant_f_baseline"]

    print("=" * 100)
    print(f"NY V14 REGIME-GATED: {Path(dataset).name}")
    print("=" * 100)
    print(
        f"{'Standalone NY V14':<26s} raw={ny['raw_trades']:>4d} kept={ny['regime_gated_kept']:>4d} "
        f"blocked={ny['regime_gated_blocked']:>4d} WR={kept['win_rate_pct']:.1f}% "
        f"PF={kept['profit_factor']:.3f} net=${kept['net_usd']:.2f} DD=${kept['max_drawdown_usd']:.2f}"
    )
    print(
        f"{'Blocked NY V14':<26s} trades={blocked['trades']:>4d} wins={blocked['wins']:>4d} "
        f"losses={blocked['losses']:>4d} net=${blocked['net_usd']:.2f} pips={blocked['net_pips']:.1f}"
    )
    print("-" * 100)
    print(
        f"{'Variant F baseline':<26s} trades={base['total_trades']:>4d} "
        f"net=${base['net_usd']:.2f} PF={base['profit_factor']:.3f} DD=${base['max_drawdown_usd']:.2f}"
    )
    print(
        f"{'Combined + NY V14':<26s} trades={comb['total_trades']:>4d} "
        f"net=${comb['net_usd']:.2f} PF={comb['profit_factor']:.3f} DD=${comb['max_drawdown_usd']:.2f}"
    )
    print(
        f"{'Delta vs Variant F':<26s} trades={delta['total_trades']:+4d} "
        f"net={delta['net_usd']:+.2f} PF={delta['pf']:+.4f} DD={delta['max_dd_usd']:+.2f}"
    )
    print(f"Output: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
