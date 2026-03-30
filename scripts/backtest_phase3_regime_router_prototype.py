#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path("/Users/codygnon/Documents/usdjpy_assistant")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import backtest_merged_integrated_tokyo_london_v2_ny as merged_engine

OUT_DIR = ROOT / "research_out"


@dataclass
class Candidate:
    trade: merged_engine.TradeRow
    strategy_name: str
    candidate_side: str
    regime_fit_score: float
    entry_quality_score: float
    blocking_reason: str | None = None


def _safe_float(x: Any, default: float = 0.0) -> float:
    return merged_engine._safe_float(x, default)


def _load_component_trades(
    *,
    dataset: str,
    v14_config: Path,
    v2_config: Path,
    v44_config: Path,
    starting_equity: float,
) -> tuple[list[merged_engine.TradeRow], dict[str, Any]]:
    m1 = merged_engine._load_m1(dataset)
    v14_report, v14_cfg = merged_engine._run_v14_in_process(v14_config, dataset)
    v2_trades_df, v2_diag, _ = merged_engine._run_v2_in_process(v2_config, m1)
    v44_results, v44_embedded = merged_engine._run_v44_in_process(v44_config, dataset)
    v44_base_eq = _safe_float(v44_embedded.get("v5", {}).get("account_size", starting_equity), starting_equity)
    trades = (
        merged_engine._extract_v14_trades(v14_report, default_entry_equity=float(starting_equity))
        + merged_engine._extract_v2_trades(v2_trades_df, default_entry_equity=float(starting_equity))
        + merged_engine._extract_v44_trades(v44_results, default_entry_equity=float(v44_base_eq))
    )
    meta = {
        "m1": m1,
        "v14_cfg": v14_cfg,
        "v2_diag": v2_diag,
        "v44_embedded": v44_embedded,
    }
    return trades, meta


def _setup_d_breakout(raw: dict[str, Any]) -> bool:
    values = " ".join(str(raw.get(k, "")) for k in ("setup", "setup_type", "pattern", "entry_type", "label")).lower()
    return ("setup d" in values) or values.endswith("d") or "breakout" in values


def _candidate_from_trade(trade: merged_engine.TradeRow) -> Candidate:
    raw = trade.raw or {}
    if trade.strategy == "v44_ny":
        signal_mode = str(raw.get("entry_signal_mode", ""))
        profile = str(raw.get("entry_profile", ""))
        if signal_mode == "news_trend_confirm":
            return Candidate(trade, "v44_news_trend", trade.side, 100.0, 100.0, None)
        if profile == "Strong":
            return Candidate(trade, "v44", trade.side, 85.0, 88.0, None)
        if profile == "Normal":
            return Candidate(trade, "v44", trade.side, 72.0, 74.0, None)
        return Candidate(trade, "v44", trade.side, 50.0, 52.0, None)
    if trade.strategy == "london_v2":
        if _setup_d_breakout(raw):
            return Candidate(trade, "london_v2_breakout", trade.side, 82.0, 85.0, None)
        return Candidate(trade, "london_v2", trade.side, 65.0, 68.0, None)
    combo = str(raw.get("combo", raw.get("entry_combo", "")))
    confluence = int(_safe_float(raw.get("confluence", raw.get("confluence_score", 0)), 0.0))
    quality = 45.0 + min(4, confluence) * 8.0 + (4.0 if "D" in combo else 0.0)
    return Candidate(trade, "v14", trade.side, 35.0, quality, None)


def _session_ok(candidate: Candidate, variant: str) -> bool:
    if variant == "A":
        return True
    t = candidate.trade
    return (
        (t.strategy == "v14" and t.entry_session == "tokyo")
        or (t.strategy == "london_v2" and t.entry_session == "london")
        or (t.strategy == "v44_ny" and t.entry_session == "ny")
    )


def _pick_winner(candidates: list[Candidate]) -> Candidate | None:
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    news = [c for c in candidates if c.strategy_name == "v44_news_trend"]
    if news:
        return news[0]
    london_breakouts = [c for c in candidates if c.strategy_name == "london_v2_breakout"]
    if london_breakouts:
        non_v14 = [c for c in london_breakouts if c.trade.strategy != "v14"]
        if non_v14:
            return sorted(non_v14, key=lambda c: (c.regime_fit_score + c.entry_quality_score), reverse=True)[0]
    v44 = [c for c in candidates if c.trade.strategy == "v44_ny"]
    v14 = [c for c in candidates if c.trade.strategy == "v14"]
    if v44 and v14:
        best_v44 = max(v44, key=lambda c: c.regime_fit_score + c.entry_quality_score)
        best_v14 = max(v14, key=lambda c: c.regime_fit_score + c.entry_quality_score)
        if best_v44.regime_fit_score >= 72.0 and best_v44.entry_quality_score >= best_v14.entry_quality_score:
            return best_v44
        if best_v14.regime_fit_score <= 40.0 and best_v14.entry_quality_score >= 65.0:
            return best_v14
        return None
    ranked = sorted(candidates, key=lambda c: (c.regime_fit_score + c.entry_quality_score, c.entry_quality_score), reverse=True)
    if len(ranked) >= 2 and abs((ranked[0].regime_fit_score + ranked[0].entry_quality_score) - (ranked[1].regime_fit_score + ranked[1].entry_quality_score)) < 3.0:
        return None
    return ranked[0]


def _run_router_variant(trades: list[merged_engine.TradeRow], variant: str, starting_equity: float, v14_max_units: int) -> dict[str, Any]:
    buckets: dict[pd.Timestamp, list[Candidate]] = defaultdict(list)
    for trade in trades:
        candidate = _candidate_from_trade(trade)
        if not _session_ok(candidate, variant):
            continue
        buckets[pd.Timestamp(trade.entry_time).floor("min")].append(candidate)

    selected_raw: list[merged_engine.TradeRow] = []
    arbitration_log: list[dict[str, Any]] = []
    for bar_time in sorted(buckets):
        group = buckets[bar_time]
        winner = _pick_winner(group)
        arbitration_log.append(
            {
                "bar_time": bar_time.isoformat(),
                "candidate_count": len(group),
                "winner": winner.trade.strategy if winner is not None else None,
                "candidates": [
                    {
                        "strategy": c.trade.strategy,
                        "strategy_name": c.strategy_name,
                        "side": c.candidate_side,
                        "regime_fit_score": c.regime_fit_score,
                        "entry_quality_score": c.entry_quality_score,
                    }
                    for c in group
                ],
            }
        )
        if winner is not None:
            selected_raw.append(merged_engine.TradeRow(**{**winner.trade.__dict__}))

    selected_scaled = merged_engine._apply_shared_equity_coupling(selected_raw, starting_equity, v14_max_units=v14_max_units)
    equity_curve = merged_engine._build_equity_curve(selected_scaled, starting_equity)
    stats = merged_engine._stats(selected_scaled, starting_equity, equity_curve)
    by_strategy = merged_engine._subset_breakdown(selected_scaled, lambda t: t.strategy)
    by_session = merged_engine._subset_breakdown(selected_scaled, lambda t: t.entry_session)
    by_month = merged_engine._group_monthly(selected_scaled)
    daily_counts: dict[str, int] = defaultdict(int)
    for trade in selected_scaled:
        daily_counts[pd.Timestamp(trade.entry_time).strftime("%Y-%m-%d")] += 1
    return {
        "summary": stats,
        "by_strategy": by_strategy,
        "by_session": by_session,
        "by_month": by_month,
        "closed_trades": [
            {
                "strategy": t.strategy,
                "entry_time": t.entry_time.isoformat(),
                "exit_time": t.exit_time.isoformat(),
                "entry_session": t.entry_session,
                "side": t.side,
                "pips": float(t.pips),
                "usd": float(t.usd),
                "exit_reason": t.exit_reason,
                "size_scale": float(t.size_scale),
            }
            for t in selected_scaled
        ],
        "arbitration_log": arbitration_log,
        "days_with_trades": len(daily_counts),
    }


def _sample_day_deltas(baseline_rows: list[dict[str, Any]], variant_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    def _group(rows: list[dict[str, Any]]) -> dict[str, float]:
        out: dict[str, float] = defaultdict(float)
        for row in rows:
            day = pd.Timestamp(row["entry_time"]).strftime("%Y-%m-%d")
            out[day] += float(row.get("usd", 0.0))
        return out

    b = _group(baseline_rows)
    v = _group(variant_rows)
    rows = []
    for day in sorted(set(b) | set(v)):
        rows.append(
            {
                "day": day,
                "baseline_usd": round(float(b.get(day, 0.0)), 2),
                "variant_usd": round(float(v.get(day, 0.0)), 2),
                "delta_usd": round(float(v.get(day, 0.0) - b.get(day, 0.0)), 2),
            }
        )
    improved = sorted(rows, key=lambda r: r["delta_usd"], reverse=True)[:10]
    worsened = sorted(rows, key=lambda r: r["delta_usd"])[:10]
    return improved, worsened


def main() -> int:
    p = argparse.ArgumentParser(description="Phase 3 regime-led router prototype backtest")
    p.add_argument("--dataset", default=str(OUT_DIR / "USDJPY_M1_OANDA_1000k.csv"))
    p.add_argument("--v14-config", default=str(OUT_DIR / "tokyo_optimized_v14_config.json"))
    p.add_argument("--london-v2-config", default=str(OUT_DIR / "v2_exp4_winner_baseline_config.json"))
    p.add_argument("--v44-config", default=str(OUT_DIR / "session_momentum_v44_base_config.json"))
    p.add_argument("--baseline-report", default=str(OUT_DIR / "phase3_regime_led_router_baseline_report.json"))
    p.add_argument("--output-prefix", default=str(OUT_DIR / "phase3_regime_router"))
    p.add_argument("--starting-equity", type=float, default=100000.0)
    args = p.parse_args()

    dataset = str(Path(args.dataset))
    trades, meta = _load_component_trades(
        dataset=dataset,
        v14_config=Path(args.v14_config),
        v2_config=Path(args.london_v2_config),
        v44_config=Path(args.v44_config),
        starting_equity=float(args.starting_equity),
    )
    v14_max_units = merged_engine._safe_int(meta["v14_cfg"].get("position_sizing", {}).get("max_units", 500000), 500000)

    baseline_output = Path(args.baseline_report)
    baseline_output.parent.mkdir(parents=True, exist_ok=True)
    baseline_cmd_result = merged_engine.main  # make linter happy about imported module use
    _ = baseline_cmd_result
    # Recreate baseline through shared-equity coupling on all raw component trades.
    baseline_scaled = merged_engine._apply_shared_equity_coupling(
        sorted(trades, key=lambda t: (t.exit_time, t.entry_time)),
        float(args.starting_equity),
        v14_max_units=v14_max_units,
    )
    baseline_equity = merged_engine._build_equity_curve(baseline_scaled, float(args.starting_equity))
    baseline_summary = merged_engine._stats(baseline_scaled, float(args.starting_equity), baseline_equity)
    baseline_closed = [
        {
            "strategy": t.strategy,
            "entry_time": t.entry_time.isoformat(),
            "exit_time": t.exit_time.isoformat(),
            "entry_session": t.entry_session,
            "side": t.side,
            "pips": float(t.pips),
            "usd": float(t.usd),
            "exit_reason": t.exit_reason,
            "size_scale": float(t.size_scale),
        }
        for t in baseline_scaled
    ]
    baseline_report = {
        "dataset": dataset,
        "starting_equity": float(args.starting_equity),
        "summary": baseline_summary,
        "closed_trades": baseline_closed,
        "notes": {
            "type": "integrated_baseline_for_regime_router_prototype",
            "limitation": "Uses existing component trade candidates; does not generate new cross-session signals.",
        },
    }
    baseline_output.write_text(json.dumps(baseline_report, indent=2), encoding="utf-8")

    output_index = {
        "dataset": dataset,
        "starting_equity": float(args.starting_equity),
        "baseline_report": str(baseline_output),
        "baseline_summary": baseline_summary,
        "variants": {},
    }

    for variant in ("A", "B"):
        result = _run_router_variant(trades, variant, float(args.starting_equity), v14_max_units)
        improved_days, worsened_days = _sample_day_deltas(baseline_closed, result["closed_trades"])
        report = {
            "variant": variant,
            "dataset": dataset,
            "baseline_summary": baseline_summary,
            "variant_summary": result["summary"],
            "delta_vs_baseline": {
                "trades": int(result["summary"]["total_trades"]) - int(baseline_summary["total_trades"]),
                "net_usd": round(float(result["summary"]["net_usd"]) - float(baseline_summary["net_usd"]), 2),
                "profit_factor": round(float(result["summary"]["profit_factor"]) - float(baseline_summary["profit_factor"]), 4),
                "max_drawdown_usd": round(float(result["summary"]["max_drawdown_usd"]) - float(baseline_summary["max_drawdown_usd"]), 2),
                "win_rate_pct": round(float(result["summary"]["win_rate_pct"]) - float(baseline_summary["win_rate_pct"]), 3),
            },
            "by_strategy_family": result["by_strategy"],
            "by_session": result["by_session"],
            "by_month": result["by_month"],
            "large_move_no_trade_days_reduced": 0,
            "sample_improved_days": improved_days,
            "sample_worsened_days": worsened_days,
            "closed_trades": result["closed_trades"],
            "arbitration_log": result["arbitration_log"],
            "notes": {
                "variant_A": "winner-take-bar with no additional session restriction",
                "variant_B": "winner-take-bar while preserving original session ownership",
                "current_limitation": "Because candidate generation still comes from the standalone strategies, A and B may be similar until a true cross-session eligibility engine is built.",
            },
        }
        out_path = Path(f"{args.output_prefix}_{variant}.json")
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        output_index["variants"][variant] = {
            "report_path": str(out_path),
            "summary": report["variant_summary"],
            "delta_vs_baseline": report["delta_vs_baseline"],
        }

    index_path = Path(f"{args.output_prefix}_index.json")
    index_path.write_text(json.dumps(output_index, indent=2), encoding="utf-8")
    print(json.dumps(output_index, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
