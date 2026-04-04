#!/usr/bin/env python3
"""Run Daily Trend + 4H pullback backtest — real spread (2.0 pip + 0.1 slippage)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from dataclasses import fields

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.regime_backtest_engine.admission import AdmissionFilter
from core.regime_backtest_engine.daily_trend_strategy import DailyTrendStrategy
from core.regime_backtest_engine.data import load_market_data
from core.regime_backtest_engine.engine import BacktestEngine
from core.regime_backtest_engine.manifest import evaluate_summary_against_manifest, freeze_manifest
from core.regime_backtest_engine.margin import MarginModel
from core.regime_backtest_engine.models import (
    AdmissionConfig,
    BacktestResult,
    ClosedTrade,
    FixedSpreadConfig,
    InstrumentSpec,
    PortfolioState,
    RunConfig,
    Signal,
    SlippageConfig,
    SpreadConfig,
    closed_trade_to_row,
    portfolio_snapshot_from_state,
)
from core.regime_backtest_engine.strategy import BarView, HistoricalDataView


class ProgressBacktestEngine(BacktestEngine):
    """BacktestEngine that prints progress every 100k bars."""

    def run(self, config: RunConfig) -> BacktestResult:
        missing = [name for name in config.active_families if name not in self.strategies]
        if missing:
            raise ValueError(f"missing strategies for active families: {missing}")

        loaded = load_market_data(config)
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = None
        if config.manifest is not None:
            manifest_path = output_dir / "run_manifest.json"
            freeze_manifest(config.manifest, manifest_path)

        margin_model = MarginModel(config.instrument)
        admission = AdmissionFilter(config.admission, margin_model)
        state = PortfolioState(
            balance=float(config.initial_balance),
            equity=float(config.initial_balance),
            unrealized_pnl=0.0,
            margin_used=0.0,
            available_margin=float(config.initial_balance),
        )

        bar_rows: list[dict[str, Any]] = []
        arbitration_rows: list[dict[str, Any]] = []
        n = len(loaded.store)
        t0 = time.perf_counter()

        for idx in range(n):
            if idx > 0 and idx % 100_000 == 0:
                elapsed = time.perf_counter() - t0
                print(
                    f"Progress: bar_index={idx} / {n} ({elapsed:.1f}s elapsed)",
                    flush=True,
                )

            current_bar = BarView(loaded.store, idx)
            fill_rejections = self._fill_pending_orders(state, current_bar, margin_model, config)
            for rejection in fill_rejections:
                arbitration_rows.append(
                    {
                        "bar_index": idx,
                        "timestamp": current_bar.timestamp,
                        "candidate_families": [rejection.signal.family],
                        "candidate_directions": [rejection.signal.direction],
                        "outcome": "fill_rejected",
                        "accepted_families": [],
                        "rejected_families": [rejection.signal.family],
                        "rejection_reasons": [rejection.reason],
                    }
                )

            self._mark_to_market(state, current_bar, margin_model)
            self._check_exits(state, current_bar, margin_model, config)
            self._mark_to_market(state, current_bar, margin_model)

            snapshot = portfolio_snapshot_from_state(state)
            history = HistoricalDataView(loaded.store, idx)
            signals: list[Signal] = []
            for family in config.active_families:
                strat = self.strategies[family]
                signals.extend(strat.evaluate_signals(current_bar, history, snapshot))

            decision = admission.decide(
                bar_index=idx,
                timestamp=current_bar.timestamp,
                signals=signals,
                portfolio=snapshot,
                reference_price=float(current_bar.mid_close),
            )
            state.pending_signals = list(decision.accepted)
            if decision.record is not None:
                arbitration_rows.append(
                    {
                        "bar_index": decision.record.bar_index,
                        "timestamp": decision.record.timestamp,
                        "candidate_families": list(decision.record.candidate_families),
                        "candidate_directions": list(decision.record.candidate_directions),
                        "outcome": decision.record.outcome,
                        "accepted_families": list(decision.record.accepted_families),
                        "rejected_families": list(decision.record.rejected_families),
                        "rejection_reasons": list(decision.record.rejection_reasons),
                    }
                )

            bar_rows.append(
                {
                    "timestamp": current_bar.timestamp,
                    "bar_index": idx,
                    "balance": state.balance,
                    "unrealized_pnl": state.unrealized_pnl,
                    "equity": state.equity,
                    "margin_used": state.margin_used,
                    "available_margin": state.available_margin,
                    "open_position_count": len(state.open_positions),
                    "pending_order_count": len(state.pending_signals),
                    "accepted_signal_count": len(decision.accepted),
                    "rejected_signal_count": len(decision.rejected) + len(fill_rejections),
                    **self._per_family_counts(state, config.active_families),
                }
            )

        elapsed = time.perf_counter() - t0
        print(f"Backtest complete: {n} bars in {elapsed:.1f}s", flush=True)

        state.pending_signals = []
        self._mark_to_market(state, BarView(loaded.store, len(loaded.store) - 1), margin_model)

        trade_columns = [field.name for field in fields(ClosedTrade)]
        trade_df = pd.DataFrame(
            [closed_trade_to_row(t) for t in state.closed_trades], columns=trade_columns
        )
        bar_df = pd.DataFrame(bar_rows)
        arbitration_df = pd.DataFrame(arbitration_rows)
        summary = self._build_summary(
            config, loaded.contract.synthetic_bid_ask, loaded.frame, trade_df, bar_df, arbitration_df, state
        )
        if config.manifest is not None:
            summary["manifest_evaluation"] = evaluate_summary_against_manifest(
                summary, config.manifest
            )

        config_snapshot_path = output_dir / "run_config.json"
        config_snapshot_path.write_text(config.model_dump_json(indent=2), encoding="utf-8")

        trade_log_path = output_dir / "trade_log.csv"
        trade_df.to_csv(trade_log_path, index=False)

        bar_log_path = output_dir / (
            f"bar_state_log.{'parquet' if config.bar_log_format == 'parquet' else 'csv'}"
        )
        if config.bar_log_format == "parquet":
            try:
                bar_df.to_parquet(bar_log_path, index=False)
            except Exception as exc:
                raise RuntimeError("bar_log_format='parquet' requires pyarrow or fastparquet") from exc
        else:
            bar_df.to_csv(bar_log_path, index=False)

        arbitration_log_path = None
        if not arbitration_df.empty:
            arbitration_log_path = output_dir / "arbitration_log.csv"
            arbitration_df.to_csv(arbitration_log_path, index=False)

        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

        return BacktestResult(
            summary=summary,
            trade_log_path=trade_log_path,
            bar_log_path=bar_log_path,
            config_snapshot_path=config_snapshot_path,
            manifest_path=manifest_path,
            arbitration_log_path=arbitration_log_path,
            final_portfolio=portfolio_snapshot_from_state(state),
        )


def _extend_summary(summary: dict, trade_df: pd.DataFrame, bar_df: pd.DataFrame) -> dict:
    out = dict(summary)

    if trade_df.empty:
        out["trade_count"] = 0
        out["win_rate"] = 0.0
        out["profit_factor"] = 0.0
        out["net_pnl"] = 0.0
        out["avg_bars_held"] = None
        out["avg_initial_stop_pips"] = None
        return out

    pnl_col = "pnl_usd" if "pnl_usd" in trade_df.columns else "pnl"
    if pnl_col not in trade_df.columns:
        for c in trade_df.columns:
            if "pnl" in c.lower():
                pnl_col = c
                break

    trip_pnl = (
        trade_df.groupby("trade_id")[pnl_col].sum()
        if "trade_id" in trade_df.columns
        else trade_df[pnl_col]
    )

    out["trade_count"] = len(trip_pnl)
    out["win_rate"] = (
        100.0 * (trip_pnl > 0).sum() / len(trip_pnl) if len(trip_pnl) > 0 else 0.0
    )

    gross_wins = trip_pnl[trip_pnl > 0].sum()
    gross_losses = abs(trip_pnl[trip_pnl < 0].sum())
    out["profit_factor"] = float(gross_wins / gross_losses) if gross_losses > 0 else 999.0
    out["net_pnl"] = float(trip_pnl.sum())
    out["gross_wins"] = float(gross_wins)
    out["gross_losses"] = float(gross_losses)
    out["avg_win"] = float(trip_pnl[trip_pnl > 0].mean()) if (trip_pnl > 0).any() else 0.0
    out["avg_loss"] = float(trip_pnl[trip_pnl < 0].mean()) if (trip_pnl < 0).any() else 0.0

    if "exit_reason" in trade_df.columns and "trade_id" in trade_df.columns:
        terminal = trade_df.groupby("trade_id").last()
        if "exit_reason" in terminal.columns:
            out["exit_reasons"] = terminal["exit_reason"].value_counts().to_dict()

    if "bars_held" in trade_df.columns and "trade_id" in trade_df.columns:
        out["avg_bars_held"] = float(trade_df.groupby("trade_id")["bars_held"].last().mean())
    else:
        out["avg_bars_held"] = None

    if (
        "entry_price" in trade_df.columns
        and "stop_loss" in trade_df.columns
        and "trade_id" in trade_df.columns
    ):
        first = trade_df.groupby("trade_id").first()
        dist = (first["entry_price"] - first["stop_loss"]).abs() * 100.0
        out["avg_initial_stop_pips"] = float(dist.mean())
    else:
        out["avg_initial_stop_pips"] = None

    return out


def _experiment13_gates(ext: dict) -> dict[str, Any]:
    tc = int(ext.get("trade_count") or 0)
    pf = float(ext.get("profit_factor") or 0.0)
    dd = float(ext.get("max_drawdown_pct") or 0.0)
    pf_ok = pf >= 1.30
    tc_ok = tc >= 80
    dd_ok = dd <= 20.0
    return {
        "minimum_profit_factor": 1.30,
        "minimum_trade_count": 80,
        "maximum_drawdown_pct_gate": 20.0,
        "profit_factor_pass": pf_ok,
        "trade_count_pass": tc_ok,
        "drawdown_pass": dd_ok,
        "experiment13_all_pass": pf_ok and tc_ok and dd_ok,
    }


def main() -> None:
    uj_path = ROOT / "research_out/USDJPY_M1_OANDA_1000k.csv"
    if not uj_path.is_file():
        print(f"ERROR: missing USDJPY data file: {uj_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = ROOT / "research_out/daily_trend_real"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run_log.txt"
    log_path.write_text("", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log("=== DAILY TREND + 4H PULLBACK — REAL SPREAD ===")
    log(f"Started at {time.strftime('%Y-%m-%d %H:%M:%S')} UTC")

    strategy = DailyTrendStrategy(
        daily_ema_fast=20,
        daily_ema_slow=50,
        daily_atr_period=14,
        proximity_atr=0.3,
        stop_atr_factor=1.5,
        trail_buffer_atr=0.5,
        pullback_lookback=5,
        trail_lookback=5,
        risk_fraction=0.01,
        account_balance=100_000.0,
        max_size=500_000,
    )

    family = strategy.family_name
    cfg = RunConfig(
        hypothesis="daily_trend_real",
        data_path=uj_path,
        output_dir=output_dir,
        mode="standalone",
        active_families=(family,),
        instrument=InstrumentSpec(symbol="USDJPY", margin_rate=(1.0 / 33.3)),
        spread=SpreadConfig(spread_source="fixed", fixed=FixedSpreadConfig(spread_pips=2.0)),
        slippage=SlippageConfig(fixed_slippage_pips=0.1),
        admission=AdmissionConfig(
            allow_opposing_exposure=False,
            max_total_open_positions=1,
            max_open_positions_per_family={family: 1},
            max_total_units=500_000,
            max_units_per_family={family: 500_000},
            family_priority=(family,),
        ),
        initial_balance=100_000.0,
        bar_log_format="csv",
    )

    log("Starting backtest...")
    engine = ProgressBacktestEngine({family: strategy})
    t0 = time.perf_counter()
    result = engine.run(cfg)
    elapsed = time.perf_counter() - t0
    log(f"Finished in {elapsed:.1f}s")

    trade_df = (
        pd.read_csv(result.trade_log_path) if result.trade_log_path.is_file() else pd.DataFrame()
    )
    bar_df = pd.read_csv(result.bar_log_path) if result.bar_log_path.is_file() else pd.DataFrame()
    ext = _extend_summary(dict(result.summary), trade_df, bar_df)
    ext["experiment13_gates"] = _experiment13_gates(ext)
    ext["methodology_notes"] = (
        "Daily trend (EMA50 22:00 UTC) + 4H pullback to daily EMA20. "
        "Trailing on daily bars. Fixed 2.0 pip spread + 0.1 pip slippage."
    )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(ext, indent=2, default=str), encoding="utf-8")
    log(f"\nSummary written to {summary_path}")
    log(json.dumps(ext, indent=2, default=str))
    log(f"\nTrade log: {result.trade_log_path}")


if __name__ == "__main__":
    main()
