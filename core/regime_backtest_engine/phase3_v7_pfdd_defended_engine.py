"""Phase3 v7_pfdd defended variant: RunConfig-driven backtest via shared bar-by-bar runner.

The canonical simulation lives in :mod:`core.phase3_v7_pfdd_defended_runner`. This engine
maps runner output into the standard trade log / bar log / summary shape used by
:class:`BacktestEngine`.
"""

from __future__ import annotations

import json
import re
from dataclasses import fields
from pathlib import Path
from typing import Any

import pandas as pd

from core.phase3_v7_pfdd_defended_runner import (
    Phase3V7PfddParams,
    execute_phase3_v7_pfdd_defended,
)

from .engine import BacktestEngine
from .models import (
    BacktestResult,
    ClosedTrade,
    ExitAction,
    PortfolioSnapshot,
    PortfolioState,
    Signal,
    closed_trade_to_row,
    portfolio_snapshot_from_state,
)
from .strategy import BarView, HistoricalDataView
from .manifest import evaluate_summary_against_manifest, freeze_manifest


PHASE3_V7_PFDD_FAMILY = "phase3_v7_pfdd_defended"


def _parse_pnl_usd(row: dict[str, Any]) -> float | None:
    v = row.get("pnl_usd")
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _infer_size(row: dict[str, Any]) -> int:
    u = row.get("units")
    if u not in (None, ""):
        return max(1, int(u))
    lots = row.get("lots")
    if lots not in (None, ""):
        return max(1, int(float(lots) * 1000))
    return 1


def _direction(row: dict[str, Any]) -> str:
    side = str(row.get("side", "buy")).lower()
    return "long" if side == "buy" else "short"


def _stop_loss_price(row: dict[str, Any], entry: float, direction: str) -> float:
    sp = row.get("sl_pips")
    if sp not in (None, ""):
        d = float(sp) * 0.01
        return entry - d if direction == "long" else entry + d
    raw = row.get("sl_price")
    if raw not in (None, ""):
        return float(raw)
    return entry - 0.05 if direction == "long" else entry + 0.05


def phase3_rows_to_closed_trades(rows: list[dict[str, Any]], *, family: str) -> list[ClosedTrade]:
    out: list[ClosedTrade] = []
    for row in rows:
        pnl = _parse_pnl_usd(row)
        if pnl is None:
            continue
        tid = int(row.get("trade_id", 0) or 0)
        entry_bar = int(row.get("entry_bar_index", 0) or 0)
        exit_bar = int(row.get("exit_bar_index", entry_bar) or entry_bar)
        ep = float(row.get("entry_price", 0.0) or 0.0)
        xp = float(row.get("exit_price", ep) or ep)
        direction = _direction(row)
        size = _infer_size(row)
        sl = _stop_loss_price(row, ep, direction)
        tp_raw = row.get("tp2_price", row.get("tp1_price"))
        tp = float(tp_raw) if tp_raw not in (None, "") else None
        pnl_pips = float(row.get("pnl_pips", 0.0) or 0.0)
        reason = str(row.get("exit_reason", "phase3"))
        out.append(
            ClosedTrade(
                trade_id=tid,
                family=family,
                direction=direction,  # type: ignore[arg-type]
                entry_time=row.get("entry_time", ""),
                exit_time=row.get("exit_time", ""),
                entry_bar=entry_bar,
                exit_bar=exit_bar,
                entry_price=ep,
                exit_price=xp,
                size=size,
                margin_held=0.0,
                stop_loss=sl,
                take_profit=tp,
                spread_cost=0.0,
                slippage_cost=0.0,
                pnl_usd=float(pnl),
                pnl_pips=pnl_pips,
                bars_held=max(0, exit_bar - entry_bar),
                exit_reason=reason,
                event_type="full",
                close_fraction=1.0,
                closed_units=size,
                remaining_units=0,
            )
        )
    return out


class V7PfddDefendedStrategy:
    """Placeholder strategy registered for RunConfig compatibility.

    The defended Phase3 path is executed by :class:`Phase3V7PfddDefendedBacktestEngine`,
    which calls the shared runner instead of stepping bar-by-bar through generic fills.
    A standard :class:`BacktestEngine` run with only this strategy would produce no trades.
    """

    family_name = PHASE3_V7_PFDD_FAMILY

    def evaluate_signals(
        self,
        current_bar: BarView,
        history: HistoricalDataView,
        portfolio: PortfolioSnapshot,
    ) -> list[Signal]:
        return []

    def evaluate(self, current_bar: BarView, history: HistoricalDataView, portfolio: PortfolioSnapshot) -> Signal | None:
        return None

    def get_exit_conditions(self, position, current_bar: BarView, history: HistoricalDataView) -> ExitAction | None:
        return None


class Phase3V7PfddDefendedBacktestEngine(BacktestEngine):
    """Runs the Phase3 defended bar-by-bar simulation and writes engine-shaped artifacts."""

    def __init__(self) -> None:
        super().__init__({PHASE3_V7_PFDD_FAMILY: V7PfddDefendedStrategy()})

    def run(self, config: Any, *, runner_quiet: bool = False) -> BacktestResult:
        """runner_quiet: if True, suppress Phase3 runner stdout (use in fast/pytest runs)."""
        if config.mode != "standalone":
            raise ValueError("Phase3V7PfddDefendedBacktestEngine requires mode='standalone'")
        if config.active_families != (PHASE3_V7_PFDD_FAMILY,):
            raise ValueError(
                f"expected active_families=('{PHASE3_V7_PFDD_FAMILY}',), got {config.active_families!r}"
            )

        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = None
        if config.manifest is not None:
            manifest_path = output_dir / "run_manifest.json"
            freeze_manifest(config.manifest, manifest_path)

        spread_mode = "realistic" if "realistic" in str(config.hypothesis) else "pipeline"
        max_bars = None
        m = re.search(r"_maxbars_(\d+)", str(config.hypothesis))
        if m:
            max_bars = int(m.group(1))

        raw = execute_phase3_v7_pfdd_defended(
            Phase3V7PfddParams(
                data_path=str(config.data_path),
                spread_mode=spread_mode,
                quiet=runner_quiet,
                max_bars=max_bars,
            )
        )

        closed = phase3_rows_to_closed_trades(raw["trades_log"], family=PHASE3_V7_PFDD_FAMILY)
        trade_columns = [f.name for f in fields(ClosedTrade)]
        trade_df = pd.DataFrame([closed_trade_to_row(t) for t in closed], columns=trade_columns)

        eq = raw["equity_rows"]
        bar_df = pd.DataFrame(eq) if eq else pd.DataFrame()
        if not bar_df.empty:
            bar_df["balance"] = bar_df["equity"]
            bar_df["margin_used"] = 0.0
            bar_df["available_margin"] = bar_df["equity"]
            bar_df["pending_order_count"] = 0
            bar_df["accepted_signal_count"] = 0
            bar_df["rejected_signal_count"] = 0
            bar_df[f"open_positions_{PHASE3_V7_PFDD_FAMILY}"] = bar_df["open_position_count"]

        ending = float(raw["summary"]["ending_equity"])
        state = PortfolioState(
            balance=ending,
            equity=ending,
            unrealized_pnl=0.0,
            margin_used=0.0,
            available_margin=ending,
            open_positions=[],
            closed_trades=list(closed),
            pending_signals=[],
            trade_id_seq=max((t.trade_id for t in closed), default=0),
        )

        frame = pd.DataFrame({"timestamp": pd.to_datetime(bar_df["timestamp"], utc=True)}) if not bar_df.empty else pd.DataFrame(columns=["timestamp"])
        arbitration_df = pd.DataFrame()
        summary = self._build_summary(
            config,
            True,
            frame,
            trade_df,
            bar_df,
            arbitration_df,
            state,
        )
        summary["processed_bar_count"] = int(raw["total_bars_processed"])
        summary["phase3_defended_runner_summary"] = raw["summary"]
        if config.manifest is not None:
            summary["manifest_evaluation"] = evaluate_summary_against_manifest(summary, config.manifest)

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

        raw_summary_path = output_dir / "phase3_defended_summary.json"
        raw_summary_path.write_text(json.dumps(raw["summary"], indent=2, default=str), encoding="utf-8")

        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

        return BacktestResult(
            summary=summary,
            trade_log_path=trade_log_path,
            bar_log_path=bar_log_path,
            config_snapshot_path=config_snapshot_path,
            manifest_path=manifest_path,
            arbitration_log_path=None,
            final_portfolio=portfolio_snapshot_from_state(state),
        )
