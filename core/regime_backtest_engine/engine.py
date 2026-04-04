from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import pandas as pd

from .admission import AdmissionFilter
from .data import load_market_data
from .manifest import evaluate_summary_against_manifest, freeze_manifest
from .margin import MarginModel
from .models import (
    BacktestResult,
    ClosedTrade,
    PendingOrderRejection,
    PortfolioState,
    Position,
    PositionSnapshot,
    RunConfig,
    Signal,
    closed_trade_to_row,
    portfolio_snapshot_from_state,
)
from .strategy import BarView, HistoricalDataView, StrategyAdapter, StrategyFamily


class BacktestEngine:
    def __init__(self, strategies: dict[str, StrategyFamily]) -> None:
        self.strategies = {
            family_name: StrategyAdapter(family_name=family_name, strategy=strategy)
            for family_name, strategy in strategies.items()
        }

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

        for idx in range(len(loaded.store)):
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

        state.pending_signals = []
        self._mark_to_market(state, BarView(loaded.store, len(loaded.store) - 1), margin_model)

        trade_columns = [field.name for field in fields(ClosedTrade)]
        trade_df = pd.DataFrame([closed_trade_to_row(t) for t in state.closed_trades], columns=trade_columns)
        bar_df = pd.DataFrame(bar_rows)
        arbitration_df = pd.DataFrame(arbitration_rows)
        summary = self._build_summary(config, loaded.contract.synthetic_bid_ask, loaded.frame, trade_df, bar_df, arbitration_df, state)
        if config.manifest is not None:
            summary["manifest_evaluation"] = evaluate_summary_against_manifest(summary, config.manifest)

        config_snapshot_path = output_dir / "run_config.json"
        config_snapshot_path.write_text(config.model_dump_json(indent=2), encoding="utf-8")

        trade_log_path = output_dir / "trade_log.csv"
        trade_df.to_csv(trade_log_path, index=False)

        bar_log_path = output_dir / f"bar_state_log.{ 'parquet' if config.bar_log_format == 'parquet' else 'csv' }"
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

    def _fill_pending_orders(
        self,
        state: PortfolioState,
        current_bar: BarView,
        margin_model: MarginModel,
        config: RunConfig,
    ) -> list[PendingOrderRejection]:
        rejections: list[PendingOrderRejection] = []
        if not state.pending_signals:
            return rejections

        pending = list(state.pending_signals)
        state.pending_signals = []
        for signal in pending:
            entry_price = self._entry_fill_price(signal, current_bar, config)
            stop_loss = self._resolve_clamped_raw_stop(
                signal=signal,
                entry_price=entry_price,
                direction=signal.direction,
                pip_size=config.instrument.pip_size,
            )
            if stop_loss is None:
                stop_loss = self._resolve_fill_relative_price(
                    signal=signal,
                    entry_price=entry_price,
                    field_name="stop_loss_pips",
                    fallback=float(signal.stop_loss),
                    direction=signal.direction,
                    pip_size=config.instrument.pip_size,
                )
            take_profit = self._resolve_fill_relative_price(
                signal=signal,
                entry_price=entry_price,
                field_name="take_profit_pips",
                fallback=float(signal.take_profit) if signal.take_profit is not None else None,
                direction=signal.direction,
                pip_size=config.instrument.pip_size,
            )
            margin_required = margin_model.margin_required_for_units(signal.size)
            if not margin_model.can_open(equity=state.equity, margin_used=state.margin_used, units=signal.size):
                rejections.append(PendingOrderRejection(signal=signal, reason="fill_margin_insufficient"))
                continue
            state.trade_id_seq += 1
            state.open_positions.append(
                Position(
                    trade_id=state.trade_id_seq,
                    family=signal.family,
                    direction=signal.direction,
                    entry_price=float(entry_price),
                    entry_bar=int(current_bar.bar_index),
                    entry_time=current_bar.timestamp,
                    size=int(signal.size),
                    initial_size=int(signal.size),
                    margin_held=float(margin_required),
                    stop_loss=float(stop_loss),
                    take_profit=float(take_profit) if take_profit is not None else None,
                )
            )
            self.strategies[signal.family].on_position_opened(self._position_snapshot(state.open_positions[-1]), signal, current_bar)
            margin_model.refresh_state(state)
        return rejections

    def _resolve_clamped_raw_stop(
        self,
        *,
        signal: Signal,
        entry_price: float,
        direction: str,
        pip_size: float,
    ) -> float | None:
        raw_stop = signal.metadata.get("raw_stop_price")
        if raw_stop is None:
            return None
        raw_stop = float(raw_stop)
        min_pips = float(signal.metadata.get("sl_min_pips", 0.0) or 0.0)
        max_pips_raw = signal.metadata.get("sl_max_pips")
        max_pips = float(max_pips_raw) if max_pips_raw is not None else None
        if direction == "long":
            stop_pips = (float(entry_price) - raw_stop) / pip_size
            if stop_pips <= 0:
                return None
            if max_pips is not None:
                stop_pips = min(stop_pips, max_pips)
            stop_pips = max(stop_pips, min_pips)
            return float(entry_price) - stop_pips * pip_size
        stop_pips = (raw_stop - float(entry_price)) / pip_size
        if stop_pips <= 0:
            return None
        if max_pips is not None:
            stop_pips = min(stop_pips, max_pips)
        stop_pips = max(stop_pips, min_pips)
        return float(entry_price) + stop_pips * pip_size

    def _resolve_fill_relative_price(
        self,
        *,
        signal: Signal,
        entry_price: float,
        field_name: str,
        fallback: float | None,
        direction: str,
        pip_size: float,
    ) -> float | None:
        raw_pips = signal.metadata.get(field_name)
        if raw_pips is None:
            return fallback
        pips = float(raw_pips)
        if direction == "long":
            return float(entry_price - pips * pip_size) if field_name == "stop_loss_pips" else float(entry_price + pips * pip_size)
        return float(entry_price + pips * pip_size) if field_name == "stop_loss_pips" else float(entry_price - pips * pip_size)

    def _entry_fill_price(self, signal: Signal, current_bar: BarView, config: RunConfig) -> float:
        slippage = float(config.slippage.fixed_slippage_pips) * config.instrument.pip_size
        if signal.direction == "long":
            return float(current_bar.ask_open) + slippage
        return float(current_bar.bid_open) - slippage

    def _price_diff_to_usd(self, entry_price: float, exit_price: float, units: int, direction: str) -> float:
        if direction == "long":
            jpy_pnl = (float(exit_price) - float(entry_price)) * int(units)
        else:
            jpy_pnl = (float(entry_price) - float(exit_price)) * int(units)
        return float(jpy_pnl / max(float(exit_price), 1e-9))

    def _mark_to_market(self, state: PortfolioState, current_bar: BarView, margin_model: MarginModel) -> None:
        unrealized = 0.0
        for pos in state.open_positions:
            mark = float(current_bar.bid_close) if pos.direction == "long" else float(current_bar.ask_close)
            pnl = self._price_diff_to_usd(pos.entry_price, mark, pos.size, pos.direction)
            pos.unrealized_pnl = float(pnl)
            unrealized += float(pnl)
        state.unrealized_pnl = float(unrealized)
        state.equity = float(state.balance + state.unrealized_pnl)
        margin_model.refresh_state(state)

    def _check_exits(
        self,
        state: PortfolioState,
        current_bar: BarView,
        margin_model: MarginModel,
        config: RunConfig,
    ) -> None:
        still_open: list[Position] = []
        for pos in state.open_positions:
            custom_exit = self.strategies[pos.family].get_exit_conditions(
                self._position_snapshot(pos),
                current_bar,
                HistoricalDataView(current_bar._store, current_bar.bar_index),
            )
            if custom_exit is not None:
                updated_stop = custom_exit.new_stop_loss if custom_exit.new_stop_loss is not None else custom_exit.stop_loss
                updated_take = custom_exit.new_take_profit if custom_exit.new_take_profit is not None else custom_exit.take_profit
                if updated_stop is not None:
                    pos.stop_loss = float(updated_stop)
                if updated_take is not None:
                    pos.take_profit = float(updated_take)

                exit_type = custom_exit.exit_type
                if exit_type == "none" and custom_exit.close_full:
                    exit_type = "full"

                if exit_type == "partial":
                    closed_fully = self._close_position(
                        state,
                        current_bar,
                        pos,
                        float(custom_exit.price if custom_exit.price is not None else (current_bar.bid_close if pos.direction == "long" else current_bar.ask_close)),
                        custom_exit.reason,
                        config,
                        event_type="partial",
                        close_fraction=float(custom_exit.close_fraction),
                    )
                    if closed_fully:
                        continue
                elif exit_type == "full":
                    self._close_position(
                        state,
                        current_bar,
                        pos,
                        float(custom_exit.price if custom_exit.price is not None else (current_bar.bid_close if pos.direction == "long" else current_bar.ask_close)),
                        custom_exit.reason,
                        config,
                        event_type="full",
                        close_fraction=1.0,
                    )
                    continue

            stop_hit, target_hit = self._exit_hits(pos, current_bar)
            if stop_hit and target_hit:
                self._close_position(state, current_bar, pos, float(pos.stop_loss), "worst_case_stop", config, event_type="full", close_fraction=1.0)
                continue
            if stop_hit:
                self._close_position(state, current_bar, pos, float(pos.stop_loss), "stop_loss", config, event_type="full", close_fraction=1.0)
                continue
            if target_hit:
                self._close_position(state, current_bar, pos, float(pos.take_profit), "take_profit", config, event_type="full", close_fraction=1.0)
                continue
            still_open.append(pos)

        state.open_positions = still_open
        self._mark_to_market(state, current_bar, margin_model)

    def _close_position(
        self,
        state: PortfolioState,
        current_bar: BarView,
        pos: Position,
        exit_price: float,
        exit_reason: str,
        config: RunConfig,
        *,
        event_type: str,
        close_fraction: float,
    ) -> bool:
        requested_units = pos.size if close_fraction >= 1.0 else int(pos.size * close_fraction)
        if requested_units <= 0:
            requested_units = 1 if pos.size > 1 else pos.size
        closed_units = min(pos.size, requested_units)
        remaining_units = int(pos.size - closed_units)
        if remaining_units <= 0:
            event_type = "full"
            close_fraction = 1.0

        pnl_usd = self._price_diff_to_usd(pos.entry_price, exit_price, closed_units, pos.direction)
        pnl_pips = ((float(exit_price) - float(pos.entry_price)) / config.instrument.pip_size) if pos.direction == "long" else ((float(pos.entry_price) - float(exit_price)) / config.instrument.pip_size)
        spread_cost = (float(current_bar.spread_pips) * config.instrument.pip_size * closed_units) / max(float(exit_price), 1e-9)
        slippage_cost = (float(config.slippage.fixed_slippage_pips) * config.instrument.pip_size * closed_units) / max(float(exit_price), 1e-9)
        margin_released = float(pos.margin_held) * (float(closed_units) / max(float(pos.size), 1.0))
        state.balance += float(pnl_usd)
        closed_trade = ClosedTrade(
                trade_id=pos.trade_id,
                event_type="partial" if event_type == "partial" and remaining_units > 0 else "full",
                close_fraction=float(closed_units / max(pos.initial_size, 1)),
                closed_units=int(closed_units),
                remaining_units=int(max(remaining_units, 0)),
                family=pos.family,
                direction=pos.direction,
                entry_time=pos.entry_time,
                exit_time=current_bar.timestamp,
                entry_bar=pos.entry_bar,
                exit_bar=int(current_bar.bar_index),
                entry_price=float(pos.entry_price),
                exit_price=float(exit_price),
                size=int(closed_units),
                margin_held=float(margin_released),
                stop_loss=float(pos.stop_loss),
                take_profit=float(pos.take_profit) if pos.take_profit is not None else None,
                spread_cost=float(spread_cost),
                slippage_cost=float(slippage_cost),
                pnl_usd=float(pnl_usd),
                pnl_pips=float(pnl_pips),
                bars_held=int(current_bar.bar_index - pos.entry_bar),
                exit_reason=exit_reason,
        )
        state.closed_trades.append(closed_trade)
        self.strategies[pos.family].on_position_closed(closed_trade)
        if remaining_units > 0:
            pos.size = int(remaining_units)
            pos.margin_held = float(max(0.0, pos.margin_held - margin_released))
            return False
        return True

    def _position_snapshot(self, pos: Position) -> PositionSnapshot:
        return PositionSnapshot(
            trade_id=pos.trade_id,
            family=pos.family,
            direction=pos.direction,
            entry_price=float(pos.entry_price),
            entry_bar=int(pos.entry_bar),
            size=int(pos.size),
            margin_held=float(pos.margin_held),
            stop_loss=float(pos.stop_loss),
            take_profit=float(pos.take_profit) if pos.take_profit is not None else None,
            unrealized_pnl=float(pos.unrealized_pnl),
        )

    def _exit_hits(self, pos: Position, current_bar: BarView) -> tuple[bool, bool]:
        if pos.direction == "long":
            stop_hit = float(current_bar.bid_low) <= float(pos.stop_loss)
            target_hit = pos.take_profit is not None and float(current_bar.bid_high) >= float(pos.take_profit)
        else:
            stop_hit = float(current_bar.ask_high) >= float(pos.stop_loss)
            target_hit = pos.take_profit is not None and float(current_bar.ask_low) <= float(pos.take_profit)
        return bool(stop_hit), bool(target_hit)

    def _per_family_counts(self, state: PortfolioState, active_families: tuple[str, ...]) -> dict[str, int]:
        counts = {f"open_positions_{family}": 0 for family in active_families}
        for pos in state.open_positions:
            key = f"open_positions_{pos.family}"
            if key in counts:
                counts[key] += 1
        return counts

    def _build_summary(
        self,
        config: RunConfig,
        synthetic_bid_ask: bool,
        frame: pd.DataFrame,
        trades: pd.DataFrame,
        bars: pd.DataFrame,
        arbitration: pd.DataFrame,
        state: PortfolioState,
    ) -> dict[str, Any]:
        trade_level = trades.groupby(["trade_id", "family"], as_index=False).agg({"pnl_usd": "sum"}) if not trades.empty else pd.DataFrame(columns=["trade_id", "family", "pnl_usd"])
        net_pnl = float(trade_level["pnl_usd"].sum()) if not trade_level.empty else 0.0
        gross_win = float(trade_level.loc[trade_level["pnl_usd"] > 0, "pnl_usd"].sum()) if not trade_level.empty else 0.0
        gross_loss = abs(float(trade_level.loc[trade_level["pnl_usd"] < 0, "pnl_usd"].sum())) if not trade_level.empty else 0.0
        profit_factor = float(gross_win / gross_loss) if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
        wins = int((trade_level["pnl_usd"] > 0).sum()) if not trade_level.empty else 0
        total = int(len(trade_level))
        peak = float(config.initial_balance)
        max_dd = 0.0
        for eq in bars["equity"].tolist() if not bars.empty else [config.initial_balance]:
            peak = max(peak, float(eq))
            max_dd = max(max_dd, peak - float(eq))
        max_dd_pct = (max_dd / peak * 100.0) if peak > 0 else 0.0
        by_family = []
        if not trade_level.empty:
            for family, group in trade_level.groupby("family"):
                by_family.append(
                    {
                        "family": str(family),
                        "trade_count": int(len(group)),
                        "net_pnl_usd": float(group["pnl_usd"].sum()),
                    }
                )
        return {
            "hypothesis": config.hypothesis,
            "mode": config.mode,
            "active_families": list(config.active_families),
            "processed_bar_count": int(len(frame)),
            "processed_start_time": pd.Timestamp(frame["timestamp"].iloc[0]).isoformat(),
            "processed_end_time": pd.Timestamp(frame["timestamp"].iloc[-1]).isoformat(),
            "synthetic_bid_ask": bool(synthetic_bid_ask),
            "initial_balance": float(config.initial_balance),
            "ending_balance": float(state.balance),
            "ending_equity": float(state.equity),
            "net_pnl_usd": net_pnl,
            "profit_factor": profit_factor,
            "trade_count": total,
            "win_rate": float(wins / total * 100.0) if total else None,
            "max_drawdown_usd": float(max_dd),
            "max_drawdown_pct": float(max_dd_pct),
            "max_concurrent_positions": int(bars["open_position_count"].max()) if not bars.empty else 0,
            "arbitration_event_count": int(len(arbitration)),
            "by_family": by_family,
        }
