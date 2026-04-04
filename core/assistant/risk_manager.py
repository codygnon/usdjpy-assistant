from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TradeProposal:
    instrument: str
    direction: str
    entry_price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    lots: Optional[float]


@dataclass(frozen=True)
class RiskAssessment:
    recommended_lots: float
    recommended_units: int
    risk_amount: float
    risk_percent: float
    stop_loss: float
    stop_distance_pips: float
    take_profit_1: Optional[float]
    take_profit_2: Optional[float]
    tp1_distance_pips: Optional[float]
    tp2_distance_pips: Optional[float]
    is_valid: bool
    warnings: list[str]
    errors: list[str]
    account_equity: float
    pip_value_per_lot: float
    current_exposure_percent: float
    exposure_after_trade_percent: float


class RiskManager:
    """Computes position sizes and validates trades against risk limits."""

    def __init__(self, config):
        self._config = config

    def compute_pip_value(self, instrument: str, price: float, lot_size: float = 1.0) -> float:
        units = float(lot_size) * 100_000.0
        if "JPY" in instrument:
            return (0.01 / float(price)) * units
        return 0.0001 * units

    def compute_stop_distance_pips(self, instrument: str, entry: float, stop: float) -> float:
        distance = abs(float(entry) - float(stop))
        if "JPY" in instrument:
            return distance / 0.01
        return distance / 0.0001

    def compute_position_size(self, equity: float, entry_price: float, stop_loss: float, instrument: str = "USD_JPY") -> tuple[float, int]:
        stop_pips = self.compute_stop_distance_pips(instrument, entry_price, stop_loss)
        if stop_pips <= 0:
            return 0.0, 0

        risk_amount = float(equity) * float(self._config.max_risk_per_trade)
        pip_value_per_unit = (0.01 / float(entry_price)) if "JPY" in instrument else 0.0001
        raw_units = risk_amount / (stop_pips * pip_value_per_unit)
        cap_units = int(float(self._config.max_position_size_lots) * 100_000)
        units = min(int(math.floor(raw_units / 1000.0) * 1000), cap_units)
        if units < 0:
            units = 0
        return units / 100_000.0, units

    def compute_auto_stop(self, direction: str, entry_price: float, atr: float, instrument: str = "USD_JPY") -> float:
        stop_distance = float(self._config.default_stop_atr_multiple) * float(atr)
        pip_size = 0.01 if "JPY" in instrument else 0.0001
        max_distance = float(self._config.catastrophic_stop_pips) * pip_size
        stop_distance = min(stop_distance, max_distance)
        if direction == "long":
            return float(entry_price) - stop_distance
        return float(entry_price) + stop_distance

    def compute_tp_levels(self, direction: str, entry_price: float, stop_loss: float) -> tuple[Optional[float], Optional[float]]:
        stop_distance = abs(float(entry_price) - float(stop_loss))
        tp1_distance = stop_distance * float(self._config.tp1_ratio)
        tp2_distance = stop_distance * float(self._config.tp2_ratio)
        if direction == "long":
            return float(entry_price) + tp1_distance, float(entry_price) + tp2_distance
        return float(entry_price) - tp1_distance, float(entry_price) - tp2_distance

    def assess(self, proposal: TradeProposal, account, open_trades: list, current_price) -> RiskAssessment:
        warnings: list[str] = []
        errors: list[str] = []

        equity = float(account.equity)
        entry = float(proposal.entry_price or current_price.mid)
        pip_size = 0.01 if "JPY" in proposal.instrument else 0.0001

        if proposal.stop_loss is not None:
            stop = float(proposal.stop_loss)
        else:
            warnings.append("No stop loss specified — using catastrophic stop")
            if proposal.direction == "long":
                stop = entry - float(self._config.catastrophic_stop_pips) * pip_size
            else:
                stop = entry + float(self._config.catastrophic_stop_pips) * pip_size

        if proposal.direction == "long" and stop >= entry:
            errors.append(f"Stop loss {stop:.3f} must be BELOW entry {entry:.3f} for long")
        if proposal.direction == "short" and stop <= entry:
            errors.append(f"Stop loss {stop:.3f} must be ABOVE entry {entry:.3f} for short")

        stop_pips = self.compute_stop_distance_pips(proposal.instrument, entry, stop)
        if stop_pips > float(self._config.catastrophic_stop_pips):
            errors.append(
                f"Stop distance {stop_pips:.0f} pips exceeds catastrophic limit {self._config.catastrophic_stop_pips}"
            )

        if proposal.lots is not None:
            lots = float(proposal.lots)
            units = int(math.floor((lots * 100_000) / 1000.0) * 1000)
            lots = units / 100_000.0
        else:
            lots, units = self.compute_position_size(equity, entry, stop, proposal.instrument)

        pip_value_per_lot = self.compute_pip_value(proposal.instrument, entry, 1.0)
        risk_amount = stop_pips * pip_value_per_lot * lots
        risk_percent = (risk_amount / equity) if equity > 0 else 0.0

        if risk_percent > float(self._config.max_risk_per_trade) + 1e-12:
            errors.append(f"Risk {risk_percent:.1%} exceeds max {self._config.max_risk_per_trade:.1%}")
        if lots > float(self._config.max_position_size_lots):
            errors.append(f"Size {lots:.1f} lots exceeds max {self._config.max_position_size_lots} lots")

        current_exposure = 0.0
        if equity > 0:
            for trade in open_trades:
                trade_stop = getattr(trade, "stop_loss", None)
                if trade_stop is not None:
                    trade_stop_pips = self.compute_stop_distance_pips(trade.instrument, float(trade.open_price), float(trade_stop))
                else:
                    trade_stop_pips = float(self._config.catastrophic_stop_pips)
                trade_lots = abs(int(trade.units)) / 100_000.0
                trade_pip_value = self.compute_pip_value(trade.instrument, float(trade.open_price), trade_lots)
                current_exposure += (trade_stop_pips * trade_pip_value) / equity
        new_trade_exposure = risk_percent
        total_exposure = current_exposure + new_trade_exposure
        if total_exposure > float(self._config.max_total_exposure):
            warnings.append(f"Total exposure high: {total_exposure:.1%} of equity")

        tp1, tp2 = self.compute_tp_levels(proposal.direction, entry, stop)
        if proposal.take_profit is not None:
            tp1 = float(proposal.take_profit)
            tp2 = None

        tp1_pips = self.compute_stop_distance_pips(proposal.instrument, entry, tp1) if tp1 is not None else None
        tp2_pips = self.compute_stop_distance_pips(proposal.instrument, entry, tp2) if tp2 is not None else None

        if stop_pips < 10:
            warnings.append(f"Stop very tight at {stop_pips:.0f} pips — likely to be hit by noise")
        if stop_pips > 100:
            warnings.append(f"Stop wide at {stop_pips:.0f} pips — large risk per trade")

        for trade in open_trades:
            if trade.instrument != proposal.instrument:
                continue
            if trade.direction != proposal.direction:
                warnings.append(
                    f"OPPOSING POSITION: you have a {trade.direction} trade open ({abs(int(trade.units)):,} units). This would hedge, not add."
                )
            else:
                warnings.append(
                    f"ADDING to existing {trade.direction} position ({abs(int(trade.units)):,} units already open)"
                )

        return RiskAssessment(
            recommended_lots=lots,
            recommended_units=units,
            risk_amount=risk_amount,
            risk_percent=risk_percent,
            stop_loss=stop,
            stop_distance_pips=stop_pips,
            take_profit_1=tp1,
            take_profit_2=tp2,
            tp1_distance_pips=tp1_pips,
            tp2_distance_pips=tp2_pips,
            is_valid=(len(errors) == 0),
            warnings=warnings,
            errors=errors,
            account_equity=equity,
            pip_value_per_lot=pip_value_per_lot,
            current_exposure_percent=current_exposure,
            exposure_after_trade_percent=total_exposure,
        )
