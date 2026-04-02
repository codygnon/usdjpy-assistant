from __future__ import annotations

from dataclasses import dataclass

from .models import ExitAction, PortfolioSnapshot, PositionSnapshot, Signal
from .strategy import BarView, HistoricalDataView, StrategyFamily


@dataclass(frozen=True)
class DummyStrategy(StrategyFamily):
    family_name: str
    every_n_bars: int = 10
    direction: str = "long"
    stop_offset_pips: float = 10.0
    target_offset_pips: float = 12.0
    size_units: int = 10_000

    def evaluate(self, current_bar: BarView, history: HistoricalDataView, portfolio: PortfolioSnapshot) -> Signal | None:
        if current_bar.bar_index == history.max_index == 0:
            return None
        if current_bar.bar_index % self.every_n_bars != 0:
            return None
        if any(pos.family == self.family_name for pos in portfolio.open_positions):
            return None
        if self.direction == "long":
            stop_loss = float(current_bar.bid_close) - self.stop_offset_pips * 0.01
            take_profit = float(current_bar.ask_close) + self.target_offset_pips * 0.01
        else:
            stop_loss = float(current_bar.ask_close) + self.stop_offset_pips * 0.01
            take_profit = float(current_bar.bid_close) - self.target_offset_pips * 0.01
        return Signal(
            family=self.family_name,
            direction=self.direction,
            stop_loss=stop_loss,
            take_profit=take_profit,
            size=self.size_units,
            metadata={"dummy": True},
        )

    def get_exit_conditions(self, position: PositionSnapshot, current_bar: BarView, history: HistoricalDataView) -> ExitAction | None:
        return None


@dataclass(frozen=True)
class CheatingStrategy(StrategyFamily):
    family_name: str = "cheater"

    def evaluate(self, current_bar: BarView, history: HistoricalDataView, portfolio: PortfolioSnapshot) -> Signal | None:
        _ = history[history.max_index + 1]
        return None

    def get_exit_conditions(self, position: PositionSnapshot, current_bar: BarView, history: HistoricalDataView) -> ExitAction | None:
        return None
