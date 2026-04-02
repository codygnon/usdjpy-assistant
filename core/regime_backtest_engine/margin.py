from __future__ import annotations

from .models import InstrumentSpec, PortfolioState, Position


class MarginModel:
    def __init__(self, instrument: InstrumentSpec) -> None:
        self.instrument = instrument

    def margin_required_for_units(self, units: int) -> float:
        return float(max(0, int(units)) * self.instrument.contract_size * self.instrument.margin_rate)

    def margin_used(self, open_positions: list[Position]) -> float:
        return float(sum(float(p.margin_held) for p in open_positions))

    def available_margin(self, *, equity: float, margin_used: float) -> float:
        return float(equity - margin_used)

    def can_open(self, *, equity: float, margin_used: float, units: int) -> bool:
        return self.margin_required_for_units(units) <= self.available_margin(equity=equity, margin_used=margin_used)

    def refresh_state(self, state: PortfolioState) -> None:
        state.margin_used = self.margin_used(state.open_positions)
        state.available_margin = self.available_margin(equity=state.equity, margin_used=state.margin_used)
