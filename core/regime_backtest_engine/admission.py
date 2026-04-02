from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .margin import MarginModel
from .models import AdmissionConfig, ArbitrationRecord, PortfolioSnapshot, RejectedSignal, Signal


@dataclass(frozen=True)
class AdmissionDecision:
    accepted: list[Signal]
    rejected: list[RejectedSignal]
    record: ArbitrationRecord | None


class AdmissionFilter:
    def __init__(self, config: AdmissionConfig, margin_model: MarginModel) -> None:
        self.config = config
        self.margin_model = margin_model

    def _priority(self, family: str) -> tuple[int, str]:
        if family in self.config.family_priority:
            return (self.config.family_priority.index(family), family)
        return (len(self.config.family_priority), family)

    def decide(self, *, bar_index: int, timestamp: Any, signals: list[Signal], portfolio: PortfolioSnapshot, reference_price: float) -> AdmissionDecision:
        if not signals:
            return AdmissionDecision([], [], None)

        ordered = sorted(signals, key=lambda s: self._priority(s.family))
        accepted: list[Signal] = []
        rejected: list[RejectedSignal] = []
        current_positions = list(portfolio.open_positions)
        accepted_units_total = 0
        accepted_by_family: dict[str, int] = {}

        for signal in ordered:
            reason: str | None = None
            all_positions = current_positions + [
                type("AcceptedPosition", (), {"family": s.family, "direction": s.direction, "size": s.size})()
                for s in accepted
            ]
            open_count_total = len(current_positions) + len(accepted)
            open_count_family = sum(1 for p in all_positions if p.family == signal.family)
            units_total = sum(int(getattr(p, "size", 0)) for p in all_positions)
            units_family = sum(int(getattr(p, "size", 0)) for p in all_positions if p.family == signal.family)

            if not self.config.allow_opposing_exposure:
                opposite = "short" if signal.direction == "long" else "long"
                if any(getattr(p, "direction", None) == opposite for p in all_positions):
                    reason = "opposing_exposure_block"
            if reason is None and open_count_total >= self.config.max_total_open_positions:
                reason = "max_total_open_positions"
            family_cap = self.config.max_open_positions_per_family.get(signal.family)
            if reason is None and family_cap is not None and open_count_family >= family_cap:
                reason = "max_open_positions_per_family"
            if reason is None and units_total + signal.size > self.config.max_total_units:
                reason = "max_total_units"
            family_units_cap = self.config.max_units_per_family.get(signal.family)
            if reason is None and family_units_cap is not None and units_family + signal.size > family_units_cap:
                reason = "max_units_per_family"
            if reason is None:
                equity = portfolio.equity
                simulated_margin = portfolio.margin_used + sum(self.margin_model.margin_required_for_units(s.size) for s in accepted)
                if not self.margin_model.can_open(equity=equity, margin_used=simulated_margin, units=signal.size):
                    reason = "margin_insufficient"

            if reason is None:
                accepted.append(signal)
                accepted_units_total += signal.size
                accepted_by_family[signal.family] = accepted_by_family.get(signal.family, 0) + signal.size
            else:
                rejected.append(RejectedSignal(family=signal.family, direction=signal.direction, reason=reason))

        record = None
        if len(signals) > 1 or rejected:
            outcome = "all_accepted"
            if rejected and accepted:
                if any(r.reason == "margin_insufficient" for r in rejected):
                    outcome = "margin_ranked"
                else:
                    outcome = "conflict_rejected"
            elif rejected and not accepted:
                outcome = "conflict_rejected"
            elif len(signals) == 1 and accepted:
                outcome = "single_accepted"
            record = ArbitrationRecord(
                bar_index=int(bar_index),
                timestamp=timestamp,
                candidate_families=tuple(s.family for s in signals),
                candidate_directions=tuple(s.direction for s in signals),
                outcome=outcome,
                accepted_families=tuple(s.family for s in accepted),
                rejected_families=tuple(r.family for r in rejected),
                rejection_reasons=tuple(r.reason for r in rejected),
            )

        return AdmissionDecision(accepted=accepted, rejected=rejected, record=record)
