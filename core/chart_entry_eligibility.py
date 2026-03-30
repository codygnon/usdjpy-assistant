"""
Shadow entry-eligibility layer on top of chart authorization.

This layer does NOT decide whether a strategy would enter.
It only decides whether the authorized archetype has the data/reference state
needed to evaluate its own entry logic on the current bar.

Strategy ids here must stay aligned with ``core.ownership_table.STRATEGY_KEYS`` and
``core.chart_shadow_contracts.ARCHETYPE_IDS``. Session vocabulary for engines vs
Phase 3 is documented in ``core.chart_shadow_contracts``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class EntryDataFlags:
    has_m1_bar: bool = True
    has_m5_context: bool = False
    has_m15_context: bool = False
    has_h1_context: bool = False
    pivot_levels_available: bool = False
    asian_range_valid: bool = False
    lor_valid: bool = False


@dataclass(frozen=True)
class EntryEligibilityDecision:
    strategy: str
    can_evaluate_entry_logic: bool
    reason: str
    data_flags: EntryDataFlags

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["data_flags"] = asdict(self.data_flags)
        return out


def evaluate_shadow_entry_eligibility(
    *,
    strategy: str | None,
    flags: EntryDataFlags,
) -> EntryEligibilityDecision:
    if not strategy:
        return EntryEligibilityDecision(
            strategy="none",
            can_evaluate_entry_logic=False,
            reason="no_authorized_strategy",
            data_flags=flags,
        )

    if strategy == "v44_ny":
        ok = bool(flags.has_m1_bar and flags.has_m5_context and flags.has_h1_context)
        return EntryEligibilityDecision(
            strategy=strategy,
            can_evaluate_entry_logic=ok,
            reason="ok" if ok else "missing_m5_or_h1_context",
            data_flags=flags,
        )

    if strategy == "v14":
        ok = bool(
            flags.has_m1_bar
            and flags.has_m5_context
            and flags.has_m15_context
            and flags.pivot_levels_available
        )
        return EntryEligibilityDecision(
            strategy=strategy,
            can_evaluate_entry_logic=ok,
            reason="ok" if ok else "missing_pivots_or_m5_m15_context",
            data_flags=flags,
        )

    if strategy == "london_v2":
        ok = bool(flags.has_m1_bar and (flags.asian_range_valid or flags.lor_valid))
        return EntryEligibilityDecision(
            strategy=strategy,
            can_evaluate_entry_logic=ok,
            reason="ok" if ok else "missing_asian_or_lor_range_reference",
            data_flags=flags,
        )

    return EntryEligibilityDecision(
        strategy=strategy,
        can_evaluate_entry_logic=False,
        reason="unknown_strategy",
        data_flags=flags,
    )
