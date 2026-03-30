"""
Chart-first authorization scaffold for the hybrid/offensive roadmap.

This module does not place trades. It answers one question per bar:

    Given the chart-state cell and current session, which archetype is
    chart-eligible, validated for this context, feasible with current data,
    and therefore authorized right now?

Current policy is intentionally conservative:
  - ownership comes from the stable conservative ownership table
  - validation reflects what is actually proven today
  - no stable owner -> no owner
  - stable no-trade -> no owner

This is the closest production-safe representation of the future chart-owned
decision loop using today's evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from core.chart_shadow_contracts import (
    NO_OWNER_NO_STABLE_OWNER,
    NO_OWNER_NOT_AUTHORIZED,
    NO_OWNER_NOT_FEASIBLE,
    NO_OWNER_NOT_VALIDATED,
    NO_OWNER_POLICY_MISSING,
    NO_OWNER_STABLE_NO_TRADE,
)
from core.ownership_table import STRATEGY_KEYS, load_conservative_table

ARCHETYPES: tuple[str, ...] = STRATEGY_KEYS


@dataclass(frozen=True)
class ArchetypePolicy:
    strategy: str
    validated_sessions: tuple[str, ...]
    feasible_sessions: tuple[str, ...] | None
    notes: str

    def is_validated(self, session: str | None) -> bool:
        return bool(session) and session in set(self.validated_sessions)

    def is_feasible(self, session: str | None) -> bool:
        if self.feasible_sessions is None:
            return True
        return bool(session) and session in set(self.feasible_sessions)


@dataclass(frozen=True)
class AuthorizationCandidate:
    strategy: str
    chart_eligible: bool
    validated_for_context: bool
    feasible_with_current_data: bool
    eligible_to_authorize: bool
    note: str = ""


@dataclass(frozen=True)
class AuthorizationDecision:
    time: str
    session: str | None
    ownership_cell: str
    ownership_type: str
    recommended_strategy: str
    candidates: tuple[AuthorizationCandidate, ...]
    authorized_strategy: str | None
    no_owner_reason: str
    mode: str = "strict_current_validation"

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["candidates"] = [asdict(c) for c in self.candidates]
        return out


def default_archetype_policies() -> dict[str, ArchetypePolicy]:
    return {
        "v44_ny": ArchetypePolicy(
            strategy="v44_ny",
            validated_sessions=("ny",),
            feasible_sessions=None,
            notes="Momentum is computable all day, but only validated in NY so far.",
        ),
        "v14": ArchetypePolicy(
            strategy="v14",
            validated_sessions=("tokyo",),
            feasible_sessions=None,
            notes="Current V14 remains session-native; cross-session portability is not validated.",
        ),
        "london_v2": ArchetypePolicy(
            strategy="london_v2",
            validated_sessions=("london",),
            feasible_sessions=("london",),
            notes="Current range-breakout implementation is London-bound and not portable yet.",
        ),
    }


def load_authorization_context(
    *,
    research_out: Path | None = None,
    policies: dict[str, ArchetypePolicy] | None = None,
) -> dict[str, Any]:
    return {
        "table": load_conservative_table(research_out=research_out),
        "policies": policies or default_archetype_policies(),
    }


def authorize_bar(
    *,
    ts_iso: str,
    session: str | None,
    ownership_cell: str,
    context: dict[str, Any],
) -> AuthorizationDecision:
    table: dict[str, dict[str, Any]] = context["table"]
    policies: dict[str, ArchetypePolicy] = context["policies"]

    cell_info = table.get(ownership_cell)
    if cell_info is None:
        candidates = tuple(
            AuthorizationCandidate(
                strategy=s,
                chart_eligible=False,
                validated_for_context=policies[s].is_validated(session),
                feasible_with_current_data=policies[s].is_feasible(session),
                eligible_to_authorize=False,
                note="no stable owner for cell",
            )
            for s in ARCHETYPES
        )
        return AuthorizationDecision(
            time=ts_iso,
            session=session,
            ownership_cell=ownership_cell,
            ownership_type="unknown",
            recommended_strategy="none",
            candidates=candidates,
            authorized_strategy=None,
            no_owner_reason=NO_OWNER_NO_STABLE_OWNER,
        )

    rec = str(cell_info.get("recommended_strategy", "none"))
    ctype = str(cell_info.get("type", "unknown"))
    if rec == "NO-TRADE":
        candidates = tuple(
            AuthorizationCandidate(
                strategy=s,
                chart_eligible=False,
                validated_for_context=policies[s].is_validated(session),
                feasible_with_current_data=policies[s].is_feasible(session),
                eligible_to_authorize=False,
                note="stable no-trade cell",
            )
            for s in ARCHETYPES
        )
        return AuthorizationDecision(
            time=ts_iso,
            session=session,
            ownership_cell=ownership_cell,
            ownership_type=ctype,
            recommended_strategy="NO-TRADE",
            candidates=candidates,
            authorized_strategy=None,
            no_owner_reason=NO_OWNER_STABLE_NO_TRADE,
        )

    candidates_list: list[AuthorizationCandidate] = []
    authorized_strategy: str | None = None
    no_owner_reason = ""

    for strategy in ARCHETYPES:
        policy = policies[strategy]
        chart_eligible = strategy == rec
        validated = policy.is_validated(session)
        feasible = policy.is_feasible(session)
        eligible = chart_eligible and validated and feasible
        note = ""
        if chart_eligible and not validated:
            note = "owner not validated in this session"
        elif chart_eligible and not feasible:
            note = "owner not feasible with current data/session constraints"
        elif chart_eligible and eligible:
            note = "authorized"
        candidates_list.append(
            AuthorizationCandidate(
                strategy=strategy,
                chart_eligible=chart_eligible,
                validated_for_context=validated,
                feasible_with_current_data=feasible,
                eligible_to_authorize=eligible,
                note=note,
            )
        )
        if eligible:
            authorized_strategy = strategy

    if authorized_strategy is None:
        owner_policy = policies.get(rec)
        if owner_policy is None:
            no_owner_reason = NO_OWNER_POLICY_MISSING
        elif not owner_policy.is_validated(session):
            no_owner_reason = NO_OWNER_NOT_VALIDATED
        elif not owner_policy.is_feasible(session):
            no_owner_reason = NO_OWNER_NOT_FEASIBLE
        else:
            no_owner_reason = NO_OWNER_NOT_AUTHORIZED

    return AuthorizationDecision(
        time=ts_iso,
        session=session,
        ownership_cell=ownership_cell,
        ownership_type=ctype,
        recommended_strategy=rec,
        candidates=tuple(candidates_list),
        authorized_strategy=authorized_strategy,
        no_owner_reason=no_owner_reason,
    )
