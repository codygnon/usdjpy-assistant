"""
Shared contracts for chart-first diagnostics and the upcoming shadow entry-engine layer.

This module is intentionally dependency-light (no scripts.* imports). It documents
and normalizes vocabulary that otherwise drifts across engines.

Session vocabulary (critical for shadow invocation)
--------------------------------------------------
1. **Phase 3 / authorization bar replay** uses ``core.phase3_integrated_engine.classify_session``:
   returns ``\"tokyo\" | \"london\" | \"ny\" | None``. Outside windows → ``None``.
   Diagnostics often coerce ``None`` → ``\"dead\"`` for JSON counters.

2. **V44 analysis labels** (``scripts.backtest_session_momentum.classify_session``) use:
   ``\"tokyo\" | \"london\" | \"ny_overlap\" | \"dead\"``. There is no ``\"ny\"`` string;
   NY window is ``\"ny_overlap\"``.

3. **Merged stack TradeRow.entry_session** for V44 is normalized to ``\"ny\"`` in
   ``backtest_merged_integrated_tokyo_london_v2_ny`` (not ``ny_overlap``).

Shadow adapters comparing “authorized in NY” vs “V44 internal session tag” must map:
``ny_overlap`` ↔ ``ny`` and treat ``None``/``dead`` as non-NY for authorization context.
"""

from __future__ import annotations

# Stable archetype ids (must match ownership table + chart authorization).
from core.ownership_table import STRATEGY_KEYS

ARCHETYPE_IDS: tuple[str, ...] = STRATEGY_KEYS

# Reasons emitted by ``core.chart_authorization.authorize_bar`` when ``authorized_strategy`` is None.
NO_OWNER_NO_STABLE_OWNER = "no_stable_owner"
NO_OWNER_STABLE_NO_TRADE = "stable_no_trade"
NO_OWNER_POLICY_MISSING = "owner_policy_missing"
NO_OWNER_NOT_VALIDATED = "owner_not_validated_in_session"
NO_OWNER_NOT_FEASIBLE = "owner_not_feasible"
NO_OWNER_NOT_AUTHORIZED = "owner_not_authorized"

NO_OWNER_REASONS: tuple[str, ...] = (
    NO_OWNER_NO_STABLE_OWNER,
    NO_OWNER_STABLE_NO_TRADE,
    NO_OWNER_POLICY_MISSING,
    NO_OWNER_NOT_VALIDATED,
    NO_OWNER_NOT_FEASIBLE,
    NO_OWNER_NOT_AUTHORIZED,
)

# Phase 3 router session labels (classify_session non-None values).
PHASE3_TRADING_SESSIONS: frozenset[str] = frozenset({"tokyo", "london", "ny"})

# V44 momentum engine analysis labels.
V44_ANALYSIS_SESSIONS: frozenset[str] = frozenset({"tokyo", "london", "ny_overlap", "dead"})


def analysis_session_to_phase3_session(label: str | None) -> str | None:
    """
    Map V44-style analysis session to the closest Phase 3 / authorization label.

    - ny_overlap → ny
    - tokyo / london unchanged
    - dead / unknown / empty → None
    """
    if not label:
        return None
    s = str(label).strip().lower()
    if s == "ny_overlap":
        return "ny"
    if s in PHASE3_TRADING_SESSIONS:
        return s
    return None


def phase3_session_to_analysis_bucket(session: str | None) -> str:
    """
    Map Phase 3 classify_session output to a coarse bucket name safe for logs.

    None → dead; ny → ny_overlap label for side-by-side with V44 harness output.
    """
    if session is None:
        return "dead"
    if session == "ny":
        return "ny_overlap"
    return str(session)
