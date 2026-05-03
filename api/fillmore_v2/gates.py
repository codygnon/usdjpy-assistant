"""Gate Layer Redesign — Phase 9 Step 5 (PHASE9.2).

Decides which setups REACH the LLM. Step 6 should first run gates without a
pre-veto summary to confirm/log raw gate eligibility, then run Step 3
pre-vetoes, then re-run/finalize gates with `pre_veto_summary` so fired vetoes
or protected bypasses are reflected in the selected gate and candidate audit.

Design verdicts from PHASE9.2:

  - **Critical Level Reaction → REDESIGN.** Split into separate buy-CLR and
    sell-CLR eligibility paths with different evidence burdens. Buy-CLR
    contains all three preserved-edge cells; sell-CLR collapsed via the
    caveat template.
  - **Momentum Continuation → KEEP-CONDITIONAL.** Only with deterministic
    sizing, profit-path room ≥ 1R, no runner-preservation prompt.
  - **Mean Reversion → KEEP-SMALL-N.** Size-cap to 1 lot until 30 forward
    closes prove non-negative expectancy. Carries `sizing_cap_lots=1.0`.
  - **Non-primary gates → KILL-BY-DEFAULT.** Not registered at all. Exception
    path requires an explicit gate_experiment_id and a separate analytics
    table — out of scope for v2 launch.

The runner-preservation language ban (PHASE9.4) and exit-extension ban
(PHASE9.7) are enforced in Step 6 (prompt) and Step 7 (exits) — they do not
gate eligibility here. Sizing constraints are enforced via `sizing_cap_lots`
which Step 4's sizing function should treat as an upper bound when present.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .pre_decision_vetoes import PreVetoRunSummary
from .snapshot import Snapshot


# --- Thresholds (PHASE9.2) ---------------------------------------------------

LEVEL_SCORE_BUY_MIN = 70.0
LEVEL_SCORE_SELL_MIN = 85.0
LEVEL_SCORE_SELL_MIXED_OVERLAP_MIN = 90.0
MOMENTUM_MIN_RR = 1.0  # path room must be >= 1R (sl_pips)
MEAN_REVERSION_MAX_LOTS = 1.0  # until 30 closes prove non-negative
SPREAD_NORMAL_MAX_PIPS = 2.5  # mean-reversion + momentum spread cap


# --- Result type --------------------------------------------------------------

@dataclass
class GateEligibility:
    gate_id: str
    eligible: bool
    reason_code: Optional[str] = None
    reason_detail: Optional[str] = None
    sizing_cap_lots: Optional[float] = None  # gate-imposed cap (mean-reversion)
    score: float = 0.0  # gate confidence in [0, 1]; for snapshot.all_gate_candidates

    def to_audit_record(self) -> dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "eligible": self.eligible,
            "reason_code": self.reason_code,
            "reason_detail": self.reason_detail,
            "sizing_cap_lots": self.sizing_cap_lots,
            "score": self.score,
        }


def _eligible(gate_id: str, *, score: float = 1.0, sizing_cap_lots: Optional[float] = None) -> GateEligibility:
    return GateEligibility(
        gate_id=gate_id, eligible=True, score=score, sizing_cap_lots=sizing_cap_lots,
    )


def _ineligible(gate_id: str, code: str, detail: str = "") -> GateEligibility:
    return GateEligibility(
        gate_id=gate_id, eligible=False, reason_code=code, reason_detail=detail, score=0.0,
    )


# --- Helpers ------------------------------------------------------------------

def _is_mixed_overlap(snapshot: Snapshot) -> bool:
    return (
        snapshot.timeframe_alignment == "mixed"
        and snapshot.session_overlap in ("london_ny", "tokyo_london")
    )


def _pre_veto_blocks(pre_veto_summary: Optional[PreVetoRunSummary]) -> Optional[str]:
    """Return a reason string if pre-vetoes block this setup, else None.

    A protected-edge bypass means a veto's eligibility check passed (with
    `bypass_applied=True`); that's NOT a block.
    """
    if pre_veto_summary is None or not pre_veto_summary.skip_before_call:
        return None
    fired_ids = [f.veto_id for f in pre_veto_summary.fires]
    return f"pre_veto_fired={fired_ids}"


def _expected_packet_side(snapshot: Snapshot) -> Optional[str]:
    if snapshot.proposed_side == "buy":
        return "buy_support"
    if snapshot.proposed_side == "sell":
        return "sell_resistance"
    return None


# --- Critical Level Reaction (split) -----------------------------------------

def buy_clr_eligible(
    snapshot: Snapshot,
    *,
    pre_veto_summary: Optional[PreVetoRunSummary] = None,
) -> GateEligibility:
    """PHASE9.2 Buy-CLR eligibility. All required:

    1. Side-normalized level packet exists.
    2. entry_wall.side == buy_support.
    3. level_quality_score >= 70.
    4. No recent broken-then-failed reclaim against the buy.
    5. Profit-path blocker distance >= planned risk distance (or marked clear).
    6. Pre-decision V1/V2 don't fire (or protected bypass).

    Buy-CLR is the home of the three protected-edge cells. The level-score
    threshold is intentionally lower than sell-CLR (70 vs 85) to preserve
    them.
    """
    gid = "buy_clr"
    if snapshot.proposed_side != "buy":
        return _ineligible(gid, "wrong_side_for_gate", f"proposed_side={snapshot.proposed_side}")
    pkt = snapshot.level_packet
    if pkt is None:
        return _ineligible(gid, "level_packet_missing")
    if pkt.side != "buy_support":
        return _ineligible(gid, "wrong_side_packet", f"packet.side={pkt.side}")
    if pkt.level_quality_score < LEVEL_SCORE_BUY_MIN:
        return _ineligible(
            gid, "level_score_below_min",
            f"score={pkt.level_quality_score} < {LEVEL_SCORE_BUY_MIN}",
        )
    age = snapshot.level_age_metadata
    if age is not None and age.broken_then_reclaimed:
        return _ineligible(gid, "broken_then_reclaimed", "level was broken and reclaimed; fast-failure risk")
    blocker = pkt.profit_path_blocker_distance_pips
    sl = snapshot.sl_pips
    if blocker is not None and sl is not None and sl > 0 and blocker < sl:
        return _ineligible(
            gid, "blocker_inside_risk",
            f"blocker={blocker}p < sl={sl}p (no room to next structure)",
        )
    pv_block = _pre_veto_blocks(pre_veto_summary)
    if pv_block:
        return _ineligible(gid, "pre_veto_fired", pv_block)
    # Score: linear lift over the buy threshold, capped at 1.0
    score = min(1.0, (pkt.level_quality_score - LEVEL_SCORE_BUY_MIN) / 30.0 + 0.5)
    return _eligible(gid, score=score)


def sell_clr_eligible(
    snapshot: Snapshot,
    *,
    pre_veto_summary: Optional[PreVetoRunSummary] = None,
) -> GateEligibility:
    """PHASE9.2 Sell-CLR eligibility. All required:

    1. level_quality_score >= 85 (or >= 90 in mixed overlap).
    2. entry_wall.side == sell_resistance.
    3. H1 and M5 not both against the short (timeframe_alignment != aligned_buy).
    4. macro_bias not explicitly bullish.
    5. Mixed-overlap requires score >= 90 AND material catalyst.
    6. No structure-only catalyst.

    Defense in depth with V1: V1 also checks structure-only catalyst and
    timeframe alignment. The gate kills bad sells *before* the pre-decision
    veto layer runs, so the audit trail shows the gate as the blocker.
    """
    gid = "sell_clr"
    if snapshot.proposed_side != "sell":
        return _ineligible(gid, "wrong_side_for_gate", f"proposed_side={snapshot.proposed_side}")
    pkt = snapshot.level_packet
    if pkt is None:
        return _ineligible(gid, "level_packet_missing")
    if pkt.side != "sell_resistance":
        return _ineligible(gid, "wrong_side_packet", f"packet.side={pkt.side}")

    mixed_overlap = _is_mixed_overlap(snapshot)
    threshold = LEVEL_SCORE_SELL_MIXED_OVERLAP_MIN if mixed_overlap else LEVEL_SCORE_SELL_MIN
    if pkt.level_quality_score < threshold:
        return _ineligible(
            gid, "level_score_below_threshold",
            f"score={pkt.level_quality_score} < {threshold} (mixed_overlap={mixed_overlap})",
        )
    if snapshot.timeframe_alignment == "aligned_buy":
        return _ineligible(gid, "h1_m5_against_short", "timeframe alignment fully buy on a sell")
    if snapshot.macro_bias == "bullish":
        return _ineligible(gid, "macro_against_short", "macro_bias=bullish on a sell")
    if mixed_overlap and snapshot.catalyst_category != "material":
        return _ineligible(
            gid, "mixed_overlap_needs_material_catalyst",
            f"mixed-overlap exception requires material catalyst, got {snapshot.catalyst_category}",
        )
    if snapshot.catalyst_category == "structure_only":
        return _ineligible(gid, "structure_only_catalyst", "")

    pv_block = _pre_veto_blocks(pre_veto_summary)
    if pv_block:
        return _ineligible(gid, "pre_veto_fired", pv_block)
    score = min(1.0, (pkt.level_quality_score - LEVEL_SCORE_SELL_MIN) / 15.0 + 0.5)
    return _eligible(gid, score=score)


# --- Momentum Continuation ---------------------------------------------------

def momentum_continuation_eligible(
    snapshot: Snapshot,
    *,
    pre_veto_summary: Optional[PreVetoRunSummary] = None,
) -> GateEligibility:
    """PHASE9.2 Momentum: KEEP-CONDITIONAL.

    1. Room to next S/R is at least 1R (sl_pips).
    2. Volatility regime is not elevated (else sizing layer halves lots —
       so we still pass eligibility but flag the cap).
    3. Spread within normal regime.
    4. Pre-decision vetoes don't fire.

    Runner-preservation language is banned at the prompt layer (Step 6),
    not the gate; this gate exists only to reject momentum setups that
    cannot earn 1R before hitting structure.
    """
    gid = "momentum_continuation"
    pkt = snapshot.level_packet
    sl = snapshot.sl_pips
    if pkt is None or pkt.profit_path_blocker_distance_pips is None:
        return _ineligible(gid, "no_path_blocker_data", "cannot prove 1R room")
    expected_side = _expected_packet_side(snapshot)
    if expected_side is None:
        return _ineligible(gid, "missing_proposed_side", f"proposed_side={snapshot.proposed_side}")
    if pkt.side != expected_side:
        return _ineligible(gid, "wrong_side_packet", f"packet.side={pkt.side} expected={expected_side}")
    if sl is None or sl <= 0:
        return _ineligible(gid, "missing_sl_pips", "")
    room = pkt.profit_path_blocker_distance_pips
    if room < MOMENTUM_MIN_RR * sl:
        return _ineligible(
            gid, "insufficient_path_room",
            f"room={room}p < {MOMENTUM_MIN_RR}R={sl}p",
        )
    if snapshot.spread_pips is not None and snapshot.spread_pips > SPREAD_NORMAL_MAX_PIPS:
        return _ineligible(
            gid, "spread_above_normal",
            f"spread={snapshot.spread_pips}p > {SPREAD_NORMAL_MAX_PIPS}p",
        )
    pv_block = _pre_veto_blocks(pre_veto_summary)
    if pv_block:
        return _ineligible(gid, "pre_veto_fired", pv_block)
    return _eligible(gid, score=0.6)


# --- Mean Reversion ----------------------------------------------------------

def mean_reversion_eligible(
    snapshot: Snapshot,
    *,
    pre_veto_summary: Optional[PreVetoRunSummary] = None,
) -> GateEligibility:
    """PHASE9.2 Mean Reversion: KEEP-SMALL-N. Carries sizing_cap_lots=1.0
    until 30 forward closes prove non-negative expectancy.

    1. Level packet indicates exhaustion at a side-relevant level.
    2. Spread and volatility inside normal regime.
    3. Sizing capped to 1 lot.

    Conservative exhaustion proxy: structural_origin contains 'exhaustion'
    or the level is a daily/weekly extremum (pdh, pdl, wh, wl). When in
    doubt, fail closed — this gate has the smallest forward sample.
    """
    gid = "mean_reversion"
    pkt = snapshot.level_packet
    if pkt is None:
        return _ineligible(gid, "level_packet_missing")
    expected_side = _expected_packet_side(snapshot)
    if expected_side is None:
        return _ineligible(gid, "missing_proposed_side", f"proposed_side={snapshot.proposed_side}")
    if pkt.side != expected_side:
        return _ineligible(gid, "wrong_side_packet", f"packet.side={pkt.side} expected={expected_side}")
    if snapshot.spread_pips is not None and snapshot.spread_pips > SPREAD_NORMAL_MAX_PIPS:
        return _ineligible(gid, "spread_above_normal", f"spread={snapshot.spread_pips}p")
    if snapshot.volatility_regime == "elevated":
        return _ineligible(gid, "volatility_elevated", "")
    origin = (pkt.structural_origin or "").lower()
    exhaustion_markers = ("exhaustion", "extremum", "pdh", "pdl", "wh", "wl")
    if not any(m in origin for m in exhaustion_markers):
        return _ineligible(
            gid, "no_exhaustion_marker",
            f"structural_origin={pkt.structural_origin!r} lacks exhaustion marker",
        )
    pv_block = _pre_veto_blocks(pre_veto_summary)
    if pv_block:
        return _ineligible(gid, "pre_veto_fired", pv_block)
    return _eligible(gid, score=0.4, sizing_cap_lots=MEAN_REVERSION_MAX_LOTS)


# --- Registry & orchestrator -------------------------------------------------

GateFn = Callable[..., GateEligibility]

# Only PRIMARY gates are registered. Non-primary gates (post_spike_retracement,
# failed_breakout, trend_expansion, etc.) are KILLED per PHASE9.2 — they can
# only be added back via an explicit gate_experiment_id with a separate
# analytics table.
PRIMARY_GATES: dict[str, GateFn] = {
    "buy_clr": buy_clr_eligible,
    "sell_clr": sell_clr_eligible,
    "momentum_continuation": momentum_continuation_eligible,
    "mean_reversion": mean_reversion_eligible,
}


@dataclass
class GateRunSummary:
    selected_gate: Optional[GateEligibility]  # the first eligible gate, or None
    candidates: list[GateEligibility] = field(default_factory=list)

    @property
    def any_eligible(self) -> bool:
        return self.selected_gate is not None and self.selected_gate.eligible

    def to_audit_records(self) -> list[dict[str, Any]]:
        return [c.to_audit_record() for c in self.candidates]


def evaluate_all_gates(
    snapshot: Snapshot,
    *,
    pre_veto_summary: Optional[PreVetoRunSummary] = None,
    preferred_order: Optional[tuple[str, ...]] = None,
) -> GateRunSummary:
    """Run every primary gate. Returns the first eligible gate (by
    `preferred_order` or registration order) plus all candidates for the
    snapshot's `all_gate_candidates` field.

    PHASE9.8 item 11: every candidate must be logged with its score and
    veto reason so multi-gate conflicts are auditable. This function
    populates the audit list whether or not any gate is eligible.
    """
    order = preferred_order or tuple(PRIMARY_GATES)
    candidates: list[GateEligibility] = []
    selected: Optional[GateEligibility] = None
    for gate_id in order:
        fn = PRIMARY_GATES.get(gate_id)
        if fn is None:
            continue
        result = fn(snapshot, pre_veto_summary=pre_veto_summary)
        candidates.append(result)
        if result.eligible and selected is None:
            selected = result
    return GateRunSummary(selected_gate=selected, candidates=candidates)


def is_non_primary_gate_killed(gate_id: str) -> bool:
    """True if the gate id is in the killed-by-default set per PHASE9.2."""
    KILLED = {
        "post_spike_retracement", "failed_breakout", "trend_expansion",
    }
    return gate_id in KILLED
