"""Pre-Decision Veto Layer — Phase 9 Step 3 (PHASE9.3).

Two pre-decision vetoes (V1, V2) that fire BEFORE the LLM is called. Run after
gate eligibility, before LLM dispatch. A fired veto means: skip-before-call,
record the reason, charge no token cost, persist a `decision='skip'` row with
`pre_veto_fired_json` populated.

V5 and V6 from PHASE9.3 § Adopted Rule Stack are already implemented in
`api.fillmore_v2.validators` (post-decision); they need the LLM output to fire.

V3 (runner v3) is retired — its lesson is encoded in the prompt removal list
and exit layer (Step 7).
V4 (Wednesday CLR) and V7 (blunt CLR sell) are rejected per PHASE9.3.

Unknown categorical inputs do not count as veto precursors by themselves. The
exception is a missing sell-side level packet in V1: without side-normalized
level evidence, the sell caveat-template risk is treated as unsafe to send to
the LLM. Blocking-field halt (Step 1) remains responsible for systemic capture
gaps; these vetoes make per-setup skip decisions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

from .snapshot import Snapshot


# --- Thresholds (PHASE9.3 / PHASE9.5) -----------------------------------------

LEVEL_SCORE_BUY_MIN = 70.0           # Protected buy-CLR floor; matches validators.py
LEVEL_SCORE_SELL_MIN = 85.0          # Sell-side burden threshold
PROTECTED_LOTS_CAP = 4.0             # Bypass cap for buy-CLR per PHASE9.3 V2
OVERLAP_SESSIONS = ("london_ny", "tokyo_london")


# --- Result type --------------------------------------------------------------

@dataclass
class PreVetoResult:
    veto_id: str
    fired: bool
    reason_code: Optional[str] = None
    reason_detail: Optional[str] = None
    bypass_applied: bool = False  # True if a protected-edge bypass prevented a fire

    def to_audit_record(self) -> dict[str, Any]:
        return asdict(self)


def _passing(veto_id: str, *, bypass: bool = False) -> PreVetoResult:
    return PreVetoResult(veto_id=veto_id, fired=False, bypass_applied=bypass)


# --- V1: Refined sell caveat-template veto ------------------------------------

def v1_sell_caveat_template_veto(
    snapshot: Snapshot,
    deterministic_lots: Optional[float] = None,
) -> PreVetoResult:
    """PHASE9.3 V1 refined: block sell setups whose ex-ante packet has caveat-
    template precursors AND (level score < 85 OR no material catalyst).

    Precursors checked at gate-time (no rationale yet):
      - Mixed/countertrend timeframe alignment
      - Thin level packet (score < sell threshold or packet missing)
      - Contradictory macro (macro_bias bullish on a sell)
      - Structure-only catalyst

    No protected-edge bypass — none of the three protected cells are sell-side.
    """
    vid = "v1_sell_caveat_template"
    if snapshot.proposed_side != "sell":
        return _passing(vid)

    precursors: list[str] = []
    align = snapshot.timeframe_alignment
    if align in ("mixed", "aligned_buy"):
        precursors.append(f"countertrend_or_mixed:{align}")
    if snapshot.level_packet is None:
        precursors.append("level_packet_missing")
    elif snapshot.level_packet.level_quality_score < LEVEL_SCORE_SELL_MIN:
        precursors.append(f"level_score_below_sell_min:{snapshot.level_packet.level_quality_score}")
    if snapshot.macro_bias == "bullish":
        precursors.append("contradictory_macro_bullish")
    if snapshot.catalyst_category == "structure_only":
        precursors.append("structure_only_catalyst")

    if not precursors:
        return _passing(vid)

    score_low = (
        snapshot.level_packet is None
        or snapshot.level_packet.level_quality_score < LEVEL_SCORE_SELL_MIN
    )
    no_material = snapshot.catalyst_category != "material"
    if score_low or no_material:
        return PreVetoResult(
            veto_id=vid,
            fired=True,
            reason_code="sell_caveat_template",
            reason_detail=(
                f"precursors={precursors} score_low={score_low} no_material={no_material}"
            ),
        )
    return _passing(vid)


# --- V2: Refined mixed-overlap entry signature with protected bypass ---------

def v2_mixed_overlap_entry_veto(
    snapshot: Snapshot,
    deterministic_lots: Optional[float] = None,
) -> PreVetoResult:
    """PHASE9.3 V2 refined: block mixed timeframe alignment in London/NY or
    Tokyo/London overlap.

    Protected bypass: do NOT fire when the setup is buy-CLR with packet
    score >= LEVEL_SCORE_BUY_MIN AND deterministic_lots <= PROTECTED_LOTS_CAP.
    Bypass preserves the three Phase 8 protected cells.

    `deterministic_lots` may be None pre-Step-4; the bypass conservatively
    requires the cap to be met OR the lots to be unknown (fail-open into
    the bypass when sizing isn't yet computed).
    """
    vid = "v2_mixed_overlap"
    if snapshot.session_overlap not in OVERLAP_SESSIONS:
        return _passing(vid)
    if snapshot.timeframe_alignment != "mixed":
        return _passing(vid)

    # Protected buy-CLR bypass
    pkt = snapshot.level_packet
    is_protected_buy_clr = (
        snapshot.proposed_side == "buy"
        and pkt is not None
        and pkt.side == "buy_support"
        and pkt.level_quality_score >= LEVEL_SCORE_BUY_MIN
        and (deterministic_lots is None or deterministic_lots <= PROTECTED_LOTS_CAP)
    )
    if is_protected_buy_clr:
        return _passing(vid, bypass=True)

    return PreVetoResult(
        veto_id=vid,
        fired=True,
        reason_code="mixed_overlap_entry",
        reason_detail=(
            f"session_overlap={snapshot.session_overlap} alignment=mixed "
            f"side={snapshot.proposed_side} "
            f"score={pkt.level_quality_score if pkt else None}"
        ),
    )


# --- Orchestrator -------------------------------------------------------------

PreVetoFn = Callable[[Snapshot, Optional[float]], PreVetoResult]

ALL_PRE_VETOES: tuple[PreVetoFn, ...] = (
    v1_sell_caveat_template_veto,
    v2_mixed_overlap_entry_veto,
)


@dataclass
class PreVetoRunSummary:
    skip_before_call: bool
    fires: list[PreVetoResult] = field(default_factory=list)
    passes: list[PreVetoResult] = field(default_factory=list)

    def to_audit_records(self) -> list[dict[str, Any]]:
        return [r.to_audit_record() for r in self.fires]


def run_pre_vetoes(
    snapshot: Snapshot,
    deterministic_lots: Optional[float] = None,
) -> PreVetoRunSummary:
    """Run all pre-decision vetoes. Returns aggregated summary.

    All vetoes run (so the audit log shows every fire), but the orchestrator
    short-circuits the LLM call if any fired.
    """
    fires: list[PreVetoResult] = []
    passes: list[PreVetoResult] = []
    for fn in ALL_PRE_VETOES:
        r = fn(snapshot, deterministic_lots)
        (fires if r.fired else passes).append(r)
    return PreVetoRunSummary(
        skip_before_call=bool(fires),
        fires=fires,
        passes=passes,
    )
