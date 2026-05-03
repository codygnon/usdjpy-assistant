"""Post-Decision Validator Layer — Phase 9 Step 2 (PHASE9.5).

Five deterministic validators. Each takes the parsed LLM JSON output plus the
v2 Snapshot and returns either pass or override-to-skip. The validator is the
structural cure for caveat laundering: the LLM may say `place`, but it cannot
execute through an invalid explanation.

Per PHASE9.5: every override logs validator_id, raw LLM payload, failed field,
skip reason. The orchestrator (`run_all`) collects these for persistence.

Order matters: validators run in the order returned by `ALL_VALIDATORS`. The
first override wins; subsequent validators still run (so the audit log shows
every fire), but the decision is locked once anything overrides.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

from .llm_output_schema import LlmDecisionOutput, LevelQualityClaim
from .snapshot import Snapshot


# --- Result type --------------------------------------------------------------

@dataclass
class ValidatorResult:
    validator_id: str
    fired: bool  # True = override to skip
    failed_field: Optional[str] = None
    reason_code: Optional[str] = None
    reason_detail: Optional[str] = None
    raw_llm_json: Optional[dict[str, Any]] = None  # for audit log

    def to_audit_record(self) -> dict[str, Any]:
        return asdict(self)


def _passing(validator_id: str) -> ValidatorResult:
    return ValidatorResult(validator_id=validator_id, fired=False)


# --- Thresholds (PHASE9.5) ----------------------------------------------------

LEVEL_SCORE_BUY_MIN = 70.0
LEVEL_SCORE_SELL_MIN = 85.0
LEVEL_SCORE_SELL_MIXED_OVERLAP_MIN = 90.0

# Generic catalysts that don't carry independent meaning. Per PHASE9.5: reject
# if the catalyst phrasing is one of these standing alone, without context.
GENERIC_CATALYST_TOKENS = (
    "support", "resistance", "reject", "rejection", "reclaim", "reclaimed",
    "fade", "pullback", "bounce",
)

# V5 hedge/conviction lexicons. Top-quartile densities per PHASE9.5;
# thresholds tuned during Step 8 Pass B from the corpus distribution.
HEDGE_TOKENS = (
    "could", "might", "maybe", "perhaps", "possibly", "may ", "appears to ",
    "seems ", "potentially", "arguably",
)
CONVICTION_TOKENS = (
    "definitely", "clearly", "obviously", "certainly", "decisive",
    "textbook", "perfect setup", "must work", "high conviction", "no question",
)
HEDGE_DENSITY_MIN = 0.020  # >= 2% of words are hedges → top quartile
CONVICTION_DENSITY_MIN = 0.015  # >= 1.5% of words are conviction → top quartile


# --- V1: Caveat-Resolution Validator ------------------------------------------

# Generic resolution phrases that don't actually cite evidence — failure mode
# from Phase 4: model says "but the catalyst overrides" without naming a field.
_GENERIC_RESOLUTION_PHRASES = (
    "but tradeable", "still tradeable", "edge remains", "overall edge",
    "overall the setup", "net positive", "tilt remains", "comfortable with",
)


def caveat_resolution_validator(
    output: LlmDecisionOutput,
    snapshot: Snapshot,
) -> ValidatorResult:
    """PHASE9.5: caveats_detected non-empty requires caveat_resolution with at
    least one valid evidence ref AND no generic-only text.

    Preserves protected buy-CLR: if caveat is resolved by level packet score
    >= LEVEL_SCORE_BUY_MIN, the validator passes even on caveat-language CLR.
    """
    vid = "caveat_resolution"
    if output.decision != "place":
        return _passing(vid)
    if not output.caveats_detected:
        return _passing(vid)
    res = (output.caveat_resolution or "").strip()
    if not res:
        return ValidatorResult(
            validator_id=vid, fired=True,
            failed_field="caveat_resolution",
            reason_code="missing_resolution",
            reason_detail=f"caveats_detected={output.caveats_detected} but no resolution provided",
            raw_llm_json=asdict(output),
        )
    if not output.evidence_refs:
        return ValidatorResult(
            validator_id=vid, fired=True,
            failed_field="evidence_refs",
            reason_code="resolution_lacks_evidence_refs",
            reason_detail="caveat_resolution provided but evidence_refs is empty",
            raw_llm_json=asdict(output),
        )
    res_lower = res.lower()
    if any(g in res_lower for g in _GENERIC_RESOLUTION_PHRASES):
        # Protected-edge bypass: buy + level score >= 70 with packet present
        if (
            snapshot.proposed_side == "buy"
            and snapshot.level_packet
            and snapshot.level_packet.level_quality_score >= LEVEL_SCORE_BUY_MIN
        ):
            return _passing(vid)
        return ValidatorResult(
            validator_id=vid, fired=True,
            failed_field="caveat_resolution",
            reason_code="generic_resolution_text",
            reason_detail=f"resolution contains generic phrase, no protected-edge bypass",
            raw_llm_json=asdict(output),
        )
    return _passing(vid)


# --- V2: Level-Language Overreach Validator -----------------------------------

def level_language_overreach_validator(
    output: LlmDecisionOutput,
    snapshot: Snapshot,
) -> ValidatorResult:
    """PHASE9.5: 'strong/clean/fresh/textbook' terms require packet score above
    threshold (>= LEVEL_SCORE_BUY_MIN for buy, >= LEVEL_SCORE_SELL_MIN for sell)
    and matching side field.

    V6 in the blueprint replay table. Preserves protected cells when packet
    evidence is valid.
    """
    vid = "level_language_overreach"
    if output.decision != "place":
        return _passing(vid)
    lqc = output.level_quality_claim
    if lqc is None or not lqc.is_overreach_term():
        return _passing(vid)
    pkt = snapshot.level_packet
    if pkt is None:
        return ValidatorResult(
            validator_id=vid, fired=True,
            failed_field="level_quality_claim",
            reason_code="strong_claim_no_packet",
            reason_detail="claim='strong' but no side-normalized level packet",
            raw_llm_json=asdict(output),
        )
    side = snapshot.proposed_side
    threshold = LEVEL_SCORE_BUY_MIN if side == "buy" else LEVEL_SCORE_SELL_MIN
    if pkt.level_quality_score < threshold:
        return ValidatorResult(
            validator_id=vid, fired=True,
            failed_field="level_quality_claim",
            reason_code="strong_claim_below_threshold",
            reason_detail=(
                f"claim='strong' but packet score {pkt.level_quality_score} < {threshold} for side={side}"
            ),
            raw_llm_json=asdict(output),
        )
    # Side mismatch: model claimed strong but cited the wrong side's packet
    expected_side_tag = "buy_support" if side == "buy" else "sell_resistance"
    if pkt.side != expected_side_tag:
        return ValidatorResult(
            validator_id=vid, fired=True,
            failed_field="level_quality_claim.evidence_field",
            reason_code="strong_claim_wrong_side_packet",
            reason_detail=f"claim='strong' but packet side {pkt.side} != expected {expected_side_tag}",
            raw_llm_json=asdict(output),
        )
    return _passing(vid)


# --- V3: Loss-Asymmetry Validator ---------------------------------------------

# Tokens we expect the loss_asymmetry_argument to mention by name.
_REQUIRED_LOSS_ASYM_TOKENS = ("sl", "tp", "blocker")  # spread/volatility checked separately
_OPTIONAL_LOSS_ASYM_TOKENS = ("spread", "volatility", "atr")


def loss_asymmetry_validator(
    output: LlmDecisionOutput,
    snapshot: Snapshot,
) -> ValidatorResult:
    """PHASE9.5: every place must explain why expected loser < expected winner.

    Checks the argument cites SL, TP, blocker distance, and at least one of
    (spread, volatility/atr). RR must be defensible: tp_pips >= 1.0 * sl_pips
    when both are present on the snapshot.
    """
    vid = "loss_asymmetry"
    if output.decision != "place":
        return _passing(vid)
    arg = (output.loss_asymmetry_argument or "").strip().lower()
    if not arg:
        return ValidatorResult(
            validator_id=vid, fired=True,
            failed_field="loss_asymmetry_argument",
            reason_code="missing_argument",
            reason_detail="place decision missing loss_asymmetry_argument",
            raw_llm_json=asdict(output),
        )
    missing_required = [t for t in _REQUIRED_LOSS_ASYM_TOKENS if t not in arg]
    if missing_required:
        return ValidatorResult(
            validator_id=vid, fired=True,
            failed_field="loss_asymmetry_argument",
            reason_code="missing_required_tokens",
            reason_detail=f"argument missing tokens: {missing_required}",
            raw_llm_json=asdict(output),
        )
    if not any(t in arg for t in _OPTIONAL_LOSS_ASYM_TOKENS):
        return ValidatorResult(
            validator_id=vid, fired=True,
            failed_field="loss_asymmetry_argument",
            reason_code="missing_volatility_context",
            reason_detail="argument cites SL/TP/blocker but no spread/volatility/ATR",
            raw_llm_json=asdict(output),
        )
    if snapshot.sl_pips is not None and snapshot.tp_pips is not None:
        if snapshot.sl_pips > 0 and snapshot.tp_pips < snapshot.sl_pips:
            return ValidatorResult(
                validator_id=vid, fired=True,
                failed_field="loss_asymmetry_argument",
                reason_code="rr_below_one",
                reason_detail=f"tp_pips {snapshot.tp_pips} < sl_pips {snapshot.sl_pips}",
                raw_llm_json=asdict(output),
            )
    return _passing(vid)


# --- V4: Sell-Side Burden Validator -------------------------------------------

def sell_side_burden_validator(
    output: LlmDecisionOutput,
    snapshot: Snapshot,
) -> ValidatorResult:
    """PHASE9.5: every sell must cite side-aligned level packet AND a material
    catalyst that beats the sell-side base rate.

    Catches structure-only sell catalysts ("rejected resistance", "reclaim
    failure") which Phase 8 identified as the sell-side template collapse.
    """
    vid = "sell_side_burden"
    if output.decision != "place":
        return _passing(vid)
    if snapshot.proposed_side != "sell":
        return _passing(vid)
    proof = (output.side_burden_proof or "").strip()
    if not proof:
        return ValidatorResult(
            validator_id=vid, fired=True,
            failed_field="side_burden_proof",
            reason_code="missing_proof",
            reason_detail="sell decision missing side_burden_proof",
            raw_llm_json=asdict(output),
        )
    pkt = snapshot.level_packet
    if pkt is None or pkt.side != "sell_resistance":
        return ValidatorResult(
            validator_id=vid, fired=True,
            failed_field="side_burden_proof",
            reason_code="missing_sell_resistance_packet",
            reason_detail="sell needs side-normalized sell_resistance packet",
            raw_llm_json=asdict(output),
        )
    if pkt.level_quality_score < LEVEL_SCORE_SELL_MIN:
        return ValidatorResult(
            validator_id=vid, fired=True,
            failed_field="level_packet.level_quality_score",
            reason_code="sell_packet_below_threshold",
            reason_detail=f"sell packet score {pkt.level_quality_score} < {LEVEL_SCORE_SELL_MIN}",
            raw_llm_json=asdict(output),
        )
    # Material catalyst check: proof can't reduce to generic structural tokens
    proof_lower = proof.lower()
    proof_words = re.findall(r"[a-z]+", proof_lower)
    non_generic = [w for w in proof_words if w not in GENERIC_CATALYST_TOKENS]
    if len(non_generic) < max(5, int(0.6 * len(proof_words))):
        return ValidatorResult(
            validator_id=vid, fired=True,
            failed_field="side_burden_proof",
            reason_code="structure_only_catalyst",
            reason_detail="sell proof reduces to generic structural tokens",
            raw_llm_json=asdict(output),
        )
    return _passing(vid)


# --- V5: Hedge + Overconfidence Validator -------------------------------------

def _density(text: str, tokens: tuple[str, ...]) -> float:
    if not text:
        return 0.0
    text_lower = text.lower()
    word_count = len(re.findall(r"\w+", text_lower))
    if word_count == 0:
        return 0.0
    hits = sum(text_lower.count(t) for t in tokens)
    return hits / word_count


def hedge_plus_overconfidence_validator(
    output: LlmDecisionOutput,
    snapshot: Snapshot,
) -> ValidatorResult:
    """PHASE9.5 V5: if hedge density AND conviction density both top-quartile,
    skip. Captures the 'definitely a strong setup, but might fail because…'
    pattern Phase 7 caught.
    """
    vid = "hedge_plus_overconfidence"
    if output.decision != "place":
        return _passing(vid)
    text_pool = " ".join(filter(None, [
        output.primary_thesis,
        output.caveat_resolution,
        output.side_burden_proof,
        output.loss_asymmetry_argument,
    ]))
    hedge_d = _density(text_pool, HEDGE_TOKENS)
    conv_d = _density(text_pool, CONVICTION_TOKENS)
    if hedge_d >= HEDGE_DENSITY_MIN and conv_d >= CONVICTION_DENSITY_MIN:
        return ValidatorResult(
            validator_id=vid, fired=True,
            failed_field="primary_thesis",
            reason_code="hedge_plus_overconfidence",
            reason_detail=f"hedge_density={hedge_d:.4f} conviction_density={conv_d:.4f}",
            raw_llm_json=asdict(output),
        )
    return _passing(vid)


# --- Orchestrator -------------------------------------------------------------

ValidatorFn = Callable[[LlmDecisionOutput, Snapshot], ValidatorResult]

ALL_VALIDATORS: tuple[ValidatorFn, ...] = (
    caveat_resolution_validator,
    level_language_overreach_validator,
    loss_asymmetry_validator,
    sell_side_burden_validator,
    hedge_plus_overconfidence_validator,
)


@dataclass
class ValidatorRunSummary:
    """Aggregated result of running all validators on one LLM output."""
    final_decision: str  # 'place' or 'skip' after overrides
    overrides: list[ValidatorResult] = field(default_factory=list)
    passes: list[ValidatorResult] = field(default_factory=list)

    @property
    def any_fired(self) -> bool:
        return bool(self.overrides)

    def to_audit_records(self) -> list[dict[str, Any]]:
        return [r.to_audit_record() for r in self.overrides]


def run_all(output: LlmDecisionOutput, snapshot: Snapshot) -> ValidatorRunSummary:
    """Run every validator. First override locks the decision to skip; later
    validators still run so the audit log shows every fire.
    """
    overrides: list[ValidatorResult] = []
    passes: list[ValidatorResult] = []
    for fn in ALL_VALIDATORS:
        r = fn(output, snapshot)
        (overrides if r.fired else passes).append(r)
    final = "skip" if overrides else output.decision
    return ValidatorRunSummary(final_decision=final, overrides=overrides, passes=passes)
