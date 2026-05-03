"""Phase 9 LLM JSON output schema (PHASE9.4).

Defined in Step 2 because the validators must type against it. Step 6 (LLM
Decision Layer) will build the actual prompt and the call-site parser; this
module is the contract between them.

Strict JSON parsing per blueprint: malformed → caller forces skip and logs
the parse failure. The validator layer never sees a partial LlmDecisionOutput.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class LevelQualityClaim:
    """Subdocument: model's claim about the side-normalized level packet."""
    claim: str  # 'none' | 'weak' | 'acceptable' | 'strong'
    evidence_field: Optional[str] = None  # 'side_normalized_level_packet.<field>'
    score_cited: Optional[float] = None

    _ALLOWED_CLAIMS = ("none", "weak", "acceptable", "strong")

    def is_overreach_term(self) -> bool:
        """Per PHASE9.5 V6 — 'strong'/'clean'/'fresh'/'textbook' need packet support.

        We collapse all overreach terms to claim=='strong' in the structured
        schema; the validator is responsible for proving the packet supports it.
        """
        return self.claim == "strong"


@dataclass
class LlmDecisionOutput:
    """The complete Phase 9 LLM output. PHASE9.4 schema verbatim.

    NOT what the model returns as a Python object — it returns JSON text. Use
    `parse(raw_text)` to get an instance, which raises on any malformed
    structure so the caller can override to skip per spec.
    """
    decision: str  # 'place' | 'skip'
    primary_thesis: str  # max 200 tokens (advisory; not enforced here)
    caveats_detected: list[str] = field(default_factory=list)
    caveat_resolution: Optional[str] = None  # required iff caveats_detected non-empty
    level_quality_claim: Optional[LevelQualityClaim] = None
    side_burden_proof: Optional[str] = None  # required for every sell
    loss_asymmetry_argument: Optional[str] = None  # required for every place
    invalid_if: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)


class LlmOutputParseError(ValueError):
    """Raised when the model's JSON is malformed or missing required keys.

    Per PHASE9.4: caller catches this, forces decision=skip, and logs the parse
    failure. Validators never run on a partial structure.
    """


_REQUIRED_KEYS = ("decision", "primary_thesis")
_ALLOWED_KEYS = {
    "decision",
    "primary_thesis",
    "caveats_detected",
    "caveat_resolution",
    "level_quality_claim",
    "side_burden_proof",
    "loss_asymmetry_argument",
    "invalid_if",
    "evidence_refs",
}


def _optional_string(payload: dict[str, Any], key: str) -> Optional[str]:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise LlmOutputParseError(f"{key} must be string or null")
    return value


def _string_list(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list):
        raise LlmOutputParseError(f"{key} must be list of strings")
    if not all(isinstance(item, str) for item in value):
        raise LlmOutputParseError(f"{key} must be list of strings")
    return value


def parse(raw_text: str) -> LlmDecisionOutput:
    """Strict JSON parser for the LLM output. Raises on any deviation.

    No fallback regex extraction, no markdown-fence stripping (the prompt
    instructs JSON-only). If the model wraps in fences, that is itself a
    schema violation we want to log.
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise LlmOutputParseError("empty model output")
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise LlmOutputParseError(f"json decode failed: {e}") from e
    if not isinstance(payload, dict):
        raise LlmOutputParseError(f"top-level must be object, got {type(payload).__name__}")
    unknown = sorted(set(payload) - _ALLOWED_KEYS)
    if unknown:
        raise LlmOutputParseError(f"unknown keys: {unknown}")
    for k in _REQUIRED_KEYS:
        if k not in payload:
            raise LlmOutputParseError(f"missing required key: {k}")
    decision = payload["decision"]
    if decision not in ("place", "skip"):
        raise LlmOutputParseError(f"decision must be 'place' or 'skip', got {decision!r}")
    if not isinstance(payload["primary_thesis"], str):
        raise LlmOutputParseError("primary_thesis must be string")

    lqc_raw = payload.get("level_quality_claim")
    lqc: Optional[LevelQualityClaim] = None
    if lqc_raw is not None:
        if not isinstance(lqc_raw, dict):
            raise LlmOutputParseError("level_quality_claim must be object")
        claim_val = lqc_raw.get("claim")
        if claim_val not in LevelQualityClaim._ALLOWED_CLAIMS:
            raise LlmOutputParseError(
                f"level_quality_claim.claim must be one of {LevelQualityClaim._ALLOWED_CLAIMS}, got {claim_val!r}"
            )
        score = lqc_raw.get("score_cited")
        if score is not None and not isinstance(score, (int, float)):
            raise LlmOutputParseError("level_quality_claim.score_cited must be number or null")
        evidence_field = lqc_raw.get("evidence_field")
        if evidence_field is not None and not isinstance(evidence_field, str):
            raise LlmOutputParseError("level_quality_claim.evidence_field must be string or null")
        lqc = LevelQualityClaim(
            claim=claim_val,
            evidence_field=evidence_field,
            score_cited=float(score) if score is not None else None,
        )

    return LlmDecisionOutput(
        decision=decision,
        primary_thesis=payload["primary_thesis"],
        caveats_detected=_string_list(payload, "caveats_detected"),
        caveat_resolution=_optional_string(payload, "caveat_resolution"),
        level_quality_claim=lqc,
        side_burden_proof=_optional_string(payload, "side_burden_proof"),
        loss_asymmetry_argument=_optional_string(payload, "loss_asymmetry_argument"),
        invalid_if=_string_list(payload, "invalid_if"),
        evidence_refs=_string_list(payload, "evidence_refs"),
    )
