"""Best-effort legacy adapter: derive Phase 9 LlmDecisionOutput from corpus rows.

Used ONLY by the shadow replay (Step 2 acceptance, Step 8 Pass B) to test
validator code against the 241-trade forensic corpus. The corpus predates the
Phase 9 structured output, so we reconstruct fields heuristically:

  - caveats_detected: scan rationale for caveat-template tokens
  - caveat_resolution: text immediately after "however"/"but"/"although"
  - level_quality_claim.claim: scan rationale for level-language adjectives
  - side_burden_proof: 'side_bias_check' field (when present) else first
    paragraph of rationale
  - loss_asymmetry_argument: 'why_not_stop' or 'low_rr_edge' fields (when
    present) else None — validator will fire 'missing_argument' on these
  - evidence_refs: heuristic — if the row has structured fields (post-rollout),
    we report which were populated; legacy rows get an empty list

This is best-effort. Per Ambiguity Register entry A1, Pass B agreement
targets are stratified by structured-field availability.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from datetime import datetime
from .llm_output_schema import LevelQualityClaim, LlmDecisionOutput
from .snapshot import LevelPacket, Snapshot, new_snapshot_id, now_utc_iso


# --- Lexicons ----------------------------------------------------------------

_CAVEAT_TOKENS = (
    # mixed alignment
    ("mixed", "mixed_alignment"),
    ("conflicting", "mixed_alignment"),
    ("divergent", "mixed_alignment"),
    # contradiction / self-correction
    ("however", "contradiction"),
    ("although", "contradiction"),
    ("though ", "contradiction"),
    ("but ", "contradiction"),
    # weak level
    ("weak ", "weak_level"),
    ("thin", "weak_level"),
    ("shallow", "weak_level"),
    ("marginal", "weak_level"),
    # unresolved chop
    ("chop", "unresolved_chop"),
    ("rangebound", "unresolved_chop"),
    ("indecisive", "unresolved_chop"),
    # adverse macro / counter-trend
    ("counter-trend", "side_conflict"),
    ("countertrend", "side_conflict"),
    ("against macro", "adverse_macro"),
    ("macro headwind", "adverse_macro"),
    # thin evidence
    ("limited evidence", "thin_evidence"),
    ("few touches", "thin_evidence"),
)

_OVERREACH_STRONG = ("strong ", "clean ", "fresh ", "textbook", "decisive", "perfect setup")
_OVERREACH_ACCEPTABLE = ("acceptable", "decent ", "reasonable ", "solid ")
_OVERREACH_WEAK = ("weak", "thin", "shallow", "marginal")

_RESOLUTION_SPLITTERS = ("however", "but ", "although", "though ")


def derive_sessions(created_utc: Optional[str]) -> tuple[list[str], Optional[str]]:
    """Map an ISO UTC timestamp to (active_sessions, overlap).

    Session windows (approximate, UTC):
      Tokyo  00:00-09:00
      London 07:00-16:00
      NY     12:00-21:00
      tokyo_london overlap 07:00-09:00
      london_ny    overlap 12:00-16:00
    """
    if not created_utc:
        return [], None
    try:
        dt = datetime.fromisoformat(str(created_utc).replace("Z", "+00:00"))
    except ValueError:
        return [], None
    h = dt.hour + dt.minute / 60.0
    active: list[str] = []
    if 0 <= h < 9:
        active.append("tokyo")
    if 7 <= h < 16:
        active.append("london")
    if 12 <= h < 21:
        active.append("ny")
    overlap: Optional[str] = None
    if "tokyo" in active and "london" in active:
        overlap = "tokyo_london"
    elif "london" in active and "ny" in active:
        overlap = "london_ny"
    return active, overlap


def normalize_timeframe_alignment(raw: Optional[str]) -> Optional[str]:
    """Map heterogeneous legacy strings to v2 categorical values."""
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if not s:
        return None
    if "mixed" in s or "conflict" in s:
        return "mixed"
    if "aligned_buy" in s or s in ("buy_aligned", "all_buy"):
        return "aligned_buy"
    if "aligned_sell" in s or s in ("sell_aligned", "all_sell"):
        return "aligned_sell"
    if "neutral" in s:
        return "neutral"
    if "buy" in s and "sell" not in s:
        return "aligned_buy"
    if "sell" in s and "buy" not in s:
        return "aligned_sell"
    return None


def derive_macro_bias(side_bias_check: Optional[str]) -> Optional[str]:
    if not side_bias_check:
        return None
    s = side_bias_check.lower()
    if any(t in s for t in ("bullish", "bull bias", "uptrend", "macro long")):
        return "bullish"
    if any(t in s for t in ("bearish", "bear bias", "downtrend", "macro short")):
        return "bearish"
    if "neutral" in s:
        return "neutral"
    return None


def derive_catalyst_category(named_catalyst: Optional[str]) -> Optional[str]:
    """Heuristic: structure-only words alone → 'structure_only'; anything more
    specific → 'material'. Empty/None → unknown.
    """
    if not named_catalyst:
        return None
    s = str(named_catalyst).strip().lower()
    if not s:
        return None
    structure_only_set = {
        "support", "resistance", "reject", "rejection", "reclaim", "reclaimed",
        "fade", "pullback", "bounce",
    }
    # Filter stop-words/connectives so "Reject at resistance" parses as
    # structure-only — short words rarely contribute material context.
    words = [w for w in re.findall(r"[a-z]+", s) if len(w) >= 4]
    if not words:
        return None
    non_structural = [w for w in words if w not in structure_only_set]
    return "material" if non_structural else "structure_only"


# --- Adapter -----------------------------------------------------------------

def detect_caveats(rationale: str) -> list[str]:
    if not rationale:
        return []
    text = rationale.lower()
    seen: set[str] = set()
    for token, label in _CAVEAT_TOKENS:
        if token in text:
            seen.add(label)
    return sorted(seen)


def extract_resolution(rationale: str) -> Optional[str]:
    """Return text after the first contradiction marker, up to a sentence end."""
    if not rationale:
        return None
    text = rationale
    lower = text.lower()
    earliest_idx = -1
    earliest_tok = None
    for tok in _RESOLUTION_SPLITTERS:
        idx = lower.find(tok)
        if idx >= 0 and (earliest_idx < 0 or idx < earliest_idx):
            earliest_idx = idx
            earliest_tok = tok
    if earliest_idx < 0 or earliest_tok is None:
        return None
    after = text[earliest_idx + len(earliest_tok):].strip()
    # Cut at first sentence boundary (. ! ? newline)
    m = re.search(r"[.!?\n]", after)
    if m:
        after = after[: m.start()].strip()
    return after or None


def detect_level_claim(rationale: str) -> str:
    text = (rationale or "").lower()
    if any(t in text for t in _OVERREACH_STRONG):
        return "strong"
    if any(t in text for t in _OVERREACH_ACCEPTABLE):
        return "acceptable"
    if any(t in text for t in _OVERREACH_WEAK):
        return "weak"
    return "none"


def adapt_corpus_row(row: dict[str, Any]) -> tuple[LlmDecisionOutput, Snapshot]:
    """Build (LlmDecisionOutput, Snapshot) from a forensic-corpus row.

    Strategy: prefer structured fields when present; fall back to rationale
    parsing when absent. Snapshot fields the validators actually consult are
    `proposed_side` and `level_packet` — both reconstructed best-effort.
    """
    rationale = row.get("rationale") or ""
    side = (row.get("side") or "").lower()
    decision = (row.get("decision") or "place").lower()
    if decision not in ("place", "skip"):
        # Legacy rows often omit explicit decision; lots>0 implies placed
        decision = "place" if (row.get("lots") or 0) > 0 else "skip"

    # Structured-when-present fallback to rationale parsing
    caveats = detect_caveats(rationale)
    resolution = (
        row.get("caveat_resolution")
        or extract_resolution(rationale)
        or None
    )
    side_burden_proof = (
        row.get("side_bias_check")
        or row.get("edge_reason")
        or (rationale[:400] if rationale else None)
    )
    loss_asym = (
        row.get("why_not_stop")
        or row.get("low_rr_edge")
        or None
    )
    claim = detect_level_claim(rationale)
    lqc = LevelQualityClaim(claim=claim, evidence_field=None, score_cited=None)

    # Heuristic evidence_refs: list any populated structured field as a ref
    refs = sorted(
        k for k in (
            "trade_thesis", "named_catalyst", "side_bias_check",
            "setup_location", "edge_reason", "caveat_resolution",
            "micro_confirmation_event",
        )
        if row.get(k)
    )

    output = LlmDecisionOutput(
        decision=decision,
        primary_thesis=(row.get("trade_thesis") or rationale[:400] or ""),
        caveats_detected=caveats,
        caveat_resolution=resolution,
        level_quality_claim=lqc,
        side_burden_proof=side_burden_proof if side == "sell" else None,
        loss_asymmetry_argument=loss_asym,
        invalid_if=[],
        evidence_refs=refs,
    )

    # Snapshot: enough to drive the validators. Level packet is the load-bearing
    # field; reconstruct from heuristics. Score=72 (just above buy threshold)
    # for buy-CLR, 80 (below sell threshold) for sell-CLR — these are the
    # corpus-typical values per Phase 4 forensics.
    trigger = (row.get("trigger_family") or "").lower()
    is_clr = "critical_level" in trigger or "clr" in trigger
    pkt: Optional[LevelPacket] = None
    if is_clr and side in ("buy", "sell"):
        pkt = LevelPacket(
            side="buy_support" if side == "buy" else "sell_resistance",
            level_price=float(row.get("limit_price") or row.get("requested_price") or 0.0),
            level_quality_score=72.0 if side == "buy" else 80.0,
            distance_pips=0.0,
        )

    active_sessions, session_overlap = derive_sessions(row.get("created_utc"))
    snapshot = Snapshot(
        snapshot_id=new_snapshot_id(),
        created_utc=row.get("created_utc") or now_utc_iso(),
        proposed_side=side if side in ("buy", "sell") else None,
        sl_pips=float(row["sl"]) if row.get("sl") not in (None, "") else None,
        tp_pips=float(row["tp"]) if row.get("tp") not in (None, "") else None,
        level_packet=pkt,
        timeframe_alignment=normalize_timeframe_alignment(row.get("timeframe_alignment")),
        macro_bias=derive_macro_bias(row.get("side_bias_check")),
        catalyst_category=derive_catalyst_category(row.get("named_catalyst")),
        active_sessions=active_sessions,
        session_overlap=session_overlap,
    )
    return output, snapshot
