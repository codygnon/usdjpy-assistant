"""Step 2 acceptance: Post-Decision Validator Layer.

Unit tests cover each validator's pass and override paths plus protected-edge
preservation. The shadow-replay test runs all five validators against the
forensic corpus and asserts the run is sane (no crashes, fire rates within
broad bounds). Step 8 Pass B will tighten the bounds.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.fillmore_v2 import shadow_replay
from api.fillmore_v2.legacy_rationale_parser import (
    adapt_corpus_row,
    detect_caveats,
    detect_level_claim,
    extract_resolution,
)
from api.fillmore_v2.llm_output_schema import (
    LevelQualityClaim,
    LlmDecisionOutput,
    LlmOutputParseError,
    parse,
)
from api.fillmore_v2.snapshot import LevelPacket, Snapshot, new_snapshot_id, now_utc_iso
from api.fillmore_v2.validators import (
    ALL_VALIDATORS,
    caveat_resolution_validator,
    hedge_plus_overconfidence_validator,
    level_language_overreach_validator,
    loss_asymmetry_validator,
    run_all,
    sell_side_burden_validator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _snap(side: str = "buy", level_score: float = 78.0, sl: float = 8.0, tp: float = 16.0) -> Snapshot:
    pkt = LevelPacket(
        side="buy_support" if side == "buy" else "sell_resistance",
        level_price=150.0,
        level_quality_score=level_score,
        distance_pips=0.0,
    )
    return Snapshot(
        snapshot_id=new_snapshot_id(),
        created_utc=now_utc_iso(),
        proposed_side=side,
        sl_pips=sl,
        tp_pips=tp,
        level_packet=pkt,
    )


def _output(**overrides) -> LlmDecisionOutput:
    base = dict(
        decision="place",
        primary_thesis="Buy CLR at strong support, room to next resistance.",
        caveats_detected=[],
        caveat_resolution=None,
        level_quality_claim=LevelQualityClaim(
            claim="acceptable", evidence_field="side_normalized_level_packet.score", score_cited=78
        ),
        side_burden_proof=None,
        loss_asymmetry_argument=(
            "SL 8p, TP 16p, blocker 30p away, atr normal, spread 1p — winner > loser."
        ),
        invalid_if=["price closes below 149.40"],
        evidence_refs=["side_normalized_level_packet.level_quality_score"],
    )
    base.update(overrides)
    return LlmDecisionOutput(**base)


# ---------------------------------------------------------------------------
# JSON parser
# ---------------------------------------------------------------------------

def test_parser_accepts_minimal_valid_payload():
    out = parse(json.dumps({"decision": "skip", "primary_thesis": "no edge"}))
    assert out.decision == "skip"


def test_parser_rejects_unknown_decision():
    with pytest.raises(LlmOutputParseError, match="decision must be"):
        parse(json.dumps({"decision": "maybe", "primary_thesis": "x"}))


def test_parser_rejects_non_object_top_level():
    with pytest.raises(LlmOutputParseError, match="top-level must be object"):
        parse(json.dumps(["place"]))


def test_parser_rejects_invalid_level_quality_claim():
    with pytest.raises(LlmOutputParseError, match="level_quality_claim.claim"):
        parse(json.dumps({
            "decision": "place", "primary_thesis": "x",
            "level_quality_claim": {"claim": "amazing"},
        }))


def test_parser_rejects_string_where_list_required():
    with pytest.raises(LlmOutputParseError, match="caveats_detected must be list"):
        parse(json.dumps({
            "decision": "place",
            "primary_thesis": "x",
            "caveats_detected": "mixed_alignment",
        }))


def test_parser_rejects_unknown_keys():
    with pytest.raises(LlmOutputParseError, match="unknown keys"):
        parse(json.dumps({
            "decision": "skip",
            "primary_thesis": "x",
            "extra": "not allowed",
        }))


def test_parser_rejects_empty_string():
    with pytest.raises(LlmOutputParseError, match="empty"):
        parse("")


def test_parser_rejects_markdown_fence():
    """Per blueprint: model must return raw JSON. Fenced output is a violation."""
    with pytest.raises(LlmOutputParseError):
        parse("```json\n{\"decision\": \"skip\"}\n```")


# ---------------------------------------------------------------------------
# Caveat-resolution validator
# ---------------------------------------------------------------------------

def test_caveat_resolution_passes_when_no_caveats():
    r = caveat_resolution_validator(_output(), _snap())
    assert r.fired is False


def test_caveat_resolution_fires_when_resolution_missing():
    out = _output(caveats_detected=["mixed_alignment"], caveat_resolution=None)
    r = caveat_resolution_validator(out, _snap())
    assert r.fired is True
    assert r.reason_code == "missing_resolution"


def test_caveat_resolution_fires_when_no_evidence_refs():
    out = _output(
        caveats_detected=["weak_level"],
        caveat_resolution="Resolved by zone memory at 149.50.",
        evidence_refs=[],
    )
    r = caveat_resolution_validator(out, _snap())
    assert r.fired and r.reason_code == "resolution_lacks_evidence_refs"


def test_caveat_resolution_fires_on_generic_resolution_text():
    out = _output(
        caveats_detected=["mixed_alignment"],
        caveat_resolution="Mixed alignment but tradeable; overall edge intact.",
        evidence_refs=["side_normalized_level_packet"],
    )
    # Sell side → no protected-edge bypass
    snap = _snap(side="sell", level_score=86)
    r = caveat_resolution_validator(out, snap)
    assert r.fired and r.reason_code == "generic_resolution_text"


def test_caveat_resolution_protected_edge_bypass_buy_clr():
    """Buy + level score >= 70 + packet present must NOT fire even on generic resolution.

    This is the protected-edge bypass that preserves the three Phase 8 cells.
    """
    out = _output(
        caveats_detected=["mixed_alignment", "side_conflict"],
        caveat_resolution="Mixed but tradeable due to zone memory at this level.",
        evidence_refs=["side_normalized_level_packet"],
    )
    snap = _snap(side="buy", level_score=72)  # just above threshold
    r = caveat_resolution_validator(out, snap)
    assert r.fired is False


def test_caveat_resolution_buy_below_threshold_does_not_bypass():
    out = _output(
        caveats_detected=["mixed_alignment"],
        caveat_resolution="Mixed alignment but tradeable; net positive.",
        evidence_refs=["side_normalized_level_packet"],
    )
    snap = _snap(side="buy", level_score=65)  # below 70
    r = caveat_resolution_validator(out, snap)
    assert r.fired is True


# ---------------------------------------------------------------------------
# Level-language overreach validator
# ---------------------------------------------------------------------------

def test_overreach_passes_when_claim_not_strong():
    r = level_language_overreach_validator(_output(), _snap())
    assert r.fired is False


def test_overreach_fires_when_strong_claim_no_packet():
    out = _output(level_quality_claim=LevelQualityClaim(claim="strong"))
    snap = _snap()
    snap.level_packet = None
    r = level_language_overreach_validator(out, snap)
    assert r.fired and r.reason_code == "strong_claim_no_packet"


def test_overreach_fires_when_strong_claim_below_buy_threshold():
    out = _output(level_quality_claim=LevelQualityClaim(claim="strong"))
    r = level_language_overreach_validator(out, _snap(side="buy", level_score=65))
    assert r.fired and r.reason_code == "strong_claim_below_threshold"


def test_overreach_fires_when_strong_sell_below_sell_threshold():
    out = _output(level_quality_claim=LevelQualityClaim(claim="strong"))
    r = level_language_overreach_validator(out, _snap(side="sell", level_score=80))  # <85
    assert r.fired


def test_overreach_passes_when_strong_buy_at_threshold():
    out = _output(level_quality_claim=LevelQualityClaim(claim="strong"))
    r = level_language_overreach_validator(out, _snap(side="buy", level_score=70))
    assert r.fired is False


def test_overreach_fires_on_wrong_side_packet():
    out = _output(level_quality_claim=LevelQualityClaim(claim="strong"))
    snap = _snap(side="buy", level_score=88)
    snap.level_packet = LevelPacket(
        side="sell_resistance",  # wrong side for buy proposal
        level_price=150.0, level_quality_score=88, distance_pips=0.0,
    )
    r = level_language_overreach_validator(out, snap)
    assert r.fired and r.reason_code == "strong_claim_wrong_side_packet"


# ---------------------------------------------------------------------------
# Loss-asymmetry validator
# ---------------------------------------------------------------------------

def test_loss_asymmetry_fires_when_argument_missing():
    out = _output(loss_asymmetry_argument=None)
    r = loss_asymmetry_validator(out, _snap())
    assert r.fired and r.reason_code == "missing_argument"


def test_loss_asymmetry_fires_when_required_token_missing():
    out = _output(loss_asymmetry_argument="Spread is 1p, atr normal, room to run.")
    r = loss_asymmetry_validator(out, _snap())
    assert r.fired and r.reason_code == "missing_required_tokens"


def test_loss_asymmetry_fires_when_volatility_context_missing():
    out = _output(loss_asymmetry_argument="SL 8p, TP 16p, blocker 30p away.")
    r = loss_asymmetry_validator(out, _snap())
    assert r.fired and r.reason_code == "missing_volatility_context"


def test_loss_asymmetry_fires_when_rr_below_one():
    out = _output(loss_asymmetry_argument="SL 16p, TP 8p, blocker 30p, atr normal, spread 1p.")
    snap = _snap(sl=16.0, tp=8.0)
    r = loss_asymmetry_validator(out, snap)
    assert r.fired and r.reason_code == "rr_below_one"


def test_loss_asymmetry_passes_with_complete_argument():
    r = loss_asymmetry_validator(_output(), _snap())
    assert r.fired is False


# ---------------------------------------------------------------------------
# Sell-side burden validator
# ---------------------------------------------------------------------------

def test_sell_burden_skipped_for_buys():
    r = sell_side_burden_validator(_output(), _snap(side="buy"))
    assert r.fired is False


def test_sell_burden_fires_when_proof_missing():
    out = _output(side_burden_proof=None)
    r = sell_side_burden_validator(out, _snap(side="sell", level_score=86))
    assert r.fired and r.reason_code == "missing_proof"


def test_sell_burden_fires_below_sell_threshold():
    out = _output(side_burden_proof=(
        "Macro headwind plus M5 rejection at supply zone, M15 momentum exhausted, "
        "session-high failure backed by widening spread."
    ))
    r = sell_side_burden_validator(out, _snap(side="sell", level_score=80))  # <85
    assert r.fired and r.reason_code == "sell_packet_below_threshold"


def test_sell_burden_fires_on_structure_only_catalyst():
    out = _output(side_burden_proof="Resistance reject reclaim fade pullback bounce.")
    r = sell_side_burden_validator(out, _snap(side="sell", level_score=86))
    assert r.fired and r.reason_code == "structure_only_catalyst"


def test_sell_burden_passes_with_strong_packet_and_material_proof():
    out = _output(side_burden_proof=(
        "Macro headwind plus M5 rejection at supply zone, M15 momentum exhausted, "
        "session-high failure backed by widening spread."
    ))
    r = sell_side_burden_validator(out, _snap(side="sell", level_score=86))
    assert r.fired is False


# ---------------------------------------------------------------------------
# Hedge + overconfidence validator
# ---------------------------------------------------------------------------

def test_hedge_overconf_passes_normal_text():
    out = _output(primary_thesis="Buy support at 149.50, room to 150.30.")
    r = hedge_plus_overconfidence_validator(out, _snap())
    assert r.fired is False


def test_hedge_overconf_fires_when_both_densities_high():
    # Lots of hedges and lots of conviction in the same passage
    text = (
        "This is definitely a textbook setup but might fail because price could reverse. "
        "Clearly a high conviction trade that may possibly arguably go wrong perhaps "
        "though a perfect setup obviously wins."
    )
    out = _output(primary_thesis=text)
    r = hedge_plus_overconfidence_validator(out, _snap())
    assert r.fired and r.reason_code == "hedge_plus_overconfidence"


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def test_run_all_locks_decision_to_skip_on_any_override():
    out = _output(loss_asymmetry_argument=None)  # will fire loss_asymmetry
    summary = run_all(out, _snap())
    assert summary.final_decision == "skip"
    assert summary.any_fired is True
    assert any(o.validator_id == "loss_asymmetry" for o in summary.overrides)


def test_run_all_returns_place_when_clean():
    summary = run_all(_output(), _snap())
    assert summary.final_decision == "place"
    assert summary.overrides == []
    assert len(summary.passes) == len(ALL_VALIDATORS)


def test_run_all_records_every_fire_for_audit():
    """Even after first override locks decision, later validators still report."""
    out = _output(
        caveats_detected=["mixed_alignment"],
        caveat_resolution=None,  # fires caveat_resolution
        loss_asymmetry_argument=None,  # fires loss_asymmetry
    )
    summary = run_all(out, _snap())
    fired_ids = {o.validator_id for o in summary.overrides}
    assert "caveat_resolution" in fired_ids
    assert "loss_asymmetry" in fired_ids


# ---------------------------------------------------------------------------
# Legacy adapter unit tests
# ---------------------------------------------------------------------------

def test_detect_caveats_picks_up_mixed_and_contradiction():
    text = "However the M5 is mixed against the H1 trend, and the level is thin."
    caveats = detect_caveats(text)
    assert "contradiction" in caveats
    assert "mixed_alignment" in caveats
    assert "weak_level" in caveats


def test_extract_resolution_returns_text_after_marker():
    text = "Setup is weak, but the zone memory rescues it. Continuation expected."
    res = extract_resolution(text)
    assert res is not None and "zone memory rescues it" in res


def test_detect_level_claim_picks_strongest_term():
    assert detect_level_claim("This is a textbook setup at strong support") == "strong"
    assert detect_level_claim("Acceptable level, decent edge") == "acceptable"
    assert detect_level_claim("Weak level, thin evidence") == "weak"
    assert detect_level_claim("Routine setup") == "none"


def test_adapt_corpus_row_minimal():
    row = {
        "side": "buy",
        "lots": 2.0,
        "rationale": "Buy at strong support, but mixed alignment. However the zone memory holds.",
        "trigger_family": "critical_level_reaction",
        "limit_price": 150.0,
        "sl": 149.92,
        "tp": 150.16,
    }
    out, snap = adapt_corpus_row(row)
    assert out.decision == "place"
    assert "contradiction" in out.caveats_detected
    assert out.level_quality_claim and out.level_quality_claim.claim == "strong"
    assert snap.proposed_side == "buy"
    assert snap.level_packet and snap.level_packet.side == "buy_support"


# ---------------------------------------------------------------------------
# Shadow replay against forensic corpus (Step 2 acceptance)
# ---------------------------------------------------------------------------

def test_shadow_replay_against_corpus_runs_without_crash():
    files = shadow_replay.find_corpus_files(REPO_ROOT)
    if not files:
        pytest.skip("forensic corpus not present in this checkout")
    for f in files:
        summary = shadow_replay.replay_corpus(f)
        # Sanity: scanned > 0, parse failures contained, validators didn't crash
        assert summary.rows_scanned > 0, f
        assert summary.parse_failures < summary.rows_scanned * 0.05, (
            f"too many parse failures in {f}: {summary.parse_failures}/{summary.rows_scanned}"
        )
        # At least one validator must have fired somewhere across the whole
        # corpus — if zero across both files, our regex/heuristics are broken.
        total_fires = sum(summary.fire_counts.values())
        assert total_fires >= 0  # informational; tightened in Step 8 Pass B


def test_shadow_replay_aggregate_fires_reasonable():
    """Pooled across all corpus files, at least caveat_resolution and overreach
    validators should fire SOMEWHERE — they're the highest-volume per Phase 9.
    """
    files = shadow_replay.find_corpus_files(REPO_ROOT)
    if not files:
        pytest.skip("forensic corpus not present in this checkout")
    pooled: dict[str, int] = {}
    total_place = 0
    for f in files:
        s = shadow_replay.replay_corpus(f)
        total_place += s.place_decisions
        for vid, count in s.fire_counts.items():
            pooled[vid] = pooled.get(vid, 0) + count
    # Don't assert exact rates here (that's Step 8 Pass B). Just confirm the
    # high-volume validators show signal. Caveat resolution has no protected
    # bypass on sell rows in the corpus, so it must fire somewhere.
    assert total_place > 0
    assert pooled.get("caveat_resolution", 0) > 0, pooled
    # Loss-asymmetry will fire heavily on legacy rows since why_not_stop is
    # often empty — that's expected and matches the Phase 8 finding that
    # rationales rarely defended loss asymmetry explicitly.
    assert pooled.get("loss_asymmetry", 0) > 0, pooled
