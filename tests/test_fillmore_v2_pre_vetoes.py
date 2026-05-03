"""Step 3 acceptance: Pre-Decision Veto Layer (V1, V2)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.fillmore_v2 import shadow_replay
from api.fillmore_v2.legacy_rationale_parser import (
    derive_catalyst_category,
    derive_macro_bias,
    derive_sessions,
    normalize_timeframe_alignment,
)
from api.fillmore_v2.pre_decision_vetoes import (
    ALL_PRE_VETOES,
    LEVEL_SCORE_BUY_MIN,
    LEVEL_SCORE_SELL_MIN,
    PROTECTED_LOTS_CAP,
    run_pre_vetoes,
    v1_sell_caveat_template_veto,
    v2_mixed_overlap_entry_veto,
)
from api.fillmore_v2.snapshot import LevelPacket, Snapshot, new_snapshot_id, now_utc_iso


def _snap(
    *,
    side: str = "sell",
    score: float = 80.0,
    align: str = "mixed",
    macro: str = "bullish",
    catalyst: str = "structure_only",
    overlap: str = "london_ny",
) -> Snapshot:
    pkt = LevelPacket(
        side="buy_support" if side == "buy" else "sell_resistance",
        level_price=150.0,
        level_quality_score=score,
        distance_pips=0.0,
    )
    return Snapshot(
        snapshot_id=new_snapshot_id(),
        created_utc=now_utc_iso(),
        proposed_side=side,
        sl_pips=8.0,
        tp_pips=16.0,
        level_packet=pkt,
        timeframe_alignment=align,
        macro_bias=macro,
        catalyst_category=catalyst,
        active_sessions=["london", "ny"] if overlap == "london_ny" else [],
        session_overlap=overlap,
    )


# ---------------------------------------------------------------------------
# V1 — Sell caveat-template veto
# ---------------------------------------------------------------------------

def test_v1_skips_buys_entirely():
    s = _snap(side="buy", score=72, align="aligned_buy", macro="bullish", catalyst="material")
    r = v1_sell_caveat_template_veto(s)
    assert r.fired is False


def test_v1_fires_on_classic_caveat_template_sell():
    """Mixed alignment + bullish macro + structure-only catalyst + score < 85 → fire."""
    s = _snap(side="sell", score=80, align="mixed", macro="bullish", catalyst="structure_only")
    r = v1_sell_caveat_template_veto(s)
    assert r.fired is True
    assert r.reason_code == "sell_caveat_template"


def test_v1_fires_when_level_packet_missing():
    s = _snap()
    s.level_packet = None
    r = v1_sell_caveat_template_veto(s)
    assert r.fired is True
    assert "level_packet_missing" in (r.reason_detail or "")


def test_v1_passes_when_no_precursors():
    """Aligned-sell + bearish macro + material catalyst + score >= 85 → no precursors → pass."""
    s = _snap(
        side="sell", score=88, align="aligned_sell",
        macro="bearish", catalyst="material",
    )
    r = v1_sell_caveat_template_veto(s)
    assert r.fired is False


def test_v1_passes_when_precursors_exist_but_score_high_and_material():
    """Has one precursor (mixed alignment) but level >= 85 AND catalyst material → no fire."""
    s = _snap(
        side="sell", score=88, align="mixed",
        macro="bearish", catalyst="material",
    )
    r = v1_sell_caveat_template_veto(s)
    # Precursors exist (mixed), but score_low=False AND no_material=False → no fire
    assert r.fired is False


def test_v1_fails_open_when_alignment_unknown():
    """If timeframe_alignment is None, it cannot count as a precursor by itself.

    But other precursors (low score, bullish macro, structure-only) can still fire V1.
    """
    s = _snap(side="sell", score=80, align=None, macro="bullish", catalyst="structure_only")
    r = v1_sell_caveat_template_veto(s)
    assert r.fired is True


def test_v1_no_protected_bypass_for_sell():
    """All three protected cells are buy-side; V1 must never bypass on a sell."""
    s = _snap(side="sell", score=72, align="mixed", macro="bullish", catalyst="structure_only")
    r = v1_sell_caveat_template_veto(s, deterministic_lots=2.0)
    assert r.fired is True


# ---------------------------------------------------------------------------
# V2 — Mixed-overlap entry veto with protected bypass
# ---------------------------------------------------------------------------

def test_v2_passes_outside_overlap_sessions():
    s = _snap(side="sell", overlap=None, align="mixed")
    r = v2_mixed_overlap_entry_veto(s)
    assert r.fired is False


def test_v2_passes_when_alignment_not_mixed():
    s = _snap(side="sell", overlap="london_ny", align="aligned_sell")
    r = v2_mixed_overlap_entry_veto(s)
    assert r.fired is False


def test_v2_fires_on_mixed_overlap_sell():
    s = _snap(side="sell", overlap="london_ny", align="mixed", score=80)
    r = v2_mixed_overlap_entry_veto(s)
    assert r.fired is True
    assert r.reason_code == "mixed_overlap_entry"


def test_v2_fires_on_mixed_overlap_buy_below_threshold():
    """Buy + score < 70 → no protected bypass → fire."""
    s = _snap(side="buy", overlap="london_ny", align="mixed", score=65)
    r = v2_mixed_overlap_entry_veto(s)
    assert r.fired is True


def test_v2_protected_bypass_buy_clr_at_threshold():
    """Buy + CLR + score >= 70 + lots <= 4 → bypass, no fire."""
    s = _snap(side="buy", overlap="london_ny", align="mixed", score=70)
    r = v2_mixed_overlap_entry_veto(s, deterministic_lots=2.0)
    assert r.fired is False
    assert r.bypass_applied is True


def test_v2_protected_bypass_denied_when_lots_above_cap():
    """Buy + score >= 70 + lots > 4 → bypass condition fails → fire."""
    s = _snap(side="buy", overlap="london_ny", align="mixed", score=80)
    r = v2_mixed_overlap_entry_veto(s, deterministic_lots=4.5)
    assert r.fired is True


def test_v2_protected_bypass_when_lots_unknown():
    """deterministic_lots None (Step 4 not yet computed) → conservative bypass for buy-CLR."""
    s = _snap(side="buy", overlap="london_ny", align="mixed", score=78)
    r = v2_mixed_overlap_entry_veto(s, deterministic_lots=None)
    assert r.fired is False
    assert r.bypass_applied is True


def test_v2_fires_for_tokyo_london_overlap_too():
    s = _snap(side="sell", overlap="tokyo_london", align="mixed", score=88)
    r = v2_mixed_overlap_entry_veto(s)
    assert r.fired is True


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def test_run_pre_vetoes_short_circuits_on_any_fire():
    s = _snap(side="sell", overlap="london_ny", align="mixed",
              macro="bullish", catalyst="structure_only", score=80)
    summary = run_pre_vetoes(s)
    assert summary.skip_before_call is True
    fired_ids = {f.veto_id for f in summary.fires}
    # Both V1 (sell caveat template) and V2 (mixed overlap) should fire
    assert "v1_sell_caveat_template" in fired_ids
    assert "v2_mixed_overlap" in fired_ids


def test_run_pre_vetoes_returns_clean_when_no_fires():
    s = _snap(side="buy", overlap=None, align="aligned_buy",
              macro="bullish", catalyst="material", score=78)
    summary = run_pre_vetoes(s, deterministic_lots=2.0)
    assert summary.skip_before_call is False
    assert summary.fires == []
    assert len(summary.passes) == len(ALL_PRE_VETOES)


def test_run_pre_vetoes_protected_buy_clr_passes_through():
    """The cardinal preservation case: buy-CLR in mixed overlap with score >= 70.

    V1 doesn't fire (not a sell). V2 bypasses on protected criteria. Net: no skip.
    """
    s = _snap(side="buy", overlap="london_ny", align="mixed",
              macro="bullish", catalyst="material", score=72)
    summary = run_pre_vetoes(s, deterministic_lots=3.0)
    assert summary.skip_before_call is False
    assert any(p.bypass_applied for p in summary.passes)


# ---------------------------------------------------------------------------
# Legacy adapter helpers
# ---------------------------------------------------------------------------

def test_derive_sessions_known_windows():
    # 13:00 UTC → London + NY → london_ny overlap
    active, ov = derive_sessions("2026-04-20T13:00:00+00:00")
    assert "london" in active and "ny" in active
    assert ov == "london_ny"
    # 08:00 UTC → Tokyo + London → tokyo_london overlap
    active2, ov2 = derive_sessions("2026-04-20T08:00:00+00:00")
    assert "tokyo" in active2 and "london" in active2
    assert ov2 == "tokyo_london"
    # 22:00 UTC → none of the three
    active3, ov3 = derive_sessions("2026-04-20T22:00:00+00:00")
    assert active3 == []
    assert ov3 is None


def test_derive_sessions_handles_z_suffix_and_garbage():
    active, _ = derive_sessions("2026-04-20T13:00:00Z")
    assert "ny" in active
    assert derive_sessions("not-a-date") == ([], None)
    assert derive_sessions(None) == ([], None)


def test_normalize_timeframe_alignment():
    assert normalize_timeframe_alignment("mixed_alignment") == "mixed"
    assert normalize_timeframe_alignment("ALIGNED_BUY") == "aligned_buy"
    assert normalize_timeframe_alignment("buy_aligned") == "aligned_buy"
    assert normalize_timeframe_alignment("neutral") == "neutral"
    assert normalize_timeframe_alignment("") is None
    assert normalize_timeframe_alignment(None) is None


def test_derive_macro_bias():
    assert derive_macro_bias("Macro long, bullish above 200dma") == "bullish"
    assert derive_macro_bias("Bear bias, macro short context") == "bearish"
    assert derive_macro_bias("Neutral chop") == "neutral"
    assert derive_macro_bias("") is None
    assert derive_macro_bias(None) is None


def test_derive_catalyst_category():
    assert derive_catalyst_category("Reject at resistance") == "structure_only"
    assert derive_catalyst_category("CPI release with M5 hammer rejection") == "material"
    assert derive_catalyst_category(None) is None
    assert derive_catalyst_category("") is None


# ---------------------------------------------------------------------------
# Shadow replay covers pre-vetoes against the corpus
# ---------------------------------------------------------------------------

def test_shadow_replay_pre_vetoes_run_without_crash():
    files = shadow_replay.find_corpus_files(REPO_ROOT)
    if not files:
        pytest.skip("forensic corpus not present in this checkout")
    for f in files:
        s = shadow_replay.replay_corpus(f)
        # Counters must exist and be coherent
        assert s.pre_veto_fire_counts is not None
        assert "v1_sell_caveat_template" in s.pre_veto_fire_counts
        assert "v2_mixed_overlap" in s.pre_veto_fire_counts
        # If anything fired, skipped_before_call must be > 0
        if any(c > 0 for c in s.pre_veto_fire_counts.values()):
            assert s.pre_veto_skipped_before_call > 0


def test_shadow_replay_pre_vetoes_aggregate_signal():
    """Pooled across the corpus, V1 must fire on at least some sells.

    Phase 9 V1 floor on the 241-trade Phase 8 universe is 80 blocks.
    The available corpus subset is different rows so we don't gate on the
    exact count here — Step 8 Pass A does that.
    """
    files = shadow_replay.find_corpus_files(REPO_ROOT)
    if not files:
        pytest.skip("forensic corpus not present in this checkout")
    pooled = {"v1_sell_caveat_template": 0, "v2_mixed_overlap": 0}
    sells_by_side = 0
    for f in files:
        s = shadow_replay.replay_corpus(f)
        for k, v in s.pre_veto_fire_counts.items():
            pooled[k] = pooled.get(k, 0) + v
        sells_by_side += s.pre_veto_by_side.get("sell", {}).get("v1_sell_caveat_template", 0)
    # V1 must produce at least one fire on the corpus
    assert pooled["v1_sell_caveat_template"] > 0, pooled
    # All V1 fires must be on sells
    assert sells_by_side == pooled["v1_sell_caveat_template"]
