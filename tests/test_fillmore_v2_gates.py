"""Step 5 acceptance: Gate Layer Redesign (PHASE9.2)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.fillmore_v2.gates import (
    LEVEL_SCORE_BUY_MIN,
    LEVEL_SCORE_SELL_MIN,
    LEVEL_SCORE_SELL_MIXED_OVERLAP_MIN,
    MEAN_REVERSION_MAX_LOTS,
    MOMENTUM_MIN_RR,
    PRIMARY_GATES,
    SPREAD_NORMAL_MAX_PIPS,
    buy_clr_eligible,
    evaluate_all_gates,
    is_non_primary_gate_killed,
    mean_reversion_eligible,
    momentum_continuation_eligible,
    sell_clr_eligible,
)
from api.fillmore_v2.pre_decision_vetoes import PreVetoResult, PreVetoRunSummary, run_pre_vetoes
from api.fillmore_v2.snapshot import (
    LevelAgeMetadata,
    LevelPacket,
    Snapshot,
    new_snapshot_id,
    now_utc_iso,
)


def _snap(
    *,
    side: str = "buy",
    score: float = 78.0,
    pkt_side: str = None,
    blocker_pips: float = 30.0,
    sl: float = 8.0,
    tp: float = 16.0,
    align: str = "aligned_buy",
    macro: str = "neutral",
    catalyst: str = "material",
    overlap: str = None,
    spread: float = 1.0,
    vol: str = "normal",
    broken: bool = False,
    structural_origin: str = "h1_swing_low",
) -> Snapshot:
    if pkt_side is None:
        pkt_side = "buy_support" if side == "buy" else "sell_resistance"
    pkt = LevelPacket(
        side=pkt_side,
        level_price=150.0,
        level_quality_score=score,
        distance_pips=0.0,
        profit_path_blocker_distance_pips=blocker_pips,
        structural_origin=structural_origin,
    )
    age = LevelAgeMetadata(broken_then_reclaimed=broken)
    return Snapshot(
        snapshot_id=new_snapshot_id(),
        created_utc=now_utc_iso(),
        proposed_side=side,
        sl_pips=sl,
        tp_pips=tp,
        spread_pips=spread,
        level_packet=pkt,
        level_age_metadata=age,
        timeframe_alignment=align,
        macro_bias=macro,
        catalyst_category=catalyst,
        active_sessions=["london", "ny"] if overlap == "london_ny" else [],
        session_overlap=overlap,
        volatility_regime=vol,
    )


# ---------------------------------------------------------------------------
# Buy-CLR
# ---------------------------------------------------------------------------

def test_buy_clr_eligible_with_protected_packet():
    s = _snap(side="buy", score=72)  # just above min
    r = buy_clr_eligible(s)
    assert r.eligible is True
    assert r.gate_id == "buy_clr"
    assert 0.5 <= r.score <= 1.0


def test_buy_clr_rejects_wrong_proposed_side():
    s = _snap(side="sell")  # also flips packet to sell_resistance
    r = buy_clr_eligible(s)
    assert r.eligible is False
    assert r.reason_code == "wrong_side_for_gate"


def test_buy_clr_rejects_wrong_packet_side():
    s = _snap(side="buy", pkt_side="sell_resistance")
    r = buy_clr_eligible(s)
    assert r.eligible is False
    assert r.reason_code == "wrong_side_packet"


def test_buy_clr_rejects_score_below_70():
    s = _snap(side="buy", score=65)
    r = buy_clr_eligible(s)
    assert r.eligible is False
    assert r.reason_code == "level_score_below_min"


def test_buy_clr_rejects_broken_then_reclaimed():
    s = _snap(side="buy", score=80, broken=True)
    r = buy_clr_eligible(s)
    assert r.eligible is False
    assert r.reason_code == "broken_then_reclaimed"


def test_buy_clr_rejects_blocker_inside_risk():
    """Profit-path blocker < SL → no room to next structure → reject."""
    s = _snap(side="buy", score=80, blocker_pips=5.0, sl=8.0)
    r = buy_clr_eligible(s)
    assert r.eligible is False
    assert r.reason_code == "blocker_inside_risk"


def test_buy_clr_passes_when_blocker_unknown():
    """If profit-path blocker isn't measured, don't fail-closed on it.

    The blueprint says 'or explicitly marked clear'; absence of data is
    treated as 'unknown', not 'blocked'.
    """
    s = _snap(side="buy", score=80, blocker_pips=None)
    r = buy_clr_eligible(s)
    assert r.eligible is True


def test_buy_clr_blocked_by_pre_veto():
    s = _snap(side="buy", score=80)
    pv = PreVetoRunSummary(
        skip_before_call=True,
        fires=[PreVetoResult(veto_id="v2_mixed_overlap", fired=True, reason_code="mixed_overlap_entry")],
    )
    r = buy_clr_eligible(s, pre_veto_summary=pv)
    assert r.eligible is False
    assert r.reason_code == "pre_veto_fired"
    assert "v2_mixed_overlap" in (r.reason_detail or "")


# ---------------------------------------------------------------------------
# Sell-CLR — higher evidence burden
# ---------------------------------------------------------------------------

def test_sell_clr_eligible_with_clean_packet():
    s = _snap(side="sell", score=86, align="aligned_sell", macro="bearish", catalyst="material")
    r = sell_clr_eligible(s)
    assert r.eligible is True


def test_sell_clr_rejects_score_below_85():
    s = _snap(side="sell", score=80, align="aligned_sell", macro="bearish", catalyst="material")
    r = sell_clr_eligible(s)
    assert r.eligible is False
    assert r.reason_code == "level_score_below_threshold"


def test_sell_clr_mixed_overlap_requires_score_90():
    """Mixed-overlap exception: needs >= 90 even though base threshold is 85."""
    s = _snap(side="sell", score=88, align="mixed", overlap="london_ny",
              macro="bearish", catalyst="material")
    r = sell_clr_eligible(s)
    assert r.eligible is False
    assert r.reason_code == "level_score_below_threshold"
    s2 = _snap(side="sell", score=92, align="mixed", overlap="london_ny",
               macro="bearish", catalyst="material")
    r2 = sell_clr_eligible(s2)
    assert r2.eligible is True


def test_sell_clr_mixed_overlap_requires_material_catalyst():
    s = _snap(side="sell", score=92, align="mixed", overlap="london_ny",
              macro="bearish", catalyst="structure_only")
    r = sell_clr_eligible(s)
    # Either structure_only check or mixed-overlap material check fires; both block.
    assert r.eligible is False


def test_sell_clr_rejects_h1_m5_against_short():
    s = _snap(side="sell", score=88, align="aligned_buy",  # both timeframes buy = bad for sell
              macro="bearish", catalyst="material")
    r = sell_clr_eligible(s)
    assert r.eligible is False
    assert r.reason_code == "h1_m5_against_short"


def test_sell_clr_rejects_bullish_macro():
    s = _snap(side="sell", score=88, align="aligned_sell", macro="bullish", catalyst="material")
    r = sell_clr_eligible(s)
    assert r.eligible is False
    assert r.reason_code == "macro_against_short"


def test_sell_clr_rejects_structure_only_catalyst():
    s = _snap(side="sell", score=88, align="aligned_sell", macro="bearish", catalyst="structure_only")
    r = sell_clr_eligible(s)
    assert r.eligible is False
    assert r.reason_code == "structure_only_catalyst"


# ---------------------------------------------------------------------------
# Momentum Continuation
# ---------------------------------------------------------------------------

def test_momentum_passes_with_1r_room_and_normal_spread():
    s = _snap(side="buy", blocker_pips=10.0, sl=8.0, spread=1.0)
    r = momentum_continuation_eligible(s)
    assert r.eligible is True


def test_momentum_rejects_below_1r_room():
    s = _snap(side="buy", blocker_pips=5.0, sl=8.0)
    r = momentum_continuation_eligible(s)
    assert r.eligible is False
    assert r.reason_code == "insufficient_path_room"


def test_momentum_rejects_missing_blocker_data():
    s = _snap(side="buy", blocker_pips=None, sl=8.0)
    r = momentum_continuation_eligible(s)
    assert r.eligible is False
    assert r.reason_code == "no_path_blocker_data"


def test_momentum_rejects_wrong_side_packet():
    s = _snap(side="buy", pkt_side="sell_resistance", blocker_pips=20.0, sl=8.0)
    r = momentum_continuation_eligible(s)
    assert r.eligible is False
    assert r.reason_code == "wrong_side_packet"


def test_momentum_rejects_wide_spread():
    s = _snap(side="buy", blocker_pips=20.0, sl=8.0, spread=SPREAD_NORMAL_MAX_PIPS + 0.5)
    r = momentum_continuation_eligible(s)
    assert r.eligible is False
    assert r.reason_code == "spread_above_normal"


# ---------------------------------------------------------------------------
# Mean Reversion — sizing-cap carrier
# ---------------------------------------------------------------------------

def test_mean_reversion_carries_one_lot_cap():
    s = _snap(side="sell", structural_origin="exhaustion_at_pdh", spread=1.0, vol="normal")
    r = mean_reversion_eligible(s)
    assert r.eligible is True
    assert r.sizing_cap_lots == MEAN_REVERSION_MAX_LOTS


def test_mean_reversion_rejects_no_exhaustion_marker():
    s = _snap(side="sell", structural_origin="ordinary_swing_high")
    r = mean_reversion_eligible(s)
    assert r.eligible is False
    assert r.reason_code == "no_exhaustion_marker"


def test_mean_reversion_rejects_wrong_side_packet():
    s = _snap(side="sell", pkt_side="buy_support", structural_origin="exhaustion_at_pdh")
    r = mean_reversion_eligible(s)
    assert r.eligible is False
    assert r.reason_code == "wrong_side_packet"


def test_mean_reversion_rejects_elevated_vol():
    s = _snap(side="sell", structural_origin="exhaustion_pdh", vol="elevated")
    r = mean_reversion_eligible(s)
    assert r.eligible is False
    assert r.reason_code == "volatility_elevated"


def test_mean_reversion_accepts_pdh_pdl_extremum_markers():
    """Common structural extremum markers count as exhaustion proxies."""
    for origin in ("pdh", "pdl", "wh", "wl"):
        s = _snap(side="sell", structural_origin=origin)
        r = mean_reversion_eligible(s)
        assert r.eligible is True, origin


# ---------------------------------------------------------------------------
# Non-primary gate kill
# ---------------------------------------------------------------------------

def test_non_primary_gates_killed_by_default():
    for gid in ("post_spike_retracement", "failed_breakout", "trend_expansion"):
        assert is_non_primary_gate_killed(gid)
        assert gid not in PRIMARY_GATES


def test_primary_gate_registry_only_has_four():
    """Buy-CLR, sell-CLR, momentum, mean reversion. No others registered."""
    assert set(PRIMARY_GATES) == {"buy_clr", "sell_clr", "momentum_continuation", "mean_reversion"}


# ---------------------------------------------------------------------------
# Orchestrator: every candidate logged, first eligible selected
# ---------------------------------------------------------------------------

def test_evaluate_all_gates_records_every_candidate():
    s = _snap(side="buy", score=78, blocker_pips=20.0, sl=8.0)
    summary = evaluate_all_gates(s)
    candidate_ids = {c.gate_id for c in summary.candidates}
    assert candidate_ids == set(PRIMARY_GATES)
    assert summary.selected_gate is not None
    assert summary.selected_gate.eligible


def test_evaluate_all_gates_picks_first_eligible_in_preferred_order():
    """If both buy_clr and momentum_continuation pass, preferred_order chooses."""
    s = _snap(side="buy", score=78, blocker_pips=20.0, sl=8.0)
    sum_clr_first = evaluate_all_gates(s, preferred_order=("buy_clr", "momentum_continuation"))
    assert sum_clr_first.selected_gate.gate_id == "buy_clr"
    sum_mom_first = evaluate_all_gates(s, preferred_order=("momentum_continuation", "buy_clr"))
    assert sum_mom_first.selected_gate.gate_id == "momentum_continuation"


def test_evaluate_all_gates_returns_no_selection_when_all_ineligible():
    s = _snap(side="buy", score=50, blocker_pips=2.0, structural_origin="ordinary",
              vol="elevated", spread=5.0)
    summary = evaluate_all_gates(s)
    assert summary.selected_gate is None
    assert summary.any_eligible is False
    # All four candidates still recorded with their reasons
    assert len(summary.candidates) == len(PRIMARY_GATES)


def test_evaluate_all_gates_propagates_pre_veto_block():
    s = _snap(side="sell", score=92, align="aligned_sell", macro="bearish", catalyst="material")
    pv = PreVetoRunSummary(
        skip_before_call=True,
        fires=[PreVetoResult(veto_id="v1_sell_caveat_template", fired=True, reason_code="sell_caveat_template")],
    )
    summary = evaluate_all_gates(s, pre_veto_summary=pv)
    sell_clr_cand = next(c for c in summary.candidates if c.gate_id == "sell_clr")
    assert sell_clr_cand.eligible is False
    assert sell_clr_cand.reason_code == "pre_veto_fired"


def test_evaluate_all_gates_protected_buy_clr_passes_through_v2_bypass():
    """Cardinal preservation: buy-CLR + score >= 70 + V2 bypass active → eligible."""
    s = _snap(side="buy", score=72, align="mixed", overlap="london_ny",
              macro="bullish", catalyst="material", blocker_pips=15.0, sl=8.0)
    pv = run_pre_vetoes(s, deterministic_lots=2.0)
    # V2 should bypass (buy-CLR, score 72 >= 70, lots <= 4)
    assert pv.skip_before_call is False
    summary = evaluate_all_gates(s, pre_veto_summary=pv)
    assert summary.selected_gate is not None
    assert summary.selected_gate.gate_id == "buy_clr"
    assert summary.selected_gate.eligible
