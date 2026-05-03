"""Step 9 acceptance: Staged rollout wiring + engine flag + tripwires."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api import suggestion_tracker
from api.autonomous_fillmore import (
    _v2_oanda_practice_place_guard,
    _v2_result_to_broker_suggestion,
)
from api.fillmore_v2 import persistence
from api.fillmore_v2.engine_flag import (
    DEFAULT_ENGINE,
    VALID_ENGINES,
    read_engine_flag,
    set_engine_flag,
)
from api.fillmore_v2.llm_client import FakeLlmClient
from api.fillmore_v2.stage_progression import (
    STAGES,
    STAGE_ORDER,
    StageObservations,
    evaluate_stage,
)
from api.fillmore_v2.tripwires import (
    T1_KILL_WR,
    T1_REQUIRED_WR,
    T1_MIN_SELL_TRADES,
    T2_MIN_SELL_CLR_CLOSES,
    T3_MIN_LLM_CALLS,
    T3_REPLAY_EXPECTATION_FIRE_RATE,
    T4_HARD_LOTS_CEILING,
    T5_MIN_SKIPS_FOR_AUDIT,
    T5_REQUIRED_OUTCOME_COVERAGE,
    check_caveat_validator_fire_rate,
    check_sell_clr_kill,
    check_sell_side_wr,
    check_sizing_ceiling,
    check_skip_outcome_coverage,
    evaluate_all,
)
from api.fillmore_v2.v1_bridge import build_snapshot_from_v1_inputs, dispatch_v2_tick


# ===========================================================================
# Engine flag
# ===========================================================================

def test_engine_flag_defaults_to_v1_when_state_missing(tmp_path):
    assert read_engine_flag(tmp_path / "missing.json") == "v1"


def test_engine_flag_defaults_to_v1_when_state_corrupt(tmp_path):
    p = tmp_path / "rs.json"
    p.write_text("{not valid json")
    assert read_engine_flag(p) == "v1"


def test_engine_flag_defaults_to_v1_on_unknown_value(tmp_path):
    p = tmp_path / "rs.json"
    p.write_text(json.dumps({"autonomous_fillmore": {"engine": "v999"}}))
    assert read_engine_flag(p) == "v1"


def test_engine_flag_round_trip(tmp_path):
    p = tmp_path / "rs.json"
    set_engine_flag(p, "v2")
    assert read_engine_flag(p) == "v2"
    set_engine_flag(p, "v1")
    assert read_engine_flag(p) == "v1"


def test_engine_flag_preserves_existing_state(tmp_path):
    p = tmp_path / "rs.json"
    p.write_text(json.dumps({"other_section": {"foo": "bar"}, "autonomous_fillmore": {"enabled": True}}))
    set_engine_flag(p, "v2")
    after = json.loads(p.read_text())
    assert after["other_section"]["foo"] == "bar"
    assert after["autonomous_fillmore"]["enabled"] is True
    assert after["autonomous_fillmore"]["engine"] == "v2"


def test_engine_flag_set_rejects_invalid(tmp_path):
    with pytest.raises(ValueError):
        set_engine_flag(tmp_path / "rs.json", "v3")  # type: ignore[arg-type]


def test_engine_flag_default_is_v1():
    """Cardinal safety: default engine MUST be v1 until operator opts in."""
    assert DEFAULT_ENGINE == "v1"
    assert "v1" in VALID_ENGINES and "v2" in VALID_ENGINES


# ===========================================================================
# Tripwires
# ===========================================================================

def test_t1_sell_wr_green_when_sample_too_small():
    r = check_sell_side_wr(sell_wins=10, sell_losses=10)
    assert r.status == "green"
    assert "insufficient sample" in r.detail


def test_t1_sell_wr_red_below_kill_threshold():
    r = check_sell_side_wr(sell_wins=8, sell_losses=22)
    assert r.status == "red"


def test_t1_sell_wr_amber_between_kill_and_target():
    r = check_sell_side_wr(sell_wins=12, sell_losses=18)  # 40%
    assert r.status == "amber"


def test_t1_sell_wr_green_above_target():
    r = check_sell_side_wr(sell_wins=15, sell_losses=15)  # 50%
    assert r.status == "green"


def test_t2_sell_clr_red_on_negative_net_pips():
    closes = [-5.0] * 16 + [4.0] * 14  # 30 closes, net -16p
    r = check_sell_clr_kill(sell_clr_closes=closes)
    assert r.status == "red"


def test_t2_sell_clr_red_on_low_wr_even_if_net_positive():
    """30 closes, 8 wins of 10p, 22 losses of 1p → net positive but WR=27%."""
    closes = [10.0] * 8 + [-1.0] * 22
    r = check_sell_clr_kill(sell_clr_closes=closes)
    assert r.status == "red"
    assert "WR" in r.detail


def test_t2_sell_clr_green_with_acceptable_record():
    closes = [4.0] * 16 + [-3.0] * 14  # WR 53%, net positive
    r = check_sell_clr_kill(sell_clr_closes=closes)
    assert r.status == "green"


def test_t3_caveat_fire_rate_red_outside_band():
    # Expectation 0.617 → band 0.308–0.926. Rate 0.20 is below.
    r = check_caveat_validator_fire_rate(llm_calls=100, caveat_validator_fires=20)
    assert r.status == "red"


def test_t3_caveat_fire_rate_green_inside_band():
    r = check_caveat_validator_fire_rate(llm_calls=100, caveat_validator_fires=60)
    assert r.status == "green"


def test_t4_sizing_ceiling_red_when_breached():
    r = check_sizing_ceiling(max_lots_seen=4.5)
    assert r.status == "red"


def test_t4_sizing_ceiling_green_at_exactly_4():
    r = check_sizing_ceiling(max_lots_seen=4.0)
    assert r.status == "green"


def test_t5_skip_coverage_red_below_98pct():
    r = check_skip_outcome_coverage(total_skips=100, skips_with_outcomes=95)
    assert r.status == "red"


def test_t5_skip_coverage_green_at_or_above_98pct():
    r = check_skip_outcome_coverage(total_skips=100, skips_with_outcomes=98)
    assert r.status == "green"


def test_evaluate_all_aggregates_and_reports_red_ids():
    snap = evaluate_all(
        sell_wins=8, sell_losses=22,  # T1 red
        sell_clr_closes=[4.0] * 30,    # T2 green
        llm_calls=100, caveat_validator_fires=60,  # T3 green
        max_lots_seen=4.0,             # T4 green
        total_skips=100, skips_with_outcomes=99,   # T5 green
    )
    assert snap.any_red is True
    assert "T1_sell_wr" in snap.red_ids()


# ===========================================================================
# Stage progression
# ===========================================================================

def _obs(**overrides):
    base = dict(
        stage="paper",
        closes_in_stage=0, net_pips_in_stage=0.0, profit_factor_in_stage=1.0,
        sell_wins=0, sell_losses=0, sell_breakevens=0,
        sell_clr_closes=[],
        llm_calls=0, caveat_validator_fires=0,
        max_lots_seen=2.0,
        total_skips=0, skips_with_outcomes=0,
        cumulative_drawdown_pips=0.0,
        any_phase8_primary_failure_reappeared=False,
    )
    base.update(overrides)
    return StageObservations(**base)


def test_stage_holds_when_below_advance_threshold():
    v = evaluate_stage(_obs(closes_in_stage=10, net_pips_in_stage=20.0, profit_factor_in_stage=1.5))
    assert v.action == "hold"
    assert "more closes" in v.reason


def test_stage_advances_when_criteria_met_and_no_red():
    v = evaluate_stage(_obs(closes_in_stage=50, net_pips_in_stage=20.0, profit_factor_in_stage=1.0))
    assert v.action == "advance"
    assert v.next_stage == "0.1x"


def test_stage_paper_kills_at_minus_150p():
    v = evaluate_stage(_obs(closes_in_stage=20, cumulative_drawdown_pips=-160.0))
    assert v.action == "kill"
    # paper has no previous stage; kill stays at paper effectively
    # (next_stage == None for paper)
    assert v.reason.startswith("drawdown")


def test_stage_light_kills_at_minus_100p_from_start():
    v = evaluate_stage(_obs(stage="0.1x", cumulative_drawdown_pips=-110.0))
    assert v.action == "kill"
    assert v.next_stage == "paper"


def test_stage_full_kills_when_phase8_failure_reappears():
    v = evaluate_stage(_obs(stage="full", any_phase8_primary_failure_reappeared=True))
    assert v.action == "kill"
    assert v.next_stage == "0.5x"


def test_stage_kills_on_red_sell_tripwire():
    v = evaluate_stage(_obs(
        stage="0.1x", closes_in_stage=40, sell_wins=8, sell_losses=22,
    ))
    assert v.action == "kill"
    assert "T1_sell_wr" in (v.tripwire_alerts or [])


def test_stage_holds_when_red_non_fatal_tripwire_present():
    """T4 sizing-ceiling breach is red but not in the kill-tripwire set —
    advance is denied, stage holds.
    """
    v = evaluate_stage(_obs(
        closes_in_stage=50, net_pips_in_stage=20.0, profit_factor_in_stage=1.0,
        max_lots_seen=4.5,  # T4 red
    ))
    # The kill-tripwire set is {T1, T2}; T4 doesn't kill but blocks advance.
    assert v.action == "hold"
    assert "T4_sizing_ceiling" in (v.tripwire_alerts or [])


def test_stage_full_does_not_advance():
    """Last stage has nowhere to advance; verdict 'hold' even if criteria met."""
    v = evaluate_stage(_obs(
        stage="full", closes_in_stage=300, net_pips_in_stage=100.0, profit_factor_in_stage=1.5,
    ))
    assert v.action == "hold"


# ===========================================================================
# v1→v2 dispatch bridge
# ===========================================================================

class _FakeProfile:
    pip_size = 0.01
    symbol = "USD_JPY"


class _FakeTick:
    def __init__(self, bid: float, ask: float):
        self.bid = bid
        self.ask = ask


class _FakeAccount:
    equity = 100_000.0


class _FakeAdapter:
    def __init__(self, *, positions=None):
        self.positions = list(positions or [])

    def get_account_info(self):
        return _FakeAccount()

    def get_open_positions(self, symbol):
        return self.positions


def _stage1_bars():
    rows = []
    base = 149.80
    for i in range(20):
        close = base + i * 0.01
        rows.append({
            "time": f"2026-05-02T00:{i:02d}:00+00:00",
            "open": close - 0.01,
            "high": close + 0.08,
            "low": close - 0.06,
            "close": close,
        })
    return {"M1": rows, "M5": rows, "H1": rows}


def test_build_snapshot_from_v1_inputs_basic_fields():
    p = _FakeProfile()
    t = _FakeTick(bid=149.99, ask=150.01)
    snap = build_snapshot_from_v1_inputs(profile=p, tick=t, proposed_side="buy", sl_pips=8.0, tp_pips=16.0)
    assert snap.tick_bid == 149.99
    assert snap.tick_ask == 150.01
    assert snap.tick_mid == 150.00
    assert snap.spread_pips == pytest.approx(2.0, abs=0.01)
    assert snap.proposed_side == "buy"
    # Adapter-owned blocking telemetry is absent; tick-owned pip value is ready.
    assert snap.account_equity is None
    assert snap.pip_value_per_lot == pytest.approx(6.6667, abs=0.0001)


def test_build_snapshot_from_v1_inputs_populates_stage1_telemetry(tmp_path):
    db = tmp_path / "ai_suggestions.sqlite"
    suggestion_tracker.init_db(db)
    persistence.init_v2_schema(db)
    adapter = _FakeAdapter(positions=[
        {"currentUnits": "100000", "unrealizedPL": "12.50"},
        {"currentUnits": "-200000", "unrealizedPL": "-7.25"},
    ])
    snap = build_snapshot_from_v1_inputs(
        profile=_FakeProfile(),
        profile_name="test",
        tick=_FakeTick(bid=149.99, ask=150.01),
        adapter=adapter,
        db_path=db,
        data_by_tf=_stage1_bars(),
    )
    assert snap.account_equity == 100_000.0
    assert snap.open_lots_buy == 1.0
    assert snap.open_lots_sell == 2.0
    assert snap.unrealized_pnl_buy == 12.50
    assert snap.unrealized_pnl_sell == -7.25
    assert snap.rolling_20_trade_pnl == 0.0
    assert snap.rolling_20_lot_weighted_pnl == 0.0
    assert snap.proposed_side == "buy"
    assert snap.sl_pips is not None and snap.tp_pips is not None
    assert snap.level_packet is not None
    assert snap.level_age_metadata is not None
    assert snap.timeframe_alignment == "aligned_buy"
    assert snap.active_sessions is not None


def test_dispatch_v2_tick_halts_within_3_ticks_when_blocking_telemetry_missing(tmp_path):
    """Cardinal safety: v2 dispatch should halt after 3 incomplete polls
    rather than place trades on missing telemetry. This documents that the
    minimal v1→v2 bridge is intentionally a halt-driver until the full
    telemetry wiring lands.
    """
    db = tmp_path / "ai_suggestions.sqlite"
    suggestion_tracker.init_db(db)
    persistence.init_v2_schema(db)
    state_path = tmp_path / "runtime_state.json"
    p = _FakeProfile()
    t = _FakeTick(bid=149.99, ask=150.01)
    fake = FakeLlmClient()
    results = []
    for _ in range(3):
        r = dispatch_v2_tick(
            profile=p, profile_name="test", state_path=state_path, tick=t,
            db_path=db, llm_client=fake,
        )
        results.append(r)
    assert results[0].final_decision == "skip"
    assert results[1].final_decision == "skip"
    assert results[2].final_decision == "halt"
    assert fake.calls == []  # LLM never invoked


def test_dispatch_v2_tick_with_stage1_telemetry_does_not_accumulate_strikes(tmp_path):
    db = tmp_path / "ai_suggestions.sqlite"
    state_path = tmp_path / "runtime_state.json"
    fake = FakeLlmClient()
    adapter = _FakeAdapter()
    results = []
    for _ in range(10):
        results.append(dispatch_v2_tick(
            profile=_FakeProfile(),
            profile_name="test",
            state_path=state_path,
            tick=_FakeTick(bid=149.99, ask=150.01),
            adapter=adapter,
            db_path=db,
            data_by_tf=_stage1_bars(),
            llm_client=fake,
        ))
    assert all(r.final_decision != "halt" for r in results)


def test_dispatch_v2_tick_blocks_non_paper_stage_by_default(tmp_path):
    db = tmp_path / "ai_suggestions.sqlite"
    res = dispatch_v2_tick(
        profile=_FakeProfile(),
        profile_name="test",
        state_path=tmp_path / "runtime_state.json",
        tick=_FakeTick(bid=149.99, ask=150.01),
        adapter=_FakeAdapter(),
        db_path=db,
        data_by_tf=_stage1_bars(),
        llm_client=FakeLlmClient(),
        stage="0.1x",
    )
    assert res.final_decision == "halt"
    assert "paper_mode_guard" in res.reason


def test_v2_practice_place_guard_allows_only_oanda_practice_paper_mode():
    p = SimpleNamespace(broker_type="oanda", oanda_environment="practice")
    assert _v2_oanda_practice_place_guard(p, "paper") == (True, "ok")

    ok, reason = _v2_oanda_practice_place_guard(p, "shadow")
    assert not ok and "mode" in reason

    ok, reason = _v2_oanda_practice_place_guard(
        SimpleNamespace(broker_type="oanda", oanda_environment="live"),
        "paper",
    )
    assert not ok and "practice" in reason

    ok, reason = _v2_oanda_practice_place_guard(
        SimpleNamespace(broker_type="mt5", oanda_environment="practice"),
        "paper",
    )
    assert not ok and "OANDA practice" in reason


def test_v2_result_to_broker_suggestion_uses_deterministic_lots_and_prices():
    snap = SimpleNamespace(
        proposed_side="buy",
        tick_mid=150.0,
        sl_pips=8.0,
        tp_pips=16.0,
        selected_gate_id="momentum_continuation",
        risk_after_fill_usd=32.0,
        timeframe_alignment="aligned_buy",
        macro_bias="neutral",
        catalyst_category="material",
    )
    out = SimpleNamespace(
        primary_thesis="good buy",
        caveat_resolution=None,
        loss_asymmetry_argument="tp bigger than sl",
        level_quality_claim={"claim": "ok"},
        invalid_if=["breaks support"],
        evidence_refs=["level_packet"],
    )
    result = SimpleNamespace(
        suggestion_id="v2-test",
        snapshot=snap,
        llm_output=out,
        sizing=SimpleNamespace(lots=2.0),
        deterministic_lots=2.0,
        to_audit_records=lambda: {"final_decision": "place"},
    )
    suggestion = _v2_result_to_broker_suggestion(result, SimpleNamespace(pip_size=0.01))
    assert suggestion["engine_version"] == "v2"
    assert suggestion["suggestion_id"] == "v2-test"
    assert suggestion["order_type"] == "market"
    assert suggestion["side"] == "buy"
    assert suggestion["lots"] == 2.0
    assert suggestion["price"] == 150.0
    assert suggestion["sl"] == pytest.approx(149.92)
    assert suggestion["tp"] == pytest.approx(150.16)
    assert suggestion["risk_after_fill_usd"] == 32.0


# ===========================================================================
# v1 hook isolation: when flag is v1, v2 modules are not invoked
# ===========================================================================

def test_v1_autonomous_fillmore_does_not_invoke_v2_when_flag_unset(tmp_path, monkeypatch):
    """Read the v1 source: it must default-skip the v2 path when the engine
    flag is absent (the test file itself can't easily run tick_autonomous_fillmore
    so this asserts the source contract).
    """
    src = (REPO_ROOT / "api" / "autonomous_fillmore.py").read_text()
    # The new wiring block must be present
    assert "read_engine_flag" in src
    assert "from api.fillmore_v2.v1_bridge import dispatch_v2_tick" in src
    # The wiring is gated on flag == 'v2' — v1 path is the default
    assert 'read_engine_flag(state_path) == "v2"' in src
    # The wiring is wrapped in a try/except so v2 import errors can't crash v1.
    # Search a window AROUND the read_engine_flag call (try/import precedes it).
    idx = src.find("read_engine_flag")
    block = src[max(0, idx - 200):idx + 2200]
    assert "try:" in block
    assert "except" in block
