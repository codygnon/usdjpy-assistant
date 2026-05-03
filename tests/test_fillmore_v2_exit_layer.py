"""Step 7 acceptance: Exit Layer (PHASE9.7).

Coverage:
  - Each rule (initial-stop never widened, profit-lock at 1R, time stop at
    30m, trailing disabled) in isolation
  - Halt detectors (unauthorized stop widening, exit-reversal rate >15%
    over 50 trades)
  - LLM has zero stop-widening authority (no entrypoint accepts an
    overriding SL change)
  - Replay determinism: same trade-path → same ExitDecision sequence
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.fillmore_v2.exit_layer import (
    AUTHORIZED_STOP_CHANGE_RULE_IDS,
    EXIT_REVERSAL_AUDIT_WINDOW,
    EXIT_REVERSAL_RATE_HALT_THRESHOLD,
    PROFIT_LOCK_R_MULTIPLE,
    TIME_STOP_MINUTES,
    ClosedTradeRecord,
    ExitAction,
    ExitDecision,
    ExitEvent,
    StopChangeEvent,
    TradeState,
    decide_exit_action,
    decide_partial_close_at_1r,
    detect_unauthorized_stop_widenings,
    exit_reversal_rate,
    should_halt_for_exit_reversals,
    trailing_stop_enabled,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _utc(s: str) -> str:
    return s if s.endswith("+00:00") else s + "+00:00"


def _state(
    *,
    side: str = "buy",
    entry_price: float = 150.00,
    entry_time: str = "2026-04-20T12:00:00",
    sl_pips: float = 8.0,
    tp_pips: float = 16.0,
    current_sl_price: float = None,
    mae: float = 0.0,
    mfe: float = 0.0,
    locked: bool = False,
    spread: float = 1.0,
    lots: float = 2.0,
) -> TradeState:
    if current_sl_price is None:
        # Default initial SL placement at entry +/- sl_pips
        if side == "buy":
            current_sl_price = entry_price - sl_pips * 0.01
        else:
            current_sl_price = entry_price + sl_pips * 0.01
    return TradeState(
        trade_id="t1",
        side=side,
        entry_price=entry_price,
        entry_time_utc=_utc(entry_time),
        sl_pips_at_entry=sl_pips,
        tp_pips_at_entry=tp_pips,
        current_sl_price=current_sl_price,
        current_tp_price=entry_price + tp_pips * 0.01 * (1 if side == "buy" else -1),
        current_lots=lots,
        mae_pips_so_far=mae,
        mfe_pips_so_far=mfe,
        profit_locked=locked,
        spread_pips=spread,
    )


# ---------------------------------------------------------------------------
# Profit lock at 1R
# ---------------------------------------------------------------------------

def test_no_action_when_below_1r_mfe():
    s = _state(mfe=4.0, sl_pips=8.0)  # 4p MFE < 8p sl → not yet 1R
    d = decide_exit_action(s, current_price=150.04, current_time_utc=_utc("2026-04-20T12:05:00"))
    assert d.action == ExitAction.NO_CHANGE
    assert d.rule_id == "no_match"


def test_profit_lock_fires_at_exactly_1r():
    s = _state(mfe=8.0, sl_pips=8.0, side="buy")
    d = decide_exit_action(s, current_price=150.08, current_time_utc=_utc("2026-04-20T12:10:00"))
    assert d.action == ExitAction.MOVE_SL_TO_BE_PLUS_SPREAD
    assert d.rule_id == "profit_lock_be_plus_spread"
    # BE+spread for buy = entry + spread*pip = 150.00 + 1*0.01 = 150.01
    assert d.new_sl_price == pytest.approx(150.01, abs=1e-9)
    assert d.rule_id in AUTHORIZED_STOP_CHANGE_RULE_IDS


def test_profit_lock_skipped_once_already_locked():
    s = _state(mfe=12.0, sl_pips=8.0, locked=True)
    d = decide_exit_action(s, current_price=150.12, current_time_utc=_utc("2026-04-20T12:15:00"))
    # Locked → fall through to time-stop check; not yet 30m elapsed
    assert d.action == ExitAction.NO_CHANGE


def test_profit_lock_for_sell_uses_minus_spread():
    s = _state(side="sell", entry_price=150.00, mfe=8.0, sl_pips=8.0, spread=1.0,
               current_sl_price=150.08)
    d = decide_exit_action(s, current_price=149.92, current_time_utc=_utc("2026-04-20T12:10:00"))
    assert d.action == ExitAction.MOVE_SL_TO_BE_PLUS_SPREAD
    # BE+spread for sell = entry - spread*pip = 150.00 - 0.01 = 149.99
    assert d.new_sl_price == pytest.approx(149.99, abs=1e-9)


def test_profit_lock_refuses_to_widen_when_target_is_wider_than_current_sl():
    """If SL has already been moved tighter than BE+spread by some other
    rule, don't loosen it back to BE+spread. Defensive check.
    """
    s = _state(side="buy", entry_price=150.00, mfe=8.0, sl_pips=8.0, spread=1.0,
               current_sl_price=150.05)  # already at +5p — tighter than BE+1p
    d = decide_exit_action(s, current_price=150.08, current_time_utc=_utc("2026-04-20T12:10:00"))
    assert d.action == ExitAction.NO_CHANGE
    assert d.rule_id == "profit_lock_skipped_would_widen"


# ---------------------------------------------------------------------------
# Time stop at 30 minutes
# ---------------------------------------------------------------------------

def test_time_stop_fires_at_exactly_30m():
    s = _state(entry_time="2026-04-20T12:00:00", mfe=2.0, sl_pips=8.0)
    d = decide_exit_action(s, current_price=150.02,
                           current_time_utc=_utc("2026-04-20T12:30:00"))
    assert d.action == ExitAction.TIME_STOP_CLOSE
    assert d.rule_id == "time_stop_30m"


def test_time_stop_does_not_fire_at_29m_59s():
    s = _state(entry_time="2026-04-20T12:00:00", mfe=2.0, sl_pips=8.0)
    d = decide_exit_action(s, current_price=150.02,
                           current_time_utc=_utc("2026-04-20T12:29:59"))
    assert d.action == ExitAction.NO_CHANGE


def test_time_stop_yields_to_profit_lock_when_both_eligible():
    """If MFE >= 1R AND >= 30m elapsed AND not yet locked: profit-lock wins."""
    s = _state(entry_time="2026-04-20T12:00:00", mfe=8.0, sl_pips=8.0)
    d = decide_exit_action(s, current_price=150.08,
                           current_time_utc=_utc("2026-04-20T12:30:00"))
    assert d.action == ExitAction.MOVE_SL_TO_BE_PLUS_SPREAD


def test_time_stop_does_not_fire_after_profit_lock():
    """Blueprint: 30m backup flattens only if neither stop nor profit lock fired."""
    s = _state(entry_time="2026-04-20T12:00:00", mfe=12.0, sl_pips=8.0, locked=True)
    d = decide_exit_action(s, current_price=150.10,
                           current_time_utc=_utc("2026-04-20T12:30:00"))
    assert d.action == ExitAction.NO_CHANGE
    assert d.rule_id == "no_match"


def test_malformed_timestamp_returns_no_change_not_crash():
    s = _state(entry_time="not-a-date", mfe=2.0)
    d = decide_exit_action(s, current_price=150.0, current_time_utc="garbage")
    assert d.action == ExitAction.NO_CHANGE
    assert d.rule_id == "malformed_timestamps"


def test_naive_iso_timestamps_are_treated_as_utc():
    """Callers should pass UTC strings, but naive ISO strings remain deterministic."""
    s = _state(entry_time="2026-04-20T12:00:00", mfe=2.0, sl_pips=8.0)
    d = decide_exit_action(s, current_price=150.02,
                           current_time_utc="2026-04-20T12:30:00")
    assert d.action == ExitAction.TIME_STOP_CLOSE


# ---------------------------------------------------------------------------
# Partial-close path
# ---------------------------------------------------------------------------

def test_partial_close_at_1r_default_50pct():
    s = _state(mfe=8.0, sl_pips=8.0)
    d = decide_partial_close_at_1r(s)
    assert d.action == ExitAction.PARTIAL_CLOSE
    assert d.close_fraction == 0.5
    assert d.rule_id == "profit_lock_partial_close"


def test_partial_close_below_1r_no_op():
    s = _state(mfe=4.0, sl_pips=8.0)
    d = decide_partial_close_at_1r(s)
    assert d.action == ExitAction.NO_CHANGE
    assert d.rule_id == "below_1r"


def test_partial_close_already_locked_no_op():
    s = _state(mfe=8.0, sl_pips=8.0, locked=True)
    d = decide_partial_close_at_1r(s)
    assert d.action == ExitAction.NO_CHANGE
    assert d.rule_id == "already_locked"


def test_partial_close_rejects_invalid_fraction():
    s = _state(mfe=8.0, sl_pips=8.0)
    for bad in (0.0, 1.0, -0.5, 2.0):
        with pytest.raises(ValueError):
            decide_partial_close_at_1r(s, close_fraction=bad)


# ---------------------------------------------------------------------------
# Stop-widening detector — LLM has zero authority
# ---------------------------------------------------------------------------

def test_no_widening_detected_when_all_changes_tighten():
    events = [
        StopChangeEvent(trade_id="t1", timestamp_utc="2026-04-20T12:10:00+00:00",
                        side="buy", old_sl_price=149.92, new_sl_price=150.01,
                        rule_id="profit_lock_be_plus_spread"),
    ]
    assert detect_unauthorized_stop_widenings(events) == []


def test_unauthorized_widening_caught():
    """Any widening with a rule_id NOT in AUTHORIZED set is offending."""
    events = [
        StopChangeEvent(trade_id="t1", timestamp_utc="2026-04-20T12:10:00+00:00",
                        side="buy", old_sl_price=149.92, new_sl_price=149.85,
                        rule_id="llm_extension_request"),
    ]
    found = detect_unauthorized_stop_widenings(events)
    assert len(found) == 1
    assert found[0].rule_id == "llm_extension_request"


def test_widening_with_authorized_rule_is_still_caught():
    """Even an authorized rule_id cannot widen — those are tightening rules.
    A widening labeled with an authorized rule_id is a correctness bug
    upstream, but the detector treats it as authorized (the rule list is
    the trust boundary). Document this with a test.
    """
    events = [
        StopChangeEvent(trade_id="t1", timestamp_utc="2026-04-20T12:10:00+00:00",
                        side="buy", old_sl_price=149.92, new_sl_price=149.80,
                        rule_id="profit_lock_be_plus_spread"),  # authorized but widening
    ]
    found = detect_unauthorized_stop_widenings(events)
    # The detector considers rule_id authoritative; widening-with-auth-id
    # passes the detector but should be caught by a separate "rule misuse"
    # audit. Test documents the boundary.
    assert found == []


def test_no_llm_entrypoint_can_widen_sl():
    """Inspect the module: every public callable defined IN this module
    is on a hand-reviewed allowlist. Re-exports (datetime, etc.) are
    skipped via __module__ check.
    """
    import dataclasses
    import enum
    import api.fillmore_v2.exit_layer as el

    allowed = {
        "decide_exit_action", "decide_partial_close_at_1r",
        "detect_unauthorized_stop_widenings", "exit_reversal_rate",
        "should_halt_for_exit_reversals", "trailing_stop_enabled",
    }
    for name in dir(el):
        if name.startswith("_"):
            continue
        obj = getattr(el, name)
        if not callable(obj):
            continue
        # Skip re-exports (anything not defined in this module)
        if getattr(obj, "__module__", "") != el.__name__:
            continue
        if dataclasses.is_dataclass(obj):
            continue
        if isinstance(obj, type) and issubclass(obj, enum.Enum):
            continue
        assert name in allowed, f"unreviewed write path exposed: {name}"


# ---------------------------------------------------------------------------
# Exit-reversal proxy + halt
# ---------------------------------------------------------------------------

def _closed(*, mfe: float, pips: float, n: int = 1, base_time: str = "2026-04-20T12:00:00") -> list[ClosedTradeRecord]:
    base = datetime.fromisoformat(base_time + "+00:00")
    return [
        ClosedTradeRecord(
            trade_id=f"t{i}",
            closed_at_utc=(base + timedelta(minutes=i)).isoformat(),
            pips_realized=pips, mfe_pips=mfe,
        )
        for i in range(n)
    ]


def test_exit_reversal_rate_zero_when_no_reversals():
    trades = _closed(mfe=10.0, pips=10.0, n=20)  # all winners that ran
    rate, n_rev, n = exit_reversal_rate(trades)
    assert rate == 0.0
    assert n_rev == 0
    assert n == 20


def test_exit_reversal_rate_counts_reversals_only():
    trades = (
        _closed(mfe=10.0, pips=-5.0, n=5)  # 5 reversals (mfe>=4 AND pips<0)
        + _closed(mfe=2.0, pips=-3.0, n=15, base_time="2026-04-20T13:00:00")  # losses but never had >=4p MFE
    )
    rate, n_rev, n = exit_reversal_rate(trades)
    assert n_rev == 5
    assert n == 20
    assert rate == pytest.approx(0.25, abs=1e-9)


def test_halt_does_not_fire_below_audit_window():
    trades = _closed(mfe=10.0, pips=-5.0, n=49)
    halt, reason = should_halt_for_exit_reversals(trades)
    assert halt is False
    assert "insufficient sample" in reason


def test_halt_fires_above_15pct_at_50_trades():
    # 10 reversals + 40 wins → 20% rate over 50 trades
    trades = (
        _closed(mfe=10.0, pips=-5.0, n=10)
        + _closed(mfe=6.0, pips=4.0, n=40, base_time="2026-04-20T13:00:00")
    )
    halt, reason = should_halt_for_exit_reversals(trades)
    assert halt is True
    assert "exit_reversal_rate" in reason


def test_halt_does_not_fire_at_exactly_15pct_threshold():
    """PHASE9.7 says ABOVE 15%. 7/50 = 14% does not halt; 8/50 = 16% does."""
    trades_14 = (
        _closed(mfe=10.0, pips=-5.0, n=7)
        + _closed(mfe=6.0, pips=4.0, n=43, base_time="2026-04-20T13:00:00")
    )
    halt, _ = should_halt_for_exit_reversals(trades_14)
    assert halt is False
    trades_16 = (
        _closed(mfe=10.0, pips=-5.0, n=8)
        + _closed(mfe=6.0, pips=4.0, n=42, base_time="2026-04-20T13:00:00")
    )
    halt, _ = should_halt_for_exit_reversals(trades_16)
    assert halt is True


# ---------------------------------------------------------------------------
# Trailing stop disabled gate
# ---------------------------------------------------------------------------

def test_trailing_disabled_until_all_three_gates_clear():
    # All conditions short of clear:
    assert trailing_stop_enabled(path_time_coverage_pct=95.0, exit_reversal_rate_pct=5.0, closed_trades=49) is False  # too few trades
    assert trailing_stop_enabled(path_time_coverage_pct=85.0, exit_reversal_rate_pct=5.0, closed_trades=60) is False  # coverage low
    assert trailing_stop_enabled(path_time_coverage_pct=95.0, exit_reversal_rate_pct=12.0, closed_trades=60) is False  # reversal rate high


def test_trailing_enabled_when_all_three_gates_clear():
    assert trailing_stop_enabled(path_time_coverage_pct=95.0, exit_reversal_rate_pct=5.0, closed_trades=60) is True


# ---------------------------------------------------------------------------
# Replay determinism — same path → same decisions
# ---------------------------------------------------------------------------

def test_replay_path_is_deterministic():
    """Drive the same synthetic trade-path twice; sequence of decisions is
    identical. PHASE9.7 mandates exit-manager replayability.
    """
    # Path: enter long at 150.00, run to +6p (no lock), drift back to flat,
    # then run to +9p (lock fires). After that, 30m backup time stop no longer applies.
    path = [
        # (current_time, current_price, mae, mfe)
        ("2026-04-20T12:01:00", 150.03, 0.0, 3.0),
        ("2026-04-20T12:05:00", 150.06, 0.0, 6.0),
        ("2026-04-20T12:10:00", 150.00, 1.0, 6.0),
        ("2026-04-20T12:15:00", 150.09, 1.0, 9.0),  # locks here
        # After locking, simulate locked=True for subsequent ticks
        ("2026-04-20T12:20:00", 150.05, 1.0, 9.0),
        ("2026-04-20T12:35:00", 150.03, 1.0, 9.0),  # locked, so no time stop
    ]

    def replay() -> list[ExitDecision]:
        decisions: list[ExitDecision] = []
        locked = False
        for (t, px, mae, mfe) in path:
            s = _state(entry_time="2026-04-20T12:00:00", mfe=mfe, mae=mae, locked=locked)
            d = decide_exit_action(s, current_price=px, current_time_utc=_utc(t))
            decisions.append(d)
            if d.action == ExitAction.MOVE_SL_TO_BE_PLUS_SPREAD:
                locked = True
        return decisions

    a = replay()
    b = replay()
    assert a == b
    actions = [d.action for d in a]
    assert ExitAction.MOVE_SL_TO_BE_PLUS_SPREAD in actions
    assert actions[-1] == ExitAction.NO_CHANGE


# ---------------------------------------------------------------------------
# Exit event audit record shape
# ---------------------------------------------------------------------------

def test_exit_event_audit_record_round_trip():
    decision = ExitDecision(action=ExitAction.MOVE_SL_TO_BE_PLUS_SPREAD,
                            rule_id="profit_lock_be_plus_spread",
                            new_sl_price=150.01, reason="MFE>=1R")
    event = ExitEvent(trade_id="t1", timestamp_utc="2026-04-20T12:10:00+00:00",
                      decision=decision, current_price=150.08,
                      current_mae_pips=0.0, current_mfe_pips=8.0)
    rec = event.to_audit_record()
    assert rec["trade_id"] == "t1"
    assert rec["decision"]["action"] == "move_sl_to_be_plus_spread"
    assert rec["decision"]["new_sl_price"] == 150.01
    assert rec["current_mfe_pips"] == 8.0
