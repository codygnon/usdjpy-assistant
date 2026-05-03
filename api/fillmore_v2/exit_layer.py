"""Exit Layer — Phase 9 Step 7 (PHASE9.7).

Deterministic exit management. The LLM has zero authority to widen stops
or extend exits. Bounded commentary is allowed elsewhere; this module only
acts on rules.

Rules (PHASE9.7):

  1. Initial stop: deterministic at entry from sl_pips, never widened.
  2. Profit lock: when MFE reaches 1R, move SL to break-even + spread
     (or partial close, depending on broker mechanics — BE-move is the
     default; partial-close is callable separately).
  3. Time stop: flatten at 30 minutes if neither stop nor profit lock
     has triggered.
  4. Trailing stop: DISABLED until path-time MAE/MFE telemetry is live
     for ≥ 50 trades and exit-reversal proxy < 10% (per rollout doc).
  5. LLM override: NONE for stop widening; bounded only for skip/close
     commentary after deterministic rules.

Halt rules:

  - Any stop widened after entry without a deterministic rule id → halt
    autonomous exits.
  - Exit-reversal proxy rate above 15% over first 50 closed trades →
    reduce to stop/TP-only exits and audit.
  - Profit-lock malfunction or missing path event logs > 2 trades in one
    day → halt new autonomous entries.

Replay treatment (PHASE9.7): no net pip recovery counted from exit
redesign in the Phase 9 pass/fail replay. The audit lacks path-time
ordering and exit-manager replay. This module is conservative and
telemetry-first: it prevents the proven bad runner clause from returning,
but does not claim untested pips.

Replay log: every exit decision (no_change too) emits an `ExitEvent` so
the next forensic audit can fully reconstruct the exit-management loop.
PHASE9.8 item 12.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Iterable, Literal, Optional


PIP_SIZE_USDJPY = 0.01

# PHASE9.7 thresholds
PROFIT_LOCK_R_MULTIPLE = 1.0  # lock at 1R MFE
TIME_STOP_MINUTES = 30
EXIT_REVERSAL_MFE_THRESHOLD_PIPS = 4.0  # Phase 6 definition
EXIT_REVERSAL_RATE_HALT_THRESHOLD = 0.15  # 15% over first 50 trades
EXIT_REVERSAL_AUDIT_WINDOW = 50

# Deterministic rule ids that authorize a stop change. Anything else is a
# halt-worthy unauthorized widening.
AUTHORIZED_STOP_CHANGE_RULE_IDS = frozenset({
    "profit_lock_be_plus_spread",
    "profit_lock_partial_close",
    # Initial-set rule for the first SL placement at order-create time.
    "initial_set_at_entry",
})


# --- Decision types ----------------------------------------------------------

class ExitAction(str, Enum):
    NO_CHANGE = "no_change"
    MOVE_SL_TO_BE_PLUS_SPREAD = "move_sl_to_be_plus_spread"
    PARTIAL_CLOSE = "partial_close"
    TIME_STOP_CLOSE = "time_stop_close"


@dataclass(frozen=True)
class TradeState:
    """Read-only state of an open trade. Constructed fresh each tick."""
    trade_id: str
    side: Literal["buy", "sell"]
    entry_price: float
    entry_time_utc: str  # ISO-8601
    sl_pips_at_entry: float
    tp_pips_at_entry: float
    current_sl_price: float
    current_tp_price: float
    current_lots: float
    mae_pips_so_far: float  # max adverse excursion seen (positive number)
    mfe_pips_so_far: float  # max favorable excursion seen (positive number)
    profit_locked: bool  # True once profit_lock has fired
    spread_pips: float  # current spread for BE-move calculation


@dataclass
class ExitDecision:
    action: ExitAction
    rule_id: str  # which deterministic rule fired (or "no_match")
    new_sl_price: Optional[float] = None
    close_fraction: Optional[float] = None  # 0.0-1.0, for partial_close
    reason: str = ""

    def to_audit_record(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "rule_id": self.rule_id,
            "new_sl_price": self.new_sl_price,
            "close_fraction": self.close_fraction,
            "reason": self.reason,
        }


@dataclass
class ExitEvent:
    """One row in the exit-manager replay stream. PHASE9.8 item 12."""
    trade_id: str
    timestamp_utc: str
    decision: ExitDecision
    current_price: float
    current_mae_pips: float
    current_mfe_pips: float

    def to_audit_record(self) -> dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "timestamp_utc": self.timestamp_utc,
            "decision": self.decision.to_audit_record(),
            "current_price": self.current_price,
            "current_mae_pips": self.current_mae_pips,
            "current_mfe_pips": self.current_mfe_pips,
        }


# --- Helpers ----------------------------------------------------------------

def _parse_iso_utc(s: str) -> datetime:
    dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _be_plus_spread_price(state: TradeState) -> float:
    """Break-even + spread, in the trade's favorable direction."""
    if state.side == "buy":
        return state.entry_price + state.spread_pips * PIP_SIZE_USDJPY
    return state.entry_price - state.spread_pips * PIP_SIZE_USDJPY


def _is_widening(*, side: str, old_sl: float, new_sl: float) -> bool:
    """A SL change is widening if the new SL is FURTHER from entry direction
    than the old SL (i.e., risks more loss).

    For buy: lower new_sl = wider. For sell: higher new_sl = wider.
    """
    if side == "buy":
        return new_sl < old_sl
    return new_sl > old_sl


# --- Decision function -------------------------------------------------------

def decide_exit_action(
    state: TradeState,
    *,
    current_price: float,
    current_time_utc: str,
) -> ExitDecision:
    """Pure function. Decides one of NO_CHANGE / MOVE_SL_TO_BE_PLUS_SPREAD /
    TIME_STOP_CLOSE based on rules 1-3 (PHASE9.7).

    PARTIAL_CLOSE is callable via `decide_partial_close_at_1r` instead — the
    blueprint allows BE-move OR partial as profit-lock implementations; the
    caller picks per broker mechanics. This decision function defaults to
    BE-move because it's the universal mechanic.
    """
    # Rule 2: profit lock at 1R MFE (only fires once)
    if not state.profit_locked and state.mfe_pips_so_far >= PROFIT_LOCK_R_MULTIPLE * state.sl_pips_at_entry:
        new_sl = _be_plus_spread_price(state)
        # Defensive: BE+spread should be a TIGHTENING relative to current SL
        if _is_widening(side=state.side, old_sl=state.current_sl_price, new_sl=new_sl):
            # Already tighter than BE+spread? Then don't loosen — no_change.
            return ExitDecision(
                action=ExitAction.NO_CHANGE,
                rule_id="profit_lock_skipped_would_widen",
                reason=(
                    f"profit-lock target {new_sl:.5f} is wider than current SL "
                    f"{state.current_sl_price:.5f}; refusing to widen"
                ),
            )
        return ExitDecision(
            action=ExitAction.MOVE_SL_TO_BE_PLUS_SPREAD,
            rule_id="profit_lock_be_plus_spread",
            new_sl_price=new_sl,
            reason=f"MFE={state.mfe_pips_so_far:.1f}p >= 1R={state.sl_pips_at_entry:.1f}p",
        )

    # Rule 3: time stop at 30 minutes
    try:
        entry_dt = _parse_iso_utc(state.entry_time_utc)
        now_dt = _parse_iso_utc(current_time_utc)
    except ValueError:
        # Malformed timestamps → don't act; let the next tick try
        return ExitDecision(
            action=ExitAction.NO_CHANGE, rule_id="malformed_timestamps",
            reason="cannot parse entry_time or current_time",
        )
    elapsed = now_dt - entry_dt
    if not state.profit_locked and elapsed >= timedelta(minutes=TIME_STOP_MINUTES):
        return ExitDecision(
            action=ExitAction.TIME_STOP_CLOSE,
            rule_id="time_stop_30m",
            reason=f"elapsed {elapsed} >= {TIME_STOP_MINUTES}m and no SL/profit-lock fired",
        )

    return ExitDecision(action=ExitAction.NO_CHANGE, rule_id="no_match")


def decide_partial_close_at_1r(
    state: TradeState,
    *,
    close_fraction: float = 0.5,
) -> ExitDecision:
    """Alternative profit-lock path: close a fraction of the position at 1R.

    Callers using broker mechanics that prefer scale-out over BE-move use
    this in place of `decide_exit_action`'s BE-move output. The fraction is
    bounded to (0, 1).
    """
    if not (0.0 < close_fraction < 1.0):
        raise ValueError("close_fraction must be in (0, 1)")
    if state.profit_locked:
        return ExitDecision(action=ExitAction.NO_CHANGE, rule_id="already_locked")
    if state.mfe_pips_so_far < PROFIT_LOCK_R_MULTIPLE * state.sl_pips_at_entry:
        return ExitDecision(action=ExitAction.NO_CHANGE, rule_id="below_1r")
    return ExitDecision(
        action=ExitAction.PARTIAL_CLOSE,
        rule_id="profit_lock_partial_close",
        close_fraction=close_fraction,
        reason=f"MFE={state.mfe_pips_so_far:.1f}p >= 1R={state.sl_pips_at_entry:.1f}p; close {close_fraction:.0%}",
    )


# --- Halt detectors ---------------------------------------------------------

@dataclass(frozen=True)
class StopChangeEvent:
    trade_id: str
    timestamp_utc: str
    side: Literal["buy", "sell"]
    old_sl_price: float
    new_sl_price: float
    rule_id: str  # must be in AUTHORIZED_STOP_CHANGE_RULE_IDS for legal widening


def detect_unauthorized_stop_widenings(events: Iterable[StopChangeEvent]) -> list[StopChangeEvent]:
    """Per PHASE9.7: any stop widened without a deterministic rule id is a
    halt-worthy event. Returns the offending events in order seen.
    """
    offenders: list[StopChangeEvent] = []
    for e in events:
        if not _is_widening(side=e.side, old_sl=e.old_sl_price, new_sl=e.new_sl_price):
            continue  # tightening is always allowed
        if e.rule_id not in AUTHORIZED_STOP_CHANGE_RULE_IDS:
            offenders.append(e)
    return offenders


@dataclass(frozen=True)
class ClosedTradeRecord:
    trade_id: str
    closed_at_utc: str
    pips_realized: float
    mfe_pips: float


def exit_reversal_rate(
    closed_trades: list[ClosedTradeRecord],
    *,
    window: int = EXIT_REVERSAL_AUDIT_WINDOW,
) -> tuple[float, int, int]:
    """Returns (rate, reversal_count, sample_size).

    Exit-reversal proxy (Phase 6): trade closed red after reaching ≥ 4p MFE.
    """
    if not closed_trades:
        return 0.0, 0, 0
    tail = sorted(closed_trades, key=lambda t: t.closed_at_utc)[-window:]
    reversals = sum(
        1 for t in tail
        if t.mfe_pips >= EXIT_REVERSAL_MFE_THRESHOLD_PIPS and t.pips_realized < 0
    )
    return reversals / len(tail), reversals, len(tail)


def should_halt_for_exit_reversals(closed_trades: list[ClosedTradeRecord]) -> tuple[bool, str]:
    """PHASE9.7: rate > 15% over first 50 trades → reduce to stop/TP-only.

    Returns (halt, reason). Halt fires once the sample reaches the audit
    window AND the rate exceeds threshold.
    """
    rate, reversals, n = exit_reversal_rate(closed_trades)
    if n < EXIT_REVERSAL_AUDIT_WINDOW:
        return False, f"insufficient sample: {n}/{EXIT_REVERSAL_AUDIT_WINDOW}"
    if rate > EXIT_REVERSAL_RATE_HALT_THRESHOLD:
        return True, (
            f"exit_reversal_rate={rate:.3f} > {EXIT_REVERSAL_RATE_HALT_THRESHOLD} "
            f"({reversals}/{n} closes); reduce to stop/TP-only and audit"
        )
    return False, f"exit_reversal_rate={rate:.3f} within tolerance"


# --- Trailing stop is intentionally absent ----------------------------------

def trailing_stop_enabled(*, path_time_coverage_pct: float, exit_reversal_rate_pct: float, closed_trades: int) -> bool:
    """Trailing stops are DISABLED until:

      - ≥ 50 closed trades
      - path-time telemetry coverage ≥ 90%
      - exit-reversal proxy rate < 10%

    Per PHASE9.7 + Step 1 rollout doc. Returns False unless all met.
    """
    return (
        closed_trades >= 50
        and path_time_coverage_pct >= 90.0
        and exit_reversal_rate_pct < 10.0
    )
