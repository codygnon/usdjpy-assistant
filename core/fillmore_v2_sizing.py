"""Deterministic Sizing Layer — Phase 9 Step 4 (PHASE9.6).

The LLM has zero sizing authority. This module is the sole determiner of
lot size for Auto Fillmore v2. Implements the PHASE9.6 pseudocode verbatim
with explicit branch flags so the audit log shows which throttles fired.

ISOLATION REQUIREMENT (Step 1 import-graph audit): this module must not
import anything from `core.runner_score`, and `core.runner_score` must not
import anything from here. The two sizing systems serve different programs
(runner_score is the manual-trial Trial #10 sizing path; this is autonomous
Fillmore). They share only stdlib types. Enforced by tests.

REFERENTIAL TRANSPARENCY: `compute_autonomous_lots` is a pure function of
its `SizingContext`. Identical inputs produce identical outputs across any
number of calls. No globals, no clock, no I/O. Enforced by tests.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Optional

# --- Constants (PHASE9.6) ----------------------------------------------------

DEFAULT_RISK_PCT = 0.0025
PAPER_OR_LIGHT_RISK_PCT = 0.0010
RAMP_RISK_PCT_CEILING = 0.0050  # only after forward 100-trade PF >= 1.10 AND net_pips_100 > 0
LOTS_HARD_MIN = 1.0
LOTS_HARD_MAX = 4.0
PROTECTED_FLOOR_LOTS = 2.0
PROTECTED_RISK_FRACTION = 0.005  # protected floor only when risk_after_fill <= 0.5% equity
ROLLING_PNL_PENALTY_USD = -50.0
ROLLING_LOT_WEIGHTED_PENALTY = -100.0
SIDE_EXPOSURE_PENALTY_LOTS = 4.0
THROTTLE_FACTOR = 0.50
LOT_STEP = 0.01

Stage = Literal["paper", "0.1x", "0.5x", "full"]
Side = Literal["buy", "sell"]
VolatilityRegime = Literal["normal", "elevated", "unknown"]


# --- Inputs / outputs --------------------------------------------------------

@dataclass(frozen=True)
class SizingContext:
    """Inputs to the deterministic sizing function.

    Frozen so the function is pure and testable. All fields are required —
    no Optional defaults — to force callers to assemble a complete context.
    Defensive defaults belong upstream in the snapshot layer.
    """
    account_equity: float
    sl_pips: float
    pip_value_per_lot: float
    proposed_side: Side
    open_lots_buy: float
    open_lots_sell: float
    rolling_20_trade_pnl: float
    rolling_20_lot_weighted_pnl: float
    risk_after_fill_usd: float
    volatility_regime: VolatilityRegime
    stage: Stage
    forward_100_trade_profit_factor: float
    net_pips_100: float
    protected_buy_clr_packet: bool


@dataclass(frozen=True)
class SizingResult:
    """Audit-grade output. Every branch the function took is recorded so the
    suggestion row's `sizing_inputs_json` plus this result is enough to
    explain any decision in a forensic audit.
    """
    lots: float
    risk_pct: float
    raw_lots: float
    rolling_throttle_applied: bool
    side_exposure_throttle_applied: bool
    volatility_throttle_applied: bool
    protected_floor_applied: bool
    cap_to_max_applied: bool
    cap_to_min_applied: bool
    notes: tuple[str, ...] = ()


# --- Helpers (pure) ----------------------------------------------------------

def round_to_step(value: float, step: float) -> float:
    """Half-up rounding to a step. Avoids Python's banker's rounding so the
    function is bit-stable across platforms.
    """
    if step <= 0:
        raise ValueError("step must be positive")
    return math.floor(value / step + 0.5) * step


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(value, hi))


def select_risk_pct(
    *, stage: Stage, forward_100_trade_pf: float, net_pips_100: float
) -> float:
    """Per PHASE9.6: paper/0.1x → 0.0010; default 0.0025; 0.0050 ceiling only
    after forward 100-trade PF ≥ 1.10 AND net_pips_100 > 0.

    Note: the ceiling clause USES `min(...)` in the blueprint pseudocode,
    which means the ramp can only LOWER the ceiling, not raise from
    paper-stage 0.0010. That's faithful to the spec: ramp gates the maximum,
    it doesn't override the stage floor.
    """
    risk_pct = DEFAULT_RISK_PCT
    if stage in ("paper", "0.1x"):
        risk_pct = PAPER_OR_LIGHT_RISK_PCT
    if forward_100_trade_pf >= 1.10 and net_pips_100 > 0:
        risk_pct = min(risk_pct, RAMP_RISK_PCT_CEILING)
    return risk_pct


# --- Main sizing function (PHASE9.6) ----------------------------------------

def compute_autonomous_lots(ctx: SizingContext) -> SizingResult:
    """Deterministic lot sizing. Pure function of `ctx`.

    Sequence (PHASE9.6 verbatim):
      1. Pick risk_pct from stage and ramp gate.
      2. Compute raw_lots from equity * risk / (sl * pip_value).
      3. Clamp to [1, 4] and round to 0.01.
      4. Apply 50% throttle for poor rolling P&L.
      5. Apply 50% throttle for high side exposure.
      6. Apply 50% throttle for elevated volatility.
      7. If protected buy-CLR AND risk-after-fill within 0.5% equity: floor at 2.0.
      8. Re-clamp to [1, 4] and round.

    Degenerate inputs (sl_pips ≤ 0, pip_value ≤ 0, equity ≤ 0) return
    lots=0.0 with a `degenerate_inputs` note. Callers must skip the trade
    and log a halt rather than place at zero size.
    """
    risk_pct = select_risk_pct(
        stage=ctx.stage,
        forward_100_trade_pf=ctx.forward_100_trade_profit_factor,
        net_pips_100=ctx.net_pips_100,
    )

    if ctx.sl_pips <= 0 or ctx.pip_value_per_lot <= 0 or ctx.account_equity <= 0:
        return SizingResult(
            lots=0.0,
            risk_pct=risk_pct,
            raw_lots=0.0,
            rolling_throttle_applied=False,
            side_exposure_throttle_applied=False,
            volatility_throttle_applied=False,
            protected_floor_applied=False,
            cap_to_max_applied=False,
            cap_to_min_applied=False,
            notes=("degenerate_inputs",),
        )

    raw_lots = (ctx.account_equity * risk_pct) / (ctx.sl_pips * ctx.pip_value_per_lot)
    rounded = round_to_step(raw_lots, LOT_STEP)
    cap_max = rounded > LOTS_HARD_MAX
    cap_min = rounded < LOTS_HARD_MIN
    lots = clamp(rounded, LOTS_HARD_MIN, LOTS_HARD_MAX)

    rolling_throttle = (
        ctx.rolling_20_trade_pnl < ROLLING_PNL_PENALTY_USD
        or ctx.rolling_20_lot_weighted_pnl < ROLLING_LOT_WEIGHTED_PENALTY
    )
    side_exposure = (
        ctx.open_lots_buy >= SIDE_EXPOSURE_PENALTY_LOTS
        if ctx.proposed_side == "buy"
        else ctx.open_lots_sell >= SIDE_EXPOSURE_PENALTY_LOTS
    )
    elevated_vol = ctx.volatility_regime == "elevated"

    if rolling_throttle:
        lots *= THROTTLE_FACTOR
    if side_exposure:
        lots *= THROTTLE_FACTOR
    if elevated_vol:
        lots *= THROTTLE_FACTOR

    protected_floor = False
    if (
        ctx.protected_buy_clr_packet
        and ctx.risk_after_fill_usd <= ctx.account_equity * PROTECTED_RISK_FRACTION
    ):
        if lots < PROTECTED_FLOOR_LOTS:
            lots = PROTECTED_FLOOR_LOTS
            protected_floor = True

    final = clamp(round_to_step(lots, LOT_STEP), LOTS_HARD_MIN, LOTS_HARD_MAX)

    return SizingResult(
        lots=final,
        risk_pct=risk_pct,
        raw_lots=raw_lots,
        rolling_throttle_applied=rolling_throttle,
        side_exposure_throttle_applied=side_exposure,
        volatility_throttle_applied=elevated_vol,
        protected_floor_applied=protected_floor,
        cap_to_max_applied=cap_max,
        cap_to_min_applied=cap_min,
    )


# --- Replay helper for Step 4 + Step 8 Pass C --------------------------------

def cap_historical_lots(original_lots: float, *, cap: float = LOTS_HARD_MAX) -> float:
    """Cap historical lots to [1, cap] for retroactive sizing replay.

    Per PHASE9.6: 'apply the Phase 9 admission filter, then cap surviving
    historical lots to [1, 4]'. This is the only sizing transformation
    available for the corpus replay because account equity and rolling P&L
    aren't reconstructable from historical rows.
    """
    return clamp(original_lots, LOTS_HARD_MIN, cap)


def rescale_pnl_for_cap(*, original_pnl: float, original_lots: float, capped_lots: float) -> float:
    """Linear-scale the realized USD P&L to the capped lot count.

    PnL scales linearly with lots when SL/TP/spread are fixed (which they
    are in the corpus). Returns the capped P&L.
    """
    if original_lots <= 0:
        return 0.0
    return original_pnl * (capped_lots / original_lots)
