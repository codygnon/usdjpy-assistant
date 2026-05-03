"""Stage progression evaluator — Phase 9 Step 9 (PHASE9.10).

Pure logic. Given the current stage + observed counters, returns whether
v2 should ADVANCE, HOLD, or KILL back to a safer stage. No I/O. The
operator (or a scheduled job) is responsible for actually applying the
verdict by updating `runtime_state.json`.

Stages (PHASE9.10):

  - paper          : full new stack, no real capital
  - 0.1x           : risk_pct multiplied by 0.4 effectively (0.0010)
  - 0.5x           : deterministic sizing × 0.5
  - full           : 0.25%–0.5% risk function, cap 4 lots

Per the user's Step 1 directive, advancing from any stage to the next is a
manual operator action. This module gives an evaluation but does NOT mutate
state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from .tripwires import T1_KILL_WR, T1_REQUIRED_WR, evaluate_all

Stage = Literal["paper", "0.1x", "0.5x", "full"]
STAGE_ORDER: tuple[Stage, ...] = ("paper", "0.1x", "0.5x", "full")


# --- Per-stage criteria (PHASE9.10) ------------------------------------------

@dataclass(frozen=True)
class StageCriteria:
    name: Stage
    advance_min_closes: int
    advance_min_net_pips: float
    advance_min_pf: float
    kill_max_neg_pips: float        # absolute pip drawdown that kills
    kill_max_neg_pips_label: str
    kill_pf_below: float            # PF below this kills


PAPER = StageCriteria(
    name="paper", advance_min_closes=50, advance_min_net_pips=0.0,
    advance_min_pf=0.9, kill_max_neg_pips=-150.0,
    kill_max_neg_pips_label="-150p cumulative", kill_pf_below=0.0,
)
LIGHT = StageCriteria(
    name="0.1x", advance_min_closes=50, advance_min_net_pips=0.0,
    advance_min_pf=0.9, kill_max_neg_pips=-100.0,
    kill_max_neg_pips_label="-100p from stage start", kill_pf_below=0.5,
)
HALF = StageCriteria(
    name="0.5x", advance_min_closes=100, advance_min_net_pips=0.0,
    advance_min_pf=1.0, kill_max_neg_pips=-150.0,
    kill_max_neg_pips_label="-150p from stage start", kill_pf_below=0.0,
)
FULL = StageCriteria(
    name="full", advance_min_closes=200, advance_min_net_pips=0.0,
    advance_min_pf=0.0, kill_max_neg_pips=float("-inf"),
    kill_max_neg_pips_label="any Phase 8 primary failure reappears",
    kill_pf_below=0.0,
)
STAGES: dict[Stage, StageCriteria] = {
    "paper": PAPER, "0.1x": LIGHT, "0.5x": HALF, "full": FULL,
}


# --- Verdict ---------------------------------------------------------------

@dataclass
class StageVerdict:
    current_stage: Stage
    action: Literal["hold", "advance", "kill"]
    next_stage: Optional[Stage] = None  # for advance / kill
    reason: str = ""
    tripwire_alerts: list[str] = None  # type: ignore[assignment]


@dataclass
class StageObservations:
    """Aggregated counters for the current stage window.

    `closes_in_stage` resets each time the operator advances/rolls back.
    """
    stage: Stage
    closes_in_stage: int
    net_pips_in_stage: float
    profit_factor_in_stage: float
    sell_wins: int = 0
    sell_losses: int = 0
    sell_breakevens: int = 0
    sell_clr_closes: Optional[list[float]] = None
    llm_calls: int = 0
    caveat_validator_fires: int = 0
    max_lots_seen: float = 0.0
    total_skips: int = 0
    skips_with_outcomes: int = 0
    cumulative_drawdown_pips: float = 0.0  # stage-start to now (negative if loss)
    any_phase8_primary_failure_reappeared: bool = False


def _next_stage(current: Stage) -> Optional[Stage]:
    try:
        idx = STAGE_ORDER.index(current)
    except ValueError:
        return None
    if idx + 1 >= len(STAGE_ORDER):
        return None
    return STAGE_ORDER[idx + 1]


def _previous_stage(current: Stage) -> Optional[Stage]:
    try:
        idx = STAGE_ORDER.index(current)
    except ValueError:
        return None
    if idx == 0:
        return None
    return STAGE_ORDER[idx - 1]


def evaluate_stage(observations: StageObservations) -> StageVerdict:
    """Pure verdict on current stage. Order of evaluation:
      1. Kill conditions (most severe wins).
      2. Tripwire-driven kills.
      3. Advance conditions.
      4. Otherwise: hold.
    """
    crit = STAGES[observations.stage]

    tripwires = evaluate_all(
        sell_wins=observations.sell_wins,
        sell_losses=observations.sell_losses,
        sell_breakevens=observations.sell_breakevens,
        sell_clr_closes=observations.sell_clr_closes or [],
        llm_calls=observations.llm_calls,
        caveat_validator_fires=observations.caveat_validator_fires,
        max_lots_seen=observations.max_lots_seen,
        total_skips=observations.total_skips,
        skips_with_outcomes=observations.skips_with_outcomes,
    )
    red_ids = tripwires.red_ids()

    # 1. Stage-specific kill (Stage "full" uses the Phase 8 reappearance flag)
    if observations.stage == "full" and observations.any_phase8_primary_failure_reappeared:
        return StageVerdict(
            current_stage=observations.stage, action="kill",
            next_stage=_previous_stage(observations.stage),
            reason="Phase 8 primary failure reappeared", tripwire_alerts=red_ids,
        )
    if observations.cumulative_drawdown_pips <= crit.kill_max_neg_pips:
        return StageVerdict(
            current_stage=observations.stage, action="kill",
            next_stage=_previous_stage(observations.stage),
            reason=(
                f"drawdown {observations.cumulative_drawdown_pips:.1f}p "
                f"<= kill threshold {crit.kill_max_neg_pips_label}"
            ),
            tripwire_alerts=red_ids,
        )
    if crit.kill_pf_below > 0 and observations.closes_in_stage >= crit.advance_min_closes \
            and observations.profit_factor_in_stage < crit.kill_pf_below:
        return StageVerdict(
            current_stage=observations.stage, action="kill",
            next_stage=_previous_stage(observations.stage),
            reason=f"PF {observations.profit_factor_in_stage:.2f} < {crit.kill_pf_below}",
            tripwire_alerts=red_ids,
        )

    # 2. Tripwire kills (sell-side WR or sell-CLR red is a kill, others alert)
    sell_kill_tripwires = {"T1_sell_wr", "T2_sell_clr_kill"}
    fatal_red = [r for r in tripwires.results if r.status == "red" and r.tripwire_id in sell_kill_tripwires]
    if fatal_red:
        return StageVerdict(
            current_stage=observations.stage, action="kill",
            next_stage=_previous_stage(observations.stage),
            reason="; ".join(r.detail for r in fatal_red),
            tripwire_alerts=red_ids,
        )

    # 3. Advance
    can_advance = (
        observations.closes_in_stage >= crit.advance_min_closes
        and observations.net_pips_in_stage >= crit.advance_min_net_pips
        and observations.profit_factor_in_stage >= crit.advance_min_pf
    )
    if can_advance and not red_ids:
        nxt = _next_stage(observations.stage)
        if nxt is not None:
            return StageVerdict(
                current_stage=observations.stage, action="advance",
                next_stage=nxt,
                reason=(
                    f"{observations.closes_in_stage} closes, "
                    f"net {observations.net_pips_in_stage:+.1f}p, "
                    f"PF {observations.profit_factor_in_stage:.2f}"
                ),
                tripwire_alerts=[],
            )

    # 4. Hold
    why_hold = []
    if observations.closes_in_stage < crit.advance_min_closes:
        why_hold.append(f"need {crit.advance_min_closes - observations.closes_in_stage} more closes")
    if observations.net_pips_in_stage < crit.advance_min_net_pips:
        why_hold.append(f"net {observations.net_pips_in_stage:+.1f}p < {crit.advance_min_net_pips}")
    if observations.profit_factor_in_stage < crit.advance_min_pf:
        why_hold.append(f"PF {observations.profit_factor_in_stage:.2f} < {crit.advance_min_pf}")
    if red_ids:
        why_hold.append(f"red tripwires: {red_ids}")
    return StageVerdict(
        current_stage=observations.stage, action="hold",
        reason="; ".join(why_hold) or "stage criteria not yet met",
        tripwire_alerts=red_ids,
    )
