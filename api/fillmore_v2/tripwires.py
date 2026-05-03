"""Always-on tripwires — Phase 9 Step 9 (PHASE9.10).

Pure evaluation of the always-active tripwires from `docs/fillmore_v2/rollout.md`.
No I/O, no state writes — callers feed in observed counters and receive a
verdict. The autonomous loop checks each tripwire after every closed trade
and after every Nth LLM call; flips of red status are operator alerts.

Tripwires:

  T1. Sell-side WR ≥ 45% within first 30 sell trades.
  T2. Sell-CLR killed if first 30 post-redesign sell-CLR closes net negative.
  T3. Caveat-resolution validator fire rate within 50%–150% of Phase 9
      replay expectation after 100 LLM calls.
  T4. Sizing function never produces > 4 lots.
  T5. Skipped setup forward outcomes present for ≥ 98% of skips.

The exit-reversal tripwire from PHASE9.7 is implemented in
`exit_layer.should_halt_for_exit_reversals` — separated because its window
is 50 closes vs the others' 30/100.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional

# --- Thresholds (PHASE9.10 + rollout.md) -------------------------------------

T1_MIN_SELL_TRADES = 30
T1_KILL_WR = 0.35           # Stage 2 hard kill below this
T1_REQUIRED_WR = 0.45       # advance criterion / tripwire alarm above this

T2_MIN_SELL_CLR_CLOSES = 30
T2_KILL_NET_NEGATIVE = True  # any net-negative net pips kills sell-CLR

T3_MIN_LLM_CALLS = 100
T3_REPLAY_EXPECTATION_FIRE_RATE = 0.617  # from step2_shadow_replay_baseline (caveat_resolution)
T3_BAND_LOWER = 0.50  # 50% of expectation
T3_BAND_UPPER = 1.50  # 150% of expectation

T4_HARD_LOTS_CEILING = 4.0

T5_MIN_SKIPS_FOR_AUDIT = 100
T5_REQUIRED_OUTCOME_COVERAGE = 0.98


@dataclass
class TripwireResult:
    tripwire_id: str
    status: Literal["green", "amber", "red"]
    detail: str


# --- T1: sell-side WR --------------------------------------------------------

def check_sell_side_wr(
    *,
    sell_wins: int, sell_losses: int, sell_breakevens: int = 0,
) -> TripwireResult:
    n = sell_wins + sell_losses + sell_breakevens
    if n < T1_MIN_SELL_TRADES:
        return TripwireResult(
            tripwire_id="T1_sell_wr",
            status="green",
            detail=f"insufficient sample: {n}/{T1_MIN_SELL_TRADES} sell trades",
        )
    wr = sell_wins / n if n else 0.0
    if wr < T1_KILL_WR:
        return TripwireResult(
            tripwire_id="T1_sell_wr",
            status="red",
            detail=f"sell WR {wr:.3f} < {T1_KILL_WR} hard-kill threshold ({sell_wins}/{n})",
        )
    if wr < T1_REQUIRED_WR:
        return TripwireResult(
            tripwire_id="T1_sell_wr",
            status="amber",
            detail=f"sell WR {wr:.3f} < {T1_REQUIRED_WR} target ({sell_wins}/{n})",
        )
    return TripwireResult(
        tripwire_id="T1_sell_wr", status="green",
        detail=f"sell WR {wr:.3f} >= {T1_REQUIRED_WR} ({sell_wins}/{n})",
    )


# --- T2: sell-CLR survival --------------------------------------------------

def check_sell_clr_kill(
    *, sell_clr_closes: list[float],
) -> TripwireResult:
    """Per PHASE9.3: 'Kill criterion: 30 post-redesign sell-CLR closes with
    negative net pips or WR below 45% kills sell-CLR until manual review.'
    """
    n = len(sell_clr_closes)
    if n < T2_MIN_SELL_CLR_CLOSES:
        return TripwireResult(
            tripwire_id="T2_sell_clr_kill",
            status="green",
            detail=f"insufficient sample: {n}/{T2_MIN_SELL_CLR_CLOSES}",
        )
    net_pips = sum(sell_clr_closes)
    wins = sum(1 for p in sell_clr_closes if p > 0)
    wr = wins / n
    if T2_KILL_NET_NEGATIVE and net_pips < 0:
        return TripwireResult(
            tripwire_id="T2_sell_clr_kill",
            status="red",
            detail=f"sell-CLR net {net_pips:+.1f}p over {n} closes — kill until manual review",
        )
    if wr < T1_REQUIRED_WR:
        return TripwireResult(
            tripwire_id="T2_sell_clr_kill",
            status="red",
            detail=f"sell-CLR WR {wr:.3f} < {T1_REQUIRED_WR} over {n} closes — kill",
        )
    return TripwireResult(
        tripwire_id="T2_sell_clr_kill", status="green",
        detail=f"sell-CLR net {net_pips:+.1f}p / WR {wr:.3f} over {n} closes",
    )


# --- T3: caveat-resolution validator fire-rate band ------------------------

def check_caveat_validator_fire_rate(
    *, llm_calls: int, caveat_validator_fires: int,
    expected_rate: float = T3_REPLAY_EXPECTATION_FIRE_RATE,
) -> TripwireResult:
    if llm_calls < T3_MIN_LLM_CALLS:
        return TripwireResult(
            tripwire_id="T3_caveat_fire_rate",
            status="green",
            detail=f"insufficient sample: {llm_calls}/{T3_MIN_LLM_CALLS} calls",
        )
    rate = caveat_validator_fires / llm_calls
    lower = expected_rate * T3_BAND_LOWER
    upper = expected_rate * T3_BAND_UPPER
    if rate < lower or rate > upper:
        return TripwireResult(
            tripwire_id="T3_caveat_fire_rate",
            status="red",
            detail=(
                f"caveat fire rate {rate:.3f} outside [{lower:.3f}, {upper:.3f}] "
                f"({caveat_validator_fires}/{llm_calls}, expected {expected_rate:.3f})"
            ),
        )
    return TripwireResult(
        tripwire_id="T3_caveat_fire_rate", status="green",
        detail=f"caveat fire rate {rate:.3f} within band ({caveat_validator_fires}/{llm_calls})",
    )


# --- T4: sizing hard ceiling ------------------------------------------------

def check_sizing_ceiling(*, max_lots_seen: float) -> TripwireResult:
    """Pure check against the stored maximum lots produced by the sizing
    function. PHASE9.6's hard cap is 4.0 — anything above is a code defect.
    """
    if max_lots_seen > T4_HARD_LOTS_CEILING:
        return TripwireResult(
            tripwire_id="T4_sizing_ceiling",
            status="red",
            detail=f"max lots {max_lots_seen} > {T4_HARD_LOTS_CEILING} ceiling — sizing layer breach",
        )
    return TripwireResult(
        tripwire_id="T4_sizing_ceiling", status="green",
        detail=f"max lots {max_lots_seen} within {T4_HARD_LOTS_CEILING} ceiling",
    )


# --- T5: skip-outcome coverage ---------------------------------------------

def check_skip_outcome_coverage(
    *, total_skips: int, skips_with_outcomes: int,
) -> TripwireResult:
    if total_skips < T5_MIN_SKIPS_FOR_AUDIT:
        return TripwireResult(
            tripwire_id="T5_skip_coverage",
            status="green",
            detail=f"insufficient sample: {total_skips}/{T5_MIN_SKIPS_FOR_AUDIT} skips",
        )
    coverage = skips_with_outcomes / total_skips
    if coverage < T5_REQUIRED_OUTCOME_COVERAGE:
        return TripwireResult(
            tripwire_id="T5_skip_coverage",
            status="red",
            detail=(
                f"skip outcome coverage {coverage:.3f} < {T5_REQUIRED_OUTCOME_COVERAGE} "
                f"({skips_with_outcomes}/{total_skips})"
            ),
        )
    return TripwireResult(
        tripwire_id="T5_skip_coverage", status="green",
        detail=f"skip outcome coverage {coverage:.3f}",
    )


# --- Aggregator -------------------------------------------------------------

@dataclass
class TripwireSnapshot:
    """Aggregated tripwire status. `any_red` triggers operator alert."""
    results: list[TripwireResult]

    @property
    def any_red(self) -> bool:
        return any(r.status == "red" for r in self.results)

    @property
    def any_amber(self) -> bool:
        return any(r.status == "amber" for r in self.results)

    def red_ids(self) -> list[str]:
        return [r.tripwire_id for r in self.results if r.status == "red"]

    def to_audit(self) -> list[dict]:
        return [
            {"id": r.tripwire_id, "status": r.status, "detail": r.detail}
            for r in self.results
        ]


def evaluate_all(
    *,
    sell_wins: int = 0, sell_losses: int = 0, sell_breakevens: int = 0,
    sell_clr_closes: Optional[list[float]] = None,
    llm_calls: int = 0, caveat_validator_fires: int = 0,
    max_lots_seen: float = 0.0,
    total_skips: int = 0, skips_with_outcomes: int = 0,
) -> TripwireSnapshot:
    return TripwireSnapshot(results=[
        check_sell_side_wr(sell_wins=sell_wins, sell_losses=sell_losses, sell_breakevens=sell_breakevens),
        check_sell_clr_kill(sell_clr_closes=sell_clr_closes or []),
        check_caveat_validator_fire_rate(llm_calls=llm_calls, caveat_validator_fires=caveat_validator_fires),
        check_sizing_ceiling(max_lots_seen=max_lots_seen),
        check_skip_outcome_coverage(total_skips=total_skips, skips_with_outcomes=skips_with_outcomes),
    ])
