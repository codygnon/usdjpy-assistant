#!/usr/bin/env python3
"""Step 8 Pass C: full v2 stack replay against the 241-trade Phase 7 corpus.

This is the diagnostic-floor gate (PHASE9.9 + PHASE9.6). If the integrated
v2 stack does not pass the floor here, the rebuild is not done and the
brief says: STOP and report which component is the gap.

Stack applied per row:
  1. Pre-decision V1 + V2 (with protected buy-CLR bypass)
  2. Post-decision V5 + V6 validators (V1 sell-burden equivalent skipped
     because pre-veto already handles it)
  3. Cap surviving historical lots to [1, 4] (PHASE9.6)

Floor (PHASE9.9):
  - Net pip recovery >= +278.4p (preferably >= +300.5p; target +324.3p)
  - Net USD recovery  >= +$5,684.56 (preferably >= +$6,000; target +$6,420.90)
  - Blocked trades   <= 110 (preferably <= 100)
  - Blocked winners  <= 52  (preferably <= 45)

Protected cells must remain positive (PHASE9.4 binding constraint):
  - CLR × buy × Phase 2 zone-memory × caveat-trade × 2-3.99 lots (N=17)
  - Tuesday × CLR × buy (N=17)
  - CLR × buy × Phase 2 zone-memory × caveat-trade (N=23)
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.fillmore_v2.legacy_rationale_parser import adapt_corpus_row
from api.fillmore_v2.pre_decision_vetoes import run_pre_vetoes
from api.fillmore_v2.validators import (
    hedge_plus_overconfidence_validator,
    level_language_overreach_validator,
)
from core.fillmore_v2_sizing import cap_historical_lots, rescale_pnl_for_cap

DEFAULT_CORPUS = ROOT / "research_out" / "autonomous_fillmore_forensic_20260501" / "phase7_interaction_dataset.csv"

EXPECTED_BASELINE_PIPS = -308.0
EXPECTED_BASELINE_USD = -7253.2365

# Floor / preferred / target per PHASE9.9
FLOOR_PIP_RECOVERY = 278.4
PREFERRED_PIP_RECOVERY = 300.5
TARGET_PIP_RECOVERY = 324.3
FLOOR_USD_RECOVERY = 5684.56
PREFERRED_USD_RECOVERY = 6000.0
TARGET_USD_RECOVERY = 6420.90
FLOOR_BLOCKED = 110
PREFERRED_BLOCKED = 100
FLOOR_BLOCKED_WINNERS = 52
PREFERRED_BLOCKED_WINNERS = 45


@dataclass
class RowResult:
    blocked: bool
    block_source: str  # 'pre_veto:V1' | 'pre_veto:V2' | 'post:overreach' | 'post:hedge' | 'kept'
    original_pips: float
    original_pnl: float
    original_lots: float
    capped_lots: float
    capped_pnl: float


def _f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    v = row.get(key, "")
    return float(v) if v not in ("", "nan", "NaN", None) else default


def replay_row(row: dict[str, str]) -> RowResult:
    pips = _f(row, "pips")
    pnl = _f(row, "pnl")
    lots = _f(row, "lots")

    try:
        llm_output, snap = adapt_corpus_row(row)
    except Exception:
        # Treat parse failures as kept at original size (most conservative)
        return RowResult(
            blocked=False, block_source="kept", original_pips=pips,
            original_pnl=pnl, original_lots=lots,
            capped_lots=lots, capped_pnl=pnl,
        )

    # 1. Pre-decision V1 + V2 (with bypass already inside the veto fns)
    pv = run_pre_vetoes(snap, deterministic_lots=None)
    if pv.skip_before_call:
        first = pv.fires[0].veto_id
        source = "pre_veto:" + first
        return RowResult(
            blocked=True, block_source=source, original_pips=pips,
            original_pnl=pnl, original_lots=lots,
            capped_lots=0.0, capped_pnl=0.0,
        )

    # 2. Post-decision V5 + V6 (V5=hedge+overconfidence, V6=overreach)
    overreach = level_language_overreach_validator(llm_output, snap)
    if overreach.fired:
        return RowResult(
            blocked=True, block_source="post:overreach", original_pips=pips,
            original_pnl=pnl, original_lots=lots,
            capped_lots=0.0, capped_pnl=0.0,
        )
    hedge = hedge_plus_overconfidence_validator(llm_output, snap)
    if hedge.fired:
        return RowResult(
            blocked=True, block_source="post:hedge", original_pips=pips,
            original_pnl=pnl, original_lots=lots,
            capped_lots=0.0, capped_pnl=0.0,
        )

    # 3. Survivor: cap lots to 4 (degenerate lots <= 0 keep their pnl unchanged)
    if lots <= 0:
        return RowResult(
            blocked=False, block_source="kept", original_pips=pips,
            original_pnl=pnl, original_lots=lots,
            capped_lots=lots, capped_pnl=pnl,
        )
    new_lots = cap_historical_lots(lots)
    new_pnl = rescale_pnl_for_cap(original_pnl=pnl, original_lots=lots, capped_lots=new_lots)
    return RowResult(
        blocked=False, block_source="kept", original_pips=pips,
        original_pnl=pnl, original_lots=lots,
        capped_lots=new_lots, capped_pnl=new_pnl,
    )


def _is_protected_cell_1(row: dict[str, str]) -> bool:
    """CLR × buy × Phase 2 zone-memory × caveat-trade × 2-3.99 lots (N=17)."""
    return (
        row.get("trigger_family") == "critical_level_reaction"
        and row.get("side") == "buy"
        and row.get("prompt_regime") == "Phase 2 zone-memory"
        and row.get("rationale_cluster") in {"critical_level_mixed_caveat_trade", "momentum_with_caveat_trade"}
        and row.get("lot_bucket") == "2-3.99"
    )


def _is_protected_cell_2(row: dict[str, str]) -> bool:
    """Tuesday × CLR × buy (N=17)."""
    return (
        row.get("day_of_week") == "Tuesday"
        and row.get("trigger_family") == "critical_level_reaction"
        and row.get("side") == "buy"
    )


def _is_protected_cell_3(row: dict[str, str]) -> bool:
    """CLR × buy × Phase 2 zone-memory × caveat-trade (N=23)."""
    return (
        row.get("trigger_family") == "critical_level_reaction"
        and row.get("side") == "buy"
        and row.get("prompt_regime") == "Phase 2 zone-memory"
        and row.get("rationale_cluster") in {"critical_level_mixed_caveat_trade", "momentum_with_caveat_trade"}
    )


PROTECTED_CELLS = (
    ("CLR×buy×Phase2-zone-memory×caveat×2-3.99lots", _is_protected_cell_1, 17),
    ("Tuesday×CLR×buy", _is_protected_cell_2, 17),
    ("CLR×buy×Phase2-zone-memory×caveat", _is_protected_cell_3, 23),
)


def run_pass_c(corpus_path: Path = DEFAULT_CORPUS) -> dict:
    rows = list(csv.DictReader(corpus_path.open(newline="", encoding="utf-8")))
    results = [replay_row(r) for r in rows]

    baseline_pips = sum(_f(r, "pips") for r in rows)
    baseline_usd = sum(_f(r, "pnl") for r in rows)

    blocked = [(r, res) for r, res in zip(rows, results) if res.blocked]
    kept = [(r, res) for r, res in zip(rows, results) if not res.blocked]

    blocked_winners = [(r, res) for r, res in blocked if _f(r, "pips") > 0]
    blocked_losers = [(r, res) for r, res in blocked if _f(r, "pips") < 0]
    saved_loser_pips = -sum(_f(r, "pips") for r, _ in blocked_losers)
    saved_loser_usd = -sum(_f(r, "pnl") for r, _ in blocked_losers)
    missed_winner_pips = sum(_f(r, "pips") for r, _ in blocked_winners)
    missed_winner_usd = sum(_f(r, "pnl") for r, _ in blocked_winners)

    net_delta_pips = saved_loser_pips - missed_winner_pips
    net_delta_usd = saved_loser_usd - missed_winner_usd

    # Sizing-only delta on survivors
    survivor_orig_usd = sum(res.original_pnl for _, res in kept)
    survivor_capped_usd = sum(res.capped_pnl for _, res in kept)
    sizing_delta_usd = survivor_orig_usd - survivor_capped_usd

    # Totals: full v2 stack USD = capped survivor USD (blocked rows zeroed)
    final_usd = survivor_capped_usd
    full_recovery_usd = baseline_usd - final_usd  # baseline is negative; final less so → recovery positive
    # We want to express as "vs baseline" — baseline -7253, final, e.g., -832 → recovery 6421
    # baseline_usd is negative; -baseline + final is the loss reduced; recovery = baseline_usd_difference
    # Actually: recovery = final_usd - baseline_usd  (final less negative than baseline → positive)
    full_recovery_usd = final_usd - baseline_usd

    # Block-source breakdown
    by_source: dict[str, int] = {}
    for _, res in blocked:
        by_source[res.block_source] = by_source.get(res.block_source, 0) + 1

    # Protected cells
    protected_results = []
    for name, predicate, expected_n in PROTECTED_CELLS:
        cell_rows = [(r, res) for r, res in zip(rows, results) if predicate(r)]
        cell_blocked = sum(1 for _, res in cell_rows if res.blocked)
        cell_survivors = [res for _, res in cell_rows if not res.blocked]
        cell_pips = sum(res.original_pips for res in cell_survivors)
        cell_orig_usd = sum(res.original_pnl for res in cell_survivors)
        cell_capped_usd = sum(res.capped_pnl for res in cell_survivors)
        protected_results.append({
            "name": name,
            "expected_n": expected_n,
            "found_n": len(cell_rows),
            "blocked": cell_blocked,
            "survivors": len(cell_survivors),
            "survivor_pips": cell_pips,
            "survivor_usd_original": cell_orig_usd,
            "survivor_usd_capped": cell_capped_usd,
        })

    return {
        "rows": len(rows),
        "baseline_pips": baseline_pips,
        "baseline_usd": baseline_usd,
        "blocked": len(blocked),
        "blocked_winners": len(blocked_winners),
        "blocked_losers": len(blocked_losers),
        "saved_loser_pips": saved_loser_pips,
        "saved_loser_usd": saved_loser_usd,
        "missed_winner_pips": missed_winner_pips,
        "missed_winner_usd": missed_winner_usd,
        "net_delta_pips": net_delta_pips,
        "net_delta_usd": net_delta_usd,
        "survivor_orig_usd": survivor_orig_usd,
        "survivor_capped_usd": survivor_capped_usd,
        "sizing_delta_usd": sizing_delta_usd,
        "final_usd_after_cap": final_usd,
        "full_recovery_usd_with_cap": full_recovery_usd,
        "block_sources": by_source,
        "protected": protected_results,
    }


def _print_report(r: dict) -> None:
    print("=" * 64)
    print("PASS C — Full v2 stack replay (PHASE9.9 diagnostic floor)")
    print("=" * 64)
    print(f"rows                         : {r['rows']}")
    print(f"baseline                     : {r['baseline_pips']:.1f}p / ${r['baseline_usd']:,.2f}")
    print()
    print("--- Admission filter (V1+V2 pre-veto + V5+V6 post-validator) ---")
    print(f"blocked                      : {r['blocked']}")
    print(f"  by source                  : {r['block_sources']}")
    print(f"blocked winners              : {r['blocked_winners']}")
    print(f"blocked losers               : {r['blocked_losers']}")
    print(f"missed winner pips           : {r['missed_winner_pips']:.1f}")
    print(f"missed winner USD            : ${r['missed_winner_usd']:,.2f}")
    print(f"saved loser pips             : {r['saved_loser_pips']:.1f}")
    print(f"saved loser USD              : ${r['saved_loser_usd']:,.2f}")
    print(f"net pip recovery (admission) : +{r['net_delta_pips']:.1f}p")
    print(f"net USD recovery (admission) : +${r['net_delta_usd']:,.2f}")
    print()
    print("--- Sizing layer (cap to [1, 4] on survivors) ---")
    print(f"survivor USD at original size: ${r['survivor_orig_usd']:,.2f}")
    print(f"survivor USD after cap-to-4  : ${r['survivor_capped_usd']:,.2f}")
    print(f"sizing-only delta            : ${r['sizing_delta_usd']:,.2f}")
    print()
    print("--- Combined: full v2 stack vs baseline ---")
    print(f"final USD with v2 stack      : ${r['final_usd_after_cap']:,.2f}")
    print(f"FULL recovery vs baseline    : +${r['full_recovery_usd_with_cap']:,.2f}")
    print()
    print("--- Floor / preferred / target ---")
    print(f"net pip recovery     : +{r['net_delta_pips']:.1f}p   (floor +{FLOOR_PIP_RECOVERY}, prefer +{PREFERRED_PIP_RECOVERY}, target +{TARGET_PIP_RECOVERY})")
    print(f"net USD recovery     : +${r['net_delta_usd']:,.2f}  (floor +${FLOOR_USD_RECOVERY:,.2f}, prefer +${PREFERRED_USD_RECOVERY:,.2f}, target +${TARGET_USD_RECOVERY:,.2f})")
    print(f"full USD w/cap-to-4  : +${r['full_recovery_usd_with_cap']:,.2f}  (target +${TARGET_USD_RECOVERY:,.2f})")
    print(f"blocked trades       : {r['blocked']}      (floor <={FLOOR_BLOCKED}, prefer <={PREFERRED_BLOCKED})")
    print(f"blocked winners      : {r['blocked_winners']}       (floor <={FLOOR_BLOCKED_WINNERS}, prefer <={PREFERRED_BLOCKED_WINNERS})")
    print()
    print("--- Protected cells (must remain positive) ---")
    for cell in r["protected"]:
        marker = "OK" if cell["survivor_usd_original"] > 0 else "** NEGATIVE **"
        print(
            f"  {cell['name']:50s}"
            f" found={cell['found_n']:3d} (expected {cell['expected_n']})"
            f" blocked={cell['blocked']:2d}"
            f" survivors={cell['survivors']:2d}"
            f" survivor pips={cell['survivor_pips']:+.1f}"
            f" survivor USD orig=${cell['survivor_usd_original']:+,.2f}"
            f" capped=${cell['survivor_usd_capped']:+,.2f}  [{marker}]"
        )
    print()
    # Pass/fail summary
    pass_pip_floor = r["net_delta_pips"] >= FLOOR_PIP_RECOVERY
    pass_usd_floor = r["net_delta_usd"] >= FLOOR_USD_RECOVERY
    pass_blocked = r["blocked"] <= FLOOR_BLOCKED
    pass_blocked_win = r["blocked_winners"] <= FLOOR_BLOCKED_WINNERS
    pass_protected_all = all(c["survivor_usd_original"] > 0 for c in r["protected"])
    overall = pass_pip_floor and pass_usd_floor and pass_blocked and pass_blocked_win and pass_protected_all
    print(f"pip-floor               : {'PASS' if pass_pip_floor else 'FAIL'}")
    print(f"usd-floor               : {'PASS' if pass_usd_floor else 'FAIL'}")
    print(f"blocked-trades ceiling  : {'PASS' if pass_blocked else 'FAIL'}")
    print(f"blocked-winners ceiling : {'PASS' if pass_blocked_win else 'FAIL'}")
    print(f"protected cells positive: {'PASS' if pass_protected_all else 'FAIL'}")
    print()
    print(f"OVERALL DIAGNOSTIC FLOOR: {'PASS' if overall else 'FAIL'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    args = parser.parse_args()
    r = run_pass_c(args.corpus)
    _print_report(r)


if __name__ == "__main__":
    main()
