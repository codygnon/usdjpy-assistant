#!/usr/bin/env python3
"""
Agent 3 / Track 3: archetype portability program report.

Builds a concrete Agent 3 artifact from the current evidence base:
  - Phase A chart-first routing diagnostic
  - offensive shadow ledger
  - narrow Phase B V44 -> V14 override result
  - conservative ownership table / portability gates

Outputs:
  - research_out/agent3_portability_program.json
  - research_out/agent3_portability_program.md

This is a planning/reporting artifact only. It does not change live routing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "research_out"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.ownership_table import cells_where_owner_is, load_conservative_table
from scripts.portability_program_gates import GATES

PHASE_A_PATH = OUT / "diagnostic_chart_first_routing.json"
LEDGER_PATH = OUT / "offensive_shadow_ledger.json"
PHASE_B_PATH = OUT / "phaseb_v44_to_v14_narrow.json"

JSON_OUT = OUT / "agent3_portability_program.json"
MD_OUT = OUT / "agent3_portability_program.md"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sum_pair_counts(ledger: dict[str, Any], pair: str) -> int:
    total = 0
    for rows in ledger.get("by_dataset", {}).values():
        for row in rows.get("portable_candidates", []) + rows.get("non_portable_london_table", []):
            if row.get("pair") == pair:
                total += int(row.get("count", 0))
    return total


def _sum_london_table_volume(ledger: dict[str, Any]) -> int:
    total = 0
    for rows in ledger.get("by_dataset", {}).values():
        for row in rows.get("non_portable_london_table", []):
            total += int(row.get("count", 0))
    return total


def build_report() -> dict[str, Any]:
    phase_a = _load_json(PHASE_A_PATH)
    ledger = _load_json(LEDGER_PATH)
    phase_b = _load_json(PHASE_B_PATH)
    table = load_conservative_table(research_out=OUT)

    phase_a_500 = phase_a["dataset_summaries"]["500k"]
    phase_a_1000 = phase_a["dataset_summaries"]["1000k"]

    v44_owner_cells = cells_where_owner_is(table, "v44_ny")
    v14_owner_cells = cells_where_owner_is(table, "v14")
    london_owner_cells = cells_where_owner_is(table, "london_v2")

    london_table_volume = _sum_london_table_volume(ledger)
    v44_to_v14_count = _sum_pair_counts(ledger, "v44_ny -> v14")
    v14_to_v44_count = _sum_pair_counts(ledger, "v14 -> v44_ny")

    report: dict[str, Any] = {
        "objective": (
            "Build archetypes that can legitimately support chart authorization, "
            "with archetype portability proven independently before router-level expansion."
        ),
        "evidence_snapshot": {
            "phase_a": {
                "500k": phase_a_500,
                "1000k": phase_a_1000,
                "verdict": phase_a["verdict"],
            },
            "offensive_shadow": {
                "portable_pair_counts": {
                    "v44_ny_to_v14": v44_to_v14_count,
                    "v14_to_v44_ny": v14_to_v44_count,
                },
                "non_portable_london_table_total_count": london_table_volume,
            },
            "phaseb_v44_to_v14": phase_b,
            "stable_owner_cells": {
                "v44_ny": v44_owner_cells,
                "v14": v14_owner_cells,
                "london_v2": london_owner_cells,
            },
        },
        "tracks": {
            "3A_momentum_portability": {
                "status": "NEXT_BUILDABLE_TRACK",
                "thesis": (
                    "V44-style momentum is the best portability candidate because its entry "
                    "logic is mostly chart-driven and less session-anchored than London V2."
                ),
                "current_read": (
                    "Promising as an archetype path, but not yet evidenced by broad disagreement "
                    "volume. We need a dedicated non-native harness, not router promotion."
                ),
                "priority_contexts": [
                    "momentum/er_mid/der_neg",
                    "momentum/er_high/der_neg",
                    "momentum/er_low/der_pos",
                ],
                "why_next": [
                    "Momentum is the most chart-native strategy family in the current stack.",
                    "Range breakout is still structurally session-bound.",
                    "V14 exact substitution failed, so MR is not the next portability bet.",
                ],
                "backtest_harness": {
                    "name": "momentum portability harness",
                    "scope": [
                        "Run V44 logic outside NY in shadow/backtest only.",
                        "Constrain evaluation to stable V44-owned chart cells.",
                        "Compare in-session vs off-session same-cell performance.",
                        "Keep session as a prior in reporting, not a hard gate in the harness.",
                    ],
                    "must_measure": [
                        "authorized bars",
                        "real fills",
                        "avg pips / net usd by session",
                        "profit factor and max drawdown under coupling",
                        "same-cell NY vs non-NY performance split",
                    ],
                    "promotion_gate": GATES["3A_momentum_portability"],
                },
                "no_go_conditions": [
                    "Off-session fills are too sparse to be decision-relevant.",
                    "Off-session same-cell performance is materially worse than NY.",
                    "Coupling-aware drawdown rises without meaningful net improvement.",
                ],
            },
            "3B_generalized_range_archetype": {
                "status": "DESIGN_ONLY_LARGEST_PROJECT",
                "thesis": (
                    "Most offensive shadow disagreement volume points to London ownership, "
                    "which means the largest offensive unlock likely requires a portable "
                    "range-breakout archetype rather than London-session handoff."
                ),
                "current_read": (
                    "This is the highest-upside offensive track, but it is an archetype rebuild, "
                    "not a router tweak."
                ),
                "evidence": {
                    "london_table_total_disagreement_count": london_table_volume,
                    "note": (
                        "High disagreement volume involving london_v2 in the ownership table "
                        "does not prove portability; it does show where the offensive upside likely lives."
                    ),
                },
                "technical_design": {
                    "replace": [
                        "Asian-range anchoring",
                        "London-open / LOR-only timing",
                        "fixed session-relative entry windows",
                    ],
                    "with": [
                        "portable range-formation detector",
                        "portable breakout qualification layer",
                        "range lifecycle state machine independent of London open",
                    ],
                    "core_components": [
                        "formation quality score",
                        "breakout quality score",
                        "range invalidation / reset rules",
                        "context modifiers for spread, volatility, and session prior",
                    ],
                    "first_outputs": [
                        "technical spec only",
                        "historical replay examples for detected ranges",
                        "formation frequency and breakout quality diagnostics",
                    ],
                    "promotion_gate": GATES["3B_generalized_range_archetype"],
                },
            },
            "3C_mean_reversion_decision": {
                "status": "FREEZE_V14_SESSION_NATIVE",
                "decision": "KEEP_TOKYO_NATIVE_ONLY",
                "why": [
                    "V14 chart-gating produced zero strict candidates and zero watchlist candidates.",
                    "Broad NY V14 research already failed.",
                    "Exact-moment V44 -> V14 Phase B produced zero replacements on both datasets.",
                ],
                "implication": (
                    "Do not spend more time trying to route current V14 offensively across sessions. "
                    "If portable MR is still desired later, treat it as a new archetype build."
                ),
                "replacement_path": {
                    "if_needed_later": [
                        "Define portable MR around chart features, not Tokyo session identity.",
                        "Require explicit non-native validation before any router integration.",
                        "Keep current Tokyo V14 untouched as a production strategy until replacement is proven.",
                    ]
                },
                "promotion_gate": GATES["3C_mean_reversion_decision"],
            },
        },
        "priority_order": [
            "3A momentum portability harness",
            "3B generalized range archetype design",
            "3C freeze V14 portability and open replacement-only path if needed later",
        ],
        "agent3_verdict": (
            "Agent 3 should advance portability through momentum first, design generalized range as the "
            "major offensive unlock, and explicitly freeze V14 as session-native for now."
        ),
    }
    criteria_path = OUT / "track3_portability_promotion_criteria.json"
    if criteria_path.is_file():
        report["track3_promotion_criteria"] = json.loads(criteria_path.read_text(encoding="utf-8"))
    return report


def _to_markdown(report: dict[str, Any]) -> str:
    phase_a = report["evidence_snapshot"]["phase_a"]
    portable = report["evidence_snapshot"]["offensive_shadow"]["portable_pair_counts"]
    london_total = report["evidence_snapshot"]["offensive_shadow"]["non_portable_london_table_total_count"]
    phase_b_verdict = report["evidence_snapshot"]["phaseb_v44_to_v14"]["verdict"]

    lines = [
        "# Agent 3 Portability Program",
        "",
        "## Objective",
        report["objective"],
        "",
    ]
    if "track3_promotion_criteria" in report:
        lines.extend([
            "## Track 3 promotion criteria (dual lens)",
            "",
            "Embedded from `research_out/track3_portability_promotion_criteria.json`. Narrative: `track3_portability_promotion_criteria.md`.",
            "",
        ])
    lines.extend([
        "## Evidence Snapshot",
        f"- Phase A agreement rate: 500k `{phase_a['500k']['agreement_rate_pct']}%`, 1000k `{phase_a['1000k']['agreement_rate_pct']}%`",
        f"- Portable shadow disagreements: `v44_ny -> v14 = {portable['v44_ny_to_v14']}`, `v14 -> v44_ny = {portable['v14_to_v44_ny']}`",
        f"- Non-portable London-table disagreement volume: `{london_total}` trades",
        f"- Narrow V44 -> V14 Phase B verdict: {phase_b_verdict}",
        "",
        "## 3A Momentum Portability",
        f"Status: `{report['tracks']['3A_momentum_portability']['status']}`",
        report["tracks"]["3A_momentum_portability"]["thesis"],
        "",
        "Why next:",
    ])
    for item in report["tracks"]["3A_momentum_portability"]["why_next"]:
        lines.append(f"- {item}")
    lines.extend([
        "",
        "Priority contexts:",
    ])
    for item in report["tracks"]["3A_momentum_portability"]["priority_contexts"]:
        lines.append(f"- `{item}`")
    lines.extend([
        "",
        "Harness must measure:",
    ])
    for item in report["tracks"]["3A_momentum_portability"]["backtest_harness"]["must_measure"]:
        lines.append(f"- {item}")
    lines.extend([
        "",
        "## 3B Generalized Range Archetype",
        f"Status: `{report['tracks']['3B_generalized_range_archetype']['status']}`",
        report["tracks"]["3B_generalized_range_archetype"]["thesis"],
        "",
        "Design focus:",
    ])
    for item in report["tracks"]["3B_generalized_range_archetype"]["technical_design"]["with"]:
        lines.append(f"- {item}")
    lines.extend([
        "",
        "Replace:",
    ])
    for item in report["tracks"]["3B_generalized_range_archetype"]["technical_design"]["replace"]:
        lines.append(f"- {item}")
    lines.extend([
        "",
        "## 3C Mean Reversion Decision",
        f"Status: `{report['tracks']['3C_mean_reversion_decision']['status']}`",
        f"Decision: `{report['tracks']['3C_mean_reversion_decision']['decision']}`",
    ])
    for item in report["tracks"]["3C_mean_reversion_decision"]["why"]:
        lines.append(f"- {item}")
    lines.extend([
        "",
        "## Priority Order",
    ])
    for i, item in enumerate(report["priority_order"], start=1):
        lines.append(f"{i}. {item}")
    lines.extend([
        "",
        "## Verdict",
        report["agent3_verdict"],
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    report = build_report()
    JSON_OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    MD_OUT.write_text(_to_markdown(report), encoding="utf-8")
    print(f"Wrote {JSON_OUT}")
    print(f"Wrote {MD_OUT}")
    print(report["agent3_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
