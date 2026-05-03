#!/usr/bin/env python3
"""Step 8 Pass B: agreement between v2 validator code and Phase 7 labels.

For every row in the 241-trade corpus:
  - Compute v2 V1 fire via `v1_sell_caveat_template_veto` against a snapshot
    built by the legacy rationale parser.
  - Compute v2 V2 fire via `v2_mixed_overlap_entry_veto`.
  - Compare with Phase 7 ground-truth labels:
      V1 truth = rationale_cluster in V1_CLUSTERS AND side == 'sell'
      V2 truth = timeframe_alignment_clean == 'mixed' AND session in V2_SESSIONS

Reports agreement rate. The user-confirmed bar is ≥ 95% V1 / ≥ 90% V2.
Disagreements are expected on V2 because the v2 validator carries a
protected-edge bypass (buy-CLR + score >= 70 + lots <= 4) that the Phase 7
label doesn't have.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.fillmore_v2.legacy_rationale_parser import adapt_corpus_row
from api.fillmore_v2.pre_decision_vetoes import (
    v1_sell_caveat_template_veto,
    v2_mixed_overlap_entry_veto,
)

DEFAULT_CORPUS = ROOT / "research_out" / "autonomous_fillmore_forensic_20260501" / "phase7_interaction_dataset.csv"

V1_CLUSTERS = {"momentum_with_caveat_trade", "critical_level_mixed_caveat_trade"}
V2_SESSIONS = {"london/ny overlap", "tokyo/london overlap"}


def _phase7_v1_label(row: dict[str, str]) -> bool:
    return row.get("rationale_cluster", "") in V1_CLUSTERS and row.get("side") == "sell"


def _phase7_v2_label(row: dict[str, str]) -> bool:
    return (
        row.get("timeframe_alignment_clean", "") == "mixed"
        and row.get("session", "") in V2_SESSIONS
    )


def run_pass_b(corpus_path: Path = DEFAULT_CORPUS) -> dict[str, float | int]:
    rows = list(csv.DictReader(corpus_path.open(newline="", encoding="utf-8")))

    v1_match = v1_truth_pos = v1_v2_pos = v1_only_truth = v1_only_v2 = 0
    v2_match = v2_truth_pos = v2_v2_pos = v2_only_truth = v2_only_v2 = 0
    n = 0

    for row in rows:
        n += 1
        # v2 code outputs
        try:
            _llm, snap = adapt_corpus_row(row)
        except Exception:
            continue
        v1_v2 = v1_sell_caveat_template_veto(snap).fired
        v2_v2 = v2_mixed_overlap_entry_veto(snap, deterministic_lots=None).fired

        # Ground truth from Phase 7 labels
        v1_t = _phase7_v1_label(row)
        v2_t = _phase7_v2_label(row)

        v1_match += int(v1_v2 == v1_t)
        v1_truth_pos += int(v1_t)
        v1_v2_pos += int(v1_v2)
        v1_only_truth += int(v1_t and not v1_v2)
        v1_only_v2 += int(v1_v2 and not v1_t)

        v2_match += int(v2_v2 == v2_t)
        v2_truth_pos += int(v2_t)
        v2_v2_pos += int(v2_v2)
        v2_only_truth += int(v2_t and not v2_v2)
        v2_only_v2 += int(v2_v2 and not v2_t)

    return {
        "rows": n,
        "v1_agreement": v1_match / n,
        "v1_truth_positive": v1_truth_pos,
        "v1_v2_positive": v1_v2_pos,
        "v1_only_truth": v1_only_truth,
        "v1_only_v2": v1_only_v2,
        "v2_agreement": v2_match / n,
        "v2_truth_positive": v2_truth_pos,
        "v2_v2_positive": v2_v2_pos,
        "v2_only_truth": v2_only_truth,
        "v2_only_v2": v2_only_v2,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    args = parser.parse_args()
    r = run_pass_b(args.corpus)
    print("PASS B — Validator-code vs Phase 7 labels agreement")
    print(f"rows scanned                       : {r['rows']}")
    print()
    print(f"V1 agreement                       : {r['v1_agreement']:.3%}")
    print(f"  Phase 7 V1 positive count        : {r['v1_truth_positive']}")
    print(f"  v2-code V1 positive count        : {r['v1_v2_positive']}")
    print(f"  Phase 7 says fire, v2 doesn't    : {r['v1_only_truth']}")
    print(f"  v2 says fire, Phase 7 doesn't    : {r['v1_only_v2']}")
    print()
    print(f"V2 agreement                       : {r['v2_agreement']:.3%}")
    print(f"  Phase 7 V2 positive count        : {r['v2_truth_positive']}")
    print(f"  v2-code V2 positive count        : {r['v2_v2_positive']}")
    print(f"  Phase 7 says fire, v2 doesn't    : {r['v2_only_truth']}")
    print(f"  v2 says fire, Phase 7 doesn't    : {r['v2_only_v2']}")
    print()
    print(f"Targets (user-confirmed): V1 >= 95.0%, V2 >= 90.0%")


if __name__ == "__main__":
    main()
