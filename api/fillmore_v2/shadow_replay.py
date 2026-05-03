"""Shadow-mode replay harness for the Step 2 acceptance test.

Runs every validator against the forensic corpus rows (without making real
trade decisions). Used to:

  1. Sanity-check that validators don't crash on real-world inputs.
  2. Report fire counts per validator for acceptance gating.
  3. Step 8 Pass B reuses this with stricter agreement targets.

This module does no trading and writes no production state. It reads JSON
files under research_out/ and returns a summary dict.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .legacy_rationale_parser import adapt_corpus_row
from .pre_decision_vetoes import ALL_PRE_VETOES, run_pre_vetoes
from .validators import ALL_VALIDATORS, run_all


@dataclass
class ShadowReplaySummary:
    rows_scanned: int
    place_decisions: int
    skip_decisions: int
    fire_counts: dict[str, int]  # validator_id -> count of overrides
    fire_rates: dict[str, float]  # validator_id -> rate over place decisions
    parse_failures: int
    by_side_fires: dict[str, dict[str, int]]  # side -> validator_id -> count
    pre_veto_fire_counts: dict[str, int] = None  # type: ignore[assignment]
    pre_veto_by_side: dict[str, dict[str, int]] = None  # type: ignore[assignment]
    pre_veto_skipped_before_call: int = 0


def replay_corpus(corpus_path: Path) -> ShadowReplaySummary:
    """Walk a forensic corpus JSON file and run all validators against each row."""
    rows = json.loads(corpus_path.read_text())
    if not isinstance(rows, list):
        raise ValueError(f"expected list of rows in {corpus_path}, got {type(rows).__name__}")

    fire_counts: Counter[str] = Counter()
    by_side: dict[str, Counter[str]] = {"buy": Counter(), "sell": Counter(), "unknown": Counter()}
    pre_veto_counts: Counter[str] = Counter()
    pre_veto_by_side: dict[str, Counter[str]] = {"buy": Counter(), "sell": Counter(), "unknown": Counter()}
    pre_veto_skipped = 0
    place_n = skip_n = parse_fail = 0

    for row in rows:
        try:
            output, snapshot = adapt_corpus_row(row)
        except Exception:
            parse_fail += 1
            continue
        if output.decision == "place":
            place_n += 1
        else:
            skip_n += 1
        side_key = snapshot.proposed_side if snapshot.proposed_side in ("buy", "sell") else "unknown"

        # Pre-decision vetoes — short-circuit if any fire (mirrors live flow).
        pre_summary = run_pre_vetoes(snapshot, deterministic_lots=None)
        for r in pre_summary.fires:
            pre_veto_counts[r.veto_id] += 1
            pre_veto_by_side[side_key][r.veto_id] += 1
        if pre_summary.skip_before_call:
            pre_veto_skipped += 1
            continue  # don't run post-decision validators on pre-vetoed rows

        summary = run_all(output, snapshot)
        for r in summary.overrides:
            fire_counts[r.validator_id] += 1
            by_side[side_key][r.validator_id] += 1

    fire_rates = {
        v: (fire_counts[v] / place_n if place_n else 0.0)
        for v in (fn.__name__.replace("_validator", "") for fn in ALL_VALIDATORS)
        if v in fire_counts or True  # always report all validators, even 0
    }
    # Re-key from function name to validator_id (which is the same after stripping)
    validator_ids = [
        "caveat_resolution",
        "level_language_overreach",
        "loss_asymmetry",
        "sell_side_burden",
        "hedge_plus_overconfidence",
    ]
    fire_counts_full = {vid: fire_counts.get(vid, 0) for vid in validator_ids}
    fire_rates_full = {
        vid: (fire_counts_full[vid] / place_n if place_n else 0.0)
        for vid in validator_ids
    }
    by_side_full = {
        side: {vid: by_side[side].get(vid, 0) for vid in validator_ids}
        for side in by_side
    }

    pre_veto_ids = ["v1_sell_caveat_template", "v2_mixed_overlap"]
    pre_veto_counts_full = {vid: pre_veto_counts.get(vid, 0) for vid in pre_veto_ids}
    pre_veto_by_side_full = {
        side: {vid: pre_veto_by_side[side].get(vid, 0) for vid in pre_veto_ids}
        for side in pre_veto_by_side
    }

    return ShadowReplaySummary(
        rows_scanned=len(rows),
        place_decisions=place_n,
        skip_decisions=skip_n,
        fire_counts=fire_counts_full,
        fire_rates=fire_rates_full,
        parse_failures=parse_fail,
        by_side_fires=by_side_full,
        pre_veto_fire_counts=pre_veto_counts_full,
        pre_veto_by_side=pre_veto_by_side_full,
        pre_veto_skipped_before_call=pre_veto_skipped,
    )


def find_corpus_files(repo_root: Path) -> list[Path]:
    """Locate every autonomous-suggestions raw JSON in the evidence directory."""
    base = repo_root / "research_out" / "autonomous_fillmore_evidence_20260429"
    if not base.exists():
        return []
    return sorted(base.glob("*/ai_suggestions_autonomous_raw.json"))
