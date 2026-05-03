#!/usr/bin/env python3
"""Read-only first-tick readiness check for Auto Fillmore v2.

Reports the latest v2 row in ai_suggestions and fails non-zero when required
Stage 1 audit fields are missing. This is intentionally SQL-only: it does not
import the live loop or mutate runtime state.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Optional


REQUIRED_COLUMNS = (
    "engine_version",
    "snapshot_version",
    "snapshot_schema_hash",
    "prompt_version",
    "rendered_prompt",
    "rendered_context_json",
    "open_lots_buy",
    "open_lots_sell",
    "unrealized_pnl_buy",
    "unrealized_pnl_sell",
    "pip_value_per_lot",
    "risk_after_fill_usd",
    "rolling_20_trade_pnl",
    "rolling_20_lot_weighted_pnl",
    "gate_candidates_json",
    "sizing_inputs_json",
    "deterministic_lots",
)

OPTIONAL_AUDIT_COLUMNS = (
    "level_packet_json",
    "level_age_metadata_json",
    "validator_overrides_json",
    "pre_veto_fired_json",
    "halt_reason",
    "trigger_family",
    "decision",
    "skip_reason",
    "rationale",
)


def _json_count(raw: Any) -> Optional[int]:
    if raw in (None, ""):
        return None
    try:
        parsed = json.loads(raw)
    except Exception:
        return None
    if isinstance(parsed, list):
        return len(parsed)
    if isinstance(parsed, dict):
        return len(parsed)
    return None


def _missing_value(row: sqlite3.Row, col: str) -> bool:
    value = row[col]
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", required=True, type=Path, help="Path to ai_suggestions.sqlite")
    p.add_argument("--profile", default=None, help="Optional profile filter")
    p.add_argument("--limit", type=int, default=1, help="How many latest v2 rows to print")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.db.exists():
        print(f"FAIL db_missing path={args.db}")
        return 2

    with sqlite3.connect(str(args.db)) as conn:
        conn.row_factory = sqlite3.Row
        cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(ai_suggestions)").fetchall()}
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in cols]
        if missing_cols:
            print(f"FAIL missing_columns={missing_cols}")
            return 2

        where = "engine_version = 'v2'"
        params: list[Any] = []
        if args.profile:
            where += " AND profile = ?"
            params.append(args.profile)
        rows = conn.execute(
            f"""
            SELECT *
            FROM ai_suggestions
            WHERE {where}
            ORDER BY created_utc DESC
            LIMIT ?
            """,
            (*params, args.limit),
        ).fetchall()

    if not rows:
        print("FAIL no_v2_rows")
        return 1

    bad = False
    for idx, row in enumerate(rows, start=1):
        missing_values = [c for c in REQUIRED_COLUMNS if _missing_value(row, c)]
        bad = bad or bool(missing_values)
        decision = row["decision"] if "decision" in row.keys() else None
        skip_reason = row["skip_reason"] if "skip_reason" in row.keys() else None
        rationale = row["rationale"] if "rationale" in row.keys() else None
        if skip_reason and str(skip_reason).startswith("parse_failure"):
            parse_status = "parse_failure"
        elif rationale:
            parse_status = "parsed_or_model_skip"
        else:
            parse_status = "not_called"

        print(f"--- latest_v2_row[{idx}] ---")
        print(f"suggestion_id={row['suggestion_id']}")
        print(f"created_utc={row['created_utc']}")
        print(f"profile={row['profile']}")
        print(f"final_decision={decision}")
        print(f"selected_gate={row['trigger_family'] if 'trigger_family' in row.keys() else None}")
        print(f"halt_reason={row['halt_reason'] if 'halt_reason' in row.keys() else None}")
        print(f"skip_reason={skip_reason}")
        print(f"llm_parse_status={parse_status}")
        print(f"gate_candidates_count={_json_count(row['gate_candidates_json'])}")
        if "pre_veto_fired_json" in row.keys():
            print(f"pre_veto_fires_count={_json_count(row['pre_veto_fired_json']) or 0}")
        if "validator_overrides_json" in row.keys():
            print(f"validator_overrides_count={_json_count(row['validator_overrides_json']) or 0}")
        print(f"deterministic_lots={row['deterministic_lots']}")
        print(f"risk_after_fill_usd={row['risk_after_fill_usd']}")
        print(f"missing_required_values={missing_values}")

        for col in OPTIONAL_AUDIT_COLUMNS:
            if col not in row.keys():
                print(f"missing_optional_column={col}")

    if bad:
        print("FAIL latest v2 row is missing required Stage 1 fields")
        return 1
    print("PASS latest v2 row has required Stage 1 fields")
    return 0


if __name__ == "__main__":
    sys.exit(main())
