#!/usr/bin/env python3
"""Pass A dry-run for Fillmore v2 Step 1.

This is intentionally not validator code. It loads the Phase 7 labeled
241-trade corpus, writes only historical/base ai_suggestions fields into a
temporary SQLite DB, runs the v2 additive migration, and recomputes the Phase 7
V1+V2 diagnostic floor from the migrated rows plus the existing labels.
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api import suggestion_tracker  # noqa: E402
from api.fillmore_v2 import persistence  # noqa: E402

DEFAULT_CORPUS = ROOT / "research_out" / "autonomous_fillmore_forensic_20260501" / "phase7_interaction_dataset.csv"

EXPECTED_TRADES = 241
EXPECTED_BASELINE_PIPS = -308.0
EXPECTED_BASELINE_USD = -7253.2365
EXPECTED_BLOCKED = 110
EXPECTED_BLOCKED_WINNERS = 52
EXPECTED_BLOCKED_LOSERS = 58
EXPECTED_NET_DELTA_PIPS = 300.5
EXPECTED_NET_DELTA_USD = 5684.56

V1_CLUSTERS = {"momentum_with_caveat_trade", "critical_level_mixed_caveat_trade"}
V2_SESSIONS = {"london/ny overlap", "tokyo/london overlap"}


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    raw = row.get(key, "")
    if raw in ("", "nan", "NaN", None):
        return default
    return float(raw)


def _insert_historical_rows(conn: sqlite3.Connection, rows: list[dict[str, str]]) -> None:
    conn.execute(
        """
        CREATE TABLE phase7_labels (
            suggestion_id TEXT PRIMARY KEY,
            rationale_cluster TEXT,
            timeframe_alignment_clean TEXT,
            session TEXT
        )
        """
    )
    for row in rows:
        suggestion_id = row["suggestion_id"]
        conn.execute(
            """
            INSERT INTO ai_suggestions (
                suggestion_id, created_utc, profile, model,
                side, limit_price, lots, rationale, prompt_version,
                trigger_family, entry_type, closed_at, pnl, pips, win_loss
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                suggestion_id,
                row["created_utc"],
                row["profile"],
                "phase7_historical",
                row["side"],
                0.0,
                _float(row, "lots"),
                row.get("rationale") or None,
                row.get("prompt_version") or None,
                row.get("trigger_family") or None,
                "ai_autonomous",
                row.get("closed_at") or None,
                _float(row, "pnl"),
                _float(row, "pips"),
                row.get("win_loss") or None,
            ),
        )
        conn.execute(
            """
            INSERT INTO phase7_labels (
                suggestion_id, rationale_cluster, timeframe_alignment_clean, session
            ) VALUES (?, ?, ?, ?)
            """,
            (
                suggestion_id,
                row.get("rationale_cluster") or "",
                row.get("timeframe_alignment_clean") or "",
                row.get("session") or "",
            ),
        )
    conn.commit()


def _assert_close(name: str, actual: float, expected: float, places: int) -> None:
    if round(actual, places) != round(expected, places):
        raise AssertionError(f"{name}: expected {expected}, got {actual}")


def run_pass_a(corpus_path: Path = DEFAULT_CORPUS) -> dict[str, float | int]:
    with corpus_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != EXPECTED_TRADES:
        raise AssertionError(f"expected {EXPECTED_TRADES} corpus rows, got {len(rows)}")

    with tempfile.TemporaryDirectory(prefix="fillmore_v2_pass_a_") as td:
        db_path = Path(td) / "ai_suggestions.sqlite"
        suggestion_tracker.init_db(db_path)
        with sqlite3.connect(str(db_path)) as conn:
            _insert_historical_rows(conn, rows)

        persistence.init_v2_schema(db_path)

        with sqlite3.connect(str(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            migrated = conn.execute(
                """
                SELECT s.suggestion_id, s.side, s.pips, s.pnl, s.engine_version,
                       s.snapshot_version, s.snapshot_schema_hash,
                       l.rationale_cluster, l.timeframe_alignment_clean, l.session
                FROM ai_suggestions s
                JOIN phase7_labels l USING (suggestion_id)
                ORDER BY s.created_utc, s.suggestion_id
                """
            ).fetchall()
            non_null_v2 = conn.execute(
                """
                SELECT COUNT(*) AS n
                FROM ai_suggestions
                WHERE engine_version IS NOT NULL
                   OR snapshot_version IS NOT NULL
                   OR snapshot_schema_hash IS NOT NULL
                """
            ).fetchone()["n"]

    if len(migrated) != EXPECTED_TRADES:
        raise AssertionError(f"expected {EXPECTED_TRADES} migrated rows, got {len(migrated)}")
    if non_null_v2:
        raise AssertionError(f"pre-cutover rows should keep v2 identity columns NULL, got {non_null_v2}")

    baseline_pips = sum(float(r["pips"]) for r in migrated)
    baseline_usd = sum(float(r["pnl"]) for r in migrated)
    _assert_close("baseline pips", baseline_pips, EXPECTED_BASELINE_PIPS, 1)
    _assert_close("baseline usd", baseline_usd, EXPECTED_BASELINE_USD, 4)

    blocked = []
    for r in migrated:
        v1 = r["rationale_cluster"] in V1_CLUSTERS and r["side"] == "sell"
        v2 = r["timeframe_alignment_clean"] == "mixed" and r["session"] in V2_SESSIONS
        if v1 or v2:
            blocked.append(r)

    blocked_winners = [r for r in blocked if float(r["pips"]) > 0]
    blocked_losers = [r for r in blocked if float(r["pips"]) < 0]
    saved_loser_pips = -sum(float(r["pips"]) for r in blocked_losers)
    saved_loser_usd = -sum(float(r["pnl"]) for r in blocked_losers)
    missed_winner_pips = sum(float(r["pips"]) for r in blocked_winners)
    missed_winner_usd = sum(float(r["pnl"]) for r in blocked_winners)
    net_delta_pips = saved_loser_pips - missed_winner_pips
    net_delta_usd = saved_loser_usd - missed_winner_usd

    if len(blocked) != EXPECTED_BLOCKED:
        raise AssertionError(f"blocked trades: expected {EXPECTED_BLOCKED}, got {len(blocked)}")
    if len(blocked_winners) != EXPECTED_BLOCKED_WINNERS:
        raise AssertionError(f"blocked winners: expected {EXPECTED_BLOCKED_WINNERS}, got {len(blocked_winners)}")
    if len(blocked_losers) != EXPECTED_BLOCKED_LOSERS:
        raise AssertionError(f"blocked losers: expected {EXPECTED_BLOCKED_LOSERS}, got {len(blocked_losers)}")
    _assert_close("net delta pips", net_delta_pips, EXPECTED_NET_DELTA_PIPS, 1)
    _assert_close("net delta usd", net_delta_usd, EXPECTED_NET_DELTA_USD, 2)

    return {
        "rows": len(migrated),
        "baseline_pips": baseline_pips,
        "baseline_usd": baseline_usd,
        "blocked_trades": len(blocked),
        "blocked_winners": len(blocked_winners),
        "blocked_losers": len(blocked_losers),
        "net_delta_pips": net_delta_pips,
        "net_delta_usd": net_delta_usd,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    args = parser.parse_args()
    result = run_pass_a(args.corpus)
    print("PASS A dry-run OK")
    print(f"rows={result['rows']}")
    print(f"baseline={result['baseline_pips']:.1f}p / ${result['baseline_usd']:,.4f}")
    print(
        "V1+V2 floor="
        f"+{result['net_delta_pips']:.1f}p / +${result['net_delta_usd']:,.2f} "
        f"({result['blocked_trades']} blocked, "
        f"{result['blocked_winners']} winners, {result['blocked_losers']} losers)"
    )


if __name__ == "__main__":
    main()
