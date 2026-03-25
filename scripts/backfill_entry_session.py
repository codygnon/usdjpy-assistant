#!/usr/bin/env python3
"""One-time backfill: derive entry_session from timestamp_utc for Phase 3 trades.

Uses the actual V14/V2/V44 session windows (DST-aware) rather than generic
Tokyo/London/NY ranges.

Usage:
    python3 scripts/backfill_entry_session.py          # dry-run (default)
    python3 scripts/backfill_entry_session.py --apply   # actually update DB
"""
import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# DST helpers (copied from core/phase3_integrated_engine.py)
# ---------------------------------------------------------------------------

def last_sunday(year: int, month: int) -> pd.Timestamp:
    d = pd.Timestamp(year=year, month=month, day=1, tz="UTC") + pd.offsets.MonthEnd(0)
    while d.weekday() != 6:
        d -= pd.Timedelta(days=1)
    return d


def nth_sunday(year: int, month: int, n: int) -> pd.Timestamp:
    d = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    while d.weekday() != 6:
        d += pd.Timedelta(days=1)
    d += pd.Timedelta(days=(n - 1) * 7)
    return d


def uk_london_open_utc(ts: pd.Timestamp) -> int:
    y = ts.year
    summer_start = last_sunday(y, 3).normalize()
    summer_end = last_sunday(y, 10).normalize()
    d = ts.normalize()
    return 7 if summer_start <= d < summer_end else 8


def us_ny_open_utc(ts: pd.Timestamp) -> int:
    y = ts.year
    summer_start = nth_sunday(y, 3, 2).normalize()
    summer_end = nth_sunday(y, 11, 1).normalize()
    d = ts.normalize()
    return 12 if summer_start <= d < summer_end else 13


# ---------------------------------------------------------------------------
# Session classification using actual strategy windows
# ---------------------------------------------------------------------------
# V14 (Tokyo):  16:00 - 22:00 UTC fixed, allowed days: Tue/Wed/Fri
# V2  (London): london_open .. london_open+4h, allowed days: Tue/Wed
# V44 (NY):     ny_open .. ny_open+3h, allowed days: Mon-Fri

TOKYO_START = 16
TOKYO_END = 22
TOKYO_DAYS = {1, 2, 4}       # Tue, Wed, Fri

LONDON_DURATION_H = 4
LONDON_DAYS = {1, 2}         # Tue, Wed

NY_DURATION_H = 3
NY_DAYS = {0, 1, 2, 3, 4}   # Mon-Fri


def classify_session(ts: pd.Timestamp) -> str | None:
    """Return 'tokyo', 'london', or 'ny' based on the actual V14/V2/V44 windows."""
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    hour = ts.hour
    weekday = ts.weekday()

    # V14 Tokyo: 16-22 UTC, Tue/Wed/Fri
    if weekday in TOKYO_DAYS and TOKYO_START <= hour < TOKYO_END:
        return "tokyo"

    # V2 London: DST-aware, Tue/Wed
    if weekday in LONDON_DAYS:
        london_open = uk_london_open_utc(ts)
        london_close = london_open + LONDON_DURATION_H
        if london_open <= hour < london_close:
            return "london"

    # V44 NY: DST-aware, Mon-Fri
    if weekday in NY_DAYS:
        ny_open = us_ny_open_utc(ts)
        ny_close = ny_open + NY_DURATION_H
        if ny_open <= hour < ny_close:
            return "ny"

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_db_files() -> list[Path]:
    base = Path(__file__).resolve().parent.parent / "logs"
    return sorted(base.rglob("assistant.db"))


def backfill(db_path: Path, apply: bool) -> int:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Check entry_session column exists
    cols = {r[1] for r in conn.execute("PRAGMA table_info(trades)").fetchall()}
    if "entry_session" not in cols:
        print(f"  [skip] {db_path}: no entry_session column")
        conn.close()
        return 0

    rows = conn.execute(
        "SELECT rowid, trade_id, timestamp_utc, entry_type, policy_type, entry_session "
        "FROM trades WHERE policy_type = 'phase3_integrated'"
    ).fetchall()

    updated = 0
    for row in rows:
        ts_str = row["timestamp_utc"]
        existing = row["entry_session"]
        if existing and existing in ("tokyo", "london", "ny"):
            continue  # already populated

        try:
            ts = pd.Timestamp(ts_str)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
        except Exception:
            continue

        session = classify_session(ts)

        # Fallback: infer from entry_type/strategy tag if timestamp doesn't match a window
        if session is None:
            entry_type = str(row["entry_type"] or "")
            if "v14" in entry_type or "mean_reversion" in entry_type:
                session = "tokyo"
            elif "london" in entry_type or "v2" in entry_type.lower():
                session = "london"
            elif "v44" in entry_type or "ny" in entry_type.lower():
                session = "ny"

        if session is None:
            print(f"  [?] {row['trade_id']}: {ts_str} -> could not classify")
            continue

        label = f"  {'[UPDATE]' if apply else '[DRY-RUN]'}"
        print(f"{label} {row['trade_id']}: {ts_str} -> {session}")

        if apply:
            conn.execute(
                "UPDATE trades SET entry_session = ? WHERE rowid = ?",
                (session, row["rowid"]),
            )
        updated += 1

    if apply:
        conn.commit()
    conn.close()
    return updated


def main():
    parser = argparse.ArgumentParser(description="Backfill entry_session for Phase 3 trades")
    parser.add_argument("--apply", action="store_true", help="Actually write to DB (default is dry-run)")
    parser.add_argument("--db", type=str, help="Path to specific assistant.db (otherwise scans logs/)")
    args = parser.parse_args()

    if args.db:
        db_files = [Path(args.db)]
    else:
        db_files = find_db_files()

    if not db_files:
        print("No assistant.db files found in logs/. Use --db to specify a path.")
        sys.exit(1)

    total = 0
    for db_path in db_files:
        print(f"\n=== {db_path} ===")
        count = backfill(db_path, apply=args.apply)
        total += count

    mode = "UPDATED" if args.apply else "WOULD UPDATE"
    print(f"\nDone. {mode} {total} trade(s).")
    if not args.apply and total > 0:
        print("Run with --apply to commit changes.")


if __name__ == "__main__":
    main()
