#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path


def _parse_line(line: str) -> dict[str, str]:
    out: dict[str, str] = {}
    parts = line.rstrip("\n").split("\t")
    if parts:
        out["ts"] = parts[0]
    for part in parts[1:]:
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _parse_filters(raw: str) -> list[str]:
    s = raw.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    if not s:
        return []
    vals: list[str] = []
    for item in s.split(";"):
        item = item.strip()
        if not item:
            continue
        name = item.split(":", 1)[0].strip()
        vals.append(name)
    return vals


def _norm_filter_name(name: str) -> str:
    n = name.strip().lower()
    n = re.sub(r"[^a-z0-9]+", "_", n).strip("_")
    aliases = {
        "active_session": "active_session",
        "session": "active_session",
        "allowed_to_trade": "allowed_to_trade",
        "active_strategy": "active_strategy",
        "spread": "spread",
    }
    return aliases.get(n, n)


def _strategy_from_row(row: dict[str, str]) -> str:
    tag = (row.get("strategy") or "").strip()
    if tag:
        return tag
    reason = (row.get("reason") or "").strip("'\" ")
    if reason.startswith("london_v2"):
        return "phase3:london_v2"
    if reason.startswith("v44_ny"):
        return "phase3:v44_ny"
    if reason.startswith("phase3:"):
        return "phase3:v14"
    return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze phase3 minute diagnostics blockers.")
    ap.add_argument("--log", required=True, help="Absolute path to phase3_minute_diagnostics.log")
    ap.add_argument(
        "--include-nonsession",
        action="store_true",
        help="Include out-of-session rows (by default only session-valid rows are analyzed).",
    )
    args = ap.parse_args()

    p = Path(args.log)
    if not p.exists():
        raise SystemExit(f"log file not found: {p}")

    rows: list[dict[str, str]] = []
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = _parse_line(line)
            row["_filters"] = ",".join(_parse_filters(row.get("filters", "")))
            rows.append(row)

    if not rows:
        raise SystemExit("no rows parsed")

    total_rows = len(rows)
    session_valid_rows: list[dict[str, str]] = []
    for r in rows:
        sess = (r.get("session") or "").strip().lower()
        raw_filters = [x for x in r.get("_filters", "").split(",") if x]
        nf = [_norm_filter_name(x) for x in raw_filters]
        if sess in {"", "none"}:
            continue
        if "active_session" in nf:
            continue
        session_valid_rows.append(r)

    use_rows = rows if args.include_nonsession else session_valid_rows
    if not use_rows:
        raise SystemExit("no session-valid rows after filtering")

    overall = Counter()
    by_session = defaultdict(Counter)
    by_session_strategy = defaultdict(Counter)
    row_count_by_session = Counter()
    row_count_by_session_strategy = Counter()

    for r in use_rows:
        sess = (r.get("session") or "none").strip().lower()
        strat = _strategy_from_row(r)
        raw_filters = [x for x in r.get("_filters", "").split(",") if x]
        nf = [_norm_filter_name(x) for x in raw_filters]
        row_count_by_session[sess] += 1
        row_count_by_session_strategy[(sess, strat)] += 1
        for name in nf:
            overall[name] += 1
            by_session[sess][name] += 1
            by_session_strategy[(sess, strat)][name] += 1

    print(f"log: {p}")
    print(f"rows_total: {total_rows}")
    print(f"rows_session_valid: {len(session_valid_rows)}")
    print("")
    print("overall_top_blockers (session-valid only):")
    for k, v in overall.most_common(20):
        print(f"  {k}: {v}")

    print("")
    print("by_session:")
    for sess in sorted(by_session.keys()):
        print(f"  {sess} (rows={row_count_by_session[sess]}):")
        for k, v in by_session[sess].most_common(12):
            print(f"    {k}: {v}")

    print("")
    print("by_session_and_strategy:")
    for key in sorted(by_session_strategy.keys()):
        sess, strat = key
        print(f"  {sess} | {strat} (rows={row_count_by_session_strategy[key]}):")
        for k, v in by_session_strategy[key].most_common(10):
            print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
