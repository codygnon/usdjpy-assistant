#!/usr/bin/env python3
"""
Paper-trading guardrail report for the frozen defensive veto pocket.

Loads research_out/defensive_paper_guardrail_profile.json, summarizes
phase3_minute_diagnostics.log (ownership_cell + reasons), and flags pause/rollback
hints from the profile. Does not change runtime behavior.

Usage:
  python3 scripts/paper_monitor_defensive_veto_pocket.py \\
    --log /path/to/phase3_minute_diagnostics.log \\
    --output-json research_out/defensive_paper_pain_report.json

Optional:
  --strict  exit 1 if audit_has_rows is false or staleness/pause flags fire
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnostic_phase3_ownership_pain_log import _parse_line, summarize_pain_log

PROFILE_PATH = ROOT / "research_out" / "defensive_paper_guardrail_profile.json"


def _log_bounds(path: Path) -> tuple[str | None, str | None, int]:
    first: str | None = None
    last: str | None = None
    n = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        n += 1
        ts = line.split("\t", 1)[0].strip()
        if first is None:
            first = ts
        last = ts
    return first, last, n


def _parse_ts(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def _count_v44_cell_rows(path: Path, target_cell: str) -> dict[str, Any]:
    """Rows where parsed ownership_cell matches and strategy looks like v44 NY."""
    n_cell = 0
    n_cell_v44 = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = _parse_line(line)
        if rec.get("ownership_cell", "") != target_cell:
            continue
        n_cell += 1
        st = (rec.get("strategy") or "").strip()
        if "v44" in st.lower():
            n_cell_v44 += 1
    return {"rows_ownership_cell": n_cell, "rows_ownership_cell_and_v44_strategy": n_cell_v44}


def evaluate_guardrails(
    profile: dict[str, Any],
    log_path: Path,
    summary: dict[str, Any],
    *,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    now = now_utc or datetime.now(timezone.utc)
    gr = profile.get("paper_guardrails") or {}
    target = (profile.get("defensive_veto_pocket") or {}).get("ownership_cell", "")
    min_lines = int(gr.get("minimum_log_lines_first_week", 1) or 1)
    stale_h = float(gr.get("staleness_hours_warning", 72) or 72)
    absence_days = int(gr.get("target_cell_absence_review_days", 5) or 5)

    first_ts_s, last_ts_s, line_count = _log_bounds(log_path)
    last_dt = _parse_ts(last_ts_s)
    hours_old: float | None = None
    staleness_warning = False
    if last_dt:
        hours_old = (now - last_dt.astimezone(timezone.utc)).total_seconds() / 3600.0
        staleness_warning = hours_old > stale_h

    audit_has_rows = line_count >= min_lines
    tc = summary.get("target_cell") or {}
    rows_cell = int(tc.get("rows_with_cell", 0) or 0)
    target_cell_seen = rows_cell > 0

    pause_recommended: list[str] = []
    if not audit_has_rows:
        pause_recommended.append("empty_diagnostics: log below minimum line count for guardrail profile")
    if staleness_warning and last_dt is not None:
        pause_recommended.append(
            f"staleness: last log line ~{hours_old:.1f}h old (threshold {stale_h}h)"
        )
    # Strong signal only: lots of bars logged but never the target cell → likely audit wiring.
    heavy_log_threshold = 2000
    if (
        line_count >= heavy_log_threshold
        and audit_has_rows
        and not target_cell_seen
        and (gr.get("target_cell_should_appear", True))
    ):
        pause_recommended.append(
            f"pocket_invisible_heavy_log: no rows with ownership_cell={target!r} "
            f"despite {line_count} log lines (threshold {heavy_log_threshold})"
        )

    review_hints: list[str] = []
    if audit_has_rows and not target_cell_seen and line_count < heavy_log_threshold:
        review_hints.append(
            f"early_paper: zero rows for ownership_cell={target!r} so far; "
            f"escalate if still true after ~{absence_days}d NY coverage (see memo)"
        )

    rollback_recommended: list[str] = []

    return {
        "evaluated_at_utc": now.isoformat(),
        "audit_has_rows": audit_has_rows,
        "log_line_count": line_count,
        "log_time_first": first_ts_s,
        "log_time_last": last_ts_s,
        "hours_since_last_log": round(hours_old, 2) if hours_old is not None else None,
        "staleness_warning": staleness_warning,
        "target_cell": target,
        "target_cell_rows": rows_cell,
        "target_cell_seen": target_cell_seen,
        "pause_recommended": pause_recommended,
        "review_hints": review_hints,
        "rollback_recommended": rollback_recommended,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Paper guardrail report for defensive veto pocket.")
    ap.add_argument("--log", required=True, type=Path, help="phase3_minute_diagnostics.log path")
    ap.add_argument("--profile", type=Path, default=PROFILE_PATH)
    ap.add_argument("--output-json", type=Path, default="")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 if audit empty or any pause_recommended entries",
    )
    args = ap.parse_args()

    if not args.log.exists():
        print(f"Missing log: {args.log}", file=sys.stderr)
        return 1
    if not args.profile.exists():
        print(f"Missing profile: {args.profile}", file=sys.stderr)
        return 1

    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    target_cell = str((profile.get("defensive_veto_pocket") or {}).get("ownership_cell") or "")

    summary = summarize_pain_log(args.log, top=25, filter_cell=target_cell or None)
    v44_counts = _count_v44_cell_rows(args.log, target_cell) if target_cell else {}
    status = evaluate_guardrails(profile, args.log, summary)

    report: dict[str, Any] = {
        "profile_path": str(args.profile.resolve()),
        "frozen_package_id": profile.get("frozen_package_id"),
        "defensive_veto_pocket": profile.get("defensive_veto_pocket"),
        "log_path": str(args.log.resolve()),
        "pain_summary": summary,
        "v44_strategy_crosscheck": v44_counts,
        "guardrail_status": status,
        "pause_triggers_reference": profile.get("pause_triggers"),
        "rollback_triggers_reference": profile.get("rollback_triggers"),
    }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {args.output_json}")

    s = status
    print(f"Log lines={s['log_line_count']} last_ts={s['log_time_last']}")
    print(f"Target cell rows={s['target_cell_rows']} ({s['target_cell']})")
    print(f"audit_has_rows={s['audit_has_rows']} staleness_warning={s['staleness_warning']}")
    if s["pause_recommended"]:
        print("PAUSE recommended:")
        for p in s["pause_recommended"]:
            print(f"  - {p}")
    else:
        print("No automated pause flags.")
    for h in s.get("review_hints") or []:
        print(f"Review hint: {h}")

    if args.strict:
        if not s["audit_has_rows"] or s["pause_recommended"]:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
