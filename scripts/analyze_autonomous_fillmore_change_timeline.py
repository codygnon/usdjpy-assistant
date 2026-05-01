#!/usr/bin/env python3
"""Build a change/performance timeline for Autonomous Fillmore.

This script intentionally separates two clocks:
1. code-change time from git commits touching Fillmore/autonomous modules;
2. behavior time from deployed AI suggestion rows and closed trade outcomes.

The goal is to show what changed, when it became visible in suggestions, and
how performance behaved between those changes.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "research_out" / "autonomous_fillmore_change_timeline_20260430"
LOCAL_TZ = ZoneInfo("America/Toronto")

DEFAULT_INPUTS = [
    Path("/tmp/newera8_ai_history_timeline.json"),
    Path("/tmp/newera8_ai_history_timeline_offset500.json"),
    Path("/tmp/kumatora2_ai_history_timeline.json"),
]

RELEVANT_GIT_PATHS = [
    "api/autonomous_fillmore.py",
    "api/ai_trading_chat.py",
    "api/suggestion_schema.py",
    "api/suggestion_tracker.py",
    "api/autonomous_performance.py",
    "scripts/export_autonomous_fillmore_evidence.py",
    "scripts/analyze_autonomous_fillmore_performance_investigation.py",
    "tests/test_autonomous_fillmore.py",
]

PROMPT_LABELS = {
    "blank": "early autonomous/no prompt version",
    "autonomous_phase_a_v1": "Phase A autonomous baseline",
    "autonomous_phase2_zone_memory_custom_exit_v1": "Phase 2 zone memory + custom exits",
    "autonomous_phase2_runner_custom_exit_v3": "Phase 2 runner/custom-exit v3",
    "autonomous_phase3_house_edge_v1": "Phase 3 house-edge prompt rewrite",
}


def parse_dt(raw: Any) -> datetime | None:
    if not raw:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def to_float(raw: Any) -> float | None:
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def load_items(paths: list[Path]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"missing input JSON: {path}")
        data = json.loads(path.read_text())
        rows = data.get("items") if isinstance(data, dict) else data
        if not isinstance(rows, list):
            raise RuntimeError(f"input {path} does not contain an items list")
        for row in rows:
            if not isinstance(row, dict):
                continue
            key = str(row.get("suggestion_id") or row.get("trade_id") or id(row))
            by_id[key] = row
    return list(by_id.values())


def is_autonomous_fillmore(row: dict[str, Any]) -> bool:
    placed = row.get("placed_order") if isinstance(row.get("placed_order"), dict) else {}
    return (
        str(row.get("prompt_version") or "").startswith("autonomous")
        or str(row.get("trade_id") or "").startswith("ai_autonomous:")
        or str(row.get("entry_type") or "").startswith("ai_autonomous")
        or placed.get("autonomous") is True
        or str(row.get("rationale") or "").startswith("AUTONOMOUS")
    )


GENERIC_TEXT = {"", "n/a", "na", "none", "null", "default", "-", "see thesis", "see analysis"}


def has_weakness(row: dict[str, Any]) -> bool:
    rr = to_float(row.get("planned_rr_estimate"))
    why = str(row.get("why_trade_despite_weakness") or "").strip().lower()
    return (
        row.get("timeframe_alignment") in {"mixed", "countertrend"}
        or row.get("repeat_trade_case") == "blind_retry"
        or row.get("zone_memory_read") in {"failing_zone", "unresolved_chop"}
        or (rr is not None and rr < 1.0)
        or why not in GENERIC_TEXT
    )


def is_placed(row: dict[str, Any]) -> bool:
    return str(row.get("action") or "").lower() == "placed"


def is_skip(row: dict[str, Any]) -> bool:
    return str(row.get("decision") or "").lower() == "skip" or str(row.get("action") or "").lower() == "skip"


def closed_with_pips(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if is_placed(row) and row.get("closed_at") and to_float(row.get("pips")) is not None
    ]


def metric_row(label: str, rows: list[dict[str, Any]], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    placed = [r for r in rows if is_placed(r)]
    skips = [r for r in rows if is_skip(r)]
    closed = closed_with_pips(rows)
    pips = [to_float(r.get("pips")) for r in closed]
    pnl = [to_float(r.get("pnl")) for r in closed]
    pips_f = [x for x in pips if x is not None]
    pnl_f = [x for x in pnl if x is not None]
    wins = [r for r in closed if (to_float(r.get("pips")) or 0.0) > 0]
    losses = [r for r in closed if (to_float(r.get("pips")) or 0.0) < 0]
    buys = [r for r in closed if str(r.get("side") or "").lower() == "buy"]
    sells = [r for r in closed if str(r.get("side") or "").lower() == "sell"]
    mixed = [r for r in closed if r.get("timeframe_alignment") == "mixed"]
    sell_weak = [r for r in closed if str(r.get("side") or "").lower() == "sell" and has_weakness(r)]
    big_lot = [r for r in closed if (to_float(r.get("lots")) or 0.0) >= 8.0]
    families = Counter(str(r.get("trigger_family") or "(blank)") for r in closed)
    out = {
        "label": label,
        "calls": len(rows),
        "placed": len(placed),
        "placement_rate": round(len(placed) / len(rows), 4) if rows else "",
        "skips": len(skips),
        "closed_with_pips": len(closed),
        "win_rate": round(len(wins) / len(closed), 4) if closed else "",
        "net_pips": round(sum(pips_f), 1) if pips_f else "",
        "avg_pips": round(sum(pips_f) / len(pips_f), 2) if pips_f else "",
        "net_pnl": round(sum(pnl_f), 2) if pnl_f else "",
        "avg_pnl": round(sum(pnl_f) / len(pnl_f), 2) if pnl_f else "",
        "loss_count": len(losses),
        "buy_closed": len(buys),
        "buy_net_pips": round(sum(to_float(r.get("pips")) or 0.0 for r in buys), 1) if buys else "",
        "sell_closed": len(sells),
        "sell_net_pips": round(sum(to_float(r.get("pips")) or 0.0 for r in sells), 1) if sells else "",
        "mixed_closed": len(mixed),
        "mixed_net_pips": round(sum(to_float(r.get("pips")) or 0.0 for r in mixed), 1) if mixed else "",
        "sell_weak_closed": len(sell_weak),
        "sell_weak_net_pips": round(sum(to_float(r.get("pips")) or 0.0 for r in sell_weak), 1) if sell_weak else "",
        "large_lot_closed": len(big_lot),
        "large_lot_net_pnl": round(sum(to_float(r.get("pnl")) or 0.0 for r in big_lot), 2) if big_lot else "",
        "top_family": families.most_common(1)[0][0] if families else "",
    }
    if extra:
        out.update(extra)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def git_commits() -> list[dict[str, Any]]:
    cmd = [
        "git", "log", "--date=iso-strict",
        "--pretty=format:%H%x1f%h%x1f%cI%x1f%s",
        "--",
        *RELEVANT_GIT_PATHS,
    ]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=True)
    commits: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        parts = line.split("\x1f")
        if len(parts) != 4:
            continue
        full, short, committed, subject = parts
        dt = parse_dt(committed)
        if not dt or dt < datetime(2026, 4, 12, tzinfo=timezone.utc):
            continue
        name_proc = subprocess.run(
            ["git", "show", "--name-only", "--format=", full, "--", *RELEVANT_GIT_PATHS],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        files = [p for p in name_proc.stdout.splitlines() if p.strip()]
        commits.append({
            "commit": short,
            "commit_full": full,
            "committed_utc": dt.isoformat(),
            "committed_local": dt.astimezone(LOCAL_TZ).isoformat(),
            "subject": subject,
            "files": ";".join(files),
        })
    commits.sort(key=lambda row: row["committed_utc"])
    return commits


def build_prompt_version_performance(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("prompt_version") or "blank")].append(row)
    out = []
    for version, group in sorted(grouped.items(), key=lambda kv: min(str(r.get("created_utc") or "") for r in kv[1])):
        times = [parse_dt(r.get("created_utc")) for r in group]
        times_f = [t for t in times if t]
        extra = {
            "prompt_version": version,
            "epoch_name": PROMPT_LABELS.get(version, version),
            "first_created_utc": min(times_f).isoformat() if times_f else "",
            "last_created_utc": max(times_f).isoformat() if times_f else "",
            "first_created_local": min(times_f).astimezone(LOCAL_TZ).isoformat() if times_f else "",
            "last_created_local": max(times_f).astimezone(LOCAL_TZ).isoformat() if times_f else "",
        }
        out.append(metric_row(PROMPT_LABELS.get(version, version), group, extra))
    return out


def build_daily_performance(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        dt = parse_dt(row.get("created_utc"))
        if not dt:
            continue
        day = dt.astimezone(LOCAL_TZ).date().isoformat()
        version = str(row.get("prompt_version") or "blank")
        grouped[(day, version)].append(row)
    out = []
    for (day, version), group in sorted(grouped.items()):
        out.append(metric_row(
            f"{day} {version}",
            group,
            {"local_date": day, "prompt_version": version, "epoch_name": PROMPT_LABELS.get(version, version)},
        ))
    return out


def build_commit_interval_performance(
    rows: list[dict[str, Any]],
    commits: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out = []
    parsed = [(parse_dt(c["committed_utc"]), c) for c in commits]
    parsed = [(dt, c) for dt, c in parsed if dt]
    for idx, (start, commit) in enumerate(parsed):
        end = parsed[idx + 1][0] if idx + 1 < len(parsed) else None
        group = []
        for row in rows:
            created = parse_dt(row.get("created_utc"))
            if not created:
                continue
            if created >= start and (end is None or created < end):
                group.append(row)
        extra = {
            "commit": commit["commit"],
            "subject": commit["subject"],
            "period_start_local": start.astimezone(LOCAL_TZ).isoformat(),
            "period_end_local": end.astimezone(LOCAL_TZ).isoformat() if end else "",
        }
        out.append(metric_row(commit["subject"], group, extra))
    return out


def build_worst_trades(rows: list[dict[str, Any]], limit: int = 40) -> list[dict[str, Any]]:
    closed = closed_with_pips(rows)
    closed.sort(key=lambda r: to_float(r.get("pnl")) if to_float(r.get("pnl")) is not None else 0.0)
    out = []
    for row in closed[:limit]:
        created = parse_dt(row.get("created_utc"))
        out.append({
            "created_local": created.astimezone(LOCAL_TZ).isoformat() if created else "",
            "profile": row.get("profile"),
            "prompt_version": row.get("prompt_version") or "blank",
            "side": row.get("side"),
            "lots": row.get("lots"),
            "pips": row.get("pips"),
            "pnl": row.get("pnl"),
            "trigger_family": row.get("trigger_family"),
            "trigger_reason": row.get("trigger_reason"),
            "timeframe_alignment": row.get("timeframe_alignment"),
            "zone_memory_read": row.get("zone_memory_read"),
            "planned_rr_estimate": row.get("planned_rr_estimate"),
            "named_catalyst": row.get("named_catalyst"),
            "trade_id": row.get("trade_id"),
        })
    return out


def build_report(
    out_dir: Path,
    rows: list[dict[str, Any]],
    version_rows: list[dict[str, Any]],
    daily_rows: list[dict[str, Any]],
    commits: list[dict[str, Any]],
) -> str:
    total = metric_row("all autonomous fillmore", rows)
    latest = version_rows[-1] if version_rows else {}
    lines = [
        "# Autonomous Fillmore Change Timeline Investigation",
        "",
        "This pass links major Fillmore code/prompt changes to realized autonomous trade performance.",
        "Performance is grouped by the deployed `prompt_version` first, because that is the cleanest behavioral boundary in the live suggestions table. Commit intervals are also exported for audit.",
        "",
        "## Headline",
        "",
        f"- Autonomous rows analyzed: {total['calls']} calls, {total['placed']} placed, {total['closed_with_pips']} pips-counted closes.",
        f"- Total pips/P&L across analyzed autonomous closes: {total['net_pips']}p / ${total['net_pnl']}.",
        f"- Latest prompt version: `{latest.get('prompt_version', '')}` from {latest.get('first_created_local', '')} to {latest.get('last_created_local', '')}.",
        f"- Latest prompt performance: {latest.get('closed_with_pips', '')} closes, WR {pct(latest.get('win_rate'))}, {latest.get('net_pips', '')}p / ${latest.get('net_pnl', '')}.",
        "",
        "## Prompt-Version Performance",
        "",
        "| Prompt / epoch | Local active window | Calls | Placed | Place rate | Closed | WR | Net pips | Net P&L | Buy pips | Sell pips | Mixed pips | Large-lot P&L |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in version_rows:
        window = f"{short_time(row.get('first_created_local'))} -> {short_time(row.get('last_created_local'))}"
        lines.append(
            f"| {row['epoch_name']} | {window} | {row['calls']} | {row['placed']} | {pct(row['placement_rate'])} | "
            f"{row['closed_with_pips']} | {pct(row['win_rate'])} | {row['net_pips']} | ${row['net_pnl']} | "
            f"{row['buy_net_pips']} | {row['sell_net_pips']} | "
            f"{row['mixed_net_pips']} | ${row['large_lot_net_pnl']} |"
        )
    lines.extend([
        "",
        "## What The Timeline Says So Far",
        "",
        "- The problem persisted through every prompt epoch. No deployed version has produced positive total pips in the current evidence set.",
        "- The Apr 29 house-edge rewrite improved win rate on paper, but still lost money and pips because losers were much larger and high-lot losses dominated.",
        "- The house-edge rewrite also increased placement density: 73 placed from 95 calls (76.8%). That is not selective enough for a prompt meant to raise the bar.",
        "- Sell-side weakness improved versus the Apr 29 runner prompt, but it did not disappear. Phase 3 still has 23 closed sell+weakness trades for -26.4p / -$447.81.",
        "- Mixed alignment was not actually killed by Phase 3. It became high win-rate but still economically bad: 53 mixed closes, -4.2p / -$1,157.23.",
        "- Large lots are a recurring damage amplifier. In Phase 3, the worst two losses were 8-lot trades for about -$1,330 combined.",
        "",
        "## Major Change Dates From Git",
        "",
        "| Local commit time | Commit | Subject |",
        "|---|---:|---|",
    ])
    for c in commits:
        lines.append(f"| {short_time(c['committed_local'])} | `{c['commit']}` | {c['subject']} |")
    lines.extend([
        "",
        "## Daily Performance By Prompt Version",
        "",
        "| Local date | Prompt version | Closed | WR | Net pips | Net P&L | Buy/Sell closed | Mixed pips | Large-lot P&L |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in daily_rows:
        lines.append(
            f"| {row['local_date']} | `{row['prompt_version']}` | {row['closed_with_pips']} | {pct(row['win_rate'])} | "
            f"{row['net_pips']} | ${row['net_pnl']} | {row['buy_closed']}/{row['sell_closed']} | "
            f"{row['mixed_net_pips']} | ${row['large_lot_net_pnl']} |"
        )
    lines.extend([
        "",
        "## Files",
        "",
        f"- `{out_dir / 'major_change_log.csv'}`",
        f"- `{out_dir / 'prompt_version_performance.csv'}`",
        f"- `{out_dir / 'daily_prompt_performance.csv'}`",
        f"- `{out_dir / 'commit_interval_performance.csv'}`",
        f"- `{out_dir / 'worst_trades.csv'}`",
        "",
        "## Caveats",
        "",
        "- Commit time is not always deploy time. Prompt-version timestamps are the stronger behavior boundary.",
        "- Some early autonomous rows have blank prompt versions but are marked autonomous in `placed_order`.",
        "- This report uses suggestion-history outcomes because they carry prompt/version/reasoning fields; trade-history rows are used as a cross-check source, not the primary grouping key.",
    ])
    report = "\n".join(lines) + "\n"
    (out_dir / "REPORT.md").write_text(report)
    return report


def pct(value: Any) -> str:
    if value == "" or value is None:
        return ""
    return f"{float(value) * 100:.1f}%"


def short_time(value: Any) -> str:
    if not value:
        return ""
    dt = parse_dt(value)
    if not dt:
        return str(value)
    return dt.astimezone(LOCAL_TZ).strftime("%b %-d %H:%M")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("inputs", nargs="*", default=[str(p) for p in DEFAULT_INPUTS])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = [Path(p) for p in args.inputs]

    raw_rows = load_items(inputs)
    rows = [r for r in raw_rows if is_autonomous_fillmore(r)]
    rows.sort(key=lambda r: parse_dt(r.get("created_utc")) or datetime.min.replace(tzinfo=timezone.utc))

    commits = git_commits()
    version_rows = build_prompt_version_performance(rows)
    daily_rows = build_daily_performance(rows)
    interval_rows = build_commit_interval_performance(rows, commits)
    worst_rows = build_worst_trades(rows)

    write_csv(out_dir / "major_change_log.csv", commits)
    write_csv(out_dir / "prompt_version_performance.csv", version_rows)
    write_csv(out_dir / "daily_prompt_performance.csv", daily_rows)
    write_csv(out_dir / "commit_interval_performance.csv", interval_rows)
    write_csv(out_dir / "worst_trades.csv", worst_rows)
    (out_dir / "source_manifest.json").write_text(json.dumps({
        "inputs": [str(p) for p in inputs],
        "raw_rows": len(raw_rows),
        "autonomous_rows": len(rows),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    report = build_report(out_dir, rows, version_rows, daily_rows, commits)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
