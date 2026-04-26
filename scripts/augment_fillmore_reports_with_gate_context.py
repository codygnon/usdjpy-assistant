#!/usr/bin/env python3
"""Add LLM/gate context sections to Week 1 and daily Fillmore forensic reports."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path("/Users/codygnon/Documents/usdjpy_assistant")
RESEARCH_OUT = ROOT / "research_out"
HISTORY_PATHS = {
    "newera8": Path("/tmp/newera8_ai_history_week1.json"),
    "kumatora2": Path("/tmp/kumatora2_ai_history_week1.json"),
}

WEEK_START = "2026-04-21"
WEEK_END = "2026-04-24"
DAILY_FILES = {
    "2026-04-21": RESEARCH_OUT / "autonomous_fillmore_20260421_report.md",
    "2026-04-22": RESEARCH_OUT / "autonomous_fillmore_20260422_report.md",
    "2026-04-23": RESEARCH_OUT / "autonomous_fillmore_20260423_report.md",
    "2026-04-24": RESEARCH_OUT / "autonomous_fillmore_20260424_report.md",
}
WEEK_FILE = RESEARCH_OUT / "week1_autonomous_fillmore_apr21_24_report.md"


@dataclass(frozen=True)
class Window:
    label: str
    day_start: str
    day_end: str
    report_path: Path


def _load_history(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [row for row in (payload.get("items") or []) if isinstance(row, dict)]


def _created_day(row: dict[str, Any]) -> str:
    return str(row.get("created_utc") or "")[:10]


def _is_autonomous(row: dict[str, Any]) -> bool:
    placed_order = row.get("placed_order") or {}
    if isinstance(placed_order, dict) and placed_order.get("autonomous") is True:
        return True
    if str(row.get("entry_type") or "").strip().lower() == "ai_autonomous":
        return True
    return str(row.get("rationale") or "").startswith("AUTONOMOUS")


def _in_window(row: dict[str, Any], day_start: str, day_end: str) -> bool:
    day = _created_day(row)
    return bool(day and day_start <= day <= day_end)


def _window_rows(all_rows: list[dict[str, Any]], day_start: str, day_end: str) -> list[dict[str, Any]]:
    return [row for row in all_rows if _is_autonomous(row) and _in_window(row, day_start, day_end)]


def _prompt_breakdown(rows: list[dict[str, Any]]) -> tuple[str, str]:
    version_counts = Counter(str(row.get("prompt_version") or "unknown") for row in rows)
    hash_counts = Counter(str(row.get("prompt_hash") or "unknown") for row in rows)
    version_text = ", ".join(f"`{k}` x{v}" for k, v in version_counts.most_common()) or "`none`"
    top_hashes = ", ".join(f"`{k}` x{v}" for k, v in hash_counts.most_common(6)) or "`none`"
    return version_text, top_hashes


def _build_llm_summary(rows: list[dict[str, Any]]) -> str:
    action_counts = Counter(str(row.get("action") or "null").lower() for row in rows)
    placed_rows = [row for row in rows if str(row.get("action") or "").lower() == "placed"]
    no_trade_rows = [row for row in rows if str(row.get("action") or "").lower() != "placed"]
    profile_counts = Counter(str(row.get("profile") or row.get("_profile") or "unknown") for row in rows)
    version_text, hash_text = _prompt_breakdown(rows)
    lines = [
        "## LLM Call Summary",
        "",
        f"- inferred autonomous LLM calls recorded in live suggestion history: `{len(rows)}`",
        f"- placed trades after LLM call: `{len(placed_rows)}`",
        f"- autonomous no-trade / non-placement rows recorded in this slice: `{len(no_trade_rows)}`",
        f"- action breakdown: " + ", ".join(f"`{k}`={v}" for k, v in action_counts.items()),
        f"- by profile: " + ", ".join(f"`{k}`={v}" for k, v in sorted(profile_counts.items())),
        f"- prompt versions observed: {version_text}",
        f"- unique prompt hashes observed: `{len(set(str(row.get('prompt_hash') or 'unknown') for row in rows))}` | top hashes: {hash_text}",
    ]
    if rows and len(rows) == len(placed_rows):
        lines.append(
            "- in this report window, the available live history shows a `1:1` relationship between recorded autonomous LLM calls and placed trades; there are no autonomous skip rows preserved in this slice."
        )
    else:
        lines.append(
            "- this section is based on autonomous suggestion rows present in live history, so it reflects recorded calls in the slice rather than the separate runtime `llm_calls_today` counter."
        )
    return "\n".join(lines)


def _build_gate_prompt_section(window_label: str, rows: list[dict[str, Any]]) -> str:
    families = Counter(str(row.get("trigger_family") or ((row.get("placed_order") or {}).get("trigger_family") if isinstance(row.get("placed_order"), dict) else "unknown") or "unknown") for row in rows)
    top_families = ", ".join(f"`{fam}` x{count}" for fam, count in families.most_common(5)) or "`none`"
    lines = [
        "## Gate + Prompt Interaction",
        "",
        "- The code gate runs first. It evaluates the active trade-family detectors and only wakes Fillmore when a trigger family arms; if no trigger passes, there is no autonomous trade row and no LLM decision recorded in this report.",
        "- When the gate passes, it forwards structured metadata into the prompt context, including `trigger_family`, `trigger_reason`, session/level information, and micro-confirmation details. In this window the placed trade mix was led by: " + top_families + ".",
        "- The system prompt then asks Fillmore to reason about the gated setup rather than blindly execute it. The prompt explicitly frames the gate as a wake-up signal, not a command to trade.",
        "- The prompt allows Fillmore to choose `0` lots and skip, but only within the operator-configured lot envelope. Absolute base size and deviation are not chosen by the model; the model expresses conviction inside the range that was configured for the profile.",
        "- Every placed trade in these reports carries the exact `prompt_version` and `prompt_hash` used for that decision, so you can tie each trade back to the prompt build that generated it.",
        "- Because this forensic report is trade-centric, it mostly shows the `gate pass -> prompt reasoning -> placed trade` path. It does not enumerate blocked gate events unless they were captured elsewhere in autonomous stats.",
        f"- For `{window_label}`, the practical implication is: the report documents what Fillmore did after the gate woke it, not the full universe of times the gate blocked before any prompt call happened.",
    ]
    return "\n".join(lines)


def _replace_or_insert_sections(body: str, llm_summary: str, gate_section: str) -> str:
    new_sections = llm_summary + "\n\n" + gate_section + "\n\n"
    markers = [
        "## LLM Call Summary\n",
        "## Gate + Prompt Interaction\n",
    ]
    if "## LLM Call Summary\n" in body:
        start = body.index("## LLM Call Summary\n")
        next_header = body.find("\n## ", start + 1)
        while next_header != -1 and body.startswith("## Gate + Prompt Interaction\n", next_header + 1):
            next_header = body.find("\n## ", next_header + 1)
        if next_header == -1:
            body = body[:start].rstrip() + "\n\n" + new_sections.rstrip() + "\n"
        else:
            body = body[:start].rstrip() + "\n\n" + new_sections + body[next_header + 1 :]
        return body

    insert_after = None
    for header in ("## Headline\n", "## Snapshot\n"):
        idx = body.find(header)
        if idx != -1:
            next_header = body.find("\n## ", idx + len(header))
            if next_header != -1:
                insert_after = next_header + 1
            break
    if insert_after is None:
        raise ValueError("Could not find insertion point for new sections")
    return body[:insert_after] + new_sections + body[insert_after:]


def _augment_report(window: Window, all_rows: list[dict[str, Any]]) -> None:
    rows = _window_rows(all_rows, window.day_start, window.day_end)
    text = window.report_path.read_text(encoding="utf-8")
    llm_summary = _build_llm_summary(rows)
    gate_section = _build_gate_prompt_section(window.label, rows)
    updated = _replace_or_insert_sections(text, llm_summary, gate_section)
    window.report_path.write_text(updated, encoding="utf-8")


def main() -> int:
    all_rows: list[dict[str, Any]] = []
    for profile, path in HISTORY_PATHS.items():
        rows = _load_history(path)
        for row in rows:
            row.setdefault("_profile", profile)
            row.setdefault("profile", profile)
        all_rows.extend(rows)

    windows = [
        Window("Week 1", WEEK_START, WEEK_END, WEEK_FILE),
        *(Window(day, day, day, path) for day, path in DAILY_FILES.items()),
    ]
    for window in windows:
        _augment_report(window, all_rows)
        print(f"updated {window.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
