#!/usr/bin/env python3
"""Autonomous Fillmore performance investigation.

Builds an investigation harness that compares high-quality vs poor autonomous
trades, traces common denominators, identifies code-gate / model-prompt
mismatches, and emits proposed prompt/gate patches for human review. Does not
change live trading behavior.

Inputs (defaults):
  - research_out/autonomous_fillmore_evidence_<date>/combined_autonomous_joined.csv
  - research_out/autonomous_fillmore_evidence_<date>/combined_autonomous_suggestions_flat.csv
  - research_out/autonomous_fillmore_evidence_<date>/combined_closed_trades_flat.csv
  - research_out/autonomous_fillmore_evidence_<date>/fix_commit_log.txt

Outputs:
  research_out/autonomous_fillmore_performance_investigation_<date>/
    REPORT.md
    trade_cohorts.csv
    common_denominators.csv
    gate_prompt_mismatch.csv
    patch_recommendations.md

Run:
  python scripts/analyze_autonomous_fillmore_performance_investigation.py \
      --evidence research_out/autonomous_fillmore_evidence_20260429 \
      --out research_out/autonomous_fillmore_performance_investigation_20260429

Daily-loss shutdown is intentionally OUT OF SCOPE for this investigation. The
analysis aims to improve model selectivity via prompt + gate metadata, not to
recommend forced shutdown.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Cohort definitions
# ---------------------------------------------------------------------------

COHORT_HIGH_QUALITY = "high_quality"
COHORT_POOR_QUALITY = "poor_quality"
COHORT_LARGE_USD_LOSS = "large_usd_loss"
COHORT_HIGH_CONVICTION_LOSER = "high_conviction_loser"
COHORT_WEAKNESS_TRADED = "weakness_traded_anyway"
COHORT_SKIP = "skip"

ALL_COHORTS = [
    COHORT_HIGH_QUALITY,
    COHORT_POOR_QUALITY,
    COHORT_LARGE_USD_LOSS,
    COHORT_HIGH_CONVICTION_LOSER,
    COHORT_WEAKNESS_TRADED,
    COHORT_SKIP,
]


def _to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _norm_lower(value: Any) -> str:
    return str(value or "").strip().lower()


def _is_placed(row: dict[str, Any]) -> bool:
    return _norm_lower(row.get("action")) == "placed" and (_to_float(row.get("lots")) or 0.0) > 0


def _is_closed(row: dict[str, Any]) -> bool:
    has_close_ts = bool(str(row.get("closed_at") or "").strip())
    return _is_placed(row) and has_close_ts and (
        _to_float(row.get("pips")) is not None or _to_float(row.get("ledger_pips")) is not None
    )


def _pips(row: dict[str, Any]) -> Optional[float]:
    p = _to_float(row.get("pips"))
    if p is None:
        p = _to_float(row.get("ledger_pips"))
    return p


def _pnl(row: dict[str, Any]) -> Optional[float]:
    p = _to_float(row.get("pnl_usd"))
    if p is None:
        p = _to_float(row.get("ledger_profit"))
    return p


def _mae(row: dict[str, Any]) -> Optional[float]:
    p = _to_float(row.get("max_adverse_pips"))
    if p is None:
        p = _to_float(row.get("ledger_max_adverse_pips"))
    return p


def _mfe(row: dict[str, Any]) -> Optional[float]:
    p = _to_float(row.get("max_favorable_pips"))
    if p is None:
        p = _to_float(row.get("ledger_max_favorable_pips"))
    return p


def _exit_reason(row: dict[str, Any]) -> str:
    return _norm_lower(row.get("ledger_exit_reason") or row.get("exit_reason"))


def is_high_quality(row: dict[str, Any]) -> bool:
    if not _is_closed(row):
        return False
    pips = _pips(row)
    pnl = _pnl(row)
    if pips is None or pnl is None:
        return False
    if pips < 4 or pnl <= 0:
        return False
    mae = _mae(row)
    mfe = _mfe(row)
    return (mae is not None and mae >= -4) or (mfe is not None and mfe >= 6)


def is_poor_quality(row: dict[str, Any]) -> bool:
    if not _is_closed(row):
        return False
    pips = _pips(row)
    mae = _mae(row)
    if pips is not None and pips <= -6:
        return True
    if mae is not None and mae <= -8:
        return True
    if _exit_reason(row) in ("hit_stop_loss", "stop_loss", "stopped_out"):
        return True
    return False


def is_large_usd_loss(row: dict[str, Any]) -> bool:
    if not _is_closed(row):
        return False
    pnl = _pnl(row)
    return pnl is not None and pnl <= -150


def is_high_conviction_loser(row: dict[str, Any]) -> bool:
    if not _is_closed(row):
        return False
    rung = str(row.get("conviction_rung") or "").strip().upper()
    pips = _pips(row)
    return rung in ("A", "B") and pips is not None and pips < 0


def is_weakness_traded_anyway(row: dict[str, Any]) -> bool:
    if not _is_placed(row):
        return False
    if _norm_lower(row.get("zone_memory_read")) in ("failing_zone", "unresolved_chop"):
        return True
    if _norm_lower(row.get("repeat_trade_case")) == "blind_retry":
        return True
    rr = _to_float(row.get("planned_rr_estimate"))
    if rr is not None and rr < 1.0:
        return True
    if _norm_lower(row.get("timeframe_alignment")) in ("mixed", "countertrend"):
        return True
    why = str(row.get("why_trade_despite_weakness") or "").strip()
    if why and why.lower() not in ("none", "null", "n/a"):
        return True
    return False


def is_skip(row: dict[str, Any]) -> bool:
    if _norm_lower(row.get("decision")) == "skip":
        return True
    if (_to_float(row.get("lots")) or 0.0) <= 0.0:
        return True
    if _norm_lower(row.get("action")) and not _is_placed(row):
        return True
    return False


COHORT_PREDICATES: dict[str, Any] = {
    COHORT_HIGH_QUALITY: is_high_quality,
    COHORT_POOR_QUALITY: is_poor_quality,
    COHORT_LARGE_USD_LOSS: is_large_usd_loss,
    COHORT_HIGH_CONVICTION_LOSER: is_high_conviction_loser,
    COHORT_WEAKNESS_TRADED: is_weakness_traded_anyway,
    COHORT_SKIP: is_skip,
}


def label_cohorts(row: dict[str, Any]) -> list[str]:
    return [name for name, pred in COHORT_PREDICATES.items() if pred(row)]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def _safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return statistics.fmean(vals) if vals else None


def _fmt(value: Optional[float], digits: int = 1, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    fmt = f"{{:+.{digits}f}}" if signed else f"{{:.{digits}f}}"
    return fmt.format(value)


def cohort_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    closed = [r for r in rows if _is_closed(r)]
    pips = [_pips(r) for r in closed]
    pnls = [_pnl(r) for r in closed]
    maes = [_mae(r) for r in closed]
    mfes = [_mfe(r) for r in closed]
    holds = [_to_float(r.get("minutes_open")) for r in closed]
    wins = [r for r in closed if _norm_lower(r.get("win_loss")) == "win"]
    return {
        "count": len(rows),
        "closed_count": len(closed),
        "win_rate": (len(wins) / len(closed)) if closed else None,
        "avg_pips": _safe_mean(pips),
        "net_pips": sum(p for p in pips if p is not None) if pips else None,
        "avg_pnl_usd": _safe_mean(pnls),
        "net_pnl_usd": sum(p for p in pnls if p is not None) if pnls else None,
        "avg_mae_pips": _safe_mean(maes),
        "avg_mfe_pips": _safe_mean(mfes),
        "avg_hold_min": _safe_mean(holds),
        "exit_reasons": Counter(_exit_reason(r) for r in closed if _exit_reason(r)).most_common(5),
    }


def grouped_summary(
    rows: list[dict[str, Any]],
    key_fn: Any,
    *,
    min_count: int = 2,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = key_fn(r)
        if key is None or key == "":
            key = "(blank)"
        groups[str(key)].append(r)
    out = []
    for key, items in groups.items():
        if len(items) < min_count:
            continue
        s = cohort_summary(items)
        s["key"] = key
        out.append(s)
    out.sort(key=lambda s: -(s.get("count") or 0))
    return out


# ---------------------------------------------------------------------------
# Common denominators
# ---------------------------------------------------------------------------

DIMENSIONS: list[tuple[str, Any]] = [
    ("trigger_family", lambda r: r.get("trigger_family") or "(unknown)"),
    ("trigger_reason", lambda r: r.get("trigger_reason") or "(unknown)"),
    ("session", lambda r: r.get("session") or "(blank)"),
    ("side", lambda r: r.get("side") or "(blank)"),
    ("conviction_rung", lambda r: r.get("conviction_rung") or "(blank)"),
    ("zone_memory_read", lambda r: r.get("zone_memory_read") or "(blank)"),
    ("repeat_trade_case", lambda r: r.get("repeat_trade_case") or "(blank)"),
    ("timeframe_alignment", lambda r: r.get("timeframe_alignment") or "(blank)"),
    ("trigger_fit", lambda r: r.get("trigger_fit") or "(blank)"),
    ("exit_strategy", lambda r: r.get("exit_strategy") or "(blank)"),
    (
        "planned_rr_bucket",
        lambda r: _rr_bucket(_to_float(r.get("planned_rr_estimate"))),
    ),
    ("prompt_version", lambda r: r.get("prompt_version") or "(blank)"),
    ("order_type", lambda r: (r.get("placed_order_json") or "")[:1] or "(blank)"),
    ("h1_regime", lambda r: r.get("h1_regime") or "(blank)"),
    ("m5_regime", lambda r: r.get("m5_regime") or "(blank)"),
    ("m1_regime", lambda r: r.get("m1_regime") or "(blank)"),
    ("vol_label", lambda r: r.get("vol_label") or "(blank)"),
    ("macro_combined_bias", lambda r: r.get("macro_combined_bias") or "(blank)"),
]


def _rr_bucket(rr: Optional[float]) -> str:
    if rr is None:
        return "(missing)"
    if rr < 0.7:
        return "<0.7"
    if rr < 1.0:
        return "0.7-1.0"
    if rr < 1.3:
        return "1.0-1.3"
    if rr < 2.0:
        return "1.3-2.0"
    return ">=2.0"


# ---------------------------------------------------------------------------
# Gate / prompt mismatch
# ---------------------------------------------------------------------------


def gate_prompt_mismatch_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return cases where the code gate woke the model and the model itself
    flagged a weak edge but still placed the trade.

    These are the highest-leverage prompt-improvement candidates: the model
    has the words for "this is weak" but the gate-induced framing of "look
    here" is overriding its own self-assessment.
    """
    out = []
    for r in rows:
        if not _is_placed(r):
            continue
        family = _norm_lower(r.get("trigger_family"))
        if not family or family == "unknown":
            continue
        weakness = []
        if _norm_lower(r.get("zone_memory_read")) in ("failing_zone", "unresolved_chop"):
            weakness.append(f"zone={r.get('zone_memory_read')}")
        if _norm_lower(r.get("repeat_trade_case")) == "blind_retry":
            weakness.append("repeat=blind_retry")
        rr = _to_float(r.get("planned_rr_estimate"))
        if rr is not None and rr < 1.0:
            weakness.append(f"rr={rr:.2f}")
        if _norm_lower(r.get("timeframe_alignment")) in ("countertrend", "mixed"):
            weakness.append(f"tf={r.get('timeframe_alignment')}")
        why = str(r.get("why_trade_despite_weakness") or "").strip()
        if why and why.lower() not in ("none", "null", "n/a"):
            weakness.append("why_trade_despite_weakness=set")
        rung = str(r.get("conviction_rung") or "").strip().upper()
        if rung == "D":
            weakness.append("rung=D")
        if not weakness:
            continue
        out.append({
            "profile": r.get("profile"),
            "created_utc": r.get("created_utc"),
            "trigger_family": r.get("trigger_family"),
            "trigger_reason": r.get("trigger_reason"),
            "side": r.get("side"),
            "lots": r.get("lots"),
            "conviction_rung": rung,
            "zone_memory_read": r.get("zone_memory_read"),
            "repeat_trade_case": r.get("repeat_trade_case"),
            "timeframe_alignment": r.get("timeframe_alignment"),
            "planned_rr_estimate": r.get("planned_rr_estimate"),
            "why_trade_despite_weakness": (why[:160] + "...") if len(why) > 160 else why,
            "weakness_signals": "; ".join(weakness),
            "outcome_pips": _pips(r),
            "outcome_pnl_usd": _pnl(r),
            "outcome_mae": _mae(r),
            "outcome_mfe": _mfe(r),
            "exit_reason": _exit_reason(r),
            "trade_thesis": (str(r.get("trade_thesis") or "")[:160]),
        })
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def write_trade_cohorts_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "profile",
        "suggestion_id",
        "trade_id",
        "created_utc",
        "closed_at",
        "action",
        "decision",
        "side",
        "lots",
        "trigger_family",
        "trigger_reason",
        "conviction_rung",
        "zone_memory_read",
        "repeat_trade_case",
        "timeframe_alignment",
        "trigger_fit",
        "planned_rr_estimate",
        "exit_strategy",
        "prompt_version",
        "session",
        "vol_label",
        "h1_regime",
        "m5_regime",
        "m1_regime",
        "pips",
        "pnl_usd",
        "max_adverse_pips",
        "max_favorable_pips",
        "minutes_open",
        "exit_reason",
        "win_loss",
        "cohort_high_quality",
        "cohort_poor_quality",
        "cohort_large_usd_loss",
        "cohort_high_conviction_loser",
        "cohort_weakness_traded_anyway",
        "cohort_skip",
        "cohort_labels",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            labels = label_cohorts(r)
            w.writerow({
                "profile": r.get("profile"),
                "suggestion_id": r.get("suggestion_id"),
                "trade_id": r.get("trade_id"),
                "created_utc": r.get("created_utc"),
                "closed_at": r.get("closed_at"),
                "action": r.get("action"),
                "decision": r.get("decision"),
                "side": r.get("side"),
                "lots": r.get("lots"),
                "trigger_family": r.get("trigger_family"),
                "trigger_reason": r.get("trigger_reason"),
                "conviction_rung": r.get("conviction_rung"),
                "zone_memory_read": r.get("zone_memory_read"),
                "repeat_trade_case": r.get("repeat_trade_case"),
                "timeframe_alignment": r.get("timeframe_alignment"),
                "trigger_fit": r.get("trigger_fit"),
                "planned_rr_estimate": r.get("planned_rr_estimate"),
                "exit_strategy": r.get("exit_strategy"),
                "prompt_version": r.get("prompt_version"),
                "session": r.get("session"),
                "vol_label": r.get("vol_label"),
                "h1_regime": r.get("h1_regime"),
                "m5_regime": r.get("m5_regime"),
                "m1_regime": r.get("m1_regime"),
                "pips": _pips(r),
                "pnl_usd": _pnl(r),
                "max_adverse_pips": _mae(r),
                "max_favorable_pips": _mfe(r),
                "minutes_open": r.get("minutes_open"),
                "exit_reason": _exit_reason(r),
                "win_loss": r.get("win_loss"),
                "cohort_high_quality": int(COHORT_HIGH_QUALITY in labels),
                "cohort_poor_quality": int(COHORT_POOR_QUALITY in labels),
                "cohort_large_usd_loss": int(COHORT_LARGE_USD_LOSS in labels),
                "cohort_high_conviction_loser": int(COHORT_HIGH_CONVICTION_LOSER in labels),
                "cohort_weakness_traded_anyway": int(COHORT_WEAKNESS_TRADED in labels),
                "cohort_skip": int(COHORT_SKIP in labels),
                "cohort_labels": "|".join(labels),
            })


def write_common_denominators_csv(
    path: Path,
    cohort_rows: dict[str, list[dict[str, Any]]],
) -> None:
    fields = [
        "cohort",
        "dimension",
        "key",
        "count",
        "closed_count",
        "win_rate",
        "avg_pips",
        "net_pips",
        "avg_pnl_usd",
        "net_pnl_usd",
        "avg_mae_pips",
        "avg_mfe_pips",
        "avg_hold_min",
        "top_exit_reason",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for cohort, rows in cohort_rows.items():
            for dim_name, key_fn in DIMENSIONS:
                groups = grouped_summary(rows, key_fn, min_count=1)
                for g in groups:
                    top_exit = ""
                    er = g.get("exit_reasons") or []
                    if er:
                        top_exit = f"{er[0][0]}({er[0][1]})"
                    w.writerow({
                        "cohort": cohort,
                        "dimension": dim_name,
                        "key": g["key"],
                        "count": g["count"],
                        "closed_count": g.get("closed_count"),
                        "win_rate": _fmt(g.get("win_rate"), 3),
                        "avg_pips": _fmt(g.get("avg_pips"), 2, signed=True),
                        "net_pips": _fmt(g.get("net_pips"), 1, signed=True),
                        "avg_pnl_usd": _fmt(g.get("avg_pnl_usd"), 2, signed=True),
                        "net_pnl_usd": _fmt(g.get("net_pnl_usd"), 2, signed=True),
                        "avg_mae_pips": _fmt(g.get("avg_mae_pips"), 2, signed=True),
                        "avg_mfe_pips": _fmt(g.get("avg_mfe_pips"), 2, signed=True),
                        "avg_hold_min": _fmt(g.get("avg_hold_min"), 1),
                        "top_exit_reason": top_exit,
                    })


def write_gate_prompt_mismatch_csv(path: Path, mismatches: list[dict[str, Any]]) -> None:
    if not mismatches:
        with path.open("w", newline="") as f:
            f.write("# no gate/prompt mismatch rows detected\n")
        return
    fields = list(mismatches[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(mismatches)


def _format_summary_block(summary: dict[str, Any]) -> str:
    if not summary or summary.get("count", 0) == 0:
        return "(no rows)"
    parts = [
        f"n={summary['count']}",
        f"closed={summary.get('closed_count') or 0}",
        f"WR={_fmt((summary.get('win_rate') or 0) * 100, 0)}%" if summary.get("win_rate") is not None else "WR=n/a",
        f"avg_pips={_fmt(summary.get('avg_pips'), 2, signed=True)}",
        f"net_pips={_fmt(summary.get('net_pips'), 1, signed=True)}",
        f"avg_pnl=${_fmt(summary.get('avg_pnl_usd'), 2, signed=True)}",
        f"net_pnl=${_fmt(summary.get('net_pnl_usd'), 2, signed=True)}",
        f"avg_MAE={_fmt(summary.get('avg_mae_pips'), 2, signed=True)}",
        f"avg_MFE={_fmt(summary.get('avg_mfe_pips'), 2, signed=True)}",
        f"hold={_fmt(summary.get('avg_hold_min'), 0)}m",
    ]
    line = " | ".join(parts)
    er = summary.get("exit_reasons") or []
    if er:
        line += " | exits: " + ", ".join(f"{k}={c}" for k, c in er)
    return line


def _top_dim_lines(rows: list[dict[str, Any]], limit: int = 5) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for dim_name, key_fn in DIMENSIONS:
        groups = grouped_summary(rows, key_fn, min_count=2)
        if not groups:
            continue
        lines = []
        for g in groups[:limit]:
            lines.append(
                f"  - {g['key']}: n={g['count']}, "
                f"avg_pips={_fmt(g.get('avg_pips'), 2, signed=True)}, "
                f"net_pips={_fmt(g.get('net_pips'), 1, signed=True)}, "
                f"WR={_fmt((g.get('win_rate') or 0) * 100, 0)}%"
            )
        out[dim_name] = lines
    return out


def write_report_md(
    path: Path,
    *,
    all_rows: list[dict[str, Any]],
    cohort_rows: dict[str, list[dict[str, Any]]],
    mismatches: list[dict[str, Any]],
    evidence_dir: Path,
) -> None:
    closed = [r for r in all_rows if _is_closed(r)]
    placed = [r for r in all_rows if _is_placed(r)]
    skipped = cohort_rows.get(COHORT_SKIP, [])
    overall = cohort_summary(placed)

    hq = cohort_rows.get(COHORT_HIGH_QUALITY, [])
    pq = cohort_rows.get(COHORT_POOR_QUALITY, [])
    large_loss = cohort_rows.get(COHORT_LARGE_USD_LOSS, [])
    weakness = cohort_rows.get(COHORT_WEAKNESS_TRADED, [])
    high_conv_lose = cohort_rows.get(COHORT_HIGH_CONVICTION_LOSER, [])

    lines: list[str] = []
    lines.append("# Autonomous Fillmore Performance Investigation")
    lines.append("")
    lines.append(f"_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}Z from `{evidence_dir}`._")
    lines.append("")
    lines.append("> **Scope.** This investigation explains why Autonomous Fillmore is")
    lines.append("> underperforming by comparing high-quality vs poor autonomous trades and")
    lines.append("> tracing the common denominators back to code-gate and prompt behavior.")
    lines.append("> **Daily-loss shutdown is intentionally out of scope** — the goal here is")
    lines.append("> better model selectivity, not a forced kill switch.")
    lines.append("")
    lines.append("## Cohort definitions")
    lines.append("")
    lines.append("Cohorts are labeled on **pips/MAE/MFE first, USD second**, so lot size does")
    lines.append("not dominate the explanation. They are not mutually exclusive — a trade can")
    lines.append("appear in several cohorts.")
    lines.append("")
    lines.append("- `high_quality`: closed placed trade with `pips ≥ +4`, `pnl > 0`, and")
    lines.append("  either `MAE ≥ -4` or `MFE ≥ +6` (genuinely clean trade, not a saved loss)")
    lines.append("- `poor_quality`: closed trade with `pips ≤ -6` OR `MAE ≤ -8` OR a")
    lines.append("  stop-loss exit")
    lines.append("- `large_usd_loss`: `pnl ≤ -$150` (sizing-amplified, kept separate)")
    lines.append("- `high_conviction_loser`: model self-rated rung A or B and lost pips")
    lines.append("- `weakness_traded_anyway`: placed trade with `failing_zone`,")
    lines.append("  `unresolved_chop`, `blind_retry`, `RR<1.0`, `mixed`/`countertrend`,")
    lines.append("  or any non-empty `why_trade_despite_weakness`")
    lines.append("- `skip`: model returned `decision=skip` or `lots≤0`")
    lines.append("")

    lines.append("## Top-line performance")
    lines.append("")
    lines.append(f"- All suggestions: **{len(all_rows)}** "
                 f"(placed {len(placed)}, closed {len(closed)}, skipped {len(skipped)})")
    lines.append(f"- Overall placed: {_format_summary_block(overall)}")
    lines.append("")

    lines.append("## Cohort summary")
    lines.append("")
    lines.append("| Cohort | n | closed | WR | avg_pips | net_pips | avg_pnl | net_pnl | avg_MAE | avg_MFE |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for cohort in ALL_COHORTS:
        s = cohort_summary(cohort_rows.get(cohort, []))
        lines.append(
            f"| `{cohort}` | {s.get('count') or 0} | {s.get('closed_count') or 0} | "
            f"{_fmt((s.get('win_rate') or 0) * 100, 0)}% | "
            f"{_fmt(s.get('avg_pips'), 2, signed=True)} | "
            f"{_fmt(s.get('net_pips'), 1, signed=True)} | "
            f"{_fmt(s.get('avg_pnl_usd'), 2, signed=True)} | "
            f"{_fmt(s.get('net_pnl_usd'), 2, signed=True)} | "
            f"{_fmt(s.get('avg_mae_pips'), 2, signed=True)} | "
            f"{_fmt(s.get('avg_mfe_pips'), 2, signed=True)} |"
        )
    lines.append("")

    # ---- HQ vs PQ common denominators ----
    lines.append("## High-quality common denominators")
    lines.append("")
    lines.append(f"_n={len(hq)}_ — pips≥+4, pnl>0, with MAE≥-4 or MFE≥+6.")
    lines.append("")
    if not hq:
        lines.append("_No high-quality trades in this evidence pack._")
    else:
        for dim, dim_lines in _top_dim_lines(hq, limit=5).items():
            lines.append(f"**By {dim}**")
            lines.extend(dim_lines)
            lines.append("")

    lines.append("## Poor-quality common denominators")
    lines.append("")
    lines.append(f"_n={len(pq)}_ — pips≤-6, MAE≤-8, or stop-loss exits.")
    lines.append("")
    if not pq:
        lines.append("_No poor-quality trades in this evidence pack._")
    else:
        for dim, dim_lines in _top_dim_lines(pq, limit=5).items():
            lines.append(f"**By {dim}**")
            lines.extend(dim_lines)
            lines.append("")

    lines.append("## Weakness-traded-anyway common denominators")
    lines.append("")
    lines.append(f"_n={len(weakness)}_ — model placed lots>0 while flagging at least one")
    lines.append("weakness signal (failing zone, unresolved chop, blind retry, RR<1, mixed/countertrend, or hedging language).")
    lines.append("")
    if not weakness:
        lines.append("_No weakness-traded-anyway rows. Either the model is clean, or these fields are absent (older runs)._")
    else:
        for dim, dim_lines in _top_dim_lines(weakness, limit=5).items():
            lines.append(f"**By {dim}**")
            lines.extend(dim_lines)
            lines.append("")

    lines.append("## Large USD losses")
    lines.append("")
    lines.append(f"_n={len(large_loss)}_ — pnl ≤ -$150. Kept separate because sizing amplifies these.")
    if not large_loss:
        lines.append("")
        lines.append("_None._")
    else:
        lines.append("")
        for r in large_loss[:25]:
            lines.append(
                f"- {r.get('profile')}/{r.get('created_utc','')[:16]} "
                f"{r.get('side','?')} lots={r.get('lots')} "
                f"family={r.get('trigger_family') or '(none)'} "
                f"rung={r.get('conviction_rung') or '?'} "
                f"zone={r.get('zone_memory_read') or '?'} "
                f"rr={r.get('planned_rr_estimate') or '?'} "
                f"-> {_fmt(_pips(r), 1, signed=True)}p / ${_fmt(_pnl(r), 0, signed=True)} "
                f"(MAE {_fmt(_mae(r), 1, signed=True)}, MFE {_fmt(_mfe(r), 1, signed=True)})"
            )

    lines.append("")
    lines.append("## High-conviction losers")
    lines.append("")
    lines.append(f"_n={len(high_conv_lose)}_ — rung A/B but pips<0. These are misjudgments,")
    lines.append("not weak setups the model already flagged.")
    lines.append("")
    if not high_conv_lose:
        lines.append("_None._")
    else:
        for r in high_conv_lose[:25]:
            lines.append(
                f"- {r.get('profile')}/{r.get('created_utc','')[:16]} rung {r.get('conviction_rung')} "
                f"{r.get('side','?')} family={r.get('trigger_family')} "
                f"trigger={r.get('trigger_reason')} -> "
                f"{_fmt(_pips(r), 1, signed=True)}p (MAE {_fmt(_mae(r), 1, signed=True)})"
            )

    # ---- HQ vs PQ contrast ----
    lines.append("")
    lines.append("## High-quality vs poor-quality contrasts")
    lines.append("")
    lines.append("This is the main signal for prompt design. Where do the distributions of")
    lines.append("the same field diverge between the two cohorts?")
    lines.append("")
    if hq and pq:
        for dim_name, key_fn in DIMENSIONS:
            hq_counts = Counter(str(key_fn(r) or "(blank)") for r in hq)
            pq_counts = Counter(str(key_fn(r) or "(blank)") for r in pq)
            keys = sorted(set(hq_counts) | set(pq_counts))
            if not keys:
                continue
            interesting = [
                k for k in keys
                if (hq_counts.get(k, 0) + pq_counts.get(k, 0)) >= 3
                and abs((hq_counts.get(k, 0) / max(1, len(hq))) - (pq_counts.get(k, 0) / max(1, len(pq)))) >= 0.1
            ]
            if not interesting:
                continue
            lines.append(f"**{dim_name}** (HQ% vs PQ%, raw counts in parens):")
            for k in interesting:
                hp = hq_counts.get(k, 0) / max(1, len(hq)) * 100
                pp = pq_counts.get(k, 0) / max(1, len(pq)) * 100
                lines.append(f"  - {k}: HQ {hp:.0f}% ({hq_counts.get(k,0)}) vs PQ {pp:.0f}% ({pq_counts.get(k,0)})")
            lines.append("")

    # ---- Gate / prompt mismatch ----
    lines.append("## Code-gate / model self-assessment mismatch")
    lines.append("")
    lines.append(f"_n={len(mismatches)}_ rows where the gate woke the model and the model's")
    lines.append("own fields said the edge is weak, yet it placed the trade anyway. These are")
    lines.append("the highest-leverage prompt patches.")
    lines.append("")
    if mismatches:
        # closed mismatches outcome
        closed_mismatch = [m for m in mismatches if m.get("outcome_pips") is not None]
        if closed_mismatch:
            pips_vals = [_to_float(m.get("outcome_pips")) for m in closed_mismatch]
            pips_vals = [p for p in pips_vals if p is not None]
            wins = sum(1 for p in pips_vals if p > 0)
            lines.append(
                f"Closed mismatches: {len(closed_mismatch)} | WR={wins/len(closed_mismatch)*100:.0f}% | "
                f"avg_pips={statistics.fmean(pips_vals):+.2f} | net_pips={sum(pips_vals):+.1f}"
            )
            lines.append("")
        # by family
        by_fam: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for m in mismatches:
            by_fam[str(m.get("trigger_family"))].append(m)
        lines.append("By gate family:")
        for fam, items in sorted(by_fam.items(), key=lambda kv: -len(kv[1])):
            closed_items = [m for m in items if m.get("outcome_pips") is not None]
            if closed_items:
                pv = [_to_float(m.get("outcome_pips")) or 0.0 for m in closed_items]
                lines.append(
                    f"  - {fam}: n={len(items)}, closed={len(closed_items)}, "
                    f"net_pips={sum(pv):+.1f}, avg_pips={statistics.fmean(pv):+.2f}"
                )
            else:
                lines.append(f"  - {fam}: n={len(items)}, no closes")
        lines.append("")
        lines.append("Top weakness-signal combinations driving placements:")
        sig_counter = Counter(m.get("weakness_signals") for m in mismatches)
        for sig, c in sig_counter.most_common(10):
            lines.append(f"  - `{sig}` × {c}")
        lines.append("")

    lines.append("## Notes on gate/prompt design")
    lines.append("")
    lines.append("- The current gate hands the model a trigger family + reason; the prompt")
    lines.append("  contains a family scorecard and a today log. The model has the field")
    lines.append("  names (`zone_memory_read`, `repeat_trade_case`, `why_trade_despite_weakness`)")
    lines.append("  to *describe* weakness, but the prompt does not currently force a")
    lines.append("  comparison against discovered HQ vs PQ fingerprints. That is the gap.")
    lines.append("- Server-side vetoes already cover a few of the most obvious patterns")
    lines.append("  (failing-zone+blind-retry, rung-D + RR<1, unresolved-chop + RR<1.1,")
    lines.append("  unclear-trigger + RR<1, repeat-fire + recent-fires≥1). Recommendations")
    lines.append("  below extend coverage where the data supports it, but prefer prompt")
    lines.append("  changes over hard vetoes.")
    lines.append("- Prompt remains under `gpt-5.4-mini` — recommendations favor short, explicit")
    lines.append("  rules over long-form judgment.")
    lines.append("")

    lines.append("## Out of scope")
    lines.append("")
    lines.append("- Daily-loss shutdown / hard kill switches.")
    lines.append("- Lot-sizing recalibration (deliberately ignored so we don't hide signal in sizing).")
    lines.append("- Live behavior changes — recommendations are for human review only.")
    lines.append("")

    lines.append("## Outputs")
    lines.append("")
    lines.append("- `trade_cohorts.csv` — every autonomous suggestion labeled with cohort flags.")
    lines.append("- `common_denominators.csv` — full per-cohort × per-dimension table.")
    lines.append("- `gate_prompt_mismatch.csv` — placed-with-flagged-weakness rows.")
    lines.append("- `patch_recommendations.md` — proposed prompt + gate metadata patches for review.")
    lines.append("")

    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Patch recommendations
# ---------------------------------------------------------------------------


def write_patch_recommendations(
    path: Path,
    *,
    cohort_rows: dict[str, list[dict[str, Any]]],
    mismatches: list[dict[str, Any]],
) -> None:
    hq = cohort_rows.get(COHORT_HIGH_QUALITY, [])
    pq = cohort_rows.get(COHORT_POOR_QUALITY, [])
    weakness = cohort_rows.get(COHORT_WEAKNESS_TRADED, [])

    def _fingerprint(rows: list[dict[str, Any]]) -> dict[str, list[tuple[str, int]]]:
        out = {}
        for dim_name, key_fn in DIMENSIONS:
            counter = Counter(str(key_fn(r) or "(blank)") for r in rows)
            out[dim_name] = counter.most_common(5)
        return out

    hq_fp = _fingerprint(hq) if hq else {}
    pq_fp = _fingerprint(pq) if pq else {}

    lines: list[str] = []
    lines.append("# Patch Recommendations (for human audit, not auto-applied)")
    lines.append("")
    lines.append("These are concrete, reviewable changes to the autonomous Fillmore prompt and")
    lines.append("code-gate metadata. They are not applied. Each item names the file/symbol to")
    lines.append("touch, the proposed change, and the data that supports it. **No daily-loss")
    lines.append("shutdown is recommended** — the goal is selectivity.")
    lines.append("")

    # --- Patch 1: Fingerprint memory block ---
    lines.append("## 1. Add a compact good-trade / bad-trade fingerprint memory block")
    lines.append("")
    lines.append("**Where:** `api/autonomous_performance.py` — add a new")
    lines.append("`build_quality_fingerprint_memory_block(db_path)` next to")
    lines.append("`build_performance_memory_block` and `build_family_scorecard_memory_block`,")
    lines.append("and wire it into prompt assembly in `api/autonomous_fillmore.py::_invoke_suggest`")
    lines.append("(the prompt builder section near `prompt_version=AUTONOMOUS_PROMPT_VERSION`).")
    lines.append("")
    lines.append("**What it should say** (kept short for `gpt-5.4-mini`, hard rules > vibes):")
    lines.append("")
    lines.append("```")
    lines.append("=== AUTONOMOUS QUALITY FINGERPRINTS ===")
    lines.append("Trades that worked (last 30d, pips>=+4 with MAE>=-4 or MFE>=+6):")
    if hq_fp:
        for dim in ("trigger_family", "session", "trigger_fit", "zone_memory_read", "timeframe_alignment", "conviction_rung"):
            top = hq_fp.get(dim, [])
            if top:
                lines.append(f"  - {dim}: " + ", ".join(f"{k}({c})" for k, c in top[:3]))
    else:
        lines.append("  - (insufficient HQ trades — fall back to default rule)")
    lines.append("Trades that failed (pips<=-6, MAE<=-8, or SL exit):")
    if pq_fp:
        for dim in ("trigger_family", "session", "trigger_fit", "zone_memory_read", "timeframe_alignment", "conviction_rung"):
            top = pq_fp.get(dim, [])
            if top:
                lines.append(f"  - {dim}: " + ", ".join(f"{k}({c})" for k, c in top[:3]))
    else:
        lines.append("  - (insufficient PQ trades)")
    lines.append("Rule: if the current setup matches >=2 fields from the failing fingerprint")
    lines.append("and 0 from the working one, demote conviction by one rung and prefer skip.")
    lines.append("```")
    lines.append("")
    lines.append("**Why:** the current `build_performance_memory_block` reports cohort-level WR")
    lines.append("and PF but does not tell the model *what specifically distinguishes a winner*.")
    lines.append("On `gpt-5.4-mini`, an explicit comparison rule outperforms long advisory prose.")
    lines.append("")

    # --- Patch 2: Hard skip rule for top mismatch combos ---
    lines.append("## 2. Tighten the prompt's binding-skip block")
    lines.append("")
    lines.append("**Where:** `api/autonomous_fillmore.py::_invoke_suggest`, around the")
    lines.append("`# Binding skip discipline` block (currently lines ~4506-4549). Add prompt")
    lines.append("text that pre-commits the model, plus optionally one new server veto.")
    lines.append("")
    lines.append("**Proposed prompt addition** (insert after the existing CONVICTION RUNG block):")
    lines.append("")
    lines.append("```")
    lines.append("BINDING SKIP RULES (server will enforce; do not try to argue around them):")
    lines.append("- failing_zone + blind_retry -> skip, regardless of rung.")
    lines.append("- unresolved_chop + planned_rr_estimate < 1.1 -> skip.")
    lines.append("- conviction_rung D + planned_rr_estimate < 1.0 -> skip.")
    lines.append("- trigger_fit in {unclear, micro_expansion_inside_chop} + RR < 1.0 -> skip.")
    lines.append("- repeat_trade_case = blind_retry + working_zone/failing_zone/unresolved_chop")
    lines.append("  while another trade in this fingerprint fired in the last 2h -> skip.")
    lines.append("If you want to take a setup that hits any of these, the answer is")
    lines.append("'wait for material change.' Write that in skip_reason.")
    lines.append("```")
    lines.append("")
    if mismatches:
        sig_counter = Counter(m.get("weakness_signals") for m in mismatches)
        top_combos = sig_counter.most_common(5)
        lines.append("**Why:** the gate/prompt mismatch CSV shows these specific combinations")
        lines.append("most often slip past the model's own self-assessment:")
        lines.append("")
        for sig, c in top_combos:
            lines.append(f"  - `{sig}` × {c}")
        lines.append("")
        lines.append("These are already partially covered by server vetoes — the patch is to make")
        lines.append("the rules visible *before* the model writes its decision so it stops")
        lines.append("forming the trade plan in the first place.")
    lines.append("")

    # --- Patch 3: Family-specific guard text ---
    lines.append("## 3. Add family-specific failure warnings to the family scorecard")
    lines.append("")
    lines.append("**Where:** `api/autonomous_performance.py::build_family_scorecard_memory_block`")
    lines.append("(the per-family loop around lines 558-605).")
    lines.append("")
    lines.append("**What:** when a family's recent net pips are clearly negative or its")
    lines.append("dominant exit reason is `hit_stop_loss`, append a one-line, family-specific")
    lines.append("'fail mode' note next to the existing `caution=` bits. For example:")
    lines.append("")
    lines.append("```")
    lines.append("- critical_level_reaction: idea=...; recent=...; FAIL MODE: tends to lose")
    lines.append("  when zone_memory_read is unresolved_chop and timeframe_alignment is mixed.")
    lines.append("```")
    lines.append("")
    lines.append("**Why:** the current scorecard exposes intent and recent stats but not the")
    lines.append("specific failure mode of each family in this evidence. The investigation's")
    lines.append("`common_denominators.csv` row set per family is exactly the input needed to")
    lines.append("produce that one-liner; we recommend computing it from rolling-30d closed")
    lines.append("trades and caching alongside the family scorecard.")
    lines.append("")

    # --- Patch 4: today block surfaces P&L context ---
    lines.append("## 4. Surface today + rolling-20 P&L more visibly")
    lines.append("")
    lines.append("**Where:** `api/suggestion_tracker.py::build_autonomous_today_block` and")
    lines.append("`api/autonomous_performance.py::build_performance_memory_block`.")
    lines.append("")
    lines.append("**What:** keep today P&L on its current line, but add a one-line callout if")
    lines.append("today P&L is < 0 *and* rolling-20 P&L is < 0 — `'Both today and rolling-20")
    lines.append("are negative; raise the bar to rung B+ and skip anything you would otherwise")
    lines.append("rationalize.'`")
    lines.append("")
    lines.append("**Why:** the request explicitly excludes hard daily-loss shutdown, but")
    lines.append("conditional selectivity is a softer lever. The model already has these stats,")
    lines.append("but they are buried — making them load-bearing in the prompt costs almost")
    lines.append("nothing on `gpt-5.4-mini`.")
    lines.append("")

    # --- Patch 5: Optional / non-preferred ---
    lines.append("## 5. (Optional / non-preferred) Hard gate on structurally invalid wakeups")
    lines.append("")
    lines.append("**Status:** only consider this if the gate-prompt mismatch CSV shows the gate")
    lines.append("repeatedly waking the model on a family/zone combination that has produced")
    lines.append("zero high-quality trades and a clear net loss. The data should justify it.")
    lines.append("")
    lines.append("**Where:** `api/autonomous_fillmore.py::evaluate_gate` — additional Layer 1")
    lines.append("hard filter rejecting wakeups for the offending family while a specific zone")
    lines.append("memory pattern is active.")
    lines.append("")
    lines.append("**Why we mark this non-preferred:** the same intent is achieved by the")
    lines.append("scorecard fail-mode line in patch 3 *plus* the binding-skip rules in patch 2.")
    lines.append("A hard gate removes the option for the model to write a coherent skip_reason,")
    lines.append("which we still want for analytics. Use only when the prompt route has been")
    lines.append("tried and a specific family is structurally unsalvageable.")
    lines.append("")

    # --- Patch 6: prompt-version naming ---
    lines.append("## 6. Bump `AUTONOMOUS_PROMPT_VERSION` on rollout")
    lines.append("")
    lines.append("**Where:** `api/autonomous_fillmore.py:66` (`AUTONOMOUS_PROMPT_VERSION = ...`).")
    lines.append("")
    lines.append("**What:** when patches 1-4 ship, bump the version string (e.g.,")
    lines.append("`autonomous_phase2_runner_custom_exit_v4_quality_fingerprints`). The")
    lines.append("`prompt_version_breakdown_json` analytics already segment by version, which")
    lines.append("makes the next investigation A/B-clean.")
    lines.append("")
    lines.append("**Why:** without a version bump, post-patch performance regressions cannot be")
    lines.append("attributed cleanly. This is cheap insurance.")
    lines.append("")

    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def find_default_evidence_dir() -> Optional[Path]:
    candidates = sorted((REPO_ROOT / "research_out").glob("autonomous_fillmore_evidence_*"))
    return candidates[-1] if candidates else None


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--evidence", type=Path, default=None, help="Evidence directory (default: latest)")
    p.add_argument("--out", type=Path, default=None, help="Output directory")
    p.add_argument("--rerun-export", action="store_true",
                   help="Re-run scripts/export_autonomous_fillmore_evidence.py before analyzing")
    args = p.parse_args(argv)

    evidence = args.evidence or find_default_evidence_dir()
    if evidence is None:
        print("error: no evidence directory found. pass --evidence or run the exporter first.",
              file=sys.stderr)
        return 2

    if args.rerun_export:
        # Best-effort: caller is responsible for env/auth.
        os.system(f"python {REPO_ROOT}/scripts/export_autonomous_fillmore_evidence.py")

    joined_path = evidence / "combined_autonomous_joined.csv"
    suggestions_path = evidence / "combined_autonomous_suggestions_flat.csv"
    if not joined_path.exists():
        print(f"error: missing {joined_path}", file=sys.stderr)
        return 2

    rows = load_csv(joined_path)
    if not rows:
        # fall back to suggestions-only
        rows = load_csv(suggestions_path)

    out_dir = args.out
    if out_dir is None:
        date_tag = datetime.now(timezone.utc).strftime("%Y%m%d")
        out_dir = REPO_ROOT / "research_out" / f"autonomous_fillmore_performance_investigation_{date_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cohort_rows: dict[str, list[dict[str, Any]]] = {c: [] for c in ALL_COHORTS}
    for r in rows:
        for c in label_cohorts(r):
            cohort_rows[c].append(r)

    mismatches = gate_prompt_mismatch_rows(rows)

    write_trade_cohorts_csv(out_dir / "trade_cohorts.csv", rows)
    write_common_denominators_csv(out_dir / "common_denominators.csv", cohort_rows)
    write_gate_prompt_mismatch_csv(out_dir / "gate_prompt_mismatch.csv", mismatches)
    write_report_md(
        out_dir / "REPORT.md",
        all_rows=rows,
        cohort_rows=cohort_rows,
        mismatches=mismatches,
        evidence_dir=evidence,
    )
    write_patch_recommendations(
        out_dir / "patch_recommendations.md",
        cohort_rows=cohort_rows,
        mismatches=mismatches,
    )

    print(f"wrote investigation pack -> {out_dir}")
    for f in ("REPORT.md", "trade_cohorts.csv", "common_denominators.csv",
              "gate_prompt_mismatch.csv", "patch_recommendations.md"):
        print(f"  - {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
