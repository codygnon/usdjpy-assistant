#!/usr/bin/env python3
"""Root-cause investigation for Autonomous Fillmore.

This is a read-only analysis harness. It consumes existing evidence exports and,
when available, refreshed AI suggestion history JSON. It does not change live
trading behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional
from zoneinfo import ZoneInfo


REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_TZ = ZoneInfo("America/Toronto")
PHASE3_VERSION = "autonomous_phase3_house_edge_v1"

DEFAULT_TIMELINE = REPO_ROOT / "research_out" / "autonomous_fillmore_change_timeline_20260430"
DEFAULT_EVIDENCE = REPO_ROOT / "research_out" / "autonomous_fillmore_evidence_20260429"
DEFAULT_PRIOR = REPO_ROOT / "research_out" / "autonomous_fillmore_performance_investigation_20260429"

GENERIC_TEXT = {"", "n/a", "na", "none", "null", "default", "-", "see thesis", "see analysis"}
GENERIC_CATALYST_PHRASES = {
    "level reject",
    "level reaction",
    "support reclaim",
    "resistance reject",
    "pullback",
    "pullback fade",
    "fade",
    "fade in chop",
    "trend continuation",
    "continuation",
    "reclaimed_support",
    "rejected_resistance",
    "support_reclaim",
    "resistance_reject",
}
STRUCTURE_TOKENS = {
    "half_yen",
    "whole_yen",
    "session_high",
    "session_low",
    "london",
    "ny",
    "tokyo",
    "oanda",
    "cluster",
    "support",
    "resistance",
    "reclaim",
    "reclaimed",
    "reject",
    "rejected",
    "level",
}
MICRO_TOKENS = {
    "micro",
    "confirmation",
    "m1",
    "m5",
    "aligned",
    "hh",
    "hl",
    "ll",
    "lh",
    "ema",
    "retest",
    "impulse",
    "pullback",
}
MATERIAL_TOKENS = {
    "intervention",
    "boj",
    "mof",
    "fed",
    "policy",
    "geopolitical",
    "war",
    "tariff",
    "flow",
    "flows",
    "fixing",
    "surprise",
    "hawkish",
    "dovish",
    "volatility",
    "regime",
    "material",
    "failed prior",
    "prior failure",
    "breakout failure",
    "stop run",
    "liquidity sweep",
    "news",
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
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def norm(raw: Any) -> str:
    return str(raw or "").strip()


def lower(raw: Any) -> str:
    return norm(raw).lower()


def local_iso(raw: Any) -> str:
    dt = parse_dt(raw)
    return dt.astimezone(LOCAL_TZ).isoformat() if dt else ""


def short(raw: Any, limit: int = 220) -> str:
    text = " ".join(norm(raw).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def load_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_json_items(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return []
    rows = data.get("items") if isinstance(data, dict) else data
    return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []


def default_live_json_paths() -> list[Path]:
    candidates = [
        Path("/tmp/newera8_ai_history_timeline.json"),
        Path("/tmp/newera8_ai_history_timeline_offset500.json"),
        Path("/tmp/kumatora2_ai_history_timeline.json"),
        Path("/tmp/newera8_ai_history_root_cause.json"),
        Path("/tmp/newera8_ai_history_root_cause_offset500.json"),
        Path("/tmp/kumatora2_ai_history_root_cause.json"),
    ]
    return [p for p in candidates if p.exists()]


def row_completeness(row: dict[str, Any]) -> int:
    score = sum(1 for v in row.values() if v not in (None, ""))
    if to_float(row.get("pips")) is not None or to_float(row.get("ledger_pips")) is not None:
        score += 20
    if row.get("named_catalyst"):
        score += 10
    if row.get("placed_order") or row.get("placed_order_json"):
        score += 5
    return score


def dedupe_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = "|".join([
            norm(row.get("suggestion_id")),
            norm(row.get("trade_id")),
            norm(row.get("created_utc")),
        ])
        if not key.replace("|", ""):
            key = f"anon:{id(row)}"
        prev = by_key.get(key)
        if prev is None or row_completeness(row) >= row_completeness(prev):
            by_key[key] = row
    return sorted(by_key.values(), key=lambda r: parse_dt(r.get("created_utc")) or datetime.min.replace(tzinfo=timezone.utc))


def load_all_rows(evidence: Path, live_jsons: list[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    sources: list[str] = []
    rows: list[dict[str, Any]] = []
    for name in ("combined_autonomous_joined.csv", "combined_autonomous_suggestions_flat.csv"):
        path = evidence / name
        loaded = load_csv(path)
        if loaded:
            sources.append(str(path))
            rows.extend(loaded)
    for path in live_jsons:
        loaded = load_json_items(path)
        if loaded:
            sources.append(str(path))
            rows.extend(loaded)
    return dedupe_rows(rows), sources


def write_csv(path: Path, rows: list[dict[str, Any]], required_columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(required_columns or [])
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def is_autonomous(row: dict[str, Any]) -> bool:
    placed = row.get("placed_order") if isinstance(row.get("placed_order"), dict) else {}
    placed_json = norm(row.get("placed_order_json"))
    return (
        lower(row.get("prompt_version")).startswith("autonomous")
        or lower(row.get("trade_id")).startswith("ai_autonomous:")
        or lower(row.get("entry_type")).startswith("ai_autonomous")
        or lower(row.get("is_autonomous")) == "true"
        or placed.get("autonomous") is True
        or '"autonomous": true' in placed_json.lower()
        or lower(row.get("rationale")).startswith("autonomous")
    )


def is_placed(row: dict[str, Any]) -> bool:
    return lower(row.get("action")) == "placed" and (to_float(row.get("lots")) or 0.0) > 0


def is_skip(row: dict[str, Any]) -> bool:
    return lower(row.get("decision")) == "skip" or lower(row.get("action")) == "skip" or (to_float(row.get("lots")) == 0.0)


def pips(row: dict[str, Any]) -> float | None:
    return to_float(row.get("pips")) if to_float(row.get("pips")) is not None else to_float(row.get("ledger_pips"))


def pnl(row: dict[str, Any]) -> float | None:
    for key in ("pnl_usd", "pnl", "profit", "ledger_profit"):
        value = to_float(row.get(key))
        if value is not None:
            return value
    return None


def mae(row: dict[str, Any]) -> float | None:
    return to_float(row.get("max_adverse_pips")) if to_float(row.get("max_adverse_pips")) is not None else to_float(row.get("ledger_max_adverse_pips"))


def mfe(row: dict[str, Any]) -> float | None:
    return to_float(row.get("max_favorable_pips")) if to_float(row.get("max_favorable_pips")) is not None else to_float(row.get("ledger_max_favorable_pips"))


def is_closed_with_pips(row: dict[str, Any]) -> bool:
    return is_placed(row) and bool(norm(row.get("closed_at") or row.get("ledger_exit_time_utc"))) and pips(row) is not None


def phase3_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if is_autonomous(r) and norm(r.get("prompt_version")) == PHASE3_VERSION]


def phase3_closed(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in phase3_rows(rows) if is_closed_with_pips(r)]


def lots_bucket(row: dict[str, Any]) -> str:
    lots = to_float(row.get("lots")) or 0.0
    if lots >= 8:
        return "8+"
    if lots >= 4:
        return "4-7.99"
    if lots >= 2:
        return "2-3.99"
    if lots > 0:
        return "0-1.99"
    return "0"


def rr_bucket(row: dict[str, Any]) -> str:
    rr = to_float(row.get("planned_rr_estimate"))
    if rr is None:
        return "(missing)"
    if rr < 1.0:
        return "<1.0"
    if rr < 1.3:
        return "1.0-1.3"
    if rr < 2.0:
        return "1.3-2.0"
    return "2.0+"


def meaningful_weakness_text(row: dict[str, Any]) -> bool:
    return lower(row.get("why_trade_despite_weakness")) not in GENERIC_TEXT


def weakness_signals(row: dict[str, Any]) -> list[str]:
    out: list[str] = []
    tf = lower(row.get("timeframe_alignment"))
    if tf in {"mixed", "countertrend"}:
        out.append(f"timeframe_alignment={tf}")
    if lower(row.get("repeat_trade_case")) == "blind_retry":
        out.append("repeat_trade_case=blind_retry")
    zone = lower(row.get("zone_memory_read"))
    if zone in {"failing_zone", "unresolved_chop"}:
        out.append(f"zone_memory_read={zone}")
    rr = to_float(row.get("planned_rr_estimate"))
    if rr is not None and rr < 1.0:
        out.append(f"planned_rr_estimate={rr:.2f}")
    if meaningful_weakness_text(row):
        out.append("why_trade_despite_weakness=set")
    return out


def has_weakness(row: dict[str, Any]) -> bool:
    return bool(weakness_signals(row))


def catalyst_text(row: dict[str, Any]) -> str:
    pieces = [
        norm(row.get("named_catalyst")),
        norm(row.get("trade_thesis")),
        norm(row.get("rationale")),
    ]
    return " | ".join(p for p in pieces if p)


def _contains_any(text: str, tokens: set[str]) -> bool:
    return any(token in text for token in tokens)


def score_text(text: Any) -> tuple[int, str]:
    text = norm(text)
    clean = " ".join(text.lower().replace("-", "_").split())
    if not clean or clean in GENERIC_TEXT:
        return 0, "empty_or_generic"
    if clean in GENERIC_CATALYST_PHRASES:
        return 0, "generic_phrase"
    has_material = _contains_any(clean, MATERIAL_TOKENS)
    has_micro = _contains_any(clean, MICRO_TOKENS)
    has_structure = _contains_any(clean, STRUCTURE_TOKENS)
    has_specific_number = any(ch.isdigit() for ch in clean)
    if has_material:
        return 3, "material_catalyst"
    if has_structure and (has_micro or "confirmation" in clean or "aligned" in clean):
        return 2, "structure_plus_micro"
    if has_micro and has_specific_number:
        return 2, "micro_confirmed_level"
    if has_structure or has_specific_number:
        return 1, "structure_only"
    if len(clean) < 18:
        return 0, "too_short_generic"
    return 1, "non_material_text"


def score_catalyst(row_or_text: dict[str, Any] | str) -> tuple[int, str]:
    if isinstance(row_or_text, dict):
        return score_text(row_or_text.get("named_catalyst"))
    return score_text(row_or_text)


def composite_catalyst_score(row: dict[str, Any]) -> tuple[int, str]:
    scored = [
        ("named", *score_text(row.get("named_catalyst"))),
        ("thesis", *score_text(row.get("trade_thesis"))),
        ("rationale", *score_text(row.get("rationale"))),
    ]
    source, score, label = max(scored, key=lambda item: item[1])
    return score, f"{source}:{label}"


def green_match_count(row: dict[str, Any]) -> int:
    count = 0
    side = lower(row.get("side"))
    family = lower(row.get("trigger_family"))
    tf = lower(row.get("timeframe_alignment"))
    h1 = lower(row.get("h1_regime"))
    session = lower(row.get("session"))
    fit = lower(row.get("trigger_fit"))
    reason = lower(row.get("trigger_reason")) + " " + lower(row.get("named_catalyst"))
    if side == "buy":
        count += 1
    if family == "critical_level_reaction":
        count += 1
    if tf == "aligned" or (tf == "mixed" and (h1 == "bull" or side == "buy")):
        count += 1
    if "london" in session or "ny" in session:
        count += 1
    if fit == "level_reaction":
        count += 1
    if any(token in reason for token in ("half_yen", "session", "oanda", "cluster", "support", "resistance")):
        count += 1
    return count


def red_match_count(row: dict[str, Any]) -> int:
    count = 0
    side = lower(row.get("side"))
    family = lower(row.get("trigger_family"))
    tf = lower(row.get("timeframe_alignment"))
    if side == "sell":
        count += 1
    if tf in {"mixed", "countertrend"}:
        count += 1
    if has_weakness(row):
        count += 1
    if family == "momentum_continuation" and side == "sell":
        count += 1
    if family == "critical_level_reaction" and tf == "mixed":
        count += 1
    if lots_bucket(row) == "8+":
        count += 1
    score, _ = score_catalyst(row)
    if score < 2:
        count += 1
    return count


def admission_failure_reasons(row: dict[str, Any]) -> list[str]:
    score, label = score_catalyst(row)
    reasons: list[str] = []
    if lower(row.get("timeframe_alignment")) == "mixed" and score < 2:
        reasons.append("mixed_with_weak_catalyst")
    if lower(row.get("timeframe_alignment")) == "mixed" and green_match_count(row) == 0:
        reasons.append("mixed_without_green_match")
    if lower(row.get("side")) == "sell" and has_weakness(row):
        reasons.append("sell_with_weakness")
    if lots_bucket(row) == "8+" and (has_weakness(row) or green_match_count(row) == 0):
        reasons.append("large_lot_without_clean_edge")
    if label in {"generic_phrase", "structure_only", "non_material_text"}:
        reasons.append("generic_or_structure_only_catalyst")
    if lower(row.get("trigger_family")) == "momentum_continuation" and lower(row.get("side")) == "sell":
        reasons.append("momentum_sell_fail_cell")
    if lower(row.get("trigger_family")) == "critical_level_reaction" and lower(row.get("timeframe_alignment")) == "mixed":
        reasons.append("critical_level_mixed_fail_cell")
    return reasons


def proposed_preventable_rule(row: dict[str, Any]) -> str:
    reasons = admission_failure_reasons(row)
    if "large_lot_without_clean_edge" in reasons:
        return "cap_lots_to_1_or_skip_when_weakness"
    if "sell_with_weakness" in reasons:
        return "sell_plus_weakness_requires_material_catalyst"
    if "mixed_with_weak_catalyst" in reasons or "mixed_without_green_match" in reasons:
        return "mixed_requires_score2_and_green_match"
    if "momentum_sell_fail_cell" in reasons:
        return "audit_or_pause_momentum_continuation_sells"
    if "critical_level_mixed_fail_cell" in reasons:
        return "tighten_critical_level_mixed_admission"
    return "review_manually"


def metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    closed = [r for r in rows if is_closed_with_pips(r)]
    placed = [r for r in rows if is_placed(r)]
    skips = [r for r in rows if is_skip(r)]
    pips_vals = [pips(r) for r in closed if pips(r) is not None]
    pnl_vals = [pnl(r) for r in closed if pnl(r) is not None]
    wins = [r for r in closed if (pips(r) or 0) > 0]
    losses = [r for r in closed if (pips(r) or 0) < 0]
    return {
        "calls": len(rows),
        "placed": len(placed),
        "placement_rate": len(placed) / len(rows) if rows else None,
        "skips": len(skips),
        "closed": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(closed) if closed else None,
        "net_pips": sum(pips_vals) if pips_vals else 0.0,
        "avg_pips": statistics.fmean(pips_vals) if pips_vals else None,
        "net_pnl": sum(pnl_vals) if pnl_vals else 0.0,
        "avg_pnl": statistics.fmean(pnl_vals) if pnl_vals else None,
    }


def phase3_loser_dossier_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    losers = [r for r in phase3_closed(rows) if (pips(r) or 0) < 0]
    losers.sort(key=lambda r: pnl(r) if pnl(r) is not None else 0)
    for row in losers:
        score, score_label = score_catalyst(row)
        out.append({
            "created_utc": row.get("created_utc"),
            "created_local": local_iso(row.get("created_utc")),
            "profile": row.get("profile"),
            "trade_id": row.get("trade_id"),
            "side": lower(row.get("side")),
            "lots": to_float(row.get("lots")) or "",
            "lot_bucket": lots_bucket(row),
            "pips": pips(row),
            "pnl_usd": pnl(row),
            "mae_pips": mae(row),
            "mfe_pips": mfe(row),
            "trigger_family": row.get("trigger_family") or "(blank)",
            "trigger_reason": row.get("trigger_reason") or "",
            "timeframe_alignment": row.get("timeframe_alignment") or "",
            "trigger_fit": row.get("trigger_fit") or "",
            "session": row.get("session") or "",
            "h1_regime": row.get("h1_regime") or "",
            "zone_memory_read": row.get("zone_memory_read") or "",
            "repeat_trade_case": row.get("repeat_trade_case") or "",
            "planned_rr_estimate": row.get("planned_rr_estimate") or "",
            "rr_bucket": rr_bucket(row),
            "exit_strategy": row.get("exit_strategy") or "",
            "exit_reason": row.get("exit_reason") or row.get("ledger_exit_reason") or "",
            "named_catalyst": row.get("named_catalyst") or "",
            "catalyst_score": score,
            "catalyst_label": score_label,
            "green_match_count": green_match_count(row),
            "red_match_count": red_match_count(row),
            "weakness_signals": "; ".join(weakness_signals(row)),
            "admission_failure_reasons": "; ".join(admission_failure_reasons(row)),
            "proposed_preventable_rule": proposed_preventable_rule(row),
            "trade_thesis": short(row.get("trade_thesis")),
            "rationale": short(row.get("rationale")),
        })
    return out


def group_metrics(rows: list[dict[str, Any]], group_key: Callable[[dict[str, Any]], str], extra: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[group_key(row)].append(row)
    out = []
    for key, group in grouped.items():
        closed = [r for r in group if is_closed_with_pips(r)]
        winners = [r for r in closed if (pips(r) or 0) > 0]
        losers = [r for r in closed if (pips(r) or 0) < 0]
        pips_vals = [pips(r) or 0.0 for r in closed]
        pnl_vals = [pnl(r) or 0.0 for r in closed]
        out.append({
            **extra,
            "key": key,
            "closed": len(closed),
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": round(len(winners) / len(closed), 4) if closed else "",
            "net_pips": round(sum(pips_vals), 1) if closed else "",
            "avg_pips": round(statistics.fmean(pips_vals), 2) if closed else "",
            "net_pnl_usd": round(sum(pnl_vals), 2) if closed else "",
            "avg_loser_pips": round(statistics.fmean([pips(r) or 0.0 for r in losers]), 2) if losers else "",
            "avg_winner_pips": round(statistics.fmean([pips(r) or 0.0 for r in winners]), 2) if winners else "",
        })
    out.sort(key=lambda r: (r["net_pips"] if isinstance(r["net_pips"], float) else 0))
    return out


def winner_loser_contrast_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    closed = phase3_closed(rows)
    dims: list[tuple[str, Callable[[dict[str, Any]], str]]] = [
        ("side", lambda r: lower(r.get("side")) or "(blank)"),
        ("trigger_family", lambda r: norm(r.get("trigger_family")) or "(blank)"),
        ("timeframe_alignment", lambda r: norm(r.get("timeframe_alignment")) or "(blank)"),
        ("trigger_fit", lambda r: norm(r.get("trigger_fit")) or "(blank)"),
        ("catalyst_score", lambda r: str(score_catalyst(r)[0])),
        ("lot_bucket", lots_bucket),
        ("family_side", lambda r: f"{norm(r.get('trigger_family')) or '(blank)'}|{lower(r.get('side')) or '(blank)'}"),
        ("family_side_tf", lambda r: f"{norm(r.get('trigger_family')) or '(blank)'}|{lower(r.get('side')) or '(blank)'}|{norm(r.get('timeframe_alignment')) or '(blank)'}"),
        ("exit_reason", lambda r: norm(r.get("exit_reason") or r.get("ledger_exit_reason")) or "(blank)"),
    ]
    out: list[dict[str, Any]] = []
    for dim, fn in dims:
        out.extend(group_metrics(closed, fn, {"dimension": dim}))
    return out


def catalyst_quality_audit_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in phase3_rows(rows):
        score, label = score_catalyst(row)
        thesis_score, thesis_label = score_text(row.get("trade_thesis"))
        rationale_score, rationale_label = score_text(row.get("rationale"))
        composite_score, composite_label = composite_catalyst_score(row)
        out.append({
            "created_utc": row.get("created_utc"),
            "profile": row.get("profile"),
            "action": row.get("action"),
            "decision": row.get("decision"),
            "side": row.get("side"),
            "lots": row.get("lots"),
            "closed": bool(row.get("closed_at")),
            "pips": pips(row),
            "pnl_usd": pnl(row),
            "trigger_family": row.get("trigger_family"),
            "timeframe_alignment": row.get("timeframe_alignment"),
            "named_catalyst": row.get("named_catalyst") or "",
            "named_catalyst_score": score,
            "named_catalyst_label": label,
            "trade_thesis_score": thesis_score,
            "trade_thesis_label": thesis_label,
            "rationale_score": rationale_score,
            "rationale_label": rationale_label,
            "composite_catalyst_score": composite_score,
            "composite_catalyst_label": composite_label,
            "green_match_count": green_match_count(row),
            "red_match_count": red_match_count(row),
            "weakness_signals": "; ".join(weakness_signals(row)),
            "text_audited": short(catalyst_text(row), 360),
            "trade_id": row.get("trade_id"),
        })
    return out


def mixed_alignment_audit_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in phase3_rows(rows):
        if lower(row.get("timeframe_alignment")) != "mixed":
            continue
        score, label = score_catalyst(row)
        failed = score < 2 or green_match_count(row) < 1 or lower(row.get("side")) == "sell" or lots_bucket(row) == "8+" or label in {"generic_phrase", "structure_only", "non_material_text"}
        out.append({
            "created_utc": row.get("created_utc"),
            "profile": row.get("profile"),
            "action": row.get("action"),
            "side": row.get("side"),
            "lots": row.get("lots"),
            "pips": pips(row),
            "pnl_usd": pnl(row),
            "trigger_family": row.get("trigger_family"),
            "trigger_reason": row.get("trigger_reason"),
            "named_catalyst": row.get("named_catalyst") or "",
            "catalyst_score": score,
            "catalyst_label": label,
            "green_match_count": green_match_count(row),
            "red_match_count": red_match_count(row),
            "failed_admission": int(failed),
            "admission_failure_reasons": "; ".join(admission_failure_reasons(row)),
            "trade_id": row.get("trade_id"),
        })
    return out


def family_side_matrix_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    closed = [r for r in rows if is_autonomous(r) and is_closed_with_pips(r)]
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in closed:
        key = (
            norm(row.get("prompt_version")) or "blank",
            norm(row.get("trigger_family")) or "(blank)",
            lower(row.get("side")) or "(blank)",
            norm(row.get("timeframe_alignment")) or "(blank)",
            norm(row.get("trigger_fit")) or "(blank)",
            norm(row.get("session")) or "(blank)",
            norm(row.get("h1_regime")) or "(blank)",
            norm(row.get("zone_memory_read")) or "(blank)",
            norm(row.get("repeat_trade_case")) or "(blank)",
            rr_bucket(row),
            lots_bucket(row),
        )
        grouped[key].append(row)
    out = []
    for key, group in grouped.items():
        pvals = [pips(r) or 0.0 for r in group]
        pnl_vals = [pnl(r) or 0.0 for r in group]
        winners = [r for r in group if (pips(r) or 0) > 0]
        losers = [r for r in group if (pips(r) or 0) < 0]
        out.append({
            "prompt_version": key[0],
            "trigger_family": key[1],
            "side": key[2],
            "timeframe_alignment": key[3],
            "trigger_fit": key[4],
            "session": key[5],
            "h1_regime": key[6],
            "zone_memory_read": key[7],
            "repeat_trade_case": key[8],
            "planned_rr_bucket": key[9],
            "lots_bucket": key[10],
            "closed": len(group),
            "win_rate": round(len(winners) / len(group), 4) if group else "",
            "net_pips": round(sum(pvals), 1),
            "net_pnl_usd": round(sum(pnl_vals), 2),
            "avg_loser_pips": round(statistics.fmean([pips(r) or 0.0 for r in losers]), 2) if losers else "",
            "loss_win_count_ratio": round(len(losers) / len(winners), 3) if winners else ("inf" if losers else ""),
        })
    out.sort(key=lambda r: (float(r["net_pips"]), float(r["net_pnl_usd"])))
    return out


def loss_asymmetry_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    closed = phase3_closed(rows)
    groups: list[tuple[str, list[dict[str, Any]]]] = [("phase3_overall", closed)]
    for side in sorted({lower(r.get("side")) or "(blank)" for r in closed}):
        groups.append((f"side={side}", [r for r in closed if (lower(r.get("side")) or "(blank)") == side]))
    for bucket in sorted({lots_bucket(r) for r in closed}):
        groups.append((f"lots={bucket}", [r for r in closed if lots_bucket(r) == bucket]))
    for family in sorted({norm(r.get("trigger_family")) or "(blank)" for r in closed}):
        groups.append((f"family={family}", [r for r in closed if (norm(r.get("trigger_family")) or "(blank)") == family]))
    for tf in sorted({norm(r.get("timeframe_alignment")) or "(blank)" for r in closed}):
        groups.append((f"timeframe={tf}", [r for r in closed if (norm(r.get("timeframe_alignment")) or "(blank)") == tf]))

    out = []
    for label, group in groups:
        winners = [r for r in group if (pips(r) or 0) > 0]
        losers = [r for r in group if (pips(r) or 0) < 0]
        out.append({
            "segment": label,
            "closed": len(group),
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": round(len(winners) / len(group), 4) if group else "",
            "net_pips": round(sum(pips(r) or 0.0 for r in group), 1) if group else "",
            "net_pnl_usd": round(sum(pnl(r) or 0.0 for r in group), 2) if group else "",
            "avg_winner_pips": round(statistics.fmean([pips(r) or 0.0 for r in winners]), 2) if winners else "",
            "avg_loser_pips": round(statistics.fmean([pips(r) or 0.0 for r in losers]), 2) if losers else "",
            "avg_winner_usd": round(statistics.fmean([pnl(r) or 0.0 for r in winners]), 2) if winners else "",
            "avg_loser_usd": round(statistics.fmean([pnl(r) or 0.0 for r in losers]), 2) if losers else "",
            "avg_winner_lots": round(statistics.fmean([to_float(r.get("lots")) or 0.0 for r in winners]), 2) if winners else "",
            "avg_loser_lots": round(statistics.fmean([to_float(r.get("lots")) or 0.0 for r in losers]), 2) if losers else "",
            "avg_winner_mae": round(statistics.fmean([mae(r) for r in winners if mae(r) is not None]), 2) if any(mae(r) is not None for r in winners) else "",
            "avg_loser_mae": round(statistics.fmean([mae(r) for r in losers if mae(r) is not None]), 2) if any(mae(r) is not None for r in losers) else "",
            "avg_winner_mfe": round(statistics.fmean([mfe(r) for r in winners if mfe(r) is not None]), 2) if any(mfe(r) is not None for r in winners) else "",
            "avg_loser_mfe": round(statistics.fmean([mfe(r) for r in losers if mfe(r) is not None]), 2) if any(mfe(r) is not None for r in losers) else "",
        })
    return out


def risk_score(row: dict[str, Any]) -> float:
    score, _ = score_catalyst(row)
    risk = 0.0
    risk += red_match_count(row) * 2.0
    risk -= green_match_count(row) * 0.8
    if score < 2:
        risk += 2.0
    if lower(row.get("side")) == "sell" and has_weakness(row):
        risk += 3.0
    if lower(row.get("timeframe_alignment")) == "mixed":
        risk += 1.5
    if lots_bucket(row) == "8+":
        risk += 3.0
    rr = to_float(row.get("planned_rr_estimate"))
    if rr is not None and rr < 1.0:
        risk += 1.5
    return risk


def rule_mixed_score2_green(row: dict[str, Any]) -> bool:
    return lower(row.get("timeframe_alignment")) == "mixed" and (score_catalyst(row)[0] < 2 or green_match_count(row) < 1)


def rule_sell_weak_requires_score3(row: dict[str, Any]) -> bool:
    return lower(row.get("side")) == "sell" and has_weakness(row) and score_catalyst(row)[0] < 3


def rule_large_lot_clean_only(row: dict[str, Any]) -> bool:
    return lots_bucket(row) == "8+" and not (lower(row.get("timeframe_alignment")) == "aligned" and not has_weakness(row) and green_match_count(row) >= 1 and score_catalyst(row)[0] >= 2)


def rule_generic_structural_catalyst(row: dict[str, Any]) -> bool:
    score, label = score_catalyst(row)
    return score <= 1 and label in {"generic_phrase", "structure_only", "non_material_text", "too_short_generic"}


def rule_momentum_sell(row: dict[str, Any]) -> bool:
    return lower(row.get("trigger_family")) == "momentum_continuation" and lower(row.get("side")) == "sell"


def rule_critical_level_mixed(row: dict[str, Any]) -> bool:
    return lower(row.get("trigger_family")) == "critical_level_reaction" and lower(row.get("timeframe_alignment")) == "mixed"


def counterfactual_row(name: str, rows: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool], mode: str = "skip") -> dict[str, Any]:
    phase = phase3_rows(rows)
    placed = [r for r in phase if is_placed(r)]
    closed = [r for r in placed if is_closed_with_pips(r)]
    blocked = [r for r in placed if predicate(r)]
    blocked_closed = [r for r in blocked if is_closed_with_pips(r)]
    blocked_winners = [r for r in blocked_closed if (pips(r) or 0.0) > 0]
    blocked_losers = [r for r in blocked_closed if (pips(r) or 0.0) < 0]
    if mode == "max_1_lot":
        affected = [r for r in blocked_closed if (to_float(r.get("lots")) or 0.0) > 1.0]
        saved_loss_usd = 0.0
        missed_winner_usd = 0.0
        for row in affected:
            lots = to_float(row.get("lots")) or 0.0
            value = pnl(row) or 0.0
            reduction = max(0.0, (lots - 1.0) / lots) if lots else 0.0
            if value < 0:
                saved_loss_usd += -value * reduction
            elif value > 0:
                missed_winner_usd += value * reduction
        return {
            "rule": name,
            "mode": mode,
            "blocked_trades": len(blocked),
            "blocked_winners": len(blocked_winners),
            "blocked_losers": len(blocked_losers),
            "saved_loss_pips": 0.0,
            "saved_loss_usd": round(saved_loss_usd, 2),
            "missed_winner_pips": 0.0,
            "missed_winner_usd": round(missed_winner_usd, 2),
            "net_delta_pips": 0.0,
            "net_delta_usd": round(saved_loss_usd - missed_winner_usd, 2),
            "new_placement_rate": round(len(placed) / len(phase), 4) if phase else "",
            "notes": "size cap only; pips unchanged, USD impact estimated by linear lot scaling",
        }
    saved_loss_pips = -sum(pips(r) or 0.0 for r in blocked_losers)
    saved_loss_usd = -sum(pnl(r) or 0.0 for r in blocked_losers)
    missed_winner_pips = sum(pips(r) or 0.0 for r in blocked_winners)
    missed_winner_usd = sum(pnl(r) or 0.0 for r in blocked_winners)
    return {
        "rule": name,
        "mode": mode,
        "blocked_trades": len(blocked),
        "blocked_winners": len(blocked_winners),
        "blocked_losers": len(blocked_losers),
        "saved_loss_pips": round(saved_loss_pips, 1),
        "saved_loss_usd": round(saved_loss_usd, 2),
        "missed_winner_pips": round(missed_winner_pips, 1),
        "missed_winner_usd": round(missed_winner_usd, 2),
        "net_delta_pips": round(saved_loss_pips - missed_winner_pips, 1),
        "net_delta_usd": round(saved_loss_usd - missed_winner_usd, 2),
        "new_placement_rate": round((len(placed) - len(blocked)) / len(phase), 4) if phase else "",
        "notes": "",
    }


def placement_target_rule(rows: list[dict[str, Any]], target_rate: float) -> dict[str, Any]:
    phase = phase3_rows(rows)
    placed = [r for r in phase if is_placed(r)]
    allowed = int(math.floor(len(phase) * target_rate))
    block_count = max(0, len(placed) - allowed)
    risky = sorted(placed, key=lambda r: (-risk_score(r), parse_dt(r.get("created_utc")) or datetime.min.replace(tzinfo=timezone.utc)))
    blocked_ids = {id(r) for r in risky[:block_count]}
    return counterfactual_row(
        f"placement_rate_target_{int(target_rate * 100)}pct_highest_risk_first",
        rows,
        lambda r: id(r) in blocked_ids,
    )


def rule_counterfactual_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rules = [
        counterfactual_row("mixed_requires_catalyst_score2_and_green_match", rows, rule_mixed_score2_green),
        counterfactual_row("sell_plus_weakness_requires_catalyst_score3_skip", rows, rule_sell_weak_requires_score3),
        counterfactual_row("sell_plus_weakness_requires_catalyst_score3_max_1_lot", rows, rule_sell_weak_requires_score3, mode="max_1_lot"),
        counterfactual_row("large_lots_only_on_aligned_no_weakness_green_score2", rows, rule_large_lot_clean_only),
        counterfactual_row("generic_structural_catalysts_skip", rows, rule_generic_structural_catalyst),
        counterfactual_row("momentum_continuation_sells_skip", rows, rule_momentum_sell),
        counterfactual_row("critical_level_reaction_mixed_skip", rows, rule_critical_level_mixed),
        placement_target_rule(rows, 0.25),
        placement_target_rule(rows, 0.35),
        placement_target_rule(rows, 0.50),
    ]
    rules.sort(key=lambda r: (float(r["net_delta_pips"]), float(r["net_delta_usd"])), reverse=True)
    return rules


def fmt_num(value: Any, digits: int = 1) -> str:
    if value is None or value == "":
        return "n/a"
    return f"{float(value):.{digits}f}"


def fmt_money(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    return f"${float(value):,.2f}"


def pct(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def load_supporting_tables(timeline: Path, prior: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        "prompt_version_performance": load_csv(timeline / "prompt_version_performance.csv"),
        "daily_prompt_performance": load_csv(timeline / "daily_prompt_performance.csv"),
        "major_change_log": load_csv(timeline / "major_change_log.csv"),
        "worst_trades": load_csv(timeline / "worst_trades.csv"),
        "trade_cohorts": load_csv(prior / "trade_cohorts.csv"),
        "gate_prompt_mismatch": load_csv(prior / "gate_prompt_mismatch.csv"),
        "common_denominators": load_csv(prior / "common_denominators.csv"),
    }


def generate_solution_ideas(out_dir: Path, counterfactuals: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    by_rule = {row["rule"]: row for row in counterfactuals}

    def bullet(row: dict[str, Any]) -> str:
        return (
            f"- **{row['rule']}** ({row['mode']}): net {row['net_delta_pips']}p / {fmt_money(row['net_delta_usd'])}; "
            f"blocks {row['blocked_trades']} trades ({row['blocked_losers']} losers, {row['blocked_winners']} winners), "
            f"new placement rate {pct(row['new_placement_rate'])}."
        )

    high_confidence_names = [
        "large_lots_only_on_aligned_no_weakness_green_score2",
        "sell_plus_weakness_requires_catalyst_score3_max_1_lot",
    ]
    medium_confidence_names = [
        "placement_rate_target_50pct_highest_risk_first",
        "critical_level_reaction_mixed_skip",
        "placement_rate_target_35pct_highest_risk_first",
        "momentum_continuation_sells_skip",
    ]
    hard_backstop_names = [
        "mixed_requires_catalyst_score2_and_green_match",
        "sell_plus_weakness_requires_catalyst_score3_skip",
        "critical_level_reaction_mixed_skip",
    ]
    lines = [
        "# Solution Ideas From Root-Cause Investigation",
        "",
        "These ideas are ranked by historical counterfactual impact and implementation cost. This is not a max-daily-loss shutdown plan.",
        "",
        "## High confidence / low opportunity cost",
        "",
    ]
    for name in high_confidence_names:
        if name in by_rule:
            lines.append(bullet(by_rule[name]))
    lines.append("- **rolling placement-rate callout**: no counterfactual pips estimate, but low risk; Phase 3 placed 76.8% of calls and should be forced to explain why continued activity is justified.")
    lines.extend([
        "",
        "Prompt guidance:",
        "- Treat structure-only catalysts as insufficient. A named level is location, not edge.",
        "- Mixed alignment needs catalyst score >=2 and at least one green-pattern match.",
        "- Sell + weakness should default to skip unless the catalyst is truly material.",
        "",
        "Server validation/backstop:",
        "- Add validation that detects generic `named_catalyst` values even when they include a level name.",
        "- Add a post-parse warning or veto for mixed alignment with catalyst score <2.",
        "",
        "Sizing discipline:",
        "- Cap large lots to clean aligned setups with no weakness and catalyst score >=2.",
        "- If a weak setup is still allowed for learning, force 1 lot rather than 8 lots.",
        "",
        "## Medium confidence / needs follow-up sample",
        "",
    ])
    for name in medium_confidence_names:
        if name in by_rule:
            lines.append(bullet(by_rule[name]))
    lines.extend([
        "- Track average loser lots vs average winner lots in the UI and prompt memory, because the win rate can improve while expectancy still worsens.",
        "- Add a rolling placement-rate callout. Phase 3 placed too often for a prompt designed to raise selectivity.",
        "",
        "## Hard backstop candidates only if prompt guidance keeps failing",
        "",
    ])
    for name in hard_backstop_names:
        if name in by_rule:
            lines.append(bullet(by_rule[name]))
    lines.extend([
        "- Hard cap lots >1 on any setup with mixed/countertrend alignment or red matches greater than green matches.",
        "",
        "## Phase 3 baseline being repaired",
        "",
        f"- Calls: {summary['calls']}; placed: {summary['placed']}; placement rate: {pct(summary['placement_rate'])}.",
        f"- Closed: {summary['closed']}; win rate: {pct(summary['win_rate'])}; net pips: {fmt_num(summary['net_pips'])}; net P&L: {fmt_money(summary['net_pnl'])}.",
    ])
    (out_dir / "SOLUTION_IDEAS.md").write_text("\n".join(lines) + "\n")


def generate_report(
    out_dir: Path,
    rows: list[dict[str, Any]],
    source_files: list[str],
    supporting: dict[str, list[dict[str, Any]]],
    counterfactuals: list[dict[str, Any]],
    loser_rows: list[dict[str, Any]],
    mixed_rows: list[dict[str, Any]],
    asymmetry_rows: list[dict[str, Any]],
) -> None:
    phase = phase3_rows(rows)
    closed = phase3_closed(rows)
    summary = metric_summary(phase)
    sell_closed = [r for r in closed if lower(r.get("side")) == "sell"]
    buy_closed = [r for r in closed if lower(r.get("side")) == "buy"]
    mixed_closed = [r for r in closed if lower(r.get("timeframe_alignment")) == "mixed"]
    large_lot_closed = [r for r in closed if lots_bucket(r) == "8+"]
    sell_weak = [r for r in closed if lower(r.get("side")) == "sell" and has_weakness(r)]
    overall_asym = next((r for r in asymmetry_rows if r["segment"] == "phase3_overall"), {})
    top_rules = counterfactuals[:5]

    lines = [
        "# Autonomous Fillmore Root-Cause Investigation",
        "",
        "This investigation focuses on why Fillmore remains net-negative after prompt changes, with Phase 3 losers as the immediate debugging target.",
        "",
        "## Executive Findings",
        "",
        f"- Phase 3 universe: {summary['calls']} calls, {summary['placed']} placed, {summary['closed']} pips-counted closes.",
        f"- Phase 3 performance: WR {pct(summary['win_rate'])}, {fmt_num(summary['net_pips'])}p, {fmt_money(summary['net_pnl'])}, placement rate {pct(summary['placement_rate'])}.",
        f"- Buy side: {len(buy_closed)} closes, {fmt_num(sum(pips(r) or 0.0 for r in buy_closed))}p, {fmt_money(sum(pnl(r) or 0.0 for r in buy_closed))}.",
        f"- Sell side: {len(sell_closed)} closes, {fmt_num(sum(pips(r) or 0.0 for r in sell_closed))}p, {fmt_money(sum(pnl(r) or 0.0 for r in sell_closed))}.",
        f"- Mixed alignment: {len(mixed_closed)} closes, {fmt_num(sum(pips(r) or 0.0 for r in mixed_closed))}p, {fmt_money(sum(pnl(r) or 0.0 for r in mixed_closed))}.",
        f"- Large lots: {len(large_lot_closed)} closes, {fmt_num(sum(pips(r) or 0.0 for r in large_lot_closed))}p, {fmt_money(sum(pnl(r) or 0.0 for r in large_lot_closed))}.",
        f"- Sell + weakness: {len(sell_weak)} closes, {fmt_num(sum(pips(r) or 0.0 for r in sell_weak))}p, {fmt_money(sum(pnl(r) or 0.0 for r in sell_weak))}.",
        "",
        "## Root-Cause Hierarchy",
        "",
        "1. **Selectivity failed.** Phase 3 placed too frequently for a prompt that was supposed to raise the bar. A high win rate did not matter because too many marginal setups still became trades.",
        "2. **Catalyst language was too easy to satisfy.** Many losing trades named a structure, but did not explain why that structure should beat the losing base rate.",
        "3. **Loss asymmetry and lot exposure dominated.** Losers were larger than winners, and high-lot losses amplified the damage.",
        "4. **Mixed alignment remained admissible.** Mixed setups often had enough words to pass, but not enough actual edge.",
        "5. **Sell-side weakness persisted.** The sell-side rule improved symptoms but did not eliminate the loss cell.",
        "",
        "## Why 60%+ Win Rate Still Lost Money",
        "",
        f"- Average winner pips: {overall_asym.get('avg_winner_pips', 'n/a')}; average loser pips: {overall_asym.get('avg_loser_pips', 'n/a')}.",
        f"- Average winner USD: {fmt_money(overall_asym.get('avg_winner_usd'))}; average loser USD: {fmt_money(overall_asym.get('avg_loser_usd'))}.",
        f"- Average winner lots: {overall_asym.get('avg_winner_lots', 'n/a')}; average loser lots: {overall_asym.get('avg_loser_lots', 'n/a')}.",
        "This isolates the issue: selection is weak, then sizing amplifies the weak selections.",
        "",
        "## Best Counterfactual Fixes",
        "",
        "| Rule | Mode | Blocked | Winners blocked | Losers blocked | Net pips | Net USD | New placement rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in top_rules:
        lines.append(
            f"| {row['rule']} | {row['mode']} | {row['blocked_trades']} | {row['blocked_winners']} | {row['blocked_losers']} | "
            f"{row['net_delta_pips']} | {fmt_money(row['net_delta_usd'])} | {pct(row['new_placement_rate'])} |"
        )
    lines.extend([
        "",
        "## Phase 3 Loser Themes",
        "",
    ])
    for rule, count in Counter(r["proposed_preventable_rule"] for r in loser_rows).most_common():
        lines.append(f"- {rule}: {count} losing trades.")
    lines.extend([
        "",
        "## Mixed-Alignment Admission",
        "",
        f"- Mixed rows audited: {len(mixed_rows)}.",
        f"- Failed mixed admissions: {sum(1 for r in mixed_rows if str(r.get('failed_admission')) == '1')}.",
        "- The recommended standard is catalyst score >=2, at least one green-pattern match, and no sell-side weakness.",
        "",
        "## Output Files",
        "",
        "- `phase3_loser_dossier.csv`",
        "- `phase3_winner_loser_contrast.csv`",
        "- `catalyst_quality_audit.csv`",
        "- `mixed_alignment_audit.csv`",
        "- `family_side_matrix.csv`",
        "- `loss_asymmetry.csv`",
        "- `rule_counterfactuals.csv`",
        "- `SOLUTION_IDEAS.md`",
        "",
        "## Data Sources",
        "",
    ])
    lines.extend(f"- `{src}`" for src in source_files)
    if supporting.get("prompt_version_performance"):
        lines.append("- Timeline performance tables were loaded and used as context.")
    lines.extend([
        "",
        "## Caveats",
        "",
        "- `prompt_version` is treated as the behavioral boundary; git commit time is context only.",
        "- Counterfactuals are historical filters, not guaranteed future performance.",
        "- Pips are primary; USD exposes sizing damage.",
        "- This is not a max-daily-loss shutdown recommendation.",
    ])
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


def run_analysis(
    *,
    evidence: Path = DEFAULT_EVIDENCE,
    timeline: Path = DEFAULT_TIMELINE,
    prior: Path = DEFAULT_PRIOR,
    out: Path,
    live_jsons: list[Path] | None = None,
) -> dict[str, Any]:
    live_jsons = live_jsons if live_jsons is not None else default_live_json_paths()
    out.mkdir(parents=True, exist_ok=True)
    rows, source_files = load_all_rows(evidence, live_jsons)
    rows = [r for r in rows if is_autonomous(r)]
    supporting = load_supporting_tables(timeline, prior)

    loser_rows = phase3_loser_dossier_rows(rows)
    contrast_rows = winner_loser_contrast_rows(rows)
    catalyst_rows = catalyst_quality_audit_rows(rows)
    mixed_rows = mixed_alignment_audit_rows(rows)
    matrix_rows = family_side_matrix_rows(rows)
    asymmetry_rows = loss_asymmetry_rows(rows)
    counterfactuals = rule_counterfactual_rows(rows)
    summary = metric_summary(phase3_rows(rows))

    write_csv(out / "phase3_loser_dossier.csv", loser_rows)
    write_csv(out / "phase3_winner_loser_contrast.csv", contrast_rows)
    write_csv(out / "catalyst_quality_audit.csv", catalyst_rows)
    write_csv(out / "mixed_alignment_audit.csv", mixed_rows)
    write_csv(out / "family_side_matrix.csv", matrix_rows)
    write_csv(out / "loss_asymmetry.csv", asymmetry_rows)
    write_csv(out / "rule_counterfactuals.csv", counterfactuals)
    generate_solution_ideas(out, counterfactuals, summary)
    generate_report(out, rows, source_files, supporting, counterfactuals, loser_rows, mixed_rows, asymmetry_rows)
    (out / "source_manifest.json").write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "evidence": str(evidence),
        "timeline": str(timeline),
        "prior": str(prior),
        "live_jsons": [str(p) for p in live_jsons],
        "source_files": source_files,
        "autonomous_rows": len(rows),
        "phase3_summary": summary,
    }, indent=2, sort_keys=True))
    return {
        "rows": rows,
        "phase3_summary": summary,
        "outputs": {
            "losers": loser_rows,
            "contrast": contrast_rows,
            "catalyst": catalyst_rows,
            "mixed": mixed_rows,
            "matrix": matrix_rows,
            "asymmetry": asymmetry_rows,
            "counterfactuals": counterfactuals,
        },
    }


def default_out_dir() -> Path:
    return REPO_ROOT / "research_out" / f"autonomous_fillmore_root_cause_{datetime.now(LOCAL_TZ).strftime('%Y%m%d')}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", default=str(DEFAULT_EVIDENCE))
    parser.add_argument("--timeline", default=str(DEFAULT_TIMELINE))
    parser.add_argument("--prior", default=str(DEFAULT_PRIOR))
    parser.add_argument("--out", default=str(default_out_dir()))
    parser.add_argument("--live-json", action="append", default=None, help="AI suggestion history JSON path; can be repeated")
    args = parser.parse_args(argv)

    live_jsons = [Path(p) for p in args.live_json] if args.live_json else None
    result = run_analysis(
        evidence=Path(args.evidence),
        timeline=Path(args.timeline),
        prior=Path(args.prior),
        out=Path(args.out),
        live_jsons=live_jsons,
    )
    summary = result["phase3_summary"]
    print(f"Phase 3: calls={summary['calls']} placed={summary['placed']} closed={summary['closed']} "
          f"WR={pct(summary['win_rate'])} net_pips={fmt_num(summary['net_pips'])} net_pnl={fmt_money(summary['net_pnl'])}")
    print(f"Wrote {Path(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
