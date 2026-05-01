#!/usr/bin/env python3
"""Reasoning deep-dive for Autonomous Fillmore.

This read-only harness looks past the trade result cohorts and audits the text
Fillmore wrote before it traded: rationale, analysis, thesis, catalyst,
side-bias check, weakness admissions, and skip reasons.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo


REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_TZ = ZoneInfo("America/Toronto")
PHASE3_VERSION = "autonomous_phase3_house_edge_v1"
DEFAULT_EVIDENCE = REPO_ROOT / "research_out" / "autonomous_fillmore_evidence_20260429"

GENERIC_TEXT = {"", "n/a", "na", "none", "null", "default", "-", "see thesis", "see analysis"}
GENERIC_CATALYSTS = {
    "level reject", "level reaction", "support reclaim", "reclaimed support",
    "resistance reject", "rejected resistance", "pullback", "pullback fade",
    "fade", "fade in chop", "trend continuation", "continuation", "support",
    "resistance", "structure", "technical setup", "price action",
    "fresh setup", "fresh autonomous setup",
    *GENERIC_TEXT,
}
STRUCTURE_TOKENS = {
    "support", "resistance", "reclaim", "reclaimed", "reject", "rejected",
    "level", "half_yen", "whole_yen", "session_high", "session_low",
    "session high", "session low", "range high", "range low", "cluster",
    "oanda", "pivot", "shelf", "trendline", "ema", "vwap",
}
MICRO_TOKENS = {
    "m1", "m3", "m5", "micro", "confirmation", "confirmed", "retest",
    "sweep", "acceptance", "close above", "close below", "breakout",
    "breakdown", "impulse", "higher low", "lower high", "higher high",
    "lower low", "ema9", "ema21",
}
MATERIAL_PHRASES = {
    "boj", "mof", "finance minister", "intervention", "rate check",
    "hawkish surprise", "dovish surprise", "policy shift", "policy divergence",
    "macro catalyst", "macro release", "macro surprise", "flow shift",
    "safe-haven", "safe haven", "safe-haven flow", "safe haven flow",
    "geopolitical risk", "war premium", "war-premium", "us-japan",
    "treasury yield", "yield spike", "yield compression", "cpi", "nfp",
    "fomc", "volatility regime", "liquidity shift", "fixing flow",
    "option expiry", "real money", "material change", "prior failure",
    "failed prior", "break of structure", "regime shift",
}
NEGATIONS = {
    "no", "not", "without", "absent", "contradict", "contradicts",
    "contradicted", "mixed", "unclear", "generic", "does not confirm",
    "doesn't confirm", "fails to confirm", "not confirmed", "not material",
}
ADVERSE_CONTEXT_PATTERNS = (
    r"\bdespite\b",
    r"\beven though\b",
    r"\bhowever\b",
    r"\balthough\b",
    r"\bbut (?:the )?(?:broader|higher|h1|m15|m5|m1|policy|macro|cross|jpy|yen|tape|trend|structure|session|intraday)\b",
    r"\bmixed(?:-| )?(?:tape|alignment|structure|backdrop|context|trend|overall)\b",
    r"\bcontradict\w*\b",
    r"\bconflict(?:s|ing)?\b",
    r"\bjpy[- ]weak\b",
    r"\byen weakness\b",
    r"\bbroader .* pressure\b",
    r"\bstructurally weaker\b",
    r"\bnot enough to cancel\b",
)
WEAK_PERMISSION_PATTERNS = (
    r"\bonly as a small\b",
    r"\bsmall .*?(?:because|despite|given|respect|due to)\b",
    r"\bsize (?:stays|is|kept) (?:small|minimal|reduced)\b",
    r"\bmust be quick\b",
    r"\bquick .*?(?:because|given|despite|while)\b",
    r"\bdisciplined .*?(?:because|given|due to|while)\b",
    r"\bprobe\b",
    r"\btactical\b",
    r"\bthin\b",
    r"\bmarginal\b",
    r"\breduced conviction\b",
    r"\blow conviction\b",
)
EVIDENCE_CLAIMS = {
    "fresh", "micro", "micro-confirmed", "confirmation", "confirmed",
    "base rate", "not generic", "specific", "clean", "bounded", "tight",
}


def parse_dt(raw: Any) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def local_iso(raw: Any) -> str:
    dt = parse_dt(raw)
    return dt.astimezone(LOCAL_TZ).isoformat() if dt else ""


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
    if isinstance(raw, (dict, list)):
        return json.dumps(raw, sort_keys=True)
    return str(raw or "").strip()


def lower(raw: Any) -> str:
    return re.sub(r"\s+", " ", norm(raw).lower())


def short(raw: Any, limit: int = 260) -> str:
    text = " ".join(norm(raw).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def phrase_regex(phrase: str) -> re.Pattern[str]:
    parts = [re.escape(part) for part in re.split(r"[\s_-]+", phrase.strip()) if part]
    body = r"[\s_-]+".join(parts)
    return re.compile(rf"(?<![a-z0-9]){body}(?![a-z0-9])")


def has_phrase(text: str, phrases: Iterable[str]) -> bool:
    return any(phrase_regex(phrase).search(text) for phrase in phrases)


def has_pattern(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def has_clean_material_phrase(text: str) -> bool:
    for phrase in MATERIAL_PHRASES:
        for match in phrase_regex(phrase).finditer(text):
            window = text[max(0, match.start() - 40): min(len(text), match.end() + 40)]
            if has_phrase(window, NEGATIONS):
                continue
            return True
    return False


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
    score = sum(1 for v in row.values() if v not in (None, "", {}, []))
    if row.get("prompt_version"):
        score += 15
    if row.get("named_catalyst"):
        score += 15
    if to_float(row.get("pips")) is not None or to_float(row.get("ledger_pips")) is not None:
        score += 12
    if isinstance(row.get("market_snapshot"), dict):
        score += 8
    if isinstance(row.get("features"), dict):
        score += 8
    return score


def dedupe_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = norm(row.get("suggestion_id"))
        if not key:
            key = "|".join([norm(row.get("trade_id")), norm(row.get("created_utc")), norm(row.get("profile"))])
        if not key.replace("|", ""):
            key = f"anon:{id(row)}"
        prev = by_key.get(key)
        if prev is None or row_completeness(row) >= row_completeness(prev):
            by_key[key] = row
    return sorted(by_key.values(), key=lambda r: parse_dt(r.get("created_utc")) or datetime.min.replace(tzinfo=timezone.utc))


def load_all_rows(evidence: Path, live_jsons: list[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    sources: list[str] = []
    for name in ("combined_autonomous_joined.csv", "combined_autonomous_suggestions_flat.csv"):
        path = evidence / name
        loaded = load_csv(path)
        if loaded:
            rows.extend(loaded)
            sources.append(str(path))
    for path in live_jsons:
        loaded = load_json_items(path)
        if loaded:
            rows.extend(loaded)
            sources.append(str(path))
    return dedupe_rows(rows), sources


def nested(row: dict[str, Any], key: str) -> Any:
    if key in row and row[key] not in (None, ""):
        return row[key]
    features = row.get("features")
    if isinstance(features, dict) and features.get(key) not in (None, ""):
        return features.get(key)
    placed = row.get("placed_order")
    if isinstance(placed, dict) and placed.get(key) not in (None, ""):
        return placed.get(key)
    return None


def is_autonomous(row: dict[str, Any]) -> bool:
    placed = row.get("placed_order") if isinstance(row.get("placed_order"), dict) else {}
    placed_json = lower(row.get("placed_order_json"))
    return (
        lower(row.get("prompt_version")).startswith("autonomous")
        or lower(row.get("trade_id")).startswith("ai_autonomous:")
        or lower(row.get("entry_type")).startswith("ai_autonomous")
        or lower(row.get("is_autonomous")) == "true"
        or placed.get("autonomous") is True
        or '"autonomous": true' in placed_json
    )


def is_placed(row: dict[str, Any]) -> bool:
    return lower(row.get("action")) == "placed" and (to_float(row.get("lots")) or 0.0) > 0


def is_trade_decision(row: dict[str, Any]) -> bool:
    decision = lower(row.get("decision"))
    return decision == "trade" or is_placed(row)


def is_skip(row: dict[str, Any]) -> bool:
    return lower(row.get("decision")) == "skip" or (to_float(row.get("lots")) == 0.0 and not is_placed(row))


def pips(row: dict[str, Any]) -> float | None:
    return to_float(row.get("pips")) if to_float(row.get("pips")) is not None else to_float(row.get("ledger_pips"))


def pnl(row: dict[str, Any]) -> float | None:
    for key in ("pnl_usd", "pnl", "profit", "ledger_profit"):
        value = to_float(row.get(key))
        if value is not None:
            return value
    return None


def is_closed(row: dict[str, Any]) -> bool:
    return is_placed(row) and pips(row) is not None and bool(row.get("closed_at") or row.get("ledger_exit_time_utc"))


def phase3(row: dict[str, Any]) -> bool:
    return norm(row.get("prompt_version")) == PHASE3_VERSION


def reasoning_text(row: dict[str, Any]) -> str:
    keys = [
        "rationale", "trade_thesis", "named_catalyst", "side_bias_check",
        "why_not_stop", "whats_different", "low_rr_edge", "countertrend_edge",
        "why_trade_despite_weakness", "skip_reason", "exit_plan",
    ]
    return " | ".join(norm(row.get(k)) for k in keys if norm(row.get(k)))


def core_reasoning_text(row: dict[str, Any]) -> str:
    """Reasoning without the full checklist-style analysis tail."""
    keys = [
        "trade_thesis", "named_catalyst", "side_bias_check", "why_not_stop",
        "whats_different", "low_rr_edge", "countertrend_edge",
        "why_trade_despite_weakness", "skip_reason", "exit_plan",
    ]
    rat = norm(row.get("rationale"))
    marker = "ANALYSIS:"
    idx = rat.upper().find(marker)
    if idx >= 0:
        rat = rat[:idx].strip()
    pieces = [rat, *[norm(row.get(k)) for k in keys if norm(row.get(k))]]
    return " | ".join(p for p in pieces if p)


def analysis_text(row: dict[str, Any]) -> str:
    rat = norm(row.get("rationale"))
    marker = "ANALYSIS:"
    idx = rat.upper().find(marker)
    return rat[idx + len(marker):].strip() if idx >= 0 else ""


def catalyst_score_text(text: Any) -> tuple[int, str]:
    clean = lower(text).strip(" .:-")
    if not clean or clean in GENERIC_TEXT:
        return 0, "empty_or_generic"
    if clean in GENERIC_CATALYSTS or len(clean) < 12:
        return 0, "generic_or_short"
    if has_clean_material_phrase(clean):
        return 3, "material_catalyst"
    has_structure = has_phrase(clean, STRUCTURE_TOKENS)
    has_micro = has_phrase(clean, MICRO_TOKENS)
    has_level = bool(re.search(r"\d", clean))
    if has_structure and has_micro:
        return 2, "structure_plus_micro"
    if has_micro and has_level:
        return 2, "micro_confirmed_level"
    if has_structure or has_level:
        return 1, "structure_or_level_only"
    if len(clean.split()) >= 4:
        return 1, "non_material_text"
    return 0, "too_short_generic"


def catalyst_score(row: dict[str, Any]) -> tuple[int, str, str]:
    candidates = [
        ("named_catalyst", *catalyst_score_text(row.get("named_catalyst"))),
        ("trade_thesis", *catalyst_score_text(row.get("trade_thesis"))),
        ("rationale", *catalyst_score_text(row.get("rationale"))),
    ]
    source, score, label = max(candidates, key=lambda item: item[1])
    return score, label, source


def timeframe(row: dict[str, Any]) -> str:
    return lower(nested(row, "timeframe_alignment"))


def family(row: dict[str, Any]) -> str:
    return lower(nested(row, "trigger_family"))


def trigger_fit(row: dict[str, Any]) -> str:
    return lower(nested(row, "trigger_fit"))


def zone(row: dict[str, Any]) -> str:
    return lower(nested(row, "zone_memory_read"))


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


def session_label(row: dict[str, Any]) -> str:
    session = lower(row.get("session") or nested(row, "session"))
    if session and session != "{}":
        return session
    snap = row.get("market_snapshot")
    if isinstance(snap, dict):
        sess = snap.get("session") or {}
        if isinstance(sess, dict):
            overlap = lower(sess.get("overlap"))
            active = "+".join(lower(x) for x in (sess.get("active_sessions") or []) if x)
            return overlap or active
    return ""


def h1_regime(row: dict[str, Any]) -> str:
    h1 = lower(row.get("h1_regime") or nested(row, "h1_regime"))
    if h1 and h1 != "unknown":
        return h1
    snap = row.get("market_snapshot")
    if isinstance(snap, dict):
        ta = snap.get("ta_snapshot") or {}
        if isinstance(ta, dict):
            h1_obj = ta.get("H1") or {}
            if isinstance(h1_obj, dict):
                return lower(h1_obj.get("regime"))
    return h1


def weakness_signals(row: dict[str, Any]) -> list[str]:
    out: list[str] = []
    tf = timeframe(row)
    if tf in {"mixed", "countertrend"}:
        out.append(f"timeframe={tf}")
    if lower(row.get("repeat_trade_case")) == "blind_retry":
        out.append("blind_retry")
    zn = zone(row)
    if zn in {"failing_zone", "unresolved_chop"}:
        out.append(f"zone={zn}")
    rr = to_float(row.get("planned_rr_estimate"))
    if rr is not None and rr < 1.0:
        out.append(f"rr={rr:.2f}")
    if lower(row.get("why_trade_despite_weakness")) not in GENERIC_TEXT:
        out.append("weakness_text")
    return out


def green_match_count(row: dict[str, Any]) -> int:
    side = lower(row.get("side"))
    fam = family(row)
    tf = timeframe(row)
    fit = trigger_fit(row)
    sess = session_label(row)
    h1 = h1_regime(row)
    text = lower(" ".join([norm(row.get("named_catalyst")), norm(row.get("trade_thesis")), norm(row.get("trigger_reason"))]))
    score = 0
    if side == "buy":
        score += 1
    if fam == "critical_level_reaction":
        score += 1
    if fit == "level_reaction":
        score += 1
    if tf == "aligned" or (tf == "mixed" and "bull" in h1 and side == "buy"):
        score += 1
    if "london" in sess or "ny" in sess or "new york" in sess:
        score += 1
    if has_phrase(text, STRUCTURE_TOKENS):
        score += 1
    return score


def reasoning_flags(row: dict[str, Any]) -> list[str]:
    text = lower(core_reasoning_text(row))
    score, label, _source = catalyst_score(row)
    flags: list[str] = []
    if score == 0:
        flags.append("empty_or_generic_catalyst")
    if score == 1:
        flags.append("location_as_edge")
    if score < 2 and (timeframe(row) in {"mixed", "countertrend"}):
        flags.append("mixed_or_countertrend_weak_catalyst")
    if lower(row.get("side")) == "sell" and weakness_signals(row) and score < 3:
        flags.append("sell_weakness_not_material")
    if lots_bucket(row) == "8+" and (weakness_signals(row) or green_match_count(row) < 4 or score < 2):
        flags.append("large_lot_reasoning_gap")
    if has_pattern(text, ADVERSE_CONTEXT_PATTERNS):
        flags.append("contradiction_admitted")
        if score < 3:
            flags.append("contradiction_not_materially_resolved")
    if has_pattern(text, WEAK_PERMISSION_PATTERNS) and is_trade_decision(row):
        flags.append("weak_trade_language_used_as_permission")
    if "base rate" in text and ("not generic" in text or "beats" in text) and score < 3:
        flags.append("base_rate_claim_without_material_edge")
    if ("fresh" in text or "new" in text) and zone(row) in {"failing_zone", "unresolved_chop"}:
        flags.append("fresh_claim_conflicts_with_zone_memory")
    if ("micro" in text or "confirmation" in text or "confirmed" in text) and not has_phrase(text, MICRO_TOKENS - {"micro", "confirmation", "confirmed"}):
        flags.append("vague_micro_confirmation")
    if family(row) == "critical_level_reaction" and timeframe(row) == "mixed":
        flags.append("critical_level_mixed")
    if family(row) == "momentum_continuation" and lower(row.get("side")) == "sell":
        flags.append("momentum_continuation_sell")
    if lower(row.get("skip_reason")) and is_trade_decision(row):
        flags.append("trade_with_skip_reason")
    return flags


def reasoning_risk_score(row: dict[str, Any]) -> int:
    weights = {
        "empty_or_generic_catalyst": 3,
        "location_as_edge": 2,
        "mixed_or_countertrend_weak_catalyst": 3,
        "sell_weakness_not_material": 4,
        "large_lot_reasoning_gap": 5,
        "contradiction_admitted": 1,
        "contradiction_not_materially_resolved": 3,
        "weak_trade_language_used_as_permission": 2,
        "base_rate_claim_without_material_edge": 2,
        "fresh_claim_conflicts_with_zone_memory": 3,
        "vague_micro_confirmation": 1,
        "critical_level_mixed": 2,
        "momentum_continuation_sell": 2,
        "trade_with_skip_reason": 5,
    }
    return sum(weights.get(flag, 1) for flag in reasoning_flags(row))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    closed = [r for r in rows if is_closed(r)]
    placed = [r for r in rows if is_placed(r)]
    wins = [r for r in closed if (pips(r) or 0) > 0]
    losses = [r for r in closed if (pips(r) or 0) < 0]
    pvals = [pips(r) for r in closed if pips(r) is not None]
    pnl_vals = [pnl(r) for r in closed if pnl(r) is not None]
    return {
        "calls": len(rows),
        "placed": len(placed),
        "placement_rate": len(placed) / len(rows) if rows else None,
        "closed": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(closed) if closed else None,
        "net_pips": sum(pvals) if pvals else 0.0,
        "avg_pips": statistics.fmean(pvals) if pvals else None,
        "net_pnl": sum(pnl_vals) if pnl_vals else 0.0,
    }


def stats_for_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    s = summarize(rows)
    closed = [r for r in rows if is_closed(r)]
    losses = [r for r in closed if (pips(r) or 0) < 0]
    wins = [r for r in closed if (pips(r) or 0) > 0]
    return {
        **s,
        "avg_winner_pips": statistics.fmean([pips(r) for r in wins if pips(r) is not None]) if wins else None,
        "avg_loser_pips": statistics.fmean([pips(r) for r in losses if pips(r) is not None]) if losses else None,
        "avg_lots": statistics.fmean([to_float(r.get("lots")) or 0.0 for r in closed]) if closed else None,
        "avg_reasoning_risk": statistics.fmean([reasoning_risk_score(r) for r in rows]) if rows else None,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(columns or [])
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def money(value: float | None) -> str:
    return "n/a" if value is None else f"${value:,.2f}"


def num(value: float | int | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def dossier_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        score, label, source = catalyst_score(row)
        flags = reasoning_flags(row)
        out.append({
            "created_utc": row.get("created_utc"),
            "created_local": local_iso(row.get("created_utc")),
            "profile": row.get("profile"),
            "prompt_version": row.get("prompt_version"),
            "suggestion_id": row.get("suggestion_id"),
            "trade_id": row.get("trade_id"),
            "action": row.get("action"),
            "decision": row.get("decision"),
            "side": lower(row.get("side")),
            "lots": to_float(row.get("lots")) or "",
            "lot_bucket": lots_bucket(row),
            "pips": pips(row) if pips(row) is not None else "",
            "pnl": pnl(row) if pnl(row) is not None else "",
            "win_loss": row.get("win_loss"),
            "trigger_family": family(row),
            "trigger_fit": trigger_fit(row),
            "timeframe_alignment": timeframe(row),
            "zone_memory_read": zone(row),
            "session": session_label(row),
            "h1_regime": h1_regime(row),
            "planned_rr_estimate": to_float(row.get("planned_rr_estimate")) if to_float(row.get("planned_rr_estimate")) is not None else "",
            "catalyst_score": score,
            "catalyst_label": label,
            "catalyst_source": source,
            "green_match_count": green_match_count(row),
            "weakness_signals": ";".join(weakness_signals(row)),
            "reasoning_risk_score": reasoning_risk_score(row),
            "reasoning_flags": ";".join(flags),
            "named_catalyst": short(row.get("named_catalyst"), 300),
            "trade_thesis": short(row.get("trade_thesis"), 380),
            "side_bias_check": short(row.get("side_bias_check"), 300),
            "why_trade_despite_weakness": short(row.get("why_trade_despite_weakness"), 260),
            "rationale": short(row.get("rationale"), 520),
            "analysis_text": short(analysis_text(row), 520),
        })
    return out


def matrix_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    closed = [r for r in rows if is_closed(r)]
    all_flags = sorted({flag for row in closed for flag in reasoning_flags(row)})
    out = []
    for flag in all_flags:
        subset = [r for r in closed if flag in reasoning_flags(r)]
        complement = [r for r in closed if flag not in reasoning_flags(r)]
        s = stats_for_group(subset)
        c = stats_for_group(complement)
        out.append({
            "reasoning_flag": flag,
            "closed": s["closed"],
            "wins": s["wins"],
            "losses": s["losses"],
            "win_rate": s["win_rate"],
            "net_pips": s["net_pips"],
            "net_pnl": s["net_pnl"],
            "avg_pips": s["avg_pips"],
            "avg_winner_pips": s["avg_winner_pips"],
            "avg_loser_pips": s["avg_loser_pips"],
            "avg_lots": s["avg_lots"],
            "complement_net_pips": c["net_pips"],
            "complement_win_rate": c["win_rate"],
        })
    return sorted(out, key=lambda r: (float(r["net_pips"] or 0), -int(r["closed"] or 0)))


def phrase_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    phrase_groups = {
        "despite/but/however": {"despite", "but", "however", "although"},
        "mixed/contradicts": {"mixed", "contradict", "contradicts"},
        "fresh": {"fresh"},
        "micro_confirmed": {"micro", "confirmed", "confirmation", "micro-confirmed"},
        "not_generic/base_rate": {"not generic", "base rate", "beats the buy-side"},
        "quick/small/disciplined": {"quick", "small", "disciplined", "discipline"},
        "mean_reversion": {"mean-reversion", "mean reversion"},
        "compressed_range": {"compressed", "compression", "range"},
        "jpy_weakness": {"jpy-weak", "yen weakness", "jpy weakness"},
        "policy_macro": {"policy", "macro", "mof", "boj", "intervention"},
        "support_resistance": {"support", "resistance", "reclaim", "reject"},
    }
    closed = [r for r in rows if is_closed(r)]
    out = []
    for label, phrases in phrase_groups.items():
        subset = [r for r in closed if has_phrase(lower(core_reasoning_text(r)), phrases)]
        s = stats_for_group(subset)
        out.append({"phrase_group": label, **s})
    return sorted(out, key=lambda r: float(r["net_pips"] or 0))


def contradiction_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row for row in dossier_rows(rows)
        if "contradiction_admitted" in row["reasoning_flags"] or "weak_trade_language_used_as_permission" in row["reasoning_flags"]
    ]


def catalyst_audit_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        score, label, source = catalyst_score(row)
        if not is_trade_decision(row) and not row.get("named_catalyst") and not row.get("trade_thesis"):
            continue
        out.append({
            "created_utc": row.get("created_utc"),
            "profile": row.get("profile"),
            "prompt_version": row.get("prompt_version"),
            "decision": row.get("decision"),
            "action": row.get("action"),
            "side": lower(row.get("side")),
            "pips": pips(row) if pips(row) is not None else "",
            "pnl": pnl(row) if pnl(row) is not None else "",
            "catalyst_score": score,
            "catalyst_label": label,
            "catalyst_source": source,
            "reasoning_flags": ";".join(reasoning_flags(row)),
            "named_catalyst": short(row.get("named_catalyst"), 360),
            "trade_thesis": short(row.get("trade_thesis"), 420),
            "rationale": short(row.get("rationale"), 520),
        })
    return sorted(out, key=lambda r: (str(r["prompt_version"]), int(r["catalyst_score"]), float(r["pips"] or 0)))


def flag_impact_counterfactuals(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    closed = [r for r in rows if is_closed(r)]
    flags = sorted({flag for row in closed for flag in reasoning_flags(row)})
    out = []
    for flag in flags:
        blocked = [r for r in closed if flag in reasoning_flags(r)]
        blocked_winners = [r for r in blocked if (pips(r) or 0) > 0]
        blocked_losers = [r for r in blocked if (pips(r) or 0) < 0]
        saved_loss_pips = abs(sum(pips(r) or 0 for r in blocked_losers))
        missed_winner_pips = sum(pips(r) or 0 for r in blocked_winners)
        saved_loss_usd = abs(sum(pnl(r) or 0 for r in blocked_losers))
        missed_winner_usd = sum(pnl(r) or 0 for r in blocked_winners)
        out.append({
            "candidate_reasoning_rule": f"skip_or_cap_if_{flag}",
            "blocked_closed": len(blocked),
            "blocked_winners": len(blocked_winners),
            "blocked_losers": len(blocked_losers),
            "saved_loss_pips": saved_loss_pips,
            "missed_winner_pips": missed_winner_pips,
            "net_delta_pips_if_skip": saved_loss_pips - missed_winner_pips,
            "saved_loss_usd": saved_loss_usd,
            "missed_winner_usd": missed_winner_usd,
            "net_delta_usd_if_skip": saved_loss_usd - missed_winner_usd,
        })
    return sorted(out, key=lambda r: float(r["net_delta_pips_if_skip"]), reverse=True)


def make_report(
    out_dir: Path,
    rows: list[dict[str, Any]],
    phase3_rows: list[dict[str, Any]],
    sources: list[str],
    matrix: list[dict[str, Any]],
    phrases: list[dict[str, Any]],
    counterfactuals: list[dict[str, Any]],
) -> str:
    all_summary = summarize(rows)
    p3_summary = summarize(phase3_rows)
    p3_closed = [r for r in phase3_rows if is_closed(r)]
    p3_losers = [r for r in p3_closed if (pips(r) or 0) < 0]
    flag_counts = Counter(flag for row in p3_losers for flag in reasoning_flags(row))
    top_flags = flag_counts.most_common(10)
    risk_sorted = sorted(p3_losers, key=lambda r: (-reasoning_risk_score(r), pnl(r) or 0))[:10]

    lines = [
        "# Autonomous Fillmore Reasoning Deep Dive",
        "",
        "This pass audits Fillmore's written reasoning, not just trade outcomes. It looks for cases where the model's own words admitted weakness, contradiction, generic catalysts, or size risk and then traded anyway.",
        "",
        "## Executive Read",
        "",
        f"- Autonomous reasoning rows loaded: {all_summary['calls']} calls; {all_summary['placed']} placed; {all_summary['closed']} closed; net {num(all_summary['net_pips'])}p / {money(all_summary['net_pnl'])}.",
        f"- Phase 3 reasoning universe: {p3_summary['calls']} calls; {p3_summary['placed']} placed; {p3_summary['closed']} closed; WR {pct(p3_summary['win_rate'])}; net {num(p3_summary['net_pips'])}p / {money(p3_summary['net_pnl'])}.",
        "- The dominant reasoning problem is not ignorance of risk. Fillmore often *names* the weakness, then treats the act of naming it as permission to trade.",
        "- The second problem is location-as-edge: support/resistance/reclaim/reject/session-high language often substitutes for an actual reason the setup should beat the losing base rate.",
        "- The third problem is contradiction laundering: words like 'despite', 'mixed', 'JPY-weak backdrop', 'quick', and 'disciplined' appear as caveats, but the decision still becomes a trade.",
        "",
        "## Top Phase 3 Loser Reasoning Flags",
        "",
    ]
    for flag, count in top_flags:
        subset = [r for r in p3_losers if flag in reasoning_flags(r)]
        lines.append(f"- **{flag}**: {count} losers, {num(sum(pips(r) or 0 for r in subset))}p, {money(sum(pnl(r) or 0 for r in subset))}.")

    lines.extend([
        "",
        "## Reasoning Pattern Impact",
        "",
        "| Reasoning flag | Closed | WR | Net pips | Avg loser | Avg lots |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in matrix[:12]:
        lines.append(
            f"| {row['reasoning_flag']} | {row['closed']} | {pct(row['win_rate'])} | "
            f"{num(row['net_pips'])} | {num(row['avg_loser_pips'])} | {num(row['avg_lots'], 2)} |"
        )

    lines.extend([
        "",
        "## Phrase-Level Evidence",
        "",
        "| Phrase group | Closed | WR | Net pips | Net USD |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in phrases:
        lines.append(f"| {row['phrase_group']} | {row['closed']} | {pct(row['win_rate'])} | {num(row['net_pips'])} | {money(row['net_pnl'])} |")

    lines.extend([
        "",
        "## Best Reasoning Counterfactuals",
        "",
        "These are historical skip counterfactuals for reasoning flags. Use them to rank prompt/schema fixes; do not treat them as guaranteed forward performance.",
        "",
        "| Candidate rule | Blocked | Winners blocked | Losers blocked | Net pips if skip | Net USD if skip |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in counterfactuals[:10]:
        lines.append(
            f"| {row['candidate_reasoning_rule']} | {row['blocked_closed']} | {row['blocked_winners']} | "
            f"{row['blocked_losers']} | {num(row['net_delta_pips_if_skip'])} | {money(row['net_delta_usd_if_skip'])} |"
        )

    lines.extend([
        "",
        "## Row-Level Examples: High-Risk Phase 3 Loser Reasoning",
        "",
    ])
    for row in risk_sorted[:8]:
        flags = ", ".join(reasoning_flags(row))
        lines.extend([
            f"### {local_iso(row.get('created_utc'))} | {lower(row.get('side')).upper()} {to_float(row.get('lots')) or 0:g} lots | {num(pips(row))}p | {money(pnl(row))}",
            f"- Family/session: `{family(row) or 'unknown'}` / `{session_label(row) or 'unknown'}` / TF `{timeframe(row) or 'unknown'}`",
            f"- Reasoning risk {reasoning_risk_score(row)}; flags: {flags or 'none'}",
            f"- Catalyst: {short(row.get('named_catalyst') or row.get('trade_thesis') or row.get('rationale'), 360)}",
            "",
        ])

    lines.extend([
        "## Root Reasoning Failures",
        "",
        "1. **Caveat-to-permission failure.** The model sees adverse context, writes 'despite/but/mixed/quick/discipline', and then treats the caveat as a risk plan. That is not edge; it is usually a reason to skip or cap.",
        "2. **Where-vs-why confusion.** A named level answers where the trade is, not why it should work. The losing text often names support/resistance/session high but does not name a material reason the base rate should improve.",
        "3. **Generic micro-confirmation.** 'Micro-confirmed' and 'fresh' appear often, but the text does not always state the actual event that changed the distribution.",
        "4. **Sell-side self-justification.** The phrase 'not a generic short' is being used as a substitute for a material catalyst. It should not pass the sell+weakness rule.",
        "5. **Size not coupled tightly enough to reasoning quality.** Large or normal size should require clean reasoning, not just a non-empty rationale.",
        "",
        "## Recommended Reasoning Fixes",
        "",
        "- Add a required `adverse_context` field: if it contains contradiction language and `material_catalyst_score < 3`, decision must be skip or max 1 lot.",
        "- Split `named_catalyst` into `location` and `edge_reason`. Reject/cap when `edge_reason` is only support/resistance/reclaim/reject/session-high language.",
        "- Add `micro_confirmation_event`: require the concrete event, not the phrase 'micro-confirmed'.",
        "- Add a `caveat_resolution` field: if the model writes despite/but/mixed/JPY-weak/quick/discipline, it must state what objectively neutralizes that caveat. Empty/generic means skip/cap.",
        "- For sells, `side_bias_check` must name a material reason. 'Not generic' and 'bounded range' should cap to 1 lot.",
        "",
        "## Output Files",
        "",
        "- `reasoning_trade_dossier.csv`",
        "- `phase3_reasoning_loser_dossier.csv`",
        "- `reasoning_vs_outcome_matrix.csv`",
        "- `reasoning_phrase_impact.csv`",
        "- `self_contradiction_audit.csv`",
        "- `catalyst_language_audit.csv`",
        "- `reasoning_counterfactuals.csv`",
        "- `PROMPT_PATCH_RECOMMENDATIONS.md`",
        "",
        "## Sources",
        "",
    ])
    lines.extend(f"- `{src}`" for src in sources)
    text = "\n".join(lines) + "\n"
    (out_dir / "REPORT.md").write_text(text)
    return text


def make_patch_recommendations(out_dir: Path, counterfactuals: list[dict[str, Any]]) -> None:
    lines = [
        "# Prompt / Code Recommendations From Reasoning Deep Dive",
        "",
        "These recommendations target Fillmore's reasoning failure mode: it identifies caveats but still converts them into trades.",
        "",
        "## Highest Confidence",
        "",
        "1. **Caveat is not edge.** Add a binding rule: if rationale/thesis contains despite, but, however, mixed, contradicts, JPY-weak, quick, small, tactical, probe, or disciplined, then the answer is skip unless `caveat_resolution` names a material catalyst. If allowed, max 1 lot.",
        "2. **Split location from edge.** Replace `named_catalyst` with `setup_location` and `edge_reason`. Support/resistance/reclaim/reject/session-high text can fill location, but cannot satisfy edge_reason.",
        "3. **Require concrete micro event.** `micro_confirmation_event` must name the event: sweep/retest/close/acceptance/EMA reclaim/higher-low/lower-high. The word 'micro-confirmed' alone is generic.",
        "4. **Sell-side proof standard.** `side_bias_check` must include a material catalyst token or max 1 lot. 'Not generic short' is not a catalyst.",
        "",
        "## Candidate Counterfactuals",
        "",
        "| Rule | Net pips if skipped | Net USD if skipped | Blocked closed |",
        "|---|---:|---:|---:|",
    ]
    for row in counterfactuals[:12]:
        lines.append(
            f"| {row['candidate_reasoning_rule']} | {num(row['net_delta_pips_if_skip'])} | "
            f"{money(row['net_delta_usd_if_skip'])} | {row['blocked_closed']} |"
        )
    lines.extend([
        "",
        "## Suggested Output Contract Additions",
        "",
        "```json",
        '{',
        '  "setup_location": "<level/zone/location only>",',
        '  "edge_reason": "<why this should beat the base rate; cannot be only a level>",',
        '  "adverse_context": "<the strongest reason not to trade, or null>",',
        '  "caveat_resolution": "<specific material reason the adverse context is neutralized, else null>",',
        '  "micro_confirmation_event": "<actual event observed, not just micro-confirmed>",',
        '  "reasoning_quality_gate": "clean" | "capped" | "skip"',
        '}',
        "```",
        "",
        "## Server Backstops",
        "",
        "- If `adverse_context` is non-empty and `caveat_resolution` is empty/generic: skip or cap to 1.",
        "- If `edge_reason` is location-only: skip/cap, even when `setup_location` is specific.",
        "- If `micro_confirmation_event` lacks an event verb: cap to 1.",
        "- If side=sell and `edge_reason` is not material: cap to 1.",
    ])
    (out_dir / "PROMPT_PATCH_RECOMMENDATIONS.md").write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> Path:
    stamp = args.date or datetime.now(LOCAL_TZ).strftime("%Y%m%d")
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "research_out" / f"autonomous_fillmore_reasoning_deep_dive_{stamp}"
    live_jsons = [Path(p) for p in args.live_json] if args.live_json else default_live_json_paths()
    rows, sources = load_all_rows(Path(args.evidence), live_jsons)
    auto_rows = [r for r in rows if is_autonomous(r)]
    phase3_rows = [r for r in auto_rows if phase3(r)]

    out_dir.mkdir(parents=True, exist_ok=True)
    dossier = dossier_rows(auto_rows)
    phase3_losers = [r for r in phase3_rows if is_closed(r) and (pips(r) or 0) < 0]
    phase3_loser_dossier = dossier_rows(phase3_losers)
    matrix = matrix_rows(phase3_rows)
    phrases = phrase_rows(phase3_rows)
    contradictions = contradiction_rows(phase3_rows)
    catalyst_audit = catalyst_audit_rows(phase3_rows)
    counterfactuals = flag_impact_counterfactuals(phase3_rows)

    write_csv(out_dir / "reasoning_trade_dossier.csv", dossier)
    write_csv(out_dir / "phase3_reasoning_loser_dossier.csv", phase3_loser_dossier)
    write_csv(out_dir / "reasoning_vs_outcome_matrix.csv", matrix)
    write_csv(out_dir / "reasoning_phrase_impact.csv", phrases)
    write_csv(out_dir / "self_contradiction_audit.csv", contradictions)
    write_csv(out_dir / "catalyst_language_audit.csv", catalyst_audit)
    write_csv(out_dir / "reasoning_counterfactuals.csv", counterfactuals)
    make_patch_recommendations(out_dir, counterfactuals)
    report = make_report(out_dir, auto_rows, phase3_rows, sources, matrix, phrases, counterfactuals)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(out_dir),
        "sources": sources,
        "autonomous_rows": len(auto_rows),
        "phase3_rows": len(phase3_rows),
        "phase3_summary": summarize(phase3_rows),
        "outputs": [
            "REPORT.md",
            "PROMPT_PATCH_RECOMMENDATIONS.md",
            "reasoning_trade_dossier.csv",
            "phase3_reasoning_loser_dossier.csv",
            "reasoning_vs_outcome_matrix.csv",
            "reasoning_phrase_impact.csv",
            "self_contradiction_audit.csv",
            "catalyst_language_audit.csv",
            "reasoning_counterfactuals.csv",
        ],
    }
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    if not args.quiet:
        print(report)
        print(f"\nWrote {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", default=str(DEFAULT_EVIDENCE))
    parser.add_argument("--live-json", action="append", default=[])
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--date", default="")
    parser.add_argument("--quiet", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
