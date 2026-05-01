#!/usr/bin/env python3
"""Round up Autonomous Fillmore evidence from the deployed API.

The exporter is intentionally broad: it keeps raw payloads, flattened CSVs,
joined suggestion/trade rows, and a manifest of local reports/commits that
explain the fixes made over time.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import urllib.parse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_BASE = "https://web-production-0de6a.up.railway.app"
OUT_DIR = Path("research_out/autonomous_fillmore_evidence_20260429")
DAYS_BACK = 3650


def http_get(url: str, timeout: int = 240) -> Any:
    proc = subprocess.run(
        ["curl", "-sS", "-m", str(timeout), url],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"curl failed {proc.returncode}: {proc.stderr[:500]}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"non-json response from {url}: {proc.stdout[:500]}") from exc


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def parse_iso(ts: Any) -> datetime | None:
    if not ts:
        return None
    text = str(ts).strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def get_nested(d: Any, *path: str, default: Any = None) -> Any:
    cur = d
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


def is_autonomous(row: dict[str, Any]) -> bool:
    placed = row.get("placed_order") if isinstance(row.get("placed_order"), dict) else {}
    return (
        placed.get("autonomous") is True
        or str(row.get("entry_type") or "").lower().startswith("ai_autonomous")
        or str(row.get("trade_id") or "").lower().startswith("ai_autonomous:")
        or str(row.get("rationale") or "").startswith("AUTONOMOUS")
    )


def fetch_all_suggestions(base: str, profile: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    offset = 0
    limit = 500
    total: int | None = None
    while True:
        q = urllib.parse.urlencode({"limit": limit, "offset": offset})
        payload = http_get(f"{base}/api/data/{urllib.parse.quote(profile)}/ai-suggestions/history?{q}")
        if not isinstance(payload, dict):
            break
        if total is None:
            try:
                total = int(payload.get("total") or 0)
            except (TypeError, ValueError):
                total = None
        items = payload.get("items") or []
        if not items:
            break
        for item in items:
            if isinstance(item, dict):
                item["_profile"] = profile
                out.append(item)
        offset += len(items)
        if total is not None and offset >= total:
            break
    return out


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True, default=str)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        if not keys:
            fh.write("")
            return
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in keys})


def flatten_suggestion(s: dict[str, Any]) -> dict[str, Any]:
    placed = s.get("placed_order") if isinstance(s.get("placed_order"), dict) else {}
    market = s.get("market_snapshot") if isinstance(s.get("market_snapshot"), dict) else {}
    features = s.get("features") if isinstance(s.get("features"), dict) else {}
    created = parse_iso(s.get("created_utc"))
    filled = parse_iso(s.get("filled_at"))
    closed = parse_iso(s.get("closed_at"))
    return {
        "profile": s.get("_profile"),
        "created_utc": s.get("created_utc"),
        "filled_at": s.get("filled_at"),
        "closed_at": s.get("closed_at"),
        "minutes_open": (
            round((closed - filled).total_seconds() / 60.0, 2)
            if closed and filled
            else (round((closed - created).total_seconds() / 60.0, 2) if closed and created else None)
        ),
        "suggestion_id": s.get("suggestion_id"),
        "trade_id": s.get("trade_id"),
        "oanda_order_id": s.get("oanda_order_id"),
        "entry_type": s.get("entry_type"),
        "is_autonomous": is_autonomous(s),
        "action": s.get("action"),
        "decision": placed.get("decision"),
        "side": s.get("side"),
        "lots": s.get("lots"),
        "requested_price": s.get("requested_price"),
        "limit_price": s.get("limit_price"),
        "fill_price": s.get("fill_price"),
        "exit_price": s.get("exit_price"),
        "sl": s.get("sl"),
        "tp": s.get("tp"),
        "exit_strategy": s.get("exit_strategy"),
        "outcome_status": s.get("outcome_status"),
        "win_loss": s.get("win_loss"),
        "pips": s.get("pips"),
        "pnl_usd": s.get("pnl"),
        "max_adverse_pips": s.get("max_adverse_pips"),
        "max_favorable_pips": s.get("max_favorable_pips"),
        "trigger_family": s.get("trigger_family") or placed.get("trigger_family"),
        "trigger_reason": s.get("trigger_reason") or placed.get("trigger_reason"),
        "thesis_fingerprint": s.get("thesis_fingerprint") or placed.get("thesis_fingerprint"),
        "zone_memory_read": s.get("zone_memory_read") or placed.get("zone_memory_read"),
        "repeat_trade_case": s.get("repeat_trade_case") or placed.get("repeat_trade_case"),
        "timeframe_alignment": s.get("timeframe_alignment") or placed.get("timeframe_alignment"),
        "trigger_fit": s.get("trigger_fit") or placed.get("trigger_fit"),
        "conviction_rung": s.get("conviction_rung") or placed.get("conviction_rung"),
        "planned_rr_estimate": s.get("planned_rr_estimate") or placed.get("planned_rr_estimate"),
        "skip_reason": s.get("skip_reason") or placed.get("skip_reason"),
        "trade_thesis": s.get("trade_thesis") or placed.get("trade_thesis"),
        "why_not_stop": s.get("why_not_stop") or placed.get("why_not_stop"),
        "whats_different": s.get("whats_different") or placed.get("whats_different"),
        "low_rr_edge": s.get("low_rr_edge") or placed.get("low_rr_edge"),
        "countertrend_edge": s.get("countertrend_edge") or placed.get("countertrend_edge"),
        "why_trade_despite_weakness": s.get("why_trade_despite_weakness") or placed.get("why_trade_despite_weakness"),
        "named_catalyst": s.get("named_catalyst") or placed.get("named_catalyst"),
        "side_bias_check": s.get("side_bias_check") or placed.get("side_bias_check"),
        "setup_location": s.get("setup_location") or placed.get("setup_location") or features.get("setup_location"),
        "edge_reason": s.get("edge_reason") or placed.get("edge_reason") or features.get("edge_reason"),
        "adverse_context": s.get("adverse_context") or placed.get("adverse_context") or features.get("adverse_context"),
        "caveat_resolution": s.get("caveat_resolution") or placed.get("caveat_resolution") or features.get("caveat_resolution"),
        "micro_confirmation_event": (
            s.get("micro_confirmation_event")
            or placed.get("micro_confirmation_event")
            or features.get("micro_confirmation_event")
        ),
        "reasoning_quality_gate": (
            s.get("reasoning_quality_gate")
            or placed.get("reasoning_quality_gate")
            or features.get("reasoning_quality_gate")
        ),
        "phase5_reasoning_flags": json.dumps(features.get("phase5_reasoning_flags") or [], sort_keys=True, default=str),
        "phase5_material_resolution_score": features.get("phase5_material_resolution_score"),
        "exit_plan": s.get("exit_plan") or placed.get("exit_plan"),
        "rationale": s.get("rationale"),
        "session": get_nested(market, "session", "active_sessions"),
        "overlap": get_nested(market, "session", "overlap"),
        "spread_pips": get_nested(market, "spread_pips"),
        "vol_label": get_nested(market, "volatility", "label"),
        "vol_ratio": get_nested(market, "volatility", "ratio"),
        "h1_regime": get_nested(market, "technicals", "H1", "regime"),
        "m5_regime": get_nested(market, "technicals", "M5", "regime"),
        "m1_regime": get_nested(market, "technicals", "M1", "regime"),
        "h1_atr": get_nested(market, "technicals", "H1", "atr_pips"),
        "m5_atr": get_nested(market, "technicals", "M5", "atr_pips"),
        "m1_atr": get_nested(market, "technicals", "M1", "atr_pips"),
        "nearest_support": get_nested(market, "order_book", "nearest_support"),
        "nearest_support_dist_p": get_nested(market, "order_book", "nearest_support_distance_pips"),
        "nearest_resistance": get_nested(market, "order_book", "nearest_resistance"),
        "nearest_resistance_dist_p": get_nested(market, "order_book", "nearest_resistance_distance_pips"),
        "macro_combined_bias": get_nested(market, "macro_bias", "combined_bias"),
        "macro_dxy_dir": get_nested(market, "macro_bias", "dxy_direction"),
        "macro_oil_dir": get_nested(market, "macro_bias", "oil_direction"),
        "features_json": json.dumps(features, sort_keys=True, default=str) if features else "",
        "placed_order_json": json.dumps(placed, sort_keys=True, default=str) if placed else "",
    }


def flatten_trade(t: dict[str, Any], profile: str) -> dict[str, Any]:
    return {
        "profile": profile,
        "trade_id": t.get("trade_id"),
        "ticket_id": t.get("ticket_id") or t.get("mt5_position_id") or t.get("oanda_trade_id"),
        "entry_type": t.get("entry_type"),
        "preset_name": t.get("preset_name"),
        "policy_type": t.get("policy_type"),
        "side": t.get("side"),
        "entry_time_utc": t.get("entry_time_utc") or t.get("timestamp_utc") or t.get("open_time_utc") or t.get("entry_time"),
        "exit_time_utc": t.get("exit_time_utc") or t.get("exit_timestamp_utc") or t.get("close_time_utc") or t.get("exit_time"),
        "duration_minutes": t.get("duration_minutes"),
        "entry_price": t.get("entry_price"),
        "exit_price": t.get("exit_price"),
        "stop_price": t.get("stop_price"),
        "take_profit": t.get("take_profit"),
        "lots": t.get("lots") or t.get("size"),
        "pips": t.get("pips"),
        "r_multiple": t.get("r_multiple"),
        "risk_pips": t.get("risk_pips"),
        "profit": t.get("profit") or t.get("pnl") or t.get("net_pnl") or t.get("profit_usd"),
        "commission": t.get("commission"),
        "swap": t.get("swap"),
        "max_adverse_pips": t.get("max_adverse_pips"),
        "max_favorable_pips": t.get("max_favorable_pips"),
        "post_sl_recovery_pips": t.get("post_sl_recovery_pips"),
        "exit_reason": t.get("exit_reason"),
        "raw_json": json.dumps(t, sort_keys=True, default=str),
    }


def extract_trade_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        rows = payload.get("trades") or payload.get("items") or []
        return rows if isinstance(rows, list) else []
    return payload if isinstance(payload, list) else []


def trade_sort_key(row: dict[str, Any]) -> str:
    return str(row.get("entry_time_utc") or row.get("exit_time_utc") or row.get("created_utc") or "")


def summarize_suggestions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    placed = [r for r in rows if str(r.get("action") or "").lower() == "placed"]
    closed = [r for r in placed if r.get("closed_at")]
    wins = [r for r in closed if str(r.get("win_loss") or "").lower() == "win"]
    losses = [r for r in closed if str(r.get("win_loss") or "").lower() == "loss"]
    pnl = sum(float(r.get("pnl_usd") or 0) for r in closed)
    pips = sum(float(r.get("pips") or 0) for r in closed)
    by_day: dict[str, dict[str, Any]] = defaultdict(lambda: {"calls": 0, "placed": 0, "closed": 0, "pnl_usd": 0.0, "pips": 0.0})
    by_family: dict[str, dict[str, Any]] = defaultdict(lambda: {"calls": 0, "placed": 0, "closed": 0, "pnl_usd": 0.0, "pips": 0.0})
    for r in rows:
        day = str(r.get("created_utc") or "")[:10] or "unknown"
        family = str(r.get("trigger_family") or "unknown")
        by_day[day]["calls"] += 1
        by_family[family]["calls"] += 1
        if str(r.get("action") or "").lower() == "placed":
            by_day[day]["placed"] += 1
            by_family[family]["placed"] += 1
        if r.get("closed_at"):
            by_day[day]["closed"] += 1
            by_day[day]["pnl_usd"] += float(r.get("pnl_usd") or 0)
            by_day[day]["pips"] += float(r.get("pips") or 0)
            by_family[family]["closed"] += 1
            by_family[family]["pnl_usd"] += float(r.get("pnl_usd") or 0)
            by_family[family]["pips"] += float(r.get("pips") or 0)
    return {
        "calls": len(rows),
        "placed": len(placed),
        "skips": len(rows) - len(placed),
        "closed": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "net_pnl_usd": round(pnl, 2),
        "net_pips": round(pips, 2),
        "by_day": {k: {kk: round(vv, 2) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in sorted(by_day.items())},
        "by_family": {k: {kk: round(vv, 2) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in sorted(by_family.items())},
    }


def git_log_for_fixes() -> str:
    paths = [
        "api/autonomous_fillmore.py",
        "api/paper_fillmore.py",
        "api/fillmore_learning.py",
        "api/fillmore_llm_guard.py",
        "api/suggestion_tracker.py",
        "run_loop.py",
        "tests/test_autonomous_fillmore.py",
        "tests/test_fillmore_learning_hooks.py",
        "tests/test_fillmore_llm_guard.py",
        "scripts/export_fillmore_apr26_27_investigation.py",
        "scripts/analyze_fillmore_apr26_27_investigation.py",
    ]
    proc = subprocess.run(
        ["git", "log", "--date=short", "--pretty=format:%h %ad %s", "--", *paths],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout


def local_evidence_files() -> list[str]:
    patterns = (
        "fillmore",
        "autonomous",
        "apr21",
        "apr22",
        "apr23",
        "apr24",
        "apr26",
        "apr27",
        "forensic",
        "journal",
    )
    files: list[str] = []
    for root in (Path("research_out"), Path("scripts"), Path("api"), Path("tests")):
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            low = str(path).lower()
            if any(p in low for p in patterns):
                files.append(str(path))
    return sorted(files)


def main() -> int:
    base = (sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASE).rstrip("/")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    generated_utc = datetime.now(timezone.utc).isoformat()

    profiles = http_get(f"{base}/api/profiles")
    if not isinstance(profiles, list):
        raise RuntimeError("profiles endpoint did not return a list")
    write_json(OUT_DIR / "profiles.json", profiles)

    summary: dict[str, Any] = {
        "base": base,
        "generated_utc": generated_utc,
        "days_back": DAYS_BACK,
        "profiles": {},
        "known_prior_evidence": local_evidence_files(),
    }
    combined_suggestions: list[dict[str, Any]] = []
    combined_all_trades: list[dict[str, Any]] = []
    combined_trades: list[dict[str, Any]] = []
    combined_joined: list[dict[str, Any]] = []

    for profile_row in profiles:
        if not isinstance(profile_row, dict):
            continue
        profile = str(profile_row.get("name") or "").strip()
        profile_path = str(profile_row.get("path") or "").strip()
        if not profile:
            continue
        pdir = OUT_DIR / safe_name(profile)
        pdir.mkdir(parents=True, exist_ok=True)
        enc_path = urllib.parse.quote(profile_path, safe="")
        enc_profile = urllib.parse.quote(profile)

        endpoint_payloads: dict[str, Any] = {}
        endpoints = {
            "trades_recent": f"{base}/api/data/{enc_profile}/trades?limit=20000&profile_path={enc_path}",
            "open_trades": f"{base}/api/data/{enc_profile}/open-trades?profile_path={enc_path}",
            "trade_history": f"{base}/api/data/{enc_profile}/trade-history?days_back={DAYS_BACK}&profile_path={enc_path}",
            "trade_history_detail": f"{base}/api/data/{enc_profile}/trade-history-detail?days_back={DAYS_BACK}&profile_path={enc_path}",
            "advanced_analytics": f"{base}/api/data/{enc_profile}/advanced-analytics?days_back={DAYS_BACK}&profile_path={enc_path}",
            "mt5_report": f"{base}/api/data/{enc_profile}/mt5-report?profile_path={enc_path}",
            "ai_suggestion_stats": f"{base}/api/data/{enc_profile}/ai-suggestions/stats?days_back={DAYS_BACK}",
            "autonomous_config": f"{base}/api/data/{enc_profile}/autonomous/config?profile_path={enc_path}",
            "autonomous_stats": f"{base}/api/data/{enc_profile}/autonomous/stats?profile_path={enc_path}",
            "autonomous_reasoning": f"{base}/api/data/{enc_profile}/autonomous/reasoning?profile_path={enc_path}",
        }
        for key, url in endpoints.items():
            try:
                endpoint_payloads[key] = http_get(url)
            except Exception as exc:
                endpoint_payloads[key] = {"_error": str(exc), "_url": url}
            write_json(pdir / f"{key}.json", endpoint_payloads[key])

        suggestions_raw = fetch_all_suggestions(base, profile)
        suggestions_auto = [s for s in suggestions_raw if is_autonomous(s)]
        write_json(pdir / "ai_suggestions_all_raw.json", suggestions_raw)
        write_json(pdir / "ai_suggestions_autonomous_raw.json", suggestions_auto)

        flat_suggestions = [flatten_suggestion(s) for s in suggestions_auto]
        flat_suggestions.sort(key=trade_sort_key)
        write_csv(pdir / "ai_suggestions_autonomous_flat.csv", flat_suggestions)
        combined_suggestions.extend(flat_suggestions)

        all_trade_rows = extract_trade_rows(endpoint_payloads.get("trades_recent"))
        flat_all_trades = [flatten_trade(t, profile) for t in all_trade_rows if isinstance(t, dict)]
        flat_all_trades.sort(key=trade_sort_key)
        write_csv(pdir / "all_trades_flat.csv", flat_all_trades)
        combined_all_trades.extend(flat_all_trades)

        trade_source = endpoint_payloads.get("advanced_analytics")
        trades_raw = extract_trade_rows(trade_source)
        if not trades_raw:
            trades_raw = extract_trade_rows(endpoint_payloads.get("trade_history_detail"))
        flat_trades = [flatten_trade(t, profile) for t in trades_raw if isinstance(t, dict)]
        flat_trades.sort(key=trade_sort_key)
        write_csv(pdir / "closed_trades_flat.csv", flat_trades)
        combined_trades.extend(flat_trades)

        trade_by_id = {str(t.get("trade_id") or ""): t for t in flat_trades if t.get("trade_id")}
        joined: list[dict[str, Any]] = []
        for s in flat_suggestions:
            trade = trade_by_id.get(str(s.get("trade_id") or ""))
            joined_row = dict(s)
            if trade:
                for key, value in trade.items():
                    if key in {"raw_json", "profile", "trade_id"}:
                        continue
                    joined_row[f"ledger_{key}"] = value
            joined.append(joined_row)
        write_csv(pdir / "autonomous_suggestions_joined_to_trades.csv", joined)
        combined_joined.extend(joined)

        summary["profiles"][profile] = {
            "profile_path": profile_path,
            "suggestions_total": len(suggestions_raw),
            "autonomous_suggestions_total": len(flat_suggestions),
            "all_trades_total": len(flat_all_trades),
            "closed_trades_total": len(flat_trades),
            "autonomous_summary": summarize_suggestions(flat_suggestions),
            "endpoint_errors": {
                key: payload.get("_error")
                for key, payload in endpoint_payloads.items()
                if isinstance(payload, dict) and payload.get("_error")
            },
        }

    combined_suggestions.sort(key=trade_sort_key)
    combined_all_trades.sort(key=trade_sort_key)
    combined_trades.sort(key=trade_sort_key)
    combined_joined.sort(key=trade_sort_key)
    write_csv(OUT_DIR / "combined_autonomous_suggestions_flat.csv", combined_suggestions)
    write_csv(OUT_DIR / "combined_all_trades_flat.csv", combined_all_trades)
    write_csv(OUT_DIR / "combined_closed_trades_flat.csv", combined_trades)
    write_csv(OUT_DIR / "combined_autonomous_joined.csv", combined_joined)
    write_json(OUT_DIR / "summary.json", summary)
    (OUT_DIR / "fix_commit_log.txt").write_text(git_log_for_fixes(), encoding="utf-8")

    manifest_lines = [
        "# Autonomous Fillmore Evidence Roundup",
        "",
        f"Generated UTC: {generated_utc}",
        f"Base URL: {base}",
        "",
        "## Primary Exports",
        "- `profiles.json` - deployed profiles exposed by the API.",
        "- `summary.json` - counts, endpoint errors, and autonomous performance aggregates.",
        "- `combined_autonomous_suggestions_flat.csv` - every autonomous suggestion/call found in the suggestions DB.",
        "- `combined_all_trades_flat.csv` - all trade rows returned by the profile trade ledger endpoint.",
        "- `combined_closed_trades_flat.csv` - every closed trade row returned by advanced analytics.",
        "- `combined_autonomous_joined.csv` - autonomous suggestion rows joined to ledger trades where `trade_id` matches.",
        "- `<profile>/*.json` - raw endpoint payloads kept for re-analysis.",
        "",
        "## Fix / Context Evidence",
        "- `fix_commit_log.txt` - git commits touching Autonomous Fillmore, suggestion tracking, learning, and runtime state.",
        "- Existing prior reports are listed in `summary.json` under `known_prior_evidence`.",
        "",
        "## Profiles",
    ]
    for name, psummary in summary["profiles"].items():
        auto = psummary["autonomous_summary"]
        manifest_lines.append(
            f"- `{name}`: {auto['calls']} autonomous calls, {auto['placed']} placed, "
            f"{auto['closed']} closed, net {auto['net_pnl_usd']} USD / {auto['net_pips']} pips."
        )
    (OUT_DIR / "MANIFEST.md").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    print(f"Wrote evidence roundup to {OUT_DIR}")
    print(json.dumps(summary["profiles"], indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
