#!/usr/bin/env python3
"""Export full Autonomous Fillmore data for kumatora2 + newera8 over UTC Apr 26-27 2026.

Writes raw JSON, normalized CSVs, and joined suggestion+ledger rows into
research_out/fillmore_apr26_27_investigation/.

Usage:
  python3 scripts/export_fillmore_apr26_27_investigation.py [BASE_URL]
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import urllib.parse
from datetime import datetime, timezone
from typing import Any

DEFAULT_BASE = "https://web-production-0de6a.up.railway.app"
PROFILES = [
    ("newera8", "/data/profiles/newera8.json"),
    ("kumatora2", "/data/profiles/kumatora2.json"),
]
OUT_DIR = "research_out/fillmore_apr26_27_investigation"

UTC_START = datetime(2026, 4, 26, 0, 0, 0, tzinfo=timezone.utc)
UTC_END = datetime(2026, 4, 28, 0, 0, 0, tzinfo=timezone.utc)  # exclusive


def http_get(url: str, timeout: float = 180.0) -> Any:
    proc = subprocess.run(
        ["curl", "-sS", "-m", str(int(timeout)), url],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"curl failed {proc.returncode}: {proc.stderr[:500]}")
    return json.loads(proc.stdout)


def parse_iso(ts: Any) -> datetime | None:
    if not ts:
        return None
    t = str(ts).strip()
    if not t:
        return None
    try:
        if t.endswith("Z"):
            t = t[:-1] + "+00:00"
        return datetime.fromisoformat(t.replace("Z", "+00:00"))
    except ValueError:
        return None


def in_window_utc(ts: Any) -> bool:
    dt = parse_iso(ts)
    if dt is None:
        return False
    dt_utc = dt.astimezone(timezone.utc)
    return UTC_START <= dt_utc < UTC_END


def is_autonomous(s: dict[str, Any]) -> bool:
    po = s.get("placed_order") or {}
    if isinstance(po, dict) and po.get("autonomous") is True:
        return True
    if str(s.get("rationale") or "").startswith("AUTONOMOUS"):
        return True
    if str(s.get("entry_type") or "").lower().startswith("ai_autonomous"):
        return True
    if str(s.get("trade_id") or "").lower().startswith("ai_autonomous:"):
        return True
    return False


def fetch_all_suggestions(base: str, profile: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    offset = 0
    limit = 500
    total: int | None = None
    while True:
        q = urllib.parse.urlencode({"limit": limit, "offset": offset})
        url = f"{base}/api/data/{urllib.parse.quote(profile)}/ai-suggestions/history?{q}"
        payload = http_get(url)
        if not isinstance(payload, dict):
            break
        if total is None:
            total = int(payload.get("total") or 0)
        items = payload.get("items") or []
        if not items:
            break
        for it in items:
            it["_profile"] = profile
            out.append(it)
        offset += len(items)
        if offset >= (total or 0):
            break
        # Early termination: if all items in this page are older than UTC_START, stop.
        try:
            last_ts = parse_iso(items[-1].get("created_utc"))
            if last_ts and last_ts.astimezone(timezone.utc) < UTC_START:
                break
        except Exception:  # pragma: no cover
            pass
    return out


def fetch_advanced_trades(base: str, profile: str, profile_path: str, days_back: int = 60) -> list[dict[str, Any]]:
    enc = urllib.parse.quote(profile_path, safe="")
    url = (
        f"{base}/api/data/{urllib.parse.quote(profile)}/advanced-analytics?"
        f"profile_path={enc}&days_back={days_back}"
    )
    payload = http_get(url, timeout=240.0)
    if not isinstance(payload, dict):
        return []
    rows = payload.get("trades") or []
    for r in rows:
        r["_profile"] = profile
    return rows


def fetch_autonomous_stats(base: str, profile: str) -> dict[str, Any]:
    url = f"{base}/api/data/{urllib.parse.quote(profile)}/autonomous/stats"
    return http_get(url, timeout=90.0)


def fetch_reasoning(base: str, profile: str) -> dict[str, Any]:
    url = f"{base}/api/data/{urllib.parse.quote(profile)}/autonomous/reasoning"
    return http_get(url, timeout=90.0)


def to_jsonable(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return json.dumps(v, ensure_ascii=False, sort_keys=True, default=str)


def flatten_suggestion(s: dict[str, Any]) -> dict[str, Any]:
    placed = s.get("placed_order") or {}
    feats = s.get("features") or {}
    custom_exit = (placed or {}).get("custom_exit_plan") if isinstance(placed, dict) else None
    market = s.get("market_snapshot") or {}
    created = parse_iso(s.get("created_utc"))
    closed = parse_iso(s.get("closed_at"))
    filled = parse_iso(s.get("filled_at"))

    def get(d: dict[str, Any], *path: str, default: Any = None) -> Any:
        cur: Any = d
        for k in path:
            if not isinstance(cur, dict):
                return default
            cur = cur.get(k)
            if cur is None:
                return default
        return cur

    return {
        "profile": s.get("_profile"),
        "created_utc": s.get("created_utc"),
        "filled_at": s.get("filled_at"),
        "closed_at": s.get("closed_at"),
        "minutes_open": (
            round((closed - filled).total_seconds() / 60.0, 2)
            if filled and closed
            else (round((closed - created).total_seconds() / 60.0, 2) if created and closed else None)
        ),
        "suggestion_id": s.get("suggestion_id"),
        "trade_id": s.get("trade_id"),
        "oanda_order_id": s.get("oanda_order_id"),
        "decision": placed.get("decision") if isinstance(placed, dict) else None,
        "action": s.get("action"),
        "outcome_status": s.get("outcome_status"),
        "side": s.get("side"),
        "lots": s.get("lots"),
        "requested_price": s.get("requested_price"),
        "limit_price": s.get("limit_price"),
        "fill_price": s.get("fill_price"),
        "exit_price": s.get("exit_price"),
        "sl": s.get("sl"),
        "tp": s.get("tp"),
        "exit_strategy": s.get("exit_strategy"),
        "trigger_family": s.get("trigger_family") or get(placed, "trigger_family"),
        "trigger_reason": s.get("trigger_reason") or get(placed, "trigger_reason"),
        "thesis_fingerprint": s.get("thesis_fingerprint") or get(placed, "thesis_fingerprint"),
        "zone_memory_read": s.get("zone_memory_read") or get(placed, "zone_memory_read"),
        "repeat_trade_case": s.get("repeat_trade_case") or get(placed, "repeat_trade_case"),
        "timeframe_alignment": s.get("timeframe_alignment") or get(placed, "timeframe_alignment"),
        "trigger_fit": s.get("trigger_fit") or get(placed, "trigger_fit"),
        "conviction_rung": s.get("conviction_rung") or get(placed, "conviction_rung"),
        "planned_rr": s.get("planned_rr_estimate") or get(placed, "planned_rr_estimate"),
        "confidence": s.get("confidence"),
        "win_loss": s.get("win_loss"),
        "pips": s.get("pips"),
        "pnl_usd": s.get("pnl"),
        "max_adverse_pips": s.get("max_adverse_pips"),
        "max_favorable_pips": s.get("max_favorable_pips"),
        "skip_reason": s.get("skip_reason") or get(placed, "skip_reason"),
        "trade_thesis": s.get("trade_thesis") or get(placed, "trade_thesis"),
        "why_not_stop": s.get("why_not_stop") or get(placed, "why_not_stop"),
        "whats_different": s.get("whats_different") or get(placed, "whats_different"),
        "low_rr_edge": s.get("low_rr_edge") or get(placed, "low_rr_edge"),
        "countertrend_edge": s.get("countertrend_edge") or get(placed, "countertrend_edge"),
        "why_trade_despite_weakness": s.get("why_trade_despite_weakness") or get(placed, "why_trade_despite_weakness"),
        "exit_plan": s.get("exit_plan") or get(placed, "exit_plan"),
        "rationale": s.get("rationale"),
        "session": get(market, "session", "active_sessions"),
        "overlap": get(market, "session", "overlap"),
        "spread_pips": get(market, "spread_pips"),
        "vol_label": get(market, "volatility", "label"),
        "vol_ratio": get(market, "volatility", "ratio"),
        "h1_regime": get(market, "technicals", "H1", "regime"),
        "m5_regime": get(market, "technicals", "M5", "regime"),
        "m1_regime": get(market, "technicals", "M1", "regime"),
        "h1_atr": get(market, "technicals", "H1", "atr_pips"),
        "m5_atr": get(market, "technicals", "M5", "atr_pips"),
        "m1_atr": get(market, "technicals", "M1", "atr_pips"),
        "h1_rsi_zone": get(market, "technicals", "H1", "rsi_zone"),
        "m5_rsi_zone": get(market, "technicals", "M5", "rsi_zone"),
        "m1_rsi_zone": get(market, "technicals", "M1", "rsi_zone"),
        "live_mid": get(market, "live_price_mid"),
        "nearest_support": get(market, "order_book", "nearest_support"),
        "nearest_support_dist_p": get(market, "order_book", "nearest_support_distance_pips"),
        "nearest_resistance": get(market, "order_book", "nearest_resistance"),
        "nearest_resistance_dist_p": get(market, "order_book", "nearest_resistance_distance_pips"),
        "macro_combined_bias": get(market, "macro_bias", "combined_bias"),
        "macro_dxy_dir": get(market, "macro_bias", "dxy_direction"),
        "macro_oil_dir": get(market, "macro_bias", "oil_direction"),
        "is_autonomous": is_autonomous(s),
        "custom_exit_plan_json": json.dumps(custom_exit, sort_keys=True) if custom_exit else "",
        "features_json": json.dumps(feats, sort_keys=True) if feats else "",
        "placed_order_json": json.dumps(placed, sort_keys=True, default=str) if placed else "",
        "market_snapshot_json": json.dumps(market, sort_keys=True, default=str) if market else "",
    }


def flatten_trade(t: dict[str, Any]) -> dict[str, Any]:
    return {
        "profile": t.get("_profile"),
        "trade_id": t.get("trade_id"),
        "ticket_id": t.get("ticket_id"),
        "side": t.get("side"),
        "entry_time_utc": t.get("entry_time_utc") or t.get("open_time_utc") or t.get("entry_time"),
        "exit_time_utc": t.get("exit_time_utc") or t.get("close_time_utc") or t.get("exit_time"),
        "duration_minutes": t.get("duration_minutes"),
        "entry_price": t.get("entry_price"),
        "exit_price": t.get("exit_price"),
        "lots": t.get("lots") or t.get("size"),
        "pips": t.get("pips"),
        "r_multiple": t.get("r_multiple"),
        "risk_pips": t.get("risk_pips"),
        "profit": t.get("profit") or t.get("pnl") or t.get("net_pnl"),
        "max_adverse_pips": t.get("max_adverse_pips"),
        "max_favorable_pips": t.get("max_favorable_pips"),
        "post_sl_recovery_pips": t.get("post_sl_recovery_pips"),
        "exit_reason": t.get("exit_reason"),
        "entry_type": t.get("entry_type"),
        "preset_name": t.get("preset_name"),
        "raw_json": json.dumps(t, default=str, sort_keys=True),
    }


def write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in keys})


def main() -> int:
    base = (sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASE).rstrip("/")
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"BASE={base} OUT_DIR={OUT_DIR}")

    summary: dict[str, Any] = {
        "base": base,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "utc_window": [UTC_START.isoformat(), UTC_END.isoformat()],
        "profiles": {},
    }

    all_combined_sug: list[dict[str, Any]] = []
    all_combined_trades: list[dict[str, Any]] = []

    for pname, ppath in PROFILES:
        print(f"\n=== {pname} ===")
        sug_all = fetch_all_suggestions(base, pname)
        sug_window = [s for s in sug_all if in_window_utc(s.get("created_utc"))]
        sug_window_auto = [s for s in sug_window if is_autonomous(s)]
        print(f"  fetched suggestion rows total: {len(sug_all)}")
        print(f"  in UTC window {UTC_START.date()}..{UTC_END.date()}: {len(sug_window)} (autonomous {len(sug_window_auto)})")

        trades_all = fetch_advanced_trades(base, pname, ppath, days_back=120)
        trades_window: list[dict[str, Any]] = []
        for t in trades_all:
            ot = (
                t.get("entry_time_utc")
                or t.get("open_time_utc")
                or t.get("entry_time")
                or t.get("exit_time_utc")
            )
            if in_window_utc(ot):
                trades_window.append(t)
        print(f"  trade rows in UTC window: {len(trades_window)} (total fetched {len(trades_all)})")
        trades_window_auto = [
            t for t in trades_window
            if str(t.get("entry_type") or "").lower().startswith("ai_autonomous")
            or str(t.get("trade_id") or "").lower().startswith("ai_autonomous:")
        ]
        print(f"    autonomous-only: {len(trades_window_auto)}")

        try:
            stats = fetch_autonomous_stats(base, pname)
        except Exception as exc:
            stats = {"_error": str(exc)}
        try:
            reasoning = fetch_reasoning(base, pname)
        except Exception as exc:
            reasoning = {"_error": str(exc)}

        # Filter recent thesis checks / gate decisions to UTC window when timestamped
        thesis_checks = []
        if isinstance(reasoning, dict):
            tc = reasoning.get("thesis_checks") or []
            for r in tc:
                ts = (r or {}).get("created_utc") or (r or {}).get("checked_at_utc")
                if not ts or in_window_utc(ts):
                    thesis_checks.append(r)
        gate_decisions = []
        if isinstance(stats, dict):
            gd = stats.get("recent_decisions") or []
            for r in gd:
                ts = (r or {}).get("created_utc") or (r or {}).get("decided_at_utc") or (r or {}).get("ts_utc")
                if not ts or in_window_utc(ts):
                    gate_decisions.append(r)

        prefix = f"{pname}_apr26_27_utc"
        with open(os.path.join(OUT_DIR, f"{prefix}_raw_suggestions.json"), "w", encoding="utf-8") as fh:
            json.dump(sug_window_auto, fh, indent=2, default=str)
        with open(os.path.join(OUT_DIR, f"{prefix}_raw_trades.json"), "w", encoding="utf-8") as fh:
            json.dump(trades_window_auto, fh, indent=2, default=str)
        with open(os.path.join(OUT_DIR, f"{prefix}_raw_trades_all.json"), "w", encoding="utf-8") as fh:
            json.dump(trades_window, fh, indent=2, default=str)
        with open(os.path.join(OUT_DIR, f"{prefix}_recent_thesis_checks.json"), "w", encoding="utf-8") as fh:
            json.dump(thesis_checks, fh, indent=2, default=str)
        with open(os.path.join(OUT_DIR, f"{prefix}_recent_gate_decisions.json"), "w", encoding="utf-8") as fh:
            json.dump(gate_decisions, fh, indent=2, default=str)
        with open(os.path.join(OUT_DIR, f"{prefix}_autonomous_stats.json"), "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2, default=str)
        with open(os.path.join(OUT_DIR, f"{prefix}_reasoning_snapshot.json"), "w", encoding="utf-8") as fh:
            json.dump(reasoning, fh, indent=2, default=str)

        flat_sug = [flatten_suggestion(s) for s in sug_window_auto]
        flat_sug.sort(key=lambda r: str(r.get("created_utc") or ""))
        write_csv(os.path.join(OUT_DIR, f"{prefix}_suggestions_flat.csv"), flat_sug)

        flat_trades = [flatten_trade(t) for t in trades_window_auto]
        flat_trades.sort(key=lambda r: str(r.get("entry_time_utc") or ""))
        write_csv(os.path.join(OUT_DIR, f"{prefix}_trades_flat.csv"), flat_trades)

        # Joined view: suggestion -> ledger trade by trade_id
        trade_by_id: dict[str, dict[str, Any]] = {}
        for t in trades_window_auto:
            tid = str(t.get("trade_id") or "").strip()
            if tid:
                trade_by_id[tid] = t
        joined: list[dict[str, Any]] = []
        for s in flat_sug:
            tid = str(s.get("trade_id") or "").strip()
            t = trade_by_id.get(tid) if tid else None
            joined.append({
                **s,
                "ledger_exit_reason": (t or {}).get("exit_reason"),
                "ledger_duration_minutes": (t or {}).get("duration_minutes"),
                "ledger_pnl": (t or {}).get("pnl"),
                "ledger_pips": (t or {}).get("pips"),
                "ledger_mae": (t or {}).get("max_adverse_pips"),
                "ledger_mfe": (t or {}).get("max_favorable_pips"),
                "ledger_entry_time_utc": (t or {}).get("entry_time_utc") or (t or {}).get("open_time_utc") or (t or {}).get("entry_time"),
                "ledger_exit_time_utc": (t or {}).get("exit_time_utc") or (t or {}).get("close_time_utc") or (t or {}).get("exit_time"),
                "ledger_r_multiple": (t or {}).get("r_multiple"),
                "ledger_risk_pips": (t or {}).get("risk_pips"),
                "ledger_post_sl_recovery_pips": (t or {}).get("post_sl_recovery_pips"),
            })
        write_csv(os.path.join(OUT_DIR, f"{prefix}_joined.csv"), joined)

        placed = [s for s in sug_window_auto if str(s.get("action") or "").lower() == "placed"]
        skips = [s for s in sug_window_auto if str(s.get("action") or "").lower() != "placed"]
        closed_rows = [s for s in sug_window_auto if s.get("closed_at")]
        wins = sum(1 for s in closed_rows if str(s.get("win_loss") or "").lower() == "win")
        losses = sum(1 for s in closed_rows if str(s.get("win_loss") or "").lower() == "loss")
        sum_pnl = sum(float(s.get("pnl") or 0) for s in closed_rows)
        sum_pips = sum(float(s.get("pips") or 0) for s in closed_rows)
        buys = sum(1 for s in placed if str(s.get("side") or "").lower() == "buy")
        sells = sum(1 for s in placed if str(s.get("side") or "").lower() == "sell")

        summary["profiles"][pname] = {
            "total_autonomous_calls": len(sug_window_auto),
            "placed": len(placed),
            "skips": len(skips),
            "buys_placed": buys,
            "sells_placed": sells,
            "closed": len(closed_rows),
            "wins": wins,
            "losses": losses,
            "win_rate_pct": round(100.0 * wins / max(len(closed_rows), 1), 2),
            "net_pnl_usd": round(sum_pnl, 2),
            "net_pips": round(sum_pips, 1),
            "trades_in_window_autonomous": len(trades_window_auto),
            "trades_in_window_total": len(trades_window),
            "thesis_checks_returned": len(thesis_checks),
            "gate_decisions_returned": len(gate_decisions),
        }

        all_combined_sug.extend(flat_sug)
        all_combined_trades.extend(flat_trades)

    all_combined_sug.sort(key=lambda r: str(r.get("created_utc") or ""))
    all_combined_trades.sort(key=lambda r: str(r.get("entry_time_utc") or ""))
    write_csv(os.path.join(OUT_DIR, "combined_suggestions_flat.csv"), all_combined_sug)
    write_csv(os.path.join(OUT_DIR, "combined_trades_flat.csv"), all_combined_trades)

    with open(os.path.join(OUT_DIR, "export_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    print("\n=== EXPORT SUMMARY ===")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
