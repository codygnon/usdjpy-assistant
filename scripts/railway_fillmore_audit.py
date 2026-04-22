#!/usr/bin/env python3
"""One-off Fillmore performance audit against Railway HTTP API.

Usage:
  python3 scripts/railway_fillmore_audit.py [BASE_URL]

Default BASE_URL: https://web-production-0de6a.up.railway.app
"""
from __future__ import annotations

import json
import subprocess
import sys
import urllib.parse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from statistics import median
from typing import Any

DEFAULT_BASE = "https://web-production-0de6a.up.railway.app"
PROFILES = [
    ("newera8", "/data/profiles/newera8.json"),
    ("kumatora2", "/data/profiles/kumatora2.json"),
]


def http_get(url: str, timeout: float = 120.0) -> dict[str, Any] | list[Any]:
    """Use curl for HTTPS (avoids local Python SSL cert store issues on some Macs)."""
    proc = subprocess.run(
        ["curl", "-sS", "-m", str(int(timeout)), url],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"curl failed {proc.returncode}: {proc.stderr[:500]}")
    return json.loads(proc.stdout)


def fetch_all_suggestions(base: str, profile: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    offset = 0
    limit = 500
    total: int | None = None
    while True:
        q = urllib.parse.urlencode({"limit": limit, "offset": offset})
        url = f"{base}/api/data/{urllib.parse.quote(profile)}/ai-suggestions/history?{q}"
        payload = http_get(url)
        assert isinstance(payload, dict)
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
    return out


def fetch_advanced_trades(base: str, profile: str, profile_path: str, days_back: int = 8000) -> list[dict[str, Any]]:
    enc = urllib.parse.quote(profile_path, safe="")
    q = urllib.parse.urlencode({"profile_path": profile_path, "days_back": days_back})
    # profile_path must be passed; use encoded path in query
    url = f"{base}/api/data/{urllib.parse.quote(profile)}/advanced-analytics?{q}"
    # FastAPI expects profile_path as query param; urllib.parse.urlencode already encodes
    url = f"{base}/api/data/{urllib.parse.quote(profile)}/advanced-analytics?profile_path={enc}&days_back={days_back}"
    payload = http_get(url, timeout=180.0)
    assert isinstance(payload, dict)
    rows = payload.get("trades") or []
    for r in rows:
        r["_profile"] = profile
    return rows


def fetch_autonomous_stats(base: str, profile: str) -> dict[str, Any]:
    url = f"{base}/api/data/{urllib.parse.quote(profile)}/autonomous/stats"
    payload = http_get(url, timeout=60.0)
    assert isinstance(payload, dict)
    return payload


def fetch_reasoning_thesis(base: str, profile: str) -> dict[str, Any]:
    url = f"{base}/api/data/{urllib.parse.quote(profile)}/autonomous/reasoning"
    return http_get(url, timeout=60.0)  # type: ignore[return-value]


def is_autonomous_suggestion(s: dict[str, Any]) -> bool:
    po = s.get("placed_order") or {}
    if isinstance(po, dict) and po.get("autonomous") is True:
        return True
    if str(s.get("rationale") or "").startswith("AUTONOMOUS"):
        return True
    return False


def pip_size_default() -> float:
    return 0.01


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


def session_bucket_utc(dt: datetime | None) -> str:
    if dt is None:
        return "unknown"
    h = dt.astimezone(timezone.utc).hour
    # Tokyo 00:00–08:00 UTC, London 08:00–13:00, NY 13:00–21:00, else off-hours
    if 0 <= h < 8:
        return "tokyo_00_08_utc"
    if 8 <= h < 13:
        return "london_08_13_utc"
    if 13 <= h < 21:
        return "ny_13_21_utc"
    return "off_hours"


def vol_regime_from_snapshot(s: dict[str, Any]) -> str:
    snap = s.get("market_snapshot")
    if not isinstance(snap, dict):
        return "unknown"
    vol = snap.get("volatility") or {}
    if isinstance(vol, dict):
        lab = vol.get("label") or vol.get("regime")
        if lab:
            return str(lab).lower()
    return "unknown"


def dxy_direction_from_snapshot(s: dict[str, Any]) -> str:
    snap = s.get("market_snapshot")
    if not isinstance(snap, dict):
        return "unknown"
    ca = snap.get("cross_assets") or snap.get("macro_bias") or {}
    # try a few shapes
    if isinstance(ca, dict):
        for k in ("dxy_trend", "dxy_bias", "dxy"):
            v = ca.get(k)
            if v:
                return str(v).lower()
        bias = ca.get("combined_bias")
        if bias:
            return str(bias).lower()
    mb = snap.get("macro_bias") or {}
    if isinstance(mb, dict):
        v = mb.get("usd_bias") or mb.get("combined_bias")
        if v:
            return str(v).lower()
    return "unknown"


def event_within_30m(s: dict[str, Any], entry_dt: datetime | None) -> bool | None:
    if entry_dt is None:
        return None
    snap = s.get("market_snapshot")
    if not isinstance(snap, dict):
        return None
    evs = snap.get("upcoming_events") or snap.get("imminent_events")
    if not isinstance(evs, list):
        return None
    # If snapshot has minutes-to-event style fields, use them; else unknown
    for ev in evs:
        if not isinstance(ev, dict):
            continue
        mins = ev.get("minutes_until") or ev.get("minutes_to_event") or ev.get("mins")
        try:
            m = float(mins)
            if m <= 30.0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def main() -> int:
    base = (sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASE).rstrip("/")
    print(f"BASE={base}\n")

    all_sug: list[dict[str, Any]] = []
    trades_by_profile: dict[str, list[dict[str, Any]]] = {}
    stats_by_profile: dict[str, dict[str, Any]] = {}
    reasoning_by_profile: dict[str, dict[str, Any]] = {}

    for pname, ppath in PROFILES:
        print(f"Fetching suggestions: {pname} …")
        all_sug.extend(fetch_all_suggestions(base, pname))
        print(f"Fetching advanced-analytics trades: {pname} …")
        trades_by_profile[pname] = fetch_advanced_trades(base, pname, ppath, days_back=8000)
        print(f"Fetching autonomous stats: {pname} …")
        try:
            stats_by_profile[pname] = fetch_autonomous_stats(base, pname)
        except Exception as e:
            stats_by_profile[pname] = {"_error": str(e)}
        try:
            reasoning_by_profile[pname] = fetch_reasoning_thesis(base, pname)
        except Exception as e:
            reasoning_by_profile[pname] = {"_error": str(e)}

    print(f"\nTOTAL suggestion rows fetched (both profiles): {len(all_sug)}")

    trade_index: dict[tuple[str, str], dict[str, Any]] = {}
    for pname, rows in trades_by_profile.items():
        for r in rows:
            tid = str(r.get("trade_id") or "").strip()
            if tid:
                trade_index[(pname, tid)] = r

    pip = pip_size_default()

    # SECTION 1 funnel (all suggestions)
    total_gen = len(all_sug)
    act_ctr = Counter((str(s.get("action") or "null")).lower() for s in all_sug)
    placed = [s for s in all_sug if str(s.get("action") or "").lower() == "placed"]
    placed_outcome = Counter(str(s.get("outcome_status") or "null").lower() for s in placed)
    placed_filled = [s for s in placed if str(s.get("outcome_status") or "").lower() == "filled"]
    filled_closed = [s for s in placed_filled if s.get("closed_at")]
    filled_open = [s for s in placed_filled if not s.get("closed_at")]
    closed = [s for s in all_sug if s.get("closed_at")]
    wl = Counter(str(s.get("win_loss") or "null").lower() for s in closed)
    wins = sum(1 for s in closed if str(s.get("win_loss") or "").lower() == "win")
    losses = sum(1 for s in closed if str(s.get("win_loss") or "").lower() == "loss")
    be = sum(1 for s in closed if str(s.get("win_loss") or "").lower() == "breakeven")
    pnl_pos = sum(1 for s in closed if isinstance(s.get("pnl"), (int, float)) and float(s["pnl"]) > 0)
    total_closed = len(closed)
    sum_pips = sum(float(s.get("pips") or 0) for s in closed)
    sum_usd = sum(float(s.get("pnl") or 0) for s in closed)

    print("\n======== SECTION 1: OVERALL SUGGESTION FUNNEL (both profiles, suggestion rows) ========")
    print(f"Total suggestions generated (rows fetched): {total_gen}")
    print("Breakdown by action (raw action field, lowercased; null string = never placed/rejected):")
    for k, v in act_ctr.most_common():
        print(f"  {k}: {v}")
    print("Of action=placed, outcome_status:")
    for k, v in placed_outcome.most_common():
        print(f"  {k}: {v}")
    print(f"Placed + filled: {len(placed_filled)}")
    print(f"Of filled, closed_at set (closed): {len(filled_closed)}")
    print(f"Of filled, closed_at null (still open in suggestion table): {len(filled_open)}")
    print("Of all rows with closed_at, win_loss:")
    for k, v in wl.most_common():
        print(f"  {k}: {v}")
    print(f"Overall win rate (win_loss=='win' / total closed): {wins}/{total_closed} = {100.0*wins/total_closed if total_closed else 0:.4f}%")
    print(f"Win rate if win = pnl>0: {pnl_pos}/{total_closed} = {100.0*pnl_pos/total_closed if total_closed else 0:.4f}%")
    print(f"Total P&L pips (sum suggestion.pips over closed): {sum_pips:.4f}")
    print(f"Total P&L USD (sum suggestion.pnl over closed): {sum_usd:.4f}")

    # SECTION 2 entry quality on closed suggestions + joined trade where possible
    print("\n======== SECTION 2: ENTRY QUALITY (closed suggestion rows; trade join by trade_id) ========")

    def entry_px(s: dict[str, Any]) -> float | None:
        fp = s.get("fill_price")
        if isinstance(fp, (int, float)) and fp:
            return float(fp)
        lp = s.get("limit_price")
        if isinstance(lp, (int, float)) and lp:
            return float(lp)
        return None

    winners = [s for s in closed if str(s.get("win_loss") or "").lower() == "win"]
    losers = [s for s in closed if str(s.get("win_loss") or "").lower() == "loss"]

    def avg_pips(rows: list[dict[str, Any]]) -> float | None:
        vals = [float(s.get("pips") or 0) for s in rows if s.get("pips") is not None]
        return sum(vals) / len(vals) if vals else None

    print(f"Avg winner pips (from suggestion.pips): {avg_pips(winners)}")
    print(f"Avg loser pips (from suggestion.pips): {avg_pips(losers)}")

    planned_rrs: list[float] = []
    realized_rrs: list[float] = []
    win_durs: list[float] = []
    lose_durs: list[float] = []
    lose_reasons = Counter()

    for s in closed:
        e = entry_px(s)
        sl = s.get("sl")
        tp = s.get("tp")
        if e and isinstance(sl, (int, float)) and isinstance(tp, (int, float)):
            sl_dist = abs(e - float(sl)) / pip
            tp_dist = abs(float(tp) - e) / pip
            if sl_dist > 1e-6:
                planned_rrs.append(tp_dist / sl_dist)
        pips_v = s.get("pips")
        if e and isinstance(sl, (int, float)) and isinstance(pips_v, (int, float)):
            risk = abs(e - float(sl)) / pip
            if risk > 1e-6:
                realized_rrs.append(float(pips_v) / risk)

        filled_dt = parse_iso(s.get("filled_at")) or parse_iso(s.get("created_utc"))
        closed_dt = parse_iso(s.get("closed_at"))
        if filled_dt and closed_dt:
            mins = (closed_dt - filled_dt).total_seconds() / 60.0
            if str(s.get("win_loss") or "").lower() == "win":
                win_durs.append(mins)
            elif str(s.get("win_loss") or "").lower() == "loss":
                lose_durs.append(mins)

        pname = str(s.get("_profile") or "")
        tid = str(s.get("trade_id") or "").strip()
        tr = trade_index.get((pname, tid)) if pname and tid else None
        if tr and str(s.get("win_loss") or "").lower() == "loss":
            er = str(tr.get("exit_reason") or "unknown")
            lose_reasons[er] += 1

    print(f"Avg planned R:R (|TP-entry|/|entry-SL| in pips): {sum(planned_rrs)/len(planned_rrs) if planned_rrs else None} (n={len(planned_rrs)})")
    print(f"Avg realized R:R (pips / risk_pips): {sum(realized_rrs)/len(realized_rrs) if realized_rrs else None} (n={len(realized_rrs)})")
    print(f"Median time-in-trade minutes (winners): {median(win_durs) if win_durs else None} (n={len(win_durs)})")
    print(f"Median time-in-trade minutes (losers): {median(lose_durs) if lose_durs else None} (n={len(lose_durs)})")
    print("Losers exit_reason (from advanced-analytics trade join, may miss if trade_id blank):")
    for k, v in lose_reasons.most_common():
        print(f"  {k}: {v}")
    print(f"Losers with no matched trade row for exit_reason: {len(losers) - sum(lose_reasons.values())}")

    # SECTION 3 segmentation tables
    print("\n======== SECTION 3: SEGMENTATION ========")

    def table_two_dim(label: str, key_fn, closed_only: bool = True):
        rows = [s for s in all_sug if (s.get("closed_at") if closed_only else True)]
        if closed_only:
            rows = [s for s in rows if s.get("closed_at")]
        by: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for s in rows:
            by[key_fn(s)].append(s)
        print(f"\n--- {label} (closed={closed_only}) ---")
        print("key | n | win_rate_% | avg_pips")
        for k in sorted(by.keys(), key=lambda x: (-len(by[x]), x)):
            g = by[k]
            w = sum(1 for s in g if str(s.get("win_loss") or "").lower() == "win")
            n = len(g)
            ap = sum(float(s.get("pips") or 0) for s in g) / n if n else 0.0
            wr = 100.0 * w / n if n else 0.0
            print(f"  {k} | {n} | {wr:.2f} | {ap:.3f}")

    table_two_dim("a) manual vs autonomous", lambda s: "autonomous" if is_autonomous_suggestion(s) else "manual_ui")
    table_two_dim("b) exit_strategy", lambda s: str(s.get("exit_strategy") or "null").lower())

    def trig(s):
        tf = s.get("trigger_family")
        if tf:
            return str(tf).lower()
        po = s.get("placed_order") or {}
        if isinstance(po, dict) and po.get("trigger_family"):
            return str(po.get("trigger_family")).lower()
        return "null"

    table_two_dim("c) trigger_family (all closed)", trig)
    # filter autonomous for c2
    by_aut = [s for s in closed if is_autonomous_suggestion(s)]
    by_trig_auto: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in by_aut:
        by_trig_auto[trig(s)].append(s)
    print("\n--- c) trigger_family (autonomous closed only) ---")
    for k in sorted(by_trig_auto.keys(), key=lambda x: (-len(by_trig_auto[x]), x)):
        g = by_trig_auto[k]
        w = sum(1 for s in g if str(s.get("win_loss") or "").lower() == "win")
        n = len(g)
        ap = sum(float(s.get("pips") or 0) for s in g) / n if n else 0.0
        wr = 100.0 * w / n if n else 0.0
        print(f"  {k} | {n} | {wr:.2f} | {ap:.3f}")

    table_two_dim("d) session bucket (UTC) at created_utc", lambda s: session_bucket_utc(parse_iso(s.get("created_utc"))))
    table_two_dim("e) confidence field", lambda s: str(s.get("confidence") or "null").lower())
    table_two_dim("f) model", lambda s: str(s.get("model") or "null"))
    table_two_dim("g) weekday at created_utc", lambda s: parse_iso(s.get("created_utc")).strftime("%A") if parse_iso(s.get("created_utc")) else "unknown")

    # SECTION 4 worst 10 losses by pips
    print("\n======== SECTION 4: LOSS AUTOPSY (10 worst by suggestion.pips among losses) ========")
    loss_sorted = sorted(losers, key=lambda s: float(s.get("pips") or 0.0))[:10]
    cluster_notes: list[str] = []
    for s in loss_sorted:
        pname = s.get("_profile")
        tid = s.get("trade_id")
        tr = trade_index.get((str(pname), str(tid or ""))) if pname else None
        print(
            f"  created={s.get('created_utc')} profile={pname} side={s.get('side')} "
            f"limit={s.get('limit_price')} sl={s.get('sl')} tp={s.get('tp')} exit={s.get('exit_price')} "
            f"pips={s.get('pips')} pnl={s.get('pnl')} conf={s.get('confidence')} exit_strat={s.get('exit_strategy')} "
            f"trig={trig(s)} sugg_id={s.get('suggestion_id')}"
        )
        if tr:
            print(
                f"    trade: exit_reason={tr.get('exit_reason')} dur_min={tr.get('duration_minutes')} "
                f"mae={tr.get('max_adverse_pips')} mfe={tr.get('max_favorable_pips')}"
            )
    # simple cluster: same calendar day
    days = [str(s.get("created_utc") or "")[:10] for s in loss_sorted]
    day_ctr = Counter(days)
    print("  Same-day clustering among worst-10 created dates:", dict(day_ctr))

    # SECTION 5 exit strategy performance (>=3 closed)
    print("\n======== SECTION 5: EXIT STRATEGY PERFORMANCE (closed, strat with n>=3) ========")
    by_es: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in closed:
        by_es[str(s.get("exit_strategy") or "null").lower()].append(s)
    for es, g in sorted(by_es.items(), key=lambda kv: -len(kv[1])):
        if len(g) < 3:
            continue
        w = sum(1 for s in g if str(s.get("win_loss") or "").lower() == "win")
        n = len(g)
        aw = avg_pips([s for s in g if str(s.get("win_loss") or "").lower() == "win"])
        al = avg_pips([s for s in g if str(s.get("win_loss") or "").lower() == "loss"])
        mae_list: list[float] = []
        mfe_list: list[float] = []
        for s in g:
            pname = str(s.get("_profile") or "")
            tid = str(s.get("trade_id") or "").strip()
            tr = trade_index.get((pname, tid)) if tid else None
            if tr:
                try:
                    if tr.get("max_adverse_pips") is not None:
                        mae_list.append(float(tr["max_adverse_pips"]))
                    if tr.get("max_favorable_pips") is not None:
                        mfe_list.append(float(tr["max_favorable_pips"]))
                except (TypeError, ValueError):
                    pass
        print(
            f"  {es}: n={n} win_rate={100.0*w/n:.2f}% avg_win_pips={aw} avg_loss_pips={al} "
            f"avg_MAE={sum(mae_list)/len(mae_list) if mae_list else None} (n={len(mae_list)}) "
            f"avg_MFE={sum(mfe_list)/len(mfe_list) if mfe_list else None} (n={len(mfe_list)})"
        )
        print(
            "    TP1 reached / trail-after-TP1: N/A — advanced-analytics trade payload does not "
            "include tp1_partial_done / managed-exit flags (only MAE/MFE, duration, exit_reason)."
        )

    # SECTION 6 thesis - only small sample via reasoning API
    print("\n======== SECTION 6: THESIS MONITOR (API /autonomous/reasoning caps checks) ========")
    for pname in stats_by_profile:
        r = reasoning_by_profile.get(pname) or {}
        if "_error" in r:
            print(f"  {pname}: reasoning fetch error: {r['_error']}")
            continue
        th = r.get("thesis_checks") or []
        print(f"  {pname}: thesis_checks returned = {len(th)} (endpoint uses small default limit; not full DB)")

    # SECTION 7 confidence calibration
    print("\n======== SECTION 7: CONFIDENCE CALIBRATION (closed rows) ========")
    for conf in ("high", "medium", "low"):
        g = [s for s in closed if str(s.get("confidence") or "").lower() == conf]
        w = sum(1 for s in g if str(s.get("win_loss") or "").lower() == "win")
        n = len(g)
        print(f"  {conf}: n={n} win_rate={100.0*w/n if n else 0:.2f}%")

    # SECTION 8 regime / timing / spread / dxy
    print("\n======== SECTION 8: TIMING & REGIME (best-effort from market_snapshot) ========")
    table_two_dim("volatility label", lambda s: vol_regime_from_snapshot(s))
    # event 30m - only if minutes_until present
    ev_y = ev_n = ev_u = 0
    for s in closed:
        entry_dt = parse_iso(s.get("filled_at")) or parse_iso(s.get("created_utc"))
        ev = event_within_30m(s, entry_dt)
        if ev is True:
            ev_y += 1
        elif ev is False:
            ev_n += 1
        else:
            ev_u += 1
    print(f"event_within_30m: true={ev_y} false={ev_n} unknown={ev_u}")
    sw: list[float] = []
    sls: list[float] = []
    for s in closed:
        snap = s.get("market_snapshot") or {}
        sp = None
        if isinstance(snap, dict):
            spot = snap.get("spot_price") or {}
            if isinstance(spot, dict):
                sp = spot.get("spread_pips")
        try:
            spf = float(sp) if sp is not None else None
        except (TypeError, ValueError):
            spf = None
        if spf is None:
            continue
        if str(s.get("win_loss") or "").lower() == "win":
            sw.append(spf)
        elif str(s.get("win_loss") or "").lower() == "loss":
            sls.append(spf)
    print(f"avg spread at snapshot (winners, n={len(sw)}): {sum(sw)/len(sw) if sw else None}")
    print(f"avg spread at snapshot (losers, n={len(sls)}): {sum(sls)/len(sls) if sls else None}")

    print("\n--- DXY / macro bias vs outcome (closed) ---")
    for side in ("buy", "sell"):
        by_dxy: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for s in closed:
            if str(s.get("side") or "").lower() != side:
                continue
            by_dxy[dxy_direction_from_snapshot(s)].append(s)
        print(f"  side={side}")
        for k in sorted(by_dxy.keys(), key=lambda x: (-len(by_dxy[x]), x)):
            g = by_dxy[k]
            w = sum(1 for s in g if str(s.get("win_loss") or "").lower() == "win")
            n = len(g)
            wr = 100.0 * w / n if n else 0.0
            ap = sum(float(s.get("pips") or 0) for s in g) / n if n else 0.0
            print(f"    bias={k} | n={n} | win_rate={wr:.2f}% | avg_pips={ap:.3f}")

    # SECTION 9 autonomous gate
    print("\n======== SECTION 9: AUTONOMOUS GATE (from /autonomous/stats recent_decisions) ========")
    for pname, st in stats_by_profile.items():
        if "_error" in st:
            print(f"  {pname}: {st['_error']}")
            continue
        dec = st.get("recent_decisions") or []
        win = st.get("window") or {}
        print(f"  {pname}: window total={win.get('total')} passes={win.get('passes')} blocks={win.get('blocks')} pass_rate={win.get('pass_rate_pct')}%")
        print(f"    recent_decisions returned: {len(dec)} rows (may be capped in backend)")
        fam_ctr = Counter(str((d.get("x") or {}).get("trigger_family")) for d in dec if isinstance(d, dict))
        if fam_ctr:
            print("    trigger_family counts in recent slice:", dict(fam_ctr))

    # SECTION 10 dump first 50 closed
    print("\n======== SECTION 10: CLOSED TRADES TABLE (first 50 by closed_at desc) ========")
    closed_sorted = sorted(closed, key=lambda s: str(s.get("closed_at") or ""), reverse=True)
    total_c = len(closed_sorted)
    print(f"total closed suggestion rows: {total_c}")
    print(
        "suggestion_id|date|side|entry|sl|tp|exit|pips|exit_reason|dur_min|exit_strat|conf|trig|session|vol|model|auto?"
    )
    for s in closed_sorted[:50]:
        pname = str(s.get("_profile") or "")
        tid = str(s.get("trade_id") or "").strip()
        tr = trade_index.get((pname, tid)) if pname and tid else None
        er = str(tr.get("exit_reason") or "") if tr else ""
        dur = tr.get("duration_minutes") if tr else None
        if dur is None:
            fa = parse_iso(s.get("filled_at")) or parse_iso(s.get("created_utc"))
            ca = parse_iso(s.get("closed_at"))
            if fa and ca:
                dur = round((ca - fa).total_seconds() / 60.0, 2)
        print(
            f"{s.get('suggestion_id')}|{str(s.get('closed_at'))[:10]}|{s.get('side')}|{entry_px(s)}|{s.get('sl')}|{s.get('tp')}|"
            f"{s.get('exit_price')}|{s.get('pips')}|{er}|{dur}|{s.get('exit_strategy')}|{s.get('confidence')}|{trig(s)}|"
            f"{session_bucket_utc(parse_iso(s.get('created_utc')))}|{vol_regime_from_snapshot(s)}|{s.get('model')}|"
            f"{'auto' if is_autonomous_suggestion(s) else 'manual'}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
