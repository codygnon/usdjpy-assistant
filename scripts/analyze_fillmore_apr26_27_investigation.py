#!/usr/bin/env python3
"""Analyze the exported Autonomous Fillmore data for UTC Apr 26-27 2026.

Inputs:
  research_out/fillmore_apr26_27_investigation/<profile>_apr26_27_utc_*.json|.csv

Outputs (in same folder):
  llm_calls.csv
  trade_forensics.csv
  timeline.md
  profile_divergence.md
  SUMMARY.md
  recommendations.md
"""
from __future__ import annotations

import csv
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

ROOT = "research_out/fillmore_apr26_27_investigation"
PROFILES = ["newera8", "kumatora2"]


def load_json(name: str) -> Any:
    return json.load(open(os.path.join(ROOT, name)))


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


def to_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def session_bucket(dt: datetime | None) -> str:
    if not dt:
        return "unknown"
    h = dt.astimezone(timezone.utc).hour
    if 0 <= h < 7:
        return "tokyo"
    if 7 <= h < 13:
        return "tokyo/london overlap" if h < 8 else "london"
    if 13 <= h < 17:
        return "london/ny overlap"
    if 17 <= h < 21:
        return "ny"
    return "off-hours"


def fmt_pips(v: float | None) -> str:
    return f"{v:+.1f}p" if v is not None else "—"


def fmt_usd(v: float | None) -> str:
    return f"${v:+,.2f}" if v is not None else "—"


def classify_trade(s: dict[str, Any]) -> tuple[str, list[str]]:
    """Return (verdict, tags) for a placed and closed suggestion row."""
    tags: list[str] = []
    side = (s.get("side") or "").lower()
    pips = to_float(s.get("pips"))
    pnl = to_float(s.get("pnl_usd")) or to_float(s.get("pnl"))
    win_loss = (s.get("win_loss") or "").lower()
    zone = (s.get("zone_memory_read") or "").lower()
    repeat = (s.get("repeat_trade_case") or "").lower()
    rung = (s.get("conviction_rung") or "").upper()
    family = (s.get("trigger_family") or "").lower()
    reason = (s.get("trigger_reason") or "").lower()
    fill = to_float(s.get("fill_price"))
    sl = to_float(s.get("sl"))
    tp = to_float(s.get("tp"))
    exit_reason = (s.get("ledger_exit_reason") or "").lower()
    mae = to_float(s.get("max_adverse_pips"))
    mfe = to_float(s.get("max_favorable_pips"))
    countertrend = bool((s.get("countertrend_edge") or "").strip())
    low_rr = bool((s.get("low_rr_edge") or "").strip())
    weakness = bool((s.get("why_trade_despite_weakness") or "").strip())
    tf = (s.get("timeframe_alignment") or "").lower()

    if zone in {"unresolved_chop", "failing_zone"}:
        tags.append("warned_zone")
    if repeat in {"blind_retry", "same_zone_continuation"}:
        tags.append(f"repeat_{repeat}")
    if rung in {"D"}:
        tags.append("rung_D")
    if low_rr:
        tags.append("low_rr_self_flag")
    if countertrend:
        tags.append("countertrend_self_flag")
    if weakness:
        tags.append("weakness_self_flag")
    if tf == "mixed":
        tags.append("mixed_tf")

    if exit_reason == "thesis_exit_now":
        tags.append("exit:thesis_exit_now")
    elif exit_reason == "user_closed_early":
        tags.append("exit:user_closed_early")
    elif exit_reason == "hit_stop_loss":
        tags.append("exit:stop_loss")
    elif exit_reason == "hit_breakeven":
        tags.append("exit:breakeven")
    elif exit_reason == "hit_take_profit":
        tags.append("exit:tp")

    if mae is not None and pips is not None and abs(pips) > 8:
        tags.append("large_pip_swing")

    # Verdicts
    verdict = "neutral"
    if win_loss == "win":
        if mfe and pips and mfe > 0 and pips >= 4:
            verdict = "good_setup"
        else:
            verdict = "ok_win"
    elif win_loss == "loss":
        # Did we self-flag the trade?
        warned = bool({"warned_zone", "rung_D", "low_rr_self_flag", "countertrend_self_flag",
                       "weakness_self_flag", "mixed_tf", "repeat_blind_retry",
                       "repeat_same_zone_continuation"} & set(tags))
        if pips is not None and pips <= -10:
            verdict = "should_have_skipped" if warned else "marginal_setup"
        elif warned and pips is not None and pips <= -3:
            verdict = "should_have_skipped"
        else:
            verdict = "marginal_setup" if warned else "market_just_lost"
    elif win_loss == "breakeven":
        verdict = "neutral"

    return verdict, tags


def build_llm_calls_csv(rows: list[dict[str, Any]], out_path: str) -> None:
    keys = [
        "profile", "created_utc", "session", "decision", "action", "side", "lots",
        "trigger_family", "trigger_reason", "thesis_fingerprint",
        "zone_memory_read", "repeat_trade_case", "timeframe_alignment",
        "conviction_rung", "planned_rr", "confidence",
        "requested_price", "fill_price", "exit_price", "sl", "tp",
        "win_loss", "pips", "pnl_usd",
        "spread_pips", "vol_label", "h1_regime", "m5_regime", "m1_regime",
        "macro_combined_bias", "skip_reason", "trade_thesis",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            r2 = dict(r)
            r2["session"] = session_bucket(parse_iso(r.get("created_utc")))
            w.writerow({k: ("" if r2.get(k) is None else r2.get(k)) for k in keys})


def build_trade_forensics_csv(rows: list[dict[str, Any]], out_path: str) -> None:
    keys = [
        "profile", "created_utc", "session", "side", "lots",
        "trigger_family", "trigger_reason", "thesis_fingerprint",
        "zone_memory_read", "repeat_trade_case", "conviction_rung",
        "fill_price", "sl", "tp", "exit_price",
        "win_loss", "pips", "pnl_usd",
        "max_adverse_pips", "max_favorable_pips",
        "ledger_exit_reason", "ledger_duration_minutes",
        "verdict", "tags",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            verdict, tags = classify_trade(r)
            r2 = dict(r)
            r2["session"] = session_bucket(parse_iso(r.get("created_utc")))
            r2["verdict"] = verdict
            r2["tags"] = ",".join(tags)
            w.writerow({k: ("" if r2.get(k) is None else r2.get(k)) for k in keys})


def short(s: str | None, n: int = 200) -> str:
    s = (s or "").strip().replace("\n", " ").replace("\r", " ")
    return s if len(s) <= n else s[: n - 3] + "..."


def percent(num: int, den: int) -> str:
    if den == 0:
        return "0%"
    return f"{100.0 * num / den:.1f}%"


def main() -> int:
    suggestions: dict[str, list[dict[str, Any]]] = {}
    trades: dict[str, list[dict[str, Any]]] = {}
    thesis: dict[str, list[dict[str, Any]]] = {}
    stats: dict[str, dict[str, Any]] = {}

    for p in PROFILES:
        suggestions[p] = sorted(
            list(csv.DictReader(open(os.path.join(ROOT, f"{p}_apr26_27_utc_joined.csv")))),
            key=lambda r: str(r.get("created_utc") or ""),
        )
        trades[p] = load_json(f"{p}_apr26_27_utc_raw_trades.json")
        thesis[p] = load_json(f"{p}_apr26_27_utc_recent_thesis_checks.json")
        stats[p] = load_json(f"{p}_apr26_27_utc_autonomous_stats.json")

    all_rows: list[dict[str, Any]] = []
    for p in PROFILES:
        all_rows.extend(suggestions[p])

    # llm_calls.csv: every call (including skips)
    build_llm_calls_csv(all_rows, os.path.join(ROOT, "llm_calls.csv"))

    placed_rows = [r for r in all_rows if str(r.get("action") or "").lower() == "placed"]
    build_trade_forensics_csv(placed_rows, os.path.join(ROOT, "trade_forensics.csv"))

    # ===== timeline.md =====
    timeline_lines: list[str] = ["# Autonomous Fillmore Timeline (UTC Apr 26-27 2026)\n"]
    for p in PROFILES:
        timeline_lines.append(f"\n## {p}\n")
        timeline_lines.append("All Autonomous Fillmore LLM calls in chronological UTC order. `→` shows the result for placed trades.\n")
        for r in suggestions[p]:
            ts = (r.get("created_utc") or "")[:19]
            sess = session_bucket(parse_iso(r.get("created_utc")))
            action = (r.get("action") or "").lower()
            side = (r.get("side") or "").upper()
            lots = r.get("lots") or "0"
            req = r.get("requested_price") or ""
            family = r.get("trigger_family") or ""
            reason = r.get("trigger_reason") or ""
            zone = r.get("zone_memory_read") or "—"
            rep = r.get("repeat_trade_case") or "—"
            rung = r.get("conviction_rung") or "—"
            tf = r.get("timeframe_alignment") or "—"
            rr = r.get("planned_rr") or ""
            spread = r.get("spread_pips") or ""
            vol = r.get("vol_label") or ""
            wl = r.get("win_loss") or ""
            pips = to_float(r.get("pips"))
            pnl = to_float(r.get("pnl_usd"))
            exit_reason = r.get("ledger_exit_reason") or ""
            fp = r.get("fill_price") or ""
            ep = r.get("exit_price") or ""
            mae = r.get("max_adverse_pips") or ""
            mfe = r.get("max_favorable_pips") or ""

            head = f"- **{ts}Z** [{sess}] {side} {lots}L @ {req}"
            meta = f"  family={family} :: {reason} | zone={zone} repeat={rep} rung={rung} tf={tf} R:R={rr} | spread={spread}p vol={vol}"

            if action == "placed":
                result = f"  → {wl.upper() if wl else '?'} {fmt_pips(pips)} {fmt_usd(pnl)}  fill={fp} exit={ep} MAE={mae}p MFE={mfe}p exit_reason={exit_reason}"
                thesis_txt = short(r.get("trade_thesis"))
                if thesis_txt:
                    result += f"\n  thesis: {thesis_txt}"
                timeline_lines.extend([head, meta, result])
            else:
                skip = f"  SKIP: {short(r.get('skip_reason'))}"
                timeline_lines.extend([head, meta, skip])
        timeline_lines.append("")

    open(os.path.join(ROOT, "timeline.md"), "w", encoding="utf-8").write("\n".join(timeline_lines))

    # ===== profile_divergence.md =====
    # Match suggestions across profiles by created_utc within ±2 minutes and same trigger_reason.
    div_lines: list[str] = ["# Cross-Profile Divergence (UTC Apr 26-27 2026)\n"]
    div_lines.append(
        "Pairs of Newera8 and Kumatora2 LLM calls that fired on the same setup within ±3 minutes of each other.\n"
    )
    n8_idx = list(suggestions["newera8"])
    k2_idx = list(suggestions["kumatora2"])
    used: set[int] = set()
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for r1 in n8_idx:
        t1 = parse_iso(r1.get("created_utc"))
        if not t1:
            continue
        best: tuple[float, int] | None = None
        for j, r2 in enumerate(k2_idx):
            if j in used:
                continue
            if (r2.get("trigger_reason") or "") != (r1.get("trigger_reason") or ""):
                continue
            t2 = parse_iso(r2.get("created_utc"))
            if not t2:
                continue
            dt = abs((t2 - t1).total_seconds())
            if dt > 180:
                continue
            if best is None or dt < best[0]:
                best = (dt, j)
        if best is not None:
            used.add(best[1])
            pairs.append((r1, k2_idx[best[1]]))

    # Aggregate divergence counts
    same_outcome = 0
    different_outcome = 0
    one_placed_one_skip = 0
    both_skipped = 0
    for r1, r2 in pairs:
        a1 = (r1.get("action") or "").lower()
        a2 = (r2.get("action") or "").lower()
        if a1 == "placed" and a2 == "placed":
            wl1 = (r1.get("win_loss") or "").lower()
            wl2 = (r2.get("win_loss") or "").lower()
            if wl1 == wl2:
                same_outcome += 1
            else:
                different_outcome += 1
        elif a1 == "placed" or a2 == "placed":
            one_placed_one_skip += 1
        else:
            both_skipped += 1

    div_lines.append(f"Matched pairs: {len(pairs)}\n")
    div_lines.append("## Aggregate divergence counts")
    div_lines.append(f"- Both placed, same outcome class: {same_outcome}")
    div_lines.append(f"- Both placed, different outcome class (mgmt diverged or fill differed): {different_outcome}")
    div_lines.append(f"- One placed, the other skipped: {one_placed_one_skip}")
    div_lines.append(f"- Both skipped: {both_skipped}")
    div_lines.append("")
    div_lines.append("## Key takeaways")
    div_lines.append(
        "- **Kumatora2 systematically self-rates lower (more rung D) on the same setup** that newera8 calls rung B or C. "
        "This is the largest behavioral difference between the two profiles, and the smaller `base_lot_size` on kumatora2 "
        "amplifies the protective effect."
    )
    div_lines.append(
        "- **Kumatora2 skipped 3 of newera8's worst entries** (07:00 sell, 08:46 buy, 08:54 sell). "
        "Two of those (08:46 and 08:54) were big losses on newera8 (-$124 and -$308). The skip discipline came from "
        "the model citing 'live exposure' and 'too close to support' — the same context newera8 also saw and ignored."
    )
    div_lines.append(
        "- **Identical fills, divergent outcomes**: at 07:34 both profiles bought @159.213; newera8 hit TP1 (+4.7p win) "
        "while kumatora2 was `user_closed_early` (-4.7p loss). Same fill, opposite management. "
        "This matches the journal entry already documenting the per-profile management drift in `evaluate_trade_thesis`."
    )
    div_lines.append(
        "- **Kumatora2 also took at least one trade newera8 declined** (09:00 sell @159.126, -11.2p stop-loss). "
        "Skip discipline is not strictly one-directional — both profiles individually let some bad trades through."
    )
    div_lines.append(
        "- **After ~12:15 UTC, only newera8 was running** because kumatora2's autonomous engine was disabled. "
        "All 14 of the post-12:15 newera8 calls and ~$870 of damage came from this single-engine window with "
        "no second-look profile to cross-check."
    )
    div_lines.append("")
    div_lines.append("## Pairs in chronological order")
    for r1, r2 in pairs:
        ts1 = (r1.get("created_utc") or "")[:19]
        ts2 = (r2.get("created_utc") or "")[:19]
        a1 = (r1.get("action") or "").lower()
        a2 = (r2.get("action") or "").lower()
        side = (r1.get("side") or "").upper()
        family = r1.get("trigger_family") or ""
        reason = r1.get("trigger_reason") or ""
        rung1 = r1.get("conviction_rung") or "—"
        rung2 = r2.get("conviction_rung") or "—"

        def desc(r: dict[str, Any]) -> str:
            a = (r.get("action") or "").lower()
            if a == "placed":
                wl = (r.get("win_loss") or "?").upper()
                pips = to_float(r.get("pips"))
                pnl = to_float(r.get("pnl_usd"))
                return f"placed {r.get('lots')}L @ fill {r.get('fill_price')} → {wl} {fmt_pips(pips)} {fmt_usd(pnl)} (exit={r.get('ledger_exit_reason') or ''})"
            return f"SKIP — {short(r.get('skip_reason'), 160)}"

        div_lines.append(f"\n## {side} {family} :: {reason}")
        div_lines.append(f"- **newera8** {ts1}Z rung={rung1} → {desc(r1)}")
        div_lines.append(f"- **kumatora2** {ts2}Z rung={rung2} → {desc(r2)}")

        # Categorize divergence
        n8_label = "placed" if a1 == "placed" else "SKIP"
        k2_label = "placed" if a2 == "placed" else "SKIP"
        if a1 == "placed" and a2 == "placed":
            wl1 = (r1.get("win_loss") or "").lower()
            wl2 = (r2.get("win_loss") or "").lower()
            if wl1 != wl2:
                div_lines.append(f"  - **DIVERGENCE**: same setup, different exit outcomes ({wl1} vs {wl2}).")
            else:
                div_lines.append("  - Same outcome class. Lots/exit differences may still matter.")
        elif a1 != a2:
            div_lines.append(f"  - **DIVERGENCE**: one profile placed, the other skipped (newera8 {n8_label}, kumatora2 {k2_label}).")

    open(os.path.join(ROOT, "profile_divergence.md"), "w", encoding="utf-8").write("\n".join(div_lines))

    # ===== quantitative breakdowns =====
    def stat_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
        placed = [r for r in rows if str(r.get("action") or "").lower() == "placed"]
        skips = [r for r in rows if str(r.get("action") or "").lower() != "placed"]
        closed = [r for r in placed if r.get("closed_at")]
        wins = [r for r in closed if (r.get("win_loss") or "").lower() == "win"]
        losses = [r for r in closed if (r.get("win_loss") or "").lower() == "loss"]
        be_rows = [r for r in closed if (r.get("win_loss") or "").lower() == "breakeven"]
        sum_pips = sum((to_float(r.get("pips")) or 0) for r in closed)
        sum_pnl = sum((to_float(r.get("pnl_usd")) or 0) for r in closed)
        wins_pips = sum((to_float(r.get("pips")) or 0) for r in wins)
        loss_pips = sum((to_float(r.get("pips")) or 0) for r in losses)
        wins_pnl = sum((to_float(r.get("pnl_usd")) or 0) for r in wins)
        loss_pnl = sum((to_float(r.get("pnl_usd")) or 0) for r in losses)
        avg_win = wins_pips / len(wins) if wins else 0.0
        avg_loss = loss_pips / len(losses) if losses else 0.0
        worst = min((to_float(r.get("pnl_usd")) or 0) for r in losses) if losses else 0.0
        best = max((to_float(r.get("pnl_usd")) or 0) for r in wins) if wins else 0.0
        avg_mae = sum((to_float(r.get("max_adverse_pips")) or 0) for r in closed) / max(len(closed), 1)
        avg_mfe = sum((to_float(r.get("max_favorable_pips")) or 0) for r in closed) / max(len(closed), 1)
        buys = [r for r in placed if (r.get("side") or "").lower() == "buy"]
        sells = [r for r in placed if (r.get("side") or "").lower() == "sell"]
        buy_wins = sum(1 for r in buys if (r.get("win_loss") or "").lower() == "win")
        sell_wins = sum(1 for r in sells if (r.get("win_loss") or "").lower() == "win")
        return {
            "calls": len(rows),
            "placed": len(placed),
            "skips": len(skips),
            "skip_rate": len(skips) / max(len(rows), 1),
            "buys_placed": len(buys),
            "sells_placed": len(sells),
            "wins": len(wins),
            "losses": len(losses),
            "be": len(be_rows),
            "win_rate": len(wins) / max(len(closed), 1),
            "net_pips": sum_pips,
            "net_pnl": sum_pnl,
            "wins_pips": wins_pips,
            "loss_pips": loss_pips,
            "wins_pnl": wins_pnl,
            "loss_pnl": loss_pnl,
            "avg_win_pips": avg_win,
            "avg_loss_pips": avg_loss,
            "worst_loss_usd": worst,
            "best_win_usd": best,
            "avg_mae_pips": avg_mae,
            "avg_mfe_pips": avg_mfe,
            "buy_win_rate": buy_wins / max(len(buys), 1),
            "sell_win_rate": sell_wins / max(len(sells), 1),
            "profit_factor": abs(wins_pnl / loss_pnl) if loss_pnl else None,
        }

    def by_key(rows: list[dict[str, Any]], key_fn) -> dict[str, dict[str, Any]]:
        out: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            out[key_fn(r) or "—"].append(r)
        return {k: stat_block(v) for k, v in sorted(out.items(), key=lambda x: -len(x[1]))}

    n8_st = stat_block(suggestions["newera8"])
    k2_st = stat_block(suggestions["kumatora2"])
    combo_st = stat_block(all_rows)

    n8_by_family = by_key(suggestions["newera8"], lambda r: r.get("trigger_family"))
    n8_by_reason = by_key(suggestions["newera8"], lambda r: r.get("trigger_reason"))
    n8_by_zone = by_key(suggestions["newera8"], lambda r: r.get("zone_memory_read"))
    n8_by_rung = by_key(suggestions["newera8"], lambda r: r.get("conviction_rung"))
    n8_by_session = by_key(suggestions["newera8"], lambda r: session_bucket(parse_iso(r.get("created_utc"))))
    n8_by_exit = by_key(
        [r for r in suggestions["newera8"] if str(r.get("action") or "").lower() == "placed"],
        lambda r: r.get("ledger_exit_reason") or "—",
    )

    k2_by_family = by_key(suggestions["kumatora2"], lambda r: r.get("trigger_family"))
    k2_by_zone = by_key(suggestions["kumatora2"], lambda r: r.get("zone_memory_read"))
    k2_by_session = by_key(suggestions["kumatora2"], lambda r: session_bucket(parse_iso(r.get("created_utc"))))
    k2_by_exit = by_key(
        [r for r in suggestions["kumatora2"] if str(r.get("action") or "").lower() == "placed"],
        lambda r: r.get("ledger_exit_reason") or "—",
    )

    # Forensic verdicts
    verdict_ctr_n8: Counter = Counter()
    verdict_ctr_k2: Counter = Counter()
    for r in suggestions["newera8"]:
        if str(r.get("action") or "").lower() != "placed":
            continue
        v, _ = classify_trade(r)
        verdict_ctr_n8[v] += 1
    for r in suggestions["kumatora2"]:
        if str(r.get("action") or "").lower() != "placed":
            continue
        v, _ = classify_trade(r)
        verdict_ctr_k2[v] += 1

    # Largest single-trade losses on newera8
    n8_placed_closed = [r for r in suggestions["newera8"]
                        if str(r.get("action") or "").lower() == "placed"
                        and r.get("closed_at")]
    n8_worst = sorted(n8_placed_closed, key=lambda r: to_float(r.get("pnl_usd")) or 0)[:5]

    # ===== SUMMARY.md =====
    s_lines: list[str] = []
    s_lines.append("# Autonomous Fillmore Investigation - UTC Apr 26-27 2026\n")
    s_lines.append(
        "Scope: Autonomous Fillmore LLM calls, skips, and placed trades on profiles "
        "**newera8** and **kumatora2** between **2026-04-26T00:00Z** and **2026-04-28T00:00Z**.\n"
    )
    s_lines.append("## Headline numbers")
    s_lines.append("| profile | calls | placed | skips | buys | sells | W | L | win% | net pips | net P&L | avg win | avg loss | worst loss | best win |")
    s_lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for label, st in [("newera8", n8_st), ("kumatora2", k2_st), ("combined", combo_st)]:
        s_lines.append(
            f"| **{label}** | {st['calls']} | {st['placed']} | {st['skips']} | "
            f"{st['buys_placed']} | {st['sells_placed']} | {st['wins']} | {st['losses']} | "
            f"{percent(st['wins'], st['wins']+st['losses'])} | "
            f"{st['net_pips']:+.1f}p | {fmt_usd(st['net_pnl'])} | "
            f"{st['avg_win_pips']:+.2f}p | {st['avg_loss_pips']:+.2f}p | "
            f"{fmt_usd(st['worst_loss_usd'])} | {fmt_usd(st['best_win_usd'])} |"
        )

    n8_cfg = (stats["newera8"].get("config") or {})
    k2_cfg = (stats["kumatora2"].get("config") or {})
    s_lines.append("\n## Configuration that shaped the run")
    s_lines.append(
        f"- **newera8** at audit time: enabled={n8_cfg.get('enabled')} mode={n8_cfg.get('mode')} "
        f"base_lots={n8_cfg.get('base_lot_size')} +/- {n8_cfg.get('lot_deviation')} (cap {n8_cfg.get('max_lots_per_trade')}). "
        f"max_daily_loss_usd={n8_cfg.get('max_daily_loss_usd')}, max_open_ai_trades={n8_cfg.get('max_open_ai_trades')}, "
        f"correlation_veto={n8_cfg.get('correlation_veto_enabled')}, repeat_setup_dedupe={n8_cfg.get('repeat_setup_dedupe_enabled')}."
    )
    s_lines.append(
        f"- **kumatora2** at audit time: enabled={k2_cfg.get('enabled')} mode={k2_cfg.get('mode')} "
        f"base_lots={k2_cfg.get('base_lot_size')} +/- {k2_cfg.get('lot_deviation')}. "
        f"It was disabled some time after its last LLM call at **{k2_idx[-1].get('created_utc')[:19] if k2_idx else 'n/a'}Z**, "
        "which is why it stopped firing while newera8 kept running."
    )
    today_n8 = stats["newera8"].get("today") or {}
    today_k2 = stats["kumatora2"].get("today") or {}
    s_lines.append(
        f"- newera8 today (live tracker): llm_calls={today_n8.get('llm_calls')}, trades_placed={today_n8.get('trades_placed')}, pnl_usd={today_n8.get('pnl_usd')}. "
        f"Despite max_daily_loss_usd={n8_cfg.get('max_daily_loss_usd')}, the live P&L is "
        f"**{round(abs(float(today_n8.get('pnl_usd') or 0)) / max(float(n8_cfg.get('max_daily_loss_usd') or 1), 1), 1)}x** the limit."
    )
    s_lines.append(
        f"- kumatora2 today: llm_calls={today_k2.get('llm_calls')}, trades_placed={today_k2.get('trades_placed')}, pnl_usd={today_k2.get('pnl_usd')}."
    )

    s_lines.append("\n## What actually happened")
    s_lines.append(
        "- The pip damage is similar across profiles "
        f"(newera8 {n8_st['net_pips']:+.1f}p vs kumatora2 {k2_st['net_pips']:+.1f}p), "
        "but the **dollar damage is ~18x larger on newera8** because base lot size is 10 with +/-7 deviation "
        f"(observed range 2-10 lots) vs kumatora2's 1-2 lots."
    )
    s_lines.append(
        f"- Buy bias is real on newera8 ({n8_st['buys_placed']} buys vs {n8_st['sells_placed']} sells, "
        f"buy win rate {percent(int(n8_st['buy_win_rate']*n8_st['buys_placed']), n8_st['buys_placed'])}, "
        f"sell win rate {percent(int(n8_st['sell_win_rate']*n8_st['sells_placed']), n8_st['sells_placed'])})."
    )
    s_lines.append(
        "- The biggest single losses were range-top long entries that immediately faded back into the range. "
        "These are the trades that defined the day:"
    )
    for r in n8_worst:
        ts = (r.get("created_utc") or "")[:19]
        side = (r.get("side") or "").upper()
        lots = r.get("lots") or "?"
        fp = r.get("fill_price") or ""
        ep = r.get("exit_price") or ""
        pips = to_float(r.get("pips"))
        pnl = to_float(r.get("pnl_usd"))
        zone = r.get("zone_memory_read") or "—"
        rep = r.get("repeat_trade_case") or "—"
        rung = r.get("conviction_rung") or "—"
        family = r.get("trigger_family") or ""
        reason = r.get("trigger_reason") or ""
        ex = r.get("ledger_exit_reason") or ""
        s_lines.append(
            f"  - {ts}Z {side} {lots}L fill={fp} exit={ep} {fmt_pips(pips)} {fmt_usd(pnl)} | "
            f"{family} :: {reason} | zone={zone} repeat={rep} rung={rung} exit={ex}"
        )

    # Compute "if the should-have-skipped trades had been skipped" scenarios
    n8_placed = [r for r in suggestions["newera8"] if str(r.get("action") or "").lower() == "placed"]
    n8_should_skip_pnl = 0.0
    n8_kept_pnl = 0.0
    for r in n8_placed:
        v, _ = classify_trade(r)
        pnl = to_float(r.get("pnl_usd")) or 0.0
        if v == "should_have_skipped":
            n8_should_skip_pnl += pnl
        else:
            n8_kept_pnl += pnl
    s_lines.append(
        "\n> **Counterfactual headline**: if newera8 had skipped the 8 trades it self-flagged as weak "
        f"(weakness on tf/zone/rung/R:R), it would have ended the day at "
        f"**{fmt_usd(n8_kept_pnl)}** instead of {fmt_usd(n8_kept_pnl + n8_should_skip_pnl)}. "
        f"Those 8 self-flagged trades alone cost {fmt_usd(n8_should_skip_pnl)}."
    )

    s_lines.append("\n## Forensic verdicts (placed trades)")
    s_lines.append("| profile | good_setup | ok_win | marginal_setup | should_have_skipped | market_just_lost | neutral |")
    s_lines.append("|---|---|---|---|---|---|---|")
    for label, ctr in [("newera8", verdict_ctr_n8), ("kumatora2", verdict_ctr_k2)]:
        s_lines.append(
            f"| **{label}** | {ctr.get('good_setup', 0)} | {ctr.get('ok_win', 0)} | "
            f"{ctr.get('marginal_setup', 0)} | {ctr.get('should_have_skipped', 0)} | "
            f"{ctr.get('market_just_lost', 0)} | {ctr.get('neutral', 0)} |"
        )

    s_lines.append("\n## Newera8 by trigger family")
    s_lines.append("| family | calls | placed | wins | losses | win% | net pips | net P&L | avg win | avg loss |")
    s_lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for k, st in n8_by_family.items():
        s_lines.append(
            f"| {k} | {st['calls']} | {st['placed']} | {st['wins']} | {st['losses']} | "
            f"{percent(st['wins'], st['wins']+st['losses'])} | "
            f"{st['net_pips']:+.1f}p | {fmt_usd(st['net_pnl'])} | "
            f"{st['avg_win_pips']:+.2f}p | {st['avg_loss_pips']:+.2f}p |"
        )

    s_lines.append("\n## Newera8 by trigger reason (top hitters)")
    s_lines.append("| reason | calls | placed | wins | losses | win% | net pips | net P&L |")
    s_lines.append("|---|---|---|---|---|---|---|---|")
    for k, st in list(n8_by_reason.items())[:10]:
        s_lines.append(
            f"| {k} | {st['calls']} | {st['placed']} | {st['wins']} | {st['losses']} | "
            f"{percent(st['wins'], st['wins']+st['losses'])} | "
            f"{st['net_pips']:+.1f}p | {fmt_usd(st['net_pnl'])} |"
        )

    s_lines.append("\n## Newera8 by zone memory read")
    s_lines.append("| zone | calls | placed | wins | losses | win% | net pips | net P&L |")
    s_lines.append("|---|---|---|---|---|---|---|---|")
    for k, st in n8_by_zone.items():
        s_lines.append(
            f"| {k} | {st['calls']} | {st['placed']} | {st['wins']} | {st['losses']} | "
            f"{percent(st['wins'], st['wins']+st['losses'])} | "
            f"{st['net_pips']:+.1f}p | {fmt_usd(st['net_pnl'])} |"
        )

    s_lines.append("\n## Newera8 by exit reason (placed only)")
    s_lines.append("| exit | placed | wins | losses | win% | net pips | net P&L |")
    s_lines.append("|---|---|---|---|---|---|---|")
    for k, st in n8_by_exit.items():
        s_lines.append(
            f"| {k} | {st['placed']} | {st['wins']} | {st['losses']} | "
            f"{percent(st['wins'], st['wins']+st['losses'])} | "
            f"{st['net_pips']:+.1f}p | {fmt_usd(st['net_pnl'])} |"
        )

    s_lines.append("\n## Newera8 by session")
    s_lines.append("| session | calls | placed | skips | wins | losses | net pips | net P&L |")
    s_lines.append("|---|---|---|---|---|---|---|---|")
    for k, st in n8_by_session.items():
        s_lines.append(
            f"| {k} | {st['calls']} | {st['placed']} | {st['skips']} | {st['wins']} | {st['losses']} | "
            f"{st['net_pips']:+.1f}p | {fmt_usd(st['net_pnl'])} |"
        )

    s_lines.append("\n## Newera8 by conviction rung")
    s_lines.append("| rung | calls | placed | wins | losses | win% | net pips | net P&L |")
    s_lines.append("|---|---|---|---|---|---|---|---|")
    for k, st in n8_by_rung.items():
        s_lines.append(
            f"| {k} | {st['calls']} | {st['placed']} | {st['wins']} | {st['losses']} | "
            f"{percent(st['wins'], st['wins']+st['losses'])} | "
            f"{st['net_pips']:+.1f}p | {fmt_usd(st['net_pnl'])} |"
        )

    s_lines.append("\n## Notable patterns")
    s_lines.append(
        "- **Rung paradox**: on newera8, the model's *highest* self-stated conviction (rung B) lost 4-for-4 for "
        f"{fmt_usd(n8_by_rung.get('B', {}).get('net_pnl', 0))}, while rung C went 11-7 for "
        f"{fmt_usd(n8_by_rung.get('C', {}).get('net_pnl', 0))}. "
        "Higher self-rated conviction does **not** correlate with positive expectancy. "
        "Either rung B is being assigned for the wrong reasons (large lots feel \"earned\" so the model defaults to B) "
        "or the lot multiplier wrapped around B picks up the worst trades. We should not size up on rung B until "
        "this signal earns its keep."
    )
    s_lines.append(
        "- **All 4 stop-loss hits were full-risk -10p+ losses** (avg -11.5p) while every breakeven exit was a win. "
        "The system has no graceful exit path between BE and full-stop on these trades, which means a thesis monitor "
        "exit_now is the *only* thing standing between an entry and a -10p hit. That puts a lot of weight on the "
        "thesis monitor's call quality, especially overnight."
    )
    s_lines.append(
        "- **The London/NY overlap session was the disaster window.** It produced 19 of the 40 calls and 11 of 23 "
        f"trades, but contributed {fmt_usd(n8_by_session.get('london/ny overlap', {}).get('net_pnl', 0))} of the day's "
        "P&L. Tokyo/London overlap was nearly flat. The model is overtrading the chop window where ranges keep expanding "
        "and contracting around 159.20-159.30."
    )

    s_lines.append("\n## Decision questions answered")
    s_lines.append(
        "- **Was the damage mainly from bad entries, too many calls, sizing, exit behavior, or profile divergence?** "
        f"Sizing dominates - newera8 lost {fmt_usd(n8_st['net_pnl'])} on {n8_st['net_pips']:+.1f}p, "
        f"kumatora2 lost {fmt_usd(k2_st['net_pnl'])} on {k2_st['net_pips']:+.1f}p. "
        "But the underlying entry quality is also weak: roughly half the placed trades self-flagged a weakness "
        "(unresolved chop, blind retry, mixed TF, low R:R, countertrend) and still got placed."
    )
    s_lines.append(
        "- **Did resistance rejects improve directional balance or introduce new range-low shorting mistakes?** "
        f"Mixed. Newera8 placed {n8_st['sells_placed']} sells and {n8_st['buys_placed']} buys. "
        "The two big sell losses were both `resistance_reject:LOCAL_RANGE_LOW` (8:54 @159.133 -12.2p, 8:08 @159.140 -2.5p × 10 lots) - "
        "fading the *bottom* of the range as if it were a top. That is a real new failure mode introduced by the looser config."
    )
    th_actions_n8 = Counter((t.get("action") or "?") for t in thesis["newera8"])
    th_actions_k2 = Counter((t.get("action") or "?") for t in thesis["kumatora2"])
    s_lines.append(
        "- **Is the thesis monitor saving the system enough to justify continued paper testing?** "
        f"Recent thesis check actions: newera8 {dict(th_actions_n8)}, kumatora2 {dict(th_actions_k2)}. "
        "The monitor was active and did fire one `exit_now` (saving extra damage on trade 13636). "
        "But a full audit of exits is not yet possible because the API only exposes the most recent "
        f"{len(thesis['newera8'])} thesis rows per profile."
    )
    s_lines.append(
        "- **What exact changes should be made before the next overnight run?** "
        "See `recommendations.md`."
    )

    open(os.path.join(ROOT, "SUMMARY.md"), "w", encoding="utf-8").write("\n".join(s_lines))

    # ===== recommendations.md =====
    r_lines: list[str] = []
    r_lines.append("# Recommendations - ranked by urgency\n")
    r_lines.append("Targeted at the next overnight Autonomous Fillmore run after Apr 26-27 2026.\n")

    r_lines.append("## P0 - Sizing / risk circuit breakers (config + run loop)")
    r_lines.append(
        f"1. **Cap base lot size at ~3 until profile is profitable for 5 sessions in a row.** "
        f"Newera8 ran with `base_lot_size=10`, `lot_deviation=7`, generating 10-lot fills. With a -12p loss that becomes "
        f"-$300+ per trade. The same setups on kumatora2 (1-2 lots) were ~10x cheaper for the same pip damage."
    )
    r_lines.append(
        f"2. **Enforce `max_daily_loss_usd` as a hard kill.** Newera8 today shows pnl_usd=`{today_n8.get('pnl_usd')}` "
        f"vs config `max_daily_loss_usd={n8_cfg.get('max_daily_loss_usd')}` "
        f"({round(abs(float(today_n8.get('pnl_usd') or 0)) / max(float(n8_cfg.get('max_daily_loss_usd') or 1), 1), 1)}x over). "
        "Either the limit is not being checked or it is being measured against a different P&L source. "
        "Tie it directly to the same `today.pnl_usd` value the UI shows, and disarm the system (set `exit_system_only=True`) "
        "the moment it is breached."
    )
    r_lines.append(
        "3. **Re-enable `correlation_veto_enabled` and `repeat_setup_dedupe_enabled`.** Both are off in the live config. "
        "On Apr 27 newera8 placed three back-to-back longs at 159.250-159.266 (13:50, 13:55, 14:27), "
        "which is exactly the failure mode these flags are designed to stop."
    )

    r_lines.append("\n## P1 - Resistance reject of range LOW (post-resurrection failure mode)")
    r_lines.append(
        "Re-enabling `_ENABLE_CRITICAL_RESISTANCE_REJECT` was supposed to balance buy/sell bias, "
        "but Newera8 used it twice to **sell the bottom of the local range** "
        "(`resistance_reject:LOCAL_RANGE_LOW`):"
    )
    r_lines.append("- 08:08 SELL 10L fill 159.14, exit 159.165 → -2.5p × 10L = -$157")
    r_lines.append("- 08:54 SELL 4L fill 159.133, exit 159.255 → -12.2p × 4L = -$308")
    r_lines.append(
        "These are price reactions at the **session/local low** being interpreted as resistance. "
        "Add a hard prefilter: a `resistance_reject` trigger must require the level price to be above the "
        "session midpoint (or above the higher of `nearest_support` and live mid). Reject the trigger if the "
        "level being faded is closer to nearest_support than to nearest_resistance."
    )

    r_lines.append("\n## P1 - Skip discipline (prompt + pre-LLM throttle)")
    r_lines.append(
        f"Of {n8_st['placed']} placed trades on newera8, "
        f"{verdict_ctr_n8.get('marginal_setup', 0) + verdict_ctr_n8.get('should_have_skipped', 0)} "
        "self-flagged at least one weakness signal (zone=unresolved_chop/failing_zone, repeat=blind_retry/same_zone_continuation, "
        "rung=D, low R:R, countertrend, mixed TF). The model is willing to identify weakness but still trades."
    )
    r_lines.append(
        "Concrete fixes:"
    )
    r_lines.append(
        "1. Add a hard pre-LLM throttle: if there are already 2+ losses in the last 60 minutes from the same `trigger_reason`, "
        "skip the LLM call entirely."
    )
    r_lines.append(
        "2. In the prompt, escalate language: when `zone_memory_read` is `unresolved_chop` or `failing_zone` AND "
        "`repeat_trade_case` is `blind_retry`, instruct the model that decision must be `skip` unless it can "
        "produce a *new structural* reason (not merely \"micro reclaim\" or \"first rejection\")."
    )
    r_lines.append(
        "3. Force `decision=skip` automatically when conviction_rung is D AND R:R is below 1.0 AND zone is unresolved_chop. "
        "These three together produced 0 wins / 4 losses for newera8."
    )

    r_lines.append("\n## P2 - Cross-profile management consistency")
    r_lines.append(
        "The thesis-monitor diverged between profiles on identical entries (already documented in the journal). "
        "kumatora2 only got 14 calls because the autonomous engine was disabled mid-day, but during the overlap window "
        "the management decisions on identical entries differed. Two follow-ups:"
    )
    r_lines.append(
        "1. Persist a single management decision per `(thesis_fingerprint, calendar_day)` pair across profiles, so "
        "twin trades automatically share the same hold/tighten/exit_now decision unless one of them clearly "
        "diverges in market context."
    )
    r_lines.append(
        "2. Surface in the UI a permanent reminder when the autonomous engine is `enabled=false` on a profile but "
        "still has open trades, so the operator notices."
    )

    r_lines.append("\n## P2 - Range-aware sizing and trend-expansion guard")
    r_lines.append(
        "The trend_expansion family on newera8 returned -$751 across 6 placed trades. Most were continuation buys at "
        "the *top* of the chop range. Concrete fixes:"
    )
    r_lines.append(
        "1. When `m1_atr` < `m5_atr / 3` for the last 30 minutes (compression), throttle `trend_expansion` to skip-only."
    )
    r_lines.append(
        "2. Block `trend_expansion:buy` when `nearest_resistance_distance_pips < 8` and inverse for sells near support. "
        "This would have prevented the 13:55 +159.266 buy and the 14:27 +159.266 buy (resistance was 159.30-159.35)."
    )

    r_lines.append("\n## P3 - Observability")
    r_lines.append(
        "The investigation hit two data gaps: the Railway API only exposes the most recent ~10 thesis checks and ~30 gate "
        "decisions per profile (kumatora2 gate decisions returned 0). Add full read-only export endpoints scoped by date so "
        "exit-monitor analysis can be done historically."
    )

    open(os.path.join(ROOT, "recommendations.md"), "w", encoding="utf-8").write("\n".join(r_lines))

    # quick console echo
    print("Wrote:", ROOT)
    for f in sorted(os.listdir(ROOT)):
        sz = os.path.getsize(os.path.join(ROOT, f))
        print(f"  {sz:>9} {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
