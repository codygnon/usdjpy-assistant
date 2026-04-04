#!/usr/bin/env python3
"""Diagnostic audit of cross-asset confluence backtest results (reads CSVs only)."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

BASE = Path("/Users/codygnon/Documents/usdjpy_assistant")


def load_trade_rows(run_name: str) -> list[dict[str, str]]:
    path = BASE / "research_out" / run_name / "trade_log.csv"
    if not path.exists():
        print(f"  !! {path} not found")
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_usdjpy_bars(n: int = 20) -> list[dict[str, str]]:
    """Load first n bars for spot-checking."""
    path = BASE / "research_out/USDJPY_M1_OANDA_1000k.csv"
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for _, row in zip(range(n), reader)]


def _float(x: str, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _int(x: str, default: int = 0) -> int:
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return default


def aggregate_round_trips(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """
    One engine trade_id can span multiple rows (tp1 partial, tp2, runner, etc.).
    Build one record per trade_id for PnL, direction, entry, and terminal exit.
    """
    by_id: dict[str, dict[str, Any]] = {}
    for t in rows:
        tid = t.get("trade_id", "")
        if not tid:
            continue
        if tid not in by_id:
            by_id[tid] = {
                "trade_id": tid,
                "direction": t.get("direction", ""),
                "entry_time": t.get("entry_time", ""),
                "entry_price": t.get("entry_price", ""),
                "entry_bar": _int(t.get("entry_bar", "")),
                "pnl_usd": 0.0,
                "max_bars_held": 0,
                "rows": [],
            }
        rec = by_id[tid]
        rec["pnl_usd"] += _float(t.get("pnl_usd", "0"))
        rec["max_bars_held"] = max(rec["max_bars_held"], _int(t.get("bars_held", "0")))
        rec["rows"].append(t)

    out: list[dict[str, Any]] = []
    for tid, rec in sorted(by_id.items(), key=lambda x: int(x[0])):
        rws = sorted(rec["rows"], key=lambda r: _int(r.get("exit_bar", "0")))
        last = rws[-1]
        first = rws[0]
        rec["exit_time"] = last.get("exit_time", "")
        rec["exit_price"] = last.get("exit_price", "")
        rec["exit_reason"] = last.get("exit_reason", "")
        rec["stop_loss"] = first.get("stop_loss", "")
        rec["final_event_type"] = last.get("event_type", "")
        # Any bias_flip / max_hold on any leg
        rec["any_exit_reasons"] = {r.get("exit_reason", "") for r in rws}
        out.append(rec)
    return out


def audit_trades(rows: list[dict[str, str]], label: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  AUDIT: {label}")
    print(f"  Raw CSV rows (events): {len(rows)}")

    if not rows:
        return

    trades = aggregate_round_trips(rows)
    print(f"  Round-trip trades (unique trade_id): {len(trades)}")

    # --- 1. First 5 round trips ---
    print("\n  --- First 5 Round Trips (Spot Check) ---")
    for t in trades[:5]:
        print(
            f"    id={t['trade_id']} Entry: {t.get('entry_time', 'N/A')} | "
            f"Dir: {t.get('direction', 'N/A')} | "
            f"Entry$: {t.get('entry_price', 'N/A')} | "
            f"Exit$: {t.get('exit_price', 'N/A')} | "
            f"SL@open: {t.get('stop_loss', 'N/A')} | "
            f"PnL: {t['pnl_usd']:.2f} | "
            f"Final exit: {t.get('exit_reason', 'N/A')} ({t.get('final_event_type', '')}) | "
            f"Max bars: {t['max_bars_held']}"
        )

    # --- 2. Exit reason (terminal, per round trip) ---
    print("\n  --- Exit Reasons (terminal, per round trip) ---")
    reasons: dict[str, int] = {}
    for t in trades:
        r = t.get("exit_reason") or "unknown"
        reasons[r] = reasons.get(r, 0) + 1
    n = len(trades)
    for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
        pct = 100 * c / n
        print(f"    {r:25s}: {c:4d} ({pct:5.1f}%)")

    # --- 2b. Event-level exit mix (all partial/full rows) ---
    print("\n  --- Exit Reasons (all CSV rows / events) ---")
    er: dict[str, int] = {}
    for t in rows:
        r = t.get("exit_reason", "unknown")
        er[r] = er.get(r, 0) + 1
    for r, c in sorted(er.items(), key=lambda x: -x[1]):
        pct = 100 * c / len(rows)
        print(f"    {r:25s}: {c:4d} ({pct:5.1f}%)")

    # --- 3. Direction split (round trip) ---
    print("\n  --- Direction Split (round trip PnL) ---")
    for direction in ("long", "short"):
        dir_trades = [t for t in trades if t.get("direction") == direction]
        if not dir_trades:
            continue
        wins = sum(1 for t in dir_trades if t["pnl_usd"] > 0)
        total_pnl = sum(t["pnl_usd"] for t in dir_trades)
        print(
            f"    {direction}: {len(dir_trades)} trades | "
            f"WR: {100 * wins / len(dir_trades):.1f}% | "
            f"Net: ${total_pnl:,.0f}"
        )

    # --- 4. SL analysis (terminal stop_loss) ---
    print("\n  --- Stop Loss Analysis (terminal exit = stop_loss) ---")
    sl_trades = [t for t in trades if (t.get("exit_reason") or "").lower() == "stop_loss"]
    if sl_trades:
        print(f"    SL exits: {len(sl_trades)} ({100 * len(sl_trades) / n:.1f}%)")
        sl_pnls = [t["pnl_usd"] for t in sl_trades]
        print(f"    SL PnL range: ${min(sl_pnls):,.0f} to ${max(sl_pnls):,.0f}")
        print(f"    SL mean PnL: ${sum(sl_pnls) / len(sl_pnls):,.0f}")

    # --- 5. Winner analysis (round trip) ---
    print("\n  --- Winner/Loser Analysis (round trip) ---")
    pnls = [t["pnl_usd"] for t in trades]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p < 0]
    scratches = [p for p in pnls if p == 0]

    if winners:
        print(
            f"    Winners: {len(winners)} | Avg: ${sum(winners) / len(winners):,.0f} | "
            f"Best: ${max(winners):,.0f}"
        )
    if losers:
        print(
            f"    Losers:  {len(losers)} | Avg: ${sum(losers) / len(losers):,.0f} | "
            f"Worst: ${min(losers):,.0f}"
        )
    if scratches:
        print(f"    Scratches: {len(scratches)}")

    if winners and losers:
        avg_win = sum(winners) / len(winners)
        avg_loss = abs(sum(losers) / len(losers))
        print(f"    Avg Win / Avg Loss ratio: {avg_win / avg_loss:.2f}")

    # --- 6. Monthly breakdown (round trip, entry month) ---
    print("\n  --- Monthly Breakdown (round trip, by entry month) ---")
    monthly: dict[str, dict[str, float | int]] = {}
    for t in trades:
        entry = (t.get("entry_time") or "")[:7]
        if len(entry) < 7:
            continue
        if entry not in monthly:
            monthly[entry] = {"trades": 0, "pnl": 0.0, "wins": 0}
        m = monthly[entry]
        m["trades"] = int(m["trades"]) + 1  # type: ignore[assignment]
        pnl = t["pnl_usd"]
        m["pnl"] = float(m["pnl"]) + pnl  # type: ignore[assignment]
        if pnl > 0:
            m["wins"] = int(m["wins"]) + 1  # type: ignore[assignment]

    for month in sorted(monthly.keys()):
        m = monthly[month]
        tr = int(m["trades"])
        wr = 100 * int(m["wins"]) / tr if tr else 0
        print(f"    {month}: {tr:3d} trades | WR: {wr:5.1f}% | Net: ${float(m['pnl']):>9,.0f}")

    # --- 7. Entry price sanity ---
    print("\n  --- Entry Price Sanity ---")
    prices = [_float(t.get("entry_price", "0")) for t in trades if t.get("entry_price")]
    if prices:
        print(f"    Range: {min(prices):.3f} - {max(prices):.3f}")
        print("    (USDJPY should be roughly 130-160 in this period)")

    # --- 8. Hold duration ---
    print("\n  --- Hold Duration (bars, max per round trip) ---")
    holds = [t["max_bars_held"] for t in trades if t["max_bars_held"] > 0]
    if holds:
        holds_sorted = sorted(holds)
        mid = holds_sorted[len(holds_sorted) // 2]
        print(
            f"    Min: {min(holds)} | Median: {mid} | "
            f"Max: {max(holds)} | Mean: {sum(holds) / len(holds):.0f}"
        )
        at_max = sum(1 for h in holds if h >= 238)
        print(f"    At/near max hold (≥238): {at_max} ({100 * at_max / len(holds):.1f}%)")

    # --- 9. Sizing distribution (first row per trade = first event size / units closed) ---
    print("\n  --- Position Sizing (first event closed_units+remaining_units) ---")
    sizes: dict[str, int] = defaultdict(int)
    by_id_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for t in rows:
        by_id_rows[t.get("trade_id", "")].append(t)
    for tid, rws in by_id_rows.items():
        if not tid:
            continue
        r0 = sorted(rws, key=lambda r: _int(r.get("exit_bar", "0")))[0]
        cu = _int(r0.get("closed_units", "0"))
        ru = _int(r0.get("remaining_units", "0"))
        initial = cu + ru
        sizes[str(initial)] += 1
    for s, c in sorted(sizes.items(), key=lambda x: -x[1]):
        print(f"    Initial units {s}: {c} trades")


def check_column_names() -> None:
    for run in ("cross_asset_confluence_real", "cross_asset_confluence_zero"):
        path = BASE / "research_out" / run / "trade_log.csv"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
            print(f"\n  {run} columns:")
            print(f"    {header}")


if __name__ == "__main__":
    print("CROSS-ASSET CONFLUENCE BACKTEST AUDIT")
    print("=" * 60)

    print("\n--- Trade Log Column Names ---")
    check_column_names()

    real_rows = load_trade_rows("cross_asset_confluence_real")
    zero_rows = load_trade_rows("cross_asset_confluence_zero")

    audit_trades(real_rows, "REAL SPREAD (2.0 pip + 0.1 slip)")
    audit_trades(zero_rows, "ZERO SPREAD")

    print(f"\n{'=' * 60}")
    print("AUDIT COMPLETE")
