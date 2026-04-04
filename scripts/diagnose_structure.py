#!/usr/bin/env python3
"""Diagnose structural problems in cross-asset confluence from trade_log only."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

BASE = Path("/Users/codygnon/Documents/usdjpy_assistant")


def _int(x: str) -> int:
    return int(float(x))


def _float(x: str) -> float:
    return float(x)


def load_round_trips(run_name: str) -> list[dict]:
    """Load trade log CSV rows, group by trade_id into round trips."""
    path = BASE / "research_out" / run_name / "trade_log.csv"
    rows_by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows_by_id[row["trade_id"]].append(row)

    trips = []
    for tid, rows in sorted(rows_by_id.items(), key=lambda x: int(x[0])):
        rows = sorted(rows, key=lambda r: _int(r["exit_bar"]))
        first = rows[0]
        last = rows[-1]

        total_pnl = sum(_float(r["pnl_usd"]) for r in rows)
        total_pips = sum(_float(r["pnl_pips"]) for r in rows)

        exit_types = [r["exit_reason"] for r in rows]
        hit_tp1 = any("tp1" in e.lower() for e in exit_types)
        hit_tp2 = any("tp2" in e.lower() for e in exit_types)
        terminal_exit = last["exit_reason"]

        trips.append(
            {
                "id": tid,
                "direction": first["direction"],
                "entry_time": first["entry_time"],
                "entry_price": _float(first["entry_price"]),
                "exit_price": _float(last["exit_price"]),
                "total_pnl": total_pnl,
                "total_pips": total_pips,
                "terminal_exit": terminal_exit,
                "hit_tp1": hit_tp1,
                "hit_tp2": hit_tp2,
                "bars_held": max(_int(r["bars_held"]) for r in rows),
                "initial_size": _int(first["closed_units"]) + _int(first["remaining_units"]),
                "n_events": len(rows),
                "events": rows,
            }
        )
    return trips


def diagnose(trips: list[dict], label: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  STRUCTURAL DIAGNOSIS: {label}")
    print(f"  {len(trips)} round trips")
    print(f"{'=' * 70}")

    # -------------------------------------------------------
    # DIAGNOSIS 1: How much damage do stop losses do?
    # -------------------------------------------------------
    print("\n  ─── DIAGNOSIS 1: Stop Loss Damage ───")

    sl_trades = [t for t in trips if t["terminal_exit"] == "stop_loss"]
    non_sl = [t for t in trips if t["terminal_exit"] != "stop_loss"]

    sl_pnl = sum(t["total_pnl"] for t in sl_trades)
    non_sl_pnl = sum(t["total_pnl"] for t in non_sl)

    print(f"    SL trades:     {len(sl_trades):4d} | Net: ${sl_pnl:>10,.0f}")
    print(f"    Non-SL trades: {len(non_sl):4d} | Net: ${non_sl_pnl:>10,.0f}")

    sl_with_tp1 = [t for t in sl_trades if t["hit_tp1"]]
    sl_no_tp1 = [t for t in sl_trades if not t["hit_tp1"]]
    print(
        f"    SL after hitting TP1: {len(sl_with_tp1):4d} | Net: ${sum(t['total_pnl'] for t in sl_with_tp1):>10,.0f}"
    )
    print(f"    SL without TP1:       {len(sl_no_tp1):4d} | Net: ${sum(t['total_pnl'] for t in sl_no_tp1):>10,.0f}")

    sl_bars = [t["bars_held"] for t in sl_trades]
    if sl_bars:
        sl_bars.sort()
        print(
            f"    SL hold bars: min={min(sl_bars)} median={sl_bars[len(sl_bars) // 2]} "
            f"mean={sum(sl_bars) / len(sl_bars):.0f} max={max(sl_bars)}"
        )
        quick_sl = [t for t in sl_trades if t["bars_held"] <= 30]
        print(f"    Quick SL (≤30 bars): {len(quick_sl)} | Net: ${sum(t['total_pnl'] for t in quick_sl):>10,.0f}")

    # -------------------------------------------------------
    # DIAGNOSIS 2: Session close — winners or losers?
    # -------------------------------------------------------
    print("\n  ─── DIAGNOSIS 2: Session Close Analysis ───")

    sc_trades = [t for t in trips if t["terminal_exit"] == "session_close"]
    sc_winners = [t for t in sc_trades if t["total_pnl"] > 0]
    sc_losers = [t for t in sc_trades if t["total_pnl"] < 0]

    print(f"    Session close total: {len(sc_trades)}")
    print(
        f"    SC winners: {len(sc_winners):4d} | Net: ${sum(t['total_pnl'] for t in sc_winners):>10,.0f} | "
        f"Avg: ${sum(t['total_pnl'] for t in sc_winners) / max(len(sc_winners), 1):>8,.0f}"
    )
    print(
        f"    SC losers:  {len(sc_losers):4d} | Net: ${sum(t['total_pnl'] for t in sc_losers):>10,.0f} | "
        f"Avg: ${sum(t['total_pnl'] for t in sc_losers) / max(len(sc_losers), 1):>8,.0f}"
    )

    sc_with_tp1 = [t for t in sc_trades if t["hit_tp1"]]
    sc_with_tp1_losing = [t for t in sc_with_tp1 if t["total_pnl"] <= 0]
    print(
        f"    SC that hit TP1 first: {len(sc_with_tp1):4d} | Net: ${sum(t['total_pnl'] for t in sc_with_tp1):>10,.0f}"
    )
    print(
        f"    SC hit TP1 but NET NEGATIVE: {len(sc_with_tp1_losing):4d} | "
        f"Net: ${sum(t['total_pnl'] for t in sc_with_tp1_losing):>10,.0f}"
    )

    print("\n    Counterfactual: What if we closed 100% at TP1 (+10 pips)?")
    tp1_full_close_pnl = 0.0
    for t in sc_with_tp1:
        pip_value = t["initial_size"] / t["entry_price"]
        hypothetical = 10.0 * pip_value
        tp1_full_close_pnl += hypothetical
    actual_sc_tp1_pnl = sum(t["total_pnl"] for t in sc_with_tp1)
    print(f"    Actual PnL (SC trades that hit TP1):       ${actual_sc_tp1_pnl:>10,.0f}")
    print(f"    Hypothetical (full close at TP1):          ${tp1_full_close_pnl:>10,.0f}")
    print(f"    Delta (hypothetical - actual):             ${tp1_full_close_pnl - actual_sc_tp1_pnl:>10,.0f}")
    print(
        "    NOTE: Hypothetical = +10 pips × full initial_size / entry for each SC trade that "
        "ever hit TP1.\n"
        "          It ignores partial profits already booked, runner path, and shorts vs longs — "
        "NOT incremental\n"
        "          or achievable PnL; do not treat the delta as a fixable 'cost'."
    )

    # -------------------------------------------------------
    # DIAGNOSIS 3: Long vs Short deep dive
    # -------------------------------------------------------
    print("\n  ─── DIAGNOSIS 3: Long vs Short ───")

    for direction in ("long", "short"):
        dir_trades = [t for t in trips if t["direction"] == direction]
        if not dir_trades:
            continue

        wins = [t for t in dir_trades if t["total_pnl"] > 0]
        wr = 100 * len(wins) / len(dir_trades)
        net = sum(t["total_pnl"] for t in dir_trades)

        sl_count = sum(1 for t in dir_trades if t["terminal_exit"] == "stop_loss")
        sl_rate = 100 * sl_count / len(dir_trades)

        tp1_count = sum(1 for t in dir_trades if t["hit_tp1"])
        tp1_rate = 100 * tp1_count / len(dir_trades)

        print(
            f"    {direction.upper():5s}: {len(dir_trades):3d} trades | WR: {wr:5.1f}% | "
            f"Net: ${net:>9,.0f} | SL%: {sl_rate:5.1f}% | TP1%: {tp1_rate:5.1f}%"
        )

    long_trades = [t for t in trips if t["direction"] == "long"]
    long_pnl = sum(t["total_pnl"] for t in long_trades)
    long_wins = sum(1 for t in long_trades if t["total_pnl"] > 0)
    long_wr = 100 * long_wins / len(long_trades) if long_trades else 0

    long_gross_w = sum(t["total_pnl"] for t in long_trades if t["total_pnl"] > 0)
    long_gross_l = abs(sum(t["total_pnl"] for t in long_trades if t["total_pnl"] < 0))
    long_pf = long_gross_w / long_gross_l if long_gross_l else 999

    print(
        f"\n    LONG-ONLY hypothetical: {len(long_trades)} trades | WR: {long_wr:.1f}% | "
        f"PF: {long_pf:.2f} | Net: ${long_pnl:>9,.0f}"
    )

    # -------------------------------------------------------
    # DIAGNOSIS 4: TP1 hit rate by sizing tier
    # -------------------------------------------------------
    print("\n  ─── DIAGNOSIS 4: Confluence Quality ───")

    for size_label, size_val in [("3-lot (300k)", 300_000), ("2-lot (200k)", 200_000), ("1-lot (100k)", 100_000)]:
        tier = [t for t in trips if t["initial_size"] == size_val]
        if not tier:
            continue

        tier_wins = sum(1 for t in tier if t["total_pnl"] > 0)
        tier_wr = 100 * tier_wins / len(tier) if tier else 0
        tier_pnl = sum(t["total_pnl"] for t in tier)
        tier_tp1 = sum(1 for t in tier if t["hit_tp1"])
        tier_sl = sum(1 for t in tier if t["terminal_exit"] == "stop_loss")

        tier_gross_w = sum(t["total_pnl"] for t in tier if t["total_pnl"] > 0)
        tier_gross_l = abs(sum(t["total_pnl"] for t in tier if t["total_pnl"] < 0))
        tier_pf = tier_gross_w / tier_gross_l if tier_gross_l else 999

        print(
            f"    {size_label}: {len(tier):3d} trades | WR: {tier_wr:5.1f}% | "
            f"PF: {tier_pf:.2f} | TP1%: {100 * tier_tp1 / len(tier):5.1f}% | "
            f"SL%: {100 * tier_sl / len(tier):5.1f}% | Net: ${tier_pnl:>9,.0f}"
        )

    # -------------------------------------------------------
    # DIAGNOSIS 5: What does a "good" trade look like?
    # -------------------------------------------------------
    print("\n  ─── DIAGNOSIS 5: Best & Worst Trades ───")

    sorted_trips = sorted(trips, key=lambda t: t["total_pnl"], reverse=True)

    print("    TOP 5:")
    for t in sorted_trips[:5]:
        print(
            f"      ${t['total_pnl']:>8,.0f} | {t['direction']:5s} | {t['entry_time'][:16]} | "
            f"TP1:{t['hit_tp1']} TP2:{t['hit_tp2']} | Exit: {t['terminal_exit']} | "
            f"Bars: {t['bars_held']} | Size: {t['initial_size'] // 1000}k"
        )

    print("    BOTTOM 5:")
    for t in sorted_trips[-5:]:
        print(
            f"      ${t['total_pnl']:>8,.0f} | {t['direction']:5s} | {t['entry_time'][:16]} | "
            f"TP1:{t['hit_tp1']} TP2:{t['hit_tp2']} | Exit: {t['terminal_exit']} | "
            f"Bars: {t['bars_held']} | Size: {t['initial_size'] // 1000}k"
        )


if __name__ == "__main__":
    print("CROSS-ASSET CONFLUENCE: STRUCTURAL DIAGNOSIS")

    for run, label in [
        ("cross_asset_confluence_zero", "ZERO SPREAD"),
        ("cross_asset_confluence_real", "REAL SPREAD (2.0 pip)"),
    ]:
        trips = load_round_trips(run)
        diagnose(trips, label)

    print(f"\n{'=' * 70}")
    print("DONE")
