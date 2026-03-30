#!/usr/bin/env python3
"""
Ownership Stability Diagnostic — Cross-Dataset Agreement Analysis.

Compares the ownership grid across 500k and 1000k datasets to find:
1. Stable ownership: same owner on both datasets (high confidence)
2. Stable no-trade: no-trade on both datasets
3. Unstable cells: different owner across datasets (low confidence)
4. Actionable clusters: groups of stable cells that could form positive rules

This is the analytical foundation for v2 positive ownership.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/Users/codygnon/Documents/usdjpy_assistant")
OUT_DIR = ROOT / "research_out"


def main() -> int:
    with open(OUT_DIR / "diagnostic_strategy_ownership.json") as f:
        data = json.load(f)

    grid_500k = data["500k"]["grid"]
    grid_1000k = data["1000k"]["grid"]
    simple_500k = data["500k"]["simple_grid"]
    simple_1000k = data["1000k"]["simple_grid"]

    all_cells = sorted(set(list(grid_500k.keys()) + list(grid_1000k.keys())))
    all_simple = sorted(set(list(simple_500k.keys()) + list(simple_1000k.keys())))

    STRATEGIES = ["v14", "london_v2", "v44_ny"]

    # ── 1. Full 30-cell grid: cross-dataset agreement ──
    print("=" * 100)
    print("  OWNERSHIP STABILITY: Full Grid (regime × ER × ΔER)")
    print("=" * 100)

    header = (f"  {'Condition':<42s} {'500k Owner':<14s} {'1000k Owner':<14s} "
              f"{'Agreement':<12s} {'500k n':<8s} {'1000k n':<8s}")
    print(header)
    print(f"  {'─' * 96}")

    stable_same = []
    stable_notrade = []
    unstable = []
    one_sided = []

    for cell in all_cells:
        c500 = grid_500k.get(cell)
        c1000 = grid_1000k.get(cell)

        owner_500 = c500["owner"] if c500 else "—"
        owner_1000 = c1000["owner"] if c1000 else "—"

        n500 = sum(c500["strategies"][s].get("count", 0) for s in STRATEGIES) if c500 else 0
        n1000 = sum(c1000["strategies"][s].get("count", 0) for s in STRATEGIES) if c1000 else 0

        if c500 is None or c1000 is None:
            agreement = "ONE-SIDED"
            one_sided.append(cell)
        elif owner_500 == owner_1000:
            if owner_500 == "NO-TRADE":
                agreement = "STABLE NT"
                stable_notrade.append(cell)
            else:
                agreement = "STABLE"
                stable_same.append(cell)
        else:
            agreement = "UNSTABLE"
            unstable.append(cell)

        print(f"  {cell:<42s} {owner_500:<14s} {owner_1000:<14s} "
              f"{agreement:<12s} {n500:<8d} {n1000:<8d}")

    print(f"\n  Summary:")
    print(f"    Stable (same owner):    {len(stable_same)} cells")
    print(f"    Stable (no-trade):      {len(stable_notrade)} cells")
    print(f"    Unstable (disagree):    {len(unstable)} cells")
    print(f"    One-sided (data gap):   {len(one_sided)} cells")

    # ── 2. Stable ownership detail ──
    print(f"\n{'─' * 100}")
    print(f"  STABLE OWNERSHIP CELLS (same owner on both datasets)")
    print(f"{'─' * 100}")

    stable_by_owner = defaultdict(list)
    for cell in stable_same:
        owner = grid_500k[cell]["owner"]
        stable_by_owner[owner].append(cell)

    for owner in STRATEGIES:
        cells = stable_by_owner.get(owner, [])
        if not cells:
            continue
        total_n_500 = 0
        total_n_1000 = 0
        total_pips_500 = 0.0
        total_pips_1000 = 0.0
        print(f"\n  {owner} stably owns {len(cells)} cells:")
        for cell in cells:
            c500 = grid_500k[cell]
            c1000 = grid_1000k[cell]
            s500 = c500["strategies"].get(owner, {})
            s1000 = c1000["strategies"].get(owner, {})
            n500 = s500.get("count", 0)
            n1000 = s1000.get("count", 0)
            avg500 = s500.get("avg_pips", 0)
            avg1000 = s1000.get("avg_pips", 0)
            net500 = s500.get("net_pips", 0)
            net1000 = s1000.get("net_pips", 0)
            total_n_500 += n500
            total_n_1000 += n1000
            total_pips_500 += net500
            total_pips_1000 += net1000
            print(f"    {cell:<42s} "
                  f"500k: {n500:>3d} trades, {avg500:>+6.1f} avg | "
                  f"1000k: {n1000:>3d} trades, {avg1000:>+6.1f} avg")
        avg_all_500 = total_pips_500 / max(1, total_n_500)
        avg_all_1000 = total_pips_1000 / max(1, total_n_1000)
        print(f"    {'TOTAL':<42s} "
              f"500k: {total_n_500:>3d} trades, {avg_all_500:>+6.1f} avg, {total_pips_500:>+7.1f} net | "
              f"1000k: {total_n_1000:>3d} trades, {avg_all_1000:>+6.1f} avg, {total_pips_1000:>+7.1f} net")

    # ── 3. Unstable cells detail ──
    print(f"\n{'─' * 100}")
    print(f"  UNSTABLE CELLS (different owner across datasets)")
    print(f"{'─' * 100}")

    for cell in unstable:
        c500 = grid_500k.get(cell)
        c1000 = grid_1000k.get(cell)
        owner_500 = c500["owner"] if c500 else "—"
        owner_1000 = c1000["owner"] if c1000 else "—"
        print(f"\n  {cell}: 500k={owner_500}, 1000k={owner_1000}")
        for strat in STRATEGIES:
            s500 = c500["strategies"].get(strat, {}) if c500 else {}
            s1000 = c1000["strategies"].get(strat, {}) if c1000 else {}
            n500 = s500.get("count", 0)
            n1000 = s1000.get("count", 0)
            avg500 = s500.get("avg_pips", 0) if n500 > 0 else None
            avg1000 = s1000.get("avg_pips", 0) if n1000 > 0 else None
            avg500_str = f"{avg500:>+6.1f}" if avg500 is not None else "   ---"
            avg1000_str = f"{avg1000:>+6.1f}" if avg1000 is not None else "   ---"
            print(f"    {strat:<14s} 500k: n={n500:>3d} avg={avg500_str} | "
                  f"1000k: n={n1000:>3d} avg={avg1000_str}")

    # ── 4. Simplified view: regime × ER stability ──
    print(f"\n{'=' * 100}")
    print(f"  SIMPLIFIED STABILITY: regime × ER (ignoring ΔER)")
    print(f"{'=' * 100}")

    header = (f"  {'Condition':<32s} {'500k Owner':<14s} {'1000k Owner':<14s} "
              f"{'Agreement':<12s}")
    print(header)
    print(f"  {'─' * 70}")

    simple_stable = []
    simple_unstable = []

    for cell in all_simple:
        c500 = simple_500k.get(cell)
        c1000 = simple_1000k.get(cell)
        owner_500 = c500["owner"] if c500 else "—"
        owner_1000 = c1000["owner"] if c1000 else "—"

        if c500 is None or c1000 is None:
            agreement = "ONE-SIDED"
        elif owner_500 == owner_1000:
            agreement = "STABLE NT" if owner_500 == "NO-TRADE" else "STABLE"
            if owner_500 != "NO-TRADE":
                simple_stable.append((cell, owner_500))
        else:
            agreement = "UNSTABLE"
            simple_unstable.append(cell)

        print(f"  {cell:<32s} {owner_500:<14s} {owner_1000:<14s} {agreement:<12s}")

    print(f"\n  Simplified: {len(simple_stable)} stable, {len(simple_unstable)} unstable")

    # ── 5. Actionable clusters ──
    print(f"\n{'=' * 100}")
    print(f"  ACTIONABLE CLUSTERS: Stable cells grouped by owner")
    print(f"{'=' * 100}")

    # For each strategy, show its stable cells and the "misallocated" trades
    # that currently trade in those cells but belong to a different strategy.
    for owner in STRATEGIES:
        owner_cells = [cell for cell in stable_same
                       if grid_500k[cell]["owner"] == owner]
        if not owner_cells:
            continue

        print(f"\n  ── {owner} stable cluster ({len(owner_cells)} cells) ──")

        # Count trades by all strategies in these cells
        for_owner = {"500k": 0, "1000k": 0}
        for_owner_pips = {"500k": 0.0, "1000k": 0.0}
        misallocated = {"500k": defaultdict(lambda: {"n": 0, "pips": 0.0}),
                        "1000k": defaultdict(lambda: {"n": 0, "pips": 0.0})}

        for cell in owner_cells:
            for dk, grid in [("500k", grid_500k), ("1000k", grid_1000k)]:
                c = grid.get(cell)
                if not c:
                    continue
                for strat in STRATEGIES:
                    s = c["strategies"].get(strat, {})
                    n = s.get("count", 0)
                    net = s.get("net_pips", 0)
                    if n == 0:
                        continue
                    if strat == owner:
                        for_owner[dk] += n
                        for_owner_pips[dk] += net
                    else:
                        misallocated[dk][strat]["n"] += n
                        misallocated[dk][strat]["pips"] += net

        print(f"    Owner trades:  500k={for_owner['500k']} ({for_owner_pips['500k']:+.1f}p) | "
              f"1000k={for_owner['1000k']} ({for_owner_pips['1000k']:+.1f}p)")
        for strat in STRATEGIES:
            if strat == owner:
                continue
            m500 = misallocated["500k"][strat]
            m1000 = misallocated["1000k"][strat]
            if m500["n"] > 0 or m1000["n"] > 0:
                print(f"    {strat} in this cluster: "
                      f"500k={m500['n']} ({m500['pips']:+.1f}p) | "
                      f"1000k={m1000['n']} ({m1000['pips']:+.1f}p)")

    # ── 6. Positive authorization candidates ──
    print(f"\n{'=' * 100}")
    print(f"  POSITIVE AUTHORIZATION CANDIDATES")
    print(f"{'=' * 100}")
    print(f"\n  Criteria: stable owner on both datasets, owner avg > +3p on both,")
    print(f"           at least one other strategy with negative avg in same cell")
    print()

    candidates = []
    for cell in stable_same:
        owner = grid_500k[cell]["owner"]
        c500 = grid_500k[cell]
        c1000 = grid_1000k[cell]
        s500_owner = c500["strategies"].get(owner, {})
        s1000_owner = c1000["strategies"].get(owner, {})
        avg500 = s500_owner.get("avg_pips", 0)
        avg1000 = s1000_owner.get("avg_pips", 0)

        if avg500 <= 3.0 or avg1000 <= 3.0:
            continue

        # Check if any other strategy is negative
        has_negative_other = False
        for strat in STRATEGIES:
            if strat == owner:
                continue
            s500_other = c500["strategies"].get(strat, {})
            s1000_other = c1000["strategies"].get(strat, {})
            n500 = s500_other.get("count", 0)
            n1000 = s1000_other.get("count", 0)
            avg500_o = s500_other.get("avg_pips", 0) if n500 > 0 else None
            avg1000_o = s1000_other.get("avg_pips", 0) if n1000 > 0 else None
            if (avg500_o is not None and avg500_o < 0) or (avg1000_o is not None and avg1000_o < 0):
                has_negative_other = True
                break

        if not has_negative_other:
            continue

        # This is a candidate: the owner has edge, and at least one non-owner is losing
        total_misalloc_n = 0
        total_misalloc_pips = 0.0
        for strat in STRATEGIES:
            if strat == owner:
                continue
            for grid in [grid_500k, grid_1000k]:
                c = grid.get(cell, {})
                s = c.get("strategies", {}).get(strat, {})
                n = s.get("count", 0)
                net = s.get("net_pips", 0)
                if n > 0 and s.get("avg_pips", 0) < 0:
                    total_misalloc_n += n
                    total_misalloc_pips += net

        candidates.append({
            "cell": cell,
            "owner": owner,
            "owner_avg_500k": avg500,
            "owner_avg_1000k": avg1000,
            "misalloc_trades": total_misalloc_n,
            "misalloc_pips": total_misalloc_pips,
        })

    candidates.sort(key=lambda c: c["misalloc_pips"])

    for c in candidates:
        cell = c["cell"]
        print(f"  {cell}")
        print(f"    Owner: {c['owner']} (500k avg={c['owner_avg_500k']:+.1f}p, "
              f"1000k avg={c['owner_avg_1000k']:+.1f}p)")
        # Show all strategies
        for dk, grid in [("500k", grid_500k), ("1000k", grid_1000k)]:
            gc = grid.get(cell, {})
            parts = []
            for strat in STRATEGIES:
                s = gc.get("strategies", {}).get(strat, {})
                n = s.get("count", 0)
                if n > 0:
                    parts.append(f"{strat}={s['avg_pips']:+.1f}({n})")
                else:
                    parts.append(f"{strat}=---")
            print(f"      {dk}: {', '.join(parts)}")
        print(f"    Misallocated losers: {c['misalloc_trades']} trades, {c['misalloc_pips']:+.1f}p")
        print()

    if not candidates:
        print("  No candidates found matching criteria.")

    # ── 7. Biggest single improvement opportunity ──
    print(f"{'─' * 100}")
    print(f"  BIGGEST IMPROVEMENT OPPORTUNITIES (blocking negative-avg non-owners)")
    print(f"{'─' * 100}")
    print()

    # For ALL stable cells (not just candidates), find non-owner strategies
    # that have negative avg on BOTH datasets
    opportunities = []
    for cell in stable_same:
        owner = grid_500k[cell]["owner"]
        for strat in STRATEGIES:
            if strat == owner:
                continue
            s500 = grid_500k[cell]["strategies"].get(strat, {})
            s1000 = grid_1000k[cell]["strategies"].get(strat, {})
            n500 = s500.get("count", 0)
            n1000 = s1000.get("count", 0)
            avg500 = s500.get("avg_pips", 0) if n500 > 0 else None
            avg1000 = s1000.get("avg_pips", 0) if n1000 > 0 else None

            # Must be negative on both datasets with at least some data
            if avg500 is not None and avg1000 is not None and avg500 < 0 and avg1000 < 0:
                net500 = s500.get("net_pips", 0)
                net1000 = s1000.get("net_pips", 0)
                opportunities.append({
                    "cell": cell,
                    "owner": owner,
                    "blocked_strategy": strat,
                    "n_500k": n500,
                    "n_1000k": n1000,
                    "avg_500k": avg500,
                    "avg_1000k": avg1000,
                    "net_500k": net500,
                    "net_1000k": net1000,
                    "total_net": net500 + net1000,
                })

    opportunities.sort(key=lambda o: o["total_net"])

    for o in opportunities:
        print(f"  Block {o['blocked_strategy']} in {o['cell']} (owner: {o['owner']})")
        print(f"    500k: {o['n_500k']} trades, {o['avg_500k']:+.1f} avg, {o['net_500k']:+.1f} net")
        print(f"    1000k: {o['n_1000k']} trades, {o['avg_1000k']:+.1f} avg, {o['net_1000k']:+.1f} net")
        print(f"    Combined: {o['total_net']:+.1f} pips saveable")
        print()

    if not opportunities:
        print("  No opportunities found where a non-owner is negative on both datasets.")

    # Save results
    result = {
        "stable_same_owner": stable_same,
        "stable_no_trade": stable_notrade,
        "unstable": unstable,
        "one_sided": one_sided,
        "candidates": candidates,
        "opportunities": opportunities,
    }
    out_path = OUT_DIR / "diagnostic_ownership_stability.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nResults written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
