#!/usr/bin/env python3
"""
Phase A: Chart-First Routing Diagnostic (Shadow Router).

Tests the core premise of chart-first routing:
  On trades that actually happened in the promoted stack,
  does a conservative chart-first ownership table disagree often enough,
  and usefully enough, to justify the next phase?

Uses the promoted Variant K baseline (F + G + I + K) with equity coupling.
Compares each trade's session-assigned strategy to the stable ownership
table's recommendation for that chart-state cell.

IMPORTANT GUARDRAILS:
  - Ownership table uses ONLY stable-across-datasets cells
  - Unstable / sparse / one-sided cells → no_table_decision
  - Coverage-gap bars are exploratory only, NOT proof of edge
  - Does NOT claim cross-session portability for any strategy
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.ownership_table import cell_key, der_bucket, er_bucket, load_conservative_table
from scripts import backtest_merged_integrated_tokyo_london_v2_ny as merged_engine
from scripts import backtest_variant_i_pbt_standdown as variant_i
from scripts import backtest_variant_k_london_cluster as variant_k

OUT_DIR = ROOT / "research_out"
STARTING_EQUITY = 100_000.0

DATASETS = [
    str(OUT_DIR / "USDJPY_M1_OANDA_500k.csv"),
    str(OUT_DIR / "USDJPY_M1_OANDA_1000k.csv"),
]


def _dataset_key(path: str) -> str:
    name = Path(path).name
    if "500k" in name:
        return "500k"
    if "1000k" in name:
        return "1000k"
    raise ValueError(f"Unknown dataset: {name}")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)


# ── Classify trades ──

def _classify_trade(
    trade: merged_engine.TradeRow,
    cell: str,
    table: dict[str, dict[str, Any]],
) -> str:
    """
    Classify a trade as agreement, disagreement, or no_table_decision.
    """
    if cell not in table:
        return "no_table_decision"

    rec = table[cell]["recommended_strategy"]

    if rec == "NO-TRADE":
        # Trade occurred in a stable no-trade cell → disagreement
        return "disagreement"

    if trade.strategy == rec:
        return "agreement"
    else:
        return "disagreement"


# ── Per-dataset analysis ──

def _analyze_dataset(
    dataset: str,
    table: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    dk = _dataset_key(dataset)
    print(f"\n{'='*60}")
    print(f"  Dataset: {dk}")
    print(f"{'='*60}")

    # Build promoted Variant K coupled trades
    kept, baseline, classified_dynamic, dyn_time_idx, _, _ = (
        variant_k.build_variant_k_pre_coupling_kept(dataset)
    )
    coupled = merged_engine._apply_shared_equity_coupling(
        sorted(kept, key=lambda t: (t.exit_time, t.entry_time)),
        STARTING_EQUITY,
        v14_max_units=baseline["v14_max_units"],
    )

    print(f"  Promoted-stack trades: {len(coupled)}")

    # Classify each trade
    agreements: list[dict[str, Any]] = []
    disagreements: list[dict[str, Any]] = []
    no_decisions: list[dict[str, Any]] = []

    cell_breakdown: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "agreement": 0, "disagreement": 0, "no_table_decision": 0}
    )

    for trade in coupled:
        # Look up regime cell
        regime_info = variant_i._lookup_regime_with_dynamic(
            classified_dynamic, dyn_time_idx, trade.entry_time,
        )
        idx = dyn_time_idx.get_indexer(
            [pd.Timestamp(trade.entry_time)], method="ffill"
        )[0]
        full_row = classified_dynamic.iloc[idx]
        er = float(full_row.get("sf_er", 0.5))
        if np.isnan(er):
            er = 0.5

        cell = cell_key(
            regime_info["regime_label"],
            er_bucket(er),
            der_bucket(regime_info["delta_er"]),
        )

        classification = _classify_trade(trade, cell, table)

        rec = table.get(cell, {}).get("recommended_strategy", "none")
        entry = {
            "entry_time": trade.entry_time.isoformat(),
            "strategy": trade.strategy,
            "cell": cell,
            "recommended": rec,
            "pips": float(trade.pips),
            "usd": float(trade.usd),
            "side": trade.side,
        }

        if classification == "agreement":
            agreements.append(entry)
        elif classification == "disagreement":
            disagreements.append(entry)
        else:
            no_decisions.append(entry)

        cell_breakdown[cell]["total"] += 1
        cell_breakdown[cell][classification] += 1

    total = len(coupled)
    n_agree = len(agreements)
    n_disagree = len(disagreements)
    n_nodec = len(no_decisions)
    agree_rate = n_agree / total * 100 if total > 0 else 0

    print(f"  Agreement:        {n_agree:4d} ({n_agree/total*100:5.1f}%)")
    print(f"  Disagreement:     {n_disagree:4d} ({n_disagree/total*100:5.1f}%)")
    print(f"  No table decision:{n_nodec:4d} ({n_nodec/total*100:5.1f}%)")

    # Compute stats for agreement and disagreement groups
    def _group_stats(trades: list[dict[str, Any]]) -> dict[str, Any]:
        if not trades:
            return {
                "count": 0, "wins": 0, "losses": 0, "win_rate": 0,
                "avg_pips": 0, "net_pips": 0, "avg_usd": 0, "net_usd": 0,
            }
        n = len(trades)
        wins = sum(1 for t in trades if t["pips"] > 0)
        return {
            "count": n,
            "wins": wins,
            "losses": n - wins,
            "win_rate": round(wins / n * 100, 1),
            "avg_pips": round(sum(t["pips"] for t in trades) / n, 2),
            "net_pips": round(sum(t["pips"] for t in trades), 2),
            "avg_usd": round(sum(t["usd"] for t in trades) / n, 2),
            "net_usd": round(sum(t["usd"] for t in trades), 2),
        }

    agree_stats = _group_stats(agreements)
    disagree_stats = _group_stats(disagreements)
    nodec_stats = _group_stats(no_decisions)

    print(f"\n  Agreement stats:    WR={agree_stats['win_rate']:5.1f}%  "
          f"avg={agree_stats['avg_pips']:+7.2f}p  net={agree_stats['net_pips']:+8.1f}p  "
          f"${agree_stats['net_usd']:+10,.2f}")
    print(f"  Disagreement stats: WR={disagree_stats['win_rate']:5.1f}%  "
          f"avg={disagree_stats['avg_pips']:+7.2f}p  net={disagree_stats['net_pips']:+8.1f}p  "
          f"${disagree_stats['net_usd']:+10,.2f}")
    print(f"  No-decision stats:  WR={nodec_stats['win_rate']:5.1f}%  "
          f"avg={nodec_stats['avg_pips']:+7.2f}p  net={nodec_stats['net_pips']:+8.1f}p  "
          f"${nodec_stats['net_usd']:+10,.2f}")

    # Disagreement pair breakdown
    pair_counter: Counter = Counter()
    pair_pips: dict[str, list[float]] = defaultdict(list)
    for d in disagreements:
        pair = f"{d['strategy']} -> {d['recommended']}"
        pair_counter[pair] += 1
        pair_pips[pair].append(d["pips"])

    pair_breakdown: list[dict[str, Any]] = []
    for pair, count in pair_counter.most_common():
        pips_list = pair_pips[pair]
        pair_breakdown.append({
            "pair": pair,
            "count": count,
            "avg_pips": round(sum(pips_list) / len(pips_list), 2),
            "net_pips": round(sum(pips_list), 2),
            "wins": sum(1 for p in pips_list if p > 0),
            "losses": sum(1 for p in pips_list if p <= 0),
        })

    print(f"\n  Disagreement pairs:")
    for pb in pair_breakdown:
        print(f"    {pb['pair']:35s}  n={pb['count']:3d}  "
              f"avg={pb['avg_pips']:+7.2f}p  net={pb['net_pips']:+8.1f}p  "
              f"W/L={pb['wins']}/{pb['losses']}")

    # Coverage gap exploratory (bars with stable recommendation but no trade)
    # Use classified_dynamic which has regime state on every M1 bar
    coverage_gap = _exploratory_coverage_gaps(
        classified_dynamic, dyn_time_idx, coupled, table, dk,
    )

    return {
        "total_trades": total,
        "agreement_count": n_agree,
        "disagreement_count": n_disagree,
        "no_table_decision_count": n_nodec,
        "agreement_rate_pct": round(agree_rate, 1),
        "agreement_stats": agree_stats,
        "disagreement_stats": disagree_stats,
        "no_table_decision_stats": nodec_stats,
        "disagreement_pair_breakdown": pair_breakdown,
        "disagreement_samples": disagreements[:30],
        "cell_breakdown": dict(cell_breakdown),
        "coverage_gap_exploratory": coverage_gap,
    }


# ── Coverage gap exploration ──

def _session_for_utc_hour(hour: int) -> str | None:
    """Map UTC hour to session (approximate, ignoring DST for exploration)."""
    if 16 <= hour < 22:
        return "tokyo"
    if 8 <= hour < 12:
        return "london"
    if 13 <= hour < 16:
        return "ny"
    return None


_SESSION_NATIVE_STRATEGY = {
    "tokyo": "v14",
    "london": "london_v2",
    "ny": "v44_ny",
}


def _exploratory_coverage_gaps(
    classified: pd.DataFrame,
    time_idx: pd.DatetimeIndex,
    trades: list[merged_engine.TradeRow],
    table: dict[str, dict[str, Any]],
    dk: str,
) -> dict[str, Any]:
    """
    Exploratory only: count bars where stable ownership table has a
    recommendation but no trade occurred.

    NOT proof of edge. NOT a success gate.
    """
    # Get entry times for quick lookup
    trade_entry_minutes = set()
    for t in trades:
        # Round to minute for matching
        ts = pd.Timestamp(t.entry_time)
        trade_entry_minutes.add(ts.floor("min"))

    # Sample classified bars at 5-minute intervals to reduce computation
    # (checking every M1 bar across 500k-1000k bars is expensive)
    sample_stride = 5  # every 5th bar
    gap_count = 0
    cross_session_gap_count = 0
    session_gap_counts: dict[str, int] = Counter()
    regime_gap_counts: dict[str, int] = Counter()
    samples: list[dict[str, Any]] = []
    total_sampled = 0

    for i in range(0, len(classified), sample_stride):
        row = classified.iloc[i]
        ts = pd.Timestamp(row["time"])

        # Skip if outside any trading session
        hour = ts.hour if ts.tzinfo is None else ts.tz_convert("UTC").hour
        session = _session_for_utc_hour(hour)
        if session is None:
            continue

        total_sampled += 1

        # Skip if a trade was entered near this bar
        bar_minute = ts.floor("min")
        if bar_minute in trade_entry_minutes:
            continue

        # Get regime cell
        label = str(row.get("regime_hysteresis", "ambiguous"))
        er = float(row.get("sf_er", 0.5))
        der = float(row.get("sf_delta_er", 0.0))
        if np.isnan(er):
            er = 0.5
        if np.isnan(der):
            der = 0.0
        cell = cell_key(label, er_bucket(er), der_bucket(der))

        # Check if stable ownership exists for this cell
        if cell not in table:
            continue
        rec = table[cell].get("recommended_strategy")
        if rec is None or rec == "NO-TRADE":
            continue

        gap_count += 1
        regime_gap_counts[cell] += 1

        # Is the recommendation different from the session-native strategy?
        native = _SESSION_NATIVE_STRATEGY.get(session)
        if native and rec != native:
            cross_session_gap_count += 1
            session_gap_counts[f"{session}:{native}->{rec}"] += 1

            if len(samples) < 20:
                samples.append({
                    "time": ts.isoformat(),
                    "session": session,
                    "cell": cell,
                    "native_strategy": native,
                    "table_recommends": rec,
                    "note": "EXPLORATORY ONLY — not proof of edge",
                })

    print(f"\n  Coverage gap exploration (sampled every {sample_stride} bars):")
    print(f"    Bars sampled in trading sessions: {total_sampled}")
    print(f"    Bars with stable recommendation, no trade: {gap_count}")
    print(f"    Of which cross-session (rec ≠ native): {cross_session_gap_count}")
    if session_gap_counts:
        for key, count in session_gap_counts.most_common(10):
            print(f"      {key}: {count}")

    return {
        "note": "EXPLORATORY ONLY — these bars are NOT proof of missed edge. "
                "They show where chart ownership would have recommended a "
                "non-native strategy. No claim of cross-session portability.",
        "sample_stride": sample_stride,
        "bars_sampled_in_sessions": total_sampled,
        "bars_with_stable_rec_no_trade": gap_count,
        "cross_session_gaps": cross_session_gap_count,
        "cross_session_breakdown": dict(session_gap_counts),
        "top_regime_cells": dict(regime_gap_counts.most_common(10)),
        "samples": samples,
    }


# ── Verdict ──

def _verdict(results: dict[str, dict[str, Any]]) -> str:
    lines: list[str] = []

    # Check agreement rate
    agree_rates = {
        dk: r["agreement_rate_pct"] for dk, r in results.items()
    }
    avg_agree = sum(agree_rates.values()) / len(agree_rates)

    lines.append("=" * 60)
    lines.append("PHASE A VERDICT")
    lines.append("=" * 60)
    lines.append("")

    # Agreement rate assessment
    for dk, rate in agree_rates.items():
        lines.append(f"{dk} agreement rate: {rate:.1f}%")
    lines.append(f"Average agreement rate: {avg_agree:.1f}%")
    lines.append("")

    if avg_agree > 90:
        lines.append("FINDING: Agreement rate is HIGH (>90%).")
        lines.append("Chart-first ownership table rarely disagrees with session routing.")
        lines.append("The current session-first + veto stack already captures most of the value.")
        lines.append("Recommendation: Phase B is NOT justified. Stay with v1.")
    elif avg_agree < 50:
        lines.append("FINDING: Agreement rate is VERY LOW (<50%).")
        lines.append("The ownership table disagrees with session routing on most trades.")
        lines.append("This may indicate the ownership table is too noisy or unstable.")
        lines.append("Recommendation: Investigate table quality before proceeding.")
    else:
        lines.append(f"FINDING: Agreement rate is in the actionable range ({avg_agree:.0f}%).")

    lines.append("")

    # Disagreement quality assessment
    pips_diffs = []
    for dk, r in results.items():
        agree_avg = r["agreement_stats"]["avg_pips"]
        disagree_avg = r["disagreement_stats"]["avg_pips"]
        diff = agree_avg - disagree_avg
        pips_diffs.append(diff)
        lines.append(
            f"{dk}: Agreement avg={agree_avg:+.2f}p, "
            f"Disagreement avg={disagree_avg:+.2f}p, "
            f"Δ={diff:+.2f}p"
        )

    lines.append("")
    avg_diff = sum(pips_diffs) / len(pips_diffs)

    if all(d >= 3.0 for d in pips_diffs):
        lines.append(f"FINDING: Disagreement trades are MATERIALLY WORSE "
                      f"(Δ ≥ 3p on both datasets, avg Δ = {avg_diff:.1f}p).")
        lines.append("The chart-first table correctly identifies the session router's worst allocations.")
    elif all(d >= 1.0 for d in pips_diffs):
        lines.append(f"FINDING: Disagreement trades are SOMEWHAT WORSE "
                      f"(Δ ≥ 1p on both datasets, avg Δ = {avg_diff:.1f}p).")
        lines.append("The chart-first table identifies worse allocations, but the signal is moderate.")
    elif any(d < 0 for d in pips_diffs):
        lines.append(f"FINDING: On at least one dataset, disagreement trades are NOT worse.")
        lines.append("The chart-first table does not reliably identify bad allocations.")
        lines.append("Recommendation: Chart-first routing does NOT pass Phase A.")
    else:
        lines.append(f"FINDING: Disagreement trades are slightly worse (avg Δ = {avg_diff:.1f}p).")
        lines.append("Signal exists but is weak.")

    lines.append("")

    # Cross-dataset consistency
    disagree_net_500k = results.get("500k", {}).get("disagreement_stats", {}).get("net_pips", 0)
    disagree_net_1000k = results.get("1000k", {}).get("disagreement_stats", {}).get("net_pips", 0)
    both_negative = disagree_net_500k < 0 and disagree_net_1000k < 0

    if both_negative:
        lines.append("CONSISTENCY: Disagreement net pips are negative on BOTH datasets.")
        lines.append("This is directionally consistent — disagreements are losers.")
    else:
        lines.append(f"CONSISTENCY: Disagreement net pips: 500k={disagree_net_500k:+.1f}p, "
                      f"1000k={disagree_net_1000k:+.1f}p.")
        if not both_negative:
            lines.append("Not consistently negative — caution warranted.")

    lines.append("")

    # Final recommendation
    passes_agree_rate = avg_agree < 90
    passes_pip_diff = all(d >= 3.0 for d in pips_diffs)
    passes_consistency = both_negative

    if passes_agree_rate and passes_pip_diff and passes_consistency:
        lines.append("VERDICT: Phase A PASSES all criteria.")
        lines.append("Chart-first routing identifies meaningful misallocation in the session router.")
        lines.append("Phase B (cross-session strategy evaluation backtest) is justified.")
    elif passes_agree_rate and (passes_pip_diff or passes_consistency):
        lines.append("VERDICT: Phase A shows PARTIAL signal.")
        lines.append("Chart-first routing has promise but does not pass all criteria cleanly.")
        lines.append("Recommended: targeted Phase B on specific disagreement pairs only.")
    else:
        lines.append("VERDICT: Phase A does NOT pass.")
        lines.append("Chart-first routing does not add enough value beyond current v1 vetoes.")
        lines.append("Recommended: Stay with v1 (session-first + defensive ownership).")

    return "\n".join(lines)


# ── Main ──

def main() -> None:
    print("Building conservative ownership table from stable cells...")
    table = load_conservative_table(research_out=OUT_DIR)
    print(f"  Stable owner cells: {sum(1 for v in table.values() if v['type'] == 'stable_owner')}")
    print(f"  Stable no-trade cells: {sum(1 for v in table.values() if v['type'] == 'stable_no_trade')}")
    print(f"  Total cells with table decision: {len(table)}")

    # Print table
    print(f"\n  Ownership table:")
    for cell, info in sorted(table.items()):
        if info["type"] == "stable_no_trade":
            print(f"    {cell:45s} → NO-TRADE")
        else:
            print(f"    {cell:45s} → {info['recommended_strategy']:12s} "
                  f"(avg 500k={info['owner_avg_500k']:+.1f}p, "
                  f"1000k={info['owner_avg_1000k']:+.1f}p)")

    results: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        dk = _dataset_key(dataset)
        results[dk] = _analyze_dataset(dataset, table)

    verdict_text = _verdict(results)

    # Print verdict
    print(f"\n{verdict_text}")

    # Build output
    output = {
        "ownership_table": {
            cell: {
                "recommended": info.get("recommended_strategy"),
                "type": info.get("type"),
                "owner_avg_500k": info.get("owner_avg_500k"),
                "owner_avg_1000k": info.get("owner_avg_1000k"),
            }
            for cell, info in table.items()
        },
        "dataset_summaries": {
            dk: {
                "total_trades": r["total_trades"],
                "agreement_count": r["agreement_count"],
                "disagreement_count": r["disagreement_count"],
                "no_table_decision_count": r["no_table_decision_count"],
                "agreement_rate_pct": r["agreement_rate_pct"],
            }
            for dk, r in results.items()
        },
        "agreement_stats": {
            dk: r["agreement_stats"] for dk, r in results.items()
        },
        "disagreement_stats": {
            dk: r["disagreement_stats"] for dk, r in results.items()
        },
        "no_table_decision_stats": {
            dk: r["no_table_decision_stats"] for dk, r in results.items()
        },
        "disagreement_pair_breakdown": {
            dk: r["disagreement_pair_breakdown"] for dk, r in results.items()
        },
        "cell_breakdown": {
            dk: r["cell_breakdown"] for dk, r in results.items()
        },
        "disagreement_samples": {
            dk: r["disagreement_samples"] for dk, r in results.items()
        },
        "coverage_gap_exploratory": {
            dk: r["coverage_gap_exploratory"] for dk, r in results.items()
        },
        "verdict": verdict_text,
    }

    out_path = OUT_DIR / "diagnostic_chart_first_routing.json"
    out_path.write_text(
        json.dumps(output, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(f"\n  Output: {out_path}")


if __name__ == "__main__":
    main()
