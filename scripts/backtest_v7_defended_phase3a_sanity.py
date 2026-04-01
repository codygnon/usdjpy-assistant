#!/usr/bin/env python3
"""
Phase 3A quick sanity: reproduce v7_pfdd defended combined delta vs $222,651.96.

Same data path as evaluate_defensive_on_freeze_leader.main / package_freeze_closeout_lib.load_context().
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import backtest_offensive_slice_additive as additive
from scripts import evaluate_defensive_on_freeze_leader as freeze_leader
from scripts import package_freeze_closeout_lib as lib
from scripts import run_offensive_slice_discovery as discovery

OUT_JSON = ROOT / "research_out" / "v7_defended_phase3a_sanity.json"
TARGET_USD = 222_651.96
PACKAGE = "v7_pfdd"
TARGET_LABEL = freeze_leader.TARGET_LABEL


def _count_by_session(trades: list[dict[str, Any]]) -> dict[str, int]:
    c = {"tokyo_v14": 0, "london_v2_setup_a": 0, "london_v2_setup_d_l1": 0, "v44_ny": 0}
    for t in trades:
        s = str(t.get("strategy", ""))
        if s == "v14":
            c["tokyo_v14"] += 1
        elif s == "v44_ny":
            c["v44_ny"] += 1
        elif s == "london_v2":
            st = str(t.get("setup_type", ""))
            if st == "A":
                c["london_v2_setup_a"] += 1
            elif st == "D":
                c["london_v2_setup_d_l1"] += 1
            else:
                c["london_v2_setup_d_l1"] += 1
    return c


def _trade_row_dict(t: dict[str, Any]) -> dict[str, Any]:
    return {
        "entry_time": str(t.get("entry_time", "")),
        "side": str(t.get("side", "")),
        "pnl_usd": round(float(t.get("usd", 0.0)), 4),
        "strategy": str(t.get("strategy", "")),
        "setup_type": str(t.get("setup_type", "")),
    }


def _collect_margin_selected(
    combined: list[dict[str, Any]],
    defensive_ctx: additive.BaselineContext,
    policy: additive.ConflictPolicy,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pol_sel, pol_st = additive._apply_conflict_policy_to_selected_trades(combined, policy)
    baseline_id = {additive._trade_identity_from_trade_row(t) for t in defensive_ctx.baseline_coupled}
    add_cand = [t for t in pol_sel if not additive._has_exact_baseline_match_by_identity(t, baseline_id)]
    m_sel, m_st = additive._apply_margin_policy_to_candidates(
        additive_candidates=add_cand,
        baseline_trades=defensive_ctx.baseline_coupled,
        starting_equity=additive.STARTING_EQUITY,
        policy=policy,
    )
    return m_sel, {**pol_st, **m_st}


def main() -> int:
    context = lib.load_context()
    scales = lib.package_scales(PACKAGE)

    followup = {ds: freeze_leader._build_followup_replacement(context, ds) for ds in ("500k", "1000k")}
    defensive_ctx: dict[str, additive.BaselineContext] = {}
    defensive_meta: dict[str, Any] = {}
    for ds in ("500k", "1000k"):
        c, m = freeze_leader._build_defensive_baseline_ctx(discovery.DATASETS[ds])
        defensive_ctx[ds] = c
        defensive_meta[ds] = m

    weekday_drop = 0
    for ds in ("500k", "1000k"):
        for t in context["trades_by_ds"][ds][TARGET_LABEL]:
            if not freeze_leader._weekday_ok(t):
                weekday_drop += 1

    before_total = 0
    conflict_tot = margin_tot = 0
    per_ds: dict[str, Any] = {}

    for ds in ("500k", "1000k"):
        tbl = dict(context["trades_by_ds"][ds])
        tbl[TARGET_LABEL] = followup[ds]
        combined = lib.scaled_combined_trades(scales, tbl)
        before_total += len(combined)

        m_sel, merged_st = _collect_margin_selected(combined, defensive_ctx[ds], context["policy"])
        conflict_tot += int(
            merged_st.get("overlap_blocked", 0)
            + merged_st.get("max_open_blocked", 0)
            + merged_st.get("max_entries_day_blocked", 0)
            + merged_st.get("opposite_side_blocked", 0)
            + merged_st.get("exact_duplicate_blocked", 0)
        )
        margin_tot += int(merged_st.get("margin_rejected", 0) + merged_st.get("margin_call_state_blocks", 0))

    pkg = freeze_leader._package_result(context, PACKAGE, defensive_ctx, followup)
    for ds in ("500k", "1000k"):
        pkg_ds = pkg["datasets"][ds]
        per_ds[ds] = {
            "delta_net_usd": pkg_ds["delta_vs_baseline"]["net_usd"],
            "selection_counts": pkg_ds["selection_counts"],
            "policy_stats": pkg_ds.get("policy_stats", {}),
            "defensive_blocked": defensive_meta[ds],
        }
    reproduced = float(pkg["combined_delta_usd"])
    diff = abs(reproduced - TARGET_USD)
    diff_pct = (diff / TARGET_USD * 100.0) if TARGET_USD else 0.0
    ok = diff <= 500.0

    after_policy = sum(int(pkg["datasets"][ds]["selection_counts"]["policy_selected_trade_count"]) for ds in ("500k", "1000k"))
    after_margin = sum(int(pkg["datasets"][ds]["selection_counts"]["margin_selected_trade_count"]) for ds in ("500k", "1000k"))
    veto_total = sum(int(defensive_meta[ds]["blocked_count"]) for ds in defensive_meta)

    by_sess = {"tokyo_v14": 0, "london_v2_setup_a": 0, "london_v2_setup_d_l1": 0, "v44_ny": 0}
    trade_lists: dict[str, list[dict[str, Any]]] = {k: [] for k in by_sess}
    for ds in ("500k", "1000k"):
        tbl = dict(context["trades_by_ds"][ds])
        tbl[TARGET_LABEL] = followup[ds]
        combined = lib.scaled_combined_trades(scales, tbl)
        m_sel, _ = _collect_margin_selected(combined, defensive_ctx[ds], context["policy"])
        for t in m_sel:
            sess = _count_by_session([t])
            for k, v in sess.items():
                by_sess[k] += v
            sk = str(t.get("strategy", ""))
            if sk == "v14":
                trade_lists["tokyo_v14"].append(_trade_row_dict(t))
            elif sk == "v44_ny":
                trade_lists["v44_ny"].append(_trade_row_dict(t))
            elif sk == "london_v2":
                st = str(t.get("setup_type", ""))
                key = "london_v2_setup_a" if st == "A" else "london_v2_setup_d_l1"
                trade_lists[key].append(_trade_row_dict(t))

    payload: dict[str, Any] = {
        "target_usd": TARGET_USD,
        "reproduced_usd": reproduced,
        "difference_usd": round(diff, 2),
        "difference_pct": round(diff_pct, 4),
        "sanity_passed": ok,
        "trade_counts": {
            "before_filters": before_total,
            "after_conflict_policy": after_policy,
            "after_margin_policy": after_margin,
            "tokyo_v14": by_sess["tokyo_v14"],
            "london_setup_a": by_sess["london_v2_setup_a"],
            "london_setup_d_l1": by_sess["london_v2_setup_d_l1"],
            "v44_ny": by_sess["v44_ny"],
        },
        "removed_by": {
            "weekday_block_l1_matrix": weekday_drop,
            "defensive_veto_v44_cell": veto_total,
            "variant_f_v44_blocked": "applied_inside_build_variant_k_pre_coupling_kept (see defensive baseline ctx)",
            "variant_k_cluster": "applied_inside_build_variant_k_pre_coupling_kept",
            "variant_i_standdown": "applied_inside_build_variant_k_pre_coupling_kept",
            "conflict_policy": conflict_tot,
            "margin": margin_tot,
        },
        "per_dataset": per_ds,
        "margin_selected_trades_by_session": trade_lists,
        "load_path": {
            "matrix_default": str(lib.DEFAULT_MATRIX),
            "datasets": {ds: str(discovery.DATASETS[ds]) for ds in ("500k", "1000k")},
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=== PHASE 3A SANITY CHECK ===")
    print(f"Target:               ${TARGET_USD:,.2f}")
    print(f"Reproduced:           ${reproduced:,.2f}")
    print(f"Difference:           ${diff:,.2f} ({diff_pct:.2f}%)")
    print()
    print("Trade counts:")
    print(f"  Before filters:     {before_total}")
    print(f"  After conflict:     {after_policy}")
    print(f"  After margin:       {after_margin}  (matches additive trade lists)")
    print(f"  Tokyo V14:          {by_sess['tokyo_v14']}")
    print(f"  London Setup A:     {by_sess['london_v2_setup_a']}")
    print(f"  London Setup D/L1:    {by_sess['london_v2_setup_d_l1']}")
    print(f"  V44 NY:             {by_sess['v44_ny']}")
    print()
    print("Removed by:")
    print(f"  Weekday block:      {weekday_drop}")
    print(f"  Defensive veto:     {veto_total}")
    print("  Variant F:          (inside defensive baseline / variant_k path; not split here)")
    print("  Variant K cluster:  (inside defensive baseline / variant_k path; not split here)")
    print("  Variant I:          (inside defensive baseline / variant_k path; not split here)")
    print(f"  Conflict policy:    {conflict_tot}")
    print(f"  Margin:             {margin_tot}")
    print()
    print(f"Wrote {OUT_JSON}")

    if not ok:
        print()
        print("DIAGNOSTIC: Reproduction differs from target by more than $500.")
        print("Compare per_dataset.delta_net_usd to expected slice totals; check policy_stats for margin/conflict.")
        print("For F/K/I trade-level counts, run scripts/backtest_variant_k_london_cluster.py or inspect defensive baseline build.")
        return 2

    print("SANITY CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
