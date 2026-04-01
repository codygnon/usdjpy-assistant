#!/usr/bin/env python3
"""
Phase 3A: Reproduce the defended v7_pfdd follow-up pipeline USD delta (target ~222,651.96).

Mirrors scripts/evaluate_defensive_on_freeze_leader.py:
  - lib.load_context() → offensive_slice_discovery_matrix + normalized trades
  - L1 replacement via run_package_slice_exit_frontier._build_followup_replacement pattern
  - Defensive baseline ctx (Variant K path + V44 cell veto)
  - additive.run_slice_additive_with_policy with lib.strict_policy()

Does not modify any existing modules; only imports them.
"""
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# package_freeze_closeout_lib.load_context — loads matrix, specs, trades_by_ds, baseline_ctx, ConflictPolicy
from scripts import backtest_offensive_slice_additive as additive
from scripts import backtest_defensive_v15_pocket_grid as def_pocket
from scripts import backtest_merged_integrated_tokyo_london_v2_ny as merged_engine
from scripts import backtest_variant_k_london_cluster as variant_k
from scripts import evaluate_defensive_on_freeze_leader as freeze_leader
from scripts import package_freeze_closeout_lib as lib
from scripts import run_offensive_slice_discovery as discovery
from scripts import run_package_slice_exit_frontier as exit_frontier

OUT_JSON = ROOT / "research_out" / "v7_defended_phase3a_reproduction.json"
TARGET_USD = 222651.96
PACKAGE_NAME = "v7_pfdd"
TARGET_LABEL = freeze_leader.TARGET_LABEL  # L1_mom_low_pos_buy


def _count_trades_by_session(trades: list[dict[str, Any]]) -> dict[str, int]:
    c = {
        "tokyo_v14": 0,
        "london_v2_setup_a": 0,
        "london_v2_setup_d_l1": 0,
        "v44_ny": 0,
    }
    for t in trades:
        s = str(t.get("strategy", ""))
        if s == "v14":
            c["tokyo_v14"] += 1
        elif s == "v44_ny":
            c["v44_ny"] += 1
        elif s == "london_v2":
            if str(t.get("setup_type", "")) == "D":
                c["london_v2_setup_d_l1"] += 1
            elif str(t.get("setup_type", "")) == "A":
                c["london_v2_setup_a"] += 1
            else:
                c["london_v2_setup_d_l1"] += 1  # default bucket for unknown
    return c


def _package_result(
    context: lib.Context,
    package_name: str,
    baseline_ctx_by_ds: dict[str, additive.BaselineContext],
    followup_replacement_by_ds: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Delegate to freeze_leader._package_result (single source of truth)."""
    out = freeze_leader._package_result(context, package_name, baseline_ctx_by_ds, followup_replacement_by_ds)
    scales = lib.package_scales(package_name)
    trades_by_ds: dict[str, list[dict[str, Any]]] = {}
    for ds in ["500k", "1000k"]:
        trades_by_label = dict(context["trades_by_ds"][ds])
        trades_by_label[TARGET_LABEL] = followup_replacement_by_ds[ds]
        trades_by_ds[ds] = lib.scaled_combined_trades(scales, trades_by_label)
    out["trades_by_ds"] = trades_by_ds
    return out


def main() -> int:
    notes: list[str] = []
    # Exact load path as evaluate_defensive_on_freeze_leader.main
    context = lib.load_context()
    notes.append(
        f"load_context: matrix={lib.DEFAULT_MATRIX.name}, "
        f"policy={context['policy'].name}, margin_leverage={context['policy'].margin_leverage}"
    )

    followup_replacement_by_ds = {
        ds: freeze_leader._build_followup_replacement(context, ds) for ds in ["500k", "1000k"]
    }

    defensive_ctx_by_ds: dict[str, additive.BaselineContext] = {}
    defensive_blocked: dict[str, Any] = {}
    for ds in ["500k", "1000k"]:
        ctx, meta = freeze_leader._build_defensive_baseline_ctx(discovery.DATASETS[ds])
        defensive_ctx_by_ds[ds] = ctx
        defensive_blocked[ds] = meta

    # Count L1 weekday drops: trades in matrix L1 list Mon/Tue excluded from replacement baseline
    weekday_dropped = 0
    for ds in ["500k", "1000k"]:
        for t in context["trades_by_ds"][ds][TARGET_LABEL]:
            if not freeze_leader._weekday_ok(t):
                weekday_dropped += 1

    scales = lib.package_scales(PACKAGE_NAME)
    raw_combined_counts = {"500k": 0, "1000k": 0}
    after_policy_counts = {"500k": 0, "1000k": 0}
    policy_margin_blocks = {"500k": 0, "1000k": 0}
    policy_conflict_blocks = {"500k": 0, "1000k": 0}

    for ds in ["500k", "1000k"]:
        trades_by_label = dict(context["trades_by_ds"][ds])
        trades_by_label[TARGET_LABEL] = followup_replacement_by_ds[ds]
        combined = lib.scaled_combined_trades(scales, trades_by_label)
        raw_combined_counts[ds] = len(combined)
        # Replay policy only to count blocks (same as inside run_slice_additive_with_policy)
        pol_sel, pol_st = additive._apply_conflict_policy_to_selected_trades(combined, context["policy"])
        after_policy_counts[ds] = len(pol_sel)
        policy_conflict_blocks[ds] = int(pol_st.get("overlap_blocked", 0))
        policy_conflict_blocks[ds] += int(pol_st.get("max_open_blocked", 0))
        policy_conflict_blocks[ds] += int(pol_st.get("max_entries_day_blocked", 0))
        policy_conflict_blocks[ds] += int(pol_st.get("opposite_side_blocked", 0))
        policy_conflict_blocks[ds] += int(pol_st.get("exact_duplicate_blocked", 0))
        baseline_id = {additive._trade_identity_from_trade_row(t) for t in defensive_ctx_by_ds[ds].baseline_coupled}
        add_cand = [t for t in pol_sel if not additive._has_exact_baseline_match_by_identity(t, baseline_id)]
        _m_sel, m_st = additive._apply_margin_policy_to_candidates(
            additive_candidates=add_cand,
            baseline_trades=defensive_ctx_by_ds[ds].baseline_coupled,
            starting_equity=additive.STARTING_EQUITY,
            policy=context["policy"],
        )
        policy_margin_blocks[ds] = int(m_st.get("margin_rejected", 0)) + int(m_st.get("margin_call_state_blocks", 0))

    pkg = _package_result(context, PACKAGE_NAME, defensive_ctx_by_ds, followup_replacement_by_ds)
    reproduced = float(pkg["combined_delta_usd"])
    diff = abs(reproduced - TARGET_USD)
    diff_pct = (diff / TARGET_USD * 100.0) if TARGET_USD else 0.0
    match = diff <= 500.0

    # Variant F / K / I counts: approximate from variant_k build (not re-run full admission on dicts)
    # Defensive veto = sum blocked_count from defensive_blocked
    veto_total = sum(int(defensive_blocked[ds]["blocked_count"]) for ds in defensive_blocked)

    total_before = sum(raw_combined_counts.values())
    total_after = sum(
        int(pkg["datasets"][ds]["selection_counts"]["policy_selected_trade_count"]) for ds in ["500k", "1000k"]
    )

    trades_by_session_combined: dict[str, int] = {
        "tokyo_v14": 0,
        "london_v2_setup_a": 0,
        "london_v2_setup_d_l1": 0,
        "v44_ny": 0,
    }
    for ds in ["500k", "1000k"]:
        for k, v in _count_trades_by_session(pkg["trades_by_ds"][ds]).items():
            trades_by_session_combined[k] += v

    payload = {
        "phase": "3A",
        "purpose": "reproduce pipeline result",
        "target_usd": TARGET_USD,
        "reproduced_usd": reproduced,
        "difference_usd": round(diff, 2),
        "difference_pct": round(diff_pct, 4),
        "match": match,
        "total_trades_before_filters": total_before,
        "total_trades_after_filters": total_after,
        "trades_by_session": trades_by_session_combined,
        "trades_removed_by": {
            "weekday_block": weekday_dropped,
            "defensive_veto": veto_total,
            "variant_f": "embedded_in_variant_k_pre_coupling_kept",
            "variant_k_cluster": "embedded_in_variant_k_pre_coupling_kept",
            "variant_i_standdown": "embedded_in_variant_k_pre_coupling_kept",
            "conflict_policy": sum(policy_conflict_blocks.values()),
            "margin": sum(policy_margin_blocks.values()),
        },
        "per_dataset": {
            ds: {
                "delta_net_usd": pkg["datasets"][ds]["delta_vs_baseline"]["net_usd"],
                "selection_counts": pkg["datasets"][ds]["selection_counts"],
                "policy_stats": pkg["datasets"][ds].get("policy_stats", {}),
                "defensive_blocked": defensive_blocked.get(ds, {}),
            }
            for ds in ["500k", "1000k"]
        },
        "notes": " | ".join(notes)
        + " | Variant F/K/I are applied inside variant_k.build_variant_k_pre_coupling_kept when building defensive baseline ctx, not as a separate pass on dict trades.",
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": str(OUT_JSON), "match": match, "reproduced_usd": reproduced}, indent=2))
    if not match:
        print(
            "STOP: reproduction differs from target by more than $500. "
            "Inspect per_dataset deltas and policy_stats in the JSON.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
