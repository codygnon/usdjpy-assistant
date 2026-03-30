#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path("/Users/codygnon/Documents/usdjpy_assistant")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.offensive_slice_spec import OffensiveSliceSpec
from scripts import backtest_offensive_slice_additive as additive
from scripts import backtest_offensive_slice_family_combo as family_combo
from scripts import backtest_variant_k_plus_v44_family_v3 as v3
from scripts import backtest_variant_k_v5_search as v5
from scripts import run_offensive_slice_discovery as discovery

OUT_DIR = ROOT / "research_out"
DEFAULT_OUTPUT = OUT_DIR / "v44_family_native_hedging_policy_rerun.json"
DEFAULT_MD = OUT_DIR / "v44_family_native_hedging_policy_rerun.md"
V4_SEARCH_ARTIFACT = OUT_DIR / "system_variant_k_plus_v44_family_v4_search.json"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return discovery._json_default(obj)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_v4_best_specs(matrix: dict[str, Any]) -> list[OffensiveSliceSpec]:
    artifact = _load_json(V4_SEARCH_ARTIFACT)
    slice_ids = list(artifact["best_variant"]["slice_ids"])
    return [v3._load_specs(matrix)[0] if False else OffensiveSliceSpec(**matrix["results"][sid]["slice_spec"]) for sid in slice_ids]


def _select_for_specs(
    specs: list[OffensiveSliceSpec],
    all_trades: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, list[dict[str, Any]]]:
    return family_combo._dedupe_selected_trades(specs, all_trades)


def _load_v3_selected(
    matrix: dict[str, Any],
    all_trades: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, list[dict[str, Any]]]:
    specs = v3._load_specs(matrix)
    per_cell = v3._select_trades_per_cell(specs, all_trades)
    variant = next(row for row in v3._build_variants() if row["name"] == "full_v3")
    out: dict[str, list[dict[str, Any]]] = {"500k": [], "1000k": []}
    for dataset_key in ["500k", "1000k"]:
        combined: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str, str]] = set()
        for cell_idx in variant["active_cells"]:
            scale = variant["scales"].get(cell_idx, 1.0)
            for trade in per_cell[dataset_key][cell_idx]:
                key = (str(trade["strategy"]), str(trade["entry_time"]), str(trade["exit_time"]), str(trade["side"]))
                if key in seen:
                    continue
                seen.add(key)
                if scale != 1.0:
                    t = dict(trade)
                    t["usd"] = float(t["usd"]) * scale
                    t["size_scale"] = float(t.get("size_scale", 1.0)) * scale
                    combined.append(t)
                else:
                    combined.append(trade)
        combined.sort(key=lambda t: (t["entry_time"], t["exit_time"]))
        out[dataset_key] = combined
    return out


def _load_v5_selected(
    matrix: dict[str, Any],
    all_trades: dict[str, dict[str, list[dict[str, Any]]]],
    variant_name: str,
) -> dict[str, list[dict[str, Any]]]:
    specs_all = v5._load_all_specs(matrix)
    trades_by_ds = v5._select_all_trades(specs_all, all_trades)
    variant = next(row for row in v5._build_variants() if row["name"] == variant_name)
    out: dict[str, list[dict[str, Any]]] = {"500k": [], "1000k": []}
    for dataset_key in ["500k", "1000k"]:
        combined: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str, str]] = set()
        scales = variant.get("scales") or {}
        for label in variant["cells"]:
            scale = scales.get(label, 1.0)
            for trade in trades_by_ds[dataset_key].get(label, []):
                key = (str(trade["strategy"]), str(trade["entry_time"]), str(trade["exit_time"]), str(trade["side"]))
                if key in seen:
                    continue
                seen.add(key)
                if scale != 1.0:
                    t = dict(trade)
                    t["usd"] = float(t["usd"]) * scale
                    t["size_scale"] = float(t.get("size_scale", 1.0)) * scale
                    combined.append(t)
                else:
                    combined.append(trade)
        combined.sort(key=lambda t: (t["entry_time"], t["exit_time"]))
        out[dataset_key] = combined
    return out


def _run_variant(
    *,
    name: str,
    selected: dict[str, list[dict[str, Any]]],
    policy: additive.ConflictPolicy,
) -> dict[str, Any]:
    out = {
        "name": name,
        "datasets": {},
    }
    for dataset_key in ["500k", "1000k"]:
        baseline_ctx = additive.build_baseline_context(discovery.DATASETS[dataset_key])
        result = additive.run_slice_additive_with_policy(
            baseline_ctx=baseline_ctx,
            slice_spec={"variant_name": name},
            selected_trades=selected[dataset_key],
            conflict_policy=policy,
            size_scale=1.0,
        )
        out["datasets"][dataset_key] = {
            "delta_vs_baseline": result["delta_vs_baseline"],
            "variant_summary": result["variant_summary"],
            "selection_counts": result["selection_counts"],
            "policy_stats": result["policy_stats"],
        }
    return out


def _score(row: dict[str, Any]) -> tuple[float, float]:
    d500 = row["datasets"]["500k"]["delta_vs_baseline"]
    d1k = row["datasets"]["1000k"]["delta_vs_baseline"]
    return (float(d500["net_usd"]) + float(d1k["net_usd"]), float(d500["profit_factor"]) + float(d1k["profit_factor"]))


def _build_md(payload: dict[str, Any]) -> str:
    lines = [
        "# V44 Family Rerun Under Native Hedging Policy",
        "",
        f"- policy: `{payload['policy']['name']}`",
        f"- hedging enabled: `{payload['policy']['hedging_enabled']}`",
        f"- allow opposite-side coexistence: `{payload['policy']['allow_opposite_side_overlap']}`",
        f"- allow internal overlap: `{payload['policy']['allow_internal_overlap']}`",
        f"- max open offensive: `{payload['policy']['max_open_offensive']}`",
        f"- max entries per day: `{payload['policy']['max_entries_per_day']}`",
        f"- margin model enabled: `{payload['policy']['margin_model_enabled']}`",
        f"- leverage: `{payload['policy']['margin_leverage']}`",
        f"- margin buffer pct: `{payload['policy']['margin_buffer_pct']}`",
        f"- max lot per trade: `{payload['policy']['max_lot_per_trade']}`",
        "",
    ]
    for row in payload["leaderboard"]:
        d500 = row["datasets"]["500k"]["delta_vs_baseline"]
        d1k = row["datasets"]["1000k"]["delta_vs_baseline"]
        s500 = row["datasets"]["500k"]["variant_summary"]
        s1k = row["datasets"]["1000k"]["variant_summary"]
        lines += [
            f"## {row['name']}",
            "",
            f"- combined delta USD: `{round(d500['net_usd'] + d1k['net_usd'], 2)}`",
            f"- combined delta PF: `{round(d500['profit_factor'] + d1k['profit_factor'], 4)}`",
            "",
            f"### 500k",
            f"- total trades: `{s500['total_trades']}`",
            f"- net USD: `{round(s500['net_usd'], 2)}`",
            f"- PF: `{round(s500['profit_factor'], 4)}`",
            f"- max DD: `{round(s500['max_drawdown_usd'], 2)}`",
            f"- raw selected: `{row['datasets']['500k']['selection_counts']['raw_selected_trade_count']}`",
            f"- policy selected: `{row['datasets']['500k']['selection_counts']['policy_selected_trade_count']}`",
            f"- margin selected: `{row['datasets']['500k']['selection_counts']['margin_selected_trade_count']}`",
            f"- baseline exact overlap: `{row['datasets']['500k']['selection_counts']['exact_baseline_overlap_count']}`",
            f"- internal overlap pairs: `{row['datasets']['500k']['selection_counts']['internal_overlap_pairs']}`",
            f"- internal opposite-side pairs: `{row['datasets']['500k']['selection_counts']['internal_opposite_side_overlap_pairs']}`",
            f"- lot-cap blocked: `{row['datasets']['500k']['policy_stats']['lot_cap_blocked']}`",
            f"- margin rejected: `{row['datasets']['500k']['policy_stats']['margin_rejected']}`",
            f"- margin-call-state blocks: `{row['datasets']['500k']['policy_stats']['margin_call_state_blocks']}`",
            "",
            f"### 1000k",
            f"- total trades: `{s1k['total_trades']}`",
            f"- net USD: `{round(s1k['net_usd'], 2)}`",
            f"- PF: `{round(s1k['profit_factor'], 4)}`",
            f"- max DD: `{round(s1k['max_drawdown_usd'], 2)}`",
            f"- raw selected: `{row['datasets']['1000k']['selection_counts']['raw_selected_trade_count']}`",
            f"- policy selected: `{row['datasets']['1000k']['selection_counts']['policy_selected_trade_count']}`",
            f"- margin selected: `{row['datasets']['1000k']['selection_counts']['margin_selected_trade_count']}`",
            f"- baseline exact overlap: `{row['datasets']['1000k']['selection_counts']['exact_baseline_overlap_count']}`",
            f"- internal overlap pairs: `{row['datasets']['1000k']['selection_counts']['internal_overlap_pairs']}`",
            f"- internal opposite-side pairs: `{row['datasets']['1000k']['selection_counts']['internal_opposite_side_overlap_pairs']}`",
            f"- lot-cap blocked: `{row['datasets']['1000k']['policy_stats']['lot_cap_blocked']}`",
            f"- margin rejected: `{row['datasets']['1000k']['policy_stats']['margin_rejected']}`",
            f"- margin-call-state blocks: `{row['datasets']['1000k']['policy_stats']['margin_call_state_blocks']}`",
            "",
        ]
    return "\n".join(lines) + "\n"


def main() -> int:
    matrix = family_combo._load_matrix(family_combo.DEFAULT_MATRIX)
    all_trades = discovery._load_all_normalized_trades({"v44_ny"})
    policy = additive.ConflictPolicy(
        name="native_v44_hedging_like",
        hedging_enabled=True,
        allow_internal_overlap=True,
        allow_opposite_side_overlap=True,
        max_open_offensive=None,
        max_entries_per_day=None,
        margin_model_enabled=True,
        margin_leverage=33.3,
        margin_buffer_pct=0.0,
        max_lot_per_trade=20.0,
    )

    v3_selected = _load_v3_selected(matrix, all_trades)
    v4_selected = _select_for_specs(_load_v4_best_specs(matrix), all_trades)
    v5_best_selected = _load_v5_selected(matrix, all_trades, "v4_plus_all_adj")
    v5_candidate_a_selected = _load_v5_selected(matrix, all_trades, "v5_candidate_A")

    leaderboard = [
        _run_variant(name="v3_full_v3", selected=v3_selected, policy=policy),
        _run_variant(name="v4_best", selected=v4_selected, policy=policy),
        _run_variant(name="v5_best_current_framework", selected=v5_best_selected, policy=policy),
        _run_variant(name="v5_candidate_A", selected=v5_candidate_a_selected, policy=policy),
    ]
    leaderboard.sort(key=_score, reverse=True)

    payload = {
        "title": "V44 family rerun under native hedging policy",
        "policy": {
            "name": policy.name,
            "hedging_enabled": policy.hedging_enabled,
            "allow_internal_overlap": policy.allow_internal_overlap,
            "allow_opposite_side_overlap": policy.allow_opposite_side_overlap,
            "max_open_offensive": policy.max_open_offensive,
            "max_entries_per_day": policy.max_entries_per_day,
            "margin_model_enabled": policy.margin_model_enabled,
            "margin_leverage": policy.margin_leverage,
            "margin_buffer_pct": policy.margin_buffer_pct,
            "max_lot_per_trade": policy.max_lot_per_trade,
        },
        "leaderboard": leaderboard,
    }

    DEFAULT_OUTPUT.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    DEFAULT_MD.write_text(_build_md(payload), encoding="utf-8")
    print(DEFAULT_OUTPUT)
    print(DEFAULT_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
