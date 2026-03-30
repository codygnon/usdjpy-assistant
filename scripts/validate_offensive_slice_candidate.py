#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path("/Users/codygnon/Documents/usdjpy_assistant")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.offensive_slice_spec import OffensiveSliceSpec
from scripts import backtest_offensive_slice_additive as additive
from scripts import run_offensive_slice_discovery as discovery

OUT_DIR = ROOT / "research_out"
DEFAULT_MATRIX = OUT_DIR / "offensive_slice_discovery_matrix.json"


def _load_matrix(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _slice_spec_from_matrix(matrix: dict[str, Any], slice_id: str) -> OffensiveSliceSpec:
    row = (matrix.get("results") or {}).get(slice_id)
    if not row:
        raise SystemExit(f"Slice id not found in matrix: {slice_id}")
    return OffensiveSliceSpec(**row["slice_spec"])


def _selected_trades_for_spec(
    spec: OffensiveSliceSpec,
    all_trades: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for dataset_key in ["500k", "1000k"]:
        trades = all_trades[dataset_key][spec.strategy]
        out[dataset_key] = [t for t in trades if discovery._passes_filters(t, spec)]
    return out


def _summarize_dataset(
    *,
    dataset_key: str,
    spec: OffensiveSliceSpec,
    selected_trades: list[dict[str, Any]],
    matrix: dict[str, Any],
    conflict_policy: additive.ConflictPolicy | None,
) -> dict[str, Any]:
    dataset = discovery.DATASETS[dataset_key]
    baseline_ctx = additive.build_baseline_context(dataset)
    coverage = discovery._load_variant_k_coverage(dataset)
    coverage_info = coverage.get(spec.ownership_cells[0], {
        "variant_k_trade_count": 0,
        "variant_k_by_strategy": {},
        "variant_k_net_pips": 0.0,
        "coverage_class": "whitespace",
    })
    per_scale = {}
    for scale_label, scale in [("50pct", 0.5), ("75pct", 0.75), ("100pct", 1.0)]:
        if conflict_policy is None:
            per_scale[scale_label] = additive.run_slice_additive(
                baseline_ctx=baseline_ctx,
                slice_spec=spec.as_dict(),
                selected_trades=selected_trades,
                size_scale=scale,
            )
        else:
            per_scale[scale_label] = additive.run_slice_additive_with_policy(
                baseline_ctx=baseline_ctx,
                slice_spec=spec.as_dict(),
                selected_trades=selected_trades,
                conflict_policy=conflict_policy,
                size_scale=scale,
            )
    expected = ((matrix.get("results") or {}).get(spec.resolved_slice_id()) or {}).get("per_dataset", {}).get(dataset_key)
    expected_100 = (((expected or {}).get("additive")) or {})
    observed_100 = per_scale["100pct"]
    regression = None
    if conflict_policy is None:
        regression = {
            "selected_trade_count_matches": bool(expected) and observed_100["selection_counts"]["selected_trade_count"] == expected_100["selection_counts"]["selected_trade_count"],
            "exact_overlap_matches": bool(expected) and observed_100["selection_counts"]["exact_baseline_overlap_count"] == expected_100["selection_counts"]["exact_baseline_overlap_count"],
            "new_additive_matches": bool(expected) and observed_100["selection_counts"]["new_additive_trades_count"] == expected_100["selection_counts"]["new_additive_trades_count"],
            "delta_net_usd_matches": bool(expected) and abs(observed_100["delta_vs_baseline"]["net_usd"] - expected_100["delta_vs_baseline"]["net_usd"]) < 0.01,
            "delta_pf_matches": bool(expected) and abs(observed_100["delta_vs_baseline"]["profit_factor"] - expected_100["delta_vs_baseline"]["profit_factor"]) < 1e-4,
            "delta_dd_matches": bool(expected) and abs(observed_100["delta_vs_baseline"]["max_drawdown_usd"] - expected_100["delta_vs_baseline"]["max_drawdown_usd"]) < 0.01,
        }
    return {
        "dataset": dataset_key,
        "source_dataset": dataset,
        "coverage": coverage_info,
        "standalone": discovery._metrics(selected_trades),
        "size_sweep": per_scale,
        "regression_vs_matrix_100pct": regression,
    }


def _count_key(selection_counts: dict[str, Any], preferred: str, fallback: str) -> Any:
    if preferred in selection_counts:
        return selection_counts[preferred]
    return selection_counts.get(fallback)


def _build_markdown(
    slice_id: str,
    spec: OffensiveSliceSpec,
    results: dict[str, Any],
    policy_name: str,
) -> str:
    lines = [
        "# Offensive Slice Validation",
        "",
        f"- slice_id: `{slice_id}`",
        f"- strategy: `{spec.strategy}`",
        f"- direction: `{spec.direction}`",
        f"- ownership cell(s): `{', '.join(spec.ownership_cells)}`",
        f"- source mode: `{spec.source_mode}`",
        f"- state realism mode: `{spec.state_realism_mode}`",
        f"- conflict policy: `{policy_name}`",
        "",
    ]
    for dataset_key in ["500k", "1000k"]:
        ds = results["datasets"][dataset_key]
        lines += [
            f"## {dataset_key}",
            "",
            f"- coverage: `{ds['coverage'].get('coverage_class', 'unknown')}`",
            f"- Variant K trades in cell: `{ds['coverage'].get('variant_k_trade_count', 0)}`",
            f"- standalone trades: `{ds['standalone']['trade_count']}`",
            f"- standalone avg pips: `{ds['standalone']['avg_pips']}`",
            f"- standalone PF: `{ds['standalone']['profit_factor']}`",
            "",
        ]
        for scale_label in ["50pct", "75pct", "100pct"]:
            scale = ds["size_sweep"][scale_label]
            delta = scale["delta_vs_baseline"]
            counts = scale["selection_counts"]
            selected_count = _count_key(counts, "margin_selected_trade_count", "selected_trade_count")
            lines.append(
                f"- {scale_label}: selected `{selected_count}`, overlap `{counts['exact_baseline_overlap_count']}`, "
                f"additive `{counts['new_additive_trades_count']}`, delta USD `{delta['net_usd']}`, "
                f"delta PF `{delta['profit_factor']}`, delta DD `{delta['max_drawdown_usd']}`"
            )
        reg = ds["regression_vs_matrix_100pct"]
        if reg is not None:
            lines += [
                "",
                f"- 100pct regression vs matrix: `{all(reg.values())}`",
                f"- regression detail: `{json.dumps(reg, sort_keys=True)}`",
                "",
            ]
        else:
            stats = ds["size_sweep"]["100pct"].get("policy_stats") or {}
            if stats:
                lines += [
                    "",
                    f"- policy stats @100pct: `{json.dumps(stats, sort_keys=True)}`",
                    "",
                ]
    return "\n".join(lines) + "\n"


def _json_default(obj: Any) -> Any:
    return discovery._json_default(obj)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice-id", required=True)
    ap.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    ap.add_argument("--output-json", default="")
    ap.add_argument("--output-md", default="")
    ap.add_argument("--policy", default="none", choices=["none", "native_v44_hedging_like"])
    args = ap.parse_args()

    matrix_path = Path(args.matrix).resolve()
    matrix = _load_matrix(matrix_path)
    spec = _slice_spec_from_matrix(matrix, args.slice_id)
    conflict_policy = None
    if args.policy == "native_v44_hedging_like":
        conflict_policy = additive.ConflictPolicy(
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
    all_trades = discovery._load_all_normalized_trades({spec.strategy})
    selected_by_dataset = _selected_trades_for_spec(spec, all_trades)

    out = {
        "title": "Offensive Slice Exact Validation",
        "slice_id": args.slice_id,
        "slice_spec": spec.as_dict(),
        "policy": args.policy,
        "datasets": {
            dataset_key: _summarize_dataset(
                dataset_key=dataset_key,
                spec=spec,
                selected_trades=selected_by_dataset[dataset_key],
                matrix=matrix,
                conflict_policy=conflict_policy,
            )
            for dataset_key in ["500k", "1000k"]
        },
    }

    output_json = Path(args.output_json) if args.output_json else OUT_DIR / f"{args.slice_id}_validation.json"
    output_md = Path(args.output_md) if args.output_md else OUT_DIR / f"{args.slice_id}_validation.md"
    output_json.write_text(json.dumps(out, indent=2, default=_json_default), encoding="utf-8")
    output_md.write_text(_build_markdown(args.slice_id, spec, out, args.policy), encoding="utf-8")
    print(output_json)
    print(output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
