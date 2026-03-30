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


def _specs_from_ids(matrix: dict[str, Any], slice_ids: list[str]) -> list[OffensiveSliceSpec]:
    out = []
    for slice_id in slice_ids:
        row = (matrix.get("results") or {}).get(slice_id)
        if not row:
            raise SystemExit(f"Slice id not found in matrix: {slice_id}")
        out.append(OffensiveSliceSpec(**row["slice_spec"]))
    return out


def _trade_key(trade: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(trade["strategy"]),
        str(trade["entry_time"]),
        str(trade["exit_time"]),
        str(trade["side"]),
    )


def _dedupe_selected_trades(
    specs: list[OffensiveSliceSpec],
    all_trades: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {"500k": [], "1000k": []}
    for dataset_key in ["500k", "1000k"]:
        seen: set[tuple[str, str, str, str]] = set()
        combined: list[dict[str, Any]] = []
        for spec in specs:
            for trade in all_trades[dataset_key][spec.strategy]:
                if not discovery._passes_filters(trade, spec):
                    continue
                key = _trade_key(trade)
                if key in seen:
                    continue
                seen.add(key)
                combined.append(trade)
        combined.sort(key=lambda t: (t["entry_time"], t["exit_time"], t["strategy"], t["side"]))
        out[dataset_key] = combined
    return out


def _metrics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    return discovery._metrics(trades)


def _coverage_summary(dataset: str, specs: list[OffensiveSliceSpec]) -> list[dict[str, Any]]:
    coverage = discovery._load_variant_k_coverage(dataset)
    out = []
    for spec in specs:
        cell = spec.ownership_cells[0] if spec.ownership_cells else "multiple"
        out.append({
            "slice_id": spec.resolved_slice_id(),
            "cell": cell,
            **coverage.get(cell, {
                "variant_k_trade_count": 0,
                "variant_k_by_strategy": {},
                "variant_k_net_pips": 0.0,
                "coverage_class": "whitespace",
            }),
        })
    return out


def _summarize_variant(
    *,
    name: str,
    specs: list[OffensiveSliceSpec],
    all_trades: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, Any]:
    selected = _dedupe_selected_trades(specs, all_trades)
    out = {
        "name": name,
        "slice_ids": [s.resolved_slice_id() for s in specs],
        "datasets": {},
    }
    for dataset_key in ["500k", "1000k"]:
        dataset = discovery.DATASETS[dataset_key]
        baseline_ctx = additive.build_baseline_context(dataset)
        size_sweep = {}
        for scale_label, scale in [("50pct", 0.5), ("75pct", 0.75), ("100pct", 1.0)]:
            size_sweep[scale_label] = additive.run_slice_additive(
                baseline_ctx=baseline_ctx,
                slice_spec={"variant_name": name, "slice_ids": out["slice_ids"]},
                selected_trades=selected[dataset_key],
                size_scale=scale,
            )
        out["datasets"][dataset_key] = {
            "coverage_by_cell": _coverage_summary(dataset, specs),
            "standalone": _metrics(selected[dataset_key]),
            "deduped_selected_trade_count": len(selected[dataset_key]),
            "size_sweep": size_sweep,
        }
    return out


def _build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Offensive Slice Family Combo Backtest",
        "",
        f"- family: `{report['family_name']}`",
        "",
    ]
    for variant in report["variants"]:
        lines += [
            f"## {variant['name']}",
            "",
            f"- slice ids: `{', '.join(variant['slice_ids'])}`",
            "",
        ]
        for dataset_key in ["500k", "1000k"]:
            ds = variant["datasets"][dataset_key]
            lines += [
                f"### {dataset_key}",
                "",
                f"- deduped selected trades: `{ds['deduped_selected_trade_count']}`",
                f"- standalone trades: `{ds['standalone']['trade_count']}`",
                f"- standalone avg pips: `{ds['standalone']['avg_pips']}`",
                f"- standalone PF: `{ds['standalone']['profit_factor']}`",
            ]
            for scale_label in ["50pct", "75pct", "100pct"]:
                scale = ds["size_sweep"][scale_label]
                delta = scale["delta_vs_baseline"]
                counts = scale["selection_counts"]
                lines.append(
                    f"- {scale_label}: selected `{counts['selected_trade_count']}`, overlap `{counts['exact_baseline_overlap_count']}`, "
                    f"additive `{counts['new_additive_trades_count']}`, displaced `{counts['displaced_trades_count']}`, "
                    f"delta USD `{delta['net_usd']}`, delta PF `{delta['profit_factor']}`, delta DD `{delta['max_drawdown_usd']}`"
                )
            lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family-name", required=True)
    ap.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    ap.add_argument("--slice-id", action="append", required=True, dest="slice_ids")
    ap.add_argument("--output-json", default="")
    ap.add_argument("--output-md", default="")
    args = ap.parse_args()

    matrix = _load_matrix(Path(args.matrix).resolve())
    specs = _specs_from_ids(matrix, args.slice_ids)
    strategies = {spec.strategy for spec in specs}
    all_trades = discovery._load_all_normalized_trades(strategies)

    variants = []
    for i in range(1, len(specs) + 1):
        variants.append(_summarize_variant(
            name=f"top{i}",
            specs=specs[:i],
            all_trades=all_trades,
        ))

    report = {
        "title": "Offensive Slice Family Combo Backtest",
        "family_name": args.family_name,
        "slice_ids_in_order": [s.resolved_slice_id() for s in specs],
        "variants": variants,
    }

    output_json = Path(args.output_json) if args.output_json else OUT_DIR / f"{args.family_name}_family_combo.json"
    output_md = Path(args.output_md) if args.output_md else OUT_DIR / f"{args.family_name}_family_combo.md"
    output_json.write_text(json.dumps(report, indent=2, default=discovery._json_default), encoding="utf-8")
    output_md.write_text(_build_markdown(report), encoding="utf-8")
    print(output_json)
    print(output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
