#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path("/Users/codygnon/Documents/usdjpy_assistant")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.offensive_slice_spec import OffensiveSliceSpec
from scripts import backtest_offensive_slice_additive as additive
from scripts import backtest_offensive_slice_family_combo as family_combo
from scripts import run_offensive_slice_discovery as discovery

OUT_DIR = ROOT / "research_out"
DEFAULT_MATRIX = OUT_DIR / "offensive_slice_discovery_matrix.json"
DEFAULT_OUTPUT = OUT_DIR / "system_variant_k_plus_v44_family_v4_search.json"
DEFAULT_MD = OUT_DIR / "system_variant_k_plus_v44_family_v4_search.md"

# Base leader: full_v3
BASE_CELLS = [
    {
        "label": "B0_ambig_high_pos_sell_strong",
        "slice_id": "v44_ny__short__cells_ambiguous_er_high_der_pos__entry_profiles_Strong",
    },
    {
        "label": "B1_ambig_low_pos_sell_base",
        "slice_id": "v44_ny__short__cells_ambiguous_er_low_der_pos",
    },
    {
        "label": "B2_mom_high_pos_sell",
        "slice_id": "v44_ny__short__cells_momentum_er_high_der_pos",
    },
    {
        "label": "B3_ambig_mid_neg_buy",
        "slice_id": "v44_ny__long__cells_ambiguous_er_mid_der_neg",
    },
    {
        "label": "B4_ambig_mid_pos_sell_strong",
        "slice_id": "v44_ny__short__cells_ambiguous_er_mid_der_pos__entry_profiles_Strong",
    },
    {
        "label": "B5_pbt_high_pos_sell",
        "slice_id": "v44_ny__short__cells_post_breakout_trend_er_high_der_pos",
    },
    {
        "label": "B6_pbt_mid_pos_sell",
        "slice_id": "v44_ny__short__cells_post_breakout_trend_er_mid_der_pos",
    },
]

# Focused adjacent search set only.
OPTION_CELLS = [
    {
        "label": "O0_ambig_high_pos_long_strong",
        "slice_id": "v44_ny__long__cells_ambiguous_er_high_der_pos__entry_profiles_Strong",
        "note": "Opposite-side add in anchor cell.",
    },
    {
        "label": "O1_ambig_low_pos_long_strong",
        "slice_id": "v44_ny__long__cells_ambiguous_er_low_der_pos__entry_profiles_Strong",
        "note": "Large nearby long add, stronger 500k than 1000k.",
    },
    {
        "label": "O2_ambig_mid_pos_long_strong",
        "slice_id": "v44_ny__long__cells_ambiguous_er_mid_der_pos__entry_profiles_Strong",
        "note": "Orthogonal long in the same mid-pos regime pocket.",
    },
    {
        "label": "O3_ambig_mid_neg_short_pullback",
        "slice_id": "v44_ny__short__cells_ambiguous_er_mid_der_neg__entry_signal_modes_pullback",
        "note": "Low-displacement short in nearby negative-der cell.",
    },
    {
        "label": "O4_ambig_low_neg_short_normal",
        "slice_id": "v44_ny__short__cells_ambiguous_er_low_der_neg__entry_profiles_Normal",
        "note": "Tiny adjacent short, zero displacement.",
    },
    {
        "label": "O5_mom_high_neg_short_strong",
        "slice_id": "v44_ny__short__cells_momentum_er_high_der_neg__entry_profiles_Strong",
        "note": "Small momentum continuation short.",
    },
    {
        "label": "O6_pbt_low_neg_long_news",
        "slice_id": "v44_ny__long__cells_post_breakout_trend_er_low_der_neg__entry_profiles_news_trend",
        "note": "Single-trade orthogonal whitespace add.",
    },
]

# Alternate base: use the broader ambig_mid_pos short base instead of Strong-only.
ALT_BASE_REPLACEMENTS = [
    {
        "name": "full_v3",
        "replace": {},
        "notes": "Current board leader.",
    },
    {
        "name": "full_v3_c4_base",
        "replace": {
            "v44_ny__short__cells_ambiguous_er_mid_der_pos__entry_profiles_Strong":
                "v44_ny__short__cells_ambiguous_er_mid_der_pos"
        },
        "notes": "Replace ambig_mid_pos short Strong-only with base.",
    },
]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return discovery._json_default(obj)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Targeted v4 search around the current V44 family leader")
    ap.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    ap.add_argument("--max-option-count", type=int, default=3)
    ap.add_argument("--output-json", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--output-md", default=str(DEFAULT_MD))
    return ap.parse_args()


def _load_matrix(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_spec(matrix: dict[str, Any], slice_id: str) -> OffensiveSliceSpec:
    row = (matrix.get("results") or {}).get(slice_id)
    if not row:
        raise SystemExit(f"Slice id not found in matrix: {slice_id}")
    return OffensiveSliceSpec(**row["slice_spec"])


def _trade_key(trade: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(trade["strategy"]),
        str(trade["entry_time"]),
        str(trade["exit_time"]),
        str(trade["side"]),
    )


def _select_trades(
    specs: list[OffensiveSliceSpec],
    all_trades: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, list[dict[str, Any]]]:
    selected: dict[str, list[dict[str, Any]]] = {"500k": [], "1000k": []}
    for dataset_key in ["500k", "1000k"]:
        combined: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str, str]] = set()
        for spec in specs:
            for trade in all_trades[dataset_key].get(spec.strategy, []):
                if not discovery._passes_filters(trade, spec):
                    continue
                key = _trade_key(trade)
                if key in seen:
                    continue
                seen.add(key)
                combined.append(trade)
        combined.sort(key=lambda t: (t["entry_time"], t["exit_time"], t["strategy"], t["side"]))
        selected[dataset_key] = combined
    return selected


def _run_variant(
    *,
    name: str,
    specs: list[OffensiveSliceSpec],
    all_trades: dict[str, dict[str, list[dict[str, Any]]]],
    baseline_ctxs: dict[str, additive.BaselineContext],
) -> dict[str, Any]:
    selected = _select_trades(specs, all_trades)
    out = {
        "name": name,
        "slice_ids": [spec.resolved_slice_id() for spec in specs],
        "passes_strict": True,
        "datasets": {},
    }
    for dataset_key in ["500k", "1000k"]:
        result = additive.run_slice_additive(
            baseline_ctx=baseline_ctxs[dataset_key],
            slice_spec={"variant_name": name, "slice_ids": out["slice_ids"]},
            selected_trades=selected[dataset_key],
            size_scale=1.0,
        )
        delta = result["delta_vs_baseline"]
        if delta["net_usd"] <= 0 or delta["profit_factor"] < 0:
            out["passes_strict"] = False
        out["datasets"][dataset_key] = {
            "standalone": discovery._metrics(selected[dataset_key]),
            "selection_counts": result["selection_counts"],
            "delta_vs_baseline": delta,
            "variant_summary": result["variant_summary"],
        }
    return out


def _variant_score(row: dict[str, Any]) -> tuple[float, float, float]:
    d500 = row["datasets"]["500k"]["delta_vs_baseline"]
    d1k = row["datasets"]["1000k"]["delta_vs_baseline"]
    counts500 = row["datasets"]["500k"]["selection_counts"]
    counts1k = row["datasets"]["1000k"]["selection_counts"]
    combined_usd = float(d500["net_usd"]) + float(d1k["net_usd"])
    combined_pf = float(d500["profit_factor"]) + float(d1k["profit_factor"])
    displacement = int(counts500["displaced_trades_count"]) + int(counts1k["displaced_trades_count"])
    return (combined_usd, combined_pf, -displacement)


def _build_variants(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    option_labels = [row["label"] for row in OPTION_CELLS]
    variants = []
    for base in ALT_BASE_REPLACEMENTS:
        base_ids = []
        for cell in BASE_CELLS:
            slice_id = base["replace"].get(cell["slice_id"], cell["slice_id"])
            base_ids.append(slice_id)
        for r in range(0, len(OPTION_CELLS) + 1):
            if r > args.max_option_count:
                break
            for combo in itertools.combinations(OPTION_CELLS, r):
                combo_ids = [item["slice_id"] for item in combo]
                combo_labels = [item["label"] for item in combo]
                variants.append({
                    "name": base["name"] if not combo else f"{base['name']}__plus__{'__'.join(combo_labels)}",
                    "base_name": base["name"],
                    "base_notes": base["notes"],
                    "option_labels": combo_labels,
                    "slice_ids": base_ids + combo_ids,
                })
    return variants


def _build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Targeted V44 Family v4 Search",
        "",
        f"- search scope: local expansion around `{report['base_leader']}`",
        f"- variants tested: `{report['variants_tested']}`",
        f"- strict-pass variants: `{report['strict_pass_variants']}`",
        "",
        "## Best Variant",
        "",
        f"- name: `{report['best_variant']['name']}`",
        f"- base: `{report['best_variant']['base_name']}`",
        f"- added options: `{', '.join(report['best_variant']['option_labels']) if report['best_variant']['option_labels'] else 'none'}`",
        "",
    ]
    for dataset_key in ["500k", "1000k"]:
        ds = report["best_variant"]["datasets"][dataset_key]
        delta = ds["delta_vs_baseline"]
        counts = ds["selection_counts"]
        lines += [
            f"### {dataset_key}",
            "",
            f"- delta USD: `{delta['net_usd']}`",
            f"- delta PF: `{delta['profit_factor']}`",
            f"- delta DD: `{delta['max_drawdown_usd']}`",
            f"- additive trades: `{counts['new_additive_trades_count']}`",
            f"- displaced trades: `{counts['displaced_trades_count']}`",
            "",
        ]
    lines += [
        "## Top 10 Strict-Pass Variants",
        "",
    ]
    for row in report["top10"]:
        d500 = row["datasets"]["500k"]["delta_vs_baseline"]
        d1k = row["datasets"]["1000k"]["delta_vs_baseline"]
        c500 = row["datasets"]["500k"]["selection_counts"]
        c1k = row["datasets"]["1000k"]["selection_counts"]
        lines.append(
            f"- `{row['name']}` | 500k `{d500['net_usd']}` / `{d500['profit_factor']}` / `{d500['max_drawdown_usd']}` "
            f"(add `{c500['new_additive_trades_count']}`, disp `{c500['displaced_trades_count']}`) | "
            f"1000k `{d1k['net_usd']}` / `{d1k['profit_factor']}` / `{d1k['max_drawdown_usd']}` "
            f"(add `{c1k['new_additive_trades_count']}`, disp `{c1k['displaced_trades_count']}`)"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    global args
    args = _parse_args()
    matrix = _load_matrix(Path(args.matrix).resolve())

    all_slice_ids = {row["slice_id"] for row in BASE_CELLS} | {row["slice_id"] for row in OPTION_CELLS}
    # include alternate replacements too
    for base in ALT_BASE_REPLACEMENTS:
        all_slice_ids.update(base["replace"].values())
    specs_by_id = {slice_id: _load_spec(matrix, slice_id) for slice_id in all_slice_ids}

    strategies = {spec.strategy for spec in specs_by_id.values()}
    all_trades = discovery._load_all_normalized_trades(strategies)
    baseline_ctxs = {dk: additive.build_baseline_context(discovery.DATASETS[dk]) for dk in ["500k", "1000k"]}

    variants = _build_variants(matrix)
    results = []
    for variant in variants:
        specs = [specs_by_id[sid] for sid in variant["slice_ids"]]
        row = _run_variant(
            name=variant["name"],
            specs=specs,
            all_trades=all_trades,
            baseline_ctxs=baseline_ctxs,
        )
        row["base_name"] = variant["base_name"]
        row["base_notes"] = variant["base_notes"]
        row["option_labels"] = variant["option_labels"]
        results.append(row)

    results.sort(key=_variant_score, reverse=True)
    strict_pass = [row for row in results if row["passes_strict"]]
    best = strict_pass[0] if strict_pass else results[0]

    report = {
        "title": "Targeted V44 Family v4 Search",
        "base_leader": "full_v3",
        "variants_tested": len(results),
        "strict_pass_variants": len(strict_pass),
        "best_variant": best,
        "top10": strict_pass[:10],
        "all_results": results,
    }

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    out_md.write_text(_build_markdown(report), encoding="utf-8")
    print(out_json)
    print(out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
