#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path('/Users/codygnon/Documents/usdjpy_assistant')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import backtest_offensive_slice_additive as additive
from scripts import backtest_offensive_slice_family_combo as family_combo
from scripts import backtest_variant_k_v6_search as v6
from scripts import run_offensive_slice_discovery as discovery

OUT_DIR = ROOT / 'research_out'
DEFAULT_OUTPUT = OUT_DIR / 'system_variant_k_v7_search.json'
DEFAULT_MD = OUT_DIR / 'system_variant_k_v7_search.md'
DEFAULT_MATRIX = OUT_DIR / 'offensive_slice_discovery_matrix.json'

# Keep the proven v5 base and strongest v6 breakout adds fixed.
FIXED_CORE = [
    'C0_sell_strong', 'C1_sell_base', 'C2_sell', 'C3_buy',
    'C4_sell_base', 'C5_pbt_sell', 'C6_pbt_sell',
    'O0_buy_strong', 'O1_buy_strong', 'O2_buy_strong',
    'ADJ_meanrev_low_neg_buy', 'ADJ_ambig_mid_neg_sell', 'ADJ_mom_high_neg_sell',
    'N1_brkout_low_neg_sell_strong', 'N2_brkout_low_pos_buy_strong',
]

# Search only the true remaining decisions around validated v6.
TUNABLE_OPTIONS: list[tuple[str, list[float]]] = [
    ('L2_brkout_mid_neg_buy', [0.0, 0.5, 1.0]),
    ('T1_ambig_high_pos_buy', [0.0, 0.5, 1.0]),
    ('T2_brkout_mid_pos_buy', [0.0, 0.5, 1.0]),
    ('N3_brkout_low_neg_buy_news', [0.0, 0.5, 1.0]),
    ('N4_pbt_low_neg_buy_news', [0.0, 0.5, 1.0]),
    ('L1_mom_low_pos_buy', [0.0, 0.25, 0.5, 1.0]),
    ('T3_ambig_mid_pos_sell', [0.0, 0.25, 0.5, 1.0]),
]

REFERENCE_VARIANTS = {
    'v6_validated_no_singletons': {
        **{label: 1.0 for label in FIXED_CORE},
        'L2_brkout_mid_neg_buy': 1.0,
        'T1_ambig_high_pos_buy': 1.0,
        'T2_brkout_mid_pos_buy': 1.0,
        'N3_brkout_low_neg_buy_news': 0.0,
        'N4_pbt_low_neg_buy_news': 0.0,
        'L1_mom_low_pos_buy': 0.0,
        'T3_ambig_mid_pos_sell': 0.0,
    },
    'v6_validated_only': {
        **{label: 1.0 for label in FIXED_CORE},
        'L2_brkout_mid_neg_buy': 1.0,
        'T1_ambig_high_pos_buy': 1.0,
        'T2_brkout_mid_pos_buy': 1.0,
        'N3_brkout_low_neg_buy_news': 1.0,
        'N4_pbt_low_neg_buy_news': 1.0,
        'L1_mom_low_pos_buy': 0.0,
        'T3_ambig_mid_pos_sell': 0.0,
    },
}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return discovery._json_default(obj)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Targeted v7 composition search around validated v6')
    ap.add_argument('--output-json', default=str(DEFAULT_OUTPUT))
    ap.add_argument('--output-md', default=str(DEFAULT_MD))
    ap.add_argument('--matrix', default=str(DEFAULT_MATRIX))
    ap.add_argument('--top-k', type=int, default=20)
    return ap.parse_args()


def _policy() -> additive.ConflictPolicy:
    return additive.ConflictPolicy(
        name='native_v44_hedging_like',
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


def _load_inputs(matrix_path: Path):
    matrix = family_combo._load_matrix(matrix_path)
    specs = v6._load_all_specs(matrix)
    wanted_labels = set(FIXED_CORE) | {label for label, _ in TUNABLE_OPTIONS}
    missing = sorted(wanted_labels - set(specs))
    if missing:
        raise SystemExit(f'Missing specs for: {missing}')
    strategies = {specs[label].strategy for label in wanted_labels}
    all_trades = discovery._load_all_normalized_trades(strategies)
    trades_by_ds = v6._select_all_trades(specs, all_trades)
    return trades_by_ds


def _scaled_combined_trades(scales: dict[str, float], trades_by_label: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    combined: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for label in FIXED_CORE + [name for name, _ in TUNABLE_OPTIONS]:
        scale = float(scales.get(label, 0.0))
        if scale <= 0.0:
            continue
        for trade in trades_by_label.get(label, []):
            key = (str(trade['strategy']), str(trade['entry_time']), str(trade['exit_time']), str(trade['side']))
            if key in seen:
                continue
            seen.add(key)
            if abs(scale - 1.0) < 1e-9:
                combined.append(trade)
            else:
                t = deepcopy(trade)
                t['usd'] = float(t['usd']) * scale
                t['size_scale'] = float(t.get('size_scale', 1.0)) * scale
                combined.append(t)
    combined.sort(key=lambda t: (t['entry_time'], t['exit_time']))
    return combined


def _evaluate_variant(
    name: str,
    scales: dict[str, float],
    trades_by_ds: dict[str, dict[str, list[dict[str, Any]]]],
    policy: additive.ConflictPolicy,
    baseline_ctx_by_ds: dict[str, additive.BaselineContext],
) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for ds in ['500k', '1000k']:
        combined = _scaled_combined_trades(scales, trades_by_ds[ds])
        result = additive.run_slice_additive_with_policy(
            baseline_ctx=baseline_ctx_by_ds[ds],
            slice_spec={'variant': name, 'cell_scales': scales},
            selected_trades=combined,
            conflict_policy=policy,
            size_scale=1.0,
        )
        datasets[ds] = {
            'summary': result['variant_summary'],
            'delta_vs_baseline': result['delta_vs_baseline'],
            'selection_counts': result['selection_counts'],
            'policy_stats': result.get('policy_stats', {}),
        }
    d5 = datasets['500k']['delta_vs_baseline']
    d1 = datasets['1000k']['delta_vs_baseline']
    return {
        'name': name,
        'cell_scales': scales,
        'datasets': datasets,
        'combined_delta_usd': round(d5['net_usd'] + d1['net_usd'], 2),
        'combined_delta_pf': round(d5['profit_factor'] + d1['profit_factor'], 4),
        'combined_delta_dd': round(d5['max_drawdown_usd'] + d1['max_drawdown_usd'], 2),
        'passes_strict': d5['net_usd'] > 0.0 and d1['net_usd'] > 0.0 and d5['profit_factor'] >= 0.0 and d1['profit_factor'] >= 0.0,
    }


def _variant_name(scales: dict[str, float]) -> str:
    changes = []
    ref = REFERENCE_VARIANTS['v6_validated_only']
    for label, _ in TUNABLE_OPTIONS:
        if abs(scales[label] - ref.get(label, 0.0)) > 1e-9:
            changes.append(f'{label}={scales[label]:g}')
    return 'v6_validated_only' if not changes else 'v7__' + '__'.join(changes)


def _reference_summary(name: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        'name': name,
        'combined_delta_usd': result['combined_delta_usd'],
        'combined_delta_pf': result['combined_delta_pf'],
        'combined_delta_dd': result['combined_delta_dd'],
        'passes_strict': result['passes_strict'],
    }


def _changed_scales(scales: dict[str, float]) -> list[dict[str, Any]]:
    ref = REFERENCE_VARIANTS['v6_validated_only']
    changes = []
    for label, _ in TUNABLE_OPTIONS:
        if abs(scales[label] - ref.get(label, 0.0)) > 1e-9:
            changes.append({'label': label, 'scale': scales[label], 'was': ref.get(label, 0.0)})
    return changes


def _build_markdown(payload: dict[str, Any]) -> str:
    best = payload['best_candidate']
    lines = [
        '# Variant K v7 Search',
        '',
        'Search method:',
        '- fixed the validated v5 base and validated V44 breakout adds',
        '- searched only the real remaining package decisions',
        '- allowed partial-size reentry for failed-but-USD-positive `L1` and `T3`',
        '- kept exact strict `native_v44_hedging_like` policy',
        '',
        '## References',
        '',
    ]
    for ref in payload['references']:
        lines.append(f"- `{ref['name']}`: USD `{ref['combined_delta_usd']}`, PF `{ref['combined_delta_pf']}`, DD `{ref['combined_delta_dd']}`")
    lines += [
        '',
        '## Best Candidate',
        '',
        f"- name: `{best['name']}`",
        f"- combined delta USD: `{best['combined_delta_usd']}`",
        f"- combined delta PF: `{best['combined_delta_pf']}`",
        f"- combined delta DD: `{best['combined_delta_dd']}`",
        '',
        'Changed cell scales vs `v6_validated_only`:',
    ]
    changes = payload['best_candidate_changes']
    if changes:
        for change in changes:
            lines.append(f"- `{change['label']}` -> `{change['scale']}` (was `{change['was']}`)")
    else:
        lines.append('- none')
    lines.append('')
    for ds in ['500k', '1000k']:
        summary = best['datasets'][ds]['summary']
        delta = best['datasets'][ds]['delta_vs_baseline']
        selection = best['datasets'][ds]['selection_counts']
        lines += [
            f'## {ds}',
            '',
            f"- net USD: `{round(summary['net_usd'], 2)}`",
            f"- PF: `{round(summary['profit_factor'], 4)}`",
            f"- max DD: `{round(summary['max_drawdown_usd'], 2)}`",
            f"- delta USD: `{delta['net_usd']}`",
            f"- delta PF: `{delta['profit_factor']}`",
            f"- delta DD: `{delta['max_drawdown_usd']}`",
            f"- additive trades: `{selection['new_additive_trades_count']}`",
            '',
        ]
    lines += [
        '## Top Candidates',
        '',
    ]
    for row in payload['top_candidates']:
        lines.append(f"- `{row['name']}`: USD `{row['combined_delta_usd']}`, PF `{row['combined_delta_pf']}`, DD `{row['combined_delta_dd']}`")
    return '\n'.join(lines) + '\n'


def main() -> int:
    args = _parse_args()
    trades_by_ds = _load_inputs(Path(args.matrix))
    policy = _policy()
    baseline_ctx_by_ds = {ds: additive.build_baseline_context(discovery.DATASETS[ds]) for ds in ['500k', '1000k']}

    references = []
    reference_results = {}
    for name, scales in REFERENCE_VARIANTS.items():
        result = _evaluate_variant(name, scales, trades_by_ds, policy, baseline_ctx_by_ds)
        references.append(_reference_summary(name, result))
        reference_results[name] = result

    tuned_labels = [label for label, _ in TUNABLE_OPTIONS]
    tuned_scales = [scales for _, scales in TUNABLE_OPTIONS]
    total = 1
    for options in tuned_scales:
        total *= len(options)

    rows: list[dict[str, Any]] = []
    best = None
    for idx, values in enumerate(product(*tuned_scales), start=1):
        scales = {label: 1.0 for label in FIXED_CORE}
        scales.update({label: value for label, value in zip(tuned_labels, values)})
        result = _evaluate_variant(_variant_name(scales), scales, trades_by_ds, policy, baseline_ctx_by_ds)
        rows.append(result)
        if result['passes_strict'] and (
            best is None or (result['combined_delta_usd'], result['combined_delta_pf'], -result['combined_delta_dd']) >
            (best['combined_delta_usd'], best['combined_delta_pf'], -best['combined_delta_dd'])
        ):
            best = result
        if idx % 250 == 0 or idx == total:
            print(f'Progress {idx}/{total}: best={best["combined_delta_usd"] if best else 0} ({best["name"] if best else "none"})', flush=True)

    rows.sort(key=lambda r: (r['passes_strict'], r['combined_delta_usd'], r['combined_delta_pf'], -r['combined_delta_dd']), reverse=True)
    payload = {
        'title': 'Variant K v7 targeted composition search',
        'search_space': {
            'fixed_core': FIXED_CORE,
            'tunable_options': [{'label': label, 'scales': scales} for label, scales in TUNABLE_OPTIONS],
            'variant_count': total,
            'policy': {
                'name': policy.name,
                'hedging_enabled': policy.hedging_enabled,
                'allow_internal_overlap': policy.allow_internal_overlap,
                'allow_opposite_side_overlap': policy.allow_opposite_side_overlap,
                'margin_model_enabled': policy.margin_model_enabled,
                'margin_leverage': policy.margin_leverage,
                'margin_buffer_pct': policy.margin_buffer_pct,
                'max_lot_per_trade': policy.max_lot_per_trade,
            },
        },
        'references': references,
        'best_candidate': best,
        'best_candidate_changes': _changed_scales(best['cell_scales']) if best else [],
        'top_candidates': [
            {
                'name': row['name'],
                'combined_delta_usd': row['combined_delta_usd'],
                'combined_delta_pf': row['combined_delta_pf'],
                'combined_delta_dd': row['combined_delta_dd'],
                'passes_strict': row['passes_strict'],
            }
            for row in rows[:args.top_k]
        ],
    }

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.write_text(json.dumps(payload, indent=2, default=_json_default), encoding='utf-8')
    output_md.write_text(_build_markdown(payload), encoding='utf-8')
    print(output_json)
    print(output_md)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
