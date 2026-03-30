#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path('/Users/codygnon/Documents/usdjpy_assistant')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import package_freeze_closeout_lib as lib

OUT_JSON = lib.OUT_DIR / 'package_pruning_frontier.json'
OUT_MD = lib.OUT_DIR / 'package_pruning_frontier.md'
CONDITIONAL_ATTRS = ['session_bucket', 'entry_hour_bucket', 'weekday', 'entry_profile', 'entry_signal_mode', 'ownership_regime']
TARGET_LABELS = ['L1_mom_low_pos_buy', 'T3_ambig_mid_pos_sell', 'N3_brkout_low_neg_buy_news', 'N4_pbt_low_neg_buy_news', 'C0_base']


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Close pruning frontier for package freeze research')
    ap.add_argument('--output-json', default=str(OUT_JSON))
    ap.add_argument('--output-md', default=str(OUT_MD))
    ap.add_argument('--package', action='append', choices=lib.PRUNING_PACKAGE_ORDER, dest='packages')
    ap.add_argument('--top-n-values', type=int, default=4)
    return ap.parse_args()


def _target_scales(base_scales: dict[str, float], label: str) -> list[float]:
    base = float(base_scales.get(label, 0.0))
    if base <= 0:
        return []
    options = [0.0, 0.25, 0.5]
    return sorted({opt for opt in options if opt <= base + 1e-9})


def _c0_filter(attr: str, value: str):
    return lib.make_value_disable_filter(attr, value, subset_predicate=lib.non_strong_c0_predicate)


def _build_md(payload: dict) -> str:
    lines = [
        '# Package Pruning Frontier',
        '',
        '- This closes the negative-space frontier for questioned slices and thin adds.',
        '- C0 pruning is applied only to the widened non-Strong portion when possible.',
        '',
    ]
    for package in payload['packages']:
        lines += [f"## {package['package']}", '']
        for target in package['targets']:
            lines.append(f"### {target['label']}")
            lines.append('')
            best = target['best_candidate']
            if best is None:
                lines.append('- no variants tested')
                lines.append('')
                continue
            lines.append(f"- best: `{best['name']}` | USD `{best['combined_delta_usd']}` | PF `{best['combined_delta_pf']}` | DD `{best['combined_delta_dd']}`")
            for row in target['top_candidates']:
                lines.append(f"- `{row['name']}`: USD `{row['combined_delta_usd']}`, PF `{row['combined_delta_pf']}`, DD `{row['combined_delta_dd']}`")
            lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> int:
    args = parse_args()
    packages = args.packages or list(lib.PRUNING_PACKAGE_ORDER)
    context = lib.load_context()
    package_results = []

    for package_name in packages:
        base_scales = lib.package_scales(package_name)
        targets = []
        for label in TARGET_LABELS:
            if float(base_scales.get(label, 0.0)) <= 0.0 and label != 'C0_base':
                continue
            rows = []
            if label == 'C0_base' and float(base_scales.get('C0_base', 0.0)) <= 0.0:
                continue
            for scale in _target_scales(base_scales, label):
                if abs(scale - float(base_scales.get(label, 0.0))) < 1e-9:
                    continue
                scales = dict(base_scales)
                scales[label] = scale
                row = lib.evaluate_package(
                    context=context,
                    name=f'{package_name}__prune__{label}_{int(scale * 100)}pct',
                    scales=scales,
                    metadata={'package': package_name, 'target_label': label, 'prune_type': 'downweight', 'target_scale': scale},
                )
                rows.append(row)

            for attr in CONDITIONAL_ATTRS:
                values = lib.allowed_filter_values(context, label, attr, top_n=args.top_n_values)
                for value in values:
                    if label == 'C0_base':
                        label_filters = {'C0_base': _c0_filter(attr, value)}
                    else:
                        label_filters = {label: lib.make_value_disable_filter(attr, value)}
                    row = lib.evaluate_package(
                        context=context,
                        name=f'{package_name}__prune__{label}__drop_{attr}_{value}',
                        scales=base_scales,
                        label_filters=label_filters,
                        metadata={'package': package_name, 'target_label': label, 'prune_type': 'conditional_disable', 'attr': attr, 'value': value},
                    )
                    rows.append(row)

            rows.sort(key=lambda r: (r['passes_strict'], r['combined_delta_usd'], r['combined_delta_pf'], -r['combined_delta_dd']), reverse=True)
            targets.append({
                'label': label,
                'variant_count': len(rows),
                'best_candidate': None if not rows else {
                    'name': rows[0]['name'],
                    'combined_delta_usd': rows[0]['combined_delta_usd'],
                    'combined_delta_pf': rows[0]['combined_delta_pf'],
                    'combined_delta_dd': rows[0]['combined_delta_dd'],
                    'passes_strict': rows[0]['passes_strict'],
                },
                'top_candidates': [
                    {
                        'name': r['name'],
                        'combined_delta_usd': r['combined_delta_usd'],
                        'combined_delta_pf': r['combined_delta_pf'],
                        'combined_delta_dd': r['combined_delta_dd'],
                        'passes_strict': r['passes_strict'],
                    }
                    for r in rows[:10]
                ],
                'rows': rows,
            })
        package_results.append({'package': package_name, 'targets': targets})

    payload = {
        'title': 'Package pruning frontier',
        'policy': {
            'name': context['policy'].name,
            'hedging_enabled': context['policy'].hedging_enabled,
            'allow_internal_overlap': context['policy'].allow_internal_overlap,
            'allow_opposite_side_overlap': context['policy'].allow_opposite_side_overlap,
            'margin_model_enabled': context['policy'].margin_model_enabled,
        },
        'packages': package_results,
    }
    lib.write_json_md(Path(args.output_json), Path(args.output_md), payload, _build_md(payload))
    print(Path(args.output_json))
    print(Path(args.output_md))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
