#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path('/Users/codygnon/Documents/usdjpy_assistant')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import package_freeze_closeout_lib as lib

OUT_JSON = lib.OUT_DIR / 'package_time_regime_frontier.json'
OUT_MD = lib.OUT_DIR / 'package_time_regime_frontier.md'
DIMENSIONS = ['weekday', 'session_bucket', 'entry_hour_bucket', 'ownership_regime']


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Close time/regime frontier for top package finalists')
    ap.add_argument('--output-json', default=str(OUT_JSON))
    ap.add_argument('--output-md', default=str(OUT_MD))
    ap.add_argument('--package', action='append', choices=lib.TIME_FRONTIER_TOP_TWO, dest='packages')
    ap.add_argument('--top-n-values', type=int, default=6)
    return ap.parse_args()


def _package_values(context: lib.Context, scales: dict[str, float], dim: str, top_n: int) -> list[str]:
    counts: dict[str, int] = {}
    for ds in ['500k', '1000k']:
        for label in lib.active_labels(scales):
            for trade in context['trades_by_ds'][ds].get(label, []):
                value = str(lib.trade_attr(trade, dim))
                counts[value] = counts.get(value, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [value for value, count in ordered if count >= 2][:top_n]


def _build_md(payload: dict) -> str:
    lines = [
        '# Package Time/Regime Frontier',
        '',
        '- One-dimension-at-a-time refinement on the two finalists.',
        '- No combined filters are promoted unless a single-dimension filter clearly improves the package first.',
        '',
    ]
    for package in payload['packages']:
        lines += [f"## {package['package']}", '']
        for dim in package['dimensions']:
            lines.append(f"### {dim['dimension']}")
            lines.append('')
            best = dim['best_candidate']
            if best is None:
                lines.append('- no candidate values tested')
                lines.append('')
                continue
            lines.append(f"- best: `{best['name']}` | USD `{best['combined_delta_usd']}` | PF `{best['combined_delta_pf']}` | DD `{best['combined_delta_dd']}`")
            for row in dim['top_candidates']:
                lines.append(f"- `{row['name']}`: USD `{row['combined_delta_usd']}`, PF `{row['combined_delta_pf']}`, DD `{row['combined_delta_dd']}`")
            lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> int:
    args = parse_args()
    packages = args.packages or list(lib.TIME_FRONTIER_TOP_TWO)
    context = lib.load_context()
    package_results = []
    for package_name in packages:
        base_scales = lib.package_scales(package_name)
        dimensions = []
        for dim in DIMENSIONS:
            rows = []
            for value in _package_values(context, base_scales, dim, args.top_n_values):
                label_filters = {label: lib.make_value_disable_filter(dim, value) for label in lib.active_labels(base_scales)}
                rows.append(lib.evaluate_package(
                    context=context,
                    name=f'{package_name}__time__drop_{dim}_{value}',
                    scales=base_scales,
                    label_filters=label_filters,
                    metadata={'package': package_name, 'dimension': dim, 'blocked_value': value},
                ))
            rows.sort(key=lambda r: (r['passes_strict'], r['combined_delta_usd'], r['combined_delta_pf'], -r['combined_delta_dd']), reverse=True)
            dimensions.append({
                'dimension': dim,
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
        package_results.append({'package': package_name, 'dimensions': dimensions})

    payload = {
        'title': 'Package time/regime frontier',
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
