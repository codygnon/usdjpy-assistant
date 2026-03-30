#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path('/Users/codygnon/Documents/usdjpy_assistant')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import package_freeze_closeout_lib as lib

OUT_JSON = lib.OUT_DIR / 'package_time_regime_followup.json'
OUT_MD = lib.OUT_DIR / 'package_time_regime_followup.md'
PACKAGES = ['v7_usd_max', 'v7_pfdd']
FOLLOWUP_DIMS = ['ownership_regime', 'session_bucket', 'entry_hour_bucket', 'weekday']
TARGET_LABEL = 'L1_mom_low_pos_buy'
BASE_WEEKDAY = 'Monday'


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Final combined time/regime follow-up centered on L1 drop Monday')
    ap.add_argument('--output-json', default=str(OUT_JSON))
    ap.add_argument('--output-md', default=str(OUT_MD))
    ap.add_argument('--package', action='append', choices=PACKAGES, dest='packages')
    ap.add_argument('--top-n-values', type=int, default=4)
    return ap.parse_args()


def _combined_filter(attr: str, value: str):
    def _keep(trade: dict[str, object]) -> bool:
        if str(lib.trade_attr(trade, 'weekday')) == BASE_WEEKDAY:
            return False
        if str(lib.trade_attr(trade, attr)) == value:
            return False
        return True
    return _keep


def _build_md(payload: dict) -> str:
    lines = [
        '# Package Time/Regime Follow-Up',
        '',
        '- Final combined follow-up centered on `L1` drop `Monday`.',
        '- This tests whether a second simple observable filter creates a credible remaining improvement axis.',
        '',
    ]
    for package in payload['packages']:
        lines += [f"## {package['package']}", '']
        best = package.get('best_candidate')
        if best is None:
            lines.append('- no candidates tested')
            lines.append('')
            continue
        lines.append(f"- best: `{best['name']}` | USD `{best['combined_delta_usd']}` | PF `{best['combined_delta_pf']}` | DD `{best['combined_delta_dd']}`")
        for row in package['top_candidates']:
            lines.append(f"- `{row['name']}`: USD `{row['combined_delta_usd']}`, PF `{row['combined_delta_pf']}`, DD `{row['combined_delta_dd']}`")
        lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> int:
    args = parse_args()
    packages = args.packages or PACKAGES
    context = lib.load_context()
    package_results = []
    for package_name in packages:
        scales = lib.package_scales(package_name)
        if float(scales.get(TARGET_LABEL, 0.0)) <= 0.0:
            package_results.append({'package': package_name, 'best_candidate': None, 'top_candidates': [], 'rows': []})
            continue
        rows = []
        base_filter = {TARGET_LABEL: lib.make_value_disable_filter('weekday', BASE_WEEKDAY)}
        base_name = f'{package_name}__followup__L1_drop_weekday_{BASE_WEEKDAY}'
        rows.append(lib.evaluate_package(
            context=context,
            name=base_name,
            scales=scales,
            label_filters=base_filter,
            metadata={'package': package_name, 'target_label': TARGET_LABEL, 'base_weekday_drop': BASE_WEEKDAY},
        ))
        for dim in FOLLOWUP_DIMS:
            values = lib.allowed_filter_values(context, TARGET_LABEL, dim, top_n=args.top_n_values)
            for value in values:
                if dim == 'weekday' and value == BASE_WEEKDAY:
                    continue
                rows.append(lib.evaluate_package(
                    context=context,
                    name=f'{package_name}__followup__L1_drop_{BASE_WEEKDAY}__drop_{dim}_{value}',
                    scales=scales,
                    label_filters={TARGET_LABEL: _combined_filter(dim, value)},
                    metadata={'package': package_name, 'target_label': TARGET_LABEL, 'base_weekday_drop': BASE_WEEKDAY, 'second_dimension': dim, 'blocked_value': value},
                ))
        rows.sort(key=lambda r: (r['passes_strict'], r['combined_delta_usd'], r['combined_delta_pf'], -r['combined_delta_dd']), reverse=True)
        package_results.append({
            'package': package_name,
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
    payload = {
        'title': 'Package time/regime follow-up',
        'packages': package_results,
        'policy': {
            'name': context['policy'].name,
            'hedging_enabled': context['policy'].hedging_enabled,
            'allow_internal_overlap': context['policy'].allow_internal_overlap,
            'allow_opposite_side_overlap': context['policy'].allow_opposite_side_overlap,
            'margin_model_enabled': context['policy'].margin_model_enabled,
        },
    }
    lib.write_json_md(Path(args.output_json), Path(args.output_md), payload, _build_md(payload))
    print(Path(args.output_json))
    print(Path(args.output_md))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
