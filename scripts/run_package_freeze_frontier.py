#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path('/Users/codygnon/Documents/usdjpy_assistant')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import package_freeze_closeout_lib as lib

OUT_JSON = lib.OUT_DIR / 'package_freeze_frontier.json'
OUT_MD = lib.OUT_DIR / 'package_freeze_frontier.md'
UNIFORM_SCALES = [0.25, 0.5, 0.75, 1.0]
DOWNWEIGHT_OPTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Close package frontier for freeze research')
    ap.add_argument('--output-json', default=str(OUT_JSON))
    ap.add_argument('--output-md', default=str(OUT_MD))
    ap.add_argument('--package', action='append', choices=lib.PACKAGE_FRONTIER_ORDER, dest='packages')
    ap.add_argument('--limit-combined', type=int, default=0, help='Optional cap for combined weak-label grid per package (0 = no cap).')
    return ap.parse_args()


def _package_specific_variants(package_name: str, base_scales: dict[str, float]) -> list[tuple[str, dict[str, float], dict]]:
    variants: list[tuple[str, dict[str, float], dict]] = []
    variants.append((f'{package_name}__base', dict(base_scales), {'package': package_name, 'kind': 'base'}))

    for scale in UNIFORM_SCALES:
        variants.append((
            f'{package_name}__uniform_{int(scale * 100)}pct',
            lib.multiply_scales(base_scales, scale),
            {'package': package_name, 'kind': 'uniform', 'uniform_scale': scale},
        ))

    weak_labels = lib.package_weak_labels(package_name)
    options_by_label: list[tuple[str, list[float]]] = []
    for label in weak_labels:
        base = float(base_scales.get(label, 0.0))
        allowed = sorted({opt for opt in DOWNWEIGHT_OPTIONS if opt <= base + 1e-9})
        options_by_label.append((label, allowed))
        for opt in allowed:
            if abs(opt - base) < 1e-9:
                continue
            scales = dict(base_scales)
            scales[label] = opt
            variants.append((
                f'{package_name}__{label}_{int(opt * 100)}pct',
                scales,
                {'package': package_name, 'kind': 'single_downweight', 'target_label': label, 'target_scale': opt},
            ))

    combined_count = 0
    if options_by_label:
        from itertools import product
        label_names = [label for label, _ in options_by_label]
        option_lists = [opts for _, opts in options_by_label]
        for values in product(*option_lists):
            scales = dict(base_scales)
            changed = []
            for label, opt in zip(label_names, values):
                scales[label] = opt
                if abs(opt - base_scales.get(label, 0.0)) > 1e-9:
                    changed.append((label, opt))
            if not changed:
                continue
            combined_count += 1
            label_part = '__'.join(f'{label}_{int(opt * 100)}pct' for label, opt in changed)
            variants.append((
                f'{package_name}__combo__{label_part}',
                scales,
                {'package': package_name, 'kind': 'combined_downweight', 'changes': [{'label': l, 'scale': s} for l, s in changed]},
            ))
    return variants


def _build_md(payload: dict) -> str:
    lines = [
        '# Package Freeze Frontier',
        '',
        '- This maps the package-sizing frontier for the four final package candidates under the strict policy.',
        '- DD-adjusted leader = lowest-DD candidate within 2% of the USD leader.',
        '',
        '## Overall Leaders',
        '',
    ]
    for key in ['usd_leader', 'pf_leader', 'dd_adjusted_leader']:
        row = payload['overall_leaders'].get(key)
        if row is None:
            lines.append(f'- `{key}`: `none`')
        else:
            lines.append(f"- `{key}`: `{row['name']}` | USD `{row['combined_delta_usd']}` | PF `{row['combined_delta_pf']}` | DD `{row['combined_delta_dd']}`")
    lines.append('')
    for package in payload['packages']:
        lines += [
            f"## {package['package']}",
            '',
            f"- variants tested: `{package['variant_count']}`",
            f"- strict-pass variants: `{package['strict_pass_count']}`",
            f"- usd leader: `{package['leaders']['usd_leader']['name'] if package['leaders']['usd_leader'] else 'none'}`",
            f"- pf leader: `{package['leaders']['pf_leader']['name'] if package['leaders']['pf_leader'] else 'none'}`",
            f"- dd-adjusted leader: `{package['leaders']['dd_adjusted_leader']['name'] if package['leaders']['dd_adjusted_leader'] else 'none'}`",
            '',
        ]
        for row in package['top_candidates']:
            lines.append(f"- `{row['name']}`: USD `{row['combined_delta_usd']}`, PF `{row['combined_delta_pf']}`, DD `{row['combined_delta_dd']}`")
        lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> int:
    args = parse_args()
    packages = args.packages or list(lib.PACKAGE_FRONTIER_ORDER)
    context = lib.load_context()

    package_rows = []
    all_rows = []
    for package_name in packages:
        base_scales = lib.package_scales(package_name)
        variants = _package_specific_variants(package_name, base_scales)
        if args.limit_combined > 0:
            filtered = []
            combined_seen = 0
            for item in variants:
                meta = item[2]
                if meta.get('kind') == 'combined_downweight':
                    combined_seen += 1
                    if combined_seen > args.limit_combined:
                        continue
                filtered.append(item)
            variants = filtered
        rows = []
        for name, scales, meta in variants:
            row = lib.evaluate_package(context=context, name=name, scales=scales, metadata=meta)
            rows.append(row)
            all_rows.append(row)
        rows.sort(key=lambda r: (r['passes_strict'], r['combined_delta_usd'], r['combined_delta_pf'], -r['combined_delta_dd']), reverse=True)
        package_rows.append({
            'package': package_name,
            'variant_count': len(rows),
            'strict_pass_count': sum(1 for r in rows if r['passes_strict']),
            'leaders': lib.choose_leaders(rows),
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
        'title': 'Package freeze frontier',
        'policy': {
            'name': context['policy'].name,
            'hedging_enabled': context['policy'].hedging_enabled,
            'allow_internal_overlap': context['policy'].allow_internal_overlap,
            'allow_opposite_side_overlap': context['policy'].allow_opposite_side_overlap,
            'margin_model_enabled': context['policy'].margin_model_enabled,
            'margin_leverage': context['policy'].margin_leverage,
            'margin_buffer_pct': context['policy'].margin_buffer_pct,
            'max_lot_per_trade': context['policy'].max_lot_per_trade,
        },
        'packages': package_rows,
        'overall_leaders': lib.choose_leaders(all_rows),
    }
    json_path = Path(args.output_json)
    md_path = Path(args.output_md)
    lib.write_json_md(json_path, md_path, payload, _build_md(payload))
    print(json_path)
    print(md_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
