#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path('/Users/codygnon/Documents/usdjpy_assistant')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import backtest_variant_k_v7_search as v7

OUT_DIR = ROOT / 'research_out'


def _json_default(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return v7._json_default(obj)


def _parse_candidate(name: str) -> dict[str, float]:
    scales = dict(v7.REFERENCE_VARIANTS['v6_validated_only'])
    if name == 'v6_validated_only':
        return scales
    if not name.startswith('v7__'):
        raise SystemExit(f'Unsupported candidate name: {name}')
    for part in name.split('__')[1:]:
        if '=' not in part:
            raise SystemExit(f'Bad candidate part: {part}')
        label, value = part.rsplit('=', 1)
        if label not in scales:
            raise SystemExit(f'Unknown label in candidate: {label}')
        scales[label] = float(value)
    return scales


def _build_md(payload: dict[str, Any]) -> str:
    lines = [
        '# V7 Focus Candidate Validation',
        '',
        f"- name: `{payload['name']}`",
        f"- combined delta USD: `{payload['combined_delta_usd']}`",
        f"- combined delta PF: `{payload['combined_delta_pf']}`",
        f"- combined delta DD: `{payload['combined_delta_dd']}`",
        f"- policy: `{payload['policy']['name']}`",
        '',
        '## Cell Scales',
        '',
    ]
    for label in sorted(payload['cell_scales']):
        lines.append(f"- `{label}`: `{payload['cell_scales'][label]}`")
    lines.append('')
    for ds in ['500k', '1000k']:
        summary = payload['datasets'][ds]['summary']
        delta = payload['datasets'][ds]['delta_vs_baseline']
        sel = payload['datasets'][ds]['selection_counts']
        lines += [
            f'## {ds}',
            '',
            f"- total trades: `{summary['total_trades']}`",
            f"- net USD: `{round(summary['net_usd'], 2)}`",
            f"- PF: `{round(summary['profit_factor'], 4)}`",
            f"- max DD: `{round(summary['max_drawdown_usd'], 2)}`",
            f"- delta USD: `{delta['net_usd']}`",
            f"- delta PF: `{delta['profit_factor']}`",
            f"- delta DD: `{delta['max_drawdown_usd']}`",
            f"- additive trades: `{sel['new_additive_trades_count']}`",
            f"- internal overlap pairs: `{sel['internal_overlap_pairs']}`",
            f"- internal opposite-side pairs: `{sel['internal_opposite_side_overlap_pairs']}`",
            '',
        ]
    return '\n'.join(lines) + '\n'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--output-json', default='')
    ap.add_argument('--output-md', default='')
    args = ap.parse_args()

    scales = _parse_candidate(args.name)
    trades_by_ds = v7._load_inputs(Path(v7.DEFAULT_MATRIX))
    policy = v7._policy()
    baseline_ctx_by_ds = {ds: v7.additive.build_baseline_context(v7.discovery.DATASETS[ds]) for ds in ['500k', '1000k']}
    result = v7._evaluate_variant(args.name, scales, trades_by_ds, policy, baseline_ctx_by_ds)
    payload = {
        'title': 'V7 focused candidate validation',
        'name': args.name,
        'cell_scales': scales,
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
        **result,
    }
    stem = args.name.replace('=', '_').replace('/', '_')
    out_json = Path(args.output_json) if args.output_json else OUT_DIR / f'{stem}_validation.json'
    out_md = Path(args.output_md) if args.output_md else OUT_DIR / f'{stem}_validation.md'
    out_json.write_text(json.dumps(payload, indent=2, default=_json_default), encoding='utf-8')
    out_md.write_text(_build_md(payload), encoding='utf-8')
    print(out_json)
    print(out_md)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
