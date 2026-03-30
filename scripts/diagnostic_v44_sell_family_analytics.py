#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path('/Users/codygnon/Documents/usdjpy_assistant')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import backtest_offensive_slice_additive as additive
from scripts import backtest_offensive_slice_family_combo as family_combo
from scripts import run_offensive_slice_discovery as discovery

OUT = ROOT / 'research_out'
FAMILY_JSON = OUT / 'v44_sell_family_combo.json'
OUTPUT_JSON = OUT / 'v44_sell_family_analytics.json'
OUTPUT_MD = OUT / 'v44_sell_family_analytics.md'
SLICE_IDS = [
    'v44_ny__short__cells_ambiguous_er_high_der_pos',
    'v44_ny__short__cells_ambiguous_er_low_der_pos',
    'v44_ny__short__cells_momentum_er_high_der_pos',
]
VALIDATION_JSONS = {
    SLICE_IDS[0]: OUT / 'v44_ambiguous_er_high_der_pos_sell_validation.json',
    SLICE_IDS[1]: OUT / 'v44_ambiguous_er_low_der_pos_sell_validation.json',
    SLICE_IDS[2]: OUT / 'v44_momentum_er_high_der_pos_sell_validation.json',
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def _month_key(ts: str) -> str:
    return pd.Timestamp(ts).strftime('%Y-%m')


def _family_top3_selected() -> tuple[list[Any], dict[str, list[dict[str, Any]]]]:
    matrix = family_combo._load_matrix(family_combo.DEFAULT_MATRIX)
    specs = family_combo._specs_from_ids(matrix, SLICE_IDS)
    all_trades = discovery._load_all_normalized_trades({'v44_ny'})
    selected = family_combo._dedupe_selected_trades(specs, all_trades)
    return specs, selected


def _monthly_added_trade_contrib(dataset_key: str, selected: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_ctx = additive.build_baseline_context(discovery.DATASETS[dataset_key])
    overlaps = []
    additive_trades = []
    for trade in selected:
        if additive._has_exact_baseline_match(trade, baseline_ctx.baseline_coupled):
            overlaps.append(trade)
        else:
            additive_trades.append(trade)
    months: dict[str, dict[str, Any]] = defaultdict(lambda: {
        'trade_count': 0,
        'total_usd_raw': 0.0,
        'total_pips_raw': 0.0,
        'wins': 0,
        'losses': 0,
        'slice_ids': defaultdict(int),
    })
    for trade in additive_trades:
        key = _month_key(trade['entry_time'])
        row = months[key]
        row['trade_count'] += 1
        row['total_usd_raw'] += float(trade['usd'])
        row['total_pips_raw'] += float(trade['pips'])
        sid = str(trade.get('slice_id') or '')
        if not sid:
            cell = str(trade.get('ownership_cell'))
            if cell == 'ambiguous/er_high/der_pos':
                sid = SLICE_IDS[0]
            elif cell == 'ambiguous/er_low/der_pos':
                sid = SLICE_IDS[1]
            elif cell == 'momentum/er_high/der_pos':
                sid = SLICE_IDS[2]
        row['slice_ids'][sid] += 1
        if float(trade['usd']) > 0:
            row['wins'] += 1
        else:
            row['losses'] += 1
    out = {}
    for month, row in sorted(months.items()):
        total_usd = round(row['total_usd_raw'], 2)
        total_pips = round(row['total_pips_raw'], 2)
        out[month] = {
            'trade_count': row['trade_count'],
            'total_usd_raw': total_usd,
            'total_pips_raw': total_pips,
            'avg_usd_raw': round(total_usd / row['trade_count'], 2),
            'avg_pips_raw': round(total_pips / row['trade_count'], 3),
            'wins': row['wins'],
            'losses': row['losses'],
            'slice_mix': dict(sorted(row['slice_ids'].items())),
        }
    return {
        'exact_overlap_count': len(overlaps),
        'new_additive_trade_count': len(additive_trades),
        'months': out,
        'top_months_by_usd_raw': sorted(
            ({'month': m, **v} for m, v in out.items()),
            key=lambda x: x['total_usd_raw'],
            reverse=True,
        )[:10],
        'worst_months_by_usd_raw': sorted(
            ({'month': m, **v} for m, v in out.items()),
            key=lambda x: x['total_usd_raw'],
        )[:10],
    }


def _by_slice_contribution() -> dict[str, Any]:
    family = _load_json(FAMILY_JSON)
    variants = {v['name']: v for v in family['variants']}
    validations = {sid: _load_json(path) for sid, path in VALIDATION_JSONS.items()}
    out = {'individual': {}, 'marginal_in_family_order': {}}
    for sid, data in validations.items():
        out['individual'][sid] = {}
        for dk in ['500k', '1000k']:
            ds = data['datasets'][dk]
            scale = ds['size_sweep']['100pct']
            out['individual'][sid][dk] = {
                'standalone': ds['standalone'],
                'coverage': ds['coverage'],
                'selection_counts': scale['selection_counts'],
                'delta_vs_baseline': scale['delta_vs_baseline'],
            }
    chain = [('top1', None), ('top2', 'top1'), ('top3', 'top2')]
    for name, prev in chain:
        out['marginal_in_family_order'][name] = {}
        for dk in ['500k', '1000k']:
            cur = variants[name]['datasets'][dk]['size_sweep']['100pct']['delta_vs_baseline']
            if prev is None:
                marginal = cur
            else:
                prv = variants[prev]['datasets'][dk]['size_sweep']['100pct']['delta_vs_baseline']
                marginal = {
                    'total_trades': cur['total_trades'] - prv['total_trades'],
                    'net_usd': round(cur['net_usd'] - prv['net_usd'], 2),
                    'profit_factor': round(cur['profit_factor'] - prv['profit_factor'], 4),
                    'max_drawdown_usd': round(cur['max_drawdown_usd'] - prv['max_drawdown_usd'], 2),
                }
            out['marginal_in_family_order'][name][dk] = marginal
    return out


def _deployment_recommendation() -> dict[str, Any]:
    family = _load_json(FAMILY_JSON)
    top3 = next(v for v in family['variants'] if v['name'] == 'top3')
    scales = {}
    for scale in ['50pct', '75pct', '100pct']:
        d500 = top3['datasets']['500k']['size_sweep'][scale]['delta_vs_baseline']
        d1k = top3['datasets']['1000k']['size_sweep'][scale]['delta_vs_baseline']
        scales[scale] = {
            '500k': d500,
            '1000k': d1k,
            'combined_net_usd': round(d500['net_usd'] + d1k['net_usd'], 2),
            'combined_pf_delta': round(d500['profit_factor'] + d1k['profit_factor'], 4),
            'worst_dd_delta': round(max(d500['max_drawdown_usd'], d1k['max_drawdown_usd']), 2),
        }
    recommendation = {
        'recommended_scale': '75pct',
        'reason': '75pct captures most of the family upside while materially limiting the 500k drawdown increase versus 100pct. 100pct is the backtest-maximizing choice; 75pct is the cleaner deployment balance.',
        'aggressive_option': '100pct',
        'conservative_option': '50pct',
        'scales': scales,
    }
    return recommendation


def _build_md(payload: dict[str, Any]) -> str:
    lines = ['# V44 Sell Family Analytics', '']
    lines += ['## By-Slice Contribution', '']
    for sid, by_ds in payload['by_slice_contribution']['individual'].items():
        lines.append(f'- `{sid}`')
        for dk in ['500k', '1000k']:
            ds = by_ds[dk]
            delta = ds['delta_vs_baseline']
            lines.append(
                f"  {dk}: standalone avg pips `{ds['standalone']['avg_pips']}`, PF `{ds['standalone']['profit_factor']}`, "
                f"delta USD `{delta['net_usd']}`, delta PF `{delta['profit_factor']}`, delta DD `{delta['max_drawdown_usd']}`"
            )
        lines.append('')
    lines += ['## Marginal Family Contribution', '']
    for name, by_ds in payload['by_slice_contribution']['marginal_in_family_order'].items():
        lines.append(f'- `{name}`')
        for dk in ['500k', '1000k']:
            ds = by_ds[dk]
            lines.append(
                f"  {dk}: trades `{ds['total_trades']}`, delta USD `{ds['net_usd']}`, delta PF `{ds['profit_factor']}`, delta DD `{ds['max_drawdown_usd']}`"
            )
        lines.append('')
    lines += ['## Month-by-Month Contribution', '']
    for dk in ['500k', '1000k']:
        lines.append(f'- `{dk}` top months by raw added-trade USD:')
        for row in payload['month_by_month'][dk]['top_months_by_usd_raw'][:5]:
            lines.append(
                f"  {row['month']}: trades `{row['trade_count']}`, raw USD `{row['total_usd_raw']}`, raw pips `{row['total_pips_raw']}`"
            )
        lines.append(f'- `{dk}` worst months by raw added-trade USD:')
        for row in payload['month_by_month'][dk]['worst_months_by_usd_raw'][:5]:
            lines.append(
                f"  {row['month']}: trades `{row['trade_count']}`, raw USD `{row['total_usd_raw']}`, raw pips `{row['total_pips_raw']}`"
            )
        lines.append('')
    rec = payload['deployment_recommendation']
    lines += ['## Deployment Recommendation', '']
    lines.append(f"- recommended: `{rec['recommended_scale']}`")
    lines.append(f"- reason: {rec['reason']}")
    lines.append(f"- aggressive option: `{rec['aggressive_option']}`")
    lines.append(f"- conservative option: `{rec['conservative_option']}`")
    return '\n'.join(lines) + '\n'


def main() -> int:
    specs, selected = _family_top3_selected()
    payload = {
        'title': 'V44 sell family analytics',
        'family_slice_ids': [s.resolved_slice_id() for s in specs],
        'by_slice_contribution': _by_slice_contribution(),
        'month_by_month': {
            dk: _monthly_added_trade_contrib(dk, selected[dk]) for dk in ['500k', '1000k']
        },
        'deployment_recommendation': _deployment_recommendation(),
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2, default=discovery._json_default), encoding='utf-8')
    OUTPUT_MD.write_text(_build_md(payload), encoding='utf-8')
    print(OUTPUT_JSON)
    print(OUTPUT_MD)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
