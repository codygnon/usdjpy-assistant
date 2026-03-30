#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
import argparse

ROOT = Path('/Users/codygnon/Documents/usdjpy_assistant')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import package_freeze_closeout_lib as lib
from scripts import run_package_slice_exit_frontier as exit_frontier
from scripts import run_offensive_slice_discovery as discovery

OUT_JSON = ROOT / 'research_out' / 'l1_followup_exit_check.json'
OUT_MD = ROOT / 'research_out' / 'l1_followup_exit_check.md'
TARGET_LABEL = 'L1_mom_low_pos_buy'


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Check L1 exit settings on the Monday+Tuesday-pruned winner candidate')
    ap.add_argument('--package', default='v7_usd_max', choices=['v7_usd_max', 'v7_pfdd'])
    ap.add_argument('--drop-weekdays', default='Monday,Tuesday')
    ap.add_argument('--tp1r-values', default='1.0,1.25')
    ap.add_argument('--be-values', default='1.0,1.5')
    ap.add_argument('--output-json', default=str(OUT_JSON))
    ap.add_argument('--output-md', default=str(OUT_MD))
    return ap.parse_args()


def _weekday_ok(trade: dict, blocked: set[str]) -> bool:
    return str(lib.trade_attr(trade, 'weekday')) not in blocked


def _build_md(payload: dict) -> str:
    lines = [
        '# L1 Follow-Up Exit Check',
        '',
        f"- Checks `L1` exits on the pruned winner candidate `{payload['package']}` with weekdays dropped: `{', '.join(payload['drop_weekdays'])}`.",
        '',
        f"- baseline follow-up USD: `{payload['baseline_followup']['combined_delta_usd']}`",
        f"- baseline follow-up PF: `{payload['baseline_followup']['combined_delta_pf']}`",
        f"- baseline follow-up DD: `{payload['baseline_followup']['combined_delta_dd']}`",
        '',
    ]
    for row in payload['rows']:
        lines.append(f"- `{row['name']}`: USD `{row['combined_delta_usd']}`, PF `{row['combined_delta_pf']}`, DD `{row['combined_delta_dd']}`")
    lines.append('')
    if payload.get('best_candidate'):
        best = payload['best_candidate']
        lines.append(f"- best candidate: `{best['name']}` | dUSD `{best['delta_vs_baseline']['usd']}` | dPF `{best['delta_vs_baseline']['pf']}` | dDD `{best['delta_vs_baseline']['dd']}`")
        lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> int:
    args = parse_args()
    context = lib.load_context()
    package_name = str(args.package)
    blocked = {x.strip() for x in str(args.drop_weekdays).split(',') if x.strip()}
    tp1r_values = [float(x) for x in str(args.tp1r_values).split(',') if x.strip()]
    be_values = [float(x) for x in str(args.be_values).split(',') if x.strip()]
    scales = lib.package_scales(package_name)

    baseline_replacement = {
        ds: [t for t in context['trades_by_ds'][ds][TARGET_LABEL] if _weekday_ok(t, blocked)]
        for ds in ['500k', '1000k']
    }
    baseline_followup = exit_frontier._replace_target_trades(scales, context, TARGET_LABEL, baseline_replacement)
    baseline_followup['name'] = f'{package_name}__followup__L1_drop_{"_".join(sorted(blocked))}__baseline_exit'

    rows = []
    for tp1_r in tp1r_values:
        for be_offset in be_values:
            replacement_by_ds = {}
            for ds in ['500k', '1000k']:
                result = exit_frontier._run_london_config(tp1_r, be_offset, 2.0, discovery.DATASETS[ds])
                normalized = exit_frontier._normalize_london_results(result, discovery.DATASETS[ds])
                spec = context['specs'][TARGET_LABEL]
                selected = [t for t in normalized if discovery._passes_filters(t, spec) and _weekday_ok(t, blocked)]
                selected_map = {exit_frontier._entry_key(t): t for t in selected}
                replacement = []
                for base_trade in baseline_replacement[ds]:
                    key = exit_frontier._entry_key(base_trade)
                    replacement.append(selected_map.get(key, base_trade))
                replacement_by_ds[ds] = replacement
            evaluated = exit_frontier._replace_target_trades(scales, context, TARGET_LABEL, replacement_by_ds)
            evaluated['name'] = f'{package_name}__followup__L1_drop_{"_".join(sorted(blocked))}__tp1r_{tp1_r:g}__be_{be_offset:g}__tp2r_2'
            evaluated['delta_vs_baseline'] = {
                'usd': round(evaluated['combined_delta_usd'] - baseline_followup['combined_delta_usd'], 2),
                'pf': round(evaluated['combined_delta_pf'] - baseline_followup['combined_delta_pf'], 4),
                'dd': round(evaluated['combined_delta_dd'] - baseline_followup['combined_delta_dd'], 2),
            }
            rows.append(evaluated)

    rows.sort(key=lambda r: (r['combined_delta_usd'], r['combined_delta_pf'], -r['combined_delta_dd']), reverse=True)
    best = rows[0] if rows else None
    payload = {
        'title': 'L1 follow-up exit check',
        'package': package_name,
        'drop_weekdays': sorted(blocked),
        'baseline_followup': {
            'name': baseline_followup['name'],
            'combined_delta_usd': baseline_followup['combined_delta_usd'],
            'combined_delta_pf': baseline_followup['combined_delta_pf'],
            'combined_delta_dd': baseline_followup['combined_delta_dd'],
        },
        'rows': [
            {
                'name': r['name'],
                'combined_delta_usd': r['combined_delta_usd'],
                'combined_delta_pf': r['combined_delta_pf'],
                'combined_delta_dd': r['combined_delta_dd'],
                'delta_vs_baseline': r['delta_vs_baseline'],
            }
            for r in rows
        ],
        'best_candidate': None if best is None else {
            'name': best['name'],
            'combined_delta_usd': best['combined_delta_usd'],
            'combined_delta_pf': best['combined_delta_pf'],
            'combined_delta_dd': best['combined_delta_dd'],
            'delta_vs_baseline': best['delta_vs_baseline'],
        },
    }
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    out_md.write_text(_build_md(payload), encoding='utf-8')
    print(out_json)
    print(out_md)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
