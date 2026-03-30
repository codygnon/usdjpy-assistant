#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

ROOT = Path('/Users/codygnon/Documents/usdjpy_assistant')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import backtest_offensive_slice_additive as additive
from scripts import backtest_offensive_setupd_additive as setupd_additive
from scripts import backtest_session_momentum as sm
from scripts import diagnostic_london_setupd_trade_outcomes as london_outcomes
from scripts import package_freeze_closeout_lib as lib
from scripts import run_offensive_slice_discovery as discovery

OUT_JSON = lib.OUT_DIR / 'package_slice_exit_frontier.json'
OUT_MD = lib.OUT_DIR / 'package_slice_exit_frontier.md'
BASE_CONFIG_PATH = lib.OUT_DIR / 'session_momentum_v44_base_config.json'
LONDON_CFG_PATH = lib.OUT_DIR / 'v2_exp4_winner_baseline_config.json'
TARGET_LABELS = ['C0_base', 'N1_brkout_low_neg_sell_strong', 'N2_brkout_low_pos_buy_strong', 'L1_mom_low_pos_buy']


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Close slice-specific exit frontier for package freeze research')
    ap.add_argument('--output-json', default=str(OUT_JSON))
    ap.add_argument('--output-md', default=str(OUT_MD))
    ap.add_argument('--package', action='append', dest='packages')
    ap.add_argument('--target-label', action='append', dest='target_labels')
    ap.add_argument('--tp1-values', default='')
    ap.add_argument('--be-values', default='')
    ap.add_argument('--trail-values', default='')
    ap.add_argument('--runner-modes', default='')
    ap.add_argument('--l1-tp1r-values', default='')
    ap.add_argument('--l1-be-values', default='')
    ap.add_argument('--l1-tp2r-values', default='')
    return ap.parse_args()


def _ts(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize('UTC')
    return ts.tz_convert('UTC')


def _build_args(config: dict[str, Any], dataset: str) -> SimpleNamespace:
    merged = dict(sm.DEFAULTS)
    merged.update(config)
    merged.setdefault('version', 'v5')
    merged.setdefault('mode', 'session')
    merged['inputs'] = [str(Path(dataset))]
    return SimpleNamespace(**merged)


def _run_v44_config(config: dict[str, Any], dataset: str) -> dict[str, Any]:
    args = _build_args(config, dataset)
    return sm.run_backtest_v5(args)


def _build_london_args(tp1_r: float, be_offset: float, tp2_r: float) -> SimpleNamespace:
    return SimpleNamespace(
        spread_pips=0.3,
        sl_buffer_pips=3.0,
        sl_min_pips=5.0,
        sl_max_pips=20.0,
        tp1_r_multiple=float(tp1_r),
        tp2_r_multiple=float(tp2_r),
        tp1_close_fraction=0.5,
        be_offset_pips=float(be_offset),
    )


def _run_london_config(tp1_r: float, be_offset: float, tp2_r: float, dataset: str) -> dict[str, Any]:
    return london_outcomes.run_dataset(dataset, _build_london_args(tp1_r, be_offset, tp2_r))


def _ownership_info(dataset: str, timestamps: list[pd.Timestamp]) -> list[dict[str, Any]]:
    if not timestamps:
        return []
    return discovery._classify_times_to_cells(dataset, timestamps)


def _normalize_v44_results(results: dict[str, Any], dataset: str) -> list[dict[str, Any]]:
    raw = results.get('closed_trades')
    if raw is None and 'results' in results:
        raw = results['results'].get('closed_trades', [])
    raw = raw or []
    entries = [_ts(t['entry_time']) for t in raw]
    cells = _ownership_info(dataset, entries)
    out = []
    for row, cell in zip(raw, cells):
        entry_ts = _ts(row['entry_time'])
        session = 'ny' if str(row.get('entry_session')) == 'ny_overlap' else str(row.get('entry_session', 'ny'))
        minutes = discovery._session_open_minutes(entry_ts, session)
        direction = 'long' if str(row['side']).lower() == 'buy' else 'short'
        out.append({
            'strategy': 'v44_ny',
            'source_mode': 'v44_trial10_exit_replay',
            'dataset_key': Path(dataset).name,
            'entry_time': entry_ts.isoformat(),
            'exit_time': _ts(row['exit_time']).isoformat(),
            'signal_time': entry_ts.isoformat(),
            'entry_session': session,
            'side': str(row['side']),
            'direction': direction,
            'pips': float(row['pips']),
            'usd': float(row['usd']),
            'exit_reason': str(row.get('exit_reason', 'unknown')),
            'ownership_cell': str(cell['ownership_cell']),
            'setup_type': None,
            'evaluator_mode': str(row.get('entry_signal_mode', 'trial10_exit')),
            'timing_gate': discovery._timing_gate_for_minutes(minutes, {'strategy': 'v44_ny'}),
            'timing_minutes': round(float(minutes), 2) if minutes is not None else None,
            'standalone_entry_equity': 100000.0,
            'raw': row,
            'entry_signal_mode': str(row.get('entry_signal_mode', '')),
            'entry_profile': str(row.get('entry_profile', '')),
            'entry_regime': str(row.get('entry_regime', '')),
            'entry_mode': int(row.get('entry_mode', 0) or 0),
            'size_scale': float(row.get('conviction_scale', row.get('wr_size_scale', 1.0)) or 1.0),
        })
    return out


def _normalize_london_results(results: dict[str, Any], dataset: str) -> list[dict[str, Any]]:
    raw = results.get('_all_trades', []) or []
    london_cfg = setupd_additive._load_london_cfg()
    out = []
    for t in raw:
        if not t.get('native_allowed', False):
            continue
        entry_ts = _ts(t['entry_time'])
        signal_ts = _ts(t.get('signal_time', t['entry_time']))
        minutes = discovery._session_open_minutes(signal_ts, 'london')
        usd, units = setupd_additive._calc_setupd_usd(t, london_cfg, size_scale=1.0)
        out.append({
            'strategy': 'london_v2',
            'source_mode': 'london_setupd_exit_replay',
            'dataset_key': Path(dataset).name,
            'entry_time': entry_ts.isoformat(),
            'exit_time': _ts(t['exit_time']).isoformat(),
            'signal_time': signal_ts.isoformat(),
            'entry_session': 'london',
            'side': 'buy' if str(t['direction']) == 'long' else 'sell',
            'direction': str(t['direction']),
            'pips': float(t['pnl_pips']),
            'usd': float(usd),
            'exit_reason': str(t['exit_reason']),
            'ownership_cell': str(t['ownership_cell']),
            'setup_type': 'D',
            'evaluator_mode': 'setup_d_trade_outcome_sim',
            'timing_gate': discovery._timing_gate_for_minutes(minutes, {'strategy': 'london_v2'}),
            'timing_minutes': round(float(minutes), 2) if minutes is not None else None,
            'native_allowed': True,
            'standalone_entry_equity': 100000.0,
            'raw': {**dict(t), 'position_units': int(units)},
            'size_scale': 1.0,
        })
    return out


def _entry_key(trade: dict[str, Any]) -> tuple[str, ...]:
    if str(trade.get('strategy')) == 'london_v2':
        return (
            str(trade['entry_time']),
            str(trade.get('side', '')),
            str(trade.get('ownership_cell', '')),
            str(trade.get('setup_type', '')),
            str(trade.get('timing_gate', '')),
        )
    return (
        str(trade['entry_time']),
        str(trade.get('side', '')),
        str(trade.get('entry_profile', '')),
        str(trade.get('entry_signal_mode', '')),
        str(trade.get('entry_regime', '')),
    )


def _replace_target_trades(
    package_scales: dict[str, float],
    context: lib.Context,
    target_label: str,
    replacement_by_ds: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    datasets = {}
    for ds in ['500k', '1000k']:
        trades_by_label = dict(context['trades_by_ds'][ds])
        trades_by_label[target_label] = replacement_by_ds[ds]
        combined = lib.scaled_combined_trades(package_scales, trades_by_label)
        result = additive.run_slice_additive_with_policy(
            baseline_ctx=context['baseline_ctx_by_ds'][ds],
            slice_spec={'variant': f'exit_replace_{target_label}', 'cell_scales': package_scales},
            selected_trades=combined,
            conflict_policy=context['policy'],
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
        'datasets': datasets,
        'combined_delta_usd': round(d5['net_usd'] + d1['net_usd'], 2),
        'combined_delta_pf': round(d5['profit_factor'] + d1['profit_factor'], 4),
        'combined_delta_dd': round(d5['max_drawdown_usd'] + d1['max_drawdown_usd'], 2),
        'passes_strict': bool(d5['net_usd'] > 0 and d1['net_usd'] > 0 and d5['profit_factor'] >= 0 and d1['profit_factor'] >= 0),
    }


def _grid_from_base(base_cfg: dict[str, Any], args: argparse.Namespace) -> tuple[list[float], list[float], list[float], list[str]]:
    base_tp1 = float(base_cfg.get('v5_trial10_exit_tp1_pips', 6.0))
    base_be = float(base_cfg.get('v5_trial10_exit_be_extra_pips', 0.0))
    base_trail = float(base_cfg.get('v5_trial10_exit_trail_buffer_pips', 3.0))
    base_mode = str(base_cfg.get('v5_trial10_exit_runner_mode', 'fixed_tp2_then_trail'))
    tp1 = [max(1.0, base_tp1 - 1.0), base_tp1, base_tp1 + 1.0] if not args.tp1_values else [float(x) for x in args.tp1_values.split(',') if x]
    be = sorted({max(0.0, round(base_be - 0.25, 2)), round(base_be, 2), round(base_be + 0.25, 2)}) if not args.be_values else [float(x) for x in args.be_values.split(',') if x]
    trail = [max(0.5, base_trail - 0.5), base_trail, base_trail + 0.5] if not args.trail_values else [float(x) for x in args.trail_values.split(',') if x]
    modes = [base_mode, 'trail_only_after_tp1' if base_mode != 'trail_only_after_tp1' else 'fixed_tp2_then_trail'] if not args.runner_modes else [x for x in args.runner_modes.split(',') if x]
    return sorted(set(tp1)), sorted(set(be)), sorted(set(trail)), list(dict.fromkeys(modes))


def _l1_grid_from_args(args: argparse.Namespace) -> tuple[list[float], list[float], list[float]]:
    tp1 = [1.0, 1.25] if not args.l1_tp1r_values else [float(x) for x in args.l1_tp1r_values.split(',') if x]
    be = [1.0, 1.5] if not args.l1_be_values else [float(x) for x in args.l1_be_values.split(',') if x]
    tp2 = [2.0] if not args.l1_tp2r_values else [float(x) for x in args.l1_tp2r_values.split(',') if x]
    return sorted(set(tp1)), sorted(set(be)), sorted(set(tp2))


def _build_md(payload: dict[str, Any]) -> str:
    lines = [
        '# Package Slice Exit Frontier',
        '',
        '- Frozen-entry slice-specific exit search for V44 driver slices.',
        '- Matching is done on baseline entry keys so only exit behavior is substituted.',
        '',
    ]
    for target in payload['targets']:
        lines += [f"## {target['package']} | {target['target_label']}", '']
        if target.get('status') == 'unsupported':
            lines.append(f"- unsupported: `{target['reason']}`")
            lines.append('')
            continue
        lines.append(f"- baseline entry count 500k/1000k: `{target['baseline_entry_counts']['500k']}` / `{target['baseline_entry_counts']['1000k']}`")
        best = target.get('best_candidate')
        if best:
            lines.append(f"- best: `{best['name']}` | USD `{best['combined_delta_usd']}` | PF `{best['combined_delta_pf']}` | DD `{best['combined_delta_dd']}`")
        for row in target['top_candidates']:
            lines.append(f"- `{row['name']}`: USD `{row['combined_delta_usd']}`, PF `{row['combined_delta_pf']}`, DD `{row['combined_delta_dd']}`")
        lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> int:
    args = parse_args()
    packages = args.packages or ['v7_pfdd', 'v7_best_combo']
    target_labels = args.target_labels or list(TARGET_LABELS)
    context = lib.load_context()
    base_cfg = json.loads(BASE_CONFIG_PATH.read_text(encoding='utf-8'))
    tp1_values, be_values, trail_values, runner_modes = _grid_from_base(base_cfg, args)
    l1_tp1_values, l1_be_values, l1_tp2_values = _l1_grid_from_args(args)

    payload_targets = []
    for package_name in packages:
        package_scales = lib.package_scales(package_name)
        for target_label in target_labels:
            if target_label not in package_scales or float(package_scales.get(target_label, 0.0)) <= 0.0:
                continue
            baseline_entries_by_ds: dict[str, list[dict[str, Any]]] = {}
            for ds in ['500k', '1000k']:
                baseline_entries_by_ds[ds] = list(context['trades_by_ds'][ds][target_label])

            rows = []
            if target_label == 'L1_mom_low_pos_buy':
                for tp1_r in l1_tp1_values:
                    for be_offset in l1_be_values:
                        for tp2_r in l1_tp2_values:
                            replacement_by_ds = {}
                            entry_match_stats = {}
                            for ds in ['500k', '1000k']:
                                result = _run_london_config(tp1_r, be_offset, tp2_r, discovery.DATASETS[ds])
                                normalized = _normalize_london_results(result, discovery.DATASETS[ds])
                                spec = context['specs'][target_label]
                                selected = [t for t in normalized if discovery._passes_filters(t, spec)]
                                selected_map = {_entry_key(t): t for t in selected}
                                baseline_list = baseline_entries_by_ds[ds]
                                replacement = []
                                matched = 0
                                for base_trade in baseline_list:
                                    key = _entry_key(base_trade)
                                    if key in selected_map:
                                        replacement.append(selected_map[key])
                                        matched += 1
                                    else:
                                        replacement.append(base_trade)
                                replacement_by_ds[ds] = replacement
                                entry_match_stats[ds] = {'baseline_entries': len(baseline_list), 'matched_variant_entries': matched, 'raw_variant_selected': len(selected)}

                            evaluated = _replace_target_trades(package_scales, context, target_label, replacement_by_ds)
                            evaluated['name'] = f'{package_name}__{target_label}__tp1r_{tp1_r:g}__be_{be_offset:g}__tp2r_{tp2_r:g}'
                            evaluated['metadata'] = {
                                'package': package_name,
                                'target_label': target_label,
                                'tp1_r_multiple': tp1_r,
                                'be_offset_pips': be_offset,
                                'tp2_r_multiple': tp2_r,
                                'entry_match_stats': entry_match_stats,
                            }
                            rows.append(evaluated)
            else:
                for tp1 in tp1_values:
                    for be in be_values:
                        for trail in trail_values:
                            for mode in runner_modes:
                                variant_cfg = dict(base_cfg)
                                variant_cfg.update({
                                    'v5_trial10_exit_enabled': True,
                                    'v5_trial10_exit_tp1_pips': tp1,
                                    'v5_trial10_exit_be_extra_pips': be,
                                    'v5_trial10_exit_trail_buffer_pips': trail,
                                    'v5_trial10_exit_runner_mode': mode,
                                    'v5_trial10_exit_apply_profiles': 'Strong,Normal',
                                    'v5_trial10_exit_skip_news': False,
                                })
                                replacement_by_ds = {}
                                entry_match_stats = {}
                                for ds in ['500k', '1000k']:
                                    result = _run_v44_config(variant_cfg, discovery.DATASETS[ds])
                                    normalized = _normalize_v44_results(result, discovery.DATASETS[ds])
                                    spec = context['specs'][target_label]
                                    selected = [t for t in normalized if discovery._passes_filters(t, spec)]
                                    selected_map = {_entry_key(t): t for t in selected}
                                    baseline_list = baseline_entries_by_ds[ds]
                                    replacement = []
                                    matched = 0
                                    for base_trade in baseline_list:
                                        key = _entry_key(base_trade)
                                        if key in selected_map:
                                            replacement.append(selected_map[key])
                                            matched += 1
                                        else:
                                            replacement.append(base_trade)
                                    replacement_by_ds[ds] = replacement
                                    entry_match_stats[ds] = {'baseline_entries': len(baseline_list), 'matched_variant_entries': matched, 'raw_variant_selected': len(selected)}

                                evaluated = _replace_target_trades(package_scales, context, target_label, replacement_by_ds)
                                evaluated['name'] = f'{package_name}__{target_label}__tp1_{tp1:g}__be_{be:g}__trail_{trail:g}__{mode}'
                                evaluated['metadata'] = {
                                    'package': package_name,
                                    'target_label': target_label,
                                    'tp1_pips': tp1,
                                    'be_extra_pips': be,
                                    'trail_buffer_pips': trail,
                                    'runner_mode': mode,
                                    'entry_match_stats': entry_match_stats,
                                }
                                rows.append(evaluated)

            rows.sort(key=lambda r: (r['passes_strict'], r['combined_delta_usd'], r['combined_delta_pf'], -r['combined_delta_dd']), reverse=True)
            payload_targets.append({
                'package': package_name,
                'target_label': target_label,
                'baseline_entry_counts': {ds: len(baseline_entries_by_ds[ds]) for ds in ['500k', '1000k']},
                'best_candidate': None if not rows else {
                    'name': rows[0]['name'],
                    'combined_delta_usd': rows[0]['combined_delta_usd'],
                    'combined_delta_pf': rows[0]['combined_delta_pf'],
                    'combined_delta_dd': rows[0]['combined_delta_dd'],
                    'passes_strict': rows[0]['passes_strict'],
                    'metadata': rows[0]['metadata'],
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
        'title': 'Package slice exit frontier',
        'grid': {
            'tp1_values': tp1_values,
            'be_values': be_values,
            'trail_values': trail_values,
            'runner_modes': runner_modes,
        },
        'targets': payload_targets,
    }
    lib.write_json_md(Path(args.output_json), Path(args.output_md), payload, _build_md(payload))
    print(Path(args.output_json))
    print(Path(args.output_md))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
