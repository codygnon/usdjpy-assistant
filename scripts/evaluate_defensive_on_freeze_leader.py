#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path('/Users/codygnon/Documents/usdjpy_assistant')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import backtest_offensive_slice_additive as additive
from scripts import backtest_defensive_v15_pocket_grid as def_pocket
from scripts import backtest_merged_integrated_tokyo_london_v2_ny as merged_engine
from scripts import backtest_variant_k_london_cluster as variant_k
from scripts import package_freeze_closeout_lib as lib
from scripts import run_offensive_slice_discovery as discovery
from scripts import run_package_slice_exit_frontier as exit_frontier

OUT_JSON = ROOT / 'research_out' / 'defensive_on_freeze_leader.json'
OUT_MD = ROOT / 'research_out' / 'defensive_on_freeze_leader.md'
TARGET_LABEL = 'L1_mom_low_pos_buy'
BLOCK_STRATEGY = 'v44_ny'
BLOCK_CELL = 'ambiguous/er_low/der_neg'
DROP_WEEKDAYS = {'Monday', 'Tuesday'}
L1_TP1R = 3.25
L1_BE = 1.0
L1_TP2R = 2.0
PACKAGES = ['v7_usd_max', 'v7_pfdd']


def _ts(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize('UTC')
    return ts.tz_convert('UTC')


def _weekday_ok(trade: dict[str, Any]) -> bool:
    return str(lib.trade_attr(trade, 'weekday')) not in DROP_WEEKDAYS


def _build_followup_replacement(context: lib.Context, ds: str) -> list[dict[str, Any]]:
    baseline_replacement = [t for t in context['trades_by_ds'][ds][TARGET_LABEL] if _weekday_ok(t)]
    result = exit_frontier._run_london_config(L1_TP1R, L1_BE, L1_TP2R, discovery.DATASETS[ds])
    normalized = exit_frontier._normalize_london_results(result, discovery.DATASETS[ds])
    spec = context['specs'][TARGET_LABEL]
    selected = [t for t in normalized if discovery._passes_filters(t, spec) and _weekday_ok(t)]
    selected_map = {exit_frontier._entry_key(t): t for t in selected}
    out = []
    for base_trade in baseline_replacement:
        key = exit_frontier._entry_key(base_trade)
        out.append(selected_map.get(key, base_trade))
    return out


def _build_defensive_baseline_ctx(dataset: str) -> tuple[additive.BaselineContext, dict[str, Any]]:
    kept_k, baseline_meta, classified_dynamic, dyn_time_idx, _, _ = variant_k.build_variant_k_pre_coupling_kept(dataset)
    kept_blocked = []
    blocked = []
    for t in kept_k:
        cell = def_pocket._trade_cell(t, classified_dynamic, dyn_time_idx)
        if t.strategy == BLOCK_STRATEGY and cell == BLOCK_CELL:
            blocked.append(t)
        else:
            kept_blocked.append(t)
    coupled = merged_engine._apply_shared_equity_coupling(
        sorted(kept_blocked, key=lambda t: (t.exit_time, t.entry_time)),
        additive.STARTING_EQUITY,
        v14_max_units=baseline_meta['v14_max_units'],
    )
    eq = merged_engine._build_equity_curve(coupled, additive.STARTING_EQUITY)
    summary = merged_engine._stats(coupled, additive.STARTING_EQUITY, eq)
    ctx = additive.BaselineContext(
        dataset=str(Path(dataset).resolve()),
        baseline_kept=kept_blocked,
        baseline_coupled=coupled,
        baseline_summary=summary,
        baseline_meta=baseline_meta,
    )
    blocked_usd = round(sum(float(t.usd) for t in blocked), 2)
    blocked_pips = round(sum(float(t.pips) for t in blocked), 2)
    return ctx, {
        'blocked_count': len(blocked),
        'blocked_winners': sum(1 for t in blocked if float(t.pips) > 0),
        'blocked_losers': sum(1 for t in blocked if float(t.pips) <= 0),
        'blocked_net_usd': blocked_usd,
        'blocked_net_pips': blocked_pips,
    }


def _package_result(
    context: lib.Context,
    package_name: str,
    baseline_ctx_by_ds: dict[str, additive.BaselineContext],
    followup_replacement_by_ds: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    scales = lib.package_scales(package_name)
    trades_by_ds = {}
    for ds in ['500k', '1000k']:
        trades_by_label = dict(context['trades_by_ds'][ds])
        trades_by_label[TARGET_LABEL] = followup_replacement_by_ds[ds]
        trades_by_ds[ds] = lib.scaled_combined_trades(scales, trades_by_label)

    datasets = {}
    for ds in ['500k', '1000k']:
        result = additive.run_slice_additive_with_policy(
            baseline_ctx=baseline_ctx_by_ds[ds],
            slice_spec={
                'variant': f'{package_name}__followup__L1_drop_Monday_Tuesday__tp1r_3.25__be_1__tp2r_2',
                'cell_scales': scales,
            },
            selected_trades=trades_by_ds[ds],
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
        'name': package_name,
        'datasets': datasets,
        'combined_delta_usd': round(float(d5['net_usd']) + float(d1['net_usd']), 2),
        'combined_delta_pf': round(float(d5['profit_factor']) + float(d1['profit_factor']), 4),
        'combined_delta_dd': round(float(d5['max_drawdown_usd']) + float(d1['max_drawdown_usd']), 2),
        'passes_strict': bool(d5['net_usd'] > 0 and d1['net_usd'] > 0 and d5['profit_factor'] >= 0 and d1['profit_factor'] >= 0),
    }


def _build_md(payload: dict[str, Any]) -> str:
    lines = [
        '# Defensive Lever On Freeze Leader',
        '',
        '- Applies the narrow defensive lever `v44_ny @ ambiguous/er_low/der_neg` on top of the tuned `v7` follow-up leader candidates.',
        '- Follow-up package assumptions: drop `L1` on `Monday,Tuesday`; use `tp1r=3.25`, `be=1.0`, `tp2r=2.0` for `L1`.',
        '',
    ]
    for ds, stats in payload['defensive_blocked'].items():
        lines.append(f"- defensive block {ds}: blocked `{stats['blocked_count']}` trades | USD `{stats['blocked_net_usd']}` | pips `{stats['blocked_net_pips']}`")
    lines.append('')
    for row in payload['packages']:
        base = row['base_followup']
        defended = row['with_defensive']
        diff = row['delta_vs_base_followup']
        lines.append(f"## {row['package']}")
        lines.append('')
        lines.append(f"- base follow-up: USD `{base['combined_delta_usd']}`, PF `{base['combined_delta_pf']}`, DD `{base['combined_delta_dd']}`")
        lines.append(f"- with defensive: USD `{defended['combined_delta_usd']}`, PF `{defended['combined_delta_pf']}`, DD `{defended['combined_delta_dd']}`")
        lines.append(f"- delta vs base follow-up: USD `{diff['combined_delta_usd']}`, PF `{diff['combined_delta_pf']}`, DD `{diff['combined_delta_dd']}`")
        lines.append('')
    best = payload.get('best_move')
    if best:
        lines.append(f"- best move: `{best['package']}` | dUSD `{best['delta_vs_base_followup']['combined_delta_usd']}` | dPF `{best['delta_vs_base_followup']['combined_delta_pf']}` | dDD `{best['delta_vs_base_followup']['combined_delta_dd']}`")
        lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> int:
    context = lib.load_context()
    base_baseline_ctx = context['baseline_ctx_by_ds']
    followup_replacement_by_ds = {
        ds: _build_followup_replacement(context, ds)
        for ds in ['500k', '1000k']
    }
    defensive_ctx_by_ds = {}
    defensive_blocked = {}
    for ds in ['500k', '1000k']:
        defensive_ctx_by_ds[ds], defensive_blocked[ds] = _build_defensive_baseline_ctx(discovery.DATASETS[ds])

    package_rows = []
    for package_name in PACKAGES:
        base_followup = _package_result(context, package_name, base_baseline_ctx, followup_replacement_by_ds)
        with_defensive = _package_result(context, package_name, defensive_ctx_by_ds, followup_replacement_by_ds)
        delta = {
            'combined_delta_usd': round(with_defensive['combined_delta_usd'] - base_followup['combined_delta_usd'], 2),
            'combined_delta_pf': round(with_defensive['combined_delta_pf'] - base_followup['combined_delta_pf'], 4),
            'combined_delta_dd': round(with_defensive['combined_delta_dd'] - base_followup['combined_delta_dd'], 2),
        }
        package_rows.append({
            'package': package_name,
            'base_followup': base_followup,
            'with_defensive': with_defensive,
            'delta_vs_base_followup': delta,
        })

    best_move = max(package_rows, key=lambda r: (r['delta_vs_base_followup']['combined_delta_usd'], r['delta_vs_base_followup']['combined_delta_pf'], -r['delta_vs_base_followup']['combined_delta_dd']))
    payload = {
        'title': 'Defensive lever on freeze leader',
        'defensive_rule': {'strategy': BLOCK_STRATEGY, 'cell': BLOCK_CELL},
        'l1_followup': {
            'drop_weekdays': sorted(DROP_WEEKDAYS),
            'tp1r': L1_TP1R,
            'be': L1_BE,
            'tp2r': L1_TP2R,
        },
        'defensive_blocked': defensive_blocked,
        'packages': package_rows,
        'best_move': best_move,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=lib._json_default), encoding='utf-8')
    OUT_MD.write_text(_build_md(payload), encoding='utf-8')
    print(OUT_JSON)
    print(OUT_MD)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
