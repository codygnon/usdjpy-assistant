#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path('/Users/codygnon/Documents/usdjpy_assistant')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import backtest_offensive_slice_additive as additive
from scripts import backtest_variant_k_v7_profile_search as v7
from scripts import backtest_offensive_slice_family_combo as family_combo
from scripts import run_offensive_slice_discovery as discovery

OUT_DIR = ROOT / 'research_out'


def _json_default(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return discovery._json_default(obj)


def _build_payload(name: str, cell_labels: list[str]) -> dict[str, Any]:
    matrix = family_combo._load_matrix(Path(v7.DEFAULT_MATRIX))
    specs = v7._load_all_specs(matrix)
    missing = [label for label in cell_labels if label not in specs]
    if missing:
        raise SystemExit(f'Missing cell labels: {missing}')
    strategies = {specs[label].strategy for label in cell_labels}
    all_trades = discovery._load_all_normalized_trades(strategies)
    trades_by_ds = v7._select_all_trades(specs, all_trades)
    baseline_ctx_by_ds = {
        ds: additive.build_baseline_context(discovery.DATASETS[ds])
        for ds in ['500k', '1000k']
    }
    policy = additive.ConflictPolicy(
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
    result = v7._run_variant(name, cell_labels, trades_by_ds, policy, baseline_ctx_by_ds)
    d5 = result['datasets']['500k']['delta_vs_baseline']
    d1 = result['datasets']['1000k']['delta_vs_baseline']
    return {
        'title': 'V7 Profile Combo Candidate Validation',
        'name': name,
        'cells': cell_labels,
        'cell_slice_ids': {label: v7.ALL_REGISTRY[label] for label in cell_labels},
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
        'combined_delta_usd': round(d5['net_usd'] + d1['net_usd'], 2),
        'combined_delta_pf': round(d5['profit_factor'] + d1['profit_factor'], 4),
        'datasets': {
            ds: {
                'summary': result['datasets'][ds]['variant_summary'],
                'delta_vs_baseline': result['datasets'][ds]['delta_vs_baseline'],
                'selection_counts': result['datasets'][ds]['selection_counts'],
                'policy_stats': result['datasets'][ds].get('policy_stats', {}),
            }
            for ds in ['500k', '1000k']
        },
    }


def _build_md(payload: dict[str, Any]) -> str:
    lines = [
        '# V7 Profile Combo Candidate Validation',
        '',
        f"- name: `{payload['name']}`",
        f"- combined delta USD: `{payload['combined_delta_usd']}`",
        f"- combined delta PF: `{payload['combined_delta_pf']}`",
        f"- policy: `{payload['policy']['name']}`",
        '',
        '## Cells',
        '',
    ]
    for label in payload['cells']:
        lines.append(f"- `{label}` -> `{payload['cell_slice_ids'][label]}`")
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
            f"- raw selected: `{sel['raw_selected_trade_count']}`",
            f"- exact overlap: `{sel['exact_baseline_overlap_count']}`",
            f"- additive trades: `{sel['new_additive_trades_count']}`",
            f"- internal overlap pairs: `{sel['internal_overlap_pairs']}`",
            f"- internal opposite-side pairs: `{sel['internal_opposite_side_overlap_pairs']}`",
            '',
        ]
    return '\n'.join(lines) + '\n'


def main() -> int:
    name = 'v7_best_combo'
    cells = [
        'C1_sell_base', 'C2_sell', 'C3_buy', 'C4_sell_base', 'C5_pbt_sell', 'C6_pbt_sell',
        'O0_buy_strong', 'O1_buy_strong', 'O2_buy_strong',
        'ADJ_meanrev_low_neg_buy', 'ADJ_ambig_mid_neg_sell', 'ADJ_mom_high_neg_sell',
        'N1_brkout_low_neg_sell_strong', 'N2_brkout_low_pos_buy_strong',
        'L2_brkout_mid_neg_buy', 'T1_ambig_high_pos_buy', 'T2_brkout_mid_pos_buy',
        'C0_base', 'NEW_v14_ambig_low_pos_sell', 'NEW_v44_ambig_low_neg_sell_normal',
    ]
    payload = _build_payload(name, cells)
    out_json = OUT_DIR / 'v7_best_combo_validation.json'
    out_md = OUT_DIR / 'v7_best_combo_validation.md'
    out_json.write_text(json.dumps(payload, indent=2, default=_json_default), encoding='utf-8')
    out_md.write_text(_build_md(payload), encoding='utf-8')
    print(out_json)
    print(out_md)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
