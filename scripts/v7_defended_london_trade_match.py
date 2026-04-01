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
from scripts import evaluate_defensive_on_freeze_leader as freeze_leader
from scripts import package_freeze_closeout_lib as lib
from scripts import run_offensive_slice_discovery as discovery

OUT = ROOT / 'research_out' / 'v7_defended_london_trade_match.json'
TRADES = ROOT / 'research_out' / 'v7_defended_bar_by_bar_trades.csv'


def _ts(x: Any) -> pd.Timestamp:
    t = pd.Timestamp(x)
    if t.tzinfo is None:
        t = t.tz_localize('UTC')
    return t.tz_convert('UTC')


def _collect_margin_selected(combined: list[dict[str, Any]], defensive_ctx: additive.BaselineContext, policy: additive.ConflictPolicy) -> list[dict[str, Any]]:
    pol_sel, _ = additive._apply_conflict_policy_to_selected_trades(combined, policy)
    baseline_id = {additive._trade_identity_from_trade_row(t) for t in defensive_ctx.baseline_coupled}
    add_cand = [t for t in pol_sel if not additive._has_exact_baseline_match_by_identity(t, baseline_id)]
    m_sel, _ = additive._apply_margin_policy_to_candidates(
        additive_candidates=add_cand,
        baseline_trades=defensive_ctx.baseline_coupled,
        starting_equity=additive.STARTING_EQUITY,
        policy=policy,
    )
    return m_sel


def _match(bar: list[dict[str, Any]], shadow: list[dict[str, Any]]) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]]]:
    used = set()
    matched = 0
    bar_only = []
    for b in bar:
        bt = _ts(b['entry_time'])
        side = str(b['side']).lower()
        best_i = None
        best_dt = None
        for i, s in enumerate(shadow):
            if i in used:
                continue
            if str(s['side']).lower() != side:
                continue
            st = _ts(s['entry_time'])
            dt = abs((bt - st).total_seconds())
            if dt <= 60 and (best_dt is None or dt < best_dt):
                best_i = i
                best_dt = dt
        if best_i is None:
            bar_only.append(b)
        else:
            used.add(best_i)
            matched += 1
    shadow_only = [s for i, s in enumerate(shadow) if i not in used]
    return matched, bar_only, shadow_only


def _reason_bar_only(r: dict[str, Any]) -> str:
    dow = str(pd.Timestamp(r['entry_time']).day_name())
    if r.get('setup_type') == 'D' and dow in {'Monday', 'Tuesday'}:
        return 'weekday mismatch or setup_type mismatch against shadow filters'
    return 'timing/side mismatch or shadow list built from additive margin-selected trades'


def _reason_shadow_only(r: dict[str, Any]) -> str:
    return 'bar-by-bar gating/margin/causal state may reject this shadow candidate'


def main() -> int:
    bar = pd.read_csv(TRADES)
    london = bar[bar['session'] == 'london_v2'].copy()
    london['entry_time'] = pd.to_datetime(london['entry_time'], utc=True, errors='coerce')
    london['setup_type'] = london['setup_type'].astype(str)
    london['side'] = london['side'].astype(str).str.lower()
    london['direction'] = london['direction'].astype(str).str.lower()

    bar_a = [
        {'entry_time': str(t), 'setup_type': 'A', 'side': s, 'day_of_week': pd.Timestamp(t).day_name()}
        for t, s in zip(london[london['setup_type'] == 'A']['entry_time'], london[london['setup_type'] == 'A']['direction'], strict=False)
    ]
    bar_d = [
        {'entry_time': str(t), 'setup_type': 'D', 'side': s, 'day_of_week': pd.Timestamp(t).day_name()}
        for t, s in zip(london[london['setup_type'] == 'D']['entry_time'], london[london['setup_type'] == 'D']['direction'], strict=False)
    ]

    context = lib.load_context()
    followup = {'1000k': freeze_leader._build_followup_replacement(context, '1000k')}
    defensive_ctx, _ = freeze_leader._build_defensive_baseline_ctx(discovery.DATASETS['1000k'])
    scales = lib.package_scales('v7_pfdd')

    tbl = dict(context['trades_by_ds']['1000k'])
    tbl[freeze_leader.TARGET_LABEL] = followup['1000k']
    combined = lib.scaled_combined_trades(scales, tbl)
    m_sel = _collect_margin_selected(combined, defensive_ctx, context['policy'])
    sh_ldn = [t for t in m_sel if str(t.get('strategy')) == 'london_v2']

    sh_a = [{'entry_time': str(t['entry_time']), 'setup_type': 'A', 'side': str(t.get('side', '')).lower()} for t in sh_ldn if str(t.get('setup_type', '')) == 'A']
    sh_d = [{'entry_time': str(t['entry_time']), 'setup_type': 'D', 'side': str(t.get('side', '')).lower()} for t in sh_ldn if str(t.get('setup_type', '')) != 'A']

    a_m, a_bo, a_so = _match(bar_a, sh_a)
    d_m, d_bo, d_so = _match(bar_d, sh_d)

    bar_only_details = [
        {
            'entry_time': r['entry_time'],
            'setup_type': r['setup_type'],
            'side': r['side'],
            'day_of_week': r.get('day_of_week', ''),
            'possible_reason_not_in_shadow': _reason_bar_only(r),
        }
        for r in (a_bo + d_bo)
    ]
    shadow_only_details = [
        {
            'entry_time': r['entry_time'],
            'setup_type': r['setup_type'],
            'side': r['side'],
            'possible_reason_not_in_bar_by_bar': _reason_shadow_only(r),
        }
        for r in (a_so + d_so)
    ]

    out = {
        'bar_by_bar_setup_a_count': len(bar_a),
        'bar_by_bar_l1_count': len(bar_d),
        'shadow_setup_a_count': len(sh_a),
        'shadow_l1_count': len(sh_d),
        'setup_a_matched': a_m,
        'setup_a_bar_only': len(a_bo),
        'setup_a_shadow_only': len(a_so),
        'l1_matched': d_m,
        'l1_bar_only': len(d_bo),
        'l1_shadow_only': len(d_so),
        'bar_only_details': bar_only_details,
        'shadow_only_details': shadow_only_details,
    }
    OUT.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f'wrote {OUT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
