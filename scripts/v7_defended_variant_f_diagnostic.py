#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/Users/codygnon/Documents/usdjpy_assistant')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import backtest_v44_conservative_router as v44_router
from scripts import backtest_tokyo_meanrev as tokyo_bt
from scripts import backtest_variant_i_pbt_standdown as variant_i
from scripts import diagnostic_chart_authorization_loop as auth_loop
from scripts import run_offensive_slice_discovery as discovery
from core.ownership_table import cell_key_from_floats
from scripts.backtest_merged_integrated_tokyo_london_v2_ny import TradeRow
from core.phase3_variant_k_baseline import build_variant_k_baseline_context
OUT = ROOT / 'research_out' / 'v7_defended_variant_f_diagnostic.json'
DATASET = str(discovery.DATASETS['1000k'])
REPORT = ROOT / 'research_out' / 'phase1_v44_baseline_1000k_report.json'


def _ts(x: object) -> pd.Timestamp:
    t = pd.Timestamp(x)
    if t.tzinfo is None:
        t = t.tz_localize('UTC')
    return t.tz_convert('UTC')


def _lookup_bar_cell(ts: pd.Timestamp, bf_times: np.ndarray, bf_reg: list[str], bf_er: list[float], bf_der: list[float]) -> tuple[str, float, float, str]:
    key = np.datetime64(pd.Timestamp(ts).asm8.astype('datetime64[ns]'))
    idx = int(np.searchsorted(bf_times, key, side='right') - 1)
    if idx < 0:
        reg, er, der = 'ambiguous', 0.5, 0.0
    else:
        reg, er, der = str(bf_reg[idx]), float(bf_er[idx]), float(bf_der[idx])
    return reg, er, der, cell_key_from_floats(reg, er, der)


def main() -> int:
    oracle_rows = json.loads(REPORT.read_text(encoding='utf-8')).get('results', {}).get('closed_trades', [])
    # Bar-by-bar inputs
    bar_frame = auth_loop._load_bar_frame(DATASET)
    if 'ownership_cell' not in bar_frame.columns:
        bar_frame = bar_frame.copy()
        bar_frame['ownership_cell'] = [
            cell_key_from_floats(str(rg), float(er), float(de))
            for rg, er, de in zip(bar_frame['regime_hysteresis'], bar_frame['sf_er'], bar_frame['delta_er'], strict=False)
        ]
    bf_times = pd.to_datetime(bar_frame['time'], utc=True).values.astype('datetime64[ns]')
    bf_reg = bar_frame['regime_hysteresis'].astype(str).tolist()
    bf_er = bar_frame['sf_er'].astype(float).tolist()
    bf_der = bar_frame['delta_er'].astype(float).tolist()

    m1 = tokyo_bt.load_m1(DATASET)
    m5 = (
        m1.set_index('time').resample('5min', label='right', closed='right')
        .agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last'))
        .dropna().reset_index()
    )
    vk_ctx = build_variant_k_baseline_context({'M1': m1, 'M5': m5})

    # Pipeline-equivalent inputs
    baseline = v44_router._build_baseline(
        DATASET,
        ROOT / 'research_out' / 'tokyo_optimized_v14_config.json',
        ROOT / 'research_out' / 'v2_exp4_winner_baseline_config.json',
        ROOT / 'research_out' / 'session_momentum_v44_base_config.json',
        100000.0,
    )
    classified_dynamic = variant_i._load_classified_with_dynamic(DATASET)
    dyn_idx = pd.DatetimeIndex(classified_dynamic['time'])
    baseline_v44 = [t for t in baseline['trades'] if t.strategy == 'v44_ny']
    by_et = {pd.Timestamp(t.entry_time).value: t for t in baseline_v44}

    disagreements: list[dict[str, object]] = []
    reg_mm = er_mm = der_mm = cell_mm = verdict_mm = 0
    bar_blocks = pipe_blocks = 0
    reason_dist = Counter()

    for row in oracle_rows:
        et = _ts(row['entry_time'])
        side = str(row.get('side', 'buy')).lower()

        # bar-by-bar verdict (exact current implementation uses dummy buy/raw)
        dummy = TradeRow(
            strategy='v44_ny', entry_time=et, exit_time=et, entry_session='v44_ny', side='buy',
            pips=0.0, usd=0.0, exit_reason='x', standalone_entry_equity=100000.0, raw={}, size_scale=1.0,
        )
        br = v44_router._filter_v44_trade(
            dummy, vk_ctx.classified_basic, vk_ctx.m5_basic,
            block_breakout=True, block_post_breakout=True, block_ambiguous_non_momentum=True,
            momentum_only=False, exhaustion_gate=False, soft_exhaustion=False,
            er_threshold=0.35, decay_threshold=0.40,
        )
        bar_reg, bar_er, bar_der, bar_cell = _lookup_bar_cell(et, bf_times, bf_reg, bf_er, bf_der)
        bar_verdict = 'BLOCK' if br.blocked else 'PASS'
        if br.blocked:
            bar_blocks += 1
            if 'breakout' in br.reason:
                reason_dist['block_breakout'] += 1
            elif 'post_breakout' in br.reason:
                reason_dist['block_post_breakout'] += 1
            elif 'ambiguous' in br.reason:
                reason_dist['block_ambiguous_non_momentum'] += 1
            else:
                reason_dist['other'] += 1

        # pipeline-equivalent verdict (real trade row from baseline)
        bt = by_et.get(et.value)
        if bt is None:
            bt = TradeRow(
                strategy='v44_ny', entry_time=et, exit_time=et, entry_session='v44_ny', side=side,
                pips=0.0, usd=0.0, exit_reason='oracle_missing_in_baseline', standalone_entry_equity=100000.0,
                raw={'entry_profile': str(row.get('entry_profile', ''))}, size_scale=1.0,
            )
        pr = v44_router._filter_v44_trade(
            bt, vk_ctx.classified_basic, vk_ctx.m5_basic,
            block_breakout=True, block_post_breakout=True, block_ambiguous_non_momentum=True,
            momentum_only=False, exhaustion_gate=False, soft_exhaustion=False,
            er_threshold=0.35, decay_threshold=0.40,
        )
        pinfo = variant_i._lookup_regime_with_dynamic(classified_dynamic, dyn_idx, bt.entry_time)
        pidx = dyn_idx.get_indexer([pd.Timestamp(bt.entry_time)], method='ffill')[0]
        prow = classified_dynamic.iloc[pidx] if pidx >= 0 else {}
        p_er = float(prow.get('sf_er', 0.5) if hasattr(prow, 'get') else 0.5)
        if np.isnan(p_er):
            p_er = 0.5
        p_der = float(pinfo.get('delta_er', 0.0))
        p_reg = str(pinfo.get('regime_label', 'ambiguous'))
        p_cell = cell_key_from_floats(p_reg, p_er, p_der)
        pipe_verdict = 'BLOCK' if pr.blocked else 'PASS'
        if pr.blocked:
            pipe_blocks += 1

        reg_match = bar_reg == p_reg
        er_match = abs(bar_er - p_er) < 0.01
        der_match = abs(bar_der - p_der) < 0.01
        cell_match = bar_cell == p_cell
        verdict_match = bar_verdict == pipe_verdict

        reg_mm += 0 if reg_match else 1
        er_mm += 0 if er_match else 1
        der_mm += 0 if der_match else 1
        cell_mm += 0 if cell_match else 1
        verdict_mm += 0 if verdict_match else 1

        if not verdict_match:
            disagreements.append(
                {
                    'entry_time': et.isoformat(),
                    'side': side,
                    'bar_regime': bar_reg,
                    'pipeline_regime': p_reg,
                    'bar_er': round(bar_er, 6),
                    'pipeline_er': round(p_er, 6),
                    'bar_delta_er': round(bar_der, 6),
                    'pipeline_delta_er': round(p_der, 6),
                    'bar_cell': bar_cell,
                    'pipeline_cell': p_cell,
                    'bar_verdict': bar_verdict,
                    'pipeline_verdict': pipe_verdict,
                    'bar_block_reason': br.reason if br.blocked else '',
                    'pipeline_block_reason': pr.reason if pr.blocked else '',
                }
            )

    total = len(oracle_rows)
    out = {
        'total_v44_oracle_entries': total,
        'bar_by_bar_variant_f_blocks': bar_blocks,
        'pipeline_equivalent_blocks': pipe_blocks,
        'verdict_agreement_rate': f"{(100.0 * (total - verdict_mm) / total if total else 0.0):.2f}%",
        'disagreements': disagreements,
        'regime_mismatch_count': reg_mm,
        'er_mismatch_count': er_mm,
        'der_mismatch_count': der_mm,
        'cell_mismatch_count': cell_mm,
        'variant_f_block_reason_distribution': {
            'block_breakout': int(reason_dist.get('block_breakout', 0)),
            'block_post_breakout': int(reason_dist.get('block_post_breakout', 0)),
            'block_ambiguous_non_momentum': int(reason_dist.get('block_ambiguous_non_momentum', 0)),
            'other': int(reason_dist.get('other', 0)),
        },
    }
    OUT.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f'wrote {OUT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
