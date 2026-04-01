#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import pandas as pd

ROOT = Path('/Users/codygnon/Documents/usdjpy_assistant')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import backtest_v7_defended_bar_by_bar as bbb
from scripts import backtest_tokyo_meanrev as tokyo_bt
from scripts import backtest_v2_multisetup_london as london_bt
from scripts.v7_defended_london_unified import LondonUnifiedDayState, advance_london_unified_bar, init_london_day_state
OUT = ROOT / 'research_out' / 'v7_defended_margin_diagnostic.json'


def _ts(x: Any) -> pd.Timestamp:
    t = pd.Timestamp(x)
    if t.tzinfo is None:
        t = t.tz_localize('UTC')
    return t.tz_convert('UTC')


def _dist(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {'min': 0.0, 'max': 0.0, 'mean': 0.0}
    return {'min': float(min(xs)), 'max': float(max(xs)), 'mean': float(sum(xs) / len(xs))}


def main() -> int:
    dataset_path = str(bbb.DATASETS['1000k'])
    m1_raw = tokyo_bt.load_m1(dataset_path)
    with bbb.TOKYO_CFG.open() as f:
        tokyo_cfg = json.load(f)
    with bbb.LONDON_CFG_PATH.open() as f:
        london_cfg = json.load(f)
    with bbb.V44_CFG_PATH.open() as f:
        v44_cfg = json.load(f)

    df = tokyo_bt.add_indicators(m1_raw, tokyo_cfg)
    m5 = tokyo_bt.resample_ohlc_continuous(m1_raw, '5min')

    bar_frame = bbb.auth_loop._load_bar_frame(dataset_path)
    if 'ownership_cell' not in bar_frame.columns:
        bar_frame = bar_frame.copy()
        bar_frame['ownership_cell'] = [
            bbb.cell_key_from_floats(str(rg), float(er), float(de))
            for rg, er, de in zip(bar_frame['regime_hysteresis'], bar_frame['sf_er'], bar_frame['delta_er'], strict=False)
        ]
    bf_times = pd.to_datetime(bar_frame['time'], utc=True).values.astype('datetime64[ns]')
    bf_cells = bar_frame['ownership_cell'].astype(str).tolist()
    bf_reg = bar_frame['regime_hysteresis'].astype(str).tolist()
    bf_er = bar_frame['sf_er'].astype(float).tolist()
    bf_der = bar_frame['delta_er'].astype(float).tolist()

    def lookup_cell(ts: pd.Timestamp) -> tuple[str, str, float, float]:
        key = pd.Timestamp(ts).to_datetime64()
        idx = int(bbb.np.searchsorted(bf_times, key, side='right') - 1)
        if idx < 0:
            return 'ambiguous/er_mid/der_pos', 'ambiguous', 0.5, 0.0
        return bf_cells[idx], bf_reg[idx], bf_er[idx], bf_der[idx]

    v44_oracle = bbb.load_v44_oracle(dataset_path)
    v44_risk_pct = float(v44_cfg.get('v5_risk_per_trade_pct', 0.5)) / 100.0
    vk_ctx = bbb.build_variant_k_baseline_context({'M1': m1_raw, 'M5': m5})

    def variant_f_allows_v44(entry_ts: pd.Timestamp) -> bool:
        tr = bbb.TradeRow(
            strategy='v44_ny', entry_time=_ts(entry_ts), exit_time=_ts(entry_ts), entry_session='v44_ny',
            side='buy', pips=0.0, usd=0.0, exit_reason='x', standalone_entry_equity=100000.0, raw={}, size_scale=1.0,
        )
        r = bbb.v44_router._filter_v44_trade(
            tr,
            vk_ctx.classified_basic,
            vk_ctx.m5_basic,
            block_breakout=True,
            block_post_breakout=True,
            block_ambiguous_non_momentum=True,
            momentum_only=False,
            exhaustion_gate=False,
            soft_exhaustion=False,
            er_threshold=0.35,
            decay_threshold=0.40,
        )
        return not r.blocked

    equity = 100_000.0
    margin_lev = 33.3
    max_lot = 20.0
    open_positions: list[dict[str, Any]] = []
    trade_id_box = [0]

    ld: Optional[LondonUnifiedDayState] = None
    cur_day: Optional[pd.Timestamp] = None
    day_start_idx = 0
    warmup = 200

    def margin_used() -> float:
        s = 0.0
        if ld is not None:
            s += sum(float(p.margin_required_usd) for p in ld.open_positions)
        for p in open_positions:
            if p['kind'] == 'l1' and p.get('margin_usd') is not None:
                s += float(p['margin_usd'])
            else:
                s += float(p.get('lots', 0.0)) * 100_000.0 / margin_lev
        return s

    def margin_ceiling(eq: float) -> float:
        return float(eq) * 0.5 * 1.0

    def margin_avail(eq: float) -> float:
        return margin_ceiling(eq) - margin_used()

    blocked_rows: list[dict[str, Any]] = []
    blocking_sessions = Counter()

    def causal_asian_lor(i0: int, i: int, t: pd.Timestamp):
        asian_min = float(london_cfg['levels']['asian_range_min_pips'])
        asian_max = float(london_cfg['levels']['asian_range_max_pips'])
        lor_min = float(london_cfg['levels']['lor_range_min_pips'])
        lor_max = float(london_cfg['levels']['lor_range_max_pips'])
        day_start = pd.Timestamp(df.iloc[i0]['time']).normalize()
        london_h = london_bt.uk_london_open_utc(day_start)
        london_open = day_start + pd.Timedelta(hours=london_h)
        lor_end = london_open + pd.Timedelta(minutes=15)
        chunk = df.iloc[i0 : i + 1]
        asian = chunk[(chunk['time'] >= day_start) & (chunk['time'] < london_open)]
        if asian.empty:
            return (0.0, 0.0, 0.0, False, 0.0, 0.0, 0.0, False)
        ah = float(asian['high'].max())
        al = float(asian['low'].min())
        ar = (ah - al) / bbb.PIP_SIZE
        av = asian_min <= ar <= asian_max
        lor = chunk[(chunk['time'] >= london_open) & (chunk['time'] < lor_end) & (chunk['time'] <= t)]
        if lor.empty:
            return (ah, al, ar, av, 0.0, 0.0, 0.0, False)
        lh = float(lor['high'].max())
        ll = float(lor['low'].min())
        lr = (lh - ll) / bbb.PIP_SIZE
        lv = lor_min <= lr <= lor_max
        return (ah, al, ar, av, lh, ll, lr, lv)

    for i in range(warmup, len(df)):
        row = df.iloc[i]
        ts = pd.Timestamp(row['time'])
        dnorm = ts.normalize()
        if cur_day is None or dnorm != cur_day:
            cur_day = dnorm
            day_start_idx = i
            ld = init_london_day_state(dnorm, london_cfg)

        # exits for v44/l1 only (same approximation as backtest)
        still = []
        for p in open_positions:
            if p['kind'] == 'v44_oracle':
                if ts >= _ts(p['oracle_exit_time']):
                    scale = float(p['lots']) / max(1e-9, float(p['oracle_lots']))
                    equity += float(p['oracle_pnl_usd']) * scale
                else:
                    still.append(p)
                continue
            if p['kind'] == 'l1':
                sim = p['sim']
                hs_bar = None
                if p.get('l1_realistic_spread'):
                    spb = london_bt.compute_spread_pips(i, ts, london_cfg)
                    hs_bar = spb * bbb.PIP_SIZE / 2.0
                if not sim.on_bar(ts, float(row['open']), float(row['high']), float(row['low']), float(row['close']), half_spread=hs_bar):
                    still.append(p)
                    continue
                _, usd = london_bt.calc_leg_pnl('long' if sim.is_long else 'short', sim.entry_price, float(row['close']), int(p['units']))
                equity += usd
                continue
            still.append(p)
        open_positions = still

        # london advance (needed because london can consume margin)
        if ld is not None:
            ah, al, ar, av, lh, ll, lr, lv = causal_asian_lor(day_start_idx, i, ts)
            nxt_ts = pd.Timestamp(df.iloc[i + 1]['time']) if i + 1 < len(df) else None
            equity, _ldn_closed, l1_payloads, _ = advance_london_unified_bar(
                ld,
                row=row,
                ts=ts,
                nxt_ts=nxt_ts,
                i_day=i - day_start_idx,
                i_global=i,
                asian_high=ah,
                asian_low=al,
                asian_range_pips=ar,
                asian_valid=av,
                lor_high=lh,
                lor_low=ll,
                lor_range_pips=lr,
                lor_valid=lv,
                equity=equity,
                margin_avail_unified=margin_avail(equity),
                extra_open_positions=len(open_positions),
                trade_id_counter=trade_id_box,
                spread_mode_pipeline=True,
                l1_tp1_r=3.25,
                l1_tp2_r=2.0,
                l1_be_offset=1.0,
                l1_tp1_close_fraction=float(london_cfg['setups']['D']['tp1_close_fraction']),
                exec_gate=lambda pe, t: (bbb.admission_checks(
                    strategy='london_v2', entry_ts=t, cell_str=lookup_cell(t)[0], regime=lookup_cell(t)[1],
                    delta_er=lookup_cell(t)[3], setup_d=(str(pe['setup_type']) == 'D'), weekday_name=str(t.day_name())
                )),
            )
            for pay in l1_payloads:
                sim = bbb.build_l1_incremental_from_entry(
                    direction=str(pay['direction']), entry_bar=pay['entry_bar_row'], lor_high=float(pay['lor_high']),
                    lor_low=float(pay['lor_low']), ny_open=pay['ny_open'], spread_pips=float(pay['spread_pips_exec']),
                    tp1_r=3.25, tp2_r=2.0, be_offset=1.0,
                    tp1_close_fraction=float(london_cfg['setups']['D']['tp1_close_fraction'])
                )
                if sim is None:
                    continue
                req_m = float(pay['margin_required_usd'])
                if req_m <= margin_avail(equity):
                    open_positions.append({'kind': 'l1', 'sim': sim, 'units': int(pay['units']), 'margin_usd': req_m, 'l1_realistic_spread': False})

        # V44 oracle admission + margin block capture
        oracle_row = v44_oracle.get(ts.value)
        if oracle_row is None:
            continue
        cell_s, reg, _er, der_v = lookup_cell(ts)
        ok, _reason = bbb.admission_checks(
            strategy='v44_ny', entry_ts=ts, cell_str=cell_s, regime=reg, delta_er=der_v,
            setup_d=False, weekday_name=str(ts.day_name()), variant_f_allows=lambda: variant_f_allows_v44(ts)
        )
        if not ok:
            continue

        raw = dict(oracle_row)
        sl_pips = float(raw.get('sl_pips', 0.0) or raw.get('sl_dist', 5.0) or 5.0)
        ep0 = float(raw.get('entry_price', 150.0))
        pip_val = bbb.pip_value_usd_per_lot(ep0)
        or_pips = float(raw.get('pips', 0.0))
        or_usd = float(raw.get('usd', 0.0))
        oracle_lots = abs(or_usd) / max(1e-9, abs(or_pips) * pip_val) if abs(or_pips) > 1e-9 else 1.0
        risk_usd = equity * v44_risk_pct
        lots = max(0.01, min(max_lot, risk_usd / max(1e-9, sl_pips * pip_val)))
        req_m = lots * 100000.0 / margin_lev

        m_used = margin_used()
        m_ceiling = margin_ceiling(equity)
        m_avail = m_ceiling - m_used
        if req_m > m_avail:
            pos_rows = []
            present = set()
            if ld is not None:
                for p in ld.open_positions:
                    sess = 'london_setup_a' if str(p.setup_type) == 'A' else 'london_l1'
                    present.add(sess)
                    pos_rows.append({
                        'session': sess,
                        'entry_time': str(p.entry_time),
                        'side': str(p.direction),
                        'lots': float(p.initial_units) / 100000.0,
                        'margin_consumed': float(p.margin_required_usd),
                    })
            for p in open_positions:
                if p['kind'] == 'v44_oracle':
                    sess = 'v44_ny'
                    lots_p = float(p.get('lots', 0.0))
                    marg = lots_p * 100000.0 / margin_lev
                    pos_rows.append({'session': sess, 'entry_time': str(p.get('entry_time')), 'side': str(p.get('side')), 'lots': lots_p, 'margin_consumed': marg})
                    present.add(sess)
                elif p['kind'] == 'l1':
                    sess = 'london_l1'
                    pos_rows.append({'session': sess, 'entry_time': str(p['sim'].entry_time), 'side': str(p['sim'].direction), 'lots': float(p['units']) / 100000.0, 'margin_consumed': float(p.get('margin_usd', 0.0))})
                    present.add(sess)
            for s in present:
                blocking_sessions[s] += 1

            blocked_rows.append(
                {
                    'entry_time': str(ts),
                    'equity': float(equity),
                    'margin_used': float(m_used),
                    'margin_ceiling': float(m_ceiling),
                    'margin_available': float(m_avail),
                    'required_margin': float(req_m),
                    'open_positions': pos_rows,
                    'naive_margin_available': float(equity - m_used),
                    'would_fit_naive': bool(req_m <= (equity - m_used)),
                    'would_fit_50pct_no_buffer': bool(req_m <= (equity * 0.5 - m_used)),
                    'oracle_lots': float(oracle_lots),
                    'requested_lots': float(lots),
                }
            )
            continue

        open_positions.append(
            {
                'kind': 'v44_oracle',
                'entry_time': ts,
                'entry_price': float(raw['entry_price']),
                'side': 'buy' if str(raw.get('side', 'buy')).lower() == 'buy' else 'sell',
                'lots': lots,
                'oracle_lots': float(oracle_lots),
                'oracle_pnl_usd': float(raw['usd']),
                'oracle_exit_time': _ts(raw['exit_time']),
            }
        )

    eqs = [float(r['equity']) for r in blocked_rows]
    avs = [float(r['margin_available']) for r in blocked_rows]
    reqs = [float(r['required_margin']) for r in blocked_rows]
    out = {
        'total_margin_blocked': len(blocked_rows),
        'would_fit_with_naive_margin': int(sum(1 for r in blocked_rows if r['would_fit_naive'])),
        'would_fit_with_50pct_no_buffer': int(sum(1 for r in blocked_rows if r['would_fit_50pct_no_buffer'])),
        'blocking_position_sessions': {
            'london_setup_a': int(blocking_sessions.get('london_setup_a', 0)),
            'london_l1': int(blocking_sessions.get('london_l1', 0)),
            'v44_ny': int(blocking_sessions.get('v44_ny', 0)),
        },
        'equity_at_block_distribution': _dist(eqs),
        'margin_available_at_block_distribution': _dist(avs),
        'required_margin_distribution': _dist(reqs),
        'sample_blocks': blocked_rows[:10],
    }
    OUT.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f'wrote {OUT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
