import copy
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from statistics import pstdev

import pandas as pd

ROOT = Path('/Users/codygnon/Documents/usdjpy_assistant')
RESEARCH = ROOT / 'research_out'
ENGINE = ROOT / 'scripts' / 'backtest_tokyo_meanrev.py'
BASE_CONFIG_PATH = RESEARCH / 'tokyo_optimized_v15_config.json'
CSV_1000 = RESEARCH / 'USDJPY_M1_OANDA_1000k.csv'
CSV_500 = RESEARCH / 'USDJPY_M1_OANDA_500k.csv'
CSV_250 = RESEARCH / 'USDJPY_M1_OANDA_250k.csv'

OUT_DIR = RESEARCH / 'tokyo_actual_round1'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str):
    print(msg, flush=True)


def months_for_csv(csv_path: Path) -> float:
    # Read only time column for span calculation.
    df = pd.read_csv(csv_path, usecols=['time'])
    t = pd.to_datetime(df['time'], utc=True, errors='coerce').dropna()
    if t.empty:
        return 1.0
    days = (t.iloc[-1] - t.iloc[0]).total_seconds() / 86400.0
    months = max(days / 30.4375, 1e-9)
    return float(months)


@dataclass
class RunResult:
    label: str
    config_path: str
    report_path: str
    trades_path: str
    equity_path: str
    metrics: dict
    day_of_week: dict
    hourly: dict
    exits: list
    signals_generated: int
    entries_taken: int
    filter_pass_rate_pct: float


def extract_metrics(report: dict, csv_path: Path):
    s = report['summary']
    maxdd = float(s['max_drawdown_usd'])
    net = float(s['net_profit_usd'])
    months = months_for_csv(csv_path)
    net_over_dd = (net / maxdd) if maxdd > 0 else 0.0
    return {
        'trades': int(s['total_trades']),
        'wr_pct': float(s['win_rate_pct']),
        'pf': float(s['profit_factor']),
        'net_usd': net,
        'maxdd_usd': maxdd,
        'maxdd_pct': float(s['max_drawdown_pct']),
        'net_over_maxdd': float(net_over_dd),
        'sharpe': float(s['sharpe_ratio']),
        'calmar': float(s['calmar_ratio']),
        'usd_per_month': float(net / months),
        'months_span': float(months),
    }


def compute_day_hour_from_trades(trades_path: Path):
    if not trades_path.exists():
        return {}, {}
    tdf = pd.read_csv(trades_path)
    if tdf.empty or 'entry_datetime' not in tdf.columns:
        return {}, {}
    tdf['entry_datetime'] = pd.to_datetime(tdf['entry_datetime'], utc=True, errors='coerce')
    tdf = tdf.dropna(subset=['entry_datetime']).copy()
    if tdf.empty:
        return {}, {}
    tdf['win'] = tdf['usd'] > 0
    tdf['dow'] = tdf['entry_datetime'].dt.day_name()
    tdf['hour'] = tdf['entry_datetime'].dt.hour

    day_out = {}
    for d in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
        g = tdf[tdf['dow'] == d]
        if len(g) == 0:
            day_out[d] = {'trades': 0, 'wr': 0.0, 'pf': 0.0, 'net_pnl': 0.0}
            continue
        gp = float(g.loc[g['usd'] > 0, 'usd'].sum())
        gl = float(g.loc[g['usd'] < 0, 'usd'].sum())
        pf = (gp / abs(gl)) if gl < 0 else float('inf')
        day_out[d] = {
            'trades': int(len(g)),
            'wr': float(g['win'].mean() * 100.0),
            'pf': float(pf),
            'net_pnl': float(g['usd'].sum()),
        }

    hour_out = {}
    for h in range(24):
        g = tdf[tdf['hour'] == h]
        if len(g) == 0:
            hour_out[str(h)] = {'trades': 0, 'wr': 0.0, 'pf': 0.0, 'net_pnl': 0.0}
            continue
        gp = float(g.loc[g['usd'] > 0, 'usd'].sum())
        gl = float(g.loc[g['usd'] < 0, 'usd'].sum())
        pf = (gp / abs(gl)) if gl < 0 else float('inf')
        hour_out[str(h)] = {
            'trades': int(len(g)),
            'wr': float(g['win'].mean() * 100.0),
            'pf': float(pf),
            'net_pnl': float(g['usd'].sum()),
        }
    return day_out, hour_out


def run_backtest(cfg: dict, label: str, csv_path: Path) -> RunResult:
    cfg = copy.deepcopy(cfg)
    # Ensure realism/margin required by prompt.
    cfg.setdefault('execution_model', {})['spread_mode'] = 'realistic'
    cfg.setdefault('margin_model', {})['enabled'] = True
    cfg.setdefault('margin_model', {})['leverage'] = 33.3

    report_path = OUT_DIR / f'{label}_report.json'
    trades_path = OUT_DIR / f'{label}_trades.csv'
    equity_path = OUT_DIR / f'{label}_equity.csv'
    cfg_path = OUT_DIR / f'{label}_config.json'

    cfg['run_sequence'] = [{
        'label': label,
        'input_csv': str(csv_path),
        'output_json': str(report_path),
        'output_trades_csv': str(trades_path),
        'output_equity_csv': str(equity_path),
    }]

    cfg_path.write_text(json.dumps(cfg, indent=2))
    log(f'RUN {label} on {csv_path.name} ...')
    subprocess.run(['python3', str(ENGINE), '--config', str(cfg_path)], check=True, cwd=str(ROOT))

    report = json.loads(report_path.read_text())
    metrics = extract_metrics(report, csv_path)

    day, hour = compute_day_hour_from_trades(trades_path)
    exits = report.get('breakdown', {}).get('exit_distribution', [])
    ec = report.get('entry_confirmation_stats', {})
    sig_gen = int(ec.get('signals_generated', 0))
    entries = int(ec.get('signals_confirmed', metrics['trades']))
    pass_rate = (entries / sig_gen * 100.0) if sig_gen > 0 else 0.0

    log(
        f"DONE {label}: trades={metrics['trades']} wr={metrics['wr_pct']:.2f}% pf={metrics['pf']:.3f} "
        f"net={metrics['net_usd']:.2f} maxdd={metrics['maxdd_usd']:.2f}"
    )

    return RunResult(
        label=label,
        config_path=str(cfg_path),
        report_path=str(report_path),
        trades_path=str(trades_path),
        equity_path=str(equity_path),
        metrics=metrics,
        day_of_week=day,
        hourly=hour,
        exits=exits,
        signals_generated=sig_gen,
        entries_taken=entries,
        filter_pass_rate_pct=pass_rate,
    )


def pick_by_pf_with_min_trades(variant_results: dict, min_trades: int = 30):
    eligible = [(k, v) for k, v in variant_results.items() if v.metrics['trades'] >= min_trades]
    if not eligible:
        # Fallback highest PF regardless of trades.
        best = max(variant_results.items(), key=lambda kv: (kv[1].metrics['pf'], kv[1].metrics['net_usd']))
        return best[0], 'No variant met min trades; selected highest PF overall.'
    best = max(eligible, key=lambda kv: (kv[1].metrics['pf'], kv[1].metrics['net_usd']))
    return best[0], 'Selected highest PF among variants with trades >= 30; tie-breaker Net USD.'


def pick_t4(variant_results: dict):
    eligible = [
        (k, v)
        for k, v in variant_results.items()
        if v.metrics['pf'] > 1.3 and v.metrics['trades'] >= 30
    ]
    if eligible:
        best = max(eligible, key=lambda kv: kv[1].metrics['net_usd'])
        return best[0], 'Selected highest Net USD among PF > 1.3 and trades >= 30.'
    best = max(variant_results.items(), key=lambda kv: (kv[1].metrics['pf'], kv[1].metrics['net_usd']))
    return best[0], 'No variant met PF/trade gate; selected highest PF then Net USD.'


def pick_t5(variant_results: dict):
    eligible = [(k, v) for k, v in variant_results.items() if v.metrics['maxdd_pct'] < 8.0]
    if eligible:
        best = max(eligible, key=lambda kv: kv[1].metrics['net_usd'])
        return best[0], 'Selected highest Net USD among MaxDD% < 8.'
    best = min(variant_results.items(), key=lambda kv: kv[1].metrics['maxdd_pct'])
    return best[0], 'No variant under MaxDD gate; selected lowest MaxDD%.'


def variant_table(results: dict):
    out = {}
    for k, rr in results.items():
        out[k] = {
            'metrics': rr.metrics,
            'config_path': rr.config_path,
            'report_path': rr.report_path,
            'signals_generated': rr.signals_generated,
            'entries_taken': rr.entries_taken,
            'filter_pass_rate_pct': rr.filter_pass_rate_pct,
            'day_of_week': rr.day_of_week,
            'hourly': rr.hourly,
            'exit_reason_breakdown': rr.exits,
        }
    return out


def weekday_rank_from_daystats(daystats: dict):
    rows = []
    for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
        d = daystats.get(day, {'trades': 0, 'pf': 0.0, 'net_pnl': 0.0, 'wr': 0.0})
        rows.append((day, float(d.get('pf', 0.0)), float(d.get('net_pnl', 0.0)), int(d.get('trades', 0))))
    # worst first by PF then net
    worst = sorted(rows, key=lambda r: (r[1], r[2]))
    best = sorted(rows, key=lambda r: (r[1], r[2]), reverse=True)
    return rows, worst, best


def apply_session(cfg: dict, start: str, end: str, days=None):
    c = copy.deepcopy(cfg)
    c.setdefault('session_filter', {})['session_start_utc'] = start
    c['session_filter']['session_end_utc'] = end
    if days is not None:
        c['session_filter']['allowed_trading_days'] = list(days)
    # Align entry window to session window for timing experiments.
    c['session_filter']['entry_start_utc'] = start
    c['session_filter']['entry_end_utc'] = end
    return c


def main():
    base_cfg = json.loads(BASE_CONFIG_PATH.read_text())

    results = {
        'baseline_reference': str(BASE_CONFIG_PATH),
        'dataset': str(CSV_1000),
        'realism_margin': {
            'spread_mode': 'realistic',
            'margin_enabled': True,
            'leverage': 33.3,
        },
        'pre_step': {},
        'experiments': {},
        'final': {},
    }

    # Pre-step baseline scan (actual Tokyo hours + all weekdays).
    pre_cfg = apply_session(base_cfg, '00:00', '06:00', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])
    pre_run = run_backtest(pre_cfg, 'tokyo_actual_prestep_1000k', CSV_1000)

    # Pre-step report payload.
    pre_payload = {
        'run': variant_table({'prestep': pre_run})['prestep'],
        'stop_condition': {},
    }

    day_net_all_nonpos = all(float(pre_run.day_of_week.get(d, {}).get('net_pnl', 0.0)) <= 0.0 for d in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])
    hour_net_all_nonpos = True
    for h in range(0, 6):
        hv = pre_run.hourly.get(str(h), pre_run.hourly.get(h, {'net_pnl': 0.0}))
        if float(hv.get('net_pnl', 0.0)) > 0.0:
            hour_net_all_nonpos = False
            break
    stop = (pre_run.metrics['pf'] < 0.8) and (pre_run.metrics['net_usd'] < 0.0) and day_net_all_nonpos and hour_net_all_nonpos
    pre_payload['stop_condition'] = {
        'pf_lt_0p8': pre_run.metrics['pf'] < 0.8,
        'net_negative': pre_run.metrics['net_usd'] < 0,
        'all_days_nonpositive': day_net_all_nonpos,
        'all_hours_00_05_nonpositive': hour_net_all_nonpos,
        'triggered': stop,
    }

    results['pre_step'] = pre_payload

    if stop:
        results['final'] = {
            'stopped_early': True,
            'reason': 'Pre-step stop condition triggered: no detectable edge during actual Tokyo hours under baseline signal stack.',
        }
        out_path = RESEARCH / 'tokyo_actual_round1_results.json'
        out_path.write_text(json.dumps(results, indent=2))
        log(f'STOP CONDITION TRIGGERED. Saved {out_path}')
        return

    # Current baseline for experiments.
    current_cfg = copy.deepcopy(pre_cfg)
    changelog = []

    # T1 session window calibration
    t1_variants = {
        'T1A': ('00:00', '06:00'),
        'T1B': ('23:00', '06:00'),
        'T1C': ('00:00', '07:00'),
        'T1D': ('01:00', '06:00'),
        'T1E': ('00:00', '05:00'),
        'T1F': ('01:00', '05:00'),
    }
    t1_runs = {}
    for k, (st, en) in t1_variants.items():
        cfg = apply_session(current_cfg, st, en)
        t1_runs[k] = run_backtest(cfg, f'tokyo_actual_{k}_1000k', CSV_1000)
    t1_winner, t1_reason = pick_by_pf_with_min_trades(t1_runs, min_trades=30)
    current_cfg = json.loads(Path(t1_runs[t1_winner].config_path).read_text())
    changelog.append({'experiment': 'T1', 'winner': t1_winner, 'reason': t1_reason})
    results['experiments']['T1'] = {
        'variants': variant_table(t1_runs),
        'winner': t1_winner,
        'winner_reason': t1_reason,
    }

    # T2 day-of-week analysis + combos.
    # Start with all weekdays for ranking.
    t2a_cfg = copy.deepcopy(current_cfg)
    t2a_cfg['session_filter']['allowed_trading_days'] = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    t2a_run = run_backtest(t2a_cfg, 'tokyo_actual_T2A_1000k', CSV_1000)

    rows, worst, best = weekday_rank_from_daystats(t2a_run.day_of_week)
    all_days = [r[0] for r in rows]
    worst1 = worst[0][0]
    worst2 = [worst[0][0], worst[1][0]]
    best2 = [best[0][0], best[1][0]]
    best3 = [best[0][0], best[1][0], best[2][0]]

    t2_day_sets = {
        'T2A': all_days,
        'T2B': [d for d in all_days if d != worst1],
        'T2C': [d for d in all_days if d not in set(worst2)],
        'T2D': best2,
        'T2E': best3,
    }

    t2_runs = {'T2A': t2a_run}
    for k in ['T2B', 'T2C', 'T2D', 'T2E']:
        cfg = copy.deepcopy(current_cfg)
        cfg['session_filter']['allowed_trading_days'] = t2_day_sets[k]
        t2_runs[k] = run_backtest(cfg, f'tokyo_actual_{k}_1000k', CSV_1000)

    t2_winner, t2_reason = pick_by_pf_with_min_trades(t2_runs, min_trades=30)
    current_cfg = json.loads(Path(t2_runs[t2_winner].config_path).read_text())
    changelog.append({'experiment': 'T2', 'winner': t2_winner, 'reason': t2_reason})
    results['experiments']['T2'] = {
        'base_day_of_week_analysis': t2a_run.day_of_week,
        'derived_day_sets': t2_day_sets,
        'variants': variant_table(t2_runs),
        'winner': t2_winner,
        'winner_reason': t2_reason,
    }

    # T3 entry signal recalibration using pivot tolerance as primary threshold.
    base_tol = float(current_cfg['entry_rules']['long']['price_zone']['tolerance_pips'])
    mults = {
        'T3A': None,  # baseline
        'T3B': 0.6,
        'T3C': 0.8,
        'T3D': 1.0,
        'T3E': 1.2,
        'T3F': 1.5,
    }
    t3_runs = {}
    for k, m in mults.items():
        cfg = copy.deepcopy(current_cfg)
        if m is not None:
            tol = float(base_tol * m)
            cfg['entry_rules']['long']['price_zone']['tolerance_pips'] = tol
            cfg['entry_rules']['short']['price_zone']['tolerance_pips'] = tol
        t3_runs[k] = run_backtest(cfg, f'tokyo_actual_{k}_1000k', CSV_1000)
    t3_winner, t3_reason = pick_by_pf_with_min_trades(t3_runs, min_trades=30)
    current_cfg = json.loads(Path(t3_runs[t3_winner].config_path).read_text())
    changelog.append({'experiment': 'T3', 'winner': t3_winner, 'reason': t3_reason, 'base_tolerance_pips': base_tol})
    results['experiments']['T3'] = {
        'primary_threshold': 'entry_rules.*.price_zone.tolerance_pips',
        'base_tolerance_pips': base_tol,
        'variants': variant_table(t3_runs),
        'winner': t3_winner,
        'winner_reason': t3_reason,
    }

    # T4 SL/TP calibration for Tokyo volatility.
    # TP is ATR-based in this engine; keep TP settings unchanged per prompt.
    t4_sl_vals = {
        'T4A': 10.0,
        'T4B': 12.0,
        'T4C': 15.0,
        'T4D': 20.0,
        'T4E': 25.0,
    }
    t4_runs = {}
    for k, slmax in t4_sl_vals.items():
        cfg = copy.deepcopy(current_cfg)
        cfg['exit_rules']['stop_loss']['hard_max_sl_pips'] = slmax
        # Ensure min sl does not exceed max sl.
        cfg['exit_rules']['stop_loss']['minimum_sl_pips'] = min(float(cfg['exit_rules']['stop_loss'].get('minimum_sl_pips', 10.0)), slmax)
        t4_runs[k] = run_backtest(cfg, f'tokyo_actual_{k}_1000k', CSV_1000)
    t4_winner, t4_reason = pick_t4(t4_runs)
    current_cfg = json.loads(Path(t4_runs[t4_winner].config_path).read_text())
    changelog.append({'experiment': 'T4', 'winner': t4_winner, 'reason': t4_reason})
    results['experiments']['T4'] = {
        'note': 'TP is ATR-based in this engine; TP scaling sub-grid skipped by design.',
        'variants': variant_table(t4_runs),
        'winner': t4_winner,
        'winner_reason': t4_reason,
    }

    # T5 risk sizing.
    t5_risk = {
        'T5A': 0.75,
        'T5B': 1.0,
        'T5C': 1.25,
        'T5D': 1.5,
        'T5E': 2.0,
    }
    t5_runs = {}
    for k, r in t5_risk.items():
        cfg = copy.deepcopy(current_cfg)
        cfg['position_sizing']['risk_per_trade_pct'] = r
        t5_runs[k] = run_backtest(cfg, f'tokyo_actual_{k}_1000k', CSV_1000)
    t5_winner, t5_reason = pick_t5(t5_runs)
    current_cfg = json.loads(Path(t5_runs[t5_winner].config_path).read_text())
    changelog.append({'experiment': 'T5', 'winner': t5_winner, 'reason': t5_reason})
    results['experiments']['T5'] = {
        'variants': variant_table(t5_runs),
        'winner': t5_winner,
        'winner_reason': t5_reason,
    }

    # T6 regime/cooldown filters.
    base_pct = float(current_cfg['indicators']['bb_width_regime_filter'].get('ranging_percentile', 0.8))
    base_cool = int(current_cfg['trade_management'].get('cooldown_minutes', 15))

    def looser(base, frac):
        # Move toward 1.0 by fraction of remaining gap.
        return min(1.0, base + (1.0 - base) * frac)

    t6_runs = {}
    # T6A baseline
    t6_runs['T6A'] = run_backtest(copy.deepcopy(current_cfg), 'tokyo_actual_T6A_1000k', CSV_1000)
    # T6B looser regime 25%
    c = copy.deepcopy(current_cfg)
    c['indicators']['bb_width_regime_filter']['ranging_percentile'] = looser(base_pct, 0.25)
    t6_runs['T6B'] = run_backtest(c, 'tokyo_actual_T6B_1000k', CSV_1000)
    # T6C looser regime 50%
    c = copy.deepcopy(current_cfg)
    c['indicators']['bb_width_regime_filter']['ranging_percentile'] = looser(base_pct, 0.50)
    t6_runs['T6C'] = run_backtest(c, 'tokyo_actual_T6C_1000k', CSV_1000)
    # T6D reduce cooldown 50%
    c = copy.deepcopy(current_cfg)
    c['trade_management']['cooldown_minutes'] = max(0, int(round(base_cool * 0.5)))
    t6_runs['T6D'] = run_backtest(c, 'tokyo_actual_T6D_1000k', CSV_1000)
    # T6E remove breakout cooldown entirely
    c = copy.deepcopy(current_cfg)
    c['trade_management']['breakout_detection_mode'] = 'disabled'
    t6_runs['T6E'] = run_backtest(c, 'tokyo_actual_T6E_1000k', CSV_1000)
    # T6F combine T6C + T6D
    c = copy.deepcopy(current_cfg)
    c['indicators']['bb_width_regime_filter']['ranging_percentile'] = looser(base_pct, 0.50)
    c['trade_management']['cooldown_minutes'] = max(0, int(round(base_cool * 0.5)))
    t6_runs['T6F'] = run_backtest(c, 'tokyo_actual_T6F_1000k', CSV_1000)

    t6_winner, t6_reason = pick_by_pf_with_min_trades(t6_runs, min_trades=30)
    current_cfg = json.loads(Path(t6_runs[t6_winner].config_path).read_text())
    changelog.append({'experiment': 'T6', 'winner': t6_winner, 'reason': t6_reason})
    results['experiments']['T6'] = {
        'base_ranging_percentile': base_pct,
        'base_cooldown_minutes': base_cool,
        'variants': variant_table(t6_runs),
        'winner': t6_winner,
        'winner_reason': t6_reason,
    }

    # Save final config.
    final_cfg_path = RESEARCH / 'tokyo_actual_v1_config.json'
    final_cfg_path.write_text(json.dumps(current_cfg, indent=2))

    # Scaling runs on 250k/500k/1000k.
    final_runs = {}
    final_runs['250k'] = run_backtest(current_cfg, 'tokyo_actual_final_250k', CSV_250)
    final_runs['500k'] = run_backtest(current_cfg, 'tokyo_actual_final_500k', CSV_500)
    final_runs['1000k'] = run_backtest(current_cfg, 'tokyo_actual_final_1000k', CSV_1000)

    pfs = [final_runs[k].metrics['pf'] for k in ['250k', '500k', '1000k']]
    pf_std = float(pstdev(pfs)) if len(pfs) > 1 else 0.0

    # Scorecard on 1000k final.
    m1000 = final_runs['1000k'].metrics
    scorecard = {
        'trades_gt_80': {'target': '> 80', 'actual': m1000['trades'], 'pass': m1000['trades'] > 80},
        'pf_gt_1p5': {'target': '> 1.5', 'actual': m1000['pf'], 'pass': m1000['pf'] > 1.5},
        'wr_gt_55': {'target': '> 55%', 'actual': m1000['wr_pct'], 'pass': m1000['wr_pct'] > 55.0},
        'maxdd_lt_8': {'target': '< 8%', 'actual': m1000['maxdd_pct'], 'pass': m1000['maxdd_pct'] < 8.0},
        'usd_per_month_gt_250': {'target': '> 250', 'actual': m1000['usd_per_month'], 'pass': m1000['usd_per_month'] > 250.0},
        'pf_std_lt_0p5': {'target': '< 0.50', 'actual': pf_std, 'pass': pf_std < 0.50},
    }

    # Compare to US-session V15 (reference only).
    us_v15_report = json.loads((RESEARCH / 'tokyo_optimized_v15_1000k_report.json').read_text())
    us_ref = {
        'trades': int(us_v15_report['summary']['total_trades']),
        'wr_pct': float(us_v15_report['summary']['win_rate_pct']),
        'pf': float(us_v15_report['summary']['profit_factor']),
        'net_usd': float(us_v15_report['summary']['net_profit_usd']),
        'maxdd_usd': float(us_v15_report['summary']['max_drawdown_usd']),
    }

    results['final'] = {
        'stopped_early': False,
        'changelog': changelog,
        'final_config_path': str(final_cfg_path),
        'final_scaling': {
            k: {
                'metrics': final_runs[k].metrics,
                'report_path': final_runs[k].report_path,
                'day_of_week': final_runs[k].day_of_week if k == '1000k' else {},
                'hourly': final_runs[k].hourly if k == '1000k' else {},
                'exit_reason_breakdown': final_runs[k].exits if k == '1000k' else [],
            }
            for k in ['250k', '500k', '1000k']
        },
        'pf_stddev': pf_std,
        'scorecard': scorecard,
        'us_session_v15_reference_1000k': us_ref,
        'tokyo_vs_us_summary': {
            'delta_trades': int(m1000['trades'] - us_ref['trades']),
            'delta_pf': float(m1000['pf'] - us_ref['pf']),
            'delta_net_usd': float(m1000['net_usd'] - us_ref['net_usd']),
            'delta_maxdd_usd': float(m1000['maxdd_usd'] - us_ref['maxdd_usd']),
        },
        'viability_assessment': {
            'edge_exists': bool(m1000['pf'] > 1.0 and m1000['net_usd'] > 0),
            'deployment_ready_by_scorecard': bool(all(v['pass'] for v in scorecard.values())),
            'honest_note': (
                'Tokyo-hour strategy passes round-1 deployment scorecard.'
                if all(v['pass'] for v in scorecard.values())
                else 'Tokyo-hour strategy does not pass deployment scorecard in round 1; further redesign may be required.'
            ),
        },
    }

    out_path = RESEARCH / 'tokyo_actual_round1_results.json'
    out_path.write_text(json.dumps(results, indent=2))
    log(f'Saved {out_path}')
    log(f'Saved final config {final_cfg_path}')


if __name__ == '__main__':
    main()
