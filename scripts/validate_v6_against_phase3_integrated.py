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

from scripts import backtest_merged_integrated_tokyo_london_v2_ny as merged_engine
from scripts import backtest_offensive_slice_additive as additive
from scripts import backtest_variant_k_v6_search as v6
from scripts import backtest_offensive_slice_family_combo as family_combo
from scripts import run_offensive_slice_discovery as discovery

OUT_DIR = ROOT / 'research_out'
OUT_JSON = OUT_DIR / 'v6_vs_phase3_integrated_validation.json'
OUT_MD = OUT_DIR / 'v6_vs_phase3_integrated_validation.md'
PHASE3_BASELINES = {
    '500k': OUT_DIR / 'phase3_integrated_500k_report.json',
    '1000k': OUT_DIR / 'phase3_integrated_1000k_report.json',
}
STARTING_EQUITY = 100000.0

V6_VALIDATED_NO_SINGLETONS = [
    'C0_sell_strong', 'C1_sell_base', 'C2_sell', 'C3_buy',
    'C4_sell_base', 'C5_pbt_sell', 'C6_pbt_sell',
    'O0_buy_strong', 'O1_buy_strong', 'O2_buy_strong',
    'ADJ_meanrev_low_neg_buy', 'ADJ_ambig_mid_neg_sell', 'ADJ_mom_high_neg_sell',
    'N1_brkout_low_neg_sell_strong', 'N2_brkout_low_pos_buy_strong',
    'L2_brkout_mid_neg_buy', 'T1_ambig_high_pos_buy', 'T2_brkout_mid_pos_buy',
]

V6_VALIDATED_ONLY = V6_VALIDATED_NO_SINGLETONS + [
    'N3_brkout_low_neg_buy_news', 'N4_pbt_low_neg_buy_news',
]

V6_KITCHEN_SINK = V6_VALIDATED_ONLY + [
    'L1_mom_low_pos_buy', 'T3_ambig_mid_pos_sell',
]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return discovery._json_default(obj)


def _ts(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize('UTC')
    return ts.tz_convert('UTC')


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def _baseline_trade_key(row: merged_engine.TradeRow) -> tuple[str, str, str, str]:
    return (
        str(row.strategy),
        pd.Timestamp(row.entry_time).isoformat(),
        pd.Timestamp(row.exit_time).isoformat(),
        str(row.side),
    )


def _offensive_trade_key(trade: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(trade.get('strategy', '')),
        _ts(trade['entry_time']).isoformat(),
        _ts(trade['exit_time']).isoformat(),
        str(trade.get('side', '')),
    )


def _normalize_trade_dict_time(trade: dict[str, Any]) -> dict[str, Any]:
    t = dict(trade)
    t['entry_time'] = _ts(t['entry_time']).isoformat()
    t['exit_time'] = _ts(t['exit_time']).isoformat()
    return t


def _load_phase3_baseline(dataset_key: str) -> dict[str, Any]:
    report = _load_json(PHASE3_BASELINES[dataset_key])
    trades: list[merged_engine.TradeRow] = []
    for t in report['closed_trades']:
        trades.append(
            merged_engine.TradeRow(
                strategy=str(t['strategy']),
                entry_time=_ts(t['entry_time']),
                exit_time=_ts(t['exit_time']),
                entry_session=str(t.get('entry_session') or 'unknown'),
                side=str(t['side']),
                pips=float(t['pips']),
                usd=float(t['usd']),
                exit_reason=str(t.get('exit_reason') or 'unknown'),
                standalone_entry_equity=STARTING_EQUITY,
                raw={},
                size_scale=float(t.get('size_scale', 1.0) or 1.0),
            )
        )
    combined = report['combined']
    return {
        'report_path': str(PHASE3_BASELINES[dataset_key]),
        'report': report,
        'baseline_actual_trades': trades,
        'baseline_summary': {
            'total_trades': int(combined['total_trades']),
            'wins': int(combined['wins']),
            'losses': int(combined['losses']),
            'win_rate_pct': float(combined['win_rate_pct']),
            'profit_factor': float(combined['profit_factor']),
            'net_usd': float(combined['net_usd']),
            'net_pips': float(combined['net_pips']),
            'max_drawdown_usd': float(combined['max_drawdown_usd']),
            'max_drawdown_pct': float(combined['max_drawdown_pct']),
            'sharpe_ratio': float(combined['sharpe_ratio']),
            'calmar_ratio': float(combined['calmar_ratio']),
            'avg_win_pips': float(combined['avg_win_pips']),
            'avg_loss_pips': float(combined['avg_loss_pips']),
            'avg_win_usd': float(combined['avg_win_usd']),
            'avg_loss_usd': float(combined['avg_loss_usd']),
        },
    }


def _scale_offensive_against_phase3_baseline(*, baseline_actual_trades: list[merged_engine.TradeRow], offensive_trades: list[dict[str, Any]]) -> list[merged_engine.TradeRow]:
    baseline = [merged_engine.TradeRow(**{**t.__dict__}) for t in baseline_actual_trades]
    offensive_rows = [additive.trade_dict_to_trade_row(_normalize_trade_dict_time(t)) for t in offensive_trades]

    events: list[tuple[pd.Timestamp, int, str, int]] = []
    for i, t in enumerate(baseline):
        events.append((pd.Timestamp(t.exit_time), 0, 'baseline', i))
    for i, t in enumerate(offensive_rows):
        events.append((pd.Timestamp(t.entry_time), 1, 'offensive', i))
        events.append((pd.Timestamp(t.exit_time), 0, 'offensive', i))
    events.sort(key=lambda x: (x[0], x[1]))

    realized_equity = float(STARTING_EQUITY)
    entry_scale: dict[int, float] = {}
    scaled_offensive: dict[int, merged_engine.TradeRow] = {}

    for _, evt_type, source, idx in events:
        if source == 'baseline':
            if evt_type == 0:
                realized_equity += float(baseline[idx].usd)
            continue

        row = offensive_rows[idx]
        if evt_type == 1:
            base_eq = float(row.standalone_entry_equity) if float(row.standalone_entry_equity) > 0 else float(STARTING_EQUITY)
            scale = float(realized_equity / base_eq) if base_eq > 0 else 1.0
            if not pd.notna(scale) or scale <= 0:
                scale = 1.0
            entry_scale[idx] = float(scale)
        else:
            sc = float(entry_scale.get(idx, 1.0))
            scaled_offensive[idx] = merged_engine.TradeRow(
                strategy=row.strategy,
                entry_time=row.entry_time,
                exit_time=row.exit_time,
                entry_session=row.entry_session,
                side=row.side,
                pips=row.pips,
                usd=float(row.usd) * sc,
                exit_reason=row.exit_reason,
                standalone_entry_equity=row.standalone_entry_equity,
                raw=dict(row.raw),
                size_scale=float(row.size_scale) * sc,
            )
            realized_equity += float(scaled_offensive[idx].usd)

    merged = sorted(baseline + list(scaled_offensive.values()), key=lambda t: (t.exit_time, t.entry_time))
    return merged


def _run_variant_against_phase3(*, name: str, selected: dict[str, list[dict[str, Any]]], policy: additive.ConflictPolicy, baselines: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out = {'name': name, 'datasets': {}}
    for dataset_key in ['500k', '1000k']:
        baseline = baselines[dataset_key]
        baseline_actual_trades = baseline['baseline_actual_trades']
        baseline_summary = baseline['baseline_summary']

        policy_selected, policy_stats = additive._apply_conflict_policy_to_selected_trades(selected[dataset_key], policy)
        baseline_keys = {_baseline_trade_key(t) for t in baseline_actual_trades}
        exact_overlap = [t for t in policy_selected if _offensive_trade_key(_normalize_trade_dict_time(t)) in baseline_keys]
        additive_candidates = [t for t in policy_selected if _offensive_trade_key(_normalize_trade_dict_time(t)) not in baseline_keys]

        merged_trades = _scale_offensive_against_phase3_baseline(
            baseline_actual_trades=baseline_actual_trades,
            offensive_trades=additive_candidates,
        )
        eq = merged_engine._build_equity_curve(merged_trades, STARTING_EQUITY)
        summary = merged_engine._stats(merged_trades, STARTING_EQUITY, eq)
        overlapping_baseline = []
        for bt in baseline_actual_trades:
            if any(additive._intervals_overlap(_ts(bt.entry_time), _ts(bt.exit_time), _ts(t['entry_time']), _ts(t['exit_time'])) for t in additive_candidates):
                overlapping_baseline.append(bt)
        internal_overlap_pairs, opposite_side_pairs = additive._count_internal_overlap_pairs(policy_selected)

        out['datasets'][dataset_key] = {
            'baseline_summary': baseline_summary,
            'variant_summary': summary,
            'delta_vs_phase3_integrated': {
                'total_trades': int(summary['total_trades'] - baseline_summary['total_trades']),
                'net_usd': round(summary['net_usd'] - baseline_summary['net_usd'], 2),
                'profit_factor': round(summary['profit_factor'] - baseline_summary['profit_factor'], 4),
                'max_drawdown_usd': round(summary['max_drawdown_usd'] - baseline_summary['max_drawdown_usd'], 2),
            },
            'selection_counts': {
                'raw_selected_trade_count': len(selected[dataset_key]),
                'policy_selected_trade_count': len(policy_selected),
                'exact_phase3_overlap_count': len(exact_overlap),
                'new_additive_trades_count': len(additive_candidates),
                'overlapping_phase3_trades_count': len({(t.strategy, t.entry_time.isoformat(), t.exit_time.isoformat(), t.side) for t in overlapping_baseline}),
                'internal_overlap_pairs': internal_overlap_pairs,
                'internal_opposite_side_overlap_pairs': opposite_side_pairs,
            },
            'policy_stats': policy_stats,
        }
    return out


def _score(row: dict[str, Any]) -> tuple[float, float]:
    d500 = row['datasets']['500k']['delta_vs_phase3_integrated']
    d1k = row['datasets']['1000k']['delta_vs_phase3_integrated']
    return (float(d500['net_usd']) + float(d1k['net_usd']), float(d500['profit_factor']) + float(d1k['profit_factor']))


def _build_md(payload: dict[str, Any]) -> str:
    lines = [
        '# V6 Validation Against Phase 3 Integrated Baseline',
        '',
        'This keeps the original `phase3_integrated` baseline trades fixed as reported, then layers the validated `v6` packages on top using the same shared-equity style entry scaling for the offensive additions.',
        '',
        '- This is more Phase-3-like than the family-only additive harness.',
        '- It is still an anchored replay, not a full bar-by-bar live Phase 3 engine rerun.',
        '',
        f"- policy: `{payload['policy']['name']}`",
        f"- hedging enabled: `{payload['policy']['hedging_enabled']}`",
        f"- opposite-side coexistence: `{payload['policy']['allow_opposite_side_overlap']}`",
        f"- internal overlap allowed: `{payload['policy']['allow_internal_overlap']}`",
        f"- margin model in anchored replay: `{payload['policy']['margin_model_enabled']}`",
        '',
    ]
    for row in payload['leaderboard']:
        d500 = row['datasets']['500k']['delta_vs_phase3_integrated']
        d1k = row['datasets']['1000k']['delta_vs_phase3_integrated']
        s500 = row['datasets']['500k']['variant_summary']
        s1k = row['datasets']['1000k']['variant_summary']
        lines += [
            f"## {row['name']}",
            '',
            f"- combined delta USD vs phase3_integrated: `{round(d500['net_usd'] + d1k['net_usd'], 2)}`",
            f"- combined delta PF vs phase3_integrated: `{round(d500['profit_factor'] + d1k['profit_factor'], 4)}`",
            '',
            '### 500k',
            f"- total trades: `{s500['total_trades']}`",
            f"- net USD: `{round(s500['net_usd'], 2)}`",
            f"- PF: `{round(s500['profit_factor'], 4)}`",
            f"- max DD: `{round(s500['max_drawdown_usd'], 2)}`",
            f"- delta USD vs phase3_integrated: `{round(d500['net_usd'], 2)}`",
            f"- delta PF vs phase3_integrated: `{round(d500['profit_factor'], 4)}`",
            f"- delta DD vs phase3_integrated: `{round(d500['max_drawdown_usd'], 2)}`",
            '',
            '### 1000k',
            f"- total trades: `{s1k['total_trades']}`",
            f"- net USD: `{round(s1k['net_usd'], 2)}`",
            f"- PF: `{round(s1k['profit_factor'], 4)}`",
            f"- max DD: `{round(s1k['max_drawdown_usd'], 2)}`",
            f"- delta USD vs phase3_integrated: `{round(d1k['net_usd'], 2)}`",
            f"- delta PF vs phase3_integrated: `{round(d1k['profit_factor'], 4)}`",
            f"- delta DD vs phase3_integrated: `{round(d1k['max_drawdown_usd'], 2)}`",
            '',
        ]
    return '\n'.join(lines) + '\n'


def main() -> int:
    matrix = family_combo._load_matrix(family_combo.DEFAULT_MATRIX)
    specs = v6._load_all_specs(matrix)
    strategies = {specs[label].strategy for label in set(V6_KITCHEN_SINK)}
    all_trades = discovery._load_all_normalized_trades(strategies)
    trades_by_ds = v6._select_all_trades(specs, all_trades)
    baselines = {k: _load_phase3_baseline(k) for k in ['500k', '1000k']}
    policy = additive.ConflictPolicy(
        name='phase3_integrated_anchored_v6_hedging_like',
        hedging_enabled=True,
        allow_internal_overlap=True,
        allow_opposite_side_overlap=True,
        max_open_offensive=None,
        max_entries_per_day=None,
        margin_model_enabled=False,
        margin_leverage=33.3,
        margin_buffer_pct=0.0,
        max_lot_per_trade=20.0,
    )

    leaderboard = [
        _run_variant_against_phase3(name='v6_validated_no_singletons', selected={ds: v6._collect_family_trades(V6_VALIDATED_NO_SINGLETONS, trades_by_ds[ds]) for ds in ['500k', '1000k']}, policy=policy, baselines=baselines),
        _run_variant_against_phase3(name='v6_validated_only', selected={ds: v6._collect_family_trades(V6_VALIDATED_ONLY, trades_by_ds[ds]) for ds in ['500k', '1000k']}, policy=policy, baselines=baselines),
        _run_variant_against_phase3(name='v6_kitchen_sink_strict_shape', selected={ds: v6._collect_family_trades(V6_KITCHEN_SINK, trades_by_ds[ds]) for ds in ['500k', '1000k']}, policy=policy, baselines=baselines),
    ]
    leaderboard.sort(key=_score, reverse=True)

    payload = {
        'title': 'V6 validation against phase3_integrated baseline',
        'policy': {
            'name': policy.name,
            'hedging_enabled': policy.hedging_enabled,
            'allow_internal_overlap': policy.allow_internal_overlap,
            'allow_opposite_side_overlap': policy.allow_opposite_side_overlap,
            'max_open_offensive': policy.max_open_offensive,
            'max_entries_per_day': policy.max_entries_per_day,
            'margin_model_enabled': policy.margin_model_enabled,
        },
        'baseline_reports': {k: str(v['report_path']) for k, v in baselines.items()},
        'leaderboard': leaderboard,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=_json_default), encoding='utf-8')
    OUT_MD.write_text(_build_md(payload), encoding='utf-8')
    print(OUT_JSON)
    print(OUT_MD)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
