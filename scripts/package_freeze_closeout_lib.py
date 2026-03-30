#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import pandas as pd

ROOT = Path('/Users/codygnon/Documents/usdjpy_assistant')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import backtest_offensive_slice_additive as additive
from scripts import backtest_variant_k_v6_search as v6
from scripts import backtest_variant_k_v7_profile_search as v7_profile
from scripts import backtest_variant_k_v7_search as v7
from scripts import backtest_offensive_slice_family_combo as family_combo
from scripts import run_offensive_slice_discovery as discovery

OUT_DIR = ROOT / 'research_out'
DEFAULT_MATRIX = OUT_DIR / 'offensive_slice_discovery_matrix.json'

UNIFIED_REGISTRY: dict[str, str] = {}
UNIFIED_REGISTRY.update(v7_profile.ALL_REGISTRY)
for label in [
    'L1_mom_low_pos_buy', 'L2_brkout_mid_neg_buy',
    'N1_brkout_low_neg_sell_strong', 'N2_brkout_low_pos_buy_strong',
    'N3_brkout_low_neg_buy_news', 'N4_pbt_low_neg_buy_news',
    'T1_ambig_high_pos_buy', 'T2_brkout_mid_pos_buy', 'T3_ambig_mid_pos_sell',
]:
    UNIFIED_REGISTRY[label] = v6.CELL_REGISTRY[label]

V7_BEST_COMBO_CELLS = [
    'C1_sell_base', 'C2_sell', 'C3_buy', 'C4_sell_base', 'C5_pbt_sell', 'C6_pbt_sell',
    'O0_buy_strong', 'O1_buy_strong', 'O2_buy_strong',
    'ADJ_meanrev_low_neg_buy', 'ADJ_ambig_mid_neg_sell', 'ADJ_mom_high_neg_sell',
    'N1_brkout_low_neg_sell_strong', 'N2_brkout_low_pos_buy_strong',
    'L2_brkout_mid_neg_buy', 'T1_ambig_high_pos_buy', 'T2_brkout_mid_pos_buy',
    'C0_base', 'NEW_v14_ambig_low_pos_sell', 'NEW_v44_ambig_low_neg_sell_normal',
]

PACKAGE_SCALES: dict[str, dict[str, float]] = {
    'v6_clean': dict(v7.REFERENCE_VARIANTS['v6_validated_no_singletons']),
    'v7_usd_max': {
        **{label: 1.0 for label in v7.FIXED_CORE},
        'L2_brkout_mid_neg_buy': 1.0,
        'T1_ambig_high_pos_buy': 1.0,
        'T2_brkout_mid_pos_buy': 1.0,
        'N3_brkout_low_neg_buy_news': 1.0,
        'N4_pbt_low_neg_buy_news': 1.0,
        'L1_mom_low_pos_buy': 1.0,
        'T3_ambig_mid_pos_sell': 0.5,
    },
    'v7_pfdd': {
        **{label: 1.0 for label in v7.FIXED_CORE},
        'L2_brkout_mid_neg_buy': 1.0,
        'T1_ambig_high_pos_buy': 1.0,
        'T2_brkout_mid_pos_buy': 1.0,
        'N3_brkout_low_neg_buy_news': 1.0,
        'N4_pbt_low_neg_buy_news': 1.0,
        'L1_mom_low_pos_buy': 1.0,
        'T3_ambig_mid_pos_sell': 0.25,
    },
    'v7_best_combo': {label: 1.0 for label in V7_BEST_COMBO_CELLS},
}

KNOWN_WEAK_LABELS = ['T3_ambig_mid_pos_sell', 'L1_mom_low_pos_buy', 'N3_brkout_low_neg_buy_news', 'N4_pbt_low_neg_buy_news', 'T1_ambig_high_pos_buy']
PACKAGE_FRONTIER_ORDER = ['v6_clean', 'v7_best_combo', 'v7_usd_max', 'v7_pfdd']
TIME_FRONTIER_TOP_TWO = ['v6_clean', 'v7_pfdd']
PRUNING_PACKAGE_ORDER = ['v7_usd_max', 'v7_pfdd', 'v7_best_combo']

Context = dict[str, Any]
TradeFilter = Callable[[dict[str, Any]], bool]
LabelFilters = dict[str, TradeFilter]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return discovery._json_default(obj)


def strict_policy() -> additive.ConflictPolicy:
    return additive.ConflictPolicy(
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


def load_context(matrix_path: Path | None = None) -> Context:
    matrix_path = matrix_path or DEFAULT_MATRIX
    matrix = family_combo._load_matrix(Path(matrix_path))
    specs_v6 = v6._load_all_specs(matrix)
    specs_v7p = v7_profile._load_all_specs(matrix)
    specs = dict(specs_v6)
    specs.update(specs_v7p)
    missing = sorted(set(UNIFIED_REGISTRY) - set(specs))
    if missing:
        raise RuntimeError(f'Missing slice specs for: {missing}')
    strategies = {specs[label].strategy for label in UNIFIED_REGISTRY}
    all_trades = discovery._load_all_normalized_trades(strategies)
    trades_by_ds = {'500k': {}, '1000k': {}}
    for ds in ['500k', '1000k']:
        for label, spec in specs.items():
            trades_by_ds[ds][label] = [t for t in all_trades[ds].get(spec.strategy, []) if discovery._passes_filters(t, spec)]
    baseline_ctx_by_ds = {
        ds: additive.build_baseline_context(discovery.DATASETS[ds])
        for ds in ['500k', '1000k']
    }
    return {
        'matrix': matrix,
        'specs': specs,
        'trades_by_ds': trades_by_ds,
        'baseline_ctx_by_ds': baseline_ctx_by_ds,
        'policy': strict_policy(),
    }


def active_labels(scales: dict[str, float]) -> list[str]:
    return sorted(label for label, scale in scales.items() if float(scale) > 0.0)


def package_scales(name: str) -> dict[str, float]:
    if name not in PACKAGE_SCALES:
        raise KeyError(f'Unknown package: {name}')
    return dict(PACKAGE_SCALES[name])


def package_weak_labels(name: str) -> list[str]:
    scales = package_scales(name)
    return [label for label in KNOWN_WEAK_LABELS if float(scales.get(label, 0.0)) > 0.0]


def multiply_scales(scales: dict[str, float], factor: float) -> dict[str, float]:
    out = {}
    for label, scale in scales.items():
        if scale <= 0:
            out[label] = 0.0
        else:
            out[label] = round(float(scale) * float(factor), 6)
    return out


def scaled_combined_trades(
    scales: dict[str, float],
    trades_by_label: dict[str, list[dict[str, Any]]],
    label_filters: LabelFilters | None = None,
) -> list[dict[str, Any]]:
    combined: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    label_filters = label_filters or {}
    for label in active_labels(scales):
        scale = float(scales[label])
        filt = label_filters.get(label)
        for trade in trades_by_label.get(label, []):
            if filt is not None and not filt(trade):
                continue
            key = (str(trade['strategy']), str(trade['entry_time']), str(trade['exit_time']), str(trade['side']))
            if key in seen:
                continue
            seen.add(key)
            if abs(scale - 1.0) < 1e-9:
                combined.append(trade)
            else:
                t = deepcopy(trade)
                t['usd'] = float(t['usd']) * scale
                t['size_scale'] = float(t.get('size_scale', 1.0) or 1.0) * scale
                combined.append(t)
    combined.sort(key=lambda t: (t['entry_time'], t['exit_time']))
    return combined


def evaluate_package(
    *,
    context: Context,
    name: str,
    scales: dict[str, float],
    label_filters: LabelFilters | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for ds in ['500k', '1000k']:
        combined = scaled_combined_trades(scales, context['trades_by_ds'][ds], label_filters=label_filters)
        result = additive.run_slice_additive_with_policy(
            baseline_ctx=context['baseline_ctx_by_ds'][ds],
            slice_spec={'variant': name, 'cell_scales': scales, 'meta': metadata or {}},
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
    out = {
        'name': name,
        'cell_scales': dict(scales),
        'datasets': datasets,
        'active_cells': active_labels(scales),
        'combined_delta_usd': round(float(d5['net_usd']) + float(d1['net_usd']), 2),
        'combined_delta_pf': round(float(d5['profit_factor']) + float(d1['profit_factor']), 4),
        'combined_delta_dd': round(float(d5['max_drawdown_usd']) + float(d1['max_drawdown_usd']), 2),
        'passes_strict': bool(d5['net_usd'] > 0.0 and d1['net_usd'] > 0.0 and d5['profit_factor'] >= 0.0 and d1['profit_factor'] >= 0.0),
    }
    if metadata:
        out['metadata'] = metadata
    return out


def choose_leaders(rows: list[dict[str, Any]]) -> dict[str, Any]:
    passing = [r for r in rows if r.get('passes_strict')]
    if not passing:
        return {'usd_leader': None, 'pf_leader': None, 'dd_adjusted_leader': None}
    usd_leader = max(passing, key=lambda r: (r['combined_delta_usd'], r['combined_delta_pf'], -r['combined_delta_dd']))
    pf_leader = max(passing, key=lambda r: (r['combined_delta_pf'], r['combined_delta_usd'], -r['combined_delta_dd']))
    usd_cutoff = float(usd_leader['combined_delta_usd']) * 0.98
    dd_pool = [r for r in passing if float(r['combined_delta_usd']) >= usd_cutoff]
    dd_adjusted = min(dd_pool, key=lambda r: (r['combined_delta_dd'], -r['combined_delta_pf'], -r['combined_delta_usd']))
    return {
        'usd_leader': {'name': usd_leader['name'], 'combined_delta_usd': usd_leader['combined_delta_usd'], 'combined_delta_pf': usd_leader['combined_delta_pf'], 'combined_delta_dd': usd_leader['combined_delta_dd']},
        'pf_leader': {'name': pf_leader['name'], 'combined_delta_usd': pf_leader['combined_delta_usd'], 'combined_delta_pf': pf_leader['combined_delta_pf'], 'combined_delta_dd': pf_leader['combined_delta_dd']},
        'dd_adjusted_leader': {'name': dd_adjusted['name'], 'combined_delta_usd': dd_adjusted['combined_delta_usd'], 'combined_delta_pf': dd_adjusted['combined_delta_pf'], 'combined_delta_dd': dd_adjusted['combined_delta_dd']},
    }


def trade_attr(trade: dict[str, Any], attr: str) -> Any:
    entry_ts = pd.Timestamp(trade['entry_time'])
    if entry_ts.tzinfo is None:
        entry_ts = entry_ts.tz_localize('UTC')
    entry_ts = entry_ts.tz_convert('UTC')
    if attr == 'weekday':
        return entry_ts.day_name()
    if attr == 'entry_hour_bucket':
        h = int(entry_ts.hour)
        if h < 6:
            return '00_05'
        if h < 12:
            return '06_11'
        if h < 18:
            return '12_17'
        return '18_23'
    if attr == 'entry_hour':
        return int(entry_ts.hour)
    if attr == 'session_bucket':
        return str(trade.get('entry_session', 'unknown'))
    if attr == 'entry_profile':
        return str(trade.get('entry_profile', ''))
    if attr == 'entry_signal_mode':
        return str(trade.get('entry_signal_mode', trade.get('evaluator_mode', '')))
    if attr == 'ownership_cell':
        return str(trade.get('ownership_cell', ''))
    if attr == 'ownership_regime':
        cell = str(trade.get('ownership_cell', ''))
        return cell.split('/')[0] if '/' in cell else cell
    raise KeyError(attr)


def label_value_counts(context: Context, label: str, attr: str, *, datasets: tuple[str, ...] = ('500k', '1000k')) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ds in datasets:
        for trade in context['trades_by_ds'][ds].get(label, []):
            value = trade_attr(trade, attr)
            key = str(value)
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def allowed_filter_values(context: Context, label: str, attr: str, *, min_count: int = 2, top_n: int = 6) -> list[str]:
    counts = label_value_counts(context, label, attr)
    out = [value for value, count in counts.items() if count >= min_count]
    return out[:top_n]


def make_value_disable_filter(attr: str, blocked_value: str, *, subset_predicate: TradeFilter | None = None) -> TradeFilter:
    def _f(trade: dict[str, Any]) -> bool:
        if subset_predicate is not None and not subset_predicate(trade):
            return True
        return str(trade_attr(trade, attr)) != str(blocked_value)
    return _f


def non_strong_c0_predicate(trade: dict[str, Any]) -> bool:
    return str(trade.get('entry_profile', '')).strip().lower() != 'strong'


def write_json_md(json_path: Path, md_path: Path, payload: dict[str, Any], md: str) -> None:
    json_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding='utf-8')
    md_path.write_text(md, encoding='utf-8')
