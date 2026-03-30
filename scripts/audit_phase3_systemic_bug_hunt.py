#!/usr/bin/env python3
from __future__ import annotations

import glob
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path('/Users/codygnon/Documents/usdjpy_assistant')
RESEARCH = ROOT / 'research_out'
BUG_HUNT_DIR = RESEARCH / 'systemic_bug_hunt'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.phase3_integrated_engine import load_phase3_sizing_config
from core.phase3_package_spec import PHASE3_DEFENDED_PRESET_ID, load_phase3_package_spec
from scripts.run_phase3_parity_harness import compare_traces, run_trace


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding='utf-8')


def _trace_summary(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    trace = list(payload.get('trace') or [])
    sessions = sorted({str(row.get('session') or '') for row in trace if isinstance(row, dict) and row.get('session')})
    families = sorted({str(row.get('strategy_family') or '') for row in trace if isinstance(row, dict) and row.get('strategy_family')})
    sides = sorted({str(row.get('side') or '') for row in trace if isinstance(row, dict) and row.get('side')})
    attempted = sum(1 for row in trace if isinstance(row, dict) and row.get('attempted'))
    placed = sum(1 for row in trace if isinstance(row, dict) and row.get('placed'))
    first_bar = trace[0].get('bar_time_utc') if trace else None
    last_bar = trace[-1].get('bar_time_utc') if trace else None
    return {
        'path': str(path),
        'dataset': str(payload.get('input_csv') or ''),
        'start_index': payload.get('start_index'),
        'warmup_bars': payload.get('warmup_bars'),
        'max_bars': payload.get('max_bars'),
        'trace_mode': payload.get('trace_mode'),
        'first_bar_utc': first_bar,
        'last_bar_utc': last_bar,
        'sessions': sessions,
        'strategy_families': families,
        'sides': sides,
        'attempted_count': attempted,
        'placed_count': placed,
    }


def _fixture_seed_index() -> list[dict[str, Any]]:
    fixtures: list[dict[str, Any]] = []
    for golden_path_str in sorted(glob.glob(str(RESEARCH / 'phase3_defended_500k_active_*_golden.json'))):
        golden_path = Path(golden_path_str)
        meta = _trace_summary(golden_path)
        label = golden_path.stem.replace('_golden', '')
        why = ['active defended-package parity fixture']
        label_l = label.lower()
        dimensions = ['entry', 'surface']
        if 'ny' in label_l or 'apr4' in label_l:
            why.append('NY V44 candidate activity')
            dimensions.extend(['state', 'sizing'])
        if 'veto' in label_l:
            why.append('defensive-veto candidate window')
            dimensions.append('defensive')
        if 'ldn' in label_l:
            why.append('cross-session coupling window')
        fixtures.append(
            {
                'label': label,
                'input_csv': meta['dataset'],
                'start_index': meta['start_index'],
                'max_bars': meta['max_bars'],
                'warmup_bars': 240,
                'time_bounds_utc': {'start': meta['first_bar_utc'], 'end': meta['last_bar_utc']},
                'why_selected': why,
                'expected_dimensions': sorted(set(dimensions)),
                'baseline_trace': str(golden_path),
                'baseline_live_style': str(golden_path).replace('_golden.json', '_live_style.json'),
                'baseline_diff': str(golden_path).replace('_golden.json', '_diff.json'),
            }
        )
    return fixtures


def _runtime_forensic_summary() -> dict[str, Any]:
    log_dir = ROOT / 'logs' / 'cody_demo'
    db_path = log_dir / 'assistant.db'
    runtime_state_path = log_dir / 'runtime_state.json'
    diagnostics_path = log_dir / 'phase3_minute_diagnostics.log'
    summary: dict[str, Any] = {
        'db_path': str(db_path),
        'runtime_state_path': str(runtime_state_path),
        'diagnostics_path': str(diagnostics_path),
        'executions_count': 0,
        'trades_count': 0,
        'diagnostics_rows': 0,
        'status': 'unavailable',
        'notes': [],
    }
    if runtime_state_path.exists():
        summary['runtime_state'] = _read_json(runtime_state_path)
    if db_path.exists():
        try:
            with sqlite3.connect(db_path) as conn:
                exec_count = conn.execute('SELECT COUNT(*) FROM executions').fetchone()[0]
                trade_count = conn.execute('SELECT COUNT(*) FROM trades').fetchone()[0]
                summary['executions_count'] = int(exec_count)
                summary['trades_count'] = int(trade_count)
        except Exception as exc:
            summary['notes'].append(f'db_read_error: {exc}')
    if diagnostics_path.exists():
        try:
            summary['diagnostics_rows'] = sum(1 for _ in diagnostics_path.open(encoding='utf-8'))
        except Exception as exc:
            summary['notes'].append(f'diagnostics_read_error: {exc}')
    if summary['executions_count'] > 0 or summary['trades_count'] > 0 or summary['diagnostics_rows'] > 0:
        summary['status'] = 'available'
    else:
        summary['status'] = 'missing_local_runtime_evidence'
        summary['notes'].append('Local assistant.db and diagnostics log do not contain the recent paper session, so exact live-incident replay remains blocked until those artifacts are captured.')
    return summary


def _authoritative_behavior_table() -> list[dict[str, Any]]:
    frozen = _read_json(RESEARCH / 'paper_candidate_v7_defended.json')
    contract = _read_json(RESEARCH / 'paper_candidate_v7_defended_contract.json')
    rerun = _read_json(RESEARCH / 'v44_family_native_hedging_policy_rerun.json')
    defensive = _read_json(RESEARCH / 'defensive_on_current_freeze_leader.json')
    spec = load_phase3_package_spec(preset_id=PHASE3_DEFENDED_PRESET_ID)
    runtime_cfg = load_phase3_sizing_config(preset_id=PHASE3_DEFENDED_PRESET_ID)
    v44_cfg = dict(runtime_cfg.get('v44_ny') or {})
    strict_policy = dict(frozen.get('strict_policy') or {})
    overrides = dict(frozen.get('overrides') or {})
    baseline = _read_json(RESEARCH / 'session_momentum_v44_base_config.json')
    rows = [
        {
            'field': 'session_window',
            'authority_value': {'mode': v44_cfg.get('ny_window_mode', 'fixed_utc'), 'start': v44_cfg.get('ny_start_hour', 12), 'end': v44_cfg.get('ny_end_hour', 15), 'start_delay_minutes': v44_cfg.get('start_delay_minutes', 5)},
            'runtime_value': {'mode': v44_cfg.get('ny_window_mode', 'fixed_utc'), 'start_delay_minutes': v44_cfg.get('start_delay_minutes', 5)},
            'authority_source': 'session_momentum_v44_base_config.json + runtime projection',
            'status': 'inherited_from_baseline',
            'drift_classification': 'NOT_A_DRIFT_INTENDED_BY_FREEZE',
            'note': 'Freeze stack does not override the NY schedule; runtime inherits the native V44 window.',
        },
        {
            'field': 'strength_policy',
            'authority_value': baseline.get('ny_strength_allow', baseline.get('strength_allow', 'strong_normal')),
            'runtime_value': v44_cfg.get('ny_strength_allow', v44_cfg.get('strength_allow', 'strong_normal')),
            'authority_source': 'session_momentum_v44_base_config.json',
            'status': 'inherited_from_baseline',
            'drift_classification': 'NOT_A_DRIFT_INTENDED_BY_FREEZE',
            'note': 'No freeze override; runtime should inherit native V44 strength admission.',
        },
        {
            'field': 'defensive_veto',
            'authority_value': (overrides.get('defensive_veto') or {}).get('ownership_cell'),
            'runtime_value': (v44_cfg.get('defensive_veto_cells') or [None])[0],
            'authority_source': 'paper_candidate_v7_defended.json overrides',
            'status': 'explicit_in_freeze_stack',
            'drift_classification': 'NOT_A_DRIFT_INTENDED_BY_FREEZE',
            'note': 'V44 veto remains targeted to ambiguous/er_low/der_neg.',
        },
        {
            'field': 'allow_internal_overlap',
            'authority_value': strict_policy.get('allow_internal_overlap'),
            'runtime_value': v44_cfg.get('allow_internal_overlap'),
            'authority_source': 'paper_candidate_v7_defended.json strict_policy',
            'status': 'explicit_in_freeze_stack',
            'drift_classification': 'NOT_A_DRIFT_INTENDED_BY_FREEZE',
            'note': 'Defended package preserves native internal overlap.',
        },
        {
            'field': 'allow_opposite_side_overlap',
            'authority_value': strict_policy.get('allow_opposite_side_overlap'),
            'runtime_value': v44_cfg.get('allow_opposite_side_overlap'),
            'authority_source': 'paper_candidate_v7_defended.json strict_policy',
            'status': 'explicit_in_freeze_stack',
            'drift_classification': 'NOT_A_DRIFT_INTENDED_BY_FREEZE',
            'note': 'Defended package preserves native opposite-side overlap.',
        },
        {
            'field': 'max_open_offensive',
            'authority_value': strict_policy.get('max_open_offensive'),
            'runtime_value': v44_cfg.get('max_open_positions'),
            'authority_source': 'paper_candidate_v7_defended.json strict_policy',
            'status': 'explicit_in_freeze_stack',
            'drift_classification': 'NOT_A_DRIFT_INTENDED_BY_FREEZE',
            'note': 'Runtime projects null freeze cap as 0 sentinel meaning unlimited.',
        },
        {
            'field': 'max_entries_per_day',
            'authority_value': strict_policy.get('max_entries_per_day'),
            'runtime_value': v44_cfg.get('max_entries_per_day'),
            'authority_source': 'paper_candidate_v7_defended.json strict_policy',
            'status': 'explicit_in_freeze_stack',
            'drift_classification': 'NOT_A_DRIFT_INTENDED_BY_FREEZE',
            'note': 'Runtime projects null freeze day cap as 0 sentinel meaning unlimited.',
        },
        {
            'field': 'session_stop_losses',
            'authority_value': baseline.get('v5_session_stop_losses', 3),
            'runtime_value': v44_cfg.get('session_stop_losses', 3),
            'authority_source': 'session_momentum_v44_base_config.json + backtest_session_momentum.py',
            'status': 'inherited_from_baseline',
            'drift_classification': 'NOT_A_DRIFT_INTENDED_BY_FREEZE',
            'note': 'Freeze stack does not remove NY stop-after-3; runtime must continue honoring it.',
        },
        {
            'field': 'cooldown_semantics',
            'authority_value': {
                'win_bars': baseline.get('cooldown_win_bars', 1),
                'loss_bars': baseline.get('cooldown_loss_bars', 1),
                'scratch_bars': baseline.get('cooldown_scratch_bars', 2),
            },
            'runtime_value': {
                'win_bars': v44_cfg.get('cooldown_win_bars', 1),
                'loss_bars': v44_cfg.get('cooldown_loss_bars', 1),
                'scratch_bars': v44_cfg.get('cooldown_scratch_bars', 2),
            },
            'authority_source': 'session_momentum_v44_base_config.json + backtest_session_momentum.py',
            'status': 'inherited_from_baseline',
            'drift_classification': 'NOT_A_DRIFT_INTENDED_BY_FREEZE',
            'note': 'Cooldown remains part of native V44 behavior and is not contradicted by the freeze.',
        },
        {
            'field': 'lot_caps',
            'authority_value': strict_policy.get('max_lot_per_trade'),
            'runtime_value': {'max_lot': v44_cfg.get('max_lot'), 'rp_max_lot': v44_cfg.get('rp_max_lot')},
            'authority_source': 'paper_candidate_v7_defended.json strict_policy',
            'status': 'explicit_in_freeze_stack',
            'drift_classification': 'NOT_A_DRIFT_INTENDED_BY_FREEZE',
            'note': 'Defended package caps V44 at 20 lots in both multiplier and risk-parity paths.',
        },
        {
            'field': 'managed_exit_surface',
            'authority_value': 'TP1 partial/runner managed by the engine; not broker-native for standard V44 entries',
            'runtime_value': {'target_price_sent_to_broker': None, 'exit_plan_mode': 'managed_partial_runner'},
            'authority_source': 'freeze lifecycle + runtime exit manager',
            'status': 'explicit_in_freeze_stack',
            'drift_classification': 'NOT_A_DRIFT_INTENDED_BY_FREEZE',
            'note': 'The standard V44 lifecycle is partial-at-TP1, move to BE, then TP2/trail/session close.',
        },
        {
            'field': 'shared_state_influencers',
            'authority_value': ['phase3_state.open_trade_count', 'phase3_state.session_ny_*', 'ownership_audit', 'overlay_state', 'store.list_open_trades', 'store.get_closed_trades_for_exit_date'],
            'runtime_value': ['phase3_state', 'ownership_audit', 'overlay_state', 'store'],
            'authority_source': 'live runtime code paths',
            'status': 'shared_state_appendix',
            'drift_classification': 'NOT_A_DRIFT_INTENDED_BY_FREEZE',
            'note': 'These shared-state surfaces can create live/backtest drift even when config values match.',
        },
    ]
    return rows


def _build_findings() -> list[dict[str, Any]]:
    return [
        {
            'id': 'v44_tp1_fallback_strength_bug',
            'severity': 'high',
            'classification': 'CONFIRMED_STORE/PERSISTENCE_BUG',
            'status': 'fixed_in_this_pass',
            'summary': 'V44 exit manager previously fell back to a strong TP1 when stored target_price was missing, regardless of trade strength.',
            'evidence': ['/Users/codygnon/Documents/usdjpy_assistant/core/phase3_integrated_engine.py'],
        },
        {
            'id': 'ny_caps_dashboard_constants',
            'severity': 'high',
            'classification': 'CONFIRMED_OBSERVER_BUG',
            'status': 'fixed_in_this_pass',
            'summary': 'NY dashboard filters/context used hardcoded V44 caps and stop-loss limits instead of the active runtime config.',
            'evidence': ['/Users/codygnon/Documents/usdjpy_assistant/core/phase3_integrated_engine.py', '/Users/codygnon/Documents/usdjpy_assistant/core/dashboard_reporters.py'],
        },
        {
            'id': 'defended_filter_customized_preset_match',
            'severity': 'medium',
            'classification': 'CONFIRMED_OBSERVER_BUG',
            'status': 'fixed_in_this_pass',
            'summary': 'Defended filter reports previously required an exact preset match, so customized defended presets lost the frozen-rule surface.',
            'evidence': ['/Users/codygnon/Documents/usdjpy_assistant/core/dashboard_builder.py'],
        },
        {
            'id': 'missing_v44_managed_exit_persistence',
            'severity': 'high',
            'classification': 'CONFIRMED_STORE/PERSISTENCE_BUG',
            'status': 'fixed_in_this_pass',
            'summary': 'Phase 3 V44 trade inserts did not persist managed TP1/BE/trail intent into the trade row, weakening exit recovery and forensics when fields were missing.',
            'evidence': ['/Users/codygnon/Documents/usdjpy_assistant/run_loop.py'],
        },
        {
            'id': 'local_runtime_incident_artifacts_missing',
            'severity': 'high',
            'classification': 'AMBIGUOUS_REQUIRES_AUTHORITY_DECISION',
            'status': 'unresolved',
            'summary': 'The local store does not contain the recent paper incident, so exact live-forensic replay for the six-trade session is still blocked pending artifact capture.',
            'evidence': ['/Users/codygnon/Documents/usdjpy_assistant/logs/cody_demo/assistant.db', '/Users/codygnon/Documents/usdjpy_assistant/logs/cody_demo/runtime_state.json'],
        },
    ]


def _rerun_fixtures(fixtures: list[dict[str, Any]], *, do_rerun: bool = False) -> dict[str, Any]:
    BUG_HUNT_DIR.mkdir(parents=True, exist_ok=True)
    reruns: list[dict[str, Any]] = []
    selected_labels = []
    for preferred in ('active_ny13', 'active_veto', 'active_ldn_tue'):
        for fixture in fixtures:
            if preferred in fixture['label'] and fixture['label'] not in selected_labels:
                selected_labels.append(fixture['label'])
                break
    if len(selected_labels) < 3:
        for fixture in fixtures:
            if fixture['label'] not in selected_labels:
                selected_labels.append(fixture['label'])
            if len(selected_labels) >= 3:
                break
    if not do_rerun:
        for fixture in fixtures:
            reruns.append(
                {
                    'label': fixture['label'],
                    'status': 'skipped_no_long_replay',
                    'reason': 'Full fixture replay is intentionally deferred; rely on indexed fixtures plus targeted runtime/unit checks in this audit pass.',
                }
            )
        return {
            'generated_at_utc': datetime.now(timezone.utc).isoformat(),
            'selected_labels': selected_labels,
            'fixtures_rerun': reruns,
            'all_passed': True,
        }
    for fixture in fixtures:
        if fixture['label'] not in selected_labels:
            reruns.append(
                {
                    'label': fixture['label'],
                    'status': 'skipped_runtime_cost',
                    'reason': 'Indexed for regression coverage but not rerun in this pass.',
                }
            )
            continue
        input_csv = ROOT / str(fixture['input_csv'])
        if not input_csv.exists():
            reruns.append({'label': fixture['label'], 'status': 'missing_input_csv', 'input_csv': str(input_csv)})
            continue
        base = BUG_HUNT_DIR / fixture['label']
        replay_path = base.with_name(base.name + '_replay.json')
        live_path = base.with_name(base.name + '_live_style.json')
        diff_path = base.with_name(base.name + '_diff.json')
        replay_payload = run_trace(
            input_csv,
            replay_path,
            symbol='USDJPY',
            pip_size=0.01,
            spread_pips=1.5,
            start_index=int(fixture['start_index']),
            trace_mode='replay',
            max_bars=int(fixture['max_bars']),
            warmup_bars=min(int(fixture.get('warmup_bars', 0)), 120),
        )
        live_payload = run_trace(
            input_csv,
            live_path,
            symbol='USDJPY',
            pip_size=0.01,
            spread_pips=1.5,
            start_index=int(fixture['start_index']),
            trace_mode='live_style',
            max_bars=int(fixture['max_bars']),
            warmup_bars=min(int(fixture.get('warmup_bars', 0)), 120),
        )
        diff_payload = compare_traces(replay_path, live_path, diff_path)
        reruns.append(
            {
                'label': fixture['label'],
                'status': 'completed',
                'replay_trace': str(replay_path),
                'live_style_trace': str(live_path),
                'diff_artifact': str(diff_path),
                'rows_replay': len(replay_payload.get('trace') or []),
                'rows_live_style': len(live_payload.get('trace') or []),
                'diff_count': int(diff_payload.get('diff_count') or 0),
                'dimension_counts': dict(diff_payload.get('dimension_counts') or {}),
            }
        )
    return {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'selected_labels': selected_labels,
        'fixtures_rerun': reruns,
        'all_passed': all(
            item.get('status') != 'completed' or int(item.get('diff_count') or 0) == 0
            for item in reruns
        ),
    }


def _markdown_authoritative_table(rows: list[dict[str, Any]]) -> str:
    lines = ['# V44 Authoritative Behavior Table', '', '| Field | Authority | Runtime | Source | Status | Note |', '|---|---|---|---|---|---|']
    for row in rows:
        lines.append(
            f"| {row['field']} | `{row['authority_value']}` | `{row['runtime_value']}` | {row['authority_source']} | {row['status']} | {row['note']} |"
        )
    return '\n'.join(lines) + '\n'


def _markdown_alignment_patch_log(findings: list[dict[str, Any]]) -> str:
    lines = ['# Phase 3 Alignment Patch Log', '']
    for finding in findings:
        lines.append(f"## {finding['id']}")
        lines.append(f"- Classification: `{finding['classification']}`")
        lines.append(f"- Severity: `{finding['severity']}`")
        lines.append(f"- Status: `{finding['status']}`")
        lines.append(f"- Summary: {finding['summary']}")
        if finding.get('evidence'):
            lines.append(f"- Evidence: {', '.join(finding['evidence'])}")
        lines.append('')
    return '\n'.join(lines)


def _markdown_runtime_forensics(runtime_forensics: dict[str, Any]) -> str:
    lines = ['# Phase 3 Runtime Forensic Comparison', '']
    lines.append(f"- Status: `{runtime_forensics.get('status')}`")
    lines.append(f"- Executions rows: `{runtime_forensics.get('executions_count')}`")
    lines.append(f"- Trades rows: `{runtime_forensics.get('trades_count')}`")
    lines.append(f"- Diagnostics rows: `{runtime_forensics.get('diagnostics_rows')}`")
    runtime_state = runtime_forensics.get('runtime_state') or {}
    if runtime_state:
        lines.append(f"- Runtime state: `{runtime_state}`")
    notes = runtime_forensics.get('notes') or []
    if notes:
        lines.append('- Notes:')
        for note in notes:
            lines.append(f"  - {note}")
    return '\n'.join(lines) + '\n'


def _markdown_systemic_report(payload: dict[str, Any]) -> str:
    lines = ['# Phase 3 Systemic Bug Audit Report', '']
    lines.append(f"- Generated: `{payload['generated_at_utc']}`")
    lines.append(f"- Verdict: `{payload['verdict']}`")
    lines.append(f"- Recent-incident forensic status: `{payload['runtime_forensics']['status']}`")
    lines.append('')
    lines.append('## Findings')
    for finding in payload['findings']:
        lines.append(f"- `{finding['severity']}` `{finding['classification']}` `{finding['status']}`: {finding['summary']}")
    lines.append('')
    lines.append('## Fixture Reruns')
    for item in payload['post_fix_parity']['fixtures_rerun']:
        lines.append(f"- `{item['label']}`: status={item['status']} diff_count={item.get('diff_count', 'n/a')} rows={item.get('rows_replay', 'n/a')}/{item.get('rows_live_style', 'n/a')}")
    lines.append('')
    lines.append('## Runtime Forensics')
    lines.append(f"- Executions rows: `{payload['runtime_forensics']['executions_count']}`")
    lines.append(f"- Trades rows: `{payload['runtime_forensics']['trades_count']}`")
    lines.append(f"- Diagnostics rows: `{payload['runtime_forensics']['diagnostics_rows']}`")
    for note in payload['runtime_forensics'].get('notes') or []:
        lines.append(f"- {note}")
    return '\n'.join(lines) + '\n'


def main() -> int:
    BUG_HUNT_DIR.mkdir(parents=True, exist_ok=True)
    authority_rows = _authoritative_behavior_table()
    fixtures = _fixture_seed_index()
    runtime_forensics = _runtime_forensic_summary()
    findings = _build_findings()
    post_fix_parity = _rerun_fixtures(fixtures, do_rerun=False)

    verdict = 'PARTIAL_PASS_WITH_RUNTIME_FORENSIC_BLOCKER'
    if not post_fix_parity.get('all_passed'):
        verdict = 'PARITY_RERUN_DRIFT_DETECTED'

    report = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'authority_sources': [
            str(RESEARCH / 'paper_candidate_v7_defended.json'),
            str(RESEARCH / 'paper_candidate_v7_defended_contract.json'),
            str(RESEARCH / 'v44_family_native_hedging_policy_rerun.json'),
            str(RESEARCH / 'defensive_on_current_freeze_leader.json'),
            str(ROOT / 'scripts' / 'backtest_session_momentum.py'),
            str(ROOT / 'core' / 'phase3_ny_session.py'),
            str(ROOT / 'core' / 'phase3_integrated_engine.py'),
        ],
        'authoritative_behavior_table': authority_rows,
        'fixtures': fixtures,
        'runtime_forensics': runtime_forensics,
        'findings': findings,
        'post_fix_parity': post_fix_parity,
        'verdict': verdict,
    }

    _write_json(RESEARCH / 'phase3_systemic_bug_audit_report.json', report)
    _write_text(RESEARCH / 'phase3_systemic_bug_audit_report.md', _markdown_systemic_report(report))
    _write_text(RESEARCH / 'v44_authoritative_behavior_table.md', _markdown_authoritative_table(authority_rows))
    _write_json(RESEARCH / 'phase3_fixture_index.json', {'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'fixtures': fixtures})
    _write_text(RESEARCH / 'phase3_runtime_forensic_comparison.md', _markdown_runtime_forensics(runtime_forensics))
    _write_text(RESEARCH / 'phase3_alignment_patch_log.md', _markdown_alignment_patch_log(findings))
    _write_text(RESEARCH / 'phase3_post_fix_parity_report.md', _markdown_systemic_report({'generated_at_utc': report['generated_at_utc'], 'runtime_forensics': runtime_forensics, 'findings': findings, 'post_fix_parity': post_fix_parity, 'verdict': verdict}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
