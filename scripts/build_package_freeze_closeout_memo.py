#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path('/Users/codygnon/Documents/usdjpy_assistant')
OUT_DIR = ROOT / 'research_out'

PACKAGE_FRONTIER_JSON = OUT_DIR / 'package_freeze_frontier.json'
PRUNING_JSON = OUT_DIR / 'package_pruning_frontier.json'
TIME_JSON = OUT_DIR / 'package_time_regime_frontier.json'
EXIT_JSON = OUT_DIR / 'package_slice_exit_frontier.json'
FOLLOWUP_JSON = OUT_DIR / 'package_time_regime_followup.json'
OUT_JSON = OUT_DIR / 'package_freeze_closeout_memo.json'
OUT_MD = OUT_DIR / 'package_freeze_closeout_memo.md'


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Build final closeout memo for package freeze research')
    ap.add_argument('--package-frontier-json', default=str(PACKAGE_FRONTIER_JSON))
    ap.add_argument('--pruning-json', default=str(PRUNING_JSON))
    ap.add_argument('--time-json', default=str(TIME_JSON))
    ap.add_argument('--exit-json', default=str(EXIT_JSON))
    ap.add_argument('--followup-json', default=str(FOLLOWUP_JSON))
    ap.add_argument('--output-json', default=str(OUT_JSON))
    ap.add_argument('--output-md', default=str(OUT_MD))
    return ap.parse_args()


def _load(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding='utf-8'))


def _row_brief(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    out = {
        'name': row.get('name'),
        'combined_delta_usd': row.get('combined_delta_usd'),
        'combined_delta_pf': row.get('combined_delta_pf'),
        'combined_delta_dd': row.get('combined_delta_dd'),
        'passes_strict': row.get('passes_strict'),
    }
    if 'metadata' in row:
        out['metadata'] = row['metadata']
    return out


def _iter_package_rows(package_frontier: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pkg in package_frontier.get('packages', []):
        rows.extend(pkg.get('rows', []))
    return rows


def _iter_pruning_rows(pruning: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pkg in pruning.get('packages', []):
        for target in pkg.get('targets', []):
            rows.extend(target.get('rows', []))
    return rows


def _iter_time_rows(time_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pkg in time_payload.get('packages', []):
        for dim in pkg.get('dimensions', []):
            rows.extend(dim.get('rows', []))
    return rows


def _iter_exit_rows(exit_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in exit_payload.get('targets', []):
        if target.get('status') == 'unsupported':
            continue
        rows.extend(target.get('rows', []))
    return rows


def _iter_followup_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pkg in payload.get('packages', []):
        rows.extend(pkg.get('rows', []))
    return rows


def _leader(rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    passing = [r for r in rows if r.get('passes_strict')]
    if not passing:
        return None
    if key == 'usd':
        winner = max(passing, key=lambda r: (float(r['combined_delta_usd']), float(r['combined_delta_pf']), -float(r['combined_delta_dd'])))
    elif key == 'pf':
        winner = max(passing, key=lambda r: (float(r['combined_delta_pf']), float(r['combined_delta_usd']), -float(r['combined_delta_dd'])))
    elif key == 'dd_adjusted':
        usd = _leader(passing, 'usd')
        if usd is None:
            return None
        cutoff = float(usd['combined_delta_usd']) * 0.98
        pool = [r for r in passing if float(r['combined_delta_usd']) >= cutoff]
        winner = min(pool, key=lambda r: (float(r['combined_delta_dd']), -float(r['combined_delta_pf']), -float(r['combined_delta_usd'])))
    else:
        raise KeyError(key)
    return _row_brief(winner)


def _best_positive_move_against_base(base: dict[str, Any] | None, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if base is None:
        return None
    base_name = str(base.get('name'))
    passing = [r for r in rows if r.get('passes_strict') and r.get('name') != base_name]
    improved = [r for r in passing if float(r['combined_delta_usd']) > float(base['combined_delta_usd']) or float(r['combined_delta_pf']) > float(base['combined_delta_pf'])]
    if not improved:
        return None
    improved.sort(key=lambda r: (float(r['combined_delta_usd']) - float(base['combined_delta_usd']), float(r['combined_delta_pf']) - float(base['combined_delta_pf']), -float(r['combined_delta_dd']) + float(base['combined_delta_dd'])), reverse=True)
    best = improved[0]
    brief = _row_brief(best)
    assert brief is not None
    brief['delta_vs_base'] = {
        'usd': round(float(best['combined_delta_usd']) - float(base['combined_delta_usd']), 2),
        'pf': round(float(best['combined_delta_pf']) - float(base['combined_delta_pf']), 4),
        'dd': round(float(best['combined_delta_dd']) - float(base['combined_delta_dd']), 2),
    }
    return brief


def _slice_exit_summary(exit_payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for target in exit_payload.get('targets', []):
        key = f"{target.get('package')}::{target.get('target_label')}"
        if target.get('status') == 'unsupported':
            out[key] = {
                'status': 'unsupported',
                'reason': target.get('reason'),
            }
            continue
        best = target.get('best_candidate')
        if best is None:
            out[key] = {'status': 'no_variants'}
            continue
        metadata = best.get('metadata', {})
        out[key] = {
            'status': 'tested',
            'best_candidate': _row_brief(best),
            'entry_match_stats': metadata.get('entry_match_stats', {}),
            'baseline_entry_counts': target.get('baseline_entry_counts', {}),
        }
    return out


def _build_stop_conditions(package_frontier: dict[str, Any], pruning: dict[str, Any], time_payload: dict[str, Any], exit_payload: dict[str, Any], followup_payload: dict[str, Any], leaders: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    overall = package_frontier.get('overall_leaders', {})
    if overall.get('usd_leader') and overall.get('pf_leader') and overall.get('dd_adjusted_leader'):
        usd = overall['usd_leader']['name']
        pf = overall['pf_leader']['name']
        dd = overall['dd_adjusted_leader']['name']
        if usd == pf == dd:
            notes.append(f'Package frontier converged on one clear leader: `{usd}`.')
        else:
            notes.append(f'Package frontier compressed into defensible leaders rather than a single monopoly: USD `{usd}`, PF `{pf}`, DD-adjusted `{dd}`.')

    exit_rows = _iter_exit_rows(exit_payload)
    if exit_rows:
        improved = [r for r in exit_rows if r.get('passes_strict')]
        best_exit = max(improved, key=lambda r: (float(r['combined_delta_usd']), float(r['combined_delta_pf']), -float(r['combined_delta_dd'])))
        notes.append(f'Exit frontier closed with best slice-specific candidate `{best_exit["name"]}`; any remaining exit work must beat that frozen-entry benchmark rather than a broad sweep.')
    else:
        notes.append('Exit frontier produced no promotable slice-specific variants beyond the package baselines.')

    prune_rows = _iter_pruning_rows(pruning)
    if prune_rows:
        best_prune = max([r for r in prune_rows if r.get('passes_strict')], key=lambda r: (float(r['combined_delta_usd']), float(r['combined_delta_pf']), -float(r['combined_delta_dd'])), default=None)
        if best_prune is not None:
            notes.append(f'Pruning frontier was explicitly searched; best negative-space candidate was `{best_prune["name"]}`.')

    time_rows = _iter_time_rows(time_payload)
    if time_rows:
        best_time = max([r for r in time_rows if r.get('passes_strict')], key=lambda r: (float(r['combined_delta_usd']), float(r['combined_delta_pf']), -float(r['combined_delta_dd'])), default=None)
        if best_time is not None:
            notes.append(f'Time/regime frontier was explicitly searched; best single-dimension refinement was `{best_time["name"]}`.')
    followup_rows = _iter_followup_rows(followup_payload)
    if followup_rows:
        best_follow = max([r for r in followup_rows if r.get('passes_strict')], key=lambda r: (float(r['combined_delta_usd']), float(r['combined_delta_pf']), -float(r['combined_delta_dd'])), default=None)
        if best_follow is not None:
            notes.append(f'Combined follow-up around `L1` Monday pruning was explicitly searched; best candidate was `{best_follow["name"]}`.')

    if leaders.get('usd_leader') and leaders.get('dd_adjusted_leader'):
        usd = float(leaders['usd_leader']['combined_delta_usd'])
        dd_usd = float(leaders['dd_adjusted_leader']['combined_delta_usd'])
        if usd > 0:
            gap_pct = 100.0 * (usd - dd_usd) / usd
            notes.append(f'DD-adjusted winner remains within `{gap_pct:.2f}%` of the USD leader by construction, satisfying the closeout policy.')
    return notes


def _build_md(payload: dict[str, Any]) -> str:
    leaders = payload['leaders']
    lines = [
        '# Package Freeze Closeout Memo',
        '',
        '- This memo closes the remaining package, exit, pruning, and time/regime frontiers for the promoted research package set.',
        '- DD-adjusted leader is defined as the lowest-DD strict-pass candidate within 2% of the USD leader.',
        '',
        '## Final Leaders',
        '',
    ]
    for key, label in [('usd_leader', 'USD leader'), ('pf_leader', 'PF leader'), ('dd_adjusted_leader', 'DD-adjusted leader')]:
        row = leaders.get(key)
        if row is None:
            lines.append(f'- {label}: `none`')
        else:
            lines.append(f"- {label}: `{row['name']}` | USD `{row['combined_delta_usd']}` | PF `{row['combined_delta_pf']}` | DD `{row['combined_delta_dd']}`")
    lines += ['', '## Frontier Bests', '']
    for key, label in [('package_frontier_best', 'Package frontier'), ('pruning_frontier_best', 'Pruning frontier'), ('time_regime_frontier_best', 'Time/regime frontier'), ('combined_followup_best', 'Combined follow-up'), ('slice_exit_frontier_best', 'Slice exit frontier')]:
        row = payload['frontier_bests'].get(key)
        if row is None:
            lines.append(f'- {label}: `none`')
        else:
            lines.append(f"- {label}: `{row['name']}` | USD `{row['combined_delta_usd']}` | PF `{row['combined_delta_pf']}` | DD `{row['combined_delta_dd']}`")
    lines += ['', '## Frontier Deltas Vs Baselines', '']
    for key, label in [('best_pruning_move_vs_v7_pfdd', 'Best pruning move vs `v7_pfdd`'), ('best_time_move_vs_v6_clean', 'Best time/regime move vs `v6_clean`'), ('best_time_move_vs_v7_pfdd', 'Best time/regime move vs `v7_pfdd`'), ('best_followup_move_vs_v7_pfdd', 'Best combined follow-up vs `v7_pfdd`')]:
        row = payload['baseline_comparisons'].get(key)
        if row is None:
            lines.append(f'- {label}: `none`')
        else:
            dv = row['delta_vs_base']
            lines.append(f"- {label}: `{row['name']}` | dUSD `{dv['usd']}` | dPF `{dv['pf']}` | dDD `{dv['dd']}`")
    lines += ['', '## Slice Exit Closure', '']
    for key, info in payload['slice_exit_summary'].items():
        if info['status'] == 'unsupported':
            lines.append(f"- `{key}`: unsupported | {info['reason']}")
        elif info['status'] == 'no_variants':
            lines.append(f"- `{key}`: no variants tested")
        else:
            best = info['best_candidate']
            lines.append(f"- `{key}`: best `{best['name']}` | USD `{best['combined_delta_usd']}` | PF `{best['combined_delta_pf']}` | DD `{best['combined_delta_dd']}`")
    lines += ['', '## Stop Conditions', '']
    for note in payload['stop_conditions']:
        lines.append(f'- {note}')
    lines += ['', '## Remaining Unclosed Risks', '']
    for note in payload['remaining_unclosed_risks']:
        lines.append(f'- {note}')
    lines.append('')
    return '\n'.join(lines)


def main() -> int:
    args = parse_args()
    package_frontier = _load(args.package_frontier_json)
    pruning = _load(args.pruning_json)
    time_payload = _load(args.time_json)
    exit_payload = _load(args.exit_json)
    followup_payload = _load(args.followup_json)

    package_rows = _iter_package_rows(package_frontier)
    pruning_rows = _iter_pruning_rows(pruning)
    time_rows = _iter_time_rows(time_payload)
    exit_rows = _iter_exit_rows(exit_payload)
    followup_rows = _iter_followup_rows(followup_payload)
    all_rows = package_rows + pruning_rows + time_rows + followup_rows + exit_rows

    leaders = {
        'usd_leader': _leader(all_rows, 'usd'),
        'pf_leader': _leader(all_rows, 'pf'),
        'dd_adjusted_leader': _leader(all_rows, 'dd_adjusted'),
    }

    base_rows = {r.get('name'): r for r in package_rows}

    frontier_bests = {
        'package_frontier_best': _leader(package_rows, 'usd'),
        'pruning_frontier_best': _leader(pruning_rows, 'usd'),
        'time_regime_frontier_best': _leader(time_rows, 'usd'),
        'combined_followup_best': _leader(followup_rows, 'usd'),
        'slice_exit_frontier_best': _leader(exit_rows, 'usd'),
    }

    baseline_comparisons = {
        'best_pruning_move_vs_v7_pfdd': _best_positive_move_against_base(base_rows.get('v7_pfdd__base'), pruning_rows),
        'best_time_move_vs_v6_clean': _best_positive_move_against_base(base_rows.get('v6_clean__base'), time_rows),
        'best_time_move_vs_v7_pfdd': _best_positive_move_against_base(base_rows.get('v7_pfdd__base'), time_rows),
        'best_followup_move_vs_v7_pfdd': _best_positive_move_against_base(base_rows.get('v7_pfdd__base'), followup_rows),
    }

    remaining_risks = []
    for key, info in _slice_exit_summary(exit_payload).items():
        if info['status'] == 'unsupported':
            remaining_risks.append(f'Slice exit frontier not implemented for `{key}`: {info["reason"]}')
    if not time_rows:
        remaining_risks.append('Time/regime frontier did not produce any tested candidates.')
    if not pruning_rows:
        remaining_risks.append('Pruning frontier did not produce any tested candidates.')
    if leaders['usd_leader'] and leaders['pf_leader'] and leaders['usd_leader']['name'] != leaders['pf_leader']['name']:
        remaining_risks.append('USD leader and PF leader still diverge; freezing one package remains a policy choice rather than a mathematical monopoly.')

    payload = {
        'title': 'Package freeze closeout memo',
        'inputs': {
            'package_frontier_json': str(Path(args.package_frontier_json)),
            'pruning_json': str(Path(args.pruning_json)),
            'time_json': str(Path(args.time_json)),
            'exit_json': str(Path(args.exit_json)),
            'followup_json': str(Path(args.followup_json)),
        },
        'leaders': leaders,
        'frontier_bests': frontier_bests,
        'baseline_comparisons': baseline_comparisons,
        'slice_exit_summary': _slice_exit_summary(exit_payload),
        'stop_conditions': _build_stop_conditions(package_frontier, pruning, time_payload, exit_payload, followup_payload, leaders),
        'remaining_unclosed_risks': remaining_risks,
    }

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    out_md.write_text(_build_md(payload), encoding='utf-8')
    print(out_json)
    print(out_md)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
