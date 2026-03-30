#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path('/Users/codygnon/Documents/usdjpy_assistant')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import package_freeze_closeout_lib as lib

DEFAULT_INPUTS = [
    ROOT / 'research_out' / 'package_slice_exit_frontier_v7_pfdd_C0_base.json',
    ROOT / 'research_out' / 'package_slice_exit_frontier_v7_pfdd_N1.json',
    ROOT / 'research_out' / 'package_slice_exit_frontier_v7_pfdd_N2.json',
    ROOT / 'research_out' / 'package_slice_exit_frontier_v7_pfdd_L1.json',
    ROOT / 'research_out' / 'package_slice_exit_frontier_v7_best_combo_C0_base.json',
    ROOT / 'research_out' / 'package_slice_exit_frontier_v7_best_combo_N1.json',
    ROOT / 'research_out' / 'package_slice_exit_frontier_v7_best_combo_N2.json',
    ROOT / 'research_out' / 'package_slice_exit_frontier_v7_best_combo_L1.json',
]
OUT_JSON = ROOT / 'research_out' / 'package_slice_exit_frontier.json'
OUT_MD = ROOT / 'research_out' / 'package_slice_exit_frontier.md'


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Merge split slice-exit frontier runs into the canonical artifact')
    ap.add_argument('--input-json', action='append', dest='inputs')
    ap.add_argument('--output-json', default=str(OUT_JSON))
    ap.add_argument('--output-md', default=str(OUT_MD))
    return ap.parse_args()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def _build_md(payload: dict[str, Any]) -> str:
    lines = [
        '# Package Slice Exit Frontier',
        '',
        '- Merged canonical slice-exit frontier from split package/target runs.',
        '- Frozen-entry replacement logic was preserved for every target-specific run.',
        '',
    ]
    for target in payload['targets']:
        lines += [f"## {target['package']} | {target['target_label']}", '']
        if target.get('status') == 'unsupported':
            lines.append(f"- unsupported: `{target['reason']}`")
            lines.append('')
            continue
        lines.append(f"- baseline entry count 500k/1000k: `{target['baseline_entry_counts']['500k']}` / `{target['baseline_entry_counts']['1000k']}`")
        best = target.get('best_candidate')
        if best:
            lines.append(f"- best: `{best['name']}` | USD `{best['combined_delta_usd']}` | PF `{best['combined_delta_pf']}` | DD `{best['combined_delta_dd']}`")
        for row in target.get('top_candidates', []):
            lines.append(f"- `{row['name']}`: USD `{row['combined_delta_usd']}`, PF `{row['combined_delta_pf']}`, DD `{row['combined_delta_dd']}`")
        lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> int:
    args = parse_args()
    inputs = [Path(p) for p in (args.inputs or [str(p) for p in DEFAULT_INPUTS])]
    targets = []
    grids = []
    for path in inputs:
        payload = _load(path)
        if 'grid' in payload:
            grids.append(payload['grid'])
        targets.extend(payload.get('targets', []))
    targets.sort(key=lambda t: (str(t.get('package', '')), str(t.get('target_label', ''))))
    merged = {
        'title': 'Package slice exit frontier',
        'merged_from_split_runs': True,
        'source_files': [str(p) for p in inputs],
        'grids_seen': grids,
        'targets': targets,
    }
    lib.write_json_md(Path(args.output_json), Path(args.output_md), merged, _build_md(merged))
    print(Path(args.output_json))
    print(Path(args.output_md))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
