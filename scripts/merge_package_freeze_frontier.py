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
    ROOT / 'research_out' / 'package_freeze_frontier_v6_clean.json',
    ROOT / 'research_out' / 'package_freeze_frontier_v7_best_combo.json',
    ROOT / 'research_out' / 'package_freeze_frontier_v7_usd_max.json',
    ROOT / 'research_out' / 'package_freeze_frontier_v7_pfdd.json',
]
OUT_JSON = ROOT / 'research_out' / 'package_freeze_frontier.json'
OUT_MD = ROOT / 'research_out' / 'package_freeze_frontier.md'


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Merge per-package frontier runs into the canonical package frontier artifact')
    ap.add_argument('--input-json', action='append', dest='inputs')
    ap.add_argument('--output-json', default=str(OUT_JSON))
    ap.add_argument('--output-md', default=str(OUT_MD))
    return ap.parse_args()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def _build_md(payload: dict[str, Any]) -> str:
    lines = [
        '# Package Freeze Frontier',
        '',
        '- Merged canonical frontier from per-package full sizing runs.',
        '- DD-adjusted leader = lowest-DD candidate within 2% of the USD leader.',
        '',
        '## Overall Leaders',
        '',
    ]
    for key in ['usd_leader', 'pf_leader', 'dd_adjusted_leader']:
        row = payload['overall_leaders'].get(key)
        if row is None:
            lines.append(f'- `{key}`: `none`')
        else:
            lines.append(f"- `{key}`: `{row['name']}` | USD `{row['combined_delta_usd']}` | PF `{row['combined_delta_pf']}` | DD `{row['combined_delta_dd']}`")
    lines.append('')
    for package in payload['packages']:
        lines += [
            f"## {package['package']}",
            '',
            f"- variants tested: `{package['variant_count']}`",
            f"- strict-pass variants: `{package['strict_pass_count']}`",
            f"- usd leader: `{package['leaders']['usd_leader']['name'] if package['leaders']['usd_leader'] else 'none'}`",
            f"- pf leader: `{package['leaders']['pf_leader']['name'] if package['leaders']['pf_leader'] else 'none'}`",
            f"- dd-adjusted leader: `{package['leaders']['dd_adjusted_leader']['name'] if package['leaders']['dd_adjusted_leader'] else 'none'}`",
            '',
        ]
        for row in package['top_candidates']:
            lines.append(f"- `{row['name']}`: USD `{row['combined_delta_usd']}`, PF `{row['combined_delta_pf']}`, DD `{row['combined_delta_dd']}`")
        lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> int:
    args = parse_args()
    inputs = [Path(p) for p in (args.inputs or [str(p) for p in DEFAULT_INPUTS])]
    payloads = [_load(path) for path in inputs]
    packages = []
    all_rows: list[dict[str, Any]] = []
    policy = None
    for payload in payloads:
        if policy is None:
            policy = payload.get('policy')
        for pkg in payload.get('packages', []):
            packages.append(pkg)
            all_rows.extend(pkg.get('rows', []))
    merged = {
        'title': 'Package freeze frontier',
        'policy': policy,
        'packages': sorted(packages, key=lambda p: lib.PACKAGE_FRONTIER_ORDER.index(p['package'])),
        'overall_leaders': lib.choose_leaders(all_rows),
    }
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    lib.write_json_md(out_json, out_md, merged, _build_md(merged))
    print(out_json)
    print(out_md)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
