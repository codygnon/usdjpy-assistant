#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path('/Users/codygnon/Documents/usdjpy_assistant')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.evaluate_defensive_on_freeze_leader as base
import scripts.package_freeze_closeout_lib as lib
import scripts.run_offensive_slice_discovery as discovery

OUT_JSON = ROOT / 'research_out' / 'defensive_on_current_freeze_leader.json'
OUT_MD = ROOT / 'research_out' / 'defensive_on_current_freeze_leader.md'
PACKAGE = 'v7_pfdd'


def main() -> int:
    context = lib.load_context()
    followup_replacement_by_ds = {
        ds: base._build_followup_replacement(context, ds)
        for ds in ['500k', '1000k']
    }
    defensive_ctx_by_ds = {}
    defensive_blocked = {}
    for ds in ['500k', '1000k']:
        defensive_ctx_by_ds[ds], defensive_blocked[ds] = base._build_defensive_baseline_ctx(discovery.DATASETS[ds])

    base_followup = base._package_result(context, PACKAGE, context['baseline_ctx_by_ds'], followup_replacement_by_ds)
    with_defensive = base._package_result(context, PACKAGE, defensive_ctx_by_ds, followup_replacement_by_ds)
    delta = {
        'combined_delta_usd': round(with_defensive['combined_delta_usd'] - base_followup['combined_delta_usd'], 2),
        'combined_delta_pf': round(with_defensive['combined_delta_pf'] - base_followup['combined_delta_pf'], 4),
        'combined_delta_dd': round(with_defensive['combined_delta_dd'] - base_followup['combined_delta_dd'], 2),
    }
    payload = {
        'title': 'Defensive lever on current freeze leader',
        'package': PACKAGE,
        'defensive_rule': {'strategy': base.BLOCK_STRATEGY, 'cell': base.BLOCK_CELL},
        'l1_followup': {
            'drop_weekdays': sorted(base.DROP_WEEKDAYS),
            'tp1r': base.L1_TP1R,
            'be': base.L1_BE,
            'tp2r': base.L1_TP2R,
        },
        'defensive_blocked': defensive_blocked,
        'base_followup': base_followup,
        'with_defensive': with_defensive,
        'delta_vs_base_followup': delta,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=lib._json_default), encoding='utf-8')
    lines = [
        '# Defensive Lever On Current Freeze Leader',
        '',
        f"- package: `{PACKAGE}`",
        f"- defensive rule: `{base.BLOCK_STRATEGY} @ {base.BLOCK_CELL}`",
        f"- L1 follow-up: drop `Monday,Tuesday`, `tp1r={base.L1_TP1R}`, `be={base.L1_BE}`, `tp2r={base.L1_TP2R}`",
        '',
    ]
    for ds, stats in defensive_blocked.items():
        lines.append(f"- defensive block {ds}: blocked `{stats['blocked_count']}` trades | USD `{stats['blocked_net_usd']}` | pips `{stats['blocked_net_pips']}`")
    lines += [
        '',
        f"- base follow-up: USD `{base_followup['combined_delta_usd']}`, PF `{base_followup['combined_delta_pf']}`, DD `{base_followup['combined_delta_dd']}`",
        f"- with defensive: USD `{with_defensive['combined_delta_usd']}`, PF `{with_defensive['combined_delta_pf']}`, DD `{with_defensive['combined_delta_dd']}`",
        f"- delta vs base follow-up: USD `{delta['combined_delta_usd']}`, PF `{delta['combined_delta_pf']}`, DD `{delta['combined_delta_dd']}`",
        '',
    ]
    OUT_MD.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(OUT_JSON)
    print(OUT_MD)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
