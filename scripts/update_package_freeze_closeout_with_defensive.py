#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path('/Users/codygnon/Documents/usdjpy_assistant')
RESEARCH = ROOT / 'research_out'
JSON_OUT = RESEARCH / 'package_freeze_closeout_memo.json'
MD_OUT = RESEARCH / 'package_freeze_closeout_memo.md'


def load(path: str) -> dict:
    return json.loads((RESEARCH / path).read_text(encoding='utf-8'))


def main() -> int:
    defended = load('defensive_on_current_freeze_leader.json')
    l1_325_pfdd = load('l1_followup_exit_check_tp1r325_350_v7_pfdd.json')
    l1_325_usd = load('l1_followup_exit_check_tp1r325_350_v7_usd_max.json')

    defended_row = defended['with_defensive']
    defended_meta = {
        'package': 'v7_pfdd',
        'target_label': 'L1_mom_low_pos_buy',
        'blocked_weekdays': ['Monday', 'Tuesday'],
        'tp1_r_multiple': 3.25,
        'be_offset_pips': 1.0,
        'tp2_r_multiple': 2.0,
        'defensive_rule': defended['defensive_rule'],
    }

    usd_row = l1_325_usd['best_candidate']
    pfdd_row = l1_325_pfdd['best_candidate']

    payload = {
        'title': 'Package freeze closeout memo',
        'inputs': {
            'defended_freeze_json': str(RESEARCH / 'defensive_on_current_freeze_leader.json'),
            'l1_pfdd_json': str(RESEARCH / 'l1_followup_exit_check_tp1r325_350_v7_pfdd.json'),
            'l1_usd_json': str(RESEARCH / 'l1_followup_exit_check_tp1r325_350_v7_usd_max.json'),
            'legacy_closeout_json': str(JSON_OUT),
        },
        'leaders': {
            'research_default_leader': {
                'name': 'v7_pfdd__followup__L1_drop_Monday_Tuesday__tp1r_3.25__be_1__tp2r_2__defensive_v44_ambiguous_low_derneg',
                'combined_delta_usd': defended_row['combined_delta_usd'],
                'combined_delta_pf': defended_row['combined_delta_pf'],
                'combined_delta_dd': defended_row['combined_delta_dd'],
                'passes_strict': defended_row['passes_strict'],
                'metadata': defended_meta,
            },
            'usd_reference_leader': {
                'name': usd_row['name'],
                'combined_delta_usd': usd_row['combined_delta_usd'],
                'combined_delta_pf': usd_row['combined_delta_pf'],
                'combined_delta_dd': usd_row['combined_delta_dd'],
                'passes_strict': True,
                'metadata': {
                    'package': 'v7_usd_max',
                    'target_label': 'L1_mom_low_pos_buy',
                    'blocked_weekdays': ['Monday', 'Tuesday'],
                    'tp1_r_multiple': 3.25,
                    'be_offset_pips': 1.0,
                    'tp2_r_multiple': 2.0,
                },
            },
            'pf_reference_leader': {
                'name': pfdd_row['name'],
                'combined_delta_usd': pfdd_row['combined_delta_usd'],
                'combined_delta_pf': pfdd_row['combined_delta_pf'],
                'combined_delta_dd': pfdd_row['combined_delta_dd'],
                'passes_strict': True,
                'metadata': {
                    'package': 'v7_pfdd',
                    'target_label': 'L1_mom_low_pos_buy',
                    'blocked_weekdays': ['Monday', 'Tuesday'],
                    'tp1_r_multiple': 3.25,
                    'be_offset_pips': 1.0,
                    'tp2_r_multiple': 2.0,
                },
            },
            'dd_adjusted_leader': {
                'name': 'v7_pfdd__followup__L1_drop_Monday_Tuesday__tp1r_3.25__be_1__tp2r_2__defensive_v44_ambiguous_low_derneg',
                'combined_delta_usd': defended_row['combined_delta_usd'],
                'combined_delta_pf': defended_row['combined_delta_pf'],
                'combined_delta_dd': defended_row['combined_delta_dd'],
                'passes_strict': defended_row['passes_strict'],
                'metadata': defended_meta,
            },
        },
        'frontier_bests': {
            'l1_followup_usd_best': {
                'name': usd_row['name'],
                'combined_delta_usd': usd_row['combined_delta_usd'],
                'combined_delta_pf': usd_row['combined_delta_pf'],
                'combined_delta_dd': usd_row['combined_delta_dd'],
            },
            'l1_followup_pf_best': {
                'name': pfdd_row['name'],
                'combined_delta_usd': pfdd_row['combined_delta_usd'],
                'combined_delta_pf': pfdd_row['combined_delta_pf'],
                'combined_delta_dd': pfdd_row['combined_delta_dd'],
            },
            'defended_current_freeze_leader': {
                'name': 'v7_pfdd__followup__L1_drop_Monday_Tuesday__tp1r_3.25__be_1__tp2r_2__defensive_v44_ambiguous_low_derneg',
                'combined_delta_usd': defended_row['combined_delta_usd'],
                'combined_delta_pf': defended_row['combined_delta_pf'],
                'combined_delta_dd': defended_row['combined_delta_dd'],
            },
        },
        'defensive_overlay_effect': {
            'package': defended['package'],
            'defensive_rule': defended['defensive_rule'],
            'blocked': defended['defensive_blocked'],
            'base_followup': {
                'combined_delta_usd': defended['base_followup']['combined_delta_usd'],
                'combined_delta_pf': defended['base_followup']['combined_delta_pf'],
                'combined_delta_dd': defended['base_followup']['combined_delta_dd'],
            },
            'with_defensive': {
                'combined_delta_usd': defended['with_defensive']['combined_delta_usd'],
                'combined_delta_pf': defended['with_defensive']['combined_delta_pf'],
                'combined_delta_dd': defended['with_defensive']['combined_delta_dd'],
            },
            'delta_vs_base_followup': defended['delta_vs_base_followup'],
        },
        'stop_conditions': [
            'The L1 follow-up exit band was extended through tp1r = 3.5; performance improved through 3.25 and reversed at 3.5, establishing a local optimum at 3.25 for the tested band.',
            'The current tested freeze leader is now the defended v7_pfdd follow-up package, not the raw follow-up alone.',
            'The narrow defensive overlay improved the current freeze leader by +2655.13 USD and +0.0523 PF with only +19.75 DD, so it is no longer just a separate defensive finding; it is additive to the promoted package.',
        ],
        'remaining_unclosed_risks': [
            'The same defensive overlay has not yet been replayed on top of the v7_usd_max follow-up package, so there may still be a slightly higher USD variant than the defended v7_pfdd leader.',
        ],
    }

    JSON_OUT.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    md = f'''# Package Freeze Closeout Memo

- This memo promotes the current tested freeze leader after the final L1 exit-band search and defensive overlay check.
- Research-default and DD-adjusted leader are now the same defended package.

## Final Leaders

- Research-default leader: `v7_pfdd__followup__L1_drop_Monday_Tuesday__tp1r_3.25__be_1__tp2r_2__defensive_v44_ambiguous_low_derneg` | USD `{defended_row['combined_delta_usd']}` | PF `{defended_row['combined_delta_pf']}` | DD `{defended_row['combined_delta_dd']}`
- USD reference leader: `{usd_row['name']}` | USD `{usd_row['combined_delta_usd']}` | PF `{usd_row['combined_delta_pf']}` | DD `{usd_row['combined_delta_dd']}`
- PF reference leader: `{pfdd_row['name']}` | USD `{pfdd_row['combined_delta_usd']}` | PF `{pfdd_row['combined_delta_pf']}` | DD `{pfdd_row['combined_delta_dd']}`
- DD-adjusted leader: `v7_pfdd__followup__L1_drop_Monday_Tuesday__tp1r_3.25__be_1__tp2r_2__defensive_v44_ambiguous_low_derneg` | USD `{defended_row['combined_delta_usd']}` | PF `{defended_row['combined_delta_pf']}` | DD `{defended_row['combined_delta_dd']}`

## Defensive Overlay Effect

- Defensive rule: `v44_ny @ ambiguous/er_low/der_neg`
- Base follow-up (`v7_pfdd` + `L1` Monday/Tuesday drop + `L1 tp1r=3.25`): USD `{defended['base_followup']['combined_delta_usd']}`, PF `{defended['base_followup']['combined_delta_pf']}`, DD `{defended['base_followup']['combined_delta_dd']}`
- With defensive overlay: USD `{defended['with_defensive']['combined_delta_usd']}`, PF `{defended['with_defensive']['combined_delta_pf']}`, DD `{defended['with_defensive']['combined_delta_dd']}`
- Delta vs base follow-up: USD `{defended['delta_vs_base_followup']['combined_delta_usd']}`, PF `{defended['delta_vs_base_followup']['combined_delta_pf']}`, DD `{defended['delta_vs_base_followup']['combined_delta_dd']}`

Blocked trades:
- `500k`: `{defended['defensive_blocked']['500k']['blocked_count']}` blocked | USD `{defended['defensive_blocked']['500k']['blocked_net_usd']}` | pips `{defended['defensive_blocked']['500k']['blocked_net_pips']}`
- `1000k`: `{defended['defensive_blocked']['1000k']['blocked_count']}` blocked | USD `{defended['defensive_blocked']['1000k']['blocked_net_usd']}` | pips `{defended['defensive_blocked']['1000k']['blocked_net_pips']}`

## L1 Exit-Band Closure

- `tp1r = 3.0` still improved over `2.75`
- `tp1r = 3.25` improved again and became the peak of the tested band
- `tp1r = 3.5` reversed on both `v7_usd_max` and `v7_pfdd`

That gives us a believable local optimum for the tested `L1` follow-up axis at `tp1r = 3.25`.

## Stop Conditions

- The L1 follow-up exit band was pushed until it reversed at `3.5`.
- The narrow defensive overlay was tested on the current freeze leader and proved additive.
- The defended `v7_pfdd` package is now the strongest fully tested freeze candidate.

## Remaining Risk

- The same defensive overlay has not yet been replayed on top of the `v7_usd_max` follow-up package, so there may still be a slightly higher USD variant than the current defended leader.
'''
    MD_OUT.write_text(md, encoding='utf-8')
    print(JSON_OUT)
    print(MD_OUT)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
