#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path('/Users/codygnon/Documents/usdjpy_assistant')
RESEARCH = ROOT / 'research_out'
JSON_OUT = RESEARCH / 'paper_candidate_v7_defended.json'
MD_OUT = RESEARCH / 'paper_candidate_v7_defended.md'
CHECKLIST_OUT = RESEARCH / 'paper_trading_go_live_checklist.md'


def load(name: str) -> dict:
    return json.loads((RESEARCH / name).read_text(encoding='utf-8'))


def main() -> int:
    freeze = load('package_freeze_closeout_memo.json')
    defended = load('defensive_on_current_freeze_leader.json')

    leader = freeze['leaders']['research_default_leader']
    effect = freeze['defensive_overlay_effect']

    payload = {
        'title': 'Paper candidate v7 defended',
        'candidate_name': leader['name'],
        'status': 'paper_candidate_frozen_for_runtime_implementation',
        'source_artifacts': {
            'freeze_memo': str(RESEARCH / 'package_freeze_closeout_memo.json'),
            'defensive_overlay_eval': str(RESEARCH / 'defensive_on_current_freeze_leader.json'),
        },
        'package_family': 'v7_pfdd',
        'base_cell_scales': {
            'C1_sell_base': 1.0,
            'C2_sell': 1.0,
            'C3_buy': 1.0,
            'C4_sell_base': 1.0,
            'C5_pbt_sell': 1.0,
            'C6_pbt_sell': 1.0,
            'O0_buy_strong': 1.0,
            'O1_buy_strong': 1.0,
            'O2_buy_strong': 1.0,
            'ADJ_meanrev_low_neg_buy': 1.0,
            'ADJ_ambig_mid_neg_sell': 1.0,
            'ADJ_mom_high_neg_sell': 1.0,
            'N1_brkout_low_neg_sell_strong': 1.0,
            'N2_brkout_low_pos_buy_strong': 1.0,
            'L2_brkout_mid_neg_buy': 1.0,
            'T1_ambig_high_pos_buy': 1.0,
            'T2_brkout_mid_pos_buy': 1.0,
            'N3_brkout_low_neg_buy_news': 1.0,
            'N4_pbt_low_neg_buy_news': 1.0,
            'L1_mom_low_pos_buy': 1.0,
            'T3_ambig_mid_pos_sell': 0.25,
        },
        'overrides': {
            'l1_weekday_disable': ['Monday', 'Tuesday'],
            'l1_exit_override': {
                'tp1_r_multiple': 3.25,
                'be_offset_pips': 1.0,
                'tp2_r_multiple': 2.0,
            },
            'defensive_veto': {
                'strategy': 'v44_ny',
                'ownership_cell': 'ambiguous/er_low/der_neg',
                'mode': 'block',
            },
        },
        'strict_policy': {
            'name': 'native_v44_hedging_like',
            'hedging_enabled': True,
            'allow_internal_overlap': True,
            'allow_opposite_side_overlap': True,
            'max_open_offensive': None,
            'max_entries_per_day': None,
            'margin_model_enabled': True,
            'margin_leverage': 33.3,
            'margin_buffer_pct': 0.0,
            'max_lot_per_trade': 20.0,
        },
        'validated_results': {
            'combined_delta_usd': leader['combined_delta_usd'],
            'combined_delta_pf': leader['combined_delta_pf'],
            'combined_delta_dd': leader['combined_delta_dd'],
            '500k': defended['with_defensive']['datasets']['500k']['delta_vs_baseline'],
            '1000k': defended['with_defensive']['datasets']['1000k']['delta_vs_baseline'],
            '500k_actual_summary': defended['with_defensive']['datasets']['500k']['summary'],
            '1000k_actual_summary': defended['with_defensive']['datasets']['1000k']['summary'],
        },
        'expected_behavior': {
            'defensive_blocked_counts': effect['blocked'],
            'selection_counts': {
                '500k': defended['with_defensive']['datasets']['500k']['selection_counts'],
                '1000k': defended['with_defensive']['datasets']['1000k']['selection_counts'],
            },
            'policy_stats': {
                '500k': defended['with_defensive']['datasets']['500k']['policy_stats'],
                '1000k': defended['with_defensive']['datasets']['1000k']['policy_stats'],
            },
        },
        'paper_mode_intent': {
            'live_mode_unchanged': True,
            'paper_mode_only': True,
            'requires_slice_attribution_logging': True,
            'requires_defensive_veto_logging': True,
            'requires_l1_override_logging': True,
        },
    }

    JSON_OUT.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    md = f'''# Paper Candidate: V7 Defended

## Canonical Candidate

- name: `{leader['name']}`
- status: `paper_candidate_frozen_for_runtime_implementation`
- package family: `v7_pfdd`

## Base Cell Scales

- all core `v7_pfdd` cells at `1.0`
- `T3_ambig_mid_pos_sell` at `0.25`

## Operational Overrides

- `L1_mom_low_pos_buy` disabled on `Monday`, `Tuesday`
- `L1` exit override:
  - `tp1r = 3.25`
  - `be = 1.0`
  - `tp2r = 2.0`
- defensive veto:
  - block `v44_ny` when `ownership_cell == ambiguous/er_low/der_neg`

## Validated Result

- combined delta USD: `{leader['combined_delta_usd']}`
- combined PF: `{leader['combined_delta_pf']}`
- combined DD: `{leader['combined_delta_dd']}`

### 500k
- delta USD: `{defended['with_defensive']['datasets']['500k']['delta_vs_baseline']['net_usd']}`
- delta PF: `{defended['with_defensive']['datasets']['500k']['delta_vs_baseline']['profit_factor']}`
- delta DD: `{defended['with_defensive']['datasets']['500k']['delta_vs_baseline']['max_drawdown_usd']}`
- actual net USD: `{defended['with_defensive']['datasets']['500k']['summary']['net_usd']}`
- actual PF: `{defended['with_defensive']['datasets']['500k']['summary']['profit_factor']}`
- actual DD: `{defended['with_defensive']['datasets']['500k']['summary']['max_drawdown_usd']}`

### 1000k
- delta USD: `{defended['with_defensive']['datasets']['1000k']['delta_vs_baseline']['net_usd']}`
- delta PF: `{defended['with_defensive']['datasets']['1000k']['delta_vs_baseline']['profit_factor']}`
- delta DD: `{defended['with_defensive']['datasets']['1000k']['delta_vs_baseline']['max_drawdown_usd']}`
- actual net USD: `{defended['with_defensive']['datasets']['1000k']['summary']['net_usd']}`
- actual PF: `{defended['with_defensive']['datasets']['1000k']['summary']['profit_factor']}`
- actual DD: `{defended['with_defensive']['datasets']['1000k']['summary']['max_drawdown_usd']}`

## Expected Runtime Evidence

- defensive blocked counts:
  - `500k`: `{effect['blocked']['500k']['blocked_count']}`
  - `1000k`: `{effect['blocked']['1000k']['blocked_count']}`
- paper mode must log:
  - slice attribution
  - `L1` weekday suppression
  - `L1` exit override usage
  - defensive veto hits and prevented trades
'''
    MD_OUT.write_text(md, encoding='utf-8')

    checklist = f'''# Paper Trading Go-Live Checklist

## Canonical Package

- [ ] Use only `/Users/codygnon/Documents/usdjpy_assistant/research_out/paper_candidate_v7_defended.json` as the package definition.
- [ ] Confirm paper mode only; live mode remains unchanged.
- [ ] Confirm the package name in logs is `{leader['name']}`.

## Required Rule Checks

- [ ] `T3_ambig_mid_pos_sell` scale is `0.25`.
- [ ] `L1_mom_low_pos_buy` is disabled on `Monday` and `Tuesday`.
- [ ] `L1` uses `tp1r=3.25`, `be=1.0`, `tp2r=2.0`.
- [ ] Defensive veto blocks `v44_ny` in `ambiguous/er_low/der_neg`.

## Logging / Attribution

- [ ] Every paper trade records its slice label.
- [ ] Every blocked trade records a block reason.
- [ ] `L1` weekday suppression events are visible in logs.
- [ ] `L1` exit override usage is visible in logs.
- [ ] Defensive veto hits are visible in logs.

## Expected Package Behavior

- [ ] `500k`-like replay should be in the neighborhood of:
  - delta USD `{defended['with_defensive']['datasets']['500k']['delta_vs_baseline']['net_usd']}`
  - delta PF `{defended['with_defensive']['datasets']['500k']['delta_vs_baseline']['profit_factor']}`
  - delta DD `{defended['with_defensive']['datasets']['500k']['delta_vs_baseline']['max_drawdown_usd']}`
- [ ] `1000k`-like replay should be in the neighborhood of:
  - delta USD `{defended['with_defensive']['datasets']['1000k']['delta_vs_baseline']['net_usd']}`
  - delta PF `{defended['with_defensive']['datasets']['1000k']['delta_vs_baseline']['profit_factor']}`
  - delta DD `{defended['with_defensive']['datasets']['1000k']['delta_vs_baseline']['max_drawdown_usd']}`

## Operational Guardrails

- [ ] If `L1` fires on `Monday` or `Tuesday`, pause and inspect.
- [ ] If the defensive veto never fires over a meaningful paper sample, audit implementation.
- [ ] If the defensive veto fires far more often than research expectation, audit classification drift.
- [ ] If slice attribution is missing on any paper trade, pause and fix logging.

## Remaining Known Risk

- [ ] Note that the defensive overlay has not yet been replayed on top of the `v7_usd_max` follow-up package.
- [ ] Treat the defended `v7_pfdd` line as the strongest fully tested paper candidate, not necessarily the final absolute USD maximum.
'''
    CHECKLIST_OUT.write_text(checklist, encoding='utf-8')

    print(JSON_OUT)
    print(MD_OUT)
    print(CHECKLIST_OUT)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
