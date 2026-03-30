#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path('/Users/codygnon/Documents/usdjpy_assistant')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import backtest_variant_k_v7_search as v7

OUT_JSON = Path('/Users/codygnon/Documents/usdjpy_assistant/research_out/system_variant_k_v7_focus_search.json')
OUT_MD = Path('/Users/codygnon/Documents/usdjpy_assistant/research_out/system_variant_k_v7_focus_search.md')


def main() -> int:
    trades_by_ds = v7._load_inputs(Path(v7.DEFAULT_MATRIX))
    policy = v7._policy()
    baseline_ctx_by_ds = {ds: v7.additive.build_baseline_context(v7.discovery.DATASETS[ds]) for ds in ['500k', '1000k']}

    references = []
    for name, scales in v7.REFERENCE_VARIANTS.items():
        result = v7._evaluate_variant(name, scales, trades_by_ds, policy, baseline_ctx_by_ds)
        references.append(v7._reference_summary(name, result))

    tunables = {
        'L2_brkout_mid_neg_buy': [0.0, 1.0],
        'T1_ambig_high_pos_buy': [0.0, 1.0],
        'T2_brkout_mid_pos_buy': [0.0, 1.0],
        'N3_brkout_low_neg_buy_news': [0.0, 0.5, 1.0],
        'N4_pbt_low_neg_buy_news': [0.0, 0.5, 1.0],
        'L1_mom_low_pos_buy': [0.5, 1.0],
        'T3_ambig_mid_pos_sell': [0.5, 1.0],
    }

    rows = []
    best = None
    labels = list(tunables)
    total = 1
    for scales in tunables.values():
        total *= len(scales)

    import itertools
    for idx, values in enumerate(itertools.product(*(tunables[label] for label in labels)), start=1):
        scales = {label: 1.0 for label in v7.FIXED_CORE}
        scales.update({label: value for label, value in zip(labels, values)})
        result = v7._evaluate_variant(v7._variant_name(scales), scales, trades_by_ds, policy, baseline_ctx_by_ds)
        rows.append(result)
        if result['passes_strict'] and (best is None or (result['combined_delta_usd'], result['combined_delta_pf'], -result['combined_delta_dd']) > (best['combined_delta_usd'], best['combined_delta_pf'], -best['combined_delta_dd'])):
            best = result
        if idx % 50 == 0 or idx == total:
            print(f'Progress {idx}/{total}: best={best["combined_delta_usd"] if best else 0} ({best["name"] if best else "none"})', flush=True)

    rows.sort(key=lambda r: (r['passes_strict'], r['combined_delta_usd'], r['combined_delta_pf'], -r['combined_delta_dd']), reverse=True)
    payload = {
        'title': 'Variant K v7 focused package search',
        'references': references,
        'focus_space': tunables,
        'best_candidate': best,
        'best_candidate_changes': v7._changed_scales(best['cell_scales']) if best else [],
        'top_candidates': [
            {
                'name': row['name'],
                'combined_delta_usd': row['combined_delta_usd'],
                'combined_delta_pf': row['combined_delta_pf'],
                'combined_delta_dd': row['combined_delta_dd'],
                'passes_strict': row['passes_strict'],
            }
            for row in rows[:20]
        ],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=v7._json_default), encoding='utf-8')
    OUT_MD.write_text(v7._build_markdown(payload), encoding='utf-8')
    print(OUT_JSON)
    print(OUT_MD)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
