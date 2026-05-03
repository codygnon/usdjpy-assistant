# Step 3 Pre-Veto Baseline

Pooled pre-decision veto fire counts on the legacy corpus (241 rows total).
**Reference numbers — not the official Phase 9 replay.** Step 8 Pass A is the
gate that asserts +278.4p / +$5,684.56 against the 241-trade Phase 8 universe.

## Pooled fire counts

| Veto | Fires | Phase 9 expected (PHASE9.9) | Notes |
| --- | ---: | ---: | --- |
| `v1_sell_caveat_template` | 89 | 80 | Within 50-150% of expectation. All fires on sells (0 buy fires) — correct architectural invariant. |
| `v2_mixed_overlap` | 43 | 44 | Near-exact match. Bypass preserves protected buy-CLR cells. |
| **Total skipped pre-call** | 100 | ~100 | Matches PHASE9.9 trade-count target. |

## By side

| | buy | sell |
| --- | ---: | ---: |
| `v1_sell_caveat_template` | 0 | 89 |
| `v2_mixed_overlap` | 11 | 32 |

The 11 buy V2 fires are non-CLR buys or buy-CLR with `level_quality_score < 70`,
which fall outside the protected bypass per design.

## Downstream effect

After pre-vetoes skip-before-call, the post-decision `sell_side_burden`
validator drops to **0 fires** on the corpus (was 57 in Step 2). The bad
sells that would have triggered it never reach the LLM. This is the
intended architectural behavior — pre-decision blocks are cheaper than
post-decision overrides because no token cost is paid.

| Validator | Step 2 alone | After Step 3 pre-vetoes |
| --- | ---: | ---: |
| `caveat_resolution` | 116 | 88 |
| `level_language_overreach` | 93 | 44 |
| `loss_asymmetry` | 188 | 127 |
| `sell_side_burden` | 57 | **0** |
| `hedge_plus_overconfidence` | 0 | 0 |

## Reproducibility

```python
from pathlib import Path
from api.fillmore_v2 import shadow_replay
for f in shadow_replay.find_corpus_files(Path('.')):
    s = shadow_replay.replay_corpus(f)
    print(s.pre_veto_fire_counts, s.pre_veto_skipped_before_call)
```
