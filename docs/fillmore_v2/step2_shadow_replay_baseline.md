# Step 2 Shadow Replay Baseline

Pooled fire counts from running all five validators against the legacy
corpus via the rationale parser. **These are baselines for Step 8 Pass B
calibration, not the Phase 9 replay numbers themselves.**

Corpus: `research_out/autonomous_fillmore_evidence_20260429/{newera8,kumatora2}/ai_suggestions_autonomous_raw.json`
(241 rows total; 188 place decisions, 53 skips).

## Fire counts (pooled)

| Validator | Fires | Rate over place | Phase 9 expected (PHASE9.5 + 9.9) |
| --- | ---: | ---: | --- |
| `caveat_resolution` | 116 | 61.7% | Covered by V1 floor (~80 sells; here covers both sides) |
| `level_language_overreach` | 93 | 49.5% | V6 individual: 39 fires |
| `loss_asymmetry` | 188 | 100.0% | Not counted separately in PHASE9.5 |
| `sell_side_burden` | 57 | 30.3% | Covered by V1 refined (~80 sells) |
| `hedge_plus_overconfidence` | 0 | 0.0% | V5: 5 fires expected |

## By side

| | buy place | sell place |
| --- | ---: | ---: |
| `caveat_resolution` | 88 | 28 |
| `level_language_overreach` | 48 | 45 |
| `loss_asymmetry` | 131 | 57 |
| `sell_side_burden` | 0 | 57 (100% of sells) |
| `hedge_plus_overconfidence` | 0 | 0 |

## Calibration notes for Step 8 Pass B

1. **`loss_asymmetry` 100% fire rate is an artifact of the legacy adapter,
   not the validator.** Legacy rows rarely populate `why_not_stop` or
   `low_rr_edge`, so the adapter passes `loss_asymmetry_argument=None` and
   the validator correctly fires `missing_argument`. v2 forward path will
   require the field. Pass B must stratify by structured-field availability
   per Ambiguity Register A1.

2. **`sell_side_burden` 100% on sells is also an adapter artifact.** The
   adapter synthesizes `level_quality_score=80` for sell-CLR rows (below the
   85 threshold) because real packet scores don't exist in legacy. Pass C
   will use real packets when v2 is wired and the rate will drop.

3. **`hedge_plus_overconfidence` zero fires.** Either the densities are too
   strict for legacy rationales (which average 1.6–2.8K chars but use less
   marketing language than the validator was tuned for), or both densities
   simply don't co-occur at top-quartile in the available corpus. Phase 9's
   estimate (5 fires individual, weak N<15) suggests the validator should
   fire rarely. Action: leave thresholds as-is; revisit only if Pass C
   under-recovers and V5 is the gap. Tracked in Ambiguity Register A5.

4. **`caveat_resolution` and `level_language_overreach` show strong signal**
   on both sides, consistent with Phase 8's primary cognitive-failure finding.

## Reproducibility

```python
from pathlib import Path
from api.fillmore_v2 import shadow_replay
for f in shadow_replay.find_corpus_files(Path('.')):
    summary = shadow_replay.replay_corpus(f)
    print(f.parent.name, summary.fire_counts)
```
