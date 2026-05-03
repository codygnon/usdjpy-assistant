# Ambiguity Register — Auto Fillmore v2

Open questions where telemetry, corpus, or design context cannot give a
definitive answer. Each entry records what's unknown, why it can't be
resolved now, and the conservative assumption v2 makes in the meantime.

The blueprint's Phase 8 ambiguity register (PHASE8_SYNTHESIS.md §8.7) seeded
this file. Entries below are additive ambiguities discovered during the v2
implementation.

---

## A1. Sparse structured fields in legacy rationale corpus

**Discovered:** Step 1 prerequisite (rationale-persistence inventory).

**Issue:** Pass B of the retroactive replay (Step 8) re-runs the new
validators against the 241-trade corpus to test the validator code itself,
not just the labels. New validators key off explicit JSON fields like
`caveats_detected` and `caveat_resolution`. Older corpus rows (Apr 16-29,
prompt versions before structured-output rollout) only have free-text
`rationale`; structured fields are NULL in 58% of newera8 rows and 77% of
kumatora2 rows.

**Why unresolvable now:** The structured fields didn't exist in the prompt
schema; we cannot reconstruct them.

**Conservative assumption:** Pass B validators run a two-tier matcher: prefer
structured fields when present, fall back to deterministic free-text regex on
`rationale` otherwise. Pass B agreement targets (≥95% V1, ≥90% V2) are
measured stratified by structured-field availability so silent regressions
in the fallback parser are catchable.

---

## A2. Schema hash incorporates a stub prompt version

**Discovered:** Step 1 implementation.

**Issue:** `compute_schema_hash()` mixes `SNAPSHOT_VERSION`, `PROMPT_VERSION`,
and the dataclass field set. PROMPT_VERSION is currently
`"v2.prompt.0-stub"` because the real system prompt lands in Step 6. Until
then, schema hash represents only the snapshot field set + stub prompt.

**Why unresolvable now:** The real prompt doesn't exist yet; pinning a final
PROMPT_VERSION would lie about silent-drift detection.

**Conservative assumption:** Step 6 must bump PROMPT_VERSION (e.g.,
`v2.prompt.1`) when the real prompt lands. The schema_hash on rows written
before that bump will differ from rows after — this is correct behavior, not
a bug. Replays must use the snapshot_version + schema_hash on each row to
look up the prompt that was active.

---

## A3. Path-time MAE/MFE telemetry is best-effort initially

**Discovered:** Step 1 design (PHASE9.8 item 8).

**Issue:** The blueprint requires `path_time_mae_mfe_buckets` (1/3/5/15 min)
to be best-effort initially and required by 50 forward trades. v1's
infrastructure stores only terminal MAE/MFE, not path-ordered buckets.

**Why unresolvable now:** Implementing path-ordered MAE/MFE requires either
high-frequency tick capture or a per-trade poll loop running for ≥15 minutes
of trade duration. That belongs to a later step (likely Step 7's exit layer
work) once an exit-management loop is wired.

**Conservative assumption:** v2 captures `PathTimeMaeMfe` with all fields
None initially. Trailing stops are disabled (per PHASE9.7) until path-time
telemetry is live for ≥50 trades. Tripwire: if 50 forward closes pass without
any path-time data, halt trailing-stop work and audit.

---

## A4. Volatility regime threshold is a placeholder

**Discovered:** Step 1 implementation (`telemetry.classify_volatility_regime`).

**Issue:** Threshold of 10p M5 ATR was chosen as the midpoint between Phase 3's
elevated regime (~15.6p) and a healthy regime (~5.1p) per Phase 7 stop-overshoot
analysis. The true boundary that justifies a sizing reduction is unknown.

**Why unresolvable now:** Forward data on regime-stratified P&L under v2's
deterministic sizing doesn't exist. The number is a heuristic.

**Conservative assumption:** Sizing function (Step 4) halves lots when
`volatility_regime == "elevated"`. Re-tune once Stage 2 (0.1x) accumulates
≥30 closes per regime.

---

## A5. V5 hedge+overconfidence thresholds untested against legacy corpus

**Discovered:** Step 2 shadow replay baseline.

**Issue:** PHASE9.5 specifies V5 thresholds as "top-quartile hedge density
AND top-quartile conviction density" without numeric values. The Step 2
implementation uses 2.0% / 1.5% as proxies. Pooled across the available
241-row legacy corpus, V5 fires zero times. Phase 9's estimate is 5
individual fires (weak N<15), so a near-zero count on a different corpus
is not necessarily wrong — but the threshold is unproven.

**Why unresolvable now:** True top-quartile densities require the live v2
output distribution, which doesn't exist yet. Legacy rationales average
1.6–2.8K chars and use less marketing language than the v2 prompt is
likely to elicit, so legacy is probably not the right calibration target.

**Conservative assumption:** Leave V5 thresholds at 2.0% / 1.5% through
Stage 2. If Step 8 Pass C under-recovers and the gap traces to setups V5
should have caught, recompute thresholds from the first 50 v2 LLM
outputs. Otherwise leave alone.
