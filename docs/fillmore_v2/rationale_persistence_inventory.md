# Rationale Persistence Inventory — Step 1 Prerequisite

**Question gated:** Is the LLM rationale text in `ai_suggestions` stored at full
length, or truncated at write time? This determines whether **Pass B** of the
Step 8 retroactive replay (running new validator code against the historical
241-trade corpus) is feasible.

## Method

Sampled the two largest evidence dumps from the forensic corpus:

- `research_out/autonomous_fillmore_evidence_20260429/newera8/ai_suggestions_autonomous_raw.json` (184 rows)
- `research_out/autonomous_fillmore_evidence_20260429/kumatora2/ai_suggestions_autonomous_raw.json` (57 rows)

Measured `len(rationale)` and `len(trade_thesis)` per row, plus presence of
all text-bearing structured fields with content > 20 chars.

## Findings

| Profile | Rows | Rationale present | min | median | max |
| --- | --- | --- | --- | --- | --- |
| newera8 | 184 | 184/184 | 203 | 1,658 | 4,444 |
| kumatora2 | 57 | 57/57 | 226 | 2,777 | 4,248 |

**Verdict:** Rationale is preserved at full length. SQLite `rationale TEXT`
column has no truncation; writers pass the full LLM response. Pass B is feasible.

## Structured-field availability (caveat)

Newer prompt versions added explicit fields (`trade_thesis`, `caveat_resolution`,
`named_catalyst`, `setup_location`, `edge_reason`, `side_bias_check`, etc.).
These are sparsely populated in older rows:

- newera8: `trade_thesis` present in 78/184 rows (42%)
- kumatora2: `trade_thesis` present in 13/57 rows (23%)

**Implication for Pass B:** Validators that key off structured fields (e.g.,
caveat-resolution validator looking for `caveats_detected`) cannot run pure
field-match logic on legacy rows. They must:

1. Use structured fields when present.
2. Fall back to free-text parsing of `rationale` when absent.
3. Mark the row's prompt version so reconciliation can stratify.

This is a known limitation, not a blocker. Logged in the Ambiguity Register.

## Always-present text-bearing fields (suitable for fallback parsing)

`rationale`, `trigger_family`, `trigger_reason`, `trigger_fit`, `prompt_version`,
`thesis_fingerprint`, `low_rr_edge`, `repeat_trade_case`, `whats_different`,
`why_not_stop`, `why_trade_despite_weakness`, `countertrend_edge`, `exit_plan`.
