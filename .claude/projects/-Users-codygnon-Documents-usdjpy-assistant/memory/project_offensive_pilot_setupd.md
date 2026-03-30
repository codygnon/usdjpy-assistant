---
name: Offensive pilot Setup D — shadow logging milestone
description: Status of the narrow Setup D offensive pilot for mean_reversion/er_low/der_neg long-only first_30min
type: project
---

Narrow offensive shadow pilot for the single best expansion candidate.

**Candidate:** london_v2 / Setup D / long-only / mean_reversion/er_low/der_neg / first_30min (signal 15-45min after London open)

**Why:** Only the first offensive slice with real coupled additive evidence — positive on both datasets, stable across 50/75/100% sizing, whitespace cell where Variant K has only 2-5 trades.

**Step 1 (offline replay validation): CLOSED 2026-03-28**
- Parity confirmed on both 500k and 1000k datasets
- Validator script: `scripts/diagnostic_offensive_setupd_replay_validation.py`
- Key parity fix at line 216: channel state consumed on earlier Setup D longs outside target cell
- Pilot script (`scripts/diagnostic_offensive_setupd_whitespace_pilot.py`) still needs the same channel-state fix ported before Step 2

**Step 2 (live shadow logging): NOT STARTED**
- Deploy logging-only into live flow, accumulate events for 60 London trading days
- Prerequisite: port channel-state fix from validator to pilot script

**Step 3 (review board): NOT STARTED**
- After 60 London days, evaluate cell appearance, candidate emission, defensive conflict gates

**User constraint:** No paper trading until system is 100% ready and backtest results are impressive. Pilot is logging-only.

**How to apply:** When working on the offensive pilot, follow this sequence strictly. Do not skip to paper trading. The pilot's only job is answering "does this candidate appear in live flow."
