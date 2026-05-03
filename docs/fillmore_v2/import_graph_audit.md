# Import-Graph Audit — Step 1 Prerequisite

**Question gated:** Does the parallel build (v2 in `api/fillmore_v2/`) share
mutable state with v1 (`api/autonomous_fillmore.py`) or with the runner-score
infrastructure used by the manual trial program?

## v1 → core/

`api/autonomous_fillmore.py` imports from `core/`:

- `core.execution_state.load_state` — file-backed (runtime_state.json)
- `core.json_state.load_json_state`, `save_json_state` — file-backed (runtime_state.json)
- `core.indicators.bollinger_bands` — pure function
- `core.book_cache.get_book_cache` — process-local OANDA order-book cache

**Mutable shared state risk:** `runtime_state.json` is the only persistent
shared surface. v2 must not write to it.

**Resolution:** v2 uses its own state file `runtime_state_fillmore_v2.json`
via `api.fillmore_v2.state`. Test
`test_state_file_is_separate_from_v1` enforces this. The process-level engine
flag prevents concurrent writes; the separate file prevents cross-restart
corruption if the flag flips mid-write.

## fillmore_v2 ↔ core/runner_score

`core/runner_score.py` is the deterministic sizing/exit module owned by the
manual trial program (Trial #10 specifically). Files importing it:

```
run_loop.py
core/runner_score.py
core/execution_engine.py
core/presets.py
core/profile.py
core/execution_state.py
core/dashboard_reporters.py
core/dashboard_builder.py
scripts/analyze_trial10_regime_live_500k.py
scripts/analyze_trial10_runner_elite.py
api/main.py
```

Notable absence: **`api/autonomous_fillmore.py` does NOT import
`runner_score`.** v1 fillmore and the manual trial program were already
decoupled. v2 inherits this decoupling.

**Resolution:** v2 sizing (Step 4) lives in `core/fillmore_v2_sizing.py` and
must not import `runner_score`. Test
`test_no_runner_score_imports_in_fillmore_v2` enforces this in both directions.

## fillmore_v2 → api/suggestion_tracker

v2 calls `api.suggestion_tracker.init_db` to ensure the base `ai_suggestions`
table exists, then `api.fillmore_v2.persistence.init_v2_schema` adds v2-only
columns idempotently. This is read-mostly coupling; v2 owns its own columns
and writes them via its own insert path. v1 writes are unaffected.

`engine_version` is the discriminator; tests confirm `WHERE engine_version =
'v2'` and `WHERE engine_version IS NULL` cleanly partition rows with no joins.
