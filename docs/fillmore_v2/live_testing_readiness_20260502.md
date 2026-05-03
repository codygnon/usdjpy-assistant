# Auto Fillmore v2 — Stage 1 Live-Testing Readiness

Status: ready for Stage 1 OANDA-practice live testing after operator opt-in.

## What is now wired

- v1 remains the default engine.
- v2 is reached only when `runtime_state.json` has
  `autonomous_fillmore.engine = "v2"`.
- The v2 bridge now populates Stage 1 telemetry from live-loop inputs:
  account equity, open lots by side, unrealized P&L by side, pip value,
  rolling-20 P&L, session labels, volatility regime, proposed side, SL/TP,
  timeframe alignment, and conservative level packet.
- `risk_after_fill_usd` is computed after deterministic sizing and before
  any LLM placement decision is persisted.
- If the adapter/account/bar inputs are absent, v2 still fails closed through
  the existing strike/halt policy.

## Stage 1 boundaries

- Stage 1 is OANDA practice-account validation only.
- The v2 dispatch path is paper-guarded. `api.autonomous_fillmore` calls
  `dispatch_v2_tick(..., stage="paper")`, and `dispatch_v2_tick` refuses
  non-paper stages unless a future caller explicitly passes
  `allow_non_paper=True`.
- v2 can call broker order-send functions only after the v2 orchestrator
  returns `final_decision='place'`, autonomous mode is `paper`, broker type is
  OANDA, and `oanda_environment='practice'`. Any other mode/broker/environment
  fails closed before placement.
- v2 practice placements use the existing broker placement path, so fills,
  broker rejects, local trade rows, and suggestion placement stamps are exercised
  during Stage 1.
- CLR is intentionally conservative: the bridge assigns bar-derived level
  packets `level_quality_score=65`, below CLR thresholds. This prevents fake
  CLR quality from getting through before the real level normalizer exists.
- Momentum-continuation can be exercised because the conservative level packet
  still provides path-room telemetry.
- LLM has no sizing authority.
- LLM has no exit-extension authority.
- Deterministic validators remain in control of final placement.

## Verified

- `pytest tests/test_fillmore_v2_*.py -q` — 248 passed.
- `pytest tests/test_autonomous_fillmore.py -q -k "engine or suggest or runner or veto"` — 26 selected passed.
- Combined targeted readiness run — 273 passed.
- `python3 -m compileall api/fillmore_v2 api/autonomous_fillmore.py` passed.
- `python3 scripts/fillmore_v2_smoke_test.py --no-call` rendered prompt/context successfully.
- Full suite: 1088 passed, 1 unrelated pre-existing failure in
  `tests/test_phase3_additive_runtime.py::test_project_runtime_config_includes_spike_fade_v4_defaults`.

## Operator flip

Only after confirming the account should be in Stage 1 paper mode:

```bash
python3 - <<'PY'
from pathlib import Path
from api.fillmore_v2.engine_flag import set_engine_flag
set_engine_flag(Path("runtime_state.json"), "v2")
PY
```

Return to v1:

```bash
python3 - <<'PY'
from pathlib import Path
from api.fillmore_v2.engine_flag import set_engine_flag
set_engine_flag(Path("runtime_state.json"), "v1")
PY
```

## Must watch during first hour

- v2 rows appear in `ai_suggestions` with `engine_version='v2'`.
- Approved v2 `place` decisions create OANDA practice orders and local
  autonomous trade rows.
- `snapshot_blocking_strikes` stays at 0 when adapter/account/bar inputs are
  healthy.
- No lots above 4.
- Any sell-side placement has validator/pre-veto audit metadata.
- Every placed row has `pip_value_per_lot`, `risk_after_fill_usd`, rendered
  prompt/context, gate candidates, and deterministic sizing inputs.
- Run the first-tick check:

```bash
python3 scripts/fillmore_v2_first_tick_check.py --db /path/to/ai_suggestions.sqlite --profile PROFILE_NAME
```

The script reports missing v2 fields, halt reason, selected gate, final
decision, LLM parse status, validator overrides, and pre-veto fires.

## Rollback

Set the engine back to v1:

```bash
python3 - <<'PY'
from pathlib import Path
from api.fillmore_v2.engine_flag import set_engine_flag, read_engine_flag
state_path = Path("runtime_state.json")
set_engine_flag(state_path, "v1")
print(read_engine_flag(state_path))
PY
```

Confirm next tick uses v1 by checking there are no new
`engine_version='v2'` rows after the rollback timestamp. v2 halt/strike state
is isolated in `runtime_state_fillmore_v2.json`; v1 keeps using
`runtime_state.json`.

## Known Debt

- Full suite currently has one unrelated known failure:
  `tests/test_phase3_additive_runtime.py::test_project_runtime_config_includes_spike_fade_v4_defaults`.
  It is documented as non-v2 launch debt and should not be used as a
  launch/no-launch signal for Stage 1.

## Still required before Stage 2 / 0.1x

- Replace conservative level packets with real side-normalized level quality.
- Upgrade macro/catalyst classification.
- Capture skip-forward outcomes; T5 requires at least 98% skip outcome
  coverage.
- Add operator dashboard/stage progression automation if desired.
- Wire richer exit-layer live replay logs for every v2 paper open/close.
- Build adversarial verifier only if rollout tripwires fire.
- Produce a 50-close Stage 1 paper report: net pips >= 0, simulated PF >= 0.9,
  zero missing blocking telemetry, no protected-cell regression, and no
  stop-widening events.
