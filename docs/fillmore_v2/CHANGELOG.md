# Auto Fillmore v2 — Changelog

Per the Phase 9 build directive: one entry per component built, citing the
corresponding blueprint section.

---

## Step 1 — Telemetry & Snapshot Layer (PHASE9.8)

**Added:**

- `api/fillmore_v2/__init__.py` — `ENGINE_VERSION="v2"`, `SNAPSHOT_VERSION`,
  `PROMPT_VERSION`, and durable v2.0.0 schema-hash baseline constants.
  PROMPT_VERSION is a stub until Step 6.
- `api/fillmore_v2/snapshot.py` — `Snapshot` dataclass with all 14 priority
  telemetry fields, schema-hash drift detection, blocking-field check, and
  3-strike halt logic against the v2 state file.
- `api/fillmore_v2/state.py` — separate runtime state file
  (`runtime_state_fillmore_v2.json`) so v2 cannot corrupt v1's `runtime_state.json`.
- `api/fillmore_v2/telemetry.py` — capture functions for exposure (lots by
  side, unrealized P&L by side), sizing inputs (pip value, risk-after-fill),
  drawdown awareness (rolling-20 P&L lot-weighted and raw), side-normalized
  level packet, level age metadata, volatility regime classifier.
- `api/fillmore_v2/persistence.py` — additive schema migration adding 23
  v2 columns to `ai_suggestions`. Idempotent. Pre-cutover rows keep NULL;
  no backfill. v1 and v2 rows distinguishable by `engine_version` alone.
- `tests/test_fillmore_v2_telemetry.py` — 22 tests covering all of the
  above. Includes the Step 1 acceptance test (paper-trade fake gate event
  end-to-end, replay from logs alone), the v1/v2 row-distinguishability
  test, the import-graph audit (no `runner_score` in either direction), and
  the schema-hash silent-drift catcher.

**Audit docs:**

- `docs/fillmore_v2/rationale_persistence_inventory.md` — gates Step 8
  Pass B. Rationale preserved at full length (median 1,650-2,800 chars).
  Structured fields sparse in older rows; Pass B uses two-tier matcher.
- `docs/fillmore_v2/import_graph_audit.md` — confirms v1 and v2 share no
  mutable state. v2 sizing (Step 4) will live at `core/fillmore_v2_sizing.py`
  and must not import `runner_score`.
- `docs/fillmore_v2/ambiguity_register.md` — four open ambiguities logged
  (sparse legacy structured fields, stub prompt version in schema hash,
  best-effort path-time telemetry, placeholder volatility threshold).
- `docs/fillmore_v2/rollout.md` — staged rollout from PHASE9.10 plus
  explicit build triggers for the deferred adversarial verifier, deferred
  path-time blocking promotion, and deferred trailing stops.
- `docs/fillmore_v2/schema_hash_baseline.md` — pins the Step 1/v2.0.0
  snapshot schema hash (`03b4e69ff188c61a`) outside runtime computation.
- `scripts/fillmore_v2_pass_a_dry_run.py` — Pass A migration dry-run harness
  against the 241-trade Phase 7 corpus. Reproduces `+300.5p / +$5,684.56`
  from V1+V2 labels after v2 schema migration.

**Acceptance:** `pytest tests/test_fillmore_v2_telemetry.py` — 22/22 pass.

**Not yet wired:** v2 modules exist in parallel; no live loop is calling
them. Integration happens implicitly as Steps 2–7 land. The process-level
engine flag (config: `autonomous_fillmore.engine = "v1" | "v2"`) is
specified but not implemented — Step 9 wires it.

---

## Step 2 — Post-Decision Validator Layer (PHASE9.5)

**Added:**

- `api/fillmore_v2/llm_output_schema.py` — `LlmDecisionOutput` dataclass
  matching the PHASE9.4 JSON schema verbatim (decision/primary_thesis/
  caveats_detected/caveat_resolution/level_quality_claim/side_burden_proof/
  loss_asymmetry_argument/invalid_if/evidence_refs). Strict `parse()` raises
  `LlmOutputParseError` on any malformed structure; caller forces skip per
  spec. No markdown-fence stripping (fenced output is itself a violation).
- `api/fillmore_v2/validators.py` — five deterministic post-decision
  validators per PHASE9.5 with full audit records. Protected-edge bypass on
  `caveat_resolution` for buy + level_quality_score >= 70 + packet present.
  Orchestrator `run_all()` runs every validator (so audit shows every fire)
  but locks `final_decision` to `skip` on the first override.
- `api/fillmore_v2/legacy_rationale_parser.py` — best-effort adapter that
  reconstructs `LlmDecisionOutput` from corpus rows that predate the
  Phase 9 structured schema. Used only by shadow replay; v2 forward path
  consumes real structured output.
- `api/fillmore_v2/shadow_replay.py` — runs all validators against the
  forensic corpus; reports fire counts and rates per validator and per side.
- `tests/test_fillmore_v2_validators.py` — 39 tests covering JSON parser
  strictness, every validator pass/fail/edge case, protected-edge bypass,
  orchestrator audit semantics, the legacy adapter, and the shadow-replay
  corpus run.

**Audit docs:**

- `docs/fillmore_v2/step2_shadow_replay_baseline.md` — pooled fire counts
  from the available 241-row corpus subset. Includes calibration notes
  explaining why `loss_asymmetry` fires 100% on legacy (adapter artifact),
  why `sell_side_burden` fires 100% on legacy sells (adapter artifact),
  and why V5 fires 0× (thresholds untested against legacy distribution).
- `docs/fillmore_v2/ambiguity_register.md` — entry A5 added on V5 thresholds.

**Acceptance:** `pytest tests/test_fillmore_v2_validators.py` — 39/39 pass.
Combined with Step 1: 62/62 pass.

**Not yet wired:** Validators are testable in isolation and via shadow
replay only. The forward path (call run_all after the LLM responds, persist
overrides via `validator_overrides_json`) lands in Step 6 when the LLM call
is rebuilt to emit Phase 9 JSON.

---

## Step 3 — Pre-Decision Veto Layer (PHASE9.3)

**Schema bump: `v2.snap.1` → `v2.snap.2`.** Snapshot grew five fields needed
by V1/V2 (timeframe_alignment, macro_bias, catalyst_category, active_sessions,
session_overlap). New pinned hash `SNAPSHOT_SCHEMA_HASH_V2_SNAP_2 =
"66e8d0344cae9740"`. Historical `SNAPSHOT_SCHEMA_HASH_V2_0_0 =
"03b4e69ff188c61a"` retained as immutable anchor; new
`SNAPSHOT_SCHEMA_HASH_CURRENT` alias points at the active baseline. Future
schema bumps follow the same pattern: add a versioned constant, repoint
CURRENT, leave older constants in place. The baseline test asserts CURRENT.

**Added:**

- `api/fillmore_v2/pre_decision_vetoes.py` — V1 (sell caveat-template)
  and V2 (mixed-overlap entry with protected buy-CLR bypass). Both run
  BEFORE the LLM is dispatched. V3 retired (lesson encoded in prompt
  removal + exit layer). V4/V7 rejected. V5/V6 already in Step 2 as
  post-decision validators.
- Snapshot fields for veto inputs (see schema bump above).
- `api/fillmore_v2/persistence.py` now persists the Step 3 veto inputs
  (`timeframe_alignment`, `macro_bias`, `catalyst_category`,
  `active_sessions_json`, `session_overlap`) so pre-veto decisions can be
  replayed from logs.
- Legacy adapter helpers: `derive_sessions` (UTC → active sessions +
  overlap tag), `normalize_timeframe_alignment`, `derive_macro_bias`
  (from `side_bias_check`), `derive_catalyst_category` (from
  `named_catalyst`; filters short stop-words so "Reject at resistance"
  → structure_only).
- `shadow_replay.replay_corpus` now runs pre-vetoes first and short-circuits
  validator execution on fired rows (mirrors live flow). Summary carries
  `pre_veto_fire_counts`, `pre_veto_by_side`, `pre_veto_skipped_before_call`.
- `tests/test_fillmore_v2_pre_vetoes.py` — 28 tests covering V1 (no buy
  fires, no protected bypass), V2 (4 bypass paths including unknown lots),
  orchestrator short-circuit, all four legacy adapter helpers, and the
  shadow-replay coverage extension.

**Audit docs:**

- `docs/fillmore_v2/step3_pre_veto_baseline.md` — pooled fire counts.
  V1=89 (target 80), V2=43 (target 44), 100 skipped pre-call (target ~100),
  all V1 fires on sells. Post-decision `sell_side_burden` drops from 57
  fires (Step 2) to 0 (Step 3) — the bad sells now never reach the LLM.

**Acceptance:** `pytest tests/test_fillmore_v2_*.py` — 90/90 pass.

**Not yet wired:** Pre-vetoes run only via shadow replay. Forward path
(call `run_pre_vetoes` after gate eligibility, before LLM dispatch, persist
fires via `pre_veto_fired_json`) lands in Step 6 + the v1/v2 engine flag.

---

## Step 4 — Deterministic Sizing Layer (PHASE9.6)

**Added:**

- `core/fillmore_v2_sizing.py` — `compute_autonomous_lots(SizingContext) →
  SizingResult`. Implements PHASE9.6 verbatim: 0.0025 default risk, 0.0010
  paper/0.1x, 0.0050 ramp ceiling gated on forward 100-trade PF ≥ 1.10 AND
  net_pips_100 > 0; raw lots clamped to [1, 4]; three 50% throttles
  (rolling P&L, side exposure ≥ 4 lots, elevated volatility); protected
  buy-CLR floor at 2.0 lots when risk-after-fill ≤ 0.5% equity; final
  re-clamp to [1, 4]. Pure function of frozen `SizingContext`. Audit-grade
  `SizingResult` records every branch taken.
- Helpers: `select_risk_pct`, `round_to_step` (half-up, bit-stable),
  `clamp`, `cap_historical_lots`, `rescale_pnl_for_cap`.
- `tests/test_fillmore_v2_sizing.py` — 32 tests:
    - Helper purity (round, clamp, risk-pct ramp semantics including
      "ramp caps but never raises paper-stage floor").
    - Every branch of `compute_autonomous_lots` (basic paths, paper stage,
      sub-1-lot clamp, degenerate inputs, each throttle in isolation, side
      exposure side-correctness, triple throttle re-clamps to MIN, every
      protected-floor path).
    - Hard-cap invariant under huge equity / tight stops / ramp.
    - Referential transparency: 1000 calls on a frozen context, identical
      output every time.
    - Import-graph isolation: no `runner_score` imports either direction
      (checks actual import statements, not bare substrings); no `api/`
      imports from `core/fillmore_v2_sizing`.
    - Cap-to-4 corpus replay: V1+V2 survivors (131 of 241) capped to 4 lots
      recovers **$334.29**, matching PHASE9.6's "Incremental survivor
      sizing improvement after filter: $334.30" within rounding. Cross-check
      assertion against the published $334.30 figure.
    - Audit invariant: zero corpus survivors have lots in (0, 1), so the
      lower clamp doesn't distort the cap-to-4 measurement.

**Acceptance:** `pytest tests/test_fillmore_v2_sizing.py` — 32/32 pass.
Combined v2 suite: 122/122 pass. Pass A dry-run still reproduces
+300.5p / +$5,684.56 against the migrated v2 schema.

**Note on full PHASE9.6 numbers:** PHASE9.6's headline figures
($6,420.90 total recovery, $832.34 final-USD-after-cap, 141 survivors at
+16.3p) require the FULL Phase 9 admission stack (V1+V2+V5+V6) applied to
corpus rows. V5+V6 don't have per-row labels in the Phase 7 dataset, so
they would need to be derived by running validators against rationales.
That replay is Step 8 Pass C. Step 4 pins the sizing-only delta ($334.29
on V1+V2 survivors) which the blueprint cross-references at $334.30.

**Not yet wired:** Sizing function is called only by tests. Forward path
(snapshot → context → `compute_autonomous_lots` → persist via
`sizing_inputs_json` and `deterministic_lots`) lands in Step 6 alongside
the LLM call refactor.

---

## Step 5 — Gate Layer Redesign (PHASE9.2)

**Added:**

- `api/fillmore_v2/gates.py` — four primary gates with PHASE9.2 verdicts:
    - `buy_clr_eligible` — REDESIGN, buy half of CLR. Score ≥ 70, packet
      side `buy_support`, no broken-then-reclaimed flag, profit-path
      blocker ≥ planned risk (or unknown), pre-decision V1/V2 don't fire
      (or protected bypass active). Lower threshold than sell-CLR (70 vs
      85) to preserve the three Phase 8 protected cells.
    - `sell_clr_eligible` — REDESIGN, sell half. Score ≥ 85 (or ≥ 90 in
      mixed overlap), packet side `sell_resistance`, timeframe alignment
      not `aligned_buy`, macro not bullish, mixed-overlap exception
      requires material catalyst, no structure-only catalyst. Defense in
      depth with V1.
    - `momentum_continuation_eligible` — KEEP-CONDITIONAL. Profit-path
      blocker ≥ 1R sl_pips, side-normalized packet matches proposed side,
      spread within normal regime.
    - `mean_reversion_eligible` — KEEP-SMALL-N. Carries
      `sizing_cap_lots=1.0` until 30 closes prove non-negative. Requires
      side-normalized packet matches proposed side and exhaustion marker on
      the level (exhaustion / pdh / pdl / wh / wl keywords in
      `structural_origin`).
    - Non-primary gates (`post_spike_retracement`, `failed_breakout`,
      `trend_expansion`) NOT registered. Helper `is_non_primary_gate_killed`
      makes the kill explicit and grep-able.
- `evaluate_all_gates(snapshot, pre_veto_summary, preferred_order)` —
  runs every primary gate, records every candidate (PHASE9.8 item 11
  multi-gate auditability), selects the first eligible by `preferred_order`.
- `tests/test_fillmore_v2_gates.py` — 30 tests:
    - Buy-CLR: protected packet passes; wrong proposed-side, wrong
      packet-side, score < 70, broken-reclaimed, blocker-inside-risk,
      pre-veto block. Blocker absent → don't fail-closed.
    - Sell-CLR: clean packet passes; sub-85 reject; mixed-overlap
      requires ≥ 90 AND material catalyst; H1+M5 against short reject;
      bullish macro reject; structure-only catalyst reject.
    - Momentum: 1R room passes; sub-1R, missing blocker, wrong-side packet,
      wide spread reject.
    - Mean reversion: exhaustion marker required; carries 1-lot cap;
      wrong-side packet and elevated vol reject; pdh/pdl/wh/wl markers count
      as exhaustion.
    - Non-primary kill: registry has exactly 4 entries; killed set is
      not present.
    - Orchestrator: every candidate logged; preferred_order chooses; no
      selection when all ineligible; pre-veto block propagates; protected
      buy-CLR with V2 bypass passes through.

**Acceptance:** `pytest tests/test_fillmore_v2_gates.py` — 30/30 pass.
Combined v2 suite: 152/152 pass. Pass A dry-run intact.

**Not yet wired:** Gates are called only by tests. Forward path is assembled
in Step 6. Required order: snapshot → preliminary `evaluate_all_gates`
without pre-veto summary to log raw candidates → `run_pre_vetoes` →
final `evaluate_all_gates(..., pre_veto_summary=...)` so fired vetoes or
protected bypasses are reflected in selected-gate eligibility → if eligible,
LLM call → post-decision validators → sizing → order placement.

---

## Step 6 — LLM Decision Layer (PHASE9.4)

**Schema bump: prompt content + version.** `PROMPT_VERSION = "v2.prompt.1"`
(was the `v2.prompt.0-stub` placeholder). Snapshot field set unchanged at
`v2.snap.2`. Combined hash recomputed:
`SNAPSHOT_SCHEMA_HASH_V2_PROMPT_1 = "c1f6863ebcc2c8ca"`.
`SNAPSHOT_SCHEMA_HASH_CURRENT` repointed to it. Older constants
(`V2_0_0`, `V2_SNAP_2`) preserved as historical anchors.

**Risk-treatment for the highest-risk step:** v1 (`api/autonomous_fillmore.py`)
was NOT modified. The orchestrator is callable from tests and the manual
smoke-test script only — no production loop registers it. The engine flag
(config: `autonomous_fillmore.engine = "v1" | "v2"`) lands in Step 9. v1
state file (`runtime_state.json`) is verified untouched by v2 in tests.

**Added:**

- `api/fillmore_v2/system_prompt.py` — `SYSTEM_PROMPT_V1` text per PHASE9
  SYSTEM PROMPT V1 (caveat law, sell-side burden, level claim law,
  loss-asymmetry law, JSON-only output). All five removed-content classes
  enforced (no runner-preservation language, no red-pattern lists without
  terminal caveats, no structure-only catalyst acceptance, no sizing
  instructions, no exit-extension authority). `render_user_context` uses a
  whitelist of visible snapshot fields and cross-checks against
  `SIZING_FORBIDDEN_FIELDS` so a future schema change can't accidentally
  leak account_equity / open_lots / risk_after_fill into the LLM call.
- `api/fillmore_v2/llm_client.py` — `LlmClient` Protocol, `OpenAILlmClient`
  using `response_format={"type": "json_object"}` and `temperature=0.0`,
  plus `FakeLlmClient` for deterministic tests. Transport errors return
  `LlmCallResult.error` instead of raising; orchestrator forces skip.
- `api/fillmore_v2/orchestrator.py` — `run_decision(snapshot, llm_client,
  profile_dir, db_path, profile, model, stage)` sequences halt check →
  blocking-field check → gate eligibility → pre-vetoes → gate
  re-evaluation with pre-veto context (matching the gates-module docstring
  contract) → deterministic sizing → prompt render → LLM call → strict
  parse → post-decision validators → persist v2 row. Pure orchestration;
  every layer is delegated to its owning module. Returns
  `OrchestrationResult` with full audit fields. Never raises on LLM or
  parse errors.
- `tests/test_fillmore_v2_orchestrator.py` — 20 tests including the
  PHASE9.4 acceptance: 10 LLM calls in dev, every schema-valid output
  parses, validators fire on intentionally weak rationales, sizing
  fields stay out of the call path (verified two ways: whitelist check
  on `render_user_context` and substring grep on the rendered prompt of
  every captured `FakeLlmClient.calls[i].user`). Plus halt-state
  strikes-then-halt path, no-eligible-gate short-circuit (across all
  four primary gates), prompt rendering without placeholder prefill,
  CLR-specific blocking after gate selection, pre-veto skip-before-call,
  LLM transport error,
  malformed-JSON parse failure, markdown-fence parse failure, validator
  override → final skip, happy-path place persists full audit, and
  isolation tests (orchestrator does not import v1, v1 state file
  untouched after a v2 decision). Sell-side SL/TP persistence is covered.
- `scripts/fillmore_v2_smoke_test.py` — manual real-LLM smoke test. NOT a
  pytest. Runs one decision through the full pipeline against the live
  OpenAI API. `--no-call` prints the rendered prompt for review without
  spending tokens. No DB writes.

**Acceptance:** `pytest tests/test_fillmore_v2_orchestrator.py` — 20/20
pass. Combined v2 suite: 174/174 pass. Pass A dry-run still reproduces
+300.5p / +$5,684.56.

**Sizing-leak defense in depth:**

  1. `SIZING_FORBIDDEN_FIELDS` set lives at module scope; `render_user_context`
     asserts no overlap with `_VISIBLE_SNAPSHOT_FIELDS` whitelist at render time.
  2. `test_sizing_fields_absent_from_user_context` asserts the set membership.
  3. `test_sizing_fields_absent_from_rendered_prompt` substring-checks the
     rendered text.
  4. `test_acceptance_ten_llm_calls_all_parse_and_route` substring-checks
     every captured user message across 10 distinct LLM call patterns.

**Adversarial verifier still deferred** per Step 1 rollout doc — build only
on an explicit trigger (Stage 2 sell WR < 45% after 30 sells, validator
fire-rate drift, protected-cell regression, or any Phase 8 primary failure
reappearing).

**Not yet wired into production:** Step 9 implements the
`autonomous_fillmore.engine = "v1" | "v2"` flag and its check inside the
live tick loop. Until then, the orchestrator is exercised only by tests
and the smoke-test script. v1 remains the live engine.

---

## Step 7 — Exit Layer (PHASE9.7)

**Added:**

- `api/fillmore_v2/exit_layer.py` — deterministic exit management:
    - `decide_exit_action(state, current_price, current_time_utc)` —
      pure function returning one of `NO_CHANGE`, `MOVE_SL_TO_BE_PLUS_SPREAD`,
      `TIME_STOP_CLOSE`. Profit lock at exactly 1R MFE; time stop at exactly
      30 minutes; both have authorized `rule_id` strings drawn from
      `AUTHORIZED_STOP_CHANGE_RULE_IDS`.
    - `decide_partial_close_at_1r(state, close_fraction)` — alternative
      profit-lock path for brokers preferring scale-out over BE-move.
    - Defensive: profit-lock REFUSES to widen when the BE+spread target
      is wider than the current SL (post-tightening invariant).
    - `detect_unauthorized_stop_widenings(events)` — halt detector for
      any SL change with `rule_id` outside the authorized set.
    - `exit_reversal_rate(closed_trades)` + `should_halt_for_exit_reversals(...)` —
      Phase 6's exit-reversal proxy (mfe ≥ 4p AND pips < 0). Halt fires
      strictly above 15% over a 50-trade audit window. Fires-at-15% test
      asserts the > (not ≥) semantics.
    - `trailing_stop_enabled(...)` — locked OFF until ≥ 50 closed trades
      AND path-time coverage ≥ 90% AND exit-reversal rate < 10%.
- `tests/test_fillmore_v2_exit_layer.py` — 28 tests covering:
    - Profit lock at exactly 1R, sell-side BE-spread sign, refuse-to-widen
      defense, already-locked no-op
    - Time stop at exactly 30m (and not at 29m59s), profit-lock priority
      when both eligible, no time-stop after profit lock has fired, and
      naive ISO timestamps normalized to UTC
    - Partial-close path including invalid-fraction rejection
    - Unauthorized-widening detector (positive + negative cases) and
      trust-boundary documentation test for widening labeled with an
      authorized rule_id
    - Public-callable allowlist test: every public callable defined in
      `exit_layer` (excluding dataclasses, enums, and re-exports) is on a
      hand-reviewed write-path allowlist. Catches accidental future
      additions of LLM-driven SL widening
    - Exit-reversal rate edge cases including the strict > 15% threshold
    - Trailing-stop disabled until all three gates clear
    - Replay determinism: same path → same `ExitDecision` sequence
    - `ExitEvent.to_audit_record` round-trip

**Acceptance:** `pytest tests/test_fillmore_v2_exit_layer.py` — 28/28 pass.
Combined v2 suite: 202/202 pass. Pass A dry-run still reproduces +300.5p / +$5,684.56.

**LLM-authority enforcement:** the allowlist test makes it impossible to
add a function that takes an LLM-supplied SL value without a CI failure.
The `AUTHORIZED_STOP_CHANGE_RULE_IDS` set is the trust boundary;
`detect_unauthorized_stop_widenings` halts on any widening labeled with a
rule_id outside it.

**Replay treatment per PHASE9.7:** no net pip recovery is claimed from
exit redesign in the Phase 9 pass/fail replay (path-time data is
unavailable in the corpus). The exit layer is conservative + telemetry-
first: prevents the proven runner-clause harm from returning, doesn't
claim untested pips. Trailing-stop enable is gated on telemetry maturity.

**Not yet wired:** `decide_exit_action` is callable from tests only. The
live trade-management loop still belongs to v1 (`run_loop.py` +
v1's exit strategies). Step 9 wires the exit layer into the v2 forward
path together with the engine flag.

---

## Step 8 — Retroactive Replay Verification (PHASE9.9 + PHASE9.6)

**Three-pass gate executed.** Full results: `docs/fillmore_v2/step8_replay_results.md`.

**Added:**

- `scripts/fillmore_v2_pass_b_validator_agreement.py` — V1 / V2 agreement
  vs Phase 7 ground-truth labels.
- `scripts/fillmore_v2_pass_c_full_stack_replay.py` — full v2 stack
  (V1+V2 pre-veto + V5+V6 post-validator + cap-to-4 sizing) against the
  241-trade Phase 7 corpus. Reports recovery, false-positive cost,
  block-source breakdown, and protected-cell preservation.
- `tests/test_fillmore_v2_pass_abc.py` — 8 tests pinning the current
  state. Pass A reproduces the floor; Pass B asserts V1 ≥ 95% and
  reconciles every V2 disagreement to the protected-edge bypass; Pass C
  asserts recovery floor + protected cells + documented false-positive
  overshoot baseline.

**Pass A status: PASS.** Reproduces +300.5p / +$5,684.56 exactly.

**Pass B status:**
  - V1 agreement: **95.02%** (≥ 95% target) — PASS, with 0 false negatives.
  - V2 agreement: 89.21% (target 90%) — 0.8pp under, fully reconciled:
    every disagreement is a buy-CLR mixed-overlap row sparing a Phase 8
    protected-edge candidate via the V2 bypass. By design.

**Pass C status: SPLIT.**
  - Recovery floor: **PASS** (+282.5p / +$6,370.77 vs floor +278.4p / +$5,684.56).
  - Full USD with cap-to-4: **BEATS TARGET** (+$6,599.37 vs target $6,420.90).
  - Protected cells: **PASS** — all three cells, exact pip/USD parity to
    PHASE8 §8.4 published values, zero rows blocked.
  - Blocked-trades ceiling: **FAIL** (131 vs ≤110 floor).
  - Blocked-winners ceiling: **FAIL** (64 vs ≤52 floor).

**Gap analysis (PHASE9.9 violation traced):**

The false-positive overshoot is entirely explained by the legacy
rationale parser, which exists only for retroactive replay. The
production orchestrator (Step 6) does NOT call it — verified by
`test_pass_c_acknowledged_diagnostic_gap_traces_to_legacy_adapter`.

  - V1 over-fires by 12 because the heuristic catches a superset of the
    Phase 4 NLP clusters (Pass B confirmed 12 false positives, 0 false
    negatives).
  - V2 under-fires by 40 because the legacy adapter assigns a uniform
    `score=72` to all buy-CLR rows, triggering the bypass for all of
    them (production sees real per-row scores).
  - V5 fires 0 (already documented in Ambiguity A5).
  - V6 within tolerance (-4 vs target).

**Per the build directive ("STOP and report"):** the user is asked to
choose between three options documented in
`docs/fillmore_v2/step8_replay_results.md`:

  1. Accept Pass C as-is — gap is corpus-only, production code unchanged.
  2. Tune the legacy adapter to reduce V1 over-firing — improves Pass C
     numbers at the cost of fidelity to the heuristic intent.
  3. **Recommended:** Defer to Step 9 wiring + Stage 1 paper validation.
     The forensic corpus predates Phase 9 structured fields; reverse-
     engineering them via heuristics is a diagnostic, not the gate. Stage 1
     paper trades produce real Phase 9 outputs and become the binding gate.

**Acceptance:** `pytest tests/test_fillmore_v2_pass_abc.py` — 8/8 pass.
Combined v2 suite: 210/210 pass.

**Step 8 status: VERIFICATION COMPLETE; decision requested before Step 9
moves to live-engine wiring.**

---

## Step 9 — Staged Rollout Wiring (PHASE9.10)

**Pre-step decision:** Step 8 Option 3 selected by user — Stage 1 paper
validation against live v2-emitted JSON is the binding gate. Step 9
wires the rollout machinery and the engine-flag dispatch but does NOT
flip any switches; default engine remains "v1".

**Added:**

- `api/fillmore_v2/engine_flag.py` — `read_engine_flag(state_path) → 'v1'|'v2'`
  with default `"v1"` on missing file, corrupt JSON, unknown value, or
  any read error. `set_engine_flag` round-trips and preserves other state
  sections. Cardinal safety invariant tested: default is v1.
- `api/fillmore_v2/tripwires.py` — five always-on tripwires from
  rollout.md as PURE functions: T1 sell-side WR, T2 sell-CLR kill,
  T3 caveat-validator fire-rate band, T4 sizing hard ceiling, T5
  skip-outcome coverage. Each returns green/amber/red with detail.
  `evaluate_all` aggregates into a `TripwireSnapshot` with `any_red` and
  `red_ids()` for operator alerting.
- `api/fillmore_v2/stage_progression.py` — pure verdict on the current
  stage given observations: hold / advance / kill. Per-stage criteria
  matrix encodes PHASE9.10's advance/kill thresholds verbatim. Sell-side
  tripwires (T1, T2) are kill-class; non-sell red tripwires (T4) block
  advancement but don't kill. Stage "full" never advances and uses the
  Phase-8-failure-reappeared flag for kills.
- `api/fillmore_v2/v1_bridge.py` — `dispatch_v2_tick` and
  `build_snapshot_from_v1_inputs`. **Deliberately minimal**: many Snapshot
  fields land as None on first wiring; v2's blocking-field check halts
  within 3 ticks. That's the intended safety behavior — full telemetry
  wiring is follow-up work tracked in
  `docs/fillmore_v2/wire_v2_to_live_loop.md` (next step after Stage 1
  shows the modules behave under real ticks).
- **Modified `api/autonomous_fillmore.py`** (the only v1 source touched in
  the entire 9-step build): inserted a try/except-wrapped engine-flag
  check in `tick_autonomous_fillmore` immediately after the `enabled`
  fast-path. When flag is "v1" (default) behavior is byte-identical to
  before. When flag is "v2" the tick dispatches to `v1_bridge`. Any
  exception in the v2 path is swallowed so v2 wiring cannot crash v1.
- `tests/test_fillmore_v2_step9.py` — 32 tests:
    - Engine flag: default v1, defensive reads on missing/corrupt/unknown,
      round-trip, preserves existing state, rejects invalid values,
      cardinal-safety constants
    - Tripwires: green/amber/red boundaries for each, aggregator behavior
    - Stage progression: hold / advance / kill paths, stage-specific kill
      thresholds, fatal vs non-fatal tripwires, full-stage never advances
    - v1 bridge: snapshot construction from minimal inputs, halt within
      3 ticks (cardinal safety check), v1 source contract (engine-flag
      check exists, gates on "v2", wrapped in try/except)

**Acceptance:** `pytest tests/test_fillmore_v2_step9.py` — 32/32 pass.
Combined v2 suite: 242/242 pass. Pass A dry-run intact.

**Live-system safety state at end of Step 9:**

  - Default engine: v1 (unchanged behavior)
  - To enable v2 in shadow: `set_engine_flag(state_path, "v2")`. v2 will
    halt within 3 ticks because blocking telemetry isn't wired in v1's
    inputs. That halt is correct and safe — operators see "v2 halted:
    missing risk_after_fill_usd, pip_value_per_lot, …" and decide.
  - Full v2-handles-live-tick wiring (populating Snapshot blocking fields
    from v1's profile/store) is the next milestone after this rebuild
    completes. It is intentionally outside Step 9's scope per the user's
    Step 1 risk-treatment directives.

**Not yet wired (deferred follow-ups):**

  - Account-equity, exposure, pip-value, risk-after-fill capture from
    OANDA in `v1_bridge.build_snapshot_from_v1_inputs`
  - Side-normalized level packet construction from v1's gate inputs
  - Rolling-20 P&L computation against ai_suggestions
  - Stage progression cron / operator dashboard
  - Adversarial verifier (per build triggers in rollout.md)

These are tracked in `docs/fillmore_v2/wire_v2_to_live_loop.md` (placeholder
backlog doc). Stage 1 paper validation against the v2 modules in
isolation is the next operator action; v1 remains live until Stage 1
confirms behavior.

---

## Post-Step 9 — Stage 1 Live-Loop Readiness Wiring

**Goal:** make v2 safe to flip into Stage 1 paper validation from the live
loop without immediately halting on telemetry that can be derived from v1's
existing adapter/bar inputs. Default engine remains `"v1"`.

**Added/changed:**

- `api/fillmore_v2/v1_bridge.py`
  - Adds an explicit Stage 1 paper-mode guard: non-paper stages return a
    `halt` result unless a future caller explicitly passes
    `allow_non_paper=True`.
  - Pulls `account_equity` from `adapter.get_account_info()`.
  - Pulls open exposure and unrealized P&L from
    `adapter.get_open_positions(profile.symbol)`.
  - Computes `pip_value_per_lot` from the live USDJPY mid.
  - Computes rolling-20 P&L from the profile's `ai_suggestions` DB.
  - Derives Stage 1 `proposed_side`, `sl_pips`, `tp_pips`,
    `timeframe_alignment`, `active_sessions`, `session_overlap`,
    `volatility_regime`, and conservative `catalyst_category`.
  - Builds a conservative bar-derived side-normalized `LevelPacket` with
    `level_quality_score=65`. This is intentionally below CLR admission
    thresholds, but gives momentum-continuation a path-room packet for
    paper validation.
  - Initializes the additive v2 persistence schema idempotently when a DB
    path is available.
- `api/fillmore_v2/orchestrator.py`
  - Treats `risk_after_fill_usd` as post-sizing blocking telemetry:
    excluded from the pre-sizing strike check, computed immediately after
    deterministic sizing, and failed closed if it cannot be computed.
- `api/autonomous_fillmore.py`
  - Passes v1 adapter, store, `data_by_tf`, and optional open-position
    snapshot into the v2 dispatch bridge when the explicit engine flag is
    set to `"v2"`.
- `docs/fillmore_v2/wire_v2_to_live_loop.md`
  - Updated B1-B5 status for Stage 1 readiness.
- `scripts/fillmore_v2_first_tick_check.py`
  - Read-only SQL checker for latest v2 rows: missing required fields, halt
    reason, selected gate, final decision, LLM parse status, gate candidates,
    sizing audit, validator overrides, and pre-veto fires.
- `docs/fillmore_v2/live_testing_readiness_20260502.md`
  - Documents paper-only boundary, rollback command, v2 state isolation, known
    unrelated full-suite debt, first-hour checks, and pre-0.1x requirements.

**Acceptance:**

- `pytest tests/test_fillmore_v2_*.py -q` — 248/248 pass.
- `pytest tests/test_autonomous_fillmore.py -q -k "engine or suggest or runner or veto"` —
  26/26 selected pass.
- `python3 -m compileall api/fillmore_v2 api/autonomous_fillmore.py` passes.
- `python3 scripts/fillmore_v2_smoke_test.py --no-call` renders the v2
  system prompt and context successfully.

**Remaining before Stage 2 / 0.1x sizing:**

- Replace the Stage 1 conservative level packet with a real side-normalized
  level-quality builder.
- Upgrade macro/catalyst classification beyond `"neutral"` /
  `"structure_only"`.
- Wire stage-progression/dashboard ops (B6) if desired.
- Build adversarial verifier only if rollout tripwires trigger (B7).

---

## Post-Step 9 Follow-up — OANDA Practice Placement

**Reason:** operator confirmed audit-row-only validation is not sufficient;
Stage 1 must exercise real OANDA practice fills, broker rejects, local trade
rows, and lifecycle accounting.

**Added/changed:**

- `api/autonomous_fillmore.py`
  - After `dispatch_v2_tick(...)` returns, approved v2
    `final_decision="place"` results are projected into the existing broker
    placement shape and sent through `_place_from_suggestion(..., "market",
    ...)`.
  - Added a hard v2 practice-placement guard: autonomous mode must be
    `paper`, `broker_type` must be `oanda`, and `oanda_environment` must be
    `practice`. Shadow/live/MT5/OANDA-live all fail closed before broker
    order-send.
  - v2 practice orders carry `engine_version="v2"`, deterministic lots,
    risk-after-fill, loss-asymmetry argument, level-quality claim, evidence
    refs, and the v2 audit record into local trade config where available.
- `api/main.py` / frontend autonomous panel
  - v2 status and UI copy now state that v2 Stage 1 can place OANDA practice
    orders after all v2 checks pass. It remains blocked from real-capital and
    0.1x/full sizing.
- `docs/fillmore_v2/live_testing_readiness_20260502.md`
  - Updated Stage 1 boundary: OANDA practice-account validation, not
    audit-row-only validation.

**Acceptance added:**

- `tests/test_fillmore_v2_step9.py` covers the practice-only placement guard
  and the v2-result-to-broker-suggestion projection, including deterministic
  lots and SL/TP price conversion.
