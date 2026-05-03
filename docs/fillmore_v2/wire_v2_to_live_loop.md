# Wire v2 to Live Loop — Backlog (post-Step 9)

The 9-step rebuild is complete. v2 modules are tested in isolation and the
engine-flag dispatch is wired. **Stage 1 bridge wiring is now in place**:
`api/fillmore_v2/v1_bridge.py` can populate account/exposure, pip value,
rolling P&L, session labels, conservative bar-derived side/SL/TP, and a
low-score side-normalized level packet from v1's live tick inputs.

Flipping `engine = "v2"` still fails closed if the adapter/account/bar inputs
are absent. With those inputs present, the blocking-field strike counter should
stay clean across ticks; approved v2 `place` decisions can send OANDA practice
orders only when autonomous mode is `paper`, broker type is OANDA, and
`oanda_environment='practice'`. v1 remains the default engine until the
operator opts in.

This file lists the remaining wiring work in priority order.

## Why Step 9 stopped here

Per Step 1 design directives (parallel build, rollback-friendly) and the
user's Step 8 Option-3 decision (Stage 1 OANDA-practice validation is the binding
gate), the highest-risk wiring — populating Snapshot blocking fields from
v1's profile/store/OANDA adapter — was deliberately deferred. Doing it
inside Step 9 would have widened the test surface by an order of
magnitude and made rollback non-trivial. The intent is to land this
wiring as a separate, narrowly-scoped follow-up after Stage 1 paper
validation confirms the v2 modules behave under real ticks and real practice
fills.

## Backlog

### B1. Populate `account_equity`, exposure, unrealized P&L — Stage 1 done

Source: OANDA account state via the existing v1 adapter in
`api/main.py`. Add accessors in `v1_bridge.build_snapshot_from_v1_inputs`
that pull these from v1's `profile.broker_account_state` (or equivalent
— exact symbol TBD; see api/main.py:_account_summary). Map to:

  - `Snapshot.account_equity`
  - `Snapshot.open_lots_buy`, `Snapshot.open_lots_sell`
    (use `telemetry.open_lots_by_side` against position rows)
  - `Snapshot.unrealized_pnl_buy`, `Snapshot.unrealized_pnl_sell`

Implemented in `v1_bridge.build_snapshot_from_v1_inputs` via
`adapter.get_account_info()` and `adapter.get_open_positions(...)`. Missing
adapter/account data still fails closed.

Acceptance: covered by `tests/test_fillmore_v2_step9.py`.

### B2. Populate `pip_value_per_lot` and `risk_after_fill_usd` — Stage 1 done

`pip_value_per_lot` from `telemetry.pip_value_per_lot(usdjpy_price)` —
trivial. `risk_after_fill_usd` from `telemetry.risk_after_fill_usd(...)`
once `proposed_lots` and `sl_pips` are known (i.e., after sizing).

Note: `risk_after_fill_usd` is computed POST-LLM in the orchestrator's
sequence, so the bridge can leave it None at snapshot-build time and
let the orchestrator fill it after `compute_autonomous_lots`.

Implemented. `pip_value_per_lot` is computed in the bridge. The orchestrator
now treats `risk_after_fill_usd` as post-sizing blocking telemetry: it is
excluded from the pre-sizing strike check, computed immediately after
deterministic sizing, and fails closed if it cannot be computed.

Acceptance: 10-tick bridge test plus orchestrator post-sizing risk test.

### B3. Populate `rolling_20_trade_pnl` + `rolling_20_lot_weighted_pnl` — Stage 1 done

Use `telemetry.fetch_closed_trades_for_rolling(db_path, profile)` then
`telemetry.rolling_pnl(...)`. Cache for ~60s if hot-path concerns arise.

Implemented without cache. Revisit only if tick-path profiling shows DB reads
matter.

### B4. Side-normalized level packet from v1's gate signals — Stage 1 conservative

Hardest piece. v1's gate produces a `setup_family` and `trigger_reason`
but not a side-normalized level packet with score/age/blocker. Two paths:

  (a) Build a separate side-normalizer module that reads v1's snapshot
      (S/R levels, swing data, cluster strengths) and produces
      `LevelPacket` per proposed side. Touches: `core/h1_level_detector.py`
      and the level-detection helpers in v1.
  (b) Defer level-aware gating in v2 — let buy_clr/sell_clr fail "level
      packet missing" and rely on momentum_continuation as the v2-eligible
      gate during Stage 1. Stage 2+ requires the real packet.

Stage 1 implementation is a conservative bridge-only packet from recent bar
highs/lows with `level_quality_score=65`. This intentionally does not qualify
buy/sell CLR (`buy>=70`, `sell>=85`), but it gives momentum-continuation the
profit-path blocker it needs for paper testing. A real side-normalizer remains
required before Stage 2 / 0.1x sizing.

### B5. Pre-decision veto inputs (timeframe_alignment, macro_bias, etc.) — Stage 1 partial

v1 has these signals scattered across `core/execution_engine.py` and
`core/dashboard_reporters.py`. The bridge needs to derive:

  - `timeframe_alignment`: from M5/M15/H1 EMA stack agreement
  - `macro_bias`: from D1 trend + recent macro events
  - `catalyst_category`: from v1's `phase4_catalyst_score` and named-catalyst signals
  - `active_sessions`, `session_overlap`: already derivable from clock; reuse `legacy_rationale_parser.derive_sessions`

Stage 1 implementation derives `timeframe_alignment` from simple M1/M5/H1
close slopes, sets `macro_bias="neutral"` conservatively, marks
`catalyst_category="structure_only"`, and reuses `derive_sessions`.
Macro/catalyst quality should be upgraded before Stage 2.

### B6. Stage progression cron + operator dashboard

`stage_progression.evaluate_stage` is callable but not scheduled. Wire as:

  - Cron: nightly call producing `StageVerdict` from accumulated
    counters in `ai_suggestions` + `ai_thesis_checks`.
  - Dashboard: surface in api/main.py reasoning feed under a
    `v2_stage_status` key.

### B7. Adversarial verifier (per Step 1 rollout.md triggers)

Build only when one of:
  - Stage 2 sell-side WR < 45% after 30 sells
  - Caveat-validator fire rate outside 50%-150% band
  - Any protected cell turns negative in forward sample
  - Any Phase 8 primary failure reappears

Spec already in `system_prompt.py` (use a second model with veto-only
authority on the proposed JSON).

## Done-criteria for "v2 wiring complete"

  - With engine="v2" and B1–B5 wired: 50 ticks produce a full
    blocking-telemetry-populated Snapshot every time.
  - First v2 paper trade (decision="place") writes a complete row to
    `ai_suggestions` with `engine_version='v2'` and all v2 columns
    populated (verified by query).
  - Stage 1 advance criteria met → operator runs `set_engine_flag` to
    persist the stage transition; sizing scaler picks up new stage.
