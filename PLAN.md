# Multi-agent research & promotion plan

This document is the working plan for coordinated work across **Agent 1 (defensive ownership)**, **Agent 2 (offensive shadow / disagreement ledger)**, and **Agent 3 (portability / momentum harness)**. It is rewritten from the **current state** (March 2026).

---

## Where we are now (completed work)

### Agent 1 — Defensive ownership v1.5

**Live / engine observability**

- **`core/phase3_integrated_engine.py`**: `compute_phase3_ownership_audit_for_data(...)` — regime, ER / ΔER buckets, `ownership_cell`, and which **defensive gates** would fire (V44 regime block, London cluster block, global standdown).
- **`run_loop.py`**: attaches **`phase3_ownership_audit`** to Phase 3 execution results; extends **`phase3_minute_diagnostics.log`** with **`ownership_cell`**, **`regime`**, **`defensive_flags`** for post-trade review.

**Candidate mining**

- **`scripts/diagnostic_defensive_ownership_v15.py`** → **`research_out/defensive_ownership_v15_candidates.json`** (negative-both-dataset cells for `v44_ny` / `london_v2`, etc.).
- **`scripts/diagnostic_phase3_ownership_pain_log.py`**: aggregates live/minute diagnostics by cell, reason, and flags (use before promoting any veto).

**Single-pocket backtests vs promoted Variant K (coupled stack)**

| Pocket | Rule | Output JSON | Result (high level) |
|--------|------|-------------|------------------------|
| **#1** | Block **`v44_ny`** only in **`ambiguous/er_low/der_neg`** | `research_out/defensive_v15_pocket1_v44_ambiguous_low_derneg.json` | **500k:** blocked 5 trades; Δ vs K **+1629 USD**, **+0.12 PF**, **+45.73 DD** (DD worse). **1000k:** blocked 20; **+613 USD**, **+0.07 PF**, **−163 DD** (DD better). Candidate for promotion **only** after live pain-log alignment and explicit DD tradeoff sign-off. |
| **#2** | Block **`v44_ny`** in **`ambiguous/er_high/der_pos`** | `research_out/defensive_v15_pocket2_v44_ambiguous_high_er_pos.json` | **Rejected for promotion:** cell is **`UNSTABLE_CROSS_DATASET`**; coupled veto **hurts net USD** on both samples (blocked sets are net contributors on the K path). **500k:** Δ **−1319 USD**; **1000k:** Δ **−4156 USD** (some PF/DD improvement at large net cost). |

**Supporting diagnostics (non-exhaustive)**

- Various **`scripts/diagnostic_*`** scripts for ownership stability, regime splits, session boundaries, etc., feeding **`research_out/*.json`** as needed for gates.

---

### Agent 2 — Offensive shadow & promotion machinery

- **`scripts/offensive_shadow_ledger.py`**: tracks shadow / disagreement vs baseline for offensive candidates.
- **`scripts/portability_program_gates.py`**: gate checks aligned with the portability program.
- **`scripts/promotion_gate_board.py`**: weekly-style **promote / defer / close** matrix inputs (pair with human review).

*Ongoing role:* run recurring Phase A shadow diagnostics, split portable vs non-portable opportunities, and keep offensive tests **narrow** until gates pass.

---

### Agent 3 — Momentum portability harness (3A scaffold)

- **`scripts/backtest_momentum_portability_harness.py`**
  - Runs **native NY-only V44** on the current path vs **latent / all-day V44** with the same entry logic and a **relaxed session window**.
  - Classifies trades by **analysis session** and **ownership cell**.
  - Restricts portability candidates to **stable V44 momentum-owned** cells:  
    `momentum/er_high/der_neg`, `momentum/er_low/der_pos`, `momentum/er_mid/der_neg`.
  - **Modes:** full harness (includes coupled shadow pilot vs **Variant K**) vs **`--raw-only`** (skips expensive K baseline/coupling; raw portability evidence only).
  - **Default output:** **`research_out/momentum_portability_harness.json`** (override with **`--output`**).
  - **Implementation:** reuses real V44 via **`scripts/backtest_merged_integrated_tokyo_london_v2_ny.py`**, conservative ownership from **`core/ownership_table.py`**, same regime/cell semantics as the rest of the stack.

**Status:** Code path and wiring are validated; runs are **long** (real V44 over large CSVs). A finished harness JSON is **not** guaranteed until a run completes to the end.

---

## Plan from here (forward)

### 1. Defensive lane — Pocket #1 only (Pocket #2 closed)

1. **Live confirmation before any Pocket #1 live veto**
   - Run **`diagnostic_phase3_ownership_pain_log.py`** (or equivalent) on production **`phase3_minute_diagnostics.log`** and confirm **`ambiguous/er_low/der_neg`** pain matches research intent.
2. **Explicit go / no-go**
   - Reconcile **500k** (DD worse, smaller sample) vs **1000k** (DD better, more blocks): decide whether a **narrow** `v44_ny` block in this cell is acceptable given your DD and frequency objectives.
3. **Do not promote Pocket #2** as a defensive block unless objectives change (e.g. DD-only pilot with signed acceptance of negative Δnet).

### 2. Portability lane — Agent 3 harness

1. Prefer **`--raw-only`** when you need **evidence quickly** or want to avoid Variant K coupling cost.
2. Run **full** mode when you need **coupled shadow vs K** for board-ready numbers.
3. Treat **`momentum_portability_harness.json`** as the **source of truth** for 3A once produced; until then, status remains “harness not finalized.”

### 3. Cross-agent promotion

1. Use **`promotion_gate_board.py`** (plus **`portability_program_gates.py`**) for **weekly** promote / defer / close decisions.
2. Keep **Agent 2** shadow ledger updated so offensive and defensive candidates are not conflated.

### 4. Next research pockets (only if needed)

1. After Pocket #1 decision, **do not** assume a third defensive pocket exists: many London negatives are **already covered by Variant K’s cluster** or are **thin n**.
2. Any new veto must repeat: **stable cell**, **both-dataset evidence**, **coupled backtest vs K**, then **live pain log**.

---

## Quick reference — key artifacts

| Artifact | Purpose |
|----------|---------|
| `research_out/defensive_ownership_v15_candidates.json` | Defensive cell candidates |
| `research_out/defensive_v15_pocket1_v44_ambiguous_low_derneg.json` | Pocket #1 backtest vs K |
| `research_out/defensive_v15_pocket2_v44_ambiguous_high_er_pos.json` | Pocket #2 (rejected) |
| `research_out/diagnostic_ownership_stability.json` | Cross-dataset stability flags |
| `research_out/momentum_portability_harness.json` | Agent 3 harness output (when run completes) |
| `research_out/defensive_v15_pocket1_promotion_rubric.json` | Pocket #1 frozen scope + acceptance thresholds |
| `research_out/defensive_v15_pocket1_board_package.md` (and `.json`) | Board recommendation + DD/frequency rationale |
| `research_out/defensive_v15_pocket1_live_pain_alignment.json` | Live pain-log capture status + command |
| `research_out/defensive_v15_pocket1_semantic_crosscheck.json` | Live vs research semantic checks |
| `research_out/defensive_v15_pocket1_rollout_checklist.md` | Pilot / defer verification checklist |
| `research_out/track3_portability_promotion_criteria.json` | Track 3 dual-lens promote vs shadow criteria |
| `research_out/track3_v44_shadow_registry_parity.json` | V44 registry parity protocol + integrity |
| `research_out/track3_v14_london_evaluator_parity_spotcheck.json` | V14/London shadow builder spot-check |
| `research_out/track3_london_surface_summary.md` | London-heavy surface (diagnostic consolidation) |
| `research_out/track3_chart_first_router_boundary.md` | Chart-first beyond shadow (design boundary) |

---

*Last updated: Track 3 near-100 push artifacts (parity scripts, promotion criteria, London/router notes); Track 1 Near-100 Pocket #1 board package; Agent 3 harness scaffold.*
