#!/usr/bin/env python3
"""Write Phase 8 synthesis documents for the Autonomous Fillmore audit.

Phase 8 is synthesis only. The numbers below are locked outputs from
Phases 1-7 and are intentionally not recomputed here.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research_out" / "autonomous_fillmore_forensic_20260501"


CAUSAL_HIERARCHY = """# PHASE 8.1 - Causal Hierarchy

_Synthesis only. No new analysis. Source set: `PHASE1_BASELINE.md` through `PHASE7_INTERACTION_EFFECTS.md` and `phase{1..7}_*.csv`._

## Ranked Root Causes

| rank | tier | root cause | attributed pip impact | attributed USD impact | impact basis | supporting evidence | confidence | mechanism |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | PRIMARY | Sell-side caveat-template collapse | +278.4p recoverable in-sample | +$4,397.69 recoverable in-sample | Diagnostic counterfactual: `V1_caveat_cluster_sell` | Phase 4 sell-CLR forensic; Phase 7 minimum veto floor; `phase4_sell_clr_vs_buy_clr.csv`; `phase7_minimum_veto_rules.csv` | High | The model uses caveat/level language on sells as permission to trade instead of as a reason to abstain. |
| 2 | PRIMARY | Random/edge-blind sizing amplification | 0.0p | -$5,187.24 | Additive dollar ledger | Phase 5 sizing decomposition; Phase 7 sizing reconciliation; `phase5_pl_decomposition.csv`; `phase7_sizing_compounding.csv` | High | Lots do not track realized edge, so a negative pip base becomes a much larger dollar loss. |
| 3 | PRIMARY | Admission layer negative pip expectancy | -308.0p | -$2,065.99 at uniform 1 lot | Additive ledger | Phase 1 baseline; Phase 5 dollar decomposition; `phase1_closed_autonomous_trades.csv`; `phase5_pl_decomposition.csv` | High | The system placed a set of trades that lost pips even before variable sizing. |
| 4 | PRIMARY | Entry-generated level failure | -305.1p across 25 trades | -$5,514.55 actual USD footprint | Diagnostic lifecycle footprint, overlaps sizing | Phase 6 lifecycle attribution; Phase 7 entry-failure concentration; `phase6_entry_exit_attribution.csv`; `phase7_sizing_compounding.csv` | High | The worst losses often had terminal MFE <=2p and MAE >=6p, meaning the entry thesis failed before the trade produced useful green. |
| 5 | SECONDARY | Exit-reversal leakage | -218.7p across 24 trades | -$3,345.82 actual USD footprint | Diagnostic lifecycle footprint, overlaps sizing | Phase 6 lifecycle attribution; Phase 7 sizing compounding; `phase6_entry_exit_attribution.csv`; `phase7_sizing_compounding.csv` | Medium | Some trades had >=4p terminal MFE but still closed red, so exit logic leaked part of the already weak expectancy. |
| 6 | SECONDARY | Runner/custom-exit v3 regime harm | -92.2p | -$1,777.09 | Regime footprint | Phase 6 runner-preservation test; Phase 7 day-of-week interaction; `phase6_runner_regime_test.csv`; `phase7_day_of_week_interaction.csv` | High | Runner-preservation language increased exposure to a negative-expectancy regime without enough winner extension. |
| 7 | CONTRIBUTORY | Snapshot lacks side-normalized level and portfolio context | Not directly additive | Not directly additive | Structural input defect | Phase 3 snapshot audit; Phase 5 required telemetry; `phase3_schema_timeline.csv`; `phase5_required_telemetry.csv` | High | The LLM sees raw support/resistance and account primitives, but not side-relevant level quality, exposure, risk-after-fill, or rolling P&L. |
| 8 | CONTRIBUTORY | Winner capture/exit asymmetry | Winners captured median 0.69x MFE; 2.45p median MFE left | Not directly additive | Lifecycle symptom | Phase 6 capture efficiency; `phase6_capture_efficiency.csv` | Medium | Winners leave green on the table while losers realize full adverse path or worse under imperfect MAE telemetry. |
| 9 | CONTRIBUTORY | Phase 3 stop-overshoot volatility regime | Phase 3 overshoot rate 41.4% | Not directly additive | Regime anomaly | Phase 7 stop overshoot anomaly; `phase7_stop_overshoot_anomaly.csv` | Medium | Phase 3 ran during much higher M5 ATR; stop overshoot is best explained by volatility/slippage or exit-fill mechanics, not proven LLM intent. |
| 10 | INCIDENTAL | Exact 10D interaction sparsity | 0 exact 10D cells with N>=10 | N/A | Evidence limitation | Phase 7 damage concentration; `phase7_damage_concentration.csv` | High | The corpus is too small for exact 10-dimensional rules; reliable signals live in lower-cardinality cells. |

## Refuted Or Downgraded Causes

| hypothesis | status | evidence | confidence |
| --- | --- | --- | --- |
| Sells failed because USDJPY macro drift punished shorts | REFUTED | Phase 2 found passive shorts had a +170.7p tailwind. | High |
| Sell-CLR snapshots were thinner than buy-CLR snapshots | REFUTED | Phase 3 found core market fields symmetric across buy/sell CLR. | High |
| Sizing was anti-Kelly | REFUTED | Phase 5 found sizing random/edge-blind; Spearman lots-vs-pips -0.043. | High |
| 8+ lot CLR is real preserved edge | REFUTED | Phase 5 found the cell compositional, buy-CLR dominated, and fragile to date tests. | High |
| Caveat-language losers were held longer at exit | REFUTED | Phase 6 found caveat laundering in entry/sizing, not exit hold behavior. | Medium |
| Entry-failure trades attracted disproportionate oversizing | REFUTED | Phase 7 found only 40.0% above median sizing. | High |
| Caveat language predicts failure type | REFUTED | Phase 7 chi-square permutation p=0.70. | High |
| Phase 3 stop overshoot is proven exit-system bug | DOWNGRADED | Phase 7 ranked slippage/microstructure best-supported. | Medium |
| Phase 3 house-edge prompt alone caused Phase 3 damage | PARTIALLY REFUTED | Phase 7 found the house-edge window overlapped high M5 ATR, so prompt and volatility are confounded for stop/exit claims. | Medium |

## Reconciliation

The hierarchy contains diagnostic footprints that overlap. The additive accounting ledger is separate and reconciles exactly:

| ledger row | pips | USD | source |
| --- | --- | --- | --- |
| Admission at uniform 1 lot | -308.0p | -$2,065.99 | Phase 5 `phase5_pl_decomposition.csv` |
| Variable sizing amplification | 0.0p | -$5,187.24 | Phase 5 `phase5_pl_decomposition.csv` |
| Overlap term | 0.0p | $0.00 | Phase 5 `phase5_pl_decomposition.csv` |
| Observed total | -308.0p | -$7,253.24 | Phase 1 baseline / Phase 5 reconciliation |

Pip sub-ledger from Phase 6 reconciles the -308.0p admission result:

| pip bucket | pips | trades | source |
| --- | --- | --- | --- |
| Entry-failure proxy | -305.1p | 25 | `phase6_entry_exit_attribution.csv` |
| Exit-reversal proxy | -218.7p | 24 | `phase6_entry_exit_attribution.csv` |
| Other losses | -474.3p | 64 | `phase6_entry_exit_attribution.csv` |
| Winner offset | +690.1p | 128 | `phase6_entry_exit_attribution.csv` |
| Observed net | -308.0p | 241 | `phase6_entry_exit_attribution.csv` |

The diagnostic veto footprints do not sum to observed P&L because they overlap the additive admission and sizing ledgers. Phase 8 therefore treats them as in-sample damage-avoidance floors, not additive accounting rows.
"""


FAILURE_NARRATIVE = """# PHASE 8.2 - Unified Failure Narrative

Auto Fillmore lost -308.0p / -$7,253.24 over 241 closed autonomous trades because several layers failed in sequence, then amplified each other.

The gate layer admitted roughly plausible setups, but it did not condition them well enough by side, regime, or reasoning quality. Critical Level Reaction was not universally broken: buy-CLR was near breakeven or positive in protected cells, while sell-CLR collapsed. That means the gate was not simply firing random noise; it was firing setups whose edge depended heavily on side and context.

The snapshot layer was not corrupt, and buy/sell CLR coverage was broadly symmetric. But it was insufficient. It gave the model raw support/resistance and account primitives, not a side-normalized level packet, level age/touch/broken metadata, portfolio exposure, rolling P&L, risk-after-fill, or path-time MAE/MFE. The LLM had to infer too much from raw fields.

The reasoning layer was the primary cognitive failure. On shorts, it collapsed into a weak caveat template: "this is risky/mixed, but tradeable." It converted caveats into permission, used structure-only phrases like reject/reclaim/support as catalysts, and often claimed strong level evidence against weak or mixed snapshot evidence. Phase 7 showed the sell-side caveat cluster veto alone would recover 278.4p and $4,397.69 in-sample.

The decision layer failed to make caveats terminal. The model could recite red flags and still place. The runner/custom-exit v3 regime added measurable harm (-92.2p / -$1,777.09), showing that prompt language could change behavior in the wrong direction.

The sizing layer then turned a bad pip base into a severe dollar drawdown. At uniform 1 lot the same admissions lost -$2,065.99; variable sizing added -$5,187.24. Sizing was random/edge-blind, not anti-Kelly, which means it did not reliably know when to press or when to shrink.

The exit layer was secondary. Winners captured only 0.69x terminal MFE and left 2.45p median MFE behind, while 24 exit-reversal trades lost -218.7p after reaching at least 4p MFE. But the largest pip bucket remained entry-generated: 25 entry-failure trades lost -305.1p, and the 17 known CLR fast failures alone lost -207.1p.

Finally, the telemetry layer prevented fast learning. No skip forward outcomes, no path-time buckets, no exit replay, no snapshot version, no open exposure, and no first-class pip value means the system could not fully diagnose itself while running.
"""


FAILURE_ARCHITECTURE = """# PHASE 8.3 - Structural Failure Architecture

```mermaid
flowchart TD
    A["Input Layer: Market Snapshot"] --> B["Cognition Layer: LLM Reasoning"]
    B --> C["Decision Layer: Trade / Skip"]
    C --> D["Action Layer: Sizing"]
    D --> E["Lifecycle Layer: Exit Management"]
    E --> F["Feedback Layer: Telemetry / Learning"]
    F -. incomplete feedback .-> A
```

| layer | confirmed defect | attributed damage / footprint | severity | evidence |
| --- | --- | --- | --- | --- |
| Input / snapshot | Raw fields are symmetric but not side-normalized; no level quality packet, open exposure, rolling P&L, risk-after-fill, path-time buckets, or snapshot_version. | Structural enabler, not directly additive. | High | Phase 3 `PHASE3_SNAPSHOT_AUDIT.md`; `phase3_schema_timeline.csv`; `phase5_required_telemetry.csv` |
| Cognition / reasoning | Sell-side caveat template collapse; caveats become permission; level-language overreach. | V1 sell-side caveat veto recovers +278.4p / +$4,397.69 in-sample. | Critical | Phase 4 `PHASE4_REASONING_FORENSICS.md`; Phase 7 `phase7_minimum_veto_rules.csv` |
| Decision / admission | The model places trades whose rationales include unresolved weakness; prompt does not make contradiction terminal. | Uniform 1-lot admission loss -308.0p / -$2,065.99. | Critical | Phase 5 `phase5_pl_decomposition.csv`; Phase 7 `phase7_damage_concentration.csv` |
| Action / sizing | Variable sizing is random/edge-blind, not positively related to realized edge. | Additive sizing amplification -$5,187.24. | Critical | Phase 5 `phase5_edge_size_correlation.csv`; `phase5_pl_decomposition.csv` |
| Lifecycle / exits | Winners cut early; some exit reversals; runner/custom-exit v3 measurable harm; Phase 3 stop overshoot likely volatility/slippage. | Exit-reversal proxy -218.7p; runner v3 -92.2p / -$1,777.09. | Medium | Phase 6 `phase6_entry_exit_attribution.csv`; Phase 7 `phase7_stop_overshoot_anomaly.csv` |
| Feedback / telemetry | Missing skip outcomes, path-time MAE/MFE, exit replay, open exposure, pip value, multi-gate candidates. | Prevents exact root-cause closure and live self-correction. | High | Phase 3, Phase 5, Phase 6 evidence gaps |

## Damage Flow

1. Gate fires on a context-sensitive setup.
2. Snapshot supplies raw facts but not side-normalized quality or portfolio state.
3. LLM rationalizes weakness, especially on sell-side caveat trades.
4. Decision layer places the trade despite unresolved caveats.
5. Sizing layer assigns lots without edge awareness.
6. Bad entries fail quickly or exits leak some reversals.
7. Telemetry cannot fully teach the system which skip/place/exit decisions were correct.
"""


PRESERVED_EDGE = """# PHASE 8.4 - Preserved Edge Catalog

_Binding Phase 9 constraint: the overhaul cannot be allowed to destroy these cells while pursuing broader damage avoidance. These are the only cells that survived leave-one-date-out with positive pips and USD._

| preserved cell | N | win rate | net pips | net USD | expectancy | LOO min pips | LOO min USD | mechanism hypothesis | Phase 9 protection requirement | evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CLR x buy x Phase 2 zone-memory x caveat-trade x 2-3.99 lots | 17 | 82.4% | +56.7p | +$787.33 | +3.34p/trade | +26.7p | +$288.68 | Buy-side critical-level reactions with moderate size and zone-memory context had enough side-aligned structure to overcome caveat language. | Preserve buy-side CLR when zone-memory context is favorable and size is moderate; do not bluntly kill all caveat-language CLR. | Phase 7 `phase7_preserved_edge_cells.csv` |
| Tuesday x CLR x buy | 17 | 82.4% | +47.7p | +$592.84 | +2.81p/trade | +17.9p | +$233.29 | Tuesday's positive day effect is not a day-only edge; it is concentrated in buy-side CLR. | Protect the buy-side CLR conditions that made Tuesday work; do not preserve Tuesday as a generic trade day. | Phase 7 `phase7_preserved_edge_cells.csv`; `phase7_day_of_week_interaction.csv` |
| CLR x buy x Phase 2 zone-memory x caveat-trade | 23 | 69.6% | +33.6p | +$500.26 | +1.46p/trade | +15.7p | +$233.29 | The broader protected cell is the same phenomenon with a little more noise: buy-side CLR plus zone-memory context. | Phase 9 must distinguish this cell from sell-side CLR caveat collapse. | Phase 7 `phase7_preserved_edge_cells.csv` |

## What Is Not Preserved Edge

| candidate | verdict | reason |
| --- | --- | --- |
| Phase A buy-CLR caveat-trade | Not robust | Positive in aggregate but fails leave-one-date-out. |
| 8+ lot CLR cell | Not robust | Phase 5 found it compositional and buy-CLR dominated; Phase 7 leave-one-date-out fails. |
| Tuesday as a full day-of-week edge | Not robust | Day-only Tuesday fails leave-one-date-out; the robust part is Tuesday buy-CLR. |
| Momentum-continuation sell, Phase 3, 0-1.99 lots | Not robust | Positive candidate but leave-one-date-out fails. |
"""


DIAGNOSTIC_FLOOR = """# PHASE 8.5 - Diagnostic Floor For Phase 9

_Binding Phase 9 constraint: any redesign that does not match the V1+V2 retroactive damage-avoidance rate with comparable or better false-positive cost is not defensible._

Phase 9 does not have to use these exact rules. But its proposed combination of gate, snapshot, reasoning, sizing, and exit changes must beat the floor below when replayed retroactively on the same evidence.

## Floor

| floor | rule set | blocked trades | blocked winners | blocked losers | missed winner pips | missed winner USD | net pip recovery | net USD recovery | evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pip floor | V1: `caveat_cluster_sell` | 80 | 34 | 46 | 170.4p | $2,283.96 | +278.4p / 90.4% | +$4,397.69 / 60.6% | Phase 7 `phase7_minimum_veto_rules.csv` |
| USD floor | V1 + V2: `caveat_cluster_sell` + `entry_signature_mixed_overlap` | 110 | 52 | 58 | 278.1p | $3,402.16 | +300.5p / 97.6% | +$5,684.56 / 78.4% | Phase 7 `phase7_minimum_veto_rules.csv` |

## False-Positive Constraint

The floor is not "delete everything." V1+V2 blocks 110 of 241 trades, including 52 winners. It removes 278.1 winning pips and $3,402.16 winner USD while still recovering $5,684.56 net. Phase 9 must either:

- match or exceed +300.5p and +$5,684.56 in-sample net recovery with fewer false positives, or
- preserve materially more of the robust buy-side CLR edge while reaching comparable net recovery.

If a Phase 9 redesign recovers less than V1+V2, or recovers it only by destroying the preserved-edge catalog, it fails the diagnostic floor.
"""


REFUTED = """# PHASE 8.6 - Refuted Hypotheses

_These are off-limits for Phase 9 as primary explanations. The overhaul should not spend effort targeting them._

| hypothesis | status | evidence | phase reference |
| --- | --- | --- | --- |
| "Sells failed because USDJPY was in a bull regime." | REFUTED | Passive shorts had a +170.7p tailwind over Apr 16-May 1, yet Fillmore sells lost heavily. | Phase 2 `PHASE2_GATE_AUDIT.md` |
| "Sell-CLR snapshots were thinner than buy-CLR." | REFUTED | Core market fields were symmetric across buy/sell CLR; snapshot coverage did not explain sell collapse. | Phase 3 `PHASE3_SNAPSHOT_AUDIT.md` |
| "Sizing was anti-Kelly." | REFUTED | Spearman lots-vs-pips was -0.043; sizing is random/edge-blind, not a stable inverse edge estimator. | Phase 5 `phase5_edge_size_correlation.csv` |
| "8+ lot CLR cell is real edge." | REFUTED | It was a compositional artifact, buy-CLR dominated, near flat, and fragile to leave-one-date-out. | Phase 5 `phase5_8lot_clr_anomaly.csv`; Phase 7 `phase7_preserved_edge_cells.csv` |
| "Caveat-language losers are held longer." | REFUTED | Caveat laundering is confirmed in entry/sizing, but not exit hold behavior. | Phase 6 `phase6_caveat_exit_interaction.csv` |
| "Entry-failure trades disproportionately attracted oversizing." | REFUTED | Only 40.0% of entry-failure trades used above-median lots. | Phase 7 `phase7_sizing_compounding.csv` |
| "Caveat language predicts failure type." | REFUTED | Rationale-cluster x outcome-type permutation chi-square p=0.70. | Phase 7 `phase7_caveat_outcome_interaction.csv` |
| "Phase 3 stop overshoot is a proven exit-system bug." | DOWNGRADED | Phase 7 ranked slippage/microstructure as best-supported; exit-system bug remains possible but unproven. | Phase 7 `phase7_stop_overshoot_anomaly.csv` |
| "Phase 3 house-edge prompt is purely the cause of its damage." | PARTIALLY REFUTED | Phase 3 coincided with much higher M5 ATR, so prompt and volatility regime are confounded. | Phase 7 `phase7_stop_overshoot_anomaly.csv` |
"""


AMBIGUITY = """# PHASE 8.7 - Genuine Ambiguity Register

_These questions remain open because telemetry is missing. Phase 9 must assume conservatively rather than pretending the audit answered them._

| ambiguity | why audit cannot answer | telemetry required to close | Phase 9 design implication |
| --- | --- | --- | --- |
| Was MAE usually reached before MFE in losers? | No path-time MAE/MFE buckets; stored MAE under-reports realized loss in 77.9% of losers. | Exit-inclusive MAE plus MAE/MFE at 1/3/5/15 minutes or tick-path replay. | Treat lifecycle counterfactuals as upper bounds; do not overfit exit logic from terminal MAE/MFE. |
| Were skips good or bad? | Skipped calls lack forward price, post-gate MAE/MFE, and expiry outcome fields. | For every skip: price_at_expiry, distance_at_expiry_pips, post-gate MAE/MFE, snapshot retention. | Selectivity changes must be tested prospectively or with new skip telemetry. |
| Did the exit manager follow the prompt internally? | Exit-manager intermediate decisions are not replayable. | Exit decision logs, stop/TP changes, trail updates, close reason, and rendered exit context. | Do not attribute stop overshoot to LLM intent without replay. |
| Did open exposure influence sizing? | Open lots, unrealized P&L, side exposure, and risk-after-fill were absent. | open_usdjpy_lots_by_side, unrealized_pnl_by_side, risk_after_fill_usd. | Any sizing redesign must include deterministic exposure/risk context. |
| Was Phase 3 house-edge damage separable from high ATR? | Prompt window overlaps M5 ATR ~15.6p vs Phase A ~5.1p. | Snapshot_version, prompt_version, volatility regime, and enough independent samples per regime. | Treat Phase 3 prompt conclusions as confounded with volatility for exit/overshoot claims. |
| Did the LLM reason over full context or only persisted snapshot subset? | Persisted snapshot is not the full final rendered prompt/context. | Store full rendered prompt or canonical context JSON per call. | Future audits must replay exactly what the model saw. |
| Were multi-gate conflicts present? | Only selected `trigger_family` was logged. | Log all candidate gate families, scores, and veto reasons per poll. | Phase 9 should not assume the selected gate was the only plausible gate. |
| What is exact dollar risk per proposed trade? | Pip value is inferred, not first-class; risk-after-fill missing. | pip_value_per_lot and proposed loss-at-stop in USD. | Size controls should be deterministic until telemetry proves model sizing can be trusted. |
"""


HANDOFF_SPEC = """# PHASE 8.8 - Phase 9 Hand-off Spec

This is the bridge from diagnosis to redesign. Phase 9 should not need to re-read the full Phase 1-7 reports to know the constraints.

## Causal Hierarchy To Build Against

1. Primary cognitive failure: sell-side caveat-template collapse. V1 would recover +278.4p / +$4,397.69 in-sample.
2. Primary dollar failure: random/edge-blind sizing amplification. Adds -$5,187.24 to the uniform-lot admission loss.
3. Primary pip failure: admission layer negative expectancy. Same trades at 1 lot still lose -308.0p / -$2,065.99.
4. Primary lifecycle source: entry-generated damage. Entry-failure proxy loses -305.1p; 17 CLR fast failures lose -207.1p.
5. Secondary lifecycle leakage: exit reversals lose -218.7p; runner/custom-exit v3 loses -92.2p / -$1,777.09.

## Preserved Edge To Protect

| cell | protection requirement |
| --- | --- |
| CLR x buy x Phase 2 zone-memory x caveat-trade x 2-3.99 lots | Do not bluntly ban all caveat-language CLR. Preserve moderate-size buy-side CLR when zone-memory context is favorable. |
| Tuesday x CLR x buy | Preserve buy-side CLR conditions; do not treat Tuesday itself as a broad edge. |
| CLR x buy x Phase 2 zone-memory x caveat-trade | Separate buy-side CLR from sell-side CLR collapse. |

## Diagnostic Floor

Phase 9 must beat:

- +278.4p / +$4,397.69 from V1 alone.
- +300.5p / +$5,684.56 from V1+V2.
- V1+V2 false-positive cost: 110 blocked trades, 52 blocked winners, 278.1 missed winner pips, $3,402.16 missed winner USD.

## Required Telemetry By Priority

| priority | field | why it matters |
| --- | --- | --- |
| 1 | open_usdjpy_lots_by_side | Prevents blind adds into concentrated exposure. |
| 2 | risk_after_fill_usd | Converts proposed size and stop into explicit dollar loss. |
| 3 | rolling_20_trade_pnl_and_lot_weighted_pnl | Enables performance-aware selectivity without a max-daily-loss kill switch. |
| 4 | unrealized_pnl_by_side | Separates new trades from hedges/adds into losing inventory. |
| 5 | pip_value_per_lot | Makes fixed-dollar risk calculable without inference. |
| 6 | side-normalized level packet | Tells the model entry-wall strength and profit-path blocker for the proposed side. |
| 7 | level age / touch count / broken-reclaimed flag | Directly addresses CLR fast failures. |
| 8 | path-time MAE/MFE buckets | Separates immediate entry failure from exit drift. |
| 9 | skip forward outcomes | Allows selectivity quality to be measured. |
| 10 | snapshot_version and full rendered prompt/context | Makes future regime comparisons exact. |
| 11 | all gate candidates and scores | Makes overlap/contradiction auditable. |
| 12 | exit-manager replay | Separates exit-system bug from slippage or LLM intent. |

## Confirmed Prompt Clauses To Remove Or Rewrite

- Runner-preservation clauses such as "Preserve wider runners" / "leave a runner." Phase 6 found measurable harm.
- House-edge clauses that name red patterns without making caveat resolution terminal. Phase 4/7 show the model can recite red flags and still place.
- Catalyst instructions that accept structure-only text such as support/reject/reclaim. Phase 4 found this language in negative-expectancy clusters.

## Cognitive Failure Modes To Design Against

- Caveat laundering: contradiction/self-correction language must not become permission to trade.
- Sell-side template collapse: sell-side CLR and sell-side caveat clusters need materially higher proof than buy-side CLR.
- Level-language overreach: strong/clean/fresh level claims must be backed by actual level-quality fields.
- Sizing confidence mismatch: lot size cannot be chosen from verbal conviction.
- Reasoning without loss-asymmetry proof: a trade rationale must address why expected loser size will be smaller than expected winner size.

## Refuted Hypotheses Off-Limits

Do not redesign around these as primary causes: macro drift punishing shorts, thinner sell snapshots, anti-Kelly sizing, real 8+ lot CLR edge, caveat-laundered exits, entry-failure oversizing, caveat language predicting failure type, or proven LLM stop disregard.

## Ambiguity Phase 9 Must Carry

The redesign must assume uncertainty around path ordering, skip quality, exit-manager behavior, open-exposure effects, Phase 3 prompt-vs-volatility confounding, and exact rendered context.
"""


SYNTHESIS = """# PHASE 8 - ROOT CAUSE SYNTHESIS

_Auto Fillmore forensic investigation, Apr 16-May 1. This is the bridge from investigation to overhaul. No new analysis, no fixes, no prompt rewrite._

## Executive Verdict

Auto Fillmore did not fail because of one bad guardrail or one missing max-loss rule. It failed because the system admitted context-sensitive setups, asked a small model to infer side-specific edge from insufficiently normalized data, allowed the model to launder caveats into trades, then let random/edge-blind sizing multiply a negative pip base into a large dollar drawdown.

The primary cognitive failure is sell-side caveat-template collapse. The primary dollar failure is random sizing amplification. The primary pip failure is bad admission, especially entry-generated CLR failures. The preserved edge is narrow and real: buy-side CLR under Phase 2 zone-memory conditions, especially moderate 2-3.99 lot size.

Phase 9's minimum bar is now explicit: beat the V1+V2 diagnostic floor of +300.5p and +$5,684.56 in-sample recovery while preserving the protected buy-side CLR cells and improving the false-positive cost.

---

""" + CAUSAL_HIERARCHY.replace("# PHASE 8.1 - Causal Hierarchy\n\n", "## 8.1 Causal Hierarchy\n\n") + "\n---\n\n" + FAILURE_NARRATIVE.replace("# PHASE 8.2 - Unified Failure Narrative\n\n", "## 8.2 Unified Failure Narrative\n\n") + "\n---\n\n" + FAILURE_ARCHITECTURE.replace("# PHASE 8.3 - Structural Failure Architecture\n\n", "## 8.3 Structural Failure Architecture\n\n") + "\n---\n\n" + PRESERVED_EDGE.replace("# PHASE 8.4 - Preserved Edge Catalog\n\n", "## 8.4 Preserved Edge Catalog\n\n") + "\n---\n\n" + DIAGNOSTIC_FLOOR.replace("# PHASE 8.5 - Diagnostic Floor For Phase 9\n\n", "## 8.5 Diagnostic Floor For Phase 9\n\n") + "\n---\n\n" + REFUTED.replace("# PHASE 8.6 - Refuted Hypotheses\n\n", "## 8.6 Refuted Hypotheses\n\n") + "\n---\n\n" + AMBIGUITY.replace("# PHASE 8.7 - Genuine Ambiguity Register\n\n", "## 8.7 Genuine Ambiguity Register\n\n") + "\n---\n\n" + HANDOFF_SPEC.replace("# PHASE 8.8 - Phase 9 Hand-off Spec\n\n", "## 8.8 Phase 9 Hand-off Spec\n\n")


FILES = {
    "PHASE8_CAUSAL_HIERARCHY.md": CAUSAL_HIERARCHY,
    "PHASE8_FAILURE_NARRATIVE.md": FAILURE_NARRATIVE,
    "PHASE8_FAILURE_ARCHITECTURE.md": FAILURE_ARCHITECTURE,
    "PHASE8_PRESERVED_EDGE_CATALOG.md": PRESERVED_EDGE,
    "PHASE8_DIAGNOSTIC_FLOOR.md": DIAGNOSTIC_FLOOR,
    "PHASE8_REFUTED_HYPOTHESES.md": REFUTED,
    "PHASE8_AMBIGUITY_REGISTER.md": AMBIGUITY,
    "PHASE8_HANDOFF_SPEC.md": HANDOFF_SPEC,
    "PHASE8_SYNTHESIS.md": SYNTHESIS,
}


def main() -> None:
    for name, text in FILES.items():
        (OUT / name).write_text(text.rstrip() + "\n", encoding="utf-8")
    manifest = {
        "phase": 8,
        "mode": "synthesis_only",
        "outputs": sorted(FILES),
        "source_reports": [f"PHASE{i}" for i in range(1, 8)],
        "no_new_analysis": True,
    }
    (OUT / "PHASE8_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(FILES)} Phase 8 markdown files to {OUT}")


if __name__ == "__main__":
    main()
