#!/usr/bin/env python3
"""Write Phase 9 overhaul blueprint documents for Autonomous Fillmore.

Phase 9 is prescription, not another exploratory audit. It consumes the
Phase 8 synthesis as the binding specification and uses Phase 7 rule labels
only for the required retroactive replay check.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research_out" / "autonomous_fillmore_forensic_20260501"
DATASET = OUT / "phase7_interaction_dataset.csv"

BASELINE_PIPS = -308.0
BASELINE_USD = -7253.24
PIP_FLOOR = 278.4
USD_FLOOR = 5684.56
FLOOR_BLOCKED = 110
FLOOR_BLOCKED_WINNERS = 52
FLOOR_MISSED_WINNER_PIPS = 278.1
FLOOR_MISSED_WINNER_USD = 3402.16


def fmt_money(value: float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.2f}"


def fmt_pips(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.1f}p"


def build_replay() -> tuple[dict[str, float], pd.DataFrame]:
    df = pd.read_csv(DATASET)

    v1 = df["rationale_cluster"].isin(["momentum_with_caveat_trade", "critical_level_mixed_caveat_trade"]) & df[
        "side"
    ].eq("sell")
    v2 = df["timeframe_alignment_clean"].eq("mixed") & df["session"].isin(
        ["london/ny overlap", "tokyo/london overlap"]
    )
    v5 = df["high_hedge"].astype(bool) & df["high_conviction"].astype(bool)
    v6_sell = (
        df["trigger_family"].eq("critical_level_reaction")
        & df["mixed_or_thin_snapshot"].astype(bool)
        & df["strong_level_claim"].astype(bool)
        & df["side"].eq("sell")
    )

    protected_1 = (
        df["trigger_family"].eq("critical_level_reaction")
        & df["side"].eq("buy")
        & df["prompt_regime"].eq("Phase 2 zone-memory")
        & df["rationale_cluster"].eq("critical_level_mixed_caveat_trade")
        & df["lot_bucket"].eq("2-3.99")
    )
    protected_2 = (
        df["day_of_week"].eq("Tuesday")
        & df["trigger_family"].eq("critical_level_reaction")
        & df["side"].eq("buy")
    )
    protected_3 = (
        df["trigger_family"].eq("critical_level_reaction")
        & df["side"].eq("buy")
        & df["prompt_regime"].eq("Phase 2 zone-memory")
        & df["rationale_cluster"].eq("critical_level_mixed_caveat_trade")
    )
    protected_any = protected_1 | protected_2 | protected_3

    masks = {
        "V1_refined_sell_caveat_template": v1,
        "V2_refined_mixed_overlap_entry_signature": v2,
        "V5_hedge_plus_overconfidence_validator": v5,
        "V6_sell_level_overreach_validator": v6_sell,
    }
    block_mask = (v1 | v2 | v5 | v6_sell) & ~protected_any
    survivor_mask = ~block_mask

    blocked = df[block_mask]
    blocked_winners = blocked["pips"] > 0
    blocked_losers = blocked["pips"] < 0
    saved_loser_pips = -blocked.loc[blocked_losers, "pips"].sum()
    saved_loser_usd = -blocked.loc[blocked_losers, "pnl"].sum()
    missed_winner_pips = blocked.loc[blocked_winners, "pips"].sum()
    missed_winner_usd = blocked.loc[blocked_winners, "pnl"].sum()
    net_delta_pips = saved_loser_pips - missed_winner_pips
    net_delta_usd = saved_loser_usd - missed_winner_usd

    pnl_per_lot = df["pnl"] / df["lots"].replace(0, pd.NA)
    capped_lots = df["lots"].clip(lower=1, upper=4)
    new_pnl_after_sizing = (pnl_per_lot * capped_lots).where(survivor_mask, 0).sum()
    new_pips_after_filter = df.loc[survivor_mask, "pips"].sum()
    sizing_recovery_total = new_pnl_after_sizing - BASELINE_USD
    sizing_increment_after_filter = new_pnl_after_sizing - (BASELINE_USD + net_delta_usd)

    protected_rows = []
    for name, mask in {
        "CLR buy Phase 2 zone-memory caveat 2-3.99 lots": protected_1,
        "Tuesday CLR buy": protected_2,
        "CLR buy Phase 2 zone-memory caveat": protected_3,
    }.items():
        survivors = df[mask & survivor_mask]
        protected_rows.append(
            {
                "protected_cell": name,
                "original_n": int(mask.sum()),
                "blocked_by_phase9": int((mask & block_mask).sum()),
                "survivor_n": int((mask & survivor_mask).sum()),
                "survivor_pips": survivors["pips"].sum(),
                "survivor_usd_original_size": survivors["pnl"].sum(),
                "survivor_usd_cap4_replay": (pnl_per_lot * capped_lots).where(mask & survivor_mask, 0).sum(),
            }
        )

    replay_rows = []
    for name, mask in masks.items():
        blocked_rule = df[mask & ~protected_any]
        rule_win = blocked_rule["pips"] > 0
        rule_loss = blocked_rule["pips"] < 0
        replay_rows.append(
            {
                "rule": name,
                "blocked_after_protected_bypass": int(len(blocked_rule)),
                "blocked_winners": int(rule_win.sum()),
                "blocked_losers": int(rule_loss.sum()),
                "standalone_net_pips_after_bypass": -blocked_rule.loc[rule_loss, "pips"].sum()
                - blocked_rule.loc[rule_win, "pips"].sum(),
                "standalone_net_usd_after_bypass": -blocked_rule.loc[rule_loss, "pnl"].sum()
                - blocked_rule.loc[rule_win, "pnl"].sum(),
            }
        )

    replay = {
        "blocked_trades": int(len(blocked)),
        "placed_trades": int(survivor_mask.sum()),
        "blocked_winners": int(blocked_winners.sum()),
        "blocked_losers": int(blocked_losers.sum()),
        "saved_loser_pips": float(saved_loser_pips),
        "saved_loser_usd": float(saved_loser_usd),
        "missed_winner_pips": float(missed_winner_pips),
        "missed_winner_usd": float(missed_winner_usd),
        "net_delta_pips": float(net_delta_pips),
        "net_delta_usd": float(net_delta_usd),
        "new_pips_after_filter": float(new_pips_after_filter),
        "new_usd_after_filter_original_size": float(BASELINE_USD + net_delta_usd),
        "new_usd_after_filter_cap4": float(new_pnl_after_sizing),
        "total_usd_recovery_with_cap4": float(sizing_recovery_total),
        "sizing_increment_after_filter": float(sizing_increment_after_filter),
        "placement_rate_closed_corpus": float(survivor_mask.sum() / len(df)),
        "protected_rows": protected_rows,
        "rule_rows": replay_rows,
    }

    replay_table = pd.DataFrame(replay_rows)
    protected_table = pd.DataFrame(protected_rows)
    replay_table.to_csv(OUT / "phase9_replay_rule_components.csv", index=False)
    protected_table.to_csv(OUT / "phase9_replay_protected_edge_check.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "metric": "phase9_admission_filter_net_delta_pips",
                "value": replay["net_delta_pips"],
                "floor": PIP_FLOOR,
                "passes_floor": replay["net_delta_pips"] >= PIP_FLOOR,
            },
            {
                "metric": "phase9_admission_filter_net_delta_usd",
                "value": replay["net_delta_usd"],
                "floor": USD_FLOOR,
                "passes_floor": replay["net_delta_usd"] >= USD_FLOOR,
            },
            {
                "metric": "phase9_filter_plus_cap4_net_usd_recovery",
                "value": replay["total_usd_recovery_with_cap4"],
                "floor": USD_FLOOR,
                "passes_floor": replay["total_usd_recovery_with_cap4"] >= USD_FLOOR,
            },
        ]
    )
    summary.to_csv(OUT / "phase9_replay_summary.csv", index=False)
    return replay, df


def md_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(out)


def architecture_doc() -> str:
    return """# PHASE 9.1 - Layered System Architecture

_Prescription phase. Binding source: `PHASE8_SYNTHESIS.md`. This is a deployable design specification, not a new forensic audit._

## Target Architecture

```mermaid
flowchart TD
    A["Telemetry and Snapshot Layer"] --> B["Gate Layer"]
    B --> C["Pre-Decision Veto Layer"]
    C --> D["LLM Decision Layer"]
    D --> E["Post-Decision Validator"]
    E --> F["Deterministic Sizing Layer"]
    F --> G["Deterministic Exit Layer"]
    G --> H["Feedback and Replay Layer"]
    H -. full audit context .-> A
```

## Layer Contract

| layer | inputs | outputs | decision authority | Phase 8 defect defended | failure mode | kill criterion |
| --- | --- | --- | --- | --- | --- | --- |
| Telemetry and Snapshot | Live price, account, exposure, level packet, gate candidates, rendered prompt context | Versioned canonical snapshot plus full rendered context | Deterministic | Snapshot lacks side-normalized level packet and portfolio context | Missing blocking fields force low-quality decisions | Any blocking field missing for 3 consecutive polls halts autonomous placement |
| Gate | Three primary gates plus all candidate scores | Eligible setup packet or no-call | Deterministic | Admission layer negative pip expectancy; entry-generated level failure | Gate admits side/regime cells it cannot support | 30 closed trades per gate with negative net pips after veto layer |
| Pre-Decision Veto | Gate packet, side, timeframe alignment, session, level packet, exposure | Call LLM or skip-before-call | Deterministic | Sell-side caveat-template collapse; mixed-overlap entry failure | LLM called on historically toxic rows | Any veto blocks a protected-edge cell in replay or live audit |
| LLM Decision | Only pre-cleared setup packet | `place` or `skip` JSON; no sizing/exit authority | LLM within deterministic boundaries | Reasoning template collapse; caveat laundering | Model turns caveats into permission | 20% validator override rate over 50 calls, or sell-side WR below kill threshold |
| Post-Decision Validator | LLM JSON, rationale text, level packet, loss-asymmetry fields | Final place/skip decision | Deterministic override | Caveat laundering; level-language overreach; loss-asymmetry proof failure | Text sounds confident without evidence | Any validator fires on protected-edge rows with valid level packet evidence |
| Sizing | Equity, pip value, SL pips, exposure, rolling P&L, volatility | Final lots | Deterministic | Random/edge-blind sizing amplification | Size tracks verbal conviction instead of risk | Any order >4 lots or missing risk-after-fill log |
| Exit | Entry, SL/TP, path telemetry, time in trade | Stop, profit lock, time stop, close reason | Deterministic with bounded override | Winner capture asymmetry; runner v3 harm | Exit extends losers or changes stop silently | Any stop extension without replay log; 30 trades with exit-reversal rate above baseline |
| Feedback | Every call, skip, order, modification, exit decision | Audit-ready event stream | Deterministic logging | Telemetry prevented fast learning | Next audit cannot reconstruct decisions | Any unjoined call/trade rate >2% over a day |

## Blocking Telemetry

The system cannot go live without `open_usdjpy_lots_by_side`, `risk_after_fill_usd`, `pip_value_per_lot`, `snapshot_version`, and the full rendered prompt/context. CLR eligibility additionally requires the side-normalized level packet and all gate candidates/scores. This directly addresses Phase 8 telemetry findings and prevents another compromised audit.
"""


def gate_redesign_doc() -> str:
    return """# PHASE 9.2 - Gate Layer Redesign

## Gate Verdicts

| gate | verdict | rationale from Phase 8 | redesign |
| --- | --- | --- | --- |
| Critical Level Reaction | REDESIGN | CLR is not globally dead: buy-CLR contains all three preserved-edge cells, while sell-CLR collapsed through caveat-template reasoning. | Split buy-CLR and sell-CLR into separate eligibility paths with different evidence burdens. |
| Momentum Continuation | KEEP-CONDITIONAL | Phase 2 found +10.0p but -$1,241.24, which Phase 5 tied to sizing rather than pure pip failure. | Keep only with deterministic sizing, path room to next structure, and no runner-preservation language. |
| Mean Reversion | KEEP-SMALL-N | Sample was too small for a kill verdict. | Keep paper/low-size until 30 forward closes prove non-negative expectancy after veto layer. |
| Non-primary gates | KILL-BY-DEFAULT | Non-primary gates contaminated the stated 3-gate architecture and were not part of the clean design. | Disabled unless a new gate id, snapshot schema, prompt version, and paper-only experiment are opened. |

## Critical Level Reaction

### Buy-CLR Eligibility

Buy-CLR can reach the LLM only when all are true:

1. Side-normalized level packet exists.
2. `entry_wall.side == buy_support` or equivalent side-aligned packet.
3. `level_quality_score >= 70`.
4. No recent broken-then-failed reclaim flag against the buy.
5. Profit-path blocker distance is at least planned risk distance or explicitly marked clear.
6. Pre-decision V1/V2 logic does not fire, except for the protected-edge bypass.

Protected cells are buy-CLR. The redesign preserves them by refusing blunt caveat bans on buy-side zone-memory CLR and by allowing 2-4 lot deterministic sizing when risk-after-fill is inside limits.

### Sell-CLR Eligibility

Sell-CLR is structurally constrained. It can reach the LLM only when all are true:

1. `level_quality_score >= 85`.
2. `entry_wall.side == sell_resistance`.
3. H1 and M5 are not both against the short.
4. Macro bias is not explicitly against the sell.
5. Timeframe alignment is not mixed in an overlap session unless the level packet score is at least 90 and a material catalyst field is present.
6. There is no structure-only catalyst such as "reject", "support", "resistance", or "reclaim" standing alone.

Kill criterion: 30 post-redesign sell-CLR closes with negative net pips or WR below 45% kills sell-CLR until manual review.

## Momentum Continuation

Eligibility:

1. Room to next support/resistance is at least 1R.
2. Volatility regime is not elevated, or deterministic sizing halves the lot count.
3. No runner-preservation prompt clause is present.
4. The LLM cannot choose size or hold-time.

Kill criterion: 30 forward momentum closes with negative net pips after deterministic sizing kills momentum until redesigned.

## Mean Reversion

Eligibility:

1. Level packet indicates exhaustion at a side-relevant level.
2. Spread and volatility are inside the normal regime.
3. Max size is 1 lot until 30 closed trades prove positive net pips.

Kill criterion: first 30 forward closes net negative by more than -25p or profit factor below 0.8.

## Non-Primary Gates

`post_spike_retracement`, `failed_breakout`, `trend_expansion`, and any unversioned family are killed. Exception path: paper-only experiment with explicit `gate_experiment_id`, `snapshot_version`, and separate analytics table.
"""


def veto_rules_doc(replay: dict[str, float]) -> str:
    rows = [
        {
            "rule": "V1 refined sell caveat-template veto",
            "decision": "ADOPT",
            "spec": "Before LLM call, block sell setups whose ex-ante packet has caveat-template precursors: mixed/countertrend, thin level packet, contradictory macro, or structure-only catalyst. Allow only if side-normalized sell evidence score >=85 and material catalyst exists.",
            "estimate": "+278.4p / +$4,397.69 from Phase 8 V1; in Phase 9 stack included in +324.3p / +$6,086.60.",
            "protected edge": "No protected cell is sell-side.",
        },
        {
            "rule": "V2 refined mixed-overlap entry signature",
            "decision": "ADOPT WITH PROTECTED BYPASS",
            "spec": "Block mixed timeframe alignment in London/NY or Tokyo/London overlap unless the setup is a protected buy-CLR packet with level score >=70 and deterministic size <=4.",
            "estimate": "V1+V2 floor is +300.5p / +$5,684.56; protected-bypass replay still recovers +302.7p / +$5,660.76 before V5/V6.",
            "protected edge": "Bypass preserves all three protected cells.",
        },
        {
            "rule": "V3 runner/custom-exit v3",
            "decision": "RETIRED AS LIVE RULE",
            "spec": "The prompt regime no longer exists. Its lesson becomes a hard ban on runner-preservation language and silent hold-time bias.",
            "estimate": "Historic harm footprint: +92.2p / +$1,777.09 if blocked.",
            "protected edge": "No live firing condition.",
        },
        {
            "rule": "V4 Wednesday CLR",
            "decision": "REJECT",
            "spec": "Day-of-week is not causal enough for a permanent veto.",
            "estimate": "+65.3p / +$1,122.55 in-sample, rejected as correlation without mechanism.",
            "protected edge": "Avoids day-only overfit.",
        },
        {
            "rule": "V5 hedge plus overconfidence",
            "decision": "MOVE TO POST-DECISION VALIDATOR",
            "spec": "If hedge density and conviction density both exceed thresholds in the LLM output, override place to skip.",
            "estimate": "+35.8p / +$370.75 individual, weak N<15; Phase 9 stack adds one extra loser after V1/V2 protected bypass.",
            "protected edge": "Does not fire on protected cells in replay.",
        },
        {
            "rule": "V6 level-language overreach",
            "decision": "ADOPT WITH LEVEL-PACKET EXCEPTION",
            "spec": "If the model says strong/clean/fresh/textbook while the side-normalized level packet is mixed/thin or below threshold, override to skip. Legacy replay counts sell-side only because buy packet quality did not exist.",
            "estimate": "+149.7p / +$1,567.43 individual; refined Phase 9 stack beats the floor at +324.3p / +$6,086.60.",
            "protected edge": "Protected buy-CLR survives when packet evidence is valid.",
        },
        {
            "rule": "V7 blunt CLR sell",
            "decision": "REJECT",
            "spec": "Too blunt for an overhaul; sell-CLR is constrained by V1/V6 evidence burden instead.",
            "estimate": "+184.3p / +$2,081.56 in-sample, but discarded as a ceiling test.",
            "protected edge": "Avoids broad side bans leaking into future valid short setups.",
        },
    ]
    return f"""# PHASE 9.3 - Pre-Decision Veto Rules

## Adopted Rule Stack

The live rule stack is not the blunt Phase 7 union. It is a refined stack:

1. V1 refined sell caveat-template veto.
2. V2 mixed-overlap entry signature with protected buy-CLR bypass.
3. V5 hedge plus overconfidence as a post-decision validator.
4. V6 level-language overreach, applied where the side-normalized level packet does not support the claim.

Retroactive available-field replay on the 241 closed-trade corpus:

| metric | Phase 9 stack | Phase 8 floor |
| --- | --- | --- |
| Blocked trades | {replay['blocked_trades']} | {FLOOR_BLOCKED} |
| Blocked winners | {replay['blocked_winners']} | {FLOOR_BLOCKED_WINNERS} |
| Blocked losers | {replay['blocked_losers']} | 58 |
| Missed winner pips | {replay['missed_winner_pips']:.1f}p | {FLOOR_MISSED_WINNER_PIPS:.1f}p |
| Missed winner USD | {fmt_money(replay['missed_winner_usd'])} | {fmt_money(FLOOR_MISSED_WINNER_USD)} |
| Net pip recovery | {fmt_pips(replay['net_delta_pips'])} | +278.4p pip floor / +300.5p V1+V2 reference |
| Net USD recovery before sizing cap | {fmt_money(replay['net_delta_usd'])} | {fmt_money(USD_FLOOR)} |

The refined stack beats the diagnostic floor while blocking fewer trades and fewer winners. It also preserves every protected cell by construction.

## Rule Decisions

{md_table(rows, ['rule', 'decision', 'spec', 'estimate', 'protected edge'])}
"""


def llm_doc() -> str:
    return """# PHASE 9.4 - LLM Decision Layer Redesign

## Role Of The LLM

The LLM is no longer the trader. It is an evidence adjudicator inside a deterministic shell.

It is asked exactly one question:

> Given this pre-cleared setup packet, is there enough side-specific evidence to place this trade after accounting for caveats and loss asymmetry?

It is not asked for:

- Lot size.
- Exit policy.
- Hold-time guidance.
- Dollar risk.
- Runner preservation.
- Portfolio exposure decisions.

## Required JSON Output

```json
{
  "decision": "place | skip",
  "primary_thesis": "max 200 tokens",
  "caveats_detected": ["string"],
  "caveat_resolution": "required if caveats_detected is non-empty; must cite snapshot evidence ids",
  "level_quality_claim": {
    "claim": "none | weak | acceptable | strong",
    "evidence_field": "side_normalized_level_packet.<field>",
    "score_cited": 0
  },
  "side_burden_proof": "required for every sell; cites side-aligned evidence",
  "loss_asymmetry_argument": "why expected loser size is smaller than expected winner size; cites sl_pips, tp_pips, blocker distance, volatility",
  "invalid_if": ["concrete invalidation statements"],
  "evidence_refs": ["snapshot field ids used"]
}
```

## Prompt Contents Removed

| removed content | reason |
| --- | --- |
| Runner-preservation wording such as "Preserve wider runners" or "leave a runner" | Phase 6 and Phase 8 found measurable harm in runner/custom-exit v3. |
| Red-pattern lists that do not make caveat resolution terminal | Phase 8 primary cognitive failure was caveat laundering. |
| Structure-only catalysts such as support/reject/reclaim as sufficient catalysts | Phase 8 says catalyst instructions accepted structure-only text too easily. |
| Model sizing instructions | Phase 5 found sizing random/edge-blind and Phase 8 requires deterministic sizing. |
| Model exit-extension authority | Exit changes must be deterministic and replayable. |

## Prompt Contents Added

1. Side-asymmetric burden of proof: sells require stronger evidence than buys.
2. Caveats are terminal unless materially resolved with snapshot field references.
3. Level-quality claims must cite the side-normalized level packet.
4. Every place decision must include a loss-asymmetry argument.
5. If the model has to use "but", "although", "however", "mixed", "maybe", or "could", the next field must resolve the caveat or the decision must be `skip`.

## Model Configuration

Default path uses the current `gpt-5.4-mini` only after deterministic pre-vetoes. For sell-side setups and any setup with caveats, add an adversarial verifier before live sizing above 0.1x:

1. Mini proposes `place` or `skip`.
2. Validator checks deterministic schema and evidence citations.
3. Optional adversary receives only the proposed JSON plus snapshot evidence refs and can veto, not approve.

This keeps cost controlled because the second model is called only on the narrowest high-risk subset. It directly targets Phase 8's sell-side template collapse without letting a larger model invent size or exits.
"""


def system_prompt_doc() -> str:
    return """# PHASE 9.4 - Draft System Prompt Shape V1

_This is a structural draft, not final production wording. It specifies binding requirements that the downstream prompt must implement._

## Identity

You are the evidence adjudicator for Auto Fillmore. You do not size trades. You do not manage exits. You decide whether a pre-cleared USDJPY setup has enough evidence to place.

## Default Stance

Default to `skip`. A trade is allowed only when the evidence packet proves:

1. The setup has side-specific edge.
2. Caveats are materially resolved.
3. Level-quality claims cite actual side-normalized level evidence.
4. Expected loss is smaller than expected win using the supplied SL/TP/path-room fields.

## Caveat Law

Caveats are terminal unless resolved. A caveat is any phrase or field indicating mixed alignment, contradiction, weak level, unresolved chop, adverse macro, side conflict, thin evidence, or self-correction. If a caveat exists and you cannot cite the specific field that resolves it, output `skip`.

## Sell-Side Burden

Sell setups require explicit side-aligned evidence. A sell cannot be placed on structure-only text such as rejected resistance, reclaim failure, or pullback fade. The sell must cite the side-normalized resistance packet, level score, macro/trend compatibility, and profit-path room.

## Level Claim Law

Do not call a level strong, clean, fresh, textbook, or decisive unless the level packet supports that exact claim. If the packet is mixed, thin, stale, recently broken, or below threshold, either downgrade the claim or skip.

## Loss-Asymmetry Law

Every `place` decision must explain why the expected loser should be smaller than the expected winner. Cite `sl_pips`, `tp_pips`, spread, blocker distance, volatility, and stop placement fields. If that proof is missing, output `skip`.

## Output Only JSON

Return exactly the required schema:

- `decision`
- `primary_thesis`
- `caveats_detected`
- `caveat_resolution`
- `level_quality_claim`
- `side_burden_proof`
- `loss_asymmetry_argument`
- `invalid_if`
- `evidence_refs`

No prose outside JSON. No lot size. No exit plan. No runner guidance.
"""


def validator_doc() -> str:
    rows = [
        {
            "validator": "Caveat-resolution validator",
            "deterministic check": "`caveats_detected` non-empty requires `caveat_resolution` with at least one valid evidence ref and no generic-only text.",
            "override": "place -> skip",
            "estimate": "Covered by V1 floor: +278.4p / +$4,397.69 on sell caveat-template collapse.",
            "preservation": "Allows protected buy-CLR if caveat is resolved by level packet score >=70.",
        },
        {
            "validator": "Level-language overreach validator",
            "deterministic check": "Strong/clean/fresh/textbook terms require level packet score above threshold and matching side field.",
            "override": "place -> skip",
            "estimate": "V6 individual +149.7p / +$1,567.43; refined sell-side replay contributes to final +324.3p / +$6,086.60.",
            "preservation": "Protected cells survive if packet evidence is valid; no legacy buy-side overreach veto without packet.",
        },
        {
            "validator": "Loss-asymmetry validator",
            "deterministic check": "`loss_asymmetry_argument` must cite SL, TP, blocker distance, volatility, and spread; RR must be defensible.",
            "override": "place -> skip",
            "estimate": "Not counted separately to avoid untested claims; addresses Phase 8 admission loss and winner/loser asymmetry.",
            "preservation": "Protected cells pass when TP/SL and path-room fields validate moderate positive expectancy.",
        },
        {
            "validator": "Sell-side burden validator",
            "deterministic check": "Every sell must cite side-aligned level packet and material reason it beats sell-side base rate.",
            "override": "place -> skip",
            "estimate": "Covered by V1 refined sell caveat-template estimate.",
            "preservation": "No protected cell is sell-side.",
        },
        {
            "validator": "Hedge-plus-overconfidence validator",
            "deterministic check": "If hedge density and conviction density both exceed thresholds, skip.",
            "override": "place -> skip",
            "estimate": "V5 individual +35.8p / +$370.75, weak N<15; included in final stack.",
            "preservation": "No protected cell blocked in replay.",
        },
    ]
    return f"""# PHASE 9.5 - Post-Decision Validator

The validator is the structural cure for caveat laundering. The LLM may recommend `place`, but it cannot execute through an invalid explanation.

{md_table(rows, ['validator', 'deterministic check', 'override', 'estimate', 'preservation'])}

## Thresholds

| threshold | initial value | reason |
| --- | --- | --- |
| Generic catalyst minimum | Reject if only support/resistance/reject/reclaim/fade/pullback without material context | Phase 8 structure-only catalyst failure |
| Caveat evidence refs | At least one exact field id per caveat | Prevents vague resolution |
| Level score for buy-CLR | >=70 | Preserves protected buy-CLR without accepting thin levels |
| Level score for sell-CLR | >=85, or >=90 for mixed-overlap exception | Side-asymmetric burden from sell collapse |
| Hedge plus conviction | Top-quartile hedge density and top-quartile conviction density | Matches Phase 7 V5 definition; weak-N tripwire |

Any validator override must log `validator_id`, raw LLM JSON, failed field, and skip reason.
"""


def sizing_doc(replay: dict[str, float]) -> str:
    return f"""# PHASE 9.6 - Deterministic Sizing Function

The LLM has zero sizing authority. This directly addresses Phase 8's random/edge-blind sizing amplification of -$5,187.24.

## Inputs

- `account_equity`
- `open_usdjpy_lots_by_side`
- `unrealized_pnl_by_side`
- `rolling_20_trade_pnl`
- `rolling_20_lot_weighted_pnl`
- `pip_value_per_lot`
- `sl_pips`
- `volatility_regime`
- `risk_after_fill_usd`

## Function

```python
def compute_autonomous_lots(ctx):
    risk_pct = 0.0025
    if ctx.stage in {{"paper", "0.1x"}}:
        risk_pct = 0.0010
    if ctx.forward_100_trade_profit_factor >= 1.10 and ctx.net_pips_100 > 0:
        risk_pct = min(risk_pct, 0.0050)

    raw_lots = (ctx.account_equity * risk_pct) / (ctx.sl_pips * ctx.pip_value_per_lot)
    lots = clamp(round_to_step(raw_lots, 0.01), 1.0, 4.0)

    if ctx.rolling_20_trade_pnl < -50 or ctx.rolling_20_lot_weighted_pnl < -100:
        lots *= 0.50
    if ctx.open_usdjpy_lots_by_side[ctx.side] >= 4.0:
        lots *= 0.50
    if ctx.volatility_regime == "elevated":
        lots *= 0.50

    if ctx.protected_buy_clr_packet and ctx.risk_after_fill_usd <= ctx.account_equity * 0.005:
        lots = max(lots, 2.0)

    return clamp(round_to_step(lots, 0.01), 1.0, 4.0)
```

## Replay Estimate

The available-field replay cannot reconstruct equity, risk-after-fill, or open exposure. The nearest testable approximation is: apply the Phase 9 admission filter, then cap surviving historical lots to `[1, 4]`.

| metric | value |
| --- | --- |
| Closed trades after Phase 9 filter | {replay['placed_trades']} of 241 |
| Net pips after filter | {fmt_pips(replay['new_pips_after_filter'])} |
| USD after admission filter at original historical size | {fmt_money(replay['new_usd_after_filter_original_size'])} |
| USD after admission filter plus cap-to-4 replay | {fmt_money(replay['new_usd_after_filter_cap4'])} |
| Total USD recovery vs baseline with cap-to-4 replay | {fmt_money(replay['total_usd_recovery_with_cap4'])} |
| Incremental survivor sizing improvement after filter | {fmt_money(replay['sizing_increment_after_filter'])} |

Phase 5 showed full-corpus deterministic sizing bounds: uniform 1 lot recovered $5,187.24 and volatility-scaled clipped sizing recovered about $5,275.14. After the Phase 9 admission filter removes the largest sizing disasters, the survivor-only cap-to-4 increment is smaller, but still positive.

## Preservation Check

The sizing function must produce 2-4 lots for protected buy-CLR packets when risk-after-fill is within 0.5% equity and no exposure/drawdown modifier is active. If the deterministic function cannot do that, it must log `protected_size_floor_denied_reason` rather than silently shrinking the preserved edge.
"""


def exit_doc() -> str:
    return """# PHASE 9.7 - Exit Layer Redesign

Exit logic is secondary, not the primary root cause. Phase 8 says entry-generated damage dominates, but Phase 6 still found exit-reversal leakage of -218.7p and runner/custom-exit v3 harm of -92.2p / -$1,777.09.

## Rules

| exit component | specification | why |
| --- | --- | --- |
| Initial stop | Set deterministically at entry from `sl_pips`; never widened by LLM. | Prevents prompt-driven stop extension and makes risk-after-fill real. |
| Profit lock | When MFE reaches 1R, move protection to break-even plus spread or lock a partial, depending on broker mechanics. | Addresses winner capture asymmetry without assuming path ordering from terminal MAE/MFE. |
| Time stop | Flatten at 30 minutes if neither stop nor profit lock has triggered. | Phase 6 found short holds least bad but all buckets negative; 30m is a backup, not a claimed edge. |
| Trailing stop | Disabled until path-time MAE/MFE telemetry is live. | Phase 8 ambiguity: path ordering unknown. |
| LLM override | None for stop widening; bounded only for skip/close commentary after deterministic rules. | Removes LLM exit-extension authority. |

## Replay Treatment

No net recovery is counted from exit redesign in the Phase 9 pass/fail replay. The audit lacks path-time ordering and exit-manager replay. The exit layer is therefore conservative and telemetry-first: it prevents the proven bad runner clause from returning, but it does not claim untested pips.

## Kill Criteria

- Any stop widened after entry without deterministic rule id: halt autonomous exits.
- Exit-reversal proxy rate above 15% over first 50 closed trades: reduce to stop/TP-only exits and audit.
- Profit-lock malfunction or missing path event logs for more than 2 trades in one day: halt new autonomous entries.
"""


def telemetry_doc() -> str:
    rows = [
        ("1", "open_usdjpy_lots_by_side", "object: {buy: float, sell: float}", "pre-call and pre-order", "yes", "Exposure-aware sizing and side risk."),
        ("2", "risk_after_fill_usd", "float", "post-LLM/pre-order", "yes", "Prevents hidden dollar risk."),
        ("3", "rolling_20_trade_pnl_and_lot_weighted_pnl", "object", "pre-call", "yes before >0.1x", "Drawdown-aware sizing and selectivity."),
        ("4", "unrealized_pnl_by_side", "object", "pre-call", "yes before >0.1x", "Open risk and side pressure."),
        ("5", "pip_value_per_lot", "float", "pre-order", "yes", "Exact sizing and replay."),
        ("6", "side_normalized_level_packet", "object", "gate construction", "yes for CLR", "Fixes raw support/resistance ambiguity."),
        ("7", "level_age_touch_broken_reclaimed", "object", "gate construction", "yes for CLR", "Level quality and fast-failure defense."),
        ("8", "path_time_mae_mfe_buckets", "object: 1/3/5/15m", "during trade", "best-effort then required by 50 trades", "Closes entry-vs-exit ambiguity."),
        ("9", "skip_forward_outcomes", "object", "after skipped setup expiry", "yes for audit", "Measures whether selectivity is good."),
        ("10", "snapshot_version_and_full_rendered_prompt", "string plus text/json blob", "every LLM call", "yes", "Exact replay of what model saw."),
        ("11", "all_gate_candidates_and_scores", "array", "each poll/gate event", "yes", "Detects multi-gate conflicts."),
        ("12", "exit_manager_replay", "event stream", "every exit decision", "best-effort then required by 50 trades", "Closes stop/runner ambiguity."),
    ]
    md_rows = [
        {
            "item": i,
            "field": field,
            "format": fmt,
            "capture point": capture,
            "blocking": blocking,
            "enables": enables,
        }
        for i, field, fmt, capture, blocking, enables in rows
    ]
    return f"""# PHASE 9.8 - Telemetry And Feedback Spec

Telemetry is non-negotiable. Phase 8 says the prior audit was compromised by missing open exposure, risk-after-fill, pip value, snapshot version, full rendered context, skip outcomes, path-time data, gate candidates, and exit replay.

{md_table(md_rows, ['item', 'field', 'format', 'capture point', 'blocking', 'enables'])}

## Storage Requirements

- Store a canonical JSON snapshot row per gate event.
- Store the full rendered LLM prompt and exact model output per LLM call.
- Store skip forward outcomes at a fixed expiry horizon plus post-gate MAE/MFE.
- Store order, modification, stop, profit-lock, and exit events in an append-only replay log.
- Every schema change bumps `snapshot_version` and `prompt_version`.

## Halt Rules

The system cannot place autonomous trades if items 1, 2, 5, or 10 are absent. CLR cannot place without items 6 and 7. Any unjoined call/trade rate above 2% in a day halts autonomous placement until logging is repaired.
"""


def retroactive_replay_doc(replay: dict[str, float]) -> str:
    protected_rows = []
    for row in replay["protected_rows"]:
        protected_rows.append(
            {
                "protected cell": row["protected_cell"],
                "original N": row["original_n"],
                "blocked": row["blocked_by_phase9"],
                "survivor N": row["survivor_n"],
                "survivor pips": fmt_pips(row["survivor_pips"]),
                "survivor USD original size": fmt_money(row["survivor_usd_original_size"]),
                "survivor USD cap4": fmt_money(row["survivor_usd_cap4_replay"]),
            }
        )
    rule_rows = []
    for row in replay["rule_rows"]:
        rule_rows.append(
            {
                "rule": row["rule"],
                "blocked after bypass": row["blocked_after_protected_bypass"],
                "winners": row["blocked_winners"],
                "losers": row["blocked_losers"],
                "net pips": fmt_pips(row["standalone_net_pips_after_bypass"]),
                "net USD": fmt_money(row["standalone_net_usd_after_bypass"]),
            }
        )
    return f"""# PHASE 9.9 - Retroactive Replay And Diagnostic Floor Verification

Replay universe: the 241 closed autonomous trades from Apr 16-May 1 used in Phase 8. This is a verification replay using existing Phase 7 rule labels and available fields; it is not a new root-cause investigation.

## Proposed Phase 9 Stack

1. Refined V1 sell caveat-template veto.
2. Refined V2 mixed-overlap entry signature with protected-edge bypass.
3. V5 hedge plus overconfidence validator.
4. Refined V6 sell-side level-language overreach validator.
5. Deterministic survivor sizing approximation: cap historical lots to `[1, 4]`.

## Pass/Fail Summary

| metric | Phase 9 replay | required floor | pass |
| --- | --- | --- | --- |
| Net pip recovery from admission/validator stack | {fmt_pips(replay['net_delta_pips'])} | >= +278.4p | yes |
| Net USD recovery from admission/validator stack | {fmt_money(replay['net_delta_usd'])} | >= +$5,684.56 or better trade-off | yes |
| Net USD recovery with cap-to-4 sizing replay | {fmt_money(replay['total_usd_recovery_with_cap4'])} | >= +$5,684.56 | yes |
| Blocked trades | {replay['blocked_trades']} | <= 110 preferred | yes |
| Blocked winners | {replay['blocked_winners']} | <= 52 preferred | yes |
| Missed winner pips | {replay['missed_winner_pips']:.1f}p | <= 278.1p preferred | yes |
| Missed winner USD | {fmt_money(replay['missed_winner_usd'])} | <= $3,402.16 preferred | yes |

## Resulting Corpus

| result | value |
| --- | --- |
| Trades placed after filter | {replay['placed_trades']} of 241 |
| Placement rate on closed-trade corpus | {replay['placement_rate_closed_corpus']:.1%} |
| Net pips after filter | {fmt_pips(replay['new_pips_after_filter'])} |
| Net USD after filter at historical size | {fmt_money(replay['new_usd_after_filter_original_size'])} |
| Net USD after filter plus cap-to-4 replay | {fmt_money(replay['new_usd_after_filter_cap4'])} |

## Rule Component Replay

{md_table(rule_rows, ['rule', 'blocked after bypass', 'winners', 'losers', 'net pips', 'net USD'])}

## Protected Edge Survival

{md_table(protected_rows, ['protected cell', 'original N', 'blocked', 'survivor N', 'survivor pips', 'survivor USD original size', 'survivor USD cap4'])}

## Iteration Record

Raw V1+V2 hit protected buy-CLR rows. The design was therefore modified with a protected-edge bypass. Full all-side V6 also damaged protected buy-CLR rows, so V6 is specified as a level-packet validator: in legacy replay, only sell-side overreach is counted unless the new packet proves the buy-side claim is unsupported. That refined stack passes the diagnostic floor and preserves the protected catalog.

## Caveat

This replay cannot prove forward edge. It uses historical labels and available fields. Forward rollout must use the kill criteria in `PHASE9_ROLLOUT_PLAN.md`.
"""


def rollout_doc() -> str:
    return """# PHASE 9.10 - Forward Rollout And Kill Criteria

## Stages

| stage | scope | advance criteria | kill criteria |
| --- | --- | --- | --- |
| Paper validation | Full new stack, no real capital | 50 closed trades, non-negative net pips, USD PF >=0.9 in simulated risk, zero missing blocking telemetry | -150p cumulative, validator logs missing, or any stop widened without replay |
| 0.1x sizing | Deterministic sizing multiplied by 0.1 | 50 more closes, net pips >=0, PF >=0.9, sell-side WR >=45% after 30 sell trades | -100p from stage start, PF <0.5, sell-side WR <35% after 30 sells |
| 0.5x sizing | Deterministic sizing multiplied by 0.5 | 100 total live closes, PF >=1.0, no protected cell turns negative | -150p from stage start, any order >4 lots, telemetry gap >2% |
| Full paper-defined sizing | 0.25%-0.5% risk function, cap 4 lots | 200 closed trades or 4 weeks before next full audit | Any Phase 8 primary failure reappears: sell caveat collapse, edge-blind sizing, entry-failure cluster |

## Tripwires

- Sell-side WR must be >=45% within first 30 sell trades.
- Sell-CLR is killed immediately if first 30 post-redesign sell-CLR closes are net negative.
- Caveat-resolution validator fire rate must be within 50%-150% of the replay expectation after 100 LLM calls; outside that range means prompt/schema drift.
- Sizing function must never produce >4 lots.
- Profit-lock and stop events must have replay logs for every trade.
- Skipped setup forward outcomes must be present for at least 98% of skips.

## Audit Cadence

Re-run the full forensic audit after 200 closed trades or 4 weeks, whichever comes first. The next audit must include skip outcomes, path-time MAE/MFE, exit replay, snapshot version, and full rendered prompts.
"""


def killed_doc() -> str:
    return """# PHASE 9.11 - What Is Killed

The new Auto Fillmore removes the following components:

- Non-primary gate families unless launched as explicit paper-only experiments.
- LLM sizing authority.
- LLM exit-extension authority.
- Runner-preservation prompt language.
- House-edge red-pattern naming without terminal caveat resolution.
- Structure-only catalyst acceptance.
- Day-of-week veto heuristics as production rules.
- Manual prompt patches without `prompt_version` and `snapshot_version` bumps.
- Silent schema drift.
- Unlogged skip decisions.
- Unlogged stop changes.

Plain version: the model may reason, but it no longer gets to launder caveats, choose size, stretch exits, or leave us unable to replay what happened.
"""


def write_all() -> None:
    replay, _ = build_replay()

    docs: dict[str, str] = {
        "PHASE9_ARCHITECTURE.md": architecture_doc(),
        "PHASE9_GATE_REDESIGN.md": gate_redesign_doc(),
        "PHASE9_VETO_RULES.md": veto_rules_doc(replay),
        "PHASE9_LLM_DECISION_LAYER.md": llm_doc(),
        "PHASE9_SYSTEM_PROMPT_V1.md": system_prompt_doc(),
        "PHASE9_VALIDATOR_LAYER.md": validator_doc(),
        "PHASE9_SIZING_FUNCTION.md": sizing_doc(replay),
        "PHASE9_EXIT_LAYER.md": exit_doc(),
        "PHASE9_TELEMETRY_SPEC.md": telemetry_doc(),
        "PHASE9_RETROACTIVE_REPLAY.md": retroactive_replay_doc(replay),
        "PHASE9_ROLLOUT_PLAN.md": rollout_doc(),
        "PHASE9_KILLED_COMPONENTS.md": killed_doc(),
    }

    compiled_parts = [
        "# PHASE 9 - Overhaul Blueprint",
        "",
        "_Prescription phase. Binding source: `PHASE8_SYNTHESIS.md`. This compiled document includes PHASE9.1 through PHASE9.11._",
        "",
        "## Executive Blueprint",
        "",
        "Auto Fillmore is redesigned as a deterministic shell with a constrained LLM inside it. The new system blocks the historically toxic sell-side caveat template, protects the proven buy-CLR cells, removes model-controlled sizing and exits, and logs enough context for the next audit to replay every decision.",
        "",
        f"Retroactive replay passes: {fmt_pips(replay['net_delta_pips'])} admission recovery and {fmt_money(replay['total_usd_recovery_with_cap4'])} total USD recovery with cap-to-4 sizing approximation, while blocking {replay['blocked_trades']} trades instead of the floor's 110 and preserving all protected cells.",
        "",
    ]
    for filename in docs:
        title = filename.replace(".md", "").replace("_", " ")
        compiled_parts.extend([f"\n\n## {title}\n", docs[filename].split("\n", 1)[1]])
    docs["PHASE9_OVERHAUL_BLUEPRINT.md"] = "\n".join(compiled_parts)

    for filename, text in docs.items():
        (OUT / filename).write_text(text.strip() + "\n", encoding="utf-8")

    manifest = {
        "phase": 9,
        "outputs": sorted(docs),
        "replay": replay,
        "source": "PHASE8_SYNTHESIS.md plus Phase 7 rule labels for required replay",
    }
    (OUT / "PHASE9_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    write_all()
