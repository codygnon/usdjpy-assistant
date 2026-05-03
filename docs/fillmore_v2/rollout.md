# Auto Fillmore v2 — Staged Rollout

Implements PHASE9.10 verbatim, with explicit build triggers for components
deferred from v1 (notably the adversarial verifier).

## Stages

| Stage | Scope | Advance criteria | Kill criteria |
| --- | --- | --- | --- |
| **Paper validation** | Full new stack, no real capital | 50 closed trades, non-negative net pips, simulated USD PF ≥ 0.9, zero missing blocking telemetry | -150p cumulative, missing validator logs, any stop widened without replay |
| **0.1x sizing** | risk_pct = 0.0010 | 50 more closes, net pips ≥ 0, PF ≥ 0.9, sell-side WR ≥ 45% after 30 sell trades | -100p from stage start, PF < 0.5, sell-side WR < 35% after 30 sells |
| **0.5x sizing** | Deterministic sizing × 0.5 | 100 total live closes, PF ≥ 1.0, no protected cell turns negative | -150p from stage start, any order > 4 lots, telemetry gap > 2% |
| **Full sizing** | 0.25%–0.5% risk, cap 4 lots | 200 closed trades or 4 weeks before next full audit | Any Phase 8 primary failure reappears |

## Always-on tripwires

- Sell-side WR ≥ 45% within first 30 sell trades.
- Sell-CLR killed immediately if first 30 post-redesign sell-CLR closes net negative.
- Caveat-resolution validator fire rate within 50%–150% of replay expectation after 100 LLM calls.
- Sizing function never produces > 4 lots.
- Profit-lock and stop events have replay logs for every trade.
- Skipped setup forward outcomes present for ≥ 98% of skips.

## Deferred components: explicit build triggers

These were deferred from v1 to minimize surface area. Each is built **only
if** its trigger fires.

### Adversarial verifier (deferred from Step 6)

PHASE9.4 specifies an optional second model that can veto (not approve)
sell-side and caveat-bearing setups. Build triggers — any one is sufficient:

1. **Sell-side WR < 45% after 30 sell trades in Stage 2.** The mini's sell
   adjudication isn't holding; verifier targets the same failure mode.
2. **Caveat-resolution validator fires outside 50–150% of Phase 9 replay
   expectation after 100 LLM calls.** Indicates prompt or schema drift the
   structured validator alone can't catch; an adversary trained on caveat
   evasion adds a second line.
3. **Any protected cell (CLR×buy×Phase 2 zone-memory×caveat,
   Tuesday×CLR×buy, the 2–3.99 lot sub-cell) turns negative in forward
   sample of ≥10 trades.** The current sizing/veto preservation isn't
   sufficient; add adversarial review on packets matching protected
   patterns to reduce false negatives.
4. **Any Phase 8 primary failure reappears** (sell caveat collapse,
   edge-blind sizing, entry-failure cluster). Forensic audit hook —
   adversary becomes a holding pattern while the primary loop is investigated.

### Path-time MAE/MFE buckets (deferred from Step 1)

Captured as best-effort (all-None) until 50 forward trades have closed.
Trigger to make blocking: at trade #50, audit how many trades captured
1/3/5/15-min buckets. If <90%, halt trailing-stop work; investigate the
exit-management loop's polling fidelity before proceeding.

### Trailing stops (deferred from Step 7)

Disabled until path-time telemetry above is reliable for ≥50 closes. Trigger
to enable: ≥90% path-time coverage on first 50 trades AND exit-reversal
proxy rate < 10%.

## Audit cadence

Re-run the full forensic audit after 200 closed trades or 4 weeks, whichever
comes first. The next audit must include: skip outcomes, path-time MAE/MFE,
exit replay, snapshot version, and full rendered prompts. v2's telemetry
layer is built to satisfy this; if any of these fields is missing for >2%
of rows in the audit window, the audit cannot proceed and the system halts
new placements.
