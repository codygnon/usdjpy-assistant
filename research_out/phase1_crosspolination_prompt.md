# Phase 1: Cross-Pollination Backtests — V14 Tokyo & V44 London+NY

## Objective
Run 6 experimental backtest variants testing feature cross-pollination between two USDJPY strategies. Each variant modifies ONE feature in isolation to measure its independent impact vs baseline. Produce a comparison table at the end.

## Codebase & File Locations
- **V14 engine**: `scripts/backtest_tokyo_meanrev.py`
- **V14 config**: `research_out/tokyo_optimized_v14_config.json` (strategy_id: `tokyo_mean_reversion_v7`)
- **V44 engine**: `scripts/backtest_session_momentum.py` (invoked with `--version v5`)
- **V44 config reference**: Extract the full parameter set from `research_out/v44_combined_winners_500k.json` — the config/parameters are embedded in the output JSON. Find how V44 was invoked by searching for experiment runner scripts (look for files matching `run_v44*`, `run_*session*`, `run_*momentum*` in `scripts/`), or reconstruct the CLI args from the embedded config. The V44 engine uses `--version v5` plus many `--v5-*` CLI arguments.
- **Primary test dataset**: `research_out/USDJPY_M1_OANDA_500k.csv`
- **Validation dataset** (run winners only): `research_out/USDJPY_M1_OANDA_1000k.csv`

## CRITICAL: Do NOT Break Originals
- Before modifying any engine file, make a backup copy (e.g., `backtest_tokyo_meanrev_BACKUP.py`)
- All new features must be gated behind config flags / CLI args that default to OFF/disabled
- When a flag is OFF, behavior must be IDENTICAL to the unmodified engine
- This ensures original baselines can always be reproduced

## Baseline Reference (500k dataset)

### V14 Baseline (500k)
| Metric | Value |
|---|---|
| Total Trades | 60 |
| Win Rate | 83.33% |
| Profit Factor | 3.19 |
| Net USD | +$6,818.91 |
| Net Pips | +208.58 |
| Max Drawdown | $1,400.40 (1.34%) |
| Sharpe | 3.54 |
| Avg Win (pips) | 6.02 |
| Avg Loss (pips) | -9.23 |

### V44 Baseline (500k — `v44_combined_winners_500k.json`)
| Metric | Value |
|---|---|
| Total Trades | 257 |
| Win Rate | 40.86% |
| Profit Factor | 1.58 |
| Net USD | +$30,529.27 |
| Net Pips | +664.98 |
| Max Drawdown | $6,942.43 |
| Avg Win (pips) | 17.19 |
| Avg Loss (pips) | -7.50 |

---

## Test V14-A: Signal-Strength-Based Position Sizing

### Rationale
V14 already computes a signal strength score (1-10 scale, tiers: weak 1-3, moderate 4-6, strong 7+) via the `signal_strength_tracking` config section, but currently does NOT use it for sizing or filtering (`filter_on_it: false`). Meanwhile, V44's risk parity sizing (which scales by conviction) is a key profit driver. This test applies strength-based multipliers to V14's position sizing.

### Modification
In `scripts/backtest_tokyo_meanrev.py`, find the position sizing calculation (search for `units_formula` or where `units = floor(...)` is computed — it's in the section that calculates units from equity, risk_pct, sl_pips, and pip_size).

Add a new config section and logic:
```json
"signal_strength_sizing": {
    "enabled": true,
    "weak_mult": 0.5,
    "moderate_mult": 1.0,
    "strong_mult": 1.4
}
```

After computing the base `units`, apply a multiplier based on the signal's strength tier:
- Strength score 1-3 (weak): multiply units by `weak_mult` (0.5)
- Strength score 4-6 (moderate): multiply units by `moderate_mult` (1.0)
- Strength score 7+ (strong): multiply units by `strong_mult` (1.4)

The signal strength score is already computed per trade — find where it's calculated (search for `signal_strength` or the scoring components: `confluence_count`, `bb_penetration_bonus`, `rsi_extreme_bonus`, `deep_pivot_zone`, `same_candle_sar_flip`, `favorable_hour`) and pass it to the sizing function.

### Config
Copy `research_out/tokyo_optimized_v14_config.json` to `research_out/phase1_v14a_config.json`. Add the `signal_strength_sizing` section above. Keep everything else identical.

### Run
```bash
python3 scripts/backtest_tokyo_meanrev.py --config research_out/phase1_v14a_config.json
```
Modify the run_sequence output paths to write to `research_out/phase1_v14a_500k_report.json` (and corresponding trades/equity CSVs). Run only the 500k entry.

### Output
Save report to `research_out/phase1_v14a_500k_report.json`

---

## Test V14-B: Session Entry Cutoff

### Rationale
V14 has no entry cutoff — it can open trades minutes before session end (22:00 UTC), which then get immediately force-closed. In the 1000k results, 36 trades exited via `session_close` for net -$4,376. V44 blocks entries in the last 60 minutes of each session, preventing this. A cutoff should eliminate these wasted late entries.

### Modification
In `scripts/backtest_tokyo_meanrev.py`, find the session check logic (search for `in_tokyo_session` or the session time check where `session_start_utc: "16:00"` and `session_end_utc: "22:00"` are used).

Add a new config parameter:
```json
"session_entry_cutoff_minutes": 90
```

Before generating any entry signal, add a check: if the current bar time is within `session_entry_cutoff_minutes` of session end (i.e., current time >= 20:30 UTC when cutoff=90), block the entry. This does NOT affect open trade management — existing positions still trail/close normally. Only NEW entries are blocked.

### Config
Copy `research_out/tokyo_optimized_v14_config.json` to `research_out/phase1_v14b_config.json`. Add `"session_entry_cutoff_minutes": 90` to the `session_filter` section. Keep everything else identical.

### Run
```bash
python3 scripts/backtest_tokyo_meanrev.py --config research_out/phase1_v14b_config.json
```
Output to `research_out/phase1_v14b_500k_report.json`.

---

## Test V14-C: ATR Percentile Filter (Replacing Flat Cap)

### Rationale
V14 uses a flat ATR cap (`atr.max_threshold_price_units: 0.3` = 30 pips). This is static and doesn't adapt to changing market conditions. V44 uses a rolling percentile rank (blocking the top 33%), which auto-adjusts. A percentile approach would be more robust across different volatility regimes.

### Modification
In `scripts/backtest_tokyo_meanrev.py`, find where the ATR filter is applied (search for `atr_max` or `require_atr_max` or `max_threshold_price_units` — it's in the entry pre-conditions section).

Add new config parameters:
```json
"atr": {
    "timeframe": "M15",
    "period": 14,
    "percentile_filter": {
        "enabled": true,
        "lookback_bars": 150,
        "max_percentile": 0.67
    },
    "max_threshold_price_units": 0.3
}
```

When `percentile_filter.enabled` is true:
1. On the current M15 bar, get the current ATR(14) value
2. Look back `lookback_bars` M15 bars and collect all ATR values
3. Compute percentile rank: `rank = count(historical_atr <= current_atr) / len(historical_values)`
4. If `rank > max_percentile` (0.67), block the entry
5. IGNORE the flat `max_threshold_price_units` when percentile filter is enabled

### Config
Copy config to `research_out/phase1_v14c_config.json`. Enable the percentile filter as shown above.

### Run
Output to `research_out/phase1_v14c_500k_report.json`.

---

## Test V44-A: Queued Entry Confirmation

### Rationale
V14 queues entry signals and requires a subsequent M1 bar to close in the trade direction within 12 bars before entering — only 14.9% of signals get confirmed. V44 enters immediately when pullback conditions are met (with just a body-size check on the current bar). Adding a confirmation queue would filter out signals where momentum immediately stalls, potentially cutting false entries.

### Modification
In `scripts/backtest_session_momentum.py`, find the pullback entry logic inside `run_backtest_v5()`. Search for where `pullback_trigger` is set to True and the trade is opened.

Add a new CLI arg:
```
--v5-queued-confirm-bars 6
```
(default: 0, meaning disabled / immediate entry as current behavior)

When `v5_queued_confirm_bars > 0`:
1. When a pullback trigger fires, do NOT immediately open a trade. Instead, record a pending signal with: timestamp, side (buy/sell), entry conditions snapshot, SL price, strength, and a `confirm_deadline = current_bar_index + v5_queued_confirm_bars`
2. On each subsequent M1 bar (while signal is pending):
   a. Check if a confirmation bar appears: for BUY, a bar where `close > open` (bullish close); for SELL, a bar where `close < open` (bearish close). The confirmation bar must also have `body >= v5_entry_min_body_pips` (already a parameter, 1.0 pip)
   b. If confirmed: open the trade at the confirmation bar's close price, with the original SL (recalculated if needed to maintain floor/cap constraints from the confirmation bar)
   c. If `current_bar_index > confirm_deadline`: discard the signal (expired)
   d. If the opposite side triggers while a signal is pending: discard the old signal
3. Only ONE pending signal at a time per session
4. Track confirmation stats: signals generated, confirmed, expired, avg confirmation delay

### Config / Invocation
Reconstruct the V44 baseline invocation and add `--v5-queued-confirm-bars 6`. Run on 500k dataset.

### Output
Save to `research_out/phase1_v44a_500k_report.json`

---

## Test V44-B: Rolling Range Exhaustion Gate

### Rationale
V14 monitors a 60-minute rolling price range and blocks entries when it exceeds 40 pips (with 15-min cooldown). This prevents entering after the move is already exhausted. V44 has no equivalent — it can enter pullbacks during extended one-way moves where reversal risk is high. In V44's 1000k data, 290 SL exits cost -$57,213; some were likely from chasing exhausted trends.

### Modification
In `scripts/backtest_session_momentum.py`, inside `run_backtest_v5()`, add rolling range tracking.

Add new CLI args:
```
--v5-exhaustion-gate-enabled true
--v5-exhaustion-gate-window-minutes 60
--v5-exhaustion-gate-max-range-pips 40.0
--v5-exhaustion-gate-cooldown-minutes 15
```
(defaults: enabled=false, preserving current behavior)

When enabled:
1. Maintain a rolling window of M1 bar high/low values for the last `window_minutes` (60) minutes within the current session. Reset at session start.
2. On each bar: compute `rolling_range = (max(highs) - min(lows)) / PIP_SIZE`
3. If `rolling_range > max_range_pips` (40.0):
   a. Block all new entries
   b. Set `exhaustion_cooldown_until = current_time + cooldown_minutes`
4. If currently in cooldown (`current_time < exhaustion_cooldown_until`): block entries
5. After cooldown expires: resume normal entry logic
6. Track stats: how many entries were blocked by exhaustion gate

This does NOT affect open trade management — only blocks new entries.

### Config / Invocation
Add the four CLI args above to the V44 baseline invocation. Run on 500k.

### Output
Save to `research_out/phase1_v44b_500k_report.json`

---

## Test V44-C: ADX Minimum for Trend Confirmation

### Rationale
V14 uses ADX(14) on M15 with a MAXIMUM threshold (≤35) to confirm ranging conditions. V44 could use the INVERSE — a MINIMUM ADX threshold to confirm that a genuine trend exists before entering pullbacks. This would block entries in flat/directionless markets where EMA alignments are noise. Currently V44 relies solely on EMA slope for strength — ADX provides independent trend confirmation.

### Modification
In `scripts/backtest_session_momentum.py`, inside `run_backtest_v5()`:

1. First, check if ADX is already computed on any timeframe. If not, compute ADX(14) on M15 data. The ADX computation logic already exists in the V14 engine (`backtest_tokyo_meanrev.py`) — you can reference or copy that implementation. ADX is computed from:
   - +DM, -DM (directional movement)
   - TR (true range)
   - Smoothed +DI, -DI (14-period)
   - DX = abs(+DI - -DI) / (+DI + -DI) * 100
   - ADX = 14-period rolling average of DX

2. Add new CLI args:
```
--v5-adx-filter-enabled true
--v5-adx-min 20.0
--v5-adx-timeframe M15
--v5-adx-period 14
```
(defaults: enabled=false)

3. Before the pullback entry is taken (after all existing checks pass), add: if `adx_filter_enabled` and `current_adx < adx_min`: block entry with reason `"v5_adx_too_low"`.

4. Track stats: entries blocked by ADX filter, average ADX at entry vs at blocked

### Config / Invocation
Add the ADX CLI args to the V44 baseline invocation. Run on 500k.

### Output
Save to `research_out/phase1_v44c_500k_report.json`

---

## Execution Order

Run all 6 variants sequentially:
1. V14-A (signal strength sizing)
2. V14-B (session entry cutoff)
3. V14-C (ATR percentile filter)
4. V44-A (queued entry confirmation)
5. V44-B (rolling range exhaustion gate)
6. V44-C (ADX minimum filter)

All on the **500k dataset** (`research_out/USDJPY_M1_OANDA_500k.csv`).

Before running any variant, verify the baseline by running the unmodified engine with the original config and confirming the output matches the baseline numbers listed above (or is very close). This ensures your code changes haven't broken anything.

## Final Comparison Output

After all 6 variants complete, create `research_out/phase1_comparison.json` with the following structure:

```json
{
    "generated_at": "<timestamp>",
    "dataset": "USDJPY_M1_OANDA_500k.csv",
    "baselines": {
        "v14": { "trades": 60, "win_rate": 83.33, "pf": 3.19, "net_usd": 6818.91, "net_pips": 208.58, "max_dd_usd": 1400.40, "sharpe": 3.54, "avg_win_pips": 6.02, "avg_loss_pips": -9.23 },
        "v44": { "trades": 257, "win_rate": 40.86, "pf": 1.58, "net_usd": 30529.27, "net_pips": 664.98, "max_dd_usd": 6942.43, "avg_win_pips": 17.19, "avg_loss_pips": -7.50 }
    },
    "variants": {
        "v14a_strength_sizing":     { "trades": null, "win_rate": null, "pf": null, "net_usd": null, "net_pips": null, "max_dd_usd": null, "sharpe": null, "avg_win_pips": null, "avg_loss_pips": null, "change_vs_baseline": { "net_usd_delta": null, "pf_delta": null, "dd_delta": null } },
        "v14b_entry_cutoff":        { "...same fields..." },
        "v14c_atr_percentile":      { "...same fields..." },
        "v44a_queued_confirm":      { "...same fields..." },
        "v44b_exhaustion_gate":     { "...same fields..." },
        "v44c_adx_minimum":         { "...same fields..." }
    },
    "recommendation": "<Which variants improved their respective baselines and should advance to 1000k validation>"
}
```

Also output a human-readable summary table to the console at the end showing all 6 variants vs their baselines, with delta columns for net_usd, profit_factor, max_drawdown, and win_rate.

## Important Notes
- Each variant tests ONE change in isolation. Do not combine features within a single variant.
- If a variant worsens ALL key metrics (PF, net USD, drawdown), note it as "negative result" — this is still valuable data.
- If you encounter errors running the V44 engine, check for required Python dependencies (numpy, pandas at minimum). The V14 engine has the same dependencies.
- The V14 engine config uses `run_sequence` with multiple dataset entries — only run the entry pointing to the 500k CSV.
- Keep detailed notes on any implementation decisions you make (e.g., "confirmation bar must have body >= 1.0 pip to count").
