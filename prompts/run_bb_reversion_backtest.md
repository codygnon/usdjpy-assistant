# Run backtest: Bollinger Band reversion (2σ, M5 confirm)

Use this prompt when you want the agent to run the BB reversion sweep backtest.

---

## Prompt for agent

Run the Bollinger Band reversion sweep backtest for USDJPY session momentum.

**What it does:** Backtests the v39 winner config with BB reversion overlay enabled:
- **2-standard-deviation** band touch (not 3σ)
- **Confirm on next M5 bar:** entry only when the next M5 bar closes back inside the band (buy: close above lower band; sell: close below upper band)
- TP at midline; no trend filter
- Grid over SL pips `{6, 8, 10, 12}` and min band width pips `{6, 10, 15}`

**How to run:**
1. From the project root `usdjpy_assistant`, run:
   ```bash
   ./scripts/run_v39_bb_reversion_sweep.sh
   ```
   Or with explicit env if needed:
   ```bash
   ROOT_DIR="$(pwd)" OUT_DIR="/tmp" ./scripts/run_v39_bb_reversion_sweep.sh
   ```
2. Ensure the venv is active or `ROOT_DIR` points to the repo so `PYTHON_BIN` resolves (default: `$ROOT_DIR/.venv/bin/python`).
3. Required inputs (script will exit with an error if missing):
   - Config: `research_out/session_momentum_v39_winner_config.json`
   - Data: `research_out/USDJPY_M1_OANDA_500k.csv`
4. Outputs go to `OUT_DIR` (default `/tmp`): for each (SL, MW) a JSON result and a log. At the end the script prints a summary table of BB-reversion trades (count, WR%, net USD, etc.) and marks profitable configs with `<<<`. It also checks that the base v39 result is unchanged (239 trades, ~$40,311).

**Optional:** To save outputs under the project, run:
```bash
OUT_DIR="$(pwd)/research_out/bb_rev_2std_confirm" ./scripts/run_v39_bb_reversion_sweep.sh
```
Create the directory first if you want: `mkdir -p research_out/bb_rev_2std_confirm`.

---

## Summary of BB reversion settings (in the script)

| Setting            | Value   | Meaning                                      |
|--------------------|---------|----------------------------------------------|
| `--bb-rev-std`     | 2.0     | 2σ Bollinger bands                           |
| `--bb-rev-confirm` | true    | Wait for next M5 bar to close back inside    |
| `--bb-rev-tp-midline` | true | Take profit at band midline                  |
| `--bb-rev-require-trend` | false | No trend filter for BB entries           |

SL and min width are swept; other BB defaults (period 20, session 6–17, etc.) come from `backtest_session_momentum.py`.
