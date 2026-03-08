# Phase 2: Merged V14 + V44 Backtesting Engine

## Objective
Build a single backtesting script (`scripts/backtest_merged_v14_v44.py`) that runs BOTH the V14 Tokyo Mean Reversion strategy and the V44 London+NY Trend Pullback strategy on the same M1 USDJPY dataset, sharing a **single equity curve**. The strategies trade in **completely non-overlapping sessions**, so there are never simultaneous positions from both strategies.

## Session Windows (UTC)
```
08:30 - 11:00  →  V44 London
11:00 - 13:00  →  Dead zone (no trading)
13:00 - 16:00  →  V44 NY Overlap
16:00 - 22:00  →  V14 Tokyo
22:00 - 08:30  →  Dead zone (no trading)
```
There is zero overlap. At any given time, at most ONE strategy is active.

## Reference Files
Read these files to understand and replicate each strategy's logic:

| File | Purpose |
|------|---------|
| `scripts/backtest_tokyo_meanrev.py` | V14 engine (3602 lines) — replicate entry/exit/sizing logic |
| `research_out/tokyo_optimized_v14_config.json` | V14 configuration (all parameters) |
| `scripts/backtest_session_momentum.py` | V44 engine (6950 lines) — replicate `run_backtest_v5()` logic |
| `research_out/v44_combined_winners_500k.json` | V44 config (embedded in `"config"` key) + baseline results |
| `research_out/v5_scheduled_events_utc.csv` | V44 news calendar (required for news filter + news trend entries) |

## CRITICAL: Do NOT Break Originals
- Do NOT modify `backtest_tokyo_meanrev.py` or `backtest_session_momentum.py`
- Create `scripts/backtest_merged_v14_v44.py` as a **new, standalone file**
- You may copy helper functions from both engines (load_m1, resample_ohlc, rolling_rsi, rolling_atr, rolling_adx, compute_psar, compute_spread_pips, etc.)

---

## Architecture

### Invocation
```bash
python scripts/backtest_merged_v14_v44.py \
  --v14-config research_out/tokyo_optimized_v14_config.json \
  --v44-config research_out/v44_combined_winners_500k.json \
  --input research_out/USDJPY_M1_OANDA_500k.csv \
  --output research_out/phase2_merged_500k_report.json \
  --starting-equity 100000
```

The `--v44-config` file is a V44 output JSON — extract the `"config"` key from it to get V44's parameters. The `--v14-config` file is a direct config JSON.

### Data Flow
```
1. Load M1 CSV → pandas DataFrame (time, open, high, low, close columns)
2. Resample to M5, M15, H1 (shared by both strategies)
3. Compute V14 indicators on the shared timeframes
4. Compute V44 indicators on the shared timeframes
5. Merge all indicators onto M1 via merge_asof(direction="backward")
6. Run single M1 bar loop with session routing
7. Generate combined report + per-strategy breakdowns
```

### Main Bar Loop Flow (per M1 bar)
```
for each M1 bar:
    1. Compute spread (use V44's realistic model — see spread section)
    2. Derive bid/ask from mid price and spread
    3. Classify session: "tokyo" | "london" | "ny_overlap" | "dead"

    4. MANAGE OPEN POSITIONS (both strategies):
       - For each V14 position: apply V14 exit rules
       - For each V44 position: apply V44 exit rules
       - Update shared equity on any close

    5. FORCE CLOSE stale positions:
       - If NOT in Tokyo session: close all V14 positions ("session_close")
       - If NOT in London/NY session: close all V44 positions ("session_close")

    6. ENTRY LOGIC (mutually exclusive by session):
       - If Tokyo session + V14 allowed trading day: run V14 entry logic
       - If London session: run V44 London entry logic
       - If NY Overlap session: run V44 NY entry logic

    7. Record equity curve point (on every trade close)
```

---

## V14 Logic to Replicate

Reference: `scripts/backtest_tokyo_meanrev.py`

### V14 Indicators (computed upfront)
Replicate `add_indicators()` (line 232-409):
- **M5 Bollinger Bands**: BB(25, 2.2) on M5 close — mid, upper, lower, width, regime
- **M5 BB Width Regime**: Rolling 80th percentile of bb_width over 100 M5 bars → "ranging"/"trending"
- **M5 RSI(14)**: Standard RSI on M5 close
- **M15 ATR(14)**: True Range → rolling mean, in price units
- **M15 ADX(14)**: +DI, -DI, DX, ADX
- **M1 Parabolic SAR**: Iterative PSAR (af_start=0.02, step=0.02, max=0.2) → sar_value, sar_direction, sar_flip flags
- **Fibonacci Pivot Points**: Daily (NY close boundary 22:00 UTC) — P, R1, R2, R3, S1, S2, S3 using Fibonacci ratios (0.382, 0.618, 1.000)

All indicators merged onto M1 using `merge_asof(direction="backward")`.

### V14 Session Rules
- Session window: 16:00 - 22:00 UTC
- Allowed trading days: Tuesday, Wednesday, Friday only
- Force close all positions at session end (22:00 UTC)
- Max 4 trades per session
- Stop after 3 consecutive losses in a session
- Session loss stop: 1.5% of equity
- Min 10 minutes between entries
- No re-entry same direction after stop for 30 minutes

### V14 Entry Logic
Replicate confluence scoring from lines 2174-2335:

**Pre-conditions** (all must pass before scoring):
1. In session + allowed day
2. Max concurrent positions not reached (2 max)
3. Max trades per session not reached (4)
4. No consecutive loss stop
5. Session loss limit not hit
6. Min time between entries (10 min)
7. BB width regime = "ranging" (not "trending")
8. ATR(14) on M15 ≤ 0.30 price units
9. ADX(14) on M15 ≤ 35
10. Not in breakout gate (rolling 60-min range > 40 pips → 15 min cooldown)
11. Not in no-reentry-after-stop window

**Confluence Conditions** (for LONG — mirror for SHORT):
- A (zone): M1 close ≤ S1 + 20 pips tolerance
- B (BB): M1 close ≤ BB lower OR M1 low ≤ BB lower
- C (SAR): Parabolic SAR flipped bullish within last 12 M1 bars
- D (RSI extreme): M5 RSI < 30
- E (S2 zone): M1 close ≤ S2 + 20 pips tolerance

Score = count of true conditions. Entry requires score ≥ 2.

**Combo filter**: Block combo "ABCD" (blocklist mode).

**Entry Confirmation** (lines 2494-2629):
When enabled (it IS enabled in V14 config), signals are queued. A subsequent M1 bar must close in the trade direction within 12 bars. If not confirmed within the window, the signal expires.

### V14 Position Sizing
```python
units = floor((equity * 0.02) / (sl_pips * (0.01 / entry_price)))
units = min(units, 500000)  # max_units cap
```
Where equity = current shared equity.

### V14 SL/TP Computation
- **SL**: Based on nearby structure levels (S2/R2, BB bands, SAR) + buffer (8 pips). Floor at 12 pips, hard max at 35 pips.
- **TP1 (partial)**: 0.5 × ATR(14) pips toward daily pivot P, clamped [6, 12] pips. Close 50% of units.
- **Breakeven**: After TP1 hit, move SL to entry ± 2.0 pips offset.
- **Trailing stop**: Activates after 8 pips profit (and requires TP1 hit), trails at 5 pip distance, never widens.
- **Session close**: Force close at session end (22:00 UTC).

### V14 Exit Priority (check in this order)
1. Stop loss hit
2. TP1 partial close (50% of units, move SL to BE)
3. TP2 full close (remaining units reach pivot target)
4. Trailing stop hit (after TP1)
5. Time decay (after TP1, if held > 120 min and profit < 3.0 pips)
6. Session close (force close at 22:00 UTC)

---

## V44 Logic to Replicate

Reference: `scripts/backtest_session_momentum.py`, function `run_backtest_v5()` (line 3217+)

### V44 Indicators (computed upfront)
Replicate from lines 3237-3327:
- **M1 EMA 8, EMA 21**: `ewm(span=8)`, `ewm(span=21)` on M1 close
- **M5 EMA 9, EMA 21**: `ewm(span=9)`, `ewm(span=21)` on M5 close
- **M5 EMA 9 (trail)**: Already covered by EMA 9 above
- **M5 EMA 21 (trail)**: Already covered by EMA 21 above
- **M5 ATR(14)**: True Range → rolling(14).mean() in pips
- **H1 EMA 20, EMA 50**: `ewm(span=20)`, `ewm(span=50)` on H1 close
- **M5 ADX(14)**: Only if you want to replicate blocked-reason tracking (not required for entries in baseline V44 — `adx_min = 0.0` means no ADX filter)

Note: V44's ATR percentile filter uses M5 ATR(14) values with a 200-bar rolling window, computing percentile rank.

### V44 Session Rules
- London: 08:30 - 11:00 UTC (active window same as session)
- NY Overlap: 13:00 - 16:00 UTC (with 5-minute start delay → effective 13:05)
- Entry cutoff: No entries within 60 minutes of session end (London: no entries after 10:00, NY: no entries after 15:00)
- Max 7 entries per day (combined London+NY)
- Max 4 entries in London session specifically
- Session stop losses: 3 consecutive losses → stop that session for the day
- Per-session daily USD loss limit: London $750, NY $750
- Force close full-risk positions (TP1 not yet hit) at session end
- Skip December entirely (`skip_months = "12"`)
- Skip Thursdays (`skip_days = "3"`, Python weekday 3)
- Cooldown after trade close: 1 M5 bar after win, 1 M5 bar after loss, 2 M5 bars after scratch (< 1.0 pip)

### V44 Regime Detection

**Session Efficiency** (dual mode):
- Track rolling history (deque maxlen=5) of session efficiency values
- Efficiency = |close - open| / (high - low) per session
- If avg efficiency ≥ 0.40 → "trend" mode (trades allowed)
- If avg efficiency ≤ 0.30 → "range_fade" mode (range_fade is DISABLED in V44 → no trades)
- Between → "neutral" mode (no trades)
- Default to "trend" if fewer than 3 history values

**H1 Trend** (line 4237-4252):
- H1 EMA 20 vs H1 EMA 50 → `trend_side` = "buy" if fast > slow, "sell" if fast < slow

**M5 Setup** (line 4256-4260):
- `setup_active` = True when H1 trend_side aligns with M5 EMA 9 vs 21
- Both must agree for regime = "Trending"

### V44 Pullback Entry Logic

**Trend strength** (M5 EMA slope):
- Slope = (EMA_now - EMA_ago) / (slope_bars × PIP_SIZE), where slope_bars=4
- |slope| > 0.5 (NY) or > 0.6 (London) → "Strong"
- |slope| > 0.2 → "Normal"
- Else → "Weak" (blocked — `skip_weak = true`)

**Strength gating**: Both London and NY allow "Strong" and "Normal" (`strong_normal`)

**Pullback trigger** (core V39, lines 5338-5344):
```python
M1_ema_fast = EMA(8)
M1_ema_slow = EMA(21)

# For BUY:
pullback_trigger = (M1_ema_fast > M1_ema_slow) AND (close > M1_ema_fast) AND (close > open)

# For SELL:
pullback_trigger = (M1_ema_fast < M1_ema_slow) AND (close < M1_ema_fast) AND (close < open)
```

**Confirmation bars**:
- London: 1 consecutive M1 bar with body ≥ 1.5 pips in trade direction
- NY: 1 consecutive M1 bar with body ≥ 1.0 pip in trade direction

**Additional entry filters** (all must pass):
- ATR percentile filter: M5 ATR(14) percentile rank over 200 bars must be ≤ 0.67
- News filter: NOT within 60 min before or 30 min after high-impact news event
- Entry spread ≤ 3.0 pips
- Not in cooldown
- Session not stopped (3 consec losses)
- Per-session daily USD loss not exceeded
- Within entry cutoff window

### V44 News Trend Entry (secondary)
When enabled (it IS enabled):
- After a high-impact news event, delay 45 minutes, then open a 90-minute window
- Look for 3 consecutive M5 bars with body ≥ 2.0 pips in same direction
- Require M5 EMA 9 > 21 alignment (for buy)
- Fixed SL: 8.0 pips, TP: 1.5 × SL = 12 pips
- TP1 close fraction: 1.0 (full close at TP, no runner)
- Risk: 0.5% of account
- Max 1 entry per news event
- Entry regime: "NEWS_TREND" (separate from normal V39 cooldown tracking)

### V44 SL Computation
```python
# M1 lookback window (6 bars) + current M5 bar
if entry_side == "buy":
    raw_stop = min(m1_6bar_low_min, m5_current_low) - 1.5 * PIP_SIZE
else:
    raw_stop = max(m1_6bar_high_max, m5_current_high) + 1.5 * PIP_SIZE

sl_distance = abs(entry_price - raw_stop) / PIP_SIZE

# Floor/cap
if sl_distance > 9.0:  # sl_cap_pips
    SKIP THIS TRADE ENTIRELY (do not enter)
if sl_distance < 7.0:  # sl_floor_pips
    sl_distance = 7.0  # widen to floor
```

### V44 TP/Trail Parameters (by strength)

| Param | Strong | Normal |
|-------|--------|--------|
| TP1 mult | 2.0× SL | 1.75× SL |
| TP2 mult | 5.0× SL | 3.0× SL |
| TP1 close % | 30% | 50% |
| BE offset | 0.5 pips | 0.5 pips |
| Trail EMA | M5 EMA 21 | M5 EMA 9 |
| Trail buffer | 4.0 pips | 3.0 pips |
| Trail start | TP1 + 0.5× SL | TP1 + 0.5× SL |

**London overrides**:
- TP1 mult: 1.2× SL, TP2: 999 (trail-only, no TP2), TP1 close: 40%
- Trail EMA: M5 EMA 9, buffer: 2.0 pips

### V44 Position Sizing (Risk Parity)
```python
base_risk_usd = equity * 0.005  # 0.5% risk
pip_value_per_lot = (100000.0 * 0.01) / entry_price
raw_lot = base_risk_usd / (sl_pips * pip_value_per_lot)

# Session multiplier
if session == "london":  sess_mult = 0.7
elif session == "ny":    sess_mult = 1.2

# Strength multiplier
if strength == "Strong":  str_mult = 1.0
elif strength == "Normal": str_mult = 0.6

rp_lot = raw_lot * sess_mult * str_mult

# Win bonus (session-scoped streak)
bonus = 1.0 + 0.25 * win_streak_steps  # capped at 2.0x

# WR scale (20-trade rolling window per session)
# If WR >= 0.38 → scale 1.0
# If WR < 0.38 → scale 0.2

final_lot = rp_lot * bonus * wr_scale
final_lot = clamp(1.0, 20.0)  # min/max lot
```

IMPORTANT: `equity` here is the **shared** equity — same variable V14 uses.

### V44 Exit Priority (check in this order)
1. Session-end close (full risk positions where TP1 not yet hit)
2. Pre-TP1: SL hit → full close, TP1 hit → partial close + move SL to BE
3. Intrabar conflict (SL and TP1 both hit same bar): nearest-to-open wins
4. Post-TP1: TP2 hit, trailing stop hit
5. End-of-data cleanup

### V44 Trailing Stop (after TP1)
- Trail does NOT arm until price reaches entry + (TP1_distance + 0.5 × SL) pips
- Trail stop = M5 trail EMA value ± trail buffer pips
- Only ratchets favorably (buy: new_stop must be > current_stop)

---

## Spread Model

Use V44's more sophisticated realistic spread model for BOTH strategies:

**Base spread** (`compute_spread_pips()` — line 1173 of V44 engine):
- Deterministic, time-of-day-based:
  - Tokyo hours (00:00-07:00 UTC): 1.551 pips
  - London hours (07:00-11:00 UTC): 2.001 pips
  - NY overlap (13:00-16:00 UTC): 2.307 pips
  - Other hours: 1.550 pips
- Clamped to [1.0, 3.0]

**V5 spread enhancements** (`_apply_v5_spread_spikes()` — line 3634 of V44 engine):
1. TOD profile: `"00:00-01:30|1.12|1.4|4.0;11:00-12:00|1.08|1.3|3.5;20:00-23:00|1.22|1.8|6.0"`
2. Rollover spike (21:57-22:09 UTC): 2.5x mult, min 3.0, max 12.0
3. Entry spread gate: block entry if spread > 3.0 pips (for V44) or apply V14's fixed 1.5 pip spread assumption

**For V14 entries**: V14's original config uses a fixed 1.5 pip spread. In the merged engine, apply the realistic spread model during Tokyo session (which naturally produces ~1.55 pips during Tokyo hours). This is a minor change and more realistic.

**Bid/ask derivation**:
```python
half_spread = spread_pips * PIP_SIZE / 2.0
bid = mid_close - half_spread
ask = mid_close + half_spread
# Same for high/low: bid_high = high - half_spread, etc.
```

---

## Shared State

### Shared Between Strategies
- `equity: float` — single equity variable, starts at `--starting-equity`
- `peak_equity: float` — tracks peak for drawdown calculation
- `all_closed_trades: list[dict]` — combined closed trade log (tagged with `strategy: "v14"` or `"v44"`)
- `equity_curve: list[dict]` — combined equity curve

### V14-Only State
- `v14_open_positions: list` — V14's open positions (max 2 concurrent)
- `v14_session_state: dict` — per-session-day tracking (trade count, consec losses, last entry time, session PnL, etc.)
- `v14_pending_signals: list` — confirmation queue for V14 signals
- `v14_diag: Counter` — V14 diagnostic counters

### V44-Only State
- `v44_open_positions: list` — V44's open positions (max 4 concurrent)
- `v44_daily_state: dict` — per-day tracking (entries opened, session stops, etc.)
- `v44_session_efficiency_history: deque(maxlen=5)` — for dual mode detection
- `v44_cooldown_until_p5: int` — M5 bar cooldown index
- `v44_wr_deques: dict` — per-session rolling WR deques (maxlen=20)
- `v44_news_events: list` — loaded from CSV, for news filter + news trend
- `v44_diag: Counter` — V44 diagnostic counters

---

## Output Format

Save to the `--output` path as JSON:

```json
{
    "engine": "merged_v14_v44",
    "dataset": "USDJPY_M1_OANDA_500k.csv",
    "date_range": {"start_utc": "...", "end_utc": "..."},
    "starting_equity": 100000,
    "ending_equity": 0,

    "combined": {
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "win_rate_pct": 0.0,
        "profit_factor": 0.0,
        "net_usd": 0.0,
        "net_pips": 0.0,
        "max_drawdown_usd": 0.0,
        "max_drawdown_pct": 0.0,
        "sharpe_ratio": 0.0,
        "calmar_ratio": 0.0,
        "avg_win_pips": 0.0,
        "avg_loss_pips": 0.0,
        "avg_win_usd": 0.0,
        "avg_loss_usd": 0.0,
        "max_consecutive_wins": 0,
        "max_consecutive_losses": 0
    },

    "v14_subset": {
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "win_rate_pct": 0.0,
        "profit_factor": 0.0,
        "net_usd": 0.0,
        "net_pips": 0.0,
        "max_drawdown_usd": 0.0,
        "avg_win_pips": 0.0,
        "avg_loss_pips": 0.0,
        "by_day_of_week": [],
        "by_exit_reason": []
    },

    "v44_subset": {
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "win_rate_pct": 0.0,
        "profit_factor": 0.0,
        "net_usd": 0.0,
        "net_pips": 0.0,
        "max_drawdown_usd": 0.0,
        "avg_win_pips": 0.0,
        "avg_loss_pips": 0.0,
        "by_session": [],
        "by_entry_signal_mode": [],
        "by_exit_reason": [],
        "by_trend_strength": [],
        "sizing_stats": {}
    },

    "by_session": [
        {"session": "tokyo", "strategy": "v14", "trades": 0, "win_rate": 0, "net_usd": 0, "pf": 0},
        {"session": "london", "strategy": "v44", "trades": 0, "win_rate": 0, "net_usd": 0, "pf": 0},
        {"session": "ny_overlap", "strategy": "v44", "trades": 0, "win_rate": 0, "net_usd": 0, "pf": 0}
    ],

    "by_month": [],

    "equity_curve": [
        {"trade_number": 1, "strategy": "v14", "entry_time": "...", "exit_time": "...", "pnl_usd": 0, "equity_after": 0, "drawdown_usd": 0}
    ],

    "closed_trades": [],

    "diagnostics": {
        "v14": {"signals_generated": 0, "signals_confirmed": 0, "signals_expired": 0, "entries_blocked_by": {}},
        "v44": {"entries_attempted": 0, "entries_blocked_by": {}, "news_trend_entries": 0, "sl_cap_skipped": 0}
    }
}
```

---

## Validation Criteria

After running on 500k (`research_out/USDJPY_M1_OANDA_500k.csv`):

### V14 Subset Should Approximate:
| Metric | Standalone Baseline | Acceptable Range |
|--------|-------------------|------------------|
| Trades | 60 | 55-65 |
| Win Rate | 83.33% | 78-88% |
| Profit Factor | 3.19 | 2.5-4.0 |
| Net USD | +$6,819 | +$5,500 - +$8,500 |

### V44 Subset Should Approximate:
| Metric | Standalone Baseline | Acceptable Range |
|--------|-------------------|------------------|
| Trades | 257 | 240-275 |
| Win Rate | 40.86% | 37-45% |
| Profit Factor | 1.58 | 1.3-1.8 |
| Net USD | +$30,529 | +$25,000 - +$36,000 |

### Expected Differences
Small deviations are expected because:
1. **Shared equity**: V14 trades on equity modified by V44 results (and vice versa). V14 uses 2% risk, so a $30k boost from V44 means larger V14 positions.
2. **Spread model**: V14's original used fixed 1.5p spread; merged uses realistic model (~1.55p during Tokyo). Effect should be minimal.
3. **Combined drawdown**: The combined max DD should be LESS THAN the sum of individual max DDs (diversification benefit from non-overlapping sessions).

### If Validation Fails
If any subset diverges by more than the acceptable ranges:
1. Run the standalone engine on the same 500k dataset and compare trade-by-trade
2. Check that session classification matches (no V14 trades during London hours or vice versa)
3. Check that position sizing uses the correct formula for each strategy
4. Check that exit rules are correctly applied per strategy

---

## Run Plan

### Step 1: Build & Run on 500k
```bash
python scripts/backtest_merged_v14_v44.py \
  --v14-config research_out/tokyo_optimized_v14_config.json \
  --v44-config research_out/v44_combined_winners_500k.json \
  --input research_out/USDJPY_M1_OANDA_500k.csv \
  --output research_out/phase2_merged_500k_report.json \
  --starting-equity 100000
```

### Step 2: Validate
Compare v14_subset and v44_subset metrics against standalone baselines (listed above).

### Step 3: Run on 1000k
```bash
python scripts/backtest_merged_v14_v44.py \
  --v14-config research_out/tokyo_optimized_v14_config.json \
  --v44-config research_out/v44_combined_winners_500k.json \
  --input research_out/USDJPY_M1_OANDA_1000k.csv \
  --output research_out/phase2_merged_1000k_report.json \
  --starting-equity 100000
```

### Step 4: Report
Print a summary table to console:
```
╔═══════════════════════════════════════════════════════════════════╗
║           MERGED V14+V44 BACKTEST RESULTS (500k)                ║
╠═══════════════╦══════════╦════════╦═══════╦═══════════╦═════════╣
║ Strategy      ║ Trades   ║ WR%    ║ PF    ║ Net USD   ║ Max DD  ║
╠═══════════════╬══════════╬════════╬═══════╬═══════════╬═════════╣
║ V14 Tokyo     ║ XX       ║ XX.X%  ║ X.XX  ║ +$XX,XXX  ║ $X,XXX  ║
║ V44 London+NY ║ XX       ║ XX.X%  ║ X.XX  ║ +$XX,XXX  ║ $X,XXX  ║
║ COMBINED      ║ XX       ║ XX.X%  ║ X.XX  ║ +$XX,XXX  ║ $X,XXX  ║
╚═══════════════╩══════════╩════════╩═══════╩═══════════╩═════════╝
Combined Sharpe: X.XX | Calmar: X.XX | Return: +X.X%
```

---

## Implementation Notes

### PnL Calculation
Both strategies use the same formula (USDJPY-specific):
```python
pnl_pips = (exit_price - entry_price) / PIP_SIZE  # for long; negate for short
pnl_usd = pnl_pips * units * (PIP_SIZE / exit_price)  # V14 style
# OR equivalently for V44's lot-based:
pnl_usd = pnl_pips * lots * pip_value_per_lot  # where pip_value_per_lot = 1000.0 / price for standard lot
```

Make sure both strategies use a consistent PnL formula. The V14 formula (units-based) and V44 formula (lots-based) are mathematically equivalent — just use one consistently.

### Timeframe Pointers
Both engines advance M5/M15/H1 pointers inside the M1 loop using a forward-walking index (`p5`, `p15`, `p1h`). The merged engine needs:
- `p5` — M5 bar pointer (shared)
- `p15` — M15 bar pointer (used by V14 for ATR/ADX)
- `p1h` — H1 bar pointer (used by V44 for trend detection)
- Track `new_m5_bar` flag (V44 uses this for cooldown counting and trail updates)

### Position Dataclass
Create a unified Position dataclass or use two separate ones. A unified one is simpler:
```python
@dataclass
class Position:
    strategy: str  # "v14" or "v44"
    side: str  # "buy" or "sell"
    entry_price: float
    entry_time: datetime
    units: int  # V14 uses units
    lots: float  # V44 uses lots
    sl_price: float
    tp1_price: float
    tp2_price: float
    tp1_hit: bool = False
    tp1_close_fraction: float = 0.5
    initial_units: int = 0
    initial_lots: float = 0.0
    be_offset_pips: float = 0.0
    trail_active: bool = False
    trail_stop: float = 0.0
    # V14-specific:
    trail_distance_pips: float = 5.0
    # V44-specific:
    trail_ema_period: int = 21
    trail_buffer_pips: float = 4.0
    trail_start_threshold: float = 0.0
    strength: str = ""
    entry_regime: str = ""
    entry_session: str = ""
    peak_mfe_pips: float = 0.0
    # ... add fields as needed
```

### News Calendar Loading
V44 uses `research_out/v5_scheduled_events_utc.csv`. Load this at startup. The CSV has columns for date, time_utc, impact (at minimum). Only "high" impact events are used.

### Helper Functions to Copy
From V14 engine (`backtest_tokyo_meanrev.py`):
- `load_m1()` (line 88) — CSV loading
- `resample_ohlc_continuous()` (line 107) — M1→M5/M15
- `rolling_rsi()` (line 118)
- `rolling_atr_price()` (line 129)
- `rolling_adx()` (line 142)
- `compute_psar()` (line 167)

From V44 engine (`backtest_session_momentum.py`):
- `compute_spread_pips()` (line 1173) — realistic spread model
- `hour_in_window()` (line 1138) — handles midnight wrapping
- `classify_session()` (line 1153)

Both engines' resampling functions are nearly identical (`resample_ohlc_continuous` vs `resample_ohlc`) — use either one.

---

## Important Warnings

1. **Do NOT implement all of V44's experimental modules** (swing, bb_reversion, ema_cross, london_orb, tokyo_orb, tokyo_fade, range_fade). These are all DISABLED in the V44 baseline config. Only implement: the core V39 pullback entry, news_trend entry, and the standard exit system.

2. **Do NOT implement V44's v2/v3/v4 logic**. Only v5 matters. The v2/v3/v4 sections in the config are legacy and unused when `--version v5`.

3. **Keep V14's entry confirmation queue** — this is a critical feature that filters 85% of raw signals.

4. **The merged engine does NOT need to support parameter sweeping or grid search.** It's a single-config runner.

5. **Fibonacci pivots must use NY close (22:00 UTC) as day boundary**, NOT calendar midnight. This means a "trading day" runs from 22:00 UTC to 22:00 UTC the next day. The V14 engine handles this with `ny_day = (time - timedelta(hours=22)).date()`.

6. **V44's session efficiency tracking**: Finalize the previous session's efficiency at the START of each new session. The V44 engine tracks session_open, session_high, session_low, session_close_price and computes efficiency when the session ends.
