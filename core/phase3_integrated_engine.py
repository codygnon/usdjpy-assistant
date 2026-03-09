"""
Phase 3 Integrated Engine – Tokyo V14 Mean Reversion.

All parameters from research_out/tokyo_optimized_v14_realism_maxopen3_walkforward_config.json
are hardcoded as module-level constants.  London V2 and NY V5 are stubbed.
"""
from __future__ import annotations

import math
import re
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Session constants
# ---------------------------------------------------------------------------
TOKYO_START_UTC = 16  # 16:00 UTC
TOKYO_END_UTC = 22    # 22:00 UTC
TOKYO_ALLOWED_DAYS = {1, 2, 4}  # Tuesday=1, Wednesday=2, Friday=4  (weekday())

# ---------------------------------------------------------------------------
# Indicator constants
# ---------------------------------------------------------------------------
BB_TF = "M5"
BB_PERIOD = 25
BB_STD = 2.2

BB_WIDTH_LOOKBACK = 100
BB_WIDTH_RANGING_PCT = 0.80  # below 80th percentile = ranging

PSAR_AF_START = 0.02
PSAR_AF_STEP = 0.02
PSAR_AF_MAX = 0.20
PSAR_FLIP_LOOKBACK = 12  # M1 bars

RSI_TF = "M5"
RSI_PERIOD = 14
RSI_LONG_ENTRY = 35
RSI_SHORT_ENTRY = 65

ATR_TF = "M15"
ATR_PERIOD = 14
ATR_MAX = 0.30  # price units (= 30 pips for USDJPY)

ADX_TF = "M15"
ADX_PERIOD = 14
ADX_MAX = 35

# ---------------------------------------------------------------------------
# Entry constants
# ---------------------------------------------------------------------------
MIN_CONFLUENCE = 2
ZONE_TOLERANCE_PIPS = 20
BLOCKED_COMBOS = {"ABCD"}

# ---------------------------------------------------------------------------
# SL / TP / Exit constants
# ---------------------------------------------------------------------------
SL_BUFFER_PIPS = 8
SL_MIN_PIPS = 12
SL_MAX_PIPS = 35

TP1_ATR_MULT = 0.5
TP1_MIN_PIPS = 6
TP1_MAX_PIPS = 12
TP1_CLOSE_PCT = 0.50  # 50 %

BE_OFFSET_PIPS = 2

TRAIL_ACTIVATE_PROFIT_PIPS = 8
TRAIL_DISTANCE_PIPS = 5
TRAIL_REQUIRES_TP1 = True
TRAIL_NEVER_WIDEN = True

TIME_DECAY_MINUTES = 120
TIME_DECAY_CAP_PIPS = 3.0

PRIMARY_TP_TARGET = "pivot_P"
MIN_TP_DISTANCE_PIPS = 8

# ---------------------------------------------------------------------------
# Sizing / risk
# ---------------------------------------------------------------------------
RISK_PCT = 0.02  # 2 %
MAX_UNITS = 500_000
MAX_CONCURRENT = 3
LEVERAGE = 33.3

# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------
MAX_TRADES_PER_SESSION = 4
COOLDOWN_MINUTES = 15
STOP_AFTER_CONSECUTIVE_LOSSES = 3
SESSION_LOSS_STOP_PCT = 0.015  # 1.5 %
MAX_ENTRY_SPREAD_PIPS = 3.0

PIP_SIZE = 0.01  # USDJPY


# ===================================================================
#  Session classifier
# ===================================================================

def classify_session(now_utc: datetime) -> Optional[str]:
    """Return 'tokyo', 'london', 'ny', or None."""
    hour = now_utc.hour
    weekday = now_utc.weekday()
    if TOKYO_START_UTC <= hour < TOKYO_END_UTC and weekday in TOKYO_ALLOWED_DAYS:
        return "tokyo"
    # London / NY stubs — not active this increment
    return None


# ===================================================================
#  Fibonacci Pivots
# ===================================================================

def compute_daily_fib_pivots(prev_day_high: float, prev_day_low: float, prev_day_close: float) -> dict:
    """Compute daily Fibonacci pivot levels {P, R1, R2, R3, S1, S2, S3}."""
    p = (prev_day_high + prev_day_low + prev_day_close) / 3.0
    r = prev_day_high - prev_day_low
    return {
        "P": p,
        "R1": p + 0.382 * r,
        "R2": p + 0.618 * r,
        "R3": p + 1.000 * r,
        "S1": p - 0.382 * r,
        "S2": p - 0.618 * r,
        "S3": p - 1.000 * r,
    }


# ===================================================================
#  BB Width Regime
# ===================================================================

def compute_bb_width_regime(m5_df: pd.DataFrame) -> str:
    """Return 'ranging' or 'trending' based on BB width percentile."""
    close = m5_df["close"].astype(float)
    sma = close.rolling(BB_PERIOD).mean()
    std = close.rolling(BB_PERIOD).std()
    upper = sma + BB_STD * std
    lower = sma - BB_STD * std
    width = (upper - lower) / sma
    if len(width.dropna()) < BB_WIDTH_LOOKBACK:
        return "ranging"  # default safe
    recent = width.iloc[-BB_WIDTH_LOOKBACK:]
    cutoff = recent.quantile(BB_WIDTH_RANGING_PCT)
    current = width.iloc[-1]
    if pd.isna(current) or pd.isna(cutoff):
        return "ranging"
    return "trending" if current >= cutoff else "ranging"


def _compute_bb(m5_df: pd.DataFrame) -> tuple[float, float, float]:
    """Return (bb_upper, bb_middle, bb_lower) for latest M5 bar."""
    close = m5_df["close"].astype(float)
    sma = close.rolling(BB_PERIOD).mean()
    std = close.rolling(BB_PERIOD).std()
    bb_upper = float(sma.iloc[-1] + BB_STD * std.iloc[-1])
    bb_lower = float(sma.iloc[-1] - BB_STD * std.iloc[-1])
    bb_mid = float(sma.iloc[-1])
    return bb_upper, bb_mid, bb_lower


# ===================================================================
#  RSI
# ===================================================================

def _compute_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Compute RSI on a price series."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


# ===================================================================
#  ATR
# ===================================================================

def _compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """True Range -> EWM ATR."""
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


# ===================================================================
#  ADX
# ===================================================================

def _compute_adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> float:
    """Return latest ADX value."""
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)
    plus_dm = (high - prev_high).clip(lower=0)
    minus_dm = (prev_low - low).clip(lower=0)
    # Zero out when the other is larger
    plus_dm = plus_dm.where(plus_dm > minus_dm, 0.0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0.0)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100.0 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
    minus_di = 100.0 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)
    dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    adx = dx.ewm(span=period, adjust=False).mean()
    val = adx.iloc[-1]
    return float(val) if pd.notna(val) else 0.0


# ===================================================================
#  Parabolic SAR
# ===================================================================

def compute_parabolic_sar(m1_df: pd.DataFrame) -> pd.Series:
    """Return PSAR series for M1 data."""
    high = m1_df["high"].astype(float).values
    low = m1_df["low"].astype(float).values
    n = len(high)
    psar = np.full(n, np.nan)
    af = PSAR_AF_START
    trend = 1  # 1 = up, -1 = down
    if n < 2:
        return pd.Series(psar, index=m1_df.index)
    psar[0] = low[0]
    ep = high[0]
    for i in range(1, n):
        prev_psar = psar[i - 1]
        if trend == 1:
            psar[i] = prev_psar + af * (ep - prev_psar)
            psar[i] = min(psar[i], low[i - 1])
            if i >= 2:
                psar[i] = min(psar[i], low[i - 2])
            if high[i] > ep:
                ep = high[i]
                af = min(af + PSAR_AF_STEP, PSAR_AF_MAX)
            if low[i] < psar[i]:
                trend = -1
                psar[i] = ep
                ep = low[i]
                af = PSAR_AF_START
        else:
            psar[i] = prev_psar + af * (ep - prev_psar)
            psar[i] = max(psar[i], high[i - 1])
            if i >= 2:
                psar[i] = max(psar[i], high[i - 2])
            if low[i] < ep:
                ep = low[i]
                af = min(af + PSAR_AF_STEP, PSAR_AF_MAX)
            if high[i] > psar[i]:
                trend = 1
                psar[i] = ep
                ep = high[i]
                af = PSAR_AF_START
    return pd.Series(psar, index=m1_df.index)


def detect_sar_flip(m1_df: pd.DataFrame, lookback: int = PSAR_FLIP_LOOKBACK) -> tuple[bool, bool]:
    """Return (bullish_flip, bearish_flip) within last N M1 bars."""
    psar = compute_parabolic_sar(m1_df)
    close = m1_df["close"].astype(float)
    if len(psar) < lookback + 1:
        return False, False
    bullish = False
    bearish = False
    for i in range(-lookback, 0):
        prev_idx = i - 1
        if abs(prev_idx) > len(psar):
            continue
        prev_above = psar.iloc[prev_idx] > close.iloc[prev_idx]
        curr_below = psar.iloc[i] < close.iloc[i]
        if prev_above and curr_below:
            bullish = True
        prev_below = psar.iloc[prev_idx] < close.iloc[prev_idx]
        curr_above = psar.iloc[i] > close.iloc[i]
        if prev_below and curr_above:
            bearish = True
    return bullish, bearish


# ===================================================================
#  Confluence scoring
# ===================================================================

def evaluate_v14_confluence(
    side: str,
    close: float,
    pivots: dict,
    bb_upper: float,
    bb_lower: float,
    sar_bullish_flip: bool,
    sar_bearish_flip: bool,
    rsi: float,
    pip_size: float,
) -> tuple[int, str]:
    """
    Score confluence for V14 entry.

    Components (labelled A-D):
      A = price in pivot zone (close <= S1+tol for long, close >= R1-tol for short)
      B = BB touch/penetration (close <= bb_lower for long, close >= bb_upper for short)
      C = SAR flip (bullish for long, bearish for short)
      D = RSI soft filter (< 35 for long, > 65 for short)

    Returns (score, combo_string) e.g. (3, "ABC").
    """
    score = 0
    combo = ""
    tol = ZONE_TOLERANCE_PIPS * pip_size

    if side == "buy":
        # A: pivot zone
        if close <= pivots["S1"] + tol:
            score += 1
            combo += "A"
        # B: BB lower touch
        if close <= bb_lower:
            score += 1
            combo += "B"
        # C: SAR bullish flip
        if sar_bullish_flip:
            score += 1
            combo += "C"
        # D: RSI
        if rsi < RSI_LONG_ENTRY:
            score += 1
            combo += "D"
    else:  # sell
        if close >= pivots["R1"] - tol:
            score += 1
            combo += "A"
        if close >= bb_upper:
            score += 1
            combo += "B"
        if sar_bearish_flip:
            score += 1
            combo += "C"
        if rsi > RSI_SHORT_ENTRY:
            score += 1
            combo += "D"

    return score, combo


# ===================================================================
#  SL / TP computation
# ===================================================================

def compute_v14_sl(side: str, entry_price: float, pivots: dict, pip_size: float) -> float:
    """Compute SL price based on pivot + buffer, clamped to min/max."""
    buffer = SL_BUFFER_PIPS * pip_size
    if side == "buy":
        # SL below nearest support
        support_levels = sorted([pivots["S1"], pivots["S2"], pivots["S3"]])
        nearest = None
        for lvl in support_levels:
            if lvl < entry_price:
                nearest = lvl
                break
        if nearest is None:
            nearest = pivots["S1"]
        raw_sl = nearest - buffer
        sl_dist = (entry_price - raw_sl) / pip_size
    else:
        resist_levels = sorted([pivots["R1"], pivots["R2"], pivots["R3"]], reverse=True)
        nearest = None
        for lvl in resist_levels:
            if lvl > entry_price:
                nearest = lvl
                break
        if nearest is None:
            nearest = pivots["R1"]
        raw_sl = nearest + buffer
        sl_dist = (raw_sl - entry_price) / pip_size

    # Clamp
    sl_dist = max(SL_MIN_PIPS, min(SL_MAX_PIPS, sl_dist))
    if side == "buy":
        return entry_price - sl_dist * pip_size
    else:
        return entry_price + sl_dist * pip_size


def compute_v14_tp1(side: str, entry_price: float, atr_value: float, pip_size: float) -> float:
    """Compute TP1 = ATR * 0.5, clamped to 6-12 pips."""
    tp_dist = atr_value * TP1_ATR_MULT
    tp_pips = tp_dist / pip_size
    tp_pips = max(TP1_MIN_PIPS, min(TP1_MAX_PIPS, tp_pips))
    if side == "buy":
        return entry_price + tp_pips * pip_size
    else:
        return entry_price - tp_pips * pip_size


# ===================================================================
#  Lot sizing
# ===================================================================

def compute_v14_lot_size(
    equity: float,
    sl_pips: float,
    current_price: float,
    pip_size: float,
    leverage: float = LEVERAGE,
) -> int:
    """Risk-based units, margin-aware.  Returns integer units."""
    if sl_pips <= 0 or current_price <= 0:
        return 0
    # Risk amount in account currency (USD for OANDA USDJPY)
    risk_amount = equity * RISK_PCT
    # For USDJPY: pip_value per unit = pip_size / current_price (USD per pip per unit)
    pip_value_per_unit = pip_size / current_price
    units = risk_amount / (sl_pips * pip_value_per_unit)
    # Margin check: required margin = units * current_price / leverage
    # (For JPY-denominated, OANDA uses quote_currency / leverage)
    max_margin_units = (equity * leverage) / current_price
    units = min(units, max_margin_units)
    units = min(units, MAX_UNITS)
    return int(math.floor(units))


# ===================================================================
#  Main execute function
# ===================================================================

def execute_phase3_integrated_policy_demo_only(
    *,
    adapter,
    profile,
    log_dir,
    policy,
    context,
    data_by_tf: dict,
    tick,
    mode: str,
    phase3_state: dict,
    store=None,
) -> dict:
    """
    Main Phase 3 entry point.  Returns dict with:
      decision: ExecutionDecision-like dict
      phase3_state_updates: dict to merge into phase3_state
      strategy_tag: str | None  (e.g. "phase3:v14")
    """
    from core.execution_engine import ExecutionDecision

    now_utc = datetime.now(timezone.utc)
    pip = float(profile.pip_size) if hasattr(profile, "pip_size") else PIP_SIZE
    no_trade = {
        "decision": ExecutionDecision(attempted=False, placed=False, reason="", side=None),
        "phase3_state_updates": {},
        "strategy_tag": None,
    }

    # 1) Session gate
    session = classify_session(now_utc)
    if session != "tokyo":
        no_trade["decision"] = ExecutionDecision(
            attempted=False, placed=False,
            reason=f"phase3: no active session (current={session})", side=None,
        )
        return no_trade

    # 2) Demo guard
    if mode != "ARMED_AUTO_DEMO":
        no_trade["decision"] = ExecutionDecision(
            attempted=False, placed=False,
            reason=f"phase3: mode={mode} (need ARMED_AUTO_DEMO)", side=None,
        )
        return no_trade

    is_demo = getattr(adapter, "is_demo", True)
    if callable(is_demo):
        is_demo = is_demo()
    if not is_demo:
        no_trade["decision"] = ExecutionDecision(
            attempted=False, placed=False,
            reason="phase3: adapter is not demo", side=None,
        )
        return no_trade

    # 3) Spread gate
    spread_pips = (tick.ask - tick.bid) / pip
    if spread_pips > MAX_ENTRY_SPREAD_PIPS:
        no_trade["decision"] = ExecutionDecision(
            attempted=False, placed=False,
            reason=f"phase3: spread {spread_pips:.1f}p > max {MAX_ENTRY_SPREAD_PIPS}p", side=None,
        )
        return no_trade

    # 4) Pivots (cache by date)
    today_str = now_utc.date().isoformat()
    pivots = phase3_state.get("pivots")
    pivots_date = phase3_state.get("pivots_date")
    state_updates: dict = {}
    if pivots is None or pivots_date != today_str:
        d_df = data_by_tf.get("D")
        if d_df is None or d_df.empty or len(d_df) < 2:
            no_trade["decision"] = ExecutionDecision(
                attempted=False, placed=False,
                reason="phase3: no daily data for pivots", side=None,
            )
            return no_trade
        d_sorted = d_df.sort_values("time")
        prev = d_sorted.iloc[-2]
        pivots = compute_daily_fib_pivots(
            float(prev["high"]), float(prev["low"]), float(prev["close"]),
        )
        state_updates["pivots"] = pivots
        state_updates["pivots_date"] = today_str

    # 5) Indicators
    m5_df = data_by_tf.get("M5")
    m15_df = data_by_tf.get("M15")
    m1_df = data_by_tf.get("M1")
    if m5_df is None or m5_df.empty or m15_df is None or m15_df.empty or m1_df is None or m1_df.empty:
        no_trade["decision"] = ExecutionDecision(
            attempted=False, placed=False,
            reason="phase3: missing M1/M5/M15 data", side=None,
        )
        no_trade["phase3_state_updates"] = state_updates
        return no_trade

    # BB + regime
    if len(m5_df) < BB_PERIOD + BB_WIDTH_LOOKBACK:
        no_trade["decision"] = ExecutionDecision(
            attempted=False, placed=False,
            reason="phase3: insufficient M5 data for BB", side=None,
        )
        no_trade["phase3_state_updates"] = state_updates
        return no_trade

    regime = compute_bb_width_regime(m5_df)
    bb_upper, bb_mid, bb_lower = _compute_bb(m5_df)

    # RSI
    m5_close = m5_df["close"].astype(float)
    rsi_series = _compute_rsi(m5_close, RSI_PERIOD)
    rsi_val = float(rsi_series.iloc[-1]) if pd.notna(rsi_series.iloc[-1]) else 50.0

    # ATR (M15)
    if len(m15_df) < ATR_PERIOD + 2:
        no_trade["decision"] = ExecutionDecision(
            attempted=False, placed=False,
            reason="phase3: insufficient M15 data for ATR", side=None,
        )
        no_trade["phase3_state_updates"] = state_updates
        return no_trade
    atr_series = _compute_atr(m15_df, ATR_PERIOD)
    atr_val = float(atr_series.iloc[-1]) if pd.notna(atr_series.iloc[-1]) else 0.0

    # ADX (M15)
    adx_val = _compute_adx(m15_df, ADX_PERIOD)

    # PSAR + flip
    sar_bull, sar_bear = detect_sar_flip(m1_df, PSAR_FLIP_LOOKBACK)

    # 6) Regime gate
    if regime != "ranging":
        no_trade["decision"] = ExecutionDecision(
            attempted=False, placed=False,
            reason=f"phase3: regime={regime} (need ranging)", side=None,
        )
        no_trade["phase3_state_updates"] = state_updates
        return no_trade

    # 7) ADX gate
    if adx_val >= ADX_MAX:
        no_trade["decision"] = ExecutionDecision(
            attempted=False, placed=False,
            reason=f"phase3: ADX={adx_val:.1f} >= {ADX_MAX}", side=None,
        )
        no_trade["phase3_state_updates"] = state_updates
        return no_trade

    # 8) ATR gate
    if atr_val >= ATR_MAX:
        no_trade["decision"] = ExecutionDecision(
            attempted=False, placed=False,
            reason=f"phase3: ATR={atr_val:.4f} >= {ATR_MAX}", side=None,
        )
        no_trade["phase3_state_updates"] = state_updates
        return no_trade

    # 9) Session management: concurrent positions, trade count, cooldown, consecutive losses
    session_key = f"session_{today_str}"
    session_data = phase3_state.get(session_key, {})
    trade_count = session_data.get("trade_count", 0)
    consecutive_losses = session_data.get("consecutive_losses", 0)
    last_entry_time = session_data.get("last_entry_time")

    if trade_count >= MAX_TRADES_PER_SESSION:
        no_trade["decision"] = ExecutionDecision(
            attempted=False, placed=False,
            reason=f"phase3: max trades/session ({trade_count}/{MAX_TRADES_PER_SESSION})", side=None,
        )
        no_trade["phase3_state_updates"] = state_updates
        return no_trade

    if consecutive_losses >= STOP_AFTER_CONSECUTIVE_LOSSES:
        no_trade["decision"] = ExecutionDecision(
            attempted=False, placed=False,
            reason=f"phase3: stopped after {consecutive_losses} consecutive losses", side=None,
        )
        no_trade["phase3_state_updates"] = state_updates
        return no_trade

    # Cooldown
    if last_entry_time is not None:
        elapsed = (now_utc - datetime.fromisoformat(last_entry_time)).total_seconds() / 60.0
        if elapsed < COOLDOWN_MINUTES:
            no_trade["decision"] = ExecutionDecision(
                attempted=False, placed=False,
                reason=f"phase3: cooldown ({elapsed:.1f}/{COOLDOWN_MINUTES}min)", side=None,
            )
            no_trade["phase3_state_updates"] = state_updates
            return no_trade

    # Count concurrent positions (Phase 3 trades)
    concurrent = phase3_state.get("open_trade_count", 0)
    if concurrent >= MAX_CONCURRENT:
        no_trade["decision"] = ExecutionDecision(
            attempted=False, placed=False,
            reason=f"phase3: max concurrent ({concurrent}/{MAX_CONCURRENT})", side=None,
        )
        no_trade["phase3_state_updates"] = state_updates
        return no_trade

    # 10) Evaluate confluence for both sides
    close_price = float(m5_close.iloc[-1])
    best_side = None
    best_score = 0
    best_combo = ""

    for side in ("buy", "sell"):
        score, combo = evaluate_v14_confluence(
            side, close_price, pivots, bb_upper, bb_lower,
            sar_bull, sar_bear, rsi_val, pip,
        )
        if score >= MIN_CONFLUENCE and combo not in BLOCKED_COMBOS and score > best_score:
            best_side = side
            best_score = score
            best_combo = combo

    if best_side is None:
        no_trade["decision"] = ExecutionDecision(
            attempted=False, placed=False,
            reason=f"phase3: no qualifying confluence (RSI={rsi_val:.1f})", side=None,
        )
        no_trade["phase3_state_updates"] = state_updates
        return no_trade

    # 11) Compute SL, TP1, lot size
    entry_price = tick.ask if best_side == "buy" else tick.bid
    sl_price = compute_v14_sl(best_side, entry_price, pivots, pip)
    sl_pips = abs(entry_price - sl_price) / pip
    tp1_price = compute_v14_tp1(best_side, entry_price, atr_val, pip)

    # Get equity from adapter
    try:
        acct = adapter.get_account_info()
        equity = float(acct.equity)
    except Exception:
        equity = 100_000.0  # fallback

    units = compute_v14_lot_size(equity, sl_pips, entry_price, pip, LEVERAGE)
    if units <= 0:
        no_trade["decision"] = ExecutionDecision(
            attempted=False, placed=False,
            reason=f"phase3: lot size 0 (equity={equity:.0f}, sl_pips={sl_pips:.1f})", side=None,
        )
        no_trade["phase3_state_updates"] = state_updates
        return no_trade

    # 12) Place order
    strategy_tag = "phase3:v14"
    comment = f"phase3_integrated:{policy.id}:v14_mean_reversion"

    try:
        order_result = adapter.place_order(
            symbol=profile.symbol,
            side=best_side,
            lots=units / 100_000.0,
            stop_price=round(sl_price, 3),
            target_price=round(tp1_price, 3),
            comment=comment,
        )
        placed = True
    except Exception as e:
        return {
            "decision": ExecutionDecision(
                attempted=True, placed=False,
                reason=f"phase3: order error: {e}", side=best_side,
            ),
            "phase3_state_updates": state_updates,
            "strategy_tag": strategy_tag,
        }

    # Update session state
    session_data["trade_count"] = trade_count + 1
    session_data["last_entry_time"] = now_utc.isoformat()
    state_updates[session_key] = session_data

    return {
        "decision": ExecutionDecision(
            attempted=True, placed=True,
            reason=f"phase3:v14 {best_side.upper()} confluence={best_score} combo={best_combo} "
                   f"SL={sl_price:.3f}({sl_pips:.1f}p) TP1={tp1_price:.3f} units={units}",
            side=best_side,
        ),
        "phase3_state_updates": state_updates,
        "strategy_tag": strategy_tag,
        "sl_price": sl_price,
        "tp1_price": tp1_price,
        "units": units,
        "entry_price": entry_price,
        "sl_pips": sl_pips,
        "pivots": pivots,
        "atr_val": atr_val,
    }


# ===================================================================
#  Exit management
# ===================================================================

def manage_phase3_exit(
    *,
    adapter,
    profile,
    store,
    tick,
    trade_row: dict,
    position,
    data_by_tf: dict,
    phase3_state: dict,
) -> dict:
    """
    Exit management for Phase 3 trades.

    Routes by strategy tag in entry_type/comment.  Currently only V14.
    Returns dict with action taken info.
    """
    pip = float(profile.pip_size) if hasattr(profile, "pip_size") else PIP_SIZE
    entry = float(trade_row["entry_price"])
    side = str(trade_row["side"]).lower()
    trade_id = str(trade_row["trade_id"])
    entry_type = trade_row.get("entry_type", "")
    now_utc = datetime.now(timezone.utc)
    mid = (tick.bid + tick.ask) / 2.0
    current_spread = tick.ask - tick.bid

    if isinstance(position, dict):
        position_id = position.get("id")
        current_units = abs(int(position.get("currentUnits") or 0))
        current_lots = current_units / 100_000.0
    else:
        position_id = getattr(position, "ticket", None)
        current_lots = float(getattr(position, "volume", 0) or 0)
        current_units = int(current_lots * 100_000)

    if current_lots <= 0:
        return {"action": "none", "reason": "no units"}

    position_type = 1 if side == "sell" else 0

    # 1) Session end force close
    session = classify_session(now_utc)
    if session != "tokyo":
        try:
            adapter.close_position(
                ticket=position_id,
                symbol=profile.symbol,
                volume=current_lots,
                position_type=position_type,
            )
            return {"action": "session_end_close", "reason": f"session ended (now={session})"}
        except Exception as e:
            return {"action": "error", "reason": f"session end close error: {e}"}

    # 2) Time decay check
    opened_at = trade_row.get("opened_at")
    if opened_at is not None:
        try:
            if isinstance(opened_at, str):
                open_time = datetime.fromisoformat(opened_at)
                if open_time.tzinfo is None:
                    open_time = open_time.replace(tzinfo=timezone.utc)
            else:
                open_time = opened_at
            elapsed_min = (now_utc - open_time).total_seconds() / 60.0
            if elapsed_min >= TIME_DECAY_MINUTES:
                profit_pips = ((mid - entry) / pip) if side == "buy" else ((entry - mid) / pip)
                if profit_pips < TIME_DECAY_CAP_PIPS:
                    try:
                        adapter.close_position(
                            ticket=position_id,
                            symbol=profile.symbol,
                            volume=current_lots,
                            position_type=position_type,
                        )
                        return {
                            "action": "time_decay_close",
                            "reason": f"time decay {elapsed_min:.0f}min, profit={profit_pips:.1f}p < {TIME_DECAY_CAP_PIPS}p",
                        }
                    except Exception as e:
                        return {"action": "error", "reason": f"time decay close error: {e}"}
        except Exception:
            pass

    # 3) Hard SL check (software SL in case broker SL didn't trigger)
    check_price = tick.bid if side == "buy" else tick.ask
    stop_price = trade_row.get("stop_price")
    if stop_price is not None:
        stop_price = float(stop_price)
        if side == "buy" and check_price <= stop_price:
            try:
                adapter.close_position(ticket=position_id, symbol=profile.symbol, volume=current_lots, position_type=position_type)
                return {"action": "hard_sl", "reason": f"hard SL hit at {check_price:.3f}"}
            except Exception as e:
                return {"action": "error", "reason": f"hard SL close error: {e}"}
        elif side == "sell" and check_price >= stop_price:
            try:
                adapter.close_position(ticket=position_id, symbol=profile.symbol, volume=current_lots, position_type=position_type)
                return {"action": "hard_sl", "reason": f"hard SL hit at {check_price:.3f}"}
            except Exception as e:
                return {"action": "error", "reason": f"hard SL close error: {e}"}

    # 4) TP1 partial close
    tp1_done = trade_row.get("tp1_partial_done") or 0
    if not tp1_done:
        target_price = trade_row.get("target_price")
        if target_price is not None:
            target_price = float(target_price)
            reached_buy = mid >= target_price and side == "buy"
            reached_sell = mid <= target_price and side == "sell"
            if reached_buy or reached_sell:
                close_lots = current_lots * TP1_CLOSE_PCT
                try:
                    adapter.close_position(
                        ticket=position_id,
                        symbol=profile.symbol,
                        volume=close_lots,
                        position_type=position_type,
                    )
                    # Move SL to BE + offset
                    be_offset = current_spread + BE_OFFSET_PIPS * pip
                    if side == "buy":
                        be_sl = entry + be_offset
                    else:
                        be_sl = entry - be_offset
                    adapter.update_position_stop_loss(position_id, profile.symbol, round(be_sl, 3))
                    store.update_trade(trade_id, {
                        "tp1_partial_done": 1,
                        "breakeven_applied": 1,
                        "breakeven_sl_price": round(be_sl, 5),
                    })
                    return {
                        "action": "tp1_partial",
                        "reason": f"TP1 {TP1_CLOSE_PCT*100:.0f}% closed, BE SL={be_sl:.3f}",
                    }
                except Exception as e:
                    return {"action": "error", "reason": f"TP1 close error: {e}"}

    # 5) Trailing stop (after TP1)
    elif tp1_done:
        profit_pips = ((mid - entry) / pip) if side == "buy" else ((entry - mid) / pip)
        if profit_pips >= TRAIL_ACTIVATE_PROFIT_PIPS:
            prev_trail = trade_row.get("breakeven_sl_price")
            if prev_trail is not None:
                prev_trail = float(prev_trail)
            if side == "buy":
                new_trail = mid - TRAIL_DISTANCE_PIPS * pip
                if prev_trail is not None and TRAIL_NEVER_WIDEN:
                    new_trail = max(new_trail, prev_trail)
            else:
                new_trail = mid + TRAIL_DISTANCE_PIPS * pip
                if prev_trail is not None and TRAIL_NEVER_WIDEN:
                    new_trail = min(new_trail, prev_trail)
            should_update = False
            if side == "buy" and (prev_trail is None or new_trail > prev_trail):
                should_update = True
            elif side == "sell" and (prev_trail is None or new_trail < prev_trail):
                should_update = True
            if should_update:
                try:
                    adapter.update_position_stop_loss(position_id, profile.symbol, round(new_trail, 3))
                    store.update_trade(trade_id, {"breakeven_sl_price": round(new_trail, 5)})
                    return {
                        "action": "trail_update",
                        "reason": f"trail SL -> {new_trail:.3f} (profit={profit_pips:.1f}p)",
                    }
                except Exception as e:
                    return {"action": "error", "reason": f"trail update error: {e}"}

    return {"action": "none", "reason": "no exit condition met"}


# ===================================================================
#  Dashboard helpers
# ===================================================================

def report_phase3_session(now_utc: datetime | None = None) -> dict:
    """Report current session classification."""
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    session = classify_session(now_utc)
    return {
        "name": "Active Session",
        "value": session or "none",
        "ok": session is not None,
        "detail": f"UTC {now_utc.strftime('%H:%M')} | day={now_utc.strftime('%A')}",
    }


def report_phase3_strategy(session: str | None) -> dict:
    """Report active strategy for this session."""
    if session == "tokyo":
        return {"name": "Active Strategy", "value": "V14 Mean Reversion", "ok": True, "detail": "Tokyo session"}
    elif session == "london":
        return {"name": "Active Strategy", "value": "London V2 (stub)", "ok": False, "detail": "Not implemented"}
    elif session == "ny":
        return {"name": "Active Strategy", "value": "NY V5 (stub)", "ok": False, "detail": "Not implemented"}
    return {"name": "Active Strategy", "value": "none", "ok": False, "detail": "No session"}


def report_phase3_regime(regime: str) -> dict:
    """Report BB width regime."""
    return {
        "name": "V14 Regime",
        "value": regime,
        "ok": regime == "ranging",
        "detail": f"BB({BB_PERIOD}, {BB_STD}) width P{int(BB_WIDTH_RANGING_PCT*100)}",
    }


def report_phase3_adx(adx_val: float) -> dict:
    """Report ADX status."""
    return {
        "name": "V14 ADX",
        "value": f"{adx_val:.1f}",
        "ok": adx_val < ADX_MAX,
        "detail": f"max {ADX_MAX}",
    }


def report_phase3_atr(atr_val: float) -> dict:
    """Report ATR status."""
    atr_pips = atr_val / PIP_SIZE
    return {
        "name": "V14 ATR",
        "value": f"{atr_pips:.1f}p",
        "ok": atr_val < ATR_MAX,
        "detail": f"max {ATR_MAX/PIP_SIZE:.0f}p",
    }
