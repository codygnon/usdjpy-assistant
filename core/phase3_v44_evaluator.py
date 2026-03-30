from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd


def _compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.astype(float).ewm(span=period, adjust=False).mean()


def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


def compute_v44_h1_trend(h1_df: pd.DataFrame, ema_fast_period: int = 20, ema_slow_period: int = 50) -> str | None:
    if h1_df is None or len(h1_df) < max(ema_fast_period, ema_slow_period) + 2:
        return None
    close = h1_df["close"].astype(float)
    ema_fast = _compute_ema(close, ema_fast_period)
    ema_slow = _compute_ema(close, ema_slow_period)
    if float(ema_fast.iloc[-1]) > float(ema_slow.iloc[-1]):
        return "up"
    if float(ema_fast.iloc[-1]) < float(ema_slow.iloc[-1]):
        return "down"
    return None


def compute_v44_m5_slope(
    m5_df: pd.DataFrame,
    slope_bars: int,
    ema_fast_period: int = 9,
    ema_slow_period: int = 21,
) -> float:
    if m5_df is None or len(m5_df) < max(ema_fast_period, ema_slow_period) + slope_bars + 2:
        return 0.0
    ema_fast = _compute_ema(m5_df["close"].astype(float), ema_fast_period)
    now = float(ema_fast.iloc[-1])
    prev = float(ema_fast.iloc[-1 - slope_bars])
    return (now - prev) / 0.01 / max(1, slope_bars)


def classify_v44_strength(slope: float, is_london: bool, *, london_threshold: float = 0.5, normal_threshold: float = 0.5) -> str:
    threshold = london_threshold if is_london else normal_threshold
    return "strong" if abs(slope) >= threshold else "normal"


def compute_v44_atr_pct_filter(
    m5_df: pd.DataFrame,
    *,
    enabled: bool = True,
    cap: float = 0.67,
    lookback: int = 200,
) -> bool:
    if not enabled or m5_df is None or len(m5_df) < int(lookback):
        return True
    atr = _compute_atr(m5_df, 14).dropna()
    if len(atr) < 20:
        return True
    current = float(atr.iloc[-1])
    look = atr.iloc[-int(lookback):]
    cutoff = float(np.quantile(look, float(cap)))
    return current <= cutoff


def compute_v44_sl(side: str, m5_df: pd.DataFrame, entry_price: float, pip_size: float) -> float:
    if m5_df is None or len(m5_df) < 7:
        raw_pips = 7.0
    else:
        w = m5_df.tail(6)
        if side == "buy":
            raw_sl = float(w["low"].min()) - 1.5 * pip_size
            raw_pips = (entry_price - raw_sl) / pip_size
        else:
            raw_sl = float(w["high"].max()) + 1.5 * pip_size
            raw_pips = (raw_sl - entry_price) / pip_size
    sl_pips = max(7.0, min(9.0, raw_pips))
    return entry_price - sl_pips * pip_size if side == "buy" else entry_price + sl_pips * pip_size


def evaluate_v44_entry(
    h1_df: pd.DataFrame,
    m5_df: pd.DataFrame,
    tick,
    pip_size: float,
    session: str,
    session_state: dict,
    *,
    now_utc: Optional[datetime] = None,
    max_entries_per_day: int = 7,
    session_stop_losses: int = 3,
    h1_ema_fast_period: int = 20,
    h1_ema_slow_period: int = 50,
    m5_ema_fast_period: int = 9,
    m5_ema_slow_period: int = 21,
    slope_bars: int = 4,
    strong_slope_threshold: float = 0.5,
    weak_slope_threshold: float = 0.2,
    min_body_pips: float = 1.5,
    atr_pct_filter_enabled: bool = True,
    atr_pct_cap: float = 0.67,
    atr_pct_lookback: int = 200,
) -> tuple[Optional[str], str, str]:
    if int(max_entries_per_day) > 0 and int(session_state.get("trade_count", 0)) >= int(max_entries_per_day):
        return None, "normal", "v44: max entries/day reached"
    if int(session_state.get("consecutive_losses", 0)) >= int(session_stop_losses):
        return None, "normal", "v44: consecutive loss stop"

    cooldown_until = session_state.get("cooldown_until")
    if cooldown_until:
        try:
            ts_now = now_utc or datetime.now(timezone.utc)
            if ts_now < datetime.fromisoformat(str(cooldown_until)):
                return None, "normal", "v44: cooldown active"
        except Exception:
            pass

    trend = compute_v44_h1_trend(h1_df, h1_ema_fast_period, h1_ema_slow_period)
    if trend is None:
        return None, "normal", "v44: no H1 trend"
    if not compute_v44_atr_pct_filter(m5_df, enabled=atr_pct_filter_enabled, cap=atr_pct_cap, lookback=atr_pct_lookback):
        return None, "normal", "v44: ATR percentile block"
    if m5_df is None or len(m5_df) < max(m5_ema_fast_period, m5_ema_slow_period) + 4:
        return None, "normal", "v44: insufficient M5"

    close = m5_df["close"].astype(float)
    open_ = m5_df["open"].astype(float)
    ema_fast = _compute_ema(close, m5_ema_fast_period)
    ema_slow = _compute_ema(close, m5_ema_slow_period)
    body_pips = abs(float(close.iloc[-1]) - float(open_.iloc[-1])) / pip_size
    slope = compute_v44_m5_slope(m5_df, slope_bars, m5_ema_fast_period, m5_ema_slow_period)
    abs_slope = abs(slope)
    strength = "strong" if abs_slope > float(strong_slope_threshold) else "normal" if abs_slope > float(weak_slope_threshold) else "weak"
    bullish_bar = float(close.iloc[-1]) > float(open_.iloc[-1]) and body_pips >= min_body_pips
    bearish_bar = float(close.iloc[-1]) < float(open_.iloc[-1]) and body_pips >= min_body_pips

    if trend == "up" and float(ema_fast.iloc[-1]) > float(ema_slow.iloc[-1]) and bullish_bar and slope > 0:
        return "buy", strength, "v44: H1 up + M5 strong bullish momentum"
    if trend == "down" and float(ema_fast.iloc[-1]) < float(ema_slow.iloc[-1]) and bearish_bar and slope < 0:
        return "sell", strength, "v44: H1 down + M5 strong bearish momentum"
    return None, strength, "v44: directional conditions not met"
