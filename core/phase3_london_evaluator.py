from __future__ import annotations

from typing import Optional

import pandas as pd


def _compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.astype(float).ewm(span=period, adjust=False).mean()


def evaluate_london_v2_arb(
    m1_df: pd.DataFrame,
    tick,
    asian_high: float,
    asian_low: float,
    pip_size: float,
    session_state: dict,
    *,
    breakout_buffer_pips: float = 7.0,
    max_trades: int = 1,
) -> tuple[Optional[str], str]:
    if m1_df is None or len(m1_df) < 2:
        return None, "london_arb: insufficient M1"
    if int(session_state.get("arb_trades", 0)) >= int(max_trades):
        return None, "london_arb: max trades reached"
    row = m1_df.iloc[-1]
    close = float(row["close"])
    if close > asian_high + float(breakout_buffer_pips) * pip_size:
        return "buy", "london_arb: breakout above asian high"
    if close < asian_low - float(breakout_buffer_pips) * pip_size:
        return "sell", "london_arb: breakout below asian low"
    return None, "london_arb: no breakout"


def evaluate_london_v2_lmp(
    m1_df: pd.DataFrame,
    m15_df: pd.DataFrame,
    tick,
    impulse_direction: str | None,
    zone_top: float,
    zone_bottom: float,
    pip_size: float,
    session_state: dict,
    *,
    ema_m15_period: int = 20,
    max_trades: int = 1,
) -> tuple[Optional[str], str]:
    if m1_df is None or len(m1_df) < 2 or m15_df is None or len(m15_df) < int(ema_m15_period) + 2:
        return None, "london_lmp: insufficient data"
    if impulse_direction is None:
        return None, "london_lmp: no valid impulse"
    if int(session_state.get("lmp_trades", 0)) >= int(max_trades):
        return None, "london_lmp: max trades reached"

    ema20 = float(_compute_ema(m15_df["close"].astype(float), int(ema_m15_period)).iloc[-1])
    row = m1_df.iloc[-1]
    hi = float(row["high"])
    lo = float(row["low"])
    close = float(row["close"])
    if impulse_direction == "up":
        touched = lo <= zone_top + 0.1 * pip_size
        if touched and close > zone_top and close > ema20:
            return "buy", "london_lmp: bullish pullback confirmation"
    elif impulse_direction == "down":
        touched = hi >= zone_bottom - 0.1 * pip_size
        if touched and close < zone_bottom and close < ema20:
            return "sell", "london_lmp: bearish pullback confirmation"
    return None, "london_lmp: no pullback signal"
