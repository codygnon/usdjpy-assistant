from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd

from .context_engine import compute_tf_context
from .indicators import atr, bollinger_bands, macd, rsi, vwap
from .profile import ProfileV1
from .timeframes import Timeframe


@dataclass(frozen=True)
class TaFrameSummary:
    timeframe: Timeframe
    regime: str
    rsi_value: float | None
    rsi_zone: str
    macd_value: float | None
    macd_signal: float | None
    macd_hist: float | None
    atr_value: float | None
    atr_state: str
    price: float | None
    recent_high: float | None
    recent_low: float | None
    summary: str
    # Optional: Bollinger Bands and VWAP (for technical analysis page only; not used in automation)
    bollinger_upper: float | None = None
    bollinger_middle: float | None = None
    bollinger_lower: float | None = None
    vwap_value: float | None = None


def _rsi_zone(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value >= 70:
        return "overbought"
    if value <= 30:
        return "oversold"
    return "neutral"


def _atr_state(series: pd.Series) -> tuple[float | None, str]:
    if series.empty:
        return None, "unknown"
    last = float(series.iloc[-1])
    window = series.dropna().tail(100)
    if window.empty:
        return last, "unknown"
    mean = float(window.mean())
    if mean == 0:
        return last, "unknown"
    ratio = last / mean
    if ratio >= 1.25:
        return last, "elevated"
    if ratio <= 0.75:
        return last, "low"
    return last, "normal"


def _frame_summary_text(tf: Timeframe, regime: str, rsi_z: str, macd_h: float | None, atr_s: str) -> str:
    parts: list[str] = []

    if regime == "bull":
        parts.append("Uptrend")
    elif regime == "bear":
        parts.append("Downtrend")
    elif regime == "sideways":
        parts.append("Sideways")
    else:
        parts.append("Regime unclear")

    if rsi_z == "overbought":
        parts.append("momentum looks stretched")
    elif rsi_z == "oversold":
        parts.append("price looks washed out")
    elif rsi_z == "neutral":
        parts.append("momentum is balanced")

    if macd_h is not None:
        if macd_h > 0:
            parts.append("MACD supports bulls")
        elif macd_h < 0:
            parts.append("MACD supports bears")

    if atr_s == "elevated":
        parts.append("volatility is elevated")
    elif atr_s == "low":
        parts.append("volatility is low")

    text = ", ".join(parts)
    return f"{tf}: {text}." if text else f"{tf}: no clear signal."


def compute_ta_for_tf(profile: ProfileV1, tf: Timeframe, df: pd.DataFrame) -> TaFrameSummary:
    df = df.copy()
    df = df.sort_values("time")
    if df.empty:
        return TaFrameSummary(
            timeframe=tf,
            regime="unknown",
            rsi_value=None,
            rsi_zone="unknown",
            macd_value=None,
            macd_signal=None,
            macd_hist=None,
            atr_value=None,
            atr_state="unknown",
            price=None,
            recent_high=None,
            recent_low=None,
            summary=f"{tf}: no data.",
            bollinger_upper=None,
            bollinger_middle=None,
            bollinger_lower=None,
            vwap_value=None,
        )

    ctx = compute_tf_context(profile, tf, df)
    close = df["close"]

    rsi_series = rsi(close, period=14)
    rsi_val = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else None
    rsi_z = _rsi_zone(rsi_val)

    macd_line, macd_signal, macd_hist = macd(close)
    macd_val = float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else None
    macd_sig = float(macd_signal.iloc[-1]) if not pd.isna(macd_signal.iloc[-1]) else None
    macd_h = float(macd_hist.iloc[-1]) if not pd.isna(macd_hist.iloc[-1]) else None

    atr_series = atr(df, period=14)
    atr_val, atr_s = _atr_state(atr_series)

    price = float(close.iloc[-1])
    recent = df.tail(100)
    recent_high = float(recent["high"].max()) if "high" in recent.columns else None
    recent_low = float(recent["low"].min()) if "low" in recent.columns else None

    bb_upper, bb_middle, bb_lower = bollinger_bands(close, period=20, std_dev=2.0)
    bb_u = float(bb_upper.iloc[-1]) if not pd.isna(bb_upper.iloc[-1]) else None
    bb_m = float(bb_middle.iloc[-1]) if not pd.isna(bb_middle.iloc[-1]) else None
    bb_l = float(bb_lower.iloc[-1]) if not pd.isna(bb_lower.iloc[-1]) else None

    vwap_series = vwap(df)
    vwap_val = float(vwap_series.iloc[-1]) if not vwap_series.empty and not pd.isna(vwap_series.iloc[-1]) else None

    summary = _frame_summary_text(tf, ctx.regime, rsi_z, macd_h, atr_s)

    return TaFrameSummary(
        timeframe=tf,
        regime=ctx.regime,
        rsi_value=rsi_val,
        rsi_zone=rsi_z,
        macd_value=macd_val,
        macd_signal=macd_sig,
        macd_hist=macd_h,
        atr_value=atr_val,
        atr_state=atr_s,
        price=price,
        recent_high=recent_high,
        recent_low=recent_low,
        summary=summary,
        bollinger_upper=bb_u,
        bollinger_middle=bb_m,
        bollinger_lower=bb_l,
        vwap_value=vwap_val,
    )


def compute_ta_multi(profile: ProfileV1, data_by_tf: Dict[Timeframe, pd.DataFrame]) -> Dict[Timeframe, TaFrameSummary]:
    out: Dict[Timeframe, TaFrameSummary] = {}
    for tf, df in data_by_tf.items():
        out[tf] = compute_ta_for_tf(profile, tf, df)
    return out

