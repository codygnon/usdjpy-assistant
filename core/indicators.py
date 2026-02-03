from __future__ import annotations

import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (simple rolling mean of True Range).

    df must have columns: high, low, close.
    """
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (RSI) using Wilder's smoothing."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    avg_loss_safe = avg_loss.replace(0, float("nan")).ffill().fillna(1e-10)
    rs = avg_gain / avg_loss_safe
    rsi = 100 - (100 / (1 + rs))
    return rsi


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, and histogram."""
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands: middle (SMA), upper, lower."""
    middle = sma(series, period)
    std = series.rolling(period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return upper, middle, lower


def vwap(df: pd.DataFrame) -> pd.Series:
    """Volume-weighted average price. Requires columns: high, low, close, and optionally tick_volume or real_volume."""
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    if "tick_volume" in df.columns and df["tick_volume"].notna().any():
        vol = pd.to_numeric(df["tick_volume"], errors="coerce").fillna(0)
    elif "real_volume" in df.columns and df["real_volume"].notna().any():
        vol = pd.to_numeric(df["real_volume"], errors="coerce").fillna(0)
    else:
        vol = pd.Series(1.0, index=df.index)
    cum_tp = (typical * vol).cumsum()
    cum_vol = vol.cumsum()
    return cum_tp / cum_vol.replace(0, float("nan")).ffill().fillna(1e-10)

