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


def detect_rsi_divergence(
    df: pd.DataFrame,
    rsi_period: int = 14,
    lookback_bars: int = 50,
) -> tuple[bool, bool, dict]:
    """Detect price-RSI divergence using rolling window comparison.

    Uses a rolling window approach similar to danger zone:
    - Splits lookback into two halves (reference period and recent period)
    - Compares rolling high/low and RSI between the two periods

    Args:
        df: DataFrame with columns 'high', 'low', 'close'
        rsi_period: RSI calculation period (default 14)
        lookback_bars: Number of bars to analyze for divergence (default 50)

    Returns:
        (has_bearish_divergence, has_bullish_divergence, details_dict)

    Bearish divergence: recent high > reference high, but recent RSI < reference RSI
    Bullish divergence: recent low < reference low, but recent RSI > reference RSI
    """
    if df is None or len(df) < lookback_bars:
        return False, False, {"error": "insufficient data"}

    # Use last N bars for analysis
    analysis_df = df.tail(lookback_bars).copy()
    if len(analysis_df) < 20:  # Need minimum bars for meaningful comparison
        return False, False, {"error": "insufficient data for rolling comparison"}

    # Calculate RSI
    rsi_values = rsi(analysis_df["close"], rsi_period)
    analysis_df = analysis_df.copy()
    analysis_df["rsi"] = rsi_values

    # Split into two halves: reference (older) and recent (newer)
    half = len(analysis_df) // 2
    reference_df = analysis_df.iloc[:half]
    recent_df = analysis_df.iloc[half:]

    has_bearish = False
    has_bullish = False
    details = {
        "method": "rolling_window",
        "lookback_bars": lookback_bars,
        "bearish_divergence": None,
        "bullish_divergence": None,
    }

    # Find rolling high in each period
    ref_high_idx = reference_df["high"].idxmax()
    ref_high_price = float(reference_df.loc[ref_high_idx, "high"])
    ref_high_rsi = float(reference_df.loc[ref_high_idx, "rsi"])

    recent_high_idx = recent_df["high"].idxmax()
    recent_high_price = float(recent_df.loc[recent_high_idx, "high"])
    recent_high_rsi = float(recent_df.loc[recent_high_idx, "rsi"])

    # Find rolling low in each period
    ref_low_idx = reference_df["low"].idxmin()
    ref_low_price = float(reference_df.loc[ref_low_idx, "low"])
    ref_low_rsi = float(reference_df.loc[ref_low_idx, "rsi"])

    recent_low_idx = recent_df["low"].idxmin()
    recent_low_price = float(recent_df.loc[recent_low_idx, "low"])
    recent_low_rsi = float(recent_df.loc[recent_low_idx, "rsi"])

    # Detect bearish divergence: recent high > reference high, but RSI is lower
    if recent_high_price > ref_high_price and recent_high_rsi < ref_high_rsi:
        has_bearish = True
        details["bearish_divergence"] = {
            "ref_price": ref_high_price,
            "ref_rsi": ref_high_rsi,
            "recent_price": recent_high_price,
            "recent_rsi": recent_high_rsi,
        }

    # Detect bullish divergence: recent low < reference low, but RSI is higher
    if recent_low_price < ref_low_price and recent_low_rsi > ref_low_rsi:
        has_bullish = True
        details["bullish_divergence"] = {
            "ref_price": ref_low_price,
            "ref_rsi": ref_low_rsi,
            "recent_price": recent_low_price,
            "recent_rsi": recent_low_rsi,
        }

    return has_bearish, has_bullish, details


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

