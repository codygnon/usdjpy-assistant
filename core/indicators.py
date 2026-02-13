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
    swing_window: int = 5,
) -> tuple[bool, bool, dict]:
    """Detect price-RSI divergence on OHLC data.

    Args:
        df: DataFrame with columns 'high', 'low', 'close'
        rsi_period: RSI calculation period (default 14)
        lookback_bars: Number of bars to analyze for divergence (default 50)
        swing_window: Window for detecting swing highs/lows (default 5)

    Returns:
        (has_bearish_divergence, has_bullish_divergence, details_dict)

    Bearish divergence: price makes higher high, RSI makes lower high (trend weakening)
    Bullish divergence: price makes lower low, RSI makes higher low (trend strengthening)
    """
    if df is None or len(df) < lookback_bars:
        return False, False, {"error": "insufficient data"}

    # Use last N bars for analysis
    analysis_df = df.tail(lookback_bars).copy()
    if len(analysis_df) < swing_window * 2 + 1:
        return False, False, {"error": "insufficient data for swing detection"}

    # Calculate RSI
    rsi_values = rsi(analysis_df["close"], rsi_period)
    analysis_df = analysis_df.copy()
    analysis_df["rsi"] = rsi_values

    # Find swing highs in price (local maxima in 'high' column)
    price_swing_highs = []
    for i in range(swing_window, len(analysis_df) - swing_window):
        window_highs = analysis_df["high"].iloc[i - swing_window : i + swing_window + 1]
        center_high = analysis_df["high"].iloc[i]
        if center_high == window_highs.max():
            price_swing_highs.append((i, center_high, analysis_df["rsi"].iloc[i]))

    # Find swing lows in price (local minima in 'low' column)
    price_swing_lows = []
    for i in range(swing_window, len(analysis_df) - swing_window):
        window_lows = analysis_df["low"].iloc[i - swing_window : i + swing_window + 1]
        center_low = analysis_df["low"].iloc[i]
        if center_low == window_lows.min():
            price_swing_lows.append((i, center_low, analysis_df["rsi"].iloc[i]))

    has_bearish = False
    has_bullish = False
    details = {
        "price_swing_highs": len(price_swing_highs),
        "price_swing_lows": len(price_swing_lows),
        "bearish_divergence": None,
        "bullish_divergence": None,
    }

    # Detect bearish divergence: price higher high + RSI lower high
    # Compare most recent swing high with previous swing highs
    if len(price_swing_highs) >= 2:
        recent_high = price_swing_highs[-1]  # (index, price_high, rsi_at_high)
        for prev_high in price_swing_highs[:-1]:
            # Price made higher high
            if recent_high[1] > prev_high[1]:
                # RSI made lower high
                if recent_high[2] < prev_high[2]:
                    has_bearish = True
                    details["bearish_divergence"] = {
                        "prev_price": float(prev_high[1]),
                        "prev_rsi": float(prev_high[2]),
                        "recent_price": float(recent_high[1]),
                        "recent_rsi": float(recent_high[2]),
                    }
                    break

    # Detect bullish divergence: price lower low + RSI higher low
    # Compare most recent swing low with previous swing lows
    if len(price_swing_lows) >= 2:
        recent_low = price_swing_lows[-1]  # (index, price_low, rsi_at_low)
        for prev_low in price_swing_lows[:-1]:
            # Price made lower low
            if recent_low[1] < prev_low[1]:
                # RSI made higher low
                if recent_low[2] > prev_low[2]:
                    has_bullish = True
                    details["bullish_divergence"] = {
                        "prev_price": float(prev_low[1]),
                        "prev_rsi": float(prev_low[2]),
                        "recent_price": float(recent_low[1]),
                        "recent_rsi": float(recent_low[2]),
                    }
                    break

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

