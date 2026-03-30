from __future__ import annotations

import math
import os
import re
from functools import lru_cache
from typing import Any, Optional

import numpy as np
import pandas as pd

from core.signal_engine import drop_incomplete_last_bar


def compute_bb(m5_df: pd.DataFrame, *, period: int, std: float) -> tuple[float, float, float]:
    close = m5_df["close"].astype(float)
    sma = close.rolling(period).mean()
    sigma = close.rolling(period, min_periods=period).std(ddof=0)
    bb_upper = float(sma.iloc[-1] + std * sigma.iloc[-1])
    bb_lower = float(sma.iloc[-1] - std * sigma.iloc[-1])
    bb_mid = float(sma.iloc[-1])
    return bb_upper, bb_mid, bb_lower


def compute_rsi(series: pd.Series, *, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_atr(df: pd.DataFrame, *, period: int) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


def compute_adx(df: pd.DataFrame, *, period: int) -> float:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).rolling(window=period, min_periods=period).mean() / atr.replace(0.0, np.nan)
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).rolling(window=period, min_periods=period).mean() / atr.replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx = dx.rolling(window=period, min_periods=period).mean()
    val = adx.iloc[-1]
    return float(val) if pd.notna(val) else 0.0


def compute_parabolic_sar(m1_df: pd.DataFrame, *, af_start: float, af_step: float, af_max: float) -> pd.Series:
    high = m1_df["high"].astype(float).values
    low = m1_df["low"].astype(float).values
    n = len(high)
    psar = np.full(n, np.nan)
    af = af_start
    trend = 1
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
                af = min(af + af_step, af_max)
            if low[i] < psar[i]:
                trend = -1
                psar[i] = ep
                ep = low[i]
                af = af_start
        else:
            psar[i] = prev_psar + af * (ep - prev_psar)
            psar[i] = max(psar[i], high[i - 1])
            if i >= 2:
                psar[i] = max(psar[i], high[i - 2])
            if low[i] < ep:
                ep = low[i]
                af = min(af + af_step, af_max)
            if high[i] > psar[i]:
                trend = 1
                psar[i] = ep
                ep = high[i]
                af = af_start
    return pd.Series(psar, index=m1_df.index)


def detect_sar_flip(
    m1_df: pd.DataFrame,
    *,
    lookback: int,
    af_start: float,
    af_step: float,
    af_max: float,
) -> tuple[bool, bool]:
    psar = compute_parabolic_sar(m1_df, af_start=af_start, af_step=af_step, af_max=af_max)
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


def compute_ema(series: pd.Series, *, period: int) -> pd.Series:
    return series.astype(float).ewm(span=period, adjust=False).mean()


def drop_incomplete_tf(df: Optional[pd.DataFrame], tf: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    if "time" in d.columns:
        d["time"] = pd.to_datetime(d["time"], utc=True, errors="coerce")
        d = d.dropna(subset=["time"]).sort_values("time")
    if d.empty:
        return d
    try:
        return drop_incomplete_last_bar(d, tf)  # type: ignore[arg-type]
    except Exception:
        return d


def compute_session_windows(
    now_utc,
    *,
    london_open_fn,
    ny_open_fn,
    lmp_impulse_minutes: int,
):
    d0 = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    london_open_hour = london_open_fn(pd.Timestamp(now_utc))
    ny_open_hour = ny_open_fn(pd.Timestamp(now_utc))
    london_open_dt = d0.replace(hour=int(london_open_hour), minute=0)
    ny_open_dt = d0.replace(hour=int(ny_open_hour), minute=0)
    return {
        "day_start": d0,
        "london_open": london_open_dt,
        "london_arb_end": london_open_dt + pd.Timedelta(minutes=90),
        "lmp_impulse_end": london_open_dt + pd.Timedelta(minutes=lmp_impulse_minutes),
        "london_end": london_open_dt + pd.Timedelta(hours=4),
        "ny_open": ny_open_dt,
        "ny_end": ny_open_dt + pd.Timedelta(hours=3),
    }


def impact_rank(v: object) -> int:
    s = str(v).strip().lower()
    if s in {"high", "h"}:
        return 3
    if s in {"medium", "med", "m"}:
        return 2
    if s in {"low", "l"}:
        return 1
    return 0


@lru_cache(maxsize=16)
def load_news_events_cached_inner(calendar_path: str, impact_floor: str, mtime_ns: int) -> tuple[pd.Timestamp, ...]:
    events: list[pd.Timestamp] = []
    try:
        news_df = pd.read_csv(calendar_path)
    except Exception:
        return tuple()
    floor = impact_rank(impact_floor)
    for _, row in news_df.iterrows():
        try:
            imp = impact_rank(row.get("impact", "high"))
            if imp < floor:
                continue
            d = pd.to_datetime(row.get("date"), utc=True, errors="coerce")
            if pd.isna(d):
                continue
            ts_event = pd.Timestamp(d).normalize()
            if "time_utc" in news_df.columns and pd.notna(row.get("time_utc")):
                t_raw = str(row.get("time_utc")).strip()
                hm = re.match(r"^(\d{1,2}):(\d{2})$", t_raw)
                if hm:
                    hh = int(hm.group(1))
                    mm = int(hm.group(2))
                    ts_event = ts_event + pd.Timedelta(hours=hh, minutes=mm)
                else:
                    continue
            else:
                mins = pd.to_timedelta(row.get("minutes"), unit="m", errors="coerce")
                if pd.isna(mins):
                    continue
                ts_event = ts_event + pd.Timedelta(minutes=int(mins / pd.Timedelta(minutes=1)))
            events.append(ts_event)
        except Exception:
            continue
    events.sort()
    return tuple(events)


def load_news_events_cached(calendar_path: str, impact_floor: str) -> tuple[pd.Timestamp, ...]:
    try:
        mtime_ns = int(os.stat(calendar_path).st_mtime_ns)
    except Exception:
        mtime_ns = -1
    return load_news_events_cached_inner(calendar_path, impact_floor, mtime_ns)


def is_in_news_window(ts_now: datetime, events: tuple[pd.Timestamp, ...], before_min: int, after_min: int) -> bool:
    if not events:
        return False
    now = pd.Timestamp(ts_now).tz_convert("UTC") if pd.Timestamp(ts_now).tzinfo is not None else pd.Timestamp(ts_now, tz="UTC")
    w_start = now - pd.Timedelta(minutes=max(0, int(before_min)))
    w_end = now + pd.Timedelta(minutes=max(0, int(after_min)))
    return any(w_start <= ev <= w_end for ev in events)


def compute_asian_range(
    m1_df: pd.DataFrame,
    london_open_utc_hour: int,
    *,
    pip_size: float,
    range_min_pips: float,
    range_max_pips: float,
) -> tuple[float, float, float, bool]:
    if m1_df is None or m1_df.empty:
        return np.nan, np.nan, 0.0, False
    d = m1_df.copy()
    d["time"] = pd.to_datetime(d["time"], utc=True, errors="coerce")
    d = d.dropna(subset=["time"])
    if d.empty:
        return np.nan, np.nan, 0.0, False
    now = pd.Timestamp(d["time"].iloc[-1])
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")
    day_start = now.floor("D")
    london_open = day_start + pd.Timedelta(hours=london_open_utc_hour)
    w = d[(d["time"] >= day_start) & (d["time"] < london_open)]
    if w.empty:
        return np.nan, np.nan, 0.0, False
    high = float(w["high"].max())
    low = float(w["low"].min())
    pips = (high - low) / pip_size
    is_valid = float(range_min_pips) <= pips <= float(range_max_pips)
    return high, low, pips, bool(is_valid)


def compute_lmp_impulse(
    m1_df: pd.DataFrame,
    session_start_utc: datetime,
    impulse_minutes: int,
    *,
    pip_size: float,
    min_impulse_pips: float,
) -> tuple[float, float, Optional[str]]:
    if m1_df is None or m1_df.empty:
        return np.nan, np.nan, None
    d = m1_df.copy()
    d["time"] = pd.to_datetime(d["time"], utc=True, errors="coerce")
    d = d.dropna(subset=["time"]).sort_values("time")
    sess_start = pd.Timestamp(session_start_utc)
    if sess_start.tzinfo is None:
        sess_start = sess_start.tz_localize("UTC")
    else:
        sess_start = sess_start.tz_convert("UTC")
    end = sess_start + pd.Timedelta(minutes=impulse_minutes)
    w = d[(d["time"] >= sess_start) & (d["time"] < end)]
    if w.empty:
        return np.nan, np.nan, None
    impulse_high = float(w["high"].max())
    impulse_low = float(w["low"].min())
    impulse_pips = (impulse_high - impulse_low) / pip_size
    if impulse_pips < min_impulse_pips:
        return impulse_high, impulse_low, None
    sub_open = float(w["open"].iloc[0])
    sub_close = float(w["close"].iloc[-1])
    t_h = pd.Timestamp(w.loc[w["high"].idxmax(), "time"])
    t_l = pd.Timestamp(w.loc[w["low"].idxmin(), "time"])
    bull = sub_close > sub_open and t_l < t_h
    bear = sub_close < sub_open and t_h < t_l
    direction = "up" if bull else "down" if bear else None
    return impulse_high, impulse_low, direction


def compute_lmp_zone(impulse_high: float, impulse_low: float, direction: str | None, fib_ratio: float) -> tuple[float, float]:
    if direction == "up":
        fib50 = impulse_low + (impulse_high - impulse_low) * fib_ratio
    elif direction == "down":
        fib50 = impulse_high - (impulse_high - impulse_low) * fib_ratio
    else:
        return np.nan, np.nan
    return float(max(fib50, fib50)), float(min(fib50, fib50))


def compute_risk_units(equity: float, risk_pct: float, sl_pips: float, entry_price: float, pip_size: float, *, max_units: int) -> int:
    if equity <= 0 or risk_pct <= 0 or sl_pips <= 0 or entry_price <= 0:
        return 0
    pip_value_per_unit = pip_size / entry_price
    units = (equity * risk_pct) / (sl_pips * pip_value_per_unit)
    units = min(units, float(max_units))
    return int(math.floor(max(0.0, units)))


def account_sizing_value(adapter, *, fallback: float = 100000.0) -> float:
    try:
        acct = adapter.get_account_info()
    except Exception:
        return float(fallback)
    bal = getattr(acct, "balance", None)
    eq = getattr(acct, "equity", None)
    try:
        if bal is not None and float(bal) > 0:
            return float(bal)
    except Exception:
        pass
    try:
        if eq is not None and float(eq) > 0:
            return float(eq)
    except Exception:
        pass
    return float(fallback)


def compute_v44_atr_rank(m5_df: pd.DataFrame, lookback: int) -> Optional[float]:
    if m5_df is None or len(m5_df) < max(20, int(lookback)):
        return None
    tr = pd.concat([
        m5_df["high"].astype(float) - m5_df["low"].astype(float),
        (m5_df["high"].astype(float) - m5_df["close"].astype(float).shift(1)).abs(),
        (m5_df["low"].astype(float) - m5_df["close"].astype(float).shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    if len(atr.dropna()) < max(20, int(lookback)):
        return None
    now = float(atr.iloc[-1])
    window = atr.iloc[-int(lookback):].to_numpy(dtype=float)
    if len(window) == 0:
        return None
    return float(np.count_nonzero(window <= now)) / float(len(window))


def determine_v44_session_mode(
    sdat: dict[str, Any],
    *,
    dual_mode_enabled: bool,
    trend_mode_efficiency_min: float,
    range_mode_efficiency_max: float,
) -> str:
    if not dual_mode_enabled:
        return "trend"
    hist = sdat.get("session_efficiency_history", [])
    if not isinstance(hist, list) or len(hist) < 3:
        return "trend"
    vals = [float(x) for x in hist if isinstance(x, (int, float))]
    if len(vals) < 3:
        return "trend"
    avg_eff = float(sum(vals) / len(vals))
    if avg_eff >= float(trend_mode_efficiency_min):
        return "trend"
    if avg_eff <= float(range_mode_efficiency_max):
        return "range_fade"
    return "neutral"


def v44_news_trend_active_event(
    now_utc: datetime,
    events: tuple[pd.Timestamp, ...],
    delay_minutes: int,
    window_minutes: int,
) -> Optional[pd.Timestamp]:
    if not events:
        return None
    now_ts = pd.Timestamp(now_utc)
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize("UTC")
    else:
        now_ts = now_ts.tz_convert("UTC")
    for ev in events:
        ev_ts = pd.Timestamp(ev)
        if ev_ts.tzinfo is None:
            ev_ts = ev_ts.tz_localize("UTC")
        else:
            ev_ts = ev_ts.tz_convert("UTC")
        start = ev_ts + pd.Timedelta(minutes=max(0, int(delay_minutes)))
        end = start + pd.Timedelta(minutes=max(1, int(window_minutes)))
        if start <= now_ts <= end:
            return ev_ts
    return None


def v44_most_recent_news_event(now_utc: datetime, events: tuple[pd.Timestamp, ...]) -> Optional[pd.Timestamp]:
    if not events:
        return None
    now_ts = pd.Timestamp(now_utc)
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize("UTC")
    else:
        now_ts = now_ts.tz_convert("UTC")
    latest: Optional[pd.Timestamp] = None
    for ev in events:
        ev_ts = pd.Timestamp(ev)
        if ev_ts.tzinfo is None:
            ev_ts = ev_ts.tz_localize("UTC")
        else:
            ev_ts = ev_ts.tz_convert("UTC")
        if ev_ts <= now_ts:
            latest = ev_ts
        else:
            break
    return latest
