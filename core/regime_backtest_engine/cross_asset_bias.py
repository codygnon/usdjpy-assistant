"""
Cross-asset directional bias for USDJPY from ``CrossAssetDataLoader`` inputs.

All lookups are causal via the loader. ADX(14) uses Wilder smoothing.
``compute_ema`` from ``synthetic_bars`` is used as an auxiliary filter on
USDJPY daily closes (see ``_ema14_last``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from .cross_asset_data import CrossAssetDataLoader
from .synthetic_bars import compute_ema

UTC = timezone.utc


@dataclass
class BiasReading:
    """Complete bias state at a point in time."""

    timestamp: datetime

    oil_signal: float
    dxy_signal: float
    gold_signal: float
    silver_signal: float

    raw_score: float
    bias: str

    oil_dxy_agree: bool
    conflict_action: str

    adx_value: float
    adxr_value: float
    regime: str

    size_multiplier: float

    oil_sma_20: float | None
    eurusd_sma_20: float | None
    gold_sma_20: float | None
    silver_sma_20: float | None
    brent_is_big_move: bool


def compute_adx(
    daily_highs: list[float],
    daily_lows: list[float],
    daily_closes: list[float],
    period: int = 14,
) -> list[Optional[float]]:
    """
    Wilder ADX(period) from daily OHLC. None until the first ADX (after 2*period-1 bars).
    """
    n = len(daily_highs)
    if n != len(daily_lows) or n != len(daily_closes):
        raise ValueError("daily_highs, daily_lows, daily_closes must match length")
    if period < 2:
        raise ValueError("period must be >= 2")
    if n == 0:
        return []

    tr: list[float] = [0.0] * n
    plus_dm: list[float] = [0.0] * n
    minus_dm: list[float] = [0.0] * n

    tr[0] = float(daily_highs[0]) - float(daily_lows[0])
    for i in range(1, n):
        h, l = float(daily_highs[i]), float(daily_lows[i])
        c_prev = float(daily_closes[i - 1])
        up_move = float(daily_highs[i]) - float(daily_highs[i - 1])
        down_move = float(daily_lows[i - 1]) - float(daily_lows[i])
        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0.0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0.0
        tr[i] = max(h - l, abs(h - c_prev), abs(l - c_prev))

    tr_s = [math.nan] * n
    p_s = [math.nan] * n
    m_s = [math.nan] * n

    tr_s[period - 1] = sum(tr[0:period])
    p_s[period - 1] = sum(plus_dm[0:period])
    m_s[period - 1] = sum(minus_dm[0:period])

    for i in range(period, n):
        tr_s[i] = tr_s[i - 1] - tr_s[i - 1] / period + tr[i]
        p_s[i] = p_s[i - 1] - p_s[i - 1] / period + plus_dm[i]
        m_s[i] = m_s[i - 1] - m_s[i - 1] / period + minus_dm[i]

    dx: list[Optional[float]] = [None] * n
    for i in range(period - 1, n):
        if tr_s[i] == 0 or math.isnan(tr_s[i]):
            continue
        pdi = 100.0 * p_s[i] / tr_s[i]
        mdi = 100.0 * m_s[i] / tr_s[i]
        denom = pdi + mdi
        dx[i] = 0.0 if denom == 0 else 100.0 * abs(pdi - mdi) / denom

    adx: list[Optional[float]] = [None] * n
    first_adx_i = 2 * period - 2
    if first_adx_i >= n:
        return adx
    slice_dx = [dx[j] for j in range(period - 1, period - 1 + period)]
    if any(v is None for v in slice_dx):
        return adx
    adx[first_adx_i] = sum(float(v) for v in slice_dx) / period

    for i in range(first_adx_i + 1, n):
        if dx[i] is None:
            continue
        prev = adx[i - 1]
        if prev is None:
            continue
        adx[i] = (prev * (period - 1) + float(dx[i])) / period

    return adx


def _ema14_last(closes: list[float]) -> float | None:
    """Last finite EMA(14) close; uses shared ``compute_ema``."""
    if len(closes) < 14:
        return None
    series = compute_ema(closes, 14)
    if not series:
        return None
    v = series[-1]
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return float(v)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _oil_signal_from_bars(bars: list[dict[str, Any]]) -> tuple[float, float | None, bool]:
    """
    Returns (oil_signal, oil_sma_20, brent_is_big_move) before conflict resolution.
    """
    if len(bars) < 21:
        return 0.0, None, False

    last20 = bars[-21:-1]
    current = bars[-1]
    closes = [float(b["close"]) for b in last20]
    oil_sma_20 = _mean(closes)
    brent_close = float(current["close"])
    ranges = [float(b["high"]) - float(b["low"]) for b in last20]
    avg_range = _mean(ranges)
    cur_range = float(current["high"]) - float(current["low"])

    uptrend = brent_close > oil_sma_20
    downtrend = brent_close < oil_sma_20
    big = avg_range > 0 and cur_range > 2.0 * avg_range
    move_up = float(current["close"]) >= float(current["open"])
    move_down = float(current["close"]) < float(current["open"])

    if uptrend and big and move_up:
        return 1.5, oil_sma_20, True
    if uptrend and big and move_down:
        return 0.0, oil_sma_20, True
    if uptrend and not big:
        return 1.0, oil_sma_20, False
    if downtrend and big and move_down:
        return -1.5, oil_sma_20, True
    if downtrend and big and move_up:
        return 0.0, oil_sma_20, True
    if downtrend and not big:
        return -1.0, oil_sma_20, False
    return 0.0, oil_sma_20, False


def _dxy_signal_from_bars(bars: list[dict[str, Any]]) -> tuple[float, float | None]:
    if len(bars) < 21:
        return 0.0, None
    last20 = bars[-21:-1]
    cur = bars[-1]
    sma = _mean([float(b["close"]) for b in last20])
    close = float(cur["close"])
    if close < sma:
        return 1.0, sma
    if close > sma:
        return -1.0, sma
    return 0.0, sma


def _metal_signal_from_bars(bars: list[dict[str, Any]]) -> tuple[float, float | None]:
    if len(bars) < 21:
        return 0.0, None
    last20 = bars[-21:-1]
    cur = bars[-1]
    sma = _mean([float(b["close"]) for b in last20])
    close = float(cur["close"])
    if close < sma:
        return 0.5, sma
    if close > sma:
        return -0.5, sma
    return 0.0, sma


def _sign_nonzero(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def _apply_oil_dxy_conflict(
    oil: float,
    dxy: float,
    brent_big_move: bool,
) -> tuple[float, float, bool, str]:
    so, sd = _sign_nonzero(oil), _sign_nonzero(dxy)
    if so == 0 or sd == 0 or so == sd:
        return oil, dxy, True, "FULL"
    if brent_big_move:
        return oil, 0.0, False, "FAVOR_OIL"
    return 0.0, 0.0, False, "SIT_OUT"


def _bias_from_raw(raw: float) -> str:
    if raw >= 2.0:
        return "STRONG_LONG"
    if raw >= 0.5:
        return "MILD_LONG"
    if raw <= -2.0:
        return "STRONG_SHORT"
    if raw <= -0.5:
        return "MILD_SHORT"
    return "NEUTRAL"


def _size_multiplier_for_bias(bias: str, oil: float, dxy: float, gold: float, silver: float) -> float:
    if bias == "NEUTRAL":
        return 0.5
    want = 1 if bias in ("STRONG_LONG", "MILD_LONG") else -1
    agree = 0
    for s in (oil, dxy, gold, silver):
        if s == 0:
            continue
        if want > 0 and s > 0:
            agree += 1
        elif want < 0 and s < 0:
            agree += 1
    if agree >= 4:
        return 1.5
    if agree == 3:
        return 1.0
    return 0.5


def _regime_from_adx(adx: float) -> str:
    if adx >= 25.0:
        return "TRENDING"
    if adx >= 20.0:
        return "WEAK_TREND"
    return "RANGING"


class CrossAssetBias:
    """
    Computes directional bias for USDJPY from cross-asset data.

    Economic logic (research defaults):
    - Oil up → JPY weakness → USDJPY bullish
    - EUR/USD down → USD strong → USDJPY bullish
    - Gold/Silver up → anti-dollar → USDJPY bearish
    """

    def __init__(self, data_loader: CrossAssetDataLoader) -> None:
        self._loader = data_loader
        _probe = compute_ema([float(i) for i in range(28)], 14)
        if not _probe:
            raise RuntimeError("compute_ema unavailable")

    def compute_bias(
        self,
        timestamp: datetime,
        usdjpy_daily_bars: list | None = None,
    ) -> BiasReading:
        ts = timestamp.astimezone(UTC) if timestamp.tzinfo else timestamp.replace(tzinfo=UTC)

        brent_hist = self._loader.get_brent_history(ts, 21)
        oil_sig, oil_sma, brent_big = _oil_signal_from_bars(brent_hist)

        eur_hist = self._loader.get_eurusd_history(ts, 21)
        dxy_sig, eur_sma = _dxy_signal_from_bars(eur_hist)

        gold_hist = self._loader.get_gold_history(ts, 21)
        gold_sig, gold_sma = _metal_signal_from_bars(gold_hist)

        silver_hist = self._loader.get_silver_history(ts, 21)
        silver_sig, silver_sma = _metal_signal_from_bars(silver_hist)

        oil_r, dxy_r, agree, conflict = _apply_oil_dxy_conflict(oil_sig, dxy_sig, brent_big)
        raw = oil_r + dxy_r + gold_sig + silver_sig
        bias = _bias_from_raw(raw)

        highs: list[float] = []
        lows: list[float] = []
        closes: list[float] = []
        if usdjpy_daily_bars is not None and len(usdjpy_daily_bars) >= 28:
            for b in usdjpy_daily_bars:
                highs.append(float(b["high"]))
                lows.append(float(b["low"]))
                closes.append(float(b["close"]))
            _ema14_last(closes)
            adx_series = compute_adx(highs, lows, closes, 14)
            last_adx = next((v for v in reversed(adx_series) if v is not None), None)
            adx_value = float(last_adx) if last_adx is not None else 0.0
            idx_last = len(adx_series) - 1
            adx_14_ago = adx_series[idx_last - 14] if idx_last >= 14 else None
            if adx_14_ago is not None and last_adx is not None:
                adxr_value = (float(last_adx) + float(adx_14_ago)) / 2.0
            else:
                adxr_value = adx_value
            regime = _regime_from_adx(adx_value)
        else:
            adx_value = 0.0
            adxr_value = 0.0
            regime = "RANGING"

        size_mult = _size_multiplier_for_bias(bias, oil_r, dxy_r, gold_sig, silver_sig)

        return BiasReading(
            timestamp=ts,
            oil_signal=oil_r,
            dxy_signal=dxy_r,
            gold_signal=gold_sig,
            silver_signal=silver_sig,
            raw_score=raw,
            bias=bias,
            oil_dxy_agree=agree,
            conflict_action=conflict,
            adx_value=adx_value,
            adxr_value=adxr_value,
            regime=regime,
            size_multiplier=size_mult,
            oil_sma_20=oil_sma,
            eurusd_sma_20=eur_sma,
            gold_sma_20=gold_sma,
            silver_sma_20=silver_sma,
            brent_is_big_move=brent_big,
        )
