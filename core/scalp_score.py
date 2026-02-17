"""Scalp Direction Score engine.

Combines sentiment, microstructure, price action, and momentum into a single
directional score from -10 to +10 for M1-M5 scalping.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .indicators import ema, rsi, session_vwap, stochastic


@dataclass
class LayerDetail:
    name: str
    score: float
    max_score: float
    components: dict[str, Any]


@dataclass
class ScalpScoreResult:
    final_score: float
    direction: str  # "LONG", "SHORT", "NEUTRAL"
    confidence: str  # "HIGH", "MEDIUM", "LOW"
    kill_switch: bool
    kill_reason: str | None
    layers: list[dict[str, Any]]
    timestamp: str


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _safe_float(v: Any) -> float | None:
    try:
        f = float(v)
        return f if pd.notna(f) else None
    except (TypeError, ValueError):
        return None


class ScalpScore:
    """Calculate the scalp direction score from market data."""

    @staticmethod
    def calculate(
        *,
        df: pd.DataFrame,
        tick_bid: float,
        tick_ask: float,
        order_book_snapshots: list[dict],
        position_book_snapshots: list[dict],
        pip_size: float = 0.01,
        timeframe: str = "M1",
        timestamp_iso: str = "",
    ) -> ScalpScoreResult:
        spread = tick_ask - tick_bid
        spread_pips = spread / pip_size
        mid = (tick_bid + tick_ask) / 2.0

        close = df["close"]
        n = len(df)

        # Pre-compute indicators
        rsi7 = rsi(close, 7)
        rsi7_val = _safe_float(rsi7.iloc[-1]) if n > 7 else None
        rsi7_prev = _safe_float(rsi7.iloc[-2]) if n > 8 else None

        stoch_k, stoch_d = stochastic(df, 5, 3, 3) if n >= 10 else (pd.Series(dtype=float), pd.Series(dtype=float))
        stoch_k_val = _safe_float(stoch_k.iloc[-1]) if len(stoch_k) > 0 else None
        stoch_d_val = _safe_float(stoch_d.iloc[-1]) if len(stoch_d) > 0 else None
        stoch_k_prev = _safe_float(stoch_k.iloc[-2]) if len(stoch_k) > 1 else None
        stoch_d_prev = _safe_float(stoch_d.iloc[-2]) if len(stoch_d) > 1 else None

        svwap = session_vwap(df)
        svwap_val = _safe_float(svwap.iloc[-1]) if len(svwap) > 0 else None
        svwap_prev = _safe_float(svwap.iloc[-5]) if len(svwap) > 5 else None

        vol = df.get("tick_volume")
        avg_vol_20 = float(vol.tail(20).mean()) if vol is not None and len(vol) >= 20 else None

        from .indicators import atr as calc_atr
        atr_series = calc_atr(df, 14)
        atr_val = _safe_float(atr_series.iloc[-1]) if len(atr_series) > 0 else None
        atr_pips = atr_val / pip_size if atr_val else None

        # ─── Kill switch ───
        kill_switch = False
        kill_reason = None
        if spread_pips > 4.0:
            kill_switch = True
            kill_reason = f"Wide spread ({spread_pips:.1f} pips)"

        # ─── Layer 1: Sentiment (±3.0) ───
        sent_score = 0.0
        sent_components: dict[str, Any] = {}
        if position_book_snapshots:
            latest_pb = position_book_snapshots[-1]
            buckets = latest_pb.get("positionBook", {}).get("buckets", [])
            total_long = sum(float(b.get("longCountPercent", 0)) for b in buckets)
            total_short = sum(float(b.get("shortCountPercent", 0)) for b in buckets)
            total = total_long + total_short
            if total > 0:
                long_pct = total_long / total
                # Contrarian: crowd mostly long -> bearish signal
                ratio = long_pct - 0.5  # positive = crowd long
                sent_score += _clamp(-ratio * 4.0, -1.5, 1.5)  # contrarian
                sent_components["position_ratio"] = round(long_pct, 3)
                sent_components["contrarian_score"] = round(-ratio * 4.0, 2)

            # Position shift (if we have 2+ snapshots)
            if len(position_book_snapshots) >= 2:
                prev_pb = position_book_snapshots[-2]
                prev_buckets = prev_pb.get("positionBook", {}).get("buckets", [])
                prev_long = sum(float(b.get("longCountPercent", 0)) for b in prev_buckets)
                prev_short = sum(float(b.get("shortCountPercent", 0)) for b in prev_buckets)
                prev_total = prev_long + prev_short
                if prev_total > 0 and total > 0:
                    shift = (total_long / total) - (prev_long / prev_total)
                    # Crowd shifting long -> bearish contrarian
                    sent_score += _clamp(-shift * 10.0, -0.75, 0.75)
                    sent_components["position_shift"] = round(shift, 4)

        if order_book_snapshots:
            latest_ob = order_book_snapshots[-1]
            buckets = latest_ob.get("orderBook", {}).get("buckets", [])
            # Stop cluster detection: look for concentration of stops near price
            near_stops_long = 0.0
            near_stops_short = 0.0
            for b in buckets:
                try:
                    bp = float(b.get("price", 0))
                    dist = abs(bp - mid)
                    if dist < pip_size * 20:
                        near_stops_short += float(b.get("shortCountPercent", 0))
                        near_stops_long += float(b.get("longCountPercent", 0))
                except (TypeError, ValueError):
                    pass
            stop_imbalance = near_stops_long - near_stops_short
            sent_score += _clamp(stop_imbalance * 2.0, -0.75, 0.75)
            sent_components["stop_imbalance"] = round(stop_imbalance, 3)

        sent_score = _clamp(sent_score, -3.0, 3.0)
        sent_components["total"] = round(sent_score, 2)

        # ─── Layer 2: Microstructure (±2.5) ───
        micro_score = 0.0
        micro_components: dict[str, Any] = {}

        # Spread modifier
        if spread_pips < 1.0:
            micro_score += 0.3  # tight spread is neutral-positive
        elif spread_pips > 2.5:
            micro_score -= 0.5  # wide spread dampens
        micro_components["spread_pips"] = round(spread_pips, 2)

        # Volume vs average
        if vol is not None and len(vol) >= 1 and avg_vol_20 is not None and avg_vol_20 > 0:
            cur_vol = float(vol.iloc[-1])
            vol_ratio = cur_vol / avg_vol_20
            # High volume on up candle -> bullish, etc.
            last_close = float(close.iloc[-1])
            last_open = float(df["open"].iloc[-1])
            candle_dir = 1.0 if last_close > last_open else -1.0 if last_close < last_open else 0.0
            vol_signal = _clamp(candle_dir * (vol_ratio - 1.0) * 1.5, -1.0, 1.0)
            micro_score += vol_signal
            micro_components["volume_ratio"] = round(vol_ratio, 2)
            micro_components["volume_signal"] = round(vol_signal, 2)

        # Volume trend (last 5 vs prev 5)
        if vol is not None and len(vol) >= 10:
            recent_vol = float(vol.tail(5).mean())
            prev_vol = float(vol.iloc[-10:-5].mean())
            if prev_vol > 0:
                vol_trend = (recent_vol - prev_vol) / prev_vol
                # Rising volume -> confirms direction
                last_5_dir = 1.0 if float(close.iloc[-1]) > float(close.iloc[-5]) else -1.0
                trend_signal = _clamp(last_5_dir * vol_trend * 1.0, -0.5, 0.5)
                micro_score += trend_signal
                micro_components["volume_trend"] = round(vol_trend, 3)

        micro_score = _clamp(micro_score, -2.5, 2.5)
        micro_components["total"] = round(micro_score, 2)

        # ─── Layer 3: Price Action (±2.5) ───
        pa_score = 0.0
        pa_components: dict[str, Any] = {}

        # Session VWAP position & distance
        if svwap_val is not None:
            vwap_dist = (mid - svwap_val) / pip_size
            if vwap_dist > 0:
                pa_score += _clamp(vwap_dist * 0.1, 0, 0.8)
            else:
                pa_score += _clamp(vwap_dist * 0.1, -0.8, 0)
            pa_components["vwap_distance_pips"] = round(vwap_dist, 1)

            # VWAP slope
            if svwap_prev is not None:
                vwap_slope = (svwap_val - svwap_prev) / pip_size
                pa_score += _clamp(vwap_slope * 0.3, -0.5, 0.5)
                pa_components["vwap_slope_pips"] = round(vwap_slope, 2)

        # Candle structure: consecutive same-direction candles (fade after 3+)
        if n >= 5:
            consec = 0
            last_dir = 0
            for i in range(1, min(6, n)):
                c = float(close.iloc[-i])
                o = float(df["open"].iloc[-i])
                d = 1 if c > o else -1 if c < o else 0
                if d == 0:
                    break
                if last_dir == 0:
                    last_dir = d
                    consec = 1
                elif d == last_dir:
                    consec += 1
                else:
                    break
            if consec >= 3:
                # Fade consecutive candles (contrarian)
                fade = _clamp(-last_dir * (consec - 2) * 0.3, -0.7, 0.7)
                pa_score += fade
                pa_components["consecutive_candles"] = consec * last_dir
                pa_components["fade_score"] = round(fade, 2)

        pa_score = _clamp(pa_score, -2.5, 2.5)
        pa_components["total"] = round(pa_score, 2)

        # ─── Layer 4: Momentum (±1.5) ───
        mom_score = 0.0
        mom_components: dict[str, Any] = {}

        # RSI(7) extremes
        if rsi7_val is not None:
            if rsi7_val > 80:
                mom_score -= _clamp((rsi7_val - 80) * 0.05, 0, 0.5)
            elif rsi7_val < 20:
                mom_score += _clamp((20 - rsi7_val) * 0.05, 0, 0.5)
            mom_components["rsi7"] = round(rsi7_val, 1)

        # RSI(7) divergence (simple: price higher but RSI lower = bearish div)
        if rsi7_val is not None and rsi7_prev is not None and n >= 10:
            price_now = float(close.iloc[-1])
            price_prev = float(close.iloc[-5])
            if price_now > price_prev and rsi7_val < rsi7_prev:
                mom_score -= 0.3  # bearish divergence
                mom_components["rsi_divergence"] = "bearish"
            elif price_now < price_prev and rsi7_val > rsi7_prev:
                mom_score += 0.3  # bullish divergence
                mom_components["rsi_divergence"] = "bullish"

        # Stochastic(5,3,3) crosses
        if stoch_k_val is not None and stoch_d_val is not None and stoch_k_prev is not None and stoch_d_prev is not None:
            # Bullish cross: %K crosses above %D in oversold
            if stoch_k_prev < stoch_d_prev and stoch_k_val > stoch_d_val:
                if stoch_k_val < 30:
                    mom_score += 0.5
                    mom_components["stoch_cross"] = "bullish_oversold"
                else:
                    mom_score += 0.2
                    mom_components["stoch_cross"] = "bullish"
            elif stoch_k_prev > stoch_d_prev and stoch_k_val < stoch_d_val:
                if stoch_k_val > 70:
                    mom_score -= 0.5
                    mom_components["stoch_cross"] = "bearish_overbought"
                else:
                    mom_score -= 0.2
                    mom_components["stoch_cross"] = "bearish"
            mom_components["stoch_k"] = round(stoch_k_val, 1)
            mom_components["stoch_d"] = round(stoch_d_val, 1)

        mom_score = _clamp(mom_score, -1.5, 1.5)
        mom_components["total"] = round(mom_score, 2)

        # ─── Context modifiers ───
        context_mod = 0.0
        if atr_pips is not None:
            if atr_pips < 3.0:
                context_mod = -0.3  # low vol dampener
            elif atr_pips > 20.0:
                context_mod = -0.2  # extreme vol dampener

        # ─── Final ───
        raw = sent_score + micro_score + pa_score + mom_score + context_mod
        final = _clamp(round(raw, 1), -10.0, 10.0)

        if kill_switch:
            direction = "NEUTRAL"
            confidence = "NONE"
        elif abs(final) >= 5.0:
            direction = "LONG" if final > 0 else "SHORT"
            confidence = "HIGH"
        elif abs(final) >= 2.5:
            direction = "LONG" if final > 0 else "SHORT"
            confidence = "MEDIUM"
        elif abs(final) >= 1.0:
            direction = "LONG" if final > 0 else "SHORT"
            confidence = "LOW"
        else:
            direction = "NEUTRAL"
            confidence = "LOW"

        layers = [
            {"name": "Sentiment", "score": round(sent_score, 2), "max": 3.0, "components": sent_components},
            {"name": "Microstructure", "score": round(micro_score, 2), "max": 2.5, "components": micro_components},
            {"name": "Price Action", "score": round(pa_score, 2), "max": 2.5, "components": pa_components},
            {"name": "Momentum", "score": round(mom_score, 2), "max": 1.5, "components": mom_components},
        ]

        return ScalpScoreResult(
            final_score=final,
            direction=direction,
            confidence=confidence,
            kill_switch=kill_switch,
            kill_reason=kill_reason,
            layers=[l for l in layers],
            timestamp=timestamp_iso,
        )
