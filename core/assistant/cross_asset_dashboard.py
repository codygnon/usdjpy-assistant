from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class AssetBias:
    instrument: str
    direction: str
    five_day_return: float
    twenty_day_return: float
    current_price: float
    last_updated: datetime


@dataclass(frozen=True)
class DashboardReading:
    oil_bias: AssetBias
    dxy_bias: AssetBias
    combined_bias: str
    usdjpy_implication: str
    confidence: str
    timestamp: datetime


class CrossAssetDashboard:
    """Cross-asset bias dashboard for USDJPY."""

    def __init__(self, client):
        self._client = client

    def get_reading(self) -> DashboardReading:
        oil_bias = self._get_oil_bias()
        dxy_bias = self._get_dxy_bias()
        oil_dir = oil_bias.direction
        dxy_dir = dxy_bias.direction

        if oil_dir == "bullish" and dxy_dir == "bullish":
            combined = "bullish"
            confidence = "high"
            implication = "OIL up and USD strong both support USDJPY longs."
        elif oil_dir == "bearish" and dxy_dir == "bearish":
            combined = "bearish"
            confidence = "high"
            implication = "OIL down and USD soft both support USDJPY shorts."
        elif oil_dir == "neutral" and dxy_dir == "neutral":
            combined = "neutral"
            confidence = "low"
            implication = "No clear macro direction. Trade technicals only."
        elif oil_dir == "neutral" or dxy_dir == "neutral":
            active = oil_dir if oil_dir != "neutral" else dxy_dir
            combined = active
            confidence = "low"
            implication = f"Weak {active} bias — only one macro factor is aligned."
        else:
            combined = "conflicting"
            confidence = "low"
            implication = f"Conflict: oil says {oil_dir}, DXY proxy says {dxy_dir}. Reduce size or skip."

        return DashboardReading(
            oil_bias=oil_bias,
            dxy_bias=dxy_bias,
            combined_bias=combined,
            usdjpy_implication=implication,
            confidence=confidence,
            timestamp=datetime.now(timezone.utc),
        )

    def _get_oil_bias(self) -> AssetBias:
        try:
            candles = [candle for candle in self._client.get_candles("BCO_USD", "D", 25) if candle.complete]
            if len(candles) < 6:
                raise ValueError("insufficient candles")
            current = float(candles[-1].close)
            five_ago = float(candles[-6].close)
            twenty_ago = float(candles[-21].close) if len(candles) >= 21 else float(candles[0].close)
            five_day = (current - five_ago) / five_ago
            twenty_day = (current - twenty_ago) / twenty_ago
            if five_day > 0.005:
                direction = "bullish"
            elif five_day < -0.005:
                direction = "bearish"
            else:
                direction = "neutral"
            return AssetBias("BCO_USD", direction, five_day, twenty_day, current, datetime.now(timezone.utc))
        except Exception:
            return AssetBias("BCO_USD", "neutral", 0.0, 0.0, 0.0, datetime.now(timezone.utc))

    def _get_dxy_bias(self) -> AssetBias:
        try:
            candles = [candle for candle in self._client.get_candles("EUR_USD", "D", 25) if candle.complete]
            if len(candles) < 6:
                raise ValueError("insufficient candles")
            current = float(candles[-1].close)
            five_ago = float(candles[-6].close)
            twenty_ago = float(candles[-21].close) if len(candles) >= 21 else float(candles[0].close)
            eur_five_day = (current - five_ago) / five_ago
            eur_twenty_day = (current - twenty_ago) / twenty_ago
            if eur_five_day < -0.003:
                direction = "bullish"
            elif eur_five_day > 0.003:
                direction = "bearish"
            else:
                direction = "neutral"
            return AssetBias(
                "DXY_proxy",
                direction,
                -eur_five_day,
                -eur_twenty_day,
                (1.0 / current) if current != 0 else 0.0,
                datetime.now(timezone.utc),
            )
        except Exception:
            return AssetBias("DXY_proxy", "neutral", 0.0, 0.0, 0.0, datetime.now(timezone.utc))

    def format_reading(self, reading: DashboardReading) -> str:
        o = reading.oil_bias
        d = reading.dxy_bias
        return (
            "\n"
            "══════════════════════════════════════════════════\n"
            "           CROSS-ASSET MACRO DASHBOARD\n"
            "══════════════════════════════════════════════════\n"
            f"  Oil (BCO/USD):  {o.direction.upper():>10}   5D: {o.five_day_return:+.2%}   20D: {o.twenty_day_return:+.2%}\n"
            f"  DXY (proxy):    {d.direction.upper():>10}   5D: {d.five_day_return:+.2%}   20D: {d.twenty_day_return:+.2%}\n"
            "──────────────────────────────────────────────────\n"
            f"  COMBINED BIAS:  {reading.combined_bias.upper()}  ({reading.confidence} confidence)\n\n"
            f"  {reading.usdjpy_implication}\n"
            "──────────────────────────────────────────────────\n"
            f"  Updated: {reading.timestamp.strftime('%Y-%m-%d %H:%M UTC')}\n"
            "══════════════════════════════════════════════════\n"
        )
