from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.assistant.cross_asset_dashboard import CrossAssetDashboard
from core.assistant.mock_oanda_client import MockOandaClient
from core.assistant.oanda_client import CandleBar


def _daily_candles(closes: list[float]) -> list[CandleBar]:
    base = datetime(2026, 3, 1, tzinfo=timezone.utc)
    candles: list[CandleBar] = []
    for idx, close in enumerate(closes):
        candles.append(
            CandleBar(
                timestamp=base + timedelta(days=idx),
                open=close,
                high=close + 0.1,
                low=close - 0.1,
                close=close,
                volume=100,
                complete=True,
            )
        )
    return candles


def test_oil_up_over_half_percent_is_bullish() -> None:
    client = MockOandaClient()
    client.set_candles("BCO_USD", _daily_candles([80] * 19 + [80.1, 81.0, 81.5]), "D")
    client.set_candles("EUR_USD", _daily_candles([1.10] * 25), "D")
    reading = CrossAssetDashboard(client).get_reading()
    assert reading.oil_bias.direction == "bullish"


def test_oil_down_over_half_percent_is_bearish() -> None:
    client = MockOandaClient()
    client.set_candles("BCO_USD", _daily_candles([81.5, 81.2, 81.0, 80.8, 80.5, 80.0, 79.8]), "D")
    client.set_candles("EUR_USD", _daily_candles([1.10] * 25), "D")
    reading = CrossAssetDashboard(client).get_reading()
    assert reading.oil_bias.direction == "bearish"


def test_oil_flat_is_neutral() -> None:
    client = MockOandaClient()
    client.set_candles("BCO_USD", _daily_candles([80.0] * 25), "D")
    client.set_candles("EUR_USD", _daily_candles([1.10] * 25), "D")
    reading = CrossAssetDashboard(client).get_reading()
    assert reading.oil_bias.direction == "neutral"


def test_eurusd_down_over_point_three_percent_is_bullish_for_usdjpy() -> None:
    client = MockOandaClient()
    client.set_candles("BCO_USD", _daily_candles([80.0] * 25), "D")
    client.set_candles("EUR_USD", _daily_candles([1.12, 1.115, 1.11, 1.105, 1.10, 1.095, 1.09]), "D")
    reading = CrossAssetDashboard(client).get_reading()
    assert reading.dxy_bias.direction == "bullish"


def test_eurusd_up_over_point_three_percent_is_bearish_for_usdjpy() -> None:
    client = MockOandaClient()
    client.set_candles("BCO_USD", _daily_candles([80.0] * 25), "D")
    client.set_candles("EUR_USD", _daily_candles([1.09, 1.095, 1.10, 1.105, 1.11, 1.115, 1.12]), "D")
    reading = CrossAssetDashboard(client).get_reading()
    assert reading.dxy_bias.direction == "bearish"


def test_eurusd_flat_is_neutral() -> None:
    client = MockOandaClient()
    client.set_candles("BCO_USD", _daily_candles([80.0] * 25), "D")
    client.set_candles("EUR_USD", _daily_candles([1.10] * 25), "D")
    reading = CrossAssetDashboard(client).get_reading()
    assert reading.dxy_bias.direction == "neutral"


def test_combined_bullish_high_confidence() -> None:
    client = MockOandaClient()
    client.set_candles("BCO_USD", _daily_candles([80, 80.2, 80.5, 80.7, 81.0, 81.2, 81.5]), "D")
    client.set_candles("EUR_USD", _daily_candles([1.12, 1.115, 1.11, 1.105, 1.10, 1.095, 1.09]), "D")
    reading = CrossAssetDashboard(client).get_reading()
    assert reading.combined_bias == "bullish"
    assert reading.confidence == "high"


def test_combined_bearish_high_confidence() -> None:
    client = MockOandaClient()
    client.set_candles("BCO_USD", _daily_candles([81.5, 81.2, 81.0, 80.8, 80.5, 80.0, 79.8]), "D")
    client.set_candles("EUR_USD", _daily_candles([1.09, 1.095, 1.10, 1.105, 1.11, 1.115, 1.12]), "D")
    reading = CrossAssetDashboard(client).get_reading()
    assert reading.combined_bias == "bearish"
    assert reading.confidence == "high"


def test_combined_conflicting_low_confidence() -> None:
    client = MockOandaClient()
    client.set_candles("BCO_USD", _daily_candles([80, 80.2, 80.5, 80.7, 81.0, 81.2, 81.5]), "D")
    client.set_candles("EUR_USD", _daily_candles([1.09, 1.095, 1.10, 1.105, 1.11, 1.115, 1.12]), "D")
    reading = CrossAssetDashboard(client).get_reading()
    assert reading.combined_bias == "conflicting"
    assert reading.confidence == "low"


def test_combined_neutral_when_both_neutral() -> None:
    client = MockOandaClient()
    client.set_candles("BCO_USD", _daily_candles([80.0] * 25), "D")
    client.set_candles("EUR_USD", _daily_candles([1.10] * 25), "D")
    reading = CrossAssetDashboard(client).get_reading()
    assert reading.combined_bias == "neutral"


def test_one_neutral_one_directional_is_weak_bias() -> None:
    client = MockOandaClient()
    client.set_candles("BCO_USD", _daily_candles([80.0] * 25), "D")
    client.set_candles("EUR_USD", _daily_candles([1.12, 1.115, 1.11, 1.105, 1.10, 1.095, 1.09]), "D")
    reading = CrossAssetDashboard(client).get_reading()
    assert reading.combined_bias == "bullish"
    assert reading.confidence == "low"
