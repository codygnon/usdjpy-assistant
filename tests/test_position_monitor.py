from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.assistant.config import RiskConfig, TrailingConfig
from core.assistant.mock_oanda_client import MockOandaClient
from core.assistant.oanda_client import CandleBar
from core.assistant.position_monitor import PositionMonitor
from core.assistant.trade_journal import TradeJournal


def _candles_for_trailing_long() -> list[CandleBar]:
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    lows = [149.55, 149.60, 149.65, 149.70, 149.75, 149.80]
    candles: list[CandleBar] = []
    prev_close = 149.60
    for idx, low in enumerate(lows):
        high = low + 0.05
        close = low + 0.03
        candles.append(
            CandleBar(
                timestamp=base + timedelta(hours=4 * idx),
                open=prev_close,
                high=high,
                low=low,
                close=close,
                volume=100,
                complete=True,
            )
        )
        prev_close = close
    return candles


def _candles_for_trailing_short() -> list[CandleBar]:
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    highs = [150.45, 150.40, 150.35, 150.30, 150.25, 150.20]
    candles: list[CandleBar] = []
    prev_close = 150.35
    for idx, high in enumerate(highs):
        low = high - 0.05
        close = high - 0.03
        candles.append(
            CandleBar(
                timestamp=base + timedelta(hours=4 * idx),
                open=prev_close,
                high=high,
                low=low,
                close=close,
                volume=100,
                complete=True,
            )
        )
        prev_close = close
    return candles


def test_register_trade_adds_to_managed_dict() -> None:
    client = MockOandaClient()
    monitor = PositionMonitor(client, RiskConfig(), TrailingConfig())
    managed = monitor.register_trade("1", 150.0, 149.5, 150.5, 151.0, "long", 100000)
    assert monitor.get_status()["managed_count"] == 1
    assert managed.trade_id == "1"


def test_register_trade_stores_all_fields() -> None:
    client = MockOandaClient()
    monitor = PositionMonitor(client, RiskConfig(), TrailingConfig())
    managed = monitor.register_trade("2", 150.1, 149.7, 150.5, 151.0, "short", 200000, False, "USD_JPY")
    assert managed.entry_price == 150.1
    assert managed.current_stop == 149.7
    assert managed.tp1 == 150.5
    assert managed.tp2 == 151.0
    assert managed.user_provided_stop is False
    assert managed.instrument == "USD_JPY"


def test_adopt_unmanaged_trade_without_stop_sets_stop_and_logs() -> None:
    client = MockOandaClient()
    client.set_price("USD_JPY", 150.00, 150.02)
    client.add_open_trade(trade_id="10", direction="long", units=100000, open_price=150.00, stop_loss=None)
    client.set_candles("USD_JPY", _candles_for_trailing_long())
    monitor = PositionMonitor(client, RiskConfig(), TrailingConfig(enabled=False))

    actions = monitor.check_and_manage("USD_JPY")

    assert any(action.action == "set_stop" for action in actions)
    assert any(action.action == "adopted" for action in actions)
    assert client.get_open_trades("USD_JPY")[0].stop_loss is not None


def test_adopt_unmanaged_trade_with_stop_preserves_stop() -> None:
    client = MockOandaClient()
    client.set_price("USD_JPY", 150.00, 150.02)
    client.add_open_trade(trade_id="11", direction="long", units=100000, open_price=150.00, stop_loss=149.50)
    monitor = PositionMonitor(client, RiskConfig(), TrailingConfig(enabled=False))

    actions = monitor.check_and_manage("USD_JPY")

    assert not any(action.action == "set_stop" for action in actions)
    assert monitor.get_status()["managed_count"] == 1
    assert client.get_open_trades("USD_JPY")[0].stop_loss == 149.50


def test_already_managed_trade_not_readopted() -> None:
    client = MockOandaClient()
    client.set_price("USD_JPY", 150.00, 150.02)
    client.add_open_trade(trade_id="12", direction="long", units=100000, open_price=150.00, stop_loss=149.50)
    monitor = PositionMonitor(client, RiskConfig(), TrailingConfig(enabled=False))
    monitor.register_trade("12", 150.0, 149.5, 150.5, 151.0, "long", 100000)

    actions = monitor.check_and_manage("USD_JPY")

    assert not any(action.action == "adopted" for action in actions)


def test_long_tp1_partial_close_and_breakeven_move() -> None:
    client = MockOandaClient()
    monitor = PositionMonitor(client, RiskConfig(), TrailingConfig(enabled=False))
    monitor.register_trade("20", 150.0, 149.5, 150.5, 151.0, "long", 100000)
    client.add_open_trade(trade_id="20", direction="long", units=100000, open_price=150.00, stop_loss=149.50)

    actions = monitor._check_tp1(monitor._managed["20"], 150.50)

    assert any(action.action == "partial_close_tp1" for action in actions)
    assert monitor._managed["20"].tp1_hit is True
    assert monitor._managed["20"].current_units == 50000
    assert monitor._managed["20"].current_stop == 150.0


def test_short_tp1_partial_close() -> None:
    client = MockOandaClient()
    monitor = PositionMonitor(client, RiskConfig(), TrailingConfig(enabled=False))
    monitor.register_trade("21", 150.0, 150.5, 149.5, 149.0, "short", 100000)
    client.add_open_trade(trade_id="21", direction="short", units=100000, open_price=150.00, stop_loss=150.50)

    actions = monitor._check_tp1(monitor._managed["21"], 149.50)

    assert any(action.action == "partial_close_tp1" for action in actions)
    assert monitor._managed["21"].current_units == 50000


def test_tp1_not_repeated_after_first_hit() -> None:
    client = MockOandaClient()
    monitor = PositionMonitor(client, RiskConfig(), TrailingConfig(enabled=False))
    managed = monitor.register_trade("22", 150.0, 149.5, 150.5, 151.0, "long", 100000)
    managed.tp1_hit = True

    assert monitor._check_tp1(managed, 150.60) == []


def test_tp1_no_action_below_target() -> None:
    client = MockOandaClient()
    monitor = PositionMonitor(client, RiskConfig(), TrailingConfig(enabled=False))
    managed = monitor.register_trade("23", 150.0, 149.5, 150.5, 151.0, "long", 100000)

    assert monitor._check_tp1(managed, 150.40) == []


def test_tp2_partial_close_after_tp1() -> None:
    client = MockOandaClient()
    monitor = PositionMonitor(client, RiskConfig(), TrailingConfig(enabled=False))
    managed = monitor.register_trade("24", 150.0, 149.5, 150.5, 151.0, "long", 100000)
    managed.tp1_hit = True
    managed.current_units = 50000
    client.add_open_trade(trade_id="24", direction="long", units=50000, open_price=150.00, stop_loss=150.0)

    actions = monitor._check_tp2(managed, 151.00)

    assert any(action.action == "partial_close_tp2" for action in actions)
    assert managed.tp2_hit is True
    assert managed.current_units == 25000


def test_tp2_only_after_tp1_done() -> None:
    client = MockOandaClient()
    client.set_price("USD_JPY", 151.00, 151.02)
    client.add_open_trade(trade_id="25", direction="long", units=100000, open_price=150.00, stop_loss=149.50)
    monitor = PositionMonitor(client, RiskConfig(), TrailingConfig(enabled=False))
    monitor.register_trade("25", 150.0, 149.5, 150.5, 151.0, "long", 100000)

    actions = monitor.check_and_manage("USD_JPY")

    assert not any(action.action == "partial_close_tp2" for action in actions)


def test_trailing_tightens_for_long() -> None:
    client = MockOandaClient()
    client.set_candles("USD_JPY", _candles_for_trailing_long())
    monitor = PositionMonitor(client, RiskConfig(), TrailingConfig(enabled=True, lookback_bars=5, atr_buffer=0.5, timeframe="H4"))
    managed = monitor.register_trade("30", 150.0, 149.4, 150.5, 151.0, "long", 100000)
    client.add_open_trade(trade_id="30", direction="long", units=100000, open_price=150.0, stop_loss=149.4)

    actions = monitor._check_trailing(managed, "USD_JPY")

    assert any(action.action == "trail_stop" for action in actions)
    assert managed.current_stop > 149.4


def test_trailing_does_not_widen() -> None:
    client = MockOandaClient()
    client.set_candles("USD_JPY", _candles_for_trailing_long())
    monitor = PositionMonitor(client, RiskConfig(), TrailingConfig(enabled=True, lookback_bars=5, atr_buffer=0.5, timeframe="H4"))
    managed = monitor.register_trade("31", 150.0, 149.9, 150.5, 151.0, "long", 100000)
    client.add_open_trade(trade_id="31", direction="long", units=100000, open_price=150.0, stop_loss=149.9)

    assert monitor._check_trailing(managed, "USD_JPY") == []


def test_trailing_computed_for_short() -> None:
    client = MockOandaClient()
    client.set_candles("USD_JPY", _candles_for_trailing_short())
    monitor = PositionMonitor(client, RiskConfig(), TrailingConfig(enabled=True, lookback_bars=5, atr_buffer=0.5, timeframe="H4"))
    managed = monitor.register_trade("32", 150.0, 150.8, 149.5, 149.0, "short", 100000)
    client.add_open_trade(trade_id="32", direction="short", units=100000, open_price=150.0, stop_loss=150.8)

    actions = monitor._check_trailing(managed, "USD_JPY")

    assert any(action.action == "trail_stop" for action in actions)
    assert managed.current_stop < 150.8


def test_emergency_stop_set_when_oanda_trade_has_no_stop() -> None:
    client = MockOandaClient()
    client.add_open_trade(trade_id="40", direction="long", units=100000, open_price=150.0, stop_loss=None)
    monitor = PositionMonitor(client, RiskConfig(), TrailingConfig(enabled=False))
    managed = monitor.register_trade("40", 150.0, 149.5, 150.5, 151.0, "long", 100000)

    actions = monitor._set_emergency_stop(managed, client.get_open_trades()[0], 150.0)

    assert actions[0].action == "catastrophic_stop"
    assert client.get_open_trades()[0].stop_loss == 149.5


def test_cleanup_removes_closed_trade_and_logs_action() -> None:
    client = MockOandaClient()
    client.set_price("USD_JPY", 150.0, 150.02)
    journal = TradeJournal(":memory:")
    journal.log_trade_opened("50", "USD_JPY", "long", 150.0, 100000, 149.5)
    monitor = PositionMonitor(client, RiskConfig(), TrailingConfig(enabled=False), journal=journal)
    monitor.register_trade("50", 150.0, 149.5, 150.5, 151.0, "long", 100000)

    actions = monitor.check_and_manage("USD_JPY")

    assert any(action.action == "trade_closed" for action in actions)
    assert monitor.get_status()["managed_count"] == 0
    journal.close()
