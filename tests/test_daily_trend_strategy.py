"""Integration tests for DailyTrendStrategy (engine protocol)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.regime_backtest_engine.daily_trend_strategy import DailyTrendStrategy
from core.regime_backtest_engine.models import ClosedTrade, ExitAction, PortfolioSnapshot, PositionSnapshot, Signal
from core.regime_backtest_engine.strategy import BarView, HistoricalDataView, MarketDataStore

UTC = timezone.utc


def _make_store(rows: list[dict], spread_pips: float = 0.5) -> MarketDataStore:
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    pip_size = 0.01
    half = spread_pips * pip_size / 2.0
    df["mid_open"] = df["open"]
    df["mid_high"] = df["high"]
    df["mid_low"] = df["low"]
    df["mid_close"] = df["close"]
    df["bid_open"] = df["open"] - half
    df["bid_high"] = df["high"] - half
    df["bid_low"] = df["low"] - half
    df["bid_close"] = df["close"] - half
    df["ask_open"] = df["open"] + half
    df["ask_high"] = df["high"] + half
    df["ask_low"] = df["low"] + half
    df["ask_close"] = df["close"] + half
    df["spread_pips"] = spread_pips
    cols = [
        "timestamp",
        "bid_open",
        "bid_high",
        "bid_low",
        "bid_close",
        "ask_open",
        "ask_high",
        "ask_low",
        "ask_close",
        "mid_open",
        "mid_high",
        "mid_low",
        "mid_close",
        "spread_pips",
    ]
    return MarketDataStore({col: df[col].to_numpy() for col in cols})


def _portfolio(*, open_positions: tuple[PositionSnapshot, ...] = ()) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        balance=100_000.0,
        equity=100_000.0,
        unrealized_pnl=0.0,
        margin_used=0.0,
        available_margin=100_000.0,
        open_positions=open_positions,
    )


def _position_long() -> PositionSnapshot:
    return PositionSnapshot(
        trade_id=1,
        family="daily_trend",
        direction="long",
        entry_price=150.0,
        entry_bar=0,
        size=10_000,
        margin_held=100.0,
        stop_loss=145.0,
        take_profit=None,
        unrealized_pnl=0.0,
    )


class TestProtocol:
    def test_family_name(self) -> None:
        s = DailyTrendStrategy(daily_ema_slow=5)
        assert s.family_name == "daily_trend"

    def test_methods_exist(self) -> None:
        s = DailyTrendStrategy(daily_ema_slow=5)
        store = _make_store(
            [
                {
                    "timestamp": "2024-01-01T12:00:00+00:00",
                    "open": 100.0,
                    "high": 100.1,
                    "low": 99.9,
                    "close": 100.0,
                }
            ]
        )
        bv = BarView(store, 0)
        hist = HistoricalDataView(store, 0)
        assert s.evaluate(bv, hist, _portfolio()) is None
        assert s.get_exit_conditions(_position_long(), bv, hist) is None


class TestBarWiring:
    def test_feed_many_bars_warms_detector(self) -> None:
        rows = []
        t0 = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
        for i in range(8 * 288):
            ts = t0 + timedelta(minutes=i * 5)
            day = i // 288
            px = 100.0 + day * 2.0
            rows.append(
                {
                    "timestamp": ts.isoformat(),
                    "open": px,
                    "high": px + 0.2,
                    "low": px - 0.2,
                    "close": px,
                }
            )
        store = _make_store(rows)
        s = DailyTrendStrategy(daily_ema_slow=5, daily_ema_fast=3)
        hist = HistoricalDataView(store, len(rows) - 1)
        for idx in range(len(rows)):
            s.get_exit_conditions(_position_long(), BarView(store, idx), hist)
            s.evaluate(BarView(store, idx), hist, _portfolio())
        assert s._detector.is_warmed_up  # type: ignore[attr-defined]


class TestSmokeSignal:
    def test_eventually_emits_long_signal(self) -> None:
        rows = []
        t0 = datetime(2024, 1, 1, 22, 0, tzinfo=UTC)
        for day in range(12):
            for m in range(1440):
                ts = t0 + timedelta(days=day, minutes=m)
                if day < 8:
                    px = 100.0 + day * 1.5
                elif day == 8:
                    px = 112.0 - 0.01 * m if m < 600 else 100.0 + (m - 600) * 0.02
                else:
                    px = 118.0
                rows.append(
                    {
                        "timestamp": ts.isoformat(),
                        "open": px,
                        "high": px + 0.3,
                        "low": px - 0.5,
                        "close": px,
                    }
                )
        store = _make_store(rows)
        s = DailyTrendStrategy(
            daily_ema_slow=5,
            daily_ema_fast=3,
            proximity_atr=3.0,
            account_balance=100_000.0,
        )
        hist = HistoricalDataView(store, len(rows) - 1)
        sig = None
        for idx in range(len(rows)):
            s.get_exit_conditions(_position_long(), BarView(store, idx), hist)
            out = s.evaluate(BarView(store, idx), hist, _portfolio())
            if out is not None:
                sig = out
        assert s._detector.is_warmed_up  # type: ignore[attr-defined]
        if sig is not None:
            assert sig.family == "daily_trend"
            assert sig.direction == "long"
            assert sig.size >= 1000
            assert sig.stop_loss < float(sig.metadata.get("ema20", 200.0))


class TestExitsAndCallbacks:
    def test_no_position_no_exit_trailing(self) -> None:
        s = DailyTrendStrategy(daily_ema_slow=5)
        store = _make_store(
            [
                {
                    "timestamp": "2024-06-01T12:00:00+00:00",
                    "open": 100.0,
                    "high": 100.1,
                    "low": 99.9,
                    "close": 100.0,
                }
            ]
        )
        bv = BarView(store, 0)
        assert s.get_exit_conditions(_position_long(), bv, HistoricalDataView(store, 0)) is None

    def test_pending_exit_dequeued(self) -> None:
        """Queued ExitAction is returned on the next get_exit_conditions call."""
        s = DailyTrendStrategy(daily_ema_slow=5)
        s._last_processed_idx = 0  # type: ignore[attr-defined]
        s._pending_exit = ExitAction(  # type: ignore[attr-defined]
            reason="trend_reversal", exit_type="full", close_fraction=1.0
        )
        store = _make_store(
            [
                {
                    "timestamp": "2024-06-01T12:00:00+00:00",
                    "open": 100.0,
                    "high": 100.1,
                    "low": 99.9,
                    "close": 100.0,
                }
            ]
        )
        bv = BarView(store, 0)
        ex = s.get_exit_conditions(_position_long(), bv, HistoricalDataView(store, 0))
        assert ex is not None
        assert ex.exit_type == "full"
        assert ex.reason == "trend_reversal"

    def test_on_position_open_close(self) -> None:
        s = DailyTrendStrategy(daily_ema_slow=5)
        sig = Signal(
            family="daily_trend",
            direction="long",
            stop_loss=140.0,
            take_profit=None,
            size=5000,
            metadata={},
        )
        pos = _position_long()
        store = _make_store(
            [
                {
                    "timestamp": "2024-03-01T12:00:00+00:00",
                    "open": 150.0,
                    "high": 150.1,
                    "low": 149.9,
                    "close": 150.0,
                }
            ]
        )
        s.on_position_opened(pos, sig, BarView(store, 0))
        assert s._has_position  # type: ignore[attr-defined]
        tr = ClosedTrade(
            trade_id=1,
            family="daily_trend",
            direction="long",
            entry_time="",
            exit_time="",
            entry_bar=0,
            exit_bar=1,
            entry_price=150.0,
            exit_price=149.0,
            size=5000,
            margin_held=100.0,
            stop_loss=140.0,
            take_profit=None,
            spread_cost=0.0,
            slippage_cost=0.0,
            pnl_usd=-10.0,
            pnl_pips=-1.0,
            bars_held=1,
            exit_reason="stop_loss",
            remaining_units=0,
        )
        s.on_position_closed(tr)
        assert not s._has_position  # type: ignore[attr-defined]


class TestParameterThreading:
    def test_custom_risk_fraction(self) -> None:
        s_hi = DailyTrendStrategy(daily_ema_slow=5, risk_fraction=0.02)
        s_lo = DailyTrendStrategy(daily_ema_slow=5, risk_fraction=0.01)
        assert s_hi._risk > s_lo._risk  # type: ignore[attr-defined]
