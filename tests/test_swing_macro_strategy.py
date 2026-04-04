"""Tests for swing_macro_strategy.py — engine-protocol integration."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.regime_backtest_engine.models import ClosedTrade, PortfolioSnapshot, PositionSnapshot, Signal
from core.regime_backtest_engine.strategy import BarView, HistoricalDataView, MarketDataStore
from core.regime_backtest_engine.swing_macro_signals import MacroBias, MacroDirection, MacroReading
from core.regime_backtest_engine.swing_macro_strategy import SwingMacroStrategy


class MockMacroSignal:
    def __init__(self, bias: MacroBias = MacroBias.LONG):
        self.bias = bias

    def compute(self, timestamp):
        if self.bias in (MacroBias.LONG, MacroBias.LEAN_LONG):
            oil_dir = MacroDirection.UP
            dxy_dir = (
                MacroDirection.UP if self.bias == MacroBias.LONG else MacroDirection.NEUTRAL
            )
        elif self.bias in (MacroBias.SHORT, MacroBias.LEAN_SHORT):
            oil_dir = MacroDirection.DOWN
            dxy_dir = (
                MacroDirection.DOWN if self.bias == MacroBias.SHORT else MacroDirection.NEUTRAL
            )
        else:
            oil_dir = MacroDirection.NEUTRAL
            dxy_dir = MacroDirection.NEUTRAL

        return MacroReading(
            timestamp=timestamp,
            oil_direction=oil_dir,
            oil_return_5d=0.01
            if oil_dir == MacroDirection.UP
            else -0.01
            if oil_dir == MacroDirection.DOWN
            else 0.0,
            dxy_direction=dxy_dir,
            dxy_return_5d=0.01
            if dxy_dir == MacroDirection.UP
            else -0.01
            if dxy_dir == MacroDirection.DOWN
            else 0.0,
            bias=self.bias,
            tradeable=self.bias != MacroBias.NEUTRAL,
        )


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


def _row(ts: str, o: float, h: float, l: float, c: float) -> dict:
    return {"timestamp": ts, "open": o, "high": h, "low": l, "close": c}


def _portfolio(*, open_positions: tuple[PositionSnapshot, ...] = ()) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        balance=100_000.0,
        equity=100_000.0,
        unrealized_pnl=0.0,
        margin_used=0.0,
        available_margin=100_000.0,
        open_positions=open_positions,
        closed_trade_count=0,
    )


def _make_strategy(bias=MacroBias.LONG, **kwargs) -> SwingMacroStrategy:
    return SwingMacroStrategy(macro_signal=MockMacroSignal(bias=bias), **kwargs)


def _feed_evaluate(strategy: SwingMacroStrategy, store: MarketDataStore, last_idx: int) -> list:
    out = []
    for i in range(last_idx + 1):
        sig = strategy.evaluate(
            BarView(store, i), HistoricalDataView(store, i), _portfolio()
        )
        out.append(sig)
    return out


class TestStrategyInit:
    def test_creates_without_error(self):
        assert _make_strategy() is not None

    def test_custom_parameters(self):
        s = _make_strategy(risk_per_trade=0.02, atr_stop_factor=2.0, cooldown_bars_4h=3)
        assert s._risk == 0.02
        assert s._atr_stop_factor == 2.0
        assert s._cooldown_bars == 3

    def test_initial_state(self):
        s = _make_strategy()
        assert s._pending_entry is None
        assert s._cooldown == 0
        assert not s._active


class TestBarProcessing:
    def test_first_evaluate_returns_none(self):
        s = _make_strategy()
        rows = [_row("2024-01-01T00:00:00+00:00", 150.0, 150.1, 149.9, 150.0)]
        store = _make_store(rows)
        assert (
            s.evaluate(BarView(store, 0), HistoricalDataView(store, 0), _portfolio()) is None
        )

    def test_warmup_no_signals(self):
        s = _make_strategy()
        rows = []
        ts0 = pd.Timestamp("2024-01-01T00:00:00", tz="UTC")
        for i in range(240):
            p = 145.0 + i * 0.001
            rows.append(
                _row(
                    (ts0 + pd.Timedelta(minutes=i)).isoformat(),
                    p,
                    p + 0.05,
                    p - 0.05,
                    p,
                )
            )
        store = _make_store(rows)
        sigs = _feed_evaluate(s, store, 239)
        assert all(x is None for x in sigs)

    def test_4h_bars_accumulate(self):
        s = _make_strategy()
        rows = []
        ts0 = pd.Timestamp("2024-01-01T00:00:00", tz="UTC")
        for i in range(480):
            p = 145.0 + i * 0.001
            rows.append(
                _row(
                    (ts0 + pd.Timedelta(minutes=i)).isoformat(),
                    p,
                    p + 0.05,
                    p - 0.05,
                    p,
                )
            )
        store = _make_store(rows)
        _feed_evaluate(s, store, 479)
        assert len(s._completed_4h) >= 1


class TestSignalGeneration:
    def test_neutral_macro_no_signals(self):
        s = _make_strategy(bias=MacroBias.NEUTRAL)
        ts0 = pd.Timestamp("2024-01-01T00:00:00", tz="UTC")
        rows = []
        for i in range(14000):
            p = 145.0 + i * 0.001
            rows.append(
                _row(
                    (ts0 + pd.Timedelta(minutes=i)).isoformat(),
                    p,
                    p + 0.08,
                    p - 0.08,
                    p,
                )
            )
        store = _make_store(rows)
        sigs = [x for x in _feed_evaluate(s, store, 13999) if x is not None]
        assert len(sigs) == 0

    def test_short_macro_blocks_long_signals(self):
        s = _make_strategy(bias=MacroBias.SHORT)
        ts0 = pd.Timestamp("2024-01-01T00:00:00", tz="UTC")
        rows = []
        for i in range(14000):
            p = 145.0 + i * 0.001
            rows.append(
                _row(
                    (ts0 + pd.Timedelta(minutes=i)).isoformat(),
                    p,
                    p + 0.08,
                    p - 0.08,
                    p,
                )
            )
        store = _make_store(rows)
        longs = [
            x
            for x in _feed_evaluate(s, store, 13999)
            if x is not None and x.direction == "long"
        ]
        assert len(longs) == 0


class TestPositionSizing:
    def test_size_positive_and_cap(self):
        from core.regime_backtest_engine.swing_macro_trend import SwingMacroEntry, TrendState

        s = _make_strategy(account_balance=100_000, risk_per_trade=0.01, max_size=500_000)
        entry = SwingMacroEntry(
            timestamp=pd.Timestamp("2024-06-01", tz="UTC").to_pydatetime(),
            direction="long",
            entry_price=150.0,
            stop_loss=149.0,
            atr_value=0.7,
            trend_state=TrendState.UP,
            ema50_value=149.5,
            ema20_value=149.8,
        )
        assert s._compute_size(entry) > 0

    def test_wider_stop_smaller_size(self):
        from core.regime_backtest_engine.swing_macro_trend import SwingMacroEntry, TrendState

        s = _make_strategy(account_balance=100_000, risk_per_trade=0.01)
        tight = SwingMacroEntry(
            timestamp=pd.Timestamp("2024-06-01", tz="UTC").to_pydatetime(),
            direction="long",
            entry_price=150.0,
            stop_loss=149.5,
            atr_value=0.35,
            trend_state=TrendState.UP,
            ema50_value=149.0,
            ema20_value=149.5,
        )
        wide = SwingMacroEntry(
            timestamp=pd.Timestamp("2024-06-01", tz="UTC").to_pydatetime(),
            direction="long",
            entry_price=150.0,
            stop_loss=148.5,
            atr_value=1.0,
            trend_state=TrendState.UP,
            ema50_value=149.0,
            ema20_value=149.5,
        )
        assert s._compute_size(tight) > s._compute_size(wide)

    def test_zero_stop_distance(self):
        from core.regime_backtest_engine.swing_macro_trend import SwingMacroEntry, TrendState

        s = _make_strategy()
        entry = SwingMacroEntry(
            timestamp=pd.Timestamp("2024-06-01", tz="UTC").to_pydatetime(),
            direction="long",
            entry_price=150.0,
            stop_loss=150.0,
            atr_value=0.5,
            trend_state=TrendState.UP,
            ema50_value=149.0,
            ema20_value=149.5,
        )
        assert s._compute_size(entry) == 0

    def test_rounded_to_1000(self):
        from core.regime_backtest_engine.swing_macro_trend import SwingMacroEntry, TrendState

        s = _make_strategy()
        entry = SwingMacroEntry(
            timestamp=pd.Timestamp("2024-06-01", tz="UTC").to_pydatetime(),
            direction="long",
            entry_price=150.0,
            stop_loss=149.0,
            atr_value=0.7,
            trend_state=TrendState.UP,
            ema50_value=149.0,
            ema20_value=149.5,
        )
        assert s._compute_size(entry) % 1000 == 0


class TestMacroConfirmation:
    def test_long_long_bias(self):
        s = _make_strategy(bias=MacroBias.LONG)
        s._current_macro = s._macro.compute(pd.Timestamp("2024-01-01", tz="UTC"))
        assert s._macro_confirms("long") is True

    def test_long_lean_when_allowed(self):
        s = _make_strategy(bias=MacroBias.LEAN_LONG, allow_lean=True)
        s._current_macro = s._macro.compute(pd.Timestamp("2024-01-01", tz="UTC"))
        assert s._macro_confirms("long") is True

    def test_long_lean_blocked_when_disabled(self):
        s = _make_strategy(bias=MacroBias.LEAN_LONG, allow_lean=False)
        s._current_macro = s._macro.compute(pd.Timestamp("2024-01-01", tz="UTC"))
        assert s._macro_confirms("long") is False

    def test_long_blocked_short_bias(self):
        s = _make_strategy(bias=MacroBias.SHORT)
        s._current_macro = s._macro.compute(pd.Timestamp("2024-01-01", tz="UTC"))
        assert s._macro_confirms("long") is False

    def test_no_macro(self):
        s = _make_strategy()
        s._current_macro = None
        assert s._macro_confirms("long") is False
        assert s._macro_confirms("short") is False


class TestLifecycleCooldown:
    def test_stop_loss_sets_cooldown(self):
        s = _make_strategy(cooldown_bars_4h=3)
        tr = ClosedTrade(
            trade_id=1,
            family=s.family_name,
            direction="long",
            entry_time=None,
            exit_time=None,
            entry_bar=0,
            exit_bar=5,
            entry_price=150.0,
            exit_price=149.0,
            size=1000,
            margin_held=100.0,
            stop_loss=149.0,
            take_profit=None,
            spread_cost=0.0,
            slippage_cost=0.0,
            pnl_usd=-5.0,
            pnl_pips=-10.0,
            bars_held=5,
            exit_reason="stop_loss",
        )
        s.on_position_closed(tr)
        assert s._cooldown == 3

    def test_trend_reversal_no_cooldown(self):
        s = _make_strategy(cooldown_bars_4h=3)
        tr = ClosedTrade(
            trade_id=1,
            family=s.family_name,
            direction="long",
            entry_time=None,
            exit_time=None,
            entry_bar=0,
            exit_bar=5,
            entry_price=150.0,
            exit_price=151.0,
            size=1000,
            margin_held=100.0,
            stop_loss=149.0,
            take_profit=None,
            spread_cost=0.0,
            slippage_cost=0.0,
            pnl_usd=1.0,
            pnl_pips=10.0,
            bars_held=5,
            exit_reason="trend_reversal",
        )
        s.on_position_closed(tr)
        assert s._cooldown == 0


class TestSignalShape:
    def test_signal_dataclass(self):
        sig = Signal(
            family="swing_macro",
            direction="long",
            stop_loss=149.0,
            take_profit=None,
            size=100_000,
            metadata={"strategy": "swing_macro"},
        )
        assert sig.direction == "long"
        assert sig.take_profit is None
