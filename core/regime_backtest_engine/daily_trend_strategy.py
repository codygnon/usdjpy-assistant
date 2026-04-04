"""
Daily trend (EMA50 direction on daily bars) + 4H pullback entries.

Implements the StrategyFamily protocol for the backtest engine.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from .daily_trend_bars import DailyBar, IncrementalDailyBarBuilder
from .daily_trend_detector import DailyTrend, DailyTrendDetector, DailyTrendEntry
from .models import ClosedTrade, ExitAction, PortfolioSnapshot, PositionSnapshot, Signal
from .strategy import BarView, HistoricalDataView
from .swing_macro_bars import Bar4H, IncrementalBar4HBuilder


def _to_utc_datetime(ts: Any) -> datetime:
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)
    s = str(ts)
    s = s.replace("Z", "+00:00")
    if "+" not in s and "Z" not in s:
        s += "+00:00"
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        import pandas as pd

        return pd.Timestamp(ts).to_pydatetime().replace(tzinfo=timezone.utc)


class DailyTrendStrategy:
    """Daily EMA50 trend; 4H pullback to daily EMA20; trailing on daily completion."""

    family_name: str = "daily_trend"

    def __init__(
        self,
        daily_ema_fast: int = 20,
        daily_ema_slow: int = 50,
        daily_atr_period: int = 14,
        proximity_atr: float = 0.3,
        stop_atr_factor: float = 1.5,
        trail_buffer_atr: float = 0.5,
        pullback_lookback: int = 5,
        trail_lookback: int = 5,
        risk_fraction: float = 0.01,
        account_balance: float = 100_000.0,
        max_size: int = 500_000,
    ) -> None:
        self._balance = account_balance
        self._risk = risk_fraction
        self._max_size = max_size

        self._daily_builder = IncrementalDailyBarBuilder()
        self._bar4h_builder = IncrementalBar4HBuilder()
        self._detector = DailyTrendDetector(
            daily_ema_fast=daily_ema_fast,
            daily_ema_slow=daily_ema_slow,
            daily_atr_period=daily_atr_period,
            proximity_atr=proximity_atr,
            stop_atr_factor=stop_atr_factor,
            trail_buffer_atr=trail_buffer_atr,
            pullback_lookback=pullback_lookback,
            trail_lookback=trail_lookback,
        )

        self._last_processed_idx: int = -1
        self._pending_entry: Optional[DailyTrendEntry] = None
        self._daily_completed_this_bar: bool = False

        self._has_position: bool = False
        self._position_direction: Optional[str] = None
        self._current_stop: Optional[float] = None
        self._last_trend_at_entry: Optional[DailyTrend] = None

        self._pending_exit: Optional[ExitAction] = None
        self._pending_stop_update: Optional[float] = None

    def evaluate(
        self,
        current_bar: BarView,
        history: HistoricalDataView,
        portfolio: PortfolioSnapshot,
    ) -> Signal | None:
        _ = history
        self._process_bar(current_bar)

        has_position = any(p.family == self.family_name for p in portfolio.open_positions)
        if has_position:
            self._pending_entry = None
            return None

        if self._pending_entry is not None:
            signal = self._create_signal(self._pending_entry)
            self._pending_entry = None
            if signal is not None:
                return signal
        return None

    def get_exit_conditions(
        self,
        position: PositionSnapshot,
        current_bar: BarView,
        history: HistoricalDataView,
    ) -> ExitAction | None:
        _ = history
        if position.family != self.family_name:
            return None

        self._process_bar(current_bar)

        if self._pending_exit is not None:
            action = self._pending_exit
            self._pending_exit = None
            self._pending_stop_update = None
            return action

        if self._pending_stop_update is not None:
            new_sl = self._pending_stop_update
            self._pending_stop_update = None
            return ExitAction(
                reason="trailing_update",
                exit_type="none",
                new_stop_loss=new_sl,
            )

        return None

    def on_position_opened(
        self, position: PositionSnapshot, signal: Signal, current_bar: BarView
    ) -> None:
        _ = current_bar
        self._has_position = True
        self._position_direction = position.direction
        self._current_stop = float(signal.stop_loss)
        self._last_trend_at_entry = self._detector.trend

    def on_position_closed(self, trade: ClosedTrade) -> None:
        if trade.family != self.family_name:
            return
        remaining = int(getattr(trade, "remaining_units", 0) or 0)
        if remaining <= 0:
            self._has_position = False
            self._position_direction = None
            self._current_stop = None
            self._last_trend_at_entry = None
            self._pending_exit = None
            self._pending_stop_update = None

    def _process_bar(self, current_bar: BarView) -> None:
        idx = int(current_bar.bar_index)
        if idx == self._last_processed_idx:
            return
        self._last_processed_idx = idx
        self._daily_completed_this_bar = False

        ts = _to_utc_datetime(current_bar.timestamp)
        o = float(current_bar.mid_open)
        h = float(current_bar.mid_high)
        l = float(current_bar.mid_low)
        c = float(current_bar.mid_close)

        daily_done = self._daily_builder.update(current_bar)
        if daily_done is not None:
            self._detector.update_daily(daily_done)
            self._daily_completed_this_bar = True

        completed_4h = self._bar4h_builder.update(ts, o, h, l, c)
        if completed_4h is not None:
            entry = self._detector.update_4h(completed_4h)
            if entry is not None and not self._has_position and self._pending_exit is None:
                self._pending_entry = entry

        if self._has_position:
            self._check_trend_reversal_exit()
            if self._daily_completed_this_bar:
                self._maybe_trailing_stop_update()

    def _check_trend_reversal_exit(self) -> None:
        if self._pending_exit is not None:
            return
        cur = self._detector.trend
        if self._position_direction == "long" and self._last_trend_at_entry == DailyTrend.UP:
            if cur != DailyTrend.UP:
                self._pending_exit = ExitAction(
                    reason="trend_reversal",
                    exit_type="full",
                    close_fraction=1.0,
                )
        elif self._position_direction == "short" and self._last_trend_at_entry == DailyTrend.DOWN:
            if cur != DailyTrend.DOWN:
                self._pending_exit = ExitAction(
                    reason="trend_reversal",
                    exit_type="full",
                    close_fraction=1.0,
                )

    def _maybe_trailing_stop_update(self) -> None:
        if self._pending_exit is not None or self._current_stop is None:
            return
        new_stop: Optional[float] = None
        if self._position_direction == "long":
            new_stop = self._detector.compute_trailing_stop_long()
        elif self._position_direction == "short":
            new_stop = self._detector.compute_trailing_stop_short()
        if new_stop is None:
            return
        if self._position_direction == "long" and new_stop > self._current_stop:
            self._current_stop = new_stop
            self._pending_stop_update = new_stop
        elif self._position_direction == "short" and new_stop < self._current_stop:
            self._current_stop = new_stop
            self._pending_stop_update = new_stop

    def _create_signal(self, entry: DailyTrendEntry) -> Signal | None:
        size = self._compute_size(entry)
        if size <= 0:
            return None
        stop_pips = abs(entry.entry_price - entry.stop_loss) * 100
        return Signal(
            family=self.family_name,
            direction=entry.direction,  # type: ignore[arg-type]
            stop_loss=entry.stop_loss,
            take_profit=None,
            size=int(size),
            metadata={
                "strategy": "daily_trend",
                **entry.metadata,
                "daily_atr": round(entry.daily_atr, 5),
                "ema20": round(entry.ema20, 3),
                "ema50": round(entry.ema50, 3),
                "stop_distance_pips": round(stop_pips, 1),
            },
        )

    def _compute_size(self, entry: DailyTrendEntry) -> int:
        risk_dollars = self._balance * self._risk
        stop_distance = abs(entry.entry_price - entry.stop_loss)
        if stop_distance <= 0:
            return 0
        size = risk_dollars * entry.entry_price / stop_distance
        size = max(1000, round(size / 1000) * 1000)
        size = min(size, self._max_size)
        return int(size)
