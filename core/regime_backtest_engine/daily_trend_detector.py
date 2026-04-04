"""
Daily trend state and 4H pullback entry logic for the daily_trend strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

from .daily_trend_bars import DailyBar
from .swing_macro_bars import Bar4H, IncrementalATR, IncrementalEMA


def _daily_as_bar4h(d: DailyBar) -> Bar4H:
    """IncrementalATR expects Bar4H; daily OHLC is sufficient."""
    return Bar4H(
        timestamp=d.timestamp,
        open=d.open,
        high=d.high,
        low=d.low,
        close=d.close,
        bar_count=1,
    )


class DailyTrend(Enum):
    NONE = auto()
    UP = auto()
    DOWN = auto()


class PullbackState(Enum):
    NONE = auto()
    ACTIVE = auto()
    EXPIRED = auto()


@dataclass(frozen=True)
class DailyTrendEntry:
    direction: str
    entry_price: float
    stop_loss: float
    daily_atr: float
    ema20: float
    ema50: float
    metadata: dict[str, Any] = field(default_factory=dict)


class DailyTrendDetector:
    """Daily EMA50/20/ATR trend; 4H pullback entries; daily ATR for stops/trailing."""

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
    ) -> None:
        self._ema_fast = IncrementalEMA(daily_ema_fast)
        self._ema_slow = IncrementalEMA(daily_ema_slow)
        self._atr = IncrementalATR(daily_atr_period)

        self._trend = DailyTrend.NONE
        self._pullback = PullbackState.NONE

        self._proximity_atr = proximity_atr
        self._stop_atr_factor = stop_atr_factor
        self._trail_buffer_atr = trail_buffer_atr
        self._pullback_lookback = pullback_lookback
        self._trail_lookback = trail_lookback
        self._warmup_period = daily_ema_slow

        self._recent_4h: list[Bar4H] = []
        self._recent_daily: list[DailyBar] = []
        self._buf_cap = max(pullback_lookback, trail_lookback) + 5

        self._daily_bars_seen: int = 0

    @property
    def is_warmed_up(self) -> bool:
        return self._daily_bars_seen >= self._warmup_period

    @property
    def trend(self) -> DailyTrend:
        return self._trend

    @property
    def ema20(self) -> Optional[float]:
        return self._ema_fast.value

    @property
    def ema50(self) -> Optional[float]:
        return self._ema_slow.value

    @property
    def daily_atr(self) -> Optional[float]:
        return self._atr.value

    def update_daily(self, bar: DailyBar) -> None:
        prev_trend = self._trend

        self._ema_fast.update(bar.close)
        self._ema_slow.update(bar.close)
        self._atr.update(_daily_as_bar4h(bar))

        slow = self._ema_slow.value
        prev_slow = self._ema_slow.prev_value

        if not self.is_warmed_up or slow is None or prev_slow is None:
            new_trend = DailyTrend.NONE
        elif bar.close > slow and slow > prev_slow:
            new_trend = DailyTrend.UP
        elif bar.close < slow and slow < prev_slow:
            new_trend = DailyTrend.DOWN
        else:
            new_trend = DailyTrend.NONE

        if new_trend != prev_trend:
            self._pullback = PullbackState.NONE

        self._trend = new_trend

        self._recent_daily.append(bar)
        if len(self._recent_daily) > self._buf_cap:
            self._recent_daily = self._recent_daily[-self._buf_cap :]

        self._daily_bars_seen += 1

    def update_4h(self, bar: Bar4H) -> Optional[DailyTrendEntry]:
        self._recent_4h.append(bar)
        if len(self._recent_4h) > self._buf_cap:
            self._recent_4h = self._recent_4h[-self._buf_cap :]

        if not self.is_warmed_up:
            return None
        if self._trend == DailyTrend.NONE:
            self._pullback = PullbackState.NONE
            return None

        ema20 = self.ema20
        atr = self.daily_atr
        if ema20 is None or atr is None or atr <= 0:
            return None

        proximity = self._proximity_atr * atr

        if self._trend == DailyTrend.UP:
            return self._check_long_pullback(bar, ema20, atr, proximity)
        if self._trend == DailyTrend.DOWN:
            return self._check_short_pullback(bar, ema20, atr, proximity)
        return None

    def _check_long_pullback(
        self, bar: Bar4H, ema20: float, atr: float, proximity: float
    ) -> Optional[DailyTrendEntry]:
        if self._pullback in (PullbackState.NONE, PullbackState.EXPIRED):
            if bar.low <= ema20 + proximity:
                self._pullback = PullbackState.ACTIVE

        if self._pullback == PullbackState.ACTIVE:
            if bar.close < ema20 - (2.0 * atr):
                self._pullback = PullbackState.EXPIRED
                return None

            if bar.close > ema20:
                swing_low = self._get_4h_swing_low()
                if swing_low is None:
                    return None
                stop = swing_low - (self._stop_atr_factor * atr)
                ema50_v = self.ema50
                if ema50_v is None:
                    return None
                self._pullback = PullbackState.NONE
                return DailyTrendEntry(
                    direction="long",
                    entry_price=bar.close,
                    stop_loss=stop,
                    daily_atr=atr,
                    ema20=ema20,
                    ema50=ema50_v,
                    metadata={
                        "trend": "UP",
                        "swing_low": swing_low,
                        "bar_timestamp": str(bar.timestamp),
                    },
                )

        return None

    def _check_short_pullback(
        self, bar: Bar4H, ema20: float, atr: float, proximity: float
    ) -> Optional[DailyTrendEntry]:
        if self._pullback in (PullbackState.NONE, PullbackState.EXPIRED):
            if bar.high >= ema20 - proximity:
                self._pullback = PullbackState.ACTIVE

        if self._pullback == PullbackState.ACTIVE:
            if bar.close > ema20 + (2.0 * atr):
                self._pullback = PullbackState.EXPIRED
                return None

            if bar.close < ema20:
                swing_high = self._get_4h_swing_high()
                if swing_high is None:
                    return None
                stop = swing_high + (self._stop_atr_factor * atr)
                ema50_v = self.ema50
                if ema50_v is None:
                    return None
                self._pullback = PullbackState.NONE
                return DailyTrendEntry(
                    direction="short",
                    entry_price=bar.close,
                    stop_loss=stop,
                    daily_atr=atr,
                    ema20=ema20,
                    ema50=ema50_v,
                    metadata={
                        "trend": "DOWN",
                        "swing_high": swing_high,
                        "bar_timestamp": str(bar.timestamp),
                    },
                )

        return None

    def _get_4h_swing_low(self) -> Optional[float]:
        bars = self._recent_4h[-self._pullback_lookback :]
        if not bars:
            return None
        return min(b.low for b in bars)

    def _get_4h_swing_high(self) -> Optional[float]:
        bars = self._recent_4h[-self._pullback_lookback :]
        if not bars:
            return None
        return max(b.high for b in bars)

    def compute_trailing_stop_long(self) -> Optional[float]:
        if len(self._recent_daily) < self._trail_lookback:
            return None
        atr = self.daily_atr
        if atr is None or atr <= 0:
            return None
        recent = self._recent_daily[-self._trail_lookback :]
        swing_low = min(b.low for b in recent)
        return swing_low - (self._trail_buffer_atr * atr)

    def compute_trailing_stop_short(self) -> Optional[float]:
        if len(self._recent_daily) < self._trail_lookback:
            return None
        atr = self.daily_atr
        if atr is None or atr <= 0:
            return None
        recent = self._recent_daily[-self._trail_lookback :]
        swing_high = max(b.high for b in recent)
        return swing_high + (self._trail_buffer_atr * atr)
