"""
Trend detection and pullback entry logic for the Swing-Macro strategy.

Components:
  - TrendState: UP / DOWN / NONE enum
  - PullbackPhase: Tracks whether price has pulled back to EMA20
  - TrendDetector: Combines EMA50 trend + EMA20 pullback into entry signals
  - SwingMacroEntry: Complete entry signal dataclass
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional

from .swing_macro_bars import Bar4H, IncrementalATR, IncrementalEMA

# Default multipliers (mirrored in SwingMacroStrategy for initial / trailing stops).
# Initial stop: swing_extreme ± (DEFAULT_ATR_STOP_FACTOR × ATR(14)) via TrendDetector.
DEFAULT_ATR_PROXIMITY_FACTOR = 0.3
DEFAULT_ATR_STOP_FACTOR = 1.5
DEFAULT_REVERSAL_WICK_RATIO = 1.5


def is_bullish_reversal_bar(
    current_bar: Bar4H,
    previous_bar: Bar4H,
    ema20_value: float,
    *,
    wick_ratio: float = DEFAULT_REVERSAL_WICK_RATIO,
) -> bool:
    """Bullish reversal bar: probed below prior low, closed above midrange,

    closed above EMA(20), lower wick >= wick_ratio × body.
    Doji (body=0) automatically passes the wick test.
    """
    o = current_bar.open
    h = current_bar.high
    l = current_bar.low
    c = current_bar.close

    if l >= previous_bar.low:
        return False
    mid = (h + l) / 2.0
    if c <= mid:
        return False
    if c <= ema20_value:
        return False

    body_size = abs(c - o)
    lower_wick = min(o, c) - l
    if body_size == 0:
        return True
    return lower_wick >= wick_ratio * body_size


def is_bearish_reversal_bar(
    current_bar: Bar4H,
    previous_bar: Bar4H,
    ema20_value: float,
    *,
    wick_ratio: float = DEFAULT_REVERSAL_WICK_RATIO,
) -> bool:
    """Bearish reversal bar: probed above prior high, closed below midrange,

    closed below EMA(20), upper wick >= wick_ratio × body.
    Doji (body=0) automatically passes the wick test.
    """
    o = current_bar.open
    h = current_bar.high
    l = current_bar.low
    c = current_bar.close

    if h <= previous_bar.high:
        return False
    mid = (h + l) / 2.0
    if c >= mid:
        return False
    if c >= ema20_value:
        return False

    body_size = abs(c - o)
    upper_wick = h - max(o, c)
    if body_size == 0:
        return True
    return upper_wick >= wick_ratio * body_size


class TrendState(Enum):
    UP = "UP"
    DOWN = "DOWN"
    NONE = "NONE"


class PullbackPhase(Enum):
    """Tracks the pullback lifecycle."""

    WAITING = "WAITING"  # In trend, waiting for pullback to start
    PULLING_BACK = "PULLING_BACK"  # Price is near/below EMA20 (for longs)
    TRIGGERED = "TRIGGERED"  # Price reclaimed EMA20 — valid entry


@dataclass(frozen=True)
class SwingMacroEntry:
    """A complete entry signal with all context needed for trade management."""

    timestamp: datetime  # Bar that triggered the signal
    direction: str  # "long" or "short"
    entry_price: float  # Close of the trigger bar (fill at next bar open)
    stop_loss: float  # 1.5 × ATR below swing low (long) or above swing high (short)
    atr_value: float  # Current ATR for reference
    trend_state: TrendState
    ema50_value: float
    ema20_value: float


class TrendDetector:
    """Detects trend direction and pullback entry opportunities on 4H bars.

    Trend rules:
      UP:   close > EMA(50) AND EMA(50) slope > 0
      DOWN: close < EMA(50) AND EMA(50) slope < 0
      NONE: everything else

    Pullback rules (for longs, inverse for shorts):
      1. WAITING: trend is UP, price is above EMA(20)
      2. PULLING_BACK: price touches or drops within atr_proximity of EMA(20)
      3. TRIGGERED: price closes back above EMA(20)

    After a TRIGGERED signal fires, reset to WAITING to prevent
    repeated signals on the same pullback.
    """

    def __init__(
        self,
        atr_proximity_factor: float = DEFAULT_ATR_PROXIMITY_FACTOR,
        atr_stop_factor: float = DEFAULT_ATR_STOP_FACTOR,
        require_reversal_bar: bool = False,
        reversal_wick_ratio: float = DEFAULT_REVERSAL_WICK_RATIO,
    ):
        """
        Args:
            atr_proximity_factor: How close to EMA20 counts as "touching it".
                                  0.3 means within 0.3 × ATR.
            atr_stop_factor: Initial stop distance as multiple of ATR(14) beyond pullback swing.
            require_reversal_bar: If True, reclaim entries must pass bullish/bearish reversal rules.
            reversal_wick_ratio: Lower/upper wick must be >= this × body (ignored when body is 0).
        """
        self.atr_proximity = atr_proximity_factor
        self.atr_stop_factor = atr_stop_factor
        self._require_reversal_bar = require_reversal_bar
        self._reversal_wick_ratio = reversal_wick_ratio

        self.ema50 = IncrementalEMA(period=50)
        self.ema20 = IncrementalEMA(period=20)
        self.atr = IncrementalATR(period=14)

        self._trend: TrendState = TrendState.NONE
        self._pullback_phase: PullbackPhase = PullbackPhase.WAITING
        self._pullback_swing_low: Optional[float] = None  # For long stop placement
        self._pullback_swing_high: Optional[float] = None  # For short stop placement
        self._last_signal_bar: Optional[datetime] = None
        self._previous_bar: Optional[Bar4H] = None

        # Track recent lows/highs for swing-based stops
        self._recent_bars: List[Bar4H] = []
        self._max_recent_bars: int = 10  # Keep last 10 for swing detection

    @property
    def trend(self) -> TrendState:
        return self._trend

    @property
    def pullback_phase(self) -> PullbackPhase:
        return self._pullback_phase

    def update(self, bar: Bar4H) -> Optional[SwingMacroEntry]:
        """Feed one completed 4H bar. Returns an entry signal if triggered.

        Call this each time a new 4H bar completes.
        Returns None most of the time — only returns a signal when
        all conditions align.
        """
        try:
            return self._update_inner(bar)
        finally:
            # Prior completed bar for next call's reversal probe (vs current bar).
            self._previous_bar = bar

    def _update_inner(self, bar: Bar4H) -> Optional[SwingMacroEntry]:
        # Update indicators
        ema50_val = self.ema50.update(bar.close)
        ema20_val = self.ema20.update(bar.close)
        atr_val = self.atr.update(bar)

        # Track recent bars for swing stops
        self._recent_bars.append(bar)
        if len(self._recent_bars) > self._max_recent_bars:
            self._recent_bars = self._recent_bars[-self._max_recent_bars :]

        # Can't do anything without all indicators ready
        if ema50_val is None or ema20_val is None or atr_val is None:
            return None

        if self.ema50.slope is None:
            return None

        # Determine trend
        prev_trend = self._trend
        self._trend = self._compute_trend(bar.close, ema50_val, self.ema50.slope)

        # If trend changed, reset pullback tracking
        if self._trend != prev_trend:
            self._pullback_phase = PullbackPhase.WAITING
            self._pullback_swing_low = None
            self._pullback_swing_high = None

        # No signals in NONE trend
        if self._trend == TrendState.NONE:
            self._pullback_phase = PullbackPhase.WAITING
            return None

        # Track pullback lifecycle
        signal = self._update_pullback(bar, ema20_val, atr_val)
        return signal

    def _compute_trend(self, close: float, ema50: float, slope: float) -> TrendState:
        """Determine trend from price position relative to EMA50 and its slope."""
        if close > ema50 and slope > 0:
            return TrendState.UP
        if close < ema50 and slope < 0:
            return TrendState.DOWN
        return TrendState.NONE

    def _update_pullback(
        self, bar: Bar4H, ema20: float, atr: float
    ) -> Optional[SwingMacroEntry]:
        """Track pullback phases and generate entry signal when triggered."""
        proximity = self.atr_proximity * atr

        if self._trend == TrendState.UP:
            return self._update_pullback_long(bar, ema20, atr, proximity)
        if self._trend == TrendState.DOWN:
            return self._update_pullback_short(bar, ema20, atr, proximity)
        return None

    def _update_pullback_long(
        self, bar: Bar4H, ema20: float, atr: float, proximity: float
    ) -> Optional[SwingMacroEntry]:
        """Pullback tracking for long setups."""

        if self._pullback_phase == PullbackPhase.WAITING:
            # Check if price is pulling back toward EMA20
            if bar.low <= ema20 + proximity:
                self._pullback_phase = PullbackPhase.PULLING_BACK
                self._pullback_swing_low = bar.low
            return None

        if self._pullback_phase == PullbackPhase.PULLING_BACK:
            # Track the lowest point of the pullback
            if self._pullback_swing_low is not None and bar.low < self._pullback_swing_low:
                self._pullback_swing_low = bar.low
            elif self._pullback_swing_low is None:
                self._pullback_swing_low = bar.low

            # Check for reclaim: bar closes above EMA20
            if bar.close > ema20:
                if self._require_reversal_bar:
                    # No prior bar yet, or failed reversal pattern: skip entry (conservative).
                    if self._previous_bar is None or not is_bullish_reversal_bar(
                        bar,
                        self._previous_bar,
                        ema20,
                        wick_ratio=self._reversal_wick_ratio,
                    ):
                        return None

                stop = self._pullback_swing_low - (self.atr_stop_factor * atr)
                ema50_now = self.ema50.value
                assert ema50_now is not None

                signal = SwingMacroEntry(
                    timestamp=bar.timestamp,
                    direction="long",
                    entry_price=bar.close,
                    stop_loss=stop,
                    atr_value=atr,
                    trend_state=self._trend,
                    ema50_value=ema50_now,
                    ema20_value=ema20,
                )

                # Reset for next pullback
                self._pullback_phase = PullbackPhase.WAITING
                self._pullback_swing_low = None
                self._last_signal_bar = bar.timestamp

                return signal

            return None

        if self._pullback_phase == PullbackPhase.TRIGGERED:
            # Shouldn't reach here — we reset to WAITING after trigger
            self._pullback_phase = PullbackPhase.WAITING
            return None

        return None

    def _update_pullback_short(
        self, bar: Bar4H, ema20: float, atr: float, proximity: float
    ) -> Optional[SwingMacroEntry]:
        """Pullback tracking for short setups."""

        if self._pullback_phase == PullbackPhase.WAITING:
            # Check if price is pulling back up toward EMA20
            if bar.high >= ema20 - proximity:
                self._pullback_phase = PullbackPhase.PULLING_BACK
                self._pullback_swing_high = bar.high
            return None

        if self._pullback_phase == PullbackPhase.PULLING_BACK:
            # Track the highest point of the pullback
            if self._pullback_swing_high is not None and bar.high > self._pullback_swing_high:
                self._pullback_swing_high = bar.high
            elif self._pullback_swing_high is None:
                self._pullback_swing_high = bar.high

            # Check for reclaim: bar closes below EMA20
            if bar.close < ema20:
                if self._require_reversal_bar:
                    if self._previous_bar is None or not is_bearish_reversal_bar(
                        bar,
                        self._previous_bar,
                        ema20,
                        wick_ratio=self._reversal_wick_ratio,
                    ):
                        return None

                stop = self._pullback_swing_high + (self.atr_stop_factor * atr)
                ema50_now = self.ema50.value
                assert ema50_now is not None

                signal = SwingMacroEntry(
                    timestamp=bar.timestamp,
                    direction="short",
                    entry_price=bar.close,
                    stop_loss=stop,
                    atr_value=atr,
                    trend_state=self._trend,
                    ema50_value=ema50_now,
                    ema20_value=ema20,
                )

                # Reset for next pullback
                self._pullback_phase = PullbackPhase.WAITING
                self._pullback_swing_high = None
                self._last_signal_bar = bar.timestamp

                return signal

            return None

        if self._pullback_phase == PullbackPhase.TRIGGERED:
            self._pullback_phase = PullbackPhase.WAITING
            return None

        return None
