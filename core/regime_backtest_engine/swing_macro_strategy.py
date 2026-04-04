"""
Swing-Macro strategy: 4H trend-pullback with weekly macro confirmation.

Implements the StrategyFamily protocol for the backtest engine.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, List, Optional

from .models import ClosedTrade, ExitAction, PortfolioSnapshot, PositionSnapshot, Signal
from .strategy import BarView, HistoricalDataView
from .swing_macro_bars import (
    Bar4H,
    IncrementalBar4HBuilder,
    find_recent_swing_high,
    find_recent_swing_low,
)
from .swing_macro_signals import (
    MacroBias,
    MacroReading,
    WeeklyMacroSignal,
    bias_supports_long,
    bias_supports_short,
)
from .swing_macro_trend import (
    DEFAULT_REVERSAL_WICK_RATIO,
    SwingMacroEntry,
    TrendDetector,
    TrendState,
)

# Initial stop distance uses TrendDetector.atr_stop_factor × ATR(14) from the pullback swing.
# Trailing stop buffer uses _trailing_buffer × ATR beyond the N-bar swing low/high (see _check_exits_4h).


def _to_utc_datetime(ts: Any) -> datetime:
    """Convert BarView timestamp to UTC datetime."""
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


class SwingMacroStrategy:
    """Swing-Macro strategy implementing StrategyFamily protocol.

    Stop parameters:
      * ``atr_stop_factor`` — passed to ``TrendDetector``; initial stop is
        swing ± (factor × ATR(14)) on entry (see ``swing_macro_trend``).
      * ``trail_buffer_atr`` — ATR multiple added beyond the N-bar swing
        when updating the trailing stop in ``_check_exits_4h``.
      * ``trailing_atr_buffer`` — optional legacy alias; if not ``None``,
        overrides ``trail_buffer_atr`` for backward compatibility.
      * ``require_reversal_bar`` — when True, pullback reclaim entries must
        pass the 4H bullish/bearish reversal bar rules in ``TrendDetector``.
      * ``reversal_wick_ratio`` — wick ≥ ratio × body for the reversal filter
        (default 1.5); only used when ``require_reversal_bar`` is True.
    """

    family_name: str = "swing_macro"

    def __init__(
        self,
        macro_signal: WeeklyMacroSignal | Any,
        account_balance: float = 100_000.0,
        risk_per_trade: float = 0.01,
        atr_stop_factor: float = 1.5,
        atr_proximity_factor: float = 0.3,
        trailing_swing_lookback: int = 5,
        trail_buffer_atr: float = 0.5,
        trailing_atr_buffer: float | None = None,
        cooldown_bars_4h: int = 2,
        allow_lean: bool = True,
        max_size: int = 500_000,
        require_reversal_bar: bool = False,
        reversal_wick_ratio: float = DEFAULT_REVERSAL_WICK_RATIO,
    ) -> None:
        self._allow_lean = allow_lean
        self._macro = macro_signal
        self._balance = account_balance
        self._risk = risk_per_trade
        self._atr_stop_factor = atr_stop_factor
        self._swing_lookback = trailing_swing_lookback
        self._trailing_buffer = (
            float(trailing_atr_buffer)
            if trailing_atr_buffer is not None
            else float(trail_buffer_atr)
        )
        self._cooldown_bars = cooldown_bars_4h
        self._max_size = max_size

        self._bar_builder = IncrementalBar4HBuilder()
        self._trend = TrendDetector(
            atr_proximity_factor=atr_proximity_factor,
            atr_stop_factor=atr_stop_factor,
            require_reversal_bar=require_reversal_bar,
            reversal_wick_ratio=reversal_wick_ratio,
        )

        self._completed_4h: List[Bar4H] = []
        self._max_history: int = 100

        self._last_processed_idx: int = -1
        self._current_macro: Optional[MacroReading] = None
        self._cooldown: int = 0

        self._pending_entry: Optional[SwingMacroEntry] = None
        self._pending_exit: Optional[ExitAction] = None
        self._pending_stop_update: Optional[float] = None

        self._active: bool = False
        self._active_direction: Optional[str] = None
        self._current_stop: Optional[float] = None
        self._last_entry_signal: Optional[SwingMacroEntry] = None

    # ──────────────────────────────────────────────
    #  Engine protocol methods
    # ──────────────────────────────────────────────

    def evaluate(
        self,
        current_bar: BarView,
        history: HistoricalDataView,
        portfolio: PortfolioSnapshot,
    ) -> Signal | None:
        _ = history
        self._process_bar(current_bar)

        has_position = any(
            p.family == self.family_name for p in portfolio.open_positions
        )
        if has_position:
            self._pending_entry = None
            return None

        if self._pending_entry is not None and self._cooldown <= 0:
            signal = self._create_signal(self._pending_entry)
            self._last_entry_signal = self._pending_entry
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
        self._active = True
        self._active_direction = position.direction
        self._current_stop = float(signal.stop_loss)

    def on_position_closed(self, trade: ClosedTrade) -> None:
        if trade.family != self.family_name:
            return
        remaining = int(getattr(trade, "remaining_units", 0) or 0)
        if remaining <= 0:
            self._active = False
            self._active_direction = None
            self._current_stop = None
            self._pending_exit = None
            self._pending_stop_update = None
            if trade.exit_reason == "stop_loss":
                self._cooldown = self._cooldown_bars

    # ──────────────────────────────────────────────
    #  Internal bar processing
    # ──────────────────────────────────────────────

    def _process_bar(self, current_bar: BarView) -> None:
        idx = int(current_bar.bar_index)
        if idx == self._last_processed_idx:
            return
        self._last_processed_idx = idx

        ts = _to_utc_datetime(current_bar.timestamp)
        o = float(current_bar.mid_open)
        h = float(current_bar.mid_high)
        l = float(current_bar.mid_low)
        c = float(current_bar.mid_close)

        completed = self._bar_builder.update(ts, o, h, l, c)

        if completed is not None:
            self._on_4h_complete(completed, ts)

    def _on_4h_complete(self, bar4h: Bar4H, current_ts: datetime) -> None:
        self._completed_4h.append(bar4h)
        if len(self._completed_4h) > self._max_history:
            self._completed_4h = self._completed_4h[-self._max_history :]

        self._current_macro = self._macro.compute(current_ts)

        if self._cooldown > 0:
            self._cooldown -= 1

        entry_signal = self._trend.update(bar4h)

        if self._active:
            self._check_exits_4h()

        if (
            entry_signal is not None
            and not self._active
            and self._cooldown <= 0
            and self._pending_entry is None
            and self._pending_exit is None
            and self._macro_confirms(entry_signal.direction)
        ):
            self._pending_entry = entry_signal

    def _check_exits_4h(self) -> None:
        trend = self._trend.trend
        if self._active_direction == "long" and trend != TrendState.UP:
            self._pending_exit = ExitAction(
                reason="trend_reversal",
                exit_type="full",
                close_fraction=1.0,
            )
            return

        if self._active_direction == "short" and trend != TrendState.DOWN:
            self._pending_exit = ExitAction(
                reason="trend_reversal",
                exit_type="full",
                close_fraction=1.0,
            )
            return

        if self._current_macro is not None:
            bias = self._current_macro.bias
            if self._active_direction == "long" and bias_supports_short(bias):
                self._pending_exit = ExitAction(
                    reason="macro_flip",
                    exit_type="full",
                    close_fraction=1.0,
                )
                return
            if self._active_direction == "short" and bias_supports_long(bias):
                self._pending_exit = ExitAction(
                    reason="macro_flip",
                    exit_type="full",
                    close_fraction=1.0,
                )
                return

        atr = self._trend.atr.value
        if atr is None or self._current_stop is None:
            return

        if self._active_direction == "long":
            swing_low = find_recent_swing_low(self._completed_4h, self._swing_lookback)
            if swing_low is not None:
                new_stop = swing_low - self._trailing_buffer * atr
                if new_stop > self._current_stop:
                    self._current_stop = new_stop
                    self._pending_stop_update = new_stop

        elif self._active_direction == "short":
            swing_high = find_recent_swing_high(self._completed_4h, self._swing_lookback)
            if swing_high is not None:
                new_stop = swing_high + self._trailing_buffer * atr
                if new_stop < self._current_stop:
                    self._current_stop = new_stop
                    self._pending_stop_update = new_stop

    def _macro_confirms(self, direction: str) -> bool:
        if self._current_macro is None:
            return False

        bias = self._current_macro.bias

        if direction == "long":
            if not bias_supports_long(bias):
                return False
            if not self._allow_lean and bias == MacroBias.LEAN_LONG:
                return False
            return True

        if direction == "short":
            if not bias_supports_short(bias):
                return False
            if not self._allow_lean and bias == MacroBias.LEAN_SHORT:
                return False
            return True

        return False

    def _create_signal(self, entry: SwingMacroEntry) -> Signal | None:
        size = self._compute_size(entry)
        if size <= 0:
            return None

        macro_bias = self._current_macro.bias.value if self._current_macro else "UNKNOWN"

        dir_lit = "long" if entry.direction == "long" else "short"
        return Signal(
            family=self.family_name,
            direction=dir_lit,
            stop_loss=entry.stop_loss,
            take_profit=None,
            size=int(size),
            metadata={
                "strategy": "swing_macro",
                "trend": entry.trend_state.value,
                "atr": round(entry.atr_value, 5),
                "ema50": round(entry.ema50_value, 3),
                "ema20": round(entry.ema20_value, 3),
                "macro_bias": macro_bias,
                "stop_distance_pips": round(abs(entry.entry_price - entry.stop_loss) * 100, 1),
            },
        )

    def _compute_size(self, signal: SwingMacroEntry) -> int:
        risk_dollars = self._balance * self._risk
        stop_distance = abs(signal.entry_price - signal.stop_loss)

        if stop_distance <= 0:
            return 0

        size = risk_dollars * signal.entry_price / stop_distance

        size = max(1000, round(size / 1000) * 1000)
        size = min(size, self._max_size)

        return int(size)
