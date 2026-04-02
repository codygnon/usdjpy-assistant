from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Protocol

import numpy as np

from .models import BarRecord, ClosedTrade, ExitAction, PortfolioSnapshot, PositionSnapshot, Signal


class MarketDataStore:
    def __init__(self, columns: dict[str, np.ndarray]) -> None:
        self._columns = columns
        self._length = len(next(iter(columns.values()))) if columns else 0

    def __len__(self) -> int:
        return self._length

    def value(self, column: str, index: int) -> Any:
        return self._columns[column][index]

    def bounded_column(self, column: str, max_index: int) -> np.ndarray:
        arr = self._columns[column][: max_index + 1]
        arr.setflags(write=False)
        return arr

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self._columns.keys())


@dataclass(frozen=True)
class BarView:
    _store: MarketDataStore
    bar_index: int

    def __getattr__(self, item: str) -> Any:
        if item.startswith("_"):
            raise AttributeError(item)
        return self._store.value(item, self.bar_index)

    def to_record(self) -> BarRecord:
        return BarRecord(
            bar_index=int(self.bar_index),
            timestamp=self.timestamp,
            bid_open=float(self.bid_open),
            bid_high=float(self.bid_high),
            bid_low=float(self.bid_low),
            bid_close=float(self.bid_close),
            ask_open=float(self.ask_open),
            ask_high=float(self.ask_high),
            ask_low=float(self.ask_low),
            ask_close=float(self.ask_close),
            mid_open=float(self.mid_open),
            mid_high=float(self.mid_high),
            mid_low=float(self.mid_low),
            mid_close=float(self.mid_close),
            spread_pips=float(self.spread_pips),
        )


class HistoricalDataView:
    def __init__(self, store: MarketDataStore, max_index: int) -> None:
        self._store = store
        self._max_index = max_index

    def __len__(self) -> int:
        return self._max_index + 1

    @property
    def max_index(self) -> int:
        return self._max_index

    def __getitem__(self, item: int | slice) -> BarView | list[BarView]:
        if isinstance(item, slice):
            start, stop, step = item.indices(self._max_index + 1)
            return [BarView(self._store, idx) for idx in range(start, stop, step)]
        if item < 0:
            item = self._max_index + 1 + item
        if item < 0 or item > self._max_index:
            raise IndexError("historical view does not expose future bars")
        return BarView(self._store, item)

    def iter_bars(self) -> Iterator[BarView]:
        for idx in range(self._max_index + 1):
            yield BarView(self._store, idx)

    def column(self, name: str) -> np.ndarray:
        return self._store.bounded_column(name, self._max_index)

    def tail(self, count: int) -> list[BarView]:
        if count <= 0:
            raise ValueError("count must be positive")
        start = max(0, self._max_index + 1 - count)
        return [BarView(self._store, idx) for idx in range(start, self._max_index + 1)]

    def window(self, start: int, end: int) -> list[BarView]:
        if start < 0 or end < start:
            raise ValueError("invalid history window")
        if end > self._max_index:
            raise IndexError("historical view does not expose future bars")
        return [BarView(self._store, idx) for idx in range(start, end + 1)]

    def rolling_mean(self, name: str, window: int, *, min_periods: int | None = None) -> float | None:
        values = self._rolling_values(name, window, min_periods=min_periods)
        if values is None:
            return None
        return float(np.mean(values))

    def rolling_min(self, name: str, window: int, *, min_periods: int | None = None) -> float | None:
        values = self._rolling_values(name, window, min_periods=min_periods)
        if values is None:
            return None
        return float(np.min(values))

    def rolling_max(self, name: str, window: int, *, min_periods: int | None = None) -> float | None:
        values = self._rolling_values(name, window, min_periods=min_periods)
        if values is None:
            return None
        return float(np.max(values))

    def rolling_std(self, name: str, window: int, *, min_periods: int | None = None, ddof: int = 0) -> float | None:
        values = self._rolling_values(name, window, min_periods=min_periods)
        if values is None:
            return None
        return float(np.std(values, ddof=ddof))

    def _rolling_values(self, name: str, window: int, *, min_periods: int | None = None) -> np.ndarray | None:
        if window <= 0:
            raise ValueError("window must be positive")
        required = window if min_periods is None else min_periods
        if required <= 0:
            raise ValueError("min_periods must be positive when provided")
        values = np.asarray(self.column(name), dtype=float)
        if len(values) < required:
            return None
        return values[-window:]


class StrategyFamily(Protocol):
    family_name: str

    def evaluate(
        self,
        current_bar: BarView,
        history: HistoricalDataView,
        portfolio: PortfolioSnapshot,
    ) -> Signal | None:
        ...

    def get_exit_conditions(
        self,
        position: PositionSnapshot,
        current_bar: BarView,
        history: HistoricalDataView,
    ) -> ExitAction | None:
        ...


class TrainableStrategyFamily(StrategyFamily, Protocol):
    def fit(self, history: HistoricalDataView) -> None:
        ...


class LifecycleStrategyFamily(StrategyFamily, Protocol):
    def on_position_opened(self, position: PositionSnapshot, signal: Signal, current_bar: BarView) -> None:
        ...

    def on_position_closed(self, trade: ClosedTrade) -> None:
        ...


class StrategyValidationError(ValueError):
    pass


@dataclass(frozen=True)
class StrategyAdapter:
    family_name: str
    strategy: StrategyFamily

    def evaluate(
        self,
        current_bar: BarView,
        history: HistoricalDataView,
        portfolio: PortfolioSnapshot,
    ) -> Signal | None:
        signal = self.strategy.evaluate(current_bar, history, portfolio)
        if signal is None:
            return None
        self._validate_signal(signal)
        return signal

    def get_exit_conditions(
        self,
        position: PositionSnapshot,
        current_bar: BarView,
        history: HistoricalDataView,
    ) -> ExitAction | None:
        action = self.strategy.get_exit_conditions(position, current_bar, history)
        if action is None:
            return None
        stop_loss = action.new_stop_loss if action.new_stop_loss is not None else action.stop_loss
        take_profit = action.new_take_profit if action.new_take_profit is not None else action.take_profit
        if stop_loss is not None and not np.isfinite(float(stop_loss)):
            raise StrategyValidationError(f"{self.family_name}: exit stop_loss must be finite")
        if take_profit is not None and not np.isfinite(float(take_profit)):
            raise StrategyValidationError(f"{self.family_name}: exit take_profit must be finite")
        if action.exit_type not in {"none", "full", "partial"}:
            raise StrategyValidationError(f"{self.family_name}: invalid exit_type {action.exit_type!r}")
        if action.exit_type == "partial":
            if not (0.0 < float(action.close_fraction) < 1.0):
                raise StrategyValidationError(f"{self.family_name}: partial close_fraction must be between 0 and 1")
        elif action.exit_type == "full" and float(action.close_fraction) != 1.0:
            raise StrategyValidationError(f"{self.family_name}: full exit close_fraction must be 1.0")
        if action.price is not None and not np.isfinite(float(action.price)):
            raise StrategyValidationError(f"{self.family_name}: exit price must be finite")
        return action

    def _validate_signal(self, signal: Signal) -> None:
        if signal.family != self.family_name:
            raise StrategyValidationError(
                f"{self.family_name}: strategy returned signal for unexpected family {signal.family!r}"
            )
        if signal.direction not in {"long", "short"}:
            raise StrategyValidationError(f"{self.family_name}: invalid signal direction {signal.direction!r}")
        if int(signal.size) <= 0:
            raise StrategyValidationError(f"{self.family_name}: signal size must be positive")
        if not np.isfinite(float(signal.stop_loss)):
            raise StrategyValidationError(f"{self.family_name}: stop_loss must be finite")
        if signal.take_profit is not None and not np.isfinite(float(signal.take_profit)):
            raise StrategyValidationError(f"{self.family_name}: take_profit must be finite")

    def on_position_opened(self, position: PositionSnapshot, signal: Signal, current_bar: BarView) -> None:
        hook = getattr(self.strategy, "on_position_opened", None)
        if callable(hook):
            hook(position, signal, current_bar)

    def on_position_closed(self, trade: ClosedTrade) -> None:
        hook = getattr(self.strategy, "on_position_closed", None)
        if callable(hook):
            hook(trade)
