from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .daily_levels import (
    compute_pdh_pdl,
    compute_pmh_pml,
    compute_pwh_pwl,
    compute_round_levels,
    find_nearest_level,
    is_level_cluster,
)
from .models import ClosedTrade, ExitAction, PortfolioSnapshot, PositionSnapshot, Signal
from .strategy import BarView, HistoricalDataView, TrainableStrategyFamily
from .synthetic_bars import BarDaily, aggregate_to_daily


Bias = Literal["long", "short", "neutral"]


class MacroReversionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    family_name: str = "macro_reversion"
    pip_size: float = 0.01
    fixed_units: int = 200_000
    lookback_days: int = 252
    short_bias_threshold: float = 0.80
    long_bias_threshold: float = 0.20
    short_entry_threshold: float = 0.75
    long_entry_threshold: float = 0.25
    level_touch_distance_pips: float = 10.0
    cluster_radius_pips: float = 20.0
    cooldown_bars: int = 50
    max_hold_bars: int = 480
    stop_loss_pips: float = 30.0
    tp1_pips: float = 15.0
    tp2_pips: float = 30.0
    london_start_hour_utc: int = 7
    london_end_hour_utc: int = 11
    ny_start_hour_utc: int = 12
    ny_end_hour_utc: int = 17
    level_recompute_hour_utc: int = 7
    day_boundary_utc_hour: int = 22
    round_level_radius_pips: float = 50.0


@dataclass
class _DailyAccumulator:
    trading_day: date
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float | None
    bar_index_start: int
    bar_index_end: int
    bar_count: int = 1

    @classmethod
    def from_bar(cls, current_bar: BarView, trading_day: date) -> "_DailyAccumulator":
        volume = _optional_bar_value(current_bar, "tick_volume", "volume")
        vol_value = float(volume) if volume is not None else None
        return cls(
            trading_day=trading_day,
            timestamp=_to_utc_timestamp(current_bar.timestamp).to_pydatetime(),
            open=float(current_bar.mid_open),
            high=float(current_bar.mid_high),
            low=float(current_bar.mid_low),
            close=float(current_bar.mid_close),
            volume=vol_value,
            bar_index_start=int(current_bar.bar_index),
            bar_index_end=int(current_bar.bar_index),
        )

    def update(self, current_bar: BarView) -> None:
        self.high = max(float(self.high), float(current_bar.mid_high))
        self.low = min(float(self.low), float(current_bar.mid_low))
        self.close = float(current_bar.mid_close)
        self.bar_index_end = int(current_bar.bar_index)
        self.bar_count += 1
        volume = _optional_bar_value(current_bar, "tick_volume", "volume")
        if volume is None or self.volume is None:
            self.volume = None
        else:
            self.volume += float(volume)

    def to_daily_bar(self) -> BarDaily:
        return BarDaily(
            trading_day=self.trading_day,
            timestamp=self.timestamp,
            open=float(self.open),
            high=float(self.high),
            low=float(self.low),
            close=float(self.close),
            volume=self.volume,
            bar_index_start=int(self.bar_index_start),
            bar_index_end=int(self.bar_index_end),
            bar_count=int(self.bar_count),
        )


@dataclass(frozen=True)
class _LevelSnapshot:
    trading_day: date
    version: int
    recompute_bar_index: int
    levels: tuple[float, ...]
    pdh: float | None
    pdl: float | None
    pwh: float | None
    pwl: float | None
    pmh: float | None
    pml: float | None
    round_levels: tuple[float, ...]
    has_cluster: bool


@dataclass
class _TradePlan:
    direction: Literal["long", "short"]
    session_end_ts: pd.Timestamp
    max_hold_until_bar: int
    tp1_price: float
    tp2_price: float
    tp1_done: bool = False
    tp2_done: bool = False
    runner_target_price: float | None = None
    last_trail_version: int = 0


def _to_utc_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _session_day_utc(ts: pd.Timestamp, day_boundary_utc_hour: int) -> date:
    return (ts - pd.Timedelta(hours=int(day_boundary_utc_hour))).date()


def _optional_bar_value(current_bar: BarView, *names: str) -> Any:
    for name in names:
        try:
            return getattr(current_bar, name)
        except (AttributeError, KeyError):
            continue
    return None


class MacroReversionStrategy(TrainableStrategyFamily):
    family_name = "macro_reversion"

    def __init__(self, config: MacroReversionConfig | None = None) -> None:
        self.config = config or MacroReversionConfig()
        self.family_name = self.config.family_name
        self._completed_daily_bars: list[BarDaily] = []
        self._current_daily: _DailyAccumulator | None = None
        self._level_snapshot: _LevelSnapshot | None = None
        self._last_recompute_day: date | None = None
        self._last_advanced_bar_index: int = -1
        self._level_version: int = 0
        self._cooldown_until_bar_index: int | None = None
        self._trade_plans: dict[int, _TradePlan] = {}

    def fit(self, history: HistoricalDataView) -> None:
        return None

    def evaluate(
        self,
        current_bar: BarView,
        history: HistoricalDataView,
        portfolio: PortfolioSnapshot,
    ) -> Signal | None:
        self._advance_state(current_bar)
        ts = _to_utc_timestamp(current_bar.timestamp)
        current_day = _session_day_utc(ts, self.config.day_boundary_utc_hour)

        if not self._in_session(ts):
            return None
        if self._cooldown_until_bar_index is not None and int(current_bar.bar_index) < int(self._cooldown_until_bar_index):
            return None
        if portfolio.open_positions:
            return None
        if self._level_snapshot is None or self._level_snapshot.trading_day != current_day:
            return None

        range_position = self.compute_range_position(float(current_bar.mid_close))
        bias = self.bias_from_range_position(range_position)
        if bias == "neutral" or range_position is None:
            return None

        if bias == "short":
            if float(range_position) <= float(self.config.short_entry_threshold):
                return None
            if not float(current_bar.mid_close) < float(current_bar.mid_open):
                return None
            nearest_level, distance = find_nearest_level(
                float(current_bar.mid_high),
                list(self._level_snapshot.levels),
                pip_size=self.config.pip_size,
            )
            if nearest_level is None or distance is None or float(distance) >= float(self.config.level_touch_distance_pips):
                return None
            nearby_levels = self._nearby_levels(float(nearest_level))
            if not is_level_cluster(
                nearby_levels,
                cluster_radius_pips=self.config.cluster_radius_pips,
                pip_size=self.config.pip_size,
            ):
                return None
            entry_price = float(current_bar.mid_close)
            return Signal(
                family=self.family_name,
                direction="short",
                stop_loss=self.stop_loss_price(entry_price, "short"),
                take_profit=self.tp1_price(entry_price, "short"),
                size=int(self.config.fixed_units),
                metadata={
                    "strategy": self.family_name,
                    "bias": bias,
                    "range_position": float(range_position),
                    "nearest_level": float(nearest_level),
                    "nearby_levels": list(nearby_levels),
                    "stop_loss_pips": float(self.config.stop_loss_pips),
                    "take_profit_pips": float(self.config.tp1_pips),
                    "tp1_pips": float(self.config.tp1_pips),
                    "tp2_pips": float(self.config.tp2_pips),
                    "fixed_units": int(self.config.fixed_units),
                    "levels_version": int(self._level_snapshot.version),
                },
            )

        if float(range_position) >= float(self.config.long_entry_threshold):
            return None
        if not float(current_bar.mid_close) > float(current_bar.mid_open):
            return None
        nearest_level, distance = find_nearest_level(
            float(current_bar.mid_low),
            list(self._level_snapshot.levels),
            pip_size=self.config.pip_size,
        )
        if nearest_level is None or distance is None or float(distance) >= float(self.config.level_touch_distance_pips):
            return None
        nearby_levels = self._nearby_levels(float(nearest_level))
        if not is_level_cluster(
            nearby_levels,
            cluster_radius_pips=self.config.cluster_radius_pips,
            pip_size=self.config.pip_size,
        ):
            return None
        entry_price = float(current_bar.mid_close)
        return Signal(
            family=self.family_name,
            direction="long",
            stop_loss=self.stop_loss_price(entry_price, "long"),
            take_profit=self.tp1_price(entry_price, "long"),
            size=int(self.config.fixed_units),
            metadata={
                "strategy": self.family_name,
                "bias": bias,
                "range_position": float(range_position),
                "nearest_level": float(nearest_level),
                "nearby_levels": list(nearby_levels),
                "stop_loss_pips": float(self.config.stop_loss_pips),
                "take_profit_pips": float(self.config.tp1_pips),
                "tp1_pips": float(self.config.tp1_pips),
                "tp2_pips": float(self.config.tp2_pips),
                "fixed_units": int(self.config.fixed_units),
                "levels_version": int(self._level_snapshot.version),
            },
        )

    def get_exit_conditions(
        self,
        position: PositionSnapshot,
        current_bar: BarView,
        history: HistoricalDataView,
    ) -> ExitAction | None:
        self._advance_state(current_bar)
        plan = self._trade_plans.get(int(position.trade_id))
        if plan is None:
            return None

        ts = _to_utc_timestamp(current_bar.timestamp)
        if ts >= plan.session_end_ts:
            return ExitAction(reason="session_close", exit_type="full", close_fraction=1.0)
        if int(current_bar.bar_index) >= int(plan.max_hold_until_bar):
            return ExitAction(reason="max_hold", exit_type="full", close_fraction=1.0)

        if not plan.tp1_done and self._target_touched(position.direction, current_bar, plan.tp1_price):
            plan.tp1_done = True
            return ExitAction(
                reason="tp1_partial",
                exit_type="partial",
                close_fraction=0.5,
                price=float(plan.tp1_price),
                new_take_profit=float(plan.tp2_price),
            )

        if plan.tp1_done and not plan.tp2_done and self._target_touched(position.direction, current_bar, plan.tp2_price):
            plan.tp2_done = True
            runner_target = plan.runner_target_price
            if runner_target is not None:
                return ExitAction(
                    reason="tp2_partial",
                    exit_type="partial",
                    close_fraction=0.5,
                    price=float(plan.tp2_price),
                    new_take_profit=float(runner_target),
                )
            return ExitAction(
                reason="tp2_partial",
                exit_type="partial",
                close_fraction=0.5,
                price=float(plan.tp2_price),
                new_take_profit=None,
            )

        if plan.tp2_done and self._level_snapshot is not None and plan.last_trail_version < int(self._level_snapshot.version):
            updated_target = self._runner_target_from_snapshot(plan.direction)
            updated_target = self._normalize_runner_target(updated_target, plan.direction, plan.tp2_price)
            if updated_target is not None:
                ratcheted = self._ratchet_runner_target(plan.runner_target_price, updated_target, plan.direction)
                plan.last_trail_version = int(self._level_snapshot.version)
                if ratcheted is not None and ratcheted != plan.runner_target_price:
                    plan.runner_target_price = float(ratcheted)
                    return ExitAction(
                        reason="runner_trail_update",
                        exit_type="none",
                        new_take_profit=float(ratcheted),
                    )
            else:
                plan.last_trail_version = int(self._level_snapshot.version)
        return None

    def on_position_opened(self, position: PositionSnapshot, signal: Signal, current_bar: BarView) -> None:
        ts = _to_utc_timestamp(current_bar.timestamp)
        direction = position.direction
        tp1 = self.tp1_price(float(position.entry_price), direction)
        tp2 = self.tp2_price(float(position.entry_price), direction)
        runner_target = self._runner_target_from_snapshot(direction)
        runner_target = self._normalize_runner_target(runner_target, direction, tp2)
        session_end = self._session_end_timestamp(ts)
        self._trade_plans[int(position.trade_id)] = _TradePlan(
            direction=direction,
            session_end_ts=session_end,
            max_hold_until_bar=int(position.entry_bar) + int(self.config.max_hold_bars),
            tp1_price=float(tp1),
            tp2_price=float(tp2),
            runner_target_price=float(runner_target) if runner_target is not None else None,
            last_trail_version=int(self._level_snapshot.version) if self._level_snapshot is not None else 0,
        )

    def on_position_closed(self, trade: ClosedTrade) -> None:
        self._cooldown_until_bar_index = int(trade.exit_bar) + int(self.config.cooldown_bars)
        if trade.event_type == "partial" and int(trade.remaining_units) > 0:
            return
        self._trade_plans.pop(int(trade.trade_id), None)

    def compute_range_position(self, current_price: float) -> float | None:
        if len(self._completed_daily_bars) < int(self.config.lookback_days):
            return None
        window = self._completed_daily_bars[-int(self.config.lookback_days) :]
        trailing_high = max(float(bar.high) for bar in window)
        trailing_low = min(float(bar.low) for bar in window)
        if trailing_high <= trailing_low:
            return None
        return (float(current_price) - trailing_low) / (trailing_high - trailing_low)

    def bias_from_range_position(self, range_position: float | None) -> Bias:
        if range_position is None:
            return "neutral"
        if float(range_position) > float(self.config.short_bias_threshold):
            return "short"
        if float(range_position) < float(self.config.long_bias_threshold):
            return "long"
        return "neutral"

    def stop_loss_price(self, entry_price: float, direction: Literal["long", "short"]) -> float:
        if direction == "long":
            return float(entry_price) - float(self.config.stop_loss_pips) * float(self.config.pip_size)
        return float(entry_price) + float(self.config.stop_loss_pips) * float(self.config.pip_size)

    def tp1_price(self, entry_price: float, direction: Literal["long", "short"]) -> float:
        if direction == "long":
            return float(entry_price) + float(self.config.tp1_pips) * float(self.config.pip_size)
        return float(entry_price) - float(self.config.tp1_pips) * float(self.config.pip_size)

    def tp2_price(self, entry_price: float, direction: Literal["long", "short"]) -> float:
        if direction == "long":
            return float(entry_price) + float(self.config.tp2_pips) * float(self.config.pip_size)
        return float(entry_price) - float(self.config.tp2_pips) * float(self.config.pip_size)

    def rebuild_daily_bars_from_history(self, history: HistoricalDataView) -> list[BarDaily]:
        bars = [history[idx].to_record() for idx in range(len(history))]
        return aggregate_to_daily(bars, day_boundary_utc_hour=self.config.day_boundary_utc_hour)

    def _advance_state(self, current_bar: BarView) -> None:
        bar_index = int(current_bar.bar_index)
        if bar_index == self._last_advanced_bar_index:
            return
        self._last_advanced_bar_index = bar_index
        self._ingest_daily_bar(current_bar)
        ts = _to_utc_timestamp(current_bar.timestamp)
        current_day = _session_day_utc(ts, self.config.day_boundary_utc_hour)
        if self._is_level_recompute_bar(ts) and current_day != self._last_recompute_day:
            self._recompute_levels(float(current_bar.mid_close), current_day, bar_index)

    def _ingest_daily_bar(self, current_bar: BarView) -> None:
        ts = _to_utc_timestamp(current_bar.timestamp)
        trading_day = _session_day_utc(ts, self.config.day_boundary_utc_hour)
        if self._current_daily is None:
            self._current_daily = _DailyAccumulator.from_bar(current_bar, trading_day)
            return
        if trading_day != self._current_daily.trading_day:
            self._completed_daily_bars.append(self._current_daily.to_daily_bar())
            self._current_daily = _DailyAccumulator.from_bar(current_bar, trading_day)
            return
        self._current_daily.update(current_bar)

    def _recompute_levels(self, current_price: float, current_day: date, bar_index: int) -> None:
        daily_bars = self._daily_bars_with_current()
        if not daily_bars:
            return
        current_day_index = len(daily_bars) - 1
        pdh, pdl = compute_pdh_pdl(daily_bars, current_day_index)
        pwh, pwl = compute_pwh_pwl(daily_bars, current_day_index)
        pmh, pml = compute_pmh_pml(daily_bars, current_day_index)
        round_levels = tuple(
            compute_round_levels(
                current_price,
                radius_pips=self.config.round_level_radius_pips,
                pip_size=self.config.pip_size,
            )
        )
        levels = tuple(float(level) for level in (pdh, pdl, pwh, pwl, pmh, pml) if level is not None) + round_levels
        self._level_version += 1
        self._level_snapshot = _LevelSnapshot(
            trading_day=current_day,
            version=int(self._level_version),
            recompute_bar_index=int(bar_index),
            levels=levels,
            pdh=float(pdh) if pdh is not None else None,
            pdl=float(pdl) if pdl is not None else None,
            pwh=float(pwh) if pwh is not None else None,
            pwl=float(pwl) if pwl is not None else None,
            pmh=float(pmh) if pmh is not None else None,
            pml=float(pml) if pml is not None else None,
            round_levels=round_levels,
            has_cluster=is_level_cluster(
                list(levels),
                cluster_radius_pips=self.config.cluster_radius_pips,
                pip_size=self.config.pip_size,
            ),
        )
        self._last_recompute_day = current_day

    def _daily_bars_with_current(self) -> list[BarDaily]:
        if self._current_daily is None:
            return list(self._completed_daily_bars)
        return list(self._completed_daily_bars) + [self._current_daily.to_daily_bar()]

    def _nearby_levels(self, anchor_level: float) -> list[float]:
        if self._level_snapshot is None:
            return []
        max_distance = float(self.config.cluster_radius_pips) * float(self.config.pip_size)
        return [
            float(level)
            for level in self._level_snapshot.levels
            if abs(float(level) - float(anchor_level)) <= max_distance + 1e-12
        ]

    def _runner_target_from_snapshot(self, direction: Literal["long", "short"]) -> float | None:
        if self._level_snapshot is None:
            return None
        if direction == "short":
            return self._level_snapshot.pdl
        return self._level_snapshot.pdh

    def _normalize_runner_target(
        self,
        target: float | None,
        direction: Literal["long", "short"],
        tp2_price: float,
    ) -> float | None:
        if target is None:
            return None
        if direction == "short" and float(target) < float(tp2_price):
            return float(target)
        if direction == "long" and float(target) > float(tp2_price):
            return float(target)
        return None

    def _ratchet_runner_target(
        self,
        current_target: float | None,
        candidate_target: float,
        direction: Literal["long", "short"],
    ) -> float | None:
        if current_target is None:
            return float(candidate_target)
        if direction == "short":
            return float(candidate_target) if float(candidate_target) < float(current_target) else None
        return float(candidate_target) if float(candidate_target) > float(current_target) else None

    def _target_touched(
        self,
        direction: Literal["long", "short"],
        current_bar: BarView,
        target_price: float,
    ) -> bool:
        if direction == "long":
            return float(current_bar.bid_high) >= float(target_price)
        return float(current_bar.ask_low) <= float(target_price)

    def _in_session(self, ts: pd.Timestamp) -> bool:
        minute_of_day = int(ts.hour) * 60 + int(ts.minute)
        london_start = int(self.config.london_start_hour_utc) * 60
        london_end = int(self.config.london_end_hour_utc) * 60
        ny_start = int(self.config.ny_start_hour_utc) * 60
        ny_end = int(self.config.ny_end_hour_utc) * 60
        return london_start <= minute_of_day < london_end or ny_start <= minute_of_day < ny_end

    def _is_level_recompute_bar(self, ts: pd.Timestamp) -> bool:
        return int(ts.hour) == int(self.config.level_recompute_hour_utc) and int(ts.minute) == 0

    def _session_end_timestamp(self, ts: pd.Timestamp) -> pd.Timestamp:
        if int(ts.hour) < int(self.config.london_end_hour_utc):
            return ts.normalize() + pd.Timedelta(hours=int(self.config.london_end_hour_utc))
        return ts.normalize() + pd.Timedelta(hours=int(self.config.ny_end_hour_utc))
