from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict

from .daily_levels import compute_pdh_pdl
from .models import BarRecord, ClosedTrade, ExitAction, PortfolioSnapshot, PositionSnapshot, Signal
from .strategy import BarView, HistoricalDataView, TrainableStrategyFamily
from .synthetic_bars import Bar5M, BarDaily, aggregate_to_5m, aggregate_to_daily, compute_ema


Direction = Literal["long", "short"]


class BBReversionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    family_name: str = "bb_reversion"
    pip_size: float = 0.01
    fixed_units: int = 200_000
    bb_period: int = 20
    bb_stddev: float = 2.0
    ema_period: int = 50
    warmup_5m_bars: int = 50
    min_warmup_1m_bars: int = 250
    min_bb_width: float = 0.001
    trend_tolerance_pips: float = 10.0
    stop_loss_pips: float = 20.0
    pdh_pdl_reject_distance_pips: float = 15.0
    cooldown_bars_after_sl: int = 10
    max_daily_trades: int = 8
    max_hold_bars: int = 120
    day_boundary_utc_hour: int = 22
    london_start_hour_utc: int = 7
    london_end_hour_utc: int = 11
    ny_start_hour_utc: int = 12
    ny_end_hour_utc: int = 17


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
        return cls(
            trading_day=trading_day,
            timestamp=_to_utc_timestamp(current_bar.timestamp).to_pydatetime(),
            open=float(current_bar.mid_open),
            high=float(current_bar.mid_high),
            low=float(current_bar.mid_low),
            close=float(current_bar.mid_close),
            volume=float(volume) if volume is not None else None,
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
class _IndicatorSnapshot:
    five_minute_timestamp: datetime
    bb_mid: float
    bb_upper: float
    bb_lower: float
    bb_width: float
    ema_5m_50: float


@dataclass
class _TradePlan:
    direction: Direction
    session_end_ts: pd.Timestamp
    max_hold_until_bar: int
    tp1_price: float
    tp2_price: float
    tp1_done: bool = False


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


def bb_reversion_session_open_utc(ts: datetime | pd.Timestamp, config: BBReversionConfig | None = None) -> bool:
    cfg = config or BBReversionConfig()
    stamp = _to_utc_timestamp(ts)
    minute_of_day = int(stamp.hour) * 60 + int(stamp.minute)
    london_start = int(cfg.london_start_hour_utc) * 60
    london_end = int(cfg.london_end_hour_utc) * 60
    ny_start = int(cfg.ny_start_hour_utc) * 60
    ny_end = int(cfg.ny_end_hour_utc) * 60
    return london_start <= minute_of_day < london_end or ny_start <= minute_of_day < ny_end


def compute_bollinger_bands(
    closes: list[float] | tuple[float, ...],
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[float | None, float | None, float | None]:
    if period <= 0 or len(closes) < period:
        return None, None, None
    window = [float(value) for value in closes[-period:]]
    mean = float(sum(window) / period)
    variance = float(sum((value - mean) ** 2 for value in window) / period)
    std_dev = math.sqrt(variance)
    return mean, float(mean + num_std * std_dev), float(mean - num_std * std_dev)


class BBReversionStrategy(TrainableStrategyFamily):
    family_name = "bb_reversion"

    def __init__(self, config: BBReversionConfig | None = None) -> None:
        self.config = config or BBReversionConfig()
        self.family_name = self.config.family_name
        self._recent_1m_records: list[BarRecord] = []
        self._completed_5m: list[Bar5M] = []
        self._completed_5m_closes: list[float] = []
        self._last_completed_5m_timestamp: pd.Timestamp | None = None
        self._ema_5m_50: float | None = None
        self._indicator_snapshot: _IndicatorSnapshot | None = None
        self._completed_daily_bars: list[BarDaily] = []
        self._current_daily: _DailyAccumulator | None = None
        self._daily_trade_count: int = 0
        self._cooldown_until_bar_index: int | None = None
        self._trade_plans: dict[int, _TradePlan] = {}
        self._last_advanced_bar_index: int = -1

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

        if int(current_bar.bar_index) < int(self.config.min_warmup_1m_bars):
            return None
        if len(self._completed_5m) < int(self.config.warmup_5m_bars):
            return None
        if not bb_reversion_session_open_utc(ts, self.config):
            return None
        if self._cooldown_until_bar_index is not None and int(current_bar.bar_index) < int(self._cooldown_until_bar_index):
            return None
        if portfolio.open_positions:
            return None
        if int(self._daily_trade_count) >= int(self.config.max_daily_trades):
            return None

        snapshot = self.current_indicator_snapshot()
        if snapshot is None:
            return None
        if float(snapshot.bb_width) <= float(self.config.min_bb_width):
            return None

        current_price = float(current_bar.mid_close)
        pdh, pdl = self.current_pdh_pdl()

        long_signal = (
            float(current_bar.mid_low) <= float(snapshot.bb_lower)
            and float(current_bar.mid_close) > float(snapshot.bb_lower)
            and self._long_trend_ok(current_price, float(snapshot.ema_5m_50))
            and not self._within_pips(current_price, pdh, self.config.pdh_pdl_reject_distance_pips)
        )
        short_signal = (
            float(current_bar.mid_high) >= float(snapshot.bb_upper)
            and float(current_bar.mid_close) < float(snapshot.bb_upper)
            and self._short_trend_ok(current_price, float(snapshot.ema_5m_50))
            and not self._within_pips(current_price, pdl, self.config.pdh_pdl_reject_distance_pips)
        )

        if long_signal and short_signal:
            return None
        if long_signal:
            return self._build_signal("long", current_price, snapshot)
        if short_signal:
            return self._build_signal("short", current_price, snapshot)
        return None

    def get_exit_conditions(
        self,
        position: PositionSnapshot,
        current_bar: BarView,
        history: HistoricalDataView,
    ) -> ExitAction | None:
        if position.family != self.family_name:
            return None

        self._advance_state(current_bar)
        plan = self._trade_plans.get(int(position.trade_id))
        if plan is None:
            return None

        ts = _to_utc_timestamp(current_bar.timestamp)
        if ts >= plan.session_end_ts:
            return ExitAction(
                reason="session_close",
                exit_type="full",
                close_fraction=1.0,
                price=float(current_bar.bid_close if position.direction == "long" else current_bar.ask_close),
            )
        if int(current_bar.bar_index) >= int(plan.max_hold_until_bar):
            return ExitAction(
                reason="max_hold",
                exit_type="full",
                close_fraction=1.0,
                price=float(current_bar.bid_close if position.direction == "long" else current_bar.ask_close),
            )

        if position.direction == "long":
            tp1_hit = float(current_bar.bid_high) >= float(plan.tp1_price)
            tp2_hit = float(current_bar.bid_high) >= float(plan.tp2_price)
        else:
            tp1_hit = float(current_bar.ask_low) <= float(plan.tp1_price)
            tp2_hit = float(current_bar.ask_low) <= float(plan.tp2_price)

        if not plan.tp1_done and tp1_hit:
            plan.tp1_done = True
            return ExitAction(
                reason="tp1",
                exit_type="partial",
                close_fraction=0.5,
                price=float(plan.tp1_price),
                new_take_profit=float(plan.tp2_price),
            )

        if plan.tp1_done and tp2_hit:
            return ExitAction(
                reason="tp2",
                exit_type="full",
                close_fraction=1.0,
                price=float(plan.tp2_price),
            )

        return None

    def on_position_opened(self, position: PositionSnapshot, signal: Signal, current_bar: BarView) -> None:
        if position.family != self.family_name:
            return

        self._advance_state(current_bar)
        snapshot = self.current_indicator_snapshot()
        if snapshot is None:
            current_ts = _to_utc_timestamp(current_bar.timestamp).to_pydatetime()
            snapshot = _IndicatorSnapshot(
                five_minute_timestamp=current_ts,
                bb_mid=float(signal.metadata.get("bb_mid", position.entry_price)),
                bb_upper=float(signal.metadata.get("bb_upper", position.entry_price)),
                bb_lower=float(signal.metadata.get("bb_lower", position.entry_price)),
                bb_width=float(signal.metadata.get("bb_width", 0.0)),
                ema_5m_50=float(signal.metadata.get("ema_5m_50", position.entry_price)),
            )

        ts = _to_utc_timestamp(current_bar.timestamp)
        self._daily_trade_count += 1
        self._trade_plans[int(position.trade_id)] = _TradePlan(
            direction=position.direction,
            session_end_ts=self._session_end_timestamp(ts),
            max_hold_until_bar=int(position.entry_bar) + int(self.config.max_hold_bars),
            tp1_price=float(snapshot.bb_mid),
            tp2_price=float(snapshot.bb_upper if position.direction == "long" else snapshot.bb_lower),
        )

    def on_position_closed(self, trade: ClosedTrade) -> None:
        if trade.family != self.family_name:
            return
        if trade.exit_reason in {"stop_loss", "worst_case_stop"}:
            self._cooldown_until_bar_index = int(trade.exit_bar) + int(self.config.cooldown_bars_after_sl)
        if int(getattr(trade, "remaining_units", 0) or 0) <= 0:
            self._trade_plans.pop(int(trade.trade_id), None)

    def current_indicator_snapshot(self) -> _IndicatorSnapshot | None:
        return self._indicator_snapshot

    def current_pdh_pdl(self) -> tuple[float | None, float | None]:
        if self._current_daily is None:
            return None, None
        daily_bars = [*self._completed_daily_bars, self._current_daily.to_daily_bar()]
        if len(daily_bars) < 2:
            return None, None
        return compute_pdh_pdl(daily_bars, len(self._completed_daily_bars))

    def stop_loss_price(self, entry_price: float, direction: Direction) -> float:
        offset = float(self.config.stop_loss_pips) * float(self.config.pip_size)
        if direction == "long":
            return float(entry_price - offset)
        return float(entry_price + offset)

    def rebuild_daily_bars_from_history(self, history: HistoricalDataView) -> list[BarDaily]:
        return aggregate_to_daily([bar.to_record() for bar in history.iter_bars()], day_boundary_utc_hour=self.config.day_boundary_utc_hour)

    def _build_signal(self, direction: Direction, current_price: float, snapshot: _IndicatorSnapshot) -> Signal:
        return Signal(
            family=self.family_name,
            direction=direction,
            stop_loss=self.stop_loss_price(current_price, direction),
            take_profit=None,
            size=int(self.config.fixed_units),
            metadata={
                "stop_loss_pips": float(self.config.stop_loss_pips),
                "bb_mid": float(snapshot.bb_mid),
                "bb_upper": float(snapshot.bb_upper),
                "bb_lower": float(snapshot.bb_lower),
                "bb_width": float(snapshot.bb_width),
                "ema_5m_50": float(snapshot.ema_5m_50),
                "fixed_units": int(self.config.fixed_units),
            },
        )

    def _advance_state(self, current_bar: BarView) -> None:
        bar_index = int(current_bar.bar_index)
        if bar_index <= int(self._last_advanced_bar_index):
            return

        ts = _to_utc_timestamp(current_bar.timestamp)
        trading_day = _session_day_utc(ts, self.config.day_boundary_utc_hour)

        if self._current_daily is None:
            self._current_daily = _DailyAccumulator.from_bar(current_bar, trading_day)
        elif self._current_daily.trading_day != trading_day:
            self._completed_daily_bars.append(self._current_daily.to_daily_bar())
            self._current_daily = _DailyAccumulator.from_bar(current_bar, trading_day)
            self._daily_trade_count = 0
        else:
            self._current_daily.update(current_bar)

        self._recent_1m_records.append(current_bar.to_record())
        if len(self._recent_1m_records) > 32:
            self._recent_1m_records = self._recent_1m_records[-32:]

        synthetic_5m = aggregate_to_5m(self._recent_1m_records)
        for bar_5m in synthetic_5m:
            bar_ts = _to_utc_timestamp(bar_5m.timestamp)
            if self._last_completed_5m_timestamp is not None and bar_ts <= self._last_completed_5m_timestamp:
                continue
            self._completed_5m.append(bar_5m)
            self._completed_5m_closes.append(float(bar_5m.close))
            self._last_completed_5m_timestamp = bar_ts
            self._update_indicator_snapshot(bar_5m)

        self._last_advanced_bar_index = bar_index

    def _update_indicator_snapshot(self, latest_5m_bar: Bar5M) -> None:
        closes = self._completed_5m_closes
        period = int(self.config.ema_period)
        if len(closes) < period:
            self._ema_5m_50 = None
            self._indicator_snapshot = None
            return

        if len(closes) == period or self._ema_5m_50 is None:
            self._ema_5m_50 = float(sum(closes[-period:]) / period)
        else:
            multiplier = 2.0 / (period + 1.0)
            self._ema_5m_50 = float((closes[-1] - self._ema_5m_50) * multiplier + self._ema_5m_50)

        bb_mid, bb_upper, bb_lower = compute_bollinger_bands(
            closes,
            period=int(self.config.bb_period),
            num_std=float(self.config.bb_stddev),
        )
        if bb_mid is None or bb_upper is None or bb_lower is None or self._ema_5m_50 is None:
            self._indicator_snapshot = None
            return

        width = 0.0 if abs(bb_mid) <= 1e-12 else float((bb_upper - bb_lower) / bb_mid)
        self._indicator_snapshot = _IndicatorSnapshot(
            five_minute_timestamp=latest_5m_bar.timestamp,
            bb_mid=float(bb_mid),
            bb_upper=float(bb_upper),
            bb_lower=float(bb_lower),
            bb_width=float(width),
            ema_5m_50=float(self._ema_5m_50),
        )

    def _long_trend_ok(self, price: float, ema_5m_50: float) -> bool:
        tolerance = float(self.config.trend_tolerance_pips) * float(self.config.pip_size)
        return float(price) >= float(ema_5m_50) or (float(price) < float(ema_5m_50) and (float(ema_5m_50) - float(price)) < tolerance)

    def _short_trend_ok(self, price: float, ema_5m_50: float) -> bool:
        tolerance = float(self.config.trend_tolerance_pips) * float(self.config.pip_size)
        return float(price) <= float(ema_5m_50) or (float(price) > float(ema_5m_50) and (float(price) - float(ema_5m_50)) < tolerance)

    def _within_pips(self, price: float, level: float | None, distance_pips: float) -> bool:
        if level is None:
            return False
        return abs(float(price) - float(level)) <= float(distance_pips) * float(self.config.pip_size)

    def _session_end_timestamp(self, ts: pd.Timestamp) -> pd.Timestamp:
        if int(ts.hour) < int(self.config.london_end_hour_utc):
            return ts.normalize() + pd.Timedelta(hours=int(self.config.london_end_hour_utc))
        return ts.normalize() + pd.Timedelta(hours=int(self.config.ny_end_hour_utc))


def verify_bb_ema_sequence(values: list[float], period: int) -> list[float]:
    return compute_ema(values, period)
