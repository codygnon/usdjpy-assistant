from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


@dataclass
class ManagedTrade:
    trade_id: str
    instrument: str
    direction: str
    entry_price: float
    initial_stop: float
    current_stop: float
    tp1: Optional[float] = None
    tp2: Optional[float] = None
    tp1_hit: bool = False
    tp2_hit: bool = False
    initial_units: int = 0
    current_units: int = 0
    entry_time: Optional[datetime] = None
    last_trail_update: Optional[datetime] = None
    user_provided_stop: bool = False


@dataclass(frozen=True)
class MonitorAction:
    timestamp: datetime
    trade_id: str
    action: str
    details: str
    old_value: Optional[float] = None
    new_value: Optional[float] = None


class PositionMonitor:
    """Monitors and manages open trades."""

    def __init__(self, client, risk_config, trailing_config, journal=None):
        self._client = client
        self._risk_config = risk_config
        self._trailing_config = trailing_config
        self._journal = journal
        self._managed: dict[str, ManagedTrade] = {}

    def register_trade(
        self,
        trade_id: str,
        entry_price: float,
        stop_loss: float,
        tp1: Optional[float],
        tp2: Optional[float],
        direction: str,
        units: int,
        user_provided_stop: bool = True,
        instrument: str = "USD_JPY",
    ) -> ManagedTrade:
        managed = ManagedTrade(
            trade_id=trade_id,
            instrument=instrument,
            direction=direction,
            entry_price=float(entry_price),
            initial_stop=float(stop_loss),
            current_stop=float(stop_loss),
            tp1=float(tp1) if tp1 is not None else None,
            tp2=float(tp2) if tp2 is not None else None,
            tp1_hit=False,
            tp2_hit=False,
            initial_units=abs(int(units)),
            current_units=abs(int(units)),
            entry_time=datetime.now(timezone.utc),
            user_provided_stop=bool(user_provided_stop),
        )
        self._managed[trade_id] = managed
        return managed

    def check_and_manage(self, instrument: str = "USD_JPY") -> list[MonitorAction]:
        actions: list[MonitorAction] = []
        try:
            price = self._client.get_price(instrument)
            open_trades = self._client.get_open_trades(instrument)
        except Exception as exc:
            return [self._record_action(MonitorAction(self._now(), "N/A", "error", f"Failed to fetch data: {exc}"))]

        open_ids = {trade.trade_id for trade in open_trades}
        closed_ids = [trade_id for trade_id in self._managed if trade_id not in open_ids]
        for trade_id in closed_ids:
            managed = self._managed.pop(trade_id)
            actions.append(
                self._record_action(
                    MonitorAction(self._now(), trade_id, "trade_closed", f"Trade {trade_id} no longer open — removed from monitor")
                )
            )
            if self._journal:
                self._journal.log_trade_closed(managed_trade=managed, trade_id=trade_id)

        for trade in open_trades:
            if trade.trade_id not in self._managed:
                actions.extend(self._adopt_trade(trade, price))

        current_mid = float(price.mid)
        for trade in self._client.get_open_trades(instrument):
            if trade.trade_id not in self._managed:
                continue
            managed = self._managed[trade.trade_id]

            if trade.stop_loss is None:
                actions.extend(self._set_emergency_stop(managed, trade, current_mid))

            tp1_actions: list[MonitorAction] = []
            if not managed.tp1_hit and managed.tp1 is not None:
                tp1_actions = self._check_tp1(managed, current_mid)
                actions.extend(tp1_actions)

            if not tp1_actions and not managed.tp2_hit and managed.tp2 is not None and managed.tp1_hit:
                actions.extend(self._check_tp2(managed, current_mid))

            if self._trailing_config.enabled:
                actions.extend(self._check_trailing(managed, managed.instrument))

        return actions

    def _adopt_trade(self, trade, price) -> list[MonitorAction]:
        actions: list[MonitorAction] = []
        stop = trade.stop_loss
        user_provided = stop is not None

        if stop is None:
            try:
                candles = self._client.get_candles(trade.instrument, "H4", 20)
                atr = self._compute_atr_from_candles(candles, 14)
                pip_size = 0.01 if "JPY" in trade.instrument else 0.0001
                stop_distance = min(
                    float(self._risk_config.default_stop_atr_multiple) * float(atr),
                    float(self._risk_config.catastrophic_stop_pips) * pip_size,
                )
                stop = trade.open_price - stop_distance if trade.direction == "long" else trade.open_price + stop_distance
            except Exception:
                pip_size = 0.01 if "JPY" in trade.instrument else 0.0001
                stop_distance = float(self._risk_config.catastrophic_stop_pips) * pip_size
                stop = trade.open_price - stop_distance if trade.direction == "long" else trade.open_price + stop_distance

            try:
                self._client.set_stop_loss(trade.trade_id, stop)
                actions.append(
                    self._record_action(
                        MonitorAction(self._now(), trade.trade_id, "set_stop", f"Adopted trade — auto stop set at {stop:.3f}", new_value=stop)
                    )
                )
            except Exception as exc:
                actions.append(self._record_action(MonitorAction(self._now(), trade.trade_id, "error", f"Failed to set stop: {exc}")))

        stop_distance = abs(float(trade.open_price) - float(stop))
        if trade.direction == "long":
            tp1 = float(trade.open_price) + stop_distance * float(self._risk_config.tp1_ratio)
            tp2 = float(trade.open_price) + stop_distance * float(self._risk_config.tp2_ratio)
        else:
            tp1 = float(trade.open_price) - stop_distance * float(self._risk_config.tp1_ratio)
            tp2 = float(trade.open_price) - stop_distance * float(self._risk_config.tp2_ratio)

        self.register_trade(
            trade_id=trade.trade_id,
            entry_price=trade.open_price,
            stop_loss=float(stop),
            tp1=tp1,
            tp2=tp2,
            direction=trade.direction,
            units=trade.units,
            user_provided_stop=user_provided,
            instrument=trade.instrument,
        )
        actions.append(
            self._record_action(
                MonitorAction(
                    self._now(),
                    trade.trade_id,
                    "adopted",
                    f"Adopted {trade.direction} {abs(int(trade.units)):,} units @ {trade.open_price:.3f} | SL {stop:.3f} | TP1 {tp1:.3f} | TP2 {tp2:.3f}",
                )
            )
        )
        return actions

    def _set_emergency_stop(self, managed: ManagedTrade, trade, current_price: float) -> list[MonitorAction]:
        try:
            self._client.set_stop_loss(trade.trade_id, managed.current_stop)
            return [
                self._record_action(
                    MonitorAction(
                        self._now(),
                        trade.trade_id,
                        "catastrophic_stop",
                        f"NO STOP DETECTED — emergency stop set at {managed.current_stop:.3f}",
                        new_value=managed.current_stop,
                    )
                )
            ]
        except Exception as exc:
            return [self._record_action(MonitorAction(self._now(), trade.trade_id, "error", f"CRITICAL: Failed to set emergency stop: {exc}"))]

    def _check_tp1(self, managed: ManagedTrade, current_price: float) -> list[MonitorAction]:
        if managed.tp1 is None or managed.tp1_hit:
            return []
        hit = (managed.direction == "long" and current_price >= managed.tp1) or (
            managed.direction == "short" and current_price <= managed.tp1
        )
        if not hit:
            return []

        close_units = int((managed.current_units * float(self._risk_config.tp1_close_fraction)) // 1000 * 1000)
        if close_units <= 0:
            return []
        try:
            self._client.close_trade(managed.trade_id, close_units)
            managed.tp1_hit = True
            managed.current_units -= close_units
            try:
                self._client.set_stop_loss(managed.trade_id, managed.entry_price)
                managed.current_stop = float(managed.entry_price)
            except Exception:
                pass
            return [
                self._record_action(
                    MonitorAction(
                        self._now(),
                        managed.trade_id,
                        "partial_close_tp1",
                        f"TP1 hit @ {current_price:.3f} — closed {close_units:,} units ({managed.current_units:,} remaining) — stop moved to BE {managed.entry_price:.3f}",
                    )
                )
            ]
        except Exception as exc:
            return [self._record_action(MonitorAction(self._now(), managed.trade_id, "error", f"Failed TP1 partial close: {exc}"))]

    def _check_tp2(self, managed: ManagedTrade, current_price: float) -> list[MonitorAction]:
        if managed.tp2 is None:
            return []
        hit = (managed.direction == "long" and current_price >= managed.tp2) or (
            managed.direction == "short" and current_price <= managed.tp2
        )
        if not hit:
            return []

        close_units = int((managed.initial_units * float(self._risk_config.tp2_close_fraction)) // 1000 * 1000)
        close_units = min(close_units, managed.current_units)
        if close_units <= 0:
            return []
        try:
            self._client.close_trade(managed.trade_id, close_units)
            managed.tp2_hit = True
            managed.current_units -= close_units
            return [
                self._record_action(
                    MonitorAction(
                        self._now(),
                        managed.trade_id,
                        "partial_close_tp2",
                        f"TP2 hit @ {current_price:.3f} — closed {close_units:,} units ({managed.current_units:,} remaining as runner)",
                    )
                )
            ]
        except Exception as exc:
            return [self._record_action(MonitorAction(self._now(), managed.trade_id, "error", f"Failed TP2 partial close: {exc}"))]

    def _check_trailing(self, managed: ManagedTrade, instrument: str) -> list[MonitorAction]:
        try:
            candles = self._client.get_candles(
                instrument,
                self._trailing_config.timeframe,
                self._trailing_config.lookback_bars + 10,
            )
            if len(candles) < self._trailing_config.lookback_bars:
                return []
            atr = self._compute_atr_from_candles(candles, 14)
            recent = candles[-self._trailing_config.lookback_bars :]
            buffer = float(self._trailing_config.atr_buffer) * atr
            if managed.direction == "long":
                swing_low = min(candle.low for candle in recent)
                new_stop = float(swing_low - buffer)
                if self._trailing_config.only_tighten and new_stop <= managed.current_stop:
                    return []
            else:
                swing_high = max(candle.high for candle in recent)
                new_stop = float(swing_high + buffer)
                if self._trailing_config.only_tighten and new_stop >= managed.current_stop:
                    return []
            old_stop = managed.current_stop
            self._client.set_stop_loss(managed.trade_id, new_stop)
            managed.current_stop = new_stop
            managed.last_trail_update = self._now()
            return [
                self._record_action(
                    MonitorAction(
                        self._now(),
                        managed.trade_id,
                        "trail_stop",
                        f"Trailing stop moved {old_stop:.3f} → {new_stop:.3f}",
                        old_value=old_stop,
                        new_value=new_stop,
                    )
                )
            ]
        except Exception as exc:
            return [self._record_action(MonitorAction(self._now(), managed.trade_id, "error", f"Trailing stop error: {exc}"))]

    @staticmethod
    def _compute_atr_from_candles(candles: list, period: int = 14) -> float:
        complete = [candle for candle in candles if candle.complete]
        if len(complete) < 2:
            return 0.0
        true_ranges: list[float] = []
        for index in range(1, len(complete)):
            high = float(complete[index].high)
            low = float(complete[index].low)
            prev_close = float(complete[index - 1].close)
            true_ranges.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
        recent = true_ranges[-period:]
        if not recent:
            return 0.0
        return sum(recent) / len(recent)

    def get_status(self) -> dict:
        return {
            "managed_count": len(self._managed),
            "trades": {
                trade_id: {
                    "direction": trade.direction,
                    "entry": trade.entry_price,
                    "current_stop": trade.current_stop,
                    "tp1": trade.tp1,
                    "tp1_hit": trade.tp1_hit,
                    "tp2": trade.tp2,
                    "tp2_hit": trade.tp2_hit,
                    "units": trade.current_units,
                    "initial_units": trade.initial_units,
                }
                for trade_id, trade in self._managed.items()
            },
        }

    def _record_action(self, action: MonitorAction) -> MonitorAction:
        if self._journal is not None:
            self._journal.log_action(action)
        return action

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)
