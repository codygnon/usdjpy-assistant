from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Optional

from .oanda_client import AccountSummary, CandleBar, OpenTrade, PendingOrder, PriceSnapshot


class MockOandaClient:
    """Drop-in replacement for OandaClient in tests."""

    def __init__(self):
        self._account_id = "mock-account"
        self._currency = "USD"
        self._balance = 100000.0
        self._unrealized_pnl = 0.0
        self._margin_used = 0.0
        self._open_trades: list[OpenTrade] = []
        self._prices: dict[str, PriceSnapshot] = {}
        self._candles: dict[tuple[str, str], list[CandleBar]] = {}
        self._trade_modifications: list[dict] = []
        self._orders_placed: list[dict] = []
        self._pending_orders: list[PendingOrder] = []

    def set_account(self, balance: float = 100000.0, unrealized_pnl: float = 0.0, margin_used: float = 0.0):
        self._balance = float(balance)
        self._unrealized_pnl = float(unrealized_pnl)
        self._margin_used = float(margin_used)

    def add_open_trade(
        self,
        *,
        trade_id: str,
        instrument: str = "USD_JPY",
        direction: str = "long",
        units: int = 100000,
        open_price: float = 150.0,
        open_time: Optional[datetime] = None,
        unrealized_pnl: float = 0.0,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        trailing_stop_distance: Optional[float] = None,
    ) -> None:
        signed_units = abs(int(units)) if direction == "long" else -abs(int(units))
        self._open_trades = [trade for trade in self._open_trades if trade.trade_id != trade_id]
        self._open_trades.append(
            OpenTrade(
                trade_id=str(trade_id),
                instrument=instrument,
                direction=direction,
                units=signed_units,
                open_price=float(open_price),
                open_time=open_time or datetime.now(timezone.utc),
                unrealized_pnl=float(unrealized_pnl),
                stop_loss=float(stop_loss) if stop_loss is not None else None,
                take_profit=float(take_profit) if take_profit is not None else None,
                trailing_stop_distance=float(trailing_stop_distance) if trailing_stop_distance is not None else None,
            )
        )

    def set_price(self, instrument: str, bid: float, ask: float):
        pip_size = 0.01 if "JPY" in instrument else 0.0001
        self._prices[instrument] = PriceSnapshot(
            instrument=instrument,
            bid=float(bid),
            ask=float(ask),
            mid=(float(bid) + float(ask)) / 2.0,
            spread_pips=(float(ask) - float(bid)) / pip_size,
            timestamp=datetime.now(timezone.utc),
        )

    def set_candles(self, instrument: str, candles: list[CandleBar], granularity: str = "H4"):
        self._candles[(instrument, granularity)] = list(candles)

    def get_account_summary(self) -> AccountSummary:
        return AccountSummary(
            account_id=self._account_id,
            balance=self._balance,
            unrealized_pnl=self._unrealized_pnl,
            equity=self._balance + self._unrealized_pnl,
            margin_used=self._margin_used,
            margin_available=(self._balance + self._unrealized_pnl - self._margin_used),
            open_trade_count=len(self._open_trades),
            currency=self._currency,
        )

    def get_open_trades(self, instrument: Optional[str] = None) -> list[OpenTrade]:
        if instrument is None:
            return list(self._open_trades)
        return [trade for trade in self._open_trades if trade.instrument == instrument]

    def get_pending_orders(self, instrument: Optional[str] = None) -> list[PendingOrder]:
        if instrument is None:
            return list(self._pending_orders)
        return [order for order in self._pending_orders if order.instrument == instrument]

    def get_price(self, instrument: str) -> PriceSnapshot:
        if instrument not in self._prices:
            self.set_price(instrument, 150.0, 150.02)
        return self._prices[instrument]

    def get_candles(self, instrument: str, granularity: str = "H4", count: int = 100) -> list[CandleBar]:
        candles = self._candles.get((instrument, granularity), self._candles.get((instrument, "H4"), []))
        return list(candles[-count:])

    def set_stop_loss(self, trade_id: str, price: float) -> dict:
        self._trade_modifications.append({"action": "set_stop_loss", "trade_id": trade_id, "price": float(price)})
        self._replace_trade(trade_id, stop_loss=float(price))
        return {"tradeID": trade_id, "stopLossOrder": {"price": f"{price:.3f}"}}

    def set_take_profit(self, trade_id: str, price: float) -> dict:
        self._trade_modifications.append({"action": "set_take_profit", "trade_id": trade_id, "price": float(price)})
        self._replace_trade(trade_id, take_profit=float(price))
        return {"tradeID": trade_id, "takeProfitOrder": {"price": f"{price:.3f}"}}

    def set_stop_and_tp(self, trade_id, stop_loss, take_profit=None) -> dict:
        self._trade_modifications.append(
            {
                "action": "set_stop_and_tp",
                "trade_id": trade_id,
                "stop_loss": float(stop_loss),
                "take_profit": float(take_profit) if take_profit is not None else None,
            }
        )
        self._replace_trade(trade_id, stop_loss=float(stop_loss), take_profit=float(take_profit) if take_profit is not None else None)
        return {"tradeID": trade_id}

    def close_trade(self, trade_id: str, units=None) -> dict:
        self._trade_modifications.append({"action": "close_trade", "trade_id": trade_id, "units": units})
        for index, trade in enumerate(self._open_trades):
            if trade.trade_id != trade_id:
                continue
            if units is None or abs(int(units)) >= abs(trade.units):
                self._open_trades.pop(index)
                return {"tradeClosed": {"tradeID": trade_id}}
            remaining_units = abs(trade.units) - abs(int(units))
            signed_units = remaining_units if trade.direction == "long" else -remaining_units
            self._open_trades[index] = replace(trade, units=signed_units)
            return {"tradeReduced": {"tradeID": trade_id, "units": str(abs(int(units)))}, "remainingUnits": str(remaining_units)}
        return {"tradeID": trade_id, "status": "not_found"}

    def cancel_order(self, order_id: str) -> dict:
        self._pending_orders = [order for order in self._pending_orders if order.order_id != str(order_id)]
        return {"orderCancelTransaction": {"orderID": str(order_id)}}

    def place_market_order(self, instrument, units, stop_loss=None, take_profit=None) -> dict:
        trade_id = str(len(self._orders_placed) + 1)
        snapshot = self.get_price(instrument)
        direction = "long" if int(units) > 0 else "short"
        open_price = snapshot.ask if direction == "long" else snapshot.bid
        self._orders_placed.append(
            {
                "trade_id": trade_id,
                "instrument": instrument,
                "units": int(units),
                "stop_loss": float(stop_loss) if stop_loss is not None else None,
                "take_profit": float(take_profit) if take_profit is not None else None,
            }
        )
        self.add_open_trade(
            trade_id=trade_id,
            instrument=instrument,
            direction=direction,
            units=abs(int(units)),
            open_price=open_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        return {"orderFillTransaction": {"tradeOpened": {"tradeID": trade_id}}}

    def place_limit_order(
        self,
        instrument,
        units,
        price,
        *,
        stop_loss=None,
        take_profit=None,
        gtd_time=None,
        comment=None,
    ) -> dict:
        order_id = str(len(self._orders_placed) + len(self._pending_orders) + 1)
        self._pending_orders.append(
            PendingOrder(
                order_id=order_id,
                instrument=instrument,
                units=int(units),
                price=float(price),
                create_time=datetime.now(timezone.utc),
                time_in_force="GTD" if gtd_time is not None else "GTC",
                gtd_time=gtd_time,
                stop_loss_on_fill=float(stop_loss) if stop_loss is not None else None,
                take_profit_on_fill=float(take_profit) if take_profit is not None else None,
                comment=str(comment) if comment is not None else None,
            )
        )
        return {"orderCreateTransaction": {"id": order_id}}

    def test_connection(self) -> bool:
        return True

    def _replace_trade(self, trade_id: str, **changes) -> None:
        for index, trade in enumerate(self._open_trades):
            if trade.trade_id == trade_id:
                self._open_trades[index] = replace(trade, **changes)
                break
