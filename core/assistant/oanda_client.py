from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import requests


class OandaAPIError(Exception):
    """Raised when OANDA API returns an error."""

    def __init__(self, status_code: int, message: str, endpoint: str):
        self.status_code = status_code
        self.endpoint = endpoint
        super().__init__(f"OANDA API error {status_code} on {endpoint}: {message}")


@dataclass(frozen=True)
class AccountSummary:
    account_id: str
    balance: float
    unrealized_pnl: float
    equity: float
    margin_used: float
    margin_available: float
    open_trade_count: int
    currency: str


@dataclass(frozen=True)
class OpenTrade:
    trade_id: str
    instrument: str
    direction: str
    units: int
    open_price: float
    open_time: datetime
    unrealized_pnl: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    trailing_stop_distance: Optional[float]


@dataclass(frozen=True)
class PriceSnapshot:
    instrument: str
    bid: float
    ask: float
    mid: float
    spread_pips: float
    timestamp: datetime


@dataclass(frozen=True)
class CandleBar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    complete: bool


class OandaClient:
    """OANDA v20 REST API client."""

    def __init__(self, account_id: str, api_token: str, environment: str = "practice"):
        self._account_id = account_id
        self._api_token = api_token
        if environment == "live":
            self._base_url = "https://api-fxtrade.oanda.com"
        else:
            self._base_url = "https://api-fxpractice.oanda.com"
        self._headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        }

    def _get(self, endpoint: str, params: Optional[dict] = None) -> dict:
        url = f"{self._base_url}{endpoint}"
        response = requests.get(url, headers=self._headers, params=params, timeout=10)
        if response.status_code != 200:
            raise OandaAPIError(response.status_code, response.text, endpoint)
        return response.json()

    def _put(self, endpoint: str, data: dict) -> dict:
        url = f"{self._base_url}{endpoint}"
        response = requests.put(url, headers=self._headers, json=data, timeout=10)
        if response.status_code not in (200, 201):
            raise OandaAPIError(response.status_code, response.text, endpoint)
        return response.json()

    def _post(self, endpoint: str, data: dict) -> dict:
        url = f"{self._base_url}{endpoint}"
        response = requests.post(url, headers=self._headers, json=data, timeout=10)
        if response.status_code not in (200, 201):
            raise OandaAPIError(response.status_code, response.text, endpoint)
        return response.json()

    def _trade_orders_put(self, endpoint: str, data: dict) -> dict:
        return self._put(endpoint, data)

    def get_account_summary(self) -> AccountSummary:
        data = self._get(f"/v3/accounts/{self._account_id}/summary")
        account = data["account"]
        balance = float(account["balance"])
        unrealized = float(account["unrealizedPL"])
        return AccountSummary(
            account_id=account["id"],
            balance=balance,
            unrealized_pnl=unrealized,
            equity=balance + unrealized,
            margin_used=float(account["marginUsed"]),
            margin_available=float(account["marginAvailable"]),
            open_trade_count=int(account["openTradeCount"]),
            currency=account["currency"],
        )

    def get_price(self, instrument: str) -> PriceSnapshot:
        data = self._get(f"/v3/accounts/{self._account_id}/pricing", params={"instruments": instrument})
        price = data["prices"][0]
        bid = float(price["bids"][0]["price"])
        ask = float(price["asks"][0]["price"])
        mid = (bid + ask) / 2.0
        pip_size = 0.01 if "JPY" in instrument else 0.0001
        return PriceSnapshot(
            instrument=instrument,
            bid=bid,
            ask=ask,
            mid=mid,
            spread_pips=(ask - bid) / pip_size,
            timestamp=datetime.fromisoformat(price["time"].replace("Z", "+00:00")),
        )

    def get_open_trades(self, instrument: Optional[str] = None) -> list[OpenTrade]:
        data = self._get(f"/v3/accounts/{self._account_id}/openTrades")
        trades: list[OpenTrade] = []
        for trade in data.get("trades", []):
            if instrument is not None and trade["instrument"] != instrument:
                continue
            units = int(trade["currentUnits"])
            stop_loss = None
            if trade.get("stopLossOrder"):
                stop_loss = float(trade["stopLossOrder"]["price"])
            take_profit = None
            if trade.get("takeProfitOrder"):
                take_profit = float(trade["takeProfitOrder"]["price"])
            trailing = None
            if trade.get("trailingStopLossOrder"):
                trailing = float(trade["trailingStopLossOrder"]["distance"])
            trades.append(
                OpenTrade(
                    trade_id=str(trade["id"]),
                    instrument=trade["instrument"],
                    direction="long" if units > 0 else "short",
                    units=units,
                    open_price=float(trade["price"]),
                    open_time=datetime.fromisoformat(trade["openTime"].replace("Z", "+00:00")),
                    unrealized_pnl=float(trade["unrealizedPL"]),
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    trailing_stop_distance=trailing,
                )
            )
        return trades

    def set_stop_loss(self, trade_id: str, price: float) -> dict:
        return self._trade_orders_put(
            f"/v3/accounts/{self._account_id}/trades/{trade_id}/orders",
            {"stopLoss": {"price": f"{price:.3f}", "timeInForce": "GTC"}},
        )

    def set_take_profit(self, trade_id: str, price: float) -> dict:
        return self._trade_orders_put(
            f"/v3/accounts/{self._account_id}/trades/{trade_id}/orders",
            {"takeProfit": {"price": f"{price:.3f}", "timeInForce": "GTC"}},
        )

    def set_stop_and_tp(self, trade_id: str, stop_loss: float, take_profit: Optional[float] = None) -> dict:
        data: dict[str, dict[str, str]] = {
            "stopLoss": {"price": f"{stop_loss:.3f}", "timeInForce": "GTC"}
        }
        if take_profit is not None:
            data["takeProfit"] = {"price": f"{take_profit:.3f}", "timeInForce": "GTC"}
        return self._trade_orders_put(f"/v3/accounts/{self._account_id}/trades/{trade_id}/orders", data)

    def close_trade(self, trade_id: str, units: Optional[int] = None) -> dict:
        data: dict[str, str] = {}
        if units is not None:
            data["units"] = str(abs(int(units)))
        return self._post(f"/v3/accounts/{self._account_id}/trades/{trade_id}/close", data)

    def get_candles(self, instrument: str, granularity: str = "H4", count: int = 100) -> list[CandleBar]:
        data = self._get(
            f"/v3/instruments/{instrument}/candles",
            params={"granularity": granularity, "count": str(count), "price": "M"},
        )
        candles: list[CandleBar] = []
        for candle in data.get("candles", []):
            mid = candle["mid"]
            candles.append(
                CandleBar(
                    timestamp=datetime.fromisoformat(candle["time"].replace("Z", "+00:00")),
                    open=float(mid["o"]),
                    high=float(mid["h"]),
                    low=float(mid["l"]),
                    close=float(mid["c"]),
                    volume=int(candle["volume"]),
                    complete=bool(candle["complete"]),
                )
            )
        return candles

    def place_market_order(
        self,
        instrument: str,
        units: int,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> dict:
        order: dict[str, object] = {
            "type": "MARKET",
            "instrument": instrument,
            "units": str(units),
            "timeInForce": "FOK",
        }
        if stop_loss is not None:
            order["stopLossOnFill"] = {"price": f"{stop_loss:.3f}", "timeInForce": "GTC"}
        if take_profit is not None:
            order["takeProfitOnFill"] = {"price": f"{take_profit:.3f}", "timeInForce": "GTC"}
        return self._post(f"/v3/accounts/{self._account_id}/orders", {"order": order})

    def test_connection(self) -> bool:
        try:
            self.get_account_summary()
            return True
        except Exception:
            return False
