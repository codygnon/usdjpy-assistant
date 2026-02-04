"""OANDA v20 REST API adapter. Same logical interface as mt5_adapter for run_loop and execution_engine."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal

import pandas as pd
import requests

from core.timeframes import Timeframe

# Re-export types compatible with mt5_adapter
@dataclass(frozen=True)
class Tick:
    time: int
    bid: float
    ask: float

@dataclass(frozen=True)
class OrderResult:
    retcode: int
    comment: str
    request_id: int | None
    order: int | None
    deal: int | None


@dataclass(frozen=True)
class PositionCloseInfo:
    """Close info for a position; same shape as mt5_adapter.PositionCloseInfo for sync_closed_trades."""
    ticket: int
    exit_price: float
    exit_time_utc: str
    profit: float
    volume: float


_BASE_URLS = {
    "practice": "https://api-fxpractice.oanda.com",
    "live": "https://api-fxtrade.oanda.com",
}


def _symbol_to_instrument(symbol: str) -> str:
    """USDJPY or USDJPY.PRO -> USD_JPY."""
    base = re.sub(r"[^A-Za-z]", "", symbol.upper())
    if len(base) == 6:
        return f"{base[:3]}_{base[3:]}"
    return base


def _timeframe_to_granularity(tf: Timeframe) -> str:
    m = {"M1": "M1", "M3": "M3", "M5": "M5", "M15": "M15", "M30": "M30", "H1": "H1", "H4": "H4"}
    if tf in m:
        return m[tf]
    raise ValueError(f"Unsupported timeframe: {tf}")


def _lots_to_units(volume_lots: float) -> int:
    """Standard forex: 1 lot = 100,000 units."""
    return int(round(volume_lots * 100_000))


def _instrument_to_symbol(instrument: str) -> str:
    """USD_JPY or USD/JPY -> USDJPY (OANDA may use either format)."""
    s = (instrument or "").replace("_", "").replace("/", "").upper()
    return s


def _order_price_precision(instrument: str) -> int:
    """Return decimal places allowed for order prices (SL/TP/limit) for this instrument.
    OANDA rejects orders with excess precision; JPY pairs use 3, most others 5."""
    u = (instrument or "").upper().replace("/", "_")
    if "JPY" in u:
        return 3
    return 5


@dataclass(frozen=True)
class OandaClosedPositionInfo:
    """Closed position from OANDA history; same shape as mt5_adapter.ClosedPositionInfo for import."""
    position_id: int
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    entry_time_utc: str
    exit_time_utc: str
    volume: float
    profit: float
    commission: float
    swap: float
    pips: float | None


class OandaAdapter:
    """Adapter for OANDA v3 REST API. Use get_oanda_adapter(profile) to build from profile."""

    def __init__(self, token: str, account_id: str | None, environment: Literal["practice", "live"]) -> None:
        self.token = token.strip()
        self._account_id = account_id
        self._base = _BASE_URLS.get(environment, _BASE_URLS["practice"])
        self._session = requests.Session()
        self._session.headers["Authorization"] = f"Bearer {self.token}"
        self._session.headers["Content-Type"] = "application/json"

    def _get_account_id(self) -> str:
        if self._account_id:
            return self._account_id
        r = self._session.get(f"{self._base}/v3/accounts")
        r.raise_for_status()
        data = r.json()
        accounts = data.get("accounts", [])
        if not accounts:
            raise RuntimeError("OANDA: no accounts found for token")
        self._account_id = accounts[0]["id"]
        return self._account_id

    def _req(self, method: str, path: str, **kwargs) -> dict:
        url = f"{self._base}{path}"
        # Default timeout so we don't hang on stuck connections
        if "timeout" not in kwargs:
            kwargs["timeout"] = 30
        last_err = None
        max_attempts = 4  # 1 initial + 3 retries
        for attempt in range(max_attempts):
            try:
                resp = self._session.request(method, url, **kwargs)
            except requests.RequestException as e:
                last_err = e
                if attempt < max_attempts - 1:
                    delay = min(2 ** (attempt + 1), 60)
                    time.sleep(delay)
                    continue
                raise RuntimeError(f"OANDA API {method} {path}: connection error after {max_attempts} attempts: {e}") from e
            if resp.status_code in (502, 503, 429):
                last_err = RuntimeError(f"OANDA API {method} {path}: {resp.status_code}")
                if attempt < max_attempts - 1:
                    delay = min(2 ** (attempt + 1), 60)
                    time.sleep(delay)
                    continue
                try:
                    err = resp.json()
                    msg = err.get("errorMessage", resp.text)
                except Exception:
                    msg = resp.text
                if msg and ("<" in msg and ">" in msg):
                    msg = f"{resp.status_code} non-JSON response (e.g. Bad Gateway)"
                elif msg and len(msg) > 300:
                    msg = msg[:300] + "..."
                raise RuntimeError(f"OANDA API {method} {path}: {resp.status_code} {msg}")
            if resp.status_code >= 400:
                try:
                    err = resp.json()
                    msg = err.get("errorMessage", resp.text)
                except Exception:
                    msg = resp.text
                if msg and ("<" in msg and ">" in msg):
                    msg = f"{resp.status_code} non-JSON response (e.g. Bad Gateway)"
                elif msg and len(msg) > 300:
                    msg = msg[:300] + "..."
                raise RuntimeError(f"OANDA API {method} {path}: {resp.status_code} {msg}")
            if resp.status_code == 204 or not resp.content:
                return {}
            return resp.json()
        raise RuntimeError(f"OANDA API {method} {path}: unexpected retry exhaustion") from last_err

    def initialize(self) -> None:
        self._get_account_id()

    def shutdown(self) -> None:
        self._session.close()

    def ensure_symbol(self, symbol: str) -> None:
        pass  # OANDA does not require symbol selection

    def get_account_info(self) -> dict:
        aid = self._get_account_id()
        data = self._req("GET", f"/v3/accounts/{aid}/summary")
        acc = data.get("account", {})
        return type("AccountInfo", (), {
            "balance": float(acc.get("balance", 0)),
            "equity": float(acc.get("NAV", acc.get("balance", 0))),
            "margin": float(acc.get("marginUsed", 0)),
            "margin_free": float(acc.get("marginAvailable", 0)),
        })()

    def get_tick(self, symbol: str) -> Tick:
        aid = self._get_account_id()
        inst = _symbol_to_instrument(symbol)
        data = self._req("GET", f"/v3/accounts/{aid}/pricing?instruments={inst}")
        prices = data.get("prices", [])
        if not prices:
            raise RuntimeError(f"OANDA: no price for {symbol} ({inst})")
        p = prices[0]
        # Use closeout for executable bid/ask
        bid = float(p.get("closeoutBid", p.get("bids", [{}])[0].get("price", 0)))
        ask = float(p.get("closeoutAsk", p.get("asks", [{}])[0].get("price", 0)))
        ts = p.get("time", "")
        from datetime import datetime, timezone
        try:
            # RFC3339
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            time_int = int(dt.timestamp())
        except Exception:
            time_int = 0
        return Tick(time=time_int, bid=bid, ask=ask)

    def get_bars(self, symbol: str, tf: Timeframe, count: int) -> pd.DataFrame:
        inst = _symbol_to_instrument(symbol)
        gran = _timeframe_to_granularity(tf)
        data = self._req("GET", f"/v3/instruments/{inst}/candles?granularity={gran}&count={count}&price=M")
        candles = data.get("candles", [])
        if not candles:
            raise RuntimeError(f"OANDA: no candles for {symbol} {tf}")
        rows = []
        for c in candles:
            mid = c.get("mid", {})
            t = c.get("time", "")
            try:
                dt = pd.Timestamp(t).tz_convert("UTC")
            except Exception:
                dt = pd.Timestamp.now(tz="UTC")
            rows.append({
                "time": dt,
                "open": float(mid.get("o", 0)),
                "high": float(mid.get("h", 0)),
                "low": float(mid.get("l", 0)),
                "close": float(mid.get("c", 0)),
                "tick_volume": int(c.get("volume", 0)),
            })
        df = pd.DataFrame(rows)
        if "time" not in df.columns and len(rows):
            df["time"] = [r["time"] for r in rows]
        return df

    def is_demo_account(self) -> bool:
        return self._base == _BASE_URLS["practice"]

    def get_open_positions(self, symbol: str | None = None):
        aid = self._get_account_id()
        data = self._req("GET", f"/v3/accounts/{aid}/openTrades")
        trades = data.get("trades", [])
        if symbol:
            inst = _symbol_to_instrument(symbol)
            trades = [t for t in trades if t.get("instrument") == inst]
        return trades

    def order_send_market(
        self,
        *,
        symbol: str,
        side: Literal["buy", "sell"],
        volume_lots: float,
        sl: float | None,
        tp: float | None,
        deviation_points: int = 20,
        magic: int = 20260128,
        comment: str = "",
    ) -> OrderResult:
        aid = self._get_account_id()
        inst = _symbol_to_instrument(symbol)
        units = _lots_to_units(volume_lots)
        if side == "sell":
            units = -units
        prec = _order_price_precision(inst)
        order: dict = {
            "type": "MARKET",
            "instrument": inst,
            "units": str(units),
            "timeInForce": "FOK",
            "positionFill": "DEFAULT",
        }
        if sl is not None:
            order["stopLossOnFill"] = {"price": str(round(sl, prec)), "timeInForce": "GTC"}
        if tp is not None:
            order["takeProfitOnFill"] = {"price": str(round(tp, prec)), "timeInForce": "GTC"}
        body = {"order": order}
        data = self._req("POST", f"/v3/accounts/{aid}/orders", json=body)
        fill = data.get("orderFillTransaction")
        create = data.get("orderCreateTransaction")
        if fill:
            trade_id = fill.get("tradeOpened", {}).get("tradeID") or fill.get("tradeOpened", {}).get("tradeID")
            if trade_id:
                trade_id = int(trade_id)
            else:
                trade_id = None
            return OrderResult(
                retcode=0,
                comment=fill.get("reason", "FILLED"),
                request_id=None,
                order=int(create["id"]) if create else None,
                deal=trade_id,
            )
        if data.get("orderRejectTransaction"):
            rej = data["orderRejectTransaction"]
            return OrderResult(
                retcode=-1,
                comment=rej.get("rejectReason", rej.get("reason", "REJECTED")),
                request_id=None,
                order=None,
                deal=None,
            )
        return OrderResult(retcode=0, comment="PENDING", request_id=None, order=int(create["id"]) if create else None, deal=None)

    def order_send_pending_limit(
        self,
        *,
        symbol: str,
        side: Literal["buy", "sell"],
        price: float,
        volume_lots: float,
        sl: float | None = None,
        tp: float | None = None,
        magic: int = 20260128,
        comment: str = "",
    ) -> OrderResult:
        aid = self._get_account_id()
        inst = _symbol_to_instrument(symbol)
        units = _lots_to_units(volume_lots)
        if side == "sell":
            units = -units
        prec = _order_price_precision(inst)
        order: dict = {
            "type": "LIMIT",
            "instrument": inst,
            "units": str(units),
            "price": str(round(price, prec)),
            "timeInForce": "GTC",
            "positionFill": "DEFAULT",
        }
        if sl is not None:
            order["stopLossOnFill"] = {"price": str(round(sl, prec)), "timeInForce": "GTC"}
        if tp is not None:
            order["takeProfitOnFill"] = {"price": str(round(tp, prec)), "timeInForce": "GTC"}
        data = self._req("POST", f"/v3/accounts/{aid}/orders", json={"order": order})
        create = data.get("orderCreateTransaction")
        if data.get("orderRejectTransaction"):
            rej = data["orderRejectTransaction"]
            return OrderResult(retcode=-1, comment=rej.get("rejectReason", "REJECTED"), request_id=None, order=None, deal=None)
        return OrderResult(
            retcode=0,
            comment="PENDING",
            request_id=None,
            order=int(create["id"]) if create else None,
            deal=None,
        )

    def update_position_stop_loss(self, trade_id: int, symbol: str, stop_loss_price: float) -> None:
        """Set or replace the stop loss order for an open trade (e.g. for breakeven)."""
        aid = self._get_account_id()
        inst = _symbol_to_instrument(symbol)
        prec = _order_price_precision(inst)
        body = {"stopLoss": {"timeInForce": "GTC", "price": str(round(stop_loss_price, prec))}}
        self._req("PUT", f"/v3/accounts/{aid}/trades/{trade_id}/orders", json=body)

    def close_position(
        self,
        *,
        ticket: int,
        symbol: str,
        volume: float,
        position_type: int,
        deviation_points: int = 20,
        magic: int = 20260128,
        comment: str = "",
    ) -> OrderResult:
        aid = self._get_account_id()
        # OANDA: ticket is trade_id; close by units (all = close full)
        units = str(int(round(volume * 100_000)))
        if position_type == 1:  # SELL position
            units = "-" + units
        data = self._req("PUT", f"/v3/accounts/{aid}/trades/{ticket}/close", json={"units": units})
        close = data.get("orderFillTransaction") or data.get("orderCreateTransaction")
        if close:
            return OrderResult(retcode=0, comment=close.get("reason", "CLOSED"), request_id=None, order=int(close.get("id", 0)) or None, deal=int(close.get("id", 0)) or None)
        return OrderResult(retcode=0, comment="closed", request_id=None, order=None, deal=None)

    def get_position_by_ticket(self, ticket: int):
        aid = self._get_account_id()
        data = self._req("GET", f"/v3/accounts/{aid}/openTrades")
        for t in data.get("trades", []):
            if str(t.get("id")) == str(ticket):
                return t
        return None

    def get_position_id_from_deal(self, deal_id: int) -> int | None:
        # For OANDA we use deal_id as trade_id in OrderResult, so it is the position id
        return deal_id

    def get_position_id_from_order(self, order_id: int) -> int | None:
        aid = self._get_account_id()
        data = self._req("GET", f"/v3/accounts/{aid}/transactions?orderID={order_id}")
        for t in data.get("transactions", []):
            if t.get("type") == "ORDER_FILL" and str(t.get("orderID")) == str(order_id):
                opened = t.get("tradeOpened", {})
                tid = opened.get("tradeID")
                if tid:
                    return int(tid)
        return None

    def get_deals_by_position(self, ticket: int) -> list:
        return []  # OANDA transaction history differs; stub for compatibility

    def get_deal_profit(self, deal_id: int) -> float | None:
        return None

    def get_position_close_info(self, ticket: int) -> PositionCloseInfo | None:
        """Fetch close details for a trade from OANDA transaction history.
        Returns None if no close transaction found (e.g. trade still open or too old).
        """
        aid = self._get_account_id()
        ticket_str = str(ticket)
        # Request last 30 days of transactions
        to_ts = datetime.now(timezone.utc)
        from_ts = to_ts - timedelta(days=30)
        from_param = from_ts.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
        to_param = to_ts.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
        try:
            data = self._req(
                "GET",
                f"/v3/accounts/{aid}/transactions",
                params={"from": from_param, "to": to_param, "pageSize": 500},
            )
        except Exception:
            return None
        pages = data.get("pages") or []
        for page_path in pages:
            if not page_path:
                continue
            if page_path.startswith("http"):
                page_url = page_path
            elif page_path.startswith("/"):
                page_url = self._base + page_path
            else:
                page_url = self._base + "/" + page_path
            try:
                page_resp = self._session.get(page_url)
                if page_resp.status_code != 200:
                    continue
                page_data = page_resp.json()
            except Exception:
                continue
            for t in page_data.get("transactions") or []:
                if t.get("type") != "ORDER_FILL":
                    continue
                # ORDER_FILL that closed a trade has tradeClosed (single) or tradesClosed (array)
                closed = t.get("tradeClosed") or {}
                if closed.get("tradeID") == ticket_str:
                    return _close_info_from_fill(ticket, t, closed)
                for tc in t.get("tradesClosed") or []:
                    if tc.get("tradeID") == ticket_str:
                        return _close_info_from_fill(ticket, t, tc)
        return None

    def get_closed_positions_from_history(self, days_back: int = 30, symbol: str | None = None, pip_size: float | None = None) -> list:
        """Fetch closed positions from OANDA transaction history (activity).
        Pairs ORDER_FILL open/close by tradeID and returns list compatible with import_mt5_history.
        """
        aid = self._get_account_id()
        to_ts = datetime.now(timezone.utc)
        from_ts = to_ts - timedelta(days=min(days_back, 90))
        from_param = from_ts.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
        to_param = to_ts.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
        try:
            data = self._req(
                "GET",
                f"/v3/accounts/{aid}/transactions",
                params={"from": from_param, "to": to_param, "pageSize": 500},
            )
        except Exception as e:
            print(f"[oanda] get_closed_positions_from_history: API request failed: {e}")
            return []
        pages = data.get("pages") or []
        count = data.get("count", 0)
        opens: dict[str, dict] = {}   # tradeID -> { instrument, price, time, units }
        closes: dict[str, dict] = {}  # tradeID -> { price, time, pl, units }

        def process_transactions(transactions: list) -> None:
            for t in transactions or []:
                if t.get("type") != "ORDER_FILL":
                    continue
                # Open: tradeOpened
                opened = t.get("tradeOpened") or {}
                tid = opened.get("tradeID")
                if tid:
                    tid_str = str(tid)
                    units = int(float(t.get("units") or 0))
                    opens[tid_str] = {
                        "instrument": t.get("instrument") or "",
                        "price": float(t.get("price") or 0),
                        "time": str(t.get("time") or ""),
                        "units": units,
                    }
                # Close: tradeClosed or tradesClosed
                closed = t.get("tradeClosed") or {}
                if closed.get("tradeID"):
                    tid_str = str(closed.get("tradeID"))
                    pl = float(t.get("pl") or closed.get("realizedPL") or 0)
                    closes[tid_str] = {
                        "price": float(t.get("price") or 0),
                        "time": str(t.get("time") or ""),
                        "pl": pl,
                        "units": int(float(closed.get("units") or 0)),
                    }
                for tc in t.get("tradesClosed") or []:
                    tid_str = str(tc.get("tradeID") or "")
                    if not tid_str:
                        continue
                    pl = float(t.get("pl") or tc.get("realizedPL") or 0)
                    closes[tid_str] = {
                        "price": float(t.get("price") or 0),
                        "time": str(t.get("time") or ""),
                        "pl": pl,
                        "units": int(float(tc.get("units") or 0)),
                    }

        # Process initial response transactions if present (some APIs return first page in body)
        process_transactions(data.get("transactions") or [])
        for page_path in pages:
            if not page_path:
                continue
            if page_path.startswith("http"):
                page_url = page_path
            elif page_path.startswith("/"):
                page_url = self._base + page_path
            else:
                page_url = self._base + "/" + page_path
            try:
                page_resp = self._session.get(page_url)
                if page_resp.status_code != 200:
                    print(f"[oanda] get_closed_positions_from_history: page returned {page_resp.status_code}")
                    continue
                page_data = page_resp.json()
                process_transactions(page_data.get("transactions") or [])
            except Exception as e:
                print(f"[oanda] get_closed_positions_from_history: page fetch failed: {e}")
                continue
        if not pages and count == 0 and not data.get("transactions"):
            print(f"[oanda] get_closed_positions_from_history: no transactions in range {from_param} to {to_param}")
        results: list[OandaClosedPositionInfo] = []
        # Compare base symbol so "USDJPY.PRO" profile matches OANDA instrument "USD_JPY" -> USDJPY
        want_symbol = (_instrument_to_symbol(symbol) if symbol else "").upper()
        want_base = (want_symbol.split(".")[0] if want_symbol else "")
        for tid_str, close_info in closes.items():
            open_info = opens.get(tid_str)
            if not open_info:
                continue
            inst = open_info.get("instrument") or ""
            sym = _instrument_to_symbol(inst)
            sym_base = (sym.split(".")[0] if sym else "")
            if want_base and sym_base != want_base:
                continue
            entry_price = open_info["price"]
            exit_price = close_info["price"]
            entry_time_utc = open_info["time"]
            exit_time_utc = close_info["time"]
            units = abs(open_info["units"])
            volume = units / 100_000.0
            profit = close_info["pl"]
            side = "buy" if open_info["units"] and int(open_info["units"]) > 0 else "sell"
            pips_val = None
            if pip_size and pip_size > 0:
                if side == "buy":
                    pips_val = (exit_price - entry_price) / pip_size
                else:
                    pips_val = (entry_price - exit_price) / pip_size
            results.append(OandaClosedPositionInfo(
                position_id=int(tid_str),
                symbol=sym,
                side=side,
                entry_price=entry_price,
                exit_price=exit_price,
                entry_time_utc=entry_time_utc,
                exit_time_utc=exit_time_utc,
                volume=volume,
                profit=profit,
                commission=0.0,
                swap=0.0,
                pips=pips_val,
            ))
        # Sort by exit time descending (most recent first)
        results.sort(key=lambda p: p.exit_time_utc or "", reverse=True)
        if count > 0 and len(results) == 0 and (opens or closes):
            print(f"[oanda] get_closed_positions_from_history: {len(opens)} opens, {len(closes)} closes -> 0 matched (check symbol filter or open/close pairing)")
        return results

    def get_mt5_report_stats(self, symbol: str | None = None, pip_size: float | None = None, days_back: int = 90):
        return None

    def get_mt5_full_report(self, symbol: str | None = None, pip_size: float | None = None, days_back: int = 90) -> dict | None:
        return None


def _close_info_from_fill(ticket: int, fill: dict, closed: dict) -> PositionCloseInfo:
    """Build PositionCloseInfo from OANDA ORDER_FILL and tradeClosed/tradesClosed entry."""
    price = float(fill.get("price") or 0)
    time_str = str(fill.get("time") or "")
    pl = float(fill.get("pl") or closed.get("realizedPL") or 0)
    units = abs(int(float(closed.get("units") or 0)))
    volume = units / 100_000.0
    return PositionCloseInfo(
        ticket=ticket,
        exit_price=price,
        exit_time_utc=time_str,
        profit=pl,
        volume=volume,
    )


def get_oanda_adapter(token: str, account_id: str | None, environment: Literal["practice", "live"]) -> OandaAdapter:
    return OandaAdapter(token=token, account_id=account_id, environment=environment)
