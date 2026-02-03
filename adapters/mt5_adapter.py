from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None  # Optional: install with pip install -r requirements-mt5.txt (Windows only)

import pandas as pd

from core.timeframes import Timeframe


def _ensure_mt5() -> None:
    """Raise a clear error if MetaTrader5 is not installed (e.g. on Linux/macOS or PaaS)."""
    if mt5 is None:
        raise RuntimeError(
            "MetaTrader5 is not installed. For MT5 trading on Windows, install it with: pip install -r requirements-mt5.txt"
        )


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


def initialize() -> None:
    _ensure_mt5()
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")


def shutdown() -> None:
    _ensure_mt5()
    mt5.shutdown()


def ensure_symbol(symbol: str) -> None:
    _ensure_mt5()
    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"symbol_select failed for {symbol}: {mt5.last_error()}")


def get_account_info():
    _ensure_mt5()
    return mt5.account_info()


def get_tick(symbol: str) -> Tick:
    _ensure_mt5()
    t = mt5.symbol_info_tick(symbol)
    if t is None:
        raise RuntimeError(f"symbol_info_tick returned None for {symbol}: {mt5.last_error()}")
    return Tick(time=int(t.time), bid=float(t.bid), ask=float(t.ask))


def timeframe_to_mt5(tf: Timeframe):
    _ensure_mt5()
    if tf == "M1":
        return mt5.TIMEFRAME_M1
    if tf == "M3":
        return mt5.TIMEFRAME_M3
    if tf == "M5":
        return mt5.TIMEFRAME_M5
    if tf == "M15":
        return mt5.TIMEFRAME_M15
    if tf == "M30":
        return mt5.TIMEFRAME_M30
    if tf == "H1":
        return mt5.TIMEFRAME_H1
    if tf == "H4":
        return mt5.TIMEFRAME_H4
    raise ValueError(f"Unsupported timeframe: {tf}")


def get_bars(symbol: str, tf: Timeframe, count: int) -> pd.DataFrame:
    _ensure_mt5()
    rates = mt5.copy_rates_from_pos(symbol, timeframe_to_mt5(tf), 0, count)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"copy_rates_from_pos returned no data for {symbol} {tf}: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df


def is_demo_account() -> bool:
    _ensure_mt5()
    acct = mt5.account_info()
    if acct is None:
        raise RuntimeError("account_info is None (not logged in?)")
    # MT5 trade_mode: 0 demo, 1 contest, 2 real (common convention)
    return int(getattr(acct, "trade_mode", -1)) == 0


def get_open_positions(symbol: str | None = None):
    _ensure_mt5()
    if symbol:
        return mt5.positions_get(symbol=symbol)
    return mt5.positions_get()


def _allowed_filling(symbol: str) -> int:
    """Return an ORDER_FILLING_* constant supported for this symbol.

    MT5 symbol_info(symbol).filling_mode is a bitmask:
    1 = FOK, 2 = IOC, 4 = RETURN.
    ORDER_FILLING_* constants are 0, 1, 2, so we map the first allowed bit.
    Fallback to IOC if info is unavailable.
    """
    _ensure_mt5()
    info = mt5.symbol_info(symbol)
    mode = int(getattr(info, "filling_mode", 0) or 0) if info is not None else 0
    # Prefer FOK, then IOC, then RETURN
    if mode & 1:
        return mt5.ORDER_FILLING_FOK
    if mode & 2:
        return mt5.ORDER_FILLING_IOC
    if mode & 4:
        return mt5.ORDER_FILLING_RETURN
    # Sensible default on unknown/zero: IOC tends to be widely supported
    return mt5.ORDER_FILLING_IOC


def order_send_market(
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
    _ensure_mt5()
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"symbol_info_tick returned None for {symbol}: {mt5.last_error()}")

    order_type = mt5.ORDER_TYPE_BUY if side == "buy" else mt5.ORDER_TYPE_SELL
    price = float(tick.ask) if side == "buy" else float(tick.bid)

    filling = _allowed_filling(symbol)
    
    # Sanitize comment - MT5 has very strict requirements for comment field
    # Use only simple alphanumeric characters, max 20 chars to be safe
    if comment:
        # Keep only letters and numbers, replace everything else
        safe_comment = "".join(c if c.isalnum() else "" for c in comment)[:20]
    else:
        safe_comment = "auto"

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(volume_lots),
        "type": order_type,
        "price": price,
        "deviation": int(deviation_points),
        "magic": int(magic),
        "comment": safe_comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling,
    }
    if sl is not None:
        request["sl"] = float(sl)
    if tp is not None:
        request["tp"] = float(tp)

    res = mt5.order_send(request)
    if res is None:
        raise RuntimeError(f"order_send returned None: {mt5.last_error()}")

    return OrderResult(
        retcode=int(res.retcode),
        comment=str(getattr(res, "comment", "")),
        request_id=int(getattr(res, "request_id", 0)) if hasattr(res, "request_id") else None,
        order=int(getattr(res, "order", 0)) if hasattr(res, "order") else None,
        deal=int(getattr(res, "deal", 0)) if hasattr(res, "deal") else None,
    )


def order_send_pending_limit(
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
    """Place a pending limit order at `price`. MT5 fills when price reaches the level.

    - Buy limit: price < bid (we want price to fall to X).
    - Sell limit: price > ask (we want price to rise to X).
    """
    _ensure_mt5()
    order_type = mt5.ORDER_TYPE_BUY_LIMIT if side == "buy" else mt5.ORDER_TYPE_SELL_LIMIT
    filling = _allowed_filling(symbol)
    
    # Sanitize comment - MT5 has very strict requirements
    if comment:
        safe_comment = "".join(c if c.isalnum() else "" for c in comment)[:20]
    else:
        safe_comment = "auto"
    
    request = {
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": symbol,
        "volume": float(volume_lots),
        "type": order_type,
        "price": float(price),
        "magic": int(magic),
        "comment": safe_comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling,
    }
    if sl is not None:
        request["sl"] = float(sl)
    if tp is not None:
        request["tp"] = float(tp)

    res = mt5.order_send(request)
    if res is None:
        raise RuntimeError(f"order_send (pending limit) returned None: {mt5.last_error()}")

    return OrderResult(
        retcode=int(res.retcode),
        comment=str(getattr(res, "comment", "")),
        request_id=int(getattr(res, "request_id", 0)) if hasattr(res, "request_id") else None,
        order=int(getattr(res, "order", 0)) if hasattr(res, "order") else None,
        deal=int(getattr(res, "deal", 0)) if hasattr(res, "deal") else None,
    )


def close_position(
    *,
    ticket: int,
    symbol: str,
    volume: float,
    position_type: int,
    deviation_points: int = 20,
    magic: int = 20260128,
    comment: str = "",
) -> OrderResult:
    """Close an open position by sending an opposite market order.
    
    Args:
        ticket: Position ticket number
        symbol: Trading symbol
        volume: Volume to close
        position_type: MT5 position type (0 = BUY, 1 = SELL)
        deviation_points: Max price deviation in points
        magic: Magic number
        comment: Order comment
    """
    _ensure_mt5()
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"symbol_info_tick returned None for {symbol}: {mt5.last_error()}")

    # Close by sending opposite order
    # If position is BUY (type 0), close with SELL at bid
    # If position is SELL (type 1), close with BUY at ask
    if position_type == 0:  # BUY position -> close with SELL
        order_type = mt5.ORDER_TYPE_SELL
        price = float(tick.bid)
    else:  # SELL position -> close with BUY
        order_type = mt5.ORDER_TYPE_BUY
        price = float(tick.ask)

    filling = _allowed_filling(symbol)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(volume),
        "type": order_type,
        "position": int(ticket),
        "price": price,
        "deviation": int(deviation_points),
        "magic": int(magic),
        "comment": comment or f"close_{ticket}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling,
    }

    res = mt5.order_send(request)
    if res is None:
        raise RuntimeError(f"order_send (close position) returned None: {mt5.last_error()}")

    return OrderResult(
        retcode=int(res.retcode),
        comment=str(getattr(res, "comment", "")),
        request_id=int(getattr(res, "request_id", 0)) if hasattr(res, "request_id") else None,
        order=int(getattr(res, "order", 0)) if hasattr(res, "order") else None,
        deal=int(getattr(res, "deal", 0)) if hasattr(res, "deal") else None,
    )


def get_position_by_ticket(ticket: int):
    """Get a specific position by ticket number (position_id). Returns None if not found."""
    _ensure_mt5()
    positions = mt5.positions_get(ticket=ticket)
    if positions is None or len(positions) == 0:
        return None
    return positions[0]


def get_position_id_from_deal(deal_id: int) -> int | None:
    """Look up a deal by its ticket (deal_id) and return the position_id.
    
    After order_send returns a deal_id, use this to get the position_id
    that we need for later sync operations.
    """
    _ensure_mt5()
    from datetime import datetime, timedelta, timezone
    
    # Search recent history for this deal
    date_from = datetime.now(timezone.utc) - timedelta(days=7)
    date_to = datetime.now(timezone.utc) + timedelta(days=1)
    
    # history_deals_get(ticket=...) gets deals by deal ticket (not order ticket)
    deals = mt5.history_deals_get(date_from, date_to)
    if deals is None:
        return None
    
    # Find the deal with matching ticket (deal_id)
    for deal in deals:
        if getattr(deal, "ticket", None) == deal_id:
            position_id = getattr(deal, "position_id", None)
            if position_id:
                return int(position_id)
    
    return None


def get_position_id_from_order(order_id: int) -> int | None:
    """Look up deals by order ticket and return the position_id.
    
    Alternative to get_position_id_from_deal when we have order_id instead of deal_id.
    """
    _ensure_mt5()
    from datetime import datetime, timedelta, timezone
    
    date_from = datetime.now(timezone.utc) - timedelta(days=7)
    date_to = datetime.now(timezone.utc) + timedelta(days=1)
    
    # Get deals associated with this order
    deals = mt5.history_deals_get(date_from, date_to, ticket=order_id)
    if deals is None or len(deals) == 0:
        # Try searching all deals for matching order
        all_deals = mt5.history_deals_get(date_from, date_to)
        if all_deals:
            for deal in all_deals:
                if getattr(deal, "order", None) == order_id:
                    position_id = getattr(deal, "position_id", None)
                    if position_id:
                        return int(position_id)
        return None
    
    # Return position_id from first matching deal
    for deal in deals:
        position_id = getattr(deal, "position_id", None)
        if position_id:
            return int(position_id)
    
    return None


@dataclass(frozen=True)
class PositionCloseInfo:
    """Information about a closed position."""
    ticket: int
    exit_price: float
    exit_time_utc: str
    profit: float
    volume: float


def get_deals_by_position(ticket: int) -> list:
    """Get all deals for a position ticket from MT5 history.
    
    Returns list of deal objects, or empty list if none found.
    """
    _ensure_mt5()
    from datetime import datetime, timedelta, timezone
    
    # Search deals from 90 days ago to now (extended for older closed positions)
    date_from = datetime.now(timezone.utc) - timedelta(days=90)
    date_to = datetime.now(timezone.utc) + timedelta(days=1)
    
    deals = mt5.history_deals_get(date_from, date_to, position=ticket)
    if deals is None:
        return []
    return list(deals)


def get_deal_profit(deal_id: int) -> float | None:
    """Fetch a deal by its ticket (deal_id) and return its profit.
    
    Returns None if deal not found.
    """
    _ensure_mt5()
    from datetime import datetime, timedelta, timezone
    
    date_from = datetime.now(timezone.utc) - timedelta(days=1)
    date_to = datetime.now(timezone.utc) + timedelta(days=1)
    
    deals = mt5.history_deals_get(date_from, date_to)
    if deals is None:
        return None
    for deal in deals:
        if getattr(deal, "ticket", None) == deal_id:
            return float(getattr(deal, "profit", 0))
    return None


def get_position_total_profit(ticket: int) -> float | None:
    """Sum profit from all closing deals for a position (handles partial closes).
    
    Returns total profit across all exit deals, or None if position not found.
    """
    deals = get_deals_by_position(ticket)
    if not deals:
        return None
    total = 0.0
    for deal in deals:
        entry = getattr(deal, "entry", None)
        if entry in (1, 3):  # DEAL_ENTRY_OUT or OUT_BY
            total += float(getattr(deal, "profit", 0))
    return total


def get_position_close_info(ticket: int) -> PositionCloseInfo | None:
    """Return exit details if position was closed, else None.
    
    Uses LAST closing deal for exit_price/time; sums profit from ALL exit deals
    (handles partial closes correctly).
    """
    from datetime import datetime, timezone
    
    deals = get_deals_by_position(ticket)
    if not deals:
        return None
    
    # Collect all closing deals (entry 1 or 3)
    closing_deals = []
    for deal in deals:
        entry = getattr(deal, "entry", None)
        if entry in (1, 3):  # OUT or OUT_BY
            closing_deals.append(deal)
    
    if not closing_deals:
        return None
    
    # Use last close for exit price/time (final close of position)
    last_deal = closing_deals[-1]
    exit_time = getattr(last_deal, "time", 0)
    exit_time_utc = datetime.fromtimestamp(exit_time, tz=timezone.utc).isoformat(timespec="seconds")
    
    # Sum profit from ALL exit deals (correct for partial closes)
    total_profit = sum(float(getattr(d, "profit", 0)) for d in closing_deals)
    
    return PositionCloseInfo(
        ticket=ticket,
        exit_price=float(getattr(last_deal, "price", 0)),
        exit_time_utc=exit_time_utc,
        profit=total_profit,
        volume=float(getattr(last_deal, "volume", 0)),
    )


@dataclass(frozen=True)
class ClosedPositionInfo:
    """Full information about a closed position built from deal history."""
    position_id: int
    symbol: str
    side: str  # "buy" or "sell"
    entry_price: float
    exit_price: float
    entry_time_utc: str
    exit_time_utc: str
    volume: float
    profit: float
    commission: float
    swap: float
    pips: float | None  # Calculated if pip_size provided


def get_closed_positions_from_history(
    days_back: int = 30,
    symbol: str | None = None,
    pip_size: float | None = None,
) -> list[ClosedPositionInfo]:
    """Fetch closed positions from MT5 deal history.
    
    Groups deals by position_id, pairs entry/exit deals, and returns
    structured info for each closed position.
    
    Args:
        days_back: How many days of history to fetch
        symbol: Optional symbol filter (e.g. "USDJPY.PRO")
        pip_size: If provided, calculates pips for each position
    
    Returns:
        List of ClosedPositionInfo for each closed position found.
    """
    _ensure_mt5()
    from datetime import datetime, timedelta, timezone
    from collections import defaultdict
    
    date_from = datetime.now(timezone.utc) - timedelta(days=days_back)
    date_to = datetime.now(timezone.utc) + timedelta(days=1)
    
    if symbol:
        deals = mt5.history_deals_get(date_from, date_to, group=f"*{symbol}*")
    else:
        deals = mt5.history_deals_get(date_from, date_to)
    
    if deals is None or len(deals) == 0:
        return []
    
    # Group deals by position_id
    deals_by_position: dict[int, list] = defaultdict(list)
    for deal in deals:
        pos_id = getattr(deal, "position_id", 0)
        if pos_id > 0:  # Skip balance operations (position_id = 0)
            deals_by_position[pos_id].append(deal)
    
    results: list[ClosedPositionInfo] = []
    
    for pos_id, pos_deals in deals_by_position.items():
        entry_deal = None
        exit_deals = []
        
        for deal in pos_deals:
            entry_type = getattr(deal, "entry", -1)
            if entry_type == 0:  # DEAL_ENTRY_IN
                entry_deal = deal
            elif entry_type in (1, 3):  # DEAL_ENTRY_OUT or OUT_BY
                exit_deals.append(deal)
        
        # Only include fully closed positions (have both entry and exit)
        if entry_deal is None or not exit_deals:
            continue
        
        # Use last exit deal for price/time; sum profit from all exit deals (partial closes)
        last_exit = exit_deals[-1]
        total_profit = sum(float(getattr(d, "profit", 0)) for d in exit_deals)
        total_commission = sum(float(getattr(d, "commission", 0)) for d in pos_deals)
        total_swap = sum(float(getattr(d, "swap", 0)) for d in pos_deals)
        
        deal_symbol = getattr(entry_deal, "symbol", "")
        deal_type = getattr(entry_deal, "type", 0)  # 0 = BUY, 1 = SELL
        side = "buy" if deal_type == 0 else "sell"
        
        entry_price = float(getattr(entry_deal, "price", 0))
        exit_price = float(getattr(last_exit, "price", 0))
        volume = float(getattr(entry_deal, "volume", 0))
        profit = total_profit
        
        entry_time = getattr(entry_deal, "time", 0)
        exit_time = getattr(last_exit, "time", 0)
        entry_time_utc = datetime.fromtimestamp(entry_time, tz=timezone.utc).isoformat(timespec="seconds")
        exit_time_utc = datetime.fromtimestamp(exit_time, tz=timezone.utc).isoformat(timespec="seconds")
        
        # Calculate pips if pip_size provided
        pips = None
        if pip_size and pip_size > 0:
            if side == "buy":
                pips = (exit_price - entry_price) / pip_size
            else:
                pips = (entry_price - exit_price) / pip_size
        
        results.append(ClosedPositionInfo(
            position_id=pos_id,
            symbol=deal_symbol,
            side=side,
            entry_price=entry_price,
            exit_price=exit_price,
            entry_time_utc=entry_time_utc,
            exit_time_utc=exit_time_utc,
            volume=volume,
            profit=profit,
            commission=total_commission,
            swap=total_swap,
            pips=pips,
        ))
    
    return results


@dataclass(frozen=True)
class Mt5ReportStats:
    """Stats derived directly from MT5 deal history (same source as View -> Reports)."""
    closed_trades: int
    wins: int
    losses: int
    win_rate: float
    total_profit: float
    total_commission: float
    total_swap: float
    gross_profit: float
    gross_loss: float
    profit_factor: float
    largest_profit_trade: float
    largest_loss_trade: float
    expected_payoff: float
    avg_pips: float | None
    total_pips: float


def get_position_financials(position_id: int) -> dict | None:
    """Return {profit, commission, swap} summed from all deals for a position."""
    deals = get_deals_by_position(position_id)
    if not deals:
        return None
    profit = 0.0
    commission = 0.0
    swap = 0.0
    for deal in deals:
        entry_type = getattr(deal, "entry", -1)
        if entry_type in (1, 3):  # OUT or OUT_BY - profit on close
            profit += float(getattr(deal, "profit", 0))
        commission += float(getattr(deal, "commission", 0))
        swap += float(getattr(deal, "swap", 0))
    return {"profit": profit, "commission": commission, "swap": swap}


def get_mt5_report_stats(
    symbol: str | None = None,
    pip_size: float | None = None,
    days_back: int = 90,
) -> Mt5ReportStats | None:
    """Fetch stats directly from MT5 deal history (equivalent to View -> Reports).

    Uses the same underlying data as MT5's Summary / Profit & Loss report.
    Returns None if MT5 is not available or fails.
    """
    if mt5 is None:
        return None
    try:
        mt5.initialize()
    except Exception:
        return None

    try:
        closed = get_closed_positions_from_history(
            days_back=days_back,
            symbol=symbol,
            pip_size=pip_size,
        )
    except Exception:
        return None
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass

    if not closed:
        return Mt5ReportStats(
            closed_trades=0,
            wins=0,
            losses=0,
            win_rate=0.0,
            total_profit=0.0,
            total_commission=0.0,
            total_swap=0.0,
            gross_profit=0.0,
            gross_loss=0.0,
            profit_factor=1.0,
            largest_profit_trade=0.0,
            largest_loss_trade=0.0,
            expected_payoff=0.0,
            avg_pips=None,
            total_pips=0.0,
        )

    total_profit = sum(p.profit for p in closed)
    total_commission = sum(p.commission for p in closed)
    total_swap = sum(p.swap for p in closed)
    wins = sum(1 for p in closed if p.profit > 0)
    losses = sum(1 for p in closed if p.profit < 0)
    total = len(closed)
    win_rate = wins / total if total > 0 else 0.0

    gross_profit = sum(p.profit for p in closed if p.profit > 0)
    gross_loss = sum(p.profit for p in closed if p.profit < 0)
    profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else (1.0 if gross_profit > 0 else 1.0)
    largest_profit = max((p.profit for p in closed if p.profit > 0), default=0.0)
    largest_loss = min((p.profit for p in closed if p.profit < 0), default=0.0)
    expected_payoff = total_profit / total if total > 0 else 0.0

    pips_list = [p.pips for p in closed if p.pips is not None]
    total_pips_val = sum(pips_list)
    avg_pips = total_pips_val / len(pips_list) if pips_list else None

    return Mt5ReportStats(
        closed_trades=total,
        wins=wins,
        losses=losses,
        win_rate=round(win_rate, 3),
        total_profit=round(total_profit, 2),
        total_commission=round(total_commission, 2),
        total_swap=round(total_swap, 2),
        gross_profit=round(gross_profit, 2),
        gross_loss=round(gross_loss, 2),
        profit_factor=round(profit_factor, 3),
        largest_profit_trade=round(largest_profit, 2),
        largest_loss_trade=round(largest_loss, 2),
        expected_payoff=round(expected_payoff, 2),
        avg_pips=round(avg_pips, 3) if avg_pips is not None else None,
        total_pips=round(total_pips_val, 2),
    )


def get_mt5_full_report(
    symbol: str | None = None,
    pip_size: float | None = None,
    days_back: int = 90,
) -> dict | None:
    """Fetch full MT5 report (Summary + Closed P/L + Long/Short).
    
    Returns dict with summary, closed_pl, long_short; or None if MT5 unavailable.
    """
    if mt5 is None:
        return None
    try:
        mt5.initialize()
    except Exception:
        return None

    try:
        acct = mt5.account_info()
        closed = get_closed_positions_from_history(
            days_back=days_back,
            symbol=symbol,
            pip_size=pip_size,
        )
    except Exception:
        return None
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass

    summary = {}
    if acct is not None:
        summary = {
            "balance": round(float(getattr(acct, "balance", 0)), 2),
            "equity": round(float(getattr(acct, "equity", 0)), 2),
            "margin": round(float(getattr(acct, "margin", 0)), 2),
            "free_margin": round(float(getattr(acct, "margin_free", 0)), 2),
        }

    total_profit = sum(p.profit for p in closed)
    total_commission = sum(p.commission for p in closed)
    total_swap = sum(p.swap for p in closed)
    wins = sum(1 for p in closed if p.profit > 0)
    losses = sum(1 for p in closed if p.profit < 0)
    total = len(closed)
    win_rate = round(wins / total, 3) if total > 0 else 0.0
    gross_profit = sum(p.profit for p in closed if p.profit > 0)
    gross_loss = sum(p.profit for p in closed if p.profit < 0)
    profit_factor = round(gross_profit / abs(gross_loss), 3) if gross_loss != 0 else 1.0
    largest_profit = max((p.profit for p in closed if p.profit > 0), default=0.0)
    largest_loss = min((p.profit for p in closed if p.profit < 0), default=0.0)
    expected_payoff = round(total_profit / total, 2) if total > 0 else 0.0
    pips_list = [p.pips for p in closed if p.pips is not None]
    total_pips_val = sum(pips_list)
    avg_pips = round(total_pips_val / len(pips_list), 3) if pips_list else None

    closed_pl = {
        "closed_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "total_profit": round(total_profit, 2),
        "total_commission": round(total_commission, 2),
        "total_swap": round(total_swap, 2),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "profit_factor": profit_factor,
        "largest_profit_trade": round(largest_profit, 2),
        "largest_loss_trade": round(largest_loss, 2),
        "expected_payoff": expected_payoff,
        "avg_pips": avg_pips,
        "total_pips": round(total_pips_val, 2),
    }

    long_trades = [p for p in closed if p.side == "buy"]
    short_trades = [p for p in closed if p.side == "sell"]
    long_wins = sum(1 for p in long_trades if p.profit > 0)
    short_wins = sum(1 for p in short_trades if p.profit > 0)
    long_short = {
        "long_trades": len(long_trades),
        "long_wins": long_wins,
        "long_win_pct": round(long_wins / len(long_trades), 3) if long_trades else 0,
        "short_trades": len(short_trades),
        "short_wins": short_wins,
        "short_win_pct": round(short_wins / len(short_trades), 3) if short_trades else 0,
    }

    return {
        "source": "mt5",
        "summary": summary,
        "closed_pl": closed_pl,
        "long_short": long_short,
    }

