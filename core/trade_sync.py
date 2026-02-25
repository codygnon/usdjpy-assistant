"""Trade synchronization with MT5.

Detects trades that were closed externally (via MT5 app) and updates the local database.
Also supports importing closed positions from MT5 history.
"""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from core.profile import ProfileV1
    from storage.sqlite_store import SqliteStore


def _safe_get(row: Any, key: str, default: Any = None) -> Any:
    """Safely get a value from a sqlite3.Row or dict-like object."""
    try:
        val = row[key]
        return val if val is not None else default
    except (KeyError, IndexError, TypeError):
        return default


def _determine_exit_reason(
    exit_price: float,
    target_price: float | None,
    stop_price: float | None,
    side: str,
    pip_size: float,
    breakeven_sl_price: float | None = None,
    breakeven_applied: int = 0,
) -> str:
    """Determine if trade was closed at TP, SL, BE, or manually by user.

    Tolerance: within 1.5 pips of target/stop counts as hitting it.
    BE tolerance is 3 pips due to spread-aware SL that varies slightly.
    """
    tolerance = pip_size * 1.5  # 1.5 pips tolerance

    if target_price and not pd.isna(target_price):
        if abs(exit_price - target_price) <= tolerance:
            return "hit_take_profit"

    # Check breakeven BEFORE original SL — BE SL is far from original stop_price
    if breakeven_applied and breakeven_sl_price and not pd.isna(breakeven_sl_price):
        if abs(exit_price - breakeven_sl_price) <= pip_size * 3:
            return "hit_breakeven"

    if stop_price and not pd.isna(stop_price):
        if abs(exit_price - stop_price) <= tolerance:
            return "hit_stop_loss"

    # If we have a target and exit is before it (for profit), user closed early
    if target_price and not pd.isna(target_price):
        if side == "buy" and exit_price < target_price:
            return "user_closed_early"
        if side == "sell" and exit_price > target_price:
            return "user_closed_early"

    return "user_closed_early"


def compute_post_sl_recovery_pips(
    adapter: Any,
    symbol: str,
    side: str,
    exit_price: float,
    exit_time_utc: str,
    pip_size: float,
    window_minutes: int = 30,
) -> float | None:
    """Fetch M1 candles after a SL/BE close and return max favorable move within window.

    BUY: how far price rose above exit_price → positive pips
    SELL: how far price fell below exit_price → positive pips
    Returns None on failure (no candles, adapter error, etc.).
    """
    try:
        df = adapter.get_bars_from_time(symbol, "M1", exit_time_utc, count=window_minutes)
        if df is None or df.empty:
            return None

        # Filter to only candles at or after exit time
        exit_ts = pd.to_datetime(exit_time_utc, utc=True)
        if "time" in df.columns:
            df = df[df["time"] >= exit_ts]

        if df.empty:
            return None

        if side == "buy":
            best_high = df["high"].max()
            recovery = (best_high - exit_price) / pip_size
        else:
            best_low = df["low"].min()
            recovery = (exit_price - best_low) / pip_size

        return round(max(0.0, float(recovery)), 2)
    except Exception as e:
        print(f"[trade_sync] compute_post_sl_recovery_pips failed: {e}")
        return None


def sync_closed_trades(profile: "ProfileV1", store: "SqliteStore", log_dir=None) -> int:
    """Check open trades in DB against broker (MT5/OANDA); update any that were closed externally.
    
    Uses mt5_position_id (preferred) or falls back to mt5_order_id for lookups.
    Returns count of trades synced.
    """
    from adapters.broker import get_adapter
    
    # Get open trades from our database
    open_trades = store.list_open_trades(profile.profile_name)
    
    if not open_trades:
        return 0
    
    try:
        adapter = get_adapter(profile)
        adapter.initialize()
        adapter.ensure_symbol(profile.symbol)
    except Exception as e:
        print(f"[trade_sync] Broker init failed: {e}")
        return 0
    
    synced_count = 0
    oanda_missing_close_info_warned = False
    pip_size = profile.pip_size
    is_oanda = getattr(profile, "broker_type", None) == "oanda"
    oanda_closed_map: dict[int, Any] | None = None
    
    for trade_row in open_trades:
        trade_id = trade_row["trade_id"]
        
        # Get position_id - prefer mt5_position_id, fallback to looking up from order/deal
        mt5_position_id = _safe_get(trade_row, "mt5_position_id")
        mt5_order_id = _safe_get(trade_row, "mt5_order_id")
        mt5_deal_id = _safe_get(trade_row, "mt5_deal_id")
        
        position_ticket = None
        
        if mt5_position_id and not pd.isna(mt5_position_id):
            position_ticket = int(mt5_position_id)
        elif mt5_deal_id and not pd.isna(mt5_deal_id):
            # Try to get position_id from deal
            position_ticket = adapter.get_position_id_from_deal(int(mt5_deal_id))
        elif mt5_order_id and not pd.isna(mt5_order_id):
            # Try to get position_id from order
            position_ticket = adapter.get_position_id_from_order(int(mt5_order_id))
        
        if position_ticket is None:
            # No valid identifier - can't sync
            continue
        
        # If we found a position_id but didn't have it stored, update it now
        if (not mt5_position_id or pd.isna(mt5_position_id)) and position_ticket:
            store.update_trade(trade_id, {"mt5_position_id": position_ticket})
        
        # Check if position is still open
        position = adapter.get_position_by_ticket(position_ticket)
        
        if position is not None:
            # Position is still open - nothing to sync
            continue
        
        # Position is not open - check if it was closed
        close_info = adapter.get_position_close_info(position_ticket)

        if close_info is None:
            # OANDA fallback: query recent closed positions once and match by trade/position id.
            if is_oanda:
                if oanda_closed_map is None:
                    try:
                        closed_positions = adapter.get_closed_positions_from_history(
                            days_back=30, symbol=profile.symbol, pip_size=float(profile.pip_size)
                        )
                        oanda_closed_map = {
                            int(getattr(pos, "position_id")): pos
                            for pos in (closed_positions or [])
                            if getattr(pos, "position_id", None) is not None
                        }
                    except Exception:
                        oanda_closed_map = {}
                matched = (oanda_closed_map or {}).get(int(position_ticket))
                if matched is not None:
                    close_info = SimpleNamespace(
                        exit_price=float(getattr(matched, "exit_price", 0.0)),
                        exit_time_utc=str(getattr(matched, "exit_time_utc", "")),
                        profit=float(getattr(matched, "profit", 0.0)),
                        volume=float(getattr(matched, "volume", 0.0)),
                    )

        if close_info is None:
            # No close info found - might be a different issue
            if is_oanda:
                if not oanda_missing_close_info_warned:
                    print("[trade_sync] OANDA close info temporarily unavailable (history endpoint may be delayed/transient); will retry next sync cycle.")
                    oanda_missing_close_info_warned = True
            else:
                print(f"[trade_sync] No close info for position {position_ticket}, trade_id={trade_id}")
            continue
        
        # Position was closed - update our database
        entry_price = trade_row["entry_price"]
        side = str(trade_row["side"]).lower()
        stop_price = _safe_get(trade_row, "stop_price")
        target_price = _safe_get(trade_row, "target_price")
        entry_ts = trade_row["timestamp_utc"]
        
        exit_price = close_info.exit_price
        exit_ts = close_info.exit_time_utc
        
        # Calculate pips
        if side == "buy":
            pips = (exit_price - entry_price) / pip_size
        else:
            pips = (entry_price - exit_price) / pip_size
        
        # Calculate R-multiple
        risk_pips = None
        r_multiple = None
        if stop_price and not pd.isna(stop_price):
            risk_pips = abs(entry_price - stop_price) / pip_size
            if risk_pips > 0:
                r_multiple = pips / risk_pips
        
        # Calculate duration
        duration_minutes = None
        if entry_ts:
            try:
                t0 = pd.to_datetime(entry_ts, utc=True)
                t1 = pd.to_datetime(exit_ts, utc=True)
                duration_minutes = float((t1 - t0).total_seconds() / 60.0)
            except Exception:
                pass
        
        # Determine exit reason
        be_sl_price = _safe_get(trade_row, "breakeven_sl_price")
        be_applied = int(_safe_get(trade_row, "breakeven_applied") or 0)
        exit_reason = _determine_exit_reason(
            exit_price=exit_price,
            target_price=target_price,
            stop_price=stop_price,
            side=side,
            pip_size=pip_size,
            breakeven_sl_price=float(be_sl_price) if be_sl_price is not None else None,
            breakeven_applied=be_applied,
        )
        
        # Update database
        store.close_trade(
            trade_id=trade_id,
            updates={
                "exit_price": float(exit_price),
                "exit_timestamp_utc": exit_ts,
                "exit_reason": exit_reason,
                "pips": float(pips),
                "risk_pips": float(risk_pips) if risk_pips else None,
                "r_multiple": float(r_multiple) if r_multiple else None,
                "duration_minutes": float(duration_minutes) if duration_minutes else None,
                "profit": float(close_info.profit),
            },
        )
        
        print(f"[trade_sync] Synced closed trade {trade_id}: exit={exit_price}, pips={pips:.2f}, reason={exit_reason}")

        # Append trade close event for dashboard
        if log_dir is not None:
            try:
                from core.dashboard_models import TradeEvent, append_trade_event
                from pathlib import Path
                event = TradeEvent(
                    event_type="close",
                    timestamp_utc=exit_ts or pd.Timestamp.now(tz="UTC").isoformat(),
                    trade_id=str(trade_id),
                    side=side,
                    entry_type=str(_safe_get(trade_row, "entry_type") or ""),
                    price=float(exit_price),
                    pips=float(pips),
                    profit=float(close_info.profit) if close_info.profit is not None else None,
                    exit_reason=exit_reason,
                )
                append_trade_event(Path(log_dir), event)
            except Exception as e:
                print(f"[trade_sync] Failed to append trade close event: {e}")

        synced_count += 1
    
    try:
        adapter.shutdown()
    except Exception:
        pass
    return synced_count


def import_mt5_history(
    profile: "ProfileV1",
    store: "SqliteStore",
    days_back: int = 30,
) -> int:
    """Import closed positions from broker history that aren't already in our database.
    
    This imports manually-opened trades (trades opened via broker app, not by our program).
    OANDA adapter returns empty list (stub); MT5 provides full history.
    
    Returns count of trades imported.
    """
    from adapters.broker import get_adapter
    
    try:
        adapter = get_adapter(profile)
        adapter.initialize()
        adapter.ensure_symbol(profile.symbol)
    except Exception as e:
        print(f"[trade_sync] Broker init failed: {e}")
        return 0
    
    pip_size = profile.pip_size
    
    try:
        closed_positions = adapter.get_closed_positions_from_history(
            days_back=days_back,
            symbol=profile.symbol,
            pip_size=pip_size,
        )
    finally:
        try:
            adapter.shutdown()
        except Exception:
            pass
    
    imported_count = 0
    
    is_oanda = getattr(profile, "broker_type", None) == "oanda"
    prefix = "oanda_" if is_oanda else "mt5_"
    notes_tag = "imported_from_oanda_history" if is_oanda else "imported_from_mt5_history"
    exit_reason_tag = "oanda_history_import" if is_oanda else "mt5_history_import"

    for pos in closed_positions:
        # Check if we already have this position
        if store.trade_exists_by_position_id(profile.profile_name, pos.position_id):
            continue
        
        # Calculate duration
        duration_minutes = None
        try:
            t0 = pd.to_datetime(pos.entry_time_utc, utc=True)
            t1 = pd.to_datetime(pos.exit_time_utc, utc=True)
            duration_minutes = float((t1 - t0).total_seconds() / 60.0)
        except Exception:
            pass
        
        # Create trade record (prefix by broker so OANDA and MT5 IDs don't collide)
        trade_id = f"{prefix}{pos.position_id}"
        
        store.insert_trade({
            "trade_id": trade_id,
            "timestamp_utc": pos.entry_time_utc,
            "profile": profile.profile_name,
            "symbol": pos.symbol,
            "side": pos.side,
            "config_json": None,
            "entry_price": pos.entry_price,
            "stop_price": None,  # Unknown for imported trades
            "target_price": None,  # Unknown for imported trades
            "size_lots": pos.volume,
            "notes": notes_tag,
            "snapshot_id": None,
            "exit_price": pos.exit_price,
            "exit_timestamp_utc": pos.exit_time_utc,
            "exit_reason": exit_reason_tag,
            "pips": pos.pips,
            "risk_pips": None,
            "r_multiple": None,
            "duration_minutes": duration_minutes,
            "mt5_order_id": None,
            "preset_name": profile.active_preset_name or "Unknown (imported)",
            "mt5_deal_id": None,
            "mt5_retcode": None,
            "mt5_position_id": pos.position_id,
            "opened_by": "manual",
            "profit": float(pos.profit),
        })
        
        pips_disp = f"{pos.pips:.2f}" if pos.pips is not None else "n/a"
        print(f"[trade_sync] Imported broker history: {trade_id}, {pos.side} {pos.symbol}, pips={pips_disp}")
        imported_count += 1
    if getattr(profile, "broker_type", None) == "oanda" and imported_count > 0:
        print(f"[trade_sync] OANDA: imported {imported_count} new closed trade(s) from history")
    
    return imported_count


def backfill_position_ids(profile: "ProfileV1", store: "SqliteStore") -> int:
    """Backfill mt5_position_id for existing trades that have deal_id or order_id but no position_id.
    
    Tries get_position_id_from_deal first, then get_position_id_from_order. Returns count updated.
    """
    from adapters.broker import get_adapter
    
    trades = store.get_trades_missing_position_id(profile.profile_name)
    
    if not trades:
        return 0
    
    try:
        adapter = get_adapter(profile)
        adapter.initialize()
    except Exception as e:
        print(f"[trade_sync] Broker init failed: {e}")
        return 0
    
    updated_count = 0
    try:
        for trade_row in trades:
            trade_id = trade_row["trade_id"]
            mt5_deal_id = _safe_get(trade_row, "mt5_deal_id")
            mt5_order_id = _safe_get(trade_row, "mt5_order_id")
            
            position_id = None
            
            if mt5_deal_id and not pd.isna(mt5_deal_id):
                position_id = adapter.get_position_id_from_deal(int(mt5_deal_id))
            
            if position_id is None and mt5_order_id and not pd.isna(mt5_order_id):
                position_id = adapter.get_position_id_from_order(int(mt5_order_id))
            
            if position_id:
                store.update_trade(trade_id, {"mt5_position_id": position_id})
                print(f"[trade_sync] Backfilled position_id={position_id} for trade {trade_id}")
                updated_count += 1
    finally:
        try:
            adapter.shutdown()
        except Exception:
            pass
    return updated_count


def backfill_profit(profile: "ProfileV1", store: "SqliteStore", force_refresh: bool = False) -> int:
    """Backfill profit for closed trades that have mt5_position_id.
    
    Fetches profit from broker (summed across partial closes). Returns count updated.
    If force_refresh=True, overwrites existing profit to fix incorrect data.
    OANDA adapter returns None for get_position_close_info (stub).
    """
    from adapters.broker import get_adapter
    
    trades = store.get_trades_missing_profit(profile.profile_name, force_refresh=force_refresh)
    
    if not trades:
        return 0
    
    try:
        adapter = get_adapter(profile)
        adapter.initialize()
    except Exception as e:
        print(f"[trade_sync] Broker init failed for profit backfill: {e}")
        return 0
    
    updated_count = 0
    try:
        oanda_profit_by_position: dict[int, float] | None = None
        is_oanda = getattr(profile, "broker_type", None) == "oanda"
        if is_oanda:
            try:
                closed_positions = adapter.get_closed_positions_from_history(
                    days_back=365,
                    symbol=profile.symbol,
                    pip_size=float(profile.pip_size),
                )
                oanda_profit_by_position = {}
                for pos in closed_positions or []:
                    pid = getattr(pos, "position_id", None)
                    if pid is None:
                        continue
                    try:
                        oanda_profit_by_position[int(pid)] = float(getattr(pos, "profit", 0.0) or 0.0)
                    except Exception:
                        continue
            except Exception as e:
                print(f"[trade_sync] OANDA history profit map failed: {e}")
                oanda_profit_by_position = {}

        for trade_row in trades:
            trade_id = trade_row["trade_id"]
            mt5_position_id = _safe_get(trade_row, "mt5_position_id")
            
            if not mt5_position_id or pd.isna(mt5_position_id):
                continue

            position_id = int(mt5_position_id)
            profit_value: float | None = None
            if oanda_profit_by_position is not None:
                profit_value = oanda_profit_by_position.get(position_id)

            if profit_value is None:
                close_info = adapter.get_position_close_info(position_id)
                if close_info:
                    profit_value = float(close_info.profit)

            if profit_value is None:
                continue

            store.update_trade(trade_id, {"profit": float(profit_value)})
            print(f"[trade_sync] Backfilled profit={profit_value:.2f} for trade {trade_id}")
            updated_count += 1
    finally:
        try:
            adapter.shutdown()
        except Exception:
            pass
    return updated_count
