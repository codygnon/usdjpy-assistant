from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is on path (fixes ModuleNotFoundError when started as subprocess)
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from adapters.broker import get_adapter
from core.context_engine import compute_tf_context
from core.execution_engine import (
    build_default_candidate_from_signal,
    execute_bollinger_policy_demo_only,
    execute_breakout_policy_demo_only,
    execute_ema_pullback_policy_demo_only,
    execute_indicator_policy_demo_only,
    execute_kt_cg_ctp_policy_demo_only,
    execute_kt_cg_hybrid_policy_demo_only,
    execute_kt_cg_trial_4_policy_demo_only,
    execute_kt_cg_trial_5_policy_demo_only,
    execute_price_level_policy_demo_only,
    execute_session_momentum_policy_demo_only,
    execute_signal_demo_only,
    execute_vwap_policy_demo_only,
)
from core.execution_state import RuntimeState, load_state, save_state
from core.models import MarketContext
from core.profile import get_effective_risk, load_profile_v1
from core.risk_engine import evaluate_trade
from core.signal_engine import (
    compute_alignment_score,
    compute_latest_diffs,
    detect_latest_confirmed_cross_signal,
    evaluate_filters,
)
from core.trade_sync import sync_closed_trades, import_mt5_history
from storage.sqlite_store import SqliteStore


def _poll_seconds(profile, cli_poll: float | None) -> float:
    cfg = profile.execution
    base = float(cfg.loop_poll_seconds)
    if cli_poll is not None:
        base = float(cli_poll)
    fast = float(cfg.loop_poll_seconds_fast)
    use_fast = any(
        getattr(p, "type", None) == "price_level_trend"
        and getattr(p, "enabled", True)
        and not getattr(p, "use_pending_order", True)
        for p in cfg.policies
    )
    if use_fast and base > fast:
        return fast
    return max(0.25, base)


def _compute_mkt(profile, tick, data_by_tf) -> MarketContext:
    spread_pips = (tick.ask - tick.bid) / float(profile.pip_size)
    diffs = compute_latest_diffs(profile, data_by_tf)
    alignment_score = int(compute_alignment_score(profile, diffs)) if diffs else 0
    return MarketContext(spread_pips=float(spread_pips), alignment_score=alignment_score)


def _insert_trade_for_policy(
    *,
    profile,
    adapter,
    store,
    policy_type: str,
    policy_id: str,
    side: str,
    entry_price: float,
    dec,
    stop_price: float | None = None,
    target_price: float | None = None,
    size_lots: float | None = None,
    entry_type: str | None = None,
) -> None:
    """Store a trade in DB when a policy places an order. Ensures preset and position_id for Performance by Preset."""
    try:
        position_id = None
        if dec.deal_id:
            position_id = adapter.get_position_id_from_deal(dec.deal_id)
        if position_id is None and dec.order_id:
            position_id = adapter.get_position_id_from_order(dec.order_id)
        trade_id = f"{policy_type}:{policy_id}:{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d%H%M%S')}"
        row = {
            "trade_id": trade_id,
            "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "profile": profile.profile_name,
            "symbol": profile.symbol,
            "side": side,
            "config_json": json.dumps(profile.model_dump()),
            "entry_price": float(entry_price),
            "stop_price": stop_price,
            "target_price": target_price,
            "size_lots": float(size_lots or get_effective_risk(profile).max_lots),
            "notes": f"auto:{policy_type}:{policy_id}",
            "snapshot_id": None,
            "mt5_order_id": dec.order_id,
            "mt5_deal_id": dec.deal_id,
            "mt5_retcode": dec.order_retcode,
            "mt5_position_id": position_id,
            "opened_by": "program",
            "preset_name": profile.active_preset_name or "Unknown",
        }
        row["breakeven_applied"] = 0
        row["tp1_partial_done"] = 0
        if entry_type:
            row["entry_type"] = entry_type
        # Capture entry slippage if fill_price available
        if getattr(dec, 'fill_price', None) is not None:
            slippage = abs(dec.fill_price - entry_price) / float(profile.pip_size)
            row["entry_slippage_pips"] = round(slippage, 3)
        store.insert_trade(row)
    except Exception:
        pass


def _run_trade_management(profile, adapter, store, tick) -> None:
    """Apply breakeven and TP1 partial close for open positions we manage. Order: breakeven first, then TP1."""
    tm = getattr(profile, "trade_management", None)
    if tm is None:
        return
    breakeven = getattr(tm, "breakeven", None)
    target = getattr(tm, "target", None)

    # Find Trial #4 spread-aware BE config if present
    t4_spread_be = None
    for pol in profile.execution.policies:
        if getattr(pol, "type", None) in ("kt_cg_trial_4", "kt_cg_trial_5") and getattr(pol, "enabled", True):
            if getattr(pol, "spread_aware_be_enabled", False):
                t4_spread_be = pol
            break

    has_simple_be = breakeven and getattr(breakeven, "enabled", False)
    has_scaled = target and getattr(target, "mode", None) == "scaled" and getattr(target, "tp1_pips", None)
    if not has_simple_be and not has_scaled and t4_spread_be is None:
        return
    try:
        open_positions = adapter.get_open_positions(profile.symbol)
    except Exception:
        return
    if not open_positions:
        return
    # sqlite3.Row doesn't support .get(); convert to dicts for safe access
    our_trades = [dict(r) for r in store.list_open_trades(profile.profile_name)]
    position_to_trade = {
        row["mt5_position_id"]: row
        for row in our_trades
        if row.get("mt5_position_id") is not None
    }
    pip = float(profile.pip_size)
    mid = (tick.bid + tick.ask) / 2.0
    current_spread = tick.ask - tick.bid
    for pos in open_positions:
        # OANDA: dict with "id", "currentUnits"; MT5: object with ticket, volume (lots)
        position_id = pos.get("id") if isinstance(pos, dict) else getattr(pos, "ticket", None)
        if position_id is None:
            continue
        try:
            position_id = int(position_id)
        except (TypeError, ValueError):
            continue
        trade_row = position_to_trade.get(position_id)
        if trade_row is None:
            continue
        trade_id = str(trade_row["trade_id"])
        entry = float(trade_row["entry_price"])
        side = str(trade_row["side"]).lower()
        entry_type = trade_row.get("entry_type")

        # 1a) Spread-Aware Breakeven for Trial #4 trades
        if t4_spread_be is not None and entry_type is not None:
            # Check scope: apply_to_zone_entry / apply_to_tiered_pullback
            apply = False
            if entry_type == "zone_entry" and getattr(t4_spread_be, "spread_aware_be_apply_to_zone_entry", True):
                apply = True
            elif entry_type == "tiered_pullback" and getattr(t4_spread_be, "spread_aware_be_apply_to_tiered_pullback", True):
                apply = True
            if apply:
                # Trial #5: hardcode spread+buffer; Trial #4 uses policy setting
                if getattr(t4_spread_be, "type", None) == "kt_cg_trial_5":
                    trigger_mode = "spread_relative"
                else:
                    trigger_mode = getattr(t4_spread_be, "spread_aware_be_trigger_mode", "fixed_pips")
                if trigger_mode == "spread_relative":
                    trigger_pips = (current_spread / pip) + getattr(t4_spread_be, "spread_aware_be_spread_buffer_pips", 1.0)
                else:
                    trigger_pips = getattr(t4_spread_be, "spread_aware_be_fixed_trigger_pips", 5.0)
                # Check if TP distance is large enough (skip if TP < trigger threshold)
                tp_price = trade_row.get("target_price")
                if tp_price is not None:
                    tp_dist_pips = abs(float(tp_price) - entry) / pip
                    if tp_dist_pips < trigger_pips:
                        pass  # Skip BE — TP too close
                    else:
                        # Use bid for BUY, ask for SELL to check profit
                        check_price = tick.bid if side == "buy" else tick.ask
                        profit_pips = ((check_price - entry) / pip) if side == "buy" else ((entry - check_price) / pip)
                        if profit_pips >= trigger_pips:
                            # Compute new SL = entry ± current_spread
                            if side == "buy":
                                new_sl = entry + current_spread
                            else:
                                new_sl = entry - current_spread
                            # Ratchet: only move SL favorably
                            prev_be_sl = trade_row.get("breakeven_sl_price")
                            if prev_be_sl is not None:
                                prev_be_sl = float(prev_be_sl)
                                if side == "buy":
                                    new_sl = max(new_sl, prev_be_sl)
                                else:
                                    new_sl = min(new_sl, prev_be_sl)
                            # Only update if SL changed
                            if prev_be_sl is None or abs(new_sl - prev_be_sl) > pip * 0.01:
                                try:
                                    adapter.update_position_stop_loss(position_id, profile.symbol, round(new_sl, 3))
                                    store.update_trade(trade_id, {"breakeven_applied": 1, "breakeven_sl_price": round(new_sl, 5)})
                                    spread_pips = current_spread / pip
                                    print(f"[{profile.profile_name}] spread-aware BE: pos {position_id} SL -> {new_sl:.3f} (spread={spread_pips:.1f}p, profit={profit_pips:.1f}p)")
                                except Exception as e:
                                    print(f"[{profile.profile_name}] spread-aware BE error pos {position_id}: {e}")
                # Skip simple BE for this trade (spread-aware handles it)
                # Fall through to TP1 and MAE/MFE below
            else:
                # Not in scope — fall through to simple BE
                if has_simple_be:
                    be_applied = trade_row.get("breakeven_applied") or 0
                    if not be_applied:
                        after_pips = float(getattr(breakeven, "after_pips", 0) or 0)
                        if after_pips > 0:
                            in_favor_buy = mid >= entry + after_pips * pip
                            in_favor_sell = mid <= entry - after_pips * pip
                            if (side == "buy" and in_favor_buy) or (side == "sell" and in_favor_sell):
                                adapter.update_position_stop_loss(position_id, profile.symbol, entry)
                                store.update_trade(trade_id, {"breakeven_applied": 1})
                                print(f"[{profile.profile_name}] breakeven applied position {position_id}")
        # 1b) Simple breakeven for non-Trial-4 or when spread-aware is disabled
        elif has_simple_be:
            be_applied = trade_row.get("breakeven_applied") or 0
            if not be_applied:
                after_pips = float(getattr(breakeven, "after_pips", 0) or 0)
                if after_pips > 0:
                    in_favor_buy = mid >= entry + after_pips * pip
                    in_favor_sell = mid <= entry - after_pips * pip
                    if (side == "buy" and in_favor_buy) or (side == "sell" and in_favor_sell):
                        adapter.update_position_stop_loss(position_id, profile.symbol, entry)
                        store.update_trade(trade_id, {"breakeven_applied": 1})
                        print(f"[{profile.profile_name}] breakeven applied position {position_id}")
        # 2) TP1 partial close
        if target and getattr(target, "mode", None) == "scaled":
            tp1_pips = getattr(target, "tp1_pips", None)
            tp1_pct = getattr(target, "tp1_close_percent", None)
            if tp1_pips is not None and tp1_pct is not None:
                tp1_done = trade_row.get("tp1_partial_done") or 0
                if not tp1_done:
                    reached_buy = mid >= entry + float(tp1_pips) * pip
                    reached_sell = mid <= entry - float(tp1_pips) * pip
                    if (side == "buy" and reached_buy) or (side == "sell" and reached_sell):
                        if isinstance(pos, dict):
                            current_units = pos.get("currentUnits") or 0
                            current_lots = abs(int(current_units)) / 100_000.0
                        else:
                            current_lots = float(getattr(pos, "volume", 0) or 0)
                        close_lots = current_lots * (float(tp1_pct) / 100.0)
                        position_type = 1 if side == "sell" else 0
                        adapter.close_position(
                            ticket=position_id,
                            symbol=profile.symbol,
                            volume=close_lots,
                            position_type=position_type,
                        )
                        store.update_trade(trade_id, {"tp1_partial_done": 1})
                        print(f"[{profile.profile_name}] TP1 partial close position {position_id} ({tp1_pct}%)")
        # 3) Update MAE/MFE watermarks
        current_pips = ((mid - entry) / pip) if side == "buy" else ((entry - mid) / pip)
        prev_mae = trade_row.get("max_adverse_pips")
        prev_mfe = trade_row.get("max_favorable_pips")
        mae = min(current_pips, prev_mae if prev_mae is not None else current_pips)
        mfe = max(current_pips, prev_mfe if prev_mfe is not None else current_pips)
        mae_mfe_updates: dict = {}
        if mae != prev_mae:
            mae_mfe_updates["max_adverse_pips"] = round(mae, 2)
        if mfe != prev_mfe:
            mae_mfe_updates["max_favorable_pips"] = round(mfe, 2)
        if mae_mfe_updates:
            store.update_trade(trade_id, mae_mfe_updates)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run v1 loop (M1 cadence).")
    ap.add_argument("--profile", required=True, help="Path to profile JSON (legacy or v1)")
    ap.add_argument("--poll-seconds", type=float, default=None, help="Override profile loop_poll_seconds")
    ap.add_argument("--once", action="store_true", help="Run a single iteration then exit")
    args = ap.parse_args()

    # Use same persistent data dir as API when set (Railway volume)
    _data_base_env = os.environ.get("RAILWAY_VOLUME_MOUNT_PATH") or os.environ.get("USDJPY_DATA_DIR")
    _base_dir = Path(_data_base_env) if _data_base_env else Path(__file__).resolve().parent
    profile = load_profile_v1(args.profile)

    log_dir = _base_dir / "logs" / profile.profile_name
    log_dir.mkdir(parents=True, exist_ok=True)
    state_path = log_dir / "runtime_state.json"

    store = SqliteStore(log_dir / "assistant.db")
    store.init_db()

    adapter = get_adapter(profile)
    adapter.initialize()
    try:
        adapter.ensure_symbol(profile.symbol)

        last_seen_m1_time: str | None = None
        poll_sec = _poll_seconds(profile, args.poll_seconds)
        has_price_level = any(
            getattr(p, "type", None) == "price_level_trend" and getattr(p, "enabled", True)
            for p in profile.execution.policies
        )
        has_indicator = any(
            getattr(p, "type", None) == "indicator_based" and getattr(p, "enabled", True)
            for p in profile.execution.policies
        )
        has_breakout = any(
            getattr(p, "type", None) == "breakout_range" and getattr(p, "enabled", True)
            for p in profile.execution.policies
        )
        has_session = any(
            getattr(p, "type", None) == "session_momentum" and getattr(p, "enabled", True)
            for p in profile.execution.policies
        )
        has_bollinger = any(
            getattr(p, "type", None) == "bollinger_bands" and getattr(p, "enabled", True)
            for p in profile.execution.policies
        )
        has_vwap = any(
            getattr(p, "type", None) == "vwap" and getattr(p, "enabled", True)
            for p in profile.execution.policies
        )
        has_ema_pullback = any(
            getattr(p, "type", None) == "ema_pullback" and getattr(p, "enabled", True)
            for p in profile.execution.policies
        )
        has_kt_cg_ctp = any(
            getattr(p, "type", None) == "kt_cg_counter_trend_pullback" and getattr(p, "enabled", True)
            for p in profile.execution.policies
        )
        has_kt_cg_hybrid = any(
            getattr(p, "type", None) == "kt_cg_hybrid" and getattr(p, "enabled", True)
            for p in profile.execution.policies
        )
        has_kt_cg_trial_4 = any(
            getattr(p, "type", None) == "kt_cg_trial_4" and getattr(p, "enabled", True)
            for p in profile.execution.policies
        )
        has_kt_cg_trial_5 = any(
            getattr(p, "type", None) == "kt_cg_trial_5" and getattr(p, "enabled", True)
            for p in profile.execution.policies
        )
        # Confirmed-cross setups can now use M5; detect if any do so we fetch M5 bars.
        m5_cross_setup_ids = {
            sid
            for sid, setup in (profile.strategy.setups or {}).items()
            if getattr(setup, "enabled", True) and getattr(setup, "timeframe", "M1") == "M5"
        }
        has_m5_confirmed_cross = any(
            getattr(p, "type", None) == "confirmed_cross"
            and getattr(p, "enabled", True)
            and getattr(p, "setup_id", "m1_cross_entry") in m5_cross_setup_ids
            for p in profile.execution.policies
        )

        loop_count = 0
        last_sync_loop = 0
        SYNC_INTERVAL_LOOPS = 12  # Sync every ~60 seconds (assuming 5s poll)
        _MAX_FETCH_RETRIES = 3  # Retry broker fetch this many times before sleeping and continuing

        # Candle cache: {tf: (timestamp_fetched, DataFrame)}
        # TTLs in seconds: M1=55, M3=175, M5=295, M15=895, H4=14395, D=300
        _candle_cache: dict[str, tuple[float, pd.DataFrame]] = {}
        _CANDLE_TTL: dict[str, float] = {
            "M1": 55.0, "M3": 175.0, "M5": 295.0, "M15": 895.0,
            "H4": 14395.0, "D": 300.0,
        }

        # Logging state for Trial #5 — reduce log spam
        _last_exhaustion_zone: str | None = None
        _last_summary_time: float = 0.0
        _SUMMARY_INTERVAL: float = 30.0  # Print periodic summary every 30s

        def _get_bars_cached(symbol: str, tf: str, count: int) -> pd.DataFrame:
            """Fetch bars with caching to reduce API calls at fast poll rates."""
            now = time.time()
            cached = _candle_cache.get(tf)
            ttl = _CANDLE_TTL.get(tf, 55.0)
            if cached is not None:
                cached_time, cached_df = cached
                if now - cached_time < ttl:
                    return cached_df
            df = adapter.get_bars(symbol, tf, count)
            _candle_cache[tf] = (now, df)
            return df

        while True:
            loop_count += 1
            if loop_count % 20 == 0:
                print(f"[{profile.profile_name}] heartbeat loop={loop_count}")
            
            # Periodic trade sync (detect externally closed trades; import from broker history)
            if loop_count - last_sync_loop >= SYNC_INTERVAL_LOOPS:
                try:
                    synced = sync_closed_trades(profile, store)
                    if synced > 0:
                        print(f"[{profile.profile_name}] synced {synced} externally closed trade(s)")
                    # Import closed trades from broker history (OANDA activity / MT5 history) every sync
                    imported = import_mt5_history(profile, store, days_back=90)
                    if imported > 0:
                        print(f"[{profile.profile_name}] imported {imported} trade(s) from broker history")
                except Exception as e:
                    print(f"[{profile.profile_name}] sync error: {e}")
                last_sync_loop = loop_count

            state = load_state(state_path)
            if state.kill_switch:
                mode = "DISARMED"
            else:
                mode = state.mode

            # Fetch market data with retries (502/503/timeouts); avoid exiting on transient broker errors
            data_by_tf = None
            tick = None
            for _fetch_attempt in range(_MAX_FETCH_RETRIES):
                try:
                    data_by_tf = {
                        "H4": _get_bars_cached(profile.symbol, "H4", 800),
                        "M15": _get_bars_cached(profile.symbol, "M15", 2000),
                        "M1": _get_bars_cached(profile.symbol, "M1", 3000),
                    }
                    # Fetch M5 data when needed by ema_pullback, kt_cg_ctp, kt_cg_hybrid, or M5 confirmed-cross setups.
                    if has_ema_pullback or has_m5_confirmed_cross or has_kt_cg_ctp or has_kt_cg_hybrid:
                        data_by_tf["M5"] = _get_bars_cached(profile.symbol, "M5", 2000)
                    # Fetch M3 data when needed by kt_cg_trial_4/5 (trend detection).
                    if has_kt_cg_trial_4 or has_kt_cg_trial_5:
                        data_by_tf["M3"] = _get_bars_cached(profile.symbol, "M3", 3000)
                    # Fetch D1 data when needed by daily H/L filter (Trial #4/5).
                    if has_kt_cg_trial_4 or has_kt_cg_trial_5:
                        data_by_tf["D"] = _get_bars_cached(profile.symbol, "D", 2)
                    # Tick always fetched fresh for live price detection
                    tick = adapter.get_tick(profile.symbol)
                    break
                except Exception as _fetch_err:
                    if _fetch_attempt < _MAX_FETCH_RETRIES - 1:
                        _delay = min(10 * (2 **_fetch_attempt), 60)
                        print(f"[{profile.profile_name}] broker fetch error (retry in {_delay}s): {_fetch_err}")
                        time.sleep(_delay)
                    else:
                        print(f"[{profile.profile_name}] broker temporarily unavailable after {_MAX_FETCH_RETRIES} attempts, sleeping 60s then continuing: {_fetch_err}")
                        time.sleep(60)
            if data_by_tf is None or tick is None:
                continue

            # Trade management: breakeven and TP1 partial close (only for positions we opened)
            _run_trade_management(profile, adapter, store, tick)

            m1_df = data_by_tf["M1"]
            m1_last_time = pd.to_datetime(m1_df["time"].iloc[-1], utc=True).isoformat()
            if last_seen_m1_time is None:
                last_seen_m1_time = m1_last_time

            is_new = m1_last_time != last_seen_m1_time

            mkt = None
            if is_new or args.once:
                spread_pips = (tick.ask - tick.bid) / float(profile.pip_size)
                h4c = compute_tf_context(profile, "H4", data_by_tf["H4"])
                m15c = compute_tf_context(profile, "M15", data_by_tf["M15"])
                m1c = compute_tf_context(profile, "M1", data_by_tf["M1"])
                diffs = compute_latest_diffs(profile, data_by_tf)
                alignment_score = int(compute_alignment_score(profile, diffs)) if diffs else 0
                mkt = MarketContext(spread_pips=float(spread_pips), alignment_score=alignment_score)

                snap_row = {
                    "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                    "profile": profile.profile_name,
                    "symbol": profile.symbol,
                    "config_json": json.dumps(profile.model_dump()),
                    "spread_pips": float(spread_pips),
                    "alignment_score": alignment_score,
                    "h4_regime": h4c.regime,
                    "h4_cross_dir": h4c.last_cross_dir,
                    "h4_cross_time": h4c.last_cross_time.isoformat() if h4c.last_cross_time is not None else None,
                    "h4_cross_price": h4c.last_cross_price,
                    "h4_trend_since": h4c.trend_since_cross,
                    "m15_regime": m15c.regime,
                    "m15_cross_dir": m15c.last_cross_dir,
                    "m15_cross_time": m15c.last_cross_time.isoformat() if m15c.last_cross_time is not None else None,
                    "m15_cross_price": m15c.last_cross_price,
                    "m15_trend_since": m15c.trend_since_cross,
                    "m1_regime": m1c.regime,
                    "m1_cross_dir": m1c.last_cross_dir,
                    "m1_cross_time": m1c.last_cross_time.isoformat() if m1c.last_cross_time is not None else None,
                    "m1_cross_price": m1c.last_cross_price,
                    "m1_trend_since": m1c.trend_since_cross,
                }
                # Capture M1 ATR-14 at snapshot time
                try:
                    from core.indicators import atr as atr_fn
                    if "M1" in data_by_tf and not data_by_tf["M1"].empty:
                        atr_series = atr_fn(data_by_tf["M1"], period=14)
                        if not atr_series.empty and pd.notna(atr_series.iloc[-1]):
                            snap_row["atr_m1_14"] = float(atr_series.iloc[-1])
                except Exception:
                    pass
                snapshot_id = store.insert_snapshot(snap_row)

                trades_df = store.read_trades_df(profile.profile_name)

                for pol in profile.execution.policies:
                    if not getattr(pol, "enabled", True):
                        continue
                    if getattr(pol, "type", None) != "confirmed_cross":
                        continue
                    setup_id = getattr(pol, "setup_id", "m1_cross_entry")
                    setup = profile.strategy.setups.get(setup_id)
                    if setup is None or not setup.enabled:
                        continue
                    setup_name = setup_id
                    tf = setup.timeframe
                    sig = detect_latest_confirmed_cross_signal(
                        profile=profile,
                        df=data_by_tf[tf],
                        tf=tf,
                        ema_period=setup.ema,
                        sma_period=setup.sma,
                        confirm_bars=setup.confirmation.confirm_bars,
                        require_close_on_correct_side=setup.confirmation.require_close_on_correct_side,
                        min_distance_pips=setup.confirmation.min_distance_pips,
                        max_wait_bars=setup.confirmation.max_wait_bars,
                    )
                    if sig is None:
                        continue

                    ok, reasons = evaluate_filters(profile, data_by_tf, sig)
                    if not ok:
                        filter_reason = "filter_reject: " + "; ".join(reasons)
                        print(f"[{profile.profile_name}] REJECTED: confirmed_cross:{setup_name} | side={sig.side} | {filter_reason}")
                        store.insert_execution(
                            {
                                "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                                "profile": profile.profile_name,
                                "symbol": profile.symbol,
                                "signal_id": sig.signal_id,
                                "mode": mode,
                                "attempted": 1,
                                "placed": 0,
                                "reason": filter_reason,
                                "mt5_retcode": None,
                                "mt5_order_id": None,
                                "mt5_deal_id": None,
                            }
                        )
                        continue

                    if mode == "ARMED_MANUAL_CONFIRM":
                        cand = build_default_candidate_from_signal(profile, sig)
                        risk_dec = evaluate_trade(profile=profile, candidate=cand, context=mkt, trades_df=trades_df)
                        if not risk_dec.allow:
                            risk_reason = "risk_reject: " + "; ".join(risk_dec.hard_reasons)
                            print(f"[{profile.profile_name}] REJECTED: confirmed_cross:{setup_name} | side={sig.side} | {risk_reason}")
                            store.insert_execution(
                                {
                                    "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                                    "profile": profile.profile_name,
                                    "symbol": profile.symbol,
                                    "signal_id": sig.signal_id,
                                    "mode": mode,
                                    "attempted": 1,
                                    "placed": 0,
                                    "reason": risk_reason,
                                    "mt5_retcode": None,
                                    "mt5_order_id": None,
                                    "mt5_deal_id": None,
                                }
                            )
                        else:
                            store.upsert_pending_signal(
                                {
                                    "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                                    "profile": profile.profile_name,
                                    "symbol": profile.symbol,
                                    "signal_id": sig.signal_id,
                                    "timeframe": sig.timeframe,
                                    "side": sig.side,
                                    "cross_time": sig.cross_time.isoformat(),
                                    "confirm_time": sig.confirm_time.isoformat(),
                                    "entry_price_hint": float(sig.entry_price_hint),
                                    "reasons_json": json.dumps(sig.reasons + reasons),
                                }
                            )
                            store.insert_execution(
                                {
                                    "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                                    "profile": profile.profile_name,
                                    "symbol": profile.symbol,
                                    "signal_id": sig.signal_id,
                                    "mode": mode,
                                    "attempted": 1,
                                    "placed": 0,
                                    "reason": "manual_confirm_required",
                                    "mt5_retcode": None,
                                    "mt5_order_id": None,
                                    "mt5_deal_id": None,
                                }
                            )
                            print(f"[{profile.profile_name}] pending {setup_name} {sig.side} confirm={sig.confirm_time}")
                        continue

                    exec_dec = execute_signal_demo_only(
                        adapter=adapter,
                        profile=profile,
                        log_dir=log_dir,
                        signal=sig,
                        context=mkt,
                        trades_df=trades_df,
                        mode=mode,
                    )
                    
                    # Log the execution result
                    if exec_dec.placed:
                        print(f"[{profile.profile_name}] TRADE PLACED: confirmed_cross:{setup_name} | side={sig.side} | entry={sig.entry_price_hint:.3f} | {exec_dec.reason}")
                    elif exec_dec.attempted:
                        print(f"[{profile.profile_name}] REJECTED: confirmed_cross:{setup_name} | side={sig.side} | {exec_dec.reason}")

                    if exec_dec.placed:
                        try:
                            cand = build_default_candidate_from_signal(profile, sig)
                            # Get position_id from deal for reliable sync later
                            position_id = None
                            if exec_dec.deal_id:
                                position_id = adapter.get_position_id_from_deal(exec_dec.deal_id)
                            if position_id is None and exec_dec.order_id:
                                position_id = adapter.get_position_id_from_order(exec_dec.order_id)
                            store.insert_trade(
                                {
                                    "trade_id": sig.signal_id,
                                    "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                                    "profile": profile.profile_name,
                                    "symbol": profile.symbol,
                                    "side": sig.side,
                                    "config_json": json.dumps(profile.model_dump()),
                                    "entry_price": float(sig.entry_price_hint),
                                    "stop_price": cand.stop_price,
                                    "target_price": cand.target_price,
                                    "size_lots": cand.size_lots,
                                    "notes": f"auto:{setup_name}",
                                    "snapshot_id": int(snapshot_id),
                                    "mt5_order_id": exec_dec.order_id,
                                    "mt5_deal_id": exec_dec.deal_id,
                                    "mt5_retcode": exec_dec.order_retcode,
                                    "mt5_position_id": position_id,
                                    "opened_by": "program",
                                    "preset_name": profile.active_preset_name or "Unknown",
                                    "breakeven_applied": 0,
                                    "tp1_partial_done": 0,
                                    **({"entry_slippage_pips": round(abs(exec_dec.fill_price - float(sig.entry_price_hint)) / float(profile.pip_size), 3)} if getattr(exec_dec, 'fill_price', None) is not None else {}),
                                }
                            )
                        except Exception:
                            pass

                    state = load_state(state_path)
                    save_state(
                        state_path,
                        RuntimeState(
                            mode=state.mode,
                            kill_switch=state.kill_switch,
                            last_processed_bar_time_utc=m1_last_time,
                        ),
                    )
                    if exec_dec.attempted:
                        print(f"[{profile.profile_name}] {setup_name} {sig.side} {sig.confirm_time} mode={mode} -> {exec_dec.reason}")

                last_seen_m1_time = m1_last_time

            if has_price_level and mkt is None:
                mkt = _compute_mkt(profile, tick, data_by_tf)
            if has_price_level and mkt is not None:
                trades_df = store.read_trades_df(profile.profile_name)
                for pol in profile.execution.policies:
                    if not getattr(pol, "enabled", True) or getattr(pol, "type", None) != "price_level_trend":
                        continue
                    dec = execute_price_level_policy_demo_only(
                        adapter=adapter,
                        profile=profile,
                        log_dir=log_dir,
                        policy=pol,
                        context=mkt,
                        data_by_tf=data_by_tf,
                        tick=tick,
                        trades_df=trades_df,
                        mode=mode,
                    )
                    if dec.attempted:
                        if dec.placed:
                            print(f"[{profile.profile_name}] TRADE PLACED: {pol.type}:{pol.id} | side={pol.side} | entry={pol.price_level} | {dec.reason}")
                            pip = float(profile.pip_size)
                            tp = (pol.price_level + pol.tp_pips * pip) if getattr(pol, "tp_pips", None) and pol.side == "buy" else (pol.price_level - pol.tp_pips * pip) if getattr(pol, "tp_pips", None) and pol.side == "sell" else None
                            sl = (pol.price_level - pol.sl_pips * pip) if getattr(pol, "sl_pips", None) is not None and pol.side == "buy" else (pol.price_level + pol.sl_pips * pip) if getattr(pol, "sl_pips", None) is not None and pol.side == "sell" else None
                            _insert_trade_for_policy(
                                profile=profile,
                                adapter=adapter,
                                store=store,
                                policy_type=pol.type,
                                policy_id=pol.id,
                                side=pol.side,
                                entry_price=pol.price_level,
                                dec=dec,
                                stop_price=sl,
                                target_price=tp,
                                size_lots=float(get_effective_risk(profile).max_lots),
                            )
                        else:
                            print(f"[{profile.profile_name}] {dec.reason}")

            if (has_indicator or has_bollinger or has_vwap or has_ema_pullback) and mkt is None:
                mkt = _compute_mkt(profile, tick, data_by_tf)
            if has_indicator and mkt is not None:
                trades_df = store.read_trades_df(profile.profile_name)
                for pol in profile.execution.policies:
                    if not getattr(pol, "enabled", True) or getattr(pol, "type", None) != "indicator_based":
                        continue
                    tf_df = data_by_tf.get(getattr(pol, "timeframe", "M15"))
                    if tf_df is None or tf_df.empty:
                        continue
                    bar_time_utc = pd.to_datetime(tf_df["time"].iloc[-1], utc=True).isoformat()
                    dec = execute_indicator_policy_demo_only(
                        adapter=adapter,
                        profile=profile,
                        log_dir=log_dir,
                        policy=pol,
                        context=mkt,
                        data_by_tf=data_by_tf,
                        tick=tick,
                        trades_df=trades_df,
                        mode=mode,
                        bar_time_utc=bar_time_utc,
                    )
                    if dec.attempted:
                        if dec.placed:
                            entry_price = tick.ask if pol.side == "buy" else tick.bid
                            print(f"[{profile.profile_name}] TRADE PLACED: {pol.type}:{pol.id} | side={pol.side} | entry={entry_price:.3f} | {dec.reason}")
                            _insert_trade_for_policy(
                                profile=profile,
                                adapter=adapter,
                                store=store,
                                policy_type=pol.type,
                                policy_id=pol.id,
                                side=pol.side,
                                entry_price=entry_price,
                                dec=dec,
                            )
                        else:
                            print(f"[{profile.profile_name}] {dec.reason}")

            # Breakout range policies
            if has_breakout and mkt is None:
                mkt = _compute_mkt(profile, tick, data_by_tf)
            if has_breakout and mkt is not None:
                trades_df = store.read_trades_df(profile.profile_name)
                for pol in profile.execution.policies:
                    if not getattr(pol, "enabled", True) or getattr(pol, "type", None) != "breakout_range":
                        continue
                    tf_df = data_by_tf.get(getattr(pol, "timeframe", "M15"))
                    if tf_df is None or tf_df.empty:
                        continue
                    bar_time_utc = pd.to_datetime(tf_df["time"].iloc[-1], utc=True).isoformat()
                    dec = execute_breakout_policy_demo_only(
                        adapter=adapter,
                        profile=profile,
                        log_dir=log_dir,
                        policy=pol,
                        context=mkt,
                        data_by_tf=data_by_tf,
                        tick=tick,
                        trades_df=trades_df,
                        mode=mode,
                        bar_time_utc=bar_time_utc,
                    )
                    if dec.attempted:
                        if dec.placed:
                            side = dec.side or "buy"
                            entry_price = tick.ask if side == "buy" else tick.bid
                            print(f"[{profile.profile_name}] TRADE PLACED: breakout_range:{pol.id} | side={side} | entry={entry_price:.3f} | {dec.reason}")
                            _insert_trade_for_policy(
                                profile=profile,
                                adapter=adapter,
                                store=store,
                                policy_type="breakout_range",
                                policy_id=pol.id,
                                side=side,
                                entry_price=entry_price,
                                dec=dec,
                            )
                        else:
                            print(f"[{profile.profile_name}] breakout_range {pol.id} mode={mode} -> {dec.reason}")

            # Session momentum policies
            if has_session and mkt is None:
                mkt = _compute_mkt(profile, tick, data_by_tf)
            if has_session and mkt is not None:
                trades_df = store.read_trades_df(profile.profile_name)
                for pol in profile.execution.policies:
                    if not getattr(pol, "enabled", True) or getattr(pol, "type", None) != "session_momentum":
                        continue
                    dec = execute_session_momentum_policy_demo_only(
                        adapter=adapter,
                        profile=profile,
                        log_dir=log_dir,
                        policy=pol,
                        context=mkt,
                        data_by_tf=data_by_tf,
                        tick=tick,
                        trades_df=trades_df,
                        mode=mode,
                    )
                    if dec.attempted:
                        if dec.placed:
                            side = dec.side or "buy"
                            entry_price = tick.ask if side == "buy" else tick.bid
                            print(f"[{profile.profile_name}] TRADE PLACED: session_momentum:{pol.id} | side={side} | entry={entry_price:.3f} | {dec.reason}")
                            _insert_trade_for_policy(
                                profile=profile,
                                adapter=adapter,
                                store=store,
                                policy_type="session_momentum",
                                policy_id=pol.id,
                                side=side,
                                entry_price=entry_price,
                                dec=dec,
                            )
                        else:
                            print(f"[{profile.profile_name}] session_momentum {pol.id} mode={mode} -> {dec.reason}")

            # Bollinger Bands policies
            if has_bollinger and mkt is not None:
                trades_df = store.read_trades_df(profile.profile_name)
                for pol in profile.execution.policies:
                    if not getattr(pol, "enabled", True) or getattr(pol, "type", None) != "bollinger_bands":
                        continue
                    tf_df = data_by_tf.get(getattr(pol, "timeframe", "M15"))
                    if tf_df is None or tf_df.empty:
                        continue
                    bar_time_utc = pd.to_datetime(tf_df["time"].iloc[-1], utc=True).isoformat()
                    dec = execute_bollinger_policy_demo_only(
                        adapter=adapter,
                        profile=profile,
                        log_dir=log_dir,
                        policy=pol,
                        context=mkt,
                        data_by_tf=data_by_tf,
                        tick=tick,
                        trades_df=trades_df,
                        mode=mode,
                        bar_time_utc=bar_time_utc,
                    )
                    if dec.attempted:
                        if dec.placed:
                            side = dec.side or "buy"
                            entry_price = tick.ask if side == "buy" else tick.bid
                            print(f"[{profile.profile_name}] TRADE PLACED: bollinger_bands:{pol.id} | side={side} | entry={entry_price:.3f} | {dec.reason}")
                            _insert_trade_for_policy(
                                profile=profile,
                                adapter=adapter,
                                store=store,
                                policy_type="bollinger_bands",
                                policy_id=pol.id,
                                side=side,
                                entry_price=entry_price,
                                dec=dec,
                            )
                        else:
                            print(f"[{profile.profile_name}] bollinger_bands {pol.id} mode={mode} -> {dec.reason}")

            # VWAP policies
            if has_vwap and mkt is not None:
                trades_df = store.read_trades_df(profile.profile_name)
                for pol in profile.execution.policies:
                    if not getattr(pol, "enabled", True) or getattr(pol, "type", None) != "vwap":
                        continue
                    tf_df = data_by_tf.get(getattr(pol, "timeframe", "M15"))
                    if tf_df is None or tf_df.empty:
                        continue
                    bar_time_utc = pd.to_datetime(tf_df["time"].iloc[-1], utc=True).isoformat()
                    dec = execute_vwap_policy_demo_only(
                        adapter=adapter,
                        profile=profile,
                        log_dir=log_dir,
                        policy=pol,
                        context=mkt,
                        data_by_tf=data_by_tf,
                        tick=tick,
                        trades_df=trades_df,
                        mode=mode,
                        bar_time_utc=bar_time_utc,
                    )
                    if dec.attempted:
                        if dec.placed:
                            side = dec.side or "buy"
                            entry_price = tick.ask if side == "buy" else tick.bid
                            print(f"[{profile.profile_name}] TRADE PLACED: vwap:{pol.id} | side={side} | entry={entry_price:.3f} | {dec.reason}")
                            _insert_trade_for_policy(
                                profile=profile,
                                adapter=adapter,
                                store=store,
                                policy_type="vwap",
                                policy_id=pol.id,
                                side=side,
                                entry_price=entry_price,
                                dec=dec,
                            )
                        else:
                            print(f"[{profile.profile_name}] vwap {pol.id} mode={mode} -> {dec.reason}")

            # EMA pullback policies (M5-M15 momentum pullback)
            if has_ema_pullback and mkt is not None:
                trades_df = store.read_trades_df(profile.profile_name)
                for pol in profile.execution.policies:
                    if not getattr(pol, "enabled", True) or getattr(pol, "type", None) != "ema_pullback":
                        continue
                    tf_df = data_by_tf.get(getattr(pol, "entry_timeframe", "M5"))
                    if tf_df is None or tf_df.empty:
                        continue
                    bar_time_utc = pd.to_datetime(tf_df["time"].iloc[-1], utc=True).isoformat()
                    dec = execute_ema_pullback_policy_demo_only(
                        adapter=adapter,
                        profile=profile,
                        log_dir=log_dir,
                        policy=pol,
                        context=mkt,
                        data_by_tf=data_by_tf,
                        tick=tick,
                        trades_df=trades_df,
                        mode=mode,
                        bar_time_utc=bar_time_utc,
                    )
                    if dec.attempted:
                        if dec.placed:
                            side = dec.side or "buy"
                            entry_price = tick.ask if side == "buy" else tick.bid
                            pip = float(profile.pip_size)
                            sl_pips = getattr(pol, "sl_pips", None)
                            if sl_pips is None:
                                sl_pips = float(get_effective_risk(profile).min_stop_pips)
                            if side == "buy":
                                tp_price = entry_price + pol.tp_pips * pip
                                sl_price = entry_price - sl_pips * pip
                            else:
                                tp_price = entry_price - pol.tp_pips * pip
                                sl_price = entry_price + sl_pips * pip
                            print(f"[{profile.profile_name}] TRADE PLACED: ema_pullback:{pol.id} | side={side} | entry={entry_price:.3f} | {dec.reason}")
                            _insert_trade_for_policy(
                                profile=profile,
                                adapter=adapter,
                                store=store,
                                policy_type="ema_pullback",
                                policy_id=pol.id,
                                side=side,
                                entry_price=entry_price,
                                dec=dec,
                                stop_price=sl_price,
                                target_price=tp_price,
                            )
                        else:
                            print(f"[{profile.profile_name}] ema_pullback {pol.id} mode={mode} -> {dec.reason}")

            # KT/CG Hybrid policies (Trial #2)
            if has_kt_cg_hybrid and mkt is not None:
                trades_df = store.read_trades_df(profile.profile_name)
                for pol in profile.execution.policies:
                    if not getattr(pol, "enabled", True) or getattr(pol, "type", None) != "kt_cg_hybrid":
                        continue
                    m1_df = data_by_tf.get("M1")
                    if m1_df is None or m1_df.empty:
                        continue
                    bar_time_utc = pd.to_datetime(m1_df["time"].iloc[-1], utc=True).isoformat()
                    dec = execute_kt_cg_hybrid_policy_demo_only(
                        adapter=adapter,
                        profile=profile,
                        log_dir=log_dir,
                        policy=pol,
                        context=mkt,
                        data_by_tf=data_by_tf,
                        tick=tick,
                        trades_df=trades_df,
                        mode=mode,
                        bar_time_utc=bar_time_utc,
                    )
                    if dec.attempted:
                        if dec.placed:
                            side = dec.side or "buy"
                            entry_price = tick.ask if side == "buy" else tick.bid
                            pip = float(profile.pip_size)
                            sl_pips = getattr(pol, "sl_pips", None)
                            if sl_pips is None:
                                sl_pips = float(get_effective_risk(profile).min_stop_pips)
                            if side == "buy":
                                tp_price = entry_price + pol.tp_pips * pip
                                sl_price = entry_price - sl_pips * pip
                            else:
                                tp_price = entry_price - pol.tp_pips * pip
                                sl_price = entry_price + sl_pips * pip
                            print(f"[{profile.profile_name}] TRADE PLACED: kt_cg_hybrid:{pol.id} | side={side} | entry={entry_price:.3f} | {dec.reason}")
                            _insert_trade_for_policy(
                                profile=profile,
                                adapter=adapter,
                                store=store,
                                policy_type="kt_cg_hybrid",
                                policy_id=pol.id,
                                side=side,
                                entry_price=entry_price,
                                dec=dec,
                                stop_price=sl_price,
                                target_price=tp_price,
                            )
                        else:
                            print(f"[{profile.profile_name}] kt_cg_hybrid {pol.id} mode={mode} -> {dec.reason}")

            # KT/CG Counter-Trend Pullback policies (Trial #3)
            if has_kt_cg_ctp and mkt is not None:
                trades_df = store.read_trades_df(profile.profile_name)
                # Load temp overrides from runtime state if present
                temp_overrides = None
                try:
                    state_data = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
                    temp_overrides = {}
                    for key in ("temp_m5_trend_ema_fast", "temp_m5_trend_ema_slow", "temp_m1_zone_entry_ema_slow", "temp_m1_pullback_cross_ema_slow"):
                        val = state_data.get(key)
                        if val is not None:
                            # Map to policy field names
                            mapped_key = key.replace("temp_", "")
                            temp_overrides[mapped_key] = int(val)
                    if not temp_overrides:
                        temp_overrides = None
                    elif temp_overrides:
                        print(f"[{profile.profile_name}] Trial #3 TEMP OVERRIDES ACTIVE: {temp_overrides}")
                except Exception:
                    temp_overrides = None

                for pol in profile.execution.policies:
                    if not getattr(pol, "enabled", True) or getattr(pol, "type", None) != "kt_cg_counter_trend_pullback":
                        continue
                    m1_df = data_by_tf.get("M1")
                    if m1_df is None or m1_df.empty:
                        continue
                    bar_time_utc = pd.to_datetime(m1_df["time"].iloc[-1], utc=True).isoformat()
                    dec = execute_kt_cg_ctp_policy_demo_only(
                        adapter=adapter,
                        profile=profile,
                        log_dir=log_dir,
                        policy=pol,
                        context=mkt,
                        data_by_tf=data_by_tf,
                        tick=tick,
                        trades_df=trades_df,
                        mode=mode,
                        bar_time_utc=bar_time_utc,
                        temp_overrides=temp_overrides,
                    )
                    if dec.attempted:
                        if dec.placed:
                            side = dec.side or "buy"
                            entry_price = tick.ask if side == "buy" else tick.bid
                            pip = float(profile.pip_size)
                            sl_pips = getattr(pol, "sl_pips", None)
                            if sl_pips is None:
                                sl_pips = float(get_effective_risk(profile).min_stop_pips)
                            if side == "buy":
                                tp_price = entry_price + pol.tp_pips * pip
                                sl_price = entry_price - sl_pips * pip
                            else:
                                tp_price = entry_price - pol.tp_pips * pip
                                sl_price = entry_price + sl_pips * pip
                            print(f"[{profile.profile_name}] TRADE PLACED: kt_cg_ctp:{pol.id} | side={side} | entry={entry_price:.3f} | {dec.reason}")
                            _insert_trade_for_policy(
                                profile=profile,
                                adapter=adapter,
                                store=store,
                                policy_type="kt_cg_counter_trend_pullback",
                                policy_id=pol.id,
                                side=side,
                                entry_price=entry_price,
                                dec=dec,
                                stop_price=sl_price,
                                target_price=tp_price,
                            )
                        else:
                            print(f"[{profile.profile_name}] kt_cg_ctp {pol.id} mode={mode} -> {dec.reason}")

            # Trial #4 execution — runs EVERY poll cycle (not just on M1 bar close)
            # so tiered pullback can detect live price touches between bar closes
            if has_kt_cg_trial_4:
                # Use existing mkt if available (on bar close), otherwise compute minimal context
                t4_mkt = mkt
                if t4_mkt is None:
                    spread_pips = (tick.ask - tick.bid) / float(profile.pip_size)
                    t4_mkt = MarketContext(spread_pips=float(spread_pips), alignment_score=0)
                trades_df = store.read_trades_df(profile.profile_name)
                # Load temp overrides and tier state from runtime state if present
                temp_overrides = None
                tier_state: dict[int, bool] = {}
                try:
                    state_data = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
                    temp_overrides = {}
                    for key in ("temp_m3_trend_ema_fast", "temp_m3_trend_ema_slow",
                                "temp_m1_t4_zone_entry_ema_fast", "temp_m1_t4_zone_entry_ema_slow"):
                        val = state_data.get(key)
                        if val is not None:
                            # Map temp_ fields to policy field names
                            mapped_key = key.replace("temp_", "").replace("_t4_", "_")
                            temp_overrides[mapped_key] = int(val)
                    if not temp_overrides:
                        temp_overrides = None
                    # Load tier state for tiered pullback (dynamic dict)
                    tier_fired_raw = state_data.get("tier_fired")
                    if isinstance(tier_fired_raw, dict):
                        tier_state = {int(k): bool(v) for k, v in tier_fired_raw.items()}
                    else:
                        # Backward compat: read old tier_X_fired keys
                        tier_state = {}
                        for key, val in state_data.items():
                            if key.startswith("tier_") and key.endswith("_fired") and key != "tier_fired":
                                try:
                                    period = int(key.replace("tier_", "").replace("_fired", ""))
                                    tier_state[period] = bool(val)
                                except ValueError:
                                    pass
                    # Load divergence block state for RSI divergence detection
                    divergence_state: dict[str, str] = {}
                    block_buy_until = state_data.get("divergence_block_buy_until")
                    block_sell_until = state_data.get("divergence_block_sell_until")
                    if block_buy_until:
                        divergence_state["block_buy_until"] = block_buy_until
                    if block_sell_until:
                        divergence_state["block_sell_until"] = block_sell_until
                except Exception:
                    temp_overrides = None
                    tier_state = {}
                    divergence_state = {}

                for pol in profile.execution.policies:
                    if not getattr(pol, "enabled", True) or getattr(pol, "type", None) != "kt_cg_trial_4":
                        continue
                    m1_df = data_by_tf.get("M1")
                    if m1_df is None or m1_df.empty:
                        continue
                    bar_time_utc = pd.to_datetime(m1_df["time"].iloc[-1], utc=True).isoformat()
                    exec_result = execute_kt_cg_trial_4_policy_demo_only(
                        adapter=adapter,
                        profile=profile,
                        log_dir=log_dir,
                        policy=pol,
                        context=t4_mkt,
                        data_by_tf=data_by_tf,
                        tick=tick,
                        trades_df=trades_df,
                        mode=mode,
                        bar_time_utc=bar_time_utc,
                        tier_state=tier_state,
                        temp_overrides=temp_overrides,
                        divergence_state=divergence_state,
                    )
                    dec = exec_result["decision"]
                    tier_updates = exec_result.get("tier_updates", {})
                    divergence_updates = exec_result.get("divergence_updates", {})
                    t4_trigger_type = exec_result.get("trigger_type")

                    # Persist tier state updates to runtime_state.json
                    if tier_updates:
                        try:
                            current_state_data = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
                            # Use tier_fired dict (migrate away from old tier_X_fired keys)
                            tf_dict = current_state_data.get("tier_fired", {})
                            if not isinstance(tf_dict, dict):
                                tf_dict = {}
                            for tier, new_state in tier_updates.items():
                                tf_dict[str(tier)] = new_state
                                tier_state[tier] = new_state  # Update local state too
                            current_state_data["tier_fired"] = tf_dict
                            state_path.write_text(json.dumps(current_state_data, indent=2) + "\n", encoding="utf-8")
                        except Exception as e:
                            print(f"[{profile.profile_name}] Failed to persist tier state: {e}")

                    # Persist divergence state updates to runtime_state.json
                    if divergence_updates:
                        try:
                            current_state_data = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
                            for key, value in divergence_updates.items():
                                state_key = f"divergence_{key}"  # e.g., "divergence_block_buy_until"
                                current_state_data[state_key] = value
                                divergence_state[key] = value  # Update local state too
                            state_path.write_text(json.dumps(current_state_data, indent=2) + "\n", encoding="utf-8")
                        except Exception as e:
                            print(f"[{profile.profile_name}] Failed to persist divergence state: {e}")

                    if dec.attempted:
                        if dec.placed:
                            side = dec.side or "buy"
                            entry_price = tick.ask if side == "buy" else tick.bid
                            pip = float(profile.pip_size)
                            sl_pips = getattr(pol, "sl_pips", None)
                            if sl_pips is None:
                                sl_pips = float(get_effective_risk(profile).min_stop_pips)
                            if side == "buy":
                                tp_price = entry_price + pol.tp_pips * pip
                                sl_price = entry_price - sl_pips * pip
                            else:
                                tp_price = entry_price - pol.tp_pips * pip
                                sl_price = entry_price + sl_pips * pip
                            print(f"[{profile.profile_name}] TRADE PLACED: kt_cg_trial_4:{pol.id} | side={side} | entry={entry_price:.3f} | {dec.reason}")
                            _insert_trade_for_policy(
                                profile=profile,
                                adapter=adapter,
                                store=store,
                                policy_type="kt_cg_trial_4",
                                policy_id=pol.id,
                                side=side,
                                entry_price=entry_price,
                                dec=dec,
                                stop_price=sl_price,
                                target_price=tp_price,
                                entry_type=t4_trigger_type,
                            )
                        else:
                            print(f"[{profile.profile_name}] kt_cg_trial_4 {pol.id} mode={mode} -> {dec.reason}")

            # Trial #5 execution — same structure as Trial #4 with dual ATR filter
            if has_kt_cg_trial_5:
                t5_mkt = mkt
                if t5_mkt is None:
                    spread_pips = (tick.ask - tick.bid) / float(profile.pip_size)
                    t5_mkt = MarketContext(spread_pips=float(spread_pips), alignment_score=0)
                trades_df = store.read_trades_df(profile.profile_name)
                temp_overrides = None
                tier_state: dict[int, bool] = {}
                try:
                    state_data = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
                    temp_overrides = {}
                    for key in ("temp_m3_trend_ema_fast", "temp_m3_trend_ema_slow",
                                "temp_m1_t4_zone_entry_ema_fast", "temp_m1_t4_zone_entry_ema_slow"):
                        val = state_data.get(key)
                        if val is not None:
                            mapped_key = key.replace("temp_", "").replace("_t4_", "_")
                            temp_overrides[mapped_key] = int(val)
                    if not temp_overrides:
                        temp_overrides = None
                    tier_fired_raw = state_data.get("tier_fired")
                    if isinstance(tier_fired_raw, dict):
                        tier_state = {int(k): bool(v) for k, v in tier_fired_raw.items()}
                    else:
                        tier_state = {}
                        for key, val in state_data.items():
                            if key.startswith("tier_") and key.endswith("_fired") and key != "tier_fired":
                                try:
                                    period = int(key.replace("tier_", "").replace("_fired", ""))
                                    tier_state[period] = bool(val)
                                except ValueError:
                                    pass
                    divergence_state: dict[str, str] = {}
                    block_buy_until = state_data.get("divergence_block_buy_until")
                    block_sell_until = state_data.get("divergence_block_sell_until")
                    if block_buy_until:
                        divergence_state["block_buy_until"] = block_buy_until
                    if block_sell_until:
                        divergence_state["block_sell_until"] = block_sell_until
                    # Load daily reset state for Trial #5
                    daily_reset_state_t5: dict = {
                        "daily_reset_date": state_data.get("daily_reset_date"),
                        "daily_reset_high": state_data.get("daily_reset_high"),
                        "daily_reset_low": state_data.get("daily_reset_low"),
                        "daily_reset_block_active": bool(state_data.get("daily_reset_block_active", False)),
                        "daily_reset_settled": bool(state_data.get("daily_reset_settled", False)),
                    }
                    # Load exhaustion state for Trial #5
                    exhaustion_state_t5: dict = {
                        "trend_flip_price": state_data.get("trend_flip_price"),
                        "trend_flip_direction": state_data.get("trend_flip_direction"),
                        "trend_flip_time": state_data.get("trend_flip_time"),
                    }
                except Exception:
                    temp_overrides = None
                    tier_state = {}
                    divergence_state = {}
                    daily_reset_state_t5 = {}
                    exhaustion_state_t5 = {}

                for pol in profile.execution.policies:
                    if not getattr(pol, "enabled", True) or getattr(pol, "type", None) != "kt_cg_trial_5":
                        continue
                    m1_df = data_by_tf.get("M1")
                    if m1_df is None or m1_df.empty:
                        continue
                    bar_time_utc = pd.to_datetime(m1_df["time"].iloc[-1], utc=True).isoformat()
                    exec_result = execute_kt_cg_trial_5_policy_demo_only(
                        adapter=adapter,
                        profile=profile,
                        log_dir=log_dir,
                        policy=pol,
                        context=t5_mkt,
                        data_by_tf=data_by_tf,
                        tick=tick,
                        trades_df=trades_df,
                        mode=mode,
                        bar_time_utc=bar_time_utc,
                        tier_state=tier_state,
                        temp_overrides=temp_overrides,
                        divergence_state=divergence_state,
                        daily_reset_state=daily_reset_state_t5,
                        exhaustion_state=exhaustion_state_t5,
                    )
                    dec = exec_result["decision"]
                    tier_updates = exec_result.get("tier_updates", {})
                    divergence_updates = exec_result.get("divergence_updates", {})
                    t5_trigger_type = exec_result.get("trigger_type")

                    # Exhaustion status — only log on zone CHANGE (Fix 3: reduce log spam)
                    if getattr(pol, "trend_exhaustion_enabled", False):
                        ex_result = exec_result.get("exhaustion_result")
                        if ex_result is not None:
                            ex_state = exec_result.get("exhaustion_state") or {}
                            zone = ex_result.get("zone", "FRESH")
                            if zone != _last_exhaustion_zone:
                                flip_price = ex_state.get("trend_flip_price")
                                last_cross = f"{flip_price:.3f}" if flip_price is not None else "—"
                                ratio_raw = ex_result.get("extension_ratio")
                                ratio_adj = ex_result.get("adjusted_ratio")
                                tf_val = ex_result.get("time_factor")
                                raw_str = f"{ratio_raw}x" if ratio_raw is not None else "—"
                                adj_str = f"{ratio_adj}x (tf={tf_val:.2f})" if ratio_adj is not None and tf_val is not None else "—"
                                print(f"[{profile.profile_name}] Exhaustion ZONE CHANGE: {_last_exhaustion_zone} -> {zone} | cross@{last_cross} | raw={raw_str} adj={adj_str}")
                                _last_exhaustion_zone = zone

                    # Persist tier RESETS immediately (value=False means price moved away).
                    # Tier FIRES (value=True) are deferred until trade is confirmed placed (Fix 6).
                    tier_resets = {t: v for t, v in tier_updates.items() if not v}
                    tier_fires = {t: v for t, v in tier_updates.items() if v}
                    if tier_resets:
                        try:
                            current_state_data = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
                            tf_dict = current_state_data.get("tier_fired", {})
                            if not isinstance(tf_dict, dict):
                                tf_dict = {}
                            for tier, new_state in tier_resets.items():
                                tf_dict[str(tier)] = new_state
                                tier_state[tier] = new_state
                            current_state_data["tier_fired"] = tf_dict
                            state_path.write_text(json.dumps(current_state_data, indent=2) + "\n", encoding="utf-8")
                        except Exception as e:
                            print(f"[{profile.profile_name}] Failed to persist tier resets: {e}")

                    # Persist divergence state updates
                    if divergence_updates:
                        try:
                            current_state_data = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
                            for key, value in divergence_updates.items():
                                state_key = f"divergence_{key}"
                                current_state_data[state_key] = value
                                divergence_state[key] = value
                            state_path.write_text(json.dumps(current_state_data, indent=2) + "\n", encoding="utf-8")
                        except Exception as e:
                            print(f"[{profile.profile_name}] Failed to persist divergence state: {e}")

                    # Persist daily reset state (updated in-place by engine)
                    if daily_reset_state_t5:
                        try:
                            current_state_data = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
                            for key in ("daily_reset_date", "daily_reset_high", "daily_reset_low", "daily_reset_block_active", "daily_reset_settled"):
                                current_state_data[key] = daily_reset_state_t5.get(key)
                            state_path.write_text(json.dumps(current_state_data, indent=2) + "\n", encoding="utf-8")
                        except Exception as e:
                            print(f"[{profile.profile_name}] Failed to persist daily reset state: {e}")

                    # Persist exhaustion state and handle tier reset on flip
                    returned_exhaustion = exec_result.get("exhaustion_state", {})
                    returned_exhaust_result = exec_result.get("exhaustion_result")
                    if returned_exhaustion:
                        try:
                            current_state_data = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
                            for key in ("trend_flip_price", "trend_flip_direction", "trend_flip_time"):
                                current_state_data[key] = returned_exhaustion.get(key)
                            # On trend flip: clear all tier_fired states
                            if returned_exhaust_result and returned_exhaust_result.get("reset_tiers"):
                                current_state_data["tier_fired"] = {}
                                tier_state.clear()
                                print(f"[{profile.profile_name}] TREND FLIP detected -> all tier_fired states cleared")
                            state_path.write_text(json.dumps(current_state_data, indent=2) + "\n", encoding="utf-8")
                            exhaustion_state_t5 = returned_exhaustion
                        except Exception as e:
                            print(f"[{profile.profile_name}] Failed to persist exhaustion state: {e}")

                    if dec.attempted:
                        if dec.placed:
                            side = dec.side or "buy"
                            entry_price = tick.ask if side == "buy" else tick.bid
                            pip = float(profile.pip_size)
                            sl_pips = getattr(pol, "sl_pips", None)
                            if sl_pips is None:
                                sl_pips = float(get_effective_risk(profile).min_stop_pips)
                            if side == "buy":
                                tp_price = entry_price + pol.tp_pips * pip
                                sl_price = entry_price - sl_pips * pip
                            else:
                                tp_price = entry_price - pol.tp_pips * pip
                                sl_price = entry_price + sl_pips * pip
                            print(f"[{profile.profile_name}] TRADE PLACED: kt_cg_trial_5:{pol.id} | side={side} | entry={entry_price:.3f} | {dec.reason}")
                            _insert_trade_for_policy(
                                profile=profile,
                                adapter=adapter,
                                store=store,
                                policy_type="kt_cg_trial_5",
                                policy_id=pol.id,
                                side=side,
                                entry_price=entry_price,
                                dec=dec,
                                stop_price=sl_price,
                                target_price=tp_price,
                                entry_type=t5_trigger_type,
                            )
                            # Fix 6: Only persist tier FIRES after trade is confirmed placed
                            if tier_fires:
                                try:
                                    current_state_data = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
                                    tf_dict = current_state_data.get("tier_fired", {})
                                    if not isinstance(tf_dict, dict):
                                        tf_dict = {}
                                    for tier, new_state in tier_fires.items():
                                        tf_dict[str(tier)] = new_state
                                        tier_state[tier] = new_state
                                    current_state_data["tier_fired"] = tf_dict
                                    state_path.write_text(json.dumps(current_state_data, indent=2) + "\n", encoding="utf-8")
                                except Exception as e:
                                    print(f"[{profile.profile_name}] Failed to persist tier fires: {e}")
                        elif dec.reason and "dead_zone" not in dec.reason:
                            # Fix 7: Only log blocks that aren't routine (skip dead zone, skip non-attempted)
                            print(f"[{profile.profile_name}] kt_cg_trial_5 BLOCKED: {dec.reason}")

                    # Fix 7: Periodic summary every 30 seconds
                    now_ts = time.time()
                    if now_ts - _last_summary_time >= _SUMMARY_INTERVAL:
                        _last_summary_time = now_ts
                        ex_zone = _last_exhaustion_zone or "—"
                        fired_tiers = [str(t) for t, v in tier_state.items() if v]
                        avail_tiers = [str(t) for t, v in tier_state.items() if not v]
                        print(f"[{profile.profile_name}] [SUMMARY] zone={ex_zone} | tiers_fired=[{','.join(fired_tiers)}] avail=[{','.join(avail_tiers)}]")

            if args.once:
                break

            time.sleep(poll_sec)

    finally:
        adapter.shutdown()


if __name__ == "__main__":
    main()

