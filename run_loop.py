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
    execute_kt_cg_hybrid_policy_demo_only,
    execute_m5_m1_ema_cross_policy_demo_only,
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
    return max(1.0, base)


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
    if (not breakeven or not getattr(breakeven, "enabled", False)) and (
        not target or getattr(target, "mode", None) != "scaled" or not getattr(target, "tp1_pips", None)
    ):
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
        # 1) Breakeven first
        if breakeven and getattr(breakeven, "enabled", False):
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
        has_kt_cg_hybrid = any(
            getattr(p, "type", None) == "kt_cg_hybrid" and getattr(p, "enabled", True)
            for p in profile.execution.policies
        )
        has_m5_m1_ema_cross = any(
            getattr(p, "type", None) == "m5_m1_ema_cross" and getattr(p, "enabled", True)
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
                        "H4": adapter.get_bars(profile.symbol, "H4", 800),
                        "M15": adapter.get_bars(profile.symbol, "M15", 2000),
                        "M1": adapter.get_bars(profile.symbol, "M1", 3000),
                    }
                    # Fetch M5 data when needed by ema_pullback, M5 confirmed-cross, or m5_m1_ema_cross.
                    if has_ema_pullback or has_m5_confirmed_cross or has_m5_m1_ema_cross:
                        data_by_tf["M5"] = adapter.get_bars(profile.symbol, "M5", 2000)
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

            # KT/CG Hybrid policies
            if has_kt_cg_hybrid and mkt is None:
                mkt = _compute_mkt(profile, tick, data_by_tf)
            if has_kt_cg_hybrid and mkt is not None:
                trades_df = store.read_trades_df(profile.profile_name)
                for pol in profile.execution.policies:
                    if not getattr(pol, "enabled", True) or getattr(pol, "type", None) != "kt_cg_hybrid":
                        continue
                    tf_df = data_by_tf.get(getattr(pol, "entry_timeframe", "M1"))
                    if tf_df is None or tf_df.empty:
                        continue
                    bar_time_utc = pd.to_datetime(tf_df["time"].iloc[-1], utc=True).isoformat()
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
                            tp_price = entry_price + pol.tp_pips * pip if side == "buy" else entry_price - pol.tp_pips * pip
                            sl_price = entry_price - pol.sl_pips * pip if side == "buy" else entry_price + pol.sl_pips * pip
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

            # M5/M1 EMA Cross policies
            if has_m5_m1_ema_cross and mkt is None:
                mkt = _compute_mkt(profile, tick, data_by_tf)
            if has_m5_m1_ema_cross and mkt is not None:
                trades_df = store.read_trades_df(profile.profile_name)
                for pol in profile.execution.policies:
                    if not getattr(pol, "enabled", True) or getattr(pol, "type", None) != "m5_m1_ema_cross":
                        continue
                    m5_df = data_by_tf.get("M5")
                    if m5_df is None or m5_df.empty:
                        continue
                    bar_time_utc = pd.to_datetime(m5_df["time"].iloc[-1], utc=True).isoformat()
                    dec = execute_m5_m1_ema_cross_policy_demo_only(
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
                            tp_pips = getattr(pol, "tp_strong", 2.0)  # Approximate; actual TP computed dynamically
                            sl_pips = getattr(pol, "sl_pips", 20.0)
                            if side == "buy":
                                tp_price = entry_price + tp_pips * pip
                                sl_price = entry_price - sl_pips * pip
                            else:
                                tp_price = entry_price - tp_pips * pip
                                sl_price = entry_price + sl_pips * pip
                            print(f"[{profile.profile_name}] TRADE PLACED: m5_m1_ema_cross:{pol.id} | side={side} | entry={entry_price:.3f} | {dec.reason}")
                            _insert_trade_for_policy(
                                profile=profile,
                                adapter=adapter,
                                store=store,
                                policy_type="m5_m1_ema_cross",
                                policy_id=pol.id,
                                side=side,
                                entry_price=entry_price,
                                dec=dec,
                                stop_price=sl_price,
                                target_price=tp_price,
                                size_lots=float(getattr(pol, "lots", 0.01)),
                            )
                        else:
                            print(f"[{profile.profile_name}] m5_m1_ema_cross {pol.id} mode={mode} -> {dec.reason}")

            if args.once:
                break

            time.sleep(poll_sec)

    finally:
        adapter.shutdown()


if __name__ == "__main__":
    main()

