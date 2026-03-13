from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

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
    execute_h1_breakout_policy_demo_only,
    execute_indicator_policy_demo_only,
    execute_kt_cg_ctp_policy_demo_only,
    execute_kt_cg_hybrid_policy_demo_only,
    execute_kt_cg_trial_4_policy_demo_only,
    execute_kt_cg_trial_5_policy_demo_only,
    execute_kt_cg_trial_6_policy_demo_only,
    execute_kt_cg_trial_7_policy_demo_only,
    execute_kt_cg_trial_8_policy_demo_only,
    execute_kt_cg_trial_9_policy_demo_only,
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
from core.dashboard_models import (
    DailySummary, DashboardState, PositionInfo, TradeEvent,
    append_trade_event, write_dashboard_state,
)
from core.dashboard_reporters import (
    collect_trial_2_context, collect_trial_3_context,
    collect_trial_4_context, collect_trial_5_context,
    collect_trial_6_context, collect_trial_7_context,
    collect_uncle_parsh_context,
)
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


def _normalized_profit_for_dashboard_row(row: dict, symbol_hint: str) -> float | None:
    """Use stored profit unless it's a clear outlier; then use price-based estimate."""
    def _f(v):
        try:
            if v is None:
                return None
            return float(v)
        except (TypeError, ValueError):
            return None

    raw = _f(row.get("profit"))
    entry = _f(row.get("entry_price"))
    exit_ = _f(row.get("exit_price"))
    lots = _f(row.get("size_lots")) or _f(row.get("volume"))
    side = str(row.get("side") or "").lower()
    symbol = "".join(ch for ch in str(row.get("symbol") or symbol_hint or "").upper() if ch.isalpha())[:6]

    est = None
    if entry is not None and exit_ is not None and lots and side in ("buy", "sell"):
        diff = (exit_ - entry) if side == "buy" else (entry - exit_)
        units = lots * 100_000.0
        if symbol.endswith("JPY") and exit_ > 0:
            est = round(diff * units / exit_, 2)
        elif len(symbol) == 6 and symbol[3:] == "USD":
            est = round(diff * units, 2)
        elif len(symbol) == 6 and symbol[:3] == "USD" and exit_ > 0:
            est = round(diff * units / exit_, 2)

    if raw is None:
        return est
    if est is None:
        return raw
    abs_raw = abs(raw)
    abs_est = abs(est)
    if abs_raw >= 250 and abs_est >= 5:
        ratio = abs_raw / abs_est if abs_est > 0 else float("inf")
        if ratio >= 6.0:
            return est
    if abs_raw >= 250 and abs_est >= 20 and (raw * est) < 0:
        return est
    return raw


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
    entry_session: str | None = None,
    risk_usd_planned: float | None = None,
) -> None:
    """Store a trade in DB when a policy places an order. Ensures preset and position_id for Performance by Preset."""
    try:
        position_id = None
        if dec.deal_id:
            position_id = adapter.get_position_id_from_deal(dec.deal_id)
        if position_id is None and dec.order_id:
            position_id = adapter.get_position_id_from_order(dec.order_id)
        if position_id is None and dec.order_id:
            time.sleep(1)
            position_id = adapter.get_position_id_from_order(dec.order_id)
        # OANDA: tradeID can appear slightly after order creation; retry a few times.
        if position_id is None and dec.order_id and getattr(profile, "broker_type", None) == "oanda":
            for _ in range(3):
                try:
                    time.sleep(0.5)
                    position_id = adapter.get_position_id_from_order(dec.order_id)
                    if position_id is not None:
                        break
                except Exception:
                    position_id = None
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
        if entry_session:
            row["entry_session"] = entry_session
        if risk_usd_planned is not None:
            row["risk_usd_planned"] = float(risk_usd_planned)
        # Capture entry slippage if fill_price available
        if getattr(dec, 'fill_price', None) is not None:
            slippage = abs(dec.fill_price - entry_price) / float(profile.pip_size)
            row["entry_slippage_pips"] = round(slippage, 3)
        store.insert_trade(row)
    except Exception:
        pass


# Minimum partial-close size (lots). OANDA typically requires >= 1000 units = 0.01 lots.
MIN_CLOSE_LOTS = 0.01


def _run_trade_management(profile, adapter, store, tick, phase3_state: dict | None = None, open_positions: list | None = None) -> None:
    """Apply breakeven and TP1 partial close for open positions we manage. Order: breakeven first, then TP1."""
    tm = getattr(profile, "trade_management", None)
    if tm is None:
        return
    breakeven = getattr(tm, "breakeven", None)
    target = getattr(tm, "target", None)

    # Find Trial #4/#5/#6/#7 spread-aware BE config if present
    t4_spread_be = None
    for pol in profile.execution.policies:
        if getattr(pol, "type", None) in ("kt_cg_trial_4", "kt_cg_trial_5", "kt_cg_trial_6", "kt_cg_trial_7", "kt_cg_trial_8") and getattr(pol, "enabled", True):
            if getattr(pol, "spread_aware_be_enabled", False):
                t4_spread_be = pol
            break

    # Find Uncle Parsh policy if present
    up_policy = None
    up_tm_overrides: dict = {}
    for pol in profile.execution.policies:
        if getattr(pol, "type", None) == "uncle_parsh_h1_breakout" and getattr(pol, "enabled", True):
            up_policy = pol
            # Load temp overrides for trade management fields
            try:
                _up_sp = Path("runtime") / profile.profile_name / "runtime_state.json"
                if _up_sp.exists():
                    _up_sd = json.loads(_up_sp.read_text(encoding="utf-8"))
                    for _fld, _sk in [
                        ("tp1_pips", "temp_up_tp1_pips"),
                        ("tp1_close_pct", "temp_up_tp1_close_pct"),
                        ("be_spread_plus_pips", "temp_up_be_spread_plus_pips"),
                        ("trail_ema_period", "temp_up_trail_ema_period"),
                    ]:
                        _v = _up_sd.get(_sk)
                        if _v is not None:
                            up_tm_overrides[_fld] = _v
            except Exception:
                pass
            break

    # Find Trial #8 policy for managed exits:
    # - tp1_be_trail: TP1 + BE + optional M1 EMA trail
    # - ema_scale_runner: H1 breakout style
    # - none: broker TP/SL only (no managed exits)
    t8_policy = None
    t8_tm_overrides: dict = {}
    for pol in profile.execution.policies:
        if getattr(pol, "type", None) == "kt_cg_trial_8" and getattr(pol, "enabled", True):
            t8_exit_strategy = str(getattr(pol, "exit_strategy", "tp1_be_trail") or "tp1_be_trail")
            if t8_exit_strategy != "none" and (getattr(pol, "trail_after_tp1", False) or t8_exit_strategy == "ema_scale_runner"):
                t8_policy = pol
                try:
                    _t8_sp = Path("runtime") / profile.profile_name / "runtime_state.json"
                    if _t8_sp.exists():
                        _t8_sd = json.loads(_t8_sp.read_text(encoding="utf-8"))
                        for _fld, _sk in [
                            ("exit_strategy", "temp_t8_exit_strategy"),
                            ("tp1_pips", "temp_t8_tp1_pips"),
                            ("tp1_close_pct", "temp_t8_tp1_close_pct"),
                            ("be_spread_plus_pips", "temp_t8_be_spread_plus_pips"),
                            ("trail_ema_period", "temp_t8_trail_ema_period"),
                            ("m1_exit_ema_fast", "temp_t8_m1_exit_ema_fast"),
                            ("m1_exit_ema_slow", "temp_t8_m1_exit_ema_slow"),
                            ("scale_out_pct", "temp_t8_scale_out_pct"),
                            ("initial_sl_spread_plus_pips", "temp_t8_initial_sl_spread_plus_pips"),
                        ]:
                            _v = _t8_sd.get(_sk)
                            if _v is not None:
                                t8_tm_overrides[_fld] = _v
                except Exception:
                    pass
            break

    # Find Trial #9 policy for managed exits (TP1 + BE + bar-close trail + Kill Switch)
    t9_policy = None
    for pol in profile.execution.policies:
        if getattr(pol, "type", None) == "kt_cg_trial_9" and getattr(pol, "enabled", True):
            t9_policy = pol
            break

    # Find Trial #7 policy (if any) that uses ema_scale_runner exit (H1 breakout style).
    t7_ema_policy = None
    for pol in profile.execution.policies:
        if getattr(pol, "type", None) == "kt_cg_trial_7" and getattr(pol, "enabled", True):
            es7 = str(getattr(pol, "exit_strategy", "tp_sl_be") or "tp_sl_be")
            if es7 == "ema_scale_runner":
                t7_ema_policy = pol
                break

    has_simple_be = breakeven and getattr(breakeven, "enabled", False)
    has_scaled = target and getattr(target, "mode", None) == "scaled" and getattr(target, "tp1_pips", None)
    has_t8_trail = t8_policy is not None
    has_t9_trail = t9_policy is not None
    has_t7_ema = t7_ema_policy is not None
    has_phase3_trail = any(
        getattr(p, "type", None) == "phase3_integrated" and getattr(p, "enabled", True)
        for p in profile.execution.policies
    )
    if not has_simple_be and not has_scaled and t4_spread_be is None and up_policy is None and not has_t7_ema and not has_t8_trail and not has_t9_trail and not has_phase3_trail:
        return
    # Reuse shared open_positions snapshot when provided; fall back to adapter call otherwise.
    if open_positions is None:
        try:
            open_positions = adapter.get_open_positions(profile.symbol)
        except Exception:
            return
    if not open_positions:
        return
    # sqlite3.Row doesn't support .get(); convert to dicts for safe access
    our_trades = [dict(r) for r in store.list_open_trades(profile.profile_name)]
    position_to_trade = {}
    for row in our_trades:
        pid = row.get("mt5_position_id")
        if pid is not None:
            try:
                position_to_trade[int(pid)] = row
            except (TypeError, ValueError):
                pass
    pip = float(profile.pip_size)
    mid = (tick.bid + tick.ask) / 2.0
    current_spread = tick.ask - tick.bid
    unmatched_position_ids: list[int] = []

    # Pre-fetch bar data ONCE before the per-position loop.
    # Without this, every policy calls adapter.get_bars() for each open position → O(N) API calls.
    # N open positions × 2 bar fetches × 0.25s rate limit = N/2 seconds of stall per loop.
    _tm_m1_df = None  # M1(250): covers T9 kill switch/trail, T8/T7/UP EMA exits (all need ≤250)
    _tm_m5_df = None  # M5(100): covers T9 kill switch (needs 50) and M5 trail (needs 100)
    _needs_m1 = has_t9_trail or has_t8_trail or has_t7_ema or up_policy is not None
    _needs_m5 = has_t9_trail
    if open_positions and (_needs_m1 or _needs_m5):
        try:
            if _needs_m1:
                _tm_m1_df = adapter.get_bars(profile.symbol, "M1", 250)
            if _needs_m5:
                _tm_m5_df = adapter.get_bars(profile.symbol, "M5", 100)
        except Exception:
            pass

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
            unmatched_position_ids.append(position_id)
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
            elif entry_type == "ema_tier" and getattr(t4_spread_be, "spread_aware_be_apply_to_ema_tier", True):
                apply = True
            elif entry_type == "bb_reversal" and getattr(t4_spread_be, "spread_aware_be_apply_to_bb_reversal", True):
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

        # 2) Trial #7 ema_scale_runner (H1 breakout style; no TP/BE moves). Run before other exits.
        if (
            t7_ema_policy is not None
            and entry_type in ("zone_entry", "tiered_pullback")
        ):
            trade_policy_type = trade_id.split(":", 1)[0] if ":" in trade_id else ""
            if trade_policy_type == "kt_cg_trial_7" and str(getattr(t7_ema_policy, "exit_strategy", "tp_sl_be")) == "ema_scale_runner":
                try:
                    from core.indicators import ema as ema_fn
                    ema_fast_p = int(getattr(t7_ema_policy, "m1_exit_ema_fast", 9))
                    ema_slow_p = int(getattr(t7_ema_policy, "m1_exit_ema_slow", 21))
                    scale_pct = float(getattr(t7_ema_policy, "scale_out_pct", 50.0))
                    m1_df_t7 = _tm_m1_df
                    if m1_df_t7 is not None and not m1_df_t7.empty and len(m1_df_t7) >= max(ema_fast_p, ema_slow_p) + 2:
                        m1_close_t7 = m1_df_t7["close"].astype(float)
                        ema_fast = ema_fn(m1_close_t7, ema_fast_p)
                        ema_slow = ema_fn(m1_close_t7, ema_slow_p)
                        if not ema_fast.empty and not ema_slow.empty and pd.notna(ema_fast.iloc[-1]) and pd.notna(ema_slow.iloc[-1]):
                            last_close = float(m1_close_t7.iloc[-1])
                            last_ema_fast = float(ema_fast.iloc[-1])
                            last_ema_slow = float(ema_slow.iloc[-1])
                            tp1_done = trade_row.get("tp1_partial_done") or 0
                            if isinstance(pos, dict):
                                current_units = pos.get("currentUnits") or 0
                                current_lots = abs(int(current_units)) / 100_000.0
                            else:
                                current_lots = float(getattr(pos, "volume", 0) or 0)
                            position_type = 1 if side == "sell" else 0
                            if not tp1_done:
                                wrong_side_fast = (side == "buy" and last_close < last_ema_fast) or (side == "sell" and last_close > last_ema_fast)
                                if wrong_side_fast and current_lots > 0:
                                    close_lots = current_lots * (scale_pct / 100.0)
                                    if close_lots > 0:
                                        adapter.close_position(
                                            ticket=position_id,
                                            symbol=profile.symbol,
                                            volume=close_lots,
                                            position_type=position_type,
                                        )
                                        store.update_trade(trade_id, {"tp1_partial_done": 1})
                                        print(f"[{profile.profile_name}] T7 ema_scale_runner scale-out: pos {position_id} ({scale_pct:.0f}%) wrong side EMA{ema_fast_p}")
                            else:
                                wrong_side_slow = (side == "buy" and last_close < last_ema_slow) or (side == "sell" and last_close > last_ema_slow)
                                if wrong_side_slow and current_lots > 0:
                                    adapter.close_position(
                                        ticket=position_id,
                                        symbol=profile.symbol,
                                        volume=current_lots,
                                        position_type=position_type,
                                    )
                                    print(f"[{profile.profile_name}] T7 ema_scale_runner runner closed: pos {position_id} wrong side EMA{ema_slow_p}")
                except Exception as e:
                    print(f"[{profile.profile_name}] T7 ema_scale_runner exit error pos {position_id}: {e}")
                continue

        # 3) Uncle Parsh H1 Breakout trade management (EMA exits; no BE moves)
        if up_policy is not None and entry_type in ("power_break", "sniper"):
            try:
                from core.indicators import ema as ema_fn

                ema_fast_p = int(getattr(up_policy, "m1_entry_ema_fast", 9))
                ema_slow_p = int(getattr(up_policy, "m1_entry_ema_slow", 21))
                scale_pct = float(getattr(up_policy, "scale_out_pct", 50.0))

                m1_df_exit = _tm_m1_df
                if m1_df_exit is None or m1_df_exit.empty or len(m1_df_exit) < max(ema_fast_p, ema_slow_p) + 2:
                    continue
                m1_close_exit = m1_df_exit["close"].astype(float)
                ema_fast = ema_fn(m1_close_exit, ema_fast_p)
                ema_slow = ema_fn(m1_close_exit, ema_slow_p)
                if ema_fast.empty or ema_slow.empty or pd.isna(ema_fast.iloc[-1]) or pd.isna(ema_slow.iloc[-1]):
                    continue
                last_close = float(m1_close_exit.iloc[-1])
                last_ema_fast = float(ema_fast.iloc[-1])
                last_ema_slow = float(ema_slow.iloc[-1])

                tp1_done = trade_row.get("tp1_partial_done") or 0

                if isinstance(pos, dict):
                    current_units = pos.get("currentUnits") or 0
                    current_lots = abs(int(current_units)) / 100_000.0
                else:
                    current_lots = float(getattr(pos, "volume", 0) or 0)
                position_type = 1 if side == "sell" else 0

                # Scale-out (50%) when M1 close goes wrong side of EMA9
                if not tp1_done:
                    wrong_side_ema9 = (side == "buy" and last_close < last_ema_fast) or (side == "sell" and last_close > last_ema_fast)
                    if wrong_side_ema9:
                        close_lots = current_lots * (scale_pct / 100.0)
                        if close_lots > 0:
                            adapter.close_position(
                                ticket=position_id,
                                symbol=profile.symbol,
                                volume=close_lots,
                                position_type=position_type,
                            )
                            store.update_trade(trade_id, {"tp1_partial_done": 1})
                            print(f"[{profile.profile_name}] UP scale-out: pos {position_id} ({scale_pct:.0f}%) close wrong side EMA{ema_fast_p}")

                # Runner exit when M1 close goes wrong side of EMA21
                else:
                    wrong_side_ema21 = (side == "buy" and last_close < last_ema_slow) or (side == "sell" and last_close > last_ema_slow)
                    if wrong_side_ema21 and current_lots > 0:
                        adapter.close_position(
                            ticket=position_id,
                            symbol=profile.symbol,
                            volume=current_lots,
                            position_type=position_type,
                        )
                        print(f"[{profile.profile_name}] UP runner closed: pos {position_id} close wrong side EMA{ema_slow_p}")
            except Exception as e:
                print(f"[{profile.profile_name}] UP EMA exit error pos {position_id}: {e}")

        # 2b) Trial #8: ema_scale_runner (H1 breakout style) or tp1_be_trail (TP1 + BE + M1 EMA trail)
        elif t8_policy is not None and entry_type in ("zone_entry", "tiered_pullback"):
            t8_exit_strategy = t8_tm_overrides.get("exit_strategy", getattr(t8_policy, "exit_strategy", "tp1_be_trail"))

            if t8_exit_strategy == "ema_scale_runner":
                # H1 breakout style: scale-out on M1 close wrong side of EMA fast, runner on wrong side of EMA slow; no TP, no BE
                try:
                    from core.indicators import ema as ema_fn
                    ema_fast_p = int(t8_tm_overrides.get("m1_exit_ema_fast", getattr(t8_policy, "m1_exit_ema_fast", 9)))
                    ema_slow_p = int(t8_tm_overrides.get("m1_exit_ema_slow", getattr(t8_policy, "m1_exit_ema_slow", 21)))
                    scale_pct = float(t8_tm_overrides.get("scale_out_pct", getattr(t8_policy, "scale_out_pct", 50.0)))
                    m1_df_t8 = _tm_m1_df
                    if m1_df_t8 is not None and not m1_df_t8.empty and len(m1_df_t8) >= max(ema_fast_p, ema_slow_p) + 2:
                        m1_close_t8 = m1_df_t8["close"].astype(float)
                        ema_fast = ema_fn(m1_close_t8, ema_fast_p)
                        ema_slow = ema_fn(m1_close_t8, ema_slow_p)
                        if not ema_fast.empty and not ema_slow.empty and pd.notna(ema_fast.iloc[-1]) and pd.notna(ema_slow.iloc[-1]):
                            last_close = float(m1_close_t8.iloc[-1])
                            last_ema_fast = float(ema_fast.iloc[-1])
                            last_ema_slow = float(ema_slow.iloc[-1])
                            tp1_done = trade_row.get("tp1_partial_done") or 0
                            if isinstance(pos, dict):
                                current_units = pos.get("currentUnits") or 0
                                current_lots = abs(int(current_units)) / 100_000.0
                            else:
                                current_lots = float(getattr(pos, "volume", 0) or 0)
                            position_type = 1 if side == "sell" else 0
                            if not tp1_done:
                                wrong_side_ema9 = (side == "buy" and last_close < last_ema_fast) or (side == "sell" and last_close > last_ema_fast)
                                if wrong_side_ema9 and current_lots > 0:
                                    close_lots = current_lots * (scale_pct / 100.0)
                                    if close_lots > 0:
                                        adapter.close_position(
                                            ticket=position_id,
                                            symbol=profile.symbol,
                                            volume=close_lots,
                                            position_type=position_type,
                                        )
                                        store.update_trade(trade_id, {"tp1_partial_done": 1})
                                        print(f"[{profile.profile_name}] T8 ema_scale_runner scale-out: pos {position_id} ({scale_pct:.0f}%) wrong side EMA{ema_fast_p}")
                            else:
                                wrong_side_ema21 = (side == "buy" and last_close < last_ema_slow) or (side == "sell" and last_close > last_ema_slow)
                                if wrong_side_ema21 and current_lots > 0:
                                    adapter.close_position(
                                        ticket=position_id,
                                        symbol=profile.symbol,
                                        volume=current_lots,
                                        position_type=position_type,
                                    )
                                    print(f"[{profile.profile_name}] T8 ema_scale_runner runner closed: pos {position_id} wrong side EMA{ema_slow_p}")
                except Exception as e:
                    print(f"[{profile.profile_name}] T8 ema_scale_runner exit error pos {position_id}: {e}")
                continue

            tp1_done = trade_row.get("tp1_partial_done") or 0
            t8_tp1_pips = t8_tm_overrides.get("tp1_pips", t8_policy.tp1_pips)
            t8_tp1_pct = t8_tm_overrides.get("tp1_close_pct", t8_policy.tp1_close_pct)

            if not tp1_done:
                reached_buy = mid >= entry + float(t8_tp1_pips) * pip
                reached_sell = mid <= entry - float(t8_tp1_pips) * pip
                if (side == "buy" and reached_buy) or (side == "sell" and reached_sell):
                    if isinstance(pos, dict):
                        current_units = pos.get("currentUnits") or 0
                        current_lots = abs(int(current_units)) / 100_000.0
                    else:
                        current_lots = float(getattr(pos, "volume", 0) or 0)
                    close_lots = current_lots * (float(t8_tp1_pct) / 100.0)
                    close_lots = max(MIN_CLOSE_LOTS, min(close_lots, current_lots))
                    if close_lots < 1e-6:
                        close_lots = min(MIN_CLOSE_LOTS, current_lots)
                    position_type = 1 if side == "sell" else 0
                    try:
                        adapter.close_position(
                            ticket=position_id,
                            symbol=profile.symbol,
                            volume=close_lots,
                            position_type=position_type,
                        )
                        print(f"[{profile.profile_name}] T8 TP1 partial close: pos {position_id} {close_lots:.3f} lots ({t8_tp1_pct}%)")
                    except Exception as e:
                        print(f"[{profile.profile_name}] T8 TP1 partial close error pos {position_id}: {e}")
                    _ov_be_pips = t8_tm_overrides.get("be_spread_plus_pips", t8_policy.be_spread_plus_pips)
                    be_offset = current_spread + _ov_be_pips * pip
                    if side == "buy":
                        be_sl = entry + be_offset
                    else:
                        be_sl = entry - be_offset
                    try:
                        adapter.update_position_stop_loss(position_id, profile.symbol, round(be_sl, 3))
                        store.update_trade(trade_id, {"tp1_partial_done": 1, "breakeven_applied": 1, "breakeven_sl_price": round(be_sl, 5)})
                        print(f"[{profile.profile_name}] T8 BE: pos {position_id} SL->{be_sl:.3f}")
                    except Exception as e:
                        print(f"[{profile.profile_name}] T8 BE error pos {position_id}: {e}")

            elif tp1_done:
                try:
                    _ov_trail_period = t8_tm_overrides.get("trail_ema_period", t8_policy.trail_ema_period)
                    m1_df_trail = _tm_m1_df
                    if m1_df_trail is not None and not m1_df_trail.empty and len(m1_df_trail) > _ov_trail_period:
                        from core.indicators import ema as ema_fn
                        m1_close_trail = m1_df_trail["close"].astype(float)
                        trail_ema = ema_fn(m1_close_trail, _ov_trail_period)
                        if not trail_ema.empty and pd.notna(trail_ema.iloc[-1]):
                            ema_val = float(trail_ema.iloc[-1])
                            prev_be_sl = trade_row.get("breakeven_sl_price")
                            if prev_be_sl is not None:
                                prev_be_sl = float(prev_be_sl)
                            if side == "buy":
                                new_trail_sl = ema_val - (1.0 * pip)
                                if prev_be_sl is not None:
                                    new_trail_sl = max(new_trail_sl, prev_be_sl)
                            else:
                                new_trail_sl = ema_val + (1.0 * pip)
                                if prev_be_sl is not None:
                                    new_trail_sl = min(new_trail_sl, prev_be_sl)
                            if prev_be_sl is None or abs(new_trail_sl - prev_be_sl) > pip * 0.1:
                                should_update = False
                                if side == "buy" and (prev_be_sl is None or new_trail_sl > prev_be_sl):
                                    should_update = True
                                elif side == "sell" and (prev_be_sl is None or new_trail_sl < prev_be_sl):
                                    should_update = True
                                if should_update:
                                    adapter.update_position_stop_loss(position_id, profile.symbol, round(new_trail_sl, 3))
                                    store.update_trade(trade_id, {"breakeven_sl_price": round(new_trail_sl, 5)})
                            last_m1_close = float(m1_close_trail.iloc[-1])
                            if side == "buy" and last_m1_close < ema_val:
                                position_type = 0
                                if isinstance(pos, dict):
                                    vol = abs(int(pos.get("currentUnits") or 0)) / 100_000.0
                                else:
                                    vol = float(getattr(pos, "volume", 0) or 0)
                                adapter.close_position(ticket=position_id, symbol=profile.symbol, volume=vol, position_type=position_type)
                                print(f"[{profile.profile_name}] T8 runner closed (BUY close < EMA): pos {position_id}")
                            elif side == "sell" and last_m1_close > ema_val:
                                position_type = 1
                                if isinstance(pos, dict):
                                    vol = abs(int(pos.get("currentUnits") or 0)) / 100_000.0
                                else:
                                    vol = float(getattr(pos, "volume", 0) or 0)
                                adapter.close_position(ticket=position_id, symbol=profile.symbol, volume=vol, position_type=position_type)
                                print(f"[{profile.profile_name}] T8 runner closed (SELL close > EMA): pos {position_id}")
                except Exception as e:
                    print(f"[{profile.profile_name}] T8 trailing EMA error pos {position_id}: {e}")

        # 2c) Trial #9: TP1 + BE + bar-close trail (M1 or M5 depending on exit_strategy) + Kill Switch
        elif t9_policy is not None and entry_type in ("zone_entry", "tiered_pullback"):
            t9_exit_strategy = str(getattr(t9_policy, "exit_strategy", "tp1_be_trail"))
            tp1_done = trade_row.get("tp1_partial_done") or 0
            t9_tp1_pips = float(getattr(t9_policy, "tp1_pips", 6.0))
            t9_tp1_pct = float(getattr(t9_policy, "tp1_close_pct", 80.0))

            if t9_exit_strategy == "tp1_be_m5_trail":
                # --- Phase A: TP1 partial close ---
                if not tp1_done:
                    reached_buy = mid >= entry + t9_tp1_pips * pip
                    reached_sell = mid <= entry - t9_tp1_pips * pip
                    if (side == "buy" and reached_buy) or (side == "sell" and reached_sell):
                        if isinstance(pos, dict):
                            current_units = pos.get("currentUnits") or 0
                            current_lots = abs(int(current_units)) / 100_000.0
                        else:
                            current_lots = float(getattr(pos, "volume", 0) or 0)
                        close_lots = current_lots * (t9_tp1_pct / 100.0)
                        close_lots = max(MIN_CLOSE_LOTS, min(close_lots, current_lots))
                        if close_lots < 1e-6:
                            close_lots = min(MIN_CLOSE_LOTS, current_lots)
                        position_type = 1 if side == "sell" else 0
                        try:
                            adapter.close_position(
                                ticket=position_id,
                                symbol=profile.symbol,
                                volume=close_lots,
                                position_type=position_type,
                            )
                            print(f"[{profile.profile_name}] T9 TP1 partial close: pos {position_id} {close_lots:.3f} lots ({t9_tp1_pct}%)")
                        except Exception as e:
                            print(f"[{profile.profile_name}] T9 TP1 partial close error pos {position_id}: {e}")
                        # --- Phase B: BE on TP1 hit ---
                        _t9_be_pips = float(getattr(t9_policy, "be_spread_plus_pips", 0.5))
                        be_offset = current_spread + _t9_be_pips * pip
                        if side == "buy":
                            be_sl = entry + be_offset
                        else:
                            be_sl = entry - be_offset
                        try:
                            adapter.update_position_stop_loss(position_id, profile.symbol, round(be_sl, 3))
                            store.update_trade(trade_id, {"tp1_partial_done": 1, "breakeven_applied": 1, "breakeven_sl_price": round(be_sl, 5)})
                            print(f"[{profile.profile_name}] T9 BE: pos {position_id} SL->{be_sl:.3f}")
                        except Exception as e:
                            print(f"[{profile.profile_name}] T9 BE error pos {position_id}: {e}")

                elif tp1_done:
                    # --- Phase C: M5 bar-close-only trailing ---
                    try:
                        _t9_m5_trail_period = int(getattr(t9_policy, "trail_m5_ema_period", 20))
                        m5_df_trail = _tm_m5_df
                        if m5_df_trail is not None and not m5_df_trail.empty and len(m5_df_trail) > _t9_m5_trail_period + 1:
                            from core.indicators import ema as ema_fn
                            m5_close_trail = m5_df_trail["close"].astype(float)
                            trail_ema = ema_fn(m5_close_trail, _t9_m5_trail_period)
                            if not trail_ema.empty and pd.notna(trail_ema.iloc[-2]):
                                ema_val = float(trail_ema.iloc[-2])
                                last_m5_close = float(m5_close_trail.iloc[-2])
                                prev_be_sl = trade_row.get("breakeven_sl_price")
                                if prev_be_sl is not None:
                                    prev_be_sl = float(prev_be_sl)
                                if side == "buy":
                                    new_trail_sl = ema_val - (1.0 * pip)
                                    if prev_be_sl is not None:
                                        new_trail_sl = max(new_trail_sl, prev_be_sl)
                                else:
                                    new_trail_sl = ema_val + (1.0 * pip)
                                    if prev_be_sl is not None:
                                        new_trail_sl = min(new_trail_sl, prev_be_sl)
                                if prev_be_sl is None or abs(new_trail_sl - prev_be_sl) > pip * 0.1:
                                    should_update = False
                                    if side == "buy" and (prev_be_sl is None or new_trail_sl > prev_be_sl):
                                        should_update = True
                                    elif side == "sell" and (prev_be_sl is None or new_trail_sl < prev_be_sl):
                                        should_update = True
                                    if should_update:
                                        adapter.update_position_stop_loss(position_id, profile.symbol, round(new_trail_sl, 3))
                                        store.update_trade(trade_id, {"breakeven_sl_price": round(new_trail_sl, 5)})
                                if side == "buy" and last_m5_close < ema_val:
                                    position_type = 0
                                    if isinstance(pos, dict):
                                        vol = abs(int(pos.get("currentUnits") or 0)) / 100_000.0
                                    else:
                                        vol = float(getattr(pos, "volume", 0) or 0)
                                    if vol > 0:
                                        adapter.close_position(ticket=position_id, symbol=profile.symbol, volume=vol, position_type=position_type)
                                        print(f"[{profile.profile_name}] T9 runner closed (BUY M5 bar-close < EMA{_t9_m5_trail_period}): pos {position_id}")
                                elif side == "sell" and last_m5_close > ema_val:
                                    position_type = 1
                                    if isinstance(pos, dict):
                                        vol = abs(int(pos.get("currentUnits") or 0)) / 100_000.0
                                    else:
                                        vol = float(getattr(pos, "volume", 0) or 0)
                                    if vol > 0:
                                        adapter.close_position(ticket=position_id, symbol=profile.symbol, volume=vol, position_type=position_type)
                                        print(f"[{profile.profile_name}] T9 runner closed (SELL M5 bar-close > EMA{_t9_m5_trail_period}): pos {position_id}")
                    except Exception as e:
                        print(f"[{profile.profile_name}] T9 M5 trailing EMA error pos {position_id}: {e}")

            else:
                # Existing tp1_be_trail logic (backward compat)
                if not tp1_done:
                    reached_buy = mid >= entry + t9_tp1_pips * pip
                    reached_sell = mid <= entry - t9_tp1_pips * pip
                    if (side == "buy" and reached_buy) or (side == "sell" and reached_sell):
                        if isinstance(pos, dict):
                            current_units = pos.get("currentUnits") or 0
                            current_lots = abs(int(current_units)) / 100_000.0
                        else:
                            current_lots = float(getattr(pos, "volume", 0) or 0)
                        close_lots = current_lots * (t9_tp1_pct / 100.0)
                        close_lots = max(MIN_CLOSE_LOTS, min(close_lots, current_lots))
                        if close_lots < 1e-6:
                            close_lots = min(MIN_CLOSE_LOTS, current_lots)
                        position_type = 1 if side == "sell" else 0
                        try:
                            adapter.close_position(
                                ticket=position_id,
                                symbol=profile.symbol,
                                volume=close_lots,
                                position_type=position_type,
                            )
                            print(f"[{profile.profile_name}] T9 TP1 partial close: pos {position_id} {close_lots:.3f} lots ({t9_tp1_pct}%)")
                        except Exception as e:
                            print(f"[{profile.profile_name}] T9 TP1 partial close error pos {position_id}: {e}")
                        _t9_be_pips = float(getattr(t9_policy, "be_spread_plus_pips", 2.0))
                        be_offset = current_spread + _t9_be_pips * pip
                        if side == "buy":
                            be_sl = entry + be_offset
                        else:
                            be_sl = entry - be_offset
                        try:
                            adapter.update_position_stop_loss(position_id, profile.symbol, round(be_sl, 3))
                            store.update_trade(trade_id, {"tp1_partial_done": 1, "breakeven_applied": 1, "breakeven_sl_price": round(be_sl, 5)})
                            print(f"[{profile.profile_name}] T9 BE: pos {position_id} SL->{be_sl:.3f}")
                        except Exception as e:
                            print(f"[{profile.profile_name}] T9 BE error pos {position_id}: {e}")

                elif tp1_done:
                    # Bar-close-only trailing on M1 EMA
                    try:
                        _t9_trail_period = int(getattr(t9_policy, "trail_ema_period", 21))
                        m1_df_trail = _tm_m1_df
                        if m1_df_trail is not None and not m1_df_trail.empty and len(m1_df_trail) > _t9_trail_period + 1:
                            from core.indicators import ema as ema_fn
                            m1_close_trail = m1_df_trail["close"].astype(float)
                            trail_ema = ema_fn(m1_close_trail, _t9_trail_period)
                            if not trail_ema.empty and pd.notna(trail_ema.iloc[-2]):
                                ema_val = float(trail_ema.iloc[-2])
                                last_m1_close = float(m1_close_trail.iloc[-2])
                                prev_be_sl = trade_row.get("breakeven_sl_price")
                                if prev_be_sl is not None:
                                    prev_be_sl = float(prev_be_sl)
                                if side == "buy":
                                    new_trail_sl = ema_val - (1.0 * pip)
                                    if prev_be_sl is not None:
                                        new_trail_sl = max(new_trail_sl, prev_be_sl)
                                else:
                                    new_trail_sl = ema_val + (1.0 * pip)
                                    if prev_be_sl is not None:
                                        new_trail_sl = min(new_trail_sl, prev_be_sl)
                                if prev_be_sl is None or abs(new_trail_sl - prev_be_sl) > pip * 0.1:
                                    should_update = False
                                    if side == "buy" and (prev_be_sl is None or new_trail_sl > prev_be_sl):
                                        should_update = True
                                    elif side == "sell" and (prev_be_sl is None or new_trail_sl < prev_be_sl):
                                        should_update = True
                                    if should_update:
                                        adapter.update_position_stop_loss(position_id, profile.symbol, round(new_trail_sl, 3))
                                        store.update_trade(trade_id, {"breakeven_sl_price": round(new_trail_sl, 5)})
                                if side == "buy" and last_m1_close < ema_val:
                                    position_type = 0
                                    if isinstance(pos, dict):
                                        vol = abs(int(pos.get("currentUnits") or 0)) / 100_000.0
                                    else:
                                        vol = float(getattr(pos, "volume", 0) or 0)
                                    if vol > 0:
                                        adapter.close_position(ticket=position_id, symbol=profile.symbol, volume=vol, position_type=position_type)
                                        print(f"[{profile.profile_name}] T9 runner closed (BUY bar-close < EMA{_t9_trail_period}): pos {position_id}")
                                elif side == "sell" and last_m1_close > ema_val:
                                    position_type = 1
                                    if isinstance(pos, dict):
                                        vol = abs(int(pos.get("currentUnits") or 0)) / 100_000.0
                                    else:
                                        vol = float(getattr(pos, "volume", 0) or 0)
                                    if vol > 0:
                                        adapter.close_position(ticket=position_id, symbol=profile.symbol, volume=vol, position_type=position_type)
                                        print(f"[{profile.profile_name}] T9 runner closed (SELL bar-close > EMA{_t9_trail_period}): pos {position_id}")
                    except Exception as e:
                        print(f"[{profile.profile_name}] T9 trailing EMA error pos {position_id}: {e}")

            # Kill Switch (M5 trend + M1-200 EMA): check on every poll, act on completed M1 bar close
            # Fires when M5 is Bull AND M1 bar close < EMA200 (BUY), or M5 is Bear AND M1 bar close > EMA200 (SELL)
            # Closes ALL T9 trades (both tiered pullback and zone entry when kill_switch_zone_entry_action="kill")
            if t9_policy is not None and getattr(t9_policy, "kill_switch_enabled", True):
                try:
                    m1_df_ks = _tm_m1_df
                    if m1_df_ks is not None and not m1_df_ks.empty and len(m1_df_ks) >= 202:
                        from core.indicators import ema as ema_fn
                        m1_close_ks = m1_df_ks["close"].astype(float)
                        ema200 = ema_fn(m1_close_ks, 200)
                        if not ema200.empty and pd.notna(ema200.iloc[-2]):
                            last_ema200 = float(ema200.iloc[-2])
                            last_m1_close_ks = float(m1_close_ks.iloc[-2])
                            # M5 trend check
                            m5_df_ks = _tm_m5_df
                            m5_is_bull_ks = None
                            if m5_df_ks is not None and not m5_df_ks.empty and len(m5_df_ks) >= 22:
                                m5_close_ks = m5_df_ks["close"].astype(float)
                                _ks_fast_p = int(getattr(t9_policy, "m5_trend_ema_fast", 9))
                                _ks_slow_p = int(getattr(t9_policy, "m5_trend_ema_slow", 21))
                                _ks_m5_fast = m5_close_ks.ewm(span=_ks_fast_p, adjust=False).mean()
                                _ks_m5_slow = m5_close_ks.ewm(span=_ks_slow_p, adjust=False).mean()
                                m5_is_bull_ks = float(_ks_m5_fast.iloc[-1]) > float(_ks_m5_slow.iloc[-1])
                            # Extract trade comment
                            comment = ""
                            if isinstance(pos, dict):
                                ce = pos.get("clientExtensions")
                                if isinstance(ce, dict):
                                    comment = str(ce.get("comment") or "")
                                if not comment:
                                    te = pos.get("tradeClientExtensions")
                                    if isinstance(te, dict):
                                        comment = str(te.get("comment") or "")
                            if "kt_cg_trial_9" not in comment:
                                pass  # Not a T9 trade
                            else:
                                is_zone_entry = "zone_entry" in comment
                                import re
                                _tier_m = re.search(r"tier_(\d+)", comment)
                                tier_num = int(_tier_m.group(1)) if _tier_m else None
                                if is_zone_entry:
                                    label = "zone_entry"
                                elif tier_num is not None:
                                    label = f"tier_{tier_num}"
                                else:
                                    label = "unknown"

                                should_kill = False
                                # Kill when M5 trend is Bull and M1 completed bar close < EMA200
                                if side == "buy" and m5_is_bull_ks is True and last_m1_close_ks < last_ema200:
                                    if is_zone_entry:
                                        if str(getattr(t9_policy, "kill_switch_zone_entry_action", "kill")) == "kill":
                                            should_kill = True
                                    else:
                                        should_kill = True  # All tiered pullback tiers killed
                                # Kill when M5 trend is Bear and M1 completed bar close > EMA200
                                elif side == "sell" and m5_is_bull_ks is False and last_m1_close_ks > last_ema200:
                                    if is_zone_entry:
                                        if str(getattr(t9_policy, "kill_switch_zone_entry_action", "kill")) == "kill":
                                            should_kill = True
                                    else:
                                        should_kill = True  # All tiered pullback tiers killed

                                if should_kill:
                                    position_type = 1 if side == "sell" else 0
                                    if isinstance(pos, dict):
                                        vol = abs(int(pos.get("currentUnits") or 0)) / 100_000.0
                                    else:
                                        vol = float(getattr(pos, "volume", 0) or 0)
                                    if vol > 0:
                                        adapter.close_position(ticket=position_id, symbol=profile.symbol, volume=vol, position_type=position_type)
                                        m5_trend_str = "Bull" if m5_is_bull_ks else "Bear"
                                        print(f"[{profile.profile_name}] T9 Kill Switch: closed {label} pos {position_id} (M5={m5_trend_str}, M1 close {last_m1_close_ks:.3f} vs EMA200 {last_ema200:.3f})")
                except Exception as e:
                    print(f"[{profile.profile_name}] T9 Kill Switch error pos {position_id}: {e}")

        # 2d) Phase 3 Integrated trade management
        elif entry_type is not None and str(entry_type).startswith("phase3:"):
            try:
                from core.phase3_integrated_engine import (
                    manage_phase3_exit,
                    load_phase3_sizing_config,
                    V44_COOLDOWN_WIN,
                    V44_COOLDOWN_LOSS,
                )
                phase3_sizing_cfg = load_phase3_sizing_config() or {}
                p3_exit = manage_phase3_exit(
                    adapter=adapter,
                    profile=profile,
                    store=store,
                    tick=tick,
                    trade_row=trade_row,
                    position=pos,
                    data_by_tf={},  # not available in _run_trade_management; engine uses adapter.get_bars internally if needed
                    phase3_state=phase3_state or {},
                    sizing_config=phase3_sizing_cfg,
                )
                if p3_exit.get("action") not in ("none", None):
                    print(f"[{profile.profile_name}] Phase3 exit: pos {position_id} -> {p3_exit.get('action')}: {p3_exit.get('reason', '')}")
                    # Keep Phase 3 runtime caps/cooldowns in sync with recent close outcomes.
                    if isinstance(phase3_state, dict):
                        action = str(p3_exit.get("action") or "")
                        full_close_actions = {"session_end_close", "time_decay_close", "hard_sl", "tp2_full", "tp1_full"}
                        if action in full_close_actions:
                            now_utc = pd.Timestamp.now(tz="UTC")
                            key_date = now_utc.date().isoformat()
                            try:
                                _ts = pd.Timestamp(trade_row.get("timestamp_utc"))
                                if _ts.tzinfo is None:
                                    _ts = _ts.tz_localize("UTC")
                                else:
                                    _ts = _ts.tz_convert("UTC")
                                key_date = _ts.date().isoformat()
                            except Exception:
                                pass
                            closed_pips_est = p3_exit.get("closed_pips_est")
                            if closed_pips_est is None:
                                _stored_pips = trade_row.get("pips")
                                if _stored_pips is not None:
                                    closed_pips_est = _stored_pips
                                else:
                                    closed_pips_est = ((mid - entry) / pip) if side == "buy" else ((entry - mid) / pip)
                            try:
                                pips_eval = float(closed_pips_est)
                            except Exception:
                                pips_eval = 0.0
                            is_loss = True if action == "hard_sl" else (pips_eval < 0.0)
                            _et = str(entry_type or "")
                            entry_session = str(trade_row.get("entry_session") or "").lower()
                            if entry_session not in {"tokyo", "london", "ny"}:
                                if _et.startswith("phase3:v14"):
                                    entry_session = "tokyo"
                                elif _et.startswith("phase3:london_v2"):
                                    entry_session = "london"
                                elif _et.startswith("phase3:v44"):
                                    entry_session = "ny"
                            if entry_session == "tokyo":
                                k = f"session_tokyo_{key_date}"
                                sd = dict(phase3_state.get(k, {}))
                                if is_loss:
                                    sd["consecutive_losses"] = int(sd.get("consecutive_losses", 0)) + 1
                                    sd["win_streak"] = 0
                                    if action == "hard_sl":
                                        sd[f"last_stopout_time_{side}"] = now_utc.isoformat()
                                else:
                                    sd["consecutive_losses"] = 0
                                    sd["wins_closed"] = int(sd.get("wins_closed", 0)) + 1
                                    sd["win_streak"] = int(sd.get("win_streak", 0)) + 1
                                phase3_state[k] = sd
                            elif entry_session == "london":
                                k = f"session_london_{key_date}"
                                sd = dict(phase3_state.get(k, {}))
                                if is_loss:
                                    sd["consecutive_losses"] = int(sd.get("consecutive_losses", 0)) + 1
                                    sd["win_streak"] = 0
                                else:
                                    sd["consecutive_losses"] = 0
                                    sd["wins_closed"] = int(sd.get("wins_closed", 0)) + 1
                                    sd["win_streak"] = int(sd.get("win_streak", 0)) + 1
                                _ldn_cfg = phase3_sizing_cfg.get("london_v2", {})
                                _disable_reset = bool(_ldn_cfg.get("disable_channel_reset_after_exit", False))
                                if not _disable_reset:
                                    _channels = dict(sd.get("channels", {}))
                                    if _et.startswith("phase3:london_v2_arb"):
                                        _channels["A_long" if side == "buy" else "A_short"] = "WAITING_RESET"
                                    elif _et.startswith("phase3:london_v2_d"):
                                        _channels["D_long" if side == "buy" else "D_short"] = "WAITING_RESET"
                                    sd["channels"] = _channels
                                phase3_state[k] = sd
                            elif entry_session == "ny":
                                k = f"session_ny_{key_date}"
                                sd = dict(phase3_state.get(k, {}))
                                sd["consecutive_losses"] = int(sd.get("consecutive_losses", 0)) + 1 if is_loss else 0
                                sd["wins_closed"] = int(sd.get("wins_closed", 0)) + (0 if is_loss else 1)
                                sd["win_streak"] = 0 if is_loss else (int(sd.get("win_streak", 0)) + 1)
                                v44_cfg = phase3_sizing_cfg.get("v44_ny", {})
                                _scope = str(v44_cfg.get("win_streak_scope", "session")).lower()
                                if _scope != "session":
                                    phase3_state["v44_win_streak_global"] = 0 if is_loss else (int(phase3_state.get("v44_win_streak_global", 0)) + 1)
                                cd_win_bars = int(v44_cfg.get("cooldown_win_bars", V44_COOLDOWN_WIN))
                                cd_loss_bars = int(v44_cfg.get("cooldown_loss_bars", V44_COOLDOWN_LOSS))
                                cd_minutes = max(0, (cd_loss_bars if is_loss else cd_win_bars) * 5)
                                if cd_minutes > 0:
                                    sd["cooldown_until"] = (now_utc + pd.Timedelta(minutes=cd_minutes)).isoformat()
                                phase3_state[k] = sd
                            # Immediately write estimated exit data to DB so that V14 session_loss_stop
                            # and V44 daily/weekly loss limits see the closed trade on the next loop poll
                            # (rather than waiting for sync_closed_trades which runs every ~12 loops).
                            try:
                                _p3_exit_px = float(tick.bid if side == "buy" else tick.ask)
                                _p3_size_lots = float(trade_row.get("size_lots") or 0)
                                # Profit estimate: pips × pip_size × units / mid_rate (USDJPY convention)
                                _p3_profit_est = (
                                    (pips_eval * pip * 100_000.0 * _p3_size_lots / mid)
                                    if mid > 0 and _p3_size_lots > 0 else None
                                )
                                store.close_trade(trade_id=trade_id, updates={
                                    "exit_price": _p3_exit_px,
                                    "exit_timestamp_utc": now_utc.isoformat(),
                                    "exit_reason": action,
                                    "pips": pips_eval,
                                    "profit": round(_p3_profit_est, 4) if _p3_profit_est is not None else None,
                                })
                            except Exception as _db_e:
                                print(f"[{profile.profile_name}] Phase3 exit DB immediate-write error: {_db_e}")
            except Exception as e:
                print(f"[{profile.profile_name}] Phase3 exit error pos {position_id}: {e}")

        # 2) TP1 partial close (for non-uncle_parsh policies)
        elif target and getattr(target, "mode", None) == "scaled":
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

    if unmatched_position_ids:
        sample = ",".join(str(pid) for pid in unmatched_position_ids[:5])
        extra = "" if len(unmatched_position_ids) <= 5 else ",..."
        print(
            f"[{profile.profile_name}] trade management skipped {len(unmatched_position_ids)} "
            f"unmatched live position(s) (sample ids: {sample}{extra})"
        )


def _record_loop_error(profile, store, message: str) -> None:
    """Record a loop error into the executions table so it appears in the dashboard execution log."""
    try:
        ts = pd.Timestamp.now(tz="UTC").isoformat()
        store.insert_execution({
            "timestamp_utc": ts,
            "profile": profile.profile_name,
            "symbol": profile.symbol,
            "signal_id": f"loop_error_{ts.replace(':', '-')}",
            "mode": "",
            "attempted": 1,
            "placed": 0,
            "reason": f"loop_error: {str(message)[:500]}",
            "mt5_retcode": None,
            "mt5_order_id": None,
            "mt5_deal_id": None,
        })
    except Exception:
        pass


def _collect_dashboard_positions(
    *,
    profile,
    store,
    tick,
    adapter=None,
    open_positions_snapshot: list | None = None,
) -> list[PositionInfo]:
    """Build dashboard position rows, reusing the loop's open-position snapshot when available."""
    from datetime import datetime, timezone

    pip_size = float(profile.pip_size)
    now_utc = datetime.now(timezone.utc)
    mid = (tick.bid + tick.ask) / 2.0
    positions: list[PositionInfo] = []

    db_by_position_id: dict[int, dict] = {}
    try:
        for row in store.list_open_trades(profile.profile_name):
            d = dict(row)
            pos_id = d.get("mt5_position_id")
            if pos_id is None:
                continue
            try:
                db_by_position_id[int(pos_id)] = d
            except Exception:
                continue
    except Exception:
        pass

    try:
        live_trades = open_positions_snapshot
        if live_trades is None and adapter is not None:
            live_trades = adapter.get_open_positions(profile.symbol)
        for t in live_trades or []:
            if isinstance(t, dict):
                units = float(t.get("currentUnits", 0) or t.get("initialUnits", 0) or 0)
                s = "buy" if units > 0 else "sell"
                size_lots = abs(units) / 100_000.0 if units else 0.0
                entry = float(t.get("price", 0) or 0)
                trade_id = str(t.get("id", ""))
                open_time_str = t.get("openTime")
                sl = None
                tp = None
                try:
                    if t.get("stopLossOrder"):
                        sl = float(t["stopLossOrder"]["price"])
                except Exception:
                    pass
                try:
                    if t.get("takeProfitOrder"):
                        tp = float(t["takeProfitOrder"]["price"])
                except Exception:
                    pass
            else:
                mt5_type = getattr(t, "type", None)
                if mt5_type == 0:
                    s = "buy"
                elif mt5_type == 1:
                    s = "sell"
                else:
                    continue
                size_lots = float(getattr(t, "volume", 0.0) or 0.0)
                entry = float(getattr(t, "price_open", 0.0) or 0.0)
                trade_id = str(getattr(t, "ticket", ""))
                open_time_raw = getattr(t, "time", None)
                open_time_str = str(open_time_raw) if open_time_raw is not None else None
                sl = float(getattr(t, "sl", 0.0) or 0.0) or None
                tp = float(getattr(t, "tp", 0.0) or 0.0) or None

            if not entry or not s:
                continue

            unrealized = (mid - entry) / pip_size if s == "buy" else (entry - mid) / pip_size
            age = 0.0
            if open_time_str:
                try:
                    import pandas as _pd2
                    t0 = _pd2.to_datetime(open_time_str, utc=True)
                    age = (now_utc - t0.to_pydatetime()).total_seconds() / 60.0
                except Exception:
                    pass

            db_row = None
            try:
                db_row = db_by_position_id.get(int(trade_id))
            except Exception:
                db_row = None

            positions.append(PositionInfo(
                trade_id=trade_id,
                side=s,
                entry_price=entry,
                size_lots=round(size_lots, 4),
                entry_type=(db_row.get("entry_type") if db_row else None),
                current_price=mid,
                unrealized_pips=round(unrealized, 1),
                age_minutes=round(age, 1),
                stop_price=(db_row.get("stop_price") if db_row and db_row.get("stop_price") is not None else sl),
                target_price=(db_row.get("target_price") if db_row and db_row.get("target_price") is not None else tp),
                breakeven_applied=bool(db_row.get("breakeven_applied")) if db_row else False,
            ))
    except Exception:
        positions = []

    if positions:
        return positions

    try:
        open_trades = store.list_open_trades(profile.profile_name)
        for row in open_trades:
            d = dict(row)
            entry = float(d.get("entry_price", 0))
            s = str(d.get("side", "")).lower()
            if s == "buy":
                unrealized = (mid - entry) / pip_size
            else:
                unrealized = (entry - mid) / pip_size
            age = 0.0
            ts = d.get("timestamp_utc")
            if ts:
                try:
                    import pandas as _pd
                    t0 = _pd.to_datetime(ts, utc=True)
                    age = (now_utc - t0.to_pydatetime()).total_seconds() / 60.0
                except Exception:
                    pass
            positions.append(PositionInfo(
                trade_id=str(d.get("trade_id", "")),
                side=s,
                entry_price=entry,
                size_lots=(float(d.get("size_lots")) if d.get("size_lots") is not None else None),
                entry_type=d.get("entry_type"),
                current_price=mid,
                unrealized_pips=round(unrealized, 1),
                age_minutes=round(age, 1),
                stop_price=d.get("stop_price"),
                target_price=d.get("target_price"),
                breakeven_applied=bool(d.get("breakeven_applied")),
            ))
    except Exception:
        pass

    return positions


def _collect_dashboard_daily_summary(profile, store) -> DailySummary:
    """Build dashboard daily summary once per loop iteration and reuse it across writes."""
    from datetime import datetime, timezone

    now_utc = datetime.now(timezone.utc)
    daily = DailySummary()
    try:
        date_str = now_utc.strftime("%Y-%m-%d")
        closed_today = store.get_trades_for_date(profile.profile_name, date_str)
        daily.trades_today = len(closed_today)
        for row in closed_today:
            d = dict(row)
            pips = d.get("pips")
            profit = _normalized_profit_for_dashboard_row(d, profile.symbol)
            if pips is not None:
                daily.total_pips += float(pips)
                if float(pips) > 0:
                    daily.wins += 1
                else:
                    daily.losses += 1
            if profit is not None:
                daily.total_profit += float(profit)
        if daily.trades_today > 0:
            daily.win_rate = round(daily.wins / daily.trades_today * 100, 1)
    except Exception:
        pass
    return daily


def _build_and_write_dashboard(
    *,
    profile,
    store,
    log_dir,
    tick,
    data_by_tf: dict,
    mode: str,
    adapter=None,
    policy=None,
    policy_type: str = "",
    tier_state: dict | None = None,
    eval_result: dict | None = None,
    divergence_state: dict | None = None,
    daily_reset_state: dict | None = None,
    exhaustion_result: dict | None = None,
    temp_overrides: dict | None = None,
    daily_level_filter=None,
    ntz_filter=None,
    phase3_state: dict | None = None,
    open_positions_snapshot: list | None = None,
    positions_snapshot: list[PositionInfo] | None = None,
    daily_summary_snapshot: DailySummary | None = None,
) -> None:
    """Assemble and write dashboard state JSON for the current poll cycle."""
    from datetime import datetime, timezone

    try:
        pip_size = float(profile.pip_size)
        now_utc = datetime.now(timezone.utc)
        spread_pips = (tick.ask - tick.bid) / pip_size

        # --- Filter reports (shared with API via core.dashboard_builder) ---
        from core.dashboard_builder import build_dashboard_filters, effective_policy_for_dashboard
        filters = build_dashboard_filters(
            profile=profile,
            tick=tick,
            data_by_tf=data_by_tf,
            policy=policy,
            policy_type=policy_type,
            eval_result=eval_result,
            divergence_state=divergence_state,
            daily_reset_state=daily_reset_state,
            exhaustion_result=exhaustion_result,
            store=store,
            adapter=adapter,
            live_positions_snapshot=open_positions_snapshot,
            temp_overrides=temp_overrides,
            daily_level_filter_snapshot=(daily_level_filter.get_state_snapshot() if daily_level_filter is not None else None),
            ntz_filter_snapshot=(ntz_filter.get_levels_snapshot() if ntz_filter is not None else None),
            phase3_state=phase3_state,
        )

        # --- Context items (use effective policy when Apply Temporary Settings active) ---
        context_items = []
        policy_for_context = effective_policy_for_dashboard(policy, temp_overrides) if policy else policy
        if policy_type == "kt_cg_trial_4":
            context_items = collect_trial_4_context(
                policy_for_context, data_by_tf, tick, tier_state or {}, eval_result, pip_size,
            )
        elif policy_type == "kt_cg_trial_5":
            context_items = collect_trial_5_context(
                policy_for_context, data_by_tf, tick, tier_state or {}, eval_result, pip_size,
                exhaustion_result=exhaustion_result,
                daily_reset_state=daily_reset_state,
            )
        elif policy_type == "kt_cg_trial_6":
            context_items = collect_trial_6_context(
                policy_for_context, data_by_tf, tick, tier_state or {}, eval_result, pip_size,
            )
        elif policy_type == "kt_cg_trial_7":
            context_items = collect_trial_7_context(
                policy_for_context, data_by_tf, tick, tier_state or {}, eval_result, pip_size,
                exhaustion_result=exhaustion_result,
            )
        elif policy_type == "kt_cg_trial_8":
            context_items = collect_trial_7_context(
                policy_for_context, data_by_tf, tick, tier_state or {}, eval_result, pip_size,
                exhaustion_result=exhaustion_result,
            )
        elif policy_type == "kt_cg_trial_9":
            context_items = collect_trial_7_context(
                policy_for_context, data_by_tf, tick, tier_state or {}, eval_result, pip_size,
                exhaustion_result=exhaustion_result,
            )
        elif policy_type == "kt_cg_hybrid":
            context_items = collect_trial_2_context(policy, data_by_tf, tick, pip_size)
        elif policy_type == "kt_cg_counter_trend_pullback":
            context_items = collect_trial_3_context(policy, data_by_tf, tick, pip_size)
        elif policy_type == "uncle_parsh_h1_breakout":
            context_items = collect_uncle_parsh_context(
                policy_for_context, data_by_tf, tick, pip_size,
                eval_result=eval_result,
            )
        elif policy_type == "phase3_integrated":
            from core.dashboard_reporters import collect_phase3_context
            context_items = collect_phase3_context(
                policy_for_context or policy,
                data_by_tf,
                tick,
                eval_result,
                phase3_state or {},
                pip_size,
            )

        candidate_side = None
        candidate_trigger = None
        if isinstance(eval_result, dict):
            raw_side = eval_result.get("candidate_side") or eval_result.get("side")
            if raw_side is None:
                dec = eval_result.get("decision")
                raw_side = getattr(dec, "side", None) if dec is not None else None
            if str(raw_side).lower() in ("buy", "sell"):
                candidate_side = str(raw_side).lower()

            raw_trigger = eval_result.get("candidate_trigger") or eval_result.get("trigger_type")
            if str(raw_trigger) in ("zone_entry", "tiered_pullback"):
                candidate_trigger = str(raw_trigger)

        positions = list(positions_snapshot) if positions_snapshot is not None else _collect_dashboard_positions(
            profile=profile,
            store=store,
            tick=tick,
            adapter=adapter,
            open_positions_snapshot=open_positions_snapshot,
        )

        daily = daily_summary_snapshot if daily_summary_snapshot is not None else _collect_dashboard_daily_summary(profile, store)

        # --- Assemble state ---
        state = DashboardState(
            timestamp_utc=now_utc.isoformat(),
            preset_name=profile.active_preset_name or "",
            mode=mode,
            loop_running=True,
            entry_candidate_side=candidate_side,
            entry_candidate_trigger=candidate_trigger,
            filters=filters,
            context=context_items,
            positions=positions,
            daily_summary=daily,
            bid=tick.bid,
            ask=tick.ask,
            spread_pips=round(spread_pips, 1),
        )
        write_dashboard_state(log_dir, state)
    except Exception as e:
        print(f"[{profile.profile_name}] Dashboard write error: {e}")
        try:
            _record_loop_error(profile, store, f"Dashboard write error: {e}")
        except Exception:
            pass


def _append_trade_open_event(
    log_dir, trade_id: str, side: str, price: float,
    trigger_type: str = "", entry_type: str = "",
    context_snapshot: dict | None = None,
) -> None:
    """Append a trade open event for dashboard display."""
    try:
        event = TradeEvent(
            event_type="open",
            timestamp_utc=pd.Timestamp.now(tz="UTC").isoformat(),
            trade_id=trade_id,
            side=side,
            entry_type=entry_type,
            price=price,
            trigger_type=trigger_type,
            context_snapshot=context_snapshot or {},
        )
        append_trade_event(log_dir, event)
    except Exception:
        pass


def _event_trigger_label(trigger_type: str | None, decision_reason: str | None) -> str:
    """Create a user-friendly trigger label for dashboard trade events."""
    base = (trigger_type or "").strip()
    if base != "tiered_pullback":
        return base
    reason = (decision_reason or "").strip()
    m = re.search(r"tier_(\d+)", reason)
    if m:
        return f"tiered_pullback_ema{m.group(1)}"
    return base


def _append_phase3_minute_diagnostics(
    log_dir: Path,
    profile: Any,
    store: Any,
    tick: Any,
    data_by_tf: dict,
    policy: Any,
    exec_result: dict,
    phase3_state: dict,
    m1_bar_time: str,
    is_new: bool,
    mode: str,
) -> None:
    """Log one line per closed M1 bar: session/strategy, is_new, placed, filter blocks, and decision reason."""
    try:
        from core.phase3_integrated_engine import classify_session, load_phase3_sizing_config
        from core.dashboard_builder import build_dashboard_filters
        cfg = load_phase3_sizing_config()
        bar_ts = pd.Timestamp(m1_bar_time)
        if bar_ts.tzinfo is None:
            bar_ts = bar_ts.tz_localize("UTC")
        else:
            bar_ts = bar_ts.tz_convert("UTC")
        session = classify_session(bar_ts.to_pydatetime(), cfg)
        strategy_tag = exec_result.get("strategy_tag")
        filters = build_dashboard_filters(
            profile=profile,
            tick=tick,
            data_by_tf=data_by_tf,
            policy=policy,
            policy_type="phase3_integrated",
            eval_result=exec_result,
            phase3_state=phase3_state,
        )
        blocking = [f for f in filters if not getattr(f, "is_clear", True)]
        blocking_count = len(blocking)
        blocking_details = [
            f"{getattr(f, 'display_name', getattr(f, 'filter_id', 'unknown_filter'))}:{getattr(f, 'block_reason', '') or getattr(f, 'current_value', '')}"
            for f in blocking
        ]
        blocking_ids = [
            str(getattr(f, "filter_id", "") or "").strip()
            for f in blocking
            if str(getattr(f, "filter_id", "") or "").strip()
        ]
        dec = exec_result.get("decision")
        reason = getattr(dec, "reason", "") if dec else ""
        placed = getattr(dec, "placed", False) if dec else False
        ts = pd.Timestamp.now(tz="UTC").isoformat()
        line = (
            f"{ts}\tbar={m1_bar_time}\tis_new={int(is_new)}\tplaced={int(placed)}\t"
            f"session={session or 'none'}\tstrategy={strategy_tag or ''}\t"
            f"blocking={blocking_count}\t"
            f"filters=[{'; '.join(blocking_details)}]\treason={reason!r}\n"
        )
        diag_path = log_dir / "phase3_minute_diagnostics.log"
        with open(diag_path, "a", encoding="utf-8") as f:
            f.write(line)

        # Persist one row per closed M1 bar so UI can load many days and group by blockers.
        # Store the blocking filter_ids as an encoded suffix in reason for fast grouping without schema changes.
        try:
            signal_id = f"eval:phase3_integrated:{getattr(policy, 'id', 'policy')}:{m1_bar_time}"
            rule_id = signal_id
            blocks_csv = ",".join(blocking_ids)
            reason_store = str(reason or "")
            if blocks_csv:
                reason_store = f"{reason_store} | blocks={blocks_csv}"
            store.insert_execution(
                {
                    "timestamp_utc": ts,
                    "profile": getattr(profile, "profile_name", "") or getattr(profile, "name", ""),
                    "symbol": getattr(profile, "symbol", ""),
                    "signal_id": signal_id,
                    "rule_id": rule_id,
                    "mode": str(mode or ""),
                    "attempted": int(getattr(dec, "attempted", False) if dec else 0),
                    "placed": int(bool(placed)),
                    "reason": reason_store,
                    "mt5_retcode": getattr(dec, "order_retcode", None) if dec else None,
                    "mt5_order_id": getattr(dec, "order_id", None) if dec else None,
                    "mt5_deal_id": getattr(dec, "deal_id", None) if dec else None,
                }
            )
        except Exception:
            pass
    except Exception as e:
        try:
            diag_path = log_dir / "phase3_minute_diagnostics.log"
            with open(diag_path, "a", encoding="utf-8") as f:
                f.write(f"{pd.Timestamp.now(tz='UTC').isoformat()}\tbar=?\tis_new=?\tplaced=?\tblocking=?\tfilters=[]\treason=diagnostics_error:{e!r}\n")
        except Exception:
            pass


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

    # Loop log ring buffer — captures last 200 log entries for dashboard display
    _loop_log: collections.deque = collections.deque(maxlen=200)
    _loop_log_path = log_dir / "loop_log.json"

    def _log(msg: str, level: str = "INFO") -> None:
        """Append a timestamped message to the loop log ring buffer and print to console."""
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        entry = {"ts": ts, "level": level, "msg": msg}
        _loop_log.append(entry)
        print(f"[{profile.profile_name}] [{level}] {msg}")
        # Persist to disk so dashboard can read it
        try:
            _loop_log_path.write_text(json.dumps(list(_loop_log), ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    adapter = get_adapter(profile)
    adapter.initialize()
    try:
        adapter.ensure_symbol(profile.symbol)
        _log("Loop started, symbol ensured")

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
        has_kt_cg_trial_6 = any(
            getattr(p, "type", None) == "kt_cg_trial_6" and getattr(p, "enabled", True)
            for p in profile.execution.policies
        )
        has_kt_cg_trial_7 = any(
            getattr(p, "type", None) == "kt_cg_trial_7" and getattr(p, "enabled", True)
            for p in profile.execution.policies
        )
        has_uncle_parsh = any(
            getattr(p, "type", None) == "uncle_parsh_h1_breakout" and getattr(p, "enabled", True)
            for p in profile.execution.policies
        )
        has_kt_cg_trial_8 = any(
            getattr(p, "type", None) == "kt_cg_trial_8" and getattr(p, "enabled", True)
            for p in profile.execution.policies
        )
        has_kt_cg_trial_9 = any(
            getattr(p, "type", None) == "kt_cg_trial_9" and getattr(p, "enabled", True)
            for p in profile.execution.policies
        )
        has_phase3_integrated = any(
            getattr(p, "type", None) == "phase3_integrated" and getattr(p, "enabled", True)
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
        last_sync_time = 0.0
        _SYNC_INTERVAL_SECONDS = 60.0
        # Keep broker fetch retries light; OANDA adapter already has internal retries.
        _MAX_FETCH_RETRIES = 1
        _last_oanda_tradeid_backfill_time: float = 0.0
        _OANDA_TRADEID_BACKFILL_INTERVAL_S: float = 30.0
        _OANDA_TRADEID_BACKFILL_MAX_PER_CYCLE: int = 5
        _OANDA_TRADEID_BACKFILL_BUDGET_S: float = 2.0
        _OANDA_TRADEID_RETRY_COOLDOWN_S: float = 300.0
        _oanda_backfill_last_attempt_by_trade: dict[str, float] = {}

        # Candle cache: {tf: (timestamp_fetched, DataFrame)}
        # TTLs in seconds: M1=28 so new bar is seen soon after minute close (fast poll); M3=175, M5=295, ...
        _candle_cache: dict[str, tuple[float, pd.DataFrame]] = {}
        _CANDLE_TTL: dict[str, float] = {
            "M1": 28.0, "M3": 175.0, "M5": 295.0, "M15": 895.0,
            "H1": 3595.0, "H4": 14395.0, "D": 300.0,
            "W": 3600.0, "MN": 3600.0,
        }
        # OANDA candles can lag; keep M1/M5 refresh frequent so loop sees the latest candle time quickly.
        if getattr(profile, "broker_type", None) == "oanda":
            _CANDLE_TTL["M1"] = 2.0
            _CANDLE_TTL["M5"] = 55.0  # Refresh M5 trend ~once/minute instead of once/5min

        # Logging state for Trial #5 — reduce log spam
        _last_exhaustion_zone: str | None = None
        _last_summary_time: float = 0.0
        _SUMMARY_INTERVAL: float = 30.0  # Print periodic summary every 30s

        # Shared daily H/L state for Trial #7 and Trial #8 (today_high/low, prev_day_high/low)
        trial7_daily_state: dict = {
            "date_utc": None,
            "today_open": None,
            "today_high": None,
            "today_low": None,
            "prev_day_high": None,
            "prev_day_low": None,
        }
        try:
            _init_state_data = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
            _saved_daily = _init_state_data.get("trial7_daily_state")
            if isinstance(_saved_daily, dict) and _saved_daily.get("date_utc") is not None:
                now_date_init = pd.Timestamp.now(tz="UTC").date().isoformat()
                if _saved_daily["date_utc"] == now_date_init:
                    trial7_daily_state.update(_saved_daily)
                    print(f"[{profile.profile_name}] Restored trial7_daily_state from disk (date={now_date_init})")
        except Exception as _e:
            print(f"[{profile.profile_name}] Could not restore trial7_daily_state: {_e}")

        # Daily Level Filter for Trial #8 / Trial #9 (one instance per profile, state persists across iterations)
        daily_level_filter = None
        if has_kt_cg_trial_8 or has_kt_cg_trial_9:
            from core.daily_level_filter import DailyLevelFilter
            # Prefer a Trial #9 policy if present; otherwise fall back to Trial #8.
            t8_policy = next((p for p in profile.execution.policies if getattr(p, "type", None) == "kt_cg_trial_8" and getattr(p, "enabled", True)), None)
            t9_policy = next((p for p in profile.execution.policies if getattr(p, "type", None) == "kt_cg_trial_9" and getattr(p, "enabled", True)), None)
            _dl_policy = t9_policy or t8_policy
            if _dl_policy is not None:
                daily_level_filter = DailyLevelFilter(
                    enabled=bool(getattr(_dl_policy, "use_daily_level_filter", False)),
                    buffer_pips=float(getattr(_dl_policy, "daily_level_buffer_pips", 3.0)),
                    breakout_candles_required=int(getattr(_dl_policy, "daily_level_breakout_candles_required", 2)),
                    pip_size=float(profile.pip_size),
                )

        # No-Trade Zone filter for Trial #9 (not used now that Trial #9 is a copy of Trial #8, but kept for future experiments)
        ntz_filter = None

        # Phase 3 Integrated state (pivots, session counters, open trade tracking)
        phase3_state: dict = {}
        if has_phase3_integrated:
            try:
                _p3_init = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
                phase3_state = _p3_init.get("phase3_state", {})
            except Exception:
                phase3_state = {}

        trades_df_cache: pd.DataFrame | None = None

        def _invalidate_trades_df_cache() -> None:
            nonlocal trades_df_cache
            trades_df_cache = None

        def _get_trades_df_cached() -> pd.DataFrame:
            nonlocal trades_df_cache
            if trades_df_cache is None:
                trades_df_cache = store.read_trades_df(profile.profile_name)
            return trades_df_cache

        def _get_bars_cached(symbol: str, tf: str, count: int, *, include_incomplete: bool = False, force_refresh: bool = False) -> pd.DataFrame:
            """Fetch bars with caching to reduce API calls at fast poll rates."""
            now = time.time()
            if not force_refresh:
                cached = _candle_cache.get(tf)
                ttl = _CANDLE_TTL.get(tf, 55.0)
                if cached is not None:
                    cached_time, cached_df = cached
                    if now - cached_time < ttl:
                        return cached_df
            try:
                df = adapter.get_bars(symbol, tf, count, include_incomplete=include_incomplete)
            except TypeError:
                # Backward compat: adapters that don't support include_incomplete
                df = adapter.get_bars(symbol, tf, count)
            _candle_cache[tf] = (now, df)
            return df

        while True:
          try:
            loop_count += 1
            _invalidate_trades_df_cache()
            iter_started = time.perf_counter()
            phase_times: dict[str, float] = {}

            def _phase_done(name: str, started_at: float) -> None:
                phase_times[name] = phase_times.get(name, 0.0) + (time.perf_counter() - started_at)

            if loop_count % 20 == 0:
                _log(f"heartbeat loop={loop_count}")

            # Periodic trade sync (detect externally closed trades; import from broker history)
            _now_sync = time.monotonic()
            if (_now_sync - last_sync_time) >= _SYNC_INTERVAL_SECONDS:
                _sync_started = time.perf_counter()
                try:
                    synced = sync_closed_trades(profile, store, log_dir=log_dir)
                    if synced > 0:
                        print(f"[{profile.profile_name}] synced {synced} externally closed trade(s)")
                    # Import from broker history (MT5 only — OANDA places all trades via bot, already in DB)
                    if getattr(profile, "broker_type", None) != "oanda":
                        imported = import_mt5_history(profile, store, days_back=90)
                        if imported > 0:
                            print(f"[{profile.profile_name}] imported {imported} trade(s) from broker history")
                except Exception as e:
                    print(f"[{profile.profile_name}] sync error: {e}")
                    try:
                        _record_loop_error(profile, store, f"sync error: {e}")
                    except Exception:
                        pass
                last_sync_time = _now_sync
                _phase_done("sync", _sync_started)

            # OANDA-only: backfill broker trade IDs (stored in trades.mt5_position_id) for older rows.
            # This is critical for correct dashboard classification + zone/tier cap accounting.
            _backfill_started = time.perf_counter()
            try:
                if getattr(profile, "broker_type", None) == "oanda":
                    _now_bf = time.time()
                    if _now_bf - _last_oanda_tradeid_backfill_time >= _OANDA_TRADEID_BACKFILL_INTERVAL_S:
                        missing = store.get_trades_missing_position_id(profile.profile_name)
                        updated = 0
                        attempted = 0
                        _backfill_deadline = time.perf_counter() + _OANDA_TRADEID_BACKFILL_BUDGET_S
                        for r in missing:
                            d = dict(r)
                            if d.get("exit_price") is not None:
                                continue  # closed trade; skip
                            trade_id = str(d.get("trade_id") or "")
                            if not trade_id:
                                continue
                            last_attempt = _oanda_backfill_last_attempt_by_trade.get(trade_id, 0.0)
                            if (_now_bf - last_attempt) < _OANDA_TRADEID_RETRY_COOLDOWN_S:
                                continue
                            if attempted >= _OANDA_TRADEID_BACKFILL_MAX_PER_CYCLE or time.perf_counter() >= _backfill_deadline:
                                break
                            attempted += 1
                            _oanda_backfill_last_attempt_by_trade[trade_id] = _now_bf
                            # If mt5_deal_id already contains the OANDA tradeID (newer behavior), use it directly.
                            deal_id = d.get("mt5_deal_id")
                            order_id = d.get("mt5_order_id")
                            pos_id = None
                            if deal_id is not None:
                                try:
                                    pos_id = adapter.get_position_id_from_deal(int(deal_id))
                                except Exception:
                                    pos_id = None
                            if pos_id is None and order_id is not None:
                                try:
                                    pos_id = adapter.get_position_id_from_order(int(order_id))
                                except Exception:
                                    pos_id = None
                            if pos_id is not None:
                                try:
                                    store.update_trade(trade_id, {"mt5_position_id": int(pos_id)})
                                    updated += 1
                                    _oanda_backfill_last_attempt_by_trade.pop(trade_id, None)
                                except Exception:
                                    pass
                        if updated:
                            _log(f"OANDA tradeID backfill updated {updated} open trade(s)")
                        _last_oanda_tradeid_backfill_time = _now_bf
            except Exception:
                pass
            _phase_done("tradeid_backfill", _backfill_started)

            state = load_state(state_path)
            if state.kill_switch:
                mode = "DISARMED"
            else:
                mode = state.mode

            # Fetch market data with retries (502/503/timeouts); avoid exiting on transient broker errors
            data_by_tf = None
            tick = None
            _fetch_started = time.perf_counter()
            _log(f"Fetching market data (attempt loop={loop_count})")
            for _fetch_attempt in range(_MAX_FETCH_RETRIES):
                try:
                    data_by_tf = {
                        "H4": _get_bars_cached(profile.symbol, "H4", 800),
                        "M15": _get_bars_cached(profile.symbol, "M15", 2000),
                        # OANDA: request fewer bars (faster) and include forming candle for fresher timestamps.
                        "M1": _get_bars_cached(
                            profile.symbol,
                            "M1",
                            800 if getattr(profile, "broker_type", None) == "oanda" else 3000,
                            include_incomplete=(getattr(profile, "broker_type", None) == "oanda"),
                            force_refresh=(getattr(profile, "broker_type", None) == "oanda"),
                        ),
                    }
                    # Fetch M5 data when needed by ema_pullback, kt_cg_ctp, kt_cg_hybrid, M5 confirmed-cross, or uncle_parsh.
                    if has_ema_pullback or has_m5_confirmed_cross or has_kt_cg_ctp or has_kt_cg_hybrid or has_kt_cg_trial_7 or has_kt_cg_trial_8 or has_kt_cg_trial_9 or has_uncle_parsh or has_phase3_integrated:
                        data_by_tf["M5"] = _get_bars_cached(profile.symbol, "M5", 2000)
                    # Fetch M3 data when needed by kt_cg_trial_4/5/6 (trend detection).
                    if has_kt_cg_trial_4 or has_kt_cg_trial_5 or has_kt_cg_trial_6:
                        data_by_tf["M3"] = _get_bars_cached(profile.symbol, "M3", 3000)
                    # Fetch D1 data when needed by daily H/L filter (Trial #4/5/8/9).
                    if has_kt_cg_trial_4 or has_kt_cg_trial_5 or has_kt_cg_trial_8 or has_kt_cg_trial_9 or has_phase3_integrated:
                        data_by_tf["D"] = _get_bars_cached(profile.symbol, "D", 5)
                    # Fetch W and MN data only when a T9 NTZ filter is actually enabled.
                    if has_kt_cg_trial_9 and any(
                        getattr(p, "type", None) == "kt_cg_trial_9"
                        and getattr(p, "enabled", True)
                        and getattr(p, "ntz_enabled", False)
                        for p in profile.execution.policies
                    ):
                        try:
                            data_by_tf["W"] = _get_bars_cached(profile.symbol, "W", 2, include_incomplete=True)
                        except Exception as _w_err:
                            _log(f"W candle fetch failed: {_w_err}", "WARN")
                        try:
                            data_by_tf["MN"] = _get_bars_cached(profile.symbol, "MN", 2, include_incomplete=True)
                        except Exception as _mn_err:
                            _log(f"MN candle fetch failed: {_mn_err}", "WARN")

                    # Fetch H1 data when needed by Uncle Parsh or Phase3 integrated NY branch.
                    if has_uncle_parsh or has_phase3_integrated:
                        data_by_tf["H1"] = _get_bars_cached(profile.symbol, "H1", 200)
                    # Tick always fetched fresh for live price detection
                    tick = adapter.get_tick(profile.symbol)
                    break
                except Exception as _fetch_err:
                    # Fast-fail on broker fetch errors so the loop and dashboard never stall for minutes.
                    print(f"[{profile.profile_name}] broker fetch error: {_fetch_err}")
                    try:
                        _record_loop_error(profile, store, f"broker temporarily unavailable: {_fetch_err}")
                    except Exception:
                        pass
                    # Let the outer loop (and poll_sec) control pacing; don't add long sleeps here.
                    break
            _phase_done("fetch_market_data", _fetch_started)
            if data_by_tf is None or tick is None:
                continue

            # Trade management: breakeven and TP1 partial close (only for positions we opened).
            # Fetch open positions once per loop so all caps/management share the same snapshot.
            _snapshot_started = time.perf_counter()
            try:
                open_positions_snapshot = adapter.get_open_positions(profile.symbol)
            except Exception:
                open_positions_snapshot = None
            dashboard_positions_snapshot = _collect_dashboard_positions(
                profile=profile,
                store=store,
                tick=tick,
                adapter=adapter,
                open_positions_snapshot=open_positions_snapshot,
            )
            dashboard_daily_snapshot = _collect_dashboard_daily_summary(profile, store)
            _phase_done("snapshot_build", _snapshot_started)
            _tm_started = time.perf_counter()
            _run_trade_management(
                profile,
                adapter,
                store,
                tick,
                phase3_state=phase3_state if has_phase3_integrated else None,
                open_positions=open_positions_snapshot,
            )
            _invalidate_trades_df_cache()
            _phase_done("trade_management", _tm_started)
            if has_phase3_integrated:
                _p3_state_started = time.perf_counter()
                try:
                    current_state_data = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
                    current_state_data["phase3_state"] = phase3_state
                    state_path.write_text(json.dumps(current_state_data, indent=2) + "\n", encoding="utf-8")
                except Exception as e:
                    print(f"[{profile.profile_name}] Failed to persist phase3_state after trade management: {e}")
                _phase_done("persist_phase3_state", _p3_state_started)

            # Refresh the dashboard immediately once we have current prices and positions.
            # Policy-specific writes later in the loop can overwrite this with richer filters/context.
            _dashboard_seed_started = time.perf_counter()
            _build_and_write_dashboard(
                profile=profile,
                store=store,
                log_dir=log_dir,
                tick=tick,
                data_by_tf=data_by_tf,
                mode=mode,
                adapter=adapter,
                phase3_state=phase3_state if has_phase3_integrated else None,
                open_positions_snapshot=open_positions_snapshot,
                positions_snapshot=dashboard_positions_snapshot,
                daily_summary_snapshot=dashboard_daily_snapshot,
            )
            _phase_done("dashboard_seed", _dashboard_seed_started)

            m1_df = data_by_tf["M1"]
            m1_last_time = pd.to_datetime(m1_df["time"].iloc[-1], utc=True).isoformat()
            now_utc = pd.Timestamp.now(tz="UTC")
            last_bar_dt = pd.to_datetime(m1_last_time, utc=True, errors="coerce")
            bar_lag_sec: float | None = None
            # Keep for diagnostics (dashboard can show how far behind candle timestamps are).
            if pd.notna(last_bar_dt):
                try:
                    bar_lag_sec = max(0.0, float((now_utc - last_bar_dt).total_seconds()))
                except Exception:
                    bar_lag_sec = None
            if last_seen_m1_time is None:
                last_seen_m1_time = m1_last_time

            is_new = m1_last_time != last_seen_m1_time
            used_clock_fallback = False
            # Clock-based fallback: if broker hasn't delivered the new bar yet (e.g. API delay),
            # treat current UTC minute as new once we're a few seconds in, so Trial 7/8 still runs.
            if not is_new and (has_kt_cg_trial_7 or has_kt_cg_trial_8 or has_kt_cg_trial_9):
                try:
                    current_minute_iso = now_utc.floor("min").isoformat()
                    last_bar_minute_iso = last_bar_dt.floor("min").isoformat()
                    if now_utc.second >= 5 and current_minute_iso > last_bar_minute_iso and last_seen_m1_time != current_minute_iso:
                        is_new = True
                        used_clock_fallback = True
                        last_seen_m1_time = current_minute_iso
                except Exception:
                    pass

            # Update shared daily H/L state for Trial #7 / Trial #8 when we have a new M1 bar
            if (has_kt_cg_trial_7 or has_kt_cg_trial_8 or has_kt_cg_trial_9) and is_new:
                now_date = pd.Timestamp.now(tz="UTC").date().isoformat()
                mid_tick = (tick.bid + tick.ask) / 2.0
                date_changed = trial7_daily_state.get("date_utc") != now_date
                if date_changed:
                    trial7_daily_state["date_utc"] = now_date
                    trial7_daily_state["today_open"] = float(mid_tick)
                    trial7_daily_state["today_high"] = float(mid_tick)
                    trial7_daily_state["today_low"] = float(mid_tick)
                else:
                    prev_hi = trial7_daily_state.get("today_high")
                    prev_lo = trial7_daily_state.get("today_low")
                    trial7_daily_state["today_high"] = float(mid_tick) if prev_hi is None else max(float(prev_hi), float(mid_tick))
                    trial7_daily_state["today_low"] = float(mid_tick) if prev_lo is None else min(float(prev_lo), float(mid_tick))
                d_df_live = data_by_tf.get("D")
                if d_df_live is not None and not d_df_live.empty and len(d_df_live) >= 2:
                    try:
                        d_local = d_df_live.copy()
                        d_local["time"] = pd.to_datetime(d_local["time"], utc=True, errors="coerce")
                        d_local = d_local.dropna(subset=["time"]).sort_values("time")
                        if len(d_local) >= 2:
                            # Log all D candles so we can debug PDH/PDL selection
                            for _di in range(len(d_local)):
                                _dr = d_local.iloc[_di]
                                _log(f"D1 candle [{_di}]: time={_dr['time']} H={float(_dr['high']):.3f} L={float(_dr['low']):.3f}")
                            if str(d_local.iloc[-1]["time"].date().isoformat()) == now_date:
                                prev_row = d_local.iloc[-2]
                                _log(f"D1 prev_row: iloc[-2] (last candle is today={now_date})")
                            else:
                                prev_row = d_local.iloc[-1]
                                _log(f"D1 prev_row: iloc[-1] (last candle is NOT today={now_date}, last={d_local.iloc[-1]['time'].date().isoformat()})")
                            _log(f"D1 selected: H={float(prev_row['high']):.3f} L={float(prev_row['low']):.3f}")
                            trial7_daily_state["prev_day_high"] = float(prev_row["high"])
                            trial7_daily_state["prev_day_low"] = float(prev_row["low"])
                    except Exception:
                        pass
                if date_changed and daily_level_filter is not None:
                    daily_level_filter.reset(trial7_daily_state.get("prev_day_high"), trial7_daily_state.get("prev_day_low"))
                try:
                    _ds_data = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
                    _ds_data["trial7_daily_state"] = trial7_daily_state
                    state_path.write_text(json.dumps(_ds_data, indent=2) + "\n", encoding="utf-8")
                except Exception:
                    pass

            # OANDA: run Trial 7/8/9 intrabar (tick-driven), not only on new M1 candles.
            intrabar_t78 = (
                getattr(profile, "broker_type", None) == "oanda"
                and (has_kt_cg_trial_7 or has_kt_cg_trial_8 or has_kt_cg_trial_9)
            )

            # Log when we skip Trial 7/8 due to no new M1 bar (so execution log shows why no evaluation)
            if (has_kt_cg_trial_7 or has_kt_cg_trial_8 or has_kt_cg_trial_9) and not is_new and not args.once and not intrabar_t78:
                try:
                    store.insert_execution(
                        {
                            "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                            "profile": profile.profile_name,
                            "symbol": profile.symbol,
                            "signal_id": f"loop:skip:{m1_last_time}",
                            "mode": mode,
                            "attempted": 0,
                            "placed": 0,
                            "reason": f"loop: is_new=False, skip Trial 7/8 (no new M1 bar; last bar={m1_last_time})",
                            "mt5_retcode": None,
                            "mt5_order_id": None,
                            "mt5_deal_id": None,
                        }
                    )
                except Exception:
                    pass

            # Trial 7/8/9 intrabar execution (OANDA): evaluate every poll so zone-entry can fire within seconds.
            if intrabar_t78 and not args.once:
                _intrabar_t78_started = time.perf_counter()
                t7_mkt = None
                spread_pips = (tick.ask - tick.bid) / float(profile.pip_size)
                t7_mkt = MarketContext(spread_pips=float(spread_pips), alignment_score=0)
                trades_df = _get_trades_df_cached()
                tier_state: dict[int, bool] = {}
                try:
                    state_data = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
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
                except Exception:
                    tier_state = {}

                for pol in profile.execution.policies:
                    pol_type = getattr(pol, "type", None)
                    if not getattr(pol, "enabled", True) or pol_type not in ("kt_cg_trial_7", "kt_cg_trial_8", "kt_cg_trial_9"):
                        continue
                    # Use poll time for bar_time_utc so multiple evaluations within a minute are distinct.
                    bar_time_utc = now_utc.isoformat()
                    if pol_type == "kt_cg_trial_9":
                        exec_result = execute_kt_cg_trial_9_policy_demo_only(
                            adapter=adapter,
                            profile=profile,
                            log_dir=log_dir,
                            policy=pol,
                            context=t7_mkt,
                            data_by_tf=data_by_tf,
                            tick=tick,
                            trades_df=trades_df,
                            mode=mode,
                            bar_time_utc=bar_time_utc,
                            tier_state=tier_state,
                            store=store,
                            daily_level_filter=daily_level_filter,
                            daily_state=trial7_daily_state,
                            open_positions=open_positions_snapshot,
                        )
                    elif pol_type == "kt_cg_trial_8":
                        exec_result = execute_kt_cg_trial_8_policy_demo_only(
                            adapter=adapter,
                            profile=profile,
                            log_dir=log_dir,
                            policy=pol,
                            context=t7_mkt,
                            data_by_tf=data_by_tf,
                            tick=tick,
                            trades_df=trades_df,
                            mode=mode,
                            bar_time_utc=bar_time_utc,
                            tier_state=tier_state,
                            store=store,
                            daily_level_filter=daily_level_filter,
                            daily_state=trial7_daily_state,
                            open_positions=open_positions_snapshot,
                        )
                    else:
                        exec_result = execute_kt_cg_trial_7_policy_demo_only(
                            adapter=adapter,
                            profile=profile,
                            log_dir=log_dir,
                            policy=pol,
                            context=t7_mkt,
                            data_by_tf=data_by_tf,
                            tick=tick,
                            trades_df=trades_df,
                            mode=mode,
                            bar_time_utc=bar_time_utc,
                            tier_state=tier_state,
                            store=store,
                            daily_level_filter=None,
                            daily_state=None,
                            open_positions=open_positions_snapshot,
                        )

                    dec = exec_result["decision"]
                    tier_updates = exec_result.get("tier_updates", {})
                    t7_trigger_type = exec_result.get("trigger_type")

                    if tier_updates:
                        try:
                            current_state_data = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
                            tf_dict = current_state_data.get("tier_fired", {})
                            if not isinstance(tf_dict, dict):
                                tf_dict = {}
                            for tier, new_state in tier_updates.items():
                                tf_dict[str(tier)] = new_state
                                tier_state[tier] = new_state
                            current_state_data["tier_fired"] = tf_dict
                            state_path.write_text(json.dumps(current_state_data, indent=2) + "\n", encoding="utf-8")
                        except Exception as e:
                            print(f"[{profile.profile_name}] Failed to persist Trial #7 tier state: {e}")

                    if dec.attempted:
                        if dec.placed:
                            side = dec.side or "buy"
                            entry_price = tick.ask if side == "buy" else tick.bid
                            pip = float(profile.pip_size)
                            if pol_type == "kt_cg_trial_9":
                                sl_price = None
                                tp_price = None
                            else:
                                sl_pips = getattr(pol, "sl_pips", None)
                                if sl_pips is None:
                                    sl_pips = float(get_effective_risk(profile).min_stop_pips)
                                if side == "buy":
                                    tp_price = entry_price + pol.tp_pips * pip
                                    sl_price = entry_price - sl_pips * pip
                                else:
                                    tp_price = entry_price - pol.tp_pips * pip
                                    sl_price = entry_price + sl_pips * pip
                                if pol_type == "kt_cg_trial_8" and getattr(pol, "trail_after_tp1", False):
                                    tp_price = None
                            print(f"[{profile.profile_name}] TRADE PLACED: {pol_type}:{pol.id} | side={side} | entry={entry_price:.3f} | {dec.reason}")
                            _insert_trade_for_policy(
                                profile=profile,
                                adapter=adapter,
                                store=store,
                                policy_type=pol_type,
                                policy_id=pol.id,
                                side=side,
                                entry_price=entry_price,
                                dec=dec,
                                stop_price=sl_price,
                                target_price=tp_price,
                                entry_type=t7_trigger_type,
                            )
                            _invalidate_trades_df_cache()
                            _append_trade_open_event(
                                log_dir, f"{pol_type}:{pol.id}:{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d%H%M%S')}",
                                side, entry_price,
                                trigger_type=_event_trigger_label(t7_trigger_type, dec.reason),
                                entry_type=t7_trigger_type or "",
                            )
                        else:
                            print(f"[{profile.profile_name}] {pol_type} {pol.id} mode={mode} -> {dec.reason}")

                    _build_and_write_dashboard(
                        profile=profile, store=store, log_dir=log_dir, tick=tick,
                        data_by_tf=data_by_tf, mode=mode, adapter=adapter, policy=pol,
                        policy_type=pol_type, tier_state=tier_state,
                        eval_result=exec_result,
                        exhaustion_result=exec_result.get("exhaustion_result"),
                        daily_level_filter=daily_level_filter if pol_type in ("kt_cg_trial_8", "kt_cg_trial_9") else None,
                        ntz_filter=None,
                        open_positions_snapshot=open_positions_snapshot,
                        positions_snapshot=dashboard_positions_snapshot,
                        daily_summary_snapshot=dashboard_daily_snapshot,
                    )
                _phase_done("trial7_9_intrabar", _intrabar_t78_started)

            mkt = None
            if is_new or args.once:
                _bar_close_started = time.perf_counter()
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

                trades_df = _get_trades_df_cached()

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
                            _invalidate_trades_df_cache()
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

                # Always mark bar as seen so we don't re-run every poll for same bar (even when clock fallback was used).
                last_seen_m1_time = m1_last_time
                _phase_done("bar_close_core", _bar_close_started)

            if has_price_level and mkt is None:
                mkt = _compute_mkt(profile, tick, data_by_tf)
            if has_price_level and mkt is not None:
                trades_df = _get_trades_df_cached()
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
                            _invalidate_trades_df_cache()
                        else:
                            print(f"[{profile.profile_name}] {dec.reason}")

            if (has_indicator or has_bollinger or has_vwap or has_ema_pullback) and mkt is None:
                mkt = _compute_mkt(profile, tick, data_by_tf)
            if has_indicator and mkt is not None:
                trades_df = _get_trades_df_cached()
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
                            _invalidate_trades_df_cache()
                        else:
                            print(f"[{profile.profile_name}] {dec.reason}")

            # Breakout range policies
            if has_breakout and mkt is None:
                mkt = _compute_mkt(profile, tick, data_by_tf)
            if has_breakout and mkt is not None:
                trades_df = _get_trades_df_cached()
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
                            _invalidate_trades_df_cache()
                        else:
                            print(f"[{profile.profile_name}] breakout_range {pol.id} mode={mode} -> {dec.reason}")

            # Session momentum policies
            if has_session and mkt is None:
                mkt = _compute_mkt(profile, tick, data_by_tf)
            if has_session and mkt is not None:
                trades_df = _get_trades_df_cached()
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
                            _invalidate_trades_df_cache()
                        else:
                            print(f"[{profile.profile_name}] session_momentum {pol.id} mode={mode} -> {dec.reason}")

            # Bollinger Bands policies
            if has_bollinger and mkt is not None:
                trades_df = _get_trades_df_cached()
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
                            _invalidate_trades_df_cache()
                        else:
                            print(f"[{profile.profile_name}] bollinger_bands {pol.id} mode={mode} -> {dec.reason}")

            # VWAP policies
            if has_vwap and mkt is not None:
                trades_df = _get_trades_df_cached()
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
                            _invalidate_trades_df_cache()
                        else:
                            print(f"[{profile.profile_name}] vwap {pol.id} mode={mode} -> {dec.reason}")

            # EMA pullback policies (M5-M15 momentum pullback)
            if has_ema_pullback and mkt is not None:
                trades_df = _get_trades_df_cached()
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
                            _invalidate_trades_df_cache()
                        else:
                            print(f"[{profile.profile_name}] ema_pullback {pol.id} mode={mode} -> {dec.reason}")

            # KT/CG Hybrid policies (Trial #2)
            if has_kt_cg_hybrid and mkt is not None:
                trades_df = _get_trades_df_cached()
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
                            _invalidate_trades_df_cache()
                            _append_trade_open_event(
                                log_dir, f"kt_cg_hybrid:{pol.id}:{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d%H%M%S')}",
                                side, entry_price,
                            )
                        else:
                            print(f"[{profile.profile_name}] kt_cg_hybrid {pol.id} mode={mode} -> {dec.reason}")

                    # Dashboard assembly for Trial #2
                    _build_and_write_dashboard(
                        profile=profile, store=store, log_dir=log_dir, tick=tick,
                        data_by_tf=data_by_tf, mode=mode, adapter=adapter, policy=pol,
                        policy_type="kt_cg_hybrid",
                        open_positions_snapshot=open_positions_snapshot,
                        positions_snapshot=dashboard_positions_snapshot,
                        daily_summary_snapshot=dashboard_daily_snapshot,
                    )

            # KT/CG Counter-Trend Pullback policies (Trial #3)
            if has_kt_cg_ctp and mkt is not None:
                trades_df = _get_trades_df_cached()
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
                            _invalidate_trades_df_cache()
                            _append_trade_open_event(
                                log_dir, f"kt_cg_ctp:{pol.id}:{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d%H%M%S')}",
                                side, entry_price,
                            )
                        else:
                            print(f"[{profile.profile_name}] kt_cg_ctp {pol.id} mode={mode} -> {dec.reason}")

                    # Dashboard assembly for Trial #3
                    _build_and_write_dashboard(
                        profile=profile, store=store, log_dir=log_dir, tick=tick,
                        data_by_tf=data_by_tf, mode=mode, adapter=adapter, policy=pol,
                        policy_type="kt_cg_counter_trend_pullback",
                        temp_overrides=temp_overrides,
                        open_positions_snapshot=open_positions_snapshot,
                        positions_snapshot=dashboard_positions_snapshot,
                        daily_summary_snapshot=dashboard_daily_snapshot,
                    )

            # Trial #4 execution — runs EVERY poll cycle (not just on M1 bar close)
            # so tiered pullback can detect live price touches between bar closes
            if has_kt_cg_trial_4:
                # Use existing mkt if available (on bar close), otherwise compute minimal context
                t4_mkt = mkt
                if t4_mkt is None:
                    spread_pips = (tick.ask - tick.bid) / float(profile.pip_size)
                    t4_mkt = MarketContext(spread_pips=float(spread_pips), alignment_score=0)
                trades_df = _get_trades_df_cached()
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
                            _invalidate_trades_df_cache()
                            _append_trade_open_event(
                                log_dir, f"kt_cg_trial_4:{pol.id}:{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d%H%M%S')}",
                                side, entry_price,
                                trigger_type=_event_trigger_label(t4_trigger_type, dec.reason),
                                entry_type=t4_trigger_type or "",
                            )
                        else:
                            print(f"[{profile.profile_name}] kt_cg_trial_4 {pol.id} mode={mode} -> {dec.reason}")

                    # Dashboard assembly for Trial #4
                    _build_and_write_dashboard(
                        profile=profile, store=store, log_dir=log_dir, tick=tick,
                        data_by_tf=data_by_tf, mode=mode, adapter=adapter, policy=pol,
                        policy_type="kt_cg_trial_4", tier_state=tier_state,
                        eval_result=exec_result, divergence_state=divergence_state,
                        temp_overrides=temp_overrides,
                        open_positions_snapshot=open_positions_snapshot,
                        positions_snapshot=dashboard_positions_snapshot,
                        daily_summary_snapshot=dashboard_daily_snapshot,
                    )

            # Trial #5 execution — same structure as Trial #4 with dual ATR filter
            if has_kt_cg_trial_5:
                t5_mkt = mkt
                if t5_mkt is None:
                    spread_pips = (tick.ask - tick.bid) / float(profile.pip_size)
                    t5_mkt = MarketContext(spread_pips=float(spread_pips), alignment_score=0)
                trades_df = _get_trades_df_cached()
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
                            _invalidate_trades_df_cache()
                            _append_trade_open_event(
                                log_dir, f"kt_cg_trial_5:{pol.id}:{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d%H%M%S')}",
                                side, entry_price,
                                trigger_type=_event_trigger_label(t5_trigger_type, dec.reason),
                                entry_type=t5_trigger_type or "",
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

                    # Dashboard assembly for Trial #5
                    _build_and_write_dashboard(
                        profile=profile, store=store, log_dir=log_dir, tick=tick,
                        data_by_tf=data_by_tf, mode=mode, adapter=adapter, policy=pol,
                        policy_type="kt_cg_trial_5", tier_state=tier_state,
                        eval_result=exec_result, divergence_state=divergence_state,
                        daily_reset_state=daily_reset_state_t5,
                        exhaustion_result=exec_result.get("exhaustion_result"),
                        temp_overrides=temp_overrides,
                        open_positions_snapshot=open_positions_snapshot,
                        positions_snapshot=dashboard_positions_snapshot,
                        daily_summary_snapshot=dashboard_daily_snapshot,
                    )

            # Trial #7 / Trial #8 execution — M5 Trend + Tiered Pullback (+ Daily Level Filter for T8)
            if (has_kt_cg_trial_7 or has_kt_cg_trial_8 or has_kt_cg_trial_9) and not intrabar_t78:
                t7_mkt = mkt
                if t7_mkt is None:
                    spread_pips = (tick.ask - tick.bid) / float(profile.pip_size)
                    t7_mkt = MarketContext(spread_pips=float(spread_pips), alignment_score=0)
                trades_df = _get_trades_df_cached()
                tier_state: dict[int, bool] = {}
                try:
                    state_data = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
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
                except Exception:
                    tier_state = {}

                for pol in profile.execution.policies:
                    pol_type = getattr(pol, "type", None)
                    if not getattr(pol, "enabled", True) or pol_type not in ("kt_cg_trial_7", "kt_cg_trial_8", "kt_cg_trial_9"):
                        continue
                    m1_df = data_by_tf.get("M1")
                    if m1_df is None or m1_df.empty:
                        continue
                    bar_time_utc = pd.to_datetime(m1_df["time"].iloc[-1], utc=True).isoformat()
                    if pol_type == "kt_cg_trial_9":
                        # Update NTZ levels from candle data
                        if ntz_filter is not None:
                            _ntz_levels: dict = {}
                            try:
                                def _prev_candle_hl(df, tf_label):
                                    """Extract high/low from the previous completed candle (sort by time, check if last is today)."""
                                    if df is None or df.empty:
                                        return None, None
                                    d_sorted = df.copy()
                                    d_sorted["_time"] = pd.to_datetime(d_sorted["time"], utc=True, errors="coerce")
                                    d_sorted = d_sorted.dropna(subset=["_time"]).sort_values("_time")
                                    if len(d_sorted) < 1:
                                        return None, None
                                    # W/MN: use CURRENT candle HL (forming week/month). We fetch with include_incomplete=True.
                                    if tf_label in ("W", "MN"):
                                        cur_row = d_sorted.iloc[-1]
                                        h = float(cur_row["high"])
                                        l = float(cur_row["low"])
                                        _log(f"NTZ {tf_label}: using CURRENT candle iloc[-1] time={cur_row['_time']} H={h:.3f} L={l:.3f}")
                                        return h, l
                                    now_date = pd.Timestamp.now(tz="UTC").date().isoformat()
                                    last_date = d_sorted.iloc[-1]["_time"].date().isoformat()
                                    if last_date == now_date:
                                        # Last candle is today's forming candle — use the one before
                                        if len(d_sorted) < 2:
                                            return None, None
                                        prev_row = d_sorted.iloc[-2]
                                        _log(f"NTZ {tf_label}: last candle is today ({now_date}), using iloc[-2]")
                                    else:
                                        # Last candle is a past completed day — use it directly
                                        prev_row = d_sorted.iloc[-1]
                                        _log(f"NTZ {tf_label}: last candle is {last_date} (not today {now_date}), using iloc[-1]")
                                    h = float(prev_row["high"])
                                    l = float(prev_row["low"])
                                    _log(f"NTZ {tf_label}: time={prev_row['_time']} H={h:.3f} L={l:.3f}")
                                    return h, l

                                d_df = data_by_tf.get("D")
                                dh, dl = _prev_candle_hl(d_df, "D1")
                                if dh is not None:
                                    _ntz_levels["prev_day_high"] = dh
                                    _ntz_levels["prev_day_low"] = dl

                                w_df = data_by_tf.get("W")
                                wh, wl = _prev_candle_hl(w_df, "W")
                                if wh is not None:
                                    _ntz_levels["weekly_high"] = wh
                                    _ntz_levels["weekly_low"] = wl

                                mn_df = data_by_tf.get("MN")
                                mh, ml = _prev_candle_hl(mn_df, "MN")
                                if mh is not None:
                                    _ntz_levels["monthly_high"] = mh
                                    _ntz_levels["monthly_low"] = ml

                                ntz_filter.update_levels(**_ntz_levels)
                            except Exception as _ntz_err:
                                _log(f"NTZ level update error: {_ntz_err}", "ERROR")
                        exec_result = execute_kt_cg_trial_9_policy_demo_only(
                            adapter=adapter,
                            profile=profile,
                            log_dir=log_dir,
                            policy=pol,
                            context=t7_mkt,
                            data_by_tf=data_by_tf,
                            tick=tick,
                            trades_df=trades_df,
                            mode=mode,
                            bar_time_utc=bar_time_utc,
                            tier_state=tier_state,
                            store=store,
                            daily_level_filter=daily_level_filter,
                            daily_state=trial7_daily_state,
                        )
                    elif pol_type == "kt_cg_trial_8":
                        exec_result = execute_kt_cg_trial_8_policy_demo_only(
                            adapter=adapter,
                            profile=profile,
                            log_dir=log_dir,
                            policy=pol,
                            context=t7_mkt,
                            data_by_tf=data_by_tf,
                            tick=tick,
                            trades_df=trades_df,
                            mode=mode,
                            bar_time_utc=bar_time_utc,
                            tier_state=tier_state,
                            store=store,
                            daily_level_filter=daily_level_filter,
                            daily_state=trial7_daily_state,
                        )
                    else:
                        exec_result = execute_kt_cg_trial_7_policy_demo_only(
                            adapter=adapter,
                            profile=profile,
                            log_dir=log_dir,
                            policy=pol,
                            context=t7_mkt,
                            data_by_tf=data_by_tf,
                            tick=tick,
                            trades_df=trades_df,
                            mode=mode,
                            bar_time_utc=bar_time_utc,
                            tier_state=tier_state,
                            store=store,
                            daily_level_filter=None,
                            daily_state=None,
                        )
                    dec = exec_result["decision"]
                    tier_updates = exec_result.get("tier_updates", {})
                    t7_trigger_type = exec_result.get("trigger_type")

                    # Persist tier state updates to runtime_state.json
                    if tier_updates:
                        try:
                            current_state_data = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
                            tf_dict = current_state_data.get("tier_fired", {})
                            if not isinstance(tf_dict, dict):
                                tf_dict = {}
                            for tier, new_state in tier_updates.items():
                                tf_dict[str(tier)] = new_state
                                tier_state[tier] = new_state
                            current_state_data["tier_fired"] = tf_dict
                            state_path.write_text(json.dumps(current_state_data, indent=2) + "\n", encoding="utf-8")
                        except Exception as e:
                            print(f"[{profile.profile_name}] Failed to persist Trial #7 tier state: {e}")

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
                            # When T8/T9 trailing exit is on, do not set broker TP so loop can partial-close + trail
                            if pol_type in ("kt_cg_trial_8", "kt_cg_trial_9") and getattr(pol, "trail_after_tp1", False):
                                tp_price = None
                            print(f"[{profile.profile_name}] TRADE PLACED: {pol_type}:{pol.id} | side={side} | entry={entry_price:.3f} | {dec.reason}")
                            _insert_trade_for_policy(
                                profile=profile,
                                adapter=adapter,
                                store=store,
                                policy_type=pol_type,
                                policy_id=pol.id,
                                side=side,
                                entry_price=entry_price,
                                dec=dec,
                                stop_price=sl_price,
                                target_price=tp_price,
                                entry_type=t7_trigger_type,
                            )
                            _invalidate_trades_df_cache()
                            _append_trade_open_event(
                                log_dir, f"{pol_type}:{pol.id}:{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d%H%M%S')}",
                                side, entry_price,
                                trigger_type=_event_trigger_label(t7_trigger_type, dec.reason),
                                entry_type=t7_trigger_type or "",
                            )
                        else:
                            print(f"[{profile.profile_name}] {pol_type} {pol.id} mode={mode} -> {dec.reason}")

                    _build_and_write_dashboard(
                        profile=profile, store=store, log_dir=log_dir, tick=tick,
                        data_by_tf=data_by_tf, mode=mode, adapter=adapter, policy=pol,
                        policy_type=pol_type, tier_state=tier_state,
                        eval_result=exec_result,
                        exhaustion_result=exec_result.get("exhaustion_result"),
                        daily_level_filter=daily_level_filter if pol_type == "kt_cg_trial_8" else None,
                        ntz_filter=ntz_filter if pol_type == "kt_cg_trial_9" else None,
                        open_positions_snapshot=open_positions_snapshot,
                        positions_snapshot=dashboard_positions_snapshot,
                        daily_summary_snapshot=dashboard_daily_snapshot,
                    )
                # After Trial 7/8/9 block: mark this bar as seen so we don't re-run every poll for same bar.
                # Always update (even when clock fallback was used) so we don't re-evaluate the same bar every poll.
                last_seen_m1_time = m1_last_time

            # Trial #6 execution — BB Slope Trend + EMA Tier Pullback + BB Reversal
            if has_kt_cg_trial_6:
                t6_mkt = mkt
                if t6_mkt is None:
                    spread_pips = (tick.ask - tick.bid) / float(profile.pip_size)
                    t6_mkt = MarketContext(spread_pips=float(spread_pips), alignment_score=0)
                trades_df = _get_trades_df_cached()
                # Load tier state from runtime_state.json
                t6_tier_state: dict[int, bool] = {}
                t6_daily_reset_state: dict = {}
                try:
                    state_data = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
                    # EMA tier state (reuse tier_fired dict)
                    tier_fired_raw = state_data.get("tier_fired")
                    if isinstance(tier_fired_raw, dict):
                        t6_tier_state = {int(k): bool(v) for k, v in tier_fired_raw.items()}
                    # Daily reset / dead zone state
                    t6_daily_reset_state = {
                        "daily_reset_block_active": bool(state_data.get("daily_reset_block_active", False)),
                    }
                except Exception:
                    t6_tier_state = {}
                    t6_daily_reset_state = {}

                for pol in profile.execution.policies:
                    if not getattr(pol, "enabled", True) or getattr(pol, "type", None) != "kt_cg_trial_6":
                        continue
                    m1_df = data_by_tf.get("M1")
                    if m1_df is None or m1_df.empty:
                        continue

                    exec_result = execute_kt_cg_trial_6_policy_demo_only(
                        adapter=adapter,
                        profile=profile,
                        log_dir=log_dir,
                        policy=pol,
                        context=t6_mkt,
                        data_by_tf=data_by_tf,
                        tick=tick,
                        trades_df=trades_df,
                        mode=mode,
                        tier_state=t6_tier_state,
                        daily_reset_state=t6_daily_reset_state,
                    )
                    dec = exec_result["decision"]
                    tier_updates = exec_result.get("tier_updates", {})
                    t6_trigger_type = exec_result.get("trigger_type")
                    t6_daily_reset_state = exec_result.get("daily_reset_state", t6_daily_reset_state)

                    # Persist tier RESETS immediately, tier FIRES after trade placed
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
                                t6_tier_state[tier] = new_state
                            current_state_data["tier_fired"] = tf_dict
                            state_path.write_text(json.dumps(current_state_data, indent=2) + "\n", encoding="utf-8")
                        except Exception as e:
                            print(f"[{profile.profile_name}] Failed to persist T6 tier resets: {e}")

                    # Persist dead zone state
                    if t6_daily_reset_state:
                        try:
                            current_state_data = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
                            current_state_data["daily_reset_block_active"] = t6_daily_reset_state.get("daily_reset_block_active", False)
                            state_path.write_text(json.dumps(current_state_data, indent=2) + "\n", encoding="utf-8")
                        except Exception as e:
                            print(f"[{profile.profile_name}] Failed to persist T6 dead zone state: {e}")

                    if dec.attempted:
                        if dec.placed:
                            side = dec.side or "buy"
                            entry_price = tick.ask if side == "buy" else tick.bid
                            pip = float(profile.pip_size)
                            t6_tp_pips = exec_result.get("tp_pips") or pol.ema_tier_tp_pips
                            t6_sl_pips = exec_result.get("sl_pips") or pol.sl_pips
                            if side == "buy":
                                tp_price = entry_price + t6_tp_pips * pip
                                sl_price = entry_price - t6_sl_pips * pip
                            else:
                                tp_price = entry_price - t6_tp_pips * pip
                                sl_price = entry_price + t6_sl_pips * pip
                            print(f"[{profile.profile_name}] TRADE PLACED: kt_cg_trial_6:{pol.id}:{t6_trigger_type} | side={side} | entry={entry_price:.3f} | {dec.reason}")
                            _insert_trade_for_policy(
                                profile=profile,
                                adapter=adapter,
                                store=store,
                                policy_type="kt_cg_trial_6",
                                policy_id=pol.id,
                                side=side,
                                entry_price=entry_price,
                                dec=dec,
                                stop_price=sl_price,
                                target_price=tp_price,
                                entry_type=t6_trigger_type,
                            )
                            _invalidate_trades_df_cache()
                            _append_trade_open_event(
                                log_dir, f"kt_cg_trial_6:{pol.id}:{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d%H%M%S')}",
                                side, entry_price, trigger_type=t6_trigger_type or "", entry_type=t6_trigger_type or "",
                            )
                            # Persist tier FIRES after trade confirmed placed
                            if tier_fires:
                                try:
                                    current_state_data = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
                                    tf_dict = current_state_data.get("tier_fired", {})
                                    if not isinstance(tf_dict, dict):
                                        tf_dict = {}
                                    for tier, new_state in tier_fires.items():
                                        tf_dict[str(tier)] = new_state
                                        t6_tier_state[tier] = new_state
                                    current_state_data["tier_fired"] = tf_dict
                                    state_path.write_text(json.dumps(current_state_data, indent=2) + "\n", encoding="utf-8")
                                except Exception as e:
                                    print(f"[{profile.profile_name}] Failed to persist T6 tier fires: {e}")
                        elif dec.reason and "dead_zone" not in dec.reason and dec.reason != "no_signal":
                            print(f"[{profile.profile_name}] kt_cg_trial_6 BLOCKED: {dec.reason}")

                    # Dashboard assembly for Trial #6
                    _build_and_write_dashboard(
                        profile=profile, store=store, log_dir=log_dir, tick=tick,
                        data_by_tf=data_by_tf, mode=mode, adapter=adapter, policy=pol,
                        policy_type="kt_cg_trial_6", tier_state=t6_tier_state,
                        eval_result=exec_result,
                        open_positions_snapshot=open_positions_snapshot,
                        positions_snapshot=dashboard_positions_snapshot,
                        daily_summary_snapshot=dashboard_daily_snapshot,
                    )

            # Uncle Parsh H1 Breakout execution
            if has_uncle_parsh:
                up_mkt = mkt
                if up_mkt is None:
                    spread_pips = (tick.ask - tick.bid) / float(profile.pip_size)
                    up_mkt = MarketContext(spread_pips=float(spread_pips), alignment_score=0)
                trades_df = _get_trades_df_cached()

                # Track M5 bar changes
                m5_df_up = data_by_tf.get("M5")
                if m5_df_up is not None and not m5_df_up.empty:
                    m5_last_time_up = pd.to_datetime(m5_df_up["time"].iloc[-1], utc=True).isoformat()
                else:
                    m5_last_time_up = ""
                is_new_m5_up = m5_last_time_up != getattr(_run_trade_management, '_last_m5_time_up', None)

                # Load level state from runtime_state.json
                up_level_state: list[dict] = []
                try:
                    state_data = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
                    up_level_state = state_data.get("h1_breakout_levels", [])
                    # Daily reset: clear levels when date changes
                    stored_scan_date = state_data.get("h1_breakout_scan_date", "")
                    current_date_up = pd.Timestamp.now(tz="UTC").date().isoformat()
                    if stored_scan_date != current_date_up:
                        up_level_state = []
                except Exception:
                    up_level_state = []

                # Build Uncle Parsh temp overrides from runtime_state.json
                up_temp_overrides: dict = {}
                try:
                    _up_state_data = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
                    _up_override_keys = [
                        ("m5_trend_ema_fast", "temp_up_m5_ema_fast"),
                        ("m5_trend_ema_slow", "temp_up_m5_ema_slow"),
                        ("major_extremes_only", "temp_up_major_extremes_only"),
                        ("h1_lookback_hours", "temp_up_h1_lookback_hours"),
                        ("h1_swing_strength", "temp_up_h1_swing_strength"),
                        ("h1_cluster_tolerance_pips", "temp_up_h1_cluster_tolerance_pips"),
                        ("h1_min_touches_for_major", "temp_up_h1_min_touches_for_major"),
                        ("power_close_body_pct", "temp_up_power_close_body_pct"),
                        ("velocity_pips", "temp_up_velocity_pips"),
                        ("initial_sl_spread_plus_pips", "temp_up_initial_sl_spread_plus_pips"),
                        ("tp1_pips", "temp_up_tp1_pips"),
                        ("tp1_close_pct", "temp_up_tp1_close_pct"),
                        ("be_spread_plus_pips", "temp_up_be_spread_plus_pips"),
                        ("trail_ema_period", "temp_up_trail_ema_period"),
                        ("max_spread_pips", "temp_up_max_spread_pips"),
                    ]
                    for field_name, state_key in _up_override_keys:
                        val = _up_state_data.get(state_key)
                        if val is not None:
                            up_temp_overrides[field_name] = val
                except Exception:
                    up_temp_overrides = {}

                for pol in profile.execution.policies:
                    if not getattr(pol, "enabled", True) or getattr(pol, "type", None) != "uncle_parsh_h1_breakout":
                        continue
                    m1_df_up = data_by_tf.get("M1")
                    if m1_df_up is None or m1_df_up.empty:
                        continue
                    bar_time_utc = pd.to_datetime(m1_df_up["time"].iloc[-1], utc=True).isoformat()
                    exec_result = execute_h1_breakout_policy_demo_only(
                        adapter=adapter,
                        profile=profile,
                        log_dir=log_dir,
                        policy=pol,
                        context=up_mkt,
                        data_by_tf=data_by_tf,
                        tick=tick,
                        trades_df=trades_df,
                        mode=mode,
                        bar_time_utc=bar_time_utc,
                        level_state=up_level_state,
                        is_new_m1=is_new,
                        is_new_m5=is_new_m5_up,
                        temp_overrides=up_temp_overrides or None,
                    )
                    dec = exec_result["decision"]
                    up_level_updates = exec_result.get("level_updates", [])
                    up_entry_type = exec_result.get("entry_type", "")

                    # Persist level state to runtime_state.json
                    try:
                        current_state_data = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
                        current_state_data["h1_breakout_levels"] = up_level_updates
                        current_state_data["h1_breakout_scan_date"] = pd.Timestamp.now(tz="UTC").date().isoformat()
                        current_state_data["last_m5_time_up"] = m5_last_time_up
                        state_path.write_text(json.dumps(current_state_data, indent=2) + "\n", encoding="utf-8")
                    except Exception as e:
                        print(f"[{profile.profile_name}] Failed to persist H1 breakout level state: {e}")

                    if dec.attempted:
                        if dec.placed:
                            side = dec.side or "buy"
                            entry_price = tick.ask if side == "buy" else tick.bid
                            pip = float(profile.pip_size)
                            current_spread = tick.ask - tick.bid
                            _ov_sl_pips = up_temp_overrides.get("initial_sl_spread_plus_pips", pol.initial_sl_spread_plus_pips)
                            sl_distance = current_spread + _ov_sl_pips * pip
                            if side == "buy":
                                sl_price = entry_price - sl_distance
                            else:
                                sl_price = entry_price + sl_distance
                            tp_price = None  # dynamic EMA exits; no broker TP
                            print(f"[{profile.profile_name}] TRADE PLACED: uncle_parsh:{pol.id} | side={side} | entry={entry_price:.3f} | {dec.reason}")
                            _insert_trade_for_policy(
                                profile=profile,
                                adapter=adapter,
                                store=store,
                                policy_type="uncle_parsh_h1_breakout",
                                policy_id=pol.id,
                                side=side,
                                entry_price=entry_price,
                                dec=dec,
                                stop_price=sl_price,
                                target_price=tp_price,
                                entry_type=up_entry_type,
                            )
                            _invalidate_trades_df_cache()
                            _append_trade_open_event(
                                log_dir, f"uncle_parsh:{pol.id}:{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d%H%M%S')}",
                                side, entry_price,
                                trigger_type=up_entry_type or "",
                                entry_type=up_entry_type or "",
                            )
                        else:
                            if dec.reason and "waiting" not in dec.reason:
                                print(f"[{profile.profile_name}] uncle_parsh {pol.id} mode={mode} -> {dec.reason}")

                    # Dashboard assembly for Uncle Parsh
                    _build_and_write_dashboard(
                        profile=profile, store=store, log_dir=log_dir, tick=tick,
                        data_by_tf=data_by_tf, mode=mode, adapter=adapter, policy=pol,
                        policy_type="uncle_parsh_h1_breakout",
                        eval_result=exec_result,
                        open_positions_snapshot=open_positions_snapshot,
                        positions_snapshot=dashboard_positions_snapshot,
                        daily_summary_snapshot=dashboard_daily_snapshot,
                    )

                # Update M5 time tracker
                _run_trade_management._last_m5_time_up = m5_last_time_up

            # Phase 3 Integrated execution
            if has_phase3_integrated:
                p3_mkt = mkt
                if p3_mkt is None:
                    spread_pips = (tick.ask - tick.bid) / float(profile.pip_size)
                    p3_mkt = MarketContext(spread_pips=float(spread_pips), alignment_score=0)

                # Count open Phase 3 trades from broker positions
                try:
                    p3_open_count = 0
                    for _p3pos in (open_positions_snapshot or []):
                        _p3_comment = ""
                        if isinstance(_p3pos, dict):
                            _p3_ce = _p3pos.get("clientExtensions") or _p3pos.get("tradeClientExtensions")
                            if isinstance(_p3_ce, dict):
                                _p3_comment = str(_p3_ce.get("comment") or "")
                        if "phase3_integrated" in _p3_comment:
                            p3_open_count += 1
                    phase3_state["open_trade_count"] = p3_open_count
                except Exception:
                    pass

                for pol in profile.execution.policies:
                    pol_type = getattr(pol, "type", None)
                    if not getattr(pol, "enabled", True) or pol_type != "phase3_integrated":
                        continue

                    from core.phase3_integrated_engine import (
                        execute_phase3_integrated_policy_demo_only,
                        load_phase3_sizing_config,
                    )
                    phase3_sizing = load_phase3_sizing_config()
                    exec_result = execute_phase3_integrated_policy_demo_only(
                        adapter=adapter,
                        profile=profile,
                        log_dir=log_dir,
                        policy=pol,
                        context=p3_mkt,
                        data_by_tf=data_by_tf,
                        tick=tick,
                        mode=mode,
                        phase3_state=phase3_state,
                        store=store,
                        sizing_config=phase3_sizing if phase3_sizing else None,
                        is_new_m1=(is_new or args.once),
                    )
                    dec = exec_result["decision"]
                    p3_state_updates = exec_result.get("phase3_state_updates", {})
                    strategy_tag = exec_result.get("strategy_tag")

                    # Persist state updates
                    if p3_state_updates:
                        phase3_state.update(p3_state_updates)
                        try:
                            current_state_data = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
                            current_state_data["phase3_state"] = phase3_state
                            state_path.write_text(json.dumps(current_state_data, indent=2) + "\n", encoding="utf-8")
                        except Exception as e:
                            print(f"[{profile.profile_name}] Failed to persist phase3_state: {e}")

                    if dec.attempted:
                        if dec.placed:
                            side = dec.side or "buy"
                            fill_price = getattr(dec, "fill_price", None)
                            entry_price = float(fill_price) if fill_price is not None else (tick.ask if side == "buy" else tick.bid)
                            sl_price = exec_result.get("sl_price")
                            tp1_price = exec_result.get("tp1_price")
                            risk_usd_planned = exec_result.get("risk_usd_planned")
                            _p3_units = exec_result.get("units")
                            _p3_size_lots = (float(_p3_units) / 100_000.0) if _p3_units is not None and int(_p3_units) > 0 else None
                            print(f"[{profile.profile_name}] TRADE PLACED: phase3_integrated:{pol.id} | side={side} | entry={entry_price:.3f} | {dec.reason}")
                            _entry_session = None
                            if strategy_tag:
                                if "v14" in strategy_tag or "mean_reversion" in strategy_tag:
                                    _entry_session = "tokyo"
                                elif "london" in strategy_tag:
                                    _entry_session = "london"
                                elif "v44_ny" in strategy_tag:
                                    _entry_session = "ny"
                            _insert_trade_for_policy(
                                profile=profile,
                                adapter=adapter,
                                store=store,
                                policy_type="phase3_integrated",
                                policy_id=pol.id,
                                side=side,
                                entry_price=entry_price,
                                dec=dec,
                                stop_price=sl_price,
                                target_price=tp1_price,
                                size_lots=_p3_size_lots,
                                entry_type=strategy_tag,
                                entry_session=_entry_session,
                                risk_usd_planned=risk_usd_planned,
                            )
                            _invalidate_trades_df_cache()
                            _append_trade_open_event(
                                log_dir, f"phase3_integrated:{pol.id}:{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d%H%M%S')}",
                                side, entry_price,
                                trigger_type=strategy_tag or "",
                                entry_type=strategy_tag or "",
                            )
                        else:
                            print(f"[{profile.profile_name}] phase3 {pol.id} mode={mode} -> {dec.reason}")

                    _build_and_write_dashboard(
                        profile=profile, store=store, log_dir=log_dir, tick=tick,
                        data_by_tf=data_by_tf, mode=mode, adapter=adapter, policy=pol,
                        policy_type="phase3_integrated",
                        eval_result=exec_result,
                        phase3_state=phase3_state,
                        open_positions_snapshot=open_positions_snapshot,
                        positions_snapshot=dashboard_positions_snapshot,
                        daily_summary_snapshot=dashboard_daily_snapshot,
                    )
                    # Per-minute diagnostics: bar time, is_new, placed, blocking filter count/reasons, decision reason
                    if is_new or args.once:
                        _append_phase3_minute_diagnostics(
                            log_dir=log_dir,
                            profile=profile,
                            store=store,
                            tick=tick,
                            data_by_tf=data_by_tf,
                            policy=pol,
                            exec_result=exec_result,
                            phase3_state=phase3_state,
                            m1_bar_time=m1_last_time,
                            is_new=is_new,
                            mode=mode,
                        )

                # Bar-close parity for Phase 3: mark the current M1 bar as seen after evaluation
                # so entries are evaluated once per newly closed M1 candle.
                last_seen_m1_time = m1_last_time

            # Catch-all dashboard write for profiles not using KT/CG policy types.
            # Runs every poll cycle so positions + prices are always fresh.
            if not (has_kt_cg_hybrid or has_kt_cg_ctp or has_kt_cg_trial_4 or has_kt_cg_trial_5 or has_kt_cg_trial_6 or has_kt_cg_trial_7 or has_kt_cg_trial_8 or has_kt_cg_trial_9 or has_uncle_parsh or has_phase3_integrated):
                _dashboard_fallback_started = time.perf_counter()
                if tick is not None and data_by_tf is not None:
                    _build_and_write_dashboard(
                        profile=profile, store=store, log_dir=log_dir, tick=tick,
                        data_by_tf=data_by_tf, mode=mode, adapter=adapter,
                        open_positions_snapshot=open_positions_snapshot,
                        positions_snapshot=dashboard_positions_snapshot,
                        daily_summary_snapshot=dashboard_daily_snapshot,
                    )
                _phase_done("dashboard_fallback", _dashboard_fallback_started)

            if args.once:
                break

            loop_elapsed = time.perf_counter() - iter_started
            if loop_elapsed >= 5.0:
                phase_summary = ", ".join(
                    f"{name}={secs:.2f}s"
                    for name, secs in sorted(phase_times.items(), key=lambda item: item[1], reverse=True)
                    if secs >= 0.05
                )
                _log(f"SLOW LOOP total={loop_elapsed:.2f}s poll={poll_sec:.2f}s {phase_summary}", "WARN")

            time.sleep(poll_sec)

          except Exception as _loop_exc:
            _log(f"LOOP EXCEPTION: {_loop_exc}\n{traceback.format_exc()}", "ERROR")
            time.sleep(5)

    finally:
        adapter.shutdown()


if __name__ == "__main__":
    main()
