from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd


def execute_tokyo_v14_session(
    *,
    adapter,
    profile,
    policy,
    data_by_tf: dict,
    tick,
    mode: str,
    phase3_state: dict,
    store=None,
    sizing_config: Optional[dict[str, Any]] = None,
    now_utc: Optional[datetime] = None,
    ownership_audit: Optional[dict[str, Any]] = None,
    overlay_state: Optional[dict[str, Any]] = None,
    base_state_updates: Optional[dict[str, Any]] = None,
    support,
) -> dict:
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    ExecutionDecision = support.ExecutionDecision
    _drop_incomplete_tf = support._drop_incomplete_tf
    compute_daily_fib_pivots = support.compute_daily_fib_pivots
    _compute_bb = support._compute_bb
    _compute_rsi = support._compute_rsi
    _compute_atr = support._compute_atr
    _compute_adx = support._compute_adx
    detect_sar_flip = support.detect_sar_flip
    _as_risk_fraction = support._as_risk_fraction
    _account_sizing_value = support._account_sizing_value
    compute_v14_units_from_config = support.compute_v14_units_from_config
    compute_v14_lot_size = support.compute_v14_lot_size
    evaluate_v14_confluence = support.evaluate_v14_confluence
    compute_v14_signal_strength = support.compute_v14_signal_strength
    compute_v14_sl = support.compute_v14_sl
    compute_v14_tp1 = support.compute_v14_tp1
    resolve_v14_cell_scale_override_from_state = support.resolve_v14_cell_scale_override_from_state
    _phase3_order_confirmed = support._phase3_order_confirmed
    parse_hhmm_to_hour = support.parse_hhmm_to_hour

    pip = float(profile.pip_size) if hasattr(profile, "pip_size") else support.PIP_SIZE
    effective_cfg = sizing_config or {}
    state_updates: dict[str, Any] = dict(base_state_updates or {})
    no_trade = {
        "decision": ExecutionDecision(attempted=False, placed=False, reason="", side=None),
        "phase3_state_updates": state_updates,
        "strategy_tag": None,
    }

    if not isinstance(ownership_audit, dict):
        no_trade["decision"] = ExecutionDecision(
            attempted=False,
            placed=False,
            reason="phase3: missing shared ownership audit",
            side=None,
        )
        return no_trade

    if mode != "ARMED_AUTO_DEMO":
        no_trade["decision"] = ExecutionDecision(
            attempted=False,
            placed=False,
            reason=f"phase3: mode={mode} (need ARMED_AUTO_DEMO)",
            side=None,
        )
        return no_trade

    is_demo = getattr(adapter, "is_demo", True)
    if callable(is_demo):
        is_demo = is_demo()
    if not is_demo:
        no_trade["decision"] = ExecutionDecision(
            attempted=False,
            placed=False,
            reason="phase3: adapter is not demo",
            side=None,
        )
        return no_trade

    spread_pips = (tick.ask - tick.bid) / pip
    v14_cfg_for_spread = effective_cfg.get("v14", {})
    v14_max_entry_spread = float(v14_cfg_for_spread.get("max_entry_spread_pips", support.MAX_ENTRY_SPREAD_PIPS))
    if spread_pips > v14_max_entry_spread:
        no_trade["decision"] = ExecutionDecision(
            attempted=False,
            placed=False,
            reason=f"phase3: spread {spread_pips:.1f}p > max {v14_max_entry_spread:.1f}p",
            side=None,
        )
        return no_trade

    current_ny_day = (pd.Timestamp(now_utc) - pd.Timedelta(hours=22)).date()
    prev_ny_day = current_ny_day - pd.Timedelta(days=1)
    today_str = current_ny_day.isoformat()
    pivots = phase3_state.get("pivots")
    pivots_date = phase3_state.get("pivots_date")
    if pivots is None or pivots_date != today_str:
        m1_for_pivots = _drop_incomplete_tf(data_by_tf.get("M1"), "M1")
        if m1_for_pivots is None or m1_for_pivots.empty:
            no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason="phase3: no M1 data for pivots", side=None)
            return no_trade
        pivot_src = m1_for_pivots.copy()
        pivot_src["time"] = pd.to_datetime(pivot_src["time"], utc=True, errors="coerce")
        pivot_src = pivot_src.dropna(subset=["time"]).sort_values("time")
        pivot_src["ny_day"] = (pivot_src["time"] - pd.Timedelta(hours=22)).dt.date
        prev_day_rows = pivot_src[pivot_src["ny_day"] == prev_ny_day]
        if prev_day_rows.empty:
            no_trade["decision"] = ExecutionDecision(
                attempted=False,
                placed=False,
                reason="phase3: no prior NY-close day data for pivots",
                side=None,
            )
            return no_trade
        pivots = compute_daily_fib_pivots(
            float(prev_day_rows["high"].max()),
            float(prev_day_rows["low"].min()),
            float(prev_day_rows["close"].iloc[-1]),
        )
        state_updates["pivots"] = pivots
        state_updates["pivots_date"] = today_str

    m5_df = _drop_incomplete_tf(data_by_tf.get("M5"), "M5")
    m15_df = _drop_incomplete_tf(data_by_tf.get("M15"), "M15")
    m1_df = _drop_incomplete_tf(data_by_tf.get("M1"), "M1")
    if m5_df is None or m5_df.empty or m15_df is None or m15_df.empty or m1_df is None or m1_df.empty:
        no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason="phase3: missing M1/M5/M15 data", side=None)
        return no_trade

    v14_config = effective_cfg.get("v14", {})
    bb_regime_mode = str(v14_config.get("bb_regime_mode", "percentile")).strip().lower()
    bb_regime_lookback = max(1, int(v14_config.get("bb_width_lookback", support.BB_WIDTH_LOOKBACK)))
    bb_regime_ranging_pct = float(v14_config.get("bb_width_ranging_pct", support.BB_WIDTH_RANGING_PCT))
    bb_required_bars = support.BB_PERIOD + (4 if bb_regime_mode in {"expanding3", "bb_width_expanding3"} else bb_regime_lookback)
    if len(m5_df) < bb_required_bars:
        no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason="phase3: insufficient M5 data for BB", side=None)
        return no_trade

    regime = support.compute_bb_width_regime(
        m5_df,
        mode=bb_regime_mode,
        lookback=bb_regime_lookback,
        ranging_pct=bb_regime_ranging_pct,
    )
    bb_upper, _, bb_lower = _compute_bb(m5_df)
    m5_close = m5_df["close"].astype(float)
    rsi_series = _compute_rsi(m5_close)
    rsi_val = float(rsi_series.iloc[-1]) if pd.notna(rsi_series.iloc[-1]) else 50.0

    if len(m15_df) < support.ATR_PERIOD + 2:
        no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason="phase3: insufficient M15 data for ATR", side=None)
        return no_trade
    atr_series = _compute_atr(m15_df)
    atr_val = float(atr_series.iloc[-1]) if pd.notna(atr_series.iloc[-1]) else 0.0
    adx_val = _compute_adx(m15_df)
    sar_bull, sar_bear = detect_sar_flip(m1_df)

    day_name = now_utc.strftime("%A")
    atr_max_entry = float(v14_config.get("atr_max_threshold_price_units", support.ATR_MAX))
    adx_max_entry = float(v14_config.get("adx_max_for_entry", support.ADX_MAX))
    adx_filter_enabled = bool(v14_config.get("adx_filter_enabled", True))
    adx_day_overrides = v14_config.get("adx_max_by_day", {})
    if isinstance(adx_day_overrides, dict):
        try:
            adx_max_entry = float(adx_day_overrides.get(day_name, adx_max_entry))
        except Exception:
            pass
    core_gate_required = int(v14_config.get("core_gate_required", v14_config.get("min_confluence", support.MIN_CONFLUENCE)))
    min_conf_long = int(v14_config.get("confluence_min_long", core_gate_required))
    min_conf_short = int(v14_config.get("confluence_min_short", core_gate_required))
    blocked_combos_cfg = v14_config.get("blocked_combos", support.BLOCKED_COMBOS)
    if isinstance(blocked_combos_cfg, (list, tuple, set)):
        blocked_combos = {str(x) for x in blocked_combos_cfg}
    elif isinstance(blocked_combos_cfg, str):
        blocked_combos = {x.strip() for x in blocked_combos_cfg.split(",") if x.strip()}
    else:
        blocked_combos = set(support.BLOCKED_COMBOS)
    zone_tolerance_pips = float(v14_config.get("zone_tolerance_pips", support.ZONE_TOLERANCE_PIPS))
    rsi_long_entry = float(v14_config.get("rsi_long_entry", support.RSI_LONG_ENTRY))
    rsi_short_entry = float(v14_config.get("rsi_short_entry", support.RSI_SHORT_ENTRY))
    sl_buffer_pips = float(v14_config.get("sl_buffer_pips", support.SL_BUFFER_PIPS))
    min_sl_pips = float(v14_config.get("min_sl_pips", support.SL_MIN_PIPS))
    max_sl_pips = float(v14_config.get("max_sl_pips", support.SL_MAX_PIPS))
    partial_tp_atr_mult = float(v14_config.get("partial_tp_atr_mult", support.TP1_ATR_MULT))
    partial_tp_min_pips = float(v14_config.get("partial_tp_min_pips", support.TP1_MIN_PIPS))
    partial_tp_max_pips = float(v14_config.get("partial_tp_max_pips", support.TP1_MAX_PIPS))
    core_gate_use_zone = bool(v14_config.get("core_gate_use_zone", True))
    core_gate_use_bb = bool(v14_config.get("core_gate_use_bb", True))
    core_gate_use_sar = bool(v14_config.get("core_gate_use_sar", True))
    core_gate_use_rsi = bool(v14_config.get("core_gate_use_rsi", True))
    confirmation_enabled = bool(v14_config.get("confirmation_enabled", False))
    confirmation_type = str(v14_config.get("confirmation_type", "m1")).lower()
    confirmation_window_bars = max(0, int(v14_config.get("confirmation_window_bars", 0)))
    if not confirmation_enabled:
        confirmation_window_bars = 0
    sst_cfg = v14_config.get("signal_strength_tracking", {}) if isinstance(v14_config.get("signal_strength_tracking"), dict) else {}
    sst_enabled = bool(sst_cfg.get("enabled", False))
    sst_filter_on = bool(sst_cfg.get("filter_on_it", False))
    sst_min_score = int(sst_cfg.get("min_strength_score", 0))
    session_loss_stop_frac = _as_risk_fraction(v14_config.get("session_loss_stop_pct", support.SESSION_LOSS_STOP_PCT), support.SESSION_LOSS_STOP_PCT)
    min_time_between_entries = int(v14_config.get("min_time_between_entries_minutes", v14_config.get("cooldown_minutes", support.COOLDOWN_MINUTES)))
    no_reentry_after_stop_min = int(v14_config.get("no_reentry_same_direction_after_stop_minutes", 0))
    breakout_disable_pips = float(v14_config.get("disable_entries_if_move_from_tokyo_open_range_exceeds_pips", 0.0))
    breakout_mode = str(v14_config.get("breakout_detection_mode", "rolling")).lower()
    breakout_window_min = max(1, int(v14_config.get("rolling_window_minutes", 60)))
    breakout_window_pips = float(v14_config.get("rolling_range_threshold_pips", 40.0))

    if regime != "ranging":
        no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason=f"phase3: regime={regime} (need ranging)", side=None)
        return no_trade
    if adx_filter_enabled and adx_val >= adx_max_entry:
        no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason=f"phase3: ADX={adx_val:.1f} >= {adx_max_entry:.1f}", side=None)
        return no_trade
    if atr_val >= atr_max_entry:
        no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason=f"phase3: ATR={atr_val:.4f} >= {atr_max_entry:.4f}", side=None)
        return no_trade

    max_trades_session = int(v14_config.get("max_trades_per_session", support.MAX_TRADES_PER_SESSION))
    stop_consec_losses = int(v14_config.get("stop_after_consecutive_losses", support.STOP_AFTER_CONSECUTIVE_LOSSES))
    cooldown_min = int(min_time_between_entries)
    max_concurrent = int(v14_config.get("max_concurrent_positions", support.MAX_CONCURRENT))

    session_key = f"session_tokyo_{today_str}"
    session_data = phase3_state.get(session_key, {})
    trade_count = session_data.get("trade_count", 0)
    consecutive_losses = session_data.get("consecutive_losses", 0)
    last_entry_time = session_data.get("last_entry_time")

    if trade_count >= max_trades_session:
        no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason=f"phase3: max trades/session ({trade_count}/{max_trades_session})", side=None)
        return no_trade
    if consecutive_losses >= stop_consec_losses:
        no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason=f"phase3: stopped after {consecutive_losses} consecutive losses", side=None)
        return no_trade

    if last_entry_time is not None:
        try:
            _lt = datetime.fromisoformat(str(last_entry_time))
            if _lt.tzinfo is None:
                _lt = _lt.replace(tzinfo=timezone.utc)
            elapsed = (now_utc - _lt).total_seconds() / 60.0
            if elapsed < cooldown_min:
                no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason=f"phase3: cooldown ({elapsed:.1f}/{cooldown_min}min)", side=None)
                return no_trade
        except Exception:
            pass

    if store is not None and session_loss_stop_frac > 0:
        try:
            profile_name = getattr(profile, "profile_name", str(profile))
            get_by_exit_date = getattr(store, "get_closed_trades_for_exit_date", None)
            if callable(get_by_exit_date):
                day_trades = get_by_exit_date(profile_name, today_str)
            else:
                day_trades = store.get_trades_for_date(profile_name, today_str)
            session_loss_usd = 0.0
            for t in day_trades:
                et = str(t["entry_type"] or "")
                if not et.startswith("phase3:v14"):
                    continue
                p = t["profit"] if "profit" in t.keys() else None
                if p is not None:
                    try:
                        session_loss_usd += float(p)
                    except Exception:
                        pass
            sizing_basis = _account_sizing_value(adapter, fallback=100000.0)
            loss_limit = -abs(float(sizing_basis) * float(session_loss_stop_frac))
            if session_loss_usd <= loss_limit:
                no_trade["decision"] = ExecutionDecision(
                    attempted=False,
                    placed=False,
                    reason=f"phase3: session loss stop ({session_loss_usd:.2f} <= {loss_limit:.2f})",
                    side=None,
                )
                return no_trade
        except Exception:
            pass

    concurrent = phase3_state.get("open_trade_count", 0)
    if concurrent >= max_concurrent:
        no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason=f"phase3: max concurrent ({concurrent}/{max_concurrent})", side=None)
        return no_trade

    if breakout_disable_pips > 0:
        if bool(session_data.get("breakout_blocked", False)):
            no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason="phase3: breakout block active for session", side=None)
            return no_trade
        try:
            m1t = m1_df.copy()
            m1t["time"] = pd.to_datetime(m1t["time"], utc=True, errors="coerce")
            m1t = m1t.dropna(subset=["time"]).sort_values("time")
            if not m1t.empty:
                ts_now = pd.Timestamp(now_utc)
                if ts_now.tzinfo is None:
                    ts_now = ts_now.tz_localize("UTC")
                else:
                    ts_now = ts_now.tz_convert("UTC")
                if breakout_mode == "rolling":
                    w_start = ts_now - pd.Timedelta(minutes=breakout_window_min)
                    w = m1t[m1t["time"] >= w_start]
                    if not w.empty:
                        rng_pips = (float(w["high"].max()) - float(w["low"].min())) / pip
                        if rng_pips >= breakout_window_pips:
                            session_data["breakout_blocked"] = True
                            session_data["breakout_blocked_reason"] = f"rolling {rng_pips:.1f}p >= {breakout_window_pips:.1f}p"
                            state_updates[session_key] = session_data
                            no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason=f"phase3: breakout rolling block ({rng_pips:.1f}p)", side=None)
                            return no_trade
                else:
                    tokyo_start_h = parse_hhmm_to_hour(v14_config.get("session_start_utc", "16:00"), float(support.TOKYO_START_UTC))
                    tokyo_start = ts_now.replace(
                        hour=int(tokyo_start_h),
                        minute=int(round((tokyo_start_h - int(tokyo_start_h)) * 60.0)),
                        second=0,
                        microsecond=0,
                    )
                    if tokyo_start > ts_now:
                        tokyo_start -= pd.Timedelta(days=1)
                    sess = m1t[m1t["time"] >= tokyo_start]
                    if not sess.empty:
                        first = sess.iloc[0]
                        open_mid = (float(first["high"]) + float(first["low"])) / 2.0
                        cur_mid = (float(tick.ask) + float(tick.bid)) / 2.0
                        move_pips = abs(cur_mid - open_mid) / pip
                        if move_pips >= breakout_disable_pips:
                            session_data["breakout_blocked"] = True
                            session_data["breakout_blocked_reason"] = f"anchor {move_pips:.1f}p >= {breakout_disable_pips:.1f}p"
                            state_updates[session_key] = session_data
                            no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason=f"phase3: breakout anchor block ({move_pips:.1f}p)", side=None)
                            return no_trade
        except Exception:
            pass

    pending_v14_raw = phase3_state.get("pending_v14_signals", [])
    pending_v14: list[dict[str, Any]] = []
    now_ts = pd.Timestamp(now_utc).tz_convert("UTC") if pd.Timestamp(now_utc).tzinfo is not None else pd.Timestamp(now_utc, tz="UTC")
    for s in pending_v14_raw if isinstance(pending_v14_raw, list) else []:
        if not isinstance(s, dict):
            continue
        if str(s.get("session_date", "")) != today_str:
            continue
        exp = s.get("expiry_time")
        if not exp:
            continue
        try:
            exp_ts = pd.Timestamp(exp)
            if exp_ts.tzinfo is None:
                exp_ts = exp_ts.tz_localize("UTC")
            else:
                exp_ts = exp_ts.tz_convert("UTC")
        except Exception:
            continue
        if exp_ts >= now_ts:
            pending_v14.append(s)

    if confirmation_window_bars > 0 and pending_v14:
        ref_df = m5_df if confirmation_type == "m5" else m1_df
        if ref_df is not None and not ref_df.empty and "time" in ref_df.columns:
            ref_bar_ts = pd.Timestamp(ref_df["time"].iloc[-1])
            if ref_bar_ts.tzinfo is None:
                ref_bar_ts = ref_bar_ts.tz_localize("UTC")
            else:
                ref_bar_ts = ref_bar_ts.tz_convert("UTC")
            bar_open = float(ref_df["open"].astype(float).iloc[-1])
            bar_close = float(ref_df["close"].astype(float).iloc[-1])
            for sig in list(pending_v14):
                try:
                    sig_side = str(sig.get("side", "")).lower()
                    if sig_side not in {"buy", "sell"}:
                        continue
                    sig_bar_ts = pd.Timestamp(sig.get("bar_time"))
                    if sig_bar_ts.tzinfo is None:
                        sig_bar_ts = sig_bar_ts.tz_localize("UTC")
                    else:
                        sig_bar_ts = sig_bar_ts.tz_convert("UTC")
                    if ref_bar_ts <= sig_bar_ts:
                        continue
                    confirmed = (bar_close > bar_open) if sig_side == "buy" else (bar_close < bar_open)
                    if not confirmed:
                        continue
                    sig_sl = float(sig.get("sl_price"))
                    sig_tp1 = float(sig.get("tp1_price"))
                    sig_units = int(sig.get("units"))
                    if sig_units <= 0:
                        continue
                    strategy_tag = "phase3:v14_mean_reversion"
                    _qv14_cell = str(ownership_audit.get("ownership_cell") or "") or None
                    if _qv14_cell:
                        strategy_tag = f"{strategy_tag}@{_qv14_cell}"
                    comment = f"phase3_integrated:{policy.id}:v14_mean_reversion"
                    try:
                        adapter.place_order(
                            symbol=profile.symbol,
                            side=sig_side,
                            lots=sig_units / 100_000.0,
                            stop_price=round(sig_sl, 3),
                            target_price=round(sig_tp1, 3),
                            comment=comment,
                        )
                    except Exception as e:
                        return {
                            "decision": ExecutionDecision(attempted=True, placed=False, reason=f"phase3: queued order error: {e}", side=sig_side),
                            "phase3_state_updates": state_updates,
                            "strategy_tag": strategy_tag,
                        }
                    pending_v14.remove(sig)
                    session_data["trade_count"] = trade_count + 1
                    session_data["last_entry_time"] = now_utc.isoformat()
                    state_updates[session_key] = session_data
                    state_updates["pending_v14_signals"] = pending_v14
                    return {
                        "decision": ExecutionDecision(
                            attempted=True,
                            placed=True,
                            reason=f"phase3:v14 queued {sig_side.upper()} confirmed confluence={sig.get('score')} combo={sig.get('combo')}",
                            side=sig_side,
                        ),
                        "phase3_state_updates": state_updates,
                        "strategy_tag": strategy_tag,
                        "sl_price": sig_sl,
                        "tp1_price": sig_tp1,
                        "units": sig_units,
                        "entry_price": tick.ask if sig_side == "buy" else tick.bid,
                        "sl_pips": abs((tick.ask if sig_side == "buy" else tick.bid) - sig_sl) / pip,
                    }
                except Exception:
                    continue

    close_price = float(m5_close.iloc[-1])
    high_price = float(m5_df["high"].astype(float).iloc[-1])
    low_price = float(m5_df["low"].astype(float).iloc[-1])
    best_side = None
    best_score = 0
    best_combo = ""
    for side in ("buy", "sell"):
        score, combo = evaluate_v14_confluence(
            side, close_price, high_price, low_price, pivots, bb_upper, bb_lower, sar_bull, sar_bear, rsi_val, pip,
            zone_tolerance_pips=zone_tolerance_pips,
            rsi_long_entry=rsi_long_entry,
            rsi_short_entry=rsi_short_entry,
            core_gate_use_zone=core_gate_use_zone,
            core_gate_use_bb=core_gate_use_bb,
            core_gate_use_sar=core_gate_use_sar,
            core_gate_use_rsi=core_gate_use_rsi,
        )
        min_conf = min_conf_long if side == "buy" else min_conf_short
        if score >= min_conf and combo not in blocked_combos and score > best_score:
            best_side = side
            best_score = score
            best_combo = combo

    if best_side is None:
        no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason=f"phase3: no qualifying confluence (RSI={rsi_val:.1f})", side=None)
        if confirmation_window_bars > 0:
            state_updates["pending_v14_signals"] = pending_v14
        return no_trade

    if no_reentry_after_stop_min > 0:
        stop_key = f"last_stopout_time_{best_side}"
        last_stop = session_data.get(stop_key)
        if last_stop:
            try:
                stop_dt = datetime.fromisoformat(str(last_stop))
                if stop_dt.tzinfo is None:
                    stop_dt = stop_dt.replace(tzinfo=timezone.utc)
                elapsed_stop = (now_utc - stop_dt).total_seconds() / 60.0
                if elapsed_stop < float(no_reentry_after_stop_min):
                    no_trade["decision"] = ExecutionDecision(
                        attempted=False,
                        placed=False,
                        reason=f"phase3: no-reentry {best_side} after stop ({elapsed_stop:.1f}/{no_reentry_after_stop_min}m)",
                        side=None,
                    )
                    return no_trade
            except Exception:
                pass

    strength_info = compute_v14_signal_strength(
        side=best_side,
        confluence_score=int(best_score),
        close=close_price,
        high=high_price,
        low=low_price,
        pivots=pivots,
        bb_upper=bb_upper,
        bb_lower=bb_lower,
        rsi=rsi_val,
        sar_bullish_flip=sar_bull,
        sar_bearish_flip=sar_bear,
        now_utc=now_utc,
        pip_size=pip,
        sst_cfg=sst_cfg if sst_enabled else {},
    ) if sst_enabled else {"score": 0, "bucket": "moderate", "size_mult": 1.0}
    if sst_enabled and sst_filter_on and int(strength_info.get("score", 0)) < int(sst_min_score):
        no_trade["decision"] = ExecutionDecision(
            attempted=False,
            placed=False,
            reason=f"phase3: signal-strength below min ({strength_info.get('score', 0)} < {sst_min_score})",
            side=None,
        )
        return no_trade

    entry_price = tick.ask if best_side == "buy" else tick.bid
    sl_price = compute_v14_sl(best_side, entry_price, pivots, pip, sl_buffer_pips=sl_buffer_pips, min_sl_pips=min_sl_pips, max_sl_pips=max_sl_pips)
    sl_pips = abs(entry_price - sl_price) / pip
    tp1_price = compute_v14_tp1(best_side, entry_price, atr_val, pip, partial_tp_atr_mult=partial_tp_atr_mult, partial_tp_min_pips=partial_tp_min_pips, partial_tp_max_pips=partial_tp_max_pips)

    equity = _account_sizing_value(adapter, fallback=100_000.0)
    if v14_config:
        units = compute_v14_units_from_config(equity, sl_pips, entry_price, pip, now_utc, v14_config)
    else:
        units = compute_v14_lot_size(equity, sl_pips, entry_price, pip, support.LEVERAGE)
    if units <= 0:
        no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason=f"phase3: lot size 0 (equity={equity:.0f}, sl_pips={sl_pips:.1f})", side=None)
        return no_trade

    size_mult = float(strength_info.get("size_mult", 1.0))
    if size_mult > 0 and size_mult != 1.0:
        units = int(max(0, math.floor(float(units) * size_mult)))
    if units <= 0:
        no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason="phase3: lot size 0 after strength multiplier", side=None)
        return no_trade

    _cs_cell = str(ownership_audit.get("ownership_cell") or "") or None
    units, _cell_scale_applied, _cs_key = resolve_v14_cell_scale_override_from_state(
        ownership_cell=_cs_cell,
        side=best_side,
        units=units,
        overlay_state=overlay_state,
    )
    if _cs_key:
        print(f"[phase3] V14 cell-scale override: {_cs_key} -> scale={_cell_scale_applied} units={units}")
        if units <= 0:
            no_trade["decision"] = ExecutionDecision(
                attempted=False,
                placed=False,
                reason=f"phase3: lot size 0 after cell-scale ({_cs_key}={_cell_scale_applied})",
                side=None,
            )
            return no_trade

    state_updates["last_v14_signal_strength"] = {
        "time": now_utc.isoformat(),
        "side": best_side,
        "confluence_score": int(best_score),
        "confluence_combo": str(best_combo),
        "signal_strength_score": int(strength_info.get("score", 0)),
        "signal_strength_bucket": str(strength_info.get("bucket", "moderate")),
        "signal_strength_mult": float(strength_info.get("size_mult", 1.0)),
    }

    if confirmation_window_bars > 0:
        same_side_pending = any(str(s.get("side", "")).lower() == best_side for s in pending_v14)
        if not same_side_pending:
            window_minutes = confirmation_window_bars if confirmation_type == "m1" else confirmation_window_bars * 5
            pending_v14.append(
                {
                    "side": best_side,
                    "bar_time": now_utc.isoformat(),
                    "expiry_time": (now_utc + pd.Timedelta(minutes=window_minutes)).isoformat(),
                    "session_date": today_str,
                    "sl_price": float(sl_price),
                    "tp1_price": float(tp1_price),
                    "units": int(units),
                    "score": int(best_score),
                    "combo": str(best_combo),
                    "signal_strength_score": int(strength_info.get("score", 0)),
                    "signal_strength_bucket": str(strength_info.get("bucket", "moderate")),
                    "signal_strength_mult": float(strength_info.get("size_mult", 1.0)),
                }
            )
        state_updates["pending_v14_signals"] = pending_v14
        return {
            "decision": ExecutionDecision(
                attempted=False,
                placed=False,
                reason=f"v14 signal queued {best_side} conf={best_score} combo={best_combo} sst={strength_info.get('score', 0)}({strength_info.get('bucket', 'moderate')})",
                side=None,
            ),
            "phase3_state_updates": state_updates,
            "strategy_tag": "phase3:v14_pending",
        }

    strategy_tag = "phase3:v14_mean_reversion"
    _v14_cell_str = str(ownership_audit.get("ownership_cell") or "") or None
    if _v14_cell_str:
        strategy_tag = f"{strategy_tag}@{_v14_cell_str}"
    comment = f"phase3_integrated:{policy.id}:v14_mean_reversion"

    try:
        order_result = adapter.place_order(
            symbol=profile.symbol,
            side=best_side,
            lots=units / 100_000.0,
            stop_price=round(sl_price, 3),
            target_price=round(tp1_price, 3),
            comment=comment,
        )
    except Exception as e:
        return {
            "decision": ExecutionDecision(attempted=True, placed=False, reason=f"phase3: order error: {e}", side=best_side),
            "phase3_state_updates": state_updates,
            "strategy_tag": strategy_tag,
        }

    confirmed_fill, confirmed_deal_id = _phase3_order_confirmed(adapter, profile, order_result)
    if not confirmed_fill:
        return {
            "decision": ExecutionDecision(
                attempted=True,
                placed=False,
                reason="phase3: broker order pending/unfilled",
                side=best_side,
                order_retcode=getattr(order_result, "order_retcode", None),
                order_id=getattr(order_result, "order_id", None),
                deal_id=getattr(order_result, "deal_id", None),
                fill_price=getattr(order_result, "fill_price", None),
            ),
            "phase3_state_updates": state_updates,
            "strategy_tag": strategy_tag,
        }

    session_data["trade_count"] = trade_count + 1
    session_data["last_entry_time"] = now_utc.isoformat()
    state_updates[session_key] = session_data

    return {
        "decision": ExecutionDecision(
            attempted=True,
            placed=True,
            reason=f"phase3:v14 {best_side.upper()} confluence={best_score} combo={best_combo} SL={sl_price:.3f}({sl_pips:.1f}p) TP1={tp1_price:.3f} units={units} sst={strength_info.get('score', 0)}({strength_info.get('bucket', 'moderate')})",
            side=best_side,
            order_retcode=getattr(order_result, "order_retcode", None),
            order_id=getattr(order_result, "order_id", None),
            deal_id=confirmed_deal_id if confirmed_deal_id is not None else getattr(order_result, "deal_id", None),
            fill_price=getattr(order_result, "fill_price", None),
        ),
        "phase3_state_updates": state_updates,
        "strategy_tag": strategy_tag,
        "sl_price": sl_price,
        "tp1_price": tp1_price,
        "units": units,
        "entry_price": entry_price,
        "sl_pips": sl_pips,
        "pivots": pivots,
        "atr_val": atr_val,
    }
