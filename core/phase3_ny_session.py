from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd


def execute_v44_ny_session(
    *,
    adapter,
    profile,
    policy,
    data_by_tf: dict,
    tick,
    phase3_state: dict,
    sizing_config: Optional[dict[str, Any]] = None,
    now_utc: Optional[datetime] = None,
    store=None,
    ownership_audit: Optional[dict[str, Any]] = None,
    overlay_state: Optional[dict[str, Any]] = None,
    support,
) -> dict:
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    ExecutionDecision = support.ExecutionDecision
    _drop_incomplete_tf = support._drop_incomplete_tf
    resolve_ny_window_hours = support.resolve_ny_window_hours
    _as_risk_fraction = support._as_risk_fraction
    _compute_ema = support._compute_ema
    _compute_adx = support._compute_adx
    _determine_v44_session_mode = support._determine_v44_session_mode
    _compute_v44_atr_rank = support._compute_v44_atr_rank
    _load_news_events_cached = support._load_news_events_cached
    is_in_news_window = support.is_in_news_window
    _v44_most_recent_news_event = support._v44_most_recent_news_event
    _v44_news_trend_active_event = support._v44_news_trend_active_event
    _account_sizing_value = support._account_sizing_value
    compute_v44_h1_trend = support.compute_v44_h1_trend
    compute_v44_sl = support.compute_v44_sl
    evaluate_v44_entry = support.evaluate_v44_entry
    v44_defensive_veto_block_from_state = support.v44_defensive_veto_block_from_state
    _phase3_order_confirmed = support._phase3_order_confirmed

    pip = float(profile.pip_size) if hasattr(profile, "pip_size") else support.PIP_SIZE
    no_trade = {
        "decision": ExecutionDecision(attempted=False, placed=False, reason="", side=None),
        "phase3_state_updates": {},
        "strategy_tag": None,
    }

    v44_config = (sizing_config or {}).get("v44_ny", {})
    v44_start_delay_min = int(v44_config.get("start_delay_minutes", 5))
    v44_max_entry_spread = float(v44_config.get("max_entry_spread_pips", support.V44_MAX_ENTRY_SPREAD))
    v44_ny_start_hour, v44_ny_end_hour = resolve_ny_window_hours(now_utc, v44_config)

    _ts_now = pd.Timestamp(now_utc)
    if _ts_now.tzinfo is None:
        _ts_now = _ts_now.tz_localize("UTC")
    else:
        _ts_now = _ts_now.tz_convert("UTC")
    _day0 = _ts_now.normalize()
    ny_start_ts = _day0 + pd.Timedelta(hours=v44_ny_start_hour) + pd.Timedelta(minutes=max(0, v44_start_delay_min))
    ny_end_ts = _day0 + pd.Timedelta(hours=v44_ny_end_hour)

    if _ts_now < ny_start_ts:
        no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason="v44_ny: start delay active", side=None)
        return no_trade
    if (tick.ask - tick.bid) / pip > v44_max_entry_spread:
        no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason="v44_ny: spread veto", side=None)
        return no_trade

    m1_df = _drop_incomplete_tf(data_by_tf.get("M1"), "M1")
    h1_df = _drop_incomplete_tf(data_by_tf.get("H1"), "H1")
    m5_df = _drop_incomplete_tf(data_by_tf.get("M5"), "M5")
    h4_df = _drop_incomplete_tf(data_by_tf.get("H4"), "H4")
    if h1_df is None or h1_df.empty or m5_df is None or m5_df.empty or m1_df is None or m1_df.empty:
        no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason="v44_ny: missing M1/H1/M5", side=None)
        return no_trade

    v44_max_open = int(v44_config.get("max_open_positions", support.V44_MAX_OPEN))
    v44_risk_pct = _as_risk_fraction(v44_config.get("risk_per_trade_pct", 0.5), 0.005)
    v44_rp_min_lot = float(v44_config.get("rp_min_lot", 1.0))
    v44_rp_max_lot = float(v44_config.get("rp_max_lot", 20.0))
    v44_max_entries_day = int(v44_config.get("max_entries_per_day", support.V44_MAX_ENTRIES_DAY))
    v44_session_stop_losses = int(v44_config.get("session_stop_losses", support.V44_SESSION_STOP_LOSSES))
    v44_ny_strength_allow = str(v44_config.get("ny_strength_allow", v44_config.get("strength_allow", "strong_normal"))).lower()
    v44_strong_tp1_pips = float(v44_config.get("strong_tp1_pips", support.V44_STRONG_TP1_PIPS))
    v44_normal_tp1_pips = float(v44_config.get("normal_tp1_pips", 1.75))
    v44_weak_tp1_pips = float(v44_config.get("weak_tp1_pips", 1.2))
    v44_h1_ema_fast = int(v44_config.get("h1_ema_fast", support.V44_H1_EMA_FAST))
    v44_h1_ema_slow = int(v44_config.get("h1_ema_slow", support.V44_H1_EMA_SLOW))
    v44_m5_ema_fast = int(v44_config.get("m5_ema_fast", support.V44_M5_EMA_FAST))
    v44_m5_ema_slow = int(v44_config.get("m5_ema_slow", support.V44_M5_EMA_SLOW))
    v44_slope_bars = int(v44_config.get("slope_bars", support.V44_SLOPE_BARS))
    v44_strong_slope = float(v44_config.get("strong_slope", support.V44_STRONG_SLOPE))
    v44_weak_slope = float(v44_config.get("weak_slope", support.V44_WEAK_SLOPE))
    v44_min_body_pips = float(v44_config.get("entry_min_body_pips", support.V44_MIN_BODY_PIPS))
    v44_atr_pct_filter_enabled = bool(v44_config.get("atr_pct_filter_enabled", True))
    v44_atr_pct_cap = float(v44_config.get("atr_pct_cap", support.V44_ATR_PCT_CAP))
    v44_atr_pct_lookback = int(v44_config.get("atr_pct_lookback", support.V44_ATR_PCT_LOOKBACK))
    v44_entry_cutoff_minutes = int(v44_config.get("session_entry_cutoff_minutes", 60))
    v44_news_filter_enabled = bool(v44_config.get("news_filter_enabled", False))
    v44_news_before = int(v44_config.get("news_window_minutes_before", 60))
    v44_news_after = int(v44_config.get("news_window_minutes_after", 30))
    v44_news_impact_min = str(v44_config.get("news_impact_min", "high"))
    v44_news_calendar_path = str(v44_config.get("news_calendar_path", "research_out/v5_scheduled_events_utc.csv"))
    v44_skip_days_raw = str(v44_config.get("skip_days", "")).strip()
    v44_skip_months_raw = str(v44_config.get("skip_months", "")).strip()
    v44_skip_weak = bool(v44_config.get("skip_weak", True))
    v44_skip_normal = bool(v44_config.get("skip_normal", False))
    v44_allow_normal_plus = bool(v44_config.get("allow_normal_plus", False))
    v44_normalplus_atr_min_pips = float(v44_config.get("normalplus_atr_min_pips", 6.0))
    v44_session_range_cap_pips = float(v44_config.get("session_range_cap_pips", 0.0))
    v44_h1_slope_min = float(v44_config.get("h1_slope_min", 0.0))
    v44_h1_slope_consistent_bars = int(v44_config.get("h1_slope_consistent_bars", 0))
    v44_dual_mode_enabled = bool(v44_config.get("dual_mode_enabled", False))
    v44_trend_mode_efficiency_min = float(v44_config.get("trend_mode_efficiency_min", 0.4))
    v44_range_mode_efficiency_max = float(v44_config.get("range_mode_efficiency_max", 0.3))
    v44_range_fade_enabled = bool(v44_config.get("range_fade_enabled", False))
    v44_queued_confirm_bars = max(0, int(v44_config.get("queued_confirm_bars", 0)))
    v44_gl_enabled = bool(v44_config.get("gl_enabled", False))
    v44_gl_min_wins = max(1, int(v44_config.get("gl_min_wins", 1)))
    v44_gl_extra_entries = max(0, int(v44_config.get("gl_extra_entries", 0)))
    v44_gl_allow_normal = bool(v44_config.get("gl_allow_normal", True))
    v44_gl_normal_atr_cap = float(v44_config.get("gl_normal_atr_cap", 0.67))
    v44_gl_slope_relax = float(v44_config.get("gl_slope_relax", 0.0))
    v44_gl_max_open = int(v44_config.get("gl_max_open", 0))
    v44_news_trend_enabled = bool(v44_config.get("news_trend_enabled", False))
    v44_news_trend_delay_minutes = int(v44_config.get("news_trend_delay_minutes", 45))
    v44_news_trend_window_minutes = int(v44_config.get("news_trend_window_minutes", 90))
    v44_news_trend_confirm_bars = max(1, int(v44_config.get("news_trend_confirm_bars", 3)))
    v44_news_trend_min_body_pips = float(v44_config.get("news_trend_min_body_pips", 1.5))
    v44_news_trend_require_ema_align = bool(v44_config.get("news_trend_require_ema_align", True))
    v44_news_trend_risk_pct = _as_risk_fraction(v44_config.get("news_trend_risk_pct", 0.5), 0.005)
    v44_news_trend_tp_rr = float(v44_config.get("news_trend_tp_rr", 1.5))
    v44_news_trend_sl_pips = float(v44_config.get("news_trend_sl_pips", 8.0))
    v44_h4_adx_min = float(v44_config.get("h4_adx_min", 0.0))
    v44_exhaustion_gate_enabled = bool(v44_config.get("exhaustion_gate_enabled", False))
    v44_exhaustion_window_minutes = max(5, int(v44_config.get("exhaustion_gate_window_minutes", 60)))
    v44_exhaustion_max_range_pips = float(v44_config.get("exhaustion_gate_max_range_pips", 40.0))
    v44_exhaustion_cooldown_minutes = max(1, int(v44_config.get("exhaustion_gate_cooldown_minutes", 15)))
    v44_regime_block_enabled = bool(v44_config.get("regime_block_enabled", True))
    v44_daily_loss_limit_pips = float(v44_config.get("daily_loss_limit_pips", 0.0))
    v44_weekly_loss_limit_pips = float(v44_config.get("weekly_loss_limit_pips", 0.0))
    v44_daily_loss_limit_usd = float(v44_config.get("daily_loss_limit_usd", 0.0))
    v44_sizing_mode = str(v44_config.get("sizing_mode", "risk_parity")).lower()
    v44_base_lot = float(v44_config.get("base_lot", 2.0))
    v44_str_size_mults = {
        "strong": float(v44_config.get("strong_size_mult", 3.0)),
        "normal": float(v44_config.get("normal_size_mult", 2.0)),
        "weak": float(v44_config.get("weak_size_mult", 0.0)),
    }
    v44_win_bonus_step = float(v44_config.get("win_bonus_per_step", 0.0))
    v44_win_streak_scope = str(v44_config.get("win_streak_scope", "session")).lower()
    v44_max_lot = float(v44_config.get("max_lot", 10.0))
    v44_rp_str_mults = {
        "strong": float(v44_config.get("rp_strong_mult", 1.0)),
        "normal": float(v44_config.get("rp_normal_mult", 1.0)),
        "weak": float(v44_config.get("rp_weak_mult", 0.0)),
    }
    v44_rp_win_bonus_pct = float(v44_config.get("rp_win_bonus_pct", 0.0))
    v44_rp_max_lot_mult = float(v44_config.get("rp_max_lot_mult", 2.0))
    v44_rp_ny_mult = float(v44_config.get("rp_ny_mult", 1.0))
    v44_hyb_str_mults = {
        "strong": float(v44_config.get("hybrid_strong_boost", 1.3)),
        "normal": float(v44_config.get("hybrid_normal_boost", 0.8)),
        "weak": float(v44_config.get("hybrid_weak_boost", 0.0)),
    }
    v44_hybrid_ny_boost = float(v44_config.get("hybrid_ny_boost", 1.0))

    today = now_utc.date().isoformat()
    prev_day = str(phase3_state.get("ny_last_day") or "")
    if prev_day and prev_day != today:
        prev_key = f"session_ny_{prev_day}"
        prev_sdat = phase3_state.get(prev_key, {})
        if isinstance(prev_sdat, dict):
            try:
                o = float(prev_sdat.get("session_open_price"))
                c = float(prev_sdat.get("session_close_price"))
                hi = float(prev_sdat.get("session_high"))
                lo = float(prev_sdat.get("session_low"))
                rng = hi - lo
                if rng > 0:
                    eff = abs(c - o) / rng
                    hist = list(phase3_state.get("ny_session_efficiency_history", []))
                    hist.append(float(eff))
                    phase3_state["ny_session_efficiency_history"] = hist[-50:]
            except Exception:
                pass
    phase3_state["ny_last_day"] = today

    session_key = f"session_ny_{today}"
    sdat = dict(phase3_state.get(session_key, {}))
    sdat.setdefault("trade_count", 0)
    sdat.setdefault("consecutive_losses", 0)
    sdat.setdefault("wins_closed", 0)
    sdat.setdefault("cooldown_until", None)
    sdat.setdefault("last_entry_time", None)
    sdat.setdefault("exhaustion_cooldown_until", None)
    sdat.setdefault("win_streak", 0)
    sdat.setdefault("session_open_price", None)
    sdat.setdefault("session_high", None)
    sdat.setdefault("session_low", None)
    sdat.setdefault("session_close_price", None)
    sdat.setdefault("session_efficiency_history", list(phase3_state.get("ny_session_efficiency_history", [])))
    sdat.setdefault("queued_pending", None)
    sdat.setdefault("queued_stats", {"generated": 0, "confirmed": 0, "expired": 0, "replaced": 0})
    sdat.setdefault("news_trend_state", {})

    m1_last = m1_df.iloc[-1]
    c_last = float(m1_last["close"])
    h_last = float(m1_last["high"])
    l_last = float(m1_last["low"])
    if sdat.get("session_open_price") is None:
        sdat["session_open_price"] = c_last
        sdat["session_high"] = h_last
        sdat["session_low"] = l_last
        sdat["session_close_price"] = c_last
    else:
        try:
            sdat["session_high"] = max(float(sdat.get("session_high", h_last)), h_last)
            sdat["session_low"] = min(float(sdat.get("session_low", l_last)), l_last)
            sdat["session_close_price"] = c_last
        except Exception:
            sdat["session_high"] = h_last
            sdat["session_low"] = l_last
            sdat["session_close_price"] = c_last

    if float(v44_session_range_cap_pips) > 0:
        try:
            sess_range_pips = (float(sdat.get("session_high")) - float(sdat.get("session_low"))) / pip
            if sess_range_pips > float(v44_session_range_cap_pips):
                no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason=f"v44_ny: session range cap ({sess_range_pips:.1f}p > {v44_session_range_cap_pips:.1f}p)", side=None)
                no_trade["phase3_state_updates"] = {session_key: sdat}
                return no_trade
        except Exception:
            pass

    session_mode = _determine_v44_session_mode(
        sdat,
        dual_mode_enabled=v44_dual_mode_enabled,
        trend_mode_efficiency_min=v44_trend_mode_efficiency_min,
        range_mode_efficiency_max=v44_range_mode_efficiency_max,
    )
    _sess_hist_vals = [float(x) for x in sdat.get("session_efficiency_history", []) if isinstance(x, (int, float))]
    _sess_eff_avg = (sum(_sess_hist_vals) / len(_sess_hist_vals)) if _sess_hist_vals else None
    sdat["session_mode"] = session_mode
    sdat["session_efficiency_avg"] = float(_sess_eff_avg) if _sess_eff_avg is not None else None
    sdat["session_efficiency_hist_len"] = int(len(_sess_hist_vals))
    sdat["range_fade_enabled"] = bool(v44_range_fade_enabled)
    sdat["trend_mode_efficiency_min"] = float(v44_trend_mode_efficiency_min)
    sdat["range_mode_efficiency_max"] = float(v44_range_mode_efficiency_max)
    if v44_dual_mode_enabled and session_mode == "neutral":
        no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason="v44_ny: dual mode neutral", side=None)
        no_trade["phase3_state_updates"] = {session_key: sdat}
        return no_trade
    if v44_dual_mode_enabled and session_mode == "range_fade" and not v44_range_fade_enabled:
        no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason="v44_ny: range mode disabled", side=None)
        no_trade["phase3_state_updates"] = {session_key: sdat}
        return no_trade

    open_count = int(phase3_state.get("open_trade_count", 0))
    green_light_active = bool(v44_gl_enabled and int(sdat.get("wins_closed", 0)) >= int(v44_gl_min_wins))
    effective_max_open = int(v44_max_open)
    if green_light_active and int(v44_gl_max_open) > effective_max_open:
        effective_max_open = int(v44_gl_max_open)
    if open_count >= effective_max_open:
        no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason=f"v44_ny: max open {open_count}/{effective_max_open}", side=None)
        no_trade["phase3_state_updates"] = {session_key: sdat}
        return no_trade

    if v44_skip_days_raw:
        try:
            skip_days = {int(x.strip()) for x in v44_skip_days_raw.split(",") if str(x).strip() != ""}
        except Exception:
            skip_days = set()
        if now_utc.weekday() in skip_days:
            no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason="v44_ny: skip day", side=None)
            return no_trade
    if v44_skip_months_raw:
        try:
            skip_months = {int(x.strip()) for x in v44_skip_months_raw.split(",") if str(x).strip() != ""}
        except Exception:
            skip_months = set()
        if now_utc.month in skip_months:
            no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason="v44_ny: skip month", side=None)
            return no_trade

    if store is not None and (v44_daily_loss_limit_pips > 0 or v44_weekly_loss_limit_pips > 0 or v44_daily_loss_limit_usd > 0):
        try:
            profile_name = getattr(profile, "profile_name", str(profile))
            get_by_exit_date = getattr(store, "get_closed_trades_for_exit_date", None)
            day_trades = None
            if v44_daily_loss_limit_pips > 0 or v44_daily_loss_limit_usd > 0:
                if callable(get_by_exit_date):
                    day_trades = get_by_exit_date(profile_name, today)
                else:
                    day_trades = store.get_trades_for_date(profile_name, today)
            if day_trades is not None:
                day_pips = sum(float(t["pips"] or 0) for t in day_trades if str(t["entry_type"] or "").startswith("phase3:v44") and t["pips"] is not None)
                day_usd = sum(float(t["profit"] or 0) for t in day_trades if str(t["entry_type"] or "").startswith("phase3:v44") and t["profit"] is not None)
                if v44_daily_loss_limit_pips > 0 and day_pips <= -v44_daily_loss_limit_pips:
                    no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason=f"v44_ny: daily loss limit ({day_pips:.1f}p <= -{v44_daily_loss_limit_pips:.0f}p)", side=None)
                    return no_trade
                if v44_daily_loss_limit_usd > 0 and day_usd <= -v44_daily_loss_limit_usd:
                    no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason=f"v44_ny: daily loss USD ({day_usd:.2f} <= -{v44_daily_loss_limit_usd:.2f})", side=None)
                    return no_trade
            if v44_weekly_loss_limit_pips > 0:
                wk_start = now_utc.date() - pd.Timedelta(days=now_utc.weekday())
                week_pips = 0.0
                for d_offset in range(7):
                    d_str = (wk_start + pd.Timedelta(days=d_offset)).isoformat()
                    if d_str > today:
                        break
                    if callable(get_by_exit_date):
                        wk_trades = get_by_exit_date(profile_name, d_str)
                    else:
                        wk_trades = store.get_trades_for_date(profile_name, d_str)
                    week_pips += sum(float(t["pips"] or 0) for t in wk_trades if str(t["entry_type"] or "").startswith("phase3:v44") and t["pips"] is not None)
                if week_pips <= -v44_weekly_loss_limit_pips:
                    no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason=f"v44_ny: weekly loss limit ({week_pips:.1f}p <= -{v44_weekly_loss_limit_pips:.0f}p)", side=None)
                    return no_trade
        except Exception:
            pass

    if v44_h4_adx_min > 0 and h4_df is not None and len(h4_df) >= 14:
        h4_adx_val = _compute_adx(h4_df)
        if h4_adx_val < v44_h4_adx_min:
            no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason=f"v44_ny: H4 ADX {h4_adx_val:.1f} < min {v44_h4_adx_min:.1f}", side=None)
            return no_trade

    if v44_h1_slope_min > 0 and h1_df is not None and len(h1_df) >= max(2, v44_slope_bars + 1):
        _h1_close = h1_df["close"].astype(float)
        _h1_ema = _compute_ema(_h1_close, period=v44_h1_ema_fast)
        _sb = max(1, int(v44_slope_bars))
        if len(_h1_ema) > _sb:
            _h1_slope_mag = abs((float(_h1_ema.iloc[-1]) - float(_h1_ema.iloc[-1 - _sb])) / (float(_sb) * pip))
            if _h1_slope_mag < float(v44_h1_slope_min):
                no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason=f"v44_ny: H1 slope weak ({_h1_slope_mag:.2f} < {v44_h1_slope_min:.2f})", side=None)
                no_trade["phase3_state_updates"] = {session_key: sdat}
                return no_trade
    if v44_h1_slope_consistent_bars > 0 and h1_df is not None and len(h1_df) >= int(v44_h1_slope_consistent_bars) + 2:
        _h1_close = h1_df["close"].astype(float)
        _h1_fast = _compute_ema(_h1_close, period=v44_h1_ema_fast)
        _h1_slow = _compute_ema(_h1_close, period=v44_h1_ema_slow)
        trend_side = None
        if float(_h1_fast.iloc[-1]) > float(_h1_slow.iloc[-1]):
            trend_side = "buy"
        elif float(_h1_fast.iloc[-1]) < float(_h1_slow.iloc[-1]):
            trend_side = "sell"
        if trend_side is not None:
            consistent = True
            n = int(v44_h1_slope_consistent_bars)
            for k in range(1, n + 1):
                e_now = float(_h1_fast.iloc[-k])
                e_prev = float(_h1_fast.iloc[-k - 1])
                bar_slope = e_now - e_prev
                if (trend_side == "buy" and bar_slope <= 0) or (trend_side == "sell" and bar_slope >= 0):
                    consistent = False
                    break
            if not consistent:
                no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason="v44_ny: H1 slope inconsistent", side=None)
                no_trade["phase3_state_updates"] = {session_key: sdat}
                return no_trade

    if v44_exhaustion_gate_enabled:
        exh_cooldown = sdat.get("exhaustion_cooldown_until")
        if exh_cooldown is not None:
            try:
                exh_cd_dt = pd.Timestamp(exh_cooldown).tz_localize("UTC") if pd.Timestamp(exh_cooldown).tzinfo is None else pd.Timestamp(exh_cooldown).tz_convert("UTC")
                if pd.Timestamp(now_utc) < exh_cd_dt:
                    no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason=f"v44_ny: exhaustion gate cooldown until {exh_cooldown}", side=None)
                    return no_trade
            except Exception:
                pass
        if m5_df is not None and not m5_df.empty and "time" in m5_df.columns:
            try:
                m5_times = pd.to_datetime(m5_df["time"])
                if m5_times.dt.tz is None:
                    m5_times = m5_times.dt.tz_localize("UTC")
                cutoff_ts = pd.Timestamp(now_utc).tz_convert("UTC") - pd.Timedelta(minutes=v44_exhaustion_window_minutes)
                m5_win = m5_df[m5_times >= cutoff_ts]
            except Exception:
                m5_win = m5_df.tail(max(1, v44_exhaustion_window_minutes // 5))
            if not m5_win.empty:
                ex_range_pips = (float(m5_win["high"].max()) - float(m5_win["low"].min())) / pip
                if ex_range_pips > v44_exhaustion_max_range_pips:
                    cd_until = (now_utc + pd.Timedelta(minutes=v44_exhaustion_cooldown_minutes)).isoformat()
                    sdat["exhaustion_cooldown_until"] = cd_until
                    no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason=f"v44_ny: exhaustion gate ({ex_range_pips:.1f}p > {v44_exhaustion_max_range_pips:.0f}p)", side=None)
                    no_trade["phase3_state_updates"] = {session_key: sdat}
                    return no_trade

    if _ts_now >= (ny_end_ts - pd.Timedelta(minutes=max(0, v44_entry_cutoff_minutes))):
        no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason=f"v44_ny: entry cutoff ({v44_entry_cutoff_minutes}m before session end)", side=None)
        no_trade["phase3_state_updates"] = {session_key: sdat}
        return no_trade

    news_events: tuple[pd.Timestamp, ...] = tuple()
    in_news_window = False
    sdat["news_filter_enabled"] = int(v44_news_filter_enabled)
    sdat["news_trend_enabled"] = int(v44_news_trend_enabled)
    sdat["news_calendar_path"] = v44_news_calendar_path
    sdat["news_status"] = "disabled"
    sdat["news_event_time"] = None
    sdat["news_wait_minutes"] = None
    sdat["news_confirm_progress"] = None
    sdat["news_trend_side"] = None
    if v44_news_filter_enabled:
        news_events = _load_news_events_cached(v44_news_calendar_path, v44_news_impact_min)
        sdat["news_events_loaded"] = int(len(news_events))
        in_news_window = bool(is_in_news_window(now_utc, news_events, v44_news_before, v44_news_after))
        latest_ev = _v44_most_recent_news_event(now_utc, news_events)
        if latest_ev is not None:
            sdat["news_event_time"] = latest_ev.isoformat()
        sdat["news_status"] = "in_window" if in_news_window else "clear"
        if in_news_window and not v44_news_trend_enabled:
            no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason="v44_ny: news window block", side=None)
            no_trade["phase3_state_updates"] = {session_key: sdat}
            return no_trade

    effective_max_entries_day = int(v44_max_entries_day) + (int(v44_gl_extra_entries) if green_light_active else 0)
    eff_strong_slope = float(v44_strong_slope)
    if green_light_active and v44_gl_slope_relax > 0:
        eff_strong_slope = max(0.0, float(v44_strong_slope) * (1.0 - float(v44_gl_slope_relax)))

    side: Optional[str] = None
    strength: str = "normal"
    reason: str = "v44_ny: news window block"
    queued = sdat.get("queued_pending")
    queued_confirmed = False
    latest_m5_ts = pd.Timestamp(m5_df["time"].iloc[-1]) if ("time" in m5_df.columns and len(m5_df) > 0) else None
    if latest_m5_ts is not None:
        if latest_m5_ts.tzinfo is None:
            latest_m5_ts = latest_m5_ts.tz_localize("UTC")
        else:
            latest_m5_ts = latest_m5_ts.tz_convert("UTC")

    if not in_news_window:
        side, strength, reason = evaluate_v44_entry(
            h1_df, m5_df, tick, pip, "ny", sdat,
            now_utc=now_utc,
            max_entries_per_day=effective_max_entries_day,
            session_stop_losses=v44_session_stop_losses,
            h1_ema_fast_period=v44_h1_ema_fast,
            h1_ema_slow_period=v44_h1_ema_slow,
            m5_ema_fast_period=v44_m5_ema_fast,
            m5_ema_slow_period=v44_m5_ema_slow,
            slope_bars=v44_slope_bars,
            strong_slope_threshold=eff_strong_slope,
            weak_slope_threshold=v44_weak_slope,
            min_body_pips=v44_min_body_pips,
            atr_pct_filter_enabled=v44_atr_pct_filter_enabled,
            atr_pct_cap=v44_atr_pct_cap,
            atr_pct_lookback=v44_atr_pct_lookback,
        )
        if isinstance(queued, dict) and latest_m5_ts is not None:
            exp_ts = pd.Timestamp(queued.get("expiry_time")) if queued.get("expiry_time") else None
            if exp_ts is not None:
                if exp_ts.tzinfo is None:
                    exp_ts = exp_ts.tz_localize("UTC")
                else:
                    exp_ts = exp_ts.tz_convert("UTC")
            if exp_ts is not None and latest_m5_ts > exp_ts:
                sdat["queued_pending"] = None
                qs = dict(sdat.get("queued_stats", {}))
                qs["expired"] = int(qs.get("expired", 0)) + 1
                sdat["queued_stats"] = qs
                queued = None
            elif str(queued.get("last_checked_m5_time", "")) != latest_m5_ts.isoformat():
                m5_open = float(m5_df["open"].astype(float).iloc[-1])
                m5_close = float(m5_df["close"].astype(float).iloc[-1])
                m5_body_pips = abs(m5_close - m5_open) / pip
                q_side = str(queued.get("side", "")).lower()
                is_confirm = (((q_side == "buy" and m5_close > m5_open) or (q_side == "sell" and m5_close < m5_open)) and m5_body_pips >= float(v44_min_body_pips))
                if is_confirm:
                    rem = int(queued.get("remaining", 1)) - 1
                    queued["remaining"] = rem
                    queued["last_checked_m5_time"] = latest_m5_ts.isoformat()
                    if rem <= 0:
                        side = q_side
                        strength = str(queued.get("strength", strength))
                        reason = f"v44: queued confirm ({queued.get('confirm_bars', 0)} bars)"
                        sdat["queued_pending"] = None
                        queued_confirmed = True
                        qs = dict(sdat.get("queued_stats", {}))
                        qs["confirmed"] = int(qs.get("confirmed", 0)) + 1
                        sdat["queued_stats"] = qs
                    else:
                        sdat["queued_pending"] = queued
                        no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason=f"v44: queued confirmation progress {max(0, rem)} remaining", side=None)
                        no_trade["phase3_state_updates"] = {session_key: sdat}
                        return no_trade
                else:
                    queued["last_checked_m5_time"] = latest_m5_ts.isoformat()
                    sdat["queued_pending"] = queued

    if side is None:
        if v44_news_filter_enabled and v44_news_trend_enabled and news_events:
            latest_ev = _v44_most_recent_news_event(now_utc, news_events)
            if latest_ev is not None:
                delay_start = latest_ev + pd.Timedelta(minutes=max(0, v44_news_trend_delay_minutes))
                delay_end = delay_start + pd.Timedelta(minutes=max(1, v44_news_trend_window_minutes))
                now_ts = pd.Timestamp(now_utc)
                if now_ts.tzinfo is None:
                    now_ts = now_ts.tz_localize("UTC")
                else:
                    now_ts = now_ts.tz_convert("UTC")
                if latest_ev <= now_ts < delay_start:
                    sdat["news_status"] = "waiting_delay"
                    sdat["news_wait_minutes"] = max(0.0, (delay_start - now_ts) / pd.Timedelta(minutes=1))
                elif delay_start <= now_ts <= delay_end:
                    sdat["news_status"] = "waiting_confirm"
            ev = _v44_news_trend_active_event(now_utc, news_events, delay_minutes=v44_news_trend_delay_minutes, window_minutes=v44_news_trend_window_minutes)
            if ev is not None and len(m1_df) >= int(v44_news_trend_confirm_bars):
                sdat["news_event_time"] = pd.Timestamp(ev).isoformat()
                h1_tr = compute_v44_h1_trend(h1_df, v44_h1_ema_fast, v44_h1_ema_slow)
                if h1_tr in {"up", "down"}:
                    nt_side = "buy" if h1_tr == "up" else "sell"
                    sdat["news_trend_side"] = nt_side
                    confirm_ok = True
                    conf_n = int(v44_news_trend_confirm_bars)
                    progress = 0
                    for j in range(len(m1_df) - conf_n, len(m1_df)):
                        oj = float(m1_df.iloc[j]["open"])
                        cj = float(m1_df.iloc[j]["close"])
                        body = abs(cj - oj) / pip
                        dir_ok = (cj > oj) if nt_side == "buy" else (cj < oj)
                        if (not dir_ok) or (body < float(v44_news_trend_min_body_pips)):
                            confirm_ok = False
                            break
                        progress += 1
                    sdat["news_confirm_progress"] = f"{progress}/{conf_n}"
                    if confirm_ok:
                        if v44_news_trend_require_ema_align:
                            cser = m5_df["close"].astype(float)
                            ef = _compute_ema(cser, period=v44_m5_ema_fast)
                            es = _compute_ema(cser, period=v44_m5_ema_slow)
                            ema_ok = float(ef.iloc[-1]) > float(es.iloc[-1]) if nt_side == "buy" else float(ef.iloc[-1]) < float(es.iloc[-1])
                            if not ema_ok:
                                sdat["news_status"] = "ema_misaligned"
                                no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason="v44_ny: news-trend EMA misaligned", side=None)
                                no_trade["phase3_state_updates"] = {session_key: sdat}
                                return no_trade
                        sdat["news_status"] = "ready"
                        side = nt_side
                        strength = "strong"
                        reason = "v44: news trend confirmation"
                    else:
                        sdat["news_status"] = "waiting_confirm"
                else:
                    sdat["news_status"] = "waiting_h1_trend"
        if side is None:
            no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason=reason, side=None)
            no_trade["phase3_state_updates"] = {session_key: sdat}
            return no_trade

    atr_rank = _compute_v44_atr_rank(m5_df, lookback=v44_atr_pct_lookback)
    allow_map = {
        "strong_only": {"strong"},
        "strong_normal": {"strong", "normal"},
        "all": {"strong", "normal", "weak"},
    }
    allowed_strengths = allow_map.get(v44_ny_strength_allow, {"strong", "normal"})
    strength_allowed = strength in allowed_strengths
    if strength == "weak" and v44_skip_weak:
        strength_allowed = False
    if strength == "normal" and v44_skip_normal:
        strength_allowed = False
    if (not strength_allowed) and strength == "normal" and v44_allow_normal_plus and atr_rank is not None:
        atr_now = float(m5_df["high"].astype(float).iloc[-1] - m5_df["low"].astype(float).iloc[-1]) / pip
        if atr_now >= float(v44_normalplus_atr_min_pips):
            strength_allowed = True
    if (not strength_allowed) and strength == "normal" and green_light_active and v44_gl_allow_normal and atr_rank is not None:
        if float(atr_rank) < float(v44_gl_normal_atr_cap):
            strength_allowed = True
    if not strength_allowed:
        no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason=f"v44_ny: strength {strength} blocked by {v44_ny_strength_allow}", side=None)
        no_trade["phase3_state_updates"] = {session_key: sdat}
        return no_trade

    if v44_queued_confirm_bars > 0 and (not queued_confirmed):
        cur_pending = sdat.get("queued_pending")
        if isinstance(cur_pending, dict):
            cur_side = str(cur_pending.get("side", "")).lower()
            if cur_side != str(side).lower():
                exp_ts = _ts_now + pd.Timedelta(minutes=max(5, int(v44_queued_confirm_bars) * 5))
                sdat["queued_pending"] = {
                    "side": side,
                    "strength": strength,
                    "confirm_bars": int(v44_queued_confirm_bars),
                    "remaining": int(v44_queued_confirm_bars),
                    "expiry_time": exp_ts.isoformat(),
                    "last_checked_m5_time": latest_m5_ts.isoformat() if latest_m5_ts is not None else "",
                }
                qs = dict(sdat.get("queued_stats", {}))
                qs["generated"] = int(qs.get("generated", 0)) + 1
                qs["replaced"] = int(qs.get("replaced", 0)) + 1
                sdat["queued_stats"] = qs
                no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason=f"v44: queued signal replaced with {side}", side=None)
                no_trade["phase3_state_updates"] = {session_key: sdat}
                return no_trade
            no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason=f"v44: queued signal already pending ({cur_side})", side=None)
            no_trade["phase3_state_updates"] = {session_key: sdat}
            return no_trade
        exp_ts = _ts_now + pd.Timedelta(minutes=max(5, int(v44_queued_confirm_bars) * 5))
        sdat["queued_pending"] = {
            "side": side,
            "strength": strength,
            "confirm_bars": int(v44_queued_confirm_bars),
            "remaining": int(v44_queued_confirm_bars),
            "expiry_time": exp_ts.isoformat(),
            "last_checked_m5_time": latest_m5_ts.isoformat() if latest_m5_ts is not None else "",
        }
        qs = dict(sdat.get("queued_stats", {}))
        qs["generated"] = int(qs.get("generated", 0)) + 1
        sdat["queued_stats"] = qs
        no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason=f"v44: signal queued for {v44_queued_confirm_bars} confirmations ({strength})", side=None)
        no_trade["phase3_state_updates"] = {session_key: sdat}
        return no_trade

    if v44_regime_block_enabled:
        if isinstance(ownership_audit, dict):
            _blocked = bool(ownership_audit.get("defensive_v44_regime_block"))
            _regime_label = str(ownership_audit.get("regime_label") or "unknown")
            _block_reason = str(ownership_audit.get("defensive_v44_regime_reason") or "")
        else:
            m15_df = _drop_incomplete_tf(data_by_tf.get("M15"), "M15")
            _blocked, _regime_label, _block_reason = support._v44_regime_block(m5_df, m15_df, h1_df, pip)
        sdat["regime_label"] = _regime_label
        if _blocked:
            no_trade["decision"] = ExecutionDecision(attempted=True, placed=False, reason=f"v44_ny: {_block_reason}", side=side)
            no_trade["phase3_state_updates"] = {session_key: sdat}
            return no_trade

    _v44_cell_str = str((ownership_audit or {}).get("ownership_cell") or "") or None
    if _v44_cell_str:
        sdat["ownership_cell"] = _v44_cell_str
    _blocked, _block_reason = v44_defensive_veto_block_from_state(ownership_cell=_v44_cell_str, overlay_state=overlay_state)
    if _blocked:
        print(f"[phase3] DEFENSIVE VETO: v44_ny blocked in cell {_v44_cell_str} (side={side}, strength={strength})")
        no_trade["decision"] = ExecutionDecision(attempted=True, placed=False, reason=_block_reason or "v44_ny: defensive veto", side=side)
        no_trade["phase3_state_updates"] = {session_key: sdat}
        return no_trade

    entry_price = tick.ask if side == "buy" else tick.bid
    is_news_trend_entry = "news trend" in str(reason).lower()
    risk_pct_for_trade = float(v44_news_trend_risk_pct) if is_news_trend_entry else float(v44_risk_pct)
    if is_news_trend_entry:
        sl_pips = max(0.1, float(v44_news_trend_sl_pips))
        sl_price = entry_price - sl_pips * pip if side == "buy" else entry_price + sl_pips * pip
        tp1_pips = max(0.1, float(v44_news_trend_tp_rr) * sl_pips)
    else:
        sl_price = compute_v44_sl(side, m5_df, entry_price, pip)
        sl_pips = abs(entry_price - sl_price) / pip
        if strength == "strong":
            tp1_pips = v44_strong_tp1_pips * sl_pips
        elif strength == "normal":
            tp1_pips = v44_normal_tp1_pips * sl_pips
        else:
            tp1_pips = v44_weak_tp1_pips * sl_pips
    tp1_price = entry_price + (tp1_pips * pip if side == "buy" else -tp1_pips * pip)

    equity = _account_sizing_value(adapter, fallback=100000.0)
    pip_value_per_lot = (pip / entry_price) * 100000.0
    if v44_win_streak_scope == "session":
        bonus_steps = max(0, int(sdat.get("win_streak", 0)))
    else:
        bonus_steps = max(0, int(phase3_state.get("v44_win_streak_global", 0)))
    units = 0
    if v44_sizing_mode == "multiplier":
        regime_lot = v44_base_lot * float(v44_str_size_mults.get(strength, 1.0))
        lot = min(v44_max_lot, max(0.01, regime_lot + (v44_win_bonus_step * bonus_steps)))
        units = int(lot * 100000.0)
    elif v44_sizing_mode == "hybrid":
        if sl_pips > 0 and pip_value_per_lot > 0:
            risk_usd = equity * risk_pct_for_trade
            raw_lot = risk_usd / (sl_pips * pip_value_per_lot)
            raw_lot *= float(v44_hyb_str_mults.get(strength, 1.0))
            raw_lot *= float(v44_hybrid_ny_boost)
            bonus_factor = min(v44_rp_max_lot_mult, 1.0 + (v44_rp_win_bonus_pct / 100.0) * bonus_steps)
            lot = max(v44_rp_min_lot, min(v44_rp_max_lot, raw_lot * bonus_factor))
            units = int(lot * 100000.0)
    else:
        if sl_pips > 0 and pip_value_per_lot > 0:
            risk_usd = equity * risk_pct_for_trade
            raw_lot = risk_usd / (sl_pips * pip_value_per_lot)
            raw_lot *= float(v44_rp_str_mults.get(strength, 1.0))
            raw_lot *= float(v44_rp_ny_mult)
            bonus_factor = min(v44_rp_max_lot_mult, 1.0 + (v44_rp_win_bonus_pct / 100.0) * bonus_steps)
            lot = max(v44_rp_min_lot, min(v44_rp_max_lot, raw_lot * bonus_factor))
            units = int(lot * 100000.0)
    if units <= 0:
        no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason="v44_ny: size=0", side=None)
        return no_trade

    strategy_tag = "phase3:v44_ny:news" if is_news_trend_entry else f"phase3:v44_ny:{strength}"
    if _v44_cell_str:
        strategy_tag = f"{strategy_tag}@{_v44_cell_str}"
        reason = f"{reason} [cell={_v44_cell_str}]"
    comment = f"phase3_integrated:{policy.id}:v44_ny"
    try:
        dec = adapter.place_order(
            symbol=profile.symbol,
            side=side,
            lots=units / 100000.0,
            stop_price=round(float(sl_price), 3),
            # V44 TP1 is managed by the Phase 3 exit engine as a partial-close trigger.
            # Sending TP1 as a broker-native TP causes the whole position to close before
            # the partial + runner logic can execute.
            target_price=None,
            comment=comment,
        )
    except Exception as e:
        return {
            "decision": ExecutionDecision(attempted=True, placed=False, reason=f"v44_ny order error: {e}", side=side),
            "phase3_state_updates": {},
            "strategy_tag": strategy_tag,
        }

    confirmed_fill, confirmed_deal_id = _phase3_order_confirmed(adapter, profile, dec)
    if not confirmed_fill:
        return {
            "decision": ExecutionDecision(
                attempted=True,
                placed=False,
                reason="v44_ny: broker order pending/unfilled",
                side=side,
                order_retcode=getattr(dec, "order_retcode", None),
                order_id=getattr(dec, "order_id", None),
                deal_id=getattr(dec, "deal_id", None),
                fill_price=getattr(dec, "fill_price", None),
            ),
            "phase3_state_updates": {},
            "strategy_tag": strategy_tag,
        }

    sdat["trade_count"] = int(sdat.get("trade_count", 0)) + 1
    sdat["last_entry_time"] = now_utc.isoformat()
    risk_usd_planned = float(sl_pips) * (pip / entry_price) * float(units) if units > 0 else 0.0

    return {
        "decision": ExecutionDecision(
            attempted=True,
            placed=True,
            reason=f"{reason} | strength={strength} TP1={tp1_pips:.2f}p SL={sl_pips:.1f}p units={units}",
            side=side,
            order_retcode=getattr(dec, "order_retcode", None),
            order_id=getattr(dec, "order_id", None),
            deal_id=confirmed_deal_id if confirmed_deal_id is not None else getattr(dec, "deal_id", None),
            fill_price=getattr(dec, "fill_price", None),
        ),
        "phase3_state_updates": {session_key: sdat},
        "strategy_tag": strategy_tag,
        "sl_price": float(sl_price),
        "tp1_price": float(tp1_price),
        "units": int(units),
        "entry_price": float(entry_price),
        "sl_pips": float(sl_pips),
        "risk_usd_planned": float(risk_usd_planned),
    }
