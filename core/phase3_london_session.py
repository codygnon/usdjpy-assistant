from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from core.ownership_table import cell_key, der_bucket, er_bucket


def execute_london_v2_session(
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
    _as_risk_fraction = support._as_risk_fraction
    _drop_incomplete_tf = support._drop_incomplete_tf
    _compute_session_windows = support._compute_session_windows
    compute_asian_range = support.compute_asian_range
    evaluate_london_v2_arb = support.evaluate_london_v2_arb
    london_setup_d_weekday_block_from_state = support.london_setup_d_weekday_block_from_state
    _account_sizing_value = support._account_sizing_value
    _compute_risk_units = support._compute_risk_units
    _phase3_order_confirmed = support._phase3_order_confirmed

    pip = float(profile.pip_size) if hasattr(profile, "pip_size") else support.PIP_SIZE
    no_trade = {
        "decision": ExecutionDecision(attempted=False, placed=False, reason="", side=None),
        "phase3_state_updates": {},
        "strategy_tag": None,
    }

    ldn_config = (sizing_config or {}).get("london_v2", {})
    ldn_max_open = int(ldn_config.get("max_open_positions", support.LDN_MAX_OPEN))
    ldn_default_risk = _as_risk_fraction(ldn_config.get("risk_per_trade_pct", support.LDN_RISK_PCT), support.LDN_RISK_PCT)
    ldn_arb_risk_pct = _as_risk_fraction(ldn_config.get("arb_risk_per_trade_pct", ldn_default_risk), ldn_default_risk)
    ldn_d_risk_pct = _as_risk_fraction(ldn_config.get("d_risk_per_trade_pct", ldn_config.get("lmp_risk_per_trade_pct", ldn_default_risk)), ldn_default_risk)
    ldn_max_total_risk_pct = _as_risk_fraction(ldn_config.get("max_total_open_risk_pct", 0.05), 0.05)
    _ldn_cap_raw = ldn_config.get("max_trades_per_day_total", support.LDN_MAX_TRADES_PER_DAY_TOTAL)
    try:
        ldn_max_trades_per_day_total = int(_ldn_cap_raw) if _ldn_cap_raw is not None else 0
    except Exception:
        ldn_max_trades_per_day_total = 0
    ldn_range_min_pips = float(ldn_config.get("arb_range_min_pips", support.LDN_ARB_RANGE_MIN_PIPS))
    ldn_range_max_pips = float(ldn_config.get("arb_range_max_pips", support.LDN_ARB_RANGE_MAX_PIPS))
    ldn_lor_min_pips = float(ldn_config.get("lor_range_min_pips", support.LDN_D_LOR_MIN_PIPS))
    ldn_lor_max_pips = float(ldn_config.get("lor_range_max_pips", support.LDN_D_LOR_MAX_PIPS))
    ldn_d_max_trades = int(ldn_config.get("d_max_trades", support.LDN_D_MAX_TRADES))
    ldn_d_allow_long = bool(ldn_config.get("d_allow_long", True))
    ldn_d_allow_short = bool(ldn_config.get("d_allow_short", False))
    ldn_a_allow_long = bool(ldn_config.get("a_allow_long", True))
    ldn_a_allow_short = bool(ldn_config.get("a_allow_short", True))
    ldn_leverage = float(ldn_config.get("leverage", 33.0))
    ldn_max_margin_frac = float(ldn_config.get("max_margin_usage_fraction_per_trade", 0.5))
    ldn_active_days = ldn_config.get("active_days_utc", ["Tuesday", "Wednesday"])
    ldn_active_days = set(str(d) for d in ldn_active_days) if isinstance(ldn_active_days, (list, tuple, set)) else {"Tuesday", "Wednesday"}
    ldn_disable_channel_reset = bool(ldn_config.get("disable_channel_reset_after_exit", False))

    ldn_max_entry_spread = float(ldn_config.get("max_entry_spread_pips", support.LDN_MAX_SPREAD_PIPS))
    if (tick.ask - tick.bid) / pip > ldn_max_entry_spread:
        no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason=f"london_v2: spread veto ({(tick.ask - tick.bid) / pip:.1f}p > {ldn_max_entry_spread:.1f}p)", side=None)
        return no_trade

    m1_df = _drop_incomplete_tf(data_by_tf.get("M1"), "M1")
    if m1_df is None or m1_df.empty:
        no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason="london_v2: missing M1", side=None)
        return no_trade

    windows = _compute_session_windows(now_utc)
    if now_utc.strftime("%A") not in ldn_active_days:
        no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason="london_v2: day not active", side=None)
        return no_trade
    today = now_utc.date().isoformat()
    session_key = f"session_london_{today}"
    sdat = dict(phase3_state.get(session_key, {}))
    sdat.setdefault("arb_trades", 0)
    sdat.setdefault("d_trades", 0)
    sdat.setdefault("total_trades", 0)
    sdat.setdefault("consecutive_losses", 0)
    sdat.setdefault("last_entry_time", None)
    channels = dict(sdat.get("channels", {}))
    for _ck in ("A_long", "A_short", "D_long", "D_short"):
        channels.setdefault(_ck, "ARMED")

    if ldn_max_trades_per_day_total > 0 and int(sdat.get("total_trades", 0)) >= ldn_max_trades_per_day_total:
        no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason=f"london_v2: daily cap {sdat.get('total_trades', 0)}/{ldn_max_trades_per_day_total}", side=None)
        no_trade["phase3_state_updates"] = {}
        return no_trade

    high, low, range_pips, range_ok = compute_asian_range(
        m1_df,
        int(windows["london_open"].hour),
        range_min_pips=ldn_range_min_pips,
        range_max_pips=ldn_range_max_pips,
    )
    state_updates = {"london_asian_range": {"date": today, "high": high, "low": low, "pips": range_pips, "is_valid": range_ok}}
    open_count = int(phase3_state.get("open_trade_count", 0))
    if open_count >= ldn_max_open:
        no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason=f"london_v2: max open {open_count}/{ldn_max_open}", side=None)
        no_trade["phase3_state_updates"] = state_updates
        return no_trade

    if bool(ldn_config.get("ownership_cluster_enabled", True)):
        if isinstance(ownership_audit, dict):
            blocked = bool(ownership_audit.get("defensive_london_cluster_block"))
            block_reason = str(ownership_audit.get("defensive_london_cluster_reason") or "")
        else:
            m5_regime = _drop_incomplete_tf(data_by_tf.get("M5"), "M5")
            m15_regime = _drop_incomplete_tf(data_by_tf.get("M15"), "M15")
            h1_regime = _drop_incomplete_tf(data_by_tf.get("H1"), "H1")
            blocked, block_reason = support._london_v2_ownership_cluster_block(m5_regime, m15_regime, h1_regime, pip)
        if blocked:
            no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason=f"london_v2: {block_reason}", side=None)
            no_trade["phase3_state_updates"] = state_updates
            return no_trade

    strategy_tag = None
    side = None
    reason = ""
    entry_price = 0.0
    sl_price = None
    tp1_price = None
    lor_high = np.nan
    lor_low = np.nan

    a_start = windows["london_open"] + support.pd.Timedelta(minutes=int(ldn_config.get("a_entry_start_min_after_london", 0)))
    a_end = windows["london_open"] + support.pd.Timedelta(minutes=int(ldn_config.get("a_entry_end_min_after_london", 90)))
    in_arb_window = a_start <= now_utc < a_end
    d_start = windows["london_open"] + support.pd.Timedelta(minutes=int(ldn_config.get("d_entry_start_min_after_london", 15)))
    d_end = windows["london_open"] + support.pd.Timedelta(minutes=int(ldn_config.get("d_entry_end_min_after_london", 120)))
    in_d_window = d_start <= now_utc < d_end

    m1_last_high = float(m1_df["high"].astype(float).iloc[-1])
    m1_last_low = float(m1_df["low"].astype(float).iloc[-1])
    if not ldn_disable_channel_reset:
        if str(channels.get("A_long", "ARMED")) == "WAITING_RESET" and m1_last_low <= float(high):
            channels["A_long"] = "ARMED"
        if str(channels.get("A_short", "ARMED")) == "WAITING_RESET" and m1_last_high >= float(low):
            channels["A_short"] = "ARMED"

    if in_arb_window:
        if range_ok:
            side, reason = evaluate_london_v2_arb(m1_df, tick, high, low, pip, sdat)
            if side:
                ch_key = "A_long" if side == "buy" else "A_short"
                if str(channels.get(ch_key, "ARMED")) != "ARMED":
                    side = None
                    reason = f"london_arb: {ch_key} waiting reset"
            if side:
                if side == "buy" and not ldn_a_allow_long:
                    side = None
                    reason = "london_arb: long disabled"
                elif side == "sell" and not ldn_a_allow_short:
                    side = None
                    reason = "london_arb: short disabled"
            if side:
                entry_price = tick.ask if side == "buy" else tick.bid
                raw_sl = low - support.LDN_ARB_SL_BUFFER_PIPS * pip if side == "buy" else high + support.LDN_ARB_SL_BUFFER_PIPS * pip
                sl_pips = abs(entry_price - raw_sl) / pip
                sl_pips = max(support.LDN_ARB_SL_MIN_PIPS, min(support.LDN_ARB_SL_MAX_PIPS, sl_pips))
                sl_price = entry_price - sl_pips * pip if side == "buy" else entry_price + sl_pips * pip
                tp1_price = entry_price + (support.LDN_ARB_TP1_R * sl_pips * pip if side == "buy" else -support.LDN_ARB_TP1_R * sl_pips * pip)
                strategy_tag = "phase3:london_v2_arb"
        else:
            reason = f"london_v2: asian range invalid ({range_pips:.1f}p)"

    _ldn_d_tp1_r = float(ldn_config.get("d_tp1_r", support.LDN_D_TP1_R))
    if side is None and in_d_window:
        lor_form_end = windows["london_open"] + support.pd.Timedelta(minutes=15)
        lor_w = m1_df[(m1_df["time"] >= windows["london_open"]) & (m1_df["time"] < lor_form_end)]
        if not lor_w.empty:
            lor_high = float(lor_w["high"].max())
            lor_low = float(lor_w["low"].min())
            if not ldn_disable_channel_reset:
                if str(channels.get("D_long", "ARMED")) == "WAITING_RESET" and m1_last_low <= float(lor_high):
                    channels["D_long"] = "ARMED"
                if str(channels.get("D_short", "ARMED")) == "WAITING_RESET" and m1_last_high >= float(lor_low):
                    channels["D_short"] = "ARMED"
            lor_pips = (lor_high - lor_low) / pip
            lor_valid = ldn_lor_min_pips <= lor_pips <= ldn_lor_max_pips
            state_updates["london_lor"] = {"date": today, "high": lor_high, "low": lor_low, "pips": lor_pips, "is_valid": lor_valid}
            if lor_valid and int(sdat.get("d_trades", 0)) < ldn_d_max_trades:
                close_px = float(m1_df.iloc[-1]["close"])
                if ldn_d_allow_long and close_px > lor_high + support.LDN_D_BREAKOUT_BUFFER_PIPS * pip:
                    if str(channels.get("D_long", "ARMED")) == "ARMED":
                        side = "buy"
                        reason = "london_v2_d: LOR long breakout"
                        entry_price = tick.ask
                        raw_sl = lor_low - support.LDN_D_SL_BUFFER_PIPS * pip
                        sl_pips = abs(entry_price - raw_sl) / pip
                        sl_pips = max(support.LDN_D_SL_MIN_PIPS, min(support.LDN_D_SL_MAX_PIPS, sl_pips))
                        sl_price = entry_price - sl_pips * pip
                        tp1_price = entry_price + (_ldn_d_tp1_r * sl_pips * pip)
                        strategy_tag = "phase3:london_v2_d"
                    else:
                        reason = "london_v2_d: D_long waiting reset"
                elif ldn_d_allow_short and close_px < lor_low - support.LDN_D_BREAKOUT_BUFFER_PIPS * pip:
                    if str(channels.get("D_short", "ARMED")) == "ARMED":
                        side = "sell"
                        reason = "london_v2_d: LOR short breakout"
                        entry_price = tick.bid
                        raw_sl = lor_high + support.LDN_D_SL_BUFFER_PIPS * pip
                        sl_pips = abs(raw_sl - entry_price) / pip
                        sl_pips = max(support.LDN_D_SL_MIN_PIPS, min(support.LDN_D_SL_MAX_PIPS, sl_pips))
                        sl_price = entry_price + sl_pips * pip
                        tp1_price = entry_price - (_ldn_d_tp1_r * sl_pips * pip)
                        strategy_tag = "phase3:london_v2_d"
                    else:
                        reason = "london_v2_d: D_short waiting reset"
                else:
                    reason = "london_v2_d: no breakout"
            elif int(sdat.get("d_trades", 0)) >= ldn_d_max_trades:
                reason = f"london_v2_d: max trades reached ({sdat.get('d_trades', 0)}/{ldn_d_max_trades})"
            else:
                reason = f"london_v2_d: invalid LOR range ({lor_pips:.1f}p)"
        elif not reason:
            reason = "london_v2_d: missing LOR window data"

    if side is None or sl_price is None or tp1_price is None:
        no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason=reason or "london_v2: no setup", side=None)
        no_trade["phase3_state_updates"] = state_updates
        return no_trade

    _blocked, _block_reason = london_setup_d_weekday_block_from_state(now_utc=now_utc, strategy_tag=strategy_tag, overlay_state=overlay_state)
    if _blocked:
        print(f"[phase3] L1 weekday suppression: blocking london_v2_d on {now_utc.strftime('%A')}")
        no_trade["decision"] = ExecutionDecision(attempted=True, placed=False, reason=_block_reason or "london_v2_d: L1 weekday suppression", side=side)
        no_trade["phase3_state_updates"] = state_updates
        return no_trade

    _ldn_cell_str = str((ownership_audit or {}).get("ownership_cell") or "") or None
    if _ldn_cell_str is None:
        _ldn_m5 = _drop_incomplete_tf(data_by_tf.get("M5"), "M5")
        _ldn_m15 = _drop_incomplete_tf(data_by_tf.get("M15"), "M15")
        _ldn_h1 = _drop_incomplete_tf(data_by_tf.get("H1"), "H1")
        if _ldn_m5 is not None and not _ldn_m5.empty:
            _ldn_label, _, _ = support._compute_regime_snapshot(_ldn_m5, _ldn_m15, _ldn_h1, pip)
            if _ldn_label != "unknown":
                _ldn_er = float(support.compute_efficiency_ratio(_ldn_m5, lookback=12))
                _ldn_der = float(support.compute_delta_efficiency_ratio(_ldn_m5, lookback=12, delta_bars=3))
                if not np.isfinite(_ldn_er):
                    _ldn_er = 0.5
                if not np.isfinite(_ldn_der):
                    _ldn_der = 0.0
                _ldn_cell_str = cell_key(_ldn_label, er_bucket(_ldn_er), der_bucket(_ldn_der))
    if _ldn_cell_str:
        strategy_tag = f"{strategy_tag}@{_ldn_cell_str}"
        reason = f"{reason} [cell={_ldn_cell_str}]"
    if _ldn_d_tp1_r != support.LDN_D_TP1_R and (strategy_tag or "").startswith("phase3:london_v2_d"):
        print(f"[phase3] L1 exit override active: d_tp1_r={_ldn_d_tp1_r} (default={support.LDN_D_TP1_R})")

    equity = _account_sizing_value(adapter, fallback=100000.0)
    acct_equity = float(equity)
    margin_used = 0.0
    try:
        acct = adapter.get_account_info()
        _eq = getattr(acct, "equity", None)
        if _eq is not None:
            acct_equity = float(_eq)
        _mu = getattr(acct, "margin_used", None)
        if _mu is not None:
            margin_used = float(_mu)
    except Exception:
        pass

    current_setup_risk_pct = ldn_arb_risk_pct if (strategy_tag or "").endswith("_arb") else ldn_d_risk_pct
    new_risk_usd = equity * current_setup_risk_pct
    open_risk_usd = 0.0
    if store is not None:
        try:
            profile_name = getattr(profile, "profile_name", str(profile))
            for ot in store.list_open_trades(profile_name):
                if not str(ot["entry_type"] or "").startswith("phase3:london_v2"):
                    continue
                rup = ot["risk_usd_planned"] if "risk_usd_planned" in ot.keys() else None
                if rup is not None:
                    try:
                        open_risk_usd += float(rup)
                        continue
                    except Exception:
                        pass
                ep = ot["entry_price"]
                sp = ot["stop_price"]
                lots = ot["size_lots"]
                if ep and sp and lots:
                    sl_pips = abs(float(ep) - float(sp)) / pip
                    open_risk_usd += sl_pips * (pip / float(ep)) * float(lots) * 100000.0
        except Exception:
            open_risk_usd = open_count * equity * current_setup_risk_pct
    else:
        open_risk_usd = open_count * equity * current_setup_risk_pct
    if open_risk_usd + new_risk_usd > equity * ldn_max_total_risk_pct:
        no_trade["decision"] = ExecutionDecision(
            attempted=False,
            placed=False,
            reason=f"london_v2: open risk cap (open={open_risk_usd:.0f}+new={new_risk_usd:.0f} > cap={equity * ldn_max_total_risk_pct:.0f} USD)",
            side=None,
        )
        no_trade["phase3_state_updates"] = state_updates
        return no_trade

    sl_pips = abs(entry_price - sl_price) / pip
    units = _compute_risk_units(equity, current_setup_risk_pct, sl_pips, entry_price, pip, max_units=support.MAX_UNITS)
    if units <= 0:
        no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason="london_v2: size=0", side=None)
        no_trade["phase3_state_updates"] = state_updates
        return no_trade
    risk_usd_planned = sl_pips * (pip / entry_price) * float(units)

    req_margin = abs(float(units)) / max(1.0, ldn_leverage)
    try:
        free_margin = max(0.0, acct_equity - float(margin_used))
    except Exception:
        free_margin = acct_equity * (1.0 - ldn_max_margin_frac)
    if req_margin > ldn_max_margin_frac * free_margin:
        no_trade["decision"] = ExecutionDecision(attempted=False, placed=False, reason="london_v2: margin constraint", side=None)
        no_trade["phase3_state_updates"] = state_updates
        return no_trade

    comment = f"phase3_integrated:{policy.id}:{strategy_tag.replace('phase3:','')}"
    try:
        dec = adapter.place_order(
            symbol=profile.symbol,
            side=side,
            lots=units / 100000.0,
            stop_price=round(float(sl_price), 3),
            target_price=round(float(tp1_price), 3),
            comment=comment,
        )
    except Exception as e:
        return {
            "decision": ExecutionDecision(attempted=True, placed=False, reason=f"london_v2 order error: {e}", side=side),
            "phase3_state_updates": state_updates,
            "strategy_tag": strategy_tag,
        }

    confirmed_fill, confirmed_deal_id = _phase3_order_confirmed(adapter, profile, dec)
    if not confirmed_fill:
        return {
            "decision": ExecutionDecision(
                attempted=True,
                placed=False,
                reason="london_v2: broker order pending/unfilled",
                side=side,
                order_retcode=getattr(dec, "order_retcode", None),
                order_id=getattr(dec, "order_id", None),
                deal_id=getattr(dec, "deal_id", None),
                fill_price=getattr(dec, "fill_price", None),
            ),
            "phase3_state_updates": state_updates,
            "strategy_tag": strategy_tag,
        }

    if strategy_tag.endswith("_arb"):
        sdat["arb_trades"] = int(sdat.get("arb_trades", 0)) + 1
        channels["A_long" if side == "buy" else "A_short"] = "FIRED"
    else:
        sdat["d_trades"] = int(sdat.get("d_trades", 0)) + 1
        channels["D_long" if side == "buy" else "D_short"] = "FIRED"
    sdat["total_trades"] = int(sdat.get("total_trades", 0)) + 1
    sdat["last_entry_time"] = now_utc.isoformat()
    sdat["channels"] = channels
    state_updates[session_key] = sdat

    return {
        "decision": ExecutionDecision(
            attempted=True,
            placed=True,
            reason=f"{reason} | SL={sl_pips:.1f}p units={units}",
            side=side,
            order_retcode=getattr(dec, "order_retcode", None),
            order_id=getattr(dec, "order_id", None),
            deal_id=confirmed_deal_id if confirmed_deal_id is not None else getattr(dec, "deal_id", None),
            fill_price=getattr(dec, "fill_price", None),
        ),
        "phase3_state_updates": state_updates,
        "strategy_tag": strategy_tag,
        "sl_price": float(sl_price),
        "tp1_price": float(tp1_price),
        "units": int(units),
        "entry_price": float(entry_price),
        "sl_pips": float(sl_pips),
        "risk_usd_planned": float(risk_usd_planned),
    }
