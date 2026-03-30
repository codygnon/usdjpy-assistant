"""Dashboard reporter functions.

Each reporter wraps an existing filter function and packages its result
into a FilterReport structure. The dashboard renders whatever reports
it receives — zero hardcoded filter names.
"""
from __future__ import annotations

from core.presets import FROZEN_PHASE3_DEFENDED_PRESET_ID

from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from core.dashboard_models import ContextItem, FilterReport


# ---------------------------------------------------------------------------
# Filter reporters
# ---------------------------------------------------------------------------


def report_session_filter(profile, now_utc: datetime) -> FilterReport:
    """Report session filter status."""
    from core.execution_engine import passes_session_filter

    f = profile.strategy.filters.session_filter
    enabled = f.enabled
    if not enabled:
        return FilterReport(
            filter_id="session_filter", display_name="Session Filter",
            enabled=False, is_clear=True,
            current_value=f"UTC {now_utc.strftime('%H:%M')}",
        )
    ok, reason = passes_session_filter(profile, now_utc)
    sessions_str = ", ".join(f.sessions)
    return FilterReport(
        filter_id="session_filter", display_name="Session Filter",
        enabled=True, is_clear=ok,
        current_value=f"UTC {now_utc.strftime('%H:%M')}",
        threshold=f"Sessions: {sessions_str}",
        block_reason=reason,
    )


def report_session_boundary_block(profile, now_utc: datetime) -> FilterReport:
    """Report session boundary block status (block 15min before/after NY/London/Tokyo open and close)."""
    from core.execution_engine import passes_session_boundary_block

    f = getattr(profile.strategy.filters, "session_boundary_block", None)
    if f is None or not getattr(f, "enabled", False):
        return FilterReport(
            filter_id="session_boundary_block",
            display_name="Session Boundary Block",
            enabled=False,
            is_clear=True,
            current_value=f"UTC {now_utc.strftime('%H:%M')}",
        )
    ok, reason = passes_session_boundary_block(profile, now_utc)
    buf = max(0, int(getattr(f, "buffer_minutes", 15)))
    return FilterReport(
        filter_id="session_boundary_block",
        display_name="Session Boundary Block",
        enabled=True,
        is_clear=ok,
        current_value=f"UTC {now_utc.strftime('%H:%M')}",
        threshold=f"Block ±{buf}min around open/close",
        block_reason=reason,
    )


def report_tiered_atr_trial_4(policy, m1_df, pip_size: float, trigger_type: str) -> FilterReport:
    """Report Tiered ATR(14) filter for Trial #4."""
    from core.execution_engine import _passes_tiered_atr_filter_trial_4
    from core.indicators import atr as atr_fn

    enabled = getattr(policy, "tiered_atr_filter_enabled", False)
    if not enabled:
        return FilterReport(
            filter_id="tiered_atr", display_name="Tiered ATR(14)",
            enabled=False, is_clear=True,
        )
    atr_pips_str = "N/A"
    if m1_df is not None and len(m1_df) >= 16:
        a = atr_fn(m1_df, 14)
        if not a.empty and not pd.isna(a.iloc[-1]):
            atr_pips_str = f"{float(a.iloc[-1]) / pip_size:.1f}p"

    ok, reason = _passes_tiered_atr_filter_trial_4(policy, m1_df, pip_size, trigger_type)
    block_below = getattr(policy, "tiered_atr_block_below_pips", 4.0)
    allow_all_max = getattr(policy, "tiered_atr_allow_all_max_pips", 12.0)
    pullback_max = getattr(policy, "tiered_atr_pullback_only_max_pips", 15.0)
    return FilterReport(
        filter_id="tiered_atr", display_name="Tiered ATR(14)",
        enabled=True, is_clear=ok,
        current_value=atr_pips_str,
        threshold=f"<{block_below}p block | {block_below}-{allow_all_max}p all | {allow_all_max}-{pullback_max}p pullback | >{pullback_max}p block",
        block_reason=reason,
    )


def report_regime_gate(regime_snapshot: Optional[dict]) -> FilterReport:
    """Report Trial #10 regime gate status."""
    if not regime_snapshot:
        return FilterReport(
            filter_id="trial10_regime_gate",
            display_name="Trial #10 Regime Gate",
            enabled=False,
            is_clear=True,
            current_value="Disabled",
        )
    label = str(regime_snapshot.get("label", "normal"))
    allowed = bool(regime_snapshot.get("allowed", True))
    mult = regime_snapshot.get("multiplier", 1.0)
    applied_mult = regime_snapshot.get("regime_multiplier_applied", regime_snapshot.get("regime_multiplier", mult))
    mult_active = bool(regime_snapshot.get("regime_multiplier_active", abs(float(applied_mult or 1.0) - 1.0) > 1e-9))
    reason = str(regime_snapshot.get("reason", ""))
    final_lots = regime_snapshot.get("final_lots")
    conviction_lots = regime_snapshot.get("conviction_lots")
    is_blocked = not allowed
    display = f"{label.upper()}"
    if allowed and mult_active:
        display += f" ({mult}x)"
    if final_lots is not None:
        display += f" | {float(final_lots):.2f} lots"
    explanation = reason
    if conviction_lots is not None and final_lots is not None:
        if mult_active:
            explanation = (
                f"{reason} Conviction {float(conviction_lots):.2f} -> final {float(final_lots):.2f} lots."
            ).strip()
        else:
            explanation = (
                f"{reason} Regime is informational for runner sizing. "
                f"Runner lots {float(conviction_lots):.2f} -> final {float(final_lots):.2f} lots."
            ).strip()
    return FilterReport(
        filter_id="trial10_regime_gate",
        display_name="Trial #10 Regime Gate",
        enabled=True,
        is_clear=not is_blocked,
        current_value=display,
        block_reason=reason if is_blocked else None,
        explanation=explanation,
        metadata=regime_snapshot,
    )


def report_trial10_directional_cap(
    *,
    base_lots: float | None,
    runner_max_lots: float | None,
    max_open_trades_per_side: int | None,
    configured_cap_lots: float | None,
    buy_open_lots: float = 0.0,
    sell_open_lots: float = 0.0,
) -> FilterReport:
    """Report Trial #10 same-side directional lot cap."""
    side_cap = int(max_open_trades_per_side) if max_open_trades_per_side is not None else None
    base = float(base_lots) if base_lots is not None else None
    runner_cap = float(runner_max_lots) if runner_max_lots is not None else None
    configured_cap = float(configured_cap_lots) if configured_cap_lots is not None else None

    auto_cap = None
    if base is not None and side_cap is not None and base > 0 and side_cap > 0:
        auto_cap = round(base * side_cap / 2.0, 2)

    display_cap = configured_cap if configured_cap is not None else auto_cap
    if display_cap is None:
        return FilterReport(
            filter_id="trial10_directional_cap",
            display_name="Trial #10 Directional Cap",
            enabled=False,
            is_clear=True,
            current_value="Disabled",
        )

    current_side_lots = max(float(buy_open_lots), float(sell_open_lots))
    is_clear = current_side_lots + 1e-9 < float(display_cap)
    mode = "MANUAL" if configured_cap is not None else "AUTO"
    current_value = f"{mode} {float(display_cap):.2f} lots/side"
    threshold_parts = []
    if base is not None:
        threshold_parts.append(f"Base {base:.2f}")
    if side_cap is not None:
        threshold_parts.append(f"Trades/side {side_cap}")
    if runner_cap is not None:
        threshold_parts.append(f"Runner max {runner_cap:.2f}")
    if auto_cap is not None:
        threshold_parts.append(f"Auto {auto_cap:.2f}")
    threshold = " | ".join(threshold_parts) if threshold_parts else None
    explanation = (
        f"Buy open {float(buy_open_lots):.2f} lots, Sell open {float(sell_open_lots):.2f} lots. "
        "Auto cap formula = runner base lots x max open trades per side / 2."
    )
    if configured_cap is not None and auto_cap is not None:
        explanation += f" Manual override active over auto default {auto_cap:.2f}."

    return FilterReport(
        filter_id="trial10_directional_cap",
        display_name="Trial #10 Directional Cap",
        enabled=True,
        is_clear=is_clear,
        current_value=current_value,
        threshold=threshold,
        explanation=explanation,
        metadata={
            "directional_cap_lots": round(float(display_cap), 4),
            "auto_directional_cap_lots": round(float(auto_cap), 4) if auto_cap is not None else None,
            "configured_directional_cap_lots": round(float(configured_cap), 4) if configured_cap is not None else None,
            "buy_open_lots": round(float(buy_open_lots), 4),
            "sell_open_lots": round(float(sell_open_lots), 4),
            "runner_base_lots": round(float(base), 4) if base is not None else None,
            "runner_max_lots": round(float(runner_cap), 4) if runner_cap is not None else None,
            "max_open_trades_per_side": side_cap,
            "mode": mode.lower(),
        },
    )


def report_trial10_stop_loss(profile, policy, data_by_tf: dict, pip_size: float) -> FilterReport:
    """Report Trial #10 active stop-loss mode and current ATR-derived stop distance."""
    from core.execution_engine import _resolve_trial10_stop_pips
    from core.indicators import atr as atr_fn
    from core.signal_engine import drop_incomplete_last_bar

    sl_cfg = getattr(profile.trade_management, "stop_loss", None)
    fallback_sl = float(getattr(policy, "sl_pips", 0.0) or getattr(profile.risk, "min_stop_pips", 0.0))
    if sl_cfg is None:
        return FilterReport(
            filter_id="trial10_stop_loss",
            display_name="Trial #10 Stop Loss",
            enabled=False,
            is_clear=True,
            current_value=f"Fixed {fallback_sl:.1f}p",
        )

    mode = str(getattr(sl_cfg, "mode", "fixed_pips") or "fixed_pips")
    if mode != "atr":
        return FilterReport(
            filter_id="trial10_stop_loss",
            display_name="Trial #10 Stop Loss",
            enabled=True,
            is_clear=True,
            current_value=f"Fixed {fallback_sl:.1f}p",
            threshold="Manual/fixed stop",
        )

    m5_df = data_by_tf.get("M5")
    atr_pips_str = "N/A"
    if m5_df is not None and not m5_df.empty:
        m5_local = drop_incomplete_last_bar(m5_df.copy(), "M5")
        if len(m5_local) >= 15:
            atr_series = atr_fn(m5_local, 14)
            if not atr_series.empty and not pd.isna(atr_series.iloc[-1]) and pip_size > 0.0:
                atr_pips_str = f"{float(atr_series.iloc[-1]) / pip_size:.1f}p"

    effective_stop = _resolve_trial10_stop_pips(profile, policy, data_by_tf)
    atr_mult = float(getattr(sl_cfg, "atr_multiplier", 1.5))
    max_sl = float(getattr(sl_cfg, "max_sl_pips", fallback_sl))
    return FilterReport(
        filter_id="trial10_stop_loss",
        display_name="Trial #10 Stop Loss",
        enabled=True,
        is_clear=True,
        current_value=f"ATR {atr_pips_str} -> Stop {effective_stop:.1f}p",
        threshold=f"ATR14 x {atr_mult:.2f}, cap {max_sl:.1f}p",
        metadata={
            "mode": mode,
            "atr_multiplier": atr_mult,
            "max_sl_pips": max_sl,
            "effective_stop_pips": round(float(effective_stop), 3),
        },
    )


def report_dual_atr_trial_5(policy, m1_df, m3_df, pip_size: float, trigger_type: str) -> FilterReport:
    """Report Dual ATR filter for Trial #5 with M1 and M3 sub-filters."""
    from core.execution_engine import _passes_atr_filter_trial_5, _get_current_sessions
    from core.indicators import atr as atr_fn

    ok, reason = _passes_atr_filter_trial_5(policy, m1_df, m3_df, pip_size, trigger_type)

    sub_filters = []

    # M1 ATR sub-filter
    m1_enabled = getattr(policy, "m1_atr_filter_enabled", True)
    m1_val = "N/A"
    m1_period = getattr(policy, "m1_atr_period", 7)
    if m1_enabled and m1_df is not None and len(m1_df) >= m1_period + 2:
        a1 = atr_fn(m1_df, m1_period)
        if not a1.empty and not pd.isna(a1.iloc[-1]):
            m1_val = f"{float(a1.iloc[-1]) / pip_size:.1f}p"
    utc_hour = datetime.now(timezone.utc).hour
    sessions = _get_current_sessions(utc_hour)
    session_str = "+".join(sessions) if sessions else "none"
    sub_filters.append(FilterReport(
        filter_id="m1_atr", display_name=f"M1 ATR({m1_period})",
        enabled=m1_enabled, is_clear="m1_atr" not in (reason or ""),
        current_value=m1_val,
        threshold=f"Session: {session_str}",
    ))

    # M3 ATR sub-filter
    m3_enabled = getattr(policy, "m3_atr_filter_enabled", False)
    m3_val = "N/A"
    m3_period = getattr(policy, "m3_atr_period", 14)
    if m3_enabled and m3_df is not None and len(m3_df) >= m3_period + 2:
        a3 = atr_fn(m3_df, m3_period)
        if not a3.empty and not pd.isna(a3.iloc[-1]):
            m3_val = f"{float(a3.iloc[-1]) / pip_size:.1f}p"
    m3_min = getattr(policy, "m3_atr_min_pips", 4.5)
    m3_max = getattr(policy, "m3_atr_max_pips", 11.0)
    sub_filters.append(FilterReport(
        filter_id="m3_atr", display_name=f"M3 ATR({m3_period})",
        enabled=m3_enabled, is_clear="m3_atr" not in (reason or ""),
        current_value=m3_val,
        threshold=f"{m3_min}-{m3_max}p",
    ))

    return FilterReport(
        filter_id="dual_atr", display_name="Dual ATR Filter",
        enabled=True, is_clear=ok,
        current_value="",
        block_reason=reason,
        sub_filters=sub_filters,
    )


def report_daily_hl_filter(
    policy, data_by_tf: dict, tick, side: str, pip_size: float,
    daily_reset_high=None, daily_reset_low=None, daily_reset_settled=False,
) -> FilterReport:
    """Report Daily H/L filter status."""
    from core.execution_engine import _passes_daily_hl_filter

    enabled = getattr(policy, "daily_hl_filter_enabled", False)
    if not enabled:
        return FilterReport(
            filter_id="daily_hl", display_name="Daily H/L Filter",
            enabled=False, is_clear=True,
        )
    ok, reason = _passes_daily_hl_filter(
        policy, data_by_tf, tick, side, pip_size,
        daily_reset_high, daily_reset_low, daily_reset_settled,
    )
    buffer = getattr(policy, "daily_hl_buffer_pips", 5.0)
    h_str = f"{daily_reset_high:.3f}" if daily_reset_high else "N/A"
    l_str = f"{daily_reset_low:.3f}" if daily_reset_low else "N/A"
    return FilterReport(
        filter_id="daily_hl", display_name="Daily H/L Filter",
        enabled=True, is_clear=ok,
        current_value=f"H:{h_str} L:{l_str}",
        threshold=f"Buffer: {buffer}p",
        block_reason=reason,
    )


def report_ema_zone_filter(policy, m1_df, pip_size: float, is_bull: bool) -> FilterReport:
    """Report EMA Zone Filter score with sub-scores."""
    from core.execution_engine import _compute_ema_zone_filter_score

    enabled = getattr(policy, "ema_zone_filter_enabled", False)
    if not enabled:
        return FilterReport(
            filter_id="ema_zone_filter", display_name="EMA Zone Filter",
            enabled=False, is_clear=True,
        )
    lookback = getattr(policy, "ema_zone_filter_lookback_bars", 3)
    threshold = getattr(policy, "ema_zone_filter_block_threshold", 0.35)
    score, details = _compute_ema_zone_filter_score(m1_df, pip_size, is_bull, lookback, policy=policy)
    if "error" in details:
        return FilterReport(
            filter_id="ema_zone_filter", display_name="EMA Zone Filter",
            enabled=True, is_clear=True,
            current_value="Insufficient data",
        )
    is_clear = score >= threshold
    sub_filters = [
        FilterReport(
            filter_id="ema_zone_spread", display_name="Spread",
            enabled=True, is_clear=True,
            current_value=f"{details['spread_pips']:.1f}p (score: {details['spread_score']:.2f})",
        ),
        FilterReport(
            filter_id="ema_zone_slope", display_name="Slope",
            enabled=True, is_clear=True,
            current_value=f"{details['slope_pips']:.1f}p (score: {details['slope_score']:.2f})",
        ),
        FilterReport(
            filter_id="ema_zone_dir", display_name="Direction",
            enabled=True, is_clear=True,
            current_value=f"{details['spread_dir_pips']:.1f}p (score: {details['dir_score']:.2f})",
        ),
    ]
    return FilterReport(
        filter_id="ema_zone_filter", display_name="EMA Zone Filter",
        enabled=True, is_clear=is_clear,
        current_value=f"Score: {score:.2f}",
        threshold=f">= {threshold}",
        block_reason=f"Score {score:.2f} < {threshold}" if not is_clear else None,
        sub_filters=sub_filters,
    )


def report_ema_zone_slope_filter_trial_7(policy, m1_df, pip_size: float, side: str) -> FilterReport:
    """Report Trial #7 slope-only EMA zone filter status."""
    from core.execution_engine import _passes_ema_zone_slope_filter_trial_7

    enabled = getattr(policy, "ema_zone_filter_enabled", False)
    if not enabled:
        return FilterReport(
            filter_id="ema_zone_slope_filter", display_name="EMA Zone Slope Filter",
            enabled=False, is_clear=True,
        )

    ok, reason, details = _passes_ema_zone_slope_filter_trial_7(policy, m1_df, pip_size, side)
    if "error" in details:
        return FilterReport(
            filter_id="ema_zone_slope_filter", display_name="EMA Zone Slope Filter",
            enabled=True, is_clear=True, current_value="Insufficient data",
        )

    sub_filters = [
        FilterReport(
            filter_id="ema5_slope", display_name="EMA5 Slope",
            enabled=True, is_clear=True,
            current_value=f"{details['ema5_slope_pips_per_bar']:.3f} p/b",
            threshold=f"{'>=' if side == 'buy' else '<='} {details['ema5_min']:.3f}" if side == "buy" else f"<= -{details['ema5_min']:.3f}",
        ),
        FilterReport(
            filter_id="ema9_slope", display_name="EMA9 Slope",
            enabled=True, is_clear=True,
            current_value=f"{details['ema9_slope_pips_per_bar']:.3f} p/b",
            threshold=f"{'>=' if side == 'buy' else '<='} {details['ema9_min']:.3f}" if side == "buy" else f"<= -{details['ema9_min']:.3f}",
        ),
        FilterReport(
            filter_id="ema21_slope", display_name="EMA21 Slope",
            enabled=True, is_clear=True,
            current_value=f"{details['ema21_slope_pips_per_bar']:.3f} p/b",
            threshold=f"{'>=' if side == 'buy' else '<='} {details['ema21_min']:.3f}" if side == "buy" else f"<= -{details['ema21_min']:.3f}",
        ),
    ]

    return FilterReport(
        filter_id="ema_zone_slope_filter", display_name="EMA Zone Slope Filter",
        enabled=True, is_clear=ok,
        current_value=f"Lookback: {details['lookback_bars']} bars",
        block_reason=reason if not ok else None,
        sub_filters=sub_filters,
    )


def report_trial7_m5_ema_distance_gate(policy, data_by_tf, pip_size: float) -> FilterReport:
    """Report Trial #7 M5 EMA distance trade gate status."""
    from core.execution_engine import _resolve_m5_trend_close_series, _trial7_m5_ema_gap_pips

    m5_df = (data_by_tf or {}).get("M5")
    if m5_df is None or m5_df.empty:
        return FilterReport(
            filter_id="m5_ema_distance_gate", display_name="M5 EMA Distance Gate",
            enabled=True, is_clear=True, current_value="No M5 data",
        )
    m5_close, m5_source = _resolve_m5_trend_close_series(policy, data_by_tf or {})
    fast_p = int(getattr(policy, "m5_trend_ema_fast", 9))
    slow_p = int(getattr(policy, "m5_trend_ema_slow", 21))
    if len(m5_close) < max(fast_p, slow_p) + 1:
        return FilterReport(
            filter_id="m5_ema_distance_gate", display_name="M5 EMA Distance Gate",
            enabled=True, is_clear=True, current_value="Insufficient data",
        )
    gap_pips, _fast_v, _slow_v = _trial7_m5_ema_gap_pips(m5_close, fast_p, slow_p, pip_size)
    min_gap = float(getattr(policy, "m5_min_ema_distance_pips", 1.0))
    is_clear = gap_pips >= min_gap
    return FilterReport(
        filter_id="m5_ema_distance_gate", display_name="M5 EMA Distance Gate",
        enabled=True, is_clear=is_clear,
        current_value=f"{gap_pips:.2f}p",
        threshold=f">= {min_gap:.2f}p | {m5_source}",
        block_reason=f"{gap_pips:.2f}p < {min_gap:.2f}p" if not is_clear else None,
    )


def report_rolling_danger_zone(policy, m1_df, tick, side: str, pip_size: float) -> FilterReport:
    """Report Rolling Danger Zone status."""
    enabled = getattr(policy, "rolling_danger_zone_enabled", False)
    if not enabled:
        return FilterReport(
            filter_id="rolling_danger", display_name="Rolling Danger Zone",
            enabled=False, is_clear=True,
        )
    lookback = getattr(policy, "rolling_danger_lookback_bars", 100)
    danger_pct = getattr(policy, "rolling_danger_zone_pct", 0.15)
    if m1_df is None or m1_df.empty or len(m1_df) < 10:
        return FilterReport(
            filter_id="rolling_danger", display_name="Rolling Danger Zone",
            enabled=True, is_clear=True,
            current_value="Insufficient data",
        )
    bars = m1_df.tail(lookback)
    rolling_high = float(bars["high"].max())
    rolling_low = float(bars["low"].min())
    price_range = rolling_high - rolling_low
    if price_range <= 0:
        return FilterReport(
            filter_id="rolling_danger", display_name="Rolling Danger Zone",
            enabled=True, is_clear=True, current_value="No range",
        )
    upper_threshold = rolling_high - (price_range * danger_pct)
    lower_threshold = rolling_low + (price_range * danger_pct)
    entry_price = tick.ask if side == "buy" else tick.bid
    blocked = False
    reason = None
    if side == "buy" and entry_price >= upper_threshold:
        blocked = True
        reason = f"BUY in upper {danger_pct*100:.0f}% zone"
    elif side == "sell" and entry_price <= lower_threshold:
        blocked = True
        reason = f"SELL in lower {danger_pct*100:.0f}% zone"
    return FilterReport(
        filter_id="rolling_danger", display_name="Rolling Danger Zone",
        enabled=True, is_clear=not blocked,
        current_value=f"{entry_price:.3f} (range: {rolling_low:.3f}-{rolling_high:.3f})",
        threshold=f"Danger: top/bottom {danger_pct*100:.0f}%",
        block_reason=reason,
    )


def report_rsi_divergence(policy, divergence_state: dict, side: str) -> FilterReport:
    """Report RSI Divergence block status."""
    enabled = getattr(policy, "rsi_divergence_enabled", False)
    if not enabled:
        return FilterReport(
            filter_id="rsi_divergence", display_name="RSI Divergence",
            enabled=False, is_clear=True,
        )
    now_utc = datetime.now(timezone.utc)
    block_key = f"block_{side}_until"
    block_until_str = divergence_state.get(block_key) if divergence_state else None
    blocked = False
    reason = None
    value = "No active block"
    if block_until_str:
        try:
            block_until = datetime.fromisoformat(block_until_str.replace("Z", "+00:00"))
            if now_utc < block_until:
                blocked = True
                remaining = (block_until - now_utc).total_seconds()
                reason = f"{side.upper()} blocked for {remaining:.0f}s"
                value = f"Blocked until {block_until.strftime('%H:%M:%S')}"
        except (ValueError, TypeError):
            pass
    return FilterReport(
        filter_id="rsi_divergence", display_name="RSI Divergence",
        enabled=True, is_clear=not blocked,
        current_value=value,
        block_reason=reason,
    )


def report_cooldown(last_placed_time: Optional[str], cooldown_min: float, trigger_type: str) -> FilterReport:
    """Report cooldown status (zone entry only)."""
    if trigger_type != "zone_entry":
        return FilterReport(
            filter_id="cooldown", display_name="Cooldown",
            enabled=True, is_clear=True,
            current_value="N/A (tiered pullback)",
        )
    if not last_placed_time:
        return FilterReport(
            filter_id="cooldown", display_name="Cooldown",
            enabled=True, is_clear=True,
            current_value="No recent trade",
            threshold=f"{cooldown_min}m",
        )
    try:
        now = datetime.now(timezone.utc)
        last = datetime.fromisoformat(last_placed_time.replace("Z", "+00:00"))
        elapsed = (now - last).total_seconds() / 60.0
        is_clear = elapsed >= cooldown_min
        return FilterReport(
            filter_id="cooldown", display_name="Cooldown",
            enabled=True, is_clear=is_clear,
            current_value=f"{elapsed:.1f}m elapsed",
            threshold=f"{cooldown_min}m",
            block_reason=f"{elapsed:.1f}m < {cooldown_min}m" if not is_clear else None,
        )
    except Exception:
        return FilterReport(
            filter_id="cooldown", display_name="Cooldown",
            enabled=True, is_clear=True, current_value="Error",
        )


def report_dead_zone(daily_reset_state: dict) -> FilterReport:
    """Report Dead Zone status (21:00-02:00 UTC)."""
    block_active = daily_reset_state.get("daily_reset_block_active", False) if daily_reset_state else False
    utc_hour = datetime.now(timezone.utc).hour
    return FilterReport(
        filter_id="dead_zone", display_name="Dead Zone (21-02 UTC)",
        enabled=True, is_clear=not block_active,
        current_value=f"UTC {utc_hour:02d}:00",
        threshold="21:00-02:00 UTC",
        block_reason="Inside dead zone" if block_active else None,
    )


def report_trend_exhaustion(exhaustion_result: Optional[dict]) -> FilterReport:
    """Report Trend Extension Exhaustion status."""
    if not exhaustion_result:
        return FilterReport(
            filter_id="trend_exhaustion", display_name="Trend Exhaustion",
            enabled=True, is_clear=True,
            current_value="No data",
        )
    zone = exhaustion_result.get("zone", "FRESH")
    ratio = exhaustion_result.get("extension_ratio")
    adj_ratio = exhaustion_result.get("adjusted_ratio")
    tf = exhaustion_result.get("time_factor")
    ratio_str = f"{ratio:.1f}x" if ratio is not None else "N/A"
    adj_str = f"{adj_ratio:.1f}x (tf={tf:.2f})" if adj_ratio is not None and tf is not None else ""
    is_clear = zone not in ("EXHAUSTED",)
    return FilterReport(
        filter_id="trend_exhaustion", display_name="Trend Exhaustion",
        enabled=True, is_clear=is_clear,
        current_value=f"Zone: {zone} | Ratio: {ratio_str} {adj_str}",
        block_reason=f"EXHAUSTED zone (ratio={ratio_str})" if zone == "EXHAUSTED" else None,
        metadata={"zone": zone, "extension_ratio": ratio},
    )


def report_trial7_adaptive_tp(policy, exhaustion_result: Optional[dict]) -> FilterReport:
    """Report Trial #7 adaptive TP status and currently effective TP."""
    base_tp = max(0.1, float(getattr(policy, "tp_pips", 4.0)))
    enabled = bool(getattr(policy, "trend_exhaustion_adaptive_tp_enabled", False))
    ext_off = max(0.0, float(getattr(policy, "trend_exhaustion_tp_extended_offset_pips", 1.0)))
    very_off = max(0.0, float(getattr(policy, "trend_exhaustion_tp_very_extended_offset_pips", 2.0)))
    min_tp = max(0.1, float(getattr(policy, "trend_exhaustion_tp_min_pips", 0.5)))
    zone = str((exhaustion_result or {}).get("zone", "normal")).lower()
    if not enabled:
        return FilterReport(
            filter_id="trial7_adaptive_tp", display_name="Trial #7 Adaptive TP",
            enabled=False, is_clear=True, current_value=f"Base {base_tp:.2f}p",
        )
    if zone == "very_extended":
        effective = max(min_tp, base_tp - very_off)
    elif zone == "extended":
        effective = max(min_tp, base_tp - ext_off)
    else:
        effective = base_tp
    return FilterReport(
        filter_id="trial7_adaptive_tp", display_name="Trial #7 Adaptive TP",
        enabled=True, is_clear=True,
        current_value=f"{effective:.2f}p (base {base_tp:.2f}p, zone {zone})",
        threshold=f"extended:-{ext_off:.2f}p | very:-{very_off:.2f}p | min:{min_tp:.2f}p",
    )


def report_trial7_reversal_risk(policy, reversal_risk_result: Optional[dict]) -> FilterReport:
    """Report Trial #7 calibrated reversal-risk score and active response."""
    enabled = bool(getattr(policy, "use_reversal_risk_score", False))
    if not enabled:
        return FilterReport(
            filter_id="trial7_reversal_risk",
            display_name="Trial #7 Reversal Risk",
            enabled=False,
            is_clear=True,
            current_value="OFF",
        )

    if not isinstance(reversal_risk_result, dict):
        return FilterReport(
            filter_id="trial7_reversal_risk",
            display_name="Trial #7 Reversal Risk",
            enabled=True,
            is_clear=True,
            current_value="Waiting for score",
        )

    score = float(reversal_risk_result.get("score", 0.0))
    tier = str(reversal_risk_result.get("tier", "low")).lower()
    regime = str(reversal_risk_result.get("regime", "transition")).lower()
    thresholds = reversal_risk_result.get("thresholds") if isinstance(reversal_risk_result.get("thresholds"), dict) else {}
    t_medium = float(thresholds.get("medium", getattr(policy, "rr_tier_medium", 58.0)))
    t_high = float(thresholds.get("high", getattr(policy, "rr_tier_high", 65.0)))
    t_critical = float(thresholds.get("critical", getattr(policy, "rr_tier_critical", 71.0)))

    response = reversal_risk_result.get("response") if isinstance(reversal_risk_result.get("response"), dict) else {}
    lot_x = float(response.get("lot_multiplier", 1.0))
    min_tier = response.get("min_tier_ema")
    zone_block = bool(response.get("block_zone_entry", False))
    managed_exit = bool(response.get("use_managed_exit", False))

    sub_filters: list[FilterReport] = []
    components = reversal_risk_result.get("components")
    if isinstance(components, dict):
        comp_labels = (
            ("rsi_divergence", "RSI Divergence"),
            ("adr_exhaustion", "ADR Exhaustion"),
            ("htf_proximity", "HTF Proximity"),
            ("ema_spread", "EMA Spread"),
        )
        for key, label in comp_labels:
            comp = components.get(key)
            if not isinstance(comp, dict):
                continue
            comp_score = float(comp.get("score", 0.0))
            sub_filters.append(
                FilterReport(
                    filter_id=f"trial7_rr_{key}",
                    display_name=label,
                    enabled=True,
                    is_clear=True,
                    current_value=f"{comp_score * 100.0:.1f}/100",
                )
            )

    min_tier_str = f"{int(min_tier)}+" if min_tier is not None else "none"
    return FilterReport(
        filter_id="trial7_reversal_risk",
        display_name="Trial #7 Reversal Risk",
        enabled=True,
        is_clear=True,
        current_value=f"Score {score:.2f} | Tier {tier.upper()} | Regime: {regime}",
        threshold=f"medium:{t_medium:.1f} high:{t_high:.1f} critical:{t_critical:.1f}",
        block_reason=None,
        sub_filters=sub_filters,
        metadata={
            "score": round(score, 2),
            "tier": tier,
            "regime": regime,
            "lot_multiplier": round(lot_x, 4),
            "min_tier_ema": int(min_tier) if min_tier is not None else None,
            "zone_block_entry": zone_block,
            "use_managed_exit": managed_exit,
        },
    )


def report_max_trades(open_count: int, max_allowed: Optional[int], side: str, side_counts: Optional[dict] = None) -> FilterReport:
    """Report max open trades status."""
    if max_allowed is None:
        return FilterReport(
            filter_id="max_trades", display_name="Max Open Trades",
            enabled=False, is_clear=True,
        )
    is_clear = open_count < max_allowed
    side_info = ""
    if side_counts:
        side_info = f" (buy:{side_counts.get('buy', 0)} sell:{side_counts.get('sell', 0)})"
    return FilterReport(
        filter_id="max_trades", display_name="Max Open Trades",
        enabled=True, is_clear=is_clear,
        current_value=f"{open_count}/{max_allowed}{side_info}",
        threshold=f"Max: {max_allowed}",
        block_reason=f"{open_count} >= {max_allowed}" if not is_clear else None,
    )


def report_max_trades_by_side(side_counts: dict[str, int], max_allowed: Optional[int]) -> list[FilterReport]:
    """Report explicit per-side max open trade caps (Buy + Sell rows)."""
    if max_allowed is None:
        return [
            FilterReport(
                filter_id="max_trades_buy",
                display_name="Max Open Trades (Buy)",
                enabled=False,
                is_clear=True,
            ),
            FilterReport(
                filter_id="max_trades_sell",
                display_name="Max Open Trades (Sell)",
                enabled=False,
                is_clear=True,
            ),
        ]
    buy_open = int(side_counts.get("buy", 0))
    sell_open = int(side_counts.get("sell", 0))
    buy_clear = buy_open < max_allowed
    sell_clear = sell_open < max_allowed
    return [
        FilterReport(
            filter_id="max_trades_buy",
            display_name="Max Open Trades (Buy)",
            enabled=True,
            is_clear=buy_clear,
            current_value=f"{buy_open}/{max_allowed}",
            threshold=f"Max: {max_allowed}",
            block_reason=f"{buy_open} >= {max_allowed}" if not buy_clear else None,
        ),
        FilterReport(
            filter_id="max_trades_sell",
            display_name="Max Open Trades (Sell)",
            enabled=True,
            is_clear=sell_clear,
            current_value=f"{sell_open}/{max_allowed}",
            threshold=f"Max: {max_allowed}",
            block_reason=f"{sell_open} >= {max_allowed}" if not sell_clear else None,
        ),
    ]

def report_trial10_entry_gates(eval_result: Optional[dict]) -> FilterReport:
    """Report Trial #10 proof-based zone/tier gate status from the latest evaluation."""
    gate_data = dict((eval_result or {}).get("trial10_entry_gates") or {})
    if not gate_data:
        return FilterReport(
            filter_id="trial10_entry_gates",
            display_name="Trial #10 Entry Gates",
            enabled=True,
            is_clear=True,
            current_value="Awaiting eval",
            explanation="No recent Trial #10 evaluation available yet.",
        )

    zone = dict(gate_data.get("zone") or {})
    tier = dict(gate_data.get("tier") or {})

    def _status_clear(status: str) -> bool:
        return status not in ("blocked", "failed")

    zone_status = str(zone.get("status") or "idle")
    tier_status = str(tier.get("status") or "idle")
    zone_msg = str(zone.get("message") or "Awaiting alignment")
    tier_msg = str(tier.get("message") or "Awaiting tier touch")

    zone_threshold = ""
    if bool(zone.get("recent_cross_required", False)):
        zone_threshold = f"Cross/reclaim within {int(zone.get('lookback_bars') or 0)} bars"

    tier_threshold = ""
    if bool(tier.get("reclaim_enabled", False)):
        tier_threshold = f"Touch then reclaim EMA{int(tier.get('reclaim_period') or 0)}"

    zone_filter = FilterReport(
        filter_id="trial10_zone_gate",
        display_name="Zone Gate",
        enabled=bool(zone.get("enabled", True)),
        is_clear=_status_clear(zone_status),
        current_value=zone_msg,
        threshold=zone_threshold,
        block_reason=zone_msg if zone_status == "blocked" else None,
        explanation=zone_msg,
        metadata={"status": zone_status, "mode": zone.get("mode")},
    )
    resumption_gate = dict(tier.get("resumption_gate") or (eval_result or {}).get("trial10_resumption_gate") or {})
    tier_meta: dict = {"status": tier_status, "tier": tier.get("tier")}
    if resumption_gate.get("resumption_gate_applicable"):
        tier_meta["resumption_gate"] = resumption_gate
    tier_filter = FilterReport(
        filter_id="trial10_tier_gate",
        display_name="Tier Gate",
        enabled=bool(tier.get("enabled", True)),
        is_clear=_status_clear(tier_status),
        current_value=tier_msg,
        threshold=tier_threshold,
        block_reason=tier_msg if tier_status == "blocked" else None,
        explanation=tier_msg,
        metadata=tier_meta,
    )

    blocking_msgs = [msg for status, msg in ((zone_status, zone_msg), (tier_status, tier_msg)) if status == "blocked"]
    overall_clear = len(blocking_msgs) == 0

    # Extract Trial 10 classification fields from eval_result
    tier_class = (eval_result or {}).get("trial10_tier_class")
    entry_class = (eval_result or {}).get("trial10_entry_class")
    setup_state = str((eval_result or {}).get("trial10_setup_state") or ("clean" if overall_clear else "blocked")).upper()

    display_parts = []
    if setup_state:
        display_parts.append(f"Setup: {setup_state}")
    if entry_class:
        display_parts.append(f"Entry: {entry_class.replace('_', ' ')}")
    if tier_class:
        display_parts.append(f"Tier: {tier_class}")
    current_display = " | ".join(display_parts) if display_parts else "Zone + Tier flow"

    return FilterReport(
        filter_id="trial10_entry_gates",
        display_name="Trial #10 Entry Gates",
        enabled=True,
        is_clear=overall_clear,
        current_value=current_display,
        block_reason=" | ".join(blocking_msgs) if blocking_msgs else None,
        explanation=" | ".join(blocking_msgs) if blocking_msgs else "Trial #10 entry flow is clear; advisory overlays may still reduce size.",
        sub_filters=[zone_filter, tier_filter],
        metadata={
            "zone_status": zone_status,
            "tier_status": tier_status,
            "tier_class": tier_class,
            "entry_class": entry_class,
            "setup_state": setup_state,
        },
    )


def report_trial10_pullback_quality(eval_result: Optional[dict]) -> FilterReport:
    """Report Trial #10 pullback quality for the latest tier touch/reclaim evaluation."""
    quality = dict((eval_result or {}).get("trial10_pullback_quality") or {})
    if not quality:
        return FilterReport(
            filter_id="trial10_pullback_quality",
            display_name="Pullback Quality",
            enabled=True,
            is_clear=True,
            current_value="Awaiting tier touch",
            explanation="No recent Trial #10 pullback-quality sample available yet.",
        )

    label = str(quality.get("label") or "neutral").lower()
    tier = quality.get("tier")
    bars = int(quality.get("pullback_bar_count") or 0)
    ratio = float(quality.get("structure_ratio") or 0.0)
    dampener = float(quality.get("dampener_multiplier") or 1.0)
    applicable = bool(quality.get("applicable", False))

    current_value = f"{label.upper()} | {bars} bars | ratio {ratio:.2f}"
    tier_class = "shallow" if tier in (17, 21) else ("standard" if tier in (27, 33) else "deep") if tier else ""
    threshold = f"Tier {tier} ({tier_class})" if tier is not None else ""
    explanation = str(quality.get("reason") or "").strip() or "Pullback quality measured from the touch bar."
    if applicable and label == "sloppy" and tier_class == "shallow":
        explanation += f" Advisory only: sloppy shallow tier {tier} will be floor-sized ({bars} bars, ratio={ratio:.2f})."
    elif applicable and label == "sloppy":
        explanation += f" Sloppy pullback on {tier_class} tier {tier} (telemetry only, no block)."
    elif not applicable and label == "sloppy":
        explanation += " Sloppy but not on an applicable tier (telemetry only)."

    return FilterReport(
        filter_id="trial10_pullback_quality",
        display_name="Pullback Quality",
        enabled=bool(quality.get("enabled", True)),
        is_clear=True,
        current_value=current_value,
        threshold=threshold,
        explanation=explanation,
        metadata={**quality, "tier_class": tier_class},
    )


def report_open_trade_cap_by_entry_type(entry_type: str, open_count: int, max_allowed: int) -> FilterReport:
    """Report open trade cap for a specific entry type."""
    label = "Zone Entry Cap" if entry_type == "zone_entry" else "Tiered Pullback Cap"
    is_clear = open_count < max_allowed
    return FilterReport(
        filter_id=f"{entry_type}_cap", display_name=label,
        enabled=True, is_clear=is_clear,
        current_value=f"{open_count}/{max_allowed}",
        threshold=f"Max: {max_allowed}",
        block_reason=f"{open_count} >= {max_allowed}" if not is_clear else None,
    )


def report_spread(spread_pips: float, max_spread: Optional[float]) -> FilterReport:
    """Report spread status."""
    if max_spread is None:
        return FilterReport(
            filter_id="spread", display_name="Spread",
            enabled=False, is_clear=True, current_value=f"{spread_pips:.1f}p",
        )
    is_clear = spread_pips <= max_spread
    return FilterReport(
        filter_id="spread", display_name="Spread",
        enabled=True, is_clear=is_clear,
        current_value=f"{spread_pips:.1f}p",
        threshold=f"Max: {max_spread}p",
        block_reason=f"Spread {spread_pips:.1f}p > {max_spread}p" if not is_clear else None,
    )


def report_t6_dead_zone(policy, daily_reset_state: dict) -> FilterReport:
    """Report Trial #6 Dead Zone status (configurable hours)."""
    block_active = daily_reset_state.get("daily_reset_block_active", False) if daily_reset_state else False
    utc_hour = datetime.now(timezone.utc).hour
    start_h = getattr(policy, "dead_zone_start_hour_utc", 21)
    end_h = getattr(policy, "dead_zone_end_hour_utc", 2)
    enabled = getattr(policy, "dead_zone_enabled", True)
    return FilterReport(
        filter_id="t6_dead_zone", display_name=f"Dead Zone ({start_h:02d}-{end_h:02d} UTC)",
        enabled=enabled, is_clear=not block_active,
        current_value=f"UTC {utc_hour:02d}:00",
        threshold=f"{start_h:02d}:00-{end_h:02d}:00 UTC",
        block_reason="Inside dead zone" if block_active else None,
    )


def report_t6_m3_trend(eval_result: Optional[dict]) -> FilterReport:
    """Report Trial #6 M3 Slope Trend status."""
    trend_result = eval_result.get("trend_result") if eval_result else None
    if not trend_result:
        return FilterReport(
            filter_id="t6_m3_trend", display_name="M3 Slope Trend",
            enabled=True, is_clear=True,
            current_value="No data",
        )
    trend = trend_result.get("trend", "NONE")
    is_clear = trend != "NONE"
    reasons = trend_result.get("reasons", [])
    reason_str = "; ".join(reasons) if reasons else None
    return FilterReport(
        filter_id="t6_m3_trend", display_name="M3 Slope Trend",
        enabled=True, is_clear=is_clear,
        current_value=trend,
        block_reason=f"NONE – {reason_str}" if not is_clear and reason_str else ("NONE – no trend" if not is_clear else None),
        metadata={"trend": trend},
    )


def report_t6_bb_reversal_cap(policy, store, profile_name: str) -> FilterReport:
    """Report Trial #6 BB Reversal position cap."""
    max_pos = getattr(policy, "max_bb_reversal_positions", 3)
    bb_count = 0
    try:
        open_trades = store.list_open_trades(profile_name)
        for t in open_trades:
            row = dict(t) if hasattr(t, "keys") else t
            if row.get("entry_type") == "bb_reversal":
                bb_count += 1
    except Exception:
        pass
    is_clear = bb_count < max_pos
    return FilterReport(
        filter_id="t6_bb_reversal_cap", display_name="BB Reversal Positions",
        enabled=getattr(policy, "bb_reversal_enabled", True),
        is_clear=is_clear,
        current_value=f"{bb_count}/{max_pos}",
        block_reason=f"At max ({bb_count}/{max_pos})" if not is_clear else None,
    )


def report_t8_exit_strategy(policy) -> FilterReport:
    """Report Trial #8/#9 exit strategy."""
    exit_strategy = str(getattr(policy, "exit_strategy", "tp1_be_trail") or "tp1_be_trail")
    if getattr(policy, "type", None) == "kt_cg_trial_10" and bool(getattr(policy, "bucketed_exit_enabled", False)):
        q_tp1 = float(getattr(policy, "quick_tp1_pips", 4.0))
        q_pct = float(getattr(policy, "quick_tp1_close_pct", 85.0))
        q_be = float(getattr(policy, "quick_be_spread_plus_pips", 0.3))
        s_tp1 = float(getattr(policy, "tp1_pips", 6.0))
        s_pct = float(getattr(policy, "tp1_close_pct", 70.0))
        s_be = float(getattr(policy, "be_spread_plus_pips", 0.5))
        r_tp1 = float(getattr(policy, "runner_tp1_pips", 8.0))
        r_pct = float(getattr(policy, "runner_tp1_close_pct", 55.0))
        r_be = float(getattr(policy, "runner_be_spread_plus_pips", 0.5))
        _trail_escalation_enabled = bool(getattr(policy, "trail_escalation_enabled", False))
        _std_trail = "M1->M5->M15 ratchet" if _trail_escalation_enabled else "M5 trail"
        _runner_trail = "M1->M5->M15 ratchet" if _trail_escalation_enabled else "M5 trail"
        label = (
            f"Quick {q_tp1:.0f}p/{q_pct:.0f}% + BE +{q_be:.1f}p + M1 trail | "
            f"Std {s_tp1:.0f}p/{s_pct:.0f}% + BE +{s_be:.1f}p + {_std_trail} | "
            f"Runner {r_tp1:.0f}p/{r_pct:.0f}% + BE +{r_be:.1f}p + {_runner_trail}"
        )
        if _trail_escalation_enabled:
            _et1 = float(getattr(policy, "trail_escalation_tier1_pips", 10.0))
            _et2 = float(getattr(policy, "trail_escalation_tier2_pips", 20.0))
            label += f" | Escalation: +{_et1:.0f}p->M5, +{_et2:.0f}p->M15"
    elif exit_strategy == "ema_scale_runner":
        ema_fast = getattr(policy, "m1_exit_ema_fast", 9)
        ema_slow = getattr(policy, "m1_exit_ema_slow", 21)
        scale_pct = getattr(policy, "scale_out_pct", 50.0)
        label = f"EMA scale-out + runner (EMA{ema_fast}/{ema_slow}, {scale_pct:.0f}%)"
    elif exit_strategy == "none":
        label = "None (broker TP/SL only)"
    elif exit_strategy == "tp1_be_m5_trail":
        tp1_pips = getattr(policy, "tp1_pips", 6.0)
        close_pct = getattr(policy, "tp1_close_pct", 80.0)
        be_pips = getattr(policy, "be_spread_plus_pips", 0.5)
        m5_period = getattr(policy, "trail_m5_ema_period", 20)
        label = f"TP1 {tp1_pips}p/{close_pct:.0f}% + BE +{be_pips}p + M5 EMA{m5_period} trail"
    elif exit_strategy == "tp1_be_hwm_trail":
        tp1_pips = getattr(policy, "tp1_pips", 6.0)
        close_pct = getattr(policy, "tp1_close_pct", 80.0)
        be_pips = getattr(policy, "be_spread_plus_pips", 0.5)
        hwm_pips = getattr(policy, "hwm_trail_pips", 3.0)
        label = f"TP1 {tp1_pips}p/{close_pct:.0f}% + BE +{be_pips}p + HWM {hwm_pips}p tick trail"
    else:
        tp1_pips = getattr(policy, "tp1_pips", 4.0)
        close_pct = getattr(policy, "tp1_close_pct", 50.0)
        be_pips = getattr(policy, "be_spread_plus_pips", 2.0)
        trail_period = getattr(policy, "trail_ema_period", 21)
        label = f"TP1 {tp1_pips}p/{close_pct:.0f}% + BE +{be_pips}p + M1 EMA{trail_period} trail"
    return FilterReport(
        filter_id="t8_exit_strategy",
        display_name="Exit",
        enabled=True,
        is_clear=True,
        current_value=label,
    )


def report_daily_level_filter(policy, tick, side: str, pip_size: float, snapshot: Optional[dict] = None) -> FilterReport:
    """Report Trial #8 Daily Level Filter status using a snapshot from DailyLevelFilter.get_state_snapshot()."""
    enabled = getattr(policy, "use_daily_level_filter", False)
    if not enabled:
        return FilterReport(
            filter_id="daily_level_filter", display_name="Daily Level Filter",
            enabled=False, is_clear=True,
        )
    if snapshot is None:
        return FilterReport(
            filter_id="daily_level_filter", display_name="Daily Level Filter",
            enabled=True, is_clear=True, current_value="Awaiting state",
        )

    watched_high = snapshot.get("watched_high")
    watched_low = snapshot.get("watched_low")
    high_confirmed = bool(snapshot.get("high_breakout_confirmed", False))
    low_confirmed = bool(snapshot.get("low_breakout_confirmed", False))
    buffer_pips = float(getattr(policy, "daily_level_filter_buffer_pips", 3.0))
    buffer = buffer_pips * pip_size
    current_price = tick.ask if side == "buy" else tick.bid

    block_reason = None
    if side == "buy" and watched_high is not None:
        if current_price >= watched_high - buffer and current_price < watched_high:
            block_reason = f"BUY within {buffer_pips}p below watched_high={watched_high:.3f}"
        elif not high_confirmed and current_price >= watched_high:
            block_reason = f"BUY above {watched_high:.3f} — breakout not yet confirmed"
    elif side == "sell" and watched_low is not None:
        if current_price <= watched_low + buffer and current_price > watched_low:
            block_reason = f"SELL within {buffer_pips}p above watched_low={watched_low:.3f}"
        elif not low_confirmed and current_price <= watched_low:
            block_reason = f"SELL below {watched_low:.3f} — breakout not yet confirmed"

    high_str = f"{watched_high:.3f}" if watched_high is not None else "—"
    low_str = f"{watched_low:.3f}" if watched_low is not None else "—"
    return FilterReport(
        filter_id="daily_level_filter", display_name="Daily Level Filter",
        enabled=True, is_clear=block_reason is None,
        current_value=f"H:{high_str} L:{low_str}",
        threshold=f"Buffer: {buffer_pips}p",
        block_reason=block_reason,
    )


def report_ntz_status(ntz_snapshot: Optional[dict], tick, pip_size: float) -> FilterReport:
    """Report Trial #9 No-Trade Zone status (including Fibonacci Pivot levels)."""
    if ntz_snapshot is None:
        return FilterReport(
            filter_id="ntz", display_name="No-Trade Zones",
            enabled=False, is_clear=True,
        )
    ntz_blocking_enabled = ntz_snapshot.get("enabled", False)
    fib_pivots_enabled = ntz_snapshot.get("fib_pivots_enabled", False)
    fib_levels_data = ntz_snapshot.get("fib_levels", {})
    # Return disabled only if both NTZ blocking AND fib pivots have no data
    if not ntz_blocking_enabled and not (fib_pivots_enabled and fib_levels_data):
        return FilterReport(
            filter_id="ntz", display_name="No-Trade Zones",
            enabled=False, is_clear=True,
        )
    levels = ntz_snapshot.get("levels", {})
    buffer_pips = float(ntz_snapshot.get("buffer_pips", 10.0))
    buffer = buffer_pips * pip_size
    current_price = (tick.bid + tick.ask) / 2.0
    fib_levels = fib_levels_data  # already fetched above

    blocked_label = None
    closest_dist = None
    level_parts = []
    for label, val in levels.items():
        if val is not None:
            dist = abs(current_price - val)
            dist_pips = dist / pip_size
            in_zone = ntz_blocking_enabled and dist <= buffer
            level_parts.append(f"{label}={val:.3f}({dist_pips:.1f}p{'*' if in_zone else ''})")
            if in_zone and (closest_dist is None or dist < closest_dist):
                closest_dist = dist
                blocked_label = label

    block_reason = None
    if ntz_blocking_enabled and blocked_label is not None and closest_dist is not None:
        block_reason = f"Price within {closest_dist / pip_size:.1f}p of {blocked_label} (buffer={buffer_pips}p)"
    explanation = None
    if level_parts:
        if not ntz_blocking_enabled:
            explanation = f"NTZ blocking OFF; showing {len(level_parts)} reference levels"
        elif block_reason is not None:
            explanation = f"Blocking at {blocked_label}; monitoring {len(level_parts)} levels"
        else:
            explanation = f"Clear; monitoring {len(level_parts)} levels"

    # Build metadata with fib pivot info
    meta: dict = {}
    if fib_pivots_enabled:
        meta["fib_pivots_enabled"] = True
        meta["fib_levels"] = fib_levels
        meta["fib_levels_count"] = len(fib_levels)
    else:
        meta["fib_pivots_enabled"] = False

    display_name = "No-Trade Zones" if ntz_blocking_enabled else "NTZ Levels (blocking OFF)"
    if fib_pivots_enabled:
        display_name += " + Fib Pivots"

    return FilterReport(
        filter_id="ntz", display_name=display_name,
        enabled=True, is_clear=block_reason is None,
        current_value=" | ".join(level_parts) if level_parts else "No levels",
        threshold=f"Buffer: {buffer_pips}p" if ntz_blocking_enabled else "Blocking disabled",
        block_reason=block_reason,
        explanation=explanation,
        metadata=meta,
    )


def report_kill_switch_status(policy, data_by_tf: dict, tick, pip_size: float, side: str) -> FilterReport:
    """Report Trial #9 Kill Switch status: M5 trend + M1-200 EMA cross.

    Active when M5 Bull + M1 completed bar close < EMA200, or M5 Bear + M1 close > EMA200.
    Blocks both exits and new entries until M1 closes back on the trend side or M5 trend changes.
    """
    enabled = getattr(policy, "kill_switch_enabled", True)
    if not enabled:
        return FilterReport(
            filter_id="kill_switch", display_name="Kill Switch (M5+M1-200)",
            enabled=False, is_clear=True,
        )

    m1_df = data_by_tf.get("M1")
    m5_df = data_by_tf.get("M5")
    if m1_df is None or m1_df.empty or len(m1_df) < 202:
        return FilterReport(
            filter_id="kill_switch", display_name="Kill Switch (M5+M1-200)",
            enabled=True, is_clear=True,
            current_value="Insufficient M1 data",
        )

    try:
        from core.indicators import ema as ema_fn
        close = m1_df["close"].astype(float)
        ema200 = ema_fn(close, 200)
        if ema200.empty or pd.isna(ema200.iloc[-2]):
            return FilterReport(
                filter_id="kill_switch", display_name="Kill Switch (M5+M1-200)",
                enabled=True, is_clear=True,
                current_value="EMA200 not ready",
            )
        ema200_val = float(ema200.iloc[-2])   # completed bar (same as exit logic)
        last_close = float(close.iloc[-2])     # completed bar
        current_mid = (tick.bid + tick.ask) / 2.0
        dist_pips = (current_mid - ema200_val) / pip_size

        # M5 trend
        m5_trend_str = "unknown"
        m5_is_bull = None
        if m5_df is not None and not m5_df.empty and len(m5_df) >= 22:
            m5_close = m5_df["close"].astype(float)
            _fast_p = int(getattr(policy, "m5_trend_ema_fast", 9))
            _slow_p = int(getattr(policy, "m5_trend_ema_slow", 21))
            _m5_fast = m5_close.ewm(span=_fast_p, adjust=False).mean()
            _m5_slow = m5_close.ewm(span=_slow_p, adjust=False).mean()
            m5_is_bull = float(_m5_fast.iloc[-1]) > float(_m5_slow.iloc[-1])
            m5_trend_str = "Bull" if m5_is_bull else "Bear"

        zone_action = str(getattr(policy, "kill_switch_zone_entry_action", "kill"))
        active = False
        if m5_is_bull is True and last_close < ema200_val:
            active = True
        elif m5_is_bull is False and last_close > ema200_val:
            active = True

        block_reason = None
        if active:
            cmp = "<" if m5_is_bull else ">"
            block_reason = (
                f"Kill Switch ACTIVE: M5={m5_trend_str}, M1 close {last_close:.3f} "
                f"{cmp} EMA200 {ema200_val:.3f} — exits+entries blocked"
            )

        return FilterReport(
            filter_id="kill_switch", display_name="Kill Switch (M5+M1-200)",
            enabled=True, is_clear=not active,
            current_value=f"M5={m5_trend_str} | EMA200={ema200_val:.3f} ({dist_pips:+.1f}p) | ZoneEntry={zone_action}",
            block_reason=block_reason,
        )
    except Exception:
        return FilterReport(
            filter_id="kill_switch", display_name="Kill Switch (M5+M1-200)",
            enabled=True, is_clear=True,
            current_value="Error computing kill switch status",
        )


# ---------------------------------------------------------------------------
# Context collectors
# ---------------------------------------------------------------------------


def collect_trial_4_context(
    policy, data_by_tf: dict, tick, tier_state: dict,
    eval_result: Optional[dict], pip_size: float,
) -> list[ContextItem]:
    """Collect context items for Trial #4 dashboard display."""
    items: list[ContextItem] = []

    # M3 Trend (compute from data)
    m3_trend = "N/A"
    m3_df = data_by_tf.get("M3")
    if m3_df is not None and not m3_df.empty:
        from core.indicators import ema as ema_fn
        m3_close = m3_df["close"]
        fast_p = getattr(policy, "m3_trend_ema_fast", 5)
        slow_p = getattr(policy, "m3_trend_ema_slow", 9)
        if len(m3_df) > slow_p:
            fast_v = float(ema_fn(m3_close, fast_p).iloc[-1])
            slow_v = float(ema_fn(m3_close, slow_p).iloc[-1])
            m3_trend = "BULL" if fast_v > slow_v else "BEAR"
            items.append(ContextItem(f"M3 EMA{fast_p}", f"{fast_v:.3f}", "trend"))
            items.append(ContextItem(f"M3 EMA{slow_p}", f"{slow_v:.3f}", "trend"))
    items.insert(0, ContextItem("M3 Trend", m3_trend, "trend"))

    # M1 Zone Entry EMAs
    m1_df = data_by_tf.get("M1")
    if m1_df is not None and not m1_df.empty:
        from core.indicators import ema as ema_fn
        m1_close = m1_df["close"]
        ze_fast = getattr(policy, "m1_zone_entry_ema_fast", 5)
        ze_slow = getattr(policy, "m1_zone_entry_ema_slow", 9)
        if len(m1_df) > ze_slow:
            zf = float(ema_fn(m1_close, ze_fast).iloc[-1])
            zs = float(ema_fn(m1_close, ze_slow).iloc[-1])
            items.append(ContextItem(f"M1 EMA{ze_fast}", f"{zf:.3f}", "zone_entry"))
            items.append(ContextItem(f"M1 EMA{ze_slow}", f"{zs:.3f}", "zone_entry"))

    # Bid/Ask
    items.append(ContextItem("Bid", f"{tick.bid:.3f}", "price"))
    items.append(ContextItem("Ask", f"{tick.ask:.3f}", "price"))
    items.append(ContextItem("Spread", f"{(tick.ask - tick.bid) / pip_size:.1f}p", "price"))

    # Tier state
    fired = [str(t) for t, v in sorted(tier_state.items()) if v]
    avail = [str(t) for t, v in sorted(tier_state.items()) if not v]
    items.append(ContextItem("Tiers Fired", ", ".join(fired) if fired else "None", "tiers"))
    items.append(ContextItem("Tiers Available", ", ".join(avail) if avail else "None", "tiers"))

    return items


def collect_trial_5_context(
    policy, data_by_tf: dict, tick, tier_state: dict,
    eval_result: Optional[dict], pip_size: float,
    exhaustion_result: Optional[dict] = None,
    daily_reset_state: Optional[dict] = None,
) -> list[ContextItem]:
    """Collect context items for Trial #5 — extends Trial #4 context."""
    items = collect_trial_4_context(policy, data_by_tf, tick, tier_state, eval_result, pip_size)

    # Exhaustion
    if exhaustion_result:
        zone = exhaustion_result.get("zone", "N/A")
        ratio = exhaustion_result.get("extension_ratio")
        items.append(ContextItem("Exhaustion Zone", zone, "exhaustion"))
        if ratio is not None:
            items.append(ContextItem("Extension Ratio", f"{ratio:.1f}x", "exhaustion"))

    # Daily Reset
    if daily_reset_state:
        h = daily_reset_state.get("daily_reset_high")
        l = daily_reset_state.get("daily_reset_low")
        settled = daily_reset_state.get("daily_reset_settled", False)
        if h is not None:
            items.append(ContextItem("Daily High", f"{h:.3f}", "daily"))
        if l is not None:
            items.append(ContextItem("Daily Low", f"{l:.3f}", "daily"))
        items.append(ContextItem("H/L Settled", str(settled), "daily"))

    # Dead zone countdown
    utc_hour = datetime.now(timezone.utc).hour
    block_active = daily_reset_state.get("daily_reset_block_active", False) if daily_reset_state else False
    if block_active:
        # Hours until 02:00 UTC
        hours_left = (2 - utc_hour) % 24
        items.append(ContextItem("Dead Zone Ends In", f"{hours_left}h", "daily"))

    return items


def collect_trial_7_context(
    policy, data_by_tf: dict, tick, tier_state: dict,
    eval_result: Optional[dict], pip_size: float,
    exhaustion_result: Optional[dict] = None,
) -> list[ContextItem]:
    """Collect context items for Trial #7 dashboard display."""
    items: list[ContextItem] = []

    # M5 Trend
    m5_df = data_by_tf.get("M5")
    trend_added = False
    if m5_df is not None and not m5_df.empty:
        from core.indicators import ema as ema_fn
        from core.execution_engine import _resolve_m5_trend_close_series, _trial7_m5_ema_gap_pips
        fast_p = getattr(policy, "m5_trend_ema_fast", 9)
        slow_p = getattr(policy, "m5_trend_ema_slow", 21)
        current_price = None
        try:
            current_price = (float(tick.bid) + float(tick.ask)) / 2.0
        except Exception:
            current_price = None
        m5_close, m5_source = _resolve_m5_trend_close_series(
            policy,
            data_by_tf,
            current_price=current_price,
        )
        if len(m5_close) >= max(fast_p, slow_p):
            fast_v = float(ema_fn(m5_close, fast_p).iloc[-1])
            slow_v = float(ema_fn(m5_close, slow_p).iloc[-1])
            trend = "BULL" if fast_v > slow_v else "BEAR"
            items.append(ContextItem("M5 Trend", trend, "trend"))
            items.append(ContextItem(f"M5 EMA{fast_p}", f"{fast_v:.3f}", "trend"))
            items.append(ContextItem(f"M5 EMA{slow_p}", f"{slow_v:.3f}", "trend"))
            items.append(
                ContextItem(
                    "M5 Trend Source",
                    "Synthetic Live M5" if m5_source == "synthetic_live_m5" else "Closed M5",
                    "trend",
                )
            )
            gap_pips, _fv, _sv = _trial7_m5_ema_gap_pips(m5_close, int(fast_p), int(slow_p), pip_size)
            min_gap = float(getattr(policy, "m5_min_ema_distance_pips", 1.0))
            items.append(ContextItem("EMA9-EMA21 Gap (pips)", f"{gap_pips:.2f}", "trend"))
            items.append(ContextItem("Min Gap Threshold (pips)", f"{min_gap:.2f}", "trend"))
            trend_added = True
    if not trend_added:
        items.append(ContextItem("M5 Trend", "N/A (warming up)", "trend"))

    # M1 Zone Entry EMAs
    m1_df = data_by_tf.get("M1")
    zone_mode = str(getattr(policy, "zone_entry_mode", "ema_cross"))
    if zone_mode not in ("ema_cross", "price_vs_ema5"):
        zone_mode = "ema_cross"
    items.append(ContextItem("Zone Entry Mode", zone_mode, "zone_entry"))
    if m1_df is not None and not m1_df.empty:
        from core.indicators import ema as ema_fn
        m1_close = m1_df["close"]
        ze_fast = getattr(policy, "m1_zone_entry_ema_fast", 5)
        ze_slow = getattr(policy, "m1_zone_entry_ema_slow", 9)
        if len(m1_df) > max(ze_fast, ze_slow):
            zf = float(ema_fn(m1_close, ze_fast).iloc[-1])
            zs = float(ema_fn(m1_close, ze_slow).iloc[-1])
            items.append(ContextItem(f"M1 EMA{ze_fast}", f"{zf:.3f}", "zone_entry"))
            items.append(ContextItem(f"M1 EMA{ze_slow}", f"{zs:.3f}", "zone_entry"))
            if zone_mode == "price_vs_ema5":
                ze_price = int(getattr(policy, "m1_zone_entry_price_ema_period", 5))
                if len(m1_df) > ze_price:
                    ema_price = float(ema_fn(m1_close, ze_price).iloc[-1])
                    # Use same price as execution: ask for BULL (buy), bid for BEAR (sell)
                    if m5_df is not None and not m5_df.empty:
                        from core.execution_engine import _resolve_m5_trend_close_series

                        fast_p = getattr(policy, "m5_trend_ema_fast", 9)
                        slow_p = getattr(policy, "m5_trend_ema_slow", 21)
                        current_price = None
                        try:
                            current_price = (float(tick.bid) + float(tick.ask)) / 2.0
                        except Exception:
                            current_price = None
                        m5_close, _m5_source = _resolve_m5_trend_close_series(
                            policy,
                            data_by_tf,
                            current_price=current_price,
                        )
                        if len(m5_close) >= max(fast_p, slow_p):
                            fast_v = float(ema_fn(m5_close, fast_p).iloc[-1])
                            slow_v = float(ema_fn(m5_close, slow_p).iloc[-1])
                            is_bull = fast_v > slow_v
                        else:
                            is_bull = True  # default
                    else:
                        is_bull = True  # default
                    if is_bull:
                        items.append(
                            ContextItem(
                                "Zone Condition Snapshot",
                                f"ask {tick.ask:.3f} vs EMA{ze_price} {ema_price:.3f}",
                                "zone_entry",
                            )
                        )
                    else:
                        items.append(
                            ContextItem(
                                "Zone Condition Snapshot",
                                f"bid {tick.bid:.3f} vs EMA{ze_price} {ema_price:.3f}",
                                "zone_entry",
                            )
                        )
            else:
                items.append(
                    ContextItem(
                        "Zone Condition Snapshot",
                        f"EMA{ze_fast} {zf:.3f} vs EMA{ze_slow} {zs:.3f}",
                        "zone_entry",
                    )
                )

    items.append(ContextItem("Bid", f"{tick.bid:.3f}", "price"))
    items.append(ContextItem("Ask", f"{tick.ask:.3f}", "price"))
    items.append(ContextItem("Spread", f"{(tick.ask - tick.bid) / pip_size:.1f}p", "price"))

    ex_enabled = bool(getattr(policy, "trend_exhaustion_enabled", False))
    ex_mode = str(getattr(policy, "trend_exhaustion_mode", "session_and_side"))
    items.append(ContextItem("Trend Exhaustion", "ON" if ex_enabled else "OFF", "exhaustion"))
    if ex_enabled:
        items.append(ContextItem("Exhaustion Mode", ex_mode, "exhaustion"))
    base_tp = max(0.1, float(getattr(policy, "tp_pips", 4.0)))
    adaptive_tp_enabled = bool(getattr(policy, "trend_exhaustion_adaptive_tp_enabled", False))
    ext_off = max(0.0, float(getattr(policy, "trend_exhaustion_tp_extended_offset_pips", 1.0)))
    very_off = max(0.0, float(getattr(policy, "trend_exhaustion_tp_very_extended_offset_pips", 2.0)))
    min_tp = max(0.1, float(getattr(policy, "trend_exhaustion_tp_min_pips", 0.5)))
    zone_for_tp = str((exhaustion_result or {}).get("zone", "normal")).lower()
    if adaptive_tp_enabled and zone_for_tp == "very_extended":
        active_tp = max(min_tp, base_tp - very_off)
    elif adaptive_tp_enabled and zone_for_tp == "extended":
        active_tp = max(min_tp, base_tp - ext_off)
    else:
        active_tp = base_tp
    items.append(ContextItem("Base TP (pips)", f"{base_tp:.2f}", "tp"))
    items.append(ContextItem("Adaptive TP", "ON" if adaptive_tp_enabled else "OFF", "tp"))
    items.append(ContextItem("Active TP (pips)", f"{active_tp:.2f}", "tp"))
    if adaptive_tp_enabled:
        items.append(
            ContextItem(
                "Adaptive TP Rule",
                f"extended:-{ext_off:.2f} very:-{very_off:.2f} min:{min_tp:.2f}",
                "tp",
            )
        )

    rr_enabled = bool(getattr(policy, "use_reversal_risk_score", False))
    items.append(ContextItem("Reversal Risk", "ON" if rr_enabled else "OFF", "reversal_risk"))
    rr_result = None
    if eval_result and isinstance(eval_result.get("reversal_risk_result"), dict):
        rr_result = eval_result.get("reversal_risk_result")
    elif exhaustion_result and (
        exhaustion_result.get("rr_score") is not None or exhaustion_result.get("rr_tier") is not None
    ):
        rr_result = {
            "score": exhaustion_result.get("rr_score"),
            "tier": exhaustion_result.get("rr_tier"),
            "response": {"lot_multiplier": exhaustion_result.get("rr_lot_multiplier")},
        }
    if rr_enabled:
        if isinstance(rr_result, dict):
            rr_score = rr_result.get("score")
            rr_tier = str(rr_result.get("tier", "low")).upper()
            rr_regime = str(rr_result.get("regime", "transition")).lower()
            rr_resp = rr_result.get("response") if isinstance(rr_result.get("response"), dict) else {}
            rr_lot_x = rr_resp.get("lot_multiplier")
            rr_min_tier = rr_resp.get("min_tier_ema")
            rr_zone_block = bool(rr_resp.get("block_zone_entry", False))
            rr_managed_exit = bool(rr_resp.get("use_managed_exit", False))
            if rr_score is not None:
                try:
                    items.append(ContextItem("RR Score", f"{float(rr_score):.2f}", "reversal_risk"))
                except Exception:
                    pass
            items.append(ContextItem("RR Tier", rr_tier, "reversal_risk"))
            items.append(ContextItem("Regime", rr_regime, "reversal_risk"))
            if rr_lot_x is not None:
                try:
                    items.append(ContextItem("RR Lot Multiplier", f"{float(rr_lot_x):.2f}x", "reversal_risk"))
                except Exception:
                    pass
            items.append(ContextItem("RR Zone Block", "ON" if rr_zone_block else "OFF", "reversal_risk"))
            items.append(ContextItem("RR Managed Exit", "ON" if rr_managed_exit else "OFF", "reversal_risk"))
            if rr_min_tier is not None:
                try:
                    items.append(ContextItem("RR Min Tier EMA", f"{int(rr_min_tier)}+", "reversal_risk"))
                except Exception:
                    pass
        else:
            items.append(ContextItem("RR Score", "PENDING", "reversal_risk"))

    if exhaustion_result:
        zone = str(exhaustion_result.get("zone", "normal"))
        stretch = exhaustion_result.get("stretch_pips")
        p80 = exhaustion_result.get("threshold_p80")
        p90 = exhaustion_result.get("threshold_p90")
        sess = exhaustion_result.get("session")
        items.append(ContextItem("Exhaustion Zone", zone.upper(), "exhaustion"))
        if stretch is not None:
            items.append(ContextItem("Stretch (pips)", f"{float(stretch):.2f}", "exhaustion"))
        if p80 is not None and p90 is not None:
            items.append(ContextItem("P80/P90", f"{float(p80):.2f} / {float(p90):.2f}", "exhaustion"))
        if sess:
            items.append(ContextItem("Exhaustion Session", str(sess), "exhaustion"))
    elif ex_enabled:
        items.append(ContextItem("Exhaustion Zone", "PENDING", "exhaustion"))

    fired = [str(t) for t, v in sorted(tier_state.items()) if v]
    avail = [str(t) for t, v in sorted(tier_state.items()) if not v]
    items.append(ContextItem("Tiers Fired", ", ".join(fired) if fired else "None", "tiers"))
    items.append(ContextItem("Tiers Available", ", ".join(avail) if avail else "None", "tiers"))

    return items


def report_intraday_fib_corridor(snapshot: Optional[dict], tick, pip_size: float) -> FilterReport:
    """Report Trial #9 Intraday Fibonacci Corridor status."""
    if snapshot is None or not snapshot.get("enabled", False):
        return FilterReport(
            filter_id="intraday_fib_corridor", display_name="Intraday Fib Corridor",
            enabled=False, is_clear=True,
        )

    lower_level = snapshot.get("lower_level", "S1")
    upper_level = snapshot.get("upper_level", "R1")
    timeframe = snapshot.get("timeframe", "M15")
    lookback_bars = snapshot.get("lookback_bars", 16)
    lower_val = snapshot.get("lower_value")
    upper_val = snapshot.get("upper_value")
    corridor_state = snapshot.get("corridor_state")
    rolling_high = snapshot.get("rolling_high")
    rolling_low = snapshot.get("rolling_low")
    source_close = snapshot.get("source_close")
    calculation_mode = str(snapshot.get("calculation_mode") or "rolling_window")
    buffer_pips = snapshot.get("boundary_buffer_pips", 1.0)
    hysteresis_pips = snapshot.get("hysteresis_pips", 1.0)

    current_price = (tick.bid + tick.ask) / 2.0

    if lower_val is None or upper_val is None:
        calc_str = f"{timeframe} prev candle" if calculation_mode == "previous_candle" else f"{timeframe} x{lookback_bars}"
        return FilterReport(
            filter_id="intraday_fib_corridor", display_name="Intraday Fib Corridor",
            enabled=True, is_clear=True,
            current_value=f"{calc_str} | Awaiting data",
            explanation="No intraday fib levels computed yet",
        )

    if lower_val >= upper_val:
        return FilterReport(
            filter_id="intraday_fib_corridor", display_name="Intraday Fib Corridor",
            enabled=True, is_clear=False,
            current_value=f"{lower_level}={lower_val:.3f} >= {upper_level}={upper_val:.3f}",
            block_reason=f"Invalid bounds: {lower_level} >= {upper_level}",
        )

    is_inside = corridor_state is True
    price_vs = ""
    if current_price > upper_val:
        price_vs = f"above {upper_level}"
    elif current_price < lower_val:
        price_vs = f"below {lower_level}"
    else:
        price_vs = "inside corridor"

    calc_label = f"{timeframe} prev candle" if calculation_mode == "previous_candle" else f"{timeframe} x{lookback_bars}"
    current_str = (
        f"{calc_label} | Price {current_price:.3f} | "
        f"{lower_level}={lower_val:.3f} .. {upper_level}={upper_val:.3f} | {price_vs}"
    )
    block_reason = None if is_inside else f"Price {price_vs} — entries blocked"
    explanation = (
        f"{'ALLOWED' if is_inside else 'BLOCKED'}: {price_vs}"
        + (f" (hysteresis holding)" if corridor_state is False and lower_val <= current_price <= upper_val else "")
    )

    meta: dict[str, Any] = {
        "lower_level": lower_level,
        "upper_level": upper_level,
        "timeframe": timeframe,
        "lookback_bars": lookback_bars,
        "calculation_mode": calculation_mode,
        "lower_value": lower_val,
        "upper_value": upper_val,
        "rolling_high": rolling_high,
        "rolling_low": rolling_low,
        "source_close": source_close,
        "corridor_state": "inside" if is_inside else ("outside" if corridor_state is False else "undecided"),
        "boundary_buffer_pips": buffer_pips,
        "hysteresis_pips": hysteresis_pips,
    }
    # Include all fib levels in metadata
    fib_levels = snapshot.get("fib_levels")
    if fib_levels:
        meta["fib_levels"] = fib_levels

    return FilterReport(
        filter_id="intraday_fib_corridor", display_name="Intraday Fib Corridor",
        enabled=True, is_clear=is_inside,
        current_value=current_str,
        threshold=f"Buffer: {buffer_pips}p | Hysteresis: {hysteresis_pips}p",
        block_reason=block_reason,
        explanation=explanation,
        metadata=meta,
    )


def report_conviction_sizing(snapshot: Optional[dict]) -> FilterReport:
    """Report Trial #9 Conviction Sizing status."""
    if snapshot is None or not snapshot.get("enabled", False):
        return FilterReport(
            filter_id="conviction_sizing", display_name="Conviction Sizing",
            enabled=False, is_clear=True,
        )

    m5_bucket = snapshot.get("m5_bucket", "normal")
    m1_bucket = snapshot.get("m1_bucket", "neutral")
    multiplier = snapshot.get("multiplier", 1.0)
    lots = snapshot.get("conviction_lots", 0.0)
    base = snapshot.get("base_lots", 0.0)
    m5_spread = snapshot.get("m5_spread_pips", 0.0)
    m5_slope = snapshot.get("m5_slope_pips_per_bar", 0.0)
    m1_spread = snapshot.get("m1_spread_pips", 0.0)
    m1_compressing = snapshot.get("m1_compressing", False)

    current_str = f"M5:{m5_bucket.upper()} M1:{m1_bucket.upper()} → {multiplier:.2f}x = {lots:.2f} lots"
    threshold_str = f"Base: {base:.2f} lots"
    explanation = (
        f"M5 spread {m5_spread:.2f}p, slope {m5_slope:.2f}p/bar. "
        f"M1 spread {m1_spread:.2f}p{', compressing' if m1_compressing else ''}."
    )

    return FilterReport(
        filter_id="conviction_sizing", display_name="Conviction Sizing",
        enabled=True, is_clear=True,  # conviction sizing never blocks, only adjusts lots
        current_value=current_str,
        threshold=threshold_str,
        explanation=explanation,
        metadata=dict(snapshot),
    )


def report_runner_score(snapshot: Optional[dict]) -> FilterReport:
    """Report Trial #10 Runner Score v2 status (logging/dashboard only)."""
    if snapshot is None:
        return FilterReport(
            filter_id="runner_score", display_name="Runner Score",
            enabled=False, is_clear=True,
        )

    bucket = str(snapshot.get("runner_bucket", "floor")).upper()
    points = int(snapshot.get("runner_points", 0))
    atr_eligible = bool(snapshot.get("atr_eligible", False))
    atr_pips = float(snapshot.get("atr_stop_pips", 0.0))
    regime_pt = bool(snapshot.get("regime_point", False))
    m5_pt = bool(snapshot.get("m5_point", False))
    struct_pt = bool(snapshot.get("structure_point", False))
    fresh = bool(snapshot.get("fresh", False))
    bars_cross = snapshot.get("bars_since_cross")
    prior_ent = snapshot.get("prior_entries")
    freshness_mode = str(snapshot.get("freshness_mode", "strict")).lower()

    features = []
    if regime_pt:
        features.append("regime")
    if m5_pt:
        features.append("m5")
    if struct_pt:
        features.append("structure")
    feature_str = ", ".join(features) if features else "none"

    # Lot sizing info from runner_score_to_lots audit trail
    final_lots = snapshot.get("final_lots")
    spread_gated = bool(snapshot.get("spread_gated", False))
    tier17_floor = bool(snapshot.get("tier17_floor_applied", False))
    sloppy_floor = bool(snapshot.get("sloppy_shallow_floor_applied", False))
    deep_base_cap = bool(snapshot.get("deep_tier_base_cap_applied", False))
    weak_zone_cap = bool(snapshot.get("weak_m5_zone_cap_applied", False))
    directional_cap = bool(snapshot.get("directional_cap_applied", False))

    current_str = f"{bucket} ({points}pt)"
    if fresh:
        current_str += " FRESH"
    if final_lots is not None:
        current_str += f" -> {final_lots:.2f} lots"
    if spread_gated:
        current_str += " [SPREAD GATE]"
    if tier17_floor:
        current_str += " [T17 FLOOR]"
    if sloppy_floor:
        current_str += " [PB FLOOR]"
    if deep_base_cap:
        current_str += " [DEEP BASE CAP]"
    if weak_zone_cap:
        current_str += " [WEAK ZONE CAP]"
    if directional_cap:
        current_str += " [DIR CAP]"
    _floor = snapshot.get("bucket_lots_floor")
    _base = snapshot.get("bucket_lots_base")
    _elev = snapshot.get("bucket_lots_elevated")
    _press = snapshot.get("bucket_lots_press")
    _elite = snapshot.get("bucket_lots_elite")
    if all(v is not None for v in (_floor, _base, _elev, _press, _elite)):
        threshold_str = (
            f"floor {_floor:.2f} | base {_base:.2f} | "
            f"elevated {_elev:.2f} | press {_press:.2f} | elite {_elite:.2f}"
        )
    else:
        threshold_str = "floor 0.03 | base 0.05 | elevated 0.07 | press 0.15 | elite 0.30"
    explanation = (
        f"ATR stop: {atr_pips:.1f}p ({'eligible' if atr_eligible else 'FLOOR — too wide'}). "
        f"Features: {feature_str}. Freshness mode: {freshness_mode}."
    )
    if bars_cross is not None or prior_ent is not None:
        bc_str = str(bars_cross) if bars_cross is not None else "?"
        pe_str = str(prior_ent) if prior_ent is not None else "?"
        explanation += f" Freshness: {bc_str} bars since cross, {pe_str} prior entries."

    return FilterReport(
        filter_id="runner_score", display_name="Runner Score",
        enabled=True, is_clear=True,
        current_value=current_str,
        threshold=threshold_str,
        explanation=explanation,
        metadata=dict(snapshot),
    )


def report_open_exposure(total_lots: float, buy_lots: float, sell_lots: float, directional_cap_lots: float | None = None) -> FilterReport:
    is_clear = True
    threshold = f"Buy {buy_lots:.2f} | Sell {sell_lots:.2f}"
    if directional_cap_lots is not None:
        cap = float(directional_cap_lots)
        threshold = f"Buy {buy_lots:.2f} | Sell {sell_lots:.2f} | Cap {cap:.2f}/side"
        is_clear = max(float(buy_lots), float(sell_lots)) < cap
    return FilterReport(
        filter_id="open_exposure",
        display_name="Open Exposure",
        enabled=True,
        is_clear=is_clear,
        current_value=f"{total_lots:.2f} lots",
        threshold=threshold,
        explanation="Total live directional exposure for operator visibility.",
        metadata={
            "total_open_lots": round(float(total_lots), 4),
            "buy_open_lots": round(float(buy_lots), 4),
            "sell_open_lots": round(float(sell_lots), 4),
            "directional_cap_lots": round(float(directional_cap_lots), 4) if directional_cap_lots is not None else None,
        },
    )


def collect_trial_9_context(
    policy,
    data_by_tf: dict,
    tick,
    tier_state: dict,
    eval_result: Optional[dict],
    pip_size: float,
    *,
    exhaustion_result: Optional[dict] = None,
    ntz_snapshot: Optional[dict] = None,
    intraday_fib_snapshot: Optional[dict] = None,
    conviction_snapshot: Optional[dict] = None,
) -> list[ContextItem]:
    """Collect Trial #9 dashboard context, including NTZ/Fibonacci/Conviction details."""
    items = collect_trial_7_context(
        policy,
        data_by_tf,
        tick,
        tier_state,
        eval_result,
        pip_size,
        exhaustion_result=exhaustion_result,
    )

    ntz_enabled = bool(getattr(policy, "ntz_enabled", False))
    fib_enabled = bool(getattr(policy, "ntz_use_fib_pivots", False))
    buffer_pips = float(getattr(policy, "ntz_buffer_pips", 10.0))
    current_price = (tick.bid + tick.ask) / 2.0

    items.append(ContextItem("NTZ", "ON" if ntz_enabled else "OFF", "filters"))
    if ntz_enabled:
        items.append(ContextItem("NTZ Buffer", f"{buffer_pips:.1f}p", "filters"))
    items.append(ContextItem("Fib Pivot NTZ", "ON" if fib_enabled else "OFF", "filters"))

    levels = dict((ntz_snapshot or {}).get("levels") or {})
    fib_levels = dict((ntz_snapshot or {}).get("fib_levels") or {})
    if fib_enabled:
        active_labels = ", ".join(fib_levels.keys()) if fib_levels else "Awaiting daily H/L/C"
        items.append(ContextItem("Fib Levels Active", active_labels, "filters"))

        nearest_label: str | None = None
        nearest_dist_pips: float | None = None
        for label, val in fib_levels.items():
            try:
                if val is None:
                    continue
                level = float(val)
                dist_pips = abs(current_price - level) / pip_size
                items.append(ContextItem(label, f"{level:.3f}", "filters"))
                if nearest_dist_pips is None or dist_pips < nearest_dist_pips:
                    nearest_dist_pips = dist_pips
                    nearest_label = label
            except Exception:
                continue
        if nearest_label is not None and nearest_dist_pips is not None:
            items.append(ContextItem("Nearest Fib Level", f"{nearest_label} ({nearest_dist_pips:.1f}p)", "filters"))

    if ntz_enabled and levels:
        blocked_label: str | None = None
        blocked_dist_pips: float | None = None
        buffer = buffer_pips * pip_size
        for label, val in levels.items():
            try:
                if val is None:
                    continue
                dist = abs(current_price - float(val))
                if dist <= buffer and (blocked_dist_pips is None or dist / pip_size < blocked_dist_pips):
                    blocked_dist_pips = dist / pip_size
                    blocked_label = label
            except Exception:
                continue
        if blocked_label is not None and blocked_dist_pips is not None:
            items.append(ContextItem("NTZ Blocking", f"{blocked_label} ({blocked_dist_pips:.1f}p)", "filters"))
        else:
            items.append(ContextItem("NTZ Blocking", "clear", "filters"))

    # Intraday Fibonacci Corridor context
    ifib = intraday_fib_snapshot or {}
    ifib_enabled = bool(ifib.get("enabled", False))
    items.append(ContextItem("Intraday Fib Corridor", "ON" if ifib_enabled else "OFF", "filters"))
    if ifib_enabled:
        tf = str(ifib.get("timeframe") or getattr(policy, "intraday_fib_timeframe", "M15"))
        lookback = int(ifib.get("lookback_bars") or getattr(policy, "intraday_fib_lookback_bars", 16))
        calc_mode = str(ifib.get("calculation_mode") or "rolling_window")
        items.append(ContextItem("IFC Calc", "Previous Candle Pivots" if calc_mode == "previous_candle" else "Rolling Window", "filters"))
        items.append(ContextItem("IFC Timeframe", tf, "filters"))
        if calc_mode != "previous_candle":
            items.append(ContextItem("IFC Lookback", str(lookback), "filters"))
        lower_level = ifib.get("lower_level", "S1")
        upper_level = ifib.get("upper_level", "R1")
        lower_val = ifib.get("lower_value")
        upper_val = ifib.get("upper_value")
        items.append(ContextItem("IFC Bounds", f"{lower_level} .. {upper_level}", "filters"))
        if lower_val is not None and upper_val is not None:
            items.append(ContextItem(f"IFC {lower_level}", f"{lower_val:.3f}", "filters"))
            items.append(ContextItem(f"IFC {upper_level}", f"{upper_val:.3f}", "filters"))
        r_high = ifib.get("rolling_high")
        r_low = ifib.get("rolling_low")
        src_close = ifib.get("source_close")
        if r_high is not None and r_low is not None:
            label = "IFC Source H/L" if calc_mode == "previous_candle" else "IFC Range"
            items.append(ContextItem(label, f"{r_low:.3f} - {r_high:.3f}", "filters"))
        if src_close is not None and calc_mode == "previous_candle":
            items.append(ContextItem("IFC Source Close", f"{float(src_close):.3f}", "filters"))
        corridor_state = ifib.get("corridor_state")
        if corridor_state is True:
            items.append(ContextItem("IFC Status", "INSIDE — entries allowed", "filters"))
        elif corridor_state is False:
            items.append(ContextItem("IFC Status", "OUTSIDE — entries blocked", "filters"))
        else:
            items.append(ContextItem("IFC Status", "Awaiting data", "filters"))

    # Conviction sizing is a Trial 9 context item. Trial 10 uses runner-score sizing instead.
    if not (
        getattr(policy, "type", None) == "kt_cg_trial_10"
        and bool(getattr(policy, "runner_score_sizing_enabled", False))
    ):
        conv = conviction_snapshot or {}
        conv_enabled = bool(conv.get("enabled", False))
        items.append(ContextItem("Conviction Sizing", "ON" if conv_enabled else "OFF", "sizing"))
        if conv_enabled:
            m5b = conv.get("m5_bucket", "normal")
            m1b = conv.get("m1_bucket", "neutral")
            mult = conv.get("multiplier", 1.0)
            lots = conv.get("conviction_lots", 0.0)
            base = conv.get("base_lots", 0.0)
            items.append(ContextItem("M5 Regime", m5b.upper(), "sizing"))
            items.append(ContextItem("M1 Health", m1b.upper(), "sizing"))
            items.append(ContextItem("Multiplier", f"{mult:.2f}x", "sizing"))
            items.append(ContextItem("Lots", f"{lots:.2f} (base {base:.2f})", "sizing"))
            m5_spread = conv.get("m5_spread_pips", 0.0)
            m5_slope = conv.get("m5_slope_pips_per_bar", 0.0)
            m1_spread = conv.get("m1_spread_pips", 0.0)
            items.append(ContextItem("M5 EMA Spread", f"{m5_spread:.2f}p", "sizing"))
            items.append(ContextItem("M5 EMA9 Slope", f"{m5_slope:.2f}p/bar", "sizing"))
            items.append(ContextItem("M1 EMA Spread", f"{m1_spread:.2f}p", "sizing"))
            if conv.get("m1_compressing"):
                items.append(ContextItem("M1 Compressing", f"YES ({conv.get('m1_compression_bars_count', 0)} bars)", "sizing"))

    gate_data = dict((eval_result or {}).get("trial10_entry_gates") or {})
    if gate_data:
        zone = dict(gate_data.get("zone") or {})
        tier = dict(gate_data.get("tier") or {})
        items.append(ContextItem("T10 Zone Gate", str(zone.get("message") or "Awaiting alignment"), "filters"))
        items.append(ContextItem("T10 Tier Gate", str(tier.get("message") or "Awaiting tier touch"), "filters"))
    quality = dict((eval_result or {}).get("trial10_pullback_quality") or {})
    if quality:
        items.append(ContextItem("PB Quality", str(quality.get("label", "neutral")).upper(), "filters"))
        items.append(ContextItem("PB Bars", str(int(quality.get("pullback_bar_count") or 0)), "filters"))
        items.append(ContextItem("PB Ratio", f"{float(quality.get('structure_ratio') or 0.0):.2f}", "filters"))

    return items


def collect_trial_2_context(
    policy, data_by_tf: dict, tick, pip_size: float,
) -> list[ContextItem]:
    """Collect context items for Trial #2."""
    items: list[ContextItem] = []

    # M5 Trend
    m5_df = data_by_tf.get("M5")
    if m5_df is not None and not m5_df.empty:
        from core.indicators import ema as ema_fn
        m5_close = m5_df["close"]
        fast_p = getattr(policy, "m5_trend_ema_fast", 9)
        slow_p = getattr(policy, "m5_trend_ema_slow", 21)
        if len(m5_df) > slow_p:
            fast_v = float(ema_fn(m5_close, fast_p).iloc[-1])
            slow_v = float(ema_fn(m5_close, slow_p).iloc[-1])
            trend = "BULL" if fast_v > slow_v else "BEAR"
            items.append(ContextItem("M5 Trend", trend, "trend"))
            items.append(ContextItem(f"M5 EMA{fast_p}", f"{fast_v:.3f}", "trend"))
            items.append(ContextItem(f"M5 EMA{slow_p}", f"{slow_v:.3f}", "trend"))

    # M1 Zone Entry EMAs
    m1_df = data_by_tf.get("M1")
    if m1_df is not None and not m1_df.empty:
        from core.indicators import ema as ema_fn
        m1_close = m1_df["close"]
        ze_fast = getattr(policy, "m1_zone_entry_ema_fast", 9)
        ze_slow = getattr(policy, "m1_zone_entry_ema_slow", 21)
        if len(m1_df) > max(ze_fast, ze_slow):
            zf = float(ema_fn(m1_close, ze_fast).iloc[-1])
            zs = float(ema_fn(m1_close, ze_slow).iloc[-1])
            items.append(ContextItem(f"M1 EMA{ze_fast}", f"{zf:.3f}", "zone_entry"))
            items.append(ContextItem(f"M1 EMA{ze_slow}", f"{zs:.3f}", "zone_entry"))

    items.append(ContextItem("Bid", f"{tick.bid:.3f}", "price"))
    items.append(ContextItem("Ask", f"{tick.ask:.3f}", "price"))
    items.append(ContextItem("Spread", f"{(tick.ask - tick.bid) / pip_size:.1f}p", "price"))

    return items


def collect_trial_3_context(
    policy, data_by_tf: dict, tick, pip_size: float,
) -> list[ContextItem]:
    """Collect context items for Trial #3."""
    # Trial #3 is similar to Trial #2 but with different EMA periods
    return collect_trial_2_context(policy, data_by_tf, tick, pip_size)


def collect_uncle_parsh_context(
    policy, data_by_tf: dict, tick, pip_size: float,
    eval_result: Optional[dict] = None,
) -> list[ContextItem]:
    """Collect context items for Uncle Parsh H1 Breakout dashboard display."""
    items: list[ContextItem] = []

    # M5 Trend (EMA 9/21)
    m5_df = data_by_tf.get("M5")
    if m5_df is not None and not m5_df.empty:
        from core.indicators import ema as ema_fn
        m5_close = m5_df["close"]
        fast_p = getattr(policy, "m5_trend_ema_fast", 9)
        slow_p = getattr(policy, "m5_trend_ema_slow", 21)
        if len(m5_df) > slow_p:
            fast_v = float(ema_fn(m5_close, fast_p).iloc[-1])
            slow_v = float(ema_fn(m5_close, slow_p).iloc[-1])
            trend = "BULL" if fast_v > slow_v else "BEAR"
            items.append(ContextItem("M5 Trend", trend, "trend"))
            items.append(ContextItem(f"M5 EMA{fast_p}", f"{fast_v:.3f}", "trend"))
            items.append(ContextItem(f"M5 EMA{slow_p}", f"{slow_v:.3f}", "trend"))

    # M1 Entry/Exit EMAs (9/21 by default)
    m1_df = data_by_tf.get("M1")
    if m1_df is not None and not m1_df.empty:
        from core.indicators import ema as ema_fn
        m1_close = m1_df["close"]
        ema_fast_p = int(getattr(policy, "m1_entry_ema_fast", 9))
        ema_slow_p = int(getattr(policy, "m1_entry_ema_slow", 21))
        for period, label in [(ema_fast_p, f"M1 EMA{ema_fast_p}"), (ema_slow_p, f"M1 EMA{ema_slow_p}")]:
            if len(m1_df) > period:
                val = float(ema_fn(m1_close, period).iloc[-1])
                items.append(ContextItem(label, f"{val:.3f}", "entry"))

    # H1 level mode (major only vs major + swing + cluster)
    major_only = getattr(policy, "major_extremes_only", False)
    level_mode = "Major only" if major_only else "Major + Swing + Cluster"
    items.append(ContextItem("H1 Level Mode", level_mode, "levels"))

    # H1 levels summary + entry window from eval result
    if eval_result and isinstance(eval_result.get("level_updates"), list):
        level_count = len(eval_result["level_updates"])
        items.append(ContextItem("H1 Levels", str(level_count), "levels"))
        # Show remaining entry window if any level is ready
        try:
            entry_window_m = float(getattr(policy, "entry_window_minutes", 15))
            now_utc = None
            if m1_df is not None and not m1_df.empty and "time" in m1_df.columns:
                now_utc = pd.to_datetime(m1_df["time"].iloc[-1], utc=True, errors="coerce")
            if now_utc is None or pd.isna(now_utc):
                now_utc = pd.Timestamp.now(tz="UTC")
            remaining_min = None
            for lv in eval_result["level_updates"]:
                if isinstance(lv, dict) and lv.get("state") == "ready" and lv.get("break_time"):
                    bt = pd.to_datetime(lv.get("break_time"), utc=True, errors="coerce")
                    if bt is None or pd.isna(bt):
                        continue
                    expiry = bt + pd.Timedelta(minutes=entry_window_m)
                    rem = (expiry - now_utc).total_seconds() / 60.0
                    remaining_min = rem if remaining_min is None else max(remaining_min, rem)
            if remaining_min is not None:
                items.append(ContextItem("Entry Window", f"{max(0.0, remaining_min):.1f}m", "levels"))
        except Exception:
            pass

    # Price / spread
    items.append(ContextItem("Bid", f"{tick.bid:.3f}", "price"))
    items.append(ContextItem("Ask", f"{tick.ask:.3f}", "price"))
    items.append(ContextItem("Spread", f"{(tick.ask - tick.bid) / pip_size:.1f}p", "price"))

    return items




def _is_defended_phase3_preset_name(active_preset_name: str | None) -> bool:
    normalized = str(active_preset_name or "").strip().lower()
    return normalized == FROZEN_PHASE3_DEFENDED_PRESET_ID or normalized.startswith(f"{FROZEN_PHASE3_DEFENDED_PRESET_ID} ")


def _phase3_defended_context_items(active_preset_name: str | None, cfg: dict[str, Any]) -> list[ContextItem]:
    if not _is_defended_phase3_preset_name(active_preset_name):
        return []
    ldn_cfg = cfg.get("london_v2", {}) if isinstance(cfg.get("london_v2"), dict) else {}
    v44_cfg = cfg.get("v44_ny", {}) if isinstance(cfg.get("v44_ny"), dict) else {}
    v14_cfg = cfg.get("v14", {}) if isinstance(cfg.get("v14"), dict) else {}
    blocked_days = ldn_cfg.get("d_suppress_weekdays") or []
    veto_cells = v44_cfg.get("defensive_veto_cells") or []
    cell_scale_overrides = v14_cfg.get("cell_scale_overrides") or {}
    t3_scale = cell_scale_overrides.get("ambiguous/er_mid/der_pos:sell")
    items = [ContextItem("Frozen Package", "Phase 3 Frozen V7 Defended", "frozen")]
    if blocked_days:
        items.append(ContextItem("L1 Weekdays", ", ".join(str(d) for d in blocked_days) + " blocked", "frozen"))
    items.append(ContextItem(
        "L1 Exit",
        f"TP1 {float(ldn_cfg.get('d_tp1_r', 0.0)):.2f}R / BE {float(ldn_cfg.get('d_be_offset_pips', 0.0)):.1f} / TP2 {float(ldn_cfg.get('d_tp2_r', 0.0)):.1f}R",
        "frozen",
    ))
    if veto_cells:
        items.append(ContextItem("Defensive Veto", f"v44_ny blocked in {', '.join(str(c) for c in veto_cells)}", "frozen"))
    if t3_scale is not None:
        items.append(ContextItem("T3 Scale", f"ambiguous/er_mid/der_pos:sell = {float(t3_scale):.2f}x", "frozen"))
    return items

def collect_phase3_context(
    policy: Any,
    data_by_tf: dict,
    tick: Any,
    eval_result: Optional[dict],
    phase3_state: dict,
    pip_size: float,
    active_preset_name: Optional[str] = None,
) -> list[ContextItem]:
    """Collect context items for Phase 3 Integrated dashboard; varies by active session."""
    from datetime import datetime, timezone
    from core.phase3_integrated_engine import (
        classify_session,
        _compute_session_windows,
        compute_bb_width_regime,
        _compute_adx,
        _compute_atr,
        compute_asian_range,
        compute_v44_h1_trend,
        compute_v44_m5_slope,
        compute_v44_atr_pct_filter,
        classify_v44_strength,
        ADX_PERIOD,
        ATR_PERIOD,
        LONDON_START_UTC,
    )

    items: list[ContextItem] = []
    now_utc = datetime.now(timezone.utc)
    # Load sizing config for session classification (matches engine's classify_session call)
    try:
        from core.phase3_integrated_engine import load_phase3_sizing_config as _lp3cfg
        _p3_sizing = _lp3cfg() or {}
    except Exception:
        _p3_sizing = {}
    session = classify_session(now_utc, _p3_sizing)
    items.extend(_phase3_defended_context_items(active_preset_name, _p3_sizing))

    # Runtime-health and current-eval context shared across all sessions.
    _cfg_hash = str(phase3_state.get("effective_phase3_config_hash") or "")
    _cfg_date = str(phase3_state.get("effective_phase3_config_date") or "")
    if _cfg_hash:
        items.append(ContextItem("Config Hash", _cfg_hash[:12], "runtime"))
    if _cfg_date:
        items.append(ContextItem("Config Date", _cfg_date, "runtime"))
    _arrival_lag = phase3_state.get("last_m1_arrival_lag_sec")
    if _arrival_lag is not None:
        try:
            items.append(ContextItem("M1 Arrival Lag", f"{float(_arrival_lag):.1f}s", "runtime"))
        except Exception:
            pass
    _retry_count = phase3_state.get("last_m1_retry_count")
    if _retry_count is not None:
        try:
            items.append(ContextItem("M1 Retry Count", str(int(_retry_count)), "runtime"))
        except Exception:
            pass

    _fallback_eval = phase3_state.get("last_phase3_eval", {}) if isinstance(phase3_state.get("last_phase3_eval"), dict) else {}
    _audit = eval_result.get("phase3_ownership_audit") if isinstance(eval_result, dict) else None
    _strategy_tag = str((eval_result or {}).get("strategy_tag") or "") if isinstance(eval_result, dict) else ""
    if not _strategy_tag:
        _strategy_tag = str(_fallback_eval.get("strategy_tag") or "")
    if _strategy_tag:
        items.append(ContextItem("Strategy Tag", _strategy_tag, "decision"))
    _audit_cell = ""
    _audit_regime = ""
    _flags: list[str] = []
    if isinstance(_audit, dict):
        _audit_cell = str(_audit.get("ownership_cell") or "")
        _audit_regime = str(_audit.get("regime_label") or "")
        if _audit.get("defensive_global_standdown"):
            _flags.append("standdown")
        if _audit.get("defensive_london_cluster_block"):
            _flags.append("ldn_cluster")
        if _audit.get("defensive_v44_regime_block"):
            _flags.append("v44_regime")
    if not _audit_cell:
        _audit_cell = str(_fallback_eval.get("ownership_cell") or "")
    if not _audit_regime:
        _audit_regime = str(_fallback_eval.get("regime_label") or "")
    if not _flags:
        _flags = [str(flag).strip() for flag in list(_fallback_eval.get("defensive_flags") or []) if str(flag).strip()]
    if _audit_cell:
        items.append(ContextItem("Ownership Cell", _audit_cell, "decision"))
    if _audit_regime:
        items.append(ContextItem("Regime Label", _audit_regime, "decision"))
    if _flags:
        items.append(ContextItem("Defensive Flags", ", ".join(_flags), "decision"))
    _decision_reason = ""
    dec = eval_result.get("decision") if eval_result else None
    if dec is not None and getattr(dec, "reason", None):
        _decision_reason = str(dec.reason)
    if not _decision_reason:
        _decision_reason = str(_fallback_eval.get("reason") or "")
    if _decision_reason:
        items.append(ContextItem("Last decision", _decision_reason, "decision"))

    # Common: time and session
    items.append(ContextItem("UTC", now_utc.strftime("%H:%M"), "session"))
    items.append(ContextItem("Day", now_utc.strftime("%A"), "session"))
    windows = _compute_session_windows(now_utc)
    d_start = windows["london_open"] + pd.Timedelta(minutes=15)
    ny_start_delayed = windows["ny_open"] + pd.Timedelta(minutes=5)
    if session is None:
        items.append(ContextItem("Active Session", "None", "session"))
        items.append(ContextItem(
            "Next",
            f"London {windows['london_open'].strftime('%H:%M')} / NY {ny_start_delayed.strftime('%H:%M')} / Tokyo 16:00 UTC",
            "session",
        ))
        items.append(ContextItem("Bid", f"{tick.bid:.3f}", "price"))
        items.append(ContextItem("Ask", f"{tick.ask:.3f}", "price"))
        items.append(ContextItem("Spread", f"{(tick.ask - tick.bid) / pip_size:.1f}p", "price"))
        return items

    from core.phase3_integrated_engine import (
        ADX_MAX, COOLDOWN_MINUTES, STOP_AFTER_CONSECUTIVE_LOSSES,
        MAX_TRADES_PER_SESSION, V44_SESSION_STOP_LOSSES, V44_MAX_ENTRIES_DAY,
        V44_COOLDOWN_WIN, V44_COOLDOWN_LOSS,
    )

    items.append(ContextItem("Active Session", session, "session"))
    open_count = int(phase3_state.get("open_trade_count", 0))
    items.append(ContextItem("Phase 3 open", str(open_count), "session"))

    if session == "tokyo":
        items.append(ContextItem("Strategy", "V14 Mean Reversion", "session"))
        m5_df = data_by_tf.get("M5")
        if m5_df is not None and not m5_df.empty and len(m5_df) >= 130:
            regime = compute_bb_width_regime(m5_df)
            items.append(ContextItem("Regime", f"{regime} ({'OK' if regime == 'ranging' else 'BLOCK'})", "v14"))
        m15_df = data_by_tf.get("M15")
        if m15_df is not None and not m15_df.empty and len(m15_df) >= ADX_PERIOD + 2:
            adx_val = _compute_adx(m15_df)
            items.append(ContextItem("ADX", f"{adx_val:.1f} ({'OK' if adx_val < ADX_MAX else 'BLOCK'})", "v14"))
            atr_series = _compute_atr(m15_df)
            atr_val = float(atr_series.iloc[-1]) if pd.notna(atr_series.iloc[-1]) else 0.0
            items.append(ContextItem("ATR (pips)", f"{atr_val / pip_size:.1f}", "v14"))
        today_str = now_utc.strftime("%Y-%m-%d")
        sk = f"session_tokyo_{today_str}"
        sd = phase3_state.get(sk, {})
        trade_count = int(sd.get("trade_count", 0))
        consec_losses = int(sd.get("consecutive_losses", 0))
        items.append(ContextItem("Trades", f"{trade_count}/{MAX_TRADES_PER_SESSION}", "v14"))
        items.append(ContextItem("Consec losses", f"{consec_losses}/{STOP_AFTER_CONSECUTIVE_LOSSES} ({'STOP' if consec_losses >= STOP_AFTER_CONSECUTIVE_LOSSES else 'OK'})", "v14"))
        items.append(ContextItem("Wins closed", str(sd.get("wins_closed", 0)), "v14"))
        items.append(ContextItem("Win streak", str(sd.get("win_streak", 0)), "v14"))
        # Cooldown: time since last entry vs threshold
        last_entry = sd.get("last_entry_time")
        if last_entry:
            try:
                from datetime import timezone as _tz
                _lt = pd.Timestamp(last_entry)
                if _lt.tzinfo is None:
                    _lt = _lt.tz_localize("UTC")
                elapsed_min = (pd.Timestamp(now_utc) - _lt).total_seconds() / 60.0
                cd_ok = elapsed_min >= COOLDOWN_MINUTES
                items.append(ContextItem("Cooldown", f"{elapsed_min:.0f}m ago (min {COOLDOWN_MINUTES}m) — {'OK' if cd_ok else 'WAIT'}", "v14"))
            except Exception:
                pass
        # Breakout block
        if sd.get("breakout_blocked"):
            reason = sd.get("breakout_blocked_reason", "active")
            items.append(ContextItem("Breakout block", f"ACTIVE ({reason})", "v14"))
        else:
            items.append(ContextItem("Breakout block", "clear", "v14"))
        # Signal Strength Tracking (SST)
        try:
            from core.phase3_integrated_engine import load_phase3_sizing_config as _lpc
            _v14_cfg = (_lpc() or {}).get("v14", {})
            _sst_cfg = _v14_cfg.get("signal_strength_tracking", {}) if isinstance(_v14_cfg.get("signal_strength_tracking"), dict) else {}
            _sst_enabled = bool(_sst_cfg.get("enabled", False))
            if _sst_enabled:
                _sst_filter = bool(_sst_cfg.get("filter_on_it", False))
                _sst_min = int(_sst_cfg.get("min_strength_score", 0))
                # SST info is stored in phase3_state["last_v14_signal_strength"] by the engine,
                # not at the top level of eval_result.
                _last_sst = phase3_state.get("last_v14_signal_strength") or {}
                _sst_score = _last_sst.get("signal_strength_score")
                _sst_bucket = str(_last_sst.get("signal_strength_bucket") or "—")
                _sst_mult = _last_sst.get("signal_strength_mult")
                if _sst_score is not None:
                    _sst_str = f"{_sst_score} ({_sst_bucket}) ×{_sst_mult:.2f}"
                    if _sst_filter and _sst_min > 0 and _sst_score < _sst_min:
                        _sst_str += f" BLOCK (<{_sst_min})"
                    elif _sst_filter and _sst_min > 0:
                        _sst_str += f" OK (≥{_sst_min})"
                    items.append(ContextItem("SST Score", _sst_str, "v14"))
                else:
                    items.append(ContextItem("SST", "enabled — no eval yet", "v14"))
            else:
                items.append(ContextItem("SST", "disabled", "v14"))
        except Exception:
            pass
        # No-reentry timer (after stopout): show if either direction is blocked
        try:
            from core.phase3_integrated_engine import load_phase3_sizing_config
            _v14_cfg = (load_phase3_sizing_config(preset_id=active_preset_name) or {}).get("v14", {})
            _no_reentry_min = int(_v14_cfg.get("no_reentry_same_direction_after_stop_minutes", 0))
            if _no_reentry_min > 0:
                for _dir in ("buy", "sell"):
                    _stop_key = f"last_stopout_time_{_dir}"
                    _last_stop = sd.get(_stop_key)
                    if _last_stop:
                        try:
                            _st = pd.Timestamp(_last_stop)
                            if _st.tzinfo is None:
                                _st = _st.tz_localize("UTC")
                            _elapsed = (pd.Timestamp(now_utc) - _st).total_seconds() / 60.0
                            if _elapsed < _no_reentry_min:
                                _remaining = _no_reentry_min - _elapsed
                                items.append(ContextItem(f"No-reentry {_dir}", f"BLOCK {_remaining:.0f}m left (after SL)", "v14"))
                            else:
                                items.append(ContextItem(f"No-reentry {_dir}", "clear", "v14"))
                        except Exception:
                            pass
        except Exception:
            pass

    elif session == "london":
        items.append(ContextItem("Strategy", "London V2 (A + D)", "session"))
        if now_utc < windows["london_arb_end"]:
            items.append(ContextItem(
                "Window",
                f"Setup A ({windows['london_open'].strftime('%H:%M')}–{windows['london_arb_end'].strftime('%H:%M')} UTC)",
                "london",
            ))
        else:
            items.append(ContextItem(
                "Window",
                f"Setup D ({d_start.strftime('%H:%M')}–{(windows['london_open'] + pd.Timedelta(minutes=120)).strftime('%H:%M')} UTC active)",
                "london",
            ))
        m1_df = data_by_tf.get("M1")
        if m1_df is not None and not m1_df.empty:
            a_hi, a_lo, a_pips, a_ok = compute_asian_range(m1_df, int(windows["london_open"].hour))
            items.append(ContextItem("Asian range (pips)", f"{a_pips:.1f} ({'OK' if a_ok else 'BLOCK'})", "london"))
            items.append(ContextItem("Asian H/L", f"{a_hi:.3f} / {a_lo:.3f}", "london"))
        today = now_utc.date().isoformat()
        sk = f"session_london_{today}"
        sd = phase3_state.get(sk, {})
        items.append(ContextItem("ARB trades", str(sd.get("arb_trades", 0)), "london"))
        items.append(ContextItem("D trades", str(sd.get("d_trades", 0)), "london"))
        items.append(ContextItem("Total trades", str(sd.get("total_trades", 0)), "london"))
        items.append(ContextItem("Consec losses", str(sd.get("consecutive_losses", 0)), "london"))
        items.append(ContextItem("Wins closed", str(sd.get("wins_closed", 0)), "london"))
        # Channel states: ARMED = ready, FIRED = used, WAITING_RESET = needs retouch
        channels = sd.get("channels", {})
        for ch_key, ch_label in [("A_long", "A↑"), ("A_short", "A↓"), ("D_long", "D↑"), ("D_short", "D↓")]:
            state = channels.get(ch_key, "ARMED")
            items.append(ContextItem(f"Ch {ch_label}", state, "london"))

    elif session == "ny":
        items.append(ContextItem("Strategy", "V44 NY Momentum", "session"))
        h1_df = data_by_tf.get("H1")
        trend = compute_v44_h1_trend(h1_df) if h1_df is not None and not h1_df.empty else None
        items.append(ContextItem("H1 Trend", trend or "—", "v44"))
        m5_df = data_by_tf.get("M5")
        if m5_df is not None and not m5_df.empty:
            slope = compute_v44_m5_slope(m5_df, 4)
            items.append(ContextItem("M5 Slope (p/bar)", f"{slope:.2f}", "v44"))
            atr_ok = compute_v44_atr_pct_filter(m5_df)
            items.append(ContextItem("ATR % filter", "pass" if atr_ok else "block", "v44"))
            # Signal strength from H1 trend + M5 slope (backtest parity: ny_strength_allow filter)
            _v44_strength = classify_v44_strength(slope, is_london=False)
            try:
                from core.phase3_integrated_engine import load_phase3_sizing_config
                _v44_cfg2 = (load_phase3_sizing_config(preset_id=active_preset_name) or {}).get("v44_ny", {})
                _strength_allow = str(_v44_cfg2.get("ny_strength_allow", "strong_normal")).lower()
                _allow_map = {
                    "strong_only": {"strong"},
                    "strong_normal": {"strong", "normal"},
                    "all": {"strong", "normal", "weak"},
                }
                _allowed = _allow_map.get(_strength_allow, {"strong", "normal"})
                _strength_ok = _v44_strength in _allowed
                items.append(ContextItem("Signal strength", f"{_v44_strength} ({'OK' if _strength_ok else 'BLOCK'})", "v44"))
            except Exception:
                items.append(ContextItem("Signal strength", _v44_strength, "v44"))
        # H4 ADX gate (backtest parity: v5_h4_adx_min)
        h4_df = data_by_tf.get("H4")
        if h4_df is not None and not h4_df.empty and len(h4_df) >= 16:
            try:
                from core.phase3_integrated_engine import load_phase3_sizing_config
                _v44_cfg3 = (load_phase3_sizing_config(preset_id=active_preset_name) or {}).get("v44_ny", {})
                _h4_adx_min = float(_v44_cfg3.get("h4_adx_min", 0.0))
                if _h4_adx_min > 0:
                    _h4_adx = _compute_adx(h4_df)
                    _h4_ok = _h4_adx >= _h4_adx_min
                    items.append(ContextItem("H4 ADX", f"{_h4_adx:.1f} ({'OK' if _h4_ok else 'BLOCK'} ≥{_h4_adx_min:.0f})", "v44"))
            except Exception:
                pass
        today = now_utc.date().isoformat()
        sk = f"session_ny_{today}"
        sd = phase3_state.get(sk, {})
        session_mode = str(sd.get("session_mode") or "—")
        eff_avg = sd.get("session_efficiency_avg")
        eff_hist_len = int(sd.get("session_efficiency_hist_len", 0))
        eff_min = sd.get("trend_mode_efficiency_min")
        eff_max = sd.get("range_mode_efficiency_max")
        range_fade_enabled = bool(sd.get("range_fade_enabled", False))
        if eff_avg is not None:
            eff_detail = f"avg {float(eff_avg):.3f} / n={eff_hist_len}"
            if eff_min is not None and eff_max is not None:
                eff_detail += f" / trend>={float(eff_min):.2f} range<={float(eff_max):.2f}"
            items.append(ContextItem("Session mode", f"{session_mode} ({'range fade on' if range_fade_enabled else 'range fade off'})", "v44"))
            items.append(ContextItem("Efficiency avg", eff_detail, "v44"))
        else:
            items.append(ContextItem("Session mode", session_mode, "v44"))
            items.append(ContextItem("Efficiency avg", f"n={eff_hist_len}", "v44"))
        news_filter_enabled = bool(sd.get("news_filter_enabled", False))
        news_trend_enabled = bool(sd.get("news_trend_enabled", False))
        news_status = str(sd.get("news_status") or ("disabled" if not news_filter_enabled else "clear"))
        news_event_time = sd.get("news_event_time")
        news_wait_minutes = sd.get("news_wait_minutes")
        news_confirm_progress = sd.get("news_confirm_progress")
        news_trend_side = sd.get("news_trend_side")
        news_mode = "disabled"
        if news_filter_enabled and news_trend_enabled:
            news_mode = "block + trend"
        elif news_filter_enabled:
            news_mode = "block only"
        items.append(ContextItem("News mode", news_mode, "v44"))
        news_status_detail = news_status
        if news_trend_side:
            news_status_detail += f" ({news_trend_side})"
        items.append(ContextItem("News status", news_status_detail, "v44"))
        if news_wait_minutes is not None:
            items.append(ContextItem("News wait", f"{float(news_wait_minutes):.1f}m", "v44"))
        if news_confirm_progress:
            items.append(ContextItem("News confirm", str(news_confirm_progress), "v44"))
        if news_event_time:
            try:
                ev_ts = pd.Timestamp(news_event_time)
                if ev_ts.tzinfo is None:
                    ev_ts = ev_ts.tz_localize("UTC")
                else:
                    ev_ts = ev_ts.tz_convert("UTC")
                items.append(ContextItem("News event UTC", ev_ts.strftime("%Y-%m-%d %H:%M"), "v44"))
            except Exception:
                items.append(ContextItem("News event UTC", str(news_event_time), "v44"))
        trade_count = int(sd.get("trade_count", 0))
        consec_losses = int(sd.get("consecutive_losses", 0))
        wins_closed = int(sd.get("wins_closed", 0))
        win_streak = int(sd.get("win_streak", 0))
        try:
            from core.phase3_integrated_engine import load_phase3_sizing_config
            _v44_cfg_live = (load_phase3_sizing_config(preset_id=active_preset_name) or {}).get("v44_ny", {})
        except Exception:
            _v44_cfg_live = {}
        _trade_cap = int(_v44_cfg_live.get("max_entries_per_day", V44_MAX_ENTRIES_DAY))
        _stop_limit = int(_v44_cfg_live.get("session_stop_losses", V44_SESSION_STOP_LOSSES))
        _trade_cap_label = str(_trade_cap) if _trade_cap > 0 else "unlimited"
        items.append(ContextItem("Trades", f"{trade_count}/{_trade_cap_label}", "v44"))
        items.append(ContextItem("Consec losses", f"{consec_losses}/{_stop_limit} ({'STOP' if consec_losses >= _stop_limit else 'OK'})", "v44"))
        items.append(ContextItem("Wins closed", str(wins_closed), "v44"))
        items.append(ContextItem("Win streak", str(win_streak), "v44"))
        _ny_cell = str(sd.get("ownership_cell") or "")
        if _ny_cell:
            items.append(ContextItem("NY Ownership Cell", _ny_cell, "v44"))
        _news_status = sd.get("news_status")
        if _news_status is not None:
            items.append(ContextItem("News Status", str(_news_status), "v44"))
        _news_wait = sd.get("news_wait_minutes")
        if _news_wait is not None:
            try:
                items.append(ContextItem("News Wait", f"{float(_news_wait):.1f}m", "v44"))
            except Exception:
                pass
        _news_confirm = sd.get("news_confirm_progress")
        if _news_confirm:
            items.append(ContextItem("News Confirm", str(_news_confirm), "v44"))
        _news_side = sd.get("news_trend_side")
        if _news_side:
            items.append(ContextItem("News Side", str(_news_side), "v44"))
        # GL (Golden Lion) mode
        try:
            from core.phase3_integrated_engine import load_phase3_sizing_config
            _v44_cfg = (load_phase3_sizing_config(preset_id=active_preset_name) or {}).get("v44_ny", {})
            _gl_enabled = bool(_v44_cfg.get("gl_enabled", False))
            _gl_min_wins = max(1, int(_v44_cfg.get("gl_min_wins", 1)))
            _gl_extra = int(_v44_cfg.get("gl_extra_entries", 0))
            if _gl_enabled:
                _gl_active = wins_closed >= _gl_min_wins
                if _gl_active:
                    items.append(ContextItem("GL Mode", f"ACTIVE ({wins_closed}/{_gl_min_wins} wins, +{_gl_extra} entries)", "v44"))
                else:
                    items.append(ContextItem("GL Mode", f"inactive ({wins_closed}/{_gl_min_wins} wins needed)", "v44"))
            else:
                items.append(ContextItem("GL Mode", "disabled", "v44"))
        except Exception:
            pass
        # Cooldown
        cooldown_until = sd.get("cooldown_until")
        if cooldown_until:
            try:
                _cd = pd.Timestamp(cooldown_until)
                if _cd.tzinfo is None:
                    _cd = _cd.tz_localize("UTC")
                remaining = (_cd - pd.Timestamp(now_utc)).total_seconds() / 60.0
                if remaining > 0:
                    items.append(ContextItem("Cooldown", f"WAIT {remaining:.0f}m remaining", "v44"))
                else:
                    items.append(ContextItem("Cooldown", "clear", "v44"))
            except Exception:
                pass
        else:
            items.append(ContextItem("Cooldown", "clear", "v44"))

    # Last decision and price
    items.append(ContextItem("Bid", f"{tick.bid:.3f}", "price"))
    items.append(ContextItem("Ask", f"{tick.ask:.3f}", "price"))
    items.append(ContextItem("Spread", f"{(tick.ask - tick.bid) / pip_size:.1f}p", "price"))
    return items


def collect_trial_6_context(
    policy, data_by_tf: dict, tick, tier_state: dict,
    eval_result: Optional[dict], pip_size: float,
) -> list[ContextItem]:
    """Collect context items for Trial #6 dashboard display."""
    items: list[ContextItem] = []

    # M3 Trend (from eval_result or computed)
    trend_result = eval_result.get("trend_result") if eval_result else None
    if trend_result:
        trend = trend_result.get("trend", "N/A")
        items.append(ContextItem("M3 Trend", trend, "trend"))
        items.append(ContextItem(f"M3 EMA{getattr(policy, 'm3_trend_ema_slow', 9)}", f"{trend_result.get('ema_slow_val', 0):.3f}", "trend"))
        items.append(ContextItem(f"M3 EMA{getattr(policy, 'm3_trend_ema_extra', 21)}", f"{trend_result.get('ema_extra_val', 0):.3f}", "trend"))
    else:
        items.append(ContextItem("M3 Trend", "N/A", "trend"))

    # Bid/Ask/Spread
    items.append(ContextItem("Bid", f"{tick.bid:.3f}", "price"))
    items.append(ContextItem("Ask", f"{tick.ask:.3f}", "price"))
    items.append(ContextItem("Spread", f"{(tick.ask - tick.bid) / pip_size:.1f}p", "price"))

    # EMA Tier state (System A)
    fired = [str(t) for t, v in sorted(tier_state.items()) if v]
    avail = [str(t) for t, v in sorted(tier_state.items()) if not v]
    items.append(ContextItem("EMA Tiers Fired", ", ".join(fired) if fired else "None", "tiers"))
    items.append(ContextItem("EMA Tiers Available", ", ".join(avail) if avail else "None", "tiers"))

    # Dead zone
    daily_reset = eval_result.get("daily_reset_state", {}) if eval_result else {}
    block_active = daily_reset.get("daily_reset_block_active", False)
    utc_hour = datetime.now(timezone.utc).hour
    items.append(ContextItem("Dead Zone", "ACTIVE" if block_active else "Clear", "dead_zone"))
    items.append(ContextItem("UTC Hour", f"{utc_hour:02d}:00", "dead_zone"))

    return items


# ---------------------------------------------------------------------------
# Uncle Parsh H1 Breakout reporters
# ---------------------------------------------------------------------------


def report_h1_levels(policy, data_by_tf, level_state) -> FilterReport:
    """Show detected H1 levels, their status (watching/broken/active), touch counts.
    Sub_filters list one row per level with type and broken status for dropdown.
    """
    major_only = getattr(policy, "major_extremes_only", False)
    level_mode = "Major only (YH/YL, weekly, monthly)" if major_only else "Major + Swing + Cluster"
    if not level_state:
        return FilterReport(
            filter_id="h1_levels", display_name="H1 S/R Levels",
            enabled=True, is_clear=True,
            current_value="No levels detected",
            explanation=f"Level mode: {level_mode}. No H1 levels in range. Check lookback and symbol.",
        )
    watching = sum(1 for lv in level_state if lv.get("state") == "watching")
    catalyst = sum(1 for lv in level_state if lv.get("state") == "catalyst")
    ready = sum(1 for lv in level_state if lv.get("state") == "ready")
    voided = sum(1 for lv in level_state if lv.get("voided", False))
    total = len(level_state)
    has_broken = (ready + catalyst) > 0

    level_details = []
    sub_filters_list: list[FilterReport] = []
    for i, lv in enumerate(level_state):
        price = lv.get("price", 0)
        lt = lv.get("level_type", "?")
        st = lv.get("state", "?")
        tc = lv.get("touch_count", 0)
        level_details.append(f"{price:.3f} ({lt}) [{st}] tc={tc}")
        broken_this = st in ("catalyst", "ready")
        threshold_str = ""
        if broken_this:
            bd = lv.get("break_direction", "")
            if bd:
                threshold_str = f"break: {bd}"
        if lv.get("voided", False):
            sub_expl = "Voided (e.g. by 35 EMA)."
        elif st == "ready":
            sub_expl = "Ready for entry."
        elif st == "catalyst":
            sub_expl = "Broken; waiting for second bar to confirm momentum."
        else:
            sub_expl = "Watching; need M5 close past level."
        sub_filters_list.append(FilterReport(
            filter_id=f"h1_level_{i}",
            display_name=f"{price:.3f} ({lt})",
            enabled=True,
            is_clear=broken_this,
            current_value=f"{st} tc={tc}",
            threshold=threshold_str,
            explanation=sub_expl,
        ))

    if has_broken:
        main_expl = "At least one level broken; ready once M1 entry aligns."
    elif voided and ready == 0 and catalyst == 0:
        main_expl = "Levels voided (e.g. by 35 EMA)."
    else:
        main_expl = "No level broken yet. Need an M5 candle to close past a level."
    main_expl = f"Level mode: {level_mode}. {main_expl}"

    # Count by level type for summary
    type_counts: dict[str, int] = {}
    for lv in level_state:
        lt = lv.get("level_type", "?")
        type_counts[lt] = type_counts.get(lt, 0) + 1
    type_summary = ", ".join(f"{k}: {v}" for k, v in sorted(type_counts.items()))

    return FilterReport(
        filter_id="h1_levels", display_name="H1 S/R Levels",
        enabled=True, is_clear=has_broken,
        current_value=f"{total} levels: {watching}W {catalyst}C {ready}R {voided}V",
        threshold=type_summary or "; ".join(level_details[:5]) if level_details else "none",
        sub_filters=sub_filters_list,
        explanation=main_expl,
    )


def report_m5_trend_alignment(policy, data_by_tf) -> FilterReport:
    """M5 EMA 9 vs 21 alignment status."""
    from core.execution_engine import _resolve_m5_trend_close_series
    from core.indicators import ema as ema_fn

    m5_df = data_by_tf.get("M5")
    if m5_df is None or m5_df.empty:
        return FilterReport(
            filter_id="m5_trend", display_name="M5 Trend (EMA)",
            enabled=True, is_clear=False,
            current_value="No M5 data",
            explanation="No M5 data yet.",
        )
    m5_close, m5_source = _resolve_m5_trend_close_series(policy, data_by_tf)
    fast = ema_fn(m5_close, policy.m5_trend_ema_fast)
    slow = ema_fn(m5_close, policy.m5_trend_ema_slow)
    if fast.empty or slow.empty or pd.isna(fast.iloc[-1]) or pd.isna(slow.iloc[-1]):
        return FilterReport(
            filter_id="m5_trend", display_name="M5 Trend (EMA)",
            enabled=True, is_clear=False,
            current_value="EMA not ready",
            explanation="EMA not ready yet.",
        )
    f_val = float(fast.iloc[-1])
    s_val = float(slow.iloc[-1])
    trend = "BULL" if f_val > s_val else ("BEAR" if f_val < s_val else "FLAT")
    gap = abs(f_val - s_val)
    if trend == "FLAT":
        expl = "Trend not clear. Need EMA 9 clearly above or below EMA 21."
    else:
        expl = f"M5 trend is {trend}. OK for aligned entries."
    return FilterReport(
        filter_id="m5_trend", display_name="M5 Trend (EMA)",
        enabled=True, is_clear=trend != "FLAT",
        current_value=f"{trend} [{m5_source}] (EMA{policy.m5_trend_ema_fast}={f_val:.3f} vs EMA{policy.m5_trend_ema_slow}={s_val:.3f}, gap={gap:.3f})",
        threshold=f"EMA {policy.m5_trend_ema_fast}/{policy.m5_trend_ema_slow}",
        explanation=expl,
    )


def report_m5_power_close_status(policy, data_by_tf, level_state) -> FilterReport:
    """Which levels have had Power Closes, velocity type, waiting for velocity check.
    Sub_filters list one row per breakout for dropdown.
    """
    broken = [lv for lv in (level_state or []) if lv.get("is_broken", False)]
    if not broken:
        return FilterReport(
            filter_id="power_close", display_name="M5 Power Close",
            enabled=True, is_clear=False,
            current_value="No breakouts yet",
            explanation="No breakout yet. Need M5 candle body to close past level.",
        )
    details = []
    sub_filters_list: list[FilterReport] = []
    any_ready = any(lv.get("state") == "ready" for lv in broken)
    any_catalyst = any(lv.get("state") == "catalyst" for lv in broken)
    for i, lv in enumerate(broken):
        price = lv.get("price", 0)
        lt = lv.get("level_type", "?")
        vel = lv.get("velocity_type") or "pending"
        st = lv.get("state", "?")
        bd = lv.get("break_direction", "?")
        details.append(f"{price:.3f}: {bd} vel={vel} [{st}]")
        if st == "ready":
            sub_expl = "Ready for entry."
        else:
            sub_expl = "Broken; waiting for second bar to confirm momentum."
        sub_filters_list.append(FilterReport(
            filter_id=f"power_close_{i}",
            display_name=f"{price:.3f} ({lt})",
            enabled=True,
            is_clear=st in ("catalyst", "ready"),
            current_value=f"{bd} vel={vel} [{st}]",
            threshold="",
            explanation=sub_expl,
        ))
    if any_ready:
        main_expl = "Breakout confirmed; ready for M1 entry."
    else:
        main_expl = "Power close done. Need one more M5 bar to confirm momentum."
    return FilterReport(
        filter_id="power_close", display_name="M5 Power Close",
        enabled=True, is_clear=any(lv.get("state") in ("catalyst", "ready") for lv in broken),
        current_value=f"{len(broken)} breakout(s)",
        threshold="; ".join(details),
        sub_filters=sub_filters_list,
        explanation=main_expl,
    )


def report_m1_entry_status(policy, data_by_tf, tick, pip_size: float = 0.01) -> FilterReport:
    """M1 EMA stack (5/9/21 or 9 vs 21) status, 35 EMA veto status (if enabled)."""
    from core.indicators import ema as ema_fn

    m1_df = data_by_tf.get("M1")
    if m1_df is None or m1_df.empty:
        return FilterReport(
            filter_id="m1_entry", display_name="M1 Entry Status",
            enabled=True, is_clear=False,
            current_value="No M1 data",
            explanation="No M1 data yet.",
        )
    m1_close = m1_df["close"].astype(float)
    ema5 = ema_fn(m1_close, policy.m1_ema_fast)
    ema9 = ema_fn(m1_close, policy.m1_ema_mid)
    ema21 = ema_fn(m1_close, policy.m1_ema_slow)
    ema35 = ema_fn(m1_close, policy.m1_ema_veto)
    if any(s.empty or pd.isna(s.iloc[-1]) for s in [ema5, ema9, ema21, ema35]):
        return FilterReport(
            filter_id="m1_entry", display_name="M1 Entry Status",
            enabled=True, is_clear=False,
            current_value="EMAs not ready",
            explanation="EMAs not ready yet.",
        )
    e5 = float(ema5.iloc[-1])
    e9 = float(ema9.iloc[-1])
    e21 = float(ema21.iloc[-1])
    e35 = float(ema35.iloc[-1])
    last_close = float(m1_close.iloc[-1])
    stack_mode = getattr(policy, "m1_entry_stack_mode", "full")
    veto_enabled = getattr(policy, "m1_ema_veto_enabled", True)
    veto_buffer_pips = float(getattr(policy, "m1_ema_veto_buffer_pips", 2.0))
    veto_buffer = veto_buffer_pips * pip_size
    if stack_mode == "nine_vs_21":
        bull_stack = e9 > e21
        bear_stack = e9 < e21
        stack_str = "BULL (9>21)" if bull_stack else ("BEAR (9<21)" if bear_stack else "MIXED")
    else:
        bull_stack = e5 > e9 > e21
        bear_stack = e5 < e9 < e21
        stack_str = "BULL" if bull_stack else ("BEAR" if bear_stack else "MIXED")
    veto_str = ""
    veto_buy = last_close < e35 - veto_buffer
    veto_sell = last_close > e35 + veto_buffer
    if veto_enabled:
        if veto_buy:
            veto_str = " | 35EMA VETO(buy)"
        elif veto_sell:
            veto_str = " | 35EMA VETO(sell)"
    veto_active = veto_enabled and (veto_buy or veto_sell)
    stack_ok = bull_stack or bear_stack
    entry_ok = stack_ok and not veto_active
    if not stack_ok:
        expl = "M1 EMAs not stacked for entry (need 5>9>21 for buy or 5<9<21 for sell)." if stack_mode == "full" else "M1 9 vs 21 not aligned for entry (need 9>21 for buy or 9<21 for sell)."
    elif veto_active:
        expl = "Price past 35 EMA on wrong side — setup voided."
    else:
        expl = "M1 stack aligned; entry allowed if other filters pass."
    thresh = f"EMA {policy.m1_ema_fast}/{policy.m1_ema_mid}/{policy.m1_ema_slow}, veto={policy.m1_ema_veto}"
    if not veto_enabled:
        thresh += " (veto off)"
    if stack_mode == "nine_vs_21":
        thresh += ", stack: 9vs21"
    return FilterReport(
        filter_id="m1_entry", display_name="M1 Entry Status",
        enabled=True, is_clear=entry_ok,
        current_value=f"Stack: {stack_str} (5={e5:.3f} 9={e9:.3f} 21={e21:.3f}){veto_str}",
        threshold=thresh,
        explanation=expl,
    )


def report_up_spread_veto(policy, tick, pip_size: float) -> FilterReport:
    """Current spread vs max_spread_pips for Uncle Parsh."""
    spread_pips = (tick.ask - tick.bid) / pip_size
    max_spread = float(policy.max_spread_pips)
    ok = spread_pips <= max_spread
    expl = "Spread OK." if ok else f"Spread too high. No entry until spread is {max_spread:.0f}p or less."
    return FilterReport(
        filter_id="up_spread", display_name="Spread Check",
        enabled=True, is_clear=ok,
        current_value=f"{spread_pips:.1f} pips",
        threshold=f"Max: {max_spread} pips",
        block_reason=None if ok else f"Spread {spread_pips:.1f}p > max {max_spread}p",
        explanation=expl,
    )
