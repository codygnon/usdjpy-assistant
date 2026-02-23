"""Dashboard reporter functions.

Each reporter wraps an existing filter function and packages its result
into a FilterReport structure. The dashboard renders whatever reports
it receives — zero hardcoded filter names.
"""
from __future__ import annotations

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
