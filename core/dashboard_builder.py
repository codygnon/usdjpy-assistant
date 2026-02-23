"""Build dashboard filter list the same way the run loop does.

Used by both run_loop (when writing dashboard_state.json) and the API
(when serving /api/data/{profile}/dashboard) so the dashboard always
shows the same filter status the loop uses.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from core.dashboard_models import FilterReport
from core.dashboard_reporters import (
    report_session_filter,
    report_spread,
    report_tiered_atr_trial_4,
    report_daily_hl_filter,
    report_ema_zone_filter,
    report_rolling_danger_zone,
    report_rsi_divergence,
    report_dual_atr_trial_5,
    report_dead_zone,
    report_trend_exhaustion,
    report_max_trades,
    report_t6_dead_zone,
    report_t6_m3_trend,
    report_t6_bb_reversal_cap,
    report_ema_zone_slope_filter_trial_7,
    report_open_trade_cap_by_entry_type,
)

# Attribute names that can be overridden by Apply Temporary Settings (same as execution engine).
# T4/T5: m3_trend_ema_*, m1_zone_entry_ema_*. T3: m5_trend_ema_*, m1_zone_entry_ema_slow, m1_pullback_cross_ema_slow.
_TEMP_OVERRIDE_ATTRS = (
    "m3_trend_ema_fast",
    "m3_trend_ema_slow",
    "m1_zone_entry_ema_fast",
    "m1_zone_entry_ema_slow",
    "m5_trend_ema_fast",
    "m5_trend_ema_slow",
    "m1_zone_entry_ema_slow",
    "m1_pullback_cross_ema_slow",
)


class _EffectivePolicy:
    """Wraps a policy and applies temp_overrides so dashboard uses same EMAs as execution."""

    def __init__(self, policy: Any, overrides: dict[str, int]) -> None:
        self._policy = policy
        self._overrides = overrides

    def __getattr__(self, name: str) -> Any:
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._policy, name)


def effective_policy_for_dashboard(policy: Any, temp_overrides: Optional[dict]) -> Any:
    """Return a policy wrapper that applies temp_overrides (Apply Temporary Settings), or policy unchanged."""
    if policy is None or not temp_overrides:
        return policy
    overrides_clean = {k: int(v) for k, v in temp_overrides.items() if v is not None and k in _TEMP_OVERRIDE_ATTRS}
    if not overrides_clean:
        return policy
    return _EffectivePolicy(policy, overrides_clean)


def _side_from_eval_result(eval_result: Optional[dict]) -> str:
    side = "buy"
    if not eval_result:
        return side
    dec = eval_result.get("decision")
    if dec and getattr(dec, "side", None):
        return dec.side
    if eval_result.get("side"):
        return eval_result["side"]
    return side


def _side_from_m3(data_by_tf: dict, policy: Any) -> str:
    """Derive side (buy/sell) from M3 trend when eval_result is not available."""
    m3_df = data_by_tf.get("M3")
    if m3_df is None or m3_df.empty:
        return "buy"
    try:
        from core.indicators import ema as ema_fn
        fast_p = getattr(policy, "m3_trend_ema_fast", 5)
        slow_p = getattr(policy, "m3_trend_ema_slow", 9)
        if len(m3_df) < slow_p + 1:
            return "buy"
        close = m3_df["close"]
        fast_v = float(ema_fn(close, fast_p).iloc[-1])
        slow_v = float(ema_fn(close, slow_p).iloc[-1])
        return "buy" if fast_v > slow_v else "sell"
    except Exception:
        return "buy"


def _side_from_m5(data_by_tf: dict, policy: Any) -> str:
    """Derive side (buy/sell) from M5 trend when eval_result is not available."""
    m5_df = data_by_tf.get("M5")
    if m5_df is None or m5_df.empty:
        return "buy"
    try:
        from core.indicators import ema as ema_fn
        fast_p = getattr(policy, "m5_trend_ema_fast", 9)
        slow_p = getattr(policy, "m5_trend_ema_slow", 21)
        if len(m5_df) < slow_p + 1:
            return "buy"
        close = m5_df["close"]
        fast_v = float(ema_fn(close, fast_p).iloc[-1])
        slow_v = float(ema_fn(close, slow_p).iloc[-1])
        return "buy" if fast_v > slow_v else "sell"
    except Exception:
        return "buy"


def build_dashboard_filters(
    *,
    profile: Any,
    tick: Any,
    data_by_tf: dict,
    policy: Optional[Any] = None,
    policy_type: str = "",
    eval_result: Optional[dict] = None,
    divergence_state: Optional[dict] = None,
    daily_reset_state: Optional[dict] = None,
    exhaustion_result: Optional[dict] = None,
    store: Optional[Any] = None,
    temp_overrides: Optional[dict[str, int]] = None,
) -> list[FilterReport]:
    """Build the filter report list the same way the run loop does.

    Call with the same profile, tick, data_by_tf, and (when available)
    policy, policy_type, eval_result, and state. If policy/eval_result
    are missing (e.g. API without a recent eval), side is derived from M3 trend.
    When temp_overrides is provided (from Apply Temporary Settings), the
    dashboard uses the same effective EMA periods as the execution engine.
    """
    now_utc = datetime.now(timezone.utc)
    pip_size = float(profile.pip_size)
    filters: list[FilterReport] = []

    # Apply temp overrides so dashboard reflects same EMAs as run loop execution.
    policy = effective_policy_for_dashboard(policy, temp_overrides)

    side = _side_from_eval_result(eval_result) if eval_result else "buy"
    trigger_type = "zone_entry"
    if eval_result:
        trigger_type = eval_result.get("trigger_type") or "zone_entry"
    if policy is not None and not eval_result:
        if policy_type == "kt_cg_trial_7":
            side = _side_from_m5(data_by_tf, policy)
        else:
            side = _side_from_m3(data_by_tf, policy)

    # Session filter (all)
    filters.append(report_session_filter(profile, now_utc))

    # Spread: run loop uses profile.strategy.filters.max_spread_pips; API sometimes has profile.risk
    spread_pips = (tick.ask - tick.bid) / pip_size
    max_spread = None
    if hasattr(profile, "strategy") and hasattr(profile.strategy, "filters"):
        max_spread = getattr(profile.strategy.filters, "max_spread_pips", None)
    if max_spread is None and hasattr(profile, "risk"):
        max_spread = getattr(profile.risk, "max_spread_pips", None)
    filters.append(report_spread(spread_pips, max_spread))

    if policy_type == "kt_cg_trial_4" and policy is not None:
        filters.append(report_tiered_atr_trial_4(policy, data_by_tf.get("M1"), pip_size, trigger_type))
        filters.append(report_daily_hl_filter(policy, data_by_tf, tick, side, pip_size))
        m1_df = data_by_tf.get("M1")
        is_bull = side == "buy"
        if m1_df is not None and not m1_df.empty:
            filters.append(report_ema_zone_filter(policy, m1_df, pip_size, is_bull))
        filters.append(report_rolling_danger_zone(policy, data_by_tf.get("M1"), tick, side, pip_size))
        filters.append(report_rsi_divergence(policy, divergence_state or {}, side))

    elif policy_type == "kt_cg_trial_5" and policy is not None:
        filters.append(report_dual_atr_trial_5(policy, data_by_tf.get("M1"), data_by_tf.get("M3"), pip_size, trigger_type))
        filters.append(report_daily_hl_filter(
            policy, data_by_tf, tick, side, pip_size,
            daily_reset_state.get("daily_reset_high") if daily_reset_state else None,
            daily_reset_state.get("daily_reset_low") if daily_reset_state else None,
            daily_reset_state.get("daily_reset_settled", False) if daily_reset_state else False,
        ))
        m1_df = data_by_tf.get("M1")
        is_bull = side == "buy"
        if m1_df is not None and not m1_df.empty:
            filters.append(report_ema_zone_filter(policy, m1_df, pip_size, is_bull))
        filters.append(report_rolling_danger_zone(policy, data_by_tf.get("M1"), tick, side, pip_size))
        filters.append(report_rsi_divergence(policy, divergence_state or {}, side))
        filters.append(report_dead_zone(daily_reset_state or {}))
        filters.append(report_trend_exhaustion(exhaustion_result))
        max_per_side = getattr(policy, "max_open_trades_per_side", None)
        if max_per_side is not None and store is not None:
            try:
                open_trades = store.list_open_trades(profile.profile_name)
                side_counts: dict[str, int] = {"buy": 0, "sell": 0}
                for t in open_trades:
                    row = dict(t) if hasattr(t, "keys") else t
                    s = str(row.get("side", "")).lower()
                    if s in side_counts:
                        side_counts[s] += 1
                sc = side_counts.get(side, 0)
                filters.append(report_max_trades(sc, max_per_side, side, side_counts))
            except Exception:
                pass

    elif policy_type == "kt_cg_trial_6" and policy is not None:
        filters.append(report_t6_m3_trend(eval_result))
        filters.append(report_t6_dead_zone(policy, daily_reset_state or {}))
        if store is not None:
            filters.append(report_t6_bb_reversal_cap(policy, store, profile.profile_name))
        max_per_side = getattr(policy, "max_open_trades_per_side", None)
        if max_per_side is not None and store is not None:
            try:
                open_trades = store.list_open_trades(profile.profile_name)
                side_counts: dict[str, int] = {"buy": 0, "sell": 0}
                for t in open_trades:
                    row = dict(t) if hasattr(t, "keys") else t
                    s = str(row.get("side", "")).lower()
                    if s in side_counts:
                        side_counts[s] += 1
                sc = side_counts.get(side, 0)
                filters.append(report_max_trades(sc, max_per_side, side, side_counts))
            except Exception:
                pass

    elif policy_type == "kt_cg_trial_7" and policy is not None:
        m1_df = data_by_tf.get("M1")
        if m1_df is not None and not m1_df.empty:
            filters.append(report_ema_zone_slope_filter_trial_7(policy, m1_df, pip_size, side))
        max_per_side = getattr(policy, "max_open_trades_per_side", None)
        if max_per_side is not None and store is not None:
            try:
                open_trades = store.list_open_trades(profile.profile_name)
                side_counts: dict[str, int] = {"buy": 0, "sell": 0}
                for t in open_trades:
                    row = dict(t) if hasattr(t, "keys") else t
                    s = str(row.get("side", "")).lower()
                    if s in side_counts:
                        side_counts[s] += 1
                sc = side_counts.get(side, 0)
                filters.append(report_max_trades(sc, max_per_side, side, side_counts))
                zone_cap = getattr(policy, "max_zone_entry_open", None)
                if zone_cap is not None:
                    zone_open = sum(1 for t in open_trades if (dict(t) if hasattr(t, "keys") else t).get("entry_type") == "zone_entry")
                    filters.append(report_open_trade_cap_by_entry_type("zone_entry", zone_open, zone_cap))
                tier_cap = getattr(policy, "max_tiered_pullback_open", None)
                if tier_cap is not None:
                    tier_open = sum(1 for t in open_trades if (dict(t) if hasattr(t, "keys") else t).get("entry_type") == "tiered_pullback")
                    filters.append(report_open_trade_cap_by_entry_type("tiered_pullback", tier_open, tier_cap))
            except Exception:
                pass

    return filters
