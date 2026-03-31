"""Build dashboard filter list the same way the run loop does.

Used by both run_loop (when writing dashboard_state.json) and the API
(when serving /api/data/{profile}/dashboard) so the dashboard always
shows the same filter status the loop uses.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from core.dashboard_models import FilterReport
from core.presets import FROZEN_PHASE3_DEFENDED_PRESET_ID
from core.signal_engine import drop_incomplete_last_bar
from core.dashboard_reporters import (
    report_session_filter,
    report_session_boundary_block,
    report_spread,
    report_tiered_atr_trial_4,
    report_trial10_stop_loss,
    report_daily_hl_filter,
    report_ema_zone_filter,
    report_rolling_danger_zone,
    report_rsi_divergence,
    report_dual_atr_trial_5,
    report_dead_zone,
    report_trend_exhaustion,
    report_max_trades,
    report_max_trades_by_side,
    report_t6_dead_zone,
    report_t6_m3_trend,
    report_t6_bb_reversal_cap,
    report_ema_zone_slope_filter_trial_7,
    report_trial7_m5_ema_distance_gate,
    report_trial7_adaptive_tp,
    report_trial7_reversal_risk,
    report_trial10_entry_gates,
    report_trial10_pullback_quality,
    report_regime_gate,
    report_open_trade_cap_by_entry_type,
    report_t8_exit_strategy,
    report_daily_level_filter,
    report_ntz_status,
    report_intraday_fib_corridor,
    report_kill_switch_status,
    report_conviction_sizing,
    report_runner_score,
    report_open_exposure,
    report_h1_levels,
    report_m5_trend_alignment,
    report_m5_power_close_status,
    report_up_spread_veto,
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
    "m5_trend_source",
    "m1_zone_entry_ema_slow",
    "m1_pullback_cross_ema_slow",
    "regime_gate_enabled",
    "regime_london_sell_veto",
    "regime_london_start_hour_et",
    "regime_london_end_hour_et",
    "regime_boost_multiplier",
    "regime_buy_base_multiplier",
    "regime_sell_base_multiplier",
    "regime_chop_pause_enabled",
    "regime_chop_pause_minutes",
    "regime_chop_pause_lookback_trades",
    "regime_chop_pause_stop_rate",
    "tier17_nonboost_multiplier",
    "max_open_trades_per_side",
    "bucketed_exit_enabled",
    "quick_tp1_pips",
    "quick_tp1_close_pct",
    "quick_be_spread_plus_pips",
    "runner_tp1_pips",
    "runner_tp1_close_pct",
    "runner_be_spread_plus_pips",
    "trail_escalation_enabled",
    "trail_escalation_tier1_pips",
    "trail_escalation_tier2_pips",
    "trail_escalation_m15_ema_period",
    "trail_escalation_m15_buffer_pips",
    "runner_score_sizing_enabled",
    "runner_base_lots",
    "runner_min_lots",
    "runner_max_lots",
    "atr_stop_enabled",
    "atr_stop_multiplier",
    "atr_stop_max_pips",
    "conviction_base_lots",
    "conviction_min_lots",
    "exit_strategy",
    "hwm_trail_pips",
    "tp1_pips",
    "tp1_close_pct",
    "be_spread_plus_pips",
    "trail_ema_period",
    "trail_m5_ema_period",
)


class _EffectivePolicy:
    """Wraps a policy and applies temp_overrides so dashboard uses same EMAs as execution."""

    def __init__(self, policy: Any, overrides: dict[str, Any]) -> None:
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
    overrides_clean = {k: v for k, v in temp_overrides.items() if v is not None and k in _TEMP_OVERRIDE_ATTRS}
    if not overrides_clean:
        return policy
    return _EffectivePolicy(policy, overrides_clean)


def _live_side_counts(
    profile: Any,
    adapter: Optional[Any],
    live_positions_snapshot: Optional[list[Any]] = None,
) -> tuple[dict[str, int], bool]:
    """Count live broker positions by side (buy/sell). Returns (counts, ok)."""
    side_counts: dict[str, int] = {"buy": 0, "sell": 0}
    if live_positions_snapshot is None and adapter is None:
        return side_counts, False
    try:
        open_positions = live_positions_snapshot
        if open_positions is None:
            open_positions = adapter.get_open_positions(profile.symbol)
        if not open_positions:
            return side_counts, True
        for pos in open_positions:
            if isinstance(pos, dict):
                units = float(pos.get("currentUnits") or pos.get("initialUnits") or 0)
                if units > 0:
                    side_counts["buy"] += 1
                elif units < 0:
                    side_counts["sell"] += 1
            else:
                mt5_type = getattr(pos, "type", None)
                if mt5_type == 0:
                    side_counts["buy"] += 1
                elif mt5_type == 1:
                    side_counts["sell"] += 1
    except Exception:
        return side_counts, False
    return side_counts, True


def _store_side_counts(store: Optional[Any], profile_name: str) -> dict[str, int]:
    """Count open DB trades by side as fallback."""
    side_counts: dict[str, int] = {"buy": 0, "sell": 0}
    if store is None:
        return side_counts
    try:
        open_trades = store.list_open_trades(profile_name)
        for t in open_trades:
            row = dict(t) if hasattr(t, "keys") else t
            s = str(row.get("side", "")).lower()
            if s in side_counts:
                side_counts[s] += 1
    except Exception:
        pass
    return side_counts


def _live_position_ids(
    profile: Any,
    adapter: Optional[Any],
    live_positions_snapshot: Optional[list[Any]] = None,
) -> Optional[set[int]]:
    """Return set of broker position ids that are currently open (for cap counts).

    Returns None when live data is unavailable (adapter missing or fetch failed) —
    callers should fall back to DB count in that case.
    Returns an empty set when data was fetched successfully but 0 positions are open.
    """
    if live_positions_snapshot is None and adapter is None:
        return None
    try:
        positions = live_positions_snapshot
        if positions is None:
            positions = adapter.get_open_positions(profile.symbol)
        ids: set[int] = set()
        for pos in (positions or []):
            pid = pos.get("id") if isinstance(pos, dict) else getattr(pos, "ticket", None)
            if pid is not None:
                try:
                    ids.add(int(pid))
                except (TypeError, ValueError):
                    pass
        return ids
    except Exception:
        return None


def _side_from_eval_result(eval_result: Optional[dict]) -> str:
    side = ""
    if not eval_result:
        return side
    if eval_result.get("candidate_side") in ("buy", "sell"):
        return eval_result["candidate_side"]
    dec = eval_result.get("decision")
    if dec and getattr(dec, "side", None):
        raw = str(dec.side).lower()
        if raw in ("buy", "sell"):
            return raw
    if eval_result.get("side"):
        raw = str(eval_result["side"]).lower()
        if raw in ("buy", "sell"):
            return raw
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
    try:
        from core.execution_engine import _resolve_m5_trend_close_series
        from core.indicators import ema as ema_fn
        fast_p = getattr(policy, "m5_trend_ema_fast", 9)
        slow_p = getattr(policy, "m5_trend_ema_slow", 21)
        close, _source = _resolve_m5_trend_close_series(policy, data_by_tf)
        if len(close) < slow_p + 1:
            return "buy"
        fast_v = float(ema_fn(close, fast_p).iloc[-1])
        slow_v = float(ema_fn(close, slow_p).iloc[-1])
        return "buy" if fast_v > slow_v else "sell"
    except Exception:
        return "buy"




def _phase3_defended_filter_reports(active_preset_name: str | None, cfg: dict[str, Any]) -> list[FilterReport]:
    normalized = str(active_preset_name or "").strip().lower()
    if not (normalized == FROZEN_PHASE3_DEFENDED_PRESET_ID or normalized.startswith(f"{FROZEN_PHASE3_DEFENDED_PRESET_ID} ")):
        return []
    reports: list[FilterReport] = []
    ldn_cfg = cfg.get("london_v2", {}) if isinstance(cfg.get("london_v2"), dict) else {}
    v44_cfg = cfg.get("v44_ny", {}) if isinstance(cfg.get("v44_ny"), dict) else {}
    v14_cfg = cfg.get("v14", {}) if isinstance(cfg.get("v14"), dict) else {}
    blocked_days = ldn_cfg.get("d_suppress_weekdays") or []
    veto_cells = v44_cfg.get("defensive_veto_cells") or []
    cell_scale_overrides = v14_cfg.get("cell_scale_overrides") or {}
    t3_scale = cell_scale_overrides.get("ambiguous/er_mid/der_pos:sell")
    reports.append(FilterReport(
        filter_id="phase3_frozen_package",
        display_name="Frozen Package",
        enabled=True,
        is_clear=True,
        current_value="Phase 3 Frozen V7 Defended",
        explanation="Promoted defended package is active for this profile.",
    ))
    reports.append(FilterReport(
        filter_id="phase3_frozen_l1_weekdays",
        display_name="L1 Weekday Rule",
        enabled=True,
        is_clear=True,
        current_value=", ".join(str(d) for d in blocked_days) + " blocked" if blocked_days else "no suppression",
        explanation="Setup D is suppressed on the defended preset weekdays.",
    ))
    reports.append(FilterReport(
        filter_id="phase3_frozen_l1_exit",
        display_name="L1 Exit Policy",
        enabled=True,
        is_clear=True,
        current_value=f"TP1 {float(ldn_cfg.get('d_tp1_r', 0.0)):.2f}R / BE {float(ldn_cfg.get('d_be_offset_pips', 0.0)):.1f} / TP2 {float(ldn_cfg.get('d_tp2_r', 0.0)):.1f}R",
        explanation="Setup D uses the defended frozen exit override.",
    ))
    reports.append(FilterReport(
        filter_id="phase3_frozen_defensive_veto",
        display_name="Defensive Veto",
        enabled=True,
        is_clear=True,
        current_value=f"v44_ny blocked in {', '.join(str(c) for c in veto_cells)}" if veto_cells else "off",
        explanation="The defended preset keeps the V44 defensive veto active.",
    ))
    reports.append(FilterReport(
        filter_id="phase3_frozen_t3_scale",
        display_name="T3 Scale",
        enabled=True,
        is_clear=True,
        current_value=f"ambiguous/er_mid/der_pos:sell = {float(t3_scale):.2f}x" if t3_scale is not None else "default",
        explanation="The defended preset keeps the reduced T3 sell sizing.",
    ))
    return reports

def _phase3_dict_to_filter_report(d: dict) -> FilterReport:
    """Convert Phase 3 report dict (name, value, ok, detail) to FilterReport."""
    name = str(d.get("name", "Phase 3"))
    filter_id = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
    ok = bool(d.get("ok", True))
    return FilterReport(
        filter_id=filter_id,
        display_name=name,
        enabled=True,
        is_clear=ok,
        current_value=str(d.get("value", "")),
        threshold=str(d.get("detail", "")),
        block_reason=None if ok else str(d.get("detail", "")),
    )


def _phase3_additive_payload(eval_result: Optional[dict], phase3_state: Optional[dict]) -> tuple[dict[str, Any], dict[str, Any]]:
    fallback_eval = phase3_state.get("last_phase3_eval", {}) if isinstance(phase3_state, dict) and isinstance(phase3_state.get("last_phase3_eval"), dict) else {}
    envelope = dict((eval_result or {}).get("phase3_additive_envelope") or {})
    if not envelope:
        envelope = dict(fallback_eval.get("additive_envelope") or {})
    truth = dict((eval_result or {}).get("phase3_additive_truth") or {})
    if not truth:
        truth = dict(fallback_eval.get("additive_truth") or {})
    return envelope, truth


def _phase3_additive_filter_reports(
    active_preset_name: str | None,
    eval_result: Optional[dict],
    phase3_state: Optional[dict],
) -> list[FilterReport]:
    if not (str(active_preset_name or "").strip().lower() == FROZEN_PHASE3_DEFENDED_PRESET_ID or str(active_preset_name or "").strip().lower().startswith(f"{FROZEN_PHASE3_DEFENDED_PRESET_ID} ")):
        return []
    envelope, truth = _phase3_additive_payload(eval_result, phase3_state)
    accepted = list(envelope.get("accepted") or [])
    rejected = list(envelope.get("rejected") or [])
    offensive_intents = list(envelope.get("offensive_intents") or [])
    filters = [
        FilterReport(
            filter_id="phase3_additive_mode",
            display_name="Additive Runtime",
            enabled=True,
            is_clear=True,
            current_value="defended_additive_runtime_v1",
            explanation="Defended preset is using additive-book routing instead of session-first routing.",
        ),
        FilterReport(
            filter_id="phase3_additive_book",
            display_name="Additive Book",
            enabled=True,
            is_clear=True,
            current_value=f"open={truth.get('open_book_count_before', 0)} / candidates={truth.get('candidate_count', 0)} / accepted={truth.get('accepted_count', len(accepted))}",
            explanation="Baseline and offensive intents are counted from the additive envelope.",
        ),
        FilterReport(
            filter_id="phase3_additive_baseline",
            display_name="Baseline Intents",
            enabled=True,
            is_clear=True,
            current_value=f"{truth.get('baseline_candidate_count', len(envelope.get('baseline_intents') or []))} generated",
            explanation="Baseline candidates bypass old session-routed filter panels.",
        ),
        FilterReport(
            filter_id="phase3_additive_offensive",
            display_name="Offensive Slices",
            enabled=True,
            is_clear=True,
            current_value=f"{truth.get('offensive_candidate_count', len(offensive_intents))} generated / {truth.get('accepted_offensive_count', sum(1 for row in accepted if str(row.get('intent_source')) == 'offensive'))} accepted",
            explanation="Offensive slice admission now comes from the additive envelope, not session-only filters.",
        ),
    ]
    if rejected:
        first = dict(rejected[0] or {})
        filters.append(
            FilterReport(
                filter_id="phase3_additive_first_block",
                display_name="First Additive Block",
                enabled=True,
                is_clear=False,
                current_value=str(first.get("intent_source") or "blocked"),
                threshold=str(first.get("slice_id") or ""),
                block_reason=str(first.get("reason") or "blocked"),
                explanation="This is the first additive reject recorded on the current bar.",
            )
        )
    else:
        filters.append(
            FilterReport(
                filter_id="phase3_additive_first_block",
                display_name="First Additive Block",
                enabled=True,
                is_clear=True,
                current_value="none",
                explanation="No additive candidates were rejected on the current bar.",
            )
        )
    filters.append(
        FilterReport(
            filter_id="phase3_additive_quarantine",
            display_name="Legacy Session Panels",
            enabled=True,
            is_clear=True,
            current_value="quarantined",
            explanation="Tokyo/London/NY session-routed panels are quarantined for the defended additive runtime until they are rebuilt from additive truth.",
        )
    )
    return filters


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
    adapter: Optional[Any] = None,
    live_positions_snapshot: Optional[list[Any]] = None,
    temp_overrides: Optional[dict[str, Any]] = None,
    daily_level_filter_snapshot: Optional[dict] = None,
    ntz_filter_snapshot: Optional[dict] = None,
    intraday_fib_corridor_snapshot: Optional[dict] = None,
    conviction_snapshot: Optional[dict] = None,
    regime_snapshot: Optional[dict] = None,
    runner_snapshot: Optional[dict] = None,
    phase3_state: Optional[dict] = None,
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

    side = _side_from_eval_result(eval_result) if eval_result else ""
    trigger_type = "zone_entry"
    if eval_result:
        trigger_type = eval_result.get("candidate_trigger") or eval_result.get("trigger_type") or "zone_entry"
    # Trial #7/#8 dashboard filter context should follow current M5 trend direction,
    # not a stale/missing eval_result side (which can default to BUY).
    if policy is not None and policy_type in ("kt_cg_trial_7", "kt_cg_trial_8"):
        if side not in ("buy", "sell"):
            side = _side_from_m5(data_by_tf, policy)
    elif policy is not None and side not in ("buy", "sell"):
        side = _side_from_m3(data_by_tf, policy)
    if side not in ("buy", "sell"):
        side = "buy"

    # Session filter (all)
    filters.append(report_session_filter(profile, now_utc))
    filters.append(report_session_boundary_block(profile, now_utc))

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
        if max_per_side is not None:
            try:
                side_counts, live_ok = _live_side_counts(profile, adapter, live_positions_snapshot)
                if not live_ok:
                    side_counts = _store_side_counts(store, profile.profile_name)
                filters.extend(report_max_trades_by_side(side_counts, max_per_side))
            except Exception:
                pass

    elif policy_type == "kt_cg_trial_6" and policy is not None:
        filters.append(report_t6_m3_trend(eval_result))
        filters.append(report_t6_dead_zone(policy, daily_reset_state or {}))
        if store is not None:
            filters.append(report_t6_bb_reversal_cap(policy, store, profile.profile_name))
        max_per_side = getattr(policy, "max_open_trades_per_side", None)
        if max_per_side is not None:
            try:
                side_counts, live_ok = _live_side_counts(profile, adapter, live_positions_snapshot)
                if not live_ok:
                    side_counts = _store_side_counts(store, profile.profile_name)
                filters.extend(report_max_trades_by_side(side_counts, max_per_side))
            except Exception:
                pass

    elif policy_type == "kt_cg_trial_7" and policy is not None:
        m5_df = data_by_tf.get("M5")
        filters.append(report_trial7_m5_ema_distance_gate(policy, data_by_tf, pip_size))
        m1_df = data_by_tf.get("M1")
        if m1_df is not None and not m1_df.empty:
            filters.append(report_ema_zone_slope_filter_trial_7(policy, m1_df, pip_size, side))
        if getattr(policy, "trend_exhaustion_enabled", False):
            filters.append(report_trend_exhaustion(exhaustion_result))
        filters.append(report_trial7_adaptive_tp(policy, exhaustion_result))
        rr_result = eval_result.get("reversal_risk_result") if eval_result else None
        filters.append(report_trial7_reversal_risk(policy, rr_result))
        cap_multiplier = 1.0
        cap_min = 1
        if (
            exhaustion_result
            and str(exhaustion_result.get("zone", "")).lower() == "very_extended"
            and bool(getattr(policy, "trend_exhaustion_very_extended_tighten_caps", True))
        ):
            cap_multiplier = max(0.05, float(getattr(policy, "trend_exhaustion_very_extended_cap_multiplier", 0.5)))
            cap_min = max(1, int(getattr(policy, "trend_exhaustion_very_extended_cap_min", 1)))
        max_per_side = getattr(policy, "max_open_trades_per_side", None)
        if max_per_side is not None:
            max_per_side = max(cap_min, int(round(float(max_per_side) * cap_multiplier)))
        if (max_per_side is not None or getattr(profile.risk, "max_open_trades", None) is not None) and store is not None:
            try:
                side_counts, live_ok = _live_side_counts(profile, adapter, live_positions_snapshot)
                if not live_ok:
                    side_counts = _store_side_counts(store, profile.profile_name)
                max_open_trades_total = getattr(profile.risk, "max_open_trades", None)
                if max_open_trades_total is not None:
                    total_open = side_counts.get("buy", 0) + side_counts.get("sell", 0)
                    filters.append(report_max_trades(total_open, int(max_open_trades_total), side, side_counts))
                if max_per_side is not None:
                    filters.extend(report_max_trades_by_side(side_counts, max_per_side))
                buy_lots = 0.0
                sell_lots = 0.0
                for pos in live_positions_snapshot or []:
                    try:
                        if isinstance(pos, dict):
                            units = float(pos.get("currentUnits") or pos.get("initialUnits") or 0)
                            lots = abs(units) / 100000.0
                            if units > 0:
                                buy_lots += lots
                            elif units < 0:
                                sell_lots += lots
                        else:
                            volume = float(getattr(pos, "volume", 0.0) or 0.0)
                            mt5_type = getattr(pos, "type", None)
                            if mt5_type == 0:
                                buy_lots += volume
                            elif mt5_type == 1:
                                sell_lots += volume
                    except Exception:
                        continue
                if buy_lots > 0 or sell_lots > 0:
                    filters.append(report_open_exposure(buy_lots + sell_lots, buy_lots, sell_lots))
                open_trades = store.list_open_trades(profile.profile_name)
                # Prefer live broker position ids so cap reflects actually open positions (not stale DB rows)
                live_ids = _live_position_ids(profile, adapter, live_positions_snapshot)

                def _trade_still_open(row: dict) -> bool:
                    pid = row.get("mt5_position_id")
                    if live_ids is None:
                        return True  # no live data: use DB count (backward compat)
                    if pid is None:
                        return False  # have live data but no position_id: don't count (avoid overcount)
                    try:
                        return int(pid) in live_ids
                    except (TypeError, ValueError):
                        return False
                zone_cap = getattr(policy, "max_zone_entry_open", None)
                if zone_cap is not None:
                    zone_cap = max(cap_min, int(round(float(zone_cap) * cap_multiplier)))
                    zone_open = sum(
                        1 for t in open_trades
                        if ((dict(t) if hasattr(t, "keys") else t).get("entry_type") == "zone_entry")
                        and _trade_still_open(dict(t) if hasattr(t, "keys") else t)
                    )
                    filters.append(report_open_trade_cap_by_entry_type("zone_entry", zone_open, zone_cap))
                tier_cap = getattr(policy, "max_tiered_pullback_open", None)
                if tier_cap is not None:
                    tier_cap = max(cap_min, int(round(float(tier_cap) * cap_multiplier)))
                    tier_open = sum(
                        1 for t in open_trades
                        if ((dict(t) if hasattr(t, "keys") else t).get("entry_type") == "tiered_pullback")
                        and _trade_still_open(dict(t) if hasattr(t, "keys") else t)
                    )
                    filters.append(report_open_trade_cap_by_entry_type("tiered_pullback", tier_open, tier_cap))
            except Exception:
                pass

    elif policy_type == "kt_cg_trial_8" and policy is not None:
        filters.append(report_t8_exit_strategy(policy))
        m5_df = data_by_tf.get("M5")
        filters.append(report_trial7_m5_ema_distance_gate(policy, data_by_tf, pip_size))
        # No EMA zone slope filter for Trial #8 (disabled by design)
        if getattr(policy, "trend_exhaustion_enabled", False):
            filters.append(report_trend_exhaustion(exhaustion_result))
        filters.append(report_trial7_adaptive_tp(policy, exhaustion_result))
        # Daily Level Filter (Trial #8 specific)
        filters.append(report_daily_level_filter(policy, tick, side, pip_size, daily_level_filter_snapshot))
        # No reversal risk for Trial #8 (disabled by design)
        cap_multiplier = 1.0
        cap_min = 1
        if (
            exhaustion_result
            and str(exhaustion_result.get("zone", "")).lower() == "very_extended"
            and bool(getattr(policy, "trend_exhaustion_very_extended_tighten_caps", True))
        ):
            cap_multiplier = max(0.05, float(getattr(policy, "trend_exhaustion_very_extended_cap_multiplier", 0.5)))
            cap_min = max(1, int(getattr(policy, "trend_exhaustion_very_extended_cap_min", 1)))
        max_per_side = getattr(policy, "max_open_trades_per_side", None)
        if max_per_side is not None:
            max_per_side = max(cap_min, int(round(float(max_per_side) * cap_multiplier)))
        if (max_per_side is not None or getattr(profile.risk, "max_open_trades", None) is not None) and store is not None:
            try:
                side_counts, live_ok = _live_side_counts(profile, adapter, live_positions_snapshot)
                if not live_ok:
                    side_counts = _store_side_counts(store, profile.profile_name)
                max_open_trades_total = getattr(profile.risk, "max_open_trades", None)
                if max_open_trades_total is not None:
                    total_open = side_counts.get("buy", 0) + side_counts.get("sell", 0)
                    filters.append(report_max_trades(total_open, int(max_open_trades_total), side, side_counts))
                if max_per_side is not None:
                    filters.extend(report_max_trades_by_side(side_counts, max_per_side))
                buy_lots = 0.0
                sell_lots = 0.0
                for pos in live_positions_snapshot or []:
                    try:
                        if isinstance(pos, dict):
                            units = float(pos.get("currentUnits") or pos.get("initialUnits") or 0)
                            lots = abs(units) / 100000.0
                            if units > 0:
                                buy_lots += lots
                            elif units < 0:
                                sell_lots += lots
                        else:
                            volume = float(getattr(pos, "volume", 0.0) or 0.0)
                            mt5_type = getattr(pos, "type", None)
                            if mt5_type == 0:
                                buy_lots += volume
                            elif mt5_type == 1:
                                sell_lots += volume
                    except Exception:
                        continue
                if buy_lots > 0 or sell_lots > 0:
                    filters.append(report_open_exposure(buy_lots + sell_lots, buy_lots, sell_lots))
                open_trades = store.list_open_trades(profile.profile_name)
                live_ids = _live_position_ids(profile, adapter, live_positions_snapshot)
                db_live_ids: set[int] = set()
                try:
                    for _t in open_trades:
                        _row = dict(_t) if hasattr(_t, "keys") else _t
                        _pid = _row.get("mt5_position_id")
                        if _pid is None:
                            continue
                        try:
                            db_live_ids.add(int(_pid))
                        except (TypeError, ValueError):
                            continue
                except Exception:
                    db_live_ids = set()
                unmatched_live = (live_ids - db_live_ids) if live_ids is not None else set()

                def _trade_still_open_t8(row: dict) -> bool:
                    pid = row.get("mt5_position_id")
                    if live_ids is None:
                        return True
                    if pid is None:
                        return False
                    try:
                        return int(pid) in live_ids
                    except (TypeError, ValueError):
                        return False

                zone_cap = getattr(policy, "max_zone_entry_open", None)
                if zone_cap is not None:
                    zone_cap = max(cap_min, int(round(float(zone_cap) * cap_multiplier)))
                    zone_open = sum(
                        1 for t in open_trades
                        if ((dict(t) if hasattr(t, "keys") else t).get("entry_type") == "zone_entry")
                        and _trade_still_open_t8(dict(t) if hasattr(t, "keys") else t)
                    )
                    fr = report_open_trade_cap_by_entry_type("zone_entry", zone_open, zone_cap)
                    if unmatched_live:
                        fr.metadata["unclassified_live_trade_ids"] = sorted(unmatched_live)
                    filters.append(fr)
                tier_cap = getattr(policy, "max_tiered_pullback_open", None)
                if tier_cap is not None:
                    tier_cap = max(cap_min, int(round(float(tier_cap) * cap_multiplier)))
                    tier_open = sum(
                        1 for t in open_trades
                        if ((dict(t) if hasattr(t, "keys") else t).get("entry_type") == "tiered_pullback")
                        and _trade_still_open_t8(dict(t) if hasattr(t, "keys") else t)
                    )
                    fr = report_open_trade_cap_by_entry_type("tiered_pullback", tier_open, tier_cap)
                    if unmatched_live:
                        fr.metadata["unclassified_live_trade_ids"] = sorted(unmatched_live)
                    filters.append(fr)
            except Exception:
                pass

    elif policy_type in ("kt_cg_trial_9", "kt_cg_trial_10") and policy is not None:
        filters.append(report_t8_exit_strategy(policy))
        m5_df = data_by_tf.get("M5")
        filters.append(report_trial7_m5_ema_distance_gate(policy, data_by_tf, pip_size))
        if policy_type == "kt_cg_trial_10":
            _buy_lots = 0.0
            _sell_lots = 0.0
            _db_trial10_position_ids: set[int] = set()
            _live_trial10_pos_meta: dict[int, tuple[str, float, bool]] = {}
            for pos in live_positions_snapshot or []:
                try:
                    _is_trial10_pos = False
                    _pos_id = None
                    _side = None
                    _lots = 0.0
                    if isinstance(pos, dict):
                        _pos_id = pos.get("id")
                        units = float(pos.get("currentUnits") or pos.get("initialUnits") or 0)
                        _lots = abs(units) / 100000.0
                        if units > 0:
                            _side = "buy"
                        elif units < 0:
                            _side = "sell"
                        _comment = ""
                        _client_ext = pos.get("clientExtensions")
                        if isinstance(_client_ext, dict):
                            _comment = str(_client_ext.get("comment") or "").strip()
                        if not _comment:
                            _trade_ext = pos.get("tradeClientExtensions")
                            if isinstance(_trade_ext, dict):
                                _comment = str(_trade_ext.get("comment") or "").strip()
                        _is_trial10_pos = _comment.startswith("kt_cg_trial_10:")
                    else:
                        _pos_id = getattr(pos, "ticket", None)
                        volume = float(getattr(pos, "volume", 0.0) or 0.0)
                        _lots = abs(volume)
                        mt5_type = getattr(pos, "type", None)
                        if mt5_type == 0:
                            _side = "buy"
                        elif mt5_type == 1:
                            _side = "sell"
                        _is_trial10_pos = str(getattr(pos, "comment", "") or "").startswith("kt_cg_trial_10:")
                    if _pos_id is not None:
                        try:
                            _live_trial10_pos_meta[int(_pos_id)] = (str(_side or ""), float(_lots), bool(_is_trial10_pos))
                        except (TypeError, ValueError):
                            pass
                except Exception:
                    continue
            if store is not None:
                try:
                    _open_trades_t10 = store.list_open_trades(profile.profile_name)
                    for _t in _open_trades_t10:
                        _row = dict(_t) if hasattr(_t, "keys") else _t
                        if str(_row.get("policy_type", "")) != "kt_cg_trial_10":
                            continue
                        _side = str(_row.get("side", "")).lower()
                        _lots = float(_row.get("size_lots") or 0.0)
                        if _side == "buy":
                            pass
                        elif _side == "sell":
                            pass
                        _pid = _row.get("mt5_position_id")
                        _live_lots = None
                        if _pid is not None:
                            try:
                                _pid_int = int(_pid)
                                _db_trial10_position_ids.add(_pid_int)
                                _live_meta = _live_trial10_pos_meta.get(_pid_int)
                                if _live_meta is not None:
                                    _live_lots = float(_live_meta[1])
                            except (TypeError, ValueError):
                                pass
                        _effective_lots = _live_lots if _live_lots is not None else _lots
                        if _side == "buy":
                            _buy_lots += _effective_lots
                        elif _side == "sell":
                            _sell_lots += _effective_lots
                except Exception:
                    _db_trial10_position_ids = set()
            for pos in live_positions_snapshot or []:
                try:
                    _is_trial10_pos = False
                    _pos_id = None
                    _side = None
                    _lots = 0.0
                    if isinstance(pos, dict):
                        _pos_id = pos.get("id")
                        units = float(pos.get("currentUnits") or pos.get("initialUnits") or 0)
                        _lots = abs(units) / 100000.0
                        if units > 0:
                            _side = "buy"
                        elif units < 0:
                            _side = "sell"
                        _comment = ""
                        _client_ext = pos.get("clientExtensions")
                        if isinstance(_client_ext, dict):
                            _comment = str(_client_ext.get("comment") or "").strip()
                        if not _comment:
                            _trade_ext = pos.get("tradeClientExtensions")
                            if isinstance(_trade_ext, dict):
                                _comment = str(_trade_ext.get("comment") or "").strip()
                        _is_trial10_pos = _comment.startswith("kt_cg_trial_10:")
                    else:
                        _pos_id = getattr(pos, "ticket", None)
                        volume = float(getattr(pos, "volume", 0.0) or 0.0)
                        _lots = abs(volume)
                        mt5_type = getattr(pos, "type", None)
                        if mt5_type == 0:
                            _side = "buy"
                        elif mt5_type == 1:
                            _side = "sell"
                        _is_trial10_pos = str(getattr(pos, "comment", "") or "").startswith("kt_cg_trial_10:")
                    if _pos_id is not None:
                        try:
                            if int(_pos_id) in _db_trial10_position_ids:
                                continue
                        except (TypeError, ValueError):
                            pass
                    if not _is_trial10_pos:
                        continue
                    if _side == "buy":
                        _buy_lots += _lots
                    elif _side == "sell":
                        _sell_lots += _lots
                except Exception:
                    continue
            filters.append(report_regime_gate(regime_snapshot))
            filters.append(report_trial10_entry_gates(eval_result))
            filters.append(report_trial10_pullback_quality(eval_result))
            filters.append(report_trial10_stop_loss(profile, policy, data_by_tf, pip_size))
            filters.append(report_runner_score(runner_snapshot))
        if getattr(policy, "trend_exhaustion_enabled", False):
            filters.append(report_trend_exhaustion(exhaustion_result))
        filters.append(report_trial7_adaptive_tp(policy, exhaustion_result))
        filters.append(report_ntz_status(ntz_filter_snapshot, tick, pip_size))
        filters.append(report_intraday_fib_corridor(intraday_fib_corridor_snapshot, tick, pip_size))
        filters.append(report_kill_switch_status(policy, data_by_tf, tick, pip_size, side))
        if not (
            getattr(policy, "type", None) == "kt_cg_trial_10"
            and bool(getattr(policy, "runner_score_sizing_enabled", False))
        ):
            filters.append(report_conviction_sizing(conviction_snapshot))
        cap_multiplier = 1.0
        cap_min = 1
        if (
            exhaustion_result
            and str(exhaustion_result.get("zone", "")).lower() == "very_extended"
            and bool(getattr(policy, "trend_exhaustion_very_extended_tighten_caps", True))
        ):
            cap_multiplier = max(0.05, float(getattr(policy, "trend_exhaustion_very_extended_cap_multiplier", 0.5)))
            cap_min = max(1, int(getattr(policy, "trend_exhaustion_very_extended_cap_min", 1)))
        max_per_side = getattr(policy, "max_open_trades_per_side", None)
        if max_per_side is not None:
            max_per_side = max(cap_min, int(round(float(max_per_side) * cap_multiplier)))
        if (max_per_side is not None or getattr(profile.risk, "max_open_trades", None) is not None) and store is not None:
            try:
                side_counts, live_ok = _live_side_counts(profile, adapter, live_positions_snapshot)
                if not live_ok:
                    side_counts = _store_side_counts(store, profile.profile_name)
                max_open_trades_total = getattr(profile.risk, "max_open_trades", None)
                if max_open_trades_total is not None:
                    total_open = side_counts.get("buy", 0) + side_counts.get("sell", 0)
                    filters.append(report_max_trades(total_open, int(max_open_trades_total), side, side_counts))
                if max_per_side is not None:
                    filters.extend(report_max_trades_by_side(side_counts, max_per_side))
                open_trades = store.list_open_trades(profile.profile_name)
                live_ids = _live_position_ids(profile, adapter, live_positions_snapshot)
                db_live_ids: set[int] = set()
                try:
                    for _t in open_trades:
                        _row = dict(_t) if hasattr(_t, "keys") else _t
                        _pid = _row.get("mt5_position_id")
                        if _pid is None:
                            continue
                        try:
                            db_live_ids.add(int(_pid))
                        except (TypeError, ValueError):
                            continue
                except Exception:
                    db_live_ids = set()
                unmatched_live = (live_ids - db_live_ids) if live_ids is not None else set()

                def _trade_still_open_t9(row: dict) -> bool:
                    pid = row.get("mt5_position_id")
                    if live_ids is None:
                        return True
                    if pid is None:
                        return False
                    try:
                        return int(pid) in live_ids
                    except (TypeError, ValueError):
                        return False

                zone_cap = getattr(policy, "max_zone_entry_open", None)
                if zone_cap is not None:
                    zone_cap = max(cap_min, int(round(float(zone_cap) * cap_multiplier)))
                    zone_open = sum(
                        1 for t in open_trades
                        if ((dict(t) if hasattr(t, "keys") else t).get("entry_type") == "zone_entry")
                        and _trade_still_open_t9(dict(t) if hasattr(t, "keys") else t)
                    )
                    fr = report_open_trade_cap_by_entry_type("zone_entry", zone_open, zone_cap)
                    if unmatched_live:
                        fr.metadata["unclassified_live_trade_ids"] = sorted(unmatched_live)
                    filters.append(fr)
                tier_cap = getattr(policy, "max_tiered_pullback_open", None)
                if tier_cap is not None:
                    tier_cap = max(cap_min, int(round(float(tier_cap) * cap_multiplier)))
                    tier_open = sum(
                        1 for t in open_trades
                        if ((dict(t) if hasattr(t, "keys") else t).get("entry_type") == "tiered_pullback")
                        and _trade_still_open_t9(dict(t) if hasattr(t, "keys") else t)
                    )
                    fr = report_open_trade_cap_by_entry_type("tiered_pullback", tier_open, tier_cap)
                    if unmatched_live:
                        fr.metadata["unclassified_live_trade_ids"] = sorted(unmatched_live)
                    filters.append(fr)
            except Exception:
                pass

    elif policy_type == "phase3_integrated" and policy is not None:
        additive_filters = _phase3_additive_filter_reports(getattr(profile, "active_preset_name", None), eval_result, phase3_state)
        _active_preset = str(getattr(profile, "active_preset_name", "") or "").strip().lower()
        _policy_id = str(getattr(policy, "id", "") or "").strip().lower()
        _effective_preset_id = (
            FROZEN_PHASE3_DEFENDED_PRESET_ID
            if (_active_preset == FROZEN_PHASE3_DEFENDED_PRESET_ID or _policy_id == FROZEN_PHASE3_DEFENDED_PRESET_ID)
            else getattr(profile, "active_preset_name", None)
        )
        if additive_filters:
            cfg = {}
            try:
                from core.phase3_integrated_engine import load_phase3_sizing_config
                cfg = load_phase3_sizing_config(preset_id=_effective_preset_id) or {}
            except Exception:
                cfg = {}
            additive_filters[:0] = _phase3_defended_filter_reports(getattr(profile, "active_preset_name", None), cfg or {})
            return additive_filters
        from core.phase3_integrated_engine import (
            classify_session, compute_bb_width_regime, _compute_adx, _compute_atr,
            report_phase3_session, report_phase3_strategy, report_phase3_regime,
            report_phase3_adx, report_phase3_atr,
            compute_asian_range, compute_v44_h1_trend, compute_v44_m5_slope, compute_v44_atr_pct_filter,
            report_phase3_london_range, report_phase3_london_levels,
            report_phase3_ny_trend, report_phase3_ny_slope, report_phase3_ny_atr_filter,
            report_phase3_last_decision,
            report_phase3_tokyo_caps, report_phase3_london_window, report_phase3_london_caps,
            report_phase3_ny_caps,
            ADX_PERIOD, ATR_PERIOD, load_phase3_sizing_config, uk_london_open_utc,
        )
        cfg = load_phase3_sizing_config(preset_id=_effective_preset_id)
        filters.extend(_phase3_defended_filter_reports(getattr(profile, "active_preset_name", None), cfg or {}))
        m1_closed = data_by_tf.get("M1")
        if m1_closed is not None and not m1_closed.empty:
            try:
                m1_closed = drop_incomplete_last_bar(m1_closed, "M1")
            except Exception:
                pass
        now_utc = datetime.now(timezone.utc)
        if m1_closed is not None and not m1_closed.empty and "time" in m1_closed.columns:
            try:
                now_utc = datetime.fromisoformat(str(m1_closed["time"].iloc[-1]).replace("Z", "+00:00"))
                if now_utc.tzinfo is None:
                    now_utc = now_utc.replace(tzinfo=timezone.utc)
            except Exception:
                now_utc = datetime.now(timezone.utc)
        session = classify_session(now_utc, cfg)
        for d in (report_phase3_session(now_utc, cfg), report_phase3_strategy(session)):
            filters.append(_phase3_dict_to_filter_report(d))
        v14_cfg = cfg.get("v14", {}) if isinstance(cfg.get("v14"), dict) else {}
        ldn_cfg = cfg.get("london_v2", {}) if isinstance(cfg.get("london_v2"), dict) else {}
        v44_cfg = cfg.get("v44_ny", {}) if isinstance(cfg.get("v44_ny"), dict) else {}
        if session == "tokyo":
            spread_limit = float(v14_cfg.get("max_entry_spread_pips", 3.0))
        elif session == "london":
            spread_limit = float(ldn_cfg.get("max_entry_spread_pips", 3.5))
        elif session == "ny":
            spread_limit = float(v44_cfg.get("max_entry_spread_pips", 3.0))
        else:
            spread_limit = 0.0
        spread_ok = session is not None and ((tick.ask - tick.bid) / pip_size <= spread_limit)
        filters.append(_phase3_dict_to_filter_report({
            "name": "Allowed to Trade",
            "value": "yes" if (session is not None and spread_ok) else "no",
            "ok": session is not None and spread_ok,
            "detail": f"session={'yes' if session else 'no'}, spread_limit={spread_limit:.1f}p",
        }))
        filters.append(_phase3_dict_to_filter_report(report_phase3_last_decision(eval_result)))
        m5_df = data_by_tf.get("M5")
        if m5_df is not None and not m5_df.empty and len(m5_df) >= 130:
            regime = compute_bb_width_regime(m5_df)
            filters.append(_phase3_dict_to_filter_report(report_phase3_regime(regime)))
        m15_df = data_by_tf.get("M15")
        if m15_df is not None and not m15_df.empty and len(m15_df) >= ADX_PERIOD + 2:
            adx_val = _compute_adx(m15_df)
            filters.append(_phase3_dict_to_filter_report(report_phase3_adx(adx_val, float(v14_cfg.get("adx_max_for_entry", 35.0)))))
            atr_series = _compute_atr(m15_df)
            import pandas as _pd
            atr_val = float(atr_series.iloc[-1]) if _pd.notna(atr_series.iloc[-1]) else 0.0
            filters.append(_phase3_dict_to_filter_report(report_phase3_atr(atr_val, float(v14_cfg.get("atr_max_threshold_price_units", 0.30)))))
        if session == "tokyo" and phase3_state is not None:
            for d in report_phase3_tokyo_caps(phase3_state, now_utc):
                filters.append(_phase3_dict_to_filter_report(d))
        if session == "london":
            filters.append(_phase3_dict_to_filter_report(report_phase3_london_window(now_utc)))
            m1_df = data_by_tf.get("M1")
            if m1_df is not None and not m1_df.empty:
                london_open_hour = int(uk_london_open_utc(now_utc))
                a_hi, a_lo, a_pips, a_ok = compute_asian_range(
                    m1_df,
                    london_open_hour,
                    range_min_pips=float(ldn_cfg.get("arb_range_min_pips", 30.0)),
                    range_max_pips=float(ldn_cfg.get("arb_range_max_pips", 60.0)),
                )
                filters.append(_phase3_dict_to_filter_report(
                    report_phase3_london_range(
                        a_pips,
                        a_ok,
                        float(ldn_cfg.get("arb_range_min_pips", 30.0)),
                        float(ldn_cfg.get("arb_range_max_pips", 60.0)),
                    )
                ))
                filters.append(_phase3_dict_to_filter_report(report_phase3_london_levels(a_hi, a_lo)))
            if phase3_state is not None:
                for d in report_phase3_london_caps(phase3_state):
                    filters.append(_phase3_dict_to_filter_report(d))
        elif session == "ny":
            h1_df = data_by_tf.get("H1")
            m5_df = data_by_tf.get("M5")
            trend = compute_v44_h1_trend(h1_df) if h1_df is not None and not h1_df.empty else None
            filters.append(_phase3_dict_to_filter_report(report_phase3_ny_trend(trend)))
            if m5_df is not None and not m5_df.empty:
                slope_bars = int(v44_cfg.get("slope_bars", 4))
                slope_threshold = float(v44_cfg.get("strong_slope", 0.5))
                slope = compute_v44_m5_slope(m5_df, slope_bars)
                filters.append(_phase3_dict_to_filter_report(report_phase3_ny_slope(slope, slope_threshold)))
                atr_enabled = bool(v44_cfg.get("atr_pct_filter_enabled", True))
                atr_cap = float(v44_cfg.get("atr_pct_cap", 0.67))
                atr_lookback = int(v44_cfg.get("atr_pct_lookback", 200))
                atr_ok = compute_v44_atr_pct_filter(
                    m5_df,
                    enabled=atr_enabled,
                    cap=atr_cap,
                    lookback=atr_lookback,
                )
                filters.append(_phase3_dict_to_filter_report(report_phase3_ny_atr_filter(atr_ok, atr_cap, atr_lookback)))
            if phase3_state is not None:
                for d in report_phase3_ny_caps(phase3_state, now_utc, v44_cfg):
                    filters.append(_phase3_dict_to_filter_report(d))

    elif policy_type == "uncle_parsh_h1_breakout" and policy is not None:
        level_state = eval_result.get("level_updates", []) if eval_result else []
        filters.append(report_h1_levels(policy, data_by_tf, level_state))
        filters.append(report_m5_trend_alignment(policy, data_by_tf))
        filters.append(report_m5_power_close_status(policy, data_by_tf, level_state))
        filters.append(report_up_spread_veto(policy, tick, pip_size))

    return filters
