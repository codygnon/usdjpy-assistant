"""
Phase 3 Integrated Engine.

Single policy routes by UTC session:
- Tokyo: V14 mean reversion
- London: London V2 (Setup A + Setup D)
- NY: V44 session momentum
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import time
import hashlib
from functools import lru_cache, partial
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)
import pandas as pd
from core.fib_pivots import compute_daily_fib_pivots
from core.phase3_overlay_resolver import (
    build_phase3_overlay_state,
    london_setup_d_weekday_block,
    london_setup_d_weekday_block_from_state,
    resolve_v14_cell_scale_override,
    resolve_v14_cell_scale_override_from_state,
    v44_defensive_veto_block,
    v44_defensive_veto_block_from_state,
)
from core.phase3_session_router import build_phase3_route_plan
from core.phase3_ownership_core import (
    compute_phase3_ownership_audit_for_data as compute_phase3_ownership_audit_for_data_shared,
    compute_regime_snapshot as compute_regime_snapshot_shared,
    global_regime_standdown as global_regime_standdown_shared,
    london_v2_ownership_cluster_block as london_v2_ownership_cluster_block_shared,
    v44_regime_block as v44_regime_block_shared,
)
from core.phase3_session_core import (
    classify_phase3_session,
    last_sunday as _shared_last_sunday,
    nth_sunday as _shared_nth_sunday,
    parse_hhmm_to_hour,
    resolve_ny_window_hours,
    resolve_phase3_bar_time,
    uk_london_open_utc as _shared_uk_london_open_utc,
    us_ny_open_utc as _shared_us_ny_open_utc,
)
from core.regime_classifier import RegimeClassifier, RegimeResult, classify_bar as _regime_classify_bar
from core.regime_features import compute_delta_efficiency_ratio, compute_efficiency_ratio
from core.phase3_london_evaluator import evaluate_london_v2_arb, evaluate_london_v2_lmp
from core.phase3_v14_evaluator import (
    compute_v14_lot_size,
    compute_v14_signal_strength,
    compute_v14_sl,
    compute_v14_tp1,
    compute_v14_units_from_config,
    evaluate_v14_confluence,
)
from core.phase3_v44_evaluator import (
    classify_v44_strength,
    compute_v44_atr_pct_filter,
    compute_v44_h1_trend,
    compute_v44_m5_slope,
    compute_v44_sl,
    evaluate_v44_entry,
)
from core.phase3_tokyo_session import execute_tokyo_v14_session
from core.phase3_london_session import execute_london_v2_session
from core.phase3_ny_session import execute_v44_ny_session
from core.phase3_session_support import build_phase3_session_support_from_mapping
from core.phase3_session_helpers import (
    account_sizing_value as _account_sizing_value,
    compute_adx as _compute_adx_impl,
    compute_asian_range as compute_asian_range_impl,
    compute_atr as _compute_atr_impl,
    compute_bb as _compute_bb_impl,
    compute_ema as _compute_ema,
    compute_session_windows as compute_session_windows_impl,
    compute_parabolic_sar as compute_parabolic_sar_impl,
    compute_risk_units as _compute_risk_units,
    compute_rsi as _compute_rsi_impl,
    compute_v44_atr_rank as _compute_v44_atr_rank,
    detect_sar_flip as detect_sar_flip_impl,
    determine_v44_session_mode as _determine_v44_session_mode,
    drop_incomplete_tf as _drop_incomplete_tf_impl,
    is_in_news_window,
    load_news_events_cached as _load_news_events_cached,
    v44_most_recent_news_event as _v44_most_recent_news_event,
    v44_news_trend_active_event as _v44_news_trend_active_event,
)

try:
    from core.execution_engine import ExecutionDecision
except Exception:  # pragma: no cover - runtime fallback
    @dataclass(frozen=True)
    class ExecutionDecision:  # type: ignore[no-redef]
        attempted: bool
        placed: bool
        reason: str
        order_retcode: Optional[int] = None
        order_id: Optional[int] = None
        deal_id: Optional[int] = None
        side: Optional[str] = None
        fill_price: Optional[float] = None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _set_nested_path(payload: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = [p for p in str(dotted_key or "").split(".") if p]
    if not parts:
        return
    cursor: dict[str, Any] = payload
    for part in parts[:-1]:
        node = cursor.get(part)
        if not isinstance(node, dict):
            node = {}
            cursor[part] = node
        cursor = node
    cursor[parts[-1]] = value


def _get_nested_path(payload: dict[str, Any], dotted_key: str) -> Any:
    parts = [p for p in str(dotted_key or "").split(".") if p]
    cursor: Any = payload
    for part in parts:
        if not isinstance(cursor, dict) or part not in cursor:
            return None
        cursor = cursor.get(part)
    return cursor


def _defended_locked_updates(
    *,
    preset_id: str | None,
    canonical_spec: Any,
    normalized_cfg: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    updates: dict[str, Any] = {}
    locked: list[str] = []
    from core.phase3_package_spec import uses_defended_phase3_package as _uses_defended
    if not _uses_defended(preset_id) or canonical_spec is None:
        return updates, locked

    can_v14 = (canonical_spec.runtime_overrides or {}).get("v14", {})
    can_ldn = (canonical_spec.runtime_overrides or {}).get("london_v2", {})
    can_v44 = (canonical_spec.runtime_overrides or {}).get("v44_ny", {})
    can_strict = canonical_spec.strict_policy if isinstance(canonical_spec.strict_policy, dict) else {}

    if isinstance(can_ldn, dict):
        for key in ("d_suppress_weekdays", "d_tp1_r", "d_be_offset_pips", "d_tp2_r"):
            if key in can_ldn:
                _set_nested_path(updates, f"london_v2.{key}", can_ldn[key])
                locked.append(f"london_v2.{key}")

    if isinstance(can_v44, dict) and "defensive_veto_cells" in can_v44:
        _set_nested_path(updates, "v44_ny.defensive_veto_cells", can_v44["defensive_veto_cells"])
        locked.append("v44_ny.defensive_veto_cells")
    if isinstance(can_v44, dict):
        for key in (
            "allow_internal_overlap",
            "allow_opposite_side_overlap",
            "max_open_positions",
            "margin_leverage",
            "margin_buffer_pct",
            "max_lot",
            "rp_max_lot",
        ):
            if key in can_v44:
                _set_nested_path(updates, f"v44_ny.{key}", can_v44[key])
                locked.append(f"v44_ny.{key}")

    if "max_entries_per_day" in can_strict:
        max_entries = can_strict.get("max_entries_per_day")
        _set_nested_path(updates, "v44_ny.max_entries_per_day", None if max_entries is None else int(max_entries))
        locked.append("v44_ny.max_entries_per_day")
    if "max_open_offensive" in can_strict:
        max_open = can_strict.get("max_open_offensive")
        _set_nested_path(updates, "v44_ny.max_open_positions", None if max_open is None else int(max_open))
        locked.append("v44_ny.max_open_positions")

    base_v44 = normalized_cfg.get("v44_ny", {}) if isinstance(normalized_cfg.get("v44_ny"), dict) else {}
    for key in ("ny_window_mode", "ny_start_hour", "ny_end_hour", "start_delay_minutes"):
        if key in base_v44:
            _set_nested_path(updates, f"v44_ny.{key}", base_v44[key])
            locked.append(f"v44_ny.{key}")

    if isinstance(can_v14, dict):
        cell_overrides = can_v14.get("cell_scale_overrides")
        if isinstance(cell_overrides, dict) and "ambiguous/er_mid/der_pos:sell" in cell_overrides:
            _set_nested_path(
                updates,
                "v14.cell_scale_overrides.ambiguous/er_mid/der_pos:sell",
                cell_overrides["ambiguous/er_mid/der_pos:sell"],
            )
            locked.append("v14.cell_scale_overrides.ambiguous/er_mid/der_pos:sell")

    return updates, locked


def apply_defended_locked_keys(
    effective_cfg: dict[str, Any],
    *,
    preset_id: str | None,
    canonical_spec: Any,
    normalized_cfg: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    updates, locked_keys = _defended_locked_updates(
        preset_id=preset_id,
        canonical_spec=canonical_spec,
        normalized_cfg=normalized_cfg,
    )
    if not updates:
        return effective_cfg, []
    overridden: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for key_path in locked_keys:
        if key_path in seen_paths:
            continue
        seen_paths.add(key_path)
        current_val = _get_nested_path(effective_cfg, key_path)
        locked_val = _get_nested_path(updates, key_path)
        if current_val != locked_val:
            overridden.append({"key": key_path, "was": current_val, "forced_to": locked_val})
    out = _deep_merge(effective_cfg, updates)
    if overridden:
        logger.warning(
            "Defended locked keys overrode %d value(s) from integrated config: %s",
            len(overridden),
            overridden,
        )
    return out, locked_keys


def _phase3_order_confirmed(adapter, profile, order_result) -> tuple[bool, Optional[int]]:
    """
    Confirm a market order actually opened on the broker before Phase 3 treats it
    as placed. For OANDA, a market order can return an order id with no immediate
    fill/deal; those must not be counted as open trades.
    """
    deal_id = getattr(order_result, "deal_id", None)
    if deal_id is not None:
        return True, int(deal_id)
    if getattr(profile, "broker_type", None) != "oanda":
        return True, None
    order_id = getattr(order_result, "order_id", None)
    if order_id is None:
        return False, None
    for _ in range(3):
        try:
            position_id = adapter.get_position_id_from_order(int(order_id))
            if position_id is not None:
                try:
                    setattr(order_result, "deal_id", int(position_id))
                except Exception:
                    pass
                return True, int(position_id)
        except Exception:
            position_id = None
        time.sleep(0.5)
    return False, None


def infer_phase3_entry_session(entry_type: Any, entry_session: Any = None) -> str:
    session = str(entry_session or "").lower()
    if session in {"tokyo", "london", "ny"}:
        return session
    et = str(entry_type or "")
    if et.startswith("phase3:v14"):
        return "tokyo"
    if et.startswith("phase3:london_v2"):
        return "london"
    if et.startswith("phase3:v44"):
        return "ny"
    return ""


def phase3_trade_key_date(trade_timestamp_utc: Any, fallback_now_utc: Any) -> str:
    key_date = pd.Timestamp(fallback_now_utc, tz="UTC").date().isoformat()
    try:
        ts = pd.Timestamp(trade_timestamp_utc)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        key_date = ts.date().isoformat()
    except Exception:
        pass
    return key_date


def apply_phase3_session_outcome(
    *,
    phase3_state: dict[str, Any],
    phase3_sizing_cfg: dict[str, Any],
    entry_session: str,
    entry_type: Any,
    is_loss: bool,
    action: str,
    side: str,
    key_date: str,
    now_utc: Any,
) -> dict[str, Any]:
    session = str(entry_session or "").lower()
    if session not in {"tokyo", "london", "ny"}:
        return {}

    now_ts = pd.Timestamp(now_utc)
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize("UTC")
    else:
        now_ts = now_ts.tz_convert("UTC")
    et = str(entry_type or "")

    if session == "tokyo":
        k = f"session_tokyo_{key_date}"
        sd = dict(phase3_state.get(k, {}))
        if is_loss:
            sd["consecutive_losses"] = int(sd.get("consecutive_losses", 0)) + 1
            sd["win_streak"] = 0
            if action == "hard_sl":
                sd[f"last_stopout_time_{side}"] = now_ts.isoformat()
        else:
            sd["consecutive_losses"] = 0
            sd["wins_closed"] = int(sd.get("wins_closed", 0)) + 1
            sd["win_streak"] = int(sd.get("win_streak", 0)) + 1
        phase3_state[k] = sd
        return sd

    if session == "london":
        k = f"session_london_{key_date}"
        sd = dict(phase3_state.get(k, {}))
        if is_loss:
            sd["consecutive_losses"] = int(sd.get("consecutive_losses", 0)) + 1
            sd["win_streak"] = 0
        else:
            sd["consecutive_losses"] = 0
            sd["wins_closed"] = int(sd.get("wins_closed", 0)) + 1
            sd["win_streak"] = int(sd.get("win_streak", 0)) + 1
        ldn_cfg = phase3_sizing_cfg.get("london_v2", {}) if isinstance(phase3_sizing_cfg, dict) else {}
        disable_reset = bool(ldn_cfg.get("disable_channel_reset_after_exit", False))
        if not disable_reset:
            channels = dict(sd.get("channels", {}))
            if et.startswith("phase3:london_v2_arb"):
                channels["A_long" if side == "buy" else "A_short"] = "WAITING_RESET"
            elif et.startswith("phase3:london_v2_d"):
                channels["D_long" if side == "buy" else "D_short"] = "WAITING_RESET"
            sd["channels"] = channels
        phase3_state[k] = sd
        return sd

    k = f"session_ny_{key_date}"
    sd = dict(phase3_state.get(k, {}))
    sd["consecutive_losses"] = int(sd.get("consecutive_losses", 0)) + 1 if is_loss else 0
    sd["wins_closed"] = int(sd.get("wins_closed", 0)) + (0 if is_loss else 1)
    sd["win_streak"] = 0 if is_loss else (int(sd.get("win_streak", 0)) + 1)
    v44_cfg = phase3_sizing_cfg.get("v44_ny", {}) if isinstance(phase3_sizing_cfg, dict) else {}
    scope = str(v44_cfg.get("win_streak_scope", "session")).lower()
    if scope != "session":
        phase3_state["v44_win_streak_global"] = 0 if is_loss else (int(phase3_state.get("v44_win_streak_global", 0)) + 1)
    session_stop_losses = int(v44_cfg.get("session_stop_losses", V44_SESSION_STOP_LOSSES))
    sd["stopped"] = bool(int(sd.get("consecutive_losses", 0)) >= int(session_stop_losses))
    cd_win_bars = int(v44_cfg.get("cooldown_win_bars", V44_COOLDOWN_WIN))
    cd_loss_bars = int(v44_cfg.get("cooldown_loss_bars", V44_COOLDOWN_LOSS))
    cd_minutes = max(0, (cd_loss_bars if is_loss else cd_win_bars) * 5)
    if cd_minutes > 0:
        sd["cooldown_until"] = (now_ts + pd.Timedelta(minutes=cd_minutes)).isoformat()
    phase3_state[k] = sd
    return sd


def _normalize_v14_source(v14_src: dict[str, Any]) -> dict[str, Any]:
    ps = v14_src.get("position_sizing", {}) if isinstance(v14_src.get("position_sizing"), dict) else {}
    tm = v14_src.get("trade_management", {}) if isinstance(v14_src.get("trade_management"), dict) else {}
    exec_model = v14_src.get("execution_model", {}) if isinstance(v14_src.get("execution_model"), dict) else {}
    ind = v14_src.get("indicators", {}) if isinstance(v14_src.get("indicators"), dict) else {}
    atr_i = ind.get("atr", {}) if isinstance(ind.get("atr"), dict) else {}
    adx_i = v14_src.get("adx_filter", {}) if isinstance(v14_src.get("adx_filter"), dict) else {}
    ent = v14_src.get("entry_rules", {}) if isinstance(v14_src.get("entry_rules"), dict) else {}
    long_e = ent.get("long", {}) if isinstance(ent.get("long"), dict) else {}
    short_e = ent.get("short", {}) if isinstance(ent.get("short"), dict) else {}
    core_gate = ent.get("core_gate", {}) if isinstance(ent.get("core_gate"), dict) else {}
    ex = v14_src.get("exit_rules", {}) if isinstance(v14_src.get("exit_rules"), dict) else {}
    ex_tp = ex.get("take_profit", {}) if isinstance(ex.get("take_profit"), dict) else {}
    ex_sl = ex.get("stop_loss", {}) if isinstance(ex.get("stop_loss"), dict) else {}
    ex_tr = ex.get("trailing_stop", {}) if isinstance(ex.get("trailing_stop"), dict) else {}
    ex_tm = ex.get("time_exit", {}) if isinstance(ex.get("time_exit"), dict) else {}
    ccf = v14_src.get("confluence_combo_filter", {}) if isinstance(v14_src.get("confluence_combo_filter"), dict) else {}
    ss = v14_src.get("signal_strength_tracking", {}) if isinstance(v14_src.get("signal_strength_tracking"), dict) else {}
    conf = v14_src.get("entry_confirmation", {}) if isinstance(v14_src.get("entry_confirmation"), dict) else {}
    sf = v14_src.get("session_filter", {}) if isinstance(v14_src.get("session_filter"), dict) else {}
    long_zone = long_e.get("price_zone", {}) if isinstance(long_e.get("price_zone"), dict) else {}
    long_rsi = long_e.get("rsi_soft_filter", {}) if isinstance(long_e.get("rsi_soft_filter"), dict) else {}
    short_rsi = short_e.get("rsi_soft_filter", {}) if isinstance(short_e.get("rsi_soft_filter"), dict) else {}
    long_conf = long_e.get("confluence_scoring", {}) if isinstance(long_e.get("confluence_scoring"), dict) else {}
    short_conf = short_e.get("confluence_scoring", {}) if isinstance(short_e.get("confluence_scoring"), dict) else {}
    bb_regime = ind.get("bb_width_regime_filter", {}) if isinstance(ind.get("bb_width_regime_filter"), dict) else {}
    allowed_days = sf.get("allowed_trading_days", ["Tuesday", "Wednesday", "Friday"])
    return {
        "risk_per_trade_pct": float(ps.get("risk_per_trade_pct", 2.0)),
        "max_units": int(ps.get("max_units", MAX_UNITS)),
        "max_concurrent_positions": int(ps.get("max_concurrent_positions", 2)),
        "leverage": float(ps.get("leverage", LEVERAGE)),
        "day_risk_multipliers": ps.get("day_risk_multipliers", {}) if isinstance(ps.get("day_risk_multipliers"), dict) else {},
        "max_trades_per_session": int(tm.get("max_trades_per_session", 4)),
        "stop_after_consecutive_losses": int(tm.get("stop_after_consecutive_losses", 3)),
        "session_loss_stop_pct": float(tm.get("session_loss_stop_pct", 1.5)),
        "cooldown_minutes": int(tm.get("cooldown_minutes", tm.get("min_time_between_entries_minutes", 10))),
        "min_time_between_entries_minutes": int(tm.get("min_time_between_entries_minutes", 10)),
        "no_reentry_same_direction_after_stop_minutes": int(tm.get("no_reentry_same_direction_after_stop_minutes", 30)),
        "disable_entries_if_move_from_tokyo_open_range_exceeds_pips": float(tm.get("disable_entries_if_move_from_tokyo_open_range_exceeds_pips", 0.0)),
        "breakout_detection_mode": str(tm.get("breakout_detection_mode", "rolling")).lower(),
        "rolling_window_minutes": int(tm.get("rolling_window_minutes", 60)),
        "rolling_range_threshold_pips": float(tm.get("rolling_range_threshold_pips", 40.0)),
        "atr_max_threshold_price_units": float(atr_i.get("max_threshold_price_units", ATR_MAX)),
        "adx_max_for_entry": float(adx_i.get("max_adx_for_entry", ADX_MAX)),
        "adx_filter_enabled": bool(adx_i.get("enabled", True)),
        "bb_regime_mode": str(bb_regime.get("mode", "percentile")).strip().lower(),
        "bb_width_lookback": int(bb_regime.get("percentile_lookback_m5_bars", BB_WIDTH_LOOKBACK)),
        "bb_width_ranging_pct": float(bb_regime.get("ranging_percentile", BB_WIDTH_RANGING_PCT)),
        "zone_tolerance_pips": float(long_zone.get("tolerance_pips", ZONE_TOLERANCE_PIPS)),
        "rsi_long_entry": float(long_rsi.get("entry_soft_threshold", RSI_LONG_ENTRY)),
        "rsi_short_entry": float(short_rsi.get("entry_soft_threshold", RSI_SHORT_ENTRY)),
        "min_confluence": int(min(long_conf.get("minimum_score", MIN_CONFLUENCE), short_conf.get("minimum_score", MIN_CONFLUENCE))),
        "confluence_min_long": int(long_conf.get("minimum_score", MIN_CONFLUENCE)),
        "confluence_min_short": int(short_conf.get("minimum_score", MIN_CONFLUENCE)),
        "blocked_combos": ccf.get("blocked_combos", list(BLOCKED_COMBOS)) if bool(ccf.get("enabled", True)) else [],
        "core_gate_required": int(core_gate.get("required_count", 2)),
        "core_gate_use_zone": bool(core_gate.get("use_zone", True)),
        "core_gate_use_bb": bool(core_gate.get("use_bb_touch", True)),
        "core_gate_use_sar": bool(core_gate.get("use_sar_flip", True)),
        "core_gate_use_rsi": bool(core_gate.get("use_rsi_soft", True)),
        "sl_buffer_pips": float(ex_sl.get("buffer_pips", SL_BUFFER_PIPS)),
        "min_sl_pips": float(ex_sl.get("minimum_sl_pips", SL_MIN_PIPS)),
        "max_sl_pips": float(ex_sl.get("hard_max_sl_pips", SL_MAX_PIPS)),
        "partial_tp_atr_mult": float(ex_tp.get("partial_tp_atr_mult", TP1_ATR_MULT)),
        "partial_tp_min_pips": float(ex_tp.get("partial_tp_min_pips", TP1_MIN_PIPS)),
        "partial_tp_max_pips": float(ex_tp.get("partial_tp_max_pips", TP1_MAX_PIPS)),
        "tp1_close_pct": float(ex_tp.get("partial_close_pct", TP1_CLOSE_PCT)),
        "breakeven_offset_pips": float(ex_tp.get("breakeven_offset_pips", BE_OFFSET_PIPS)),
        "trail_activate_profit_pips": float(ex_tr.get("activate_after_profit_pips", TRAIL_ACTIVATE_PROFIT_PIPS)),
        "trail_distance_pips": float(ex_tr.get("trail_distance_pips", TRAIL_DISTANCE_PIPS)),
        "time_decay_minutes": int(ex_tm.get("time_decay_minutes", TIME_DECAY_MINUTES)),
        "time_decay_profit_cap_pips": float(ex_tm.get("time_decay_profit_cap_pips", TIME_DECAY_CAP_PIPS)),
        "min_tp_distance_pips": float(ex_tp.get("minimum_tp_distance_pips", MIN_TP_DISTANCE_PIPS)),
        "confirmation_enabled": bool(conf.get("enabled", False)),
        "confirmation_type": str(conf.get("type", "m1")).lower(),
        "confirmation_window_bars": int(conf.get("window_bars", 0)),
        "signal_strength_tracking": ss if isinstance(ss, dict) else {},
        "session_start_utc": str(sf.get("session_start_utc", "16:00")),
        "session_end_utc": str(sf.get("session_end_utc", "22:00")),
        "allowed_trading_days": list(allowed_days) if isinstance(allowed_days, list) else ["Tuesday", "Wednesday", "Friday"],
        # Backtest parity: Tokyo backtest defaults to 10.0 when this key is absent.
        "max_entry_spread_pips": float(exec_model.get("max_entry_spread_pips", 10.0)),
    }


def _normalize_london_source(v2_src: dict[str, Any]) -> dict[str, Any]:
    account = v2_src.get("account", {}) if isinstance(v2_src.get("account"), dict) else {}
    session = v2_src.get("session", {}) if isinstance(v2_src.get("session"), dict) else {}
    execution_model = v2_src.get("execution_model", {}) if isinstance(v2_src.get("execution_model"), dict) else {}
    risk = v2_src.get("risk", {}) if isinstance(v2_src.get("risk"), dict) else {}
    limits = v2_src.get("entry_limits", {}) if isinstance(v2_src.get("entry_limits"), dict) else {}
    levels = v2_src.get("levels", {}) if isinstance(v2_src.get("levels"), dict) else {}
    setups = v2_src.get("setups", {}) if isinstance(v2_src.get("setups"), dict) else {}
    sa = setups.get("A", {}) if isinstance(setups.get("A"), dict) else {}
    sd = setups.get("D", {}) if isinstance(setups.get("D"), dict) else {}
    return {
        "risk_per_trade_pct": float(risk.get("risk_per_trade_pct", 0.01)),
        "arb_risk_per_trade_pct": float(sa.get("risk_per_trade_pct", risk.get("risk_per_trade_pct", 0.01))),
        "d_risk_per_trade_pct": float(sd.get("risk_per_trade_pct", risk.get("risk_per_trade_pct", 0.01))),
        "max_total_open_risk_pct": float(risk.get("max_total_open_risk_pct", 0.05)),
        "max_trades_per_day_total": limits.get("max_trades_per_day_total", 1),
        "max_trades_per_setup_per_day": limits.get("max_trades_per_setup_per_day", None),
        "max_trades_per_setup_direction_per_day": limits.get("max_trades_per_setup_direction_per_day", None),
        "disable_channel_reset_after_exit": bool(limits.get("disable_channel_reset_after_exit", False)),
        "active_days_utc": session.get("active_days_utc", ["Tuesday", "Wednesday"]),
        "arb_range_min_pips": float(levels.get("asian_range_min_pips", 30.0)),
        "arb_range_max_pips": float(levels.get("asian_range_max_pips", 60.0)),
        "lor_range_min_pips": float(levels.get("lor_range_min_pips", 4.0)),
        "lor_range_max_pips": float(levels.get("lor_range_max_pips", 20.0)),
        "d_entry_start_min_after_london": int(sd.get("entry_start_min_after_london", 15)),
        "d_entry_end_min_after_london": int(sd.get("entry_end_min_after_london", 120)),
        "d_allow_long": bool(sd.get("allow_long", True)),
        "d_allow_short": bool(sd.get("allow_short", False)),
        "hard_close_at_ny_open": bool(session.get("hard_close_at_ny_open", True)),
        "arb_tp1_r": float(sa.get("tp1_r_multiple", 1.0)),
        "arb_tp2_r": float(sa.get("tp2_r_multiple", 2.0)),
        "arb_tp1_close_pct": float(sa.get("tp1_close_fraction", 0.5)),
        "arb_be_offset_pips": float(sa.get("be_offset_pips", 1.0)),
        "d_tp1_r": float(sd.get("tp1_r_multiple", 1.0)),
        "d_tp2_r": float(sd.get("tp2_r_multiple", 2.0)),
        "d_tp1_close_pct": float(sd.get("tp1_close_fraction", 0.5)),
        "d_be_offset_pips": float(sd.get("be_offset_pips", 1.0)),
        "leverage": float(account.get("leverage", 33.0)),
        "max_margin_usage_fraction_per_trade": float(account.get("max_margin_usage_fraction_per_trade", 0.5)),
        "max_open_positions": int(account.get("max_open_positions", 5)),
        "a_allow_long": bool(sa.get("allow_long", True)),
        "a_allow_short": bool(sa.get("allow_short", True)),
        "a_entry_start_min_after_london": int(sa.get("entry_start_min_after_london", 0)),
        "a_entry_end_min_after_london": int(sa.get("entry_end_min_after_london", 90)),
        "a_reset_mode": str(sa.get("reset_mode", "touch_level")),
        "d_reset_mode": str(sd.get("reset_mode", "touch_level")),
        "max_entry_spread_pips": float(execution_model.get("spread_max_pips", LDN_MAX_SPREAD_PIPS)),
    }


def _normalize_v44_source(v44_src: dict[str, Any]) -> dict[str, Any]:
    ny_start_raw = v44_src.get("ny_start", v44_src.get("v5_ny_start"))
    ny_end_raw = v44_src.get("ny_end", v44_src.get("v5_ny_end"))
    # The benchmark session_momentum backtest uses fixed UTC ny_start/ny_end values.
    # Default live normalization to fixed_utc unless the source config explicitly opts
    # into DST-aware routing.
    ny_window_mode = str(v44_src.get("ny_window_mode", "fixed_utc")).lower()
    ny_daily_loss_usd = v44_src.get("v5_ny_daily_loss_usd", v44_src.get("v5_daily_loss_limit_usd", 0.0))
    return {
        "risk_per_trade_pct": float(v44_src.get("v5_risk_per_trade_pct", V44_RISK_PCT) or V44_RISK_PCT),
        "rp_min_lot": float(v44_src.get("v5_rp_min_lot", 1.0)),
        "rp_max_lot": float(v44_src.get("v5_rp_max_lot", 20.0)),
        "max_open_positions": int(v44_src.get("v5_max_open", V44_MAX_OPEN)),
        "max_entries_per_day": int(v44_src.get("v5_max_entries_day", V44_MAX_ENTRIES_DAY)),
        "session_stop_losses": int(v44_src.get("v5_session_stop_losses", V44_SESSION_STOP_LOSSES)),
        "session_entry_cutoff_minutes": int(v44_src.get("v5_session_entry_cutoff_minutes", 60)),
        "session_range_cap_pips": float(v44_src.get("v5_session_range_cap_pips", 0.0)),
        "cooldown_win_bars": int(v44_src.get("v5_cooldown_win", V44_COOLDOWN_WIN)),
        "cooldown_loss_bars": int(v44_src.get("v5_cooldown_loss", V44_COOLDOWN_LOSS)),
        "cooldown_scratch_bars": int(v44_src.get("v5_cooldown_scratch", 2)),
        "start_delay_minutes": int(v44_src.get("v5_ny_start_delay_minutes", 5)),
        "max_entry_spread_pips": float(v44_src.get("v5_max_entry_spread_pips", v44_src.get("max_entry_spread_pips", V44_MAX_ENTRY_SPREAD))),
        "ny_strength_allow": str(v44_src.get("v5_ny_strength_allow", v44_src.get("v5_strength_allow", "strong_normal"))),
        "skip_weak": bool(v44_src.get("v5_skip_weak", True)),
        "skip_normal": bool(v44_src.get("v5_skip_normal", False)),
        "allow_normal_plus": bool(v44_src.get("v5_allow_normal_plus", False)),
        "normalplus_atr_min_pips": float(v44_src.get("v5_normalplus_atr_min_pips", 6.0)),
        "normalplus_slope_min": float(v44_src.get("v5_normalplus_slope_min", 0.45)),
        "news_filter_enabled": bool(v44_src.get("v5_news_filter_enabled", False)),
        "news_window_minutes_before": int(v44_src.get("v5_news_window_minutes_before", 60)),
        "news_window_minutes_after": int(v44_src.get("v5_news_window_minutes_after", 30)),
        "news_impact_min": str(v44_src.get("v5_news_impact_min", "high")),
        "news_calendar_path": str(v44_src.get("v5_news_calendar_path", "research_out/v5_scheduled_events_utc.csv")),
        "news_trend_enabled": bool(v44_src.get("v5_news_trend_enabled", False)),
        "news_trend_delay_minutes": int(v44_src.get("v5_news_trend_delay_minutes", 45)),
        "news_trend_window_minutes": int(v44_src.get("v5_news_trend_window_minutes", 90)),
        "news_trend_confirm_bars": int(v44_src.get("v5_news_trend_confirm_bars", 3)),
        "news_trend_min_body_pips": float(v44_src.get("v5_news_trend_min_body_pips", 1.5)),
        "news_trend_require_ema_align": bool(v44_src.get("v5_news_trend_require_ema_align", True)),
        "news_trend_risk_pct": float(v44_src.get("v5_news_trend_risk_pct", 0.5)),
        "news_trend_tp_rr": float(v44_src.get("v5_news_trend_tp_rr", 1.5)),
        "news_trend_sl_pips": float(v44_src.get("v5_news_trend_sl_pips", 8.0)),
        "atr_pct_filter_enabled": bool(v44_src.get("v5_atr_pct_filter_enabled", False)),
        "atr_pct_cap": float(v44_src.get("v5_atr_pct_cap", V44_ATR_PCT_CAP)),
        "atr_pct_lookback": int(v44_src.get("v5_atr_pct_lookback", V44_ATR_PCT_LOOKBACK)),
        "skip_days": str(v44_src.get("v5_skip_days", "")),
        "skip_months": str(v44_src.get("v5_skip_months", "")),
        "strong_tp1_pips": float(v44_src.get("v5_strong_tp1", V44_STRONG_TP1_PIPS)),
        "normal_tp1_pips": float(v44_src.get("v5_normal_tp1", 1.75)),
        "weak_tp1_pips": float(v44_src.get("v5_weak_tp1", 1.2)),
        "strong_tp2_pips": float(v44_src.get("v5_strong_tp2", V44_STRONG_TP2_PIPS)),
        "normal_tp2_pips": float(v44_src.get("v5_normal_tp2", 3.0)),
        "weak_tp2_pips": float(v44_src.get("v5_weak_tp2", 2.0)),
        "strong_tp1_close_pct": float(v44_src.get("v5_strong_tp1_close_pct", V44_STRONG_TP1_CLOSE_PCT)),
        "normal_tp1_close_pct": float(v44_src.get("v5_normal_tp1_close_pct", 0.5)),
        "weak_tp1_close_pct": float(v44_src.get("v5_weak_tp1_close_pct", 0.6)),
        "be_offset_pips": float(v44_src.get("v5_be_offset", V44_BE_OFFSET_PIPS)),
        "strong_trail_buffer": float(v44_src.get("v5_strong_trail_buffer_pips", v44_src.get("v5_strong_trail_buffer", V44_STRONG_TRAIL_BUFFER))),
        "normal_trail_buffer": float(v44_src.get("v5_normal_trail_buffer", 3.0)),
        "weak_trail_buffer": float(v44_src.get("v5_weak_trail_buffer", 2.0)),
        "strong_trail_ema": int(v44_src.get("v5_strong_trail_ema", 21)),
        "normal_trail_ema": int(v44_src.get("v5_normal_trail_ema", 9)),
        "trail_start_after_tp1_mult": float(v44_src.get("v5_trail_start_after_tp1_mult", 0.5)),
        "h1_ema_fast": int(v44_src.get("h1_ema_fast", V44_H1_EMA_FAST)),
        "h1_ema_slow": int(v44_src.get("h1_ema_slow", V44_H1_EMA_SLOW)),
        "h1_slope_min": float(v44_src.get("v5_h1_slope_min", 0.0)),
        "h1_slope_consistent_bars": int(v44_src.get("v5_h1_slope_consistent_bars", 0)),
        "m5_ema_fast": int(v44_src.get("v5_m5_ema_fast", V44_M5_EMA_FAST)),
        "m5_ema_slow": int(v44_src.get("v5_m5_ema_slow", V44_M5_EMA_SLOW)),
        "slope_bars": int(v44_src.get("v5_slope_bars", V44_SLOPE_BARS)),
        "strong_slope": float(v44_src.get("v5_strong_slope", V44_STRONG_SLOPE)),
        "weak_slope": float(v44_src.get("v5_weak_slope", 0.2)),
        "entry_min_body_pips": float(v44_src.get("v5_entry_min_body_pips", V44_MIN_BODY_PIPS)),
        "dual_mode_enabled": bool(v44_src.get("v5_dual_mode_enabled", False)),
        "trend_mode_efficiency_min": float(v44_src.get("v5_trend_mode_efficiency_min", 0.4)),
        "range_mode_efficiency_max": float(v44_src.get("v5_range_mode_efficiency_max", 0.3)),
        "range_fade_enabled": bool(v44_src.get("v5_range_fade_enabled", False)),
        "queued_confirm_bars": int(v44_src.get("v5_queued_confirm_bars", 0)),
        "london_confirm_bars": int(v44_src.get("v5_london_confirm_bars", 1)),
        "sizing_mode": str(v44_src.get("v5_sizing_mode", "risk_parity")).lower(),
        "base_lot": float(v44_src.get("v5_base_lot", 2.0)),
        "strong_size_mult": float(v44_src.get("v5_strong_size_mult", 3.0)),
        "normal_size_mult": float(v44_src.get("v5_normal_size_mult", 2.0)),
        "weak_size_mult": float(v44_src.get("v5_weak_size_mult", 0.0)),
        "win_bonus_per_step": float(v44_src.get("v5_win_bonus_per_step", 0.0)),
        "win_streak_scope": str(v44_src.get("v5_win_streak_scope", "session")),
        "max_lot": float(v44_src.get("v5_max_lot", 10.0)),
        "rp_strong_mult": float(v44_src.get("v5_rp_strong_mult", 1.0)),
        "rp_normal_mult": float(v44_src.get("v5_rp_normal_mult", 1.0)),
        "rp_weak_mult": float(v44_src.get("v5_rp_weak_mult", 0.0)),
        "rp_london_mult": float(v44_src.get("v5_rp_london_mult", 1.0)),
        "rp_ny_mult": float(v44_src.get("v5_rp_ny_mult", 1.0)),
        "rp_win_bonus_pct": float(v44_src.get("v5_rp_win_bonus_pct", 0.0)),
        "rp_max_lot_mult": float(v44_src.get("v5_rp_max_lot_mult", 2.0)),
        "hybrid_strong_boost": float(v44_src.get("v5_hybrid_strong_boost", 1.3)),
        "hybrid_normal_boost": float(v44_src.get("v5_hybrid_normal_boost", 0.8)),
        "hybrid_weak_boost": float(v44_src.get("v5_hybrid_weak_boost", 0.0)),
        "hybrid_ny_boost": float(v44_src.get("v5_hybrid_ny_boost", 1.0)),
        "h4_adx_min": float(v44_src.get("v5_h4_adx_min", 0.0)),
        "exhaustion_gate_enabled": bool(v44_src.get("v5_exhaustion_gate_enabled", False)),
        "exhaustion_gate_window_minutes": int(v44_src.get("v5_exhaustion_gate_window_minutes", 60)),
        "exhaustion_gate_max_range_pips": float(v44_src.get("v5_exhaustion_gate_max_range_pips", 40.0)),
        "exhaustion_gate_cooldown_minutes": int(v44_src.get("v5_exhaustion_gate_cooldown_minutes", 15)),
        "daily_loss_limit_pips": float(v44_src.get("v5_daily_loss_limit_pips", 0.0)),
        "daily_loss_limit_usd": float(ny_daily_loss_usd or 0.0),
        "weekly_loss_limit_pips": float(v44_src.get("v5_weekly_loss_limit_pips", 0.0)),
        "gl_enabled": bool(v44_src.get("v5_gl_enabled", False)),
        "gl_min_wins": int(v44_src.get("v5_gl_min_wins", 1)),
        "gl_extra_entries": int(v44_src.get("v5_gl_extra_entries", 2)),
        "gl_allow_normal": bool(v44_src.get("v5_gl_allow_normal", True)),
        "gl_normal_atr_cap": float(v44_src.get("v5_gl_normal_atr_cap", 0.67)),
        "gl_slope_relax": float(v44_src.get("v5_gl_slope_relax", 0.0)),
        "gl_max_open": int(v44_src.get("v5_gl_max_open", 0)),
        "ny_window_mode": ny_window_mode,
        "ny_start_hour": float(ny_start_raw) if ny_start_raw is not None else None,
        "ny_end_hour": float(ny_end_raw) if ny_end_raw is not None else None,
        "ny_duration_hours": float(v44_src.get("ny_duration_hours", 3.0)),
    }


def load_phase3_sizing_config(
    config_path: Optional[Path] = None,
    source_paths: Optional[dict[str, Path]] = None,
    preset_id: str | None = None,
) -> dict[str, Any]:
    """
    Load effective Phase 3 config with precedence:
      source-of-truth configs -> optional runtime override -> code defaults.
    """
    root = Path(__file__).resolve().parent.parent
    src_paths = {
        "v14": root / "research_out" / "tokyo_optimized_v14_config.json",
        "london_v2": root / "research_out" / "v2_exp4_winner_baseline_config.json",
        "v44_ny": root / "research_out" / "session_momentum_v44_base_config.json",
    }
    if isinstance(source_paths, dict):
        for k in ("v14", "london_v2", "v44_ny"):
            p = source_paths.get(k)
            if isinstance(p, Path):
                src_paths[k] = p

    v14_src = _read_json(src_paths["v14"])
    ldn_src = _read_json(src_paths["london_v2"])
    v44_src = _read_json(src_paths["v44_ny"])

    normalized = {
        "v14": _normalize_v14_source(v14_src),
        "london_v2": _normalize_london_source(ldn_src),
        "v44_ny": _normalize_v44_source(v44_src),
    }

    if config_path is None:
        config_path = root / "research_out" / "phase3_integrated_sizing_config.json"
    overrides = _read_json(config_path) if config_path.exists() else {}
    effective = dict(normalized)
    canonical_spec = None
    try:
        from core.phase3_package_spec import load_phase3_package_spec

        canonical_spec = load_phase3_package_spec(preset_id=preset_id)
        if canonical_spec.runtime_overrides:
            effective = _deep_merge(effective, canonical_spec.runtime_overrides)
    except Exception:
        pass
    effective = _deep_merge(effective, overrides if isinstance(overrides, dict) else {})

    # Defended preset: lock contract-critical keys after all file-based merges.
    locked_keys: list[str] = []
    try:
        effective, locked_keys = apply_defended_locked_keys(
            effective,
            preset_id=preset_id,
            canonical_spec=canonical_spec,
            normalized_cfg=normalized,
        )
    except Exception:
        pass

    hash_input = json.dumps(effective, sort_keys=True, separators=(",", ":"))
    effective_hash = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()[:16]
    effective["_meta"] = {
        "effective_hash": effective_hash,
        "source_paths": {k: str(v) for k, v in src_paths.items()},
        "override_path": str(config_path),
        "loaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "locked_keys": locked_keys,
    }
    return effective


# ---------------------------------------------------------------------------
# Session constants (single source of truth; aligned with backtest)
# ---------------------------------------------------------------------------
# London V2: UK-DST aware (07:00-11:00 UTC summer, 08:00-12:00 UTC winter)
#   ARB: London open -> +90m
#   Setup D: per config minutes after London open
# V44 NY: US-DST aware by default (12:00-15:00 UTC summer, 13:00-16:00 UTC winter)
#   plus configurable start delay (default +5m)
# Tokyo V14: fixed 16:00-22:00 UTC
TOKYO_START_UTC = 16  # 16:00 UTC
TOKYO_END_UTC = 22    # 22:00 UTC
TOKYO_ALLOWED_DAYS = {1, 2, 4}  # Tuesday=1, Wednesday=2, Friday=4  (weekday())

LONDON_START_UTC = 8
LONDON_END_UTC = 12
LONDON_ALLOWED_DAYS = {1, 2}  # Tue/Wed

NY_START_UTC = 13
NY_END_UTC = 16
NY_ALLOWED_DAYS = {0, 1, 2, 3, 4}  # Mon-Fri

# ---------------------------------------------------------------------------
# Indicator constants
# ---------------------------------------------------------------------------
BB_TF = "M5"
BB_PERIOD = 25
BB_STD = 2.2

BB_WIDTH_LOOKBACK = 100
BB_WIDTH_RANGING_PCT = 0.80  # below 80th percentile = ranging

PSAR_AF_START = 0.02
PSAR_AF_STEP = 0.02
PSAR_AF_MAX = 0.20
PSAR_FLIP_LOOKBACK = 12  # M1 bars

RSI_TF = "M5"
RSI_PERIOD = 14
RSI_LONG_ENTRY = 35
RSI_SHORT_ENTRY = 65

ATR_TF = "M15"
ATR_PERIOD = 14
ATR_MAX = 0.30  # price units (= 30 pips for USDJPY)

ADX_TF = "M15"
ADX_PERIOD = 14
ADX_MAX = 35

# ---------------------------------------------------------------------------
# Entry constants
# ---------------------------------------------------------------------------
MIN_CONFLUENCE = 2
ZONE_TOLERANCE_PIPS = 20
BLOCKED_COMBOS = {"ABCD"}

# ---------------------------------------------------------------------------
# SL / TP / Exit constants
# ---------------------------------------------------------------------------
SL_BUFFER_PIPS = 8
SL_MIN_PIPS = 12
SL_MAX_PIPS = 35

TP1_ATR_MULT = 0.5
TP1_MIN_PIPS = 6
TP1_MAX_PIPS = 12
TP1_CLOSE_PCT = 0.50  # 50 %

BE_OFFSET_PIPS = 2

TRAIL_ACTIVATE_PROFIT_PIPS = 8
TRAIL_DISTANCE_PIPS = 5
TRAIL_REQUIRES_TP1 = True
TRAIL_NEVER_WIDEN = True

TIME_DECAY_MINUTES = 120
TIME_DECAY_CAP_PIPS = 3.0

PRIMARY_TP_TARGET = "pivot_P"
MIN_TP_DISTANCE_PIPS = 8

# ---------------------------------------------------------------------------
# Sizing / risk
# ---------------------------------------------------------------------------
RISK_PCT = 0.02  # 2 %
MAX_UNITS = 500_000
MAX_CONCURRENT = 3
LEVERAGE = 33.3

# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------
MAX_TRADES_PER_SESSION = 4
COOLDOWN_MINUTES = 15
STOP_AFTER_CONSECUTIVE_LOSSES = 3
SESSION_LOSS_STOP_PCT = 0.015  # 1.5 %
MAX_ENTRY_SPREAD_PIPS = 3.0

PIP_SIZE = 0.01  # USDJPY

# ---------------------------------------------------------------------------
# London V2 constants (ARB + LMP)
# ---------------------------------------------------------------------------
LDN_ARB_BREAKOUT_BUFFER_PIPS = 7.0
LDN_ARB_SL_BUFFER_PIPS = 3.0
LDN_ARB_RANGE_MIN_PIPS = 30.0
LDN_ARB_RANGE_MAX_PIPS = 60.0
LDN_ARB_SL_MIN_PIPS = 15.0
LDN_ARB_SL_MAX_PIPS = 40.0
LDN_ARB_TP1_R = 1.0
LDN_ARB_TP2_R = 2.0
LDN_ARB_TP1_CLOSE_PCT = 0.5
LDN_ARB_BE_OFFSET_PIPS = 1.0
LDN_ARB_MAX_TRADES = 1

LDN_LMP_IMPULSE_MINUTES = 90
LDN_LMP_IMPULSE_MIN_PIPS = 20.0
LDN_LMP_ZONE_FIB = 0.5
LDN_LMP_SL_BUFFER_PIPS = 5.0
LDN_LMP_SL_MIN_PIPS = 12.0
LDN_LMP_SL_MAX_PIPS = 30.0
LDN_LMP_TP2_EXTENSION = 0.618
LDN_LMP_TP1_CLOSE_PCT = 0.5
LDN_LMP_BE_OFFSET_PIPS = 1.0
LDN_LMP_EMA_M15_PERIOD = 20
LDN_LMP_MAX_TRADES = 1

LDN_RISK_PCT = 0.0075
LDN_MAX_OPEN = 2
LDN_MAX_SPREAD_PIPS = 3.5
LDN_FORCE_CLOSE_AT_NY_OPEN = True

# London V2 Setup D (LOR breakout, long-only in exp4 baseline winner)
LDN_D_LOR_MIN_PIPS = 4.0
LDN_D_LOR_MAX_PIPS = 20.0
LDN_D_BREAKOUT_BUFFER_PIPS = 3.0
LDN_D_SL_BUFFER_PIPS = 3.0
LDN_D_SL_MIN_PIPS = 5.0
LDN_D_SL_MAX_PIPS = 20.0
LDN_D_TP1_R = 1.0
LDN_D_TP2_R = 2.0
LDN_D_TP1_CLOSE_PCT = 0.5
LDN_D_BE_OFFSET_PIPS = 1.0
LDN_D_MAX_TRADES = 1
LDN_MAX_TRADES_PER_DAY_TOTAL = 1

# ---------------------------------------------------------------------------
# V44 NY constants
# ---------------------------------------------------------------------------
V44_H1_EMA_FAST = 20
V44_H1_EMA_SLOW = 50

# Unused for Phase 3 session routing (we use LONDON_* / NY_* above)
# V44_LONDON_START = 8.5
# V44_LONDON_END = 11.0
V44_NY_START = 13.0
V44_NY_END = 16.0

V44_M5_EMA_FAST = 9
V44_M5_EMA_SLOW = 21
V44_SLOPE_BARS = 4
V44_STRONG_SLOPE = 0.5
V44_WEAK_SLOPE = 0.2
V44_LONDON_STRONG_SLOPE = 0.6
V44_MIN_BODY_PIPS = 1.5

V44_SL_LOOKBACK = 6
V44_SL_BUFFER_PIPS = 1.5
V44_SL_FLOOR_PIPS = 7.0
V44_SL_CAP_PIPS = 9.0

V44_STRONG_TP1_PIPS = 2.0
V44_STRONG_TP2_PIPS = 5.0
V44_STRONG_TP1_CLOSE_PCT = 0.3
V44_STRONG_TRAIL_BUFFER = 4.0
V44_LONDON_TP1_PIPS = 1.2
V44_LONDON_TRAIL_BUFFER = 2.0

V44_RISK_PCT = 0.005
V44_MAX_OPEN = 3
V44_MAX_ENTRIES_DAY = 7
V44_COOLDOWN_WIN = 1
V44_COOLDOWN_LOSS = 1
V44_SESSION_STOP_LOSSES = 3
V44_BE_OFFSET_PIPS = 0.5
V44_MAX_ENTRY_SPREAD = 3.0

V44_ATR_PCT_CAP = 0.67
V44_ATR_PCT_LOOKBACK = 200


# ===================================================================
#  Session classifier
# ===================================================================

def classify_session(now_utc: datetime, effective_config: Optional[dict[str, Any]] = None) -> Optional[str]:
    """Return 'tokyo', 'london', 'ny', or None using shared session core."""
    return classify_phase3_session(now_utc, effective_config)



# ===================================================================
#  BB Width Regime
# ===================================================================

def compute_bb_width_regime(
    m5_df: pd.DataFrame,
    *,
    mode: str = "percentile",
    lookback: int = BB_WIDTH_LOOKBACK,
    ranging_pct: float = BB_WIDTH_RANGING_PCT,
) -> str:
    """Return 'ranging' or 'trending' using the configured backtest regime mode."""
    close = m5_df["close"].astype(float)
    sma = close.rolling(BB_PERIOD).mean()
    std = close.rolling(BB_PERIOD, min_periods=BB_PERIOD).std(ddof=0)
    upper = sma + BB_STD * std
    lower = sma - BB_STD * std
    width = (upper - lower) / sma
    mode_norm = str(mode).strip().lower()
    if mode_norm in {"expanding3", "bb_width_expanding3"}:
        valid = width.dropna()
        if len(valid) < 4:
            return "ranging"
        expanding3 = bool(valid.iloc[-2] > valid.iloc[-3] and valid.iloc[-3] > valid.iloc[-4])
        return "trending" if expanding3 else "ranging"
    lookback_n = max(1, int(lookback))
    if len(width.dropna()) < lookback_n:
        return "ranging"
    recent = width.iloc[-lookback_n:]
    cutoff = recent.quantile(float(ranging_pct))
    current = width.iloc[-1]
    if pd.isna(current) or pd.isna(cutoff):
        return "ranging"
    return "trending" if current >= cutoff else "ranging"


_compute_bb = partial(_compute_bb_impl, period=BB_PERIOD, std=BB_STD)
_compute_rsi = partial(_compute_rsi_impl, period=RSI_PERIOD)
_compute_atr = partial(_compute_atr_impl, period=ATR_PERIOD)
_compute_adx = partial(_compute_adx_impl, period=ADX_PERIOD)
compute_parabolic_sar = partial(
    compute_parabolic_sar_impl,
    af_start=PSAR_AF_START,
    af_step=PSAR_AF_STEP,
    af_max=PSAR_AF_MAX,
)
detect_sar_flip = partial(
    detect_sar_flip_impl,
    lookback=PSAR_FLIP_LOOKBACK,
    af_start=PSAR_AF_START,
    af_step=PSAR_AF_STEP,
    af_max=PSAR_AF_MAX,
)
_drop_incomplete_tf = _drop_incomplete_tf_impl
compute_asian_range = partial(
    compute_asian_range_impl,
    pip_size=PIP_SIZE,
    range_min_pips=LDN_ARB_RANGE_MIN_PIPS,
    range_max_pips=LDN_ARB_RANGE_MAX_PIPS,
)
_compute_session_windows = partial(
    compute_session_windows_impl,
    london_open_fn=_shared_uk_london_open_utc,
    ny_open_fn=_shared_us_ny_open_utc,
    lmp_impulse_minutes=LDN_LMP_IMPULSE_MINUTES,
)


def _as_risk_fraction(value: Any, default_fraction: float) -> float:
    """Accept either percent-style (2.0) or fraction-style (0.02)."""
    try:
        v = float(value)
    except Exception:
        return float(default_fraction)
    if v <= 0:
        return 0.0
    return v / 100.0 if v > 1.0 else v

def last_sunday(year: int, month: int) -> pd.Timestamp:
    return _shared_last_sunday(year, month)


def nth_sunday(year: int, month: int, n: int) -> pd.Timestamp:
    return _shared_nth_sunday(year, month, n)


def uk_london_open_utc(ts_day: pd.Timestamp) -> int:
    return _shared_uk_london_open_utc(ts_day)


def us_ny_open_utc(ts_day: pd.Timestamp) -> int:
    return _shared_us_ny_open_utc(ts_day)


# ── Conservative V44 regime block ────────────────────────────────────
# Blocks V44 entries when the regime classifier labels the current bar
# as "breakout" or "post_breakout_trend".  Also blocks "ambiguous"
# only when momentum is not the top-scoring regime. This is the
# promoted Variant F router rule from the aligned Phase 3 backtests.
#
# Uses the same classifier constants as the backtest validation:
#   BB: period=25, std=2.2, width lookback=100
#   M5 slope: EMA(9), 4-bar slope in pips/bar
#   ADX: 14-period on M15 (resampled from M5 if M15 not available)
#   H1 alignment: side-aware (H1 EMA20 vs EMA50 agrees with M5 slope)
#
# Promoted from backtest Variant F (2026-03-26 aligned analysis).

_REGIME_BB_PERIOD = 25
_REGIME_BB_STD = 2.2
_REGIME_BB_WIDTH_LOOKBACK = 100
_REGIME_M5_EMA_FAST = 9
_REGIME_SLOPE_BARS = 4
_REGIME_H1_EMA_FAST = 20
_REGIME_H1_EMA_SLOW = 50
_REGIME_V44_BLOCK_LABELS = frozenset({"breakout", "post_breakout_trend"})
_LONDON_V2_OWNERSHIP_BLOCK_CELLS = frozenset(
    {
        ("breakout", "er_low", "der_neg"),
        ("momentum", "er_mid", "der_neg"),
        ("ambiguous", "er_high", "der_neg"),
    }
)


_compute_regime_snapshot = compute_regime_snapshot_shared
_v44_regime_block = v44_regime_block_shared
_london_v2_ownership_cluster_block = london_v2_ownership_cluster_block_shared


# ── Global regime stand-down (Variant I) ────────────────────────────
# Blocks ALL strategies when regime = post_breakout_trend AND ΔER < 0.
# Promoted from backtest Variant I (2026-03-27 ownership diagnostic).
# In the integrated system this primarily protects London V2 from entering
# during fading post-breakout conditions.  V44 is already blocked in
# post_breakout_trend by Variant F, so this adds cross-strategy coverage.

def compute_phase3_ownership_audit_for_data(
    data_by_tf: dict[str, Any],
    pip_size: float,
) -> dict[str, Any]:
    return compute_phase3_ownership_audit_for_data_shared(data_by_tf, pip_size)


_global_regime_standdown = global_regime_standdown_shared




def execute_london_v2_entry(
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
) -> dict:
    return execute_london_v2_session(
        adapter=adapter,
        profile=profile,
        policy=policy,
        data_by_tf=data_by_tf,
        tick=tick,
        phase3_state=phase3_state,
        sizing_config=sizing_config,
        now_utc=now_utc,
        store=store,
        ownership_audit=ownership_audit,
        overlay_state=overlay_state,
        support=build_phase3_session_support_from_mapping(globals()),
    )


def execute_v44_ny_entry(
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
) -> dict:
    return execute_v44_ny_session(
        adapter=adapter,
        profile=profile,
        policy=policy,
        data_by_tf=data_by_tf,
        tick=tick,
        phase3_state=phase3_state,
        sizing_config=sizing_config,
        now_utc=now_utc,
        store=store,
        ownership_audit=ownership_audit,
        overlay_state=overlay_state,
        support=build_phase3_session_support_from_mapping(globals()),
    )


# ===================================================================
#  Main execute function
# ===================================================================


def execute_tokyo_v14_entry(
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
) -> dict:
    return execute_tokyo_v14_session(
        adapter=adapter,
        profile=profile,
        policy=policy,
        data_by_tf=data_by_tf,
        tick=tick,
        mode=mode,
        phase3_state=phase3_state,
        store=store,
        sizing_config=sizing_config,
        now_utc=now_utc,
        ownership_audit=ownership_audit,
        overlay_state=overlay_state,
        base_state_updates=base_state_updates,
        support=build_phase3_session_support_from_mapping(globals()),
    )


def execute_phase3_integrated_policy_demo_only(
    *,
    adapter,
    profile,
    log_dir,
    policy,
    context,
    data_by_tf: dict,
    tick,
    mode: str,
    phase3_state: dict,
    store=None,
    sizing_config: Optional[dict[str, Any]] = None,
    is_new_m1: bool = True,
    ownership_audit: Optional[dict[str, Any]] = None,
    overlay_state: Optional[dict[str, Any]] = None,
) -> dict:
    """
    Main Phase 3 entry point.  Returns dict with:
      decision: ExecutionDecision-like dict
      phase3_state_updates: dict to merge into phase3_state
      strategy_tag: str | None  (e.g. "phase3:v14")
    """
    now_utc = datetime.now(timezone.utc)
    effective_cfg = sizing_config if isinstance(sizing_config, dict) and ("v14" in sizing_config or "london_v2" in sizing_config or "v44_ny" in sizing_config) else load_phase3_sizing_config()
    pip = float(profile.pip_size) if hasattr(profile, "pip_size") else PIP_SIZE
    effective_overlay_state = overlay_state if isinstance(overlay_state, dict) else build_phase3_overlay_state(effective_cfg)
    no_trade = {
        "decision": ExecutionDecision(attempted=False, placed=False, reason="", side=None),
        "phase3_state_updates": {},
        "strategy_tag": None,
    }

    route_plan = build_phase3_route_plan(
        data_by_tf=data_by_tf,
        effective_cfg=effective_cfg,
        phase3_state=phase3_state,
        is_new_m1=is_new_m1,
        ownership_audit=ownership_audit,
    )
    now_utc = route_plan["now_utc"]
    session = route_plan["session"]
    base_state_updates = dict(route_plan["base_state_updates"])
    no_trade["phase3_state_updates"] = dict(base_state_updates)

    if route_plan["blocked_reason"]:
        no_trade["decision"] = ExecutionDecision(
            attempted=bool(route_plan["blocked_attempted"]),
            placed=False,
            reason=str(route_plan["blocked_reason"]),
            side=None,
        )
        no_trade["phase3_state_updates"] = base_state_updates
        return no_trade

    if session == "london":
        ldn_res = execute_london_v2_entry(
            adapter=adapter,
            profile=profile,
            policy=policy,
            data_by_tf=data_by_tf,
            tick=tick,
            phase3_state=phase3_state,
            sizing_config=effective_cfg,
            now_utc=now_utc,
            store=store,
            ownership_audit=ownership_audit,
            overlay_state=effective_overlay_state,
        )
        merged = dict(base_state_updates)
        merged.update(ldn_res.get("phase3_state_updates") or {})
        ldn_res["phase3_state_updates"] = merged
        return ldn_res
    if session == "ny":
        ny_res = execute_v44_ny_entry(
            adapter=adapter,
            profile=profile,
            policy=policy,
            data_by_tf=data_by_tf,
            tick=tick,
            phase3_state=phase3_state,
            sizing_config=effective_cfg,
            now_utc=now_utc,
            store=store,
            ownership_audit=ownership_audit,
            overlay_state=effective_overlay_state,
        )
        merged = dict(base_state_updates)
        merged.update(ny_res.get("phase3_state_updates") or {})
        ny_res["phase3_state_updates"] = merged
        return ny_res
    if session != "tokyo":
        no_trade["decision"] = ExecutionDecision(
            attempted=False, placed=False,
            reason=f"phase3: no active session (current={session})", side=None,
        )
        no_trade["phase3_state_updates"] = base_state_updates
        return no_trade

    tokyo_res = execute_tokyo_v14_entry(
        adapter=adapter,
        profile=profile,
        policy=policy,
        data_by_tf=data_by_tf,
        tick=tick,
        mode=mode,
        phase3_state=phase3_state,
        store=store,
        sizing_config=effective_cfg,
        now_utc=now_utc,
        ownership_audit=ownership_audit,
        overlay_state=effective_overlay_state,
        base_state_updates=base_state_updates,
    )
    merged = dict(base_state_updates)
    merged.update(tokyo_res.get("phase3_state_updates") or {})
    tokyo_res["phase3_state_updates"] = merged
    return tokyo_res


# ===================================================================
#  Exit management
# ===================================================================

def _phase3_position_meta(position, side: str) -> tuple[Optional[int], float, int]:
    if isinstance(position, dict):
        position_id = position.get("id")
        current_units = abs(int(position.get("currentUnits") or 0))
        current_lots = current_units / 100_000.0
    else:
        position_id = getattr(position, "ticket", None)
        current_lots = float(getattr(position, "volume", 0) or 0)
        current_units = int(current_lots * 100_000)
    return position_id, current_lots, current_units


def _close_full(adapter, profile, position_id, current_lots: float, side: str) -> None:
    position_type = 1 if side == "sell" else 0
    adapter.close_position(
        ticket=position_id,
        symbol=profile.symbol,
        volume=current_lots,
        position_type=position_type,
    )


def _london_v2_entry_is_setup_d(entry_type: str) -> bool:
    """Setup D (L1) vs arb. entry_type is strategy_tag and may include @ownership_cell."""
    et = str(entry_type or "")
    if "london_v2_arb" in et:
        return False
    return et.endswith("london_v2_d") or "london_v2_d@" in et


def _manage_london_v2_exit(
    *,
    adapter,
    profile,
    store,
    tick,
    trade_row: dict,
    position,
    london_config: Optional[dict[str, Any]] = None,
) -> dict:
    pip = float(profile.pip_size) if hasattr(profile, "pip_size") else PIP_SIZE
    side = str(trade_row["side"]).lower()
    entry = float(trade_row["entry_price"])
    trade_id = str(trade_row["trade_id"])
    now_utc = datetime.now(timezone.utc)
    position_id, current_lots, _ = _phase3_position_meta(position, side)
    if current_lots <= 0:
        return {"action": "none", "reason": "no units"}

    day_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    ny_open_hour = us_ny_open_utc(pd.Timestamp(now_utc))
    ny_open = day_start + pd.Timedelta(hours=int(ny_open_hour))
    tp_check_price = tick.bid if side == "buy" else tick.ask
    cfg = london_config or {}
    entry_type = str(trade_row.get("entry_type") or "")
    is_setup_d = _london_v2_entry_is_setup_d(entry_type)
    force_close_at_ny_open = bool(cfg.get("hard_close_at_ny_open", LDN_FORCE_CLOSE_AT_NY_OPEN))
    tp1_r = float(cfg.get("d_tp1_r", LDN_D_TP1_R)) if is_setup_d else float(cfg.get("arb_tp1_r", LDN_ARB_TP1_R))
    tp2_r = float(cfg.get("d_tp2_r", LDN_D_TP2_R)) if is_setup_d else float(cfg.get("arb_tp2_r", LDN_ARB_TP2_R))
    tp1_close_pct = float(cfg.get("d_tp1_close_pct", LDN_D_TP1_CLOSE_PCT)) if is_setup_d else float(cfg.get("arb_tp1_close_pct", LDN_ARB_TP1_CLOSE_PCT))
    be_offset_pips = float(cfg.get("d_be_offset_pips", LDN_D_BE_OFFSET_PIPS)) if is_setup_d else float(cfg.get("arb_be_offset_pips", LDN_ARB_BE_OFFSET_PIPS))

    if force_close_at_ny_open and now_utc >= ny_open:
        try:
            _close_full(adapter, profile, position_id, current_lots, side)
            close_pips = ((tp_check_price - entry) / pip) if side == "buy" else ((entry - tp_check_price) / pip)
            return {"action": "session_end_close", "reason": "london_v2 hard close at NY open", "closed_pips_est": float(close_pips)}
        except Exception as e:
            return {"action": "error", "reason": f"london_v2 hard close error: {e}"}

    stop_price = trade_row.get("stop_price")
    if stop_price is None:
        return {"action": "none", "reason": "no stop in trade row"}
    stop_price = float(stop_price)
    r_pips = abs(entry - stop_price) / pip
    tp1_price = entry + (tp1_r * r_pips * pip if side == "buy" else -tp1_r * r_pips * pip)
    tp2_price = entry + (tp2_r * r_pips * pip if side == "buy" else -tp2_r * r_pips * pip)

    tp1_done = int(trade_row.get("tp1_partial_done") or 0) == 1
    reached_tp1 = tp_check_price >= tp1_price if side == "buy" else tp_check_price <= tp1_price
    reached_tp2 = tp_check_price >= tp2_price if side == "buy" else tp_check_price <= tp2_price

    if (not tp1_done) and reached_tp1:
        try:
            position_type = 1 if side == "sell" else 0
            close_lots = current_lots * tp1_close_pct
            adapter.close_position(ticket=position_id, symbol=profile.symbol, volume=close_lots, position_type=position_type)
            be_sl = entry + (be_offset_pips * pip if side == "buy" else -be_offset_pips * pip)
            adapter.update_position_stop_loss(position_id, profile.symbol, round(be_sl, 3))
            store.update_trade(trade_id, {"tp1_partial_done": 1, "breakeven_applied": 1, "breakeven_sl_price": round(be_sl, 5)})
            return {"action": "tp1_partial", "reason": f"london_v2 TP1 partial + BE ({be_sl:.3f})"}
        except Exception as e:
            return {"action": "error", "reason": f"london_v2 TP1 error: {e}"}

    if tp1_done and reached_tp2:
        try:
            _close_full(adapter, profile, position_id, current_lots, side)
            close_pips = float(tp2_r * r_pips)
            return {"action": "tp2_full", "reason": "london_v2 TP2 runner close", "closed_pips_est": close_pips}
        except Exception as e:
            return {"action": "error", "reason": f"london_v2 TP2 close error: {e}"}

    return {"action": "none", "reason": "no london_v2 exit condition met"}


def _manage_v44_exit(
    *,
    adapter,
    profile,
    store,
    tick,
    trade_row: dict,
    position,
    v44_config: Optional[dict[str, Any]] = None,
) -> dict:
    pip = float(profile.pip_size) if hasattr(profile, "pip_size") else PIP_SIZE
    side = str(trade_row["side"]).lower()
    entry = float(trade_row["entry_price"])
    trade_id = str(trade_row["trade_id"])
    now_utc = datetime.now(timezone.utc)
    position_id, current_lots, _ = _phase3_position_meta(position, side)
    if current_lots <= 0:
        return {"action": "none", "reason": "no units"}

    day_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    cfg = v44_config or {}
    _ny_start_hour, ny_end_hour = resolve_ny_window_hours(now_utc, cfg)
    ny_end = day_start + pd.Timedelta(hours=ny_end_hour)
    tp_check_price = tick.bid if side == "buy" else tick.ask
    trail_mark_price = tick.bid if side == "buy" else tick.ask

    if now_utc >= ny_end:
        try:
            _close_full(adapter, profile, position_id, current_lots, side)
            close_pips = ((tp_check_price - entry) / pip) if side == "buy" else ((entry - tp_check_price) / pip)
            return {"action": "session_end_close", "reason": "v44_ny hard close at session end", "closed_pips_est": float(close_pips)}
        except Exception as e:
            return {"action": "error", "reason": f"v44_ny hard close error: {e}"}

    be_offset_pips = float(cfg.get("be_offset_pips", V44_BE_OFFSET_PIPS))
    strong_tp2_pips = float(cfg.get("strong_tp2_pips", V44_STRONG_TP2_PIPS))
    normal_tp2_pips = float(cfg.get("normal_tp2_pips", 3.0))
    weak_tp2_pips = float(cfg.get("weak_tp2_pips", 2.0))
    strong_tp1_close_pct = float(cfg.get("strong_tp1_close_pct", V44_STRONG_TP1_CLOSE_PCT))
    normal_tp1_close_pct = float(cfg.get("normal_tp1_close_pct", 0.5))
    weak_tp1_close_pct = float(cfg.get("weak_tp1_close_pct", 0.6))
    strong_trail_buffer = float(cfg.get("strong_trail_buffer", V44_STRONG_TRAIL_BUFFER))
    normal_trail_buffer = float(cfg.get("normal_trail_buffer", 3.0))
    weak_trail_buffer = float(cfg.get("weak_trail_buffer", 2.0))
    trail_start_after_tp1_mult = float(cfg.get("trail_start_after_tp1_mult", 0.5))

    tp1_done = int(trade_row.get("tp1_partial_done") or 0) == 1
    entry_type = str(trade_row.get("entry_type") or "")
    is_news = ":news" in entry_type
    stop_price = trade_row.get("stop_price")
    if ":weak" in entry_type:
        tp1_close_pct = weak_tp1_close_pct
    elif ":normal" in entry_type:
        tp1_close_pct = normal_tp1_close_pct
    else:
        tp1_close_pct = strong_tp1_close_pct
    tp1_price = trade_row.get("target_price")
    if tp1_price is None:
        tp1_pips = trade_row.get("managed_tp1_pips")
        if tp1_pips is None:
            try:
                sl_pips = abs(float(entry) - float(stop_price)) / pip if stop_price is not None else 7.0
            except Exception:
                sl_pips = 7.0
            if is_news:
                tp1_pips = float(cfg.get("news_trend_tp_rr", 1.5)) * float(sl_pips)
            elif ":weak" in entry_type:
                tp1_pips = float(cfg.get("weak_tp1_pips", 1.2)) * float(sl_pips)
            elif ":normal" in entry_type:
                tp1_pips = float(cfg.get("normal_tp1_pips", 1.75)) * float(sl_pips)
            else:
                tp1_pips = float(cfg.get("strong_tp1_pips", V44_STRONG_TP1_PIPS)) * float(sl_pips)
        tp1_price = entry + (float(tp1_pips) * pip if side == "buy" else -float(tp1_pips) * pip)
    tp1_price = float(tp1_price)
    reached_tp1 = tp_check_price >= tp1_price if side == "buy" else tp_check_price <= tp1_price

    if (not tp1_done) and reached_tp1:
        if is_news:
            try:
                _close_full(adapter, profile, position_id, current_lots, side)
                close_pips = ((tp1_price - entry) / pip) if side == "buy" else ((entry - tp1_price) / pip)
                return {"action": "tp1_full", "reason": "v44_ny news-trend TP1 full close", "closed_pips_est": float(close_pips)}
            except Exception as e:
                return {"action": "error", "reason": f"v44_ny news TP1 error: {e}"}
        try:
            position_type = 1 if side == "sell" else 0
            close_lots = current_lots * tp1_close_pct
            adapter.close_position(ticket=position_id, symbol=profile.symbol, volume=close_lots, position_type=position_type)
            be_sl = entry + (be_offset_pips * pip if side == "buy" else -be_offset_pips * pip)
            adapter.update_position_stop_loss(position_id, profile.symbol, round(be_sl, 3))
            store.update_trade(trade_id, {"tp1_partial_done": 1, "breakeven_applied": 1, "breakeven_sl_price": round(be_sl, 5)})
            return {"action": "tp1_partial", "reason": f"v44_ny TP1 partial + BE ({be_sl:.3f})"}
        except Exception as e:
            return {"action": "error", "reason": f"v44_ny TP1 error: {e}"}

    if tp1_done:
        if ":weak" in entry_type:
            tp2_pips = weak_tp2_pips
        elif ":normal" in entry_type:
            tp2_pips = normal_tp2_pips
        else:
            tp2_pips = strong_tp2_pips
        # TP2 runner target is measured beyond TP1 (not from entry) to avoid
        # immediate runner closes right after TP1 partial on fast ticks.
        tp1_pips = abs(float(tp1_price) - float(entry)) / pip
        tp2_total_pips = float(tp1_pips) + float(tp2_pips)
        tp2_price = entry + (tp2_total_pips * pip if side == "buy" else -tp2_total_pips * pip)
        reached_tp2 = tp_check_price >= tp2_price if side == "buy" else tp_check_price <= tp2_price
        if reached_tp2:
            try:
                _close_full(adapter, profile, position_id, current_lots, side)
                close_pips = float(tp2_total_pips)
                return {"action": "tp2_full", "reason": f"v44_ny TP2 hit (+{tp2_pips:.2f}p runner, total {tp2_total_pips:.2f}p)", "closed_pips_est": close_pips}
            except Exception as e:
                return {"action": "error", "reason": f"v44_ny TP2 error: {e}"}

        # Trailing stop fallback (only reaches here if TP2 not yet hit)
        stop_price = trade_row.get("stop_price")
        if stop_price is not None and trail_start_after_tp1_mult > 0:
            try:
                sl_pips = abs(float(entry) - float(stop_price)) / pip
                profit_pips = ((tp_check_price - entry) / pip) if side == "buy" else ((entry - tp_check_price) / pip)
                tp1_pips = abs(float(tp1_price) - float(entry)) / pip
                arm_pips = tp1_pips + trail_start_after_tp1_mult * sl_pips
                if profit_pips < arm_pips:
                    return {"action": "none", "reason": "v44_ny trail not armed yet"}
            except Exception:
                pass
        if ":normal" in entry_type:
            trail_buffer_pips = normal_trail_buffer
        elif ":weak" in entry_type:
            trail_buffer_pips = weak_trail_buffer
        else:
            trail_buffer_pips = strong_trail_buffer
        prev_trail = trade_row.get("breakeven_sl_price")
        prev_trail = float(prev_trail) if prev_trail is not None else None
        if side == "buy":
            new_trail = trail_mark_price - trail_buffer_pips * pip
            if prev_trail is not None:
                new_trail = max(new_trail, prev_trail)
            should_update = prev_trail is None or new_trail > prev_trail
        else:
            new_trail = trail_mark_price + trail_buffer_pips * pip
            if prev_trail is not None:
                new_trail = min(new_trail, prev_trail)
            should_update = prev_trail is None or new_trail < prev_trail
        if should_update:
            try:
                adapter.update_position_stop_loss(position_id, profile.symbol, round(new_trail, 3))
                store.update_trade(trade_id, {"breakeven_sl_price": round(new_trail, 5)})
                return {"action": "trail_update", "reason": f"v44_ny trail -> {new_trail:.3f}"}
            except Exception as e:
                return {"action": "error", "reason": f"v44_ny trail error: {e}"}

    return {"action": "none", "reason": "no v44_ny exit condition met"}

def manage_phase3_exit(
    *,
    adapter,
    profile,
    store,
    tick,
    trade_row: dict,
    position,
    data_by_tf: dict,
    phase3_state: dict,
    sizing_config: Optional[dict[str, Any]] = None,
) -> dict:
    """
    Exit management for Phase 3 trades.

    Routes by strategy tag in entry_type/comment:
    - phase3:v14*
    - phase3:london_v2*
    - phase3:v44*
    Returns dict with action taken info.
    """
    pip = float(profile.pip_size) if hasattr(profile, "pip_size") else PIP_SIZE
    entry = float(trade_row["entry_price"])
    side = str(trade_row["side"]).lower()
    trade_id = str(trade_row["trade_id"])
    entry_type = str(trade_row.get("entry_type", "") or "")
    now_utc = datetime.now(timezone.utc)
    exit_check_price = tick.bid if side == "buy" else tick.ask
    current_spread = tick.ask - tick.bid

    if entry_type.startswith("phase3:london_v2"):
        return _manage_london_v2_exit(
            adapter=adapter,
            profile=profile,
            store=store,
            tick=tick,
            trade_row=trade_row,
            position=position,
            london_config=(sizing_config or {}).get("london_v2", {}),
        )
    if entry_type.startswith("phase3:v44"):
        return _manage_v44_exit(
            adapter=adapter,
            profile=profile,
            store=store,
            tick=tick,
            trade_row=trade_row,
            position=position,
            v44_config=(sizing_config or {}).get("v44_ny", {}),
        )

    v14_cfg = (sizing_config or {}).get("v14", {})
    tp1_close_pct = float(v14_cfg.get("tp1_close_pct", TP1_CLOSE_PCT))
    be_offset_pips = float(v14_cfg.get("breakeven_offset_pips", BE_OFFSET_PIPS))
    trail_activate_profit_pips = float(v14_cfg.get("trail_activate_profit_pips", TRAIL_ACTIVATE_PROFIT_PIPS))
    trail_distance_pips = float(v14_cfg.get("trail_distance_pips", TRAIL_DISTANCE_PIPS))
    time_decay_minutes = int(v14_cfg.get("time_decay_minutes", TIME_DECAY_MINUTES))
    time_decay_profit_cap_pips = float(v14_cfg.get("time_decay_profit_cap_pips", TIME_DECAY_CAP_PIPS))
    min_tp_distance_pips = float(v14_cfg.get("min_tp_distance_pips", MIN_TP_DISTANCE_PIPS))

    if isinstance(position, dict):
        position_id = position.get("id")
        current_units = abs(int(position.get("currentUnits") or 0))
        current_lots = current_units / 100_000.0
    else:
        position_id = getattr(position, "ticket", None)
        current_lots = float(getattr(position, "volume", 0) or 0)
        current_units = int(current_lots * 100_000)

    if current_lots <= 0:
        return {"action": "none", "reason": "no units"}

    position_type = 1 if side == "sell" else 0

    # 1) Session end force close
    session = classify_session(now_utc, sizing_config if isinstance(sizing_config, dict) else None)
    if session != "tokyo":
        try:
            adapter.close_position(
                ticket=position_id,
                symbol=profile.symbol,
                volume=current_lots,
                position_type=position_type,
            )
            close_pips = ((exit_check_price - entry) / pip) if side == "buy" else ((entry - exit_check_price) / pip)
            return {"action": "session_end_close", "reason": f"session ended (now={session})", "closed_pips_est": float(close_pips)}
        except Exception as e:
            return {"action": "error", "reason": f"session end close error: {e}"}

    tp1_done = int(trade_row.get("tp1_partial_done") or 0)

    # 2) Time decay check (backtest parity: only after TP1 and only for small positive profit)
    opened_at = trade_row.get("opened_at") or trade_row.get("timestamp_utc")
    if tp1_done and opened_at is not None:
        try:
            if isinstance(opened_at, str):
                open_time = datetime.fromisoformat(opened_at)
                if open_time.tzinfo is None:
                    open_time = open_time.replace(tzinfo=timezone.utc)
            else:
                open_time = opened_at
            elapsed_min = (now_utc - open_time).total_seconds() / 60.0
            if elapsed_min >= time_decay_minutes:
                profit_pips = ((exit_check_price - entry) / pip) if side == "buy" else ((entry - exit_check_price) / pip)
                if 0.0 <= profit_pips < time_decay_profit_cap_pips:
                    try:
                        adapter.close_position(
                            ticket=position_id,
                            symbol=profile.symbol,
                            volume=current_lots,
                            position_type=position_type,
                        )
                        return {
                            "action": "time_decay_close",
                            "reason": f"time decay {elapsed_min:.0f}min, profit={profit_pips:.1f}p < {time_decay_profit_cap_pips}p",
                            "closed_pips_est": float(profit_pips),
                        }
                    except Exception as e:
                        return {"action": "error", "reason": f"time decay close error: {e}"}
        except Exception:
            pass

    # 3) Hard SL check (software SL in case broker SL didn't trigger)
    check_price = tick.bid if side == "buy" else tick.ask
    stop_price = trade_row.get("stop_price")
    if stop_price is not None:
        stop_price = float(stop_price)
        if side == "buy" and check_price <= stop_price:
            try:
                adapter.close_position(ticket=position_id, symbol=profile.symbol, volume=current_lots, position_type=position_type)
                close_pips = (stop_price - entry) / pip
                return {"action": "hard_sl", "reason": f"hard SL hit at {check_price:.3f}", "closed_pips_est": float(close_pips)}
            except Exception as e:
                return {"action": "error", "reason": f"hard SL close error: {e}"}
        elif side == "sell" and check_price >= stop_price:
            try:
                adapter.close_position(ticket=position_id, symbol=profile.symbol, volume=current_lots, position_type=position_type)
                close_pips = (entry - stop_price) / pip
                return {"action": "hard_sl", "reason": f"hard SL hit at {check_price:.3f}", "closed_pips_est": float(close_pips)}
            except Exception as e:
                return {"action": "error", "reason": f"hard SL close error: {e}"}

    # 4) TP1 partial close
    if not tp1_done:
        target_price = trade_row.get("target_price")
        if target_price is not None:
            target_price = float(target_price)
            reached_buy = exit_check_price >= target_price and side == "buy"
            reached_sell = exit_check_price <= target_price and side == "sell"
            if reached_buy or reached_sell:
                close_lots = current_lots * tp1_close_pct
                try:
                    adapter.close_position(
                        ticket=position_id,
                        symbol=profile.symbol,
                        volume=close_lots,
                        position_type=position_type,
                    )
                    # Move SL to BE + offset
                    be_offset = current_spread + be_offset_pips * pip
                    if side == "buy":
                        be_sl = entry + be_offset
                    else:
                        be_sl = entry - be_offset
                    adapter.update_position_stop_loss(position_id, profile.symbol, round(be_sl, 3))
                    store.update_trade(trade_id, {
                        "tp1_partial_done": 1,
                        "breakeven_applied": 1,
                        "breakeven_sl_price": round(be_sl, 5),
                    })
                    return {
                        "action": "tp1_partial",
                        "reason": f"TP1 {tp1_close_pct*100:.0f}% closed, BE SL={be_sl:.3f}",
                    }
                except Exception as e:
                    return {"action": "error", "reason": f"TP1 close error: {e}"}

    # 5) Trailing stop (after TP1)
    elif tp1_done:
        # TP2 runner target at daily pivot P (with minimum TP floor), then trail as fallback.
        piv = phase3_state.get("pivots") if isinstance(phase3_state, dict) else None
        pivot_p = None
        if isinstance(piv, dict):
            try:
                pivot_p = float(piv.get("P"))
            except Exception:
                pivot_p = None
        if pivot_p is not None and np.isfinite(pivot_p):
            min_tp = float(min_tp_distance_pips) * pip
            if side == "buy":
                tp2_price = max(pivot_p, entry + min_tp)
                reached_tp2 = exit_check_price >= tp2_price
            else:
                tp2_price = min(pivot_p, entry - min_tp)
                reached_tp2 = exit_check_price <= tp2_price
            if reached_tp2:
                try:
                    adapter.close_position(
                        ticket=position_id,
                        symbol=profile.symbol,
                        volume=current_lots,
                        position_type=position_type,
                    )
                    close_pips = ((tp2_price - entry) / pip) if side == "buy" else ((entry - tp2_price) / pip)
                    return {"action": "tp2_full", "reason": f"pivot TP2 hit at {tp2_price:.3f}", "closed_pips_est": float(close_pips)}
                except Exception as e:
                    return {"action": "error", "reason": f"TP2 close error: {e}"}

        profit_pips = ((exit_check_price - entry) / pip) if side == "buy" else ((entry - exit_check_price) / pip)
        if profit_pips >= trail_activate_profit_pips:
            prev_trail = trade_row.get("breakeven_sl_price")
            if prev_trail is not None:
                prev_trail = float(prev_trail)
            if side == "buy":
                new_trail = exit_check_price - trail_distance_pips * pip
                if prev_trail is not None and TRAIL_NEVER_WIDEN:
                    new_trail = max(new_trail, prev_trail)
            else:
                new_trail = exit_check_price + trail_distance_pips * pip
                if prev_trail is not None and TRAIL_NEVER_WIDEN:
                    new_trail = min(new_trail, prev_trail)
            should_update = False
            if side == "buy" and (prev_trail is None or new_trail > prev_trail):
                should_update = True
            elif side == "sell" and (prev_trail is None or new_trail < prev_trail):
                should_update = True
            if should_update:
                try:
                    adapter.update_position_stop_loss(position_id, profile.symbol, round(new_trail, 3))
                    store.update_trade(trade_id, {"breakeven_sl_price": round(new_trail, 5)})
                    return {
                        "action": "trail_update",
                        "reason": f"trail SL -> {new_trail:.3f} (profit={profit_pips:.1f}p)",
                    }
                except Exception as e:
                    return {"action": "error", "reason": f"trail update error: {e}"}

    return {"action": "none", "reason": "no exit condition met"}


# ===================================================================
#  Dashboard helpers
# ===================================================================

def report_phase3_session(now_utc: datetime | None = None, effective_config: Optional[dict[str, Any]] = None) -> dict:
    """Report current session classification."""
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    cfg = effective_config if isinstance(effective_config, dict) else load_phase3_sizing_config()
    session = classify_session(now_utc, cfg)
    return {
        "name": "Active Session",
        "value": session or "none",
        "ok": session is not None,
        "detail": f"UTC {now_utc.strftime('%H:%M')} | day={now_utc.strftime('%A')}",
    }


def report_phase3_strategy(session: str | None) -> dict:
    """Report active strategy for this session."""
    if session == "tokyo":
        return {"name": "Active Strategy", "value": "V14 Mean Reversion", "ok": True, "detail": "Tokyo session"}
    elif session == "london":
        return {"name": "Active Strategy", "value": "London V2 (A+D)", "ok": True, "detail": "London session"}
    elif session == "ny":
        return {"name": "Active Strategy", "value": "V44 NY Momentum", "ok": True, "detail": "NY session"}
    return {"name": "Active Strategy", "value": "none", "ok": False, "detail": "No session"}


def report_phase3_regime(regime: str) -> dict:
    """Report BB width regime."""
    return {
        "name": "V14 Regime",
        "value": regime,
        "ok": regime == "ranging",
        "detail": f"BB({BB_PERIOD}, {BB_STD}) width P{int(BB_WIDTH_RANGING_PCT*100)}",
    }


def report_phase3_adx(adx_val: float, max_adx: float | None = None) -> dict:
    """Report ADX status."""
    limit = float(ADX_MAX if max_adx is None else max_adx)
    return {
        "name": "V14 ADX",
        "value": f"{adx_val:.1f}",
        "ok": adx_val < limit,
        "detail": f"max {limit:.1f}",
    }


def report_phase3_atr(atr_val: float, atr_max: float | None = None) -> dict:
    """Report ATR status."""
    limit = float(ATR_MAX if atr_max is None else atr_max)
    atr_pips = atr_val / PIP_SIZE
    return {
        "name": "V14 ATR",
        "value": f"{atr_pips:.1f}p",
        "ok": atr_val < limit,
        "detail": f"max {limit/PIP_SIZE:.0f}p",
    }


def report_phase3_london_range(
    asian_pips: float,
    is_valid: bool,
    min_pips: float | None = None,
    max_pips: float | None = None,
) -> dict:
    min_limit = float(LDN_ARB_RANGE_MIN_PIPS if min_pips is None else min_pips)
    max_limit = float(LDN_ARB_RANGE_MAX_PIPS if max_pips is None else max_pips)
    return {
        "name": "London Asian Range",
        "value": f"{asian_pips:.1f}p",
        "ok": bool(is_valid),
        "detail": f"valid={min_limit:.0f}-{max_limit:.0f}p",
    }


def report_phase3_london_levels(asian_high: float, asian_low: float) -> dict:
    return {
        "name": "London ARB Levels",
        "value": f"H:{asian_high:.3f} / L:{asian_low:.3f}" if np.isfinite(asian_high) and np.isfinite(asian_low) else "n/a",
        "ok": np.isfinite(asian_high) and np.isfinite(asian_low),
        "detail": f"buffers ±{LDN_ARB_BREAKOUT_BUFFER_PIPS:.1f}p",
    }


def report_phase3_ny_trend(trend: str | None) -> dict:
    return {
        "name": "NY H1 Trend",
        "value": trend or "none",
        "ok": trend in {"up", "down"},
        "detail": f"EMA{V44_H1_EMA_FAST}/{V44_H1_EMA_SLOW}",
    }


def report_phase3_ny_slope(slope_pips_per_bar: float, threshold: float | None = None) -> dict:
    limit = float(V44_STRONG_SLOPE if threshold is None else threshold)
    return {
        "name": "NY M5 Slope",
        "value": f"{slope_pips_per_bar:.2f} p/bar",
        "ok": abs(slope_pips_per_bar) >= limit,
        "detail": f"strong>={limit:.2f}",
    }


def report_phase3_ny_atr_filter(ok: bool, cap: float | None = None, lookback: int | None = None) -> dict:
    cap_v = float(V44_ATR_PCT_CAP if cap is None else cap)
    lookback_v = int(V44_ATR_PCT_LOOKBACK if lookback is None else lookback)
    return {
        "name": "NY ATR PCT Filter",
        "value": "pass" if ok else "blocked",
        "ok": bool(ok),
        "detail": f"cap P{int(cap_v*100)} lookback={lookback_v}",
    }


def report_phase3_last_decision(eval_result: Optional[dict]) -> dict:
    """Report last decision reason so user sees why trade was/wasn't taken."""
    reason = ""
    if eval_result and isinstance(eval_result, dict):
        dec = eval_result.get("decision")
        if dec is not None and getattr(dec, "reason", None):
            reason = str(dec.reason)
    return {
        "name": "Last decision",
        "value": reason or "—",
        "ok": True,
        "detail": "Reason from this poll",
    }


def report_phase3_tokyo_caps(phase3_state: dict, now_utc: datetime) -> list[dict]:
    """Tokyo session caps: trade count, cooldown, consecutive losses, max concurrent."""
    reports = []
    today_str = now_utc.strftime("%Y-%m-%d")
    session_key = f"session_tokyo_{today_str}"
    session_data = phase3_state.get(session_key, {})
    trade_count = session_data.get("trade_count", 0)
    consecutive_losses = session_data.get("consecutive_losses", 0)
    last_entry = session_data.get("last_entry_time")
    open_count = int(phase3_state.get("open_trade_count", 0))

    reports.append({
        "name": "V14 Trades this session",
        "value": f"{trade_count}/{MAX_TRADES_PER_SESSION}",
        "ok": trade_count < MAX_TRADES_PER_SESSION,
        "detail": f"max {MAX_TRADES_PER_SESSION}",
    })
    reports.append({
        "name": "V14 Consecutive losses",
        "value": f"{consecutive_losses}/{STOP_AFTER_CONSECUTIVE_LOSSES}",
        "ok": consecutive_losses < STOP_AFTER_CONSECUTIVE_LOSSES,
        "detail": f"stop after {STOP_AFTER_CONSECUTIVE_LOSSES}",
    })
    reports.append({
        "name": "V14 Max concurrent",
        "value": f"{open_count}/{MAX_CONCURRENT}",
        "ok": open_count < MAX_CONCURRENT,
        "detail": "Phase 3 open positions",
    })
    cooldown_ok = True
    if last_entry:
        try:
            from datetime import datetime as dt
            t = dt.fromisoformat(last_entry)
            if t.tzinfo is None:
                import pandas as pd
                t = pd.Timestamp(t).tz_localize("UTC").to_pydatetime()
            elapsed = (now_utc - t).total_seconds() / 60.0
            cooldown_ok = elapsed >= COOLDOWN_MINUTES
        except Exception:
            pass
    reports.append({
        "name": "V14 Cooldown",
        "value": "clear" if cooldown_ok else "active",
        "ok": cooldown_ok,
        "detail": f"{COOLDOWN_MINUTES} min between entries",
    })
    return reports


def report_phase3_london_window(now_utc: datetime) -> dict:
    """Report which London sub-window we're in (A vs D)."""
    windows = _compute_session_windows(now_utc)
    fmt = lambda t: t.strftime("%H:%M")
    if now_utc < windows["london_open"]:
        return {"name": "London Window", "value": "pre-open", "ok": False, "detail": f"Before {fmt(windows['london_open'])} UTC"}
    if now_utc < windows["london_arb_end"]:
        return {
            "name": "London Window",
            "value": f"Setup A ({fmt(windows['london_open'])}–{fmt(windows['london_arb_end'])} UTC)",
            "ok": True,
            "detail": "Asian range breakout",
        }
    if now_utc < windows["london_end"]:
        d_start = windows["london_open"] + pd.Timedelta(minutes=15)
        d_end = windows["london_open"] + pd.Timedelta(minutes=120)
        return {
            "name": "London Window",
            "value": f"Setup D ({fmt(d_start)}–{fmt(d_end)} UTC active)",
            "ok": True,
            "detail": "LOR breakout",
        }
    return {"name": "London Window", "value": "closed", "ok": False, "detail": f"After {fmt(windows['london_end'])} UTC"}


def report_phase3_london_caps(phase3_state: dict) -> list[dict]:
    """London session caps: max open, A/D trade counts."""
    reports = []
    today = datetime.now(timezone.utc).date().isoformat()
    session_key = f"session_london_{today}"
    sdat = phase3_state.get(session_key, {})
    arb = int(sdat.get("arb_trades", 0))
    d_trades = int(sdat.get("d_trades", 0))
    total_trades = int(sdat.get("total_trades", 0))
    open_count = int(phase3_state.get("open_trade_count", 0))

    reports.append({
        "name": "London Max open",
        "value": f"{open_count}/{LDN_MAX_OPEN}",
        "ok": open_count < LDN_MAX_OPEN,
        "detail": "Phase 3 open positions",
    })
    reports.append({
        "name": "London ARB trades",
        "value": f"{arb}/{LDN_ARB_MAX_TRADES}",
        "ok": arb < LDN_ARB_MAX_TRADES,
        "detail": "Asian range breakout",
    })
    reports.append({
        "name": "London D trades",
        "value": f"{d_trades}/{LDN_D_MAX_TRADES}",
        "ok": d_trades < LDN_D_MAX_TRADES,
        "detail": "LOR breakout (long-only)",
    })
    reports.append({
        "name": "London total trades",
        "value": f"{total_trades}/{LDN_MAX_TRADES_PER_DAY_TOTAL}",
        "ok": total_trades < LDN_MAX_TRADES_PER_DAY_TOTAL,
        "detail": "Daily cap",
    })
    return reports


def report_phase3_ny_caps(phase3_state: dict, now_utc: datetime, v44_cfg: Optional[dict[str, Any]] = None) -> list[dict]:
    """NY session caps: max open, session stop losses, cooldown."""
    reports = []
    today = now_utc.date().isoformat()
    session_key = f"session_ny_{today}"
    sdat = phase3_state.get(session_key, {})
    trade_count = int(sdat.get("trade_count", 0))
    consecutive_losses = int(sdat.get("consecutive_losses", 0))
    open_count = int(phase3_state.get("open_trade_count", 0))
    cfg = v44_cfg or {}
    raw_max_open = cfg.get("max_open_positions", V44_MAX_OPEN)
    max_open = 0 if raw_max_open is None else int(raw_max_open)
    session_stop_losses = int(cfg.get("session_stop_losses", V44_SESSION_STOP_LOSSES))

    reports.append({
        "name": "NY Max open",
        "value": f"{open_count}/{max_open if max_open > 0 else 'unlimited'}",
        "ok": True if max_open <= 0 else open_count < max_open,
        "detail": "Phase 3 open positions",
    })
    reports.append({
        "name": "NY Session stop losses",
        "value": f"{consecutive_losses}/{session_stop_losses}",
        "ok": consecutive_losses < session_stop_losses,
        "detail": f"stop after {session_stop_losses}",
    })
    return reports
