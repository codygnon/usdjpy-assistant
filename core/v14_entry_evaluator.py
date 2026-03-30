"""
Extracted V14 (Tokyo mean-reversion) entry signal evaluator.

This module contains the exact entry evaluation logic from
``scripts/backtest_tokyo_meanrev.py``, extracted into callable functions
so that both the native backtest loop and the shadow invocation layer
can use the same source of truth.

**No thresholds or conditions were changed during extraction.**
The native backtest loop calls ``evaluate_v14_entry_signal()`` and
receives the same candidate dict it previously built inline.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

PIP_SIZE = 0.01


# ---------------------------------------------------------------------------
# Regime gate scoring (V14-native)
# ---------------------------------------------------------------------------
def _score_regime_gate(
    *,
    row: Any,
    mid_close: float,
    R1: float,
    R2: float,
    S1: float,
    S2: float,
    regime_enabled: bool,
    atr_ratio_trend: float,
    atr_ratio_calm: float,
    adx_trend: float,
    adx_range: float,
    favorable_min_score: int,
    neutral_min_score: int,
    neutral_size_mult: float,
) -> tuple[str, float, bool]:
    """
    Returns (regime_label, regime_size_mult, blocked).
    blocked=True means the bar should be skipped.
    """
    if not regime_enabled:
        return "neutral", 1.0, False

    score = 0
    atr_fast = float(row.get("atr_m15", np.nan))
    atr_slow = float(row.get("atr_m15_slow", np.nan))
    atr_ratio = (atr_fast / atr_slow) if (not pd.isna(atr_fast) and not pd.isna(atr_slow) and atr_slow > 0) else np.nan
    if not pd.isna(atr_ratio):
        if atr_ratio > atr_ratio_trend:
            score -= 1
        elif atr_ratio < atr_ratio_calm:
            score += 1
    bbw = float(row.get("bb_width", np.nan))
    bbw_hi = float(row.get("bb_width_q_high", np.nan))
    bbw_lo = float(row.get("bb_width_q_low", np.nan))
    if not pd.isna(bbw) and not pd.isna(bbw_hi) and bbw > bbw_hi:
        score -= 1
    elif not pd.isna(bbw) and not pd.isna(bbw_lo) and bbw < bbw_lo:
        score += 1
    adx = float(row.get("adx_m15", np.nan))
    if not pd.isna(adx):
        if adx > adx_trend:
            score -= 1
        elif adx < adx_range:
            score += 1
    if S1 <= mid_close <= R1:
        score += 1
    if (mid_close <= S2) or (mid_close >= R2):
        score -= 1
    if score >= favorable_min_score:
        return "favorable", 1.0, False
    elif score == neutral_min_score:
        return "neutral", neutral_size_mult, False
    else:
        return "unfavorable", 0.0, True


# ---------------------------------------------------------------------------
# Confluence scoring
# ---------------------------------------------------------------------------
def _score_confluence_v2(
    *,
    cond_long_zone: bool,
    cond_long_s2: bool,
    cond_long_bb: bool,
    cond_long_sar: bool,
    cond_long_rsi_soft: bool,
    cond_long_rsi_ext: bool,
    cond_atr_low: bool,
    cond_short_zone: bool,
    cond_short_r2: bool,
    cond_short_bb: bool,
    cond_short_sar: bool,
    cond_short_rsi_soft: bool,
    cond_short_rsi_ext: bool,
    confluence_min_long: int,
    confluence_min_short: int,
) -> tuple[bool, bool, int, int, dict, dict]:
    """Tokyo V2 scoring model. Returns (long_signal, short_signal, long_score, short_score, long_conditions, short_conditions)."""
    long_conditions = {
        "pivot_S1": bool(cond_long_zone),
        "pivot_S2": bool(cond_long_s2),
        "bb_lower": bool(cond_long_bb),
        "sar_flip": bool(cond_long_sar),
        "rsi_oversold": bool(cond_long_rsi_soft),
        "rsi_extreme": bool(cond_long_rsi_ext),
        "atr_low": bool(cond_atr_low),
    }
    short_conditions = {
        "pivot_R1": bool(cond_short_zone),
        "pivot_R2": bool(cond_short_r2),
        "bb_upper": bool(cond_short_bb),
        "sar_flip": bool(cond_short_sar),
        "rsi_overbought": bool(cond_short_rsi_soft),
        "rsi_extreme": bool(cond_short_rsi_ext),
        "atr_low": bool(cond_atr_low),
    }
    long_score = int(sum(1 for v in long_conditions.values() if v))
    short_score = int(sum(1 for v in short_conditions.values() if v))
    long_signal = long_score >= confluence_min_long
    short_signal = short_score >= confluence_min_short
    return long_signal, short_signal, long_score, short_score, long_conditions, short_conditions


def _score_confluence_legacy(
    *,
    cond_long_zone: bool,
    cond_long_bb: bool,
    cond_long_sar: bool,
    cond_long_rsi_ext: bool,
    cond_long_s2: bool,
    cond_short_zone: bool,
    cond_short_bb: bool,
    cond_short_sar: bool,
    cond_short_rsi_ext: bool,
    cond_short_r2: bool,
    cond_long_rsi_soft: bool,
    cond_short_rsi_soft: bool,
    core_gate_use_zone: bool,
    core_gate_use_bb: bool,
    core_gate_use_sar: bool,
    core_gate_use_rsi: bool,
    core_gate_required: int,
    confluence_min_long: int,
    confluence_min_short: int,
) -> tuple[bool, bool, int, int]:
    """Legacy core-gate scoring model. Returns (long_signal, short_signal, long_score, short_score)."""
    long_core_flags = []
    if core_gate_use_zone:
        long_core_flags.append(bool(cond_long_zone))
    if core_gate_use_bb:
        long_core_flags.append(bool(cond_long_bb))
    if core_gate_use_sar:
        long_core_flags.append(bool(cond_long_sar))
    if core_gate_use_rsi:
        long_core_flags.append(bool(cond_long_rsi_soft))
    long_core_ok = (sum(1 for x in long_core_flags if x) >= max(1, min(core_gate_required, len(long_core_flags)))) if long_core_flags else True

    long_signal = False
    long_score = 0
    if long_core_ok:
        long_score += 1 if cond_long_zone else 0
        long_score += 1 if cond_long_bb else 0
        long_score += 1 if cond_long_sar else 0
        long_score += 1 if cond_long_rsi_ext else 0
        long_score += 1 if cond_long_s2 else 0
        long_signal = long_score >= confluence_min_long

    short_core_flags = []
    if core_gate_use_zone:
        short_core_flags.append(bool(cond_short_zone))
    if core_gate_use_bb:
        short_core_flags.append(bool(cond_short_bb))
    if core_gate_use_sar:
        short_core_flags.append(bool(cond_short_sar))
    if core_gate_use_rsi:
        short_core_flags.append(bool(cond_short_rsi_soft))
    short_core_ok = (sum(1 for x in short_core_flags if x) >= max(1, min(core_gate_required, len(short_core_flags)))) if short_core_flags else True

    short_signal = False
    short_score = 0
    if short_core_ok:
        short_score += 1 if cond_short_zone else 0
        short_score += 1 if cond_short_bb else 0
        short_score += 1 if cond_short_sar else 0
        short_score += 1 if cond_short_rsi_ext else 0
        short_score += 1 if cond_short_r2 else 0
        short_signal = short_score >= confluence_min_short

    return long_signal, short_signal, long_score, short_score


# ---------------------------------------------------------------------------
# Signal strength scoring
# ---------------------------------------------------------------------------
def _compute_signal_strength(
    *,
    direction: str,
    row: Any,
    rsi: float,
    bb_l: float,
    bb_u: float,
    mid_close: float,
    E: bool,
    long_score: int,
    short_score: int,
    ss_enabled: bool,
    ss_comp: dict,
) -> tuple[int, str]:
    """Returns (signal_strength_score, signal_strength_tier)."""
    if not ss_enabled:
        return 0, "weak"

    signal_strength_score = 0
    cc_map = ss_comp.get("confluence_count", {"2": 1, "3": 2, "4": 3, "5": 4})
    cscore = int(long_score if direction == "long" else short_score)
    signal_strength_score += int(cc_map.get(str(cscore), 0))
    bb_pen = ((bb_l - mid_close) / PIP_SIZE) if direction == "long" else ((mid_close - bb_u) / PIP_SIZE)
    if float(bb_pen) > float(ss_comp.get("bb_penetration_bonus_pips", 2)):
        signal_strength_score += 1
    rsi_ext = float(ss_comp.get("rsi_extreme_bonus_threshold", 25))
    if (direction == "long" and rsi < rsi_ext) or (direction == "short" and rsi > (100.0 - rsi_ext)):
        signal_strength_score += 1
    if E:
        signal_strength_score += 1
    same_flip = bool(row.get("sar_flip_bullish", False)) if direction == "long" else bool(row.get("sar_flip_bearish", False))
    if same_flip:
        signal_strength_score += 1
    fav_hours = set(int(h) for h in ss_comp.get("favorable_hour", [17, 18, 21]))
    if int(row["hour_utc"]) in fav_hours:
        signal_strength_score += 1

    if signal_strength_score <= 3:
        signal_strength_tier = "weak"
    elif signal_strength_score <= 6:
        signal_strength_tier = "moderate"
    else:
        signal_strength_tier = "strong"
    return signal_strength_score, signal_strength_tier


# ---------------------------------------------------------------------------
# Quality markers
# ---------------------------------------------------------------------------
def _compute_quality_markers(
    *,
    direction: str,
    row: Any,
    mid_close: float,
    rejection_bonus_enabled: bool,
    div_track_enabled: bool,
    session_env_enabled: bool,
    session_env_log_ir_pos: bool,
    sst: dict,
) -> dict:
    """
    Returns dict with: rejection_confirmed, rejection_low, rejection_high,
    rejection_wick_ratio, divergence_present, session_midpoint, inside_ir,
    dist_ir_boundary, dist_to_midpoint, quality_markers.
    """
    rejection_confirmed = False
    rejection_low = np.nan
    rejection_high = np.nan
    rejection_wick_ratio = np.nan
    if rejection_bonus_enabled:
        if direction == "long":
            rejection_confirmed = bool(row.get("rej_bull_recent", False))
            rejection_low = float(row.get("rej_bull_low_recent", np.nan))
            rejection_wick_ratio = float(row.get("rej_wick_ratio_bull", np.nan))
        else:
            rejection_confirmed = bool(row.get("rej_bear_recent", False))
            rejection_high = float(row.get("rej_bear_high_recent", np.nan))
            rejection_wick_ratio = float(row.get("rej_wick_ratio_bear", np.nan))

    divergence_present = False
    if div_track_enabled:
        divergence_present = bool(row.get("rsi_div_bull_recent", False)) if direction == "long" else bool(row.get("rsi_div_bear_recent", False))

    session_midpoint = np.nan
    inside_ir = False
    dist_ir_boundary = np.nan
    if session_env_enabled:
        s_hi = float(sst.get("session_high", np.nan))
        s_lo = float(sst.get("session_low", np.nan))
        if np.isfinite(s_hi) and np.isfinite(s_lo):
            session_midpoint = (s_hi + s_lo) / 2.0
        if session_env_log_ir_pos and bool(sst.get("ir_ready", False)):
            ir_hi = float(sst.get("ir_high", np.nan))
            ir_lo = float(sst.get("ir_low", np.nan))
            if np.isfinite(ir_hi) and np.isfinite(ir_lo):
                inside_ir = bool(ir_lo <= mid_close <= ir_hi)
                dist_ir_boundary = min(abs(mid_close - ir_lo), abs(mid_close - ir_hi)) / PIP_SIZE

    dist_to_midpoint = abs(mid_close - session_midpoint) / PIP_SIZE if np.isfinite(session_midpoint) else np.nan

    marker_list = []
    if rejection_confirmed:
        marker_list.append("rejection")
    if divergence_present:
        marker_list.append("divergence")
    if inside_ir:
        marker_list.append("inside_ir")

    return {
        "rejection_confirmed": bool(rejection_confirmed),
        "rejection_low": float(rejection_low) if np.isfinite(rejection_low) else np.nan,
        "rejection_high": float(rejection_high) if np.isfinite(rejection_high) else np.nan,
        "rejection_wick_ratio": float(rejection_wick_ratio) if np.isfinite(rejection_wick_ratio) else np.nan,
        "divergence_present": bool(divergence_present),
        "session_midpoint": float(session_midpoint) if np.isfinite(session_midpoint) else np.nan,
        "inside_ir": bool(inside_ir),
        "dist_ir_boundary": float(dist_ir_boundary) if np.isfinite(dist_ir_boundary) else np.nan,
        "dist_to_midpoint": float(dist_to_midpoint) if np.isfinite(dist_to_midpoint) else np.nan,
        "quality_markers": ",".join(marker_list),
    }


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------
def evaluate_v14_entry_signal(
    *,
    row: Any,
    mid_close: float,
    mid_open: float,
    mid_high: float,
    mid_low: float,
    pivot_levels: dict,
    cfg_params: dict,
    sst: dict,
) -> dict | None:
    """
    Evaluate a V14 entry signal on a single bar.

    Parameters
    ----------
    row : pandas Series or dict-like
        The current M1 bar with indicator columns attached.
    mid_close, mid_open, mid_high, mid_low : float
        Mid-price OHLC for the bar.
    pivot_levels : dict
        Must contain keys: P, R1, R2, R3, S1, S2, S3.
    cfg_params : dict
        Pre-extracted config parameters. Required keys:
        - tokyo_v2_scoring, confluence_min_long, confluence_min_short,
          long_rsi_soft_entry, long_rsi_bonus, short_rsi_soft_entry,
          short_rsi_bonus, tol (in price units, not pips),
          atr_max,
          core_gate_use_zone, core_gate_use_bb, core_gate_use_sar,
          core_gate_use_rsi, core_gate_required,
          regime_enabled, atr_ratio_trend, atr_ratio_calm,
          adx_trend, adx_range, favorable_min_score, neutral_min_score,
          neutral_size_mult,
          ss_enabled, ss_comp,
          combo_filter_enabled, combo_filter_mode, combo_allow, combo_block,
          ss_filter_enabled, ss_filter_min_score,
          cq_enabled, top_combos, bottom_combos,
          high_quality_mult, medium_quality_mult, low_quality_skip,
          rejection_bonus_enabled, div_track_enabled,
          session_env_enabled, session_env_log_ir_pos
    sst : dict
        Session state dict (for session envelope / IR tracking).

    Returns
    -------
    dict or None
        None if no entry candidate.
        A candidate dict with all native payload fields if a signal fired.
        The ``_blocked_reason`` key is set if the signal was blocked by a
        post-confluence filter (combo, signal-strength, quality).
    """
    P = float(pivot_levels["P"])
    R1 = float(pivot_levels["R1"])
    R2 = float(pivot_levels["R2"])
    R3 = float(pivot_levels["R3"])
    S1 = float(pivot_levels["S1"])
    S2 = float(pivot_levels["S2"])
    S3 = float(pivot_levels["S3"])
    bb_u = float(row["bb_upper"])
    bb_l = float(row["bb_lower"])
    rsi = float(row["rsi_m5"])
    sar = float(row["sar_value"])
    sar_dir = str(row["sar_direction"])

    tol = float(cfg_params["tol"])
    confluence_min_long = int(cfg_params["confluence_min_long"])
    confluence_min_short = int(cfg_params["confluence_min_short"])
    long_rsi_soft_entry = float(cfg_params["long_rsi_soft_entry"])
    long_rsi_bonus = float(cfg_params["long_rsi_bonus"])
    short_rsi_soft_entry = float(cfg_params["short_rsi_soft_entry"])
    short_rsi_bonus = float(cfg_params["short_rsi_bonus"])
    tokyo_v2_scoring = bool(cfg_params["tokyo_v2_scoring"])
    atr_max = float(cfg_params["atr_max"])

    # Regime gate
    regime_label, regime_size_mult, regime_blocked = _score_regime_gate(
        row=row,
        mid_close=mid_close,
        R1=R1, R2=R2, S1=S1, S2=S2,
        regime_enabled=bool(cfg_params["regime_enabled"]),
        atr_ratio_trend=float(cfg_params["atr_ratio_trend"]),
        atr_ratio_calm=float(cfg_params["atr_ratio_calm"]),
        adx_trend=float(cfg_params["adx_trend"]),
        adx_range=float(cfg_params["adx_range"]),
        favorable_min_score=int(cfg_params["favorable_min_score"]),
        neutral_min_score=int(cfg_params["neutral_min_score"]),
        neutral_size_mult=float(cfg_params["neutral_size_mult"]),
    )
    if regime_blocked:
        return None

    # Individual conditions
    cond_long_zone = mid_close <= (S1 + tol)
    cond_short_zone = mid_close >= (R1 - tol)
    cond_long_bb = (mid_close <= bb_l) or (mid_low <= bb_l)
    cond_short_bb = (mid_close >= bb_u) or (mid_high >= bb_u)
    cond_long_sar = bool(row.get("sar_flip_bullish_recent", False))
    cond_short_sar = bool(row.get("sar_flip_bearish_recent", False))
    cond_long_rsi_soft = rsi < long_rsi_soft_entry
    cond_short_rsi_soft = rsi > short_rsi_soft_entry
    cond_long_rsi_ext = rsi < long_rsi_bonus
    cond_short_rsi_ext = rsi > short_rsi_bonus
    cond_long_s2 = mid_close <= (S2 + tol)
    cond_short_r2 = mid_close >= (R2 - tol)
    cond_atr_low = float(row["atr_m15"]) <= atr_max

    # Confluence scoring
    long_conditions: dict | None = None
    short_conditions: dict | None = None
    if tokyo_v2_scoring:
        long_signal, short_signal, long_score, short_score, long_conditions, short_conditions = _score_confluence_v2(
            cond_long_zone=cond_long_zone,
            cond_long_s2=cond_long_s2,
            cond_long_bb=cond_long_bb,
            cond_long_sar=cond_long_sar,
            cond_long_rsi_soft=cond_long_rsi_soft,
            cond_long_rsi_ext=cond_long_rsi_ext,
            cond_atr_low=cond_atr_low,
            cond_short_zone=cond_short_zone,
            cond_short_r2=cond_short_r2,
            cond_short_bb=cond_short_bb,
            cond_short_sar=cond_short_sar,
            cond_short_rsi_soft=cond_short_rsi_soft,
            cond_short_rsi_ext=cond_short_rsi_ext,
            confluence_min_long=confluence_min_long,
            confluence_min_short=confluence_min_short,
        )
    else:
        long_signal, short_signal, long_score, short_score = _score_confluence_legacy(
            cond_long_zone=cond_long_zone,
            cond_long_bb=cond_long_bb,
            cond_long_sar=cond_long_sar,
            cond_long_rsi_ext=cond_long_rsi_ext,
            cond_long_s2=cond_long_s2,
            cond_short_zone=cond_short_zone,
            cond_short_bb=cond_short_bb,
            cond_short_sar=cond_short_sar,
            cond_short_rsi_ext=cond_short_rsi_ext,
            cond_short_r2=cond_short_r2,
            cond_long_rsi_soft=cond_long_rsi_soft,
            cond_short_rsi_soft=cond_short_rsi_soft,
            core_gate_use_zone=bool(cfg_params["core_gate_use_zone"]),
            core_gate_use_bb=bool(cfg_params["core_gate_use_bb"]),
            core_gate_use_sar=bool(cfg_params["core_gate_use_sar"]),
            core_gate_use_rsi=bool(cfg_params["core_gate_use_rsi"]),
            core_gate_required=int(cfg_params["core_gate_required"]),
            confluence_min_long=confluence_min_long,
            confluence_min_short=confluence_min_short,
        )

    # Ambiguous both-signals block
    if long_signal and short_signal:
        return None
    if not long_signal and not short_signal:
        return None

    direction = "long" if long_signal else "short"

    # Combo string
    A = bool(cond_long_zone if direction == "long" else cond_short_zone)
    B = bool(cond_long_bb if direction == "long" else cond_short_bb)
    C = bool(cond_long_sar if direction == "long" else cond_short_sar)
    D = bool((rsi < long_rsi_bonus) if direction == "long" else (rsi > short_rsi_bonus))
    E = bool((mid_close <= (S2 + tol)) if direction == "long" else (mid_close >= (R2 - tol)))
    combo = "".join(x for x, ok in [("A", A), ("B", B), ("C", C), ("D", D), ("E", E)] if ok)

    # Signal strength
    signal_strength_score, signal_strength_tier = _compute_signal_strength(
        direction=direction,
        row=row,
        rsi=rsi,
        bb_l=bb_l,
        bb_u=bb_u,
        mid_close=mid_close,
        E=E,
        long_score=long_score,
        short_score=short_score,
        ss_enabled=bool(cfg_params["ss_enabled"]),
        ss_comp=cfg_params.get("ss_comp", {}),
    )

    # Post-confluence filters -- these return a blocked candidate so the
    # caller can still update diagnostics the same way it did inline.
    blocked_reason = None

    # Combo filter
    if bool(cfg_params.get("combo_filter_enabled", False)):
        combo_filter_mode = str(cfg_params.get("combo_filter_mode", "allowlist"))
        combo_allow = cfg_params.get("combo_allow", set())
        combo_block = cfg_params.get("combo_block", set())
        allow = True
        if combo_filter_mode == "allowlist":
            allow = combo in combo_allow
        elif combo_filter_mode == "blocklist":
            allow = combo not in combo_block
        if not allow:
            blocked_reason = "combo_filter"

    # Signal-strength filter
    if blocked_reason is None and bool(cfg_params.get("ss_filter_enabled", False)):
        if int(signal_strength_score) < int(cfg_params.get("ss_filter_min_score", 0)):
            blocked_reason = "signal_strength_filter"

    # Confluence quality filter
    quality_label = "medium"
    quality_mult = float(cfg_params.get("medium_quality_mult", 0.75))
    if bool(cfg_params.get("cq_enabled", False)):
        top_combos = cfg_params.get("top_combos", set())
        bottom_combos = cfg_params.get("bottom_combos", set())
        high_quality_mult = float(cfg_params.get("high_quality_mult", 1.0))
        if combo in top_combos:
            quality_label = "high"
            quality_mult = high_quality_mult
        elif combo in bottom_combos:
            quality_label = "low"
            quality_mult = 0.0
            if bool(cfg_params.get("low_quality_skip", True)):
                blocked_reason = "low_quality_combo"

    # Quality markers
    qm = _compute_quality_markers(
        direction=direction,
        row=row,
        mid_close=mid_close,
        rejection_bonus_enabled=bool(cfg_params.get("rejection_bonus_enabled", False)),
        div_track_enabled=bool(cfg_params.get("div_track_enabled", False)),
        session_env_enabled=bool(cfg_params.get("session_env_enabled", False)),
        session_env_log_ir_pos=bool(cfg_params.get("session_env_log_ir_pos", True)),
        sst=sst,
    )

    dist_to_pivot = abs(mid_close - P) / PIP_SIZE

    candidate = {
        "direction": direction,
        "confluence_score": int(long_score if long_signal else short_score),
        "confluence_combo": combo,
        "long_signal": long_signal,
        "short_signal": short_signal,
        "long_score": long_score,
        "short_score": short_score,
        "long_conditions": long_conditions,
        "short_conditions": short_conditions,
        "signal_strength_score": int(signal_strength_score),
        "signal_strength_tier": str(signal_strength_tier),
        "quality_label": quality_label,
        "quality_mult": float(quality_mult),
        "regime_label": regime_label,
        "regime_mult": float(regime_size_mult),
        "from_zone": bool(mid_close <= (S2 + tol) if direction == "long" else mid_close >= (R2 - tol)),
        "P": P,
        "R1": R1,
        "R2": R2,
        "R3": R3,
        "S1": S1,
        "S2": S2,
        "S3": S3,
        "bb_upper": bb_u,
        "bb_lower": bb_l,
        "sar_value": sar,
        "sar_direction": sar_dir,
        "rsi_m5": rsi,
        "atr_m15": float(row["atr_m15"]),
        "bb_mid": float(row["bb_mid"]),
        "rejection_confirmed": qm["rejection_confirmed"],
        "rejection_low": qm["rejection_low"],
        "rejection_high": qm["rejection_high"],
        "rejection_wick_ratio": qm["rejection_wick_ratio"],
        "divergence_present": qm["divergence_present"],
        "inside_ir": qm["inside_ir"],
        "quality_markers": qm["quality_markers"],
        "session_midpoint": qm["session_midpoint"],
        "distance_to_ir_boundary_pips": qm["dist_ir_boundary"],
        "distance_to_midpoint_pips": qm["dist_to_midpoint"],
        "distance_to_pivot_pips": float(dist_to_pivot),
        # Conditions for diagnostics
        "cond_long_zone": cond_long_zone,
        "cond_short_zone": cond_short_zone,
        "cond_long_bb": cond_long_bb,
        "cond_short_bb": cond_short_bb,
        "cond_long_sar": cond_long_sar,
        "cond_short_sar": cond_short_sar,
        "cond_long_rsi_soft": cond_long_rsi_soft,
        "cond_short_rsi_soft": cond_short_rsi_soft,
        "cond_atr_low": cond_atr_low,
        # Post-confluence filter result
        "_blocked_reason": blocked_reason,
    }
    return candidate
