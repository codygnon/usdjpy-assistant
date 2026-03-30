from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import numpy as np


def evaluate_v14_confluence(
    side: str,
    close: float,
    high: float,
    low: float,
    pivots: dict,
    bb_upper: float,
    bb_lower: float,
    sar_bullish_flip: bool,
    sar_bearish_flip: bool,
    rsi: float,
    pip_size: float,
    *,
    zone_tolerance_pips: float = 20.0,
    rsi_long_entry: float = 35.0,
    rsi_short_entry: float = 65.0,
    core_gate_use_zone: bool = True,
    core_gate_use_bb: bool = True,
    core_gate_use_sar: bool = True,
    core_gate_use_rsi: bool = True,
) -> tuple[int, str]:
    score = 0
    combo = ""
    tol = float(zone_tolerance_pips) * pip_size

    if side == "buy":
        if core_gate_use_zone and close <= pivots["S1"] + tol:
            score += 1
            combo += "A"
        if core_gate_use_bb and (close <= bb_lower or low <= bb_lower):
            score += 1
            combo += "B"
        if core_gate_use_sar and sar_bullish_flip:
            score += 1
            combo += "C"
        if core_gate_use_rsi and rsi < float(rsi_long_entry):
            score += 1
            combo += "D"
    else:
        if core_gate_use_zone and close >= pivots["R1"] - tol:
            score += 1
            combo += "A"
        if core_gate_use_bb and (close >= bb_upper or high >= bb_upper):
            score += 1
            combo += "B"
        if core_gate_use_sar and sar_bearish_flip:
            score += 1
            combo += "C"
        if core_gate_use_rsi and rsi > float(rsi_short_entry):
            score += 1
            combo += "D"

    return score, combo


def compute_v14_signal_strength(
    *,
    side: str,
    confluence_score: int,
    close: float,
    high: float,
    low: float,
    pivots: dict[str, Any],
    bb_upper: float,
    bb_lower: float,
    rsi: float,
    sar_bullish_flip: bool,
    sar_bearish_flip: bool,
    now_utc: datetime,
    pip_size: float,
    sst_cfg: dict[str, Any],
) -> dict[str, Any]:
    components = sst_cfg.get("components", {}) if isinstance(sst_cfg.get("components"), dict) else {}
    cc_map = components.get("confluence_count", {}) if isinstance(components.get("confluence_count"), dict) else {}
    score = int(cc_map.get(str(int(confluence_score)), 0))

    bb_pen_threshold = float(components.get("bb_penetration_bonus_pips", 0.0))
    if bb_pen_threshold > 0:
        pen = (bb_lower - low) / pip_size if side == "buy" else (high - bb_upper) / pip_size
        if pen >= bb_pen_threshold:
            score += 1

    if bool(components.get("deep_pivot_zone", False)):
        if side == "buy" and close <= float(pivots.get("S2", np.nan)):
            score += 1
        if side == "sell" and close >= float(pivots.get("R2", np.nan)):
            score += 1

    if bool(components.get("same_candle_sar_flip", False)):
        if side == "buy" and sar_bullish_flip:
            score += 1
        if side == "sell" and sar_bearish_flip:
            score += 1

    rsi_ext = float(components.get("rsi_extreme_bonus_threshold", 0.0))
    if rsi_ext > 0:
        if side == "buy" and rsi <= rsi_ext:
            score += 1
        if side == "sell" and rsi >= (100.0 - rsi_ext):
            score += 1

    favorable = components.get("favorable_hour", [])
    if isinstance(favorable, (list, tuple, set)):
        fav_set = set()
        for hour in favorable:
            try:
                fav_set.add(int(hour))
            except Exception:
                continue
        if now_utc.hour in fav_set:
            score += 1

    weak_max = int(sst_cfg.get("weak_max_score", 2))
    strong_min = int(sst_cfg.get("strong_min_score", 5))
    bucket = "strong" if score >= strong_min else "weak" if score <= weak_max else "moderate"

    size_mults = sst_cfg.get("size_multipliers", {}) if isinstance(sst_cfg.get("size_multipliers"), dict) else {}
    default_mults = {"weak": 1.0, "moderate": 1.0, "strong": 1.0}
    for key in ("weak", "moderate", "strong"):
        try:
            if key in size_mults:
                default_mults[key] = float(size_mults[key])
        except Exception:
            pass
    return {"score": int(score), "bucket": bucket, "size_mult": float(default_mults.get(bucket, 1.0))}


def compute_v14_sl(
    side: str,
    entry_price: float,
    pivots: dict,
    pip_size: float,
    *,
    sl_buffer_pips: float = 8.0,
    min_sl_pips: float = 12.0,
    max_sl_pips: float = 35.0,
) -> float:
    buffer = float(sl_buffer_pips) * pip_size
    if side == "buy":
        support_levels = sorted([pivots["S1"], pivots["S2"], pivots["S3"]])
        nearest = next((lvl for lvl in support_levels if lvl < entry_price), pivots["S1"])
        raw_sl = nearest - buffer
        sl_dist = (entry_price - raw_sl) / pip_size
    else:
        resist_levels = sorted([pivots["R1"], pivots["R2"], pivots["R3"]], reverse=True)
        nearest = next((lvl for lvl in resist_levels if lvl > entry_price), pivots["R1"])
        raw_sl = nearest + buffer
        sl_dist = (raw_sl - entry_price) / pip_size

    sl_dist = max(float(min_sl_pips), min(float(max_sl_pips), sl_dist))
    return entry_price - sl_dist * pip_size if side == "buy" else entry_price + sl_dist * pip_size


def compute_v14_tp1(
    side: str,
    entry_price: float,
    atr_value: float,
    pip_size: float,
    *,
    partial_tp_atr_mult: float = 0.5,
    partial_tp_min_pips: float = 6.0,
    partial_tp_max_pips: float = 12.0,
) -> float:
    tp_dist = atr_value * float(partial_tp_atr_mult)
    tp_pips = tp_dist / pip_size
    tp_pips = max(float(partial_tp_min_pips), min(float(partial_tp_max_pips), tp_pips))
    return entry_price + tp_pips * pip_size if side == "buy" else entry_price - tp_pips * pip_size


def compute_v14_lot_size(
    equity: float,
    sl_pips: float,
    current_price: float,
    pip_size: float,
    leverage: float = 33.3,
    risk_pct: float = 0.02,
    max_units: int = 500_000,
) -> int:
    if sl_pips <= 0 or current_price <= 0:
        return 0
    risk_amount = equity * risk_pct
    pip_value_per_unit = pip_size / current_price
    units = risk_amount / (sl_pips * pip_value_per_unit)
    max_margin_units = (equity * leverage) / current_price
    units = min(units, max_margin_units, float(max_units))
    return int(math.floor(units))


def compute_v14_units_from_config(
    equity: float,
    sl_pips: float,
    current_price: float,
    pip_size: float,
    now_utc: datetime,
    v14_config: dict[str, Any],
) -> int:
    if sl_pips <= 0 or current_price <= 0:
        return 0
    risk_pct = float(v14_config.get("risk_per_trade_pct", 2.0)) / 100.0
    max_units = int(v14_config.get("max_units", 500_000))
    leverage = float(v14_config.get("leverage", 33.3))
    day_multipliers = v14_config.get("day_risk_multipliers") or {}
    day_name = now_utc.strftime("%A")
    day_mult = float(day_multipliers.get(day_name, 1.0)) if isinstance(day_multipliers.get(day_name), (int, float)) else 1.0
    risk_amount = equity * risk_pct * day_mult
    pip_value_per_unit = pip_size / current_price
    units = risk_amount / (sl_pips * pip_value_per_unit)
    max_margin_units = (equity * leverage) / current_price
    units = min(units, max_margin_units, float(max_units))
    return int(math.floor(max(0.0, units)))
