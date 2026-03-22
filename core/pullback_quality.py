"""Pullback-quality scoring for Trial #10 tier entries.

This measures whether the pullback into a touched EMA tier was orderly
or sloppy, using only closed M1 candles:

- pullback_bar_count: number of bars since the most recent local swing extreme
- structure_ratio: fraction of pullback bars that progressed cleanly
  (lower-highs for bull pullbacks, higher-lows for bear pullbacks)

The first rollout is intentionally simple and discrete:
- orderly
- neutral
- sloppy

The result can be used as a dashboard diagnostic and as a shallow-tier
lot dampener without blocking the entry outright.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class PullbackQualityResult:
    enabled: bool = False
    applicable: bool = False
    side: str = ""
    tier: Optional[int] = None
    label: str = "neutral"
    pullback_bar_count: int = 0
    structure_ratio: float = 0.0
    lookback_bars: int = 30
    swing_index: Optional[int] = None
    touch_index: Optional[int] = None
    swing_time: Optional[pd.Timestamp] = None
    touch_time: Optional[pd.Timestamp] = None
    swing_price: Optional[float] = None
    touch_price: Optional[float] = None
    dampener_multiplier: float = 1.0
    dampener_applied: bool = False
    reason: str = ""


def _last_extreme_offset(values: list[float], want_max: bool) -> int:
    if not values:
        return 0
    target = max(values) if want_max else min(values)
    for idx in range(len(values) - 1, -1, -1):
        if values[idx] == target:
            return idx
    return 0


def analyze_pullback_quality(
    *,
    m1_df: pd.DataFrame,
    touch_index: int,
    side: str,
    tier: Optional[int],
    enabled: bool = True,
    lookback_bars: int = 30,
    orderly_bar_count_min: int = 6,
    sloppy_bar_count_max: int = 2,
    orderly_structure_ratio_min: float = 0.6,
    sloppy_structure_ratio_max: float = 0.3,
    shallow_tiers: tuple[int, ...] = (17, 21),
    sloppy_lot_multiplier: float = 0.5,
) -> PullbackQualityResult:
    result = PullbackQualityResult(
        enabled=bool(enabled),
        applicable=bool(tier in tuple(int(x) for x in shallow_tiers)),
        side=str(side or ""),
        tier=int(tier) if tier is not None else None,
        lookback_bars=max(1, int(lookback_bars)),
    )
    if not enabled:
        result.reason = "pullback_quality_disabled"
        return result
    if m1_df is None or m1_df.empty or touch_index <= 0 or touch_index >= len(m1_df):
        result.reason = "insufficient_m1_data"
        return result
    if side not in {"bull", "bear"}:
        result.reason = "unknown_side"
        return result
    if "high" not in m1_df.columns or "low" not in m1_df.columns:
        result.reason = "missing_ohlc_columns"
        return result

    window_start = max(0, int(touch_index) - result.lookback_bars)
    window = m1_df.iloc[window_start : touch_index + 1]
    if window.empty:
        result.reason = "empty_window"
        return result

    if side == "bull":
        swing_values = window["high"].astype(float).tolist()
        swing_rel = _last_extreme_offset(swing_values, want_max=True)
        price_series = m1_df["high"].astype(float).tolist()
        compare_col = "high"
        compare_fn = lambda cur, prev: cur < prev
    else:
        swing_values = window["low"].astype(float).tolist()
        swing_rel = _last_extreme_offset(swing_values, want_max=False)
        price_series = m1_df["low"].astype(float).tolist()
        compare_col = "low"
        compare_fn = lambda cur, prev: cur > prev

    swing_index = window_start + swing_rel
    pullback_bar_count = max(0, int(touch_index) - int(swing_index))
    result.swing_index = swing_index
    result.touch_index = int(touch_index)
    result.pullback_bar_count = pullback_bar_count
    result.swing_price = float(price_series[swing_index]) if 0 <= swing_index < len(price_series) else None
    result.touch_price = float(price_series[touch_index]) if 0 <= touch_index < len(price_series) else None

    try:
        if "time" in m1_df.columns:
            result.swing_time = pd.to_datetime(m1_df["time"].iloc[swing_index], utc=True)
            result.touch_time = pd.to_datetime(m1_df["time"].iloc[touch_index], utc=True)
        elif isinstance(m1_df.index, pd.DatetimeIndex):
            result.swing_time = pd.to_datetime(m1_df.index[swing_index], utc=True)
            result.touch_time = pd.to_datetime(m1_df.index[touch_index], utc=True)
    except Exception:
        pass

    comparisons = 0
    orderly_steps = 0
    if pullback_bar_count > 0:
        pullback_vals = m1_df[compare_col].iloc[swing_index : touch_index + 1].astype(float).tolist()
        for i in range(1, len(pullback_vals)):
            comparisons += 1
            if compare_fn(pullback_vals[i], pullback_vals[i - 1]):
                orderly_steps += 1
    result.structure_ratio = float(orderly_steps / comparisons) if comparisons > 0 else 0.0

    if result.pullback_bar_count >= int(orderly_bar_count_min) and result.structure_ratio >= float(orderly_structure_ratio_min):
        result.label = "orderly"
    elif result.pullback_bar_count <= int(sloppy_bar_count_max) or result.structure_ratio < float(sloppy_structure_ratio_max):
        result.label = "sloppy"
    else:
        result.label = "neutral"

    if result.applicable and result.label == "sloppy":
        result.dampener_multiplier = float(sloppy_lot_multiplier)
        result.dampener_applied = True

    result.reason = (
        f"{result.label}: {result.pullback_bar_count} bars since swing, "
        f"structure_ratio={result.structure_ratio:.2f}"
    )
    return result


def pullback_quality_snapshot(result: PullbackQualityResult | None) -> Optional[dict]:
    if result is None:
        return None
    return {
        "enabled": result.enabled,
        "applicable": result.applicable,
        "side": result.side,
        "tier": result.tier,
        "label": result.label,
        "pullback_bar_count": result.pullback_bar_count,
        "structure_ratio": round(result.structure_ratio, 4),
        "lookback_bars": result.lookback_bars,
        "swing_index": result.swing_index,
        "touch_index": result.touch_index,
        "swing_time": result.swing_time.isoformat() if result.swing_time is not None else None,
        "touch_time": result.touch_time.isoformat() if result.touch_time is not None else None,
        "swing_price": result.swing_price,
        "touch_price": result.touch_price,
        "dampener_multiplier": result.dampener_multiplier,
        "dampener_applied": result.dampener_applied,
        "reason": result.reason,
    }
