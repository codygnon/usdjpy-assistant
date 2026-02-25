from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

import pandas as pd

from core.indicators import adx as adx_fn
from core.indicators import atr as atr_fn
from core.indicators import detect_rsi_divergence as detect_rsi_divergence_fn
from core.indicators import ema as ema_fn
from core.indicators import rsi as rsi_fn


_RISK_TIERS = ("low", "medium", "high", "critical")


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _tier_rank(tier: str) -> int:
    t = str(tier or "").lower()
    if t == "medium":
        return 1
    if t == "high":
        return 2
    if t == "critical":
        return 3
    return 0


def _session_name_utc(ts_utc: pd.Timestamp) -> str:
    h = int(ts_utc.hour)
    if 0 <= h < 8:
        return "tokyo"
    if 8 <= h < 13:
        return "london"
    return "ny"


def compute_regime(
    m5_df: Optional[pd.DataFrame],
    adx_period: int = 14,
    trending_threshold: float = 25.0,
    ranging_threshold: float = 20.0,
    atr_ratio_lookback: int = 20,
    atr_ratio_trending: float = 1.2,
    use_atr_fallback: bool = False,
) -> Literal["trending", "ranging", "transition"]:
    """Classify market regime from M5: trending (ADX high), ranging (ADX low), or transition.

    When ADX is in transition band, optional ATR fallback: range/ATR ratio > atr_ratio_trending -> trending.
    """
    if m5_df is None or m5_df.empty or len(m5_df) < adx_period + atr_ratio_lookback:
        return "transition"
    try:
        adx_series = adx_fn(m5_df, period=adx_period)
        adx_val = float(adx_series.iloc[-1]) if len(adx_series) else 0.0
    except Exception:
        return "transition"

    if adx_val >= trending_threshold:
        return "trending"
    if adx_val <= ranging_threshold:
        return "ranging"

    if use_atr_fallback and len(m5_df) >= atr_ratio_lookback:
        try:
            window = m5_df.tail(atr_ratio_lookback)
            rng = (window["high"].astype(float) - window["low"].astype(float)).mean()
            atr_series = atr_fn(m5_df, period=adx_period)
            atr_val = float(atr_series.iloc[-1]) if len(atr_series) else 1e-10
            if atr_val > 0 and rng / atr_val >= atr_ratio_trending:
                return "trending"
        except Exception:
            pass
    return "transition"


def _find_price_rsi_divergence(
    close: pd.Series,
    rsi: pd.Series,
    lookback: int,
    min_pivot_separation_bars: int = 5,
) -> dict:
    """Pivot-based divergence detection on latest lookback bars.

    Requires at least min_pivot_separation_bars between last two highs/lows to filter noise.
    Returns deltas and last pivot index (for confirmation check).
    """
    if close is None or rsi is None or len(close) < max(lookback, 8):
        return {
            "bearish_found": False,
            "bullish_found": False,
            "bearish_delta": 0.0,
            "bullish_delta": 0.0,
            "bearish_last_pivot_idx": None,
            "bullish_last_pivot_idx": None,
            "bearish_separation": 0,
            "bullish_separation": 0,
        }

    c = close.tail(lookback).astype(float).reset_index(drop=True)
    rv = rsi.tail(lookback).astype(float).reset_index(drop=True)
    n = len(c)
    if n < 8:
        return {
            "bearish_found": False,
            "bullish_found": False,
            "bearish_delta": 0.0,
            "bullish_delta": 0.0,
            "bearish_last_pivot_idx": None,
            "bullish_last_pivot_idx": None,
            "bearish_separation": 0,
            "bullish_separation": 0,
        }

    sep = max(1, int(min_pivot_separation_bars))
    highs: list[int] = []
    lows: list[int] = []
    wing = 2
    for i in range(wing, n - wing):
        w = c.iloc[i - wing : i + wing + 1]
        v = float(c.iloc[i])
        if v >= float(w.max()):
            highs.append(i)
        if v <= float(w.min()):
            lows.append(i)

    bearish_found = False
    bullish_found = False
    bearish_delta = 0.0
    bullish_delta = 0.0
    bearish_last_pivot_idx: Optional[int] = None
    bullish_last_pivot_idx: Optional[int] = None

    bearish_separation = 0
    bullish_separation = 0
    if len(highs) >= 2:
        i1, i2 = highs[-2], highs[-1]
        if (i2 - i1) >= sep:
            p1, p2 = float(c.iloc[i1]), float(c.iloc[i2])
            r1, r2 = float(rv.iloc[i1]), float(rv.iloc[i2])
            if p2 > p1 and r2 < r1:
                bearish_found = True
                bearish_delta = abs(r1 - r2)
                bearish_last_pivot_idx = i2
                bearish_separation = i2 - i1

    if len(lows) >= 2:
        i1, i2 = lows[-2], lows[-1]
        if (i2 - i1) >= sep:
            p1, p2 = float(c.iloc[i1]), float(c.iloc[i2])
            r1, r2 = float(rv.iloc[i1]), float(rv.iloc[i2])
            if p2 < p1 and r2 > r1:
                bullish_found = True
                bullish_delta = abs(r2 - r1)
                bullish_last_pivot_idx = i2
                bullish_separation = i2 - i1

    return {
        "bearish_found": bearish_found,
        "bullish_found": bullish_found,
        "bearish_delta": round(bearish_delta, 3),
        "bullish_delta": round(bullish_delta, 3),
        "bearish_last_pivot_idx": bearish_last_pivot_idx,
        "bullish_last_pivot_idx": bullish_last_pivot_idx,
        "bearish_separation": bearish_separation,
        "bullish_separation": bullish_separation,
    }


def _rsi_severity_from_delta(delta: float, min_delta: float) -> float:
    """Stepped severity: 10->0.2, 18->0.5, 36->1.0. Below min_delta -> 0."""
    if delta < min_delta:
        return 0.0
    if delta < 10.0:
        return 0.0
    if delta <= 18.0:
        return 0.2
    if delta <= 36.0:
        return _clamp(0.2 + (delta - 18.0) / 18.0 * 0.8, 0.2, 1.0)
    return 1.0


def _compute_rsi_divergence_component(
    *,
    m5_df: Optional[pd.DataFrame],
    trend_side: str,
    rsi_period: int,
    lookback: int,
    severity_midpoint: float,
    use_rolling_fallback: bool = True,
    require_confirmation_bar: bool = True,
    min_delta_for_score: float = 8.0,
    min_pivot_separation_bars: int = 5,
) -> dict:
    if m5_df is None or m5_df.empty or len(m5_df) < max(lookback + 2, rsi_period + 5):
        return {
            "name": "rsi_divergence",
            "score": 0.0,
            "found": False,
            "severity": 0.0,
            "confidence": 0.0,
            "direction": None,
            "details": "insufficient_m5_data",
        }

    lb = max(8, int(lookback))
    close = m5_df["close"].astype(float)
    high = m5_df["high"].astype(float) if "high" in m5_df.columns else close
    low = m5_df["low"].astype(float) if "low" in m5_df.columns else close
    rsi = rsi_fn(close, max(2, int(rsi_period)))
    div = _find_price_rsi_divergence(
        close, rsi, lb, min_pivot_separation_bars=int(min_pivot_separation_bars)
    )

    side = str(trend_side or "bull").lower()
    if side == "bull":
        found = bool(div["bearish_found"])
        delta = float(div["bearish_delta"])
        direction = "bearish"
        last_pivot_idx = div.get("bearish_last_pivot_idx")
        check_high = True
    else:
        found = bool(div["bullish_found"])
        delta = float(div["bullish_delta"])
        direction = "bullish"
        last_pivot_idx = div.get("bullish_last_pivot_idx")
        check_high = False

    if found and require_confirmation_bar and last_pivot_idx is not None:
        c_tail = close.tail(lb).reset_index(drop=True)
        h_tail = high.tail(lb).reset_index(drop=True)
        l_tail = low.tail(lb).reset_index(drop=True)
        n = len(c_tail)
        if last_pivot_idx >= n:
            last_pivot_idx = n - 1
        if check_high:
            pivot_price = float(h_tail.iloc[last_pivot_idx])
            after_highs = h_tail.iloc[last_pivot_idx + 1 : n]
            if len(after_highs) > 0 and float(after_highs.max()) > pivot_price:
                found = False
                delta = 0.0
        else:
            pivot_price = float(l_tail.iloc[last_pivot_idx])
            after_lows = l_tail.iloc[last_pivot_idx + 1 : n]
            if len(after_lows) > 0 and float(after_lows.min()) < pivot_price:
                found = False
                delta = 0.0

    if not found and use_rolling_fallback and len(m5_df) >= 50:
        has_bearish, has_bullish, roll_details = detect_rsi_divergence_fn(
            m5_df, rsi_period=int(rsi_period), lookback_bars=min(50, lb * 2)
        )
        if side == "bull" and has_bearish and isinstance(roll_details.get("bearish_divergence"), dict):
            d = roll_details["bearish_divergence"]
            delta = abs(float(d.get("recent_rsi", 0)) - float(d.get("ref_rsi", 0)))
            found = delta >= min_delta_for_score
            direction = "bearish"
        elif side != "bull" and has_bullish and isinstance(roll_details.get("bullish_divergence"), dict):
            d = roll_details["bullish_divergence"]
            delta = abs(float(d.get("recent_rsi", 0)) - float(d.get("ref_rsi", 0)))
            found = delta >= min_delta_for_score
            direction = "bullish"

    min_delta = max(0.0, float(min_delta_for_score))
    if found and delta < min_delta:
        found = False
        severity = 0.0
    else:
        severity = _rsi_severity_from_delta(delta, min_delta) if found else 0.0

    confidence = 0.0
    if found:
        sep = div.get("bearish_separation", 0) if direction == "bearish" else div.get("bullish_separation", 0)
        confidence = min(1.0, max(0.0, int(sep)) / 10.0)

    return {
        "name": "rsi_divergence",
        "score": round(severity, 4),
        "found": found,
        "severity": round(severity, 4),
        "confidence": round(confidence, 4),
        "delta_rsi_points": round(delta, 3),
        "direction": direction,
        "details": f"{direction}_divergence" if found else "none",
    }


def _compute_adr_exhaustion_component(
    *,
    d_df: Optional[pd.DataFrame],
    trial7_daily_state: Optional[dict],
    pip_size: float,
    adr_period: int,
    ramp_start_pct: float,
    score_100: float,
    score_120: float,
    score_150: float,
) -> dict:
    if d_df is None or d_df.empty or len(d_df) < max(adr_period + 2, 5):
        return {
            "name": "adr_exhaustion",
            "score": 0.0,
            "consumed_pct": None,
            "remaining_pips_estimate": None,
            "adr_pips": None,
            "details": "insufficient_daily_data",
        }

    d = d_df.copy()
    d["time"] = pd.to_datetime(d["time"], utc=True, errors="coerce")
    d = d.dropna(subset=["time"]).sort_values("time")
    if d.empty:
        return {
            "name": "adr_exhaustion",
            "score": 0.0,
            "consumed_pct": None,
            "remaining_pips_estimate": None,
            "adr_pips": None,
            "details": "invalid_daily_time",
        }

    now_date = datetime.now(timezone.utc).date().isoformat()
    if len(d) >= 2 and str(d.iloc[-1]["time"].date().isoformat()) == now_date:
        hist = d.iloc[:-1].copy()
    else:
        hist = d.copy()
    if len(hist) < max(adr_period, 3):
        return {
            "name": "adr_exhaustion",
            "score": 0.0,
            "consumed_pct": None,
            "remaining_pips_estimate": None,
            "adr_pips": None,
            "details": "insufficient_hist_daily_data",
        }

    hist_range = (hist["high"].astype(float) - hist["low"].astype(float)).abs() / float(pip_size)
    adr = float(hist_range.tail(max(2, int(adr_period))).mean())
    if adr <= 0:
        return {
            "name": "adr_exhaustion",
            "score": 0.0,
            "consumed_pct": None,
            "remaining_pips_estimate": None,
            "adr_pips": None,
            "details": "invalid_adr",
        }

    day_high = None
    day_low = None
    if isinstance(trial7_daily_state, dict):
        day_high = trial7_daily_state.get("today_high")
        day_low = trial7_daily_state.get("today_low")
    if day_high is None or day_low is None:
        # Fallback: current D candle if available
        day_row = d.iloc[-1]
        day_high = float(day_row["high"])
        day_low = float(day_row["low"])

    day_range_pips = abs(float(day_high) - float(day_low)) / float(pip_size)
    consumed = (day_range_pips / adr) * 100.0

    x = consumed
    x0 = float(ramp_start_pct)
    if x <= x0:
        score = 0.0
    elif x <= 100.0:
        score = (x - x0) / max(1e-6, (100.0 - x0)) * float(score_100)
    elif x <= 120.0:
        score = float(score_100) + ((x - 100.0) / 20.0) * (float(score_120) - float(score_100))
    elif x <= 150.0:
        score = float(score_120) + ((x - 120.0) / 30.0) * (float(score_150) - float(score_120))
    elif x <= 180.0:
        score = float(score_150) + ((x - 150.0) / 30.0) * (1.0 - float(score_150))
    else:
        score = 1.0
    score = _clamp(score, 0.0, 1.0)

    return {
        "name": "adr_exhaustion",
        "score": round(score, 4),
        "consumed_pct": round(consumed, 2),
        "remaining_pips_estimate": round(adr - day_range_pips, 2),
        "adr_pips": round(adr, 2),
        "details": "ok",
    }


def _compute_ema_spread_component(
    *,
    m5_df: Optional[pd.DataFrame],
    pip_size: float,
    threshold_pips: float,
    max_pips: float,
) -> dict:
    if m5_df is None or m5_df.empty or len(m5_df) < 24:
        return {
            "name": "ema_spread",
            "score": 0.0,
            "spread_pips": None,
            "overextended": False,
            "details": "insufficient_m5_data",
        }

    close = m5_df["close"].astype(float)
    ema9 = ema_fn(close, 9)
    ema21 = ema_fn(close, 21)
    spread_pips = abs(float(ema9.iloc[-1]) - float(ema21.iloc[-1])) / float(pip_size)

    lo = 3.0
    hi = max(lo + 0.001, float(max_pips))
    score = _clamp((spread_pips - lo) / (hi - lo), 0.0, 1.0)

    return {
        "name": "ema_spread",
        "score": round(score, 4),
        "spread_pips": round(spread_pips, 3),
        "overextended": bool(spread_pips > float(threshold_pips)),
        "details": "ok",
    }


def _round_number_levels(price: float) -> list[tuple[str, float, str]]:
    """Return (name, price, level_type)."""
    levels: list[tuple[str, float, str]] = []
    whole = int(price)
    for base in (whole - 1, whole, whole + 1, whole + 2):
        levels.append((f"round_{base}.000", float(base), "round"))
        levels.append((f"round_{base}.500", float(base) + 0.5, "round"))
    return levels


def _swing_high_low_local_extrema(df: pd.DataFrame, lookback: int, wing: int = 2) -> tuple[Optional[float], Optional[float]]:
    """Swing high/low as local extrema (peak/trough in wing-neighborhood). Returns (swing_high, swing_low)."""
    if df is None or len(df) < lookback or wing < 1:
        return None, None
    window = df.tail(lookback)
    high = window["high"].astype(float)
    low = window["low"].astype(float)
    n = len(high)
    swing_highs: list[float] = []
    swing_lows: list[float] = []
    for i in range(wing, n - wing):
        h_val = float(high.iloc[i])
        l_val = float(low.iloc[i])
        h_win = high.iloc[i - wing : i + wing + 1]
        l_win = low.iloc[i - wing : i + wing + 1]
        if h_val >= float(h_win.max()):
            swing_highs.append(h_val)
        if l_val <= float(l_win.min()):
            swing_lows.append(l_val)
    sh = float(max(swing_highs)) if swing_highs else None
    sl = float(min(swing_lows)) if swing_lows else None
    return sh, sl


def _compute_htf_proximity_component(
    *,
    m15_df: Optional[pd.DataFrame],
    d_df: Optional[pd.DataFrame],
    h4_df: Optional[pd.DataFrame],
    trial7_daily_state: Optional[dict],
    current_price: float,
    trend_side: str,
    pip_size: float,
    buffer_prev_day_pips: float,
    buffer_round_pips: float,
    buffer_swing_pips: float,
    swing_lookback: int,
    score_decay_pips: float,
    use_h4_levels: bool = True,
) -> dict:
    """HTF proximity with level-type buffers and exponential decay score."""
    import math
    levels: list[tuple[str, float, str]] = []

    prev_day_high = None
    prev_day_low = None
    if isinstance(trial7_daily_state, dict):
        prev_day_high = trial7_daily_state.get("prev_day_high")
        prev_day_low = trial7_daily_state.get("prev_day_low")
    if (prev_day_high is None or prev_day_low is None) and d_df is not None and len(d_df) >= 2:
        d = d_df.copy()
        d["time"] = pd.to_datetime(d["time"], utc=True, errors="coerce")
        d = d.dropna(subset=["time"]).sort_values("time")
        if len(d) >= 2:
            now_date = datetime.now(timezone.utc).date().isoformat()
            if str(d.iloc[-1]["time"].date().isoformat()) == now_date and len(d) >= 2:
                ref = d.iloc[-2]
            else:
                ref = d.iloc[-1]
            prev_day_high = float(ref["high"])
            prev_day_low = float(ref["low"])

    if prev_day_high is not None:
        levels.append(("prev_day_high", float(prev_day_high), "prev_day"))
    if prev_day_low is not None:
        levels.append(("prev_day_low", float(prev_day_low), "prev_day"))

    levels.extend(_round_number_levels(float(current_price)))

    lb = max(5, int(swing_lookback))
    if m15_df is not None and not m15_df.empty:
        sh, sl = _swing_high_low_local_extrema(m15_df, lb)
        if sh is not None:
            levels.append(("m15_swing_high", sh, "swing"))
        if sl is not None:
            levels.append(("m15_swing_low", sl, "swing"))

    if use_h4_levels and h4_df is not None and not h4_df.empty and len(h4_df) >= lb:
        h4_lookback = min(lb, 20)
        sh, sl = _swing_high_low_local_extrema(h4_df, h4_lookback)
        if sh is not None:
            levels.append(("h4_swing_high", sh, "swing"))
        if sl is not None:
            levels.append(("h4_swing_low", sl, "swing"))

    side = str(trend_side or "bull").lower()
    opp: list[tuple[str, float, str]] = []
    for name, lv, ltype in levels:
        if side == "bull":
            if lv >= current_price:
                opp.append((name, lv, ltype))
        else:
            if lv <= current_price:
                opp.append((name, lv, ltype))

    if not opp:
        return {
            "name": "htf_proximity",
            "score": 0.0,
            "nearest_level": None,
            "distance_pips": None,
            "details": "no_opposing_levels",
        }

    buf_prev = max(0.1, float(buffer_prev_day_pips))
    buf_round = max(0.1, float(buffer_round_pips))
    buf_swing = max(0.1, float(buffer_swing_pips))
    decay = max(0.1, float(score_decay_pips))
    best_score = 0.0
    nearest_name = None
    nearest_dist = None
    for name, lv, ltype in opp:
        dist_pips = abs(float(lv) - float(current_price)) / float(pip_size)
        buf = buf_prev if ltype == "prev_day" else (buf_round if ltype == "round" else buf_swing)
        if dist_pips > buf:
            continue
        score = math.exp(-dist_pips / decay)
        if score > best_score:
            best_score = score
            nearest_name = name
            nearest_dist = dist_pips

    if nearest_dist is None:
        return {
            "name": "htf_proximity",
            "score": 0.0,
            "nearest_level": None,
            "distance_pips": None,
            "details": "no_level_within_buffer",
        }

    return {
        "name": "htf_proximity",
        "score": round(_clamp(best_score, 0.0, 1.0), 4),
        "nearest_level": nearest_name,
        "distance_pips": round(float(nearest_dist), 2),
        "details": "ok",
    }


def _compute_weighted_score(
    *,
    components: dict,
    w_rsi: int,
    w_adr: int,
    w_htf: int,
    w_spread: int,
) -> tuple[float, dict]:
    w = {
        "rsi_divergence": max(0, int(w_rsi)),
        "adr_exhaustion": max(0, int(w_adr)),
        "htf_proximity": max(0, int(w_htf)),
        "ema_spread": max(0, int(w_spread)),
    }
    total_w = sum(w.values())
    if total_w <= 0:
        w = {"rsi_divergence": 55, "adr_exhaustion": 20, "htf_proximity": 15, "ema_spread": 10}
        total_w = 100

    raw = (
        float(components["rsi_divergence"]["score"]) * w["rsi_divergence"]
        + float(components["adr_exhaustion"]["score"]) * w["adr_exhaustion"]
        + float(components["htf_proximity"]["score"]) * w["htf_proximity"]
        + float(components["ema_spread"]["score"]) * w["ema_spread"]
    ) / float(total_w)
    return _clamp(raw * 100.0, 0.0, 100.0), w


def _score_to_tier(score: float, medium: float, high: float, critical: float) -> str:
    if score < medium:
        return "low"
    if score < high:
        return "medium"
    if score < critical:
        return "high"
    return "critical"


def _tier_at_or_above(current_tier: str, threshold_tier: str) -> bool:
    return _tier_rank(current_tier) >= _tier_rank(threshold_tier)


def build_reversal_risk_response(policy, tier: str) -> dict:
    tier = str(tier or "low").lower()
    if tier not in _RISK_TIERS:
        tier = "low"

    lot_multiplier = 1.0
    min_tier_ema: Optional[int] = None
    if tier == "medium":
        lot_multiplier = max(0.01, float(getattr(policy, "rr_medium_lot_multiplier", 0.75)))
    elif tier == "high":
        lot_multiplier = max(0.01, float(getattr(policy, "rr_high_lot_multiplier", 0.50)))
        min_tier_ema = int(getattr(policy, "rr_high_min_tier_ema", 21))
    elif tier == "critical":
        lot_multiplier = max(0.01, float(getattr(policy, "rr_critical_lot_multiplier", 0.25)))
        min_tier_ema = int(getattr(policy, "rr_critical_min_tier_ema", 26))

    zone_block_threshold = str(getattr(policy, "rr_block_zone_entry_above_tier", "high")).lower()
    if zone_block_threshold not in ("medium", "high", "critical"):
        zone_block_threshold = "high"
    block_zone_entry = _tier_at_or_above(tier, zone_block_threshold)

    managed_exit_threshold = str(getattr(policy, "rr_use_managed_exit_at", "high")).lower()
    if managed_exit_threshold not in ("medium", "high", "critical"):
        managed_exit_threshold = "high"
    use_managed_exit = _tier_at_or_above(tier, managed_exit_threshold)

    return {
        "allowed": True,
        "lot_multiplier": round(float(lot_multiplier), 4),
        "min_tier_ema": int(min_tier_ema) if min_tier_ema is not None else None,
        "block_zone_entry": bool(block_zone_entry),
        "use_managed_exit": bool(use_managed_exit),
        "reason": (
            f"tier={tier} lot_x{lot_multiplier:.2f}"
            + (f" min_tier_ema>={int(min_tier_ema)}" if min_tier_ema is not None else "")
            + (" zone_block=ON" if block_zone_entry else " zone_block=OFF")
            + (" managed_exit=ON" if use_managed_exit else " managed_exit=OFF")
        ),
    }


def get_rr_exhaustion_threshold_boost_pips(policy, tier: str) -> float:
    if not bool(getattr(policy, "rr_adjust_exhaustion_thresholds", True)):
        return 0.0
    t = str(tier or "low").lower()
    if t == "medium":
        return max(0.0, float(getattr(policy, "rr_exhaustion_medium_threshold_boost_pips", 0.5)))
    if t == "high":
        return max(0.0, float(getattr(policy, "rr_exhaustion_high_threshold_boost_pips", 1.0)))
    if t == "critical":
        return max(0.0, float(getattr(policy, "rr_exhaustion_critical_threshold_boost_pips", 1.5)))
    return 0.0


def compute_reversal_risk_score(
    *,
    policy,
    data_by_tf: dict,
    tick,
    pip_size: float,
    trend_side: str,
    trial7_daily_state: Optional[dict] = None,
) -> dict:
    """Compute calibrated Trial #7 reversal-risk score and adaptive response."""
    m5_df = data_by_tf.get("M5")
    m15_df = data_by_tf.get("M15")
    h4_df = data_by_tf.get("H4")
    d_df = data_by_tf.get("D")
    current_price = (float(tick.bid) + float(tick.ask)) / 2.0

    regime = compute_regime(
        m5_df,
        adx_period=int(getattr(policy, "rr_regime_adx_period", 14)),
        trending_threshold=float(getattr(policy, "rr_regime_trending_threshold", 25.0)),
        ranging_threshold=float(getattr(policy, "rr_regime_ranging_threshold", 20.0)),
        atr_ratio_lookback=int(getattr(policy, "rr_regime_atr_ratio_lookback", 20)),
        atr_ratio_trending=float(getattr(policy, "rr_regime_atr_ratio_trending", 1.2)),
        use_atr_fallback=bool(getattr(policy, "rr_regime_use_atr_fallback", False)),
    )

    components = {
        "rsi_divergence": _compute_rsi_divergence_component(
            m5_df=m5_df,
            trend_side=trend_side,
            rsi_period=int(getattr(policy, "rr_rsi_period", 9)),
            lookback=int(getattr(policy, "rr_rsi_lookback_bars", 20)),
            severity_midpoint=float(getattr(policy, "rr_rsi_severity_midpoint", 18.0)),
            use_rolling_fallback=bool(getattr(policy, "rr_rsi_use_rolling_fallback", True)),
            require_confirmation_bar=bool(getattr(policy, "rr_rsi_require_confirmation_bar", True)),
            min_delta_for_score=float(getattr(policy, "rr_rsi_min_delta_for_score", 8.0)),
            min_pivot_separation_bars=int(getattr(policy, "rr_rsi_min_pivot_separation_bars", 5)),
        ),
        "adr_exhaustion": _compute_adr_exhaustion_component(
            d_df=d_df,
            trial7_daily_state=trial7_daily_state,
            pip_size=float(pip_size),
            adr_period=int(getattr(policy, "rr_adr_period", 14)),
            ramp_start_pct=float(getattr(policy, "rr_adr_ramp_start_pct", 75.0)),
            score_100=float(getattr(policy, "rr_adr_score_at_100_pct", 0.3)),
            score_120=float(getattr(policy, "rr_adr_score_at_120_pct", 0.6)),
            score_150=float(getattr(policy, "rr_adr_score_at_150_pct", 0.9)),
        ),
        "htf_proximity": _compute_htf_proximity_component(
            m15_df=m15_df,
            d_df=d_df,
            h4_df=h4_df,
            trial7_daily_state=trial7_daily_state,
            current_price=current_price,
            trend_side=trend_side,
            pip_size=float(pip_size),
            buffer_prev_day_pips=float(getattr(policy, "rr_htf_buffer_prev_day_pips", 8.0)),
            buffer_round_pips=float(getattr(policy, "rr_htf_buffer_round_pips", 5.0)),
            buffer_swing_pips=float(getattr(policy, "rr_htf_buffer_swing_pips", 6.0)),
            swing_lookback=int(getattr(policy, "rr_htf_swing_lookback", 30)),
            score_decay_pips=float(getattr(policy, "rr_htf_score_decay_pips", 6.0)),
            use_h4_levels=bool(getattr(policy, "rr_htf_use_h4_levels", True)),
        ),
        "ema_spread": _compute_ema_spread_component(
            m5_df=m5_df,
            pip_size=float(pip_size),
            threshold_pips=float(getattr(policy, "rr_ema_spread_threshold_pips", 4.22)),
            max_pips=float(getattr(policy, "rr_ema_spread_max_pips", 8.0)),
        ),
    }

    w_rsi = int(getattr(policy, "rr_weight_rsi_divergence", 55))
    w_adr = int(getattr(policy, "rr_weight_adr_exhaustion", 20))
    w_htf = int(getattr(policy, "rr_weight_htf_proximity", 15))
    w_spread = int(getattr(policy, "rr_weight_ema_spread", 10))

    if bool(getattr(policy, "rr_regime_adaptive_enabled", True)):
        if regime == "trending":
            w_rsi = max(1, int(w_rsi * float(getattr(policy, "rr_regime_trending_rsi_weight_mult", 0.8))))
            w_htf = max(1, int(w_htf * float(getattr(policy, "rr_regime_trending_htf_weight_mult", 0.7))))
        elif regime == "ranging":
            w_rsi = max(1, int(w_rsi * float(getattr(policy, "rr_regime_ranging_rsi_weight_mult", 1.2))))
            w_htf = max(1, int(w_htf * float(getattr(policy, "rr_regime_ranging_htf_weight_mult", 1.3))))

    score, weights = _compute_weighted_score(
        components=components,
        w_rsi=w_rsi,
        w_adr=w_adr,
        w_htf=w_htf,
        w_spread=w_spread,
    )

    t_medium = float(getattr(policy, "rr_tier_medium", 58.0))
    t_high = float(getattr(policy, "rr_tier_high", 65.0))
    t_critical = float(getattr(policy, "rr_tier_critical", 71.0))
    if bool(getattr(policy, "rr_regime_adaptive_enabled", True)):
        if regime == "trending":
            if getattr(policy, "rr_regime_trending_tier_medium", None) is not None:
                t_medium = float(policy.rr_regime_trending_tier_medium)
            if getattr(policy, "rr_regime_trending_tier_high", None) is not None:
                t_high = float(policy.rr_regime_trending_tier_high)
            if getattr(policy, "rr_regime_trending_tier_critical", None) is not None:
                t_critical = float(policy.rr_regime_trending_tier_critical)
        elif regime == "ranging":
            if getattr(policy, "rr_regime_ranging_tier_medium", None) is not None:
                t_medium = float(policy.rr_regime_ranging_tier_medium)
            if getattr(policy, "rr_regime_ranging_tier_high", None) is not None:
                t_high = float(policy.rr_regime_ranging_tier_high)
            if getattr(policy, "rr_regime_ranging_tier_critical", None) is not None:
                t_critical = float(policy.rr_regime_ranging_tier_critical)

    tier = _score_to_tier(score, t_medium, t_high, t_critical)
    response = build_reversal_risk_response(policy, tier)

    now_utc = pd.Timestamp.now(tz="UTC")
    return {
        "enabled": bool(getattr(policy, "use_reversal_risk_score", False)),
        "score": round(float(score), 2),
        "tier": tier,
        "regime": regime,
        "trend_side": str(trend_side or "").lower(),
        "session": _session_name_utc(now_utc),
        "weights": weights,
        "components": components,
        "response": response,
        "thresholds": {
            "medium": t_medium,
            "high": t_high,
            "critical": t_critical,
        },
    }
