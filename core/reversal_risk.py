from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

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


def _find_price_rsi_divergence(
    close: pd.Series,
    rsi: pd.Series,
    lookback: int,
) -> dict:
    """Simple pivot-based divergence detection on latest lookback bars.

    Returns both bullish/bearish divergence deltas so caller can select by trend side.
    """
    if close is None or rsi is None or len(close) < max(lookback, 8):
        return {
            "bearish_found": False,
            "bullish_found": False,
            "bearish_delta": 0.0,
            "bullish_delta": 0.0,
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
        }

    highs: list[int] = []
    lows: list[int] = []
    wing = 2
    for i in range(wing, n - wing):
        w = c.iloc[i - wing: i + wing + 1]
        v = float(c.iloc[i])
        if v >= float(w.max()):
            highs.append(i)
        if v <= float(w.min()):
            lows.append(i)

    bearish_found = False
    bullish_found = False
    bearish_delta = 0.0
    bullish_delta = 0.0

    if len(highs) >= 2:
        i1, i2 = highs[-2], highs[-1]
        p1, p2 = float(c.iloc[i1]), float(c.iloc[i2])
        r1, r2 = float(rv.iloc[i1]), float(rv.iloc[i2])
        if p2 > p1 and r2 < r1:
            bearish_found = True
            bearish_delta = abs(r1 - r2)

    if len(lows) >= 2:
        i1, i2 = lows[-2], lows[-1]
        p1, p2 = float(c.iloc[i1]), float(c.iloc[i2])
        r1, r2 = float(rv.iloc[i1]), float(rv.iloc[i2])
        if p2 < p1 and r2 > r1:
            bullish_found = True
            bullish_delta = abs(r2 - r1)

    return {
        "bearish_found": bearish_found,
        "bullish_found": bullish_found,
        "bearish_delta": round(bearish_delta, 3),
        "bullish_delta": round(bullish_delta, 3),
    }


def _compute_rsi_divergence_component(
    *,
    m5_df: Optional[pd.DataFrame],
    trend_side: str,
    rsi_period: int,
    lookback: int,
    severity_midpoint: float,
) -> dict:
    if m5_df is None or m5_df.empty or len(m5_df) < max(lookback + 2, rsi_period + 5):
        return {
            "name": "rsi_divergence",
            "score": 0.0,
            "found": False,
            "severity": 0.0,
            "direction": None,
            "details": "insufficient_m5_data",
        }

    close = m5_df["close"].astype(float)
    rsi = rsi_fn(close, max(2, int(rsi_period)))
    div = _find_price_rsi_divergence(close, rsi, max(8, int(lookback)))

    side = str(trend_side or "bull").lower()
    if side == "bull":
        found = bool(div["bearish_found"])
        delta = float(div["bearish_delta"])
        direction = "bearish"
    else:
        found = bool(div["bullish_found"])
        delta = float(div["bullish_delta"])
        direction = "bullish"

    # Calibration: 18 RSI points -> 0.5 severity, 36 -> 1.0
    denom = max(0.1, float(severity_midpoint) * 2.0)
    severity = _clamp(delta / denom, 0.0, 1.0) if found else 0.0

    return {
        "name": "rsi_divergence",
        "score": severity,
        "found": found,
        "severity": round(severity, 4),
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


def _round_number_levels(price: float) -> list[tuple[str, float]]:
    levels: list[tuple[str, float]] = []
    whole = int(price)
    for base in (whole - 1, whole, whole + 1, whole + 2):
        levels.append((f"round_{base}.000", float(base)))
        levels.append((f"round_{base}.500", float(base) + 0.5))
    return levels


def _compute_htf_proximity_component(
    *,
    m15_df: Optional[pd.DataFrame],
    d_df: Optional[pd.DataFrame],
    trial7_daily_state: Optional[dict],
    current_price: float,
    trend_side: str,
    pip_size: float,
    buffer_pips: float,
    swing_lookback: int,
) -> dict:
    levels: list[tuple[str, float]] = []

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
        levels.append(("prev_day_high", float(prev_day_high)))
    if prev_day_low is not None:
        levels.append(("prev_day_low", float(prev_day_low)))

    levels.extend(_round_number_levels(float(current_price)))

    if m15_df is not None and not m15_df.empty:
        m15 = m15_df.copy().sort_values("time")
        lookback = max(5, int(swing_lookback))
        window = m15.tail(lookback)
        try:
            swing_high = float(window["high"].astype(float).max())
            swing_low = float(window["low"].astype(float).min())
            levels.append(("m15_swing_high", swing_high))
            levels.append(("m15_swing_low", swing_low))
        except Exception:
            pass

    side = str(trend_side or "bull").lower()
    opp: list[tuple[str, float]] = []
    for name, lv in levels:
        if side == "bull":
            if lv >= current_price:
                opp.append((name, lv))
        else:
            if lv <= current_price:
                opp.append((name, lv))

    if not opp:
        return {
            "name": "htf_proximity",
            "score": 0.0,
            "nearest_level": None,
            "distance_pips": None,
            "details": "no_opposing_levels",
        }

    nearest_name = None
    nearest_dist = None
    for name, lv in opp:
        dist = abs(float(lv) - float(current_price)) / float(pip_size)
        if nearest_dist is None or dist < nearest_dist:
            nearest_dist = dist
            nearest_name = name

    if nearest_dist is None:
        return {
            "name": "htf_proximity",
            "score": 0.0,
            "nearest_level": None,
            "distance_pips": None,
            "details": "distance_calc_failed",
        }

    score = _clamp(1.0 - (nearest_dist / max(0.1, float(buffer_pips))), 0.0, 1.0)
    return {
        "name": "htf_proximity",
        "score": round(score, 4),
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
    d_df = data_by_tf.get("D")
    current_price = (float(tick.bid) + float(tick.ask)) / 2.0

    components = {
        "rsi_divergence": _compute_rsi_divergence_component(
            m5_df=m5_df,
            trend_side=trend_side,
            rsi_period=int(getattr(policy, "rr_rsi_period", 9)),
            lookback=int(getattr(policy, "rr_rsi_lookback_bars", 20)),
            severity_midpoint=float(getattr(policy, "rr_rsi_severity_midpoint", 18.0)),
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
            trial7_daily_state=trial7_daily_state,
            current_price=current_price,
            trend_side=trend_side,
            pip_size=float(pip_size),
            buffer_pips=float(getattr(policy, "rr_htf_buffer_pips", 5.0)),
            swing_lookback=int(getattr(policy, "rr_htf_swing_lookback", 30)),
        ),
        "ema_spread": _compute_ema_spread_component(
            m5_df=m5_df,
            pip_size=float(pip_size),
            threshold_pips=float(getattr(policy, "rr_ema_spread_threshold_pips", 4.22)),
            max_pips=float(getattr(policy, "rr_ema_spread_max_pips", 8.0)),
        ),
    }

    score, weights = _compute_weighted_score(
        components=components,
        w_rsi=int(getattr(policy, "rr_weight_rsi_divergence", 55)),
        w_adr=int(getattr(policy, "rr_weight_adr_exhaustion", 20)),
        w_htf=int(getattr(policy, "rr_weight_htf_proximity", 15)),
        w_spread=int(getattr(policy, "rr_weight_ema_spread", 10)),
    )

    tier = _score_to_tier(
        score,
        float(getattr(policy, "rr_tier_medium", 58.0)),
        float(getattr(policy, "rr_tier_high", 65.0)),
        float(getattr(policy, "rr_tier_critical", 71.0)),
    )
    response = build_reversal_risk_response(policy, tier)

    now_utc = pd.Timestamp.now(tz="UTC")
    return {
        "enabled": bool(getattr(policy, "use_reversal_risk_score", False)),
        "score": round(float(score), 2),
        "tier": tier,
        "trend_side": str(trend_side or "").lower(),
        "session": _session_name_utc(now_utc),
        "weights": weights,
        "components": components,
        "response": response,
        "thresholds": {
            "medium": float(getattr(policy, "rr_tier_medium", 58.0)),
            "high": float(getattr(policy, "rr_tier_high", 65.0)),
            "critical": float(getattr(policy, "rr_tier_critical", 71.0)),
        },
    }
