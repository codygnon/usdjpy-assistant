"""Runner Score v2 for Trial #10.

Predicts likelihood of a trade becoming a 60m+ profitable runner based on
entry-time features. Derived from 250k backtest attribution analysis.

NOT wired into sizing in v1 — output is for logging, dashboard, and
backtest validation only.

Inputs (all knowable at entry time):
    1. ATR stop pips  — hard floor gate (>12p = floor bucket)
    2. Regime label    — buy_boost_hour is the strongest contextual edge
    3. M5 bucket       — weak/normal > strong (counterintuitive but validated)
    4. Structure ratio  — 0.4–0.6 sweet spot from pullback quality

Elite overlay (trend freshness):
    5. Bars since M1 5/9 cross — how recently momentum turned
    6. Prior same-side entries since last M5 9/21 recross — trend exploitation count

Bucket mapping:
    floor    — ATR > 12p (forced, ~6% runner rate in backtest)
    base     — ATR <= 12p, 0 bonus points (~19% runner rate)
    elevated — ATR <= 12p, 1 bonus point (~24% runner rate)
    press    — ATR <= 12p, 2–3 bonus points (~29% runner rate)
    elite    — press + fresh trend (+0.49 pips after spread, 58.9% runner rate)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


RUNNER_BUCKETS = ("floor", "base", "elevated", "press", "elite")

# ATR gate threshold (pips). Above this → floor bucket regardless.
ATR_GATE_PIPS = 12.0

# Freshness thresholds (strict — validated on 56-trade slice)
FRESHNESS_MAX_BARS_SINCE_CROSS_STRICT = 5
FRESHNESS_MAX_PRIOR_ENTRIES_STRICT = 1

# Freshness thresholds (relaxed — for validation testing)
FRESHNESS_MAX_BARS_SINCE_CROSS_RELAXED = 7
FRESHNESS_MAX_PRIOR_ENTRIES_RELAXED = 2


@dataclass
class RunnerScoreResult:
    bucket: str = "floor"          # floor | base | elevated | press | elite
    points: int = 0                # 0–3 bonus points (meaningful only when not floor)
    atr_eligible: bool = False     # True if ATR <= gate
    regime_point: bool = False     # True if regime == buy_boost_hour
    m5_point: bool = False         # True if m5_bucket in {weak, normal}
    structure_point: bool = False  # True if 0.4 <= structure_ratio <= 0.6
    fresh: bool = False            # True if trend freshness qualifies for elite
    bars_since_cross: Optional[int] = None   # M1 5/9 cross recency
    prior_entries: Optional[int] = None      # same-side entries since M5 9/21 recross
    freshness_mode: str = "strict"           # strict | relaxed
    atr_stop_pips: float = 0.0
    regime_label: str = ""
    m5_bucket: str = ""
    structure_ratio: Optional[float] = None


def compute_runner_score(
    *,
    atr_stop_pips: float,
    regime_label: str,
    m5_bucket: str,
    structure_ratio: Optional[float] = None,
    bars_since_cross: Optional[int] = None,
    prior_entries: Optional[int] = None,
    freshness_mode: str = "strict",
) -> RunnerScoreResult:
    """Compute runner score from entry-time features.

    Parameters
    ----------
    atr_stop_pips : float
        ATR-derived stop distance in pips (from _resolve_trial10_stop_pips).
    regime_label : str
        Operator regime label (buy_boost_hour / buy_base_hour / sell_base_hour).
    m5_bucket : str
        M5 regime bucket (strong / normal / weak).
    structure_ratio : float or None
        Pullback quality structure ratio (0.0–1.0). None for zone entries.
    bars_since_cross : int or None
        Bars since last M1 EMA5/9 cross in trend direction. None if unknown.
    prior_entries : int or None
        Same-side entries since last M5 EMA9/21 recross. None if unknown.
    freshness_mode : str
        "strict" (0-5 bars, 0-1 entries) or "relaxed" (0-7 bars, 0-2 entries).

    Returns
    -------
    RunnerScoreResult with bucket, points, freshness, and per-feature flags.
    """
    result = RunnerScoreResult(
        atr_stop_pips=round(float(atr_stop_pips), 2),
        regime_label=str(regime_label),
        m5_bucket=str(m5_bucket),
        structure_ratio=round(float(structure_ratio), 3) if structure_ratio is not None else None,
        bars_since_cross=int(bars_since_cross) if bars_since_cross is not None else None,
        prior_entries=int(prior_entries) if prior_entries is not None else None,
        freshness_mode="relaxed" if str(freshness_mode).lower() == "relaxed" else "strict",
    )

    # --- Gate: ATR > 12p → floor ---
    if atr_stop_pips > ATR_GATE_PIPS:
        result.bucket = "floor"
        result.atr_eligible = False
        return result

    result.atr_eligible = True
    points = 0

    # +1 for buy_boost_hour
    if str(regime_label).lower() == "buy_boost_hour":
        result.regime_point = True
        points += 1

    # +1 for weak/normal M5 (not strong)
    if str(m5_bucket).lower() in ("weak", "normal"):
        result.m5_point = True
        points += 1

    # +1 for structure ratio sweet spot 0.4–0.6
    if structure_ratio is not None and 0.4 <= float(structure_ratio) <= 0.6:
        result.structure_point = True
        points += 1

    result.points = points

    if points >= 2:
        result.bucket = "press"
    elif points == 1:
        result.bucket = "elevated"
    else:
        result.bucket = "base"

    # --- Elite overlay: freshness on top of press ---
    if result.bucket == "press" and bars_since_cross is not None and prior_entries is not None:
        if result.freshness_mode == "relaxed":
            max_bars = FRESHNESS_MAX_BARS_SINCE_CROSS_RELAXED
            max_entries = FRESHNESS_MAX_PRIOR_ENTRIES_RELAXED
        else:
            max_bars = FRESHNESS_MAX_BARS_SINCE_CROSS_STRICT
            max_entries = FRESHNESS_MAX_PRIOR_ENTRIES_STRICT

        if int(bars_since_cross) <= max_bars and int(prior_entries) <= max_entries:
            result.fresh = True
            result.bucket = "elite"

    return result


def compute_freshness(
    *,
    m1_close: "pd.Series",
    m5_close: "pd.Series",
    side: str,
    trades_df: "Optional[pd.DataFrame]" = None,
    policy_type: str = "kt_cg_trial_10",
) -> tuple[Optional[int], Optional[int]]:
    """Compute trend freshness inputs for elite gate.

    Parameters
    ----------
    m1_close : pd.Series
        M1 close prices (at least 10 bars).
    m5_close : pd.Series
        M5 close prices (at least 22 bars).
    side : str
        "buy" or "sell".
    trades_df : pd.DataFrame or None
        Trade history with columns: side, entry_timestamp_utc, policy_type.
    policy_type : str
        Filter trades to this policy type.

    Returns
    -------
    (bars_since_cross, prior_entries) — None if data insufficient.
    """
    import pandas as pd

    if len(m1_close) < 10 or len(m5_close) < 22:
        return None, None

    # --- Bars since last M1 EMA5/9 cross in trend direction ---
    m1_f = m1_close.astype(float)
    ema5 = m1_f.ewm(span=5, adjust=False).mean()
    ema9 = m1_f.ewm(span=9, adjust=False).mean()
    diff = ema5 - ema9

    bars_since_cross: Optional[int] = None
    vals = diff.values
    if side == "buy":
        for i in range(len(vals) - 1, 0, -1):
            if vals[i] > 0 and vals[i - 1] <= 0:
                bars_since_cross = len(vals) - 1 - i
                break
    else:
        for i in range(len(vals) - 1, 0, -1):
            if vals[i] < 0 and vals[i - 1] >= 0:
                bars_since_cross = len(vals) - 1 - i
                break

    # --- Prior same-side entries since last M5 EMA9/21 recross ---
    m5_f = m5_close.astype(float)
    e9 = m5_f.ewm(span=9, adjust=False).mean()
    e21 = m5_f.ewm(span=21, adjust=False).mean()
    m5_diff = e9 - e21

    # Find the M5 bar time of the last recross into the current trend direction
    m5_vals = m5_diff.values
    recross_idx: Optional[int] = None
    if side == "buy":
        for i in range(len(m5_vals) - 1, 0, -1):
            if m5_vals[i] > 0 and m5_vals[i - 1] <= 0:
                recross_idx = i
                break
    else:
        for i in range(len(m5_vals) - 1, 0, -1):
            if m5_vals[i] < 0 and m5_vals[i - 1] >= 0:
                recross_idx = i
                break

    prior_entries: Optional[int] = None
    if recross_idx is not None and trades_df is not None and not trades_df.empty:
        recross_time = m5_close.index[recross_idx]
        if hasattr(recross_time, "tz") and recross_time.tz is None:
            recross_time = pd.Timestamp(recross_time).tz_localize("UTC")
        else:
            recross_time = pd.Timestamp(recross_time)

        df = trades_df.copy()
        if "policy_type" in df.columns:
            df = df[df["policy_type"].astype(str) == policy_type]
        df = df[df["side"].astype(str).str.lower() == side.lower()]
        if "entry_timestamp_utc" in df.columns:
            df["_entry_dt"] = pd.to_datetime(df["entry_timestamp_utc"], utc=True, errors="coerce")
            df = df[df["_entry_dt"].notna()]
            prior_entries = int((df["_entry_dt"] >= recross_time).sum())
        else:
            prior_entries = 0
    elif recross_idx is not None:
        prior_entries = 0

    return bars_since_cross, prior_entries


def runner_score_snapshot(result: RunnerScoreResult) -> dict:
    """Serialise a RunnerScoreResult for logging / dashboard / CSV."""
    return {
        "runner_bucket": result.bucket,
        "runner_points": result.points,
        "atr_eligible": result.atr_eligible,
        "regime_point": result.regime_point,
        "m5_point": result.m5_point,
        "structure_point": result.structure_point,
        "fresh": result.fresh,
        "bars_since_cross": result.bars_since_cross,
        "prior_entries": result.prior_entries,
        "freshness_mode": result.freshness_mode,
        "atr_stop_pips": result.atr_stop_pips,
        "regime_label": result.regime_label,
        "m5_bucket": result.m5_bucket,
        "structure_ratio": result.structure_ratio,
    }


def runner_score_to_lots(
    *,
    result: RunnerScoreResult,
    bucket_lots: dict[str, float],
    regime_multiplier: float = 1.0,
    spread_pips: float = 0.0,
    spread_gate_pips: float = 3.0,
    tier: Optional[int] = None,
    is_boost_hour: bool = False,
    force_floor_tier17_nonboost: bool = True,
    min_lots: float = 0.03,
    max_lots: float = 0.50,
) -> tuple[float, dict]:
    """Map a RunnerScoreResult to lot size with spread gate and tier17 override.

    Returns (final_lots, audit_dict) with full trail for logging/dashboard.
    """
    floor_lots = bucket_lots.get("floor", min_lots)
    base_lots = bucket_lots.get(result.bucket, floor_lots)

    spread_gated = False
    tier17_floor_applied = False

    # Spread gate: wide spread → force floor
    if spread_pips > spread_gate_pips:
        base_lots = floor_lots
        spread_gated = True

    # Tier 17 non-boost: force floor
    if tier == 17 and not is_boost_hour and force_floor_tier17_nonboost:
        base_lots = floor_lots
        tier17_floor_applied = True

    pre_clamp = round(base_lots * regime_multiplier, 4)
    final = round(min(max_lots, max(min_lots, pre_clamp)), 2)

    audit = {
        "runner_bucket": result.bucket,
        "bucket_base_lots": round(bucket_lots.get(result.bucket, floor_lots), 4),
        "bucket_lots_floor": round(float(bucket_lots.get("floor", floor_lots)), 4),
        "bucket_lots_base": round(float(bucket_lots.get("base", floor_lots)), 4),
        "bucket_lots_elevated": round(float(bucket_lots.get("elevated", floor_lots)), 4),
        "bucket_lots_press": round(float(bucket_lots.get("press", floor_lots)), 4),
        "bucket_lots_elite": round(float(bucket_lots.get("elite", floor_lots)), 4),
        "spread_gate_pips": round(float(spread_gate_pips), 4),
        "spread_gated": spread_gated,
        "tier17_floor_applied": tier17_floor_applied,
        "regime_multiplier": round(regime_multiplier, 4),
        "pre_clamp_lots": pre_clamp,
        "final_lots": final,
    }
    return final, audit
