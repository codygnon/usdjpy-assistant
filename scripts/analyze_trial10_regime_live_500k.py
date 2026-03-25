#!/usr/bin/env python3
from __future__ import annotations

import json
import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(REPO_ROOT))

from core.fib_pivots import compute_previous_candle_fib_levels, resample_intraday_ohlc
from core.no_trade_zone import IntradayFibCorridorFilter
from core.presets import PresetId, apply_preset
from core.profile import ExecutionPolicyKtCgTrial10, default_profile_for_name, get_effective_risk
from core.regime_gate import check_chop_pause, evaluate_regime_gate
from core.runner_score import compute_runner_score, runner_score_to_lots
from scripts.backtest_trial10 import (
    PIP_SIZE,
    Position,
    choose_first_event,
    close_leg,
    hour_et,
    load_m1,
    map_features_to_m1,
    pips_to_usd,
    prepare_features,
    resample_ohlc,
    unrealized_usd,
)


OUTPUT_JSON = REPO_ROOT / "research_out" / "trial10_regime_live_500k.json"
OUTPUT_MD = REPO_ROOT / "research_out" / "trial10_regime_live_500k.md"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest live-style Trial 10 regime variants on 500k M1.")
    p.add_argument(
        "--input-csv",
        default=str(REPO_ROOT / "research_out" / "USDJPY_M1_OANDA_500k.csv"),
        help="Input M1 CSV path.",
    )
    p.add_argument(
        "--max-bars",
        type=int,
        default=0,
        help="Optional max number of M1 bars to load from the tail of the CSV for profiling/debug.",
    )
    p.add_argument(
        "--variants",
        default="both_off,london_on_chop_off,london_off_chop_on,both_on",
        help="Comma-separated variant names to run.",
    )
    p.add_argument("--out-json", default=str(OUTPUT_JSON), help="Summary JSON output path.")
    p.add_argument("--out-md", default=str(OUTPUT_MD), help="Summary Markdown output path.")
    return p.parse_args()


@dataclass(frozen=True)
class Variant:
    name: str
    london_sell_veto: bool
    chop_pause_enabled: bool


@dataclass
class LivePosition(Position):
    runner_bucket: str = "floor"
    exit_profile: str = "standard"
    managed_trail_mode: str = "hwm"
    trail_escalation_level: str = ""
    peak_price: Optional[float] = None
    be_plus_pips: float = 0.5


def _pullback_quality_snapshot_fast(
    *,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    touch_index: int,
    side: str,
    tier: int,
    policy: ExecutionPolicyKtCgTrial10,
) -> Optional[dict]:
    enabled = bool(getattr(policy, "pullback_quality_enabled", True))
    shallow_tiers = tuple(int(x) for x in getattr(policy, "pullback_quality_shallow_tier_periods", (17, 21)))
    applicable = tier in shallow_tiers
    if not enabled:
        return {
            "enabled": False,
            "applicable": applicable,
            "side": side,
            "tier": tier,
            "label": "neutral",
            "pullback_bar_count": 0,
            "structure_ratio": 0.0,
            "dampener_multiplier": 1.0,
            "dampener_applied": False,
            "reason": "pullback_quality_disabled",
        }
    if touch_index <= 0:
        return None

    lookback_bars = max(1, int(getattr(policy, "pullback_quality_lookback_bars", 30)))
    orderly_bar_count_min = int(getattr(policy, "pullback_quality_orderly_bar_count_min", 6))
    sloppy_bar_count_max = int(getattr(policy, "pullback_quality_sloppy_bar_count_max", 2))
    orderly_structure_ratio_min = float(getattr(policy, "pullback_quality_orderly_structure_ratio_min", 0.6))
    sloppy_structure_ratio_max = float(getattr(policy, "pullback_quality_sloppy_structure_ratio_max", 0.3))
    sloppy_lot_multiplier = float(getattr(policy, "pullback_quality_sloppy_lot_multiplier", 0.5))

    window_start = max(0, touch_index - lookback_bars)
    if side == "bull":
        window = high_arr[window_start : touch_index + 1]
        if window.size == 0:
            return None
        swing_rel = int(np.max(np.flatnonzero(window == np.max(window))))
        pullback_vals = high_arr[window_start + swing_rel : touch_index + 1]
        orderly_steps = int(np.sum(pullback_vals[1:] < pullback_vals[:-1])) if pullback_vals.size > 1 else 0
    else:
        window = low_arr[window_start : touch_index + 1]
        if window.size == 0:
            return None
        swing_rel = int(np.max(np.flatnonzero(window == np.min(window))))
        pullback_vals = low_arr[window_start + swing_rel : touch_index + 1]
        orderly_steps = int(np.sum(pullback_vals[1:] > pullback_vals[:-1])) if pullback_vals.size > 1 else 0
    swing_index = window_start + swing_rel
    pullback_bar_count = max(0, touch_index - swing_index)
    comparisons = max(0, pullback_vals.size - 1)
    structure_ratio = float(orderly_steps / comparisons) if comparisons > 0 else 0.0

    if pullback_bar_count >= orderly_bar_count_min and structure_ratio >= orderly_structure_ratio_min:
        label = "orderly"
    elif pullback_bar_count <= sloppy_bar_count_max or structure_ratio < sloppy_structure_ratio_max:
        label = "sloppy"
    else:
        label = "neutral"

    dampener_applied = applicable and label == "sloppy"
    dampener_multiplier = sloppy_lot_multiplier if dampener_applied else 1.0
    return {
        "enabled": True,
        "applicable": applicable,
        "side": side,
        "tier": tier,
        "label": label,
        "pullback_bar_count": pullback_bar_count,
        "structure_ratio": round(structure_ratio, 4),
        "dampener_multiplier": dampener_multiplier,
        "dampener_applied": dampener_applied,
        "reason": f"{label}: {pullback_bar_count} bars since swing, structure_ratio={structure_ratio:.2f}",
    }


def _evaluate_signal_cached(
    *,
    m1f: pd.DataFrame,
    arrays: dict[str, object],
    i: int,
    tier_state: dict[int, bool],
    policy: ExecutionPolicyKtCgTrial10,
    pq_cache: dict[tuple[int, str, int], Optional[dict]],
) -> tuple[Optional[dict], Counter, dict[int, bool]]:
    reasons = Counter()
    tier_updates: dict[int, bool] = {}
    if i < 205:
        reasons["warmup"] += 1
        return None, reasons, tier_updates

    gap_pips = arrays["gap_pips"][i]
    if pd.isna(gap_pips) or float(gap_pips) < float(policy.m5_min_ema_distance_pips):
        reasons["m5_gap_block"] += 1
        return None, reasons, tier_updates

    trend = str(arrays["trend"][i] or "")
    if trend not in {"bull", "bear"}:
        reasons["m5_trend_missing"] += 1
        return None, reasons, tier_updates
    is_bull = trend == "bull"
    side_key = "bull" if is_bull else "bear"
    last_close = float(arrays["close"][i])
    prev_low = float(arrays["low"][i - 1])
    prev_high = float(arrays["high"][i - 1])
    reclaim_ema_now = float(arrays["reclaim_ema"][i])
    strong_only_tiers = {int(x) for x in getattr(policy, "strong_m5_only_tier_periods", tuple())}
    m5_bucket = str(arrays["m5_bucket"][i] or "normal")
    reset_buffer = float(policy.tier_reset_buffer_pips) * PIP_SIZE

    for tier in (int(x) for x in policy.tier_ema_periods):
        tier_ema = arrays["tier_ema"][tier]
        ema_now = float(tier_ema[i])
        ema_prev = float(tier_ema[i - 1])
        tier_fired = bool(tier_state.get(tier, False))
        requires_strong = tier in strong_only_tiers
        m5_ok = (not requires_strong) or m5_bucket == "strong"

        if is_bull:
            touched_prev = prev_low <= ema_prev
            reclaim_ok = (not policy.tier_reclaim_confirmation_enabled) or (last_close > reclaim_ema_now)
            moved_away = last_close > ema_now + reset_buffer
        else:
            touched_prev = prev_high >= ema_prev
            reclaim_ok = (not policy.tier_reclaim_confirmation_enabled) or (last_close < reclaim_ema_now)
            moved_away = last_close < ema_now - reset_buffer

        pq_snapshot = None
        if touched_prev and not tier_fired:
            cache_key = (i - 1, side_key, tier)
            if cache_key not in pq_cache:
                pq_cache[cache_key] = _pullback_quality_snapshot_fast(
                    high_arr=arrays["high"],
                    low_arr=arrays["low"],
                    touch_index=i - 1,
                    side=side_key,
                    tier=tier,
                    policy=policy,
                )
            pq_snapshot = pq_cache.get(cache_key)

        if moved_away and tier_fired:
            tier_updates[tier] = False

        if not touched_prev or tier_fired:
            continue
        if not m5_ok:
            reasons[f"tier_{tier}_needs_strong_m5"] += 1
            continue
        if not reclaim_ok:
            reasons[f"tier_{tier}_reclaim_failed"] += 1
            continue
        tier_updates[tier] = True
        return (
            {
                "side": "buy" if is_bull else "sell",
                "entry_type": "tiered_pullback",
                "tier": tier,
                "m5_bucket": m5_bucket,
                "pullback_quality": pq_snapshot,
            },
            reasons,
            tier_updates,
        )

    if not bool(policy.zone_entry_enabled):
        reasons["zone_disabled"] += 1
        return None, reasons, tier_updates

    zone_fast_arr = arrays["zone_fast"]
    zone_slow_arr = arrays["zone_slow"]
    zone_fast = float(zone_fast_arr[i])
    zone_slow = float(zone_slow_arr[i])
    aligned_now = zone_fast > zone_slow if is_bull else zone_fast < zone_slow
    recent_cross = False
    lookback = max(1, int(policy.zone_entry_max_cross_lookback_bars))
    for offset in range(1, min(lookback, i) + 1):
        prev_fast = float(zone_fast_arr[i - offset])
        prev_slow = float(zone_slow_arr[i - offset])
        curr_fast = float(zone_fast_arr[i - offset + 1])
        curr_slow = float(zone_slow_arr[i - offset + 1])
        if is_bull and prev_fast <= prev_slow and curr_fast > curr_slow:
            recent_cross = True
            break
        if (not is_bull) and prev_fast >= prev_slow and curr_fast < curr_slow:
            recent_cross = True
            break
    if aligned_now and ((not policy.zone_entry_require_recent_cross) or recent_cross):
        return (
            {
                "side": "buy" if is_bull else "sell",
                "entry_type": "zone_entry",
                "tier": None,
                "m5_bucket": m5_bucket,
                "pullback_quality": None,
            },
            reasons,
            tier_updates,
        )
    if aligned_now and policy.zone_entry_require_recent_cross and not recent_cross:
        reasons["zone_no_recent_cross"] += 1
    else:
        reasons["zone_alignment_missing"] += 1
    return None, reasons, tier_updates


def precompute_signal_sequence(
    *,
    m1f: pd.DataFrame,
    mapped: pd.DataFrame,
    policy: ExecutionPolicyKtCgTrial10,
) -> tuple[list[Optional[dict]], Counter]:
    signals: list[Optional[dict]] = [None] * len(mapped)
    block_counts = Counter()
    tier_state = {int(x): False for x in policy.tier_ema_periods}
    pq_cache: dict[tuple[int, str, int], Optional[dict]] = {}
    arrays: dict[str, object] = {
        "gap_pips": mapped["gap_pips"].to_numpy(),
        "trend": mapped["trend"].tolist(),
        "m5_bucket": mapped["m5_bucket"].tolist(),
        "close": m1f["close"].to_numpy(dtype=float),
        "low": m1f["low"].to_numpy(dtype=float),
        "high": m1f["high"].to_numpy(dtype=float),
        "reclaim_ema": m1f[f"ema_{int(policy.tier_reclaim_ema_period)}"].to_numpy(dtype=float),
        "zone_fast": m1f[f"ema_{int(policy.m1_zone_entry_ema_fast)}"].to_numpy(dtype=float),
        "zone_slow": m1f[f"ema_{int(policy.m1_zone_entry_ema_slow)}"].to_numpy(dtype=float),
        "tier_ema": {
            int(tier): m1f[f"ema_{int(tier)}"].to_numpy(dtype=float)
            for tier in policy.tier_ema_periods
        },
    }
    for i in range(1, len(mapped)):
        signal, signal_reason_counts, tier_updates = _evaluate_signal_cached(
            m1f=m1f,
            arrays=arrays,
            i=i,
            tier_state=tier_state,
            policy=policy,
            pq_cache=pq_cache,
        )
        block_counts.update(signal_reason_counts)
        for tier, value in tier_updates.items():
            tier_state[int(tier)] = bool(value)
        signals[i] = signal
    return signals, block_counts


def build_live_trial10_policy() -> tuple[object, ExecutionPolicyKtCgTrial10]:
    base = default_profile_for_name("trial10_live_500k")
    base = base.model_copy()
    base.symbol = "USDJPY"
    base.pip_size = PIP_SIZE
    base.deposit_amount = 100_000.0
    base.display_currency = "USD"
    base = base.model_copy(
        update={
            "risk": base.risk.model_copy(
                update={
                    "max_lots": 7.0,
                    "require_stop": True,
                    "min_stop_pips": 12.0,
                    "max_spread_pips": 6.0,
                    "max_trades_per_day": 100,
                    "max_open_trades": 20,
                    "cooldown_minutes_after_loss": 0,
                }
            )
        }
    )
    profile = apply_preset(base, PresetId.KT_CG_TRIAL_10)
    policy = profile.execution.policies[0]
    if not isinstance(policy, ExecutionPolicyKtCgTrial10):
        raise TypeError("Expected Trial 10 policy")
    target_max_open_trades_per_side = 5
    directional_cap = round(1.0 * float(target_max_open_trades_per_side) / 2.0, 2)
    policy = policy.model_copy(
        update={
            "exit_strategy": "tp1_be_hwm_trail",
            "hwm_trail_pips": 6.0,
            "conviction_sizing_enabled": False,
            "runner_score_sizing_enabled": True,
            "conviction_base_lots": 1.8,
            "conviction_min_lots": 0.03,
            "bucketed_exit_enabled": True,
            "trail_escalation_enabled": True,
            "intraday_fib_enabled": True,
            "intraday_fib_timeframe": "H1",
            "intraday_fib_lower_level": "S2",
            "intraday_fib_upper_level": "R2",
            "intraday_fib_boundary_buffer_pips": 1.0,
            "intraday_fib_hysteresis_pips": 1.5,
            "regime_london_sell_veto": False,
            "regime_chop_pause_enabled": True,
            "regime_chop_pause_minutes": 45,
            "regime_chop_pause_lookback_trades": 5,
            "regime_chop_pause_stop_rate": 0.6,
            "max_open_trades_per_side": target_max_open_trades_per_side,
            "max_zone_entry_open": 4,
            "max_tiered_pullback_open": 10,
            "max_directional_lots_per_side": directional_cap,
        }
    )
    return profile, policy


def bucket_exit_profile(policy: ExecutionPolicyKtCgTrial10, runner_bucket: str) -> dict[str, float | str]:
    exit_strategy = str(getattr(policy, "exit_strategy", "tp1_be_m5_trail") or "tp1_be_m5_trail")
    default_trail_mode = "m5"
    if exit_strategy == "tp1_be_trail":
        default_trail_mode = "m1"
    elif exit_strategy == "tp1_be_hwm_trail":
        default_trail_mode = "hwm"
    bucket = str(runner_bucket or "").lower()
    quick_buckets = {str(x).lower() for x in getattr(policy, "quick_exit_buckets", ("floor", "base"))}
    runner_buckets = {str(x).lower() for x in getattr(policy, "runner_exit_buckets", ("press", "elite"))}
    result = {
        "tp1_pips": float(getattr(policy, "tp1_pips", 6.0)),
        "tp1_close_pct": float(getattr(policy, "tp1_close_pct", 70.0)),
        "be_plus_pips": float(getattr(policy, "be_spread_plus_pips", 0.5)),
        "trail_mode": default_trail_mode,
        "exit_profile": "standard",
    }
    if not bool(getattr(policy, "bucketed_exit_enabled", False)):
        return result
    if bucket in quick_buckets:
        result.update(
            {
                "tp1_pips": float(getattr(policy, "quick_tp1_pips", 4.0)),
                "tp1_close_pct": float(getattr(policy, "quick_tp1_close_pct", 85.0)),
                "be_plus_pips": float(getattr(policy, "quick_be_spread_plus_pips", 0.3)),
                "trail_mode": "m1",
                "exit_profile": "quick",
            }
        )
    elif bucket in runner_buckets:
        result.update(
            {
                "tp1_pips": float(getattr(policy, "runner_tp1_pips", 8.0)),
                "tp1_close_pct": float(getattr(policy, "runner_tp1_close_pct", 55.0)),
                "be_plus_pips": float(getattr(policy, "runner_be_spread_plus_pips", 0.5)),
                "trail_mode": "m1" if bool(getattr(policy, "trail_escalation_enabled", False)) else "m5",
                "exit_profile": "runner",
            }
        )
    return result


def prepare_intraday_corridor_map(m1: pd.DataFrame) -> pd.DataFrame:
    h1 = resample_intraday_ohlc(m1, "H1")
    records: list[dict[str, object]] = []
    if h1 is None or h1.empty:
        return pd.DataFrame(columns=["fib_S2", "fib_R2"]).set_index(pd.DatetimeIndex([], tz="UTC"))
    for i in range(len(h1)):
        levels = compute_previous_candle_fib_levels(h1.iloc[: i + 1])
        records.append(
            {
                "time": h1.index[i],
                "fib_S2": None if levels is None else levels.get("S2"),
                "fib_R2": None if levels is None else levels.get("R2"),
            }
        )
    return pd.DataFrame(records).set_index("time")


def build_entry_history(entries: list[dict[str, object]]) -> pd.DataFrame:
    if not entries:
        return pd.DataFrame(columns=["side", "entry_timestamp_utc", "policy_type"])
    return pd.DataFrame(entries)


def precompute_m1_cross_recency(m1f: pd.DataFrame) -> tuple[list[Optional[int]], list[Optional[int]]]:
    close = m1f["close"].astype(float)
    ema5 = close.ewm(span=5, adjust=False).mean()
    ema9 = close.ewm(span=9, adjust=False).mean()
    diff = ema5 - ema9
    buy_cross_idx: Optional[int] = None
    sell_cross_idx: Optional[int] = None
    buy_recency: list[Optional[int]] = [None] * len(diff)
    sell_recency: list[Optional[int]] = [None] * len(diff)
    for i in range(1, len(diff)):
        prev = float(diff.iloc[i - 1])
        curr = float(diff.iloc[i])
        if curr > 0 and prev <= 0:
            buy_cross_idx = i
        if curr < 0 and prev >= 0:
            sell_cross_idx = i
        buy_recency[i] = None if buy_cross_idx is None else i - buy_cross_idx
        sell_recency[i] = None if sell_cross_idx is None else i - sell_cross_idx
    return buy_recency, sell_recency


def simulate_variant(
    *,
    m1: pd.DataFrame,
    features: dict[str, pd.DataFrame],
    mapped: pd.DataFrame,
    profile,
    policy: ExecutionPolicyKtCgTrial10,
    variant: Variant,
    precomputed_signals: list[Optional[dict]],
    precomputed_signal_blocks: Counter,
    spread_pips: float = 2.0,
) -> dict:
    policy = policy.model_copy(
        update={
            "regime_london_sell_veto": variant.london_sell_veto,
            "regime_chop_pause_enabled": variant.chop_pause_enabled,
        }
    )
    effective_risk = get_effective_risk(profile)
    spread_price = spread_pips * PIP_SIZE
    half_spread = spread_price / 2.0
    open_positions: list[LivePosition] = []
    closed_rows: list[dict] = []
    pending_entry: Optional[dict] = None
    block_counts: Counter = Counter(precomputed_signal_blocks)
    signal_counts: Counter = Counter()
    balance = 100_000.0
    max_balance = balance
    max_drawdown_usd = 0.0
    trade_id = 0
    last_trade_time: Optional[pd.Timestamp] = None
    trades_per_day: Counter = Counter()
    m5_index = features["m5"].index
    m5_close_set = set(m5_index)
    m5_index_pos = {ts: i for i, ts in enumerate(m5_index)}
    buy_bars_since_cross, sell_bars_since_cross = precompute_m1_cross_recency(features["m1"])
    m5_close_series = features["m5"]["close"].astype(float)
    m5_e9 = m5_close_series.ewm(span=9, adjust=False).mean()
    m5_e21 = m5_close_series.ewm(span=21, adjust=False).mean()
    buy_entries_since_recross = 0
    sell_entries_since_recross = 0
    m15 = resample_ohlc(m1, "15min")
    m15[f"ema_{int(policy.trail_escalation_m15_ema_period)}"] = (
        m15["close"].astype(float).ewm(span=int(policy.trail_escalation_m15_ema_period), adjust=False).mean()
    )
    m15_index = m15.index
    m15_close_set = set(m15_index)
    m15_index_pos = {ts: i for i, ts in enumerate(m15_index)}
    h1_levels = prepare_intraday_corridor_map(m1).reindex(mapped.index, method="ffill")
    corridor_filter = IntradayFibCorridorFilter(
        enabled=bool(policy.intraday_fib_enabled),
        lower_level=str(policy.intraday_fib_lower_level),
        upper_level=str(policy.intraday_fib_upper_level),
        timeframe=str(policy.intraday_fib_timeframe),
        lookback_bars=int(policy.intraday_fib_lookback_bars),
        boundary_buffer_pips=float(policy.intraday_fib_boundary_buffer_pips),
        hysteresis_pips=float(policy.intraday_fib_hysteresis_pips),
        pip_size=PIP_SIZE,
    )
    side_pause_until: dict[str, Optional[pd.Timestamp]] = {"buy": None, "sell": None}

    times = list(mapped.index)
    for j in range(1, len(mapped)):
        ts = times[j]
        row = mapped.iloc[j]
        open_mid = float(row["open"])
        high_mid = float(row["high"])
        low_mid = float(row["low"])
        close_mid = float(row["close"])
        open_bid = open_mid - half_spread
        open_ask = open_mid + half_spread
        high_bid = high_mid - half_spread
        low_bid = low_mid - half_spread
        high_ask = high_mid + half_spread
        low_ask = low_mid + half_spread
        close_bid = close_mid - half_spread
        close_ask = close_mid + half_spread

        if ts in m5_close_set:
            m5_idx = m5_index_pos.get(ts)
            if m5_idx is not None and m5_idx >= 1:
                prev_diff = float(m5_e9.iloc[m5_idx - 1] - m5_e21.iloc[m5_idx - 1])
                curr_diff = float(m5_e9.iloc[m5_idx] - m5_e21.iloc[m5_idx])
                if curr_diff > 0 and prev_diff <= 0:
                    buy_entries_since_recross = 0
                if curr_diff < 0 and prev_diff >= 0:
                    sell_entries_since_recross = 0

        if pending_entry is not None:
            trade_id += 1
            side = str(pending_entry["side"])
            entry_price = open_ask if side == "buy" else open_bid
            stop_pips = float(pending_entry["stop_pips"])
            stop_price = entry_price - stop_pips * PIP_SIZE if side == "buy" else entry_price + stop_pips * PIP_SIZE
            exit_profile = bucket_exit_profile(policy, str(pending_entry["runner_bucket"]))
            tp1_pips = float(exit_profile["tp1_pips"])
            tp1_price = entry_price + tp1_pips * PIP_SIZE if side == "buy" else entry_price - tp1_pips * PIP_SIZE
            lots = float(pending_entry["lots"])
            pos = LivePosition(
                trade_id=trade_id,
                side=side,
                entry_type=str(pending_entry["entry_type"]),
                tier_period=pending_entry["tier"],
                signal_time=pending_entry["signal_time"],
                entry_time=ts,
                entry_price=entry_price,
                initial_lots=lots,
                remaining_lots=lots,
                stop_price=stop_price,
                initial_stop_price=stop_price,
                tp1_price=tp1_price,
                tp1_done=False,
                tp1_close_pct=float(exit_profile["tp1_close_pct"]),
                be_stop_price=None,
                m5_bucket=str(pending_entry["m5_bucket"]),
                m1_bucket=str(pending_entry["m1_bucket"]),
                conviction_multiplier=float(pending_entry["multiplier"]),
                conviction_lots=lots,
                atr_stop_pips=stop_pips,
                pullback_quality_label=str((pending_entry.get("pullback_quality") or {}).get("label") or ""),
                pullback_quality_bar_count=int((pending_entry.get("pullback_quality") or {}).get("pullback_bar_count") or 0),
                pullback_quality_structure_ratio=float((pending_entry.get("pullback_quality") or {}).get("structure_ratio") or 0.0),
                pullback_quality_dampener=float((pending_entry.get("pullback_quality") or {}).get("dampener_multiplier") or 1.0),
                regime_multiplier=float(pending_entry.get("regime_multiplier") or 1.0),
                regime_label=str(pending_entry.get("regime_label") or "baseline"),
                runner_bucket=str(pending_entry["runner_bucket"]),
                exit_profile=str(exit_profile["exit_profile"]),
                managed_trail_mode=str(exit_profile["trail_mode"]),
                be_plus_pips=float(exit_profile["be_plus_pips"]),
            )
            open_positions.append(pos)
            if side == "buy":
                buy_entries_since_recross += 1
            else:
                sell_entries_since_recross += 1
            trades_per_day[ts.floor("D")] += 1
            last_trade_time = ts
            signal_counts[str(pending_entry["entry_type"])] += 1
            pending_entry = None

        survivors: list[LivePosition] = []
        for pos in open_positions:
            closed = False
            ref_price = close_mid
            if not pos.tp1_done:
                if pos.side == "buy":
                    stop_hit = low_bid <= pos.stop_price
                    tp1_hit = high_bid >= pos.tp1_price
                else:
                    stop_hit = high_ask >= pos.stop_price
                    tp1_hit = low_ask <= pos.tp1_price
                if stop_hit and tp1_hit:
                    first = choose_first_event(pos.side, open_mid, pos.stop_price, pos.tp1_price, spread_price)
                    if first == "stop":
                        tp1_hit = False
                    else:
                        stop_hit = False
                if stop_hit:
                    close_leg(pos, pos.remaining_lots, pos.stop_price, "initial_stop", ref_price)
                    closed = True
                elif tp1_hit:
                    tp1_lots = round(pos.initial_lots * (pos.tp1_close_pct / 100.0), 2)
                    tp1_lots = max(0.01, min(tp1_lots, pos.remaining_lots))
                    close_leg(pos, tp1_lots, pos.tp1_price, "tp1_partial", ref_price)
                    pos.tp1_done = True
                    be_offset = spread_price + pos.be_plus_pips * PIP_SIZE
                    pos.be_stop_price = pos.entry_price + be_offset if pos.side == "buy" else pos.entry_price - be_offset
                    pos.stop_price = float(pos.be_stop_price)
                    if pos.remaining_lots <= 0.0:
                        closed = True

            if not closed and pos.tp1_done:
                if pos.side == "buy" and low_bid <= pos.stop_price:
                    close_leg(pos, pos.remaining_lots, pos.stop_price, "runner_stop", ref_price)
                    closed = True
                elif pos.side == "sell" and high_ask >= pos.stop_price:
                    close_leg(pos, pos.remaining_lots, pos.stop_price, "runner_stop", ref_price)
                    closed = True

            if not closed and pos.tp1_done:
                trail_mode = pos.managed_trail_mode
                if bool(policy.trail_escalation_enabled) and trail_mode in ("m1", "m5") and pos.exit_profile != "quick":
                    profit_pips = ((close_mid - pos.entry_price) / PIP_SIZE) if pos.side == "buy" else ((pos.entry_price - close_mid) / PIP_SIZE)
                    if profit_pips >= float(policy.trail_escalation_tier2_pips):
                        trail_mode = "m15"
                    elif profit_pips >= float(policy.trail_escalation_tier1_pips):
                        trail_mode = "m5"
                    else:
                        trail_mode = "m1"
                    ranks = {"": -1, "m1": 0, "m5": 1, "m15": 2}
                    stored = str(pos.trail_escalation_level or "").lower()
                    if ranks.get(trail_mode, 0) > ranks.get(stored, -1):
                        pos.trail_escalation_level = trail_mode
                    elif stored in ("m1", "m5", "m15"):
                        trail_mode = stored

                if trail_mode == "hwm":
                    if pos.side == "buy":
                        pos.peak_price = max(float(pos.peak_price or pos.entry_price), close_mid)
                        new_sl = max(float(pos.be_stop_price or pos.entry_price), float(pos.peak_price) - float(policy.hwm_trail_pips) * PIP_SIZE)
                    else:
                        pos.peak_price = min(float(pos.peak_price or pos.entry_price), close_mid)
                        new_sl = min(float(pos.be_stop_price or pos.entry_price), float(pos.peak_price) + float(policy.hwm_trail_pips) * PIP_SIZE)
                    if pos.side == "buy":
                        pos.stop_price = max(pos.stop_price, new_sl)
                    else:
                        pos.stop_price = min(pos.stop_price, new_sl)
                elif trail_mode == "m1":
                    period = int(policy.trail_ema_period)
                    if j >= period + 1 and pd.notna(features["m1"][f"ema_{period}"].iat[j - 1]):
                        ema_val = float(features["m1"][f"ema_{period}"].iat[j - 1])
                        last_close = float(features["m1"]["close"].iat[j - 1])
                        new_sl = max(float(pos.be_stop_price or pos.stop_price), ema_val - PIP_SIZE) if pos.side == "buy" else min(float(pos.be_stop_price or pos.stop_price), ema_val + PIP_SIZE)
                        pos.stop_price = max(pos.stop_price, new_sl) if pos.side == "buy" else min(pos.stop_price, new_sl)
                        if (pos.side == "buy" and last_close < ema_val) or (pos.side == "sell" and last_close > ema_val):
                            close_leg(pos, pos.remaining_lots, close_bid if pos.side == "buy" else close_ask, "m1_trail_close", ref_price)
                            closed = True
                elif trail_mode == "m5" and ts in m5_close_set:
                    period = int(policy.trail_m5_ema_period)
                    m5_idx = m5_index_pos.get(ts)
                    if m5_idx is not None and m5_idx > period and pd.notna(features["m5"][f"ema_{period}"].iloc[m5_idx - 1]):
                        ema_val = float(features["m5"][f"ema_{period}"].iloc[m5_idx - 1])
                        last_close = float(features["m5"]["close"].iloc[m5_idx - 1])
                        new_sl = max(float(pos.be_stop_price or pos.stop_price), ema_val - PIP_SIZE) if pos.side == "buy" else min(float(pos.be_stop_price or pos.stop_price), ema_val + PIP_SIZE)
                        pos.stop_price = max(pos.stop_price, new_sl) if pos.side == "buy" else min(pos.stop_price, new_sl)
                        if (pos.side == "buy" and last_close < ema_val) or (pos.side == "sell" and last_close > ema_val):
                            close_leg(pos, pos.remaining_lots, close_bid if pos.side == "buy" else close_ask, "m5_trail_close", ref_price)
                            closed = True
                elif trail_mode == "m15" and ts in m15_close_set:
                    period = int(policy.trail_escalation_m15_ema_period)
                    m15_idx = m15_index_pos.get(ts)
                    if m15_idx is not None and m15_idx > period and pd.notna(m15[f"ema_{period}"].iloc[m15_idx - 1]):
                        ema_val = float(m15[f"ema_{period}"].iloc[m15_idx - 1])
                        last_close = float(m15["close"].iloc[m15_idx - 1])
                        buffer_price = float(policy.trail_escalation_m15_buffer_pips) * PIP_SIZE
                        new_sl = max(float(pos.be_stop_price or pos.stop_price), ema_val - buffer_price) if pos.side == "buy" else min(float(pos.be_stop_price or pos.stop_price), ema_val + buffer_price)
                        pos.stop_price = max(pos.stop_price, new_sl) if pos.side == "buy" else min(pos.stop_price, new_sl)
                        if (pos.side == "buy" and last_close < ema_val) or (pos.side == "sell" and last_close > ema_val):
                            close_leg(pos, pos.remaining_lots, close_bid if pos.side == "buy" else close_ask, "m15_trail_close", ref_price)
                            closed = True

            if closed or pos.remaining_lots <= 0.0:
                balance += pos.weighted_usd
                weighted_pips = pos.weighted_pips_x_lots / pos.initial_lots if pos.initial_lots > 0 else 0.0
                closed_rows.append(
                    {
                        "trade_id": pos.trade_id,
                        "side": pos.side,
                        "entry_type": pos.entry_type,
                        "tier_period": pos.tier_period,
                        "entry_time": pos.entry_time.isoformat(),
                        "exit_time": ts.isoformat(),
                        "initial_lots": round(pos.initial_lots, 2),
                        "weighted_pips": round(weighted_pips, 3),
                        "profit_usd": round(pos.weighted_usd, 4),
                        "exit_reason": pos.exit_reason,
                        "tp1_done": pos.tp1_done,
                        "m5_bucket": pos.m5_bucket,
                        "runner_bucket": pos.runner_bucket,
                        "regime_label": pos.regime_label,
                        "hold_minutes": round((ts - pos.entry_time).total_seconds() / 60.0, 1),
                        "close_time_utc": ts.isoformat(),
                    }
                )
            else:
                survivors.append(pos)
        open_positions = survivors

        equity = balance + sum(unrealized_usd(pos, close_mid) for pos in open_positions)
        max_balance = max(max_balance, equity)
        max_drawdown_usd = max(max_drawdown_usd, max_balance - equity)

        fib_s2 = h1_levels["fib_S2"].iat[j] if "fib_S2" in h1_levels.columns else None
        fib_r2 = h1_levels["fib_R2"].iat[j] if "fib_R2" in h1_levels.columns else None
        if pd.notna(fib_s2) and pd.notna(fib_r2):
            corridor_filter.update_levels(
                {"S2": float(fib_s2), "R2": float(fib_r2)},
                source_close=close_mid,
                calculation_mode="previous_h1",
            )
        else:
            corridor_filter.update_levels(None, source_close=close_mid, calculation_mode="previous_h1")

        signal = precomputed_signals[j]
        if signal is None or j >= len(mapped) - 1:
            continue

        corridor_ok, corridor_reason = corridor_filter.check_corridor(close_mid)
        if not corridor_ok:
            block_counts["intraday_fib_block"] += 1
            block_counts[corridor_reason or "intraday_fib_block"] += 1
            continue

        current_day = ts.floor("D")
        if trades_per_day[current_day] >= int(effective_risk.max_trades_per_day):
            block_counts["max_trades_per_day"] += 1
            continue
        if len(open_positions) >= int(effective_risk.max_open_trades):
            block_counts["max_open_trades"] += 1
            continue
        side = str(signal["side"])
        side_open = sum(1 for p in open_positions if p.side == side)
        if side_open >= int(policy.max_open_trades_per_side or 999):
            block_counts["max_open_trades_per_side"] += 1
            continue
        if signal["entry_type"] == "zone_entry":
            zone_open = sum(1 for p in open_positions if p.entry_type == "zone_entry")
            if zone_open >= int(policy.max_zone_entry_open or 999):
                block_counts["max_zone_entry_open"] += 1
                continue
            if last_trade_time is not None:
                elapsed = (ts - last_trade_time).total_seconds() / 60.0
                if elapsed < float(policy.cooldown_minutes):
                    block_counts["zone_cooldown"] += 1
                    continue
        else:
            tier_open = sum(1 for p in open_positions if p.entry_type == "tiered_pullback")
            if tier_open >= int(policy.max_tiered_pullback_open or 999):
                block_counts["max_tiered_pullback_open"] += 1
                continue

        m5_bucket = str(signal["m5_bucket"])
        recent_closed = [
            {"exit_reason": str(r["exit_reason"]), "close_time_utc": r["close_time_utc"]}
            for r in closed_rows[-max(int(policy.regime_chop_pause_lookback_trades), 20):]
        ]
        current_pause_start = side_pause_until.get(side)
        chop_paused, chop_reason = check_chop_pause(
            side=side,
            recent_trades=recent_closed,
            now_utc=ts.to_pydatetime(),
            lookback_trades=int(policy.regime_chop_pause_lookback_trades),
            stop_rate_threshold=float(policy.regime_chop_pause_stop_rate),
            pause_minutes=int(policy.regime_chop_pause_minutes),
            current_pause_start=current_pause_start.to_pydatetime() if current_pause_start is not None else None,
            m5_bucket=m5_bucket,
        )
        if chop_paused and current_pause_start is None:
            side_pause_until[side] = ts
        elif not chop_paused and current_pause_start is not None:
            side_pause_until[side] = None

        regime = evaluate_regime_gate(
            hour_et=hour_et(ts),
            side=side,
            m5_bucket=m5_bucket,
            enabled=bool(policy.regime_gate_enabled),
            london_sell_veto=bool(policy.regime_london_sell_veto),
            london_start_hour_et=int(policy.regime_london_start_hour_et),
            london_end_hour_et=int(policy.regime_london_end_hour_et),
            boost_hours_et=tuple(int(x) for x in policy.regime_boost_hours_et),
            boost_multiplier=float(policy.regime_boost_multiplier),
            buy_base_multiplier=float(policy.regime_buy_base_multiplier),
            sell_base_multiplier=float(policy.regime_sell_base_multiplier),
            worst_hours_et=tuple(int(x) for x in policy.regime_worst_hours_et),
            worst_multiplier=float(policy.regime_worst_multiplier),
            weak_regime_multiplier=float(policy.regime_weak_multiplier),
            chop_paused=bool(policy.regime_chop_pause_enabled and chop_paused),
            chop_pause_reason=chop_reason,
        )
        if not regime.allowed:
            if regime.label == "VETO_LONDON_SELL":
                block_counts["london_sell_veto"] += 1
            elif regime.label == "CHOP_PAUSE":
                block_counts["chop_pause"] += 1
            else:
                block_counts["regime_gate_block"] += 1
            continue

        atr_pips = mapped["atr_pips"].iat[j]
        stop_pips = min(float(atr_pips) * float(profile.trade_management.stop_loss.atr_multiplier), float(profile.trade_management.stop_loss.max_sl_pips))
        stop_pips = max(stop_pips, float(effective_risk.min_stop_pips))

        structure_ratio = None
        pq = signal.get("pullback_quality") or {}
        if pq:
            try:
                structure_ratio = float(pq.get("structure_ratio"))
            except Exception:
                structure_ratio = None
        bars_since_cross = buy_bars_since_cross[j] if side == "buy" else sell_bars_since_cross[j]
        prior_entries = buy_entries_since_recross if side == "buy" else sell_entries_since_recross
        rs = compute_runner_score(
            atr_stop_pips=stop_pips,
            regime_label=regime.label.lower(),
            m5_bucket=m5_bucket,
            structure_ratio=structure_ratio,
            bars_since_cross=bars_since_cross,
            prior_entries=prior_entries,
            freshness_mode="strict",
        )
        runner_bucket_lots = {
            "floor": round(max(float(policy.conviction_min_lots), 1.8 * float(policy.runner_bucket_mult_floor)), 2),
            "base": round(max(float(policy.conviction_min_lots), 1.8 * float(policy.runner_bucket_mult_base)), 2),
            "elevated": round(max(float(policy.conviction_min_lots), 1.8 * float(policy.runner_bucket_mult_elevated)), 2),
            "press": round(max(float(policy.conviction_min_lots), 1.8 * float(policy.runner_bucket_mult_press)), 2),
            "elite": round(max(float(policy.conviction_min_lots), 1.8 * float(policy.runner_bucket_mult_elite)), 2),
        }
        lots, _audit = runner_score_to_lots(
            result=rs,
            bucket_lots=runner_bucket_lots,
            regime_multiplier=float(regime.multiplier),
            spread_pips=spread_pips,
            spread_gate_pips=float(policy.runner_spread_gate_pips),
            tier=int(signal["tier"]) if signal.get("tier") is not None else None,
            is_boost_hour=str(regime.label).upper() == "BUY_BOOST_HOUR",
            force_floor_tier17_nonboost=bool(policy.runner_tier17_nonboost_force_floor),
            min_lots=float(policy.conviction_min_lots),
            max_lots=float(effective_risk.max_lots),
        )
        if signal["entry_type"] == "tiered_pullback" and pq.get("applicable") and str(pq.get("label", "")).lower() == "sloppy":
            lots = min(lots, runner_bucket_lots["floor"])
        if signal["entry_type"] == "zone_entry" and bool(policy.runner_weak_m5_zone_cap_enabled) and m5_bucket == "weak":
            lots = min(lots, runner_bucket_lots["elevated"])

        directional_cap = float(policy.max_directional_lots_per_side or 0.0)
        same_side_open_lots = sum(p.remaining_lots for p in open_positions if p.side == side)
        if directional_cap > 0:
            available = directional_cap - same_side_open_lots
            if available < float(policy.conviction_min_lots):
                block_counts["directional_cap_block"] += 1
                continue
            if lots > available:
                lots = round(max(float(policy.conviction_min_lots), available), 2)

        pending_entry = {
            "signal_time": ts,
            "side": side,
            "entry_type": str(signal["entry_type"]),
            "tier": signal["tier"],
            "m5_bucket": m5_bucket,
            "m1_bucket": "runner",
            "multiplier": float(regime.multiplier),
            "lots": float(lots),
            "stop_pips": float(stop_pips),
            "pullback_quality": signal.get("pullback_quality"),
            "regime_multiplier": float(regime.multiplier),
            "regime_label": str(regime.label).lower(),
            "runner_bucket": rs.bucket,
        }

    if len(mapped) and open_positions:
        final_ts = mapped.index[-1]
        final_close = float(mapped["close"].iat[-1])
        final_bid = final_close - half_spread
        final_ask = final_close + half_spread
        for pos in open_positions:
            exit_price = final_bid if pos.side == "buy" else final_ask
            close_leg(pos, pos.remaining_lots, exit_price, "end_of_data", final_close)
            balance += pos.weighted_usd
            weighted_pips = pos.weighted_pips_x_lots / pos.initial_lots if pos.initial_lots > 0 else 0.0
            closed_rows.append(
                {
                    "trade_id": pos.trade_id,
                    "side": pos.side,
                    "entry_type": pos.entry_type,
                    "tier_period": pos.tier_period,
                    "entry_time": pos.entry_time.isoformat(),
                    "exit_time": final_ts.isoformat(),
                    "initial_lots": round(pos.initial_lots, 2),
                    "weighted_pips": round(weighted_pips, 3),
                    "profit_usd": round(pos.weighted_usd, 4),
                    "exit_reason": pos.exit_reason,
                    "tp1_done": pos.tp1_done,
                    "m5_bucket": pos.m5_bucket,
                    "runner_bucket": pos.runner_bucket,
                    "regime_label": pos.regime_label,
                    "hold_minutes": round((final_ts - pos.entry_time).total_seconds() / 60.0, 1),
                    "close_time_utc": final_ts.isoformat(),
                }
            )

    trades_df = pd.DataFrame(closed_rows)
    wins = trades_df[trades_df["profit_usd"] > 0] if not trades_df.empty else pd.DataFrame()
    losses = trades_df[trades_df["profit_usd"] < 0] if not trades_df.empty else pd.DataFrame()
    gross_profit = float(wins["profit_usd"].sum()) if not wins.empty else 0.0
    gross_loss = float(losses["profit_usd"].sum()) if not losses.empty else 0.0
    profit_factor = (gross_profit / abs(gross_loss)) if gross_loss < 0 else None
    return {
        "variant": variant.name,
        "trades": int(len(trades_df)),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "win_rate_pct": round((len(wins) / len(trades_df) * 100.0), 2) if len(trades_df) else 0.0,
        "ending_balance_usd": round(balance, 4),
        "net_profit_usd": round(balance - 100_000.0, 4),
        "max_drawdown_usd": round(max_drawdown_usd, 4),
        "avg_trade_usd": round(float(trades_df["profit_usd"].mean()), 4) if len(trades_df) else 0.0,
        "avg_trade_pips": round(float(trades_df["weighted_pips"].mean()), 4) if len(trades_df) else 0.0,
        "profit_factor": round(profit_factor, 4) if profit_factor is not None else None,
        "by_entry_type": trades_df["entry_type"].value_counts().to_dict() if len(trades_df) else {},
        "by_exit_reason": trades_df["exit_reason"].value_counts().to_dict() if len(trades_df) else {},
        "by_runner_bucket": trades_df["runner_bucket"].value_counts().to_dict() if len(trades_df) else {},
        "block_counts": dict(sorted(block_counts.items(), key=lambda kv: kv[1], reverse=True)[:25]),
        "signal_counts": dict(signal_counts),
    }


def main() -> None:
    args = parse_args()
    m1 = load_m1(str(Path(args.input_csv)))
    if int(args.max_bars or 0) > 0 and len(m1) > int(args.max_bars):
        m1 = m1.iloc[-int(args.max_bars):].copy()
    profile, policy = build_live_trial10_policy()
    features = prepare_features(m1, policy)
    mapped = map_features_to_m1(features["m1"], features)
    precomputed_signals, precomputed_signal_blocks = precompute_signal_sequence(
        m1f=features["m1"],
        mapped=mapped,
        policy=policy,
    )

    all_variants = [
        Variant("both_off", False, False),
        Variant("london_on_chop_off", True, False),
        Variant("london_off_chop_on", False, True),
        Variant("both_on", True, True),
    ]
    selected = {v.strip() for v in str(args.variants or "").split(",") if v.strip()}
    variants = [v for v in all_variants if v.name in selected] or all_variants
    results = []
    for variant in variants:
        result = simulate_variant(
            m1=m1,
            features=features,
            mapped=mapped,
            profile=profile,
            policy=policy,
            variant=variant,
            precomputed_signals=precomputed_signals,
            precomputed_signal_blocks=precomputed_signal_blocks,
        )
        results.append(result)
        print(
            f"[trial10-500k] {variant.name}: trades={result['trades']} "
            f"net=${result['net_profit_usd']} avg_pips={result['avg_trade_pips']} "
            f"win_rate={result['win_rate_pct']}%"
        )

    summary = {
        "input_csv": str(Path(args.input_csv).resolve()),
        "bars": int(len(mapped)),
        "start": mapped.index[0].isoformat() if len(mapped) else None,
        "end": mapped.index[-1].isoformat() if len(mapped) else None,
        "assumptions": {
            "spread_pips": 2.0,
            "runner_score_base_lots": 1.8,
            "runner_max_lots": 7.0,
            "base_lots_for_directional_cap_formula": 1.0,
            "directional_cap_formula": "base_lots * max_open_trades_per_side / 2",
            "calculated_directional_cap_lots_per_side": float(policy.max_directional_lots_per_side or 0.0),
            "current_live_variant": "london_off_chop_on",
        },
        "variants": results,
    }
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    by_name = {r["variant"]: r for r in results}
    md = [
        "# Trial 10 Regime Gate + Live Config 500k",
        "",
        f"- Input: `{summary['input_csv']}`",
        f"- Bars: `{summary['bars']}`",
        f"- Current live comparison variant: `{summary['assumptions']['current_live_variant']}`",
        f"- Directional cap formula: `{summary['assumptions']['directional_cap_formula']}` -> `{summary['assumptions']['calculated_directional_cap_lots_per_side']}` lots/side",
        "",
        "## Variants",
        "",
    ]
    for result in results:
        md.extend(
            [
                f"### {result['variant']}",
                f"- trades: `{result['trades']}`",
                f"- net: `${result['net_profit_usd']}`",
                f"- avg pips: `{result['avg_trade_pips']}`",
                f"- win rate: `{result['win_rate_pct']}%`",
                f"- max DD: `${result['max_drawdown_usd']}`",
                "",
            ]
        )
    if "london_off_chop_on" in by_name and "both_off" in by_name:
        delta = round(by_name["london_off_chop_on"]["net_profit_usd"] - by_name["both_off"]["net_profit_usd"], 2)
        md.extend(
            [
                "## Quick Read",
                "",
                f"- Chop Pause effect vs both_off: `${delta}` on net P/L.",
                f"- London Sell Veto effect can be read by comparing `both_on` vs `london_off_chop_on`, and `london_on_chop_off` vs `both_off`.",
                "",
            ]
        )
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
