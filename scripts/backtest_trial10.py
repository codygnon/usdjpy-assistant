#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.conviction_sizing import MULTIPLIER_MATRIX
from core.fib_pivots import compute_rolling_intraday_fib_levels
from core.indicators import atr as atr_fn
from core.indicators import ema as ema_fn
from core.no_trade_zone import IntradayFibCorridorFilter, NoTradeZoneFilter
from core.pullback_quality import analyze_pullback_quality, pullback_quality_snapshot
from core.presets import PresetId, apply_preset
from core.profile import (
    ExecutionPolicyKtCgTrial10,
    ProfileV1,
    default_profile_for_name,
    get_effective_risk,
)


PIP_SIZE = 0.01
PIP_VALUE_UNITS = 100_000.0


@dataclass
class Position:
    trade_id: int
    side: str
    entry_type: str
    tier_period: Optional[int]
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    entry_price: float
    initial_lots: float
    remaining_lots: float
    stop_price: float
    initial_stop_price: float
    tp1_price: float
    tp1_done: bool
    tp1_close_pct: float
    be_stop_price: Optional[float]
    exit_reason: str = ""
    weighted_pips_x_lots: float = 0.0
    weighted_usd: float = 0.0
    m5_bucket: str = "normal"
    m1_bucket: str = "neutral"
    conviction_multiplier: float = 1.0
    conviction_lots: float = 0.0
    atr_stop_pips: float = 0.0
    pullback_quality_label: str = ""
    pullback_quality_bar_count: int = 0
    pullback_quality_structure_ratio: float = 0.0
    pullback_quality_dampener: float = 1.0
    regime_multiplier: float = 1.0
    regime_label: str = "baseline"


TORONTO_TZ = ZoneInfo("America/Toronto")
BEST_BUY_HOURS_ET = {6, 7, 12, 13, 15}
LONDON_HOURS_ET = {7, 8, 9, 10, 11}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Close-only Trial #10 backtest on USDJPY M1 CSV.")
    p.add_argument(
        "--in",
        dest="input_csv",
        default="research_out/USDJPY_M1_OANDA_250k.csv",
        help="Input M1 CSV with columns time,open,high,low,close",
    )
    p.add_argument(
        "--out",
        default="research_out/trial10_250k_summary.json",
        help="Summary JSON output path",
    )
    p.add_argument(
        "--trades-out",
        default="research_out/trial10_250k_trades.csv",
        help="Closed trades CSV output path",
    )
    p.add_argument(
        "--equity-out",
        default="research_out/trial10_250k_equity.csv",
        help="Equity curve CSV output path",
    )
    p.add_argument("--spread-pips", type=float, default=2.0, help="Fixed spread in pips")
    p.add_argument("--starting-balance", type=float, default=10_000.0, help="Starting USD balance")
    p.add_argument(
        "--operator-regime",
        action="store_true",
        help="Enable operator-style regime sizing and pause rules.",
    )
    p.add_argument(
        "--buy-boost-hours-et",
        default="6,7,12,13,15",
        help="Comma-separated ET hours where buys get boosted under operator regime.",
    )
    p.add_argument("--buy-boost-mult", type=float, default=1.35, help="Lot multiplier for boosted buy hours.")
    p.add_argument(
        "--buy-base-mult",
        type=float,
        default=0.75,
        help="Lot multiplier for buys outside boosted hours under operator regime.",
    )
    p.add_argument(
        "--sell-base-mult",
        type=float,
        default=0.50,
        help="Lot multiplier for sells outside veto hours under operator regime.",
    )
    p.add_argument(
        "--london-sell-veto",
        action="store_true",
        help="Block sells during London ET hours (07-11 ET).",
    )
    p.add_argument(
        "--autopause-minutes",
        type=float,
        default=45.0,
        help="Minutes to look back for same-side stop/sloppy clustering, and pause duration.",
    )
    p.add_argument(
        "--autopause-stop-threshold",
        type=int,
        default=2,
        help="Pause a side after this many initial-stop trades in the lookback window.",
    )
    p.add_argument(
        "--tier17-nonboost-mult",
        type=float,
        default=1.0,
        help="Extra lot multiplier for tier 17 entries outside buy boost hours.",
    )
    p.add_argument(
        "--side-hour-mults",
        default="",
        help="Comma-separated side:hour=mult overrides, e.g. sell:3=0.15,sell:12=0.15,buy:11=0.25,buy:16=0.25",
    )
    p.add_argument(
        "--tier-periods",
        default="",
        help="Comma-separated tier EMA periods override, e.g. 17,21,27,50",
    )
    p.add_argument(
        "--tp1-close-pct",
        type=float,
        default=None,
        help="Override TP1 close percentage for runner management.",
    )
    return p.parse_args()


def parse_hour_list(raw: str) -> set[int]:
    hours: set[int] = set()
    for part in str(raw or "").split(","):
        token = part.strip()
        if not token:
            continue
        try:
            hour = int(token)
        except ValueError:
            continue
        if 0 <= hour <= 23:
            hours.add(hour)
    return hours


def parse_int_list(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for part in str(raw or "").split(","):
        token = part.strip()
        if not token:
            continue
        try:
            values.append(int(token))
        except ValueError:
            continue
    return tuple(sorted(dict.fromkeys(values)))


def parse_side_hour_mults(raw: str) -> dict[tuple[str, int], float]:
    overrides: dict[tuple[str, int], float] = {}
    for part in str(raw or "").split(","):
        token = part.strip()
        if not token or "=" not in token or ":" not in token:
            continue
        lhs, rhs = token.split("=", 1)
        side_raw, hour_raw = lhs.split(":", 1)
        side = side_raw.strip().lower()
        if side not in {"buy", "sell"}:
            continue
        try:
            hour = int(hour_raw.strip())
            mult = float(rhs.strip())
        except ValueError:
            continue
        if 0 <= hour <= 23 and mult >= 0.0:
            overrides[(side, hour)] = mult
    return overrides


def hour_et(ts: pd.Timestamp) -> int:
    return int(ts.tz_convert(TORONTO_TZ).hour)


def resolve_regime_multiplier(
    *,
    ts: pd.Timestamp,
    side: str,
    operator_regime: bool,
    buy_boost_hours_et: set[int],
    buy_boost_mult: float,
    buy_base_mult: float,
    sell_base_mult: float,
    side_hour_mults: dict[tuple[str, int], float],
) -> tuple[float, str]:
    if not operator_regime:
        return 1.0, "baseline"
    current_hour = hour_et(ts)
    override = side_hour_mults.get((str(side).lower(), current_hour))
    if override is not None:
        return float(override), f"{str(side).lower()}_hour_override"
    if side == "buy":
        if current_hour in buy_boost_hours_et:
            return float(buy_boost_mult), "buy_boost_hour"
        return float(buy_base_mult), "buy_base_hour"
    return float(sell_base_mult), "sell_base_hour"


def load_m1(path: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    need = {"time", "open", "high", "low", "close"}
    if not need.issubset(df.columns):
        raise ValueError(f"{csv_path} missing required columns: {sorted(need)}")
    df = df[["time", "open", "high", "low", "close"]].copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().sort_values("time").drop_duplicates(subset=["time"], keep="last")
    df = df.set_index("time")
    return df


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return (
        df[["open", "high", "low", "close"]]
        .resample(rule, label="right", closed="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna(subset=["open", "high", "low", "close"])
    )


def build_profile(args: argparse.Namespace) -> tuple[ProfileV1, ExecutionPolicyKtCgTrial10]:
    base = default_profile_for_name("trial10_backtest")
    base = base.model_copy()
    base.symbol = "USDJPY"
    base.pip_size = PIP_SIZE
    base.deposit_amount = 10_000.0
    base.display_currency = "USD"
    base.risk = base.risk.model_copy(
        update={
            "max_lots": 10.0,
            "require_stop": True,
            "min_stop_pips": 5.0,
            "max_spread_pips": 20.0,
            "max_trades_per_day": 1000,
            "max_open_trades": 100,
        }
    )
    profile = apply_preset(base, PresetId.KT_CG_TRIAL_10)
    policy = profile.execution.policies[0]
    if not isinstance(policy, ExecutionPolicyKtCgTrial10):
        raise TypeError("Expected KT/CG Trial #10 policy")
    tier_override = parse_int_list(args.tier_periods)
    updates: dict[str, object] = {}
    if tier_override:
        updates["tier_ema_periods"] = tier_override
    if args.tp1_close_pct is not None:
        updates["tp1_close_pct"] = float(args.tp1_close_pct)
    if updates:
        policy = policy.model_copy(update=updates)
    return profile, policy


def prepare_features(m1: pd.DataFrame, policy: ExecutionPolicyKtCgTrial10) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    m1f = m1.copy()
    ema_periods = sorted(
        set(
            [
                int(policy.m1_zone_entry_ema_fast),
                int(policy.m1_zone_entry_ema_slow),
                int(policy.tier_reclaim_ema_period),
                int(policy.m1_zone_entry_price_ema_period),
                200,
            ]
            + [int(x) for x in policy.tier_ema_periods]
        )
    )
    for period in ema_periods:
        m1f[f"ema_{period}"] = ema_fn(m1f["close"].astype(float), period)
    m1f["date_utc"] = m1f.index.floor("D")
    out["m1"] = m1f

    m5 = resample_ohlc(m1, "5min")
    m5["ema_9"] = ema_fn(m5["close"].astype(float), 9)
    m5["ema_21"] = ema_fn(m5["close"].astype(float), 21)
    m5["ema_20"] = ema_fn(m5["close"].astype(float), int(policy.trail_m5_ema_period))
    m5["gap_pips"] = (m5["ema_9"] - m5["ema_21"]).abs() / PIP_SIZE
    m5["atr_14"] = atr_fn(m5, 14)
    m5["atr_pips"] = m5["atr_14"] / PIP_SIZE
    m5["ema9_raw_slope"] = m5["ema_9"] - m5["ema_9"].shift(3)
    m5["ema21_raw_slope"] = m5["ema_21"] - m5["ema_21"].shift(3)
    m5["ema9_slope_pips_per_bar"] = (m5["ema9_raw_slope"].abs() / 3.0) / PIP_SIZE
    m5["trend"] = m5.apply(lambda row: "bull" if float(row["ema_9"]) > float(row["ema_21"]) else "bear", axis=1)
    m5["slope_aligned"] = (
        ((m5["ema9_raw_slope"] > 0) & (m5["trend"] == "bull"))
        | ((m5["ema9_raw_slope"] < 0) & (m5["trend"] == "bear"))
    )
    m5["slope_disagrees_ema21"] = (
        ((m5["ema9_raw_slope"] > 0) & (m5["ema21_raw_slope"] < 0))
        | ((m5["ema9_raw_slope"] < 0) & (m5["ema21_raw_slope"] > 0))
    )
    m5["m5_bucket"] = "normal"
    strong_mask = (
        (m5["gap_pips"] > 3.0)
        & (m5["ema9_slope_pips_per_bar"] > 0.80)
        & (m5["slope_aligned"])
    )
    weak_mask = (
        (m5["gap_pips"] < 1.1)
        | (m5["ema9_slope_pips_per_bar"] < 0.30)
        | (m5["slope_disagrees_ema21"])
    )
    m5.loc[strong_mask, "m5_bucket"] = "strong"
    m5.loc[~strong_mask & weak_mask, "m5_bucket"] = "weak"
    out["m5"] = m5

    m15 = resample_ohlc(m1, "15min")
    m15_levels: list[dict[str, float | pd.Timestamp | None]] = []
    for i in range(len(m15)):
        fib = compute_rolling_intraday_fib_levels(m15.iloc[: i + 1].reset_index(), int(policy.intraday_fib_lookback_bars))
        if fib is None:
            m15_levels.append({"time": m15.index[i], "lower": None, "upper": None})
        else:
            lower = fib.get(str(policy.intraday_fib_lower_level))
            upper = fib.get(str(policy.intraday_fib_upper_level))
            m15_levels.append({"time": m15.index[i], "lower": lower, "upper": upper})
    m15_levels_df = pd.DataFrame(m15_levels).set_index("time")
    out["m15_levels"] = m15_levels_df

    daily = m1.groupby(m1.index.floor("D")).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    )
    out["daily"] = daily
    out["prev_daily"] = daily.shift(1)
    return out


def map_features_to_m1(m1: pd.DataFrame, features: dict[str, pd.DataFrame]) -> pd.DataFrame:
    mapped = m1.copy()
    m5_map = features["m5"][["close", "trend", "gap_pips", "m5_bucket", "ema_20", "atr_pips"]].rename(
        columns={"close": "m5_close"}
    ).reindex(mapped.index, method="ffill")
    mapped = mapped.join(m5_map.rename(columns={"ema_20": "m5_trail_ema_20"}))

    m15_map = features["m15_levels"].reindex(mapped.index, method="ffill")
    mapped = mapped.join(m15_map.rename(columns={"lower": "fib_lower", "upper": "fib_upper"}))

    prev_daily = features["prev_daily"]
    day_keys = mapped.index.floor("D")
    mapped["prev_day_high"] = prev_daily["high"].reindex(day_keys).to_numpy()
    mapped["prev_day_low"] = prev_daily["low"].reindex(day_keys).to_numpy()
    return mapped


def unrealized_usd(pos: Position, close_mid: float) -> float:
    pips = ((close_mid - pos.entry_price) / PIP_SIZE) if pos.side == "buy" else ((pos.entry_price - close_mid) / PIP_SIZE)
    return pips_to_usd(pips, pos.remaining_lots, close_mid)


def pips_to_usd(pips: float, lots: float, ref_price: float) -> float:
    if lots <= 0.0 or ref_price <= 0.0:
        return 0.0
    return float(pips) * PIP_SIZE * PIP_VALUE_UNITS * float(lots) / float(ref_price)


def close_leg(pos: Position, lots: float, exit_price: float, reason: str, ref_price: float) -> tuple[float, float]:
    if lots <= 0.0:
        return 0.0, 0.0
    pips = ((exit_price - pos.entry_price) / PIP_SIZE) if pos.side == "buy" else ((pos.entry_price - exit_price) / PIP_SIZE)
    usd = pips_to_usd(pips, lots, ref_price)
    pos.weighted_pips_x_lots += pips * lots
    pos.weighted_usd += usd
    pos.remaining_lots = round(max(0.0, pos.remaining_lots - lots), 6)
    pos.exit_reason = reason
    return pips, usd


def choose_first_event(side: str, open_mid: float, stop_price: float, tp1_price: float, spread_price: float) -> str:
    half = spread_price / 2.0
    if side == "buy":
        open_px = open_mid - half
        stop_dist = abs(open_px - stop_price)
        tp_dist = abs(tp1_price - open_px)
    else:
        open_px = open_mid + half
        stop_dist = abs(stop_price - open_px)
        tp_dist = abs(open_px - tp1_price)
    if stop_dist <= tp_dist:
        return "stop"
    return "tp1"


def compute_conviction_for_bar(
    m1f: pd.DataFrame,
    i: int,
    m5_bucket: str,
    side: str,
    policy: ExecutionPolicyKtCgTrial10,
    max_lots: float,
) -> tuple[str, float, float]:
    is_bull = side == "buy"
    ema5 = float(m1f["ema_5"].iat[i])
    ema9 = float(m1f["ema_9"].iat[i])
    spread_pips = abs(ema5 - ema9) / PIP_SIZE
    aligned = ema5 > ema9 if is_bull else ema5 < ema9
    compressing = False
    if i >= 3:
        recent = ((m1f["ema_5"].iloc[i - 3 : i + 1] - m1f["ema_9"].iloc[i - 3 : i + 1]).abs() / PIP_SIZE).tolist()
        compressing = recent[1] < recent[0] and recent[2] < recent[1] and recent[3] < recent[2]
    if spread_pips < 0.23:
        compressing = True
    if (not aligned) or compressing:
        m1_bucket = "dampen"
    elif spread_pips > 0.53:
        m1_bucket = "confirm"
    else:
        m1_bucket = "neutral"
    mult = float(MULTIPLIER_MATRIX.get((str(m5_bucket), m1_bucket), 1.0))
    raw = float(policy.conviction_base_lots) * mult
    lots = round(max(float(policy.conviction_min_lots), min(float(max_lots), raw)), 2)
    return m1_bucket, mult, lots


def evaluate_signal(
    *,
    m1f: pd.DataFrame,
    mapped: pd.DataFrame,
    i: int,
    tier_state: dict[int, bool],
    policy: ExecutionPolicyKtCgTrial10,
) -> tuple[Optional[dict], Counter, dict[int, bool]]:
    reasons = Counter()
    tier_updates: dict[int, bool] = {}
    if i < 205:
        reasons["warmup"] += 1
        return None, reasons, tier_updates

    gap_pips = mapped["gap_pips"].iat[i]
    if pd.isna(gap_pips) or float(gap_pips) < float(policy.m5_min_ema_distance_pips):
        reasons["m5_gap_block"] += 1
        return None, reasons, tier_updates

    trend = str(mapped["trend"].iat[i] or "")
    if trend not in {"bull", "bear"}:
        reasons["m5_trend_missing"] += 1
        return None, reasons, tier_updates
    is_bull = trend == "bull"
    last_close = float(m1f["close"].iat[i])
    prev_low = float(m1f["low"].iat[i - 1])
    prev_high = float(m1f["high"].iat[i - 1])
    reclaim_ema_now = float(m1f[f"ema_{int(policy.tier_reclaim_ema_period)}"].iat[i])
    strong_only_tiers = {int(x) for x in getattr(policy, "strong_m5_only_tier_periods", tuple())}
    m5_bucket = str(mapped["m5_bucket"].iat[i] or "normal")
    reset_buffer = float(policy.tier_reset_buffer_pips) * PIP_SIZE

    for tier in (int(x) for x in policy.tier_ema_periods):
        ema_now = float(m1f[f"ema_{tier}"].iat[i])
        ema_prev = float(m1f[f"ema_{tier}"].iat[i - 1])
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
            pq_result = analyze_pullback_quality(
                m1_df=m1f,
                touch_index=i - 1,
                side="bull" if is_bull else "bear",
                tier=tier,
                enabled=bool(getattr(policy, "pullback_quality_enabled", True)),
                lookback_bars=int(getattr(policy, "pullback_quality_lookback_bars", 30)),
                orderly_bar_count_min=int(getattr(policy, "pullback_quality_orderly_bar_count_min", 6)),
                sloppy_bar_count_max=int(getattr(policy, "pullback_quality_sloppy_bar_count_max", 2)),
                orderly_structure_ratio_min=float(getattr(policy, "pullback_quality_orderly_structure_ratio_min", 0.6)),
                sloppy_structure_ratio_max=float(getattr(policy, "pullback_quality_sloppy_structure_ratio_max", 0.3)),
                shallow_tiers=tuple(int(x) for x in getattr(policy, "pullback_quality_shallow_tier_periods", (17, 21))),
                sloppy_lot_multiplier=float(getattr(policy, "pullback_quality_sloppy_lot_multiplier", 0.5)),
            )
            pq_snapshot = pullback_quality_snapshot(pq_result)

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

    zone_fast = float(m1f[f"ema_{int(policy.m1_zone_entry_ema_fast)}"].iat[i])
    zone_slow = float(m1f[f"ema_{int(policy.m1_zone_entry_ema_slow)}"].iat[i])
    aligned_now = zone_fast > zone_slow if is_bull else zone_fast < zone_slow
    recent_cross = False
    lookback = max(1, int(policy.zone_entry_max_cross_lookback_bars))
    for offset in range(1, min(lookback, i) + 1):
        prev_fast = float(m1f[f"ema_{int(policy.m1_zone_entry_ema_fast)}"].iat[i - offset])
        prev_slow = float(m1f[f"ema_{int(policy.m1_zone_entry_ema_slow)}"].iat[i - offset])
        curr_fast = float(m1f[f"ema_{int(policy.m1_zone_entry_ema_fast)}"].iat[i - offset + 1])
        curr_slow = float(m1f[f"ema_{int(policy.m1_zone_entry_ema_slow)}"].iat[i - offset + 1])
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
            },
            reasons,
            tier_updates,
        )
    if aligned_now and policy.zone_entry_require_recent_cross and not recent_cross:
        reasons["zone_no_recent_cross"] += 1
    else:
        reasons["zone_alignment_missing"] += 1
    return None, reasons, tier_updates


def main() -> None:
    args = parse_args()
    m1 = load_m1(args.input_csv)
    profile, policy = build_profile(args)
    features = prepare_features(m1, policy)
    m1f = features["m1"]
    mapped = map_features_to_m1(m1f, features)
    effective_risk = get_effective_risk(profile)

    ntz_filter = NoTradeZoneFilter(
        enabled=bool(policy.ntz_enabled),
        buffer_pips=float(policy.ntz_buffer_pips),
        pip_size=PIP_SIZE,
        use_prev_day_hl=bool(policy.ntz_use_prev_day_hl),
        use_weekly_hl=bool(policy.ntz_use_weekly_hl),
        use_monthly_hl=bool(policy.ntz_use_monthly_hl),
        use_fib_pivots=bool(policy.ntz_use_fib_pivots),
        use_fib_pp=bool(policy.ntz_use_fib_pp),
        use_fib_r1=bool(policy.ntz_use_fib_r1),
        use_fib_r2=bool(policy.ntz_use_fib_r2),
        use_fib_r3=bool(policy.ntz_use_fib_r3),
        use_fib_s1=bool(policy.ntz_use_fib_s1),
        use_fib_s2=bool(policy.ntz_use_fib_s2),
        use_fib_s3=bool(policy.ntz_use_fib_s3),
    )
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

    spread_price = float(args.spread_pips) * PIP_SIZE
    half_spread = spread_price / 2.0
    be_offset = spread_price + float(policy.be_spread_plus_pips) * PIP_SIZE

    open_positions: list[Position] = []
    closed_rows: list[dict] = []
    equity_rows: list[dict] = []
    pending_entry: Optional[dict] = None
    tier_state = {int(x): False for x in policy.tier_ema_periods}
    block_counts: Counter = Counter()
    signal_counts: Counter = Counter()
    balance = float(args.starting_balance)
    max_balance = balance
    max_drawdown_usd = 0.0
    trade_id = 0
    last_trade_time: Optional[pd.Timestamp] = None
    trades_per_day: Counter = Counter()
    m5_close_set = set(features["m5"].index)
    buy_boost_hours_et = parse_hour_list(args.buy_boost_hours_et) or set(BEST_BUY_HOURS_ET)
    side_hour_mults = parse_side_hour_mults(args.side_hour_mults)
    side_pause_until: dict[str, Optional[pd.Timestamp]] = {"buy": None, "sell": None}
    recent_side_events: dict[str, list[dict[str, object]]] = {"buy": [], "sell": []}

    times = list(m1f.index)
    for j in range(1, len(m1f)):
        bar = m1f.iloc[j]
        ts = times[j]
        open_mid = float(bar["open"])
        high_mid = float(bar["high"])
        low_mid = float(bar["low"])
        close_mid = float(bar["close"])
        open_bid = open_mid - half_spread
        open_ask = open_mid + half_spread
        high_bid = high_mid - half_spread
        low_bid = low_mid - half_spread
        high_ask = high_mid + half_spread
        low_ask = low_mid + half_spread
        close_bid = close_mid - half_spread
        close_ask = close_mid + half_spread

        if pending_entry is not None:
            trade_id += 1
            stop_pips = float(pending_entry["stop_pips"])
            side = str(pending_entry["side"])
            entry_price = open_ask if side == "buy" else open_bid
            stop_price = entry_price - stop_pips * PIP_SIZE if side == "buy" else entry_price + stop_pips * PIP_SIZE
            tp1_price = entry_price + float(policy.tp1_pips) * PIP_SIZE if side == "buy" else entry_price - float(policy.tp1_pips) * PIP_SIZE
            open_positions.append(
                Position(
                    trade_id=trade_id,
                    side=side,
                    entry_type=str(pending_entry["entry_type"]),
                    tier_period=pending_entry["tier"],
                    signal_time=pending_entry["signal_time"],
                    entry_time=ts,
                    entry_price=entry_price,
                    initial_lots=float(pending_entry["lots"]),
                    remaining_lots=float(pending_entry["lots"]),
                    stop_price=stop_price,
                    initial_stop_price=stop_price,
                    tp1_price=tp1_price,
                    tp1_done=False,
                    tp1_close_pct=float(policy.tp1_close_pct),
                    be_stop_price=None,
                    m5_bucket=str(pending_entry["m5_bucket"]),
                    m1_bucket=str(pending_entry["m1_bucket"]),
                    conviction_multiplier=float(pending_entry["multiplier"]),
                    conviction_lots=float(pending_entry["lots"]),
                    atr_stop_pips=stop_pips,
                    pullback_quality_label=str((pending_entry.get("pullback_quality") or {}).get("label") or ""),
                    pullback_quality_bar_count=int((pending_entry.get("pullback_quality") or {}).get("pullback_bar_count") or 0),
                    pullback_quality_structure_ratio=float((pending_entry.get("pullback_quality") or {}).get("structure_ratio") or 0.0),
                    pullback_quality_dampener=float((pending_entry.get("pullback_quality") or {}).get("dampener_multiplier") or 1.0),
                    regime_multiplier=float(pending_entry.get("regime_multiplier") or 1.0),
                    regime_label=str(pending_entry.get("regime_label") or "baseline"),
                )
            )
            trades_per_day[ts.floor("D")] += 1
            last_trade_time = ts
            signal_counts[str(pending_entry["entry_type"])] += 1
            pending_entry = None

        survivors: list[Position] = []
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
                    pos.be_stop_price = pos.entry_price + be_offset if pos.side == "buy" else pos.entry_price - be_offset
                    pos.stop_price = float(pos.be_stop_price)
                    if pos.remaining_lots <= 0.0:
                        closed = True
                    else:
                        if pos.side == "buy" and low_bid <= pos.stop_price:
                            close_leg(pos, pos.remaining_lots, pos.stop_price, "tp1_then_be_same_bar", ref_price)
                            closed = True
                        elif pos.side == "sell" and high_ask >= pos.stop_price:
                            close_leg(pos, pos.remaining_lots, pos.stop_price, "tp1_then_be_same_bar", ref_price)
                            closed = True

            if not closed and pos.tp1_done:
                if pos.side == "buy" and low_bid <= pos.stop_price:
                    close_leg(pos, pos.remaining_lots, pos.stop_price, "runner_stop", ref_price)
                    closed = True
                elif pos.side == "sell" and high_ask >= pos.stop_price:
                    close_leg(pos, pos.remaining_lots, pos.stop_price, "runner_stop", ref_price)
                    closed = True

            if not closed and bool(policy.kill_switch_enabled):
                m5_trend = str(mapped["trend"].iat[j] or "")
                ema200 = float(m1f["ema_200"].iat[j])
                if pos.side == "buy" and m5_trend == "bull" and close_mid < ema200:
                    close_leg(pos, pos.remaining_lots, close_bid, "kill_switch", ref_price)
                    closed = True
                elif pos.side == "sell" and m5_trend == "bear" and close_mid > ema200:
                    close_leg(pos, pos.remaining_lots, close_ask, "kill_switch", ref_price)
                    closed = True

            # Trial #10's M5 trail is a runner exit and should only activate after TP1.
            if not closed and pos.tp1_done and ts in m5_close_set:
                m5_close = mapped["m5_close"].iat[j]
                trail_ema = mapped["m5_trail_ema_20"].iat[j]
                if pd.notna(trail_ema) and pd.notna(m5_close):
                    if pos.side == "buy" and float(m5_close) < float(trail_ema):
                        close_leg(pos, pos.remaining_lots, close_bid, "m5_trail_close", ref_price)
                        closed = True
                    elif pos.side == "sell" and float(m5_close) > float(trail_ema):
                        close_leg(pos, pos.remaining_lots, close_ask, "m5_trail_close", ref_price)
                        closed = True

            if closed or pos.remaining_lots <= 0.0:
                balance += pos.weighted_usd
                weighted_pips = pos.weighted_pips_x_lots / pos.initial_lots if pos.initial_lots > 0 else 0.0
                if bool(args.operator_regime):
                    current_side_events = recent_side_events.setdefault(pos.side, [])
                    current_side_events.append(
                        {
                            "time": ts,
                            "initial_stop": pos.exit_reason == "initial_stop",
                            "sloppy_shallow": (
                                str(pos.pullback_quality_label).lower() == "sloppy"
                                and pos.tier_period in {17, 21}
                            ),
                        }
                    )
                    lookback_cutoff = ts - pd.Timedelta(minutes=float(args.autopause_minutes))
                    current_side_events[:] = [
                        e for e in current_side_events if pd.Timestamp(e["time"]) >= lookback_cutoff
                    ]
                    stop_count = sum(1 for e in current_side_events if bool(e["initial_stop"]))
                    sloppy_count = sum(1 for e in current_side_events if bool(e["sloppy_shallow"]))
                    sloppy_stop_mix = any(bool(e["sloppy_shallow"]) for e in current_side_events) and stop_count >= 2
                    if stop_count >= int(args.autopause_stop_threshold) or sloppy_count >= 2 or sloppy_stop_mix:
                        side_pause_until[pos.side] = ts + pd.Timedelta(minutes=float(args.autopause_minutes))
                closed_rows.append(
                    {
                        "trade_id": pos.trade_id,
                        "side": pos.side,
                        "entry_type": pos.entry_type,
                        "tier_period": pos.tier_period,
                        "signal_time": pos.signal_time.isoformat(),
                        "entry_time": pos.entry_time.isoformat(),
                        "exit_time": ts.isoformat(),
                        "entry_price": round(pos.entry_price, 5),
                        "initial_stop_price": round(pos.initial_stop_price, 5),
                        "initial_lots": round(pos.initial_lots, 2),
                        "weighted_pips": round(weighted_pips, 3),
                        "profit_usd": round(pos.weighted_usd, 4),
                        "exit_reason": pos.exit_reason,
                        "tp1_done": pos.tp1_done,
                        "m5_bucket": pos.m5_bucket,
                        "m1_bucket": pos.m1_bucket,
                        "conviction_multiplier": pos.conviction_multiplier,
                        "conviction_lots": pos.conviction_lots,
                        "atr_stop_pips": round(pos.atr_stop_pips, 3),
                        "pullback_quality_label": pos.pullback_quality_label,
                        "pullback_quality_bar_count": pos.pullback_quality_bar_count,
                        "pullback_quality_structure_ratio": round(pos.pullback_quality_structure_ratio, 4),
                        "pullback_quality_dampener": pos.pullback_quality_dampener,
                        "regime_multiplier": pos.regime_multiplier,
                        "regime_label": pos.regime_label,
                        "hold_minutes": round((ts - pos.entry_time).total_seconds() / 60.0, 1),
                    }
                )
            else:
                survivors.append(pos)
        open_positions = survivors

        equity = balance + sum(unrealized_usd(pos, close_mid) for pos in open_positions)
        max_balance = max(max_balance, equity)
        max_drawdown_usd = max(max_drawdown_usd, max_balance - equity)
        equity_rows.append({"time": ts.isoformat(), "balance": round(balance, 4), "equity": round(equity, 4), "open_positions": len(open_positions)})

        lower = mapped["fib_lower"].iat[j]
        upper = mapped["fib_upper"].iat[j]
        corridor_filter.update_levels(
            {"S1": float(lower), "R1": float(upper)} if pd.notna(lower) and pd.notna(upper) else None,
            None,
            None,
            source_close=close_mid,
            calculation_mode="rolling_window",
        )
        corridor_ok, corridor_reason = corridor_filter.check_corridor(close_mid)

        ntz_filter.update_levels(
            prev_day_high=float(mapped["prev_day_high"].iat[j]) if pd.notna(mapped["prev_day_high"].iat[j]) else None,
            prev_day_low=float(mapped["prev_day_low"].iat[j]) if pd.notna(mapped["prev_day_low"].iat[j]) else None,
        )

        signal, signal_reason_counts, tier_updates = evaluate_signal(
            m1f=m1f,
            mapped=mapped,
            i=j,
            tier_state=tier_state,
            policy=policy,
        )
        block_counts.update(signal_reason_counts)
        for tier, value in tier_updates.items():
            tier_state[int(tier)] = bool(value)

        if signal is None or j >= len(m1f) - 1:
            continue

        current_price = close_mid
        current_hour_et = hour_et(ts)
        blocked, ntz_reason = ntz_filter.is_in_no_trade_zone(current_price)
        if blocked:
            block_counts["ntz_block"] += 1
            block_counts[ntz_reason] += 1
            continue
        if not corridor_ok:
            block_counts["intraday_fib_block"] += 1
            block_counts[corridor_reason or "intraday_fib_block"] += 1
            continue

        m5_trend = str(mapped["trend"].iat[j] or "")
        ema200 = float(m1f["ema_200"].iat[j])
        if bool(policy.kill_switch_enabled):
            if signal["side"] == "buy" and m5_trend == "bull" and close_mid < ema200:
                block_counts["kill_switch_entry_block"] += 1
                continue
            if signal["side"] == "sell" and m5_trend == "bear" and close_mid > ema200:
                block_counts["kill_switch_entry_block"] += 1
                continue
        if bool(args.operator_regime):
            pause_until = side_pause_until.get(str(signal["side"]))
            if pause_until is not None and ts < pause_until:
                block_counts[f"{signal['side']}_autopause"] += 1
                continue
        if bool(args.london_sell_veto) and str(signal["side"]) == "sell" and current_hour_et in LONDON_HOURS_ET:
            block_counts["london_sell_veto"] += 1
            continue

        current_day = ts.floor("D")
        if trades_per_day[current_day] >= int(effective_risk.max_trades_per_day):
            block_counts["max_trades_per_day"] += 1
            continue
        if len(open_positions) >= int(effective_risk.max_open_trades):
            block_counts["max_open_trades"] += 1
            continue
        side_open = sum(1 for p in open_positions if p.side == signal["side"])
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

        atr_pips = mapped["atr_pips"].iat[j]
        fallback_sl = float(policy.sl_pips)
        stop_pips = fallback_sl
        if profile.trade_management.stop_loss and profile.trade_management.stop_loss.mode == "atr" and pd.notna(atr_pips):
            stop_pips = min(float(atr_pips) * float(profile.trade_management.stop_loss.atr_multiplier), float(profile.trade_management.stop_loss.max_sl_pips))
            stop_pips = max(stop_pips, float(effective_risk.min_stop_pips))

        m1_bucket, mult, lots = compute_conviction_for_bar(
            m1f=m1f,
            i=j,
            m5_bucket=str(signal["m5_bucket"]),
            side=str(signal["side"]),
            policy=policy,
            max_lots=float(effective_risk.max_lots),
        )
        pending_entry = {
            "signal_time": ts,
            "side": str(signal["side"]),
            "entry_type": str(signal["entry_type"]),
            "tier": signal["tier"],
            "m5_bucket": str(signal["m5_bucket"]),
            "m1_bucket": m1_bucket,
            "multiplier": mult,
            "lots": lots,
            "stop_pips": float(stop_pips),
            "pullback_quality": signal.get("pullback_quality"),
        }
        pq = signal.get("pullback_quality") or {}
        if (
            signal["entry_type"] == "tiered_pullback"
            and bool(pq.get("applicable", False))
            and str(pq.get("label", "")).lower() == "sloppy"
        ):
            damp = max(0.0, float(pq.get("dampener_multiplier", 1.0)))
            pending_entry["lots"] = round(max(float(policy.conviction_min_lots), float(pending_entry["lots"]) * damp), 2)
        regime_mult, regime_label = resolve_regime_multiplier(
            ts=ts,
            side=str(signal["side"]),
            operator_regime=bool(args.operator_regime),
            buy_boost_hours_et=buy_boost_hours_et,
            buy_boost_mult=float(args.buy_boost_mult),
            buy_base_mult=float(args.buy_base_mult),
            sell_base_mult=float(args.sell_base_mult),
            side_hour_mults=side_hour_mults,
        )
        pending_entry["regime_multiplier"] = regime_mult
        pending_entry["regime_label"] = regime_label
        pending_entry["lots"] = round(
            min(
                float(effective_risk.max_lots),
                max(float(policy.conviction_min_lots), float(pending_entry["lots"]) * regime_mult),
            ),
            2,
        )
        if (
            signal["entry_type"] == "tiered_pullback"
            and int(signal.get("tier") or 0) == 17
            and regime_label != "buy_boost_hour"
            and float(args.tier17_nonboost_mult) != 1.0
        ):
            pending_entry["lots"] = round(
                min(
                    float(effective_risk.max_lots),
                    max(
                        float(policy.conviction_min_lots),
                        float(pending_entry["lots"]) * float(args.tier17_nonboost_mult),
                    ),
                ),
                2,
            )

    if len(m1f) > 0 and open_positions:
        final_ts = m1f.index[-1]
        final_close = float(m1f["close"].iat[-1])
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
                    "signal_time": pos.signal_time.isoformat(),
                    "entry_time": pos.entry_time.isoformat(),
                    "exit_time": final_ts.isoformat(),
                    "entry_price": round(pos.entry_price, 5),
                    "initial_stop_price": round(pos.initial_stop_price, 5),
                    "initial_lots": round(pos.initial_lots, 2),
                    "weighted_pips": round(weighted_pips, 3),
                    "profit_usd": round(pos.weighted_usd, 4),
                    "exit_reason": pos.exit_reason,
                    "tp1_done": pos.tp1_done,
                    "m5_bucket": pos.m5_bucket,
                    "m1_bucket": pos.m1_bucket,
                    "conviction_multiplier": pos.conviction_multiplier,
                    "conviction_lots": pos.conviction_lots,
                    "atr_stop_pips": round(pos.atr_stop_pips, 3),
                    "pullback_quality_label": pos.pullback_quality_label,
                    "pullback_quality_bar_count": pos.pullback_quality_bar_count,
                    "pullback_quality_structure_ratio": round(pos.pullback_quality_structure_ratio, 4),
                    "pullback_quality_dampener": pos.pullback_quality_dampener,
                    "regime_multiplier": pos.regime_multiplier,
                    "regime_label": pos.regime_label,
                    "hold_minutes": round((final_ts - pos.entry_time).total_seconds() / 60.0, 1),
                }
            )

    trades_df = pd.DataFrame(closed_rows)
    equity_df = pd.DataFrame(equity_rows)
    wins = trades_df[trades_df["profit_usd"] > 0] if not trades_df.empty else pd.DataFrame()
    losses = trades_df[trades_df["profit_usd"] < 0] if not trades_df.empty else pd.DataFrame()
    gross_profit = float(wins["profit_usd"].sum()) if not wins.empty else 0.0
    gross_loss = float(losses["profit_usd"].sum()) if not losses.empty else 0.0
    profit_factor = (gross_profit / abs(gross_loss)) if gross_loss < 0 else None

    ntz_examples = {k: int(v) for k, v in block_counts.items() if str(k).startswith("NTZ:")}
    corridor_examples = {k: int(v) for k, v in block_counts.items() if str(k).startswith("intraday_fib_corridor:")}
    core_block_counts = {
        str(k): int(v)
        for k, v in block_counts.items()
        if not str(k).startswith("NTZ:") and not str(k).startswith("intraday_fib_corridor:")
    }
    summary = {
        "input_csv": str(Path(args.input_csv).resolve()),
        "bars": int(len(m1f)),
        "start": m1f.index[0].isoformat() if len(m1f) else None,
        "end": m1f.index[-1].isoformat() if len(m1f) else None,
        "spread_pips": float(args.spread_pips),
        "starting_balance_usd": float(args.starting_balance),
        "ending_balance_usd": round(balance, 4),
        "net_profit_usd": round(balance - float(args.starting_balance), 4),
        "max_drawdown_usd": round(max_drawdown_usd, 4),
        "trades": int(len(trades_df)),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "win_rate_pct": round((len(wins) / len(trades_df) * 100.0), 2) if len(trades_df) else 0.0,
        "profit_factor": round(profit_factor, 4) if profit_factor is not None else None,
        "avg_trade_usd": round(float(trades_df["profit_usd"].mean()), 4) if len(trades_df) else 0.0,
        "avg_trade_pips": round(float(trades_df["weighted_pips"].mean()), 4) if len(trades_df) else 0.0,
        "by_entry_type": trades_df["entry_type"].value_counts().to_dict() if len(trades_df) else {},
        "by_exit_reason": trades_df["exit_reason"].value_counts().to_dict() if len(trades_df) else {},
        "by_tier": {
            str(k): int(v)
            for k, v in trades_df["tier_period"].dropna().astype(int).value_counts().to_dict().items()
        } if len(trades_df) else {},
        "signal_counts": dict(signal_counts),
        "block_counts": dict(sorted(core_block_counts.items(), key=lambda kv: kv[1], reverse=True)[:50]),
        "ntz_block_examples": dict(sorted(ntz_examples.items(), key=lambda kv: kv[1], reverse=True)[:15]),
        "corridor_block_examples": dict(sorted(corridor_examples.items(), key=lambda kv: kv[1], reverse=True)[:15]),
        "config": {
            "preset": "kt_cg_trial_10",
            "operator_regime": bool(args.operator_regime),
            "buy_boost_hours_et": sorted(int(x) for x in buy_boost_hours_et),
            "buy_boost_mult": float(args.buy_boost_mult),
            "buy_base_mult": float(args.buy_base_mult),
            "sell_base_mult": float(args.sell_base_mult),
            "side_hour_mults": {
                f"{side}:{hour}": float(mult)
                for (side, hour), mult in sorted(side_hour_mults.items())
            },
            "london_sell_veto": bool(args.london_sell_veto),
            "autopause_minutes": float(args.autopause_minutes),
            "autopause_stop_threshold": int(args.autopause_stop_threshold),
            "tier17_nonboost_mult": float(args.tier17_nonboost_mult),
            "zone_entry_mode": str(policy.zone_entry_mode),
            "tier_ema_periods": [int(x) for x in policy.tier_ema_periods],
            "strong_m5_only_tier_periods": [int(x) for x in policy.strong_m5_only_tier_periods],
            "kill_switch_enabled": bool(policy.kill_switch_enabled),
            "pullback_quality_enabled": bool(getattr(policy, "pullback_quality_enabled", True)),
            "pullback_quality_shallow_tier_periods": [int(x) for x in getattr(policy, "pullback_quality_shallow_tier_periods", (17, 21))],
            "pullback_quality_sloppy_lot_multiplier": float(getattr(policy, "pullback_quality_sloppy_lot_multiplier", 0.5)),
            "tp1_pips": float(policy.tp1_pips),
            "tp1_close_pct": float(policy.tp1_close_pct),
            "trail_m5_ema_period": int(policy.trail_m5_ema_period),
            "atr_mode": str(profile.trade_management.stop_loss.mode) if profile.trade_management.stop_loss else "fixed",
            "atr_multiplier": float(profile.trade_management.stop_loss.atr_multiplier) if profile.trade_management.stop_loss else None,
            "atr_max_sl_pips": float(profile.trade_management.stop_loss.max_sl_pips) if profile.trade_management.stop_loss else None,
            "conviction_enabled": bool(policy.conviction_sizing_enabled),
            "conviction_base_lots": float(policy.conviction_base_lots),
            "conviction_min_lots": float(policy.conviction_min_lots),
            "max_lots": float(effective_risk.max_lots),
        },
    }

    out_path = Path(args.out)
    trades_out = Path(args.trades_out)
    equity_out = Path(args.equity_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trades_out.parent.mkdir(parents=True, exist_ok=True)
    equity_out.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    trades_df.to_csv(trades_out, index=False)
    equity_df.to_csv(equity_out, index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
