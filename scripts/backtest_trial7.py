#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pandas as pd

from core.execution_engine import (
    _evaluate_kt_cg_trial_7_zone_only_candidate,
    evaluate_kt_cg_trial_7_conditions,
)
from core.reversal_risk import compute_reversal_risk_score, get_rr_exhaustion_threshold_boost_pips


PIP_SIZE = 0.01


@dataclass
class Position:
    trade_id: int
    side: str  # buy/sell
    entry_type: str  # zone_entry/tiered_pullback
    tier_period: Optional[int]
    entry_time: pd.Timestamp
    entry_price: float
    stop_price: float
    target_price: float
    lots: float
    rr_score: Optional[float]
    rr_tier: Optional[str]
    rr_use_managed_exit: bool
    rr_lot_multiplier: float
    exhaustion_zone: str


@dataclass
class ClosedTrade:
    trade_id: int
    side: str
    entry_type: str
    tier_period: Optional[int]
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    lots: float
    pips: float
    usd: float
    exit_reason: str
    rr_tier: Optional[str]
    rr_score: Optional[float]
    exhaustion_zone: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest Trial #7 with calibrated RR and exhaustion settings.")
    p.add_argument("--in", dest="inputs", action="append", required=True, help="Input M1 CSV path (repeatable)")
    p.add_argument("--spread-pips", type=float, default=1.8, help="Fixed spread in pips (default: 1.8)")
    p.add_argument("--base-lot", type=float, default=0.1, help="Base lot size before RR multipliers (default: 0.1)")
    p.add_argument("--tp-pips", type=float, default=6.0, help="TP in pips (default: 6.0)")
    p.add_argument("--sl-pips", type=float, default=20.0, help="SL in pips (default: 20.0)")
    p.add_argument("--m5-min-gap-pips", type=float, default=1.5, help="Trial #7 M5 EMA9/21 minimum distance in pips")
    p.add_argument("--out", default="research_out/trial7_backtest_report.json", help="Output JSON report path")
    return p.parse_args()


def load_m1(paths: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for p in paths:
        f = Path(p)
        if not f.exists():
            raise FileNotFoundError(f"Missing input CSV: {f}")
        df = pd.read_csv(f)
        need = {"time", "open", "high", "low", "close"}
        if not need.issubset(df.columns):
            raise ValueError(f"{f} missing required columns: {sorted(need)}")
        df = df[["time", "open", "high", "low", "close"]].copy()
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        for c in ("open", "high", "low", "close"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna().sort_values("time")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["time"], keep="last").sort_values("time").reset_index(drop=True)
    return out


def resample_ohlc(m1: pd.DataFrame, rule: str) -> pd.DataFrame:
    d = m1.set_index("time").sort_index()
    out = (
        d.resample(rule, label="right", closed="right")
        .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
        .dropna()
        .reset_index()
    )
    return out


def session_name(ts: pd.Timestamp) -> str:
    h = int(ts.hour)
    if 0 <= h < 8:
        return "tokyo"
    if 8 <= h < 13:
        return "london"
    return "ny"


def build_policy(*, tp_pips: float, sl_pips: float, m5_min_gap_pips: float) -> SimpleNamespace:
    # Exactly as requested by user for first pass backtest.
    return SimpleNamespace(
        type="kt_cg_trial_7",
        id="kt_cg_trial_7_backtest",
        enabled=True,
        # Core entry/trend
        m5_trend_ema_fast=9,
        m5_trend_ema_slow=21,
        m5_min_ema_distance_pips=float(m5_min_gap_pips),
        zone_entry_enabled=True,
        zone_entry_mode="ema_cross",
        m1_zone_entry_ema_fast=5,
        m1_zone_entry_ema_slow=9,
        tiered_pullback_enabled=True,
        tier_ema_periods=tuple([9] + list(range(11, 35))),
        tier_reset_buffer_pips=1.0,
        cooldown_minutes=3.0,
        sl_pips=float(sl_pips),
        tp_pips=float(tp_pips),
        confirm_bars=1,
        # Caps
        max_open_trades_per_side=5,
        max_zone_entry_open=3,
        max_tiered_pullback_open=6,
        # Keep zone slope filter ON per latest request
        ema_zone_filter_enabled=True,
        ema_zone_filter_lookback_bars=3,
        ema_zone_filter_ema5_min_slope_pips_per_bar=0.10,
        ema_zone_filter_ema9_min_slope_pips_per_bar=0.08,
        ema_zone_filter_ema21_min_slope_pips_per_bar=0.05,
        spread_aware_be_enabled=False,
        session_boundary_block_enabled=False,
        # Trend exhaustion
        trend_exhaustion_enabled=True,
        trend_exhaustion_mode="session_and_side",
        trend_exhaustion_use_current_price=True,
        trend_exhaustion_hysteresis_pips=0.5,
        trend_exhaustion_p80_global=10.03,
        trend_exhaustion_p90_global=13.02,
        trend_exhaustion_p80_tokyo=12.67,
        trend_exhaustion_p90_tokyo=17.63,
        trend_exhaustion_p80_london=11.06,
        trend_exhaustion_p90_london=14.41,
        trend_exhaustion_p80_ny=12.66,
        trend_exhaustion_p90_ny=18.83,
        trend_exhaustion_p80_bull_tokyo=11.85,
        trend_exhaustion_p90_bull_tokyo=15.52,
        trend_exhaustion_p80_bull_london=10.21,
        trend_exhaustion_p90_bull_london=12.97,
        trend_exhaustion_p80_bull_ny=11.21,
        trend_exhaustion_p90_bull_ny=15.84,
        trend_exhaustion_p80_bear_tokyo=13.44,
        trend_exhaustion_p90_bear_tokyo=19.73,
        trend_exhaustion_p80_bear_london=12.01,
        trend_exhaustion_p90_bear_london=17.44,
        trend_exhaustion_p80_bear_ny=13.97,
        trend_exhaustion_p90_bear_ny=21.51,
        trend_exhaustion_extended_disable_zone_entry=True,
        trend_exhaustion_very_extended_disable_zone_entry=True,
        trend_exhaustion_extended_min_tier_period=21,
        trend_exhaustion_very_extended_min_tier_period=29,
        trend_exhaustion_very_extended_tighten_caps=True,
        trend_exhaustion_very_extended_cap_multiplier=0.5,
        trend_exhaustion_very_extended_cap_min=1,
        trend_exhaustion_adaptive_tp_enabled=False,
        trend_exhaustion_tp_extended_offset_pips=1.0,
        trend_exhaustion_tp_very_extended_offset_pips=2.0,
        trend_exhaustion_tp_min_pips=0.5,
        # Reversal risk
        use_reversal_risk_score=True,
        rr_weight_rsi_divergence=55,
        rr_weight_adr_exhaustion=20,
        rr_weight_htf_proximity=15,
        rr_weight_ema_spread=10,
        rr_rsi_period=9,
        rr_rsi_lookback_bars=20,
        rr_rsi_severity_midpoint=18.0,
        rr_adr_period=14,
        rr_adr_ramp_start_pct=75.0,
        rr_adr_score_at_100_pct=0.3,
        rr_adr_score_at_120_pct=0.6,
        rr_adr_score_at_150_pct=0.9,
        rr_ema_spread_threshold_pips=4.22,
        rr_ema_spread_max_pips=8.0,
        rr_htf_buffer_pips=5.0,
        rr_htf_swing_lookback=30,
        rr_tier_medium=58.0,
        rr_tier_high=65.0,
        rr_tier_critical=71.0,
        rr_medium_lot_multiplier=0.75,
        rr_high_lot_multiplier=0.50,
        rr_critical_lot_multiplier=0.25,
        rr_high_min_tier_ema=21,
        rr_critical_min_tier_ema=26,
        rr_block_zone_entry_above_tier="high",
        rr_adjust_exhaustion_thresholds=True,
        rr_exhaustion_medium_threshold_boost_pips=0.5,
        rr_exhaustion_high_threshold_boost_pips=1.0,
        rr_exhaustion_critical_threshold_boost_pips=1.5,
        rr_use_managed_exit_at="high",
        rr_managed_exit_hard_sl_pips=50.0,
        rr_managed_exit_max_hold_underwater_min=30.0,
        rr_managed_exit_trail_activation_pips=4.0,
        rr_managed_exit_trail_distance_pips=2.5,
        # Keep regime-adaptive ON as requested
        rr_regime_adaptive_enabled=True,
    )


def _rr_tier_rank(t: Optional[str]) -> int:
    x = str(t or "").lower()
    if x == "medium":
        return 1
    if x == "high":
        return 2
    if x == "critical":
        return 3
    return 0


def _rr_tier_at_or_above(t: Optional[str], threshold: str) -> bool:
    return _rr_tier_rank(t) >= _rr_tier_rank(threshold)


def compute_exhaustion(
    policy: SimpleNamespace,
    m5_df: pd.DataFrame,
    current_price: float,
    pip_size: float,
    trend_side: str,
    ts: pd.Timestamp,
    rr_tier: Optional[str] = None,
) -> dict:
    out = {
        "zone": "normal",
        "stretch_pips": None,
        "threshold_p80": None,
        "threshold_p90": None,
        "session": session_name(ts).upper(),
        "trend_side": trend_side,
    }
    if m5_df is None or m5_df.empty or len(m5_df) < 22:
        out["reason"] = "insufficient_m5_bars"
        return out

    close = m5_df["close"].astype(float)
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema21_last = float(ema21.iloc[-1])
    price_ref = float(current_price if bool(getattr(policy, "trend_exhaustion_use_current_price", True)) else close.iloc[-1])
    stretch_pips = abs(price_ref - ema21_last) / pip_size

    sess = session_name(ts)
    mode = str(getattr(policy, "trend_exhaustion_mode", "session_and_side"))
    if mode == "global":
        p80 = float(getattr(policy, "trend_exhaustion_p80_global", 12.03))
        p90 = float(getattr(policy, "trend_exhaustion_p90_global", 17.02))
    elif mode == "session":
        p80 = float(getattr(policy, f"trend_exhaustion_p80_{sess}", getattr(policy, "trend_exhaustion_p80_global", 12.03)))
        p90 = float(getattr(policy, f"trend_exhaustion_p90_{sess}", getattr(policy, "trend_exhaustion_p90_global", 17.02)))
    else:
        side_key = "bull" if trend_side == "bull" else "bear"
        p80 = float(getattr(policy, f"trend_exhaustion_p80_{side_key}_{sess}", getattr(policy, "trend_exhaustion_p80_global", 12.03)))
        p90 = float(getattr(policy, f"trend_exhaustion_p90_{side_key}_{sess}", getattr(policy, "trend_exhaustion_p90_global", 17.02)))

    if bool(getattr(policy, "rr_adjust_exhaustion_thresholds", True)) and rr_tier:
        boost = float(get_rr_exhaustion_threshold_boost_pips(policy, rr_tier))
        p80 += boost
        p90 += boost
        out["rr_threshold_boost_pips"] = round(boost, 3)

    if p90 < p80:
        p80, p90 = p90, p80
    hyst = max(0.0, float(getattr(policy, "trend_exhaustion_hysteresis_pips", 0.5)))

    if stretch_pips >= p90 + hyst:
        zone = "very_extended"
    elif stretch_pips >= p80 + hyst:
        zone = "extended"
    else:
        zone = "normal"

    out.update(
        {
            "zone": zone,
            "stretch_pips": round(stretch_pips, 3),
            "threshold_p80": round(p80, 3),
            "threshold_p90": round(p90, 3),
        }
    )
    return out


def pip_value_usd_per_lot(price: float) -> float:
    # Approx for USDJPY: 1 pip (0.01) on 1 lot (100k) ~= 1000 / price USD
    p = max(1e-6, float(price))
    return 1000.0 / p


def backtest(m1: pd.DataFrame, spread_pips: float, base_lot: float, tp_pips: float, sl_pips: float, m5_min_gap_pips: float) -> dict:
    policy = build_policy(tp_pips=tp_pips, sl_pips=sl_pips, m5_min_gap_pips=m5_min_gap_pips)
    profile = SimpleNamespace(profile_name="backtest", symbol="USDJPY", pip_size=PIP_SIZE)

    m5 = resample_ohlc(m1, "5min")
    m15 = resample_ohlc(m1, "15min")
    h4 = resample_ohlc(m1, "4h")
    d1 = resample_ohlc(m1, "1D")

    m5_times = m5["time"].tolist()
    m15_times = m15["time"].tolist()
    h4_times = h4["time"].tolist()
    d1_times = d1["time"].tolist()

    p5 = p15 = p4 = pdx = -1

    tier_state: dict[int, bool] = {}
    open_positions: list[Position] = []
    closed_trades: list[ClosedTrade] = []
    blocked_reasons: dict[str, int] = {}

    trade_id_seq = 0
    last_trade_time: Optional[pd.Timestamp] = None

    daily_state = {
        "date_utc": None,
        "today_open": None,
        "today_high": None,
        "today_low": None,
        "prev_day_high": None,
        "prev_day_low": None,
    }

    max_open_total = 0
    max_open_buy = 0
    max_open_sell = 0

    half_spread = (spread_pips * PIP_SIZE) / 2.0

    def close_position(pos: Position, ts: pd.Timestamp, exit_price: float, reason: str) -> None:
        nonlocal open_positions
        pips = ((exit_price - pos.entry_price) / PIP_SIZE) if pos.side == "buy" else ((pos.entry_price - exit_price) / PIP_SIZE)
        usd = pips * pip_value_usd_per_lot(exit_price) * pos.lots
        closed_trades.append(
            ClosedTrade(
                trade_id=pos.trade_id,
                side=pos.side,
                entry_type=pos.entry_type,
                tier_period=pos.tier_period,
                entry_time=pos.entry_time,
                exit_time=ts,
                entry_price=pos.entry_price,
                exit_price=exit_price,
                lots=pos.lots,
                pips=float(pips),
                usd=float(usd),
                exit_reason=reason,
                rr_tier=pos.rr_tier,
                rr_score=pos.rr_score,
                exhaustion_zone=pos.exhaustion_zone,
            )
        )
        open_positions = [x for x in open_positions if x.trade_id != pos.trade_id]

    for i in range(len(m1)):
        row = m1.iloc[i]
        ts = pd.Timestamp(row["time"])
        op = float(row["open"])
        hi = float(row["high"])
        lo = float(row["low"])
        cl = float(row["close"])

        mid = cl
        bid = mid - half_spread
        ask = mid + half_spread
        tick = SimpleNamespace(bid=bid, ask=ask)

        # Advance TF pointers
        while p5 + 1 < len(m5_times) and m5_times[p5 + 1] <= ts:
            p5 += 1
        while p15 + 1 < len(m15_times) and m15_times[p15 + 1] <= ts:
            p15 += 1
        while p4 + 1 < len(h4_times) and h4_times[p4 + 1] <= ts:
            p4 += 1
        while pdx + 1 < len(d1_times) and d1_times[pdx + 1] <= ts:
            pdx += 1

        # Daily state update
        dkey = ts.date().isoformat()
        if daily_state["date_utc"] != dkey:
            daily_state["date_utc"] = dkey
            daily_state["today_open"] = mid
            daily_state["today_high"] = mid
            daily_state["today_low"] = mid
            # prev day from available complete D1 rows before current day
            prev = d1[d1["time"] < pd.Timestamp(ts.date(), tz="UTC")]
            if not prev.empty:
                last_prev = prev.iloc[-1]
                daily_state["prev_day_high"] = float(last_prev["high"])
                daily_state["prev_day_low"] = float(last_prev["low"])
        else:
            daily_state["today_high"] = max(float(daily_state["today_high"]), mid)
            daily_state["today_low"] = min(float(daily_state["today_low"]), mid)

        # Process open positions for TP/SL and managed-exit
        bid_low = lo - half_spread
        bid_high = hi - half_spread
        ask_low = lo + half_spread
        ask_high = hi + half_spread

        # iterate over snapshot list to allow closes
        for pos in list(open_positions):
            # 1) hard TP/SL checks (conservative: if both hit same bar -> SL first)
            if pos.side == "buy":
                sl_hit = bid_low <= pos.stop_price
                tp_hit = bid_high >= pos.target_price
                if sl_hit:
                    close_position(pos, ts, pos.stop_price, "sl")
                    continue
                if tp_hit:
                    close_position(pos, ts, pos.target_price, "tp")
                    continue
            else:
                sl_hit = ask_high >= pos.stop_price
                tp_hit = ask_low <= pos.target_price
                if sl_hit:
                    close_position(pos, ts, pos.stop_price, "sl")
                    continue
                if tp_hit:
                    close_position(pos, ts, pos.target_price, "tp")
                    continue

            # 2) managed exit for high/critical RR entries
            if pos.rr_use_managed_exit:
                current_exit = bid if pos.side == "buy" else ask
                current_pips = ((current_exit - pos.entry_price) / PIP_SIZE) if pos.side == "buy" else ((pos.entry_price - current_exit) / PIP_SIZE)
                age_min = (ts - pos.entry_time).total_seconds() / 60.0

                # hard disaster close
                if current_pips <= -float(policy.rr_managed_exit_hard_sl_pips):
                    close_position(pos, ts, current_exit, "managed_exit_hard_sl")
                    continue

                # time-based underwater close
                if age_min >= float(policy.rr_managed_exit_max_hold_underwater_min) and current_pips < 0:
                    close_position(pos, ts, current_exit, "managed_exit_time_underwater")
                    continue

                # trailing stop update
                if current_pips >= float(policy.rr_managed_exit_trail_activation_pips):
                    if pos.side == "buy":
                        new_sl = bid - float(policy.rr_managed_exit_trail_distance_pips) * PIP_SIZE
                        if new_sl > pos.stop_price:
                            pos.stop_price = new_sl
                    else:
                        new_sl = ask + float(policy.rr_managed_exit_trail_distance_pips) * PIP_SIZE
                        if new_sl < pos.stop_price:
                            pos.stop_price = new_sl

        # Build TF slices (trimmed windows for speed)
        m1_slice = m1.iloc[max(0, i - 900) : i + 1][["time", "open", "high", "low", "close"]].copy()
        m5_slice = m5.iloc[max(0, p5 - 600) : p5 + 1][["time", "open", "high", "low", "close"]].copy() if p5 >= 0 else pd.DataFrame(columns=["time", "open", "high", "low", "close"])
        m15_slice = m15.iloc[max(0, p15 - 600) : p15 + 1][["time", "open", "high", "low", "close"]].copy() if p15 >= 0 else pd.DataFrame(columns=["time", "open", "high", "low", "close"])
        h4_slice = h4.iloc[max(0, p4 - 200) : p4 + 1][["time", "open", "high", "low", "close"]].copy() if p4 >= 0 else pd.DataFrame(columns=["time", "open", "high", "low", "close"])
        # D for RR: exclude current incomplete day by construction
        d_cut = pd.Timestamp(ts.date(), tz="UTC")
        d_rr = d1[d1["time"] < d_cut][["time", "open", "high", "low", "close"]].copy()

        data_eval = {
            "M1": m1_slice,
            "M5": m5_slice,
            "M15": m15_slice,
            "H4": h4_slice,
            "D": d_rr,
            "_trial7_daily_state": dict(daily_state),
        }

        # Evaluate candidate
        result = evaluate_kt_cg_trial_7_conditions(
            profile,
            policy,
            data_eval,
            current_bid=bid,
            current_ask=ask,
            tier_state=tier_state,
        )

        # Persist tier resets immediately
        for t, st in (result.get("tier_updates") or {}).items():
            if not bool(st):
                tier_state[int(t)] = False

        if not result.get("passed") or result.get("side") is None:
            continue

        side = str(result.get("side"))
        trigger_type = str(result.get("trigger_type") or "")
        tier_period = result.get("tiered_pullback_tier")
        trend_side = str(result.get("m5_trend") or ("bull" if side == "buy" else "bear"))

        # Reversal risk score
        rr_result: Optional[dict] = None
        rr_response: dict = {}
        rr_tier: Optional[str] = None
        rr_lot_mult = 1.0
        rr_min_tier_ema: Optional[int] = None

        if bool(getattr(policy, "use_reversal_risk_score", False)):
            try:
                rr_result = compute_reversal_risk_score(
                    policy=policy,
                    data_by_tf=data_eval,
                    tick=tick,
                    pip_size=PIP_SIZE,
                    trend_side=trend_side,
                    trial7_daily_state=dict(daily_state),
                )
                rr_response = (rr_result.get("response") or {}) if isinstance(rr_result, dict) else {}
                rr_tier = str(rr_result.get("tier") or "low") if isinstance(rr_result, dict) else "low"
                rr_lot_mult = float(rr_response.get("lot_multiplier", 1.0))
                rr_min_tier_ema = rr_response.get("min_tier_ema")
            except Exception as e:
                blocked_reasons[f"rr_error:{type(e).__name__}"] = blocked_reasons.get(f"rr_error:{type(e).__name__}", 0) + 1

        # Trend exhaustion
        exhaustion = compute_exhaustion(
            policy=policy,
            m5_df=m5_slice,
            current_price=(bid + ask) / 2.0,
            pip_size=PIP_SIZE,
            trend_side=trend_side,
            ts=ts,
            rr_tier=rr_tier,
        )
        ex_zone = str(exhaustion.get("zone", "normal"))

        block_zone_by_ex = False
        ex_min_tier = None
        cap_multiplier = 1.0
        cap_min = 1
        if bool(getattr(policy, "trend_exhaustion_enabled", False)):
            if ex_zone == "extended":
                block_zone_by_ex = bool(getattr(policy, "trend_exhaustion_extended_disable_zone_entry", True))
                ex_min_tier = int(getattr(policy, "trend_exhaustion_extended_min_tier_period", 21))
            elif ex_zone == "very_extended":
                block_zone_by_ex = bool(getattr(policy, "trend_exhaustion_very_extended_disable_zone_entry", True))
                ex_min_tier = int(getattr(policy, "trend_exhaustion_very_extended_min_tier_period", 29))
                if bool(getattr(policy, "trend_exhaustion_very_extended_tighten_caps", True)):
                    cap_multiplier = max(0.05, float(getattr(policy, "trend_exhaustion_very_extended_cap_multiplier", 0.5)))
                    cap_min = max(1, int(getattr(policy, "trend_exhaustion_very_extended_cap_min", 1)))

        def try_zone_fallback() -> tuple[bool, Optional[str], Optional[int], str]:
            z = _evaluate_kt_cg_trial_7_zone_only_candidate(
                profile=profile,
                policy=policy,
                data_by_tf=data_eval,
                tick=tick,
                tier_state=tier_state,
            )
            if not z.get("passed") or z.get("side") is None or str(z.get("trigger_type") or "") != "zone_entry":
                return False, None, None, "zone_fallback_not_valid"
            z_side = str(z.get("side"))
            if z_side != side:
                return False, None, None, "zone_fallback_side_mismatch"
            return True, z_side, None, "zone_entry"

        # Apply tier minimums (exhaustion + RR)
        needed_min_tier = None
        if ex_min_tier is not None:
            needed_min_tier = int(ex_min_tier)
        if rr_min_tier_ema is not None:
            needed_min_tier = int(rr_min_tier_ema) if needed_min_tier is None else max(int(needed_min_tier), int(rr_min_tier_ema))

        if trigger_type == "tiered_pullback" and tier_period is not None and needed_min_tier is not None and int(tier_period) < int(needed_min_tier):
            ok_fb, fb_side, fb_tier, fb_trigger = try_zone_fallback()
            if ok_fb:
                trigger_type, tier_period = fb_trigger, fb_tier
            else:
                blocked_reasons[f"min_tier_block<{needed_min_tier}"] = blocked_reasons.get(f"min_tier_block<{needed_min_tier}", 0) + 1
                continue

        # Zone blocking rules
        if trigger_type == "zone_entry" and block_zone_by_ex:
            blocked_reasons[f"exhaustion_zone_block:{ex_zone}"] = blocked_reasons.get(f"exhaustion_zone_block:{ex_zone}", 0) + 1
            continue
        if trigger_type == "zone_entry" and bool(rr_response.get("block_zone_entry", False)):
            blocked_reasons[f"rr_zone_block:{rr_tier}"] = blocked_reasons.get(f"rr_zone_block:{rr_tier}", 0) + 1
            continue

        # Cooldown
        cooldown_min = float(getattr(policy, "cooldown_minutes", 0.0))
        if cooldown_min > 0 and last_trade_time is not None:
            elapsed = (ts - last_trade_time).total_seconds() / 60.0
            if elapsed < cooldown_min:
                blocked_reasons["cooldown"] = blocked_reasons.get("cooldown", 0) + 1
                continue

        # Apply caps
        max_side = max(cap_min, int(round(float(getattr(policy, "max_open_trades_per_side", 5)) * cap_multiplier)))
        max_zone = max(cap_min, int(round(float(getattr(policy, "max_zone_entry_open", 3)) * cap_multiplier)))
        max_tier = max(cap_min, int(round(float(getattr(policy, "max_tiered_pullback_open", 6)) * cap_multiplier)))

        buy_open = sum(1 for p in open_positions if p.side == "buy")
        sell_open = sum(1 for p in open_positions if p.side == "sell")
        zone_open = sum(1 for p in open_positions if p.entry_type == "zone_entry")
        tier_open = sum(1 for p in open_positions if p.entry_type == "tiered_pullback")

        side_open = buy_open if side == "buy" else sell_open
        if side_open >= max_side:
            blocked_reasons[f"side_cap_{side}"] = blocked_reasons.get(f"side_cap_{side}", 0) + 1
            continue

        if trigger_type == "zone_entry" and zone_open >= max_zone:
            blocked_reasons["zone_cap"] = blocked_reasons.get("zone_cap", 0) + 1
            continue

        if trigger_type == "tiered_pullback" and tier_open >= max_tier:
            ok_fb, fb_side, fb_tier, fb_trigger = try_zone_fallback()
            if ok_fb and zone_open < max_zone:
                trigger_type, tier_period = fb_trigger, fb_tier
            else:
                blocked_reasons["tier_cap"] = blocked_reasons.get("tier_cap", 0) + 1
                continue

        # Place trade
        trade_id_seq += 1
        entry_price = ask if side == "buy" else bid
        sl_pips = float(getattr(policy, "sl_pips", 20.0))
        tp_pips = float(getattr(policy, "tp_pips", 6.0))
        stop = entry_price - sl_pips * PIP_SIZE if side == "buy" else entry_price + sl_pips * PIP_SIZE
        target = entry_price + tp_pips * PIP_SIZE if side == "buy" else entry_price - tp_pips * PIP_SIZE
        lots = max(0.01, float(base_lot) * max(0.01, rr_lot_mult))
        rr_use_managed = _rr_tier_at_or_above(rr_tier, str(getattr(policy, "rr_use_managed_exit_at", "high")))

        open_positions.append(
            Position(
                trade_id=trade_id_seq,
                side=side,
                entry_type=trigger_type,
                tier_period=int(tier_period) if tier_period is not None else None,
                entry_time=ts,
                entry_price=entry_price,
                stop_price=stop,
                target_price=target,
                lots=lots,
                rr_score=float(rr_result.get("score")) if isinstance(rr_result, dict) and rr_result.get("score") is not None else None,
                rr_tier=rr_tier,
                rr_use_managed_exit=rr_use_managed,
                rr_lot_multiplier=float(rr_lot_mult),
                exhaustion_zone=ex_zone,
            )
        )

        if trigger_type == "tiered_pullback" and tier_period is not None:
            tier_state[int(tier_period)] = True

        last_trade_time = ts

        max_open_total = max(max_open_total, len(open_positions))
        max_open_buy = max(max_open_buy, sum(1 for p in open_positions if p.side == "buy"))
        max_open_sell = max(max_open_sell, sum(1 for p in open_positions if p.side == "sell"))

    # Close remaining at final price
    if len(m1):
        last = m1.iloc[-1]
        ts = pd.Timestamp(last["time"])
        mid = float(last["close"])
        bid = mid - half_spread
        ask = mid + half_spread
        for pos in list(open_positions):
            exit_px = bid if pos.side == "buy" else ask
            close_position(pos, ts, exit_px, "eod")

    if not closed_trades:
        return {
            "summary": {
                "trades": 0,
                "win_rate": None,
                "net_pips": 0.0,
                "net_usd": 0.0,
                "max_open_total": max_open_total,
                "max_open_buy": max_open_buy,
                "max_open_sell": max_open_sell,
            },
            "blocked_reasons": blocked_reasons,
            "trades": [],
        }

    tdf = pd.DataFrame([x.__dict__ for x in closed_trades])
    tdf["is_win"] = tdf["pips"] > 0

    net_pips = float(tdf["pips"].sum())
    net_usd = float(tdf["usd"].sum())
    win_rate = float(tdf["is_win"].mean())
    avg_win = float(tdf.loc[tdf["is_win"], "pips"].mean()) if (tdf["is_win"]).any() else 0.0
    avg_loss = float(tdf.loc[~tdf["is_win"], "pips"].mean()) if (~tdf["is_win"]).any() else 0.0

    gross_win = float(tdf.loc[tdf["pips"] > 0, "pips"].sum())
    gross_loss = abs(float(tdf.loc[tdf["pips"] < 0, "pips"].sum()))
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else None

    tdf = tdf.sort_values("exit_time").reset_index(drop=True)
    tdf["cum_usd"] = tdf["usd"].cumsum()
    tdf["cum_peak"] = tdf["cum_usd"].cummax()
    tdf["dd_usd"] = tdf["cum_peak"] - tdf["cum_usd"]
    max_dd_usd = float(tdf["dd_usd"].max())

    by_entry = (
        tdf.groupby("entry_type", dropna=False)
        .agg(trades=("trade_id", "size"), win_rate=("is_win", "mean"), net_pips=("pips", "sum"), net_usd=("usd", "sum"))
        .reset_index()
        .sort_values("trades", ascending=False)
    )
    by_rr = (
        tdf.groupby("rr_tier", dropna=False)
        .agg(trades=("trade_id", "size"), win_rate=("is_win", "mean"), net_pips=("pips", "sum"), net_usd=("usd", "sum"))
        .reset_index()
        .sort_values("trades", ascending=False)
    )
    by_ex = (
        tdf.groupby("exhaustion_zone", dropna=False)
        .agg(trades=("trade_id", "size"), win_rate=("is_win", "mean"), net_pips=("pips", "sum"), net_usd=("usd", "sum"))
        .reset_index()
        .sort_values("trades", ascending=False)
    )
    by_exit = (
        tdf.groupby("exit_reason", dropna=False)
        .agg(trades=("trade_id", "size"), avg_pips=("pips", "mean"), net_pips=("pips", "sum"), net_usd=("usd", "sum"))
        .reset_index()
        .sort_values("trades", ascending=False)
    )

    return {
        "summary": {
            "trades": int(len(tdf)),
            "wins": int(tdf["is_win"].sum()),
            "losses": int((~tdf["is_win"]).sum()),
            "win_rate": round(win_rate * 100.0, 3),
            "net_pips": round(net_pips, 3),
            "net_usd": round(net_usd, 3),
            "avg_win_pips": round(avg_win, 3),
            "avg_loss_pips": round(avg_loss, 3),
            "profit_factor": round(profit_factor, 4) if profit_factor is not None else None,
            "max_drawdown_usd": round(max_dd_usd, 3),
            "max_open_total": int(max_open_total),
            "max_open_buy": int(max_open_buy),
            "max_open_sell": int(max_open_sell),
        },
        "blocked_reasons": dict(sorted(blocked_reasons.items(), key=lambda kv: kv[1], reverse=True)),
        "by_entry_type": by_entry.to_dict("records"),
        "by_rr_tier": by_rr.to_dict("records"),
        "by_exhaustion_zone": by_ex.to_dict("records"),
        "by_exit_reason": by_exit.to_dict("records"),
        "trades_sample": tdf.tail(20).to_dict("records"),
    }


def main() -> int:
    args = parse_args()
    m1 = load_m1(args.inputs)

    report = {
        "config": {
            "inputs": args.inputs,
            "bars_m1": int(len(m1)),
            "start_utc": str(m1["time"].min()),
            "end_utc": str(m1["time"].max()),
            "spread_pips": float(args.spread_pips),
            "base_lot": float(args.base_lot),
            "tp_pips": float(args.tp_pips),
            "sl_pips": float(args.sl_pips),
            "m5_min_gap_pips": float(args.m5_min_gap_pips),
        },
        "results": backtest(
            m1=m1,
            spread_pips=float(args.spread_pips),
            base_lot=float(args.base_lot),
            tp_pips=float(args.tp_pips),
            sl_pips=float(args.sl_pips),
            m5_min_gap_pips=float(args.m5_min_gap_pips),
        ),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")

    s = report["results"]["summary"]
    print(
        f"Backtest complete | trades={s['trades']} win_rate={s['win_rate']}% "
        f"net_pips={s['net_pips']} net_usd={s['net_usd']} "
        f"maxDD_usd={s['max_drawdown_usd']}"
    )
    print(f"Wrote report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
