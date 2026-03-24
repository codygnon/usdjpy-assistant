#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.indicators import ema as ema_fn
from scripts.backtest_trial10 import (
    PIP_SIZE,
    Position,
    build_profile,
    choose_first_event,
    close_leg,
    load_m1,
    resample_ohlc,
)


@dataclass(frozen=True)
class Config:
    name: str
    escalation_enabled: bool


@dataclass
class SimMetrics:
    weighted_pips: float
    profit_usd: float
    exit_reason: str
    tp1_done: bool
    hold_minutes: float
    runner_bucket: str
    trail_level_used: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare current live Trial #10 trailing with and without escalation.")
    p.add_argument("--in", dest="input_csv", default="research_out/USDJPY_M1_OANDA_250k.csv")
    p.add_argument("--trades", default="research_out/trial10_250k_pause1_trades.csv")
    p.add_argument("--runner-labels", default="research_out/trial10_pause1_runner_elite_labeled.csv")
    p.add_argument("--out-json", default="research_out/trial10_live_trail_compare_250k.json")
    p.add_argument("--out-md", default="research_out/trial10_live_trail_compare_250k.md")
    p.add_argument("--spread-pips", type=float, default=2.0)
    p.add_argument("--max-hold-minutes", type=float, default=720.0)
    return p.parse_args()


def classify_trade(row: pd.Series) -> str:
    entry_type = str(row.get("entry_type") or "")
    tier = row.get("tier_period")
    if entry_type == "zone_entry":
        return "zone"
    if pd.isna(tier):
        return "tier_unknown"
    tier_int = int(float(tier))
    if tier_int in {17, 21}:
        return f"shallow_{tier_int}"
    if tier_int in {27, 33}:
        return f"standard_{tier_int}"
    return f"tier_{tier_int}"


def load_trade_set(trades_csv: str, runner_labels_csv: str) -> pd.DataFrame:
    trades = pd.read_csv(trades_csv)
    labels = pd.read_csv(runner_labels_csv, usecols=["trade_id", "strict_bucket"])
    labels = labels.rename(columns={"strict_bucket": "runner_bucket"})
    merged = trades.merge(labels, on="trade_id", how="left")
    merged["entry_time"] = pd.to_datetime(merged["entry_time"], utc=True)
    merged["signal_time"] = pd.to_datetime(merged["signal_time"], utc=True)
    merged["entry_class"] = merged.apply(classify_trade, axis=1)
    merged["runner_bucket"] = merged["runner_bucket"].fillna("floor").astype(str).str.lower()
    return merged.sort_values("entry_time").reset_index(drop=True)


def prepare_frames(m1: pd.DataFrame, *, m1_period: int, m5_period: int, m15_period: int, h1_period: int) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    m1f = m1.copy()
    m1f[f"ema_{m1_period}"] = ema_fn(m1f["close"].astype(float), m1_period)
    frames["m1"] = m1f

    m5 = resample_ohlc(m1, "5min")
    m5[f"ema_{m5_period}"] = ema_fn(m5["close"].astype(float), m5_period)
    frames["m5"] = m5

    m15 = resample_ohlc(m1, "15min")
    m15[f"ema_{m15_period}"] = ema_fn(m15["close"].astype(float), m15_period)
    frames["m15"] = m15

    h1 = resample_ohlc(m1, "1h")
    h1[f"ema_{h1_period}"] = ema_fn(h1["close"].astype(float), h1_period)
    frames["h1"] = h1
    return frames


def bucket_exit_profile(policy, runner_bucket: str) -> dict[str, float | str]:
    bucket = str(runner_bucket or "").lower()
    quick_buckets = {str(x).lower() for x in getattr(policy, "quick_exit_buckets", ("floor", "base"))}
    runner_buckets = {str(x).lower() for x in getattr(policy, "runner_exit_buckets", ("press", "elite"))}
    profile = {
        "tp1_pips": float(getattr(policy, "tp1_pips", 6.0)),
        "tp1_close_pct": float(getattr(policy, "tp1_close_pct", 70.0)),
        "be_plus_pips": float(getattr(policy, "be_spread_plus_pips", 0.5)),
        "trail_mode": "m5",
        "exit_profile": "standard",
    }
    if bucket in quick_buckets:
        profile.update(
            {
                "tp1_pips": float(getattr(policy, "quick_tp1_pips", 4.0)),
                "tp1_close_pct": float(getattr(policy, "quick_tp1_close_pct", 85.0)),
                "be_plus_pips": float(getattr(policy, "quick_be_spread_plus_pips", 0.3)),
                "trail_mode": "m1",
                "exit_profile": "quick",
            }
        )
    elif bucket in runner_buckets:
        profile.update(
            {
                "tp1_pips": float(getattr(policy, "runner_tp1_pips", 8.0)),
                "tp1_close_pct": float(getattr(policy, "runner_tp1_close_pct", 55.0)),
                "be_plus_pips": float(getattr(policy, "runner_be_spread_plus_pips", 0.5)),
                "trail_mode": "m5",
                "exit_profile": "runner",
            }
        )
    return profile


def favorable_profit_pips(side: str, entry_price: float, high_bid: float, low_ask: float) -> float:
    if side == "buy":
        return max(0.0, (high_bid - entry_price) / PIP_SIZE)
    return max(0.0, (entry_price - low_ask) / PIP_SIZE)


def simulate_trade(
    trade: pd.Series,
    *,
    config: Config,
    policy,
    frames: dict[str, pd.DataFrame],
    spread_pips: float,
    max_hold_minutes: float,
) -> Optional[SimMetrics]:
    entry_time = pd.Timestamp(trade["entry_time"])
    m1f = frames["m1"]
    m5 = frames["m5"]
    m15 = frames["m15"]
    h1 = frames["h1"]
    if entry_time not in m1f.index:
        return None
    j0 = int(m1f.index.get_loc(entry_time))

    runner_bucket = str(trade.get("runner_bucket") or "floor").lower()
    exit_profile = bucket_exit_profile(policy, runner_bucket)
    tp1_pips = float(exit_profile["tp1_pips"])
    tp1_close_pct = float(exit_profile["tp1_close_pct"])
    be_plus_pips = float(exit_profile["be_plus_pips"])
    base_trail_mode = str(exit_profile["trail_mode"])
    quick_buckets = {str(x).lower() for x in getattr(policy, "quick_exit_buckets", ("floor", "base"))}

    spread_price = float(spread_pips) * PIP_SIZE
    half_spread = spread_price / 2.0
    be_offset = spread_price + be_plus_pips * PIP_SIZE

    side = str(trade["side"])
    entry_price = float(trade["entry_price"])
    stop_price = float(trade["initial_stop_price"])
    tp1_price = entry_price + tp1_pips * PIP_SIZE if side == "buy" else entry_price - tp1_pips * PIP_SIZE

    pos = Position(
        trade_id=int(trade["trade_id"]),
        side=side,
        entry_type=str(trade["entry_type"]),
        tier_period=(None if pd.isna(trade["tier_period"]) else int(float(trade["tier_period"]))),
        signal_time=pd.Timestamp(trade["signal_time"]),
        entry_time=entry_time,
        entry_price=entry_price,
        initial_lots=float(trade["initial_lots"]),
        remaining_lots=float(trade["initial_lots"]),
        stop_price=stop_price,
        initial_stop_price=stop_price,
        tp1_price=tp1_price,
        tp1_done=False,
        tp1_close_pct=tp1_close_pct,
        be_stop_price=None,
        m5_bucket=str(trade.get("m5_bucket") or "normal"),
        m1_bucket=str(trade.get("m1_bucket") or "neutral"),
        conviction_multiplier=float(trade.get("conviction_multiplier") or 1.0),
        conviction_lots=float(trade.get("conviction_lots") or trade["initial_lots"]),
        atr_stop_pips=float(trade["atr_stop_pips"]),
        pullback_quality_label=str(trade.get("pullback_quality_label") or ""),
        pullback_quality_bar_count=int(float(trade.get("pullback_quality_bar_count") or 0)),
        pullback_quality_structure_ratio=float(trade.get("pullback_quality_structure_ratio") or 0.0),
        pullback_quality_dampener=float(trade.get("pullback_quality_dampener") or 1.0),
        regime_multiplier=float(trade.get("regime_multiplier") or 1.0),
        regime_label=str(trade.get("regime_label") or "baseline"),
    )

    m5_close_set = set(m5.index)
    m15_close_set = set(m15.index)
    h1_close_set = set(h1.index)
    peak_profit_pips = 0.0
    effective_trail_tf = base_trail_mode
    max_hold_delta = pd.Timedelta(minutes=float(max_hold_minutes))
    last_ts = entry_time

    m1_period = int(getattr(policy, "trail_ema_period", 21))
    m5_period = int(getattr(policy, "trail_m5_ema_period", 20))
    m15_period = int(getattr(policy, "trail_escalation_m15_ema_period", 21))
    h1_period = int(getattr(policy, "trail_escalation_h1_ema_period", 9))
    m15_buffer = float(getattr(policy, "trail_escalation_m15_buffer_pips", 1.5))
    h1_buffer = float(getattr(policy, "trail_escalation_h1_buffer_pips", 2.0))
    tier1 = float(getattr(policy, "trail_escalation_tier1_pips", 10.0))
    tier2 = float(getattr(policy, "trail_escalation_tier2_pips", 20.0))

    for j in range(j0, len(m1f)):
        ts = m1f.index[j]
        if ts - entry_time > max_hold_delta:
            break
        row = m1f.iloc[j]
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
        ref_price = close_mid
        closed = False
        last_ts = ts

        peak_profit_pips = max(peak_profit_pips, favorable_profit_pips(side, entry_price, high_bid, low_ask))

        if not pos.tp1_done:
            if side == "buy":
                stop_hit = low_bid <= pos.stop_price
                tp1_hit = high_bid >= pos.tp1_price
            else:
                stop_hit = high_ask >= pos.stop_price
                tp1_hit = low_ask <= pos.tp1_price

            if stop_hit and tp1_hit:
                first = choose_first_event(side, open_mid, pos.stop_price, pos.tp1_price, spread_price)
                if first == "stop":
                    tp1_hit = False
                else:
                    stop_hit = False

            if stop_hit:
                close_leg(pos, pos.remaining_lots, pos.stop_price, "initial_stop", ref_price)
                closed = True
            elif tp1_hit:
                tp1_lots = round(pos.initial_lots * (tp1_close_pct / 100.0), 2)
                tp1_lots = max(0.01, min(tp1_lots, pos.remaining_lots))
                close_leg(pos, tp1_lots, pos.tp1_price, "tp1_partial", ref_price)
                pos.tp1_done = True
                pos.be_stop_price = pos.entry_price + be_offset if side == "buy" else pos.entry_price - be_offset
                pos.stop_price = float(pos.be_stop_price)
                if pos.remaining_lots <= 0.0:
                    closed = True
                else:
                    if side == "buy" and low_bid <= pos.stop_price:
                        close_leg(pos, pos.remaining_lots, pos.stop_price, "tp1_then_be_same_bar", ref_price)
                        closed = True
                    elif side == "sell" and high_ask >= pos.stop_price:
                        close_leg(pos, pos.remaining_lots, pos.stop_price, "tp1_then_be_same_bar", ref_price)
                        closed = True

        if not closed and pos.tp1_done:
            if side == "buy" and low_bid <= pos.stop_price:
                close_leg(pos, pos.remaining_lots, pos.stop_price, "runner_stop", ref_price)
                closed = True
            elif side == "sell" and high_ask >= pos.stop_price:
                close_leg(pos, pos.remaining_lots, pos.stop_price, "runner_stop", ref_price)
                closed = True

        if not closed and pos.tp1_done:
            effective_trail_tf = base_trail_mode
            esc_allowed = (
                config.escalation_enabled
                and bool(getattr(policy, "trail_escalation_enabled", False))
                and base_trail_mode in {"m1", "m5"}
                and runner_bucket not in quick_buckets
            )
            if esc_allowed:
                if base_trail_mode == "m1":
                    if peak_profit_pips >= tier2:
                        effective_trail_tf = "m15"
                    elif peak_profit_pips >= tier1:
                        effective_trail_tf = "m5"
                elif base_trail_mode == "m5":
                    if peak_profit_pips >= tier2:
                        effective_trail_tf = "h1"
                    elif peak_profit_pips >= tier1:
                        effective_trail_tf = "m15"

            if effective_trail_tf == "m1":
                ema_val = float(m1f.iloc[j][f"ema_{m1_period}"])
                prev_be = float(pos.be_stop_price) if pos.be_stop_price is not None else None
                new_sl = ema_val - (1.0 * PIP_SIZE) if side == "buy" else ema_val + (1.0 * PIP_SIZE)
                if prev_be is not None:
                    new_sl = max(new_sl, prev_be) if side == "buy" else min(new_sl, prev_be)
                pos.stop_price = new_sl
                if side == "buy" and close_mid < ema_val:
                    close_leg(pos, pos.remaining_lots, close_bid, "m1_trail_close", ref_price)
                    closed = True
                elif side == "sell" and close_mid > ema_val:
                    close_leg(pos, pos.remaining_lots, close_ask, "m1_trail_close", ref_price)
                    closed = True

            elif effective_trail_tf == "m5" and ts in m5_close_set:
                ema_val = float(m5.loc[ts, f"ema_{m5_period}"])
                tf_close = float(m5.loc[ts, "close"])
                prev_be = float(pos.be_stop_price) if pos.be_stop_price is not None else None
                new_sl = ema_val - (1.0 * PIP_SIZE) if side == "buy" else ema_val + (1.0 * PIP_SIZE)
                if prev_be is not None:
                    new_sl = max(new_sl, prev_be) if side == "buy" else min(new_sl, prev_be)
                pos.stop_price = new_sl
                if side == "buy" and tf_close < ema_val:
                    close_leg(pos, pos.remaining_lots, close_bid, "m5_trail_close", ref_price)
                    closed = True
                elif side == "sell" and tf_close > ema_val:
                    close_leg(pos, pos.remaining_lots, close_ask, "m5_trail_close", ref_price)
                    closed = True

            elif effective_trail_tf == "m15" and ts in m15_close_set:
                ema_val = float(m15.loc[ts, f"ema_{m15_period}"])
                tf_close = float(m15.loc[ts, "close"])
                prev_be = float(pos.be_stop_price) if pos.be_stop_price is not None else None
                new_sl = ema_val - (m15_buffer * PIP_SIZE) if side == "buy" else ema_val + (m15_buffer * PIP_SIZE)
                if prev_be is not None:
                    new_sl = max(new_sl, prev_be) if side == "buy" else min(new_sl, prev_be)
                pos.stop_price = new_sl
                if side == "buy" and tf_close < ema_val:
                    close_leg(pos, pos.remaining_lots, close_bid, "m15_trail_close", ref_price)
                    closed = True
                elif side == "sell" and tf_close > ema_val:
                    close_leg(pos, pos.remaining_lots, close_ask, "m15_trail_close", ref_price)
                    closed = True

            elif effective_trail_tf == "h1" and ts in h1_close_set:
                ema_val = float(h1.loc[ts, f"ema_{h1_period}"])
                tf_close = float(h1.loc[ts, "close"])
                prev_be = float(pos.be_stop_price) if pos.be_stop_price is not None else None
                new_sl = ema_val - (h1_buffer * PIP_SIZE) if side == "buy" else ema_val + (h1_buffer * PIP_SIZE)
                if prev_be is not None:
                    new_sl = max(new_sl, prev_be) if side == "buy" else min(new_sl, prev_be)
                pos.stop_price = new_sl
                if side == "buy" and tf_close < ema_val:
                    close_leg(pos, pos.remaining_lots, close_bid, "h1_trail_close", ref_price)
                    closed = True
                elif side == "sell" and tf_close > ema_val:
                    close_leg(pos, pos.remaining_lots, close_ask, "h1_trail_close", ref_price)
                    closed = True

        if closed or pos.remaining_lots <= 0.0:
            weighted_pips = pos.weighted_pips_x_lots / pos.initial_lots if pos.initial_lots > 0 else 0.0
            return SimMetrics(
                weighted_pips=float(weighted_pips),
                profit_usd=float(pos.weighted_usd),
                exit_reason=str(pos.exit_reason or ""),
                tp1_done=bool(pos.tp1_done),
                hold_minutes=float((ts - entry_time).total_seconds() / 60.0),
                runner_bucket=runner_bucket,
                trail_level_used=effective_trail_tf,
            )

    final_close = float(m1f.iloc[min(len(m1f) - 1, j0)]["close"])
    final_exit = final_close - half_spread if side == "buy" else final_close + half_spread
    close_leg(pos, pos.remaining_lots, final_exit, "end_of_horizon", final_close)
    weighted_pips = pos.weighted_pips_x_lots / pos.initial_lots if pos.initial_lots > 0 else 0.0
    return SimMetrics(
        weighted_pips=float(weighted_pips),
        profit_usd=float(pos.weighted_usd),
        exit_reason=str(pos.exit_reason or "end_of_horizon"),
        tp1_done=bool(pos.tp1_done),
        hold_minutes=float((last_ts - entry_time).total_seconds() / 60.0),
        runner_bucket=runner_bucket,
        trail_level_used=effective_trail_tf,
    )


def summarize(df: pd.DataFrame) -> dict[str, object]:
    return {
        "trades": int(len(df)),
        "avg_pips": round(float(df["weighted_pips"].mean()), 3),
        "total_pips": round(float(df["weighted_pips"].sum()), 3),
        "avg_usd": round(float(df["profit_usd"].mean()), 4),
        "total_usd": round(float(df["profit_usd"].sum()), 4),
        "win_rate_pct": round(float((df["weighted_pips"] > 0).mean() * 100.0), 2),
        "tp1_rate_pct": round(float(df["tp1_done"].mean() * 100.0), 2),
        "avg_hold_minutes": round(float(df["hold_minutes"].mean()), 2),
        "by_exit_reason": df["exit_reason"].value_counts().to_dict(),
        "by_runner_bucket": df["runner_bucket"].value_counts().to_dict(),
        "by_trail_level": df["trail_level_used"].value_counts().to_dict(),
    }


def main() -> None:
    args = parse_args()
    trades = load_trade_set(args.trades, args.runner_labels)
    m1 = load_m1(args.input_csv)
    _, policy = build_profile(type("Args", (), {"tier_periods": "", "tp1_close_pct": None})())
    frames = prepare_frames(
        m1,
        m1_period=int(getattr(policy, "trail_ema_period", 21)),
        m5_period=int(getattr(policy, "trail_m5_ema_period", 20)),
        m15_period=int(getattr(policy, "trail_escalation_m15_ema_period", 21)),
        h1_period=int(getattr(policy, "trail_escalation_h1_ema_period", 9)),
    )

    configs = [
        Config(name="live_no_escalation", escalation_enabled=False),
        Config(name="live_with_escalation", escalation_enabled=True),
    ]

    rows: list[dict] = []
    for config in configs:
        for _, trade in trades.iterrows():
            sim = simulate_trade(
                trade,
                config=config,
                policy=policy,
                frames=frames,
                spread_pips=float(args.spread_pips),
                max_hold_minutes=float(args.max_hold_minutes),
            )
            if sim is None:
                continue
            rows.append(
                {
                    "config": config.name,
                    "entry_class": str(trade["entry_class"]),
                    "weighted_pips": sim.weighted_pips,
                    "profit_usd": sim.profit_usd,
                    "exit_reason": sim.exit_reason,
                    "tp1_done": sim.tp1_done,
                    "hold_minutes": sim.hold_minutes,
                    "runner_bucket": sim.runner_bucket,
                    "trail_level_used": sim.trail_level_used,
                }
            )

    df = pd.DataFrame(rows)
    overall = {name: summarize(g) for name, g in df.groupby("config")}
    by_entry_class = {
        cls: {name: summarize(g2) for name, g2 in g.groupby("config")}
        for cls, g in df.groupby("entry_class")
    }

    payload = {
        "input_csv": args.input_csv,
        "trades_csv": args.trades,
        "runner_labels_csv": args.runner_labels,
        "spread_pips": float(args.spread_pips),
        "max_hold_minutes": float(args.max_hold_minutes),
        "overall": overall,
        "by_entry_class": by_entry_class,
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    base = overall["live_no_escalation"]
    esc = overall["live_with_escalation"]
    md = [
        "# Trial 10 Live Trail Comparison",
        "",
        "- Baseline: exact current bucketed exits, no trail escalation.",
        "- Variant: exact current bucketed exits, trail escalation enabled.",
        "",
        "## Overall",
        "",
        f"- Baseline: {base['trades']} trades, avg {base['avg_pips']} pips, total ${base['total_usd']}, TP1 {base['tp1_rate_pct']}%, hold {base['avg_hold_minutes']} min",
        f"- Escalated: {esc['trades']} trades, avg {esc['avg_pips']} pips, total ${esc['total_usd']}, TP1 {esc['tp1_rate_pct']}%, hold {esc['avg_hold_minutes']} min",
        "",
    ]
    Path(args.out_md).write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"out_json": args.out_json, "out_md": args.out_md, "rows": int(len(df))}, indent=2))


if __name__ == "__main__":
    main()
