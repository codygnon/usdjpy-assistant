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
    choose_first_event,
    close_leg,
    load_m1,
    resample_ohlc,
)


@dataclass(frozen=True)
class TrailConfig:
    name: str
    escalated: bool
    level1_pips: float = 10.0
    level2_pips: float = 20.0
    m1_period: int = 21
    m5_period: int = 21
    m15_period: int = 21


@dataclass
class SimMetrics:
    weighted_pips: float
    profit_usd: float
    exit_reason: str
    tp1_done: bool
    hold_minutes: float
    peak_favorable_pips: float
    trail_mode_used: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay Trial #10 entries with escalated runner trailing.")
    p.add_argument("--in", dest="input_csv", default="research_out/USDJPY_M1_OANDA_250k.csv")
    p.add_argument("--trades", default="research_out/trial10_250k_pause1_trades.csv")
    p.add_argument("--out-json", default="research_out/trial10_trail_escalation_250k.json")
    p.add_argument("--out-md", default="research_out/trial10_trail_escalation_250k.md")
    p.add_argument("--spread-pips", type=float, default=2.0)
    p.add_argument("--tp1-pips", type=float, default=6.0)
    p.add_argument("--tp1-close-pct", type=float, default=70.0)
    p.add_argument("--be-spread-plus-pips", type=float, default=0.5)
    p.add_argument("--baseline-m5-period", type=int, default=20)
    p.add_argument("--level1-pips", type=float, default=10.0)
    p.add_argument("--level2-pips", type=float, default=20.0)
    p.add_argument("--min-trades-per-class", type=int, default=100)
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


def prepare_frames(m1: pd.DataFrame, baseline_m5_period: int) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    m1f = m1.copy()
    m1f["ema_21"] = ema_fn(m1f["close"].astype(float), 21)
    out["m1"] = m1f

    m5 = resample_ohlc(m1, "5min")
    m5["ema_baseline"] = ema_fn(m5["close"].astype(float), baseline_m5_period)
    m5["ema_21"] = ema_fn(m5["close"].astype(float), 21)
    out["m5"] = m5

    m15 = resample_ohlc(m1, "15min")
    m15["ema_21"] = ema_fn(m15["close"].astype(float), 21)
    out["m15"] = m15
    return out


def load_trades(path: str) -> pd.DataFrame:
    trades = pd.read_csv(path)
    trades["signal_time"] = pd.to_datetime(trades["signal_time"], utc=True)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades["entry_class"] = trades.apply(classify_trade, axis=1)
    return trades.sort_values("entry_time").reset_index(drop=True)


def favorable_pips(side: str, entry_price: float, high_bid: float, low_ask: float) -> float:
    if side == "buy":
        return max(0.0, (high_bid - entry_price) / PIP_SIZE)
    return max(0.0, (entry_price - low_ask) / PIP_SIZE)


def simulate_trade(
    trade: pd.Series,
    *,
    config: TrailConfig,
    m1f: pd.DataFrame,
    m5: pd.DataFrame,
    m15: pd.DataFrame,
    spread_pips: float,
    tp1_pips: float,
    tp1_close_pct: float,
    be_spread_plus_pips: float,
) -> Optional[SimMetrics]:
    entry_time = pd.Timestamp(trade["entry_time"])
    if entry_time not in m1f.index:
        return None
    j0 = int(m1f.index.get_loc(entry_time))
    spread_price = float(spread_pips) * PIP_SIZE
    half_spread = spread_price / 2.0
    be_offset = spread_price + float(be_spread_plus_pips) * PIP_SIZE

    side = str(trade["side"])
    entry_price = float(trade["entry_price"])
    stop_pips = float(trade["atr_stop_pips"])
    stop_price = entry_price - stop_pips * PIP_SIZE if side == "buy" else entry_price + stop_pips * PIP_SIZE
    tp1_price = entry_price + float(tp1_pips) * PIP_SIZE if side == "buy" else entry_price - float(tp1_pips) * PIP_SIZE
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
        tp1_close_pct=float(tp1_close_pct),
        be_stop_price=None,
        m5_bucket=str(trade.get("m5_bucket") or "normal"),
        m1_bucket=str(trade.get("m1_bucket") or "neutral"),
        conviction_multiplier=float(trade.get("conviction_multiplier") or 1.0),
        conviction_lots=float(trade.get("conviction_lots") or trade["initial_lots"]),
        atr_stop_pips=stop_pips,
        pullback_quality_label=str(trade.get("pullback_quality_label") or ""),
        pullback_quality_bar_count=int(float(trade.get("pullback_quality_bar_count") or 0)),
        pullback_quality_structure_ratio=float(trade.get("pullback_quality_structure_ratio") or 0.0),
        pullback_quality_dampener=float(trade.get("pullback_quality_dampener") or 1.0),
        regime_multiplier=float(trade.get("regime_multiplier") or 1.0),
        regime_label=str(trade.get("regime_label") or "baseline"),
    )

    m5_close_set = set(m5.index)
    m15_close_set = set(m15.index)
    peak_favorable = 0.0
    trail_mode_used = "baseline_m5"
    last_ts = entry_time

    for j in range(j0, len(m1f)):
        ts = m1f.index[j]
        bar = m1f.iloc[j]
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
        ref_price = close_mid
        closed = False
        last_ts = ts

        peak_favorable = max(peak_favorable, favorable_pips(side, entry_price, high_bid, low_ask))

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

        if not closed and pos.tp1_done:
            if not config.escalated:
                if ts in m5_close_set:
                    trail_mode_used = "m5"
                    m5_close = float(m5.loc[ts, "close"])
                    trail_ema = float(m5.loc[ts, "ema_baseline"])
                    if side == "buy" and m5_close < trail_ema:
                        close_leg(pos, pos.remaining_lots, close_bid, "m5_trail_close", ref_price)
                        closed = True
                    elif side == "sell" and m5_close > trail_ema:
                        close_leg(pos, pos.remaining_lots, close_ask, "m5_trail_close", ref_price)
                        closed = True
            else:
                if peak_favorable >= config.level2_pips:
                    trail_mode_used = "m15"
                    if ts in m15_close_set:
                        tf_close = float(m15.loc[ts, "close"])
                        trail_ema = float(m15.loc[ts, f"ema_{config.m15_period}"])
                        if side == "buy" and tf_close < trail_ema:
                            close_leg(pos, pos.remaining_lots, close_bid, "m15_trail_close", ref_price)
                            closed = True
                        elif side == "sell" and tf_close > trail_ema:
                            close_leg(pos, pos.remaining_lots, close_ask, "m15_trail_close", ref_price)
                            closed = True
                elif peak_favorable >= config.level1_pips:
                    trail_mode_used = "m5"
                    if ts in m5_close_set:
                        tf_close = float(m5.loc[ts, "close"])
                        trail_ema = float(m5.loc[ts, f"ema_{config.m5_period}"])
                        if side == "buy" and tf_close < trail_ema:
                            close_leg(pos, pos.remaining_lots, close_bid, "m5_trail_close", ref_price)
                            closed = True
                        elif side == "sell" and tf_close > trail_ema:
                            close_leg(pos, pos.remaining_lots, close_ask, "m5_trail_close", ref_price)
                            closed = True
                else:
                    trail_mode_used = "m1"
                    tf_close = close_mid
                    trail_ema = float(m1f.iloc[j][f"ema_{config.m1_period}"])
                    if side == "buy" and tf_close < trail_ema:
                        close_leg(pos, pos.remaining_lots, close_bid, "m1_trail_close", ref_price)
                        closed = True
                    elif side == "sell" and tf_close > trail_ema:
                        close_leg(pos, pos.remaining_lots, close_ask, "m1_trail_close", ref_price)
                        closed = True

        if closed or pos.remaining_lots <= 0.0:
            weighted_pips = pos.weighted_pips_x_lots / pos.initial_lots if pos.initial_lots > 0 else 0.0
            return SimMetrics(
                weighted_pips=float(weighted_pips),
                profit_usd=float(pos.weighted_usd),
                exit_reason=str(pos.exit_reason),
                tp1_done=bool(pos.tp1_done),
                hold_minutes=float((ts - entry_time).total_seconds() / 60.0),
                peak_favorable_pips=float(peak_favorable),
                trail_mode_used=trail_mode_used,
            )

    final_close = float(m1f.iloc[-1]["close"])
    final_exit = final_close - half_spread if side == "buy" else final_close + half_spread
    close_leg(pos, pos.remaining_lots, final_exit, "end_of_data", final_close)
    weighted_pips = pos.weighted_pips_x_lots / pos.initial_lots if pos.initial_lots > 0 else 0.0
    return SimMetrics(
        weighted_pips=float(weighted_pips),
        profit_usd=float(pos.weighted_usd),
        exit_reason=str(pos.exit_reason or "end_of_data"),
        tp1_done=bool(pos.tp1_done),
        hold_minutes=float((last_ts - entry_time).total_seconds() / 60.0),
        peak_favorable_pips=float(peak_favorable),
        trail_mode_used=trail_mode_used,
    )


def summarize(df: pd.DataFrame) -> dict[str, float | int | dict]:
    return {
        "trades": int(len(df)),
        "avg_pips": round(float(df["weighted_pips"].mean()), 3),
        "total_pips": round(float(df["weighted_pips"].sum()), 3),
        "avg_usd": round(float(df["profit_usd"].mean()), 4),
        "total_usd": round(float(df["profit_usd"].sum()), 4),
        "win_rate_pct": round(float((df["weighted_pips"] > 0).mean() * 100.0), 2),
        "tp1_rate_pct": round(float(df["tp1_done"].mean() * 100.0), 2),
        "avg_hold_minutes": round(float(df["hold_minutes"].mean()), 2),
        "avg_peak_favorable_pips": round(float(df["peak_favorable_pips"].mean()), 2),
        "by_exit_reason": df["exit_reason"].value_counts().to_dict(),
        "by_trail_mode": df["trail_mode_used"].value_counts().to_dict(),
    }


def main() -> None:
    args = parse_args()
    m1 = load_m1(args.input_csv)
    frames = prepare_frames(m1, int(args.baseline_m5_period))
    trades = load_trades(args.trades)

    configs = [
        TrailConfig(name="baseline_m5_ema20", escalated=False),
        TrailConfig(
            name="escalated_m1_21_to_m5_21_to_m15_21",
            escalated=True,
            level1_pips=float(args.level1_pips),
            level2_pips=float(args.level2_pips),
            m1_period=21,
            m5_period=21,
            m15_period=21,
        ),
    ]

    rows: list[dict] = []
    for config in configs:
        for _, trade in trades.iterrows():
            sim = simulate_trade(
                trade,
                config=config,
                m1f=frames["m1"],
                m5=frames["m5"],
                m15=frames["m15"],
                spread_pips=float(args.spread_pips),
                tp1_pips=float(args.tp1_pips),
                tp1_close_pct=float(args.tp1_close_pct),
                be_spread_plus_pips=float(args.be_spread_plus_pips),
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
                    "peak_favorable_pips": sim.peak_favorable_pips,
                    "trail_mode_used": sim.trail_mode_used,
                }
            )

    df = pd.DataFrame(rows)
    overall = {name: summarize(g) for name, g in df.groupby("config")}
    by_class = {
        cls: {name: summarize(g2) for name, g2 in g.groupby("config")}
        for cls, g in df.groupby("entry_class")
    }

    payload = {
        "input_csv": args.input_csv,
        "trades_csv": args.trades,
        "spread_pips": float(args.spread_pips),
        "tp1_pips": float(args.tp1_pips),
        "tp1_close_pct": float(args.tp1_close_pct),
        "be_spread_plus_pips": float(args.be_spread_plus_pips),
        "level1_pips": float(args.level1_pips),
        "level2_pips": float(args.level2_pips),
        "overall": overall,
        "by_entry_class": by_class,
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    base = overall["baseline_m5_ema20"]
    esc = overall["escalated_m1_21_to_m5_21_to_m15_21"]
    md = [
        "# Trial 10 Trailing Escalation Replay",
        "",
        "- Baseline: post-TP1 runner trails on `M5 EMA20`.",
        f"- Escalation: `<{args.level1_pips:.0f}p` use `M1 EMA21`, `{args.level1_pips:.0f}-{args.level2_pips:.0f}p` use `M5 EMA21`, `>{args.level2_pips:.0f}p` use `M15 EMA21`.",
        "",
        "## Overall",
        "",
        f"- Baseline: {base['trades']} trades, avg {base['avg_pips']} pips, total ${base['total_usd']}, TP1 {base['tp1_rate_pct']}%, avg hold {base['avg_hold_minutes']} min",
        f"- Escalated: {esc['trades']} trades, avg {esc['avg_pips']} pips, total ${esc['total_usd']}, TP1 {esc['tp1_rate_pct']}%, avg hold {esc['avg_hold_minutes']} min",
        "",
    ]
    Path(args.out_md).write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"out_json": args.out_json, "out_md": args.out_md, "rows": int(len(df))}, indent=2))


if __name__ == "__main__":
    main()
