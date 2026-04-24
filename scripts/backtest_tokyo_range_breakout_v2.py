#!/usr/bin/env python3
"""
Bar-by-bar M1 backtest of the Tokyo Range Breakout strategy for USDJPY.

Realistic parameters calibrated to USDJPY volatility:
  - Tokyo range window: configurable (default 00:00-06:00 UTC)
  - Range width filter: 15-50 pips (USDJPY reality)
  - Breakout window: 06:00-09:00 UTC (pre-London + London open)
  - Entry on M1 close beyond range boundary
  - SL at opposite side of range (capped)
  - TP at 1:1 or configurable R:R
  - True bar-by-bar forward walk — no lookahead
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "research_out"

PIP_SIZE = 0.01


@dataclass
class TradeRecord:
    entry_time: str
    exit_time: str
    direction: str  # "buy" or "sell"
    entry_price: float
    exit_price: float
    sl_price: float
    tp_price: float
    pips: float
    outcome: str  # "tp", "sl", "timeout"
    tokyo_high: float
    tokyo_low: float
    range_width_pips: float
    duration_bars: int


@dataclass
class Config:
    tokyo_start_hour: int = 0
    tokyo_end_hour: int = 6
    breakout_start_hour: int = 6
    breakout_end_hour: int = 9
    min_range_pips: float = 15.0
    max_range_pips: float = 50.0
    breakout_pips_beyond: float = 1.0
    sl_buffer_pips: float = 2.0
    max_sl_pips: float = 20.0
    rr_ratio: float = 1.5
    fixed_tp_pips: float = 0.0
    max_hold_bars: int = 120
    spread_pips: float = 1.2
    require_ema_alignment: bool = True
    ema_fast: int = 5
    ema_slow: int = 9
    one_trade_per_range: bool = True
    require_body_ratio: float = 0.0
    min_breakout_bar_pips: float = 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tokyo Range Breakout v2 — bar-by-bar M1 backtest")
    p.add_argument("--dataset", default=str(OUT_DIR / "USDJPY_M1_OANDA_250k.csv"))
    p.add_argument("--tokyo-start", type=int, default=0, help="Tokyo range start hour UTC")
    p.add_argument("--tokyo-end", type=int, default=6, help="Tokyo range end hour UTC")
    p.add_argument("--breakout-start", type=int, default=6, help="Breakout window start hour UTC")
    p.add_argument("--breakout-end", type=int, default=9, help="Breakout window end hour UTC")
    p.add_argument("--min-range", type=float, default=15.0, help="Min Tokyo range width pips")
    p.add_argument("--max-range", type=float, default=50.0, help="Max Tokyo range width pips")
    p.add_argument("--breakout-beyond", type=float, default=1.0, help="Pips beyond range for entry")
    p.add_argument("--sl-buffer", type=float, default=2.0, help="SL buffer pips beyond opposite range boundary")
    p.add_argument("--max-sl", type=float, default=20.0, help="Max SL pips (cap)")
    p.add_argument("--rr", type=float, default=1.5, help="Risk:reward ratio for TP (0 = use fixed-tp)")
    p.add_argument("--fixed-tp", type=float, default=0.0, help="Fixed TP pips (overrides RR if > 0)")
    p.add_argument("--max-hold", type=int, default=120, help="Max hold bars before timeout")
    p.add_argument("--spread", type=float, default=1.2, help="Assumed spread pips")
    p.add_argument("--no-ema", action="store_true", help="Disable EMA alignment filter")
    p.add_argument("--body-ratio", type=float, default=0.0, help="Min body/range ratio on breakout bar")
    p.add_argument("--min-bar-size", type=float, default=0.0, help="Min breakout bar size in pips")
    p.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    return p.parse_args()


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    df["hour"] = df["time"].dt.hour
    df["ema_fast"] = df["close"].ewm(span=5, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=9, adjust=False).mean()
    return df


def compute_tokyo_ranges(df: pd.DataFrame, cfg: Config) -> dict:
    """For each trading day, compute Tokyo session high/low."""
    ranges = {}
    dates = df["time"].dt.date.unique()

    for date in dates:
        day_start = pd.Timestamp(date, tz="UTC")
        tokyo_start = day_start + pd.Timedelta(hours=cfg.tokyo_start_hour)
        tokyo_end = day_start + pd.Timedelta(hours=cfg.tokyo_end_hour)

        mask = (df["time"] >= tokyo_start) & (df["time"] < tokyo_end)
        tokyo_bars = df.loc[mask]
        if len(tokyo_bars) < 10:
            continue

        h = tokyo_bars["high"].max()
        l = tokyo_bars["low"].min()
        width = (h - l) / PIP_SIZE

        if cfg.min_range_pips <= width <= cfg.max_range_pips:
            ranges[date] = {"high": h, "low": l, "width_pips": width}

    return ranges


def run_backtest(df: pd.DataFrame, cfg: Config) -> list[TradeRecord]:
    tokyo_ranges = compute_tokyo_ranges(df, cfg)
    trades: list[TradeRecord] = []
    in_trade = False
    trade_entry_date = None

    for i in range(len(df)):
        row = df.iloc[i]
        t = row["time"]
        date = t.date()
        hour = row["hour"]

        # Check exit conditions for open trade
        if in_trade:
            bars_held += 1
            if trade_dir == "buy":
                if row["low"] <= trade_sl:
                    trades.append(TradeRecord(
                        entry_time=str(trade_entry_time),
                        exit_time=str(t),
                        direction=trade_dir,
                        entry_price=trade_entry,
                        exit_price=trade_sl,
                        sl_price=trade_sl,
                        tp_price=trade_tp,
                        pips=round((trade_sl - trade_entry) / PIP_SIZE - cfg.spread_pips, 2),
                        outcome="sl",
                        tokyo_high=trade_tokyo_high,
                        tokyo_low=trade_tokyo_low,
                        range_width_pips=trade_range_width,
                        duration_bars=bars_held,
                    ))
                    in_trade = False
                    continue
                if row["high"] >= trade_tp:
                    trades.append(TradeRecord(
                        entry_time=str(trade_entry_time),
                        exit_time=str(t),
                        direction=trade_dir,
                        entry_price=trade_entry,
                        exit_price=trade_tp,
                        sl_price=trade_sl,
                        tp_price=trade_tp,
                        pips=round((trade_tp - trade_entry) / PIP_SIZE - cfg.spread_pips, 2),
                        outcome="tp",
                        tokyo_high=trade_tokyo_high,
                        tokyo_low=trade_tokyo_low,
                        range_width_pips=trade_range_width,
                        duration_bars=bars_held,
                    ))
                    in_trade = False
                    continue
            else:  # sell
                if row["high"] >= trade_sl:
                    trades.append(TradeRecord(
                        entry_time=str(trade_entry_time),
                        exit_time=str(t),
                        direction=trade_dir,
                        entry_price=trade_entry,
                        exit_price=trade_sl,
                        sl_price=trade_sl,
                        tp_price=trade_tp,
                        pips=round((trade_entry - trade_sl) / PIP_SIZE - cfg.spread_pips, 2),
                        outcome="sl",
                        tokyo_high=trade_tokyo_high,
                        tokyo_low=trade_tokyo_low,
                        range_width_pips=trade_range_width,
                        duration_bars=bars_held,
                    ))
                    in_trade = False
                    continue
                if row["low"] <= trade_tp:
                    trades.append(TradeRecord(
                        entry_time=str(trade_entry_time),
                        exit_time=str(t),
                        direction=trade_dir,
                        entry_price=trade_entry,
                        exit_price=trade_tp,
                        sl_price=trade_sl,
                        tp_price=trade_tp,
                        pips=round((trade_entry - trade_tp) / PIP_SIZE - cfg.spread_pips, 2),
                        outcome="tp",
                        tokyo_high=trade_tokyo_high,
                        tokyo_low=trade_tokyo_low,
                        range_width_pips=trade_range_width,
                        duration_bars=bars_held,
                    ))
                    in_trade = False
                    continue

            if bars_held >= cfg.max_hold_bars:
                exit_price = row["close"]
                if trade_dir == "buy":
                    pips = (exit_price - trade_entry) / PIP_SIZE - cfg.spread_pips
                else:
                    pips = (trade_entry - exit_price) / PIP_SIZE - cfg.spread_pips
                trades.append(TradeRecord(
                    entry_time=str(trade_entry_time),
                    exit_time=str(t),
                    direction=trade_dir,
                    entry_price=trade_entry,
                    exit_price=exit_price,
                    sl_price=trade_sl,
                    tp_price=trade_tp,
                    pips=round(pips, 2),
                    outcome="timeout",
                    tokyo_high=trade_tokyo_high,
                    tokyo_low=trade_tokyo_low,
                    range_width_pips=trade_range_width,
                    duration_bars=bars_held,
                ))
                in_trade = False
                continue
            continue

        # Only look for entries during breakout window
        if not (cfg.breakout_start_hour <= hour < cfg.breakout_end_hour):
            continue

        # Check if we have a valid Tokyo range for today
        if date not in tokyo_ranges:
            continue

        # One trade per range
        if cfg.one_trade_per_range and trade_entry_date == date:
            continue

        rng = tokyo_ranges[date]
        tokyo_high = rng["high"]
        tokyo_low = rng["low"]
        range_width = rng["width_pips"]

        close = row["close"]
        open_ = row["open"]
        bar_body = abs(close - open_)
        bar_range = row["high"] - row["low"]

        # Breakout body ratio filter
        if cfg.require_body_ratio > 0 and bar_range > 0:
            if bar_body / bar_range < cfg.require_body_ratio:
                continue

        # Min bar size filter
        if cfg.min_breakout_bar_pips > 0:
            if bar_range / PIP_SIZE < cfg.min_breakout_bar_pips:
                continue

        # Check for upside breakout
        breakout_above = close > tokyo_high + cfg.breakout_pips_beyond * PIP_SIZE
        breakout_below = close < tokyo_low - cfg.breakout_pips_beyond * PIP_SIZE

        if not breakout_above and not breakout_below:
            continue

        # EMA alignment
        if cfg.require_ema_alignment:
            ema_bull = row["ema_fast"] > row["ema_slow"]
            if breakout_above and not ema_bull:
                continue
            if breakout_below and ema_bull:
                continue

        # Determine direction and levels
        if breakout_above:
            direction = "buy"
            entry = close
            raw_sl_dist = (entry - tokyo_low) / PIP_SIZE + cfg.sl_buffer_pips
            sl_dist = min(raw_sl_dist, cfg.max_sl_pips)
            sl = entry - sl_dist * PIP_SIZE
            if cfg.fixed_tp_pips > 0:
                tp_dist = cfg.fixed_tp_pips
            else:
                tp_dist = sl_dist * cfg.rr_ratio
            tp = entry + tp_dist * PIP_SIZE
        else:
            direction = "sell"
            entry = close
            raw_sl_dist = (tokyo_high - entry) / PIP_SIZE + cfg.sl_buffer_pips
            sl_dist = min(raw_sl_dist, cfg.max_sl_pips)
            sl = entry + sl_dist * PIP_SIZE
            if cfg.fixed_tp_pips > 0:
                tp_dist = cfg.fixed_tp_pips
            else:
                tp_dist = sl_dist * cfg.rr_ratio
            tp = entry - tp_dist * PIP_SIZE

        in_trade = True
        trade_dir = direction
        trade_entry = entry
        trade_sl = sl
        trade_tp = tp
        trade_entry_time = t
        trade_entry_date = date
        trade_tokyo_high = tokyo_high
        trade_tokyo_low = tokyo_low
        trade_range_width = range_width
        bars_held = 0

    return trades


def print_results(trades: list[TradeRecord], cfg: Config, label: str = ""):
    if label:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

    if not trades:
        print("  No trades generated.")
        return

    df = pd.DataFrame([asdict(t) for t in trades])
    total = len(df)
    wins = (df["pips"] > 0).sum()
    losses = (df["pips"] <= 0).sum()
    wr = wins / total * 100
    avg_pips = df["pips"].mean()
    net_pips = df["pips"].sum()
    avg_win = df.loc[df["pips"] > 0, "pips"].mean() if wins > 0 else 0
    avg_loss = df.loc[df["pips"] <= 0, "pips"].mean() if losses > 0 else 0
    gross_win = df.loc[df["pips"] > 0, "pips"].sum()
    gross_loss = abs(df.loc[df["pips"] <= 0, "pips"].sum())
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

    tp_count = (df["outcome"] == "tp").sum()
    sl_count = (df["outcome"] == "sl").sum()
    to_count = (df["outcome"] == "timeout").sum()

    buy_count = (df["direction"] == "buy").sum()
    sell_count = (df["direction"] == "sell").sum()
    buy_pips = df.loc[df["direction"] == "buy", "pips"].sum()
    sell_pips = df.loc[df["direction"] == "sell", "pips"].sum()

    avg_duration = df["duration_bars"].mean()
    avg_range = df["range_width_pips"].mean()

    print(f"  Trades: {total} | Win Rate: {wr:.1f}% | Avg: {avg_pips:+.1f}p | Net: {net_pips:+.1f}p | PF: {pf:.2f}")
    print(f"  Avg Win: {avg_win:+.1f}p | Avg Loss: {avg_loss:+.1f}p | Avg Duration: {avg_duration:.0f} bars")
    print(f"  TP: {tp_count} ({tp_count/total*100:.0f}%) | SL: {sl_count} ({sl_count/total*100:.0f}%) | Timeout: {to_count} ({to_count/total*100:.0f}%)")
    print(f"  Buy: {buy_count} ({buy_pips:+.1f}p) | Sell: {sell_count} ({sell_pips:+.1f}p)")
    print(f"  Avg Tokyo Range: {avg_range:.1f}p")

    # Monthly breakdown
    df["month"] = pd.to_datetime(df["entry_time"]).dt.to_period("M")
    monthly = df.groupby("month").agg(
        trades=("pips", "count"),
        net=("pips", "sum"),
        wr=("pips", lambda x: (x > 0).mean() * 100),
    )
    print("\n  Monthly:")
    for idx, row in monthly.iterrows():
        print(f"    {idx}: {int(row['trades'])} trades | {row['net']:+.1f}p | {row['wr']:.0f}% WR")

    return df


def run_sweep(data: pd.DataFrame):
    print("\n" + "=" * 80)
    print("  PARAMETER SWEEP")
    print("=" * 80)

    results = []
    for tokyo_window in [(0, 6), (1, 6), (2, 6), (3, 6)]:
        for min_r, max_r in [(15, 40), (15, 50), (20, 50), (20, 60), (25, 60)]:
            for rr in [1.0, 1.5, 2.0]:
                for max_sl in [15, 20, 25]:
                    for ema in [True, False]:
                        cfg = Config(
                            tokyo_start_hour=tokyo_window[0],
                            tokyo_end_hour=tokyo_window[1],
                            min_range_pips=min_r,
                            max_range_pips=max_r,
                            rr_ratio=rr,
                            max_sl_pips=max_sl,
                            require_ema_alignment=ema,
                        )
                        trades = run_backtest(data, cfg)
                        if len(trades) < 5:
                            continue
                        df = pd.DataFrame([asdict(t) for t in trades])
                        net = df["pips"].sum()
                        wr = (df["pips"] > 0).mean() * 100
                        gw = df.loc[df["pips"] > 0, "pips"].sum()
                        gl = abs(df.loc[df["pips"] <= 0, "pips"].sum())
                        pf = gw / gl if gl > 0 else 999
                        results.append({
                            "tokyo": f"{tokyo_window[0]:02d}-{tokyo_window[1]:02d}",
                            "range": f"{min_r}-{max_r}",
                            "rr": rr,
                            "max_sl": max_sl,
                            "ema": ema,
                            "trades": len(trades),
                            "wr": round(wr, 1),
                            "net_pips": round(net, 1),
                            "pf": round(pf, 2),
                            "avg_pips": round(df["pips"].mean(), 1),
                        })

    if not results:
        print("  No configurations produced >= 5 trades.")
        return

    rdf = pd.DataFrame(results).sort_values("net_pips", ascending=False)
    print(f"\n  Top 20 configs by net pips (out of {len(rdf)} tested):\n")
    print(f"  {'Tokyo':>6} {'Range':>6} {'RR':>4} {'SL':>4} {'EMA':>5} {'Trades':>6} {'WR%':>5} {'Net':>8} {'PF':>6} {'Avg':>6}")
    print(f"  {'-'*6} {'-'*6} {'-'*4} {'-'*4} {'-'*5} {'-'*6} {'-'*5} {'-'*8} {'-'*6} {'-'*6}")
    for _, row in rdf.head(20).iterrows():
        print(f"  {row['tokyo']:>6} {row['range']:>6} {row['rr']:>4.1f} {row['max_sl']:>4} {str(row['ema']):>5} {row['trades']:>6} {row['wr']:>5.1f} {row['net_pips']:>+8.1f} {row['pf']:>6.2f} {row['avg_pips']:>+6.1f}")

    print(f"\n  Bottom 5 configs:")
    for _, row in rdf.tail(5).iterrows():
        print(f"  {row['tokyo']:>6} {row['range']:>6} {row['rr']:>4.1f} {row['max_sl']:>4} {str(row['ema']):>5} {row['trades']:>6} {row['wr']:>5.1f} {row['net_pips']:>+8.1f} {row['pf']:>6.2f} {row['avg_pips']:>+6.1f}")

    # Save full sweep
    sweep_path = OUT_DIR / "tokyo_range_breakout_v2_sweep.csv"
    rdf.to_csv(sweep_path, index=False)
    print(f"\n  Full sweep saved to {sweep_path}")


def main():
    args = parse_args()

    print("Loading data...")
    data = load_data(args.dataset)
    print(f"Loaded {len(data)} M1 bars: {data['time'].iloc[0]} to {data['time'].iloc[-1]}")

    if args.sweep:
        run_sweep(data)
        return

    cfg = Config(
        tokyo_start_hour=args.tokyo_start,
        tokyo_end_hour=args.tokyo_end,
        breakout_start_hour=args.breakout_start,
        breakout_end_hour=args.breakout_end,
        min_range_pips=args.min_range,
        max_range_pips=args.max_range,
        breakout_pips_beyond=args.breakout_beyond,
        sl_buffer_pips=args.sl_buffer,
        max_sl_pips=args.max_sl,
        rr_ratio=args.rr,
        fixed_tp_pips=args.fixed_tp,
        max_hold_bars=args.max_hold,
        spread_pips=args.spread,
        require_ema_alignment=not args.no_ema,
        require_body_ratio=args.body_ratio,
        min_breakout_bar_pips=args.min_bar_size,
    )

    print(f"\nConfig: Tokyo {cfg.tokyo_start_hour:02d}-{cfg.tokyo_end_hour:02d} UTC | "
          f"Breakout {cfg.breakout_start_hour:02d}-{cfg.breakout_end_hour:02d} UTC | "
          f"Range {cfg.min_range_pips}-{cfg.max_range_pips}p | "
          f"RR {cfg.rr_ratio} | MaxSL {cfg.max_sl_pips}p | "
          f"EMA {'on' if cfg.require_ema_alignment else 'off'}")

    trades = run_backtest(data, cfg)
    trade_df = print_results(trades, cfg, "Tokyo Range Breakout v2")

    if trade_df is not None:
        csv_path = OUT_DIR / "tokyo_range_breakout_v2_trades.csv"
        trade_df.to_csv(csv_path, index=False)
        print(f"\n  Trades saved to {csv_path}")


if __name__ == "__main__":
    main()
