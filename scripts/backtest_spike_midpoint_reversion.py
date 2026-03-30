#!/usr/bin/env python3
"""
M1 spike → midpoint reversion backtest (research).

Spike definition (per bar i) — configurable:

  **Absolute range (primary):**  (high - low) >= min_spike_pips
  Example: ``--min-spike-pips 70`` = one M1 candle spans at least 70 pips.

  **Optional ATR filter:**  if ``--atr-mult`` > 0, also require
  range >= atr_mult * ATR(14).  Default atr-mult is 0 (disabled) so huge
  candles are identified by pips alone, not relative volatility.

  **Direction:**
  - For range >= mega_relax_pips (default 50): only require a net bearish /
    bullish bar (close vs open) so hammer-like mega spikes still qualify.
  - For smaller ranges: close must sit in the extreme fraction of the bar
    (close_extreme_frac) plus direction.

Bottom / top: spike_low = low[i], spike_high = high[i] for the impulse bar.
50% reversion target = midpoint of that bar's range: (high + low) / 2.

Entry: open of bar i+1 (immediately after spike closes), with half-spread for entry.
Exit: limit at midpoint, or stop beyond spike extreme + buffer.
Intrabar: if both SL and TP touched same bar, assume SL first (conservative).

This does not modify any live strategy — research only.

Usage:
  python scripts/backtest_spike_midpoint_reversion.py \\
    --inputs research_out/USDJPY_M1_OANDA_50k.csv \\
    --min-spike-pips 70 --atr-mult 0 \\
    --out research_out/spike_midpoint_bt_50k.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PIP_SIZE = 0.01


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat(
        [
            h - l,
            (h - prev_c).abs(),
            (l - prev_c).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def _load_m1(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna().sort_values("time").reset_index(drop=True)


@dataclass
class TradeResult:
    direction: str  # "long" | "short"
    spike_time: str
    entry_time: str
    exit_time: str
    spike_high: float
    spike_low: float
    midpoint: float
    entry: float
    exit_price: float
    exit_reason: str  # "tp" | "sl" | "time"
    pips: float
    bars_in_trade: int


def simulate_trade_long(
    df: pd.DataFrame,
    entry_idx: int,
    entry_fill: float,
    sl: float,
    tp: float,
    max_bars: int,
) -> tuple[float, str, int, str, float]:
    """Long from entry_fill (includes spread). Returns (exit_price, reason, bars, exit_time, pips)."""
    n = len(df)
    entry = entry_fill
    if entry <= sl or entry >= tp:
        return entry, "invalid", 0, "", 0.0
    last = min(entry_idx + max_bars, n - 1)
    for j in range(entry_idx, last + 1):
        lo = float(df.loc[j, "low"])
        hi = float(df.loc[j, "high"])
        t = df.loc[j, "time"]
        if lo <= sl:
            pips = (sl - entry) / PIP_SIZE
            return sl, "sl", j - entry_idx + 1, str(t), pips
        if hi >= tp:
            pips = (tp - entry) / PIP_SIZE
            return tp, "tp", j - entry_idx + 1, str(t), pips
    ex = float(df.loc[last, "close"])
    pips = (ex - entry) / PIP_SIZE
    return ex, "time", last - entry_idx + 1, str(df.loc[last, "time"]), pips


def simulate_trade_short(
    df: pd.DataFrame,
    entry_idx: int,
    entry_fill: float,
    sl: float,
    tp: float,
    max_bars: int,
) -> tuple[float, str, int, str, float]:
    entry = entry_fill
    if entry >= sl or entry <= tp:
        return entry, "invalid", 0, "", 0.0
    n = len(df)
    last = min(entry_idx + max_bars, n - 1)
    for j in range(entry_idx, last + 1):
        lo = float(df.loc[j, "low"])
        hi = float(df.loc[j, "high"])
        t = df.loc[j, "time"]
        if hi >= sl:
            pips = (entry - sl) / PIP_SIZE
            return sl, "sl", j - entry_idx + 1, str(t), pips
        if lo <= tp:
            pips = (entry - tp) / PIP_SIZE
            return tp, "tp", j - entry_idx + 1, str(t), pips
    ex = float(df.loc[last, "close"])
    pips = (entry - ex) / PIP_SIZE
    return ex, "time", last - entry_idx + 1, str(df.loc[last, "time"]), pips


def run_backtest(
    df: pd.DataFrame,
    *,
    spike_atr_mult: float = 0.0,
    min_spike_pips: float = 70.0,
    mega_relax_pips: float = 50.0,
    close_extreme_frac: float = 0.25,
    spread_pips: float = 0.35,
    stop_buffer_pips: float = 3.0,
    cooldown_bars: int = 30,
    max_hold_bars: int = 60,
    long_only: bool = False,
    short_only: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    df = df.copy().reset_index(drop=True)
    atr = _atr(df, 14)
    rng = df["high"] - df["low"]

    spread = spread_pips * PIP_SIZE
    buf = stop_buffer_pips * PIP_SIZE
    min_r = min_spike_pips * PIP_SIZE
    mega_r = mega_relax_pips * PIP_SIZE

    trades: list[TradeResult] = []
    next_allowed = 0

    for i in range(15, len(df) - 2):
        if i < next_allowed:
            continue
        r = rng.iloc[i]
        if r < min_r:
            continue
        if spike_atr_mult > 0:
            a = atr.iloc[i]
            if pd.isna(a) or a <= 0:
                continue
            if r < spike_atr_mult * a:
                continue

        o = float(df.loc[i, "open"])
        h = float(df.loc[i, "high"])
        l = float(df.loc[i, "low"])
        c = float(df.loc[i, "close"])
        midpoint = (h + l) / 2.0
        mega_bar = r >= mega_r

        # Down spike → long reversion
        close_near_low = c <= l + close_extreme_frac * r
        bearish = c < o
        down_ok = bearish and (mega_bar or close_near_low)
        if not short_only and down_ok:
            entry_idx = i + 1
            entry_raw = float(df.loc[entry_idx, "open"])
            entry = entry_raw + spread  # pay spread on buy
            sl = l - buf
            tp = midpoint
            if entry >= tp - 1e-9:
                continue
            if entry <= sl:
                continue
            ex, reason, nb, ext, pips = simulate_trade_long(
                df, entry_idx, entry, sl, tp, max_hold_bars
            )
            if reason == "invalid":
                continue
            trades.append(
                TradeResult(
                    direction="long",
                    spike_time=str(df.loc[i, "time"]),
                    entry_time=str(df.loc[entry_idx, "time"]),
                    exit_time=ext,
                    spike_high=h,
                    spike_low=l,
                    midpoint=tp,
                    entry=entry,
                    exit_price=ex,
                    exit_reason=reason,
                    pips=pips,
                    bars_in_trade=nb,
                )
            )
            next_allowed = i + 1 + cooldown_bars
            continue

        # Up spike → short reversion
        close_near_high = c >= h - close_extreme_frac * r
        bullish = c > o
        up_ok = bullish and (mega_bar or close_near_high)
        if not long_only and up_ok:
            entry_idx = i + 1
            entry_raw = float(df.loc[entry_idx, "open"])
            entry = entry_raw - spread  # short fill
            sl = h + buf
            tp = midpoint
            if entry <= tp + 1e-9:
                continue
            if entry >= sl:
                continue
            ex, reason, nb, ext, pips = simulate_trade_short(
                df, entry_idx, entry, sl, tp, max_hold_bars
            )
            if reason == "invalid":
                continue
            trades.append(
                TradeResult(
                    direction="short",
                    spike_time=str(df.loc[i, "time"]),
                    entry_time=str(df.loc[entry_idx, "time"]),
                    exit_time=ext,
                    spike_high=h,
                    spike_low=l,
                    midpoint=tp,
                    entry=entry,
                    exit_price=ex,
                    exit_reason=reason,
                    pips=pips,
                    bars_in_trade=nb,
                )
            )
            next_allowed = i + 1 + cooldown_bars

    rows = [asdict(t) for t in trades]
    wins = [t for t in trades if t.pips > 0]
    losses = [t for t in trades if t.pips <= 0]
    tp_hits = [t for t in trades if t.exit_reason == "tp"]
    sl_hits = [t for t in trades if t.exit_reason == "sl"]

    summary = {
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(100.0 * len(wins) / len(trades), 2) if trades else 0.0,
        "total_pips": round(sum(t.pips for t in trades), 2),
        "avg_pips": round(np.mean([t.pips for t in trades]), 4) if trades else 0.0,
        "tp_exits": len(tp_hits),
        "sl_exits": len(sl_hits),
        "time_exits": len([t for t in trades if t.exit_reason == "time"]),
        "long_trades": len([t for t in trades if t.direction == "long"]),
        "short_trades": len([t for t in trades if t.direction == "short"]),
        "params": {
            "spike_atr_mult": spike_atr_mult,
            "min_spike_pips": min_spike_pips,
            "mega_relax_pips": mega_relax_pips,
            "close_extreme_frac": close_extreme_frac,
            "spread_pips": spread_pips,
            "stop_buffer_pips": stop_buffer_pips,
            "cooldown_bars": cooldown_bars,
            "max_hold_bars": max_hold_bars,
        },
    }
    return rows, summary


def main() -> int:
    p = argparse.ArgumentParser(description="M1 spike midpoint reversion backtest")
    p.add_argument(
        "--inputs",
        nargs="+",
        type=str,
        default=[str(ROOT / "research_out" / "USDJPY_M1_OANDA_50k.csv")],
        help="One or more M1 CSV paths",
    )
    p.add_argument("--out", type=str, default="", help="Write JSON summary + trades here")
    p.add_argument(
        "--atr-mult",
        type=float,
        default=0.0,
        help="If >0, also require range >= atr_mult * ATR(14). 0 = off (absolute pips only).",
    )
    p.add_argument(
        "--min-spike-pips",
        type=float,
        default=70.0,
        help="Minimum single-bar range in pips (e.g. 70 = one M1 candle spans 70+ pips).",
    )
    p.add_argument(
        "--mega-relax-pips",
        type=float,
        default=50.0,
        help="If bar range >= this many pips, only net direction (close vs open) is required.",
    )
    p.add_argument("--close-extreme-frac", type=float, default=0.25)
    p.add_argument("--spread-pips", type=float, default=0.35)
    p.add_argument("--stop-buffer-pips", type=float, default=3.0)
    p.add_argument("--cooldown-bars", type=int, default=30)
    p.add_argument("--max-hold-bars", type=int, default=60)
    p.add_argument("--long-only", action="store_true")
    p.add_argument("--short-only", action="store_true")
    args = p.parse_args()

    all_rows: list[dict[str, Any]] = []
    per_file: list[dict[str, Any]] = []

    for path_str in args.inputs:
        path = Path(path_str)
        if not path.is_file():
            print(f"SKIP missing file: {path}", file=sys.stderr)
            continue
        df = _load_m1(path)
        rows, summary = run_backtest(
            df,
            spike_atr_mult=args.atr_mult,
            min_spike_pips=args.min_spike_pips,
            mega_relax_pips=args.mega_relax_pips,
            close_extreme_frac=args.close_extreme_frac,
            spread_pips=args.spread_pips,
            stop_buffer_pips=args.stop_buffer_pips,
            cooldown_bars=args.cooldown_bars,
            max_hold_bars=args.max_hold_bars,
            long_only=args.long_only,
            short_only=args.short_only,
        )
        summary["bars"] = len(df)
        summary["source"] = str(path)
        per_file.append(summary)
        for r in rows:
            r["source_file"] = str(path)
        all_rows.extend(rows)
        print(f"\n=== {path.name} ===")
        print(json.dumps(summary, indent=2))

    if not per_file:
        print("No data processed.", file=sys.stderr)
        return 1

    total_trades = sum(s["total_trades"] for s in per_file)
    total_pips = sum(s["total_pips"] for s in per_file)
    agg = {
        "files": len(per_file),
        "total_trades": total_trades,
        "total_pips": round(total_pips, 2),
        "avg_pips_per_trade": round(total_pips / total_trades, 4) if total_trades else 0.0,
        "per_file": per_file,
    }
    print("\n=== AGGREGATE ===")
    print(json.dumps(agg, indent=2))

    out_payload = {"aggregate": agg, "trades": all_rows}
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
        print(f"\nWrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
