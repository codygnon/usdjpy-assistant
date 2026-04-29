#!/usr/bin/env python3
"""Lightweight screen for the autonomous momentum-continuation runner.

This is intentionally not a production backtest. It checks whether the gate can
find plausible run pullbacks on historical M1 bars, then applies a simple
partial-plus-trail exit approximation so we can sanity-check frequency and
shape before live/shadow testing.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api import autonomous_fillmore


PIP_SIZE = 0.01


@dataclass
class ScreenTrade:
    time: str
    side: str
    entry: float
    exit: float
    pips: float
    tp1_hit: bool
    reason: str
    score: float
    entry_pattern: str
    m5_extension_pips: float
    clear_path_pips: float | None


def _resample(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    out = (
        frame.set_index("time")
        .resample(rule)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
        .reset_index()
    )
    return out


def _session_label(ts: pd.Timestamp) -> str:
    hour = int(ts.hour)
    if 0 <= hour < 7:
        return "tokyo"
    if 7 <= hour < 12:
        return "london"
    if 12 <= hour < 17:
        return "london/ny"
    if 17 <= hour < 21:
        return "ny"
    return "off"


def _simulate_runner(
    frame: pd.DataFrame,
    start_idx: int,
    side: str,
    *,
    horizon_bars: int,
    stop_pips: float,
    tp1_pips: float,
    trail_pips: float,
    partial_pct: float,
) -> tuple[float, float, bool, str]:
    if start_idx >= len(frame):
        return 0.0, float(frame["close"].iloc[-1]), False, "no_forward_bars"
    entry = float(frame["open"].iloc[start_idx])
    stop = entry - stop_pips * PIP_SIZE if side == "buy" else entry + stop_pips * PIP_SIZE
    tp1 = entry + tp1_pips * PIP_SIZE if side == "buy" else entry - tp1_pips * PIP_SIZE
    runner_stop = stop
    tp1_hit = False
    best_price = entry
    partial_weight = max(0.0, min(float(partial_pct), 90.0)) / 100.0
    realized_partial_pips = 0.0
    end = min(len(frame), start_idx + horizon_bars)

    for _, row in frame.iloc[start_idx:end].iterrows():
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        if side == "buy":
            if low <= runner_stop:
                runner_pips = (runner_stop - entry) / PIP_SIZE
                return realized_partial_pips + (1.0 - partial_weight) * runner_pips, runner_stop, tp1_hit, "runner_stop"
            if not tp1_hit and high >= tp1:
                tp1_hit = True
                realized_partial_pips = partial_weight * tp1_pips
                runner_stop = max(runner_stop, entry + 0.5 * PIP_SIZE)
            if tp1_hit:
                best_price = max(best_price, high)
                runner_stop = max(runner_stop, best_price - trail_pips * PIP_SIZE)
        else:
            if high >= runner_stop:
                runner_pips = (entry - runner_stop) / PIP_SIZE
                return realized_partial_pips + (1.0 - partial_weight) * runner_pips, runner_stop, tp1_hit, "runner_stop"
            if not tp1_hit and low <= tp1:
                tp1_hit = True
                realized_partial_pips = partial_weight * tp1_pips
                runner_stop = min(runner_stop, entry - 0.5 * PIP_SIZE)
            if tp1_hit:
                best_price = min(best_price, low)
                runner_stop = min(runner_stop, best_price + trail_pips * PIP_SIZE)

    final = float(frame["close"].iloc[end - 1])
    final_pips = (final - entry) / PIP_SIZE if side == "buy" else (entry - final) / PIP_SIZE
    if tp1_hit:
        return realized_partial_pips + (1.0 - partial_weight) * final_pips, final, True, "horizon"
    return final_pips, final, False, "horizon"


def run_screen(args: argparse.Namespace) -> dict[str, Any]:
    src = Path(args.csv)
    df = pd.read_csv(src)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)
    if args.rows and args.rows > 0:
        df = df.tail(args.rows).reset_index(drop=True)

    thresholds = autonomous_fillmore.GATE_THRESHOLDS["balanced"]
    trades: list[ScreenTrade] = []
    last_signal_idx = -10_000
    cooldown = int(args.cooldown_bars)

    for i in range(max(80, args.warmup_bars), len(df) - args.horizon_bars - 1):
        if i - last_signal_idx < cooldown:
            continue
        window = df.iloc[: i + 1].tail(args.context_bars).copy()
        m5 = _resample(window, "5min")
        m15 = _resample(window, "15min")
        if len(m5) < 25 or len(m15) < 10:
            continue
        ts = pd.Timestamp(window["time"].iloc[-1])
        trig = autonomous_fillmore._momentum_continuation_trigger(
            float(window["close"].iloc[-1]),
            {"M1": window, "M5": m5, "M15": m15},
            _session_label(ts),
            min_m5_atr_pips=float(thresholds["require_min_m5_atr_pips"]),
            adx_min=float(thresholds["momentum_adx_min"]),
            clear_path_pips=float(thresholds["momentum_clear_path_pips"]),
            lookback_bars=int(thresholds["momentum_lookback_bars"]),
            pullback_zone_pips=float(thresholds["momentum_pullback_zone_pips"]),
            extension_atr_mult=float(thresholds["momentum_extension_atr_mult"]),
            pip_size=PIP_SIZE,
        )
        if not trig:
            continue
        side = "buy" if trig.get("bias") == "buy" else "sell"
        pips, exit_price, tp1_hit, reason = _simulate_runner(
            df,
            i + 1,
            side,
            horizon_bars=args.horizon_bars,
            stop_pips=args.stop_pips,
            tp1_pips=args.tp1_pips,
            trail_pips=args.trail_pips,
            partial_pct=args.partial_pct,
        )
        trades.append(
            ScreenTrade(
                time=str(ts),
                side=side,
                entry=float(df["open"].iloc[i + 1]),
                exit=float(exit_price),
                pips=round(float(pips), 2),
                tp1_hit=tp1_hit,
                reason=reason,
                score=float(trig.get("trigger_score") or 0.0),
                entry_pattern=str(trig.get("entry_pattern") or ""),
                m5_extension_pips=float(trig.get("m5_extension_pips") or 0.0),
                clear_path_pips=(
                    float(trig["clear_path_pips"])
                    if isinstance(trig.get("clear_path_pips"), (int, float))
                    else None
                ),
            )
        )
        last_signal_idx = i

    total = len(trades)
    wins = sum(1 for t in trades if t.pips > 0)
    tp1_hits = sum(1 for t in trades if t.tp1_hit)
    net = round(sum(t.pips for t in trades), 2)
    avg = round(net / total, 2) if total else 0.0
    by_side = {
        side: {
            "count": len(items),
            "net_pips": round(sum(t.pips for t in items), 2),
            "avg_pips": round(sum(t.pips for t in items) / len(items), 2) if items else 0.0,
        }
        for side, items in (
            ("buy", [t for t in trades if t.side == "buy"]),
            ("sell", [t for t in trades if t.side == "sell"]),
        )
    }
    return {
        "csv": str(src),
        "rows_screened": int(len(df)),
        "signals": total,
        "win_rate_pct": round(100.0 * wins / total, 1) if total else 0.0,
        "tp1_hit_rate_pct": round(100.0 * tp1_hits / total, 1) if total else 0.0,
        "net_pips": net,
        "avg_pips": avg,
        "by_side": by_side,
        "sample": [t.__dict__ for t in trades[:10]],
        "last_signals": [t.__dict__ for t in trades[-10:]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="research_out/USDJPY_M1_OANDA_500k.csv")
    parser.add_argument("--rows", type=int, default=50000)
    parser.add_argument("--context-bars", type=int, default=360)
    parser.add_argument("--warmup-bars", type=int, default=360)
    parser.add_argument("--horizon-bars", type=int, default=90)
    parser.add_argument("--cooldown-bars", type=int, default=20)
    parser.add_argument("--stop-pips", type=float, default=12.0)
    parser.add_argument("--tp1-pips", type=float, default=6.0)
    parser.add_argument("--trail-pips", type=float, default=4.0)
    parser.add_argument("--partial-pct", type=float, default=33.0)
    args = parser.parse_args()
    result = run_screen(args)
    print(pd.Series({k: v for k, v in result.items() if k not in {"sample", "last_signals", "by_side"}}).to_string())
    print("by_side=", result["by_side"])
    print("sample=", result["sample"])
    print("last_signals=", result["last_signals"])


if __name__ == "__main__":
    main()
