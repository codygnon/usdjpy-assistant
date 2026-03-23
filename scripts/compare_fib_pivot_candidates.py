#!/usr/bin/env python3
"""Compare candidate Fibonacci pivot constructions against target levels.

This is intended to reverse-engineer external platform pivot lines such as
ProRealTime's "Fib support/resistance 1H/2H/3H" labels. It evaluates multiple
candidate constructions from M1 OHLC data:

- Previous completed period, standard Fibonacci pivots (PP uses H/L/C)
- Rolling-window range with midpoint PP
- Rolling-window range with close-based PP (same range window, but PP uses H/L/C)

It also tests shifted intraday anchors (for example, H1 starting on :15 or :30)
because session/cutover alignment is often the real source of platform mismatch.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Iterable

import pandas as pd


LEVELS = ("PP", "R1", "R2", "R3", "S1", "S2", "S3")


@dataclass
class Candidate:
    name: str
    mode: str
    timeframe: str
    offset_minutes: int
    lookback_bars: int | None
    levels: dict[str, float]
    source_high: float
    source_low: float
    source_close: float
    bars_used: int
    score_rmse: float
    score_max_abs: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare fib pivot candidates against target levels.")
    p.add_argument("--csv", required=True, help="Path to M1 OHLC CSV with time/open/high/low/close columns.")
    p.add_argument("--timestamp", required=True, help="Target timestamp, for example '2026-03-23 11:10'.")
    p.add_argument("--tz", default="America/New_York", help="Timezone for --timestamp if naive. Default: America/New_York")
    p.add_argument("--top", type=int, default=12, help="How many top matches to print.")
    p.add_argument("--pp", type=float, required=True)
    p.add_argument("--r1", type=float, required=True)
    p.add_argument("--r2", type=float, required=True)
    p.add_argument("--r3", type=float, required=True)
    p.add_argument("--s1", type=float, required=True)
    p.add_argument("--s2", type=float, required=True)
    p.add_argument("--s3", type=float, required=True)
    p.add_argument(
        "--timeframes",
        default="M15,M30,H1,H2,H3,H4,D",
        help="Comma-separated candidate source timeframes to test.",
    )
    p.add_argument(
        "--lookbacks",
        default="2,3,4,6,8,12,16",
        help="Comma-separated rolling lookbacks to test.",
    )
    p.add_argument(
        "--offset-step-minutes",
        type=int,
        default=15,
        help="Minutes between shifted anchor tests for intraday timeframes.",
    )
    return p.parse_args()


def load_m1_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"time", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time")
    return df


def parse_timestamp(raw: str, tz_name: str) -> pd.Timestamp:
    ts = pd.Timestamp(raw)
    if ts.tzinfo is None:
        ts = ts.tz_localize(tz_name)
    return ts.tz_convert("UTC")


def timeframe_minutes(tf: str) -> int:
    mapping = {
        "M15": 15,
        "M30": 30,
        "H1": 60,
        "H2": 120,
        "H3": 180,
        "H4": 240,
        "D": 1440,
    }
    if tf not in mapping:
        raise ValueError(f"Unsupported timeframe: {tf}")
    return mapping[tf]


def standard_fib_from_hlc(high: float, low: float, close: float) -> dict[str, float] | None:
    rng = float(high) - float(low)
    if rng <= 0:
        return None
    pp = (float(high) + float(low) + float(close)) / 3.0
    return {
        "PP": pp,
        "R1": pp + 0.382 * rng,
        "R2": pp + 0.618 * rng,
        "R3": pp + 1.000 * rng,
        "S1": pp - 0.382 * rng,
        "S2": pp - 0.618 * rng,
        "S3": pp - 1.000 * rng,
    }


def midpoint_fib_from_range(high: float, low: float) -> dict[str, float] | None:
    rng = float(high) - float(low)
    if rng <= 0:
        return None
    pp = (float(high) + float(low)) / 2.0
    return {
        "PP": pp,
        "R1": pp + 0.382 * rng,
        "R2": pp + 0.618 * rng,
        "R3": pp + 1.000 * rng,
        "S1": pp - 0.382 * rng,
        "S2": pp - 0.618 * rng,
        "S3": pp - 1.000 * rng,
    }


def resample_ohlc(df: pd.DataFrame, tf: str, offset_minutes: int) -> pd.DataFrame:
    rule = "1D" if tf == "D" else f"{timeframe_minutes(tf)}min"
    out = df.set_index("time")[["open", "high", "low", "close"]].resample(
        rule,
        label="right",
        closed="right",
        origin="epoch",
        offset=f"{offset_minutes}min",
    ).agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    out = out.dropna(subset=["open", "high", "low", "close"]).reset_index()
    return out


def drop_incomplete_rows(resampled: pd.DataFrame, target_ts: pd.Timestamp) -> pd.DataFrame:
    if resampled.empty:
        return resampled
    completed = resampled[resampled["time"] <= target_ts].copy()
    return completed


def score_levels(candidate: dict[str, float], target: dict[str, float]) -> tuple[float, float]:
    diffs = [candidate[k] - target[k] for k in LEVELS if k in candidate and k in target]
    mse = sum(d * d for d in diffs) / len(diffs)
    rmse = math.sqrt(mse)
    max_abs = max(abs(d) for d in diffs)
    return rmse, max_abs


def build_previous_period_candidate(
    completed: pd.DataFrame,
    tf: str,
    offset_minutes: int,
    target: dict[str, float],
) -> Candidate | None:
    if len(completed) < 1:
        return None
    row = completed.iloc[-1]
    levels = standard_fib_from_hlc(float(row["high"]), float(row["low"]), float(row["close"]))
    if levels is None:
        return None
    rmse, max_abs = score_levels(levels, target)
    return Candidate(
        name=f"prev_{tf}_offset{offset_minutes}",
        mode="previous_period_standard",
        timeframe=tf,
        offset_minutes=offset_minutes,
        lookback_bars=None,
        levels=levels,
        source_high=float(row["high"]),
        source_low=float(row["low"]),
        source_close=float(row["close"]),
        bars_used=1,
        score_rmse=rmse,
        score_max_abs=max_abs,
    )


def build_rolling_candidates(
    completed: pd.DataFrame,
    tf: str,
    offset_minutes: int,
    lookbacks: Iterable[int],
    target: dict[str, float],
) -> list[Candidate]:
    out: list[Candidate] = []
    for lookback in lookbacks:
        if len(completed) < lookback:
            continue
        window = completed.iloc[-lookback:]
        high = float(window["high"].max())
        low = float(window["low"].min())
        close = float(window["close"].iloc[-1])

        midpoint_levels = midpoint_fib_from_range(high, low)
        if midpoint_levels is not None:
            rmse, max_abs = score_levels(midpoint_levels, target)
            out.append(
                Candidate(
                    name=f"roll_mid_{tf}_{lookback}_offset{offset_minutes}",
                    mode="rolling_midpoint",
                    timeframe=tf,
                    offset_minutes=offset_minutes,
                    lookback_bars=lookback,
                    levels=midpoint_levels,
                    source_high=high,
                    source_low=low,
                    source_close=close,
                    bars_used=lookback,
                    score_rmse=rmse,
                    score_max_abs=max_abs,
                )
            )

        standard_levels = standard_fib_from_hlc(high, low, close)
        if standard_levels is not None:
            rmse, max_abs = score_levels(standard_levels, target)
            out.append(
                Candidate(
                    name=f"roll_std_{tf}_{lookback}_offset{offset_minutes}",
                    mode="rolling_standard_close",
                    timeframe=tf,
                    offset_minutes=offset_minutes,
                    lookback_bars=lookback,
                    levels=standard_levels,
                    source_high=high,
                    source_low=low,
                    source_close=close,
                    bars_used=lookback,
                    score_rmse=rmse,
                    score_max_abs=max_abs,
                )
            )
    return out


def candidate_offsets(tf: str, step_minutes: int) -> list[int]:
    if tf == "D":
        return [0]
    tf_mins = timeframe_minutes(tf)
    step = max(1, step_minutes)
    offsets = list(range(0, tf_mins, step))
    return offsets if offsets else [0]


def print_candidate(c: Candidate) -> None:
    print(
        f"{c.name:28s} mode={c.mode:24s} rmse={c.score_rmse:.5f} "
        f"max_abs={c.score_max_abs:.5f} src=H:{c.source_high:.3f} L:{c.source_low:.3f} C:{c.source_close:.3f}"
    )
    print(
        f"  PP={c.levels['PP']:.3f} R1={c.levels['R1']:.3f} R2={c.levels['R2']:.3f} R3={c.levels['R3']:.3f} "
        f"S1={c.levels['S1']:.3f} S2={c.levels['S2']:.3f} S3={c.levels['S3']:.3f}"
    )


def main() -> None:
    args = parse_args()
    df = load_m1_csv(args.csv)
    target_ts = parse_timestamp(args.timestamp, args.tz)
    target = {
        "PP": args.pp,
        "R1": args.r1,
        "R2": args.r2,
        "R3": args.r3,
        "S1": args.s1,
        "S2": args.s2,
        "S3": args.s3,
    }

    timeframes = [x.strip().upper() for x in args.timeframes.split(",") if x.strip()]
    lookbacks = [int(x.strip()) for x in args.lookbacks.split(",") if x.strip()]

    visible = df[df["time"] <= target_ts].copy()
    if visible.empty:
        raise SystemExit("No M1 rows available at or before target timestamp")

    results: list[Candidate] = []
    for tf in timeframes:
        for offset in candidate_offsets(tf, args.offset_step_minutes):
            resampled = resample_ohlc(visible, tf, offset)
            completed = drop_incomplete_rows(resampled, target_ts)
            if completed.empty:
                continue
            prev = build_previous_period_candidate(completed, tf, offset, target)
            if prev is not None:
                results.append(prev)
            results.extend(build_rolling_candidates(completed, tf, offset, lookbacks, target))

    results.sort(key=lambda c: (c.score_rmse, c.score_max_abs))

    print(f"Target timestamp (UTC): {target_ts.isoformat()}")
    print(
        f"Target levels: PP={target['PP']:.3f} R1={target['R1']:.3f} R2={target['R2']:.3f} R3={target['R3']:.3f} "
        f"S1={target['S1']:.3f} S2={target['S2']:.3f} S3={target['S3']:.3f}"
    )
    print()
    print(f"Top {min(args.top, len(results))} matches:")
    for cand in results[: args.top]:
        print_candidate(cand)


if __name__ == "__main__":
    main()
