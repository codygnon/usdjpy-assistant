#!/usr/bin/env python3
"""
Research backtest for the Tokyo range breakout family.

Hypothesis:
  - USDJPY often builds a bounded Tokyo-session range.
  - Fresh London liquidity can break that range with follow-through.
  - The best signals are clean M5 breakout closes near the bar extreme,
    with the M1 EMA stack already leaning in breakout direction.

This is research-only and intentionally separate from the live autonomous gate.
It uses the same fast first-touch forward outcome proxy as the other gate-family
research tools:
  - enter on the M5 breakout bar close
  - win if +target_pips hits before -stop_pips within horizon_bars
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import calibrate_autonomous_gate as cag

OUT_DIR = ROOT / "research_out"
DEFAULT_DATASETS = [
    OUT_DIR / "USDJPY_M1_OANDA_500k.csv",
    OUT_DIR / "USDJPY_M1_OANDA_1000k.csv",
]
DEFAULT_JSON = OUT_DIR / "tokyo_range_breakout_backtest.json"
DEFAULT_MD = OUT_DIR / "tokyo_range_breakout_backtest.md"
PIP_SIZE = cag.PIP_SIZE
TOKYO_RANGE_BREAKOUT_FAMILY = "tokyo_range_breakout_v1"


@dataclass
class FamilySummary:
    trades: int
    win_rate_pct: float
    avg_pips: float
    avg_final_pips: float
    avg_mfe_pips: float
    avg_mae_pips: float
    net_pips: float
    profit_factor: float | None
    tp_hit_rate_pct: float
    stop_hit_rate_pct: float
    timeout_rate_pct: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest the Tokyo range breakout family on historical USDJPY M1 data")
    p.add_argument(
        "--datasets",
        nargs="*",
        default=[str(p) for p in DEFAULT_DATASETS if p.exists()],
        help="Dataset path(s). Defaults to existing 500k/1000k research datasets.",
    )
    p.add_argument("--target-pips", type=float, default=6.0, help="Forward outcome take-profit proxy")
    p.add_argument("--stop-pips", type=float, default=10.0, help="Forward outcome stop proxy")
    p.add_argument("--horizon-bars", type=int, default=30, help="Forward outcome horizon in M1 bars")
    p.add_argument(
        "--train-bars",
        type=int,
        default=700000,
        help="Most-recent historical bars to use for the train slice before the test slice.",
    )
    p.add_argument(
        "--test-bars",
        type=int,
        default=300000,
        help="Most-recent historical bars to reserve for the out-of-sample test slice.",
    )
    p.add_argument(
        "--assume-spread-pips",
        type=float,
        default=1.2,
        help="Fallback spread when dataset lacks spread_pips column.",
    )
    p.add_argument("--min-range-pips", type=float, default=8.0, help="Minimum Tokyo range width in pips")
    p.add_argument("--max-range-pips", type=float, default=20.0, help="Maximum Tokyo range width in pips")
    p.add_argument(
        "--breakout-body-ratio",
        type=float,
        default=0.60,
        help="Minimum body/range ratio on the breakout M5 bar",
    )
    p.add_argument(
        "--max-close-to-extreme-pips",
        type=float,
        default=2.0,
        help="Maximum distance from breakout close to the bar extreme",
    )
    p.add_argument(
        "--max-breakout-close-excursion-pips",
        type=float,
        default=8.0,
        help="Maximum close excursion beyond the Tokyo range",
    )
    p.add_argument(
        "--require-ema-alignment",
        action="store_true",
        default=True,
        help="Require M1 EMA(5) vs EMA(9) alignment in breakout direction (default on)",
    )
    p.add_argument("--output-json", default=str(DEFAULT_JSON))
    p.add_argument("--output-md", default=str(DEFAULT_MD))
    return p.parse_args()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)


def _dataset_key(path: Path) -> str:
    name = path.name
    if "500k" in name:
        return "500k"
    if "1000k" in name:
        return "1000k"
    return name


def _minute_of_day(ts: pd.Series) -> pd.Series:
    utc = ts.dt.tz_convert("UTC")
    return utc.dt.hour * 60 + utc.dt.minute


def _session_date(ts: pd.Series) -> pd.Series:
    return ts.dt.tz_convert("UTC").dt.floor("D")


def _tokyo_session_key(ts: pd.Series) -> pd.Series:
    minute = _minute_of_day(ts)
    session_date = _session_date(ts)
    return session_date - pd.to_timedelta(np.where(minute <= 7 * 60, 1, 0), unit="D")


def _prepare_m5_frame(m1_frame: pd.DataFrame) -> pd.DataFrame:
    work = m1_frame[["time", "open", "high", "low", "close"]].copy()
    work["m1_e5"] = work["close"].astype(float).ewm(span=5, adjust=False).mean()
    work["m1_e9"] = work["close"].astype(float).ewm(span=9, adjust=False).mean()

    m5 = cag._resample_ohlc(work, "5min")
    ema_source = work[["time", "m1_e5", "m1_e9"]].sort_values("time")
    m5 = pd.merge_asof(
        m5.sort_values("time"),
        ema_source,
        on="time",
        direction="backward",
    )
    m5["session_label"] = m5["time"].map(cag._classify_session_label)
    m5["minute_of_day"] = _minute_of_day(m5["time"])
    m5["bar_body_ratio"] = [
        cag._body_ratio(o, h, l, c) for o, h, l, c in zip(m5["open"], m5["high"], m5["low"], m5["close"])
    ]
    m5["tokyo_session_key"] = _tokyo_session_key(m5["time"])
    return m5


def scan_tokyo_range_breakout_signals(
    m5: pd.DataFrame,
    *,
    min_range_pips: float,
    max_range_pips: float,
    breakout_body_ratio: float,
    max_close_to_extreme_pips: float,
    max_breakout_close_excursion_pips: float,
    require_ema_alignment: bool,
) -> pd.DataFrame:
    if m5.empty:
        return pd.DataFrame(
            columns=[
                "time",
                "signal_direction",
                "trigger_family",
                "breakout_side",
                "tokyo_range_high",
                "tokyo_range_low",
                "tokyo_range_width_pips",
                "breakout_close_excursion_pips",
                "breakout_bar_body_ratio",
                "session_label",
            ]
        )

    tokyo_mask = (m5["minute_of_day"] > 23 * 60) | (m5["minute_of_day"] <= 7 * 60)
    tokyo_ranges = (
        m5.loc[tokyo_mask]
        .groupby("tokyo_session_key", dropna=False)
        .agg(
            tokyo_range_high=("high", "max"),
            tokyo_range_low=("low", "min"),
            tokyo_bar_count=("time", "count"),
        )
        .reset_index()
    )
    tokyo_ranges["tokyo_range_width_pips"] = (
        (tokyo_ranges["tokyo_range_high"] - tokyo_ranges["tokyo_range_low"]) / PIP_SIZE
    )

    breakout = m5.copy()
    breakout["trade_day"] = _session_date(breakout["time"])
    breakout["breakout_session_key"] = breakout["trade_day"] - pd.Timedelta(days=1)
    breakout = breakout.merge(
        tokyo_ranges,
        left_on="breakout_session_key",
        right_on="tokyo_session_key",
        how="left",
    )

    london_open_mask = (breakout["minute_of_day"] > 7 * 60) & (breakout["minute_of_day"] <= 9 * 60)
    breakout = breakout.loc[london_open_mask].copy()
    breakout["tokyo_range_width_pips"] = pd.to_numeric(breakout["tokyo_range_width_pips"], errors="coerce")
    breakout = breakout.loc[
        breakout["tokyo_range_width_pips"].between(float(min_range_pips), float(max_range_pips), inclusive="both")
    ].copy()
    if breakout.empty:
        return pd.DataFrame(
            columns=[
                "time",
                "signal_direction",
                "trigger_family",
                "breakout_side",
                "tokyo_range_high",
                "tokyo_range_low",
                "tokyo_range_width_pips",
                "breakout_close_excursion_pips",
                "breakout_bar_body_ratio",
                "session_label",
            ]
        )

    breakout["up_close_excursion_pips"] = ((breakout["close"] - breakout["tokyo_range_high"]) / PIP_SIZE).clip(lower=0.0)
    breakout["down_close_excursion_pips"] = ((breakout["tokyo_range_low"] - breakout["close"]) / PIP_SIZE).clip(lower=0.0)
    breakout["up_close_to_extreme_pips"] = ((breakout["high"] - breakout["close"]) / PIP_SIZE).clip(lower=0.0)
    breakout["down_close_to_extreme_pips"] = ((breakout["close"] - breakout["low"]) / PIP_SIZE).clip(lower=0.0)
    breakout["ema_bull"] = breakout["m1_e5"] > breakout["m1_e9"]
    breakout["ema_bear"] = breakout["m1_e5"] < breakout["m1_e9"]

    can_buy = (
        (breakout["close"] > breakout["tokyo_range_high"])
        & (breakout["bar_body_ratio"] >= float(breakout_body_ratio))
        & (breakout["up_close_to_extreme_pips"] <= float(max_close_to_extreme_pips))
        & (breakout["up_close_excursion_pips"] <= float(max_breakout_close_excursion_pips))
    )
    can_sell = (
        (breakout["close"] < breakout["tokyo_range_low"])
        & (breakout["bar_body_ratio"] >= float(breakout_body_ratio))
        & (breakout["down_close_to_extreme_pips"] <= float(max_close_to_extreme_pips))
        & (breakout["down_close_excursion_pips"] <= float(max_breakout_close_excursion_pips))
    )
    if require_ema_alignment:
        can_buy &= breakout["ema_bull"]
        can_sell &= breakout["ema_bear"]

    breakout["signal_direction"] = np.select([can_buy, can_sell], [1, -1], default=0)
    breakout = breakout.loc[breakout["signal_direction"] != 0].copy()
    if breakout.empty:
        return pd.DataFrame(
            columns=[
                "time",
                "signal_direction",
                "trigger_family",
                "breakout_side",
                "tokyo_range_high",
                "tokyo_range_low",
                "tokyo_range_width_pips",
                "breakout_close_excursion_pips",
                "breakout_bar_body_ratio",
                "session_label",
            ]
        )

    breakout["breakout_side"] = np.where(breakout["signal_direction"] > 0, "up", "down")
    breakout["breakout_close_excursion_pips"] = np.where(
        breakout["signal_direction"] > 0,
        breakout["up_close_excursion_pips"],
        breakout["down_close_excursion_pips"],
    )
    breakout["trigger_family"] = TOKYO_RANGE_BREAKOUT_FAMILY
    breakout["breakout_bar_body_ratio"] = breakout["bar_body_ratio"].astype(float)
    breakout = (
        breakout.sort_values("time")
        .drop_duplicates(subset=["breakout_session_key"], keep="first")
        [
            [
                "time",
                "signal_direction",
                "trigger_family",
                "breakout_side",
                "tokyo_range_high",
                "tokyo_range_low",
                "tokyo_range_width_pips",
                "breakout_close_excursion_pips",
                "breakout_bar_body_ratio",
                "session_label",
            ]
        ]
        .reset_index(drop=True)
    )
    return breakout


def attach_directional_outcomes(
    frame: pd.DataFrame,
    *,
    target_pips: float,
    stop_pips: float,
    horizon_bars: int,
) -> pd.DataFrame:
    return cag.attach_bidirectional_outcomes(
        frame,
        target_pips=target_pips,
        stop_pips=stop_pips,
        horizon_bars=horizon_bars,
    )


def materialize_signal_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    chosen = frame.loc[frame["signal_direction"].isin([-1, 1])].copy()
    if chosen.empty:
        chosen["outcome_pips"] = pd.Series(dtype=float)
        chosen["outcome_code"] = pd.Series(dtype=object)
        chosen["final_pips"] = pd.Series(dtype=float)
        chosen["mfe_pips"] = pd.Series(dtype=float)
        chosen["mae_pips"] = pd.Series(dtype=float)
        return chosen

    long_mask = chosen["signal_direction"] > 0
    chosen["outcome_pips"] = np.where(long_mask, chosen["long_outcome_pips"], chosen["short_outcome_pips"])
    chosen["outcome_code"] = np.where(long_mask, chosen["long_outcome_code"], chosen["short_outcome_code"])
    chosen["final_pips"] = np.where(long_mask, chosen["long_final_pips"], chosen["short_final_pips"])
    chosen["mfe_pips"] = np.where(long_mask, chosen["long_mfe_pips"], chosen["short_mfe_pips"])
    chosen["mae_pips"] = np.where(long_mask, chosen["long_mae_pips"], chosen["short_mae_pips"])
    return chosen


def summarize_signals(frame: pd.DataFrame) -> FamilySummary:
    chosen = materialize_signal_outcomes(frame)
    if chosen.empty:
        return FamilySummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None, 0.0, 0.0, 0.0)

    wins = chosen["outcome_pips"] > 0
    gross_win = float(chosen.loc[chosen["outcome_pips"] > 0, "outcome_pips"].sum())
    gross_loss = abs(float(chosen.loc[chosen["outcome_pips"] < 0, "outcome_pips"].sum()))
    pf = gross_win / gross_loss if gross_loss > 0 else (999.0 if gross_win > 0 else None)
    return FamilySummary(
        trades=int(len(chosen)),
        win_rate_pct=round(float(wins.mean()) * 100.0, 2),
        avg_pips=round(float(chosen["outcome_pips"].mean()), 4),
        avg_final_pips=round(float(chosen["final_pips"].mean()), 4),
        avg_mfe_pips=round(float(chosen["mfe_pips"].mean()), 4),
        avg_mae_pips=round(float(chosen["mae_pips"].mean()), 4),
        net_pips=round(float(chosen["outcome_pips"].sum()), 2),
        profit_factor=None if pf is None else round(float(pf), 4),
        tp_hit_rate_pct=round(float((chosen["outcome_code"] == "target").mean()) * 100.0, 2),
        stop_hit_rate_pct=round(float((chosen["outcome_code"].isin(["stop", "stop_ambiguous"])).mean()) * 100.0, 2),
        timeout_rate_pct=round(float((chosen["outcome_code"] == "timeout").mean()) * 100.0, 2),
    )


def breakdown_counts(frame: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    chosen = materialize_signal_outcomes(frame)
    if chosen.empty or column not in chosen.columns:
        return []
    rows: list[dict[str, Any]] = []
    for value, subset in chosen.groupby(column, dropna=False):
        rows.append({column: str(value), **asdict(summarize_signals(subset))})
    rows.sort(key=lambda row: (-row["trades"], str(row[column])))
    return rows


def _range_bucket(width_pips: float) -> str:
    if pd.isna(width_pips):
        return "unknown"
    if width_pips < 12.0:
        return "8-12"
    if width_pips < 16.0:
        return "12-16"
    return "16-20"


def run_dataset(
    dataset_path: Path,
    *,
    assumed_spread_pips: float,
    target_pips: float,
    stop_pips: float,
    horizon_bars: int,
    train_bars: int,
    test_bars: int,
    min_range_pips: float,
    max_range_pips: float,
    breakout_body_ratio: float,
    max_close_to_extreme_pips: float,
    max_breakout_close_excursion_pips: float,
    require_ema_alignment: bool,
) -> dict[str, Any]:
    raw = cag._load_dataset(dataset_path, assumed_spread_pips=assumed_spread_pips)
    frame = attach_directional_outcomes(
        raw,
        target_pips=target_pips,
        stop_pips=stop_pips,
        horizon_bars=horizon_bars,
    )
    m5 = _prepare_m5_frame(raw)
    signals = scan_tokyo_range_breakout_signals(
        m5,
        min_range_pips=min_range_pips,
        max_range_pips=max_range_pips,
        breakout_body_ratio=breakout_body_ratio,
        max_close_to_extreme_pips=max_close_to_extreme_pips,
        max_breakout_close_excursion_pips=max_breakout_close_excursion_pips,
        require_ema_alignment=require_ema_alignment,
    )
    merged = frame.merge(signals, on="time", how="left")
    merged["signal_direction"] = pd.to_numeric(merged["signal_direction"], errors="coerce").fillna(0).astype(int)
    merged["trigger_family"] = merged["trigger_family"].fillna("none")
    merged["breakout_side"] = merged["breakout_side"].fillna("none")
    merged["session_label"] = merged["time"].map(cag._classify_session_label)
    merged["range_bucket"] = merged["tokyo_range_width_pips"].map(_range_bucket)

    train_frame, test_frame = cag.split_frame_train_test(merged, train_bars=train_bars, test_bars=test_bars)
    return {
        "dataset_meta": {
            "bars": int(len(merged)),
            "m5_bars": int(len(m5)),
            "signal_rows": int((merged["signal_direction"] != 0).sum()),
            "spread_source": str(raw["spread_source"].iloc[0]) if len(raw) else "unknown",
        },
        "full": {
            "summary": asdict(summarize_signals(merged)),
            "by_direction": breakdown_counts(
                merged.assign(
                    direction_label=np.where(
                        merged["signal_direction"] > 0,
                        "buy",
                        np.where(merged["signal_direction"] < 0, "sell", "none"),
                    )
                ),
                "direction_label",
            ),
            "by_breakout_side": breakdown_counts(merged, "breakout_side"),
            "by_range_bucket": breakdown_counts(merged, "range_bucket"),
        },
        "train": {"summary": asdict(summarize_signals(train_frame))},
        "test": {"summary": asdict(summarize_signals(test_frame))},
    }


def build_markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Tokyo Range Breakout Backtest",
        "",
        "Research-only test of the `tokyo_range_breakout` family on historical USDJPY M1 data.",
        "",
        "## Config",
        "",
        f"- target pips: `{payload['config']['target_pips']}`",
        f"- stop pips: `{payload['config']['stop_pips']}`",
        f"- horizon bars: `{payload['config']['horizon_bars']}`",
        f"- train bars: `{payload['config']['train_bars']}`",
        f"- test bars: `{payload['config']['test_bars']}`",
        f"- Tokyo range width: `{payload['config']['min_range_pips']}` to `{payload['config']['max_range_pips']}` pips",
        f"- breakout body ratio: `{payload['config']['breakout_body_ratio']}`",
        f"- max close-to-extreme: `{payload['config']['max_close_to_extreme_pips']}` pips",
        f"- max breakout close excursion: `{payload['config']['max_breakout_close_excursion_pips']}` pips",
        f"- require EMA alignment: `{payload['config']['require_ema_alignment']}`",
        "",
    ]

    for ds_key, ds_payload in payload["datasets"].items():
        meta = ds_payload["dataset_meta"]
        full = ds_payload["full"]
        train = ds_payload["train"]["summary"]
        test = ds_payload["test"]["summary"]
        full_summary = full["summary"]
        lines.extend(
            [
                f"## Dataset: `{ds_key}`",
                "",
                f"- bars: `{meta['bars']}` | m5 bars: `{meta['m5_bars']}` | signal rows: `{meta['signal_rows']}` | spread source: `{meta['spread_source']}`",
                f"- full sample: `{full_summary['trades']}` trades | `{full_summary['win_rate_pct']}%` WR | avg `{full_summary['avg_pips']}p` | net `{full_summary['net_pips']}p` | PF `{full_summary['profit_factor']}`",
                f"- train: `{train['trades']}` trades | `{train['win_rate_pct']}%` WR | avg `{train['avg_pips']}p` | net `{train['net_pips']}p` | PF `{train['profit_factor']}`",
                f"- test: `{test['trades']}` trades | `{test['win_rate_pct']}%` WR | avg `{test['avg_pips']}p` | net `{test['net_pips']}p` | PF `{test['profit_factor']}`",
                "",
                "### Direction Breakdown",
                "",
            ]
        )
        for row in full["by_direction"]:
            lines.append(
                f"- `{row['direction_label']}`: trades `{row['trades']}` | WR `{row['win_rate_pct']}%` | avg `{row['avg_pips']}p` | net `{row['net_pips']}p` | PF `{row['profit_factor']}`"
            )
        lines.extend(["", "### Breakout Side Breakdown", ""])
        for row in full["by_breakout_side"]:
            lines.append(
                f"- `{row['breakout_side']}`: trades `{row['trades']}` | WR `{row['win_rate_pct']}%` | avg `{row['avg_pips']}p` | net `{row['net_pips']}p` | PF `{row['profit_factor']}`"
            )
        lines.extend(["", "### Tokyo Range Width Breakdown", ""])
        for row in full["by_range_bucket"]:
            lines.append(
                f"- `{row['range_bucket']}`: trades `{row['trades']}` | WR `{row['win_rate_pct']}%` | avg `{row['avg_pips']}p` | net `{row['net_pips']}p` | PF `{row['profit_factor']}`"
            )
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def main() -> int:
    args = parse_args()
    dataset_paths = [Path(p).resolve() for p in args.datasets]
    if not dataset_paths:
        raise SystemExit("No datasets provided.")

    payload: dict[str, Any] = {
        "datasets_requested": [str(p) for p in dataset_paths],
        "config": {
            "target_pips": float(args.target_pips),
            "stop_pips": float(args.stop_pips),
            "horizon_bars": int(args.horizon_bars),
            "train_bars": int(args.train_bars),
            "test_bars": int(args.test_bars),
            "assumed_spread_pips": float(args.assume_spread_pips),
            "min_range_pips": float(args.min_range_pips),
            "max_range_pips": float(args.max_range_pips),
            "breakout_body_ratio": float(args.breakout_body_ratio),
            "max_close_to_extreme_pips": float(args.max_close_to_extreme_pips),
            "max_breakout_close_excursion_pips": float(args.max_breakout_close_excursion_pips),
            "require_ema_alignment": bool(args.require_ema_alignment),
        },
        "datasets": {},
    }

    for dataset_path in dataset_paths:
        payload["datasets"][_dataset_key(dataset_path)] = run_dataset(
            dataset_path,
            assumed_spread_pips=float(args.assume_spread_pips),
            target_pips=float(args.target_pips),
            stop_pips=float(args.stop_pips),
            horizon_bars=int(args.horizon_bars),
            train_bars=int(args.train_bars),
            test_bars=int(args.test_bars),
            min_range_pips=float(args.min_range_pips),
            max_range_pips=float(args.max_range_pips),
            breakout_body_ratio=float(args.breakout_body_ratio),
            max_close_to_extreme_pips=float(args.max_close_to_extreme_pips),
            max_breakout_close_excursion_pips=float(args.max_breakout_close_excursion_pips),
            require_ema_alignment=bool(args.require_ema_alignment),
        )

    json_path = Path(args.output_json)
    md_path = Path(args.output_md)
    json_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    md_path.write_text(build_markdown_report(payload), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
