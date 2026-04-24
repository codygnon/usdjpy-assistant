#!/usr/bin/env python3
"""
Research backtest for the session handoff fade family.

Hypothesis:
  - When London spends most of its session pushing in one direction and ends near
    the session extreme, NY-open flow often fades part of that move.
  - The edge should appear only when London was truly one-sided, not just noisy.

This is research-only and intentionally separate from the live autonomous gate.
It uses the same fast first-touch forward outcome proxy as the other gate-family
research tools:
  - enter on the NY reversal bar close
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
DEFAULT_JSON = OUT_DIR / "session_handoff_fade_backtest.json"
DEFAULT_MD = OUT_DIR / "session_handoff_fade_backtest.md"
PIP_SIZE = cag.PIP_SIZE
SESSION_HANDOFF_FADE_FAMILY = "session_handoff_fade_v1"


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
    p = argparse.ArgumentParser(description="Backtest the session handoff fade family on historical USDJPY M1 data")
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
    p.add_argument("--min-london-range-pips", type=float, default=12.0, help="Minimum London session range")
    p.add_argument(
        "--min-london-onesidedness",
        type=float,
        default=0.70,
        help="Minimum fraction of London range consumed in one direction from open to close",
    )
    p.add_argument(
        "--max-distance-to-extreme-pips",
        type=float,
        default=3.0,
        help="London close must finish within this distance of the session high/low",
    )
    p.add_argument(
        "--ny-window-bars",
        type=int,
        default=6,
        help="Only consider first N M5 bars of NY for the reversal confirmation",
    )
    p.add_argument(
        "--reversal-body-ratio",
        type=float,
        default=0.50,
        help="Minimum body/range ratio on the NY reversal bar",
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


def _session_date(ts: pd.Series) -> pd.Series:
    return ts.dt.tz_convert("UTC").dt.floor("D")


def _minute_of_day(ts: pd.Series) -> pd.Series:
    utc = ts.dt.tz_convert("UTC")
    return utc.dt.hour * 60 + utc.dt.minute


def _prepare_m5_frame(m1_frame: pd.DataFrame) -> pd.DataFrame:
    work = m1_frame[["time", "open", "high", "low", "close"]].copy()
    m5 = cag._resample_ohlc(work, "5min")
    m5["session_label"] = m5["time"].map(cag._classify_session_label)
    m5["session_day"] = _session_date(m5["time"])
    m5["minute_of_day"] = _minute_of_day(m5["time"])
    m5["bar_body_ratio"] = [
        cag._body_ratio(o, h, l, c) for o, h, l, c in zip(m5["open"], m5["high"], m5["low"], m5["close"])
    ]
    return m5


def scan_session_handoff_fade_signals(
    m5: pd.DataFrame,
    *,
    min_london_range_pips: float,
    min_london_onesidedness: float,
    max_distance_to_extreme_pips: float,
    ny_window_bars: int,
    reversal_body_ratio: float,
) -> pd.DataFrame:
    empty = pd.DataFrame(
        columns=[
            "time",
            "signal_direction",
            "trigger_family",
            "london_direction",
            "london_range_pips",
            "london_onesidedness",
            "distance_to_london_extreme_pips",
            "ny_bar_index",
            "session_label",
        ]
    )
    if m5.empty:
        return empty

    records: list[dict[str, Any]] = []
    for day, day_df in m5.groupby("session_day", sort=True):
        london = day_df.loc[(day_df["minute_of_day"] >= 7 * 60) & (day_df["minute_of_day"] < 12 * 60)].copy()
        ny_open = day_df.loc[(day_df["minute_of_day"] >= 12 * 60) & (day_df["minute_of_day"] < 12 * 60 + int(ny_window_bars) * 5)].copy()
        if london.empty or ny_open.empty:
            continue

        london_open = float(london.iloc[0]["open"])
        london_close = float(london.iloc[-1]["close"])
        london_high = float(london["high"].max())
        london_low = float(london["low"].min())
        london_range_pips = (london_high - london_low) / PIP_SIZE
        if london_range_pips < float(min_london_range_pips):
            continue

        london_net_pips = (london_close - london_open) / PIP_SIZE
        if abs(london_net_pips) <= 0:
            continue
        london_direction = 1 if london_net_pips > 0 else -1
        london_onesidedness = abs(london_net_pips) / max(london_range_pips, 1e-9)
        if london_onesidedness < float(min_london_onesidedness):
            continue

        if london_direction > 0:
            distance_to_extreme = (london_high - london_close) / PIP_SIZE
        else:
            distance_to_extreme = (london_close - london_low) / PIP_SIZE
        if distance_to_extreme > float(max_distance_to_extreme_pips):
            continue

        for ny_idx, row in enumerate(ny_open.itertuples(index=False), start=1):
            close = float(row.close)
            open_ = float(row.open)
            if float(row.bar_body_ratio) < float(reversal_body_ratio):
                continue
            if london_direction > 0:
                if close < open_:
                    records.append(
                        {
                            "time": row.time,
                            "signal_direction": -1,
                            "trigger_family": SESSION_HANDOFF_FADE_FAMILY,
                            "london_direction": "up",
                            "london_range_pips": round(float(london_range_pips), 2),
                            "london_onesidedness": round(float(london_onesidedness), 3),
                            "distance_to_london_extreme_pips": round(float(distance_to_extreme), 2),
                            "ny_bar_index": int(ny_idx),
                            "session_label": str(row.session_label),
                        }
                    )
                    break
            else:
                if close > open_:
                    records.append(
                        {
                            "time": row.time,
                            "signal_direction": 1,
                            "trigger_family": SESSION_HANDOFF_FADE_FAMILY,
                            "london_direction": "down",
                            "london_range_pips": round(float(london_range_pips), 2),
                            "london_onesidedness": round(float(london_onesidedness), 3),
                            "distance_to_london_extreme_pips": round(float(distance_to_extreme), 2),
                            "ny_bar_index": int(ny_idx),
                            "session_label": str(row.session_label),
                        }
                    )
                    break

    if not records:
        return empty
    return pd.DataFrame.from_records(records)


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


def run_dataset(
    dataset_path: Path,
    *,
    assumed_spread_pips: float,
    target_pips: float,
    stop_pips: float,
    horizon_bars: int,
    train_bars: int,
    test_bars: int,
    min_london_range_pips: float,
    min_london_onesidedness: float,
    max_distance_to_extreme_pips: float,
    ny_window_bars: int,
    reversal_body_ratio: float,
) -> dict[str, Any]:
    raw = cag._load_dataset(dataset_path, assumed_spread_pips=assumed_spread_pips)
    frame = attach_directional_outcomes(
        raw,
        target_pips=target_pips,
        stop_pips=stop_pips,
        horizon_bars=horizon_bars,
    )
    m5 = _prepare_m5_frame(raw)
    signals = scan_session_handoff_fade_signals(
        m5,
        min_london_range_pips=min_london_range_pips,
        min_london_onesidedness=min_london_onesidedness,
        max_distance_to_extreme_pips=max_distance_to_extreme_pips,
        ny_window_bars=ny_window_bars,
        reversal_body_ratio=reversal_body_ratio,
    )
    merged = frame.merge(signals, on="time", how="left")
    merged["signal_direction"] = pd.to_numeric(merged["signal_direction"], errors="coerce").fillna(0).astype(int)
    merged["trigger_family"] = merged["trigger_family"].fillna("none")
    merged["london_direction"] = merged["london_direction"].fillna("none")
    merged["session_label"] = merged["time"].map(cag._classify_session_label)

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
            "by_london_direction": breakdown_counts(merged, "london_direction"),
            "by_ny_bar_index": breakdown_counts(
                merged.assign(ny_bar_bucket=merged["ny_bar_index"].fillna(-1).astype(int).astype(str)),
                "ny_bar_bucket",
            ),
        },
        "train": {"summary": asdict(summarize_signals(train_frame))},
        "test": {"summary": asdict(summarize_signals(test_frame))},
    }


def build_markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Session Handoff Fade Backtest",
        "",
        "Research-only test of the `session_handoff_fade` family on historical USDJPY M1 data.",
        "",
        "## Config",
        "",
        f"- target pips: `{payload['config']['target_pips']}`",
        f"- stop pips: `{payload['config']['stop_pips']}`",
        f"- horizon bars: `{payload['config']['horizon_bars']}`",
        f"- train bars: `{payload['config']['train_bars']}`",
        f"- test bars: `{payload['config']['test_bars']}`",
        f"- min London range: `{payload['config']['min_london_range_pips']}` pips",
        f"- min London onesidedness: `{payload['config']['min_london_onesidedness']}`",
        f"- max distance to London extreme: `{payload['config']['max_distance_to_extreme_pips']}` pips",
        f"- NY reversal window: first `{payload['config']['ny_window_bars']}` M5 bars",
        f"- reversal body ratio: `{payload['config']['reversal_body_ratio']}`",
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
        lines.extend(["", "### London Direction Breakdown", ""])
        for row in full["by_london_direction"]:
            lines.append(
                f"- `{row['london_direction']}`: trades `{row['trades']}` | WR `{row['win_rate_pct']}%` | avg `{row['avg_pips']}p` | net `{row['net_pips']}p` | PF `{row['profit_factor']}`"
            )
        lines.extend(["", "### NY Reversal Bar Breakdown", ""])
        for row in full["by_ny_bar_index"]:
            lines.append(
                f"- `{row['ny_bar_bucket']}`: trades `{row['trades']}` | WR `{row['win_rate_pct']}%` | avg `{row['avg_pips']}p` | net `{row['net_pips']}p` | PF `{row['profit_factor']}`"
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
            "min_london_range_pips": float(args.min_london_range_pips),
            "min_london_onesidedness": float(args.min_london_onesidedness),
            "max_distance_to_extreme_pips": float(args.max_distance_to_extreme_pips),
            "ny_window_bars": int(args.ny_window_bars),
            "reversal_body_ratio": float(args.reversal_body_ratio),
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
            min_london_range_pips=float(args.min_london_range_pips),
            min_london_onesidedness=float(args.min_london_onesidedness),
            max_distance_to_extreme_pips=float(args.max_distance_to_extreme_pips),
            ny_window_bars=int(args.ny_window_bars),
            reversal_body_ratio=float(args.reversal_body_ratio),
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
