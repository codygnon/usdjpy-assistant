#!/usr/bin/env python3
"""
Research backtest for the failed-breakout reversal / recapture family.

This is intentionally separate from the live autonomous gate and the main
threshold calibration harness. The goal is to answer a narrower question:

Does a simple trapped-trader failed-breakout family around intra-session highs
and lows show enough edge on historical USDJPY data to justify deeper work?

The backtest uses the same fast first-touch forward outcome proxy as the
autonomous gate calibration:
  - enter on the recapture bar close
  - win if +target_pips hits before -stop_pips within horizon_bars
"""
from __future__ import annotations

import argparse
import json
import math
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
DEFAULT_JSON = OUT_DIR / "failed_breakout_family_backtest.json"
DEFAULT_MD = OUT_DIR / "failed_breakout_family_backtest.md"
PIP_SIZE = cag.PIP_SIZE
VALID_SESSIONS = ("london", "london/ny", "ny")


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
    p = argparse.ArgumentParser(description="Backtest the failed-breakout reversal / recapture family on USDJPY M1 data")
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
    p.add_argument("--min-break-pips", type=float, default=2.0, help="Minimum breakout excursion beyond the session level")
    p.add_argument("--max-break-pips", type=float, default=5.0, help="Maximum breakout excursion beyond the session level")
    p.add_argument("--max-hold-bars", type=int, default=3, help="Recapture must happen within this many M5 bars after breakout")
    p.add_argument(
        "--min-session-bars",
        type=int,
        default=6,
        help="Minimum completed M5 bars inside the session block before breakout signals are allowed",
    )
    p.add_argument(
        "--recapture-body-ratio",
        type=float,
        default=0.5,
        help="Minimum body/range ratio on the recapture bar",
    )
    p.add_argument(
        "--continuation-invalidation-pips",
        type=float,
        default=8.0,
        help="Invalidate the setup if continuation extends this far beyond the broken level before recapture",
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


def _body_ratio(open_: float, high: float, low: float, close: float) -> float:
    return cag._body_ratio(open_, high, low, close)


def _prepare_m5_frame(m1_frame: pd.DataFrame) -> pd.DataFrame:
    return cag._prepare_failed_breakout_m5_frame(m1_frame)


def scan_failed_breakout_signals(
    m5: pd.DataFrame,
    *,
    min_break_pips: float,
    max_break_pips: float,
    max_hold_bars: int,
    min_session_bars: int,
    recapture_body_ratio: float,
    continuation_invalidation_pips: float,
) -> pd.DataFrame:
    raw = cag.scan_failed_breakout_reversal_signals(
        m5,
        min_break_pips=min_break_pips,
        max_break_pips=max_break_pips,
        max_hold_bars=max_hold_bars,
        min_session_bars=min_session_bars,
        recapture_body_ratio=recapture_body_ratio,
        continuation_invalidation_pips=continuation_invalidation_pips,
    )
    if raw.empty:
        return pd.DataFrame(
            columns=[
                "time",
                "signal_direction",
                "trigger_family",
                "breakout_side",
                "reference_level",
                "breakout_excursion_pips",
                "hold_bars",
                "session_label",
            ]
        )
    return raw.rename(
        columns={
            "failed_breakout_direction": "signal_direction",
            "failed_breakout_family": "trigger_family",
            "failed_breakout_side": "breakout_side",
            "failed_breakout_reference_level": "reference_level",
            "failed_breakout_excursion_pips": "breakout_excursion_pips",
            "failed_breakout_hold_bars": "hold_bars",
            "failed_breakout_session_label": "session_label",
        }
    )


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
        return FamilySummary(
            trades=0,
            win_rate_pct=0.0,
            avg_pips=0.0,
            avg_final_pips=0.0,
            avg_mfe_pips=0.0,
            avg_mae_pips=0.0,
            net_pips=0.0,
            profit_factor=None,
            tp_hit_rate_pct=0.0,
            stop_hit_rate_pct=0.0,
            timeout_rate_pct=0.0,
        )

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
        summary = summarize_signals(subset)
        rows.append(
            {
                column: str(value),
                **asdict(summary),
            }
        )
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
    min_break_pips: float,
    max_break_pips: float,
    max_hold_bars: int,
    min_session_bars: int,
    recapture_body_ratio: float,
    continuation_invalidation_pips: float,
) -> dict[str, Any]:
    raw = cag._load_dataset(dataset_path, assumed_spread_pips=assumed_spread_pips)
    frame = attach_directional_outcomes(
        raw,
        target_pips=target_pips,
        stop_pips=stop_pips,
        horizon_bars=horizon_bars,
    )
    m5 = _prepare_m5_frame(raw)
    signals = scan_failed_breakout_signals(
        m5,
        min_break_pips=min_break_pips,
        max_break_pips=max_break_pips,
        max_hold_bars=max_hold_bars,
        min_session_bars=min_session_bars,
        recapture_body_ratio=recapture_body_ratio,
        continuation_invalidation_pips=continuation_invalidation_pips,
    )
    merged = frame.merge(
        signals,
        on="time",
        how="left",
    )
    merged["signal_direction"] = pd.to_numeric(merged["signal_direction"], errors="coerce").fillna(0).astype(int)
    merged["trigger_family"] = merged["trigger_family"].fillna("none")
    merged["breakout_side"] = merged["breakout_side"].fillna("none")
    merged["hold_bars"] = pd.to_numeric(merged["hold_bars"], errors="coerce")
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
            "by_session": breakdown_counts(merged, "session_label"),
            "by_direction": breakdown_counts(merged.assign(direction_label=np.where(merged["signal_direction"] > 0, "buy", np.where(merged["signal_direction"] < 0, "sell", "none"))), "direction_label"),
            "by_breakout_side": breakdown_counts(merged, "breakout_side"),
            "by_hold_bars": breakdown_counts(merged.assign(hold_bucket=merged["hold_bars"].fillna(-1).astype(int).astype(str)), "hold_bucket"),
        },
        "train": {"summary": asdict(summarize_signals(train_frame))},
        "test": {"summary": asdict(summarize_signals(test_frame))},
    }


def build_markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Failed Breakout Family Backtest",
        "",
        "Research-only test of the `failed_breakout_reversal / failed_breakout_recapture` family on historical USDJPY M1 data.",
        "",
        "## Config",
        "",
        f"- target pips: `{payload['config']['target_pips']}`",
        f"- stop pips: `{payload['config']['stop_pips']}`",
        f"- horizon bars: `{payload['config']['horizon_bars']}`",
        f"- train bars: `{payload['config']['train_bars']}`",
        f"- test bars: `{payload['config']['test_bars']}`",
        f"- breakout excursion: `{payload['config']['min_break_pips']}` to `{payload['config']['max_break_pips']}` pips",
        f"- max hold bars: `{payload['config']['max_hold_bars']}` M5 bars",
        f"- min session bars before eligible: `{payload['config']['min_session_bars']}` M5 bars",
        f"- recapture body ratio: `{payload['config']['recapture_body_ratio']}`",
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
                "### Session Breakdown",
                "",
            ]
        )
        for row in full["by_session"]:
            lines.append(
                f"- `{row['session_label']}`: trades `{row['trades']}` | WR `{row['win_rate_pct']}%` | avg `{row['avg_pips']}p` | net `{row['net_pips']}p` | PF `{row['profit_factor']}`"
            )
        lines.extend(["", "### Direction Breakdown", ""])
        for row in full["by_direction"]:
            lines.append(
                f"- `{row['direction_label']}`: trades `{row['trades']}` | WR `{row['win_rate_pct']}%` | avg `{row['avg_pips']}p` | net `{row['net_pips']}p` | PF `{row['profit_factor']}`"
            )
        lines.extend(["", "### Breakout Side Breakdown", ""])
        for row in full["by_breakout_side"]:
            lines.append(
                f"- `{row['breakout_side']}`: trades `{row['trades']}` | WR `{row['win_rate_pct']}%` | avg `{row['avg_pips']}p` | net `{row['net_pips']}p` | PF `{row['profit_factor']}`"
            )
        lines.extend(["", "### Hold-Bars Breakdown", ""])
        for row in full["by_hold_bars"]:
            lines.append(
                f"- `{row['hold_bucket']}`: trades `{row['trades']}` | WR `{row['win_rate_pct']}%` | avg `{row['avg_pips']}p` | net `{row['net_pips']}p` | PF `{row['profit_factor']}`"
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
            "min_break_pips": float(args.min_break_pips),
            "max_break_pips": float(args.max_break_pips),
            "max_hold_bars": int(args.max_hold_bars),
            "min_session_bars": int(args.min_session_bars),
            "recapture_body_ratio": float(args.recapture_body_ratio),
            "continuation_invalidation_pips": float(args.continuation_invalidation_pips),
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
            min_break_pips=float(args.min_break_pips),
            max_break_pips=float(args.max_break_pips),
            max_hold_bars=int(args.max_hold_bars),
            min_session_bars=int(args.min_session_bars),
            recapture_body_ratio=float(args.recapture_body_ratio),
            continuation_invalidation_pips=float(args.continuation_invalidation_pips),
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
