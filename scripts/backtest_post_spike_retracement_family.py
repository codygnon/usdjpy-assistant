#!/usr/bin/env python3
"""
Research backtest for the post-spike retracement family.

Hypothesis:
  - Sharp multi-bar M1 spikes create a short-lived liquidity vacuum.
  - When the impulse stalls and the first opposite close prints, a partial
    retracement often follows before the original move either resumes or dies.

This is research-only and intentionally separate from the live autonomous gate.
It uses the same fast first-touch forward outcome proxy as the other gate-family
research tools:
  - enter on the confirmation bar close
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
DEFAULT_JSON = OUT_DIR / "post_spike_retracement_backtest.json"
DEFAULT_MD = OUT_DIR / "post_spike_retracement_backtest.md"
PIP_SIZE = cag.PIP_SIZE
POST_SPIKE_RETRACEMENT_FAMILY = "post_spike_retracement_v1"


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
    p = argparse.ArgumentParser(description="Backtest the post-spike retracement family on historical USDJPY M1 data")
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
    p.add_argument("--spike-window-bars", type=int, default=5, help="Lookback window for the M1 spike leg")
    p.add_argument("--min-spike-pips", type=float, default=12.0, help="Minimum net move across the spike window")
    p.add_argument(
        "--min-directional-consistency",
        type=float,
        default=0.60,
        help="Minimum fraction of closes moving in spike direction inside the spike window",
    )
    p.add_argument(
        "--max-confirmation-bars",
        type=int,
        default=3,
        help="Number of bars after the spike allowed for stall + opposite-close confirmation",
    )
    p.add_argument(
        "--stall-body-fraction",
        type=float,
        default=0.60,
        help="At least one post-spike bar must have body <= this fraction of average spike-leg body",
    )
    p.add_argument(
        "--max-extension-after-spike-pips",
        type=float,
        default=4.0,
        help="Invalidate if price keeps extending this far past the spike close before confirmation",
    )
    p.add_argument(
        "--cooldown-bars",
        type=int,
        default=10,
        help="Skip new spike entries for this many M1 bars after a signal fires",
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


def _prepare_m1_frame(m1_frame: pd.DataFrame) -> pd.DataFrame:
    work = m1_frame[["time", "open", "high", "low", "close"]].copy()
    work["session_label"] = work["time"].map(cag._classify_session_label)
    work["body_pips"] = ((work["close"] - work["open"]).abs() / PIP_SIZE).astype(float)
    work["delta_pips"] = work["close"].diff().fillna(0.0) / PIP_SIZE
    return work.reset_index(drop=True)


def scan_post_spike_retracement_signals(
    m1: pd.DataFrame,
    *,
    spike_window_bars: int,
    min_spike_pips: float,
    min_directional_consistency: float,
    max_confirmation_bars: int,
    stall_body_fraction: float,
    max_extension_after_spike_pips: float,
    cooldown_bars: int,
) -> pd.DataFrame:
    if m1.empty or len(m1) < int(spike_window_bars) + 2:
        return pd.DataFrame(
            columns=[
                "time",
                "signal_direction",
                "trigger_family",
                "spike_direction",
                "spike_window_bars",
                "spike_move_pips",
                "directional_consistency",
                "confirmation_bars",
                "session_label",
            ]
        )

    records: list[dict[str, Any]] = []
    next_allowed_idx = 0
    closes = m1["close"].to_numpy(dtype=float)
    opens = m1["open"].to_numpy(dtype=float)
    highs = m1["high"].to_numpy(dtype=float)
    lows = m1["low"].to_numpy(dtype=float)
    body_pips = m1["body_pips"].to_numpy(dtype=float)

    for i in range(int(spike_window_bars), len(m1) - 1):
        if i < next_allowed_idx:
            continue

        start_idx = i - int(spike_window_bars)
        net_move_pips = (closes[i] - closes[start_idx]) / PIP_SIZE
        abs_move_pips = abs(float(net_move_pips))
        if abs_move_pips < float(min_spike_pips):
            continue

        spike_direction = 1 if net_move_pips > 0 else -1
        deltas = np.diff(closes[start_idx : i + 1]) / PIP_SIZE
        directional_hits = np.sum(deltas * spike_direction > 0)
        consistency = float(directional_hits) / max(len(deltas), 1)
        if consistency < float(min_directional_consistency):
            continue

        avg_spike_body = float(np.mean(body_pips[start_idx + 1 : i + 1]))
        spike_close = float(closes[i])
        saw_stall = False
        signal_idx: int | None = None

        for j in range(i + 1, min(i + 1 + int(max_confirmation_bars), len(m1))):
            if spike_direction > 0:
                if ((highs[j] - spike_close) / PIP_SIZE) > float(max_extension_after_spike_pips):
                    break
            else:
                if ((spike_close - lows[j]) / PIP_SIZE) > float(max_extension_after_spike_pips):
                    break

            if body_pips[j] <= avg_spike_body * float(stall_body_fraction):
                saw_stall = True

            confirm_direction = np.sign(closes[j] - opens[j])
            if confirm_direction == 0 or confirm_direction == spike_direction:
                continue
            if not saw_stall:
                continue

            signal_idx = j
            break

        if signal_idx is None:
            continue

        signal_direction = -spike_direction
        records.append(
            {
                "time": m1.iloc[signal_idx]["time"],
                "signal_direction": int(signal_direction),
                "trigger_family": POST_SPIKE_RETRACEMENT_FAMILY,
                "spike_direction": "up" if spike_direction > 0 else "down",
                "spike_window_bars": int(spike_window_bars),
                "spike_move_pips": round(abs_move_pips, 2),
                "directional_consistency": round(consistency, 3),
                "confirmation_bars": int(signal_idx - i),
                "session_label": str(m1.iloc[signal_idx]["session_label"]),
            }
        )
        next_allowed_idx = signal_idx + int(cooldown_bars)

    if not records:
        return pd.DataFrame(
            columns=[
                "time",
                "signal_direction",
                "trigger_family",
                "spike_direction",
                "spike_window_bars",
                "spike_move_pips",
                "directional_consistency",
                "confirmation_bars",
                "session_label",
            ]
        )
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


def _spike_bucket(move_pips: float) -> str:
    if pd.isna(move_pips):
        return "unknown"
    if move_pips < 16.0:
        return "12-16"
    if move_pips < 20.0:
        return "16-20"
    return "20+"


def run_dataset(
    dataset_path: Path,
    *,
    assumed_spread_pips: float,
    target_pips: float,
    stop_pips: float,
    horizon_bars: int,
    train_bars: int,
    test_bars: int,
    spike_window_bars: int,
    min_spike_pips: float,
    min_directional_consistency: float,
    max_confirmation_bars: int,
    stall_body_fraction: float,
    max_extension_after_spike_pips: float,
    cooldown_bars: int,
) -> dict[str, Any]:
    raw = cag._load_dataset(dataset_path, assumed_spread_pips=assumed_spread_pips)
    frame = attach_directional_outcomes(
        raw,
        target_pips=target_pips,
        stop_pips=stop_pips,
        horizon_bars=horizon_bars,
    )
    m1 = _prepare_m1_frame(raw)
    signals = scan_post_spike_retracement_signals(
        m1,
        spike_window_bars=spike_window_bars,
        min_spike_pips=min_spike_pips,
        min_directional_consistency=min_directional_consistency,
        max_confirmation_bars=max_confirmation_bars,
        stall_body_fraction=stall_body_fraction,
        max_extension_after_spike_pips=max_extension_after_spike_pips,
        cooldown_bars=cooldown_bars,
    )
    merged = frame.merge(signals, on="time", how="left")
    merged["signal_direction"] = pd.to_numeric(merged["signal_direction"], errors="coerce").fillna(0).astype(int)
    merged["trigger_family"] = merged["trigger_family"].fillna("none")
    merged["spike_direction"] = merged["spike_direction"].fillna("none")
    merged["session_label"] = merged["time"].map(cag._classify_session_label)
    merged["spike_bucket"] = merged["spike_move_pips"].map(_spike_bucket)

    train_frame, test_frame = cag.split_frame_train_test(merged, train_bars=train_bars, test_bars=test_bars)
    return {
        "dataset_meta": {
            "bars": int(len(merged)),
            "signal_rows": int((merged["signal_direction"] != 0).sum()),
            "spread_source": str(raw["spread_source"].iloc[0]) if len(raw) else "unknown",
        },
        "full": {
            "summary": asdict(summarize_signals(merged)),
            "by_session": breakdown_counts(merged, "session_label"),
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
            "by_spike_direction": breakdown_counts(merged, "spike_direction"),
            "by_spike_bucket": breakdown_counts(merged, "spike_bucket"),
            "by_confirmation_bars": breakdown_counts(
                merged.assign(confirm_bucket=merged["confirmation_bars"].fillna(-1).astype(int).astype(str)),
                "confirm_bucket",
            ),
        },
        "train": {"summary": asdict(summarize_signals(train_frame))},
        "test": {"summary": asdict(summarize_signals(test_frame))},
    }


def build_markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Post Spike Retracement Backtest",
        "",
        "Research-only test of the `post_spike_retracement` family on historical USDJPY M1 data.",
        "",
        "## Config",
        "",
        f"- target pips: `{payload['config']['target_pips']}`",
        f"- stop pips: `{payload['config']['stop_pips']}`",
        f"- horizon bars: `{payload['config']['horizon_bars']}`",
        f"- train bars: `{payload['config']['train_bars']}`",
        f"- test bars: `{payload['config']['test_bars']}`",
        f"- spike window bars: `{payload['config']['spike_window_bars']}`",
        f"- min spike pips: `{payload['config']['min_spike_pips']}`",
        f"- min directional consistency: `{payload['config']['min_directional_consistency']}`",
        f"- max confirmation bars: `{payload['config']['max_confirmation_bars']}`",
        f"- stall body fraction: `{payload['config']['stall_body_fraction']}`",
        f"- max extension after spike: `{payload['config']['max_extension_after_spike_pips']}` pips",
        f"- cooldown bars: `{payload['config']['cooldown_bars']}`",
        "",
        "_Note: this research version does not use an external event calendar filter; it is pure price-action detection._",
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
                f"- bars: `{meta['bars']}` | signal rows: `{meta['signal_rows']}` | spread source: `{meta['spread_source']}`",
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
        lines.extend(["", "### Spike Direction Breakdown", ""])
        for row in full["by_spike_direction"]:
            lines.append(
                f"- `{row['spike_direction']}`: trades `{row['trades']}` | WR `{row['win_rate_pct']}%` | avg `{row['avg_pips']}p` | net `{row['net_pips']}p` | PF `{row['profit_factor']}`"
            )
        lines.extend(["", "### Spike Magnitude Breakdown", ""])
        for row in full["by_spike_bucket"]:
            lines.append(
                f"- `{row['spike_bucket']}`: trades `{row['trades']}` | WR `{row['win_rate_pct']}%` | avg `{row['avg_pips']}p` | net `{row['net_pips']}p` | PF `{row['profit_factor']}`"
            )
        lines.extend(["", "### Confirmation Bars Breakdown", ""])
        for row in full["by_confirmation_bars"]:
            lines.append(
                f"- `{row['confirm_bucket']}`: trades `{row['trades']}` | WR `{row['win_rate_pct']}%` | avg `{row['avg_pips']}p` | net `{row['net_pips']}p` | PF `{row['profit_factor']}`"
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
            "spike_window_bars": int(args.spike_window_bars),
            "min_spike_pips": float(args.min_spike_pips),
            "min_directional_consistency": float(args.min_directional_consistency),
            "max_confirmation_bars": int(args.max_confirmation_bars),
            "stall_body_fraction": float(args.stall_body_fraction),
            "max_extension_after_spike_pips": float(args.max_extension_after_spike_pips),
            "cooldown_bars": int(args.cooldown_bars),
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
            spike_window_bars=int(args.spike_window_bars),
            min_spike_pips=float(args.min_spike_pips),
            min_directional_consistency=float(args.min_directional_consistency),
            max_confirmation_bars=int(args.max_confirmation_bars),
            stall_body_fraction=float(args.stall_body_fraction),
            max_extension_after_spike_pips=float(args.max_extension_after_spike_pips),
            cooldown_bars=int(args.cooldown_bars),
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
