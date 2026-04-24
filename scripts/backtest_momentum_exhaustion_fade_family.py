#!/usr/bin/env python3
"""
Research backtest for the momentum exhaustion fade family.

Hypothesis:
  - Some intraday USDJPY pushes travel far enough to become extended, then begin
    to lose momentum before actually reversing.
  - A usable fade requires three things:
      1. a real directional move,
      2. visible deceleration / exhaustion near the move extreme,
      3. an actual reversal close before entry.

This is research-only and intentionally separate from the live autonomous gate.
It uses the same fast first-touch forward outcome proxy as the other family
research tools:
  - enter on the reversal-confirmation bar close
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
DEFAULT_JSON = OUT_DIR / "momentum_exhaustion_fade_backtest.json"
DEFAULT_MD = OUT_DIR / "momentum_exhaustion_fade_backtest.md"
PIP_SIZE = cag.PIP_SIZE
MOMENTUM_EXHAUSTION_FADE_FAMILY = "momentum_exhaustion_fade_v1"


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
    p = argparse.ArgumentParser(description="Backtest the momentum exhaustion fade family on historical USDJPY M1 data")
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
    p.add_argument("--move-window-bars", type=int, default=8, help="M5 bars used to define the prior directional move")
    p.add_argument("--min-move-pips", type=float, default=18.0, help="Minimum prior directional move in pips")
    p.add_argument(
        "--min-directional-consistency",
        type=float,
        default=0.62,
        help="Minimum fraction of prior closes that moved in the same direction",
    )
    p.add_argument(
        "--decel-bars",
        type=int,
        default=3,
        help="Trailing and leading subwindows used to test body-size deceleration",
    )
    p.add_argument(
        "--max-late-body-ratio",
        type=float,
        default=0.80,
        help="Late-window avg body must be <= early-window avg body times this ratio",
    )
    p.add_argument(
        "--min-exhaustion-wick-ratio",
        type=float,
        default=0.35,
        help="Minimum move-side wick/range ratio in the late exhaustion window",
    )
    p.add_argument(
        "--max-distance-to-window-extreme-pips",
        type=float,
        default=2.0,
        help="Last pre-confirmation close must still be close to the move extreme",
    )
    p.add_argument(
        "--confirmation-body-ratio",
        type=float,
        default=0.45,
        help="Minimum body/range ratio on the reversal confirmation bar",
    )
    p.add_argument(
        "--allowed-sessions",
        nargs="*",
        default=["tokyo", "london", "ny", "london/ny"],
        help="Session labels allowed to emit signals.",
    )
    p.add_argument("--cooldown-bars", type=int, default=6, help="Skip new entries for this many M5 bars after a signal")
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


def _prepare_m5_frame(m1_frame: pd.DataFrame) -> pd.DataFrame:
    work = m1_frame[["time", "open", "high", "low", "close"]].copy()
    m5 = cag._resample_ohlc(work, "5min")
    m5["session_label"] = m5["time"].map(cag._classify_session_label)
    m5["body_pips"] = ((m5["close"] - m5["open"]).abs() / PIP_SIZE).astype(float)
    m5["body_ratio"] = [
        cag._body_ratio(o, h, l, c) for o, h, l, c in zip(m5["open"], m5["high"], m5["low"], m5["close"])
    ]
    rng = (m5["high"] - m5["low"]).clip(lower=1e-9)
    m5["upper_wick_ratio"] = ((m5["high"] - m5[["open", "close"]].max(axis=1)) / rng).astype(float)
    m5["lower_wick_ratio"] = ((m5[["open", "close"]].min(axis=1) - m5["low"]) / rng).astype(float)
    return m5.reset_index(drop=True)


def scan_momentum_exhaustion_fade_signals(
    m5: pd.DataFrame,
    *,
    move_window_bars: int,
    min_move_pips: float,
    min_directional_consistency: float,
    decel_bars: int,
    max_late_body_ratio: float,
    min_exhaustion_wick_ratio: float,
    max_distance_to_window_extreme_pips: float,
    confirmation_body_ratio: float,
    allowed_sessions: set[str],
    cooldown_bars: int,
) -> pd.DataFrame:
    empty = pd.DataFrame(
        columns=[
            "time",
            "signal_direction",
            "trigger_family",
            "move_direction",
            "move_window_bars",
            "move_pips",
            "directional_consistency",
            "early_avg_body_pips",
            "late_avg_body_pips",
            "exhaustion_wick_ratio",
            "confirmation_body_ratio",
            "session_label",
        ]
    )
    if m5.empty or len(m5) < max(int(move_window_bars), int(decel_bars) * 2) + 1:
        return empty

    records: list[dict[str, Any]] = []
    next_allowed_idx = 0
    closes = m5["close"].to_numpy(dtype=float)
    opens = m5["open"].to_numpy(dtype=float)
    highs = m5["high"].to_numpy(dtype=float)
    lows = m5["low"].to_numpy(dtype=float)
    body_pips = m5["body_pips"].to_numpy(dtype=float)
    body_ratio = m5["body_ratio"].to_numpy(dtype=float)
    upper_wick_ratio = m5["upper_wick_ratio"].to_numpy(dtype=float)
    lower_wick_ratio = m5["lower_wick_ratio"].to_numpy(dtype=float)

    for i in range(int(move_window_bars), len(m5)):
        if i < next_allowed_idx:
            continue
        if str(m5.iloc[i]["session_label"]) not in allowed_sessions:
            continue

        start_idx = i - int(move_window_bars)
        trend_slice = slice(start_idx, i)
        if (i - start_idx) < max(int(move_window_bars), int(decel_bars) * 2):
            continue

        net_move_pips = (closes[i - 1] - opens[start_idx]) / PIP_SIZE
        abs_move_pips = abs(float(net_move_pips))
        if abs_move_pips < float(min_move_pips):
            continue

        move_direction = 1 if net_move_pips > 0 else -1
        deltas = np.diff(closes[start_idx:i]) / PIP_SIZE
        directional_hits = np.sum(deltas * move_direction > 0)
        consistency = float(directional_hits) / max(len(deltas), 1)
        if consistency < float(min_directional_consistency):
            continue

        early_bodies = body_pips[start_idx : start_idx + int(decel_bars)]
        late_bodies = body_pips[i - int(decel_bars) : i]
        early_avg_body = float(np.mean(early_bodies))
        late_avg_body = float(np.mean(late_bodies))
        if early_avg_body <= 0:
            continue
        if late_avg_body > early_avg_body * float(max_late_body_ratio):
            continue

        if move_direction > 0:
            exhaustion_wick = float(np.max(upper_wick_ratio[i - int(decel_bars) : i]))
            distance_to_extreme = float((np.max(highs[start_idx:i]) - closes[i - 1]) / PIP_SIZE)
            confirm_ok = closes[i] < opens[i] and body_ratio[i] >= float(confirmation_body_ratio)
            signal_direction = -1
            move_dir_label = "up"
        else:
            exhaustion_wick = float(np.max(lower_wick_ratio[i - int(decel_bars) : i]))
            distance_to_extreme = float((closes[i - 1] - np.min(lows[start_idx:i])) / PIP_SIZE)
            confirm_ok = closes[i] > opens[i] and body_ratio[i] >= float(confirmation_body_ratio)
            signal_direction = 1
            move_dir_label = "down"

        if exhaustion_wick < float(min_exhaustion_wick_ratio):
            continue
        if distance_to_extreme > float(max_distance_to_window_extreme_pips):
            continue
        if not confirm_ok:
            continue

        records.append(
            {
                "time": m5.iloc[i]["time"],
                "signal_direction": int(signal_direction),
                "trigger_family": MOMENTUM_EXHAUSTION_FADE_FAMILY,
                "move_direction": move_dir_label,
                "move_window_bars": int(move_window_bars),
                "move_pips": round(abs_move_pips, 2),
                "directional_consistency": round(consistency, 3),
                "early_avg_body_pips": round(early_avg_body, 3),
                "late_avg_body_pips": round(late_avg_body, 3),
                "exhaustion_wick_ratio": round(exhaustion_wick, 3),
                "confirmation_body_ratio": round(float(body_ratio[i]), 3),
                "session_label": str(m5.iloc[i]["session_label"]),
            }
        )
        next_allowed_idx = i + int(cooldown_bars)

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


def _move_bucket(move_pips: float) -> str:
    if pd.isna(move_pips):
        return "unknown"
    if move_pips < 24.0:
        return "18-24"
    if move_pips < 30.0:
        return "24-30"
    return "30+"


def run_dataset(
    dataset_path: Path,
    *,
    assumed_spread_pips: float,
    target_pips: float,
    stop_pips: float,
    horizon_bars: int,
    train_bars: int,
    test_bars: int,
    move_window_bars: int,
    min_move_pips: float,
    min_directional_consistency: float,
    decel_bars: int,
    max_late_body_ratio: float,
    min_exhaustion_wick_ratio: float,
    max_distance_to_window_extreme_pips: float,
    confirmation_body_ratio: float,
    allowed_sessions: set[str],
    cooldown_bars: int,
) -> dict[str, Any]:
    raw = cag._load_dataset(dataset_path, assumed_spread_pips=assumed_spread_pips)
    frame = attach_directional_outcomes(
        raw,
        target_pips=target_pips,
        stop_pips=stop_pips,
        horizon_bars=horizon_bars,
    )
    m5 = _prepare_m5_frame(raw)
    signals = scan_momentum_exhaustion_fade_signals(
        m5,
        move_window_bars=move_window_bars,
        min_move_pips=min_move_pips,
        min_directional_consistency=min_directional_consistency,
        decel_bars=decel_bars,
        max_late_body_ratio=max_late_body_ratio,
        min_exhaustion_wick_ratio=min_exhaustion_wick_ratio,
        max_distance_to_window_extreme_pips=max_distance_to_window_extreme_pips,
        confirmation_body_ratio=confirmation_body_ratio,
        allowed_sessions=allowed_sessions,
        cooldown_bars=cooldown_bars,
    )
    merged = frame.merge(signals, on="time", how="left")
    merged["signal_direction"] = pd.to_numeric(merged["signal_direction"], errors="coerce").fillna(0).astype(int)
    merged["trigger_family"] = merged["trigger_family"].fillna("none")
    merged["move_direction"] = merged["move_direction"].fillna("none")
    merged["session_label"] = merged["time"].map(cag._classify_session_label)
    merged["move_bucket"] = merged["move_pips"].map(_move_bucket)

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
            "by_move_direction": breakdown_counts(merged, "move_direction"),
            "by_move_bucket": breakdown_counts(merged, "move_bucket"),
        },
        "train": {"summary": asdict(summarize_signals(train_frame))},
        "test": {"summary": asdict(summarize_signals(test_frame))},
    }


def build_markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Momentum Exhaustion Fade Backtest",
        "",
        "Research-only test of the `momentum_exhaustion_fade` family on historical USDJPY M1 data.",
        "",
        "## Config",
        "",
        f"- target pips: `{payload['config']['target_pips']}`",
        f"- stop pips: `{payload['config']['stop_pips']}`",
        f"- horizon bars: `{payload['config']['horizon_bars']}`",
        f"- train bars: `{payload['config']['train_bars']}`",
        f"- test bars: `{payload['config']['test_bars']}`",
        f"- move window bars: `{payload['config']['move_window_bars']}`",
        f"- min move pips: `{payload['config']['min_move_pips']}`",
        f"- min directional consistency: `{payload['config']['min_directional_consistency']}`",
        f"- deceleration bars: `{payload['config']['decel_bars']}`",
        f"- max late-body ratio: `{payload['config']['max_late_body_ratio']}`",
        f"- min exhaustion wick ratio: `{payload['config']['min_exhaustion_wick_ratio']}`",
        f"- max distance to window extreme: `{payload['config']['max_distance_to_window_extreme_pips']}` pips",
        f"- confirmation body ratio: `{payload['config']['confirmation_body_ratio']}`",
        f"- allowed sessions: `{', '.join(payload['config']['allowed_sessions'])}`",
        f"- cooldown bars: `{payload['config']['cooldown_bars']}`",
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
        lines.extend(["", "### Prior Move Direction Breakdown", ""])
        for row in full["by_move_direction"]:
            lines.append(
                f"- `{row['move_direction']}`: trades `{row['trades']}` | WR `{row['win_rate_pct']}%` | avg `{row['avg_pips']}p` | net `{row['net_pips']}p` | PF `{row['profit_factor']}`"
            )
        lines.extend(["", "### Move Magnitude Breakdown", ""])
        for row in full["by_move_bucket"]:
            lines.append(
                f"- `{row['move_bucket']}`: trades `{row['trades']}` | WR `{row['win_rate_pct']}%` | avg `{row['avg_pips']}p` | net `{row['net_pips']}p` | PF `{row['profit_factor']}`"
            )
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def main() -> int:
    args = parse_args()
    dataset_paths = [Path(p).resolve() for p in args.datasets]
    if not dataset_paths:
        raise SystemExit("No datasets provided.")

    allowed_sessions = {str(s) for s in args.allowed_sessions}
    payload: dict[str, Any] = {
        "datasets_requested": [str(p) for p in dataset_paths],
        "config": {
            "target_pips": float(args.target_pips),
            "stop_pips": float(args.stop_pips),
            "horizon_bars": int(args.horizon_bars),
            "train_bars": int(args.train_bars),
            "test_bars": int(args.test_bars),
            "assumed_spread_pips": float(args.assume_spread_pips),
            "move_window_bars": int(args.move_window_bars),
            "min_move_pips": float(args.min_move_pips),
            "min_directional_consistency": float(args.min_directional_consistency),
            "decel_bars": int(args.decel_bars),
            "max_late_body_ratio": float(args.max_late_body_ratio),
            "min_exhaustion_wick_ratio": float(args.min_exhaustion_wick_ratio),
            "max_distance_to_window_extreme_pips": float(args.max_distance_to_window_extreme_pips),
            "confirmation_body_ratio": float(args.confirmation_body_ratio),
            "allowed_sessions": sorted(allowed_sessions),
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
            move_window_bars=int(args.move_window_bars),
            min_move_pips=float(args.min_move_pips),
            min_directional_consistency=float(args.min_directional_consistency),
            decel_bars=int(args.decel_bars),
            max_late_body_ratio=float(args.max_late_body_ratio),
            min_exhaustion_wick_ratio=float(args.min_exhaustion_wick_ratio),
            max_distance_to_window_extreme_pips=float(args.max_distance_to_window_extreme_pips),
            confirmation_body_ratio=float(args.confirmation_body_ratio),
            allowed_sessions=allowed_sessions,
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
