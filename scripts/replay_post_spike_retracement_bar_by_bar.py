#!/usr/bin/env python3
"""
Bar-by-bar replay for the narrowed post-spike retracement research family.

This uses the strongest slice from the fast backtest:
  - session in {"ny", "london/ny"}
  - confirmation_bars in {2, 3}

It is still a research replay, not a full broker/margin/portfolio simulator.
Each signal is evaluated independently.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import backtest_post_spike_retracement_family as psr
from scripts import calibrate_autonomous_gate as cag

OUT_DIR = ROOT / "research_out"
DEFAULT_SOURCE = OUT_DIR / "USDJPY_M1_OANDA_250k.csv"
DEFAULT_JSON = OUT_DIR / "post_spike_retracement_bar_by_bar_250k.json"
DEFAULT_MD = OUT_DIR / "post_spike_retracement_bar_by_bar_250k.md"
DEFAULT_CSV = OUT_DIR / "post_spike_retracement_bar_by_bar_250k_trades.csv"
VALID_SESSIONS = {"ny", "london/ny"}
VALID_CONFIRMATION_BARS = {2, 3}


@dataclass
class ReplayTrade:
    signal_time: str
    session_label: str
    spike_direction: str
    spike_move_pips: float
    confirmation_bars: int
    side: str
    entry_price: float
    entry_bar_index: int
    exit_time: str
    exit_bar_index: int
    exit_price: float
    outcome_code: str
    pnl_pips: float
    mfe_pips: float
    mae_pips: float
    bars_held: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a bar-by-bar replay of post_spike_retracement_ny_overlap_v1 on a 250k USDJPY slice")
    p.add_argument("--source-dataset", default=str(DEFAULT_SOURCE), help="Source M1 CSV dataset")
    p.add_argument("--bars", type=int, default=250000, help="Use the most recent N bars from the source dataset")
    p.add_argument("--target-pips", type=float, default=6.0)
    p.add_argument("--stop-pips", type=float, default=10.0)
    p.add_argument("--horizon-bars", type=int, default=30)
    p.add_argument("--assume-spread-pips", type=float, default=1.2)
    p.add_argument("--output-json", default=str(DEFAULT_JSON))
    p.add_argument("--output-md", default=str(DEFAULT_MD))
    p.add_argument("--output-csv", default=str(DEFAULT_CSV))
    return p.parse_args()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return str(obj)


def load_recent_slice(path: Path, *, bars: int, assume_spread_pips: float) -> pd.DataFrame:
    raw = cag._load_dataset(path, assumed_spread_pips=assume_spread_pips)
    if bars > 0 and len(raw) > bars:
        return raw.iloc[-bars:].reset_index(drop=True).copy()
    return raw.copy()


def detect_narrowed_signals(frame: pd.DataFrame) -> pd.DataFrame:
    m1 = psr._prepare_m1_frame(frame)
    signals = psr.scan_post_spike_retracement_signals(
        m1,
        spike_window_bars=5,
        min_spike_pips=12.0,
        min_directional_consistency=0.6,
        max_confirmation_bars=3,
        stall_body_fraction=0.6,
        max_extension_after_spike_pips=4.0,
        cooldown_bars=10,
    )
    if signals.empty:
        return signals
    return signals.loc[
        signals["session_label"].isin(VALID_SESSIONS)
        & signals["confirmation_bars"].isin(VALID_CONFIRMATION_BARS)
    ].reset_index(drop=True)


def replay_trades_bar_by_bar(
    frame: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    target_pips: float,
    stop_pips: float,
    horizon_bars: int,
) -> list[ReplayTrade]:
    if frame.empty or signals.empty:
        return []

    time_to_index = {ts: idx for idx, ts in enumerate(frame["time"])}
    target_px = float(target_pips) * cag.PIP_SIZE
    stop_px = float(stop_pips) * cag.PIP_SIZE
    trades: list[ReplayTrade] = []

    for sig in signals.itertuples(index=False):
        entry_idx = time_to_index.get(sig.time)
        if entry_idx is None or entry_idx >= len(frame) - 1:
            continue
        entry_row = frame.iloc[entry_idx]
        entry = float(entry_row["close"])
        direction = int(sig.signal_direction)
        side = "buy" if direction > 0 else "sell"
        tp = entry + target_px if direction > 0 else entry - target_px
        sl = entry - stop_px if direction > 0 else entry + stop_px

        mfe = 0.0
        mae = 0.0
        exit_idx = entry_idx
        exit_price = entry
        outcome = "timeout"

        for j in range(entry_idx + 1, min(len(frame), entry_idx + 1 + int(horizon_bars))):
            bar = frame.iloc[j]
            high = float(bar["high"])
            low = float(bar["low"])
            close = float(bar["close"])
            if direction > 0:
                mfe = max(mfe, (high - entry) / cag.PIP_SIZE)
                mae = max(mae, (entry - low) / cag.PIP_SIZE)
                hit_tp = high >= tp
                hit_sl = low <= sl
                if hit_tp and hit_sl:
                    outcome = "stop_ambiguous"
                    exit_idx = j
                    exit_price = sl
                    break
                if hit_tp:
                    outcome = "target"
                    exit_idx = j
                    exit_price = tp
                    break
                if hit_sl:
                    outcome = "stop"
                    exit_idx = j
                    exit_price = sl
                    break
            else:
                mfe = max(mfe, (entry - low) / cag.PIP_SIZE)
                mae = max(mae, (high - entry) / cag.PIP_SIZE)
                hit_tp = low <= tp
                hit_sl = high >= sl
                if hit_tp and hit_sl:
                    outcome = "stop_ambiguous"
                    exit_idx = j
                    exit_price = sl
                    break
                if hit_tp:
                    outcome = "target"
                    exit_idx = j
                    exit_price = tp
                    break
                if hit_sl:
                    outcome = "stop"
                    exit_idx = j
                    exit_price = sl
                    break
            exit_idx = j
            exit_price = close

        pnl_pips = ((exit_price - entry) / cag.PIP_SIZE) if direction > 0 else ((entry - exit_price) / cag.PIP_SIZE)
        trades.append(
            ReplayTrade(
                signal_time=pd.Timestamp(sig.time).isoformat(),
                session_label=str(sig.session_label),
                spike_direction=str(sig.spike_direction),
                spike_move_pips=float(sig.spike_move_pips),
                confirmation_bars=int(sig.confirmation_bars),
                side=side,
                entry_price=round(entry, 3),
                entry_bar_index=int(entry_idx),
                exit_time=pd.Timestamp(frame.iloc[exit_idx]["time"]).isoformat(),
                exit_bar_index=int(exit_idx),
                exit_price=round(float(exit_price), 3),
                outcome_code=outcome,
                pnl_pips=round(float(pnl_pips), 4),
                mfe_pips=round(float(mfe), 4),
                mae_pips=round(float(mae), 4),
                bars_held=int(exit_idx - entry_idx),
            )
        )
    return trades


def summarize_trades(trades: list[ReplayTrade]) -> dict[str, Any]:
    if not trades:
        return {
            "trades": 0,
            "win_rate_pct": 0.0,
            "avg_pips": 0.0,
            "net_pips": 0.0,
            "profit_factor": None,
            "avg_mfe_pips": 0.0,
            "avg_mae_pips": 0.0,
            "avg_bars_held": 0.0,
            "tp_hit_rate_pct": 0.0,
            "stop_hit_rate_pct": 0.0,
            "timeout_rate_pct": 0.0,
        }
    pnl = [t.pnl_pips for t in trades]
    wins = [x for x in pnl if x > 0]
    losses = [x for x in pnl if x < 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    pf = gross_win / gross_loss if gross_loss > 0 else (999.0 if gross_win > 0 else None)
    return {
        "trades": len(trades),
        "win_rate_pct": round(sum(1 for x in pnl if x > 0) / len(trades) * 100.0, 2),
        "avg_pips": round(sum(pnl) / len(trades), 4),
        "net_pips": round(sum(pnl), 2),
        "profit_factor": None if pf is None else round(float(pf), 4),
        "avg_mfe_pips": round(sum(t.mfe_pips for t in trades) / len(trades), 4),
        "avg_mae_pips": round(sum(t.mae_pips for t in trades) / len(trades), 4),
        "avg_bars_held": round(sum(t.bars_held for t in trades) / len(trades), 2),
        "tp_hit_rate_pct": round(sum(1 for t in trades if t.outcome_code == "target") / len(trades) * 100.0, 2),
        "stop_hit_rate_pct": round(sum(1 for t in trades if t.outcome_code in {"stop", "stop_ambiguous"}) / len(trades) * 100.0, 2),
        "timeout_rate_pct": round(sum(1 for t in trades if t.outcome_code == "timeout") / len(trades) * 100.0, 2),
    }


def summarize_by(trades: list[ReplayTrade], key: str) -> list[dict[str, Any]]:
    buckets: dict[str, list[ReplayTrade]] = {}
    for trade in trades:
        buckets.setdefault(str(getattr(trade, key)), []).append(trade)
    rows: list[dict[str, Any]] = []
    for bucket, items in buckets.items():
        row = summarize_trades(items)
        row[key] = bucket
        rows.append(row)
    rows.sort(key=lambda r: (-r["trades"], str(r[key])))
    return rows


def write_trade_csv(path: Path, trades: list[ReplayTrade]) -> None:
    if not trades:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(trades[0]).keys()))
        writer.writeheader()
        for trade in trades:
            writer.writerow(asdict(trade))


def build_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Post Spike Retracement Bar-by-Bar Replay (250k)",
        "",
        "Causal bar-by-bar replay of the narrowed `post_spike_retracement` variant on the 250k USDJPY dataset.",
        "",
        "Narrowed variant:",
        "",
        "- session in `ny` or `london/ny`",
        "- confirmation bars in `2` or `3`",
        "",
        "## Config",
        "",
        f"- source dataset: `{payload['config']['source_dataset']}`",
        f"- bars used: `{payload['config']['bars']}`",
        f"- target pips: `{payload['config']['target_pips']}`",
        f"- stop pips: `{payload['config']['stop_pips']}`",
        f"- horizon bars: `{payload['config']['horizon_bars']}`",
        "",
        "## Summary",
        "",
        f"- trades: `{summary['trades']}`",
        f"- win rate: `{summary['win_rate_pct']}%`",
        f"- avg pips: `{summary['avg_pips']}p`",
        f"- net pips: `{summary['net_pips']}p`",
        f"- profit factor: `{summary['profit_factor']}`",
        f"- avg MFE: `{summary['avg_mfe_pips']}p`",
        f"- avg MAE: `{summary['avg_mae_pips']}p`",
        f"- avg bars held: `{summary['avg_bars_held']}`",
        "",
        "## By Session",
        "",
    ]
    for row in payload["by_session"]:
        lines.append(
            f"- `{row['session_label']}`: trades `{row['trades']}` | WR `{row['win_rate_pct']}%` | avg `{row['avg_pips']}p` | net `{row['net_pips']}p` | PF `{row['profit_factor']}`"
        )
    lines.extend(["", "## By Confirmation Bars", ""])
    for row in payload["by_confirmation_bars"]:
        lines.append(
            f"- `{row['confirmation_bars']}`: trades `{row['trades']}` | WR `{row['win_rate_pct']}%` | avg `{row['avg_pips']}p` | net `{row['net_pips']}p` | PF `{row['profit_factor']}`"
        )
    lines.extend(["", "## By Spike Direction", ""])
    for row in payload["by_spike_direction"]:
        lines.append(
            f"- `{row['spike_direction']}`: trades `{row['trades']}` | WR `{row['win_rate_pct']}%` | avg `{row['avg_pips']}p` | net `{row['net_pips']}p` | PF `{row['profit_factor']}`"
        )
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    args = parse_args()
    source = Path(args.source_dataset).resolve()
    frame = load_recent_slice(source, bars=int(args.bars), assume_spread_pips=float(args.assume_spread_pips))
    signals = detect_narrowed_signals(frame)
    trades = replay_trades_bar_by_bar(
        frame,
        signals,
        target_pips=float(args.target_pips),
        stop_pips=float(args.stop_pips),
        horizon_bars=int(args.horizon_bars),
    )
    payload = {
        "config": {
            "source_dataset": str(source),
            "bars": int(len(frame)),
            "target_pips": float(args.target_pips),
            "stop_pips": float(args.stop_pips),
            "horizon_bars": int(args.horizon_bars),
        },
        "signals_detected": int(len(signals)),
        "summary": summarize_trades(trades),
        "by_session": summarize_by(trades, "session_label"),
        "by_confirmation_bars": summarize_by(trades, "confirmation_bars"),
        "by_spike_direction": summarize_by(trades, "spike_direction"),
    }

    json_path = Path(args.output_json)
    md_path = Path(args.output_md)
    csv_path = Path(args.output_csv)
    json_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    md_path.write_text(build_markdown(payload), encoding="utf-8")
    write_trade_csv(csv_path, trades)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
