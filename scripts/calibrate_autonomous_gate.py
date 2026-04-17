#!/usr/bin/env python3
"""
Autonomous Fillmore gate calibration harness.

This is a research tool, not a live-trading component. It replays the current
autonomous gate logic over historical USDJPY M1 data, simulates simple
trend-follow outcomes from each candidate bar, and sweeps threshold grids to
identify gate settings that look stronger than the current defaults.

Why this exists:
  - Autonomous Fillmore's gate is currently hand-tuned.
  - The repo already has deep M1 history (500k / 1000k datasets) and a strong
    research convention under research_out/.
  - Before changing live gate thresholds, we want cross-dataset evidence.

Scope of this first pass:
  - Session-aware spread scaling
  - Minimum M5 ATR floor
  - Structure proximity threshold
  - Pullback-zone minimum width
  - Pullback lookback bars

It intentionally does NOT attempt to emulate the full LLM or exit manager.
Instead it uses a first-touch forward outcome proxy:
  - gate picks a direction from the same trend signals it already uses
  - bar "wins" if +target_pips is hit before -stop_pips inside horizon bars

That makes this harness fast, repeatable, and useful for threshold selection.
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

from api import autonomous_fillmore

OUT_DIR = ROOT / "research_out"
DEFAULT_DATASETS = [
    OUT_DIR / "USDJPY_M1_OANDA_500k.csv",
    OUT_DIR / "USDJPY_M1_OANDA_1000k.csv",
]
DEFAULT_JSON = OUT_DIR / "autonomous_gate_calibration.json"
DEFAULT_MD = OUT_DIR / "autonomous_gate_calibration.md"
PIP_SIZE = 0.01


@dataclass
class MaskSummary:
    pass_count: int
    pass_rate_pct: float
    win_rate_pct: float
    avg_pips: float
    net_pips: float
    profit_factor: float | None
    tp_hit_rate_pct: float
    stop_hit_rate_pct: float
    timeout_rate_pct: float
    early_net_pips: float
    middle_net_pips: float
    late_net_pips: float
    positive_splits: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate Autonomous Fillmore gate thresholds on historical USDJPY M1 data")
    p.add_argument(
        "--datasets",
        nargs="*",
        default=[str(p) for p in DEFAULT_DATASETS if p.exists()],
        help="Dataset path(s). Defaults to existing 500k/1000k research datasets.",
    )
    p.add_argument("--modes", nargs="*", default=["balanced", "aggressive"], help="Gate modes to calibrate")
    p.add_argument("--target-pips", type=float, default=6.0, help="Forward outcome take-profit proxy")
    p.add_argument("--stop-pips", type=float, default=10.0, help="Forward outcome stop proxy")
    p.add_argument("--horizon-bars", type=int, default=30, help="Forward outcome horizon in M1 bars")
    p.add_argument(
        "--assume-spread-pips",
        type=float,
        default=1.2,
        help="Fallback spread when dataset lacks spread_pips column (used by 500k dataset).",
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


def _classify_session_label(ts: pd.Timestamp) -> str:
    hour = int(ts.tz_convert("UTC").hour)
    in_tokyo = 0 <= hour < 9
    in_london = 7 <= hour < 16
    in_ny = 12 <= hour < 21
    labels: list[str] = []
    if in_tokyo:
        labels.append("tokyo")
    if in_london:
        labels.append("london")
    if in_ny:
        labels.append("ny")
    return "/".join(labels) if labels else "off-hours"


def _load_dataset(path: Path, assumed_spread_pips: float) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time"], low_memory=False)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).copy()
    if "spread_pips" in df.columns:
        df["spread_pips"] = pd.to_numeric(df["spread_pips"], errors="coerce").fillna(float(assumed_spread_pips))
        df["spread_source"] = "dataset"
    else:
        df["spread_pips"] = float(assumed_spread_pips)
        df["spread_source"] = "assumed"
    return df


def _resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return (
        df.set_index("time")[["open", "high", "low", "close"]]
        .resample(rule, label="right", closed="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
        .reset_index()
    )


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _atr_pips_df(df: pd.DataFrame, period: int, pip_size: float = PIP_SIZE) -> pd.Series:
    highs = df["high"].astype(float)
    lows = df["low"].astype(float)
    closes = df["close"].astype(float)
    prev_close = closes.shift(1)
    tr = pd.concat(
        [
            highs - lows,
            (highs - prev_close).abs(),
            (lows - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr / pip_size


def _merge_asof_feature(base: pd.DataFrame, source: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return pd.merge_asof(
        base.sort_values("time"),
        source[["time", *cols]].sort_values("time"),
        on="time",
        direction="backward",
    )


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["session_label"] = work["time"].map(_classify_session_label)

    close = work["close"].astype(float)
    work["m1_e5"] = _ema(close, 5)
    work["m1_e9"] = _ema(close, 9)
    work["m1_e13"] = _ema(close, 13)
    work["m1_e17"] = _ema(close, 17)
    work["m1_e21"] = _ema(close, 21)
    work["m1_atr_pips"] = _atr_pips_df(work, 14)
    work["m1_stack"] = np.select(
        [
            (work["m1_e5"] > work["m1_e9"]) & (work["m1_e9"] > work["m1_e21"]),
            (work["m1_e5"] < work["m1_e9"]) & (work["m1_e9"] < work["m1_e21"]),
        ],
        ["bull", "bear"],
        default="none",
    )

    m3 = _resample_ohlc(work, "3min")
    m3["m3_e9"] = _ema(m3["close"].astype(float), 9)
    m3["m3_e21"] = _ema(m3["close"].astype(float), 21)
    m3["m3_trend"] = np.select(
        [
            (m3["close"] > m3["m3_e9"]) & (m3["m3_e9"] > m3["m3_e21"]),
            (m3["close"] < m3["m3_e9"]) & (m3["m3_e9"] < m3["m3_e21"]),
        ],
        ["bull", "bear"],
        default="none",
    )
    work = _merge_asof_feature(work, m3, ["m3_trend"])

    m5 = _resample_ohlc(work, "5min")
    m5["m5_atr_pips"] = _atr_pips_df(m5, 14)
    work = _merge_asof_feature(work, m5, ["m5_atr_pips"])

    work["day"] = work["time"].dt.floor("D")
    work["today_high"] = work.groupby("day")["high"].cummax()
    work["today_low"] = work.groupby("day")["low"].cummin()
    daily = work.groupby("day").agg(day_high=("high", "max"), day_low=("low", "min")).reset_index()
    daily["prev_day_high"] = daily["day_high"].shift(1)
    daily["prev_day_low"] = daily["day_low"].shift(1)
    work = work.merge(daily[["day", "prev_day_high", "prev_day_low"]], on="day", how="left")

    round_floor = np.floor(work["close"] * 2.0) / 2.0
    round_ceil = np.ceil(work["close"] * 2.0) / 2.0
    round_dist_pips = np.minimum((work["close"] - round_floor).abs(), (round_ceil - work["close"]).abs()) / PIP_SIZE
    structure_candidates = pd.concat(
        [
            ((work["today_high"] - work["close"]).abs() / PIP_SIZE),
            ((work["today_low"] - work["close"]).abs() / PIP_SIZE),
            ((work["prev_day_high"] - work["close"]).abs() / PIP_SIZE),
            ((work["prev_day_low"] - work["close"]).abs() / PIP_SIZE),
            round_dist_pips,
        ],
        axis=1,
    )
    work["nearest_structure_pips"] = structure_candidates.min(axis=1, skipna=True)
    work["near_daily_hl_pips"] = pd.concat(
        [
            ((work["today_high"] - work["close"]).abs() / PIP_SIZE),
            ((work["today_low"] - work["close"]).abs() / PIP_SIZE),
        ],
        axis=1,
    ).min(axis=1, skipna=True)
    return work


def compute_direction(frame: pd.DataFrame, mode: str) -> np.ndarray:
    m3 = frame["m3_trend"].astype(str)
    m1 = frame["m1_stack"].astype(str)
    out = np.zeros(len(frame), dtype=np.int8)
    if mode in {"conservative", "balanced"}:
        bull = (m3 == "bull") & (m1 == "bull")
        bear = (m3 == "bear") & (m1 == "bear")
        out[bull.to_numpy()] = 1
        out[bear.to_numpy()] = -1
        return out

    mismatch = (
        m3.isin(["bull", "bear"])
        & m1.isin(["bull", "bear"])
        & (m3 != m1)
    )
    bull = ((m3 == "bull") | (m1 == "bull")) & ~mismatch
    bear = ((m3 == "bear") | (m1 == "bear")) & ~mismatch
    out[bull.to_numpy()] = 1
    out[bear.to_numpy()] = -1
    return out


def compute_pullback_flag(
    frame: pd.DataFrame,
    *,
    lookback_bars: int,
    zone_min_pips: float,
    zone_max_pips: float = 4.0,
) -> pd.Series:
    work = frame
    atr_pips = work["m1_atr_pips"].copy()
    zone_width_pips = (atr_pips * 0.5).clip(lower=zone_min_pips, upper=zone_max_pips)
    zone_width_pips = zone_width_pips.fillna(float(zone_min_pips))
    zone_width = zone_width_pips * PIP_SIZE
    zone = (work["close"] - work["m1_e9"]).abs() <= zone_width

    lookback = max(1, int(lookback_bars))
    recent_low = work["low"].rolling(lookback, min_periods=1).min()
    recent_high = work["high"].rolling(lookback, min_periods=1).max()
    upper_ema = pd.concat([work["m1_e13"], work["m1_e17"]], axis=1).max(axis=1)
    lower_ema = pd.concat([work["m1_e13"], work["m1_e17"]], axis=1).min(axis=1)
    bull_pull = recent_low <= upper_ema
    bear_pull = recent_high >= lower_ema

    return pd.Series(
        np.where(
            work["direction"] > 0,
            zone | bull_pull,
            np.where(work["direction"] < 0, zone | bear_pull, False),
        ),
        index=work.index,
    )


def simulate_first_touch_outcomes(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    direction: np.ndarray,
    *,
    target_pips: float,
    stop_pips: float,
    horizon_bars: int,
    pip_size: float = PIP_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(close)
    out_pips = np.full(n, np.nan, dtype=float)
    out_code = np.full(n, "no_trade", dtype=object)
    tp_px = float(target_pips) * pip_size
    sl_px = float(stop_pips) * pip_size
    horizon = max(1, int(horizon_bars))

    for i in range(n):
        side = int(direction[i])
        if side == 0:
            continue
        end = min(n, i + horizon + 1)
        if i + 1 >= end:
            continue
        entry = float(close[i])
        future_high = high[i + 1 : end]
        future_low = low[i + 1 : end]
        if side > 0:
            tp = entry + tp_px
            sl = entry - sl_px
            for h, l in zip(future_high, future_low):
                hit_tp = h >= tp
                hit_sl = l <= sl
                if hit_tp and hit_sl:
                    out_pips[i] = -float(stop_pips)
                    out_code[i] = "stop_ambiguous"
                    break
                if hit_tp:
                    out_pips[i] = float(target_pips)
                    out_code[i] = "target"
                    break
                if hit_sl:
                    out_pips[i] = -float(stop_pips)
                    out_code[i] = "stop"
                    break
            else:
                final_close = float(close[end - 1])
                out_pips[i] = (final_close - entry) / pip_size
                out_code[i] = "timeout"
        else:
            tp = entry - tp_px
            sl = entry + sl_px
            for h, l in zip(future_high, future_low):
                hit_tp = l <= tp
                hit_sl = h >= sl
                if hit_tp and hit_sl:
                    out_pips[i] = -float(stop_pips)
                    out_code[i] = "stop_ambiguous"
                    break
                if hit_tp:
                    out_pips[i] = float(target_pips)
                    out_code[i] = "target"
                    break
                if hit_sl:
                    out_pips[i] = -float(stop_pips)
                    out_code[i] = "stop"
                    break
            else:
                final_close = float(close[end - 1])
                out_pips[i] = (entry - final_close) / pip_size
                out_code[i] = "timeout"
    return out_pips, out_code


def build_pass_mask(
    frame: pd.DataFrame,
    *,
    mode: str,
    spread_scale: float,
    min_m5_atr_pips: float,
    level_proximity_pips: float,
    pullback_zone_min_pips: float,
    pullback_lookback_bars: int,
) -> pd.Series:
    thresholds = autonomous_fillmore.GATE_THRESHOLDS[mode]
    session_cap = frame["session_label"].map(
        lambda s: autonomous_fillmore._session_spread_limit_pips(s) * float(spread_scale)
    )
    mask = frame["session_label"] != "off-hours"
    mask &= frame["spread_pips"] <= session_cap
    mask &= frame["direction"] != 0

    if thresholds.get("require_m3_trend", False):
        mask &= frame["m3_trend"].isin(["bull", "bear"])
    if thresholds.get("require_m1_stack", False):
        mask &= frame["m1_stack"].isin(["bull", "bear"])
    if thresholds.get("reject_m3_m1_mismatch_if_both_present", False):
        both = frame["m3_trend"].isin(["bull", "bear"]) & frame["m1_stack"].isin(["bull", "bear"])
        mask &= ~(both & (frame["m3_trend"] != frame["m1_stack"]))
    if thresholds.get("require_any_trend_signal", False):
        mask &= frame["direction"] != 0

    if float(min_m5_atr_pips) > 0:
        mask &= frame["m5_atr_pips"].fillna(float(min_m5_atr_pips)) >= float(min_m5_atr_pips)
    if float(level_proximity_pips) > 0:
        mask &= frame["nearest_structure_pips"].fillna(9999.0) <= float(level_proximity_pips)
    if float(thresholds.get("require_daily_hl_buffer_pips") or 0.0) > 0:
        mask &= frame["near_daily_hl_pips"].fillna(9999.0) > float(thresholds["require_daily_hl_buffer_pips"])
    if thresholds.get("require_pullback_or_zone", False):
        pullback = compute_pullback_flag(
            frame,
            lookback_bars=pullback_lookback_bars,
            zone_min_pips=pullback_zone_min_pips,
        )
        mask &= pullback
    return mask.fillna(False)


def summarize_mask(frame: pd.DataFrame, mask: pd.Series) -> MaskSummary:
    chosen = frame.loc[mask.fillna(False)].copy()
    total = len(frame)
    if chosen.empty:
        return MaskSummary(
            pass_count=0,
            pass_rate_pct=0.0,
            win_rate_pct=0.0,
            avg_pips=0.0,
            net_pips=0.0,
            profit_factor=None,
            tp_hit_rate_pct=0.0,
            stop_hit_rate_pct=0.0,
            timeout_rate_pct=0.0,
            early_net_pips=0.0,
            middle_net_pips=0.0,
            late_net_pips=0.0,
            positive_splits=0,
        )
    wins = chosen["outcome_pips"] > 0
    gross_win = float(chosen.loc[chosen["outcome_pips"] > 0, "outcome_pips"].sum())
    gross_loss = abs(float(chosen.loc[chosen["outcome_pips"] < 0, "outcome_pips"].sum()))
    pf = gross_win / gross_loss if gross_loss > 0 else (999.0 if gross_win > 0 else None)

    splits = np.array_split(chosen.index.to_numpy(), 3)
    split_nets: list[float] = []
    for idx in splits:
        if len(idx) == 0:
            split_nets.append(0.0)
        else:
            split_nets.append(float(chosen.loc[idx, "outcome_pips"].sum()))

    return MaskSummary(
        pass_count=int(len(chosen)),
        pass_rate_pct=round(len(chosen) / max(total, 1) * 100.0, 2),
        win_rate_pct=round(float(wins.mean()) * 100.0, 2),
        avg_pips=round(float(chosen["outcome_pips"].mean()), 4),
        net_pips=round(float(chosen["outcome_pips"].sum()), 2),
        profit_factor=None if pf is None else round(float(pf), 4),
        tp_hit_rate_pct=round(float((chosen["outcome_code"] == "target").mean()) * 100.0, 2),
        stop_hit_rate_pct=round(float(chosen["outcome_code"].isin(["stop", "stop_ambiguous"]).mean()) * 100.0, 2),
        timeout_rate_pct=round(float((chosen["outcome_code"] == "timeout").mean()) * 100.0, 2),
        early_net_pips=round(split_nets[0], 2),
        middle_net_pips=round(split_nets[1], 2),
        late_net_pips=round(split_nets[2], 2),
        positive_splits=sum(1 for x in split_nets if x > 0),
    )


def _score_candidate(candidate: dict[str, Any]) -> float:
    ds_rows = list((candidate.get("datasets") or {}).values())
    if not ds_rows:
        return -1e9
    avg_delta_net = float(np.mean([float(r.get("delta_net_pips") or 0.0) for r in ds_rows]))
    avg_delta_pf = float(np.mean([float(r.get("delta_profit_factor") or 0.0) for r in ds_rows]))
    min_ret = float(min(float(r.get("trade_retention_pct") or 0.0) for r in ds_rows))
    all_positive = all(float(r.get("delta_net_pips") or 0.0) >= 0.0 for r in ds_rows)
    positive_splits = sum(int(r.get("positive_splits") or 0) for r in ds_rows)
    score = avg_delta_net + (avg_delta_pf * 100.0) + (positive_splits * 25.0)
    if all_positive:
        score += 1000.0
    if min_ret < 35.0:
        score -= (35.0 - min_ret) * 10.0
    return round(score, 4)


def evaluate_mode_on_dataset(
    frame: pd.DataFrame,
    *,
    mode: str,
    spread_scales: list[float],
    atr_floors: list[float],
    level_thresholds: list[float],
    pullback_zone_mins: list[float],
    pullback_lookbacks: list[int],
) -> dict[str, Any]:
    thresholds = autonomous_fillmore.GATE_THRESHOLDS[mode]
    base = {
        "spread_scale": 1.0,
        "min_m5_atr_pips": float(thresholds.get("require_min_m5_atr_pips") or 0.0),
        "level_proximity_pips": float(thresholds.get("require_level_proximity_pips") or 0.0),
        "pullback_zone_min_pips": 1.5,
        "pullback_lookback_bars": 3,
    }
    baseline_mask = build_pass_mask(frame, mode=mode, **base)
    baseline = summarize_mask(frame, baseline_mask)
    candidates: list[dict[str, Any]] = []

    zone_grid = pullback_zone_mins if thresholds.get("require_pullback_or_zone", False) else [base["pullback_zone_min_pips"]]
    lookback_grid = pullback_lookbacks if thresholds.get("require_pullback_or_zone", False) else [base["pullback_lookback_bars"]]

    for spread_scale in spread_scales:
        for atr_floor in atr_floors:
            for prox in level_thresholds:
                for zone_min in zone_grid:
                    for lookback in lookback_grid:
                        params = {
                            "spread_scale": float(spread_scale),
                            "min_m5_atr_pips": float(atr_floor),
                            "level_proximity_pips": float(prox),
                            "pullback_zone_min_pips": float(zone_min),
                            "pullback_lookback_bars": int(lookback),
                        }
                        mask = build_pass_mask(frame, mode=mode, **params)
                        summary = summarize_mask(frame, mask)
                        candidates.append(
                            {
                                "params": params,
                                "summary": asdict(summary),
                                "delta_net_pips": round(summary.net_pips - baseline.net_pips, 2),
                                "delta_profit_factor": round((summary.profit_factor or 0.0) - (baseline.profit_factor or 0.0), 4),
                                "trade_retention_pct": round((summary.pass_count / max(baseline.pass_count, 1)) * 100.0, 2),
                            }
                        )

    return {
        "baseline_params": base,
        "baseline_summary": asdict(baseline),
        "candidates": candidates,
    }


def run_dataset_mode(
    dataset_path: Path,
    *,
    assumed_spread_pips: float,
    mode: str,
    target_pips: float,
    stop_pips: float,
    horizon_bars: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = _load_dataset(dataset_path, assumed_spread_pips=assumed_spread_pips)
    frame = build_feature_frame(raw)
    frame["direction"] = compute_direction(frame, mode)
    out_pips, out_code = simulate_first_touch_outcomes(
        frame["close"].to_numpy(dtype=float),
        frame["high"].to_numpy(dtype=float),
        frame["low"].to_numpy(dtype=float),
        frame["direction"].to_numpy(dtype=np.int8),
        target_pips=target_pips,
        stop_pips=stop_pips,
        horizon_bars=horizon_bars,
    )
    frame["outcome_pips"] = out_pips
    frame["outcome_code"] = out_code
    meta = {
        "bars": int(len(frame)),
        "spread_source": str(raw["spread_source"].iloc[0]) if len(raw) else "unknown",
        "directional_bars": int((frame["direction"] != 0).sum()),
    }
    return frame, meta


def aggregate_mode_results(
    dataset_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    candidate_count = min(len(v["candidates"]) for v in dataset_results.values())
    aggregated: list[dict[str, Any]] = []
    for idx in range(candidate_count):
        first = next(iter(dataset_results.values()))["candidates"][idx]
        params = dict(first["params"])
        ds_payload: dict[str, Any] = {}
        for ds_key, result in dataset_results.items():
            cand = result["candidates"][idx]
            ds_payload[ds_key] = {
                "delta_net_pips": cand["delta_net_pips"],
                "delta_profit_factor": cand["delta_profit_factor"],
                "trade_retention_pct": cand["trade_retention_pct"],
                "net_pips": cand["summary"]["net_pips"],
                "profit_factor": cand["summary"]["profit_factor"],
                "pass_count": cand["summary"]["pass_count"],
                "positive_splits": cand["summary"]["positive_splits"],
            }
        row = {
            "params": params,
            "datasets": ds_payload,
        }
        row["score"] = _score_candidate(row)
        aggregated.append(row)

    aggregated.sort(key=lambda row: row["score"], reverse=True)
    return {
        "top_candidates": aggregated[:20],
    }


def build_markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Autonomous Gate Calibration",
        "",
        "This report calibrates Autonomous Fillmore gate thresholds against historical USDJPY M1 data using a forward first-touch outcome proxy.",
        "",
        "## Run Config",
        "",
        f"- target pips: `{payload['config']['target_pips']}`",
        f"- stop pips: `{payload['config']['stop_pips']}`",
        f"- horizon bars: `{payload['config']['horizon_bars']}`",
        f"- assumed spread fallback: `{payload['config']['assumed_spread_pips']}`",
        "",
    ]
    for mode, mode_payload in payload["modes"].items():
        lines.extend([
            f"## Mode: `{mode}`",
            "",
            "### Baselines",
            "",
        ])
        for ds_key, ds_result in mode_payload["datasets"].items():
            base = ds_result["baseline_summary"]
            lines.append(
                f"- `{ds_key}`: {base['pass_count']} passes | {base['win_rate_pct']}% WR | "
                f"avg {base['avg_pips']}p | net {base['net_pips']}p | PF {base['profit_factor']}"
            )
        lines.extend(["", "### Top Candidates", ""])
        for idx, cand in enumerate(mode_payload["aggregate"]["top_candidates"][:8], start=1):
            params = cand["params"]
            lines.append(
                f"{idx}. score `{cand['score']}` | spread_scale `{params['spread_scale']}` | "
                f"m5_atr `{params['min_m5_atr_pips']}` | structure `{params['level_proximity_pips']}` | "
                f"pullback_min `{params['pullback_zone_min_pips']}` | lookback `{params['pullback_lookback_bars']}`"
            )
            for ds_key, ds_row in cand["datasets"].items():
                lines.append(
                    f"   - `{ds_key}`: delta net `{ds_row['delta_net_pips']}`p | "
                    f"delta PF `{ds_row['delta_profit_factor']}` | "
                    f"retention `{ds_row['trade_retention_pct']}%` | passes `{ds_row['pass_count']}`"
                )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    args = parse_args()
    dataset_paths = [Path(p).resolve() for p in args.datasets]
    mode_names = [m for m in args.modes if m in autonomous_fillmore.GATE_THRESHOLDS]
    if not dataset_paths:
        raise SystemExit("No datasets provided.")
    if not mode_names:
        raise SystemExit("No valid modes provided.")

    spread_scales = [0.85, 1.0, 1.15, 1.3]
    atr_floors = [2.5, 3.0, 3.5, 4.0]
    level_thresholds = [5.0, 8.0, 10.0, 12.0]
    pullback_zone_mins = [1.5, 2.0, 2.5]
    pullback_lookbacks = [1, 2, 3]

    payload: dict[str, Any] = {
        "datasets": [str(p) for p in dataset_paths],
        "config": {
            "target_pips": args.target_pips,
            "stop_pips": args.stop_pips,
            "horizon_bars": args.horizon_bars,
            "assumed_spread_pips": args.assume_spread_pips,
            "spread_scales": spread_scales,
            "atr_floors": atr_floors,
            "level_thresholds": level_thresholds,
            "pullback_zone_mins": pullback_zone_mins,
            "pullback_lookbacks": pullback_lookbacks,
        },
        "modes": {},
    }

    for mode in mode_names:
        mode_payload: dict[str, Any] = {"datasets": {}}
        for dataset_path in dataset_paths:
            frame, meta = run_dataset_mode(
                dataset_path,
                assumed_spread_pips=float(args.assume_spread_pips),
                mode=mode,
                target_pips=float(args.target_pips),
                stop_pips=float(args.stop_pips),
                horizon_bars=int(args.horizon_bars),
            )
            ds_result = evaluate_mode_on_dataset(
                frame,
                mode=mode,
                spread_scales=spread_scales,
                atr_floors=atr_floors,
                level_thresholds=level_thresholds,
                pullback_zone_mins=pullback_zone_mins,
                pullback_lookbacks=pullback_lookbacks,
            )
            ds_result["dataset_meta"] = meta
            mode_payload["datasets"][_dataset_key(dataset_path)] = ds_result
        mode_payload["aggregate"] = aggregate_mode_results(mode_payload["datasets"])
        payload["modes"][mode] = mode_payload

    json_path = Path(args.output_json)
    md_path = Path(args.output_md)
    json_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    md_path.write_text(build_markdown_report(payload), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
