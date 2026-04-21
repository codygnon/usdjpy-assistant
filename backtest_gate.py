#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api import autonomous_fillmore as af
from core.indicators import bollinger_bands

PIP_SIZE = 0.01
DEFAULT_TP_PIPS = 8.0
DEFAULT_SL_PIPS = 12.0
DEFAULT_HORIZON_MIN = 60
DEFAULT_LIMIT_FILL_MIN = 10
DEFAULT_FALSE_BREAKOUT_MIN = 15
DEFAULT_EXPANSION_FOLLOW_MIN = 30
DEFAULT_RETEST_TOLERANCE_PIPS = 2.0
DEFAULT_PROGRESS_EVERY = 10_000
DEFAULT_M1_WINDOW = 900
DEFAULT_M3_WINDOW = 120
DEFAULT_M5_WINDOW = 120
DEFAULT_M15_WINDOW = 120
DEFAULT_D_WINDOW = 5
DEFAULT_W_WINDOW = 4

DEFAULT_OUTPUT_REPORT = ROOT / "gate_backtest_report.json"
DEFAULT_OUTPUT_WAKES = ROOT / "gate_backtest_wakes.csv"
DEFAULT_OUTPUT_COMPRESSION = ROOT / "gate_backtest_compression.csv"
DEFAULT_OUTPUT_SUMMARY = ROOT / "gate_backtest_summary.txt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay USDJPY M1 history through the live autonomous hybrid gate")
    p.add_argument(
        "--data",
        nargs="*",
        default=None,
        help="CSV file(s) to load. Defaults to the best available research_out USDJPY M1 datasets.",
    )
    p.add_argument("--start", default=None, help="Optional UTC start timestamp filter")
    p.add_argument("--end", default=None, help="Optional UTC end timestamp filter")
    p.add_argument("--max-bars", type=int, default=None, help="Optional tail limit after filtering")
    p.add_argument("--aggressiveness", default="balanced", choices=["conservative", "balanced", "aggressive", "very_aggressive"])
    p.add_argument("--tp-pips", type=float, default=DEFAULT_TP_PIPS)
    p.add_argument("--sl-pips", type=float, default=DEFAULT_SL_PIPS)
    p.add_argument("--timeout-min", type=int, default=DEFAULT_HORIZON_MIN)
    p.add_argument("--limit-fill-min", type=int, default=DEFAULT_LIMIT_FILL_MIN)
    p.add_argument("--assume-spread-pips", type=float, default=1.2)
    p.add_argument("--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY)
    p.add_argument("--with-sensitivity", action="store_true", help="Run a compact parameter sensitivity sweep after the main replay")
    p.add_argument("--sensitivity-max-bars", type=int, default=100_000, help="Tail bars to use for sensitivity runs")
    p.add_argument("--output-report", default=str(DEFAULT_OUTPUT_REPORT))
    p.add_argument("--output-wakes", default=str(DEFAULT_OUTPUT_WAKES))
    p.add_argument("--output-compression", default=str(DEFAULT_OUTPUT_COMPRESSION))
    p.add_argument("--output-summary", default=str(DEFAULT_OUTPUT_SUMMARY))
    return p.parse_args()


def _discover_default_data_files() -> list[Path]:
    extended = ROOT / "research_out" / "USDJPY_M1_OANDA_extended.csv"
    if extended.exists():
        return [extended]
    preferred = [
        ROOT / "research_out" / "USDJPY_M1_OANDA_1000k.csv",
        ROOT / "research_out" / "USDJPY_M1_OANDA_500k.csv",
    ]
    found = [p for p in preferred if p.exists()]
    if found:
        return found
    all_csv = sorted((ROOT / "research_out").glob("USDJPY_M1_OANDA*.csv"))
    filtered = [
        p for p in all_csv
        if "split" not in p.name
        and "train" not in p.name
        and "test" not in p.name
        and "50k" not in p.name
        and "200k" not in p.name
        and "300k" not in p.name
        and "400k" not in p.name
        and "600k" not in p.name
        and "700k" not in p.name
        and "800k" not in p.name
    ]
    return filtered


def _report_session_label(raw: str) -> str:
    label = str(raw or "off-hours")
    if label == "ny":
        return "newyork"
    if label == "london/ny":
        return "london/newyork"
    if label == "off-hours":
        return "off_hours"
    return label


def _load_one_csv(path: Path, assume_spread_pips: float) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time"], low_memory=False)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).copy()
    if "spread_pips" in df.columns:
        df["spread_pips"] = pd.to_numeric(df["spread_pips"], errors="coerce").fillna(float(assume_spread_pips))
    else:
        df["spread_pips"] = float(assume_spread_pips)
    df["source_file"] = path.name
    return df[["time", "open", "high", "low", "close", "spread_pips", "source_file"]]


def load_history(paths: list[Path], assume_spread_pips: float, start: str | None, end: str | None, max_bars: int | None) -> pd.DataFrame:
    if not paths:
        raise FileNotFoundError("No USDJPY M1 datasets found.")
    chunks = [_load_one_csv(p, assume_spread_pips) for p in paths]
    df = pd.concat(chunks, ignore_index=True)
    df = df.sort_values("time").drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)
    if start:
        start_ts = pd.Timestamp(start, tz="UTC")
        df = df.loc[df["time"] >= start_ts].copy()
    if end:
        end_ts = pd.Timestamp(end, tz="UTC")
        df = df.loc[df["time"] <= end_ts].copy()
    if max_bars:
        df = df.tail(int(max_bars)).copy()
    if df.empty:
        raise ValueError("No bars remain after applying filters.")
    return df.reset_index(drop=True)


def _resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    out = (
        df.set_index("time")[["open", "high", "low", "close"]]
        .resample(rule, label="right", closed="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
        .reset_index()
    )
    return out


def build_frames(m1: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "M1": m1[["time", "open", "high", "low", "close"]].copy(),
        "M3": _resample_ohlc(m1, "3min"),
        "M5": _resample_ohlc(m1, "5min"),
        "M15": _resample_ohlc(m1, "15min"),
        "D": _resample_ohlc(m1, "1D"),
        "W": _resample_ohlc(m1, "1W"),
    }


@dataclass
class PointerState:
    M3: int = 0
    M5: int = 0
    M15: int = 0
    D: int = 0
    W: int = 0


def _advance_pointer(frame: pd.DataFrame, ptr: int, t: pd.Timestamp) -> int:
    times = frame["time"]
    n = len(frame)
    while ptr < n and times.iloc[ptr] <= t:
        ptr += 1
    return ptr


def _snapshot(frames: dict[str, pd.DataFrame], idx: int, ptrs: PointerState) -> dict[str, pd.DataFrame]:
    t = frames["M1"].iloc[idx]["time"]
    ptrs.M3 = _advance_pointer(frames["M3"], ptrs.M3, t)
    ptrs.M5 = _advance_pointer(frames["M5"], ptrs.M5, t)
    ptrs.M15 = _advance_pointer(frames["M15"], ptrs.M15, t)
    ptrs.D = _advance_pointer(frames["D"], ptrs.D, t)
    ptrs.W = _advance_pointer(frames["W"], ptrs.W, t)

    out = {
        "M1": frames["M1"].iloc[max(0, idx + 1 - DEFAULT_M1_WINDOW) : idx + 1],
        "M3": frames["M3"].iloc[max(0, ptrs.M3 - DEFAULT_M3_WINDOW) : ptrs.M3],
        "M5": frames["M5"].iloc[max(0, ptrs.M5 - DEFAULT_M5_WINDOW) : ptrs.M5],
        "M15": frames["M15"].iloc[max(0, ptrs.M15 - DEFAULT_M15_WINDOW) : ptrs.M15],
        "D": frames["D"].iloc[max(0, ptrs.D - DEFAULT_D_WINDOW) : ptrs.D],
        "W": frames["W"].iloc[max(0, ptrs.W - DEFAULT_W_WINDOW) : ptrs.W],
    }
    return out


_SIM_NOW_UTC: pd.Timestamp | None = None


def _historical_session_flag_now(trading_hours: dict[str, Any]) -> tuple[bool, str]:
    ts = _SIM_NOW_UTC
    if ts is None:
        return af._session_flag_now(trading_hours)
    hour = int(ts.tz_convert("UTC").hour)
    in_tokyo = hour >= 23 or hour < 9
    in_london = 7 <= hour < 16
    in_ny = 12 <= hour < 21
    allowed = False
    labels: list[str] = []
    if in_tokyo:
        labels.append("tokyo")
        if trading_hours.get("tokyo", True):
            allowed = True
    if in_london:
        labels.append("london")
        if trading_hours.get("london", True):
            allowed = True
    if in_ny:
        labels.append("ny")
        if trading_hours.get("ny", True):
            allowed = True
    return allowed, "/".join(labels) if labels else "off-hours"


@contextmanager
def patched_historical_clock() -> Any:
    original = af._session_flag_now
    af._session_flag_now = _historical_session_flag_now
    try:
        yield
    finally:
        af._session_flag_now = original


def _make_config(aggressiveness: str) -> dict[str, Any]:
    cfg = dict(af.DEFAULT_CONFIG)
    cfg["enabled"] = True
    cfg["mode"] = "shadow"
    cfg["aggressiveness"] = aggressiveness
    cfg["trading_hours"] = {"tokyo": True, "london": True, "ny": True}
    cfg["event_blackout_enabled"] = False
    cfg["repeat_setup_dedupe_enabled"] = False
    return cfg


def _make_runtime() -> dict[str, Any]:
    return {
        "daily_pnl_usd": 0.0,
        "llm_spend_today_usd": 0.0,
        "last_llm_call_utc": None,
        "throttle_until_utc": None,
        "throttle_reason": None,
        "recent_gate_blocks": {},
    }


def _safe_round(value: Any, digits: int = 2) -> Any:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return round(float(value), digits)
    return value


def _pip_move(entry: float, price: float, direction: str) -> float:
    if direction == "buy":
        return (price - entry) / PIP_SIZE
    return (entry - price) / PIP_SIZE


def _horizon_end_idx(ts_ns: np.ndarray, start_idx: int, start_time: pd.Timestamp, minutes: int) -> int:
    end_time = start_time + pd.Timedelta(minutes=int(minutes))
    return int(np.searchsorted(ts_ns, end_time.value, side="right"))


def _price_band_touched(bar_high: float, bar_low: float, level: float, tol_price: float = 0.0) -> bool:
    return bar_low <= level + tol_price and bar_high >= level - tol_price


def _simulate_market_trade(
    ts: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    start_idx: int,
    entry_price: float,
    direction: str,
    tp_pips: float,
    sl_pips: float,
    timeout_min: int,
) -> dict[str, Any]:
    entry_time = pd.Timestamp(ts[start_idx], tz="UTC")
    end_idx = _horizon_end_idx(ts.view("i8"), start_idx, entry_time, timeout_min)
    if end_idx <= start_idx + 1:
        return {"status": "insufficient_forward", "filled": True}
    tp_px = float(tp_pips) * PIP_SIZE
    sl_px = float(sl_pips) * PIP_SIZE

    for j in range(start_idx + 1, end_idx):
        high = float(highs[j])
        low = float(lows[j])
        if direction == "buy":
            hit_tp = high >= entry_price + tp_px
            hit_sl = low <= entry_price - sl_px
        else:
            hit_tp = low <= entry_price - tp_px
            hit_sl = high >= entry_price + sl_px
        if hit_tp and hit_sl:
            return {
                "status": "loss",
                "filled": True,
                "fill_idx": start_idx,
                "exit_idx": j,
                "fill_time_utc": entry_time.isoformat(),
                "exit_time_utc": pd.Timestamp(ts[j], tz="UTC").isoformat(),
                "time_to_fill_min": 0.0,
                "hold_minutes": (pd.Timestamp(ts[j], tz="UTC") - entry_time).total_seconds() / 60.0,
                "pnl_pips": -float(sl_pips),
                "outcome": "loss",
                "exit_reason": "ambiguous_bar_stop",
            }
        if hit_tp:
            return {
                "status": "win",
                "filled": True,
                "fill_idx": start_idx,
                "exit_idx": j,
                "fill_time_utc": entry_time.isoformat(),
                "exit_time_utc": pd.Timestamp(ts[j], tz="UTC").isoformat(),
                "time_to_fill_min": 0.0,
                "hold_minutes": (pd.Timestamp(ts[j], tz="UTC") - entry_time).total_seconds() / 60.0,
                "pnl_pips": float(tp_pips),
                "outcome": "win",
                "exit_reason": "tp",
            }
        if hit_sl:
            return {
                "status": "loss",
                "filled": True,
                "fill_idx": start_idx,
                "exit_idx": j,
                "fill_time_utc": entry_time.isoformat(),
                "exit_time_utc": pd.Timestamp(ts[j], tz="UTC").isoformat(),
                "time_to_fill_min": 0.0,
                "hold_minutes": (pd.Timestamp(ts[j], tz="UTC") - entry_time).total_seconds() / 60.0,
                "pnl_pips": -float(sl_pips),
                "outcome": "loss",
                "exit_reason": "sl",
            }
    exit_idx = end_idx - 1
    exit_time = pd.Timestamp(ts[exit_idx], tz="UTC")
    pnl_pips = _pip_move(entry_price, float(closes[exit_idx]), direction)
    return {
        "status": "scratch",
        "filled": True,
        "fill_idx": start_idx,
        "exit_idx": exit_idx,
        "fill_time_utc": entry_time.isoformat(),
        "exit_time_utc": exit_time.isoformat(),
        "time_to_fill_min": 0.0,
        "hold_minutes": (exit_time - entry_time).total_seconds() / 60.0,
        "pnl_pips": float(pnl_pips),
        "outcome": "scratch",
        "exit_reason": "timeout",
    }


def _simulate_limit_trade(
    ts: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    start_idx: int,
    entry_level: float,
    direction: str,
    tp_pips: float,
    sl_pips: float,
    timeout_min: int,
    fill_window_min: int,
) -> dict[str, Any]:
    start_time = pd.Timestamp(ts[start_idx], tz="UTC")
    fill_end_idx = _horizon_end_idx(ts.view("i8"), start_idx, start_time, fill_window_min)
    if fill_end_idx <= start_idx + 1:
        return {"status": "no_fill", "filled": False}
    fill_idx: int | None = None
    for j in range(start_idx + 1, fill_end_idx):
        if _price_band_touched(float(highs[j]), float(lows[j]), float(entry_level)):
            fill_idx = j
            break
    if fill_idx is None:
        return {"status": "no_fill", "filled": False}

    fill_time = pd.Timestamp(ts[fill_idx], tz="UTC")
    post = _simulate_market_trade(
        ts=ts,
        highs=highs,
        lows=lows,
        closes=closes,
        start_idx=fill_idx,
        entry_price=float(entry_level),
        direction=direction,
        tp_pips=tp_pips,
        sl_pips=sl_pips,
        timeout_min=max(1, timeout_min - int((fill_time - start_time).total_seconds() / 60.0)),
    )
    if post.get("status") == "insufficient_forward":
        return {"status": "insufficient_forward", "filled": True}
    post["fill_idx"] = fill_idx
    post["fill_time_utc"] = fill_time.isoformat()
    post["time_to_fill_min"] = (fill_time - start_time).total_seconds() / 60.0
    return post


def _window_directional_stats(
    ts_ns: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    idx: int,
    direction: str,
    minutes: int,
) -> tuple[float | None, float | None, float | None]:
    start_time = pd.Timestamp(ts_ns[idx], tz="UTC")
    end_idx = _horizon_end_idx(ts_ns, idx, start_time, minutes)
    if end_idx <= idx + 1:
        return None, None, None
    entry = float(closes[idx])
    future_high = highs[idx + 1 : end_idx]
    future_low = lows[idx + 1 : end_idx]
    end_close = float(closes[end_idx - 1])
    if direction == "buy":
        mfe = np.max((future_high - entry) / PIP_SIZE)
        mae = np.max((entry - future_low) / PIP_SIZE)
    else:
        mfe = np.max((entry - future_low) / PIP_SIZE)
        mae = np.max((future_high - entry) / PIP_SIZE)
    net = _pip_move(entry, end_close, direction)
    return float(mfe), float(mae), float(net)


def _derive_compression_metrics(data_by_tf: dict[str, pd.DataFrame], extras: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    m1 = data_by_tf.get("M1")
    if m1 is None or len(m1) < 25:
        return out
    close = m1["close"].astype(float)
    high = m1["high"].astype(float)
    low = m1["low"].astype(float)
    upper, middle, lower = bollinger_bands(close, period=20, std_dev=2.0)
    last_close = float(close.iloc[-1])
    bb_mid = float(middle.iloc[-1]) if len(middle) else math.nan
    if math.isfinite(bb_mid) and abs(bb_mid) > 1e-9:
        bandwidth = float((upper.iloc[-1] - lower.iloc[-1]) / bb_mid)
        out["bollinger_bandwidth"] = bandwidth

    window = max(4, int(thresholds.get("compression_window_bars") or 12))
    m5_atr_pips = af._atr_pips(data_by_tf, "M5", period=14, pip_size=PIP_SIZE) or 0.0
    cap_pips = max(1.5, float(m5_atr_pips) * float(thresholds.get("compression_range_atr_mult") or 0.9))

    squeeze_duration = 0
    max_lookback = min(len(m1) - window + 1, 240)
    for offset in range(max_lookback):
        start = len(m1) - window - offset
        if start < 0:
            break
        seg = m1.iloc[start : start + window]
        seg_range_pips = (float(seg["high"].max()) - float(seg["low"].min())) / PIP_SIZE
        if seg_range_pips <= cap_pips:
            squeeze_duration += 1
        else:
            break
    out["squeeze_duration_bars"] = squeeze_duration

    recent = m1.tail(window)
    if not recent.empty:
        box_high = float(recent["high"].max())
        box_low = float(recent["low"].min())
        bias = str(extras.get("trigger_bias") or "")
        out["breakout_level"] = box_high if bias == "buy" else box_low
        out["breakout_direction"] = bias
        bar_range_pips = (float(high.iloc[-1]) - float(low.iloc[-1])) / PIP_SIZE
        m1_atr_pips = af._atr_pips(data_by_tf, "M1", period=14, pip_size=PIP_SIZE)
        if isinstance(m1_atr_pips, (int, float)) and m1_atr_pips > 0:
            out["breakout_bar_atr_multiple"] = bar_range_pips / float(m1_atr_pips)
    return out


def replay_gate(df: pd.DataFrame, aggressiveness: str, progress_every: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames = build_frames(df)
    cfg = _make_config(aggressiveness)
    rt = _make_runtime()
    ptrs = PointerState()
    decisions: list[dict[str, Any]] = []
    wakes: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    block_counts: Counter[str] = Counter()

    with patched_historical_clock():
        for idx in range(len(frames["M1"])):
            if progress_every > 0 and idx and idx % progress_every == 0:
                print(f"[gate] processed {idx:,}/{len(frames['M1']):,} bars", flush=True)

            bar = frames["M1"].iloc[idx]
            _set_sim_now(bar["time"])
            snapshot = _snapshot(frames, idx, ptrs)
            if len(snapshot["M15"]) < 40 or len(snapshot["M5"]) < 40 or len(snapshot["M3"]) < 25 or len(snapshot["D"]) < 2 or len(snapshot["W"]) < 2:
                continue
            spread_pips = float(df.iloc[idx]["spread_pips"])
            inputs = af.GateInputs(
                spread_pips=spread_pips,
                tick_mid=float(bar["close"]),
                open_ai_trade_count=0,
                data_by_tf=snapshot,
                ntz_active=False,
                suggestions_db_path=None,
                upcoming_events=None,
            )
            decision = af.evaluate_gate(cfg, rt, inputs, now_utc=bar["time"].to_pydatetime())
            decisions.append(decision.to_dict())
            if decision.result != "pass":
                block_counts[str(decision.reason)] += 1
                continue

            reason_counts[str((decision.extras or {}).get("trigger_reason") or "unknown")] += 1
            rt["last_llm_call_utc"] = bar["time"].isoformat()

            extras = dict(decision.extras or {})
            session_internal = str(extras.get("session") or af._classify_session_label_from_hour(int(bar["time"].hour)))
            wake: dict[str, Any] = {
                "bar_index": idx,
                "timestamp_utc": bar["time"].isoformat(),
                "price": float(bar["close"]),
                "bid": float(bar["close"]) - (spread_pips * PIP_SIZE / 2.0),
                "ask": float(bar["close"]) + (spread_pips * PIP_SIZE / 2.0),
                "spread_pips": spread_pips,
                "session_internal": session_internal,
                "session": _report_session_label(session_internal),
                "trigger_family": extras.get("trigger_family"),
                "trigger_reason": extras.get("trigger_reason"),
                "direction": extras.get("trigger_bias"),
                "trigger_level_label": extras.get("trigger_level_label"),
                "trigger_level_price": extras.get("trigger_level_price"),
                "nearest_level_pips": extras.get("nearest_level_pips"),
                "overhead_level_label": extras.get("overhead_level_label"),
                "underfoot_level_label": extras.get("underfoot_level_label"),
                "overhead_level_pips": extras.get("overhead_level_pips"),
                "underfoot_level_pips": extras.get("underfoot_level_pips"),
                "m3": extras.get("m3"),
                "m1": extras.get("m1"),
                "m5": extras.get("m5"),
                "m5_atr_pips": extras.get("m5_atr_pips"),
                "adx_m5": extras.get("adx_m5"),
                "adx_m15": extras.get("adx_m15"),
                "extension_pips": extras.get("extension_pips"),
                "extension_limit_pips": extras.get("extension_limit_pips"),
                "compression_range_pips": extras.get("compression_range_pips"),
                "compression_cap_pips": extras.get("compression_cap_pips"),
                "micro_confirmation": extras.get("trigger_micro_confirmation"),
            }
            if wake["trigger_family"] == "compression_breakout":
                thresholds = af._resolve_gate_thresholds(aggressiveness, session_internal)
                wake.update(_derive_compression_metrics(snapshot, extras, thresholds))
            wakes.append(wake)

    metadata = {
        "bars_processed": int(len(df)),
        "wake_events": int(len(wakes)),
        "top_trigger_reasons": reason_counts.most_common(10),
        "block_reason_counts": dict(block_counts.most_common(20)),
        "data_start_utc": df["time"].iloc[0].isoformat(),
        "data_end_utc": df["time"].iloc[-1].isoformat(),
    }
    return pd.DataFrame(wakes), metadata


def _set_sim_now(ts: pd.Timestamp) -> None:
    global _SIM_NOW_UTC
    _SIM_NOW_UTC = ts


def annotate_wakes(df: pd.DataFrame, wakes: pd.DataFrame, aggressiveness: str, tp_pips: float, sl_pips: float, timeout_min: int, limit_fill_min: int) -> pd.DataFrame:
    if wakes.empty:
        out = wakes.copy()
        for col in (
            "squeeze_duration_bars",
            "bollinger_bandwidth",
            "breakout_bar_atr_multiple",
            "breakout_direction",
            "breakout_level",
            "false_breakout",
            "retest_of_breakout_level",
            "expansion_followed",
        ):
            if col not in out.columns:
                out[col] = pd.Series(dtype="object")
        return out
    ts = df["time"].to_numpy(dtype="datetime64[ns]")
    ts_ns = ts.view("i8")
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    thresholds_by_session: dict[str, dict[str, Any]] = {}

    records: list[dict[str, Any]] = []
    for row in wakes.to_dict(orient="records"):
        idx = int(row["bar_index"])
        direction = str(row.get("direction") or "")
        rec = dict(row)
        end60 = _horizon_end_idx(ts_ns, idx, pd.Timestamp(ts[idx], tz="UTC"), timeout_min)
        rec["forward_complete"] = bool(end60 > idx + 1)

        for minutes in (5, 10, 15, 30, 60):
            mfe, mae, net = _window_directional_stats(ts_ns, highs, lows, closes, idx, direction, minutes)
            rec[f"mfe_{minutes}m_pips"] = _safe_round(mfe, 2) if mfe is not None else None
            rec[f"mae_{minutes}m_pips"] = _safe_round(mae, 2) if mae is not None else None
            rec[f"net_{minutes}m_pips"] = _safe_round(net, 2) if net is not None else None

        if rec["forward_complete"]:
            strat_a = _simulate_market_trade(ts, highs, lows, closes, idx, float(row["price"]), direction, tp_pips, sl_pips, timeout_min)
            rec.update(_flatten_strategy("strategy_a", strat_a))
            rec["hit_tp_60m"] = bool(strat_a.get("outcome") == "win")
            rec["hit_sl_60m"] = bool(strat_a.get("outcome") == "loss")
            rec["outcome_60m"] = strat_a.get("outcome")
        else:
            rec.update(_flatten_strategy("strategy_a", {"status": "insufficient_forward", "filled": True}))
            rec["hit_tp_60m"] = False
            rec["hit_sl_60m"] = False
            rec["outcome_60m"] = "insufficient_forward"

        if row["trigger_family"] == "critical_level_reaction" and isinstance(row.get("trigger_level_price"), (int, float)) and rec["forward_complete"]:
            strat_b = _simulate_limit_trade(ts, highs, lows, closes, idx, float(row["trigger_level_price"]), direction, tp_pips, sl_pips, timeout_min, limit_fill_min)
        elif row["trigger_family"] in {"trend_expansion", "compression_breakout"} and rec["forward_complete"]:
            strat_b = dict(strat_a)
        else:
            strat_b = {"status": "n/a", "filled": False}
        rec.update(_flatten_strategy("strategy_b", strat_b))

        if row["trigger_family"] == "compression_breakout" and isinstance(row.get("breakout_level"), (int, float)) and rec["forward_complete"]:
            strat_c = _simulate_limit_trade(ts, highs, lows, closes, idx, float(row["breakout_level"]), direction, tp_pips, sl_pips, timeout_min, limit_fill_min)
        elif row["trigger_family"] == "compression_breakout" and rec["forward_complete"]:
            strat_c = {"status": "n/a", "filled": False}
        else:
            strat_c = {"status": "n/a", "filled": False}
        rec.update(_flatten_strategy("strategy_c", strat_c))

        if row["trigger_family"] == "compression_breakout":
            breakout_level = row.get("breakout_level")
            false_breakout = False
            retest = False
            if isinstance(breakout_level, (int, float)):
                end15 = _horizon_end_idx(ts_ns, idx, pd.Timestamp(ts[idx], tz="UTC"), DEFAULT_FALSE_BREAKOUT_MIN)
                end10 = _horizon_end_idx(ts_ns, idx, pd.Timestamp(ts[idx], tz="UTC"), DEFAULT_LIMIT_FILL_MIN)
                tol = DEFAULT_RETEST_TOLERANCE_PIPS * PIP_SIZE
                for j in range(idx + 1, end15):
                    if direction == "buy" and float(closes[j]) <= float(breakout_level):
                        false_breakout = True
                        break
                    if direction == "sell" and float(closes[j]) >= float(breakout_level):
                        false_breakout = True
                        break
                for j in range(idx + 1, end10):
                    if _price_band_touched(float(highs[j]), float(lows[j]), float(breakout_level), tol):
                        retest = True
                        break
            rec["false_breakout"] = false_breakout
            rec["retest_of_breakout_level"] = retest
        else:
            rec["false_breakout"] = None
            rec["retest_of_breakout_level"] = None

        session_key = str(row.get("session_internal") or "off-hours")
        if session_key not in thresholds_by_session:
            thresholds_by_session[session_key] = af._resolve_gate_thresholds(aggressiveness, session_key)
        rec["gate_thresholds_json"] = json.dumps(thresholds_by_session[session_key], sort_keys=True)
        records.append(rec)

    annotated = pd.DataFrame(records)
    annotated = _annotate_trigger_overlap(annotated)
    for col in (
        "squeeze_duration_bars",
        "bollinger_bandwidth",
        "breakout_bar_atr_multiple",
        "breakout_direction",
        "breakout_level",
        "false_breakout",
        "retest_of_breakout_level",
        "expansion_followed",
    ):
        if col not in annotated.columns:
            annotated[col] = None
    return annotated


def _flatten_strategy(prefix: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_status": result.get("status"),
        f"{prefix}_filled": result.get("filled"),
        f"{prefix}_pnl_pips": _safe_round(result.get("pnl_pips"), 2) if result.get("pnl_pips") is not None else None,
        f"{prefix}_outcome": result.get("outcome"),
        f"{prefix}_exit_reason": result.get("exit_reason"),
        f"{prefix}_fill_time_utc": result.get("fill_time_utc"),
        f"{prefix}_exit_time_utc": result.get("exit_time_utc"),
        f"{prefix}_hold_minutes": _safe_round(result.get("hold_minutes"), 2) if result.get("hold_minutes") is not None else None,
        f"{prefix}_time_to_fill_min": _safe_round(result.get("time_to_fill_min"), 2) if result.get("time_to_fill_min") is not None else None,
    }


def _annotate_trigger_overlap(wakes: pd.DataFrame) -> pd.DataFrame:
    if wakes.empty:
        return wakes
    wakes = wakes.sort_values("timestamp_utc").reset_index(drop=True).copy()
    wakes["timestamp_dt"] = pd.to_datetime(wakes["timestamp_utc"], utc=True)
    ts = wakes["timestamp_dt"].to_numpy(dtype="datetime64[ns]")
    ts_ns = ts.view("i8")
    expansion_followed: list[bool | None] = []

    for i, row in wakes.iterrows():
        if row["trigger_family"] != "compression_breakout":
            expansion_followed.append(None)
            continue
        limit_ns = (row["timestamp_dt"] + pd.Timedelta(minutes=DEFAULT_EXPANSION_FOLLOW_MIN)).value
        j = i + 1
        found = False
        while j < len(wakes) and ts_ns[j] <= limit_ns:
            other = wakes.iloc[j]
            if other["trigger_family"] == "trend_expansion" and other["direction"] == row["direction"]:
                found = True
                break
            j += 1
        expansion_followed.append(found)
    wakes["expansion_followed"] = expansion_followed
    return wakes.drop(columns=["timestamp_dt"])


def _rate(numerator: int | float, denominator: int | float) -> float | None:
    if not denominator:
        return None
    return float(numerator) / float(denominator)


def _pnl_summary(df: pd.DataFrame, prefix: str) -> dict[str, Any]:
    status_col = f"{prefix}_status"
    pnl_col = f"{prefix}_pnl_pips"
    fill_col = f"{prefix}_filled"
    fill_time_col = f"{prefix}_time_to_fill_min"
    subset = df.copy()
    fills = subset.loc[subset[fill_col] == True].copy()
    pnl = pd.to_numeric(fills[pnl_col], errors="coerce").dropna()
    wins = pnl.loc[pnl > 0]
    losses = pnl.loc[pnl < 0]
    gross_win = float(wins.sum()) if not wins.empty else 0.0
    gross_loss = abs(float(losses.sum())) if not losses.empty else 0.0
    pf = None
    if gross_loss > 0:
        pf = gross_win / gross_loss
    elif gross_win > 0:
        pf = 999.0
    fill_rate = _rate(int((subset[fill_col] == True).sum()), len(subset))
    no_fill_rate = _rate(int((subset[status_col] == "no_fill").sum()), len(subset))
    return {
        "trades": int(len(fills)),
        "fill_rate": fill_rate,
        "no_fill_rate": no_fill_rate,
        "win_rate": _rate(int((fills[f"{prefix}_outcome"] == "win").sum()), len(fills)),
        "avg_pnl_pips": float(pnl.mean()) if not pnl.empty else None,
        "median_pnl_pips": float(pnl.median()) if not pnl.empty else None,
        "total_pnl_pips": float(pnl.sum()) if not pnl.empty else 0.0,
        "profit_factor": pf,
        "avg_time_to_fill_min": float(pd.to_numeric(fills[fill_time_col], errors="coerce").dropna().mean()) if fill_time_col in fills else None,
    }


def _interval_summary(df: pd.DataFrame, prefix: str) -> dict[str, Any]:
    values = pd.to_numeric(df[prefix], errors="coerce").dropna()
    if values.empty:
        return {"mean": None, "median": None}
    return {"mean": float(values.mean()), "median": float(values.median())}


def _bucket_squeeze(value: Any) -> str:
    try:
        x = int(value)
    except Exception:
        return "unknown"
    if x < 10:
        return "short"
    if x <= 30:
        return "medium"
    return "long"


def build_report(df: pd.DataFrame, wakes: pd.DataFrame, metadata: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if wakes.empty:
        return {
            "config": {
                "aggressiveness": args.aggressiveness,
                "tp_pips": args.tp_pips,
                "sl_pips": args.sl_pips,
                "timeout_min": args.timeout_min,
                "limit_fill_min": args.limit_fill_min,
            },
            "overall": metadata,
        }

    wakes = wakes.copy()
    wakes["timestamp_dt"] = pd.to_datetime(wakes["timestamp_utc"], utc=True)
    wakes["date_utc"] = wakes["timestamp_dt"].dt.strftime("%Y-%m-%d")
    wakes["hour_utc"] = wakes["timestamp_dt"].dt.hour
    wakes["day_of_week"] = wakes["timestamp_dt"].dt.day_name()

    start = pd.to_datetime(metadata["data_start_utc"], utc=True)
    end = pd.to_datetime(metadata["data_end_utc"], utc=True)
    total_hours = max((end - start).total_seconds() / 3600.0, 1.0)
    active_days = max((end.normalize() - start.normalize()).days + 1, 1)

    overall = {
        "bars_processed": metadata["bars_processed"],
        "wake_events": int(len(wakes)),
        "wake_rate_per_hour": float(len(wakes) / total_hours),
        "wake_rate_per_day": float(len(wakes) / active_days),
        "wake_rate_by_session": {
            session: {
                "wake_count": int(len(g)),
                "wake_rate_per_day": float(len(g) / active_days),
            }
            for session, g in wakes.groupby("session", dropna=False)
        },
        "trigger_family_breakdown": {
            fam: {
                "count": int(len(g)),
                "pct": float(len(g) / len(wakes)),
            }
            for fam, g in wakes.groupby("trigger_family", dropna=False)
        },
        "top_trigger_reasons": metadata["top_trigger_reasons"],
        "block_reason_counts": metadata["block_reason_counts"],
    }

    per_family: dict[str, Any] = {}
    for family, fam_df in wakes.groupby("trigger_family", dropna=False):
        fam_key = str(family)
        fam_summary: dict[str, Any] = {
            "count": int(len(fam_df)),
            "strategy_a": _pnl_summary(fam_df, "strategy_a"),
            "strategy_b": _pnl_summary(fam_df, "strategy_b"),
            "strategy_c": _pnl_summary(fam_df, "strategy_c"),
        }
        for minutes in (5, 10, 15, 30, 60):
            fam_summary[f"mfe_{minutes}m"] = _interval_summary(fam_df, f"mfe_{minutes}m_pips")
            fam_summary[f"mae_{minutes}m"] = _interval_summary(fam_df, f"mae_{minutes}m_pips")
            fam_summary[f"net_{minutes}m"] = _interval_summary(fam_df, f"net_{minutes}m_pips")
        per_family[fam_key] = fam_summary

    compression_df = wakes.loc[wakes["trigger_family"] == "compression_breakout"].copy()
    compression: dict[str, Any] = {}
    if not compression_df.empty:
        comp_sessions = {}
        for session, g in compression_df.groupby("session", dropna=False):
            comp_sessions[str(session)] = {
                "count": int(len(g)),
                "false_breakout_rate": _rate(int(g["false_breakout"].fillna(False).sum()), len(g)),
            }
        squeeze_bucket = compression_df["squeeze_duration_bars"].map(_bucket_squeeze)
        compression = {
            "false_breakout_rate": _rate(int(compression_df["false_breakout"].fillna(False).sum()), len(compression_df)),
            "expansion_follow_through_rate": _rate(int(compression_df["expansion_followed"].fillna(False).sum()), len(compression_df)),
            "retest_rate": _rate(int(compression_df["retest_of_breakout_level"].fillna(False).sum()), len(compression_df)),
            "strategy_market": _pnl_summary(compression_df, "strategy_a"),
            "strategy_pullback_limit": _pnl_summary(compression_df, "strategy_c"),
            "avg_squeeze_duration_bars": float(pd.to_numeric(compression_df["squeeze_duration_bars"], errors="coerce").dropna().mean()),
            "squeeze_duration_distribution": {
                "mean": float(pd.to_numeric(compression_df["squeeze_duration_bars"], errors="coerce").dropna().mean()),
                "median": float(pd.to_numeric(compression_df["squeeze_duration_bars"], errors="coerce").dropna().median()),
            },
            "breakout_bar_atr_multiple_distribution": _quantile_summary(pd.to_numeric(compression_df["breakout_bar_atr_multiple"], errors="coerce").dropna()),
            "false_breakout_rate_by_session": comp_sessions,
            "win_rate_by_squeeze_duration_bucket": {
                bucket: _rate(int((g["strategy_a_outcome"] == "win").sum()), int((g["strategy_a_filled"] == True).sum()))
                for bucket, g in compression_df.groupby(squeeze_bucket, dropna=False)
            },
        }

    session_family: dict[str, Any] = {}
    grouped = wakes.groupby(["session", "trigger_family"], dropna=False)
    for (session, family), g in grouped:
        session_family[f"{session}__{family}"] = {
            "wake_count": int(len(g)),
            "strategy_a_win_rate": _rate(int((g["strategy_a_outcome"] == "win").sum()), int((g["strategy_a_filled"] == True).sum())),
            "avg_mfe_15m_pips": float(pd.to_numeric(g["mfe_15m_pips"], errors="coerce").dropna().mean()) if g["mfe_15m_pips"].notna().any() else None,
            "avg_mae_15m_pips": float(pd.to_numeric(g["mae_15m_pips"], errors="coerce").dropna().mean()) if g["mae_15m_pips"].notna().any() else None,
            "avg_strategy_a_pnl_pips": float(pd.to_numeric(g["strategy_a_pnl_pips"], errors="coerce").dropna().mean()) if g["strategy_a_pnl_pips"].notna().any() else None,
            "false_breakout_rate": _rate(int(g["false_breakout"].fillna(False).sum()), len(g)) if family == "compression_breakout" else None,
        }

    sequencing = _build_sequencing_metrics(wakes)
    worst_best = _build_extremes(wakes)
    histograms = _build_histograms(wakes)

    report = {
        "config": {
            "aggressiveness": args.aggressiveness,
            "tp_pips": args.tp_pips,
            "sl_pips": args.sl_pips,
            "timeout_min": args.timeout_min,
            "limit_fill_min": args.limit_fill_min,
            "data_start_utc": metadata["data_start_utc"],
            "data_end_utc": metadata["data_end_utc"],
        },
        "overall": overall,
        "per_trigger_family": per_family,
        "compression_breakout": compression,
        "session_x_trigger_family": session_family,
        "trigger_family_overlap_and_sequencing": sequencing,
        "worst_wake_analysis": worst_best,
        "time_distribution": histograms,
    }
    return report


def _quantile_summary(series: pd.Series) -> dict[str, Any]:
    if series.empty:
        return {"mean": None, "median": None, "p25": None, "p75": None}
    return {
        "mean": float(series.mean()),
        "median": float(series.median()),
        "p25": float(series.quantile(0.25)),
        "p75": float(series.quantile(0.75)),
    }


def _build_sequencing_metrics(wakes: pd.DataFrame) -> dict[str, Any]:
    if wakes.empty:
        return {}
    df = wakes.sort_values("timestamp_utc").copy()
    df["timestamp_dt"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    rows = df.to_dict(orient="records")
    comp_before_trend_same_side = 0
    comp_before_critical_opp_side = 0
    multiple_within_5m = 0

    for i, row in enumerate(rows):
        t = row["timestamp_dt"]
        fam = row["trigger_family"]
        side = row["direction"]
        seen_multi = False
        for j in range(i + 1, len(rows)):
            other = rows[j]
            delta_min = (other["timestamp_dt"] - t).total_seconds() / 60.0
            if delta_min > 30:
                break
            if delta_min <= 5 and other["trigger_family"] != fam:
                seen_multi = True
            if fam == "compression_breakout" and other["trigger_family"] == "trend_expansion" and other["direction"] == side:
                comp_before_trend_same_side += 1
                break
        if seen_multi:
            multiple_within_5m += 1

    for i, row in enumerate(rows):
        if row["trigger_family"] != "compression_breakout":
            continue
        t = row["timestamp_dt"]
        side = row["direction"]
        for j in range(i + 1, len(rows)):
            other = rows[j]
            delta_min = (other["timestamp_dt"] - t).total_seconds() / 60.0
            if delta_min > 30:
                break
            if other["trigger_family"] == "critical_level_reaction" and other["direction"] != side:
                comp_before_critical_opp_side += 1
                break

    return {
        "compression_breakout_before_trend_expansion_same_side_30m": comp_before_trend_same_side,
        "compression_breakout_before_critical_level_reaction_opposite_side_30m": comp_before_critical_opp_side,
        "multiple_trigger_families_within_5m": multiple_within_5m,
    }


def _build_extremes(wakes: pd.DataFrame) -> dict[str, Any]:
    base_cols = [
        "timestamp_utc",
        "trigger_family",
        "trigger_reason",
        "session",
        "price",
        "direction",
        "mfe_15m_pips",
        "mae_15m_pips",
        "outcome_60m",
        "strategy_a_pnl_pips",
        "trigger_level_label",
        "trigger_level_price",
    ]
    worst = wakes.dropna(subset=["mae_15m_pips"]).sort_values("mae_15m_pips", ascending=False).head(20)
    best = wakes.dropna(subset=["mfe_15m_pips"]).sort_values("mfe_15m_pips", ascending=False).head(20)
    comp = wakes.loc[wakes["trigger_family"] == "compression_breakout"].dropna(subset=["mae_15m_pips"]).sort_values("mae_15m_pips", ascending=False).head(10)
    comp_cols = base_cols + ["squeeze_duration_bars", "bollinger_bandwidth", "breakout_bar_atr_multiple", "false_breakout"]
    worst_cols = [c for c in base_cols if c in worst.columns]
    best_cols = [c for c in base_cols if c in best.columns]
    comp_cols = [c for c in comp_cols if c in comp.columns]
    return {
        "worst_by_mae_15m": worst[worst_cols].to_dict(orient="records"),
        "best_by_mfe_15m": best[best_cols].to_dict(orient="records"),
        "worst_compression_breakouts": comp[comp_cols].to_dict(orient="records") if comp_cols else [],
    }


def _build_histograms(wakes: pd.DataFrame) -> dict[str, Any]:
    hour_all = wakes.groupby("hour_utc").size().to_dict()
    dow_all = wakes.groupby("day_of_week").size().to_dict()
    by_family_hour: dict[str, dict[str, int]] = {}
    by_family_dow: dict[str, dict[str, int]] = {}
    for fam, g in wakes.groupby("trigger_family", dropna=False):
        by_family_hour[str(fam)] = {str(int(k)): int(v) for k, v in g.groupby("hour_utc").size().to_dict().items()}
        by_family_dow[str(fam)] = {str(k): int(v) for k, v in g.groupby("day_of_week").size().to_dict().items()}
    return {
        "wakes_by_hour_utc": {str(int(k)): int(v) for k, v in hour_all.items()},
        "wakes_by_day_of_week": {str(k): int(v) for k, v in dow_all.items()},
        "wakes_by_hour_utc_by_trigger_family": by_family_hour,
        "wakes_by_day_of_week_by_trigger_family": by_family_dow,
    }


def build_summary_text(report: dict[str, Any], wakes: pd.DataFrame) -> str:
    lines: list[str] = []
    overall = report.get("overall") or {}
    lines.append("Gate Backtest Summary")
    lines.append("====================")
    lines.append(f"Bars processed: {overall.get('bars_processed', 0):,}")
    lines.append(f"Wake events: {overall.get('wake_events', 0):,}")
    rate_hr = overall.get("wake_rate_per_hour")
    rate_day = overall.get("wake_rate_per_day")
    lines.append(f"Wake rate: {rate_hr:.2f}/hour, {rate_day:.2f}/day" if isinstance(rate_hr, (int, float)) and isinstance(rate_day, (int, float)) else "Wake rate: n/a")
    lines.append("")
    lines.append("Trigger family breakdown:")
    for fam, bucket in (overall.get("trigger_family_breakdown") or {}).items():
        count = bucket.get("count")
        pct = bucket.get("pct")
        lines.append(f"- {fam}: {count} ({pct:.1%})" if isinstance(pct, (int, float)) else f"- {fam}: {count}")
    lines.append("")
    lines.append("Top trigger reasons:")
    for reason, count in overall.get("top_trigger_reasons") or []:
        lines.append(f"- {reason}: {count}")
    lines.append("")
    lines.append("Per trigger family Strategy A:")
    for fam, bucket in (report.get("per_trigger_family") or {}).items():
        strat = bucket.get("strategy_a") or {}
        wr = strat.get("win_rate")
        avg = strat.get("avg_pnl_pips")
        total = strat.get("total_pnl_pips")
        pf = strat.get("profit_factor")
        lines.append(
            f"- {fam}: trades={strat.get('trades', 0)} "
            f"win_rate={wr:.1%} avg_pnl={avg:.2f}p total_pnl={total:.1f}p pf={pf:.2f}"
            if all(isinstance(v, (int, float)) for v in (wr, avg, total)) and isinstance(pf, (int, float))
            else f"- {fam}: trades={strat.get('trades', 0)}"
        )
    compression = report.get("compression_breakout") or {}
    if compression:
        lines.append("")
        lines.append("Compression breakout diagnostics:")
        for label, value in [
            ("False breakout rate", compression.get("false_breakout_rate")),
            ("Expansion follow-through rate", compression.get("expansion_follow_through_rate")),
            ("Retest rate", compression.get("retest_rate")),
        ]:
            lines.append(f"- {label}: {value:.1%}" if isinstance(value, (int, float)) else f"- {label}: n/a")
    worst = (report.get("worst_wake_analysis") or {}).get("worst_by_mae_15m") or []
    if worst:
        lines.append("")
        lines.append("Worst wake sample:")
        sample = worst[0]
        lines.append(
            f"- {sample.get('timestamp_utc')} {sample.get('trigger_family')} {sample.get('trigger_reason')} "
            f"{sample.get('session')} MAE15={sample.get('mae_15m_pips')}p outcome={sample.get('outcome_60m')}"
        )
    lines.append("")
    lines.append(f"CSV rows written: {len(wakes):,}")
    return "\n".join(lines).strip() + "\n"


def run_sensitivity(df: pd.DataFrame, base_args: argparse.Namespace) -> dict[str, Any]:
    tail = df.tail(int(base_args.sensitivity_max_bars)).copy()
    if tail.empty:
        return {}
    mode = base_args.aggressiveness
    base = dict(af.GATE_THRESHOLDS.get(mode) or af.GATE_THRESHOLDS["balanced"])
    specs: list[tuple[str, list[tuple[str, Any]]]] = [
        ("trend_adx_min", [("default_minus_5", max(1.0, float(base.get("trend_adx_min", 20.0)) - 5.0)), ("default", float(base.get("trend_adx_min", 20.0))), ("default_plus_5", float(base.get("trend_adx_min", 20.0)) + 5.0)]),
        ("require_min_m5_atr_pips", [("x0.8", float(base.get("require_min_m5_atr_pips", 3.0)) * 0.8), ("default", float(base.get("require_min_m5_atr_pips", 3.0))), ("x1.2", float(base.get("require_min_m5_atr_pips", 3.0)) * 1.2)]),
        ("critical_level_max_pips", [("x0.75", float(base.get("critical_level_max_pips", 6.0)) * 0.75), ("default", float(base.get("critical_level_max_pips", 6.0))), ("x1.25", float(base.get("critical_level_max_pips", 6.0)) * 1.25)]),
        ("trend_extension_atr_mult", [("x0.8", float(base.get("trend_extension_atr_mult", 1.0)) * 0.8), ("default", float(base.get("trend_extension_atr_mult", 1.0))), ("x1.2", float(base.get("trend_extension_atr_mult", 1.0)) * 1.2)]),
        ("compression_range_atr_mult", [("x0.8", float(base.get("compression_range_atr_mult", 0.9)) * 0.8), ("default", float(base.get("compression_range_atr_mult", 0.9))), ("x1.2", float(base.get("compression_range_atr_mult", 0.9)) * 1.2)]),
    ]
    unavailable = {
        "bollinger_bandwidth_squeeze_threshold": "not wired into the live gate yet",
        "minimum_squeeze_duration_bars": "not wired into the live gate yet",
        "breakout_bar_min_atr_multiple": "not wired into the live gate yet",
    }
    out: dict[str, Any] = {"available": {}, "unavailable": unavailable}
    for param, settings in specs:
        rows: list[dict[str, Any]] = []
        for label, value in settings:
            original_base = af.GATE_THRESHOLDS[mode].get(param)
            override_snapshot = {k: dict(v) for k, v in af.SESSION_GATE_OVERRIDES.items()}
            af.GATE_THRESHOLDS[mode][param] = value
            for key, bucket in af.SESSION_GATE_OVERRIDES.items():
                if key[0] == mode and param in bucket:
                    bucket[param] = value
            try:
                wakes, meta = replay_gate(tail, mode, progress_every=0)
                annotated = annotate_wakes(tail, wakes, mode, base_args.tp_pips, base_args.sl_pips, base_args.timeout_min, base_args.limit_fill_min)
                strat = _pnl_summary(annotated, "strategy_a")
                rows.append({
                    "setting": label,
                    "value": value,
                    "wake_count": int(len(annotated)),
                    "win_rate": strat.get("win_rate"),
                    "avg_pnl_pips": strat.get("avg_pnl_pips"),
                    "total_pnl_pips": strat.get("total_pnl_pips"),
                })
            finally:
                if original_base is None:
                    af.GATE_THRESHOLDS[mode].pop(param, None)
                else:
                    af.GATE_THRESHOLDS[mode][param] = original_base
                af.SESSION_GATE_OVERRIDES.clear()
                af.SESSION_GATE_OVERRIDES.update(override_snapshot)
        out["available"][param] = rows
    return out


def write_outputs(report: dict[str, Any], wakes: pd.DataFrame, args: argparse.Namespace, summary_text: str) -> None:
    report_path = Path(args.output_report)
    wakes_path = Path(args.output_wakes)
    compression_path = Path(args.output_compression)
    summary_path = Path(args.output_summary)

    report_path.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    wakes.to_csv(wakes_path, index=False)
    if "trigger_family" in wakes.columns:
        wakes.loc[wakes["trigger_family"] == "compression_breakout"].to_csv(compression_path, index=False)
    else:
        pd.DataFrame().to_csv(compression_path, index=False)
    summary_path.write_text(summary_text, encoding="utf-8")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return str(obj)


def main() -> int:
    args = parse_args()
    paths = [Path(p) for p in (args.data or [])] or _discover_default_data_files()
    print(f"[load] using {len(paths)} dataset(s):", flush=True)
    for p in paths:
        print(f"  - {p}", flush=True)
    df = load_history(paths, args.assume_spread_pips, args.start, args.end, args.max_bars)
    print(f"[load] {len(df):,} M1 bars from {df['time'].iloc[0]} to {df['time'].iloc[-1]}", flush=True)

    wakes, meta = replay_gate(df, args.aggressiveness, args.progress_every)
    print(f"[gate] wake events: {len(wakes):,}", flush=True)
    annotated = annotate_wakes(df, wakes, args.aggressiveness, args.tp_pips, args.sl_pips, args.timeout_min, args.limit_fill_min)
    report = build_report(df, annotated, meta, args)
    if args.with_sensitivity:
        print("[sensitivity] running compact sweep", flush=True)
        report["calibration_sensitivity"] = run_sensitivity(df, args)
    summary_text = build_summary_text(report, annotated)
    write_outputs(report, annotated, args, summary_text)
    print(summary_text, end="")
    print(f"[write] report -> {args.output_report}", flush=True)
    print(f"[write] wakes -> {args.output_wakes}", flush=True)
    print(f"[write] compression -> {args.output_compression}", flush=True)
    print(f"[write] summary -> {args.output_summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
