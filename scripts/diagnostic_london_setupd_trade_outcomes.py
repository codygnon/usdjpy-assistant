#!/usr/bin/env python3
"""
London Setup D trade outcome simulator for surviving shadow candidates.

Takes the surviving candidates from the shadow state simulation
(diagnostic_london_setupd_shadow_state.py) and simulates full trade
lifecycle: entry on next-bar open, SL/TP1/TP2 with partial close,
breakeven move, hard close at NY open.

This is research/diagnostic only -- no live code changes.
"""
from __future__ import annotations

import argparse
import json
import pickle
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.london_v2_entry_evaluator import evaluate_london_setup_d
from core.ownership_table import cell_key, der_bucket, er_bucket
from core.phase3_integrated_engine import classify_session, load_phase3_sizing_config
from core.regime_features import compute_delta_efficiency_ratio, compute_efficiency_ratio
from core.regime_classifier import RegimeThresholds
from scripts import backtest_v2_multisetup_london as london_v2
from scripts import validate_regime_classifier as regime_validation

PIP_SIZE = 0.01
OUT_DIR = ROOT / "research_out"
DEFAULT_OUTPUT = str(OUT_DIR / "diagnostic_london_setupd_trade_outcomes.json")
BAR_FRAME_CACHE_VERSION = "v1"


# ── DST helpers (US rules for NY open) ──────────────────────────────────────

def _nth_sunday(year: int, month: int, n: int) -> pd.Timestamp:
    d = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    while d.weekday() != 6:
        d += pd.Timedelta(days=1)
    d += pd.Timedelta(days=(n - 1) * 7)
    return d


def _us_dst_active(day: pd.Timestamp) -> bool:
    """US DST: second Sunday of March to first Sunday of November."""
    y = day.year
    dst_start = _nth_sunday(y, 3, 2).normalize()
    dst_end = _nth_sunday(y, 11, 1).normalize()
    d = day.normalize()
    return dst_start <= d < dst_end


def _ny_open_utc_hour(day: pd.Timestamp) -> int:
    """NY open: 12 UTC during US DST, 13 UTC otherwise."""
    return 12 if _us_dst_active(day) else 13


# ── Config & data loading (reused from shadow state script) ─────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="London Setup D trade outcome simulator")
    p.add_argument("--dataset", nargs="+", default=[
        str(OUT_DIR / "USDJPY_M1_OANDA_500k.csv"),
        str(OUT_DIR / "USDJPY_M1_OANDA_1000k.csv"),
    ])
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--spread-pips", type=float, default=0.3)
    p.add_argument("--sl-buffer-pips", type=float, default=3.0)
    p.add_argument("--sl-min-pips", type=float, default=5.0)
    p.add_argument("--sl-max-pips", type=float, default=20.0)
    p.add_argument("--tp1-r-multiple", type=float, default=1.0)
    p.add_argument("--tp2-r-multiple", type=float, default=2.0)
    p.add_argument("--tp1-close-fraction", type=float, default=0.5)
    p.add_argument("--be-offset-pips", type=float, default=1.0)
    return p.parse_args()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return str(obj)


def _dataset_key(path: str) -> str:
    name = Path(path).name
    if "500k" in name:
        return "500k"
    if "1000k" in name:
        return "1000k"
    return name


def _dataset_cache_dir(dataset: str) -> Path:
    cache_dir = Path(dataset).resolve().parent / "shadow_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _load_bar_frame(dataset: str) -> pd.DataFrame:
    """Load M1 bar frame with regime + ER/DER features (cached)."""
    cache_path = _dataset_cache_dir(dataset) / f"bar_frame_{Path(dataset).stem}_{BAR_FRAME_CACHE_VERSION}.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as fh:
            cached = pickle.load(fh)
        if isinstance(cached, pd.DataFrame):
            return cached

    m1 = regime_validation._load_m1(dataset)
    featured = regime_validation.compute_features(m1)
    classified = regime_validation.classify_all_bars(featured, RegimeThresholds())

    m5 = (
        m1.set_index("time")
        .resample("5min", label="right", closed="right")
        .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
        .dropna()
        .reset_index()
    )

    er_vals: list[float] = []
    der_vals: list[float] = []
    for i in range(len(m5)):
        window = m5.iloc[: i + 1]
        er_vals.append(float(compute_efficiency_ratio(window, lookback=12)))
        der_vals.append(float(compute_delta_efficiency_ratio(window, lookback=12, delta_bars=3)))
    m5["sf_er"] = er_vals
    m5["delta_er"] = der_vals

    merged = pd.merge_asof(
        classified.sort_values("time"),
        m5[["time", "sf_er", "delta_er"]].sort_values("time"),
        on="time",
        direction="backward",
    )
    merged["sf_er"] = merged["sf_er"].fillna(0.5)
    merged["delta_er"] = merged["delta_er"].fillna(0.0)
    with cache_path.open("wb") as fh:
        pickle.dump(merged, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return merged


def _load_london_config() -> dict[str, Any]:
    cfg_path = OUT_DIR / "v2_exp4_winner_baseline_config.json"
    with cfg_path.open() as f:
        return json.load(f)


# ── Trade simulation ────────────────────────────────────────────────────────

def simulate_trade(
    *,
    direction: str,
    signal_bar_idx: int,
    m1_day: pd.DataFrame,
    lor_high: float,
    lor_low: float,
    ny_open_time: pd.Timestamp,
    spread_pips: float,
    sl_buffer_pips: float,
    sl_min_pips: float,
    sl_max_pips: float,
    tp1_r_multiple: float,
    tp2_r_multiple: float,
    tp1_close_fraction: float,
    be_offset_pips: float,
) -> dict[str, Any] | None:
    """
    Simulate a single trade from signal bar to exit.

    Returns None if no next-bar data available for entry.
    """
    half_spread = spread_pips * PIP_SIZE / 2.0

    # Entry bar is the next M1 bar after signal
    entry_bar_idx = signal_bar_idx + 1
    if entry_bar_idx >= len(m1_day):
        return None

    entry_bar = m1_day.iloc[entry_bar_idx]
    entry_time = pd.Timestamp(entry_bar["time"])
    entry_open = float(entry_bar["open"])

    is_long = direction == "long"

    # Entry price with spread
    if is_long:
        entry_price = entry_open + half_spread  # buy at ask
    else:
        entry_price = entry_open - half_spread  # sell at bid

    # Stop loss
    if is_long:
        raw_sl = lor_low - sl_buffer_pips * PIP_SIZE
        sl_dist_pips = (entry_price - raw_sl) / PIP_SIZE
    else:
        raw_sl = lor_high + sl_buffer_pips * PIP_SIZE
        sl_dist_pips = (raw_sl - entry_price) / PIP_SIZE

    # Clamp SL distance
    sl_dist_pips = max(sl_min_pips, min(sl_max_pips, sl_dist_pips))

    if is_long:
        sl_price = entry_price - sl_dist_pips * PIP_SIZE
    else:
        sl_price = entry_price + sl_dist_pips * PIP_SIZE

    risk_pips = sl_dist_pips

    # Take profit levels
    if is_long:
        tp1_price = entry_price + tp1_r_multiple * risk_pips * PIP_SIZE
        tp2_price = entry_price + tp2_r_multiple * risk_pips * PIP_SIZE
    else:
        tp1_price = entry_price - tp1_r_multiple * risk_pips * PIP_SIZE
        tp2_price = entry_price - tp2_r_multiple * risk_pips * PIP_SIZE

    # BE price (after TP1)
    if is_long:
        be_price = entry_price + be_offset_pips * PIP_SIZE
    else:
        be_price = entry_price - be_offset_pips * PIP_SIZE

    # Bar-by-bar simulation
    tp1_hit = False
    exit_price = None
    exit_time = None
    exit_reason = None
    mfe_pips = 0.0
    mae_pips = 0.0

    # Track per-leg PnL
    leg1_exit_price = None  # first 50% (or 100% if SL before TP1)
    leg2_exit_price = None  # second 50% (after TP1)

    sim_start = entry_bar_idx + 1  # start checking from bar after entry

    for bar_i in range(sim_start, len(m1_day)):
        bar = m1_day.iloc[bar_i]
        bar_time = pd.Timestamp(bar["time"])
        bar_open = float(bar["open"])
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])

        # Hard close at NY open
        if bar_time >= ny_open_time:
            if is_long:
                close_px = bar_open - half_spread  # sell at bid
            else:
                close_px = bar_open + half_spread  # buy at ask

            if tp1_hit:
                leg2_exit_price = close_px
                exit_reason = "TP1_PARTIAL_THEN_HARD_CLOSE"
            else:
                leg1_exit_price = close_px
                exit_reason = "HARD_CLOSE"
            exit_price = close_px
            exit_time = bar_time
            break

        # Update MFE/MAE
        if is_long:
            fav = (bar_high - entry_price) / PIP_SIZE
            adv = (entry_price - bar_low) / PIP_SIZE
        else:
            fav = (entry_price - bar_low) / PIP_SIZE
            adv = (bar_high - entry_price) / PIP_SIZE
        mfe_pips = max(mfe_pips, fav)
        mae_pips = max(mae_pips, adv)

        current_sl = be_price if tp1_hit else sl_price

        # Check SL
        sl_hit_this_bar = False
        if is_long and bar_low <= current_sl:
            sl_hit_this_bar = True
        elif not is_long and bar_high >= current_sl:
            sl_hit_this_bar = True

        # Check TP
        tp_hit_this_bar = False
        tp_level_hit = None
        if not tp1_hit:
            # Check TP1
            if is_long and bar_high >= tp1_price:
                tp_hit_this_bar = True
                tp_level_hit = "TP1"
            elif not is_long and bar_low <= tp1_price:
                tp_hit_this_bar = True
                tp_level_hit = "TP1"
        else:
            # Check TP2 (after TP1 already hit)
            if is_long and bar_high >= tp2_price:
                tp_hit_this_bar = True
                tp_level_hit = "TP2"
            elif not is_long and bar_low <= tp2_price:
                tp_hit_this_bar = True
                tp_level_hit = "TP2"

        # Resolve conflicts: both SL and TP on same bar
        if sl_hit_this_bar and tp_hit_this_bar:
            # Which is closer to open?
            if is_long:
                sl_dist_from_open = abs(bar_open - current_sl)
                if tp_level_hit == "TP1":
                    tp_dist_from_open = abs(tp1_price - bar_open)
                else:
                    tp_dist_from_open = abs(tp2_price - bar_open)
            else:
                sl_dist_from_open = abs(current_sl - bar_open)
                if tp_level_hit == "TP1":
                    tp_dist_from_open = abs(bar_open - tp1_price)
                else:
                    tp_dist_from_open = abs(bar_open - tp2_price)

            if sl_dist_from_open <= tp_dist_from_open:
                tp_hit_this_bar = False  # SL wins
            else:
                sl_hit_this_bar = False  # TP wins

        # Process TP hit
        if tp_hit_this_bar and tp_level_hit == "TP1":
            tp1_hit = True
            leg1_exit_price = tp1_price
            # Continue tracking for TP2 / BE stop
            # SL is now BE
            continue

        if tp_hit_this_bar and tp_level_hit == "TP2":
            leg2_exit_price = tp2_price
            exit_price = tp2_price
            exit_time = bar_time
            exit_reason = "TP1_PARTIAL_THEN_TP2"
            break

        # Process SL hit
        if sl_hit_this_bar:
            if tp1_hit:
                leg2_exit_price = be_price
                exit_price = be_price
                exit_time = bar_time
                exit_reason = "TP1_PARTIAL_THEN_BE"
            else:
                leg1_exit_price = sl_price
                exit_price = sl_price
                exit_time = bar_time
                exit_reason = "SL_FULL"
            break

    # If we ran out of bars without an exit (shouldn't happen if NY open is in data)
    if exit_price is None:
        last_bar = m1_day.iloc[-1]
        bar_close = float(last_bar["close"])
        if is_long:
            exit_price = bar_close - half_spread
        else:
            exit_price = bar_close + half_spread
        exit_time = pd.Timestamp(last_bar["time"])
        if tp1_hit:
            leg2_exit_price = exit_price
            exit_reason = "TP1_PARTIAL_THEN_HARD_CLOSE"
        else:
            leg1_exit_price = exit_price
            exit_reason = "HARD_CLOSE"

    # Calculate PnL
    if tp1_hit:
        # Two legs
        if is_long:
            pnl_leg1 = (leg1_exit_price - entry_price) / PIP_SIZE
            pnl_leg2 = (leg2_exit_price - entry_price) / PIP_SIZE
        else:
            pnl_leg1 = (entry_price - leg1_exit_price) / PIP_SIZE
            pnl_leg2 = (entry_price - leg2_exit_price) / PIP_SIZE
        total_pnl = tp1_close_fraction * pnl_leg1 + (1.0 - tp1_close_fraction) * pnl_leg2
    else:
        # Single leg (full position)
        if is_long:
            pnl_leg1 = (leg1_exit_price - entry_price) / PIP_SIZE
        else:
            pnl_leg1 = (entry_price - leg1_exit_price) / PIP_SIZE
        pnl_leg2 = None
        total_pnl = pnl_leg1

    duration_minutes = (exit_time - entry_time).total_seconds() / 60.0

    return {
        "entry_time": entry_time,
        "exit_time": exit_time,
        "direction": direction,
        "entry_price": round(entry_price, 5),
        "exit_price": round(exit_price, 5),
        "sl_price": round(sl_price, 5),
        "tp1_price": round(tp1_price, 5),
        "tp2_price": round(tp2_price, 5),
        "be_price": round(be_price, 5),
        "risk_pips": round(risk_pips, 2),
        "pnl_pips": round(total_pnl, 2),
        "pnl_pips_leg1": round(pnl_leg1, 2),
        "pnl_pips_leg2": round(pnl_leg2, 2) if pnl_leg2 is not None else None,
        "exit_reason": exit_reason,
        "mfe_pips": round(mfe_pips, 2),
        "mae_pips": round(mae_pips, 2),
        "trade_duration_minutes": round(duration_minutes, 1),
    }


# ── Main simulation loop ───────────────────────────────────────────────────

def run_dataset(dataset: str, args: argparse.Namespace) -> dict[str, Any]:
    """Run shadow state + trade simulation on one dataset."""
    print(f"\n{'='*70}")
    print(f"Processing {Path(dataset).name}")
    print(f"{'='*70}")

    cfg = _load_london_config()
    cfg_d = cfg["setups"]["D"]
    sizing_cfg = load_phase3_sizing_config()

    # Native directional constraints
    native_allow_long = bool(cfg_d.get("allow_long", True))
    native_allow_short = bool(cfg_d.get("allow_short", False))

    # Load raw M1
    m1_raw = regime_validation._load_m1(dataset)
    m1_raw["time"] = pd.to_datetime(m1_raw["time"], utc=True, errors="coerce")
    m1_raw = m1_raw.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    # Load bar frame with regime features
    bars = _load_bar_frame(dataset)
    bars["time"] = pd.to_datetime(bars["time"], utc=True, errors="coerce")
    bars = bars.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    # Config ranges
    asian_min = float(cfg["levels"]["asian_range_min_pips"])
    asian_max = float(cfg["levels"]["asian_range_max_pips"])
    lor_min = float(cfg["levels"]["lor_range_min_pips"])
    lor_max = float(cfg["levels"]["lor_range_max_pips"])

    # Group by day
    m1_raw["day_utc"] = m1_raw["time"].dt.floor("D")
    bars["day_utc"] = bars["time"].dt.floor("D")

    m1_by_day = {k: g.copy().reset_index(drop=True) for k, g in m1_raw.groupby("day_utc")}
    bars_by_day = {k: g.copy().reset_index(drop=True) for k, g in bars.groupby("day_utc")}

    # Trade results collection
    all_trades: list[dict[str, Any]] = []

    all_days = sorted(m1_by_day.keys())

    for day in all_days:
        day_m1 = m1_by_day[day]
        if day_m1.empty:
            continue

        # Session windows
        london_h = london_v2.uk_london_open_utc(day)
        ny_h = london_v2.us_ny_open_utc(day)
        london_open = day + pd.Timedelta(hours=london_h)
        ny_open = day + pd.Timedelta(hours=ny_h)

        # NY open for hard close (using US DST rules)
        ny_open_hard_close_h = _ny_open_utc_hour(day)
        ny_open_hard_close = day + pd.Timedelta(hours=ny_open_hard_close_h)

        # Asian range
        asian = day_m1[(day_m1["time"] >= day) & (day_m1["time"] < london_open)]
        if asian.empty:
            continue
        asian_high = float(asian["high"].max())
        asian_low = float(asian["low"].min())
        asian_range_pips = (asian_high - asian_low) / PIP_SIZE
        asian_valid = asian_min <= asian_range_pips <= asian_max

        # LOR range (first 15 min of London)
        lor_end_time = london_open + pd.Timedelta(minutes=15)
        lor = day_m1[(day_m1["time"] >= london_open) & (day_m1["time"] < lor_end_time)]
        if lor.empty:
            continue
        lor_high = float(lor["high"].max())
        lor_low = float(lor["low"].min())
        lor_range_pips = (lor_high - lor_low) / PIP_SIZE
        lor_valid = lor_min <= lor_range_pips <= lor_max

        if not lor_valid:
            continue

        # Setup D window
        d_start = london_open + pd.Timedelta(minutes=int(cfg_d["entry_start_min_after_london"]))
        d_end_raw = london_open + pd.Timedelta(minutes=int(cfg_d["entry_end_min_after_london"]))
        d_end = min(d_end_raw, ny_open)

        # Channel state per direction (realistic simulation)
        channel_state = {"long": "ARMED", "short": "ARMED"}
        channel_entries = {"long": 0, "short": 0}
        pending_direction: str | None = None

        # Get the day's bars from the featured bar frame
        day_bars = bars_by_day.get(day)
        if day_bars is None or day_bars.empty:
            continue

        # Filter to Setup D window bars from raw M1
        window_bars = day_m1[(day_m1["time"] >= d_start) & (day_m1["time"] < d_end)]
        if window_bars.empty:
            continue

        # We need all bars from entry through NY open for trade simulation
        # Use the full day_m1 for that, indexed by position in day_m1

        for idx in range(len(window_bars)):
            row = window_bars.iloc[idx]
            ts = pd.Timestamp(row["time"])

            # ---- Process pending entry from previous bar ----
            if pending_direction is not None:
                channel_state[pending_direction] = "FIRED"
                channel_entries[pending_direction] += 1
                pending_direction = None

            # ---- Channel reset logic ----
            for direction in ["long", "short"]:
                if channel_state[direction] != "WAITING_RESET":
                    continue
                if direction == "long" and float(row["low"]) <= lor_high:
                    channel_state[direction] = "ARMED"
                if direction == "short" and float(row["high"]) >= lor_low:
                    channel_state[direction] = "ARMED"

            # ---- Evaluate Setup D (both directions, ignoring allow_short for shadow) ----
            if idx + 1 >= len(window_bars):
                continue
            nxt_ts = pd.Timestamp(window_bars.iloc[idx + 1]["time"])

            # Get regime cell
            bar_match = day_bars[day_bars["time"] == ts]
            if bar_match.empty:
                bar_match = day_bars.iloc[(day_bars["time"] - ts).abs().argsort()[:1]]

            regime_label = str(bar_match.iloc[0].get("regime_hysteresis", "ambiguous")) if not bar_match.empty else "ambiguous"
            sf_er = float(bar_match.iloc[0].get("sf_er", 0.5)) if not bar_match.empty else 0.5
            delta_er = float(bar_match.iloc[0].get("delta_er", 0.0)) if not bar_match.empty else 0.0
            cell = cell_key(regime_label, er_bucket(sf_er), der_bucket(delta_er))

            # Evaluate with realistic channel state but allow_short=True for shadow
            # We override cfg_d temporarily to allow both directions
            cfg_d_shadow = dict(cfg_d)
            cfg_d_shadow["allow_long"] = True
            cfg_d_shadow["allow_short"] = True

            realistic_results = evaluate_london_setup_d(
                row=row,
                cfg_d=cfg_d_shadow,
                lor_high=lor_high,
                lor_low=lor_low,
                asian_range_pips=asian_range_pips if asian_valid else None,
                lor_range_pips=lor_range_pips,
                lor_valid=lor_valid,
                asian_valid=asian_valid,
                ts=ts,
                nxt_ts=nxt_ts,
                d_start=d_start,
                d_end=d_end,
                channel_long_state=channel_state["long"],
                channel_short_state=channel_state["short"],
                channel_long_entries=channel_entries["long"],
                channel_short_entries=channel_entries["short"],
            )

            if not realistic_results:
                continue

            for entry_sig in realistic_results:
                direction = entry_sig["direction"]

                # Find the signal bar's absolute index in day_m1
                signal_abs_idx_matches = day_m1.index[day_m1["time"] == ts]
                if len(signal_abs_idx_matches) == 0:
                    continue
                signal_abs_idx = signal_abs_idx_matches[0]

                # Simulate trade
                trade = simulate_trade(
                    direction=direction,
                    signal_bar_idx=signal_abs_idx,
                    m1_day=day_m1,
                    lor_high=lor_high,
                    lor_low=lor_low,
                    ny_open_time=ny_open_hard_close,
                    spread_pips=args.spread_pips,
                    sl_buffer_pips=args.sl_buffer_pips,
                    sl_min_pips=args.sl_min_pips,
                    sl_max_pips=args.sl_max_pips,
                    tp1_r_multiple=args.tp1_r_multiple,
                    tp2_r_multiple=args.tp2_r_multiple,
                    tp1_close_fraction=args.tp1_close_fraction,
                    be_offset_pips=args.be_offset_pips,
                )

                if trade is None:
                    continue

                # Determine native_allowed
                if direction == "long":
                    native_allowed = native_allow_long
                else:
                    native_allowed = native_allow_short

                trade["ownership_cell"] = cell
                trade["native_allowed"] = native_allowed
                trade["signal_time"] = ts
                all_trades.append(trade)

                # State transition: ARMED -> PENDING
                channel_state[direction] = "PENDING"
                pending_direction = direction

            # Simulate exit for FIRED channels -> WAITING_RESET
            for direction in ["long", "short"]:
                if channel_state[direction] == "FIRED":
                    channel_state[direction] = "WAITING_RESET"

    ds_key = _dataset_key(dataset)
    print(f"  Total simulated trades: {len(all_trades)}")

    return _build_report(ds_key, all_trades)


def _build_report(ds_key: str, trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Build comprehensive report from trade list."""

    def _metrics(tlist: list[dict[str, Any]]) -> dict[str, Any]:
        if not tlist:
            return {
                "count": 0, "win_rate": 0, "avg_pnl_pips": 0,
                "median_pnl_pips": 0, "total_pnl_pips": 0,
                "profit_factor": 0, "avg_duration_minutes": 0,
                "avg_mfe_pips": 0, "avg_mae_pips": 0,
            }
        pnls = [t["pnl_pips"] for t in tlist]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        pf = round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf") if gross_profit > 0 else 0

        return {
            "count": len(tlist),
            "win_rate": round(100.0 * len(wins) / len(tlist), 2),
            "avg_pnl_pips": round(statistics.mean(pnls), 2),
            "median_pnl_pips": round(statistics.median(pnls), 2),
            "total_pnl_pips": round(sum(pnls), 2),
            "profit_factor": pf,
            "avg_duration_minutes": round(statistics.mean([t["trade_duration_minutes"] for t in tlist]), 1),
            "avg_mfe_pips": round(statistics.mean([t["mfe_pips"] for t in tlist]), 2),
            "avg_mae_pips": round(statistics.mean([t["mae_pips"] for t in tlist]), 2),
        }

    def _exit_dist(tlist: list[dict[str, Any]]) -> dict[str, int]:
        c: Counter = Counter()
        for t in tlist:
            c[t["exit_reason"]] += 1
        return dict(c)

    # ── dataset_summary ──
    native_trades = [t for t in trades if t["native_allowed"]]
    summary = _metrics(trades)
    summary["total_simulated_trades"] = len(trades)
    summary["total_simulated_native_only"] = len(native_trades)
    summary["exit_reason_distribution"] = _exit_dist(trades)

    # ── by_cell ──
    by_cell: dict[str, Any] = {}
    cell_groups: dict[str, list] = defaultdict(list)
    for t in trades:
        cell_groups[t["ownership_cell"]].append(t)
    for cell_name in sorted(cell_groups.keys()):
        clist = cell_groups[cell_name]
        m = _metrics(clist)
        m["exit_reason_distribution"] = _exit_dist(clist)
        by_cell[cell_name] = m

    # ── by_direction ──
    longs = [t for t in trades if t["direction"] == "long"]
    shorts = [t for t in trades if t["direction"] == "short"]
    longs_native = [t for t in longs if t["native_allowed"]]

    by_direction = {
        "long": _metrics(longs),
        "short": _metrics(shorts),
        "long_native_only": _metrics(longs_native),
    }

    # ── by_exit_reason ──
    exit_groups: dict[str, list] = defaultdict(list)
    for t in trades:
        exit_groups[t["exit_reason"]].append(t)
    by_exit_reason = {}
    for reason in sorted(exit_groups.keys()):
        rlist = exit_groups[reason]
        by_exit_reason[reason] = {
            "count": len(rlist),
            "avg_pnl_pips": round(statistics.mean([t["pnl_pips"] for t in rlist]), 2),
            "total_pnl_pips": round(sum(t["pnl_pips"] for t in rlist), 2),
        }

    # ── samples (first 30) ──
    samples = []
    for t in trades[:30]:
        samples.append({
            "entry_time": t["entry_time"],
            "exit_time": t["exit_time"],
            "signal_time": t["signal_time"],
            "direction": t["direction"],
            "entry_price": t["entry_price"],
            "exit_price": t["exit_price"],
            "sl_price": t["sl_price"],
            "tp1_price": t["tp1_price"],
            "tp2_price": t["tp2_price"],
            "pnl_pips": t["pnl_pips"],
            "pnl_pips_leg1": t["pnl_pips_leg1"],
            "pnl_pips_leg2": t["pnl_pips_leg2"],
            "exit_reason": t["exit_reason"],
            "ownership_cell": t["ownership_cell"],
            "mfe_pips": t["mfe_pips"],
            "mae_pips": t["mae_pips"],
            "trade_duration_minutes": t["trade_duration_minutes"],
            "native_allowed": t["native_allowed"],
        })

    return {
        "dataset_key": ds_key,
        "dataset_summary": summary,
        "by_cell": by_cell,
        "by_direction": by_direction,
        "by_exit_reason": by_exit_reason,
        "samples": samples,
        "_all_trades": trades,  # internal, not written to JSON
    }


def _build_quality_comparison(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare shadow trade quality against expectations."""
    comp = {}
    for r in reports:
        ds = r["dataset_key"]
        s = r["dataset_summary"]
        native_long = r["by_direction"].get("long_native_only", {})
        comp[ds] = {
            "shadow_all_avg_pnl": s.get("avg_pnl_pips", 0),
            "shadow_all_win_rate": s.get("win_rate", 0),
            "shadow_all_profit_factor": s.get("profit_factor", 0),
            "shadow_native_long_avg_pnl": native_long.get("avg_pnl_pips", 0),
            "shadow_native_long_win_rate": native_long.get("win_rate", 0),
            "note": (
                "Defensive stack baseline expectation: ~0.5-1.5 pip avg PnL, "
                "~45-55% win rate. Compare shadow Setup D against this range."
            ),
            "assessment": (
                "BETTER" if native_long.get("avg_pnl_pips", 0) > 1.5
                else "SIMILAR" if native_long.get("avg_pnl_pips", 0) > 0
                else "WORSE"
            ),
        }
    return comp


def _build_verdict(reports: list[dict[str, Any]]) -> str:
    """Build plain-language verdict."""
    lines = []

    for r in reports:
        ds = r["dataset_key"]
        s = r["dataset_summary"]
        native_long = r["by_direction"].get("long_native_only", {})
        lines.append(
            f"{ds}: {s['count']} trades, avg PnL {s['avg_pnl_pips']} pips, "
            f"win rate {s['win_rate']}%, PF {s['profit_factor']}, "
            f"total PnL {s['total_pnl_pips']} pips"
        )
        if native_long.get("count", 0) > 0:
            lines.append(
                f"  Long-only (native): {native_long['count']} trades, "
                f"avg PnL {native_long['avg_pnl_pips']} pips, "
                f"win rate {native_long['win_rate']}%"
            )

    lines.append("")

    # Aggregate assessment
    all_trades_combined = []
    for r in reports:
        all_trades_combined.extend(r["_all_trades"])

    if not all_trades_combined:
        lines.append("VERDICT: No trades simulated. Cannot assess.")
        return "\n".join(lines)

    avg_pnl = statistics.mean([t["pnl_pips"] for t in all_trades_combined])
    long_trades = [t for t in all_trades_combined if t["direction"] == "long" and t["native_allowed"]]
    short_trades = [t for t in all_trades_combined if t["direction"] == "short"]

    if long_trades:
        long_avg = statistics.mean([t["pnl_pips"] for t in long_trades])
    else:
        long_avg = 0
    if short_trades:
        short_avg = statistics.mean([t["pnl_pips"] for t in short_trades])
    else:
        short_avg = 0

    # Q1: Positive expectancy?
    if avg_pnl > 0.5:
        lines.append(
            f"EXPECTANCY: POSITIVE. Surviving Setup D candidates show +{avg_pnl:.2f} pips "
            f"avg PnL across both directions. This is a real signal, not noise."
        )
    elif avg_pnl > -0.5:
        lines.append(
            f"EXPECTANCY: MARGINAL. Avg PnL is {avg_pnl:+.2f} pips -- near breakeven. "
            "After realistic spread, the edge is thin or nonexistent. "
            "The candidate surface is not obviously profitable."
        )
    else:
        lines.append(
            f"EXPECTANCY: NEGATIVE. Avg PnL is {avg_pnl:+.2f} pips. "
            "Surviving Setup D candidates do not have edge under these assumptions. "
            "The state-realistic candidate surface appears to be noise."
        )

    # Q2: Which cells carry edge?
    lines.append("")
    cell_winners = []
    cell_losers = []
    for r in reports:
        for cell_name, cell_data in r["by_cell"].items():
            if cell_data["count"] >= 10:
                if cell_data["avg_pnl_pips"] > 1.0:
                    cell_winners.append((cell_name, cell_data["avg_pnl_pips"], cell_data["count"]))
                elif cell_data["avg_pnl_pips"] < -1.0:
                    cell_losers.append((cell_name, cell_data["avg_pnl_pips"], cell_data["count"]))

    if cell_winners:
        lines.append("CELL EDGE: The following cells show positive edge (avg PnL > 1.0, n >= 10):")
        for cw in sorted(cell_winners, key=lambda x: x[1], reverse=True)[:10]:
            lines.append(f"  {cw[0]}: +{cw[1]:.2f} pips (n={cw[2]})")
    else:
        lines.append("CELL EDGE: No individual cell shows strong positive edge (avg > 1.0 pip, n >= 10).")

    if cell_losers:
        lines.append("CELL DRAG: The following cells are net negative:")
        for cl in sorted(cell_losers, key=lambda x: x[1])[:10]:
            lines.append(f"  {cl[0]}: {cl[1]:.2f} pips (n={cl[2]})")

    # Q3: Long-only constraint justified?
    lines.append("")
    if long_trades and short_trades:
        if long_avg > short_avg + 1.0:
            lines.append(
                f"DIRECTIONAL CONSTRAINT: JUSTIFIED. Long avg {long_avg:+.2f} vs short avg "
                f"{short_avg:+.2f}. The native long-only constraint for Setup D is correct."
            )
        elif abs(long_avg - short_avg) < 1.0:
            lines.append(
                f"DIRECTIONAL CONSTRAINT: QUESTIONABLE. Long avg {long_avg:+.2f} vs short avg "
                f"{short_avg:+.2f}. The difference is small. The long-only constraint may be "
                "leaving edge on the table, or both directions are similarly marginal."
            )
        else:
            lines.append(
                f"DIRECTIONAL CONSTRAINT: SHORT IS BETTER. Long avg {long_avg:+.2f} vs short avg "
                f"{short_avg:+.2f}. The native long-only constraint is actively harmful."
            )
    elif not short_trades:
        lines.append(
            "DIRECTIONAL CONSTRAINT: No short trades survived. Cannot evaluate."
        )

    # Q4: Competitive with defensive stack?
    lines.append("")
    if long_avg > 1.5:
        lines.append(
            "VS DEFENSIVE: Setup D shadow trades OUTPERFORM typical defensive stack "
            f"expectations (long avg {long_avg:+.2f} pips vs defensive ~0.5-1.5 range)."
        )
    elif long_avg > 0:
        lines.append(
            "VS DEFENSIVE: Setup D shadow trades are WITHIN defensive stack range "
            f"(long avg {long_avg:+.2f} pips). Competitive but not a clear upgrade."
        )
    else:
        lines.append(
            "VS DEFENSIVE: Setup D shadow trades UNDERPERFORM defensive stack "
            f"(long avg {long_avg:+.2f} pips). Not competitive."
        )

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    datasets = [str(Path(d).resolve()) for d in args.dataset]
    output = Path(args.output)

    all_reports = []
    for dataset in datasets:
        if not Path(dataset).exists():
            print(f"WARN: dataset {dataset} not found, skipping")
            continue
        report = run_dataset(dataset, args)
        all_reports.append(report)

    if not all_reports:
        print("ERROR: no datasets processed")
        return 1

    quality = _build_quality_comparison(all_reports)
    verdict = _build_verdict(all_reports)

    print(f"\n{'='*70}")
    print("VERDICT:")
    print(verdict)

    # Build combined output (strip _all_trades before writing)
    payload = {
        "config": {
            "spread_pips": args.spread_pips,
            "sl_buffer_pips": args.sl_buffer_pips,
            "sl_min_pips": args.sl_min_pips,
            "sl_max_pips": args.sl_max_pips,
            "tp1_r_multiple": args.tp1_r_multiple,
            "tp2_r_multiple": args.tp2_r_multiple,
            "tp1_close_fraction": args.tp1_close_fraction,
            "be_offset_pips": args.be_offset_pips,
        },
        "dataset_summaries": {
            r["dataset_key"]: r["dataset_summary"] for r in all_reports
        },
        "by_cell": {
            r["dataset_key"]: r["by_cell"] for r in all_reports
        },
        "by_direction": {
            r["dataset_key"]: r["by_direction"] for r in all_reports
        },
        "by_exit_reason": {
            r["dataset_key"]: r["by_exit_reason"] for r in all_reports
        },
        "quality_comparison": quality,
        "samples": {
            r["dataset_key"]: r["samples"] for r in all_reports
        },
        "verdict": verdict,
    }

    output.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    print(f"\nWrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
