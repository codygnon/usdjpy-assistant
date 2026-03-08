#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PIP_SIZE = 0.01
ROUND_UNITS = 100


DEFAULT_CONFIG: dict[str, Any] = {
    "account": {
        "starting_balance": 100000.0,
        "leverage": 50.0,
        "max_margin_usage_fraction_per_trade": 0.5,
        "max_open_positions": 5,
    },
    "execution_model": {
        "spread_mode": "realistic",
        "spread_avg_pips": 1.6,
        "spread_min_pips": 1.0,
        "spread_max_pips": 3.5,
    },
    "session": {
        "active_days_utc": ["Tuesday", "Wednesday", "Thursday"],
        "hard_close_at_ny_open": True,
    },
    "risk": {
        "risk_per_trade_pct": 0.01,
        "max_total_open_risk_pct": 0.05,
    },
    "entry_limits": {
        "max_trades_per_day_total": None,
        "max_trades_per_setup_per_day": None,
        "max_trades_per_setup_direction_per_day": None,
        # If true, channels remain FIRED after exit (no reset/re-entry loop).
        "disable_channel_reset_after_exit": False,
    },
    "levels": {
        "asian_range_min_pips": 30.0,
        "asian_range_max_pips": 60.0,
        "lor_range_min_pips": 4.0,
        "lor_range_max_pips": 20.0,
    },
    "setups": {
        "A": {
            "enabled": True,
            "allow_long": True,
            "allow_short": True,
            "risk_per_trade_pct": 0.01,
            "entry_start_min_after_london": 0,
            "entry_end_min_after_london": 90,
            "breakout_buffer_pips": 7.0,
            "sl_buffer_pips": 3.0,
            "sl_min_pips": 15.0,
            "sl_max_pips": 40.0,
            "tp1_r_multiple": 1.0,
            "tp2_r_multiple": 2.0,
            "tp1_close_fraction": 0.5,
            "be_offset_pips": 1.0,
            "reset_mode": "touch_level",
        },
        "B": {
            "enabled": True,
            "allow_long": True,
            "allow_short": True,
            "risk_per_trade_pct": 0.01,
            "entry_start_min_after_london": 0,
            "entry_end_min_before_ny": 60,
            "confirm_window_candles": 3,
            "reenter_cooldown_minutes": 10,
            "close_back_inside_pips": 3.0,
            "sl_wick_buffer_pips": 5.0,
            "sl_min_pips": 5.0,
            "sl_max_pips": 20.0,
            "tp1_r_multiple": 1.0,
            "tp2_r_multiple": 2.0,
            "tp1_close_fraction": 0.5,
            "be_offset_pips": 1.0,
            "reset_mode": "cooldown",
        },
        "C": {
            "enabled": True,
            "allow_long": True,
            "allow_short": True,
            "risk_per_trade_pct": 0.01,
            "entry_start_min_after_london": 0,
            "entry_end_min_before_ny": 60,
            "reenter_cooldown_minutes": 10,
            "bounce_zone_pips": 5.0,
            "min_close_offset_pips": 3.0,
            "min_body_pips": 2.0,
            "sl_buffer_pips": 5.0,
            "sl_min_pips": 5.0,
            "sl_max_pips": 15.0,
            "tp1_r_multiple": 1.0,
            "tp2_r_multiple": 2.0,
            "tp1_close_fraction": 0.5,
            "be_offset_pips": 1.0,
            "reset_mode": "cooldown",
        },
        "D": {
            "enabled": True,
            "allow_long": True,
            "allow_short": True,
            "risk_per_trade_pct": 0.01,
            "entry_start_min_after_london": 15,
            "entry_end_min_after_london": 120,
            "breakout_buffer_pips": 3.0,
            "sl_buffer_pips": 3.0,
            "sl_min_pips": 5.0,
            "sl_max_pips": 20.0,
            "tp1_r_multiple": 1.0,
            "tp2_r_multiple": 2.0,
            "tp1_close_fraction": 0.5,
            "be_offset_pips": 1.0,
            "reset_mode": "touch_level",
        },
    },
}


@dataclass
class Position:
    trade_id: int
    setup_type: str
    direction: str
    entry_time: pd.Timestamp
    entry_price: float
    sl_price: float
    tp1_price: float
    tp2_price: float
    initial_units: int
    remaining_units: int
    sl_pips: float
    risk_usd_planned: float
    risk_pct: float
    margin_used_pct: float
    margin_required_usd: float
    day_name: str
    asian_range_pips: float | None
    lor_range_pips: float | None
    entry_hour_utc: int
    trade_sequence_number: int
    is_reentry: bool
    tp1_hit: bool = False
    tp1_time: pd.Timestamp | None = None
    tp1_units_closed: int = 0
    be_price_after_tp1: float | None = None
    pnl_usd_realized: float = 0.0
    weighted_pips_sum: float = 0.0
    mfe_pips: float = 0.0
    mae_pips: float = 0.0
    exit_reason: str | None = None
    exit_time: pd.Timestamp | None = None
    exit_price_last: float | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V2 baseline multi-setup London backtest")
    p.add_argument("--input", required=True, help="Input USDJPY M1 CSV")
    p.add_argument("--config", required=True, help="Config JSON")
    p.add_argument("--out-prefix", required=True, help="Output prefix path")
    return p.parse_args()


def merge_config(user_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    for k, v in user_cfg.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
            for kk, vv in v.items():
                if isinstance(vv, dict) and isinstance(cfg[k].get(kk), dict):
                    cfg[k][kk].update(vv)
        else:
            cfg[k] = v
    return cfg


def last_sunday(year: int, month: int) -> pd.Timestamp:
    d = pd.Timestamp(year=year, month=month, day=1, tz="UTC") + pd.offsets.MonthEnd(0)
    while d.weekday() != 6:
        d -= pd.Timedelta(days=1)
    return d


def nth_sunday(year: int, month: int, n: int) -> pd.Timestamp:
    d = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    while d.weekday() != 6:
        d += pd.Timedelta(days=1)
    d += pd.Timedelta(days=(n - 1) * 7)
    return d


def uk_london_open_utc(ts_day: pd.Timestamp) -> int:
    y = ts_day.year
    summer_start = last_sunday(y, 3).normalize()
    summer_end = last_sunday(y, 10).normalize()
    d = ts_day.normalize()
    return 7 if summer_start <= d < summer_end else 8


def us_ny_open_utc(ts_day: pd.Timestamp) -> int:
    y = ts_day.year
    summer_start = nth_sunday(y, 3, 2).normalize()
    summer_end = nth_sunday(y, 11, 1).normalize()
    d = ts_day.normalize()
    return 12 if summer_start <= d < summer_end else 13


def day_name(ts: pd.Timestamp) -> str:
    return ts.day_name()


def ts_hour(ts: pd.Timestamp) -> float:
    return float(ts.hour) + float(ts.minute) / 60.0 + float(ts.second) / 3600.0


def compute_spread_pips(i: int, ts: pd.Timestamp, cfg: dict[str, Any]) -> float:
    mode = str(cfg["execution_model"].get("spread_mode", "realistic")).lower()
    avg = float(cfg["execution_model"].get("spread_avg_pips", 1.6))
    mn = float(cfg["execution_model"].get("spread_min_pips", 1.0))
    mx = float(cfg["execution_model"].get("spread_max_pips", 3.5))
    if mode == "fixed":
        return max(mn, min(mx, avg))
    if mode == "realistic":
        h = ts_hour(ts)
        if 21.0 <= h or h < 0.5:
            base = 2.6
        elif 0.5 <= h < 2.0:
            base = 1.9
        elif 2.0 <= h < 6.0:
            base = 1.5
        elif 6.0 <= h < 9.0:
            base = 1.7
        elif 13.0 <= h < 16.0:
            base = 2.3
        else:
            base = 1.6
        wiggle = 0.18 * math.sin(i * 0.017) + 0.07 * math.sin(i * 0.071)
        return max(mn, min(mx, base + wiggle))
    x = avg + 0.35 * math.sin(i * 0.017) + 0.15 * math.sin(i * 0.071)
    return max(mn, min(mx, x))


def to_bid_ask(mid: float, spread_pips: float) -> tuple[float, float]:
    half = spread_pips * PIP_SIZE / 2.0
    return mid - half, mid + half


def pip_value_per_unit(usd_jpy_rate: float) -> float:
    return PIP_SIZE / max(1e-9, usd_jpy_rate)


def calc_leg_pnl(direction: str, entry: float, exit_px: float, units: int) -> tuple[float, float]:
    if direction == "long":
        pips = (exit_px - entry) / PIP_SIZE
    else:
        pips = (entry - exit_px) / PIP_SIZE
    usd = pips * units * pip_value_per_unit(exit_px)
    return float(pips), float(usd)


def clamp_sl(entry: float, raw_sl: float, direction: str, min_pips: float, max_pips: float) -> tuple[float, float]:
    if direction == "long":
        raw_dist = (entry - raw_sl) / PIP_SIZE
        dist = min(max(raw_dist, min_pips), max_pips)
        sl = entry - dist * PIP_SIZE
    else:
        raw_dist = (raw_sl - entry) / PIP_SIZE
        dist = min(max(raw_dist, min_pips), max_pips)
        sl = entry + dist * PIP_SIZE
    return float(sl), float(dist)


def seq_max(values: list[int]) -> int:
    best = 0
    cur = 0
    for v in values:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def performance_metrics(
    trades: pd.DataFrame,
    equity_curve: pd.DataFrame,
    start_balance: float,
    t0: pd.Timestamp,
    t1: pd.Timestamp,
) -> dict[str, Any]:
    if equity_curve.empty:
        equity_curve = pd.DataFrame({"timestamp": [t0, t1], "equity": [start_balance, start_balance]})
    equity_curve = equity_curve.sort_values("timestamp").copy()
    net = float(trades["pnl_usd"].sum()) if not trades.empty else 0.0
    wins = trades[trades["pnl_usd"] > 0] if not trades.empty else pd.DataFrame()
    losses = trades[trades["pnl_usd"] < 0] if not trades.empty else pd.DataFrame()
    gross_profit = float(wins["pnl_usd"].sum()) if not wins.empty else 0.0
    gross_loss = float(losses["pnl_usd"].sum()) if not losses.empty else 0.0
    pf = float(gross_profit / abs(gross_loss)) if gross_loss < 0 else (float("inf") if gross_profit > 0 else 0.0)
    wr = float((trades["pnl_usd"] > 0).mean() * 100.0) if not trades.empty else 0.0
    expectancy = float(trades["pnl_usd"].mean()) if not trades.empty else 0.0
    avg_win_pips = float(wins["pnl_pips"].mean()) if not wins.empty else 0.0
    avg_loss_pips = float(losses["pnl_pips"].mean()) if not losses.empty else 0.0
    avg_win_usd = float(wins["pnl_usd"].mean()) if not wins.empty else 0.0
    avg_loss_usd = float(losses["pnl_usd"].mean()) if not losses.empty else 0.0

    outcomes = (trades["pnl_usd"] > 0).astype(int).tolist() if not trades.empty else []
    max_wins = seq_max(outcomes)
    max_losses = seq_max([1 - x for x in outcomes])

    peak = equity_curve["equity"].cummax()
    dd = equity_curve["equity"] - peak
    dd_pct = np.where(peak > 0, dd / peak * 100.0, 0.0)
    max_dd_usd = float(-dd.min()) if len(dd) else 0.0
    max_dd_pct = float(-dd_pct.min()) if len(dd_pct) else 0.0

    daily = equity_curve.copy()
    daily["date"] = daily["timestamp"].dt.floor("D")
    daily = daily.groupby("date", as_index=False)["equity"].last()
    daily["ret"] = daily["equity"].pct_change().fillna(0.0)
    mu = float(daily["ret"].mean()) if len(daily) > 1 else 0.0
    sd = float(daily["ret"].std(ddof=0)) if len(daily) > 1 else 0.0
    downside = daily["ret"].where(daily["ret"] < 0, 0.0)
    dsd = float(downside.std(ddof=0)) if len(daily) > 1 else 0.0
    sharpe = (mu / sd) * math.sqrt(252.0) if sd > 0 else 0.0
    sortino = (mu / dsd) * math.sqrt(252.0) if dsd > 0 else 0.0

    span_days = max((t1 - t0).total_seconds() / 86400.0, 1.0)
    years = span_days / 365.25
    end_equity = float(equity_curve.iloc[-1]["equity"])
    ann = ((end_equity / start_balance) ** (1.0 / years) - 1.0) * 100.0 if years > 0 else 0.0
    avg_trades_week = float(len(trades) / (span_days / 7.0)) if span_days > 0 else 0.0
    avg_trades_month = float(len(trades) / (span_days / 30.4375)) if span_days > 0 else 0.0

    return {
        "total_trades": int(len(trades)),
        "win_rate_pct": wr,
        "avg_winner_pips": avg_win_pips,
        "avg_winner_usd": avg_win_usd,
        "avg_loser_pips": avg_loss_pips,
        "avg_loser_usd": avg_loss_usd,
        "profit_factor": pf,
        "expectancy_per_trade_usd": expectancy,
        "max_consecutive_wins": int(max_wins),
        "max_consecutive_losses": int(max_losses),
        "max_drawdown_usd": max_dd_usd,
        "max_drawdown_pct": max_dd_pct,
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "net_pnl_usd": net,
        "net_pnl_pct": float((end_equity - start_balance) / start_balance * 100.0),
        "annualized_return_pct": float(ann),
        "avg_trades_per_week": avg_trades_week,
        "avg_trades_per_month": avg_trades_month,
    }


def run_backtest(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    start_balance = float(cfg["account"]["starting_balance"])
    leverage = float(cfg["account"]["leverage"])
    max_margin_frac = float(cfg["account"]["max_margin_usage_fraction_per_trade"])
    max_open_positions = int(cfg["account"]["max_open_positions"])
    risk_pct = float(cfg["risk"]["risk_per_trade_pct"])
    max_total_open_risk_pct = float(cfg["risk"]["max_total_open_risk_pct"])
    active_days = set(cfg["session"]["active_days_utc"])

    asian_min = float(cfg["levels"]["asian_range_min_pips"])
    asian_max = float(cfg["levels"]["asian_range_max_pips"])
    lor_min = float(cfg["levels"]["lor_range_min_pips"])
    lor_max = float(cfg["levels"]["lor_range_max_pips"])

    trades_out: list[dict[str, Any]] = []
    equity_events: list[tuple[pd.Timestamp, float]] = []
    equity = start_balance
    trade_id = 0
    diagnostics: dict[str, Any] = {
        "days_total": 0,
        "days_active": 0,
        "days_asian_valid": 0,
        "days_lor_valid": 0,
        "trades_skipped_margin_constraint": 0,
        "trades_skipped_open_risk_cap": 0,
        "trades_skipped_entry_limits": 0,
    }

    df = df.copy()
    df["day_utc"] = df["time"].dt.floor("D")
    grouped = {k: g.copy().reset_index(drop=True) for k, g in df.groupby("day_utc")}
    all_days = sorted(grouped.keys())

    for day in all_days:
        diagnostics["days_total"] += 1
        day_df = grouped[day]
        if day_df.empty:
            continue
        if day.day_name() not in active_days:
            continue
        diagnostics["days_active"] += 1

        day_start = day
        london_h = uk_london_open_utc(day)
        ny_h = us_ny_open_utc(day)
        london_open = day_start + pd.Timedelta(hours=london_h)
        ny_open = day_start + pd.Timedelta(hours=ny_h)
        hard_close = ny_open

        if day_df["time"].max() < hard_close:
            continue

        asian = day_df[(day_df["time"] >= day_start) & (day_df["time"] < london_open)]
        if asian.empty:
            continue
        asian_high = float(asian["high"].max())
        asian_low = float(asian["low"].min())
        asian_range_pips = (asian_high - asian_low) / PIP_SIZE
        asian_valid = asian_min <= asian_range_pips <= asian_max
        if asian_valid:
            diagnostics["days_asian_valid"] += 1

        lor_end = london_open + pd.Timedelta(minutes=15)
        lor = day_df[(day_df["time"] >= london_open) & (day_df["time"] < lor_end)]
        lor_high = float(lor["high"].max()) if not lor.empty else np.nan
        lor_low = float(lor["low"].min()) if not lor.empty else np.nan
        lor_range_pips = (lor_high - lor_low) / PIP_SIZE if not lor.empty else np.nan
        lor_valid = (not lor.empty) and (lor_min <= lor_range_pips <= lor_max)
        if lor_valid:
            diagnostics["days_lor_valid"] += 1

        def _window_for_setup(setup_key: str) -> tuple[pd.Timestamp, pd.Timestamp]:
            s = cfg["setups"][setup_key]
            start = london_open + pd.Timedelta(minutes=int(s["entry_start_min_after_london"]))
            if s.get("entry_end_min_before_ny") is not None:
                end = ny_open - pd.Timedelta(minutes=int(s["entry_end_min_before_ny"]))
            else:
                end = london_open + pd.Timedelta(minutes=int(s["entry_end_min_after_london"]))
            if end > ny_open:
                end = ny_open
            return start, end

        windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {
            "A": _window_for_setup("A"),
            "B": _window_for_setup("B"),
            "C": _window_for_setup("C"),
            "D": _window_for_setup("D"),
        }

        channels: dict[tuple[str, str], dict[str, Any]] = {}
        for setup in ["A", "B", "C", "D"]:
            for d in ["long", "short"]:
                channels[(setup, d)] = {
                    "state": "ARMED",
                    "cooldown_until": None,
                    "entries": 0,
                    "resets": 0,
                }

        b_candidates: dict[str, list[dict[str, Any]]] = {"long": [], "short": []}
        pending_entries: list[dict[str, Any]] = []
        open_positions: list[Position] = []
        daily_trade_sequence = 0
        day_entries_total = 0
        day_entries_setup = {"A": 0, "B": 0, "C": 0, "D": 0}
        day_entries_setup_dir: dict[tuple[str, str], int] = {
            ("A", "long"): 0,
            ("A", "short"): 0,
            ("B", "long"): 0,
            ("B", "short"): 0,
            ("C", "long"): 0,
            ("C", "short"): 0,
            ("D", "long"): 0,
            ("D", "short"): 0,
        }

        def rearm_if_pending(setup: str, direction: str) -> None:
            st = channels[(setup, direction)]["state"]
            if st == "PENDING":
                channels[(setup, direction)]["state"] = "ARMED"

        def mark_post_exit(p: Position, ts: pd.Timestamp) -> None:
            c = channels[(p.setup_type, p.direction)]
            if bool(cfg.get("entry_limits", {}).get("disable_channel_reset_after_exit", False)):
                c["state"] = "FIRED"
                c["cooldown_until"] = None
                return
            if p.setup_type in ("B", "C"):
                c["state"] = "WAITING_RESET"
                c["cooldown_until"] = ts + pd.Timedelta(minutes=int(cfg["setups"][p.setup_type]["reenter_cooldown_minutes"]))
            else:
                c["state"] = "WAITING_RESET"
                c["cooldown_until"] = None

        for i in range(len(day_df)):
            row = day_df.iloc[i]
            ts = pd.Timestamp(row["time"])
            sp = compute_spread_pips(i, ts, cfg)
            bid_o, ask_o = to_bid_ask(float(row["open"]), sp)
            bid_h, ask_h = to_bid_ask(float(row["high"]), sp)
            bid_l, ask_l = to_bid_ask(float(row["low"]), sp)

            # execute pending entries
            to_exec = [x for x in pending_entries if x["execute_time"] == ts]
            pending_entries = [x for x in pending_entries if x["execute_time"] != ts]
            for pe in to_exec:
                setup = str(pe["setup_type"])
                direction = str(pe["direction"])
                w_start, w_end = windows[setup]
                if not (w_start <= ts < w_end):
                    rearm_if_pending(setup, direction)
                    continue
                limits = cfg.get("entry_limits", {})
                lim_total = limits.get("max_trades_per_day_total")
                lim_setup = limits.get("max_trades_per_setup_per_day")
                lim_setup_dir = limits.get("max_trades_per_setup_direction_per_day")
                if lim_total is not None and day_entries_total >= int(lim_total):
                    diagnostics["trades_skipped_entry_limits"] += 1
                    rearm_if_pending(setup, direction)
                    continue
                if lim_setup is not None and day_entries_setup[setup] >= int(lim_setup):
                    diagnostics["trades_skipped_entry_limits"] += 1
                    rearm_if_pending(setup, direction)
                    continue
                if lim_setup_dir is not None and day_entries_setup_dir[(setup, direction)] >= int(lim_setup_dir):
                    diagnostics["trades_skipped_entry_limits"] += 1
                    rearm_if_pending(setup, direction)
                    continue
                if len(open_positions) >= max_open_positions:
                    rearm_if_pending(setup, direction)
                    continue

                entry_price = ask_o if direction == "long" else bid_o
                s_cfg = cfg["setups"][setup]
                sl, sl_pips = clamp_sl(
                    entry_price,
                    float(pe["raw_sl"]),
                    direction,
                    float(s_cfg["sl_min_pips"]),
                    float(s_cfg["sl_max_pips"]),
                )
                sign = 1.0 if direction == "long" else -1.0
                tp1 = entry_price + sign * float(s_cfg["tp1_r_multiple"]) * sl_pips * PIP_SIZE
                tp2 = entry_price + sign * float(s_cfg["tp2_r_multiple"]) * sl_pips * PIP_SIZE

                risk_pct_trade = float(s_cfg.get("risk_per_trade_pct", risk_pct))
                risk_usd = equity * risk_pct_trade
                open_risk = float(sum(p.risk_usd_planned for p in open_positions))
                if open_risk + risk_usd > equity * max_total_open_risk_pct:
                    diagnostics["trades_skipped_open_risk_cap"] += 1
                    rearm_if_pending(setup, direction)
                    continue

                units = int(math.floor(risk_usd / max(1e-9, sl_pips * pip_value_per_unit(entry_price)) / ROUND_UNITS) * ROUND_UNITS)
                if units <= 0:
                    rearm_if_pending(setup, direction)
                    continue

                # Use requested margin model:
                # margin = (position_units * USDJPY_rate * 0.01) / leverage
                used_margin = float(sum(p.margin_required_usd for p in open_positions))
                free_margin = max(0.0, equity - used_margin)
                req_margin = (units * float(entry_price) * PIP_SIZE) / leverage
                if req_margin > max_margin_frac * free_margin:
                    diagnostics["trades_skipped_margin_constraint"] += 1
                    rearm_if_pending(setup, direction)
                    continue
                margin_used_pct = (req_margin / max(1e-9, equity)) * 100.0

                trade_id += 1
                daily_trade_sequence += 1
                p = Position(
                    trade_id=trade_id,
                    setup_type=setup,
                    direction=direction,
                    entry_time=ts,
                    entry_price=float(entry_price),
                    sl_price=float(sl),
                    tp1_price=float(tp1),
                    tp2_price=float(tp2),
                    initial_units=units,
                    remaining_units=units,
                    sl_pips=float(sl_pips),
                    risk_usd_planned=float(risk_usd),
                    risk_pct=float(risk_pct_trade),
                    margin_used_pct=float(margin_used_pct),
                    margin_required_usd=float(req_margin),
                    day_name=day_name(ts),
                    asian_range_pips=float(pe["asian_range_pips"]) if pe.get("asian_range_pips") is not None else None,
                    lor_range_pips=float(pe["lor_range_pips"]) if pe.get("lor_range_pips") is not None else None,
                    entry_hour_utc=int(ts.hour),
                    trade_sequence_number=daily_trade_sequence,
                    is_reentry=bool(pe.get("is_reentry", False)),
                )
                close_fraction = float(s_cfg["tp1_close_fraction"])
                tp1_units = int(math.floor(units * close_fraction / ROUND_UNITS) * ROUND_UNITS)
                p.tp1_units_closed = max(0, min(units, tp1_units))
                open_positions.append(p)

                c = channels[(setup, direction)]
                c["state"] = "FIRED"
                c["entries"] += 1
                day_entries_total += 1
                day_entries_setup[setup] += 1
                day_entries_setup_dir[(setup, direction)] += 1

            # hard close at NY open
            if ts == hard_close:
                for p in open_positions:
                    exit_px = bid_o if p.direction == "long" else ask_o
                    u = p.remaining_units
                    lp, lu = calc_leg_pnl(p.direction, p.entry_price, float(exit_px), u)
                    p.pnl_usd_realized += lu
                    p.weighted_pips_sum += lp * u
                    p.remaining_units = 0
                    p.exit_time = ts
                    p.exit_price_last = float(exit_px)
                    p.exit_reason = "TP1_PARTIAL" if p.tp1_hit else "HARD_CLOSE"
                    equity += p.pnl_usd_realized
                    equity_events.append((ts, p.pnl_usd_realized))
                    trades_out.append(
                        {
                            "trade_id": p.trade_id,
                            "setup_type": p.setup_type,
                            "date": p.entry_time.date().isoformat(),
                            "day_of_week": p.day_name,
                            "direction": p.direction,
                            "entry_time_utc": p.entry_time,
                            "exit_time_utc": p.exit_time,
                            "entry_price": p.entry_price,
                            "exit_price": p.exit_price_last,
                            "sl_price": p.sl_price,
                            "tp1_price": p.tp1_price,
                            "tp2_price": p.tp2_price,
                            "sl_pips": p.sl_pips,
                            "risk_pct": p.risk_pct,
                            "pnl_pips": (p.weighted_pips_sum / p.initial_units) if p.initial_units > 0 else 0.0,
                            "pnl_usd": p.pnl_usd_realized,
                            "exit_reason": p.exit_reason,
                            "asian_range_pips": p.asian_range_pips,
                            "lor_range_pips": p.lor_range_pips,
                            "MAE_pips": p.mae_pips,
                            "MFE_pips": p.mfe_pips,
                            "position_units": p.initial_units,
                            "trade_duration_minutes": int((p.exit_time - p.entry_time).total_seconds() / 60.0),
                            "trade_sequence_number": p.trade_sequence_number,
                            "is_reentry": bool(p.is_reentry),
                            "entry_hour_utc": p.entry_hour_utc,
                            "margin_used_pct": p.margin_used_pct,
                        }
                    )
                    mark_post_exit(p, ts)
                open_positions = []
                break

            # manage open positions
            survivors: list[Position] = []
            for p in open_positions:
                if p.direction == "long":
                    p.mfe_pips = max(p.mfe_pips, (float(row["high"]) - p.entry_price) / PIP_SIZE)
                    p.mae_pips = max(p.mae_pips, (p.entry_price - float(row["low"])) / PIP_SIZE)
                    px_open, px_high, px_low = bid_o, bid_h, bid_l
                else:
                    p.mfe_pips = max(p.mfe_pips, (p.entry_price - float(row["low"])) / PIP_SIZE)
                    p.mae_pips = max(p.mae_pips, (float(row["high"]) - p.entry_price) / PIP_SIZE)
                    px_open, px_high, px_low = ask_o, ask_h, ask_l

                def choose_first(level_a: float, level_b: float) -> str:
                    da = abs(px_open - level_a)
                    db = abs(px_open - level_b)
                    return "a" if da <= db else "b"

                def close_remaining(reason: str, exit_px: float) -> None:
                    nonlocal equity
                    u = p.remaining_units
                    lp, lu = calc_leg_pnl(p.direction, p.entry_price, float(exit_px), u)
                    p.pnl_usd_realized += lu
                    p.weighted_pips_sum += lp * u
                    p.remaining_units = 0
                    p.exit_time = ts
                    p.exit_price_last = float(exit_px)
                    p.exit_reason = reason
                    equity += p.pnl_usd_realized
                    equity_events.append((ts, p.pnl_usd_realized))
                    trades_out.append(
                        {
                            "trade_id": p.trade_id,
                            "setup_type": p.setup_type,
                            "date": p.entry_time.date().isoformat(),
                            "day_of_week": p.day_name,
                            "direction": p.direction,
                            "entry_time_utc": p.entry_time,
                            "exit_time_utc": p.exit_time,
                            "entry_price": p.entry_price,
                            "exit_price": p.exit_price_last,
                            "sl_price": p.sl_price,
                            "tp1_price": p.tp1_price,
                            "tp2_price": p.tp2_price,
                            "sl_pips": p.sl_pips,
                            "risk_pct": p.risk_pct,
                            "pnl_pips": (p.weighted_pips_sum / p.initial_units) if p.initial_units > 0 else 0.0,
                            "pnl_usd": p.pnl_usd_realized,
                            "exit_reason": p.exit_reason,
                            "asian_range_pips": p.asian_range_pips,
                            "lor_range_pips": p.lor_range_pips,
                            "MAE_pips": p.mae_pips,
                            "MFE_pips": p.mfe_pips,
                            "position_units": p.initial_units,
                            "trade_duration_minutes": int((p.exit_time - p.entry_time).total_seconds() / 60.0),
                            "trade_sequence_number": p.trade_sequence_number,
                            "is_reentry": bool(p.is_reentry),
                            "entry_hour_utc": p.entry_hour_utc,
                            "margin_used_pct": p.margin_used_pct,
                        }
                    )
                    mark_post_exit(p, ts)

                if not p.tp1_hit:
                    tp1_hit = px_high >= p.tp1_price if p.direction == "long" else px_low <= p.tp1_price
                    sl_hit = px_low <= p.sl_price if p.direction == "long" else px_high >= p.sl_price
                    if tp1_hit and sl_hit:
                        first = choose_first(p.tp1_price, p.sl_price)
                    elif sl_hit:
                        first = "b"
                    elif tp1_hit:
                        first = "a"
                    else:
                        first = ""

                    if first == "b":
                        close_remaining("SL_FULL", p.sl_price)
                        continue
                    if first == "a":
                        u_close = p.tp1_units_closed if p.tp1_units_closed > 0 else int(p.remaining_units / 2)
                        u_close = max(0, min(p.remaining_units, u_close))
                        if u_close > 0:
                            lp, lu = calc_leg_pnl(p.direction, p.entry_price, p.tp1_price, u_close)
                            p.pnl_usd_realized += lu
                            p.weighted_pips_sum += lp * u_close
                            p.remaining_units -= u_close
                        p.tp1_hit = True
                        p.tp1_time = ts
                        be_offset = float(cfg["setups"][p.setup_type]["be_offset_pips"])
                        p.be_price_after_tp1 = p.entry_price + (be_offset * PIP_SIZE if p.direction == "long" else -be_offset * PIP_SIZE)
                        if p.direction == "long":
                            p.sl_price = max(p.sl_price, p.be_price_after_tp1)
                        else:
                            p.sl_price = min(p.sl_price, p.be_price_after_tp1)

                if p.tp1_hit and p.remaining_units > 0:
                    tp2_hit = px_high >= p.tp2_price if p.direction == "long" else px_low <= p.tp2_price
                    sl2_hit = px_low <= p.sl_price if p.direction == "long" else px_high >= p.sl_price
                    if tp2_hit and sl2_hit:
                        first2 = choose_first(p.tp2_price, p.sl_price)
                    elif tp2_hit:
                        first2 = "a"
                    elif sl2_hit:
                        first2 = "b"
                    else:
                        first2 = ""
                    if first2 == "a":
                        close_remaining("TP2_FULL", p.tp2_price)
                        continue
                    if first2 == "b":
                        close_remaining("BE_STOP", p.sl_price)
                        continue

                survivors.append(p)

            open_positions = survivors

            # channel reset logic
            for setup in ["A", "B", "C", "D"]:
                w_start, w_end = windows[setup]
                for direction in ["long", "short"]:
                    c = channels[(setup, direction)]
                    if c["state"] != "WAITING_RESET":
                        continue
                    if setup in ("B", "C"):
                        cooldown_until = c["cooldown_until"]
                        if cooldown_until is not None and ts >= cooldown_until and ts < w_end:
                            c["state"] = "ARMED"
                            c["resets"] += 1
                    else:
                        if ts >= w_end:
                            continue
                        if setup == "A":
                            if direction == "long" and float(row["low"]) <= asian_high:
                                c["state"] = "ARMED"
                                c["resets"] += 1
                            if direction == "short" and float(row["high"]) >= asian_low:
                                c["state"] = "ARMED"
                                c["resets"] += 1
                        if setup == "D" and lor_valid:
                            if direction == "long" and float(row["low"]) <= lor_high:
                                c["state"] = "ARMED"
                                c["resets"] += 1
                            if direction == "short" and float(row["high"]) >= lor_low:
                                c["state"] = "ARMED"
                                c["resets"] += 1

            # signal generation on closed candle
            if i + 1 >= len(day_df):
                continue
            nxt_ts = pd.Timestamp(day_df.iloc[i + 1]["time"])

            # Setup A (breakout)
            if cfg["setups"]["A"]["enabled"] and asian_valid:
                a_start, a_end = windows["A"]
                if a_start <= ts < a_end:
                    long_break = float(row["close"]) > asian_high + float(cfg["setups"]["A"]["breakout_buffer_pips"]) * PIP_SIZE
                    short_break = float(row["close"]) < asian_low - float(cfg["setups"]["A"]["breakout_buffer_pips"]) * PIP_SIZE
                    if long_break and bool(cfg["setups"]["A"].get("allow_long", True)) and channels[("A", "long")]["state"] == "ARMED":
                        pending_entries.append(
                            {
                                "setup_type": "A",
                                "direction": "long",
                                "execute_time": nxt_ts,
                                "raw_sl": asian_low - float(cfg["setups"]["A"]["sl_buffer_pips"]) * PIP_SIZE,
                                "asian_range_pips": asian_range_pips,
                                "lor_range_pips": lor_range_pips if lor_valid else None,
                                "is_reentry": channels[("A", "long")]["entries"] > 0,
                            }
                        )
                        channels[("A", "long")]["state"] = "PENDING"
                    if short_break and bool(cfg["setups"]["A"].get("allow_short", True)) and channels[("A", "short")]["state"] == "ARMED":
                        pending_entries.append(
                            {
                                "setup_type": "A",
                                "direction": "short",
                                "execute_time": nxt_ts,
                                "raw_sl": asian_high + float(cfg["setups"]["A"]["sl_buffer_pips"]) * PIP_SIZE,
                                "asian_range_pips": asian_range_pips,
                                "lor_range_pips": lor_range_pips if lor_valid else None,
                                "is_reentry": channels[("A", "short")]["entries"] > 0,
                            }
                        )
                        channels[("A", "short")]["state"] = "PENDING"

            # Setup B (false breakout reversal)
            if cfg["setups"]["B"]["enabled"] and asian_valid:
                b_start, b_end = windows["B"]
                if b_start <= ts < b_end:
                    # short candidate: upside break then close back inside
                    if float(row["high"]) > asian_high:
                        b_candidates["short"].append({"expire_i": i + 2, "wick_high": float(row["high"])})
                    for cand in b_candidates["short"]:
                        cand["wick_high"] = max(float(cand["wick_high"]), float(row["high"]))
                    short_confirm = float(row["close"]) < asian_high - float(cfg["setups"]["B"]["close_back_inside_pips"]) * PIP_SIZE
                    if short_confirm and bool(cfg["setups"]["B"].get("allow_short", True)) and channels[("B", "short")]["state"] == "ARMED":
                        active = [c for c in b_candidates["short"] if c["expire_i"] >= i]
                        if active:
                            wick_high = max(float(c["wick_high"]) for c in active)
                            pending_entries.append(
                                {
                                    "setup_type": "B",
                                    "direction": "short",
                                    "execute_time": nxt_ts,
                                    "raw_sl": wick_high + float(cfg["setups"]["B"]["sl_wick_buffer_pips"]) * PIP_SIZE,
                                    "asian_range_pips": asian_range_pips,
                                    "lor_range_pips": lor_range_pips if lor_valid else None,
                                    "is_reentry": channels[("B", "short")]["entries"] > 0,
                                }
                            )
                            channels[("B", "short")]["state"] = "PENDING"
                            b_candidates["short"] = []
                    b_candidates["short"] = [c for c in b_candidates["short"] if c["expire_i"] >= i]

                    # long candidate: downside break then close back inside
                    if float(row["low"]) < asian_low:
                        b_candidates["long"].append({"expire_i": i + 2, "wick_low": float(row["low"])})
                    for cand in b_candidates["long"]:
                        cand["wick_low"] = min(float(cand["wick_low"]), float(row["low"]))
                    long_confirm = float(row["close"]) > asian_low + float(cfg["setups"]["B"]["close_back_inside_pips"]) * PIP_SIZE
                    if long_confirm and bool(cfg["setups"]["B"].get("allow_long", True)) and channels[("B", "long")]["state"] == "ARMED":
                        active = [c for c in b_candidates["long"] if c["expire_i"] >= i]
                        if active:
                            wick_low = min(float(c["wick_low"]) for c in active)
                            pending_entries.append(
                                {
                                    "setup_type": "B",
                                    "direction": "long",
                                    "execute_time": nxt_ts,
                                    "raw_sl": wick_low - float(cfg["setups"]["B"]["sl_wick_buffer_pips"]) * PIP_SIZE,
                                    "asian_range_pips": asian_range_pips,
                                    "lor_range_pips": lor_range_pips if lor_valid else None,
                                    "is_reentry": channels[("B", "long")]["entries"] > 0,
                                }
                            )
                            channels[("B", "long")]["state"] = "PENDING"
                            b_candidates["long"] = []
                    b_candidates["long"] = [c for c in b_candidates["long"] if c["expire_i"] >= i]

            # Setup C (level bounce)
            if cfg["setups"]["C"]["enabled"] and asian_valid:
                c_start, c_end = windows["C"]
                if c_start <= ts < c_end:
                    body = abs(float(row["close"]) - float(row["open"]))
                    long_cond = (
                        float(row["low"]) <= asian_low + float(cfg["setups"]["C"]["bounce_zone_pips"]) * PIP_SIZE
                        and float(row["low"]) >= asian_low
                        and float(row["close"]) > float(row["open"])
                        and float(row["close"]) >= asian_low + float(cfg["setups"]["C"]["min_close_offset_pips"]) * PIP_SIZE
                        and body >= float(cfg["setups"]["C"]["min_body_pips"]) * PIP_SIZE
                    )
                    short_cond = (
                        float(row["high"]) >= asian_high - float(cfg["setups"]["C"]["bounce_zone_pips"]) * PIP_SIZE
                        and float(row["high"]) <= asian_high
                        and float(row["close"]) < float(row["open"])
                        and float(row["close"]) <= asian_high - float(cfg["setups"]["C"]["min_close_offset_pips"]) * PIP_SIZE
                        and body >= float(cfg["setups"]["C"]["min_body_pips"]) * PIP_SIZE
                    )
                    if long_cond and bool(cfg["setups"]["C"].get("allow_long", True)) and channels[("C", "long")]["state"] == "ARMED":
                        pending_entries.append(
                            {
                                "setup_type": "C",
                                "direction": "long",
                                "execute_time": nxt_ts,
                                "raw_sl": asian_low - float(cfg["setups"]["C"]["sl_buffer_pips"]) * PIP_SIZE,
                                "asian_range_pips": asian_range_pips,
                                "lor_range_pips": lor_range_pips if lor_valid else None,
                                "is_reentry": channels[("C", "long")]["entries"] > 0,
                            }
                        )
                        channels[("C", "long")]["state"] = "PENDING"
                    if short_cond and bool(cfg["setups"]["C"].get("allow_short", True)) and channels[("C", "short")]["state"] == "ARMED":
                        pending_entries.append(
                            {
                                "setup_type": "C",
                                "direction": "short",
                                "execute_time": nxt_ts,
                                "raw_sl": asian_high + float(cfg["setups"]["C"]["sl_buffer_pips"]) * PIP_SIZE,
                                "asian_range_pips": asian_range_pips,
                                "lor_range_pips": lor_range_pips if lor_valid else None,
                                "is_reentry": channels[("C", "short")]["entries"] > 0,
                            }
                        )
                        channels[("C", "short")]["state"] = "PENDING"

            # Setup D (LOR breakout)
            if cfg["setups"]["D"]["enabled"] and lor_valid:
                d_start, d_end = windows["D"]
                if d_start <= ts < d_end:
                    long_break = float(row["close"]) > lor_high + float(cfg["setups"]["D"]["breakout_buffer_pips"]) * PIP_SIZE
                    short_break = float(row["close"]) < lor_low - float(cfg["setups"]["D"]["breakout_buffer_pips"]) * PIP_SIZE
                    if long_break and bool(cfg["setups"]["D"].get("allow_long", True)) and channels[("D", "long")]["state"] == "ARMED":
                        pending_entries.append(
                            {
                                "setup_type": "D",
                                "direction": "long",
                                "execute_time": nxt_ts,
                                "raw_sl": lor_low - float(cfg["setups"]["D"]["sl_buffer_pips"]) * PIP_SIZE,
                                "asian_range_pips": asian_range_pips if asian_valid else None,
                                "lor_range_pips": lor_range_pips,
                                "is_reentry": channels[("D", "long")]["entries"] > 0,
                            }
                        )
                        channels[("D", "long")]["state"] = "PENDING"
                    if short_break and bool(cfg["setups"]["D"].get("allow_short", True)) and channels[("D", "short")]["state"] == "ARMED":
                        pending_entries.append(
                            {
                                "setup_type": "D",
                                "direction": "short",
                                "execute_time": nxt_ts,
                                "raw_sl": lor_high + float(cfg["setups"]["D"]["sl_buffer_pips"]) * PIP_SIZE,
                                "asian_range_pips": asian_range_pips if asian_valid else None,
                                "lor_range_pips": lor_range_pips,
                                "is_reentry": channels[("D", "short")]["entries"] > 0,
                            }
                        )
                        channels[("D", "short")]["state"] = "PENDING"

    trades_df = pd.DataFrame(trades_out)
    if trades_df.empty:
        trades_df = pd.DataFrame(
            columns=[
                "trade_id",
                "setup_type",
                "date",
                "day_of_week",
                "direction",
                "entry_time_utc",
                "exit_time_utc",
                "entry_price",
                "exit_price",
                "sl_price",
                "tp1_price",
                "tp2_price",
                "sl_pips",
                "risk_pct",
                "pnl_pips",
                "pnl_usd",
                "exit_reason",
                "asian_range_pips",
                "lor_range_pips",
                "MAE_pips",
                "MFE_pips",
                "position_units",
                "trade_duration_minutes",
                "trade_sequence_number",
                "is_reentry",
                "entry_hour_utc",
                "margin_used_pct",
            ]
        )
    else:
        trades_df = trades_df.sort_values(["exit_time_utc", "trade_id"]).reset_index(drop=True)

    equity_events_df = pd.DataFrame(equity_events, columns=["timestamp", "delta_usd"]) if equity_events else pd.DataFrame(columns=["timestamp", "delta_usd"])
    m1_times = df[["time"]].drop_duplicates().sort_values("time").reset_index(drop=True)
    if equity_events_df.empty:
        m1_times["equity"] = start_balance
    else:
        ev = equity_events_df.groupby("timestamp", as_index=False)["delta_usd"].sum()
        m1_times = m1_times.merge(ev, left_on="time", right_on="timestamp", how="left")
        if "timestamp" in m1_times.columns:
            m1_times = m1_times.drop(columns=["timestamp"])
        m1_times["delta_usd"] = m1_times["delta_usd"].fillna(0.0)
        m1_times["equity"] = start_balance + m1_times["delta_usd"].cumsum()
    equity_curve = m1_times[["time", "equity"]].rename(columns={"time": "timestamp"})
    return trades_df, equity_curve, diagnostics


def _setup_metrics(setup_trades: pd.DataFrame) -> dict[str, Any]:
    if setup_trades.empty:
        return {
            "trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "net_pnl_usd": 0.0,
            "expectancy_per_trade_usd": 0.0,
            "avg_winner_pips": 0.0,
            "avg_loser_pips": 0.0,
        }
    wins = setup_trades[setup_trades["pnl_usd"] > 0]
    losses = setup_trades[setup_trades["pnl_usd"] < 0]
    gp = float(wins["pnl_usd"].sum()) if not wins.empty else 0.0
    gl = float(losses["pnl_usd"].sum()) if not losses.empty else 0.0
    pf = gp / abs(gl) if gl < 0 else (float("inf") if gp > 0 else 0.0)
    return {
        "trades": int(len(setup_trades)),
        "win_rate_pct": float((setup_trades["pnl_usd"] > 0).mean() * 100.0),
        "profit_factor": float(pf),
        "net_pnl_usd": float(setup_trades["pnl_usd"].sum()),
        "expectancy_per_trade_usd": float(setup_trades["pnl_usd"].mean()),
        "avg_winner_pips": float(wins["pnl_pips"].mean()) if not wins.empty else 0.0,
        "avg_loser_pips": float(losses["pnl_pips"].mean()) if not losses.empty else 0.0,
    }


def write_outputs(
    out_prefix: str,
    cfg: dict[str, Any],
    trades: pd.DataFrame,
    equity_curve: pd.DataFrame,
    diagnostics: dict[str, Any],
    dataset_path: str,
) -> dict[str, Any]:
    out_base = Path(out_prefix)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    report_path = str(out_base.with_name(out_base.name + "_report.json"))
    trades_path = str(out_base.with_name(out_base.name + "_trades.csv"))
    equity_path = str(out_base.with_name(out_base.name + "_equity.csv"))

    start_balance = float(cfg["account"]["starting_balance"])
    t0 = pd.Timestamp(equity_curve["timestamp"].min()) if not equity_curve.empty else pd.Timestamp.utcnow().tz_localize("UTC")
    t1 = pd.Timestamp(equity_curve["timestamp"].max()) if not equity_curve.empty else t0

    combined_metrics = performance_metrics(trades, equity_curve, start_balance, t0, t1)
    setup_breakdown = {s: _setup_metrics(trades[trades["setup_type"] == s].copy()) for s in ["A", "B", "C", "D"]}

    day_pnl_rows = []
    for d in ["Tuesday", "Wednesday", "Thursday"]:
        x = trades[trades["day_of_week"] == d]
        gp = float(x.loc[x["pnl_usd"] > 0, "pnl_usd"].sum()) if not x.empty else 0.0
        gl = float(x.loc[x["pnl_usd"] < 0, "pnl_usd"].sum()) if not x.empty else 0.0
        pf = gp / abs(gl) if gl < 0 else (float("inf") if gp > 0 else 0.0)
        day_pnl_rows.append(
            {
                "day": d,
                "trades": int(len(x)),
                "win_rate_pct": float((x["pnl_usd"] > 0).mean() * 100.0) if not x.empty else 0.0,
                "profit_factor": float(pf),
                "net_pnl_usd": float(x["pnl_usd"].sum()) if not x.empty else 0.0,
            }
        )

    hour_rows = []
    if not trades.empty:
        for h, g in trades.groupby("entry_hour_utc"):
            gp = float(g.loc[g["pnl_usd"] > 0, "pnl_usd"].sum()) if not g.empty else 0.0
            gl = float(g.loc[g["pnl_usd"] < 0, "pnl_usd"].sum()) if not g.empty else 0.0
            pf = gp / abs(gl) if gl < 0 else (float("inf") if gp > 0 else 0.0)
            hour_rows.append(
                {
                    "entry_hour_utc": int(h),
                    "trades": int(len(g)),
                    "win_rate_pct": float((g["pnl_usd"] > 0).mean() * 100.0),
                    "profit_factor": float(pf),
                    "net_pnl_usd": float(g["pnl_usd"].sum()),
                }
            )
    hour_rows = sorted(hour_rows, key=lambda r: r["entry_hour_utc"])

    seq_rows = []
    if not trades.empty:
        for seq, g in trades.groupby("trade_sequence_number"):
            seq_rows.append(
                {
                    "trade_sequence_number": int(seq),
                    "trades": int(len(g)),
                    "win_rate_pct": float((g["pnl_usd"] > 0).mean() * 100.0),
                    "net_pnl_usd": float(g["pnl_usd"].sum()),
                }
            )

    exit_reasons = ["TP1_PARTIAL", "TP2_FULL", "SL_FULL", "BE_STOP", "HARD_CLOSE"]
    exit_by_setup: dict[str, list[dict[str, Any]]] = {}
    for s in ["A", "B", "C", "D"]:
        rows = []
        x = trades[trades["setup_type"] == s]
        for r in exit_reasons:
            y = x[x["exit_reason"] == r]
            rows.append({"exit_reason": r, "count": int(len(y)), "total_pnl_usd": float(y["pnl_usd"].sum()) if not y.empty else 0.0})
        exit_by_setup[s] = rows

    monthly_combined: list[dict[str, Any]] = []
    monthly_per_setup: dict[str, list[dict[str, Any]]] = {}
    if not trades.empty:
        t = trades.copy()
        t["month"] = t["exit_time_utc"].dt.to_period("M").astype(str)
        g = t.groupby("month", as_index=False)["pnl_usd"].sum().rename(columns={"pnl_usd": "net_pnl_usd"})
        monthly_combined = g.to_dict(orient="records")
        for s in ["A", "B", "C", "D"]:
            xs = t[t["setup_type"] == s]
            gs = xs.groupby("month", as_index=False)["pnl_usd"].sum().rename(columns={"pnl_usd": "net_pnl_usd"})
            monthly_per_setup[s] = gs.to_dict(orient="records")
    else:
        monthly_per_setup = {"A": [], "B": [], "C": [], "D": []}

    if trades.empty:
        trade_dist = [{"trades_in_day": 0, "days": diagnostics["days_active"]}]
        avg_trades_per_eligible_day = 0.0
        avg_trades_per_calendar_month = 0.0
        total_reentries = {"A": 0, "B": 0, "C": 0, "D": 0}
    else:
        day_counts = trades.groupby("date", as_index=False).size().rename(columns={"size": "trade_count"})
        day_range = pd.date_range(pd.Timestamp(t0).floor("D"), pd.Timestamp(t1).floor("D"), freq="D", tz="UTC")
        all_active_dates = [d.date().isoformat() for d in day_range if d.day_name() in cfg["session"]["active_days_utc"]]
        day_full = pd.DataFrame({"date": all_active_dates}).merge(day_counts, on="date", how="left").fillna({"trade_count": 0})
        day_full["trade_count"] = day_full["trade_count"].astype(int)
        dist = day_full["trade_count"].value_counts().sort_index()
        trade_dist = [{"trades_in_day": int(k), "days": int(v)} for k, v in dist.items()]
        eligible_days = max(diagnostics["days_asian_valid"], diagnostics["days_lor_valid"])
        avg_trades_per_eligible_day = float(len(trades) / max(1, eligible_days))
        span_days = max((t1 - t0).total_seconds() / 86400.0, 1.0)
        avg_trades_per_calendar_month = float(len(trades) / (span_days / 30.4375))
        total_reentries = {s: int(len(trades[(trades["setup_type"] == s) & (trades["is_reentry"] == True)])) for s in ["A", "B", "C", "D"]}

    total_trades_per_setup = {s: int(len(trades[trades["setup_type"] == s])) for s in ["A", "B", "C", "D"]}
    direction_breakdown = {}
    for s in ["A", "B", "C", "D"]:
        for d in ["long", "short"]:
            key = f"{s}_{d}"
            x = trades[(trades["setup_type"] == s) & (trades["direction"] == d)]
            direction_breakdown[key] = {
                "count": int(len(x)),
                "win_rate_pct": float((x["pnl_usd"] > 0).mean() * 100.0) if not x.empty else 0.0,
                "net_pnl_usd": float(x["pnl_usd"].sum()) if not x.empty else 0.0,
            }

    margin_utilization = {
        "avg_margin_used_pct": float(trades["margin_used_pct"].mean()) if not trades.empty and "margin_used_pct" in trades.columns else 0.0,
        "max_margin_used_pct": float(trades["margin_used_pct"].max()) if not trades.empty and "margin_used_pct" in trades.columns else 0.0,
        "trades_skipped_margin_constraint": int(diagnostics.get("trades_skipped_margin_constraint", 0)),
    }

    report = {
        "dataset": dataset_path,
        "config_snapshot": cfg,
        "diagnostics": diagnostics,
        "aggregate_combined": combined_metrics,
        "per_setup_breakdown": setup_breakdown,
        "frequency_analysis": {
            "total_trades_per_setup": total_trades_per_setup,
            "trades_per_day_distribution": trade_dist,
            "average_trades_per_eligible_day": avg_trades_per_eligible_day,
            "average_trades_per_calendar_month": avg_trades_per_calendar_month,
            "total_reentries_per_setup": total_reentries,
        },
        "direction_breakdown": direction_breakdown,
        "margin_utilization": margin_utilization,
        "pnl_by_day_of_week": day_pnl_rows,
        "pnl_by_entry_hour_utc": hour_rows,
        "pnl_by_trade_sequence_number": seq_rows,
        "exit_reason_breakdown_per_setup": exit_by_setup,
        "monthly_pnl": {
            "combined": monthly_combined,
            "per_setup": monthly_per_setup,
        },
        "paths": {
            "report_json": report_path,
            "trade_log_csv": trades_path,
            "equity_curve_csv": equity_path,
        },
    }

    export_cols = [
        "trade_id",
        "setup_type",
        "date",
        "day_of_week",
        "direction",
        "entry_time_utc",
        "exit_time_utc",
        "entry_price",
        "exit_price",
        "sl_price",
        "tp1_price",
        "tp2_price",
        "sl_pips",
        "risk_pct",
        "pnl_pips",
        "pnl_usd",
        "exit_reason",
        "asian_range_pips",
        "lor_range_pips",
        "MAE_pips",
        "MFE_pips",
        "position_units",
        "trade_duration_minutes",
        "trade_sequence_number",
        "is_reentry",
        "margin_used_pct",
        "entry_hour_utc",
    ]
    trades[export_cols].to_csv(trades_path, index=False)
    equity_curve.to_csv(equity_path, index=False)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    return report


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg_user = json.load(f)
    cfg = merge_config(cfg_user)

    df = pd.read_csv(args.input)
    required = {"time", "open", "high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError("Input CSV must include columns: time, open, high, low, close")
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)

    trades, equity_curve, diagnostics = run_backtest(df, cfg)
    report = write_outputs(args.out_prefix, cfg, trades, equity_curve, diagnostics, args.input)

    print(
        json.dumps(
            {
                "dataset": args.input,
                "aggregate_combined": {
                    "trades": report["aggregate_combined"]["total_trades"],
                    "wr_pct": round(report["aggregate_combined"]["win_rate_pct"], 2),
                    "pf": (
                        round(report["aggregate_combined"]["profit_factor"], 3)
                        if math.isfinite(report["aggregate_combined"]["profit_factor"])
                        else "inf"
                    ),
                    "net_usd": round(report["aggregate_combined"]["net_pnl_usd"], 2),
                    "max_dd_usd": round(report["aggregate_combined"]["max_drawdown_usd"], 2),
                },
                "per_setup_trades": {k: v["trades"] for k, v in report["per_setup_breakdown"].items()},
                "paths": report["paths"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
