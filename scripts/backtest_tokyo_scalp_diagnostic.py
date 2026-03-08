#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PIP = 0.01


@dataclass
class Trade:
    trade_id: int
    system_id: int
    direction: str
    entry_time: pd.Timestamp
    entry_idx: int
    entry_price: float
    entry_hour: int
    entry_day: str
    tp_pips: float
    sl_pips: float
    time_stop_min: int
    signal_time: pd.Timestamp
    signal_body_pips: float
    signal_range_pips: float
    spread_at_entry: float
    units: int
    session_range_at_entry: Optional[float] = None
    compression_range_pips: Optional[float] = None
    three_candle_total_pips: Optional[float] = None
    session_low_at_entry: Optional[float] = None
    session_high_at_entry: Optional[float] = None
    max_fav: float = 0.0
    max_adv: float = 0.0
    mfe_time: Optional[pd.Timestamp] = None


def hhmm_to_min(s: str) -> int:
    hh, mm = s.split(":")
    return int(hh) * 60 + int(mm)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()


def load_m1(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"time", "open", "high", "low", "close"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Missing required columns in {path}: {miss}")
    keep = ["time", "open", "high", "low", "close"] + (["spread_pips"] if "spread_pips" in df.columns else [])
    out = df[keep].copy()
    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
    for c in ["open", "high", "low", "close"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if "spread_pips" in out.columns:
        out["spread_pips"] = pd.to_numeric(out["spread_pips"], errors="coerce")
    out = out.dropna(subset=["time", "open", "high", "low", "close"]).sort_values("time").reset_index(drop=True)
    out["date_utc"] = out["time"].dt.date.astype(str)
    out["day_of_week"] = out["time"].dt.day_name()
    out["hour_utc"] = out["time"].dt.hour
    out["minute_utc"] = out["time"].dt.minute
    out["minute_of_day"] = out["hour_utc"] * 60 + out["minute_utc"]
    out["body_pips"] = (out["close"] - out["open"]).abs() / PIP
    out["range_pips"] = (out["high"] - out["low"]) / PIP
    out["bull"] = out["close"] > out["open"]
    out["bear"] = out["close"] < out["open"]
    return out


SYSTEMS = {
    1: {
        "name": "Spike Fade",
        "tp": 3.0,
        "sl": 4.0,
        "time_stop": 15,
        "max_entries_session": 10,
        "min_gap": 5,
    },
    2: {
        "name": "Three-Candle Fade",
        "tp": 3.0,
        "sl": 5.0,
        "time_stop": 20,
        "max_entries_session": 8,
        "min_gap": 10,
    },
    3: {
        "name": "Three-Candle Continuation",
        "tp": 3.0,
        "sl": 5.0,
        "time_stop": 20,
        "max_entries_session": 8,
        "min_gap": 10,
    },
    4: {
        "name": "Compression Breakout",
        "tp": 4.0,
        "sl": 3.0,
        "time_stop": 15,
        "max_entries_session": 8,
        "min_gap": 10,
    },
    5: {
        "name": "Session Range Fade",
        "tp": 4.0,
        "sl": 5.0,
        "time_stop": 30,
        "max_entries_session": 6,
        "min_gap": 15,
    },
    6: {
        "name": "Random Entry Control",
        "tp": 3.0,
        "sl": 4.0,
        "time_stop": 20,
        "max_entries_session": 13,
        "min_gap": 0,
    },
}


def model_spread(minute_of_day: int) -> float:
    if 0 <= minute_of_day < 60:
        return 1.2
    if 60 <= minute_of_day < 150:
        return 0.9
    if 150 <= minute_of_day < 210:
        return 1.5
    if 210 <= minute_of_day < 360:
        return 1.0
    if 360 <= minute_of_day < 540:
        return 0.8
    return 1.2


def get_spread_pips(row: pd.Series) -> float:
    sp = row.get("spread_pips", np.nan)
    if not pd.isna(sp):
        return float(sp)
    return model_spread(int(row["minute_of_day"]))


def in_session(day: str, minute_of_day: int, allowed_days: set[str]) -> bool:
    return (day in allowed_days) and (0 <= minute_of_day < 540)


def is_lunch(minute_of_day: int) -> bool:
    return 150 <= minute_of_day < 210


def entry_blocked_time(minute_of_day: int) -> bool:
    if minute_of_day >= 510:  # no new entries after 08:30
        return True
    if is_lunch(minute_of_day):
        return True
    return False


def should_force_close(minute_of_day: int) -> bool:
    return minute_of_day >= 530  # 08:50


def session_key(ts: pd.Timestamp) -> str:
    return ts.date().isoformat()


def summarize_months(df: pd.DataFrame) -> float:
    span_days = (df["time"].max() - df["time"].min()).total_seconds() / 86400.0
    return max(span_days / 30.4375, 1e-9)


def grp_stats(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "trades": 0,
            "wr": 0.0,
            "pf": 0.0,
            "net_usd": 0.0,
            "avg_pips": 0.0,
            "avg_duration_min": 0.0,
            "maxdd_usd": 0.0,
            "maxdd_pct": 0.0,
            "sharpe": 0.0,
            "calmar": 0.0,
        }
    gp = float(trades.loc[trades["pnl_usd"] > 0, "pnl_usd"].sum())
    gl = abs(float(trades.loc[trades["pnl_usd"] < 0, "pnl_usd"].sum()))
    pf = gp / gl if gl > 0 else float("inf")
    wr = float((trades["pnl_usd"] > 0).mean() * 100.0)
    net = float(trades["pnl_usd"].sum())
    eq = 100000.0 + trades["pnl_usd"].cumsum().to_numpy()
    peaks = np.maximum.accumulate(eq)
    dd = peaks - eq
    dd_pct = np.where(peaks > 0, dd / peaks * 100.0, 0.0)
    ret = trades["pnl_usd"] / (100000.0 + trades["pnl_usd"].cumsum().shift(1).fillna(0.0))
    sharpe = float((ret.mean() / ret.std(ddof=0)) * np.sqrt(len(ret))) if ret.std(ddof=0) > 0 else 0.0
    calmar = float(((eq[-1] - 100000.0) / 100000.0 * 100.0) / max(1e-9, dd_pct.max())) if len(dd_pct) else 0.0
    return {
        "trades": int(len(trades)),
        "wr": wr,
        "pf": float(pf),
        "net_usd": net,
        "avg_pips": float(trades["net_pips"].mean()),
        "avg_duration_min": float(trades["trade_duration_minutes"].mean()),
        "maxdd_usd": float(dd.max()) if len(dd) else 0.0,
        "maxdd_pct": float(dd_pct.max()) if len(dd_pct) else 0.0,
        "sharpe": sharpe,
        "calmar": calmar,
    }


def backtest_system(
    df: pd.DataFrame,
    system_id: int,
    units: int,
    allowed_days: set[str],
    tp_override: Optional[float] = None,
    sl_override: Optional[float] = None,
) -> tuple[pd.DataFrame, dict]:
    cfg = SYSTEMS[system_id].copy()
    if tp_override is not None:
        cfg["tp"] = float(tp_override)
    if sl_override is not None:
        cfg["sl"] = float(sl_override)

    trades: list[dict] = []
    diag = {
        "system_id": system_id,
        "system_name": cfg["name"],
        "total_signals_generated": 0,
        "entries_taken": 0,
        "entries_blocked_lunch": 0,
        "entries_blocked_max_concurrent": 0,
        "entries_blocked_min_time": 0,
        "entries_blocked_consecutive_loss_pause": 0,
        "entries_blocked_entry_cutoff": 0,
        "entries_blocked_same_dir_reentry": 0,
        "entries_blocked_session_cap": 0,
        "sessions_total": 0,
        "sessions_with_zero_signals": 0,
        "signals_per_session": [],
        "exit_counts": {},
        "mirror_meta": {},
    }

    open_trade: Optional[Trade] = None
    pending: Optional[dict] = None
    last_entry_time: Optional[pd.Timestamp] = None
    last_close_time_by_dir: dict[str, pd.Timestamp] = {}

    current_session = None
    session_entries = 0
    session_signals = 0
    session_loss_streak = 0
    pause_until: Optional[pd.Timestamp] = None
    pause_hour_until: Optional[pd.Timestamp] = None
    session_high = -1e9
    session_low = 1e9
    # sys6 alternating direction
    rand_next_long = True

    trade_id = 0

    # Track breakout first direction for system 5 not needed; for all maybe none

    # Helper to close trade
    def close_trade(t: Trade, ts: pd.Timestamp, exit_price: float, reason: str, irow: pd.Series):
        nonlocal session_loss_streak, pause_until, pause_hour_until
        gross_pips = (exit_price - t.entry_price) / PIP if t.direction == "long" else (t.entry_price - exit_price) / PIP
        net_pips = gross_pips - t.spread_at_entry
        spread_cost_usd = t.spread_at_entry * t.units * 0.01 / max(1e-9, t.entry_price)
        pnl_usd = net_pips * t.units * 0.01 / max(1e-9, t.entry_price)
        dur = (ts - t.entry_time).total_seconds() / 60.0
        ttmfe = (t.mfe_time - t.entry_time).total_seconds() / 60.0 if t.mfe_time is not None else np.nan

        rec = {
            "trade_id": t.trade_id,
            "system_id": t.system_id,
            "entry_datetime": str(t.entry_time),
            "exit_datetime": str(ts),
            "direction": t.direction,
            "entry_price": float(t.entry_price),
            "exit_price": float(exit_price),
            "sl_price": float(t.entry_price - t.sl_pips * PIP if t.direction == "long" else t.entry_price + t.sl_pips * PIP),
            "tp_price": float(t.entry_price + t.tp_pips * PIP if t.direction == "long" else t.entry_price - t.tp_pips * PIP),
            "exit_reason": reason,
            "gross_pips": float(gross_pips),
            "spread_at_entry": float(t.spread_at_entry),
            "net_pips": float(net_pips),
            "pnl_usd": float(pnl_usd),
            "spread_cost_usd": float(spread_cost_usd),
            "signal_candle_datetime": str(t.signal_time),
            "signal_candle_body_pips": float(t.signal_body_pips),
            "signal_candle_range_pips": float(t.signal_range_pips),
            "mfe_pips": float(t.max_fav),
            "mae_pips": float(t.max_adv),
            "time_to_mfe_minutes": float(ttmfe) if not pd.isna(ttmfe) else np.nan,
            "trade_duration_minutes": float(dur),
            "position_units": int(t.units),
            "hour_utc": int(t.entry_hour),
            "day_of_week": t.entry_day,
            "session_range_at_entry": float(t.session_range_at_entry) if t.session_range_at_entry is not None else np.nan,
            "compression_range_pips": float(t.compression_range_pips) if t.compression_range_pips is not None else np.nan,
            "three_candle_total_pips": float(t.three_candle_total_pips) if t.three_candle_total_pips is not None else np.nan,
        }
        trades.append(rec)
        diag["exit_counts"][reason] = diag["exit_counts"].get(reason, 0) + 1

        # global trade-management streak rules
        if pnl_usd < 0:
            session_loss_streak += 1
            if session_loss_streak >= 3:
                # pause remainder of hour
                boundary = ts.floor("h") + pd.Timedelta(hours=1)
                pause_hour_until = boundary
                session_loss_streak = 0
            elif session_loss_streak >= 2:
                pause_until = ts + pd.Timedelta(minutes=10)
            last_close_time_by_dir[t.direction] = ts
        else:
            session_loss_streak = 0
            last_close_time_by_dir[t.direction] = ts

    # precompute schedule times for system 6
    sys6_times = {(0, 30), (1, 0), (1, 30), (2, 0), (4, 0), (4, 30), (5, 0), (5, 30), (6, 0), (6, 30), (7, 0), (7, 30), (8, 0)}

    n = len(df)
    for i in range(n):
        row = df.iloc[i]
        ts = row["time"]
        day = row["day_of_week"]
        mod = int(row["minute_of_day"])

        # new session reset at 00:00 weekday
        s_key = session_key(ts)
        if current_session != s_key and mod < 540:
            # close prior session stats
            if current_session is not None:
                diag["signals_per_session"].append(session_signals)
                if session_signals == 0:
                    diag["sessions_with_zero_signals"] += 1
            current_session = s_key
            if day in allowed_days:
                diag["sessions_total"] += 1
            session_entries = 0
            session_signals = 0
            session_loss_streak = 0
            pause_until = None
            pause_hour_until = None
            session_high = -1e9
            session_low = 1e9
            rand_next_long = True

        # update session range tracking
        if in_session(day, mod, allowed_days):
            session_high = max(session_high, float(row["high"]))
            session_low = min(session_low, float(row["low"]))

        # manage open trade exits
        if open_trade is not None:
            # update mfe/mae using current candle
            if open_trade.direction == "long":
                fav = (float(row["high"]) - open_trade.entry_price) / PIP
                adv = (open_trade.entry_price - float(row["low"])) / PIP
            else:
                fav = (open_trade.entry_price - float(row["low"])) / PIP
                adv = (float(row["high"]) - open_trade.entry_price) / PIP
            if fav > open_trade.max_fav:
                open_trade.max_fav = float(fav)
                open_trade.mfe_time = ts
            open_trade.max_adv = max(open_trade.max_adv, float(adv))

            # session close forced
            if should_force_close(mod):
                close_trade(open_trade, ts, float(row["open"]), "Session close", row)
                open_trade = None
            else:
                # check SL first
                sl_hit = False
                tp_hit = False
                if open_trade.direction == "long":
                    sl_level = open_trade.entry_price - open_trade.sl_pips * PIP
                    tp_level = open_trade.entry_price + open_trade.tp_pips * PIP
                    if float(row["low"]) <= sl_level:
                        close_trade(open_trade, ts, sl_level, "SL hit", row)
                        open_trade = None
                        sl_hit = True
                    if (not sl_hit) and system_id == 5 and open_trade.session_low_at_entry is not None and float(row["low"]) < float(open_trade.session_low_at_entry):
                        close_trade(open_trade, ts, float(row["open"]), "Range invalid", row)
                        open_trade = None
                    elif (not sl_hit) and float(row["high"]) >= tp_level:
                        close_trade(open_trade, ts, tp_level, "TP hit", row)
                        open_trade = None
                        tp_hit = True
                else:
                    sl_level = open_trade.entry_price + open_trade.sl_pips * PIP
                    tp_level = open_trade.entry_price - open_trade.tp_pips * PIP
                    if float(row["high"]) >= sl_level:
                        close_trade(open_trade, ts, sl_level, "SL hit", row)
                        open_trade = None
                        sl_hit = True
                    if (not sl_hit) and system_id == 5 and open_trade.session_high_at_entry is not None and float(row["high"]) > float(open_trade.session_high_at_entry):
                        close_trade(open_trade, ts, float(row["open"]), "Range invalid", row)
                        open_trade = None
                    elif (not sl_hit) and float(row["low"]) <= tp_level:
                        close_trade(open_trade, ts, tp_level, "TP hit", row)
                        open_trade = None
                        tp_hit = True

                # time stop at next candle open
                if open_trade is not None:
                    elapsed = int((ts - open_trade.entry_time).total_seconds() // 60)
                    if elapsed >= open_trade.time_stop_min:
                        close_trade(open_trade, ts, float(row["open"]), "Time stop", row)
                        open_trade = None

        # entry from pending signal (next candle open)
        if pending is not None and pending.get("entry_idx") == i:
            # entry gating at entry time
            if not in_session(day, mod, allowed_days):
                pass
            elif entry_blocked_time(mod):
                if is_lunch(mod):
                    diag["entries_blocked_lunch"] += 1
                else:
                    diag["entries_blocked_entry_cutoff"] += 1
            elif open_trade is not None:
                diag["entries_blocked_max_concurrent"] += 1
            elif session_entries >= cfg["max_entries_session"]:
                diag["entries_blocked_session_cap"] += 1
            elif last_entry_time is not None and cfg["min_gap"] > 0 and (ts - last_entry_time).total_seconds() < cfg["min_gap"] * 60:
                diag["entries_blocked_min_time"] += 1
            elif pause_hour_until is not None and ts < pause_hour_until:
                diag["entries_blocked_consecutive_loss_pause"] += 1
            elif pause_until is not None and ts < pause_until:
                diag["entries_blocked_consecutive_loss_pause"] += 1
            elif pending["direction"] in last_close_time_by_dir and (ts - last_close_time_by_dir[pending["direction"]]).total_seconds() < 3 * 60:
                diag["entries_blocked_same_dir_reentry"] += 1
            else:
                spread = get_spread_pips(row)
                trade_id += 1
                open_trade = Trade(
                    trade_id=trade_id,
                    system_id=system_id,
                    direction=pending["direction"],
                    entry_time=ts,
                    entry_idx=i,
                    entry_price=float(row["open"]),
                    entry_hour=int(row["hour_utc"]),
                    entry_day=str(day),
                    tp_pips=float(pending["tp_pips"]),
                    sl_pips=float(pending["sl_pips"]),
                    time_stop_min=int(pending["time_stop_min"]),
                    signal_time=pending["signal_time"],
                    signal_body_pips=float(pending.get("signal_body_pips", np.nan)),
                    signal_range_pips=float(pending.get("signal_range_pips", np.nan)),
                    spread_at_entry=float(spread),
                    units=units,
                    session_range_at_entry=pending.get("session_range_at_entry"),
                    compression_range_pips=pending.get("compression_range_pips"),
                    three_candle_total_pips=pending.get("three_candle_total_pips"),
                    session_low_at_entry=pending.get("session_low_at_entry"),
                    session_high_at_entry=pending.get("session_high_at_entry"),
                )
                last_entry_time = ts
                session_entries += 1
                diag["entries_taken"] += 1
            pending = None

        # generate signal on completed current candle for next candle
        if i >= n - 1:
            continue
        next_row = df.iloc[i + 1]
        next_day = next_row["day_of_week"]
        next_mod = int(next_row["minute_of_day"])

        # only generate signals during session days; pending may still get blocked by cutoff/lunch rules later
        if not in_session(day, mod, allowed_days):
            continue

        direction = None
        signal_meta = {
            "signal_time": ts,
            "signal_body_pips": float(row["body_pips"]),
            "signal_range_pips": float(row["range_pips"]),
            "tp_pips": cfg["tp"],
            "sl_pips": cfg["sl"],
            "time_stop_min": cfg["time_stop"],
        }

        if system_id == 1:
            if float(row["body_pips"]) >= 3.0:
                direction = "short" if bool(row["bull"]) else ("long" if bool(row["bear"]) else None)

        elif system_id in (2, 3):
            if i >= 2:
                r0, r1, r2 = df.iloc[i], df.iloc[i - 1], df.iloc[i - 2]
                all_bull = bool(r0["bull"] and r1["bull"] and r2["bull"])
                all_bear = bool(r0["bear"] and r1["bear"] and r2["bear"])
                total = float(r0["body_pips"] + r1["body_pips"] + r2["body_pips"])
                if (all_bull or all_bear) and total >= 4.0:
                    signal_meta["three_candle_total_pips"] = total
                    if system_id == 2:  # fade
                        direction = "short" if all_bull else "long"
                    else:  # continue
                        direction = "long" if all_bull else "short"

        elif system_id == 4:
            if i >= 5:
                prev5 = df.iloc[i - 5:i]
                compress_ok = bool((prev5["range_pips"] < 1.5).all())
                exp_ok = float(row["range_pips"]) >= 2.0
                if compress_ok and exp_ok:
                    ch = float(prev5["high"].max())
                    cl = float(prev5["low"].min())
                    signal_meta["compression_range_pips"] = (ch - cl) / PIP
                    if float(row["close"]) > ch:
                        direction = "long"
                    elif float(row["close"]) < cl:
                        direction = "short"

        elif system_id == 5:
            # active after 01:30 UTC
            if mod >= 90:
                srange = max(0.0, (session_high - session_low) / PIP)
                if srange >= 5.0:
                    lo_near = (float(row["low"]) - session_low) / PIP <= 2.0
                    hi_near = (session_high - float(row["high"])) / PIP <= 2.0
                    # close in upper/lower half
                    rng = max(1e-9, float(row["high"] - row["low"]))
                    close_upper = float(row["close"]) >= float(row["low"] + 0.5 * rng)
                    close_lower = float(row["close"]) <= float(row["high"] - 0.5 * rng)
                    if lo_near and close_upper:
                        direction = "long"
                    elif hi_near and close_lower:
                        direction = "short"
                    if direction is not None:
                        signal_meta["session_range_at_entry"] = srange
                        signal_meta["session_low_at_entry"] = session_low
                        signal_meta["session_high_at_entry"] = session_high

        elif system_id == 6:
            # schedule entries directly by next candle open timestamps
            h, m = int(next_row["hour_utc"]), int(next_row["minute_utc"])
            if in_session(str(next_day), next_mod, allowed_days) and (h, m) in sys6_times:
                direction = "long" if rand_next_long else "short"
                rand_next_long = not rand_next_long

        if direction is not None:
            diag["total_signals_generated"] += 1
            session_signals += 1
            pending = {
                "entry_idx": i + 1,
                "direction": direction,
                **signal_meta,
            }

    # session tail
    if current_session is not None:
        diag["signals_per_session"].append(session_signals)
        if session_signals == 0:
            diag["sessions_with_zero_signals"] += 1

    tdf = pd.DataFrame(trades)
    if tdf.empty:
        return tdf, diag

    # add equity/drawdown
    tdf["entry_datetime"] = pd.to_datetime(tdf["entry_datetime"], utc=True)
    tdf["exit_datetime"] = pd.to_datetime(tdf["exit_datetime"], utc=True)
    tdf = tdf.sort_values("entry_datetime").reset_index(drop=True)
    tdf["equity_after"] = 100000.0 + tdf["pnl_usd"].cumsum()
    peaks = tdf["equity_after"].cummax()
    tdf["drawdown_usd"] = peaks - tdf["equity_after"]
    tdf["drawdown_pct"] = np.where(peaks > 0, tdf["drawdown_usd"] / peaks * 100.0, 0.0)

    return tdf, diag


def summarize_system(trades: pd.DataFrame, diag: dict, months: float) -> dict:
    s = grp_stats(trades)
    out = {
        "system_id": int(diag["system_id"]),
        "system_name": diag["system_name"],
        "trades": s["trades"],
        "wr_pct": s["wr"],
        "pf": s["pf"],
        "net_usd": s["net_usd"],
        "maxdd_usd": s["maxdd_usd"],
        "maxdd_pct": s["maxdd_pct"],
        "avg_pips": s["avg_pips"],
        "avg_duration_min": s["avg_duration_min"],
        "sharpe": s["sharpe"],
        "calmar": s["calmar"],
        "usd_per_month": s["net_usd"] / months,
        "signals_generated": int(diag["total_signals_generated"]),
        "entries_taken": int(diag["entries_taken"]),
    }

    if trades.empty:
        out["spread_stats"] = {
            "total_spread_cost_usd": 0.0,
            "avg_spread_pips": 0.0,
            "spread_as_pct_avg_gross_winner": 0.0,
            "hypothetical_pf": {"0.0": 0.0, "0.5": 0.0, "1.0": 0.0, "1.5": 0.0},
        }
        return out

    wins = trades[trades["pnl_usd"] > 0]
    losses = trades[trades["pnl_usd"] <= 0]
    gross_win_pips = wins["gross_pips"].mean() if len(wins) else np.nan
    avg_spread = float(trades["spread_at_entry"].mean())
    spread_pct = float((avg_spread / gross_win_pips) * 100.0) if (len(wins) and gross_win_pips and gross_win_pips > 0) else 0.0

    hypo = {}
    for sp in [0.0, 0.5, 1.0, 1.5]:
        pnl = (trades["gross_pips"] - sp) * trades["position_units"] * 0.01 / trades["entry_price"]
        gp = float(pnl[pnl > 0].sum())
        gl = abs(float(pnl[pnl < 0].sum()))
        hypo[str(sp)] = gp / gl if gl > 0 else float("inf")

    out["spread_stats"] = {
        "total_spread_cost_usd": float(trades["spread_cost_usd"].sum()),
        "avg_spread_pips": avg_spread,
        "spread_as_pct_avg_gross_winner": spread_pct,
        "hypothetical_pf": hypo,
    }

    return out


def build_detail(trades: pd.DataFrame, diag: dict) -> dict:
    if trades.empty:
        return {
            "signal_statistics": diag,
            "exit_distribution": [],
            "day_of_week": [],
            "hour_of_entry": [],
            "mfe_mae": {},
            "monthly": [],
        }

    t = trades.copy()
    t["is_win"] = t["pnl_usd"] > 0
    t["month"] = t["entry_datetime"].dt.to_period("M").astype(str)

    # exit distribution
    ex = []
    for e, g in t.groupby("exit_reason"):
        ex.append({
            "exit_type": e,
            "count": int(len(g)),
            "net_usd": float(g["pnl_usd"].sum()),
            "avg_pips": float(g["net_pips"].mean()),
            "avg_duration_min": float(g["trade_duration_minutes"].mean()),
        })

    # day/hour
    day_rows = []
    for d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
        g = t[t["day_of_week"] == d]
        s = grp_stats(g)
        day_rows.append({"day": d, "trades": s["trades"], "wr_pct": s["wr"], "pf": s["pf"], "net_usd": s["net_usd"]})

    hour_rows = []
    for h in range(0, 9):
        g = t[t["hour_utc"] == h]
        s = grp_stats(g)
        hour_rows.append({"hour_utc": h, "trades": s["trades"], "wr_pct": s["wr"], "pf": s["pf"], "net_usd": s["net_usd"]})

    # mfe/mae
    w = t[t["is_win"]]
    l = t[~t["is_win"]]
    cap = (w["net_pips"] / w["mfe_pips"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    mfe = {
        "winners": {
            "avg_mfe_pips": float(w["mfe_pips"].mean()) if len(w) else 0.0,
            "avg_mae_pips": float(w["mae_pips"].mean()) if len(w) else 0.0,
            "capture_ratio_mean": float(cap.mean()) if len(w) else 0.0,
            "capture_ratio_median": float(cap.median()) if len(w) else 0.0,
            "avg_time_to_mfe_minutes": float(w["time_to_mfe_minutes"].mean()) if len(w) else 0.0,
        },
        "losers": {
            "avg_mfe_pips": float(l["mfe_pips"].mean()) if len(l) else 0.0,
            "avg_mae_pips": float(l["mae_pips"].mean()) if len(l) else 0.0,
            "pct_touched_plus1": float((l["mfe_pips"] >= 1.0).mean() * 100.0) if len(l) else 0.0,
            "pct_touched_plus2": float((l["mfe_pips"] >= 2.0).mean() * 100.0) if len(l) else 0.0,
            "avg_time_to_plus1_minutes": float(l.loc[l["mfe_pips"] >= 1.0, "time_to_mfe_minutes"].mean()) if len(l) else 0.0,
        },
    }

    monthly = []
    for m, g in t.groupby("month"):
        monthly.append({"month": m, "net_usd": float(g["pnl_usd"].sum()), "trades": int(len(g)), "pf": grp_stats(g)["pf"]})

    return {
        "signal_statistics": {
            "total_signals_generated": int(diag["total_signals_generated"]),
            "entries_taken": int(diag["entries_taken"]),
            "entries_blocked_lunch": int(diag["entries_blocked_lunch"]),
            "entries_blocked_max_concurrent": int(diag["entries_blocked_max_concurrent"]),
            "entries_blocked_min_time_between": int(diag["entries_blocked_min_time"]),
            "entries_blocked_consecutive_loss_pause": int(diag["entries_blocked_consecutive_loss_pause"]),
            "entries_blocked_entry_cutoff": int(diag["entries_blocked_entry_cutoff"]),
            "entries_blocked_same_dir_reentry": int(diag["entries_blocked_same_dir_reentry"]),
            "entries_blocked_session_cap": int(diag["entries_blocked_session_cap"]),
            "signal_frequency_avg_per_session": float(np.mean(diag["signals_per_session"])) if len(diag["signals_per_session"]) else 0.0,
            "sessions_with_0_signals": int(diag["sessions_with_zero_signals"]),
            "sessions_total": int(diag["sessions_total"]),
        },
        "exit_distribution": ex,
        "day_of_week": day_rows,
        "hour_of_entry": hour_rows,
        "mfe_mae": mfe,
        "monthly": sorted(monthly, key=lambda x: x["month"]),
    }


def run_dataset(path: str, units: int, allowed_days: set[str]) -> dict:
    df = load_m1(path)
    months = summarize_months(df)

    dataset_out = {
        "input_csv": path,
        "months": months,
        "systems": {},
        "master_table": [],
    }

    sys_trades = {}
    sys_diag = {}
    for sid in range(1, 7):
        t, d = backtest_system(df, sid, units, allowed_days)
        sys_trades[sid] = t
        sys_diag[sid] = d
        summary = summarize_system(t, d, months)
        detail = build_detail(t, d)
        dataset_out["systems"][str(sid)] = {
            "summary": summary,
            "details": detail,
        }
        dataset_out["master_table"].append({
            "system": f"Sys {sid}",
            "name": SYSTEMS[sid]["name"],
            "trades": summary["trades"],
            "wr_pct": summary["wr_pct"],
            "pf": summary["pf"],
            "net_usd": summary["net_usd"],
            "maxdd_pct": summary["maxdd_pct"],
            "avg_pips": summary["avg_pips"],
            "avg_duration_min": summary["avg_duration_min"],
            "spread_cost_usd": summary["spread_stats"]["total_spread_cost_usd"],
        })

    # mirror test 2 vs 3
    s2 = dataset_out["systems"]["2"]["summary"]
    s3 = dataset_out["systems"]["3"]["summary"]
    if s2["pf"] > s3["pf"] + 0.05:
        verdict = "REVERTS"
    elif s3["pf"] > s2["pf"] + 0.05:
        verdict = "CONTINUES"
    else:
        verdict = "NEITHER"
    dataset_out["mirror_test"] = {
        "system2": {k: s2[k] for k in ["trades", "wr_pct", "pf", "net_usd", "avg_pips"]},
        "system3": {k: s3[k] for k in ["trades", "wr_pct", "pf", "net_usd", "avg_pips"]},
        "verdict": verdict,
    }

    # edge over random
    ctrl_pf = dataset_out["systems"]["6"]["summary"]["pf"]
    edge = []
    for sid in range(1, 6):
        ss = dataset_out["systems"][str(sid)]["summary"]
        imp = ss["pf"] - ctrl_pf
        meaningful = (imp > 0.15) and (ss["trades"] > 100)
        edge.append({
            "system": sid,
            "system_pf": ss["pf"],
            "control_pf": ctrl_pf,
            "pf_improvement": imp,
            "statistically_meaningful": meaningful,
        })
    dataset_out["edge_over_random"] = edge

    return dataset_out, sys_trades


def run_grid_on_best(df1000: pd.DataFrame, best_sid: int, units: int, allowed_days: set[str]) -> list[dict]:
    grid = [
        (2, 3), (2, 4), (3, 3), (3, 4), (3, 5), (4, 4), (4, 5), (4, 6),
        (5, 5), (5, 6), (5, 7), (6, 6), (6, 8), (8, 8), (8, 10),
    ]
    # baseline control PF from system 6 with original settings on 1000k
    ctrl_t, ctrl_d = backtest_system(df1000, 6, units, allowed_days)
    ctrl_pf = summarize_system(ctrl_t, ctrl_d, summarize_months(df1000))["pf"]

    out = []
    for tp, sl in grid:
        t, d = backtest_system(df1000, best_sid, units, allowed_days, tp_override=float(tp), sl_override=float(sl))
        s = summarize_system(t, d, summarize_months(df1000))
        out.append({
            "tp": tp,
            "sl": sl,
            "trades": s["trades"],
            "wr_pct": s["wr_pct"],
            "pf": s["pf"],
            "net_usd": s["net_usd"],
            "avg_trade_usd": (s["net_usd"] / s["trades"]) if s["trades"] else 0.0,
            "edge_over_random_pf": s["pf"] - ctrl_pf,
        })
    return out


def main():
    args = parse_args()
    cfg = json.loads(Path(args.config).read_text())

    out_dir = Path(cfg["outputs"]["base_dir"]) 
    out_dir.mkdir(parents=True, exist_ok=True)

    units = int(cfg["account"]["fixed_units"]) 
    allowed_days = set(cfg["session"]["allowed_days"])

    datasets = cfg["run_sequence"]
    all_results = {
        "config_path": args.config,
        "datasets": {},
    }
    system_logs_1000 = {}

    for run in datasets:
        label = run["label"]
        ds_res, sys_trades = run_dataset(run["input_csv"], units, allowed_days)
        all_results["datasets"][label] = ds_res
        print(f"[{label}] done")
        if label == "1000k":
            for sid in range(1, 7):
                system_logs_1000[sid] = sys_trades[sid]

    # write per-system 1000k logs
    for sid in range(1, 7):
        t = system_logs_1000.get(sid, pd.DataFrame())
        t.to_csv(out_dir / f"tokyo_scalp_diagnostic_system{sid}_trade_logs.csv", index=False)

    # best system among 1-5 on 1000k by PF
    s1000 = all_results["datasets"]["1000k"]["systems"]
    candidates = []
    for sid in range(1, 6):
        sm = s1000[str(sid)]["summary"]
        candidates.append((sid, sm["pf"], sm["trades"]))
    best_sid = sorted(candidates, key=lambda x: (x[1], x[2]), reverse=True)[0][0]

    df1000 = load_m1(cfg["run_sequence"][-1]["input_csv"])
    grid_rows = run_grid_on_best(df1000, best_sid, units, allowed_days)
    all_results["best_system_1000k"] = {
        "system_id": best_sid,
        "system_name": SYSTEMS[best_sid]["name"],
    }
    all_results["tp_sl_grid_best_system_1000k"] = grid_rows

    # triage flags on 1000k
    s1000sum = {sid: s1000[str(sid)]["summary"] for sid in range(1, 7)}
    pf_ctrl = s1000sum[6]["pf"]
    all_below = all(s1000sum[sid]["pf"] < 0.85 for sid in range(1, 6))
    none_outperform = all((s1000sum[sid]["pf"] - pf_ctrl) <= 0.15 for sid in range(1, 6))

    positive = [sid for sid in range(1, 6) if s1000sum[sid]["pf"] > 1.0 and s1000sum[sid]["trades"] >= 200 and (s1000sum[sid]["pf"] - pf_ctrl) > 0.20]
    marginal = [sid for sid in range(1, 6) if 0.90 <= s1000sum[sid]["pf"] < 1.0 and (s1000sum[sid]["pf"] - pf_ctrl) > 0.15]

    if all_below and none_outperform:
        triage = "NO SCALPING EDGE DETECTED. Tokyo USDJPY shows no exploitable micro-structure at any TP/SL level tested. Combined with 4 prior failures across mean reversion and breakout approaches, this constitutes definitive evidence that USDJPY Tokyo hours do not offer a retail-accessible technical edge. Recommendation: Close USDJPY Tokyo investigation permanently. Redirect development resources to: 1) Enhancing existing US-session V15 system 2) Building London-session system on a different pair 3) Cross-pair or multi-session approaches"
    elif positive:
        sid = positive[0]
        triage = f"SCALPING EDGE DETECTED: System {sid}, PF {s1000sum[sid]['pf']:.3f}, {s1000sum[sid]['trades']} trades. Edge over random: +{(s1000sum[sid]['pf']-pf_ctrl):.3f} PF."
    elif marginal:
        sid = marginal[0]
        triage = f"MARGINAL SCALPING SIGNAL: System {sid}, PF {s1000sum[sid]['pf']:.3f}. Edge over random: +{(s1000sum[sid]['pf']-pf_ctrl):.3f} PF."
    else:
        triage = "NO SCALPING EDGE DETECTED."

    all_results["triage_flag"] = triage

    # additional required global lines helpers
    # best system min100
    min100 = [sid for sid in range(1, 7) if s1000sum[sid]["trades"] >= 100]
    best_min100 = sorted(min100, key=lambda sid: s1000sum[sid]["pf"], reverse=True)[0] if min100 else 6

    # best day across all systems min30 trades
    best_day = None
    best_hour = None
    day_candidates = []
    hour_candidates = []
    for sid in range(1, 7):
        details = s1000[str(sid)]["details"]
        for r in details["day_of_week"]:
            if r["trades"] >= 30:
                day_candidates.append((r["pf"], sid, r["day"], r["trades"]))
        for r in details["hour_of_entry"]:
            if r["trades"] >= 20:
                hour_candidates.append((r["pf"], sid, r["hour_utc"], r["trades"]))
    if day_candidates:
        p, sid, d, n = sorted(day_candidates, reverse=True)[0]
        best_day = {"day": d, "system": sid, "pf": p, "trades": n}
    if hour_candidates:
        p, sid, h, n = sorted(hour_candidates, reverse=True)[0]
        best_hour = {"hour_utc": h, "system": sid, "pf": p, "trades": n}

    # spread impact overall best system
    bs = s1000[str(best_min100)]["summary"]
    grid_best = sorted(grid_rows, key=lambda x: x["pf"], reverse=True)[0]

    all_results["required_summary_lines"] = {
        "best_system_min100": {
            "system": best_min100,
            "pf": s1000sum[best_min100]["pf"],
            "trades": s1000sum[best_min100]["trades"],
        },
        "mirror_test": all_results["datasets"]["1000k"]["mirror_test"],
        "random_control": {
            "pf": s1000sum[6]["pf"],
            "trades": s1000sum[6]["trades"],
        },
        "best_system_spread": s1000[str(best_min100)]["summary"]["spread_stats"],
        "optimal_tp_sl_grid": grid_best,
        "best_day_min30": best_day,
        "best_hour_min20": best_hour,
    }

    # advanced file shortcut
    adv = {
        "dataset_1000k": all_results["datasets"]["1000k"],
        "tp_sl_grid_best_system_1000k": grid_rows,
        "triage_flag": triage,
        "required_summary_lines": all_results["required_summary_lines"],
    }

    (out_dir / "tokyo_scalp_diagnostic_results.json").write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
    (out_dir / "tokyo_scalp_diagnostic_advanced.json").write_text(json.dumps(adv, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
