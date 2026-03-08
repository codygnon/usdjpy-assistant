#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PIP_SIZE = 0.01
ENTRY_RISK_PCT = 0.0075
DAILY_RISK_CAP_PCT = 0.015


DEFAULT_CONFIG: dict[str, Any] = {
    "account": {
        "starting_balance": 100000.0,
        "leverage": 50.0,
        "max_margin_usage_fraction_per_trade": 0.5,
        "max_open_positions": 2,
        "max_daily_risk_pct": DAILY_RISK_CAP_PCT,
    },
    "execution_model": {
        "spread_mode": "realistic",
        "spread_avg_pips": 1.6,
        "spread_min_pips": 1.0,
        "spread_max_pips": 3.5,
    },
    "session": {
        "active_days_utc": ["Tuesday", "Wednesday", "Thursday"],
        "entry_cutoff_minutes_before_ny_open": 60,
        "force_close_at_ny_open": True,
    },
    "arb": {
        "enabled": True,
        "breakout_buffer_pips": 7.0,
        "sl_buffer_pips": 3.0,
        "asian_range_min_pips": 25.0,
        "asian_range_max_pips": 65.0,
        "sl_min_pips": 15.0,
        "sl_max_pips": 40.0,
        "tp1_r_multiple": 1.0,
        "tp2_r_multiple": 2.0,
        "tp1_close_fraction": 0.5,
        "be_offset_pips_after_tp1": 1.0,
        "max_trades_per_day": 1,
    },
    "lmp": {
        "enabled": True,
        "impulse_minutes": 90,
        "impulse_range_min_pips": 20.0,
        "zone_fib_ratio": 0.5,
        "entry_touch_same_or_next_close": True,
        "sl_buffer_pips": 5.0,
        "sl_min_pips": 12.0,
        "sl_max_pips": 30.0,
        "tp2_extension_ratio": 0.618,
        "tp1_close_fraction": 0.5,
        "be_offset_pips_after_tp1": 1.0,
        "max_trades_per_day": 1,
        "ema_period_15m": 20,
    },
    "position_sizing": {
        "risk_per_trade_pct": ENTRY_RISK_PCT,
        "round_to_units": 100,
    },
    "reporting": {
        "exit_reasons": [
            "TP1_ONLY",
            "TP2_FULL",
            "SL_FULL",
            "SL_AFTER_TP1",
            "BE_STOP",
            "HARD_CLOSE",
        ]
    },
}


@dataclass
class Position:
    trade_id: int
    strategy: str
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
    day_name: str
    asian_range_pips: float | None = None
    impulse_range_pips: float | None = None
    entry_hour_utc: int = 0
    tp1_hit: bool = False
    tp1_time: pd.Timestamp | None = None
    tp1_units_closed: int = 0
    pnl_usd_realized: float = 0.0
    weighted_pips_sum: float = 0.0
    mfe_pips: float = 0.0
    mae_pips: float = 0.0
    exit_reason: str | None = None
    exit_time: pd.Timestamp | None = None
    exit_price_last: float | None = None
    be_price_after_tp1: float | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="London-only ARB+LMP preset backtest")
    p.add_argument("--input", required=True, help="Input USDJPY M1 CSV")
    p.add_argument("--config", required=True, help="Config JSON path")
    p.add_argument("--out-prefix", required=True, help="Output path prefix")
    return p.parse_args()


def day_name(ts: pd.Timestamp) -> str:
    return ts.day_name()


def is_active_day(ts: pd.Timestamp, active_days: set[str]) -> bool:
    return day_name(ts) in active_days


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


def make_m15(df: pd.DataFrame, ema_period: int) -> pd.DataFrame:
    x = df.set_index("time").resample("15min", label="left", closed="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    )
    x = x.dropna().reset_index()
    x["ema20"] = x["close"].ewm(span=ema_period, adjust=False).mean()
    x["bar_end"] = x["time"] + pd.Timedelta(minutes=14)
    return x


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
    largest_win = float(trades["pnl_usd"].max()) if not trades.empty else 0.0
    largest_loss = float(trades["pnl_usd"].min()) if not trades.empty else 0.0

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
    calmar = (mu * 252.0) / (max_dd_pct / 100.0) if max_dd_pct > 0 else 0.0

    span_days = max((t1 - t0).total_seconds() / 86400.0, 1.0)
    years = span_days / 365.25
    end_equity = float(equity_curve.iloc[-1]["equity"])
    ann = ((end_equity / start_balance) ** (1.0 / years) - 1.0) * 100.0 if years > 0 else 0.0
    avg_trades_week = float(len(trades) / (span_days / 7.0)) if span_days > 0 else 0.0

    outcomes = (trades["pnl_usd"] > 0).astype(int).tolist() if not trades.empty else []
    max_wins = seq_max(outcomes)
    max_losses = seq_max([1 - x for x in outcomes])

    return {
        "total_trades": int(len(trades)),
        "win_rate_pct": wr,
        "avg_winner_pips": avg_win_pips,
        "avg_winner_usd": avg_win_usd,
        "avg_loser_pips": avg_loss_pips,
        "avg_loser_usd": avg_loss_usd,
        "largest_win_usd": largest_win,
        "largest_loss_usd": largest_loss,
        "profit_factor": pf,
        "expectancy_per_trade_usd": expectancy,
        "max_consecutive_wins": int(max_wins),
        "max_consecutive_losses": int(max_losses),
        "max_drawdown_usd": max_dd_usd,
        "max_drawdown_pct": max_dd_pct,
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(calmar),
        "net_pnl_usd": net,
        "net_pnl_pct": float((end_equity - start_balance) / start_balance * 100.0),
        "annualized_return_pct": float(ann),
        "avg_trades_per_week": float(avg_trades_week),
        "avg_trade_duration_minutes": float(trades["trade_duration_minutes"].mean()) if not trades.empty else 0.0,
    }


def month_breakdown(trades: pd.DataFrame) -> list[dict[str, Any]]:
    if trades.empty:
        return []
    t = trades.copy()
    t["month"] = t["exit_time_utc"].dt.to_period("M").astype(str)
    g = t.groupby("month", as_index=False)["pnl_usd"].sum().rename(columns={"pnl_usd": "net_usd"})
    return g.to_dict(orient="records")


def day_breakdown(trades: pd.DataFrame) -> list[dict[str, Any]]:
    if trades.empty:
        return [{"day": d, "trades": 0, "net_usd": 0.0} for d in ["Tuesday", "Wednesday", "Thursday"]]
    rows: list[dict[str, Any]] = []
    for d in ["Tuesday", "Wednesday", "Thursday"]:
        x = trades[trades["day_of_week"] == d]
        gp = float(x.loc[x["pnl_usd"] > 0, "pnl_usd"].sum())
        gl = float(x.loc[x["pnl_usd"] < 0, "pnl_usd"].sum())
        pf = gp / abs(gl) if gl < 0 else (float("inf") if gp > 0 else 0.0)
        rows.append(
            {
                "day": d,
                "trades": int(len(x)),
                "win_rate_pct": float((x["pnl_usd"] > 0).mean() * 100.0) if not x.empty else 0.0,
                "profit_factor": float(pf),
                "net_usd": float(x["pnl_usd"].sum()) if not x.empty else 0.0,
            }
        )
    return rows


def hour_breakdown(trades: pd.DataFrame) -> list[dict[str, Any]]:
    if trades.empty:
        return []
    g = trades.groupby("entry_hour_utc", as_index=False).agg(
        trades=("trade_id", "count"),
        win_rate_pct=("pnl_usd", lambda s: float((s > 0).mean() * 100.0)),
        net_usd=("pnl_usd", "sum"),
    )
    gp = trades[trades["pnl_usd"] > 0].groupby("entry_hour_utc")["pnl_usd"].sum()
    gl = trades[trades["pnl_usd"] < 0].groupby("entry_hour_utc")["pnl_usd"].sum()
    pf_map: dict[int, float] = {}
    for h in g["entry_hour_utc"].tolist():
        p = float(gp.get(h, 0.0))
        l = float(gl.get(h, 0.0))
        pf_map[int(h)] = p / abs(l) if l < 0 else (float("inf") if p > 0 else 0.0)
    g["profit_factor"] = g["entry_hour_utc"].map(pf_map)
    return g.to_dict(orient="records")


def exit_breakdown(trades: pd.DataFrame, reasons: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in reasons:
        x = trades[trades["exit_reason"] == r]
        rows.append(
            {
                "exit_reason": r,
                "count": int(len(x)),
                "avg_pnl_usd": float(x["pnl_usd"].mean()) if not x.empty else 0.0,
                "net_usd": float(x["pnl_usd"].sum()) if not x.empty else 0.0,
            }
        )
    return rows


def _as_records(df: pd.DataFrame, cols: list[str]) -> list[dict[str, Any]]:
    if df.empty:
        return []
    out = df[cols].copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return out.to_dict(orient="records")


def _latest_completed_m15_ema(ema_by_end: pd.Series, ts: pd.Timestamp) -> float | None:
    idx = ema_by_end.index.searchsorted(ts, side="right") - 1
    if idx < 0:
        return None
    v = ema_by_end.iloc[idx]
    return None if pd.isna(v) else float(v)


def run_backtest(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    start_balance = float(cfg["account"]["starting_balance"])
    leverage = float(cfg["account"]["leverage"])
    max_margin_frac = float(cfg["account"]["max_margin_usage_fraction_per_trade"])
    max_open_positions = int(cfg["account"]["max_open_positions"])
    daily_risk_cap_pct = float(cfg["account"]["max_daily_risk_pct"])
    round_units = int(cfg["position_sizing"]["round_to_units"])
    risk_pct = float(cfg["position_sizing"]["risk_per_trade_pct"])
    active_days = set(cfg["session"]["active_days_utc"])
    entry_cutoff_min = int(cfg["session"]["entry_cutoff_minutes_before_ny_open"])

    ema_period = int(cfg["lmp"]["ema_period_15m"])
    m15 = make_m15(df, ema_period)
    ema_by_end = m15.set_index("bar_end")["ema20"].sort_index()

    trades_out: list[dict[str, Any]] = []
    equity_events: list[tuple[pd.Timestamp, float]] = []
    equity = start_balance
    trade_id = 0

    df = df.copy()
    df["day_utc"] = df["time"].dt.floor("D")
    grouped = {k: g.copy().reset_index(drop=True) for k, g in df.groupby("day_utc")}
    all_days = sorted(grouped.keys())

    for day in all_days:
        day_df = grouped[day]
        if day_df.empty:
            continue
        if not is_active_day(day, active_days):
            continue
        day_start = day
        london_h = uk_london_open_utc(day)
        ny_h = us_ny_open_utc(day)
        london_open = day_start + pd.Timedelta(hours=london_h)
        ny_open = day_start + pd.Timedelta(hours=ny_h)
        entry_start = london_open
        entry_end = ny_open - pd.Timedelta(minutes=entry_cutoff_min)
        hard_close = ny_open

        if day_df["time"].max() < hard_close:
            continue

        asian_mask = (day_df["time"] >= day_start) & (day_df["time"] < london_open)
        asian = day_df.loc[asian_mask]
        if asian.empty:
            continue
        asian_high = float(asian["high"].max())
        asian_low = float(asian["low"].min())
        asian_range_pips = (asian_high - asian_low) / PIP_SIZE

        impulse_end = london_open + pd.Timedelta(minutes=int(cfg["lmp"]["impulse_minutes"]))
        m15_impulse = m15[(m15["time"] >= london_open) & (m15["time"] < impulse_end)]
        lmp_mode: str | None = None
        impulse_high = impulse_low = impulse_range_pips = np.nan
        fib50 = np.nan
        if not m15_impulse.empty:
            impulse_high = float(m15_impulse["high"].max())
            impulse_low = float(m15_impulse["low"].min())
            impulse_range_pips = (impulse_high - impulse_low) / PIP_SIZE
            idx_h = int(m15_impulse["high"].idxmax())
            idx_l = int(m15_impulse["low"].idxmin())
            t_h = pd.Timestamp(m15.loc[idx_h, "time"])
            t_l = pd.Timestamp(m15.loc[idx_l, "time"])
            w = day_df[(day_df["time"] >= london_open) & (day_df["time"] < impulse_end)]
            if not w.empty:
                sub_open = float(w.iloc[0]["open"])
                sub_close = float(w.iloc[-1]["close"])
                min_range = float(cfg["lmp"]["impulse_range_min_pips"])
                bull = impulse_range_pips >= min_range and sub_close > sub_open and t_l < t_h
                bear = impulse_range_pips >= min_range and sub_close < sub_open and t_h < t_l
                if bull:
                    lmp_mode = "bull"
                    fib50 = impulse_low + (impulse_high - impulse_low) * float(cfg["lmp"]["zone_fib_ratio"])
                elif bear:
                    lmp_mode = "bear"
                    fib50 = impulse_high - (impulse_high - impulse_low) * float(cfg["lmp"]["zone_fib_ratio"])

        open_positions: list[Position] = []
        pending_entries: list[dict[str, Any]] = []
        daily_risk_committed = 0.0
        arb_done = False
        lmp_done = False
        lmp_touch_state: dict[str, Any] | None = None

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
                if not (entry_start <= ts < entry_end):
                    continue
                if len(open_positions) >= max_open_positions:
                    continue
                if any(p.strategy == pe["strategy"] for p in open_positions):
                    continue

                entry_price = ask_o if pe["direction"] == "long" else bid_o
                if pe["strategy"] == "ARB":
                    raw_sl = pe["raw_sl"]
                    sl, sl_pips = clamp_sl(
                        entry_price,
                        raw_sl,
                        pe["direction"],
                        float(cfg["arb"]["sl_min_pips"]),
                        float(cfg["arb"]["sl_max_pips"]),
                    )
                    risk_dist = sl_pips
                    sign = 1.0 if pe["direction"] == "long" else -1.0
                    tp1 = entry_price + sign * float(cfg["arb"]["tp1_r_multiple"]) * risk_dist * PIP_SIZE
                    tp2 = entry_price + sign * float(cfg["arb"]["tp2_r_multiple"]) * risk_dist * PIP_SIZE
                else:
                    raw_sl = pe["raw_sl"]
                    sl, sl_pips = clamp_sl(
                        entry_price,
                        raw_sl,
                        pe["direction"],
                        float(cfg["lmp"]["sl_min_pips"]),
                        float(cfg["lmp"]["sl_max_pips"]),
                    )
                    if pe["direction"] == "long":
                        tp1 = float(pe["impulse_high"])
                        tp2 = float(pe["impulse_high"] + float(cfg["lmp"]["tp2_extension_ratio"]) * (pe["impulse_high"] - pe["impulse_low"]))
                    else:
                        tp1 = float(pe["impulse_low"])
                        tp2 = float(pe["impulse_low"] - float(cfg["lmp"]["tp2_extension_ratio"]) * (pe["impulse_high"] - pe["impulse_low"]))

                risk_usd = equity * risk_pct
                if daily_risk_committed + risk_usd > equity * daily_risk_cap_pct:
                    continue
                units = int(math.floor(risk_usd / max(1e-9, sl_pips * pip_value_per_unit(entry_price)) / round_units) * round_units)
                if units <= 0:
                    continue
                used_margin = sum(p.remaining_units for p in open_positions) / leverage
                free_margin = max(0.0, equity - used_margin)
                req_margin = units / leverage
                if req_margin > max_margin_frac * free_margin:
                    continue

                trade_id += 1
                p = Position(
                    trade_id=trade_id,
                    strategy=pe["strategy"],
                    direction=pe["direction"],
                    entry_time=ts,
                    entry_price=float(entry_price),
                    sl_price=float(sl),
                    tp1_price=float(tp1),
                    tp2_price=float(tp2),
                    initial_units=units,
                    remaining_units=units,
                    sl_pips=float(sl_pips),
                    risk_usd_planned=float(risk_usd),
                    day_name=day_name(ts),
                    asian_range_pips=pe.get("asian_range_pips"),
                    impulse_range_pips=pe.get("impulse_range_pips"),
                    entry_hour_utc=int(ts.hour),
                )
                tp1_units = int(math.floor(units * float(cfg["arb"]["tp1_close_fraction" if pe["strategy"] == "ARB" else "tp1_close_fraction"]) / round_units) * round_units)
                p.tp1_units_closed = max(0, min(units, tp1_units))
                open_positions.append(p)
                daily_risk_committed += risk_usd
                if pe["strategy"] == "ARB":
                    arb_done = True
                else:
                    lmp_done = True

            # hard close
            if ts == hard_close:
                still: list[Position] = []
                for p in open_positions:
                    exit_px = bid_o if p.direction == "long" else ask_o
                    u = p.remaining_units
                    lpips, lusd = calc_leg_pnl(p.direction, p.entry_price, float(exit_px), u)
                    p.pnl_usd_realized += lusd
                    p.weighted_pips_sum += lpips * u
                    p.remaining_units = 0
                    p.exit_time = ts
                    p.exit_price_last = float(exit_px)
                    p.exit_reason = "TP1_ONLY" if p.tp1_hit else "HARD_CLOSE"
                    equity += p.pnl_usd_realized
                    equity_events.append((ts, p.pnl_usd_realized))
                    trades_out.append(
                        {
                            "trade_id": p.trade_id,
                            "strategy": p.strategy,
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
                            "pnl_pips": (p.weighted_pips_sum / p.initial_units) if p.initial_units > 0 else 0.0,
                            "pnl_usd": p.pnl_usd_realized,
                            "exit_reason": p.exit_reason,
                            "asian_range_pips": p.asian_range_pips,
                            "impulse_range_pips": p.impulse_range_pips,
                            "MAE_pips": p.mae_pips,
                            "MFE_pips": p.mfe_pips,
                            "position_units": p.initial_units,
                            "trade_duration_minutes": int((p.exit_time - p.entry_time).total_seconds() / 60.0),
                            "entry_hour_utc": p.entry_hour_utc,
                        }
                    )
                open_positions = still
                break

            # manage open positions on this candle
            survivors: list[Position] = []
            for p in open_positions:
                if p.direction == "long":
                    p.mfe_pips = max(p.mfe_pips, (float(row["high"]) - p.entry_price) / PIP_SIZE)
                    p.mae_pips = max(p.mae_pips, (p.entry_price - float(row["low"])) / PIP_SIZE)
                else:
                    p.mfe_pips = max(p.mfe_pips, (p.entry_price - float(row["low"])) / PIP_SIZE)
                    p.mae_pips = max(p.mae_pips, (float(row["high"]) - p.entry_price) / PIP_SIZE)

                def close_remaining(reason: str, exit_px: float) -> None:
                    nonlocal equity
                    u_ = p.remaining_units
                    lp, lu = calc_leg_pnl(p.direction, p.entry_price, float(exit_px), u_)
                    p.pnl_usd_realized += lu
                    p.weighted_pips_sum += lp * u_
                    p.remaining_units = 0
                    p.exit_time = ts
                    p.exit_price_last = float(exit_px)
                    p.exit_reason = reason
                    equity += p.pnl_usd_realized
                    equity_events.append((ts, p.pnl_usd_realized))
                    trades_out.append(
                        {
                            "trade_id": p.trade_id,
                            "strategy": p.strategy,
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
                            "pnl_pips": (p.weighted_pips_sum / p.initial_units) if p.initial_units > 0 else 0.0,
                            "pnl_usd": p.pnl_usd_realized,
                            "exit_reason": p.exit_reason,
                            "asian_range_pips": p.asian_range_pips,
                            "impulse_range_pips": p.impulse_range_pips,
                            "MAE_pips": p.mae_pips,
                            "MFE_pips": p.mfe_pips,
                            "position_units": p.initial_units,
                            "trade_duration_minutes": int((p.exit_time - p.entry_time).total_seconds() / 60.0),
                            "entry_hour_utc": p.entry_hour_utc,
                        }
                    )

                if p.direction == "long":
                    px_open = bid_o
                    px_high = bid_h
                    px_low = bid_l
                else:
                    px_open = ask_o
                    px_high = ask_h
                    px_low = ask_l

                def choose_first(level_a: float, level_b: float) -> str:
                    da = abs(px_open - level_a)
                    db = abs(px_open - level_b)
                    return "a" if da <= db else "b"

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
                        p.be_price_after_tp1 = p.entry_price + (PIP_SIZE if p.direction == "long" else -PIP_SIZE)
                        p.sl_price = p.be_price_after_tp1
                        if p.remaining_units <= 0:
                            close_remaining("TP1_ONLY", p.tp1_price)
                            continue

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
                        be = p.be_price_after_tp1 if p.be_price_after_tp1 is not None else p.entry_price
                        reason = "BE_STOP" if abs(p.sl_price - be) <= (0.2 * PIP_SIZE) else "SL_AFTER_TP1"
                        close_remaining(reason, p.sl_price)
                        continue

                survivors.append(p)
            open_positions = survivors

            # signal generation uses closed candle
            if not (entry_start <= ts < entry_end):
                continue

            # ARB first valid signal
            if bool(cfg["arb"]["enabled"]) and not arb_done:
                if float(cfg["arb"]["asian_range_min_pips"]) <= asian_range_pips <= float(cfg["arb"]["asian_range_max_pips"]):
                    long_break = float(row["close"]) > asian_high + float(cfg["arb"]["breakout_buffer_pips"]) * PIP_SIZE
                    short_break = float(row["close"]) < asian_low - float(cfg["arb"]["breakout_buffer_pips"]) * PIP_SIZE
                    if long_break or short_break:
                        if i + 1 < len(day_df):
                            nxt_ts = pd.Timestamp(day_df.iloc[i + 1]["time"])
                            direction = "long" if long_break else "short"
                            raw_sl = asian_low - float(cfg["arb"]["sl_buffer_pips"]) * PIP_SIZE if direction == "long" else asian_high + float(cfg["arb"]["sl_buffer_pips"]) * PIP_SIZE
                            pending_entries.append(
                                {
                                    "strategy": "ARB",
                                    "direction": direction,
                                    "execute_time": nxt_ts,
                                    "raw_sl": raw_sl,
                                    "asian_range_pips": asian_range_pips,
                                }
                            )
                            arb_done = True

            # LMP first valid signal
            if bool(cfg["lmp"]["enabled"]) and not lmp_done and lmp_mode is not None and ts >= impulse_end:
                ema_now = _latest_completed_m15_ema(ema_by_end, ts)
                if ema_now is not None:
                    zone_upper = max(ema_now, fib50)
                    zone_lower = min(ema_now, fib50)

                    if lmp_touch_state is not None:
                        if ts == lmp_touch_state["confirm_ts"]:
                            if lmp_mode == "bull":
                                ok = float(row["close"]) > zone_upper
                                direction = "long"
                            else:
                                ok = float(row["close"]) < zone_lower
                                direction = "short"
                            if ok and i + 1 < len(day_df):
                                nxt_ts = pd.Timestamp(day_df.iloc[i + 1]["time"])
                                pending_entries.append(
                                    {
                                        "strategy": "LMP",
                                        "direction": direction,
                                        "execute_time": nxt_ts,
                                        "raw_sl": lmp_touch_state["raw_sl"],
                                        "impulse_high": impulse_high,
                                        "impulse_low": impulse_low,
                                        "impulse_range_pips": impulse_range_pips,
                                    }
                                )
                                lmp_done = True
                            lmp_touch_state = None

                    if not lmp_done and lmp_touch_state is None:
                        if lmp_mode == "bull":
                            touched = float(row["low"]) <= zone_upper
                            same_confirm = float(row["close"]) > zone_upper
                            raw_sl = float(row["low"]) - float(cfg["lmp"]["sl_buffer_pips"]) * PIP_SIZE
                            direction = "long"
                        else:
                            touched = float(row["high"]) >= zone_lower
                            same_confirm = float(row["close"]) < zone_lower
                            raw_sl = float(row["high"]) + float(cfg["lmp"]["sl_buffer_pips"]) * PIP_SIZE
                            direction = "short"
                        if touched:
                            if same_confirm and i + 1 < len(day_df):
                                nxt_ts = pd.Timestamp(day_df.iloc[i + 1]["time"])
                                pending_entries.append(
                                    {
                                        "strategy": "LMP",
                                        "direction": direction,
                                        "execute_time": nxt_ts,
                                        "raw_sl": raw_sl,
                                        "impulse_high": impulse_high,
                                        "impulse_low": impulse_low,
                                        "impulse_range_pips": impulse_range_pips,
                                    }
                                )
                                lmp_done = True
                            elif i + 1 < len(day_df):
                                lmp_touch_state = {"confirm_ts": pd.Timestamp(day_df.iloc[i + 1]["time"]), "raw_sl": raw_sl}

    trades_df = pd.DataFrame(trades_out)
    if trades_df.empty:
        trades_df = pd.DataFrame(
            columns=[
                "trade_id",
                "strategy",
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
                "pnl_pips",
                "pnl_usd",
                "exit_reason",
                "asian_range_pips",
                "impulse_range_pips",
                "MAE_pips",
                "MFE_pips",
                "position_units",
                "trade_duration_minutes",
                "entry_hour_utc",
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
    return trades_df, equity_curve


def build_strategy_equity_curve(
    full_timeline: pd.DataFrame, trades: pd.DataFrame, start_balance: float
) -> pd.DataFrame:
    if full_timeline.empty:
        return pd.DataFrame(columns=["timestamp", "equity"])
    x = full_timeline[["timestamp"]].copy()
    if trades.empty:
        x["equity"] = start_balance
        return x
    ev = trades.groupby("exit_time_utc", as_index=False)["pnl_usd"].sum().rename(columns={"exit_time_utc": "timestamp", "pnl_usd": "delta"})
    x = x.merge(ev, on="timestamp", how="left")
    x["delta"] = x["delta"].fillna(0.0)
    x["equity"] = start_balance + x["delta"].cumsum()
    return x[["timestamp", "equity"]]


def write_outputs(
    out_prefix: str,
    cfg: dict[str, Any],
    trades: pd.DataFrame,
    equity_curve: pd.DataFrame,
    dataset_path: str,
) -> dict[str, Any]:
    start_balance = float(cfg["account"]["starting_balance"])
    t0 = pd.Timestamp(equity_curve["timestamp"].min()) if not equity_curve.empty else pd.Timestamp.utcnow().tz_localize("UTC")
    t1 = pd.Timestamp(equity_curve["timestamp"].max()) if not equity_curve.empty else t0

    arb_trades = trades[trades["strategy"] == "ARB"].copy()
    lmp_trades = trades[trades["strategy"] == "LMP"].copy()

    arb_eq = build_strategy_equity_curve(equity_curve, arb_trades, start_balance)
    lmp_eq = build_strategy_equity_curve(equity_curve, lmp_trades, start_balance)

    sections = {
        "ARB": (arb_trades, arb_eq),
        "LMP": (lmp_trades, lmp_eq),
        "COMBINED": (trades, equity_curve),
    }
    report: dict[str, Any] = {
        "dataset": dataset_path,
        "config_snapshot": cfg,
        "results": {},
    }
    reasons = list(cfg["reporting"]["exit_reasons"])
    for name, (tdf, eq) in sections.items():
        report["results"][name] = {
            "aggregate_metrics": performance_metrics(tdf, eq, start_balance, t0, t1),
            "monthly_pnl_usd": month_breakdown(tdf),
            "pnl_by_day_of_week": day_breakdown(tdf),
            "pnl_by_entry_hour_utc": hour_breakdown(tdf),
            "exit_reason_distribution": exit_breakdown(tdf, reasons),
        }

    out_base = Path(out_prefix)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    trades_path = str(out_base.with_name(out_base.name + "_trades.csv"))
    equity_path = str(out_base.with_name(out_base.name + "_equity.csv"))
    report_path = str(out_base.with_name(out_base.name + "_report.json"))

    export_cols = [
        "trade_id",
        "strategy",
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
        "pnl_pips",
        "pnl_usd",
        "exit_reason",
        "asian_range_pips",
        "impulse_range_pips",
        "MAE_pips",
        "MFE_pips",
        "position_units",
        "trade_duration_minutes",
    ]
    trades.to_csv(trades_path, index=False)
    equity_curve.to_csv(equity_path, index=False)

    report["paths"] = {"report_json": report_path, "trades_csv": trades_path, "equity_csv": equity_path}
    report["records"] = {
        "trade_log": _as_records(trades, export_cols)[:1000],  # cap inline payload; full data is in CSV
        "equity_curve_preview": _as_records(equity_curve.tail(2000), ["timestamp", "equity"]),
        "equity_curve_full_csv": equity_path,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    return report


def merge_config(user_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    for k, v in user_cfg.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg_user = json.load(f)
    cfg = merge_config(cfg_user)

    df = pd.read_csv(args.input)
    if not {"time", "open", "high", "low", "close"}.issubset(df.columns):
        raise ValueError("Input CSV must include columns: time, open, high, low, close")
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)

    trades, equity_curve = run_backtest(df, cfg)
    report = write_outputs(args.out_prefix, cfg, trades, equity_curve, args.input)

    print(json.dumps(
        {
            "dataset": args.input,
            "results": {
                k: {
                    "trades": report["results"][k]["aggregate_metrics"]["total_trades"],
                    "wr_pct": round(report["results"][k]["aggregate_metrics"]["win_rate_pct"], 2),
                    "pf": round(report["results"][k]["aggregate_metrics"]["profit_factor"], 3) if math.isfinite(report["results"][k]["aggregate_metrics"]["profit_factor"]) else "inf",
                    "net_usd": round(report["results"][k]["aggregate_metrics"]["net_pnl_usd"], 2),
                    "max_dd_usd": round(report["results"][k]["aggregate_metrics"]["max_drawdown_usd"], 2),
                }
                for k in ["ARB", "LMP", "COMBINED"]
            },
            "paths": report["paths"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
