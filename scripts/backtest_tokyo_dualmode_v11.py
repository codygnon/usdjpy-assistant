#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PIP_SIZE = 0.01
UTC = "UTC"


@dataclass
class Position:
    trade_id: int
    mode: str
    direction: str
    phase: str
    entry_time: pd.Timestamp
    entry_session_day: str
    entry_price: float
    sl_price: float
    tp1_price: float
    tp2_price: float
    tp1_close_pct: float
    units_initial: int
    units_remaining: int
    trail_activate_pips: float
    trail_dist_pips: float
    entry_indicators: dict
    tp1_hit: bool = False
    trail_active: bool = False
    trail_stop_price: Optional[float] = None
    realized_usd: float = 0.0
    realized_pip_units: float = 0.0
    max_profit_seen_pips: float = 0.0
    max_adverse_seen_pips: float = 0.0
    exit_reason: Optional[str] = None
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tokyo dual-mode V11 backtest")
    p.add_argument("--config", required=True)
    return p.parse_args()


def load_m1(path: str) -> pd.DataFrame:
    f = Path(path)
    if not f.exists():
        raise FileNotFoundError(f"Missing CSV: {f}")
    df = pd.read_csv(f)
    need = {"time", "open", "high", "low", "close"}
    if not need.issubset(df.columns):
        raise ValueError(f"CSV missing columns: {need}")
    cols = ["time", "open", "high", "low", "close"] + (["volume"] if "volume" in df.columns else [])
    out = df[cols].copy()
    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
    for c in ["open", "high", "low", "close"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if "volume" in out.columns:
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    out = out.dropna(subset=["time", "open", "high", "low", "close"]).sort_values("time").reset_index(drop=True)
    if "volume" not in out.columns:
        out["volume"] = 1.0
    out["volume"] = out["volume"].fillna(1.0)
    return out


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return (
        df.set_index("time")
        .resample(rule, label="right", closed="right")
        .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("volume", "sum"))
        .dropna()
        .reset_index()
        .sort_values("time")
        .reset_index(drop=True)
    )


def rolling_rsi(series: pd.Series, period: int) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0.0)
    dn = -d.clip(upper=0.0)
    au = up.rolling(period, min_periods=period).mean()
    ad = dn.rolling(period, min_periods=period).mean()
    rs = au / ad.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def rolling_atr(df: pd.DataFrame, period: int) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([(df["high"] - df["low"]).abs(), (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def rolling_adx(df: pd.DataFrame, period: int) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    up = h.diff()
    dn = -l.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=period).mean()
    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).rolling(period, min_periods=period).mean() / atr.replace(0.0, np.nan)
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).rolling(period, min_periods=period).mean() / atr.replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.rolling(period, min_periods=period).mean()


def compute_psar(df: pd.DataFrame, af_start: float, af_step: float, af_max: float) -> tuple[pd.Series, pd.Series]:
    n = len(df)
    if n == 0:
        return pd.Series(dtype=float), pd.Series(dtype=object)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    psar = np.zeros(n, dtype=float)
    dirn = np.empty(n, dtype=object)
    bull = True if n < 2 else close[1] >= close[0]
    ep = high[0] if bull else low[0]
    psar[0] = low[0] if bull else high[0]
    af = af_start
    dirn[0] = "bullish" if bull else "bearish"
    for i in range(1, n):
        prev_psar = psar[i - 1]
        cand = prev_psar + af * (ep - prev_psar)
        if bull:
            cand = min(cand, low[i - 1], low[i - 2] if i > 1 else low[i - 1])
            if low[i] < cand:
                bull = False
                psar[i] = ep
                ep = low[i]
                af = af_start
            else:
                psar[i] = cand
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_step, af_max)
        else:
            cand = max(cand, high[i - 1], high[i - 2] if i > 1 else high[i - 1])
            if high[i] > cand:
                bull = True
                psar[i] = ep
                ep = high[i]
                af = af_start
            else:
                psar[i] = cand
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_step, af_max)
        dirn[i] = "bullish" if bull else "bearish"
    return pd.Series(psar, index=df.index), pd.Series(dirn, index=df.index)


def get_bid_ask(mid_price: float, spread_pips: float) -> tuple[float, float]:
    hs = spread_pips * PIP_SIZE / 2.0
    return float(mid_price) - hs, float(mid_price) + hs


def calc_leg_usd_pips(direction: str, entry_price: float, exit_price: float, units: int) -> tuple[float, float]:
    if direction == "long":
        pips = (float(exit_price) - float(entry_price)) / PIP_SIZE
    else:
        pips = (float(entry_price) - float(exit_price)) / PIP_SIZE
    usd = pips * float(units) * (PIP_SIZE / max(1e-9, float(exit_price)))
    return float(pips), float(usd)


def hhmm_to_min(s: str) -> int:
    hh, mm = s.split(":")
    return int(hh) * 60 + int(mm)


def build_indicators(m1: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = m1.copy()
    m5 = resample_ohlc(out, "5min")
    m15 = resample_ohlc(out, "15min")

    bb_cfg = cfg["indicators"]["bollinger_bands"]
    rsi_cfg = cfg["indicators"]["rsi"]
    atr_cfg = cfg["indicators"]["atr"]
    adx_cfg = cfg["indicators"]["adx"]

    bb_p = int(bb_cfg.get("period", 20))
    bb_std = float(bb_cfg.get("std_dev", 2.0))
    mid = m5["close"].rolling(bb_p, min_periods=bb_p).mean()
    std = m5["close"].rolling(bb_p, min_periods=bb_p).std(ddof=0)
    m5["bb_mid"] = mid
    m5["bb_upper"] = mid + bb_std * std
    m5["bb_lower"] = mid - bb_std * std
    m5["rsi_m5"] = rolling_rsi(m5["close"], int(rsi_cfg.get("period", 14)))
    m5["ema20_m5"] = m5["close"].ewm(span=int(cfg["indicators"]["ema"]["period"]), adjust=False).mean()
    roc_p = int(cfg["indicators"]["roc"]["period"])
    m5["roc10_m5"] = (m5["close"] - m5["close"].shift(roc_p)) / m5["close"].shift(roc_p)

    m15["atr_m15"] = rolling_atr(m15, int(atr_cfg.get("period", 14)))
    m15["adx_m15"] = rolling_adx(m15, int(adx_cfg.get("period", 14)))

    ps = cfg["indicators"]["parabolic_sar"]
    out["sar_value"], out["sar_direction"] = compute_psar(out, float(ps["af_start"]), float(ps["af_increment"]), float(ps["af_max"]))

    out = pd.merge_asof(
        out.sort_values("time"),
        m5[["time", "open", "high", "low", "close", "volume", "bb_upper", "bb_mid", "bb_lower", "rsi_m5", "ema20_m5", "roc10_m5"]]
        .rename(columns={"open": "m5_open", "high": "m5_high", "low": "m5_low", "close": "m5_close", "volume": "m5_volume"})
        .sort_values("time"),
        on="time",
        direction="backward",
    )
    out = pd.merge_asof(out.sort_values("time"), m15[["time", "atr_m15", "adx_m15"]].sort_values("time"), on="time", direction="backward")

    out["ny_day"] = (out["time"] - pd.Timedelta(hours=22)).dt.date
    day = out.groupby("ny_day").agg(day_high=("high", "max"), day_low=("low", "min"), day_close=("close", "last")).reset_index()
    day["prev_high"] = day["day_high"].shift(1)
    day["prev_low"] = day["day_low"].shift(1)
    day["prev_close"] = day["day_close"].shift(1)
    day["pivot_P"] = (day["prev_high"] + day["prev_low"] + day["prev_close"]) / 3.0
    rng = day["prev_high"] - day["prev_low"]
    day["pivot_R1"] = day["pivot_P"] + 0.382 * rng
    day["pivot_R2"] = day["pivot_P"] + 0.618 * rng
    day["pivot_R3"] = day["pivot_P"] + 1.0 * rng
    day["pivot_S1"] = day["pivot_P"] - 0.382 * rng
    day["pivot_S2"] = day["pivot_P"] - 0.618 * rng
    day["pivot_S3"] = day["pivot_P"] - 1.0 * rng
    out = out.merge(day[["ny_day", "pivot_P", "pivot_R1", "pivot_R2", "pivot_R3", "pivot_S1", "pivot_S2", "pivot_S3"]], on="ny_day", how="left")
    return out


def load_news_events(cfg: dict) -> pd.DataFrame:
    nf = cfg.get("news_filter", {})
    if not bool(nf.get("enabled", False)):
        return pd.DataFrame(columns=["event_ts", "impact", "event_id", "source"]) 
    p = Path(str(nf.get("calendar_path", "research_out/v5_scheduled_events_utc.csv")))
    if not p.exists():
        return pd.DataFrame(columns=["event_ts", "impact", "event_id", "source"]) 
    ev = pd.read_csv(p)
    if not {"date", "time_utc", "event_id", "impact"}.issubset(ev.columns):
        return pd.DataFrame(columns=["event_ts", "impact", "event_id", "source"]) 
    ev["event_ts"] = pd.to_datetime(ev["date"].astype(str) + " " + ev["time_utc"].astype(str), utc=True, errors="coerce")
    ev = ev.dropna(subset=["event_ts"]).copy()
    if "source" not in ev.columns:
        ev["source"] = ""
    return ev.sort_values("event_ts").reset_index(drop=True)


def summarize_metrics(tdf: pd.DataFrame, starting_equity: float) -> tuple[dict, pd.DataFrame]:
    if tdf.empty:
        return {
            "total_trades": 0,
            "win_rate_pct": 0.0,
            "average_win_pips": 0.0,
            "average_win_usd": 0.0,
            "average_loss_pips": 0.0,
            "average_loss_usd": 0.0,
            "largest_win_usd": 0.0,
            "largest_loss_usd": 0.0,
            "profit_factor": 0.0,
            "net_profit_usd": 0.0,
            "net_profit_pips": 0.0,
            "return_on_starting_equity_pct": 0.0,
            "max_drawdown_usd": 0.0,
            "max_drawdown_pct": 0.0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
            "average_trade_duration_minutes": 0.0,
            "sharpe_ratio": 0.0,
            "calmar_ratio": 0.0,
            "starting_equity_usd": float(starting_equity),
            "ending_equity_usd": float(starting_equity),
        }, pd.DataFrame(columns=["trade_number", "time", "equity", "drawdown_usd", "drawdown_pct"])

    wins = tdf[tdf["usd"] > 0]
    losses = tdf[tdf["usd"] < 0]
    gross_pos = wins["usd"].sum()
    gross_neg = abs(losses["usd"].sum())
    pf = float(gross_pos / gross_neg) if gross_neg > 0 else float("inf")
    net_usd = float(tdf["usd"].sum())
    net_pips = float(tdf["pips"].sum())
    wr = float((tdf["usd"] > 0).mean() * 100.0)

    eq = tdf["equity_after"].to_numpy(dtype=float)
    peak = -1e18
    dd = []
    dd_pct = []
    max_dd = 0.0
    max_dd_pct = 0.0
    for v in eq:
        peak = max(peak, float(v))
        d = max(0.0, peak - float(v))
        dp = (d / peak * 100.0) if peak > 0 else 0.0
        dd.append(d)
        dd_pct.append(dp)
        max_dd = max(max_dd, d)
        max_dd_pct = max(max_dd_pct, dp)

    returns = (tdf["usd"] / tdf["equity_before"].replace(0.0, np.nan)).fillna(0.0)
    sharpe = float((returns.mean() / returns.std(ddof=0)) * np.sqrt(len(returns))) if returns.std(ddof=0) > 0 else 0.0
    ret_pct = (float(eq[-1]) - starting_equity) / starting_equity * 100.0
    calmar = float(ret_pct / max_dd_pct) if max_dd_pct > 0 else 0.0

    consec_w = consec_l = max_w = max_l = 0
    last_win = None
    for u in tdf["usd"].to_list():
        win = u > 0
        if win:
            consec_w = consec_w + 1 if last_win else 1
            consec_l = 0
            max_w = max(max_w, consec_w)
        else:
            consec_l = consec_l + 1 if (last_win is False) else 1
            consec_w = 0
            max_l = max(max_l, consec_l)
        last_win = win

    summary = {
        "total_trades": int(len(tdf)),
        "win_rate_pct": wr,
        "average_win_pips": float(wins["pips"].mean()) if len(wins) else 0.0,
        "average_win_usd": float(wins["usd"].mean()) if len(wins) else 0.0,
        "average_loss_pips": float(losses["pips"].mean()) if len(losses) else 0.0,
        "average_loss_usd": float(losses["usd"].mean()) if len(losses) else 0.0,
        "largest_win_usd": float(tdf["usd"].max()),
        "largest_loss_usd": float(tdf["usd"].min()),
        "profit_factor": pf,
        "net_profit_usd": net_usd,
        "net_profit_pips": net_pips,
        "return_on_starting_equity_pct": ret_pct,
        "max_drawdown_usd": float(max_dd),
        "max_drawdown_pct": float(max_dd_pct),
        "max_consecutive_wins": int(max_w),
        "max_consecutive_losses": int(max_l),
        "average_trade_duration_minutes": float(tdf["duration_minutes"].mean()),
        "sharpe_ratio": sharpe,
        "calmar_ratio": calmar,
        "starting_equity_usd": float(starting_equity),
        "ending_equity_usd": float(eq[-1]),
    }
    eq_df = pd.DataFrame(
        {
            "trade_number": np.arange(1, len(tdf) + 1, dtype=int),
            "time": tdf["exit_datetime"],
            "equity": eq,
            "drawdown_usd": dd,
            "drawdown_pct": dd_pct,
        }
    )
    return summary, eq_df


def run_one_v11(cfg: dict, run_cfg: dict, events: pd.DataFrame) -> dict:
    df = build_indicators(load_m1(run_cfg["input_csv"]), cfg)
    df["time_utc"] = df["time"].dt.tz_convert("UTC")
    df["utc_date"] = df["time_utc"].dt.date
    df["utc_day_name"] = df["time_utc"].dt.day_name()
    df["hour_utc"] = df["time_utc"].dt.hour
    df["minute_utc"] = df["time_utc"].dt.minute
    df["minute_of_day_utc"] = df["hour_utc"] * 60 + df["minute_utc"]
    df["is_m5_close"] = (df["minute_utc"] % 5) == 0

    sf = cfg["session_filter"]
    allowed_days = set(sf.get("allowed_trading_days", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]))

    phases = cfg["session_phases"]
    p_m_start, p_m_end = hhmm_to_min(phases["morning_active"]["start_utc"]), hhmm_to_min(phases["morning_active"]["end_utc"])
    p_d_start, p_d_end = hhmm_to_min(phases["lunch_deadzone"]["start_utc"]), hhmm_to_min(phases["lunch_deadzone"]["end_utc"])
    p_a_start, p_a_end = hhmm_to_min(phases["afternoon_active"]["start_utc"]), hhmm_to_min(phases["afternoon_active"]["end_utc"])

    def phase_of(minute_of_day: int) -> str:
        if p_m_start <= minute_of_day < p_m_end:
            return "morning_active"
        if p_d_start <= minute_of_day < p_d_end:
            return "lunch_deadzone"
        if p_a_start <= minute_of_day < p_a_end:
            return "afternoon_active"
        return "off"

    df["phase"] = df["minute_of_day_utc"].apply(phase_of)
    df["in_v11_session"] = df["phase"].isin(["morning_active", "lunch_deadzone", "afternoon_active"])
    df["trading_day_ok"] = df["utc_day_name"].isin(allowed_days)

    # Daily ATR for day-type detection denominator.
    daily = (
        df.groupby("utc_date", as_index=False)
        .agg(day_open=("open", "first"), day_high=("high", "max"), day_low=("low", "min"), day_close=("close", "last"), avg_atr_m15=("atr_m15", "mean"))
        .sort_values("utc_date")
        .reset_index(drop=True)
    )
    daily["daily_range_pips"] = (daily["day_high"] - daily["day_low"]) / PIP_SIZE
    daily["atr14d"] = daily["avg_atr_m15"].rolling(14, min_periods=1).mean()
    daily_map = daily.set_index("utc_date").to_dict(orient="index")

    nf = cfg.get("news_filter", {})
    news_enabled = bool(nf.get("enabled", False))
    pre_block = int(nf.get("pre_event_block_minutes", 30))
    post_block = int(nf.get("post_event_block_minutes", 60))
    hi_pre = int(nf.get("high_impact_block_minutes_pre", 60))
    hi_post = int(nf.get("high_impact_block_minutes_post", 90))
    news_trend_cfg = nf.get("news_trend_mode", {})
    news_trend_enabled = bool(news_trend_cfg.get("enabled", True))
    news_move_min = float(news_trend_cfg.get("min_move_pips", 15.0))
    news_entry_window = int(news_trend_cfg.get("entry_window_after_block_minutes", 30))

    starting_equity = float(cfg.get("starting_equity_usd", 100000.0))
    equity = starting_equity
    spread_pips = float(cfg.get("execution_model", {}).get("spread_pips", 1.6))

    sizing = cfg["position_sizing"]
    base_risk = float(sizing.get("base_risk_pct", 1.0)) / 100.0
    mode_adj = sizing.get("mode_adjustments", {"mean_reversion": 1.0, "breakout": 0.75, "news_reaction": 0.5})
    max_units = int(sizing.get("max_position_units", 1000000))
    max_open = int(sizing.get("max_concurrent_positions", 2))
    max_trades_phase = int(sizing.get("max_trades_per_phase", 3))
    max_trades_session = int(sizing.get("max_trades_per_session", 5))

    tm = cfg["trade_management"]
    min_gap_min = int(tm.get("min_time_between_entries_minutes", 10))

    er_mr = cfg["mode_rules"]["mean_reversion"]
    er_bo = cfg["mode_rules"]["breakout"]

    open_positions: list[Position] = []
    closed_rows: list[dict] = []
    diag = Counter()
    trade_id = 0

    session_state: dict[str, dict] = {}
    last_entry_ts: Optional[pd.Timestamp] = None

    # Pre-group events by utc_date for speed.
    event_by_date = defaultdict(list)
    if news_enabled and not events.empty:
        for _, e in events.iterrows():
            event_by_date[pd.Timestamp(e["event_ts"]).date()].append(
                {
                    "event_ts": pd.Timestamp(e["event_ts"]),
                    "impact": str(e.get("impact", "")).lower(),
                    "event_id": str(e.get("event_id", "")),
                    "source": str(e.get("source", "")),
                    "handled_news_trend": False,
                }
            )

    detected_news_prints = []
    classified_prints = []
    first_trades_by_mode = {"mean_reversion": [], "breakout": [], "news_reaction": []}

    def get_session_key(ts_utc: pd.Timestamp) -> str:
        return str(ts_utc.date())

    def ensure_session(sk: str, row: pd.Series) -> dict:
        if sk not in session_state:
            d_events = event_by_date.get(pd.Timestamp(row["time_utc"]).date(), [])
            session_state[sk] = {
                "trades": 0,
                "morning_trades": 0,
                "afternoon_trades": 0,
                "morning_pnl": 0.0,
                "session_pnl": 0.0,
                "phase_stop_afternoon": False,
                "sl_consec_mode": {"mean_reversion": 0, "breakout": 0, "news_reaction": 0},
                "switch_mode_to": None,
                "session_open": float(row["open"]),
                "session_high": float(row["high"]),
                "session_low": float(row["low"]),
                "twap_sum": 0.0,
                "twap_n": 0,
                "vwap_num": 0.0,
                "vwap_den": 0.0,
                "first_hour_open": float(row["open"]),
                "first_hour_high": float(row["high"]),
                "first_hour_low": float(row["low"]),
                "first_hour_close": float(row["close"]),
                "first_hour_done": False,
                "day_type": "UNCLASSIFIED",
                "day_type_metrics": {},
                "first_hour_range_high": None,
                "first_hour_range_low": None,
                "pending_breakout": None,
                "events": d_events,
                "news_pre_price": {},
            }
        return session_state[sk]

    def in_news_block(ts: pd.Timestamp, sst: dict) -> tuple[bool, Optional[dict], int, int]:
        if not news_enabled:
            return False, None, 0, 0
        for ev in sst["events"]:
            e_ts = pd.Timestamp(ev["event_ts"])
            imp = str(ev.get("impact", "")).lower()
            bpre = hi_pre if imp == "high" else pre_block
            bpost = hi_post if imp == "high" else post_block
            if (e_ts - pd.Timedelta(minutes=bpre)) <= ts <= (e_ts + pd.Timedelta(minutes=bpost)):
                return True, ev, bpre, bpost
        return False, None, 0, 0

    def current_session_mean(sst: dict) -> float:
        if sst["vwap_den"] > 0:
            return sst["vwap_num"] / sst["vwap_den"]
        if sst["twap_n"] > 0:
            return sst["twap_sum"] / sst["twap_n"]
        return np.nan

    def close_position(pos: Position, ts: pd.Timestamp, exit_px: float, reason: str, equity_before: float) -> None:
        nonlocal equity
        if pos.units_remaining > 0:
            pips, usd = calc_leg_usd_pips(pos.direction, pos.entry_price, exit_px, pos.units_remaining)
            pos.realized_pip_units += pips * pos.units_remaining
            pos.realized_usd += usd
            pos.units_remaining = 0
        pos.exit_reason = reason
        pos.exit_time = ts
        pos.exit_price = float(exit_px)
        total_pips = pos.realized_pip_units / max(1, pos.units_initial)
        equity += pos.realized_usd

        sk = pos.entry_session_day
        sst = session_state.get(sk, None)
        if sst is not None:
            sst["session_pnl"] += pos.realized_usd
            if pos.phase == "morning_active":
                sst["morning_pnl"] += pos.realized_usd
            if "sl" in reason:
                sst["sl_consec_mode"][pos.mode] = int(sst["sl_consec_mode"].get(pos.mode, 0)) + 1
                if sst["sl_consec_mode"][pos.mode] >= 2:
                    if pos.mode == "mean_reversion":
                        sst["switch_mode_to"] = "breakout"
                    elif pos.mode == "breakout":
                        sst["switch_mode_to"] = "mean_reversion"
            else:
                sst["sl_consec_mode"][pos.mode] = 0
            if sst["morning_pnl"] <= (-0.015 * starting_equity):
                sst["phase_stop_afternoon"] = True

        row = {
            "trade_id": pos.trade_id,
            "entry_datetime": str(pos.entry_time),
            "exit_datetime": str(ts),
            "direction": pos.direction,
            "mode": pos.mode,
            "phase": pos.phase,
            "entry_price": pos.entry_price,
            "exit_price": pos.exit_price,
            "sl_price": pos.sl_price,
            "tp1_price": pos.tp1_price,
            "tp2_price": pos.tp2_price,
            "exit_reason": pos.exit_reason,
            "pips": float(total_pips),
            "usd": float(pos.realized_usd),
            "position_size_units": int(pos.units_initial),
            "duration_minutes": (ts - pos.entry_time).total_seconds() / 60.0,
            "equity_before": float(equity_before),
            "equity_after": float(equity),
            "mfe_pips": float(pos.max_profit_seen_pips),
            "mae_pips": float(pos.max_adverse_seen_pips),
            "entry_session_day": pos.entry_session_day,
            "entry_regime": "TOKYO_DUAL_MODE_V11",
            "entry_signal_mode": pos.mode,
            "entry_session": "tokyo",
        }
        row.update(pos.entry_indicators)
        closed_rows.append(row)

        if len(first_trades_by_mode.get(pos.mode, [])) < 3:
            first_trades_by_mode[pos.mode].append(
                {
                    "entry": str(pos.entry_time),
                    "direction": pos.direction,
                    "entry_price": pos.entry_price,
                    "sl": pos.sl_price,
                    "tp1": pos.tp1_price,
                    "tp2": pos.tp2_price,
                }
            )

    def try_open(direction: str, mode: str, phase: str, ts: pd.Timestamp, row: pd.Series, sst: dict, entry_px: float, sl_px: float, tp1_px: float, tp2_px: float, tp1_close_pct: float, trail_activate: float, trail_dist: float, entry_indicators: dict) -> bool:
        nonlocal trade_id, last_entry_ts
        if len(open_positions) >= max_open:
            diag["blocked_max_open"] += 1
            return False
        if sst["trades"] >= max_trades_session:
            diag["blocked_max_trades_session"] += 1
            return False
        if phase == "morning_active" and sst["morning_trades"] >= max_trades_phase:
            diag["blocked_max_trades_phase"] += 1
            return False
        if phase == "afternoon_active" and sst["afternoon_trades"] >= int(cfg["trade_management"].get("afternoon_max_trades", 2)):
            diag["blocked_max_trades_phase"] += 1
            return False
        if phase == "afternoon_active" and bool(sst.get("phase_stop_afternoon", False)):
            diag["blocked_afternoon_loss_stop"] += 1
            return False
        if last_entry_ts is not None and (ts - last_entry_ts).total_seconds() < min_gap_min * 60.0:
            diag["blocked_min_gap"] += 1
            return False

        sl_pips = abs(entry_px - sl_px) / PIP_SIZE
        if sl_pips <= 0:
            return False

        risk = base_risk * float(mode_adj.get(mode, 1.0))
        # Friday afternoon half-size.
        if str(row["utc_day_name"]) == "Friday" and phase == "afternoon_active":
            risk *= 0.5

        units = math.floor((equity * risk) / (sl_pips * (PIP_SIZE / max(entry_px, 1e-9))))
        units = int(max(0, min(max_units, units)))
        if units < 1:
            diag["blocked_units_lt_1"] += 1
            return False

        trade_id += 1
        pos = Position(
            trade_id=trade_id,
            mode=mode,
            direction=direction,
            phase=phase,
            entry_time=ts,
            entry_session_day=get_session_key(ts),
            entry_price=float(entry_px),
            sl_price=float(sl_px),
            tp1_price=float(tp1_px),
            tp2_price=float(tp2_px),
            tp1_close_pct=float(tp1_close_pct),
            units_initial=units,
            units_remaining=units,
            trail_activate_pips=float(trail_activate),
            trail_dist_pips=float(trail_dist),
            entry_indicators=entry_indicators,
        )
        open_positions.append(pos)
        sst["trades"] += 1
        if phase == "morning_active":
            sst["morning_trades"] += 1
        elif phase == "afternoon_active":
            sst["afternoon_trades"] += 1
        last_entry_ts = ts
        diag["entries_total"] += 1
        diag[f"entries_{mode}"] += 1
        return True

    # Verification print #1.
    print("Session phase mapping UTC: 00:00-02:00 morning_active, 02:00-04:00 lunch_deadzone, 04:00-06:00 afternoon_active")
    print(f"Account start equity: ${starting_equity:,.0f}")

    for i, row in df.iterrows():
        ts = pd.Timestamp(row["time_utc"])
        minute = int(row["minute_of_day_utc"])
        phase = str(row["phase"])
        in_session = bool(row["in_v11_session"]) and bool(row["trading_day_ok"])

        bid_o, ask_o = get_bid_ask(float(row["open"]), spread_pips)
        bid_h, ask_h = get_bid_ask(float(row["high"]), spread_pips)
        bid_l, ask_l = get_bid_ask(float(row["low"]), spread_pips)
        bid_c, ask_c = get_bid_ask(float(row["close"]), spread_pips)

        # Update and process open positions first.
        for pos in list(open_positions):
            if pos.direction == "long":
                fav = (bid_h - pos.entry_price) / PIP_SIZE
                adv = (pos.entry_price - bid_l) / PIP_SIZE
            else:
                fav = (pos.entry_price - ask_l) / PIP_SIZE
                adv = (ask_h - pos.entry_price) / PIP_SIZE
            pos.max_profit_seen_pips = max(pos.max_profit_seen_pips, float(fav))
            pos.max_adverse_seen_pips = max(pos.max_adverse_seen_pips, float(adv))

            # Trail activate rules.
            if (not pos.trail_active) and pos.max_profit_seen_pips >= pos.trail_activate_pips:
                pos.trail_active = True
            if pos.trail_active:
                if pos.direction == "long":
                    nt = bid_c - pos.trail_dist_pips * PIP_SIZE
                    pos.trail_stop_price = nt if pos.trail_stop_price is None else max(pos.trail_stop_price, nt)
                else:
                    nt = ask_c + pos.trail_dist_pips * PIP_SIZE
                    pos.trail_stop_price = nt if pos.trail_stop_price is None else min(pos.trail_stop_price, nt)

        if not in_session:
            # Force close outside session boundary.
            for pos in list(open_positions):
                px = bid_c if pos.direction == "long" else ask_c
                eq_before = equity
                close_position(pos, ts, px, "session_end_close", eq_before)
                open_positions.remove(pos)
            continue

        sk = get_session_key(ts)
        sst = ensure_session(sk, row)

        # Session mean update.
        sst["session_high"] = max(float(sst["session_high"]), float(row["high"]))
        sst["session_low"] = min(float(sst["session_low"]), float(row["low"]))
        sst["twap_sum"] += float(row["close"])
        sst["twap_n"] += 1
        vol = float(row.get("volume", 1.0) if not pd.isna(row.get("volume", np.nan)) else 1.0)
        sst["vwap_num"] += float(row["close"]) * max(1.0, vol)
        sst["vwap_den"] += max(1.0, vol)

        # First hour stats.
        if minute < 60:
            sst["first_hour_high"] = max(float(sst["first_hour_high"]), float(row["high"]))
            sst["first_hour_low"] = min(float(sst["first_hour_low"]), float(row["low"]))
            sst["first_hour_close"] = float(row["close"])
        if (not sst["first_hour_done"]) and minute >= 60:
            fh_range = (float(sst["first_hour_high"]) - float(sst["first_hour_low"])) / PIP_SIZE
            fh_dir = (float(sst["first_hour_close"]) - float(sst["first_hour_open"])) / PIP_SIZE
            dstat = daily_map.get(pd.Timestamp(row["utc_date"]), {})
            d_atr = float(dstat.get("atr14d", np.nan))
            fh_atr_ratio = (fh_range * PIP_SIZE / d_atr) if np.isfinite(d_atr) and d_atr > 0 else np.nan
            adx_now = float(row.get("adx_m15", np.nan))

            has_news_remaining = False
            for ev in sst["events"]:
                if pd.Timestamp(ev["event_ts"]) >= ts and pd.Timestamp(ev["event_ts"]).hour < 6:
                    has_news_remaining = True
                    break

            if has_news_remaining:
                dtyp = "NEWS"
            elif fh_range < float(cfg["day_type_detection"]["ranging_criteria"]["max_first_hour_range_pips"]) and abs(fh_dir) < float(cfg["day_type_detection"]["ranging_criteria"]["max_first_hour_direction_pips"]) and (not np.isfinite(adx_now) or adx_now < float(cfg["day_type_detection"]["ranging_criteria"]["max_adx"])):
                dtyp = "RANGING"
            elif fh_range >= float(cfg["day_type_detection"]["trending_criteria"]["min_first_hour_range_pips"]) or abs(fh_dir) >= float(cfg["day_type_detection"]["trending_criteria"]["min_first_hour_direction_pips"]) or (np.isfinite(adx_now) and adx_now >= float(cfg["day_type_detection"]["trending_criteria"]["min_adx"])):
                dtyp = "TRENDING"
            else:
                dtyp = "RANGING"

            sst["day_type"] = dtyp
            sst["day_type_metrics"] = {
                "first_hour_range_pips": float(fh_range),
                "first_hour_direction_pips": float(fh_dir),
                "first_hour_atr_ratio": float(fh_atr_ratio) if np.isfinite(fh_atr_ratio) else None,
                "adx_value": float(adx_now) if np.isfinite(adx_now) else None,
                "news_scheduled": bool(has_news_remaining),
            }
            sst["first_hour_range_high"] = float(sst["first_hour_high"])
            sst["first_hour_range_low"] = float(sst["first_hour_low"])
            sst["first_hour_done"] = True
            if len(classified_prints) < 10:
                classified_prints.append({"session": sk, "day_type": dtyp, **sst["day_type_metrics"]})

        # News block detection and handling.
        news_block, active_event, bpre, bpost = in_news_block(ts, sst)
        if news_block:
            diag["trades_blocked_by_news"] += 1
            # tighten existing trails to 3 pips.
            for pos in open_positions:
                pos.trail_active = True
                pos.trail_dist_pips = min(float(pos.trail_dist_pips), 3.0)
            if active_event is not None:
                e_ts = pd.Timestamp(active_event["event_ts"])
                if ts <= e_ts and str(active_event["event_id"]) not in sst["news_pre_price"]:
                    sst["news_pre_price"][str(active_event["event_id"])] = float(row["close"])
                if len(detected_news_prints) < 5:
                    detected_news_prints.append(
                        {
                            "event": str(active_event["event_id"]),
                            "impact": str(active_event["impact"]),
                            "event_ts": str(e_ts),
                            "bar_ts": str(ts),
                            "action": f"block_entries+tighten_trail (pre={bpre} post={bpost})",
                        }
                    )

        # Manage exits after potential trail tighten.
        for pos in list(open_positions):
            # lunch handling at start of deadzone.
            if phase == "lunch_deadzone" and int(row["minute_utc"]) == 0 and int(row["hour_utc"]) == 2:
                cp = (bid_c - pos.entry_price) / PIP_SIZE if pos.direction == "long" else (pos.entry_price - ask_c) / PIP_SIZE
                if cp > 5.0:
                    pos.trail_active = True
                    pos.trail_dist_pips = min(pos.trail_dist_pips, 4.0)
                else:
                    eq_before = equity
                    px = bid_c if pos.direction == "long" else ask_c
                    close_position(pos, ts, px, "phase_lunch_close", eq_before)
                    open_positions.remove(pos)
                    continue

            if pos.direction == "long":
                hit_sl = bid_l <= pos.sl_price
                hit_tp1 = (not pos.tp1_hit) and (bid_h >= pos.tp1_price)
                hit_tp2 = bid_h >= pos.tp2_price
                hit_trail = pos.trail_active and pos.trail_stop_price is not None and bid_l <= pos.trail_stop_price
            else:
                hit_sl = ask_h >= pos.sl_price
                hit_tp1 = (not pos.tp1_hit) and (ask_l <= pos.tp1_price)
                hit_tp2 = ask_l <= pos.tp2_price
                hit_trail = pos.trail_active and pos.trail_stop_price is not None and ask_h >= pos.trail_stop_price

            if hit_sl:
                eq_before = equity
                close_position(pos, ts, float(pos.sl_price), "sl", eq_before)
                open_positions.remove(pos)
                continue

            if hit_tp1:
                cu = int(max(1, min(pos.units_remaining, math.floor(pos.units_initial * pos.tp1_close_pct))))
                leg_pips, leg_usd = calc_leg_usd_pips(pos.direction, pos.entry_price, pos.tp1_price, cu)
                pos.realized_pip_units += leg_pips * cu
                pos.realized_usd += leg_usd
                pos.units_remaining -= cu
                pos.tp1_hit = True
                pos.trail_active = True
                if pos.units_remaining <= 0:
                    eq_before = equity
                    close_position(pos, ts, float(pos.tp1_price), "tp", eq_before)
                    open_positions.remove(pos)
                    continue

            if pos.units_remaining > 0 and hit_tp2:
                eq_before = equity
                close_position(pos, ts, float(pos.tp2_price), "tp2", eq_before)
                open_positions.remove(pos)
                continue

            if pos.units_remaining > 0 and hit_trail:
                eq_before = equity
                close_position(pos, ts, float(pos.trail_stop_price), "trailing_stop", eq_before)
                open_positions.remove(pos)
                continue

        # No new entries in deadzone.
        if phase == "lunch_deadzone":
            continue

        # Force close at 06:00.
        if minute >= p_a_end:
            for pos in list(open_positions):
                eq_before = equity
                px = bid_c if pos.direction == "long" else ask_c
                close_position(pos, ts, px, "session_end_close", eq_before)
                open_positions.remove(pos)
            continue

        # Entry logic only on M5 closes.
        if not bool(row["is_m5_close"]):
            continue
        if news_block:
            continue

        if not np.isfinite(float(row.get("pivot_P", np.nan))) or not np.isfinite(float(row.get("atr_m15", np.nan))) or not np.isfinite(float(row.get("bb_upper", np.nan))):
            continue

        P = float(row["pivot_P"])
        R1, R2, R3 = float(row["pivot_R1"]), float(row["pivot_R2"]), float(row["pivot_R3"])
        S1, S2, S3 = float(row["pivot_S1"]), float(row["pivot_S2"]), float(row["pivot_S3"])
        bb_u, bb_l = float(row["bb_upper"]), float(row["bb_lower"])
        rsi = float(row["rsi_m5"])
        atr_pips = float(row["atr_m15"]) / PIP_SIZE
        adx = float(row.get("adx_m15", np.nan))
        ema20 = float(row.get("ema20_m5", np.nan))
        roc10 = float(row.get("roc10_m5", np.nan))
        m5o = float(row.get("m5_open", row["open"]))
        m5c = float(row.get("m5_close", row["close"]))

        # Mode availability.
        day_type = str(sst.get("day_type", "UNCLASSIFIED"))
        allowed_modes = set(phases[phase].get("modes_allowed", []))
        if minute < 60:
            mode_pref = ["mean_reversion", "breakout"]
        else:
            if day_type == "RANGING":
                mode_pref = ["mean_reversion"]
            elif day_type == "TRENDING":
                mode_pref = ["breakout"]
            elif day_type == "NEWS":
                mode_pref = ["news_reaction", "breakout", "mean_reversion"]
            else:
                mode_pref = ["mean_reversion", "breakout"]

        if sst.get("switch_mode_to") in {"mean_reversion", "breakout"}:
            mode_pref = [sst["switch_mode_to"]] + [m for m in mode_pref if m != sst["switch_mode_to"]]

        mode_pref = [m for m in mode_pref if m in allowed_modes or m == "news_reaction"]

        session_mean = current_session_mean(sst)

        opened_any = False

        for mode in mode_pref:
            if opened_any:
                break

            # NEWS REACTION mode.
            if mode == "news_reaction" and news_trend_enabled:
                for ev in sst["events"]:
                    if ev.get("handled_news_trend", False):
                        continue
                    e_ts = pd.Timestamp(ev["event_ts"])
                    imp = str(ev.get("impact", "")).lower()
                    bpost2 = hi_post if imp == "high" else post_block
                    wnd_start = e_ts + pd.Timedelta(minutes=bpost2)
                    wnd_end = wnd_start + pd.Timedelta(minutes=news_entry_window)
                    if not (wnd_start <= ts <= wnd_end):
                        continue
                    pre_px = float(sst["news_pre_price"].get(str(ev["event_id"]), np.nan))
                    if not np.isfinite(pre_px):
                        pre_px = float(row["open"])
                    move_pips = (float(row["close"]) - pre_px) / PIP_SIZE
                    if abs(move_pips) < news_move_min:
                        ev["handled_news_trend"] = True
                        continue
                    direction = "long" if move_pips > 0 else "short"
                    entry_px = ask_c if direction == "long" else bid_c
                    # 50% retracement stop from pre-news anchor.
                    sl_pips_raw = max(8.0, min(30.0, abs(move_pips) * 0.5))
                    sl_px = entry_px - sl_pips_raw * PIP_SIZE if direction == "long" else entry_px + sl_pips_raw * PIP_SIZE
                    tp_pips = max(8.0, abs(move_pips) * 0.618)
                    tp1_px = entry_px + tp_pips * PIP_SIZE if direction == "long" else entry_px - tp_pips * PIP_SIZE
                    tp2_px = tp1_px
                    ok = try_open(
                        direction=direction,
                        mode="news_reaction",
                        phase=phase,
                        ts=ts,
                        row=row,
                        sst=sst,
                        entry_px=entry_px,
                        sl_px=sl_px,
                        tp1_px=tp1_px,
                        tp2_px=tp2_px,
                        tp1_close_pct=1.0,
                        trail_activate=8.0,
                        trail_dist=6.0,
                        entry_indicators={
                            "day_type": day_type,
                            "news_event_id": str(ev.get("event_id", "")),
                            "news_move_pips": float(move_pips),
                            "phase": phase,
                        },
                    )
                    if ok:
                        ev["handled_news_trend"] = True
                        opened_any = True
                        break
                continue

            # MEAN REVERSION mode.
            if mode == "mean_reversion":
                tol = float(er_mr.get("pivot_tolerance_pips", 10.0)) * PIP_SIZE
                c_zone_l = float(row["close"]) <= (S1 + tol)
                c_bb_l = (float(row["close"]) <= bb_l) or (float(row["low"]) <= bb_l)
                c_rsi_l = rsi < float(er_mr.get("rsi_long_max", 40.0))
                sc_l = int(c_zone_l) + int(c_bb_l) + int(c_rsi_l)

                c_zone_s = float(row["close"]) >= (R1 - tol)
                c_bb_s = (float(row["close"]) >= bb_u) or (float(row["high"]) >= bb_u)
                c_rsi_s = rsi > float(er_mr.get("rsi_short_min", 60.0))
                sc_s = int(c_zone_s) + int(c_bb_s) + int(c_rsi_s)

                if sc_l >= 2 and m5c > m5o:
                    direction = "long"
                elif sc_s >= 2 and m5c < m5o:
                    direction = "short"
                else:
                    continue

                entry_px = ask_c if direction == "long" else bid_c
                sl_buf = float(cfg["exit_rules"]["mean_reversion"]["sl_buffer_pips"]) * PIP_SIZE
                if direction == "long":
                    deep = float(row["close"]) <= (S2 + tol)
                    sl_raw = (S3 - sl_buf) if deep else (S2 - sl_buf)
                    sl_pips = (entry_px - sl_raw) / PIP_SIZE
                else:
                    deep = float(row["close"]) >= (R2 - tol)
                    sl_raw = (R3 + sl_buf) if deep else (R2 + sl_buf)
                    sl_pips = (sl_raw - entry_px) / PIP_SIZE
                sl_pips = max(float(cfg["exit_rules"]["mean_reversion"]["sl_min_pips"]), min(float(cfg["exit_rules"]["mean_reversion"]["sl_max_pips"]), sl_pips))
                sl_px = entry_px - sl_pips * PIP_SIZE if direction == "long" else entry_px + sl_pips * PIP_SIZE

                target = session_mean if np.isfinite(session_mean) else P
                if direction == "long":
                    tp_dist = (target - entry_px) / PIP_SIZE
                    if tp_dist < float(cfg["exit_rules"]["mean_reversion"]["tp_min_pips"]):
                        target = P
                else:
                    tp_dist = (entry_px - target) / PIP_SIZE
                    if tp_dist < float(cfg["exit_rules"]["mean_reversion"]["tp_min_pips"]):
                        target = P
                tp1_px = float(target)
                # runner by trail only.
                tp2_px = entry_px + (999.0 * PIP_SIZE if direction == "long" else -999.0 * PIP_SIZE)

                opened_any = try_open(
                    direction=direction,
                    mode="mean_reversion",
                    phase=phase,
                    ts=ts,
                    row=row,
                    sst=sst,
                    entry_px=entry_px,
                    sl_px=sl_px,
                    tp1_px=tp1_px,
                    tp2_px=tp2_px,
                    tp1_close_pct=float(cfg["exit_rules"]["mean_reversion"]["partial_close_pct"]),
                    trail_activate=float(cfg["exit_rules"]["mean_reversion"]["trail_activate_pips"]),
                    trail_dist=float(cfg["exit_rules"]["mean_reversion"]["trail_distance_pips"]),
                    entry_indicators={
                        "pivot_P": P,
                        "pivot_R1": R1,
                        "pivot_R2": R2,
                        "pivot_S1": S1,
                        "pivot_S2": S2,
                        "bb_upper": bb_u,
                        "bb_mid": float(row["bb_mid"]),
                        "bb_lower": bb_l,
                        "rsi_m5": rsi,
                        "atr_m15": float(row["atr_m15"]),
                        "adx_m15": adx if np.isfinite(adx) else None,
                        "session_mean": float(session_mean) if np.isfinite(session_mean) else None,
                        "phase": phase,
                        "day_type": day_type,
                    },
                )
                continue

            # BREAKOUT mode.
            if mode == "breakout":
                if not sst.get("first_hour_done", False):
                    continue
                fh_hi = float(sst["first_hour_range_high"])
                fh_lo = float(sst["first_hour_range_low"])
                bo_buf = float(er_bo.get("break_buffer_pips", 3.0)) * PIP_SIZE

                # pending confirm one-candle logic.
                pb = sst.get("pending_breakout", None)
                if pb is not None:
                    # one confirmation candle only
                    dir_pb = pb["direction"]
                    lvl = pb["level"]
                    confirmed = (float(row["close"]) > lvl) if dir_pb == "long" else (float(row["close"]) < lvl)
                    if confirmed:
                        direction = dir_pb
                        entry_px = ask_c if direction == "long" else bid_c
                        if direction == "long":
                            sl_raw = fh_lo - bo_buf
                            sl_pips = (entry_px - sl_raw) / PIP_SIZE
                        else:
                            sl_raw = fh_hi + bo_buf
                            sl_pips = (sl_raw - entry_px) / PIP_SIZE
                        if sl_pips > float(cfg["exit_rules"]["breakout"]["sl_max_pips"]):
                            sst["pending_breakout"] = None
                            continue
                        sl_pips = max(float(cfg["exit_rules"]["breakout"]["sl_min_pips"]), sl_pips)
                        sl_px = entry_px - sl_pips * PIP_SIZE if direction == "long" else entry_px + sl_pips * PIP_SIZE
                        tp_pips = max(float(cfg["exit_rules"]["breakout"]["tp_min_pips"]), min(float(cfg["exit_rules"]["breakout"]["tp_max_pips"]), float(cfg["exit_rules"]["breakout"]["tp_atr_mult"]) * atr_pips))
                        tp1_px = entry_px + tp_pips * PIP_SIZE if direction == "long" else entry_px - tp_pips * PIP_SIZE
                        tp2_px = tp1_px
                        opened_any = try_open(
                            direction=direction,
                            mode="breakout",
                            phase=phase,
                            ts=ts,
                            row=row,
                            sst=sst,
                            entry_px=entry_px,
                            sl_px=sl_px,
                            tp1_px=tp1_px,
                            tp2_px=tp2_px,
                            tp1_close_pct=1.0,
                            trail_activate=float(cfg["exit_rules"]["breakout"]["trail_activate_pips"]),
                            trail_dist=float(cfg["exit_rules"]["breakout"]["trail_distance_pips"]),
                            entry_indicators={
                                "first_hour_high": fh_hi,
                                "first_hour_low": fh_lo,
                                "ema20_m5": ema20 if np.isfinite(ema20) else None,
                                "roc10_m5": roc10 if np.isfinite(roc10) else None,
                                "adx_m15": adx if np.isfinite(adx) else None,
                                "phase": phase,
                                "day_type": day_type,
                            },
                        )
                    sst["pending_breakout"] = None
                    continue

                long_break = float(row["close"]) > (fh_hi + bo_buf)
                short_break = float(row["close"]) < (fh_lo - bo_buf)
                if not (long_break or short_break):
                    continue
                if not np.isfinite(adx) or adx <= float(er_bo.get("min_adx", 20.0)):
                    continue
                if long_break and (not np.isfinite(ema20) or float(row["close"]) <= ema20 or not np.isfinite(roc10) or roc10 <= 0):
                    continue
                if short_break and (not np.isfinite(ema20) or float(row["close"]) >= ema20 or not np.isfinite(roc10) or roc10 >= 0):
                    continue

                # Require one confirmation candle beyond level.
                sst["pending_breakout"] = {
                    "direction": "long" if long_break else "short",
                    "level": (fh_hi + bo_buf) if long_break else (fh_lo - bo_buf),
                }
                continue

    # End of file close.
    if open_positions:
        last = df.iloc[-1]
        ts = pd.Timestamp(last["time_utc"])
        bid, ask = get_bid_ask(float(last["close"]), spread_pips)
        for pos in list(open_positions):
            eq_before = equity
            px = bid if pos.direction == "long" else ask
            close_position(pos, ts, px, "session_end_close", eq_before)
            open_positions.remove(pos)

    tdf = pd.DataFrame(closed_rows)
    summary, eq_df = summarize_metrics(tdf, starting_equity)

    # Basic breakdowns.
    def pf_of(g: pd.DataFrame) -> float:
        if g.empty:
            return 0.0
        gp = g[g["usd"] > 0]["usd"].sum()
        gl = abs(g[g["usd"] < 0]["usd"].sum())
        return float(gp / gl) if gl > 0 else (float("inf") if gp > 0 else 0.0)

    mode_perf = {}
    for mode in ["mean_reversion", "breakout", "news_reaction"]:
        g = tdf[tdf["mode"] == mode] if not tdf.empty else tdf
        mode_perf[mode] = {
            "trades": int(len(g)),
            "wr": float((g["usd"] > 0).mean() * 100.0) if len(g) else 0.0,
            "pf": pf_of(g),
            "net": float(g["usd"].sum()) if len(g) else 0.0,
            "avg_win_pips": float(g[g["pips"] > 0]["pips"].mean()) if len(g[g["pips"] > 0]) else 0.0,
            "avg_loss_pips": float(g[g["pips"] < 0]["pips"].mean()) if len(g[g["pips"] < 0]) else 0.0,
            "morning_pf": pf_of(g[g["phase"] == "morning_active"]) if len(g) else 0.0,
            "afternoon_pf": pf_of(g[g["phase"] == "afternoon_active"]) if len(g) else 0.0,
        }

    phase_perf = {}
    for ph in ["morning_active", "afternoon_active"]:
        g = tdf[tdf["phase"] == ph] if not tdf.empty else tdf
        phase_perf[ph] = {
            "trades": int(len(g)),
            "wr": float((g["usd"] > 0).mean() * 100.0) if len(g) else 0.0,
            "pf": pf_of(g),
            "net": float(g["usd"].sum()) if len(g) else 0.0,
            "mode_distribution": {m: int((g["mode"] == m).sum()) if len(g) else 0 for m in ["mean_reversion", "breakout", "news_reaction"]},
        }

    # Verify deadzone has zero entries.
    dead_viol = int((tdf["phase"] == "lunch_deadzone").sum()) if not tdf.empty else 0

    # Day/Month/Hour analytics.
    if tdf.empty:
        day_table = []
        month_table = []
        hour_table = []
        cross_day_month = []
    else:
        tdf["entry_ts"] = pd.to_datetime(tdf["entry_datetime"], utc=True)
        tdf["day_name"] = tdf["entry_ts"].dt.day_name()
        tdf["month"] = tdf["entry_ts"].dt.month_name()
        tdf["month_num"] = tdf["entry_ts"].dt.month
        tdf["utc_hour"] = tdf["entry_ts"].dt.hour

        day_table = []
        for d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
            g = tdf[tdf["day_name"] == d]
            if g.empty:
                day_table.append({"day": d, "total_trades": 0, "mean_reversion_trades": 0, "breakout_trades": 0, "news_trades": 0, "overall_wr": 0.0, "overall_pf": 0.0, "overall_net": 0.0, "mr_wr": 0.0, "mr_pf": 0.0, "breakout_wr": 0.0, "breakout_pf": 0.0, "best_phase": None, "worst_phase": None})
                continue
            by_phase = g.groupby("phase")["usd"].sum().to_dict()
            bp = max(by_phase.items(), key=lambda kv: kv[1])[0] if by_phase else None
            wp = min(by_phase.items(), key=lambda kv: kv[1])[0] if by_phase else None
            gm = g[g["mode"] == "mean_reversion"]
            gb = g[g["mode"] == "breakout"]
            day_table.append(
                {
                    "day": d,
                    "total_trades": int(len(g)),
                    "mean_reversion_trades": int((g["mode"] == "mean_reversion").sum()),
                    "breakout_trades": int((g["mode"] == "breakout").sum()),
                    "news_trades": int((g["mode"] == "news_reaction").sum()),
                    "overall_wr": float((g["usd"] > 0).mean() * 100.0),
                    "overall_pf": pf_of(g),
                    "overall_net": float(g["usd"].sum()),
                    "mr_wr": float((gm["usd"] > 0).mean() * 100.0) if len(gm) else 0.0,
                    "mr_pf": pf_of(gm),
                    "breakout_wr": float((gb["usd"] > 0).mean() * 100.0) if len(gb) else 0.0,
                    "breakout_pf": pf_of(gb),
                    "best_phase": bp,
                    "worst_phase": wp,
                }
            )

        month_table = []
        dtypes = pd.DataFrame([
            {
                "session": k,
                "day_type": v.get("day_type", "UNCLASSIFIED"),
                "session_pnl": v.get("session_pnl", 0.0),
            }
            for k, v in session_state.items()
        ])
        for m in range(1, 13):
            g = tdf[tdf["month_num"] == m]
            if g.empty:
                continue
            mname = g.iloc[0]["month"]
            # day type distribution by session month.
            sess_month = dtypes.copy()
            sess_month["month_num"] = pd.to_datetime(sess_month["session"]).dt.month
            ds = sess_month[sess_month["month_num"] == m]
            dcount = ds["day_type"].value_counts().to_dict()
            md = g.groupby("mode")["usd"].sum().to_dict()
            month_table.append(
                {
                    "month": mname,
                    "total_trades": int(len(g)),
                    "overall_wr": float((g["usd"] > 0).mean() * 100.0),
                    "overall_pf": pf_of(g),
                    "net_pnl": float(g["usd"].sum()),
                    "avg_daily_range_pips": float(daily[daily["utc_date"].apply(lambda x: x.month == m)]["daily_range_pips"].mean()) if len(daily) else 0.0,
                    "day_type_distribution": {
                        "ranging": int(dcount.get("RANGING", 0)),
                        "trending": int(dcount.get("TRENDING", 0)),
                        "news": int(dcount.get("NEWS", 0)),
                    },
                    "best_mode": max(md.items(), key=lambda kv: kv[1])[0] if md else None,
                    "worst_mode": min(md.items(), key=lambda kv: kv[1])[0] if md else None,
                }
            )

        hour_table = []
        for h in range(0, 6):
            g = tdf[tdf["utc_hour"] == h]
            if g.empty:
                hour_table.append({"utc_hour": h, "jst_hour": h + 9, "phase": "morning_active" if h < 2 else ("lunch_deadzone" if h < 4 else "afternoon_active"), "trades": 0, "wr": 0.0, "pf": 0.0, "net": 0.0, "avg_trade_duration_min": 0.0, "mode_breakdown": {m: {"trades": 0, "wr": 0.0, "pf": 0.0} for m in ["mean_reversion", "breakout", "news_reaction"]}})
                continue
            mb = {}
            for m in ["mean_reversion", "breakout", "news_reaction"]:
                gm = g[g["mode"] == m]
                mb[m] = {"trades": int(len(gm)), "wr": float((gm["usd"] > 0).mean() * 100.0) if len(gm) else 0.0, "pf": pf_of(gm)}
            hour_table.append(
                {
                    "utc_hour": h,
                    "jst_hour": h + 9,
                    "phase": "morning_active" if h < 2 else ("lunch_deadzone" if h < 4 else "afternoon_active"),
                    "trades": int(len(g)),
                    "wr": float((g["usd"] > 0).mean() * 100.0),
                    "pf": pf_of(g),
                    "net": float(g["usd"].sum()),
                    "avg_trade_duration_min": float(g["duration_minutes"].mean()),
                    "mode_breakdown": mb,
                }
            )

        cross_day_month = []
        cross = tdf.groupby(["day_name", "month"])
        for (d, m), g in cross:
            cell = {"day": d, "month": m, "trades": int(len(g)), "pf": pf_of(g), "net": float(g["usd"].sum()), "flag_structural_weakness": bool(len(g) > 10 and pf_of(g) < 0.5)}
            cross_day_month.append(cell)

    # day type stats.
    session_records = []
    for sk, sst in session_state.items():
        dtyp = str(sst.get("day_type", "UNCLASSIFIED"))
        session_records.append({"session": sk, "day_type": dtyp, "session_pnl": float(sst.get("session_pnl", 0.0)), "trades": int(sst.get("trades", 0))})
    srec = pd.DataFrame(session_records)

    def pf_sessions(g: pd.DataFrame) -> float:
        if g.empty:
            return 0.0
        gp = g[g["session_pnl"] > 0]["session_pnl"].sum()
        gl = abs(g[g["session_pnl"] < 0]["session_pnl"].sum())
        return float(gp / gl) if gl > 0 else (float("inf") if gp > 0 else 0.0)

    dts = {
        "total_sessions": int(len(srec)),
        "ranging_days": int((srec["day_type"] == "RANGING").sum()) if len(srec) else 0,
        "trending_days": int((srec["day_type"] == "TRENDING").sum()) if len(srec) else 0,
        "news_days": int((srec["day_type"] == "NEWS").sum()) if len(srec) else 0,
        "ranging_day_pf": pf_sessions(srec[srec["day_type"] == "RANGING"]) if len(srec) else 0.0,
        "trending_day_pf": pf_sessions(srec[srec["day_type"] == "TRENDING"]) if len(srec) else 0.0,
        "news_day_pf": pf_sessions(srec[srec["day_type"] == "NEWS"]) if len(srec) else 0.0,
        "day_type_accuracy": float((srec[srec["session_pnl"] > 0].shape[0] / len(srec) * 100.0)) if len(srec) else 0.0,
    }

    # News filter stats.
    news_trades = tdf[tdf["mode"] == "news_reaction"] if not tdf.empty else tdf
    nfs = {
        "total_news_events": int(sum(len(v.get("events", [])) for v in session_state.values())),
        "high_impact_events": int(sum(sum(1 for e in v.get("events", []) if str(e.get("impact", "")).lower() == "high") for v in session_state.values())),
        "trades_blocked_by_news": int(diag["trades_blocked_by_news"]),
        "news_trend_trades_taken": int(len(news_trades)),
        "news_trend_wr": float((news_trades["usd"] > 0).mean() * 100.0) if len(news_trades) else 0.0,
        "news_trend_pf": pf_of(news_trades),
        "news_trend_net": float(news_trades["usd"].sum()) if len(news_trades) else 0.0,
        "estimated_pnl_of_blocked_trades": 0.0,
    }

    trades_out = (
        tdf.drop(columns=[c for c in ["entry_ts", "day_name", "month", "month_num", "utc_hour"] if c in tdf.columns], errors="ignore")
        if not tdf.empty
        else tdf
    )

    report = {
        "strategy_id": cfg.get("strategy_id", "tokyo_dual_mode_v11"),
        "run_label": run_cfg["label"],
        "input_csv": run_cfg["input_csv"],
        "summary": summary,
        "mode_performance": mode_perf,
        "phase_performance": {**phase_perf, "deadzone_violations": dead_viol},
        "day_type_stats": dts,
        "news_filter_stats": nfs,
        "day_of_week_table": day_table,
        "month_table": month_table,
        "hour_table": hour_table,
        "cross_day_month": cross_day_month,
        "diagnostics": {"counts": dict(diag), "session_count": int(len(session_state))},
        "equity_curve": eq_df.to_dict(orient="records"),
        "drawdown_curve": eq_df[["trade_number", "drawdown_usd", "drawdown_pct"]].to_dict(orient="records") if len(eq_df) else [],
        "trades": trades_out.to_dict(orient="records") if not trades_out.empty else [],
    }

    # Verification prints.
    print("News filter first 5 events detected/handled:")
    for x in detected_news_prints[:5]:
        print(f"  {x}")
    print("Day type first 10 sessions:")
    for x in classified_prints[:10]:
        print(f"  {x}")
    print("First 3 trades by mode:")
    for m in ["mean_reversion", "breakout", "news_reaction"]:
        print(f"  {m}:")
        for tr in first_trades_by_mode.get(m, [])[:3]:
            print(f"    {tr}")

    print(
        "Mode funnel summary: "
        f"MR={diag.get('entries_mean_reversion',0)} "
        f"BO={diag.get('entries_breakout',0)} "
        f"NEWS={diag.get('entries_news_reaction',0)}"
    )
    print(f"Deadzone violations: {dead_viol}")

    return report


def write_outputs(report: dict, run_cfg: dict) -> None:
    Path(run_cfg["output_json"]).write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(report.get("trades", [])).to_csv(run_cfg["output_trades_csv"], index=False)
    pd.DataFrame(report.get("equity_curve", [])).to_csv(run_cfg["output_equity_csv"], index=False)


def build_walkforward_comparison(train_report: dict, test_report: dict) -> dict:
    tr = train_report["summary"]
    te = test_report["summary"]
    pf_train = float(tr.get("profit_factor", 0.0))
    pf_test = float(te.get("profit_factor", 0.0))
    return {
        "walk_forward_validation": {
            "training": {
                "candles": 700000,
                "trades": int(tr.get("total_trades", 0)),
                "wr": float(tr.get("win_rate_pct", 0.0)),
                "pf": pf_train,
                "net": float(tr.get("net_profit_usd", 0.0)),
                "maxdd": float(tr.get("max_drawdown_usd", 0.0)),
            },
            "testing": {
                "candles": 300000,
                "trades": int(te.get("total_trades", 0)),
                "wr": float(te.get("win_rate_pct", 0.0)),
                "pf": pf_test,
                "net": float(te.get("net_profit_usd", 0.0)),
                "maxdd": float(te.get("max_drawdown_usd", 0.0)),
            },
            "degradation": {
                "pf_change": pf_test - pf_train,
                "pf_change_pct": ((pf_test - pf_train) / pf_train * 100.0) if pf_train else 0.0,
                "wr_change": float(te.get("win_rate_pct", 0.0) - tr.get("win_rate_pct", 0.0)),
                "pf_retention_pct": (pf_test / pf_train * 100.0) if pf_train else 0.0,
                "verdict": "ROBUST" if (pf_train and pf_test >= 0.7 * pf_train) else ("INCONCLUSIVE" if pf_train == 0 else "OVERFIT"),
            },
        }
    }


def main() -> int:
    args = parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    runs = list(cfg.get("run_sequence", []))
    if not runs:
        raise RuntimeError("Config missing run_sequence")

    events = load_news_events(cfg)
    reports = {}
    summaries = []

    for r in runs:
        report = run_one_v11(cfg, r, events)
        write_outputs(report, r)
        reports[str(r["label"]).lower()] = report
        s = report["summary"]
        summaries.append((str(r["label"]).lower(), s))
        print(
            f"[{r['label']}] trades={s['total_trades']} wr={s['win_rate_pct']:.2f}% "
            f"pf={s['profit_factor']:.3f} net={s['net_profit_usd']:.2f} maxdd={s['max_drawdown_usd']:.2f}"
        )

    # write walkforward comparison if both exist
    if "walkforward_700k" in reports and "walkforward_300k" in reports:
        cmp = build_walkforward_comparison(reports["walkforward_700k"], reports["walkforward_300k"])
        out = cfg.get("walkforward_comparison_output", "research_out/tokyo_dualmode_v11_walkforward_comparison.json")
        Path(out).write_text(json.dumps(cmp, indent=2), encoding="utf-8")
        print(f"walkforward comparison -> {out}")

    # extra day/month/hour outputs from 1000k run.
    if "1000k" in reports:
        r1000 = reports["1000k"]
        Path(cfg.get("day_analysis_output", "research_out/tokyo_dualmode_v11_day_analysis.json")).write_text(
            json.dumps(r1000.get("day_of_week_table", []), indent=2), encoding="utf-8"
        )
        Path(cfg.get("month_analysis_output", "research_out/tokyo_dualmode_v11_month_analysis.json")).write_text(
            json.dumps(r1000.get("month_table", []), indent=2), encoding="utf-8"
        )
        Path(cfg.get("hour_analysis_output", "research_out/tokyo_dualmode_v11_hour_analysis.json")).write_text(
            json.dumps(r1000.get("hour_table", []), indent=2), encoding="utf-8"
        )

    # benchmark prints
    if summaries:
        print("\nV11 scaling summary:")
        for lb, s in summaries:
            print(
                f"  {lb:16s} trades={int(s['total_trades']):4d} wr={s['win_rate_pct']:.2f}% "
                f"pf={s['profit_factor']:.3f} net=${s['net_profit_usd']:,.0f} maxdd=${s['max_drawdown_usd']:,.0f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
