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


@dataclass
class Position:
    trade_id: int
    strategy: str  # strat_a_meanrev | strat_b_breakout
    direction: str  # long | short
    entry_time: pd.Timestamp
    session_day: str
    phase: str
    entry_price: float
    sl_price: float
    tp1_price: float
    tp2_price: float
    tp1_close_pct: float
    units_initial: int
    units_remaining: int
    be_offset_pips: float
    trail_activate_pips: float
    trail_distance_pips: float
    time_decay_minutes: int
    time_decay_profit_cap_pips: float
    early_exit_minutes: int
    early_exit_min_profit_pips: float
    entry_indicators: dict
    moved_to_be: bool = False
    tp1_hit: bool = False
    trail_active: bool = False
    trail_stop_price: Optional[float] = None
    news_tightened: bool = False
    realized_usd: float = 0.0
    realized_pip_units: float = 0.0
    max_profit_seen_pips: float = 0.0
    max_adverse_seen_pips: float = 0.0
    exit_reason: Optional[str] = None
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tokyo dual strategy V12 backtest")
    p.add_argument("--config", required=True)
    return p.parse_args()


def load_m1(path: str) -> pd.DataFrame:
    f = Path(path)
    if not f.exists():
        raise FileNotFoundError(f"Missing CSV: {f}")
    df = pd.read_csv(f)
    need = {"time", "open", "high", "low", "close"}
    if not need.issubset(df.columns):
        raise ValueError(f"Missing required columns in {f}")
    cols = ["time", "open", "high", "low", "close"] + (["volume"] if "volume" in df.columns else [])
    out = df[cols].copy()
    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
    for c in ["open", "high", "low", "close"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if "volume" not in out.columns:
        out["volume"] = 1.0
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(1.0)
    out = out.dropna(subset=["time", "open", "high", "low", "close"]).sort_values("time").reset_index(drop=True)
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
        prev = psar[i - 1]
        cand = prev + af * (ep - prev)
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


def get_bid_ask(mid: float, spread_pips: float) -> tuple[float, float]:
    hs = spread_pips * PIP_SIZE / 2.0
    return float(mid) - hs, float(mid) + hs


def calc_leg_usd_pips(direction: str, entry_price: float, exit_price: float, units: int) -> tuple[float, float]:
    if direction == "long":
        pips = (float(exit_price) - float(entry_price)) / PIP_SIZE
    else:
        pips = (float(entry_price) - float(exit_price)) / PIP_SIZE
    usd = pips * float(units) * (PIP_SIZE / max(1e-9, float(exit_price)))
    return float(pips), float(usd)


def build_indicators(m1: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = m1.copy()
    m5 = resample_ohlc(out, "5min")
    m15 = resample_ohlc(out, "15min")

    ind = cfg["indicators"]
    bb_p = int(ind["bollinger_bands"]["period"])
    bb_std = float(ind["bollinger_bands"]["std_dev"])
    rsi_p = int(ind["rsi"]["period"])
    atr_p = int(ind["atr"]["period"])
    adx_p = int(ind["adx"]["period"])
    ema_p = int(ind["ema"]["period"])
    roc_p = int(ind["roc"]["period"])

    mid = m5["close"].rolling(bb_p, min_periods=bb_p).mean()
    std = m5["close"].rolling(bb_p, min_periods=bb_p).std(ddof=0)
    m5["bb_mid"] = mid
    m5["bb_upper"] = mid + bb_std * std
    m5["bb_lower"] = mid - bb_std * std
    m5["bb_width"] = (m5["bb_upper"] - m5["bb_lower"]) / m5["bb_mid"].replace(0.0, np.nan)
    bb_pct_window = int(cfg.get("strategy_a", {}).get("bb_regime_percentile_lookback_bars", 100))
    bb_pct_cutoff = float(cfg.get("strategy_a", {}).get("bb_regime_ranging_percentile", 0.8))
    m5["bb_width_cutoff"] = m5["bb_width"].rolling(bb_pct_window, min_periods=bb_pct_window).quantile(bb_pct_cutoff)
    m5["bb_regime"] = np.where(m5["bb_width"] < m5["bb_width_cutoff"], "ranging", "trending")
    m5["rsi_m5"] = rolling_rsi(m5["close"], rsi_p)
    m5["ema20_m5"] = m5["close"].ewm(span=ema_p, adjust=False).mean()
    m5["roc10_m5"] = (m5["close"] - m5["close"].shift(roc_p)) / m5["close"].shift(roc_p)

    m15["atr_m15"] = rolling_atr(m15, atr_p)
    m15["adx_m15"] = rolling_adx(m15, adx_p)

    ps = ind["parabolic_sar"]
    out["sar_value"], out["sar_direction"] = compute_psar(out, float(ps["af_start"]), float(ps["af_increment"]), float(ps["af_max"]))
    out["sar_flip_bull"] = (out["sar_direction"] == "bullish") & (out["sar_direction"].shift(1) == "bearish")
    out["sar_flip_bear"] = (out["sar_direction"] == "bearish") & (out["sar_direction"].shift(1) == "bullish")
    sar_lookback = int(cfg.get("strategy_a", {}).get("sar_flip_lookback_bars", 12))
    out["sar_flip_bull_recent"] = out["sar_flip_bull"].rolling(sar_lookback, min_periods=1).max().astype(bool)
    out["sar_flip_bear_recent"] = out["sar_flip_bear"].rolling(sar_lookback, min_periods=1).max().astype(bool)

    out = pd.merge_asof(
        out.sort_values("time"),
        m5[["time", "open", "high", "low", "close", "volume", "bb_upper", "bb_mid", "bb_lower", "bb_width", "bb_width_cutoff", "bb_regime", "rsi_m5", "ema20_m5", "roc10_m5"]]
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


def pf_from_usd(s: pd.Series) -> float:
    if len(s) == 0:
        return 0.0
    gp = s[s > 0].sum()
    gl = abs(s[s < 0].sum())
    return float(gp / gl) if gl > 0 else (float("inf") if gp > 0 else 0.0)


def summarize_strategy(tdf: pd.DataFrame, start_eq: float) -> dict:
    if tdf.empty:
        return {
            "trades": 0,
            "wr": 0.0,
            "pf": 0.0,
            "net_pnl": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "max_dd": 0.0,
            "avg_win_pips": 0.0,
            "avg_loss_pips": 0.0,
            "avg_rr_realized": 0.0,
            "expectancy_per_trade": 0.0,
            "avg_duration_minutes": 0.0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
            "sharpe_monthly": 0.0,
        }

    wins = tdf[tdf["usd"] > 0]
    losses = tdf[tdf["usd"] < 0]
    eq = tdf["equity_after"].to_numpy(dtype=float)
    peak = -1e18
    max_dd = 0.0
    for v in eq:
        peak = max(peak, float(v))
        max_dd = max(max_dd, max(0.0, peak - float(v)))

    consec_w = consec_l = max_w = max_l = 0
    last_win = None
    for u in tdf["usd"].to_list():
        w = u > 0
        if w:
            consec_w = consec_w + 1 if last_win else 1
            consec_l = 0
            max_w = max(max_w, consec_w)
        else:
            consec_l = consec_l + 1 if (last_win is False) else 1
            consec_w = 0
            max_l = max(max_l, consec_l)
        last_win = w

    tm = tdf.copy()
    tm["month"] = pd.to_datetime(tm["entry_datetime"], utc=True).dt.to_period("M").astype(str)
    monthly = tm.groupby("month")["usd"].sum()
    sharpe_monthly = float(monthly.mean() / monthly.std(ddof=0)) if monthly.std(ddof=0) > 0 else 0.0

    return {
        "trades": int(len(tdf)),
        "wr": float((tdf["usd"] > 0).mean() * 100.0),
        "pf": pf_from_usd(tdf["usd"]),
        "net_pnl": float(tdf["usd"].sum()),
        "gross_profit": float(tdf[tdf["usd"] > 0]["usd"].sum()),
        "gross_loss": float(tdf[tdf["usd"] < 0]["usd"].sum()),
        "max_dd": float(max_dd),
        "avg_win_pips": float(wins["pips"].mean()) if len(wins) else 0.0,
        "avg_loss_pips": float(losses["pips"].mean()) if len(losses) else 0.0,
        "avg_rr_realized": float(tdf["R"].mean()) if "R" in tdf.columns else 0.0,
        "expectancy_per_trade": float(tdf["usd"].mean()),
        "avg_duration_minutes": float(tdf["duration_minutes"].mean()),
        "max_consecutive_wins": int(max_w),
        "max_consecutive_losses": int(max_l),
        "sharpe_monthly": sharpe_monthly,
    }


def run_one(cfg: dict, run_cfg: dict, events: pd.DataFrame) -> dict:
    df = build_indicators(load_m1(run_cfg["input_csv"]), cfg)
    df["time_utc"] = df["time"].dt.tz_convert("UTC")
    df["utc_date"] = df["time_utc"].dt.date
    df["utc_day_name"] = df["time_utc"].dt.day_name()
    df["hour_utc"] = df["time_utc"].dt.hour
    df["minute_utc"] = df["time_utc"].dt.minute
    df["minute_of_day"] = df["hour_utc"] * 60 + df["minute_utc"]
    df["is_m5_close"] = (df["minute_utc"] % 5) == 0

    start_eq = float(cfg.get("starting_equity_usd", 100000.0))
    equity = start_eq
    spread = float(cfg["execution_model"].get("spread_pips", 1.6))
    b_time_exit_reason = f"time_exit_{int(cfg['strategy_b'].get('time_exit_minutes', 45))}min"

    # Windows.
    a_start = 16 * 60 + 30
    a_end = 22 * 60
    b_start = 0
    b_end = 2 * 60
    deadzone_start = 2 * 60
    deadzone_end = 4 * 60
    gap_start = 22 * 60
    gap_end = 24 * 60

    allowed_days = set(cfg["strategy_a"]["days"])

    # Build event map.
    event_map = defaultdict(list)
    for _, e in events.iterrows():
        d = pd.Timestamp(e["event_ts"]).date()
        event_map[d].append(
            {
                "event_ts": pd.Timestamp(e["event_ts"]),
                "impact": str(e.get("impact", "")).lower(),
                "event_id": str(e.get("event_id", "")),
                "source": str(e.get("source", "")),
            }
        )

    # Asian range map for Strategy B (20:00-00:00 UTC before session date).
    asian_map = {}
    for d in sorted(df["utc_date"].unique()):
        d_ts = pd.Timestamp(d, tz="UTC")
        st = d_ts - pd.Timedelta(hours=4)
        en = d_ts
        g = df[(df["time_utc"] >= st) & (df["time_utc"] < en)]
        if g.empty:
            continue
        hi = float(g["high"].max())
        lo = float(g["low"].min())
        mid = (hi + lo) / 2.0
        w = (hi - lo) / PIP_SIZE
        if w < 12:
            bucket = "tight_under_12"
        elif w < 30:
            bucket = "normal_12_30"
        elif w <= 40:
            bucket = "wide_30_40"
        else:
            bucket = "over_40"
        asian_map[d] = {"high": hi, "low": lo, "mid": mid, "width_pips": w, "bucket": bucket}

    # Verification prints.
    print("Strategy A window UTC: 16:30-22:00")
    print("Strategy B window UTC: 00:00-02:00")
    print(f"Starting equity = ${start_eq:,.0f}")
    print("First 5 Asian ranges:")
    for d in sorted(asian_map.keys())[:5]:
        a = asian_map[d]
        print(f"  {d} high={a['high']:.5f} low={a['low']:.5f} mid={a['mid']:.5f} width_pips={a['width_pips']:.1f} bucket={a['bucket']}")

    print(f"Total news events detected: {len(events)}")
    if len(events):
        print("First 5 news events:")
        for _, e in events.head(5).iterrows():
            print(f"  {pd.Timestamp(e['event_ts'])} impact={e['impact']} id={e['event_id']}")

    # State.
    trade_id = 0
    open_positions: list[Position] = []
    closed = []
    diag = Counter()
    news_stats = Counter()
    b_range_allowed_days = set()
    b_range_skipped_tight_days = set()
    b_range_skipped_wide_days = set()

    # Session-state keyed by utc_date string.
    state = {}

    pending_a = []  # strategy A confirmation signals

    def get_state(day_str: str) -> dict:
        if day_str not in state:
            state[day_str] = {
                "a_trades": 0,
                "b_trades": 0,
                "a_consec_losses": 0,
                "b_fails": 0,
                "a_stopped": False,
                "b_stopped": False,
                "a_session_start_equity": float(equity),
                "a_session_pnl": 0.0,
                "a_last_entry": None,
                "b_last_entry": None,
                "a_vwap_num": 0.0,
                "a_vwap_den": 0.0,
                "a_twap_sum": 0.0,
                "a_twap_n": 0,
                "a_breakout_rw": [],
                "a_breakout_cooldown_until": None,
                "a_last_stop_long": None,
                "a_last_stop_short": None,
                "b_attempts": 0,
                "b_pending_breakout": None,
                "b_first_break": None,
                "b_whipsaw": False,
            }
        return state[day_str]

    def current_a_mean(s: dict) -> float:
        if s["a_vwap_den"] > 0:
            return s["a_vwap_num"] / s["a_vwap_den"]
        if s["a_twap_n"] > 0:
            return s["a_twap_sum"] / s["a_twap_n"]
        return np.nan

    def open_risk_usd(existing_positions: list[Position], mark_price: float) -> float:
        total = 0.0
        for p in existing_positions:
            if p.units_remaining <= 0:
                continue
            sl_pips = abs(p.entry_price - p.sl_price) / PIP_SIZE
            total += sl_pips * p.units_remaining * (PIP_SIZE / max(1e-9, mark_price))
        return float(total)

    def in_news_block(ts: pd.Timestamp, day: pd.Timestamp.date):
        if not cfg["news_filter"].get("enabled", False):
            return False, None
        for e in event_map.get(day, []):
            impact = str(e["impact"]).lower()
            if impact == "high":
                pre = int(cfg["news_filter"]["high_impact_pre_block_minutes"])
                post = int(cfg["news_filter"]["high_impact_post_block_minutes"])
            else:
                pre = int(cfg["news_filter"]["medium_impact_pre_block_minutes"])
                post = int(cfg["news_filter"]["medium_impact_post_block_minutes"])
            if pd.Timestamp(e["event_ts"]) - pd.Timedelta(minutes=pre) <= ts <= pd.Timestamp(e["event_ts"]) + pd.Timedelta(minutes=post):
                return True, e
        return False, None

    def close_pos(pos: Position, ts: pd.Timestamp, exit_px: float, reason: str, equity_before: float):
        nonlocal equity
        if pos.units_remaining > 0:
            pips, usd = calc_leg_usd_pips(pos.direction, pos.entry_price, exit_px, pos.units_remaining)
            pos.realized_pip_units += pips * pos.units_remaining
            pos.realized_usd += usd
            pos.units_remaining = 0
        pos.exit_time = ts
        pos.exit_price = float(exit_px)
        pos.exit_reason = reason
        total_pips = pos.realized_pip_units / max(1, pos.units_initial)
        equity += pos.realized_usd

        sd = pos.session_day
        s = state.get(sd, None)
        if s is not None:
            if pos.strategy == "strat_a_meanrev":
                s["a_session_pnl"] += pos.realized_usd
                if reason in {"sl_hit", "news_tightened_stop"}:
                    s["a_consec_losses"] += 1
                    if pos.direction == "long":
                        s["a_last_stop_long"] = ts
                    else:
                        s["a_last_stop_short"] = ts
                    if s["a_consec_losses"] >= 3:
                        s["a_stopped"] = True
                else:
                    s["a_consec_losses"] = 0
                if s["a_session_pnl"] <= (-0.015 * s["a_session_start_equity"]):
                    s["a_stopped"] = True
            else:
                if reason in {"sl_hit", "news_tightened_stop", "time_exit_45min"}:
                    s["b_fails"] += 1
                    if s["b_fails"] >= 2:
                        s["b_stopped"] = True

        row = {
            "trade_id": pos.trade_id,
            "strategy": pos.strategy,
            "entry_datetime": str(pos.entry_time),
            "exit_datetime": str(ts),
            "direction": pos.direction,
            "phase": pos.phase,
            "entry_price": pos.entry_price,
            "exit_price": pos.exit_price,
            "sl_price": pos.sl_price,
            "tp1_price": pos.tp1_price,
            "tp2_price": pos.tp2_price,
            "exit_reason": reason,
            "pips": float(total_pips),
            "usd": float(pos.realized_usd),
            "position_size_units": int(pos.units_initial),
            "duration_minutes": (ts - pos.entry_time).total_seconds() / 60.0,
            "equity_before": float(equity_before),
            "equity_after": float(equity),
            "mfe_pips": float(pos.max_profit_seen_pips),
            "mae_pips": float(pos.max_adverse_seen_pips),
            "session_day": pos.session_day,
        }
        row.update(pos.entry_indicators)
        row["sl_pips"] = abs(pos.entry_price - pos.sl_price) / PIP_SIZE
        row["R"] = row["pips"] / row["sl_pips"] if row["sl_pips"] > 0 else 0.0
        closed.append(row)

    def try_open(strategy: str, direction: str, ts: pd.Timestamp, row: pd.Series, day_state: dict, entry_px: float, sl_px: float, tp1_px: float, tp2_px: float, tp1_close: float, be_offset: float, trail_activate: float, trail_dist: float, time_decay_minutes: int, time_decay_profit_cap: float, early_exit_min: int, early_exit_minprofit: float, entry_info: dict) -> bool:
        nonlocal trade_id

        # Combined risk cap 2%.
        new_sl_pips = abs(entry_px - sl_px) / PIP_SIZE
        if new_sl_pips <= 0:
            return False
        risk_pct = float(entry_info.get("risk_pct", 0.0))
        est_new_risk_usd = equity * risk_pct
        cur_risk = open_risk_usd(open_positions, float(row["close"]))
        if cur_risk + est_new_risk_usd > equity * 0.02:
            diag["blocked_combined_risk_cap"] += 1
            return False

        if strategy == "strat_a_meanrev":
            if day_state["a_stopped"]:
                diag["a_blocked_stopped"] += 1
                return False
            if day_state["a_trades"] >= int(cfg["strategy_a"]["max_trades_per_session"]):
                diag["a_blocked_max_trades"] += 1
                return False
            if day_state["a_last_entry"] is not None and (ts - day_state["a_last_entry"]).total_seconds() < int(cfg["strategy_a"]["min_time_between_entries_minutes"]) * 60:
                diag["a_blocked_min_gap"] += 1
                return False
            if len([p for p in open_positions if p.strategy == "strat_a_meanrev"]) >= int(cfg["strategy_a"]["max_concurrent_positions"]):
                diag["a_blocked_max_open"] += 1
                return False
            reentry_block_min = int(cfg["strategy_a"].get("no_reentry_same_direction_after_stop_minutes", 30))
            last_stop = day_state["a_last_stop_long"] if direction == "long" else day_state["a_last_stop_short"]
            if last_stop is not None and (ts - pd.Timestamp(last_stop)).total_seconds() < reentry_block_min * 60:
                diag["a_blocked_reentry_after_stop"] += 1
                return False
        else:
            if day_state["b_stopped"]:
                diag["b_blocked_stopped"] += 1
                return False
            if day_state["b_trades"] >= int(cfg["strategy_b"]["max_trades_per_session"]):
                diag["b_blocked_max_trades"] += 1
                return False
            if day_state["b_last_entry"] is not None and (ts - day_state["b_last_entry"]).total_seconds() < int(cfg["strategy_b"]["min_time_between_entries_minutes"]) * 60:
                diag["b_blocked_min_gap"] += 1
                return False
            if len([p for p in open_positions if p.strategy == "strat_b_breakout"]) >= int(cfg["strategy_b"]["max_concurrent_positions"]):
                diag["b_blocked_max_open"] += 1
                return False

        # Position size.
        units = math.floor((equity * risk_pct) / (new_sl_pips * (PIP_SIZE / max(1e-9, entry_px))))
        units = int(max(0, min(int(cfg["portfolio"]["max_position_units"]), units)))
        if units < 1:
            diag["blocked_units_lt_1"] += 1
            return False

        trade_id += 1
        pos = Position(
            trade_id=trade_id,
            strategy=strategy,
            direction=direction,
            entry_time=ts,
            session_day=str(ts.date()),
            phase=entry_info.get("phase", ""),
            entry_price=float(entry_px),
            sl_price=float(sl_px),
            tp1_price=float(tp1_px),
            tp2_price=float(tp2_px),
            tp1_close_pct=float(tp1_close),
            units_initial=units,
            units_remaining=units,
            be_offset_pips=float(be_offset),
            trail_activate_pips=float(trail_activate),
            trail_distance_pips=float(trail_dist),
            time_decay_minutes=int(time_decay_minutes),
            time_decay_profit_cap_pips=float(time_decay_profit_cap),
            early_exit_minutes=int(early_exit_min),
            early_exit_min_profit_pips=float(early_exit_minprofit),
            entry_indicators=entry_info,
        )
        open_positions.append(pos)
        if strategy == "strat_a_meanrev":
            day_state["a_trades"] += 1
            day_state["a_last_entry"] = ts
            diag["a_entries"] += 1
            diag["bars_with_entry_triggered"] += 1
        else:
            day_state["b_trades"] += 1
            day_state["b_last_entry"] = ts
            day_state["b_attempts"] += 1
            diag["b_entries"] += 1
        return True

    # Track first trade examples for verification.
    first_a = []
    first_b = []

    for i, row in df.iterrows():
        diag["total_bars"] += 1
        ts = pd.Timestamp(row["time_utc"])
        day = ts.date()
        day_name = str(row["utc_day_name"])
        minute = int(row["minute_of_day"])

        # Windows.
        in_a = (a_start <= minute < a_end)
        in_b = (b_start <= minute < b_end)
        in_deadzone = (deadzone_start <= minute < deadzone_end)
        in_gap = (gap_start <= minute < gap_end)

        # Manage open positions first.
        bid_h, ask_h = get_bid_ask(float(row["high"]), spread)
        bid_l, ask_l = get_bid_ask(float(row["low"]), spread)
        bid_c, ask_c = get_bid_ask(float(row["close"]), spread)

        # News block action: tighten stops on existing positions.
        news_block, active_event = in_news_block(ts, day)
        if news_block:
            for pos in open_positions:
                old = pos.sl_price
                if pos.direction == "long":
                    new_sl = max(pos.sl_price, bid_c - 5.0 * PIP_SIZE)
                else:
                    new_sl = min(pos.sl_price, ask_c + 5.0 * PIP_SIZE)
                if new_sl != old:
                    pos.sl_price = new_sl
                    pos.news_tightened = True
                    news_stats["positions_tightened"] += 1

        # Position lifecycle.
        for pos in list(open_positions):
            if pos.direction == "long":
                fav = (bid_h - pos.entry_price) / PIP_SIZE
                adv = (pos.entry_price - bid_l) / PIP_SIZE
            else:
                fav = (pos.entry_price - ask_l) / PIP_SIZE
                adv = (ask_h - pos.entry_price) / PIP_SIZE
            pos.max_profit_seen_pips = max(pos.max_profit_seen_pips, float(fav))
            pos.max_adverse_seen_pips = max(pos.max_adverse_seen_pips, float(adv))

            # trailing update
            if pos.tp1_hit and pos.max_profit_seen_pips >= pos.trail_activate_pips:
                pos.trail_active = True
            if pos.trail_active:
                if pos.direction == "long":
                    nt = bid_c - pos.trail_distance_pips * PIP_SIZE
                    pos.trail_stop_price = nt if pos.trail_stop_price is None else max(pos.trail_stop_price, nt)
                else:
                    nt = ask_c + pos.trail_distance_pips * PIP_SIZE
                    pos.trail_stop_price = nt if pos.trail_stop_price is None else min(pos.trail_stop_price, nt)

            held = (ts - pos.entry_time).total_seconds() / 60.0

            if pos.direction == "long":
                hit_sl = bid_l <= pos.sl_price
                hit_tp1 = (not pos.tp1_hit) and (bid_h >= pos.tp1_price)
                hit_tp2 = bid_h >= pos.tp2_price
                hit_trail = pos.tp1_hit and pos.trail_active and pos.trail_stop_price is not None and (bid_l <= pos.trail_stop_price)
                cur_pips = (bid_c - pos.entry_price) / PIP_SIZE
            else:
                hit_sl = ask_h >= pos.sl_price
                hit_tp1 = (not pos.tp1_hit) and (ask_l <= pos.tp1_price)
                hit_tp2 = ask_l <= pos.tp2_price
                hit_trail = pos.tp1_hit and pos.trail_active and pos.trail_stop_price is not None and (ask_h >= pos.trail_stop_price)
                cur_pips = (pos.entry_price - ask_c) / PIP_SIZE

            # B failure fast time exit.
            if pos.strategy == "strat_b_breakout" and held >= pos.early_exit_minutes and pos.max_profit_seen_pips < pos.early_exit_min_profit_pips:
                eq_before = equity
                px = bid_c if pos.direction == "long" else ask_c
                close_pos(pos, ts, px, b_time_exit_reason, eq_before)
                open_positions.remove(pos)
                continue

            # A time decay post partial.
            if pos.strategy == "strat_a_meanrev" and pos.tp1_hit and held >= pos.time_decay_minutes and cur_pips < pos.time_decay_profit_cap_pips:
                eq_before = equity
                px = bid_c if pos.direction == "long" else ask_c
                close_pos(pos, ts, px, "partial_tp_then_time_decay", eq_before)
                open_positions.remove(pos)
                continue

            if hit_sl:
                eq_before = equity
                reason = "news_tightened_stop" if pos.news_tightened else "sl_hit"
                close_pos(pos, ts, float(pos.sl_price), reason, eq_before)
                open_positions.remove(pos)
                continue

            if hit_tp1:
                cu = int(max(1, min(pos.units_remaining, math.floor(pos.units_initial * pos.tp1_close_pct))))
                leg_pips, leg_usd = calc_leg_usd_pips(pos.direction, pos.entry_price, pos.tp1_price, cu)
                pos.realized_pip_units += leg_pips * cu
                pos.realized_usd += leg_usd
                pos.units_remaining -= cu
                pos.tp1_hit = True
                # Move SL to BE for strategy A only.
                if pos.strategy == "strat_a_meanrev":
                    if pos.direction == "long":
                        pos.sl_price = pos.entry_price + pos.be_offset_pips * PIP_SIZE
                    else:
                        pos.sl_price = pos.entry_price - pos.be_offset_pips * PIP_SIZE
                    pos.moved_to_be = True
                if pos.units_remaining <= 0:
                    eq_before = equity
                    close_pos(pos, ts, float(pos.tp1_price), "tp_hit", eq_before)
                    open_positions.remove(pos)
                    continue

            if pos.units_remaining > 0 and hit_tp2 and pos.tp2_price != pos.tp1_price:
                eq_before = equity
                close_pos(pos, ts, float(pos.tp2_price), "tp_hit", eq_before)
                open_positions.remove(pos)
                continue

            if pos.units_remaining > 0 and hit_trail:
                eq_before = equity
                reason = "partial_tp_then_be" if (pos.strategy == "strat_a_meanrev" and pos.moved_to_be) else "partial_tp_then_trail"
                close_pos(pos, ts, float(pos.trail_stop_price), reason, eq_before)
                open_positions.remove(pos)
                continue

            # Hard phase/session closes.
            if pos.strategy == "strat_b_breakout" and minute >= b_end:
                eq_before = equity
                px = bid_c if pos.direction == "long" else ask_c
                reason = "partial_tp_then_session_close" if pos.tp1_hit else "session_close"
                close_pos(pos, ts, px, reason, eq_before)
                open_positions.remove(pos)
                continue

            if pos.strategy == "strat_a_meanrev" and minute >= a_end:
                eq_before = equity
                px = bid_c if pos.direction == "long" else ask_c
                reason = "partial_tp_then_session_close" if pos.tp1_hit else "session_close"
                close_pos(pos, ts, px, reason, eq_before)
                open_positions.remove(pos)
                continue

        # Skip day filter.
        if day_name not in allowed_days:
            continue

        # keep deadzone/gap clean for entries
        if in_deadzone or in_gap:
            continue

        if in_a:
            diag["bars_in_session"] += 1
            diag["bars_passed_session_filter"] += 1

        ds = get_state(str(day))

        # Update Strategy A VWAP/TWAP while in A session.
        if in_a:
            vol = float(row.get("volume", 1.0) if not pd.isna(row.get("volume", np.nan)) else 1.0)
            ds["a_vwap_num"] += float(row["close"]) * max(1.0, vol)
            ds["a_vwap_den"] += max(1.0, vol)
            ds["a_twap_sum"] += float(row["close"])
            ds["a_twap_n"] += 1

        # Process pending A confirmations.
        for sig in list(pending_a):
            if sig["day"] != str(day):
                continue
            if i <= sig["sig_idx"]:
                continue
            if i > sig["exp_idx"]:
                pending_a.remove(sig)
                diag["a_signal_expired"] += 1
                continue
            if not in_a:
                continue
            if news_block:
                news_stats["trades_blocked_strat_a"] += 1
                continue
            # confirmation candle
            ok = (float(row["close"]) > float(row["open"])) if sig["direction"] == "long" else (float(row["close"]) < float(row["open"]))
            if not ok:
                continue
            entry_px = ask_c if sig["direction"] == "long" else bid_c

            # SL from triggering pivot + 8 pips, min10 max28
            sl_buf = float(cfg["strategy_a"]["sl_buffer_pips"]) * PIP_SIZE
            if sig["direction"] == "long":
                trigger = float(sig["trigger_pivot"])
                sl_raw = trigger - sl_buf
                sl_pips = (entry_px - sl_raw) / PIP_SIZE
            else:
                trigger = float(sig["trigger_pivot"])
                sl_raw = trigger + sl_buf
                sl_pips = (sl_raw - entry_px) / PIP_SIZE
            sl_pips = max(float(cfg["strategy_a"]["sl_min_pips"]), min(float(cfg["strategy_a"]["sl_max_pips"]), sl_pips))
            sl_px = entry_px - sl_pips * PIP_SIZE if sig["direction"] == "long" else entry_px + sl_pips * PIP_SIZE

            # TP target: closer of pivot P and session mean, with 6..15 pips clamp
            mean_px = current_a_mean(ds)
            P = float(sig["P"])
            if np.isfinite(mean_px):
                t_raw = mean_px if abs(mean_px - entry_px) <= abs(P - entry_px) else P
            else:
                t_raw = P
            if sig["direction"] == "long":
                dist = (t_raw - entry_px) / PIP_SIZE
                dist = max(float(cfg["strategy_a"]["tp_min_pips"]), min(float(cfg["strategy_a"]["tp_max_pips"]), dist))
                tp1_px = entry_px + dist * PIP_SIZE
                tp2_px = entry_px + 999.0 * PIP_SIZE
            else:
                dist = (entry_px - t_raw) / PIP_SIZE
                dist = max(float(cfg["strategy_a"]["tp_min_pips"]), min(float(cfg["strategy_a"]["tp_max_pips"]), dist))
                tp1_px = entry_px - dist * PIP_SIZE
                tp2_px = entry_px - 999.0 * PIP_SIZE

            opened = try_open(
                strategy="strat_a_meanrev",
                direction=sig["direction"],
                ts=ts,
                row=row,
                day_state=ds,
                entry_px=entry_px,
                sl_px=sl_px,
                tp1_px=tp1_px,
                tp2_px=tp2_px,
                tp1_close=float(cfg["strategy_a"]["partial_close_pct"]),
                be_offset=float(cfg["strategy_a"]["be_offset_pips"]),
                trail_activate=float(cfg["strategy_a"]["trail_activate_pips"]),
                trail_dist=float(cfg["strategy_a"]["trail_distance_pips"]),
                time_decay_minutes=int(cfg["strategy_a"]["time_decay_minutes"]),
                time_decay_profit_cap=float(cfg["strategy_a"]["time_decay_profit_cap_pips"]),
                early_exit_min=9999,
                early_exit_minprofit=0.0,
                entry_info={
                    "strategy_tag": "strat_a_meanrev",
                    "risk_pct": float(cfg["strategy_a"]["risk_pct"]) / 100.0,
                    "phase": "strat_a_window",
                    "pivot_P": P,
                    "pivot_R1": sig["R1"],
                    "pivot_S1": sig["S1"],
                    "bb_upper": sig["bb_upper"],
                    "bb_lower": sig["bb_lower"],
                    "rsi_m5": sig["rsi"],
                    "adx_m15": sig["adx"],
                    "combo": sig["combo"],
                },
            )
            if opened and len(first_a) < 3:
                first_a.append({"entry": str(ts), "dir": sig["direction"], "entry_px": entry_px, "sl": sl_px, "tp1": tp1_px, "combo": sig["combo"]})
            pending_a.remove(sig)

        # Strategy A signal generation.
        if in_a:
            if news_block:
                news_stats["trades_blocked_strat_a"] += 1
            else:
                # Basic indicator availability.
                has_ind = (
                    np.isfinite(float(row.get("pivot_P", np.nan)))
                    and np.isfinite(float(row.get("bb_upper", np.nan)))
                    and np.isfinite(float(row.get("bb_lower", np.nan)))
                    and np.isfinite(float(row.get("rsi_m5", np.nan)))
                    and np.isfinite(float(row.get("adx_m15", np.nan)))
                    and np.isfinite(float(row.get("atr_m15", np.nan)))
                    and (str(row.get("bb_regime", "")) in {"ranging", "trending"})
                )
                if not has_ind:
                    diag["a_blocked_missing_indicators"] += 1
                else:
                    diag["bars_passed_indicator_check"] += 1
                    logic_mode = str(cfg["strategy_a"].get("signal_logic_mode", "legacy")).strip().lower()
                    if logic_mode == "strict_v7":
                        # V7-like rolling breakout cooldown gate.
                        rw = ds["a_breakout_rw"]
                        rw.append((ts, float(row["high"]), float(row["low"])))
                        roll_min = int(cfg["strategy_a"].get("breakout_rolling_window_minutes", 60))
                        cutoff = ts - pd.Timedelta(minutes=roll_min)
                        while rw and pd.Timestamp(rw[0][0]) < cutoff:
                            rw.pop(0)
                        cd_until = ds.get("a_breakout_cooldown_until")
                        if cd_until is not None and ts < pd.Timestamp(cd_until):
                            diag["a_blocked_breakout_cooldown"] += 1
                            continue
                        highs = [x[1] for x in rw]
                        lows = [x[2] for x in rw]
                        roll_rng = (max(highs) - min(lows)) / PIP_SIZE if highs and lows else 0.0
                        roll_thresh = float(cfg["strategy_a"].get("breakout_rolling_range_threshold_pips", 40.0))
                        if roll_rng > roll_thresh:
                            cd_min = int(cfg["strategy_a"].get("breakout_cooldown_minutes", 15))
                            ds["a_breakout_cooldown_until"] = ts + pd.Timedelta(minutes=cd_min)
                            diag["a_blocked_breakout_rolling"] += 1
                            continue
                        ds["a_breakout_cooldown_until"] = None
                        diag["bars_passed_breakout_check"] += 1

                        # Regime + ATR + ADX filters.
                        if str(row.get("bb_regime", "")) != "ranging":
                            diag["a_blocked_regime_not_ranging"] += 1
                            continue
                        diag["bars_passed_regime_filter"] += 1
                        atr_max = float(cfg["strategy_a"].get("atr_max_price_units", 0.3))
                        if float(row["atr_m15"]) > atr_max:
                            diag["a_blocked_atr"] += 1
                            continue
                        if float(row["adx_m15"]) >= float(cfg["strategy_a"]["adx_max"]):
                            diag["a_blocked_adx"] += 1
                            continue
                        diag["bars_passed_adx_filter"] += 1
                        diag["bars_reached_signal_scoring"] += 1

                        P = float(row["pivot_P"])
                        R1 = float(row["pivot_R1"])
                        S1 = float(row["pivot_S1"])
                        S2 = float(row["pivot_S2"])
                        R2 = float(row["pivot_R2"])
                        tol = float(cfg["strategy_a"]["pivot_tolerance_pips"]) * PIP_SIZE

                        zone_l = float(row["close"]) <= (S1 + tol)
                        bb_l = (float(row["close"]) <= float(row["bb_lower"])) or (float(row["low"]) <= float(row["bb_lower"]))
                        sar_l = bool(row.get("sar_flip_bull_recent", False))
                        rsi_l_soft = float(row["rsi_m5"]) < 35.0
                        if zone_l:
                            diag["bars_with_long_signal_zone"] += 1
                        if bb_l:
                            diag["bars_with_bb_touch"] += 1
                        if sar_l:
                            diag["bars_with_sar_flip"] += 1
                        long_ok = False
                        combo_l = ""
                        if zone_l and bb_l and sar_l and rsi_l_soft:
                            score_l = int(zone_l) + int(bb_l) + int(sar_l) + int(float(row["rsi_m5"]) < 30.0) + int(float(row["close"]) <= (S2 + tol))
                            long_ok = score_l >= 2
                            combo_l = "".join(
                                x
                                for x, ok in [
                                    ("A", zone_l),
                                    ("B", bb_l),
                                    ("C", sar_l),
                                    ("D", float(row["rsi_m5"]) < 30.0),
                                    ("E", float(row["close"]) <= (S2 + tol)),
                                ]
                                if ok
                            )

                        zone_s = float(row["close"]) >= (R1 - tol)
                        bb_s = (float(row["close"]) >= float(row["bb_upper"])) or (float(row["high"]) >= float(row["bb_upper"]))
                        sar_s = bool(row.get("sar_flip_bear_recent", False))
                        rsi_s_soft = float(row["rsi_m5"]) > 65.0
                        if zone_s:
                            diag["bars_with_short_signal_zone"] += 1
                        if bb_s:
                            diag["bars_with_bb_touch"] += 1
                        if sar_s:
                            diag["bars_with_sar_flip"] += 1
                        short_ok = False
                        combo_s = ""
                        if zone_s and bb_s and sar_s and rsi_s_soft:
                            score_s = int(zone_s) + int(bb_s) + int(sar_s) + int(float(row["rsi_m5"]) > 70.0) + int(float(row["close"]) >= (R2 - tol))
                            short_ok = score_s >= 2
                            combo_s = "".join(
                                x
                                for x, ok in [
                                    ("A", zone_s),
                                    ("B", bb_s),
                                    ("C", sar_s),
                                    ("D", float(row["rsi_m5"]) > 70.0),
                                    ("E", float(row["close"]) >= (R2 - tol)),
                                ]
                                if ok
                            )

                        if long_ok or short_ok:
                            diag["bars_with_confluence_met"] += 1
                        if long_ok and short_ok:
                            diag["a_blocked_ambiguous"] += 1
                            continue
                    else:
                        # Legacy V12-A behavior (for diagnostic A/B comparison).
                        if float(row["adx_m15"]) >= float(cfg["strategy_a"]["adx_max"]):
                            diag["a_blocked_adx"] += 1
                            continue
                        diag["bars_passed_adx_filter"] += 1
                        diag["bars_passed_regime_filter"] += 1
                        diag["bars_reached_signal_scoring"] += 1
                        P = float(row["pivot_P"])
                        R1 = float(row["pivot_R1"])
                        S1 = float(row["pivot_S1"])
                        S2 = float(row["pivot_S2"])
                        R2 = float(row["pivot_R2"])
                        tol = float(cfg["strategy_a"]["pivot_tolerance_pips"]) * PIP_SIZE
                        zone_l = float(row["close"]) <= (S1 + tol)
                        bb_l = (float(row["close"]) <= float(row["bb_lower"])) or (float(row["low"]) <= float(row["bb_lower"]))
                        rsi_l = float(row["rsi_m5"]) < float(cfg["strategy_a"]["rsi_long_max"])
                        sar_l = bool(row.get("sar_flip_bull_recent", False))
                        if zone_l:
                            diag["bars_with_long_signal_zone"] += 1
                        if bb_l:
                            diag["bars_with_bb_touch"] += 1
                        if sar_l:
                            diag["bars_with_sar_flip"] += 1
                        score_l = int(zone_l) + int(bb_l) + int(rsi_l)
                        combo_l = "".join(x for x, ok in [("A", zone_l), ("B", bb_l), ("C", rsi_l), ("D", sar_l)] if ok)
                        zone_s = float(row["close"]) >= (R1 - tol)
                        bb_s = (float(row["close"]) >= float(row["bb_upper"])) or (float(row["high"]) >= float(row["bb_upper"]))
                        rsi_s = float(row["rsi_m5"]) > float(cfg["strategy_a"]["rsi_short_min"])
                        sar_s = bool(row.get("sar_flip_bear_recent", False))
                        if zone_s:
                            diag["bars_with_short_signal_zone"] += 1
                        if bb_s:
                            diag["bars_with_bb_touch"] += 1
                        if sar_s:
                            diag["bars_with_sar_flip"] += 1
                        score_s = int(zone_s) + int(bb_s) + int(rsi_s)
                        combo_s = "".join(x for x, ok in [("A", zone_s), ("B", bb_s), ("C", rsi_s), ("D", sar_s)] if ok)
                        long_ok = (score_l >= 2) and (combo_l != "ABCD")
                        short_ok = (score_s >= 2) and (combo_s != "ABCD")
                        if long_ok or short_ok:
                            diag["bars_with_confluence_met"] += 1
                        if long_ok and short_ok:
                            diag["a_blocked_ambiguous"] += 1
                            continue

                    # no ambiguous both
                    if long_ok ^ short_ok:
                        direction = "long" if long_ok else "short"
                        existing = any(p["day"] == str(day) and p["direction"] == direction and i <= p["exp_idx"] for p in pending_a)
                        if not existing:
                            pending_a.append(
                                {
                                    "day": str(day),
                                    "direction": direction,
                                    "sig_idx": i,
                                    "exp_idx": i + int(cfg["strategy_a"]["confirm_bars"]),
                                    "P": P,
                                    "R1": R1,
                                    "S1": S1,
                                    "trigger_pivot": S1 if direction == "long" else R1,
                                    "bb_upper": float(row["bb_upper"]),
                                    "bb_lower": float(row["bb_lower"]),
                                    "rsi": float(row["rsi_m5"]),
                                    "adx": float(row["adx_m15"]),
                                    "combo": combo_l if direction == "long" else combo_s,
                                }
                            )
                            diag["a_signal_generated"] += 1

        # Strategy B entries on M5 only.
        if in_b and bool(row["is_m5_close"]):
            if news_block:
                news_stats["trades_blocked_strat_b"] += 1
            else:
                # no entry first 15m
                if minute >= 15:
                    ainfo = asian_map.get(day)
                    if ainfo is None:
                        diag["b_blocked_missing_asian"] += 1
                    elif ainfo["width_pips"] < float(cfg["strategy_b"].get("asian_range_min_pips", 25.0)):
                        diag["b_blocked_asian_too_tight"] += 1
                        b_range_skipped_tight_days.add(str(day))
                    elif ainfo["width_pips"] > float(cfg["strategy_b"].get("asian_range_max_pips", 45.0)):
                        diag["b_blocked_asian_too_wide"] += 1
                        b_range_skipped_wide_days.add(str(day))
                    elif ds["b_stopped"]:
                        diag["b_blocked_stopped"] += 1
                    else:
                        b_range_allowed_days.add(str(day))
                        # pending confirmation handling
                        pb = ds.get("b_pending_breakout")
                        if pb is not None:
                            lvl = float(pb["level"])
                            ddir = pb["direction"]
                            confirmed = (float(row["close"]) > lvl) if ddir == "long" else (float(row["close"]) < lvl)
                            if confirmed:
                                entry_px = ask_c if ddir == "long" else bid_c
                                # Structural SL at 38.2% retracement of Asian range.
                                r_hi = float(ainfo["high"])
                                r_lo = float(ainfo["low"])
                                r_w = r_hi - r_lo
                                if ddir == "long":
                                    sl_raw = r_hi - 0.382 * r_w
                                else:
                                    sl_raw = r_lo + 0.382 * r_w
                                if ddir == "long":
                                    sl_pips = (entry_px - sl_raw) / PIP_SIZE
                                else:
                                    sl_pips = (sl_raw - entry_px) / PIP_SIZE
                                if sl_pips > float(cfg["strategy_b"]["sl_max_pips"]):
                                    ds["b_pending_breakout"] = None
                                    diag["b_blocked_sl_too_wide"] += 1
                                else:
                                    sl_pips = max(float(cfg["strategy_b"]["sl_min_pips"]), sl_pips)
                                    sl_px = entry_px - sl_pips * PIP_SIZE if ddir == "long" else entry_px + sl_pips * PIP_SIZE
                                    atr_pips = float(row.get("atr_m15", np.nan)) / PIP_SIZE if np.isfinite(float(row.get("atr_m15", np.nan))) else 0.0
                                    tp_pips = max(float(cfg["strategy_b"]["tp_min_pips"]), min(float(cfg["strategy_b"]["tp_max_pips"]), float(cfg["strategy_b"]["tp_atr_mult"]) * atr_pips))
                                    tp1_px = entry_px + tp_pips * PIP_SIZE if ddir == "long" else entry_px - tp_pips * PIP_SIZE
                                    tp2_px = entry_px + 999.0 * PIP_SIZE if ddir == "long" else entry_px - 999.0 * PIP_SIZE

                                    # risk by asian range width adjustment.
                                    if ainfo["width_pips"] < 30:
                                        rpct = 0.5
                                    else:
                                        rpct = 0.35
                                    opened = try_open(
                                        strategy="strat_b_breakout",
                                        direction=ddir,
                                        ts=ts,
                                        row=row,
                                        day_state=ds,
                                        entry_px=entry_px,
                                        sl_px=sl_px,
                                        tp1_px=tp1_px,
                                        tp2_px=tp2_px,
                                        tp1_close=float(cfg["strategy_b"]["partial_close_pct"]),
                                        be_offset=0.0,
                                        trail_activate=0.0,
                                        trail_dist=float(cfg["strategy_b"]["trail_distance_pips"]),
                                        time_decay_minutes=9999,
                                        time_decay_profit_cap=9999.0,
                                        early_exit_min=int(cfg["strategy_b"]["time_exit_minutes"]),
                                        early_exit_minprofit=float(cfg["strategy_b"]["time_exit_min_profit_pips"]),
                                        entry_info={
                                            "strategy_tag": "strat_b_breakout",
                                            "risk_pct": rpct / 100.0,
                                            "phase": "strat_b_window",
                                            "asian_high": float(ainfo["high"]),
                                            "asian_low": float(ainfo["low"]),
                                            "asian_mid": float(ainfo["mid"]),
                                            "asian_width_pips": float(ainfo["width_pips"]),
                                            "asian_bucket": str(ainfo["bucket"]),
                                            "adx_m15": float(row.get("adx_m15", np.nan)) if np.isfinite(float(row.get("adx_m15", np.nan))) else None,
                                            "ema20_m5": float(row.get("ema20_m5", np.nan)) if np.isfinite(float(row.get("ema20_m5", np.nan))) else None,
                                            "roc10_m5": float(row.get("roc10_m5", np.nan)) if np.isfinite(float(row.get("roc10_m5", np.nan))) else None,
                                        },
                                    )
                                    if opened and len(first_b) < 3:
                                        first_b.append({"entry": str(ts), "dir": ddir, "entry_px": entry_px, "sl": sl_px, "tp1": tp1_px, "asian_width_pips": ainfo["width_pips"]})
                                    elif not opened:
                                        diag["b_open_failed"] += 1
                            else:
                                diag["b_fakeouts_avoided"] += 1
                            ds["b_pending_breakout"] = None

                        # detect new breakout if no pending
                        if ds.get("b_pending_breakout") is None and ds["b_trades"] < int(cfg["strategy_b"]["max_trades_per_session"]):
                            hi = float(ainfo["high"])
                            lo = float(ainfo["low"])
                            buf = float(cfg["strategy_b"]["break_buffer_pips"]) * PIP_SIZE
                            long_break = float(row["close"]) > (hi + buf)
                            short_break = float(row["close"]) < (lo - buf)
                            if long_break or short_break:
                                side = "up" if long_break else "down"
                                # whipsaw rule
                                fb = ds.get("b_first_break")
                                if fb is None:
                                    ds["b_first_break"] = {"side": side, "time": ts}
                                else:
                                    opposite = fb["side"] != side
                                    within30 = (ts - fb["time"]).total_seconds() <= 30 * 60
                                    if opposite and within30:
                                        ds["b_whipsaw"] = True
                                        diag["b_whipsaw_detected"] += 1
                                        continue

                                adx = float(row.get("adx_m15", np.nan))
                                ema_now = float(row.get("ema20_m5", np.nan))
                                roc = float(row.get("roc10_m5", np.nan))
                                # EMA trend slope from 6 M5 bars ago
                                prev_ema = float(df.iloc[max(0, i - 30)].get("ema20_m5", np.nan))
                                cond_common = np.isfinite(adx) and (adx > float(cfg["strategy_b"]["min_adx"])) and (adx <= float(cfg["strategy_b"]["max_adx"]))
                                if long_break:
                                    cond = cond_common and np.isfinite(ema_now) and np.isfinite(prev_ema) and (ema_now > prev_ema) and np.isfinite(roc) and roc > 0
                                    if cond:
                                        ds["b_pending_breakout"] = {"direction": "long", "level": hi + buf}
                                        diag["b_breakout_signal"] += 1
                                if short_break:
                                    cond = cond_common and np.isfinite(ema_now) and np.isfinite(prev_ema) and (ema_now < prev_ema) and np.isfinite(roc) and roc < 0
                                    if cond:
                                        ds["b_pending_breakout"] = {"direction": "short", "level": lo - buf}
                                        diag["b_breakout_signal"] += 1

    # End close leftovers.
    if open_positions:
        last = df.iloc[-1]
        ts = pd.Timestamp(last["time_utc"])
        bid_c, ask_c = get_bid_ask(float(last["close"]), spread)
        for p in list(open_positions):
            eq_before = equity
            px = bid_c if p.direction == "long" else ask_c
            reason = "partial_tp_then_session_close" if p.tp1_hit else "session_close"
            close_pos(p, ts, px, reason, eq_before)
            open_positions.remove(p)

    tdf = pd.DataFrame(closed)
    if tdf.empty:
        tdf = pd.DataFrame(columns=["trade_id", "strategy", "entry_datetime", "exit_datetime", "direction", "phase", "entry_price", "exit_price", "sl_price", "tp1_price", "tp2_price", "exit_reason", "pips", "usd", "position_size_units", "duration_minutes", "equity_before", "equity_after", "mfe_pips", "mae_pips", "session_day", "sl_pips", "R"])

    # Verification prints.
    dead_trades = len(tdf[(pd.to_datetime(tdf["entry_datetime"], utc=True).dt.hour >= 2) & (pd.to_datetime(tdf["entry_datetime"], utc=True).dt.hour < 4)]) if not tdf.empty else 0
    gap_trades = len(tdf[(pd.to_datetime(tdf["entry_datetime"], utc=True).dt.hour >= 22) & (pd.to_datetime(tdf["entry_datetime"], utc=True).dt.hour < 24)]) if not tdf.empty else 0
    mon_a = len(tdf[(tdf["strategy"] == "strat_a_meanrev") & (pd.to_datetime(tdf["entry_datetime"], utc=True).dt.day_name() == "Monday")]) if not tdf.empty else 0
    mon_b = len(tdf[(tdf["strategy"] == "strat_b_breakout") & (pd.to_datetime(tdf["entry_datetime"], utc=True).dt.day_name() == "Monday")]) if not tdf.empty else 0
    print(f"Deadzone trades (02:00-04:00): {dead_trades}; gap trades (22:00-00:00): {gap_trades}")
    print(f"Monday trades excluded check: strat_a={mon_a}, strat_b={mon_b}")
    print("First 3 Strategy A trades:")
    for x in first_a[:3]:
        print(f"  {x}")
    print("First 3 Strategy B trades:")
    for x in first_b[:3]:
        print(f"  {x}")

    # Strategy summaries.
    a_df = tdf[tdf["strategy"] == "strat_a_meanrev"].copy()
    b_df = tdf[tdf["strategy"] == "strat_b_breakout"].copy()

    # Build combined equity curve from chronological exits.
    if not tdf.empty:
        eq_sorted = tdf.sort_values("exit_datetime").copy()
        eq_curve = eq_sorted[["exit_datetime", "equity_after"]].copy()
        eq_curve = eq_curve.rename(columns={"exit_datetime": "time", "equity_after": "equity"})
        peak = eq_curve["equity"].cummax()
        eq_curve["drawdown_usd"] = (peak - eq_curve["equity"]).clip(lower=0.0)
        eq_curve["drawdown_pct"] = np.where(peak > 0, eq_curve["drawdown_usd"] / peak * 100.0, 0.0)
        eq_curve["trade_number"] = np.arange(1, len(eq_curve) + 1)
    else:
        eq_curve = pd.DataFrame(columns=["time", "equity", "drawdown_usd", "drawdown_pct", "trade_number"])

    strat_a_summary = summarize_strategy(a_df, start_eq)
    strat_b_summary = summarize_strategy(b_df, start_eq)
    combined_summary = summarize_strategy(tdf, start_eq)

    # Correlation of daily returns A vs B.
    if not tdf.empty:
        tdf["entry_ts"] = pd.to_datetime(tdf["entry_datetime"], utc=True)
        dr = (
            tdf.groupby([tdf["entry_ts"].dt.date, "strategy"], as_index=False)["usd"].sum()
            .pivot(index="entry_ts", columns="strategy", values="usd")
            .fillna(0.0)
            .rename_axis("date")
            .reset_index()
        )
        if {"strat_a_meanrev", "strat_b_breakout"}.issubset(dr.columns) and len(dr) > 2:
            corr = float(dr["strat_a_meanrev"].corr(dr["strat_b_breakout"]))
        else:
            corr = 0.0
    else:
        corr = 0.0

    div_benefit = "YES" if (combined_summary["max_dd"] < (strat_a_summary["max_dd"] + strat_b_summary["max_dd"])) else "NO"

    # Exit breakdown per strategy.
    def exit_break(g: pd.DataFrame) -> dict:
        keys = [
            "tp_hit",
            "partial_tp_then_trail",
            "partial_tp_then_be",
            "partial_tp_then_session_close",
            "partial_tp_then_time_decay",
            "sl_hit",
            "time_exit_45min",
            "time_exit_30min",
            "session_close",
            "news_tightened_stop",
        ]
        out = {}
        for k in keys:
            gg = g[g["exit_reason"] == k]
            out[k] = {"count": int(len(gg)), "net": float(gg["usd"].sum()) if len(gg) else 0.0}
        return out

    exit_by_strategy = {
        "strat_a_meanrev": exit_break(a_df),
        "strat_b_breakout": exit_break(b_df),
    }

    # Day analysis Tue-Fri per strategy+combined.
    day_rows = []
    for d in ["Tuesday", "Wednesday", "Thursday", "Friday"]:
        ga = a_df[pd.to_datetime(a_df["entry_datetime"], utc=True).dt.day_name() == d] if len(a_df) else a_df
        gb = b_df[pd.to_datetime(b_df["entry_datetime"], utc=True).dt.day_name() == d] if len(b_df) else b_df
        gc = tdf[pd.to_datetime(tdf["entry_datetime"], utc=True).dt.day_name() == d] if len(tdf) else tdf
        day_rows.append(
            {
                "day": d,
                "strat_a": {"trades": int(len(ga)), "wr": float((ga["usd"] > 0).mean() * 100.0) if len(ga) else 0.0, "pf": pf_from_usd(ga["usd"]) if len(ga) else 0.0, "net": float(ga["usd"].sum()) if len(ga) else 0.0},
                "strat_b": {"trades": int(len(gb)), "wr": float((gb["usd"] > 0).mean() * 100.0) if len(gb) else 0.0, "pf": pf_from_usd(gb["usd"]) if len(gb) else 0.0, "net": float(gb["usd"].sum()) if len(gb) else 0.0},
                "combined": {"trades": int(len(gc)), "wr": float((gc["usd"] > 0).mean() * 100.0) if len(gc) else 0.0, "pf": pf_from_usd(gc["usd"]) if len(gc) else 0.0, "net": float(gc["usd"].sum()) if len(gc) else 0.0},
            }
        )

    # Month analysis.
    month_rows = []
    if len(tdf):
        tdf["month"] = pd.to_datetime(tdf["entry_datetime"], utc=True).dt.to_period("M").astype(str)
    for m in sorted(tdf["month"].unique()) if len(tdf) else []:
        ga = a_df[pd.to_datetime(a_df["entry_datetime"], utc=True).dt.to_period("M").astype(str) == m] if len(a_df) else a_df
        gb = b_df[pd.to_datetime(b_df["entry_datetime"], utc=True).dt.to_period("M").astype(str) == m] if len(b_df) else b_df
        gc = tdf[tdf["month"] == m] if len(tdf) else tdf
        # context metrics
        m_dt = pd.Period(m)
        dd = df[pd.to_datetime(df["time_utc"]).dt.to_period("M") == m_dt]
        avg_daily_range = float(dd.groupby(dd["time_utc"].dt.date).apply(lambda g: (g["high"].max() - g["low"].min()) / PIP_SIZE).mean()) if len(dd) else 0.0
        ar = [v["width_pips"] for d, v in asian_map.items() if pd.Period(pd.Timestamp(d), freq='D').to_timestamp().to_period('M') == m_dt]
        avg_asian = float(np.mean(ar)) if ar else 0.0
        ne = sum(1 for _, e in events.iterrows() if pd.Timestamp(e["event_ts"]).to_period("M") == m_dt)
        month_rows.append(
            {
                "month": m,
                "strat_a": {"trades": int(len(ga)), "wr": float((ga["usd"] > 0).mean() * 100.0) if len(ga) else 0.0, "pf": pf_from_usd(ga["usd"]) if len(ga) else 0.0, "net": float(ga["usd"].sum()) if len(ga) else 0.0},
                "strat_b": {"trades": int(len(gb)), "wr": float((gb["usd"] > 0).mean() * 100.0) if len(gb) else 0.0, "pf": pf_from_usd(gb["usd"]) if len(gb) else 0.0, "net": float(gb["usd"].sum()) if len(gb) else 0.0},
                "combined": {"trades": int(len(gc)), "wr": float((gc["usd"] > 0).mean() * 100.0) if len(gc) else 0.0, "pf": pf_from_usd(gc["usd"]) if len(gc) else 0.0, "net": float(gc["usd"].sum()) if len(gc) else 0.0},
                "avg_asian_range_pips": avg_asian,
                "avg_daily_range_pips": avg_daily_range,
                "news_events": int(ne),
            }
        )

    # Hour analysis.
    hour_rows = []
    for h in [0, 1, 16, 17, 18, 19, 20, 21]:
        ga = a_df[pd.to_datetime(a_df["entry_datetime"], utc=True).dt.hour == h] if len(a_df) else a_df
        gb = b_df[pd.to_datetime(b_df["entry_datetime"], utc=True).dt.hour == h] if len(b_df) else b_df
        gc = tdf[pd.to_datetime(tdf["entry_datetime"], utc=True).dt.hour == h] if len(tdf) else tdf
        hour_rows.append(
            {
                "utc_hour": h,
                "strat_a": {"trades": int(len(ga)), "wr": float((ga["usd"] > 0).mean() * 100.0) if len(ga) else 0.0, "pf": pf_from_usd(ga["usd"]) if len(ga) else 0.0, "net": float(ga["usd"].sum()) if len(ga) else 0.0},
                "strat_b": {"trades": int(len(gb)), "wr": float((gb["usd"] > 0).mean() * 100.0) if len(gb) else 0.0, "pf": pf_from_usd(gb["usd"]) if len(gb) else 0.0, "net": float(gb["usd"].sum()) if len(gb) else 0.0},
                "combined": {"trades": int(len(gc)), "wr": float((gc["usd"] > 0).mean() * 100.0) if len(gc) else 0.0, "pf": pf_from_usd(gc["usd"]) if len(gc) else 0.0, "net": float(gc["usd"].sum()) if len(gc) else 0.0},
            }
        )

    # Asian range analysis for strategy B.
    b_ranges = []
    for d, a in asian_map.items():
        b_ranges.append({"day": str(d), **a})
    ar_df = pd.DataFrame(b_ranges)
    if len(b_df):
        b_df2 = b_df.copy()
        b_df2["day"] = pd.to_datetime(b_df2["entry_datetime"], utc=True).dt.date.astype(str)
        b_df2 = b_df2.merge(ar_df, on="day", how="left")
    else:
        b_df2 = pd.DataFrame(columns=["usd", "bucket"])

    perf_bucket = {}
    for k in ["tight_under_12", "normal_12_30", "wide_30_40"]:
        g = b_df2[b_df2["bucket"] == k] if len(b_df2) else b_df2
        perf_bucket[k] = {"trades": int(len(g)), "wr": float((g["usd"] > 0).mean() * 100.0) if len(g) else 0.0, "pf": pf_from_usd(g["usd"]) if len(g) else 0.0, "net": float(g["usd"].sum()) if len(g) else 0.0}

    asian_stats = {
        "avg_width_pips": float(ar_df["width_pips"].mean()) if len(ar_df) else 0.0,
        "median_width_pips": float(ar_df["width_pips"].median()) if len(ar_df) else 0.0,
        "min_width": float(ar_df["width_pips"].min()) if len(ar_df) else 0.0,
        "max_width": float(ar_df["width_pips"].max()) if len(ar_df) else 0.0,
        "pct_under_12_pips": float((ar_df["width_pips"] < 12).mean() * 100.0) if len(ar_df) else 0.0,
        "pct_12_to_30_pips": float(((ar_df["width_pips"] >= 12) & (ar_df["width_pips"] < 30)).mean() * 100.0) if len(ar_df) else 0.0,
        "pct_30_to_40_pips": float(((ar_df["width_pips"] >= 30) & (ar_df["width_pips"] <= 40)).mean() * 100.0) if len(ar_df) else 0.0,
        "pct_over_40_pips": float((ar_df["width_pips"] > 40).mean() * 100.0) if len(ar_df) else 0.0,
        "performance_by_range_bucket": perf_bucket,
    }

    fakeout_stats = {
        "total_breakout_signals": int(diag["b_breakout_signal"]),
        "confirmed_breakouts": int(diag["b_entries"]),
        "fakeouts_avoided": int(diag["b_fakeouts_avoided"]),
        "whipsaws_detected": int(diag["b_whipsaw_detected"]),
        "sessions_stopped_after_2_fails": int(sum(1 for v in state.values() if v.get("b_stopped", False))),
        "time_exit_triggers": int((b_df["exit_reason"] == b_time_exit_reason).sum()) if len(b_df) else 0,
        "time_exit_reason": b_time_exit_reason,
        "avg_breakout_speed_pips_per_5min": {
            "winners": float((b_df[b_df["usd"] > 0]["pips"] / (b_df[b_df["usd"] > 0]["duration_minutes"].replace(0, np.nan) / 5.0)).mean()) if len(b_df[b_df["usd"] > 0]) else 0.0,
            "losers": float((b_df[b_df["usd"] <= 0]["pips"] / (b_df[b_df["usd"] <= 0]["duration_minutes"].replace(0, np.nan) / 5.0)).mean()) if len(b_df[b_df["usd"] <= 0]) else 0.0,
        },
    }

    news_out = {
        "news_events_total": int(len(events)),
        "trades_blocked_strat_a": int(news_stats["trades_blocked_strat_a"]),
        "trades_blocked_strat_b": int(news_stats["trades_blocked_strat_b"]),
        "positions_tightened": int(news_stats["positions_tightened"]),
        "estimated_pnl_impact": 0.0,
    }
    b_total_days = max(1, len({str(x) for x in df["utc_date"].unique()}))
    b_range_filter_impact = {
        "sessions_with_25_45_range": int(len(b_range_allowed_days)),
        "sessions_skipped_too_tight": int(len(b_range_skipped_tight_days)),
        "sessions_skipped_too_wide": int(len(b_range_skipped_wide_days)),
        "pct_sessions_tradeable": float(100.0 * len(b_range_allowed_days) / b_total_days),
    }

    # If either strategy <30 trades on 250k, print funnel.
    if str(run_cfg["label"]).lower() == "250k":
        if len(a_df) < 30:
            print("Strategy A funnel (250k, <30 trades):")
            for k in ["a_signal_generated", "a_entries", "a_blocked_adx", "a_blocked_stopped", "a_blocked_max_trades", "a_blocked_min_gap"]:
                print(f"  {k}: {diag.get(k,0)}")
        if len(b_df) < 30:
            print("Strategy B funnel (250k, <30 trades):")
            for k in ["b_breakout_signal", "b_entries", "b_fakeouts_avoided", "b_whipsaw_detected", "b_blocked_asian_too_wide", "b_blocked_stopped"]:
                print(f"  {k}: {diag.get(k,0)}")

    report = {
        "strategy_id": cfg.get("strategy_id", "tokyo_dual_strategy_v12"),
        "run_label": run_cfg["label"],
        "input_csv": run_cfg["input_csv"],
        "summary": combined_summary,
        "strategy_summaries": {
            "strat_a_meanrev": strat_a_summary,
            "strat_b_breakout": strat_b_summary,
        },
        "combined_portfolio": {
            "combined_trades": int(len(tdf)),
            "combined_net": float(combined_summary["net_pnl"]),
            "combined_pf": float(combined_summary["pf"]),
            "combined_max_dd": float(combined_summary["max_dd"]),
            "combined_sharpe": float(combined_summary["sharpe_monthly"]),
            "correlation_of_daily_returns": float(corr if np.isfinite(corr) else 0.0),
            "diversification_benefit": div_benefit,
        },
        "exit_type_breakdown": exit_by_strategy,
        "day_analysis": day_rows,
        "month_analysis": month_rows,
        "hour_analysis": hour_rows,
        "asian_range_analysis": {
            "asian_range_stats": asian_stats,
            "fakeout_stats": fakeout_stats,
        },
        "strategy_b_range_filter_impact": b_range_filter_impact,
        "news_filter_stats": news_out,
        "diagnostics": {"counts": dict(diag)},
        "equity_curve": eq_curve.to_dict(orient="records"),
        "drawdown_curve": eq_curve[["trade_number", "drawdown_usd", "drawdown_pct"]].to_dict(orient="records") if len(eq_curve) else [],
        "trades": tdf.drop(columns=[c for c in ["entry_ts", "month"] if c in tdf.columns], errors="ignore").to_dict(orient="records"),
    }
    return report


def write_outputs(report: dict, run_cfg: dict):
    Path(run_cfg["output_json"]).write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(report.get("trades", [])).to_csv(run_cfg["output_trades_csv"], index=False)
    pd.DataFrame(report.get("equity_curve", [])).to_csv(run_cfg["output_equity_csv"], index=False)


def build_walkforward_cmp(train: dict, test: dict) -> dict:
    def one(prefix: str, rep: dict):
        s = rep
        return {
            "trades": int(s.get("trades", 0)),
            "wr": float(s.get("wr", 0.0)),
            "pf": float(s.get("pf", 0.0)),
            "net": float(s.get("net_pnl", 0.0)),
            "maxdd": float(s.get("max_dd", 0.0)),
        }

    out = {
        "walkforward": {
            "strat_a": {
                "train": one("a", train["strategy_summaries"]["strat_a_meanrev"]),
                "test": one("a", test["strategy_summaries"]["strat_a_meanrev"]),
            },
            "strat_b": {
                "train": one("b", train["strategy_summaries"]["strat_b_breakout"]),
                "test": one("b", test["strategy_summaries"]["strat_b_breakout"]),
            },
            "combined": {
                "train": {
                    "trades": int(train["combined_portfolio"]["combined_trades"]),
                    "pf": float(train["combined_portfolio"]["combined_pf"]),
                    "net": float(train["combined_portfolio"]["combined_net"]),
                    "maxdd": float(train["combined_portfolio"]["combined_max_dd"]),
                },
                "test": {
                    "trades": int(test["combined_portfolio"]["combined_trades"]),
                    "pf": float(test["combined_portfolio"]["combined_pf"]),
                    "net": float(test["combined_portfolio"]["combined_net"]),
                    "maxdd": float(test["combined_portfolio"]["combined_max_dd"]),
                },
            },
        }
    }

    # retention
    for k in ["strat_a", "strat_b"]:
        pf_tr = out["walkforward"][k]["train"]["pf"]
        pf_te = out["walkforward"][k]["test"]["pf"]
        out["walkforward"][k]["pf_retention_pct"] = (pf_te / pf_tr * 100.0) if pf_tr else 0.0
        out["walkforward"][k]["verdict"] = "ROBUST" if (pf_tr and pf_te >= 0.6 * pf_tr) else ("INCONCLUSIVE" if pf_tr == 0 else "DECAY")

    c_tr = out["walkforward"]["combined"]["train"]["pf"]
    c_te = out["walkforward"]["combined"]["test"]["pf"]
    out["walkforward"]["combined"]["pf_retention_pct"] = (c_te / c_tr * 100.0) if c_tr else 0.0
    out["walkforward"]["combined"]["verdict"] = "ROBUST" if (c_tr and c_te >= 0.7 * c_tr) else ("INCONCLUSIVE" if c_tr == 0 else "DECAY")
    return out


def main() -> int:
    args = parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    runs = list(cfg.get("run_sequence", []))
    if not runs:
        raise RuntimeError("Config missing run_sequence")

    events = load_news_events(cfg)
    reports = {}

    for r in runs:
        rep = run_one(cfg, r, events)
        write_outputs(rep, r)
        reports[str(r["label"]).lower()] = rep

        s = rep["combined_portfolio"]
        a = rep["strategy_summaries"]["strat_a_meanrev"]
        b = rep["strategy_summaries"]["strat_b_breakout"]
        print(
            f"[{r['label']}] combined trades={s['combined_trades']} pf={s['combined_pf']:.3f} net={s['combined_net']:.2f} maxdd={s['combined_max_dd']:.2f} | "
            f"A: trades={a['trades']} pf={a['pf']:.3f} | B: trades={b['trades']} pf={b['pf']:.3f}"
        )

    # write walkforward comparison
    if "walkforward_700k" in reports and "walkforward_300k" in reports:
        cmp = build_walkforward_cmp(reports["walkforward_700k"], reports["walkforward_300k"])
        Path(cfg["outputs"]["walkforward_comparison_json"]).write_text(json.dumps(cmp, indent=2), encoding="utf-8")

    # write day/month/hour/asian analysis from 1000k
    if "1000k" in reports:
        r = reports["1000k"]
        Path(cfg["outputs"]["day_analysis_json"]).write_text(json.dumps(r["day_analysis"], indent=2), encoding="utf-8")
        Path(cfg["outputs"]["month_analysis_json"]).write_text(json.dumps(r["month_analysis"], indent=2), encoding="utf-8")
        Path(cfg["outputs"]["hour_analysis_json"]).write_text(json.dumps(r["hour_analysis"], indent=2), encoding="utf-8")
        Path(cfg["outputs"]["asian_range_analysis_json"]).write_text(json.dumps(r["asian_range_analysis"], indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
