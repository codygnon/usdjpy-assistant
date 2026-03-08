#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PIP_SIZE = 0.01


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()


def load_m1(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    req = {"time", "open", "high", "low", "close"}
    if not req.issubset(df.columns):
        raise ValueError(f"Missing required columns in {path}: {req - set(df.columns)}")
    cols = ["time", "open", "high", "low", "close"] + (["volume"] if "volume" in df.columns else [])
    out = df[cols].copy()
    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
    for c in ["open", "high", "low", "close"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if "volume" in out.columns:
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    out = out.dropna(subset=["time", "open", "high", "low", "close"]).sort_values("time").reset_index(drop=True)
    out["date_utc"] = out["time"].dt.date.astype(str)
    out["day_name"] = out["time"].dt.day_name()
    out["hour"] = out["time"].dt.hour
    out["minute"] = out["time"].dt.minute
    out["minute_of_day"] = out["hour"] * 60 + out["minute"]
    return out


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "volume" in df.columns:
        agg["volume"] = "sum"
    r = (
        df.set_index("time")
        .resample(rule, closed="right", label="right")
        .agg(agg)
        .dropna(subset=["open", "high", "low", "close"])
        .reset_index()
    )
    r["hour"] = r["time"].dt.hour
    r["minute"] = r["time"].dt.minute
    return r


def rsi(series: pd.Series, period: int) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0.0)
    down = -d.clip(upper=0.0)
    au = up.rolling(period, min_periods=period).mean()
    ad = down.rolling(period, min_periods=period).mean()
    rs = au / ad.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_spread_pips(i: int, ts: pd.Timestamp, cfg: dict) -> float:
    mode = str(cfg.get("spread_mode", "realistic")).lower()
    avg = float(cfg.get("spread_pips", 1.6))
    mn = float(cfg.get("spread_min_pips", 1.0))
    mx = float(cfg.get("spread_max_pips", 3.5))
    if mode == "fixed":
        return max(mn, min(mx, avg))
    h = ts.hour + ts.minute / 60.0
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


def get_bid_ask(mid: float, spread_pips: float) -> Tuple[float, float]:
    hs = spread_pips * PIP_SIZE / 2.0
    return float(mid - hs), float(mid + hs)


def in_window(m: int, s: int, e: int) -> bool:
    if s < e:
        return s <= m < e
    return m >= s or m < e


def hhmm_to_min(s: str) -> int:
    hh, mm = s.split(":")
    return int(hh) * 60 + int(mm)


@dataclass
class Position:
    trade_id: int
    direction: str
    entry_time: pd.Timestamp
    entry_price: float
    entry_level_name: str
    entry_level_tier: int
    entry_level_price: float
    zone_distance_at_entry_pips: float
    confirmation_signals_present: List[str]
    confluence_count: int
    ib_classification: str
    session_phase_at_entry: str
    rsi_at_entry: float
    vwap_at_entry: float
    sl_price: float
    tp_price: float
    sl_pips: float
    tp_pips: float
    units: int
    trailing_enabled: bool
    trailing_activate_pips: float
    trail_dist_pips: float
    late_profile: bool
    overlap_tier1: bool
    moved_to_trail: bool = False
    trail_price: Optional[float] = None
    max_fav_pips: float = 0.0
    max_adv_pips: float = 0.0
    mfe_time: Optional[pd.Timestamp] = None
    level_retouch_start: Optional[pd.Timestamp] = None


class Backtester:
    def __init__(self, cfg: dict, run_cfg: dict):
        self.cfg = cfg
        self.run = run_cfg
        self.df = load_m1(run_cfg["input_csv"])
        self.m5 = resample_ohlc(self.df, "5min")
        self.m15 = resample_ohlc(self.df, "15min")
        self._add_indicators()

        self.start_equity = float(cfg["account"]["starting_balance_usd"])
        self.equity = self.start_equity
        self.max_units = int(cfg["position_sizing"]["max_position_units"])
        self.max_open = int(cfg["position_sizing"]["max_concurrent_positions"])
        self.base_risk = float(cfg["position_sizing"]["risk_per_trade_pct"]) / 100.0
        self.leverage = float(cfg["account"]["leverage"])
        self.margin_enabled = bool(cfg["execution"]["margin_enabled"])

        sess = cfg["session"]
        self.sess_start = hhmm_to_min(sess["start_utc"])
        self.sess_end = hhmm_to_min(sess["end_utc"])
        self.lunch_start = hhmm_to_min(sess["lunch_block_start_utc"])
        self.lunch_end = hhmm_to_min(sess["lunch_block_end_utc"])
        self.wind_start = hhmm_to_min(sess["wind_down_start_utc"])
        self.force_close_min = hhmm_to_min(sess["force_close_utc"])
        self.allowed_days = set(sess["allowed_days"])

        self.tp_base = float(cfg["exits"]["normal"]["tp_pips"])
        self.sl_base = float(cfg["exits"]["normal"]["sl_pips"])
        self.tp_overlap = float(cfg["exits"]["normal"]["tp_pips_overlap_tier1"])
        self.trail_activate_base = float(cfg["exits"]["normal"]["trail_activate_pips"])
        self.trail_dist_base = float(cfg["exits"]["normal"]["trail_distance_pips"])

        self.tp_late = float(cfg["exits"]["late_session"]["tp_pips"])
        self.sl_late = float(cfg["exits"]["late_session"]["sl_pips"])
        self.trail_activate_late = float(cfg["exits"]["late_session"]["trail_activate_pips"])
        self.trail_dist_late = float(cfg["exits"]["late_session"]["trail_distance_pips"])

        tm = cfg["trade_management"]
        self.min_gap = int(tm["min_time_between_entries_minutes"])
        self.reentry_sl_same_level = int(tm["no_reentry_same_level_after_sl_minutes"])
        self.pause_losses = int(tm["consecutive_loss_pause"]["loss_count"])
        self.pause_mins = int(tm["consecutive_loss_pause"]["pause_minutes"])
        self.breakout_kill_pips = float(tm["breakout_session_kill_pips"])

        self.positions: List[Position] = []
        self.closed: List[dict] = []

        self.trade_id = 0
        self.last_entry_ts: Optional[pd.Timestamp] = None
        self.session_state: Dict[str, dict] = {}

        self.diag = {
            "total_m1_candles": int(len(self.df)),
            "candles_inside_session_window": 0,
            "candles_tradeable_phase": 0,
            "price_in_zone_events": 0,
            "zone_touches": {k: 0 for k in ["PTS_High","PTS_Low","PTS_Mid","IB_High","IB_Low","IB_Mid"]},
            "signal_fires": {"A_M5_rejection": 0, "B_RSI": 0, "C_VWAP": 0, "D_Round": 0},
            "entries_qualifying": 0,
            "entries_taken": 0,
            "blocked_max_concurrent": 0,
            "blocked_cooldown": 0,
            "blocked_level_invalidation": 0,
            "blocked_breakout_detection": 0,
            "blocked_ib_classification": 0,
            "blocked_late_session_requirements": 0,
            "blocked_pause": 0,
            "near_miss_candidates": [],
            "breakout_session_kill_count": 0,
        }

    def _add_indicators(self):
        # M15 RSI
        self.m15["rsi14"] = rsi(self.m15["close"], 14)
        self.m15["rsi14_l1"] = self.m15["rsi14"].shift(1)
        self.m15["rsi14_l2"] = self.m15["rsi14"].shift(2)
        self.m15["rsi14_l3"] = self.m15["rsi14"].shift(3)

        # M5 rejection features
        rng = (self.m5["high"] - self.m5["low"]).replace(0.0, np.nan)
        upper_close_th = self.m5["low"] + 0.67 * rng
        lower_close_th = self.m5["high"] - 0.67 * rng
        lower_wick = self.m5[["open", "close"]].min(axis=1) - self.m5["low"]
        upper_wick = self.m5["high"] - self.m5[["open", "close"]].max(axis=1)
        self.m5["bull_rej_shape"] = (self.m5["close"] > upper_close_th) & (lower_wick >= 0.60 * rng)
        self.m5["bear_rej_shape"] = (self.m5["close"] < lower_close_th) & (upper_wick >= 0.60 * rng)

        # merge-asof to m1
        m5_cols = ["time", "open", "high", "low", "close", "bull_rej_shape", "bear_rej_shape"]
        m5r = self.m5[m5_cols].copy().rename(columns={
            "open":"m5_open","high":"m5_high","low":"m5_low","close":"m5_close",
            "bull_rej_shape":"m5_bull_rej_shape","bear_rej_shape":"m5_bear_rej_shape"
        })
        m5r["m5_time"] = m5r["time"]
        m5r = m5r.drop(columns=["time"]).sort_values("m5_time")

        # add previous m5 candle features
        for c in ["m5_open","m5_high","m5_low","m5_close","m5_bull_rej_shape","m5_bear_rej_shape"]:
            m5r[c+"_prev1"] = m5r[c].shift(1)

        self.df = pd.merge_asof(self.df.sort_values("time"), m5r, left_on="time", right_on="m5_time", direction="backward")

        m15r = self.m15[["time","rsi14","rsi14_l1","rsi14_l2","rsi14_l3"]].copy().rename(columns={"time":"m15_time"}).sort_values("m15_time")
        self.df = pd.merge_asof(self.df.sort_values("time"), m15r, left_on="time", right_on="m15_time", direction="backward")

        # running VWAP or proxy from session start each day
        tp = (self.df["high"] + self.df["low"] + self.df["close"]) / 3.0
        if "volume" in self.df.columns and self.df["volume"].notna().any():
            v = self.df["volume"].fillna(0.0)
            self.df["vwap"] = np.nan
            for d, gidx in self.df.groupby("date_utc").groups.items():
                gi = list(gidx)
                ctpv = (tp.iloc[gi] * v.iloc[gi]).cumsum()
                cv = v.iloc[gi].cumsum().replace(0.0, np.nan)
                self.df.loc[self.df.index[gi], "vwap"] = (ctpv / cv).values
        else:
            self.df["vwap"] = np.nan
            for d, gidx in self.df.groupby("date_utc").groups.items():
                gi = list(gidx)
                self.df.loc[self.df.index[gi], "vwap"] = tp.iloc[gi].expanding(min_periods=1).mean().values

    def _phase(self, minute_of_day: int) -> str:
        if 0 <= minute_of_day < 60:
            return "IB_Formation_PTS"
        if 60 <= minute_of_day < 150:
            return "Morning_Active"
        if 150 <= minute_of_day < 210:
            return "Lunch_Block"
        if 210 <= minute_of_day < 420:
            return "Afternoon_Active"
        if 420 <= minute_of_day < 510:
            return "Late_Session"
        if 510 <= minute_of_day < 525:
            return "Wind_Down"
        if 525 <= minute_of_day < 540:
            return "Force_Close_Window"
        return "Outside"

    def _ensure_session(self, date_key: str, row: pd.Series):
        if date_key in self.session_state:
            return self.session_state[date_key]

        # prior session map from completed sessions in dataset
        s = {
            "date": date_key,
            "session_open_mid": float((row["open"] + row["close"]) / 2.0),
            "breakout_killed": False,
            "level_disabled": set(),
            "level_sl_cooldown_until": {},
            "consec_losses": 0,
            "pause_until": None,
            "ib_ready": False,
            "ib_class": "unknown",
            "ib": None,
            "pts": None,
            "phase_entries": Counter(),
            "signals_generated": 0,
            "entries_taken": 0,
            "wins": 0,
            "losses": 0,
            "session_pnl": 0.0,
            "entry_scores": [],
        }

        # PTS from previous available weekday session summary
        # build lazily from precomputed map
        if hasattr(self, "prior_pts_map") and date_key in self.prior_pts_map:
            s["pts"] = self.prior_pts_map[date_key]

        self.session_state[date_key] = s
        return s

    def _build_prior_pts_map(self):
        # session summary by UTC date over 00:00-09:00 window
        d = self.df.copy()
        in_sess = (d["minute_of_day"] >= 0) & (d["minute_of_day"] < 540) & d["day_name"].isin(self.allowed_days)
        ds = d[in_sess].groupby("date_utc").agg(session_high=("high", "max"), session_low=("low", "min"))
        ds = ds.sort_index()
        self.prior_pts_map = {}
        idx = list(ds.index)
        for i, date_key in enumerate(idx):
            # previous available session (Friday for Monday naturally via index)
            if i == 0:
                continue
            prev = ds.iloc[i - 1]
            rng = float(prev["session_high"] - prev["session_low"])
            if rng <= 0:
                self.prior_pts_map[date_key] = None
            else:
                self.prior_pts_map[date_key] = {
                    "PTS_High": float(prev["session_high"]),
                    "PTS_Low": float(prev["session_low"]),
                    "PTS_Mid": float((prev["session_high"] + prev["session_low"]) / 2.0),
                    "PTS_Range": rng,
                }

    def _compute_ib_for_session(self, date_key: str):
        # first 60 min of date session
        dd = self.df[(self.df["date_utc"] == date_key) & (self.df["minute_of_day"] >= 0) & (self.df["minute_of_day"] < 60)]
        if dd.empty:
            return None
        hi = float(dd["high"].max())
        lo = float(dd["low"].min())
        rng = hi - lo
        if rng < 0.03:
            cls = "dead"
        elif rng > 0.30:
            cls = "volatile"
        elif rng < 0.10:
            cls = "tight"
        elif rng < 0.20:
            cls = "normal"
        else:
            cls = "wide"
        return {
            "IB_High": hi,
            "IB_Low": lo,
            "IB_Mid": float((hi + lo) / 2.0),
            "IB_Range": rng,
            "classification": cls,
        }

    def _used_margin(self):
        if not self.margin_enabled:
            return 0.0
        return float(sum(p.units / self.leverage for p in self.positions))

    def _margin_ok(self, units: int):
        if not self.margin_enabled:
            return True
        need = units / self.leverage
        free = self.equity - self._used_margin()
        return free >= need

    def _current_round_levels(self, price: float):
        above = math.ceil(price / 0.5) * 0.5
        below = math.floor(price / 0.5) * 0.5
        return below, above

    def _zones_for_row(self, sstate: dict, row: pd.Series):
        zones = []
        # PTS available from 00:00
        pts = sstate.get("pts")
        if pts:
            zones.append(("PTS_High", pts["PTS_High"], 1, 0.04))
            zones.append(("PTS_Low", pts["PTS_Low"], 1, 0.04))
            zones.append(("PTS_Mid", pts["PTS_Mid"], 2, 0.03))

        # IB available from 01:00 and depending classification
        if sstate.get("ib_ready") and sstate.get("ib"):
            ib = sstate["ib"]
            cls = ib["classification"]
            if cls not in {"dead", "volatile"}:
                zones.append(("IB_High", ib["IB_High"], 1, 0.03))
                zones.append(("IB_Low", ib["IB_Low"], 1, 0.03))
                if cls != "wide":
                    zones.append(("IB_Mid", ib["IB_Mid"], 2, 0.02))

        # VWAP not entry level (tier3)
        # Round numbers not entry level (tier3)
        return zones

    def _level_direction(self, level_name: str, level_price: float, tol: float, row: pd.Series) -> Optional[str]:
        # support/resistance fixed for high/low. Mid by approach from prior completed M5 close.
        if level_name.endswith("_High"):
            return "short"
        if level_name.endswith("_Low"):
            return "long"
        if level_name.endswith("_Mid"):
            prev_close = row.get("m5_close_prev1", np.nan)
            if pd.isna(prev_close):
                return None
            zlo, zhi = level_price - tol, level_price + tol
            if prev_close > zhi:
                return "long"  # dropped into mid from above => support
            if prev_close < zlo:
                return "short" # rose into mid from below => resistance
            return None
        return None

    def _confirmations(self, direction: str, level_price: float, tol: float, row: pd.Series) -> List[str]:
        sigs = []

        # A: rejection shape on one of last 2 completed M5 candles + touched zone
        def m5_touch(low, high):
            zlo, zhi = level_price - tol, level_price + tol
            return (low <= zhi) and (high >= zlo)

        a = False
        for suffix in ["", "_prev1"]:
            low = row.get(f"m5_low{suffix}", np.nan)
            high = row.get(f"m5_high{suffix}", np.nan)
            if pd.isna(low) or pd.isna(high):
                continue
            touched = m5_touch(float(low), float(high))
            if not touched:
                continue
            if direction == "long" and bool(row.get(f"m5_bull_rej_shape{suffix}", False)):
                a = True
            if direction == "short" and bool(row.get(f"m5_bear_rej_shape{suffix}", False)):
                a = True
        if a:
            sigs.append("A")
            self.diag["signal_fires"]["A_M5_rejection"] += 1

        # B: RSI divergence from extreme logic (M15)
        r = row.get("rsi14", np.nan)
        r1 = row.get("rsi14_l1", np.nan)
        r2 = row.get("rsi14_l2", np.nan)
        r3 = row.get("rsi14_l3", np.nan)
        b = False
        if not pd.isna(r):
            if direction == "long":
                if r < 40:
                    b = True
                else:
                    prev = [x for x in [r1, r2, r3] if not pd.isna(x)]
                    if prev:
                        mn = min(prev)
                        if mn < 35 and r > mn:
                            b = True
            else:
                if r > 60:
                    b = True
                else:
                    prev = [x for x in [r1, r2, r3] if not pd.isna(x)]
                    if prev:
                        mx = max(prev)
                        if mx > 65 and r < mx:
                            b = True
        if b:
            sigs.append("B")
            self.diag["signal_fires"]["B_RSI"] += 1

        # C: VWAP alignment (tier3 confluence)
        vwap = row.get("vwap", np.nan)
        c = False
        if not pd.isna(vwap):
            px = float(row["close"])
            if direction == "long":
                c = px < float(vwap)
            else:
                c = px > float(vwap)
        if c:
            sigs.append("C")
            self.diag["signal_fires"]["C_VWAP"] += 1

        # D: Round number proximity
        px = float(row["close"])
        rbelow, rabove = self._current_round_levels(px)
        d = False
        if direction == "long":
            d = 0 <= (px - rbelow) / PIP_SIZE <= 5.0
        else:
            d = 0 <= (rabove - px) / PIP_SIZE <= 5.0
        if d:
            sigs.append("D")
            self.diag["signal_fires"]["D_Round"] += 1

        return sigs

    def _phase_multipliers(self, phase: str):
        if phase == "Late_Session":
            return self.tp_late, self.sl_late, self.trail_activate_late, self.trail_dist_late
        return self.tp_base, self.sl_base, self.trail_activate_base, self.trail_dist_base

    def _ib_size_mult(self, ib_class: str) -> float:
        return {
            "dead": 0.75,
            "tight": 1.0,
            "normal": 1.0,
            "wide": 0.75,
            "volatile": 0.50,
            "unknown": 1.0,
        }.get(ib_class, 1.0)

    def _open_trade(self, row: pd.Series, direction: str, level_name: str, level_tier: int, level_price: float, tol: float, confs: List[str], overlap_tier1: bool, sstate: dict, phase: str):
        spread_pips = compute_spread_pips(int(row.name), row["time"], self.cfg["account"])
        bid, ask = get_bid_ask(float(row["close"]), spread_pips)
        entry = ask if direction == "long" else bid

        tp_pips, sl_pips, tr_act, tr_dist = self._phase_multipliers(phase)
        if overlap_tier1:
            tp_pips = self.tp_overlap

        sl = entry - sl_pips * PIP_SIZE if direction == "long" else entry + sl_pips * PIP_SIZE
        tp = entry + tp_pips * PIP_SIZE if direction == "long" else entry - tp_pips * PIP_SIZE

        risk_pct = self.base_risk * self._ib_size_mult(sstate.get("ib_class", "unknown"))
        units = math.floor((self.equity * risk_pct) / (sl_pips * (PIP_SIZE / max(1e-9, entry))))
        units = int(max(0, min(self.max_units, units)))
        if units < 1:
            self.diag["blocked_cooldown"] += 1
            return False
        if not self._margin_ok(units):
            self.diag["blocked_max_concurrent"] += 1
            return False

        self.trade_id += 1
        self.positions.append(Position(
            trade_id=self.trade_id,
            direction=direction,
            entry_time=row["time"],
            entry_price=entry,
            entry_level_name=level_name,
            entry_level_tier=level_tier,
            entry_level_price=level_price,
            zone_distance_at_entry_pips=abs(float(row["close"]) - level_price) / PIP_SIZE,
            confirmation_signals_present=confs,
            confluence_count=len(confs) + (1 if overlap_tier1 else 0),
            ib_classification=sstate.get("ib_class", "unknown"),
            session_phase_at_entry=phase,
            rsi_at_entry=float(row.get("rsi14", np.nan)) if not pd.isna(row.get("rsi14", np.nan)) else np.nan,
            vwap_at_entry=float(row.get("vwap", np.nan)) if not pd.isna(row.get("vwap", np.nan)) else np.nan,
            sl_price=sl,
            tp_price=tp,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            units=units,
            trailing_enabled=True,
            trailing_activate_pips=tr_act,
            trail_dist_pips=tr_dist,
            late_profile=(phase == "Late_Session"),
            overlap_tier1=overlap_tier1,
        ))
        self.last_entry_ts = row["time"]
        sstate["entries_taken"] += 1
        sstate["phase_entries"][phase] += 1
        sstate["entry_scores"].append(len(confs) + (1 if overlap_tier1 else 0))
        self.diag["entries_taken"] += 1
        return True

    def _close_trade(self, pos: Position, ts: pd.Timestamp, exit_price: float, reason: str, sstate: dict):
        pips = (exit_price - pos.entry_price) / PIP_SIZE if pos.direction == "long" else (pos.entry_price - exit_price) / PIP_SIZE
        usd = pips * pos.units * (PIP_SIZE / max(1e-9, exit_price))
        self.equity += usd

        duration = (ts - pos.entry_time).total_seconds() / 60.0
        time_to_mfe = (pos.mfe_time - pos.entry_time).total_seconds() / 60.0 if pos.mfe_time is not None else np.nan

        self.closed.append({
            "trade_id": pos.trade_id,
            "entry_datetime": str(pos.entry_time),
            "exit_datetime": str(ts),
            "direction": pos.direction,
            "entry_price": pos.entry_price,
            "exit_price": float(exit_price),
            "sl_price": pos.sl_price,
            "tp_price": pos.tp_price,
            "exit_reason": reason,
            "pips": float(pips),
            "pnl_usd": float(usd),
            "usd": float(usd),
            "position_units": int(pos.units),
            "entry_level_name": pos.entry_level_name,
            "entry_level_tier": pos.entry_level_tier,
            "entry_level_price": pos.entry_level_price,
            "zone_distance_at_entry_pips": pos.zone_distance_at_entry_pips,
            "confirmation_signals_present": ",".join(pos.confirmation_signals_present),
            "confluence_count": int(pos.confluence_count),
            "ib_classification": pos.ib_classification,
            "session_phase_at_entry": pos.session_phase_at_entry,
            "rsi_at_entry": pos.rsi_at_entry,
            "vwap_at_entry": pos.vwap_at_entry,
            "mfe_pips": float(pos.max_fav_pips),
            "mae_pips": float(pos.max_adv_pips),
            "time_to_mfe_minutes": float(time_to_mfe) if not pd.isna(time_to_mfe) else np.nan,
            "trade_duration_minutes": float(duration),
            "day_of_week": pd.Timestamp(pos.entry_time).day_name(),
            "hour_utc": int(pd.Timestamp(pos.entry_time).hour),
        })

        sstate["session_pnl"] += float(usd)
        if usd > 0:
            sstate["wins"] += 1
            sstate["consec_losses"] = 0
        else:
            sstate["losses"] += 1
            sstate["consec_losses"] += 1
            if reason.startswith("SL"):
                sstate["level_sl_cooldown_until"][pos.entry_level_name] = ts + pd.Timedelta(minutes=self.reentry_sl_same_level)
            if sstate["consec_losses"] >= self.pause_losses:
                sstate["pause_until"] = ts + pd.Timedelta(minutes=self.pause_mins)
                sstate["consec_losses"] = 0

    def _update_positions(self, row: pd.Series, sstate: dict, phase: str):
        if not self.positions:
            return
        spread_pips = compute_spread_pips(int(row.name), row["time"], self.cfg["account"])
        bid_h, ask_h = get_bid_ask(float(row["high"]), spread_pips)
        bid_l, ask_l = get_bid_ask(float(row["low"]), spread_pips)
        bid_c, ask_c = get_bid_ask(float(row["close"]), spread_pips)

        to_close = []
        for pos in self.positions:
            if pos.direction == "long":
                fav = (bid_h - pos.entry_price) / PIP_SIZE
                adv = (pos.entry_price - bid_l) / PIP_SIZE
                cur = (bid_c - pos.entry_price) / PIP_SIZE
            else:
                fav = (pos.entry_price - ask_l) / PIP_SIZE
                adv = (ask_h - pos.entry_price) / PIP_SIZE
                cur = (pos.entry_price - ask_c) / PIP_SIZE

            if fav > pos.max_fav_pips:
                pos.max_fav_pips = float(fav)
                pos.mfe_time = row["time"]
            pos.max_adv_pips = max(pos.max_adv_pips, float(adv))

            # trailing
            if not pos.moved_to_trail and fav >= pos.trailing_activate_pips:
                pos.moved_to_trail = True
            if pos.moved_to_trail:
                if pos.direction == "long":
                    trail = bid_c - pos.trail_dist_pips * PIP_SIZE
                    pos.trail_price = trail if pos.trail_price is None else max(pos.trail_price, trail)
                else:
                    trail = ask_c + pos.trail_dist_pips * PIP_SIZE
                    pos.trail_price = trail if pos.trail_price is None else min(pos.trail_price, trail)

            # level failure logic
            lvl = pos.entry_level_price
            tol = {"PTS_High":0.04,"PTS_Low":0.04,"PTS_Mid":0.03,"IB_High":0.03,"IB_Low":0.03,"IB_Mid":0.02}.get(pos.entry_level_name,0.03)
            in_zone = (float(row["low"]) <= lvl + tol) and (float(row["high"]) >= lvl - tol)
            if in_zone and (-7.0 <= cur <= -3.0):
                if pos.level_retouch_start is None:
                    pos.level_retouch_start = row["time"]
                elif (row["time"] - pos.level_retouch_start).total_seconds() >= 15 * 60:
                    to_close.append((pos, bid_c if pos.direction=="long" else ask_c, "Level_Failure_Exit"))
                    sstate["level_disabled"].add(pos.entry_level_name)
                    continue
            else:
                pos.level_retouch_start = None

            # wind-down handling
            if phase == "Wind_Down":
                if cur > 0:
                    to_close.append((pos, bid_c if pos.direction=="long" else ask_c, "Wind_Down_Close_Profit"))
                    continue
                if cur < -8:
                    to_close.append((pos, bid_c if pos.direction=="long" else ask_c, "Wind_Down_Close_Large_Loss"))
                    continue
                if cur > -3:
                    to_close.append((pos, bid_c if pos.direction=="long" else ask_c, "Wind_Down_Close_Small_Loss"))
                    continue

            # force close
            if row["minute_of_day"] >= self.force_close_min:
                to_close.append((pos, bid_c if pos.direction=="long" else ask_c, "Force_Close_08_45"))
                continue

            # TP/SL/trail checks
            if pos.direction == "long":
                if bid_l <= pos.sl_price:
                    reason = "SL_Hit_8" if pos.late_profile else "SL_Hit_10"
                    to_close.append((pos, pos.sl_price, reason))
                    continue
                if bid_h >= pos.tp_price:
                    reason = "TP_Hit_10_overlap" if pos.overlap_tier1 else ("TP_Hit_5_late" if pos.late_profile else "TP_Hit_7")
                    to_close.append((pos, pos.tp_price, reason))
                    continue
                if pos.moved_to_trail and pos.trail_price is not None and bid_l <= pos.trail_price:
                    to_close.append((pos, pos.trail_price, "Trailing_Stop"))
                    continue
            else:
                if ask_h >= pos.sl_price:
                    reason = "SL_Hit_8" if pos.late_profile else "SL_Hit_10"
                    to_close.append((pos, pos.sl_price, reason))
                    continue
                if ask_l <= pos.tp_price:
                    reason = "TP_Hit_10_overlap" if pos.overlap_tier1 else ("TP_Hit_5_late" if pos.late_profile else "TP_Hit_7")
                    to_close.append((pos, pos.tp_price, reason))
                    continue
                if pos.moved_to_trail and pos.trail_price is not None and ask_h >= pos.trail_price:
                    to_close.append((pos, pos.trail_price, "Trailing_Stop"))
                    continue

        if to_close:
            for pos, px, reason in to_close:
                if pos in self.positions:
                    self._close_trade(pos, row["time"], float(px), reason, sstate)
                    self.positions.remove(pos)

    def _level_invalidation_update(self, row: pd.Series, sstate: dict):
        # 2 consecutive M5 closes beyond level invalidates side
        # use completed m5 close and prev m5 close
        c0 = row.get("m5_close", np.nan)
        c1 = row.get("m5_close_prev1", np.nan)
        if pd.isna(c0) or pd.isna(c1):
            return
        lvls = {}
        if sstate.get("pts"):
            lvls.update({k: sstate["pts"][k] for k in ["PTS_High", "PTS_Low", "PTS_Mid"]})
        if sstate.get("ib") and sstate.get("ib_ready"):
            lvls.update({k: sstate["ib"][k] for k in ["IB_High", "IB_Low", "IB_Mid"] if k in sstate["ib"]})
        for ln, lp in lvls.items():
            if ln.endswith("_High") and c0 > lp and c1 > lp:
                sstate["level_disabled"].add(ln)
            if ln.endswith("_Low") and c0 < lp and c1 < lp:
                sstate["level_disabled"].add(ln)
            if ln.endswith("_Mid"):
                # mid invalidation not directional in spec; keep tradable.
                pass

    def _entry_logic(self, row: pd.Series, sstate: dict, phase: str):
        m = int(row["minute_of_day"])
        day = str(row["day_name"])
        if day not in self.allowed_days:
            return
        if not in_window(m, self.sess_start, self.sess_end):
            return

        self.diag["candles_inside_session_window"] += 1

        # tradeable phase excludes lunch + winddown/force-close windows
        if phase in {"Lunch_Block", "Wind_Down", "Force_Close_Window", "Outside"}:
            return
        self.diag["candles_tradeable_phase"] += 1

        # ensure IB after 01:00 once
        if (not sstate["ib_ready"]) and m >= 60:
            ib = self._compute_ib_for_session(sstate["date"])
            sstate["ib"] = ib
            sstate["ib_ready"] = ib is not None
            sstate["ib_class"] = ib["classification"] if ib else "unknown"

        # breakout session kill
        cur_mid = float((row["open"] + row["close"]) / 2.0)
        if abs(cur_mid - float(sstate["session_open_mid"])) / PIP_SIZE > self.breakout_kill_pips:
            if not sstate["breakout_killed"]:
                self.diag["breakout_session_kill_count"] += 1
            sstate["breakout_killed"] = True
        if sstate["breakout_killed"]:
            self.diag["blocked_breakout_detection"] += 1
            return

        # consecutive loss pause
        if sstate.get("pause_until") is not None and row["time"] < sstate["pause_until"]:
            self.diag["blocked_pause"] += 1
            return

        # max concurrent
        if len(self.positions) >= self.max_open:
            self.diag["blocked_max_concurrent"] += 1
            return

        # min gap
        if self.last_entry_ts is not None and (row["time"] - self.last_entry_ts).total_seconds() < self.min_gap * 60:
            self.diag["blocked_cooldown"] += 1
            return

        zones = self._zones_for_row(sstate, row)

        # 00:00-01:00 only PTS
        if m < 60:
            zones = [z for z in zones if z[0].startswith("PTS")]

        # IB classification blocks IB entries in dead/volatile
        ib_class = sstate.get("ib_class", "unknown")
        if ib_class in {"dead", "volatile"}:
            before = len(zones)
            zones = [z for z in zones if z[0].startswith("PTS")]
            if len(zones) < before:
                self.diag["blocked_ib_classification"] += 1

        if not zones:
            return

        price = float(row["close"])
        touched = []
        for ln, lp, tier, tol in zones:
            if ln in sstate["level_disabled"]:
                self.diag["blocked_level_invalidation"] += 1
                continue
            cd = sstate["level_sl_cooldown_until"].get(ln)
            if cd is not None and row["time"] < cd:
                self.diag["blocked_cooldown"] += 1
                continue
            if (price >= lp - tol) and (price <= lp + tol):
                touched.append((ln, lp, tier, tol))
                self.diag["zone_touches"][ln] += 1

        if not touched:
            return

        self.diag["price_in_zone_events"] += 1

        # anti-stacking: choose highest-tier touched level (tier1 preferred), nearest price
        touched_sorted = sorted(touched, key=lambda x: (x[2], abs(price - x[1])))
        ln, lp, tier, tol = touched_sorted[0]
        overlap_tier1 = sum(1 for x in touched if x[2] == 1) >= 2

        direction = self._level_direction(ln, lp, tol, row)
        if direction is None:
            return

        confs = self._confirmations(direction, lp, tol, row)
        # overlap zone bonus
        conf_count = len(confs) + (1 if overlap_tier1 else 0)

        required = 1 if tier == 1 else 2
        if phase == "Late_Session":
            required += 1
            if required > 4:
                required = 4

        sstate["signals_generated"] += 1
        self.diag["entries_qualifying"] += int(conf_count >= required)

        if conf_count < required:
            # near miss log for tier1 only and no confirmation signal
            if tier == 1 and len(confs) == 0:
                self.diag["near_miss_candidates"].append({
                    "datetime": str(row["time"]),
                    "level_name": ln,
                    "direction": direction,
                    "zone_distance_pips": abs(price - lp) / PIP_SIZE,
                    "which_confirmations_checked": ["A","B","C","D"],
                    "which_failed": ["A","B","C","D"],
                    "entry_price": price,
                    "entry_index": int(row.name),
                })
            if phase == "Late_Session":
                self.diag["blocked_late_session_requirements"] += 1
            return

        self._open_trade(row, direction, ln, tier, lp, tol, confs, overlap_tier1, sstate, phase)

    def run_backtest(self):
        self._build_prior_pts_map()

        for i, row in self.df.iterrows():
            date_key = row["date_utc"]
            sstate = self._ensure_session(date_key, row)
            phase = self._phase(int(row["minute_of_day"]))

            # level invalidation using completed m5 closes
            self._level_invalidation_update(row, sstate)

            # manage open first
            self._update_positions(row, sstate, phase)

            # entries
            self._entry_logic(row, sstate, phase)

        # close leftovers at last bar
        if self.positions:
            last = self.df.iloc[-1]
            spread_pips = compute_spread_pips(int(last.name), last["time"], self.cfg["account"])
            bid, ask = get_bid_ask(float(last["close"]), spread_pips)
            sstate = self._ensure_session(last["date_utc"], last)
            for p in list(self.positions):
                px = bid if p.direction == "long" else ask
                self._close_trade(p, last["time"], px, "Force_Close_08_45", sstate)
                self.positions.remove(p)

        return self._build_report()

    def _build_report(self):
        t = pd.DataFrame(self.closed)
        if t.empty:
            raise RuntimeError("No trades generated")

        t["entry_datetime"] = pd.to_datetime(t["entry_datetime"], utc=True)
        t["exit_datetime"] = pd.to_datetime(t["exit_datetime"], utc=True)
        t["duration_min"] = (t["exit_datetime"] - t["entry_datetime"]).dt.total_seconds() / 60.0
        t["is_win"] = t["pnl_usd"] > 0
        t["month"] = t["entry_datetime"].dt.to_period("M").astype(str)
        t["dow"] = t["entry_datetime"].dt.day_name()

        gp = float(t.loc[t["pnl_usd"] > 0, "pnl_usd"].sum())
        gl = abs(float(t.loc[t["pnl_usd"] < 0, "pnl_usd"].sum()))
        pf = gp / gl if gl > 0 else float("inf")
        net = float(t["pnl_usd"].sum())
        wr = float((t["is_win"]).mean() * 100.0)

        # equity curve
        eq = [self.start_equity]
        for x in t["pnl_usd"].tolist():
            eq.append(eq[-1] + float(x))
        equity_after = np.array(eq[1:], dtype=float)
        peaks = np.maximum.accumulate(equity_after)
        dd = peaks - equity_after
        dd_pct = np.where(peaks > 0, dd / peaks * 100.0, 0.0)

        t["equity_after"] = equity_after
        t["drawdown_usd"] = dd
        t["drawdown_pct"] = dd_pct

        returns = t["pnl_usd"] / t["equity_after"].shift(1).fillna(self.start_equity)
        sharpe = float((returns.mean() / returns.std(ddof=0)) * np.sqrt(len(returns))) if returns.std(ddof=0) > 0 else 0.0
        calmar = float(((equity_after[-1] - self.start_equity) / self.start_equity * 100.0) / max(1e-9, dd_pct.max())) if len(dd_pct) else 0.0

        # breakdowns
        def grp_stats(g):
            if len(g) == 0:
                return {"trades": 0, "wr_pct": 0.0, "pf": 0.0, "net_usd": 0.0, "avg_pips": 0.0}
            gp = float(g.loc[g["pnl_usd"] > 0, "pnl_usd"].sum())
            gl = abs(float(g.loc[g["pnl_usd"] < 0, "pnl_usd"].sum()))
            pf = gp / gl if gl > 0 else float("inf")
            return {
                "trades": int(len(g)),
                "wr_pct": float(g["is_win"].mean() * 100.0),
                "pf": float(pf),
                "net_usd": float(g["pnl_usd"].sum()),
                "avg_pips": float(g["pips"].mean()),
            }

        level_perf = []
        for (lvl, d), g in t.groupby(["entry_level_name", "direction"]):
            s = grp_stats(g)
            level_perf.append({
                "level": lvl,
                "direction": d,
                **s,
                "wins": int(g["is_win"].sum()),
                "losses": int((~g["is_win"]).sum()),
                "avg_mfe": float(g["mfe_pips"].mean()),
                "avg_mae": float(g["mae_pips"].mean()),
            })

        phase_perf = []
        for ph, g in t.groupby("session_phase_at_entry"):
            s = grp_stats(g)
            phase_perf.append({"phase": ph, **s})

        sig_map = {"A": "Signal_A_M5_rejection", "B": "Signal_B_RSI", "C": "Signal_C_VWAP", "D": "Signal_D_Round"}
        signal_value = []
        for key, nm in sig_map.items():
            g = t[t["confirmation_signals_present"].fillna("").str.contains(key)]
            s = grp_stats(g)
            signal_value.append({"signal": nm, **s})

        conf_count_perf = []
        for c, g in t.groupby("confluence_count"):
            s = grp_stats(g)
            conf_count_perf.append({"confluence_count": int(c), **s})

        exit_dist = []
        for ex, g in t.groupby("exit_reason"):
            exit_dist.append({
                "exit_type": ex,
                "count": int(len(g)),
                "net_usd": float(g["pnl_usd"].sum()),
                "avg_pips": float(g["pips"].mean()),
                "pct_total": float(100.0 * len(g) / len(t)),
            })

        # ensure required rows present
        required_ex = [
            "TP_Hit_7", "TP_Hit_10_overlap", "SL_Hit_10", "SL_Hit_8", "Trailing_Stop",
            "Level_Failure_Exit", "Wind_Down_Close_Profit", "Wind_Down_Close_Small_Loss",
            "Wind_Down_Close_Large_Loss", "Force_Close_08_45", "Breakout_Session_Kill", "Consecutive_Loss_Pause_Closed"
        ]
        present = {x["exit_type"] for x in exit_dist}
        for ex in required_ex:
            if ex not in present:
                exit_dist.append({"exit_type": ex, "count": 0, "net_usd": 0.0, "avg_pips": 0.0, "pct_total": 0.0})

        # MFE/MAE analytics
        wins = t[t["is_win"]]
        losses = t[~t["is_win"]]
        win_capture = wins["pips"] / wins["mfe_pips"].replace(0.0, np.nan)

        mfe_mae = {
            "winners": {
                "avg_mfe_pips": float(wins["mfe_pips"].mean()) if len(wins) else 0.0,
                "avg_mae_pips": float(wins["mae_pips"].mean()) if len(wins) else 0.0,
                "capture_ratio_mean": float(win_capture.replace([np.inf, -np.inf], np.nan).mean()) if len(wins) else 0.0,
                "capture_ratio_median": float(win_capture.replace([np.inf, -np.inf], np.nan).median()) if len(wins) else 0.0,
                "time_to_peak_mfe_minutes_mean": float(wins["time_to_mfe_minutes"].mean()) if len(wins) else 0.0,
                "time_to_peak_mfe_minutes_median": float(wins["time_to_mfe_minutes"].median()) if len(wins) else 0.0,
            },
            "losers": {
                "avg_mfe_pips": float(losses["mfe_pips"].mean()) if len(losses) else 0.0,
                "avg_mae_pips": float(losses["mae_pips"].mean()) if len(losses) else 0.0,
                "pct_losers_touched_plus1pip": float((losses["mfe_pips"] >= 1.0).mean() * 100.0) if len(losses) else 0.0,
                "pct_losers_touched_plus3pip": float((losses["mfe_pips"] >= 3.0).mean() * 100.0) if len(losses) else 0.0,
                "avg_time_to_plus1pip_minutes": float(losses.loc[losses["mfe_pips"] >= 1.0, "time_to_mfe_minutes"].mean()) if len(losses) else 0.0,
            },
        }

        # day/month/ib class
        day_tbl = []
        for d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
            g = t[t["dow"] == d]
            s = grp_stats(g)
            day_tbl.append({"day": d, **s})

        month_tbl = []
        for m, g in t.groupby("month"):
            s = grp_stats(g)
            month_tbl.append({"month": m, **s})

        ib_tbl = []
        for cls, g in t.groupby("ib_classification"):
            s = grp_stats(g)
            # sessions count and avg range from session_state
            sess = [x for x in self.session_state.values() if x.get("ib_class") == cls]
            ranges = [float(x["ib"]["IB_Range"]/PIP_SIZE) for x in sess if x.get("ib")]
            ib_tbl.append({
                "classification": cls,
                "sessions": int(len(sess)),
                **s,
                "avg_ib_range_pips": float(np.mean(ranges)) if ranges else 0.0,
            })

        # near miss log: cap 200 evenly sampled and add next 30m move
        nm = self.diag["near_miss_candidates"]
        near_out = []
        if nm:
            if len(nm) > 200:
                idxs = np.linspace(0, len(nm) - 1, 200).astype(int)
                sample = [nm[i] for i in idxs]
            else:
                sample = nm
            for x in sample:
                i = x.get("entry_index", -1)
                if 0 <= i < len(self.df):
                    j0 = i + 1
                    j1 = min(len(self.df), i + 31)
                    seg = self.df.iloc[j0:j1]
                    ep = float(x["entry_price"])
                    if x["direction"] == "long":
                        move = float((seg["high"].max() - ep) / PIP_SIZE) if not seg.empty else np.nan
                    else:
                        move = float((ep - seg["low"].min()) / PIP_SIZE) if not seg.empty else np.nan
                else:
                    move = np.nan
                near_out.append({
                    "datetime": x["datetime"],
                    "level_name": x["level_name"],
                    "direction": x["direction"],
                    "zone_distance_pips": float(x["zone_distance_pips"]),
                    "which_confirmations_checked": x["which_confirmations_checked"],
                    "which_failed": x["which_failed"],
                    "price_movement_next_30min_pips": move,
                })

        # session summaries and hour table
        sess_rows = []
        for d, s in sorted(self.session_state.items()):
            sess_rows.append({
                "date": d,
                "signals_generated": int(s["signals_generated"]),
                "entries_taken": int(s["entries_taken"]),
                "trades_won": int(s["wins"]),
                "trades_lost": int(s["losses"]),
                "session_pnl": float(s["session_pnl"]),
                "confluence_scores_of_entries": list(s["entry_scores"]),
            })

        hour_rows = []
        for h in range(0, 9):
            g = t[t["hour_utc"] == h]
            hour_rows.append({
                "hour_utc": h,
                "entries": int(len(g)),
                "wins": int(g["is_win"].sum()) if len(g) else 0,
                "losses": int((~g["is_win"]).sum()) if len(g) else 0,
                "win_rate": float(g["is_win"].mean() * 100.0) if len(g) else 0.0,
                "avg_pips": float(g["pips"].mean()) if len(g) else 0.0,
                "net_pnl": float(g["pnl_usd"].sum()) if len(g) else 0.0,
                "avg_confluence_score": float(g["confluence_count"].mean()) if len(g) else 0.0,
            })

        # triage flags
        level_df = pd.DataFrame(level_perf)
        all_levels_negative = bool((level_df["net_usd"] <= 0).all()) if not level_df.empty else True
        all_days_negative = bool(all(x["net_usd"] <= 0 for x in day_tbl))

        if pf < 0.80 and all_days_negative and all_levels_negative:
            triage = "NO EDGE DETECTED. Session-derived levels do not provide mean reversion edge on USDJPY Tokyo hours with this entry/exit structure."
        elif 0.80 <= pf < 1.0 and any((x["pf"] > 1.2 and x["trades"] >= 15) for x in level_perf):
            good_levels = [x["level"] + "(" + x["direction"] + ")" for x in level_perf if x["pf"] > 1.2 and x["trades"] >= 15]
            triage = f"MARGINAL EDGE. Specific levels show promise. Recommend targeted follow-up on: {good_levels}"
        elif pf >= 1.0:
            triage = "POSITIVE EXPECTANCY DETECTED. Proceed to optimization round."
        else:
            triage = "NO EDGE DETECTED."

        best_level = None
        lvl_candidates = [x for x in level_perf if x["trades"] >= 10]
        if lvl_candidates:
            best_level = sorted(lvl_candidates, key=lambda x: x["pf"], reverse=True)[0]

        best_phase = None
        ph_candidates = [x for x in phase_perf if x["trades"] >= 10]
        if ph_candidates:
            best_phase = sorted(ph_candidates, key=lambda x: x["pf"], reverse=True)[0]

        best_signal = None
        sig_candidates = [x for x in signal_value if x["trades"] >= 10]
        if sig_candidates:
            best_signal = sorted(sig_candidates, key=lambda x: x["pf"], reverse=True)[0]

        best_ib = None
        ib_candidates = [x for x in ib_tbl if x["trades"] >= 10]
        if ib_candidates:
            best_ib = sorted(ib_candidates, key=lambda x: x["pf"], reverse=True)[0]

        max_wins = 0
        max_losses = 0
        cur_wins = 0
        cur_losses = 0
        for is_win in t["is_win"].tolist():
            if bool(is_win):
                cur_wins += 1
                cur_losses = 0
            else:
                cur_losses += 1
                cur_wins = 0
            if cur_wins > max_wins:
                max_wins = cur_wins
            if cur_losses > max_losses:
                max_losses = cur_losses

        report = {
            "run_label": self.run["label"],
            "input_csv": self.run["input_csv"],
            "summary": {
                "total_trades": int(len(t)),
                "win_rate_pct": wr,
                "profit_factor": float(pf),
                "net_profit_usd": net,
                "net_profit_pips": float(t["pips"].sum()),
                "average_win_pips": float(t.loc[t["is_win"], "pips"].mean()) if (t["is_win"]).any() else 0.0,
                "average_loss_pips": float(t.loc[~t["is_win"], "pips"].mean()) if (~t["is_win"]).any() else 0.0,
                "largest_single_win_usd": float(t["pnl_usd"].max()),
                "largest_single_loss_usd": float(t["pnl_usd"].min()),
                "max_drawdown_usd": float(dd.max()) if len(dd) else 0.0,
                "max_drawdown_pct": float(dd_pct.max()) if len(dd_pct) else 0.0,
                "max_consecutive_wins": int(max_wins),
                "max_consecutive_losses": int(max_losses),
                "average_trade_duration_minutes": float(t["duration_min"].mean()),
                "sharpe_ratio": sharpe,
                "calmar_ratio": calmar,
                "starting_equity": self.start_equity,
                "ending_equity": float(equity_after[-1]) if len(equity_after) else self.start_equity,
            },
            "long_short_breakdown": {
                "long": grp_stats(t[t["direction"] == "long"]),
                "short": grp_stats(t[t["direction"] == "short"]),
            },
            "day_of_week": day_tbl,
            "monthly": month_tbl,
            "equity_curve": t[["trade_id", "entry_datetime", "exit_datetime", "equity_after"]].to_dict("records"),
            "drawdown_curve": t[["trade_id", "drawdown_usd", "drawdown_pct"]].to_dict("records"),
            "diagnostics": {
                "signal_funnel": {
                    **{k: v for k, v in self.diag.items() if k not in ["near_miss_candidates", "zone_touches", "signal_fires"]},
                    "zone_touches": self.diag["zone_touches"],
                    "signal_fires": self.diag["signal_fires"],
                },
                "session_summary_table": sess_rows,
                "hour_of_session_breakdown": hour_rows,
                "near_miss_log": near_out,
                "level_performance": level_perf,
                "phase_performance": phase_perf,
                "confirmation_signal_value": signal_value,
                "confluence_count_performance": conf_count_perf,
                "exit_distribution": exit_dist,
                "mfe_mae_analysis": mfe_mae,
                "ib_classification_breakdown": ib_tbl,
                "triage_flag": triage,
                "best_level_min10": best_level,
                "best_phase_min10": best_phase,
                "best_confirmation_min10": best_signal,
                "best_ib_class_min10": best_ib,
            },
            "trades": t.to_dict("records"),
        }
        return report


def write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def main():
    args = parse_args()
    cfg = json.loads(Path(args.config).read_text())

    out_dir = Path(cfg["outputs"]["base_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    scaling_rows = []

    for run in cfg["run_sequence"]:
        bt = Backtester(cfg, run)
        rep = bt.run_backtest()
        label = run["label"]
        all_results[label] = rep

        rp = out_dir / f"tokyo_trrs_v1_{label}_report.json"
        tp = out_dir / f"tokyo_trrs_v1_{label}_trade_log.csv"
        ep = out_dir / f"tokyo_trrs_v1_{label}_equity.csv"
        write_json(rp, rep)
        pd.DataFrame(rep["trades"]).to_csv(tp, index=False)
        pd.DataFrame(rep["equity_curve"]).to_csv(ep, index=False)

        # $/month based on dataset span
        src = load_m1(run["input_csv"])
        months = max(1e-9, (src["time"].max() - src["time"].min()).total_seconds() / 86400.0 / 30.4375)

        s = rep["summary"]
        scaling_rows.append({
            "dataset": label,
            "trades": s["total_trades"],
            "wr_pct": s["win_rate_pct"],
            "pf": s["profit_factor"],
            "net_usd": s["net_profit_usd"],
            "maxdd_usd": s["max_drawdown_usd"],
            "maxdd_pct": s["max_drawdown_pct"],
            "usd_per_month": s["net_profit_usd"] / months,
            "avg_trade_pips": s["net_profit_pips"] / max(1, s["total_trades"]),
        })

    # master outputs
    results = {
        "config_path": args.config,
        "scaling_table": scaling_rows,
        "runs": {k: {"summary": v["summary"], "long_short_breakdown": v["long_short_breakdown"], "day_of_week": v["day_of_week"], "monthly": v["monthly"]} for k, v in all_results.items()},
    }

    # 1000k detailed files
    r1000 = all_results.get("1000k")
    if r1000 is not None:
        write_json(out_dir / "tokyo_trrs_v1_diagnostics.json", r1000["diagnostics"])
        write_json(out_dir / "tokyo_trrs_v1_advanced_analytics.json", {
            "mfe_mae": r1000["diagnostics"]["mfe_mae_analysis"],
            "day_of_week": r1000["day_of_week"],
            "monthly": r1000["monthly"],
            "level_performance": r1000["diagnostics"]["level_performance"],
            "phase_performance": r1000["diagnostics"]["phase_performance"],
            "confirmation_signal_value": r1000["diagnostics"]["confirmation_signal_value"],
            "ib_classification_breakdown": r1000["diagnostics"]["ib_classification_breakdown"],
            "triage_flag": r1000["diagnostics"]["triage_flag"],
        })
        pd.DataFrame(r1000["trades"]).to_csv(out_dir / "tokyo_trrs_v1_trade_log.csv", index=False)

    write_json(out_dir / "tokyo_trrs_v1_results.json", results)

    # Print compact scaling rows
    for r in scaling_rows:
        print(
            f"[{r['dataset']}] trades={r['trades']} wr={r['wr_pct']:.2f}% pf={r['pf']:.3f} "
            f"net={r['net_usd']:.2f} maxdd={r['maxdd_usd']:.2f} maxdd%={r['maxdd_pct']:.2f} $/mo={r['usd_per_month']:.2f}"
        )


if __name__ == "__main__":
    main()
