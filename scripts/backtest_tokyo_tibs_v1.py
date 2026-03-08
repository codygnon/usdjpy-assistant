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
    entry_type: int
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
    tp1_price: float
    tp2_price: Optional[float]
    sl_pips: float
    tp1_pips: float
    tp2_pips: Optional[float]
    units: int
    trailing_enabled: bool
    trailing_activate_pips: float
    trail_dist_pips: float
    late_profile: bool
    breakout_candle_body_pct: float
    partial_enabled: bool
    partial_ratio: float
    units_open: int = 0
    partial_executed: bool = False
    structural_stop_triggered: bool = False
    time_profit_protect_triggered: bool = False
    realized_partial_usd: float = 0.0
    realized_partial_pips_weighted: float = 0.0
    overlap_tier1: bool = False
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
        self.late_start = hhmm_to_min(sess["late_session_start_utc"])
        self.wind_start = hhmm_to_min(sess["wind_down_start_utc"])
        self.force_close_min = hhmm_to_min(sess["force_close_utc"])
        self.allowed_days = set(sess["allowed_days"])

        self.ex_normal = cfg["exits"]["default"]
        self.ex_type2 = cfg["exits"]["type2_retest"]
        self.ex_type3 = cfg["exits"]["type3_failed_breakout"]
        self.ex_late = cfg["exits"]["late_session"]

        tm = cfg["trade_management"]
        self.min_gap = int(tm["min_time_between_entries_minutes"])
        self.reentry_sl_same_level = int(tm["no_reentry_same_direction_type1_after_sl_minutes"])
        self.reentry_structural = int(tm["no_reentry_same_direction_after_structural_stop_minutes"])
        self.reentry_type2_after_breakout = int(tm["type2_retest_delay_after_breakout_minutes"])
        self.pause_losses = int(tm["consecutive_loss_pause"]["loss_count"])
        self.pause_mins = int(tm["consecutive_loss_pause"]["pause_minutes"])
        self.breakout_kill_pips = float(tm["breakout_session_kill_pips"])
        self.max_session_total = int(tm["max_entries_per_session_total"])
        self.max_per_type = {int(k): int(v) for k, v in tm["max_entries_per_type"].items()}

        self.positions: List[Position] = []
        self.closed: List[dict] = []

        self.trade_id = 0
        self.last_entry_ts: Optional[pd.Timestamp] = None
        self.session_state: Dict[str, dict] = {}

        self.diag = {
            "total_m1_candles": int(len(self.df)),
            "candles_inside_session_window": 0,
            "candles_tradeable_phase": 0,
            "sessions_total": 0,
            "sessions_skipped_ib_dead": 0,
            "sessions_skipped_ib_volatile": 0,
            "ib_breakout_upside": 0,
            "ib_breakout_upside_body_ok": 0,
            "ib_breakout_downside": 0,
            "ib_breakout_downside_body_ok": 0,
            "type1_long_entries": 0,
            "type1_short_entries": 0,
            "type2_long_setups": 0,
            "type2_short_setups": 0,
            "type2_long_entries": 0,
            "type2_short_entries": 0,
            "type3_long_setups": 0,
            "type3_short_setups": 0,
            "type3_long_entries": 0,
            "type3_short_entries": 0,
            "type4_long_entries": 0,
            "type4_short_entries": 0,
            "entries_taken": 0,
            "blocked_max_concurrent": 0,
            "blocked_cooldown": 0,
            "blocked_session_limits": 0,
            "blocked_whipsaw": 0,
            "blocked_conflicting_signal": 0,
            "blocked_breakout_detection": 0,
            "blocked_ib_classification": 0,
            "blocked_late_session_requirements": 0,
            "blocked_pause": 0,
            "blocked_min_confirm": 0,
            "near_miss_candidates": [],
            "breakout_session_kill_count": 0,
            "sessions_whipsaw": 0,
        }

    def _add_indicators(self):
        # M15 RSI
        self.m15["rsi14"] = rsi(self.m15["close"], 14)
        self.m15["rsi14_l1"] = self.m15["rsi14"].shift(1)
        self.m15["rsi14_l2"] = self.m15["rsi14"].shift(2)
        self.m15["rsi14_l3"] = self.m15["rsi14"].shift(3)

        # M5 rejection features + body ratio + rolling volume baseline
        rng = (self.m5["high"] - self.m5["low"]).replace(0.0, np.nan)
        upper_close_th = self.m5["low"] + 0.67 * rng
        lower_close_th = self.m5["high"] - 0.67 * rng
        lower_wick = self.m5[["open", "close"]].min(axis=1) - self.m5["low"]
        upper_wick = self.m5["high"] - self.m5[["open", "close"]].max(axis=1)
        self.m5["bull_rej_shape"] = (self.m5["close"] > upper_close_th) & (lower_wick >= 0.60 * rng)
        self.m5["bear_rej_shape"] = (self.m5["close"] < lower_close_th) & (upper_wick >= 0.60 * rng)
        self.m5["body_ratio"] = (self.m5["close"] - self.m5["open"]).abs() / rng
        if "volume" in self.m5.columns:
            self.m5["vol_avg12"] = self.m5["volume"].rolling(12, min_periods=1).mean().shift(1)
        else:
            self.m5["vol_avg12"] = np.nan

        # merge-asof to m1
        m5_cols = ["time", "open", "high", "low", "close", "bull_rej_shape", "bear_rej_shape", "body_ratio", "volume", "vol_avg12"] if "volume" in self.m5.columns else ["time", "open", "high", "low", "close", "bull_rej_shape", "bear_rej_shape", "body_ratio", "vol_avg12"]
        m5r = self.m5[m5_cols].copy().rename(columns={
            "open":"m5_open","high":"m5_high","low":"m5_low","close":"m5_close",
            "bull_rej_shape":"m5_bull_rej_shape","bear_rej_shape":"m5_bear_rej_shape",
            "body_ratio":"m5_body_ratio",
            "volume":"m5_volume",
            "vol_avg12":"m5_vol_avg12"
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
            return "IB_Formation"
        if 60 <= minute_of_day < 65:
            return "IB_Assessment"
        if 65 <= minute_of_day < 150:
            return "Active_Trading"
        if 150 <= minute_of_day < 210:
            return "Lunch_Block"
        if 210 <= minute_of_day < 390:
            return "Afternoon_Trading"
        if 390 <= minute_of_day < 480:
            return "Late_Session"
        if 480 <= minute_of_day < 525:
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
            "dir_type1_cooldown_until": {"long": None, "short": None},
            "dir_struct_cooldown_until": {"long": None, "short": None},
            "consec_losses": 0,
            "pause_until": None,
            "ib_ready": False,
            "ib_class": "unknown",
            "ib": None,
            "pts": None,
            "first_m5_above_ib": False,
            "first_m5_below_ib": False,
            "whipsaw": False,
            "breakout_state": {
                "long": {"occurred": False, "entry_time": None, "max_move_pips": 0.0, "entries": 0, "retest_entries": 0, "failed_entries": 0, "pts_entries": 0},
                "short": {"occurred": False, "entry_time": None, "max_move_pips": 0.0, "entries": 0, "retest_entries": 0, "failed_entries": 0, "pts_entries": 0},
            },
            "phase_entries": Counter(),
            "type_entries": Counter(),
            "signals_generated": 0,
            "entries_taken": 0,
            "wins": 0,
            "losses": 0,
            "session_pnl": 0.0,
            "entry_scores": [],
            "first_breakout_direction": None,
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
            "IB_Ext_1_High": float(hi + rng),
            "IB_Ext_1_5_High": float(hi + 1.5 * rng),
            "IB_Ext_1_Low": float(lo - rng),
            "IB_Ext_1_5_Low": float(lo - 1.5 * rng),
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

    def _breakout_conf_signals(self, direction: str, row: pd.Series, sstate: dict) -> List[str]:
        sigs: List[str] = []
        px = float(row["m5_close"])
        vwap = row.get("vwap", np.nan)
        rsi_now = row.get("rsi14", np.nan)
        rsi_prev = row.get("rsi14_l1", np.nan)
        pts = sstate.get("pts")

        if not pd.isna(vwap):
            if direction == "long" and px > float(vwap):
                sigs.append("VWAP_support")
            if direction == "short" and px < float(vwap):
                sigs.append("VWAP_support")

        if not pd.isna(rsi_now) and not pd.isna(rsi_prev):
            if direction == "long" and float(rsi_now) > 50 and float(rsi_now) > float(rsi_prev):
                sigs.append("RSI_momentum")
            if direction == "short" and float(rsi_now) < 50 and float(rsi_now) < float(rsi_prev):
                sigs.append("RSI_momentum")

        if pts:
            if direction == "long" and (px > float(pts["PTS_High"]) or px > float(pts["PTS_Mid"])):
                sigs.append("PTS_alignment")
            if direction == "short" and (px < float(pts["PTS_Low"]) or px < float(pts["PTS_Mid"])):
                sigs.append("PTS_alignment")

        vol = row.get("m5_volume", np.nan)
        vol_avg = row.get("m5_vol_avg12", np.nan)
        if not pd.isna(vol) and not pd.isna(vol_avg) and float(vol_avg) > 0 and float(vol) > 1.5 * float(vol_avg):
            sigs.append("Volume_surge")
        return sigs

    def _phase_exit_profile(self, phase: str, entry_type: int):
        if phase == "Late_Session":
            return {
                "sl_pips": float(self.ex_late["sl_pips"]),
                "tp1_pips": float(self.ex_late["tp_pips"]),
                "tp2_pips": None,
                "trail_activate": float(self.ex_late["trail_activate_pips"]),
                "trail_dist": float(self.ex_late["trail_distance_pips"]),
                "partial_enabled": False,
                "partial_ratio": 0.0,
            }
        if entry_type == 2:
            return {
                "sl_pips": float(self.ex_type2["sl_pips"]),
                "tp1_pips": float(self.ex_type2["tp_pips"]),
                "tp2_pips": None,
                "trail_activate": float(self.ex_type2["trail_activate_pips"]),
                "trail_dist": float(self.ex_type2["trail_distance_pips"]),
                "partial_enabled": False,
                "partial_ratio": 0.0,
            }
        if entry_type == 3:
            return {
                "sl_pips": float(self.ex_type3["sl_pips"]),
                "tp1_pips": None,
                "tp2_pips": None,
                "trail_activate": float(self.ex_type3["trail_activate_pips"]),
                "trail_dist": float(self.ex_type3["trail_distance_pips"]),
                "partial_enabled": False,
                "partial_ratio": 0.0,
            }
        return {
            "sl_pips": float(self.ex_normal["sl_pips"]),
            "tp1_pips": None,
            "tp2_pips": None,
            "trail_activate": float(self.ex_normal["trail_activate_pips"]),
            "trail_dist": float(self.ex_normal["trail_distance_pips"]),
            "partial_enabled": True,
            "partial_ratio": float(self.ex_normal["partial_close_ratio"]),
        }

    def _ib_size_mult(self, ib_class: str) -> float:
        return {
            "dead": 0.75,
            "tight": 1.0,
            "normal": 1.0,
            "wide": 0.60,
            "volatile": 0.50,
            "unknown": 1.0,
        }.get(ib_class, 1.0)

    def _open_trade(
        self,
        row: pd.Series,
        direction: str,
        entry_type: int,
        level_name: str,
        level_tier: int,
        level_price: float,
        confs: List[str],
        conf_score: int,
        body_pct: float,
        sstate: dict,
        phase: str,
    ):
        spread_pips = compute_spread_pips(int(row.name), row["time"], self.cfg["account"])
        bid, ask = get_bid_ask(float(row["close"]), spread_pips)
        entry = ask if direction == "long" else bid

        profile = self._phase_exit_profile(phase, entry_type)
        sl_pips = float(profile["sl_pips"])
        tp1_pips = profile["tp1_pips"]
        tp2_pips = profile["tp2_pips"]
        tr_act = float(profile["trail_activate"])
        tr_dist = float(profile["trail_dist"])
        partial_enabled = bool(profile["partial_enabled"])
        partial_ratio = float(profile["partial_ratio"])

        ib = sstate.get("ib")
        if entry_type in (1, 4) and phase != "Late_Session" and ib is not None:
            if direction == "long":
                tp1_calc = max(5.0, min(20.0, (float(ib["IB_Ext_1_High"]) - entry) / PIP_SIZE))
                tp2_calc = max(tp1_calc + 1.0, (float(ib["IB_Ext_1_5_High"]) - entry) / PIP_SIZE)
            else:
                tp1_calc = max(5.0, min(20.0, (entry - float(ib["IB_Ext_1_Low"])) / PIP_SIZE))
                tp2_calc = max(tp1_calc + 1.0, (entry - float(ib["IB_Ext_1_5_Low"])) / PIP_SIZE)
            tp1_pips = 7.0 if tp1_calc < 5.0 else tp1_calc
            tp2_pips = tp2_calc
        elif entry_type == 3 and ib is not None:
            if direction == "long":
                tpp = (float(ib["IB_Mid"]) - entry) / PIP_SIZE
            else:
                tpp = (entry - float(ib["IB_Mid"])) / PIP_SIZE
            tp1_pips = max(5.0, min(15.0, tpp if tpp > 0 else 7.0))

        sl = entry - sl_pips * PIP_SIZE if direction == "long" else entry + sl_pips * PIP_SIZE
        tp1 = None if tp1_pips is None else (entry + float(tp1_pips) * PIP_SIZE if direction == "long" else entry - float(tp1_pips) * PIP_SIZE)
        tp2 = None if tp2_pips is None else (entry + float(tp2_pips) * PIP_SIZE if direction == "long" else entry - float(tp2_pips) * PIP_SIZE)

        risk_pct = self.base_risk * self._ib_size_mult(sstate.get("ib_class", "unknown"))
        if entry_type == 2:
            risk_pct *= 1.25
        if entry_type == 3:
            risk_pct *= 0.75
        if phase == "Late_Session":
            risk_pct *= 0.50
        if conf_score >= 3:
            risk_pct *= 1.20
        units = math.floor((self.equity * risk_pct) / (sl_pips * (PIP_SIZE / max(1e-9, entry))))
        units = int(max(0, min(self.max_units, units)))
        if units < 1:
            self.diag["blocked_session_limits"] += 1
            return False
        if not self._margin_ok(units):
            self.diag["blocked_max_concurrent"] += 1
            return False

        self.trade_id += 1
        units_open = int(units)
        self.positions.append(Position(
            trade_id=self.trade_id,
            direction=direction,
            entry_time=row["time"],
            entry_price=entry,
            entry_type=entry_type,
            entry_level_name=level_name,
            entry_level_tier=level_tier,
            entry_level_price=level_price,
            zone_distance_at_entry_pips=abs(float(row["close"]) - level_price) / PIP_SIZE,
            confirmation_signals_present=confs,
            confluence_count=conf_score,
            ib_classification=sstate.get("ib_class", "unknown"),
            session_phase_at_entry=phase,
            rsi_at_entry=float(row.get("rsi14", np.nan)) if not pd.isna(row.get("rsi14", np.nan)) else np.nan,
            vwap_at_entry=float(row.get("vwap", np.nan)) if not pd.isna(row.get("vwap", np.nan)) else np.nan,
            sl_price=sl,
            tp1_price=tp1 if tp1 is not None else np.nan,
            tp2_price=tp2,
            sl_pips=sl_pips,
            tp1_pips=float(tp1_pips) if tp1_pips is not None else np.nan,
            tp2_pips=float(tp2_pips) if tp2_pips is not None else None,
            units=units,
            units_open=units_open,
            trailing_enabled=True,
            trailing_activate_pips=tr_act,
            trail_dist_pips=tr_dist,
            late_profile=(phase == "Late_Session"),
            breakout_candle_body_pct=float(body_pct),
            partial_enabled=partial_enabled,
            partial_ratio=partial_ratio,
        ))
        self.last_entry_ts = row["time"]
        sstate["entries_taken"] += 1
        sstate["phase_entries"][phase] += 1
        sstate["type_entries"][entry_type] += 1
        sstate["entry_scores"].append(conf_score)
        self.diag["entries_taken"] += 1
        if entry_type == 1 and direction == "long":
            self.diag["type1_long_entries"] += 1
        if entry_type == 1 and direction == "short":
            self.diag["type1_short_entries"] += 1
        if entry_type == 2 and direction == "long":
            self.diag["type2_long_entries"] += 1
        if entry_type == 2 and direction == "short":
            self.diag["type2_short_entries"] += 1
        if entry_type == 3 and direction == "long":
            self.diag["type3_long_entries"] += 1
        if entry_type == 3 and direction == "short":
            self.diag["type3_short_entries"] += 1
        if entry_type == 4 and direction == "long":
            self.diag["type4_long_entries"] += 1
        if entry_type == 4 and direction == "short":
            self.diag["type4_short_entries"] += 1
        return True

    def _close_trade(self, pos: Position, ts: pd.Timestamp, exit_price: float, reason: str, sstate: dict):
        pips_final = (exit_price - pos.entry_price) / PIP_SIZE if pos.direction == "long" else (pos.entry_price - exit_price) / PIP_SIZE
        usd_final = pips_final * pos.units_open * (PIP_SIZE / max(1e-9, exit_price))
        usd = float(pos.realized_partial_usd + usd_final)
        weighted_pips = float(pos.realized_partial_pips_weighted + pips_final * pos.units_open)
        pips = weighted_pips / max(1, pos.units)
        self.equity += usd

        duration = (ts - pos.entry_time).total_seconds() / 60.0
        time_to_mfe = (pos.mfe_time - pos.entry_time).total_seconds() / 60.0 if pos.mfe_time is not None else np.nan

        self.closed.append({
            "trade_id": pos.trade_id,
            "entry_datetime": str(pos.entry_time),
            "exit_datetime": str(ts),
            "direction": pos.direction,
            "entry_type": pos.entry_type,
            "entry_price": pos.entry_price,
            "exit_price": float(exit_price),
            "sl_price": pos.sl_price,
            "tp1_price": pos.tp1_price,
            "tp2_price": pos.tp2_price,
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
            "breakout_candle_body_pct": pos.breakout_candle_body_pct,
            "mfe_pips": float(pos.max_fav_pips),
            "mae_pips": float(pos.max_adv_pips),
            "time_to_mfe_minutes": float(time_to_mfe) if not pd.isna(time_to_mfe) else np.nan,
            "trade_duration_minutes": float(duration),
            "day_of_week": pd.Timestamp(pos.entry_time).day_name(),
            "hour_utc": int(pd.Timestamp(pos.entry_time).hour),
            "partial_close_executed": "y" if pos.partial_executed else "n",
            "structural_stop_triggered": "y" if pos.structural_stop_triggered else "n",
        })

        sstate["session_pnl"] += float(usd)
        if usd > 0:
            sstate["wins"] += 1
            sstate["consec_losses"] = 0
        else:
            sstate["losses"] += 1
            sstate["consec_losses"] += 1
            if reason.startswith("SL_Hit") and pos.entry_type == 1:
                sstate["dir_type1_cooldown_until"][pos.direction] = ts + pd.Timedelta(minutes=self.reentry_sl_same_level)
            if reason == "Structural_Stop":
                sstate["dir_struct_cooldown_until"][pos.direction] = ts + pd.Timedelta(minutes=self.reentry_structural)
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

        is_new_m5_close = (not pd.isna(row.get("m5_time", np.nan))) and (row["time"] == row.get("m5_time"))
        m5_close = row.get("m5_close", np.nan)
        ib = sstate.get("ib")
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

            # dynamic time-based protection
            age_min = (row["time"] - pos.entry_time).total_seconds() / 60.0
            if age_min >= 60 and cur > 0:
                pos.trail_dist_pips = min(pos.trail_dist_pips, 2.0)
                pos.time_profit_protect_triggered = True
            if age_min >= 90 and cur > 0 and (not pos.partial_executed):
                # close 50% and tighten on remainder
                close_units = max(1, int(pos.units_open * 0.5))
                p_part = cur
                usd_part = p_part * close_units * (PIP_SIZE / max(1e-9, bid_c if pos.direction == "long" else ask_c))
                pos.realized_partial_usd += float(usd_part)
                pos.realized_partial_pips_weighted += float(p_part * close_units)
                pos.units_open -= close_units
                pos.partial_executed = True
                pos.time_profit_protect_triggered = True
                pos.trail_dist_pips = min(pos.trail_dist_pips, 2.0)

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

            # wind-down handling
            if phase == "Wind_Down":
                if cur > 2:
                    to_close.append((pos, bid_c if pos.direction=="long" else ask_c, "Wind_Down_Close_Profit"))
                    continue
                if cur <= -4:
                    to_close.append((pos, bid_c if pos.direction=="long" else ask_c, "Wind_Down_Close_Large_Loss"))
                    continue
                # hold small loss until force close; tiny profit tight trail
                if 0 <= cur <= 2:
                    pos.moved_to_trail = True
                    pos.trail_dist_pips = min(pos.trail_dist_pips, 1.0)

            # force close
            if row["minute_of_day"] >= self.force_close_min:
                to_close.append((pos, bid_c if pos.direction=="long" else ask_c, "Force_Close_08_45"))
                continue

            # structural stop for breakout types when price returns inside IB
            if ib is not None and pos.entry_type in (1, 4) and is_new_m5_close and not pd.isna(m5_close):
                if pos.direction == "long" and float(m5_close) < float(ib["IB_High"]):
                    pos.structural_stop_triggered = True
                    to_close.append((pos, bid_c if pos.direction=="long" else ask_c, "Structural_Stop"))
                    continue
                if pos.direction == "short" and float(m5_close) > float(ib["IB_Low"]):
                    pos.structural_stop_triggered = True
                    to_close.append((pos, bid_c if pos.direction=="long" else ask_c, "Structural_Stop"))
                    continue

            # TP/SL/trail checks
            if pos.direction == "long":
                if bid_l <= pos.sl_price:
                    reason = "SL_Hit_PipBased"
                    to_close.append((pos, pos.sl_price, reason))
                    continue
                if (not pos.partial_executed) and (not pd.isna(pos.tp1_price)) and bid_h >= float(pos.tp1_price):
                    if pos.partial_enabled:
                        close_units = max(1, int(pos.units * pos.partial_ratio))
                        close_units = min(close_units, pos.units_open)
                        p_part = (float(pos.tp1_price) - pos.entry_price) / PIP_SIZE
                        usd_part = p_part * close_units * (PIP_SIZE / max(1e-9, float(pos.tp1_price)))
                        pos.realized_partial_usd += float(usd_part)
                        pos.realized_partial_pips_weighted += float(p_part * close_units)
                        pos.units_open -= close_units
                        pos.partial_executed = True
                        pos.moved_to_trail = True
                        if pos.tp2_price is not None and not pd.isna(pos.tp2_price):
                            pass
                        else:
                            # no runner target: close entire
                            to_close.append((pos, float(pos.tp1_price), "Full_TP"))
                            continue
                    else:
                        to_close.append((pos, float(pos.tp1_price), "Full_TP"))
                        continue
                if pos.partial_executed and pos.tp2_price is not None and bid_h >= float(pos.tp2_price):
                    to_close.append((pos, float(pos.tp2_price), "TP1_partial_plus_TP2_final"))
                    continue
                if pos.moved_to_trail and pos.trail_price is not None and bid_l <= pos.trail_price:
                    reason = "TP1_partial_plus_trail_final" if pos.partial_executed else "Trailing_Stop"
                    to_close.append((pos, pos.trail_price, reason))
                    continue
            else:
                if ask_h >= pos.sl_price:
                    reason = "SL_Hit_PipBased"
                    to_close.append((pos, pos.sl_price, reason))
                    continue
                if (not pos.partial_executed) and (not pd.isna(pos.tp1_price)) and ask_l <= float(pos.tp1_price):
                    if pos.partial_enabled:
                        close_units = max(1, int(pos.units * pos.partial_ratio))
                        close_units = min(close_units, pos.units_open)
                        p_part = (pos.entry_price - float(pos.tp1_price)) / PIP_SIZE
                        usd_part = p_part * close_units * (PIP_SIZE / max(1e-9, float(pos.tp1_price)))
                        pos.realized_partial_usd += float(usd_part)
                        pos.realized_partial_pips_weighted += float(p_part * close_units)
                        pos.units_open -= close_units
                        pos.partial_executed = True
                        pos.moved_to_trail = True
                        if pos.tp2_price is not None and not pd.isna(pos.tp2_price):
                            pass
                        else:
                            to_close.append((pos, float(pos.tp1_price), "Full_TP"))
                            continue
                    else:
                        to_close.append((pos, float(pos.tp1_price), "Full_TP"))
                        continue
                if pos.partial_executed and pos.tp2_price is not None and ask_l <= float(pos.tp2_price):
                    to_close.append((pos, float(pos.tp2_price), "TP1_partial_plus_TP2_final"))
                    continue
                if pos.moved_to_trail and pos.trail_price is not None and ask_h >= pos.trail_price:
                    reason = "TP1_partial_plus_trail_final" if pos.partial_executed else "Trailing_Stop"
                    to_close.append((pos, pos.trail_price, reason))
                    continue

        if to_close:
            for pos, px, reason in to_close:
                if pos in self.positions:
                    self._close_trade(pos, row["time"], float(px), reason, sstate)
                    self.positions.remove(pos)

    def _level_invalidation_update(self, row: pd.Series, sstate: dict):
        ib = sstate.get("ib")
        if not ib:
            return
        c = row.get("m5_close", np.nan)
        if pd.isna(c):
            return
        if float(c) > float(ib["IB_High"]):
            sstate["first_m5_above_ib"] = True
        if float(c) < float(ib["IB_Low"]):
            sstate["first_m5_below_ib"] = True
        if sstate["first_m5_above_ib"] and sstate["first_m5_below_ib"] and (not sstate["whipsaw"]):
            sstate["whipsaw"] = True
            self.diag["sessions_whipsaw"] += 1

    def _entry_logic(self, row: pd.Series, sstate: dict, phase: str):
        m = int(row["minute_of_day"])
        day = str(row["day_name"])
        if day not in self.allowed_days:
            return
        if not in_window(m, self.sess_start, self.sess_end):
            return

        self.diag["candles_inside_session_window"] += 1

        # tradeable phase excludes ib formation/assessment, lunch, wind down
        if phase in {"IB_Formation", "IB_Assessment", "Lunch_Block", "Wind_Down", "Force_Close_Window", "Outside"}:
            return
        self.diag["candles_tradeable_phase"] += 1

        # ensure IB after 01:00 once
        if (not sstate["ib_ready"]) and m >= 60:
            ib = self._compute_ib_for_session(sstate["date"])
            sstate["ib"] = ib
            sstate["ib_ready"] = ib is not None
            sstate["ib_class"] = ib["classification"] if ib else "unknown"
            self.diag["sessions_total"] += 1
            if sstate["ib_class"] == "dead":
                self.diag["sessions_skipped_ib_dead"] += 1
            if sstate["ib_class"] == "volatile":
                self.diag["sessions_skipped_ib_volatile"] += 1

        # breakout session kill
        cur_mid = float((row["open"] + row["close"]) / 2.0)
        if abs(cur_mid - float(sstate["session_open_mid"])) / PIP_SIZE > self.breakout_kill_pips:
            if not sstate["breakout_killed"]:
                self.diag["breakout_session_kill_count"] += 1
            sstate["breakout_killed"] = True
        if sstate["breakout_killed"]:
            self.diag["blocked_breakout_detection"] += 1
            return

        # entries only at completed m5 close
        if pd.isna(row.get("m5_time", np.nan)) or row["time"] != row.get("m5_time"):
            return
        if row.get("m5_time", None) == sstate.get("last_signal_m5_time", None):
            return
        sstate["last_signal_m5_time"] = row.get("m5_time")

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

        # session limits
        if sstate["entries_taken"] >= self.max_session_total:
            self.diag["blocked_session_limits"] += 1
            return

        # whipsaw disables types 1 and 2
        if sstate.get("whipsaw", False):
            self.diag["blocked_whipsaw"] += 1

        # IB data required for types 1-3
        ib = sstate.get("ib")
        if ib is None:
            return

        ib_class = sstate.get("ib_class", "unknown")
        m5_close = float(row["m5_close"])
        m5_open = float(row["m5_open"])
        m5_high = float(row["m5_high"])
        m5_low = float(row["m5_low"])
        body_pct = float(row.get("m5_body_ratio", 0.0)) if not pd.isna(row.get("m5_body_ratio", np.nan)) else 0.0
        buf = 0.02
        if ib_class == "wide":
            buf = 0.04
        if phase == "Late_Session":
            buf = max(buf, 0.02)

        # precompute scores
        conf_long = self._breakout_conf_signals("long", row, sstate)
        conf_short = self._breakout_conf_signals("short", row, sstate)
        score_long = len(conf_long)
        score_short = len(conf_short)

        # cooldown by direction
        for d in ("long", "short"):
            cd = sstate["dir_type1_cooldown_until"].get(d)
            if cd is not None and row["time"] < cd:
                if d == "long":
                    score_long = -999
                else:
                    score_short = -999
            cd2 = sstate["dir_struct_cooldown_until"].get(d)
            if cd2 is not None and row["time"] < cd2:
                if d == "long":
                    score_long = -999
                else:
                    score_short = -999

        # conflicting signal against open thesis
        if self.positions:
            open_dirs = {p.direction for p in self.positions}
            if "long" in open_dirs and "short" not in open_dirs:
                score_short = -999
                self.diag["blocked_conflicting_signal"] += 1
            elif "short" in open_dirs and "long" not in open_dirs:
                score_long = -999
                self.diag["blocked_conflicting_signal"] += 1

        # --- Type 1 IB breakout ---
        took_trade = False
        allow_type1 = phase in {"Active_Trading", "Afternoon_Trading", "Late_Session"} and (not sstate.get("whipsaw", False))
        if ib_class in {"dead", "volatile"}:
            allow_type1 = False
            self.diag["blocked_ib_classification"] += 1
        if ib_class == "wide" and phase != "Late_Session":
            min_score_type1 = 2
        else:
            min_score_type1 = 0
        if phase == "Late_Session":
            min_score_type1 = max(min_score_type1, 2)
        if allow_type1 and body_pct >= 0.40:
            # long
            if m5_close > float(ib["IB_High"]) + buf:
                self.diag["ib_breakout_upside"] += 1
                self.diag["ib_breakout_upside_body_ok"] += 1
                sstate["signals_generated"] += 1
                if score_long >= min_score_type1 and sstate["type_entries"][1] < self.max_per_type.get(1, 999):
                    ok = self._open_trade(row, "long", 1, "IB_High", 1, float(ib["IB_High"]), conf_long, max(0, score_long), body_pct, sstate, phase)
                    if ok:
                        took_trade = True
                        sstate["breakout_state"]["long"]["occurred"] = True
                        sstate["breakout_state"]["long"]["entry_time"] = row["time"]
                        sstate["breakout_state"]["long"]["entries"] += 1
                        if sstate["first_breakout_direction"] is None:
                            sstate["first_breakout_direction"] = "LONG"
                elif score_long < min_score_type1:
                    self.diag["blocked_min_confirm"] += 1
                else:
                    self.diag["blocked_session_limits"] += 1

            # short
            if (not took_trade) and m5_close < float(ib["IB_Low"]) - buf:
                self.diag["ib_breakout_downside"] += 1
                self.diag["ib_breakout_downside_body_ok"] += 1
                sstate["signals_generated"] += 1
                if score_short >= min_score_type1 and sstate["type_entries"][1] < self.max_per_type.get(1, 999):
                    ok = self._open_trade(row, "short", 1, "IB_Low", 1, float(ib["IB_Low"]), conf_short, max(0, score_short), body_pct, sstate, phase)
                    if ok:
                        took_trade = True
                        sstate["breakout_state"]["short"]["occurred"] = True
                        sstate["breakout_state"]["short"]["entry_time"] = row["time"]
                        sstate["breakout_state"]["short"]["entries"] += 1
                        if sstate["first_breakout_direction"] is None:
                            sstate["first_breakout_direction"] = "SHORT"
                elif score_short < min_score_type1:
                    self.diag["blocked_min_confirm"] += 1
                else:
                    self.diag["blocked_session_limits"] += 1

        # track max moves from breakout edges for retest/failed-breakout prerequisites
        if sstate["breakout_state"]["long"]["occurred"]:
            mv = max(0.0, (m5_high - float(ib["IB_High"])) / PIP_SIZE)
            sstate["breakout_state"]["long"]["max_move_pips"] = max(sstate["breakout_state"]["long"]["max_move_pips"], mv)
        if sstate["breakout_state"]["short"]["occurred"]:
            mv = max(0.0, (float(ib["IB_Low"]) - m5_low) / PIP_SIZE)
            sstate["breakout_state"]["short"]["max_move_pips"] = max(sstate["breakout_state"]["short"]["max_move_pips"], mv)

        # --- Type 2 retest ---
        if (not took_trade) and phase in {"Active_Trading", "Afternoon_Trading"} and m >= 90 and ib_class in {"tight", "normal", "wide"} and (not sstate.get("whipsaw", False)):
            # long retest
            bl = sstate["breakout_state"]["long"]
            if bl["occurred"] and bl["max_move_pips"] >= 5 and bl["retest_entries"] < self.max_per_type.get(2, 999):
                self.diag["type2_long_setups"] += 1
                wait_ok = (row["time"] - bl["entry_time"]).total_seconds() >= self.reentry_type2_after_breakout * 60 if bl["entry_time"] is not None else False
                in_zone = (m5_low <= float(ib["IB_High"]) + 0.03) and (m5_high >= float(ib["IB_High"]) - 0.03)
                close_upper40 = m5_close >= (m5_low + 0.60 * max(1e-9, m5_high - m5_low))
                if wait_ok and in_zone and close_upper40 and m5_close > float(ib["IB_High"]):
                    ok = self._open_trade(row, "long", 2, "IB_High_Retest", 1, float(ib["IB_High"]), conf_long, max(0, score_long), body_pct, sstate, phase)
                    if ok:
                        took_trade = True
                        bl["retest_entries"] += 1

            # short retest
            bs = sstate["breakout_state"]["short"]
            if (not took_trade) and bs["occurred"] and bs["max_move_pips"] >= 5 and bs["retest_entries"] < self.max_per_type.get(2, 999):
                self.diag["type2_short_setups"] += 1
                wait_ok = (row["time"] - bs["entry_time"]).total_seconds() >= self.reentry_type2_after_breakout * 60 if bs["entry_time"] is not None else False
                in_zone = (m5_low <= float(ib["IB_Low"]) + 0.03) and (m5_high >= float(ib["IB_Low"]) - 0.03)
                close_lower40 = m5_close <= (m5_high - 0.60 * max(1e-9, m5_high - m5_low))
                if wait_ok and in_zone and close_lower40 and m5_close < float(ib["IB_Low"]):
                    ok = self._open_trade(row, "short", 2, "IB_Low_Retest", 1, float(ib["IB_Low"]), conf_short, max(0, score_short), body_pct, sstate, phase)
                    if ok:
                        took_trade = True
                        bs["retest_entries"] += 1

        # --- Type 3 failed breakout reversal ---
        if (not took_trade) and phase in {"Afternoon_Trading"} and m >= 120 and ib_class in {"tight", "normal", "wide"}:
            # long reversal after failed short breakout
            bs = sstate["breakout_state"]["short"]
            if bs["occurred"] and bs["max_move_pips"] >= 3 and bs["failed_entries"] < self.max_per_type.get(3, 999):
                self.diag["type3_long_setups"] += 1
                wait_ok = (row["time"] - bs["entry_time"]).total_seconds() >= 30 * 60 if bs["entry_time"] is not None else False
                rsi_now = row.get("rsi14", np.nan)
                if wait_ok and m5_close > float(ib["IB_Low"]) + 0.03 and (pd.isna(rsi_now) or float(rsi_now) > 40):
                    ok = self._open_trade(row, "long", 3, "IB_FailedBreak_Rev_Long", 1, float(ib["IB_Low"]), conf_long, max(0, score_long), body_pct, sstate, phase)
                    if ok:
                        took_trade = True
                        bs["failed_entries"] += 1

            # short reversal after failed long breakout
            bl = sstate["breakout_state"]["long"]
            if (not took_trade) and bl["occurred"] and bl["max_move_pips"] >= 3 and bl["failed_entries"] < self.max_per_type.get(3, 999):
                self.diag["type3_short_setups"] += 1
                wait_ok = (row["time"] - bl["entry_time"]).total_seconds() >= 30 * 60 if bl["entry_time"] is not None else False
                rsi_now = row.get("rsi14", np.nan)
                if wait_ok and m5_close < float(ib["IB_High"]) - 0.03 and (pd.isna(rsi_now) or float(rsi_now) < 60):
                    ok = self._open_trade(row, "short", 3, "IB_FailedBreak_Rev_Short", 1, float(ib["IB_High"]), conf_short, max(0, score_short), body_pct, sstate, phase)
                    if ok:
                        took_trade = True
                        bl["failed_entries"] += 1

        # --- Type 4 PTS breakout with IB alignment ---
        pts = sstate.get("pts")
        if (not took_trade) and pts and phase in {"Active_Trading", "Afternoon_Trading"}:
            if body_pct >= 0.40:
                bl = sstate["breakout_state"]["long"]
                if bl["occurred"] and bl["pts_entries"] < self.max_per_type.get(4, 999) and m5_close > float(pts["PTS_High"]) + 0.03:
                    ok = self._open_trade(row, "long", 4, "PTS_High_Breakout", 1, float(pts["PTS_High"]), conf_long, max(0, score_long), body_pct, sstate, phase)
                    if ok:
                        took_trade = True
                        bl["pts_entries"] += 1
                bs = sstate["breakout_state"]["short"]
                if (not took_trade) and bs["occurred"] and bs["pts_entries"] < self.max_per_type.get(4, 999) and m5_close < float(pts["PTS_Low"]) - 0.03:
                    ok = self._open_trade(row, "short", 4, "PTS_Low_Breakout", 1, float(pts["PTS_Low"]), conf_short, max(0, score_short), body_pct, sstate, phase)
                    if ok:
                        took_trade = True
                        bs["pts_entries"] += 1

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

        # entry type performance
        entry_type_perf = []
        for (et, d), g in t.groupby(["entry_type", "direction"]):
            s = grp_stats(g)
            entry_type_perf.append({
                "entry_type": int(et),
                "direction": d,
                **s,
                "avg_mfe": float(g["mfe_pips"].mean()),
                "avg_mae": float(g["mae_pips"].mean()),
                "avg_duration": float(g["duration_min"].mean()),
            })

        phase_perf = []
        for ph, g in t.groupby("session_phase_at_entry"):
            s = grp_stats(g)
            phase_perf.append({"phase": ph, **s})

        sig_map = ["VWAP_support", "RSI_momentum", "PTS_alignment", "Volume_surge"]
        signal_value = []
        for nm in sig_map:
            g = t[t["confirmation_signals_present"].fillna("").str.contains(nm)]
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
            "TP1_partial_plus_TP2_final",
            "TP1_partial_plus_trail_final",
            "TP1_partial_session_close",
            "Full_TP",
            "SL_Hit_PipBased",
            "Structural_Stop",
            "Time_Profit_Protect_Close",
            "Wind_Down_Close_Profit",
            "Wind_Down_Close_Small_Loss",
            "Wind_Down_Close_Large_Loss",
            "Force_Close_08_45",
            "Consecutive_Loss_Pause_Closed",
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

        all_entry_types_negative = False
        if entry_type_perf:
            et_df = pd.DataFrame(entry_type_perf)
            all_entry_types_negative = bool((et_df["net_usd"] <= 0).all())

        if pf < 0.80 and all_entry_types_negative:
            triage = "NO EDGE DETECTED in breakout approach. Tokyo USDJPY may not offer any retail-accessible edge regardless of strategy type. Recommend: 1) Test on AUDJPY or EURJPY Tokyo session, or 2) Accept Tokyo as untradeable for this account."
        elif 0.80 <= pf < 1.10 and any((x["pf"] > 1.3 and x["trades"] >= 20) for x in entry_type_perf):
            good_types = [f"Type{x['entry_type']}_{x['direction']}" for x in entry_type_perf if x["pf"] > 1.3 and x["trades"] >= 20]
            triage = f"PARTIAL EDGE. Specific entry type(s) show promise. Recommend isolating: {good_types}. Drop unprofitable entry types and re-run."
        elif pf >= 1.10:
            et_best = sorted([x for x in entry_type_perf if x["trades"] >= 10], key=lambda x: x["pf"], reverse=True)
            etb = et_best[0] if et_best else None
            strongest = f"Type {etb['entry_type']} {etb['direction']}, PF {etb['pf']:.3f}, {etb['trades']} trades" if etb else "N/A"
            triage = f"POSITIVE EXPECTANCY DETECTED. Proceed to optimization. Strongest entry type: {strongest}."
        else:
            triage = "NO EDGE DETECTED."

        best_entry_type = None
        et_candidates = [x for x in entry_type_perf if x["trades"] >= 10]
        if et_candidates:
            best_entry_type = sorted(et_candidates, key=lambda x: x["pf"], reverse=True)[0]

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

        # breakout direction bias
        sess_bias = []
        for dkey, ss in self.session_state.items():
            fbd = ss.get("first_breakout_direction")
            pnl = float(ss.get("session_pnl", 0.0))
            wh = bool(ss.get("whipsaw", False))
            sess_bias.append({"date": dkey, "first_breakout_direction": fbd, "session_pnl": pnl, "whipsaw": wh})
        sb = pd.DataFrame(sess_bias) if sess_bias else pd.DataFrame(columns=["first_breakout_direction", "session_pnl", "whipsaw"])
        long_sessions = sb[sb["first_breakout_direction"] == "LONG"]
        short_sessions = sb[sb["first_breakout_direction"] == "SHORT"]
        no_break = sb[sb["first_breakout_direction"].isna()]
        whips = sb[sb["whipsaw"] == True]

        def sess_pf(df):
            if len(df) == 0:
                return 0.0
            gp = float(df.loc[df["session_pnl"] > 0, "session_pnl"].sum())
            gl = abs(float(df.loc[df["session_pnl"] < 0, "session_pnl"].sum()))
            return gp / gl if gl > 0 else float("inf")

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
                    **{k: v for k, v in self.diag.items() if k not in ["near_miss_candidates"]},
                },
                "session_summary_table": sess_rows,
                "hour_of_session_breakdown": hour_rows,
                "near_miss_log": near_out,
                "level_performance": level_perf,
                "entry_type_performance": entry_type_perf,
                "phase_performance": phase_perf,
                "confirmation_signal_value": signal_value,
                "confluence_count_performance": conf_count_perf,
                "exit_distribution": exit_dist,
                "mfe_mae_analysis": mfe_mae,
                "ib_classification_breakdown": ib_tbl,
                "breakout_direction_bias": {
                    "sessions_first_breakout_long": int(len(long_sessions)),
                    "pf_first_breakout_long_sessions": float(sess_pf(long_sessions)),
                    "sessions_first_breakout_short": int(len(short_sessions)),
                    "pf_first_breakout_short_sessions": float(sess_pf(short_sessions)),
                    "sessions_no_breakout": int(len(no_break)),
                    "sessions_whipsaw": int(len(whips)),
                    "pf_whipsaw_sessions": float(sess_pf(whips)),
                    "avg_time_to_first_breakout_minutes": float(np.mean([
                        (x["breakout_state"]["long"]["entry_time"] or x["breakout_state"]["short"]["entry_time"]).hour * 60
                        + (x["breakout_state"]["long"]["entry_time"] or x["breakout_state"]["short"]["entry_time"]).minute
                        if ((x["breakout_state"]["long"]["entry_time"] is not None) or (x["breakout_state"]["short"]["entry_time"] is not None))
                        else np.nan
                        for x in self.session_state.values()
                    ])) if self.session_state else np.nan,
                },
                "triage_flag": triage,
                "best_entry_type_min10": best_entry_type,
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

        rp = out_dir / f"tokyo_tibs_v1_{label}_report.json"
        tp = out_dir / f"tokyo_tibs_v1_{label}_trade_log.csv"
        ep = out_dir / f"tokyo_tibs_v1_{label}_equity.csv"
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
        write_json(out_dir / "tokyo_tibs_v1_diagnostics.json", r1000["diagnostics"])
        write_json(out_dir / "tokyo_tibs_v1_advanced_analytics.json", {
            "mfe_mae": r1000["diagnostics"]["mfe_mae_analysis"],
            "day_of_week": r1000["day_of_week"],
            "monthly": r1000["monthly"],
            "level_performance": r1000["diagnostics"]["level_performance"],
            "entry_type_performance": r1000["diagnostics"]["entry_type_performance"],
            "phase_performance": r1000["diagnostics"]["phase_performance"],
            "confirmation_signal_value": r1000["diagnostics"]["confirmation_signal_value"],
            "ib_classification_breakdown": r1000["diagnostics"]["ib_classification_breakdown"],
            "breakout_direction_bias": r1000["diagnostics"]["breakout_direction_bias"],
            "triage_flag": r1000["diagnostics"]["triage_flag"],
        })
        pd.DataFrame(r1000["trades"]).to_csv(out_dir / "tokyo_tibs_v1_trade_log.csv", index=False)

    write_json(out_dir / "tokyo_tibs_v1_results.json", results)

    # Print compact scaling rows
    for r in scaling_rows:
        print(
            f"[{r['dataset']}] trades={r['trades']} wr={r['wr_pct']:.2f}% pf={r['pf']:.3f} "
            f"net={r['net_usd']:.2f} maxdd={r['maxdd_usd']:.2f} maxdd%={r['maxdd_pct']:.2f} $/mo={r['usd_per_month']:.2f}"
        )


if __name__ == "__main__":
    main()
