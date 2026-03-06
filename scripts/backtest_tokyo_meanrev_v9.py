#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PIP_SIZE = 0.01


@dataclass
class Position:
    trade_id: int
    direction: str
    entry_time: pd.Timestamp
    entry_session_day: str
    entry_price: float
    sl_price: float
    tp1_price: float
    units_initial: int
    units_remaining: int
    sl_pips: float
    rejection_high: float
    rejection_low: float
    rejection_ratio: float
    confirmations: list[str]
    entry_midpoint: float
    stage1_hit: bool = False
    trail_dist_pips: float = 6.0
    trail_stop: Optional[float] = None
    realized_usd: float = 0.0
    realized_pip_units: float = 0.0
    max_fav_pips: float = 0.0
    max_adv_pips: float = 0.0
    exit_reason: Optional[str] = None
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tokyo mean reversion V9")
    p.add_argument("--config", required=True)
    return p.parse_args()


def load_m1(path: str) -> pd.DataFrame:
    f = Path(path)
    if not f.exists():
        raise FileNotFoundError(f"Missing CSV: {f}")
    df = pd.read_csv(f)
    need = {"time", "open", "high", "low", "close"}
    if not need.issubset(df.columns):
        raise ValueError(f"Missing required cols in {f}: {need}")
    keep = ["time", "open", "high", "low", "close"] + (["volume"] if "volume" in df.columns else [])
    out = df[keep].copy()
    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
    for c in ["open", "high", "low", "close"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if "volume" in out.columns:
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    out = out.dropna(subset=["time", "open", "high", "low", "close"]).sort_values("time").reset_index(drop=True)
    return out


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg["volume"] = "sum"
    r = df.set_index("time").resample(rule, label="right", closed="right").agg(agg).dropna().reset_index()
    return r


def rolling_rsi(series: pd.Series, period: int) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    avg_up = up.rolling(period, min_periods=period).mean()
    avg_dn = dn.rolling(period, min_periods=period).mean()
    rs = avg_up / avg_dn.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def rolling_atr(df: pd.DataFrame, period: int) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - pc).abs(),
            (df["low"] - pc).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def get_bid_ask(mid: float, spread_pips: float) -> tuple[float, float]:
    hs = spread_pips * PIP_SIZE / 2.0
    return float(mid) - hs, float(mid) + hs


def leg_usd_pips(direction: str, entry: float, exit_px: float, units: int) -> tuple[float, float]:
    if direction == "long":
        pips = (exit_px - entry) / PIP_SIZE
    else:
        pips = (entry - exit_px) / PIP_SIZE
    usd = pips * units * (PIP_SIZE / max(1e-9, exit_px))
    return float(pips), float(usd)


def compute_m5_features(m1: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    m5 = resample_ohlc(m1, "5min")
    m15 = resample_ohlc(m1, "15min")

    # RSI divergence stack
    rsi_p = int(cfg["rsi_divergence"]["rsi_period"])
    m5["rsi"] = rolling_rsi(m5["close"], rsi_p)

    lb = int(cfg["rsi_divergence"]["lookback_bars"])
    min_lb = int(cfg["rsi_divergence"]["min_lookback_bars"])
    bull_max = float(cfg["rsi_divergence"]["bullish_rsi_max"])
    bear_min = float(cfg["rsi_divergence"]["bearish_rsi_min"])

    bull_div = np.zeros(len(m5), dtype=bool)
    bear_div = np.zeros(len(m5), dtype=bool)
    for i in range(len(m5)):
        if i < lb:
            continue
        w0 = max(0, i - lb)
        w1 = max(0, i - min_lb)
        if w1 <= w0:
            continue
        prev = m5.iloc[w0:w1]
        if prev.empty or pd.isna(m5.iloc[i]["rsi"]):
            continue
        # bullish
        j_low = prev["low"].idxmin()
        if m5.iloc[i]["low"] < m5.loc[j_low, "low"] and m5.iloc[i]["rsi"] > m5.loc[j_low, "rsi"] and m5.iloc[i]["rsi"] < bull_max:
            bull_div[i] = True
        # bearish
        j_high = prev["high"].idxmax()
        if m5.iloc[i]["high"] > m5.loc[j_high, "high"] and m5.iloc[i]["rsi"] < m5.loc[j_high, "rsi"] and m5.iloc[i]["rsi"] > bear_min:
            bear_div[i] = True
    m5["bull_div"] = bull_div
    m5["bear_div"] = bear_div
    m5["bull_div_recent3"] = m5["bull_div"].rolling(3, min_periods=1).max().astype(bool)
    m5["bear_div_recent3"] = m5["bear_div"].rolling(3, min_periods=1).max().astype(bool)

    # ATR M15 (context/sizing)
    m15["atr_m15"] = rolling_atr(m15, 14)
    m5 = pd.merge_asof(m5.sort_values("time"), m15[["time", "atr_m15"]].sort_values("time"), on="time", direction="backward")

    # Volume or fallback climax
    vc = cfg["volume_climax"]
    if bool(vc.get("use_volume", True)) and "volume" in m5.columns and m5["volume"].notna().any():
        sma_p = int(vc.get("sma_period", 20))
        mult = float(vc.get("climax_multiplier", 2.0))
        m5["vol_sma"] = m5["volume"].rolling(sma_p, min_periods=sma_p).mean()
        m5["climax"] = m5["volume"] > (mult * m5["vol_sma"])
    else:
        m1a = m1.copy()
        m1a["atr_m1"] = rolling_atr(m1a, 14)
        m5 = pd.merge_asof(m5.sort_values("time"), m1a[["time", "atr_m1"]].sort_values("time"), on="time", direction="backward")
        mult = float(vc.get("climax_multiplier", 2.0))
        m5_range = (m5["high"] - m5["low"])
        m5["climax"] = (m5_range / m5["atr_m1"].replace(0, np.nan)) > mult

    return m5


def run_one(cfg: dict, run_cfg: dict) -> dict:
    m1 = load_m1(run_cfg["input_csv"])
    m1["time_utc"] = m1["time"].dt.tz_convert("UTC")
    m1["hour"] = m1["time_utc"].dt.hour
    m1["minute"] = m1["time_utc"].dt.minute
    m1["mod"] = m1["hour"] * 60 + m1["minute"]
    m1["weekday"] = m1["time_utc"].dt.day_name()
    m1["date_utc"] = m1["time_utc"].dt.date

    sf = cfg["session_filter"]
    start_h, start_m = [int(x) for x in sf["session_start_utc"].split(":")]
    end_h, end_m = [int(x) for x in sf["session_end_utc"].split(":")]
    start_mod = start_h * 60 + start_m
    end_mod = end_h * 60 + end_m
    allowed_days = set(sf["allowed_trading_days"])
    m1["in_session"] = (m1["mod"] >= start_mod) & (m1["mod"] < end_mod) & (m1["weekday"].isin(allowed_days))
    m1["session_day"] = m1["time_utc"].dt.date.astype(str)

    m5 = compute_m5_features(m1, cfg)
    m5_map = {t: row for t, row in zip(m5["time"], m5.to_dict("records"))}

    eq0 = float(cfg["starting_equity_usd"])
    equity = eq0
    spread = float(cfg["account"]["spread_pips"])
    base_risk_pct = float(cfg["position_sizing"]["risk_per_trade_pct"]) / 100.0
    max_units = int(cfg["position_sizing"]["max_units"])
    max_open = int(cfg["position_sizing"]["max_concurrent_positions"])
    max_trades_session = int(cfg["trade_management"]["max_trades_per_session"])
    min_gap_min = int(cfg["trade_management"]["min_time_between_entries_minutes"])
    no_reentry_min = int(cfg["trade_management"]["no_reentry_same_direction_after_loss_minutes"])

    env = cfg["session_envelope"]
    warmup_minutes = int(env["warmup_minutes"])
    uzs = float(env["upper_zone_start_mult"])
    uze = float(env["upper_zone_end_mult"])
    lzs = float(env["lower_zone_start_mult"])
    lze = float(env["lower_zone_end_mult"])

    rej_cfg = cfg["rejection_candle"]
    wick_ratio_req = float(rej_cfg["wick_to_body_ratio"])
    close_pos = float(rej_cfg["close_position_pct"])

    sig_cfg = cfg["signal"]
    div_window = int(sig_cfg["divergence_recent_m5_bars"])

    sl_cfg = cfg["exit_rules"]["stop_loss"]
    sl_buf = float(sl_cfg["rejection_extreme_buffer_pips"]) * PIP_SIZE
    sl_min = float(sl_cfg["min_sl_pips"])
    sl_max = float(sl_cfg["max_sl_pips"])

    tp_cfg = cfg["exit_rules"]["take_profit"]
    tp1_close_pct = float(tp_cfg["stage1_close_pct"])
    tp1_min = float(tp_cfg["stage1_min_pips"])
    tp1_max = float(tp_cfg["stage1_max_pips"])
    trail_init = float(tp_cfg["trail_distance_pips"])
    trail_tight = float(tp_cfg["tightened_trail_distance_pips"])

    time_cfg = cfg["exit_rules"]["time_exit"]
    tighten_after = int(time_cfg["tighten_trail_after_minutes"])

    early_cfg = cfg["exit_rules"]["early_exit"]
    ee_enabled = bool(early_cfg["enabled"])
    ee_after = int(early_cfg["minutes_after_entry"])
    ee_loss = float(early_cfg["loss_pips_threshold"])

    open_positions: list[Position] = []
    closed = []
    eq_curve = []

    session_state = {}

    # stats
    diag = {
        "bars_in_session": 0,
        "m5_bars_in_session": 0,
        "rejections_detected": 0,
        "rejections_in_zone": 0,
        "rejections_with_confirmation": 0,
        "entries_taken": 0,
        "signals_bottleneck_no_confirmation": 0,
        "signals_bottleneck_constraints": 0,
    }
    rej_samples = []
    trade_samples = []

    div_stats = {
        "bullish_divergences_detected": 0,
        "bearish_divergences_detected": 0,
        "divergence_as_confirmation_count": 0,
        "with_divergence_outcomes": [],
        "without_divergence_outcomes": [],
    }
    volume_stats = {"climaxes_detected": 0, "climax_as_confirmation_count": 0}
    sl_stats = {"trades_skipped_sl_too_wide": 0, "sl_list": []}
    envelope_stats = {
        "initial_ranges": [],
        "zone_widths": [],
        "sessions_with_lower_zone_touch": 0,
        "sessions_with_upper_zone_touch": 0,
        "sessions_with_both_zone_touch": 0,
    }

    max_cw = max_cl = cw = cl = 0
    peak_eq = equity
    drawdowns = []

    def ensure_session(sday: str, ts: pd.Timestamp, row: pd.Series):
        if sday not in session_state:
            session_state[sday] = {
                "start_ts": ts,
                "warmup_end": ts.normalize() + pd.Timedelta(minutes=start_mod + warmup_minutes),
                "warmup_high": float(row["high"]),
                "warmup_low": float(row["low"]),
                "ir_ready": False,
                "ir_high": np.nan,
                "ir_low": np.nan,
                "ir_range": np.nan,
                "session_high": float(row["high"]),
                "session_low": float(row["low"]),
                "trades": 0,
                "losses": 0,
                "stop": False,
                "last_entry": None,
                "last_loss_long": None,
                "last_loss_short": None,
                "risk_override_next": None,
                "first_trade_done": False,
                "lower_touches": 0,
                "upper_touches": 0,
                "lower_touch_flag": False,
                "upper_touch_flag": False,
                "session_pnl": 0.0,
                "ir_printed": False,
            }
        return session_state[sday]

    def close_position(pos: Position, ts: pd.Timestamp, exit_px: float, reason: str):
        nonlocal equity, cw, cl, max_cw, max_cl, peak_eq
        if pos.units_remaining > 0:
            pips, usd = leg_usd_pips(pos.direction, pos.entry_price, exit_px, pos.units_remaining)
            pos.realized_pip_units += pips * pos.units_remaining
            pos.realized_usd += usd
            pos.units_remaining = 0
            pos.exit_price = float(exit_px)
        pos.exit_reason = reason
        pos.exit_time = ts
        avg_pips = pos.realized_pip_units / max(1, pos.units_initial)
        before = equity
        equity += pos.realized_usd
        peak_eq = max(peak_eq, equity)
        dd = max(0.0, peak_eq - equity)
        drawdowns.append(dd)

        sst = session_state[pos.entry_session_day]
        sst["session_pnl"] += float(pos.realized_usd)
        if pos.realized_usd > 0:
            cw += 1
            cl = 0
        else:
            cl += 1
            cw = 0
            sst["losses"] += 1
            if pos.direction == "long":
                sst["last_loss_long"] = ts
            else:
                sst["last_loss_short"] = ts
            if (not sst["first_trade_done"]) and reason == "sl":
                sst["risk_override_next"] = 0.005
            if sst["losses"] >= 2:
                sst["stop"] = True
        sst["first_trade_done"] = True
        max_cw = max(max_cw, cw)
        max_cl = max(max_cl, cl)

        closed.append({
            "trade_id": pos.trade_id,
            "entry_datetime": str(pos.entry_time),
            "exit_datetime": str(pos.exit_time),
            "direction": pos.direction,
            "entry_price": pos.entry_price,
            "exit_price": pos.exit_price,
            "sl_price": pos.sl_price,
            "tp_price": pos.tp1_price,
            "exit_reason": pos.exit_reason,
            "pips": float(avg_pips),
            "usd": float(pos.realized_usd),
            "position_size_units": int(pos.units_initial),
            "sl_pips": float(pos.sl_pips),
            "mfe_pips": float(pos.max_fav_pips),
            "mae_pips": float(pos.max_adv_pips),
            "rejection_wick_body_ratio": float(pos.rejection_ratio),
            "confirmation_types": ",".join(pos.confirmations),
            "entry_session_day": pos.entry_session_day,
            "duration_minutes": float((pos.exit_time - pos.entry_time).total_seconds()/60.0),
            "equity_before": float(before),
            "equity_after": float(equity),
            "entry_regime": "TOKYO_MEANREV_V9",
            "entry_signal_mode": "rejection_envelope",
            "entry_session": "tokyo",
        })
        eq_curve.append({"trade_number": len(closed), "time": str(ts), "equity": equity})

    trade_id = 0

    printed_first_session_envelope = False

    for _, row in m1.iterrows():
        ts = pd.Timestamp(row["time"])
        in_session = bool(row["in_session"])
        sday = str(row["session_day"])
        mid_o, mid_h, mid_l, mid_c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        bid_h, ask_h = get_bid_ask(mid_h, spread)
        bid_l, ask_l = get_bid_ask(mid_l, spread)
        bid_c, ask_c = get_bid_ask(mid_c, spread)

        # manage open positions
        for pos in list(open_positions):
            held_min = (ts - pos.entry_time).total_seconds() / 60.0
            if pos.direction == "long":
                fav = (bid_h - pos.entry_price) / PIP_SIZE
                adv = (pos.entry_price - bid_l) / PIP_SIZE
                pos.max_fav_pips = max(pos.max_fav_pips, float(fav))
                pos.max_adv_pips = max(pos.max_adv_pips, float(adv))
                cur_pips = (bid_c - pos.entry_price) / PIP_SIZE
            else:
                fav = (pos.entry_price - ask_l) / PIP_SIZE
                adv = (ask_h - pos.entry_price) / PIP_SIZE
                pos.max_fav_pips = max(pos.max_fav_pips, float(fav))
                pos.max_adv_pips = max(pos.max_adv_pips, float(adv))
                cur_pips = (pos.entry_price - ask_c) / PIP_SIZE

            # early exit before stage1
            if ee_enabled and (not pos.stage1_hit) and held_min >= ee_after and cur_pips <= -ee_loss:
                sst = session_state.get(pos.entry_session_day, {})
                cur_mid = float((sst.get("session_high", mid_h) + sst.get("session_low", mid_l)) / 2.0)
                moved_against = (cur_mid < pos.entry_midpoint) if pos.direction == "long" else (cur_mid > pos.entry_midpoint)
                if moved_against:
                    px = bid_c if pos.direction == "long" else ask_c
                    close_position(pos, ts, px, "early_exit_midpoint_shift")
                    open_positions.remove(pos)
                    continue

            # stop
            if pos.direction == "long" and bid_l <= pos.sl_price:
                close_position(pos, ts, pos.sl_price, "sl" if not pos.stage1_hit else "tp1_then_sl")
                open_positions.remove(pos)
                continue
            if pos.direction == "short" and ask_h >= pos.sl_price:
                close_position(pos, ts, pos.sl_price, "sl" if not pos.stage1_hit else "tp1_then_sl")
                open_positions.remove(pos)
                continue

            # stage1 tp
            if not pos.stage1_hit:
                hit_tp1 = (bid_h >= pos.tp1_price) if pos.direction == "long" else (ask_l <= pos.tp1_price)
                if hit_tp1:
                    close_units = max(1, min(pos.units_remaining, int(math.floor(pos.units_initial * tp1_close_pct))))
                    pips, usd = leg_usd_pips(pos.direction, pos.entry_price, pos.tp1_price, close_units)
                    pos.realized_pip_units += pips * close_units
                    pos.realized_usd += usd
                    pos.units_remaining -= close_units
                    pos.stage1_hit = True
                    pos.trail_dist_pips = trail_init
                    if pos.direction == "long":
                        pos.trail_stop = bid_c - trail_init * PIP_SIZE
                    else:
                        pos.trail_stop = ask_c + trail_init * PIP_SIZE
                    if pos.units_remaining <= 0:
                        close_position(pos, ts, pos.tp1_price, "tp")
                        open_positions.remove(pos)
                        continue

            # trail on remainder
            if pos.stage1_hit and pos.units_remaining > 0:
                if held_min >= tighten_after:
                    pos.trail_dist_pips = trail_tight
                if pos.direction == "long":
                    new_trail = bid_c - pos.trail_dist_pips * PIP_SIZE
                    pos.trail_stop = max(float(pos.trail_stop), float(new_trail)) if pos.trail_stop is not None else float(new_trail)
                    if bid_l <= float(pos.trail_stop):
                        close_position(pos, ts, float(pos.trail_stop), "tp1_then_trail")
                        open_positions.remove(pos)
                        continue
                else:
                    new_trail = ask_c + pos.trail_dist_pips * PIP_SIZE
                    pos.trail_stop = min(float(pos.trail_stop), float(new_trail)) if pos.trail_stop is not None else float(new_trail)
                    if ask_h >= float(pos.trail_stop):
                        close_position(pos, ts, float(pos.trail_stop), "tp1_then_trail")
                        open_positions.remove(pos)
                        continue

        # force close outside session
        if not in_session:
            if open_positions:
                for pos in list(open_positions):
                    px = bid_c if pos.direction == "long" else ask_c
                    close_position(pos, ts, px, "session_close")
                    open_positions.remove(pos)
            continue

        diag["bars_in_session"] += 1
        sst = ensure_session(sday, ts, row)
        prev_session_high = float(sst["session_high"])
        prev_session_low = float(sst["session_low"])
        sst["session_high"] = max(prev_session_high, mid_h)
        sst["session_low"] = min(prev_session_low, mid_l)

        # build IR warmup
        if ts < pd.Timestamp(sst["warmup_end"]):
            sst["warmup_high"] = max(float(sst["warmup_high"]), mid_h)
            sst["warmup_low"] = min(float(sst["warmup_low"]), mid_l)
            continue
        if not sst["ir_ready"]:
            sst["ir_high"] = float(sst["warmup_high"])
            sst["ir_low"] = float(sst["warmup_low"])
            sst["ir_ready"] = True
            ir_range = max(PIP_SIZE, sst["ir_high"] - sst["ir_low"])
            sst["ir_range"] = float(ir_range)
            envelope_stats["initial_ranges"].append(ir_range / PIP_SIZE)
            if not printed_first_session_envelope:
                upper_start = sst["session_high"] + uzs * ir_range
                upper_end = sst["session_high"] + uze * ir_range
                lower_start = sst["session_low"] - lzs * ir_range
                lower_end = sst["session_low"] - lze * ir_range
                midpoint = (sst["session_high"] + sst["session_low"]) / 2.0
                print(
                    f"First session envelope: IR_high={sst['ir_high']:.5f} IR_low={sst['ir_low']:.5f} "
                    f"upper_zone=[{upper_start:.5f},{upper_end:.5f}] lower_zone=[{lower_end:.5f},{lower_start:.5f}] "
                    f"midpoint={midpoint:.5f}"
                )
                printed_first_session_envelope = True

        # only evaluate entries on M5 close bars
        m5r = m5_map.get(ts)
        if m5r is None:
            continue
        diag["m5_bars_in_session"] += 1

        if sst["stop"]:
            continue
        if len(open_positions) >= max_open:
            continue
        if sst["trades"] >= max_trades_session:
            continue
        if sst["last_entry"] is not None and (ts - pd.Timestamp(sst["last_entry"])).total_seconds() < min_gap_min * 60.0:
            continue

        # adaptive envelope zones (use pre-bar session extremes to avoid self-referential zones)
        zone_high_ref = prev_session_high
        zone_low_ref = prev_session_low
        ir_zone_range = float(sst.get("ir_range", np.nan))
        if not np.isfinite(ir_zone_range) or ir_zone_range <= 0:
            ir_zone_range = max(PIP_SIZE, float(zone_high_ref - zone_low_ref))
        upper_start = float(zone_high_ref + uzs * ir_zone_range)
        upper_end = float(zone_high_ref + uze * ir_zone_range)
        lower_start = float(zone_low_ref - lzs * ir_zone_range)
        lower_end = float(zone_low_ref - lze * ir_zone_range)
        midpoint = float((zone_high_ref + zone_low_ref) / 2.0)
        envelope_stats["zone_widths"].append((upper_end - upper_start) / PIP_SIZE)

        # rejection candle detection
        o,h,l,c = float(m5r["open"]), float(m5r["high"]), float(m5r["low"]), float(m5r["close"])
        rng = max(1e-9, h - l)
        body = abs(c - o)
        upper_wick = h - max(o,c)
        lower_wick = min(o,c) - l
        wick_ratio_up = upper_wick / max(1e-9, body)
        wick_ratio_lo = lower_wick / max(1e-9, body)
        close_upper_pct = (c - l) / rng
        close_lower_pct = (h - c) / rng

        bull_rej = (wick_ratio_lo >= wick_ratio_req) and (close_upper_pct >= (1.0 - close_pos)) and (l <= lower_start)
        bear_rej = (wick_ratio_up >= wick_ratio_req) and (close_lower_pct >= (1.0 - close_pos)) and (h >= upper_start)

        if bull_rej or bear_rej:
            diag["rejections_detected"] += 1

        if l <= lower_start and (not sst["lower_touch_flag"]):
            sst["lower_touches"] += 1
            sst["lower_touch_flag"] = True
        if h >= upper_start and (not sst["upper_touch_flag"]):
            sst["upper_touches"] += 1
            sst["upper_touch_flag"] = True
        # reset touch flag once bar away from zone to avoid duplicate counting in same push
        if l > lower_start:
            sst["lower_touch_flag"] = False
        if h < upper_start:
            sst["upper_touch_flag"] = False

        if not (bull_rej or bear_rej):
            continue
        diag["rejections_in_zone"] += 1

        # confirmations
        idx = int(m5.index[m5["time"] == ts][0])
        div_conf = False
        if bull_rej:
            div_conf = bool(m5.iloc[max(0, idx - div_window + 1): idx + 1]["bull_div"].any())
        else:
            div_conf = bool(m5.iloc[max(0, idx - div_window + 1): idx + 1]["bear_div"].any())
        climax_conf = bool(m5.iloc[max(0, idx - 1): idx + 1]["climax"].any())
        double_tap_conf = (sst["lower_touches"] >= 2) if bull_rej else (sst["upper_touches"] >= 2)

        if bool(m5r.get("bull_div", False)):
            div_stats["bullish_divergences_detected"] += 1
        if bool(m5r.get("bear_div", False)):
            div_stats["bearish_divergences_detected"] += 1
        if bool(m5r.get("climax", False)):
            volume_stats["climaxes_detected"] += 1

        confs = []
        if div_conf:
            confs.append("divergence")
            div_stats["divergence_as_confirmation_count"] += 1
        if climax_conf:
            confs.append("climax")
            volume_stats["climax_as_confirmation_count"] += 1
        if double_tap_conf:
            confs.append("double_tap")

        if len(confs) < 1:
            diag["signals_bottleneck_no_confirmation"] += 1
            continue

        diag["rejections_with_confirmation"] += 1

        direction = "long" if bull_rej else "short"
        ratio = wick_ratio_lo if bull_rej else wick_ratio_up
        if len(rej_samples) < 3:
            rej_samples.append({
                "time": str(ts),
                "ohlc": [o, h, l, c],
                "wick_body_ratio": float(ratio),
                "zone": "lower" if bull_rej else "upper",
                "confirmations": confs,
            })

        # re-entry cooldown per direction
        if direction == "long" and sst["last_loss_long"] is not None:
            if (ts - pd.Timestamp(sst["last_loss_long"])).total_seconds() < no_reentry_min * 60.0:
                diag["signals_bottleneck_constraints"] += 1
                continue
        if direction == "short" and sst["last_loss_short"] is not None:
            if (ts - pd.Timestamp(sst["last_loss_short"])).total_seconds() < no_reentry_min * 60.0:
                diag["signals_bottleneck_constraints"] += 1
                continue

        # market entry at m5 close
        entry = ask_c if direction == "long" else bid_c
        if direction == "long":
            sl_raw = l - sl_buf
            sl_pips = (entry - sl_raw) / PIP_SIZE
        else:
            sl_raw = h + sl_buf
            sl_pips = (sl_raw - entry) / PIP_SIZE

        if sl_pips > sl_max:
            sl_stats["trades_skipped_sl_too_wide"] += 1
            continue
        sl_pips = max(sl_min, sl_pips)
        if direction == "long":
            sl = entry - sl_pips * PIP_SIZE
        else:
            sl = entry + sl_pips * PIP_SIZE

        # tp1 to midpoint with bounds
        if direction == "long":
            tp1_dist = max(tp1_min, min(tp1_max, (midpoint - entry) / PIP_SIZE))
            if tp1_dist <= 0:
                tp1_dist = tp1_min
            tp1 = entry + tp1_dist * PIP_SIZE
        else:
            tp1_dist = max(tp1_min, min(tp1_max, (entry - midpoint) / PIP_SIZE))
            if tp1_dist <= 0:
                tp1_dist = tp1_min
            tp1 = entry - tp1_dist * PIP_SIZE

        # risk sizing
        risk_pct = sst["risk_override_next"] if sst["risk_override_next"] is not None else base_risk_pct
        units = math.floor((equity * risk_pct) / (sl_pips * (PIP_SIZE / max(1e-9, entry))))
        units = int(max(0, min(max_units, units)))
        if units < 1:
            continue

        trade_id += 1
        pos = Position(
            trade_id=trade_id,
            direction=direction,
            entry_time=ts,
            entry_session_day=sday,
            entry_price=float(entry),
            sl_price=float(sl),
            tp1_price=float(tp1),
            units_initial=units,
            units_remaining=units,
            sl_pips=float(sl_pips),
            rejection_high=float(h),
            rejection_low=float(l),
            rejection_ratio=float(ratio),
            confirmations=list(confs),
            entry_midpoint=float(midpoint),
        )
        open_positions.append(pos)
        sst["trades"] += 1
        sst["last_entry"] = ts
        sst["risk_override_next"] = None
        diag["entries_taken"] += 1

        if len(trade_samples) < 3:
            trade_samples.append({
                "time": str(ts),
                "direction": direction,
                "entry": float(entry),
                "sl": float(sl),
                "tp1": float(tp1),
                "confirmations": confs,
            })

    # close end-of-file
    if open_positions:
        last = m1.iloc[-1]
        ts = pd.Timestamp(last["time"])
        bid, ask = get_bid_ask(float(last["close"]), spread)
        for pos in list(open_positions):
            close_position(pos, ts, bid if pos.direction == "long" else ask, "session_close")
            open_positions.remove(pos)

    # session touch summaries
    for sst in session_state.values():
        if sst["lower_touches"] > 0:
            envelope_stats["sessions_with_lower_zone_touch"] += 1
        if sst["upper_touches"] > 0:
            envelope_stats["sessions_with_upper_zone_touch"] += 1
        if sst["lower_touches"] > 0 and sst["upper_touches"] > 0:
            envelope_stats["sessions_with_both_zone_touch"] += 1

    tdf = pd.DataFrame(closed)
    if tdf.empty:
        report = {
            "strategy_id": cfg.get("strategy_id", "tokyo_mean_reversion_v9"),
            "run_label": run_cfg["label"],
            "input_csv": run_cfg["input_csv"],
            "summary": {
                "total_trades": 0, "win_rate_pct": 0.0, "profit_factor": 0.0,
                "net_profit_usd": 0.0, "net_profit_pips": 0.0,
                "max_drawdown_usd": 0.0, "max_drawdown_pct": 0.0,
                "average_win_pips": 0.0, "average_loss_pips": 0.0,
                "average_win_usd": 0.0, "average_loss_usd": 0.0,
                "largest_win_usd": 0.0, "largest_loss_usd": 0.0,
                "return_on_starting_equity_pct": 0.0,
                "max_consecutive_wins": 0, "max_consecutive_losses": 0,
                "average_trade_duration_minutes": 0.0,
                "sharpe_ratio": 0.0, "calmar_ratio": 0.0,
                "starting_equity_usd": eq0, "ending_equity_usd": equity,
            },
            "breakdown": {
                "day_of_week": [], "monthly": [], "hourly": [], "exit_distribution": [],
                "average_trades_per_session": 0.0, "pct_sessions_with_zero_trades": 100.0,
            },
            "rejection_candle_stats": {
                "total_rejections_detected": int(diag["rejections_detected"]),
                "rejections_in_zone": int(diag["rejections_in_zone"]),
                "rejections_with_confirmation": int(diag["rejections_with_confirmation"]),
                "entries_taken": int(diag["entries_taken"]),
                "avg_wick_to_body_ratio_winners": 0.0,
                "avg_wick_to_body_ratio_losers": 0.0,
            },
            "session_envelope_stats": {
                "avg_initial_range_pips": float(np.mean(envelope_stats["initial_ranges"])) if envelope_stats["initial_ranges"] else 0.0,
                "avg_zone_width_pips": float(np.mean(envelope_stats["zone_widths"])) if envelope_stats["zone_widths"] else 0.0,
                **{k:int(v) for k,v in envelope_stats.items() if k.startswith("sessions_with_")}
            },
            "divergence_stats": {
                "bullish_divergences_detected": int(div_stats["bullish_divergences_detected"]),
                "bearish_divergences_detected": int(div_stats["bearish_divergences_detected"]),
                "divergence_as_confirmation_count": int(div_stats["divergence_as_confirmation_count"]),
                "wr_with_divergence": 0.0,
                "wr_without_divergence": 0.0,
            },
            "volume_climax_stats": {k:int(v) for k,v in volume_stats.items()},
            "sl_placement_stats": {
                "avg_sl_distance_pips": 0.0,
                "trades_skipped_sl_too_wide": int(sl_stats["trades_skipped_sl_too_wide"]),
                "sl_distance_distribution": {"6-10":0,"10-15":0,"15-20":0,"20-25":0},
            },
            "funnel": diag,
            "verification_samples": {"rejections": rej_samples, "trades": trade_samples},
            "equity_curve": [],
            "drawdown_curve": [],
            "trades": [],
        }
        return report

    wins = tdf[tdf["usd"] > 0]
    losses = tdf[tdf["usd"] < 0]
    gp = float(wins["usd"].sum())
    gl = float(abs(losses["usd"].sum()))
    pf = float(gp / gl) if gl > 0 else (float("inf") if gp > 0 else 0.0)
    wr = float((tdf["usd"] > 0).mean() * 100.0)
    net = float(tdf["usd"].sum())
    net_pips = float(tdf["pips"].sum())

    eq = tdf["equity_after"].to_numpy(dtype=float)
    peak = float(eq0)
    dd = []
    ddp = []
    maxdd = 0.0
    maxddp = 0.0
    for v in eq:
        peak = max(peak, float(v))
        d = max(0.0, peak - float(v))
        p = (d / peak * 100.0) if peak > 0 else 0.0
        dd.append(d)
        ddp.append(p)
        maxdd = max(maxdd, d)
        maxddp = max(maxddp, p)

    eqdf = pd.DataFrame({
        "trade_number": np.arange(1, len(tdf) + 1),
        "entry_datetime": tdf["entry_datetime"],
        "exit_datetime": tdf["exit_datetime"],
        "equity_after": eq,
        "drawdown_usd": dd,
        "drawdown_pct": ddp,
    })

    rets = (tdf["usd"] / tdf["equity_before"].replace(0, np.nan)).fillna(0)
    sharpe = float((rets.mean() / rets.std(ddof=0)) * math.sqrt(len(rets))) if rets.std(ddof=0) > 0 else 0.0
    calmar = float(((equity - eq0) / eq0 * 100.0) / maxddp) if maxddp > 0 else 0.0

    tdf["entry_ts"] = pd.to_datetime(tdf["entry_datetime"], utc=True)
    tdf["day"] = tdf["entry_ts"].dt.day_name().str[:3]
    tdf["month"] = tdf["entry_ts"].dt.to_period("M").astype(str)
    tdf["hour"] = tdf["entry_ts"].dt.hour

    dow = tdf.groupby("day").agg(trades=("trade_id","count"), wr=("usd", lambda s: float((s>0).mean()*100.0)), net=("usd","sum")).reset_index().to_dict("records")
    monthly = tdf.groupby("month").agg(trades=("trade_id","count"), wr=("usd", lambda s: float((s>0).mean()*100.0)), net=("usd","sum")).reset_index().to_dict("records")
    hourly = tdf.groupby("hour").agg(trades=("trade_id","count"), wr=("usd", lambda s: float((s>0).mean()*100.0)), net=("usd","sum")).reset_index().to_dict("records")

    exit_dist = tdf.groupby("exit_reason").agg(count=("trade_id","count"), net_usd=("usd","sum")).reset_index().to_dict("records")

    # extra stats
    with_div = tdf[tdf["confirmation_types"].str.contains("divergence", na=False)]
    without_div = tdf[~tdf["confirmation_types"].str.contains("divergence", na=False)]

    bins = {"6-10":0,"10-15":0,"15-20":0,"20-25":0}
    for x in tdf["sl_pips"].tolist():
        if x < 10: bins["6-10"] += 1
        elif x < 15: bins["10-15"] += 1
        elif x < 20: bins["15-20"] += 1
        else: bins["20-25"] += 1

    sessions = list(session_state.values())
    zero_sessions = sum(1 for s in sessions if int(s["trades"]) == 0)

    print("First 3 rejection candles detected:")
    for s in rej_samples:
        print(f"  {s['time']} OHLC={s['ohlc']} wick/body={s['wick_body_ratio']:.2f} zone={s['zone']} conf={s['confirmations']}")
    print("First 3 trades:")
    for s in trade_samples:
        print(f"  {s['time']} {s['direction']} entry={s['entry']:.5f} sl={s['sl']:.5f} tp1={s['tp1']:.5f} conf={s['confirmations']}")
    print(
        "Funnel: rejections_detected="
        f"{diag['rejections_detected']} -> in_zone={diag['rejections_in_zone']} -> "
        f"with_confirmation={diag['rejections_with_confirmation']} -> entered={diag['entries_taken']}"
    )

    report = {
        "strategy_id": cfg.get("strategy_id", "tokyo_mean_reversion_v9"),
        "run_label": run_cfg["label"],
        "input_csv": run_cfg["input_csv"],
        "summary": {
            "total_trades": int(len(tdf)),
            "win_rate_pct": float(wr),
            "average_win_pips": float(wins["pips"].mean()) if len(wins) else 0.0,
            "average_win_usd": float(wins["usd"].mean()) if len(wins) else 0.0,
            "average_loss_pips": float(losses["pips"].mean()) if len(losses) else 0.0,
            "average_loss_usd": float(losses["usd"].mean()) if len(losses) else 0.0,
            "largest_win_usd": float(tdf["usd"].max()),
            "largest_loss_usd": float(tdf["usd"].min()),
            "profit_factor": float(pf),
            "net_profit_usd": float(net),
            "net_profit_pips": float(net_pips),
            "return_on_starting_equity_pct": float((equity - eq0) / eq0 * 100.0),
            "max_drawdown_usd": float(maxdd),
            "max_drawdown_pct": float(maxddp),
            "max_consecutive_wins": int(max_cw),
            "max_consecutive_losses": int(max_cl),
            "average_trade_duration_minutes": float(tdf["duration_minutes"].mean()),
            "sharpe_ratio": float(sharpe),
            "calmar_ratio": float(calmar),
            "starting_equity_usd": float(eq0),
            "ending_equity_usd": float(equity),
        },
        "breakdown": {
            "day_of_week": dow,
            "monthly": monthly,
            "hourly": hourly,
            "exit_distribution": exit_dist,
            "average_trades_per_session": float(np.mean([s["trades"] for s in sessions])) if sessions else 0.0,
            "pct_sessions_with_zero_trades": float(100.0 * zero_sessions / len(sessions)) if sessions else 0.0,
        },
        "rejection_candle_stats": {
            "total_rejections_detected": int(diag["rejections_detected"]),
            "rejections_in_zone": int(diag["rejections_in_zone"]),
            "rejections_with_confirmation": int(diag["rejections_with_confirmation"]),
            "entries_taken": int(diag["entries_taken"]),
            "avg_wick_to_body_ratio_winners": float(wins["rejection_wick_body_ratio"].mean()) if len(wins) else 0.0,
            "avg_wick_to_body_ratio_losers": float(losses["rejection_wick_body_ratio"].mean()) if len(losses) else 0.0,
        },
        "session_envelope_stats": {
            "avg_initial_range_pips": float(np.mean(envelope_stats["initial_ranges"])) if envelope_stats["initial_ranges"] else 0.0,
            "avg_zone_width_pips": float(np.mean(envelope_stats["zone_widths"])) if envelope_stats["zone_widths"] else 0.0,
            "sessions_with_lower_zone_touch": int(envelope_stats["sessions_with_lower_zone_touch"]),
            "sessions_with_upper_zone_touch": int(envelope_stats["sessions_with_upper_zone_touch"]),
            "sessions_with_both_zone_touch": int(envelope_stats["sessions_with_both_zone_touch"]),
        },
        "divergence_stats": {
            "bullish_divergences_detected": int(div_stats["bullish_divergences_detected"]),
            "bearish_divergences_detected": int(div_stats["bearish_divergences_detected"]),
            "divergence_as_confirmation_count": int(div_stats["divergence_as_confirmation_count"]),
            "wr_with_divergence": float((with_div["usd"] > 0).mean() * 100.0) if len(with_div) else 0.0,
            "wr_without_divergence": float((without_div["usd"] > 0).mean() * 100.0) if len(without_div) else 0.0,
        },
        "volume_climax_stats": {
            "climaxes_detected": int(volume_stats["climaxes_detected"]),
            "climax_as_confirmation_count": int(volume_stats["climax_as_confirmation_count"]),
        },
        "sl_placement_stats": {
            "avg_sl_distance_pips": float(tdf["sl_pips"].mean()) if len(tdf) else 0.0,
            "trades_skipped_sl_too_wide": int(sl_stats["trades_skipped_sl_too_wide"]),
            "sl_distance_distribution": bins,
        },
        "funnel": diag,
        "verification_samples": {"rejections": rej_samples, "trades": trade_samples},
        "equity_curve": eqdf.to_dict("records"),
        "drawdown_curve": eqdf[["trade_number", "drawdown_usd", "drawdown_pct"]].to_dict("records"),
        "trades": tdf.drop(columns=["entry_ts", "day", "month", "hour"], errors="ignore").to_dict("records"),
    }

    if str(run_cfg.get("label", "")).lower() == "250k" and int(len(tdf)) < 30:
        bottlenecks = {
            "rejections_detected": diag["rejections_detected"],
            "rejections_in_zone": diag["rejections_in_zone"],
            "rejections_with_confirmation": diag["rejections_with_confirmation"],
            "entries_taken": diag["entries_taken"],
            "no_confirmation": diag["signals_bottleneck_no_confirmation"],
            "constraints": diag["signals_bottleneck_constraints"],
        }
        print(f"250k trades < 30 bottleneck: {bottlenecks}")

    return report


def write_outputs(report: dict, run_cfg: dict):
    Path(run_cfg["output_json"]).write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(report["trades"]).to_csv(run_cfg["output_trades_csv"], index=False)
    pd.DataFrame(report["equity_curve"]).to_csv(run_cfg["output_equity_csv"], index=False)


def main() -> int:
    args = parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    runs = cfg.get("run_sequence", [])
    if not runs:
        raise RuntimeError("run_sequence missing")

    pfs = []
    rows = []
    for r in runs:
        rep = run_one(cfg, r)
        write_outputs(rep, r)
        s = rep["summary"]
        dd = float(s["max_drawdown_usd"])
        ndd = float(s["net_profit_usd"]) / dd if dd > 0 else 0.0
        print(
            f"[{r['label']}] trades={s['total_trades']} wr={s['win_rate_pct']:.2f}% "
            f"net_usd={s['net_profit_usd']:.2f} pf={s['profit_factor']:.3f} maxdd={s['max_drawdown_usd']:.2f} -> {r['output_json']}"
        )
        rows.append((str(r['label']).lower(), int(s['total_trades']), float(s['win_rate_pct']), float(s['profit_factor']), float(s['net_profit_usd']), float(s['max_drawdown_usd']), float(ndd)))
        pfs.append(float(s["profit_factor"]))

    print("SCALING CHECK:")
    print("| Dataset | Trades | WR | PF | Net | MaxDD | Net/MaxDD |")
    for r in rows:
        print(f"| {r[0]} | {r[1]} | {r[2]:.2f}% | {r[3]:.3f} | {r[4]:.2f} | {r[5]:.2f} | {r[6]:.3f} |")
    print(f"PF StdDev across datasets: {float(np.std(pfs, ddof=0)):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
