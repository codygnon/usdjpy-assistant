#!/usr/bin/env python3
"""
Enrich newera8 manual trades with M1 bar mechanics (parity with kumatora_manual_enriched.csv).

Computes M5 EMA9/21, M5 ATR14 (Wilder), M15 EMA9/21 for higher-timeframe bias, MFE/MAE,
and swing-macro reversal flags on the entry M1 bar.

Outputs:
  research_out/trade_analysis/newera8_deep_dive/newera8_manual_enriched.csv
  research_out/trade_analysis/newera8_deep_dive/NEWERA8_DEEP_DIVE.md
  research_out/trade_analysis/newera8_deep_dive/NEWERA8_PLAYBOOK_SPEC.md (static; narrative ↔ metrics ↔ gaps)

Inputs (defaults under repo research_out/):
  railway_exports/railway_trades_export_all_profiles.csv
  USDJPY_M1_OANDA_extended.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from core.regime_backtest_engine.swing_macro_bars import Bar4H
from core.regime_backtest_engine.swing_macro_trend import (
    is_bearish_reversal_bar,
    is_bullish_reversal_bar,
)
from core.reversal_risk import _session_name_utc

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRADES = ROOT / "research_out/railway_exports/railway_trades_export_all_profiles.csv"
DEFAULT_M1 = ROOT / "research_out/USDJPY_M1_OANDA_extended.csv"
OUT_DIR = ROOT / "research_out/trade_analysis/newera8_deep_dive"

PIP = 100.0  # USDJPY: 1 pip = 0.01 price


def _parse_ts(s: str) -> pd.Timestamp:
    t = pd.Timestamp(s)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t.as_unit("ns")


def _session_from_open(t: pd.Timestamp) -> str:
    return _session_name_utc(t)


def _prep_m1(path: Path) -> pd.DataFrame:
    m1 = pd.read_csv(path, parse_dates=["time"])
    m1["time"] = pd.to_datetime(m1["time"], utc=True)
    m1 = m1.set_index("time").sort_index()
    # Normalize resolution so trade timestamps (often ns) match resampled indices.
    m1.index = m1.index.as_unit("ns")
    return m1


def _build_m5(m1: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    oh = (
        m1.resample("5min", label="right", closed="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )
    oh.index = oh.index.as_unit("ns")
    ema9 = oh["close"].ewm(span=9, adjust=False).mean()
    ema21 = oh["close"].ewm(span=21, adjust=False).mean()
    prev_close = oh["close"].shift(1)
    tr = pd.concat(
        [
            oh["high"] - oh["low"],
            (oh["high"] - prev_close).abs(),
            (oh["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14 = tr.ewm(alpha=1 / 14, adjust=False).mean()
    return oh, ema9, ema21, atr14


def _build_m15(m1: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    o15 = (
        m1.resample("15min", label="right", closed="right")
        .agg({"high": "max", "low": "min", "close": "last"})
        .dropna()
    )
    o15.index = o15.index.as_unit("ns")
    e9 = o15["close"].ewm(span=9, adjust=False).mean()
    e21 = o15["close"].ewm(span=21, adjust=False).mean()
    return e9, e21, o15


def _m15_range_stats(
    o15: pd.DataFrame, m15_ts: pd.Timestamp, open_px: float, lookback: int = 20
) -> tuple[float, float, float, float, float]:
    """Return range_high, range_low, width_pips, entry_pct_in_range, dist_mid_pips."""
    pos = int(o15.index.searchsorted(m15_ts, side="right")) - 1
    if pos < 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    start = max(0, pos - lookback + 1)
    win = o15.iloc[start : pos + 1]
    rh = float(win["high"].max())
    rl = float(win["low"].min())
    width = (rh - rl) * PIP
    if rh > rl:
        pct = (open_px - rl) / (rh - rl)
    else:
        pct = 0.5
    mid = (rh + rl) / 2.0
    dist_mid = (open_px - mid) * PIP
    return rh, rl, width, float(np.clip(pct, 0.0, 1.0)), dist_mid


def _m5_channel_levels(oh5: pd.DataFrame, m5_ts: pd.Timestamp, lookback: int = 48) -> tuple[float, float]:
    """Rolling min low / max high over last `lookback` M5 bars ending at m5_ts (inclusive)."""
    pos = int(oh5.index.searchsorted(m5_ts, side="right")) - 1
    if pos < 0:
        return np.nan, np.nan
    start = max(0, pos - lookback + 1)
    win = oh5.iloc[start : pos + 1]
    return float(win["low"].min()), float(win["high"].max())


def _adverse_two_m5_closes(
    oh5: pd.DataFrame,
    open_t: pd.Timestamp,
    close_t: pd.Timestamp,
    direction: str,
    level_low: float,
    level_high: float,
) -> tuple[bool, float]:
    """Your rule proxy: 2 consecutive M5 closes beyond the played level (against position)."""
    if np.isnan(level_low) or np.isnan(level_high):
        return False, np.nan
    times = oh5.index
    i0 = int(times.searchsorted(open_t, side="right"))
    i1 = int(times.searchsorted(close_t, side="right")) - 1
    if i0 >= len(oh5) or i0 > i1:
        return False, np.nan
    closes = oh5["close"].iloc[i0 : i1 + 1].to_numpy(dtype=float)
    d = direction.lower()
    eps = 5e-5
    if d == "buy":
        cond = closes < (level_low - eps)
    else:
        cond = closes > (level_high + eps)
    run = 0
    for k, ok in enumerate(cond):
        if ok:
            run += 1
            if run >= 2:
                bar_end = times[i0 + k]
                minutes = (bar_end - open_t).total_seconds() / 60.0
                return True, float(minutes)
        else:
            run = 0
    return False, np.nan


def _last_ts_leq(idx: pd.DatetimeIndex, t: pd.Timestamp) -> pd.Timestamp:
    pos = idx.searchsorted(t, side="right")
    if pos == 0:
        raise ValueError(f"No M5/M15 bar at or before {t}")
    return idx[pos - 1]


def _entry_exit_idx(m1: pd.DataFrame, open_t: pd.Timestamp, close_t: pd.Timestamp) -> tuple[int, int]:
    open_floor = open_t.floor("1min")
    close_floor = close_t.floor("1min")
    try:
        i0 = int(m1.index.get_loc(open_floor))
        i1 = int(m1.index.get_loc(close_floor))
    except KeyError as e:
        raise KeyError(str(e)) from e
    if i1 < i0:
        i1 = i0
    return i0, i1


def _mfe_mae(
    m1: pd.DataFrame,
    i0: int,
    i1: int,
    direction: str,
    open_px: float,
) -> tuple[float, float, float, float]:
    """Return mfe_pips, mae_pips, time_to_mfe_min, time_to_mae_min."""
    sl = m1.iloc[i0 : i1 + 1]
    if sl.empty:
        return 0.0, 0.0, 0.0, 0.0
    d = direction.lower()
    highs = sl["high"].to_numpy(dtype=float)
    lows = sl["low"].to_numpy(dtype=float)
    n = len(sl)
    if d == "buy":
        fav = (highs - open_px) * PIP
        adv = (open_px - lows) * PIP
    else:
        fav = (open_px - lows) * PIP
        adv = (highs - open_px) * PIP
    mfe = float(np.max(fav))
    mae = float(np.max(adv))
    # First bar where path reaches final MFE / MAE
    t_mfe = 0.0
    t_mae = 0.0
    if d == "buy":
        run_fav = np.maximum.accumulate(fav)
        run_adv = np.maximum.accumulate(adv)
    else:
        run_fav = np.maximum.accumulate(fav)
        run_adv = np.maximum.accumulate(adv)
    hit_f = np.where(np.isclose(run_fav, mfe) | (run_fav >= mfe - 1e-6))[0]
    hit_a = np.where(np.isclose(run_adv, mae) | (run_adv >= mae - 1e-6))[0]
    if len(hit_f):
        t_mfe = float(hit_f[0])
    if len(hit_a):
        t_mae = float(hit_a[0])
    return mfe, mae, t_mfe, t_mae


def _ema_channel_bucket(open_px: float, e9: float, e21: float) -> str:
    lo, hi = (e9, e21) if e9 <= e21 else (e21, e9)
    if open_px < lo:
        return "below_both"
    if open_px > hi:
        return "above_both"
    return "between_emas"


def _m15_bias(close_px: float, e9: float, e21: float) -> str:
    if close_px > e9 > e21:
        return "bull_stack"
    if close_px < e9 < e21:
        return "bear_stack"
    return "mixed"


def _counter_m15_mean_reversion(m15_b: str, direction: str) -> bool:
    """True when trade direction opposes a clean 15m stack (fade-the-trend setup)."""
    d = direction.lower()
    if m15_b == "bull_stack" and d == "sell":
        return True
    if m15_b == "bear_stack" and d == "buy":
        return True
    return False


def _reversal_swing_macro(m1: pd.DataFrame, ema20_m1: pd.Series, idx: int, direction: str) -> bool:
    if idx <= 0:
        return False
    row = m1.iloc[idx]
    prev = m1.iloc[idx - 1]
    cur_bar = Bar4H(
        timestamp=m1.index[idx].to_pydatetime(),
        open=float(row.open),
        high=float(row.high),
        low=float(row.low),
        close=float(row.close),
        bar_count=1,
    )
    prev_bar = Bar4H(
        timestamp=m1.index[idx - 1].to_pydatetime(),
        open=float(prev.open),
        high=float(prev.high),
        low=float(prev.low),
        close=float(prev.close),
        bar_count=1,
    )
    e20 = float(ema20_m1.iloc[idx])
    d = direction.lower()
    if d == "buy":
        return is_bullish_reversal_bar(cur_bar, prev_bar, e20)
    return is_bearish_reversal_bar(cur_bar, prev_bar, e20)


def _error_row(
    tid: str,
    direction: str,
    o_t: pd.Timestamp,
    c_t: pd.Timestamp,
    o_px: float,
    c_px: float,
    r: pd.Series,
    err: str,
) -> dict:
    return {
        "trade_id": tid,
        "direction": direction,
        "open_time": o_t.isoformat(),
        "close_time": c_t.isoformat(),
        "open_price": o_px,
        "close_price": c_px,
        "p_l": float(r["P&L"]) if pd.notna(r["P&L"]) else np.nan,
        "winner": False,
        "pips_captured": np.nan,
        "duration_minutes": np.nan,
        "entry_bar_index": np.nan,
        "exit_bar_index": np.nan,
        "ema9_entry": np.nan,
        "ema21_entry": np.nan,
        "ema_distance_entry_pips": np.nan,
        "ema_gap_entry_pips": np.nan,
        "atr14_5m_entry_pips": np.nan,
        "ret15_before_entry_pips": np.nan,
        "ret60_before_entry_pips": np.nan,
        "momentum_5bar_pips": np.nan,
        "m15_ema9_entry": np.nan,
        "m15_ema21_entry": np.nan,
        "m15_bias": "",
        "ema9_m15_distance_entry_pips": np.nan,
        "session": "",
        "hour_utc": np.nan,
        "m1_open_entry": np.nan,
        "m1_high_entry": np.nan,
        "m1_low_entry": np.nan,
        "m1_close_entry": np.nan,
        "ema9_exit": np.nan,
        "ema_distance_exit_pips": np.nan,
        "mae_pips": np.nan,
        "mfe_pips": np.nan,
        "time_to_mfe_min": np.nan,
        "time_to_mae_min": np.nan,
        "entry_bar_range_pips": np.nan,
        "reversal_bar_swing_macro": False,
        "ema_channel_bucket": "",
        "lots": float(r["Lots"]) if pd.notna(r["Lots"]) else np.nan,
        "exit_reason": r["Exit Reason"],
        "preset": r["Preset"],
        "position_id": r["Position ID"],
        "minutes_since_prev_manual_same_direction": np.nan,
        "dca_candidate_same_dir_within_120m": False,
        "pips_per_atr": np.nan,
        "m15_range_high_20": np.nan,
        "m15_range_low_20": np.nan,
        "m15_range_width_pips": np.nan,
        "m15_range_entry_pct_0_1": np.nan,
        "m15_dist_to_mid_pips": np.nan,
        "dist_to_m15_range_low_pips": np.nan,
        "dist_to_m15_range_high_pips": np.nan,
        "m5_channel_low_48": np.nan,
        "m5_channel_high_48": np.nan,
        "dist_to_m5_channel_low_pips": np.nan,
        "dist_to_m5_channel_high_pips": np.nan,
        "anti_chase_abs_ret60_over_atr": np.nan,
        "anti_chase_ema_dist_over_atr": np.nan,
        "counter_m15_fade": False,
        "adverse_2x_m5_close_vs_channel": False,
        "minutes_to_adverse_2x_m5": np.nan,
        "entry_m1_range_over_m5_atr": np.nan,
        "news_spike_bar_proxy": False,
        "match_error": err,
    }


@dataclass
class EnrichConfig:
    trades_csv: Path
    m1_csv: Path
    out_dir: Path
    dca_window_min: float = 120.0


def run(cfg: EnrichConfig) -> Path:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    trades = pd.read_csv(cfg.trades_csv)
    manual = trades[
        (trades["Profile"] == "newera8")
        & (trades["Opened By"].astype(str).str.lower() == "manual")
    ].copy()
    manual["open_ts"] = manual["Open Time"].map(_parse_ts)
    manual["close_ts"] = manual["Close Time"].map(_parse_ts)
    manual = manual.sort_values("open_ts").reset_index(drop=True)

    m1 = _prep_m1(cfg.m1_csv)
    oh5, ema9_5, ema21_5, atr14_5 = _build_m5(m1)
    ema9_15, ema21_15, o15 = _build_m15(m1)
    ema20_m1 = m1["close"].ewm(span=20, adjust=False).mean()

    rows: list[dict] = []
    prev_same_dir_times: dict[str, pd.Timestamp] = {}

    for i, r in manual.iterrows():
        tid = str(r["Trade ID"])
        direction = str(r["Direction"]).lower()
        o_t = r["open_ts"]
        c_t = r["close_ts"]
        o_px = float(r["Open Price"])
        c_px = float(r["Close Price"])
        lots = float(r["Lots"]) if pd.notna(r["Lots"]) else np.nan

        try:
            ei, xi = _entry_exit_idx(m1, o_t, c_t)
        except KeyError:
            rows.append(
                _error_row(
                    tid,
                    direction,
                    o_t,
                    c_t,
                    o_px,
                    c_px,
                    r,
                    "bar_missing_open_or_close",
                )
            )
            continue

        m5_ts = _last_ts_leq(ema9_5.index, o_t)
        e9e = float(ema9_5.loc[m5_ts])
        e21e = float(ema21_5.loc[m5_ts])
        atr_e = float(atr14_5.loc[m5_ts]) * PIP

        m15_ts = _last_ts_leq(ema9_15.index, o_t)
        e9_15 = float(ema9_15.loc[m15_ts])
        e21_15 = float(ema21_15.loc[m15_ts])

        entry_row = m1.iloc[ei]
        m15_close_ref = float(m1["close"].iloc[ei])

        ret15 = (float(entry_row["close"]) - float(m1["close"].iloc[ei - 15])) * PIP if ei >= 15 else np.nan
        ret60 = (float(entry_row["close"]) - float(m1["close"].iloc[ei - 60])) * PIP if ei >= 60 else np.nan
        mom5 = (float(entry_row["close"]) - float(m1["close"].iloc[ei - 5])) * PIP if ei >= 5 else np.nan

        if pd.notna(r["Pips"]):
            pips_cap = float(r["Pips"])
        else:
            pips_cap = (c_px - o_px) * PIP if direction == "buy" else (o_px - c_px) * PIP

        mfe, mae, t_mfe, t_mae = _mfe_mae(m1, ei, xi, direction, o_px)

        m5_ts_x = _last_ts_leq(ema9_5.index, c_t)
        e9x = float(ema9_5.loc[m5_ts_x])
        ema_dist_exit = (c_px - e9x) * PIP

        rev = _reversal_swing_macro(m1, ema20_m1, ei, direction)
        bucket = _ema_channel_bucket(o_px, e9e, e21e)
        m15_b = _m15_bias(m15_close_ref, e9_15, e21_15)
        ema9_m15_dist = (o_px - e9_15) * PIP

        prev_t = prev_same_dir_times.get(direction)
        mins_since_prev = (o_t - prev_t).total_seconds() / 60.0 if prev_t is not None else np.nan
        dca_hint = (
            bool(prev_t is not None and mins_since_prev <= cfg.dca_window_min and mins_since_prev > 0)
            if pd.notna(mins_since_prev)
            else False
        )
        prev_same_dir_times[direction] = o_t

        rh15, rl15, w15, pct15, dmid = _m15_range_stats(o15, m15_ts, o_px, lookback=20)
        ch_low, ch_high = _m5_channel_levels(oh5, m5_ts, lookback=48)
        dist_m15_lo = (o_px - rl15) * PIP if pd.notna(rl15) else np.nan
        dist_m15_hi = (rh15 - o_px) * PIP if pd.notna(rh15) else np.nan
        dist_ch_lo = (o_px - ch_low) * PIP if pd.notna(ch_low) else np.nan
        dist_ch_hi = (ch_high - o_px) * PIP if pd.notna(ch_high) else np.nan
        ac_r60 = abs(ret60) / atr_e if atr_e > 1e-6 and pd.notna(ret60) else np.nan
        ac_ema = abs((o_px - e9e) * PIP) / atr_e if atr_e > 1e-6 else np.nan
        fade = _counter_m15_mean_reversion(m15_b, direction)
        brk, brk_min = _adverse_two_m5_closes(oh5, o_t, c_t, direction, ch_low, ch_high)
        ebr = (float(entry_row["high"]) - float(entry_row["low"])) * PIP
        em1_over_atr = ebr / atr_e if atr_e > 1e-6 else np.nan
        spike = bool(em1_over_atr >= 2.0) if pd.notna(em1_over_atr) else False

        rows.append(
            {
                "trade_id": tid,
                "direction": direction,
                "open_time": o_t.isoformat(),
                "close_time": c_t.isoformat(),
                "open_price": o_px,
                "close_price": c_px,
                "p_l": float(r["P&L"]) if pd.notna(r["P&L"]) else np.nan,
                "winner": bool(float(r["P&L"]) > 0) if pd.notna(r["P&L"]) else False,
                "pips_captured": pips_cap,
                "duration_minutes": float(r["Duration Minutes"]) if pd.notna(r["Duration Minutes"]) else np.nan,
                "entry_bar_index": ei,
                "exit_bar_index": xi,
                "ema9_entry": e9e,
                "ema21_entry": e21e,
                "ema_distance_entry_pips": (o_px - e9e) * PIP,
                "ema_gap_entry_pips": abs(e9e - e21e) * PIP,
                "atr14_5m_entry_pips": atr_e,
                "ret15_before_entry_pips": ret15,
                "ret60_before_entry_pips": ret60,
                "momentum_5bar_pips": mom5,
                "m15_ema9_entry": e9_15,
                "m15_ema21_entry": e21_15,
                "m15_bias": m15_b,
                "ema9_m15_distance_entry_pips": ema9_m15_dist,
                "session": _session_from_open(o_t),
                "hour_utc": int(o_t.hour),
                "m1_open_entry": float(entry_row["open"]),
                "m1_high_entry": float(entry_row["high"]),
                "m1_low_entry": float(entry_row["low"]),
                "m1_close_entry": float(entry_row["close"]),
                "ema9_exit": e9x,
                "ema_distance_exit_pips": ema_dist_exit,
                "mae_pips": mae,
                "mfe_pips": mfe,
                "time_to_mfe_min": t_mfe,
                "time_to_mae_min": t_mae,
                "entry_bar_range_pips": (float(entry_row["high"]) - float(entry_row["low"])) * PIP,
                "reversal_bar_swing_macro": rev,
                "ema_channel_bucket": bucket,
                "lots": lots,
                "exit_reason": r["Exit Reason"],
                "preset": r["Preset"],
                "position_id": r["Position ID"],
                "minutes_since_prev_manual_same_direction": mins_since_prev,
                "dca_candidate_same_dir_within_120m": dca_hint,
                "pips_per_atr": (pips_cap / atr_e) if atr_e > 1e-6 else np.nan,
                "m15_range_high_20": rh15,
                "m15_range_low_20": rl15,
                "m15_range_width_pips": w15,
                "m15_range_entry_pct_0_1": pct15,
                "m15_dist_to_mid_pips": dmid,
                "dist_to_m15_range_low_pips": dist_m15_lo,
                "dist_to_m15_range_high_pips": dist_m15_hi,
                "m5_channel_low_48": ch_low,
                "m5_channel_high_48": ch_high,
                "dist_to_m5_channel_low_pips": dist_ch_lo,
                "dist_to_m5_channel_high_pips": dist_ch_hi,
                "anti_chase_abs_ret60_over_atr": ac_r60,
                "anti_chase_ema_dist_over_atr": ac_ema,
                "counter_m15_fade": fade,
                "adverse_2x_m5_close_vs_channel": brk,
                "minutes_to_adverse_2x_m5": brk_min,
                "entry_m1_range_over_m5_atr": em1_over_atr,
                "news_spike_bar_proxy": spike,
                "match_error": "",
            }
        )

    out = pd.DataFrame(rows)
    csv_path = cfg.out_dir / "newera8_manual_enriched.csv"
    out.to_csv(csv_path, index=False)

    md_path = cfg.out_dir / "NEWERA8_DEEP_DIVE.md"
    md_path.write_text(_build_markdown(out, manual, cfg), encoding="utf-8")
    return csv_path


def _build_markdown(enriched: pd.DataFrame, raw_manual: pd.DataFrame, cfg: EnrichConfig) -> str:
    ok = enriched[enriched["match_error"].fillna("") == ""].copy()
    n_bad = int((enriched["match_error"].fillna("") != "").sum())
    lines = [
        "# newera8 manual — trade-by-trade mechanics",
        "",
        f"**Trades file:** `{cfg.trades_csv.relative_to(ROOT)}`",
        f"**M1 data:** `{cfg.m1_csv.relative_to(ROOT)}`",
        f"**Matched with full features:** {len(ok)} / {len(enriched)}",
        f"**Unmatched / bar errors:** {n_bad}",
        "",
        "Enriched CSV: `newera8_manual_enriched.csv`",
        "",
        "**Playbook (full narrative → columns → bot gaps):** `NEWERA8_PLAYBOOK_SPEC.md`",
        "",
        "## Methodology (parity with kumatora enrich)",
        "",
        "- M5 bars: `resample(5min, label='right', closed='right')`, EMA9/EMA21 on close (`ewm(span, adjust=False)`).",
        "- ATR14 on M5: Wilder smoothing `ewm(alpha=1/14, adjust=False)` on true range.",
        "- M15 EMA9/21 on 15m close for higher-timeframe bias (`m15_bias`).",
        "- M15 **20-bar** range (high/low/width) ending at the last M15 bar ≤ entry: `m15_range_*`, `dist_to_m15_range_*`.",
        "- M5 **48-bar** channel: min low / max high before entry → `m5_channel_low_48` / `m5_channel_high_48` (S/R **proxy**, not visual levels).",
        "- **Breakout proxy (your 2× M5 close rule):** `adverse_2x_m5_close_vs_channel` — during the trade, two consecutive M5 closes **below** channel low (long) or **above** channel high (short).",
        "- **Anti-chase:** `anti_chase_abs_ret60_over_atr` (impulse vs vol), `anti_chase_ema_dist_over_atr` (tight vs extended from M5 EMA9).",
        "- **Fade 15m trend:** `counter_m15_fade` — buy into bear_stack or sell into bull_stack.",
        "- **News spike scalp proxy:** `news_spike_bar_proxy` if entry M1 range ≥ **2×** M5 ATR14 (arbitrary but stable bar-shock flag).",
        "- Entry M1 bar: `open_time` floored to the minute; same for exit.",
        "- `reversal_bar_swing_macro`: `swing_macro_trend` bullish/bearish reversal on the **entry M1 bar** vs prior bar with **M1 EMA20**.",
        "  - This may differ from any external `kumatora_manual_enriched` reversal flag if that used a different spec.",
        "",
        f"- `dca_candidate_same_dir_within_120m`: prior **newera8 manual** trade in the **same direction** within **{cfg.dca_window_min:.0f} minutes** (sorted by open time).",
        "  - This flags **any** quick re-entry / scale-in sequence, not only losers followed by adds (see `minutes_since_prev_manual_same_direction` + `lots`).",
        "",
        "## Your playbook → columns",
        "",
        "| You said | Where it shows up in the CSV |",
        "| --- | --- |",
        "| S/R mean reversion in a range | `ema_channel_bucket`, `ret60_before_entry_pips`, `atr14_5m_entry_pips`, `entry_bar_range_pips` |",
        "| 15m bias + 5m/1m entry | `m15_bias`, `m15_ema9_entry`, `ema9_m15_distance_entry_pips` (M1 bar OHLC in `m1_*_entry`) |",
        "| Pullback vs chase | `momentum_5bar_pips`, `ret15_before_entry_pips`, `anti_chase_*_over_atr`, `ema_distance_entry_pips` |",
        "| Range TP vs opposite side | `m15_range_width_pips`, `dist_to_m15_range_low_pips` / `dist_to_m15_range_high_pips`, `m15_range_entry_pct_0_1` |",
        "| Trailing / occasional runners | `mfe_pips`, `pips_captured`, `pips_per_atr`, `duration_minutes` |",
        "| Scale up on great price; DCA in range | `lots`, `dca_candidate_same_dir_within_120m`, `minutes_since_prev_manual_same_direction` |",
        "| Breakout exit (2× M5 close beyond level) | `adverse_2x_m5_close_vs_channel`, `minutes_to_adverse_2x_m5` (level = 48-bar M5 channel) |",
        "| News spike mean-revert | `news_spike_bar_proxy`, `entry_m1_range_over_m5_atr` |",
        "| Fade extended 15m trend | `counter_m15_fade` |",
        "",
        "## Narrative mapping (from your playbook)",
        "",
        "- **Range mean reversion / S&R:** use `ema_channel_bucket`, `ret60_before_entry_pips`, `m15_bias`, `atr14_5m_entry_pips`.",
        "- **Higher TF bias (15m):** `m15_bias`, `ema9_m15_distance_entry_pips`.",
        "- **Pullback vs chase:** large `|ret15_before_entry_pips|` or `|ret60_before_entry_pips|` with small `ema_distance_entry_pips` may indicate dip entries vs extension.",
        "- **Trailing / wide outcomes:** `mfe_pips` vs `pips_captured` (capture ratio), `mae_pips`, `duration_minutes`.",
        "- **DCA hypothesis:** `dca_candidate_same_dir_within_120m` + rising `lots` vs prior leg (inspect adjacent rows in CSV).",
        "- **Breakout pain:** compare `adverse_2x_m5_close_vs_channel` with `p_l` — bot would have had a **hypothetical** structural exit signal.",
        "",
        "## Section 1 — Summary",
        "",
    ]
    if len(ok) == 0:
        lines.append("_No matched trades._")
        return "\n".join(lines)

    w = ok["winner"]
    lines.append(f"- Trades: **{len(ok)}** | Win%: **{100 * w.mean():.1f}%** | Sum P&L: **${ok['p_l'].sum():,.2f}**")
    lines.append(
        f"- Median pips: **{ok['pips_captured'].median():.2f}** | Median duration (min): **{ok['duration_minutes'].median():.1f}**"
    )
    lines.append(
        f"- Median MFE: **{ok['mfe_pips'].median():.2f}** | Median MAE: **{ok['mae_pips'].median():.2f}**"
    )
    cap = ok["pips_captured"].replace(0, np.nan)
    mfe = ok["mfe_pips"].replace(0, np.nan)
    ratio = (cap / mfe).replace([np.inf, -np.inf], np.nan).dropna()
    if len(ratio):
        lines.append(f"- Median capture/MFE (where MFE>0): **{ratio.median():.3f}**")
    lines.append(
        f"- `dca_candidate_same_dir_within_120m` true: **{ok['dca_candidate_same_dir_within_120m'].sum()}** "
        f"({100 * ok['dca_candidate_same_dir_within_120m'].mean():.1f}%)"
    )
    lines.append(
        f"- `reversal_bar_swing_macro` true: **{ok['reversal_bar_swing_macro'].sum()}** "
        f"({100 * ok['reversal_bar_swing_macro'].mean():.1f}%)"
    )
    lose_prev = ok[~ok["winner"]]
    adv_l = int(lose_prev["adverse_2x_m5_close_vs_channel"].sum()) if len(lose_prev) else 0
    lines.append(
        f"- `adverse_2x_m5_close_vs_channel` true: **{ok['adverse_2x_m5_close_vs_channel'].sum()}** "
        f"({100 * ok['adverse_2x_m5_close_vs_channel'].mean():.1f}%) — among losers: **{adv_l}** / {len(lose_prev)}"
    )
    lines.append(
        f"- `counter_m15_fade` true: **{ok['counter_m15_fade'].sum()}** | "
        f"`news_spike_bar_proxy` true: **{ok['news_spike_bar_proxy'].sum()}**"
    )
    lines.append("")
    lines.append("### By session")
    lines.append("")
    lines.append("| session | n | Win% | Sum P&L | Median pips | Median MFE | Median MAE |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for s, g in ok.groupby("session"):
        lines.append(
            f"| {s} | {len(g)} | {100 * g['winner'].mean():.1f}% | ${g['p_l'].sum():,.2f} | "
            f"{g['pips_captured'].median():.2f} | {g['mfe_pips'].median():.2f} | {g['mae_pips'].median():.2f} |"
        )
    lines.append("")
    lines.append("### By m15_bias")
    lines.append("")
    lines.append("| m15_bias | n | Win% | Sum P&L | Median ema9_m15_dist |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for s, g in ok.groupby("m15_bias"):
        lines.append(
            f"| {s} | {len(g)} | {100 * g['winner'].mean():.1f}% | ${g['p_l'].sum():,.2f} | "
            f"{g['ema9_m15_distance_entry_pips'].median():.2f} |"
        )
    lines.append("")
    lines.append("### Winners vs losers (mean)")
    lines.append("")
    win = ok[ok["winner"]]
    lose = ok[~ok["winner"]]
    if len(lose):
        def mean_col(df: pd.DataFrame, c: str) -> float:
            return float(df[c].mean())

        cols = [
            "ema_distance_entry_pips",
            "ema_gap_entry_pips",
            "atr14_5m_entry_pips",
            "momentum_5bar_pips",
            "ret15_before_entry_pips",
            "ret60_before_entry_pips",
            "mfe_pips",
            "mae_pips",
            "duration_minutes",
            "pips_captured",
        ]
        lines.append("| feature | winners | losers |")
        lines.append("| --- | ---: | ---: |")
        for c in cols:
            lines.append(f"| {c} | {mean_col(win, c):.3f} | {mean_col(lose, c):.3f} |")
    lines.append("")
    lines.append("### Largest losses (inspect for breakout / DCA)")
    lines.append("")
    cols = [
        "trade_id",
        "open_time",
        "direction",
        "session",
        "duration_minutes",
        "lots",
        "pips_captured",
        "p_l",
        "mae_pips",
        "mfe_pips",
        "dca_candidate_same_dir_within_120m",
    ]
    worst = ok.nsmallest(15, "p_l")[cols]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, wr in worst.iterrows():
        cells = [
            str(wr[c]) if c != "dca_candidate_same_dir_within_120m" else str(bool(wr[c]))
            for c in cols
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Enrich newera8 manual trades with M1/M5/M15 features.")
    ap.add_argument("--trades-csv", type=Path, default=DEFAULT_TRADES)
    ap.add_argument("--m1-csv", type=Path, default=DEFAULT_M1)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--dca-window-min", type=float, default=120.0)
    args = ap.parse_args()
    cfg = EnrichConfig(
        trades_csv=args.trades_csv,
        m1_csv=args.m1_csv,
        out_dir=args.out_dir,
        dca_window_min=args.dca_window_min,
    )
    path = run(cfg)
    print(f"Wrote {path}")
    print(f"Wrote {cfg.out_dir / 'NEWERA8_DEEP_DIVE.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
