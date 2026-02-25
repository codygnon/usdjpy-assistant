#!/usr/bin/env python3
"""
Build a research dataset and summary CSVs for USDJPY exhaustion-threshold design.

Usage:
  python scripts/export_usdjpy_exhaustion_research.py \
    --in USDJPY_M1_50k.csv \
    --out-dir ./research_out

Input CSV columns required:
  time,open,high,low,close
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PIP_SIZE = 0.01


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export USDJPY exhaustion research metrics from M1 candles.")
    p.add_argument("--in", dest="inp", required=True, help="Input M1 CSV (time,open,high,low,close)")
    p.add_argument("--out-dir", default="research_out", help="Output directory for CSV artifacts")
    p.add_argument("--reversal-pips", type=float, default=8.0, help="Reversal threshold in pips")
    p.add_argument("--horizons", default="10,20", help="M5 horizons in bars, comma-separated")
    p.add_argument("--tp-targets", default="4,6,8,10", help="TP targets in pips, comma-separated")
    p.add_argument("--sl-pips", type=float, default=20.0, help="SL in pips for dry-run response metrics")
    p.add_argument("--max-hold-bars", type=int, default=36, help="Max hold bars (M5) in dry-run")
    return p.parse_args()


def ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False).mean()


def rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    ma_up = up.ewm(alpha=1.0 / period, adjust=False).mean()
    ma_down = down.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(50.0)


def rolling_adr(daily_range_pips: pd.Series, window: int) -> pd.Series:
    return daily_range_pips.shift(1).rolling(window=window, min_periods=max(5, window // 4)).mean()


def m1_to_m5(df: pd.DataFrame) -> pd.DataFrame:
    d = df.set_index("time").sort_index()
    m5 = d.resample("5min", label="right", closed="right").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    )
    m5 = m5.dropna().reset_index()
    return m5


def session_of_hour(hour: int) -> str:
    # UTC buckets commonly used for FX session approximations.
    if 0 <= hour < 8:
        return "tokyo"
    if 8 <= hour < 13:
        return "london"
    if 13 <= hour < 22:
        return "ny"
    return "off"


def build_features(m1: pd.DataFrame, reversal_pips: float, horizons: list[int]) -> pd.DataFrame:
    m5 = m1_to_m5(m1)
    m5["ema5"] = ema(m5["close"], 5)
    m5["ema9"] = ema(m5["close"], 9)
    m5["ema21"] = ema(m5["close"], 21)
    m5["trend_side"] = np.where(m5["ema9"] > m5["ema21"], "bull", "bear")
    m5["ema_spread_pips"] = (m5["ema9"] - m5["ema21"]).abs() / PIP_SIZE
    m5["ema_spread_delta_pips"] = m5["ema_spread_pips"].diff()
    m5["ema_spread_slope3"] = (m5["ema_spread_pips"] - m5["ema_spread_pips"].shift(3)) / 3.0

    for p in (9, 14, 21):
        m5[f"rsi_{p}"] = rsi(m5["close"], p)

    # Wick metrics
    body = (m5["close"] - m5["open"]).abs()
    range_ = (m5["high"] - m5["low"]).replace(0, np.nan)
    upper = m5["high"] - m5[["open", "close"]].max(axis=1)
    lower = m5[["open", "close"]].min(axis=1) - m5["low"]
    m5["wick_ratio_total"] = ((upper + lower) / range_).fillna(0.0)
    m5["wick_ratio_upper"] = (upper / range_).fillna(0.0)
    m5["wick_ratio_lower"] = (lower / range_).fillna(0.0)
    m5["wick_body_ratio"] = ((upper + lower) / body.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Daily / ADR features
    m5["date"] = m5["time"].dt.date
    daily = m5.groupby("date").agg(day_high=("high", "max"), day_low=("low", "min"), day_open=("open", "first"))
    daily["day_range_pips"] = (daily["day_high"] - daily["day_low"]) / PIP_SIZE
    daily["adr30"] = rolling_adr(daily["day_range_pips"], 30)
    daily["adr60"] = rolling_adr(daily["day_range_pips"], 60)
    daily["adr90"] = rolling_adr(daily["day_range_pips"], 90)
    m5 = m5.merge(daily[["adr30", "adr60", "adr90", "day_high", "day_low", "day_open"]], left_on="date", right_index=True, how="left")
    m5["adr_ref"] = m5["adr30"].combine_first(m5["adr60"]).combine_first(m5["adr90"])
    m5["day_so_far_high"] = m5.groupby("date")["high"].cummax()
    m5["day_so_far_low"] = m5.groupby("date")["low"].cummin()
    m5["day_so_far_range_pips"] = (m5["day_so_far_high"] - m5["day_so_far_low"]) / PIP_SIZE
    m5["adr_consumed"] = (m5["day_so_far_range_pips"] / m5["adr_ref"]).replace([np.inf, -np.inf], np.nan)
    m5["adr_exceed_100"] = m5["adr_consumed"] > 1.0
    m5["adr_exceed_120"] = m5["adr_consumed"] > 1.2

    # Divergence flags (pivot-style in recent windows)
    for p in (9, 14, 21):
        rv = m5[f"rsi_{p}"]
        price_hh = m5["high"].rolling(10, min_periods=10).max() > m5["high"].rolling(20, min_periods=20).max().shift(10)
        rsi_lh = rv.rolling(10, min_periods=10).max() < rv.rolling(20, min_periods=20).max().shift(10)
        price_ll = m5["low"].rolling(10, min_periods=10).min() < m5["low"].rolling(20, min_periods=20).min().shift(10)
        rsi_hl = rv.rolling(10, min_periods=10).min() > rv.rolling(20, min_periods=20).min().shift(10)
        m5[f"bear_div_{p}"] = (price_hh & rsi_lh)
        m5[f"bull_div_{p}"] = (price_ll & rsi_hl)

    # Reversal labels / timing
    for h in horizons:
        fut_low = m5["low"].shift(-1).rolling(h, min_periods=h).min()
        fut_high = m5["high"].shift(-1).rolling(h, min_periods=h).max()
        m5[f"rev_down_{h}"] = (m5["close"] - fut_low) / PIP_SIZE >= reversal_pips
        m5[f"rev_up_{h}"] = (fut_high - m5["close"]) / PIP_SIZE >= reversal_pips

        def bars_to_hit_down(i: int) -> float:
            end = min(len(m5) - 1, i + h)
            target = m5.at[i, "close"] - reversal_pips * PIP_SIZE
            for j in range(i + 1, end + 1):
                if m5.at[j, "low"] <= target:
                    return float(j - i)
            return np.nan

        def bars_to_hit_up(i: int) -> float:
            end = min(len(m5) - 1, i + h)
            target = m5.at[i, "close"] + reversal_pips * PIP_SIZE
            for j in range(i + 1, end + 1):
                if m5.at[j, "high"] >= target:
                    return float(j - i)
            return np.nan

        m5[f"bars_to_rev_down_{h}"] = [bars_to_hit_down(i) if i < len(m5) - h - 1 else np.nan for i in range(len(m5))]
        m5[f"bars_to_rev_up_{h}"] = [bars_to_hit_up(i) if i < len(m5) - h - 1 else np.nan for i in range(len(m5))]

    # Session labels
    m5["hour_utc"] = m5["time"].dt.hour
    m5["dow"] = m5["time"].dt.day_name()
    m5["session"] = m5["hour_utc"].map(session_of_hour)

    # Composite dry-run score components (z-ish scaling)
    spread_pct = m5["ema_spread_pips"].rank(pct=True)
    narrowing = (-m5["ema_spread_slope3"]).clip(lower=0)
    narrow_pct = narrowing.rank(pct=True)
    adr_pct = m5["adr_consumed"].clip(lower=0).rank(pct=True)
    wick_pct = m5["wick_body_ratio"].rank(pct=True)
    div_sig = (
        m5["bear_div_14"].astype(int)
        + m5["bull_div_14"].astype(int)
        + m5["bear_div_9"].astype(int)
        + m5["bull_div_9"].astype(int)
    ) / 4.0
    m5["exhaustion_score_raw"] = (
        0.30 * spread_pct
        + 0.25 * narrow_pct
        + 0.20 * adr_pct
        + 0.15 * wick_pct
        + 0.10 * div_sig
    )
    m5["exhaustion_score"] = (m5["exhaustion_score_raw"] * 100.0).round(3)

    return m5


def summarize_rsi_divergence(feat: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    rows = []
    for p in (9, 14, 21):
        for h in horizons:
            bear = feat[feat[f"bear_div_{p}"]].copy()
            bull = feat[feat[f"bull_div_{p}"]].copy()

            bear_drop = (bear[f"rsi_{p}"] - bear[f"rsi_{p}"].shift(-1).rolling(h, min_periods=h).min()).dropna()
            bull_rise = (bull[f"rsi_{p}"].shift(-1).rolling(h, min_periods=h).max() - bull[f"rsi_{p}"]).dropna()

            bear_hit = bear[f"rev_down_{h}"].dropna()
            bull_hit = bull[f"rev_up_{h}"].dropna()

            bear_t = bear[f"bars_to_rev_down_{h}"].dropna()
            bull_t = bull[f"bars_to_rev_up_{h}"].dropna()

            rows.append({
                "rsi_period": p,
                "horizon_m5_bars": h,
                "bear_div_count": int(len(bear)),
                "bull_div_count": int(len(bull)),
                "median_rsi_drop_after_bear_div": round(float(bear_drop.median()), 3) if len(bear_drop) else np.nan,
                "median_rsi_rise_after_bull_div": round(float(bull_rise.median()), 3) if len(bull_rise) else np.nan,
                "bear_div_to_reversal_hit_rate": round(float(bear_hit.mean()), 4) if len(bear_hit) else np.nan,
                "bull_div_to_reversal_hit_rate": round(float(bull_hit.mean()), 4) if len(bull_hit) else np.nan,
                "median_bars_to_bear_reversal": round(float(bear_t.median()), 2) if len(bear_t) else np.nan,
                "median_bars_to_bull_reversal": round(float(bull_t.median()), 2) if len(bull_t) else np.nan,
            })
    out = pd.DataFrame(rows)
    out["mean_hit_rate"] = out[["bear_div_to_reversal_hit_rate", "bull_div_to_reversal_hit_rate"]].mean(axis=1)
    return out.sort_values(["horizon_m5_bars", "mean_hit_rate"], ascending=[True, False])


def summarize_ema_spread(feat: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Cross events as concrete reversal markers.
    sign = np.sign(feat["ema9"] - feat["ema21"])
    cross_idx = feat.index[(sign * sign.shift(1) < 0).fillna(False)].tolist()

    peak_rows = []
    narrowing_rows = []

    prev = 0
    for i in cross_idx:
        run = feat.iloc[prev:i + 1]
        if len(run) < 5:
            prev = i
            continue
        peak_rows.append({
            "cross_time": feat.at[i, "time"],
            "peak_spread_before_cross_pips": float(run["ema_spread_pips"].max()),
            "bars_in_run": int(len(run)),
        })

        nar = run["ema_spread_delta_pips"] < 0
        tail_narrow = 0
        for v in nar.iloc[::-1]:
            if bool(v):
                tail_narrow += 1
            else:
                break
        narrowing_rows.append({
            "cross_time": feat.at[i, "time"],
            "bars_narrowing_before_cross": int(tail_narrow),
            "mean_deceleration_pips_per_bar": float(run["ema_spread_slope3"].tail(max(3, min(10, len(run)))).mean()),
        })
        prev = i

    peak_df = pd.DataFrame(peak_rows)
    nar_df = pd.DataFrame(narrowing_rows)

    # NARROWING -> reversal likelihood
    lead = 6
    narrowing_now = feat["ema_spread_delta_pips"] < 0
    future_cross = pd.Series(False, index=feat.index)
    for i in range(len(feat)):
        end = min(len(feat), i + lead + 1)
        if any((j in cross_idx) for j in range(i + 1, end)):
            future_cross.iat[i] = True

    summary = pd.DataFrame([
        {
            "median_peak_spread_before_reversal_pips": round(float(peak_df["peak_spread_before_cross_pips"].median()), 3) if not peak_df.empty else np.nan,
            "median_bars_narrowing_before_cross": round(float(nar_df["bars_narrowing_before_cross"].median()), 3) if not nar_df.empty else np.nan,
            "mean_deceleration_at_reversals": round(float(nar_df["mean_deceleration_pips_per_bar"].mean()), 5) if not nar_df.empty else np.nan,
            "mean_deceleration_overall": round(float(feat["ema_spread_slope3"].mean()), 5),
            "narrowing_leads_to_reversal_rate_6bars": round(float(future_cross[narrowing_now].mean()), 4) if narrowing_now.any() else np.nan,
            "narrowing_sample_count": int(narrowing_now.sum()),
        }
    ])
    return summary, nar_df


def summarize_adr(feat: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = feat.groupby("date").agg(
        day_range_pips=("day_so_far_range_pips", "max"),
        adr_ref=("adr_ref", "max"),
        dow=("dow", "first"),
    ).dropna(subset=["day_range_pips"])
    d["adr_consumed_close"] = d["day_range_pips"] / d["adr_ref"]

    main = pd.DataFrame([
        {
            "adr30_median": round(float(d["day_range_pips"].tail(30).median()), 3) if len(d) >= 5 else np.nan,
            "adr30_p75": round(float(d["day_range_pips"].tail(30).quantile(0.75)), 3) if len(d) >= 5 else np.nan,
            "adr30_p90": round(float(d["day_range_pips"].tail(30).quantile(0.90)), 3) if len(d) >= 5 else np.nan,
            "adr60_median": round(float(d["day_range_pips"].tail(60).median()), 3) if len(d) >= 5 else np.nan,
            "adr60_p75": round(float(d["day_range_pips"].tail(60).quantile(0.75)), 3) if len(d) >= 5 else np.nan,
            "adr60_p90": round(float(d["day_range_pips"].tail(60).quantile(0.90)), 3) if len(d) >= 5 else np.nan,
            "adr90_median": round(float(d["day_range_pips"].tail(90).median()), 3) if len(d) >= 5 else np.nan,
            "adr90_p75": round(float(d["day_range_pips"].tail(90).quantile(0.75)), 3) if len(d) >= 5 else np.nan,
            "adr90_p90": round(float(d["day_range_pips"].tail(90).quantile(0.90)), 3) if len(d) >= 5 else np.nan,
            "freq_exceed_100pct": round(float((d["adr_consumed_close"] > 1.0).mean()), 4),
            "freq_exceed_120pct": round(float((d["adr_consumed_close"] > 1.2).mean()), 4),
        }
    ])

    by_dow = d.groupby("dow", dropna=False).agg(
        count=("day_range_pips", "size"),
        median_day_range_pips=("day_range_pips", "median"),
        p75_day_range_pips=("day_range_pips", lambda s: s.quantile(0.75)),
        p90_day_range_pips=("day_range_pips", lambda s: s.quantile(0.90)),
        exceed_100_rate=("adr_consumed_close", lambda s: (s > 1.0).mean()),
        exceed_120_rate=("adr_consumed_close", lambda s: (s > 1.2).mean()),
    ).reset_index()

    # % ADR consumed when day's final high/low first formed
    hi_rows = []
    lo_rows = []
    for day, g in feat.groupby("date"):
        g = g.sort_values("time")
        if g.empty:
            continue
        final_high = float(g["high"].max())
        final_low = float(g["low"].min())
        adr = float(g["adr_ref"].dropna().iloc[-1]) if g["adr_ref"].notna().any() else np.nan
        if not np.isnan(adr) and adr > 0:
            g2 = g.reset_index(drop=True)
            i_hi = int(g2.index[g2["high"] == final_high][0])
            i_lo = int(g2.index[g2["low"] == final_low][0])
            c_hi = float(g2.at[i_hi, "day_so_far_range_pips"]) / adr
            c_lo = float(g2.at[i_lo, "day_so_far_range_pips"]) / adr
            hi_rows.append(c_hi)
            lo_rows.append(c_lo)

    consumed = pd.DataFrame([
        {
            "median_adr_consumed_when_day_high_forms": round(float(np.median(hi_rows)), 4) if hi_rows else np.nan,
            "median_adr_consumed_when_day_low_forms": round(float(np.median(lo_rows)), 4) if lo_rows else np.nan,
        }
    ])

    intraday = feat.groupby(["session", "hour_utc"], dropna=False).agg(
        samples=("adr_consumed", "size"),
        median_adr_consumed=("adr_consumed", "median"),
        p75_adr_consumed=("adr_consumed", lambda s: s.quantile(0.75)),
        p90_adr_consumed=("adr_consumed", lambda s: s.quantile(0.90)),
    ).reset_index().sort_values(["session", "hour_utc"])

    return main, by_dow, consumed, intraday


def summarize_wick(feat: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sign = np.sign(feat["ema9"] - feat["ema21"])
    cross = (sign * sign.shift(1) < 0).fillna(False)
    pre_rev_idx = set()
    for i in feat.index[cross]:
        for k in range(1, 6):
            if i - k >= 0:
                pre_rev_idx.add(i - k)
    pre_rev = feat.loc[sorted(pre_rev_idx)] if pre_rev_idx else feat.iloc[0:0]

    run_mask = (feat["ema_spread_pips"] >= feat["ema_spread_pips"].quantile(0.7)) & (~cross)
    baseline = feat[run_mask]

    if baseline.empty:
        baseline = feat

    def q(s: pd.Series, qv: float) -> float:
        return round(float(s.quantile(qv)), 5) if len(s) else np.nan

    stats = pd.DataFrame([
        {
            "baseline_wick_body_median": q(baseline["wick_body_ratio"], 0.5),
            "baseline_wick_body_p75": q(baseline["wick_body_ratio"], 0.75),
            "pre_reversal_wick_body_median": q(pre_rev["wick_body_ratio"], 0.5),
            "pre_reversal_wick_body_p75": q(pre_rev["wick_body_ratio"], 0.75),
            "baseline_wick_total_median": q(baseline["wick_ratio_total"], 0.5),
            "pre_reversal_wick_total_median": q(pre_rev["wick_ratio_total"], 0.5),
            "median_shift_wick_body": round(float(pre_rev["wick_body_ratio"].median() - baseline["wick_body_ratio"].median()), 5) if len(pre_rev) and len(baseline) else np.nan,
        }
    ])

    # Separation by lookback: correlation with cross-in-next-N
    lookbacks = [3, 5, 8, 13, 21]
    rows = []
    cross_idx = set(feat.index[cross])
    for lb in lookbacks:
        rej = feat["wick_body_ratio"].rolling(lb, min_periods=lb).mean()
        y = []
        x = []
        for i in range(len(feat)):
            if i + 1 >= len(feat):
                continue
            lead_has_cross = any((j in cross_idx) for j in range(i + 1, min(len(feat), i + lb + 1)))
            if not np.isnan(rej.iat[i]):
                x.append(float(rej.iat[i]))
                y.append(1.0 if lead_has_cross else 0.0)
        corr = np.corrcoef(x, y)[0, 1] if len(x) > 10 and np.std(x) > 0 and np.std(y) > 0 else np.nan
        rows.append({"lookback": lb, "corr_rej_to_future_cross": round(float(corr), 5) if not np.isnan(corr) else np.nan, "samples": len(x)})
    sep = pd.DataFrame(rows).sort_values("corr_rej_to_future_cross", ascending=False, na_position="last")
    return stats, sep


def composite_tiers(feat: pd.DataFrame, horizons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    score = feat["exhaustion_score"].dropna()
    dist = score.describe(percentiles=[0.5, 0.7, 0.8, 0.9, 0.95]).to_frame("value").reset_index().rename(columns={"index": "metric"})

    h = max(horizons)
    # Reversal if either direction reversal prints (regime-agnostic dry run)
    rev_any = (feat[f"rev_down_{h}"].fillna(False) | feat[f"rev_up_{h}"].fillna(False)).astype(int)

    bins = [-np.inf, 40, 50, 60, 70, 80, 90, np.inf]
    labels = ["<40", "40-50", "50-60", "60-70", "70-80", "80-90", "90+"]
    grp = pd.cut(feat["exhaustion_score"], bins=bins, labels=labels)
    prob = feat.assign(score_bin=grp, reversal=rev_any).groupby("score_bin", dropna=False).agg(
        samples=("reversal", "size"),
        reversal_prob=("reversal", "mean"),
    ).reset_index()

    p70 = float(score.quantile(0.70)) if len(score) else np.nan
    p80 = float(score.quantile(0.80)) if len(score) else np.nan
    p90 = float(score.quantile(0.90)) if len(score) else np.nan
    tiers = pd.DataFrame([
        {"tier": "normal", "min_score": 0.0, "max_score": round(p80, 3)},
        {"tier": "extended", "min_score": round(p80, 3), "max_score": round(p90, 3)},
        {"tier": "very_extended", "min_score": round(p90, 3), "max_score": 100.0},
        {"tier": "alt_warn", "min_score": round(p70, 3), "max_score": round(p80, 3)},
    ])
    return dist, prob, tiers


def dry_run_response(feat: pd.DataFrame, tp_targets: list[float], sl_pips: float, max_hold_bars: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Entry model: each M5 bar enters with current trend direction (for comparative scoring only).
    entries = feat.copy()
    entries = entries[(entries["trend_side"].isin(["bull", "bear"])) & entries["close"].notna()].reset_index(drop=True)

    # Tier by score
    p80 = float(entries["exhaustion_score"].quantile(0.80))
    p90 = float(entries["exhaustion_score"].quantile(0.90))

    def tier(s: float) -> str:
        if s >= p90:
            return "very_extended"
        if s >= p80:
            return "extended"
        return "normal"

    entries["score_tier"] = entries["exhaustion_score"].map(tier)

    # Precompute MAE/MFE over hold window in pips
    mae_list = []
    mfe_list = []
    for i in range(len(entries)):
        side = entries.at[i, "trend_side"]
        entry = float(entries.at[i, "close"])
        end = min(len(entries) - 1, i + max_hold_bars)
        g = entries.iloc[i + 1:end + 1]
        if g.empty:
            mae_list.append(np.nan)
            mfe_list.append(np.nan)
            continue
        if side == "bull":
            adverse = ((entry - g["low"]) / PIP_SIZE).max()
            favorable = ((g["high"] - entry) / PIP_SIZE).max()
        else:
            adverse = ((g["high"] - entry) / PIP_SIZE).max()
            favorable = ((entry - g["low"]) / PIP_SIZE).max()
        mae_list.append(float(adverse))
        mfe_list.append(float(favorable))
    entries["mae_pips"] = mae_list
    entries["mfe_pips"] = mfe_list

    # Win rate by TP target (SL fixed)
    tp_rows = []
    for tp in tp_targets:
        wins = (entries["mfe_pips"] >= tp)
        losses = (entries["mae_pips"] >= sl_pips) & (~wins)
        decided = wins | losses
        wr = wins[decided].mean() if decided.any() else np.nan
        tp_rows.append({
            "tp_pips": tp,
            "sl_pips": sl_pips,
            "decided_trades": int(decided.sum()),
            "win_rate": round(float(wr), 4) if not np.isnan(wr) else np.nan,
        })
    by_tp = pd.DataFrame(tp_rows)

    # Win rate by tier during adverse conditions
    adverse = entries[entries["mae_pips"] >= 8.0].copy()
    tier_rows = []
    for tname, g in adverse.groupby("score_tier"):
        wins = (g["mfe_pips"] >= 6.0)
        tier_rows.append({
            "score_tier": tname,
            "trades": int(len(g)),
            "win_rate_tp6": round(float(wins.mean()), 4) if len(g) else np.nan,
            "median_mae_pips": round(float(g["mae_pips"].median()), 3) if len(g) else np.nan,
        })
    by_tier = pd.DataFrame(tier_rows).sort_values("score_tier")

    # Trapped-at-extreme % of losses
    # Define loss when MAE hits SL before hitting TP6.
    tp6_win = entries["mfe_pips"] >= 6.0
    sl_loss = (entries["mae_pips"] >= sl_pips) & (~tp6_win)
    trapped_losses = sl_loss & (entries["score_tier"] == "very_extended")
    trapped = pd.DataFrame([
        {
            "losses_total": int(sl_loss.sum()),
            "losses_trapped_at_extreme": int(trapped_losses.sum()),
            "trapped_loss_share": round(float(trapped_losses.sum() / sl_loss.sum()), 4) if sl_loss.sum() > 0 else np.nan,
        }
    ])

    # Survivable lot sizes in adverse tiers (risk budget fractions)
    adv = entries[entries["score_tier"].isin(["extended", "very_extended"]) & entries["mae_pips"].notna()].copy()
    worst_mae = float(adv["mae_pips"].quantile(0.95)) if len(adv) else np.nan
    median_price = float(entries["close"].median()) if len(entries) else 150.0
    pip_value_usd_per_lot = 1000.0 / median_price  # USDJPY approximation

    acct_sizes = [2500, 5000, 10000]
    risk_fracs = [0.005, 0.01, 0.02]
    surv_rows = []
    for eq in acct_sizes:
        for rf in risk_fracs:
            risk_usd = eq * rf
            max_lot = risk_usd / (worst_mae * pip_value_usd_per_lot) if (not np.isnan(worst_mae) and worst_mae > 0) else np.nan
            surv_rows.append({
                "equity_usd": eq,
                "risk_fraction": rf,
                "risk_usd": round(risk_usd, 2),
                "worst_mae_p95_adverse": round(worst_mae, 3) if not np.isnan(worst_mae) else np.nan,
                "max_survivable_lot": round(float(max_lot), 4) if not np.isnan(max_lot) else np.nan,
            })
    survivable = pd.DataFrame(surv_rows)

    return by_tp, by_tier, trapped, survivable


def write_csv(df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)


def main() -> int:
    args = parse_args()
    in_path = Path(args.inp)
    out_dir = Path(args.out_dir)
    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    tp_targets = [float(x.strip()) for x in args.tp_targets.split(",") if x.strip()]

    if not in_path.exists():
        print(f"Error: input not found: {in_path}")
        return 1

    m1 = pd.read_csv(in_path)
    required = {"time", "open", "high", "low", "close"}
    if not required.issubset(set(m1.columns)):
        print(f"Error: input must include columns: {sorted(required)}")
        return 1

    m1 = m1.copy()
    m1["time"] = pd.to_datetime(m1["time"], utc=True, errors="coerce")
    m1 = m1.dropna(subset=["time", "open", "high", "low", "close"]).sort_values("time")
    for c in ("open", "high", "low", "close"):
        m1[c] = pd.to_numeric(m1[c], errors="coerce")
    m1 = m1.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)

    feat = build_features(m1, reversal_pips=args.reversal_pips, horizons=horizons)
    write_csv(feat, out_dir / "USDJPY_M5_exhaustion_features.csv")

    rsi_df = summarize_rsi_divergence(feat, horizons)
    write_csv(rsi_df, out_dir / "USDJPY_rsi_divergence_summary.csv")

    ema_summary, ema_narrow = summarize_ema_spread(feat)
    write_csv(ema_summary, out_dir / "USDJPY_ema_spread_summary.csv")
    write_csv(ema_narrow, out_dir / "USDJPY_ema_narrowing_before_cross.csv")

    adr_main, adr_dow, adr_consumed, adr_curve = summarize_adr(feat)
    write_csv(adr_main, out_dir / "USDJPY_adr_summary.csv")
    write_csv(adr_dow, out_dir / "USDJPY_adr_by_dow.csv")
    write_csv(adr_consumed, out_dir / "USDJPY_adr_consumed_when_extremes_form.csv")
    write_csv(adr_curve, out_dir / "USDJPY_adr_intraday_consumption_curve.csv")

    wick_main, wick_sep = summarize_wick(feat)
    write_csv(wick_main, out_dir / "USDJPY_wick_rejection_summary.csv")
    write_csv(wick_sep, out_dir / "USDJPY_wick_rejection_lookback_separation.csv")

    score_dist, score_prob, tiers = composite_tiers(feat, horizons)
    write_csv(score_dist, out_dir / "USDJPY_composite_score_distribution.csv")
    write_csv(score_prob, out_dir / "USDJPY_reversal_probability_by_score_range.csv")
    write_csv(tiers, out_dir / "USDJPY_composite_tier_breakpoints.csv")

    by_tp, by_tier, trapped, survivable = dry_run_response(
        feat,
        tp_targets=tp_targets,
        sl_pips=args.sl_pips,
        max_hold_bars=args.max_hold_bars,
    )
    write_csv(by_tp, out_dir / "USDJPY_response_winrate_by_tp.csv")
    write_csv(by_tier, out_dir / "USDJPY_response_winrate_by_tier_adverse.csv")
    write_csv(trapped, out_dir / "USDJPY_response_trapped_at_extreme_losses.csv")
    write_csv(survivable, out_dir / "USDJPY_response_survivable_lot_sizes.csv")

    manifest = pd.DataFrame([
        {"file": p.name, "rows": sum(1 for _ in open(p, "r", encoding="utf-8")) - 1}
        for p in sorted(out_dir.glob("*.csv"))
    ])
    write_csv(manifest, out_dir / "USDJPY_research_manifest.csv")

    print(f"Wrote {len(list(out_dir.glob('*.csv')))} files -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
