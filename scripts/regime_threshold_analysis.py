#!/usr/bin/env python3
"""
Regime Threshold Analysis
=========================
Compute (bb_regime, adx, m5_slope_abs, h1_aligned, bb_width_pctile) on every
M1 bar, then label each bar with which standalone strategy actually traded
profitably.  Produce percentile summaries so we can set data-driven thresholds
for the regime-led eligibility router.

Usage:
    python scripts/regime_threshold_analysis.py \
        --input research_out/USDJPY_M1_OANDA_250k.csv \
        --output research_out/regime_threshold_250k.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import backtest_tokyo_meanrev as v14_engine
from scripts import backtest_v2_multisetup_london as v2_engine
from scripts import backtest_session_momentum as v44_engine

PIP_SIZE = 0.01
BB_PERIOD = 25
BB_STD = 2.2
RSI_PERIOD = 14
ATR_PERIOD = 14
ADX_PERIOD = 14
M5_EMA_FAST = 9
M5_EMA_SLOW = 21
H1_EMA_FAST = 20
H1_EMA_SLOW = 50
SLOPE_BARS = 4
BB_WIDTH_LOOKBACK = 100

V14_CONFIG = ROOT / "research_out" / "tokyo_optimized_v14_config.json"
V2_CONFIG = ROOT / "research_out" / "v2_exp4_winner_baseline_config.json"
V44_CONFIG = ROOT / "research_out" / "session_momentum_v44_base_config.json"


# ── helpers ──────────────────────────────────────────────────────────

def _load_m1(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna().sort_values("time").reset_index(drop=True)


def _resample(m1: pd.DataFrame, rule: str) -> pd.DataFrame:
    r = (
        m1.set_index("time")
        .resample(rule, label="right", closed="right")
        .agg(open=("open", "first"), high=("high", "max"),
             low=("low", "min"), close=("close", "last"))
        .dropna()
        .reset_index()
    )
    return r.sort_values("time").reset_index(drop=True)


def _rolling_atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev).abs(),
        (df["low"] - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def _rolling_adx(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    prev = close.shift(1)
    tr = pd.concat([
        (high - low).abs(), (high - prev).abs(), (low - prev).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=period).mean()
    pdi = 100 * pd.Series(plus_dm, index=df.index).rolling(period, min_periods=period).mean() / atr.replace(0, np.nan)
    mdi = 100 * pd.Series(minus_dm, index=df.index).rolling(period, min_periods=period).mean() / atr.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.rolling(period, min_periods=period).mean()


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


# ── Step 1: compute regime indicators on every M1 bar ────────────────

def compute_regime_features(m1: pd.DataFrame) -> pd.DataFrame:
    """Return M1 DataFrame with regime indicator columns merged in."""
    out = m1.copy()

    m5 = _resample(out, "5min")
    m15 = _resample(out, "15min")
    h1 = _resample(out, "1h")

    # M5 BB
    sma = m5["close"].rolling(BB_PERIOD, min_periods=BB_PERIOD).mean()
    std = m5["close"].rolling(BB_PERIOD, min_periods=BB_PERIOD).std(ddof=0)
    m5["bb_upper"] = sma + BB_STD * std
    m5["bb_lower"] = sma - BB_STD * std
    m5["bb_mid"] = sma
    m5["bb_width"] = (m5["bb_upper"] - m5["bb_lower"]) / m5["bb_mid"].replace(0, np.nan)
    m5["bb_width_pctile"] = m5["bb_width"].rolling(BB_WIDTH_LOOKBACK, min_periods=BB_WIDTH_LOOKBACK).rank(pct=True)
    cutoff = m5["bb_width"].rolling(BB_WIDTH_LOOKBACK, min_periods=BB_WIDTH_LOOKBACK).quantile(0.80)
    m5["bb_regime"] = np.where(m5["bb_width"] < cutoff, "ranging", "trending")

    # M5 EMA slope
    m5_ema_fast = _ema(m5["close"], M5_EMA_FAST)
    m5_ema_slow = _ema(m5["close"], M5_EMA_SLOW)
    m5["m5_slope"] = (m5_ema_fast - m5_ema_fast.shift(SLOPE_BARS)) / PIP_SIZE / max(1, SLOPE_BARS)
    m5["m5_slope_abs"] = m5["m5_slope"].abs()

    # M15 ADX + ATR
    m15["adx"] = _rolling_adx(m15, ADX_PERIOD)
    m15["atr"] = _rolling_atr(m15, ATR_PERIOD)

    # H1 EMA direction (raw, not yet side-aware)
    h1_fast = _ema(h1["close"], H1_EMA_FAST)
    h1_slow = _ema(h1["close"], H1_EMA_SLOW)
    h1["h1_ema_fast"] = h1_fast
    h1["h1_ema_slow"] = h1_slow
    h1["h1_bullish"] = (h1_fast > h1_slow).astype(int)
    h1["h1_trend_dir"] = np.where(h1_fast > h1_slow, "up", "down")

    # Merge to M1 via merge_asof (backward)
    m5_cols = ["time", "bb_regime", "bb_width", "bb_width_pctile", "m5_slope", "m5_slope_abs"]
    m15_cols = ["time", "adx", "atr"]
    h1_cols = ["time", "h1_bullish", "h1_trend_dir"]

    out = pd.merge_asof(out.sort_values("time"), m5[m5_cols].sort_values("time"),
                        on="time", direction="backward")
    out = pd.merge_asof(out.sort_values("time"), m15[m15_cols].sort_values("time"),
                        on="time", direction="backward")
    out = pd.merge_asof(out.sort_values("time"), h1[h1_cols].sort_values("time"),
                        on="time", direction="backward")

    # Side-aware H1 alignment: H1 direction agrees with M5 slope direction
    out["h1_aligned"] = (
        ((out["m5_slope"] > 0) & (out["h1_trend_dir"] == "up"))
        | ((out["m5_slope"] < 0) & (out["h1_trend_dir"] == "down"))
    ).astype(int)

    return out


# ── Step 2: run standalone backtests and extract entry bars ──────────

def _run_v14(input_csv: str) -> list[dict]:
    cfg = json.loads(V14_CONFIG.read_text())
    run_cfg = {"label": "regime", "input_csv": input_csv,
               "output_json": "", "output_trades_csv": "", "output_equity_csv": ""}
    report = v14_engine.run_one(cfg, run_cfg)
    trades = []
    for t in report.get("trades", []):
        entry_ts = pd.Timestamp(t.get("entry_datetime", t.get("entry_ts")))
        if entry_ts.tzinfo is None:
            entry_ts = entry_ts.tz_localize("UTC")
        pips = float(t.get("pips", 0.0))
        usd = float(t.get("usd", 0.0))
        trades.append({
            "strategy": "v14",
            "entry_time": entry_ts.tz_convert("UTC"),
            "side": "buy" if str(t.get("direction", "long")).lower() in {"long", "buy"} else "sell",
            "pips": pips,
            "usd": usd,
            "profitable": pips > 0,
        })
    return trades


def _run_london_v2(m1: pd.DataFrame) -> list[dict]:
    user_cfg = json.loads(V2_CONFIG.read_text())
    cfg = v2_engine.merge_config(user_cfg)
    trades_df, _, _ = v2_engine.run_backtest(m1, cfg)
    trades = []
    if trades_df is not None and not trades_df.empty:
        for _, t in trades_df.iterrows():
            entry_ts = pd.Timestamp(t["entry_time_utc"])
            if entry_ts.tzinfo is None:
                entry_ts = entry_ts.tz_localize("UTC")
            pips = float(t.get("pnl_pips", 0.0))
            trades.append({
                "strategy": "london_v2",
                "entry_time": entry_ts.tz_convert("UTC"),
                "side": "buy" if str(t.get("direction", "long")).lower() == "long" else "sell",
                "pips": pips,
                "usd": float(t.get("pnl_usd", 0.0)),
                "profitable": pips > 0,
            })
    return trades


def _convert_v44_embedded_to_flat(
    embedded: dict[str, Any], input_csv: str, out_json: str,
) -> dict[str, Any]:
    """Mirror the merged backtest's conversion from nested V44 config to flat."""
    cfg = dict(embedded)
    v5 = dict(cfg.pop("v5", {}))
    sessions_utc = dict(cfg.pop("sessions_utc", {}))
    trend = dict(cfg.pop("trend", {}))
    caps = dict(cfg.pop("caps", {}))
    risk = dict(cfg.pop("risk", {}))

    flat: dict[str, Any] = {
        "version": "v5",
        "mode": cfg.get("mode", "session"),
        "inputs": [input_csv],
        "out": out_json,
        "spread_mode": cfg.get("spread_mode", "realistic"),
        "spread_pips": cfg.get("spread_avg_target_pips", cfg.get("spread_pips", 2.0)),
        "spread_min_pips": cfg.get("spread_min_pips", 1.0),
        "spread_max_pips": cfg.get("spread_max_pips", 3.0),
        "max_entry_spread_pips": cfg.get("max_entry_spread_pips", 3.0),
        "london_start": float(sessions_utc.get("london", [8.5, 11.0])[0]),
        "london_end": float(sessions_utc.get("london", [8.5, 11.0])[1]),
        "ny_start": float(sessions_utc.get("ny_overlap", [13.0, 16.0])[0]),
        "ny_end": float(sessions_utc.get("ny_overlap", [13.0, 16.0])[1]),
        "h1_ema_fast": int(trend.get("h1_ema_fast", 20)),
        "h1_ema_slow": int(trend.get("h1_ema_slow", 50)),
        "h1_allow_slope_direction": bool(trend.get("h1_allow_slope_direction", True)),
        "h1_slope_bars": int(trend.get("h1_slope_bars", 6)),
        "max_open_positions": int(caps.get("max_open_positions", 1)),
        "max_entries_per_day": int(caps.get("max_entries_per_day", 3)),
        "loss_after_first_full_sl_lot_mult": float(
            caps.get("loss_after_first_full_sl_lot_mult", 1.0)
        ),
    }
    for k, v in risk.items():
        flat[k] = v
    for k, v in v5.items():
        flat[f"v5_{k}"] = v
    flat["v5_sessions"] = "ny_only"

    # Also copy any top-level keys not already handled (the V44 base config
    # is already flat, so many keys are directly at the top level).
    for k, v in cfg.items():
        if k not in flat and not isinstance(v, dict):
            flat[k] = v

    return flat


def _run_v44(input_csv: str) -> list[dict]:
    raw = json.loads(V44_CONFIG.read_text())
    embedded = raw.get("config", raw) if isinstance(raw, dict) else {}
    with tempfile.TemporaryDirectory(prefix="regime_v44_") as td:
        tmp_cfg = Path(td) / "v44_flat.json"
        flat = _convert_v44_embedded_to_flat(
            embedded, input_csv, str(Path(td) / "v44_out.json"),
        )
        tmp_cfg.write_text(json.dumps(flat, indent=2))
        args = v44_engine.parse_args(["--config", str(tmp_cfg)])
        results = v44_engine.run_backtest_v5(args)
    trades = []
    for t in results.get("closed_trades", []):
        entry_ts = pd.Timestamp(t.get("entry_time"))
        if entry_ts.tzinfo is None:
            entry_ts = entry_ts.tz_localize("UTC")
        pips = float(t.get("pips", 0.0))
        trades.append({
            "strategy": "v44_ny",
            "entry_time": entry_ts.tz_convert("UTC"),
            "side": str(t.get("side", "")).lower(),
            "pips": pips,
            "usd": float(t.get("usd", 0.0)),
            "profitable": pips > 0,
        })
    return trades


# ── Step 3: label M1 bars with trade outcomes ────────────────────────

def label_bars(m1_features: pd.DataFrame, trades: list[dict]) -> pd.DataFrame:
    """Add per-strategy label columns to the M1 features DataFrame.

    For each strategy, we match each trade's entry_time to the nearest M1 bar
    and mark that bar with the strategy name and profit flag.
    """
    out = m1_features.copy()
    out["v14_entry"] = False
    out["v14_win"] = False
    out["london_v2_entry"] = False
    out["london_v2_win"] = False
    out["v44_entry"] = False
    out["v44_win"] = False

    time_idx = pd.DatetimeIndex(out["time"])

    for t in trades:
        entry = pd.Timestamp(t["entry_time"])
        if entry.tzinfo is None:
            entry = entry.tz_localize("UTC")
        idx = time_idx.get_indexer([entry], method="ffill")[0]
        if idx < 0:
            idx = time_idx.get_indexer([entry], method="bfill")[0]
        if idx < 0:
            continue
        strat = t["strategy"]
        won = t["profitable"]
        if strat == "v14":
            out.loc[out.index[idx], "v14_entry"] = True
            if won:
                out.loc[out.index[idx], "v14_win"] = True
        elif strat == "london_v2":
            out.loc[out.index[idx], "london_v2_entry"] = True
            if won:
                out.loc[out.index[idx], "london_v2_win"] = True
        elif strat == "v44_ny":
            out.loc[out.index[idx], "v44_entry"] = True
            if won:
                out.loc[out.index[idx], "v44_win"] = True

    return out


# ── Step 4: threshold analysis ───────────────────────────────────────

def _safe_pctiles(series: pd.Series) -> dict:
    s = series.dropna()
    if s.empty:
        return {"count": 0}
    return {
        "count": int(len(s)),
        "mean": round(float(s.mean()), 4),
        "std": round(float(s.std()), 4),
        "p10": round(float(s.quantile(0.10)), 4),
        "p25": round(float(s.quantile(0.25)), 4),
        "p50": round(float(s.quantile(0.50)), 4),
        "p75": round(float(s.quantile(0.75)), 4),
        "p90": round(float(s.quantile(0.90)), 4),
        "min": round(float(s.min()), 4),
        "max": round(float(s.max()), 4),
    }


def _regime_breakdown(subset: pd.DataFrame) -> dict:
    total = len(subset)
    if total == 0:
        return {"total": 0}
    ranging = int((subset["bb_regime"] == "ranging").sum())
    return {
        "total": total,
        "ranging_pct": round(100 * ranging / total, 1),
        "trending_pct": round(100 * (total - ranging) / total, 1),
        "h1_aligned_pct": round(100 * subset["h1_aligned"].sum() / total, 1),
    }


def analyze_thresholds(labeled: pd.DataFrame) -> dict:
    """Compute indicator distributions at profitable entry bars per strategy."""
    features = ["adx", "m5_slope_abs", "bb_width_pctile", "atr"]
    results: dict[str, Any] = {}

    for strat, entry_col, win_col in [
        ("v14", "v14_entry", "v14_win"),
        ("london_v2", "london_v2_entry", "london_v2_win"),
        ("v44_ny", "v44_entry", "v44_win"),
    ]:
        all_entries = labeled[labeled[entry_col] == True]
        wins = labeled[labeled[win_col] == True]
        losses = all_entries[~all_entries.index.isin(wins.index)]

        strat_result: dict[str, Any] = {
            "entries": int(len(all_entries)),
            "wins": int(len(wins)),
            "losses": int(len(losses)),
            "win_rate": round(100 * len(wins) / max(1, len(all_entries)), 1),
        }

        strat_result["regime_at_wins"] = _regime_breakdown(wins)
        strat_result["regime_at_losses"] = _regime_breakdown(losses)
        strat_result["regime_at_all_entries"] = _regime_breakdown(all_entries)

        for feat in features:
            strat_result[f"{feat}_at_wins"] = _safe_pctiles(wins[feat])
            strat_result[f"{feat}_at_losses"] = _safe_pctiles(losses[feat])
            strat_result[f"{feat}_at_all_entries"] = _safe_pctiles(all_entries[feat])

        results[strat] = strat_result

    # Global distributions for context
    valid = labeled.dropna(subset=["adx", "m5_slope_abs", "bb_width_pctile"])
    results["global"] = {
        "total_bars": int(len(valid)),
    }
    for feat in features:
        results["global"][feat] = _safe_pctiles(valid[feat])

    # Suggested thresholds based on profitable-entry clustering
    suggestions: dict[str, Any] = {}

    v14w = labeled[labeled["v14_win"] == True]
    v44w = labeled[labeled["v44_win"] == True]
    ldnw = labeled[labeled["london_v2_win"] == True]

    if not v14w.empty:
        suggestions["mean_reversion"] = {
            "adx_max": round(float(v14w["adx"].quantile(0.75)), 1),
            "m5_slope_abs_max": round(float(v14w["m5_slope_abs"].quantile(0.75)), 3),
            "bb_width_pctile_max": round(float(v14w["bb_width_pctile"].quantile(0.75)), 3),
            "note": "V14 wins cluster in low-ADX, low-slope, low-width environments",
        }

    if not v44w.empty:
        suggestions["momentum"] = {
            "adx_min": round(float(v44w["adx"].quantile(0.25)), 1),
            "m5_slope_abs_min": round(float(v44w["m5_slope_abs"].quantile(0.25)), 3),
            "h1_aligned_pct_at_wins": round(100 * v44w["h1_aligned"].sum() / len(v44w), 1),
            "note": "V44 wins cluster in moderate-to-high ADX with directional slope",
        }

    if not ldnw.empty:
        suggestions["breakout"] = {
            "adx_min": round(float(ldnw["adx"].quantile(0.25)), 1),
            "bb_width_pctile_min": round(float(ldnw["bb_width_pctile"].quantile(0.25)), 3),
            "m5_slope_abs_min": round(float(ldnw["m5_slope_abs"].quantile(0.25)), 3),
            "note": "London V2 wins cluster around volatility expansion / breakout",
        }

    results["suggested_thresholds"] = suggestions

    return results


# ── main ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regime threshold analysis for eligibility router")
    p.add_argument("--input", required=True, help="M1 OANDA CSV")
    p.add_argument("--output", required=True, help="Output JSON path")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_csv = str(Path(args.input).resolve())
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loading M1 data from {input_csv} ...")
    m1 = _load_m1(input_csv)
    print(f"       {len(m1)} bars loaded ({m1['time'].iloc[0]} → {m1['time'].iloc[-1]})")

    print("[2/5] Computing regime features (BB, ADX, M5 slope, H1 alignment) ...")
    featured = compute_regime_features(m1)
    print(f"       Features computed. NaN-free bars: {featured.dropna(subset=['adx','m5_slope_abs','bb_width_pctile']).shape[0]}")

    print("[3/5] Running standalone backtests ...")
    print("       V14 Tokyo ...")
    v14_trades = _run_v14(input_csv)
    print(f"       → {len(v14_trades)} trades ({sum(1 for t in v14_trades if t['profitable'])} wins)")

    print("       London V2 ...")
    v2_trades = _run_london_v2(m1)
    print(f"       → {len(v2_trades)} trades ({sum(1 for t in v2_trades if t['profitable'])} wins)")

    print("       V44 NY ...")
    v44_trades = _run_v44(input_csv)
    print(f"       → {len(v44_trades)} trades ({sum(1 for t in v44_trades if t['profitable'])} wins)")

    all_trades = v14_trades + v2_trades + v44_trades

    print("[4/5] Labeling bars with trade outcomes ...")
    labeled = label_bars(featured, all_trades)

    print("[5/5] Analyzing thresholds ...")
    results = analyze_thresholds(labeled)

    dataset_name = Path(input_csv).stem
    results["_meta"] = {
        "dataset": dataset_name,
        "bars": int(len(m1)),
        "date_range": {
            "start": m1["time"].iloc[0].isoformat(),
            "end": m1["time"].iloc[-1].isoformat(),
        },
        "trades_total": len(all_trades),
        "v14_trades": len(v14_trades),
        "london_v2_trades": len(v2_trades),
        "v44_trades": len(v44_trades),
    }

    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nResults written to {out_path}")

    # Print compact summary
    print("\n" + "=" * 70)
    print("REGIME THRESHOLD SUMMARY")
    print("=" * 70)
    for strat in ["v14", "london_v2", "v44_ny"]:
        s = results.get(strat, {})
        print(f"\n{strat}: {s.get('entries',0)} entries, {s.get('wins',0)} wins ({s.get('win_rate',0):.1f}%)")
        for feat in ["adx", "m5_slope_abs", "bb_width_pctile"]:
            w = s.get(f"{feat}_at_wins", {})
            if w.get("count", 0) > 0:
                print(f"  {feat:20s} wins  p25={w['p25']:8.3f}  p50={w['p50']:8.3f}  p75={w['p75']:8.3f}")
            l = s.get(f"{feat}_at_losses", {})
            if l.get("count", 0) > 0:
                print(f"  {feat:20s} loss  p25={l['p25']:8.3f}  p50={l['p50']:8.3f}  p75={l['p75']:8.3f}")
        rw = s.get("regime_at_wins", {})
        rl = s.get("regime_at_losses", {})
        if rw.get("total", 0) > 0:
            print(f"  regime at wins:  ranging={rw.get('ranging_pct',0):.0f}%  trending={rw.get('trending_pct',0):.0f}%  h1_aligned={rw.get('h1_aligned_pct',0):.0f}%")
        if rl.get("total", 0) > 0:
            print(f"  regime at losses: ranging={rl.get('ranging_pct',0):.0f}%  trending={rl.get('trending_pct',0):.0f}%  h1_aligned={rl.get('h1_aligned_pct',0):.0f}%")

    sugg = results.get("suggested_thresholds", {})
    if sugg:
        print("\n" + "-" * 70)
        print("SUGGESTED STARTING THRESHOLDS")
        print("-" * 70)
        for regime, vals in sugg.items():
            print(f"\n  {regime}:")
            for k, v in vals.items():
                if k != "note":
                    print(f"    {k}: {v}")
            print(f"    ({vals.get('note', '')})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
