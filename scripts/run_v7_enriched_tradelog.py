#!/usr/bin/env python3
"""Post-processing enrichment of phase3_v7_pfdd_defended trade_log.csv.

Adds MFE/MAE excursion, path snapshots, market context at entry,
sub-strategy classification, and prior-trade context — all computed
from M1 price data without re-running the engine.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

TRADE_LOG = ROOT / "research_out/phase3_v7_pfdd_defended_real/trade_log.csv"
M1_DATA = ROOT / "research_out/USDJPY_M1_OANDA_1000k.csv"
OUTPUT = ROOT / "research_out/phase3_v7_pfdd_defended_real/v7_enriched_trade_log.csv"

PIP = 0.01  # USDJPY pip size


# ---------------------------------------------------------------------------
# Sub-strategy classification from entry hour
# ---------------------------------------------------------------------------

def classify_sub_strategy(trade: pd.Series) -> str:
    """Classify sub-strategy using exit_reason and entry_bar.

    - exit_reason == "oracle" → v44_ny (214 trades)
    - exit_reason == "phase3" → london_setup_d_l1 (68, L1 incremental exit)
    - entry_bar == 0 AND exit_bar == 0 → london_setup_a (28, bar indices not set by runner)
    - remainder → tokyo_v14 (88 trades)
    """
    reason = str(trade["exit_reason"])
    if reason == "oracle":
        return "v44_ny"
    if reason == "phase3":
        return "london_setup_d_l1"
    if int(trade["entry_bar"]) == 0 and int(trade["exit_bar"]) == 0:
        return "london_setup_a"
    return "tokyo_v14"


# ---------------------------------------------------------------------------
# Indicator helpers
# ---------------------------------------------------------------------------

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


# ---------------------------------------------------------------------------
# Pre-compute M5 and M15 indicators
# ---------------------------------------------------------------------------

def build_m5_indicators(m1: pd.DataFrame) -> pd.DataFrame:
    """Resample M1 to M5 and compute indicators."""
    m1_ts = m1.set_index("time_utc")
    m5 = m1_ts.resample("5min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
    }).dropna()
    m5["atr14"] = atr(m5["high"], m5["low"], m5["close"], 14)
    m5["ema9"] = ema(m5["close"], 9)
    m5["ema21"] = ema(m5["close"], 21)
    m5["rsi14"] = rsi(m5["close"], 14)
    m5["bar_range"] = m5["high"] - m5["low"]
    return m5


def build_m15_indicators(m1: pd.DataFrame) -> pd.DataFrame:
    m1_ts = m1.set_index("time_utc")
    m15 = m1_ts.resample("15min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
    }).dropna()
    m15["ema50"] = ema(m15["close"], 50)
    return m15


def lookup_indicator(indicator_df: pd.DataFrame, entry_time: pd.Timestamp,
                     column: str) -> float:
    """Get the most recent indicator value at or before entry_time."""
    mask = indicator_df.index <= entry_time
    if not mask.any():
        return np.nan
    return float(indicator_df.loc[mask, column].iloc[-1])


# ---------------------------------------------------------------------------
# Per-trade enrichment
# ---------------------------------------------------------------------------

def enrich_trade(trade: pd.Series, m1: pd.DataFrame, m5: pd.DataFrame,
                 m15: pd.DataFrame, m1_time_index: pd.DatetimeIndex) -> dict:
    entry_bar = int(trade["entry_bar"])
    exit_bar = int(trade["exit_bar"])
    entry_price = float(trade["entry_price"])
    exit_price = float(trade["exit_price"])
    direction = trade["direction"]
    is_long = direction == "long"
    # Recompute pnl_pips from prices (oracle trades have pnl_pips=0 in trade_log)
    if is_long:
        pnl_pips = (exit_price - entry_price) / PIP
    else:
        pnl_pips = (entry_price - exit_price) / PIP
    entry_time = pd.Timestamp(trade["entry_time"])
    exit_time = pd.Timestamp(trade["exit_time"])

    # For trades with entry_bar=0 (london_setup_a), resolve via timestamp
    if entry_bar == 0 and exit_bar == 0:
        entry_idx = m1_time_index.searchsorted(entry_time, side="left")
        exit_idx = m1_time_index.searchsorted(exit_time, side="left")
        entry_bar = int(min(entry_idx, len(m1) - 1))
        exit_bar = int(min(exit_idx, len(m1) - 1))

    # If entry_bar == exit_bar, extend by 1 so we have at least one bar
    if exit_bar <= entry_bar:
        exit_bar = entry_bar + 1

    # Clamp to data bounds
    max_bar = len(m1) - 1
    entry_bar = min(entry_bar, max_bar)
    exit_bar = min(exit_bar, max_bar)

    # Extract M1 segment for this trade
    seg = m1.iloc[entry_bar:exit_bar + 1]

    result = {}

    # --- EXCURSION DATA ---
    if len(seg) > 0:
        if is_long:
            favorable = (seg["high"].values - entry_price) / PIP
            adverse = (entry_price - seg["low"].values) / PIP
        else:
            favorable = (entry_price - seg["low"].values) / PIP
            adverse = (seg["high"].values - entry_price) / PIP

        # Floor at 0: entry price may differ from M1 mid due to spread
        favorable = np.maximum(favorable, 0.0)
        adverse = np.maximum(adverse, 0.0)

        mfe = float(np.max(favorable))
        mae = float(np.max(adverse))
        mfe_idx = int(np.argmax(favorable))
        mae_idx = int(np.argmax(adverse))

        result["mfe_pips"] = round(mfe, 2)
        result["mae_pips"] = round(mae, 2)
        result["mfe_bar"] = mfe_idx
        result["mae_bar"] = mae_idx

        # Time to MFE/MAE
        if len(seg) > mfe_idx:
            result["time_to_mfe_min"] = mfe_idx  # M1 bars = minutes
        else:
            result["time_to_mfe_min"] = 0
        if len(seg) > mae_idx:
            result["time_to_mae_min"] = mae_idx
        else:
            result["time_to_mae_min"] = 0

        # MFE capture
        result["mfe_capture"] = round(pnl_pips / mfe, 4) if mfe > 0 else 0.0
    else:
        result.update({
            "mfe_pips": 0.0, "mae_pips": 0.0, "mfe_bar": 0, "mae_bar": 0,
            "time_to_mfe_min": 0, "time_to_mae_min": 0, "mfe_capture": 0.0,
        })

    # --- PATH DATA ---
    def pnl_at_offset(minutes: int) -> float:
        idx = entry_bar + minutes
        if idx > max_bar or idx > exit_bar:
            return np.nan
        bar = m1.iloc[idx]
        mid = float(bar["close"])
        if is_long:
            return round((mid - entry_price) / PIP, 2)
        else:
            return round((entry_price - mid) / PIP, 2)

    result["pnl_at_5min"] = pnl_at_offset(5)
    result["pnl_at_15min"] = pnl_at_offset(15)
    result["pnl_at_30min"] = pnl_at_offset(30)
    result["pnl_at_60min"] = pnl_at_offset(60)

    # Max PnL in first 30 min
    end_30 = min(entry_bar + 30, exit_bar, max_bar)
    seg30 = m1.iloc[entry_bar:end_30 + 1]
    if len(seg30) > 0:
        if is_long:
            result["max_pnl_first_30min"] = round(
                float(np.max((seg30["high"].values - entry_price) / PIP)), 2)
        else:
            result["max_pnl_first_30min"] = round(
                float(np.max((entry_price - seg30["low"].values) / PIP)), 2)
    else:
        result["max_pnl_first_30min"] = 0.0

    # Crossed breakeven
    if len(seg) > 1:
        if is_long:
            went_positive = np.any(seg["high"].values[1:] > entry_price)
            went_negative = np.any(seg["low"].values[1:] < entry_price)
        else:
            went_positive = np.any(seg["low"].values[1:] < entry_price)
            went_negative = np.any(seg["high"].values[1:] > entry_price)
        result["crossed_breakeven"] = bool(went_positive and went_negative)
    else:
        result["crossed_breakeven"] = False

    # --- MARKET CONTEXT AT ENTRY ---
    result["m5_atr14"] = round(lookup_indicator(m5, entry_time, "atr14") / PIP, 2)
    result["m5_bar_range"] = round(lookup_indicator(m5, entry_time, "bar_range") / PIP, 2)
    m5_ema9 = lookup_indicator(m5, entry_time, "ema9")
    m5_ema21 = lookup_indicator(m5, entry_time, "ema21")
    result["m5_ema9"] = round(m5_ema9, 5) if not np.isnan(m5_ema9) else np.nan
    result["m5_ema21"] = round(m5_ema21, 5) if not np.isnan(m5_ema21) else np.nan
    result["m5_ema_spread"] = (
        round((m5_ema9 - m5_ema21) / PIP, 2)
        if not (np.isnan(m5_ema9) or np.isnan(m5_ema21)) else np.nan
    )
    result["m15_ema50"] = round(lookup_indicator(m15, entry_time, "ema50"), 5)
    result["m5_rsi14"] = round(lookup_indicator(m5, entry_time, "rsi14"), 2)

    result["hour_utc"] = entry_time.hour
    result["day_of_week"] = entry_time.dayofweek  # 0=Mon..4=Fri

    # --- SUB-STRATEGY ---
    result["sub_strategy"] = classify_sub_strategy(trade)
    result["direction"] = direction
    result["pnl_pips_calc"] = round(pnl_pips, 2)

    return result


# ---------------------------------------------------------------------------
# Prior trade context (computed after all trades enriched)
# ---------------------------------------------------------------------------

def add_prior_trade_context(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling prior-trade context columns."""
    pnl = df["pnl_pips_calc"].values
    n = len(pnl)

    prev_pnl = np.full(n, np.nan)
    prev_winner = np.full(n, np.nan)
    consec_losses = np.zeros(n, dtype=int)
    trades_last_2hrs = np.zeros(n, dtype=int)
    net_pnl_last5 = np.full(n, np.nan)

    entry_times = pd.to_datetime(df["entry_time"], utc=True)

    for i in range(n):
        if i > 0:
            prev_pnl[i] = pnl[i - 1]
            prev_winner[i] = 1 if pnl[i - 1] > 0 else 0

        # Consecutive losses
        cl = 0
        for j in range(i - 1, -1, -1):
            if pnl[j] <= 0:
                cl += 1
            else:
                break
        consec_losses[i] = cl

        # Trades in last 2 hours
        t_now = entry_times.iloc[i]
        t_2h = t_now - pd.Timedelta(hours=2)
        count = 0
        for j in range(i - 1, -1, -1):
            if entry_times.iloc[j] >= t_2h:
                count += 1
            else:
                break
        trades_last_2hrs[i] = count

        # Net PnL last 5 trades
        if i >= 5:
            net_pnl_last5[i] = float(np.sum(pnl[i - 5:i]))

    df["prev_trade_pnl"] = np.round(prev_pnl, 2)
    df["prev_trade_winner"] = pd.array(prev_winner, dtype=pd.Int64Dtype())
    df["consecutive_losses"] = consec_losses
    df["trades_last_2hrs"] = trades_last_2hrs
    df["net_pnl_last_5trades"] = np.round(net_pnl_last5, 2)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.perf_counter()

    print("Loading M1 data...", flush=True)
    m1 = pd.read_csv(M1_DATA)
    m1["time_utc"] = pd.to_datetime(m1["time"], utc=True)
    print(f"  {len(m1):,} M1 bars loaded")

    print("Building M5 indicators...", flush=True)
    m5 = build_m5_indicators(m1)
    print(f"  {len(m5):,} M5 bars")

    print("Building M15 indicators...", flush=True)
    m15 = build_m15_indicators(m1)
    print(f"  {len(m15):,} M15 bars")

    print("Loading trade log...", flush=True)
    trades = pd.read_csv(TRADE_LOG)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True)
    print(f"  {len(trades)} trades")

    print("Enriching trades...", flush=True)
    m1_time_index = m1["time_utc"]
    enrichments = []
    for idx in range(len(trades)):
        row = trades.iloc[idx]
        enrichments.append(enrich_trade(row, m1, m5, m15, m1_time_index))
        if (idx + 1) % 50 == 0:
            print(f"  {idx + 1}/{len(trades)}", flush=True)

    enrich_df = pd.DataFrame(enrichments)
    result = pd.concat([trades.reset_index(drop=True), enrich_df], axis=1)

    print("Adding prior trade context...", flush=True)
    result = add_prior_trade_context(result)

    # Drop duplicate 'direction' column (already in original)
    if result.columns.tolist().count("direction") > 1:
        cols = result.columns.tolist()
        # Keep the first 'direction', remove the enrichment one
        seen = False
        drop_idx = []
        for i, c in enumerate(cols):
            if c == "direction":
                if seen:
                    drop_idx.append(i)
                seen = True
        if drop_idx:
            result = result.iloc[:, [i for i in range(len(cols)) if i not in drop_idx]]

    # Verify no nulls in enrichment columns
    enrich_cols = list(enrichments[0].keys())
    enrich_cols.remove("direction")  # already in original
    null_counts = result[enrich_cols].isnull().sum()
    nulls = null_counts[null_counts > 0]
    if len(nulls) > 0:
        print(f"\n  WARNING: nulls found in enrichment columns:")
        for col, cnt in nulls.items():
            print(f"    {col}: {cnt} nulls")
        # Path data nulls are expected (trade ended before snapshot time)
        path_cols = {"pnl_at_5min", "pnl_at_15min", "pnl_at_30min", "pnl_at_60min"}
        non_path = nulls.drop(labels=list(path_cols & set(nulls.index)), errors="ignore")
        if len(non_path) > 0:
            print("  WARNING: unexpected nulls in non-path columns!")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Output: {OUTPUT}")
    print(f"  {len(result)} trades × {len(result.columns)} columns")

    # Quick summary
    print("\n--- Sub-strategy breakdown ---")
    for sub, grp in result.groupby("sub_strategy"):
        winners = grp[grp["pnl_pips_calc"] > 0]
        win_capture = winners["mfe_capture"].mean() if len(winners) > 0 else 0.0
        print(f"  {sub}: {len(grp)} trades ({len(winners)}W/{len(grp)-len(winners)}L), "
              f"avg MFE={grp['mfe_pips'].mean():.1f}p, "
              f"avg MAE={grp['mae_pips'].mean():.1f}p, "
              f"winner capture={win_capture:.2f}")


if __name__ == "__main__":
    main()
