"""
Chop detector analysis for USDJPY Trial 10.

Joins trade outcomes with M1 calibration data at entry time,
then tests multiple chop-detection approaches to find the best
signal for pausing entries during choppy conditions.
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CAL_PATH = ROOT / "research_out" / "trial10_indicator_calibration_250k.csv"
TRADES_PATH = ROOT / "research_out" / "trial10_250k_pause1_trades.csv"


def load_data():
    cal = pd.read_csv(CAL_PATH, parse_dates=["time"])
    trades = pd.read_csv(TRADES_PATH, parse_dates=["signal_time", "entry_time", "exit_time"])
    return cal, trades


def merge_trades_with_conditions(cal: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """For each trade, find the calibration bar at signal_time."""
    # Round signal_time to nearest minute to match cal.time
    trades = trades.copy()
    trades["signal_time_rounded"] = trades["signal_time"].dt.floor("min")

    cal_cols = ["time", "m5_bucket", "m5_gap_pips", "m5_atr_pips",
                "m5_ema9_slope_pips_per_bar", "m5_slope_aligned",
                "m1_compressing", "m1_bucket", "m1_aligned"]
    merged = trades.merge(
        cal[cal_cols],
        left_on="signal_time_rounded",
        right_on="time",
        how="left",
        suffixes=("_trade", "_cal"),
    )
    # Use calibration versions of overlapping columns
    for col in ["m5_bucket", "m1_bucket"]:
        if f"{col}_cal" in merged.columns:
            merged[col] = merged[f"{col}_cal"]
    print(f"Trades total: {len(trades)}")
    print(f"Merged (have cal data): {merged['time'].notna().sum()}")
    print(f"Missing cal data: {merged['time'].isna().sum()}")
    return merged


def evaluate_approach(merged: pd.DataFrame, mask: pd.Series, name: str) -> dict:
    """Evaluate a chop detection approach given a boolean mask (True = flagged as chop)."""
    total = len(merged)
    baseline_pnl = merged["profit_usd"].sum()

    blocked = merged[mask]
    allowed = merged[~mask]

    n_blocked = len(blocked)
    n_allowed = len(allowed)

    if n_blocked == 0:
        return {
            "approach": name,
            "blocked_trades": 0,
            "blocked_pct": 0,
            "allowed_trades": total,
            "avg_pnl_blocked": np.nan,
            "avg_pnl_allowed": np.nan,
            "baseline_pnl": baseline_pnl,
            "allowed_pnl": baseline_pnl,
            "pnl_delta": 0,
            "blocked_losers": 0,
            "blocked_winners": 0,
            "true_positive_rate": np.nan,
            "false_positive_rate": np.nan,
        }

    avg_blocked = blocked["profit_usd"].mean()
    avg_allowed = allowed["profit_usd"].mean() if n_allowed > 0 else 0

    allowed_pnl = allowed["profit_usd"].sum()
    pnl_delta = allowed_pnl - baseline_pnl  # positive = improvement

    # "Losers" = trades that lost money, "Winners" = trades that made money
    total_losers = (merged["profit_usd"] < 0).sum()
    total_winners = (merged["profit_usd"] > 0).sum()

    blocked_losers = (blocked["profit_usd"] < 0).sum()
    blocked_winners = (blocked["profit_usd"] > 0).sum()

    tp_rate = blocked_losers / total_losers if total_losers > 0 else 0
    fp_rate = blocked_winners / total_winners if total_winners > 0 else 0

    return {
        "approach": name,
        "blocked_trades": n_blocked,
        "blocked_pct": round(100 * n_blocked / total, 1),
        "allowed_trades": n_allowed,
        "avg_pnl_blocked": round(avg_blocked, 4),
        "avg_pnl_allowed": round(avg_allowed, 4),
        "baseline_pnl": round(baseline_pnl, 2),
        "allowed_pnl": round(allowed_pnl, 2),
        "pnl_delta": round(pnl_delta, 2),
        "blocked_losers": blocked_losers,
        "blocked_winners": blocked_winners,
        "true_positive_rate": round(tp_rate, 4),
        "false_positive_rate": round(fp_rate, 4),
    }


def main():
    cal, trades = load_data()
    merged = merge_trades_with_conditions(cal, trades)

    # Drop rows without calibration data
    merged = merged[merged["time"].notna()].copy()
    print(f"\nAnalyzing {len(merged)} trades with calibration data")
    print(f"Baseline P&L: ${merged['profit_usd'].sum():.2f}")
    print(f"Total losers: {(merged['profit_usd'] < 0).sum()}, "
          f"Total winners: {(merged['profit_usd'] > 0).sum()}, "
          f"Breakeven: {(merged['profit_usd'] == 0).sum()}")
    print()

    results = []

    # ─── A: M5 bucket = "weak" ───
    mask_weak = merged["m5_bucket"] == "weak"
    results.append(evaluate_approach(merged, mask_weak, "A: m5_bucket=weak"))

    # ─── B: M5 EMA gap compression ───
    for thresh in [0.5, 1.0, 1.5, 2.0, 3.0]:
        mask = merged["m5_gap_pips"].abs() < thresh
        results.append(evaluate_approach(merged, mask, f"B: |m5_gap|<{thresh}p"))

    # ─── C: M5 slope flat ───
    for thresh in [0.01, 0.02, 0.05, 0.1]:
        mask = merged["m5_ema9_slope_pips_per_bar"].abs() < thresh
        results.append(evaluate_approach(merged, mask, f"C: |m5_slope|<{thresh}"))

    # ─── D: Combined weak + gap compressed ───
    for gap_thresh in [1.0, 2.0, 3.0]:
        mask = (merged["m5_bucket"] == "weak") & (merged["m5_gap_pips"].abs() < gap_thresh)
        results.append(evaluate_approach(merged, mask, f"D: weak+|gap|<{gap_thresh}p"))

    # ─── E: Rolling stop rate ───
    # Sort by entry time, compute rolling stop rate
    merged_sorted = merged.sort_values("entry_time").reset_index(drop=True)
    is_stop = (merged_sorted["exit_reason"] == "initial_stop").astype(float)
    for window in [5, 10, 15, 20]:
        for stop_thresh in [0.5, 0.6, 0.7]:
            rolling_rate = is_stop.rolling(window, min_periods=window).mean().shift(1)
            mask = rolling_rate > stop_thresh
            mask = mask.fillna(False)
            results.append(evaluate_approach(
                merged_sorted, mask,
                f"E: roll_{window}_stop>{int(stop_thresh*100)}%"
            ))

    # ─── F: M5 ATR low ───
    # First check ATR distribution
    atr_vals = merged["m5_atr_pips"].dropna()
    print(f"M5 ATR stats: mean={atr_vals.mean():.2f}, "
          f"median={atr_vals.median():.2f}, "
          f"p10={atr_vals.quantile(0.1):.2f}, "
          f"p25={atr_vals.quantile(0.25):.2f}, "
          f"p75={atr_vals.quantile(0.75):.2f}")

    for thresh in [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]:
        mask = merged["m5_atr_pips"] < thresh
        results.append(evaluate_approach(merged, mask, f"F: m5_atr<{thresh}p"))

    # ─── G: M1 compression ───
    mask_compress = merged["m1_compressing"] == True
    results.append(evaluate_approach(merged, mask_compress, "G: m1_compressing"))

    # ─── H: Combined: weak + slope flat ───
    for slope_thresh in [0.02, 0.05]:
        mask = ((merged["m5_bucket"] == "weak") &
                (merged["m5_ema9_slope_pips_per_bar"].abs() < slope_thresh))
        results.append(evaluate_approach(merged, mask, f"H: weak+|slope|<{slope_thresh}"))

    # ─── I: M5 slope disagrees (counter-trend momentum) ───
    mask_disagree = merged["m5_slope_aligned"] == False
    results.append(evaluate_approach(merged, mask_disagree, "I: m5_slope_not_aligned"))

    # ─── J: Combined: low ATR + weak bucket ───
    for atr_thresh in [5.0, 6.0, 7.0]:
        mask = (merged["m5_bucket"] == "weak") & (merged["m5_atr_pips"] < atr_thresh)
        results.append(evaluate_approach(merged, mask, f"J: weak+atr<{atr_thresh}p"))

    # ─── K: Combined: low ATR + gap compressed ───
    for atr_thresh in [5.0, 6.0, 7.0]:
        for gap_thresh in [2.0, 3.0]:
            mask = ((merged["m5_atr_pips"] < atr_thresh) &
                    (merged["m5_gap_pips"].abs() < gap_thresh))
            results.append(evaluate_approach(
                merged, mask,
                f"K: atr<{atr_thresh}+|gap|<{gap_thresh}p"
            ))

    # ─── L: M1 bucket = dampen (already weakish) ───
    mask_dampen = merged["m1_bucket"] == "dampen"
    results.append(evaluate_approach(merged, mask_dampen, "L: m1_bucket=dampen"))

    # ─── M: Slope flat + M1 compressing (true chop signature) ───
    for slope_thresh in [0.02, 0.05]:
        mask = ((merged["m5_ema9_slope_pips_per_bar"].abs() < slope_thresh) &
                (merged["m1_compressing"] == True))
        results.append(evaluate_approach(
            merged, mask,
            f"M: |slope|<{slope_thresh}+m1_compress"
        ))

    # ─── Build results table ───
    df = pd.DataFrame(results)

    # Sort by P&L delta (best improvement first)
    df = df.sort_values("pnl_delta", ascending=False).reset_index(drop=True)

    # Compute "edge ratio" = losers blocked per winner blocked
    df["loser_per_winner"] = (df["blocked_losers"] / df["blocked_winners"].replace(0, np.nan)).round(2)

    print("\n" + "=" * 140)
    print("CHOP DETECTION ANALYSIS — Sorted by P&L Improvement")
    print("=" * 140)

    display_cols = [
        "approach", "blocked_trades", "blocked_pct", "avg_pnl_blocked",
        "avg_pnl_allowed", "pnl_delta", "blocked_losers", "blocked_winners",
        "loser_per_winner", "true_positive_rate", "false_positive_rate",
    ]
    pd.set_option("display.max_colwidth", 40)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 100)
    print(df[display_cols].to_string(index=False))

    # ─── Top 10 detail ───
    print("\n" + "=" * 100)
    print("TOP 10 APPROACHES BY P&L IMPROVEMENT")
    print("=" * 100)
    for i, row in df.head(10).iterrows():
        print(f"\n#{i+1}: {row['approach']}")
        print(f"  Blocks {row['blocked_trades']} trades ({row['blocked_pct']}%)")
        print(f"  Avg P&L blocked: ${row['avg_pnl_blocked']:.4f}  |  Avg P&L allowed: ${row['avg_pnl_allowed']:.4f}")
        print(f"  P&L delta: ${row['pnl_delta']:.2f}  (baseline ${row['baseline_pnl']:.2f} -> ${row['allowed_pnl']:.2f})")
        print(f"  Blocked {row['blocked_losers']} losers, {row['blocked_winners']} winners  "
              f"(ratio: {row['loser_per_winner']})")
        print(f"  TP rate: {row['true_positive_rate']:.2%}  |  FP rate: {row['false_positive_rate']:.2%}")

    # ─── Selectivity ranking ───
    print("\n" + "=" * 100)
    print("SELECTIVITY RANKING — Approaches that block more losers than winners")
    print("(loser_per_winner > 1.0 and blocked_trades >= 100)")
    print("=" * 100)
    selective = df[(df["loser_per_winner"] > 1.0) & (df["blocked_trades"] >= 100)].copy()
    selective = selective.sort_values("pnl_delta", ascending=False)
    for _, row in selective.iterrows():
        ratio = row["loser_per_winner"]
        print(f"  {row['approach']:40s}  block {row['blocked_trades']:5d} ({row['blocked_pct']:5.1f}%)  "
              f"L/W={ratio:.2f}  delta=${row['pnl_delta']:+8.2f}  "
              f"TP={row['true_positive_rate']:.1%}  FP={row['false_positive_rate']:.1%}")

    # ─── Efficiency metric: pnl_delta per trade blocked ───
    print("\n" + "=" * 100)
    print("EFFICIENCY: P&L improvement per trade blocked (higher = better targeting)")
    print("(blocked_trades >= 50)")
    print("=" * 100)
    efficient = df[df["blocked_trades"] >= 50].copy()
    efficient["pnl_per_block"] = efficient["pnl_delta"] / efficient["blocked_trades"]
    efficient = efficient.sort_values("pnl_per_block", ascending=False)
    for _, row in efficient.head(15).iterrows():
        print(f"  {row['approach']:40s}  block {row['blocked_trades']:5d}  "
              f"delta=${row['pnl_delta']:+8.2f}  "
              f"$/block=${row['pnl_per_block']:.4f}  L/W={row['loser_per_winner']:.2f}")

    # ─── Drill into best approach: by entry_type ───
    print("\n" + "=" * 100)
    print("BEST APPROACH BREAKDOWN BY ENTRY TYPE")
    print("=" * 100)
    best = df.iloc[0]
    best_name = best["approach"]
    print(f"Best approach: {best_name}")

    # Reconstruct the best mask
    # (We'll just pick the top approach and rebuild it)
    # For simplicity, let's also show zone_entry vs tiered_pullback for the top 3
    for idx in range(min(3, len(df))):
        row = df.iloc[idx]
        approach_name = row["approach"]
        print(f"\n--- {approach_name} ---")
        # Rebuild mask (simplified — we know the pattern from the name)
        mask = _rebuild_mask(merged, approach_name)
        if mask is None:
            print("  (cannot reconstruct mask for breakdown)")
            continue
        for etype in ["zone_entry", "tiered_pullback"]:
            sub = merged[merged["entry_type"] == etype]
            sub_mask = mask[sub.index]
            blocked = sub[sub_mask]
            allowed = sub[~sub_mask]
            if len(blocked) == 0:
                print(f"  {etype}: 0 blocked")
                continue
            print(f"  {etype}: blocked {len(blocked)}/{len(sub)} "
                  f"(avg blocked ${blocked['profit_usd'].mean():.4f}, "
                  f"avg allowed ${allowed['profit_usd'].mean():.4f}, "
                  f"delta ${allowed['profit_usd'].sum() - sub['profit_usd'].sum():.2f})")


def _rebuild_mask(merged, name):
    """Reconstruct boolean mask from approach name."""
    if name == "A: m5_bucket=weak":
        return merged["m5_bucket"] == "weak"
    if name.startswith("B: |m5_gap|<"):
        t = float(name.split("<")[1].rstrip("p"))
        return merged["m5_gap_pips"].abs() < t
    if name.startswith("C: |m5_slope|<"):
        t = float(name.split("<")[1])
        return merged["m5_ema9_slope_pips_per_bar"].abs() < t
    if name.startswith("D: weak+|gap|<"):
        t = float(name.split("<")[1].rstrip("p"))
        return (merged["m5_bucket"] == "weak") & (merged["m5_gap_pips"].abs() < t)
    if name.startswith("F: m5_atr<"):
        t = float(name.split("<")[1].rstrip("p"))
        return merged["m5_atr_pips"] < t
    if name == "G: m1_compressing":
        return merged["m1_compressing"] == True
    if name.startswith("H: weak+|slope|<"):
        t = float(name.split("<")[1])
        return (merged["m5_bucket"] == "weak") & (merged["m5_ema9_slope_pips_per_bar"].abs() < t)
    if name == "I: m5_slope_not_aligned":
        return merged["m5_slope_aligned"] == False
    if name.startswith("J: weak+atr<"):
        t = float(name.split("<")[1].rstrip("p"))
        return (merged["m5_bucket"] == "weak") & (merged["m5_atr_pips"] < t)
    if name.startswith("K: atr<"):
        parts = name.replace("K: atr<", "").split("+|gap|<")
        atr_t = float(parts[0])
        gap_t = float(parts[1].rstrip("p"))
        return (merged["m5_atr_pips"] < atr_t) & (merged["m5_gap_pips"].abs() < gap_t)
    if name == "L: m1_bucket=dampen":
        return merged["m1_bucket"] == "dampen"
    if name.startswith("M: |slope|<"):
        t = float(name.split("<")[1].split("+")[0])
        return ((merged["m5_ema9_slope_pips_per_bar"].abs() < t) &
                (merged["m1_compressing"] == True))
    return None


if __name__ == "__main__":
    main()
