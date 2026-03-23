#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.regime_gate import evaluate_regime_gate
from scripts.backtest_trial10 import (
    PIP_SIZE,
    build_profile,
    compute_conviction_for_bar,
    hour_et,
    load_m1,
    map_features_to_m1,
    prepare_features,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Trial #10 indicator calibration CSV from M1 OHLC data.")
    p.add_argument(
        "--in",
        dest="input_csv",
        default="research_out/USDJPY_M1_OANDA_250k.csv",
        help="Input M1 CSV with columns time,open,high,low,close",
    )
    p.add_argument(
        "--out",
        default="research_out/trial10_indicator_calibration_250k.csv",
        help="Output CSV path",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for quick smoke runs",
    )
    return p.parse_args()


def _bars_since_directional_cross(diff: pd.Series, side: str) -> pd.Series:
    result: list[float] = []
    last_cross_idx: int | None = None
    vals = diff.astype(float).tolist()
    for i, cur in enumerate(vals):
        crossed = False
        if i > 0:
            prev = vals[i - 1]
            if side == "buy":
                crossed = cur > 0 and prev <= 0
            else:
                crossed = cur < 0 and prev >= 0
        if crossed:
            last_cross_idx = i
        if last_cross_idx is None:
            result.append(float("nan"))
        else:
            result.append(float(i - last_cross_idx))
    return pd.Series(result, index=diff.index)


def _m1_health_for_bar(m1f: pd.DataFrame, i: int, side: str) -> tuple[float, bool, bool, str]:
    ema5 = float(m1f["ema_5"].iat[i])
    ema9 = float(m1f["ema_9"].iat[i])
    spread_pips = abs(ema5 - ema9) / PIP_SIZE
    aligned = ema5 > ema9 if side == "buy" else ema5 < ema9
    compressing = False
    if i >= 3:
        recent = (
            (m1f["ema_5"].iloc[i - 3 : i + 1] - m1f["ema_9"].iloc[i - 3 : i + 1]).abs() / PIP_SIZE
        ).tolist()
        compressing = recent[1] < recent[0] and recent[2] < recent[1] and recent[3] < recent[2]
    if spread_pips < 0.23:
        compressing = True
    if (not aligned) or compressing:
        bucket = "dampen"
    elif spread_pips > 0.53:
        bucket = "confirm"
    else:
        bucket = "neutral"
    return spread_pips, aligned, compressing, bucket


def _zone_state_for_bar(m1f: pd.DataFrame, i: int, side: str, policy) -> tuple[bool, bool]:
    fast_col = f"ema_{int(policy.m1_zone_entry_ema_fast)}"
    slow_col = f"ema_{int(policy.m1_zone_entry_ema_slow)}"
    fast_now = float(m1f[fast_col].iat[i])
    slow_now = float(m1f[slow_col].iat[i])
    is_bull = side == "buy"
    aligned_now = fast_now > slow_now if is_bull else fast_now < slow_now
    recent_cross = False
    lookback = max(1, int(policy.zone_entry_max_cross_lookback_bars))
    for offset in range(1, min(lookback, i) + 1):
        prev_fast = float(m1f[fast_col].iat[i - offset])
        prev_slow = float(m1f[slow_col].iat[i - offset])
        curr_fast = float(m1f[fast_col].iat[i - offset + 1])
        curr_slow = float(m1f[slow_col].iat[i - offset + 1])
        if is_bull and prev_fast <= prev_slow and curr_fast > curr_slow:
            recent_cross = True
            break
        if (not is_bull) and prev_fast >= prev_slow and curr_fast < curr_slow:
            recent_cross = True
            break
    return aligned_now, recent_cross


def _resolve_trial10_stop_pips(profile, mapped: pd.DataFrame, i: int) -> float:
    atr_pips = mapped["atr_pips"].iat[i] if "atr_pips" in mapped.columns else None
    fallback_sl = float(getattr(profile.execution.policies[0], "sl_pips", 20.0))
    sl_cfg = getattr(profile.trade_management, "stop_loss", None)
    if sl_cfg is None or str(getattr(sl_cfg, "mode", "")).lower() != "atr" or pd.isna(atr_pips):
        return max(float(profile.risk.min_stop_pips), float(fallback_sl))
    atr_stop = min(
        float(atr_pips) * float(getattr(sl_cfg, "atr_multiplier", 1.5)),
        float(getattr(sl_cfg, "max_sl_pips", fallback_sl)),
    )
    return max(float(profile.risk.min_stop_pips), float(atr_stop))


def _ntz_metrics(snapshot: dict | None, price: float) -> tuple[str, float | None, bool]:
    if not snapshot:
        return "", None, False
    levels = snapshot.get("levels") or {}
    nearest_label = ""
    nearest_dist = None
    for label, level in levels.items():
        if level is None:
            continue
        dist = abs(float(price) - float(level)) / PIP_SIZE
        if nearest_dist is None or dist < nearest_dist:
            nearest_dist = dist
            nearest_label = str(label)
    return nearest_label, nearest_dist, bool(snapshot.get("enabled", False))


def main() -> None:
    args = parse_args()
    m1 = load_m1(args.input_csv)
    if args.limit is not None and args.limit > 0:
        m1 = m1.iloc[: int(args.limit)].copy()
    profile, policy = build_profile(argparse.Namespace(tier_periods="", tp1_close_pct=None))
    features = prepare_features(m1, policy)
    m1f = features["m1"]
    mapped = map_features_to_m1(m1f, features)
    m5 = features["m5"].copy()

    m1_diff = m1f["ema_5"].astype(float) - m1f["ema_9"].astype(float)
    m1f["bars_since_m1_cross_buy"] = _bars_since_directional_cross(m1_diff, "buy")
    m1f["bars_since_m1_cross_sell"] = _bars_since_directional_cross(m1_diff, "sell")
    mapped["bars_since_m1_cross_buy"] = m1f["bars_since_m1_cross_buy"]
    mapped["bars_since_m1_cross_sell"] = m1f["bars_since_m1_cross_sell"]

    m5_diff = m5["ema_9"].astype(float) - m5["ema_21"].astype(float)
    m5["bars_since_m5_recross_buy"] = _bars_since_directional_cross(m5_diff, "buy")
    m5["bars_since_m5_recross_sell"] = _bars_since_directional_cross(m5_diff, "sell")
    mapped["bars_since_m5_recross_buy"] = m5["bars_since_m5_recross_buy"].reindex(mapped.index, method="ffill")
    mapped["bars_since_m5_recross_sell"] = m5["bars_since_m5_recross_sell"].reindex(mapped.index, method="ffill")
    mapped["m5_ema9"] = m5["ema_9"].reindex(mapped.index, method="ffill")
    mapped["m5_ema21"] = m5["ema_21"].reindex(mapped.index, method="ffill")
    mapped["m5_ema9_slope_pips_per_bar"] = m5["ema9_slope_pips_per_bar"].reindex(mapped.index, method="ffill")
    mapped["m5_slope_aligned"] = m5["slope_aligned"].reindex(mapped.index, method="ffill")
    mapped["m5_slope_disagrees_ema21"] = m5["slope_disagrees_ema21"].reindex(mapped.index, method="ffill")

    ntz_buffer_pips = float(policy.ntz_buffer_pips)
    fib_boundary_buffer_pips = float(policy.intraday_fib_boundary_buffer_pips)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer: csv.DictWriter | None = None

        for i in range(len(mapped)):
            ts = mapped.index[i]
            close_price = float(mapped["close"].iat[i])
            trend = str(mapped["trend"].iat[i] or "")
            side = "buy" if trend == "bull" else "sell"
            prev_day_high = None if pd.isna(mapped["prev_day_high"].iat[i]) else float(mapped["prev_day_high"].iat[i])
            prev_day_low = None if pd.isna(mapped["prev_day_low"].iat[i]) else float(mapped["prev_day_low"].iat[i])
            ntz_nearest_label = ""
            ntz_nearest_dist_pips = None
            ntz_blocked = False
            ntz_reason = ""
            for label, level in (("PDH", prev_day_high), ("PDL", prev_day_low)):
                if level is None:
                    continue
                dist_pips = abs(close_price - level) / PIP_SIZE
                if ntz_nearest_dist_pips is None or dist_pips < ntz_nearest_dist_pips:
                    ntz_nearest_dist_pips = dist_pips
                    ntz_nearest_label = label
                if dist_pips <= ntz_buffer_pips:
                    ntz_blocked = True
                    ntz_reason = f"NTZ: within {dist_pips:.1f}p of {label}"

            fib_lower = None if pd.isna(mapped["fib_lower"].iat[i]) else float(mapped["fib_lower"].iat[i])
            fib_upper = None if pd.isna(mapped["fib_upper"].iat[i]) else float(mapped["fib_upper"].iat[i])
            if fib_lower is None or fib_upper is None:
                fib_allowed = True
                fib_reason = "awaiting_fib_levels"
            else:
                fib_allowed = (fib_lower - fib_boundary_buffer_pips * PIP_SIZE) <= close_price <= (fib_upper + fib_boundary_buffer_pips * PIP_SIZE)
                fib_reason = "" if fib_allowed else "outside_intraday_fib_corridor_raw"

            m1_spread_pips, m1_aligned, m1_compressing, m1_bucket = _m1_health_for_bar(m1f, i, side)
            conviction_m1_bucket, conviction_multiplier, conviction_lots = compute_conviction_for_bar(
                m1f,
                i,
                str(mapped["m5_bucket"].iat[i] or "normal"),
                side,
                policy,
                float(profile.risk.max_lots),
            )
            zone_aligned, zone_recent_cross = _zone_state_for_bar(m1f, i, side, policy)
            atr_stop_pips = _resolve_trial10_stop_pips(profile, mapped, i)
            regime_result = evaluate_regime_gate(
                hour_et=hour_et(ts),
                side=side,
                m5_bucket=str(mapped["m5_bucket"].iat[i] or "normal"),
                enabled=bool(getattr(policy, "regime_gate_enabled", False)),
                london_sell_veto=bool(getattr(policy, "regime_london_sell_veto", True)),
                london_start_hour_et=int(getattr(policy, "regime_london_start_hour_et", 3)),
                london_end_hour_et=int(getattr(policy, "regime_london_end_hour_et", 12)),
                boost_hours_et=tuple(int(x) for x in getattr(policy, "regime_boost_hours_et", (6, 7, 12, 13, 14, 15))),
                boost_multiplier=float(getattr(policy, "regime_boost_multiplier", 1.35)),
                buy_base_multiplier=float(getattr(policy, "regime_buy_base_multiplier", 0.65)),
                sell_base_multiplier=float(getattr(policy, "regime_sell_base_multiplier", 0.35)),
                chop_paused=False,
                chop_pause_reason="",
            )

            bars_since_cross = mapped[f"bars_since_m1_cross_{side}"].iat[i]
            bars_since_recross = mapped[f"bars_since_m5_recross_{side}"].iat[i]

            row: dict[str, object] = {
                "time": ts.isoformat(),
                "hour_et": hour_et(ts),
                "open": float(mapped["open"].iat[i]),
                "high": float(mapped["high"].iat[i]),
                "low": float(mapped["low"].iat[i]),
                "close": close_price,
                "trend": trend,
                "side": side,
                "m5_close": None if pd.isna(mapped["m5_close"].iat[i]) else float(mapped["m5_close"].iat[i]),
                "m5_ema9": None if pd.isna(mapped["m5_ema9"].iat[i]) else float(mapped["m5_ema9"].iat[i]),
                "m5_ema21": None if pd.isna(mapped["m5_ema21"].iat[i]) else float(mapped["m5_ema21"].iat[i]),
                "m5_gap_pips": None if pd.isna(mapped["gap_pips"].iat[i]) else float(mapped["gap_pips"].iat[i]),
                "m5_bucket": str(mapped["m5_bucket"].iat[i] or ""),
                "m5_atr_pips": None if pd.isna(mapped["atr_pips"].iat[i]) else float(mapped["atr_pips"].iat[i]),
                "m5_ema9_slope_pips_per_bar": None if pd.isna(mapped["m5_ema9_slope_pips_per_bar"].iat[i]) else float(mapped["m5_ema9_slope_pips_per_bar"].iat[i]),
                "m5_slope_aligned": bool(mapped["m5_slope_aligned"].iat[i]) if not pd.isna(mapped["m5_slope_aligned"].iat[i]) else None,
                "m5_slope_disagrees_ema21": bool(mapped["m5_slope_disagrees_ema21"].iat[i]) if not pd.isna(mapped["m5_slope_disagrees_ema21"].iat[i]) else None,
                "m1_ema5": None if pd.isna(m1f["ema_5"].iat[i]) else float(m1f["ema_5"].iat[i]),
                "m1_ema9": None if pd.isna(m1f["ema_9"].iat[i]) else float(m1f["ema_9"].iat[i]),
                "m1_spread_pips": round(float(m1_spread_pips), 4),
                "m1_aligned": bool(m1_aligned),
                "m1_compressing": bool(m1_compressing),
                "m1_bucket": str(m1_bucket),
                "zone_aligned": bool(zone_aligned),
                "zone_recent_cross": bool(zone_recent_cross),
                "conviction_m1_bucket": str(conviction_m1_bucket),
                "conviction_multiplier": float(conviction_multiplier),
                "conviction_lots": float(conviction_lots),
                "atr_stop_pips": round(float(atr_stop_pips), 4),
                "regime_allowed": bool(regime_result.allowed),
                "regime_label": str(regime_result.label),
                "regime_multiplier": float(regime_result.multiplier),
                "regime_reason": str(regime_result.reason),
                "bars_since_m1_cross": None if pd.isna(bars_since_cross) else int(bars_since_cross),
                "bars_since_m5_recross": None if pd.isna(bars_since_recross) else int(bars_since_recross),
                "ntz_blocked": bool(ntz_blocked),
                "ntz_reason": str(ntz_reason),
                "ntz_nearest_level": str(ntz_nearest_label),
                "ntz_nearest_dist_pips": ntz_nearest_dist_pips,
                "prev_day_high": prev_day_high,
                "prev_day_low": prev_day_low,
                "dist_prev_day_high_pips": None if prev_day_high is None else abs(close_price - prev_day_high) / PIP_SIZE,
                "dist_prev_day_low_pips": None if prev_day_low is None else abs(close_price - prev_day_low) / PIP_SIZE,
                "fib_lower": fib_lower,
                "fib_upper": fib_upper,
                "fib_allowed": bool(fib_allowed),
                "fib_reason": str(fib_reason),
                "candidate_exists": False,
                "candidate_type": None,
                "candidate_tier": None,
                "candidate_pullback_label": None,
                "candidate_pullback_bar_count": None,
                "candidate_pullback_structure_ratio": None,
                "eval_reason_counts": "",
            }

            for tier in (int(x) for x in policy.tier_ema_periods):
                ema_prev = float(m1f[f"ema_{tier}"].iat[i - 1]) if i > 0 else float("nan")
                ema_now = float(m1f[f"ema_{tier}"].iat[i])
                prev_low = float(m1f["low"].iat[i - 1]) if i > 0 else float("nan")
                prev_high = float(m1f["high"].iat[i - 1]) if i > 0 else float("nan")
                reclaim_ema_now = float(m1f[f"ema_{int(policy.tier_reclaim_ema_period)}"].iat[i])
                requires_strong = tier in {int(x) for x in getattr(policy, "strong_m5_only_tier_periods", tuple())}
                m5_ok = (not requires_strong) or str(mapped["m5_bucket"].iat[i] or "normal") == "strong"
                if side == "buy":
                    touched_prev = False if i == 0 else prev_low <= ema_prev
                    reclaim_ok = close_price > reclaim_ema_now
                else:
                    touched_prev = False if i == 0 else prev_high >= ema_prev
                    reclaim_ok = close_price < reclaim_ema_now
                prefix = f"tier_{tier}"
                row[f"{prefix}_ema"] = ema_now
                row[f"{prefix}_touched_prev"] = bool(touched_prev)
                row[f"{prefix}_reclaim_ok"] = bool(reclaim_ok)
                row[f"{prefix}_m5_ok"] = bool(m5_ok)
                row[f"{prefix}_pq_label"] = None
                row[f"{prefix}_pq_bar_count"] = None
                row[f"{prefix}_pq_structure_ratio"] = None

            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
            writer.writerow(row)

    print(f"Wrote {len(mapped)} rows to {out_path}")


if __name__ == "__main__":
    main()
