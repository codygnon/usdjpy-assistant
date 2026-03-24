#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.backtest_trial10 import (
    PIP_SIZE,
    Position,
    build_profile,
    choose_first_event,
    close_leg,
    load_m1,
    map_features_to_m1,
    pips_to_usd,
    prepare_features,
    unrealized_usd,
)


@dataclass(frozen=True)
class StopFormula:
    atr_mult: float
    floor_pips: float
    cap_pips: float

    @property
    def key(self) -> str:
        return f"atr{self.atr_mult:.2f}_floor{self.floor_pips:.1f}_cap{self.cap_pips:.1f}"


@dataclass
class SimMetrics:
    weighted_pips: float
    profit_usd: float
    exit_reason: str
    tp1_done: bool
    hold_minutes: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate Trial #10 ATR stop formulas on historical USDJPY entries.")
    p.add_argument("--in", dest="input_csv", default="research_out/USDJPY_M1_OANDA_250k.csv")
    p.add_argument("--trades", default="research_out/trial10_250k_pause1_trades.csv")
    p.add_argument("--out-json", default="research_out/trial10_atr_stop_calibration_250k.json")
    p.add_argument("--out-md", default="research_out/trial10_atr_stop_calibration_250k.md")
    p.add_argument("--spread-pips", type=float, default=2.0)
    p.add_argument("--atr-mults", default="0.8,1.0,1.2,1.3,1.4,1.6")
    p.add_argument("--floors", default="3,4,5")
    p.add_argument("--caps", default="10,12,15,20")
    p.add_argument("--max-hold-minutes", type=float, default=12 * 60.0)
    p.add_argument("--min-trades-per-class", type=int, default=150)
    return p.parse_args()


def parse_float_list(raw: str) -> list[float]:
    values: list[float] = []
    for part in str(raw or "").split(","):
        token = part.strip()
        if not token:
            continue
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


def classify_trade(row: pd.Series) -> str:
    entry_type = str(row.get("entry_type") or "")
    tier = row.get("tier_period")
    if entry_type == "zone_entry":
        return "zone"
    if pd.isna(tier):
        return "tier_unknown"
    tier_int = int(float(tier))
    if tier_int in {17, 21}:
        return f"shallow_{tier_int}"
    if tier_int in {27, 33}:
        return f"standard_{tier_int}"
    return f"tier_{tier_int}"


def aggregate_name(class_name: str) -> str:
    if class_name.startswith("shallow_"):
        return "shallow_all"
    if class_name.startswith("standard_"):
        return "standard_all"
    if class_name.startswith("tier_"):
        return "other_tiers"
    return class_name


def make_formulas(args: argparse.Namespace) -> list[StopFormula]:
    formulas: list[StopFormula] = []
    for mult in parse_float_list(args.atr_mults):
        for floor in parse_float_list(args.floors):
            for cap in parse_float_list(args.caps):
                if cap < floor:
                    continue
                formulas.append(StopFormula(atr_mult=float(mult), floor_pips=float(floor), cap_pips=float(cap)))
    return formulas


def load_trade_set(path: str) -> pd.DataFrame:
    trades = pd.read_csv(path)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades["signal_time"] = pd.to_datetime(trades["signal_time"], utc=True)
    trades["entry_class"] = trades.apply(classify_trade, axis=1)
    trades["entry_group"] = trades["entry_class"].map(aggregate_name)
    return trades.sort_values("entry_time").reset_index(drop=True)


def build_mapped(input_csv: str):
    class Args:
        tier_periods = ""
        tp1_close_pct = None

    m1 = load_m1(input_csv)
    profile, policy = build_profile(Args())
    features = prepare_features(m1, policy)
    m1f = features["m1"]
    mapped = map_features_to_m1(m1f, features)
    return m1f, mapped, features, policy


def simulate_trade(
    trade: pd.Series,
    *,
    stop_formula: StopFormula,
    m1f: pd.DataFrame,
    mapped: pd.DataFrame,
    m5_close_set: set[pd.Timestamp],
    policy,
    spread_pips: float,
    max_hold_minutes: float,
) -> Optional[SimMetrics]:
    entry_time = pd.Timestamp(trade["entry_time"])
    if entry_time not in m1f.index:
        return None
    j0 = int(m1f.index.get_loc(entry_time))
    entry_price = float(trade["entry_price"])
    side = str(trade["side"])
    atr_pips = mapped["atr_pips"].iat[j0]
    if pd.isna(atr_pips):
        return None
    stop_pips = min(max(float(atr_pips) * stop_formula.atr_mult, stop_formula.floor_pips), stop_formula.cap_pips)
    spread_price = float(spread_pips) * PIP_SIZE
    half_spread = spread_price / 2.0
    be_offset = spread_price + float(policy.be_spread_plus_pips) * PIP_SIZE
    stop_price = entry_price - stop_pips * PIP_SIZE if side == "buy" else entry_price + stop_pips * PIP_SIZE
    tp1_price = entry_price + float(policy.tp1_pips) * PIP_SIZE if side == "buy" else entry_price - float(policy.tp1_pips) * PIP_SIZE
    pos = Position(
        trade_id=int(trade["trade_id"]),
        side=side,
        entry_type=str(trade["entry_type"]),
        tier_period=(None if pd.isna(trade["tier_period"]) else int(float(trade["tier_period"]))),
        signal_time=pd.Timestamp(trade["signal_time"]),
        entry_time=entry_time,
        entry_price=entry_price,
        initial_lots=float(trade["initial_lots"]),
        remaining_lots=float(trade["initial_lots"]),
        stop_price=stop_price,
        initial_stop_price=stop_price,
        tp1_price=tp1_price,
        tp1_done=False,
        tp1_close_pct=float(policy.tp1_close_pct),
        be_stop_price=None,
        m5_bucket=str(trade.get("m5_bucket") or "normal"),
        m1_bucket=str(trade.get("m1_bucket") or "neutral"),
        conviction_multiplier=float(trade.get("conviction_multiplier") or 1.0),
        conviction_lots=float(trade.get("conviction_lots") or trade["initial_lots"]),
        atr_stop_pips=stop_pips,
        pullback_quality_label=str(trade.get("pullback_quality_label") or ""),
        pullback_quality_bar_count=int(float(trade.get("pullback_quality_bar_count") or 0)),
        pullback_quality_structure_ratio=float(trade.get("pullback_quality_structure_ratio") or 0.0),
        pullback_quality_dampener=float(trade.get("pullback_quality_dampener") or 1.0),
        regime_multiplier=float(trade.get("regime_multiplier") or 1.0),
        regime_label=str(trade.get("regime_label") or "baseline"),
    )

    max_hold_delta = pd.Timedelta(minutes=float(max_hold_minutes))
    last_ts = entry_time
    last_close_mid = float(m1f["close"].iat[j0])
    for j in range(j0, len(m1f)):
        ts = m1f.index[j]
        if ts - entry_time > max_hold_delta:
            break
        bar = m1f.iloc[j]
        open_mid = float(bar["open"])
        high_mid = float(bar["high"])
        low_mid = float(bar["low"])
        close_mid = float(bar["close"])
        last_ts = ts
        last_close_mid = close_mid
        open_bid = open_mid - half_spread
        open_ask = open_mid + half_spread
        high_bid = high_mid - half_spread
        low_bid = low_mid - half_spread
        high_ask = high_mid + half_spread
        low_ask = low_mid + half_spread
        close_bid = close_mid - half_spread
        close_ask = close_mid + half_spread
        ref_price = close_mid
        closed = False

        if not pos.tp1_done:
            if pos.side == "buy":
                stop_hit = low_bid <= pos.stop_price
                tp1_hit = high_bid >= pos.tp1_price
            else:
                stop_hit = high_ask >= pos.stop_price
                tp1_hit = low_ask <= pos.tp1_price

            if stop_hit and tp1_hit:
                first = choose_first_event(pos.side, open_mid, pos.stop_price, pos.tp1_price, spread_price)
                if first == "stop":
                    tp1_hit = False
                else:
                    stop_hit = False

            if stop_hit:
                close_leg(pos, pos.remaining_lots, pos.stop_price, "initial_stop", ref_price)
                closed = True
            elif tp1_hit:
                tp1_lots = round(pos.initial_lots * (pos.tp1_close_pct / 100.0), 2)
                tp1_lots = max(0.01, min(tp1_lots, pos.remaining_lots))
                close_leg(pos, tp1_lots, pos.tp1_price, "tp1_partial", ref_price)
                pos.tp1_done = True
                pos.be_stop_price = pos.entry_price + be_offset if pos.side == "buy" else pos.entry_price - be_offset
                pos.stop_price = float(pos.be_stop_price)
                if pos.remaining_lots <= 0.0:
                    closed = True
                else:
                    if pos.side == "buy" and low_bid <= pos.stop_price:
                        close_leg(pos, pos.remaining_lots, pos.stop_price, "tp1_then_be_same_bar", ref_price)
                        closed = True
                    elif pos.side == "sell" and high_ask >= pos.stop_price:
                        close_leg(pos, pos.remaining_lots, pos.stop_price, "tp1_then_be_same_bar", ref_price)
                        closed = True

        if not closed and pos.tp1_done:
            if pos.side == "buy" and low_bid <= pos.stop_price:
                close_leg(pos, pos.remaining_lots, pos.stop_price, "runner_stop", ref_price)
                closed = True
            elif pos.side == "sell" and high_ask >= pos.stop_price:
                close_leg(pos, pos.remaining_lots, pos.stop_price, "runner_stop", ref_price)
                closed = True

        if not closed and pos.tp1_done and ts in m5_close_set:
            m5_close = mapped["m5_close"].iat[j]
            trail_ema = mapped["m5_trail_ema_20"].iat[j]
            if pd.notna(trail_ema) and pd.notna(m5_close):
                if pos.side == "buy" and float(m5_close) < float(trail_ema):
                    close_leg(pos, pos.remaining_lots, close_bid, "m5_trail_close", ref_price)
                    closed = True
                elif pos.side == "sell" and float(m5_close) > float(trail_ema):
                    close_leg(pos, pos.remaining_lots, close_ask, "m5_trail_close", ref_price)
                    closed = True

        if closed or pos.remaining_lots <= 0.0:
            weighted_pips = pos.weighted_pips_x_lots / pos.initial_lots if pos.initial_lots > 0 else 0.0
            hold_minutes = (ts - entry_time).total_seconds() / 60.0
            return SimMetrics(
                weighted_pips=float(weighted_pips),
                profit_usd=float(pos.weighted_usd),
                exit_reason=str(pos.exit_reason or ""),
                tp1_done=bool(pos.tp1_done),
                hold_minutes=float(hold_minutes),
            )

    final_exit = last_close_mid - half_spread if side == "buy" else last_close_mid + half_spread
    close_leg(pos, pos.remaining_lots, final_exit, "end_of_horizon", last_close_mid)
    weighted_pips = pos.weighted_pips_x_lots / pos.initial_lots if pos.initial_lots > 0 else 0.0
    return SimMetrics(
        weighted_pips=float(weighted_pips),
        profit_usd=float(pos.weighted_usd),
        exit_reason=str(pos.exit_reason or "end_of_horizon"),
        tp1_done=bool(pos.tp1_done),
        hold_minutes=float((last_ts - entry_time).total_seconds() / 60.0),
    )


def summarize_group(df: pd.DataFrame) -> dict[str, float | int]:
    n = int(len(df))
    if n == 0:
        return {"trades": 0}
    return {
        "trades": n,
        "avg_pips": round(float(df["weighted_pips"].mean()), 3),
        "total_pips": round(float(df["weighted_pips"].sum()), 3),
        "avg_usd": round(float(df["profit_usd"].mean()), 4),
        "total_usd": round(float(df["profit_usd"].sum()), 4),
        "win_rate_pct": round(float((df["weighted_pips"] > 0).mean() * 100.0), 2),
        "tp1_rate_pct": round(float(df["tp1_done"].mean() * 100.0), 2),
        "initial_stop_rate_pct": round(float((df["exit_reason"] == "initial_stop").mean() * 100.0), 2),
        "avg_hold_minutes": round(float(df["hold_minutes"].mean()), 2),
    }


def choose_recommendation(summary_by_formula: dict[str, dict], *, min_trades: int) -> Optional[dict]:
    candidates = [item for item in summary_by_formula.values() if int(item["trades"]) >= min_trades]
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            float(item["avg_pips"]),
            float(item["tp1_rate_pct"]),
            -float(item["initial_stop_rate_pct"]),
        ),
        reverse=True,
    )
    return candidates[0]


def render_markdown(
    *,
    baseline_key: str,
    overall_summary: dict[str, dict],
    class_recommendations: dict[str, Optional[dict]],
    entry_class_results: dict[str, dict[str, dict]],
    min_trades: int,
) -> str:
    lines: list[str] = []
    lines.append("# Trial 10 ATR Stop Calibration")
    lines.append("")
    lines.append(f"Baseline formula reference: `{baseline_key}`")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    labels: list[str] = []
    if baseline_key in overall_summary:
        labels.append(baseline_key)
    best = choose_recommendation(overall_summary, min_trades=min_trades)
    if best is not None and best["formula"] not in labels:
        labels.append(best["formula"])
    for label in labels:
        item = overall_summary[label]
        lines.append(
            f"- `{label}`: {item['trades']} trades, avg {item['avg_pips']} pips, total ${item['total_usd']}, "
            f"TP1 {item['tp1_rate_pct']}%, initial-stop {item['initial_stop_rate_pct']}%"
        )
    lines.append("")
    lines.append("## Entry-Class Recommendations")
    lines.append("")
    for entry_class in sorted(class_recommendations):
        rec = class_recommendations[entry_class]
        if rec is None:
            lines.append(f"- `{entry_class}`: no formula met the `{min_trades}` trade minimum.")
            continue
        base = entry_class_results[entry_class].get(baseline_key)
        if base:
            lines.append(
                f"- `{entry_class}`: recommend `{rec['formula']}` over baseline `{baseline_key}`. "
                f"Baseline avg {base['avg_pips']}p on {base['trades']} trades; "
                f"recommended avg {rec['avg_pips']}p on {rec['trades']} trades."
            )
        else:
            lines.append(
                f"- `{entry_class}`: recommend `{rec['formula']}` with avg {rec['avg_pips']}p on {rec['trades']} trades."
            )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Trades are replayed from historical Trial 10 entries on real USDJPY M1 OHLC data.")
    lines.append("- TP1/BE/M5-trail logic stays fixed; only the ATR stop formula changes.")
    lines.append("- Recommendations prefer formulas with at least the configured trade minimum per entry class.")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    formulas = make_formulas(args)
    baseline_key = StopFormula(atr_mult=1.3, floor_pips=5.0, cap_pips=20.0).key

    trades = load_trade_set(args.trades)
    m1f, mapped, features, policy = build_mapped(args.input_csv)
    m5_close_set = set(features["m5"].index)

    rows: list[dict] = []
    total_work = len(trades) * len(formulas)
    completed = 0
    for _, trade in trades.iterrows():
        for formula in formulas:
            sim = simulate_trade(
                trade,
                stop_formula=formula,
                m1f=m1f,
                mapped=mapped,
                m5_close_set=m5_close_set,
                policy=policy,
                spread_pips=float(args.spread_pips),
                max_hold_minutes=float(args.max_hold_minutes),
            )
            completed += 1
            if sim is None:
                continue
            rows.append(
                {
                    "formula": formula.key,
                    "entry_class": str(trade["entry_class"]),
                    "entry_group": str(trade["entry_group"]),
                    "weighted_pips": sim.weighted_pips,
                    "profit_usd": sim.profit_usd,
                    "exit_reason": sim.exit_reason,
                    "tp1_done": sim.tp1_done,
                    "hold_minutes": sim.hold_minutes,
                }
            )

    sim_df = pd.DataFrame(rows)
    if sim_df.empty:
        raise RuntimeError("No simulation rows produced.")

    overall_summary: dict[str, dict] = {}
    for formula, g in sim_df.groupby("formula"):
        overall_summary[str(formula)] = {"formula": str(formula), **summarize_group(g)}

    entry_class_results: dict[str, dict[str, dict]] = {}
    class_recommendations: dict[str, Optional[dict]] = {}
    for entry_class, g_class in sim_df.groupby("entry_class"):
        per_formula: dict[str, dict] = {}
        for formula, g_formula in g_class.groupby("formula"):
            per_formula[str(formula)] = {"formula": str(formula), **summarize_group(g_formula)}
        entry_class_results[str(entry_class)] = per_formula
        class_recommendations[str(entry_class)] = choose_recommendation(per_formula, min_trades=int(args.min_trades_per_class))

    aggregate_results: dict[str, dict[str, dict]] = {}
    aggregate_recommendations: dict[str, Optional[dict]] = {}
    for group_name, g_group in sim_df.groupby("entry_group"):
        per_formula: dict[str, dict] = {}
        for formula, g_formula in g_group.groupby("formula"):
            per_formula[str(formula)] = {"formula": str(formula), **summarize_group(g_formula)}
        aggregate_results[str(group_name)] = per_formula
        aggregate_recommendations[str(group_name)] = choose_recommendation(per_formula, min_trades=int(args.min_trades_per_class))

    best_overall = choose_recommendation(overall_summary, min_trades=1)
    payload = {
        "input_csv": args.input_csv,
        "trades_csv": args.trades,
        "spread_pips": float(args.spread_pips),
        "max_hold_minutes": float(args.max_hold_minutes),
        "formula_grid": [f.key for f in formulas],
        "baseline_formula": baseline_key,
        "overall": {
            "baseline": overall_summary.get(baseline_key),
            "best": best_overall,
            "all_formulas": overall_summary,
        },
        "entry_class_recommendations": class_recommendations,
        "entry_class_results": entry_class_results,
        "aggregate_group_recommendations": aggregate_recommendations,
        "aggregate_group_results": aggregate_results,
    }

    out_json = Path(args.out_json)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    Path(args.out_md).write_text(
        render_markdown(
            baseline_key=baseline_key,
            overall_summary=overall_summary,
            class_recommendations=class_recommendations,
            entry_class_results=entry_class_results,
            min_trades=int(args.min_trades_per_class),
        ),
        encoding="utf-8",
    )

    print(json.dumps({"out_json": str(out_json), "out_md": args.out_md, "rows": int(len(sim_df)), "work_items": total_work}, indent=2))


if __name__ == "__main__":
    main()
