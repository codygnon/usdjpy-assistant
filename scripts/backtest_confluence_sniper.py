#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


PIP_SIZE = 0.01


@dataclass
class OpenPosition:
    trade_id: int
    side: str
    entry_mode: str
    entry_session: str
    entry_time: pd.Timestamp
    entry_day: str
    entry_index_in_day: int
    entry_price: float
    lots_initial: float
    lots_remaining: float
    stop_price: float
    tp1_price: float
    tp2_price: float
    sl_pips: float
    tp1_pips: float
    tp2_pips: float
    tp1_filled: bool
    level_name: str
    level_price: Optional[float]
    realized_pips: float
    realized_usd: float
    exit_price_last: Optional[float]


@dataclass
class ClosedTrade:
    trade_id: int
    side: str
    entry_mode: str
    entry_session: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    lots: float
    sl_pips: float
    tp1_pips: float
    tp2_pips: float
    pips: float
    usd: float
    exit_reason: str
    level_name: str
    level_price: Optional[float]
    entry_day: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Confluence Sniper backtest for USDJPY on M1 CSV.")
    p.add_argument("--in", dest="inputs", action="append", required=True, help="Input M1 CSV path (repeatable)")
    p.add_argument("--out", default="research_out/confluence_sniper_backtest.json", help="Output JSON path")
    p.add_argument("--base-lot", type=float, default=0.1, help="Base lot size")
    p.add_argument("--spread-mode", choices=["fixed", "variable"], default="fixed", help="Spread model")
    p.add_argument("--spread-pips", type=float, default=1.8, help="Average/fixed spread in pips")
    p.add_argument("--spread-min-pips", type=float, default=1.0, help="Minimum spread in pips")
    p.add_argument("--spread-max-pips", type=float, default=2.2, help="Maximum spread in pips")
    p.add_argument("--d1-ema", type=int, default=20, help="D1 EMA period")
    p.add_argument("--h4-ema-fast", type=int, default=9, help="H4 fast EMA period")
    p.add_argument("--h4-ema-slow", type=int, default=21, help="H4 slow EMA period")
    p.add_argument("--h4-min-gap-pips", type=float, default=3.0, help="Minimum H4 EMA gap in pips")
    p.add_argument("--d1-neutral-pips", type=float, default=0.5, help="No-trade D1 EMA neutrality threshold in pips")
    p.add_argument("--d1-neutral-use-h4", action="store_true", help="If D1 is neutral, use H4 direction instead of blocking")
    p.add_argument("--tokyo-start", type=float, default=0.0, help="Tokyo start UTC hour (fractional supported)")
    p.add_argument("--tokyo-end", type=float, default=7.0, help="Tokyo end UTC hour (exclusive, fractional supported)")
    p.add_argument("--london-start", type=float, default=7.0, help="London start UTC hour (fractional supported)")
    p.add_argument("--london-end", type=float, default=11.5, help="London end UTC hour (exclusive, fractional supported)")
    p.add_argument("--london-end-hour", dest="london_end", type=float, default=argparse.SUPPRESS, help="Alias for --london-end")
    p.add_argument("--ny-start", type=float, default=12.5, help="NY overlap start UTC hour (fractional supported)")
    p.add_argument("--ny-start-hour", dest="ny_start", type=float, default=argparse.SUPPRESS, help="Alias for --ny-start")
    p.add_argument("--ny-end", type=float, default=16.0, help="NY overlap end UTC hour (exclusive, fractional supported)")
    p.add_argument("--mode-a-breakout-pips", type=float, default=2.0, help="Mode A breakout threshold beyond Tokyo range")
    p.add_argument("--mode-a-retest-pips", type=float, default=2.0, help="Mode A retest zone half-width in pips")
    p.add_argument("--mode-a-retest-window-bars", type=int, default=90, help="Mode A retest expiry in M1 bars")
    p.add_argument("--level-proximity-pips", type=float, default=5.0, help="Mode B key-level proximity in pips")
    p.add_argument("--mode-b-pullback-bars", type=int, default=3, help="M5 bars window for Mode B pullback cross/touch logic")
    p.add_argument("--confirm-proximity-pips", type=float, default=5.0, help="Mode B confirm-bar proximity in pips")
    p.add_argument("--mode-a-only", action="store_true", help="Disable Mode B and trade only Mode A")
    p.add_argument("--mode-b-london-only", action="store_true", help="Allow Mode B only during London session")
    p.add_argument("--mode-b-half-size", action="store_true", help="Use half-size lots for Mode B entries")
    p.add_argument(
        "--mode-b-require-rejection",
        action="store_true",
        help="Require level rejection on confirm bar for Mode B (wick through level then close back)",
    )
    p.add_argument(
        "--mode-b-rejection-probe-pips",
        type=float,
        default=1.0,
        help="Required wick penetration through key level for Mode B rejection rule",
    )
    p.add_argument("--h4-swing-lookback", type=int, default=20, help="H4 bars used for swing detection")
    p.add_argument("--h4-swing-keep", type=int, default=3, help="Number of most recent swing highs/lows retained")
    p.add_argument(
        "--h4-swing-levels-count",
        type=int,
        default=None,
        help="Alias for h4-swing-keep; if set overrides --h4-swing-keep",
    )
    p.add_argument("--round-step-price", type=float, default=0.5, help="Round-number step in price units")
    p.add_argument(
        "--round-number-pip-step",
        type=float,
        default=None,
        help="Single round-number step in pips (e.g., 25 or 50). Overrides round-step-price.",
    )
    p.add_argument(
        "--round-number-pip-steps",
        type=str,
        default="",
        help="Comma-separated round-number steps in pips (e.g., 25,50).",
    )
    p.add_argument("--sl-lookback-m5", type=int, default=20, help="M5 bars for structure stop")
    p.add_argument("--sl-buffer-pips", type=float, default=1.5, help="Structure stop buffer in pips")
    p.add_argument("--sl-max-pips", type=float, default=15.0, help="Skip entries with SL distance above this")
    p.add_argument("--sl-min-pips", type=float, default=None, help="Optional minimum structural SL distance; skip if below")
    p.add_argument("--sl-floor-pips", type=float, default=5.0, help="Floor SL distance")
    p.add_argument("--tp1-mult", type=float, default=1.5, help="TP1 multiplier of SL distance")
    p.add_argument("--tp2-mult", type=float, default=3.0, help="TP2 multiplier of SL distance")
    p.add_argument("--tp2-multiplier", dest="tp2_mult", type=float, default=argparse.SUPPRESS, help="Alias for --tp2-mult")
    p.add_argument("--tp1-close-fraction", type=float, default=0.60, help="Fraction closed at TP1")
    p.add_argument("--trail-buffer-pips", type=float, default=0.5, help="Trailing stop buffer from M5 EMA9 after TP1")
    p.add_argument("--max-open-positions", type=int, default=1, help="Max concurrent open positions")
    p.add_argument("--max-entries-per-day", type=int, default=2, help="Max entries per UTC day")
    p.add_argument("--second-entry-loss-mult", type=float, default=0.5, help="Lot multiplier for second entry if first trade full-SL loss")
    return p.parse_args()


def load_m1(paths: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for p in paths:
        f = Path(p)
        if not f.exists():
            raise FileNotFoundError(f"Missing CSV: {f}")
        df = pd.read_csv(f)
        need = {"time", "open", "high", "low", "close"}
        if not need.issubset(df.columns):
            raise ValueError(f"{f} missing required columns: {sorted(need)}")
        df = df[["time", "open", "high", "low", "close"]].copy()
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        for c in ("open", "high", "low", "close"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna().sort_values("time")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["time"], keep="last").sort_values("time").reset_index(drop=True)
    return out


def resample_ohlc(m1: pd.DataFrame, rule: str) -> pd.DataFrame:
    d = m1.set_index("time").sort_index()
    out = (
        d.resample(rule, label="right", closed="right")
        .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
        .dropna()
        .reset_index()
    )
    return out


def pip_value_usd_per_lot(price: float) -> float:
    return 1000.0 / max(1e-6, float(price))


def hour_in_window(hour: float, start: float, end: float) -> bool:
    h = float(hour) % 24.0
    s = float(start) % 24.0
    e = float(end) % 24.0
    if s == e:
        return False
    if s < e:
        return s <= h < e
    return h >= s or h < e


def classify_session(ts: pd.Timestamp, london_start: float, london_end: float, ny_start: float, ny_end: float) -> str:
    h = float(ts.hour) + float(ts.minute) / 60.0 + float(ts.second) / 3600.0
    if hour_in_window(h, london_start, london_end):
        return "london"
    if hour_in_window(h, ny_start, ny_end):
        return "ny_overlap"
    return "dead"


def compute_spread_pips(i: int, ts: pd.Timestamp, mode: str, avg: float, mn: float, mx: float) -> float:
    if mode == "fixed":
        return max(1.0, float(avg))
    # Deterministic wave around avg spread.
    x = float(avg) + 0.35 * math.sin(i * 0.017) + 0.15 * math.sin(i * 0.071)
    h = int(ts.hour)
    if h >= 16 or h < 7:
        x += 0.08
    return max(1.0, min(float(mx), max(float(mn), x)))


def recent_h4_swings(h4: pd.DataFrame, p4: int, lookback: int, keep: int) -> tuple[list[float], list[float]]:
    if p4 < 2:
        return [], []
    start = max(0, p4 - int(lookback) + 1)
    w = h4.iloc[start : p4 + 1].reset_index(drop=True)
    highs: list[float] = []
    lows: list[float] = []
    for j in range(1, len(w) - 1):
        h = float(w.at[j, "high"])
        hp = float(w.at[j - 1, "high"])
        hn = float(w.at[j + 1, "high"])
        l = float(w.at[j, "low"])
        lp = float(w.at[j - 1, "low"])
        ln = float(w.at[j + 1, "low"])
        if h > hp and h > hn:
            highs.append(h)
        if l < lp and l < ln:
            lows.append(l)
    return highs[-int(keep) :], lows[-int(keep) :]


def select_key_level(
    *,
    side: str,
    mid: float,
    pip_size: float,
    threshold_pips: float,
    prev_day_high: Optional[float],
    prev_day_low: Optional[float],
    current_day_high: Optional[float],
    current_day_low: Optional[float],
    swing_highs: list[float],
    swing_lows: list[float],
    round_step_prices: list[float],
) -> tuple[Optional[str], Optional[float], Optional[float]]:
    cands: list[tuple[str, float]] = []
    if side == "buy":
        if prev_day_low is not None:
            cands.append(("prev_day_low", float(prev_day_low)))
        if current_day_low is not None:
            cands.append(("current_day_low", float(current_day_low)))
        for i, v in enumerate(swing_lows):
            cands.append((f"h4_swing_low_{i+1}", float(v)))
        for step in round_step_prices:
            if step <= 0:
                continue
            below = math.floor(mid / float(step)) * float(step)
            cands.append((f"round_below_{step:.4f}", float(below)))
        filtered = []
        for name, lvl in cands:
            if lvl <= mid:
                d = (mid - lvl) / pip_size
                if d <= float(threshold_pips):
                    filtered.append((name, lvl, d))
    else:
        if prev_day_high is not None:
            cands.append(("prev_day_high", float(prev_day_high)))
        if current_day_high is not None:
            cands.append(("current_day_high", float(current_day_high)))
        for i, v in enumerate(swing_highs):
            cands.append((f"h4_swing_high_{i+1}", float(v)))
        for step in round_step_prices:
            if step <= 0:
                continue
            above = math.ceil(mid / float(step)) * float(step)
            cands.append((f"round_above_{step:.4f}", float(above)))
        filtered = []
        for name, lvl in cands:
            if lvl >= mid:
                d = (lvl - mid) / pip_size
                if d <= float(threshold_pips):
                    filtered.append((name, lvl, d))
    if not filtered:
        return None, None, None
    filtered.sort(key=lambda x: x[2])
    return filtered[0]


def m5_pullback_ok(side: str, m5: pd.DataFrame, p5: int, lookback_bars: int) -> bool:
    lb = max(2, int(lookback_bars))
    if p5 < lb:
        return False
    w = m5.iloc[p5 - lb : p5 + 1].reset_index(drop=True)
    cross = False
    price_flip = False
    for j in range(1, len(w)):
        e9p = float(w.at[j - 1, "ema9"])
        e21p = float(w.at[j - 1, "ema21"])
        cp = float(w.at[j - 1, "close"])
        e9 = float(w.at[j, "ema9"])
        e21 = float(w.at[j, "ema21"])
        c = float(w.at[j, "close"])
        if side == "buy":
            if e9p <= e21p and e9 > e21:
                cross = True
            if cp < e9p and c > e9:
                price_flip = True
        else:
            if e9p >= e21p and e9 < e21:
                cross = True
            if cp > e9p and c < e9:
                price_flip = True
    return bool(cross or price_flip)


def resolve_round_steps_price(args: argparse.Namespace) -> list[float]:
    steps_pips: list[float] = []
    if getattr(args, "round_number_pip_steps", ""):
        for raw in str(args.round_number_pip_steps).split(","):
            raw = raw.strip()
            if not raw:
                continue
            try:
                steps_pips.append(float(raw))
            except ValueError:
                continue
    if getattr(args, "round_number_pip_step", None) is not None:
        steps_pips = [float(args.round_number_pip_step)]
    if steps_pips:
        return [float(s) * PIP_SIZE for s in steps_pips if float(s) > 0]
    return [float(args.round_step_price)]


def run_backtest(args: argparse.Namespace) -> dict:
    m1 = load_m1(args.inputs)
    m5 = resample_ohlc(m1, "5min")
    h4 = resample_ohlc(m1, "4h")
    d1 = resample_ohlc(m1, "1D")

    m5["ema9"] = m5["close"].ewm(span=9, adjust=False).mean()
    m5["ema21"] = m5["close"].ewm(span=21, adjust=False).mean()
    h4["ema9"] = h4["close"].ewm(span=int(args.h4_ema_fast), adjust=False).mean()
    h4["ema21"] = h4["close"].ewm(span=int(args.h4_ema_slow), adjust=False).mean()
    d1["ema20"] = d1["close"].ewm(span=int(args.d1_ema), adjust=False).mean()

    h4_swing_keep = int(args.h4_swing_levels_count) if args.h4_swing_levels_count is not None else int(args.h4_swing_keep)
    round_step_prices = resolve_round_steps_price(args)

    tokyo_ranges: dict[str, tuple[float, float]] = {}
    m1_time_hour = m1["time"].dt.hour + (m1["time"].dt.minute / 60.0) + (m1["time"].dt.second / 3600.0)
    for day, g in m1.groupby(m1["time"].dt.date):
        hours = m1_time_hour.loc[g.index]
        g_tok = g[hours.apply(lambda h: hour_in_window(float(h), float(args.tokyo_start), float(args.tokyo_end)))]
        if not g_tok.empty:
            tokyo_ranges[str(day)] = (float(g_tok["high"].max()), float(g_tok["low"].min()))

    m5_times = m5["time"].tolist()
    h4_times = h4["time"].tolist()
    d1_times = d1["time"].tolist()
    p5 = p4 = pdx = -1

    daily_state: dict[str, dict] = {}
    current_day_high: Optional[float] = None
    current_day_low: Optional[float] = None
    last_day_key: Optional[str] = None

    open_pos: Optional[OpenPosition] = None
    closed: list[ClosedTrade] = []
    blocked_reasons: dict[str, int] = {}
    spread_used: list[float] = []
    trade_id_seq = 0
    max_open_positions = 0

    def day_bucket(day_key: str) -> dict:
        if day_key not in daily_state:
            daily_state[day_key] = {
                "entries_opened": 0,
                "first_trade_full_sl": None,
                "mode_a_attempted": False,
                "mode_a_breakout_side": None,
                "mode_a_level": None,
                "mode_a_breakout_time": None,
                "mode_a_expiry_time": None,
                "mode_a_filled": False,
                "mode_a_disabled": False,
                "mode_b_filled_london": False,
                "mode_b_filled_ny": False,
            }
        return daily_state[day_key]

    def add_block(reason: str) -> None:
        blocked_reasons[reason] = blocked_reasons.get(reason, 0) + 1

    def close_leg(pos: OpenPosition, exit_price: float, leg_lots: float) -> tuple[float, float]:
        if leg_lots <= 0:
            return 0.0, 0.0
        pips_leg = ((float(exit_price) - float(pos.entry_price)) / PIP_SIZE) if pos.side == "buy" else ((float(pos.entry_price) - float(exit_price)) / PIP_SIZE)
        usd_leg = pips_leg * pip_value_usd_per_lot(float(exit_price)) * float(leg_lots)
        portion = float(leg_lots) / max(1e-12, float(pos.lots_initial))
        pos.realized_pips += pips_leg * portion
        pos.realized_usd += usd_leg
        pos.exit_price_last = float(exit_price)
        return pips_leg, usd_leg

    def finalize_trade(pos: OpenPosition, ts: pd.Timestamp, reason: str, full_sl: bool) -> None:
        nonlocal open_pos
        day_key = str(pos.entry_day)
        b = day_bucket(day_key)
        if pos.entry_index_in_day == 1 and b.get("first_trade_full_sl") is None:
            b["first_trade_full_sl"] = bool(full_sl)
        closed.append(
            ClosedTrade(
                trade_id=int(pos.trade_id),
                side=str(pos.side),
                entry_mode=str(pos.entry_mode),
                entry_session=str(pos.entry_session),
                entry_time=pd.Timestamp(pos.entry_time),
                exit_time=pd.Timestamp(ts),
                entry_price=float(pos.entry_price),
                exit_price=float(pos.exit_price_last if pos.exit_price_last is not None else pos.entry_price),
                lots=float(pos.lots_initial),
                sl_pips=float(pos.sl_pips),
                tp1_pips=float(pos.tp1_pips),
                tp2_pips=float(pos.tp2_pips),
                pips=float(pos.realized_pips),
                usd=float(pos.realized_usd),
                exit_reason=str(reason),
                level_name=str(pos.level_name),
                level_price=float(pos.level_price) if pos.level_price is not None else None,
                entry_day=day_key,
            )
        )
        open_pos = None

    for i in range(len(m1)):
        r = m1.iloc[i]
        ts = pd.Timestamp(r["time"])
        op = float(r["open"])
        hi = float(r["high"])
        lo = float(r["low"])
        cl = float(r["close"])
        day_key = str(ts.date())

        spread_pips = compute_spread_pips(i, ts, str(args.spread_mode), float(args.spread_pips), float(args.spread_min_pips), float(args.spread_max_pips))
        spread_used.append(spread_pips)
        half_spread = spread_pips * PIP_SIZE / 2.0
        bid = cl - half_spread
        ask = cl + half_spread
        mid = cl

        if day_key != last_day_key:
            current_day_high = mid
            current_day_low = mid
            last_day_key = day_key
        else:
            current_day_high = max(float(current_day_high), mid)
            current_day_low = min(float(current_day_low), mid)

        old5 = p5
        while p5 + 1 < len(m5_times) and m5_times[p5 + 1] <= ts:
            p5 += 1
        new_m5_bar = p5 != old5
        while p4 + 1 < len(h4_times) and h4_times[p4 + 1] <= ts:
            p4 += 1
        while pdx + 1 < len(d1_times) and d1_times[pdx + 1] <= ts:
            pdx += 1

        # Manage existing position.
        if open_pos is not None:
            # Post-TP1 trailing update from latest completed M5 EMA9.
            if open_pos.tp1_filled and p5 >= 0:
                ema9 = float(m5.iloc[p5]["ema9"])
                if open_pos.side == "buy":
                    new_stop = ema9 - float(args.trail_buffer_pips) * PIP_SIZE
                    if new_stop > float(open_pos.stop_price):
                        open_pos.stop_price = float(new_stop)
                else:
                    new_stop = ema9 + float(args.trail_buffer_pips) * PIP_SIZE
                    if new_stop < float(open_pos.stop_price):
                        open_pos.stop_price = float(new_stop)

            if not open_pos.tp1_filled:
                # (a) Full SL first.
                sl_hit = (lo <= float(open_pos.stop_price)) if open_pos.side == "buy" else (hi >= float(open_pos.stop_price))
                if sl_hit:
                    close_leg(open_pos, float(open_pos.stop_price), float(open_pos.lots_remaining))
                    full_sl = True
                    finalize_trade(open_pos, ts, "sl", full_sl=full_sl)
                else:
                    # (b) TP1 partial.
                    tp1_hit = (hi >= float(open_pos.tp1_price)) if open_pos.side == "buy" else (lo <= float(open_pos.tp1_price))
                    if tp1_hit:
                        leg = float(open_pos.lots_initial) * float(args.tp1_close_fraction)
                        close_leg(open_pos, float(open_pos.tp1_price), leg)
                        open_pos.lots_remaining = max(0.0, float(open_pos.lots_initial) - leg)
                        open_pos.tp1_filled = True
                        open_pos.stop_price = float(open_pos.entry_price)
            else:
                # (c) TP1 already filled: TP2 or trail stop (whichever first; TP2 priority on same bar).
                tp2_hit = (hi >= float(open_pos.tp2_price)) if open_pos.side == "buy" else (lo <= float(open_pos.tp2_price))
                if tp2_hit:
                    close_leg(open_pos, float(open_pos.tp2_price), float(open_pos.lots_remaining))
                    finalize_trade(open_pos, ts, "tp1_then_tp2", full_sl=False)
                else:
                    stop_hit = (lo <= float(open_pos.stop_price)) if open_pos.side == "buy" else (hi >= float(open_pos.stop_price))
                    if stop_hit:
                        close_leg(open_pos, float(open_pos.stop_price), float(open_pos.lots_remaining))
                        finalize_trade(open_pos, ts, "tp1_then_trail", full_sl=False)

        # Entry logic only when flat.
        if open_pos is not None:
            max_open_positions = max(max_open_positions, 1)
            continue

        sess = classify_session(ts, float(args.london_start), float(args.london_end), float(args.ny_start), float(args.ny_end))
        if sess not in {"london", "ny_overlap"}:
            continue

        # Trend hard gate: D1 + H4 agreement (with optional D1-neutral fallback to H4).
        if pdx < 0 or p4 < 0:
            add_block("trend_insufficient_bars")
            continue

        h4_row = h4.iloc[p4]
        h4_9 = float(h4_row["ema9"])
        h4_21 = float(h4_row["ema21"])
        h4_gap_pips = abs(h4_9 - h4_21) / PIP_SIZE
        if h4_gap_pips < float(args.h4_min_gap_pips):
            add_block("h4_gap")
            continue
        h4_side = "buy" if h4_9 > h4_21 else "sell" if h4_9 < h4_21 else None
        if h4_side is None:
            add_block("h4_no_side")
            continue

        d1_row = d1.iloc[pdx]
        d1_close = float(d1_row["close"])
        d1_ema = float(d1_row["ema20"])
        d1_delta_pips = abs(d1_close - d1_ema) / PIP_SIZE
        if d1_delta_pips <= float(args.d1_neutral_pips):
            if bool(args.d1_neutral_use_h4):
                side = str(h4_side)
            else:
                add_block("d1_neutral")
                continue
        else:
            d1_side = "buy" if d1_close > d1_ema else "sell"
            if h4_side != d1_side:
                add_block("d1_h4_disagree")
                continue
            side = str(d1_side)

        day_cfg = day_bucket(day_key)
        if int(day_cfg["entries_opened"]) >= int(args.max_entries_per_day):
            add_block("daily_entry_cap")
            continue

        lot_mult = 1.0
        if int(day_cfg["entries_opened"]) == 1 and bool(day_cfg.get("first_trade_full_sl")):
            lot_mult = float(args.second_entry_loss_mult)

        prev_completed_day = d1[d1["time"] < pd.Timestamp(ts.date(), tz="UTC")]
        prev_day_high = float(prev_completed_day.iloc[-1]["high"]) if not prev_completed_day.empty else None
        prev_day_low = float(prev_completed_day.iloc[-1]["low"]) if not prev_completed_day.empty else None
        swing_highs, swing_lows = recent_h4_swings(h4, p4, int(args.h4_swing_lookback), int(h4_swing_keep))

        # Mode A state update/detection (London only).
        if sess == "london":
            if day_cfg["mode_a_breakout_side"] is not None and day_cfg["mode_a_expiry_time"] is not None and ts > pd.Timestamp(day_cfg["mode_a_expiry_time"]):
                day_cfg["mode_a_breakout_side"] = None
                day_cfg["mode_a_disabled"] = True

            if (
                new_m5_bar
                and p5 >= 0
                and not bool(day_cfg["mode_a_attempted"])
                and not bool(day_cfg["mode_a_filled"])
                and not bool(day_cfg["mode_a_disabled"])
            ):
                tr = tokyo_ranges.get(day_key)
                if tr is not None:
                    tk_high, tk_low = tr
                    m5_close = float(m5.iloc[p5]["close"])
                    up_break = m5_close > (float(tk_high) + float(args.mode_a_breakout_pips) * PIP_SIZE)
                    dn_break = m5_close < (float(tk_low) - float(args.mode_a_breakout_pips) * PIP_SIZE)
                    if side == "buy" and up_break:
                        day_cfg["mode_a_attempted"] = True
                        day_cfg["mode_a_breakout_side"] = "buy"
                        day_cfg["mode_a_level"] = float(tk_high)
                        day_cfg["mode_a_breakout_time"] = ts
                        day_cfg["mode_a_expiry_time"] = ts + pd.Timedelta(minutes=int(args.mode_a_retest_window_bars))
                    elif side == "sell" and dn_break:
                        day_cfg["mode_a_attempted"] = True
                        day_cfg["mode_a_breakout_side"] = "sell"
                        day_cfg["mode_a_level"] = float(tk_low)
                        day_cfg["mode_a_breakout_time"] = ts
                        day_cfg["mode_a_expiry_time"] = ts + pd.Timedelta(minutes=int(args.mode_a_retest_window_bars))
                else:
                    day_cfg["mode_a_disabled"] = True

        def open_trade(entry_mode: str, level_name: str, level_price: Optional[float]) -> bool:
            nonlocal open_pos, trade_id_seq, max_open_positions
            if open_pos is not None:
                return False
            if int(args.max_open_positions) <= 0:
                return False
            if p5 < int(args.sl_lookback_m5) - 1:
                add_block("sl_structure_insufficient")
                return False
            m5w = m5.iloc[p5 - int(args.sl_lookback_m5) + 1 : p5 + 1]
            entry_price = float(ask if side == "buy" else bid)
            if side == "buy":
                raw_stop = float(m5w["low"].min()) - float(args.sl_buffer_pips) * PIP_SIZE
                if raw_stop >= entry_price:
                    add_block("sl_invalid")
                    return False
            else:
                raw_stop = float(m5w["high"].max()) + float(args.sl_buffer_pips) * PIP_SIZE
                if raw_stop <= entry_price:
                    add_block("sl_invalid")
                    return False

            sl_dist_pips = abs(entry_price - raw_stop) / PIP_SIZE
            if sl_dist_pips > float(args.sl_max_pips):
                add_block("sl_too_wide")
                return False
            if args.sl_min_pips is not None and sl_dist_pips < float(args.sl_min_pips):
                add_block("sl_too_tight")
                return False
            if sl_dist_pips < float(args.sl_floor_pips):
                sl_dist_pips = float(args.sl_floor_pips)
                raw_stop = entry_price - sl_dist_pips * PIP_SIZE if side == "buy" else entry_price + sl_dist_pips * PIP_SIZE

            tp1_pips = float(args.tp1_mult) * sl_dist_pips
            tp2_pips = float(args.tp2_mult) * sl_dist_pips
            tp1_price = entry_price + tp1_pips * PIP_SIZE if side == "buy" else entry_price - tp1_pips * PIP_SIZE
            tp2_price = entry_price + tp2_pips * PIP_SIZE if side == "buy" else entry_price - tp2_pips * PIP_SIZE

            trade_id_seq += 1
            mode_mult = 0.5 if (entry_mode == "B" and bool(args.mode_b_half_size)) else 1.0
            lots = max(0.01, float(args.base_lot) * float(lot_mult) * float(mode_mult))
            day_cfg["entries_opened"] = int(day_cfg["entries_opened"]) + 1
            open_pos = OpenPosition(
                trade_id=int(trade_id_seq),
                side=side,
                entry_mode=entry_mode,
                entry_session=sess,
                entry_time=ts,
                entry_day=day_key,
                entry_index_in_day=int(day_cfg["entries_opened"]),
                entry_price=float(entry_price),
                lots_initial=float(lots),
                lots_remaining=float(lots),
                stop_price=float(raw_stop),
                tp1_price=float(tp1_price),
                tp2_price=float(tp2_price),
                sl_pips=float(sl_dist_pips),
                tp1_pips=float(tp1_pips),
                tp2_pips=float(tp2_pips),
                tp1_filled=False,
                level_name=level_name,
                level_price=float(level_price) if level_price is not None else None,
                realized_pips=0.0,
                realized_usd=0.0,
                exit_price_last=None,
            )
            max_open_positions = max(max_open_positions, 1)
            return True

        opened = False

        # Mode A first in London.
        if sess == "london" and not bool(day_cfg["mode_a_filled"]) and not bool(day_cfg["mode_a_disabled"]):
            if day_cfg["mode_a_breakout_side"] is not None and side == str(day_cfg["mode_a_breakout_side"]):
                lvl = float(day_cfg["mode_a_level"])
                z = float(args.mode_a_retest_pips) * PIP_SIZE
                if side == "buy":
                    retest = (bid <= lvl + z) and (bid >= lvl - z)
                    confirm = cl > op
                else:
                    retest = (ask >= lvl - z) and (ask <= lvl + z)
                    confirm = cl < op
                if retest and confirm:
                    opened = open_trade("A", "tokyo_retest", lvl)
                    if opened:
                        day_cfg["mode_a_filled"] = True
                        day_cfg["mode_a_breakout_side"] = None
                        day_cfg["mode_a_disabled"] = True
                        day_cfg["mode_b_filled_london"] = True

        # Mode B
        if not opened and not bool(args.mode_a_only):
            if bool(args.mode_b_london_only) and sess != "london":
                add_block("mode_b_session_restricted")
                continue
            if sess == "london" and bool(day_cfg["mode_a_filled"]):
                pass
            elif sess == "london" and bool(day_cfg["mode_b_filled_london"]):
                pass
            elif sess == "ny_overlap" and bool(day_cfg["mode_b_filled_ny"]):
                pass
            else:
                name, lvl, dist = select_key_level(
                    side=side,
                    mid=mid,
                    pip_size=PIP_SIZE,
                    threshold_pips=float(args.level_proximity_pips),
                    prev_day_high=prev_day_high,
                    prev_day_low=prev_day_low,
                    current_day_high=current_day_high,
                    current_day_low=current_day_low,
                    swing_highs=swing_highs,
                    swing_lows=swing_lows,
                    round_step_prices=round_step_prices,
                )
                if name is not None and lvl is not None:
                    pb = m5_pullback_ok(side, m5, p5, int(args.mode_b_pullback_bars))
                    if pb:
                        confirm = (cl > op) if side == "buy" else (cl < op)
                        dist_confirm = abs(mid - float(lvl)) / PIP_SIZE
                        rejection_ok = True
                        if bool(args.mode_b_require_rejection):
                            # Rejection definition: wick probes >= probe pips beyond level and closes back through level.
                            probe = float(args.mode_b_rejection_probe_pips) * PIP_SIZE
                            if side == "buy":
                                rejection_ok = bool(lo <= float(lvl) - probe and cl > float(lvl))
                            else:
                                rejection_ok = bool(hi >= float(lvl) + probe and cl < float(lvl))
                        if confirm and rejection_ok and dist_confirm <= float(args.confirm_proximity_pips):
                            opened = open_trade("B", str(name), float(lvl))
                            if opened:
                                if sess == "london":
                                    day_cfg["mode_b_filled_london"] = True
                                else:
                                    day_cfg["mode_b_filled_ny"] = True

    # End of file close.
    if open_pos is not None:
        last = m1.iloc[-1]
        ts_last = pd.Timestamp(last["time"])
        cl_last = float(last["close"])
        sp_last = compute_spread_pips(len(m1) - 1, ts_last, str(args.spread_mode), float(args.spread_pips), float(args.spread_min_pips), float(args.spread_max_pips))
        hs_last = sp_last * PIP_SIZE / 2.0
        bid_last = cl_last - hs_last
        ask_last = cl_last + hs_last
        exit_px = bid_last if open_pos.side == "buy" else ask_last
        close_leg(open_pos, float(exit_px), float(open_pos.lots_remaining))
        reason = "tp1_then_eod" if open_pos.tp1_filled else "eod"
        finalize_trade(open_pos, ts_last, reason, full_sl=False)

    tdf = pd.DataFrame([x.__dict__ for x in closed])
    if tdf.empty:
        return {
            "summary": {
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": None,
                "net_pips": 0.0,
                "net_usd": 0.0,
                "avg_win_pips": None,
                "avg_loss_pips": None,
                "median_win_pips": None,
                "median_loss_pips": None,
                "breakeven_win_rate_est": None,
                "profit_factor": None,
                "max_drawdown_usd": 0.0,
                "max_open_positions": int(max_open_positions),
            },
            "by_entry_mode": [],
            "by_session": [],
            "by_exit_reason": [],
            "exit_reason_counts": {},
            "closed_trades": [],
            "daily": [],
            "blocked_reasons": dict(sorted(blocked_reasons.items(), key=lambda kv: kv[1], reverse=True)),
        }

    tdf["is_win"] = tdf["pips"] > 0
    tdf = tdf.sort_values("exit_time").reset_index(drop=True)
    tdf["cum_usd"] = tdf["usd"].cumsum()
    tdf["cum_peak"] = tdf["cum_usd"].cummax()
    tdf["dd_usd"] = tdf["cum_peak"] - tdf["cum_usd"]

    wins = int(tdf["is_win"].sum())
    losses = int((~tdf["is_win"]).sum())
    win_rate = float(tdf["is_win"].mean()) * 100.0
    net_pips = float(tdf["pips"].sum())
    net_usd = float(tdf["usd"].sum())
    avg_win = float(tdf.loc[tdf["is_win"], "pips"].mean()) if wins else None
    avg_loss = float(tdf.loc[~tdf["is_win"], "pips"].mean()) if losses else None
    med_win = float(tdf.loc[tdf["is_win"], "pips"].median()) if wins else None
    med_loss = float(tdf.loc[~tdf["is_win"], "pips"].median()) if losses else None
    abs_avg_loss = abs(avg_loss) if avg_loss is not None else None
    be_wr = (abs_avg_loss / (float(avg_win) + abs_avg_loss) * 100.0) if (avg_win is not None and abs_avg_loss is not None and (float(avg_win) + abs_avg_loss) > 0.0) else None
    gross_win = float(tdf.loc[tdf["pips"] > 0, "pips"].sum())
    gross_loss = abs(float(tdf.loc[tdf["pips"] < 0, "pips"].sum()))
    pf = (gross_win / gross_loss) if gross_loss > 0 else None
    max_dd = float(tdf["dd_usd"].max()) if len(tdf) else 0.0

    by_mode = (
        tdf.groupby("entry_mode", dropna=False)
        .agg(trades=("trade_id", "size"), win_rate=("is_win", "mean"), net_pips=("pips", "sum"), net_usd=("usd", "sum"))
        .reset_index()
        .sort_values("trades", ascending=False)
    )
    by_mode["win_rate"] = by_mode["win_rate"] * 100.0

    by_sess = (
        tdf.groupby("entry_session", dropna=False)
        .agg(trades=("trade_id", "size"), win_rate=("is_win", "mean"), net_pips=("pips", "sum"), net_usd=("usd", "sum"))
        .reset_index()
        .sort_values("trades", ascending=False)
    )
    by_sess["win_rate"] = by_sess["win_rate"] * 100.0

    by_day = (
        tdf.groupby("entry_day", dropna=False)
        .agg(trades=("trade_id", "size"), wins=("is_win", "sum"), net_pips=("pips", "sum"), net_usd=("usd", "sum"))
        .reset_index()
        .sort_values("entry_day")
    )
    by_exit = (
        tdf.groupby("exit_reason", dropna=False)
        .agg(trades=("trade_id", "size"), win_rate=("is_win", "mean"), net_pips=("pips", "sum"), net_usd=("usd", "sum"))
        .reset_index()
        .sort_values("trades", ascending=False)
    )
    by_exit["win_rate"] = by_exit["win_rate"] * 100.0
    exit_reason_counts = {str(k): int(v) for k, v in tdf["exit_reason"].value_counts(dropna=False).to_dict().items()}

    return {
        "summary": {
            "trades": int(len(tdf)),
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 3),
            "net_pips": round(net_pips, 3),
            "net_usd": round(net_usd, 3),
            "avg_win_pips": round(avg_win, 3) if avg_win is not None else None,
            "avg_loss_pips": round(avg_loss, 3) if avg_loss is not None else None,
            "median_win_pips": round(med_win, 3) if med_win is not None else None,
            "median_loss_pips": round(med_loss, 3) if med_loss is not None else None,
            "breakeven_win_rate_est": round(be_wr, 3) if be_wr is not None else None,
            "profit_factor": round(pf, 4) if pf is not None else None,
            "max_drawdown_usd": round(max_dd, 3),
            "max_open_positions": int(max_open_positions),
        },
        "by_entry_mode": by_mode.to_dict("records"),
        "by_session": by_sess.to_dict("records"),
        "by_exit_reason": by_exit.to_dict("records"),
        "exit_reason_counts": exit_reason_counts,
        "closed_trades": tdf[
            [
                "trade_id",
                "side",
                "entry_mode",
                "entry_session",
                "entry_time",
                "exit_time",
                "entry_price",
                "exit_price",
                "sl_pips",
                "tp1_pips",
                "tp2_pips",
                "pips",
                "usd",
                "exit_reason",
                "level_name",
                "level_price",
                "entry_day",
            ]
        ].to_dict("records"),
        "daily": by_day.to_dict("records"),
        "blocked_reasons": dict(sorted(blocked_reasons.items(), key=lambda kv: kv[1], reverse=True)),
    }


def main() -> int:
    args = parse_args()
    if float(args.spread_min_pips) < 1.0:
        args.spread_min_pips = 1.0
    if float(args.spread_pips) < 1.0:
        args.spread_pips = 1.0
    if float(args.spread_max_pips) < float(args.spread_min_pips):
        args.spread_max_pips = args.spread_min_pips
    if args.h4_swing_levels_count is not None and int(args.h4_swing_levels_count) < 1:
        args.h4_swing_levels_count = 1
    if args.sl_min_pips is not None and float(args.sl_min_pips) < 0:
        args.sl_min_pips = 0.0
    if args.sl_min_pips is not None and float(args.sl_min_pips) > float(args.sl_max_pips):
        args.sl_min_pips = float(args.sl_max_pips)

    m1 = load_m1(args.inputs)
    resolved_round_steps_price = resolve_round_steps_price(args)
    resolved_round_steps_pips = [round(float(x) / PIP_SIZE, 6) for x in resolved_round_steps_price]
    spreads = [
        compute_spread_pips(i, pd.Timestamp(m1.iloc[i]["time"]), str(args.spread_mode), float(args.spread_pips), float(args.spread_min_pips), float(args.spread_max_pips))
        for i in range(len(m1))
    ]
    report = {
        "config": {
            "inputs": args.inputs,
            "bars_m1": int(len(m1)),
            "start_utc": str(m1["time"].min()),
            "end_utc": str(m1["time"].max()),
            "pip_size": float(PIP_SIZE),
            "spread_mode": str(args.spread_mode),
            "spread_avg_target_pips": float(args.spread_pips),
            "spread_avg_actual_pips": round(float(sum(spreads) / max(1, len(spreads))), 6),
            "spread_min_pips": float(args.spread_min_pips),
            "spread_max_pips": float(args.spread_max_pips),
            "base_lot": float(args.base_lot),
            "d1_ema_period": int(args.d1_ema),
            "h4_ema_fast": int(args.h4_ema_fast),
            "h4_ema_slow": int(args.h4_ema_slow),
            "h4_min_gap_pips": float(args.h4_min_gap_pips),
            "d1_neutral_pips": float(args.d1_neutral_pips),
            "d1_neutral_use_h4": bool(args.d1_neutral_use_h4),
            "sessions_utc": {
                "tokyo_range": [float(args.tokyo_start), float(args.tokyo_end)],
                "london": [float(args.london_start), float(args.london_end)],
                "ny_overlap": [float(args.ny_start), float(args.ny_end)],
            },
            "mode_a": {
                "breakout_pips": float(args.mode_a_breakout_pips),
                "retest_pips": float(args.mode_a_retest_pips),
                "retest_window_bars": int(args.mode_a_retest_window_bars),
                "enabled": True,
            },
            "mode_b": {
                "level_proximity_pips": float(args.level_proximity_pips),
                "pullback_bars": int(args.mode_b_pullback_bars),
                "confirm_proximity_pips": float(args.confirm_proximity_pips),
                "enabled": not bool(args.mode_a_only),
                "london_only": bool(args.mode_b_london_only),
                "half_size": bool(args.mode_b_half_size),
                "require_rejection": bool(args.mode_b_require_rejection),
                "rejection_probe_pips": float(args.mode_b_rejection_probe_pips),
            },
            "key_levels": {
                "h4_swing_lookback": int(args.h4_swing_lookback),
                "h4_swing_keep": int(args.h4_swing_levels_count) if args.h4_swing_levels_count is not None else int(args.h4_swing_keep),
                "round_step_price": float(args.round_step_price),
                "round_steps_price": resolved_round_steps_price,
                "round_steps_pips": resolved_round_steps_pips,
            },
            "risk": {
                "sl_lookback_m5": int(args.sl_lookback_m5),
                "sl_buffer_pips": float(args.sl_buffer_pips),
                "sl_max_pips": float(args.sl_max_pips),
                "sl_min_pips": float(args.sl_min_pips) if args.sl_min_pips is not None else None,
                "sl_floor_pips": float(args.sl_floor_pips),
                "tp1_mult": float(args.tp1_mult),
                "tp2_mult": float(args.tp2_mult),
                "tp1_close_fraction": float(args.tp1_close_fraction),
                "trail_buffer_pips": float(args.trail_buffer_pips),
            },
            "profiles": {
                "mode_a_only": bool(args.mode_a_only),
                "mode_b_london_only": bool(args.mode_b_london_only),
                "mode_b_half_size": bool(args.mode_b_half_size),
            },
            "caps": {
                "max_open_positions": int(args.max_open_positions),
                "max_entries_per_day": int(args.max_entries_per_day),
                "second_entry_loss_mult": float(args.second_entry_loss_mult),
            },
        },
        "results": run_backtest(args),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")

    s = report["results"]["summary"]
    print(
        f"Confluence Sniper complete | trades={s['trades']} win_rate={s['win_rate']}% "
        f"net_pips={s['net_pips']} net_usd={s['net_usd']} maxDD_usd={s['max_drawdown_usd']}"
    )
    print(f"Wrote report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
