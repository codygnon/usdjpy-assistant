from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import math
import time

import numpy as np
import pandas as pd


ROOT = Path("/Users/codygnon/Documents/usdjpy_assistant")
M1_PATH = ROOT / "research_out/USDJPY_M1_OANDA_extended.csv"
STANDALONE_PATH = ROOT / "research_out/trade_analysis/spike_fade_v4/spike_v4_trades.csv"
V7_PATH = ROOT / "research_out/phase3_v7_pfdd_defended_real/v7_enriched_trade_log.csv"
OUT_DIR = ROOT / "research_out/trade_analysis/spike_fade_v4/backtest"
TRADES_OUT = OUT_DIR / "spike_fade_v4_backtest_v2_trades.csv"
SUMMARY_OUT = OUT_DIR / "spike_fade_v4_backtest_v2_summary.txt"
MATCHING_OUT = OUT_DIR / "spike_fade_v4_backtest_v2_matching.txt"

PIP = 0.01
LOT_UNITS = 100_000.0
ENTRY_SPREAD_PIPS = 1.6
STARTING_EQUITY_USD = 100_000.0
MAX_MARGIN_USD = 100_000.0
MARGIN_LEVERAGE = 50.0
ENTRY_WINDOW_MINUTES = 10
STOP_BUFFER_PIPS = 2.0
STOP_CLAMP_MIN_PIPS = 15.0
STOP_CLAMP_MAX_PIPS = 35.0
TP_FRACTION = 0.50
TRAIL_TRIGGER_PIPS = 10.0
TRAIL_DISTANCE_PIPS = 5.0
PROVE_IT_MINUTES = 15
PROVE_IT_PIPS = -5.0
TIME_STOP_MINUTES = 240
PROGRESS_EVERY_BARS = 200_000


@dataclass
class M5Bar:
    end_time: pd.Timestamp
    end_idx: int
    open: float
    high: float
    low: float
    close: float
    atr14_pips: float | None
    ema20: float | None
    m15_ema50: float | None
    range_pips: float
    body_pips: float
    range_atr_ratio: float | None
    body_atr_ratio: float | None
    prior_12_high: float | None
    prior_12_low: float | None
    broad_candidate: bool
    spike_direction: str | None


@dataclass
class PendingOrder:
    armed_time: pd.Timestamp
    expiry_time: pd.Timestamp
    side: str
    trigger_level: float
    spike_time: pd.Timestamp
    spike_direction: str
    spike_high: float
    spike_low: float
    spike_range_pips: float
    prior_12_high: float
    prior_12_low: float
    raw_stop_distance_pips: float
    family_name: str = "Family C"
    model_name: str = "Model 4"
    execution_style: str = "trigger_touch"
    margin_required_usd: float = 0.0


@dataclass
class Position:
    entry_time: pd.Timestamp
    direction: str
    entry_price: float
    confirmation_time: pd.Timestamp
    confirmation_level: float
    spike_time: pd.Timestamp
    spike_direction: str
    session_name: str
    stop_distance_pips: float
    raw_stop_distance_pips: float
    tp_distance_pips: float
    stop_price: float
    tp_price: float
    margin_required_usd: float
    equity_before_entry: float
    mfe_pips: float = 0.0
    mae_pips: float = 0.0
    trail_stop: float | None = None
    high_water: float | None = None
    low_water: float | None = None


@dataclass
class Funnel:
    m1_bars_processed: int = 0
    m5_bars_processed: int = 0
    broad_spike_candidates: int = 0
    family_c_events: int = 0
    limit_orders_armed: int = 0
    limit_orders_filled: int = 0
    limit_orders_expired: int = 0
    blocked_active: int = 0
    blocked_margin: int = 0
    trades_completed: int = 0
    exit_tp: int = 0
    exit_sl: int = 0
    exit_pif: int = 0
    exit_trail: int = 0
    exit_time: int = 0


def pip_value_usd(rate: float) -> float:
    return LOT_UNITS * PIP / float(rate)


def calc_pf(values: pd.Series) -> float:
    wins = values[values > 0].sum()
    losses = abs(values[values < 0].sum())
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return float(wins / losses)


def max_drawdown(values: pd.Series) -> tuple[float, float]:
    if values.empty:
        return 0.0, 0.0
    cum = values.cumsum()
    peak = cum.cummax()
    dd = cum - peak
    max_dd = float(abs(dd.min()))
    peak_at_dd = float(peak.loc[dd.idxmin()]) if len(dd) else 0.0
    dd_pct = (max_dd / peak_at_dd * 100.0) if peak_at_dd > 0 else 0.0
    return max_dd, dd_pct


def compute_stats(trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty:
        return {
            "trade_count": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "net_pnl_usd": 0.0,
            "gross_win_usd": 0.0,
            "gross_loss_usd": 0.0,
            "avg_win_usd": 0.0,
            "avg_loss_usd": 0.0,
            "avg_win_pips": 0.0,
            "avg_loss_pips": 0.0,
            "avg_hold_minutes": 0.0,
            "max_drawdown_usd": 0.0,
            "max_drawdown_pct": 0.0,
        }
    pnl = trades["pnl_usd"].astype(float)
    wins = trades[trades["pnl_usd"] > 0]
    losses = trades[trades["pnl_usd"] < 0]
    equity = 100_000.0 + pnl.cumsum()
    running_peak = equity.cummax()
    drawdown = equity - running_peak
    max_dd = float(abs(drawdown.min())) if not drawdown.empty else 0.0
    peak_before_dd = float(running_peak.loc[drawdown.idxmin()]) if not drawdown.empty else 100_000.0
    max_dd_pct = (max_dd / peak_before_dd * 100.0) if peak_before_dd > 0 else 0.0
    return {
        "trade_count": int(len(trades)),
        "win_rate": float((trades["pnl_usd"] > 0).mean() * 100.0),
        "profit_factor": calc_pf(pnl),
        "net_pnl_usd": float(pnl.sum()),
        "gross_win_usd": float(wins["pnl_usd"].sum()) if not wins.empty else 0.0,
        "gross_loss_usd": float(abs(losses["pnl_usd"].sum())) if not losses.empty else 0.0,
        "avg_win_usd": float(wins["pnl_usd"].mean()) if not wins.empty else 0.0,
        "avg_loss_usd": float(losses["pnl_usd"].mean()) if not losses.empty else 0.0,
        "avg_win_pips": float(wins["pnl_pips"].mean()) if not wins.empty else 0.0,
        "avg_loss_pips": float(losses["pnl_pips"].mean()) if not losses.empty else 0.0,
        "avg_hold_minutes": float(trades["duration_minutes"].mean()),
        "max_drawdown_usd": max_dd,
        "max_drawdown_pct": max_dd_pct,
    }


def bucket_end(ts: pd.Timestamp, freq_minutes: int) -> pd.Timestamp:
    return ts.ceil(f"{freq_minutes}min")


def fmt_money(x: float) -> str:
    return f"${x:,.2f}"


def load_m1(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    time_col = cols.get("time") or cols.get("datetime") or cols.get("timestamp")
    if time_col is None:
        raise ValueError("Could not find time column in M1 file")
    rename = {time_col: "time"}
    for col in ["open", "high", "low", "close"]:
        src = cols.get(col)
        if src is None:
            raise ValueError(f"Missing {col} column in M1 file")
        rename[src] = col
    df = df.rename(columns=rename)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["time", "open", "high", "low", "close"]).sort_values("time").reset_index(drop=True)
    return df


def make_m5_bar(
    end_time: pd.Timestamp,
    end_idx: int,
    open_: float,
    high: float,
    low: float,
    close: float,
    m15_ema50: float | None,
    tr_queue: deque[float],
    tr_sum: float,
    ema20_prev: float | None,
    prior_bars: list[M5Bar],
) -> tuple[M5Bar, float, float | None]:
    prev_close = prior_bars[-1].close if prior_bars else close
    tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
    tr_queue.append(tr)
    tr_sum += tr
    if len(tr_queue) > 14:
        tr_sum -= tr_queue.popleft()
    atr14_pips = (tr_sum / 14.0 / PIP) if len(tr_queue) == 14 else None

    if ema20_prev is None:
        ema20 = close
    else:
        k = 2.0 / 21.0
        ema20 = close * k + ema20_prev * (1.0 - k)

    range_pips = (high - low) / PIP
    body_pips = abs(close - open_) / PIP
    range_atr_ratio = (range_pips / atr14_pips) if atr14_pips and atr14_pips > 0 else None
    body_atr_ratio = (body_pips / atr14_pips) if atr14_pips and atr14_pips > 0 else None

    prev12 = prior_bars[-12:]
    prior_12_high = max((b.high for b in prev12), default=None)
    prior_12_low = min((b.low for b in prev12), default=None)

    broad_candidate = bool(
        atr14_pips is not None
        and range_pips >= 15.0
        and range_atr_ratio is not None
        and range_atr_ratio >= 1.5
        and body_atr_ratio is not None
        and body_atr_ratio >= 1.0
        and close != open_
    )
    spike_direction = None
    if broad_candidate:
        spike_direction = "bullish" if close > open_ else "bearish"

    return (
        M5Bar(
            end_time=end_time,
            end_idx=end_idx,
            open=open_,
            high=high,
            low=low,
            close=close,
            atr14_pips=atr14_pips,
            ema20=ema20,
            m15_ema50=m15_ema50,
            range_pips=range_pips,
            body_pips=body_pips,
            range_atr_ratio=range_atr_ratio,
            body_atr_ratio=body_atr_ratio,
            prior_12_high=prior_12_high,
            prior_12_low=prior_12_low,
            broad_candidate=broad_candidate,
            spike_direction=spike_direction,
        ),
        tr_sum,
        ema20,
    )


def family_c_event(spike: M5Bar, fade: M5Bar) -> dict[str, object] | None:
    if not spike.broad_candidate or spike.atr14_pips is None or spike.ema20 is None or spike.m15_ema50 is None:
        return None
    if spike.prior_12_high is None or spike.prior_12_low is None:
        return None

    spike_mid = spike.low + 0.5 * (spike.high - spike.low)
    stretch_atr_ratio_m5 = abs((spike.close - spike.ema20) / PIP) / spike.atr14_pips
    dist_from_m15_ema50_pips = (spike.close - spike.m15_ema50) / PIP

    if spike.spike_direction == "bullish":
        fade_direction = "short"
        reclaim_mid = fade.close <= spike_mid
        reclaim_prior = fade.close <= spike.prior_12_high
        trigger = spike.prior_12_high
    else:
        fade_direction = "long"
        reclaim_mid = fade.close >= spike_mid
        reclaim_prior = fade.close >= spike.prior_12_low
        trigger = spike.prior_12_low

    family_c = (
        stretch_atr_ratio_m5 >= 1.25
        and abs(dist_from_m15_ema50_pips) >= 20.0
        and reclaim_mid
        and reclaim_prior
    )
    if not family_c:
        return None

    return {
        "spike_time": spike.end_time,
        "fade_time": fade.end_time,
        "spike_direction": spike.spike_direction,
        "fade_direction": fade_direction,
        "trigger_level": float(trigger),
        "spike_high": float(spike.high),
        "spike_low": float(spike.low),
        "spike_range_pips": float(spike.range_pips),
        "prior_12_high": float(spike.prior_12_high),
        "prior_12_low": float(spike.prior_12_low),
        "stretch_atr_ratio_m5": float(stretch_atr_ratio_m5),
        "dist_from_m15_ema50_pips": float(dist_from_m15_ema50_pips),
        "session_name": session_name_for_time(fade.end_time),
    }


def session_name_for_time(ts: pd.Timestamp) -> str:
    hour = int(ts.hour)
    if 0 <= hour < 8:
        return "tokyo"
    if 8 <= hour < 13:
        return "london"
    if 13 <= hour < 21:
        return "ny"
    return "off"


def actual_entry_price(raw_level: float, direction: str) -> float:
    spread_px = ENTRY_SPREAD_PIPS * PIP
    return raw_level + spread_px if direction == "long" else raw_level - spread_px


def required_margin_usd(units: float = LOT_UNITS, leverage: float = MARGIN_LEVERAGE) -> float:
    return float(max(0.0, units) / max(1.0, leverage))


def stop_distances(event: dict[str, object], entry_price: float, direction: str) -> tuple[float | None, float]:
    if direction == "long":
        raw_stop = (entry_price - (float(event["spike_low"]) - STOP_BUFFER_PIPS * PIP)) / PIP
    else:
        raw_stop = ((float(event["spike_high"]) + STOP_BUFFER_PIPS * PIP) - entry_price) / PIP
    if raw_stop > STOP_CLAMP_MAX_PIPS:
        return None, float(raw_stop)
    return max(STOP_CLAMP_MIN_PIPS, float(raw_stop)), float(raw_stop)


def summarize_group(trades: pd.DataFrame, key: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=[key, "trades", "win_rate", "profit_factor", "net_pnl_usd"])
    rows = []
    for value, g in trades.groupby(key):
        rows.append(
            {
                key: value,
                "trades": int(len(g)),
                "win_rate": float((g["pnl_usd"] > 0).mean() * 100.0),
                "profit_factor": calc_pf(g["pnl_usd"]),
                "net_pnl_usd": float(g["pnl_usd"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(key).reset_index(drop=True)


def trade_year_quarter_tables(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if trades.empty:
        empty = pd.DataFrame(columns=["period", "trades", "win_rate", "profit_factor", "net_pnl_usd"])
        return empty, empty
    tmp = trades.copy()
    tmp["entry_dt"] = pd.to_datetime(tmp["entry_time"], utc=True)
    tmp["year"] = tmp["entry_dt"].dt.year
    tmp["quarter"] = tmp["entry_dt"].dt.year.astype(str) + "-Q" + tmp["entry_dt"].dt.quarter.astype(str)
    yearly = []
    for year, g in tmp.groupby("year"):
        yearly.append(
            {
                "period": str(year),
                "trades": int(len(g)),
                "win_rate": float((g["pnl_usd"] > 0).mean() * 100.0),
                "profit_factor": calc_pf(g["pnl_usd"]),
                "net_pnl_usd": float(g["pnl_usd"].sum()),
            }
        )
    quarterly = []
    for q, g in tmp.groupby("quarter"):
        quarterly.append(
            {
                "period": q,
                "trades": int(len(g)),
                "win_rate": float((g["pnl_usd"] > 0).mean() * 100.0),
                "profit_factor": calc_pf(g["pnl_usd"]),
                "net_pnl_usd": float(g["pnl_usd"].sum()),
            }
        )
    return pd.DataFrame(yearly), pd.DataFrame(quarterly)


def find_col(df: pd.DataFrame, names: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def match_trades(standalone: pd.DataFrame, bb: pd.DataFrame) -> dict[str, object]:
    std = standalone.copy().sort_values("entry_time").reset_index(drop=True)
    bar = bb.copy().sort_values("entry_time").reset_index(drop=True)
    std["entry_time"] = pd.to_datetime(std["entry_time"], utc=True)
    std["exit_time"] = pd.to_datetime(std["exit_time"], utc=True)
    bar["entry_time"] = pd.to_datetime(bar["entry_time"], utc=True)
    bar["exit_time"] = pd.to_datetime(bar["exit_time"], utc=True)

    used_std: set[int] = set()
    matches: list[tuple[int, int]] = []
    for i, row in bar.iterrows():
        candidates = []
        for j, srow in std.iterrows():
            if j in used_std:
                continue
            diff = abs((row["entry_time"] - srow["entry_time"]).total_seconds()) / 60.0
            if diff <= 10.0:
                candidates.append((diff, j))
        if not candidates:
            continue
        _, best_j = min(candidates, key=lambda x: x[0])
        used_std.add(best_j)
        matches.append((i, best_j))

    matched_bar = bar.iloc[[i for i, _ in matches]].reset_index(drop=True) if matches else pd.DataFrame(columns=bar.columns)
    matched_std = std.iloc[[j for _, j in matches]].reset_index(drop=True) if matches else pd.DataFrame(columns=std.columns)

    side_mismatches = 0
    entry_diffs = []
    exit_diffs = []
    pnl_corr = float("nan")
    if not matched_bar.empty:
        side_mismatches = int((matched_bar["direction"].astype(str) != matched_std["direction"].astype(str)).sum())
        entry_diffs = ((matched_bar["entry_price"].astype(float) - matched_std["entry_price"].astype(float)).abs() / PIP).tolist()
        exit_diffs = ((matched_bar["exit_price"].astype(float) - matched_std["exit_price"].astype(float)).abs() / PIP).tolist()
        if len(matched_bar) >= 2:
            pnl_corr = float(matched_bar["pnl_usd"].astype(float).corr(matched_std["pnl_usd"].astype(float)))

    return {
        "matched": len(matches),
        "bb_only": len(bar) - len(matches),
        "standalone_only": len(std) - len(matches),
        "entry_price_diff_mean_pips": float(np.mean(entry_diffs)) if entry_diffs else float("nan"),
        "exit_price_diff_mean_pips": float(np.mean(exit_diffs)) if exit_diffs else float("nan"),
        "pnl_corr": pnl_corr,
        "side_mismatches": side_mismatches,
    }


def combined_equity(v7_path: Path, v4_trades: pd.DataFrame) -> dict[str, object]:
    v7 = pd.read_csv(v7_path)
    v7["exit_time"] = pd.to_datetime(v7["exit_time"], utc=True)
    v7["pnl_usd"] = pd.to_numeric(v7["pnl_usd"], errors="coerce")
    v7 = v7.dropna(subset=["exit_time", "pnl_usd"])

    v4 = v4_trades.copy()
    v4["exit_time"] = pd.to_datetime(v4["exit_time"], utc=True)

    def daily_curve(df: pd.DataFrame) -> pd.Series:
        s = df.groupby(df["exit_time"].dt.floor("D"))["pnl_usd"].sum().sort_index()
        return s

    v7_daily = daily_curve(v7)
    v4_daily = daily_curve(v4)
    full_index = v7_daily.index.union(v4_daily.index).sort_values()
    v7_daily = v7_daily.reindex(full_index, fill_value=0.0)
    v4_daily = v4_daily.reindex(full_index, fill_value=0.0)
    combined_daily = v7_daily + v4_daily

    def daily_max_dd(s: pd.Series) -> float:
        cum = s.cumsum()
        peak = cum.cummax()
        dd = cum - peak
        return float(abs(dd.min())) if len(dd) else 0.0

    return {
        "v7_trade_count": int(len(v7)),
        "v7_pf": calc_pf(v7["pnl_usd"]),
        "v7_net": float(v7["pnl_usd"].sum()),
        "v7_maxdd": daily_max_dd(v7_daily),
        "v4_trade_count": int(len(v4)),
        "v4_pf": calc_pf(v4["pnl_usd"]),
        "v4_net": float(v4["pnl_usd"].sum()),
        "v4_maxdd": daily_max_dd(v4_daily),
        "combined_trade_count": int(len(v7) + len(v4)),
        "combined_pf": calc_pf(pd.concat([v7["pnl_usd"], v4["pnl_usd"]], ignore_index=True)),
        "combined_net": float(v7["pnl_usd"].sum() + v4["pnl_usd"].sum()),
        "combined_maxdd": daily_max_dd(combined_daily),
    }


def run_backtest(m1: pd.DataFrame) -> tuple[pd.DataFrame, Funnel, dict[str, object]]:
    funnel = Funnel()
    trades: list[dict[str, object]] = []

    times = m1["time"].tolist()
    opens = m1["open"].to_numpy(np.float64)
    highs = m1["high"].to_numpy(np.float64)
    lows = m1["low"].to_numpy(np.float64)
    closes = m1["close"].to_numpy(np.float64)

    current_5_end: pd.Timestamp | None = None
    current_15_end: pd.Timestamp | None = None
    m5_open = m5_high = m5_low = m5_close = 0.0
    m15_open = m15_high = m15_low = m15_close = 0.0
    tr_queue: deque[float] = deque()
    tr_sum = 0.0
    ema20_prev: float | None = None
    ema50_prev: float | None = None
    last_completed_m15_ema50: float | None = None
    m5_bars: list[M5Bar] = []
    pending_spike: M5Bar | None = None
    pending_order: PendingOrder | None = None
    position: Position | None = None

    realized_equity = STARTING_EQUITY_USD

    def available_margin_usd(current_equity: float, current_position: Position | None) -> float:
        used = float(current_position.margin_required_usd) if current_position is not None else 0.0
        return float(min(MAX_MARGIN_USD, current_equity) - used)

    def finalize_15(end_idx: int) -> None:
        nonlocal ema50_prev, last_completed_m15_ema50
        if current_15_end is None:
            return
        close = float(m15_close)
        if ema50_prev is None:
            ema50_prev = close
        else:
            k = 2.0 / 51.0
            ema50_prev = close * k + ema50_prev * (1.0 - k)
        last_completed_m15_ema50 = ema50_prev

    def maybe_finalize_5(end_idx: int) -> None:
        nonlocal tr_sum, ema20_prev, pending_spike, pending_order
        if current_5_end is None:
            return
        bar, new_tr_sum, new_ema20 = make_m5_bar(
            end_time=current_5_end,
            end_idx=end_idx,
            open_=float(m5_open),
            high=float(m5_high),
            low=float(m5_low),
            close=float(m5_close),
            m15_ema50=last_completed_m15_ema50,
            tr_queue=tr_queue,
            tr_sum=tr_sum,
            ema20_prev=ema20_prev,
            prior_bars=m5_bars,
        )
        tr_sum = new_tr_sum
        ema20_prev = new_ema20
        m5_bars.append(bar)
        funnel.m5_bars_processed += 1
        if bar.broad_candidate:
            funnel.broad_spike_candidates += 1

        if pending_spike is not None:
            event = family_c_event(pending_spike, bar)
            if event is not None:
                funnel.family_c_events += 1
                stop_distance, raw_stop = stop_distances(event, actual_entry_price(float(event["trigger_level"]), str(event["fade_direction"])), str(event["fade_direction"]))
                if stop_distance is not None:
                    if position is None and pending_order is None:
                        pending_order = PendingOrder(
                            armed_time=bar.end_time,
                            expiry_time=bar.end_time + pd.Timedelta(minutes=ENTRY_WINDOW_MINUTES),
                            side=str(event["fade_direction"]),
                            trigger_level=float(event["trigger_level"]),
                            spike_time=pd.Timestamp(event["spike_time"]),
                            spike_direction=str(event["spike_direction"]),
                            spike_high=float(event["spike_high"]),
                            spike_low=float(event["spike_low"]),
                            spike_range_pips=float(event["spike_range_pips"]),
                            prior_12_high=float(event["prior_12_high"]),
                            prior_12_low=float(event["prior_12_low"]),
                            raw_stop_distance_pips=float(raw_stop),
                            margin_required_usd=required_margin_usd(),
                        )
                        funnel.limit_orders_armed += 1
                    else:
                        funnel.blocked_active += 1
            pending_spike = None

        pending_spike = bar if bar.broad_candidate else None

    start_wall = time.time()
    for i, ts in enumerate(times):
        funnel.m1_bars_processed += 1

        next_15 = bucket_end(ts, 15)
        if current_15_end is None:
            current_15_end = next_15
            m15_open = float(opens[i])
            m15_high = float(highs[i])
            m15_low = float(lows[i])
            m15_close = float(closes[i])
        elif next_15 != current_15_end:
            finalize_15(i - 1)
            current_15_end = next_15
            m15_open = float(opens[i])
            m15_high = float(highs[i])
            m15_low = float(lows[i])
            m15_close = float(closes[i])
        else:
            m15_high = max(m15_high, float(highs[i]))
            m15_low = min(m15_low, float(lows[i]))
            m15_close = float(closes[i])

        next_5 = bucket_end(ts, 5)
        if current_5_end is None:
            current_5_end = next_5
            m5_open = float(opens[i])
            m5_high = float(highs[i])
            m5_low = float(lows[i])
            m5_close = float(closes[i])
        elif next_5 != current_5_end:
            maybe_finalize_5(i - 1)
            current_5_end = next_5
            m5_open = float(opens[i])
            m5_high = float(highs[i])
            m5_low = float(lows[i])
            m5_close = float(closes[i])
        else:
            m5_high = max(m5_high, float(highs[i]))
            m5_low = min(m5_low, float(lows[i]))
            m5_close = float(closes[i])

        bar_time = pd.Timestamp(ts)
        high = float(highs[i])
        low = float(lows[i])
        close = float(closes[i])

        if position is not None:
            direction = position.direction
            if direction == "long":
                position.mfe_pips = max(position.mfe_pips, (high - position.entry_price) / PIP)
                position.mae_pips = max(position.mae_pips, (position.entry_price - low) / PIP)
                if low <= position.stop_price:
                    exit_price = position.stop_price
                    exit_reason = "stop_loss"
                elif high >= position.tp_price:
                    exit_price = position.tp_price
                    exit_reason = "take_profit"
                else:
                    exit_price = None
                    exit_reason = None
                if exit_reason is None and position.mfe_pips >= TRAIL_TRIGGER_PIPS:
                    position.high_water = max(position.high_water or position.entry_price, high)
                    candidate = float(position.high_water) - TRAIL_DISTANCE_PIPS * PIP
                    position.trail_stop = candidate if position.trail_stop is None else max(position.trail_stop, candidate)
                    if low <= float(position.trail_stop):
                        exit_price = float(position.trail_stop)
                        exit_reason = "trailing_stop"
                current_pnl_pips = (close - position.entry_price) / PIP
            else:
                position.mfe_pips = max(position.mfe_pips, (position.entry_price - low) / PIP)
                position.mae_pips = max(position.mae_pips, (high - position.entry_price) / PIP)
                if high >= position.stop_price:
                    exit_price = position.stop_price
                    exit_reason = "stop_loss"
                elif low <= position.tp_price:
                    exit_price = position.tp_price
                    exit_reason = "take_profit"
                else:
                    exit_price = None
                    exit_reason = None
                if exit_reason is None and position.mfe_pips >= TRAIL_TRIGGER_PIPS:
                    position.low_water = min(position.low_water or position.entry_price, low)
                    candidate = float(position.low_water) + TRAIL_DISTANCE_PIPS * PIP
                    position.trail_stop = candidate if position.trail_stop is None else min(position.trail_stop, candidate)
                    if high >= float(position.trail_stop):
                        exit_price = float(position.trail_stop)
                        exit_reason = "trailing_stop"
                current_pnl_pips = (position.entry_price - close) / PIP

            held_minutes = int((bar_time - position.entry_time) / pd.Timedelta(minutes=1))
            if exit_reason is None and held_minutes >= PROVE_IT_MINUTES and current_pnl_pips < PROVE_IT_PIPS:
                exit_price = close
                exit_reason = "prove_it_fast"
            if exit_reason is None and held_minutes >= TIME_STOP_MINUTES:
                exit_price = close
                exit_reason = "time_stop"

            if exit_reason is not None:
                pnl_pips = (float(exit_price) - position.entry_price) / PIP if direction == "long" else (position.entry_price - float(exit_price)) / PIP
                pnl_usd = pnl_pips * pip_value_usd(position.entry_price)
                trades.append(
                    {
                        "entry_time": position.entry_time,
                        "exit_time": bar_time,
                        "direction": direction,
                        "entry_price": round(position.entry_price, 6),
                        "exit_price": round(float(exit_price), 6),
                        "pnl_pips": round(float(pnl_pips), 2),
                        "pnl_usd": round(float(pnl_usd), 2),
                        "exit_reason": exit_reason,
                        "mfe_pips": round(float(position.mfe_pips), 2),
                        "mae_pips": round(float(position.mae_pips), 2),
                        "duration_minutes": held_minutes,
                        "spike_time": position.spike_time,
                        "spike_direction": position.spike_direction,
                        "stop_distance_pips": round(float(position.stop_distance_pips), 2),
                        "tp_distance_pips": round(float(position.tp_distance_pips), 2),
                        "family_name": "Family C",
                        "model_name": "Model 4",
                        "execution_style": "trigger_touch",
                        "session_name": position.session_name,
                        "entry_hour": int(position.entry_time.hour),
                        "raw_stop_distance_pips": round(float(position.raw_stop_distance_pips), 2),
                        "confirmation_time": position.confirmation_time,
                        "confirmation_level": round(float(position.confirmation_level), 6),
                        "confirmation_bars": int((position.confirmation_time - position.spike_time) / pd.Timedelta(minutes=1)) if position.confirmation_time >= position.spike_time else 0,
                        "margin_required_usd": round(float(position.margin_required_usd), 2),
                        "equity_before_entry": round(float(position.equity_before_entry), 2),
                        "equity_after_exit": round(float(realized_equity + pnl_usd), 2),
                    }
                )
                funnel.trades_completed += 1
                if exit_reason == "take_profit":
                    funnel.exit_tp += 1
                elif exit_reason == "stop_loss":
                    funnel.exit_sl += 1
                elif exit_reason == "prove_it_fast":
                    funnel.exit_pif += 1
                elif exit_reason == "trailing_stop":
                    funnel.exit_trail += 1
                elif exit_reason == "time_stop":
                    funnel.exit_time += 1
                realized_equity += float(pnl_usd)
                position = None

        if pending_order is not None and position is None:
            if bar_time > pending_order.expiry_time:
                funnel.limit_orders_expired += 1
                pending_order = None
            else:
                touched = low <= pending_order.trigger_level if pending_order.side == "short" else high >= pending_order.trigger_level
                if touched:
                    free_margin = available_margin_usd(realized_equity, None)
                    if pending_order.margin_required_usd > free_margin:
                        funnel.blocked_margin += 1
                        pending_order = None
                        continue
                    entry_price = actual_entry_price(pending_order.trigger_level, pending_order.side)
                    stop_distance, raw_stop = stop_distances(
                        {
                            "spike_low": pending_order.spike_low,
                            "spike_high": pending_order.spike_high,
                        },
                        entry_price,
                        pending_order.side,
                    )
                    if stop_distance is None:
                        pending_order = None
                    else:
                        tp_distance_pips = pending_order.spike_range_pips * TP_FRACTION
                        if pending_order.side == "long":
                            stop_price = entry_price - stop_distance * PIP
                            tp_price = entry_price + tp_distance_pips * PIP
                            high_water = entry_price
                            low_water = None
                        else:
                            stop_price = entry_price + stop_distance * PIP
                            tp_price = entry_price - tp_distance_pips * PIP
                            high_water = None
                            low_water = entry_price
                        position = Position(
                            entry_time=bar_time,
                            direction=pending_order.side,
                            entry_price=entry_price,
                            confirmation_time=bar_time,
                            confirmation_level=pending_order.trigger_level,
                            spike_time=pending_order.spike_time,
                            spike_direction=pending_order.spike_direction,
                            session_name=session_name_for_time(bar_time),
                            stop_distance_pips=stop_distance,
                            raw_stop_distance_pips=raw_stop,
                            tp_distance_pips=tp_distance_pips,
                            stop_price=stop_price,
                            tp_price=tp_price,
                            margin_required_usd=pending_order.margin_required_usd,
                            equity_before_entry=realized_equity,
                            high_water=high_water,
                            low_water=low_water,
                        )
                        funnel.limit_orders_filled += 1
                        pending_order = None

        if (i + 1) % PROGRESS_EVERY_BARS == 0:
            elapsed = time.time() - start_wall
            print(
                f"progress {i+1:,}/{len(m1):,} bars | "
                f"m5={funnel.m5_bars_processed:,} broad={funnel.broad_spike_candidates:,} "
                f"family_c={funnel.family_c_events:,} filled={funnel.limit_orders_filled:,} "
                f"elapsed={elapsed:,.1f}s"
            )

    if current_15_end is not None:
        finalize_15(len(m1) - 1)
    if current_5_end is not None:
        maybe_finalize_5(len(m1) - 1)

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df = trades_df.sort_values("entry_time").reset_index(drop=True)

    meta = {
        "rule_extraction": {
            "family_c": {
                "broad_spike_candidate": [
                    "spike_range_pips >= 15",
                    "range_atr_ratio >= 1.5",
                    "body_atr_ratio >= 1.0",
                    "close != open",
                ],
                "family_c_filter": [
                    "stretch_atr_ratio_m5 >= 1.25",
                    "abs(dist_from_m15_ema50_pips) >= 20.0",
                    "fade_reclaims_spike_mid == True",
                    "fade_reclaims_prior_range == True",
                ],
            },
            "model_4": {
                "trigger_level": "prior_12_high for bullish spikes, prior_12_low for bearish spikes",
                "window_minutes": ENTRY_WINDOW_MINUTES,
                "execution": "trigger_touch",
                "spread_pips": ENTRY_SPREAD_PIPS,
            },
            "exits": {
                "stop_buffer_pips": STOP_BUFFER_PIPS,
                "stop_clamp": [STOP_CLAMP_MIN_PIPS, STOP_CLAMP_MAX_PIPS],
                "tp_fraction": TP_FRACTION,
                "trail_trigger_pips": TRAIL_TRIGGER_PIPS,
                "trail_distance_pips": TRAIL_DISTANCE_PIPS,
                "prove_it_minutes": PROVE_IT_MINUTES,
                "prove_it_pips": PROVE_IT_PIPS,
                "time_stop_minutes": TIME_STOP_MINUTES,
            },
        }
    }
    return trades_df, funnel, meta


def build_summary(
    trades: pd.DataFrame,
    funnel: Funnel,
    matching: dict[str, object],
    combined: dict[str, object],
    meta: dict[str, object],
    standalone_df: pd.DataFrame,
) -> str:
    stats = compute_stats(trades)
    yearly, quarterly = trade_year_quarter_tables(trades)
    exit_breakdown = trades.groupby("exit_reason")["pnl_usd"].agg(["count", "sum"]).reset_index() if not trades.empty else pd.DataFrame(columns=["exit_reason", "count", "sum"])

    lines = []
    lines.append("SPIKE FADE V4 — EXACT RULE EXTRACTION")
    lines.append("=====================================")
    lines.append("")
    lines.append("1. FAMILY C EVENT FILTER")
    lines.append("Broad spike candidate:")
    for rule in meta["rule_extraction"]["family_c"]["broad_spike_candidate"]:
        lines.append(f"  - {rule}")
    lines.append("Family C filter:")
    for rule in meta["rule_extraction"]["family_c"]["family_c_filter"]:
        lines.append(f"  - {rule}")
    lines.append("")
    lines.append("2. MODEL 4 CONFIRMATION")
    lines.append("  - Trigger level = prior_12_high for bullish spikes, prior_12_low for bearish spikes")
    lines.append(f"  - Window = {ENTRY_WINDOW_MINUTES} minutes after fade close")
    lines.append("  - Fill style = trigger_touch")
    lines.append("")
    lines.append("3. STOP / TP / EXIT")
    lines.append(f"  - Entry spread = {ENTRY_SPREAD_PIPS:.1f} pips adverse on fill only")
    lines.append(f"  - Starting equity = {fmt_money(STARTING_EQUITY_USD)}")
    lines.append(f"  - Margin cap = {fmt_money(MAX_MARGIN_USD)}")
    lines.append(f"  - Margin leverage = {MARGIN_LEVERAGE:.1f}x")
    lines.append(f"  - Required margin per 1-lot trade = {fmt_money(required_margin_usd())}")
    lines.append(f"  - Stop = spike extreme + {STOP_BUFFER_PIPS:.1f} pips from actual fill, clamp [{STOP_CLAMP_MIN_PIPS:.0f}, {STOP_CLAMP_MAX_PIPS:.0f}]")
    lines.append(f"  - TP = {TP_FRACTION:.2f} * spike range")
    lines.append(f"  - Trailing = arm at +{TRAIL_TRIGGER_PIPS:.0f}, trail {TRAIL_DISTANCE_PIPS:.0f}")
    lines.append(f"  - Prove-it-fast = {PROVE_IT_MINUTES} minutes, exit if worse than {PROVE_IT_PIPS:.0f} pips from fill price")
    lines.append("")
    lines.append("4. STANDALONE TRADE LOG SCHEMA")
    lines.append("Columns:")
    lines.append("  - " + ", ".join(standalone_df.columns.tolist()))
    lines.append("First 5 rows:")
    lines.append(standalone_df.head(5).to_string(index=False))
    lines.append("")
    lines.append("SPIKE FADE V4 — DETECTION FUNNEL (V2 rules)")
    lines.append("===========================================")
    lines.append(f"Total M1 bars processed:              {funnel.m1_bars_processed}")
    lines.append(f"Total M5 bars processed:              {funnel.m5_bars_processed}")
    lines.append("")
    lines.append(f"Spike candidates detected:            {funnel.broad_spike_candidates}")
    lines.append(f"Exhaustion confirmed (spike+1):       {funnel.family_c_events}")
    lines.append(f"Limit orders armed:                   {funnel.limit_orders_armed}")
    lines.append(f"Limit orders filled:                  {funnel.limit_orders_filled}")
    lines.append(f"Limit orders expired:                 {funnel.limit_orders_expired}")
    lines.append(f"Blocked (position already open):      {funnel.blocked_active}")
    lines.append(f"Blocked (margin cap):                 {funnel.blocked_margin}")
    lines.append("")
    lines.append(f"Trades completed:                     {funnel.trades_completed}")
    lines.append(f"  TP:              {funnel.exit_tp}")
    lines.append(f"  SL:              {funnel.exit_sl}")
    lines.append(f"  prove_it_fast:   {funnel.exit_pif}")
    lines.append(f"  trailing:        {funnel.exit_trail}")
    lines.append(f"  time_stop:       {funnel.exit_time}")
    lines.append("")
    lines.append("V2 SUMMARY")
    lines.append("==========")
    lines.append(f"Total trades: {stats['trade_count']}")
    lines.append(f"Win rate: {stats['win_rate']:.2f}%")
    lines.append(f"Profit factor: {stats['profit_factor']:.2f}")
    lines.append(f"Net P&L: {fmt_money(stats['net_pnl_usd'])}")
    lines.append(f"Max drawdown: {fmt_money(stats['max_drawdown_usd'])} ({stats['max_drawdown_pct']:.2f}%)")
    lines.append(f"Average winner: {stats['avg_win_pips']:.2f} pips / {fmt_money(stats['avg_win_usd'])}")
    lines.append(f"Average loser: {stats['avg_loss_pips']:.2f} pips / {fmt_money(stats['avg_loss_usd'])}")
    lines.append(f"Avg hold time: {stats['avg_hold_minutes']:.2f} minutes")
    lines.append("")
    lines.append("Trades per year:")
    for row in yearly.itertuples(index=False):
        lines.append(f"  - {row.period}: {int(row.trades)} trades | PF {row.profit_factor:.2f} | {fmt_money(row.net_pnl_usd)}")
    lines.append("")
    lines.append("Exit breakdown:")
    for row in exit_breakdown.itertuples(index=False):
        lines.append(f"  - {row.exit_reason}: {int(row.count)} trades | {fmt_money(row.sum)}")
    lines.append("")
    lines.append("Yearly P&L breakdown:")
    for row in yearly.itertuples(index=False):
        lines.append(f"  - {row.period}: {fmt_money(row.net_pnl_usd)} | PF {row.profit_factor:.2f} | profitable={row.net_pnl_usd > 0}")
    lines.append("")
    prof_q = int((quarterly["net_pnl_usd"] > 0).sum()) if not quarterly.empty else 0
    lines.append("Quarterly P&L breakdown:")
    for row in quarterly.itertuples(index=False):
        lines.append(f"  - {row.period}: {fmt_money(row.net_pnl_usd)} | PF {row.profit_factor:.2f} | profitable={row.net_pnl_usd > 0}")
    lines.append(f"Profitable quarters: {prof_q}/{len(quarterly)}")
    lines.append("")
    lines.append("COMPARISON:")
    lines.append("  Standalone:  221 trades | 79.6% WR | PF 3.52 | +$10,110")
    lines.append("  V1 BB:        48 trades | 39.6% WR | PF 0.33 | -$1,649")
    lines.append(
        f"  V2 BB:       {stats['trade_count']:>3d} trades | {stats['win_rate']:.1f}% WR | PF {stats['profit_factor']:.2f} | {fmt_money(stats['net_pnl_usd'])}"
    )
    lines.append("")
    lines.append("TRADE MATCHING:")
    lines.append(f"  Matched: {matching['matched']} / 221")
    lines.append(f"  BB only: {matching['bb_only']}")
    lines.append(f"  Standalone only: {matching['standalone_only']}")
    lines.append(f"  Entry price mean diff: {matching['entry_price_diff_mean_pips']:.2f} pips")
    lines.append(f"  Exit price mean diff: {matching['exit_price_diff_mean_pips']:.2f} pips")
    lines.append(f"  P&L correlation: {matching['pnl_corr']:.4f}")
    lines.append(f"  Side mismatches: {matching['side_mismatches']}")
    lines.append("")
    lines.append("COMBINED V7.1 + V4 EQUITY CURVE:")
    lines.append(
        f"  V7.1 alone:    {combined['v7_trade_count']} trades | PF {combined['v7_pf']:.2f} | {fmt_money(combined['v7_net'])} | MaxDD {fmt_money(combined['v7_maxdd'])}"
    )
    lines.append(
        f"  V4 alone:      {combined['v4_trade_count']} trades | PF {combined['v4_pf']:.2f} | {fmt_money(combined['v4_net'])} | MaxDD {fmt_money(combined['v4_maxdd'])}"
    )
    lines.append(
        f"  Combined:      {combined['combined_trade_count']} trades | PF {combined['combined_pf']:.2f} | {fmt_money(combined['combined_net'])} | MaxDD {fmt_money(combined['combined_maxdd'])}"
    )
    return "\n".join(lines) + "\n"


def build_matching_text(matching: dict[str, object]) -> str:
    return "\n".join(
        [
            "SPIKE FADE V4 V2 — MATCHING VS STANDALONE",
            "=========================================",
            f"Matched: {matching['matched']} / 221",
            f"BB only: {matching['bb_only']}",
            f"Standalone only: {matching['standalone_only']}",
            f"Entry price mean diff: {matching['entry_price_diff_mean_pips']:.4f} pips",
            f"Exit price mean diff: {matching['exit_price_diff_mean_pips']:.4f} pips",
            f"P&L correlation: {matching['pnl_corr']:.6f}",
            f"Side mismatches: {matching['side_mismatches']}",
        ]
    ) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading M1 from {M1_PATH}")
    m1 = load_m1(M1_PATH)
    print(f"Loaded {len(m1):,} M1 bars")
    start = time.time()
    trades, funnel, meta = run_backtest(m1)
    elapsed = time.time() - start
    print(f"Backtest finished in {elapsed:,.1f}s")

    if not trades.empty:
        trades.to_csv(TRADES_OUT, index=False)
    else:
        pd.DataFrame().to_csv(TRADES_OUT, index=False)

    standalone = pd.read_csv(STANDALONE_PATH)
    matching = match_trades(standalone, trades)
    combined = combined_equity(V7_PATH, trades)
    summary = build_summary(trades, funnel, matching, combined, meta, standalone)
    matching_text = build_matching_text(matching)

    SUMMARY_OUT.write_text(summary, encoding="utf-8")
    MATCHING_OUT.write_text(matching_text, encoding="utf-8")

    print(summary)
    print(f"Wrote {TRADES_OUT}")
    print(f"Wrote {SUMMARY_OUT}")
    print(f"Wrote {MATCHING_OUT}")


if __name__ == "__main__":
    main()
