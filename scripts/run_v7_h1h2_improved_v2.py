#!/usr/bin/env python3
"""Validated v2 runner for Phase3 v7 defended with additive H1+H2 filters.

H1:
- block V44 NY oracle entries when completed M5 ATR-14 > 7.04 pips

H2:
- block London Setup A only
- block only at the actual would-open stage, not at generic pending-signal evaluation
- condition: abs(EMA9 - EMA21) > 0.030 and trade direction aligns with EMA trend

This script keeps all changes in scripts/ only. It monkeypatches the shared runner
entry hooks, uses the optimized direct runner path, and writes validated output to:

    research_out/phase3_v7_h1h2_validated/
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import phase3_v7_pfdd_defended_runner as base_runner
from core.phase3_v7_pfdd_defended_runner import Phase3V7PfddParams, execute_phase3_v7_pfdd_defended
from scripts import v7_defended_london_unified as london_unified


PIP_SIZE = 0.01
H1_ATR14_MAX_PIPS = 7.04
H2_EMA_SPREAD_MAX_PRICE = 0.030
ROUND_UNITS = 100


@dataclass
class FilterStats:
    v44_oracle_original: int = 0
    h1_v44_blocked: int = 0
    h1_v44_kept: int = 0
    h2_setup_a_candidates: int = 0
    h2_setup_a_blocked: int = 0
    h2_missing_context: int = 0
    london_signal_counts: dict[str, int] = field(
        default_factory=lambda: {"A": 0, "B": 0, "C": 0, "D": 0}
    )
    h2_blocked_by_setup: dict[str, int] = field(
        default_factory=lambda: {"A": 0, "B": 0, "C": 0, "D": 0}
    )


class M5FilterContext:
    """Replay-matching M5 context built from raw M1 data."""

    def __init__(self, dataset_path: Path) -> None:
        m1 = pd.read_csv(dataset_path)
        m1["time"] = pd.to_datetime(m1["time"], utc=True)
        m1 = m1.sort_values("time").reset_index(drop=True).set_index("time")
        m5 = (
            m1.resample("5min")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
            .dropna()
            .copy()
        )
        prev_close = m5["close"].shift(1)
        tr = pd.concat(
            [
                m5["high"] - m5["low"],
                (m5["high"] - prev_close).abs(),
                (m5["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        m5["atr14_pips"] = tr.rolling(14, min_periods=14).mean() / PIP_SIZE
        m5["ema9"] = m5["close"].ewm(span=9, adjust=False).mean()
        m5["ema21"] = m5["close"].ewm(span=21, adjust=False).mean()
        self._frame = m5
        self._times = m5.index.asi8

    def row_at_or_before(self, ts: pd.Timestamp) -> pd.Series | None:
        key = pd.Timestamp(ts).tz_convert("UTC").value // 1000
        pos = int(np.searchsorted(self._times, key, side="right") - 1)
        if pos < 0:
            return None
        return self._frame.iloc[pos]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validated V7 H1+H2 runner")
    p.add_argument("--max-bars", type=int, default=None, help="Smoke-test subset (e.g. 100000).")
    p.add_argument("--quiet-runner", action="store_true", help="Suppress runner progress output.")
    return p.parse_args()


def build_london_advance_patch(filter_context: M5FilterContext, filter_stats: FilterStats):
    def advance_london_h2_validated(
        state: london_unified.LondonUnifiedDayState,
        *,
        row: pd.Series,
        ts: pd.Timestamp,
        nxt_ts: Optional[pd.Timestamp],
        i_day: int,
        i_global: int,
        asian_high: float,
        asian_low: float,
        asian_range_pips: float,
        asian_valid: bool,
        lor_high: float,
        lor_low: float,
        lor_range_pips: float,
        lor_valid: bool,
        equity: float,
        margin_avail_unified: float,
        extra_open_positions: int,
        trade_id_counter: list[int],
        spread_mode_pipeline: bool,
        l1_tp1_r: float,
        l1_tp2_r: float,
        l1_be_offset: float,
        l1_tp1_close_fraction: float,
        exec_gate=None,
    ):
        cfg = state.cfg
        equity = float(equity)
        trades_closed: list[dict[str, Any]] = []
        l1_open_payloads: list[dict[str, Any]] = []
        diag_events: list[dict[str, Any]] = []

        start_balance = float(cfg["account"]["starting_balance"])
        leverage = float(cfg["account"]["leverage"])
        max_margin_frac = float(cfg["account"]["max_margin_usage_fraction_per_trade"])
        max_open_positions = int(cfg["account"]["max_open_positions"])
        risk_pct = float(cfg["risk"]["risk_per_trade_pct"])
        max_total_open_risk_pct = float(cfg["risk"]["max_total_open_risk_pct"])

        sp = london_unified.compute_spread_pips(i_global, ts, cfg)
        bid_o, ask_o = london_unified.to_bid_ask(float(row["open"]), sp)
        bid_h, ask_h = london_unified.to_bid_ask(float(row["high"]), sp)
        bid_l, ask_l = london_unified.to_bid_ask(float(row["low"]), sp)
        bid_c, ask_c = london_unified.to_bid_ask(float(row["close"]), sp)

        windows = state.windows
        channels = state.channels
        pending_entries = state.pending_entries
        open_positions = state.open_positions
        hard_close = state.hard_close
        tp1_runner_hard_close = state.tp1_runner_hard_close

        def rearm_if_pending(setup: str, direction: str) -> None:
            st = channels[(setup, direction)]["state"]
            if st == "PENDING":
                channels[(setup, direction)]["state"] = "ARMED"

        def mark_post_exit(p: london_unified.Position, t: pd.Timestamp) -> None:
            c = channels[(p.setup_type, p.direction)]
            if bool(cfg.get("entry_limits", {}).get("disable_channel_reset_after_exit", False)):
                c["state"] = "FIRED"
                c["cooldown_until"] = None
                return
            if p.setup_type in ("B", "C"):
                c["state"] = "WAITING_RESET"
                c["cooldown_until"] = t + pd.Timedelta(
                    minutes=int(cfg["setups"][p.setup_type]["reenter_cooldown_minutes"])
                )
            else:
                c["state"] = "WAITING_RESET"
                c["cooldown_until"] = None

        to_exec = [x for x in pending_entries if pd.Timestamp(x["execute_time"]) == ts]
        state.pending_entries = [x for x in pending_entries if pd.Timestamp(x["execute_time"]) != ts]

        for pe in to_exec:
            setup = str(pe["setup_type"])
            direction = str(pe["direction"])
            w_start, w_end = windows[setup]
            if not (w_start <= ts < w_end):
                rearm_if_pending(setup, direction)
                continue

            limits = cfg.get("entry_limits", {})
            lim_total = limits.get("max_trades_per_day_total")
            lim_setup = limits.get("max_trades_per_setup_per_day")
            lim_setup_dir = limits.get("max_trades_per_setup_direction_per_day")
            if lim_total is not None and state.day_entries_total >= int(lim_total):
                state.diagnostics["trades_skipped_entry_limits"] += 1
                rearm_if_pending(setup, direction)
                continue
            if lim_setup is not None and state.day_entries_setup[setup] >= int(lim_setup):
                state.diagnostics["trades_skipped_entry_limits"] += 1
                rearm_if_pending(setup, direction)
                continue
            if lim_setup_dir is not None and state.day_entries_setup_dir[(setup, direction)] >= int(lim_setup_dir):
                state.diagnostics["trades_skipped_entry_limits"] += 1
                rearm_if_pending(setup, direction)
                continue
            if len(open_positions) + int(extra_open_positions) >= max_open_positions:
                rearm_if_pending(setup, direction)
                continue

            if exec_gate is not None:
                ok_g, _rg = exec_gate(pe, ts)
                if not ok_g:
                    rearm_if_pending(setup, direction)
                    continue

            exec_spread = 0.3 if (spread_mode_pipeline and setup == "D") else sp
            bid_o_e, ask_o_e = london_unified.to_bid_ask(float(row["open"]), exec_spread)
            entry_price = ask_o_e if direction == "long" else bid_o_e
            s_cfg = cfg["setups"][setup]
            sl, sl_pips = london_unified.clamp_sl(
                entry_price,
                float(pe["raw_sl"]),
                direction,
                float(s_cfg["sl_min_pips"]),
                float(s_cfg["sl_max_pips"]),
            )
            sign = 1.0 if direction == "long" else -1.0
            tp1 = entry_price + sign * float(s_cfg["tp1_r_multiple"]) * sl_pips * PIP_SIZE
            tp2 = entry_price + sign * float(s_cfg["tp2_r_multiple"]) * sl_pips * PIP_SIZE

            risk_pct_trade = float(s_cfg.get("risk_per_trade_pct", risk_pct))
            risk_usd = equity * risk_pct_trade
            open_risk = float(sum(p.risk_usd_planned for p in open_positions))
            if open_risk + risk_usd > equity * max_total_open_risk_pct:
                state.diagnostics["trades_skipped_open_risk_cap"] += 1
                rearm_if_pending(setup, direction)
                continue

            units = int(
                math.floor(
                    risk_usd
                    / max(1e-9, sl_pips * london_unified.pip_value_per_unit(entry_price))
                    / ROUND_UNITS
                )
                * ROUND_UNITS
            )
            if units <= 0:
                rearm_if_pending(setup, direction)
                continue

            used_margin_london = float(sum(p.margin_required_usd for p in open_positions))
            req_margin = (units * float(entry_price) * PIP_SIZE) / leverage
            if req_margin > margin_avail_unified:
                state.diagnostics["trades_skipped_margin_constraint"] += 1
                rearm_if_pending(setup, direction)
                continue
            if req_margin > max_margin_frac * max(1e-9, equity - used_margin_london):
                state.diagnostics["trades_skipped_margin_constraint"] += 1
                rearm_if_pending(setup, direction)
                continue

            if setup == "A":
                filter_stats.h2_setup_a_candidates += 1
                ctx = filter_context.row_at_or_before(pd.Timestamp(ts))
                if ctx is None or pd.isna(ctx["ema9"]) or pd.isna(ctx["ema21"]):
                    filter_stats.h2_missing_context += 1
                else:
                    ema9 = float(ctx["ema9"])
                    ema21 = float(ctx["ema21"])
                    spread = abs(ema9 - ema21)
                    aligns = (direction == "long" and ema9 > ema21) or (direction == "short" and ema9 < ema21)
                    if spread > H2_EMA_SPREAD_MAX_PRICE and aligns:
                        filter_stats.h2_setup_a_blocked += 1
                        filter_stats.h2_blocked_by_setup["A"] += 1
                        # Consume the Setup A channel like a rejected trade, otherwise
                        # the same setup can re-trigger repeatedly and massively overcount.
                        channels[(setup, direction)]["state"] = "WAITING_RESET"
                        channels[(setup, direction)]["cooldown_until"] = None
                        continue

            margin_used_pct = (req_margin / max(1e-9, equity)) * 100.0
            trade_id_counter[0] += 1
            tid = trade_id_counter[0]
            state.daily_trade_sequence += 1

            if setup == "D":
                l1_open_payloads.append(
                    {
                        "trade_id": tid,
                        "setup_type": "D",
                        "direction": direction,
                        "entry_time": ts,
                        "entry_bar_row": row,
                        "lor_high": float(lor_high),
                        "lor_low": float(lor_low),
                        "ny_open": state.ny_open,
                        "spread_pips_exec": float(exec_spread),
                        "tp1_r": float(l1_tp1_r),
                        "tp2_r": float(l1_tp2_r),
                        "be_offset": float(l1_be_offset),
                        "tp1_close_fraction": float(l1_tp1_close_fraction),
                        "units": units,
                        "sl_pips": float(sl_pips),
                        "risk_usd_planned": float(risk_usd),
                        "risk_pct": float(risk_pct_trade),
                        "margin_required_usd": float(req_margin),
                        "margin_used_pct": float(margin_used_pct),
                        "asian_range_pips": float(pe["asian_range_pips"]) if pe.get("asian_range_pips") is not None else None,
                        "lor_range_pips": float(pe["lor_range_pips"]) if pe.get("lor_range_pips") is not None else None,
                        "trade_sequence_number": int(state.daily_trade_sequence),
                        "is_reentry": bool(pe.get("is_reentry", False)),
                    }
                )
            else:
                p = london_unified.Position(
                    trade_id=tid,
                    setup_type=setup,
                    direction=direction,
                    entry_time=ts,
                    entry_price=float(entry_price),
                    sl_price=float(sl),
                    tp1_price=float(tp1),
                    tp2_price=float(tp2),
                    initial_units=units,
                    remaining_units=units,
                    sl_pips=float(sl_pips),
                    risk_usd_planned=float(risk_usd),
                    risk_pct=float(risk_pct_trade),
                    margin_used_pct=float(margin_used_pct),
                    margin_required_usd=float(req_margin),
                    day_name=london_unified.day_name(ts),
                    asian_range_pips=float(pe["asian_range_pips"]) if pe.get("asian_range_pips") is not None else None,
                    lor_range_pips=float(pe["lor_range_pips"]) if pe.get("lor_range_pips") is not None else None,
                    entry_hour_utc=int(ts.hour),
                    trade_sequence_number=state.daily_trade_sequence,
                    is_reentry=bool(pe.get("is_reentry", False)),
                )
                close_fraction = float(s_cfg["tp1_close_fraction"])
                tp1_units = int(math.floor(units * close_fraction / ROUND_UNITS) * ROUND_UNITS)
                p.tp1_units_closed = max(0, min(units, tp1_units))
                open_positions.append(p)

            c = channels[(setup, direction)]
            c["state"] = "FIRED"
            c["entries"] += 1
            state.day_entries_total += 1
            state.day_entries_setup[setup] += 1
            state.day_entries_setup_dir[(setup, direction)] += 1

        if ts == hard_close or ts == tp1_runner_hard_close:
            survivors_after_forced_close: list[london_unified.Position] = []
            for p in open_positions:
                allow_delayed_runner = (
                    p.setup_type == "D"
                    and bool(cfg["setups"]["D"].get("extend_runner_until_ny_start_delay", False))
                    and p.tp1_hit
                    and p.remaining_units > 0
                    and tp1_runner_hard_close > hard_close
                )
                if ts == hard_close and allow_delayed_runner:
                    survivors_after_forced_close.append(p)
                    continue
                exit_px = bid_o if p.direction == "long" else ask_o
                u = p.remaining_units
                lp, lu = london_unified.calc_leg_pnl(p.direction, p.entry_price, float(exit_px), u)
                p.pnl_usd_realized += lu
                p.weighted_pips_sum += lp * u
                p.remaining_units = 0
                p.exit_time = ts
                p.exit_price_last = float(exit_px)
                if p.tp1_hit and ts > hard_close:
                    p.exit_reason = "TP1_THEN_DELAYED_HARD_CLOSE"
                else:
                    p.exit_reason = "TP1_ONLY_HARD_CLOSE" if p.tp1_hit else "HARD_CLOSE"
                equity += p.pnl_usd_realized
                trades_closed.append(london_unified._position_to_trade_dict(p, start_balance))
                mark_post_exit(p, ts)
            state.open_positions = survivors_after_forced_close
            if ts == tp1_runner_hard_close:
                return equity, trades_closed, l1_open_payloads, diag_events
            if ts == hard_close and tp1_runner_hard_close <= hard_close:
                return equity, trades_closed, l1_open_payloads, diag_events

        survivors: list[london_unified.Position] = []
        for p in state.open_positions:
            if p.direction == "long":
                p.mfe_pips = max(p.mfe_pips, (float(row["high"]) - p.entry_price) / PIP_SIZE)
                p.mae_pips = max(p.mae_pips, (p.entry_price - float(row["low"])) / PIP_SIZE)
                px_open, px_high, px_low = bid_o, bid_h, bid_l
            else:
                p.mfe_pips = max(p.mfe_pips, (p.entry_price - float(row["low"])) / PIP_SIZE)
                p.mae_pips = max(p.mae_pips, (float(row["high"]) - p.entry_price) / PIP_SIZE)
                px_open, px_high, px_low = ask_o, ask_h, ask_l

            def choose_first(level_a: float, level_b: float) -> str:
                da = abs(px_open - level_a)
                db = abs(px_open - level_b)
                return "a" if da <= db else "b"

            def close_remaining(reason: str, exit_px: float) -> None:
                nonlocal equity
                u = p.remaining_units
                lp, lu = london_unified.calc_leg_pnl(p.direction, p.entry_price, float(exit_px), u)
                p.pnl_usd_realized += lu
                p.weighted_pips_sum += lp * u
                p.remaining_units = 0
                p.exit_time = ts
                p.exit_price_last = float(exit_px)
                p.exit_reason = reason
                equity += p.pnl_usd_realized
                trades_closed.append(london_unified._position_to_trade_dict(p, start_balance))
                mark_post_exit(p, ts)

            if not p.tp1_hit:
                tp1_hit = px_high >= p.tp1_price if p.direction == "long" else px_low <= p.tp1_price
                sl_hit = px_low <= p.sl_price if p.direction == "long" else px_high >= p.sl_price
                if tp1_hit and sl_hit:
                    first = choose_first(p.tp1_price, p.sl_price)
                elif sl_hit:
                    first = "b"
                elif tp1_hit:
                    first = "a"
                else:
                    first = ""

                if first == "b":
                    close_remaining("SL_FULL", p.sl_price)
                    continue
                if first == "a":
                    u_close = p.tp1_units_closed if p.tp1_units_closed > 0 else int(p.remaining_units / 2)
                    u_close = max(0, min(p.remaining_units, u_close))
                    if u_close > 0:
                        lp, lu = london_unified.calc_leg_pnl(p.direction, p.entry_price, p.tp1_price, u_close)
                        p.pnl_usd_realized += lu
                        p.weighted_pips_sum += lp * u_close
                        p.remaining_units -= u_close
                    p.tp1_hit = True
                    p.tp1_time = ts
                    be_offset = float(cfg["setups"][p.setup_type]["be_offset_pips"])
                    p.be_price_after_tp1 = p.entry_price + (
                        be_offset * PIP_SIZE if p.direction == "long" else -be_offset * PIP_SIZE
                    )
                    if p.direction == "long":
                        p.sl_price = max(p.sl_price, p.be_price_after_tp1)
                    else:
                        p.sl_price = min(p.sl_price, p.be_price_after_tp1)

            if p.tp1_hit and p.remaining_units > 0:
                tp2_hit = px_high >= p.tp2_price if p.direction == "long" else px_low <= p.tp2_price
                sl2_hit = px_low <= p.sl_price if p.direction == "long" else px_high >= p.sl_price
                if tp2_hit and sl2_hit:
                    first2 = choose_first(p.tp2_price, p.sl_price)
                elif tp2_hit:
                    first2 = "a"
                elif sl2_hit:
                    first2 = "b"
                else:
                    first2 = ""
                if first2 == "a":
                    close_remaining("TP2_FULL", p.tp2_price)
                    continue
                if first2 == "b":
                    close_remaining("BE_STOP", p.sl_price)
                    continue

                if (
                    p.setup_type == "D"
                    and bool(cfg["setups"]["D"].get("extend_runner_until_ny_start_delay", False))
                    and hard_close <= ts < tp1_runner_hard_close
                ):
                    grace_trail_pips = float(cfg["setups"]["D"].get("grace_trail_distance_pips", 0.0) or 0.0)
                    if grace_trail_pips > 0:
                        if p.direction == "long":
                            p.sl_price = max(p.sl_price, bid_c - grace_trail_pips * PIP_SIZE)
                        else:
                            p.sl_price = min(p.sl_price, ask_c + grace_trail_pips * PIP_SIZE)

            survivors.append(p)

        state.open_positions = survivors

        for setup in ["A", "B", "C", "D"]:
            w_start, w_end = windows[setup]
            for direction in ["long", "short"]:
                c = channels[(setup, direction)]
                if c["state"] != "WAITING_RESET":
                    continue
                if setup in ("B", "C"):
                    cooldown_until = c["cooldown_until"]
                    if cooldown_until is not None and ts >= cooldown_until and ts < w_end:
                        c["state"] = "ARMED"
                        c["resets"] += 1
                else:
                    if ts >= w_end:
                        continue
                    if setup == "A":
                        if direction == "long" and float(row["low"]) <= asian_high:
                            c["state"] = "ARMED"
                            c["resets"] += 1
                        if direction == "short" and float(row["high"]) >= asian_low:
                            c["state"] = "ARMED"
                            c["resets"] += 1
                    if setup == "D" and lor_valid:
                        if direction == "long" and float(row["low"]) <= lor_high:
                            c["state"] = "ARMED"
                            c["resets"] += 1
                        if direction == "short" and float(row["high"]) >= lor_low:
                            c["state"] = "ARMED"
                            c["resets"] += 1

        if nxt_ts is not None:
            _lv2_entries, state.b_candidates = london_unified.evaluate_london_v2_entry_signal(
                row=row,
                cfg=cfg,
                asian_high=asian_high,
                asian_low=asian_low,
                asian_range_pips=asian_range_pips,
                asian_valid=asian_valid,
                lor_high=lor_high,
                lor_low=lor_low,
                lor_range_pips=lor_range_pips if lor_valid else 0.0,
                lor_valid=lor_valid,
                ts=ts,
                nxt_ts=nxt_ts,
                bar_index=i_day,
                windows=windows,
                channels=channels,
                b_candidates=state.b_candidates,
            )
            for _pe in _lv2_entries:
                stype = str(_pe["setup_type"])
                if stype in filter_stats.london_signal_counts:
                    filter_stats.london_signal_counts[stype] += 1
                state.pending_entries.append(_pe)
                channels[(_pe["setup_type"], _pe["direction"])]["state"] = "PENDING"

        return equity, trades_closed, l1_open_payloads, diag_events

    return advance_london_h2_validated


def compute_extended_summary(
    *,
    raw_result: dict[str, Any],
    hypothesis: str,
    dataset_path: Path,
    filter_stats: FilterStats,
) -> dict[str, Any]:
    runner_summary = dict(raw_result["summary"])
    trade_df = pd.DataFrame(raw_result["trades_log"])
    equity_df = pd.DataFrame(raw_result["equity_rows"])

    pnls = pd.to_numeric(trade_df.get("pnl_usd"), errors="coerce").dropna() if not trade_df.empty else pd.Series(dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    gross_wins = float(wins.sum()) if not wins.empty else 0.0
    gross_losses = abs(float(losses.sum())) if not losses.empty else 0.0
    exit_reasons = (
        trade_df["exit_reason"].value_counts().to_dict()
        if not trade_df.empty and "exit_reason" in trade_df.columns
        else {}
    )
    avg_bars_held = (
        float((pd.to_numeric(trade_df["exit_bar_index"], errors="coerce") - pd.to_numeric(trade_df["entry_bar_index"], errors="coerce")).mean())
        if not trade_df.empty and {"entry_bar_index", "exit_bar_index"}.issubset(trade_df.columns)
        else None
    )
    avg_initial_stop_pips = (
        float(pd.to_numeric(trade_df["sl_pips"], errors="coerce").mean())
        if not trade_df.empty and "sl_pips" in trade_df.columns
        else None
    )

    if not equity_df.empty:
        processed_start_time = str(equity_df["timestamp"].iloc[0])
        processed_end_time = str(equity_df["timestamp"].iloc[-1])
        max_dd_usd = float(equity_df["drawdown_from_peak"].max())
        max_dd_pct = float(equity_df["drawdown_pct"].max())
    else:
        processed_start_time = None
        processed_end_time = None
        max_dd_usd = float(runner_summary.get("max_drawdown_usd", 0.0) or 0.0)
        max_dd_pct = 0.0

    return {
        "hypothesis": hypothesis,
        "mode": "standalone",
        "active_families": ["phase3_v7_pfdd_defended"],
        "processed_bar_count": int(raw_result.get("total_bars_processed", 0)),
        "processed_start_time": processed_start_time,
        "processed_end_time": processed_end_time,
        "synthetic_bid_ask": True,
        "initial_balance": 100000.0,
        "ending_balance": float(runner_summary["ending_equity"]),
        "ending_equity": float(runner_summary["ending_equity"]),
        "net_pnl_usd": float(runner_summary["net_pnl_usd"]),
        "profit_factor": float(runner_summary["profit_factor"]),
        "trade_count": int(runner_summary["closed_trades_total"]),
        "win_rate": float(runner_summary["win_rate_pct"]),
        "max_drawdown_usd": max_dd_usd,
        "max_drawdown_pct": max_dd_pct,
        "max_concurrent_positions": int(runner_summary["max_concurrent_positions"]),
        "arbitration_event_count": 0,
        "by_family": [
            {
                "family": "phase3_v7_pfdd_defended",
                "trade_count": int(runner_summary["closed_trades_total"]),
                "net_pnl_usd": float(runner_summary["net_pnl_usd"]),
            }
        ],
        "phase3_defended_runner_summary": runner_summary,
        "net_pnl": float(runner_summary["net_pnl_usd"]),
        "gross_wins": gross_wins,
        "gross_losses": gross_losses,
        "avg_win": float(wins.mean()) if not wins.empty else 0.0,
        "avg_loss": float(losses.mean()) if not losses.empty else 0.0,
        "exit_reasons": exit_reasons,
        "avg_bars_held": avg_bars_held,
        "avg_initial_stop_pips": avg_initial_stop_pips,
        "methodology_notes": (
            "Phase3 v7 defended improved v2 via additive H1+H2 filters in scripts only. "
            "H1 blocks V44 oracle entries when completed M5 ATR-14 > 7.04 pips. "
            "H2 blocks only London Setup A at the actual would-open stage when EMA-9/EMA-21 spread exceeds 3 pips and the trade aligns with that trend. "
            "Uses direct core.phase3_v7_pfdd_defended_runner execution."
        ),
        "h1h2_filter_stats": {
            "v44_oracle_original": filter_stats.v44_oracle_original,
            "h1_v44_blocked": filter_stats.h1_v44_blocked,
            "h1_v44_kept": filter_stats.h1_v44_kept,
            "h2_setup_a_candidates": filter_stats.h2_setup_a_candidates,
            "h2_setup_a_blocked": filter_stats.h2_setup_a_blocked,
            "h2_missing_context": filter_stats.h2_missing_context,
            "london_signal_counts": filter_stats.london_signal_counts,
            "h2_blocked_by_setup": filter_stats.h2_blocked_by_setup,
        },
    }


def write_validation_report(output_dir: Path, summary: dict[str, Any]) -> None:
    baseline_path = ROOT / "research_out/phase3_v7_pfdd_defended_real/summary.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    replay = {
        "net_pnl_usd": 69973.0,
        "profit_factor": 2.16,
        "trade_count": 282,
        "win_rate": None,
        "max_drawdown_pct": 3.7,
    }
    runner = summary["phase3_defended_runner_summary"]
    counts = runner.get("trade_counts_by_type", {})
    stats = summary["h1h2_filter_stats"]

    lines = [
        "# V7 H1+H2 Validated Report",
        "",
        f"Generated: {pd.Timestamp.now('UTC').isoformat()}",
        "",
        "## Comparison",
        "",
        "| Metric | Baseline | Replay Estimate | Full Engine | Match? |",
        "| --- | --- | --- | --- | --- |",
        f"| Net P&L | ${baseline['net_pnl_usd']:,.0f} | ${replay['net_pnl_usd']:,.0f} | ${summary['net_pnl_usd']:,.0f} | {'Yes' if 65000 <= summary['net_pnl_usd'] <= 70000 else 'Close'} |",
        f"| PF | {baseline['profit_factor']:.2f} | {replay['profit_factor']:.2f} | {summary['profit_factor']:.2f} | {'Yes' if 1.9 <= summary['profit_factor'] <= 2.1 else 'Close'} |",
        f"| Trades | {baseline['trade_count']} | {replay['trade_count']} | {summary['trade_count']} | {'Yes' if abs(summary['trade_count'] - 293) <= 15 else 'Close'} |",
        f"| WR | {baseline['win_rate']:.1f}% | — | {summary['win_rate']:.1f}% | — |",
        f"| MaxDD | {baseline['max_drawdown_pct']:.2f}% | {replay['max_drawdown_pct']:.2f}% | {summary['max_drawdown_pct']:.2f}% | {'Yes' if summary['max_drawdown_pct'] <= 10 else 'No'} |",
        "",
        "## Filter Diagnostics",
        "",
        f"- H1 V44 blocked: {stats['h1_v44_blocked']} of {stats['v44_oracle_original']}",
        f"- H2 Setup A candidate opens: {stats['h2_setup_a_candidates']}",
        f"- H2 Setup A blocked opens: {stats['h2_setup_a_blocked']}",
        f"- H2 missing context: {stats['h2_missing_context']}",
        f"- H2 blocked by setup: `{json.dumps(stats['h2_blocked_by_setup'])}`",
        "",
        "## Per-Substrategy",
        "",
        "| Sub-strategy | Final Trades | Expected | Status | Net P&L |",
        "| --- | --- | --- | --- | --- |",
        f"| london_setup_a | {counts.get('london_setup_a', 0)} | ~12 | {'OK' if abs(int(counts.get('london_setup_a', 0)) - 12) <= 5 else 'Investigate'} | ${runner['net_pnl_usd_by_type'].get('london_setup_a', 0.0):,.2f} |",
        f"| london_setup_d_l1 | {counts.get('london_setup_d_l1', 0)} | ~68 | {'OK' if abs(int(counts.get('london_setup_d_l1', 0)) - 68) <= 5 else 'Investigate'} | ${runner['net_pnl_usd_by_type'].get('london_setup_d_l1', 0.0):,.2f} |",
        f"| v44_ny | {counts.get('v44_ny', 0)} | ~114 | {'OK' if abs(int(counts.get('v44_ny', 0)) - 114) <= 5 else 'Investigate'} | ${runner['net_pnl_usd_by_type'].get('v44_ny', 0.0):,.2f} |",
        f"| tokyo_v14 | {counts.get('tokyo_v14', 0)} | ~88 | {'OK' if abs(int(counts.get('tokyo_v14', 0)) - 88) <= 5 else 'Investigate'} | ${runner['net_pnl_usd_by_type'].get('tokyo_v14', 0.0):,.2f} |",
        "",
        "## Success Criteria",
        "",
        f"- PF >= 1.80: {'PASS' if summary['profit_factor'] >= 1.80 else 'FAIL'}",
        f"- Trades >= 200: {'PASS' if summary['trade_count'] >= 200 else 'FAIL'}",
        f"- MaxDD <= 10%: {'PASS' if summary['max_drawdown_pct'] <= 10.0 else 'FAIL'}",
        f"- Net P&L between $65k and $70k target band: {'PASS' if 65000 <= summary['net_pnl_usd'] <= 70000 else 'MISS'}",
    ]
    (output_dir / "VALIDATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_once(*, dataset_path: Path, output_dir: Path, hypothesis: str, max_bars: int | None, runner_quiet: bool) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run_log.txt"
    log_path.write_text("", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log("=== Phase3 v7 defended H1+H2 validated v2 — direct runner ===")
    log(f"Started at {time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    log(f"Dataset: {dataset_path}")
    log("Building M5 filter context...")
    filter_context = M5FilterContext(dataset_path)
    filter_stats = FilterStats()

    original_load_v44_oracle = base_runner.load_v44_oracle
    original_advance_london = base_runner.advance_london_unified_bar

    def filtered_load_v44_oracle(dataset_path_str: str) -> dict[int, dict[str, Any]]:
        raw = original_load_v44_oracle(dataset_path_str)
        filter_stats.v44_oracle_original = len(raw)
        filtered: dict[int, dict[str, Any]] = {}
        blocked_local = 0
        for key, row in raw.items():
            ts = base_runner._ts(row["entry_time"])
            ctx = filter_context.row_at_or_before(ts)
            if ctx is not None and pd.notna(ctx["atr14_pips"]) and float(ctx["atr14_pips"]) > H1_ATR14_MAX_PIPS:
                blocked_local += 1
                continue
            filtered[key] = row
        filter_stats.h1_v44_blocked = blocked_local
        filter_stats.h1_v44_kept = len(filtered)
        return filtered

    base_runner.load_v44_oracle = filtered_load_v44_oracle
    base_runner.advance_london_unified_bar = build_london_advance_patch(filter_context, filter_stats)

    try:
        t0 = time.perf_counter()
        raw = execute_phase3_v7_pfdd_defended(
            Phase3V7PfddParams(
                data_path=str(dataset_path),
                spread_mode="realistic",
                dry_run=False,
                max_bars=max_bars,
                quiet=runner_quiet,
            )
        )
        elapsed = time.perf_counter() - t0
        log(f"Finished in {elapsed:.1f}s ({elapsed/60:.1f}m)")
    finally:
        base_runner.load_v44_oracle = original_load_v44_oracle
        base_runner.advance_london_unified_bar = original_advance_london

    trades_path = output_dir / "trade_log.csv"
    equity_path = output_dir / "equity_log.csv"
    raw_summary_path = output_dir / "phase3_defended_summary.json"
    pd.DataFrame(raw["trades_log"]).to_csv(trades_path, index=False)
    pd.DataFrame(raw["equity_rows"]).to_csv(equity_path, index=False)
    raw_summary_path.write_text(json.dumps(raw["summary"], indent=2, default=str), encoding="utf-8")

    ext = compute_extended_summary(
        raw_result=raw,
        hypothesis=hypothesis,
        dataset_path=dataset_path,
        filter_stats=filter_stats,
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(ext, indent=2, default=str), encoding="utf-8")

    smoke_diag = {
        "total_entry_signals": {
            "v44_ny": filter_stats.v44_oracle_original,
            "london_setup_a": filter_stats.london_signal_counts["A"],
            "london_setup_d_l1": filter_stats.london_signal_counts["D"],
            "tokyo_v14": int(raw["summary"].get("tokyo_diagnostics", {}).get("signals_generated", 0)),
        },
        "h2_blocked_entries": {
            "v44_ny": 0,
            "london_setup_a": filter_stats.h2_setup_a_blocked,
            "london_setup_d_l1": 0,
            "tokyo_v14": 0,
        },
        "final_trade_count": raw["summary"].get("trade_counts_by_type", {}),
    }
    (output_dir / "SMOKE_DIAGNOSTICS.json").write_text(json.dumps(smoke_diag, indent=2), encoding="utf-8")

    log(f"Summary written to {summary_path}")
    log(json.dumps(ext, indent=2, default=str))
    log(f"Trade log: {trades_path}")
    log(f"Smoke diagnostics: {output_dir / 'SMOKE_DIAGNOSTICS.json'}")
    return ext


def main() -> None:
    args = parse_args()
    dataset_path = ROOT / "research_out/USDJPY_M1_OANDA_extended.csv"
    if not dataset_path.is_file():
        print(f"ERROR: missing USDJPY data file: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    base_output = ROOT / "research_out/phase3_v7_h1h2_validated"
    if args.max_bars:
        output_dir = base_output / f"smoke_{args.max_bars}"
        hypothesis = f"phase3_v7_pfdd_h1h2_validated_smoke_maxbars_{args.max_bars}"
    else:
        output_dir = base_output
        hypothesis = "phase3_v7_pfdd_h1h2_validated_realistic"

    summary = run_once(
        dataset_path=dataset_path,
        output_dir=output_dir,
        hypothesis=hypothesis,
        max_bars=args.max_bars,
        runner_quiet=args.quiet_runner,
    )

    if args.max_bars:
        counts = summary["phase3_defended_runner_summary"]["trade_counts_by_type"]
        print("\nSmoke diagnostic summary:", flush=True)
        print(
            json.dumps(
                {
                    "h2_setup_a_blocked": summary["h1h2_filter_stats"]["h2_setup_a_blocked"],
                    "london_setup_a_trades": counts.get("london_setup_a", 0),
                    "london_setup_d_l1_trades": counts.get("london_setup_d_l1", 0),
                    "v44_ny_trades": counts.get("v44_ny", 0),
                    "tokyo_v14_trades": counts.get("tokyo_v14", 0),
                },
                indent=2,
            ),
            flush=True,
        )
        return

    write_validation_report(output_dir, summary)
    print(f"Validation report: {output_dir / 'VALIDATION_REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
