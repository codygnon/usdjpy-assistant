"""
One-bar advance for London V2 inside the v7 defended unified M1 loop.

Ports the per-bar body of scripts/backtest_v2_multisetup_london.run_backtest with:
- Unified margin: pending fills require req_margin <= margin_avail_unified (caller).
- Unified open cap: len(london_positions) + extra_open_positions >= max_open skips.
- Setup D: does not open a native Position; returns an L1 payload for the parent to
  attach L1Incremental (spread 0.3 pipeline vs realistic profile at execute bar).
- Causal asian/LOR levels are supplied by the caller each bar.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import pandas as pd

from core.london_v2_entry_evaluator import evaluate_london_v2_entry_signal

from scripts.backtest_v2_multisetup_london import (
    Position,
    calc_leg_pnl,
    clamp_sl,
    compute_spread_pips,
    day_name,
    pip_value_per_unit,
    to_bid_ask,
    uk_london_open_utc,
    us_ny_open_utc,
)

PIP_SIZE = 0.01
ROUND_UNITS = 100


@dataclass
class LondonUnifiedDayState:
    cfg: dict[str, Any]
    day_utc: pd.Timestamp
    windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]]
    channels: dict[tuple[str, str], dict[str, Any]]
    b_candidates: dict[str, list[dict[str, Any]]]
    pending_entries: list[dict[str, Any]]
    open_positions: list[Position]
    daily_trade_sequence: int = 0
    day_entries_total: int = 0
    day_entries_setup: dict[str, int] = field(default_factory=lambda: {"A": 0, "B": 0, "C": 0, "D": 0})
    day_entries_setup_dir: dict[tuple[str, str], int] = field(
        default_factory=lambda: {("A", "long"): 0, ("A", "short"): 0, ("B", "long"): 0, ("B", "short"): 0, ("C", "long"): 0, ("C", "short"): 0, ("D", "long"): 0, ("D", "short"): 0}
    )
    london_open: pd.Timestamp = field(default_factory=lambda: pd.Timestamp(0, tz="UTC"))
    ny_open: pd.Timestamp = field(default_factory=lambda: pd.Timestamp(0, tz="UTC"))
    hard_close: pd.Timestamp = field(default_factory=lambda: pd.Timestamp(0, tz="UTC"))
    tp1_runner_hard_close: pd.Timestamp = field(default_factory=lambda: pd.Timestamp(0, tz="UTC"))
    diagnostics: dict[str, Any] = field(default_factory=dict)


def init_london_day_state(day_utc: pd.Timestamp, cfg: dict[str, Any]) -> Optional[LondonUnifiedDayState]:
    active_days = set(cfg["session"]["active_days_utc"])
    dnorm = pd.Timestamp(day_utc).normalize()
    if dnorm.day_name() not in active_days:
        return None
    day_start = dnorm
    london_h = uk_london_open_utc(day_start)
    ny_h = us_ny_open_utc(day_start)
    london_open = day_start + pd.Timedelta(hours=london_h)
    ny_open = day_start + pd.Timedelta(hours=ny_h)
    hard_close = ny_open
    tp1_runner_hard_close = ny_open + pd.Timedelta(
        minutes=int(cfg["session"].get("tp1_runner_hard_close_delay_minutes_after_ny_open", 0) or 0)
    )

    def _window_for_setup(setup_key: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        s = cfg["setups"][setup_key]
        start = london_open + pd.Timedelta(minutes=int(s["entry_start_min_after_london"]))
        if s.get("entry_end_min_before_ny") is not None:
            end = ny_open - pd.Timedelta(minutes=int(s["entry_end_min_before_ny"]))
        else:
            end = london_open + pd.Timedelta(minutes=int(s["entry_end_min_after_london"]))
        if end > ny_open:
            end = ny_open
        return start, end

    windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {
        "A": _window_for_setup("A"),
        "B": _window_for_setup("B"),
        "C": _window_for_setup("C"),
        "D": _window_for_setup("D"),
    }
    channels: dict[tuple[str, str], dict[str, Any]] = {}
    for setup in ["A", "B", "C", "D"]:
        for direc in ["long", "short"]:
            channels[(setup, direc)] = {
                "state": "ARMED",
                "cooldown_until": None,
                "entries": 0,
                "resets": 0,
            }
    diagnostics = {
        "trades_skipped_margin_constraint": 0,
        "trades_skipped_open_risk_cap": 0,
        "trades_skipped_entry_limits": 0,
    }
    return LondonUnifiedDayState(
        cfg=cfg,
        day_utc=dnorm,
        windows=windows,
        channels=channels,
        b_candidates={"long": [], "short": []},
        pending_entries=[],
        open_positions=[],
        london_open=london_open,
        ny_open=ny_open,
        hard_close=hard_close,
        tp1_runner_hard_close=tp1_runner_hard_close,
        diagnostics=diagnostics,
    )


def advance_london_unified_bar(
    state: LondonUnifiedDayState,
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
    exec_gate: Optional[Callable[[dict[str, Any], pd.Timestamp], tuple[bool, str]]] = None,
) -> tuple[float, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Returns
    -------
    equity_after : float
    trades_closed : native-style dicts (parent maps to CSV)
    l1_open_payloads : dicts with keys for parent L1Incremental build
    diag_events : optional single-element dicts for debugging
    """
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

    sp = compute_spread_pips(i_global, ts, cfg)
    bid_o, ask_o = to_bid_ask(float(row["open"]), sp)
    bid_h, ask_h = to_bid_ask(float(row["high"]), sp)
    bid_l, ask_l = to_bid_ask(float(row["low"]), sp)
    bid_c, ask_c = to_bid_ask(float(row["close"]), sp)

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

    def mark_post_exit(p: Position, t: pd.Timestamp) -> None:
        c = channels[(p.setup_type, p.direction)]
        if bool(cfg.get("entry_limits", {}).get("disable_channel_reset_after_exit", False)):
            c["state"] = "FIRED"
            c["cooldown_until"] = None
            return
        if p.setup_type in ("B", "C"):
            c["state"] = "WAITING_RESET"
            c["cooldown_until"] = t + pd.Timedelta(minutes=int(cfg["setups"][p.setup_type]["reenter_cooldown_minutes"]))
        else:
            c["state"] = "WAITING_RESET"
            c["cooldown_until"] = None

    # --- execute pending ---
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
        bid_o_e, ask_o_e = to_bid_ask(float(row["open"]), exec_spread)
        entry_price = ask_o_e if direction == "long" else bid_o_e
        s_cfg = cfg["setups"][setup]
        sl, sl_pips = clamp_sl(
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
            math.floor(risk_usd / max(1e-9, sl_pips * pip_value_per_unit(entry_price)) / ROUND_UNITS) * ROUND_UNITS
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
            p = Position(
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
                day_name=day_name(ts),
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

    # --- hard close ---
    if ts == hard_close or ts == tp1_runner_hard_close:
        survivors_after_forced_close: list[Position] = []
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
            lp, lu = calc_leg_pnl(p.direction, p.entry_price, float(exit_px), u)
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
            trades_closed.append(_position_to_trade_dict(p, start_balance))
            mark_post_exit(p, ts)
        state.open_positions = survivors_after_forced_close
        if ts == tp1_runner_hard_close:
            return equity, trades_closed, l1_open_payloads, diag_events
        if ts == hard_close and tp1_runner_hard_close <= hard_close:
            return equity, trades_closed, l1_open_payloads, diag_events

    # --- manage open positions ---
    survivors: list[Position] = []
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
            lp, lu = calc_leg_pnl(p.direction, p.entry_price, float(exit_px), u)
            p.pnl_usd_realized += lu
            p.weighted_pips_sum += lp * u
            p.remaining_units = 0
            p.exit_time = ts
            p.exit_price_last = float(exit_px)
            p.exit_reason = reason
            equity += p.pnl_usd_realized
            trades_closed.append(_position_to_trade_dict(p, start_balance))
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
                    lp, lu = calc_leg_pnl(p.direction, p.entry_price, p.tp1_price, u_close)
                    p.pnl_usd_realized += lu
                    p.weighted_pips_sum += lp * u_close
                    p.remaining_units -= u_close
                p.tp1_hit = True
                p.tp1_time = ts
                be_offset = float(cfg["setups"][p.setup_type]["be_offset_pips"])
                p.be_price_after_tp1 = p.entry_price + (be_offset * PIP_SIZE if p.direction == "long" else -be_offset * PIP_SIZE)
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

    # --- channel reset ---
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

    # --- signal generation ---
    if nxt_ts is not None:
        _lv2_entries, state.b_candidates = evaluate_london_v2_entry_signal(
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
            state.pending_entries.append(_pe)
            channels[(_pe["setup_type"], _pe["direction"])]["state"] = "PENDING"

    return equity, trades_closed, l1_open_payloads, diag_events


def _position_to_trade_dict(p: Position, start_balance: float) -> dict[str, Any]:
    return {
        "session": "london_v2",
        "trade_id": p.trade_id,
        "setup_type": p.setup_type,
        "direction": p.direction,
        "entry_time": p.entry_time,
        "exit_time": p.exit_time,
        "entry_price": p.entry_price,
        "exit_price": p.exit_price_last,
        "sl_price": p.sl_price,
        "tp1_price": p.tp1_price,
        "tp2_price": p.tp2_price,
        "sl_pips": p.sl_pips,
        "risk_pct": p.risk_pct,
        "pnl_pips": (p.weighted_pips_sum / p.initial_units) if p.initial_units > 0 else 0.0,
        "pnl_usd": p.pnl_usd_realized,
        "exit_reason": p.exit_reason,
        "asian_range_pips": p.asian_range_pips,
        "lor_range_pips": p.lor_range_pips,
        "MAE_pips": p.mae_pips,
        "MFE_pips": p.mfe_pips,
        "position_units": p.initial_units,
        "trade_sequence_number": p.trade_sequence_number,
        "is_reentry": bool(p.is_reentry),
        "entry_hour_utc": p.entry_hour_utc,
        "margin_used_pct": p.margin_used_pct,
        "margin_required_usd": p.margin_required_usd,
        "starting_balance": start_balance,
    }
