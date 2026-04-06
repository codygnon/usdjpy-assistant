#!/usr/bin/env python3
"""Bar-by-bar combined V7.1 + H1 + Spike Fade V4 margin engine.

This script intentionally avoids replaying recorded trades. It runs the actual
defended Phase3/V7 bar-by-bar loop, applies the H1-only V44 filter used by the
promoted package, and embeds the exact validated Spike Fade V4 state machine
inside the same M1 iteration with a shared account and shared margin model.

Notes
- V7 sizing logic is preserved from the defended runner.
- V4 uses a fixed lot size (`--v4-lots`, default 20).
- Shared margin uses one account at a configurable leverage (`--leverage`,
  default 33.0).
- Entry admission uses a realized-balance-based hard margin cap to avoid
  pyramiding off floating MTM.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import math
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import phase3_v7_pfdd_defended_runner as base_runner
from core.phase3_integrated_engine import execute_v44_ny_entry, compute_phase3_ownership_audit_for_data
from core.phase3_overlay_resolver import build_phase3_overlay_state
from core.phase3_package_spec import PHASE3_DEFENDED_PRESET_ID
from core.phase3_v7_pfdd_defended_runner import (
    DROP_WEEKDAYS,
    L1Incremental,
    PIP_SIZE,
    TARGET_CELL_BLOCK,
    TOKYO_CFG,
    V44_CFG_PATH,
    LONDON_CFG_PATH,
    _m5_times_sorted_ns,
    _ts,
    admission_checks,
    assert_m5_no_lookahead_fast,
    build_l1_incremental_from_entry,
    load_v44_oracle,
    pip_value_usd_per_lot,
)
from core.ownership_table import cell_key_from_floats
from core.phase3_variant_k_baseline import build_variant_k_baseline_context
from scripts import backtest_tokyo_meanrev as tokyo_bt
from scripts import backtest_v2_multisetup_london as london_bt
from scripts import backtest_v44_conservative_router as v44_router
from scripts import diagnostic_chart_authorization_loop as auth_loop
from scripts import run_offensive_slice_discovery as discovery
from scripts.backtest_tokyo_meanrev import calc_leg_usd_pips as tokyo_calc_leg_usd_pips
from scripts.backtest_variant_i_pbt_standdown import _is_variant_i_blocked
from scripts.backtest_variant_k_london_cluster import LONDON_BLOCK_CLUSTER
from scripts.run_spike_fade_v4_backtest_v2 import bucket_end, family_c_event, make_m5_bar
from scripts.run_v7_h1h2_improved_v2 import H1_ATR14_MAX_PIPS, M5FilterContext
from scripts.v7_defended_london_unified import LondonUnifiedDayState, advance_london_unified_bar, init_london_day_state
from scripts.v7_defended_tokyo_unified import (
    TokyoState,
    _get_tokyo_spread,
    advance_tokyo_bar,
    compute_tokyo_indicators,
    init_tokyo_config,
)
import scripts.v7_defended_tokyo_unified as tokyo_unified


OUT_DIR = ROOT / "research_out/trade_analysis/spike_fade_v4/backtest/combined_v71_h1_v4_margin_bar_by_bar"
COMBINED_TRADES_OUT = OUT_DIR / "combined_trade_log.csv"
V4_TRADES_OUT = OUT_DIR / "v4_trade_log.csv"
EQUITY_OUT = OUT_DIR / "equity_log.csv"
SUMMARY_JSON_OUT = OUT_DIR / "summary.json"
SUMMARY_TXT_OUT = OUT_DIR / "summary.txt"

V4_ENTRY_SPREAD_PIPS = 1.6
V4_ENTRY_WINDOW_MINUTES = 10
V4_STOP_BUFFER_PIPS = 2.0
V4_STOP_CLAMP_MIN_PIPS = 15.0
V4_STOP_CLAMP_MAX_PIPS = 35.0
V4_TP_FRACTION = 0.50
V4_TRAIL_TRIGGER_PIPS = 10.0
V4_TRAIL_DISTANCE_PIPS = 5.0
V4_PROVE_IT_MINUTES = 15
V4_PROVE_IT_PIPS = -5.0
V4_TIME_STOP_MINUTES = 240


@dataclass
class CombinedParams:
    data_path: str
    starting_equity: float = 100_000.0
    leverage: float = 33.0
    v4_lots: float = 20.0
    entry_margin_cap_pct: float = 100.0
    spread_mode: str = "realistic"
    max_bars: Optional[int] = None
    quiet: bool = False


@dataclass
class V4PendingOrder:
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
    session_name: str
    units: int
    margin_required_usd: float


@dataclass
class V4Position:
    entry_time: pd.Timestamp
    direction: str
    entry_price: float
    confirmation_time: pd.Timestamp
    confirmation_level: float
    spike_time: pd.Timestamp
    spike_direction: str
    session_name: str
    units: int
    stop_distance_pips: float
    raw_stop_distance_pips: float
    tp_distance_pips: float
    stop_price: float
    tp_price: float
    margin_required_usd: float
    mfe_pips: float = 0.0
    mae_pips: float = 0.0
    trail_stop: float | None = None
    high_water: float | None = None
    low_water: float | None = None


@dataclass
class V4State:
    current_5_end: pd.Timestamp | None = None
    current_15_end: pd.Timestamp | None = None
    m5_open: float = 0.0
    m5_high: float = 0.0
    m5_low: float = 0.0
    m5_close: float = 0.0
    m15_open: float = 0.0
    m15_high: float = 0.0
    m15_low: float = 0.0
    m15_close: float = 0.0
    tr_queue: deque[float] = field(default_factory=deque)
    tr_sum: float = 0.0
    ema20_prev: float | None = None
    ema50_prev: float | None = None
    last_completed_m15_ema50: float | None = None
    m5_bars: list[Any] = field(default_factory=list)
    pending_spike: Any | None = None
    pending_order: V4PendingOrder | None = None
    position: V4Position | None = None
    trades: list[dict[str, Any]] = field(default_factory=list)
    broad_candidates: int = 0
    family_c_events: int = 0
    armed_orders: int = 0
    fills: int = 0
    blocked_active: int = 0
    blocked_margin: int = 0
    expired: int = 0


@dataclass
class V44ManagedPosition:
    entry_i: int
    entry_time: pd.Timestamp
    entry_price: float
    side: str
    lots_initial: float
    lots_remaining: float
    units: int
    exit_mode: str
    initial_stop_price: float
    stop_price: float
    tp1_price: float
    tp2_price: float
    tp2_pips: float
    tp1_close_fraction: float
    be_offset_pips: float
    trail_buffer_pips: float
    trail_ema_period: int
    strength: str
    strategy_tag: str
    ownership_cell: str | None
    entry_reason: str
    tp1_filled: bool = False
    realized_usd: float = 0.0
    realized_pips_lot_weighted: float = 0.0
    peak_price: float | None = None
    exit_reason: str | None = None


class LocalPhase3Adapter:
    def __init__(self) -> None:
        self.balance = 100000.0
        self.equity = 100000.0
        self.margin_used = 0.0
        self._next_id = 1

    def is_demo(self) -> bool:
        return True

    def get_account_info(self):
        return SimpleNamespace(balance=float(self.balance), equity=float(self.equity), margin_used=float(self.margin_used))

    def place_order(self, *, symbol, side, lots, stop_price, target_price, comment):
        order_id = self._next_id
        self._next_id += 1
        return SimpleNamespace(order_id=order_id, deal_id=order_id, order_retcode=0, fill_price=None)

    def get_position_id_from_order(self, order_id: int) -> int:
        return int(order_id)

    def get_position_id_from_deal(self, deal_id: int) -> int:
        return int(deal_id)


class LocalPhase3Store:
    def __init__(self, getter) -> None:
        self._getter = getter

    def list_open_trades(self, _profile_name: str) -> list[dict[str, Any]]:
        return list(self._getter())


@dataclass
class MarginCallEvent:
    timestamp: pd.Timestamp
    strategy: str
    pnl_usd: float
    nav_before: float
    margin_used_before: float
    margin_level_before: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Combined V7.1 + H1 + V4 bar-by-bar margin engine")
    p.add_argument("--data-path", default=str(ROOT / "research_out/USDJPY_M1_OANDA_extended.csv"))
    p.add_argument("--starting-equity", type=float, default=100000.0)
    p.add_argument("--leverage", type=float, default=33.0)
    p.add_argument("--v4-lots", type=float, default=20.0)
    p.add_argument("--entry-margin-cap-pct", type=float, default=100.0)
    p.add_argument("--spread-mode", choices=["pipeline", "realistic"], default="realistic")
    p.add_argument("--max-bars", type=int, default=None)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def v4_units_for_lots(lots: float) -> int:
    return int(round(float(lots) * 100000.0))


def margin_required(units: int, leverage: float) -> float:
    return float(abs(units)) / max(1.0, float(leverage))


def pip_value_for_units(price: float, units: int) -> float:
    return abs(int(units)) * PIP_SIZE / max(1e-9, float(price))


def calc_pf(series: pd.Series) -> float:
    wins = series[series > 0].sum()
    losses = abs(series[series < 0].sum())
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return float(wins / losses)


def session_name_for_time(ts: pd.Timestamp) -> str:
    hour = int(ts.hour)
    if 0 <= hour < 8:
        return "tokyo"
    if 8 <= hour < 13:
        return "london"
    if 13 <= hour < 21:
        return "ny"
    return "off"


def v4_entry_fill(raw_level: float, direction: str) -> float:
    spread_px = V4_ENTRY_SPREAD_PIPS * PIP_SIZE
    return raw_level + spread_px if direction == "long" else raw_level - spread_px


def v4_stop_distances(spike_low: float, spike_high: float, entry_price: float, direction: str) -> tuple[float | None, float]:
    if direction == "long":
        raw_stop = (entry_price - (float(spike_low) - V4_STOP_BUFFER_PIPS * PIP_SIZE)) / PIP_SIZE
    else:
        raw_stop = ((float(spike_high) + V4_STOP_BUFFER_PIPS * PIP_SIZE) - entry_price) / PIP_SIZE
    if raw_stop > V4_STOP_CLAMP_MAX_PIPS:
        return None, float(raw_stop)
    return max(V4_STOP_CLAMP_MIN_PIPS, float(raw_stop)), float(raw_stop)


def v44_leg_pnl_usd(side: str, entry_price: float, exit_price: float, lots: float) -> float:
    if side == "buy":
        pips = (float(exit_price) - float(entry_price)) / PIP_SIZE
    else:
        pips = (float(entry_price) - float(exit_price)) / PIP_SIZE
    return float(pips * float(lots) * pip_value_usd_per_lot(float(exit_price)))


def v44_total_pnl_usd(position: V44ManagedPosition, bid_c: float, ask_c: float) -> float:
    mark_price = float(bid_c) if position.side == "buy" else float(ask_c)
    return float(position.realized_usd) + v44_leg_pnl_usd(position.side, position.entry_price, mark_price, position.lots_remaining)


def v44_total_pnl_pips(position: V44ManagedPosition, exit_price: float) -> float:
    total_usd = float(position.realized_usd) + v44_leg_pnl_usd(position.side, position.entry_price, float(exit_price), position.lots_remaining)
    denom = max(1e-9, float(position.lots_initial) * pip_value_usd_per_lot(float(exit_price)))
    return float(total_usd / denom)


def v44_close_trade(position: V44ManagedPosition, ts: pd.Timestamp, exit_price: float, exit_reason: str) -> dict[str, Any]:
    pnl_usd = float(position.realized_usd) + v44_leg_pnl_usd(position.side, position.entry_price, float(exit_price), position.lots_remaining)
    pnl_pips = v44_total_pnl_pips(position, float(exit_price))
    return {
        "trade_id": "",
        "strategy": "v44_ny",
        "session": "v44_ny",
        "setup_type": "native",
        "entry_bar_index": position.entry_i,
        "entry_time": str(position.entry_time),
        "entry_price": float(position.entry_price),
        "side": position.side,
        "lots": float(position.lots_initial),
        "exit_bar_index": "",
        "exit_time": str(ts),
        "exit_price": float(exit_price),
        "exit_reason": str(exit_reason),
        "pnl_usd": float(pnl_usd),
        "pnl_pips": float(pnl_pips),
        "strength": position.strength,
        "strategy_tag": position.strategy_tag,
        "ownership_cell": position.ownership_cell or "",
        "entry_reason": position.entry_reason,
    }


def _h1_filter_v44_oracle(raw_oracle: dict[int, dict[str, Any]], filter_context: M5FilterContext) -> tuple[dict[int, dict[str, Any]], dict[str, int]]:
    filtered: dict[int, dict[str, Any]] = {}
    blocked = 0
    for key, row in raw_oracle.items():
        ts = _ts(row["entry_time"])
        ctx = filter_context.row_at_or_before(ts)
        if ctx is not None and pd.notna(ctx["atr14_pips"]) and float(ctx["atr14_pips"]) > H1_ATR14_MAX_PIPS:
            blocked += 1
            continue
        filtered[key] = row
    return filtered, {
        "v44_oracle_original": len(raw_oracle),
        "h1_v44_blocked": blocked,
        "h1_v44_kept": len(filtered),
    }


def mtm_v44(position: dict[str, Any], bid_c: float, ask_c: float) -> float:
    if position["kind"] == "v44_live":
        return v44_total_pnl_usd(position["position"], bid_c, ask_c)
    ep = float(position["entry_price"])
    lot = float(position["lots"])
    if position["side"] == "buy":
        return (bid_c - ep) / PIP_SIZE * lot * pip_value_usd_per_lot(bid_c)
    return (ep - ask_c) / PIP_SIZE * lot * pip_value_usd_per_lot(ask_c)


def mtm_l1(position: dict[str, Any], bid_c: float, ask_c: float) -> float:
    sim: L1Incremental = position["sim"]
    units = int(position["units"])
    direction = "long" if sim.is_long else "short"
    current_px = bid_c if sim.is_long else ask_c
    if not sim.tp1_hit:
        _pips, usd = london_bt.calc_leg_pnl(direction, sim.entry_price, current_px, units)
        return float(usd)
    closed_units = int(position.get("tp1_units_closed", math.floor(units * sim.tp1_close_fraction)))
    closed_units = max(0, min(units, closed_units))
    rem_units = max(0, units - closed_units)
    usd_total = 0.0
    if closed_units > 0 and sim.leg1_exit is not None:
        _p1, usd1 = london_bt.calc_leg_pnl(direction, sim.entry_price, sim.leg1_exit, closed_units)
        usd_total += float(usd1)
    if rem_units > 0:
        _p2, usd2 = london_bt.calc_leg_pnl(direction, sim.entry_price, current_px, rem_units)
        usd_total += float(usd2)
    return float(usd_total)


def mtm_london_native(state: LondonUnifiedDayState | None, bid_c: float, ask_c: float) -> float:
    if state is None:
        return 0.0
    total = 0.0
    for pos in state.open_positions:
        current_px = bid_c if pos.direction == "long" else ask_c
        _pips, usd = london_bt.calc_leg_pnl(pos.direction, pos.entry_price, current_px, pos.remaining_units)
        total += float(pos.pnl_usd_realized) + float(usd)
    return float(total)


def mtm_tokyo(state: TokyoState, bid_c: float, ask_c: float) -> float:
    total = 0.0
    for pos in state.open_positions:
        current_px = bid_c if pos.direction == "long" else ask_c
        _pips, usd = tokyo_calc_leg_usd_pips(pos.direction, pos.entry_price, current_px, pos.units_remaining)
        total += float(pos.realized_usd) + float(usd)
    return float(total)


def mtm_v4(position: V4Position | None, bid_c: float, ask_c: float) -> float:
    if position is None:
        return 0.0
    close = float(bid_c) if position.direction == "long" else float(ask_c)
    if position.direction == "long":
        pnl_pips = (close - position.entry_price) / PIP_SIZE
    else:
        pnl_pips = (position.entry_price - close) / PIP_SIZE
    return float(pnl_pips * pip_value_for_units(close, position.units))


def london_margin_used(ld: LondonUnifiedDayState | None) -> float:
    if ld is None:
        return 0.0
    return float(sum(float(p.margin_required_usd) for p in ld.open_positions))


def l1_v44_margin_used(open_positions: list[dict[str, Any]], leverage: float) -> float:
    total = 0.0
    for p in open_positions:
        if p["kind"] == "l1":
            total += float(p["margin_usd"])
        elif p["kind"] == "v44_oracle":
            total += float(p["lots"]) * 100000.0 / max(1.0, leverage)
        elif p["kind"] == "v44_live":
            total += float(abs(p["position"].units)) / max(1.0, leverage)
    return float(total)


def margin_used_total(
    *,
    ld: LondonUnifiedDayState | None,
    tokyo_state: TokyoState,
    open_positions: list[dict[str, Any]],
    v4_position: V4Position | None,
    leverage: float,
) -> float:
    tokyo_used = float(sum(max(0, int(p.units_remaining)) / max(1.0, leverage) for p in tokyo_state.open_positions))
    v4_used = float(v4_position.margin_required_usd) if v4_position is not None else 0.0
    return london_margin_used(ld) + l1_v44_margin_used(open_positions, leverage) + tokyo_used + v4_used


def hard_margin_available_from_balance(
    *,
    balance: float,
    ld: LondonUnifiedDayState | None,
    tokyo_state: TokyoState,
    open_positions: list[dict[str, Any]],
    v4_position: V4Position | None,
    leverage: float,
    entry_margin_cap_pct: float,
) -> float:
    used = margin_used_total(
        ld=ld,
        tokyo_state=tokyo_state,
        open_positions=open_positions,
        v4_position=v4_position,
        leverage=leverage,
    )
    cap_balance = float(balance) * max(0.0, float(entry_margin_cap_pct)) / 100.0
    return float(cap_balance - used)


def nav_current(
    *,
    balance: float,
    ld: LondonUnifiedDayState | None,
    tokyo_state: TokyoState,
    open_positions: list[dict[str, Any]],
    v4_position: V4Position | None,
    bid_c_tokyo: float,
    ask_c_tokyo: float,
    bid_c_london: float,
    ask_c_london: float,
    bid_c_v4: float,
    ask_c_v4: float,
) -> tuple[float, float]:
    unreal = 0.0
    for p in open_positions:
        if p["kind"] in {"v44_oracle", "v44_live"}:
            unreal += mtm_v44(p, bid_c_london, ask_c_london)
        elif p["kind"] == "l1":
            unreal += mtm_l1(p, bid_c_london, ask_c_london)
    unreal += mtm_london_native(ld, bid_c_london, ask_c_london)
    unreal += mtm_tokyo(tokyo_state, bid_c_tokyo, ask_c_tokyo)
    unreal += mtm_v4(v4_position, bid_c_v4, ask_c_v4)
    return float(balance + unreal), float(unreal)


def margin_level_pct(nav: float, margin_used: float) -> float:
    if margin_used <= 0:
        return float("inf")
    return float(nav / margin_used * 100.0)


def maybe_finalize_v4_bars(
    *,
    i: int,
    ts: pd.Timestamp,
    open_px: float,
    high_px: float,
    low_px: float,
    close_px: float,
    v4: V4State,
    leverage: float,
    v4_units: int,
) -> None:
    next_15 = bucket_end(ts, 15)
    if v4.current_15_end is None:
        v4.current_15_end = next_15
        v4.m15_open = open_px
        v4.m15_high = high_px
        v4.m15_low = low_px
        v4.m15_close = close_px
    elif next_15 != v4.current_15_end:
        close = float(v4.m15_close)
        if v4.ema50_prev is None:
            v4.ema50_prev = close
        else:
            k = 2.0 / 51.0
            v4.ema50_prev = close * k + v4.ema50_prev * (1.0 - k)
        v4.last_completed_m15_ema50 = v4.ema50_prev
        v4.current_15_end = next_15
        v4.m15_open = open_px
        v4.m15_high = high_px
        v4.m15_low = low_px
        v4.m15_close = close_px
    else:
        v4.m15_high = max(v4.m15_high, high_px)
        v4.m15_low = min(v4.m15_low, low_px)
        v4.m15_close = close_px

    next_5 = bucket_end(ts, 5)
    if v4.current_5_end is None:
        v4.current_5_end = next_5
        v4.m5_open = open_px
        v4.m5_high = high_px
        v4.m5_low = low_px
        v4.m5_close = close_px
        return

    if next_5 == v4.current_5_end:
        v4.m5_high = max(v4.m5_high, high_px)
        v4.m5_low = min(v4.m5_low, low_px)
        v4.m5_close = close_px
        return

    bar, new_tr_sum, new_ema20 = make_m5_bar(
        end_time=v4.current_5_end,
        end_idx=i - 1,
        open_=float(v4.m5_open),
        high=float(v4.m5_high),
        low=float(v4.m5_low),
        close=float(v4.m5_close),
        m15_ema50=v4.last_completed_m15_ema50,
        tr_queue=v4.tr_queue,
        tr_sum=v4.tr_sum,
        ema20_prev=v4.ema20_prev,
        prior_bars=v4.m5_bars,
    )
    v4.tr_sum = new_tr_sum
    v4.ema20_prev = new_ema20
    v4.m5_bars.append(bar)
    if bar.broad_candidate:
        v4.broad_candidates += 1

    if v4.pending_spike is not None:
        event = family_c_event(v4.pending_spike, bar)
        if event is not None:
            v4.family_c_events += 1
            assumed_fill = v4_entry_fill(float(event["trigger_level"]), str(event["fade_direction"]))
            stop_distance, raw_stop = v4_stop_distances(
                float(event["spike_low"]),
                float(event["spike_high"]),
                assumed_fill,
                str(event["fade_direction"]),
            )
            if stop_distance is not None:
                if v4.position is None and v4.pending_order is None:
                    v4.pending_order = V4PendingOrder(
                        armed_time=bar.end_time,
                        expiry_time=bar.end_time + pd.Timedelta(minutes=V4_ENTRY_WINDOW_MINUTES),
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
                        session_name=str(event["session_name"]),
                        units=v4_units,
                        margin_required_usd=margin_required(v4_units, leverage),
                    )
                    v4.armed_orders += 1
                else:
                    v4.blocked_active += 1
        v4.pending_spike = None

    v4.pending_spike = bar if bar.broad_candidate else None
    v4.current_5_end = next_5
    v4.m5_open = open_px
    v4.m5_high = high_px
    v4.m5_low = low_px
    v4.m5_close = close_px


def close_v4_margin_call(position: V4Position, ts: pd.Timestamp, bid_px: float, ask_px: float) -> dict[str, Any]:
    close_px = float(bid_px) if position.direction == "long" else float(ask_px)
    if position.direction == "long":
        pnl_pips = (close_px - position.entry_price) / PIP_SIZE
    else:
        pnl_pips = (position.entry_price - close_px) / PIP_SIZE
    pnl_usd = pnl_pips * pip_value_for_units(close_px, position.units)
    return {
        "strategy": "spike_fade_v4",
        "session": position.session_name,
        "setup_type": "spike_fade_v4",
        "entry_time": str(position.entry_time),
        "exit_time": str(ts),
        "entry_price": position.entry_price,
        "exit_price": close_px,
        "side": "buy" if position.direction == "long" else "sell",
        "lots": position.units / 100000.0,
        "pnl_pips": float(pnl_pips),
        "pnl_usd": float(pnl_usd),
        "exit_reason": "margin_call",
        "trade_id": "",
    }


def maybe_close_v4_position_on_bar(position: V4Position, ts: pd.Timestamp, open_px: float, high_px: float, low_px: float, close_bid: float, close_ask: float) -> dict[str, Any] | None:
    high = float(high_px)
    low = float(low_px)
    close = float(close_bid) if position.direction == "long" else float(close_ask)

    if position.direction == "long":
        position.mfe_pips = max(position.mfe_pips, (high - position.entry_price) / PIP_SIZE)
        position.mae_pips = max(position.mae_pips, (position.entry_price - low) / PIP_SIZE)
        exit_price = None
        exit_reason = None
        if low <= position.stop_price:
            exit_price = float(position.stop_price)
            exit_reason = "stop_loss"
        elif high >= position.tp_price:
            exit_price = float(position.tp_price)
            exit_reason = "take_profit"
        if exit_reason is None and position.mfe_pips >= V4_TRAIL_TRIGGER_PIPS:
            position.high_water = max(position.high_water or position.entry_price, high)
            candidate = float(position.high_water) - V4_TRAIL_DISTANCE_PIPS * PIP_SIZE
            position.trail_stop = candidate if position.trail_stop is None else max(position.trail_stop, candidate)
            if low <= float(position.trail_stop):
                exit_price = float(position.trail_stop)
                exit_reason = "trailing_stop"
        current_pnl_pips = (close - position.entry_price) / PIP_SIZE
    else:
        position.mfe_pips = max(position.mfe_pips, (position.entry_price - low) / PIP_SIZE)
        position.mae_pips = max(position.mae_pips, (high - position.entry_price) / PIP_SIZE)
        exit_price = None
        exit_reason = None
        if high >= position.stop_price:
            exit_price = float(position.stop_price)
            exit_reason = "stop_loss"
        elif low <= position.tp_price:
            exit_price = float(position.tp_price)
            exit_reason = "take_profit"
        if exit_reason is None and position.mfe_pips >= V4_TRAIL_TRIGGER_PIPS:
            position.low_water = min(position.low_water or position.entry_price, low)
            candidate = float(position.low_water) + V4_TRAIL_DISTANCE_PIPS * PIP_SIZE
            position.trail_stop = candidate if position.trail_stop is None else min(position.trail_stop, candidate)
            if high >= float(position.trail_stop):
                exit_price = float(position.trail_stop)
                exit_reason = "trailing_stop"
        current_pnl_pips = (position.entry_price - close) / PIP_SIZE

    held_minutes = int((ts - position.entry_time) / pd.Timedelta(minutes=1))
    if exit_reason is None and held_minutes >= V4_PROVE_IT_MINUTES and current_pnl_pips < V4_PROVE_IT_PIPS:
        exit_price = close
        exit_reason = "prove_it_fast"
    if exit_reason is None and held_minutes >= V4_TIME_STOP_MINUTES:
        exit_price = close
        exit_reason = "time_stop"
    if exit_reason is None:
        return None

    pnl_pips = (float(exit_price) - position.entry_price) / PIP_SIZE if position.direction == "long" else (position.entry_price - float(exit_price)) / PIP_SIZE
    pnl_usd = pnl_pips * pip_value_for_units(float(exit_price), position.units)
    return {
        "trade_id": "",
        "strategy": "spike_fade_v4",
        "session": position.session_name,
        "setup_type": "spike_fade_v4",
        "entry_time": str(position.entry_time),
        "exit_time": str(ts),
        "entry_price": position.entry_price,
        "exit_price": float(exit_price),
        "side": "buy" if position.direction == "long" else "sell",
        "lots": position.units / 100000.0,
        "pnl_pips": float(pnl_pips),
        "pnl_usd": float(pnl_usd),
        "exit_reason": exit_reason,
        "spike_time": str(position.spike_time),
        "spike_direction": position.spike_direction,
    }


def maybe_advance_v44_position_on_bar(position: V44ManagedPosition, ts: pd.Timestamp, bid_high: float, ask_high: float, bid_low: float, ask_low: float, bid_close: float, ask_close: float, trail_mark: float | None) -> dict[str, Any] | None:
    if position.side == "buy":
        stop_hit = float(bid_low) <= float(position.stop_price)
    else:
        stop_hit = float(ask_high) >= float(position.stop_price)

    if not position.tp1_filled:
        tp1_hit = (float(bid_high) >= float(position.tp1_price)) if position.side == "buy" else (float(ask_low) <= float(position.tp1_price))
        if stop_hit:
            position.exit_reason = "sl"
            exit_px = float(position.stop_price)
            return v44_close_trade(position, ts, exit_px, "sl")
        if tp1_hit:
            if position.exit_mode == "news_full_tp1":
                position.exit_reason = "tp1_full"
                exit_px = float(position.tp1_price)
                return v44_close_trade(position, ts, exit_px, "tp1_full")
            close_lots = max(0.0, min(float(position.lots_remaining), float(position.lots_initial) * float(position.tp1_close_fraction)))
            if close_lots > 0:
                position.realized_usd += v44_leg_pnl_usd(position.side, position.entry_price, float(position.tp1_price), close_lots)
                position.lots_remaining = max(0.0, float(position.lots_remaining) - close_lots)
            position.tp1_filled = True
            position.stop_price = float(position.entry_price) + (float(position.be_offset_pips) * PIP_SIZE if position.side == "buy" else -float(position.be_offset_pips) * PIP_SIZE)
        return None

    if trail_mark is not None:
        if position.side == "buy":
            new_stop = float(trail_mark) - float(position.trail_buffer_pips) * PIP_SIZE
            if new_stop > float(position.stop_price):
                position.stop_price = float(new_stop)
        else:
            new_stop = float(trail_mark) + float(position.trail_buffer_pips) * PIP_SIZE
            if new_stop < float(position.stop_price):
                position.stop_price = float(new_stop)

    tp1_pips = abs(float(position.tp1_price) - float(position.entry_price)) / PIP_SIZE
    tp2_total_pips = float(tp1_pips) + float(position.tp2_pips)
    tp2_price = float(position.entry_price) + (tp2_total_pips * PIP_SIZE if position.side == "buy" else -tp2_total_pips * PIP_SIZE)
    tp2_hit = (float(bid_high) >= tp2_price) if position.side == "buy" else (float(ask_low) <= tp2_price)

    if stop_hit:
        position.exit_reason = "tp1_then_trail"
        return v44_close_trade(position, ts, float(position.stop_price), "tp1_then_trail")
    if tp2_hit:
        position.exit_reason = "tp1_then_tp2"
        return v44_close_trade(position, ts, float(tp2_price), "tp1_then_tp2")
    return None


def collect_margin_call_candidates(
    *,
    ts: pd.Timestamp,
    ld: LondonUnifiedDayState | None,
    tokyo_state: TokyoState,
    open_positions: list[dict[str, Any]],
    v4: V4State,
    bid_c_tokyo: float,
    ask_c_tokyo: float,
    bid_c_london: float,
    ask_c_london: float,
    bid_c_v4: float,
    ask_c_v4: float,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    for idx, p in enumerate(open_positions):
        if p["kind"] in {"v44_oracle", "v44_live"}:
            pnl = mtm_v44(p, bid_c_london, ask_c_london)
            candidates.append({"bucket": "v44", "idx": idx, "pnl_usd": pnl, "strategy": "v44_ny"})
        elif p["kind"] == "l1":
            pnl = mtm_l1(p, bid_c_london, ask_c_london)
            candidates.append({"bucket": "l1", "idx": idx, "pnl_usd": pnl, "strategy": "london_setup_d_l1"})

    if ld is not None:
        for idx, p in enumerate(ld.open_positions):
            current_px = bid_c_london if p.direction == "long" else ask_c_london
            _pips, usd = london_bt.calc_leg_pnl(p.direction, p.entry_price, current_px, p.remaining_units)
            pnl = float(p.pnl_usd_realized) + float(usd)
            candidates.append({"bucket": "london", "idx": idx, "pnl_usd": pnl, "strategy": f"london_setup_{p.setup_type.lower()}"})

    for idx, p in enumerate(tokyo_state.open_positions):
        current_px = bid_c_tokyo if p.direction == "long" else ask_c_tokyo
        _pips, usd = tokyo_calc_leg_usd_pips(p.direction, p.entry_price, current_px, p.units_remaining)
        pnl = float(p.realized_usd) + float(usd)
        candidates.append({"bucket": "tokyo", "idx": idx, "pnl_usd": pnl, "strategy": "tokyo_v14"})

    if v4.position is not None:
        candidates.append({"bucket": "v4", "idx": 0, "pnl_usd": mtm_v4(v4.position, bid_c_v4, ask_c_v4), "strategy": "spike_fade_v4"})

    candidates.sort(key=lambda x: float(x["pnl_usd"]))
    return candidates


def execute_combined(params: CombinedParams) -> dict[str, Any]:
    dataset_path = str(params.data_path)
    spread_pipeline = params.spread_mode == "pipeline"
    v4_units = v4_units_for_lots(params.v4_lots)

    t0 = time.time()
    if not params.quiet:
        print(
            f"Loading {dataset_path} | start={params.starting_equity:.0f} | leverage={params.leverage:.1f} | "
            f"V4={params.v4_lots:.2f} lots | spread={params.spread_mode}",
            flush=True,
        )

    m1_raw = tokyo_bt.load_m1(dataset_path)
    if params.max_bars is not None:
        m1_raw = m1_raw.iloc[: int(params.max_bars)].copy()
    if not params.quiet:
        print(f"  M1 rows={len(m1_raw)}", flush=True)

    tokyo_cfg = init_tokyo_config(TOKYO_CFG)
    with LONDON_CFG_PATH.open() as f:
        london_cfg = json.load(f)
    with V44_CFG_PATH.open() as f:
        v44_cfg = json.load(f)

    # Force a unified leverage assumption across the account.
    tokyo_cfg["position_sizing"]["max_concurrent_positions"] = int(tokyo_cfg["position_sizing"].get("max_concurrent_positions", 3))
    london_cfg["account"]["leverage"] = float(params.leverage)
    tokyo_unified.tokyo_margin_used_open_positions = (
        lambda positions, leverage=params.leverage: float(sum(max(0, int(p.units_remaining)) / max(1.0, leverage) for p in positions))
    )

    m5 = tokyo_bt.resample_ohlc_continuous(m1_raw, "5min")
    m15 = tokyo_bt.resample_ohlc_continuous(m1_raw, "15min")
    h1 = tokyo_bt.resample_ohlc_continuous(m1_raw, "1h")
    h4 = tokyo_bt.resample_ohlc_continuous(m1_raw, "4h")
    df, m5, m15 = compute_tokyo_indicators(m1_raw, m5, m15, tokyo_cfg)
    m5 = m5.copy()
    h1 = h1.copy()
    h4 = h4.copy()
    m5["ema_fast_v44_9"] = m5["close"].astype(float).ewm(span=int(v44_cfg.get("v5_m5_ema_fast", 9)), adjust=False).mean()
    m5["ema_fast_v44_21"] = m5["close"].astype(float).ewm(span=int(v44_cfg.get("v5_strong_trail_ema", 21)), adjust=False).mean()
    df["time_utc"] = df["time"].dt.tz_convert("UTC")
    df["time_jst"] = df["time"].dt.tz_convert(tokyo_bt.TOKYO_TZ)
    df["session_day_jst"] = df["time_jst"].dt.date.astype(str)
    df["utc_day_name"] = df["time_utc"].dt.day_name()
    df["hour_utc"] = df["time_utc"].dt.hour
    df["minute_utc"] = df["time_utc"].dt.minute
    df["minute_of_day_utc"] = df["hour_utc"] * 60 + df["minute_utc"]

    sf = tokyo_cfg.get("session_filter", {})
    allowed_days = set(sf.get("allowed_trading_days", []))
    session_start_utc = str(sf.get("session_start_utc", "15:00"))
    session_end_utc = str(sf.get("session_end_utc", "00:00"))

    def hhmm_to_minutes(s: str) -> int:
        hh, mm = s.strip().split(":")
        return int(hh) * 60 + int(mm)

    start_min = hhmm_to_minutes(session_start_utc)
    end_min = hhmm_to_minutes(session_end_utc)
    if start_min < end_min:
        df["in_tokyo_session"] = (df["minute_of_day_utc"] >= start_min) & (df["minute_of_day_utc"] < end_min)
    else:
        df["in_tokyo_session"] = (df["minute_of_day_utc"] >= start_min) | (df["minute_of_day_utc"] < end_min)
    df["allowed_trading_day"] = df["utc_day_name"].isin(allowed_days)

    day_rng = (
        df.assign(utc_date=df["time_utc"].dt.date)
        .groupby("utc_date")
        .agg(day_high=("high", "max"), day_low=("low", "min"))
        .reset_index()
    )
    day_rng["range_pips"] = (day_rng["day_high"] - day_rng["day_low"]) / PIP_SIZE
    day_rng["prior_day_range_pips"] = day_rng["range_pips"].shift(1)
    df["prior_day_range_pips"] = df["time_utc"].dt.date.map(dict(zip(day_rng["utc_date"], day_rng["prior_day_range_pips"])))
    df["news_blocked"] = False

    bar_frame = auth_loop._load_bar_frame(dataset_path)
    if "ownership_cell" not in bar_frame.columns:
        bar_frame = bar_frame.copy()
        bar_frame["ownership_cell"] = [
            cell_key_from_floats(str(rg), float(er), float(de))
            for rg, er, de in zip(
                bar_frame["regime_hysteresis"],
                bar_frame["sf_er"],
                bar_frame["delta_er"],
                strict=False,
            )
        ]
    bf_times = pd.to_datetime(bar_frame["time"], utc=True).values.astype("datetime64[ns]")
    bf_cells = bar_frame["ownership_cell"].astype(str).tolist()
    bf_reg = bar_frame["regime_hysteresis"].astype(str).tolist()
    bf_er = bar_frame["sf_er"].astype(float).tolist()
    bf_der = bar_frame["delta_er"].astype(float).tolist()

    def lookup_cell(ts: pd.Timestamp) -> tuple[str, str, float, float]:
        key = np.datetime64(pd.Timestamp(ts).asm8.astype("datetime64[ns]"))
        idx = int(np.searchsorted(bf_times, key, side="right") - 1)
        if idx < 0:
            return "ambiguous/er_mid/der_pos", "ambiguous", 0.5, 0.0
        return bf_cells[idx], bf_reg[idx], bf_er[idx], bf_der[idx]

    v44_risk_pct = float(v44_cfg.get("v5_risk_per_trade_pct", 0.5)) / 100.0
    phase3_state: dict[str, Any] = {}
    local_adapter = LocalPhase3Adapter()
    phase3_profile = SimpleNamespace(
        symbol="USDJPY",
        pip_size=PIP_SIZE,
        broker_type="sim",
        active_preset_name=PHASE3_DEFENDED_PRESET_ID,
        profile_name="combined_barbybar",
    )
    phase3_policy = SimpleNamespace(id=PHASE3_DEFENDED_PRESET_ID)
    overlay_state = build_phase3_overlay_state({"v44_ny": v44_cfg})

    vk_ctx = build_variant_k_baseline_context({"M1": m1_raw, "M5": m5})
    m5_times_ns = _m5_times_sorted_ns(m5)
    m1_times_ns = pd.to_datetime(df["time"], utc=True).values.astype("datetime64[ns]")
    m5_times_all_ns = pd.to_datetime(m5["time"], utc=True).values.astype("datetime64[ns]")
    m15_times_all_ns = pd.to_datetime(m15["time"], utc=True).values.astype("datetime64[ns]")
    h1_times_all_ns = pd.to_datetime(h1["time"], utc=True).values.astype("datetime64[ns]")
    h4_times_all_ns = pd.to_datetime(h4["time"], utc=True).values.astype("datetime64[ns]")

    time_ns_all = pd.to_datetime(df["time"], utc=True).values.astype("datetime64[ns]")
    high_all = np.asarray(df["high"], dtype=np.float64)
    low_all = np.asarray(df["low"], dtype=np.float64)
    day_ns_bounds: dict[str, np.datetime64] = {}

    def _set_day_bounds(i0: int) -> None:
        day_start = pd.Timestamp(df.iloc[i0]["time"]).normalize()
        london_h = london_bt.uk_london_open_utc(day_start)
        london_open = day_start + pd.Timedelta(hours=london_h)
        lor_end = london_open + pd.Timedelta(minutes=15)

        def _utc_ns(ts: pd.Timestamp) -> np.datetime64:
            p = pd.Timestamp(ts)
            if p.tzinfo is None:
                p = p.tz_localize("UTC")
            else:
                p = p.tz_convert("UTC")
            return np.datetime64(p.asm8.astype("datetime64[ns]"))

        day_ns_bounds["asian_lo"] = _utc_ns(day_start)
        day_ns_bounds["asian_hi"] = _utc_ns(london_open)
        day_ns_bounds["lor_hi"] = _utc_ns(lor_end)

    def variant_f_allows_v44(entry_ts: pd.Timestamp) -> bool:
        tr = base_runner.TradeRow(
            strategy="v44_ny",
            entry_time=_ts(entry_ts),
            exit_time=_ts(entry_ts),
            entry_session="v44_ny",
            side="buy",
            pips=0.0,
            usd=0.0,
            exit_reason="x",
            standalone_entry_equity=params.starting_equity,
            raw={},
            size_scale=1.0,
        )
        r = v44_router._filter_v44_trade(
            tr,
            vk_ctx.classified_basic,
            vk_ctx.m5_basic,
            block_breakout=True,
            block_post_breakout=True,
            block_ambiguous_non_momentum=True,
            momentum_only=False,
            exhaustion_gate=False,
            soft_exhaustion=False,
            er_threshold=0.35,
            decay_threshold=0.40,
        )
        return not r.blocked

    def _slice_tf(frame: pd.DataFrame, times_ns: np.ndarray, cutoff_ts: pd.Timestamp, tail: int) -> pd.DataFrame:
        cutoff_ns = np.datetime64(pd.Timestamp(cutoff_ts).asm8.astype("datetime64[ns]"))
        idx = int(np.searchsorted(times_ns, cutoff_ns, side="right"))
        if idx <= 0:
            return frame.iloc[:0].copy()
        start = max(0, idx - int(tail))
        return frame.iloc[start:idx].copy()

    def _v44_data_by_tf(ts_now: pd.Timestamp) -> dict[str, pd.DataFrame]:
        m1_cut = ts_now
        m5_cut = ts_now.floor("5min")
        m15_cut = ts_now.floor("15min")
        h1_cut = ts_now.floor("1h")
        h4_cut = ts_now.floor("4h")
        return {
            "M1": _slice_tf(df, m1_times_ns, m1_cut, 720),
            "M5": _slice_tf(m5, m5_times_all_ns, m5_cut, 500),
            "M15": _slice_tf(m15, m15_times_all_ns, m15_cut, 250),
            "H1": _slice_tf(h1, h1_times_all_ns, h1_cut, 220),
            "H4": _slice_tf(h4, h4_times_all_ns, h4_cut, 120),
        }

    asian_min = float(london_cfg["levels"]["asian_range_min_pips"])
    asian_max = float(london_cfg["levels"]["asian_range_max_pips"])
    lor_min = float(london_cfg["levels"]["lor_range_min_pips"])
    lor_max = float(london_cfg["levels"]["lor_range_max_pips"])
    l1_tp1_r = 3.25
    l1_tp2_r = 2.0
    l1_be_off = 1.0
    l1_tp1_close_fraction = float(london_cfg["setups"]["D"]["tp1_close_fraction"])

    balance = float(params.starting_equity)
    peak_nav = balance
    trade_id_box = [0]
    combined_trades: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    blocked = Counter()
    open_positions: list[dict[str, Any]] = []
    margin_calls: list[MarginCallEvent] = []
    local_phase3_store = LocalPhase3Store(
        lambda: [
            {
                "entry_type": str(p["position"].strategy_tag),
                "side": str(p["position"].side),
            }
            for p in open_positions
            if p["kind"] == "v44_live"
        ]
    )

    ld: Optional[LondonUnifiedDayState] = None
    tokyo_state = TokyoState()
    v4_state = V4State()
    cur_day: Optional[pd.Timestamp] = None
    day_start_idx = 0

    def causal_asian_lor(i0: int, i: int, t: pd.Timestamp) -> tuple[float, float, float, bool, float, float, float, bool]:
        asian_lo = day_ns_bounds["asian_lo"]
        asian_hi = day_ns_bounds["asian_hi"]
        lor_lo = asian_hi
        lor_hi = day_ns_bounds["lor_hi"]
        ts_ns = time_ns_all[i]
        tseg = time_ns_all[i0 : i + 1]
        hseg = high_all[i0 : i + 1]
        lseg = low_all[i0 : i + 1]
        ma = (tseg >= asian_lo) & (tseg < asian_hi)
        if not ma.any():
            return (0.0, 0.0, 0.0, False, 0.0, 0.0, 0.0, False)
        ah = float(np.max(hseg[ma]))
        al = float(np.min(lseg[ma]))
        ar = (ah - al) / PIP_SIZE
        av = asian_min <= ar <= asian_max
        ml = (tseg >= lor_lo) & (tseg < lor_hi) & (tseg <= ts_ns)
        if not ml.any():
            return (ah, al, ar, av, 0.0, 0.0, 0.0, False)
        lh = float(np.max(hseg[ml]))
        ll = float(np.min(lseg[ml]))
        lr = (lh - ll) / PIP_SIZE
        lv = lor_min <= lr <= lor_max
        return (ah, al, ar, av, lh, ll, lr, lv)

    def margin_metrics(row: pd.Series, i: int) -> tuple[float, float, float, float, float, float, float]:
        sp_tokyo = tokyo_bt.compute_spread_pips(
            i,
            pd.Timestamp(row["time"]),
            str(tokyo_cfg["execution_model"].get("spread_mode", "fixed")),
            float(tokyo_cfg["execution_model"].get("spread_pips", 1.5)),
            float(tokyo_cfg["execution_model"].get("spread_min_pips", 1.0)),
            float(tokyo_cfg["execution_model"].get("spread_max_pips", 3.0)),
        )
        bid_c_tokyo, ask_c_tokyo = tokyo_bt.get_bid_ask(float(row["close"]), sp_tokyo)
        sp_london = london_bt.compute_spread_pips(i, pd.Timestamp(row["time"]), london_cfg)
        bid_c_london, ask_c_london = london_bt.to_bid_ask(float(row["close"]), sp_london)
        bid_c_v4, ask_c_v4 = bid_c_london, ask_c_london
        nav, unreal = nav_current(
            balance=balance,
            ld=ld,
            tokyo_state=tokyo_state,
            open_positions=open_positions,
            v4_position=v4_state.position,
            bid_c_tokyo=bid_c_tokyo,
            ask_c_tokyo=ask_c_tokyo,
            bid_c_london=bid_c_london,
            ask_c_london=ask_c_london,
            bid_c_v4=bid_c_v4,
            ask_c_v4=ask_c_v4,
        )
        used = margin_used_total(
            ld=ld,
            tokyo_state=tokyo_state,
            open_positions=open_positions,
            v4_position=v4_state.position,
            leverage=params.leverage,
        )
        level = margin_level_pct(nav, used)
        avail = nav - used
        return nav, unreal, used, avail, level, bid_c_tokyo, ask_c_tokyo

    def margin_metrics_full(row: pd.Series, i: int) -> tuple[float, float, float, float, float, float, float, float, float]:
        nav, unreal, used, avail, level, bid_c_tokyo, ask_c_tokyo = margin_metrics(row, i)
        sp_london = london_bt.compute_spread_pips(i, pd.Timestamp(row["time"]), london_cfg)
        bid_c_london, ask_c_london = london_bt.to_bid_ask(float(row["close"]), sp_london)
        return nav, unreal, used, avail, level, bid_c_tokyo, ask_c_tokyo, bid_c_london, ask_c_london

    warmup = 200
    n = len(df)
    max_open_peak = 0

    loop_t0 = time.time()
    for i in range(warmup, n):
        row = df.iloc[i]
        ts = pd.Timestamp(row["time"])
        dnorm = ts.normalize()
        if cur_day is None or dnorm != cur_day:
            cur_day = dnorm
            day_start_idx = i
            ld = init_london_day_state(dnorm, london_cfg)
            _set_day_bounds(day_start_idx)

        maybe_finalize_v4_bars(
            i=i,
            ts=ts,
            open_px=float(row["open"]),
            high_px=float(row["high"]),
            low_px=float(row["low"]),
            close_px=float(row["close"]),
            v4=v4_state,
            leverage=params.leverage,
            v4_units=v4_units,
        )

        cell_s, reg, er_v, der_v = lookup_cell(ts)

        def tokyo_exec_gate(_pe: dict[str, Any]) -> tuple[bool, str]:
            ok, reason = admission_checks(
                strategy="tokyo_v14",
                entry_ts=ts,
                cell_str=cell_s,
                regime=reg,
                delta_er=der_v,
                setup_d=False,
                weekday_name=str(ts.day_name()),
            )
            if not ok:
                blocked[f"tokyo_{reason}"] += 1
            return ok, reason

        tokyo_margin_other = london_margin_used(ld) + l1_v44_margin_used(open_positions, params.leverage)
        if v4_state.position is not None:
            tokyo_margin_other += float(v4_state.position.margin_required_usd)

        tokyo_res = advance_tokyo_bar(
            bar_idx=i,
            m1_row=row,
            m5_indicators=m5,
            m15_indicators=m15,
            pivots={},
            cfg=tokyo_cfg,
            state=tokyo_state,
            equity=balance,
            exec_gate=tokyo_exec_gate,
            margin_avail=None,
            spread_pips=_get_tokyo_spread(ts, tokyo_cfg),
            pip_size=PIP_SIZE,
            pip_value=pip_value_usd_per_lot(float(row["close"])),
            m1_full=df,
            peak_equity=peak_nav,
            margin_used_other=tokyo_margin_other,
            trade_id_counter=trade_id_box,
            ownership_cell=cell_s,
            regime_label=reg,
        )
        balance += tokyo_res.equity_delta
        for tr in tokyo_res.exits:
            tr = dict(tr)
            tr["strategy"] = "tokyo_v14"
            combined_trades.append(tr)

        sp_tokyo = tokyo_bt.compute_spread_pips(
            i, ts, str(tokyo_cfg["execution_model"].get("spread_mode", "fixed")),
            float(tokyo_cfg["execution_model"].get("spread_pips", 1.5)),
            float(tokyo_cfg["execution_model"].get("spread_min_pips", 1.0)),
            float(tokyo_cfg["execution_model"].get("spread_max_pips", 3.0)),
        )
        bid_c_tokyo, ask_c_tokyo = tokyo_bt.get_bid_ask(float(row["close"]), sp_tokyo)
        bid_h_tokyo, ask_h_tokyo = tokyo_bt.get_bid_ask(float(row["high"]), sp_tokyo)
        bid_l_tokyo, ask_l_tokyo = tokyo_bt.get_bid_ask(float(row["low"]), sp_tokyo)
        bid_o_tokyo, ask_o_tokyo = tokyo_bt.get_bid_ask(float(row["open"]), sp_tokyo)
        sp_london_bar = london_bt.compute_spread_pips(i, ts, london_cfg)
        bid_c_london_bar, ask_c_london_bar = london_bt.to_bid_ask(float(row["close"]), sp_london_bar)
        bid_h_london_bar, ask_h_london_bar = london_bt.to_bid_ask(float(row["high"]), sp_london_bar)
        bid_l_london_bar, ask_l_london_bar = london_bt.to_bid_ask(float(row["low"]), sp_london_bar)

        assert_m5_no_lookahead_fast(m5_times_ns, ts, "m5")

        still: list[dict[str, Any]] = []
        for p in open_positions:
            if p["kind"] == "v44_live":
                pos: V44ManagedPosition = p["position"]
                trail_mark = None
                if pos.tp1_filled and len(m5) > 0:
                    m5_cut = ts.floor("5min")
                    p5_idx = int(np.searchsorted(m5_times_all_ns, np.datetime64(pd.Timestamp(m5_cut).asm8.astype("datetime64[ns]")), side="right") - 1)
                    if p5_idx >= 0:
                        trail_col = "ema_fast_v44_21" if int(pos.trail_ema_period) >= 20 else "ema_fast_v44_9"
                        trail_mark = float(m5.iloc[p5_idx][trail_col])
                v44_trade = maybe_advance_v44_position_on_bar(
                    pos,
                    ts,
                    bid_h_london_bar,
                    ask_h_london_bar,
                    bid_l_london_bar,
                    ask_l_london_bar,
                    bid_c_london_bar,
                    ask_c_london_bar,
                    trail_mark,
                )
                if v44_trade is not None:
                    trade_id_box[0] += 1
                    v44_trade["trade_id"] = trade_id_box[0]
                    v44_trade["exit_bar_index"] = i
                    balance += float(v44_trade["pnl_usd"])
                    combined_trades.append(v44_trade)
                else:
                    still.append(p)
                continue

            if p["kind"] == "l1":
                sim: L1Incremental = p["sim"]
                hs_bar: Optional[float] = None
                if p.get("l1_realistic_spread"):
                    spb = london_bt.compute_spread_pips(i, ts, london_cfg)
                    hs_bar = spb * PIP_SIZE / 2.0
                if not sim.on_bar(
                    ts,
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    half_spread=hs_bar,
                ):
                    still.append(p)
                    continue
                pips = sim.pnl_pips()
                units = int(p["units"])
                exit_px = float(row["close"])
                _, usd = london_bt.calc_leg_pnl("long" if sim.is_long else "short", sim.entry_price, exit_px, units)
                balance += usd
                trade_id_box[0] += 1
                meta = dict(p.get("meta", {}))
                meta.update(
                    {
                        "trade_id": trade_id_box[0],
                        "strategy": "london_setup_d_l1",
                        "pnl_usd": usd,
                        "pnl_pips": pips,
                        "exit_time": str(sim.exit_time),
                        "exit_bar_index": i,
                        "exit_price": exit_px,
                        "exit_reason": sim.exit_reason,
                    }
                )
                combined_trades.append(meta)
                continue
            still.append(p)
        open_positions = still

        v4_processed_this_bar = False

        nav, unreal, margin_used, margin_available, mlevel, bid_c_tokyo, ask_c_tokyo, bid_c_london, ask_c_london = margin_metrics_full(row, i)
        hard_margin_available = hard_margin_available_from_balance(
            balance=balance,
            ld=ld,
            tokyo_state=tokyo_state,
            open_positions=open_positions,
            v4_position=v4_state.position,
            leverage=params.leverage,
            entry_margin_cap_pct=params.entry_margin_cap_pct,
        )

        if ld is not None:
            ah, al, ar, av, lh, ll, lr, lv = causal_asian_lor(day_start_idx, i, ts)
            i_day = i - day_start_idx
            nxt_ts = pd.Timestamp(df.iloc[i + 1]["time"]) if i + 1 < n else None

            def london_exec_gate(pe: dict[str, Any], t: pd.Timestamp) -> tuple[bool, str]:
                setup = str(pe["setup_type"])
                cell_s2, reg2, _er2, der_v2 = lookup_cell(t)
                ok, reason = admission_checks(
                    strategy="london_v2",
                    entry_ts=t,
                    cell_str=cell_s2,
                    regime=reg2,
                    delta_er=der_v2,
                    setup_d=(setup == "D"),
                    weekday_name=str(t.day_name()),
                )
                if not ok:
                    blocked[f"london_{reason}"] += 1
                return ok, reason

            balance, ldn_closed, l1_payloads, _ = advance_london_unified_bar(
                ld,
                row=row,
                ts=ts,
                nxt_ts=nxt_ts,
                i_day=i_day,
                i_global=i,
                asian_high=ah,
                asian_low=al,
                asian_range_pips=ar,
                asian_valid=av,
                lor_high=lh,
                lor_low=ll,
                lor_range_pips=lr,
                lor_valid=lv,
                equity=balance,
                margin_avail_unified=hard_margin_available,
                extra_open_positions=len(open_positions) + len(tokyo_state.open_positions) + (1 if v4_state.position is not None else 0),
                trade_id_counter=trade_id_box,
                spread_mode_pipeline=spread_pipeline,
                l1_tp1_r=l1_tp1_r,
                l1_tp2_r=l1_tp2_r,
                l1_be_offset=l1_be_off,
                l1_tp1_close_fraction=l1_tp1_close_fraction,
                exec_gate=london_exec_gate,
            )
            for tr in ldn_closed:
                tr_out = {k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in tr.items()}
                tr_out["strategy"] = f"london_setup_{str(tr_out.get('setup_type', '')).lower()}"
                combined_trades.append(tr_out)
            for pay in l1_payloads:
                sim = build_l1_incremental_from_entry(
                    direction=str(pay["direction"]),
                    entry_bar=pay["entry_bar_row"],
                    lor_high=float(pay["lor_high"]),
                    lor_low=float(pay["lor_low"]),
                    ny_open=pay["ny_open"],
                    spread_pips=float(pay["spread_pips_exec"]),
                    tp1_r=float(l1_tp1_r),
                    tp2_r=float(l1_tp2_r),
                    be_offset=float(l1_be_off),
                    tp1_close_fraction=float(l1_tp1_close_fraction),
                )
                if sim is None:
                    continue
                req_m = float(pay["margin_required_usd"])
                hard_avail_l1 = hard_margin_available_from_balance(
                    balance=balance,
                    ld=ld,
                    tokyo_state=tokyo_state,
                    open_positions=open_positions,
                    v4_position=v4_state.position,
                    leverage=params.leverage,
                    entry_margin_cap_pct=params.entry_margin_cap_pct,
                )
                if req_m > hard_avail_l1:
                    blocked["margin_l1"] += 1
                    continue
                cell_s2, reg2, er_v2, der_v2 = lookup_cell(ts)
                units = int(pay["units"])
                tp1_units_closed = max(0, min(units, int(math.floor(units * l1_tp1_close_fraction))))
                open_positions.append(
                    {
                        "kind": "l1",
                        "sim": sim,
                        "units": units,
                        "tp1_units_closed": tp1_units_closed,
                        "margin_usd": req_m,
                        "entry_i": i,
                        "l1_realistic_spread": params.spread_mode == "realistic",
                        "meta": {
                            "trade_id": int(pay["trade_id"]),
                            "session": "london_v2",
                            "setup_type": "D",
                            "entry_bar_index": i,
                            "entry_time": str(ts),
                            "entry_price": sim.entry_price,
                            "side": "buy" if sim.is_long else "sell",
                            "lots": units / 100000.0,
                            "sl_price": sim.sl_price,
                            "sl_pips": sim.risk_pips,
                            "tp1_price": sim.tp1_price,
                            "tp2_price": sim.tp2_price,
                            "ownership_cell": cell_s2,
                            "regime_label": reg2,
                            "er": er_v2,
                            "delta_er": der_v2,
                        },
                    }
                )

        session_key_ny = f"session_ny_{ts.date()}"
        ny_sdat = phase3_state.get(session_key_ny, {})
        should_eval_v44 = 11 <= int(ts.hour) <= 21 or bool((ny_sdat or {}).get("queued_pending"))
        if should_eval_v44:
            ok, reason = admission_checks(
                strategy="v44_ny",
                entry_ts=ts,
                cell_str=cell_s,
                regime=reg,
                delta_er=der_v,
                setup_d=False,
                weekday_name=str(ts.day_name()),
                variant_f_allows=lambda: variant_f_allows_v44(ts),
            )
            if not ok:
                blocked[reason] += 1
            else:
                trial_state = copy.deepcopy(phase3_state)
                total_open_count = (
                    len(open_positions)
                    + (len(ld.open_positions) if ld is not None else 0)
                    + len(tokyo_state.open_positions)
                    + (1 if v4_state.position is not None else 0)
                )
                trial_state["open_trade_count"] = int(total_open_count)
                local_adapter.balance = float(balance)
                pre_nav, _pre_unreal, pre_used, _pre_avail, _pre_level, _btk, _atk, _bld, _ald = margin_metrics_full(row, i)
                local_adapter.equity = float(pre_nav)
                local_adapter.margin_used = float(pre_used)
                data_by_tf = _v44_data_by_tf(ts)
                ownership_audit = None
                try:
                    ownership_audit = compute_phase3_ownership_audit_for_data(data_by_tf, PIP_SIZE)
                except Exception:
                    ownership_audit = None
                tick = SimpleNamespace(bid=float(bid_c_london_bar), ask=float(ask_c_london_bar), time=ts.to_pydatetime())
                with contextlib.redirect_stdout(io.StringIO()):
                    v44_res = execute_v44_ny_entry(
                        adapter=local_adapter,
                        profile=phase3_profile,
                        policy=phase3_policy,
                        data_by_tf=data_by_tf,
                        tick=tick,
                        phase3_state=trial_state,
                        sizing_config={"v44_ny": v44_cfg},
                        now_utc=ts.to_pydatetime(),
                        store=local_phase3_store,
                        ownership_audit=ownership_audit,
                        overlay_state=overlay_state,
                    )
                decision = v44_res.get("decision")
                attempted = bool(getattr(decision, "attempted", False))
                placed = bool(getattr(decision, "placed", False))
                if placed:
                    units = int(v44_res.get("units", 0))
                    req_m = margin_required(units, params.leverage)
                    hard_avail_v44 = hard_margin_available_from_balance(
                        balance=balance,
                        ld=ld,
                        tokyo_state=tokyo_state,
                        open_positions=open_positions,
                        v4_position=v4_state.position,
                        leverage=params.leverage,
                        entry_margin_cap_pct=params.entry_margin_cap_pct,
                    )
                    if req_m > hard_avail_v44:
                        blocked["margin_v44"] += 1
                    else:
                        phase3_state = trial_state
                        parity_context = dict(v44_res.get("v44_parity_context") or {})
                        exit_plan = dict(v44_res.get("v44_exit_plan") or {})
                        strength = str(parity_context.get("strength") or "normal")
                        if strength == "strong":
                            trail_ema = int(v44_cfg.get("v5_strong_trail_ema", 21))
                        elif strength == "weak":
                            trail_ema = int(v44_cfg.get("v5_normal_trail_ema", 9))
                        else:
                            trail_ema = int(v44_cfg.get("v5_normal_trail_ema", 9))
                        pos = V44ManagedPosition(
                            entry_i=i,
                            entry_time=ts,
                            entry_price=float(v44_res.get("entry_price", tick.ask if getattr(decision, "side", "buy") == "buy" else tick.bid)),
                            side=str(getattr(decision, "side", "buy")).lower(),
                            lots_initial=float(units) / 100000.0,
                            lots_remaining=float(units) / 100000.0,
                            units=int(units),
                            exit_mode=str(exit_plan.get("mode", "managed_partial_runner")),
                            initial_stop_price=float(v44_res.get("sl_price")),
                            stop_price=float(v44_res.get("sl_price")),
                            tp1_price=float(v44_res.get("tp1_price")),
                            tp2_price=0.0,
                            tp2_pips=float(exit_plan.get("tp2_pips", 0.0)),
                            tp1_close_fraction=float(exit_plan.get("tp1_close_pct", 100.0)) / 100.0,
                            be_offset_pips=float(exit_plan.get("be_offset_pips", 0.0)),
                            trail_buffer_pips=float(exit_plan.get("trail_buffer_pips", 0.0)),
                            trail_ema_period=int(trail_ema),
                            strength=strength,
                            strategy_tag=str(v44_res.get("strategy_tag", "phase3:v44_ny")),
                            ownership_cell=str(parity_context.get("ownership_cell") or parity_context.get("_v44_cell_str") or ""),
                            entry_reason=str(getattr(decision, "reason", "")),
                        )
                        open_positions.append({"kind": "v44_live", "position": pos})
                else:
                    phase3_state = trial_state

        if v4_state.pending_order is not None and v4_state.position is None:
            if ts > v4_state.pending_order.expiry_time:
                v4_state.expired += 1
                v4_state.pending_order = None
            else:
                touched = float(row["low"]) <= v4_state.pending_order.trigger_level if v4_state.pending_order.side == "short" else float(row["high"]) >= v4_state.pending_order.trigger_level
                if touched:
                    hard_avail_v4 = hard_margin_available_from_balance(
                        balance=balance,
                        ld=ld,
                        tokyo_state=tokyo_state,
                        open_positions=open_positions,
                        v4_position=v4_state.position,
                        leverage=params.leverage,
                        entry_margin_cap_pct=params.entry_margin_cap_pct,
                    )
                    if v4_state.pending_order.margin_required_usd > hard_avail_v4:
                        v4_state.blocked_margin += 1
                        v4_state.pending_order = None
                    else:
                        entry_price = v4_entry_fill(v4_state.pending_order.trigger_level, v4_state.pending_order.side)
                        stop_distance, raw_stop = v4_stop_distances(
                            v4_state.pending_order.spike_low,
                            v4_state.pending_order.spike_high,
                            entry_price,
                            v4_state.pending_order.side,
                        )
                        if stop_distance is None:
                            v4_state.pending_order = None
                        else:
                            tp_distance_pips = v4_state.pending_order.spike_range_pips * V4_TP_FRACTION
                            if v4_state.pending_order.side == "long":
                                stop_price = entry_price - stop_distance * PIP_SIZE
                                tp_price = entry_price + tp_distance_pips * PIP_SIZE
                                high_water = entry_price
                                low_water = None
                            else:
                                stop_price = entry_price + stop_distance * PIP_SIZE
                                tp_price = entry_price - tp_distance_pips * PIP_SIZE
                                high_water = None
                                low_water = entry_price
                            v4_state.position = V4Position(
                                entry_time=ts,
                                direction=v4_state.pending_order.side,
                                entry_price=entry_price,
                                confirmation_time=ts,
                                confirmation_level=v4_state.pending_order.trigger_level,
                                spike_time=v4_state.pending_order.spike_time,
                                spike_direction=v4_state.pending_order.spike_direction,
                                session_name=v4_state.pending_order.session_name,
                                units=v4_state.pending_order.units,
                                stop_distance_pips=stop_distance,
                                raw_stop_distance_pips=raw_stop,
                                tp_distance_pips=tp_distance_pips,
                                stop_price=stop_price,
                                tp_price=tp_price,
                                margin_required_usd=v4_state.pending_order.margin_required_usd,
                                high_water=high_water,
                                low_water=low_water,
                            )
                            v4_state.fills += 1
                            v4_state.pending_order = None
                            v4_trade = maybe_close_v4_position_on_bar(
                                v4_state.position,
                                ts,
                                float(row["open"]),
                                bid_h_london_bar if v4_state.position.direction == "long" else ask_h_london_bar,
                                bid_l_london_bar if v4_state.position.direction == "long" else ask_l_london_bar,
                                bid_c_london_bar,
                                ask_c_london_bar,
                            )
                            if v4_trade is not None:
                                v4_state.trades.append(v4_trade)
                                combined_trades.append(v4_trade)
                                balance += float(v4_trade["pnl_usd"])
                                v4_state.position = None
                            v4_processed_this_bar = True

        if v4_state.position is not None and not v4_processed_this_bar:
            v4_trade = maybe_close_v4_position_on_bar(
                v4_state.position,
                ts,
                float(row["open"]),
                bid_h_london_bar if v4_state.position.direction == "long" else ask_h_london_bar,
                bid_l_london_bar if v4_state.position.direction == "long" else ask_l_london_bar,
                bid_c_london_bar,
                ask_c_london_bar,
            )
            if v4_trade is not None:
                v4_state.trades.append(v4_trade)
                combined_trades.append(v4_trade)
                balance += float(v4_trade["pnl_usd"])
                v4_state.position = None

        nav2, unreal2, used2, avail2, level2, bid_c_tokyo2, ask_c_tokyo2, bid_c_london2, ask_c_london2 = margin_metrics_full(row, i)
        intrabar_nav2, _ = nav_current(
            balance=balance,
            ld=ld,
            tokyo_state=tokyo_state,
            open_positions=open_positions,
            v4_position=v4_state.position,
            bid_c_tokyo=bid_l_tokyo,
            ask_c_tokyo=ask_h_tokyo,
            bid_c_london=bid_l_london_bar,
            ask_c_london=ask_h_london_bar,
            bid_c_v4=bid_l_london_bar,
            ask_c_v4=ask_h_london_bar,
        )
        intrabar_level2 = margin_level_pct(intrabar_nav2, used2)
        while used2 > 0 and intrabar_level2 < 100.0:
            candidates = collect_margin_call_candidates(
                ts=ts,
                ld=ld,
                tokyo_state=tokyo_state,
                open_positions=open_positions,
                v4=v4_state,
                bid_c_tokyo=bid_l_tokyo,
                ask_c_tokyo=ask_h_tokyo,
                bid_c_london=bid_l_london_bar,
                ask_c_london=ask_h_london_bar,
                bid_c_v4=bid_l_london_bar,
                ask_c_v4=ask_h_london_bar,
            )
            if not candidates:
                break
            worst = candidates[0]
            margin_calls.append(
                MarginCallEvent(
                    timestamp=ts,
                    strategy=str(worst["strategy"]),
                    pnl_usd=float(worst["pnl_usd"]),
                    nav_before=intrabar_nav2,
                    margin_used_before=used2,
                    margin_level_before=intrabar_level2,
                )
            )
            if worst["bucket"] == "v44":
                p = open_positions.pop(int(worst["idx"]))
                if p["kind"] == "v44_live":
                    pos: V44ManagedPosition = p["position"]
                    exit_px = float(bid_l_london_bar) if pos.side == "buy" else float(ask_h_london_bar)
                    trade = v44_close_trade(pos, ts, exit_px, "margin_call")
                    balance += float(trade["pnl_usd"])
                    combined_trades.append(trade)
                else:
                    pnl = mtm_v44(p, bid_l_tokyo, ask_h_tokyo)
                    balance += pnl
                    combined_trades.append(
                        {
                            "trade_id": "",
                            "strategy": "v44_ny",
                            "session": "v44_ny",
                            "setup_type": "oracle",
                            "entry_time": str(p["entry_time"]),
                            "exit_time": str(ts),
                            "entry_price": p["entry_price"],
                            "exit_price": bid_l_tokyo if p["side"] == "buy" else ask_h_tokyo,
                            "side": p["side"],
                            "lots": p["lots"],
                            "pnl_pips": "",
                            "pnl_usd": pnl,
                            "exit_reason": "margin_call",
                        }
                    )
            elif worst["bucket"] == "l1":
                p = open_positions.pop(int(worst["idx"]))
                pnl = mtm_l1(p, bid_l_london_bar, ask_h_london_bar)
                balance += pnl
                sim: L1Incremental = p["sim"]
                combined_trades.append(
                    {
                        "trade_id": "",
                        "strategy": "london_setup_d_l1",
                        "session": "london_v2",
                        "setup_type": "D",
                        "entry_time": str(sim.entry_time),
                        "exit_time": str(ts),
                        "entry_price": sim.entry_price,
                        "exit_price": bid_l_london_bar if sim.is_long else ask_h_london_bar,
                        "side": "buy" if sim.is_long else "sell",
                        "lots": int(p["units"]) / 100000.0,
                        "pnl_pips": "",
                        "pnl_usd": pnl,
                        "exit_reason": "margin_call",
                    }
                )
            elif worst["bucket"] == "london" and ld is not None:
                p = ld.open_positions.pop(int(worst["idx"]))
                current_px = bid_l_london_bar if p.direction == "long" else ask_h_london_bar
                _pips, usd = london_bt.calc_leg_pnl(p.direction, p.entry_price, current_px, p.remaining_units)
                pnl = float(p.pnl_usd_realized) + float(usd)
                balance += pnl
                combined_trades.append(
                    {
                        "trade_id": "",
                        "strategy": f"london_setup_{p.setup_type.lower()}",
                        "session": "london_v2",
                        "setup_type": p.setup_type,
                        "entry_time": str(p.entry_time),
                        "exit_time": str(ts),
                        "entry_price": p.entry_price,
                        "exit_price": current_px,
                        "side": "buy" if p.direction == "long" else "sell",
                        "lots": p.initial_units / 100000.0,
                        "pnl_pips": "",
                        "pnl_usd": pnl,
                        "exit_reason": "margin_call",
                    }
                )
            elif worst["bucket"] == "tokyo":
                p = tokyo_state.open_positions.pop(int(worst["idx"]))
                current_px = bid_l_tokyo if p.direction == "long" else ask_h_tokyo
                _pips, usd = tokyo_calc_leg_usd_pips(p.direction, p.entry_price, current_px, p.units_remaining)
                pnl = float(p.realized_usd) + float(usd)
                balance += pnl
                combined_trades.append(
                    {
                        "trade_id": "",
                        "strategy": "tokyo_v14",
                        "session": "tokyo_v14",
                        "setup_type": "tokyo_v14",
                        "entry_time": str(p.entry_time),
                        "exit_time": str(ts),
                        "entry_price": p.entry_price,
                        "exit_price": current_px,
                        "side": "buy" if p.direction == "long" else "sell",
                        "lots": p.units_initial / 100000.0,
                        "pnl_pips": "",
                        "pnl_usd": pnl,
                        "exit_reason": "margin_call",
                    }
                )
            elif worst["bucket"] == "v4" and v4_state.position is not None:
                trade = close_v4_margin_call(v4_state.position, ts, float(bid_l_london_bar), float(ask_h_london_bar))
                balance += float(trade["pnl_usd"])
                v4_state.trades.append(trade)
                combined_trades.append(trade)
                v4_state.position = None
            nav2, unreal2, used2, avail2, level2, bid_c_tokyo2, ask_c_tokyo2, bid_c_london2, ask_c_london2 = margin_metrics_full(row, i)
            intrabar_nav2, _ = nav_current(
                balance=balance,
                ld=ld,
                tokyo_state=tokyo_state,
                open_positions=open_positions,
                v4_position=v4_state.position,
                bid_c_tokyo=bid_l_tokyo,
                ask_c_tokyo=ask_h_tokyo,
                bid_c_london=bid_l_london_bar,
                ask_c_london=ask_h_london_bar,
                bid_c_v4=bid_l_london_bar,
                ask_c_v4=ask_h_london_bar,
            )
            intrabar_level2 = margin_level_pct(intrabar_nav2, used2)

        peak_nav = max(peak_nav, nav2)
        open_ct = (
            len(open_positions)
            + (len(ld.open_positions) if ld is not None else 0)
            + len(tokyo_state.open_positions)
            + (1 if v4_state.position is not None else 0)
        )
        max_open_peak = max(max_open_peak, open_ct)
        equity_rows.append(
            {
                "bar_index": i,
                "timestamp": str(ts),
                "balance": float(balance),
                "nav": float(nav2),
                "unrealized_pnl": float(unreal2),
                "margin_used": float(used2),
                "margin_available": float(avail2),
                "margin_level_pct": float(level2),
                "open_position_count": int(open_ct),
                "drawdown_from_peak_nav": float(peak_nav - nav2),
                "drawdown_pct_nav": float((peak_nav - nav2) / peak_nav * 100.0) if peak_nav > 0 else 0.0,
                "v4_pending_order": int(v4_state.pending_order is not None),
                "v4_open_position": int(v4_state.position is not None),
            }
        )

        if not params.quiet and i % 200000 == 0:
            elapsed = time.time() - loop_t0
            print(
                f"progress {i:,}/{n:,} | balance={balance:,.2f} nav={nav2:,.2f} "
                f"margin_used={used2:,.2f} margin_level={level2:,.2f}% "
                f"v4_fills={v4_state.fills} elapsed={elapsed:,.1f}s",
                flush=True,
            )

    trades_df = pd.DataFrame(combined_trades)
    v4_df = pd.DataFrame(v4_state.trades)
    equity_df = pd.DataFrame(equity_rows)

    return {
        "combined_trades": trades_df,
        "v4_trades": v4_df,
        "equity": equity_df,
        "summary_meta": {
            "starting_equity": params.starting_equity,
            "entry_margin_cap_pct": params.entry_margin_cap_pct,
            "ending_balance": float(balance),
            "ending_nav": float(equity_df["nav"].iloc[-1]) if not equity_df.empty else float(balance),
            "max_concurrent_positions": int(max_open_peak),
            "v4_state": {
                "broad_candidates": v4_state.broad_candidates,
                "family_c_events": v4_state.family_c_events,
                "armed_orders": v4_state.armed_orders,
                "fills": v4_state.fills,
                "blocked_active": v4_state.blocked_active,
                "blocked_margin": v4_state.blocked_margin,
                "expired": v4_state.expired,
            },
            "blocked_counts": dict(blocked),
            "margin_call_count": len(margin_calls),
            "elapsed_seconds": round(time.time() - t0, 2),
        },
        "margin_calls": [mc.__dict__ for mc in margin_calls],
    }


def build_summary_text(result: dict[str, Any], params: CombinedParams) -> str:
    trades = result["combined_trades"].copy()
    if not trades.empty:
        trades["pnl_usd"] = pd.to_numeric(trades["pnl_usd"], errors="coerce")
    pnls = trades["pnl_usd"].dropna() if not trades.empty else pd.Series(dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    equity = result["equity"]
    meta = result["summary_meta"]
    v4 = result["v4_trades"]

    lines = []
    lines.append("V7.1 + H1 + V4 BAR-BY-BAR MARGIN ENGINE")
    lines.append("=======================================")
    lines.append(f"Data: {params.data_path}")
    lines.append(f"Starting equity: ${params.starting_equity:,.2f}")
    lines.append(f"Leverage: {params.leverage:.1f}:1")
    lines.append(f"V4 lots: {params.v4_lots:.2f}")
    lines.append(
        f"Entry admission: hard open-notional cap from realized balance "
        f"({params.entry_margin_cap_pct:.0f}% cap)"
    )
    lines.append("Execution: native bar-by-bar V44 entries/exits + adverse same-bar V4 handling")
    lines.append("")
    lines.append("Top line:")
    lines.append(f"  Combined trades: {len(trades)}")
    lines.append(f"  Combined PF: {calc_pf(pnls):.2f}")
    lines.append(f"  Combined net: ${pnls.sum():,.2f}")
    if not equity.empty:
        max_dd = float(equity['drawdown_from_peak_nav'].max())
        max_dd_pct = float(equity['drawdown_pct_nav'].max())
        lines.append(f"  Max DD (NAV): ${max_dd:,.2f} ({max_dd_pct:.2f}%)")
        lines.append(f"  Peak margin used: ${float(equity['margin_used'].max()):,.2f}")
        lines.append(f"  Lowest margin level: {float(equity['margin_level_pct'].replace([np.inf], np.nan).min()):,.2f}%")
    lines.append("")
    lines.append("Breakdown by strategy:")
    if not trades.empty:
        for strat, g in trades.groupby("strategy"):
            p = g["pnl_usd"].dropna()
            lines.append(f"  - {strat}: {len(g)} trades | PF {calc_pf(p):.2f} | ${p.sum():,.2f}")
    lines.append("")
    lines.append("H1 filter stats:")
    for k, v in meta["h1_filter_stats"].items():
        lines.append(f"  - {k}: {v}")
    lines.append("")
    lines.append("V4 funnel:")
    for k, v in meta["v4_state"].items():
        lines.append(f"  - {k}: {v}")
    lines.append("")
    lines.append("Margin blocks:")
    if meta["blocked_counts"]:
        for k, v in sorted(meta["blocked_counts"].items()):
            lines.append(f"  - {k}: {v}")
    else:
        lines.append("  - none")
    lines.append("")
    lines.append(f"Margin calls: {meta['margin_call_count']}")
    for mc in result["margin_calls"][:20]:
        lines.append(
            f"  - {mc['timestamp']} | {mc['strategy']} | pnl ${mc['pnl_usd']:,.2f} | "
            f"NAV ${mc['nav_before']:,.2f} | used ${mc['margin_used_before']:,.2f} | "
            f"level {mc['margin_level_before']:.2f}%"
        )
    lines.append("")
    lines.append("V4 results:")
    if not v4.empty:
        v4p = pd.to_numeric(v4["pnl_usd"], errors="coerce").dropna()
        lines.append(f"  - trades: {len(v4)}")
        lines.append(f"  - PF: {calc_pf(v4p):.2f}")
        lines.append(f"  - net: ${v4p.sum():,.2f}")
    else:
        lines.append("  - no V4 trades closed")
    lines.append("")
    lines.append(f"Elapsed: {meta['elapsed_seconds']:.1f}s")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = execute_combined(
        CombinedParams(
            data_path=str(args.data_path),
            starting_equity=float(args.starting_equity),
            leverage=float(args.leverage),
            v4_lots=float(args.v4_lots),
            entry_margin_cap_pct=float(args.entry_margin_cap_pct),
            spread_mode=str(args.spread_mode),
            max_bars=args.max_bars,
            quiet=bool(args.quiet),
        )
    )
    result["combined_trades"].to_csv(COMBINED_TRADES_OUT, index=False)
    result["v4_trades"].to_csv(V4_TRADES_OUT, index=False)
    result["equity"].to_csv(EQUITY_OUT, index=False)
    SUMMARY_JSON_OUT.write_text(json.dumps(result["summary_meta"], indent=2, default=str), encoding="utf-8")
    SUMMARY_TXT_OUT.write_text(
        build_summary_text(
            result,
            CombinedParams(
                data_path=str(args.data_path),
                starting_equity=float(args.starting_equity),
                leverage=float(args.leverage),
                v4_lots=float(args.v4_lots),
                entry_margin_cap_pct=float(args.entry_margin_cap_pct),
                spread_mode=str(args.spread_mode),
                max_bars=args.max_bars,
                quiet=bool(args.quiet),
            ),
        ),
        encoding="utf-8",
    )
    print(f"Wrote {COMBINED_TRADES_OUT}")
    print(f"Wrote {V4_TRADES_OUT}")
    print(f"Wrote {EQUITY_OUT}")
    print(f"Wrote {SUMMARY_JSON_OUT}")
    print(f"Wrote {SUMMARY_TXT_OUT}")


if __name__ == "__main__":
    main()
