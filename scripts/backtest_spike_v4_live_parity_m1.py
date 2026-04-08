#!/usr/bin/env python3
"""Spike Fade V4 — live-path backtest on M1 (isolated, incremental).

This is the “do we match live?” harness for Spike Fade V4.

It mirrors the live stack’s key semantics:

- **Signal semantics**: Family-C spike+fade logic from ``core/phase3_spike_fade_v4._event_from_spike_fade``.
- **Order semantics**: a GTD **LIMIT** resting at ``trigger_level``.
- **Fill semantics**: fill price is **the limit price** (``trigger_level``).
- **Stop/TP semantics**: use ``V4Event.stop_price`` / ``tp_price`` computed at arm time (same as live).
- **Exit semantics**: reuse the intrabar close logic from
  ``scripts/run_v71_h1_v4_margin_bar_by_bar.py::maybe_close_v4_position_on_bar`` (TP/SL/trail/prove-it/time stop).

Limit touch rule (standard limit behavior, using mid OHLC):

- buy: bar ``low <= trigger_level``
- sell: bar ``high >= trigger_level``

Spread model for bid/ask conversion uses the London execution model helper
(``scripts/backtest_v2_multisetup_london.py``), same as the combined v71 runner.

Performance: this script builds M5/M15 incrementally (O(bars)), so a full 1M-bar run is feasible.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.phase3_spike_fade_v4 import V4Event, _event_from_spike_fade, default_v4_runtime_config
from core.phase3_v7_pfdd_defended_runner import LONDON_CFG_PATH, PIP_SIZE
from scripts import backtest_v2_multisetup_london as london_bt
from scripts.run_spike_fade_v4_backtest_v2 import bucket_end, make_m5_bar
from scripts.run_v71_h1_v4_margin_bar_by_bar import V4Position, maybe_close_v4_position_on_bar


@dataclass
class _Pending:
    event: V4Event
    expiry_time: pd.Timestamp
    trigger_level: float
    units: int
    source: str
    event_id: str


def _load_london_cfg() -> dict[str, Any]:
    return json.loads(LONDON_CFG_PATH.read_text(encoding="utf-8"))


def _pending_touched(side: str, row: pd.Series, trigger: float) -> bool:
    if side == "buy":
        return float(row["low"]) <= float(trigger)
    if side == "sell":
        return float(row["high"]) >= float(trigger)
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Spike V4 live-path M1 backtest (isolated)")
    ap.add_argument(
        "--data-path",
        default=str(ROOT / "research_out/USDJPY_M1_OANDA_1000k.csv"),
        help="M1 CSV with time,open,high,low,close",
    )
    ap.add_argument("--max-bars", type=int, default=None)
    ap.add_argument("--starting-equity", type=float, default=100_000.0)
    ap.add_argument("--leverage", type=float, default=33.0)
    ap.add_argument("--v4-lots", type=float, default=20.0)
    ap.add_argument(
        "--entry-spread-pips",
        type=float,
        default=1.6,
        help="Entry spread pips used inside V4Event stop/TP computation (matches live cfg entry_spread_pips).",
    )
    ap.add_argument("--shared-margin-cap-pct", type=float, default=75.0)
    ap.add_argument(
        "--ladder-level-2-pips",
        type=float,
        default=0.0,
        help="Second limit level offset (pips) from trigger. 0 disables ladder.",
    )
    ap.add_argument(
        "--ladder-level-2-fraction",
        type=float,
        default=0.5,
        help="Fraction of size allocated to level-2 when enabled (0..1).",
    )
    ap.add_argument(
        "--fallback-market-minutes",
        type=int,
        default=0,
        help="If >0, place reduced market entry N minutes before pending expiry when still unfilled.",
    )
    ap.add_argument(
        "--fallback-market-fraction",
        type=float,
        default=0.5,
        help="Fraction of total units used for market fallback (0..1).",
    )
    ap.add_argument(
        "--fallback-max-spread-pips",
        type=float,
        default=2.5,
        help="Skip market fallback if current spread exceeds this value.",
    )
    ap.add_argument(
        "--output-dir",
        default=str(
            ROOT / "research_out/trade_analysis/spike_fade_v4/backtest/spike_v4_live_parity_m1"
        ),
    )
    args = ap.parse_args()

    data_path = Path(args.data_path)
    if not data_path.is_file():
        print(f"ERROR: missing data file: {data_path}", file=sys.stderr)
        sys.exit(1)

    london_cfg = _load_london_cfg()
    df = pd.read_csv(data_path)
    if "time" not in df.columns:
        print("ERROR: CSV must have a 'time' column", file=sys.stderr)
        sys.exit(1)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).reset_index(drop=True)
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)

    n = len(df) if args.max_bars is None else min(len(df), int(args.max_bars))
    cfg = default_v4_runtime_config()
    cfg["enabled"] = True
    cfg["lots"] = float(args.v4_lots)
    cfg["entry_spread_pips"] = float(args.entry_spread_pips)

    runtime: dict[str, Any] = {}
    pending_orders: list[_Pending] = []
    position: V4Position | None = None
    trades: list[dict[str, Any]] = []
    balance = float(args.starting_equity)
    units = int(round(float(args.v4_lots) * 100_000.0))
    margin_req = float(abs(units)) / max(1.0, float(args.leverage))
    lvl2_frac = max(0.0, min(1.0, float(args.ladder_level_2_fraction)))
    fallback_frac = max(0.0, min(1.0, float(args.fallback_market_fraction)))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Incremental M15 EMA50 and M5 featured-bar stream (matches v71’s bar pipeline).
    current_15_end: pd.Timestamp | None = None
    m15_close: float = 0.0
    ema50_prev: float | None = None
    last_completed_m15_ema50: float | None = None

    current_5_end: pd.Timestamp | None = None
    m5_open: float = 0.0
    m5_high: float = 0.0
    m5_low: float = 0.0
    m5_close: float = 0.0

    tr_queue: deque[float] = deque()
    tr_sum: float = 0.0
    ema20_prev: float | None = None
    m5_bars: list[Any] = []

    # Spike/fade detection uses spike=prev_m5 and fade=last_m5.
    last_m5_spike_candidate: Any | None = None

    warmup_ready: bool = False
    last_processed_m5_end: pd.Timestamp | None = None
    cluster_block_until: pd.Timestamp | None = None
    fills_limit = 0
    fills_market_fallback = 0

    t0 = time.perf_counter()
    for i in range(n):
        row = df.iloc[i]
        ts = pd.Timestamp(row["time"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")

        sp = london_bt.compute_spread_pips(i, ts, london_cfg)
        bid_h, ask_h = london_bt.to_bid_ask(float(row["high"]), sp)
        bid_l, ask_l = london_bt.to_bid_ask(float(row["low"]), sp)
        bid_c, ask_c = london_bt.to_bid_ask(float(row["close"]), sp)

        if position is not None:
            tr = maybe_close_v4_position_on_bar(
                position,
                ts,
                float(row["open"]),
                bid_h if position.direction == "long" else ask_h,
                bid_l if position.direction == "long" else ask_l,
                bid_c,
                ask_c,
            )
            if tr is not None:
                balance += float(tr["pnl_usd"])
                tr["exit_bar_index"] = i
                tr["entry_bar_index"] = ""
                trades.append(tr)
                position = None
                runtime.pop("active_trade_id", None)
                runtime["lifecycle_state"] = "READY"

        # ---- Incremental M15 state (EMA50 on 15-min closes) ----
        next_15 = bucket_end(ts, 15)
        if current_15_end is None:
            current_15_end = next_15
            m15_close = float(row["close"])
        elif next_15 != current_15_end:
            close15 = float(m15_close)
            if ema50_prev is None:
                ema50_prev = close15
            else:
                k = 2.0 / 51.0
                ema50_prev = close15 * k + ema50_prev * (1.0 - k)
            last_completed_m15_ema50 = ema50_prev
            current_15_end = next_15
            m15_close = float(row["close"])
        else:
            m15_close = float(row["close"])

        # ---- Incremental M5 featured bars (ATR14, EMA20, prior-12 extremes, broad-candidate) ----
        next_5 = bucket_end(ts, 5)
        if current_5_end is None:
            current_5_end = next_5
            m5_open = float(row["open"])
            m5_high = float(row["high"])
            m5_low = float(row["low"])
            m5_close = float(row["close"])
        elif next_5 == current_5_end:
            m5_high = max(m5_high, float(row["high"]))
            m5_low = min(m5_low, float(row["low"]))
            m5_close = float(row["close"])
        else:
            bar, tr_sum, ema20_prev = make_m5_bar(
                end_time=current_5_end,
                end_idx=i - 1,
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
            m5_bars.append(bar)
            last_processed_m5_end = current_5_end

            current_5_end = next_5
            m5_open = float(row["open"])
            m5_high = float(row["high"])
            m5_low = float(row["low"])
            m5_close = float(row["close"])

            warmup_ready = bool(
                len(m5_bars) >= 150
                and (bar.m15_ema50 is not None)
                and (bar.atr14_pips is not None)
                and (bar.prior_12_high is not None)
                and (bar.prior_12_low is not None)
            )

            # Live’s cluster block: don’t arm a new event until the cooldown expires.
            if cluster_block_until is not None and current_5_end <= cluster_block_until:
                last_m5_spike_candidate = bar if bar.broad_candidate else None
            else:
                # Event requires prev spike + current fade.
                if warmup_ready and last_m5_spike_candidate is not None:
                    spike_s = pd.Series(
                        {
                            "time": last_m5_spike_candidate.end_time,
                            "open": last_m5_spike_candidate.open,
                            "high": last_m5_spike_candidate.high,
                            "low": last_m5_spike_candidate.low,
                            "close": last_m5_spike_candidate.close,
                            "atr14_pips": last_m5_spike_candidate.atr14_pips,
                            "ema20": last_m5_spike_candidate.ema20,
                            "m15_ema50": last_m5_spike_candidate.m15_ema50,
                            "prior_12_high": last_m5_spike_candidate.prior_12_high,
                            "prior_12_low": last_m5_spike_candidate.prior_12_low,
                            "range_pips": last_m5_spike_candidate.range_pips,
                            "broad_candidate": bool(last_m5_spike_candidate.broad_candidate),
                            "spike_direction": last_m5_spike_candidate.spike_direction,
                        }
                    )
                    fade_s = pd.Series(
                        {
                            "time": bar.end_time,
                            "open": bar.open,
                            "high": bar.high,
                            "low": bar.low,
                            "close": bar.close,
                            "atr14_pips": bar.atr14_pips,
                            "ema20": bar.ema20,
                            "m15_ema50": bar.m15_ema50,
                            "prior_12_high": bar.prior_12_high,
                            "prior_12_low": bar.prior_12_low,
                            "range_pips": bar.range_pips,
                            "broad_candidate": bool(bar.broad_candidate),
                            "spike_direction": bar.spike_direction,
                        }
                    )
                    event = _event_from_spike_fade(spike_s, fade_s, cfg)
                    if event is not None and not pending_orders and position is None:
                        exp = pd.Timestamp(event.expiry_time)
                        exp = exp.tz_localize("UTC") if exp.tzinfo is None else exp.tz_convert("UTC")
                        if margin_req / max(1e-9, balance) <= float(args.shared_margin_cap_pct) / 100.0:
                            event_id = f"{event.armed_time}|{event.side}|{event.trigger_level:.3f}"
                            units_l1 = units
                            units_l2 = 0
                            if float(args.ladder_level_2_pips) > 0.0 and lvl2_frac > 0.0:
                                units_l2 = int(round(units * lvl2_frac))
                                units_l1 = max(0, units - units_l2)
                            if units_l1 > 0:
                                pending_orders.append(
                                    _Pending(
                                        event=event,
                                        expiry_time=exp,
                                        trigger_level=float(event.trigger_level),
                                        units=units_l1,
                                        source="limit_l1",
                                        event_id=event_id,
                                    )
                                )
                            if units_l2 > 0:
                                lvl2_px = float(args.ladder_level_2_pips) * PIP_SIZE
                                trigger2 = (
                                    float(event.trigger_level) - lvl2_px
                                    if str(event.side) == "buy"
                                    else float(event.trigger_level) + lvl2_px
                                )
                                pending_orders.append(
                                    _Pending(
                                        event=event,
                                        expiry_time=exp,
                                        trigger_level=float(trigger2),
                                        units=units_l2,
                                        source="limit_l2",
                                        event_id=event_id,
                                    )
                                )

                last_m5_spike_candidate = bar if bar.broad_candidate else None

        if pending_orders and position is None:
            pending_orders = [p for p in pending_orders if ts <= p.expiry_time]
            if pending_orders:
                # Prefer best limit if multiple levels are touched in same bar.
                if str(pending_orders[0].event.side) == "buy":
                    pending_orders.sort(key=lambda p: p.trigger_level)
                else:
                    pending_orders.sort(key=lambda p: p.trigger_level, reverse=True)
            filled_pending: _Pending | None = None
            for p in pending_orders:
                if _pending_touched(p.event.side, row, p.trigger_level):
                    filled_pending = p
                    break

            chosen_event_id: str | None = None
            entry_px: float | None = None
            fill_source: str | None = None
            fill_units: int = 0
            ev: V4Event | None = None

            if filled_pending is not None:
                chosen_event_id = filled_pending.event_id
                entry_px = float(filled_pending.trigger_level)
                fill_source = filled_pending.source
                fill_units = int(filled_pending.units)
                ev = filled_pending.event
            elif int(args.fallback_market_minutes) > 0 and pending_orders:
                probe = pending_orders[0]
                minutes_to_expiry = (probe.expiry_time - ts).total_seconds() / 60.0
                if 0.0 <= minutes_to_expiry <= float(args.fallback_market_minutes) and float(sp) <= float(args.fallback_max_spread_pips):
                    fallback_units = int(round(units * fallback_frac))
                    fallback_units = max(0, min(fallback_units, sum(int(x.units) for x in pending_orders)))
                    if fallback_units > 0:
                        chosen_event_id = probe.event_id
                        entry_px = float(ask_c) if str(probe.event.side) == "buy" else float(bid_c)
                        fill_source = "market_fallback"
                        fill_units = fallback_units
                        ev = probe.event

            if ev is not None and entry_px is not None and fill_units > 0:
                leg_margin_req = float(abs(fill_units)) / max(1.0, float(args.leverage))
                if leg_margin_req / max(1e-9, balance) <= float(args.shared_margin_cap_pct) / 100.0:
                    direction = "long" if ev.side == "buy" else "short"
                    # Preserve arm-time stop/tp distance from event trigger.
                    stop_delta = abs(float(ev.stop_price) - float(ev.trigger_level))
                    tp_base = float(ev.tp_price) if ev.tp_price is not None else float(ev.trigger_level)
                    tp_delta = abs(float(tp_base) - float(ev.trigger_level))
                    if direction == "long":
                        stop_px = float(entry_px - stop_delta)
                        tp_px = float(entry_px + tp_delta)
                    else:
                        stop_px = float(entry_px + stop_delta)
                        tp_px = float(entry_px - tp_delta)
                    raw_stop = abs(float(entry_px) - float(stop_px)) / PIP_SIZE
                    tp_dist = abs(float(tp_px) - float(entry_px)) / PIP_SIZE
                    position = V4Position(
                        entry_time=ts,
                        direction=direction,
                        entry_price=float(entry_px),
                        confirmation_time=pd.Timestamp(ev.confirmation_time),
                        confirmation_level=float(ev.confirmation_level),
                        spike_time=pd.Timestamp(ev.spike_time),
                        spike_direction=str(ev.spike_direction),
                        session_name=str(ev.session_name),
                        units=int(fill_units),
                        stop_distance_pips=float(raw_stop),
                        raw_stop_distance_pips=float(raw_stop),
                        tp_distance_pips=float(tp_dist),
                        stop_price=float(stop_px),
                        tp_price=float(tp_px),
                        margin_required_usd=float(leg_margin_req),
                    )
                    if fill_source == "market_fallback":
                        fills_market_fallback += 1
                    else:
                        fills_limit += 1
                    cluster_min = int(cfg.get("cluster_block_minutes", 120))
                    cluster_block_until = ts + pd.Timedelta(minutes=cluster_min)
                    runtime["cluster_block_until"] = cluster_block_until.isoformat()
                    runtime["lifecycle_state"] = "POSITION_OPEN"
                    tr_same = maybe_close_v4_position_on_bar(
                        position,
                        ts,
                        float(row["open"]),
                        bid_h if position.direction == "long" else ask_h,
                        bid_l if position.direction == "long" else ask_l,
                        bid_c,
                        ask_c,
                    )
                    if tr_same is not None:
                        balance += float(tr_same["pnl_usd"])
                        tr_same["exit_bar_index"] = i
                        tr_same["entry_bar_index"] = ""
                        tr_same["fill_source"] = str(fill_source)
                        trades.append(tr_same)
                        position = None
                        runtime.pop("active_trade_id", None)
                        runtime["lifecycle_state"] = "READY"
                # Cancel siblings after any fill attempt for this event (one active at a time).
                if chosen_event_id is not None:
                    pending_orders = [p for p in pending_orders if p.event_id != chosen_event_id]

        if (i + 1) % 200_000 == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / max(1e-9, elapsed)
            print(
                f"progress {i+1:,}/{n:,} | ~{rate:,.0f} bars/s | "
                f"balance={balance:,.2f} pending={len(pending_orders)} pos={int(position is not None)} "
                f"m5={len(m5_bars):,}",
                flush=True,
            )

    trades_path = out_dir / "v4_live_parity_trades.csv"
    if trades:
        pd.DataFrame(trades).to_csv(trades_path, index=False)
    else:
        trades_path.write_text("", encoding="utf-8")

    pnl = pd.to_numeric(pd.Series([float(t.get("pnl_usd", 0.0)) for t in trades]), errors="coerce").fillna(0.0)
    summary = {
        "bars": int(n),
        "starting_equity": float(args.starting_equity),
        "ending_balance": float(balance),
        "net_pnl_usd": float(pnl.sum()),
        "trades": int(len(trades)),
        "data_path": str(data_path.resolve()),
        "touch_rule": "buy: low<=trigger, sell: high>=trigger",
        "fill_price": "trigger_level (limit)",
        "entry_spread_pips_for_event": float(args.entry_spread_pips),
        "ladder_level_2_pips": float(args.ladder_level_2_pips),
        "ladder_level_2_fraction": float(lvl2_frac),
        "fallback_market_minutes": int(args.fallback_market_minutes),
        "fallback_market_fraction": float(fallback_frac),
        "fallback_max_spread_pips": float(args.fallback_max_spread_pips),
        "fills_limit": int(fills_limit),
        "fills_market_fallback": int(fills_market_fallback),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {trades_path}")


if __name__ == "__main__":
    main()
