from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict

from core.london_v2_entry_evaluator import evaluate_london_v2_entry_signal
from scripts.backtest_v2_multisetup_london import (
    ROUND_UNITS,
    PIP_SIZE,
    clamp_sl,
    merge_config,
    pip_value_per_unit,
    uk_london_open_utc,
    us_ny_open_utc,
)

from .models import ClosedTrade, ExitAction, PortfolioSnapshot, PositionSnapshot, Signal
from .strategy import BarView, HistoricalDataView, TrainableStrategyFamily


class V2LondonStrategyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    family_name: str = "london_v2"
    config: dict[str, Any]

    @classmethod
    def from_json(cls, path: str | Path, *, family_name: str = "london_v2") -> "V2LondonStrategyConfig":
        raw = json.loads(Path(path).read_text())
        return cls(family_name=family_name, config=merge_config(raw))


@dataclass
class _ChannelState:
    state: str = "ARMED"
    cooldown_until: pd.Timestamp | None = None
    entries: int = 0
    resets: int = 0


@dataclass
class _TradePlan:
    setup_type: str
    direction: str
    tp1_price: float
    tp2_price: float
    tp1_close_fraction: float
    be_price_after_tp1: float
    hard_close: pd.Timestamp
    delayed_hard_close: pd.Timestamp
    grace_trail_distance_pips: float
    extend_runner_until_ny_start_delay: bool
    risk_usd_planned: float
    tp1_hit: bool = False


@dataclass
class _DayState:
    day_start: pd.Timestamp
    london_open: pd.Timestamp
    ny_open: pd.Timestamp
    hard_close: pd.Timestamp
    delayed_hard_close: pd.Timestamp
    asian_high: float = float("-inf")
    asian_low: float = float("inf")
    asian_valid: bool = False
    asian_range_pips: float | None = None
    lor_high: float = float("-inf")
    lor_low: float = float("inf")
    lor_valid: bool = False
    lor_range_pips: float | None = None
    day_entries_total: int = 0
    day_entries_setup: dict[str, int] = field(default_factory=lambda: {k: 0 for k in ("A", "B", "C", "D")})
    day_entries_setup_dir: dict[tuple[str, str], int] = field(
        default_factory=lambda: {(s, d): 0 for s in ("A", "B", "C", "D") for d in ("long", "short")}
    )
    daily_trade_sequence: int = 0
    channels: dict[tuple[str, str], _ChannelState] = field(
        default_factory=lambda: {(s, d): _ChannelState() for s in ("A", "B", "C", "D") for d in ("long", "short")}
    )
    b_candidates: dict[str, list[dict[str, Any]]] = field(default_factory=lambda: {"long": [], "short": []})


class V2LondonStrategy(TrainableStrategyFamily):
    family_name = "london_v2"

    def __init__(self, config: V2LondonStrategyConfig) -> None:
        self.config = config
        self.family_name = config.family_name
        self.cfg = config.config
        self._current_day: pd.Timestamp | None = None
        self._day_state: _DayState | None = None
        self._trade_plans: dict[int, _TradePlan] = {}
        self._bar_index = -1

    def fit(self, history: HistoricalDataView) -> None:
        return None

    def evaluate(self, current_bar: BarView, history: HistoricalDataView, portfolio: PortfolioSnapshot) -> Signal | None:
        ts = pd.Timestamp(current_bar.timestamp)
        self._bar_index = int(current_bar.bar_index)
        self._reset_day_if_needed(ts)
        assert self._day_state is not None

        state = self._day_state
        day_name = ts.day_name()
        if day_name not in set(self.cfg["session"]["active_days_utc"]):
            return None

        self._update_ranges(current_bar, ts)
        self._apply_channel_resets(current_bar, ts)

        if ts >= state.hard_close:
            return None
        if int(current_bar.bar_index) + 1 >= len(history):
            return None

        row = {
            "open": float(current_bar.mid_open),
            "high": float(current_bar.mid_high),
            "low": float(current_bar.mid_low),
            "close": float(current_bar.mid_close),
        }
        windows = self._windows_for_day(state)
        channels = {
            key: {"state": value.state, "entries": value.entries, "resets": value.resets}
            for key, value in state.channels.items()
        }
        nxt_ts = pd.Timestamp(history[int(current_bar.bar_index) + 1].timestamp)
        entries, state.b_candidates = evaluate_london_v2_entry_signal(
            row=row,
            cfg=self.cfg,
            asian_high=float(state.asian_high) if state.asian_high != float("-inf") else float("nan"),
            asian_low=float(state.asian_low) if state.asian_low != float("inf") else float("nan"),
            asian_range_pips=float(state.asian_range_pips or 0.0),
            asian_valid=bool(state.asian_valid),
            lor_high=float(state.lor_high) if state.lor_high != float("-inf") else float("nan"),
            lor_low=float(state.lor_low) if state.lor_low != float("inf") else float("nan"),
            lor_range_pips=float(state.lor_range_pips or 0.0),
            lor_valid=bool(state.lor_valid),
            ts=ts,
            nxt_ts=nxt_ts,
            bar_index=int(current_bar.bar_index),
            windows=windows,
            channels=channels,
            b_candidates=state.b_candidates,
        )
        if not entries:
            return None

        entry = self._first_fillable_entry(entries, nxt_ts, state, portfolio)
        if entry is None:
            return None

        setup = str(entry["setup_type"])
        direction = str(entry["direction"])
        s_cfg = self.cfg["setups"][setup]
        entry_price = float(current_bar.ask_close) if direction == "long" else float(current_bar.bid_close)
        sl_preview, sl_pips = clamp_sl(
            entry_price,
            float(entry["raw_sl"]),
            direction,
            float(s_cfg["sl_min_pips"]),
            float(s_cfg["sl_max_pips"]),
        )
        risk_pct_trade = float(s_cfg.get("risk_per_trade_pct", self.cfg["risk"]["risk_per_trade_pct"]))
        risk_usd = float(portfolio.equity) * risk_pct_trade
        open_risk = sum(
            float(plan.risk_usd_planned)
            for plan in self._trade_plans.values()
        )
        if open_risk + risk_usd > float(portfolio.equity) * float(self.cfg["risk"]["max_total_open_risk_pct"]):
            return None
        units = int(math.floor(risk_usd / max(1e-9, sl_pips * pip_value_per_unit(entry_price)) / ROUND_UNITS) * ROUND_UNITS)
        if units <= 0:
            return None
        leverage = float(self.cfg["account"]["leverage"])
        req_margin = (units * entry_price * PIP_SIZE) / leverage
        if req_margin > float(self.cfg["account"]["max_margin_usage_fraction_per_trade"]) * max(0.0, float(portfolio.available_margin)):
            return None
        sign = 1.0 if direction == "long" else -1.0
        tp1_pips = float(s_cfg["tp1_r_multiple"]) * sl_pips
        tp2_pips = float(s_cfg["tp2_r_multiple"]) * sl_pips
        return Signal(
            family=self.family_name,
            direction=direction,
            stop_loss=float(sl_preview),
            take_profit=None,
            size=int(units),
            metadata={
                "strategy": self.family_name,
                "setup_type": setup,
                "raw_stop_price": float(entry["raw_sl"]),
                "sl_min_pips": float(s_cfg["sl_min_pips"]),
                "sl_max_pips": float(s_cfg["sl_max_pips"]),
                "tp1_r_multiple": float(s_cfg["tp1_r_multiple"]),
                "tp2_r_multiple": float(s_cfg["tp2_r_multiple"]),
                "tp1_close_fraction": float(s_cfg["tp1_close_fraction"]),
                "be_offset_pips": float(s_cfg["be_offset_pips"]),
                "risk_usd_planned": float(risk_usd),
                "asian_range_pips": entry.get("asian_range_pips"),
                "lor_range_pips": entry.get("lor_range_pips"),
                "is_reentry": bool(entry.get("is_reentry", False)),
                "execute_time": nxt_ts.isoformat(),
                "grace_trail_distance_pips": float(s_cfg.get("grace_trail_distance_pips", 0.0) or 0.0),
                "extend_runner_until_ny_start_delay": bool(s_cfg.get("extend_runner_until_ny_start_delay", False)),
                "stop_loss_pips": float(sl_pips),
                "tp1_pips": float(tp1_pips),
                "tp2_pips": float(tp2_pips),
                "trade_sequence_number": int(state.daily_trade_sequence + 1),
            },
        )

    def get_exit_conditions(self, position: PositionSnapshot, current_bar: BarView, history: HistoricalDataView) -> ExitAction | None:
        plan = self._trade_plans.get(int(position.trade_id))
        if plan is None:
            return None
        ts = pd.Timestamp(current_bar.timestamp)
        px_open, px_high, px_low, px_close = self._px_for_position(position, current_bar)

        if ts >= plan.delayed_hard_close or (ts >= plan.hard_close and not (plan.setup_type == "D" and plan.extend_runner_until_ny_start_delay and plan.tp1_hit and ts < plan.delayed_hard_close)):
            exit_price = float(px_open)
            reason = "TP1_THEN_DELAYED_HARD_CLOSE" if plan.tp1_hit and ts >= plan.delayed_hard_close else ("TP1_ONLY_HARD_CLOSE" if plan.tp1_hit else "HARD_CLOSE")
            return ExitAction(reason=reason, exit_type="full", close_fraction=1.0, price=exit_price)

        stop_price = float(position.stop_loss)
        if not plan.tp1_hit:
            tp1_hit = px_high >= plan.tp1_price if position.direction == "long" else px_low <= plan.tp1_price
            sl_hit = px_low <= stop_price if position.direction == "long" else px_high >= stop_price
            if tp1_hit and sl_hit:
                first = self._choose_first(float(px_open), plan.tp1_price, stop_price)
            elif sl_hit:
                first = "b"
            elif tp1_hit:
                first = "a"
            else:
                first = ""
            if first == "b":
                return ExitAction(reason="SL_FULL", exit_type="full", close_fraction=1.0, price=stop_price)
            if first == "a":
                plan.tp1_hit = True
                return ExitAction(
                    reason="tp1_partial",
                    exit_type="partial",
                    close_fraction=float(plan.tp1_close_fraction),
                    price=float(plan.tp1_price),
                    new_stop_loss=float(plan.be_price_after_tp1),
                )
            return None

        tp2_hit = px_high >= plan.tp2_price if position.direction == "long" else px_low <= plan.tp2_price
        sl2_hit = px_low <= stop_price if position.direction == "long" else px_high >= stop_price
        if tp2_hit and sl2_hit:
            first2 = self._choose_first(float(px_open), plan.tp2_price, stop_price)
        elif tp2_hit:
            first2 = "a"
        elif sl2_hit:
            first2 = "b"
        else:
            first2 = ""
        if first2 == "a":
            return ExitAction(reason="TP2_FULL", exit_type="full", close_fraction=1.0, price=float(plan.tp2_price))
        if first2 == "b":
            return ExitAction(reason="BE_STOP", exit_type="full", close_fraction=1.0, price=stop_price)
        if (
            plan.setup_type == "D"
            and plan.extend_runner_until_ny_start_delay
            and ts >= plan.hard_close
            and ts < plan.delayed_hard_close
            and plan.grace_trail_distance_pips > 0
        ):
            if position.direction == "long":
                new_stop = max(stop_price, float(current_bar.bid_close) - plan.grace_trail_distance_pips * PIP_SIZE)
            else:
                new_stop = min(stop_price, float(current_bar.ask_close) + plan.grace_trail_distance_pips * PIP_SIZE)
            if abs(new_stop - stop_price) > 1e-9:
                return ExitAction(reason="grace_trail_update", exit_type="none", new_stop_loss=float(new_stop))
        return None

    def on_position_opened(self, position: PositionSnapshot, signal: Signal, current_bar: BarView) -> None:
        if self._day_state is None:
            return
        setup = str(signal.metadata.get("setup_type", "A"))
        tp1_pips = float(signal.metadata.get("tp1_pips", 0.0))
        tp2_pips = float(signal.metadata.get("tp2_pips", 0.0))
        if position.direction == "long":
            tp1_price = float(position.entry_price) + tp1_pips * PIP_SIZE
            tp2_price = float(position.entry_price) + tp2_pips * PIP_SIZE
            be_price = float(position.entry_price) + float(signal.metadata.get("be_offset_pips", 0.0)) * PIP_SIZE
        else:
            tp1_price = float(position.entry_price) - tp1_pips * PIP_SIZE
            tp2_price = float(position.entry_price) - tp2_pips * PIP_SIZE
            be_price = float(position.entry_price) - float(signal.metadata.get("be_offset_pips", 0.0)) * PIP_SIZE
        self._trade_plans[int(position.trade_id)] = _TradePlan(
            setup_type=setup,
            direction=position.direction,
            tp1_price=float(tp1_price),
            tp2_price=float(tp2_price),
            tp1_close_fraction=float(signal.metadata.get("tp1_close_fraction", 0.5)),
            be_price_after_tp1=float(be_price),
            hard_close=self._day_state.hard_close,
            delayed_hard_close=self._day_state.delayed_hard_close,
            grace_trail_distance_pips=float(signal.metadata.get("grace_trail_distance_pips", 0.0)),
            extend_runner_until_ny_start_delay=bool(signal.metadata.get("extend_runner_until_ny_start_delay", False)),
            risk_usd_planned=float(signal.metadata.get("risk_usd_planned", 0.0)),
        )
        channel = self._day_state.channels[(setup, position.direction)]
        channel.state = "FIRED"
        channel.entries += 1
        self._day_state.day_entries_total += 1
        self._day_state.day_entries_setup[setup] += 1
        self._day_state.day_entries_setup_dir[(setup, position.direction)] += 1
        self._day_state.daily_trade_sequence += 1

    def on_position_closed(self, trade: ClosedTrade) -> None:
        if trade.event_type == "partial" and int(trade.remaining_units) > 0:
            return
        plan = self._trade_plans.pop(int(trade.trade_id), None)
        if plan is None or self._day_state is None:
            return
        channel = self._day_state.channels[(plan.setup_type, trade.direction)]
        if bool(self.cfg.get("entry_limits", {}).get("disable_channel_reset_after_exit", False)):
            channel.state = "FIRED"
            channel.cooldown_until = None
            return
        if plan.setup_type in {"B", "C"}:
            channel.state = "WAITING_RESET"
            channel.cooldown_until = pd.Timestamp(trade.exit_time) + pd.Timedelta(
                minutes=int(self.cfg["setups"][plan.setup_type]["reenter_cooldown_minutes"])
            )
        else:
            channel.state = "WAITING_RESET"
            channel.cooldown_until = None

    def _reset_day_if_needed(self, ts: pd.Timestamp) -> None:
        day = ts.normalize()
        if self._current_day is not None and day == self._current_day:
            return
        self._current_day = day
        london_h = uk_london_open_utc(day)
        ny_h = us_ny_open_utc(day)
        london_open = day + pd.Timedelta(hours=london_h)
        ny_open = day + pd.Timedelta(hours=ny_h)
        self._day_state = _DayState(
            day_start=day,
            london_open=london_open,
            ny_open=ny_open,
            hard_close=ny_open,
            delayed_hard_close=ny_open + pd.Timedelta(
                minutes=int(self.cfg["session"].get("tp1_runner_hard_close_delay_minutes_after_ny_open", 0) or 0)
            ),
        )

    def _windows_for_day(self, state: _DayState) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
        windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
        for setup in ("A", "B", "C", "D"):
            s_cfg = self.cfg["setups"][setup]
            start = state.london_open + pd.Timedelta(minutes=int(s_cfg["entry_start_min_after_london"]))
            if s_cfg.get("entry_end_min_before_ny") is not None:
                end = state.ny_open - pd.Timedelta(minutes=int(s_cfg["entry_end_min_before_ny"]))
            else:
                end = state.london_open + pd.Timedelta(minutes=int(s_cfg["entry_end_min_after_london"]))
            if end > state.ny_open:
                end = state.ny_open
            windows[setup] = (start, end)
        return windows

    def _update_ranges(self, current_bar: BarView, ts: pd.Timestamp) -> None:
        assert self._day_state is not None
        state = self._day_state
        if state.day_start <= ts < state.london_open:
            state.asian_high = max(float(state.asian_high), float(current_bar.mid_high))
            state.asian_low = min(float(state.asian_low), float(current_bar.mid_low))
            if state.asian_high != float("-inf") and state.asian_low != float("inf"):
                rng = (state.asian_high - state.asian_low) / PIP_SIZE
                state.asian_range_pips = float(rng)
                state.asian_valid = float(self.cfg["levels"]["asian_range_min_pips"]) <= rng <= float(self.cfg["levels"]["asian_range_max_pips"])
        lor_end = state.london_open + pd.Timedelta(minutes=15)
        if state.london_open <= ts < lor_end:
            state.lor_high = max(float(state.lor_high), float(current_bar.mid_high))
            state.lor_low = min(float(state.lor_low), float(current_bar.mid_low))
            if state.lor_high != float("-inf") and state.lor_low != float("inf"):
                rng = (state.lor_high - state.lor_low) / PIP_SIZE
                state.lor_range_pips = float(rng)
                state.lor_valid = float(self.cfg["levels"]["lor_range_min_pips"]) <= rng <= float(self.cfg["levels"]["lor_range_max_pips"])

    def _apply_channel_resets(self, current_bar: BarView, ts: pd.Timestamp) -> None:
        assert self._day_state is not None
        state = self._day_state
        windows = self._windows_for_day(state)
        for setup in ("A", "B", "C", "D"):
            _w_start, w_end = windows[setup]
            for direction in ("long", "short"):
                channel = state.channels[(setup, direction)]
                if channel.state != "WAITING_RESET":
                    continue
                if setup in {"B", "C"}:
                    if channel.cooldown_until is not None and ts >= channel.cooldown_until and ts < w_end:
                        channel.state = "ARMED"
                        channel.resets += 1
                    continue
                if ts >= w_end:
                    continue
                if setup == "A" and state.asian_high != float("-inf") and state.asian_low != float("inf"):
                    if direction == "long" and float(current_bar.mid_low) <= float(state.asian_high):
                        channel.state = "ARMED"
                        channel.resets += 1
                    if direction == "short" and float(current_bar.mid_high) >= float(state.asian_low):
                        channel.state = "ARMED"
                        channel.resets += 1
                if setup == "D" and state.lor_valid and state.lor_high != float("-inf") and state.lor_low != float("inf"):
                    if direction == "long" and float(current_bar.mid_low) <= float(state.lor_high):
                        channel.state = "ARMED"
                        channel.resets += 1
                    if direction == "short" and float(current_bar.mid_high) >= float(state.lor_low):
                        channel.state = "ARMED"
                        channel.resets += 1

    def _first_fillable_entry(
        self,
        entries: list[dict[str, Any]],
        nxt_ts: pd.Timestamp,
        state: _DayState,
        portfolio: PortfolioSnapshot,
    ) -> dict[str, Any] | None:
        limits = self.cfg.get("entry_limits", {})
        for entry in entries:
            setup = str(entry["setup_type"])
            direction = str(entry["direction"])
            w_start, w_end = self._windows_for_day(state)[setup]
            if not (w_start <= nxt_ts < w_end):
                continue
            lim_total = limits.get("max_trades_per_day_total")
            lim_setup = limits.get("max_trades_per_setup_per_day")
            lim_setup_dir = limits.get("max_trades_per_setup_direction_per_day")
            if lim_total is not None and state.day_entries_total >= int(lim_total):
                continue
            if lim_setup is not None and state.day_entries_setup[setup] >= int(lim_setup):
                continue
            if lim_setup_dir is not None and state.day_entries_setup_dir[(setup, direction)] >= int(lim_setup_dir):
                continue
            if len(portfolio.open_positions) >= int(self.cfg["account"]["max_open_positions"]):
                continue
            return entry
        return None

    def _px_for_position(self, position: PositionSnapshot, current_bar: BarView) -> tuple[float, float, float, float]:
        if position.direction == "long":
            return float(current_bar.bid_open), float(current_bar.bid_high), float(current_bar.bid_low), float(current_bar.bid_close)
        return float(current_bar.ask_open), float(current_bar.ask_high), float(current_bar.ask_low), float(current_bar.ask_close)

    def _choose_first(self, px_open: float, level_a: float, level_b: float) -> str:
        da = abs(px_open - level_a)
        db = abs(px_open - level_b)
        return "a" if da <= db else "b"
