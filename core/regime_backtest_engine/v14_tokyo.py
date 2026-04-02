from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from core.v14_entry_evaluator import evaluate_v14_entry_signal
from scripts.backtest_tokyo_meanrev import PIP_SIZE, TOKYO_TZ, add_indicators, load_m1, load_news_events

from .models import ClosedTrade, ExitAction, PortfolioSnapshot, PositionSnapshot, Signal
from .strategy import BarView, HistoricalDataView, TrainableStrategyFamily


class V14TokyoStrategyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    family_name: str = "v14"
    config: dict[str, Any]

    @classmethod
    def from_json(cls, path: str | Path, *, family_name: str = "v14") -> "V14TokyoStrategyConfig":
        raw = json.loads(Path(path).read_text())
        return cls(family_name=family_name, config=raw)


@dataclass
class _PendingSignal:
    payload: dict[str, Any]


@dataclass
class _TradePlan:
    direction: str
    entry_session_day: str
    entry_time: pd.Timestamp
    tp1_price: float
    tp2_price: float
    tp1_close_fraction: float
    be_offset_pips: float
    trail_activate_pips: float
    trail_distance_pips: float
    trail_requires_tp1: bool
    time_decay_minutes: int
    time_decay_profit_cap_pips: float
    tp1_hit: bool = False
    moved_to_breakeven: bool = False
    trail_active: bool = False
    trail_stop_price: float | None = None
    max_profit_seen_pips: float = 0.0
    max_adverse_seen_pips: float = 0.0


def prepare_v14_augmented_data(input_path: str | Path, config_path: str | Path, output_path: str | Path) -> Path:
    cfg = json.loads(Path(config_path).read_text())
    df = load_m1(str(input_path))
    df = add_indicators(df, cfg)
    df["time_utc"] = pd.to_datetime(df["time"], utc=True)
    df["time_jst"] = df["time_utc"].dt.tz_convert(TOKYO_TZ)
    df["weekday_jst"] = df["time_jst"].dt.weekday
    df["day_name_jst"] = df["time_jst"].dt.day_name()
    df["session_day_jst"] = df["time_jst"].dt.date.astype(str)
    df["utc_day_name"] = df["time_utc"].dt.day_name()
    df["hour_utc"] = df["time_utc"].dt.hour
    df["minute_utc"] = df["time_utc"].dt.minute
    df["minute_of_day_utc"] = df["hour_utc"] * 60 + df["minute_utc"]

    sf = cfg.get("session_filter", {})
    session_start_utc = str(sf.get("session_start_utc", "15:00"))
    session_end_utc = str(sf.get("session_end_utc", "00:00"))
    allowed_days_set = set(sf.get("allowed_trading_days", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]))

    def hhmm_to_minutes(value: str) -> int:
        hh, mm = value.strip().split(":")
        return int(hh) * 60 + int(mm)

    start_min = hhmm_to_minutes(session_start_utc)
    end_min = hhmm_to_minutes(session_end_utc)
    if start_min < end_min:
        df["in_tokyo_session"] = (df["minute_of_day_utc"] >= start_min) & (df["minute_of_day_utc"] < end_min)
    else:
        df["in_tokyo_session"] = (df["minute_of_day_utc"] >= start_min) | (df["minute_of_day_utc"] < end_min)
    df["allowed_trading_day"] = df["utc_day_name"].isin(allowed_days_set)
    df["news_blocked"] = False
    if bool(cfg.get("news_filter", {}).get("enabled", False)):
        ev = load_news_events(cfg)
        if not ev.empty:
            block_mask = np.zeros(len(df), dtype=bool)
            tcol = df["time_utc"]
            news_cfg = cfg.get("news_filter", {})
            mode = str(news_cfg.get("mode", "tiered")).strip().lower()
            all_pre = int(news_cfg.get("all_pre_block_minutes", 15))
            all_post = int(news_cfg.get("all_post_block_minutes", 15))
            high_pre = int(news_cfg.get("high_impact_pre_block_minutes", 60))
            high_post = int(news_cfg.get("high_impact_post_block_minutes", 60))
            med_pre = int(news_cfg.get("medium_impact_pre_block_minutes", 15))
            med_post = int(news_cfg.get("medium_impact_post_block_minutes", 15))
            low_pre = int(news_cfg.get("low_impact_pre_block_minutes", med_pre))
            low_post = int(news_cfg.get("low_impact_post_block_minutes", med_post))
            impacts = {str(x).lower() for x in news_cfg.get("block_impacts", ["high", "medium", "low"])}
            for _, er in ev.iterrows():
                impact = str(er.get("impact", "")).lower()
                if impact not in impacts:
                    continue
                if mode == "all":
                    pre_min, post_min = all_pre, all_post
                elif impact == "high":
                    pre_min, post_min = high_pre, high_post
                elif impact == "medium":
                    pre_min, post_min = med_pre, med_post
                else:
                    pre_min, post_min = low_pre, low_post
                evt = pd.Timestamp(er["event_ts"])
                ws = evt - pd.Timedelta(minutes=max(0, int(pre_min)))
                we = evt + pd.Timedelta(minutes=max(0, int(post_min)))
                block_mask |= ((tcol >= ws) & (tcol <= we)).to_numpy()
            df["news_blocked"] = block_mask

    bool_cols = df.select_dtypes(include=["bool"]).columns
    for col in bool_cols:
        df[col] = df[col].astype(int)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


class V14TokyoStrategy(TrainableStrategyFamily):
    family_name = "v14"

    def __init__(self, config: V14TokyoStrategyConfig) -> None:
        self.config = config
        self.family_name = config.family_name
        self.cfg = config.config
        self._pending_signals: list[dict[str, Any]] = []
        self._trade_plans: dict[int, _TradePlan] = {}
        self._session_state: dict[str, dict[str, Any]] = {}
        self._peak_equity: float = 0.0

        sf = self.cfg.get("session_filter", {})
        self._start_min = self._hhmm_to_minutes(str(sf.get("session_start_utc", "15:00")))
        self._end_min = self._hhmm_to_minutes(str(sf.get("session_end_utc", "00:00")))
        self._entry_start_min = self._hhmm_to_minutes(str(sf.get("entry_start_utc", sf.get("session_start_utc", "15:00"))))
        self._entry_end_min = self._hhmm_to_minutes(str(sf.get("entry_end_utc", sf.get("session_end_utc", "00:00"))))
        self._block_new_entries_minutes_before_end = int(sf.get("block_new_entries_minutes_before_end", sf.get("session_entry_cutoff_minutes", 0)))
        self._lunch_block_enabled = bool(sf.get("lunch_block_enabled", False))
        self._lunch_start_min = self._hhmm_to_minutes(str(sf.get("lunch_block_start_utc", "02:30")))
        self._lunch_end_min = self._hhmm_to_minutes(str(sf.get("lunch_block_end_utc", "03:30")))

        tm = self.cfg.get("trade_management", {})
        self._max_trades_session = int(tm.get("max_trades_per_session", 4))
        self._min_entry_gap_min = int(tm.get("min_time_between_entries_minutes", 10))
        self._no_reentry_stop_min = int(tm.get("no_reentry_same_direction_after_stop_minutes", 30))
        self._session_loss_stop_pct = float(tm.get("session_loss_stop_pct", 0.0)) / 100.0
        self._stop_after_consecutive_losses = int(tm.get("stop_after_consecutive_losses", 3))
        self._breakout_disable_pips = float(tm.get("disable_entries_if_move_from_tokyo_open_range_exceeds_pips", 30.0))
        self._breakout_mode = str(tm.get("breakout_detection_mode", "rolling")).strip().lower()
        self._rolling_window_minutes = int(tm.get("rolling_window_minutes", 60))
        self._rolling_range_threshold_pips = float(tm.get("rolling_range_threshold_pips", 40.0))
        self._breakout_cooldown_minutes = int(tm.get("cooldown_minutes", 15))

        self._atr_max = float(self.cfg["indicators"]["atr"]["max_threshold_price_units"])
        self._atr_gate_enabled = bool(self.cfg["indicators"]["atr"].get("use_as_hard_gate", True))
        self._adx_filter_enabled = bool(self.cfg.get("adx_filter", {}).get("enabled", False))
        self._adx_max_for_entry = float(self.cfg.get("adx_filter", {}).get("max_adx_for_entry", 35.0))
        self._entry_confirmation_enabled = bool(self.cfg.get("entry_confirmation", {}).get("enabled", True))
        self._confirmation_type = str(self.cfg.get("entry_confirmation", {}).get("type", "m1")).strip().lower()
        if self._confirmation_type not in {"m1", "m5"}:
            self._confirmation_type = "m1"
        self._confirmation_window_bars = int(self.cfg.get("entry_confirmation", {}).get("window_bars", 5)) if self._entry_confirmation_enabled else 0
        self._tokyo_v2_scoring = str(self.cfg.get("entry_rules", {}).get("scoring_model", "")).strip().lower() in {"tokyo_v2", "v2", "tokyo_actual_v2"}

        self._risk_pct = float(self.cfg["position_sizing"]["risk_per_trade_pct"]) / 100.0
        self._max_units = int(self.cfg["position_sizing"]["max_units"])
        self._max_open = int(self.cfg["position_sizing"]["max_concurrent_positions"])
        self._day_risk_multipliers = {str(k): float(v) for k, v in self.cfg.get("position_sizing", {}).get("day_risk_multipliers", {}).items()}

        self._min_tp_pips = 8.0
        self._min_sl_pips = float(self.cfg["exit_rules"]["stop_loss"].get("minimum_sl_pips", 10.0))
        self._max_sl_pips = float(self.cfg["exit_rules"]["stop_loss"].get("hard_max_sl_pips", 28.0))
        self._sl_buf = float(self.cfg["exit_rules"]["stop_loss"].get("buffer_pips", 8.0)) * PIP_SIZE
        self._trail_activate_pips = float(self.cfg["exit_rules"]["trailing_stop"].get("activate_after_profit_pips", 10.0))
        self._trail_dist_pips = float(self.cfg["exit_rules"]["trailing_stop"].get("trail_distance_pips", 8.0))
        self._trail_enabled = bool(self.cfg["exit_rules"]["trailing_stop"].get("enabled", True))
        self._trail_requires_tp1 = bool(self.cfg["exit_rules"]["trailing_stop"].get("requires_tp1_hit", True))
        self._partial_close_pct = float(self.cfg["exit_rules"]["take_profit"].get("partial_close_pct", 0.5))
        self._partial_tp_min_pips = float(self.cfg["exit_rules"]["take_profit"].get("partial_tp_min_pips", 6.0))
        self._partial_tp_max_pips = float(self.cfg["exit_rules"]["take_profit"].get("partial_tp_max_pips", 12.0))
        self._partial_tp_atr_mult = float(self.cfg["exit_rules"]["take_profit"].get("partial_tp_atr_mult", 0.5))
        self._tp_mode = str(self.cfg["exit_rules"]["take_profit"].get("mode", "partial")).strip().lower()
        self._single_tp_atr_mult = float(self.cfg["exit_rules"]["take_profit"].get("single_tp_atr_mult", 1.0))
        self._single_tp_min_pips = float(self.cfg["exit_rules"]["take_profit"].get("single_tp_min_pips", self._min_tp_pips))
        self._single_tp_max_pips = float(self.cfg["exit_rules"]["take_profit"].get("single_tp_max_pips", 40.0))
        self._breakeven_offset_pips = float(self.cfg["exit_rules"]["take_profit"].get("breakeven_offset_pips", 1.0))
        self._time_decay_minutes = int(self.cfg["exit_rules"]["time_exit"].get("time_decay_minutes", 120))
        self._time_decay_profit_cap_pips = float(self.cfg["exit_rules"]["time_exit"].get("time_decay_profit_cap_pips", 3.0))

        self._tol = float(self.cfg["entry_rules"]["long"].get("price_zone", {}).get("tolerance_pips", 10.0)) * PIP_SIZE
        self._confluence_min_long = int(self.cfg["entry_rules"]["long"].get("confluence_scoring", {}).get("minimum_score", 2))
        self._confluence_min_short = int(self.cfg["entry_rules"]["short"].get("confluence_scoring", {}).get("minimum_score", 2))
        self._long_rsi_soft_entry = float(self.cfg["entry_rules"]["long"].get("rsi_soft_filter", {}).get("entry_soft_threshold", 35.0))
        self._long_rsi_bonus = float(self.cfg["entry_rules"]["long"].get("rsi_soft_filter", {}).get("bonus_threshold", 30.0))
        self._short_rsi_soft_entry = float(self.cfg["entry_rules"]["short"].get("rsi_soft_filter", {}).get("entry_soft_threshold", 65.0))
        self._short_rsi_bonus = float(self.cfg["entry_rules"]["short"].get("rsi_soft_filter", {}).get("bonus_threshold", 70.0))
        core_gate_cfg = self.cfg.get("entry_rules", {}).get("core_gate", {})
        self._core_gate_required = int(core_gate_cfg.get("required_count", 4))
        self._core_gate_use_zone = bool(core_gate_cfg.get("use_zone", True))
        self._core_gate_use_bb = bool(core_gate_cfg.get("use_bb_touch", True))
        self._core_gate_use_sar = bool(core_gate_cfg.get("use_sar_flip", True))
        self._core_gate_use_rsi = bool(core_gate_cfg.get("use_rsi_soft", True))
        self._regime_filter_mode = str(self.cfg["indicators"].get("bb_width_regime_filter", {}).get("mode", "percentile")).strip().lower()
        regime_gate = self.cfg.get("regime_gate", {})
        self._regime_enabled = bool(regime_gate.get("enabled", False))
        self._atr_ratio_trend = float(regime_gate.get("atr_ratio_trending_threshold", 1.3))
        self._atr_ratio_calm = float(regime_gate.get("atr_ratio_calm_threshold", 0.8))
        self._adx_trend = float(regime_gate.get("adx_trending_threshold", 25.0))
        self._adx_range = float(regime_gate.get("adx_ranging_threshold", 20.0))
        self._favorable_min_score = int(regime_gate.get("favorable_min_score", 1))
        self._neutral_min_score = int(regime_gate.get("neutral_min_score", 0))
        self._neutral_size_mult = float(regime_gate.get("neutral_size_multiplier", 0.5))

        cq = self.cfg.get("confluence_quality", {})
        self._cq_enabled = bool(cq.get("enabled", False))
        self._top_combos = set(cq.get("top_combos", []))
        self._bottom_combos = set(cq.get("bottom_combos", []))
        self._high_quality_mult = float(cq.get("high_quality_size_mult", 1.0))
        self._medium_quality_mult = float(cq.get("medium_quality_size_mult", 0.75))
        self._low_quality_skip = bool(cq.get("low_quality_skip", True))

    def fit(self, history: HistoricalDataView) -> None:
        return None

    def evaluate(self, current_bar: BarView, history: HistoricalDataView, portfolio: PortfolioSnapshot) -> Signal | None:
        ts = pd.Timestamp(current_bar.timestamp)
        self._peak_equity = max(float(self._peak_equity), float(portfolio.equity))
        i = int(current_bar.bar_index)

        for sig in list(self._pending_signals):
            if i > int(sig["expiry_index"]):
                self._pending_signals.remove(sig)

        if not self._truthy(current_bar.in_tokyo_session) or not self._truthy(current_bar.allowed_trading_day):
            return None

        session_day = str(current_bar.session_day_jst)
        sst = self._ensure_session(session_day, current_bar, portfolio.equity)
        minute_now = int(current_bar.minute_of_day_utc)
        if self._lunch_block_enabled and self._in_window(minute_now, self._lunch_start_min, self._lunch_end_min):
            return None

        sst["session_high"] = max(float(sst.get("session_high", float(current_bar.mid_high))), float(current_bar.mid_high))
        sst["session_low"] = min(float(sst.get("session_low", float(current_bar.mid_low))), float(current_bar.mid_low))

        if self._stop_after_consecutive_losses > 0 and int(sst.get("consec_losses", 0)) >= self._stop_after_consecutive_losses:
            sst["stopped"] = True
            return None
        if self._session_loss_stop_pct > 0 and float(sst.get("session_pnl_usd", 0.0)) <= (-self._session_loss_stop_pct * float(sst.get("session_start_equity", portfolio.equity))):
            sst["stopped"] = True
            return None

        if self._missing_indicators(current_bar):
            return None

        if not self._passes_breakout_gate(current_bar, ts, sst):
            return None
        regime_col = "bb_regime_expanding3" if self._regime_filter_mode in {"expanding3", "bb_width_expanding3"} else "bb_regime"
        if str(getattr(current_bar, regime_col)) != "ranging":
            return None
        if self._atr_gate_enabled and float(current_bar.atr_m15) > self._atr_max:
            return None
        if self._adx_filter_enabled:
            adx_now = float(current_bar.adx_m15)
            if not np.isfinite(adx_now) or adx_now > self._adx_max_for_entry:
                return None
        if self._truthy(getattr(current_bar, "news_blocked", 0)):
            return None

        candidate = evaluate_v14_entry_signal(
            row=self._row_dict(current_bar),
            mid_close=float(current_bar.mid_close),
            mid_open=float(current_bar.mid_open),
            mid_high=float(current_bar.mid_high),
            mid_low=float(current_bar.mid_low),
            pivot_levels={
                "P": float(current_bar.pivot_P),
                "R1": float(current_bar.pivot_R1),
                "R2": float(current_bar.pivot_R2),
                "R3": float(current_bar.pivot_R3),
                "S1": float(current_bar.pivot_S1),
                "S2": float(current_bar.pivot_S2),
                "S3": float(current_bar.pivot_S3),
            },
            cfg_params=self._cfg_params(),
            sst=sst,
        )
        if candidate is not None and candidate.get("_blocked_reason") is None:
            direction = str(candidate["direction"])
            sig_obj = {
                "direction": direction,
                "session_day": session_day,
                "signal_index": i,
                "signal_time": ts,
                "expiry_index": int(i + (self._confirmation_window_bars if self._confirmation_type == "m1" else self._confirmation_window_bars * 5)),
                "confluence_score": int(candidate["confluence_score"]),
                "confluence_combo": str(candidate["confluence_combo"]),
                "signal_strength_score": int(candidate["signal_strength_score"]),
                "signal_strength_tier": str(candidate["signal_strength_tier"]),
                "quality_label": str(candidate["quality_label"]),
                "quality_mult": float(candidate["quality_mult"]),
                "regime_label": str(candidate["regime_label"]),
                "regime_mult": float(candidate["regime_mult"]),
                "from_zone": bool(candidate["from_zone"]),
                "P": float(candidate["P"]),
                "R1": float(candidate["R1"]),
                "R2": float(candidate["R2"]),
                "R3": float(candidate["R3"]),
                "S1": float(candidate["S1"]),
                "S2": float(candidate["S2"]),
                "S3": float(candidate["S3"]),
                "bb_upper": float(candidate["bb_upper"]),
                "bb_lower": float(candidate["bb_lower"]),
                "bb_mid": float(candidate["bb_mid"]),
                "sar_value": float(candidate["sar_value"]),
                "sar_direction": str(candidate["sar_direction"]),
                "rsi_m5": float(candidate["rsi_m5"]),
                "atr_m15": float(candidate["atr_m15"]),
                "entry_delay_type": "immediate",
                "confirmation_delay_candles": 0,
                "confirmation_close": float(current_bar.mid_close),
                "rejection_confirmed": bool(candidate["rejection_confirmed"]),
                "rejection_low": candidate["rejection_low"],
                "rejection_high": candidate["rejection_high"],
                "divergence_present": bool(candidate["divergence_present"]),
                "inside_ir": bool(candidate["inside_ir"]),
                "quality_markers": str(candidate["quality_markers"]),
                "session_midpoint": candidate["session_midpoint"],
                "distance_to_ir_boundary_pips": candidate["distance_to_ir_boundary_pips"],
                "distance_to_midpoint_pips": candidate["distance_to_midpoint_pips"],
                "distance_to_pivot_pips": float(candidate["distance_to_pivot_pips"]),
            }
            if self._entry_confirmation_enabled and self._confirmation_window_bars > 0:
                has_pending_same = any(
                    ps["session_day"] == session_day and ps["direction"] == direction and i <= int(ps["expiry_index"])
                    for ps in self._pending_signals
                )
                if not has_pending_same:
                    self._pending_signals.append(sig_obj)
            else:
                signal = self._build_signal(sig_obj, current_bar, portfolio, sst)
                if signal is not None:
                    return signal

        for sig in list(self._pending_signals):
            if sig["session_day"] != session_day:
                continue
            if i <= int(sig["signal_index"]):
                continue
            if i > int(sig["expiry_index"]):
                self._pending_signals.remove(sig)
                continue
            sdir = str(sig["direction"])
            if self._confirmation_type == "m5":
                if int(current_bar.minute_utc) % 5 != 0:
                    continue
                m5_open = getattr(current_bar, "m5_open", np.nan)
                m5_close = getattr(current_bar, "m5_close", np.nan)
                if pd.isna(m5_open) or pd.isna(m5_close):
                    continue
                confirmed = (float(m5_close) > float(m5_open)) if sdir == "long" else (float(m5_close) < float(m5_open))
            else:
                confirmed = (float(current_bar.mid_close) > float(current_bar.mid_open)) if sdir == "long" else (float(current_bar.mid_close) < float(current_bar.mid_open))
            if not confirmed:
                continue
            sig["confirmation_delay_candles"] = int(i - int(sig["signal_index"]))
            sig["confirmation_close"] = float(current_bar.mid_close)
            signal = self._build_signal(sig, current_bar, portfolio, sst)
            self._pending_signals.remove(sig)
            if signal is not None:
                return signal
        return None

    def get_exit_conditions(self, position: PositionSnapshot, current_bar: BarView, history: HistoricalDataView) -> ExitAction | None:
        plan = self._trade_plans.get(int(position.trade_id))
        if plan is None:
            return None
        in_session = self._truthy(getattr(current_bar, "in_tokyo_session", 0)) and self._truthy(getattr(current_bar, "allowed_trading_day", 0))
        if position.direction == "long":
            current_pips = (float(current_bar.bid_close) - float(position.entry_price)) / PIP_SIZE
            bar_fav = (float(current_bar.bid_high) - float(position.entry_price)) / PIP_SIZE
            bar_adv = (float(position.entry_price) - float(current_bar.bid_low)) / PIP_SIZE
        else:
            current_pips = (float(position.entry_price) - float(current_bar.ask_close)) / PIP_SIZE
            bar_fav = (float(position.entry_price) - float(current_bar.ask_low)) / PIP_SIZE
            bar_adv = (float(current_bar.ask_high) - float(position.entry_price)) / PIP_SIZE
        plan.max_profit_seen_pips = max(float(plan.max_profit_seen_pips), float(bar_fav))
        plan.max_adverse_seen_pips = max(float(plan.max_adverse_seen_pips), float(bar_adv))
        can_trail = (not plan.trail_requires_tp1) or plan.tp1_hit
        if self._trail_enabled and can_trail and bar_fav >= plan.trail_activate_pips:
            plan.trail_active = True
        if plan.trail_active:
            if position.direction == "long":
                new_trail = float(current_bar.bid_close) - plan.trail_distance_pips * PIP_SIZE
                plan.trail_stop_price = new_trail if plan.trail_stop_price is None else max(float(plan.trail_stop_price), new_trail)
            else:
                new_trail = float(current_bar.ask_close) + plan.trail_distance_pips * PIP_SIZE
                plan.trail_stop_price = new_trail if plan.trail_stop_price is None else min(float(plan.trail_stop_price), new_trail)

        if not in_session:
            reason = "tp1_then_session_close" if plan.tp1_hit else "session_close"
            exit_px = float(current_bar.bid_close) if position.direction == "long" else float(current_bar.ask_close)
            return ExitAction(reason=reason, exit_type="full", close_fraction=1.0, price=exit_px)

        ts = pd.Timestamp(current_bar.timestamp)
        held_minutes = (ts - pd.Timestamp(plan.entry_time)).total_seconds() / 60.0
        stop_price = float(position.stop_loss)
        hit_sl = (float(current_bar.bid_low) <= stop_price) if position.direction == "long" else (float(current_bar.ask_high) >= stop_price)
        if hit_sl:
            be_px = float(position.entry_price) + plan.be_offset_pips * PIP_SIZE if position.direction == "long" else float(position.entry_price) - plan.be_offset_pips * PIP_SIZE
            if plan.tp1_hit and plan.moved_to_breakeven and abs(stop_price - be_px) <= 1e-9:
                reason = "tp1_then_be_stop"
            else:
                reason = "tp1_then_sl" if plan.tp1_hit else "sl"
            return ExitAction(reason=reason, exit_type="full", close_fraction=1.0, price=stop_price)

        hit_tp1 = (not plan.tp1_hit) and ((float(current_bar.bid_high) >= plan.tp1_price) if position.direction == "long" else (float(current_bar.ask_low) <= plan.tp1_price))
        if hit_tp1:
            plan.tp1_hit = True
            plan.moved_to_breakeven = True
            be_stop = float(position.entry_price) + plan.be_offset_pips * PIP_SIZE if position.direction == "long" else float(position.entry_price) - plan.be_offset_pips * PIP_SIZE
            return ExitAction(
                reason="tp1_partial",
                exit_type="partial",
                close_fraction=float(plan.tp1_close_fraction),
                price=float(plan.tp1_price),
                new_stop_loss=float(be_stop),
            )

        if plan.tp1_hit:
            hit_tp2 = (float(current_bar.bid_high) >= plan.tp2_price) if position.direction == "long" else (float(current_bar.ask_low) <= plan.tp2_price)
            if hit_tp2:
                return ExitAction(reason="tp1_then_tp2", exit_type="full", close_fraction=1.0, price=float(plan.tp2_price))
            hit_trail = plan.trail_active and plan.trail_stop_price is not None and (((float(current_bar.bid_low) <= float(plan.trail_stop_price)) if position.direction == "long" else (float(current_bar.ask_high) >= float(plan.trail_stop_price))))
            if hit_trail:
                return ExitAction(reason="tp1_then_trailing_stop", exit_type="full", close_fraction=1.0, price=float(plan.trail_stop_price))
            if held_minutes >= plan.time_decay_minutes and (0.0 <= current_pips < plan.time_decay_profit_cap_pips):
                exit_px = float(current_bar.bid_close) if position.direction == "long" else float(current_bar.ask_close)
                return ExitAction(reason="tp1_then_time_decay", exit_type="full", close_fraction=1.0, price=exit_px)
        if plan.trail_active and plan.trail_stop_price is not None and abs(float(plan.trail_stop_price) - stop_price) > 1e-9:
            return ExitAction(reason="trail_update", exit_type="none", new_stop_loss=float(plan.trail_stop_price))
        return None

    def on_position_opened(self, position: PositionSnapshot, signal: Signal, current_bar: BarView) -> None:
        meta = signal.metadata
        tp1_price, tp2_price, tp1_close_pct = self._compute_targets_from_fill(position, meta)
        self._trade_plans[int(position.trade_id)] = _TradePlan(
            direction=position.direction,
            entry_session_day=str(meta.get("entry_session_day")),
            entry_time=pd.Timestamp(current_bar.timestamp),
            tp1_price=float(tp1_price),
            tp2_price=float(tp2_price),
            tp1_close_fraction=float(tp1_close_pct),
            be_offset_pips=float(meta.get("breakeven_offset_pips", self._breakeven_offset_pips)),
            trail_activate_pips=float(meta.get("trail_activate_pips", self._trail_activate_pips)),
            trail_distance_pips=float(meta.get("trail_distance_pips", self._trail_dist_pips)),
            trail_requires_tp1=bool(meta.get("trail_requires_tp1", self._trail_requires_tp1)),
            time_decay_minutes=int(meta.get("time_decay_minutes", self._time_decay_minutes)),
            time_decay_profit_cap_pips=float(meta.get("time_decay_profit_cap_pips", self._time_decay_profit_cap_pips)),
        )
        sst = self._session_state.setdefault(str(meta.get("entry_session_day")), {})
        sst["trades"] = int(sst.get("trades", 0)) + 1
        sst["last_entry_time"] = pd.Timestamp(current_bar.timestamp)
        sst.setdefault("entry_confluence_scores", []).append(int(meta.get("confluence_score", 0)))

    def on_position_closed(self, trade: ClosedTrade) -> None:
        if trade.event_type == "partial" and int(trade.remaining_units) > 0:
            return
        plan = self._trade_plans.pop(int(trade.trade_id), None)
        if plan is None:
            return
        sst = self._session_state.setdefault(plan.entry_session_day, {})
        equity_before = float(sst.get("last_known_equity", 0.0)) or float(sst.get("session_start_equity", 0.0))
        sst["session_pnl_usd"] = float(sst.get("session_pnl_usd", 0.0)) + float(trade.pnl_usd)
        sess_start_eq = float(sst.get("session_start_equity", equity_before or 0.0))
        if self._session_loss_stop_pct > 0 and sess_start_eq > 0 and float(sst["session_pnl_usd"]) <= (-self._session_loss_stop_pct * sess_start_eq):
            sst["stopped"] = True
        win = float(trade.pnl_usd) > 0
        if win:
            sst["consec_losses"] = 0
            sst["wins"] = int(sst.get("wins", 0)) + 1
        else:
            sst["consec_losses"] = int(sst.get("consec_losses", 0)) + 1
            sst["losses"] = int(sst.get("losses", 0)) + 1
            if self._stop_after_consecutive_losses > 0 and int(sst["consec_losses"]) >= self._stop_after_consecutive_losses:
                sst["stopped"] = True
            if "sl" in str(trade.exit_reason):
                if trade.direction == "long":
                    sst["last_stop_time_long"] = pd.Timestamp(trade.exit_time)
                else:
                    sst["last_stop_time_short"] = pd.Timestamp(trade.exit_time)

    def _cfg_params(self) -> dict[str, Any]:
        return {
            "tokyo_v2_scoring": self._tokyo_v2_scoring,
            "confluence_min_long": self._confluence_min_long,
            "confluence_min_short": self._confluence_min_short,
            "long_rsi_soft_entry": self._long_rsi_soft_entry,
            "long_rsi_bonus": self._long_rsi_bonus,
            "short_rsi_soft_entry": self._short_rsi_soft_entry,
            "short_rsi_bonus": self._short_rsi_bonus,
            "tol": self._tol,
            "atr_max": self._atr_max,
            "core_gate_use_zone": self._core_gate_use_zone,
            "core_gate_use_bb": self._core_gate_use_bb,
            "core_gate_use_sar": self._core_gate_use_sar,
            "core_gate_use_rsi": self._core_gate_use_rsi,
            "core_gate_required": self._core_gate_required,
            "regime_enabled": self._regime_enabled,
            "atr_ratio_trend": self._atr_ratio_trend,
            "atr_ratio_calm": self._atr_ratio_calm,
            "adx_trend": self._adx_trend,
            "adx_range": self._adx_range,
            "favorable_min_score": self._favorable_min_score,
            "neutral_min_score": self._neutral_min_score,
            "neutral_size_mult": self._neutral_size_mult,
            "ss_enabled": bool(self.cfg.get("signal_strength_tracking", {}).get("enabled", False)),
            "ss_comp": self.cfg.get("signal_strength_tracking", {}).get("components", {}),
            "combo_filter_enabled": bool(self.cfg.get("confluence_combo_filter", {}).get("enabled", False)),
            "combo_filter_mode": str(self.cfg.get("confluence_combo_filter", {}).get("mode", "allowlist")),
            "combo_allow": set(self.cfg.get("confluence_combo_filter", {}).get("allowed_combos", [])),
            "combo_block": set(self.cfg.get("confluence_combo_filter", {}).get("blocked_combos", [])),
            "ss_filter_enabled": bool(self.cfg.get("signal_strength_filter", {}).get("enabled", False)),
            "ss_filter_min_score": int(self.cfg.get("signal_strength_filter", {}).get("min_score", 0)),
            "cq_enabled": self._cq_enabled,
            "top_combos": self._top_combos,
            "bottom_combos": self._bottom_combos,
            "high_quality_mult": self._high_quality_mult,
            "medium_quality_mult": self._medium_quality_mult,
            "low_quality_skip": self._low_quality_skip,
            "rejection_bonus_enabled": bool(self.cfg.get("rejection_bonus", {}).get("enabled", False)),
            "div_track_enabled": bool(self.cfg.get("rsi_divergence_tracking", {}).get("enabled", False)),
            "session_env_enabled": bool(self.cfg.get("session_envelope", {}).get("enabled", False)),
            "session_env_log_ir_pos": bool(self.cfg.get("session_envelope", {}).get("log_ir_position", True)),
        }

    def _build_signal(self, sig: dict[str, Any], current_bar: BarView, portfolio: PortfolioSnapshot, sst: dict[str, Any]) -> Signal | None:
        ts = pd.Timestamp(current_bar.timestamp)
        if bool(sst.get("stopped", False)):
            return None
        minute_now = int(current_bar.minute_of_day_utc)
        entry_window_ok = self._in_window(minute_now, self._entry_start_min, self._entry_end_min)
        if self._block_new_entries_minutes_before_end > 0:
            mins_to_end = self._minutes_to_session_end(minute_now)
            if mins_to_end <= self._block_new_entries_minutes_before_end:
                entry_window_ok = False
        if not entry_window_ok:
            return None
        if self._stop_after_consecutive_losses > 0 and int(sst.get("consec_losses", 0)) >= self._stop_after_consecutive_losses:
            sst["stopped"] = True
            return None
        if self._session_loss_stop_pct > 0 and float(sst.get("session_pnl_usd", 0.0)) <= (-self._session_loss_stop_pct * float(sst.get("session_start_equity", portfolio.equity))):
            sst["stopped"] = True
            return None
        if self._max_trades_session > 0 and int(sst.get("trades", 0)) >= self._max_trades_session:
            return None
        if len(portfolio.open_positions) >= self._max_open:
            return None
        if float(current_bar.spread_pips) > float(self.cfg.get("execution_model", {}).get("max_entry_spread_pips", 10.0)):
            return None
        if sst.get("last_entry_time") is not None and (ts - pd.Timestamp(sst["last_entry_time"])).total_seconds() < self._min_entry_gap_min * 60.0:
            return None
        last_stop_key = "last_stop_time_long" if str(sig["direction"]) == "long" else "last_stop_time_short"
        if sst.get(last_stop_key) is not None and (ts - pd.Timestamp(sst[last_stop_key])).total_seconds() < self._no_reentry_stop_min * 60.0:
            return None
        if self._adx_filter_enabled:
            adx_now = float(current_bar.adx_m15)
            if not np.isfinite(adx_now) or adx_now > self._adx_max_for_entry:
                return None

        entry_price_preview = float(current_bar.ask_close) if str(sig["direction"]) == "long" else float(current_bar.bid_close)
        raw_stop_price = self._raw_stop_price(sig, entry_price_preview)
        if raw_stop_price is None:
            return None
        sl_pips = self._clamped_stop_pips(str(sig["direction"]), entry_price_preview, raw_stop_price)
        if sl_pips <= 0:
            return None
        total_size_mult = float(sig.get("regime_mult", 1.0)) * float(sig.get("quality_mult", 1.0))
        day_name_here = str(current_bar.utc_day_name)
        risk_pct_local = float(self._risk_pct) * float(self._day_risk_multipliers.get(day_name_here, 1.0))
        units = math.floor((float(portfolio.equity) * risk_pct_local) / (sl_pips * (PIP_SIZE / max(1e-9, entry_price_preview))))
        units = int(math.floor(units * max(0.0, total_size_mult)))
        units = int(max(0, min(self._max_units, units)))
        if units < 1:
            return None
        return Signal(
            family=self.family_name,
            direction=str(sig["direction"]),
            stop_loss=float(entry_price_preview - sl_pips * PIP_SIZE) if str(sig["direction"]) == "long" else float(entry_price_preview + sl_pips * PIP_SIZE),
            take_profit=None,
            size=int(units),
            metadata={
                "strategy": self.family_name,
                "entry_session_day": str(sig["session_day"]),
                "signal_time": pd.Timestamp(sig["signal_time"]).isoformat(),
                "confluence_score": int(sig["confluence_score"]),
                "confluence_combo": str(sig["confluence_combo"]),
                "signal_strength_score": int(sig.get("signal_strength_score", 0)),
                "signal_strength_tier": str(sig.get("signal_strength_tier", "weak")),
                "quality_label": str(sig.get("quality_label", "medium")),
                "quality_mult": float(sig.get("quality_mult", 1.0)),
                "regime_label": str(sig.get("regime_label", "neutral")),
                "regime_mult": float(sig.get("regime_mult", 1.0)),
                "raw_stop_price": float(raw_stop_price),
                "sl_min_pips": float(self._min_sl_pips),
                "sl_max_pips": float(self._max_sl_pips),
                "stop_loss_pips": float(sl_pips),
                "tp_mode": self._tp_mode,
                "from_zone": bool(sig["from_zone"]),
                "pivot_P": float(sig["P"]),
                "pivot_R1": float(sig["R1"]),
                "pivot_S1": float(sig["S1"]),
                "atr_m15": float(sig["atr_m15"]),
                "session_midpoint": sig.get("session_midpoint"),
                "partial_close_pct": float(self._partial_close_pct),
                "partial_tp_min_pips": float(self._partial_tp_min_pips),
                "partial_tp_max_pips": float(self._partial_tp_max_pips),
                "partial_tp_atr_mult": float(self._partial_tp_atr_mult),
                "single_tp_atr_mult": float(self._single_tp_atr_mult),
                "single_tp_min_pips": float(self._single_tp_min_pips),
                "single_tp_max_pips": float(self._single_tp_max_pips),
                "breakeven_offset_pips": float(self._breakeven_offset_pips),
                "trail_activate_pips": float(self._trail_activate_pips),
                "trail_distance_pips": float(self._trail_dist_pips),
                "trail_requires_tp1": bool(self._trail_requires_tp1),
                "time_decay_minutes": int(self._time_decay_minutes),
                "time_decay_profit_cap_pips": float(self._time_decay_profit_cap_pips),
                "confirmation_delay_candles": int(sig.get("confirmation_delay_candles", 0)),
                "entry_delay_type": str(sig.get("entry_delay_type", "immediate")),
            },
        )

    def _compute_targets_from_fill(self, position: PositionSnapshot, meta: dict[str, Any]) -> tuple[float, float, float]:
        entry_price = float(position.entry_price)
        direction = str(position.direction)
        from_zone = bool(meta.get("from_zone", False))
        tp_mode = str(meta.get("tp_mode", "partial")).strip().lower()
        atr_pips = float(meta.get("atr_m15", 0.0)) / PIP_SIZE
        if direction == "long":
            tp2 = float(meta.get("pivot_P", entry_price))
            if (tp2 - entry_price) / PIP_SIZE < self._min_tp_pips:
                tp2 = max(float(meta.get("pivot_R1", entry_price + self._min_tp_pips * PIP_SIZE)), entry_price + self._min_tp_pips * PIP_SIZE)
            if tp_mode == "trail_only":
                tp1_pips = 9999.0
                tp2 = entry_price + tp1_pips * PIP_SIZE
                tp1_close = 0.0
            elif tp_mode == "pivot_v2":
                if from_zone:
                    tp1 = float(meta.get("pivot_S1", tp2))
                    tp1_pips = max(0.0, (tp1 - entry_price) / PIP_SIZE)
                    tp1_close = float(meta.get("partial_close_pct", self._partial_close_pct))
                else:
                    tp1 = float(tp2)
                    tp1_pips = max(0.0, (tp1 - entry_price) / PIP_SIZE)
                    tp1_close = 1.0
            elif tp_mode == "single_pivot":
                tp1 = float(tp2)
                tp1_pips = max(0.0, (tp1 - entry_price) / PIP_SIZE)
                tp1_close = 1.0
            elif tp_mode == "single_atr":
                tp1_pips = min(float(meta.get("single_tp_max_pips", self._single_tp_max_pips)), max(float(meta.get("single_tp_min_pips", self._single_tp_min_pips)), float(meta.get("single_tp_atr_mult", self._single_tp_atr_mult)) * atr_pips))
                tp2 = entry_price + tp1_pips * PIP_SIZE
                tp1_close = 1.0
            else:
                tp1_pips = min(float(meta.get("partial_tp_max_pips", self._partial_tp_max_pips)), max(float(meta.get("partial_tp_min_pips", self._partial_tp_min_pips)), float(meta.get("partial_tp_atr_mult", self._partial_tp_atr_mult)) * atr_pips))
                tp1_close = float(meta.get("partial_close_pct", self._partial_close_pct))
            tp1_price = entry_price + tp1_pips * PIP_SIZE
            return float(tp1_price), float(tp2), float(tp1_close)
        tp2 = float(meta.get("pivot_P", entry_price))
        if (entry_price - tp2) / PIP_SIZE < self._min_tp_pips:
            tp2 = min(float(meta.get("pivot_S1", entry_price - self._min_tp_pips * PIP_SIZE)), entry_price - self._min_tp_pips * PIP_SIZE)
        if tp_mode == "trail_only":
            tp1_pips = 9999.0
            tp2 = entry_price - tp1_pips * PIP_SIZE
            tp1_close = 0.0
        elif tp_mode == "pivot_v2":
            if from_zone:
                tp1 = float(meta.get("pivot_R1", tp2))
                tp1_pips = max(0.0, (entry_price - tp1) / PIP_SIZE)
                tp1_close = float(meta.get("partial_close_pct", self._partial_close_pct))
            else:
                tp1 = float(tp2)
                tp1_pips = max(0.0, (entry_price - tp1) / PIP_SIZE)
                tp1_close = 1.0
        elif tp_mode == "single_pivot":
            tp1 = float(tp2)
            tp1_pips = max(0.0, (entry_price - tp1) / PIP_SIZE)
            tp1_close = 1.0
        elif tp_mode == "single_atr":
            tp1_pips = min(float(meta.get("single_tp_max_pips", self._single_tp_max_pips)), max(float(meta.get("single_tp_min_pips", self._single_tp_min_pips)), float(meta.get("single_tp_atr_mult", self._single_tp_atr_mult)) * atr_pips))
            tp2 = entry_price - tp1_pips * PIP_SIZE
            tp1_close = 1.0
        else:
            tp1_pips = min(float(meta.get("partial_tp_max_pips", self._partial_tp_max_pips)), max(float(meta.get("partial_tp_min_pips", self._partial_tp_min_pips)), float(meta.get("partial_tp_atr_mult", self._partial_tp_atr_mult)) * atr_pips))
            tp1_close = float(meta.get("partial_close_pct", self._partial_close_pct))
        tp1_price = entry_price - tp1_pips * PIP_SIZE
        return float(tp1_price), float(tp2), float(tp1_close)

    def _ensure_session(self, session_day: str, current_bar: BarView, equity: float) -> dict[str, Any]:
        if session_day not in self._session_state:
            self._session_state[session_day] = {
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "consec_losses": 0,
                "stopped": False,
                "last_entry_time": None,
                "last_stop_time_long": None,
                "last_stop_time_short": None,
                "session_pnl_usd": 0.0,
                "session_start_equity": float(equity),
                "session_open_price": float(current_bar.mid_open),
                "session_high": float(current_bar.mid_high),
                "session_low": float(current_bar.mid_low),
                "rolling_window": deque(),
                "breakout_cooldown_until": None,
                "last_known_equity": float(equity),
            }
        self._session_state[session_day]["last_known_equity"] = float(equity)
        return self._session_state[session_day]

    def _passes_breakout_gate(self, current_bar: BarView, ts: pd.Timestamp, sst: dict[str, Any]) -> bool:
        if self._breakout_mode == "from_open":
            moved_up_pips = (float(sst["session_high"]) - float(sst["session_open_price"])) / PIP_SIZE
            moved_dn_pips = (float(sst["session_open_price"]) - float(sst["session_low"])) / PIP_SIZE
            moved_from_open_pips = max(moved_up_pips, moved_dn_pips)
            if moved_from_open_pips > self._breakout_disable_pips:
                sst["stopped"] = True
                return False
            return not bool(sst.get("stopped", False))
        if self._breakout_mode == "rolling":
            rw: deque = sst["rolling_window"]
            rw.append((ts, float(current_bar.mid_high), float(current_bar.mid_low)))
            cutoff = ts - pd.Timedelta(minutes=self._rolling_window_minutes)
            while rw and pd.Timestamp(rw[0][0]) < cutoff:
                rw.popleft()
            if sst.get("breakout_cooldown_until") is not None and ts < pd.Timestamp(sst["breakout_cooldown_until"]):
                return False
            highs = [x[1] for x in rw]
            lows = [x[2] for x in rw]
            rolling_range_pips = (max(highs) - min(lows)) / PIP_SIZE if highs and lows else 0.0
            if rolling_range_pips > self._rolling_range_threshold_pips:
                sst["breakout_cooldown_until"] = ts + pd.Timedelta(minutes=self._breakout_cooldown_minutes)
                return False
            sst["breakout_cooldown_until"] = None
            return True
        if self._breakout_mode == "disabled":
            return True
        raise ValueError(f"Unsupported breakout_detection_mode: {self._breakout_mode}")

    def _raw_stop_price(self, sig: dict[str, Any], entry_price: float) -> float | None:
        if str(sig["direction"]) == "long":
            return (float(sig["S3"]) - self._sl_buf) if bool(sig["from_zone"]) else (float(sig["S2"]) - self._sl_buf)
        return (float(sig["R3"]) + self._sl_buf) if bool(sig["from_zone"]) else (float(sig["R2"]) + self._sl_buf)

    def _clamped_stop_pips(self, direction: str, entry_price: float, raw_stop_price: float) -> float:
        if direction == "long":
            sl_pips = (entry_price - raw_stop_price) / PIP_SIZE
        else:
            sl_pips = (raw_stop_price - entry_price) / PIP_SIZE
        return max(self._min_sl_pips, min(self._max_sl_pips, sl_pips))

    def _missing_indicators(self, current_bar: BarView) -> bool:
        cols = [
            "pivot_P", "pivot_R1", "pivot_R2", "pivot_R3", "pivot_S1", "pivot_S2", "pivot_S3",
            "bb_upper", "bb_lower", "bb_mid", "rsi_m5", "atr_m15", "sar_value", "sar_direction",
        ]
        for col in cols:
            value = getattr(current_bar, col, np.nan)
            if isinstance(value, str):
                if not value:
                    return True
                continue
            if pd.isna(value):
                return True
        return False

    def _row_dict(self, current_bar: BarView) -> dict[str, Any]:
        return {col: getattr(current_bar, col) for col in current_bar._store.columns}

    def _truthy(self, value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "t", "yes"}
        return bool(int(value)) if isinstance(value, (np.integer, int)) else bool(value)

    def _hhmm_to_minutes(self, value: str) -> int:
        hh, mm = value.strip().split(":")
        return int(hh) * 60 + int(mm)

    def _in_window(self, minute_of_day_utc: int, win_start: int, win_end: int) -> bool:
        minute = int(minute_of_day_utc)
        if win_start < win_end:
            return win_start <= minute < win_end
        return minute >= win_start or minute < win_end

    def _minutes_to_session_end(self, minute_of_day_utc: int) -> int:
        minute = int(minute_of_day_utc)
        if self._start_min < self._end_min:
            return max(0, self._end_min - minute)
        if minute >= self._start_min:
            return (1440 - minute) + self._end_min
        return max(0, self._end_min - minute)
