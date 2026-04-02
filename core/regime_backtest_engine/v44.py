from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from core.phase3_v44_evaluator import compute_v44_sl, evaluate_v44_entry

from .models import ClosedTrade, ExitAction, PortfolioSnapshot, PositionSnapshot, Signal
from .strategy import BarView, HistoricalDataView, TrainableStrategyFamily


class V44StrategyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    family_name: str = "v44_ny"
    ny_start_hour: float = 13.0
    ny_end_hour: float = 16.0
    ny_start_delay_minutes: int = 5
    session_entry_cutoff_minutes: int = 60
    max_entry_spread_pips: float = 3.0
    max_entries_per_day: int = 7
    max_open_positions: int = 3
    session_stop_losses: int = 3
    cooldown_win_bars: int = 1
    cooldown_loss_bars: int = 1
    cooldown_scratch_bars: int = 2
    scratch_threshold_pips: float = 1.0
    h1_ema_fast_period: int = 20
    h1_ema_slow_period: int = 50
    m5_ema_fast_period: int = 9
    m5_ema_slow_period: int = 21
    slope_bars: int = 4
    strong_slope_threshold: float = 0.5
    weak_slope_threshold: float = 0.2
    min_body_pips: float = 1.0
    atr_pct_filter_enabled: bool = True
    atr_pct_cap: float = 0.67
    atr_pct_lookback: int = 200
    skip_weak: bool = True
    skip_normal: bool = False
    ny_strength_allow: str = "strong_normal"
    risk_per_trade_pct: float = 0.5
    session_risk_mult: float = 1.2
    strong_risk_mult: float = 1.0
    normal_risk_mult: float = 0.6
    rp_min_lot: float = 1.0
    rp_max_lot: float = 20.0
    strong_tp1_pips: float = 2.0
    strong_tp2_pips: float = 5.0
    strong_tp1_close_pct: float = 0.5
    normal_tp1_pips: float = 1.75
    normal_tp2_pips: float = 3.0
    normal_tp1_close_pct: float = 0.5
    be_offset_pips: float = 0.5
    close_risk_at_session_end: bool = True
    strong_trail_buffer_pips: float = 4.0
    normal_trail_buffer_pips: float = 3.0
    strong_trail_ema_period: int = 21
    normal_trail_ema_period: int = 9
    news_filter_enabled: bool = True
    news_window_minutes_before: int = 60
    news_window_minutes_after: int = 30
    news_impact_min: str = "high"
    news_calendar_path: Optional[Path] = None

    @classmethod
    def from_v44_json(cls, path: str | Path, *, family_name: str = "v44_ny") -> "V44StrategyConfig":
        raw = json.loads(Path(path).read_text())
        news_path = raw.get("v5_news_calendar_path")
        return cls(
            family_name=family_name,
            ny_start_hour=float(raw.get("ny_start", 13.0)),
            ny_end_hour=float(raw.get("ny_end", 16.0)),
            ny_start_delay_minutes=int(raw.get("v5_ny_start_delay_minutes", 5)),
            session_entry_cutoff_minutes=int(raw.get("v5_session_entry_cutoff_minutes", 60)),
            max_entry_spread_pips=float(raw.get("v5_max_entry_spread_pips", raw.get("max_entry_spread_pips", 3.0))),
            max_entries_per_day=int(raw.get("v5_max_entries_day", 7)),
            max_open_positions=int(raw.get("v5_max_open", 3)),
            session_stop_losses=int(raw.get("v5_session_stop_losses", 3)),
            cooldown_win_bars=int(raw.get("v5_cooldown_win", 1)),
            cooldown_loss_bars=int(raw.get("v5_cooldown_loss", 1)),
            cooldown_scratch_bars=int(raw.get("v5_cooldown_scratch", 2)),
            scratch_threshold_pips=float(raw.get("v5_scratch_threshold", 1.0)),
            h1_ema_fast_period=int(raw.get("h1_ema_fast", 20)),
            h1_ema_slow_period=int(raw.get("h1_ema_slow", 50)),
            m5_ema_fast_period=int(raw.get("v5_m5_ema_fast", 9)),
            m5_ema_slow_period=int(raw.get("v5_m5_ema_slow", 21)),
            slope_bars=int(raw.get("v5_slope_bars", 4)),
            strong_slope_threshold=float(raw.get("v5_strong_slope", 0.5)),
            weak_slope_threshold=float(raw.get("v5_weak_slope", 0.2)),
            min_body_pips=float(raw.get("v5_entry_min_body_pips", 1.0)),
            atr_pct_filter_enabled=bool(raw.get("v5_atr_pct_filter_enabled", True)),
            atr_pct_cap=float(raw.get("v5_atr_pct_cap", 0.67)),
            atr_pct_lookback=int(raw.get("v5_atr_pct_lookback", 200)),
            skip_weak=bool(raw.get("v5_skip_weak", True)),
            skip_normal=bool(raw.get("v5_skip_normal", False)),
            ny_strength_allow=str(raw.get("v5_ny_strength_allow", "strong_normal")),
            risk_per_trade_pct=float(raw.get("v5_risk_per_trade_pct", 0.5)),
            session_risk_mult=float(raw.get("v5_rp_ny_mult", 1.2)),
            strong_risk_mult=float(raw.get("v5_rp_strong_mult", 1.0)),
            normal_risk_mult=float(raw.get("v5_rp_normal_mult", 0.6)),
            rp_min_lot=float(raw.get("v5_rp_min_lot", 1.0)),
            rp_max_lot=float(raw.get("v5_rp_max_lot", 20.0)),
            strong_tp1_pips=float(raw.get("v5_strong_tp1", 2.0)),
            strong_tp2_pips=float(raw.get("v5_strong_tp2", 5.0)),
            strong_tp1_close_pct=float(raw.get("v5_strong_tp1_close_pct", 0.5)),
            normal_tp1_pips=float(raw.get("v5_normal_tp1", 1.75)),
            normal_tp2_pips=float(raw.get("v5_normal_tp2", 3.0)),
            normal_tp1_close_pct=float(raw.get("v5_normal_tp1_close_pct", 0.5)),
            be_offset_pips=float(raw.get("v5_be_offset", 0.5)),
            close_risk_at_session_end=bool(raw.get("v5_close_full_risk_at_session_end", True)),
            strong_trail_buffer_pips=float(raw.get("v5_strong_trail_buffer_pips", raw.get("v5_strong_trail_buffer", 4.0))),
            normal_trail_buffer_pips=float(raw.get("v5_normal_trail_buffer", 3.0)),
            strong_trail_ema_period=int(raw.get("v5_strong_trail_ema", 21)),
            normal_trail_ema_period=int(raw.get("v5_normal_trail_ema", 9)),
            news_filter_enabled=bool(raw.get("v5_news_filter_enabled", True)),
            news_window_minutes_before=int(raw.get("v5_news_window_minutes_before", 60)),
            news_window_minutes_after=int(raw.get("v5_news_window_minutes_after", 30)),
            news_impact_min=str(raw.get("v5_news_impact_min", "high")),
            news_calendar_path=Path(news_path) if news_path else None,
        )


@dataclass
class _TradePlan:
    direction: str
    strength: str
    tp1_price: float
    tp2_price: float
    be_stop_price: float
    tp1_close_fraction: float
    tp1_hit: bool = False


class V44NYStrategy(TrainableStrategyFamily):
    family_name = "v44_ny"

    def __init__(self, config: V44StrategyConfig, *, pip_size: float = 0.01, lot_size_units: int = 100_000) -> None:
        self.config = config
        self.family_name = config.family_name
        self.pip_size = pip_size
        self.lot_size_units = lot_size_units
        self._m5_bars: deque[dict[str, Any]] = deque(maxlen=512)
        self._h1_bars: deque[dict[str, Any]] = deque(maxlen=128)
        self._current_m5: Optional[dict[str, Any]] = None
        self._current_h1: Optional[dict[str, Any]] = None
        self._m5_just_closed = False
        self._current_day: Optional[pd.Timestamp] = None
        self._trade_count = 0
        self._consecutive_losses = 0
        self._cooldown_until: Optional[pd.Timestamp] = None
        self._trade_plans: dict[int, _TradePlan] = {}
        self._news_events = self._load_news_calendar(config.news_calendar_path)

    def fit(self, history: HistoricalDataView) -> None:
        return None

    def evaluate(self, current_bar: BarView, history: HistoricalDataView, portfolio: PortfolioSnapshot) -> Signal | None:
        self._ingest_bar(current_bar)
        ts = pd.Timestamp(current_bar.timestamp)
        self._reset_day_if_needed(ts)

        if not self._m5_just_closed:
            return None
        if not self._is_entry_time(ts):
            return None
        if float(current_bar.spread_pips) > float(self.config.max_entry_spread_pips):
            return None
        if self.config.news_filter_enabled and self._news_block_active(ts):
            return None

        m5_df = self._bars_to_frame(self._m5_bars)
        h1_df = self._bars_to_frame(self._h1_bars)
        session_state = {
            "trade_count": self._trade_count,
            "consecutive_losses": self._consecutive_losses,
            "cooldown_until": self._cooldown_until.isoformat() if self._cooldown_until is not None else None,
        }
        side, strength, _reason = evaluate_v44_entry(
            h1_df,
            m5_df,
            tick=None,
            pip_size=self.pip_size,
            session="ny",
            session_state=session_state,
            now_utc=ts.to_pydatetime(),
            max_entries_per_day=self.config.max_entries_per_day,
            session_stop_losses=self.config.session_stop_losses,
            h1_ema_fast_period=self.config.h1_ema_fast_period,
            h1_ema_slow_period=self.config.h1_ema_slow_period,
            m5_ema_fast_period=self.config.m5_ema_fast_period,
            m5_ema_slow_period=self.config.m5_ema_slow_period,
            slope_bars=self.config.slope_bars,
            strong_slope_threshold=self.config.strong_slope_threshold,
            weak_slope_threshold=self.config.weak_slope_threshold,
            min_body_pips=self.config.min_body_pips,
            atr_pct_filter_enabled=self.config.atr_pct_filter_enabled,
            atr_pct_cap=self.config.atr_pct_cap,
            atr_pct_lookback=self.config.atr_pct_lookback,
        )
        if side is None:
            return None
        if strength == "weak" and self.config.skip_weak:
            return None
        if strength == "normal" and self.config.skip_normal:
            return None
        if not self._strength_allowed(strength):
            return None

        entry_price = float(current_bar.ask_close) if side == "buy" else float(current_bar.bid_close)
        stop_loss = compute_v44_sl(side, m5_df, entry_price, self.pip_size)
        stop_pips = max(abs(entry_price - stop_loss) / self.pip_size, 1.0)
        size_units = self._size_for_trade(portfolio.equity, stop_pips, entry_price, strength)
        if size_units <= 0:
            return None

        tp1_pips, tp2_pips = self._tp_targets(strength)
        take_profit = entry_price + tp2_pips * self.pip_size if side == "buy" else entry_price - tp2_pips * self.pip_size
        return Signal(
            family=self.family_name,
            direction="long" if side == "buy" else "short",
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
            size=int(size_units),
            metadata={
                "strategy": "v44_ny",
                "strength": strength,
                "stop_loss_pips": stop_pips,
                "tp1_pips": tp1_pips,
                "tp2_pips": tp2_pips,
                "take_profit_pips": tp2_pips,
                "be_offset_pips": float(self.config.be_offset_pips),
            },
        )

    def get_exit_conditions(self, position: PositionSnapshot, current_bar: BarView, history: HistoricalDataView) -> ExitAction | None:
        plan = self._trade_plans.get(int(position.trade_id))
        if plan is None:
            return None
        ts = pd.Timestamp(current_bar.timestamp)
        if self.config.close_risk_at_session_end and self._is_after_session_end(ts):
            return ExitAction(reason="session_end_close", exit_type="full", close_fraction=1.0)

        if not plan.tp1_hit and self._tp1_touched(position, current_bar, plan):
            plan.tp1_hit = True
            return ExitAction(
                reason="tp1_partial",
                exit_type="partial",
                close_fraction=float(plan.tp1_close_fraction),
                price=float(plan.tp1_price),
                new_stop_loss=float(plan.be_stop_price),
                new_take_profit=float(plan.tp2_price),
            )

        if plan.tp1_hit:
            trailed = self._trail_stop(position, plan)
            if trailed is not None:
                return ExitAction(
                    reason="trail_update",
                    exit_type="none",
                    new_stop_loss=float(trailed),
                    new_take_profit=float(plan.tp2_price),
                )
        return None

    def on_position_opened(self, position: PositionSnapshot, signal: Signal, current_bar: BarView) -> None:
        strength = str(signal.metadata.get("strength") or "normal")
        tp1_pips = float(signal.metadata.get("tp1_pips") or self.config.normal_tp1_pips)
        tp2_pips = float(signal.metadata.get("tp2_pips") or self.config.normal_tp2_pips)
        tp1_close_fraction = self._tp1_close_fraction(strength)
        if position.direction == "long":
            tp1_price = float(position.entry_price) + tp1_pips * self.pip_size
            tp2_price = float(position.entry_price) + tp2_pips * self.pip_size
            be_stop = float(position.entry_price) + float(self.config.be_offset_pips) * self.pip_size
        else:
            tp1_price = float(position.entry_price) - tp1_pips * self.pip_size
            tp2_price = float(position.entry_price) - tp2_pips * self.pip_size
            be_stop = float(position.entry_price) - float(self.config.be_offset_pips) * self.pip_size
        self._trade_count += 1
        self._trade_plans[int(position.trade_id)] = _TradePlan(
            direction=position.direction,
            strength=strength,
            tp1_price=tp1_price,
            tp2_price=tp2_price,
            be_stop_price=be_stop,
            tp1_close_fraction=tp1_close_fraction,
        )

    def on_position_closed(self, trade: ClosedTrade) -> None:
        if trade.event_type == "partial" and int(trade.remaining_units) > 0:
            return

        self._trade_plans.pop(int(trade.trade_id), None)
        pnl_pips = float(trade.pnl_pips)
        ts = pd.Timestamp(trade.exit_time)
        if pnl_pips > 0:
            self._consecutive_losses = 0
            self._cooldown_until = ts + pd.Timedelta(minutes=int(self.config.cooldown_win_bars))
        elif abs(pnl_pips) <= float(self.config.scratch_threshold_pips):
            self._cooldown_until = ts + pd.Timedelta(minutes=int(self.config.cooldown_scratch_bars))
        else:
            self._consecutive_losses += 1
            self._cooldown_until = ts + pd.Timedelta(minutes=int(self.config.cooldown_loss_bars))

    def _strength_allowed(self, strength: str) -> bool:
        allow = str(self.config.ny_strength_allow or "strong_normal").lower()
        if allow == "strong_only":
            return strength == "strong"
        if allow in {"strong_normal", "normal_strong"}:
            return strength in {"strong", "normal"}
        return strength in {"strong", "normal", "weak"}

    def _size_for_trade(self, equity: float, stop_pips: float, entry_price: float, strength: str) -> int:
        risk_pct = float(self.config.risk_per_trade_pct) / 100.0
        risk_usd = float(equity) * risk_pct * float(self.config.session_risk_mult)
        risk_usd *= float(self.config.strong_risk_mult if strength == "strong" else self.config.normal_risk_mult)
        usd_per_unit_at_stop = stop_pips * self.pip_size / max(entry_price, 1e-9)
        raw_units = risk_usd / max(usd_per_unit_at_stop, 1e-9)
        lots = raw_units / self.lot_size_units
        lots = max(float(self.config.rp_min_lot), min(float(self.config.rp_max_lot), lots))
        return int(lots * self.lot_size_units)

    def _tp_targets(self, strength: str) -> tuple[float, float]:
        if strength == "strong":
            return float(self.config.strong_tp1_pips), float(self.config.strong_tp2_pips)
        return float(self.config.normal_tp1_pips), float(self.config.normal_tp2_pips)

    def _tp1_close_fraction(self, strength: str) -> float:
        if strength == "strong":
            return float(self.config.strong_tp1_close_pct)
        return float(self.config.normal_tp1_close_pct)

    def _is_entry_time(self, ts: pd.Timestamp) -> bool:
        hour = float(ts.hour) + float(ts.minute) / 60.0
        start = float(self.config.ny_start_hour) + float(self.config.ny_start_delay_minutes) / 60.0
        cutoff = float(self.config.ny_end_hour) - float(self.config.session_entry_cutoff_minutes) / 60.0
        return start <= hour <= cutoff

    def _is_after_session_end(self, ts: pd.Timestamp) -> bool:
        hour = float(ts.hour) + float(ts.minute) / 60.0
        return hour >= float(self.config.ny_end_hour)

    def _reset_day_if_needed(self, ts: pd.Timestamp) -> None:
        current_day = ts.normalize()
        if self._current_day is None or current_day != self._current_day:
            self._current_day = current_day
            self._trade_count = 0
            self._consecutive_losses = 0
            self._cooldown_until = None

    def _ingest_bar(self, current_bar: BarView) -> None:
        self._m5_just_closed = False
        ts = pd.Timestamp(current_bar.timestamp)
        self._update_bucket(current_bar, ts.floor("5min"), "m5")
        self._update_bucket(current_bar, ts.floor("1h"), "h1")
        if ts.minute % 5 == 4:
            self._finalize_bucket("m5")
            self._m5_just_closed = True
        if ts.minute == 59:
            self._finalize_bucket("h1")

    def _update_bucket(self, current_bar: BarView, bucket_start: pd.Timestamp, kind: str) -> None:
        current = self._current_m5 if kind == "m5" else self._current_h1
        if current is None or current["bucket_start"] != bucket_start:
            current = {
                "bucket_start": bucket_start,
                "timestamp": current_bar.timestamp,
                "open": float(current_bar.mid_open),
                "high": float(current_bar.mid_high),
                "low": float(current_bar.mid_low),
                "close": float(current_bar.mid_close),
            }
        else:
            current["timestamp"] = current_bar.timestamp
            current["high"] = max(float(current["high"]), float(current_bar.mid_high))
            current["low"] = min(float(current["low"]), float(current_bar.mid_low))
            current["close"] = float(current_bar.mid_close)
        if kind == "m5":
            self._current_m5 = current
        else:
            self._current_h1 = current

    def _finalize_bucket(self, kind: str) -> None:
        if kind == "m5" and self._current_m5 is not None:
            self._m5_bars.append(dict(self._current_m5))
            self._current_m5 = None
        if kind == "h1" and self._current_h1 is not None:
            self._h1_bars.append(dict(self._current_h1))
            self._current_h1 = None

    def _bars_to_frame(self, bars: deque[dict[str, Any]]) -> pd.DataFrame:
        if not bars:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])
        return pd.DataFrame(list(bars))

    def _tp1_touched(self, position: PositionSnapshot, current_bar: BarView, plan: _TradePlan) -> bool:
        if position.direction == "long":
            return float(current_bar.bid_high) >= float(plan.tp1_price)
        return float(current_bar.ask_low) <= float(plan.tp1_price)

    def _trail_stop(self, position: PositionSnapshot, plan: _TradePlan) -> float | None:
        if not self._m5_bars:
            return None
        m5_df = self._bars_to_frame(self._m5_bars)
        period = self.config.strong_trail_ema_period if plan.strength == "strong" else self.config.normal_trail_ema_period
        buffer_pips = self.config.strong_trail_buffer_pips if plan.strength == "strong" else self.config.normal_trail_buffer_pips
        if len(m5_df) < period:
            return None
        ema = m5_df["close"].astype(float).ewm(span=period, adjust=False).mean()
        if position.direction == "long":
            return float(ema.iloc[-1]) - float(buffer_pips) * self.pip_size
        return float(ema.iloc[-1]) + float(buffer_pips) * self.pip_size

    def _load_news_calendar(self, path: Path | None) -> list[pd.Timestamp]:
        if path is None:
            return []
        full_path = path if path.is_absolute() else (Path("/Users/codygnon/Documents/usdjpy_assistant") / path)
        if not full_path.exists():
            return []
        df = pd.read_csv(full_path)
        if not {"date", "time_utc", "impact"}.issubset(df.columns):
            return []
        impacts = {"low": 1, "medium": 2, "high": 3}
        min_rank = impacts.get(str(self.config.news_impact_min).lower(), 3)
        df["impact_rank"] = df["impact"].astype(str).str.lower().map(impacts).fillna(0)
        df = df[df["impact_rank"] >= min_rank].copy()
        if df.empty:
            return []
        ts = pd.to_datetime(df["date"].astype(str) + " " + df["time_utc"].astype(str), utc=True, errors="coerce")
        return [pd.Timestamp(x) for x in ts.dropna().tolist()]

    def _news_block_active(self, ts: pd.Timestamp) -> bool:
        if not self._news_events:
            return False
        before = pd.Timedelta(minutes=int(self.config.news_window_minutes_before))
        after = pd.Timedelta(minutes=int(self.config.news_window_minutes_after))
        for event_ts in self._news_events:
            if event_ts - before <= ts <= event_ts + after:
                return True
        return False
