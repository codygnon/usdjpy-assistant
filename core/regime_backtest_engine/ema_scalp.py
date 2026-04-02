"""
EMA Scalp strategy — multi-timeframe EMA momentum scalper for the regime backtest engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from .daily_levels import compute_pdh_pdl
from .models import (
    AdmissionConfig,
    ClosedTrade,
    ExitAction,
    FixedSpreadConfig,
    InstrumentSpec,
    PortfolioSnapshot,
    PositionSnapshot,
    RunConfig,
    Signal,
    SlippageConfig,
    SpreadConfig,
)
from .strategy import BarView, HistoricalDataView
from .synthetic_bars import (
    Bar15M,
    Bar5M,
    BarDaily,
    _aggregate_window,
    _emit_from_sequence,
    _fifteen_minute_anchor_close_label,
    _five_minute_anchor_close_label,
    _minute_key,
    _required_1m_keys_15m,
    _required_1m_keys_5m,
    _session_date_utc,
    _to_utc_datetime,
    compute_ema,
)


def _utc_dt(ts: Any) -> datetime:
    return _to_utc_datetime(ts)


def _bar_mid_ohlc(bv: BarView) -> tuple[float, float, float, float]:
    return (
        float(bv.mid_open),
        float(bv.mid_high),
        float(bv.mid_low),
        float(bv.mid_close),
    )


@dataclass
class EMAScalpConfig:
    variant: Literal["A", "B"]
    pip_size: float = 0.01
    position_units: int = 200_000
    day_boundary_utc_hour: int = 22
    pdh_pdl_buffer_pips: float = 15.0
    breakout_confirm_bars: int = 5
    cooldown_bars_after_sl: int = 9
    max_trades_per_day: int = 10
    trail_buffer: float = 0.02

    @property
    def tp1_pips(self) -> float:
        return 4.0 if self.variant == "A" else 8.0

    @property
    def tp2_pips(self) -> float:
        return 7.0 if self.variant == "A" else 15.0

    @property
    def max_hold_bars(self) -> int:
        return 60 if self.variant == "A" else 90


class _Row:
    __slots__ = ("ts", "idx", "o", "h", "l", "c")

    def __init__(self, ts: datetime, idx: int, o: float, h: float, l: float, c: float) -> None:
        self.ts = ts
        self.idx = idx
        self.o = o
        self.h = h
        self.l = l
        self.c = c

    @property
    def timestamp(self) -> datetime:
        return self.ts

    @property
    def open(self) -> float:
        return self.o

    @property
    def high(self) -> float:
        return self.h

    @property
    def low(self) -> float:
        return self.l

    @property
    def close(self) -> float:
        return self.c


class _IncEMA:
    __slots__ = ("period", "_closes", "_ema", "_prev", "k")

    def __init__(self, period: int) -> None:
        self.period = period
        self._closes: list[float] = []
        self._ema: Optional[float] = None
        self._prev: Optional[float] = None
        self.k = 2.0 / (period + 1)

    def push(self, close: float) -> Optional[float]:
        self._prev = self._ema
        self._closes.append(close)
        if len(self._closes) < self.period:
            self._ema = None
            return None
        if len(self._closes) == self.period:
            self._ema = sum(self._closes) / self.period
            return self._ema
        assert self._ema is not None
        self._ema = close * self.k + self._ema * (1.0 - self.k)
        return self._ema

    @property
    def value(self) -> Optional[float]:
        return self._ema

    @property
    def prev(self) -> Optional[float]:
        return self._prev


def _in_trading_session_utc(ts: datetime) -> bool:
    t = ts.astimezone(timezone.utc)
    minutes = t.hour * 60 + t.minute
    london = 7 * 60 <= minutes < 11 * 60
    ny = 12 * 60 <= minutes < 17 * 60
    return london or ny


def ema_scalp_session_open_utc(ts: datetime) -> bool:
    """True when new entries are allowed (London or NY window, UTC)."""
    return _in_trading_session_utc(ts)


def _session_close_window_utc(ts: datetime) -> bool:
    """Last 5 minutes before London 11:00 or NY 17:00 UTC."""
    t = ts.astimezone(timezone.utc)
    minutes = t.hour * 60 + t.minute
    london_end = (10 * 60 + 55) <= minutes < 11 * 60
    ny_end = (16 * 60 + 55) <= minutes < 17 * 60
    return london_end or ny_end


def tp1_price(*, entry: float, direction: str, tp1_pips: float, pip: float) -> float:
    if direction == "long":
        return entry + tp1_pips * pip
    return entry - tp1_pips * pip


def tp2_price(*, entry: float, direction: str, tp2_pips: float, pip: float) -> float:
    if direction == "long":
        return entry + tp2_pips * pip
    return entry - tp2_pips * pip


def runner_dummy_take_profit(*, direction: str, **_kwargs: Any) -> float:
    if direction == "long":
        return 1.0e9
    return 1.0


class EMAScalpStrategy:
    family_name: str
    cfg: EMAScalpConfig

    def __init__(self, family_name: str, cfg: EMAScalpConfig) -> None:
        self.family_name = family_name
        self.cfg = cfg
        self._rows: list[_Row] = []
        self._by5: dict[datetime, dict[datetime, tuple[Any, int]]] = {}
        self._by15: dict[datetime, dict[datetime, tuple[Any, int]]] = {}
        self._completed_5m: list[Bar5M] = []
        self._completed_15m: list[Bar15M] = []
        self._ema5_9 = _IncEMA(9)
        self._ema5_21 = _IncEMA(21)
        self._ema5_27 = _IncEMA(27)
        self._ema5_33 = _IncEMA(33)
        self._ema15_9 = _IncEMA(9)
        self._ema15_21 = _IncEMA(21)
        self._align15: list[bool] = []
        self._daily: list[BarDaily] = []
        self._d_day: Optional[date] = None
        self._d_bucket: list[tuple[_Row, int]] = []
        self._cooldown_sl_exit_bar: int = -10_000
        self._trades_today: int = 0
        self._trade_day: Optional[date] = None
        self._above_pdh_streak: int = 0
        self._below_pdl_streak: int = 0
        self._last_pdh: Optional[float] = None
        self._last_pdl: Optional[float] = None
        self._pos: Optional[dict[str, Any]] = None
        self._last_trail_5m_ts: Optional[datetime] = None

    def _reset_soft(self) -> None:
        self._rows.clear()
        self._by5.clear()
        self._by15.clear()
        self._completed_5m.clear()
        self._completed_15m.clear()
        self._ema5_9 = _IncEMA(9)
        self._ema5_21 = _IncEMA(21)
        self._ema5_27 = _IncEMA(27)
        self._ema5_33 = _IncEMA(33)
        self._ema15_9 = _IncEMA(9)
        self._ema15_21 = _IncEMA(21)
        self._align15.clear()
        self._daily.clear()
        self._d_day = None
        self._d_bucket.clear()
        self._above_pdh_streak = 0
        self._below_pdl_streak = 0
        self._last_pdh = None
        self._last_pdl = None
        self._last_trail_5m_ts = None

    def _sync_rows(self, current_bar: BarView, history: HistoricalDataView) -> None:
        idx = int(current_bar.bar_index)
        if len(self._rows) > idx + 1:
            self._reset_soft()
            self._cooldown_sl_exit_bar = -10_000
            self._trades_today = 0
            self._trade_day = None
        if len(self._rows) == idx + 1:
            return
        while len(self._rows) <= idx:
            bi = len(self._rows)
            bv = current_bar if bi == idx else history[bi]
            ts = _utc_dt(bv.timestamp)
            o, h, l, c = _bar_mid_ohlc(bv)
            row = _Row(ts, bi, o, h, l, c)
            self._rows.append(row)
            self._ingest_5m(row)
            self._ingest_15m(row)
            self._ingest_daily(row)

    def _ingest_5m(self, row: _Row) -> None:
        ts = row.ts
        anchor = _five_minute_anchor_close_label(ts)
        mk = _minute_key(ts)
        self._by5.setdefault(anchor, {})[mk] = (row, row.idx)
        req = _required_1m_keys_5m(anchor)
        got = _aggregate_window(self._by5[anchor], req)
        if got is None:
            return
        seq, first_ts = got
        b5 = _emit_from_sequence(seq, first_ts, Bar5M)
        self._completed_5m.append(b5)
        self._ema5_9.push(b5.close)
        self._ema5_21.push(b5.close)
        self._ema5_27.push(b5.close)
        self._ema5_33.push(b5.close)
        del self._by5[anchor]

    def _ingest_15m(self, row: _Row) -> None:
        ts = row.ts
        anchor = _fifteen_minute_anchor_close_label(ts)
        mk = _minute_key(ts)
        self._by15.setdefault(anchor, {})[mk] = (row, row.idx)
        req = _required_1m_keys_15m(anchor)
        got = _aggregate_window(self._by15[anchor], req)
        if got is None:
            return
        seq, first_ts = got
        b15 = _emit_from_sequence(seq, first_ts, Bar15M)
        self._completed_15m.append(b15)
        e9 = self._ema15_9.push(b15.close)
        e21 = self._ema15_21.push(b15.close)
        if e9 is not None and e21 is not None:
            self._align15.append(bool(e9 > e21))
            if len(self._align15) > 8:
                self._align15.pop(0)

    def _flush_daily(self) -> None:
        if self._d_day is None or not self._d_bucket:
            self._d_bucket.clear()
            return
        bars = [r for r, _ in self._d_bucket]
        idxs = [i for _, i in self._d_bucket]
        o = bars[0].o
        highs = [x.h for x in bars]
        lows = [x.l for x in bars]
        c = bars[-1].c
        self._daily.append(
            BarDaily(
                trading_day=self._d_day,
                timestamp=bars[0].ts,
                open=o,
                high=max(highs),
                low=min(lows),
                close=c,
                volume=None,
                bar_index_start=min(idxs),
                bar_index_end=max(idxs),
                bar_count=len(bars),
            )
        )
        self._d_bucket.clear()

    def _effective_daily(self) -> list[BarDaily]:
        out = list(self._daily)
        if self._d_day is not None and self._d_bucket:
            bars = [r for r, _ in self._d_bucket]
            idxs = [i for _, i in self._d_bucket]
            o = bars[0].o
            highs = [x.h for x in bars]
            lows = [x.l for x in bars]
            c = bars[-1].c
            out.append(
                BarDaily(
                    trading_day=self._d_day,
                    timestamp=bars[0].ts,
                    open=o,
                    high=max(highs),
                    low=min(lows),
                    close=c,
                    volume=None,
                    bar_index_start=min(idxs),
                    bar_index_end=max(idxs),
                    bar_count=len(bars),
                )
            )
        return out

    def _ingest_daily(self, row: _Row) -> None:
        d = _session_date_utc(row.ts, self.cfg.day_boundary_utc_hour)
        if self._d_day is None:
            self._d_day = d
        elif d != self._d_day:
            self._flush_daily()
            self._d_day = d
        self._d_bucket.append((row, row.idx))

    def _update_pdh_pdl_breakout(self, mid_close: float) -> None:
        eff = self._effective_daily()
        if not eff:
            return
        cur_i = len(eff) - 1
        pdh, pdl = compute_pdh_pdl(eff, cur_i)
        self._last_pdh, self._last_pdl = pdh, pdl
        if pdh is not None and mid_close > float(pdh):
            self._above_pdh_streak += 1
        else:
            self._above_pdh_streak = 0
        if pdl is not None and mid_close < float(pdl):
            self._below_pdl_streak += 1
        else:
            self._below_pdl_streak = 0

    def _pdh_pdl_blocks_long(self, price: float) -> bool:
        pdh = self._last_pdh
        if pdh is None:
            return False
        if self._above_pdh_streak >= self.cfg.breakout_confirm_bars:
            return False
        dist = float(pdh) - price
        return dist >= 0 and dist / self.cfg.pip_size <= self.cfg.pdh_pdl_buffer_pips

    def _pdh_pdl_blocks_short(self, price: float) -> bool:
        pdl = self._last_pdl
        if pdl is None:
            return False
        if self._below_pdl_streak >= self.cfg.breakout_confirm_bars:
            return False
        dist = price - float(pdl)
        return dist >= 0 and dist / self.cfg.pip_size <= self.cfg.pdh_pdl_buffer_pips

    def _warmup_ok(self) -> bool:
        return len(self._rows) >= 315

    def _ema_slopes_5m(self) -> tuple[Optional[float], Optional[float]]:
        e9, e21 = self._ema5_9.value, self._ema5_21.value
        p9, p21 = self._ema5_9.prev, self._ema5_21.prev
        if e9 is None or e21 is None or p9 is None or p21 is None:
            return None, None
        return e9 - p9, e21 - p21

    def _standard_long(self, row: _Row) -> bool:
        e9, e21 = self._ema5_9.value, self._ema5_21.value
        h9, h21 = self._ema15_9.value, self._ema15_21.value
        if e9 is None or e21 is None or h9 is None or h21 is None:
            return False
        s9, s21 = self._ema_slopes_5m()
        if s9 is None or s21 is None:
            return False
        if not (e9 > e21 and s9 > 0 and s21 > 0 and h9 > h21):
            return False
        return row.l <= e21 and row.c > e21

    def _standard_short(self, row: _Row) -> bool:
        e9, e21 = self._ema5_9.value, self._ema5_21.value
        h9, h21 = self._ema15_9.value, self._ema15_21.value
        if e9 is None or e21 is None or h9 is None or h21 is None:
            return False
        s9, s21 = self._ema_slopes_5m()
        if s9 is None or s21 is None:
            return False
        if not (e9 < e21 and s9 < 0 and s21 < 0 and h9 < h21):
            return False
        return row.h >= e21 and row.c < e21

    def _deep_long(self, row: _Row) -> bool:
        e9, e21 = self._ema5_9.value, self._ema5_21.value
        e27, e33 = self._ema5_27.value, self._ema5_33.value
        if e9 is None or e21 is None or e27 is None or e33 is None:
            return False
        if not (e9 > e21):
            return False
        if len(self._align15) < 3 or not (self._align15[-1] and self._align15[-2] and self._align15[-3]):
            return False
        touched = row.l <= e27 or row.l <= e33
        return touched and row.c > e27

    def _deep_short(self, row: _Row) -> bool:
        e9, e21 = self._ema5_9.value, self._ema5_21.value
        e27, e33 = self._ema5_27.value, self._ema5_33.value
        if e9 is None or e21 is None or e27 is None or e33 is None:
            return False
        if not (e9 < e21):
            return False
        if len(self._align15) < 3 or any(self._align15[-3:]):
            return False
        touched = row.h >= e27 or row.h >= e33
        return touched and row.c < e27

    def _entry_decision(self, current_bar: BarView, portfolio: PortfolioSnapshot) -> Signal | None:
        row = self._rows[current_bar.bar_index]
        self._update_pdh_pdl_breakout(row.c)

        if any(p.family == self.family_name for p in portfolio.open_positions):
            return None

        if not self._warmup_ok():
            return None

        if not _in_trading_session_utc(row.ts):
            return None

        if self._cooldown_sl_exit_bar >= 0 and row.idx <= self._cooldown_sl_exit_bar + self.cfg.cooldown_bars_after_sl:
            return None

        td = _session_date_utc(row.ts, self.cfg.day_boundary_utc_hour)
        if self._trade_day != td:
            self._trade_day = td
            self._trades_today = 0
        if self._trades_today >= self.cfg.max_trades_per_day:
            return None

        std_l = self._standard_long(row)
        std_s = self._standard_short(row)
        deep_l = (not std_l) and self._deep_long(row)
        deep_s = (not std_s) and self._deep_short(row)

        direction: Optional[str] = None
        if std_l:
            direction = "long"
        elif std_s:
            direction = "short"
        elif deep_l:
            direction = "long"
        elif deep_s:
            direction = "short"
        else:
            return None

        mid = row.c
        if direction == "long" and self._pdh_pdl_blocks_long(mid):
            return None
        if direction == "short" and self._pdh_pdl_blocks_short(mid):
            return None

        pip = self.cfg.pip_size
        if direction == "long":
            sl = mid - 15 * pip
            tp = mid + self.cfg.tp1_pips * pip
        else:
            sl = mid + 15 * pip
            tp = mid - self.cfg.tp1_pips * pip

        return Signal(
            family=self.family_name,
            direction=direction,
            stop_loss=float(sl),
            take_profit=float(tp),
            size=int(self.cfg.position_units),
            metadata={
                "stop_loss_pips": 15.0,
                "take_profit_pips": float(self.cfg.tp1_pips),
                "ema_scalp_variant": self.cfg.variant,
            },
        )

    def evaluate(
        self,
        current_bar: BarView,
        history: HistoricalDataView,
        portfolio: PortfolioSnapshot,
    ) -> Signal | None:
        self._sync_rows(current_bar, history)
        return self._entry_decision(current_bar, portfolio)

    def on_position_opened(self, position: PositionSnapshot, signal: Signal, current_bar: BarView) -> None:
        self._trades_today += 1
        pip = self.cfg.pip_size
        far = runner_dummy_take_profit(direction=position.direction)
        self._pos = {
            "entry_bar": int(position.entry_bar),
            "entry_price": float(position.entry_price),
            "direction": position.direction,
            "phase": "tp1",
            "initial_size": int(position.size),
            "tp1": tp1_price(entry=float(position.entry_price), direction=position.direction, tp1_pips=self.cfg.tp1_pips, pip=pip),
            "tp2": tp2_price(entry=float(position.entry_price), direction=position.direction, tp2_pips=self.cfg.tp2_pips, pip=pip),
            "runner_tp": float(far),
        }
        self._last_trail_5m_ts = self._completed_5m[-1].timestamp if self._completed_5m else None

    def on_position_closed(self, trade: ClosedTrade) -> None:
        if trade.family != self.family_name:
            return
        if trade.exit_reason == "stop_loss":
            self._cooldown_sl_exit_bar = int(trade.exit_bar)
        if int(getattr(trade, "remaining_units", 0) or 0) <= 0:
            self._pos = None
            self._last_trail_5m_ts = None

    def get_exit_conditions(
        self,
        position: PositionSnapshot,
        current_bar: BarView,
        history: HistoricalDataView,
    ) -> ExitAction | None:
        if position.family != self.family_name:
            return None
        self._sync_rows(current_bar, history)
        st = self._pos
        if st is None:
            st = {
                "entry_bar": int(position.entry_bar),
                "entry_price": float(position.entry_price),
                "direction": position.direction,
                "phase": "tp1",
                "initial_size": int(getattr(position, "initial_size", position.size)),
                "tp1": tp1_price(
                    entry=float(position.entry_price),
                    direction=position.direction,
                    tp1_pips=self.cfg.tp1_pips,
                    pip=self.cfg.pip_size,
                ),
                "tp2": tp2_price(
                    entry=float(position.entry_price),
                    direction=position.direction,
                    tp2_pips=self.cfg.tp2_pips,
                    pip=self.cfg.pip_size,
                ),
                "runner_tp": runner_dummy_take_profit(direction=position.direction),
            }
            self._pos = st

        row = self._rows[current_bar.bar_index]
        pip = self.cfg.pip_size
        direction = str(st["direction"])
        entry_bar = int(st["entry_bar"])
        bars_held = int(row.idx - entry_bar)

        if _session_close_window_utc(row.ts):
            return ExitAction(
                reason="session_close",
                exit_type="full",
                close_fraction=1.0,
                price=float(current_bar.bid_close if direction == "long" else current_bar.ask_close),
            )

        max_hold = self.cfg.max_hold_bars
        if bars_held >= max_hold:
            return ExitAction(
                reason="max_hold",
                exit_type="full",
                close_fraction=1.0,
                price=float(current_bar.bid_close if direction == "long" else current_bar.ask_close),
            )

        phase = str(st["phase"])
        tp1 = float(st["tp1"])
        tp2 = float(st["tp2"])

        if phase == "tp1":
            if direction == "long":
                hit = float(current_bar.bid_high) >= tp1
            else:
                hit = float(current_bar.ask_low) <= tp1
            if hit:
                st["phase"] = "tp2"
                return ExitAction(
                    reason="tp1",
                    exit_type="partial",
                    close_fraction=0.5,
                    price=tp1,
                    new_take_profit=float(st["tp2"]),
                )

        if phase == "tp2":
            if direction == "long":
                hit = float(current_bar.bid_high) >= tp2
            else:
                hit = float(current_bar.ask_low) <= tp2
            if hit:
                st["phase"] = "runner"
                return ExitAction(
                    reason="tp2",
                    exit_type="partial",
                    close_fraction=0.5,
                    price=tp2,
                    new_take_profit=float(st["runner_tp"]),
                )

        if phase == "runner":
            e9 = self._ema5_9.value
            if e9 is None:
                return None
            new_5m_ts = self._completed_5m[-1].timestamp if self._completed_5m else None
            if new_5m_ts is not None and self._last_trail_5m_ts != new_5m_ts:
                self._last_trail_5m_ts = new_5m_ts
                cur_sl = float(position.stop_loss)
                if direction == "long":
                    raw = float(e9) - self.cfg.trail_buffer
                    new_sl = max(cur_sl, raw)
                    if new_sl <= cur_sl + 1e-12:
                        return None
                else:
                    raw = float(e9) + self.cfg.trail_buffer
                    new_sl = min(cur_sl, raw)
                    if new_sl >= cur_sl - 1e-12:
                        return None
                return ExitAction(
                    reason="trail_5m9",
                    exit_type="none",
                    close_fraction=1.0,
                    new_stop_loss=float(new_sl),
                )

        return None


def verify_ema_sequence(values: list[float], period: int) -> list[float]:
    """Test helper: EMA via shared compute_ema."""
    return compute_ema(values, period)


def build_ema_scalp_run_config(
    *,
    hypothesis: str,
    data_path: Path,
    output_dir: Path,
    family_name: str,
    variant: Literal["A", "B"],
    spread_pips: float,
    slippage_pips: float,
    initial_balance: float = 100_000.0,
    bar_log_format: Literal["parquet", "csv"] = "csv",
) -> RunConfig:
    if spread_pips <= 0:
        raise ValueError("spread_pips must be > 0 for FixedSpreadConfig; use a tiny epsilon for zero-spread runs")
    return RunConfig(
        hypothesis=hypothesis,
        data_path=data_path,
        output_dir=output_dir,
        mode="standalone",
        active_families=(family_name,),
        instrument=InstrumentSpec(symbol="USDJPY"),
        spread=SpreadConfig(spread_source="fixed", fixed=FixedSpreadConfig(spread_pips=float(spread_pips))),
        slippage=SlippageConfig(fixed_slippage_pips=float(slippage_pips)),
        admission=AdmissionConfig(
            allow_opposing_exposure=False,
            max_total_open_positions=1,
            max_open_positions_per_family={family_name: 1},
            max_total_units=500_000,
            max_units_per_family={family_name: 250_000},
            family_priority=(family_name,),
        ),
        manifest=None,
        initial_balance=float(initial_balance),
        bar_log_format=bar_log_format,
    )
