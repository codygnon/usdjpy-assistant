"""
Cross-Asset Confluence — macro bias + ADX regime + multi-timeframe USDJPY entries.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Literal, Optional

from .cross_asset_bias import BiasReading, CrossAssetBias
from .cross_asset_data import CrossAssetDataLoader
from .daily_levels import compute_pdh_pdl, compute_round_levels, find_nearest_level
from .ema_scalp import runner_dummy_take_profit, tp1_price, tp2_price
from .models import ClosedTrade, ExitAction, PortfolioSnapshot, PositionSnapshot, Signal
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
)

UTC = timezone.utc


def _utc_dt(ts: Any) -> datetime:
    return _to_utc_datetime(ts)


def _bar_mid_ohlc(bv: BarView) -> tuple[float, float, float, float]:
    return (
        float(bv.mid_open),
        float(bv.mid_high),
        float(bv.mid_low),
        float(bv.mid_close),
    )


def _in_trading_session_utc(ts: datetime) -> bool:
    t = ts.astimezone(UTC)
    minutes = t.hour * 60 + t.minute
    london = 7 * 60 <= minutes < 11 * 60
    ny = 12 * 60 <= minutes < 17 * 60
    return london or ny


def _session_close_window_utc(ts: datetime) -> bool:
    t = ts.astimezone(UTC)
    minutes = t.hour * 60 + t.minute
    london_end = (10 * 60 + 55) <= minutes < 11 * 60
    ny_end = (16 * 60 + 55) <= minutes < 17 * 60
    return london_end or ny_end


@dataclass
class Bar1H:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float]
    bar_index_start: int
    bar_index_end: int
    complete: bool = True


def _required_1m_keys_1h(hour_start: datetime) -> list[datetime]:
    return [hour_start + timedelta(minutes=m) for m in range(60)]


def _emit_1h_from_sequence(seq: list[tuple[Any, int]], hour_start: datetime) -> Bar1H:
    bars = [b for b, _ in seq]
    idxs = [i for _, i in seq]
    o, _, _, _ = _bar_mid_ohlc_from_obj(bars[0])
    _, _, _, c = _bar_mid_ohlc_from_obj(bars[-1])
    highs = [_bar_mid_ohlc_from_obj(b)[1] for b in bars]
    lows = [_bar_mid_ohlc_from_obj(b)[2] for b in bars]
    start_i, end_i = min(idxs), max(idxs)
    return Bar1H(
        timestamp=hour_start,
        open=o,
        high=max(highs),
        low=min(lows),
        close=c,
        volume=None,
        bar_index_start=start_i,
        bar_index_end=end_i,
        complete=True,
    )


def _bar_mid_ohlc_from_obj(bar: Any) -> tuple[float, float, float, float]:
    if hasattr(bar, "mid_open"):
        return (float(bar.mid_open), float(bar.mid_high), float(bar.mid_low), float(bar.mid_close))
    return (float(bar.o), float(bar.h), float(bar.l), float(bar.c))


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


@dataclass
class CrossAssetConfluenceConfig:
    pip_size: float = 0.01
    base_position_units: int = 200_000
    day_boundary_utc_hour: int = 22
    warmup_1m_bars: int = 40_000
    min_completed_1h: int = 20
    min_completed_15m: int = 21
    min_completed_daily: int = 28
    stop_loss_pips: float = 20.0
    tp1_pips: float = 10.0
    tp2_pips: float = 25.0
    trail_buffer_pips: float = 2.0
    level_proximity_pips: float = 10.0
    cooldown_bars_after_sl: int = 15
    max_trades_per_day: int = 8
    max_hold_bars: int = 240
    swing_lookback_1h: int = 20
    weak_trend_min_raw: float = 2.0


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


def _bias_side(bias: str) -> int:
    if bias in ("STRONG_LONG", "MILD_LONG"):
        return 1
    if bias in ("STRONG_SHORT", "MILD_SHORT"):
        return -1
    return 0


def _swing_levels_1h(bars: list[Bar1H], lookback: int) -> list[tuple[float, str]]:
    if len(bars) < 3:
        return []
    window = bars[-lookback:] if len(bars) >= lookback else bars
    out: list[tuple[float, str]] = []
    for i in range(1, len(window) - 1):
        h_prev, h_i, h_next = window[i - 1].high, window[i].high, window[i + 1].high
        l_prev, l_i, l_next = window[i - 1].low, window[i].low, window[i + 1].low
        if h_i > h_prev and h_i > h_next:
            out.append((float(h_i), "swing_high"))
        if l_i < l_prev and l_i < l_next:
            out.append((float(l_i), "swing_low"))
    return out


def _latest_swing_low_15m(bars: list[Bar15M]) -> Optional[float]:
    if len(bars) < 3:
        return None
    last: Optional[float] = None
    for i in range(1, len(bars) - 1):
        l_prev, l_i, l_next = bars[i - 1].low, bars[i].low, bars[i + 1].low
        if l_i < l_prev and l_i < l_next:
            last = float(l_i)
    return last


def _latest_swing_high_15m(bars: list[Bar15M]) -> Optional[float]:
    if len(bars) < 3:
        return None
    last: Optional[float] = None
    for i in range(1, len(bars) - 1):
        h_prev, h_i, h_next = bars[i - 1].high, bars[i].high, bars[i + 1].high
        if h_i > h_prev and h_i > h_next:
            last = float(h_i)
    return last


class CrossAssetConfluenceStrategy:
    family_name = "cross_asset_confluence"

    def __init__(
        self,
        data_loader: CrossAssetDataLoader,
        config: str | CrossAssetConfluenceConfig = "default",
    ) -> None:
        self._loader = data_loader
        self.bias_engine = CrossAssetBias(data_loader)
        if isinstance(config, CrossAssetConfluenceConfig):
            self.cfg = config
        else:
            self.cfg = CrossAssetConfluenceConfig()

        self._rows: list[_Row] = []
        self._by5: dict[datetime, dict[datetime, tuple[Any, int]]] = {}
        self._by15: dict[datetime, dict[datetime, tuple[Any, int]]] = {}
        self._by1h: dict[datetime, dict[datetime, tuple[Any, int]]] = {}
        self._completed_5m: list[Bar5M] = []
        self._completed_15m: list[Bar15M] = []
        self._completed_1h: list[Bar1H] = []
        self._ema5_9 = _IncEMA(9)
        self._ema5_21 = _IncEMA(21)
        self._ema15_9 = _IncEMA(9)
        self._ema15_21 = _IncEMA(21)

        self._daily: list[BarDaily] = []
        self._d_day: Optional[date] = None
        self._d_bucket: list[tuple[_Row, int]] = []

        self._structural: list[tuple[float, str]] = []
        self._last_bias_token: Optional[tuple[Any, Any]] = None
        self._current_bias: Optional[BiasReading] = None

        self._cooldown_sl_exit_bar: int = -10_000
        self._trades_today: int = 0
        self._trade_day: Optional[date] = None

        self._pos: Optional[dict[str, Any]] = None
        self._last_trail_15m_ts: Optional[datetime] = None
        self._entry_bias_sign: int = 0

    def _reset_soft(self) -> None:
        self._rows.clear()
        self._by5.clear()
        self._by15.clear()
        self._by1h.clear()
        self._completed_5m.clear()
        self._completed_15m.clear()
        self._completed_1h.clear()
        self._ema5_9 = _IncEMA(9)
        self._ema5_21 = _IncEMA(21)
        self._ema15_9 = _IncEMA(9)
        self._ema15_21 = _IncEMA(21)
        self._daily.clear()
        self._d_day = None
        self._d_bucket.clear()
        self._structural.clear()
        self._last_bias_token = None
        self._current_bias = None
        self._last_trail_15m_ts = None

    def _daily_dicts_completed(self) -> list[dict[str, float]]:
        return [{"high": float(b.high), "low": float(b.low), "close": float(b.close)} for b in self._daily]

    def _maybe_refresh_bias(self, ts: datetime) -> None:
        br = self._loader.get_brent_at(ts)
        eu = self._loader.get_eurusd_at(ts)
        if br is None or eu is None:
            return
        token = (br["completed_at"], eu["completed_at"])
        if token == self._last_bias_token:
            return
        self._last_bias_token = token
        dlist = self._daily_dicts_completed()
        self._current_bias = self.bias_engine.compute_bias(timestamp=ts, usdjpy_daily_bars=dlist if dlist else None)

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
            self._ingest_1h(row)
            self._ingest_daily(row)
            self._maybe_refresh_bias(ts)

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
        self._ema5_9.push(float(b5.close))
        self._ema5_21.push(float(b5.close))
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
        self._ema15_9.push(float(b15.close))
        self._ema15_21.push(float(b15.close))
        del self._by15[anchor]

    def _ingest_1h(self, row: _Row) -> None:
        hs = row.ts.replace(minute=0, second=0, microsecond=0)
        mk = _minute_key(row.ts)
        self._by1h.setdefault(hs, {})[mk] = (row, row.idx)
        req = _required_1m_keys_1h(hs)
        got = _aggregate_window(self._by1h[hs], req)
        if got is None:
            return
        seq, _ft = got
        b1 = _emit_1h_from_sequence(seq, hs)
        self._completed_1h.append(b1)
        self._structural = _swing_levels_1h(self._completed_1h, self.cfg.swing_lookback_1h)
        del self._by1h[hs]

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
            self._trades_today = 0
        self._d_bucket.append((row, row.idx))

    def _warmup_ok(self) -> bool:
        if len(self._rows) < self.cfg.warmup_1m_bars:
            return False
        if len(self._completed_1h) < self.cfg.min_completed_1h:
            return False
        if len(self._completed_15m) < self.cfg.min_completed_15m:
            return False
        if len(self._daily) < self.cfg.min_completed_daily:
            return False
        return True

    def _regime_allows_entry(self, reading: BiasReading, direction: Literal["long", "short"]) -> bool:
        if reading.bias == "NEUTRAL":
            return False
        if reading.conflict_action == "SIT_OUT":
            return False
        if reading.regime == "RANGING":
            return False
        if direction == "long":
            if reading.bias not in ("MILD_LONG", "STRONG_LONG"):
                return False
            if reading.regime == "WEAK_TREND" and reading.bias != "STRONG_LONG":
                return False
        else:
            if reading.bias not in ("MILD_SHORT", "STRONG_SHORT"):
                return False
            if reading.regime == "WEAK_TREND" and reading.bias != "STRONG_SHORT":
                return False
        if reading.regime == "WEAK_TREND" and abs(float(reading.raw_score)) < self.cfg.weak_trend_min_raw - 1e-9:
            return False
        return True

    def _trend_15m_long(self) -> bool:
        e9, e21 = self._ema15_9.value, self._ema15_21.value
        p9, p21 = self._ema15_9.prev, self._ema15_21.prev
        if e9 is None or e21 is None or p9 is None or p21 is None:
            return False
        return e9 > e21 and (e9 - p9) > 0 and (e21 - p21) > 0

    def _trend_15m_short(self) -> bool:
        e9, e21 = self._ema15_9.value, self._ema15_21.value
        p9, p21 = self._ema15_9.prev, self._ema15_21.prev
        if e9 is None or e21 is None or p9 is None or p21 is None:
            return False
        return e9 < e21 and (e9 - p9) < 0 and (e21 - p21) < 0

    def _pullback_5m_long(self, row: _Row) -> bool:
        e21 = self._ema5_21.value
        if e21 is None:
            return False
        return row.l <= float(e21)

    def _pullback_5m_short(self, row: _Row) -> bool:
        e21 = self._ema5_21.value
        if e21 is None:
            return False
        return row.h >= float(e21)

    def _level_context_long(self, row: _Row) -> bool:
        pip = self.cfg.pip_size
        max_d = self.cfg.level_proximity_pips * pip
        for lvl, typ in self._structural:
            if typ == "swing_low" and abs(row.l - float(lvl)) <= max_d:
                return True
        eff = self._effective_daily()
        if len(eff) >= 2:
            cur_i = len(eff) - 1
            pdh, pdl = compute_pdh_pdl(eff, cur_i)
            for lv in (pdh, pdl):
                if lv is not None and abs(row.c - float(lv)) <= max_d:
                    return True
        rounds = compute_round_levels(row.c, radius_pips=50.0, pip_size=pip)
        nearest, dist_p = find_nearest_level(row.c, rounds, pip_size=pip)
        if nearest is not None and dist_p is not None and dist_p <= self.cfg.level_proximity_pips:
            return True
        return False

    def _level_context_short(self, row: _Row) -> bool:
        pip = self.cfg.pip_size
        max_d = self.cfg.level_proximity_pips * pip
        for lvl, typ in self._structural:
            if typ == "swing_high" and abs(row.h - float(lvl)) <= max_d:
                return True
        eff = self._effective_daily()
        if len(eff) >= 2:
            cur_i = len(eff) - 1
            pdh, pdl = compute_pdh_pdl(eff, cur_i)
            for lv in (pdh, pdl):
                if lv is not None and abs(row.c - float(lv)) <= max_d:
                    return True
        rounds = compute_round_levels(row.c, radius_pips=50.0, pip_size=pip)
        nearest, dist_p = find_nearest_level(row.c, rounds, pip_size=pip)
        if nearest is not None and dist_p is not None and dist_p <= self.cfg.level_proximity_pips:
            return True
        return False

    def _position_units(self, reading: BiasReading) -> int:
        m = float(reading.size_multiplier)
        return int(round(self.cfg.base_position_units * m))

    def _entry_decision(self, current_bar: BarView, portfolio: PortfolioSnapshot) -> Signal | None:
        row = self._rows[current_bar.bar_index]
        reading = self._current_bias
        if reading is None:
            return None

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

        direction: Optional[Literal["long", "short"]] = None
        if (
            self._regime_allows_entry(reading, "long")
            and self._trend_15m_long()
            and self._pullback_5m_long(row)
            and self._level_context_long(row)
            and row.c > row.o
        ):
            direction = "long"
        elif (
            self._regime_allows_entry(reading, "short")
            and self._trend_15m_short()
            and self._pullback_5m_short(row)
            and self._level_context_short(row)
            and row.c < row.o
        ):
            direction = "short"
        else:
            return None

        pip = self.cfg.pip_size
        mid = row.c
        if direction == "long":
            sl = mid - self.cfg.stop_loss_pips * pip
            tp = mid + self.cfg.tp1_pips * pip
        else:
            sl = mid + self.cfg.stop_loss_pips * pip
            tp = mid - self.cfg.tp1_pips * pip

        sz = self._position_units(reading)
        return Signal(
            family=self.family_name,
            direction=direction,
            stop_loss=float(sl),
            take_profit=float(tp),
            size=int(sz),
            metadata={
                "stop_loss_pips": float(self.cfg.stop_loss_pips),
                "take_profit_pips": float(self.cfg.tp1_pips),
                "bias": str(reading.bias),
                "regime": str(reading.regime),
                "raw_score": float(reading.raw_score),
                "size_multiplier": float(reading.size_multiplier),
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
        self._sync_rows(current_bar, HistoricalDataView(current_bar._store, current_bar.bar_index))
        self._trades_today += 1
        pip = self.cfg.pip_size
        reading = self._current_bias
        self._entry_bias_sign = _bias_side(str(reading.bias)) if reading is not None else _bias_side(str(signal.metadata.get("bias", "NEUTRAL")))
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
        self._last_trail_15m_ts = self._completed_15m[-1].timestamp if self._completed_15m else None

    def on_position_closed(self, trade: ClosedTrade) -> None:
        if trade.family != self.family_name:
            return
        if trade.exit_reason == "stop_loss":
            self._cooldown_sl_exit_bar = int(trade.exit_bar)
        if int(getattr(trade, "remaining_units", 0) or 0) <= 0:
            self._pos = None
            self._last_trail_15m_ts = None
            self._entry_bias_sign = 0

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
            pip = self.cfg.pip_size
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
                    pip=pip,
                ),
                "tp2": tp2_price(
                    entry=float(position.entry_price),
                    direction=position.direction,
                    tp2_pips=self.cfg.tp2_pips,
                    pip=pip,
                ),
                "runner_tp": runner_dummy_take_profit(direction=position.direction),
            }
            self._pos = st

        row = self._rows[current_bar.bar_index]
        pip = self.cfg.pip_size
        direction = str(st["direction"])
        entry_bar = int(st["entry_bar"])
        bars_held = int(row.idx - entry_bar)
        buf = self.cfg.trail_buffer_pips * pip

        if _session_close_window_utc(row.ts):
            return ExitAction(
                reason="session_close",
                exit_type="full",
                close_fraction=1.0,
                price=float(current_bar.bid_close if direction == "long" else current_bar.ask_close),
            )

        if bars_held >= self.cfg.max_hold_bars:
            return ExitAction(
                reason="max_hold",
                exit_type="full",
                close_fraction=1.0,
                price=float(current_bar.bid_close if direction == "long" else current_bar.ask_close),
            )

        reading = self._current_bias
        if reading is not None and str(st.get("phase")) == "runner":
            cur_sign = _bias_side(str(reading.bias))
            eb = int(self._entry_bias_sign)
            if eb != 0 and cur_sign != 0 and cur_sign != eb:
                return ExitAction(
                    reason="bias_flip",
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
            new_15m_ts = self._completed_15m[-1].timestamp if self._completed_15m else None
            if new_15m_ts is not None and self._last_trail_15m_ts != new_15m_ts:
                self._last_trail_15m_ts = new_15m_ts
                cur_sl = float(position.stop_loss)
                if direction == "long":
                    swing = _latest_swing_low_15m(self._completed_15m)
                    if swing is None:
                        return None
                    raw_sl = float(swing) - buf
                    new_sl = max(cur_sl, raw_sl)
                    if new_sl <= cur_sl + 1e-12:
                        return None
                else:
                    swing = _latest_swing_high_15m(self._completed_15m)
                    if swing is None:
                        return None
                    raw_sl = float(swing) + buf
                    new_sl = min(cur_sl, raw_sl)
                    if new_sl >= cur_sl - 1e-12:
                        return None
                return ExitAction(
                    reason="trail_15m_swing",
                    exit_type="none",
                    close_fraction=1.0,
                    new_stop_loss=float(new_sl),
                )

        return None


CrossAssetConfluence = CrossAssetConfluenceStrategy
