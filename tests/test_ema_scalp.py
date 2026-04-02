from __future__ import annotations

import math
import types
from datetime import timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from datetime import timedelta

from core.regime_backtest_engine.ema_scalp import (
    EMAScalpConfig,
    EMAScalpStrategy,
    build_ema_scalp_run_config,
    ema_scalp_session_open_utc,
    tp1_price,
    verify_ema_sequence,
)
from core.regime_backtest_engine.engine import BacktestEngine
from core.regime_backtest_engine.models import PositionSnapshot
from core.regime_backtest_engine.strategy import BarView, HistoricalDataView, MarketDataStore
from core.regime_backtest_engine.synthetic_bars import Bar5M, _session_date_utc


FAMILY = "ema_scalp"
PIP = 0.01


def _store_from_df(df: pd.DataFrame) -> MarketDataStore:
    spread_pips = 1.2
    half = spread_pips * PIP / 2.0
    df = df.copy()
    df["mid_open"] = df["open"]
    df["mid_high"] = df["high"]
    df["mid_low"] = df["low"]
    df["mid_close"] = df["close"]
    df["bid_open"] = df["open"] - half
    df["bid_high"] = df["high"] - half
    df["bid_low"] = df["low"] - half
    df["bid_close"] = df["close"] - half
    df["ask_open"] = df["open"] + half
    df["ask_high"] = df["high"] + half
    df["ask_low"] = df["low"] + half
    df["ask_close"] = df["close"] + half
    df["spread_pips"] = spread_pips
    cols = [
        "timestamp",
        "bid_open",
        "bid_high",
        "bid_low",
        "bid_close",
        "ask_open",
        "ask_high",
        "ask_low",
        "ask_close",
        "mid_open",
        "mid_high",
        "mid_low",
        "mid_close",
        "spread_pips",
    ]
    return MarketDataStore({c: df[c].to_numpy() for c in cols})


def _session_rows(
    start: pd.Timestamp,
    n: int,
    *,
    price: float = 150.0,
) -> pd.DataFrame:
    rows = []
    for i in range(n):
        ts = start + pd.Timedelta(minutes=i)
        o = price
        c = price + 0.001
        rows.append({"timestamp": ts, "open": o, "high": c + 0.01, "low": o - 0.01, "close": c})
        price = c
    return pd.DataFrame(rows)


def _portfolio_empty() -> SimpleNamespace:
    return SimpleNamespace(open_positions=[])


def _warmup(s: EMAScalpStrategy, store: MarketDataStore, last_idx: int) -> None:
    for i in range(last_idx + 1):
        s.evaluate(BarView(store, i), HistoricalDataView(store, i), _portfolio_empty())


def _patch_bull_ema(s: EMAScalpStrategy) -> None:
    s._ema5_9._ema = 150.10
    s._ema5_9._prev = 150.05
    s._ema5_21._ema = 150.00
    s._ema5_21._prev = 149.95
    s._ema15_9._ema = 150.20
    s._ema15_21._ema = 149.90


def test_adapter_init_config_a() -> None:
    s = EMAScalpStrategy(FAMILY, EMAScalpConfig(variant="A"))
    assert s.cfg.tp1_pips == 4.0


def test_adapter_init_config_b() -> None:
    s = EMAScalpStrategy(FAMILY, EMAScalpConfig(variant="B"))
    assert s.cfg.tp1_pips == 8.0


def test_evaluate_valid_bar_no_crash() -> None:
    df = _session_rows(pd.Timestamp("2025-06-04T08:00:00Z"), 400)
    store = _store_from_df(df)
    s = EMAScalpStrategy(FAMILY, EMAScalpConfig(variant="A"))
    for idx in range(len(store)):
        out = s.evaluate(BarView(store, idx), HistoricalDataView(store, idx), _portfolio_empty())
        assert out is None or (out.family == FAMILY and out.direction in ("long", "short"))


def test_warmup_no_signal_before_315_bars() -> None:
    df = _session_rows(pd.Timestamp("2025-06-04T08:00:00Z"), 320)
    store = _store_from_df(df)
    s = EMAScalpStrategy(FAMILY, EMAScalpConfig(variant="A"))
    for idx in range(314):
        assert s.evaluate(BarView(store, idx), HistoricalDataView(store, idx), _portfolio_empty()) is None


@pytest.mark.parametrize(
    "ts,expect_open",
    [
        ("2025-06-04T06:59:00Z", False),
        ("2025-06-04T11:01:00Z", False),
        ("2025-06-04T11:59:00Z", False),
        ("2025-06-04T17:01:00Z", False),
        ("2025-06-04T07:00:00Z", True),
        ("2025-06-04T10:59:00Z", True),
        ("2025-06-04T12:00:00Z", True),
        ("2025-06-04T16:59:00Z", True),
    ],
)
def test_session_gate_utc(ts: str, expect_open: bool) -> None:
    t = pd.Timestamp(ts).to_pydatetime()
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    assert ema_scalp_session_open_utc(t) == expect_open


def test_ema_known_sequence_period3() -> None:
    vals = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0]
    out = verify_ema_sequence(vals, 3)
    assert math.isnan(out[0]) and math.isnan(out[1])
    k = 2.0 / 4.0
    sma = sum(vals[:3]) / 3.0
    assert out[2] == pytest.approx(sma)
    assert out[3] == pytest.approx((vals[3] - sma) * k + sma)
    assert out[4] == pytest.approx((vals[4] - out[3]) * k + out[3])


def test_flat_ema_slopes_no_signal() -> None:
    df = _session_rows(pd.Timestamp("2025-06-04T08:00:00Z"), 400)
    for i in range(len(df)):
        df.loc[i, ["open", "high", "low", "close"]] = 150.0
    store = _store_from_df(df)
    s = EMAScalpStrategy(FAMILY, EMAScalpConfig(variant="A"))
    idx = len(store) - 1
    _warmup(s, store, idx)
    assert s.evaluate(BarView(store, idx), HistoricalDataView(store, idx), _portfolio_empty()) is None


def test_standard_pullback_long_touch_generates_signal() -> None:
    start = pd.Timestamp("2025-06-04T08:00:00Z")
    df = _session_rows(start, 320)
    df.loc[319, "low"] = 149.99
    df.loc[319, "close"] = 150.05
    df.loc[319, "open"] = 150.02
    df.loc[319, "high"] = 150.06
    store = _store_from_df(df)
    s = EMAScalpStrategy(FAMILY, EMAScalpConfig(variant="A"))
    _warmup(s, store, 318)
    s._sync_rows(BarView(store, 319), HistoricalDataView(store, 319))
    _patch_bull_ema(s)
    sig = s._entry_decision(BarView(store, 319), _portfolio_empty())
    assert sig is not None and sig.direction == "long"


def test_standard_pullback_no_touch_no_signal() -> None:
    df = _session_rows(pd.Timestamp("2025-06-04T08:00:00Z"), 320)
    df.loc[319, "low"] = 150.05
    df.loc[319, "close"] = 150.08
    store = _store_from_df(df)
    s = EMAScalpStrategy(FAMILY, EMAScalpConfig(variant="A"))
    _warmup(s, store, 318)
    s._sync_rows(BarView(store, 319), HistoricalDataView(store, 319))
    _patch_bull_ema(s)
    assert s._entry_decision(BarView(store, 319), _portfolio_empty()) is None


def test_15m_misaligned_blocks_long() -> None:
    df = _session_rows(pd.Timestamp("2025-06-04T08:00:00Z"), 320)
    df.loc[319, "low"] = 149.99
    df.loc[319, "close"] = 150.05
    store = _store_from_df(df)
    s = EMAScalpStrategy(FAMILY, EMAScalpConfig(variant="A"))
    _warmup(s, store, 318)
    s._sync_rows(BarView(store, 319), HistoricalDataView(store, 319))
    s._ema5_9._ema = 150.10
    s._ema5_9._prev = 150.05
    s._ema5_21._ema = 150.00
    s._ema5_21._prev = 149.95
    s._ema15_9._ema = 149.80
    s._ema15_21._ema = 149.90
    assert s._entry_decision(BarView(store, 319), _portfolio_empty()) is None


def test_pdh_filter_blocks_long_near_ceiling() -> None:
    df = _session_rows(pd.Timestamp("2025-06-04T08:00:00Z"), 320)
    df.loc[319, "low"] = 149.99
    df.loc[319, "close"] = 150.05
    store = _store_from_df(df)
    s = EMAScalpStrategy(FAMILY, EMAScalpConfig(variant="A"))
    _warmup(s, store, 318)
    s._sync_rows(BarView(store, 319), HistoricalDataView(store, 319))
    _patch_bull_ema(s)
    s._update_pdh_pdl_breakout = lambda _c: None  # type: ignore[method-assign]
    s._last_pdh = 150.15
    s._above_pdh_streak = 0
    assert s._entry_decision(BarView(store, 319), _portfolio_empty()) is None


def test_pdh_breakout_allows_long() -> None:
    df = _session_rows(pd.Timestamp("2025-06-04T08:00:00Z"), 320)
    df.loc[319, "low"] = 149.99
    df.loc[319, "close"] = 150.05
    store = _store_from_df(df)
    s = EMAScalpStrategy(FAMILY, EMAScalpConfig(variant="A"))
    _warmup(s, store, 318)
    s._sync_rows(BarView(store, 319), HistoricalDataView(store, 319))
    _patch_bull_ema(s)
    s._last_pdh = 150.15
    s._above_pdh_streak = 5
    sig = s._entry_decision(BarView(store, 319), _portfolio_empty())
    assert sig is not None


def test_cooldown_after_sl() -> None:
    start = pd.Timestamp("2025-06-02T08:00:00Z")
    df = _session_rows(start, 400)
    for i in range(315, 321):
        df.loc[i, ["open", "high", "low", "close"]] = 150.0
    for i in range(321, 400):
        df.loc[i, "low"] = 149.99
        df.loc[i, "close"] = 150.05
    store = _store_from_df(df)
    s = EMAScalpStrategy(FAMILY, EMAScalpConfig(variant="A"))
    s._update_pdh_pdl_breakout = types.MethodType(lambda self, _mc: None, s)
    p = _portfolio_empty()
    s._cooldown_sl_exit_bar = 320
    for i in range(326):
        if i > 320:
            _patch_bull_ema(s)
        s.evaluate(BarView(store, i), HistoricalDataView(store, i), p)
    assert s._entry_decision(BarView(store, 325), p) is None
    for i in range(326, 331):
        if i > 320:
            _patch_bull_ema(s)
        s.evaluate(BarView(store, i), HistoricalDataView(store, i), p)
    _patch_bull_ema(s)
    assert s._entry_decision(BarView(store, 330), p) is not None


def test_daily_trade_limit_blocks() -> None:
    df = _session_rows(pd.Timestamp("2025-06-04T08:00:00Z"), 350)
    df.loc[320, "low"] = 149.99
    df.loc[320, "close"] = 150.05
    store = _store_from_df(df)
    s = EMAScalpStrategy(FAMILY, EMAScalpConfig(variant="A"))
    _warmup(s, store, 319)
    s._sync_rows(BarView(store, 320), HistoricalDataView(store, 320))
    ts = pd.Timestamp(df.loc[320, "timestamp"]).to_pydatetime()
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    s._trade_day = _session_date_utc(ts, 22)
    s._trades_today = 10
    _patch_bull_ema(s)
    assert s._entry_decision(BarView(store, 320), _portfolio_empty()) is None


def test_tp1_tp2_prices() -> None:
    assert tp1_price(entry=150.0, direction="long", tp1_pips=4.0, pip=PIP) == pytest.approx(150.04)
    assert tp1_price(entry=150.0, direction="long", tp1_pips=8.0, pip=PIP) == pytest.approx(150.08)


def test_tp1_partial_remaining_units() -> None:
    assert int(200_000 * 0.5) == 100_000


def test_trail_ratchet_long_moves_up_only() -> None:
    df = _session_rows(pd.Timestamp("2025-06-04T08:00:00Z"), 400)
    store = _store_from_df(df)
    s = EMAScalpStrategy(FAMILY, EMAScalpConfig(variant="A"))
    _warmup(s, store, 339)
    assert s._completed_5m, "need completed 5m bars"
    ts0 = s._completed_5m[-1].timestamp
    if hasattr(ts0, "to_pydatetime"):
        ts0 = ts0.to_pydatetime()
    ts1 = ts0 + timedelta(minutes=5)
    s._completed_5m.append(
        Bar5M(ts1, 150.0, 150.2, 149.9, 150.1, None, 300, 304, True)
    )
    s._pos = {
        "entry_bar": 300,
        "entry_price": 150.0,
        "direction": "long",
        "phase": "runner",
        "initial_size": 50_000,
        "tp1": 150.04,
        "tp2": 150.07,
        "runner_tp": 1e9,
    }
    s._ema5_9._ema = 150.50
    s._last_trail_5m_ts = None
    pos = PositionSnapshot(
        trade_id=1,
        family=FAMILY,
        direction="long",
        entry_price=150.0,
        entry_bar=300,
        size=50_000,
        margin_held=100.0,
        stop_loss=149.85,
        take_profit=1e9,
        unrealized_pnl=0.0,
    )
    b1 = BarView(store, 339)
    h1 = HistoricalDataView(store, 339)
    a1 = s.get_exit_conditions(pos, b1, h1)
    assert a1 is not None and a1.new_stop_loss is not None
    assert a1.new_stop_loss > pos.stop_loss


def test_max_hold_exit_config_a_and_b() -> None:
    df = _session_rows(pd.Timestamp("2025-06-04T08:00:00Z"), 200)
    store = _store_from_df(df)
    s_a = EMAScalpStrategy(FAMILY, EMAScalpConfig(variant="A"))
    _warmup(s_a, store, 120)
    s_a._pos = {
        "entry_bar": 60,
        "entry_price": 150.0,
        "direction": "long",
        "phase": "runner",
        "initial_size": 10_000,
        "tp1": 150.04,
        "tp2": 150.07,
        "runner_tp": 1e9,
    }
    pos = PositionSnapshot(
        trade_id=1,
        family=FAMILY,
        direction="long",
        entry_price=150.0,
        entry_bar=60,
        size=10_000,
        margin_held=1.0,
        stop_loss=149.0,
        take_profit=None,
        unrealized_pnl=0.0,
    )
    ex = s_a.get_exit_conditions(pos, BarView(store, 120), HistoricalDataView(store, 120))
    assert ex is not None and ex.reason == "max_hold"

    s_b = EMAScalpStrategy(FAMILY, EMAScalpConfig(variant="B"))
    _warmup(s_b, store, 150)
    s_b._pos = dict(s_a._pos)
    s_b._pos["entry_bar"] = 60
    ex_b = s_b.get_exit_conditions(pos, BarView(store, 150), HistoricalDataView(store, 150))
    assert ex_b is not None and ex_b.reason == "max_hold"


def test_session_close_london() -> None:
    rows = []
    base = pd.Timestamp("2025-06-04T10:54:00Z")
    for i in range(3):
        tsi = base + pd.Timedelta(minutes=i)
        p = 150.0
        rows.append({"timestamp": tsi, "open": p, "high": p + 0.02, "low": p - 0.02, "close": p})
    df = pd.DataFrame(rows)
    store = _store_from_df(df)
    s = EMAScalpStrategy(FAMILY, EMAScalpConfig(variant="A"))
    _warmup(s, store, 1)
    s._pos = {
        "entry_bar": 0,
        "entry_price": 150.0,
        "direction": "long",
        "phase": "runner",
        "initial_size": 10_000,
        "tp1": 150.04,
        "tp2": 150.07,
        "runner_tp": 1e9,
    }
    ex = s.get_exit_conditions(
        PositionSnapshot(
            trade_id=1,
            family=FAMILY,
            direction="long",
            entry_price=150.0,
            entry_bar=0,
            size=10_000,
            margin_held=1.0,
            stop_loss=149.0,
            take_profit=None,
            unrealized_pnl=0.0,
        ),
        BarView(store, 1),
        HistoricalDataView(store, 1),
    )
    assert ex is not None and ex.exit_type == "full" and ex.reason == "session_close"


def test_deep_pullback_requires_three_15m_alignment() -> None:
    df = _session_rows(pd.Timestamp("2025-06-04T08:00:00Z"), 320)
    df.loc[319, "low"] = 149.5
    df.loc[319, "close"] = 150.10
    store = _store_from_df(df)
    s = EMAScalpStrategy(FAMILY, EMAScalpConfig(variant="A"))
    _warmup(s, store, 318)
    s._sync_rows(BarView(store, 319), HistoricalDataView(store, 319))
    s._align15[:] = [True, True]
    s._ema5_9._ema = 150.20
    s._ema5_21._ema = 150.00
    s._ema5_27._ema = 149.90
    s._ema5_33._ema = 149.85
    assert not s._deep_long(s._rows[-1])


def test_deep_pullback_three_aligned_can_signal() -> None:
    df = _session_rows(pd.Timestamp("2025-06-04T08:00:00Z"), 320)
    df.loc[319, "low"] = 149.5
    df.loc[319, "close"] = 150.10
    store = _store_from_df(df)
    s = EMAScalpStrategy(FAMILY, EMAScalpConfig(variant="A"))
    _warmup(s, store, 318)
    s._sync_rows(BarView(store, 319), HistoricalDataView(store, 319))
    s._align15[:] = [True, True, True]
    s._ema5_9._ema = 150.20
    s._ema5_21._ema = 150.00
    s._ema5_27._ema = 149.90
    s._ema5_33._ema = 149.85
    assert s._deep_long(s._rows[-1])


def test_causal_evaluate_matches_truncated_history() -> None:
    start = pd.Timestamp("2025-06-04T08:00:00Z")
    df_full = _session_rows(start, 500)
    n = 350
    df_trunc = df_full.iloc[: n + 1].copy().reset_index(drop=True)
    store_full = _store_from_df(df_full)
    store_t = _store_from_df(df_trunc)
    p = _portfolio_empty()

    s1 = EMAScalpStrategy(FAMILY, EMAScalpConfig(variant="A"))
    sig1 = None
    for i in range(n + 1):
        sig1 = s1.evaluate(BarView(store_full, i), HistoricalDataView(store_full, i), p)

    s2 = EMAScalpStrategy(FAMILY, EMAScalpConfig(variant="A"))
    sig2 = None
    for i in range(n + 1):
        sig2 = s2.evaluate(BarView(store_t, i), HistoricalDataView(store_t, i), p)

    assert (sig1 is None) == (sig2 is None)
    if sig1 is not None and sig2 is not None:
        assert sig1.direction == sig2.direction


def test_build_run_config_rejects_nonpositive_spread() -> None:
    with pytest.raises(ValueError):
        build_ema_scalp_run_config(
            hypothesis="x",
            data_path=Path("/tmp/x.csv"),
            output_dir=Path("/tmp/o"),
            family_name=FAMILY,
            variant="A",
            spread_pips=0.0,
            slippage_pips=0.0,
        )


def test_engine_smoke_run(tmp_path: Path) -> None:
    df = _session_rows(pd.Timestamp("2025-06-04T08:00:00Z"), 400)
    path = tmp_path / "e.csv"
    df.to_csv(path, index=False)
    cfg = build_ema_scalp_run_config(
        hypothesis="ema_smoke",
        data_path=path,
        output_dir=tmp_path / "out_e",
        family_name=FAMILY,
        variant="A",
        spread_pips=1.2,
        slippage_pips=0.0,
    )
    engine = BacktestEngine({FAMILY: EMAScalpStrategy(FAMILY, EMAScalpConfig(variant="A"))})
    result = engine.run(cfg)
    assert "net_pnl_usd" in result.summary
    assert (tmp_path / "out_e" / "summary.json").exists()
