from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from core.regime_backtest_engine.cross_asset_bias import BiasReading
from core.regime_backtest_engine.cross_asset_confluence import (
    CrossAssetConfluenceConfig,
    CrossAssetConfluenceStrategy,
    CrossAssetConfluence,
    _in_trading_session_utc,
    _latest_swing_low_15m,
    _session_close_window_utc,
)
from core.regime_backtest_engine.cross_asset_data import CrossAssetDataLoader
from core.regime_backtest_engine.models import PortfolioSnapshot, PositionSnapshot
from core.regime_backtest_engine.strategy import BarView, HistoricalDataView, MarketDataStore
from core.regime_backtest_engine.synthetic_bars import Bar15M

UTC = timezone.utc
FAMILY = "cross_asset_confluence"
PIP = 0.01


def _reading(
    *,
    bias: str = "STRONG_LONG",
    regime: str = "TRENDING",
    raw: float = 2.5,
    conflict: str = "FULL",
    adx: float = 28.0,
    size_mult: float = 1.0,
) -> BiasReading:
    ts = datetime(2024, 6, 4, 8, 0, tzinfo=UTC)
    return BiasReading(
        timestamp=ts,
        oil_signal=1.0,
        dxy_signal=0.5,
        gold_signal=-0.5,
        silver_signal=-0.5,
        raw_score=raw,
        bias=bias,
        oil_dxy_agree=True,
        conflict_action=conflict,
        adx_value=adx,
        adxr_value=adx,
        regime=regime,
        size_multiplier=size_mult,
        oil_sma_20=70.0,
        eurusd_sma_20=1.1,
        gold_sma_20=2000.0,
        silver_sma_20=25.0,
        brent_is_big_move=False,
    )


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


def _session_rows(start: pd.Timestamp, n: int, *, price: float = 150.0) -> pd.DataFrame:
    rows = []
    for i in range(n):
        ts = start + pd.Timedelta(minutes=i)
        o = price
        c = price + 0.002
        rows.append({"timestamp": ts, "open": o, "high": c + 0.02, "low": o - 0.02, "close": c})
        price = c
    return pd.DataFrame(rows)


def _write_cross_bundle(root: Path, *, h1_periods: int = 80, d_periods: int = 40) -> None:
    import numpy as np

    root.mkdir(parents=True, exist_ok=True)
    h1_times = pd.date_range("2023-06-20", periods=h1_periods, freq="h", tz="UTC")
    d_times = pd.date_range("2023-06-20", periods=d_periods, freq="D", tz="UTC")
    for name, times in [
        ("BCO_USD_H1_OANDA.csv", h1_times),
        ("EUR_USD_H1_OANDA.csv", h1_times),
    ]:
        df = pd.DataFrame(
            {
                "timestamp": times.strftime("%Y-%m-%d %H:%M:%S"),
                "open": np.linspace(80.0, 85.0, len(times)),
                "high": np.linspace(81.0, 86.0, len(times)),
                "low": np.linspace(79.0, 84.0, len(times)),
                "close": np.linspace(80.5, 85.5, len(times)),
                "volume": 1000,
            }
        )
        df.to_csv(root / name, index=False)
    for name, times in [
        ("XAU_USD_D_OANDA.csv", d_times),
        ("XAG_USD_D_OANDA.csv", d_times),
    ]:
        df = pd.DataFrame(
            {
                "timestamp": times.strftime("%Y-%m-%d %H:%M:%S"),
                "open": np.linspace(1900.0, 1950.0, len(times)),
                "high": np.linspace(1910.0, 1960.0, len(times)),
                "low": np.linspace(1890.0, 1940.0, len(times)),
                "close": np.linspace(1920.0, 1955.0, len(times)),
                "volume": 500,
            }
        )
        df.to_csv(root / name, index=False)


def _write_min_usdjpy(path: Path, n: int) -> None:
    import numpy as np

    start = pd.Timestamp("2023-06-20T00:00:00Z")
    rows = []
    p = 150.0
    for i in range(n):
        ts = start + pd.Timedelta(minutes=i)
        o = p
        c = p + 0.001
        rows.append(
            {
                "time": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "open": o,
                "high": c + 0.01,
                "low": o - 0.01,
                "close": c,
                "spread_pips": 1.0,
            }
        )
        p = c
    pd.DataFrame(rows).to_csv(path, index=False)


def _loader(tmp_path) -> CrossAssetDataLoader:
    cross = tmp_path / "cross"
    _write_cross_bundle(cross)
    uj = tmp_path / "uj.csv"
    _write_min_usdjpy(uj, 50_000)
    return CrossAssetDataLoader(
        usdjpy_path=str(uj),
        brent_path=str(cross / "BCO_USD_H1_OANDA.csv"),
        eurusd_path=str(cross / "EUR_USD_H1_OANDA.csv"),
        gold_path=str(cross / "XAU_USD_D_OANDA.csv"),
        silver_path=str(cross / "XAG_USD_D_OANDA.csv"),
    )


def _portfolio_empty() -> PortfolioSnapshot:
    return PortfolioSnapshot(
        balance=100_000.0,
        equity=100_000.0,
        unrealized_pnl=0.0,
        margin_used=0.0,
        available_margin=100_000.0,
        open_positions=(),
    )


def _portfolio_with_open() -> PortfolioSnapshot:
    pos = PositionSnapshot(
        trade_id=1,
        family=FAMILY,
        direction="long",
        entry_price=150.0,
        entry_bar=0,
        size=200_000,
        margin_held=100.0,
        stop_loss=149.5,
        take_profit=151.0,
        unrealized_pnl=0.0,
    )
    return PortfolioSnapshot(
        balance=100_000.0,
        equity=100_000.0,
        unrealized_pnl=0.0,
        margin_used=100.0,
        available_margin=99_900.0,
        open_positions=(pos,),
    )


class TunableCrossAssetConfluence(CrossAssetConfluenceStrategy):
    """Test hook: pin bias reading without loader token churn."""

    def __init__(self, *args, pinned_reading: BiasReading | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._pinned_reading = pinned_reading

    def _maybe_refresh_bias(self, ts: datetime) -> None:
        if self._pinned_reading is not None:
            self._current_bias = self._pinned_reading
        else:
            super()._maybe_refresh_bias(ts)


@pytest.fixture()
def tiny_cfg() -> CrossAssetConfluenceConfig:
    return CrossAssetConfluenceConfig(
        warmup_1m_bars=50,
        min_completed_1h=2,
        min_completed_15m=2,
        min_completed_daily=2,
        swing_lookback_1h=10,
    )


def test_adapter_initializes(tmp_path) -> None:
    s = CrossAssetConfluenceStrategy(_loader(tmp_path))
    assert s.family_name == FAMILY


def test_cross_asset_confluence_alias() -> None:
    assert CrossAssetConfluence is CrossAssetConfluenceStrategy


def test_evaluate_returns_signal_or_none_no_crash(tmp_path, tiny_cfg) -> None:
    df = _session_rows(pd.Timestamp("2024-06-04T08:00:00Z"), 120)
    store = _store_from_df(df)
    s = CrossAssetConfluenceStrategy(_loader(tmp_path), tiny_cfg)
    for idx in range(len(store)):
        out = s.evaluate(BarView(store, idx), HistoricalDataView(store, idx), _portfolio_empty())
        assert out is None or (out.family == FAMILY and out.direction in ("long", "short"))


def test_no_signal_during_warmup_config_40000(tmp_path) -> None:
    cfg = CrossAssetConfluenceConfig(warmup_1m_bars=40_000, min_completed_1h=1, min_completed_15m=1, min_completed_daily=1)
    df = _session_rows(pd.Timestamp("2024-06-04T08:00:00Z"), 500)
    store = _store_from_df(df)
    s = TunableCrossAssetConfluence(_loader(tmp_path), cfg, pinned_reading=_reading())
    for idx in range(len(store)):
        assert s.evaluate(BarView(store, idx), HistoricalDataView(store, idx), _portfolio_empty()) is None


@pytest.mark.parametrize(
    "ts,expect",
    [
        ("2024-06-04T06:30:00Z", False),
        ("2024-06-04T11:30:00Z", False),
        ("2024-06-04T07:00:00Z", True),
        ("2024-06-04T12:00:00Z", True),
    ],
)
def test_session_gate(ts: str, expect: bool) -> None:
    t = pd.Timestamp(ts).to_pydatetime().replace(tzinfo=UTC)
    assert _in_trading_session_utc(t) is expect


def test_session_close_window() -> None:
    t = pd.Timestamp("2024-06-04T10:56:00Z").to_pydatetime().replace(tzinfo=UTC)
    assert _session_close_window_utc(t) is True


def test_regime_ranging_blocks_signal(tmp_path, tiny_cfg) -> None:
    df = _session_rows(pd.Timestamp("2024-06-04T08:00:00Z"), 80)
    store = _store_from_df(df)
    s = TunableCrossAssetConfluence(_loader(tmp_path), tiny_cfg, pinned_reading=_reading(regime="RANGING", adx=15.0))
    with patch.object(s, "_warmup_ok", return_value=True), patch.object(s, "_trend_15m_long", return_value=True), patch.object(
        s, "_pullback_5m_long", return_value=True
    ), patch.object(s, "_level_context_long", return_value=True):
        idx = len(store) - 1
        assert s.evaluate(BarView(store, idx), HistoricalDataView(store, idx), _portfolio_empty()) is None


def test_regime_trending_mild_long_allows_with_mocks(tmp_path, tiny_cfg) -> None:
    df = _session_rows(pd.Timestamp("2024-06-04T08:00:00Z"), 80)
    store = _store_from_df(df)
    s = TunableCrossAssetConfluence(_loader(tmp_path), tiny_cfg, pinned_reading=_reading(bias="MILD_LONG", raw=1.0, regime="TRENDING"))
    with patch.object(s, "_warmup_ok", return_value=True), patch.object(s, "_trend_15m_long", return_value=True), patch.object(
        s, "_pullback_5m_long", return_value=True
    ), patch.object(s, "_level_context_long", return_value=True):
        idx = len(store) - 1
        sig = s.evaluate(BarView(store, idx), HistoricalDataView(store, idx), _portfolio_empty())
        assert sig is not None and sig.direction == "long"


def test_regime_weak_mild_blocks(tmp_path, tiny_cfg) -> None:
    df = _session_rows(pd.Timestamp("2024-06-04T08:00:00Z"), 80)
    store = _store_from_df(df)
    s = TunableCrossAssetConfluence(_loader(tmp_path), tiny_cfg, pinned_reading=_reading(bias="MILD_LONG", raw=1.0, regime="WEAK_TREND", adx=22.0))
    with patch.object(s, "_warmup_ok", return_value=True), patch.object(s, "_trend_15m_long", return_value=True), patch.object(
        s, "_pullback_5m_long", return_value=True
    ), patch.object(s, "_level_context_long", return_value=True):
        idx = len(store) - 1
        assert s.evaluate(BarView(store, idx), HistoricalDataView(store, idx), _portfolio_empty()) is None


def test_regime_weak_strong_allows(tmp_path, tiny_cfg) -> None:
    df = _session_rows(pd.Timestamp("2024-06-04T08:00:00Z"), 80)
    store = _store_from_df(df)
    s = TunableCrossAssetConfluence(_loader(tmp_path), tiny_cfg, pinned_reading=_reading(bias="STRONG_LONG", raw=2.5, regime="WEAK_TREND", adx=22.0))
    with patch.object(s, "_warmup_ok", return_value=True), patch.object(s, "_trend_15m_long", return_value=True), patch.object(
        s, "_pullback_5m_long", return_value=True
    ), patch.object(s, "_level_context_long", return_value=True):
        idx = len(store) - 1
        sig = s.evaluate(BarView(store, idx), HistoricalDataView(store, idx), _portfolio_empty())
        assert sig is not None


def test_bias_neutral_blocks(tmp_path, tiny_cfg) -> None:
    df = _session_rows(pd.Timestamp("2024-06-04T08:00:00Z"), 80)
    store = _store_from_df(df)
    s = TunableCrossAssetConfluence(_loader(tmp_path), tiny_cfg, pinned_reading=_reading(bias="NEUTRAL", raw=0.0, regime="TRENDING"))
    with patch.object(s, "_warmup_ok", return_value=True), patch.object(s, "_trend_15m_long", return_value=True), patch.object(
        s, "_pullback_5m_long", return_value=True
    ), patch.object(s, "_level_context_long", return_value=True):
        idx = len(store) - 1
        assert s.evaluate(BarView(store, idx), HistoricalDataView(store, idx), _portfolio_empty()) is None


def test_conflict_sit_out_blocks(tmp_path, tiny_cfg) -> None:
    df = _session_rows(pd.Timestamp("2024-06-04T08:00:00Z"), 80)
    store = _store_from_df(df)
    s = TunableCrossAssetConfluence(
        _loader(tmp_path), tiny_cfg, pinned_reading=_reading(bias="STRONG_LONG", conflict="SIT_OUT")
    )
    with patch.object(s, "_warmup_ok", return_value=True), patch.object(s, "_trend_15m_long", return_value=True), patch.object(
        s, "_pullback_5m_long", return_value=True
    ), patch.object(s, "_level_context_long", return_value=True):
        idx = len(store) - 1
        assert s.evaluate(BarView(store, idx), HistoricalDataView(store, idx), _portfolio_empty()) is None


def test_15m_bearish_blocks_long(tmp_path, tiny_cfg) -> None:
    df = _session_rows(pd.Timestamp("2024-06-04T08:00:00Z"), 80)
    store = _store_from_df(df)
    s = TunableCrossAssetConfluence(_loader(tmp_path), tiny_cfg, pinned_reading=_reading())
    with patch.object(s, "_warmup_ok", return_value=True), patch.object(s, "_trend_15m_long", return_value=False), patch.object(
        s, "_pullback_5m_long", return_value=True
    ), patch.object(s, "_level_context_long", return_value=True):
        idx = len(store) - 1
        assert s.evaluate(BarView(store, idx), HistoricalDataView(store, idx), _portfolio_empty()) is None


def test_level_not_near_blocks(tmp_path, tiny_cfg) -> None:
    df = _session_rows(pd.Timestamp("2024-06-04T08:00:00Z"), 80)
    store = _store_from_df(df)
    s = TunableCrossAssetConfluence(_loader(tmp_path), tiny_cfg, pinned_reading=_reading())
    with patch.object(s, "_warmup_ok", return_value=True), patch.object(s, "_trend_15m_long", return_value=True), patch.object(
        s, "_pullback_5m_long", return_value=True
    ), patch.object(s, "_level_context_long", return_value=False):
        idx = len(store) - 1
        assert s.evaluate(BarView(store, idx), HistoricalDataView(store, idx), _portfolio_empty()) is None


def test_level_near_round_allows(tmp_path, tiny_cfg) -> None:
    start = pd.Timestamp("2024-06-04T08:00:00Z")
    rows = []
    p = 150.0
    for i in range(80):
        ts = start + pd.Timedelta(minutes=i)
        o, c = 150.001, 150.003
        rows.append({"timestamp": ts, "open": o, "high": c + 0.01, "low": o - 0.01, "close": c})
    df = pd.DataFrame(rows)
    store = _store_from_df(df)
    s = TunableCrossAssetConfluence(_loader(tmp_path), tiny_cfg, pinned_reading=_reading())
    with patch.object(s, "_warmup_ok", return_value=True), patch.object(s, "_trend_15m_long", return_value=True), patch.object(
        s, "_pullback_5m_long", return_value=True
    ):
        idx = len(store) - 1
        sig = s.evaluate(BarView(store, idx), HistoricalDataView(store, idx), _portfolio_empty())
        assert sig is not None


def test_1m_rejection_bearish_blocks_long(tmp_path, tiny_cfg) -> None:
    start = pd.Timestamp("2024-06-04T08:00:00Z")
    rows = []
    for i in range(80):
        ts = start + pd.Timedelta(minutes=i)
        o, c = 150.02, 150.01
        rows.append({"timestamp": ts, "open": o, "high": o + 0.01, "low": c - 0.01, "close": c})
    store = _store_from_df(pd.DataFrame(rows))
    s = TunableCrossAssetConfluence(_loader(tmp_path), tiny_cfg, pinned_reading=_reading())
    with patch.object(s, "_warmup_ok", return_value=True), patch.object(s, "_trend_15m_long", return_value=True), patch.object(
        s, "_pullback_5m_long", return_value=True
    ), patch.object(s, "_level_context_long", return_value=True):
        idx = len(store) - 1
        assert s.evaluate(BarView(store, idx), HistoricalDataView(store, idx), _portfolio_empty()) is None


def test_1m_rejection_bullish_allows(tmp_path, tiny_cfg) -> None:
    df = _session_rows(pd.Timestamp("2024-06-04T08:00:00Z"), 80)
    store = _store_from_df(df)
    s = TunableCrossAssetConfluence(_loader(tmp_path), tiny_cfg, pinned_reading=_reading())
    with patch.object(s, "_warmup_ok", return_value=True), patch.object(s, "_trend_15m_long", return_value=True), patch.object(
        s, "_pullback_5m_long", return_value=True
    ), patch.object(s, "_level_context_long", return_value=True):
        idx = len(store) - 1
        assert s.evaluate(BarView(store, idx), HistoricalDataView(store, idx), _portfolio_empty()) is not None


def test_cooldown_after_sl(tmp_path, tiny_cfg) -> None:
    df = _session_rows(pd.Timestamp("2024-06-04T08:00:00Z"), 80)
    store = _store_from_df(df)
    s = TunableCrossAssetConfluence(_loader(tmp_path), tiny_cfg, pinned_reading=_reading())
    s._cooldown_sl_exit_bar = 40
    with patch.object(s, "_warmup_ok", return_value=True), patch.object(s, "_trend_15m_long", return_value=True), patch.object(
        s, "_pullback_5m_long", return_value=True
    ), patch.object(s, "_level_context_long", return_value=True):
        for idx in range(41, 56):
            assert s.evaluate(BarView(store, idx), HistoricalDataView(store, idx), _portfolio_empty()) is None


def test_max_daily_trades(tmp_path, tiny_cfg) -> None:
    df = _session_rows(pd.Timestamp("2024-06-04T08:00:00Z"), 80)
    store = _store_from_df(df)
    s = TunableCrossAssetConfluence(_loader(tmp_path), tiny_cfg, pinned_reading=_reading())
    s._trades_today = 8
    from core.regime_backtest_engine.synthetic_bars import _session_date_utc

    s._trade_day = _session_date_utc(df["timestamp"].iloc[-1].to_pydatetime().replace(tzinfo=UTC), tiny_cfg.day_boundary_utc_hour)
    with patch.object(s, "_warmup_ok", return_value=True), patch.object(s, "_trend_15m_long", return_value=True), patch.object(
        s, "_pullback_5m_long", return_value=True
    ), patch.object(s, "_level_context_long", return_value=True):
        idx = len(store) - 1
        assert s.evaluate(BarView(store, idx), HistoricalDataView(store, idx), _portfolio_empty()) is None


def test_max_concurrent_one(tmp_path, tiny_cfg) -> None:
    df = _session_rows(pd.Timestamp("2024-06-04T08:00:00Z"), 80)
    store = _store_from_df(df)
    s = TunableCrossAssetConfluence(_loader(tmp_path), tiny_cfg, pinned_reading=_reading())
    with patch.object(s, "_warmup_ok", return_value=True):
        idx = len(store) - 1
        assert s.evaluate(BarView(store, idx), HistoricalDataView(store, idx), _portfolio_with_open()) is None


def test_stop_loss_20_pips(tmp_path, tiny_cfg) -> None:
    df = _session_rows(pd.Timestamp("2024-06-04T08:00:00Z"), 80)
    store = _store_from_df(df)
    s = TunableCrossAssetConfluence(_loader(tmp_path), tiny_cfg, pinned_reading=_reading())
    with patch.object(s, "_warmup_ok", return_value=True), patch.object(s, "_trend_15m_long", return_value=True), patch.object(
        s, "_pullback_5m_long", return_value=True
    ), patch.object(s, "_level_context_long", return_value=True):
        idx = len(store) - 1
        sig = s.evaluate(BarView(store, idx), HistoricalDataView(store, idx), _portfolio_empty())
        assert sig is not None
        mid = float(df["close"].iloc[idx])
        assert abs(mid - float(sig.stop_loss)) / PIP == pytest.approx(20.0)


def test_tp1_metadata_10_pips(tmp_path, tiny_cfg) -> None:
    df = _session_rows(pd.Timestamp("2024-06-04T08:00:00Z"), 80)
    store = _store_from_df(df)
    s = TunableCrossAssetConfluence(_loader(tmp_path), tiny_cfg, pinned_reading=_reading())
    with patch.object(s, "_warmup_ok", return_value=True), patch.object(s, "_trend_15m_long", return_value=True), patch.object(
        s, "_pullback_5m_long", return_value=True
    ), patch.object(s, "_level_context_long", return_value=True):
        idx = len(store) - 1
        sig = s.evaluate(BarView(store, idx), HistoricalDataView(store, idx), _portfolio_empty())
        assert float(sig.metadata["take_profit_pips"]) == 10.0
        mid = float(df["close"].iloc[idx])
        assert abs(float(sig.take_profit) - mid) / PIP == pytest.approx(10.0)


def test_tp2_price_distance(tmp_path, tiny_cfg) -> None:
    from core.regime_backtest_engine.ema_scalp import tp2_price

    entry = 150.0
    tp2 = tp2_price(entry=entry, direction="long", tp2_pips=tiny_cfg.tp2_pips, pip=tiny_cfg.pip_size)
    assert abs(tp2 - entry) / PIP == pytest.approx(25.0)


def test_position_sizing_multipliers(tmp_path, tiny_cfg) -> None:
    df = _session_rows(pd.Timestamp("2024-06-04T08:00:00Z"), 80)
    store = _store_from_df(df)
    for mult, expect in [(1.5, 300_000), (1.0, 200_000), (0.5, 100_000)]:
        s = TunableCrossAssetConfluence(_loader(tmp_path), tiny_cfg, pinned_reading=_reading(size_mult=mult))
        with patch.object(s, "_warmup_ok", return_value=True), patch.object(s, "_trend_15m_long", return_value=True), patch.object(
            s, "_pullback_5m_long", return_value=True
        ), patch.object(s, "_level_context_long", return_value=True):
            idx = len(store) - 1
            sig = s.evaluate(BarView(store, idx), HistoricalDataView(store, idx), _portfolio_empty())
            assert sig is not None and sig.size == expect


def test_runner_trail_ratchet_long_only_tightens() -> None:
    from datetime import datetime as dt

    tz = UTC
    # Confirmed swing low at i=1 (1.0 < 2.0 and 1.0 < 1.5); later swing low at i=3 (0.5 < 1.5 and 0.5 < 0.6)
    bars = [
        Bar15M(dt(2024, 1, 1, 10, 0, tzinfo=tz), 2, 2, 2.0, 2.0, None, 0, 0, True),
        Bar15M(dt(2024, 1, 1, 10, 15, tzinfo=tz), 2, 2, 1.0, 1.1, None, 0, 0, True),
        Bar15M(dt(2024, 1, 1, 10, 30, tzinfo=tz), 2, 2, 1.5, 1.5, None, 0, 0, True),
        Bar15M(dt(2024, 1, 1, 10, 45, tzinfo=tz), 2, 2, 0.5, 0.55, None, 0, 0, True),
        Bar15M(dt(2024, 1, 1, 11, 0, tzinfo=tz), 2, 2, 0.6, 0.6, None, 0, 0, True),
    ]
    assert _latest_swing_low_15m(bars) == pytest.approx(0.5)


def test_runner_trail_stop_long_ratchets_up_with_higher_swing_low(tmp_path, tiny_cfg) -> None:
    """new_stop = max(current, swing_low - buffer) only moves up for longs."""
    from datetime import datetime as dt

    df = _session_rows(pd.Timestamp("2024-06-04T08:00:00Z"), 30)
    store = _store_from_df(df)
    s = TunableCrossAssetConfluence(_loader(tmp_path), tiny_cfg, pinned_reading=_reading())
    for i in range(len(store)):
        s.evaluate(BarView(store, i), HistoricalDataView(store, i), _portfolio_empty())
    tz = UTC
    s._completed_15m = [
        Bar15M(dt(2024, 6, 4, 8, 0, tzinfo=tz), 150, 150, 149.0, 149.5, None, 0, 0, True),
        Bar15M(dt(2024, 6, 4, 8, 15, tzinfo=tz), 150, 150, 148.0, 148.5, None, 0, 0, True),
        Bar15M(dt(2024, 6, 4, 8, 30, tzinfo=tz), 150, 150, 148.5, 149.0, None, 0, 0, True),
    ]
    s._pos = {
        "entry_bar": 0,
        "entry_price": 150.0,
        "direction": "long",
        "phase": "runner",
        "initial_size": 200_000,
        "tp1": 151.0,
        "tp2": 152.0,
        "runner_tp": 1e9,
    }
    s._last_trail_15m_ts = None
    pos = PositionSnapshot(
        trade_id=1,
        family=FAMILY,
        direction="long",
        entry_price=150.0,
        entry_bar=0,
        size=50_000,
        margin_held=1.0,
        stop_loss=147.0,
        take_profit=None,
        unrealized_pnl=0.0,
    )
    idx = len(store) - 1
    act = s.get_exit_conditions(pos, BarView(store, idx), HistoricalDataView(store, idx))
    assert act is not None and act.exit_type == "none" and act.new_stop_loss is not None
    assert float(act.new_stop_loss) > 147.0
    # Second 15m bar: same swing structure but stop must not loosen
    s._completed_15m.append(
        Bar15M(dt(2024, 6, 4, 8, 45, tzinfo=tz), 150, 150, 147.5, 148.0, None, 0, 0, True),
    )
    pos2 = PositionSnapshot(
        trade_id=1,
        family=FAMILY,
        direction="long",
        entry_price=150.0,
        entry_bar=0,
        size=50_000,
        margin_held=1.0,
        stop_loss=float(act.new_stop_loss),
        take_profit=None,
        unrealized_pnl=0.0,
    )
    act2 = s.get_exit_conditions(pos2, BarView(store, idx), HistoricalDataView(store, idx))
    if act2 is not None and act2.new_stop_loss is not None:
        assert float(act2.new_stop_loss) >= float(act.new_stop_loss) - 1e-9


def test_bias_flip_requests_full_exit(tmp_path, tiny_cfg) -> None:
    df = _session_rows(pd.Timestamp("2024-06-04T10:56:00Z"), 20)
    store = _store_from_df(df)
    s = TunableCrossAssetConfluence(_loader(tmp_path), tiny_cfg, pinned_reading=_reading(bias="STRONG_SHORT", raw=-2.5))
    s._rows.clear()
    s._by5.clear()
    s._by15.clear()
    s._by1h.clear()
    for i in range(len(store)):
        bv = BarView(store, i)
        hist = HistoricalDataView(store, i)
        s.evaluate(bv, hist, _portfolio_empty())
    s._pos = {
        "entry_bar": 0,
        "entry_price": 150.0,
        "direction": "long",
        "phase": "runner",
        "initial_size": 200_000,
        "tp1": 151.0,
        "tp2": 152.0,
        "runner_tp": 1e9,
    }
    s._entry_bias_sign = 1
    pos = PositionSnapshot(
        trade_id=1,
        family=FAMILY,
        direction="long",
        entry_price=150.0,
        entry_bar=0,
        size=50_000,
        margin_held=1.0,
        stop_loss=149.5,
        take_profit=None,
        unrealized_pnl=0.0,
    )
    idx = len(store) - 1
    act = s.get_exit_conditions(pos, BarView(store, idx), HistoricalDataView(store, idx))
    assert act is not None and act.exit_type == "full" and act.reason == "bias_flip"


def test_max_hold_exit(tmp_path, tiny_cfg) -> None:
    df = _session_rows(pd.Timestamp("2024-06-04T08:00:00Z"), 300)
    store = _store_from_df(df)
    s = TunableCrossAssetConfluence(_loader(tmp_path), tiny_cfg, pinned_reading=_reading())
    for i in range(len(store)):
        s.evaluate(BarView(store, i), HistoricalDataView(store, i), _portfolio_empty())
    s._pos = {
        "entry_bar": 0,
        "entry_price": 150.0,
        "direction": "long",
        "phase": "runner",
        "initial_size": 200_000,
        "tp1": 151.0,
        "tp2": 152.0,
        "runner_tp": 1e9,
    }
    pos = PositionSnapshot(
        trade_id=1,
        family=FAMILY,
        direction="long",
        entry_price=150.0,
        entry_bar=0,
        size=50_000,
        margin_held=1.0,
        stop_loss=149.0,
        take_profit=None,
        unrealized_pnl=0.0,
    )
    idx = 250
    act = s.get_exit_conditions(pos, BarView(store, idx), HistoricalDataView(store, idx))
    assert act is not None and act.reason == "max_hold"


def test_session_close_exit(tmp_path, tiny_cfg) -> None:
    # Last bar must be in [10:55, 11:00) UTC so session-close logic arms; 11:00 exactly is out of the window.
    df = _session_rows(pd.Timestamp("2024-06-04T10:55:00Z"), 5)
    store = _store_from_df(df)
    s = TunableCrossAssetConfluence(_loader(tmp_path), tiny_cfg, pinned_reading=_reading())
    for i in range(len(store)):
        s.evaluate(BarView(store, i), HistoricalDataView(store, i), _portfolio_empty())
    s._pos = {
        "entry_bar": 0,
        "entry_price": 150.0,
        "direction": "long",
        "phase": "runner",
        "initial_size": 200_000,
        "tp1": 151.0,
        "tp2": 152.0,
        "runner_tp": 1e9,
    }
    pos = PositionSnapshot(
        trade_id=1,
        family=FAMILY,
        direction="long",
        entry_price=150.0,
        entry_bar=0,
        size=50_000,
        margin_held=1.0,
        stop_loss=149.0,
        take_profit=None,
        unrealized_pnl=0.0,
    )
    idx = len(store) - 1
    act = s.get_exit_conditions(pos, BarView(store, idx), HistoricalDataView(store, idx))
    assert act is not None and act.reason == "session_close"


def test_no_future_bar_index_in_history(tmp_path, tiny_cfg) -> None:
    df = _session_rows(pd.Timestamp("2024-06-04T08:00:00Z"), 60)
    store = _store_from_df(df)
    s = CrossAssetConfluenceStrategy(_loader(tmp_path), tiny_cfg)
    idx = 30
    hist = HistoricalDataView(store, idx)
    assert len(hist) == idx + 1
    last = hist[idx]
    assert int(last.bar_index) == idx


def test_tp_partial_sequence_uses_half_fractions(tmp_path, tiny_cfg) -> None:
    """Matches engine: 50% then 50% of remainder => 25% of initial."""
    df = _session_rows(pd.Timestamp("2024-06-04T08:05:00Z"), 40)
    store = _store_from_df(df)
    s = TunableCrossAssetConfluence(_loader(tmp_path), tiny_cfg, pinned_reading=_reading())
    for i in range(len(store)):
        s.evaluate(BarView(store, i), HistoricalDataView(store, i), _portfolio_empty())
    entry_bar = 5
    s._pos = None
    pos = PositionSnapshot(
        trade_id=1,
        family=FAMILY,
        direction="long",
        entry_price=150.0,
        entry_bar=entry_bar,
        size=100_000,
        margin_held=1.0,
        stop_loss=148.0,
        take_profit=151.0,
        unrealized_pnl=0.0,
    )
    # Synthetic bar: high reaches TP1
    ts = df["timestamp"].iloc[10]
    row = {"timestamp": ts, "open": 150.5, "high": 151.5, "low": 150.4, "close": 151.2}
    df2 = df.copy()
    for k, v in row.items():
        df2.loc[df2["timestamp"] == ts, k] = v
    store2 = _store_from_df(df2)
    a1 = s.get_exit_conditions(pos, BarView(store2, 10), HistoricalDataView(store2, 10))
    assert a1 is not None and a1.exit_type == "partial" and a1.close_fraction == 0.5
