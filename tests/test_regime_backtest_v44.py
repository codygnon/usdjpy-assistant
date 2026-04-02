from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from core.regime_backtest_engine import (
    AdmissionConfig,
    BacktestEngine,
    FixedSpreadConfig,
    InstrumentSpec,
    RunConfig,
    SlippageConfig,
    SpreadConfig,
    V44NYStrategy,
    V44StrategyConfig,
)
from core.regime_backtest_engine.strategy import BarView, HistoricalDataView, MarketDataStore


def _make_history_store(rows: list[dict[str, float | str]]) -> MarketDataStore:
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    spread_pips = 1.2
    pip_size = 0.01
    half = spread_pips * pip_size / 2.0
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
    return MarketDataStore({col: df[col].to_numpy() for col in cols})


def _make_uptrend_rows() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    ts = pd.Timestamp("2025-01-01T09:00:00Z")
    price = 150.0
    for i in range(360):
        drift = 0.015 if i < 120 else 0.03
        open_px = price
        close_px = price + drift
        rows.append(
            {
                "timestamp": (ts + pd.Timedelta(minutes=i)).isoformat(),
                "open": open_px,
                "high": close_px + 0.01,
                "low": open_px - 0.005,
                "close": close_px,
            }
        )
        price = close_px
    return rows


def _base_v44_config() -> V44StrategyConfig:
    return V44StrategyConfig(
        ny_start_hour=13.0,
        ny_end_hour=16.0,
        ny_start_delay_minutes=0,
        session_entry_cutoff_minutes=30,
        h1_ema_fast_period=1,
        h1_ema_slow_period=2,
        m5_ema_fast_period=2,
        m5_ema_slow_period=3,
        slope_bars=2,
        strong_slope_threshold=0.05,
        weak_slope_threshold=0.01,
        min_body_pips=0.1,
        atr_pct_filter_enabled=False,
        skip_weak=True,
        skip_normal=False,
        ny_strength_allow="strong_normal",
        news_filter_enabled=False,
        rp_min_lot=0.1,
        rp_max_lot=1.0,
    )


def test_v44_strategy_generates_signal_only_in_ny_window() -> None:
    rows = _make_uptrend_rows()
    store = _make_history_store(rows)
    strategy = V44NYStrategy(_base_v44_config())

    signal_seen = None
    before_ny_signal = None
    for idx in range(len(store)):
        current_bar = BarView(store, idx)
        history = HistoricalDataView(store, idx)
        signal = strategy.evaluate(
            current_bar,
            history,
            portfolio=type(
                "Portfolio",
                (),
                {"equity": 100000.0, "open_positions": (), "balance": 100000.0, "unrealized_pnl": 0.0, "margin_used": 0.0, "available_margin": 100000.0, "closed_trade_count": 0},
            )(),
        )
        ts = pd.Timestamp(current_bar.timestamp)
        if ts.hour < 13 and signal is not None:
            before_ny_signal = signal
        if ts.hour >= 13 and signal is not None:
            signal_seen = signal
            break

    assert before_ny_signal is None
    assert signal_seen is not None
    assert signal_seen.family == "v44_ny"
    assert signal_seen.direction == "long"


def test_v44_news_filter_blocks_signal() -> None:
    rows = _make_uptrend_rows()
    store = _make_history_store(rows)
    strategy = V44NYStrategy(
        _base_v44_config().model_copy(update={"news_filter_enabled": True, "news_calendar_path": None})
    )
    strategy._news_events = [pd.Timestamp("2025-01-01T13:30:00Z")]

    blocked = False
    for idx in range(len(store)):
        current_bar = BarView(store, idx)
        ts = pd.Timestamp(current_bar.timestamp)
        if ts < pd.Timestamp("2025-01-01T13:00:00Z") or ts > pd.Timestamp("2025-01-01T13:40:00Z"):
            continue
        history = HistoricalDataView(store, idx)
        signal = strategy.evaluate(
            current_bar,
            history,
            portfolio=type(
                "Portfolio",
                (),
                {"equity": 100000.0, "open_positions": (), "balance": 100000.0, "unrealized_pnl": 0.0, "margin_used": 0.0, "available_margin": 100000.0, "closed_trade_count": 0},
            )(),
        )
        if signal is None:
            blocked = True
            break
    assert blocked is True


def test_v44_standalone_engine_run_writes_manifest_and_summary(tmp_path: Path) -> None:
    source = Path("/Users/codygnon/Documents/usdjpy_assistant/research_out/USDJPY_M1_OANDA_500k.csv")
    cfg_json = tmp_path / "v44_config.json"
    cfg_json.write_text(json.dumps({"ny_start": 13.0, "ny_end": 16.0, "v5_ny_start_delay_minutes": 0}), encoding="utf-8")
    v44_cfg = V44StrategyConfig.from_v44_json("/Users/codygnon/Documents/usdjpy_assistant/research_out/session_momentum_v44_base_config.json").model_copy(
        update={
            "h1_ema_fast_period": 3,
            "h1_ema_slow_period": 5,
            "m5_ema_fast_period": 3,
            "m5_ema_slow_period": 5,
            "slope_bars": 2,
            "strong_slope_threshold": 0.05,
            "weak_slope_threshold": 0.01,
            "min_body_pips": 0.1,
            "atr_pct_filter_enabled": False,
            "news_filter_enabled": False,
            "rp_min_lot": 0.1,
            "rp_max_lot": 1.0,
        }
    )
    strategy = V44NYStrategy(v44_cfg)
    engine = BacktestEngine({"v44_ny": strategy})
    config = RunConfig(
        hypothesis="phase4_v44_subset_validation",
        data_path=source,
        output_dir=tmp_path / "v44_run",
        mode="standalone",
        active_families=("v44_ny",),
        instrument=InstrumentSpec(symbol="USDJPY", margin_rate=(1.0 / 33.3)),
        spread=SpreadConfig(spread_source="fixed", fixed=FixedSpreadConfig(spread_pips=2.0)),
        slippage=SlippageConfig(fixed_slippage_pips=0.1),
        admission=AdmissionConfig(
            allow_opposing_exposure=False,
            max_total_open_positions=3,
            max_open_positions_per_family={"v44_ny": 3},
            max_total_units=300_000,
            max_units_per_family={"v44_ny": 300_000},
            family_priority=("v44_ny",),
        ),
        start_index=0,
        end_index=15_000,
        bar_log_format="csv",
    )
    result = engine.run(config)
    assert result.trade_log_path.exists()
    assert result.bar_log_path.exists()
    assert result.summary["mode"] == "standalone"
    assert result.summary["processed_bar_count"] == 15_001
