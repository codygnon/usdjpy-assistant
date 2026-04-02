from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .models import DeterministicSpreadModelConfig, NormalizedDataContract, RunConfig
from .strategy import MarketDataStore


@dataclass(frozen=True)
class LoadedMarketData:
    store: MarketDataStore
    contract: NormalizedDataContract
    frame: pd.DataFrame


def _detect_mid_columns(df: pd.DataFrame) -> dict[str, str] | None:
    if {"open", "high", "low", "close"}.issubset(df.columns):
        return {"open": "open", "high": "high", "low": "low", "close": "close"}
    if {"mid_open", "mid_high", "mid_low", "mid_close"}.issubset(df.columns):
        return {"open": "mid_open", "high": "mid_high", "low": "mid_low", "close": "mid_close"}
    return None


def _hour_in_window(hour: float, start: float, end: float) -> bool:
    if start <= end:
        return start <= hour < end
    return hour >= start or hour < end


def _spread_for_timestamp(ts: pd.Timestamp, model: DeterministicSpreadModelConfig) -> float:
    hour = float(ts.hour) + float(ts.minute) / 60.0
    for window in model.session_windows:
        if _hour_in_window(hour, window.start_hour_utc, window.end_hour_utc):
            return float(window.spread_pips)
    return float(model.default_spread_pips)


def _normalize_bid_ask(df: pd.DataFrame, cfg: RunConfig) -> tuple[pd.DataFrame, bool]:
    bid_cols = {"bid_open", "bid_high", "bid_low", "bid_close"}
    ask_cols = {"ask_open", "ask_high", "ask_low", "ask_close"}
    if bid_cols.issubset(df.columns) and ask_cols.issubset(df.columns):
        if cfg.spread.spread_source != "from_data":
            raise ValueError("bid/ask input requires spread_source='from_data'")
        out = df.copy()
        out["mid_open"] = (out["bid_open"] + out["ask_open"]) / 2.0
        out["mid_high"] = (out["bid_high"] + out["ask_high"]) / 2.0
        out["mid_low"] = (out["bid_low"] + out["ask_low"]) / 2.0
        out["mid_close"] = (out["bid_close"] + out["ask_close"]) / 2.0
        out["spread_pips"] = (out["ask_close"] - out["bid_close"]) / cfg.instrument.pip_size
        return out, False

    mid_map = _detect_mid_columns(df)
    if mid_map is None:
        raise ValueError("market data must provide either bid/ask OHLC or mid OHLC")
    if cfg.spread.spread_source == "from_data":
        raise ValueError("spread_source='from_data' requires bid/ask OHLC columns")

    out = df.copy()
    out["mid_open"] = pd.to_numeric(out[mid_map["open"]], errors="raise")
    out["mid_high"] = pd.to_numeric(out[mid_map["high"]], errors="raise")
    out["mid_low"] = pd.to_numeric(out[mid_map["low"]], errors="raise")
    out["mid_close"] = pd.to_numeric(out[mid_map["close"]], errors="raise")
    if cfg.spread.spread_source == "fixed":
        spread_series = pd.Series(float(cfg.spread.fixed.spread_pips), index=out.index)
    else:
        spread_series = out["timestamp"].apply(lambda ts: _spread_for_timestamp(pd.Timestamp(ts), cfg.spread.model))
    half = spread_series * cfg.instrument.pip_size / 2.0
    out["spread_pips"] = spread_series
    out["bid_open"] = out["mid_open"] - half
    out["bid_high"] = out["mid_high"] - half
    out["bid_low"] = out["mid_low"] - half
    out["bid_close"] = out["mid_close"] - half
    out["ask_open"] = out["mid_open"] + half
    out["ask_high"] = out["mid_high"] + half
    out["ask_low"] = out["mid_low"] + half
    out["ask_close"] = out["mid_close"] + half
    return out, True


def _apply_subset(df: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    out = df
    if cfg.start_time is not None:
        start_ts = pd.Timestamp(cfg.start_time, tz="UTC") if pd.Timestamp(cfg.start_time).tzinfo is None else pd.Timestamp(cfg.start_time).tz_convert("UTC")
        if start_ts > out["timestamp"].iloc[-1]:
            raise ValueError("start_time is beyond available data")
        out = out[out["timestamp"] >= start_ts]
    if cfg.end_time is not None:
        end_ts = pd.Timestamp(cfg.end_time, tz="UTC") if pd.Timestamp(cfg.end_time).tzinfo is None else pd.Timestamp(cfg.end_time).tz_convert("UTC")
        if end_ts < out["timestamp"].iloc[0]:
            raise ValueError("end_time is before available data")
        out = out[out["timestamp"] <= end_ts]
    if cfg.start_index is not None or cfg.end_index is not None:
        start = int(cfg.start_index or 0)
        end = int(cfg.end_index if cfg.end_index is not None else len(out) - 1)
        if start >= len(out):
            raise ValueError("start_index is beyond available data")
        if end >= len(out):
            raise ValueError("end_index is beyond available data")
        out = out.iloc[start : end + 1]
    if out.empty:
        raise ValueError("requested data subset is empty")
    return out.reset_index(drop=True)


def load_market_data(cfg: RunConfig) -> LoadedMarketData:
    path = Path(cfg.data_path)
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        if "time" in df.columns:
            df = df.rename(columns={"time": "timestamp"})
        else:
            raise ValueError("market data requires a timestamp or time column")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="raise")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = _apply_subset(df, cfg)
    normalized, synthetic = _normalize_bid_ask(df, cfg)
    core_columns = [
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
    extra_columns = [col for col in normalized.columns if col not in core_columns]
    needed = core_columns + extra_columns
    columns = {col: normalized[col].to_numpy() for col in needed}
    store = MarketDataStore(columns)
    contract = NormalizedDataContract(
        bar_count=len(normalized),
        start_time=normalized["timestamp"].iloc[0].isoformat(),
        end_time=normalized["timestamp"].iloc[-1].isoformat(),
        columns=tuple(needed),
        synthetic_bid_ask=synthetic,
    )
    return LoadedMarketData(store=store, contract=contract, frame=normalized[needed].copy())
