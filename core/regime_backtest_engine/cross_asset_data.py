"""
Load and align multi-timeframe cross-asset data for USDJPY 1M backtests.

All lookups are CAUSAL: at query time T, only bars whose completion time <= T
are visible. H1 bars complete at open + 1 hour; daily bars at 22:00 UTC on the
bar's calendar date.
"""

from __future__ import annotations

import bisect
import csv
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

UTC = timezone.utc


@dataclass(frozen=True)
class CrossAssetBar:
    timestamp: datetime  # bar open time (UTC)
    completed_at: datetime  # when this bar becomes available (UTC)
    open: float
    high: float
    low: float
    close: float
    volume: float
    instrument: str
    timeframe: str


def _parse_usdjpy_time(value: str) -> datetime:
    s = value.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    else:
        dt = dt.astimezone(UTC)
    return dt


def _parse_cross_timestamp_flexible(value: str) -> datetime:
    """Parse OANDA cross-asset CSV time: ``2023-06-20 00:00:00`` or ISO with ``T`` / ``Z``."""
    s = value.strip()
    if "T" in s:
        iso = s[:-1] + "+00:00" if s.endswith("Z") else s
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        else:
            dt = dt.astimezone(UTC)
        return dt
    dt = datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
    return dt


def _h1_completed_at(open_ts: datetime) -> datetime:
    return open_ts + timedelta(hours=1)


def _daily_completed_at(open_ts: datetime) -> datetime:
    d = open_ts.astimezone(UTC).date()
    return datetime(d.year, d.month, d.day, 22, 0, 0, tzinfo=UTC)


def _bar_to_dict(bar: CrossAssetBar) -> dict[str, Any]:
    return {
        "timestamp": bar.timestamp,
        "completed_at": bar.completed_at,
        "open": bar.open,
        "high": bar.high,
        "low": bar.low,
        "close": bar.close,
        "volume": bar.volume,
        "instrument": bar.instrument,
        "timeframe": bar.timeframe,
    }


def _read_usdjpy_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"USDJPY file not found or not a file: {path}")
    rows: list[dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"time", "open", "high", "low", "close", "spread_pips"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"{path}: expected columns {sorted(required)}, got {reader.fieldnames}")
        for i, row in enumerate(reader, start=2):
            try:
                rows.append(
                    {
                        "time": _parse_usdjpy_time(row["time"]),
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "spread_pips": float(row["spread_pips"]),
                    }
                )
            except (KeyError, ValueError, TypeError) as e:
                logger.warning("USDJPY %s row %s: skip (%s)", path, i, e)
    if not rows:
        raise ValueError(f"USDJPY file is empty or no valid rows: {path}")
    return rows


def _read_cross_asset_bars(
    path: Path,
    instrument: str,
    timeframe: str,
    completed_fn,
) -> tuple[list[CrossAssetBar], list[datetime]]:
    if not path.is_file():
        raise FileNotFoundError(f"Cross-asset file not found or not a file: {path}")
    bars: list[CrossAssetBar] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"{path}: expected columns {sorted(required)}, got {reader.fieldnames}")
        for i, row in enumerate(reader, start=2):
            try:
                open_ts = _parse_cross_timestamp_flexible(row["timestamp"])
                bars.append(
                    CrossAssetBar(
                        timestamp=open_ts,
                        completed_at=completed_fn(open_ts),
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row["volume"]),
                        instrument=instrument,
                        timeframe=timeframe,
                    )
                )
            except (KeyError, ValueError, TypeError) as e:
                logger.warning("%s row %s: skip (%s)", path, i, e)
    if not bars:
        raise ValueError(f"Cross-asset file is empty or no valid rows: {path}")
    bars.sort(key=lambda b: (b.completed_at, b.timestamp))
    completed_times = [b.completed_at for b in bars]
    return bars, completed_times


class CrossAssetDataLoader:
    """
    Loads and aligns multi-timeframe, multi-asset data for backtesting.

    Architecture:
    - All cross-asset data is loaded into memory at init
    - Cross-asset bars are stored time-sorted by completion time
    - Lookup returns the most recent COMPLETED cross-asset bar at or before query time
    - CAUSAL: never returns data from the future relative to query time
    """

    def __init__(
        self,
        usdjpy_path: str,
        brent_path: str,
        eurusd_path: str,
        gold_path: str,
        silver_path: str,
    ) -> None:
        up = Path(usdjpy_path)
        self._usdjpy_bars = _read_usdjpy_rows(up)

        self._brent_bars, self._brent_completed = _read_cross_asset_bars(
            Path(brent_path), "BCO_USD", "H1", _h1_completed_at
        )
        self._eur_bars, self._eur_completed = _read_cross_asset_bars(
            Path(eurusd_path), "EUR_USD", "H1", _h1_completed_at
        )
        self._gold_bars, self._gold_completed = _read_cross_asset_bars(
            Path(gold_path), "XAU_USD", "D", _daily_completed_at
        )
        self._silver_bars, self._silver_completed = _read_cross_asset_bars(
            Path(silver_path), "XAG_USD", "D", _daily_completed_at
        )

    def get_usdjpy_bars(self) -> list[dict[str, Any]]:
        """Return all USDJPY 1M bars as list of dicts."""
        return self._usdjpy_bars

    def _last_completed_index(self, completed_list: list[datetime], query: datetime) -> int:
        if not completed_list:
            return -1
        q = query.astimezone(UTC) if query.tzinfo else query.replace(tzinfo=UTC)
        idx = bisect.bisect_right(completed_list, q) - 1
        return int(idx)

    def _get_at(
        self,
        bars: list[CrossAssetBar],
        completed_list: list[datetime],
        query: datetime,
    ) -> dict[str, Any] | None:
        idx = self._last_completed_index(completed_list, query)
        if idx < 0:
            return None
        return _bar_to_dict(bars[idx])

    def get_brent_at(self, timestamp: datetime) -> dict[str, Any] | None:
        """
        Most recent COMPLETED Brent H1 bar at or before timestamp.
        H1 bar starting at T completes at T + 1 hour.
        """
        return self._get_at(self._brent_bars, self._brent_completed, timestamp)

    def get_eurusd_at(self, timestamp: datetime) -> dict[str, Any] | None:
        return self._get_at(self._eur_bars, self._eur_completed, timestamp)

    def get_gold_at(self, timestamp: datetime) -> dict[str, Any] | None:
        """
        Most recent COMPLETED Gold daily bar. Daily bar for date D completes at 22:00 UTC on D.
        """
        return self._get_at(self._gold_bars, self._gold_completed, timestamp)

    def get_silver_at(self, timestamp: datetime) -> dict[str, Any] | None:
        return self._get_at(self._silver_bars, self._silver_completed, timestamp)

    def _history(
        self,
        bars: list[CrossAssetBar],
        completed_list: list[datetime],
        timestamp: datetime,
        n_bars: int,
    ) -> list[dict[str, Any]]:
        if n_bars <= 0:
            return []
        idx = self._last_completed_index(completed_list, timestamp)
        if idx < 0:
            return []
        start = max(0, idx - (n_bars - 1))
        return [_bar_to_dict(bars[i]) for i in range(start, idx + 1)]

    def get_brent_history(self, timestamp: datetime, n_bars: int) -> list[dict[str, Any]]:
        return self._history(self._brent_bars, self._brent_completed, timestamp, n_bars)

    def get_eurusd_history(self, timestamp: datetime, n_bars: int) -> list[dict[str, Any]]:
        return self._history(self._eur_bars, self._eur_completed, timestamp, n_bars)

    def get_gold_history(self, timestamp: datetime, n_bars: int) -> list[dict[str, Any]]:
        return self._history(self._gold_bars, self._gold_completed, timestamp, n_bars)

    def get_silver_history(self, timestamp: datetime, n_bars: int) -> list[dict[str, Any]]:
        return self._history(self._silver_bars, self._silver_completed, timestamp, n_bars)
