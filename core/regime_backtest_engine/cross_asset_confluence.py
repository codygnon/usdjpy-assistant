"""
Cross-asset confluence for USDJPY using OANDA CSVs (BCO, EUR_USD, XAU, XAG).

Loads midpoint OHLCV files from ``research_out/cross_assets`` (see
``scripts/download_cross_assets.py``) and, on each USDJPY M1 bar, takes the
**last complete** H1 or daily candle at or before that bar's time (as-of).

Vote convention (research defaults — tune in ``CrossAssetConfluenceConfig``):
  - **BCO_USD** (H1): positive return → lean **long USDJPY** (+1).
  - **EUR_USD** (H1): euro strength (positive return) → **short USDJPY** (-1 for long score).
  - **XAU_USD / XAG_USD** (D): metals up → **short USDJPY** (-1 each for long score).

Sum of votes in ``{-4, …, 4}``. Enter long if sum >= ``min_long_score``,
short if sum <= ``max_short_score``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .models import ClosedTrade, ExitAction, PortfolioSnapshot, PositionSnapshot, Signal
from .strategy import BarView, HistoricalDataView

Direction = Literal["long", "short"]

# Default filenames produced by scripts/download_cross_assets.py
DEFAULT_CROSS_ASSET_FILES = {
    "BCO_USD": "BCO_USD_H1_OANDA.csv",
    "EUR_USD": "EUR_USD_H1_OANDA.csv",
    "XAU_USD": "XAU_USD_D_OANDA.csv",
    "XAG_USD": "XAG_USD_D_OANDA.csv",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _to_utc_ts(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


@dataclass(frozen=True)
class _TimeCloseIndex:
    """Sorted UTC candle open times (ns) and mid closes for as-of lookups."""

    times_ns: np.ndarray
    close: np.ndarray

    @classmethod
    def from_csv(cls, path: Path) -> "_TimeCloseIndex":
        if not path.is_file():
            raise FileNotFoundError(f"cross-asset file not found: {path}")
        df = pd.read_csv(path)
        if "timestamp" not in df.columns:
            raise ValueError(f"{path}: expected 'timestamp' column")
        idx = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], utc=True))
        raw = idx.asi8.astype(np.int64)
        order = np.argsort(raw)
        times_ns = raw[order]
        close = df["close"].astype(float).to_numpy()[order]
        return cls(times_ns=times_ns, close=close)

    def asof_index(self, ts_ns: int) -> int:
        """Largest i with times_ns[i] <= ts_ns, or -1 if none."""
        return int(np.searchsorted(self.times_ns, ts_ns, side="right") - 1)

    def log_return_over_bars(self, ts_ns: int, bars_back: int) -> float | None:
        if bars_back <= 0:
            return None
        i = self.asof_index(ts_ns)
        if i < bars_back:
            return None
        c0 = float(self.close[i])
        c1 = float(self.close[i - bars_back])
        if c0 <= 0 or c1 <= 0:
            return None
        return float(np.log(c0 / c1))


@dataclass(frozen=True)
class CrossAssetBundle:
    bco: _TimeCloseIndex
    eur: _TimeCloseIndex
    xau: _TimeCloseIndex
    xag: _TimeCloseIndex


def load_cross_asset_bundle(
    directory: Path | None = None,
    filenames: dict[str, str] | None = None,
) -> CrossAssetBundle:
    """
    Load the four OANDA CSVs into memory-aligned as-of indexes.

    Args:
        directory: Folder containing CSVs (default: ``<repo>/research_out/cross_assets``).
        filenames: Optional map keys BCO_USD, EUR_USD, XAU_USD, XAG_USD -> file basename.
    """
    root = directory if directory is not None else _repo_root() / "research_out" / "cross_assets"
    names = filenames or DEFAULT_CROSS_ASSET_FILES
    missing = [k for k in DEFAULT_CROSS_ASSET_FILES if k not in names]
    if missing:
        raise ValueError(f"filenames map missing keys: {missing}")
    return CrossAssetBundle(
        bco=_TimeCloseIndex.from_csv(root / names["BCO_USD"]),
        eur=_TimeCloseIndex.from_csv(root / names["EUR_USD"]),
        xau=_TimeCloseIndex.from_csv(root / names["XAU_USD"]),
        xag=_TimeCloseIndex.from_csv(root / names["XAG_USD"]),
    )


class CrossAssetConfluenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    family_name: str = "cross_asset_confluence"
    cross_assets_dir: Path = Field(default_factory=lambda: _repo_root() / "research_out" / "cross_assets")
    cross_asset_filenames: dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_CROSS_ASSET_FILES))

    pip_size: float = 0.01
    fixed_units: int = 200_000
    h1_lookback_bars: int = Field(default=12, ge=1, description="H1 momentum lookback (bars)")
    d_lookback_bars: int = Field(default=5, ge=1, description="Daily momentum lookback (bars)")
    return_epsilon: float = Field(default=1e-5, ge=0.0)

    min_long_score: int = Field(default=2, ge=1, le=4)
    max_short_score: int = Field(default=-2, ge=-4, le=-1)

    min_warmup_1m_bars: int = Field(default=500, ge=0)
    cooldown_bars_after_sl: int = Field(default=30, ge=0)
    max_hold_bars: int = Field(default=240, ge=1)
    stop_loss_pips: float = Field(default=25.0, gt=0.0)
    take_profit_pips: float = Field(default=20.0, gt=0.0)

    london_start_hour_utc: int = 7
    london_end_hour_utc: int = 11
    ny_start_hour_utc: int = 12
    ny_end_hour_utc: int = 17


def cross_asset_session_open_utc(ts: pd.Timestamp, config: CrossAssetConfluenceConfig | None = None) -> bool:
    cfg = config or CrossAssetConfluenceConfig()
    stamp = _to_utc_ts(ts)
    minute_of_day = int(stamp.hour) * 60 + int(stamp.minute)
    london_start = int(cfg.london_start_hour_utc) * 60
    london_end = int(cfg.london_end_hour_utc) * 60
    ny_start = int(cfg.ny_start_hour_utc) * 60
    ny_end = int(cfg.ny_end_hour_utc) * 60
    return london_start <= minute_of_day < london_end or ny_start <= minute_of_day < ny_end


def _sign_vote(ret: float | None, eps: float) -> int:
    if ret is None:
        return 0
    if ret > eps:
        return 1
    if ret < -eps:
        return -1
    return 0


class CrossAssetConfluenceStrategy:
    """USDJPY strategy driven by as-of cross-asset momentum votes."""

    family_name: str

    def __init__(
        self,
        config: CrossAssetConfluenceConfig | None = None,
        bundle: CrossAssetBundle | None = None,
    ) -> None:
        self.config = config or CrossAssetConfluenceConfig()
        self.family_name = self.config.family_name
        self._bundle = bundle or load_cross_asset_bundle(
            self.config.cross_assets_dir,
            self.config.cross_asset_filenames,
        )
        self._cooldown_until_bar: int | None = None

    def evaluate(
        self,
        current_bar: BarView,
        history: HistoricalDataView,
        portfolio: PortfolioSnapshot,
    ) -> Signal | None:
        _ = history
        if int(current_bar.bar_index) < int(self.config.min_warmup_1m_bars):
            return None
        if self._cooldown_until_bar is not None and int(current_bar.bar_index) < int(self._cooldown_until_bar):
            return None
        if portfolio.open_positions:
            return None

        ts = _to_utc_ts(current_bar.timestamp)
        if not cross_asset_session_open_utc(ts, self.config):
            return None

        ts_ns = int(ts.value)

        oil_r = self._bundle.bco.log_return_over_bars(ts_ns, int(self.config.h1_lookback_bars))
        eur_r = self._bundle.eur.log_return_over_bars(ts_ns, int(self.config.h1_lookback_bars))
        gold_r = self._bundle.xau.log_return_over_bars(ts_ns, int(self.config.d_lookback_bars))
        silver_r = self._bundle.xag.log_return_over_bars(ts_ns, int(self.config.d_lookback_bars))

        eps = float(self.config.return_epsilon)
        oil_v = _sign_vote(oil_r, eps)
        eur_v = -_sign_vote(eur_r, eps)
        gold_v = -_sign_vote(gold_r, eps)
        silver_v = -_sign_vote(silver_r, eps)
        score = int(oil_v + eur_v + gold_v + silver_v)

        direction: Direction | None = None
        if score >= int(self.config.min_long_score):
            direction = "long"
        elif score <= int(self.config.max_short_score):
            direction = "short"

        if direction is None:
            return None

        entry = float(current_bar.mid_close)
        pip = float(self.config.pip_size)
        sl_off = float(self.config.stop_loss_pips) * pip
        tp_off = float(self.config.take_profit_pips) * pip
        if direction == "long":
            stop = entry - sl_off
            tp = entry + tp_off
        else:
            stop = entry + sl_off
            tp = entry - tp_off

        return Signal(
            family=self.family_name,
            direction=direction,
            stop_loss=float(stop),
            take_profit=float(tp),
            size=int(self.config.fixed_units),
            metadata={
                "confluence_score": score,
                "oil_ret": oil_r,
                "eur_ret": eur_r,
                "gold_ret": gold_r,
                "silver_ret": silver_r,
                "votes": {"oil": oil_v, "eur_usd": eur_v, "gold": gold_v, "silver": silver_v},
            },
        )

    def get_exit_conditions(
        self,
        position: PositionSnapshot,
        current_bar: BarView,
        history: HistoricalDataView,
    ) -> ExitAction | None:
        _ = history
        if position.family != self.family_name:
            return None
        if int(current_bar.bar_index) >= int(position.entry_bar) + int(self.config.max_hold_bars):
            return ExitAction(
                reason="max_hold",
                exit_type="full",
                close_fraction=1.0,
                price=float(current_bar.bid_close if position.direction == "long" else current_bar.ask_close),
            )
        return None

    def on_position_closed(self, trade: ClosedTrade) -> None:
        if trade.family != self.family_name:
            return
        if trade.exit_reason in {"stop_loss", "worst_case_stop"}:
            self._cooldown_until_bar = int(trade.exit_bar) + int(self.config.cooldown_bars_after_sl)
