"""Capture functions for the 14 priority telemetry fields (PHASE9.8).

Each function takes explicit inputs (no globals, no broker handle reaching into
ambient state) so the unit test can drive every code path with synthetic data.
The autonomous loop wires these together; this module only computes.
"""
from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from .snapshot import LevelAgeMetadata, LevelPacket


# --- Exposure (PHASE9.8 items 1, 4) -------------------------------------------

@dataclass
class OpenPositionRow:
    side: str  # 'buy' | 'sell'
    units: float  # OANDA units; 100_000 units = 1 standard lot
    unrealized_pnl_usd: float


def open_lots_by_side(positions: Iterable[OpenPositionRow]) -> tuple[float, float]:
    """Return (buy_lots, sell_lots). Lots = units / 100_000."""
    buy = sell = 0.0
    for p in positions:
        lots = abs(p.units) / 100_000.0
        if p.side == "buy":
            buy += lots
        elif p.side == "sell":
            sell += lots
    return round(buy, 2), round(sell, 2)


def unrealized_pnl_by_side(positions: Iterable[OpenPositionRow]) -> tuple[float, float]:
    buy = sell = 0.0
    for p in positions:
        if p.side == "buy":
            buy += p.unrealized_pnl_usd
        elif p.side == "sell":
            sell += p.unrealized_pnl_usd
    return round(buy, 2), round(sell, 2)


# --- Sizing inputs (PHASE9.8 items 2, 5) --------------------------------------

def pip_value_per_lot(usdjpy_price: float, lot_units: int = 100_000) -> float:
    """USD pip value per 1.00 lot (100k units) of USDJPY.

    Pip = 0.01 for USDJPY. Pip value in USD = (0.01 / price) * units.
    Example: at 150.00 → (0.01 / 150) * 100_000 = $6.67/pip per lot.
    """
    if usdjpy_price <= 0:
        raise ValueError("usdjpy_price must be positive")
    return round((0.01 / usdjpy_price) * lot_units, 4)


def risk_after_fill_usd(
    *,
    proposed_lots: float,
    sl_pips: float,
    pip_value_per_lot_usd: float,
) -> float:
    """Dollar loss if SL hits at the proposed size.

    Computed post-LLM/pre-order. Blocking field — no order if absent.
    """
    if proposed_lots < 0 or sl_pips < 0:
        raise ValueError("lots and sl_pips must be non-negative")
    return round(proposed_lots * sl_pips * pip_value_per_lot_usd, 2)


# --- Drawdown awareness (PHASE9.8 item 3) -------------------------------------

@dataclass
class ClosedTradeRow:
    closed_at_utc: str
    pnl_usd: float
    pips: float
    lots: float


def rolling_pnl(
    closed_trades: list[ClosedTradeRow],
    window: int = 20,
) -> tuple[float, float]:
    """Return (rolling_N_pnl_usd, rolling_N_lot_weighted_pnl_pips).

    Lot-weighted pnl = sum(pips * lots) over the last N trades. This is the
    risk-adjusted view the sizing function uses to throttle.
    """
    if not closed_trades:
        return 0.0, 0.0
    tail = sorted(closed_trades, key=lambda t: t.closed_at_utc)[-window:]
    pnl_usd = round(sum(t.pnl_usd for t in tail), 2)
    lot_weighted = round(sum(t.pips * t.lots for t in tail), 2)
    return pnl_usd, lot_weighted


def fetch_closed_trades_for_rolling(
    db_path: Path,
    profile: str,
    *,
    limit: int = 50,
) -> list[ClosedTradeRow]:
    """Read recent closed trades from ai_suggestions for rolling P&L.

    Pulls more than `window` so callers can filter by entry_type/engine if
    they want. Returns rows with non-null pnl/pips/lots only.
    """
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT closed_at, pnl, pips, lots
            FROM ai_suggestions
            WHERE profile = ?
              AND closed_at IS NOT NULL
              AND pnl IS NOT NULL
              AND lots IS NOT NULL
            ORDER BY closed_at DESC
            LIMIT ?
            """,
            (profile, limit),
        ).fetchall()
    return [
        ClosedTradeRow(
            closed_at_utc=str(r["closed_at"]),
            pnl_usd=float(r["pnl"]),
            pips=float(r["pips"] or 0.0),
            lots=float(r["lots"]),
        )
        for r in rows
    ]


# --- Side-normalized level packet (PHASE9.8 item 6) ---------------------------

def build_level_packet(
    *,
    proposed_side: str,
    nearest_support: Optional[dict] = None,
    nearest_resistance: Optional[dict] = None,
    tick_mid: float,
    profit_path_blocker_pips: Optional[float] = None,
) -> Optional[LevelPacket]:
    """Construct a side-normalized level packet from raw S/R candidates.

    Buy: anchor on nearest support (entry_wall.side == buy_support).
    Sell: anchor on nearest resistance (entry_wall.side == sell_resistance).
    Returns None if the side-relevant level is missing — caller must skip CLR.

    `nearest_support`/`nearest_resistance` are dicts with at least:
      {price: float, quality_score: float, structural_origin: str}
    quality_score is the level_quality_score in [0, 100] used by the gate
    layer's eligibility thresholds (buy>=70, sell>=85).
    """
    if proposed_side == "buy":
        lvl = nearest_support
        side_tag = "buy_support"
    elif proposed_side == "sell":
        lvl = nearest_resistance
        side_tag = "sell_resistance"
    else:
        return None
    if not lvl:
        return None
    price = float(lvl["price"])
    pip_size = 0.01  # USDJPY
    distance = (price - tick_mid) / pip_size
    if proposed_side == "sell":
        distance = -distance  # signed: + means level is in trade direction
    return LevelPacket(
        side=side_tag,
        level_price=price,
        level_quality_score=float(lvl.get("quality_score", 0.0)),
        distance_pips=round(distance, 1),
        profit_path_blocker_distance_pips=profit_path_blocker_pips,
        structural_origin=lvl.get("structural_origin"),
    )


def build_level_age_metadata(
    *,
    level_age_minutes: Optional[float] = None,
    touch_count: Optional[int] = None,
    broken_then_reclaimed: bool = False,
    last_touch_utc: Optional[str] = None,
) -> LevelAgeMetadata:
    """Trivial constructor — wrapped here so the loop calls one telemetry API."""
    return LevelAgeMetadata(
        level_age_minutes=level_age_minutes,
        touch_count=touch_count,
        broken_then_reclaimed=broken_then_reclaimed,
        last_touch_utc=last_touch_utc,
    )


# --- Volatility regime --------------------------------------------------------

def classify_volatility_regime(atr_m5_pips: Optional[float]) -> str:
    """Maps M5 ATR into a regime tag the sizing function consumes.

    Threshold from Phase 7 stop-overshoot anomaly: Phase 3's regime ran at
    ~15.6p M5 ATR, healthy regime at ~5.1p. 10p chosen as the elevated
    boundary; revisit when path-time telemetry lands.
    """
    if atr_m5_pips is None or math.isnan(atr_m5_pips):
        return "unknown"
    return "elevated" if atr_m5_pips >= 10.0 else "normal"
