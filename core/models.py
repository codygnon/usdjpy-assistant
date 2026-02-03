from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


Side = Literal["buy", "sell"]


@dataclass(frozen=True)
class MarketContext:
    spread_pips: Optional[float]
    alignment_score: Optional[int]


@dataclass(frozen=True)
class TradeCandidate:
    symbol: str
    side: Side
    entry_price: float
    stop_price: Optional[float] = None
    target_price: Optional[float] = None
    size_lots: Optional[float] = None


@dataclass(frozen=True)
class RiskSizing:
    suggested_size_lots: Optional[float] = None
    risk_per_trade_pct: Optional[float] = None
    risk_amount: Optional[float] = None
    risk_pips: Optional[float] = None
    pip_value_per_lot: Optional[float] = None


@dataclass(frozen=True)
class RiskDecision:
    allow: bool
    hard_reasons: list[str]
    warnings: list[str]
    sizing: RiskSizing

