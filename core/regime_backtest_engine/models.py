from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class InstrumentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str
    base_currency: str = "USD"
    quote_currency: str = "JPY"
    account_currency: str = "USD"
    pip_size: float = 0.01
    contract_size: int = 1
    margin_rate: float = Field(default=0.03, gt=0.0)


class FixedSpreadConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    spread_pips: float = Field(gt=0.0)


class SessionSpreadWindow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    start_hour_utc: float = Field(ge=0.0, lt=24.0)
    end_hour_utc: float = Field(ge=0.0, le=24.0)
    spread_pips: float = Field(gt=0.0)


class DeterministicSpreadModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    default_spread_pips: float = Field(gt=0.0)
    session_windows: tuple[SessionSpreadWindow, ...] = ()


class SpreadConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    spread_source: Literal["from_data", "fixed", "model"]
    fixed: Optional[FixedSpreadConfig] = None
    model: Optional[DeterministicSpreadModelConfig] = None

    @model_validator(mode="after")
    def validate_source(self) -> "SpreadConfig":
        if self.spread_source == "fixed" and self.fixed is None:
            raise ValueError("fixed spread_source requires fixed config")
        if self.spread_source == "model" and self.model is None:
            raise ValueError("model spread_source requires model config")
        if self.spread_source == "from_data" and (self.fixed is not None or self.model is not None):
            raise ValueError("from_data spread_source cannot also define fixed/model config")
        return self


class SlippageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    fixed_slippage_pips: float = Field(default=0.0, ge=0.0)


class AdmissionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    allow_opposing_exposure: bool = False
    max_total_open_positions: int = Field(default=3, ge=1)
    max_open_positions_per_family: dict[str, int] = Field(default_factory=dict)
    max_total_units: int = Field(default=1_000_000, ge=1)
    max_units_per_family: dict[str, int] = Field(default_factory=dict)
    family_priority: tuple[str, ...] = ()


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    hypothesis: str
    minimum_trade_count: int = Field(ge=1)
    minimum_profit_factor: float = Field(gt=0.0)
    maximum_drawdown_usd: Optional[float] = Field(default=None, ge=0.0)
    maximum_drawdown_pct: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    expected_win_rate_min: float = Field(ge=0.0, le=100.0)
    expected_win_rate_max: float = Field(ge=0.0, le=100.0)

    @model_validator(mode="after")
    def validate_thresholds(self) -> "RunManifest":
        if self.expected_win_rate_max < self.expected_win_rate_min:
            raise ValueError("expected_win_rate_max must be >= expected_win_rate_min")
        if self.maximum_drawdown_usd is None and self.maximum_drawdown_pct is None:
            raise ValueError("manifest requires maximum_drawdown_usd or maximum_drawdown_pct")
        return self


class WalkForwardWindow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    in_sample_start_index: int = Field(ge=0)
    in_sample_end_index: int = Field(ge=0)
    out_sample_start_index: int = Field(ge=0)
    out_sample_end_index: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_indices(self) -> "WalkForwardWindow":
        if self.in_sample_end_index < self.in_sample_start_index:
            raise ValueError("in-sample end index must be >= in-sample start index")
        if self.out_sample_end_index < self.out_sample_start_index:
            raise ValueError("out-of-sample end index must be >= out-of-sample start index")
        if self.out_sample_start_index <= self.in_sample_end_index:
            raise ValueError("out-of-sample window must start after in-sample window ends")
        return self


class WalkForwardConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    windows: tuple[WalkForwardWindow, ...]
    output_dir: Path
    aggregate_out_of_sample_only: bool = True

    @model_validator(mode="after")
    def validate_windows(self) -> "WalkForwardConfig":
        if not self.windows:
            raise ValueError("walk-forward config requires at least one window")
        return self


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    hypothesis: str = "phase1_engine_validation"
    data_path: Path
    output_dir: Path
    mode: Literal["standalone", "integrated"]
    active_families: tuple[str, ...]
    instrument: InstrumentSpec
    spread: SpreadConfig
    slippage: SlippageConfig = Field(default_factory=SlippageConfig)
    admission: AdmissionConfig = Field(default_factory=AdmissionConfig)
    manifest: Optional[RunManifest] = None
    initial_balance: float = Field(default=100_000.0, gt=0.0)
    seed: int = 1337
    start_index: Optional[int] = Field(default=None, ge=0)
    end_index: Optional[int] = Field(default=None, ge=0)
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    bar_log_format: Literal["parquet", "csv"] = "parquet"

    @model_validator(mode="after")
    def validate_mode(self) -> "RunConfig":
        if self.mode == "standalone" and len(self.active_families) != 1:
            raise ValueError("standalone mode requires exactly one active family")
        if self.end_index is not None and self.start_index is not None and self.end_index < self.start_index:
            raise ValueError("end_index must be >= start_index")
        return self


class NormalizedDataContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    bar_count: int
    start_time: str
    end_time: str
    columns: tuple[str, ...]
    synthetic_bid_ask: bool


class PositionSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    trade_id: int
    family: str
    direction: Literal["long", "short"]
    entry_price: float
    entry_bar: int
    size: int
    margin_held: float
    stop_loss: float
    take_profit: Optional[float]
    unrealized_pnl: float


class PortfolioSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    balance: float
    equity: float
    unrealized_pnl: float
    margin_used: float
    available_margin: float
    open_positions: tuple[PositionSnapshot, ...] = ()
    closed_trade_count: int = 0


@dataclass(frozen=True)
class Signal:
    family: str
    direction: Literal["long", "short"]
    stop_loss: float
    take_profit: Optional[float]
    size: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExitAction:
    reason: str
    exit_type: Literal["none", "full", "partial"] = "none"
    close_fraction: float = 1.0
    price: Optional[float] = None
    new_stop_loss: Optional[float] = None
    new_take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    close_full: bool = False


@dataclass(frozen=True)
class BarRecord:
    bar_index: int
    timestamp: Any
    bid_open: float
    bid_high: float
    bid_low: float
    bid_close: float
    ask_open: float
    ask_high: float
    ask_low: float
    ask_close: float
    mid_open: float
    mid_high: float
    mid_low: float
    mid_close: float
    spread_pips: float


@dataclass
class Position:
    trade_id: int
    family: str
    direction: Literal["long", "short"]
    entry_price: float
    entry_bar: int
    entry_time: Any
    size: int
    initial_size: int
    margin_held: float
    stop_loss: float
    take_profit: Optional[float]
    unrealized_pnl: float = 0.0


@dataclass(frozen=True)
class ClosedTrade:
    trade_id: int
    family: str
    direction: Literal["long", "short"]
    entry_time: Any
    exit_time: Any
    entry_bar: int
    exit_bar: int
    entry_price: float
    exit_price: float
    size: int
    margin_held: float
    stop_loss: float
    take_profit: Optional[float]
    spread_cost: float
    slippage_cost: float
    pnl_usd: float
    pnl_pips: float
    bars_held: int
    exit_reason: str
    event_type: Literal["full", "partial"] = "full"
    close_fraction: float = 1.0
    closed_units: int = 0
    remaining_units: int = 0


@dataclass(frozen=True)
class RejectedSignal:
    family: str
    direction: Literal["long", "short"]
    reason: str


@dataclass(frozen=True)
class ArbitrationRecord:
    bar_index: int
    timestamp: Any
    candidate_families: tuple[str, ...]
    candidate_directions: tuple[str, ...]
    outcome: str
    accepted_families: tuple[str, ...]
    rejected_families: tuple[str, ...]
    rejection_reasons: tuple[str, ...]


@dataclass
class PortfolioState:
    balance: float
    equity: float
    unrealized_pnl: float
    margin_used: float
    available_margin: float
    open_positions: list[Position] = field(default_factory=list)
    closed_trades: list[ClosedTrade] = field(default_factory=list)
    pending_signals: list[Signal] = field(default_factory=list)
    trade_id_seq: int = 0


@dataclass(frozen=True)
class BacktestResult:
    summary: dict[str, Any]
    trade_log_path: Path
    bar_log_path: Path
    config_snapshot_path: Path
    manifest_path: Optional[Path]
    arbitration_log_path: Optional[Path]
    final_portfolio: PortfolioSnapshot


@dataclass(frozen=True)
class WalkForwardSegmentResult:
    label: str
    in_sample_result: BacktestResult
    out_of_sample_result: BacktestResult


@dataclass(frozen=True)
class WalkForwardResult:
    summary: dict[str, Any]
    summary_path: Path
    manifest_path: Optional[Path]
    segments_path: Path
    segment_results: tuple[WalkForwardSegmentResult, ...]


@dataclass(frozen=True)
class PendingOrderRejection:
    signal: Signal
    reason: str


@dataclass(frozen=True)
class PendingFill:
    signal: Signal
    filled: bool
    reason: Optional[str] = None


def portfolio_snapshot_from_state(state: PortfolioState) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        balance=float(state.balance),
        equity=float(state.equity),
        unrealized_pnl=float(state.unrealized_pnl),
        margin_used=float(state.margin_used),
        available_margin=float(state.available_margin),
        open_positions=tuple(
            PositionSnapshot(
                trade_id=p.trade_id,
                family=p.family,
                direction=p.direction,
                entry_price=float(p.entry_price),
                entry_bar=int(p.entry_bar),
                size=int(p.size),
                margin_held=float(p.margin_held),
                stop_loss=float(p.stop_loss),
                take_profit=float(p.take_profit) if p.take_profit is not None else None,
                unrealized_pnl=float(p.unrealized_pnl),
            )
            for p in state.open_positions
        ),
        closed_trade_count=len(state.closed_trades),
    )


def closed_trade_to_row(trade: ClosedTrade) -> dict[str, Any]:
    return asdict(trade)


def arbitration_to_row(record: ArbitrationRecord) -> dict[str, Any]:
    row = asdict(record)
    row["candidate_families"] = list(record.candidate_families)
    row["candidate_directions"] = list(record.candidate_directions)
    row["accepted_families"] = list(record.accepted_families)
    row["rejected_families"] = list(record.rejected_families)
    row["rejection_reasons"] = list(record.rejection_reasons)
    return row
