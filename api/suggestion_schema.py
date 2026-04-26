from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from api import suggestion_tracker


class FillmoreSuggestionBase(BaseModel):
    model_config = ConfigDict(extra="allow")

    side: Literal["buy", "sell"]
    price: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    lots: float = Field(ge=0.0, le=15.0)
    rationale: str
    exit_strategy: str | None = None
    exit_params: dict[str, Any] = Field(default_factory=dict)
    entry_type: str

    @field_validator("side", mode="before")
    @classmethod
    def _normalize_side(cls, value: Any) -> str:
        side = str(value or "").strip().lower()
        if side not in {"buy", "sell"}:
            raise ValueError("side must be 'buy' or 'sell'")
        return side

    @field_validator("rationale", mode="before")
    @classmethod
    def _normalize_rationale(cls, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("rationale is required")
        return text

    @field_validator("entry_type", mode="before")
    @classmethod
    def _normalize_entry_type(cls, value: Any) -> str:
        entry_type = str(value or "").strip().lower()
        if entry_type not in {
            suggestion_tracker.ENTRY_TYPE_FILLMORE_MANUAL,
            suggestion_tracker.ENTRY_TYPE_FILLMORE_AUTONOMOUS,
        }:
            raise ValueError("entry_type must be ai_manual or ai_autonomous")
        return entry_type

    @field_validator("exit_strategy", mode="before")
    @classmethod
    def _normalize_exit_strategy(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


class ManualSuggestion(FillmoreSuggestionBase):
    entry_type: Literal["ai_manual"] = suggestion_tracker.ENTRY_TYPE_FILLMORE_MANUAL
    confidence: Literal["low", "medium", "high"]
    time_in_force: Literal["GTC", "GTD"] = "GTD"
    gtd_time_utc: str | None = None

    @model_validator(mode="after")
    def _validate_pricing(self) -> "ManualSuggestion":
        if self.price <= 0 or self.sl <= 0 or self.tp <= 0:
            raise ValueError("manual suggestions must include positive price/sl/tp")
        if self.lots <= 0:
            raise ValueError("manual suggestions must include positive lots")
        return self


class AutonomousSuggestion(FillmoreSuggestionBase):
    entry_type: Literal["ai_autonomous"] = suggestion_tracker.ENTRY_TYPE_FILLMORE_AUTONOMOUS
    order_type: Literal["market", "limit"] = "market"
    quality: Literal["A", "B", "C", "A+", "B+", "C+"] = "C"
    exit_plan: str = "default"
    zone_memory_read: str | None = None
    repeat_trade_case: str | None = None
    planned_rr_estimate: float | None = None
    low_rr_edge: str | None = None
    timeframe_alignment: str | None = None
    countertrend_edge: str | None = None
    trigger_fit: str | None = None
    why_trade_despite_weakness: str | None = None
    custom_exit_plan: dict[str, Any] = Field(default_factory=dict)

    @field_validator("order_type", mode="before")
    @classmethod
    def _normalize_order_type(cls, value: Any) -> str:
        order_type = str(value or "market").strip().lower()
        if order_type not in {"market", "limit"}:
            return "market"
        return order_type

    @field_validator("quality", mode="before")
    @classmethod
    def _normalize_quality(cls, value: Any) -> str:
        quality = str(value or "C").strip().upper()
        if quality not in {"A", "B", "C", "A+", "B+", "C+"}:
            return "C"
        return quality

    @field_validator("exit_plan", mode="before")
    @classmethod
    def _normalize_exit_plan(cls, value: Any) -> str:
        text = str(value or "default").strip()
        return text or "default"

    @field_validator(
        "zone_memory_read",
        "repeat_trade_case",
        "low_rr_edge",
        "timeframe_alignment",
        "countertrend_edge",
        "trigger_fit",
        "why_trade_despite_weakness",
        mode="before",
    )
    @classmethod
    def _normalize_optional_text(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("custom_exit_plan", mode="before")
    @classmethod
    def _normalize_custom_exit_plan(cls, value: Any) -> dict[str, Any]:
        return value if isinstance(value, dict) else {}

    @model_validator(mode="after")
    def _validate_placed_trade_pricing(self) -> "AutonomousSuggestion":
        if self.lots > 0 and (self.price <= 0 or self.sl <= 0 or self.tp <= 0):
            raise ValueError("autonomous trades with lots > 0 must include positive price/sl/tp")
        return self


def validate_manual_suggestion(payload: dict[str, Any]) -> dict[str, Any]:
    suggestion = ManualSuggestion.model_validate(payload)
    return suggestion.model_dump()


def validate_autonomous_suggestion(payload: dict[str, Any]) -> dict[str, Any]:
    suggestion = AutonomousSuggestion.model_validate(payload)
    return suggestion.model_dump()


__all__ = [
    "AutonomousSuggestion",
    "FillmoreSuggestionBase",
    "ManualSuggestion",
    "ValidationError",
    "validate_autonomous_suggestion",
    "validate_manual_suggestion",
]
