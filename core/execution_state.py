from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


Mode = Literal["DISARMED", "ARMED_MANUAL_CONFIRM", "ARMED_AUTO_DEMO"]


@dataclass(frozen=True)
class RuntimeState:
    mode: Mode = "DISARMED"
    kill_switch: bool = False

    # Used for idempotency / loop progress tracking
    last_processed_bar_time_utc: Optional[str] = None

    # Temporary EMA overrides for Trial #3 (Apply Temporary Settings)
    temp_m5_trend_ema_fast: Optional[int] = None
    temp_m5_trend_ema_slow: Optional[int] = None
    temp_m1_zone_entry_ema_slow: Optional[int] = None
    temp_m1_pullback_cross_ema_slow: Optional[int] = None

    # Temporary EMA overrides for Trial #4 (Apply Temporary Settings)
    temp_m3_trend_ema_fast: Optional[int] = None
    temp_m3_trend_ema_slow: Optional[int] = None
    temp_m1_t4_zone_entry_ema_fast: Optional[int] = None
    temp_m1_t4_zone_entry_ema_slow: Optional[int] = None

    # Trial #4 Tiered Pullback State (dynamic dict: EMA period -> fired bool)
    tier_fired: dict = field(default_factory=dict)

    # RSI Divergence Block State (Trial #4)
    # ISO timestamps indicating when the block expires
    divergence_block_buy_until: Optional[str] = None
    divergence_block_sell_until: Optional[str] = None

    # Daily Reset Block (Trial #5)
    daily_reset_date: Optional[str] = None        # "YYYY-MM-DD" of current tracking day
    daily_reset_high: Optional[float] = None       # Tracked daily high from ticks
    daily_reset_low: Optional[float] = None        # Tracked daily low from ticks
    daily_reset_block_active: bool = False          # True during dead zone (21:00-02:00 UTC)
    daily_reset_settled: bool = False               # True when outside dead zone (H/L is usable)

    # Trend Extension Exhaustion (Trial #5)
    trend_flip_price: Optional[float] = None       # Price at last M3 EMA9/EMA21 trend flip
    trend_flip_direction: Optional[str] = None     # "bull" or "bear"


def _load_tier_fired(data: dict) -> dict:
    """Load tier_fired from JSON data with backward compat for old tier_X_fired keys."""
    # New format: single dict
    if "tier_fired" in data and isinstance(data["tier_fired"], dict):
        return {int(k): bool(v) for k, v in data["tier_fired"].items()}
    # Legacy format: individual tier_X_fired keys
    result = {}
    for key, val in data.items():
        if key.startswith("tier_") and key.endswith("_fired") and key != "tier_fired":
            try:
                period = int(key.replace("tier_", "").replace("_fired", ""))
                result[period] = bool(val)
            except ValueError:
                pass
    return result


def load_state(path: str | Path) -> RuntimeState:
    p = Path(path)
    if not p.exists():
        return RuntimeState()
    data = json.loads(p.read_text(encoding="utf-8"))
    return RuntimeState(
        mode=data.get("mode", "DISARMED"),
        kill_switch=bool(data.get("kill_switch", False)),
        last_processed_bar_time_utc=data.get("last_processed_bar_time_utc"),
        temp_m5_trend_ema_fast=data.get("temp_m5_trend_ema_fast"),
        temp_m5_trend_ema_slow=data.get("temp_m5_trend_ema_slow"),
        temp_m1_zone_entry_ema_slow=data.get("temp_m1_zone_entry_ema_slow"),
        temp_m1_pullback_cross_ema_slow=data.get("temp_m1_pullback_cross_ema_slow"),
        temp_m3_trend_ema_fast=data.get("temp_m3_trend_ema_fast"),
        temp_m3_trend_ema_slow=data.get("temp_m3_trend_ema_slow"),
        temp_m1_t4_zone_entry_ema_fast=data.get("temp_m1_t4_zone_entry_ema_fast"),
        temp_m1_t4_zone_entry_ema_slow=data.get("temp_m1_t4_zone_entry_ema_slow"),
        tier_fired=_load_tier_fired(data),
        divergence_block_buy_until=data.get("divergence_block_buy_until"),
        divergence_block_sell_until=data.get("divergence_block_sell_until"),
        daily_reset_date=data.get("daily_reset_date"),
        daily_reset_high=data.get("daily_reset_high"),
        daily_reset_low=data.get("daily_reset_low"),
        daily_reset_block_active=bool(data.get("daily_reset_block_active", False)),
        daily_reset_settled=bool(data.get("daily_reset_settled", False)),
        trend_flip_price=data.get("trend_flip_price"),
        trend_flip_direction=data.get("trend_flip_direction"),
    )


def save_state(path: str | Path, state: RuntimeState) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {
                "mode": state.mode,
                "kill_switch": state.kill_switch,
                "last_processed_bar_time_utc": state.last_processed_bar_time_utc,
                "temp_m5_trend_ema_fast": state.temp_m5_trend_ema_fast,
                "temp_m5_trend_ema_slow": state.temp_m5_trend_ema_slow,
                "temp_m1_zone_entry_ema_slow": state.temp_m1_zone_entry_ema_slow,
                "temp_m1_pullback_cross_ema_slow": state.temp_m1_pullback_cross_ema_slow,
                "temp_m3_trend_ema_fast": state.temp_m3_trend_ema_fast,
                "temp_m3_trend_ema_slow": state.temp_m3_trend_ema_slow,
                "temp_m1_t4_zone_entry_ema_fast": state.temp_m1_t4_zone_entry_ema_fast,
                "temp_m1_t4_zone_entry_ema_slow": state.temp_m1_t4_zone_entry_ema_slow,
                "tier_fired": state.tier_fired,
                "divergence_block_buy_until": state.divergence_block_buy_until,
                "divergence_block_sell_until": state.divergence_block_sell_until,
                "daily_reset_date": state.daily_reset_date,
                "daily_reset_high": state.daily_reset_high,
                "daily_reset_low": state.daily_reset_low,
                "daily_reset_block_active": state.daily_reset_block_active,
                "daily_reset_settled": state.daily_reset_settled,
                "trend_flip_price": state.trend_flip_price,
                "trend_flip_direction": state.trend_flip_direction,
            },
            indent=2,
            sort_keys=False,
        )
        + "\n",
        encoding="utf-8",
    )

