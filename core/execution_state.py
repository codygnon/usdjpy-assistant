from __future__ import annotations

import json
from dataclasses import dataclass
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
            },
            indent=2,
            sort_keys=False,
        )
        + "\n",
        encoding="utf-8",
    )

