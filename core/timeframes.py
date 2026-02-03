from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Timeframe = Literal["M1", "M3", "M5", "M15", "M30", "H1", "H4"]


@dataclass(frozen=True)
class TimeframeSpec:
    name: Timeframe
    seconds: int


TIMEFRAMES: dict[Timeframe, TimeframeSpec] = {
    "M1": TimeframeSpec("M1", 60),
    "M3": TimeframeSpec("M3", 3 * 60),
    "M5": TimeframeSpec("M5", 5 * 60),
    "M15": TimeframeSpec("M15", 15 * 60),
    "M30": TimeframeSpec("M30", 30 * 60),
    "H1": TimeframeSpec("H1", 60 * 60),
    "H4": TimeframeSpec("H4", 4 * 60 * 60),
}

