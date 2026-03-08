"""No-Trade Zone (NTZ) filter for Trial #9.

Blocks entries when price is within a configurable buffer of major liquidity levels:
- Previous Day High/Low (D1 candle)
- Weekly High/Low (W candle)
- Monthly High/Low (MN candle)
"""
from __future__ import annotations


class NoTradeZoneFilter:
    """Check whether current price is inside a no-trade zone around key levels."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        buffer_pips: float = 10.0,
        pip_size: float = 0.01,
        use_prev_day_hl: bool = True,
        use_weekly_hl: bool = True,
        use_monthly_hl: bool = True,
    ):
        self.enabled = enabled
        self.buffer_pips = buffer_pips
        self.pip_size = pip_size
        self.use_prev_day_hl = use_prev_day_hl
        self.use_weekly_hl = use_weekly_hl
        self.use_monthly_hl = use_monthly_hl

        # Level storage (set via update_levels)
        self.prev_day_high: float | None = None
        self.prev_day_low: float | None = None
        self.weekly_high: float | None = None
        self.weekly_low: float | None = None
        self.monthly_high: float | None = None
        self.monthly_low: float | None = None

    def update_levels(
        self,
        *,
        prev_day_high: float | None = None,
        prev_day_low: float | None = None,
        weekly_high: float | None = None,
        weekly_low: float | None = None,
        monthly_high: float | None = None,
        monthly_low: float | None = None,
    ) -> None:
        """Store the 6 major levels. Called when new candle data is fetched."""
        self.prev_day_high = prev_day_high
        self.prev_day_low = prev_day_low
        self.weekly_high = weekly_high
        self.weekly_low = weekly_low
        self.monthly_high = monthly_high
        self.monthly_low = monthly_low

    def is_in_no_trade_zone(self, current_price: float) -> tuple[bool, str]:
        """Check if price is within buffer_pips of any enabled level.

        Returns (blocked, reason).  When blocked is False, reason is empty.
        """
        if not self.enabled:
            return False, ""

        buffer = self.buffer_pips * self.pip_size
        closest_dist: float | None = None
        closest_label: str = ""

        levels: list[tuple[str, float | None]] = []
        if self.use_prev_day_hl:
            levels.append(("PDH", self.prev_day_high))
            levels.append(("PDL", self.prev_day_low))
        if self.use_weekly_hl:
            levels.append(("WH", self.weekly_high))
            levels.append(("WL", self.weekly_low))
        if self.use_monthly_hl:
            levels.append(("MH", self.monthly_high))
            levels.append(("ML", self.monthly_low))

        for label, level in levels:
            if level is None:
                continue
            dist = abs(current_price - level)
            if dist <= buffer:
                if closest_dist is None or dist < closest_dist:
                    closest_dist = dist
                    closest_label = label

        if closest_dist is not None:
            dist_pips = closest_dist / self.pip_size
            return True, f"NTZ: price within {dist_pips:.1f}p of {closest_label} (buffer={self.buffer_pips}p)"
        return False, ""

    def get_levels_snapshot(self) -> dict:
        """Return all active levels and buffer for dashboard display."""
        levels = {}
        if self.use_prev_day_hl:
            levels["PDH"] = self.prev_day_high
            levels["PDL"] = self.prev_day_low
        if self.use_weekly_hl:
            levels["WH"] = self.weekly_high
            levels["WL"] = self.weekly_low
        if self.use_monthly_hl:
            levels["MH"] = self.monthly_high
            levels["ML"] = self.monthly_low
        return {
            "enabled": self.enabled,
            "buffer_pips": self.buffer_pips,
            "levels": levels,
        }
