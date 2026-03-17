"""No-Trade Zone (NTZ) filter for Trial #9.

Blocks entries when price is within a configurable buffer of major liquidity levels:
- Previous Day High/Low (D1 candle)
- Weekly High/Low (W candle)
- Monthly High/Low (MN candle)
- Fibonacci Pivot levels (PP/R1/R2/R3/S1/S2/S3) computed from previous daily H/L/C
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
        # Fibonacci Pivot extension
        use_fib_pivots: bool = False,
        use_fib_pp: bool = True,
        use_fib_r1: bool = True,
        use_fib_r2: bool = True,
        use_fib_r3: bool = True,
        use_fib_s1: bool = True,
        use_fib_s2: bool = True,
        use_fib_s3: bool = True,
    ):
        self.enabled = enabled
        self.buffer_pips = buffer_pips
        self.pip_size = pip_size
        self.use_prev_day_hl = use_prev_day_hl
        self.use_weekly_hl = use_weekly_hl
        self.use_monthly_hl = use_monthly_hl

        # Fib pivot toggles
        self.use_fib_pivots = use_fib_pivots
        self.use_fib_pp = use_fib_pp
        self.use_fib_r1 = use_fib_r1
        self.use_fib_r2 = use_fib_r2
        self.use_fib_r3 = use_fib_r3
        self.use_fib_s1 = use_fib_s1
        self.use_fib_s2 = use_fib_s2
        self.use_fib_s3 = use_fib_s3

        # Level storage (set via update_levels)
        self.prev_day_high: float | None = None
        self.prev_day_low: float | None = None
        self.weekly_high: float | None = None
        self.weekly_low: float | None = None
        self.monthly_high: float | None = None
        self.monthly_low: float | None = None

        # Fibonacci pivot levels (set via update_fib_levels)
        self.fib_levels: dict[str, float] = {}  # e.g. {"PP": 150.5, "R1": 150.8, ...}

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

    def update_fib_levels(self, fib_levels: dict[str, float] | None) -> None:
        """Store computed Fibonacci pivot levels. Keys: P, R1, R2, R3, S1, S2, S3."""
        self.fib_levels = dict(fib_levels) if fib_levels else {}

    def _get_active_fib_levels(self) -> list[tuple[str, float]]:
        """Return list of (label, value) for enabled fib levels."""
        if not self.use_fib_pivots or not self.fib_levels:
            return []
        mapping = {
            "PP": self.use_fib_pp,
            "R1": self.use_fib_r1,
            "R2": self.use_fib_r2,
            "R3": self.use_fib_r3,
            "S1": self.use_fib_s1,
            "S2": self.use_fib_s2,
            "S3": self.use_fib_s3,
        }
        result = []
        for key, enabled in mapping.items():
            if enabled:
                # Fib levels use "P" internally, display as "PP"
                source_key = "P" if key == "PP" else key
                val = self.fib_levels.get(source_key)
                if val is not None:
                    result.append((f"Fib-{key}", val))
        return result

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

        # Add active fib pivot levels
        for label, val in self._get_active_fib_levels():
            levels.append((label, val))

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

        # Add fib levels to snapshot
        fib_snapshot: dict = {}
        if self.use_fib_pivots and self.fib_levels:
            for label, val in self._get_active_fib_levels():
                levels[label] = val
                fib_snapshot[label] = val

        return {
            "enabled": self.enabled,
            "buffer_pips": self.buffer_pips,
            "levels": levels,
            "fib_pivots_enabled": self.use_fib_pivots,
            "fib_levels": fib_snapshot,
        }
