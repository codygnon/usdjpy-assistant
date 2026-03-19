"""No-Trade Zone (NTZ) filter for Trial #9.

Blocks entries when price is within a configurable buffer of major liquidity levels:
- Previous Day High/Low (D1 candle)
- Weekly High/Low (W candle)
- Monthly High/Low (MN candle)
- Fibonacci Pivot levels (PP/R1/R2/R3/S1/S2/S3) computed from previous daily H/L/C

Also provides IntradayFibCorridorFilter: an allowed-corridor mode using rolling
intraday fib levels (M15/M5).  Entries are permitted only while price is inside the
selected fib corridor (e.g. S1–R1), with hysteresis to prevent flapping.
"""
from __future__ import annotations

from typing import Optional


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


class IntradayFibCorridorFilter:
    """Allow entries only while price is inside a selected fib corridor.

    Uses rolling intraday fib levels (M15 or M5) to define a lower and upper
    boundary.  Entries are allowed when price is between them, blocked otherwise.

    Hysteresis prevents rapid flapping: once allowed, price must move *beyond*
    the boundary by ``hysteresis_pips`` before switching to blocked (and vice
    versa, price must come back inside by ``hysteresis_pips`` to re-allow).
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        lower_level: str = "S1",
        upper_level: str = "R1",
        timeframe: str = "M15",
        lookback_bars: int = 16,
        boundary_buffer_pips: float = 1.0,
        hysteresis_pips: float = 1.0,
        pip_size: float = 0.01,
    ):
        self.enabled = enabled
        self.lower_level = lower_level
        self.upper_level = upper_level
        self.timeframe = timeframe
        self.lookback_bars = lookback_bars
        self.boundary_buffer_pips = boundary_buffer_pips
        self.hysteresis_pips = hysteresis_pips
        self.pip_size = pip_size

        # Current computed fib levels (set via update_levels)
        self.fib_levels: Optional[dict[str, float]] = None
        # Rolling range used for computation (for reporting)
        self.rolling_high: Optional[float] = None
        self.rolling_low: Optional[float] = None

        # Hysteresis state: None = undecided, True = inside/allowed, False = outside/blocked
        self._corridor_state: Optional[bool] = None

    def update_levels(
        self,
        fib_levels: Optional[dict[str, float]],
        rolling_high: Optional[float] = None,
        rolling_low: Optional[float] = None,
    ) -> None:
        """Update the computed intraday fib levels. Called each poll cycle."""
        self.fib_levels = dict(fib_levels) if fib_levels else None
        self.rolling_high = rolling_high
        self.rolling_low = rolling_low

    def _resolve_bounds(self) -> tuple[Optional[float], Optional[float]]:
        """Resolve the lower and upper corridor boundaries from fib level names."""
        if not self.fib_levels:
            return None, None
        lower = self.fib_levels.get(self.lower_level)
        upper = self.fib_levels.get(self.upper_level)
        return lower, upper

    def check_corridor(self, current_price: float) -> tuple[bool, str]:
        """Check if price is inside the allowed corridor.

        Returns (allowed, reason).
        - allowed=True  means entries may proceed.
        - allowed=False means entries should be blocked; reason explains why.
        """
        if not self.enabled:
            return True, ""

        lower, upper = self._resolve_bounds()
        if lower is None or upper is None:
            return True, "intraday_fib_corridor: awaiting level data"

        # Guard against invalid config (lower >= upper)
        if lower >= upper:
            return False, (
                f"intraday_fib_corridor: invalid bounds {self.lower_level}={lower:.3f} >= "
                f"{self.upper_level}={upper:.3f}"
            )

        buffer = self.boundary_buffer_pips * self.pip_size
        hysteresis = self.hysteresis_pips * self.pip_size

        # Determine raw position (with buffer expanding the corridor slightly)
        price_above_upper = current_price > upper + buffer
        price_below_lower = current_price < lower - buffer

        raw_inside = not price_above_upper and not price_below_lower

        # Apply hysteresis
        if self._corridor_state is None:
            # First evaluation: use raw position
            self._corridor_state = raw_inside
        elif self._corridor_state:
            # Currently allowed (inside). Only flip to blocked if price moves
            # beyond boundary by hysteresis_pips past the buffer edge.
            if current_price > upper + buffer + hysteresis:
                self._corridor_state = False
            elif current_price < lower - buffer - hysteresis:
                self._corridor_state = False
        else:
            # Currently blocked (outside). Only flip back to allowed if price
            # comes back inside by hysteresis_pips past the buffer edge.
            if lower - buffer + hysteresis <= current_price <= upper + buffer - hysteresis:
                self._corridor_state = True

        if self._corridor_state:
            return True, ""

        # Blocked — determine direction
        if current_price > upper:
            return False, (
                f"intraday_fib_corridor: price {current_price:.3f} above "
                f"{self.upper_level}={upper:.3f} (blocked)"
            )
        elif current_price < lower:
            return False, (
                f"intraday_fib_corridor: price {current_price:.3f} below "
                f"{self.lower_level}={lower:.3f} (blocked)"
            )
        else:
            return False, (
                f"intraday_fib_corridor: price {current_price:.3f} outside corridor "
                f"(hysteresis holding blocked state)"
            )

    def reset_state(self) -> None:
        """Reset hysteresis state (e.g. on settings change)."""
        self._corridor_state = None

    def get_snapshot(self) -> dict:
        """Return current state for dashboard/reporting."""
        lower, upper = self._resolve_bounds()
        return {
            "enabled": self.enabled,
            "timeframe": self.timeframe,
            "lookback_bars": self.lookback_bars,
            "lower_level": self.lower_level,
            "upper_level": self.upper_level,
            "lower_value": lower,
            "upper_value": upper,
            "boundary_buffer_pips": self.boundary_buffer_pips,
            "hysteresis_pips": self.hysteresis_pips,
            "rolling_high": self.rolling_high,
            "rolling_low": self.rolling_low,
            "corridor_state": self._corridor_state,  # True=inside, False=outside, None=undecided
            "fib_levels": dict(self.fib_levels) if self.fib_levels else None,
        }
