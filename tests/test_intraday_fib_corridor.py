"""Tests for the Intraday Fibonacci Corridor feature (Trial #9)."""
from __future__ import annotations

import pytest
import pandas as pd
import numpy as np

from core.fib_pivots import compute_rolling_intraday_fib_levels, resolve_fib_level, FIB_LEVEL_NAMES
from core.no_trade_zone import IntradayFibCorridorFilter


# ---------------------------------------------------------------------------
# Helper: build a simple M15-like DataFrame
# ---------------------------------------------------------------------------
def _make_m15_df(n: int, base_high: float = 150.5, base_low: float = 150.0) -> pd.DataFrame:
    """Create a DataFrame with n rows of OHLC data."""
    highs = [base_high + i * 0.01 for i in range(n)]
    lows = [base_low - i * 0.01 for i in range(n)]
    opens = [(h + l) / 2 for h, l in zip(highs, lows)]
    closes = opens[:]
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes})


# ===========================================================================
# 1. Rolling intraday fib calculation
# ===========================================================================
class TestComputeRollingIntradayFibLevels:
    def test_basic_computation(self):
        """Verify fib levels are computed from rolling high/low."""
        df = _make_m15_df(20)
        result = compute_rolling_intraday_fib_levels(df, lookback_bars=16)
        assert result is not None
        assert "PP" in result
        assert "S1" in result
        assert "R1" in result
        assert "S3" in result
        assert "R3" in result

    def test_pp_is_midpoint(self):
        """PP should be the midpoint of the rolling range."""
        df = _make_m15_df(20, base_high=151.0, base_low=150.0)
        result = compute_rolling_intraday_fib_levels(df, lookback_bars=16)
        assert result is not None
        # Completed bars are [0..18], window is [3..18]
        window = df.iloc[:-1].iloc[-16:]
        expected_pp = (float(window["high"].max()) + float(window["low"].min())) / 2.0
        assert abs(result["PP"] - expected_pp) < 1e-10

    def test_level_ordering(self):
        """S3 < S2 < S1 < PP < R1 < R2 < R3."""
        df = _make_m15_df(20)
        result = compute_rolling_intraday_fib_levels(df, lookback_bars=16)
        assert result is not None
        assert result["S3"] < result["S2"] < result["S1"] < result["PP"]
        assert result["PP"] < result["R1"] < result["R2"] < result["R3"]

    def test_none_when_empty(self):
        """Returns None for empty DataFrame."""
        df = pd.DataFrame({"open": [], "high": [], "low": [], "close": []})
        assert compute_rolling_intraday_fib_levels(df) is None

    def test_none_when_none(self):
        """Returns None when df is None."""
        assert compute_rolling_intraday_fib_levels(None) is None

    def test_none_when_insufficient_data(self):
        """Returns None when fewer bars than lookback."""
        df = _make_m15_df(5)
        assert compute_rolling_intraday_fib_levels(df, lookback_bars=16) is None

    def test_excludes_current_bar(self):
        """Last row (in-progress) should not be used."""
        df = _make_m15_df(20)
        # Add a wild bar at the end
        df.loc[len(df)] = {"open": 200, "high": 200, "low": 100, "close": 150}
        result_with_wild = compute_rolling_intraday_fib_levels(df, lookback_bars=16)
        result_without = compute_rolling_intraday_fib_levels(df.iloc[:-1], lookback_bars=16)
        # The wild bar at the end is dropped (current in-progress), so the
        # completed bars for result_with_wild are df[:-1], and for
        # result_without they are df[:-2]. These differ, but the key point
        # is that result_with_wild did NOT use the 200/100 values.
        assert result_with_wild is not None
        assert result_with_wild["R3"] < 155  # Would be ~200 if wild bar included

    def test_zero_range_returns_none(self):
        """If all highs equal all lows, range is 0 → returns None."""
        df = pd.DataFrame({
            "open": [150.0] * 20,
            "high": [150.0] * 20,
            "low": [150.0] * 20,
            "close": [150.0] * 20,
        })
        assert compute_rolling_intraday_fib_levels(df, lookback_bars=16) is None


# ===========================================================================
# 2. Corridor boundary resolution
# ===========================================================================
class TestResolveFibLevel:
    def test_resolve_known_level(self):
        levels = {"PP": 150.0, "S1": 149.5, "R1": 150.5}
        assert resolve_fib_level(levels, "PP") == 150.0
        assert resolve_fib_level(levels, "S1") == 149.5

    def test_resolve_unknown_level(self):
        levels = {"PP": 150.0}
        assert resolve_fib_level(levels, "R3") is None


class TestFibLevelNames:
    def test_ordering(self):
        assert FIB_LEVEL_NAMES == ("S3", "S2", "S1", "PP", "R1", "R2", "R3")


# ===========================================================================
# 3. Allow/block behavior inside vs outside corridor
# ===========================================================================
class TestCorridorAllowBlock:
    def _make_filter(self, **kw) -> IntradayFibCorridorFilter:
        defaults = dict(
            enabled=True, lower_level="S1", upper_level="R1",
            boundary_buffer_pips=0.0, hysteresis_pips=0.0, pip_size=0.01,
        )
        defaults.update(kw)
        f = IntradayFibCorridorFilter(**defaults)
        f.update_levels({"PP": 150.0, "S1": 149.5, "R1": 150.5, "S2": 149.0, "R2": 151.0, "S3": 148.5, "R3": 151.5})
        return f

    def test_inside_corridor_allowed(self):
        f = self._make_filter()
        allowed, reason = f.check_corridor(150.0)
        assert allowed is True
        assert reason == ""

    def test_above_corridor_blocked(self):
        f = self._make_filter()
        allowed, reason = f.check_corridor(150.8)
        assert allowed is False
        assert "above" in reason

    def test_below_corridor_blocked(self):
        f = self._make_filter()
        allowed, reason = f.check_corridor(149.2)
        assert allowed is False
        assert "below" in reason

    def test_at_boundary_allowed(self):
        """Price exactly at boundary should be inside."""
        f = self._make_filter()
        allowed, _ = f.check_corridor(150.5)  # exactly at R1
        assert allowed is True

    def test_disabled_always_allows(self):
        f = self._make_filter(enabled=False)
        allowed, _ = f.check_corridor(200.0)  # way outside
        assert allowed is True

    def test_missing_levels_allows(self):
        f = self._make_filter()
        f.update_levels(None)
        allowed, reason = f.check_corridor(150.0)
        assert allowed is True
        assert "awaiting" in reason

    def test_invalid_bounds_blocks(self):
        """If lower >= upper, should block with clear reason."""
        f = self._make_filter(lower_level="R1", upper_level="S1")
        allowed, reason = f.check_corridor(150.0)
        assert allowed is False
        assert "invalid" in reason.lower()

    def test_buffer_expands_corridor(self):
        """With buffer_pips=2, price 2p outside should still be allowed."""
        f = self._make_filter(boundary_buffer_pips=2.0)
        # R1=150.5, buffer=0.02 → effective upper = 150.52
        allowed, _ = f.check_corridor(150.51)
        assert allowed is True
        # But 150.53 should be blocked
        f.reset_state()
        allowed, _ = f.check_corridor(150.53)
        assert allowed is False


# ===========================================================================
# 4. Hysteresis behavior
# ===========================================================================
class TestCorridorHysteresis:
    def _make_filter(self) -> IntradayFibCorridorFilter:
        f = IntradayFibCorridorFilter(
            enabled=True, lower_level="S1", upper_level="R1",
            boundary_buffer_pips=0.0, hysteresis_pips=2.0,  # 2 pips = 0.02
            pip_size=0.01,
        )
        f.update_levels({"PP": 150.0, "S1": 149.5, "R1": 150.5})
        return f

    def test_no_flap_on_boundary(self):
        """Once inside, should not flip to blocked until price moves beyond by hysteresis."""
        f = self._make_filter()
        # Start inside
        allowed, _ = f.check_corridor(150.0)
        assert allowed is True
        # Price at R1 boundary — still inside (need hysteresis to flip)
        allowed, _ = f.check_corridor(150.5)
        assert allowed is True
        # Price slightly past R1 — still inside (within hysteresis)
        allowed, _ = f.check_corridor(150.51)
        assert allowed is True
        # Price past R1 + hysteresis (0.02) = 150.52 → NOW blocked
        allowed, _ = f.check_corridor(150.53)
        assert allowed is False

    def test_no_flap_back_to_allowed(self):
        """Once blocked, should not flip back until price is inside by hysteresis."""
        f = self._make_filter()
        # Start outside
        allowed, _ = f.check_corridor(150.6)
        assert allowed is False
        # Price back at R1 boundary — still blocked
        allowed, _ = f.check_corridor(150.5)
        assert allowed is False
        # Price slightly inside — still blocked (within hysteresis)
        allowed, _ = f.check_corridor(150.49)
        assert allowed is False
        # Price well inside — NOW allowed (need to be inside by hysteresis)
        allowed, _ = f.check_corridor(150.47)
        assert allowed is True

    def test_reset_clears_hysteresis(self):
        """After reset, state is undecided and uses raw position."""
        f = self._make_filter()
        # Make it blocked
        f.check_corridor(150.6)
        # Reset
        f.reset_state()
        # Now inside should be allowed immediately
        allowed, _ = f.check_corridor(150.0)
        assert allowed is True


# ===========================================================================
# 5. Missing or insufficient M15 data
# ===========================================================================
class TestInsufficientData:
    def test_returns_none_for_short_df(self):
        df = _make_m15_df(3)
        assert compute_rolling_intraday_fib_levels(df, lookback_bars=16) is None

    def test_filter_handles_no_levels_gracefully(self):
        f = IntradayFibCorridorFilter(enabled=True, pip_size=0.01)
        # No levels set
        allowed, reason = f.check_corridor(150.0)
        assert allowed is True
        assert "awaiting" in reason

    def test_single_row_df(self):
        """Single row: no completed bars to split off, uses the row as-is."""
        df = _make_m15_df(1)
        # With 1 row, completed==df (no split), lookback=1 → uses that row
        result = compute_rolling_intraday_fib_levels(df, lookback_bars=1)
        assert result is not None  # valid since the single row has range > 0

    def test_two_rows_lookback_1(self):
        """2 rows → 1 completed bar, lookback=1 should work."""
        df = _make_m15_df(2, base_high=151.0, base_low=150.0)
        result = compute_rolling_intraday_fib_levels(df, lookback_bars=1)
        assert result is not None


# ===========================================================================
# 6. Default behavior unchanged when mode is off
# ===========================================================================
class TestDefaultBehaviorUnchanged:
    def test_disabled_filter_always_allows(self):
        f = IntradayFibCorridorFilter(enabled=False, pip_size=0.01)
        f.update_levels({"PP": 150.0, "S1": 149.5, "R1": 150.5})
        allowed, reason = f.check_corridor(200.0)
        assert allowed is True
        assert reason == ""

    def test_snapshot_shows_disabled(self):
        f = IntradayFibCorridorFilter(enabled=False, pip_size=0.01)
        snap = f.get_snapshot()
        assert snap["enabled"] is False
        assert snap["corridor_state"] is None

    def test_profile_defaults_off(self):
        """ExecutionPolicyKtCgTrial9 should default intraday_fib_enabled=False."""
        from core.profile import ExecutionPolicyKtCgTrial9
        pol = ExecutionPolicyKtCgTrial9()
        assert pol.intraday_fib_enabled is False
        assert pol.intraday_fib_timeframe == "M15"
        assert pol.intraday_fib_lookback_bars == 16
        assert pol.intraday_fib_lower_level == "S1"
        assert pol.intraday_fib_upper_level == "R1"


# ===========================================================================
# 7. Snapshot for dashboard
# ===========================================================================
class TestSnapshot:
    def test_snapshot_contains_all_fields(self):
        f = IntradayFibCorridorFilter(
            enabled=True, lower_level="S2", upper_level="R2",
            timeframe="M5", lookback_bars=24,
            boundary_buffer_pips=1.5, hysteresis_pips=2.0, pip_size=0.01,
        )
        f.update_levels(
            {"PP": 150.0, "S1": 149.5, "R1": 150.5, "S2": 149.0, "R2": 151.0},
            rolling_high=151.0, rolling_low=149.0,
        )
        f.check_corridor(150.0)  # trigger state evaluation
        snap = f.get_snapshot()
        assert snap["enabled"] is True
        assert snap["timeframe"] == "M5"
        assert snap["lookback_bars"] == 24
        assert snap["lower_level"] == "S2"
        assert snap["upper_level"] == "R2"
        assert snap["lower_value"] == 149.0
        assert snap["upper_value"] == 151.0
        assert snap["rolling_high"] == 151.0
        assert snap["rolling_low"] == 149.0
        assert snap["corridor_state"] is True
        assert snap["fib_levels"] is not None
