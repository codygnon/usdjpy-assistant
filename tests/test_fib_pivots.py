"""Tests for Fibonacci pivot calculation and NTZ integration."""
import pytest


class TestComputeDailyFibPivots:
    """Unit tests for the shared fib pivot calculation."""

    def test_basic_calculation(self):
        from core.fib_pivots import compute_daily_fib_pivots

        result = compute_daily_fib_pivots(150.0, 148.0, 149.0)
        p = (150.0 + 148.0 + 149.0) / 3.0
        r = 150.0 - 148.0  # 2.0

        assert abs(result["P"] - p) < 1e-10
        assert abs(result["R1"] - (p + 0.382 * r)) < 1e-10
        assert abs(result["R2"] - (p + 0.618 * r)) < 1e-10
        assert abs(result["R3"] - (p + 1.0 * r)) < 1e-10
        assert abs(result["S1"] - (p - 0.382 * r)) < 1e-10
        assert abs(result["S2"] - (p - 0.618 * r)) < 1e-10
        assert abs(result["S3"] - (p - 1.0 * r)) < 1e-10

    def test_all_keys_present(self):
        from core.fib_pivots import compute_daily_fib_pivots

        result = compute_daily_fib_pivots(100.0, 99.0, 99.5)
        assert set(result.keys()) == {"P", "R1", "R2", "R3", "S1", "S2", "S3"}

    def test_zero_range(self):
        """When H == L, all levels collapse to P."""
        from core.fib_pivots import compute_daily_fib_pivots

        result = compute_daily_fib_pivots(150.0, 150.0, 150.0)
        p = 150.0
        for key in result:
            assert abs(result[key] - p) < 1e-10

    def test_r_levels_ascending(self):
        from core.fib_pivots import compute_daily_fib_pivots

        result = compute_daily_fib_pivots(152.0, 148.0, 150.0)
        assert result["S3"] < result["S2"] < result["S1"] < result["P"] < result["R1"] < result["R2"] < result["R3"]


class TestNoTradeZoneFibPivots:
    """Tests for NTZ filter with Fibonacci pivot levels."""

    def _make_filter(self, **kwargs):
        from core.no_trade_zone import NoTradeZoneFilter
        defaults = dict(
            enabled=True,
            buffer_pips=5.0,
            pip_size=0.01,
            use_prev_day_hl=False,
            use_weekly_hl=False,
            use_monthly_hl=False,
            use_fib_pivots=True,
        )
        defaults.update(kwargs)
        return NoTradeZoneFilter(**defaults)

    def test_fib_disabled_no_blocking(self):
        """When fib pivots disabled, fib levels don't block."""
        ntz = self._make_filter(use_fib_pivots=False)
        ntz.update_fib_levels({"P": 150.0, "R1": 150.5, "S1": 149.5})
        blocked, reason = ntz.is_in_no_trade_zone(150.0)
        assert not blocked
        assert reason == ""

    def test_fib_enabled_blocks_near_pp(self):
        """Price near PP should be blocked."""
        ntz = self._make_filter()
        ntz.update_fib_levels({"P": 150.0, "R1": 150.764, "S1": 149.236})
        blocked, reason = ntz.is_in_no_trade_zone(150.02)
        assert blocked
        assert "Fib-PP" in reason

    def test_fib_blocks_near_r1(self):
        ntz = self._make_filter()
        ntz.update_fib_levels({"P": 150.0, "R1": 150.764})
        blocked, reason = ntz.is_in_no_trade_zone(150.76)
        assert blocked
        assert "Fib-R1" in reason

    def test_fib_blocks_near_s1(self):
        ntz = self._make_filter()
        ntz.update_fib_levels({"P": 150.0, "S1": 149.236})
        blocked, reason = ntz.is_in_no_trade_zone(149.24)
        assert blocked
        assert "Fib-S1" in reason

    def test_fib_clear_when_far_away(self):
        ntz = self._make_filter()
        ntz.update_fib_levels({"P": 150.0, "R1": 150.764, "S1": 149.236})
        blocked, reason = ntz.is_in_no_trade_zone(150.4)
        assert not blocked

    def test_per_level_toggle_pp_off(self):
        """Disabling PP should not block near PP."""
        ntz = self._make_filter(use_fib_pp=False)
        ntz.update_fib_levels({"P": 150.0, "R1": 150.764})
        blocked, _ = ntz.is_in_no_trade_zone(150.01)
        assert not blocked

    def test_per_level_toggle_r1_off(self):
        """Disabling R1 should not block near R1."""
        ntz = self._make_filter(use_fib_r1=False)
        ntz.update_fib_levels({"P": 150.0, "R1": 150.764})
        blocked, _ = ntz.is_in_no_trade_zone(150.76)
        assert not blocked

    def test_snapshot_includes_fib(self):
        ntz = self._make_filter()
        ntz.update_fib_levels({"P": 150.0, "R1": 150.764, "S1": 149.236})
        snap = ntz.get_levels_snapshot()
        assert snap["fib_pivots_enabled"] is True
        assert "Fib-PP" in snap["fib_levels"]
        assert "Fib-R1" in snap["fib_levels"]
        assert "Fib-S1" in snap["fib_levels"]

    def test_snapshot_no_fib_when_disabled(self):
        ntz = self._make_filter(use_fib_pivots=False)
        ntz.update_fib_levels({"P": 150.0})
        snap = ntz.get_levels_snapshot()
        assert snap["fib_pivots_enabled"] is False
        assert len(snap["fib_levels"]) == 0

    def test_missing_fib_data(self):
        """No fib levels set => no blocking from fibs."""
        ntz = self._make_filter()
        # Don't call update_fib_levels
        blocked, reason = ntz.is_in_no_trade_zone(150.0)
        assert not blocked

    def test_combined_hl_and_fib(self):
        """Both traditional NTZ and fib levels can block."""
        ntz = self._make_filter(use_prev_day_hl=True)
        ntz.update_levels(prev_day_high=151.0, prev_day_low=149.0)
        ntz.update_fib_levels({"P": 150.0})
        # Near PDH
        blocked, reason = ntz.is_in_no_trade_zone(151.01)
        assert blocked
        assert "PDH" in reason
        # Near PP
        blocked, reason = ntz.is_in_no_trade_zone(150.02)
        assert blocked
        assert "Fib-PP" in reason


class TestTrialNineUnchangedWhenFibOff:
    """Verify Trial #9 behavior unchanged when fib pivots disabled."""

    def test_ntz_filter_no_fib_same_as_original(self):
        """NTZ filter with fib off should behave identically to original."""
        from core.no_trade_zone import NoTradeZoneFilter

        ntz = NoTradeZoneFilter(
            enabled=True,
            buffer_pips=10.0,
            pip_size=0.01,
            use_prev_day_hl=True,
            use_weekly_hl=True,
            use_monthly_hl=True,
            use_fib_pivots=False,  # default OFF
        )
        ntz.update_levels(
            prev_day_high=151.0,
            prev_day_low=149.0,
            weekly_high=152.0,
            weekly_low=148.0,
        )
        # Near PDH
        blocked, reason = ntz.is_in_no_trade_zone(151.05)
        assert blocked
        assert "PDH" in reason
        # Far from levels
        blocked, reason = ntz.is_in_no_trade_zone(150.0)
        assert not blocked

    def test_snapshot_backward_compatible(self):
        """Snapshot has new keys but fib_pivots_enabled=False by default."""
        from core.no_trade_zone import NoTradeZoneFilter

        ntz = NoTradeZoneFilter(enabled=True, buffer_pips=10.0, pip_size=0.01)
        snap = ntz.get_levels_snapshot()
        assert "fib_pivots_enabled" in snap
        assert snap["fib_pivots_enabled"] is False
        assert snap["fib_levels"] == {}


class TestMissingPrevDayClose:
    """Test graceful handling when prev_day_close is unavailable."""

    def test_fib_levels_empty_without_data(self):
        from core.no_trade_zone import NoTradeZoneFilter

        ntz = NoTradeZoneFilter(
            enabled=True, buffer_pips=5.0, pip_size=0.01,
            use_fib_pivots=True,
        )
        # Never call update_fib_levels
        blocked, reason = ntz.is_in_no_trade_zone(150.0)
        assert not blocked

    def test_update_fib_levels_none(self):
        from core.no_trade_zone import NoTradeZoneFilter

        ntz = NoTradeZoneFilter(
            enabled=True, buffer_pips=5.0, pip_size=0.01,
            use_fib_pivots=True,
        )
        ntz.update_fib_levels(None)
        assert ntz.fib_levels == {}
        blocked, _ = ntz.is_in_no_trade_zone(150.0)
        assert not blocked
