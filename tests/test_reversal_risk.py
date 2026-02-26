"""
Unit tests for core reversal risk calculations.

Tests are self-contained — they implement or mock the minimal logic needed
so the suite runs without live broker connections or large data files.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.reversal_risk import (
    _clamp,
    _compute_adr_exhaustion_component,
    _compute_weighted_score,
    _rsi_severity_from_delta,
    _score_to_tier,
    _tier_rank,
    build_reversal_risk_response,
)


class TestClamp(unittest.TestCase):
    def test_below_lo(self):
        self.assertEqual(_clamp(-1.0, 0.0, 1.0), 0.0)

    def test_above_hi(self):
        self.assertEqual(_clamp(2.0, 0.0, 1.0), 1.0)

    def test_within(self):
        self.assertAlmostEqual(_clamp(0.5, 0.0, 1.0), 0.5)


class TestTierRank(unittest.TestCase):
    def test_ordering(self):
        self.assertLess(_tier_rank("low"), _tier_rank("medium"))
        self.assertLess(_tier_rank("medium"), _tier_rank("high"))
        self.assertLess(_tier_rank("high"), _tier_rank("critical"))

    def test_unknown_is_low(self):
        self.assertEqual(_tier_rank("unknown"), 0)
        self.assertEqual(_tier_rank(""), 0)


class TestScoreToTier(unittest.TestCase):
    def test_below_medium(self):
        self.assertEqual(_score_to_tier(50.0, 58.0, 65.0, 71.0), "low")

    def test_at_medium(self):
        self.assertEqual(_score_to_tier(58.0, 58.0, 65.0, 71.0), "medium")

    def test_at_high(self):
        self.assertEqual(_score_to_tier(65.0, 58.0, 65.0, 71.0), "high")

    def test_at_critical(self):
        self.assertEqual(_score_to_tier(71.0, 58.0, 65.0, 71.0), "critical")

    def test_above_critical(self):
        self.assertEqual(_score_to_tier(99.0, 58.0, 65.0, 71.0), "critical")


class TestRsiSeverity(unittest.TestCase):
    def test_below_min_delta(self):
        self.assertEqual(_rsi_severity_from_delta(5.0, 8.0), 0.0)

    def test_below_10(self):
        # delta=9, min_delta=8 → passes min_delta but < 10 → 0.0
        self.assertEqual(_rsi_severity_from_delta(9.0, 8.0), 0.0)

    def test_at_10(self):
        self.assertEqual(_rsi_severity_from_delta(10.0, 8.0), 0.2)

    def test_at_18(self):
        self.assertEqual(_rsi_severity_from_delta(18.0, 8.0), 0.2)

    def test_at_36(self):
        # 0.2 + (36-18)/18 * 0.8 = 0.2 + 0.8 = 1.0
        self.assertAlmostEqual(_rsi_severity_from_delta(36.0, 8.0), 1.0, places=5)

    def test_above_36(self):
        self.assertAlmostEqual(_rsi_severity_from_delta(50.0, 8.0), 1.0, places=5)


class TestWeightedScore(unittest.TestCase):
    def _components(self, rsi=0.0, adr=0.0, htf=0.0, spread=0.0):
        return {
            "rsi_divergence": {"score": rsi},
            "adr_exhaustion": {"score": adr},
            "htf_proximity": {"score": htf},
            "ema_spread": {"score": spread},
        }

    def test_all_zero_components(self):
        score, _ = _compute_weighted_score(
            components=self._components(),
            w_rsi=55, w_adr=20, w_htf=15, w_spread=10,
        )
        self.assertAlmostEqual(score, 0.0)

    def test_all_one_components(self):
        score, _ = _compute_weighted_score(
            components=self._components(1.0, 1.0, 1.0, 1.0),
            w_rsi=55, w_adr=20, w_htf=15, w_spread=10,
        )
        self.assertAlmostEqual(score, 100.0)

    def test_weight_normalization_independent_of_magnitude(self):
        # Score should be the same regardless of whether weights are 55/20/15/10 or 550/200/150/100
        comps = self._components(0.5, 0.5, 0.5, 0.5)
        score1, _ = _compute_weighted_score(components=comps, w_rsi=55, w_adr=20, w_htf=15, w_spread=10)
        score2, _ = _compute_weighted_score(components=comps, w_rsi=550, w_adr=200, w_htf=150, w_spread=100)
        self.assertAlmostEqual(score1, score2, places=4)

    def test_all_zero_weights_falls_back_to_defaults(self):
        # Zero total weight should fall back to default weights, not crash
        comps = self._components(1.0, 0.0, 0.0, 0.0)
        score, _ = _compute_weighted_score(components=comps, w_rsi=0, w_adr=0, w_htf=0, w_spread=0)
        # With default weights (55/20/15/10), all-RSI components → 55/100 = 0.55 → score 55
        self.assertAlmostEqual(score, 55.0, places=1)

    def test_score_clamped_0_to_100(self):
        comps = self._components(2.0, 2.0, 2.0, 2.0)  # components > 1 (shouldn't happen but test clamping)
        score, _ = _compute_weighted_score(components=comps, w_rsi=55, w_adr=20, w_htf=15, w_spread=10)
        self.assertLessEqual(score, 100.0)
        self.assertGreaterEqual(score, 0.0)


class TestBuildReversalRiskResponse(unittest.TestCase):
    def _policy(self, **kwargs):
        defaults = {
            "rr_medium_lot_multiplier": 0.75,
            "rr_high_lot_multiplier": 0.50,
            "rr_critical_lot_multiplier": 0.25,
            "rr_high_min_tier_ema": 21,
            "rr_critical_min_tier_ema": 26,
            "rr_block_zone_entry_above_tier": "high",
            "rr_adjust_exhaustion_thresholds": True,
            "rr_exhaustion_medium_threshold_boost_pips": 0.5,
            "rr_exhaustion_high_threshold_boost_pips": 1.0,
            "rr_exhaustion_critical_threshold_boost_pips": 1.5,
            "rr_use_managed_exit_at": "high",
        }
        defaults.update(kwargs)
        return SimpleNamespace(**defaults)

    def test_low_tier_full_lots(self):
        resp = build_reversal_risk_response(self._policy(), "low")
        self.assertAlmostEqual(resp["lot_multiplier"], 1.0)
        self.assertIsNone(resp["min_tier_ema"])
        self.assertFalse(resp["block_zone_entry"])
        self.assertFalse(resp["use_managed_exit"])

    def test_medium_tier_reduced_lots(self):
        resp = build_reversal_risk_response(self._policy(), "medium")
        self.assertAlmostEqual(resp["lot_multiplier"], 0.75)
        self.assertIsNone(resp["min_tier_ema"])

    def test_high_tier_min_tier_ema_set(self):
        resp = build_reversal_risk_response(self._policy(), "high")
        self.assertAlmostEqual(resp["lot_multiplier"], 0.50)
        self.assertEqual(resp["min_tier_ema"], 21)
        self.assertTrue(resp["block_zone_entry"])  # "high" >= "high" threshold
        self.assertTrue(resp["use_managed_exit"])  # "high" >= "high" threshold

    def test_critical_tier(self):
        resp = build_reversal_risk_response(self._policy(), "critical")
        self.assertAlmostEqual(resp["lot_multiplier"], 0.25)
        self.assertEqual(resp["min_tier_ema"], 26)
        self.assertTrue(resp["block_zone_entry"])
        self.assertTrue(resp["use_managed_exit"])

    def test_lot_multiplier_floor_at_0_01(self):
        resp = build_reversal_risk_response(self._policy(rr_critical_lot_multiplier=0.0), "critical")
        self.assertGreaterEqual(resp["lot_multiplier"], 0.01)


class TestAdrExhaustionComponent(unittest.TestCase):
    def _make_daily_df(self, n_days: int = 20, daily_range: float = 0.80) -> pd.DataFrame:
        """Create a synthetic daily OHLC DataFrame."""
        from datetime import datetime, timezone, timedelta
        rows = []
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for i in range(n_days):
            d = base + timedelta(days=i)
            rows.append({
                "time": d.isoformat(),
                "open": 150.0,
                "high": 150.0 + daily_range,
                "low": 150.0,
                "close": 150.0 + daily_range / 2,
            })
        return pd.DataFrame(rows)

    def test_insufficient_data_returns_zero(self):
        result = _compute_adr_exhaustion_component(
            d_df=pd.DataFrame(),
            trial7_daily_state=None,
            pip_size=0.01,
            adr_period=14,
            ramp_start_pct=75.0,
            score_100=0.3,
            score_120=0.6,
            score_150=0.9,
        )
        self.assertEqual(result["score"], 0.0)
        self.assertIn("insufficient", result["details"])

    def test_zero_consumed_zero_score(self):
        d_df = self._make_daily_df(20, daily_range=0.80)
        state = {"today_high": 150.0, "today_low": 150.0}  # 0 pips consumed
        result = _compute_adr_exhaustion_component(
            d_df=d_df,
            trial7_daily_state=state,
            pip_size=0.01,
            adr_period=14,
            ramp_start_pct=75.0,
            score_100=0.3,
            score_120=0.6,
            score_150=0.9,
        )
        self.assertAlmostEqual(result["score"], 0.0, places=3)

    def test_100_pct_consumed_score_at_score_100(self):
        # ADR = 80 pips, today_range = 80 pips → consumed = 100%
        d_df = self._make_daily_df(20, daily_range=0.80)
        state = {"today_high": 150.80, "today_low": 150.00}
        result = _compute_adr_exhaustion_component(
            d_df=d_df,
            trial7_daily_state=state,
            pip_size=0.01,
            adr_period=14,
            ramp_start_pct=75.0,
            score_100=0.3,
            score_120=0.6,
            score_150=0.9,
        )
        self.assertAlmostEqual(result["score"], 0.3, places=2)

    def test_150_pct_consumed_score_at_score_150(self):
        # ADR = 80 pips, today_range = 120 pips → consumed = 150%
        d_df = self._make_daily_df(20, daily_range=0.80)
        state = {"today_high": 151.20, "today_low": 150.00}
        result = _compute_adr_exhaustion_component(
            d_df=d_df,
            trial7_daily_state=state,
            pip_size=0.01,
            adr_period=14,
            ramp_start_pct=75.0,
            score_100=0.3,
            score_120=0.6,
            score_150=0.9,
        )
        self.assertAlmostEqual(result["score"], 0.9, places=2)

    def test_above_180_pct_score_capped_at_1(self):
        # ADR = 80 pips, today_range = 160 pips → consumed = 200%
        d_df = self._make_daily_df(20, daily_range=0.80)
        state = {"today_high": 151.60, "today_low": 150.00}
        result = _compute_adr_exhaustion_component(
            d_df=d_df,
            trial7_daily_state=state,
            pip_size=0.01,
            adr_period=14,
            ramp_start_pct=75.0,
            score_100=0.3,
            score_120=0.6,
            score_150=0.9,
        )
        self.assertAlmostEqual(result["score"], 1.0, places=4)


class TestManagedExitLogic(unittest.TestCase):
    """Verify managed exit trigger conditions without calling broker."""

    def _make_trade_row(self, side: str, entry: float, tier: str = "high") -> dict:
        return {
            "trade_id": "test_trade_001",
            "side": side,
            "entry_price": entry,
            "reversal_risk_tier": tier,
            "timestamp_utc": "2026-02-25T10:00:00+00:00",
            "notes": "auto:kt_cg_trial_7:test_policy",
            "managed_exit_sl_price": None,
            "breakeven_sl_price": None,
            "stop_price": None,
        }

    def _make_policy(self, **kwargs):
        defaults = {
            "use_reversal_risk_score": True,
            "rr_use_managed_exit_at": "high",
            "rr_managed_exit_hard_sl_pips": 72.0,
            "rr_managed_exit_max_hold_underwater_min": 30.0,
            "rr_managed_exit_trail_activation_pips": 4.0,
            "rr_managed_exit_trail_distance_pips": 2.5,
        }
        defaults.update(kwargs)
        return SimpleNamespace(**defaults)

    def test_disabled_when_use_reversal_risk_score_false(self):
        from core.managed_exit import apply_trial7_managed_exit_for_position
        adapter = MagicMock()
        store = MagicMock()
        profile = SimpleNamespace(profile_name="test", symbol="USDJPY", pip_size=0.01)
        policy = self._make_policy(use_reversal_risk_score=False)
        tick = SimpleNamespace(bid=150.0, ask=150.02)

        apply_trial7_managed_exit_for_position(
            adapter=adapter, profile=profile, store=store, policy=policy,
            position_id=1, live_position={}, trade_row=self._make_trade_row("buy", 150.0),
            tick=tick, pip_size=0.01,
        )
        adapter.close_position.assert_not_called()
        adapter.update_position_stop_loss.assert_not_called()

    def test_disabled_when_tier_is_low(self):
        from core.managed_exit import apply_trial7_managed_exit_for_position
        adapter = MagicMock()
        store = MagicMock()
        profile = SimpleNamespace(profile_name="test", symbol="USDJPY", pip_size=0.01)
        policy = self._make_policy()
        tick = SimpleNamespace(bid=150.0, ask=150.02)

        apply_trial7_managed_exit_for_position(
            adapter=adapter, profile=profile, store=store, policy=policy,
            position_id=1, live_position={}, trade_row=self._make_trade_row("buy", 150.0, tier="low"),
            tick=tick, pip_size=0.01,
        )
        adapter.close_position.assert_not_called()

    def test_hard_sl_triggers_close(self):
        from core.managed_exit import apply_trial7_managed_exit_for_position
        adapter = MagicMock()
        store = MagicMock()
        profile = SimpleNamespace(profile_name="test", symbol="USDJPY", pip_size=0.01)
        policy = self._make_policy(rr_managed_exit_hard_sl_pips=20.0)
        # Buy entry at 150.0, current bid at 149.79 → -21 pips (below -20 hard SL)
        tick = SimpleNamespace(bid=149.79, ask=149.81)
        live_pos = {"currentUnits": "10000"}  # 0.1 lots

        apply_trial7_managed_exit_for_position(
            adapter=adapter, profile=profile, store=store, policy=policy,
            position_id=42, live_position=live_pos,
            trade_row=self._make_trade_row("buy", 150.0, tier="high"),
            tick=tick, pip_size=0.01,
        )
        adapter.close_position.assert_called_once()

    def test_trail_sl_activates_at_threshold(self):
        from core.managed_exit import apply_trial7_managed_exit_for_position
        adapter = MagicMock()
        store = MagicMock()
        profile = SimpleNamespace(profile_name="test", symbol="USDJPY", pip_size=0.01)
        policy = self._make_policy(
            rr_managed_exit_trail_activation_pips=4.0,
            rr_managed_exit_trail_distance_pips=2.5,
        )
        # Buy entry at 150.0, current bid at 150.05 → +5 pips (above 4 pip activation)
        tick = SimpleNamespace(bid=150.05, ask=150.07)

        apply_trial7_managed_exit_for_position(
            adapter=adapter, profile=profile, store=store, policy=policy,
            position_id=42, live_position={"currentUnits": "10000"},
            trade_row=self._make_trade_row("buy", 150.0, tier="high"),
            tick=tick, pip_size=0.01,
        )
        adapter.update_position_stop_loss.assert_called_once()
        # Trail SL should be bid - 2.5 pips = 150.05 - 0.025 = 150.025
        args = adapter.update_position_stop_loss.call_args
        self.assertAlmostEqual(args[0][2], 150.025, places=3)


if __name__ == "__main__":
    unittest.main()
