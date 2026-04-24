from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api import ai_trading_chat, autonomous_fillmore, autonomous_performance, suggestion_tracker


def _ctx() -> dict:
    return {
        "spot_price": {"mid": 150.123, "bid": 150.121, "ask": 150.125, "spread_pips": 0.4},
        "session": {"active_sessions": ["New York"], "overlap": None, "warnings": []},
        "cross_asset_bias": {"combined_bias": "bearish", "confidence": "medium"},
        "volatility": {"label": "compressed", "ratio": 0.8, "recent_avg_pips": 6.2},
        "ta_snapshot": {"M5": {"regime": "pullback", "atr_pips": 4.0}},
    }


def _suggestion(*, exit_strategy: str = "none") -> dict:
    return {
        "side": "buy",
        "price": 159.250,
        "sl": 159.100,
        "tp": 159.450,
        "lots": 1.0,
        "time_in_force": "GTC",
        "gtd_time_utc": None,
        "quality": "A",
        "rationale": "Autonomous test setup.",
        "exit_strategy": exit_strategy,
        "exit_params": {},
    }


class _DummyStore:
    def __init__(self) -> None:
        self.inserted: list[dict] = []

    def insert_trade(self, row: dict) -> None:
        self.inserted.append(dict(row))


class _FakeMarketAdapter:
    def initialize(self) -> None:
        return None

    def shutdown(self) -> None:
        return None

    def order_send_market(self, **kwargs):
        return SimpleNamespace(retcode=0, comment="ok", fill_price=159.263, order=555001, deal=None)

    def get_position_id_from_deal(self, deal_id: int):
        return None

    def get_position_id_from_order(self, order_id: int):
        assert order_id == 555001
        return 777001


class _FakeLimitAdapter:
    def initialize(self) -> None:
        return None

    def shutdown(self) -> None:
        return None

    def order_send_pending_limit(self, **kwargs):
        return SimpleNamespace(retcode=0, comment="ok", order=888002)


def _install_fake_main(monkeypatch, db_path: Path) -> None:
    fake_main = ModuleType("api.main")
    fake_main._suggestions_db_path = lambda profile_name: db_path
    monkeypatch.setitem(sys.modules, "api.main", fake_main)


def _gate_cfg(aggressiveness: str) -> dict:
    cfg = dict(autonomous_fillmore.DEFAULT_CONFIG)
    cfg.update({
        "enabled": True,
        "mode": "paper",
        "aggressiveness": aggressiveness,
    })
    return cfg


def test_default_autonomous_config_enables_tokyo() -> None:
    assert autonomous_fillmore.DEFAULT_CONFIG["trading_hours"]["tokyo"] is True


def _force_allowed_session(monkeypatch) -> None:
    monkeypatch.setattr(autonomous_fillmore, "_session_flag_now", lambda trading_hours: (True, "ny"))


def _critical_trigger(**overrides) -> dict:
    base = {
        "family": "critical_level_reaction",
        "reason": "support_reclaim",
        "bias": "buy",
        "level_label": "WHOLE_YEN:159.00",
        "level_price": 159.000,
        "nearest_level_pips": 3.0,
        "micro_confirmation": "reclaim after touch",
    }
    base.update(overrides)
    return base


def _trend_trigger(**overrides) -> dict:
    base = {
        "family": "trend_expansion",
        "reason": "adx_trend_expansion",
        "bias": "buy",
        "adx": 27.0,
        "m5_atr_pips": 4.2,
        "extension_pips": 1.8,
        "extension_limit_pips": 4.0,
    }
    base.update(overrides)
    return base


def _compression_trigger(**overrides) -> dict:
    base = {
        "family": "compression_breakout",
        "reason": "compression_press:PDH",
        "bias": "buy",
        "level_label": "PDH",
        "level_price": 159.050,
        "nearest_level_pips": 2.0,
        "micro_confirmation": "compressed_range_pressing_boundary",
        "compression_range_pips": 3.2,
        "compression_cap_pips": 4.0,
        "adx": 18.0,
        "m5_atr_pips": 4.1,
    }
    base.update(overrides)
    return base


def _tokyo_meanrev_trigger(**overrides) -> dict:
    base = {
        "family": "tight_range_mean_reversion",
        "reason": "tokyo_range_reclaim:session_low",
        "bias": "buy",
        "level_label": "TOKYO_SESSION_LOW",
        "level_price": 159.000,
        "nearest_level_pips": 1.2,
        "micro_confirmation": "tokyo_range_low_reclaim",
        "session_range_pips": 8.0,
        "session_mid_price": 159.040,
        "reward_to_mid_pips": 3.0,
        "bb_width": 0.0005,
        "adx": 16.0,
    }
    base.update(overrides)
    return base


def _failed_breakout_trigger(**overrides) -> dict:
    base = {
        "family": "failed_breakout_reversal_overlap_v1",
        "reason": "failed_breakout_recapture:LONDON_NY_SESSION_HIGH",
        "bias": "sell",
        "level_label": "LONDON_NY_SESSION_HIGH",
        "level_price": 159.120,
        "nearest_level_pips": 2.0,
        "micro_confirmation": "failed_breakout_recapture",
        "breakout_side": "up",
        "breakout_excursion_pips": 3.2,
        "hold_bars": 2,
    }
    base.update(overrides)
    return base


def test_critical_level_reaction_trigger_keeps_support_reclaim(monkeypatch) -> None:
    m1 = pd.DataFrame(
        {
            "open": [159.04, 159.03, 159.02, 159.01, 159.02, 159.01, 159.00, 159.01, 159.00, 159.00],
            "high": [159.05, 159.04, 159.03, 159.02, 159.03, 159.02, 159.01, 159.02, 159.03, 159.04],
            "low": [159.02, 159.01, 159.00, 158.99, 159.00, 158.99, 158.98, 158.99, 158.98, 158.98],
            "close": [159.03, 159.02, 159.01, 159.00, 159.01, 159.00, 159.00, 159.01, 159.01, 159.03],
        }
    )
    monkeypatch.setattr(
        autonomous_fillmore,
        "_nearest_structure_pips",
        lambda *args, **kwargs: {
            "underfoot_pips": 1.0,
            "underfoot_price": 159.000,
            "underfoot_label": "WHOLE_YEN:159.00",
            "overhead_pips": None,
            "overhead_price": None,
            "overhead_label": None,
        },
    )

    trig = autonomous_fillmore._critical_level_reaction_trigger(
        159.03,
        {"M1": m1},
        "london/ny",
        max_level_pips=6.0,
        micro_window_bars=3,
        touch_tolerance_pips=0.8,
    )

    assert trig is not None
    assert trig["reason"] == "support_reclaim:WHOLE_YEN:159.00"
    assert trig["bias"] == "buy"


def test_critical_level_reaction_trigger_disables_resistance_reject(monkeypatch) -> None:
    m1 = pd.DataFrame(
        {
            "open": [159.03, 159.04, 159.05, 159.06, 159.07, 159.08, 159.07, 159.08, 159.07, 159.06],
            "high": [159.05, 159.06, 159.08, 159.09, 159.10, 159.11, 159.10, 159.11, 159.10, 159.10],
            "low": [159.02, 159.03, 159.04, 159.05, 159.06, 159.07, 159.06, 159.06, 159.05, 159.04],
            "close": [159.04, 159.05, 159.06, 159.07, 159.08, 159.07, 159.08, 159.07, 159.06, 159.05],
        }
    )
    monkeypatch.setattr(
        autonomous_fillmore,
        "_nearest_structure_pips",
        lambda *args, **kwargs: {
            "underfoot_pips": None,
            "underfoot_price": None,
            "underfoot_label": None,
            "overhead_pips": 1.0,
            "overhead_price": 159.100,
            "overhead_label": "WHOLE_YEN:159.10",
        },
    )

    trig = autonomous_fillmore._critical_level_reaction_trigger(
        159.05,
        {"M1": m1},
        "london/ny",
        max_level_pips=6.0,
        micro_window_bars=3,
        touch_tolerance_pips=0.8,
    )

    assert trig is None


def test_critical_level_reaction_trigger_prunes_disabled_support_labels(monkeypatch) -> None:
    m1 = pd.DataFrame(
        {
            "open": [159.04, 159.03, 159.02, 159.01, 159.02, 159.01, 159.00, 159.01, 159.00, 159.00],
            "high": [159.05, 159.04, 159.03, 159.02, 159.03, 159.02, 159.01, 159.02, 159.03, 159.04],
            "low": [159.02, 159.01, 159.00, 158.99, 159.00, 158.99, 158.98, 158.99, 158.98, 158.98],
            "close": [159.03, 159.02, 159.01, 159.00, 159.01, 159.00, 159.00, 159.01, 159.01, 159.03],
        }
    )
    monkeypatch.setattr(
        autonomous_fillmore,
        "_nearest_structure_pips",
        lambda *args, **kwargs: {
            "underfoot_pips": 1.0,
            "underfoot_price": 159.000,
            "underfoot_label": "TOKYO_SESSION_LOW",
            "overhead_pips": None,
            "overhead_price": None,
            "overhead_label": None,
        },
    )

    trig = autonomous_fillmore._critical_level_reaction_trigger(
        159.03,
        {"M1": m1},
        "tokyo",
        max_level_pips=6.0,
        micro_window_bars=3,
        touch_tolerance_pips=0.8,
    )

    assert trig is None


def test_failed_breakout_reversal_trigger_detects_overlap_recapture() -> None:
    m5 = pd.DataFrame(
        {
            "time": pd.to_datetime(
                [
                    "2026-04-24T11:15:00Z",
                    "2026-04-24T11:20:00Z",
                    "2026-04-24T11:25:00Z",
                    "2026-04-24T11:30:00Z",
                    "2026-04-24T11:35:00Z",
                    "2026-04-24T11:40:00Z",
                    "2026-04-24T11:45:00Z",
                    "2026-04-24T11:50:00Z",
                    "2026-04-24T11:55:00Z",
                    "2026-04-24T12:00:00Z",
                    "2026-04-24T12:05:00Z",
                    "2026-04-24T12:10:00Z",
                    "2026-04-24T12:15:00Z",
                    "2026-04-24T12:20:00Z",
                    "2026-04-24T12:25:00Z",
                    "2026-04-24T12:30:00Z",
                    "2026-04-24T12:35:00Z",
                    "2026-04-24T12:40:00Z",
                    "2026-04-24T12:45:00Z",
                    "2026-04-24T12:50:00Z",
                ],
                utc=True,
            ),
            "open": [158.90, 158.91, 158.92, 158.93, 158.94, 158.95, 158.96, 158.97, 158.98, 159.00, 159.01, 159.02, 159.03, 159.04, 159.05, 159.06, 159.07, 159.08, 159.10, 159.11],
            "high": [158.92, 158.93, 158.94, 158.95, 158.96, 158.97, 158.98, 158.99, 159.00, 159.02, 159.03, 159.04, 159.05, 159.06, 159.07, 159.08, 159.09, 159.10, 159.13, 159.12],
            "low": [158.89, 158.90, 158.91, 158.92, 158.93, 158.94, 158.95, 158.96, 158.97, 158.99, 159.00, 159.01, 159.02, 159.03, 159.04, 159.05, 159.06, 159.07, 159.09, 159.05],
            "close": [158.91, 158.92, 158.93, 158.94, 158.95, 158.96, 158.97, 158.98, 158.99, 159.01, 159.02, 159.03, 159.04, 159.05, 159.06, 159.07, 159.08, 159.09, 159.11, 159.06],
        }
    )

    trig = autonomous_fillmore._failed_breakout_reversal_trigger(
        159.06,
        {"M5": m5},
        "london/ny",
    )

    assert trig is not None
    assert trig["family"] == "failed_breakout_reversal_overlap_v1"
    assert trig["bias"] == "sell"
    assert trig["breakout_side"] == "up"
    assert trig["hold_bars"] == 1


def test_failed_breakout_reversal_trigger_is_overlap_only() -> None:
    m5 = pd.DataFrame(
        {
            "time": pd.to_datetime(["2026-04-24T14:00:00Z"] * 20, utc=True),
            "open": [159.0] * 20,
            "high": [159.1] * 20,
            "low": [158.9] * 20,
            "close": [159.0] * 20,
        }
    )

    trig = autonomous_fillmore._failed_breakout_reversal_trigger(
        159.0,
        {"M5": m5},
        "ny",
    )

    assert trig is None


def test_tokyo_tight_range_mean_reversion_trigger_fires_on_low_reclaim(monkeypatch) -> None:
    times = pd.date_range("2026-01-05 00:00:00+00:00", periods=60, freq="min")
    closes = [159.05] * 50 + [159.03, 159.02, 159.01, 159.00, 158.99, 158.98, 158.99, 158.98, 158.99, 159.00]
    m1 = pd.DataFrame(
        {
            "time": times,
            "open": closes,
            "high": [c + 0.03 for c in closes],
            "low": [c - 0.01 for c in closes],
            "close": closes,
        }
    )
    m5 = pd.DataFrame(
        {
            "open": [159.03] * 30,
            "high": [159.08] * 30,
            "low": [159.00] * 30,
            "close": [159.04] * 30,
        }
    )
    monkeypatch.setattr(autonomous_fillmore, "_atr_pips", lambda *args, **kwargs: 4.0)
    monkeypatch.setattr(autonomous_fillmore, "_adx_value", lambda *args, **kwargs: 16.0)

    trig = autonomous_fillmore._tokyo_tight_range_mean_reversion_trigger(
        159.03,
        {"M1": m1, "M5": m5},
        "tokyo",
        range_window_bars=20,
        min_session_bars=45,
        max_range_pips=18.0,
        range_atr_mult=3.6,
        edge_fraction=0.24,
        touch_tolerance_pips=0.8,
        adx_ceiling=22.0,
        bb_width_max=0.0008,
        min_reward_pips=2.5,
    )

    assert trig is not None
    assert trig["family"] == "tight_range_mean_reversion"
    assert trig["bias"] == "buy"
    assert trig["reason"] == "tokyo_range_reclaim:session_low"


def test_tokyo_tight_range_mean_reversion_trigger_is_tokyo_only(monkeypatch) -> None:
    times = pd.date_range("2026-01-05 00:00:00+00:00", periods=60, freq="min")
    m1 = pd.DataFrame(
        {
            "time": times,
            "open": [159.03] * 60,
            "high": [159.05] * 60,
            "low": [159.01] * 60,
            "close": [159.03] * 60,
        }
    )
    m5 = pd.DataFrame(
        {
            "open": [159.03] * 30,
            "high": [159.05] * 30,
            "low": [159.01] * 30,
            "close": [159.03] * 30,
        }
    )
    monkeypatch.setattr(autonomous_fillmore, "_atr_pips", lambda *args, **kwargs: 4.0)
    monkeypatch.setattr(autonomous_fillmore, "_adx_value", lambda *args, **kwargs: 16.0)

    trig = autonomous_fillmore._tokyo_tight_range_mean_reversion_trigger(
        159.03,
        {"M1": m1, "M5": m5},
        "london",
        range_window_bars=20,
        min_session_bars=45,
        max_range_pips=18.0,
        range_atr_mult=3.6,
        edge_fraction=0.24,
        touch_tolerance_pips=0.8,
        adx_ceiling=22.0,
        bb_width_max=0.0008,
        min_reward_pips=2.5,
    )

    assert trig is None


def test_invoke_suggest_persists_autonomous_suggestion_history(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "ai_suggestions.sqlite"
    fake_chat = ModuleType("api.ai_trading_chat")
    fake_chat.build_trading_context = lambda profile, profile_name: _ctx()
    fake_chat.build_trade_suggestion_news_block = lambda **kwargs: "NEWS BLOCK"
    fake_chat.resolve_ai_suggest_model = lambda configured: "gpt-5.4-mini"
    fake_chat.system_prompt_from_context = lambda ctx, model: "SYSTEM PROMPT"
    fake_chat.autonomous_system_prompt_from_context = lambda ctx, model, **kwargs: "AUTONOMOUS SYSTEM PROMPT"
    monkeypatch.setitem(sys.modules, "api.ai_trading_chat", fake_chat)
    _install_fake_main(monkeypatch, db_path)

    class _FakeOpenAIClient:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            content = json.dumps(_suggestion())
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    fake_openai = ModuleType("openai")
    fake_openai.OpenAI = _FakeOpenAIClient
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    profile = SimpleNamespace(symbol="USDJPY")
    out = autonomous_fillmore._invoke_suggest(
        profile,
        "kumatora2",
        {"model": "gpt-5.4-mini", "aggressiveness": "balanced", "mode": "paper"},
    )
    out0 = out[0]

    assert out0["suggestion_id"]
    assert out0["order_type"] in ("market", "limit")
    history = suggestion_tracker.get_history(db_path, limit=10, offset=0)
    assert history["total"] == 1
    row = history["items"][0]
    assert row["suggestion_id"] == out0["suggestion_id"]
    assert row["profile"] == "kumatora2"
    assert row["market_snapshot"]["macro_bias"]["combined_bias"] == "bearish"


def test_min_confidence_removed_from_default_config() -> None:
    assert "min_confidence" not in autonomous_fillmore.DEFAULT_CONFIG


def test_default_config_has_base_lot_size_and_deviation() -> None:
    assert autonomous_fillmore.DEFAULT_CONFIG["base_lot_size"] == 5.0
    assert autonomous_fillmore.DEFAULT_CONFIG["lot_deviation"] == 4.0


def test_autonomous_exit_calibration_uses_tokyo_tighter_tp1() -> None:
    params = autonomous_fillmore._apply_autonomous_exit_calibration(
        "tp1_be_m5_trail",
        {
            "tp1_pips": 4.0,
            "tp1_close_pct": 80.0,
            "be_plus_pips": 0.5,
            "trail_ema_period": 20,
        },
        {"session": {"active_sessions": ["Tokyo"], "overlap": None}},
    )

    assert params["tp1_pips"] == 5.0
    assert params["tp1_close_pct"] == 50.0
    assert params["trail_ema_period"] == 20


def test_autonomous_exit_calibration_keeps_more_runner_weight_in_london_ny() -> None:
    params = autonomous_fillmore._apply_autonomous_exit_calibration(
        "tp1_be_hwm_trail",
        {
            "tp1_pips": 4.0,
            "tp1_close_pct": 70.0,
            "be_plus_pips": 0.5,
            "hwm_trail_pips": 3.0,
        },
        {"session": {"active_sessions": ["New York"], "overlap": None}},
    )

    assert params["tp1_pips"] == 6.0
    assert params["tp1_close_pct"] == 33.0
    assert params["hwm_trail_pips"] == 3.0


def test_invoke_suggest_stage4_sharpens_confidence_prompt_and_exit_calibration(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "ai_suggestions.sqlite"
    captured: dict[str, object] = {}
    suggestion_tracker.log_reflection(
        db_path,
        profile="kumatora2",
        suggestion_id="sid-auto",
        trade_id="trade-auto",
        model="gpt-5.4-mini",
        what_read_right="Trend alignment was real.",
        what_missed="Ignored resistance overhead.",
        summary_text="BUY loss (-4.0p, $-8.00, exit=hit_stop_loss)",
        autonomous=True,
    )

    fake_chat = ModuleType("api.ai_trading_chat")
    fake_chat.build_trading_context = lambda profile, profile_name: _ctx()
    fake_chat.build_trade_suggestion_news_block = lambda **kwargs: "NEWS BLOCK"
    fake_chat.resolve_ai_suggest_model = lambda configured: "gpt-5.4-mini"
    fake_chat.autonomous_system_prompt_from_context = lambda ctx, model, **kwargs: "AUTONOMOUS SYSTEM PROMPT"
    monkeypatch.setitem(sys.modules, "api.ai_trading_chat", fake_chat)
    _install_fake_main(monkeypatch, db_path)

    class _FakeOpenAIClient:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            captured["messages"] = kwargs["messages"]
            content = json.dumps({
                **_suggestion(exit_strategy="tp1_be_m5_trail"),
                "exit_strategy": "tp1_be_m5_trail",
                "exit_params": {"tp1_pips": 4.0, "tp1_close_pct": 80.0},
            })
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    fake_openai = ModuleType("openai")
    fake_openai.OpenAI = _FakeOpenAIClient
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    out = autonomous_fillmore._invoke_suggest(
        SimpleNamespace(symbol="USDJPY"),
        "kumatora2",
        {"model": "gpt-5.4-mini", "aggressiveness": "balanced", "mode": "paper", "min_confidence": "high"},
    )
    out0 = out[0]

    user_prompt = str((captured.get("messages") or [{}, {}])[1]["content"])
    assert "LOT SIZING" in user_prompt
    assert "CONVICTION SHOULD REFLECT SELECTIVITY" in user_prompt
    assert "0 lots: use this freely" in user_prompt
    assert '"quality": "A" | "B" | "C"' in user_prompt
    assert "London/NY trend profile" in user_prompt
    system_prompt = str((captured.get("messages") or [{}, {}])[0]["content"])
    assert "SELF-REFLECTION MEMORY" in system_prompt
    assert "Trend alignment was real." in system_prompt
    assert out0["exit_params"]["tp1_pips"] == 6.0
    assert out0["exit_params"]["tp1_close_pct"] == 33.0


def test_autonomous_system_prompt_aligns_with_near_touch_execution() -> None:
    prompt = ai_trading_chat.autonomous_system_prompt_from_context(
        _ctx(),
        "gpt-5.4-mini",
        autonomous_config={
            "multi_trade_enabled": True,
            "max_suggestions_per_call": 2,
            "correlation_distance_pips": 10,
        },
        risk_regime={"label": "defensive_soft"},
    )

    assert "return UP TO 2 trade objects" in prompt
    assert "within ~10 pips of your proposed entry" in prompt
    assert "~$6.66/pip/lot at 150.123" in prompt
    assert "RISK REGIME: DEFENSIVE_SOFT" in prompt
    assert "clamp autonomous limits into a near-market band" in prompt
    assert "compression-breakout setups also favor market execution" in prompt
    assert "You are allowed to pass freely" in prompt


def test_fit_aux_memory_blocks_keeps_required_and_drops_low_priority_when_budget_tight() -> None:
    packed = autonomous_fillmore._fit_aux_memory_blocks(
        [
            ("learning", "LEARNING " * 120, True),
            ("performance", "PERFORMANCE " * 80, True),
            ("reflections", "REFLECTIONS " * 140, False),
            ("today", "TODAY " * 140, False),
        ],
        budget_words=260,
    )

    assert "LEARNING" in packed
    assert "PERFORMANCE" in packed
    assert "Some lower-priority history blocks were omitted" in packed
    assert "reflections" in packed or "today" in packed


def test_invoke_suggest_snaps_limit_price_into_near_market_band(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "ai_suggestions.sqlite"
    captured: dict[str, object] = {}

    fake_chat = ModuleType("api.ai_trading_chat")
    fake_chat.build_trading_context = lambda profile, profile_name: {
        **_ctx(),
        "spot_price": {"mid": 159.123, "bid": 159.121, "ask": 159.125, "spread_pips": 0.4},
    }
    fake_chat.build_trade_suggestion_news_block = lambda **kwargs: "NEWS BLOCK"
    fake_chat.resolve_ai_suggest_model = lambda configured: "gpt-5.4-mini"
    fake_chat.autonomous_system_prompt_from_context = lambda ctx, model, **kwargs: "AUTONOMOUS SYSTEM PROMPT"
    monkeypatch.setitem(sys.modules, "api.ai_trading_chat", fake_chat)
    _install_fake_main(monkeypatch, db_path)

    class _FakeOpenAIClient:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            captured["messages"] = kwargs["messages"]
            content = json.dumps({
                **_suggestion(),
                "order_type": "limit",
                "price": 159.050,
            })
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    fake_openai = ModuleType("openai")
    fake_openai.OpenAI = _FakeOpenAIClient
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    out = autonomous_fillmore._invoke_suggest(
        SimpleNamespace(symbol="USDJPY", pip_size=0.01),
        "kumatora2",
        {"model": "gpt-5.4-mini", "aggressiveness": "balanced", "mode": "paper"},
    )
    out0 = out[0]

    user_prompt = str((captured.get("messages") or [{}, {}])[1]["content"])
    assert '"order_type": "market" | "limit"' in user_prompt
    assert "near-touch passive entries" in user_prompt
    assert out0["order_type"] == "limit"
    assert out0["requested_price"] == 159.050
    assert out0["price"] == 159.116
    assert out0["snap_distance_pips"] == 6.6
    assert out0["time_in_force"] == "GTD"
    assert out0["gtd_time_utc"] is not None

    history = suggestion_tracker.get_history(db_path, limit=10, offset=0)
    row = history["items"][0]
    assert row["requested_price"] == 159.050
    assert row["limit_price"] == 159.116
    assert row["snap_distance_pips"] == 6.6


def test_invoke_suggest_sets_quality_and_strips_confidence(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "ai_suggestions.sqlite"

    fake_chat = ModuleType("api.ai_trading_chat")
    fake_chat.build_trading_context = lambda profile, profile_name: _ctx()
    fake_chat.build_trade_suggestion_news_block = lambda **kwargs: ""
    fake_chat.resolve_ai_suggest_model = lambda configured: "gpt-5.4-mini"
    fake_chat.autonomous_system_prompt_from_context = lambda ctx, model, **kwargs: "SYS"
    monkeypatch.setitem(sys.modules, "api.ai_trading_chat", fake_chat)
    _install_fake_main(monkeypatch, db_path)

    class _Client:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        def _create(self, **kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content=json.dumps({**_suggestion(), "quality": "B", "lots": 5}),
            ))])

    fake_openai = ModuleType("openai")
    fake_openai.OpenAI = _Client
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "k")

    out = autonomous_fillmore._invoke_suggest(
        SimpleNamespace(symbol="USDJPY"), "p1",
        {"model": "gpt-5.4-mini", "aggressiveness": "balanced", "mode": "paper"},
    )
    assert out[0]["quality"] == "B"
    assert "confidence" not in out[0]


def test_invoke_suggest_zero_lots_maps_to_quality_c(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "ai_suggestions.sqlite"

    fake_chat = ModuleType("api.ai_trading_chat")
    fake_chat.build_trading_context = lambda profile, profile_name: _ctx()
    fake_chat.build_trade_suggestion_news_block = lambda **kwargs: ""
    fake_chat.resolve_ai_suggest_model = lambda configured: "gpt-5.4-mini"
    fake_chat.autonomous_system_prompt_from_context = lambda ctx, model, **kwargs: "SYS"
    monkeypatch.setitem(sys.modules, "api.ai_trading_chat", fake_chat)
    _install_fake_main(monkeypatch, db_path)

    class _Client:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        def _create(self, **kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content=json.dumps({**_suggestion(), "quality": "C", "lots": 0}),
            ))])

    fake_openai = ModuleType("openai")
    fake_openai.OpenAI = _Client
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "k")

    out = autonomous_fillmore._invoke_suggest(
        SimpleNamespace(symbol="USDJPY"), "p1",
        {"model": "gpt-5.4-mini", "aggressiveness": "balanced", "mode": "paper"},
    )
    assert out[0]["lots"] == 0
    assert out[0]["quality"] == "C"
    assert "confidence" not in out[0]


def test_invoke_suggest_defaults_market_order_type_when_omitted(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "ai_suggestions.sqlite"

    fake_chat = ModuleType("api.ai_trading_chat")
    fake_chat.build_trading_context = lambda profile, profile_name: _ctx()
    fake_chat.build_trade_suggestion_news_block = lambda **kwargs: "NEWS BLOCK"
    fake_chat.resolve_ai_suggest_model = lambda configured: "gpt-5.4-mini"
    fake_chat.autonomous_system_prompt_from_context = lambda ctx, model, **kwargs: "AUTONOMOUS SYSTEM PROMPT"
    monkeypatch.setitem(sys.modules, "api.ai_trading_chat", fake_chat)
    _install_fake_main(monkeypatch, db_path)

    class _FakeOpenAIClient:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            content = json.dumps({**_suggestion()})
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    fake_openai = ModuleType("openai")
    fake_openai.OpenAI = _FakeOpenAIClient
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    out = autonomous_fillmore._invoke_suggest(
        SimpleNamespace(symbol="USDJPY"),
        "kumatora2",
        {"model": "gpt-5.4-mini", "aggressiveness": "balanced", "mode": "paper"},
    )
    assert out[0]["order_type"] == "market"


def test_market_order_with_no_exit_strategy_still_links_trade_and_fill(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "ai_suggestions.sqlite"
    _install_fake_main(monkeypatch, db_path)

    fake_broker = ModuleType("adapters.broker")
    fake_broker.get_adapter = lambda profile: _FakeMarketAdapter()
    monkeypatch.setitem(sys.modules, "adapters.broker", fake_broker)

    profile = SimpleNamespace(
        profile_name="kumatora2",
        symbol="USDJPY",
        broker_type="oanda",
        active_preset_name=None,
    )
    store = _DummyStore()
    state_path = tmp_path / "kumatora2" / "runtime_state.json"

    sid = suggestion_tracker.log_generated(
        db_path,
        profile="kumatora2",
        model="gpt-5.4",
        suggestion=_suggestion(),
        ctx=_ctx(),
    )
    suggestion = _suggestion()
    suggestion["suggestion_id"] = sid

    out = autonomous_fillmore._place_from_suggestion(
        profile,
        "kumatora2",
        state_path,
        suggestion,
        "market",
        store,
    )

    assert out["status"] == "filled"
    assert len(store.inserted) == 1
    row = store.inserted[0]
    assert row["trade_id"].startswith("ai_autonomous:555001:")
    assert row["entry_type"] == "ai_autonomous"
    assert row["managed_trail_mode"] == "none"

    tracked = suggestion_tracker.get_by_order_id(db_path, "555001")
    assert tracked is not None
    assert tracked["trade_id"] == row["trade_id"]
    assert tracked["placed_order"]["entry_type"] == "ai_autonomous"
    assert tracked["placed_order"]["order_type"] == "market"
    assert tracked["placed_order"]["exit_strategy"] == "none"


def test_limit_order_with_no_exit_strategy_still_registers_pending_watch(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "ai_suggestions.sqlite"
    _install_fake_main(monkeypatch, db_path)

    fake_broker = ModuleType("adapters.broker")
    fake_broker.get_adapter = lambda profile: _FakeLimitAdapter()
    monkeypatch.setitem(sys.modules, "adapters.broker", fake_broker)

    profile = SimpleNamespace(profile_name="kumatora2", symbol="USDJPY")
    store = _DummyStore()
    state_path = tmp_path / "kumatora2" / "runtime_state.json"

    sid = suggestion_tracker.log_generated(
        db_path,
        profile="kumatora2",
        model="gpt-5.4",
        suggestion=_suggestion(),
        ctx=_ctx(),
    )
    suggestion = _suggestion()
    suggestion["suggestion_id"] = sid

    out = autonomous_fillmore._place_from_suggestion(
        profile,
        "kumatora2",
        state_path,
        suggestion,
        "limit",
        store,
    )

    assert out["status"] == "placed"
    assert store.inserted == []

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert len(state["managed_pending_orders"]) == 1
    pending = state["managed_pending_orders"][0]
    assert pending["order_id"] == 888002
    assert pending["exit_strategy"] == "none"
    assert pending["trail_mode"] == "none"
    assert pending["exit_params"] == {}

    tracked = suggestion_tracker.get_by_order_id(db_path, "888002")
    assert tracked is not None
    assert tracked["placed_order"]["order_type"] == "limit"
    assert tracked["placed_order"]["exit_strategy"] == "none"


def test_aggressive_gate_blocks_when_m3_and_m1_disagree(monkeypatch) -> None:
    _force_allowed_session(monkeypatch)
    monkeypatch.setattr(autonomous_fillmore, "_m3_trend", lambda data_by_tf: "bull")
    monkeypatch.setattr(autonomous_fillmore, "_m1_stack", lambda data_by_tf: "bear")
    monkeypatch.setattr(autonomous_fillmore, "_critical_level_reaction_trigger", lambda *args, **kwargs: None)
    monkeypatch.setattr(autonomous_fillmore, "_trend_expansion_trigger", lambda *args, **kwargs: None)

    decision = autonomous_fillmore.evaluate_gate(
        _gate_cfg("aggressive"),
        {"daily_pnl_usd": 0.0, "llm_spend_today_usd": 0.0},
        autonomous_fillmore.GateInputs(
            spread_pips=0.8,
            tick_mid=159.25,
            open_ai_trade_count=0,
            data_by_tf={},
            ntz_active=False,
        ),
    )

    assert decision.result == "block"
    assert decision.layer == "signal"
    assert decision.reason == "no_hybrid_trigger"
    assert decision.extras.get("m3") == "bull"
    assert decision.extras.get("m1") == "bear"


def test_aggressive_gate_blocks_without_hybrid_trigger(monkeypatch) -> None:
    _force_allowed_session(monkeypatch)
    monkeypatch.setattr(autonomous_fillmore, "_m3_trend", lambda data_by_tf: "bull")
    monkeypatch.setattr(autonomous_fillmore, "_m1_stack", lambda data_by_tf: None)
    monkeypatch.setattr(autonomous_fillmore, "_critical_level_reaction_trigger", lambda *args, **kwargs: None)
    monkeypatch.setattr(autonomous_fillmore, "_trend_expansion_trigger", lambda *args, **kwargs: None)

    decision = autonomous_fillmore.evaluate_gate(
        _gate_cfg("aggressive"),
        {"daily_pnl_usd": 0.0, "llm_spend_today_usd": 0.0},
        autonomous_fillmore.GateInputs(
            spread_pips=0.8,
            tick_mid=159.25,
            open_ai_trade_count=0,
            data_by_tf={},
            ntz_active=False,
        ),
    )

    assert decision.result == "block"
    assert decision.layer == "signal"
    assert decision.reason == "no_hybrid_trigger"


def test_very_aggressive_gate_still_requires_some_trend_signal(monkeypatch) -> None:
    _force_allowed_session(monkeypatch)
    monkeypatch.setattr(autonomous_fillmore, "_m3_trend", lambda data_by_tf: None)
    monkeypatch.setattr(autonomous_fillmore, "_m1_stack", lambda data_by_tf: None)
    monkeypatch.setattr(autonomous_fillmore, "_critical_level_reaction_trigger", lambda *args, **kwargs: None)
    monkeypatch.setattr(autonomous_fillmore, "_trend_expansion_trigger", lambda *args, **kwargs: None)

    decision = autonomous_fillmore.evaluate_gate(
        _gate_cfg("very_aggressive"),
        {"daily_pnl_usd": 0.0, "llm_spend_today_usd": 0.0},
        autonomous_fillmore.GateInputs(
            spread_pips=0.8,
            tick_mid=159.25,
            open_ai_trade_count=0,
            data_by_tf={},
            ntz_active=False,
        ),
    )

    assert decision.result == "block"
    assert decision.layer == "signal"
    assert decision.reason == "no_hybrid_trigger:no_trend_signal"


def test_loss_streak_throttle_does_not_rearm_forever_for_same_streak() -> None:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    rt = {
        "loss_streak_last_throttled_day_utc": today,
        "loss_streak_last_throttled_value": 4,
        "throttle_until_utc": "2000-01-01T00:00:00+00:00",
        "throttle_reason": "loss_streak=4",
    }

    armed = autonomous_fillmore._maybe_arm_loss_streak_throttle(
        rt,
        dict(autonomous_fillmore.DEFAULT_CONFIG),
        4,
    )

    assert armed is False
    assert rt["throttle_until_utc"] == "2000-01-01T00:00:00+00:00"
    assert rt["throttle_reason"] == "loss_streak=4"


def test_set_config_turning_autonomous_off_clears_active_throttle(tmp_path: Path) -> None:
    state_path = tmp_path / "kumatora2" / "runtime_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "autonomous_fillmore": {
                    "config": {"enabled": True, "mode": "paper"},
                    "runtime": {
                        "throttle_until_utc": "2099-01-01T00:00:00+00:00",
                        "throttle_reason": "loss_streak=4",
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    autonomous_fillmore.set_config(state_path, {"enabled": False, "mode": "off"})

    state = autonomous_fillmore._load_state(state_path)
    rt = (state.get("autonomous_fillmore") or {}).get("runtime") or {}
    assert rt.get("throttle_until_utc") is None
    assert rt.get("throttle_reason") is None


def test_set_config_turning_autonomous_off_clears_stale_llm_error_alert_state(tmp_path: Path) -> None:
    state_path = tmp_path / "kumatora2" / "runtime_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "autonomous_fillmore": {
                    "config": {"enabled": True, "mode": "paper"},
                    "runtime": {
                        "consecutive_llm_errors": 8,
                        "last_error_msg": "Error code: 429 - quota exceeded",
                        "last_error_utc": "2026-04-21T12:34:56+00:00",
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    autonomous_fillmore.set_config(state_path, {"enabled": False, "mode": "off"})

    state = autonomous_fillmore._load_state(state_path)
    rt = (state.get("autonomous_fillmore") or {}).get("runtime") or {}
    assert rt.get("consecutive_llm_errors") == 0
    assert rt.get("last_error_msg") is None
    assert rt.get("last_error_utc") is None


def test_get_config_sanitizes_dangerous_saved_autonomous_settings(tmp_path: Path) -> None:
    state_path = tmp_path / "newera8" / "runtime_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "autonomous_fillmore": {
                    "config": {
                        "enabled": True,
                        "mode": "paper",
                        "aggressiveness": "very_aggressive",
                        "trading_hours": {"tokyo": True, "london": True, "ny": True},
                        "max_daily_loss_usd": 5000,
                        "max_lots_per_trade": 25.04,
                        "max_consecutive_errors": 10,
                        "limit_gtd_minutes": 45,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    cfg = autonomous_fillmore.get_config(state_path)

    assert cfg["aggressiveness"] == "balanced"
    assert cfg["trading_hours"]["tokyo"] is True
    assert cfg["trading_hours"]["london"] is True
    assert cfg["trading_hours"]["ny"] is True
    assert cfg["max_daily_loss_usd"] == 50.0
    assert cfg["max_lots_per_trade"] == 15.0
    assert cfg["max_consecutive_errors"] == 5
    assert cfg["limit_gtd_minutes"] == 15
    assert cfg["max_open_ai_trades"] == 6


def test_set_config_sanitizes_patch_into_safe_autonomous_envelope(tmp_path: Path) -> None:
    state_path = tmp_path / "newera8" / "runtime_state.json"

    cfg = autonomous_fillmore.set_config(
        state_path,
        {
            "enabled": True,
            "mode": "paper",
            "aggressiveness": "very_aggressive",
            "trading_hours": {"tokyo": True, "london": True, "ny": True},
            "max_daily_loss_usd": 5000,
            "max_lots_per_trade": 25.04,
            "max_consecutive_errors": 10,
            "limit_gtd_minutes": 45,
        },
    )

    assert cfg["aggressiveness"] == "balanced"
    assert cfg["trading_hours"]["tokyo"] is True
    assert cfg["max_daily_loss_usd"] == 50.0
    assert cfg["max_lots_per_trade"] == 15.0
    assert cfg["max_consecutive_errors"] == 5
    assert cfg["limit_gtd_minutes"] == 15
    assert cfg["max_open_ai_trades"] == 6


def test_get_config_lifts_stale_lower_open_trade_cap_to_shared_default(tmp_path: Path) -> None:
    state_path = tmp_path / "newera8" / "runtime_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "autonomous_fillmore": {
                    "config": {
                        "enabled": True,
                        "mode": "paper",
                        "aggressiveness": "balanced",
                        "max_open_ai_trades": 2,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    cfg = autonomous_fillmore.get_config(state_path)

    assert cfg["max_open_ai_trades"] == 6


def test_extract_json_object_prefers_fenced_block_and_keeps_analysis(monkeypatch=None) -> None:
    text = (
        "ANALYSIS:\n"
        "- M3 trend bull, M1 stack bull, JPY crosses confirm.\n"
        "- Anchoring on PDL at 159.10.\n"
        "\n"
        "DECISION:\n"
        "```json\n"
        '{"side": "buy", "price": 159.10, "lots": 2.0}\n'
        "```\n"
    )
    json_str, analysis = autonomous_fillmore._extract_json_object(text)
    parsed = json.loads(json_str)
    assert parsed["side"] == "buy"
    assert parsed["lots"] == 2.0
    assert analysis is not None and "PDL at 159.10" in analysis


def test_extract_json_object_falls_back_to_brace_balance() -> None:
    text = 'analysis text without fence\n{"side":"sell","price":159.50,"lots":1.5}\n'
    json_str, analysis = autonomous_fillmore._extract_json_object(text)
    parsed = json.loads(json_str)
    assert parsed["side"] == "sell"
    assert parsed["price"] == 159.50
    assert analysis == "analysis text without fence"


def test_proximity_gate_blocks_when_price_far_from_structure(monkeypatch) -> None:
    _force_allowed_session(monkeypatch)
    monkeypatch.setattr(autonomous_fillmore, "_m3_trend", lambda data_by_tf: "bull")
    monkeypatch.setattr(autonomous_fillmore, "_m1_stack", lambda data_by_tf: "bull")
    monkeypatch.setattr(autonomous_fillmore, "_critical_level_reaction_trigger", lambda *args, **kwargs: None)
    monkeypatch.setattr(autonomous_fillmore, "_trend_expansion_trigger", lambda *args, **kwargs: None)

    decision = autonomous_fillmore.evaluate_gate(
        _gate_cfg("balanced"),
        {"daily_pnl_usd": 0.0, "llm_spend_today_usd": 0.0},
        autonomous_fillmore.GateInputs(
            spread_pips=0.8,
            tick_mid=159.23,  # 23p above 159.00, 27p below 159.50 -> nearest round 23p > 8p threshold
            open_ai_trade_count=0,
            data_by_tf={},  # no daily candles -> only round levels considered
            ntz_active=False,
        ),
    )

    assert decision.result == "block"
    assert decision.layer == "signal"
    assert decision.reason == "no_hybrid_trigger"
    assert decision.extras.get("nearest_level_pips") == 23.0


def test_proximity_gate_passes_when_price_near_round_level(monkeypatch) -> None:
    _force_allowed_session(monkeypatch)
    monkeypatch.setattr(autonomous_fillmore, "_m3_trend", lambda data_by_tf: "bull")
    monkeypatch.setattr(autonomous_fillmore, "_m1_stack", lambda data_by_tf: "bull")
    monkeypatch.setattr(autonomous_fillmore, "_critical_level_reaction_trigger", lambda *args, **kwargs: _critical_trigger())
    monkeypatch.setattr(autonomous_fillmore, "_trend_expansion_trigger", lambda *args, **kwargs: None)

    decision = autonomous_fillmore.evaluate_gate(
        _gate_cfg("balanced"),
        {"daily_pnl_usd": 0.0, "llm_spend_today_usd": 0.0},
        autonomous_fillmore.GateInputs(
            spread_pips=0.8,
            tick_mid=159.03,  # 3p above 159.00 round level, well within 8p threshold
            open_ai_trade_count=0,
            data_by_tf={},
            ntz_active=False,
        ),
    )

    assert decision.result == "pass"
    assert decision.extras.get("nearest_level_pips") == 3.0


def test_gate_does_not_refire_same_critical_setup_while_still_active(monkeypatch) -> None:
    _force_allowed_session(monkeypatch)
    monkeypatch.setattr(autonomous_fillmore, "_m3_trend", lambda data_by_tf: "bull")
    monkeypatch.setattr(autonomous_fillmore, "_m1_stack", lambda data_by_tf: "bull")
    monkeypatch.setattr(autonomous_fillmore, "_critical_level_reaction_trigger", lambda *args, **kwargs: _critical_trigger())
    monkeypatch.setattr(autonomous_fillmore, "_compression_breakout_trigger", lambda *args, **kwargs: None)
    monkeypatch.setattr(autonomous_fillmore, "_trend_expansion_trigger", lambda *args, **kwargs: None)

    rt = {"daily_pnl_usd": 0.0, "llm_spend_today_usd": 0.0}
    inputs = autonomous_fillmore.GateInputs(
        spread_pips=0.8,
        tick_mid=159.03,
        open_ai_trade_count=0,
        data_by_tf={},
        ntz_active=False,
    )

    first = autonomous_fillmore.evaluate_gate(_gate_cfg("balanced"), rt, inputs)
    second = autonomous_fillmore.evaluate_gate(_gate_cfg("balanced"), rt, inputs)

    assert first.result == "pass"
    assert second.result == "block"
    assert second.reason == "trigger_still_active"


def test_gate_blocks_same_critical_setup_after_reset_during_setup_cooldown(monkeypatch) -> None:
    _force_allowed_session(monkeypatch)
    monkeypatch.setattr(autonomous_fillmore, "_m3_trend", lambda data_by_tf: "bull")
    monkeypatch.setattr(autonomous_fillmore, "_m1_stack", lambda data_by_tf: "bull")
    state = {"critical": True}

    def _critical(*args, **kwargs):
        return _critical_trigger() if state["critical"] else None

    monkeypatch.setattr(autonomous_fillmore, "_critical_level_reaction_trigger", _critical)
    monkeypatch.setattr(autonomous_fillmore, "_compression_breakout_trigger", lambda *args, **kwargs: None)
    monkeypatch.setattr(autonomous_fillmore, "_trend_expansion_trigger", lambda *args, **kwargs: None)

    rt = {"daily_pnl_usd": 0.0, "llm_spend_today_usd": 0.0}
    inputs = autonomous_fillmore.GateInputs(
        spread_pips=0.8,
        tick_mid=159.03,
        open_ai_trade_count=0,
        data_by_tf={},
        ntz_active=False,
    )
    now = datetime(2026, 4, 20, 12, 0, tzinfo=timezone.utc)

    first = autonomous_fillmore.evaluate_gate(_gate_cfg("balanced"), rt, inputs, now_utc=now)
    state["critical"] = False
    reset = autonomous_fillmore.evaluate_gate(_gate_cfg("balanced"), rt, inputs, now_utc=now + timedelta(minutes=1))
    state["critical"] = True
    retry = autonomous_fillmore.evaluate_gate(_gate_cfg("balanced"), rt, inputs, now_utc=now + timedelta(minutes=2))

    assert first.result == "pass"
    assert reset.result == "block"
    assert retry.result == "block"
    assert retry.reason == "trigger_setup_cooldown"


def test_gate_can_fire_different_family_while_previous_family_remains_active(monkeypatch) -> None:
    monkeypatch.setattr(autonomous_fillmore, "_session_flag_now", lambda trading_hours: (True, "london/ny"))
    monkeypatch.setattr(autonomous_fillmore, "_m3_trend", lambda data_by_tf: "bull")
    monkeypatch.setattr(autonomous_fillmore, "_m1_stack", lambda data_by_tf: "bull")
    phase = {"step": 1}

    def _critical(*args, **kwargs):
        return _critical_trigger()

    def _trend(*args, **kwargs):
        if phase["step"] >= 2:
            return _trend_trigger()
        return None

    monkeypatch.setattr(autonomous_fillmore, "_critical_level_reaction_trigger", _critical)
    monkeypatch.setattr(autonomous_fillmore, "_compression_breakout_trigger", lambda *args, **kwargs: None)
    monkeypatch.setattr(autonomous_fillmore, "_trend_expansion_trigger", _trend)

    rt = {"daily_pnl_usd": 0.0, "llm_spend_today_usd": 0.0}
    inputs = autonomous_fillmore.GateInputs(
        spread_pips=0.8,
        tick_mid=159.03,
        open_ai_trade_count=0,
        data_by_tf={},
        ntz_active=False,
    )

    first = autonomous_fillmore.evaluate_gate(_gate_cfg("balanced"), rt, inputs)
    phase["step"] = 2
    second = autonomous_fillmore.evaluate_gate(_gate_cfg("balanced"), rt, inputs)

    assert first.result == "pass"
    assert first.extras.get("trigger_family") == "critical_level_reaction"
    assert second.result == "pass"
    assert second.extras.get("trigger_family") == "trend_expansion"


def test_trend_expansion_is_restricted_to_london_ny_overlap(monkeypatch) -> None:
    monkeypatch.setattr(autonomous_fillmore, "_session_flag_now", lambda trading_hours: (True, "ny"))
    monkeypatch.setattr(autonomous_fillmore, "_m3_trend", lambda data_by_tf: "bull")
    monkeypatch.setattr(autonomous_fillmore, "_m1_stack", lambda data_by_tf: "bull")
    monkeypatch.setattr(autonomous_fillmore, "_critical_level_reaction_trigger", lambda *args, **kwargs: None)
    monkeypatch.setattr(autonomous_fillmore, "_compression_breakout_trigger", lambda *args, **kwargs: None)
    monkeypatch.setattr(autonomous_fillmore, "_trend_expansion_trigger", lambda *args, **kwargs: _trend_trigger())

    decision = autonomous_fillmore.evaluate_gate(
        _gate_cfg("balanced"),
        {"daily_pnl_usd": 0.0, "llm_spend_today_usd": 0.0},
        autonomous_fillmore.GateInputs(
            spread_pips=0.8,
            tick_mid=159.25,
            open_ai_trade_count=0,
            data_by_tf={},
            ntz_active=False,
        ),
    )

    assert decision.result == "block"
    assert decision.reason == "no_hybrid_trigger"
    assert decision.extras.get("trend_session_veto") == "ny"


def test_trend_expansion_can_still_fire_in_london_ny_overlap(monkeypatch) -> None:
    monkeypatch.setattr(autonomous_fillmore, "_session_flag_now", lambda trading_hours: (True, "london/ny"))
    monkeypatch.setattr(autonomous_fillmore, "_m3_trend", lambda data_by_tf: "bull")
    monkeypatch.setattr(autonomous_fillmore, "_m1_stack", lambda data_by_tf: "bull")
    monkeypatch.setattr(autonomous_fillmore, "_critical_level_reaction_trigger", lambda *args, **kwargs: None)
    monkeypatch.setattr(autonomous_fillmore, "_compression_breakout_trigger", lambda *args, **kwargs: None)
    monkeypatch.setattr(autonomous_fillmore, "_trend_expansion_trigger", lambda *args, **kwargs: _trend_trigger())

    decision = autonomous_fillmore.evaluate_gate(
        _gate_cfg("balanced"),
        {"daily_pnl_usd": 0.0, "llm_spend_today_usd": 0.0},
        autonomous_fillmore.GateInputs(
            spread_pips=0.8,
            tick_mid=159.25,
            open_ai_trade_count=0,
            data_by_tf={},
            ntz_active=False,
        ),
    )

    assert decision.result == "pass"
    assert decision.extras.get("trigger_family") == "trend_expansion"


def test_critical_level_setup_key_buckets_nearby_levels_together(monkeypatch) -> None:
    _force_allowed_session(monkeypatch)
    monkeypatch.setattr(autonomous_fillmore, "_m3_trend", lambda data_by_tf: "bull")
    monkeypatch.setattr(autonomous_fillmore, "_m1_stack", lambda data_by_tf: "bull")
    state = {"level_price": 159.03}

    def _critical(*args, **kwargs):
        return _critical_trigger(level_label="HALF_YEN:159.00", level_price=state["level_price"])

    monkeypatch.setattr(autonomous_fillmore, "_critical_level_reaction_trigger", _critical)
    monkeypatch.setattr(autonomous_fillmore, "_compression_breakout_trigger", lambda *args, **kwargs: None)
    monkeypatch.setattr(autonomous_fillmore, "_trend_expansion_trigger", lambda *args, **kwargs: None)

    rt = {"daily_pnl_usd": 0.0, "llm_spend_today_usd": 0.0}
    inputs = autonomous_fillmore.GateInputs(
        spread_pips=0.8,
        tick_mid=159.03,
        open_ai_trade_count=0,
        data_by_tf={},
        ntz_active=False,
    )
    now = datetime(2026, 4, 20, 12, 0, tzinfo=timezone.utc)

    first = autonomous_fillmore.evaluate_gate(_gate_cfg("balanced"), rt, inputs, now_utc=now)
    monkeypatch.setattr(autonomous_fillmore, "_critical_level_reaction_trigger", lambda *args, **kwargs: None)
    _ = autonomous_fillmore.evaluate_gate(_gate_cfg("balanced"), rt, inputs, now_utc=now + timedelta(minutes=1))
    monkeypatch.setattr(
        autonomous_fillmore,
        "_critical_level_reaction_trigger",
        lambda *args, **kwargs: _critical_trigger(level_label="WHOLE_YEN:159.00", level_price=158.98),
    )
    retry = autonomous_fillmore.evaluate_gate(_gate_cfg("balanced"), rt, inputs, now_utc=now + timedelta(minutes=2))

    assert first.result == "pass"
    assert retry.result == "block"
    assert retry.reason == "trigger_setup_cooldown"


def test_clear_throttle_resets_runtime_cooldown_fields(tmp_path: Path) -> None:
    state_path = tmp_path / "kumatora2" / "runtime_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "autonomous_fillmore": {
                    "config": {"enabled": True, "mode": "paper"},
                    "runtime": {
                        "throttle_until_utc": "2099-01-01T00:00:00+00:00",
                        "throttle_reason": "loss_streak=4",
                        "consecutive_no_trade_replies": 5,
                        "consecutive_errors": 2,
                        "loss_streak_last_throttled_day_utc": "2099-01-01",
                        "loss_streak_last_throttled_value": 4,
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    stats = autonomous_fillmore.clear_throttle(state_path)
    rt = (autonomous_fillmore._load_state(state_path).get("autonomous_fillmore") or {}).get("runtime") or {}

    assert stats["throttle"]["active"] is False
    assert rt.get("throttle_until_utc") is None
    assert rt.get("throttle_reason") is None
    assert rt.get("consecutive_no_trade_replies") == 0
    assert rt.get("consecutive_errors") == 0
    assert rt.get("loss_streak_last_throttled_day_utc") is None
    assert rt.get("loss_streak_last_throttled_value") == 0


def test_no_trade_streak_resets_after_throttle_cooldown_expires() -> None:
    rt = {
        "consecutive_no_trade_replies": 47,
        "throttle_until_utc": (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat(),
        "throttle_reason": "no_trade_streak=47",
    }
    autonomous_fillmore._clear_expired_no_trade_throttle(rt)
    assert rt["consecutive_no_trade_replies"] == 0
    assert rt["throttle_until_utc"] is None
    assert rt["throttle_reason"] is None


def test_no_trade_streak_not_reset_while_throttle_still_active() -> None:
    rt = {
        "consecutive_no_trade_replies": 10,
        "throttle_until_utc": (datetime.now(timezone.utc) + timedelta(seconds=300)).isoformat(),
        "throttle_reason": "no_trade_streak=10",
    }
    autonomous_fillmore._clear_expired_no_trade_throttle(rt)
    assert rt["consecutive_no_trade_replies"] == 10
    assert rt["throttle_until_utc"] is not None


def test_no_trade_streak_reset_does_not_touch_loss_streak_throttle() -> None:
    rt = {
        "consecutive_no_trade_replies": 20,
        "throttle_until_utc": (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat(),
        "throttle_reason": "loss_streak=4",
    }
    autonomous_fillmore._clear_expired_no_trade_throttle(rt)
    assert rt["consecutive_no_trade_replies"] == 20
    assert rt["throttle_reason"] == "loss_streak=4"


# -----------------------------------------------------------------------------
# Stage 3: dedupe gate + correlation veto + today-block prompt helper
# -----------------------------------------------------------------------------


def _seed_placed_suggestion(
    db_path: Path,
    *,
    side: str = "buy",
    price: float = 159.10,
    fill_price: float | None = None,
    closed: bool = False,
    minutes_ago: int = 5,
    autonomous: bool = True,
) -> str:
    """Seed a placed (and optionally filled/closed) suggestion row directly."""
    suggestion_tracker.init_db(db_path)
    sid = suggestion_tracker.log_generated(
        db_path,
        profile="kumatora2",
        model="gpt-5.4-mini",
        suggestion={
            "side": side,
            "price": price,
            "sl": price - (0.10 if side == "buy" else -0.10),
            "tp": price + (0.20 if side == "buy" else -0.20),
            "lots": 2.0,
            "time_in_force": "GTC",
            "gtd_time_utc": None,
            "quality": "A",
            "rationale": "seed",
            "exit_strategy": "none",
            "exit_params": {},
        },
        ctx=_ctx(),
    )
    suggestion_tracker.log_action(
        db_path,
        suggestion_id=sid,
        action="placed",
        edited_fields=None,
        placed_order={"order_type": "limit", "side": side, "price": price, "lots": 2.0, "autonomous": autonomous},
        oanda_order_id="987001",
    )
    if fill_price is not None:
        suggestion_tracker.mark_filled(db_path, oanda_order_id="987001", fill_price=fill_price, trade_id="t-1")
    if closed:
        suggestion_tracker.mark_closed_by_suggestion_id(
            db_path, suggestion_id=sid, exit_price=price, pnl=-15.0, pips=-2.4,
        )
    # Backdate created_utc so dedupe-window math sees this as `minutes_ago` old.
    import sqlite3
    when = (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("UPDATE ai_suggestions SET created_utc=?, filled_at=? WHERE suggestion_id=?", (when, when, sid))
        conn.commit()
    return sid


def test_dedupe_gate_blocks_when_recent_placement_in_same_bucket(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "ai_suggestions.sqlite"
    # Seed a placed sell at 159.05 — 2p from tick_mid 159.03 which is also near round 159.00
    # so proximity check passes and dedupe fires.
    _seed_placed_suggestion(db_path, side="sell", price=159.05, minutes_ago=10)

    _force_allowed_session(monkeypatch)
    monkeypatch.setattr(autonomous_fillmore, "_m3_trend", lambda data_by_tf: "bull")
    monkeypatch.setattr(autonomous_fillmore, "_m1_stack", lambda data_by_tf: "bull")
    monkeypatch.setattr(autonomous_fillmore, "_critical_level_reaction_trigger", lambda *args, **kwargs: _critical_trigger())
    monkeypatch.setattr(autonomous_fillmore, "_trend_expansion_trigger", lambda *args, **kwargs: None)

    decision = autonomous_fillmore.evaluate_gate(
        _gate_cfg("balanced"),
        {"daily_pnl_usd": 0.0, "llm_spend_today_usd": 0.0},
        autonomous_fillmore.GateInputs(
            spread_pips=0.8,
            tick_mid=159.03,  # within 8p of 159.00 round (proximity OK), within 25p bucket of 159.05 (dedupe blocks)
            open_ai_trade_count=0,
            data_by_tf={},
            ntz_active=False,
            suggestions_db_path=db_path,
        ),
    )

    assert decision.result == "block"
    assert decision.layer == "signal"
    assert decision.reason.startswith("repeat_setup_within_")
    assert decision.extras.get("dedupe_prior_side") == "sell"


def test_dedupe_gate_passes_when_no_recent_placement(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "ai_suggestions.sqlite"
    suggestion_tracker.init_db(db_path)  # empty DB

    _force_allowed_session(monkeypatch)
    monkeypatch.setattr(autonomous_fillmore, "_m3_trend", lambda data_by_tf: "bull")
    monkeypatch.setattr(autonomous_fillmore, "_m1_stack", lambda data_by_tf: "bull")
    monkeypatch.setattr(autonomous_fillmore, "_critical_level_reaction_trigger", lambda *args, **kwargs: _critical_trigger())
    monkeypatch.setattr(autonomous_fillmore, "_trend_expansion_trigger", lambda *args, **kwargs: None)

    decision = autonomous_fillmore.evaluate_gate(
        _gate_cfg("balanced"),
        {"daily_pnl_usd": 0.0, "llm_spend_today_usd": 0.0},
        autonomous_fillmore.GateInputs(
            spread_pips=0.8,
            tick_mid=159.03,  # near round level so proximity passes
            open_ai_trade_count=0,
            data_by_tf={},
            ntz_active=False,
            suggestions_db_path=db_path,
        ),
    )

    assert decision.result == "pass"


def test_dedupe_gate_ignores_old_suggestions_outside_window(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "ai_suggestions.sqlite"
    _seed_placed_suggestion(db_path, side="sell", price=159.05, minutes_ago=120)  # outside 30m window

    _force_allowed_session(monkeypatch)
    monkeypatch.setattr(autonomous_fillmore, "_m3_trend", lambda data_by_tf: "bull")
    monkeypatch.setattr(autonomous_fillmore, "_m1_stack", lambda data_by_tf: "bull")
    monkeypatch.setattr(autonomous_fillmore, "_critical_level_reaction_trigger", lambda *args, **kwargs: _critical_trigger())
    monkeypatch.setattr(autonomous_fillmore, "_trend_expansion_trigger", lambda *args, **kwargs: None)

    decision = autonomous_fillmore.evaluate_gate(
        _gate_cfg("balanced"),
        {"daily_pnl_usd": 0.0, "llm_spend_today_usd": 0.0},
        autonomous_fillmore.GateInputs(
            spread_pips=0.8,
            tick_mid=159.03,
            open_ai_trade_count=0,
            data_by_tf={},
            ntz_active=False,
            suggestions_db_path=db_path,
        ),
    )

    assert decision.result == "pass"


def test_correlated_open_position_detects_same_side_within_distance(tmp_path: Path) -> None:
    db_path = tmp_path / "ai_suggestions.sqlite"
    _seed_placed_suggestion(db_path, side="buy", price=159.10, fill_price=159.105, minutes_ago=20)

    hit = autonomous_fillmore._correlated_open_position(db_path, "buy", 159.06, distance_pips=15.0)
    assert hit is not None
    assert hit.get("side") == "buy"

    miss_far = autonomous_fillmore._correlated_open_position(db_path, "buy", 158.50, distance_pips=15.0)
    assert miss_far is None

    miss_other_side = autonomous_fillmore._correlated_open_position(db_path, "sell", 159.06, distance_pips=15.0)
    assert miss_other_side is None


def test_correlated_open_position_ignores_closed_positions(tmp_path: Path) -> None:
    db_path = tmp_path / "ai_suggestions.sqlite"
    _seed_placed_suggestion(
        db_path, side="buy", price=159.10, fill_price=159.105, closed=True, minutes_ago=60,
    )

    assert autonomous_fillmore._correlated_open_position(db_path, "buy", 159.06, distance_pips=15.0) is None


def test_correlated_open_position_ignores_manual_fillmore_positions(tmp_path: Path) -> None:
    db_path = tmp_path / "ai_suggestions.sqlite"
    _seed_placed_suggestion(
        db_path,
        side="buy",
        price=159.10,
        fill_price=159.105,
        minutes_ago=20,
        autonomous=False,
    )

    assert autonomous_fillmore._correlated_open_position(db_path, "buy", 159.06, distance_pips=15.0) is None


def test_build_autonomous_today_block_renders_recent_history(tmp_path: Path) -> None:
    db_path = tmp_path / "ai_suggestions.sqlite"
    _seed_placed_suggestion(db_path, side="sell", price=159.18, fill_price=159.18, closed=True, minutes_ago=45)
    _seed_placed_suggestion(db_path, side="sell", price=159.10, fill_price=159.10, minutes_ago=15)

    block = suggestion_tracker.build_autonomous_today_block(db_path, max_items=5)
    assert "AUTONOMOUS RUN TODAY" in block
    assert "SELL @159.180" in block
    assert "SELL @159.100" in block
    assert "FILLED — still open" in block or "still open" in block
    assert "CLOSED loss" in block


def test_build_autonomous_today_block_handles_empty_db(tmp_path: Path) -> None:
    db_path = tmp_path / "ai_suggestions.sqlite"
    suggestion_tracker.init_db(db_path)
    block = suggestion_tracker.build_autonomous_today_block(db_path)
    assert "No prior autonomous suggestions today" in block


def test_count_open_ai_trades_ignores_manual_ai_manual_rows() -> None:
    class _Store:
        def list_open_trades(self, profile_name: str):
            return [
                {
                    "entry_type": "ai_manual",
                    "notes": "ai_manual:none:order_1",
                    "config_json": json.dumps({"source": "ai_manual"}),
                },
                {
                    "entry_type": "ai_manual",
                    "notes": "autonomous_fillmore:none:order_2",
                    "config_json": json.dumps({"source": "autonomous_fillmore"}),
                },
                {
                    "entry_type": "ai_autonomous",
                    "notes": "autonomous_fillmore:none:order_3",
                    "config_json": json.dumps({"source": "autonomous_fillmore"}),
                },
            ]

    assert autonomous_fillmore._count_open_ai_trades(_Store(), "kumatora2") == 2


# -----------------------------------------------------------------------------
# Stage 6: event blackout, JSON array parsing, exit_plan, reasoning feed
# -----------------------------------------------------------------------------


def test_event_blackout_gate_blocks_when_imminent_high_impact_event(monkeypatch) -> None:
    _force_allowed_session(monkeypatch)
    monkeypatch.setattr(autonomous_fillmore, "_m3_trend", lambda data_by_tf: "bull")
    monkeypatch.setattr(autonomous_fillmore, "_m1_stack", lambda data_by_tf: "bull")
    monkeypatch.setattr(autonomous_fillmore, "_critical_level_reaction_trigger", lambda *args, **kwargs: _critical_trigger())
    monkeypatch.setattr(autonomous_fillmore, "_trend_expansion_trigger", lambda *args, **kwargs: None)

    cfg = _gate_cfg("balanced")
    cfg["event_blackout_enabled"] = True
    cfg["event_blackout_minutes"] = 30

    decision = autonomous_fillmore.evaluate_gate(
        cfg,
        {"daily_pnl_usd": 0.0, "llm_spend_today_usd": 0.0},
        autonomous_fillmore.GateInputs(
            spread_pips=0.8,
            tick_mid=159.03,
            open_ai_trade_count=0,
            data_by_tf={},
            ntz_active=False,
            upcoming_events=[
                {"event": "NFP", "impact": "high", "currency": "USD", "minutes_to_event": 15},
            ],
        ),
    )

    assert decision.result == "block"
    assert decision.layer == "hard"
    assert "event_blackout" in decision.reason
    assert "NFP" in decision.reason


def test_event_blackout_gate_ignores_low_impact_and_other_currencies(monkeypatch) -> None:
    _force_allowed_session(monkeypatch)
    monkeypatch.setattr(autonomous_fillmore, "_m3_trend", lambda data_by_tf: "bull")
    monkeypatch.setattr(autonomous_fillmore, "_m1_stack", lambda data_by_tf: "bull")
    monkeypatch.setattr(autonomous_fillmore, "_critical_level_reaction_trigger", lambda *args, **kwargs: _critical_trigger())
    monkeypatch.setattr(autonomous_fillmore, "_trend_expansion_trigger", lambda *args, **kwargs: None)

    cfg = _gate_cfg("balanced")
    cfg["event_blackout_enabled"] = True
    cfg["event_blackout_minutes"] = 30

    decision = autonomous_fillmore.evaluate_gate(
        cfg,
        {"daily_pnl_usd": 0.0, "llm_spend_today_usd": 0.0},
        autonomous_fillmore.GateInputs(
            spread_pips=0.8,
            tick_mid=159.03,
            open_ai_trade_count=0,
            data_by_tf={},
            ntz_active=False,
            upcoming_events=[
                {"event": "EU CPI", "impact": "high", "currency": "EUR", "minutes_to_event": 10},
                {"event": "Minor Report", "impact": "low", "currency": "USD", "minutes_to_event": 5},
            ],
        ),
    )

    assert decision.result == "pass"


def test_extract_json_object_handles_array() -> None:
    text = (
        "ANALYSIS:\n"
        "Two setups look good.\n"
        "\n"
        "```json\n"
        '[{"side":"buy","price":159.10},{"side":"sell","price":159.50}]\n'
        "```\n"
    )
    json_str, analysis = autonomous_fillmore._extract_json_object(text)
    parsed = json.loads(json_str)
    assert isinstance(parsed, list)
    assert len(parsed) == 2
    assert parsed[0]["side"] == "buy"
    assert parsed[1]["side"] == "sell"
    assert analysis is not None and "Two setups" in analysis


def test_extract_json_object_array_fallback_without_fence() -> None:
    text = 'Some analysis\n[{"side":"buy","price":159.10}]\n'
    json_str, analysis = autonomous_fillmore._extract_json_object(text)
    parsed = json.loads(json_str)
    assert isinstance(parsed, list)
    assert parsed[0]["side"] == "buy"


def test_exit_plan_stored_in_suggestion_tracker(tmp_path: Path) -> None:
    db_path = tmp_path / "ai_suggestions.sqlite"
    suggestion_tracker.init_db(db_path)

    sugg = _suggestion()
    sugg["exit_plan"] = "Hold if M3 stays bull; tighten SL to -3p if M1 flips bear; close on any M5 reversal bar."

    sid = suggestion_tracker.log_generated(
        db_path,
        profile="kumatora2",
        model="gpt-5.4-mini",
        suggestion=sugg,
        ctx=_ctx(),
    )

    history = suggestion_tracker.get_history(db_path, limit=1, offset=0)
    row = history["items"][0]
    assert row["suggestion_id"] == sid
    assert row.get("exit_plan") == sugg["exit_plan"]


def test_reasoning_feed_returns_structured_data(tmp_path: Path) -> None:
    db_path = tmp_path / "ai_suggestions.sqlite"
    suggestion_tracker.init_db(db_path)

    sugg = _suggestion()
    sugg["exit_plan"] = "trail on M1 21 EMA"
    sugg["trigger_family"] = "tight_range_mean_reversion"
    sugg["trigger_reason"] = "tokyo_range_reclaim:session_low"
    suggestion_tracker.log_generated(
        db_path,
        profile="kumatora2",
        model="gpt-5.4-mini",
        suggestion=sugg,
        ctx=_ctx(),
    )

    feed = suggestion_tracker.get_reasoning_feed(db_path)
    assert "suggestions" in feed
    assert "thesis_checks" in feed
    assert "reflections" in feed
    assert len(feed["suggestions"]) == 1
    assert feed["suggestions"][0]["exit_plan"] == "trail on M1 21 EMA"
    assert feed["suggestions"][0]["side"] == "buy"
    assert feed["suggestions"][0]["trigger_family"] == "tight_range_mean_reversion"
    assert feed["suggestions"][0]["trigger_reason"] == "tokyo_range_reclaim:session_low"


def test_compute_risk_regime_hysteresis_and_daily_drawdown() -> None:
    cfg = dict(autonomous_fillmore.DEFAULT_CONFIG)
    cfg["max_daily_loss_usd"] = 50.0

    rt = {
        "consecutive_losses": 3,
        "consecutive_wins": 0,
        "daily_pnl_usd": 0.0,
        "previous_streak_regime_label": "normal",
    }
    hard = autonomous_fillmore._compute_risk_regime(rt, cfg)
    assert hard["label"] == "defensive_hard"
    assert hard["risk_multiplier"] == 0.5

    recovering = {
        "consecutive_losses": 0,
        "consecutive_wins": 1,
        "daily_pnl_usd": 0.0,
        "previous_streak_regime_label": "defensive_hard",
    }
    soft = autonomous_fillmore._compute_risk_regime(recovering, cfg)
    assert soft["label"] == "defensive_soft"

    recovered = {
        "consecutive_losses": 0,
        "consecutive_wins": 2,
        "daily_pnl_usd": 0.0,
        "previous_streak_regime_label": "defensive_hard",
    }
    normal = autonomous_fillmore._compute_risk_regime(recovered, cfg)
    assert normal["label"] == "normal"

    dd = {
        "consecutive_losses": 0,
        "consecutive_wins": 2,
        "daily_pnl_usd": -31.0,
        "previous_streak_regime_label": "normal",
    }
    drawdown = autonomous_fillmore._compute_risk_regime(dd, cfg)
    assert drawdown["label"] == "defensive_hard"
    assert drawdown["daily_drawdown_active"] is True

    drawdown_cleared = {
        "consecutive_losses": 0,
        "consecutive_wins": 0,
        "daily_pnl_usd": 0.0,
        "previous_streak_regime_label": "normal",
        "previous_regime_label": "defensive_hard",
    }
    cleared = autonomous_fillmore._compute_risk_regime(drawdown_cleared, cfg)
    assert cleared["label"] == "normal"
    assert cleared["streak_label"] == "normal"


def test_m1_pullback_or_zone_accepts_recent_pullback_touch() -> None:
    base = [100.0 + (i * 0.01) for i in range(40)]
    close = pd.Series(base, dtype=float)
    low = close.copy()
    high = close + 0.01

    e13 = float(close.ewm(span=13, adjust=False).mean().iloc[-1])
    low.iloc[-3] = e13 - 0.005

    df = pd.DataFrame({
        "close": close,
        "low": low,
        "high": high,
    })

    assert autonomous_fillmore._m1_pullback_or_zone({"M1": df}, "bull") is True


def test_evaluate_gate_passes_trend_expansion_trigger_metadata(monkeypatch) -> None:
    monkeypatch.setattr(autonomous_fillmore, "_session_flag_now", lambda trading_hours: (True, "london/ny"))
    monkeypatch.setattr(autonomous_fillmore, "_m3_trend", lambda data_by_tf: "bull")
    monkeypatch.setattr(autonomous_fillmore, "_m1_stack", lambda data_by_tf: "bull")
    monkeypatch.setattr(autonomous_fillmore, "_critical_level_reaction_trigger", lambda *args, **kwargs: None)
    monkeypatch.setattr(autonomous_fillmore, "_trend_expansion_trigger", lambda *args, **kwargs: _trend_trigger())

    decision = autonomous_fillmore.evaluate_gate(
        _gate_cfg("aggressive"),
        {"daily_pnl_usd": 0.0, "llm_spend_today_usd": 0.0, "last_llm_call_utc": None},
        autonomous_fillmore.GateInputs(
            spread_pips=0.8,
            tick_mid=159.03,
            open_ai_trade_count=0,
            data_by_tf={},
            ntz_active=False,
        ),
    )

    assert decision.result == "pass"
    assert decision.extras.get("trigger_family") == "trend_expansion"
    assert decision.extras.get("trigger_reason") == "adx_trend_expansion"


def test_evaluate_gate_passes_tokyo_mean_reversion_trigger_metadata(monkeypatch) -> None:
    monkeypatch.setattr(autonomous_fillmore, "_session_flag_now", lambda trading_hours: (True, "tokyo"))
    monkeypatch.setattr(autonomous_fillmore, "_m3_trend", lambda data_by_tf: None)
    monkeypatch.setattr(autonomous_fillmore, "_m1_stack", lambda data_by_tf: None)
    monkeypatch.setattr(autonomous_fillmore, "_critical_level_reaction_trigger", lambda *args, **kwargs: None)
    monkeypatch.setattr(autonomous_fillmore, "_tokyo_tight_range_mean_reversion_trigger", lambda *args, **kwargs: _tokyo_meanrev_trigger())
    monkeypatch.setattr(autonomous_fillmore, "_trend_expansion_trigger", lambda *args, **kwargs: None)

    decision = autonomous_fillmore.evaluate_gate(
        _gate_cfg("balanced"),
        {"daily_pnl_usd": 0.0, "llm_spend_today_usd": 0.0, "last_llm_call_utc": None},
        autonomous_fillmore.GateInputs(
            spread_pips=0.8,
            tick_mid=159.03,
            open_ai_trade_count=0,
            data_by_tf={},
            ntz_active=False,
        ),
    )

    assert decision.result == "pass"
    assert decision.extras.get("trigger_family") == "tight_range_mean_reversion"
    assert decision.extras.get("trigger_reason") == "tokyo_range_reclaim:session_low"


def test_evaluate_gate_passes_failed_breakout_trigger_metadata(monkeypatch) -> None:
    monkeypatch.setattr(autonomous_fillmore, "_session_flag_now", lambda trading_hours: (True, "london/ny"))
    monkeypatch.setattr(autonomous_fillmore, "_m3_trend", lambda data_by_tf: None)
    monkeypatch.setattr(autonomous_fillmore, "_m1_stack", lambda data_by_tf: None)
    monkeypatch.setattr(autonomous_fillmore, "_critical_level_reaction_trigger", lambda *args, **kwargs: None)
    monkeypatch.setattr(autonomous_fillmore, "_tokyo_tight_range_mean_reversion_trigger", lambda *args, **kwargs: None)
    monkeypatch.setattr(autonomous_fillmore, "_compression_breakout_trigger", lambda *args, **kwargs: None)
    monkeypatch.setattr(autonomous_fillmore, "_failed_breakout_reversal_trigger", lambda *args, **kwargs: _failed_breakout_trigger())
    monkeypatch.setattr(autonomous_fillmore, "_trend_expansion_trigger", lambda *args, **kwargs: None)

    decision = autonomous_fillmore.evaluate_gate(
        _gate_cfg("balanced"),
        {"daily_pnl_usd": 0.0, "llm_spend_today_usd": 0.0, "last_llm_call_utc": None},
        autonomous_fillmore.GateInputs(
            spread_pips=0.8,
            tick_mid=159.06,
            open_ai_trade_count=0,
            data_by_tf={},
            ntz_active=False,
        ),
    )

    assert decision.result == "pass"
    assert decision.extras.get("trigger_family") == "failed_breakout_reversal_overlap_v1"
    assert decision.extras.get("trigger_reason") == "failed_breakout_recapture:LONDON_NY_SESSION_HIGH"
    assert decision.extras.get("trigger_breakout_side") == "up"
    assert decision.extras.get("trigger_hold_bars") == 2


def test_evaluate_gate_tokyo_controlled_experiment_blocks_non_meanrev_families(monkeypatch) -> None:
    monkeypatch.setattr(autonomous_fillmore, "_session_flag_now", lambda trading_hours: (True, "tokyo"))
    monkeypatch.setattr(autonomous_fillmore, "_m3_trend", lambda data_by_tf: "bull")
    monkeypatch.setattr(autonomous_fillmore, "_m1_stack", lambda data_by_tf: "bull")
    monkeypatch.setattr(autonomous_fillmore, "_critical_level_reaction_trigger", lambda *args, **kwargs: _critical_trigger())
    monkeypatch.setattr(autonomous_fillmore, "_tokyo_tight_range_mean_reversion_trigger", lambda *args, **kwargs: None)
    monkeypatch.setattr(autonomous_fillmore, "_trend_expansion_trigger", lambda *args, **kwargs: _trend_trigger())

    decision = autonomous_fillmore.evaluate_gate(
        _gate_cfg("balanced"),
        {"daily_pnl_usd": 0.0, "llm_spend_today_usd": 0.0, "last_llm_call_utc": None},
        autonomous_fillmore.GateInputs(
            spread_pips=0.8,
            tick_mid=159.03,
            open_ai_trade_count=0,
            data_by_tf={},
            ntz_active=False,
        ),
    )

    assert decision.result == "block"
    assert decision.reason == "no_hybrid_trigger"


def test_evaluate_gate_blocks_when_compression_breakout_is_disabled(monkeypatch) -> None:
    _force_allowed_session(monkeypatch)
    monkeypatch.setattr(autonomous_fillmore, "_m3_trend", lambda data_by_tf: "bull")
    monkeypatch.setattr(autonomous_fillmore, "_m1_stack", lambda data_by_tf: "bull")
    monkeypatch.setattr(autonomous_fillmore, "_critical_level_reaction_trigger", lambda *args, **kwargs: None)
    monkeypatch.setattr(autonomous_fillmore, "_compression_breakout_trigger", lambda *args, **kwargs: _compression_trigger())
    monkeypatch.setattr(autonomous_fillmore, "_trend_expansion_trigger", lambda *args, **kwargs: None)

    decision = autonomous_fillmore.evaluate_gate(
        _gate_cfg("balanced"),
        {"daily_pnl_usd": 0.0, "llm_spend_today_usd": 0.0, "last_llm_call_utc": None},
        autonomous_fillmore.GateInputs(
            spread_pips=0.8,
            tick_mid=159.03,
            open_ai_trade_count=0,
            data_by_tf={},
            ntz_active=False,
        ),
    )

    assert decision.result == "block"
    assert decision.reason == "no_hybrid_trigger"


def test_compression_breakout_limit_is_normalized_to_market() -> None:
    suggestion = {
        **_suggestion(),
        "order_type": "limit",
        "price": 159.180,
    }
    out = autonomous_fillmore._apply_autonomous_order_policy(
        suggestion,
        {
            "spot_price": {"mid": 159.123, "bid": 159.121, "ask": 159.125, "spread_pips": 0.4},
        },
        autonomous_fillmore.GateDecision(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            result="pass",
            layer="pass",
            reason="ok",
            mode="paper",
            aggressiveness="balanced",
            extras={"trigger_family": "compression_breakout"},
        ),
    )

    assert out["order_type"] == "market"
    assert out["requested_order_type"] == "limit"
    assert out["order_policy_reason"] == "compression_breakout_market_only"
    assert out["price"] == 159.123


def test_resolve_gate_thresholds_session_overrides() -> None:
    # NY balanced: ATR 3.5, critical max 7.0 (overrides base 3.0, 6.0)
    bal_ny = autonomous_fillmore._resolve_gate_thresholds("balanced", "ny")
    assert bal_ny["require_min_m5_atr_pips"] == 3.5
    assert bal_ny["critical_level_max_pips"] == 7.0
    assert bal_ny["require_m3_trend"] is True  # inherited from base

    # London balanced: no override, uses base
    bal_lon = autonomous_fillmore._resolve_gate_thresholds("balanced", "london")
    assert bal_lon["require_min_m5_atr_pips"] == 3.0
    assert bal_lon["critical_level_max_pips"] == 6.0

    # Aggressive London: wider critical level distance, looser ADX floor
    agg_lon = autonomous_fillmore._resolve_gate_thresholds("aggressive", "london")
    assert agg_lon["critical_level_max_pips"] == 7.0
    assert agg_lon["trend_adx_min"] == 20.0
    assert agg_lon["critical_micro_window_bars"] == 2

    # very_aggressive is sanitized legacy behavior but still resolves cleanly.
    va_ny = autonomous_fillmore._resolve_gate_thresholds("very_aggressive", "ny")
    assert va_ny["require_min_m5_atr_pips"] == 3.0
    assert va_ny["critical_level_max_pips"] == 6.0


def test_evaluate_gate_blocks_on_malformed_cooldown_timestamp(monkeypatch) -> None:
    _force_allowed_session(monkeypatch)
    monkeypatch.setattr(autonomous_fillmore, "_m3_trend", lambda data_by_tf: "bull")
    monkeypatch.setattr(autonomous_fillmore, "_m1_stack", lambda data_by_tf: "bull")

    decision = autonomous_fillmore.evaluate_gate(
        _gate_cfg("balanced"),
        {"daily_pnl_usd": 0.0, "llm_spend_today_usd": 0.0, "last_llm_call_utc": "not-a-timestamp"},
        autonomous_fillmore.GateInputs(
            spread_pips=0.8,
            tick_mid=159.03,
            open_ai_trade_count=0,
            data_by_tf={},
            ntz_active=False,
        ),
    )

    assert decision.result == "block"
    assert decision.reason == "llm_cooldown:malformed_timestamp"


def test_evaluate_gate_does_not_hard_block_on_wide_spread(monkeypatch) -> None:
    monkeypatch.setattr(autonomous_fillmore, "_session_flag_now", lambda trading_hours: (True, "london/ny"))
    monkeypatch.setattr(autonomous_fillmore, "_m3_trend", lambda data_by_tf: "bull")
    monkeypatch.setattr(autonomous_fillmore, "_m1_stack", lambda data_by_tf: "bull")
    monkeypatch.setattr(autonomous_fillmore, "_m1_pullback_or_zone", lambda data_by_tf, trend, **kwargs: True)
    monkeypatch.setattr(autonomous_fillmore, "_critical_level_reaction_trigger", lambda *args, **kwargs: _critical_trigger())
    monkeypatch.setattr(autonomous_fillmore, "_tokyo_tight_range_mean_reversion_trigger", lambda *args, **kwargs: None)
    monkeypatch.setattr(autonomous_fillmore, "_compression_breakout_trigger", lambda *args, **kwargs: None)
    monkeypatch.setattr(autonomous_fillmore, "_trend_expansion_trigger", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        autonomous_fillmore,
        "_nearest_structure_pips",
        lambda *args, **kwargs: {
            "nearest_pips": 1.5,
            "overhead_pips": None,
            "underfoot_pips": 1.5,
            "overhead_label": None,
            "underfoot_label": "WHOLE_YEN:159.00",
        },
    )

    decision = autonomous_fillmore.evaluate_gate(
        _gate_cfg("balanced"),
        {"daily_pnl_usd": 0.0, "llm_spend_today_usd": 0.0, "last_llm_call_utc": None},
        autonomous_fillmore.GateInputs(
            spread_pips=3.1,
            tick_mid=159.03,
            open_ai_trade_count=0,
            data_by_tf={},
            ntz_active=False,
        ),
    )

    assert decision.result == "pass"
    assert decision.reason == "ok"
    assert decision.extras.get("spread") == 3.1


def test_evaluate_gate_blocks_low_volatility(monkeypatch) -> None:
    _force_allowed_session(monkeypatch)
    monkeypatch.setattr(autonomous_fillmore, "_m3_trend", lambda data_by_tf: "bull")
    monkeypatch.setattr(autonomous_fillmore, "_m1_stack", lambda data_by_tf: "bull")
    monkeypatch.setattr(autonomous_fillmore, "_nearest_structure_pips", lambda tick_mid, data_by_tf: {
        "nearest_pips": 2.0,
        "overhead_pips": 2.0,
        "underfoot_pips": 4.0,
    })

    base = 159.000
    rows = []
    for i in range(25):
        close = base + (i * 0.001)
        rows.append({
            "open": close - 0.0005,
            "high": close + 0.002,
            "low": close - 0.002,
            "close": close,
        })
    m5 = pd.DataFrame(rows)

    decision = autonomous_fillmore.evaluate_gate(
        _gate_cfg("balanced"),
        {"daily_pnl_usd": 0.0, "llm_spend_today_usd": 0.0, "last_llm_call_utc": None},
        autonomous_fillmore.GateInputs(
            spread_pips=0.8,
            tick_mid=159.03,
            open_ai_trade_count=0,
            data_by_tf={"M5": m5},
            ntz_active=False,
        ),
    )

    assert decision.result == "block"
    assert decision.layer == "signal"
    assert decision.reason == "low_volatility"


def test_evaluate_gate_does_not_hard_block_on_open_trade_count(monkeypatch) -> None:
    _force_allowed_session(monkeypatch)
    monkeypatch.setattr(autonomous_fillmore, "_m3_trend", lambda data_by_tf: "bull")
    monkeypatch.setattr(autonomous_fillmore, "_m1_stack", lambda data_by_tf: "bull")
    monkeypatch.setattr(autonomous_fillmore, "_m1_pullback_or_zone", lambda data_by_tf, trend, **kwargs: True)
    monkeypatch.setattr(autonomous_fillmore, "_critical_level_reaction_trigger", lambda *args, **kwargs: _critical_trigger())
    monkeypatch.setattr(autonomous_fillmore, "_tokyo_tight_range_mean_reversion_trigger", lambda *args, **kwargs: None)
    monkeypatch.setattr(autonomous_fillmore, "_compression_breakout_trigger", lambda *args, **kwargs: None)
    monkeypatch.setattr(autonomous_fillmore, "_trend_expansion_trigger", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        autonomous_fillmore,
        "_nearest_structure_pips",
        lambda *args, **kwargs: {
            "nearest_pips": 1.5,
            "overhead_pips": None,
            "underfoot_pips": 1.5,
            "overhead_label": None,
            "underfoot_label": "WHOLE_YEN:159.00",
        },
    )

    decision = autonomous_fillmore.evaluate_gate(
        _gate_cfg("balanced"),
        {"daily_pnl_usd": -125.0, "llm_spend_today_usd": 0.0, "last_llm_call_utc": None},
        autonomous_fillmore.GateInputs(
            spread_pips=0.8,
            tick_mid=159.03,
            open_ai_trade_count=1,
            data_by_tf={},
            ntz_active=False,
        ),
        risk_regime={
            "label": "defensive_hard",
            "effective_min_llm_cooldown_sec": 120,
            "effective_max_open_ai_trades": 1,
        },
    )

    assert decision.result == "pass"
    assert decision.reason == "ok"


def test_compute_risk_regime_keeps_drawdown_informational_not_forced_defensive() -> None:
    cfg = _gate_cfg("balanced")
    rt = {
        "daily_pnl_usd": -95.0,
        "previous_streak_regime_label": "normal",
        "previous_regime_label": "normal",
    }

    regime = autonomous_fillmore._compute_risk_regime(rt, cfg)

    assert regime["daily_drawdown_active"] is True
    assert regime["label"] == "normal"
    assert regime["effective_max_open_ai_trades"] == int(cfg["max_open_ai_trades"])


def test_recompute_performance_stats_materializes_prompt_and_mae_metrics(tmp_path: Path) -> None:
    from storage.sqlite_store import SqliteStore

    suggestions_db = tmp_path / "ai_suggestions.sqlite"
    assistant_db = tmp_path / "assistant.db"
    store = SqliteStore(assistant_db)
    store.init_db()

    sugg = _suggestion(exit_strategy="tp1_be_m5_trail")
    sugg["prompt_version"] = "autonomous_phase_a_v1"
    sugg["prompt_hash"] = "abc123hash"
    sid = suggestion_tracker.log_generated(
        suggestions_db,
        profile="kumatora2",
        model="gpt-5.4-mini",
        suggestion=sugg,
        ctx=_ctx(),
    )
    suggestion_tracker.log_action(
        suggestions_db,
        suggestion_id=sid,
        action="placed",
        edited_fields=None,
        placed_order={"order_type": "limit", "autonomous": True, "side": "buy", "price": 159.25},
        oanda_order_id="701",
    )
    suggestion_tracker.mark_filled(
        suggestions_db,
        oanda_order_id="701",
        fill_price=159.25,
        filled_at="2026-04-16T10:01:00+00:00",
        trade_id="ai_manual:701:1",
    )
    suggestion_tracker.mark_closed(
        suggestions_db,
        oanda_order_id="701",
        exit_price=159.31,
        pnl=12.5,
        pips=6.0,
        closed_at="2026-04-16T10:08:00+00:00",
    )
    store.insert_trade(
        {
            "trade_id": "ai_manual:701:1",
            "timestamp_utc": "2026-04-16T10:01:00+00:00",
            "exit_timestamp_utc": "2026-04-16T10:08:00+00:00",
            "profile": "kumatora2",
            "symbol": "USDJPY",
            "side": "buy",
            "entry_price": 159.25,
            "exit_price": 159.31,
            "profit": 12.5,
            "pips": 6.0,
            "max_adverse_pips": 1.4,
            "max_favorable_pips": 7.2,
            "mae_mfe_estimated": 0,
        }
    )

    autonomous_performance.recompute_performance_stats(
        profile="kumatora2",
        suggestions_db_path=suggestions_db,
        assistant_db_path=assistant_db,
    )
    stats = autonomous_performance.get_materialized_stats(suggestions_db)
    rolling = stats["rolling_20"]
    assert rolling["closed_count"] == 1
    assert rolling["avg_mae_pips"] == 1.4
    assert rolling["avg_mfe_pips"] == 7.2
    breakdown = json.loads(rolling["prompt_version_breakdown_json"])
    assert breakdown["autonomous_phase_a_v1"]["count"] == 1


def test_build_stats_refreshes_runtime_and_materialized_performance_from_history(tmp_path: Path) -> None:
    from storage.sqlite_store import SqliteStore

    state_path = tmp_path / "kumatora2" / "runtime_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "autonomous_fillmore": {
                    "config": {
                        **autonomous_fillmore.DEFAULT_CONFIG,
                        "enabled": True,
                        "mode": "paper",
                    },
                    "runtime": {
                        "daily_pnl_usd": 0.0,
                        "last_stats_recompute_utc": None,
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    suggestions_db = state_path.parent / "ai_suggestions.sqlite"
    assistant_db = state_path.parent / "assistant.db"
    store = SqliteStore(assistant_db)
    store.init_db()

    now = datetime.now(timezone.utc)
    opened = now - timedelta(minutes=12)
    closed = now - timedelta(minutes=4)

    sugg = _suggestion(exit_strategy="tp1_be_m5_trail")
    sugg["prompt_version"] = "autonomous_phase_a_v1"
    sugg["prompt_hash"] = "abc123hash"
    sid = suggestion_tracker.log_generated(
        suggestions_db,
        profile="kumatora2",
        model="gpt-5.4-mini",
        suggestion=sugg,
        ctx=_ctx(),
    )
    suggestion_tracker.log_action(
        suggestions_db,
        suggestion_id=sid,
        action="placed",
        edited_fields=None,
        placed_order={"order_type": "market", "autonomous": True, "side": "buy", "price": 159.25},
        oanda_order_id="701",
    )
    suggestion_tracker.mark_filled(
        suggestions_db,
        oanda_order_id="701",
        fill_price=159.25,
        filled_at=opened.isoformat(),
        trade_id="ai_manual:701:1",
    )
    suggestion_tracker.mark_closed(
        suggestions_db,
        oanda_order_id="701",
        exit_price=159.31,
        pnl=12.5,
        pips=6.0,
        closed_at=closed.isoformat(),
    )
    store.insert_trade(
        {
            "trade_id": "ai_manual:701:1",
            "timestamp_utc": opened.isoformat(),
            "exit_timestamp_utc": closed.isoformat(),
            "profile": "kumatora2",
            "symbol": "USDJPY",
            "side": "buy",
            "entry_price": 159.25,
            "exit_price": 159.31,
            "profit": 12.5,
            "pips": 6.0,
            "max_adverse_pips": 1.4,
            "max_favorable_pips": 7.2,
            "mae_mfe_estimated": 0,
        }
    )

    stats = autonomous_fillmore.build_stats(state_path)

    assert stats["today"]["pnl_usd"] == 12.5
    assert stats["throttle"]["consecutive_wins"] == 1
    assert stats["performance"]["rolling_20"]["closed_count"] == 1
    assert stats["performance"]["rolling_20"]["avg_mae_pips"] == 1.4
    assert stats["performance"]["rolling_20"]["avg_mfe_pips"] == 7.2


def test_build_stats_backfills_missing_closed_autonomous_suggestion_from_trade_history(tmp_path: Path) -> None:
    from storage.sqlite_store import SqliteStore

    state_path = tmp_path / "kumatora2" / "runtime_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "autonomous_fillmore": {
                    "config": {
                        **autonomous_fillmore.DEFAULT_CONFIG,
                        "enabled": True,
                        "mode": "paper",
                    },
                    "runtime": {
                        "daily_pnl_usd": 0.0,
                        "last_stats_recompute_utc": None,
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    suggestions_db = state_path.parent / "ai_suggestions.sqlite"
    assistant_db = state_path.parent / "assistant.db"
    store = SqliteStore(assistant_db)
    store.init_db()

    now = datetime.now(timezone.utc)
    opened = now - timedelta(minutes=15)
    closed = now - timedelta(minutes=5)

    sugg = _suggestion()
    sid = suggestion_tracker.log_generated(
        suggestions_db,
        profile="kumatora2",
        model="gpt-5.4-mini",
        suggestion=sugg,
        ctx=_ctx(),
    )
    suggestion_tracker.log_action(
        suggestions_db,
        suggestion_id=sid,
        action="placed",
        edited_fields=None,
        placed_order={"order_type": "market", "autonomous": True, "side": "buy", "price": 159.25},
        oanda_order_id="701",
    )
    suggestion_tracker.mark_filled(
        suggestions_db,
        oanda_order_id="701",
        fill_price=159.25,
        filled_at=opened.isoformat(),
        trade_id="ai_manual:701:1",
    )
    store.insert_trade(
        {
            "trade_id": "ai_manual:701:1",
            "timestamp_utc": opened.isoformat(),
            "exit_timestamp_utc": closed.isoformat(),
            "profile": "kumatora2",
            "symbol": "USDJPY",
            "side": "buy",
            "entry_price": 159.25,
            "exit_price": 159.31,
            "profit": 12.5,
            "pips": 6.0,
            "max_adverse_pips": 1.4,
            "max_favorable_pips": 7.2,
            "mae_mfe_estimated": 0,
        }
    )

    stats = autonomous_fillmore.build_stats(state_path)
    history = suggestion_tracker.get_history(suggestions_db, limit=5)["items"]
    row = next(item for item in history if item["suggestion_id"] == sid)

    assert row["closed_at"] == closed.isoformat()
    assert row["pnl"] == 12.5
    assert row["pips"] == 6.0
    assert row["win_loss"] == "win"
    assert stats["today"]["pnl_usd"] == 12.5
    assert stats["performance"]["rolling_20"]["closed_count"] == 1


def test_build_stats_rebuilds_today_call_and_placement_counters_from_history(tmp_path: Path) -> None:
    state_path = tmp_path / "kumatora2" / "runtime_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "autonomous_fillmore": {
                    "config": {
                        **autonomous_fillmore.DEFAULT_CONFIG,
                        "enabled": True,
                        "mode": "paper",
                        "model": "gpt-5.4-mini",
                    },
                    "runtime": {
                        "llm_calls_today": 0,
                        "llm_spend_today_usd": 0.0,
                        "trades_placed_today": 0,
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    suggestions_db = state_path.parent / "ai_suggestions.sqlite"
    sugg = _suggestion()
    sid = suggestion_tracker.log_generated(
        suggestions_db,
        profile="kumatora2",
        model="gpt-5.4-mini",
        suggestion=sugg,
        ctx=_ctx(),
    )
    suggestion_tracker.log_action(
        suggestions_db,
        suggestion_id=sid,
        action="placed",
        edited_fields=None,
        placed_order={"order_type": "market", "autonomous": True, "side": "buy", "price": 159.25},
        oanda_order_id="701",
    )

    stats = autonomous_fillmore.build_stats(state_path)

    assert stats["today"]["llm_calls"] == 1
    assert stats["today"]["trades_placed"] == 1
    assert stats["today"]["spend_usd"] > 0


def test_tick_autonomous_fillmore_blocks_below_floor_risk_scaled_lot(tmp_path: Path, monkeypatch) -> None:
    state_path = tmp_path / "kumatora2" / "runtime_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "autonomous_fillmore": {
                    "config": {
                        **autonomous_fillmore.DEFAULT_CONFIG,
                        "enabled": True,
                        "mode": "paper",
                        "min_lot_size": 0.01,
                    },
                    "runtime": {
                        "risk_regime_override": "defensive_hard",
                        "risk_regime_override_until_utc": "2099-01-01T00:00:00+00:00",
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    class _TickStore:
        def list_open_trades(self, profile_name: str):
            return []

    monkeypatch.setattr(autonomous_fillmore, "_session_flag_now", lambda trading_hours: (True, "ny"))
    monkeypatch.setattr(autonomous_fillmore, "_m3_trend", lambda data_by_tf: "bull")
    monkeypatch.setattr(autonomous_fillmore, "_m1_stack", lambda data_by_tf: "bull")
    monkeypatch.setattr(autonomous_fillmore, "_critical_level_reaction_trigger", lambda *args, **kwargs: _critical_trigger())
    monkeypatch.setattr(autonomous_fillmore, "_trend_expansion_trigger", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        autonomous_fillmore,
        "_invoke_suggest",
        lambda profile, profile_name, cfg, risk_regime=None, gate_decision=None: [{
            **_suggestion(),
            "lots": 0.01,
        }],
    )
    monkeypatch.setattr(
        autonomous_fillmore,
        "_place_from_suggestion",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("placement should be vetoed")),
    )

    profile = SimpleNamespace(symbol="USDJPY", pip_size=0.01)
    autonomous_fillmore.tick_autonomous_fillmore(
        profile,
        "kumatora2",
        state_path,
        _TickStore(),
        SimpleNamespace(bid=159.020, ask=159.034),
        data_by_tf={},
        ntz_active=False,
    )

    stats = autonomous_fillmore.build_stats(state_path)
    assert stats["risk_regime"]["label"] == "defensive_hard"
    assert "risk_regime_lot_veto" in stats["recent_gate_blocks"]
