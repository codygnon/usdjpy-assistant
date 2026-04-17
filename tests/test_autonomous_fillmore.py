from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api import autonomous_fillmore, suggestion_tracker


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
        "confidence": "high",
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


def _force_allowed_session(monkeypatch) -> None:
    monkeypatch.setattr(autonomous_fillmore, "_session_flag_now", lambda trading_hours: (True, "ny"))


def test_invoke_suggest_persists_autonomous_suggestion_history(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "ai_suggestions.sqlite"
    fake_chat = ModuleType("api.ai_trading_chat")
    fake_chat.build_trading_context = lambda profile, profile_name: _ctx()
    fake_chat.build_trade_suggestion_news_block = lambda **kwargs: "NEWS BLOCK"
    fake_chat.resolve_ai_suggest_model = lambda configured: "gpt-5.4-mini"
    fake_chat.system_prompt_from_context = lambda ctx, model: "SYSTEM PROMPT"
    fake_chat.autonomous_system_prompt_from_context = lambda ctx, model: "AUTONOMOUS SYSTEM PROMPT"
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
        {"model": "gpt-5.4-mini", "order_type": "market", "aggressiveness": "balanced", "mode": "paper"},
    )
    out0 = out[0]

    assert out0["suggestion_id"]
    history = suggestion_tracker.get_history(db_path, limit=10, offset=0)
    assert history["total"] == 1
    row = history["items"][0]
    assert row["suggestion_id"] == out0["suggestion_id"]
    assert row["profile"] == "kumatora2"
    assert row["market_snapshot"]["macro_bias"]["combined_bias"] == "bearish"


def test_stage4_default_min_confidence_is_high() -> None:
    assert autonomous_fillmore.DEFAULT_CONFIG["min_confidence"] == "high"


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
    fake_chat.autonomous_system_prompt_from_context = lambda ctx, model: "AUTONOMOUS SYSTEM PROMPT"
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
                "confidence": "high",
            })
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    fake_openai = ModuleType("openai")
    fake_openai.OpenAI = _FakeOpenAIClient
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    out = autonomous_fillmore._invoke_suggest(
        SimpleNamespace(symbol="USDJPY"),
        "kumatora2",
        {"model": "gpt-5.4-mini", "order_type": "limit", "aggressiveness": "balanced", "mode": "paper", "min_confidence": "high"},
    )
    out0 = out[0]

    user_prompt = str((captured.get("messages") or [{}, {}])[1]["content"])
    assert "Current placement threshold is 'high'" in user_prompt
    assert "high = rare. A-tier only" in user_prompt
    assert "medium = B setup / watchlist quality" in user_prompt
    assert "London/NY trend profile" in user_prompt
    system_prompt = str((captured.get("messages") or [{}, {}])[0]["content"])
    assert "SELF-REFLECTION MEMORY" in system_prompt
    assert "Trend alignment was real." in system_prompt
    assert out0["exit_params"]["tp1_pips"] == 6.0
    assert out0["exit_params"]["tp1_close_pct"] == 33.0


def test_invoke_suggest_snaps_limit_price_into_near_spread_band(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "ai_suggestions.sqlite"
    captured: dict[str, object] = {}

    fake_chat = ModuleType("api.ai_trading_chat")
    fake_chat.build_trading_context = lambda profile, profile_name: {
        **_ctx(),
        "spot_price": {"mid": 159.123, "bid": 159.121, "ask": 159.125, "spread_pips": 0.4},
    }
    fake_chat.build_trade_suggestion_news_block = lambda **kwargs: "NEWS BLOCK"
    fake_chat.resolve_ai_suggest_model = lambda configured: "gpt-5.4-mini"
    fake_chat.autonomous_system_prompt_from_context = lambda ctx, model: "AUTONOMOUS SYSTEM PROMPT"
    monkeypatch.setitem(sys.modules, "api.ai_trading_chat", fake_chat)
    _install_fake_main(monkeypatch, db_path)

    class _FakeOpenAIClient:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            captured["messages"] = kwargs["messages"]
            content = json.dumps({
                **_suggestion(),
                "price": 159.250,
                "confidence": "high",
            })
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    fake_openai = ModuleType("openai")
    fake_openai.OpenAI = _FakeOpenAIClient
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    out = autonomous_fillmore._invoke_suggest(
        SimpleNamespace(symbol="USDJPY", pip_size=0.01),
        "kumatora2",
        {"model": "gpt-5.4-mini", "order_type": "limit", "aggressiveness": "balanced", "mode": "paper"},
    )
    out0 = out[0]

    user_prompt = str((captured.get("messages") or [{}, {}])[1]["content"])
    assert "0.1-0.5 pips below current bid" in user_prompt
    assert out0["requested_price"] == 159.250
    assert out0["price"] == 159.120

    history = suggestion_tracker.get_history(db_path, limit=10, offset=0)
    row = history["items"][0]
    assert row["requested_price"] == 159.250
    assert row["limit_price"] == 159.120


def test_limit_snap_guard_downgrades_far_requested_entry_to_pass(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "ai_suggestions.sqlite"
    fake_chat = ModuleType("api.ai_trading_chat")
    fake_chat.build_trading_context = lambda profile, profile_name: {
        **_ctx(),
        "spot_price": {"mid": 159.123, "bid": 159.121, "ask": 159.125, "spread_pips": 0.4},
    }
    fake_chat.build_trade_suggestion_news_block = lambda **kwargs: "NEWS BLOCK"
    fake_chat.resolve_ai_suggest_model = lambda configured: "gpt-5.4-mini"
    fake_chat.autonomous_system_prompt_from_context = lambda ctx, model: "AUTONOMOUS SYSTEM PROMPT"
    monkeypatch.setitem(sys.modules, "api.ai_trading_chat", fake_chat)
    _install_fake_main(monkeypatch, db_path)

    class _FakeOpenAIClient:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            content = json.dumps({
                **_suggestion(),
                "side": "sell",
                "price": 159.451,
                "confidence": "high",
                "rationale": "Sell the 159.45 supply wall after rejection.",
            })
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    fake_openai = ModuleType("openai")
    fake_openai.OpenAI = _FakeOpenAIClient
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    out = autonomous_fillmore._invoke_suggest(
        SimpleNamespace(symbol="USDJPY", pip_size=0.01),
        "kumatora2",
        {"model": "gpt-5.4-mini", "order_type": "limit", "aggressiveness": "balanced", "mode": "paper"},
    )
    out0 = out[0]

    assert out0["requested_price"] == 159.451
    assert out0["price"] == 159.130
    assert out0["snap_distance_pips"] == 32.1
    assert out0["confidence"] == "low"
    assert "too far from the original thesis" in str(out0["rationale"])


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
    assert row["trade_id"].startswith("ai_manual:555001:")
    assert row["managed_trail_mode"] == "none"

    tracked = suggestion_tracker.get_by_order_id(db_path, "555001")
    assert tracked is not None
    assert tracked["trade_id"] == row["trade_id"]
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
    monkeypatch.setattr(autonomous_fillmore, "_m1_pullback_or_zone", lambda data_by_tf, trend: True)

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
    assert decision.reason == "m3_m1_mismatch:bull/bear"


def test_aggressive_gate_requires_pullback_or_zone(monkeypatch) -> None:
    _force_allowed_session(monkeypatch)
    monkeypatch.setattr(autonomous_fillmore, "_m3_trend", lambda data_by_tf: "bull")
    monkeypatch.setattr(autonomous_fillmore, "_m1_stack", lambda data_by_tf: None)
    monkeypatch.setattr(autonomous_fillmore, "_m1_pullback_or_zone", lambda data_by_tf, trend: False)

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
    assert decision.reason == "no_pullback_or_zone"


def test_very_aggressive_gate_still_requires_some_trend_signal(monkeypatch) -> None:
    _force_allowed_session(monkeypatch)
    monkeypatch.setattr(autonomous_fillmore, "_m3_trend", lambda data_by_tf: None)
    monkeypatch.setattr(autonomous_fillmore, "_m1_stack", lambda data_by_tf: None)

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
    assert decision.reason == "no_trend_signal"


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
    monkeypatch.setattr(autonomous_fillmore, "_m1_pullback_or_zone", lambda data_by_tf, trend: True)

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
    assert decision.reason.startswith("no_structure_within_")


def test_proximity_gate_passes_when_price_near_round_level(monkeypatch) -> None:
    _force_allowed_session(monkeypatch)
    monkeypatch.setattr(autonomous_fillmore, "_m3_trend", lambda data_by_tf: "bull")
    monkeypatch.setattr(autonomous_fillmore, "_m1_stack", lambda data_by_tf: "bull")
    monkeypatch.setattr(autonomous_fillmore, "_m1_pullback_or_zone", lambda data_by_tf, trend: True)

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
            "confidence": "high",
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
        placed_order={"order_type": "limit", "side": side, "price": price, "lots": 2.0},
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
    monkeypatch.setattr(autonomous_fillmore, "_m1_pullback_or_zone", lambda data_by_tf, trend: True)

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
    monkeypatch.setattr(autonomous_fillmore, "_m1_pullback_or_zone", lambda data_by_tf, trend: True)

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
    monkeypatch.setattr(autonomous_fillmore, "_m1_pullback_or_zone", lambda data_by_tf, trend: True)

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


# -----------------------------------------------------------------------------
# Stage 6: event blackout, JSON array parsing, exit_plan, reasoning feed
# -----------------------------------------------------------------------------


def test_event_blackout_gate_blocks_when_imminent_high_impact_event(monkeypatch) -> None:
    _force_allowed_session(monkeypatch)
    monkeypatch.setattr(autonomous_fillmore, "_m3_trend", lambda data_by_tf: "bull")
    monkeypatch.setattr(autonomous_fillmore, "_m1_stack", lambda data_by_tf: "bull")
    monkeypatch.setattr(autonomous_fillmore, "_m1_pullback_or_zone", lambda data_by_tf, trend: True)

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
    monkeypatch.setattr(autonomous_fillmore, "_m1_pullback_or_zone", lambda data_by_tf, trend: True)

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
