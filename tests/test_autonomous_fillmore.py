from __future__ import annotations

import json
import os
import sys
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


def test_invoke_suggest_persists_autonomous_suggestion_history(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "ai_suggestions.sqlite"
    fake_chat = ModuleType("api.ai_trading_chat")
    fake_chat.build_trading_context = lambda profile, profile_name: _ctx()
    fake_chat.build_trade_suggestion_news_block = lambda **kwargs: "NEWS BLOCK"
    fake_chat.resolve_ai_suggest_model = lambda configured: "gpt-5.4-mini"
    fake_chat.system_prompt_from_context = lambda ctx, model: "SYSTEM PROMPT"
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

    assert out["suggestion_id"]
    history = suggestion_tracker.get_history(db_path, limit=10, offset=0)
    assert history["total"] == 1
    row = history["items"][0]
    assert row["suggestion_id"] == out["suggestion_id"]
    assert row["profile"] == "kumatora2"
    assert row["market_snapshot"]["macro_bias"]["combined_bias"] == "bearish"


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
