from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api import fillmore_learning
from api.fillmore_llm_guard import (
    FillmoreLLMCircuitOpenError,
    get_fillmore_llm_health,
    reset_fillmore_llm_health,
    run_guarded_fillmore_llm_call,
)
from api import suggestion_tracker


def test_fillmore_llm_circuit_breaker_opens_and_skips_after_failures() -> None:
    reset_fillmore_llm_health()

    def _boom():
        raise RuntimeError("quota exceeded")

    for _ in range(3):
        try:
            run_guarded_fillmore_llm_call("autonomous_suggest", _boom)
        except RuntimeError:
            pass

    snap = get_fillmore_llm_health()
    assert snap["state"] == "open"
    assert snap["consecutive_failures"] == 3

    try:
        run_guarded_fillmore_llm_call("autonomous_suggest", lambda: {"ok": True})
    except FillmoreLLMCircuitOpenError:
        pass
    else:
        raise AssertionError("expected breaker to skip call while open")


def test_evaluate_trade_thesis_returns_none_when_breaker_open(tmp_path: Path, monkeypatch) -> None:
    reset_fillmore_llm_health()

    db_path = tmp_path / "ai_suggestions.sqlite"
    sid = suggestion_tracker.log_generated(
        db_path,
        profile="kumatora2",
        model="gpt-5.4-mini",
        suggestion={
            "side": "buy",
            "price": 160.0,
            "sl": 159.9,
            "tp": 160.2,
            "lots": 0.1,
            "rationale": "Original thesis.",
            "exit_strategy": "none",
            "exit_params": {},
        },
        ctx={},
    )
    suggestion_tracker.log_action(
        db_path,
        suggestion_id=sid,
        action="placed",
        edited_fields={},
        placed_order={"side": "buy", "price": 160.0, "lots": 0.1, "autonomous": True},
        oanda_order_id="12345",
    )
    suggestion_tracker.mark_filled(db_path, oanda_order_id="12345", fill_price=160.0, trade_id="ai_autonomous:test:1")

    fake_chat = ModuleType("api.ai_trading_chat")
    fake_chat.build_trading_context = lambda profile, profile_name: {}
    fake_chat.resolve_ai_suggest_model = lambda model: model
    monkeypatch.setitem(sys.modules, "api.ai_trading_chat", fake_chat)

    class _PromptBuilder:
        @classmethod
        def for_thesis_monitor(cls, **kwargs):
            return cls()

        def build(self, *, user, prompt_version):
            return SimpleNamespace(model="gpt-5.4-mini", system="sys", user=user)

    fake_prompt_builder = ModuleType("api.prompt_builder")
    fake_prompt_builder.PromptBuilder = _PromptBuilder
    monkeypatch.setitem(sys.modules, "api.prompt_builder", fake_prompt_builder)

    class _FailingOpenAIClient:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            raise RuntimeError("quota exceeded")

    fake_openai = ModuleType("openai")
    fake_openai.OpenAI = _FailingOpenAIClient
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    profile = SimpleNamespace(pip_size=0.01)
    trade_row = {
        "trade_id": "ai_autonomous:test:1",
        "mt5_order_id": "12345",
        "side": "buy",
        "entry_price": 160.0,
        "tp1_partial_done": 0,
        "breakeven_applied": 0,
        "size_lots": 0.1,
        "timestamp_utc": "2026-04-21T00:00:00+00:00",
    }
    position = {"currentUnits": 10000, "unrealizedPL": 1.5}
    tick = SimpleNamespace(bid=160.05, ask=160.07)

    for _ in range(3):
        try:
            fillmore_learning.evaluate_trade_thesis(
                profile=profile,
                profile_name="kumatora2",
                trade_row=trade_row,
                position=position,
                tick=tick,
                db_path=db_path,
            )
        except RuntimeError:
            pass

    assert get_fillmore_llm_health()["state"] == "open"

    second = fillmore_learning.evaluate_trade_thesis(
        profile=profile,
        profile_name="kumatora2",
        trade_row=trade_row,
        position=position,
        tick=tick,
        db_path=db_path,
    )
    assert second is None
