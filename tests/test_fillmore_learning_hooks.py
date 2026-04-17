from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api import main as api_main
from api import paper_fillmore, suggestion_tracker


def _install_fake_openai(monkeypatch) -> None:
    class _FakeOpenAIClient:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps(
                                {
                                    "what_read_right": "Context alignment was valid.",
                                    "what_missed": "The exit area was too crowded.",
                                }
                            )
                        )
                    )
                ]
            )

    fake_openai = ModuleType("openai")
    fake_openai.OpenAI = _FakeOpenAIClient
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def _seed_closed_trade_link(db_path: Path, *, profile: str, trade_id: str, order_id: str, side: str = "buy") -> str:
    sid = suggestion_tracker.log_generated(
        db_path,
        profile=profile,
        model="gpt-5.4-mini",
        suggestion={
            "side": side,
            "price": 160.0,
            "sl": 159.9,
            "tp": 160.2,
            "lots": 0.1,
            "confidence": "high",
            "rationale": "Original Fillmore thesis.",
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
        placed_order={"side": side, "price": 160.0, "lots": 0.1, "autonomous": False},
        oanda_order_id=order_id,
    )
    suggestion_tracker.mark_filled(db_path, oanda_order_id=order_id, fill_price=160.0, trade_id=trade_id)
    return sid


class _PaperStore:
    def __init__(self) -> None:
        self.closed: list[tuple[str, dict]] = []

    def close_trade(self, trade_id: str, updates: dict) -> None:
        self.closed.append((trade_id, dict(updates)))


class _SyncStore:
    def __init__(self, open_rows: list[dict]) -> None:
        self._open_rows = open_rows
        self.closed: list[tuple[str, dict]] = []

    def list_open_trades(self, profile_name: str) -> list[dict]:
        return list(self._open_rows)

    def close_trade(self, trade_id: str, updates: dict) -> None:
        self.closed.append((trade_id, dict(updates)))


def test_finalize_paper_close_creates_reflection(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "ai_suggestions.sqlite"
    monkeypatch.setattr(api_main, "_suggestions_db_path", lambda profile_name: db_path)
    _install_fake_openai(monkeypatch)

    trade_id = "ai_manual:paper:42:1"
    sid = _seed_closed_trade_link(db_path, profile="kumatora2", trade_id=trade_id, order_id="paper:sid")
    trade_row = {
        "trade_id": trade_id,
        "side": "buy",
        "entry_price": 160.0,
        "size_lots": 0.1,
        "timestamp_utc": "2026-04-16T10:00:00+00:00",
        "config_json": json.dumps({"suggestion_id": sid, "paper_order_id": "paper:sid"}),
    }

    paper_fillmore.finalize_paper_close(
        profile_name="kumatora2",
        state_path=tmp_path / "runtime_state.json",
        store=_PaperStore(),
        trade_row=trade_row,
        exit_price=160.08,
        exit_reason="paper_tp",
        pip_size=0.01,
        mid=160.08,
    )

    assert suggestion_tracker.reflection_exists(
        db_path,
        profile="kumatora2",
        suggestion_id=sid,
        trade_id=trade_id,
    ) is True


def test_sync_open_trades_with_broker_creates_reflection(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "ai_suggestions.sqlite"
    monkeypatch.setattr(api_main, "_suggestions_db_path", lambda profile_name: db_path)
    monkeypatch.setattr(api_main, "_apply_phase3_sync_close_state_update", lambda **kwargs: None)
    _install_fake_openai(monkeypatch)

    class _FakeAdapter:
        def initialize(self) -> None:
            return None

        def ensure_symbol(self, symbol: str) -> None:
            return None

        def get_open_positions(self, symbol: str):
            return []

        def get_position_close_info(self, position_id: int):
            return SimpleNamespace(exit_price=160.07, exit_time_utc="2026-04-16T10:05:00+00:00", profit=7.5)

        def shutdown(self) -> None:
            return None

    fake_broker = ModuleType("adapters.broker")
    fake_broker.get_adapter = lambda profile: _FakeAdapter()
    monkeypatch.setitem(sys.modules, "adapters.broker", fake_broker)

    trade_id = "ai_manual:12345:1"
    sid = _seed_closed_trade_link(db_path, profile="kumatora2", trade_id=trade_id, order_id="12345")
    store = _SyncStore(
        [
            {
                "trade_id": trade_id,
                "mt5_position_id": 42,
                "mt5_order_id": 12345,
                "entry_type": "ai_manual",
                "side": "buy",
                "entry_price": 160.0,
                "stop_price": 159.9,
                "target_price": 160.2,
                "timestamp_utc": "2026-04-16T10:00:00+00:00",
            }
        ]
    )

    synced = api_main._sync_open_trades_with_broker(
        SimpleNamespace(profile_name="kumatora2", symbol="USDJPY", pip_size=0.01),
        store,
    )

    assert synced == 1
    tracked = suggestion_tracker.get_by_trade_id(db_path, trade_id)
    assert tracked is not None
    assert tracked["closed_at"] == "2026-04-16T10:05:00+00:00"
    assert suggestion_tracker.reflection_exists(
        db_path,
        profile="kumatora2",
        suggestion_id=sid,
        trade_id=trade_id,
    ) is True
