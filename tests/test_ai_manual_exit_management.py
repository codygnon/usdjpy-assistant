from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api import suggestion_tracker
import run_loop


class _DummyStore:
    def __init__(self, open_rows: list[dict] | None = None) -> None:
        self._open_rows = open_rows or []
        self.inserted: list[dict] = []
        self.updated: list[tuple[str, dict]] = []

    def insert_trade(self, row: dict) -> None:
        self.inserted.append(row)

    def list_open_trades(self, profile: str) -> list[dict]:
        return list(self._open_rows)

    def update_trade(self, trade_id: str, updates: dict) -> None:
        self.updated.append((trade_id, dict(updates)))


class _NoFillAdapter:
    def list_pending_orders(self, symbol: str):
        return []

    def get_position_id_from_order(self, order_id: int):
        return None


class _TrailAdapter:
    def __init__(self) -> None:
        self.stop_updates: list[tuple[int, str, float]] = []
        self.close_calls: list[tuple[int, str, float, int]] = []

    def update_position_stop_loss(self, trade_id: int, symbol: str, stop_loss_price: float) -> None:
        self.stop_updates.append((trade_id, symbol, stop_loss_price))

    def close_position(self, ticket: int, symbol: str, volume: float, position_type: int) -> None:
        self.close_calls.append((ticket, symbol, volume, position_type))


def _install_fake_main(monkeypatch, db_path: Path) -> None:
    fake_main = ModuleType("api.main")
    fake_main._suggestions_db_path = lambda profile_name: db_path
    monkeypatch.setitem(sys.modules, "api.main", fake_main)


def _seed_linked_suggestion(db_path: Path, *, trade_id: str, order_id: str) -> str:
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
        placed_order={"side": "buy", "price": 160.0, "lots": 0.1, "autonomous": False},
        oanda_order_id=order_id,
    )
    suggestion_tracker.mark_filled(db_path, oanda_order_id=order_id, fill_price=160.0, trade_id=trade_id)
    return sid


def test_watch_ai_pending_orders_waits_before_marking_cancelled(tmp_path: Path) -> None:
    profile = SimpleNamespace(profile_name="kumatora2", symbol="USDJPY")
    state_path = tmp_path / "kumatora2" / "runtime_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)

    suggestions_db = state_path.parent / "ai_suggestions.sqlite"
    sid = suggestion_tracker.log_generated(
        suggestions_db,
        profile="kumatora2",
        model="gpt-5.4",
        suggestion={
            "side": "buy",
            "price": 159.2,
            "sl": 159.1,
            "tp": 159.4,
            "lots": 0.1,
            "confidence": "A",
            "rationale": "Test pending AI limit.",
            "exit_strategy": "tp1_be_hwm_trail",
            "exit_params": {},
        },
        ctx={},
    )
    suggestion_tracker.log_action(
        suggestions_db,
        suggestion_id=sid,
        action="placed",
        edited_fields={},
        placed_order={"side": "buy", "price": 159.2, "lots": 0.1, "exit_strategy": "tp1_be_hwm_trail"},
        oanda_order_id="12345",
    )

    state_path.write_text(
        json.dumps(
            {
                "managed_pending_orders": [
                    {"order_id": 12345, "side": "buy", "price": 159.2, "lots": 0.1, "exit_strategy": "tp1_be_hwm_trail"}
                ]
            }
        ),
        encoding="utf-8",
    )

    run_loop._watch_ai_managed_pending_orders(profile, _NoFillAdapter(), _DummyStore(), state_path)

    state1 = json.loads(state_path.read_text(encoding="utf-8"))
    pending1 = state1["managed_pending_orders"]
    assert len(pending1) == 1
    assert pending1[0].get("fill_lookup_started_at_utc")

    row1 = suggestion_tracker.get_by_order_id(suggestions_db, "12345")
    assert row1 is not None
    assert row1.get("outcome_status") is None

    pending1[0]["fill_lookup_started_at_utc"] = "2000-01-01T00:00:00+00:00"
    state_path.write_text(json.dumps(state1), encoding="utf-8")

    run_loop._watch_ai_managed_pending_orders(profile, _NoFillAdapter(), _DummyStore(), state_path)

    state2 = json.loads(state_path.read_text(encoding="utf-8"))
    assert state2["managed_pending_orders"] == []

    row2 = suggestion_tracker.get_by_order_id(suggestions_db, "12345")
    assert row2 is not None
    assert row2.get("outcome_status") == "cancelled"


def test_manage_ai_manual_trades_uses_hwm_override_from_config_json() -> None:
    profile = SimpleNamespace(profile_name="kumatora2", symbol="USDJPY", pip_size=0.01)
    store = _DummyStore(
        open_rows=[
            {
                "trade_id": "ai_manual:12345:1",
                "mt5_position_id": 42,
                "entry_type": "ai_manual",
                "side": "buy",
                "entry_price": 160.000,
                "managed_trail_mode": "hwm",
                "managed_tp1_pips": 6.0,
                "managed_tp1_close_pct": 70.0,
                "managed_be_plus_pips": 0.5,
                "tp1_partial_done": 1,
                "tp1_triggered": 1,
                "breakeven_applied": 1,
                "breakeven_sl_price": 160.020,
                "peak_price": None,
                "config_json": json.dumps(
                    {
                        "source": "ai_manual",
                        "order_id": 12345,
                        "exit_strategy": "tp1_be_hwm_trail",
                        "exit_params": {"hwm_trail_pips": 2.0},
                    }
                ),
            }
        ]
    )
    adapter = _TrailAdapter()
    tick = SimpleNamespace(bid=160.120, ask=160.140)

    run_loop._manage_ai_manual_trades(
        profile,
        adapter,
        store,
        tick,
        open_positions=[{"id": "42"}],
        data_by_tf={},
    )

    assert adapter.stop_updates == [(42, "USDJPY", 160.11)]
    assert any(trade_id == "ai_manual:12345:1" and "peak_price" in updates for trade_id, updates in store.updated)


def test_manage_ai_manual_trades_skips_thesis_monitor_until_cadence_elapsed(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "ai_suggestions.sqlite"
    _install_fake_main(monkeypatch, db_path)
    trade_id = "ai_manual:12345:1"
    _seed_linked_suggestion(db_path, trade_id=trade_id, order_id="12345")
    suggestion_tracker.log_thesis_check(
        db_path,
        profile="kumatora2",
        suggestion_id="sid",
        trade_id=trade_id,
        position_id="42",
        model="gpt-5.4-mini",
        action="hold",
        reason="Recent check.",
        execution_succeeded=True,
        created_utc=datetime.now(timezone.utc).isoformat(),
    )

    called = {"value": False}

    def _fake_eval(*args, **kwargs):
        called["value"] = True
        return {"action": "hold", "reason": "stay in", "confidence": "medium", "model_used": "gpt-5.4-mini"}

    monkeypatch.setattr(run_loop, "_evaluate_fillmore_thesis_monitor", _fake_eval)

    profile = SimpleNamespace(profile_name="kumatora2", symbol="USDJPY", pip_size=0.01)
    opened_at = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    store = _DummyStore(
        open_rows=[
            {
                "trade_id": trade_id,
                "mt5_position_id": 42,
                "mt5_order_id": 12345,
                "entry_type": "ai_manual",
                "side": "buy",
                "entry_price": 160.000,
                "managed_trail_mode": "none",
                "timestamp_utc": opened_at,
                "config_json": json.dumps({"source": "ai_manual", "order_id": 12345}),
            }
        ]
    )

    run_loop._manage_ai_manual_trades(
        profile,
        _TrailAdapter(),
        store,
        SimpleNamespace(bid=160.120, ask=160.140),
        open_positions=[{"id": "42", "currentUnits": "10000"}],
        data_by_tf={},
    )

    assert called["value"] is False


def test_manage_ai_manual_trades_executes_tighten_sl_from_thesis_monitor(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "ai_suggestions.sqlite"
    _install_fake_main(monkeypatch, db_path)
    trade_id = "ai_manual:12345:1"
    _seed_linked_suggestion(db_path, trade_id=trade_id, order_id="12345")

    monkeypatch.setattr(
        run_loop,
        "_evaluate_fillmore_thesis_monitor",
        lambda *args, **kwargs: {
            "action": "tighten_sl",
            "new_sl": 160.050,
            "reason": "Momentum softened.",
            "confidence": "high",
            "model_used": "gpt-5.4-mini",
        },
    )

    profile = SimpleNamespace(profile_name="kumatora2", symbol="USDJPY", pip_size=0.01)
    opened_at = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    store = _DummyStore(
        open_rows=[
            {
                "trade_id": trade_id,
                "mt5_position_id": 42,
                "mt5_order_id": 12345,
                "entry_type": "ai_manual",
                "side": "buy",
                "entry_price": 160.000,
                "managed_trail_mode": "none",
                "stop_price": 159.900,
                "breakeven_sl_price": 160.020,
                "timestamp_utc": opened_at,
                "config_json": json.dumps({"source": "ai_manual", "order_id": 12345}),
            }
        ]
    )
    adapter = _TrailAdapter()

    run_loop._manage_ai_manual_trades(
        profile,
        adapter,
        store,
        SimpleNamespace(bid=160.120, ask=160.140),
        open_positions=[{"id": "42", "currentUnits": "10000", "stopLossOrder": {"price": "160.020"}}],
        data_by_tf={},
    )

    assert adapter.stop_updates == [(42, "USDJPY", 160.05)]
    checks = suggestion_tracker.list_thesis_checks(db_path, trade_id=trade_id, limit=5)
    assert checks[0]["action"] == "tighten_sl"
    assert checks[0]["execution_succeeded"] is True


def test_manage_ai_manual_trades_refuses_looser_thesis_sl(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "ai_suggestions.sqlite"
    _install_fake_main(monkeypatch, db_path)
    trade_id = "ai_manual:12345:1"
    _seed_linked_suggestion(db_path, trade_id=trade_id, order_id="12345")

    monkeypatch.setattr(
        run_loop,
        "_evaluate_fillmore_thesis_monitor",
        lambda *args, **kwargs: {
            "action": "tighten_sl",
            "new_sl": 160.000,
            "reason": "Bad tighten.",
            "confidence": "medium",
            "model_used": "gpt-5.4-mini",
        },
    )

    profile = SimpleNamespace(profile_name="kumatora2", symbol="USDJPY", pip_size=0.01)
    opened_at = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    store = _DummyStore(
        open_rows=[
            {
                "trade_id": trade_id,
                "mt5_position_id": 42,
                "mt5_order_id": 12345,
                "entry_type": "ai_manual",
                "side": "buy",
                "entry_price": 160.000,
                "managed_trail_mode": "none",
                "stop_price": 159.900,
                "breakeven_sl_price": 160.020,
                "timestamp_utc": opened_at,
                "config_json": json.dumps({"source": "ai_manual", "order_id": 12345}),
            }
        ]
    )
    adapter = _TrailAdapter()

    run_loop._manage_ai_manual_trades(
        profile,
        adapter,
        store,
        SimpleNamespace(bid=160.120, ask=160.140),
        open_positions=[{"id": "42", "currentUnits": "10000", "stopLossOrder": {"price": "160.020"}}],
        data_by_tf={},
    )

    assert adapter.stop_updates == []
    checks = suggestion_tracker.list_thesis_checks(db_path, trade_id=trade_id, limit=5)
    assert checks[0]["execution_succeeded"] is False


def test_manage_ai_manual_trades_executes_scale_out_and_exit_now_from_thesis_monitor(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "ai_suggestions.sqlite"
    _install_fake_main(monkeypatch, db_path)
    trade_id_scale = "ai_manual:12345:1"
    trade_id_exit = "ai_manual:12346:1"
    _seed_linked_suggestion(db_path, trade_id=trade_id_scale, order_id="12345")
    _seed_linked_suggestion(db_path, trade_id=trade_id_exit, order_id="12346")

    decisions = [
        {
            "action": "scale_out",
            "scale_out_pct": 50,
            "reason": "Take some off.",
            "confidence": "medium",
            "model_used": "gpt-5.4-mini",
        },
        {
            "action": "exit_now",
            "reason": "Thesis broken.",
            "confidence": "high",
            "model_used": "gpt-5.4-mini",
        },
    ]

    def _fake_eval(*args, **kwargs):
        return decisions.pop(0)

    monkeypatch.setattr(run_loop, "_evaluate_fillmore_thesis_monitor", _fake_eval)

    profile = SimpleNamespace(profile_name="kumatora2", symbol="USDJPY", pip_size=0.01)
    opened_at = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    store = _DummyStore(
        open_rows=[
            {
                "trade_id": trade_id_scale,
                "mt5_position_id": 42,
                "mt5_order_id": 12345,
                "entry_type": "ai_manual",
                "side": "buy",
                "entry_price": 160.000,
                "managed_trail_mode": "none",
                "stop_price": 159.900,
                "timestamp_utc": opened_at,
                "config_json": json.dumps({"source": "ai_manual", "order_id": 12345}),
            },
            {
                "trade_id": trade_id_exit,
                "mt5_position_id": 43,
                "mt5_order_id": 12346,
                "entry_type": "ai_manual",
                "side": "buy",
                "entry_price": 160.000,
                "managed_trail_mode": "none",
                "stop_price": 159.900,
                "timestamp_utc": opened_at,
                "config_json": json.dumps({"source": "ai_manual", "order_id": 12346}),
            },
        ]
    )
    adapter = _TrailAdapter()

    run_loop._manage_ai_manual_trades(
        profile,
        adapter,
        store,
        SimpleNamespace(bid=160.120, ask=160.140),
        open_positions=[
            {"id": "42", "currentUnits": "10000"},
            {"id": "43", "currentUnits": "10000"},
        ],
        data_by_tf={},
    )

    assert adapter.close_calls == [
        (42, "USDJPY", 0.05, 0),
        (43, "USDJPY", 0.1, 0),
    ]
