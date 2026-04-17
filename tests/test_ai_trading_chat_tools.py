from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api import ai_trading_chat, autonomous_fillmore, suggestion_tracker
from storage.sqlite_store import SqliteStore


def _ctx() -> dict:
    return {
        "spot_price": {"mid": 159.032, "bid": 159.025, "ask": 159.04, "spread_pips": 1.5},
        "session": {"active_sessions": ["Tokyo", "London"], "overlap": "Tokyo/London overlap", "warnings": []},
        "cross_asset_bias": {
            "combined_bias": "bearish",
            "confidence": "high",
            "usdjpy_implication": "bearish USDJPY",
            "oil": {"direction": "bearish"},
            "dxy": {"direction": "bearish"},
        },
        "volatility": {"label": "Above average", "ratio": 1.3, "recent_avg_pips": 2.8},
    }


def _suggestion() -> dict:
    return {
        "side": "sell",
        "price": 159.3,
        "sl": 159.43,
        "tp": 159.22,
        "lots": 0.05,
        "time_in_force": "GTC",
        "gtd_time_utc": None,
        "confidence": "medium",
        "rationale": "Sell the pop into resistance.",
        "exit_strategy": "tp1_be_m5_trail",
        "exit_params": {"tp1_pips": 6.0},
    }


def test_exec_get_ai_suggestion_history_reads_tracker_rows(tmp_path: Path, monkeypatch) -> None:
    profile_name = "kumatora2"
    log_dir = tmp_path / profile_name
    log_dir.mkdir(parents=True, exist_ok=True)
    db_path = log_dir / "ai_suggestions.sqlite"

    sid = suggestion_tracker.log_generated(
        db_path,
        profile=profile_name,
        model="gpt-5.4",
        suggestion=_suggestion(),
        ctx=_ctx(),
    )
    suggestion_tracker.log_action(
        db_path,
        suggestion_id=sid,
        action="placed",
        edited_fields={"lots": {"before": 0.05, "after": 5}},
        placed_order={
            "side": "sell",
            "price": 159.3,
            "lots": 5.0,
            "sl": 159.43,
            "tp": 159.22,
            "time_in_force": "GTC",
            "gtd_time_utc": None,
            "exit_strategy": "tp1_be_m5_trail",
            "exit_params": {"tp1_pips": 6.0},
        },
        oanda_order_id="127777",
    )

    monkeypatch.setattr(ai_trading_chat, "LOGS_DIR", tmp_path)

    out = ai_trading_chat._exec_get_ai_suggestion_history(
        {"days_back": 30, "limit": 5, "oanda_order_id": "127777"},
        profile_name=profile_name,
    )

    assert "Fillmore suggestion for order 127777:" in out
    assert "GTP-5" not in out
    assert "gpt-5.4" in out
    assert "SELL 159.3" in out
    assert "edited: lots: 0.05 -> 5" in out
    assert "context: session=Tokyo/London overlap, macro=bearish, vol=Above average, spread=1.5p" in out
    assert "rationale: Sell the pop into resistance." in out


def test_exec_get_ai_suggestion_history_resolves_closed_trade_results_from_assistant_db(tmp_path: Path, monkeypatch) -> None:
    profile_name = "kumatora2"
    log_dir = tmp_path / profile_name
    log_dir.mkdir(parents=True, exist_ok=True)
    suggestions_db = log_dir / "ai_suggestions.sqlite"
    assistant_db = log_dir / "assistant.db"

    sid = suggestion_tracker.log_generated(
        suggestions_db,
        profile=profile_name,
        model="gpt-5.4",
        suggestion=_suggestion(),
        ctx=_ctx(),
    )
    suggestion_tracker.log_action(
        suggestions_db,
        suggestion_id=sid,
        action="placed",
        edited_fields={},
        placed_order={"side": "sell", "price": 159.3, "lots": 5.0, "exit_strategy": "tp1_be_m5_trail"},
        oanda_order_id="127777",
    )

    store = SqliteStore(assistant_db)
    store.init_db()
    store.insert_trade(
        {
            "trade_id": "ai_manual:127777:1",
            "timestamp_utc": "2026-04-14T07:16:40+00:00",
            "profile": profile_name,
            "symbol": "USD_JPY",
            "side": "sell",
            "entry_price": 159.301,
            "stop_price": 159.43,
            "target_price": 159.22,
            "size_lots": 5.0,
            "notes": "ai_manual:tp1_be_m5_trail:order_127777",
            "snapshot_id": None,
            "exit_price": 159.215,
            "exit_timestamp_utc": "2026-04-14T07:28:00+00:00",
            "exit_reason": "tp_hit",
            "pips": 8.6,
            "mt5_order_id": 127777,
            "profit": 270.5,
            "entry_type": "ai_manual",
        }
    )

    monkeypatch.setattr(ai_trading_chat, "LOGS_DIR", tmp_path)

    out = ai_trading_chat._exec_get_ai_suggestion_history(
        {"days_back": 30, "limit": 5, "oanda_order_id": "127777"},
        profile_name=profile_name,
    )

    assert "outcome=closed/win" in out
    assert "linked_trade=ai_manual:127777:1" in out
    assert "result: pips=+8.6 | pnl=$+270.50 | exit=tp_hit | closed=2026-04-14T07:28" in out


def test_exec_get_ai_suggestion_history_falls_back_to_order_id_match_for_legacy_trade_rows(tmp_path: Path, monkeypatch) -> None:
    profile_name = "newera8"
    log_dir = tmp_path / profile_name
    log_dir.mkdir(parents=True, exist_ok=True)
    suggestions_db = log_dir / "ai_suggestions.sqlite"
    assistant_db = log_dir / "assistant.db"

    sid = suggestion_tracker.log_generated(
        suggestions_db,
        profile=profile_name,
        model="gpt-5.4",
        suggestion=_suggestion(),
        ctx=_ctx(),
    )
    suggestion_tracker.log_action(
        suggestions_db,
        suggestion_id=sid,
        action="placed",
        edited_fields={},
        placed_order={"side": "sell", "price": 159.3, "lots": 5.0, "exit_strategy": "tp1_be_m5_trail"},
        oanda_order_id="99123",
    )

    store = SqliteStore(assistant_db)
    store.init_db()
    store.insert_trade(
        {
            "trade_id": "legacy_trade_99123",
            "timestamp_utc": "2026-04-10T12:01:00+00:00",
            "profile": profile_name,
            "symbol": "USD_JPY",
            "side": "sell",
            "entry_price": 159.303,
            "stop_price": 159.43,
            "target_price": 159.22,
            "size_lots": 5.0,
            "notes": "manual import from broker history",
            "snapshot_id": None,
            "exit_price": 159.251,
            "exit_timestamp_utc": "2026-04-10T12:14:00+00:00",
            "exit_reason": "manual_close",
            "pips": 5.2,
            "mt5_order_id": 99123,
            "profit": 155.0,
            "entry_type": "",
            "opened_by": "",
            "policy_type": "",
        }
    )

    monkeypatch.setattr(ai_trading_chat, "LOGS_DIR", tmp_path)

    out = ai_trading_chat._exec_get_ai_suggestion_history(
        {"days_back": 30, "limit": 5, "oanda_order_id": "99123"},
        profile_name=profile_name,
    )

    assert "outcome=closed/win" in out
    assert "linked_trade=legacy_trade_99123" in out
    assert "result: pips=+5.2 | pnl=$+155.00 | exit=manual_close | closed=2026-04-10T12:14" in out


def test_exec_get_fillmore_system_info_describes_trade_suggestion_flow() -> None:
    out = ai_trading_chat._exec_get_fillmore_system_info(
        {"topic": "trade_suggestion", "include_runtime": False},
        profile_name="kumatora2",
    )

    assert "FILLMORE TRADE SUGGESTION" in out
    assert "manual/operator-assisted flow" in out
    assert "ai_suggestions.sqlite" in out
    assert "suggestion_id" in out
    assert "filled/pending/cancelled/expired" in out


def test_exec_get_fillmore_system_info_includes_autonomous_runtime_snapshot(tmp_path: Path, monkeypatch) -> None:
    profile_name = "kumatora2"
    log_dir = tmp_path / profile_name
    log_dir.mkdir(parents=True, exist_ok=True)
    state_path = log_dir / "runtime_state.json"

    autonomous_fillmore.set_config(
        state_path,
        {
            "mode": "paper",
            "enabled": True,
            "aggressiveness": "balanced",
            "min_confidence": "medium",
            "model": "gpt-5.4-mini",
            "daily_budget_usd": 3.5,
            "max_open_ai_trades": 3,
            "max_daily_loss_usd": 75.0,
            "max_lots_per_trade": 2.5,
        },
    )

    monkeypatch.setattr(ai_trading_chat, "LOGS_DIR", tmp_path)

    out = ai_trading_chat._exec_get_fillmore_system_info(
        {"topic": "autonomous_fillmore", "include_runtime": True},
        profile_name=profile_name,
    )

    assert "AUTONOMOUS FILLMORE" in out
    assert "CURRENT AUTONOMOUS SNAPSHOT" in out
    assert "mode=paper enabled=True aggressiveness=balanced" in out
    assert "aggressiveness=balanced" in out
    assert "daily_budget=$3.50" in out
    assert "max_open_ai_trades=3" in out
