from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.assistant.position_monitor import MonitorAction
from core.assistant.trade_journal import TradeJournal


def test_new_journal_creates_tables_without_error(tmp_path: Path) -> None:
    journal = TradeJournal(str(tmp_path / "journal.sqlite"))
    tables = {
        row[0]
        for row in journal._conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    }
    assert "trades" in tables
    assert "actions" in tables
    journal.close()


def test_reopening_existing_db_does_not_crash(tmp_path: Path) -> None:
    path = tmp_path / "journal.sqlite"
    TradeJournal(str(path)).close()
    TradeJournal(str(path)).close()


def test_log_trade_opened_inserts_row() -> None:
    journal = TradeJournal(":memory:")
    journal.log_trade_opened("1", "USD_JPY", "long", 150.0, 100000, 149.5, 150.5)
    row = journal._conn.execute("SELECT * FROM trades WHERE trade_id = '1'").fetchone()
    assert row["entry_price"] == 150.0
    assert row["initial_units"] == 100000
    journal.close()


def test_log_trade_closed_populates_exit_fields_for_long() -> None:
    journal = TradeJournal(":memory:")
    journal.log_trade_opened("2", "USD_JPY", "long", 150.0, 100000, 149.5)
    journal.log_trade_closed(trade_id="2", exit_price=150.2, pnl=133.0, exit_reason="tp1")
    row = journal._conn.execute("SELECT * FROM trades WHERE trade_id = '2'").fetchone()
    assert row["exit_price"] == 150.2
    assert row["pnl_pips"] == 20.0
    journal.close()


def test_log_trade_closed_populates_exit_fields_for_short() -> None:
    journal = TradeJournal(":memory:")
    journal.log_trade_opened("3", "USD_JPY", "short", 150.2, 100000, 150.7)
    journal.log_trade_closed(trade_id="3", exit_price=150.0, pnl=133.0, exit_reason="tp1")
    row = journal._conn.execute("SELECT * FROM trades WHERE trade_id = '3'").fetchone()
    assert row["pnl_pips"] == 20.0
    journal.close()


def test_log_action_inserts_row() -> None:
    journal = TradeJournal(":memory:")
    action = MonitorAction(datetime.now(timezone.utc), "1", "trail_stop", "Moved stop", 149.5, 149.8)
    journal.log_action(action)
    row = journal._conn.execute("SELECT * FROM actions").fetchone()
    assert row["action"] == "trail_stop"
    journal.close()


def test_multiple_actions_for_same_trade_all_stored() -> None:
    journal = TradeJournal(":memory:")
    journal.log_action(MonitorAction(datetime.now(timezone.utc), "1", "set_stop", "First"))
    journal.log_action(MonitorAction(datetime.now(timezone.utc), "1", "trail_stop", "Second"))
    count = journal._conn.execute("SELECT COUNT(*) FROM actions WHERE trade_id = '1'").fetchone()[0]
    assert count == 2
    journal.close()


def test_all_time_stats_empty_are_zeroed() -> None:
    journal = TradeJournal(":memory:")
    stats = journal.get_all_time_stats()
    assert stats["total_trades"] == 0
    assert stats["total_pnl"] == 0.0
    journal.close()


def test_all_time_stats_mixed_wins_losses_calculate_correctly() -> None:
    journal = TradeJournal(":memory:")
    for trade_id, pnl in [("1", 100.0), ("2", -50.0), ("3", 150.0)]:
        journal.log_trade_opened(trade_id, "USD_JPY", "long", 150.0, 100000, 149.5)
        journal.log_trade_closed(trade_id=trade_id, exit_price=150.1, pnl=pnl, exit_reason="manual")
    stats = journal.get_all_time_stats()
    assert stats["total_trades"] == 3
    assert stats["wins"] == 2
    assert stats["losses"] == 1
    assert stats["win_rate"] == 2 / 3
    assert stats["profit_factor"] == 5.0
    journal.close()


def test_get_streak_reports_win_and_loss_streaks() -> None:
    journal = TradeJournal(":memory:")
    for idx, pnl in enumerate([100.0, 50.0, -20.0], start=1):
        journal.log_trade_opened(str(idx), "USD_JPY", "long", 150.0, 100000, 149.5)
        journal.log_trade_closed(trade_id=str(idx), exit_price=150.1, pnl=pnl, exit_reason="manual")
        journal._conn.execute(
            "UPDATE trades SET exit_time = ? WHERE trade_id = ?",
            ((datetime.now(timezone.utc) + timedelta(minutes=idx)).isoformat(), str(idx)),
        )
    journal._conn.commit()
    assert journal.get_streak() == ("loss", 1)
    journal.close()


def test_daily_report_with_trades_has_counts_and_pnl() -> None:
    journal = TradeJournal(":memory:")
    target = date(2026, 4, 3)
    journal.log_trade_opened("10", "USD_JPY", "long", 150.0, 100000, 149.5)
    journal.log_trade_closed(trade_id="10", exit_price=150.2, pnl=200.0, exit_reason="tp1")
    journal._conn.execute(
        "UPDATE trades SET exit_time = ?, entry_time = ? WHERE trade_id = '10'",
        (
            datetime(2026, 4, 3, 13, 0, tzinfo=timezone.utc).isoformat(),
            datetime(2026, 4, 3, 9, 0, tzinfo=timezone.utc).isoformat(),
        ),
    )
    journal.log_daily_snapshot(101000.0, 101000.0, snapshot_date=target)
    report = journal.get_daily_report(target)
    assert report["trades_closed"] == 1
    assert report["pnl"] == 200.0
    assert report["equity"] == 101000.0
    journal.close()


def test_daily_report_with_no_trades_is_zero() -> None:
    journal = TradeJournal(":memory:")
    report = journal.get_daily_report(date(2026, 4, 3))
    assert report["trades_closed"] == 0
    assert report["pnl"] == 0
    journal.close()


def test_trade_history_returns_most_recent_first_and_honors_limit() -> None:
    journal = TradeJournal(":memory:")
    for idx in range(3):
        trade_id = str(idx + 1)
        journal.log_trade_opened(trade_id, "USD_JPY", "long", 150.0, 100000, 149.5)
        journal.log_trade_closed(trade_id=trade_id, exit_price=150.1, pnl=10.0 * idx, exit_reason="manual")
        journal._conn.execute(
            "UPDATE trades SET exit_time = ? WHERE trade_id = ?",
            ((datetime(2026, 4, 3, 10, 0, tzinfo=timezone.utc) + timedelta(minutes=idx)).isoformat(), trade_id),
        )
    journal._conn.commit()
    history = journal.get_trade_history(2)
    assert [row["trade_id"] for row in history] == ["3", "2"]
    journal.close()
