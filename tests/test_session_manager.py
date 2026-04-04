from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.assistant.config import SessionConfig
from core.assistant.session_manager import SessionManager


def test_0300_utc_tokyo_only() -> None:
    status = SessionManager(SessionConfig()).get_status(datetime(2026, 4, 3, 3, 0, tzinfo=timezone.utc))
    assert status.tokyo_active is True
    assert status.london_active is False
    assert status.ny_active is False


def test_0800_utc_tokyo_london_overlap() -> None:
    status = SessionManager(SessionConfig()).get_status(datetime(2026, 4, 3, 8, 0, tzinfo=timezone.utc))
    assert status.overlap == "Tokyo-London"
    assert status.active_sessions == ["Tokyo", "London"]


def test_1300_utc_london_ny_overlap() -> None:
    status = SessionManager(SessionConfig()).get_status(datetime(2026, 4, 3, 13, 0, tzinfo=timezone.utc))
    assert status.overlap == "London-NY"
    assert status.active_sessions == ["London", "New York"]


def test_1800_utc_ny_only() -> None:
    status = SessionManager(SessionConfig()).get_status(datetime(2026, 4, 3, 18, 0, tzinfo=timezone.utc))
    assert status.ny_active is True
    assert status.london_active is False


def test_2200_utc_no_session_active() -> None:
    status = SessionManager(SessionConfig()).get_status(datetime(2026, 4, 3, 22, 0, tzinfo=timezone.utc))
    assert status.active_sessions == []
    assert status.next_close_session == "None"


def test_minutes_to_close_calculated_correctly() -> None:
    status = SessionManager(SessionConfig()).get_status(datetime(2026, 4, 3, 15, 30, tzinfo=timezone.utc))
    assert status.next_close_session == "London"
    assert status.minutes_to_next_close == 30


def test_close_warning_appears_within_threshold() -> None:
    manager = SessionManager(SessionConfig(warn_before_close_minutes=15))
    status = manager.get_status(datetime(2026, 4, 3, 15, 50, tzinfo=timezone.utc))
    text = manager.format_status(status)
    assert "closing in 10 minutes" in text
