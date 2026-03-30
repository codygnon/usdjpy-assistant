from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from core.phase3_session_core import classify_phase3_session, resolve_phase3_bar_time


def test_classify_phase3_session_uses_configured_tokyo_window() -> None:
    cfg = {
        "v14": {
            "session_start_utc": "16:00",
            "session_end_utc": "22:00",
            "allowed_trading_days": ["Tuesday", "Wednesday", "Friday"],
        },
        "v44_ny": {
            "ny_window_mode": "fixed_utc",
            "ny_start_hour": 12,
            "ny_end_hour": 15,
        },
    }
    # 21:30 UTC Tuesday — after NY window, still inside Tokyo 16:00-22:00
    now_utc = datetime(2024, 10, 22, 21, 30, tzinfo=timezone.utc)
    assert classify_phase3_session(now_utc, cfg) == "tokyo"


def test_classify_phase3_session_uses_london_active_days_and_window() -> None:
    cfg = {
        "london_v2": {
            "active_days_utc": ["Tuesday", "Wednesday"],
            "a_entry_end_min_after_london": 90,
            "d_entry_end_min_after_london": 120,
        }
    }
    now_utc = datetime(2024, 10, 22, 8, 30, tzinfo=timezone.utc)  # Tuesday, London winter open
    assert classify_phase3_session(now_utc, cfg) == "london"


def test_classify_phase3_session_uses_fixed_utc_ny_window() -> None:
    cfg = {
        "v44_ny": {
            "ny_window_mode": "fixed_utc",
            "ny_start_hour": 13,
            "ny_end_hour": 16,
        }
    }
    now_utc = datetime(2024, 10, 22, 14, 15, tzinfo=timezone.utc)
    assert classify_phase3_session(now_utc, cfg) == "ny"


def test_resolve_phase3_bar_time_uses_latest_closed_m1_bar() -> None:
    df = pd.DataFrame(
        {
            "time": [
                pd.Timestamp("2024-10-22T10:00:00Z"),
                pd.Timestamp("2024-10-22T10:01:00Z"),
            ],
            "close": [150.0, 150.1],
        }
    )
    fallback = datetime(2024, 10, 22, 10, 5, tzinfo=timezone.utc)
    resolved = resolve_phase3_bar_time({"M1": df}, fallback)
    assert resolved == datetime(2024, 10, 22, 10, 1, tzinfo=timezone.utc)
