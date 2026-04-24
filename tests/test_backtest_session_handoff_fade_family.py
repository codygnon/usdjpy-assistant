from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import backtest_session_handoff_fade_family as shf


def _m5_rows(rows: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["time", "open", "high", "low", "close"],
    ).assign(time=lambda df: pd.to_datetime(df["time"], utc=True))


def test_scan_session_handoff_fade_detects_london_up_then_ny_bearish_reversal() -> None:
    m5 = shf._prepare_m5_frame(
        _m5_rows(
            [
                ("2026-04-24T07:00:00Z", 100.00, 100.04, 99.99, 100.03),
                ("2026-04-24T08:00:00Z", 100.03, 100.08, 100.02, 100.07),
                ("2026-04-24T09:00:00Z", 100.07, 100.12, 100.06, 100.11),
                ("2026-04-24T10:00:00Z", 100.11, 100.15, 100.10, 100.14),
                ("2026-04-24T11:55:00Z", 100.14, 100.16, 100.13, 100.155),
                ("2026-04-24T12:00:00Z", 100.155, 100.156, 100.10, 100.11),
            ]
        )
    )

    signals = shf.scan_session_handoff_fade_signals(
        m5,
        min_london_range_pips=12.0,
        min_london_onesidedness=0.70,
        max_distance_to_extreme_pips=3.0,
        ny_window_bars=6,
        reversal_body_ratio=0.50,
    )

    assert len(signals) == 1
    row = signals.iloc[0]
    assert row["signal_direction"] == -1
    assert row["london_direction"] == "up"
    assert row["ny_bar_index"] == 1


def test_scan_session_handoff_fade_rejects_when_london_not_onesided() -> None:
    m5 = shf._prepare_m5_frame(
        _m5_rows(
            [
                ("2026-04-24T07:00:00Z", 100.00, 100.06, 99.98, 100.02),
                ("2026-04-24T08:00:00Z", 100.02, 100.10, 100.00, 100.04),
                ("2026-04-24T09:00:00Z", 100.04, 100.11, 100.01, 100.03),
                ("2026-04-24T10:00:00Z", 100.03, 100.09, 100.00, 100.04),
                ("2026-04-24T11:55:00Z", 100.04, 100.12, 100.02, 100.05),
                ("2026-04-24T12:00:00Z", 100.05, 100.06, 100.00, 100.01),
            ]
        )
    )

    signals = shf.scan_session_handoff_fade_signals(
        m5,
        min_london_range_pips=12.0,
        min_london_onesidedness=0.70,
        max_distance_to_extreme_pips=3.0,
        ny_window_bars=6,
        reversal_body_ratio=0.50,
    )

    assert signals.empty


def test_summarize_signals_reports_expected_profit_factor() -> None:
    frame = pd.DataFrame(
        {
            "signal_direction": [1, -1, 1],
            "long_outcome_pips": [6.0, 0.0, -10.0],
            "short_outcome_pips": [0.0, 6.0, 0.0],
            "long_outcome_code": ["target", "no_trade", "stop"],
            "short_outcome_code": ["no_trade", "target", "no_trade"],
            "long_final_pips": [4.0, 0.0, -7.0],
            "short_final_pips": [0.0, 5.0, 0.0],
            "long_mfe_pips": [8.0, 0.0, 3.0],
            "short_mfe_pips": [0.0, 7.0, 0.0],
            "long_mae_pips": [2.0, 0.0, 10.0],
            "short_mae_pips": [0.0, 1.0, 0.0],
        }
    )

    summary = shf.summarize_signals(frame)

    assert summary.trades == 3
    assert summary.win_rate_pct == pytest.approx(66.67, abs=0.02)
    assert summary.net_pips == pytest.approx(2.0)
    assert summary.profit_factor == pytest.approx(1.2)
