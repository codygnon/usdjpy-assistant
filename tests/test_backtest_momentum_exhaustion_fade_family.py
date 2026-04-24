from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import backtest_momentum_exhaustion_fade_family as mef


def _m5_rows(rows: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["time", "open", "high", "low", "close"],
    ).assign(time=lambda df: pd.to_datetime(df["time"], utc=True))


def test_scan_momentum_exhaustion_detects_up_move_then_bearish_reversal() -> None:
    m5 = mef._prepare_m5_frame(
        _m5_rows(
            [
                ("2026-04-24T12:00:00Z", 100.00, 100.07, 99.99, 100.06),
                ("2026-04-24T12:05:00Z", 100.06, 100.13, 100.05, 100.12),
                ("2026-04-24T12:10:00Z", 100.12, 100.18, 100.11, 100.17),
                ("2026-04-24T12:15:00Z", 100.17, 100.22, 100.16, 100.21),
                ("2026-04-24T12:20:00Z", 100.21, 100.255, 100.20, 100.245),
                ("2026-04-24T12:25:00Z", 100.245, 100.275, 100.235, 100.255),
                ("2026-04-24T12:30:00Z", 100.255, 100.290, 100.248, 100.265),
                ("2026-04-24T12:35:00Z", 100.265, 100.315, 100.260, 100.296),
                ("2026-04-24T12:40:00Z", 100.296, 100.300, 100.255, 100.265),
            ]
        )
    )

    signals = mef.scan_momentum_exhaustion_fade_signals(
        m5,
        move_window_bars=8,
        min_move_pips=18.0,
        min_directional_consistency=0.62,
        decel_bars=3,
        max_late_body_ratio=0.80,
        min_exhaustion_wick_ratio=0.35,
        max_distance_to_window_extreme_pips=2.0,
        confirmation_body_ratio=0.45,
        allowed_sessions={"ny", "london/ny"},
        cooldown_bars=6,
    )

    assert len(signals) == 1
    row = signals.iloc[0]
    assert row["signal_direction"] == -1
    assert row["move_direction"] == "up"
    assert row["session_label"] == "london/ny"


def test_scan_momentum_exhaustion_rejects_without_deceleration() -> None:
    m5 = mef._prepare_m5_frame(
        _m5_rows(
            [
                ("2026-04-24T12:00:00Z", 100.00, 100.05, 99.99, 100.04),
                ("2026-04-24T12:05:00Z", 100.04, 100.09, 100.03, 100.08),
                ("2026-04-24T12:10:00Z", 100.08, 100.14, 100.07, 100.13),
                ("2026-04-24T12:15:00Z", 100.13, 100.19, 100.12, 100.18),
                ("2026-04-24T12:20:00Z", 100.18, 100.25, 100.17, 100.24),
                ("2026-04-24T12:25:00Z", 100.24, 100.31, 100.23, 100.30),
                ("2026-04-24T12:30:00Z", 100.30, 100.37, 100.29, 100.36),
                ("2026-04-24T12:35:00Z", 100.36, 100.43, 100.35, 100.42),
                ("2026-04-24T12:40:00Z", 100.42, 100.43, 100.36, 100.37),
            ]
        )
    )

    signals = mef.scan_momentum_exhaustion_fade_signals(
        m5,
        move_window_bars=8,
        min_move_pips=18.0,
        min_directional_consistency=0.62,
        decel_bars=3,
        max_late_body_ratio=0.80,
        min_exhaustion_wick_ratio=0.35,
        max_distance_to_window_extreme_pips=2.0,
        confirmation_body_ratio=0.45,
        allowed_sessions={"ny", "london/ny"},
        cooldown_bars=6,
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

    summary = mef.summarize_signals(frame)

    assert summary.trades == 3
    assert summary.win_rate_pct == pytest.approx(66.67, abs=0.02)
    assert summary.net_pips == pytest.approx(2.0)
    assert summary.profit_factor == pytest.approx(1.2)
