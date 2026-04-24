from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import backtest_post_spike_retracement_family as psr


def _m1_rows(rows: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["time", "open", "high", "low", "close"],
    ).assign(time=lambda df: pd.to_datetime(df["time"], utc=True))


def test_scan_post_spike_retracement_detects_bull_spike_then_short_confirmation() -> None:
    m1 = psr._prepare_m1_frame(
        _m1_rows(
            [
                ("2026-04-24T12:00:00Z", 100.00, 100.03, 99.99, 100.02),
                ("2026-04-24T12:01:00Z", 100.02, 100.06, 100.01, 100.05),
                ("2026-04-24T12:02:00Z", 100.05, 100.09, 100.04, 100.08),
                ("2026-04-24T12:03:00Z", 100.08, 100.12, 100.07, 100.11),
                ("2026-04-24T12:04:00Z", 100.11, 100.15, 100.10, 100.14),
                ("2026-04-24T12:05:00Z", 100.14, 100.15, 100.13, 100.145),
                ("2026-04-24T12:06:00Z", 100.145, 100.146, 100.11, 100.12),
                ("2026-04-24T12:07:00Z", 100.12, 100.13, 100.10, 100.11),
            ]
        )
    )

    signals = psr.scan_post_spike_retracement_signals(
        m1,
        spike_window_bars=5,
        min_spike_pips=12.0,
        min_directional_consistency=0.6,
        max_confirmation_bars=3,
        stall_body_fraction=0.6,
        max_extension_after_spike_pips=4.0,
        cooldown_bars=10,
    )

    assert len(signals) == 1
    row = signals.iloc[0]
    assert row["signal_direction"] == -1
    assert row["spike_direction"] == "up"
    assert row["confirmation_bars"] == 2


def test_scan_post_spike_retracement_respects_cooldown() -> None:
    m1 = psr._prepare_m1_frame(
        _m1_rows(
            [
                ("2026-04-24T12:00:00Z", 100.00, 100.03, 99.99, 100.02),
                ("2026-04-24T12:01:00Z", 100.02, 100.06, 100.01, 100.05),
                ("2026-04-24T12:02:00Z", 100.05, 100.09, 100.04, 100.08),
                ("2026-04-24T12:03:00Z", 100.08, 100.12, 100.07, 100.11),
                ("2026-04-24T12:04:00Z", 100.11, 100.15, 100.10, 100.14),
                ("2026-04-24T12:05:00Z", 100.14, 100.15, 100.13, 100.145),
                ("2026-04-24T12:06:00Z", 100.145, 100.146, 100.11, 100.12),
                ("2026-04-24T12:07:00Z", 100.12, 100.13, 100.10, 100.11),
                ("2026-04-24T12:08:00Z", 100.11, 100.15, 100.10, 100.14),
                ("2026-04-24T12:09:00Z", 100.14, 100.18, 100.13, 100.17),
                ("2026-04-24T12:10:00Z", 100.17, 100.21, 100.16, 100.20),
                ("2026-04-24T12:11:00Z", 100.20, 100.24, 100.19, 100.23),
                ("2026-04-24T12:12:00Z", 100.23, 100.27, 100.22, 100.26),
                ("2026-04-24T12:13:00Z", 100.26, 100.27, 100.25, 100.265),
                ("2026-04-24T12:14:00Z", 100.265, 100.266, 100.23, 100.24),
            ]
        )
    )

    signals = psr.scan_post_spike_retracement_signals(
        m1,
        spike_window_bars=5,
        min_spike_pips=12.0,
        min_directional_consistency=0.6,
        max_confirmation_bars=3,
        stall_body_fraction=0.6,
        max_extension_after_spike_pips=4.0,
        cooldown_bars=10,
    )

    assert len(signals) == 1


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

    summary = psr.summarize_signals(frame)

    assert summary.trades == 3
    assert summary.win_rate_pct == pytest.approx(66.67, abs=0.02)
    assert summary.net_pips == pytest.approx(2.0)
    assert summary.profit_factor == pytest.approx(1.2)
