from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import replay_post_spike_retracement_bar_by_bar as replay


def test_replay_trades_bar_by_bar_hits_target_for_short_trade() -> None:
    frame = pd.DataFrame(
        {
            "time": pd.to_datetime(
                [
                    "2026-04-24T12:00:00Z",
                    "2026-04-24T12:01:00Z",
                    "2026-04-24T12:02:00Z",
                    "2026-04-24T12:03:00Z",
                ],
                utc=True,
            ),
            "open": [100.20, 100.18, 100.15, 100.12],
            "high": [100.21, 100.19, 100.16, 100.13],
            "low": [100.19, 100.10, 100.08, 100.07],
            "close": [100.20, 100.12, 100.09, 100.08],
        }
    )
    signals = pd.DataFrame(
        {
            "time": pd.to_datetime(["2026-04-24T12:00:00Z"], utc=True),
            "signal_direction": [-1],
            "session_label": ["ny"],
            "spike_direction": ["up"],
            "spike_move_pips": [18.0],
            "confirmation_bars": [2],
        }
    )

    trades = replay.replay_trades_bar_by_bar(frame, signals, target_pips=6.0, stop_pips=10.0, horizon_bars=3)

    assert len(trades) == 1
    trade = trades[0]
    assert trade.side == "sell"
    assert trade.outcome_code == "target"
    assert trade.pnl_pips == pytest.approx(6.0)


def test_detect_narrowed_signals_filters_to_ny_overlap_and_confirm_2_3() -> None:
    frame = pd.DataFrame(
        {
            "time": pd.to_datetime(
                [
                    "2026-04-24T12:00:00Z",
                    "2026-04-24T12:01:00Z",
                ],
                utc=True,
            ),
            "open": [100.0, 100.0],
            "high": [100.0, 100.0],
            "low": [100.0, 100.0],
            "close": [100.0, 100.0],
        }
    )

    original = replay.psr.scan_post_spike_retracement_signals
    try:
        replay.psr.scan_post_spike_retracement_signals = lambda *args, **kwargs: pd.DataFrame(
            {
                "time": pd.to_datetime(
                    [
                        "2026-04-24T12:00:00Z",
                        "2026-04-24T12:01:00Z",
                        "2026-04-24T12:01:00Z",
                    ],
                    utc=True,
                ),
                "signal_direction": [1, -1, -1],
                "trigger_family": ["post_spike_retracement_v1"] * 3,
                "spike_direction": ["down", "up", "up"],
                "spike_window_bars": [5, 5, 5],
                "spike_move_pips": [15.0, 18.0, 18.0],
                "directional_consistency": [0.8, 0.8, 0.8],
                "confirmation_bars": [2, 1, 3],
                "session_label": ["ny", "ny", "tokyo"],
            }
        )
        narrowed = replay.detect_narrowed_signals(frame)
    finally:
        replay.psr.scan_post_spike_retracement_signals = original

    assert len(narrowed) == 1
    row = narrowed.iloc[0]
    assert row["session_label"] == "ny"
    assert row["confirmation_bars"] == 2
