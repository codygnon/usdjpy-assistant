from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import replay_failed_breakout_bar_by_bar as replay


def test_replay_trades_bar_by_bar_hits_target_for_buy_signal() -> None:
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
            "open": [100.00, 100.00, 100.04, 100.06],
            "high": [100.00, 100.05, 100.08, 100.06],
            "low": [100.00, 99.99, 100.03, 100.02],
            "close": [100.00, 100.04, 100.06, 100.03],
        }
    )
    signals = pd.DataFrame(
        {
            "time": pd.to_datetime(["2026-04-24T12:00:00Z"], utc=True),
            "failed_breakout_direction": [1],
            "failed_breakout_side": ["down"],
            "failed_breakout_excursion_pips": [3.2],
            "failed_breakout_hold_bars": [2],
            "failed_breakout_session_label": ["london/ny"],
        }
    )

    trades = replay.replay_trades_bar_by_bar(
        frame,
        signals,
        target_pips=6.0,
        stop_pips=10.0,
        horizon_bars=3,
    )

    assert len(trades) == 1
    trade = trades[0]
    assert trade.side == "buy"
    assert trade.outcome_code == "target"
    assert trade.pnl_pips == pytest.approx(6.0)
    assert trade.exit_bar_index == 2
