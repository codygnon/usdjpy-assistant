from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import backtest_tokyo_range_breakout_family as trb


def _scan_ready_m5(rows: list[tuple[str, float, float, float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["time", "open", "high", "low", "close", "m1_e5", "m1_e9"],
    ).assign(time=lambda df: pd.to_datetime(df["time"], utc=True)).pipe(
        lambda df: df.assign(
            session_label=df["time"].map(trb.cag._classify_session_label),
            minute_of_day=trb._minute_of_day(df["time"]),
            bar_body_ratio=[
                trb.cag._body_ratio(o, h, l, c) for o, h, l, c in zip(df["open"], df["high"], df["low"], df["close"])
            ],
            tokyo_session_key=trb._tokyo_session_key(df["time"]),
        )
    )


def test_scan_tokyo_range_breakout_detects_upside_breakout() -> None:
    m5 = _scan_ready_m5(
        [
            ("2026-04-22T23:05:00Z", 100.00, 100.04, 99.98, 100.02, 100.02, 100.01),
            ("2026-04-23T00:00:00Z", 100.02, 100.08, 100.00, 100.05, 100.05, 100.03),
            ("2026-04-23T06:55:00Z", 100.05, 100.10, 100.02, 100.08, 100.08, 100.05),
            ("2026-04-23T07:00:00Z", 100.08, 100.09, 100.05, 100.07, 100.07, 100.06),
            ("2026-04-23T07:05:00Z", 100.07, 100.14, 100.06, 100.13, 100.13, 100.10),
            ("2026-04-23T07:10:00Z", 100.13, 100.15, 100.11, 100.14, 100.14, 100.12),
        ]
    )
    signals = trb.scan_tokyo_range_breakout_signals(
        m5,
        min_range_pips=8.0,
        max_range_pips=20.0,
        breakout_body_ratio=0.6,
        max_close_to_extreme_pips=2.0,
        max_breakout_close_excursion_pips=8.0,
        require_ema_alignment=True,
    )

    assert len(signals) == 1
    row = signals.iloc[0]
    assert row["signal_direction"] == 1
    assert row["breakout_side"] == "up"
    assert row["tokyo_range_width_pips"] == pytest.approx(12.0)


def test_scan_tokyo_range_breakout_rejects_when_ema_alignment_disagrees() -> None:
    m5 = _scan_ready_m5(
        [
            ("2026-04-22T23:05:00Z", 100.00, 100.04, 99.98, 100.02, 100.02, 100.01),
            ("2026-04-23T00:00:00Z", 100.02, 100.08, 100.00, 100.05, 100.05, 100.03),
            ("2026-04-23T06:55:00Z", 100.05, 100.10, 100.02, 100.08, 100.08, 100.05),
            ("2026-04-23T07:00:00Z", 100.08, 100.09, 100.05, 100.07, 100.07, 100.06),
            ("2026-04-23T07:05:00Z", 100.07, 100.14, 100.06, 100.13, 100.09, 100.12),
        ]
    )
    signals = trb.scan_tokyo_range_breakout_signals(
        m5,
        min_range_pips=8.0,
        max_range_pips=20.0,
        breakout_body_ratio=0.6,
        max_close_to_extreme_pips=2.0,
        max_breakout_close_excursion_pips=8.0,
        require_ema_alignment=True,
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

    summary = trb.summarize_signals(frame)

    assert summary.trades == 3
    assert summary.win_rate_pct == pytest.approx(66.67, abs=0.02)
    assert summary.net_pips == pytest.approx(2.0)
    assert summary.profit_factor == pytest.approx(1.2)
