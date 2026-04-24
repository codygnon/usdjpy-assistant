from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import backtest_failed_breakout_family as fb


def _m5_rows(rows: list[tuple[str, float, float, float, float, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["time", "open", "high", "low", "close", "session_label"],
    ).assign(time=lambda df: pd.to_datetime(df["time"], utc=True))


def test_scan_failed_breakout_signals_detects_upside_break_then_short_recapture() -> None:
    m5 = _m5_rows(
        [
            ("2026-04-23T12:00:00Z", 100.00, 100.00, 99.96, 99.98, "london/ny"),
            ("2026-04-23T12:05:00Z", 99.99, 100.04, 99.99, 100.02, "london/ny"),
            ("2026-04-23T12:10:00Z", 100.02, 100.03, 100.01, 100.01, "london/ny"),
            ("2026-04-23T12:15:00Z", 100.01, 100.02, 99.96, 99.97, "london/ny"),
        ]
    )
    m5["session_day"] = m5["time"].dt.floor("D")
    m5["session_block"] = m5["session_day"].astype(str) + "::" + m5["session_label"].astype(str)
    m5["session_bar_index"] = m5.groupby("session_block").cumcount() + 1
    m5["prior_session_high"] = m5.groupby("session_block")["high"].cummax().shift(1)
    m5["prior_session_low"] = m5.groupby("session_block")["low"].cummin().shift(1)
    m5["bar_body_ratio"] = [
        fb._body_ratio(o, h, l, c) for o, h, l, c in zip(m5["open"], m5["high"], m5["low"], m5["close"])
    ]

    signals = fb.scan_failed_breakout_signals(
        m5,
        min_break_pips=2.0,
        max_break_pips=5.0,
        max_hold_bars=3,
        min_session_bars=1,
        recapture_body_ratio=0.5,
        continuation_invalidation_pips=8.0,
    )

    assert len(signals) == 1
    row = signals.iloc[0]
    assert row["signal_direction"] == -1
    assert row["breakout_side"] == "up"
    assert row["hold_bars"] == 2


def test_scan_failed_breakout_signals_detects_downside_break_then_long_recapture() -> None:
    m5 = _m5_rows(
        [
            ("2026-04-23T13:00:00Z", 100.00, 100.03, 100.00, 100.01, "london/ny"),
            ("2026-04-23T13:05:00Z", 100.00, 100.00, 99.96, 99.98, "london/ny"),
            ("2026-04-23T13:10:00Z", 99.98, 99.99, 99.97, 99.97, "london/ny"),
            ("2026-04-23T13:15:00Z", 99.97, 100.02, 99.97, 100.01, "london/ny"),
        ]
    )
    m5["session_day"] = m5["time"].dt.floor("D")
    m5["session_block"] = m5["session_day"].astype(str) + "::" + m5["session_label"].astype(str)
    m5["session_bar_index"] = m5.groupby("session_block").cumcount() + 1
    m5["prior_session_high"] = m5.groupby("session_block")["high"].cummax().shift(1)
    m5["prior_session_low"] = m5.groupby("session_block")["low"].cummin().shift(1)
    m5["bar_body_ratio"] = [
        fb._body_ratio(o, h, l, c) for o, h, l, c in zip(m5["open"], m5["high"], m5["low"], m5["close"])
    ]

    signals = fb.scan_failed_breakout_signals(
        m5,
        min_break_pips=2.0,
        max_break_pips=5.0,
        max_hold_bars=3,
        min_session_bars=1,
        recapture_body_ratio=0.5,
        continuation_invalidation_pips=8.0,
    )

    assert len(signals) == 1
    row = signals.iloc[0]
    assert row["signal_direction"] == 1
    assert row["breakout_side"] == "down"
    assert row["hold_bars"] == 2


def test_materialize_signal_outcomes_uses_side_specific_metrics() -> None:
    frame = pd.DataFrame(
        {
            "signal_direction": [1, -1],
            "long_outcome_pips": [6.0, 3.0],
            "short_outcome_pips": [-4.0, 5.0],
            "long_outcome_code": ["target", "timeout"],
            "short_outcome_code": ["stop", "target"],
            "long_final_pips": [5.0, 1.0],
            "short_final_pips": [-3.0, 4.0],
            "long_mfe_pips": [7.0, 4.0],
            "short_mfe_pips": [1.0, 6.0],
            "long_mae_pips": [2.0, 3.0],
            "short_mae_pips": [5.0, 1.0],
        }
    )

    chosen = fb.materialize_signal_outcomes(frame)

    assert chosen["outcome_pips"].tolist() == [6.0, 5.0]
    assert chosen["outcome_code"].tolist() == ["target", "target"]
    assert chosen["final_pips"].tolist() == [5.0, 4.0]


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

    summary = fb.summarize_signals(frame)

    assert summary.trades == 3
    assert summary.win_rate_pct == pytest.approx(66.67, abs=0.02)
    assert summary.net_pips == pytest.approx(2.0)
    assert summary.profit_factor == pytest.approx(1.2)
