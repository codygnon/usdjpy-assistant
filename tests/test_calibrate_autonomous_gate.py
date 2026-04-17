from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import calibrate_autonomous_gate as cag


def test_classify_session_label_handles_overlap() -> None:
    ts = pd.Timestamp("2025-01-01T12:30:00Z")
    assert cag._classify_session_label(ts) == "london/ny"


def test_simulate_first_touch_outcomes_hits_target_before_stop_for_buy() -> None:
    close = np.array([100.00, 100.00, 100.08, 100.02], dtype=float)
    high = np.array([100.01, 100.07, 100.09, 100.03], dtype=float)
    low = np.array([99.99, 99.98, 100.00, 99.95], dtype=float)
    direction = np.array([1, 0, 0, 0], dtype=np.int8)

    out_pips, out_code = cag.simulate_first_touch_outcomes(
        close,
        high,
        low,
        direction,
        target_pips=6.0,
        stop_pips=10.0,
        horizon_bars=3,
        pip_size=0.01,
    )

    assert out_pips[0] == 6.0
    assert out_code[0] == "target"


def test_build_pass_mask_blocks_spread_and_low_volatility() -> None:
    frame = pd.DataFrame(
        {
            "session_label": ["london/ny", "london/ny"],
            "spread_pips": [1.6, 1.0],
            "direction": [1, 1],
            "m3_trend": ["bull", "bull"],
            "m1_stack": ["bull", "bull"],
            "m5_atr_pips": [4.0, 2.0],
            "nearest_structure_pips": [3.0, 3.0],
            "near_daily_hl_pips": [10.0, 10.0],
            "close": [159.00, 159.01],
            "m1_e9": [159.00, 159.01],
            "m1_e13": [158.99, 159.00],
            "m1_e17": [158.98, 158.99],
            "m1_atr_pips": [3.0, 3.0],
            "low": [158.98, 158.99],
            "high": [159.02, 159.03],
        }
    )

    mask = cag.build_pass_mask(
        frame,
        mode="balanced",
        spread_scale=1.0,
        min_m5_atr_pips=3.0,
        level_proximity_pips=8.0,
        pullback_zone_min_pips=1.5,
        pullback_lookback_bars=3,
    )

    assert mask.tolist() == [False, False]


def test_split_frame_train_test_uses_requested_recent_window() -> None:
    frame = pd.DataFrame({"x": np.arange(1000)})

    train, test = cag.split_frame_train_test(frame, train_bars=700, test_bars=200)

    assert len(train) == 700
    assert len(test) == 200
    assert train["x"].iloc[0] == 100
    assert test["x"].iloc[0] == 800


def test_choose_recommendation_prefers_positive_out_of_sample_candidate() -> None:
    mode_payload = {
        "datasets": {
            "1000k": {
                "train": {
                    "baseline_summary": {"pass_count": 100},
                    "candidates": [
                        {
                            "params": {"spread_scale": 1.0},
                            "summary": {"pass_count": 90, "positive_splits": 3},
                            "delta_net_pips": 20.0,
                            "delta_profit_factor": 0.2,
                            "trade_retention_pct": 90.0,
                        },
                        {
                            "params": {"spread_scale": 0.85},
                            "summary": {"pass_count": 80, "positive_splits": 2},
                            "delta_net_pips": 50.0,
                            "delta_profit_factor": 0.4,
                            "trade_retention_pct": 80.0,
                        },
                    ],
                },
                "test": {
                    "baseline_summary": {"pass_count": 40},
                    "candidates": [
                        {
                            "params": {"spread_scale": 1.0},
                            "summary": {"pass_count": 30, "positive_splits": 2},
                            "delta_net_pips": 18.0,
                            "delta_profit_factor": 0.15,
                            "trade_retention_pct": 75.0,
                        },
                        {
                            "params": {"spread_scale": 0.85},
                            "summary": {"pass_count": 6, "positive_splits": 1},
                            "delta_net_pips": -10.0,
                            "delta_profit_factor": -0.2,
                            "trade_retention_pct": 15.0,
                        },
                    ],
                },
            }
        }
    }

    recommendation = cag.choose_recommendation(mode_payload)

    assert recommendation is not None
    assert recommendation["primary_dataset"] == "1000k"
    assert recommendation["selected"]["params"]["spread_scale"] == 1.0
