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
