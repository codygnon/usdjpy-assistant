from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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


def test_simulate_forward_metrics_tracks_mfe_mae_and_final_pips() -> None:
    close = np.array([100.00, 100.00, 100.08, 100.02], dtype=float)
    high = np.array([100.01, 100.07, 100.09, 100.03], dtype=float)
    low = np.array([99.99, 99.98, 100.00, 99.95], dtype=float)
    direction = np.array([1, 0, 0, 0], dtype=np.int8)

    metrics = cag.simulate_forward_metrics(
        close,
        high,
        low,
        direction,
        target_pips=6.0,
        stop_pips=10.0,
        horizon_bars=3,
        pip_size=0.01,
    )

    assert metrics["outcome_pips"][0] == 6.0
    assert metrics["outcome_code"][0] == "target"
    assert metrics["mfe_pips"][0] == pytest.approx(9.0)
    assert metrics["mae_pips"][0] == pytest.approx(5.0)
    assert metrics["final_pips"][0] == pytest.approx(2.0)


def test_build_pass_mask_blocks_spread_and_low_volatility() -> None:
    frame = pd.DataFrame(
        {
            "session_label": ["london/ny", "london/ny"],
            "spread_pips": [3.1, 1.0],
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


def test_filter_frame_by_session_returns_only_requested_rows() -> None:
    frame = pd.DataFrame(
        {
            "session_label": ["tokyo", "london", "tokyo", "ny"],
            "x": [1, 2, 3, 4],
        }
    )

    filtered = cag.filter_frame_by_session(frame, "tokyo")

    assert filtered["x"].tolist() == [1, 3]


def test_apply_quality_labels_marks_top_train_slice_setups() -> None:
    train = pd.DataFrame(
        {
            "direction": [1, 1, 1, 1],
            "outcome_pips": [8.0, 4.0, -2.0, -6.0],
            "final_pips": [5.0, 2.0, -1.0, -4.0],
            "mfe_pips": [10.0, 6.0, 3.0, 1.0],
            "mae_pips": [1.0, 2.0, 6.0, 8.0],
        }
    )
    test = pd.DataFrame(
        {
            "direction": [1, 1],
            "outcome_pips": [7.0, -3.0],
            "final_pips": [4.0, -2.0],
            "mfe_pips": [8.0, 2.0],
            "mae_pips": [1.0, 5.0],
        }
    )

    meta = cag.apply_quality_labels(train, test, target_pips=6.0, stop_pips=10.0, optimal_quantile=0.75)

    assert meta["quality_threshold"] > 0.0
    assert int(train["optimal_setup"].sum()) == 1
    assert bool(train["optimal_setup"].iloc[0]) is True
    assert bool(test["optimal_setup"].iloc[0]) is True


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


def test_choose_recommendation_precision_first_can_prefer_higher_precision_candidate() -> None:
    mode_payload = {
        "datasets": {
            "1000k": {
                "train": {
                    "baseline_summary": {"pass_count": 100},
                    "candidates": [
                        {
                            "params": {"name": "balanced"},
                            "summary": {"pass_count": 60, "positive_splits": 2},
                            "delta_net_pips": 5.0,
                            "delta_profit_factor": 0.05,
                            "delta_quality_score": 0.01,
                            "delta_optimal_precision_pct": 0.5,
                            "delta_optimal_recall_pct": 6.0,
                            "delta_optimal_f1_pct": 1.0,
                            "trade_retention_pct": 60.0,
                        },
                        {
                            "params": {"name": "precise"},
                            "summary": {"pass_count": 35, "positive_splits": 1},
                            "delta_net_pips": 2.0,
                            "delta_profit_factor": 0.02,
                            "delta_quality_score": 0.02,
                            "delta_optimal_precision_pct": 8.0,
                            "delta_optimal_recall_pct": 0.0,
                            "delta_optimal_f1_pct": 6.0,
                            "trade_retention_pct": 35.0,
                        },
                    ],
                },
                "test": {
                    "baseline_summary": {"pass_count": 40},
                    "candidates": [
                        {
                            "params": {"name": "balanced"},
                            "summary": {"pass_count": 25, "positive_splits": 2},
                            "delta_net_pips": 6.0,
                            "delta_profit_factor": 0.04,
                            "delta_quality_score": 0.015,
                            "delta_optimal_precision_pct": 0.5,
                            "delta_optimal_recall_pct": 7.0,
                            "delta_optimal_f1_pct": 1.2,
                            "trade_retention_pct": 62.5,
                        },
                        {
                            "params": {"name": "precise"},
                            "summary": {"pass_count": 18, "positive_splits": 1},
                            "delta_net_pips": 1.0,
                            "delta_profit_factor": 0.01,
                            "delta_quality_score": 0.015,
                            "delta_optimal_precision_pct": 10.0,
                            "delta_optimal_recall_pct": 0.0,
                            "delta_optimal_f1_pct": 7.0,
                            "trade_retention_pct": 45.0,
                        },
                    ],
                },
            }
        }
    }

    recommendation = cag.choose_recommendation(mode_payload, objective="precision_first")

    assert recommendation is not None
    assert recommendation["selected"]["params"]["name"] == "precise"


def test_choose_safe_recommendation_requires_non_negative_test_precision_and_net() -> None:
    mode_payload = {
        "datasets": {
            "1000k": {
                "train": {
                    "candidates": [
                        {
                            "params": {"name": "safe"},
                            "summary": {"pass_count": 60, "positive_splits": 2},
                            "delta_net_pips": 5.0,
                            "delta_quality_score": 0.02,
                            "delta_optimal_precision_pct": 0.0,
                            "delta_optimal_recall_pct": -0.1,
                        },
                        {
                            "params": {"name": "unsafe"},
                            "summary": {"pass_count": 80, "positive_splits": 2},
                            "delta_net_pips": 10.0,
                            "delta_quality_score": 0.03,
                            "delta_optimal_precision_pct": -0.5,
                            "delta_optimal_recall_pct": 2.0,
                        },
                    ]
                },
                "test": {
                    "candidates": [
                        {
                            "params": {"name": "safe"},
                            "summary": {"pass_count": 20, "positive_splits": 1},
                            "delta_net_pips": 3.0,
                            "delta_quality_score": 0.015,
                            "delta_optimal_precision_pct": 0.0,
                            "delta_optimal_recall_pct": 0.0,
                            "delta_optimal_f1_pct": 0.1,
                        },
                        {
                            "params": {"name": "unsafe"},
                            "summary": {"pass_count": 30, "positive_splits": 1},
                            "delta_net_pips": 8.0,
                            "delta_quality_score": 0.03,
                            "delta_optimal_precision_pct": -0.25,
                            "delta_optimal_recall_pct": 3.0,
                            "delta_optimal_f1_pct": 1.2,
                        },
                    ]
                },
            }
        }
    }

    recommendation = cag.choose_safe_recommendation(mode_payload)

    assert recommendation is not None
    assert recommendation["selected"]["params"]["name"] == "safe"


def test_build_session_calibration_returns_per_session_payloads(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_apply_quality_labels(train_frame: pd.DataFrame, test_frame: pd.DataFrame, **_: object) -> dict[str, float]:
        train_frame["setup_quality_score"] = 1.0
        train_frame["optimal_setup"] = True
        test_frame["setup_quality_score"] = 1.0
        test_frame["optimal_setup"] = True
        return {"quality_threshold": 0.5, "optimal_quantile": 0.85, "mfe_floor_pips": 4.5}

    def fake_evaluate_mode_on_dataset(frame: pd.DataFrame, **_: object) -> dict[str, object]:
        passes = int(len(frame))
        return {
            "baseline_summary": {
                "pass_count": passes,
                "avg_quality_score": 1.0,
                "optimal_precision_pct": 50.0,
                "net_pips": 10.0,
            },
                "candidates": [
                    {
                        "params": {
                            "spread_scale": 0.85,
                            "min_m5_atr_pips": 3.0,
                            "level_proximity_pips": 8.0,
                            "pullback_zone_min_pips": 1.5,
                            "pullback_lookback_bars": 3,
                        },
                        "summary": {
                            "pass_count": max(passes - 1, 1),
                            "positive_splits": 2,
                            "net_pips": 12.0,
                            "profit_factor": 1.1,
                            "avg_quality_score": 1.02,
                            "optimal_precision_pct": 50.0,
                            "optimal_recall_pct": 51.0,
                            "optimal_f1_pct": 50.5,
                        },
                        "delta_net_pips": 2.0,
                        "delta_profit_factor": 0.1,
                        "delta_quality_score": 0.02,
                        "delta_optimal_precision_pct": 0.0,
                    "delta_optimal_recall_pct": 1.0,
                    "delta_optimal_f1_pct": 0.5,
                    "trade_retention_pct": 90.0,
                }
            ],
        }

    monkeypatch.setattr(cag, "apply_quality_labels", fake_apply_quality_labels)
    monkeypatch.setattr(cag, "evaluate_mode_on_dataset", fake_evaluate_mode_on_dataset)

    dataset_frames = {
        "1000k": {
            "train": pd.DataFrame(
                {
                    "session_label": ["tokyo", "tokyo", "london"],
                    "direction": [1, 1, 1],
                }
            ),
            "test": pd.DataFrame(
                {
                    "session_label": ["tokyo", "london/ny"],
                    "direction": [1, 1],
                }
            ),
        }
    }

    payload = cag.build_session_calibration(
        dataset_frames,
        mode="balanced",
        target_pips=6.0,
        stop_pips=10.0,
        optimal_quantile=0.85,
        spread_scales=[0.85],
        atr_floors=[3.0],
        level_thresholds=[8.0],
        pullback_zone_mins=[1.5],
        pullback_lookbacks=[3],
    )

    tokyo = payload["tokyo"]
    assert tokyo["recommendation"] is not None
    assert tokyo["recommendation"]["selected"]["params"]["spread_scale"] == 0.85
    assert tokyo["datasets"]["1000k"]["insufficient_data"] is False

    ny = payload["ny"]
    assert ny["datasets"]["1000k"]["insufficient_data"] is True
