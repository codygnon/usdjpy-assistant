from __future__ import annotations

from datetime import datetime, timezone

from core.dashboard_builder import _phase3_defended_filter_reports
from core.phase3_integrated_engine import report_phase3_ny_caps


def test_defended_filter_reports_allow_customized_preset_name() -> None:
    reports = _phase3_defended_filter_reports(
        "phase3_integrated_v7_defended (customized)",
        {
            "london_v2": {"d_suppress_weekdays": ["Monday", "Tuesday"], "d_tp1_r": 3.25, "d_be_offset_pips": 1.0, "d_tp2_r": 2.0},
            "v44_ny": {"defensive_veto_cells": ["ambiguous/er_low/der_neg"]},
            "v14": {"cell_scale_overrides": {"ambiguous/er_mid/der_pos:sell": 0.25}},
        },
    )

    filter_ids = {report.filter_id for report in reports}
    assert "phase3_frozen_package" in filter_ids
    assert "phase3_frozen_defensive_veto" in filter_ids


def test_report_phase3_ny_caps_uses_runtime_config_values() -> None:
    phase3_state = {
        "open_trade_count": 5,
        "session_ny_2025-04-04": {
            "trade_count": 2,
            "consecutive_losses": 3,
        },
    }

    reports = report_phase3_ny_caps(
        phase3_state,
        datetime(2025, 4, 4, 14, 0, tzinfo=timezone.utc),
        {"max_open_positions": 0, "session_stop_losses": 3},
    )

    by_name = {row["name"]: row for row in reports}
    assert by_name["NY Max open"]["value"] == "5/unlimited"
    assert by_name["NY Max open"]["ok"] is True
    assert by_name["NY Session stop losses"]["value"] == "3/3"
    assert by_name["NY Session stop losses"]["ok"] is False
