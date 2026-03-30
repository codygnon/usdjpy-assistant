from __future__ import annotations

from datetime import datetime, timezone

from types import SimpleNamespace

from core.dashboard_builder import _phase3_additive_filter_reports, _phase3_defended_filter_reports
from core.dashboard_reporters import collect_phase3_context
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


def test_defended_additive_filter_reports_quarantine_session_panels() -> None:
    reports = _phase3_additive_filter_reports(
        "phase3_integrated_v7_defended",
        {
            "phase3_additive_envelope": {
                "baseline_intents": [{"strategy_tag": "baseline:v44_ny:buy@cell"}],
                "offensive_intents": [{"slice_id": "T3_ambig_mid_pos_sell", "size_scale": 0.25}],
                "accepted": [{"intent_source": "offensive", "slice_id": "T3_ambig_mid_pos_sell"}],
                "rejected": [],
            },
            "phase3_additive_truth": {
                "open_book_count_before": 1,
                "candidate_count": 2,
                "baseline_candidate_count": 1,
                "offensive_candidate_count": 1,
                "accepted_count": 1,
                "accepted_offensive_count": 1,
            },
        },
        {"last_phase3_eval": {}},
    )

    filter_ids = {report.filter_id for report in reports}
    assert "phase3_additive_mode" in filter_ids
    assert "phase3_additive_quarantine" in filter_ids


def test_collect_phase3_context_uses_additive_categories_for_defended_preset() -> None:
    items = collect_phase3_context(
        policy=SimpleNamespace(id="phase3_integrated_v7_defended"),
        data_by_tf={},
        tick=SimpleNamespace(bid=159.5, ask=159.51),
        eval_result={
            "phase3_additive_envelope": {
                "baseline_intents": [{"strategy_tag": "baseline:v44_ny:buy@cell"}],
                "offensive_intents": [{"slice_id": "T3_ambig_mid_pos_sell", "size_scale": 0.25}],
                "accepted": [{"intent_source": "offensive", "slice_id": "T3_ambig_mid_pos_sell"}],
                "rejected": [{"reason": "margin_blocked"}],
            },
            "phase3_additive_truth": {
                "open_book_count_before": 1,
                "candidate_count": 2,
                "baseline_candidate_count": 1,
                "offensive_candidate_count": 1,
                "accepted_count": 1,
                "accepted_offensive_count": 1,
            },
        },
        phase3_state={"last_phase3_eval": {"additive_mode": "defended_additive_runtime_v1"}},
        pip_size=0.01,
        active_preset_name="phase3_integrated_v7_defended",
    )

    categories = {item.category for item in items}
    assert "additive" in categories
    assert "baseline" in categories
    assert "offensive" in categories
    assert "quarantine" in categories
