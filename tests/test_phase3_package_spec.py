from __future__ import annotations

import json
from pathlib import Path

from core.phase3_package_spec import (
    DEFENDED_PACKAGE_ID,
    PHASE3_DEFENDED_PRESET_ID,
    load_phase3_package_spec,
    project_runtime_config_from_spec,
    uses_defended_phase3_package,
)


def test_load_phase3_package_spec_projects_defended_runtime_overrides(tmp_path: Path) -> None:
    paper = {
        "candidate_name": DEFENDED_PACKAGE_ID,
        "package_family": "v7_pfdd",
        "status": "paper_candidate_frozen_for_runtime_implementation",
        "base_cell_scales": {
            "L1_mom_low_pos_buy": 1.0,
            "T3_ambig_mid_pos_sell": 0.25,
        },
        "overrides": {
            "l1_weekday_disable": ["Monday", "Tuesday"],
            "l1_exit_override": {
                "tp1_r_multiple": 3.25,
                "be_offset_pips": 1.0,
                "tp2_r_multiple": 2.0,
            },
            "defensive_veto": {
                "strategy": "v44_ny",
                "ownership_cell": "ambiguous/er_low/der_neg",
            },
        },
        "strict_policy": {
            "name": "native_v44_hedging_like",
            "allow_internal_overlap": True,
            "allow_opposite_side_overlap": True,
            "max_open_offensive": None,
            "max_entries_per_day": None,
            "max_lot_per_trade": 20.0,
        },
    }
    runtime_cfg = {"v44_ny": {"risk_per_trade_pct": 0.5}}
    paper_path = tmp_path / "paper_candidate_v7_defended.json"
    runtime_path = tmp_path / "phase3_integrated_sizing_config.json"
    paper_path.write_text(json.dumps(paper), encoding="utf-8")
    runtime_path.write_text(json.dumps(runtime_cfg), encoding="utf-8")

    spec = load_phase3_package_spec(
        preset_id=PHASE3_DEFENDED_PRESET_ID,
        paper_candidate_path=paper_path,
        runtime_config_path=runtime_path,
    )

    assert spec.package_id == DEFENDED_PACKAGE_ID
    projected = project_runtime_config_from_spec(spec)
    assert projected["london_v2"]["d_suppress_weekdays"] == ["Monday", "Tuesday"]
    assert projected["london_v2"]["d_tp1_r"] == 3.25
    assert projected["v44_ny"]["defensive_veto_cells"] == ["ambiguous/er_low/der_neg"]
    assert projected["v44_ny"]["strict_policy_name"] == "native_v44_hedging_like"
    assert projected["v44_ny"]["allow_internal_overlap"] is True
    assert projected["v44_ny"]["allow_opposite_side_overlap"] is True
    assert projected["v44_ny"]["max_open_positions"] == 0
    assert projected["v44_ny"]["max_entries_per_day"] == 0
    assert projected["v44_ny"]["max_lot"] == 20.0
    assert projected["v44_ny"]["rp_max_lot"] == 20.0
    assert projected["v14"]["cell_scale_overrides"]["ambiguous/er_mid/der_pos:sell"] == 0.25
    assert spec.runtime_overrides["v44_ny"]["risk_per_trade_pct"] == 0.5


def test_load_phase3_package_spec_does_not_apply_defended_overrides_to_generic_preset(tmp_path: Path) -> None:
    paper = {
        "candidate_name": DEFENDED_PACKAGE_ID,
        "package_family": "v7_pfdd",
        "status": "paper_candidate_frozen_for_runtime_implementation",
        "base_cell_scales": {
            "T3_ambig_mid_pos_sell": 0.25,
        },
        "overrides": {
            "l1_weekday_disable": ["Monday", "Tuesday"],
            "l1_exit_override": {
                "tp1_r_multiple": 3.25,
                "be_offset_pips": 1.0,
                "tp2_r_multiple": 2.0,
            },
            "defensive_veto": {
                "strategy": "v44_ny",
                "ownership_cell": "ambiguous/er_low/der_neg",
            },
        },
        "strict_policy": {
            "name": "native_v44_hedging_like",
            "allow_internal_overlap": True,
            "allow_opposite_side_overlap": True,
            "max_open_offensive": None,
            "max_entries_per_day": None,
            "max_lot_per_trade": 20.0,
        },
    }
    runtime_cfg = {"v44_ny": {"risk_per_trade_pct": 0.5}}
    paper_path = tmp_path / "paper_candidate_v7_defended.json"
    runtime_path = tmp_path / "phase3_integrated_sizing_config.json"
    paper_path.write_text(json.dumps(paper), encoding="utf-8")
    runtime_path.write_text(json.dumps(runtime_cfg), encoding="utf-8")

    spec = load_phase3_package_spec(
        preset_id="phase3_integrated_usd_jpy",
        paper_candidate_path=paper_path,
        runtime_config_path=runtime_path,
    )

    assert uses_defended_phase3_package("phase3_integrated_usd_jpy") is False
    assert spec.frozen_modifiers == []
    assert spec.strict_policy == {}
    assert spec.base_cell_scales == {}
    assert spec.runtime_overrides["v44_ny"] == {"risk_per_trade_pct": 0.5}
    assert "london_v2" not in spec.runtime_overrides or "d_suppress_weekdays" not in spec.runtime_overrides.get("london_v2", {})
