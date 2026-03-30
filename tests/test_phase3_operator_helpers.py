from __future__ import annotations

import json
from pathlib import Path

from core.phase3_operator import (
    build_phase3_acceptance_payload,
    build_phase3_defensive_monitor_payload,
    build_phase3_provenance_payload,
    parse_phase3_blocking_filter_ids,
    parse_phase3_strategy_tag,
)


def test_parse_phase3_strategy_tag_variants() -> None:
    v14 = parse_phase3_strategy_tag("phase3:v14_mean_reversion@ambiguous/er_mid/der_pos")
    assert v14["is_phase3"] is True
    assert v14["session"] == "tokyo"
    assert v14["strategy_family"] == "v14"
    assert v14["ownership_cell"] == "ambiguous/er_mid/der_pos"
    assert v14["has_cell_attribution"] is True

    arb = parse_phase3_strategy_tag("phase3:london_v2_arb")
    assert arb["session"] == "london"
    assert arb["strategy_family"] == "london_v2_arb"
    assert arb["strategy_variant"] == "arb"

    l1 = parse_phase3_strategy_tag("phase3:london_v2_d@ambiguous/er_low/der_neg")
    assert l1["session"] == "london"
    assert l1["strategy_family"] == "london_v2_d"
    assert l1["strategy_variant"] == "d"
    assert l1["ownership_cell"] == "ambiguous/er_low/der_neg"

    v44 = parse_phase3_strategy_tag("phase3:v44_ny:strong@ambiguous/er_low/der_neg")
    assert v44["session"] == "ny"
    assert v44["strategy_family"] == "v44_ny"
    assert v44["strategy_variant"] == "strong"
    assert v44["strength"] == "strong"


def test_parse_phase3_blocking_filter_ids() -> None:
    reason = "blocked: veto | blocks=phase3_frozen_defensive_veto,london_setup_d"
    assert parse_phase3_blocking_filter_ids(reason) == ["phase3_frozen_defensive_veto", "london_setup_d"]
    assert parse_phase3_blocking_filter_ids("plain reason") == []


def test_build_phase3_acceptance_payload_handles_present_and_missing(tmp_path: Path) -> None:
    missing = build_phase3_acceptance_payload(tmp_path / "missing.json")
    assert missing["available"] is False
    assert missing["rules"] == []

    artifact = {
        "package_under_test": "pkg",
        "verdict": "CONDITIONALLY_READY_AWAITING_OBSERVED_FIRES",
        "verdict_note": "waiting",
        "observed_summary": {"OBSERVED_count": 1, "IMPLEMENTED_AND_INSTRUMENTED_AWAITING_OBSERVED_FIRE_count": 4, "BROKEN_count": 0},
        "rules": [
            {
                "id": "l1_exit_override",
                "requirement": "tp1r=3.25",
                "observed_status": "OBSERVED",
                "evidence_pointer": "memo#1",
            }
        ],
        "immediate_next_action": "run paper",
    }
    path = tmp_path / "paper_acceptance_validation.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    payload = build_phase3_acceptance_payload(path)
    assert payload["available"] is True
    assert payload["rules"][0]["label"] == "L1 exit override"
    assert payload["rules"][0]["status"] == "OBSERVED"


def test_build_phase3_defensive_monitor_payload_handles_missing_pain_report(tmp_path: Path) -> None:
    guardrail = {
        "frozen_package_id": "pkg",
        "defensive_veto_pocket": {"strategy": "v44_ny", "ownership_cell": "ambiguous/er_low/der_neg"},
        "research_baselines_for_sanity_checks": {
            "blocked_trades_reference": {
                "500k": {"blocked_count": 12, "blocked_net_usd": 1234.5},
                "1000k": {"blocked_count": 20, "blocked_net_usd": 2345.6},
            }
        },
        "rollback_triggers": [{"id": "rollback_1"}],
        "monitoring_commands": {"paper_guardrail_report": "python3 scripts/paper_monitor_defensive_veto_pocket.py"},
    }
    guardrail_path = tmp_path / "defensive_paper_guardrail_profile.json"
    guardrail_path.write_text(json.dumps(guardrail), encoding="utf-8")

    payload = build_phase3_defensive_monitor_payload(guardrail_path, tmp_path / "missing.json")
    assert payload["available"] is True
    assert payload["strategy"] == "v44_ny"
    assert payload["ownership_cell"] == "ambiguous/er_low/der_neg"
    assert payload["paper_monitor_executed"] is False
    assert payload["rollback_reference"] == {"id": "rollback_1"}


def test_build_phase3_provenance_payload_uses_defended_package_and_modifiers() -> None:
    sizing_cfg = {
        "london_v2": {
            "d_tp1_r": 3.25,
            "d_be_offset_pips": 1.0,
            "d_tp2_r": 2.0,
            "d_suppress_weekdays": ["Monday", "Tuesday"],
        },
        "v44_ny": {"defensive_veto_cells": ["ambiguous/er_low/der_neg"]},
        "v14": {"cell_scale_overrides": {"ambiguous/er_mid/der_pos:sell": 0.25}},
    }
    context_items = [
        {"key": "Active Session", "value": "london"},
        {"key": "Window", "value": "setup_d"},
        {"key": "Strategy Tag", "value": "phase3:london_v2_d@ambiguous/er_low/der_neg"},
        {"key": "Ownership Cell", "value": "ambiguous/er_low/der_neg"},
        {"key": "Regime Label", "value": "ambiguous"},
        {"key": "Defensive Flags", "value": "veto_armed, overlay_active"},
        {"key": "L1 Exit", "value": "TP1 3.25R / BE 1.0 / TP2 2.0"},
    ]
    latest_decision = {
        "timestamp_utc": "2026-03-30T00:00:00+00:00",
        "signal_id": "eval:phase3_integrated:foo:2026-03-30T00:00:00+00:00",
        "attempted": 1,
        "placed": 0,
        "reason": "blocked by weekday | blocks=phase3_frozen_l1_weekday",
    }

    payload = build_phase3_provenance_payload(
        preset_name="phase3_integrated_v7_defended",
        context_items=context_items,
        filters=[{"filter_id": "phase3_frozen_l1_weekday", "is_clear": False, "display_name": "L1 weekday suppression"}],
        latest_decision=latest_decision,
        sizing_cfg=sizing_cfg,
        dashboard_timestamp_utc="2026-03-30T00:00:05+00:00",
        last_block_reason=None,
    )
    assert payload["package_id"].startswith("v7_pfdd__followup__L1_drop_Monday_Tuesday")
    assert payload["strategy_family"] == "london_v2_d"
    assert payload["outcome"] == "blocked"
    assert payload["blocking_filter_ids"] == ["phase3_frozen_l1_weekday"]
    assert len(payload["frozen_modifiers"]) == 3
    assert payload["exit_policy"]["tp1_r"] == 3.25
