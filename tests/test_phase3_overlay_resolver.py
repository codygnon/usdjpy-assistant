from __future__ import annotations

from datetime import datetime, timezone

from core.phase3_overlay_resolver import (
    london_setup_d_weekday_block,
    resolve_v14_cell_scale_override,
    v44_defensive_veto_block,
)


def test_london_setup_d_weekday_block_respects_attributed_strategy_tag() -> None:
    blocked, reason = london_setup_d_weekday_block(
        now_utc=datetime(2024, 10, 21, 8, 0, tzinfo=timezone.utc),  # Monday
        strategy_tag="phase3:london_v2_d@ambiguous/er_low/der_neg",
        london_config={"d_suppress_weekdays": ["Monday", "Tuesday"]},
    )
    assert blocked is True
    assert reason == "london_v2_d: L1 weekday suppression (Monday)"


def test_v44_defensive_veto_block_matches_frozen_cell() -> None:
    blocked, reason = v44_defensive_veto_block(
        ownership_cell="ambiguous/er_low/der_neg",
        v44_config={"defensive_veto_cells": ["ambiguous/er_low/der_neg"]},
    )
    assert blocked is True
    assert reason == "v44_ny: defensive veto (ambiguous/er_low/der_neg)"


def test_resolve_v14_cell_scale_override_scales_units_and_reports_key() -> None:
    units, scale, key = resolve_v14_cell_scale_override(
        ownership_cell="ambiguous/er_mid/der_pos",
        side="sell",
        units=40000,
        v14_config={"cell_scale_overrides": {"ambiguous/er_mid/der_pos:sell": 0.25}},
    )
    assert units == 10000
    assert scale == 0.25
    assert key == "ambiguous/er_mid/der_pos:sell"
