from __future__ import annotations

from datetime import datetime
from typing import Any


def build_phase3_overlay_state(sizing_config: dict[str, Any] | None) -> dict[str, Any]:
    cfg = sizing_config if isinstance(sizing_config, dict) else {}
    ldn_cfg = cfg.get("london_v2", {}) if isinstance(cfg.get("london_v2"), dict) else {}
    v44_cfg = cfg.get("v44_ny", {}) if isinstance(cfg.get("v44_ny"), dict) else {}
    v14_cfg = cfg.get("v14", {}) if isinstance(cfg.get("v14"), dict) else {}
    return {
        "london_setup_d_suppressed_weekdays": tuple(str(day) for day in (ldn_cfg.get("d_suppress_weekdays") or [])),
        "v44_defensive_veto_cells": tuple(str(cell) for cell in (v44_cfg.get("defensive_veto_cells") or [])),
        "v14_cell_scale_overrides": dict(v14_cfg.get("cell_scale_overrides") or {}),
    }


def _is_london_setup_d(strategy_tag: str | None) -> bool:
    tag = str(strategy_tag or "")
    if "london_v2_arb" in tag:
        return False
    return tag.endswith("london_v2_d") or "london_v2_d@" in tag


def london_setup_d_weekday_block(
    *,
    now_utc: datetime,
    strategy_tag: str | None,
    london_config: dict[str, Any] | None,
) -> tuple[bool, str | None]:
    cfg = london_config if isinstance(london_config, dict) else {}
    blocked_days = cfg.get("d_suppress_weekdays")
    if not blocked_days or not _is_london_setup_d(strategy_tag):
        return False, None
    day_name = now_utc.strftime("%A")
    if isinstance(blocked_days, (list, tuple, set)) and day_name in blocked_days:
        return True, f"london_v2_d: L1 weekday suppression ({day_name})"
    return False, None


def london_setup_d_weekday_block_from_state(
    *,
    now_utc: datetime,
    strategy_tag: str | None,
    overlay_state: dict[str, Any] | None,
) -> tuple[bool, str | None]:
    state = overlay_state if isinstance(overlay_state, dict) else {}
    blocked_days = state.get("london_setup_d_suppressed_weekdays") or ()
    if not blocked_days or not _is_london_setup_d(strategy_tag):
        return False, None
    day_name = now_utc.strftime("%A")
    if day_name in blocked_days:
        return True, f"london_v2_d: L1 weekday suppression ({day_name})"
    return False, None


def v44_defensive_veto_block(
    *,
    ownership_cell: str | None,
    v44_config: dict[str, Any] | None,
) -> tuple[bool, str | None]:
    cfg = v44_config if isinstance(v44_config, dict) else {}
    veto_cells = cfg.get("defensive_veto_cells")
    if not ownership_cell or not isinstance(veto_cells, (list, tuple, set)):
        return False, None
    if ownership_cell in veto_cells:
        return True, f"v44_ny: defensive veto ({ownership_cell})"
    return False, None


def v44_defensive_veto_block_from_state(
    *,
    ownership_cell: str | None,
    overlay_state: dict[str, Any] | None,
) -> tuple[bool, str | None]:
    state = overlay_state if isinstance(overlay_state, dict) else {}
    veto_cells = state.get("v44_defensive_veto_cells") or ()
    if not ownership_cell or ownership_cell not in veto_cells:
        return False, None
    return True, f"v44_ny: defensive veto ({ownership_cell})"


def resolve_v14_cell_scale_override(
    *,
    ownership_cell: str | None,
    side: str | None,
    units: int,
    v14_config: dict[str, Any] | None,
) -> tuple[int, float, str | None]:
    cfg = v14_config if isinstance(v14_config, dict) else {}
    overrides = cfg.get("cell_scale_overrides")
    if not ownership_cell or not side or not isinstance(overrides, dict):
        return int(units), 1.0, None
    key = f"{ownership_cell}:{side}"
    if key not in overrides:
        return int(units), 1.0, None
    scale = float(overrides[key])
    scaled_units = int(max(0, round(float(units) * scale)))
    return scaled_units, scale, key


def resolve_v14_cell_scale_override_from_state(
    *,
    ownership_cell: str | None,
    side: str | None,
    units: int,
    overlay_state: dict[str, Any] | None,
) -> tuple[int, float, str | None]:
    state = overlay_state if isinstance(overlay_state, dict) else {}
    overrides = state.get("v14_cell_scale_overrides")
    if not ownership_cell or not side or not isinstance(overrides, dict):
        return int(units), 1.0, None
    key = f"{ownership_cell}:{side}"
    if key not in overrides:
        return int(units), 1.0, None
    scale = float(overrides[key])
    scaled_units = int(max(0, round(float(units) * scale)))
    return scaled_units, scale, key
