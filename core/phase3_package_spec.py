from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


PHASE3_DEFENDED_PRESET_ID = "phase3_integrated_v7_defended"
DEFENDED_PACKAGE_ID = "v7_pfdd__followup__L1_drop_Monday_Tuesday__tp1r_3.25__be_1__tp2r_2__defensive_v44_ambiguous_low_derneg"


@dataclass(frozen=True)
class Phase3FrozenModifier:
    id: str
    value: Any


@dataclass(frozen=True)
class Phase3PackageSpec:
    package_id: str
    package_family: str
    preset_id: str | None
    status: str
    source_artifact: str | None
    base_cell_scales: dict[str, float] = field(default_factory=dict)
    runtime_overrides: dict[str, Any] = field(default_factory=dict)
    frozen_modifiers: list[Phase3FrozenModifier] = field(default_factory=list)
    strict_policy: dict[str, Any] = field(default_factory=dict)


def _root() -> Path:
    return Path(__file__).resolve().parent.parent


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def default_phase3_artifact_paths() -> dict[str, Path]:
    root = _root()
    research = root / "research_out"
    return {
        "paper_candidate": research / "paper_candidate_v7_defended.json",
        "runtime_config": research / "phase3_integrated_sizing_config.json",
    }


def uses_defended_phase3_package(preset_id: str | None) -> bool:
    return str(preset_id or "").strip().lower() == PHASE3_DEFENDED_PRESET_ID


def load_phase3_package_spec(
    *,
    preset_id: str | None = None,
    paper_candidate_path: Path | None = None,
    runtime_config_path: Path | None = None,
) -> Phase3PackageSpec:
    paths = default_phase3_artifact_paths()
    paper_candidate_path = paper_candidate_path or paths["paper_candidate"]
    runtime_config_path = runtime_config_path or paths["runtime_config"]

    candidate = _read_json(paper_candidate_path)
    runtime_cfg = _read_json(runtime_config_path)
    package_id = str(candidate.get("candidate_name") or DEFENDED_PACKAGE_ID)
    package_family = str(candidate.get("package_family") or "phase3_integrated")
    status = str(candidate.get("status") or "runtime_projection")
    defended_active = uses_defended_phase3_package(preset_id)
    base_cell_scales = {
        str(k): float(v)
        for k, v in dict(candidate.get("base_cell_scales") or {}).items()
        if isinstance(v, (int, float))
    }
    overrides = dict(candidate.get("overrides") or {}) if defended_active else {}
    strict_policy = dict(candidate.get("strict_policy") or {}) if defended_active else {}
    defended_base_cell_scales = base_cell_scales if defended_active else {}

    frozen_modifiers: list[Phase3FrozenModifier] = []
    l1_weekdays = list(overrides.get("l1_weekday_disable") or [])
    if l1_weekdays:
        frozen_modifiers.append(Phase3FrozenModifier("l1_weekday_suppression", l1_weekdays))
    l1_exit = dict(overrides.get("l1_exit_override") or {})
    if l1_exit:
        frozen_modifiers.append(Phase3FrozenModifier("l1_exit_override", l1_exit))
    defensive_veto = dict(overrides.get("defensive_veto") or {})
    if defensive_veto:
        frozen_modifiers.append(Phase3FrozenModifier("defensive_veto", defensive_veto))
    if "T3_ambig_mid_pos_sell" in defended_base_cell_scales:
        frozen_modifiers.append(
            Phase3FrozenModifier(
                "t3_cell_scale_override",
                {"ambiguous/er_mid/der_pos:sell": float(defended_base_cell_scales["T3_ambig_mid_pos_sell"])},
            )
        )

    merged_runtime_overrides = project_runtime_config_from_spec(
        Phase3PackageSpec(
            package_id=package_id,
            package_family=package_family,
            preset_id=preset_id,
            status=status,
            source_artifact=str(paper_candidate_path),
            base_cell_scales=defended_base_cell_scales,
            runtime_overrides=overrides,
            frozen_modifiers=frozen_modifiers,
            strict_policy=strict_policy,
        )
    )
    if runtime_cfg:
        merged_runtime_overrides = _deep_merge(merged_runtime_overrides, runtime_cfg)

    return Phase3PackageSpec(
        package_id=package_id if (preset_id or "").strip().lower() == PHASE3_DEFENDED_PRESET_ID else (preset_id or package_family),
        package_family=package_family,
        preset_id=preset_id,
        status=status,
        source_artifact=str(paper_candidate_path) if paper_candidate_path.exists() else None,
        base_cell_scales=defended_base_cell_scales,
        runtime_overrides=merged_runtime_overrides,
        frozen_modifiers=frozen_modifiers,
        strict_policy=strict_policy,
    )


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def project_runtime_config_from_spec(spec: Phase3PackageSpec) -> dict[str, Any]:
    overrides = dict(spec.runtime_overrides or {})
    strict_policy = dict(spec.strict_policy or {})
    projected: dict[str, Any] = {
        "v14": {
            "cell_scale_overrides": {},
        },
        "london_v2": {},
        "v44_ny": {},
    }
    for key in ("v14", "london_v2", "v44_ny"):
        if isinstance(overrides.get(key), dict):
            projected[key] = _deep_merge(projected.get(key, {}), dict(overrides.get(key) or {}))
    l1_weekdays = [str(day).strip() for day in list(overrides.get("l1_weekday_disable") or []) if str(day).strip()]
    if l1_weekdays:
        projected["london_v2"]["d_suppress_weekdays"] = l1_weekdays

    l1_exit = dict(overrides.get("l1_exit_override") or {})
    if l1_exit:
        projected["london_v2"]["d_tp1_r"] = float(l1_exit.get("tp1_r_multiple", 0.0))
        projected["london_v2"]["d_be_offset_pips"] = float(l1_exit.get("be_offset_pips", 0.0))
        projected["london_v2"]["d_tp2_r"] = float(l1_exit.get("tp2_r_multiple", 0.0))

    defensive_veto = dict(overrides.get("defensive_veto") or {})
    if defensive_veto.get("strategy") == "v44_ny" and defensive_veto.get("ownership_cell"):
        projected["v44_ny"]["defensive_veto_cells"] = [str(defensive_veto["ownership_cell"]).strip().lower()]

    if strict_policy:
        projected["v44_ny"]["strict_policy_name"] = str(strict_policy.get("name") or "")
        if "hedging_enabled" in strict_policy:
            projected["v44_ny"]["hedging_enabled"] = bool(strict_policy.get("hedging_enabled"))
        if "allow_internal_overlap" in strict_policy:
            projected["v44_ny"]["allow_internal_overlap"] = bool(strict_policy.get("allow_internal_overlap"))
        if "allow_opposite_side_overlap" in strict_policy:
            projected["v44_ny"]["allow_opposite_side_overlap"] = bool(strict_policy.get("allow_opposite_side_overlap"))
        if "max_open_offensive" in strict_policy:
            max_open = strict_policy.get("max_open_offensive")
            projected["v44_ny"]["max_open_positions"] = None if max_open is None else int(max_open)
        if "max_entries_per_day" in strict_policy:
            max_entries = strict_policy.get("max_entries_per_day")
            projected["v44_ny"]["max_entries_per_day"] = None if max_entries is None else int(max_entries)
        if "margin_model_enabled" in strict_policy:
            projected["v44_ny"]["margin_model_enabled"] = bool(strict_policy.get("margin_model_enabled"))
        if "margin_leverage" in strict_policy and strict_policy.get("margin_leverage") is not None:
            projected["v44_ny"]["margin_leverage"] = float(strict_policy.get("margin_leverage"))
        if "margin_buffer_pct" in strict_policy and strict_policy.get("margin_buffer_pct") is not None:
            projected["v44_ny"]["margin_buffer_pct"] = float(strict_policy.get("margin_buffer_pct"))
        if "max_lot_per_trade" in strict_policy and strict_policy.get("max_lot_per_trade") is not None:
            max_lot = float(strict_policy["max_lot_per_trade"])
            projected["v44_ny"]["max_lot"] = max_lot
            projected["v44_ny"]["rp_max_lot"] = max_lot

    t3_scale = spec.base_cell_scales.get("T3_ambig_mid_pos_sell")
    if t3_scale is not None:
        projected["v14"]["cell_scale_overrides"]["ambiguous/er_mid/der_pos:sell"] = float(t3_scale)

    return projected


def phase3_package_spec_to_dict(spec: Phase3PackageSpec) -> dict[str, Any]:
    payload = asdict(spec)
    payload["frozen_modifiers"] = [asdict(mod) for mod in spec.frozen_modifiers]
    return payload
