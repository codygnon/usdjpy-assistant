from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_phase3_strategy_tag(strategy_tag: str | None) -> dict[str, Any]:
    raw = str(strategy_tag or "").strip()
    result: dict[str, Any] = {
        "is_phase3": False,
        "session": None,
        "strategy_family": None,
        "strategy_variant": None,
        "strength": None,
        "ownership_cell": None,
        "has_cell_attribution": False,
        "strategy_tag": raw or None,
    }
    if not raw.startswith("phase3:"):
        return result

    result["is_phase3"] = True
    base, sep, cell = raw.partition("@")
    if sep and cell:
        result["ownership_cell"] = cell
        result["has_cell_attribution"] = True

    body = base[len("phase3:") :]
    parts = [p for p in body.split(":") if p]
    if not parts:
        return result

    head = parts[0]
    if head.startswith("v14"):
        result["session"] = "tokyo"
        result["strategy_family"] = "v14"
        result["strategy_variant"] = head.replace("v14_", "", 1) if "_" in head else head
    elif head == "london_v2_arb":
        result["session"] = "london"
        result["strategy_family"] = "london_v2_arb"
        result["strategy_variant"] = "arb"
    elif head == "london_v2_d":
        result["session"] = "london"
        result["strategy_family"] = "london_v2_d"
        result["strategy_variant"] = "d"
    elif head == "v44_ny":
        result["session"] = "ny"
        result["strategy_family"] = "v44_ny"
        if len(parts) >= 2:
            result["strategy_variant"] = parts[1]
            if parts[1] in {"weak", "normal", "strong", "news"}:
                result["strength"] = parts[1]
    elif head == "spike_fade_v4":
        result["session"] = "ny"
        result["strategy_family"] = "spike_fade_v4"
        result["strategy_variant"] = "family_c_model4"
    else:
        result["strategy_family"] = head

    return result


def parse_phase3_blocking_filter_ids(reason: str | None) -> list[str]:
    text = str(reason or "")
    if "blocks=" not in text:
        return []
    try:
        blocks_part = text.split("blocks=", 1)[1].strip()
        for stop in (" | ", "\t", "\n"):
            if stop in blocks_part:
                blocks_part = blocks_part.split(stop, 1)[0]
        return [b.strip() for b in blocks_part.split(",") if b.strip()]
    except Exception:
        return []


def extract_phase3_reason_text(reason: str | None) -> str:
    text = str(reason or "").strip()
    if " | blocks=" in text:
        return text.split(" | blocks=", 1)[0].strip()
    return text


def infer_phase3_session_from_signal_id(signal_id: str | None, sizing_cfg: dict[str, Any] | None = None) -> str | None:
    raw = str(signal_id or "")
    prefix = "eval:phase3_integrated:"
    if not raw.startswith(prefix):
        return None
    try:
        _, _, rest = raw.partition(prefix)
        _, _, bar_time = rest.partition(":")
        if not bar_time:
            return None
        ts = datetime.fromisoformat(bar_time.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        from core.phase3_integrated_engine import classify_session

        return classify_session(ts, sizing_cfg or {})
    except Exception:
        return None


def enrich_phase3_attribution(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    parsed = parse_phase3_strategy_tag(str(row.get("entry_type") or row.get("strategy_tag") or ""))
    out.update({
        "phase3_session": parsed["session"],
        "phase3_strategy_family": parsed["strategy_family"],
        "phase3_strategy_variant": parsed["strategy_variant"],
        "ownership_cell": parsed["ownership_cell"],
        "has_cell_attribution": parsed["has_cell_attribution"],
    })
    return out


def load_json_artifact(path: Path) -> dict[str, Any] | None:
    try:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def build_phase3_acceptance_payload(path: Path) -> dict[str, Any]:
    artifact = load_json_artifact(path)
    if not artifact:
        return {
            "available": False,
            "generated_at_utc": None,
            "package_under_test": None,
            "verdict": None,
            "verdict_note": "paper_acceptance_validation.json not found",
            "observed_summary": None,
            "rules": [],
            "immediate_next_action": None,
        }

    label_map = {
        "l1_exit_override": "L1 exit override",
        "l1_weekday_suppression": "L1 weekday suppression",
        "defensive_veto": "Defensive veto",
        "slice_attribution": "Slice attribution",
        "t3_v14_scale": "T3 scale override",
    }
    return {
        "available": True,
        "generated_at_utc": artifact.get("observed_fire_closeout_at_utc") or artifact.get("validated_at_utc"),
        "package_under_test": artifact.get("package_under_test"),
        "verdict": artifact.get("verdict"),
        "verdict_note": artifact.get("verdict_note"),
        "observed_summary": artifact.get("observed_summary"),
        "rules": [
            {
                "id": str(rule.get("id") or ""),
                "label": label_map.get(str(rule.get("id") or ""), str(rule.get("id") or "").replace("_", " ").title()),
                "requirement": str(rule.get("requirement") or ""),
                "status": str(rule.get("observed_status") or rule.get("status") or ""),
                "evidence_pointer": rule.get("evidence_pointer"),
            }
            for rule in list(artifact.get("rules") or [])
            if isinstance(rule, dict)
        ],
        "immediate_next_action": artifact.get("immediate_next_action"),
    }


def build_phase3_defensive_monitor_payload(guardrail_path: Path, pain_report_path: Path) -> dict[str, Any]:
    profile = load_json_artifact(guardrail_path)
    pain_report = load_json_artifact(pain_report_path)
    if not profile:
        return {
            "available": False,
            "generated_at_utc": None,
            "frozen_package_id": None,
            "strategy": None,
            "ownership_cell": None,
            "paper_monitor_executed": None,
            "paper_monitor_skip_reason": "defensive_paper_guardrail_profile.json not found",
            "log_path_used": None,
            "guardrail_status": None,
            "pause_recommended": None,
            "rollback_reference": None,
            "next_command_when_log_exists": None,
            "searched_locations": [],
            "research_baseline_blocked_trade_counts": None,
        }

    veto = dict(profile.get("defensive_veto_pocket") or {})
    baselines = (((profile.get("research_baselines_for_sanity_checks") or {}).get("blocked_trades_reference")) or None)
    rollback_triggers = list(profile.get("rollback_triggers") or [])
    return {
        "available": True,
        "generated_at_utc": pain_report.get("generated_at_utc") if pain_report else None,
        "frozen_package_id": profile.get("frozen_package_id"),
        "strategy": veto.get("strategy"),
        "ownership_cell": veto.get("ownership_cell"),
        "paper_monitor_executed": pain_report.get("paper_monitor_executed") if pain_report else False,
        "paper_monitor_skip_reason": pain_report.get("paper_monitor_skip_reason") if pain_report else "defensive_paper_pain_report.json not found",
        "log_path_used": pain_report.get("log_path_used") if pain_report else None,
        "guardrail_status": pain_report.get("guardrail_status") if pain_report else None,
        "pause_recommended": ((pain_report or {}).get("guardrail_status") or {}).get("pause_recommended"),
        "rollback_reference": rollback_triggers[0] if rollback_triggers else None,
        "next_command_when_log_exists": (pain_report or {}).get("next_command_when_log_exists") or ((profile.get("monitoring_commands") or {}).get("paper_guardrail_report")),
        "searched_locations": (pain_report or {}).get("searched_locations") or [],
        "research_baseline_blocked_trade_counts": baselines,
    }


def build_phase3_provenance_payload(
    *,
    preset_name: str,
    context_items: list[dict[str, Any]],
    filters: list[dict[str, Any]],
    latest_decision: dict[str, Any] | None,
    sizing_cfg: dict[str, Any] | None,
    dashboard_timestamp_utc: str | None,
    last_block_reason: str | None,
) -> dict[str, Any]:
    sizing_cfg = sizing_cfg or {}
    defended_package_id = "v7_pfdd__followup__L1_drop_Monday_Tuesday__tp1r_3.25__be_1__tp2r_2__defensive_v44_ambiguous_low_derneg"
    context_map = {str(item.get("key") or ""): item for item in context_items if isinstance(item, dict)}
    decision = latest_decision or {}
    strategy_tag = str(context_map.get("Strategy Tag", {}).get("value") or "")
    parsed = parse_phase3_strategy_tag(strategy_tag)
    ldn_cfg = sizing_cfg.get("london_v2", {}) if isinstance(sizing_cfg.get("london_v2"), dict) else {}
    v44_cfg = sizing_cfg.get("v44_ny", {}) if isinstance(sizing_cfg.get("v44_ny"), dict) else {}
    v14_cfg = sizing_cfg.get("v14", {}) if isinstance(sizing_cfg.get("v14"), dict) else {}
    blocked_filters = parse_phase3_blocking_filter_ids(decision.get("reason"))
    frozen_modifiers = []
    if ldn_cfg.get("d_suppress_weekdays"):
        frozen_modifiers.append({
            "id": "l1_weekday_suppression",
            "label": "L1 weekday suppression",
            "value": ", ".join(str(day) for day in ldn_cfg.get("d_suppress_weekdays") or []),
        })
    if v44_cfg.get("defensive_veto_cells"):
        frozen_modifiers.append({
            "id": "defensive_veto",
            "label": "Defensive veto",
            "value": f"v44_ny blocked in {', '.join(str(cell) for cell in v44_cfg.get('defensive_veto_cells') or [])}",
        })
    t3_scale = ((v14_cfg.get("cell_scale_overrides") or {}).get("ambiguous/er_mid/der_pos:sell"))
    if t3_scale is not None:
        frozen_modifiers.append({
            "id": "t3_scale_override",
            "label": "T3 scale override",
            "value": f"ambiguous/er_mid/der_pos:sell = {float(t3_scale):.2f}x",
        })

    defensive_flags = [
        flag.strip()
        for flag in str(context_map.get("Defensive Flags", {}).get("value") or "").split(",")
        if flag.strip()
    ]
    phase3_filters = [flt for flt in filters if isinstance(flt, dict) and (str(flt.get("filter_id") or "").startswith("phase3_") or str(flt.get("filter_id") or "").startswith("ny_") or str(flt.get("filter_id") or "").startswith("london") or str(flt.get("filter_id") or "").startswith("tokyo"))]
    if not last_block_reason:
        for flt in phase3_filters:
            if not bool(flt.get("is_clear", True)):
                last_block_reason = str(flt.get("display_name") or flt.get("block_reason") or "")
                break

    placed = bool(decision.get("placed")) if decision else False
    attempted = bool(decision.get("attempted")) if decision else False
    outcome = "waiting"
    if attempted and placed:
        outcome = "placed"
    elif attempted and not placed:
        outcome = "blocked"
    package_id = defended_package_id if str(preset_name or "").strip().lower() == "phase3_integrated_v7_defended" else (str(preset_name or "").strip() or "phase3_integrated")

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_id": package_id,
        "preset_name": preset_name,
        "session": context_map.get("Active Session", {}).get("value") or infer_phase3_session_from_signal_id(decision.get("signal_id"), sizing_cfg),
        "strategy_tag": strategy_tag or None,
        "strategy_family": parsed.get("strategy_family"),
        "window_label": context_map.get("Window", {}).get("value"),
        "ownership_cell": context_map.get("Ownership Cell", {}).get("value") or context_map.get("NY Ownership Cell", {}).get("value"),
        "regime_label": context_map.get("Regime Label", {}).get("value"),
        "defensive_flags": defensive_flags,
        "attempted": attempted,
        "placed": placed,
        "outcome": outcome,
        "reason": extract_phase3_reason_text(decision.get("reason")),
        "blocking_filter_ids": blocked_filters,
        "last_block_reason": last_block_reason,
        "exit_policy": {
            "label": context_map.get("L1 Exit", {}).get("value") or context_map.get("L1 Exit Policy", {}).get("value") or "Phase 3 runtime exit policy",
            "tp1_r": float(ldn_cfg.get("d_tp1_r", 0.0)),
            "be_offset_pips": float(ldn_cfg.get("d_be_offset_pips", 0.0)),
            "tp2_r": float(ldn_cfg.get("d_tp2_r", 0.0)),
        },
        "frozen_modifiers": frozen_modifiers,
        "data_freshness": {
            "dashboard_timestamp_utc": dashboard_timestamp_utc,
            "decision_timestamp_utc": decision.get("timestamp_utc"),
        },
    }
