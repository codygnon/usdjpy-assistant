from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from core.phase3_session_core import classify_phase3_session, resolve_phase3_bar_time


def build_phase3_route_plan(
    *,
    data_by_tf: dict[str, Any],
    effective_cfg: dict[str, Any],
    phase3_state: dict[str, Any],
    is_new_m1: bool,
    ownership_audit: Optional[dict[str, Any]],
) -> dict[str, Any]:
    now_utc = datetime.now(timezone.utc)
    m1_ref = data_by_tf.get("M1")
    if m1_ref is not None and not getattr(m1_ref, "empty", True):
        now_utc = resolve_phase3_bar_time({"M1": m1_ref}, now_utc)

    session = classify_phase3_session(now_utc, effective_cfg)
    base_state_updates: dict[str, Any] = {}
    meta = effective_cfg.get("_meta", {}) if isinstance(effective_cfg, dict) else {}
    eff_hash = str(meta.get("effective_hash", "") or "")
    today_str_for_cfg = now_utc.date().isoformat()
    if eff_hash:
        prev_hash = str(phase3_state.get("effective_phase3_config_hash", "") or "")
        prev_day = str(phase3_state.get("effective_phase3_config_date", "") or "")
        if prev_hash != eff_hash or prev_day != today_str_for_cfg:
            base_state_updates["effective_phase3_config_hash"] = eff_hash
            base_state_updates["effective_phase3_config_date"] = today_str_for_cfg
            base_state_updates["effective_phase3_config"] = {
                "hash": eff_hash,
                "loaded_at_utc": meta.get("loaded_at_utc"),
                "source_paths": meta.get("source_paths", {}),
                "global": effective_cfg.get("global", {}),
                "v14": effective_cfg.get("v14", {}),
                "london_v2": effective_cfg.get("london_v2", {}),
                "v44_ny": effective_cfg.get("v44_ny", {}),
            }

    blocked_reason = None
    blocked_attempted = False
    if session in {"tokyo", "london", "ny"} and not is_new_m1:
        blocked_reason = "phase3: waiting for new closed M1 bar"

    global_cfg = effective_cfg.get("global", {}) if isinstance(effective_cfg, dict) else {}
    if blocked_reason is None and bool(global_cfg.get("pbt_standdown_enabled", True)):
        if isinstance(ownership_audit, dict) and bool(ownership_audit.get("defensive_global_standdown")):
            blocked_reason = f"phase3: {ownership_audit.get('defensive_global_standdown_reason') or ''}".rstrip()
            blocked_attempted = True

    return {
        "now_utc": now_utc,
        "session": session,
        "base_state_updates": base_state_updates,
        "blocked_reason": blocked_reason,
        "blocked_attempted": blocked_attempted,
    }
