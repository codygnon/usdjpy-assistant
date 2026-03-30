from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from core.phase3_session_router import build_phase3_route_plan


def test_build_phase3_route_plan_blocks_until_new_m1_bar() -> None:
    m1 = pd.DataFrame({"time": [pd.Timestamp("2024-10-22T08:30:00Z")]})
    plan = build_phase3_route_plan(
        data_by_tf={"M1": m1},
        effective_cfg={"london_v2": {"active_days_utc": ["Tuesday", "Wednesday"]}},
        phase3_state={},
        is_new_m1=False,
        ownership_audit={},
    )
    assert plan["session"] == "london"
    assert plan["blocked_reason"] == "phase3: waiting for new closed M1 bar"


def test_build_phase3_route_plan_uses_shared_global_standdown() -> None:
    m1 = pd.DataFrame({"time": [pd.Timestamp("2024-10-22T08:30:00Z")]})
    plan = build_phase3_route_plan(
        data_by_tf={"M1": m1},
        effective_cfg={"global": {"pbt_standdown_enabled": True}, "london_v2": {"active_days_utc": ["Tuesday", "Wednesday"]}},
        phase3_state={},
        is_new_m1=True,
        ownership_audit={
            "defensive_global_standdown": True,
            "defensive_global_standdown_reason": "global standdown (post_breakout_trend + ΔER=-0.100)",
        },
    )
    assert plan["blocked_reason"] == "phase3: global standdown (post_breakout_trend + ΔER=-0.100)"
    assert plan["blocked_attempted"] is True
