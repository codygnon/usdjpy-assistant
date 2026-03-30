"""
Frozen rule and helpers for the first offensive Setup D pilot candidate.

This module is intentionally narrow. It exists to keep the current best
offensive slice stable while we move from research to a shadow/paper pilot.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from scripts import backtest_v2_multisetup_london as london_v2_engine


@dataclass(frozen=True)
class OffensiveSetupDPilotRule:
    name: str
    strategy: str
    setup_type: str
    direction: str
    ownership_cell: str
    required_session: str
    min_minutes_after_london_open: int
    max_minutes_after_london_open: int
    notes: str


def default_offensive_setupd_pilot_rule() -> OffensiveSetupDPilotRule:
    return OffensiveSetupDPilotRule(
        name="setupd_whitespace_mr_low_der_neg_first30",
        strategy="london_v2",
        setup_type="D",
        direction="long",
        ownership_cell="mean_reversion/er_low/der_neg",
        required_session="london",
        min_minutes_after_london_open=15,
        max_minutes_after_london_open=45,
        notes=(
            "First offensive pilot slice: long-only London Setup D in the "
            "mean_reversion/er_low/der_neg whitespace / near-whitespace cell, "
            "restricted to the first 30 minutes of the Setup D window."
        ),
    )


def london_minutes_since_open(ts: pd.Timestamp) -> float:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    day = ts.floor("D")
    london_open_hour = london_v2_engine.uk_london_open_utc(day)
    london_open = day + pd.Timedelta(hours=london_open_hour)
    return (ts - london_open).total_seconds() / 60.0


def pilot_window_matches_timestamp(ts: pd.Timestamp, rule: OffensiveSetupDPilotRule) -> bool:
    mins = london_minutes_since_open(ts)
    return rule.min_minutes_after_london_open <= mins <= rule.max_minutes_after_london_open


def london_session_window_matches_timestamp(ts: pd.Timestamp) -> bool:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    day = ts.floor("D")
    london_open_hour = london_v2_engine.uk_london_open_utc(day)
    ny_open_hour = london_v2_engine.us_ny_open_utc(day)
    london_open = day + pd.Timedelta(hours=london_open_hour)
    ny_open = day + pd.Timedelta(hours=ny_open_hour)
    return london_open <= ts < ny_open


def pilot_bar_matches_scope(
    *,
    ts: pd.Timestamp,
    session: str | None,
    ownership_cell: str,
    rule: OffensiveSetupDPilotRule,
) -> bool:
    return (
        session == rule.required_session
        and ownership_cell == rule.ownership_cell
        and pilot_window_matches_timestamp(ts, rule)
    )


def pilot_candidate_matches_rule(
    *,
    candidate: dict[str, Any] | None,
    rule: OffensiveSetupDPilotRule,
) -> bool:
    if not candidate:
        return False
    raw = dict(candidate.get("raw") or {})
    side = str(candidate.get("side") or "").lower()
    trigger_type = str(candidate.get("trigger_type") or "")
    setup_type = str(raw.get("setup_type") or "")
    direction = str(raw.get("direction") or "").lower()
    bar_time = candidate.get("bar_time")
    if not bar_time:
        return False
    ts = pd.Timestamp(bar_time)
    return (
        trigger_type == f"setup_{rule.setup_type}"
        and setup_type == rule.setup_type
        and side == "buy"
        and direction == rule.direction
        and pilot_window_matches_timestamp(ts, rule)
    )
