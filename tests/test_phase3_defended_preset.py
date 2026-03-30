from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from core.dashboard_builder import build_dashboard_filters
from core.dashboard_reporters import collect_phase3_context
from core.presets import PresetId, apply_preset, list_presets
from core.profile import default_profile_for_name


def test_phase3_defended_preset_is_listed() -> None:
    presets = list_presets()
    ids = {item["id"] for item in presets}
    assert PresetId.PHASE3_INTEGRATED_V7_DEFENDED.value in ids


def test_phase3_defended_preset_preserves_oanda_fields() -> None:
    profile = default_profile_for_name("demo")
    profile = profile.model_copy(update={
        "broker_type": "oanda",
        "oanda_token": "token",
        "oanda_account_id": "acct",
        "oanda_environment": "practice",
    })

    updated = apply_preset(profile, PresetId.PHASE3_INTEGRATED_V7_DEFENDED)

    assert updated.active_preset_name == PresetId.PHASE3_INTEGRATED_V7_DEFENDED.value
    assert updated.broker_type == "oanda"
    assert updated.oanda_token == "token"
    assert updated.oanda_account_id == "acct"
    assert updated.oanda_environment == "practice"
    assert any(getattr(pol, "type", None) == "phase3_integrated" and getattr(pol, "enabled", False) for pol in updated.execution.policies)


def test_phase3_defended_dashboard_filters_are_additive() -> None:
    profile = default_profile_for_name("demo").model_copy(update={"active_preset_name": PresetId.PHASE3_INTEGRATED_V7_DEFENDED.value})
    tick = SimpleNamespace(bid=150.0, ask=150.02)
    filters = build_dashboard_filters(
        profile=profile,
        tick=tick,
        data_by_tf={"M1": pd.DataFrame(columns=["time"]), "M5": pd.DataFrame(), "M15": pd.DataFrame(), "H1": pd.DataFrame()},
        policy=SimpleNamespace(),
        policy_type="phase3_integrated",
        eval_result={},
        phase3_state={},
    )
    names = {f.display_name for f in filters}
    assert "Frozen Package" in names
    assert "L1 Weekday Rule" in names
    assert "L1 Exit Policy" in names
    assert "Defensive Veto" in names
    assert "T3 Scale" in names


def test_phase3_generic_preset_does_not_show_defended_package_indicators() -> None:
    profile = default_profile_for_name("demo").model_copy(update={"active_preset_name": PresetId.PHASE3_INTEGRATED_USD_JPY.value})
    tick = SimpleNamespace(bid=150.0, ask=150.02)
    filters = build_dashboard_filters(
        profile=profile,
        tick=tick,
        data_by_tf={"M1": pd.DataFrame(columns=["time"]), "M5": pd.DataFrame(), "M15": pd.DataFrame(), "H1": pd.DataFrame()},
        policy=SimpleNamespace(),
        policy_type="phase3_integrated",
        eval_result={},
        phase3_state={},
    )
    names = {f.display_name for f in filters}
    assert "Frozen Package" not in names


def test_phase3_defended_context_is_visible_only_when_active() -> None:
    tick = SimpleNamespace(bid=150.0, ask=150.02)
    items = collect_phase3_context(SimpleNamespace(), {"M5": pd.DataFrame(), "M15": pd.DataFrame()}, tick, {}, {}, 0.01, active_preset_name=PresetId.PHASE3_INTEGRATED_V7_DEFENDED.value)
    assert any(item.key == "Frozen Package" for item in items)
    other = collect_phase3_context(SimpleNamespace(), {"M5": pd.DataFrame(), "M15": pd.DataFrame()}, tick, {}, {}, 0.01, active_preset_name=PresetId.PHASE3_INTEGRATED_USD_JPY.value)
    assert all(item.key != "Frozen Package" for item in other)


def test_phase3_context_includes_runtime_and_ownership_details() -> None:
    tick = SimpleNamespace(bid=150.0, ask=150.02)
    phase3_state = {
        "effective_phase3_config_hash": "abcdef1234567890",
        "effective_phase3_config_date": "2026-03-30",
        "last_m1_arrival_lag_sec": 2.5,
        "last_m1_retry_count": 1,
        "session_ny_2026-03-30": {
            "ownership_cell": "ambiguous/er_low/der_neg",
            "news_status": "waiting_confirm",
            "news_wait_minutes": 12.0,
            "news_confirm_progress": "1/3",
            "news_trend_side": "buy",
        },
    }
    eval_result = {
        "strategy_tag": "phase3:v44_ny:strong@ambiguous/er_low/der_neg",
        "phase3_ownership_audit": {
            "ownership_cell": "ambiguous/er_low/der_neg",
            "regime_label": "ambiguous",
            "defensive_v44_regime_block": True,
        },
    }
    items = collect_phase3_context(
        SimpleNamespace(),
        {"M5": pd.DataFrame(), "M15": pd.DataFrame(), "H1": pd.DataFrame()},
        tick,
        eval_result,
        phase3_state,
        0.01,
        active_preset_name=PresetId.PHASE3_INTEGRATED_V7_DEFENDED.value,
    )
    keys = {item.key for item in items}
    assert "Config Hash" in keys
    assert "Config Date" in keys
    assert "M1 Arrival Lag" in keys
    assert "M1 Retry Count" in keys
    assert "Strategy Tag" in keys
    assert "Ownership Cell" in keys
    assert "Regime Label" in keys
    assert "Defensive Flags" in keys
