from __future__ import annotations

from types import SimpleNamespace

import core.phase3_integrated_engine as phase3_engine
import core.phase3_shared_engine as shared_engine
import core.phase3_additive_runtime as additive_runtime
from core.phase3_shared_engine import compare_phase3_envelopes, normalize_phase3_decision_envelope


def test_normalize_phase3_decision_envelope_extracts_family_and_exit_policy() -> None:
    exec_result = {
        "decision": SimpleNamespace(attempted=True, placed=False, reason="blocked | blocks=phase3_frozen_l1_weekday", side="buy"),
        "strategy_tag": "phase3:london_v2_d@ambiguous/er_low/der_neg",
        "units": 25000,
        "entry_price": 150.1,
        "sl_price": 149.9,
        "tp1_price": 150.75,
    }
    env = normalize_phase3_decision_envelope(
        exec_result=exec_result,
        policy=SimpleNamespace(id="phase3_integrated_v7_defended"),
        preset_id="phase3_integrated_v7_defended",
        sizing_config={"london_v2": {"d_tp1_r": 3.25, "d_be_offset_pips": 1.0, "d_tp2_r": 2.0}},
        ownership_audit={"ownership_cell": "ambiguous/er_low/der_neg"},
    )
    assert env.strategy_family == "london_v2_d"
    assert env.session == "london"
    assert env.ownership_cell == "ambiguous/er_low/der_neg"
    assert env.blocking_filter_ids == ["phase3_frozen_l1_weekday"]
    assert env.exit_policy is not None
    assert env.exit_policy.tp1_r == 3.25


def test_compare_phase3_envelopes_reports_field_drift() -> None:
    left = {
        "session": "london",
        "strategy_tag": "phase3:london_v2_d@cell",
        "strategy_family": "london_v2_d",
        "ownership_cell": "cell",
        "attempted": True,
        "placed": False,
        "blocking_filter_ids": ["phase3_frozen_l1_weekday"],
        "size_units": 25000,
        "entry_price": 150.1,
        "sl_price": 149.9,
        "tp1_price": 150.75,
        "exit_policy": {"label": "L1", "tp1_r": 3.25, "be_offset_pips": 1.0, "tp2_r": 2.0},
    }
    right = dict(left)
    right["ownership_cell"] = "other_cell"
    diff = compare_phase3_envelopes(left, right)
    assert diff.matches is False
    assert any("ownership_cell" in row for row in diff.mismatches)


def test_compare_phase3_envelopes_reports_reason_and_attribution_drift() -> None:
    left = {
        "session": "ny",
        "strategy_tag": "phase3:v44_ny:strong@cell",
        "strategy_family": "v44_ny",
        "ownership_cell": "cell",
        "attempted": True,
        "placed": False,
        "blocking_filter_ids": [],
        "size_units": 100000,
        "entry_price": 150.1,
        "sl_price": 149.9,
        "tp1_price": 150.4,
        "reason": "v44: ATR percentile block",
        "attribution": {"ownership_audit": {"ownership_cell": "cell"}},
        "exit_policy": {"label": "V44 session exit"},
    }
    right = dict(left)
    right["reason"] = "v44: directional conditions not met"
    right["attribution"] = {"ownership_audit": {"ownership_cell": "other_cell"}}

    diff = compare_phase3_envelopes(left, right)

    assert diff.matches is False
    assert any("reason:" in row for row in diff.mismatches)
    assert any("attribution differs" in row for row in diff.mismatches)


def test_evaluate_phase3_bar_precomputes_and_forwards_ownership_audit(monkeypatch) -> None:
    fake_audit = {
        "schema": "phase3_ownership_audit_v1",
        "ownership_cell": "ambiguous/er_low/der_neg",
        "regime_label": "ambiguous",
    }

    monkeypatch.setattr(shared_engine, "compute_phase3_ownership_audit_for_data", lambda data_by_tf, pip_size: fake_audit)
    monkeypatch.setattr(phase3_engine, "load_phase3_sizing_config", lambda: {})
    monkeypatch.setattr(shared_engine, "build_phase3_overlay_state", lambda sizing_config: {"v14_cell_scale_overrides": {}})

    def _fake_exec(**kwargs):
        assert kwargs["ownership_audit"] == fake_audit
        assert kwargs["overlay_state"] == {"v14_cell_scale_overrides": {}}
        return {
            "decision": SimpleNamespace(attempted=False, placed=False, reason="phase3: no active session", side=None),
            "strategy_tag": "phase3:v14_mean_reversion",
        }

    monkeypatch.setattr(phase3_engine, "execute_phase3_integrated_policy_demo_only", _fake_exec)

    result = shared_engine.evaluate_phase3_bar(
        adapter=shared_engine.ReplayAdapter(),
        profile=SimpleNamespace(symbol="USDJPY", pip_size=0.01, active_preset_name="phase3_integrated_v7_defended"),
        log_dir=None,
        policy=SimpleNamespace(id="phase3_integrated_v7_defended"),
        context={},
        data_by_tf={},
        tick=SimpleNamespace(bid=150.0, ask=150.01),
        mode="ARMED_AUTO_DEMO",
        phase3_state={},
        preset_id="phase3_integrated_v7_defended",
    )

    assert result["phase3_ownership_audit"] == fake_audit
    assert result["phase3_overlay_state"] == {"v14_cell_scale_overrides": {}}
    assert result["decision_envelope"]["ownership_cell"] == "ambiguous/er_low/der_neg"


def test_evaluate_phase3_bar_uses_defended_path_when_policy_is_defended(monkeypatch) -> None:
    monkeypatch.setattr(shared_engine, "compute_phase3_ownership_audit_for_data", lambda data_by_tf, pip_size: {"ownership_cell": "x"})
    monkeypatch.setattr(shared_engine, "build_phase3_overlay_state", lambda sizing_config: {})
    monkeypatch.setattr(phase3_engine, "load_phase3_sizing_config", lambda preset_id=None: {"v44_ny": {"max_entries_per_day": 7}})
    additive_called = {"ok": False}

    def _fake_additive(**kwargs):
        additive_called["ok"] = True
        return {
            "decision": SimpleNamespace(attempted=False, placed=False, reason="additive route", side=None),
            "strategy_tag": None,
        }

    def _fake_classic(**kwargs):
        raise AssertionError("classic session router should not be used for defended policy id")

    monkeypatch.setattr(additive_runtime, "execute_phase3_defended_additive_policy", _fake_additive)
    monkeypatch.setattr(phase3_engine, "execute_phase3_integrated_policy_demo_only", _fake_classic)

    result = shared_engine.evaluate_phase3_bar(
        adapter=shared_engine.ReplayAdapter(),
        profile=SimpleNamespace(symbol="USDJPY", pip_size=0.01, active_preset_name="newera8"),
        log_dir=None,
        policy=SimpleNamespace(id="phase3_integrated_v7_defended"),
        context={},
        data_by_tf={},
        tick=SimpleNamespace(bid=150.0, ask=150.01),
        mode="ARMED_AUTO_DEMO",
        phase3_state={},
        preset_id="newera8",
    )

    assert additive_called["ok"] is True
    assert result["decision"].reason == "additive route"
    assert result["decision_envelope"]["preset_id"] == "phase3_integrated_v7_defended"
