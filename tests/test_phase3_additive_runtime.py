from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from core.phase3_additive_runtime import (
    Phase3AdditiveCandidate,
    _margin_state,
    _split_candidate_intents,
    execute_phase3_defended_additive_policy,
)


class _Adapter:
    def __init__(self):
        self.orders = []

    def is_demo(self):
        return True

    def get_account_info(self):
        return SimpleNamespace(balance=100000.0, equity=100000.0, margin_used=0.0)

    def place_order(self, *, symbol, side, lots, stop_price, target_price, comment):
        self.orders.append(
            {
                "symbol": symbol,
                "side": side,
                "lots": lots,
                "stop_price": stop_price,
                "target_price": target_price,
                "comment": comment,
            }
        )
        return SimpleNamespace(order_id=len(self.orders), deal_id=len(self.orders), order_retcode=0, fill_price=None)

    def get_position_id_from_order(self, order_id: int):
        return int(order_id)

    def get_position_id_from_deal(self, deal_id: int):
        return int(deal_id)


class _Store:
    def list_open_trades(self, profile_name: str):
        return []


class _StoreWithOpen:
    def __init__(self, rows):
        self._rows = rows

    def list_open_trades(self, profile_name: str):
        return list(self._rows)


def test_defended_additive_runtime_activates_for_any_preset_when_contract_exists():
    adapter = _Adapter()
    profile = SimpleNamespace(active_preset_name="phase3_integrated_usd_jpy", profile_name="demo", name="demo", symbol="USDJPY", pip_size=0.01)
    policy = SimpleNamespace(id="phase3_integrated_v14")
    result = execute_phase3_defended_additive_policy(
        adapter=adapter,
        profile=profile,
        log_dir=None,
        policy=policy,
        context={},
        data_by_tf={},
        tick=SimpleNamespace(bid=159.5, ask=159.51),
        mode="ARMED_AUTO_DEMO",
        phase3_state={},
        store=_Store(),
        sizing_config={},
        ownership_audit={},
        overlay_state={},
    )
    assert result["decision"].placed is False
    assert "no defended candidates" in result["decision"].reason or "generic phase3 preset disabled" in result["decision"].reason


def test_margin_state_honors_max_lot_cap():
    spec = SimpleNamespace(strict_policy={"margin_leverage": 33.3, "max_lot_per_trade": 1.0})
    state = _margin_state(_Adapter(), spec, 2.0)
    assert state.blocked is True
    assert state.reason == "lot_cap>1.00"


def test_defended_additive_runtime_places_accepted_candidate(monkeypatch):
    import core.phase3_integrated_engine as engine

    def _tokyo(*, adapter, profile, policy, data_by_tf, tick, mode, phase3_state, store=None, sizing_config=None, now_utc=None, ownership_audit=None, overlay_state=None, base_state_updates=None):
        dec = engine.ExecutionDecision(attempted=True, placed=True, reason="phase3:v14 buy", side="buy", order_retcode=0, order_id=1, deal_id=1)
        return {
            "decision": dec,
            "strategy_tag": "phase3:v14_mean_reversion@ambiguous/er_mid/der_pos",
            "sl_price": 159.4,
            "tp1_price": 159.7,
            "entry_price": 159.55,
            "units": 250000,
            "phase3_state_updates": {"session_tokyo_2026-03-30": {"trade_count": 1}},
        }

    def _no_trade(**kwargs):
        return {
            "decision": engine.ExecutionDecision(attempted=False, placed=False, reason="none", side=None),
            "strategy_tag": None,
            "phase3_state_updates": {},
        }

    monkeypatch.setattr(engine, "execute_tokyo_v14_entry", _tokyo)
    monkeypatch.setattr(engine, "execute_london_v2_entry", _no_trade)
    monkeypatch.setattr(engine, "execute_v44_ny_entry", _no_trade)

    adapter = _Adapter()
    profile = SimpleNamespace(active_preset_name="phase3_integrated_v7_defended", profile_name="demo", name="demo", symbol="USDJPY", pip_size=0.01)
    policy = SimpleNamespace(id="phase3_integrated_v7_defended")
    result = execute_phase3_defended_additive_policy(
        adapter=adapter,
        profile=profile,
        log_dir=None,
        policy=policy,
        context={},
        data_by_tf={},
        tick=SimpleNamespace(bid=159.54, ask=159.55),
        mode="ARMED_AUTO_DEMO",
        phase3_state={},
        store=_Store(),
        sizing_config={},
        ownership_audit={"ownership_cell": "ambiguous/er_mid/der_pos"},
        overlay_state={},
    )
    assert result["decision"].placed is True
    assert len(result["placements"]) == 1
    assert adapter.orders[0]["side"] == "buy"
    assert result["phase3_additive_truth"]["accepted_count"] == 1


def test_split_candidate_intents_distinguishes_baseline_and_offensive() -> None:
    spec = SimpleNamespace(base_cell_scales={"T3_ambig_mid_pos_sell": 0.25})
    candidates = [
        Phase3AdditiveCandidate(
            identity="tokyo-sell",
            intent_source="reconstructed_family_candidate",
            slice_id=None,
            strategy_tag="phase3:v14_mean_reversion@ambiguous/er_mid/der_pos",
            strategy_family="v14",
            side="sell",
            ownership_cell="ambiguous/er_mid/der_pos",
            entry_time_utc="2026-03-30T16:00:00+00:00",
            units=25000,
            lots=0.25,
            entry_price=159.55,
            sl_price=159.70,
            tp1_price=159.40,
            reason="tokyo sell",
        ),
        Phase3AdditiveCandidate(
            identity="ny-buy",
            intent_source="reconstructed_family_candidate",
            slice_id=None,
            strategy_tag="phase3:v44_ny:normal@breakout/er_mid/der_pos",
            strategy_family="v44_ny",
            side="buy",
            ownership_cell="breakout/er_mid/der_pos",
            entry_time_utc="2026-03-30T14:00:00+00:00",
            units=100000,
            lots=1.0,
            entry_price=159.55,
            sl_price=159.40,
            tp1_price=159.80,
            reason="ny buy",
        ),
    ]

    baseline_intents, offensive_intents, baseline_candidates, offensive_candidates = _split_candidate_intents(candidates, spec)

    assert len(offensive_intents) == 1
    assert offensive_intents[0].slice_id == "T3_ambig_mid_pos_sell"
    assert len(offensive_candidates) == 1
    assert offensive_candidates[0].intent_source == "offensive"
    assert len(baseline_intents) == 1
    assert baseline_intents[0].strategy_tag == "phase3:v44_ny:normal@breakout/er_mid/der_pos"
    assert len(baseline_candidates) == 1
    assert baseline_candidates[0].intent_source == "baseline"


def test_baseline_candidate_respects_conflict_overlap(monkeypatch):
    import core.phase3_additive_runtime as ar
    import core.phase3_integrated_engine as engine

    baseline = Phase3AdditiveCandidate(
        identity="ny-baseline",
        intent_source="baseline",
        slice_id="baseline:v44_ny:sell@",
        strategy_tag="phase3:v44_ny:strong@momentum/er_low/der_pos",
        strategy_family="v44_ny",
        side="sell",
        ownership_cell="momentum/er_low/der_pos",
        entry_time_utc="2026-03-30T16:08:00+00:00",
        units=100000,
        lots=1.0,
        entry_price=159.44,
        sl_price=159.53,
        tp1_price=159.26,
        reason="baseline",
    )

    def _fake_eval(**kwargs):
        return [baseline], []

    monkeypatch.setattr(ar, "_evaluate_family_candidates", _fake_eval)
    monkeypatch.setattr(
        ar,
        "apply_variant_k_baseline_admission",
        lambda candidates, data_by_tf: SimpleNamespace(accepted=list(candidates), rejected=[], adjustments={}, diagnostics=[]),
    )
    monkeypatch.setattr(
        ar,
        "load_phase3_package_spec",
        lambda preset_id: SimpleNamespace(
            package_id="pkg",
            strict_policy={"allow_internal_overlap": False, "allow_opposite_side_overlap": True},
            base_cell_scales={},
        ),
    )

    profile = SimpleNamespace(active_preset_name="phase3_integrated_v7_defended", profile_name="demo", name="demo", symbol="USDJPY", pip_size=0.01)
    policy = SimpleNamespace(id="phase3_integrated_v7_defended")
    store = _StoreWithOpen([{"entry_type": "phase3:v44_ny:normal@momentum/er_low/der_pos", "side": "sell"}])
    result = execute_phase3_defended_additive_policy(
        adapter=_Adapter(),
        profile=profile,
        log_dir=None,
        policy=policy,
        context={},
        data_by_tf={},
        tick=SimpleNamespace(bid=159.44, ask=159.45),
        mode="ARMED_AUTO_DEMO",
        phase3_state={},
        store=store,
        sizing_config={},
        ownership_audit={},
        overlay_state={},
    )
    assert result["phase3_additive_truth"]["accepted_count"] == 0
    assert len(result["phase3_additive_rejected"]) == 1
    assert result["phase3_additive_rejected"][0]["reason"] == "internal_overlap_blocked"


def test_max_entries_day_counts_baseline_before_offensive(monkeypatch):
    import core.phase3_additive_runtime as ar

    baseline = Phase3AdditiveCandidate(
        identity="ny-baseline",
        intent_source="baseline",
        slice_id="baseline:v44_ny:sell@",
        strategy_tag="phase3:v44_ny:strong@momentum/er_low/der_pos",
        strategy_family="v44_ny",
        side="sell",
        ownership_cell="momentum/er_low/der_pos",
        entry_time_utc="2026-03-30T16:08:00+00:00",
        units=100000,
        lots=1.0,
        entry_price=159.44,
        sl_price=159.53,
        tp1_price=159.26,
        reason="baseline",
    )
    offensive = replace(
        baseline,
        identity="off-1",
        intent_source="offensive",
        strategy_tag="phase3:london_v2_d@momentum/er_low/der_pos",
        strategy_family="london_v2",
        slice_id="L1_mom_low_pos_buy",
        side="buy",
    )

    def _fake_eval(**kwargs):
        return [baseline, offensive], []

    monkeypatch.setattr(ar, "_evaluate_family_candidates", _fake_eval)
    monkeypatch.setattr(
        ar,
        "apply_variant_k_baseline_admission",
        lambda candidates, data_by_tf: SimpleNamespace(accepted=list(candidates), rejected=[], adjustments={}, diagnostics=[]),
    )
    monkeypatch.setattr(
        ar,
        "load_phase3_package_spec",
        lambda preset_id: SimpleNamespace(
            package_id="pkg",
            strict_policy={"max_entries_per_day": 1, "allow_internal_overlap": True, "allow_opposite_side_overlap": True},
            base_cell_scales={},
        ),
    )

    profile = SimpleNamespace(active_preset_name="phase3_integrated_v7_defended", profile_name="demo", name="demo", symbol="USDJPY", pip_size=0.01)
    policy = SimpleNamespace(id="phase3_integrated_v7_defended")
    result = execute_phase3_defended_additive_policy(
        adapter=_Adapter(),
        profile=profile,
        log_dir=None,
        policy=policy,
        context={},
        data_by_tf={},
        tick=SimpleNamespace(bid=159.44, ask=159.45),
        mode="ARMED_AUTO_DEMO",
        phase3_state={},
        store=_Store(),
        sizing_config={},
        ownership_audit={},
        overlay_state={},
    )
    assert result["phase3_additive_truth"]["accepted_count"] == 1
    reasons = [row.get("reason") for row in result["phase3_additive_rejected"]]
    assert "max_entries_day_1/1" in reasons
