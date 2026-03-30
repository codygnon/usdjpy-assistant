from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from core.phase3_additive_runtime import Phase3AdditiveCandidate
from core.phase3_variant_k_baseline import (
    VariantKBaselineContext,
    apply_variant_k_baseline_admission,
)


def _candidate(*, identity: str, strategy_family: str, side: str, cell: str) -> Phase3AdditiveCandidate:
    return Phase3AdditiveCandidate(
        identity=identity,
        intent_source="baseline",
        slice_id=f"baseline:{strategy_family}",
        strategy_tag=f"phase3:{strategy_family}@{cell}",
        strategy_family=strategy_family,
        side=side,
        ownership_cell=cell,
        entry_time_utc="2026-03-30T14:00:00+00:00",
        units=100000,
        lots=1.0,
        entry_price=159.55,
        sl_price=159.40,
        tp1_price=159.80,
        reason="candidate",
        raw_result={},
    )


def _empty_context() -> VariantKBaselineContext:
    empty = pd.DataFrame({"time": pd.to_datetime([], utc=True)})
    return VariantKBaselineContext(
        m1=empty,
        m1_idx=pd.DatetimeIndex([]),
        m5_basic=empty,
        classified_basic=empty,
        classified_dynamic=empty,
        dyn_time_idx=pd.DatetimeIndex([]),
    )


def test_variant_k_blocks_v44_candidate_when_variant_f_blocks(monkeypatch) -> None:
    import core.phase3_variant_k_baseline as baseline

    monkeypatch.setattr(baseline, "build_variant_k_baseline_context", lambda data_by_tf: _empty_context())
    monkeypatch.setattr(
        baseline.v44_router,
        "_filter_v44_trade",
        lambda *args, **kwargs: SimpleNamespace(
            blocked=True,
            reason="blocked_breakout",
            regime_label="breakout",
            er=0.42,
            decay=0.18,
        ),
    )

    outcome = apply_variant_k_baseline_admission(
        [_candidate(identity="v44-1", strategy_family="v44_ny", side="buy", cell="breakout/er_mid/der_pos")],
        data_by_tf={},
    )

    assert outcome.accepted == []
    assert len(outcome.rejected) == 1
    assert outcome.rejected[0]["variant_k_stage"] == "variant_f_v44_filter"
    assert "blocked_breakout" in outcome.rejected[0]["reason"]


def test_variant_k_marks_london_holdthrough_when_candidate_survives(monkeypatch) -> None:
    import core.phase3_variant_k_baseline as baseline

    times = pd.to_datetime(["2026-03-30T14:00:00+00:00"], utc=True)
    dynamic = pd.DataFrame(
        {
            "time": times,
            "regime_hysteresis": ["momentum"],
            "sf_er": [0.55],
            "sf_delta_er": [0.02],
        }
    )
    ctx = VariantKBaselineContext(
        m1=dynamic[["time"]].copy(),
        m1_idx=pd.DatetimeIndex(times),
        m5_basic=dynamic[["time"]].copy(),
        classified_basic=dynamic.copy(),
        classified_dynamic=dynamic.copy(),
        dyn_time_idx=pd.DatetimeIndex(times),
    )
    monkeypatch.setattr(baseline, "build_variant_k_baseline_context", lambda data_by_tf: ctx)

    outcome = apply_variant_k_baseline_admission(
        [_candidate(identity="ldn-1", strategy_family="london_v2", side="buy", cell="momentum/er_low/der_pos")],
        data_by_tf={},
    )

    assert len(outcome.accepted) == 1
    assert outcome.rejected == []
    assert outcome.adjustments["ldn-1"]["raw_result_updates"]["variant_k_holdthrough_enabled"] is True


def test_variant_k_blocks_global_standdown(monkeypatch) -> None:
    import core.phase3_variant_k_baseline as baseline

    times = pd.to_datetime(["2026-03-30T14:00:00+00:00"], utc=True)
    dynamic = pd.DataFrame(
        {
            "time": times,
            "regime_hysteresis": ["post_breakout_trend"],
            "sf_er": [0.51],
            "sf_delta_er": [-0.05],
        }
    )
    ctx = VariantKBaselineContext(
        m1=dynamic[["time"]].copy(),
        m1_idx=pd.DatetimeIndex(times),
        m5_basic=dynamic[["time"]].copy(),
        classified_basic=dynamic.copy(),
        classified_dynamic=dynamic.copy(),
        dyn_time_idx=pd.DatetimeIndex(times),
    )
    monkeypatch.setattr(baseline, "build_variant_k_baseline_context", lambda data_by_tf: ctx)

    outcome = apply_variant_k_baseline_admission(
        [_candidate(identity="tokyo-1", strategy_family="v14", side="buy", cell="ambiguous/er_mid/der_pos")],
        data_by_tf={},
    )

    assert outcome.accepted == []
    assert len(outcome.rejected) == 1
    assert outcome.rejected[0]["variant_k_stage"] == "variant_i_global_standdown"
