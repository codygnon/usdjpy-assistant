from __future__ import annotations

from core.phase3_additive_contract import (
    classify_defended_slice_label,
    defended_slice_source,
)


def test_classify_defended_slice_label_identifies_offensive_t3() -> None:
    label = classify_defended_slice_label(
        strategy_family="v14",
        side="sell",
        ownership_cell="ambiguous/er_mid/der_pos",
        strategy_tag="phase3:v14_mean_reversion@ambiguous/er_mid/der_pos",
        reason="tokyo sell",
        active_slice_scales={"T3_ambig_mid_pos_sell": 0.25},
    )

    assert label == "T3_ambig_mid_pos_sell"
    assert defended_slice_source(label) == "offensive"


def test_classify_defended_slice_label_identifies_baseline_v44_cell() -> None:
    label = classify_defended_slice_label(
        strategy_family="v44_ny",
        side="buy",
        ownership_cell="ambiguous/er_high/der_pos",
        strategy_tag="phase3:v44_ny:strong@ambiguous/er_high/der_pos",
        reason="v44 strong buy",
        active_slice_scales={"O0_buy_strong": 1.0, "T3_ambig_mid_pos_sell": 0.25},
    )

    assert label == "O0_buy_strong"
    assert defended_slice_source(label) == "baseline"
