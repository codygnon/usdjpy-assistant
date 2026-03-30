from __future__ import annotations

from dataclasses import dataclass
from typing import Any


PHASE3_DEFENDED_PACKAGE_FAMILY = "v7_pfdd"

PHASE3_DEFENDED_BASELINE_SLICE_LABELS: tuple[str, ...] = (
    "C0_sell_strong",
    "C1_sell_base",
    "C2_sell",
    "C3_buy",
    "C4_sell_base",
    "C5_pbt_sell",
    "C6_pbt_sell",
    "O0_buy_strong",
    "O1_buy_strong",
    "O2_buy_strong",
    "ADJ_meanrev_low_neg_buy",
    "ADJ_ambig_mid_neg_sell",
    "ADJ_mom_high_neg_sell",
    "N1_brkout_low_neg_sell_strong",
    "N2_brkout_low_pos_buy_strong",
    "L2_brkout_mid_neg_buy",
    "T1_ambig_high_pos_buy",
    "T2_brkout_mid_pos_buy",
)

PHASE3_DEFENDED_OFFENSIVE_SLICE_LABELS: tuple[str, ...] = (
    "N3_brkout_low_neg_buy_news",
    "N4_pbt_low_neg_buy_news",
    "L1_mom_low_pos_buy",
    "T3_ambig_mid_pos_sell",
)


@dataclass(frozen=True)
class Phase3SliceIntentRule:
    label: str
    strategy_family: str
    side: str
    ownership_cell: str
    tag_tokens: tuple[str, ...] = ()
    reason_tokens: tuple[str, ...] = ()


PHASE3_DEFENDED_SLICE_INTENT_RULES: tuple[Phase3SliceIntentRule, ...] = (
    Phase3SliceIntentRule("L1_mom_low_pos_buy", "london_v2", "buy", "momentum/er_low/der_pos", tag_tokens=("london_v2_d",)),
    Phase3SliceIntentRule("L2_brkout_mid_neg_buy", "london_v2", "buy", "breakout/er_mid/der_neg", tag_tokens=("london_v2_d",)),
    Phase3SliceIntentRule("T1_ambig_high_pos_buy", "v14", "buy", "ambiguous/er_high/der_pos"),
    Phase3SliceIntentRule("T2_brkout_mid_pos_buy", "v14", "buy", "breakout/er_mid/der_pos"),
    Phase3SliceIntentRule("T3_ambig_mid_pos_sell", "v14", "sell", "ambiguous/er_mid/der_pos"),
    Phase3SliceIntentRule("N1_brkout_low_neg_sell_strong", "v44_ny", "sell", "breakout/er_low/der_neg", tag_tokens=(":strong",)),
    Phase3SliceIntentRule("N2_brkout_low_pos_buy_strong", "v44_ny", "buy", "breakout/er_low/der_pos", tag_tokens=(":strong",)),
    Phase3SliceIntentRule("N3_brkout_low_neg_buy_news", "v44_ny", "buy", "breakout/er_low/der_neg", tag_tokens=(":news",), reason_tokens=("news",)),
    Phase3SliceIntentRule("N4_pbt_low_neg_buy_news", "v44_ny", "buy", "post_breakout_trend/er_low/der_neg", tag_tokens=(":news",), reason_tokens=("news",)),
    Phase3SliceIntentRule("O0_buy_strong", "v44_ny", "buy", "ambiguous/er_high/der_pos", tag_tokens=(":strong",)),
    Phase3SliceIntentRule("O1_buy_strong", "v44_ny", "buy", "ambiguous/er_low/der_pos", tag_tokens=(":strong",)),
    Phase3SliceIntentRule("O2_buy_strong", "v44_ny", "buy", "ambiguous/er_mid/der_pos", tag_tokens=(":strong",)),
    Phase3SliceIntentRule("C1_sell_base", "v44_ny", "sell", "ambiguous/er_low/der_pos"),
    Phase3SliceIntentRule("C2_sell", "v44_ny", "sell", "momentum/er_high/der_pos"),
    Phase3SliceIntentRule("C3_buy", "v44_ny", "buy", "ambiguous/er_mid/der_neg"),
    Phase3SliceIntentRule("C4_sell_base", "v44_ny", "sell", "ambiguous/er_mid/der_pos"),
    Phase3SliceIntentRule("C5_pbt_sell", "v44_ny", "sell", "post_breakout_trend/er_high/der_pos"),
    Phase3SliceIntentRule("C6_pbt_sell", "v44_ny", "sell", "post_breakout_trend/er_mid/der_pos"),
    Phase3SliceIntentRule("ADJ_meanrev_low_neg_buy", "v44_ny", "buy", "mean_reversion/er_low/der_neg"),
    Phase3SliceIntentRule("ADJ_ambig_mid_neg_sell", "v44_ny", "sell", "ambiguous/er_mid/der_neg"),
    Phase3SliceIntentRule("ADJ_mom_high_neg_sell", "v44_ny", "sell", "momentum/er_high/der_neg"),
)


def defended_baseline_slice_labels() -> set[str]:
    return set(PHASE3_DEFENDED_BASELINE_SLICE_LABELS)


def defended_offensive_slice_labels() -> set[str]:
    return set(PHASE3_DEFENDED_OFFENSIVE_SLICE_LABELS)


def classify_defended_slice_label(
    *,
    strategy_family: str,
    side: str,
    ownership_cell: str | None,
    strategy_tag: str | None = None,
    reason: str | None = None,
    active_slice_scales: dict[str, float] | None = None,
) -> str | None:
    allowed = {
        str(label): float(scale)
        for label, scale in dict(active_slice_scales or {}).items()
        if float(scale) > 0.0
    }
    tag = str(strategy_tag or "")
    reason_text = str(reason or "").lower()
    for rule in PHASE3_DEFENDED_SLICE_INTENT_RULES:
        if allowed and rule.label not in allowed:
            continue
        if strategy_family != rule.strategy_family:
            continue
        if str(side or "").lower() != rule.side:
            continue
        if str(ownership_cell or "") != rule.ownership_cell:
            continue
        if rule.tag_tokens and not all(token in tag for token in rule.tag_tokens):
            continue
        if rule.reason_tokens and not all(token in reason_text for token in rule.reason_tokens):
            continue
        return rule.label
    return None


def defended_slice_source(label: str | None) -> str:
    text = str(label or "")
    if text in defended_offensive_slice_labels():
        return "offensive"
    return "baseline"

