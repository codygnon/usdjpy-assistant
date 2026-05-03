"""Step 4 acceptance: Deterministic Sizing Layer (PHASE9.6).

Coverage:
  - Unit tests for every branch of compute_autonomous_lots
  - Referential transparency: 1000 calls on frozen inputs → identical outputs
  - Import-graph isolation: no cross-imports with core/runner_score
  - Cap-to-4 replay against the V1+V2-survivor corpus (matches PHASE9.6's
    $334.30 "incremental survivor sizing improvement" within $0.01).

The full PHASE9.6 cap-to-4 replay against the FULL Phase 9 admission filter
($6,420.90 total recovery, $832.34 final-USD-after-cap) is asserted in
Step 8 Pass C, where V5/V6 also run on corpus rows.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core import fillmore_v2_sizing as sizing  # noqa: E402
from core.fillmore_v2_sizing import (  # noqa: E402
    DEFAULT_RISK_PCT,
    LOT_STEP,
    LOTS_HARD_MAX,
    LOTS_HARD_MIN,
    PAPER_OR_LIGHT_RISK_PCT,
    PROTECTED_FLOOR_LOTS,
    PROTECTED_RISK_FRACTION,
    SIDE_EXPOSURE_PENALTY_LOTS,
    SizingContext,
    SizingResult,
    cap_historical_lots,
    clamp,
    compute_autonomous_lots,
    rescale_pnl_for_cap,
    round_to_step,
    select_risk_pct,
)


def _ctx(**overrides) -> SizingContext:
    base = dict(
        account_equity=100_000.0,
        sl_pips=8.0,
        pip_value_per_lot=6.6667,
        proposed_side="buy",
        open_lots_buy=0.0,
        open_lots_sell=0.0,
        rolling_20_trade_pnl=0.0,
        rolling_20_lot_weighted_pnl=0.0,
        risk_after_fill_usd=100.0,
        volatility_regime="normal",
        stage="full",
        forward_100_trade_profit_factor=0.0,
        net_pips_100=0.0,
        protected_buy_clr_packet=False,
    )
    base.update(overrides)
    return SizingContext(**base)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_round_to_step_half_up():
    """Round-half-up so behavior is bit-stable (no banker's rounding).

    Note: 2.005 in IEEE754 is actually 2.00499999... so it rounds to 2.00.
    Test only on values exactly representable to avoid asserting against
    float repr quirks; the key invariant is monotonic + bit-stable.
    """
    assert round_to_step(0.99999, 0.01) == 1.00
    assert round_to_step(1.345, 0.01) == 1.35
    assert round_to_step(2.5, 1.0) == 3.0
    # Bit-stable: same input → same output, every call
    assert round_to_step(1.234567, 0.01) == round_to_step(1.234567, 0.01)


def test_round_to_step_rejects_zero():
    with pytest.raises(ValueError):
        round_to_step(1.0, 0.0)


def test_clamp_basic():
    assert clamp(0.5, 1.0, 4.0) == 1.0
    assert clamp(2.5, 1.0, 4.0) == 2.5
    assert clamp(5.0, 1.0, 4.0) == 4.0


# ---------------------------------------------------------------------------
# select_risk_pct (PHASE9.6 ramp)
# ---------------------------------------------------------------------------

def test_select_risk_pct_default_full_stage():
    assert select_risk_pct(stage="full", forward_100_trade_pf=0.0, net_pips_100=0.0) == DEFAULT_RISK_PCT


def test_select_risk_pct_paper_overrides_default():
    assert select_risk_pct(stage="paper", forward_100_trade_pf=0.0, net_pips_100=0.0) == PAPER_OR_LIGHT_RISK_PCT
    assert select_risk_pct(stage="0.1x", forward_100_trade_pf=0.0, net_pips_100=0.0) == PAPER_OR_LIGHT_RISK_PCT


def test_select_risk_pct_ramp_uses_min_does_not_raise_floor():
    """Per blueprint: ramp clause is `min(risk_pct, 0.0050)` — it caps, never raises."""
    rp_paper_with_ramp = select_risk_pct(stage="paper", forward_100_trade_pf=1.5, net_pips_100=200.0)
    assert rp_paper_with_ramp == PAPER_OR_LIGHT_RISK_PCT  # ramp does NOT raise paper to 0.005


def test_select_risk_pct_ramp_caps_full_stage():
    rp = select_risk_pct(stage="full", forward_100_trade_pf=1.5, net_pips_100=200.0)
    assert rp == DEFAULT_RISK_PCT  # 0.0025 < 0.0050 → ramp doesn't change anything


# ---------------------------------------------------------------------------
# compute_autonomous_lots — basic paths
# ---------------------------------------------------------------------------

def test_basic_full_stage_buy():
    """Equity 100k, risk 0.25%, sl 8p, pip $6.6667 → raw 4.687 lots → cap 4."""
    r = compute_autonomous_lots(_ctx())
    assert r.lots == 4.0
    assert r.risk_pct == DEFAULT_RISK_PCT
    assert r.cap_to_max_applied is True


def test_paper_stage_lower_risk_pct():
    r = compute_autonomous_lots(_ctx(stage="paper"))
    # raw = 100k * 0.001 / (8 * 6.6667) = 1.875 → no cap
    assert r.risk_pct == PAPER_OR_LIGHT_RISK_PCT
    assert pytest.approx(r.lots, abs=0.01) == 1.88
    assert r.cap_to_max_applied is False


def test_minimum_lot_clamp_when_raw_below_one():
    """Tiny equity → raw < 1 lot → clamp to 1.0 minimum (memory feedback)."""
    r = compute_autonomous_lots(_ctx(account_equity=20_000.0, stage="paper"))
    # raw = 20_000 * 0.001 / (8*6.6667) = 0.375 → clamp to 1.0
    assert r.lots == LOTS_HARD_MIN
    assert r.cap_to_min_applied is True


def test_degenerate_inputs_return_zero_with_note():
    r = compute_autonomous_lots(_ctx(sl_pips=0.0))
    assert r.lots == 0.0
    assert "degenerate_inputs" in r.notes
    r2 = compute_autonomous_lots(_ctx(pip_value_per_lot=0.0))
    assert r2.lots == 0.0
    r3 = compute_autonomous_lots(_ctx(account_equity=0.0))
    assert r3.lots == 0.0


# ---------------------------------------------------------------------------
# Throttles (50% multipliers)
# ---------------------------------------------------------------------------

def test_rolling_pnl_throttle_fires_when_either_threshold_breached():
    r = compute_autonomous_lots(_ctx(rolling_20_trade_pnl=-60.0))
    assert r.rolling_throttle_applied is True
    r2 = compute_autonomous_lots(_ctx(rolling_20_lot_weighted_pnl=-150.0))
    assert r2.rolling_throttle_applied is True


def test_rolling_pnl_throttle_skipped_when_in_bounds():
    r = compute_autonomous_lots(_ctx(rolling_20_trade_pnl=-49.99, rolling_20_lot_weighted_pnl=-99.99))
    assert r.rolling_throttle_applied is False


def test_side_exposure_throttle_uses_correct_side():
    """Buy proposal looks at open_lots_buy; sell proposal looks at open_lots_sell."""
    r_buy = compute_autonomous_lots(_ctx(proposed_side="buy", open_lots_buy=4.0, open_lots_sell=0.0))
    assert r_buy.side_exposure_throttle_applied is True
    r_sell = compute_autonomous_lots(_ctx(proposed_side="sell", open_lots_buy=4.0, open_lots_sell=0.0))
    # buy exposure shouldn't trigger sell throttle
    assert r_sell.side_exposure_throttle_applied is False
    r_sell2 = compute_autonomous_lots(_ctx(proposed_side="sell", open_lots_sell=SIDE_EXPOSURE_PENALTY_LOTS))
    assert r_sell2.side_exposure_throttle_applied is True


def test_volatility_throttle_only_on_elevated():
    r_norm = compute_autonomous_lots(_ctx(volatility_regime="normal"))
    assert r_norm.volatility_throttle_applied is False
    r_unk = compute_autonomous_lots(_ctx(volatility_regime="unknown"))
    assert r_unk.volatility_throttle_applied is False
    r_elev = compute_autonomous_lots(_ctx(volatility_regime="elevated"))
    assert r_elev.volatility_throttle_applied is True


def test_triple_throttle_clamps_back_up_to_min():
    """4 lots × 0.5³ = 0.5 → re-clamp to LOTS_HARD_MIN=1.0 at the end."""
    r = compute_autonomous_lots(_ctx(
        rolling_20_trade_pnl=-100.0,
        open_lots_buy=4.0,
        volatility_regime="elevated",
    ))
    assert r.rolling_throttle_applied
    assert r.side_exposure_throttle_applied
    assert r.volatility_throttle_applied
    assert r.lots == LOTS_HARD_MIN  # clamped back up from 0.5 → 1.0


# ---------------------------------------------------------------------------
# Protected buy-CLR floor
# ---------------------------------------------------------------------------

def test_protected_floor_raises_below_two_lots():
    """Protected packet + risk-after-fill ≤ 0.5% equity + lots < 2 → floor at 2."""
    r = compute_autonomous_lots(_ctx(
        account_equity=20_000.0,  # raw_lots = 0.375 → clamps to 1.0
        stage="paper",
        risk_after_fill_usd=50.0,  # <= 20_000 * 0.005 = 100
        protected_buy_clr_packet=True,
    ))
    assert r.protected_floor_applied is True
    assert r.lots == PROTECTED_FLOOR_LOTS


def test_protected_floor_does_not_lower_already_high_lots():
    """Already at 4 lots: protected floor must not raise OR lower it."""
    r = compute_autonomous_lots(_ctx(
        account_equity=100_000.0,
        risk_after_fill_usd=200.0,  # under 0.5%
        protected_buy_clr_packet=True,
    ))
    assert r.lots == LOTS_HARD_MAX
    assert r.protected_floor_applied is False  # was already above floor


def test_protected_floor_skipped_when_risk_too_high():
    """Risk-after-fill > 0.5% equity → floor doesn't engage even on protected packet."""
    r = compute_autonomous_lots(_ctx(
        account_equity=20_000.0,
        stage="paper",
        risk_after_fill_usd=200.0,  # > 20_000 * 0.005 = 100
        protected_buy_clr_packet=True,
    ))
    assert r.protected_floor_applied is False


def test_protected_floor_skipped_when_packet_flag_off():
    r = compute_autonomous_lots(_ctx(
        account_equity=20_000.0, stage="paper",
        risk_after_fill_usd=50.0,
        protected_buy_clr_packet=False,
    ))
    assert r.protected_floor_applied is False
    assert r.lots == LOTS_HARD_MIN


# ---------------------------------------------------------------------------
# Hard cap invariant: never > 4 lots regardless of inputs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("equity,sl,risk_pct_target", [
    (1_000_000.0, 4.0, "huge"),  # huge equity, tight stop, default risk
    (10_000_000.0, 8.0, "ramp"),
    (100_000.0, 1.0, "tight"),
])
def test_lots_never_exceed_hard_cap(equity, sl, risk_pct_target):
    r = compute_autonomous_lots(_ctx(
        account_equity=equity, sl_pips=sl, stage="full",
        forward_100_trade_profit_factor=2.0, net_pips_100=500.0,
    ))
    assert r.lots <= LOTS_HARD_MAX, f"hard cap violated for {risk_pct_target}: {r.lots}"


# ---------------------------------------------------------------------------
# Referential transparency (Step 1 audit requirement #5)
# ---------------------------------------------------------------------------

def test_referential_transparency_1000_calls_identical():
    """Pure function: 1000 invocations on a frozen context must return identical
    output every time. No clock, no globals, no I/O.
    """
    ctx = _ctx(
        account_equity=100_000.0, sl_pips=8.0, pip_value_per_lot=6.6667,
        proposed_side="buy", rolling_20_trade_pnl=-75.0,
        open_lots_buy=2.0, open_lots_sell=1.0,
        volatility_regime="elevated", stage="full",
        risk_after_fill_usd=120.0, protected_buy_clr_packet=True,
        forward_100_trade_profit_factor=1.2, net_pips_100=50.0,
        rolling_20_lot_weighted_pnl=-30.0,
    )
    first = compute_autonomous_lots(ctx)
    for _ in range(1000):
        r = compute_autonomous_lots(ctx)
        assert r == first, "compute_autonomous_lots is not referentially transparent"


# ---------------------------------------------------------------------------
# Import-graph isolation
# ---------------------------------------------------------------------------

def test_no_runner_score_imports_in_fillmore_v2_sizing():
    """Check actual import statements, not bare substrings (the module's
    isolation docstring legitimately mentions runner_score by name).
    """
    text = Path(sizing.__file__).read_text()
    forbidden = ("import core.runner_score", "from core.runner_score", "from core import runner_score")
    found = [p for p in forbidden if p in text]
    assert found == [], f"fillmore_v2_sizing must not import runner_score; found: {found}"


def test_runner_score_does_not_import_fillmore_v2_sizing():
    text = (REPO_ROOT / "core" / "runner_score.py").read_text()
    forbidden = (
        "import core.fillmore_v2_sizing",
        "from core.fillmore_v2_sizing",
        "from core import fillmore_v2_sizing",
    )
    found = [p for p in forbidden if p in text]
    assert found == [], f"runner_score must not import fillmore_v2_sizing; found: {found}"


def test_fillmore_v2_sizing_has_no_api_imports():
    """core/ modules should not depend on api/ — keep layering clean."""
    text = Path(sizing.__file__).read_text()
    assert "from api" not in text
    assert "import api" not in text


# ---------------------------------------------------------------------------
# Cap-to-4 historical replay helpers
# ---------------------------------------------------------------------------

def test_cap_historical_lots_basic():
    assert cap_historical_lots(0.5) == LOTS_HARD_MIN
    assert cap_historical_lots(2.5) == 2.5
    assert cap_historical_lots(7.0) == LOTS_HARD_MAX
    assert cap_historical_lots(8.0, cap=6.0) == 6.0


def test_rescale_pnl_for_cap_linear():
    # 2 lots, $100 pnl → cap to 1 lot → $50
    assert rescale_pnl_for_cap(original_pnl=100.0, original_lots=2.0, capped_lots=1.0) == 50.0
    # No change when not capped
    assert rescale_pnl_for_cap(original_pnl=100.0, original_lots=3.0, capped_lots=3.0) == 100.0


def test_rescale_pnl_handles_zero_lots():
    assert rescale_pnl_for_cap(original_pnl=100.0, original_lots=0.0, capped_lots=1.0) == 0.0


# ---------------------------------------------------------------------------
# Corpus-level acceptance: V1+V2-survivor cap-to-4 delta == $334.29
# ---------------------------------------------------------------------------

# Constants pulled from the V1+V2-survivor cap-to-4 calc on the 241-trade
# Phase 7 corpus. Matches PHASE9.6's "Incremental survivor sizing improvement
# after filter: $334.30" within $0.01 (the blueprint rounded to whole cents).
CORPUS_PATH = REPO_ROOT / "research_out" / "autonomous_fillmore_forensic_20260501" / "phase7_interaction_dataset.csv"
EXPECTED_SURVIVOR_COUNT = 131  # 241 - 110 V1+V2 blocked
EXPECTED_ORIGINAL_USD = -1568.6719
EXPECTED_CAPPED_USD = -1234.3770
EXPECTED_CAP_DELTA = 334.2949  # original - capped (positive = recovery)
EXPECTED_ROWS_CAPPED = 33

V1_CLUSTERS = {"momentum_with_caveat_trade", "critical_level_mixed_caveat_trade"}
V2_SESSIONS = {"london/ny overlap", "tokyo/london overlap"}


def _load_corpus() -> list[dict[str, str]]:
    if not CORPUS_PATH.exists():
        return []
    with CORPUS_PATH.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _v1_v2_survivors(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    survivors = []
    for r in rows:
        side = r.get("side", "")
        cluster = r.get("rationale_cluster", "")
        align = r.get("timeframe_alignment_clean", "")
        sess = r.get("session", "")
        v1 = cluster in V1_CLUSTERS and side == "sell"
        v2 = align == "mixed" and sess in V2_SESSIONS
        if not (v1 or v2):
            survivors.append(r)
    return survivors


def _f(row: dict[str, str], key: str) -> float:
    v = row.get(key, "")
    return float(v) if v not in ("", "nan", "NaN", None) else 0.0


def test_cap_to_four_replay_against_v1_v2_survivors():
    """Pin: cap-to-4 on V1+V2 survivors recovers $334.29 — matches PHASE9.6
    'Incremental survivor sizing improvement after filter: $334.30' within
    rounding. The full-stack ($6,420.90) target is asserted by Step 8 Pass C.
    """
    rows = _load_corpus()
    if not rows:
        pytest.skip("forensic corpus not present in this checkout")

    survivors = _v1_v2_survivors(rows)
    assert len(survivors) == EXPECTED_SURVIVOR_COUNT

    original_usd = sum(_f(r, "pnl") for r in survivors)
    capped_usd = 0.0
    rows_capped = 0
    for r in survivors:
        lots = _f(r, "lots")
        pnl = _f(r, "pnl")
        if lots <= 0:
            capped_usd += pnl
            continue
        new_lots = cap_historical_lots(lots)
        if new_lots != lots:
            rows_capped += 1
        capped_usd += rescale_pnl_for_cap(
            original_pnl=pnl, original_lots=lots, capped_lots=new_lots
        )

    assert original_usd == pytest.approx(EXPECTED_ORIGINAL_USD, abs=0.001)
    assert capped_usd == pytest.approx(EXPECTED_CAPPED_USD, abs=0.001)
    # Recovery is positive when capped USD is less negative than original.
    recovery = capped_usd - original_usd
    assert recovery == pytest.approx(EXPECTED_CAP_DELTA, abs=0.001)
    # Cross-check against PHASE9.6's published $334.30 within rounding tolerance
    assert recovery == pytest.approx(334.30, abs=0.01)
    assert rows_capped == EXPECTED_ROWS_CAPPED


def test_no_corpus_lots_below_one_so_min_clamp_doesnt_distort_replay():
    """Audit invariant: the replay assumes cap_historical_lots only touches
    rows with lots > 4. If any survivor row had lots in (0, 1) the lower
    clamp would also fire and the test target would shift. Verify it doesn't.
    """
    rows = _load_corpus()
    if not rows:
        pytest.skip("forensic corpus not present in this checkout")
    survivors = _v1_v2_survivors(rows)
    below_one = [r for r in survivors if 0 < _f(r, "lots") < 1.0]
    assert below_one == []
