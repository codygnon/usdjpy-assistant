from __future__ import annotations

import pandas as pd

from core.phase3_regime_owned_backtest import (
    RegimeOwnedCandidate,
    UnifiedExecutionPlan,
    PortfolioBacktestState,
    _normalize_quality,
    _route_candidates,
)


def _candidate(strategy: str, side: str, q: float) -> RegimeOwnedCandidate:
    return RegimeOwnedCandidate(
        strategy=strategy,
        session_owner="test",
        side=side,
        signal_time=pd.Timestamp("2025-01-01T00:00:00Z"),
        execute_time=pd.Timestamp("2025-01-01T00:01:00Z"),
        entry_price_basis="next_bar_open",
        sl_pips=10.0,
        raw_quality=q,
        quality_normalized=q,
        regime_label="momentum",
        regime_margin=1.0,
        ownership_cell="momentum/er_high/der_pos",
        reason="test",
        source_features={"tp1_offset_pips": 10.0, "tp2_offset_pips": 20.0},
        execution_plan=UnifiedExecutionPlan(
            sl_price=100.0,
            tp1_price=110.0,
            tp2_price=120.0,
            tp1_close_fraction=0.5,
            be_offset_pips=1.0,
            trail_activate_pips=10.0,
            trail_distance_pips=5.0,
            trail_requires_tp1=True,
            session_close_time=None,
        ),
        risk_pct=0.01,
        max_spread_pips=3.0,
        source_family="tokyo",
    )


def test_router_tie_inside_epsilon_is_no_trade() -> None:
    decision = _route_candidates(
        ts=pd.Timestamp("2025-01-01T00:00:00Z"),
        regime_label="ambiguous",
        regime_margin=0.2,
        ownership_cell="ambiguous/er_high/der_pos",
        candidates=[_candidate("v44_ny", "buy", 0.90), _candidate("london_v2", "buy", 0.89)],
        quality_margin_epsilon=0.05,
    )
    assert decision.winner_strategy is None
    assert decision.no_trade_reason == "quality_tie_no_trade"


def test_quality_normalization_is_causal_and_bounded() -> None:
    state = PortfolioBacktestState(equity=100000.0, peak_equity=100000.0)
    first = _normalize_quality(state, "v14", 10.0)
    assert 0.0 <= first <= 1.0
    for idx in range(1, 30):
        score = _normalize_quality(state, "v14", float(idx))
        assert 0.0 <= score <= 1.0
