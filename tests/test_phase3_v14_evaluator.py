from __future__ import annotations

from datetime import datetime, timezone

from core.phase3_v14_evaluator import compute_v14_lot_size, compute_v14_units_from_config


def test_compute_v14_units_from_config_uses_same_margin_cap_basis_as_lot_size() -> None:
    equity = 100_000.0
    sl_pips = 25.0
    current_price = 159.50
    pip_size = 0.01
    now_utc = datetime(2025, 4, 4, 14, 0, tzinfo=timezone.utc)
    v14_config = {
        "risk_per_trade_pct": 2.0,
        "max_units": 5_000_000,
        "leverage": 33.3,
    }

    units_from_config = compute_v14_units_from_config(
        equity=equity,
        sl_pips=sl_pips,
        current_price=current_price,
        pip_size=pip_size,
        now_utc=now_utc,
        v14_config=v14_config,
    )
    units_from_lot_size = compute_v14_lot_size(
        equity=equity,
        sl_pips=sl_pips,
        current_price=current_price,
        pip_size=pip_size,
        leverage=33.3,
        risk_pct=0.02,
        max_units=5_000_000,
    )

    assert units_from_config == units_from_lot_size
