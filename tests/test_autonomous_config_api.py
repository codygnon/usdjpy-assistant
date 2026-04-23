from __future__ import annotations

from api.main import AutonomousConfigPatch


def test_autonomous_config_patch_accepts_lot_sizing_fields() -> None:
    patch = AutonomousConfigPatch(
        base_lot_size=7,
        lot_deviation=2,
        min_lot_size=0.1,
        max_lots_per_trade=9,
    )

    data = patch.model_dump(exclude_none=True)

    assert data["base_lot_size"] == 7
    assert data["lot_deviation"] == 2
    assert data["min_lot_size"] == 0.1
    assert data["max_lots_per_trade"] == 9
