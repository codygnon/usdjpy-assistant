from __future__ import annotations

import pandas as pd

from core.phase3_ownership_core import compute_phase3_ownership_audit_for_data


def test_compute_phase3_ownership_audit_for_data_empty_shape() -> None:
    audit = compute_phase3_ownership_audit_for_data({"M5": pd.DataFrame()}, 0.01)
    assert audit["schema"] == "phase3_ownership_audit_v1"
    assert audit["regime_label"] == "unknown"
    assert audit["ownership_cell"] is None
    assert audit["defensive_v44_regime_block"] is False
