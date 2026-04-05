"""Dynamic pip-value sanity for NewEra8 v4 backtester."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V4 = ROOT / "research_out/trade_analysis/newera8_strategy_v4"
sys.path.insert(0, str(V4))


def test_newera8_v4_pip_sanity() -> None:
    from backtest_v4 import assert_pip_sanity

    assert_pip_sanity()
