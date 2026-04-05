"""Dynamic pip-value sanity for NewEra8 v3 backtester."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "research_out/trade_analysis/newera8_strategy_v3"
sys.path.insert(0, str(V3))


def test_newera8_v3_pip_sanity() -> None:
    from backtest_v3 import assert_pip_sanity

    assert_pip_sanity()
