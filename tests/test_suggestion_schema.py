from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.suggestion_schema import ValidationError, validate_autonomous_suggestion, validate_manual_suggestion


def test_validate_manual_suggestion_accepts_valid_payload() -> None:
    out = validate_manual_suggestion(
        {
            "side": "BUY",
            "price": 150.12,
            "sl": 149.98,
            "tp": 150.36,
            "lots": 1.5,
            "rationale": "Buy reclaim.",
            "confidence": "high",
            "entry_type": "ai_manual",
            "time_in_force": "GTD",
        }
    )

    assert out["side"] == "buy"
    assert out["entry_type"] == "ai_manual"
    assert out["confidence"] == "high"


def test_validate_autonomous_suggestion_allows_skip_without_sl_tp() -> None:
    out = validate_autonomous_suggestion(
        {
            "side": "sell",
            "price": 0,
            "lots": 0,
            "rationale": "No clean setup.",
            "quality": "c",
            "entry_type": "ai_autonomous",
        }
    )

    assert out["lots"] == 0
    assert out["quality"] == "C"
    assert out["entry_type"] == "ai_autonomous"


def test_validate_autonomous_suggestion_rejects_positive_lots_without_levels() -> None:
    with pytest.raises(ValidationError):
        validate_autonomous_suggestion(
            {
                "side": "buy",
                "price": 150.12,
                "sl": 0,
                "tp": 150.35,
                "lots": 2,
                "rationale": "Take the breakout.",
                "quality": "B",
                "entry_type": "ai_autonomous",
            }
        )
