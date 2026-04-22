from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api import suggestion_tracker
from api.prompt_builder import PromptBuilder


def _ctx() -> dict:
    return {
        "spot_price": {"mid": 150.123, "bid": 150.121, "ask": 150.125, "spread_pips": 0.4},
        "session": {"active_sessions": ["New York"], "overlap": None, "warnings": []},
        "cross_asset_bias": {"combined_bias": "bearish", "confidence": "medium"},
        "volatility": {"label": "compressed", "ratio": 0.8, "recent_avg_pips": 6.2},
        "ta_snapshot": {"M5": {"regime": "pullback", "atr_pips": 4.0}},
    }


def test_prompt_builder_manual_tracks_hashes() -> None:
    builder = PromptBuilder.for_manual_suggest(
        profile=object(),
        profile_name="kumatora2",
        ctx=_ctx(),
        model="gpt-5.4-mini",
    )
    builder.append_system_block("=== NEWS ===\nheadline", kind="news")
    builder.append_system_block("=== LEARNING ===\npattern", kind="learning")
    assembly = builder.build(
        user="Return one trade setup.",
        prompt_version="manual_suggest_v1",
    )

    assert assembly.mode == "suggest_manual"
    assert assembly.news_block_hash
    assert assembly.learning_block_hash
    assert assembly.context_hash
    assert assembly.prompt_hash


def test_prompt_builder_autonomous_memory_builds_system(tmp_path: Path) -> None:
    db_path = tmp_path / "ai_suggestions.sqlite"
    suggestion_tracker.init_db(db_path)

    builder = PromptBuilder.for_autonomous_suggest(
        profile=object(),
        profile_name="newera8",
        ctx=_ctx(),
        model="gpt-5.4-mini",
        autonomous_config={"multi_trade_enabled": True},
        risk_regime={"label": "normal"},
    )
    builder.append_autonomous_memory(
        db_path=db_path,
        risk_regime={"label": "normal"},
    )
    assembly = builder.build(
        user="Return 0-2 opportunities.",
        prompt_version="autonomous_phase_a_v1",
    )

    assert assembly.mode == "suggest_autonomous"
    assert "No prior autonomous suggestions today" in assembly.system
    assert assembly.learning_block_hash
