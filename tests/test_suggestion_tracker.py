from __future__ import annotations

import sqlite3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api import suggestion_tracker


def _ctx(*, bias: str = "bullish", session: str = "London", vol: str = "compressed") -> dict:
    return {
        "spot_price": {"mid": 150.123, "bid": 150.121, "ask": 150.125, "spread_pips": 0.4},
        "session": {"active_sessions": [session], "overlap": None, "warnings": []},
        "cross_asset_bias": {
            "combined_bias": bias,
            "confidence": "medium",
            "usdjpy_implication": "test",
            "oil": {"direction": "up"},
            "dxy": {"direction": "up"},
        },
        "volatility": {"label": vol, "ratio": 0.8, "recent_avg_pips": 6.2},
        "ta_snapshot": {
            "H1": {"regime": "trend", "rsi_value": 58, "rsi_zone": "mid", "macd_direction": "up", "atr_pips": 12, "atr_state": "normal", "adx": 20, "adxr": 19},
            "M5": {"regime": "pullback", "rsi_value": 49, "rsi_zone": "mid", "macd_direction": "flat", "atr_pips": 4, "atr_state": "low", "adx": 18, "adxr": 17},
            "M1": {"regime": "compression", "rsi_value": 46, "rsi_zone": "mid", "macd_direction": "down", "atr_pips": 2, "atr_state": "low", "adx": 14, "adxr": 13},
        },
    }


def _suggestion(*, side: str = "buy", price: float = 150.0, exit_strategy: str = "tp1_be_hwm_trail") -> dict:
    return {
        "side": side,
        "price": price,
        "sl": price - 0.12 if side == "buy" else price + 0.12,
        "tp": price + 0.08 if side == "buy" else price - 0.08,
        "lots": 0.05,
        "time_in_force": "GTC",
        "gtd_time_utc": None,
        "confidence": "medium",
        "rationale": "Test setup.",
        "exit_strategy": exit_strategy,
        "exit_params": {"tp1_pips": 6.0},
    }


def test_init_db_adds_learning_columns(tmp_path: Path) -> None:
    db_path = tmp_path / "ai_suggestions.sqlite"
    suggestion_tracker.init_db(db_path)

    with sqlite3.connect(db_path) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(ai_suggestions)").fetchall()}

    assert "placed_order_json" in cols
    assert "trade_id" in cols


def test_stats_capture_generated_placed_edited_rejected_and_closed(tmp_path: Path) -> None:
    db_path = tmp_path / "ai_suggestions.sqlite"

    sid1 = suggestion_tracker.log_generated(
        db_path,
        profile="demo",
        model="gpt-5.4-mini",
        suggestion=_suggestion(side="buy", price=150.0),
        ctx=_ctx(bias="bullish", session="London", vol="compressed"),
    )
    sid2 = suggestion_tracker.log_generated(
        db_path,
        profile="demo",
        model="gpt-5.4-mini",
        suggestion=_suggestion(side="sell", price=150.5, exit_strategy="tp1_be_only"),
        ctx=_ctx(bias="bearish", session="New York", vol="elevated"),
    )
    sid3 = suggestion_tracker.log_generated(
        db_path,
        profile="demo",
        model="gpt-5.4-mini",
        suggestion=_suggestion(side="buy", price=149.8),
        ctx=_ctx(bias="neutral", session="Tokyo", vol="normal"),
    )

    assert suggestion_tracker.log_action(
        db_path,
        suggestion_id=sid1,
        action="placed",
        edited_fields={},
        placed_order={
            "side": "buy",
            "price": 150.0,
            "lots": 0.05,
            "sl": 149.88,
            "tp": 150.08,
            "time_in_force": "GTC",
            "gtd_time_utc": None,
            "exit_strategy": "tp1_be_hwm_trail",
            "exit_params": {"tp1_pips": 6.0},
        },
        oanda_order_id="101",
    )
    assert suggestion_tracker.mark_filled(db_path, oanda_order_id="101", fill_price=149.99, trade_id="ai_manual:101:1")
    assert suggestion_tracker.mark_closed(db_path, oanda_order_id="101", exit_price=150.09, pnl=42.5, pips=10.0)

    assert suggestion_tracker.log_action(
        db_path,
        suggestion_id=sid2,
        action="placed",
        edited_fields={"price": {"before": 150.5, "after": 150.55}},
        placed_order={
            "side": "sell",
            "price": 150.55,
            "lots": 0.03,
            "sl": 150.67,
            "tp": 150.47,
            "time_in_force": "GTC",
            "gtd_time_utc": None,
            "exit_strategy": "tp1_be_only",
            "exit_params": {"tp1_pips": 4.0},
        },
        oanda_order_id="102",
    )
    assert suggestion_tracker.mark_cancelled_or_expired(db_path, oanda_order_id="102", status="cancelled")

    assert suggestion_tracker.log_action(
        db_path,
        suggestion_id=sid3,
        action="rejected",
        edited_fields={"lots": {"before": 0.05, "after": 0.02}},
    )

    stats = suggestion_tracker.get_stats(db_path, days_back=365)
    overall = stats["overall"]

    assert overall["generated"] == 3
    assert overall["placed"] == 2
    assert overall["rejected"] == 1
    assert overall["edited_before_placed"] == 1
    assert overall["edited_before_rejected"] == 1
    assert overall["filled"] == 1
    assert overall["cancelled"] == 1
    assert overall["closed"] == 1
    assert overall["wins"] == 1
    assert overall["fill_rate_after_placement_pct"] == 50.0
    assert overall["cancel_or_expire_rate_after_placement_pct"] == 50.0
    assert overall["win_rate_closed_pct"] == 100.0

    linked = suggestion_tracker.get_by_order_id(db_path, "101")
    assert linked is not None
    assert linked["trade_id"] == "ai_manual:101:1"
    assert linked["placed_order"]["exit_strategy"] == "tp1_be_hwm_trail"


def test_learning_prompt_block_surfaces_behavioral_feedback(tmp_path: Path) -> None:
    db_path = tmp_path / "ai_suggestions.sqlite"

    sid1 = suggestion_tracker.log_generated(
        db_path,
        profile="demo",
        model="gpt-5.4-mini",
        suggestion=_suggestion(side="buy", price=150.0),
        ctx=_ctx(bias="bullish", session="London", vol="compressed"),
    )
    sid2 = suggestion_tracker.log_generated(
        db_path,
        profile="demo",
        model="gpt-5.4-mini",
        suggestion=_suggestion(side="sell", price=150.4),
        ctx=_ctx(bias="bearish", session="New York", vol="elevated"),
    )

    suggestion_tracker.log_action(
        db_path,
        suggestion_id=sid1,
        action="placed",
        edited_fields={"price": {"before": 150.0, "after": 149.98}},
        placed_order={"side": "buy", "price": 149.98, "exit_strategy": "tp1_be_hwm_trail"},
        oanda_order_id="201",
    )
    suggestion_tracker.mark_filled(db_path, oanda_order_id="201", fill_price=149.98, trade_id="ai_manual:201:1")
    suggestion_tracker.mark_closed(db_path, oanda_order_id="201", exit_price=150.06, pnl=25.0, pips=8.0)

    sid3 = suggestion_tracker.log_generated(
        db_path,
        profile="demo",
        model="gpt-5.4-mini",
        suggestion=_suggestion(side="buy", price=150.1, exit_strategy="tp1_be_m5_trail"),
        ctx=_ctx(bias="bullish", session="London", vol="compressed"),
    )
    suggestion_tracker.log_action(
        db_path,
        suggestion_id=sid3,
        action="placed",
        edited_fields={},
        placed_order={"side": "buy", "price": 150.1, "exit_strategy": "tp1_be_m5_trail"},
        oanda_order_id="203",
    )
    suggestion_tracker.mark_filled(db_path, oanda_order_id="203", fill_price=150.1, trade_id="ai_manual:203:1")
    suggestion_tracker.mark_closed(db_path, oanda_order_id="203", exit_price=150.18, pnl=22.0, pips=8.0)

    sid4 = suggestion_tracker.log_generated(
        db_path,
        profile="demo",
        model="gpt-5.4-mini",
        suggestion=_suggestion(side="buy", price=150.2, exit_strategy="tp1_be_hwm_trail"),
        ctx=_ctx(bias="bullish", session="London", vol="compressed"),
    )
    suggestion_tracker.log_action(
        db_path,
        suggestion_id=sid4,
        action="placed",
        edited_fields={},
        placed_order={"side": "buy", "price": 150.2, "exit_strategy": "tp1_be_hwm_trail"},
        oanda_order_id="204",
    )
    suggestion_tracker.mark_filled(db_path, oanda_order_id="204", fill_price=150.2, trade_id="ai_manual:204:1")
    suggestion_tracker.mark_closed(db_path, oanda_order_id="204", exit_price=150.15, pnl=-12.0, pips=-5.0)

    suggestion_tracker.log_action(
        db_path,
        suggestion_id=sid2,
        action="rejected",
        edited_fields={"lots": {"before": 0.05, "after": 0.03}},
    )

    block = suggestion_tracker.build_learning_prompt_block(
        db_path,
        days_back=365,
        max_recent_examples=4,
        current_ctx=_ctx(bias="bullish", session="London", vol="compressed"),
    )

    assert "FILLMORE LEARNING MEMORY" in block
    assert "Matched analog cohort" in block
    assert "session=london, bias=bullish, vol=compressed" in block
    assert "Matched analog outcomes" in block
    assert "Most-edited fields" in block
    assert "Exit strategy results in this cohort:" in block
    assert "tp1_be_m5_trail" in block
    assert "tp1_be_hwm_trail" in block
    assert "Recent matched examples:" in block
    assert "bias=bullish" in block
    assert "session=london" in block


def test_learning_prompt_block_handles_empty_history(tmp_path: Path) -> None:
    db_path = tmp_path / "ai_suggestions.sqlite"

    block = suggestion_tracker.build_learning_prompt_block(db_path, days_back=365)

    assert "No prior AI suggestion history is available yet" in block


def test_learning_prompt_block_falls_back_when_no_close_context_match(tmp_path: Path) -> None:
    db_path = tmp_path / "ai_suggestions.sqlite"

    sid = suggestion_tracker.log_generated(
        db_path,
        profile="demo",
        model="gpt-5.4-mini",
        suggestion=_suggestion(side="sell", price=150.4),
        ctx=_ctx(bias="bearish", session="New York", vol="elevated"),
    )
    suggestion_tracker.log_action(
        db_path,
        suggestion_id=sid,
        action="placed",
        edited_fields={},
        placed_order={"side": "sell", "price": 150.4, "exit_strategy": "tp1_be_only"},
        oanda_order_id="401",
    )
    suggestion_tracker.mark_cancelled_or_expired(db_path, oanda_order_id="401", status="cancelled")

    block = suggestion_tracker.build_learning_prompt_block(
        db_path,
        days_back=365,
        max_recent_examples=4,
        current_ctx=_ctx(bias="bullish", session="London", vol="compressed"),
    )

    assert "falling back to broad AI history" in block


def test_get_history_returns_row_level_details_with_parsed_json(tmp_path: Path) -> None:
    db_path = tmp_path / "ai_suggestions.sqlite"

    sid = suggestion_tracker.log_generated(
        db_path,
        profile="kumatora2",
        model="gpt-5.4",
        suggestion=_suggestion(side="sell", price=150.45),
        ctx=_ctx(bias="bearish", session="New York", vol="elevated"),
    )
    suggestion_tracker.log_action(
        db_path,
        suggestion_id=sid,
        action="placed",
        edited_fields={"price": {"before": 150.45, "after": 150.5}},
        placed_order={"side": "sell", "price": 150.5, "exit_strategy": "tp1_be_only"},
        oanda_order_id="301",
    )
    suggestion_tracker.mark_filled(db_path, oanda_order_id="301", fill_price=150.5, trade_id="ai_manual:301:1")

    history = suggestion_tracker.get_history(db_path, limit=10, offset=0)

    assert history["total"] == 1
    assert history["limit"] == 10
    assert history["offset"] == 0
    assert len(history["items"]) == 1

    row = history["items"][0]
    assert row["suggestion_id"] == sid
    assert row["profile"] == "kumatora2"
    assert row["market_snapshot"]["macro_bias"]["combined_bias"] == "bearish"
    assert row["edited_fields"]["price"]["after"] == 150.5
    assert row["placed_order"]["exit_strategy"] == "tp1_be_only"
    assert row["trade_id"] == "ai_manual:301:1"
