from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api import ai_trading_chat, suggestion_tracker


def _ctx() -> dict:
    return {
        "spot_price": {"mid": 159.032, "bid": 159.025, "ask": 159.04, "spread_pips": 1.5},
        "session": {"active_sessions": ["Tokyo", "London"], "overlap": "Tokyo/London overlap", "warnings": []},
        "cross_asset_bias": {
            "combined_bias": "bearish",
            "confidence": "high",
            "usdjpy_implication": "bearish USDJPY",
            "oil": {"direction": "bearish"},
            "dxy": {"direction": "bearish"},
        },
        "volatility": {"label": "Above average", "ratio": 1.3, "recent_avg_pips": 2.8},
    }


def _suggestion() -> dict:
    return {
        "side": "sell",
        "price": 159.3,
        "sl": 159.43,
        "tp": 159.22,
        "lots": 0.05,
        "time_in_force": "GTC",
        "gtd_time_utc": None,
        "confidence": "medium",
        "rationale": "Sell the pop into resistance.",
        "exit_strategy": "tp1_be_m5_trail",
        "exit_params": {"tp1_pips": 6.0},
    }


def test_exec_get_ai_suggestion_history_reads_tracker_rows(tmp_path: Path, monkeypatch) -> None:
    profile_name = "kumatora2"
    log_dir = tmp_path / profile_name
    log_dir.mkdir(parents=True, exist_ok=True)
    db_path = log_dir / "ai_suggestions.sqlite"

    sid = suggestion_tracker.log_generated(
        db_path,
        profile=profile_name,
        model="gpt-5.4",
        suggestion=_suggestion(),
        ctx=_ctx(),
    )
    suggestion_tracker.log_action(
        db_path,
        suggestion_id=sid,
        action="placed",
        edited_fields={"lots": {"before": 0.05, "after": 5}},
        placed_order={
            "side": "sell",
            "price": 159.3,
            "lots": 5.0,
            "sl": 159.43,
            "tp": 159.22,
            "time_in_force": "GTC",
            "gtd_time_utc": None,
            "exit_strategy": "tp1_be_m5_trail",
            "exit_params": {"tp1_pips": 6.0},
        },
        oanda_order_id="127777",
    )

    monkeypatch.setattr(ai_trading_chat, "LOGS_DIR", tmp_path)

    out = ai_trading_chat._exec_get_ai_suggestion_history(
        {"days_back": 30, "limit": 5, "oanda_order_id": "127777"},
        profile_name=profile_name,
    )

    assert "Fillmore suggestion for order 127777:" in out
    assert "GTP-5" not in out
    assert "gpt-5.4" in out
    assert "SELL 159.3" in out
    assert "edited: lots: 0.05 -> 5" in out
    assert "context: session=Tokyo/London overlap, macro=bearish, vol=Above average, spread=1.5p" in out
    assert "rationale: Sell the pop into resistance." in out
