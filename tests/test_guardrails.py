from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.assistant.config import RiskConfig, SessionConfig
from core.assistant.guardrails import Guardrails
from core.assistant.oanda_client import OpenTrade
from core.assistant.risk_manager import RiskAssessment, TradeProposal
from core.assistant.session_manager import SessionStatus


def _assessment(**overrides) -> RiskAssessment:
    base = RiskAssessment(
        recommended_lots=2.0,
        recommended_units=200000,
        risk_amount=500.0,
        risk_percent=0.005,
        stop_loss=149.5,
        stop_distance_pips=50.0,
        take_profit_1=150.5,
        take_profit_2=151.0,
        tp1_distance_pips=50.0,
        tp2_distance_pips=100.0,
        is_valid=True,
        warnings=[],
        errors=[],
        account_equity=100000.0,
        pip_value_per_lot=6.67,
        current_exposure_percent=0.01,
        exposure_after_trade_percent=0.03,
    )
    return replace(base, **overrides)


def _session(**overrides) -> SessionStatus:
    base = SessionStatus(
        tokyo_active=False,
        london_active=True,
        ny_active=False,
        active_sessions=["London"],
        overlap=None,
        minutes_to_next_close=120,
        next_close_session="London",
        current_utc=datetime.now(timezone.utc),
    )
    return replace(base, **overrides)


def _proposal() -> TradeProposal:
    return TradeProposal("USD_JPY", "long", 150.0, 149.5, None, None)


def _trade(trade_id: str) -> OpenTrade:
    return OpenTrade(
        trade_id=trade_id,
        instrument="USD_JPY",
        direction="long",
        units=100000,
        open_price=150.0,
        open_time=datetime.now(timezone.utc),
        unrealized_pnl=0.0,
        stop_loss=None,
        take_profit=None,
        trailing_stop_distance=None,
    )


def test_clean_trade_can_proceed_without_confirmations() -> None:
    result = Guardrails(RiskConfig(), SessionConfig()).check(_proposal(), _assessment(), _session(), {}, [])
    assert result.can_proceed is True
    assert result.confirmations_needed == []
    assert result.blocks == []


def test_large_position_needs_confirmation() -> None:
    result = Guardrails(RiskConfig(), SessionConfig()).check(_proposal(), _assessment(recommended_lots=10.0), _session(), {}, [])
    assert any("LARGE POSITION" in item for item in result.confirmations_needed)


def test_near_session_close_needs_confirmation() -> None:
    result = Guardrails(RiskConfig(), SessionConfig()).check(
        _proposal(),
        _assessment(),
        _session(minutes_to_next_close=10),
        {},
        [],
    )
    assert any("closes in 10 minutes" in item for item in result.confirmations_needed)


def test_losing_streak_confirmation() -> None:
    result = Guardrails(RiskConfig(), SessionConfig()).check(
        _proposal(),
        _assessment(),
        _session(),
        {"current_streak": ("loss", 3)},
        [],
    )
    assert any("LOSING STREAK" in item for item in result.confirmations_needed)


def test_daily_loss_confirmation() -> None:
    result = Guardrails(RiskConfig(), SessionConfig()).check(
        _proposal(),
        _assessment(),
        _session(),
        {"today_pnl": -2500.0},
        [],
    )
    assert any("Daily loss exceeds 2%" in item for item in result.confirmations_needed)


def test_three_open_positions_confirmation() -> None:
    result = Guardrails(RiskConfig(), SessionConfig()).check(
        _proposal(),
        _assessment(),
        _session(),
        {},
        [_trade("1"), _trade("2"), _trade("3")],
    )
    assert any("3 USDJPY positions" in item for item in result.confirmations_needed)


def test_assessment_errors_block_trade() -> None:
    result = Guardrails(RiskConfig(), SessionConfig()).check(
        _proposal(),
        _assessment(errors=["bad stop"]),
        _session(),
        {},
        [],
    )
    assert result.can_proceed is False
    assert result.blocks == ["bad stop"]


def test_multiple_guardrails_all_appear() -> None:
    result = Guardrails(RiskConfig(), SessionConfig()).check(
        _proposal(),
        _assessment(recommended_lots=12.0, warnings=["watch spread"]),
        _session(active_sessions=[], minutes_to_next_close=5, next_close_session="London"),
        {"current_streak": ("loss", 4), "today_pnl": -3000.0},
        [_trade("1"), _trade("2"), _trade("3")],
    )
    assert len(result.confirmations_needed) >= 5
    assert result.warnings == ["watch spread"]
