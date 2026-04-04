from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.assistant.config import RiskConfig
from core.assistant.oanda_client import AccountSummary, OpenTrade, PriceSnapshot
from core.assistant.risk_manager import RiskManager, TradeProposal


def _account(equity: float = 100000.0) -> AccountSummary:
    return AccountSummary(
        account_id="acct",
        balance=equity,
        unrealized_pnl=0.0,
        equity=equity,
        margin_used=0.0,
        margin_available=equity,
        open_trade_count=0,
        currency="USD",
    )


def _price(mid: float = 150.0) -> PriceSnapshot:
    return PriceSnapshot(
        instrument="USD_JPY",
        bid=mid - 0.01,
        ask=mid + 0.01,
        mid=mid,
        spread_pips=2.0,
        timestamp=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
    )


def _trade(direction: str, units: int = 100000, open_price: float = 150.0) -> OpenTrade:
    dt = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
    signed = abs(units) if direction == "long" else -abs(units)
    return OpenTrade(
        trade_id="1",
        instrument="USD_JPY",
        direction=direction,
        units=signed,
        open_price=open_price,
        open_time=dt,
        unrealized_pnl=0.0,
        stop_loss=None,
        take_profit=None,
        trailing_stop_distance=None,
    )


def test_usdjpy_pip_value_at_150_is_667_per_lot() -> None:
    rm = RiskManager(RiskConfig())
    assert rm.compute_pip_value("USD_JPY", 150.0, 1.0) == pytest.approx(6.6666666667)


def test_usdjpy_pip_value_at_140_is_714_per_lot() -> None:
    rm = RiskManager(RiskConfig())
    assert rm.compute_pip_value("USD_JPY", 140.0, 1.0) == pytest.approx(7.1428571429)


def test_usdjpy_pip_value_at_160_is_625_per_lot() -> None:
    rm = RiskManager(RiskConfig())
    assert rm.compute_pip_value("USD_JPY", 160.0, 1.0) == pytest.approx(6.25)


def test_pip_value_scales_with_lot_size() -> None:
    rm = RiskManager(RiskConfig())
    assert rm.compute_pip_value("USD_JPY", 150.0, 0.5) == pytest.approx(3.3333333333)
    assert rm.compute_pip_value("USD_JPY", 150.0, 10.0) == pytest.approx(66.6666666667)


def test_stop_distance_pips_examples() -> None:
    rm = RiskManager(RiskConfig())
    assert rm.compute_stop_distance_pips("USD_JPY", 150.500, 150.000) == pytest.approx(50.0)
    assert rm.compute_stop_distance_pips("USD_JPY", 150.000, 150.500) == pytest.approx(50.0)
    assert rm.compute_stop_distance_pips("USD_JPY", 150.500, 150.400) == pytest.approx(10.0)
    assert rm.compute_stop_distance_pips("USD_JPY", 150.000, 148.000) == pytest.approx(200.0)


def test_position_size_100k_1pct_50pip_stop_is_3_lots() -> None:
    rm = RiskManager(RiskConfig())
    lots, units = rm.compute_position_size(100000.0, 150.0, 149.5)
    assert lots == pytest.approx(3.0)
    assert units == 300000


def test_wider_stop_means_smaller_position() -> None:
    rm = RiskManager(RiskConfig())
    tight_lots, _ = rm.compute_position_size(100000.0, 150.0, 149.5)
    wide_lots, _ = rm.compute_position_size(100000.0, 150.0, 148.5)
    assert wide_lots < tight_lots


def test_tighter_stop_means_larger_position_capped_at_max() -> None:
    rm = RiskManager(RiskConfig(max_position_size_lots=20))
    lots, units = rm.compute_position_size(100000.0, 150.0, 149.99)
    assert lots == pytest.approx(20.0)
    assert units == 2000000


def test_units_rounded_down_to_nearest_1000() -> None:
    rm = RiskManager(RiskConfig(max_risk_per_trade=0.01))
    lots, units = rm.compute_position_size(100000.0, 150.0, 149.47)
    assert units % 1000 == 0
    assert lots == pytest.approx(units / 100000.0)


def test_zero_stop_distance_returns_zero_size() -> None:
    rm = RiskManager(RiskConfig())
    assert rm.compute_position_size(100000.0, 150.0, 150.0) == (0.0, 0)


def test_auto_stop_long_and_short() -> None:
    rm = RiskManager(RiskConfig(default_stop_atr_multiple=2.0, catastrophic_stop_pips=200.0))
    assert rm.compute_auto_stop("long", 150.0, 0.20) == pytest.approx(149.6)
    assert rm.compute_auto_stop("short", 150.0, 0.20) == pytest.approx(150.4)


def test_auto_stop_capped_at_catastrophic_limit() -> None:
    rm = RiskManager(RiskConfig(default_stop_atr_multiple=2.0, catastrophic_stop_pips=50.0))
    assert rm.compute_auto_stop("long", 150.0, 1.0) == pytest.approx(149.5)


def test_tp_levels_long_and_short() -> None:
    rm = RiskManager(RiskConfig(tp1_ratio=1.0, tp2_ratio=2.0))
    assert rm.compute_tp_levels("long", 150.0, 149.5) == pytest.approx((150.5, 151.0))
    assert rm.compute_tp_levels("short", 150.0, 150.5) == pytest.approx((149.5, 149.0))


def test_assess_valid_trade_has_no_errors() -> None:
    rm = RiskManager(RiskConfig())
    assessment = rm.assess(
        TradeProposal("USD_JPY", "long", 150.0, 149.5, None, None),
        _account(),
        [],
        _price(),
    )
    assert assessment.is_valid is True
    assert assessment.errors == []


def test_assess_long_with_stop_above_entry_errors() -> None:
    rm = RiskManager(RiskConfig())
    assessment = rm.assess(TradeProposal("USD_JPY", "long", 150.0, 150.1, None, None), _account(), [], _price())
    assert any("BELOW entry" in error for error in assessment.errors)


def test_assess_short_with_stop_below_entry_errors() -> None:
    rm = RiskManager(RiskConfig())
    assessment = rm.assess(TradeProposal("USD_JPY", "short", 150.0, 149.9, None, None), _account(), [], _price())
    assert any("ABOVE entry" in error for error in assessment.errors)


def test_assess_risk_exceeds_max_errors() -> None:
    rm = RiskManager(RiskConfig(max_risk_per_trade=0.01))
    assessment = rm.assess(TradeProposal("USD_JPY", "long", 150.0, 149.0, None, 10.0), _account(), [], _price())
    assert any("Risk" in error for error in assessment.errors)


def test_assess_size_exceeds_max_errors() -> None:
    rm = RiskManager(RiskConfig(max_position_size_lots=2))
    assessment = rm.assess(TradeProposal("USD_JPY", "long", 150.0, 149.5, None, 3.0), _account(), [], _price())
    assert any("exceeds max" in error for error in assessment.errors)


def test_assess_stop_beyond_catastrophic_errors() -> None:
    rm = RiskManager(RiskConfig(catastrophic_stop_pips=100))
    assessment = rm.assess(TradeProposal("USD_JPY", "long", 150.0, 148.0, None, None), _account(), [], _price())
    assert any("catastrophic limit" in error for error in assessment.errors)


def test_assess_warns_on_opposing_position() -> None:
    rm = RiskManager(RiskConfig())
    assessment = rm.assess(
        TradeProposal("USD_JPY", "long", 150.0, 149.5, None, None),
        _account(),
        [_trade("short")],
        _price(),
    )
    assert any("OPPOSING POSITION" in warning for warning in assessment.warnings)


def test_assess_warns_on_adding_to_existing_position() -> None:
    rm = RiskManager(RiskConfig())
    assessment = rm.assess(
        TradeProposal("USD_JPY", "long", 150.0, 149.5, None, None),
        _account(),
        [_trade("long")],
        _price(),
    )
    assert any("ADDING to existing" in warning for warning in assessment.warnings)


def test_assess_warns_on_tight_and_wide_stops() -> None:
    rm = RiskManager(RiskConfig())
    tight = rm.assess(TradeProposal("USD_JPY", "long", 150.0, 149.95, None, None), _account(), [], _price())
    wide = rm.assess(TradeProposal("USD_JPY", "long", 150.0, 148.9, None, None), _account(), [], _price())
    assert any("very tight" in warning for warning in tight.warnings)
    assert any("wide" in warning for warning in wide.warnings)


def test_assess_auto_sizes_when_lots_missing() -> None:
    rm = RiskManager(RiskConfig())
    assessment = rm.assess(TradeProposal("USD_JPY", "long", 150.0, 149.5, None, None), _account(), [], _price())
    assert assessment.recommended_lots == pytest.approx(3.0)
    assert assessment.recommended_units == 300000


def test_assess_uses_user_specified_lots_and_stop() -> None:
    rm = RiskManager(RiskConfig())
    assessment = rm.assess(TradeProposal("USD_JPY", "long", 150.0, 149.5, None, 1.5), _account(), [], _price())
    assert assessment.recommended_lots == pytest.approx(1.5)
    assert assessment.recommended_units == 150000
    assert assessment.stop_loss == pytest.approx(149.5)


def test_assess_without_stop_uses_catastrophic_stop_with_warning() -> None:
    rm = RiskManager(RiskConfig(catastrophic_stop_pips=200))
    assessment = rm.assess(TradeProposal("USD_JPY", "long", 150.0, None, None, None), _account(), [], _price())
    assert assessment.stop_loss == pytest.approx(148.0)
    assert any("No stop loss specified" in warning for warning in assessment.warnings)
