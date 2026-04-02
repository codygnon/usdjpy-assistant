from __future__ import annotations

from datetime import date, datetime, timezone

from core.regime_backtest_engine.daily_levels import (
    compute_pdh_pdl,
    compute_pmh_pml,
    compute_pwh_pwl,
    compute_round_levels,
    find_nearest_level,
    get_all_levels,
    is_level_cluster,
)
from core.regime_backtest_engine.synthetic_bars import BarDaily


def _d(y: int, m: int, d: int) -> date:
    return date(y, m, d)


def _bd(
    trading_day: date,
    hi: float,
    lo: float,
    o: float = 0.0,
    c: float = 0.0,
) -> BarDaily:
    ts = datetime(trading_day.year, trading_day.month, trading_day.day, tzinfo=timezone.utc)
    return BarDaily(
        trading_day=trading_day,
        timestamp=ts,
        open=o,
        high=hi,
        low=lo,
        close=c,
        volume=None,
        bar_index_start=0,
        bar_index_end=0,
        bar_count=1,
    )


def test_pdh_pdl_prior_day_only() -> None:
    days = [
        _bd(_d(2025, 2, 10), 110.0, 100.0, 105.0, 108.0),
        _bd(_d(2025, 2, 11), 120.0, 115.0, 116.0, 118.0),
    ]
    pdh, pdl = compute_pdh_pdl(days, 1)
    assert pdh == 110.0
    assert pdl == 100.0


def test_pdh_pdl_index_zero_returns_none() -> None:
    days = [_bd(_d(2025, 1, 1), 1, 1)]
    assert compute_pdh_pdl(days, 0) == (None, None)


def test_pwh_pwl_prior_week_only() -> None:
    # Current: Wed 2025-02-19; prior Mon–Fri block Mon Feb 10 – Fri Feb 14
    days = [
        _bd(_d(2025, 2, 10), 10.0, 5.0),
        _bd(_d(2025, 2, 11), 12.0, 6.0),
        _bd(_d(2025, 2, 12), 11.0, 7.0),
        _bd(_d(2025, 2, 13), 13.0, 8.0),
        _bd(_d(2025, 2, 14), 9.0, 4.0),
        _bd(_d(2025, 2, 17), 50.0, 40.0),
        _bd(_d(2025, 2, 18), 60.0, 55.0),
        _bd(_d(2025, 2, 19), 100.0, 90.0),
    ]
    pwh, pwl = compute_pwh_pwl(days, 7)
    assert pwh == 13.0
    assert pwl == 4.0


def test_pmh_pml_prior_month() -> None:
    days = [
        _bd(_d(2025, 1, 28), 200.0, 190.0),
        _bd(_d(2025, 1, 29), 210.0, 195.0),
        _bd(_d(2025, 2, 3), 300.0, 250.0),
    ]
    pmh, pml = compute_pmh_pml(days, 2)
    assert pmh == 210.0
    assert pml == 190.0


def test_round_levels_150_230_radius_50() -> None:
    assert compute_round_levels(150.230, radius_pips=50, pip_size=0.01) == [150.0, 150.5]


def test_round_levels_150_010_radius_5() -> None:
    assert compute_round_levels(150.010, radius_pips=5, pip_size=0.01) == [150.0]


def test_find_nearest_level_empty() -> None:
    assert find_nearest_level(150.0, [], pip_size=0.01) == (None, None)


def test_level_cluster_true_and_false() -> None:
    assert is_level_cluster([150.000, 150.150, 151.000], cluster_radius_pips=20, pip_size=0.01) is True
    assert is_level_cluster([150.000, 151.000], cluster_radius_pips=20, pip_size=0.01) is False


def test_get_all_levels_insufficient_history_no_crash() -> None:
    days = [_bd(_d(2025, 3, 1), 1.0, 0.5)]
    out = get_all_levels(days, 0, 150.23)
    assert out["pdh"] is None
    assert out["pdl"] is None
    assert out["pwh"] is None
    assert out["pwl"] is None
    assert out["pmh"] is None
    assert out["pml"] is None
    assert out["nearest_level"] is not None or out["round_levels"] is not None
