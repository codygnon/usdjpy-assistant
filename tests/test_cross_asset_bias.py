from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from core.regime_backtest_engine.cross_asset_bias import CrossAssetBias, compute_adx
from core.regime_backtest_engine.cross_asset_data import CrossAssetDataLoader

UTC = timezone.utc


def _write_csv(path: Path, header: str, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


def _make_loader(tmp_path: Path, *, brent: list[str], eur: list[str], gold: list[str], silver: list[str]) -> CrossAssetDataLoader:
    usd = tmp_path / "u.csv"
    _write_csv(usd, "time,open,high,low,close,spread_pips", ["2024-01-10T12:00:00Z,150,150,150,150,0.5"])
    b = tmp_path / "b.csv"
    e = tmp_path / "e.csv"
    g = tmp_path / "g.csv"
    s = tmp_path / "s.csv"
    _write_csv(b, "timestamp,open,high,low,close,volume", brent)
    _write_csv(e, "timestamp,open,high,low,close,volume", eur)
    _write_csv(g, "timestamp,open,high,low,close,volume", gold)
    _write_csv(s, "timestamp,open,high,low,close,volume", silver)
    return CrossAssetDataLoader(str(usd), str(b), str(e), str(g), str(s))


def _query_ts() -> datetime:
    return datetime(2024, 1, 10, 20, 0, tzinfo=UTC)


def _flat_h1_rows(n: int = 21, *, price: float = 80.0) -> list[str]:
    out: list[str] = []
    t0 = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    for i in range(n):
        ts = t0 + timedelta(hours=i)
        out.append(ts.strftime("%Y-%m-%d %H:%M:%S") + f",{price},{price},{price},{price},100")
    return out


def test_oil_positive_when_brent_above_sma(tmp_path: Path) -> None:
    rows = [f"2024-01-0{(i // 24) + 1} {i % 24:02d}:00:00,{70+i},{71+i},{69+i},{70.5+i},100" for i in range(21)]
    ld = _make_loader(
        tmp_path,
        brent=rows,
        eur=_flat_h1_rows(),
        gold=["2024-01-01 00:00:00,1,1,1,1,1", "2024-01-02 00:00:00,1,1,1,1,1"] * 11,
        silver=["2024-01-01 00:00:00,1,1,1,1,1", "2024-01-02 00:00:00,1,1,1,1,1"] * 11,
    )
    bias = CrossAssetBias(ld).compute_bias(_query_ts(), None)
    assert bias.oil_signal > 0


def test_oil_negative_when_brent_below_sma(tmp_path: Path) -> None:
    rows = [f"2024-01-0{(i // 24) + 1} {i % 24:02d}:00:00,{100-i},{101-i},{99-i},{100.5-i},100" for i in range(21)]
    ld = _make_loader(
        tmp_path,
        brent=rows,
        eur=_flat_h1_rows(),
        gold=["2024-01-01 00:00:00,1,1,1,1,1"] * 21,
        silver=["2024-01-01 00:00:00,1,1,1,1,1"] * 21,
    )
    bias = CrossAssetBias(ld).compute_bias(_query_ts(), None)
    assert bias.oil_signal < 0


def test_oil_big_move_magnitude_1_5(tmp_path: Path) -> None:
    t0 = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    small = [
        (t0 + timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S") + ",100,100.5,99.5,100,10"
        for i in range(20)
    ]
    last_ts = (t0 + timedelta(hours=20)).strftime("%Y-%m-%d %H:%M:%S")
    rows = small + [f"{last_ts},100,104,99,103,10"]
    flat_eur = [
        (t0 + timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S") + ",1.1,1.1,1.1,1.1,10"
        for i in range(21)
    ]
    ld = _make_loader(
        tmp_path,
        brent=rows,
        eur=flat_eur,
        gold=["2024-01-01 00:00:00,1,1,1,1,1"] * 21,
        silver=["2024-01-01 00:00:00,1,1,1,1,1"] * 21,
    )
    q = t0 + timedelta(hours=21)
    bias = CrossAssetBias(ld).compute_bias(q, None)
    assert bias.brent_is_big_move is True
    assert abs(bias.oil_signal) == 1.5


def test_dxy_positive_when_eurusd_below_sma(tmp_path: Path) -> None:
    rows = [f"2024-01-0{(i // 24) + 1} {i % 24:02d}:00:00,{1.1-i*0.001},{1.11-i*0.001},{1.09-i*0.001},{1.1-i*0.001},100" for i in range(21)]
    flat_oil = [f"2024-01-0{(i // 24) + 1} {i % 24:02d}:00:00,80,80,80,80,100" for i in range(21)]
    ld = _make_loader(
        tmp_path,
        brent=flat_oil,
        eur=rows,
        gold=["2024-01-01 00:00:00,1,1,1,1,1"] * 21,
        silver=["2024-01-01 00:00:00,1,1,1,1,1"] * 21,
    )
    bias = CrossAssetBias(ld).compute_bias(_query_ts(), None)
    assert bias.dxy_signal > 0


def test_dxy_negative_when_eurusd_above_sma(tmp_path: Path) -> None:
    rows = [f"2024-01-0{(i // 24) + 1} {i % 24:02d}:00:00,{1.0+i*0.001},{1.01+i*0.001},{0.99+i*0.001},{1.0+i*0.001},100" for i in range(21)]
    flat_oil = [f"2024-01-0{(i // 24) + 1} {i % 24:02d}:00:00,80,80,80,80,100" for i in range(21)]
    ld = _make_loader(
        tmp_path,
        brent=flat_oil,
        eur=rows,
        gold=["2024-01-01 00:00:00,1,1,1,1,1"] * 21,
        silver=["2024-01-01 00:00:00,1,1,1,1,1"] * 21,
    )
    bias = CrossAssetBias(ld).compute_bias(_query_ts(), None)
    assert bias.dxy_signal < 0


def test_gold_below_sma_positive_half(tmp_path: Path) -> None:
    drows = [f"2024-01-{(i%28)+1:02d} 00:00:00,{2000-i},{2010-i},{1990-i},{2005-i},10" for i in range(21)]
    flat = [f"2024-01-0{(i // 24) + 1} {i % 24:02d}:00:00,80,80,80,80,100" for i in range(21)]
    ld = _make_loader(
        tmp_path,
        brent=flat,
        eur=flat,
        gold=drows,
        silver=["2024-01-01 00:00:00,1,1,1,1,1"] * 21,
    )
    bias = CrossAssetBias(ld).compute_bias(datetime(2024, 1, 28, 23, 0, tzinfo=UTC), None)
    assert bias.gold_signal == 0.5


def test_gold_above_sma_negative_half(tmp_path: Path) -> None:
    drows = [f"2024-01-{(i%28)+1:02d} 00:00:00,{1900+i},{1910+i},{1890+i},{1905+i},10" for i in range(21)]
    flat = [f"2024-01-0{(i // 24) + 1} {i % 24:02d}:00:00,80,80,80,80,100" for i in range(21)]
    ld = _make_loader(
        tmp_path,
        brent=flat,
        eur=flat,
        gold=drows,
        silver=["2024-01-01 00:00:00,1,1,1,1,1"] * 21,
    )
    bias = CrossAssetBias(ld).compute_bias(datetime(2024, 1, 28, 23, 0, tzinfo=UTC), None)
    assert bias.gold_signal == -0.5


def test_silver_same_as_gold(tmp_path: Path) -> None:
    drows = [f"2024-01-{(i%28)+1:02d} 00:00:00,{30-i},{31-i},{29-i},{30.5-i},10" for i in range(21)]
    flat = [f"2024-01-0{(i // 24) + 1} {i % 24:02d}:00:00,80,80,80,80,100" for i in range(21)]
    ld = _make_loader(
        tmp_path,
        brent=flat,
        eur=flat,
        gold=["2024-01-01 00:00:00,1,1,1,1,1"] * 21,
        silver=drows,
    )
    bias = CrossAssetBias(ld).compute_bias(datetime(2024, 1, 28, 23, 0, tzinfo=UTC), None)
    assert bias.silver_signal == 0.5


def test_conflict_agree_full(tmp_path: Path) -> None:
    oil_up = [f"2024-01-0{(i // 24) + 1} {i % 24:02d}:00:00,{70+i},{71+i},{69+i},{70.5+i},100" for i in range(21)]
    eur_dn = [f"2024-01-0{(i // 24) + 1} {i % 24:02d}:00:00,{1.1-i*0.001},{1.11-i*0.001},{1.09-i*0.001},{1.1-i*0.001},100" for i in range(21)]
    g = ["2024-01-01 00:00:00,1,1,1,1,1"] * 21
    ld = _make_loader(tmp_path, brent=oil_up, eur=eur_dn, gold=g, silver=g)
    bias = CrossAssetBias(ld).compute_bias(_query_ts(), None)
    assert bias.oil_dxy_agree is True
    assert bias.conflict_action == "FULL"


def test_conflict_favor_oil_big_move(tmp_path: Path) -> None:
    small = [f"2024-01-01 {h:02d}:00:00,100,100.5,99.5,100,10" for h in range(0, 20)]
    last = "2024-01-01 20:00:00,100,104,99,103,10"
    oil_rows = small + [last]
    eur_up = [f"2024-01-01 {h:02d}:00:00,{1.0+h*0.0001},{1.01},{0.99},{1.0+h*0.0001},10" for h in range(21)]
    g = ["2024-01-01 00:00:00,1,1,1,1,1"] * 21
    ld = _make_loader(tmp_path, brent=oil_rows, eur=eur_up, gold=g, silver=g)
    bias = CrossAssetBias(ld).compute_bias(datetime(2024, 1, 1, 21, 0, tzinfo=UTC), None)
    assert bias.brent_is_big_move is True
    assert bias.conflict_action == "FAVOR_OIL"
    assert bias.oil_signal != 0
    assert bias.dxy_signal == 0.0


def test_conflict_sit_out_no_big_move(tmp_path: Path) -> None:
    oil_up = [f"2024-01-0{(i // 24) + 1} {i % 24:02d}:00:00,{70+i},{71+i},{69+i},{70.5+i},100" for i in range(21)]
    eur_up = [f"2024-01-0{(i // 24) + 1} {i % 24:02d}:00:00,{1.0+i*0.001},{1.01},{0.99},{1.0+i*0.001},100" for i in range(21)]
    g = ["2024-01-01 00:00:00,1,1,1,1,1"] * 21
    ld = _make_loader(tmp_path, brent=oil_up, eur=eur_up, gold=g, silver=g)
    bias = CrossAssetBias(ld).compute_bias(_query_ts(), None)
    assert bias.brent_is_big_move is False
    assert bias.conflict_action == "SIT_OUT"
    assert bias.oil_signal == 0.0
    assert bias.dxy_signal == 0.0


def _daily_falling_closes(base: float, n: int = 21) -> list[str]:
    """Descending closes so last bar is below 20-SMA → +0.5 USDJPY-long signal for metals."""
    return [
        f"2024-01-{(i + 1):02d} 00:00:00,{base+10-i},{base+12-i},{base+8-i},{base-i},10"
        for i in range(n)
    ]


def test_composite_strong_mild_neutral_short(tmp_path: Path) -> None:
    flat = _flat_h1_rows()
    oil_up = [f"2024-01-0{(i // 24) + 1} {i % 24:02d}:00:00,{70+i},{71+i},{69+i},{70.5+i},100" for i in range(21)]
    eur_dn = [f"2024-01-0{(i // 24) + 1} {i % 24:02d}:00:00,{1.1-i*0.001},{1.11-i*0.001},{1.09-i*0.001},{1.1-i*0.001},100" for i in range(21)]
    g_bull = _daily_falling_closes(2000.0, 21)
    s_bull = _daily_falling_closes(40.0, 21)
    ld = _make_loader(tmp_path, brent=oil_up, eur=eur_dn, gold=g_bull, silver=s_bull)
    b = CrossAssetBias(ld).compute_bias(datetime(2024, 1, 28, 23, 0, tzinfo=UTC), None)
    assert b.bias == "STRONG_LONG"
    assert b.raw_score >= 2.0

    g_flat = [f"2024-01-{(i + 1):02d} 00:00:00,50,51,49,50,10" for i in range(21)]
    ld2 = _make_loader(tmp_path, brent=flat, eur=flat, gold=g_flat, silver=g_flat)
    b2 = CrossAssetBias(ld2).compute_bias(datetime(2024, 1, 28, 23, 0, tzinfo=UTC), None)
    assert b2.bias == "NEUTRAL"

    oil_dn = [f"2024-01-0{(i // 24) + 1} {i % 24:02d}:00:00,{100-i},{101-i},{99-i},{99.5-i},100" for i in range(21)]
    eur_up2 = [f"2024-01-0{(i // 24) + 1} {i % 24:02d}:00:00,{1.0+i*0.001},{1.01},{0.99},{1.0+i*0.001},100" for i in range(21)]
    g_bear = [f"2024-01-{(i + 1):02d} 00:00:00,{1900+i},{1910+i},{1890+i},{1905+i},10" for i in range(21)]
    s_bear = [f"2024-01-{(i + 1):02d} 00:00:00,{30+i},{31+i},{29+i},{30.5+i},10" for i in range(21)]
    ld3 = _make_loader(tmp_path, brent=oil_dn, eur=eur_up2, gold=g_bear, silver=s_bear)
    b3 = CrossAssetBias(ld3).compute_bias(datetime(2024, 1, 28, 23, 0, tzinfo=UTC), None)
    assert b3.bias == "STRONG_SHORT"
    assert b3.raw_score <= -2.0


def test_mild_long_band(tmp_path: Path) -> None:
    flat = _flat_h1_rows()
    g_bull = _daily_falling_closes(2000.0, 21)
    ld = _make_loader(tmp_path, brent=flat, eur=flat, gold=g_bull, silver=flat[:21])
    bias = CrossAssetBias(ld).compute_bias(datetime(2024, 1, 28, 23, 0, tzinfo=UTC), None)
    assert 0.5 <= bias.raw_score < 2.0
    assert bias.bias == "MILD_LONG"


def test_adx_regime_buckets() -> None:
    n = 80
    highs = [100.0 + i * 0.5 for i in range(n)]
    lows = [99.0 + i * 0.5 for i in range(n)]
    closes = [99.5 + i * 0.5 for i in range(n)]
    adx = compute_adx(highs, lows, closes, 14)
    last = next(x for x in reversed(adx) if x is not None)
    assert last >= 25.0

    flat_h = [100.0] * n
    flat_l = [99.5] * n
    flat_c = [99.75] * n
    adx_f = compute_adx(flat_h, flat_l, flat_c, 14)
    last_f = next(x for x in reversed(adx_f) if x is not None)
    assert last_f < 20.0


def test_regime_via_compute_bias(tmp_path: Path) -> None:
    flat = [f"2024-01-0{(i // 24) + 1} {i % 24:02d}:00:00,80,80,80,80,100" for i in range(21)]
    g = ["2024-01-01 00:00:00,1,1,1,1,1"] * 21
    ld = _make_loader(tmp_path, brent=flat, eur=flat, gold=g, silver=g)
    n = 80
    trend_daily = [
        {"high": 100.0 + i * 0.5, "low": 99.0 + i * 0.5, "close": 99.5 + i * 0.5}
        for i in range(n)
    ]
    b = CrossAssetBias(ld).compute_bias(_query_ts(), trend_daily)
    assert b.regime == "TRENDING"
    assert b.adx_value >= 25.0

    flat_daily = [{"high": 100.0, "low": 99.5, "close": 99.75} for _ in range(n)]
    b2 = CrossAssetBias(ld).compute_bias(_query_ts(), flat_daily)
    assert b2.regime == "RANGING"

    mild_h = [100.0 + (i % 3) * 0.1 for i in range(n)]
    mild_l = [99.0 + (i % 3) * 0.1 for i in range(n)]
    mild_c = [99.5 + (i % 3) * 0.1 for i in range(n)]
    mild_daily = [{"high": h, "low": l, "close": c} for h, l, c in zip(mild_h, mild_l, mild_c)]
    b3 = CrossAssetBias(ld).compute_bias(_query_ts(), mild_daily)
    assert b3.regime in ("WEAK_TREND", "RANGING", "TRENDING")


def test_size_multipliers(tmp_path: Path) -> None:
    flat = _flat_h1_rows()
    oil_up = [f"2024-01-0{(i // 24) + 1} {i % 24:02d}:00:00,{70+i},{71+i},{69+i},{70.5+i},100" for i in range(21)]
    eur_dn = [f"2024-01-0{(i // 24) + 1} {i % 24:02d}:00:00,{1.1-i*0.001},{1.11-i*0.001},{1.09-i*0.001},{1.1-i*0.001},100" for i in range(21)]
    g_bull = _daily_falling_closes(2000.0, 21)
    s_bull = _daily_falling_closes(40.0, 21)
    ld = _make_loader(tmp_path, brent=oil_up, eur=eur_dn, gold=g_bull, silver=s_bull)
    b = CrossAssetBias(ld).compute_bias(datetime(2024, 1, 28, 23, 0, tzinfo=UTC), None)
    assert b.bias == "STRONG_LONG"
    assert b.size_multiplier == 1.5

    ld2 = _make_loader(tmp_path, brent=oil_up, eur=eur_dn, gold=g_bull, silver=flat[:21])
    b2 = CrossAssetBias(ld2).compute_bias(datetime(2024, 1, 28, 23, 0, tzinfo=UTC), None)
    assert b2.bias == "STRONG_LONG"
    assert b2.raw_score >= 2.0
    assert b2.size_multiplier == 1.0

    g_disagree = [f"2024-01-{(i + 1):02d} 00:00:00,{1900+i},{1910+i},{1890+i},{1905+i},10" for i in range(21)]
    ld3 = _make_loader(tmp_path, brent=oil_up, eur=eur_dn, gold=g_disagree, silver=flat[:21])
    b3 = CrossAssetBias(ld3).compute_bias(datetime(2024, 1, 28, 23, 0, tzinfo=UTC), None)
    assert b3.bias == "MILD_LONG"
    assert b3.raw_score == 1.5
    assert b3.size_multiplier == 0.5


def test_insufficient_history_zero_signals(tmp_path: Path) -> None:
    short = ["2024-01-01 12:00:00,80,81,79,80,10"]
    ld = _make_loader(tmp_path, brent=short, eur=short, gold=short, silver=short)
    bias = CrossAssetBias(ld).compute_bias(datetime(2024, 1, 1, 15, 0, tzinfo=UTC), None)
    assert bias.oil_signal == 0.0
    assert bias.dxy_signal == 0.0


def test_usdjpy_daily_none_no_crash(tmp_path: Path) -> None:
    flat = [f"2024-01-0{(i // 24) + 1} {i % 24:02d}:00:00,80,80,80,80,100" for i in range(21)]
    g = ["2024-01-01 00:00:00,1,1,1,1,1"] * 21
    ld = _make_loader(tmp_path, brent=flat, eur=flat, gold=g, silver=g)
    b = CrossAssetBias(ld).compute_bias(_query_ts(), None)
    assert b.adx_value == 0.0
    assert b.regime == "RANGING"


def test_causality_brent_not_visible_before_completion(tmp_path: Path) -> None:
    t0 = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    rows = [(t0 + timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S") + ",100,101,99,100,10" for i in range(21)]
    flat = _flat_h1_rows()
    g = ["2024-01-01 00:00:00,1,1,1,1,1"] * 21
    ld = _make_loader(tmp_path, brent=rows, eur=flat, gold=g, silver=g)
    q_mid = t0 + timedelta(minutes=30)
    b = CrossAssetBias(ld).compute_bias(q_mid, None)
    assert b.oil_signal == 0.0


def test_performance_ten_thousand_bias_calls(tmp_path: Path) -> None:
    import time

    rows = [f"2024-01-0{(i // 24) + 1} {i % 24:02d}:00:00,{70+i},{71+i},{69+i},{70.5+i},100" for i in range(21)]
    g = ["2024-01-01 00:00:00,1,1,1,1,1"] * 21
    ld = _make_loader(tmp_path, brent=rows, eur=rows, gold=g, silver=g)
    cb = CrossAssetBias(ld)
    t0 = time.perf_counter()
    base = datetime(2024, 1, 10, 12, 0, tzinfo=UTC)
    for i in range(10_000):
        cb.compute_bias(base + timedelta(minutes=i), None)
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0, f"10k bias calls took {elapsed:.3f}s"
