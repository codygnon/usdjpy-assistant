from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from core.regime_backtest_engine.cross_asset_data import (
    CrossAssetDataLoader,
    _parse_cross_timestamp_flexible,
    _parse_usdjpy_time,
    _daily_completed_at,
    _h1_completed_at,
)

REPO = Path(__file__).resolve().parents[1]
USDJPY_PATH = REPO / "research_out" / "USDJPY_M1_OANDA_1000k.csv"
BRENT_PATH = REPO / "research_out" / "cross_assets" / "BCO_USD_H1_OANDA.csv"
EUR_PATH = REPO / "research_out" / "cross_assets" / "EUR_USD_H1_OANDA.csv"
GOLD_PATH = REPO / "research_out" / "cross_assets" / "XAU_USD_D_OANDA.csv"
SILVER_PATH = REPO / "research_out" / "cross_assets" / "XAG_USD_D_OANDA.csv"


def _all_research_files_exist() -> bool:
    return all(
        p.is_file()
        for p in (USDJPY_PATH, BRENT_PATH, EUR_PATH, GOLD_PATH, SILVER_PATH)
    )


requires_full_dataset = pytest.mark.skipif(
    not _all_research_files_exist(),
    reason="research_out USDJPY + cross_assets CSVs not all present",
)


def _write_csv(path: Path, header: str, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


@pytest.fixture
def tiny_h1_causality_paths(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    """USDJPY stub + H1 Brent/EUR with bars 12:00, 13:00, 14:00 on 2024-01-01."""
    usd = tmp_path / "usdjpy.csv"
    _write_csv(
        usd,
        "time,open,high,low,close,spread_pips",
        ["2024-01-01T12:00:00Z,1,1,1,1,0.5"],
    )
    h1_rows = [
        "2024-01-01 12:00:00,1,1,1,1,10",
        "2024-01-01 13:00:00,1,1,1,1,10",
        "2024-01-01 14:00:00,1,1,1,1,10",
    ]
    brent = tmp_path / "bco.csv"
    eur = tmp_path / "eur.csv"
    _write_csv(brent, "timestamp,open,high,low,close,volume", h1_rows)
    _write_csv(eur, "timestamp,open,high,low,close,volume", h1_rows)
    gold = tmp_path / "gold.csv"
    silver = tmp_path / "silver.csv"
    d_rows = [
        "2024-01-01 00:00:00,1,1,1,1,1",
        "2024-01-02 00:00:00,1,1,1,1,1",
    ]
    _write_csv(gold, "timestamp,open,high,low,close,volume", d_rows)
    _write_csv(silver, "timestamp,open,high,low,close,volume", d_rows)
    return usd, brent, eur, gold, silver


@pytest.fixture
def tiny_daily_causality_paths(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    usd = tmp_path / "usdjpy.csv"
    _write_csv(
        usd,
        "time,open,high,low,close,spread_pips",
        ["2024-01-16T10:00:00Z,1,1,1,1,0.5"],
    )
    h1_rows = ["2024-01-15 12:00:00,1,1,1,1,10"]
    brent = tmp_path / "bco.csv"
    eur = tmp_path / "eur.csv"
    _write_csv(brent, "timestamp,open,high,low,close,volume", h1_rows)
    _write_csv(eur, "timestamp,open,high,low,close,volume", h1_rows)
    gold = tmp_path / "gold.csv"
    silver = tmp_path / "silver.csv"
    d_rows = [
        "2024-01-14 00:00:00,1,1,1,1,1",
        "2024-01-15 00:00:00,2,2,2,2,1",
    ]
    _write_csv(gold, "timestamp,open,high,low,close,volume", d_rows)
    _write_csv(silver, "timestamp,open,high,low,close,volume", d_rows)
    return usd, brent, eur, gold, silver


@requires_full_dataset
def test_loads_all_five_files_without_error() -> None:
    CrossAssetDataLoader(
        str(USDJPY_PATH),
        str(BRENT_PATH),
        str(EUR_PATH),
        str(GOLD_PATH),
        str(SILVER_PATH),
    )


@requires_full_dataset
def test_bar_counts_match_expected() -> None:
    ld = CrossAssetDataLoader(
        str(USDJPY_PATH),
        str(BRENT_PATH),
        str(EUR_PATH),
        str(GOLD_PATH),
        str(SILVER_PATH),
    )
    u = ld.get_usdjpy_bars()
    assert 999_000 <= len(u) <= 1_000_100
    assert len(ld._brent_bars) == 15_419
    assert len(ld._eur_bars) == 16_801
    assert len(ld._gold_bars) == 699
    assert len(ld._silver_bars) == 699


def test_usdjpy_timestamp_parsing() -> None:
    dt = _parse_usdjpy_time("2023-06-20T18:28:00Z")
    assert dt == datetime(2023, 6, 20, 18, 28, tzinfo=timezone.utc)


def test_cross_asset_timestamp_parsing() -> None:
    dt = _parse_cross_timestamp_flexible("2023-06-20 00:00:00")
    assert dt == datetime(2023, 6, 20, 0, 0, tzinfo=timezone.utc)


def test_h1_causality_known_bar_at_14_00(tiny_h1_causality_paths: tuple) -> None:
    usd, brent, eur, gold, silver = tiny_h1_causality_paths
    ld = CrossAssetDataLoader(str(usd), str(brent), str(eur), str(gold), str(silver))

    q_1430 = datetime(2024, 1, 1, 14, 30, tzinfo=timezone.utc)
    b = ld.get_brent_at(q_1430)
    assert b is not None
    assert b["timestamp"] == datetime(2024, 1, 1, 13, 0, tzinfo=timezone.utc)
    assert b["completed_at"] == datetime(2024, 1, 1, 14, 0, tzinfo=timezone.utc)

    q_1500 = datetime(2024, 1, 1, 15, 0, tzinfo=timezone.utc)
    b2 = ld.get_brent_at(q_1500)
    assert b2 is not None
    assert b2["timestamp"] == datetime(2024, 1, 1, 14, 0, tzinfo=timezone.utc)

    q_1501 = datetime(2024, 1, 1, 15, 1, tzinfo=timezone.utc)
    b3 = ld.get_brent_at(q_1501)
    assert b3 is not None
    assert b3["timestamp"] == datetime(2024, 1, 1, 14, 0, tzinfo=timezone.utc)


def test_daily_causality_gold_bar_jan_15(tiny_daily_causality_paths: tuple) -> None:
    usd, brent, eur, gold, silver = tiny_daily_causality_paths
    ld = CrossAssetDataLoader(str(usd), str(brent), str(eur), str(gold), str(silver))

    q_morning = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
    g = ld.get_gold_at(q_morning)
    assert g is not None
    assert g["timestamp"].date().isoformat() == "2024-01-14"

    q_2200 = datetime(2024, 1, 15, 22, 0, tzinfo=timezone.utc)
    g2 = ld.get_gold_at(q_2200)
    assert g2 is not None
    assert g2["timestamp"].date().isoformat() == "2024-01-15"

    q_next = datetime(2024, 1, 16, 5, 0, tzinfo=timezone.utc)
    g3 = ld.get_gold_at(q_next)
    assert g3 is not None
    assert g3["timestamp"].date().isoformat() == "2024-01-15"


def test_brent_history_exactly_twenty_when_available(tmp_path: Path) -> None:
    usd = tmp_path / "u.csv"
    _write_csv(usd, "time,open,high,low,close,spread_pips", ["2024-01-02T12:00:00Z,1,1,1,1,1"])
    h1_lines = [f"2024-01-01 {h:02d}:00:00,1,1,1,1,1" for h in range(0, 24)]
    h1_lines += [f"2024-01-02 {h:02d}:00:00,1,1,1,1,1" for h in range(0, 12)]
    brent = tmp_path / "b.csv"
    eur = tmp_path / "e.csv"
    _write_csv(brent, "timestamp,open,high,low,close,volume", h1_lines)
    _write_csv(eur, "timestamp,open,high,low,close,volume", h1_lines)
    g = tmp_path / "g.csv"
    s = tmp_path / "s.csv"
    d = ["2024-01-01 00:00:00,1,1,1,1,1", "2024-01-02 00:00:00,1,1,1,1,1"]
    _write_csv(g, "timestamp,open,high,low,close,volume", d)
    _write_csv(s, "timestamp,open,high,low,close,volume", d)
    ld = CrossAssetDataLoader(str(usd), str(brent), str(eur), str(g), str(s))
    q = datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc)
    hist = ld.get_brent_history(q, 20)
    assert len(hist) == 20
    assert hist[0]["timestamp"] < hist[-1]["timestamp"]


def test_brent_history_fewer_when_insufficient(tmp_path: Path) -> None:
    usd = tmp_path / "u.csv"
    _write_csv(usd, "time,open,high,low,close,spread_pips", ["2024-01-01T15:00:00Z,1,1,1,1,1"])
    h1_lines = [
        "2024-01-01 12:00:00,1,1,1,1,1",
        "2024-01-01 13:00:00,1,1,1,1,1",
        "2024-01-01 14:00:00,1,1,1,1,1",
    ]
    brent = tmp_path / "b.csv"
    eur = tmp_path / "e.csv"
    _write_csv(brent, "timestamp,open,high,low,close,volume", h1_lines)
    _write_csv(eur, "timestamp,open,high,low,close,volume", h1_lines)
    g = tmp_path / "g.csv"
    s = tmp_path / "s.csv"
    d = ["2024-01-01 00:00:00,1,1,1,1,1"]
    _write_csv(g, "timestamp,open,high,low,close,volume", d)
    _write_csv(s, "timestamp,open,high,low,close,volume", d)
    ld = CrossAssetDataLoader(str(usd), str(brent), str(eur), str(g), str(s))
    q = datetime(2024, 1, 1, 15, 0, tzinfo=timezone.utc)
    hist = ld.get_brent_history(q, 20)
    assert len(hist) == 3
    assert hist[0]["timestamp"] == datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    assert hist[-1]["timestamp"] == datetime(2024, 1, 1, 14, 0, tzinfo=timezone.utc)


def test_query_before_any_cross_asset_data(tmp_path: Path) -> None:
    usd = tmp_path / "u.csv"
    _write_csv(usd, "time,open,high,low,close,spread_pips", ["2020-01-01T12:00:00Z,1,1,1,1,1"])
    brent = tmp_path / "b.csv"
    eur = tmp_path / "e.csv"
    _write_csv(brent, "timestamp,open,high,low,close,volume", ["2024-01-01 12:00:00,1,1,1,1,1"])
    _write_csv(eur, "timestamp,open,high,low,close,volume", ["2024-01-01 12:00:00,1,1,1,1,1"])
    g = tmp_path / "g.csv"
    s = tmp_path / "s.csv"
    _write_csv(g, "timestamp,open,high,low,close,volume", ["2024-01-01 00:00:00,1,1,1,1,1"])
    _write_csv(s, "timestamp,open,high,low,close,volume", ["2024-01-01 00:00:00,1,1,1,1,1"])
    ld = CrossAssetDataLoader(str(usd), str(brent), str(eur), str(g), str(s))
    q = datetime(2020, 1, 1, 12, 0, tzinfo=timezone.utc)
    assert ld.get_brent_at(q) is None
    assert ld.get_brent_history(q, 5) == []


def test_performance_ten_thousand_lookups(tmp_path: Path) -> None:
    usd = tmp_path / "u.csv"
    _write_csv(usd, "time,open,high,low,close,spread_pips", ["2024-06-01T12:00:00Z,1,1,1,1,1"])
    t0h = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    h1_lines = [(t0h + timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S") + ",1,1,1,1,1" for i in range(4000)]
    brent = tmp_path / "b.csv"
    eur = tmp_path / "e.csv"
    _write_csv(brent, "timestamp,open,high,low,close,volume", h1_lines)
    _write_csv(eur, "timestamp,open,high,low,close,volume", h1_lines)
    dlines = [f"2024-{m:02d}-01 00:00:00,1,1,1,1,1" for m in range(1, 8)]
    g = tmp_path / "g.csv"
    s = tmp_path / "s.csv"
    _write_csv(g, "timestamp,open,high,low,close,volume", dlines)
    _write_csv(s, "timestamp,open,high,low,close,volume", dlines)
    ld = CrossAssetDataLoader(str(usd), str(brent), str(eur), str(g), str(s))
    base = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)
    t0 = time.perf_counter()
    for i in range(10_000):
        ld.get_brent_at(base.replace(minute=i % 60))
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0, f"10k lookups took {elapsed:.3f}s"


@requires_full_dataset
def test_usdjpy_rows_have_required_fields() -> None:
    ld = CrossAssetDataLoader(
        str(USDJPY_PATH),
        str(BRENT_PATH),
        str(EUR_PATH),
        str(GOLD_PATH),
        str(SILVER_PATH),
    )
    row = ld.get_usdjpy_bars()[0]
    for k in ("time", "open", "high", "low", "close", "spread_pips"):
        assert k in row


@requires_full_dataset
def test_cross_asset_bars_have_required_fields() -> None:
    ld = CrossAssetDataLoader(
        str(USDJPY_PATH),
        str(BRENT_PATH),
        str(EUR_PATH),
        str(GOLD_PATH),
        str(SILVER_PATH),
    )
    ts = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)
    b = ld.get_brent_at(ts)
    assert b is not None
    for k in ("timestamp", "open", "high", "low", "close", "volume", "completed_at"):
        assert k in b


def test_init_missing_file_raises(tmp_path: Path) -> None:
    usd = tmp_path / "missing.csv"
    brent = tmp_path / "b.csv"
    _write_csv(brent, "timestamp,open,high,low,close,volume", ["2024-01-01 00:00:00,1,1,1,1,1"])
    eur, g, s = brent, brent, brent
    with pytest.raises(FileNotFoundError):
        CrossAssetDataLoader(str(usd), str(brent), str(eur), str(g), str(s))


def test_init_empty_cross_file_raises(tmp_path: Path) -> None:
    usd = tmp_path / "u.csv"
    _write_csv(usd, "time,open,high,low,close,spread_pips", ["2024-01-01T12:00:00Z,1,1,1,1,1"])
    brent = tmp_path / "b.csv"
    brent.write_text("timestamp,open,high,low,close,volume\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty|valid rows"):
        CrossAssetDataLoader(str(usd), str(brent), str(brent), str(brent), str(brent))


def test_cross_asset_bar_dataclass_completion_helpers() -> None:
    o = datetime(2024, 1, 15, 14, 0, tzinfo=timezone.utc)
    assert _h1_completed_at(o) == datetime(2024, 1, 15, 15, 0, tzinfo=timezone.utc)
    d0 = datetime(2024, 1, 15, 0, 0, tzinfo=timezone.utc)
    assert _daily_completed_at(d0) == datetime(2024, 1, 15, 22, 0, tzinfo=timezone.utc)
