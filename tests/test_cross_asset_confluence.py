from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.regime_backtest_engine.cross_asset_confluence import (
    CrossAssetConfluenceConfig,
    CrossAssetConfluenceStrategy,
    _TimeCloseIndex,
    load_cross_asset_bundle,
)


def _write_minimal_cross_csv(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    h1_times = pd.date_range("2023-06-20", periods=50, freq="h", tz="UTC")
    d_times = pd.date_range("2023-06-20", periods=30, freq="D", tz="UTC")
    for name, times in [
        ("BCO_USD_H1_OANDA.csv", h1_times),
        ("EUR_USD_H1_OANDA.csv", h1_times),
    ]:
        df = pd.DataFrame(
            {
                "timestamp": times.strftime("%Y-%m-%d %H:%M:%S"),
                "open": np.linspace(80.0, 85.0, len(times)),
                "high": np.linspace(81.0, 86.0, len(times)),
                "low": np.linspace(79.0, 84.0, len(times)),
                "close": np.linspace(80.5, 85.5, len(times)),
                "volume": 1000,
            }
        )
        df.to_csv(root / name, index=False)
    for name, times in [
        ("XAU_USD_D_OANDA.csv", d_times),
        ("XAG_USD_D_OANDA.csv", d_times),
    ]:
        df = pd.DataFrame(
            {
                "timestamp": times.strftime("%Y-%m-%d %H:%M:%S"),
                "open": np.linspace(1900.0, 1950.0, len(times)),
                "high": np.linspace(1910.0, 1960.0, len(times)),
                "low": np.linspace(1890.0, 1940.0, len(times)),
                "close": np.linspace(1920.0, 1955.0, len(times)),
                "volume": 500,
            }
        )
        df.to_csv(root / name, index=False)


def test_time_close_index_asof_and_return(tmp_path: Path) -> None:
    p = tmp_path / "s.csv"
    df = pd.DataFrame(
        {
            "timestamp": ["2023-01-01 00:00:00", "2023-01-01 01:00:00", "2023-01-01 02:00:00"],
            "open": [1.0, 1.0, 1.0],
            "high": [1.1, 1.1, 1.1],
            "low": [0.9, 0.9, 0.9],
            "close": [100.0, 110.0, 121.0],
            "volume": [1, 1, 1],
        }
    )
    df.to_csv(p, index=False)
    idx = _TimeCloseIndex.from_csv(p)
    t2 = pd.Timestamp("2023-01-01 02:30:00", tz="UTC")
    ret = idx.log_return_over_bars(int(t2.value), 2)
    assert ret is not None
    assert ret > 0  # 121 vs 100


def test_load_cross_asset_bundle(tmp_path: Path) -> None:
    _write_minimal_cross_csv(tmp_path)
    b = load_cross_asset_bundle(tmp_path)
    assert len(b.bco.times_ns) == 50
    assert len(b.xau.times_ns) == 30


def test_strategy_construct_with_explicit_dir(tmp_path: Path) -> None:
    _write_minimal_cross_csv(tmp_path)
    cfg = CrossAssetConfluenceConfig(
        cross_assets_dir=tmp_path,
        min_warmup_1m_bars=0,
        min_long_score=4,
        max_short_score=-4,
    )
    s = CrossAssetConfluenceStrategy(cfg)
    assert s.family_name == "cross_asset_confluence"


_REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(
    not (_REPO_ROOT / "research_out/cross_assets/BCO_USD_H1_OANDA.csv").is_file(),
    reason="full cross-asset download not present",
)
def test_load_default_research_out_bundle() -> None:
    b = load_cross_asset_bundle()
    assert len(b.bco.close) > 100
