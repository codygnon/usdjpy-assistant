from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.regime_backtest_engine import (
    AdmissionConfig,
    BacktestEngine,
    DummyStrategy,
    FixedSpreadConfig,
    InstrumentSpec,
    RunConfig,
    RunManifest,
    SlippageConfig,
    SpreadConfig,
    WalkForwardConfig,
    WalkForwardRunner,
    WalkForwardWindow,
    build_walk_forward_windows,
)
from core.regime_backtest_engine.strategy import HistoricalDataView


class FitAwareDummyStrategy(DummyStrategy):
    def __init__(self, fit_lengths: list[int], **kwargs) -> None:
        super().__init__(**kwargs)
        self._fit_lengths = fit_lengths

    def fit(self, history: HistoricalDataView) -> None:
        self._fit_lengths.append(len(history))


def _write_mid_csv(path: Path, bars: int = 40) -> Path:
    rows = []
    price = 150.0
    ts = pd.Timestamp("2025-01-01T00:00:00Z")
    for i in range(bars):
        open_px = price
        close_px = price + (0.03 if i % 2 == 0 else -0.01)
        high_px = max(open_px, close_px) + 0.08
        low_px = min(open_px, close_px) - 0.08
        rows.append(
            {
                "timestamp": ts + pd.Timedelta(minutes=i),
                "open": open_px,
                "high": high_px,
                "low": low_px,
                "close": close_px,
            }
        )
        price = close_px
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _base_config(tmp_path: Path, data_path: Path, manifest: RunManifest | None = None) -> RunConfig:
    return RunConfig(
        data_path=data_path,
        output_dir=tmp_path / "engine_run",
        mode="standalone",
        active_families=("dummy",),
        instrument=InstrumentSpec(symbol="USDJPY"),
        spread=SpreadConfig(spread_source="fixed", fixed=FixedSpreadConfig(spread_pips=1.2)),
        slippage=SlippageConfig(fixed_slippage_pips=0.1),
        admission=AdmissionConfig(
            allow_opposing_exposure=False,
            max_total_open_positions=3,
            max_open_positions_per_family={"dummy": 1},
            max_total_units=50_000,
            max_units_per_family={"dummy": 10_000},
            family_priority=("dummy",),
        ),
        manifest=manifest,
        bar_log_format="csv",
    )


def test_engine_manifest_is_frozen_and_evaluated(tmp_path: Path) -> None:
    data_path = _write_mid_csv(tmp_path / "bars.csv", bars=30)
    manifest = RunManifest(
        hypothesis="dummy_phase3_manifest",
        minimum_trade_count=100,
        minimum_profit_factor=50.0,
        maximum_drawdown_usd=1000.0,
        expected_win_rate_min=0.0,
        expected_win_rate_max=100.0,
    )
    engine = BacktestEngine({"dummy": DummyStrategy(family_name="dummy", every_n_bars=5, direction="long")})
    result = engine.run(_base_config(tmp_path, data_path, manifest=manifest))

    assert result.manifest_path is not None
    assert result.manifest_path.exists()
    assert "manifest_evaluation" in result.summary
    assert result.summary["manifest_evaluation"]["overall_pass"] is False
    assert result.summary["manifest_evaluation"]["checks"]["minimum_trade_count"]["passed"] is False


def test_walk_forward_aggregates_out_of_sample_only_and_fits_on_in_sample(tmp_path: Path) -> None:
    data_path = _write_mid_csv(tmp_path / "bars_wf.csv", bars=30)
    fit_lengths: list[int] = []
    manifest = RunManifest(
        hypothesis="wf_dummy_validation",
        minimum_trade_count=1,
        minimum_profit_factor=0.1,
        maximum_drawdown_usd=100_000.0,
        expected_win_rate_min=0.0,
        expected_win_rate_max=100.0,
    )
    base_config = _base_config(tmp_path, data_path, manifest=manifest)
    wf_config = WalkForwardConfig(
        windows=build_walk_forward_windows(
            total_bars=30,
            in_sample_bars=10,
            out_sample_bars=5,
            step_bars=5,
            anchored=False,
        ),
        output_dir=tmp_path / "wf_run",
    )
    runner = WalkForwardRunner(
        {
            "dummy": lambda: FitAwareDummyStrategy(
                fit_lengths=fit_lengths,
                family_name="dummy",
                every_n_bars=4,
                direction="long",
            )
        }
    )

    result = runner.run(base_config, wf_config)

    assert result.manifest_path is not None
    assert result.summary_path.exists()
    assert result.segments_path.exists()
    assert result.summary["aggregate_scope"] == "out_of_sample_only"
    assert result.summary["processed_bar_count"] == 20
    assert fit_lengths == [10] * 8
    assert len(result.segment_results) == 4
    assert "manifest_evaluation" in result.summary


def test_build_walk_forward_windows_requires_valid_windows() -> None:
    windows = build_walk_forward_windows(total_bars=25, in_sample_bars=10, out_sample_bars=5, step_bars=5, anchored=True)
    assert windows[0] == WalkForwardWindow(
        label="wf_001",
        in_sample_start_index=0,
        in_sample_end_index=9,
        out_sample_start_index=10,
        out_sample_end_index=14,
    )
    assert windows[1].in_sample_end_index == 14
