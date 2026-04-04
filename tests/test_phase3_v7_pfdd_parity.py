from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _research_fixtures_ready() -> bool:
    paths = [
        ROOT / "research_out/USDJPY_M1_OANDA_1000k.csv",
        ROOT / "research_out/tokyo_optimized_v14_config.json",
        ROOT / "research_out/v2_exp4_winner_baseline_config.json",
        ROOT / "research_out/session_momentum_v44_base_config.json",
    ]
    return all(p.is_file() for p in paths)


@pytest.mark.skipif(not _research_fixtures_ready(), reason="research_out Phase3 inputs missing")
def test_phase3_execute_is_deterministic(tmp_path: Path) -> None:
    from core.phase3_v7_pfdd_defended_runner import Phase3V7PfddParams, execute_phase3_v7_pfdd_defended

    uj = ROOT / "research_out/USDJPY_M1_OANDA_1000k.csv"
    params = Phase3V7PfddParams(data_path=str(uj), spread_mode="pipeline", max_bars=800, quiet=True)
    a = execute_phase3_v7_pfdd_defended(params)
    b = execute_phase3_v7_pfdd_defended(params)
    assert a["summary"]["net_pnl_usd"] == b["summary"]["net_pnl_usd"]
    assert a["summary"]["closed_trades_total"] == b["summary"]["closed_trades_total"]


@pytest.mark.skipif(not _research_fixtures_ready(), reason="research_out Phase3 inputs missing")
def test_phase3_engine_matches_runner_summary(tmp_path: Path) -> None:
    from core.phase3_v7_pfdd_defended_runner import Phase3V7PfddParams, execute_phase3_v7_pfdd_defended
    from core.regime_backtest_engine import (
        AdmissionConfig,
        FixedSpreadConfig,
        InstrumentSpec,
        PHASE3_V7_PFDD_FAMILY,
        Phase3V7PfddDefendedBacktestEngine,
        RunConfig,
        SlippageConfig,
        SpreadConfig,
    )

    uj = ROOT / "research_out/USDJPY_M1_OANDA_1000k.csv"
    params = Phase3V7PfddParams(data_path=str(uj), spread_mode="pipeline", max_bars=800, quiet=True)
    raw = execute_phase3_v7_pfdd_defended(params)

    out_dir = tmp_path / "phase3_engine_out"
    cfg = RunConfig(
        hypothesis="phase3_v7_pfdd_defended_pipeline_maxbars_800",
        data_path=uj,
        output_dir=out_dir,
        mode="standalone",
        active_families=(PHASE3_V7_PFDD_FAMILY,),
        instrument=InstrumentSpec(symbol="USDJPY", margin_rate=(1.0 / 33.3)),
        spread=SpreadConfig(spread_source="fixed", fixed=FixedSpreadConfig(spread_pips=1.0)),
        slippage=SlippageConfig(fixed_slippage_pips=0.0),
        admission=AdmissionConfig(
            allow_opposing_exposure=False,
            max_total_open_positions=100,
            max_open_positions_per_family={PHASE3_V7_PFDD_FAMILY: 100},
            max_total_units=50_000_000,
            max_units_per_family={PHASE3_V7_PFDD_FAMILY: 50_000_000},
            family_priority=(PHASE3_V7_PFDD_FAMILY,),
        ),
        initial_balance=100_000.0,
        bar_log_format="csv",
    )
    eng = Phase3V7PfddDefendedBacktestEngine()
    result = eng.run(cfg)
    inner = result.summary["phase3_defended_runner_summary"]
    assert float(inner["net_pnl_usd"]) == pytest.approx(float(raw["summary"]["net_pnl_usd"]), rel=0, abs=1e-6)
    assert int(inner["closed_trades_total"]) == int(raw["summary"]["closed_trades_total"])
