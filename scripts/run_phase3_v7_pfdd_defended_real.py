#!/usr/bin/env python3
"""Run Phase3 v7_pfdd defended simulation via RunConfig + Phase3V7PfddDefendedBacktestEngine.

Inner spread: L1 uses the v2_exp4 realistic execution profile (see --spread-mode realistic
in core.phase3_v7_pfdd_defended_runner). RunConfig spread/slippage apply only to metadata;
the defended runner owns Tokyo/London/V44 mechanics.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
from scripts.run_daily_trend_real import _extend_summary


def main() -> None:
    uj_path = ROOT / "research_out/USDJPY_M1_OANDA_1000k.csv"
    if not uj_path.is_file():
        print(f"ERROR: missing USDJPY data file: {uj_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = ROOT / "research_out/phase3_v7_pfdd_defended_real"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run_log.txt"
    log_path.write_text("", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log("=== Phase3 v7_pfdd defended — realistic L1 spread (runner) ===")
    log(f"Started at {time.strftime('%Y-%m-%d %H:%M:%S')} UTC")

    cfg = RunConfig(
        hypothesis="phase3_v7_pfdd_defended_realistic",
        data_path=uj_path,
        output_dir=output_dir,
        mode="standalone",
        active_families=(PHASE3_V7_PFDD_FAMILY,),
        instrument=InstrumentSpec(symbol="USDJPY", margin_rate=(1.0 / 33.3)),
        spread=SpreadConfig(spread_source="fixed", fixed=FixedSpreadConfig(spread_pips=2.0)),
        slippage=SlippageConfig(fixed_slippage_pips=0.1),
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

    log("Starting backtest...")
    engine = Phase3V7PfddDefendedBacktestEngine()
    t0 = time.perf_counter()
    result = engine.run(cfg)
    elapsed = time.perf_counter() - t0
    log(f"Finished in {elapsed:.1f}s")

    trade_df = (
        pd.read_csv(result.trade_log_path) if result.trade_log_path.is_file() else pd.DataFrame()
    )
    bar_df = pd.read_csv(result.bar_log_path) if result.bar_log_path.is_file() else pd.DataFrame()
    ext = _extend_summary(dict(result.summary), trade_df, bar_df)
    ext["methodology_notes"] = (
        "Phase3 v7_pfdd defended via shared runner (Tokyo V14 → London V2 → V44 oracle). "
        "Hypothesis selects realistic L1 spread inside the runner."
    )
    ext["phase3_defended_runner_summary"] = result.summary.get("phase3_defended_runner_summary", {})

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(ext, indent=2, default=str), encoding="utf-8")
    log(f"\nSummary written to {summary_path}")
    log(json.dumps(ext, indent=2, default=str))
    log(f"\nTrade log: {result.trade_log_path}")


if __name__ == "__main__":
    main()
