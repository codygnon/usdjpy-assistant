#!/usr/bin/env python3
"""Run Daily Trend + 4H pullback — near-zero spread (diagnostic)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from core.regime_backtest_engine.daily_trend_strategy import DailyTrendStrategy
from core.regime_backtest_engine.models import (
    AdmissionConfig,
    FixedSpreadConfig,
    InstrumentSpec,
    RunConfig,
    SlippageConfig,
    SpreadConfig,
)

from run_daily_trend_real import ProgressBacktestEngine, _experiment13_gates, _extend_summary


def main() -> None:
    uj_path = ROOT / "research_out/USDJPY_M1_OANDA_1000k.csv"
    if not uj_path.is_file():
        print(f"ERROR: missing USDJPY data file: {uj_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = ROOT / "research_out/daily_trend_zero"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run_log.txt"
    log_path.write_text("", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log("=== DAILY TREND + 4H PULLBACK (ZERO SPREAD) ===")
    log(f"Started at {time.strftime('%Y-%m-%d %H:%M:%S')} UTC")

    strategy = DailyTrendStrategy(
        daily_ema_fast=20,
        daily_ema_slow=50,
        daily_atr_period=14,
        proximity_atr=0.3,
        stop_atr_factor=1.5,
        trail_buffer_atr=0.5,
        pullback_lookback=5,
        trail_lookback=5,
        risk_fraction=0.01,
        account_balance=100_000.0,
        max_size=500_000,
    )

    family = strategy.family_name
    cfg = RunConfig(
        hypothesis="daily_trend_zero",
        data_path=uj_path,
        output_dir=output_dir,
        mode="standalone",
        active_families=(family,),
        instrument=InstrumentSpec(symbol="USDJPY", margin_rate=(1.0 / 33.3)),
        spread=SpreadConfig(spread_source="fixed", fixed=FixedSpreadConfig(spread_pips=0.000001)),
        slippage=SlippageConfig(fixed_slippage_pips=0.0),
        admission=AdmissionConfig(
            allow_opposing_exposure=False,
            max_total_open_positions=1,
            max_open_positions_per_family={family: 1},
            max_total_units=500_000,
            max_units_per_family={family: 500_000},
            family_priority=(family,),
        ),
        initial_balance=100_000.0,
        bar_log_format="csv",
    )

    log("Starting backtest...")
    engine = ProgressBacktestEngine({family: strategy})
    t0 = time.perf_counter()
    result = engine.run(cfg)
    elapsed = time.perf_counter() - t0
    log(f"Finished in {elapsed:.1f}s")

    trade_df = (
        pd.read_csv(result.trade_log_path) if result.trade_log_path.is_file() else pd.DataFrame()
    )
    bar_df = pd.read_csv(result.bar_log_path) if result.bar_log_path.is_file() else pd.DataFrame()
    ext = _extend_summary(dict(result.summary), trade_df, bar_df)
    ext["experiment13_gates"] = _experiment13_gates(ext)
    ext["methodology_notes"] = (
        "Daily trend + 4H pullback. Near-zero spread (1e-6 pip), zero slippage."
    )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(ext, indent=2, default=str), encoding="utf-8")
    log(f"\nSummary written to {summary_path}")
    log(json.dumps(ext, indent=2, default=str))
    log(f"\nTrade log: {result.trade_log_path}")


if __name__ == "__main__":
    main()
