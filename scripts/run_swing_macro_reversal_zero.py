#!/usr/bin/env python3
"""Run Swing-Macro REVERSAL BAR filter backtest — near-zero spread (diagnostic)."""

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

from core.regime_backtest_engine.models import (
    AdmissionConfig,
    FixedSpreadConfig,
    InstrumentSpec,
    RunConfig,
    SlippageConfig,
    SpreadConfig,
)
from core.regime_backtest_engine.swing_macro_signals import WeeklyMacroSignal
from core.regime_backtest_engine.swing_macro_strategy import SwingMacroStrategy

from run_swing_macro_reversal_real import (
    ProgressBacktestEngine,
    _benchmark_evaluation,
    _extend_summary,
    _load_h1_as_daily,
)


def main() -> None:
    uj_path = ROOT / "research_out/USDJPY_M1_OANDA_1000k.csv"
    cross = ROOT / "research_out/cross_assets"
    oil_path = cross / "BCO_USD_H1_OANDA.csv"
    eurusd_path = cross / "EUR_USD_H1_OANDA.csv"

    for label, p in [("USDJPY", uj_path), ("Oil", oil_path), ("EUR/USD", eurusd_path)]:
        if not p.is_file():
            print(f"ERROR: missing {label} data file: {p}", file=sys.stderr)
            sys.exit(1)

    output_dir = ROOT / "research_out/swing_macro_reversal_zero"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run_log.txt"
    log_path.write_text("", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log("=== SWING-MACRO REVERSAL BAR — ZERO SPREAD ===")
    log(f"Started at {time.strftime('%Y-%m-%d %H:%M:%S')} UTC")

    log("Loading cross-asset data...")
    oil_daily = _load_h1_as_daily(oil_path)
    eurusd_daily = _load_h1_as_daily(eurusd_path)
    log(f"  Oil daily bars: {len(oil_daily)}")
    log(f"  EUR/USD daily bars: {len(eurusd_daily)}")

    macro = WeeklyMacroSignal(oil_daily=oil_daily, eurusd_daily=eurusd_daily)
    strategy = SwingMacroStrategy(
        macro_signal=macro,
        account_balance=100_000.0,
        risk_per_trade=0.01,
        atr_stop_factor=1.5,
        atr_proximity_factor=0.3,
        trailing_swing_lookback=5,
        trail_buffer_atr=0.5,
        cooldown_bars_4h=2,
        allow_lean=True,
        max_size=500_000,
        require_reversal_bar=True,
    )

    family = strategy.family_name
    cfg = RunConfig(
        hypothesis="swing_macro_reversal_zero",
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
    ext["benchmark_evaluation"] = _benchmark_evaluation(ext)
    ext["methodology_notes"] = (
        "REVERSAL BAR: require_reversal_bar=True, baseline stops (1.5× ATR, 0.5 trail). "
        "Near-zero spread (1e-6 pip), zero slippage."
    )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(ext, indent=2, default=str), encoding="utf-8")
    log(f"\nSummary written to {summary_path}")
    log(json.dumps(ext, indent=2, default=str))
    log(f"\nTrade log: {result.trade_log_path}")


if __name__ == "__main__":
    main()
