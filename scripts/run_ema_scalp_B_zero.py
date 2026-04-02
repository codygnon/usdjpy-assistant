#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.regime_backtest_engine import BacktestEngine
from core.regime_backtest_engine.ema_scalp import (
    EMAScalpConfig,
    EMAScalpStrategy,
    build_ema_scalp_run_config,
)

FAMILY = "ema_scalp"
DEFAULT_DATA = ROOT / "research_out/USDJPY_M1_OANDA_500k.csv"
DEFAULT_OUT = ROOT / "research_out/ema_scalp_B_zero"


def main() -> None:
    data_path = Path(DEFAULT_DATA)
    output_dir = Path(DEFAULT_OUT)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = build_ema_scalp_run_config(
        hypothesis="ema_scalp_B_zero",
        data_path=data_path,
        output_dir=output_dir,
        family_name=FAMILY,
        variant="B",
        spread_pips=1e-9,
        slippage_pips=0.0,
    )
    strat = EMAScalpStrategy(FAMILY, EMAScalpConfig(variant="B"))
    result = BacktestEngine({FAMILY: strat}).run(cfg)
    print(json.dumps(result.summary, indent=2, default=str))
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(result.summary, indent=2, default=str), encoding="utf-8")
    print("trade_log:", result.trade_log_path)
    print("bar_log:", result.bar_log_path)
    print("summary:", summary_path)


if __name__ == "__main__":
    main()
