#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path("/Users/codygnon/Documents/usdjpy_assistant")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.phase3_regime_owned_backtest import _load_defended_overlays, run_regime_owned_backtest, write_backtest_package


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run unified regime-owned 500k M1 portfolio backtest")
    p.add_argument("--input", default="/Users/codygnon/Documents/usdjpy_assistant/research_out/USDJPY_M1_OANDA_500k.csv")
    p.add_argument("--tokyo-config", default="/Users/codygnon/Documents/usdjpy_assistant/research_out/tokyo_optimized_v14_config.json")
    p.add_argument("--london-config", default="/Users/codygnon/Documents/usdjpy_assistant/research_out/v2_exp4_winner_baseline_config.json")
    p.add_argument("--v44-config", default="/Users/codygnon/Documents/usdjpy_assistant/research_out/session_momentum_v44_base_config.json")
    p.add_argument("--starting-equity", type=float, default=100000.0)
    p.add_argument("--out-prefix", default="/Users/codygnon/Documents/usdjpy_assistant/research_out/regime_owned_portfolio_500k")
    p.add_argument("--defended-overlays", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    defended_overlays = _load_defended_overlays() if bool(args.defended_overlays) else None
    result = run_regime_owned_backtest(
        input_csv=args.input,
        tokyo_config_path=args.tokyo_config,
        london_config_path=args.london_config,
        v44_config_path=args.v44_config,
        start_equity=float(args.starting_equity),
        defended_overlays=defended_overlays,
        variant_name="regime_owned_plus_v7_defended_overlays" if defended_overlays else "regime_owned_base",
    )
    paths = write_backtest_package(result, args.out_prefix)
    print(json.dumps({"summary": result.summary, "artifacts": paths}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
