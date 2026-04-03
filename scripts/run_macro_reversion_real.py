from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.regime_backtest_engine.engine import BacktestEngine
from core.regime_backtest_engine.macro_reversion import MacroReversionConfig, MacroReversionStrategy
from core.regime_backtest_engine.models import AdmissionConfig, FixedSpreadConfig, InstrumentSpec, RunConfig, SlippageConfig, SpreadConfig

DATA_PATH = ROOT / "research_out" / "USDJPY_M1_OANDA_1000k.csv"
OUTPUT_DIR = ROOT / "research_out" / "macro_reversion_real"


def _benchmark_evaluation(summary: dict) -> dict[str, object]:
    trade_count = int(summary.get("trade_count") or 0)
    if trade_count < 100:
        return {
            "status": "INSUFFICIENT DATA",
            "reason": "trade_count < 100",
            "checks": {
                "trade_count": trade_count,
                "win_rate": summary.get("win_rate"),
                "profit_factor": summary.get("profit_factor"),
                "max_drawdown_usd": summary.get("max_drawdown_usd"),
                "max_drawdown_pct": summary.get("max_drawdown_pct"),
            },
        }

    checks = {
        "trade_count_gte_100": trade_count >= 100,
        "win_rate_gte_55": float(summary.get("win_rate") or 0.0) >= 55.0,
        "profit_factor_gte_1_50": float(summary.get("profit_factor") or 0.0) >= 1.50,
        "max_drawdown_pct_lte_20": float(summary.get("max_drawdown_pct") or 0.0) <= 20.0,
        "max_drawdown_usd_lte_20000": float(summary.get("max_drawdown_usd") or 0.0) <= 20000.0,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    strategy = MacroReversionStrategy(MacroReversionConfig())
    config = RunConfig(
        hypothesis="macro_reversion_real",
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR,
        mode="standalone",
        active_families=(strategy.family_name,),
        instrument=InstrumentSpec(symbol="USDJPY", margin_rate=(1.0 / 33.3)),
        spread=SpreadConfig(spread_source="fixed", fixed=FixedSpreadConfig(spread_pips=2.0)),
        slippage=SlippageConfig(fixed_slippage_pips=0.1),
        admission=AdmissionConfig(
            allow_opposing_exposure=False,
            max_total_open_positions=1,
            max_open_positions_per_family={strategy.family_name: 1},
            max_total_units=200_000,
            max_units_per_family={strategy.family_name: 200_000},
            family_priority=(strategy.family_name,),
        ),
        initial_balance=100000.0,
        bar_log_format="csv",
    )
    result = BacktestEngine({strategy.family_name: strategy}).run(config)
    summary = dict(result.summary)
    summary["benchmark_evaluation"] = _benchmark_evaluation(summary)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
