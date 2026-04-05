#!/usr/bin/env python3
"""Fast-iteration version of run_v7_h1h2_improved.py.

Accepts --bars N to limit the number of M1 bars processed, enabling
rapid directional testing of filter changes (~20% runtime for 200k bars).

Usage:
    python3 scripts/run_v7_h1h2_fast_iter.py --bars 200000
    python3 scripts/run_v7_h1h2_fast_iter.py                # full dataset

The engine supports max_bars via the hypothesis name pattern '_maxbars_N'.
This script injects that pattern and also truncates the M5 filter context
to match, so filter lookups stay aligned with the bar window.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import phase3_v7_pfdd_defended_runner as base_runner
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


PIP_SIZE = 0.01
H1_ATR14_MAX_PIPS = 7.04
H2_EMA_SPREAD_MAX_PRICE = 0.030


@dataclass
class FilterStats:
    v44_oracle_original: int = 0
    h1_v44_blocked: int = 0
    h1_v44_kept: int = 0
    h2_setup_a_blocked: int = 0
    h2_missing_context: int = 0


class M5FilterContext:
    def __init__(self, dataset_path: Path) -> None:
        m1 = pd.read_csv(dataset_path)
        m1["time"] = pd.to_datetime(m1["time"], utc=True)
        m1 = m1.sort_values("time").reset_index(drop=True)
        m1 = m1.set_index("time")
        m5 = (
            m1.resample("5min")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
            .dropna()
            .copy()
        )
        prev_close = m5["close"].shift(1)
        tr = pd.concat(
            [
                m5["high"] - m5["low"],
                (m5["high"] - prev_close).abs(),
                (m5["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        m5["atr14_pips"] = tr.rolling(14, min_periods=14).mean() / PIP_SIZE
        m5["ema9"] = m5["close"].ewm(span=9, adjust=False).mean()
        m5["ema21"] = m5["close"].ewm(span=21, adjust=False).mean()
        self._frame = m5
        self._times = m5.index.asi8

    def row_at_or_before(self, ts: pd.Timestamp) -> pd.Series | None:
        key = pd.Timestamp(ts).tz_convert("UTC").value // 1000
        pos = int(np.searchsorted(self._times, key, side="right") - 1)
        if pos < 0:
            return None
        return self._frame.iloc[pos]


def main() -> None:
    parser = argparse.ArgumentParser(description="V7 H1+H2 fast iteration runner")
    parser.add_argument(
        "--bars", type=int, default=None,
        help="Max M1 bars to process (e.g. 200000 for ~1yr). Default: full dataset.",
    )
    args = parser.parse_args()

    uj_path = ROOT / "research_out/USDJPY_M1_OANDA_extended.csv"
    if not uj_path.is_file():
        print(f"ERROR: missing USDJPY data file: {uj_path}", file=sys.stderr)
        sys.exit(1)

    max_bars = args.bars
    suffix = f"_maxbars_{max_bars}" if max_bars else ""
    output_dir = ROOT / f"research_out/v7_h1h2_fast_iter{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run_log.txt"
    log_path.write_text("", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    bars_desc = f"{max_bars:,} bars" if max_bars else "full dataset"
    log(f"=== Phase3 v7 defended H1+H2 fast-iter — {bars_desc} ===")
    log(f"Started at {time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    log(f"Dataset: {uj_path}")
    log("Building M5 filter context...")
    filter_context = M5FilterContext(uj_path)
    filter_stats = FilterStats()

    # --- Monkeypatch H1: V44 high-vol block ---
    original_load_v44_oracle = base_runner.load_v44_oracle

    def filtered_load_v44_oracle(dataset_path: str) -> dict[int, dict[str, Any]]:
        raw = original_load_v44_oracle(dataset_path)
        filter_stats.v44_oracle_original = len(raw)
        filtered: dict[int, dict[str, Any]] = {}
        blocked_local = 0
        for key, row in raw.items():
            ts = base_runner._ts(row["entry_time"])
            ctx = filter_context.row_at_or_before(ts)
            if ctx is not None and pd.notna(ctx["atr14_pips"]) and float(ctx["atr14_pips"]) > H1_ATR14_MAX_PIPS:
                blocked_local += 1
                continue
            filtered[key] = row
        filter_stats.h1_v44_blocked = blocked_local
        filter_stats.h1_v44_kept = len(filtered)
        return filtered

    # --- Monkeypatch H2: London Setup A with-trend block ---
    original_advance_london = base_runner.advance_london_unified_bar

    def filtered_advance_london_unified_bar(*args: Any, **kwargs: Any):
        original_exec_gate = kwargs.get("exec_gate")

        def wrapped_exec_gate(pe: dict[str, Any], t: pd.Timestamp) -> tuple[bool, str]:
            if original_exec_gate is not None:
                ok, reason = original_exec_gate(pe, t)
                if not ok:
                    return ok, reason
            if str(pe.get("setup_type")) == "A":
                ctx = filter_context.row_at_or_before(pd.Timestamp(t))
                if ctx is None or pd.isna(ctx["ema9"]) or pd.isna(ctx["ema21"]):
                    filter_stats.h2_missing_context += 1
                    return True, "ok"
                ema9 = float(ctx["ema9"])
                ema21 = float(ctx["ema21"])
                spread = abs(ema9 - ema21)
                direction = str(pe.get("direction", "long")).lower()
                aligns = (direction == "long" and ema9 > ema21) or (direction == "short" and ema9 < ema21)
                if spread > H2_EMA_SPREAD_MAX_PRICE and aligns:
                    filter_stats.h2_setup_a_blocked += 1
                    return False, "h2_setup_a_with_trend"
            return True, "ok"

        kwargs["exec_gate"] = wrapped_exec_gate
        return original_advance_london(*args, **kwargs)

    base_runner.load_v44_oracle = filtered_load_v44_oracle
    base_runner.advance_london_unified_bar = filtered_advance_london_unified_bar

    # Build hypothesis name with maxbars tag so engine picks it up
    hypothesis = f"phase3_v7_pfdd_h1h2_improved_realistic{suffix}"

    cfg = RunConfig(
        hypothesis=hypothesis,
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

    log(f"Starting backtest (hypothesis={hypothesis})...")
    engine = Phase3V7PfddDefendedBacktestEngine()
    t0 = time.perf_counter()
    result = engine.run(cfg)
    elapsed = time.perf_counter() - t0
    log(f"Finished in {elapsed:.1f}s ({elapsed/60:.1f}m)")

    trade_df = pd.read_csv(result.trade_log_path) if result.trade_log_path.is_file() else pd.DataFrame()
    bar_df = pd.read_csv(result.bar_log_path) if result.bar_log_path.is_file() else pd.DataFrame()
    ext = _extend_summary(dict(result.summary), trade_df, bar_df)
    ext["methodology_notes"] = (
        f"Phase3 v7 defended H1+H2 fast-iter ({bars_desc}). "
        "H1 blocks V44 oracle entries when M5 ATR-14 > 7.04 pips. "
        "H2 blocks London Setup A entries when EMA spread > 3 pips with-trend."
    )
    ext["h1h2_filter_stats"] = {
        "v44_oracle_original": filter_stats.v44_oracle_original,
        "h1_v44_blocked": filter_stats.h1_v44_blocked,
        "h1_v44_kept": filter_stats.h1_v44_kept,
        "h2_setup_a_blocked": filter_stats.h2_setup_a_blocked,
        "h2_missing_context": filter_stats.h2_missing_context,
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(ext, indent=2, default=str), encoding="utf-8")

    log(f"\nSummary written to {summary_path}")
    log(json.dumps(ext, indent=2, default=str))
    log(f"\nTrade log: {result.trade_log_path}")


if __name__ == "__main__":
    main()
