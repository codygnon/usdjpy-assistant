#!/usr/bin/env python3
"""Run the Phase3 v7 defended strategy with additive H1+H2 filters.

H1: block V44 NY oracle entries when M5 ATR-14 > 7.04 pips.
H2: block London Setup A entries when M5 EMA-9/EMA-21 are > 3 pips apart and
the entry aligns with that trend.

This script does not modify any files under core/. It monkeypatches the exact
runner hooks used during the run, then restores nothing because the process is
single-use.
"""

from __future__ import annotations

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
        # Match the replay-analysis enrichment exactly:
        # load raw M1, set UTC index, then use plain pandas resample("5min").
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


def build_validation_report(
    *,
    output_dir: Path,
    improved_summary: dict[str, Any],
    filter_stats: FilterStats,
) -> Path:
    baseline_path = ROOT / "research_out/phase3_v7_pfdd_defended_real/summary.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    replay = {
        "net_pnl_usd": 69973.0,
        "profit_factor": 2.1584,
        "trade_count": 282,
        "win_rate": None,
        "max_drawdown_pct": 3.7,
    }

    runner = improved_summary.get("phase3_defended_runner_summary", {})
    counts = runner.get("trade_counts_by_type", {})
    pnl_by_type = runner.get("net_pnl_usd_by_type", {})

    def _metric_row(name: str, baseline_v: str, replay_v: str, full_v: str, match: str) -> str:
        return f"| {name} | {baseline_v} | {replay_v} | {full_v} | {match} |"

    pf_full = float(improved_summary.get("profit_factor", 0.0) or 0.0)
    dd_full_pct = float(improved_summary.get("max_drawdown_pct", 0.0) or 0.0)
    pf_dev = abs(pf_full - replay["profit_factor"]) / replay["profit_factor"] if replay["profit_factor"] else 0.0
    dd_dev = (
        abs(dd_full_pct - replay["max_drawdown_pct"]) / replay["max_drawdown_pct"]
        if replay["max_drawdown_pct"]
        else 0.0
    )

    discrepancy_lines = []
    if pf_dev > 0.10 or dd_dev > 0.20:
        discrepancy_lines.append(
            "- Full engine deviates materially from replay. The replay was trade-log based and did not fully reproduce equity-dependent sizing and unified margin interactions."
        )
        discrepancy_lines.append(
            "- This run uses `USDJPY_M1_OANDA_extended.csv`, so it includes an extra out-of-sample month beyond the original 1000k baseline window."
        )
        discrepancy_lines.append(
            "- V44 is still oracle-driven from the 1000k phase1 report, so the extended tail contributes Tokyo/London behavior without new V44 oracle entries."
        )
        discrepancy_lines.append(
            "- H2 is applied at actual entry-execution time through London's `exec_gate`, which is slightly stricter than post-hoc trade-log replay enrichment."
        )
    else:
        discrepancy_lines.append("- Full engine results are close enough to the replay estimate that no additional discrepancy investigation was required.")

    expected_counts = {
        "v44_ny": 114,
        "london_setup_a": 12,
        "london_setup_d_l1": 68,
        "tokyo_v14": 88,
    }
    sub_lines = []
    for key in ["v44_ny", "london_setup_a", "london_setup_d_l1", "tokyo_v14"]:
        actual = int(counts.get(key, 0) or 0)
        expected = expected_counts[key]
        delta = actual - expected
        status = "OK" if abs(delta) <= 5 else "Investigate"
        sub_lines.append(
            f"| {key} | ~{expected} | {actual} | {delta:+d} | {status} | {pnl_by_type.get(key, 0.0):,.2f} |"
        )

    lines = [
        "# V7 H1+H2 Improved Validation Report",
        "",
        f"Run directory: `{output_dir}`",
        f"Generated: {pd.Timestamp.utcnow().isoformat()}",
        "",
        "## Filters Implemented",
        "",
        f"- H1 V44 high-volatility block: ATR-14 on completed M5 bars must be <= `{H1_ATR14_MAX_PIPS}` pips.",
        f"- H2 London Setup A with-trend block: `abs(EMA9-EMA21) <= {H2_EMA_SPREAD_MAX_PRICE:.3f}` or entry must be counter to the EMA trend.",
        "- H2 should still be treated as monitor-in-forward-testing because the original sample was small.",
        "",
        "## Top-Level Comparison",
        "",
        "| Metric | Baseline | Replay Estimate | Full Engine | Match? |",
        "| --- | --- | --- | --- | --- |",
        _metric_row(
            "Net P&L",
            f"${baseline['net_pnl_usd']:,.0f}",
            f"${replay['net_pnl_usd']:,.0f}",
            f"${improved_summary['net_pnl_usd']:,.0f}",
            "Yes" if abs(float(improved_summary["net_pnl_usd"]) - replay["net_pnl_usd"]) <= replay["net_pnl_usd"] * 0.15 else "No",
        ),
        _metric_row(
            "PF",
            f"{baseline['profit_factor']:.2f}",
            f"{replay['profit_factor']:.2f}",
            f"{pf_full:.2f}",
            "Yes" if pf_dev <= 0.10 else "No",
        ),
        _metric_row(
            "Trades",
            f"{baseline['trade_count']}",
            f"{replay['trade_count']}",
            f"{improved_summary['trade_count']}",
            "Yes" if abs(int(improved_summary["trade_count"]) - replay["trade_count"]) <= 20 else "No",
        ),
        _metric_row(
            "WR",
            f"{baseline['win_rate']:.1f}%",
            "—",
            f"{float(improved_summary['win_rate']):.1f}%",
            "—",
        ),
        _metric_row(
            "MaxDD",
            f"{baseline['max_drawdown_pct']:.2f}%",
            f"{replay['max_drawdown_pct']:.2f}%",
            f"{dd_full_pct:.2f}%",
            "Yes" if dd_dev <= 0.20 else "No",
        ),
        "",
        "## Discrepancy Notes",
        "",
        *discrepancy_lines,
        "",
        "## Filter Stats",
        "",
        f"- V44 oracle entries before H1: {filter_stats.v44_oracle_original}",
        f"- V44 oracle entries blocked by H1: {filter_stats.h1_v44_blocked}",
        f"- V44 oracle entries kept after H1: {filter_stats.h1_v44_kept}",
        f"- London Setup A entries blocked by H2: {filter_stats.h2_setup_a_blocked}",
        f"- H2 entries skipped due to missing M5 context: {filter_stats.h2_missing_context}",
        "",
        "## Per-Sub-Strategy Validation",
        "",
        "| Sub-strategy | Expected Trades | Full Engine Trades | Delta | Status | Net P&L |",
        "| --- | --- | --- | --- | --- | --- |",
        *sub_lines,
        "",
        "## Success Criteria Check",
        "",
        f"- PF >= 1.80: {'PASS' if pf_full >= 1.80 else 'FAIL'}",
        f"- Trades >= 200: {'PASS' if int(improved_summary['trade_count']) >= 200 else 'FAIL'}",
        f"- MaxDD <= 10%: {'PASS' if dd_full_pct <= 10.0 else 'FAIL'}",
        f"- Net P&L within 15% of baseline: {'PASS' if 57400 <= float(improved_summary['net_pnl_usd']) <= 78000 else 'FAIL'}",
        f"- All 4 sub-strategies net positive: {'PASS' if all(float(pnl_by_type.get(k, 0.0)) > 0 for k in expected_counts) else 'FAIL'}",
        "",
        "## Notes",
        "",
        "- The baseline numbers come from `research_out/phase3_v7_pfdd_defended_real/summary.json`.",
        "- The replay estimate comes from the v7 improvement analysis docs and should be treated as optimistic in-sample guidance, not a guaranteed full-engine target.",
    ]

    report_path = output_dir / "VALIDATION_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    uj_path = ROOT / "research_out/USDJPY_M1_OANDA_extended.csv"
    if not uj_path.is_file():
        print(f"ERROR: missing USDJPY data file: {uj_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = ROOT / "research_out/v7_h1h2_improved"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run_log.txt"
    log_path.write_text("", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log("=== Phase3 v7 defended H1+H2 improved — realistic spread ===")
    log(f"Started at {time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    log(f"Dataset: {uj_path}")
    log("Building M5 filter context...")
    filter_context = M5FilterContext(uj_path)
    filter_stats = FilterStats()

    original_load_v44_oracle = base_runner.load_v44_oracle
    original_advance_london = base_runner.advance_london_unified_bar

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

    cfg = RunConfig(
        hypothesis="phase3_v7_pfdd_h1h2_improved_realistic",
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

    log("Starting backtest with H1+H2 filters...")
    engine = Phase3V7PfddDefendedBacktestEngine()
    t0 = time.perf_counter()
    result = engine.run(cfg)
    elapsed = time.perf_counter() - t0
    log(f"Finished in {elapsed:.1f}s")

    trade_df = pd.read_csv(result.trade_log_path) if result.trade_log_path.is_file() else pd.DataFrame()
    bar_df = pd.read_csv(result.bar_log_path) if result.bar_log_path.is_file() else pd.DataFrame()
    ext = _extend_summary(dict(result.summary), trade_df, bar_df)
    ext["methodology_notes"] = (
        "Phase3 v7 defended improved via additive H1+H2 filters in the run script. "
        "H1 blocks V44 oracle entries when completed M5 ATR-14 > 7.04 pips. "
        "H2 blocks London Setup A entries when EMA-9/EMA-21 spread exceeds 3 pips and the trade aligns with that trend. "
        "Core runner and engine files were not modified."
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
    report_path = build_validation_report(output_dir=output_dir, improved_summary=ext, filter_stats=filter_stats)

    log(f"\nSummary written to {summary_path}")
    log(json.dumps(ext, indent=2, default=str))
    log(f"\nTrade log: {result.trade_log_path}")
    log(f"Validation report: {report_path}")


if __name__ == "__main__":
    main()
