#!/usr/bin/env python3
"""Generate a Phase 3 parity baseline CSV from individual strategy backtest reports.

The output CSV can be passed to phase3_parity_check.py --compare to evaluate
live replay parity against the backtest source of truth.

Three strategy report formats are supported:

  V14 (tokyo_meanrev backtest output):
    Top-level "trades" list; fields: direction (long/short), entry_datetime,
    sl_price, tp_price, entry_session.

  London V2 / V44 (session_momentum / london_only_preset backtest output):
    results.closed_trades list; fields: side (buy/sell), entry_time,
    entry_price, sl_pips, tp1_pips, entry_session.

Usage:
    python scripts/generate_phase3_baseline.py \\
        --v14-report research_out/tokyo_actual_v2_500k_report.json \\
        --london-report research_out/london_trail3_500k.json \\
        --v44-report research_out/session_momentum_v32a_500k.json \\
        --start 2024-10-01 --end 2024-12-31 \\
        --output research_out/phase3_baseline_2024-10-01_2024-12-31.csv

Omit any --*-report flag to skip that strategy's contribution.
If --output is not given, the path is auto-derived from --start/--end.

IMPORTANT - Indicator drift limitation:
  This script reconstructs a baseline from per-strategy backtest report exports.
  Those backtests were run on OANDA-fetched OHLC bars, while phase3_parity_check.py
  uses M1 bars resampled to M15/H4/D.  The resulting indicator values (Bollinger Bands,
  ATR, SAR, pivots) can differ, causing the integrated engine to fire on different bars
  than the backtests — even with identical confluence thresholds.

  PREFERRED alternative: run phase3_parity_check.py once with --save-baseline to capture
  the integrated engine's own decisions as the reference.  Subsequent runs can then
  compare against that self-derived baseline for true regression testing.

  V14 SL/TP are intentionally excluded from this baseline (set to None) because the V14
  backtest stores them in limit-order format incompatible with the integrated engine's
  pivot-based market-order SL and ATR-based TP1.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PIP_SIZE = 0.01  # USDJPY pip size

# Session membership for filtering
_NY_SESSIONS = {"ny", "ny_overlap", "new_york", "us", "us_open"}
_LONDON_SESSIONS = {"london", "london_open", "europe", "ldn"}
_TOKYO_SESSIONS = {"tokyo", "asia", "asian", "tokyo_open"}


def _to_utc(ts: Any) -> pd.Timestamp | None:
    try:
        t = pd.Timestamp(ts)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        else:
            t = t.tz_convert("UTC")
        return t
    except Exception:
        return None


def _pips_to_price(entry: float, pips: float, side: str, direction: str = "sl") -> float:
    """Convert sl/tp pips to absolute price.

    For SL: buy → entry - pips*PIP; sell → entry + pips*PIP
    For TP: buy → entry + pips*PIP; sell → entry - pips*PIP
    """
    delta = float(pips) * PIP_SIZE
    if direction == "sl":
        return float(entry) - delta if side == "buy" else float(entry) + delta
    else:  # tp
        return float(entry) + delta if side == "buy" else float(entry) - delta


def _load_v14_trades(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> list[dict[str, Any]]:
    """Load V14 (tokyo_meanrev) backtest trades.

    Expected: JSON with top-level "trades" list where each record has:
      direction (long/short), entry_datetime, sl_price, tp_price, entry_session
    """
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)

    raw_trades = d.get("trades", [])
    if not isinstance(raw_trades, list):
        raise ValueError(f"Expected 'trades' list in {path}")

    records: list[dict[str, Any]] = []
    skipped = 0
    session_hours_warned = False
    for t in raw_trades:
        ts = _to_utc(t.get("entry_datetime") or t.get("signal_datetime"))
        if ts is None or ts < start or ts > end:
            skipped += 1
            continue

        session_raw = str(t.get("entry_session") or "").lower()
        if session_raw and session_raw not in _TOKYO_SESSIONS:
            # Skip non-tokyo if entry_session is present and clearly not tokyo
            skipped += 1
            continue

        # Warn once if the report uses traditional Tokyo hours (00:00-08:00 UTC).
        # The live Phase 3 engine's "tokyo" session is 16:00-22:00 UTC (from source config).
        # Use phase1_v14_baseline_500k_report.json or phase1_v14_baseline_1000k_report.json
        # (which fire in 16:00-22:00 UTC) instead of tokyo_actual_v2_500k_report.json.
        if not session_hours_warned and ts.hour < 10:
            print(
                f"  WARNING: V14 trade at {ts} has hour={ts.hour} (traditional Tokyo hours 00-08 UTC).\n"
                f"  Live Phase 3 'tokyo' session = 16:00-22:00 UTC. Use phase1_v14_baseline_500k_report.json\n"
                f"  or phase1_v14_baseline_1000k_report.json to match live session routing.",
                file=sys.stderr,
            )
            session_hours_warned = True

        direction = str(t.get("direction") or "").lower()
        side = "buy" if direction in ("long", "buy") else "sell" if direction in ("short", "sell") else None
        if side is None:
            skipped += 1
            continue

        # NOTE: V14 backtest SL/TP are NOT included in the baseline because they are
        # semantically incompatible with the integrated Phase 3 engine's computation:
        #   sl_price: stored in limit-order format — placed on the WRONG side of entry
        #             (sl > entry for buy), unusable as a conventional stop reference.
        #   tp_price: the full take-profit target (~130 pips), not the ATR-based TP1
        #             partial-close level (6-12 pips) that the integrated engine uses.
        # The parity compare path will skip SL/TP comparison for rows with sl=None.
        # Use --save-baseline from phase3_parity_check.py to capture engine-native SL/TP.
        records.append({
            "bar_time": ts.isoformat(),
            "placed": True,
            "side": side,
            "strategy_tag": "phase3:v14_mean_reversion",
            "strategy": "v14",
            "session": "tokyo",
            "entry_price": t.get("entry_price"),
            "sl_price": None,
            "tp1_price": None,
        })

    print(f"  V14: {len(records)} trades kept, {skipped} skipped (out-of-range / non-tokyo / bad-side)", file=sys.stderr)
    return records


def _load_session_momentum_trades(
    path: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    session_filter: set[str],
    strategy_tag: str,
    session_label: str,
) -> list[dict[str, Any]]:
    """Load London V2 or V44 trades from session_momentum / london_only_preset format.

    Expected: JSON with results.closed_trades list where each record has:
      side (buy/sell), entry_time, entry_price, sl_pips, tp1_pips, entry_session
    """
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)

    raw_trades: list[Any] = []
    if "results" in d and isinstance(d["results"], dict):
        raw_trades = d["results"].get("closed_trades", [])
    elif "closed_trades" in d:
        raw_trades = d["closed_trades"]
    elif "trades" in d and isinstance(d["trades"], list):
        # Some london exports use a flat "trades" list with same schema
        first = d["trades"][0] if d["trades"] else {}
        if "entry_time" in first or "sl_pips" in first:
            raw_trades = d["trades"]

    if not isinstance(raw_trades, list):
        raise ValueError(f"Could not locate closed_trades list in {path}")

    records: list[dict[str, Any]] = []
    skipped = 0
    for t in raw_trades:
        ts = _to_utc(t.get("entry_time") or t.get("entry_datetime"))
        if ts is None or ts < start or ts > end:
            skipped += 1
            continue

        session_raw = str(t.get("entry_session") or "").lower().replace("-", "_").replace(" ", "_")
        if session_filter and session_raw not in session_filter:
            skipped += 1
            continue

        side = str(t.get("side") or "").lower()
        if side not in ("buy", "sell"):
            skipped += 1
            continue

        entry_price = t.get("entry_price")
        sl_pips = t.get("sl_pips")
        tp1_pips = t.get("tp1_pips") or t.get("tp_pips")

        sl_price = None
        tp1_price = None
        if entry_price is not None and sl_pips is not None:
            sl_price = _pips_to_price(float(entry_price), float(sl_pips), side, direction="sl")
        elif t.get("sl_price") is not None:
            sl_price = float(t["sl_price"])

        if entry_price is not None and tp1_pips is not None:
            tp1_price = _pips_to_price(float(entry_price), float(tp1_pips), side, direction="tp")
        elif t.get("tp1_price") is not None or t.get("tp_price") is not None:
            tp1_price = float(t.get("tp1_price") or t.get("tp_price"))

        records.append({
            "bar_time": ts.isoformat(),
            "placed": True,
            "side": side,
            "strategy_tag": strategy_tag,
            "strategy": session_label if session_label != "ny" else "v44_ny",
            "session": session_label,
            "entry_price": float(entry_price) if entry_price is not None else None,
            "sl_price": sl_price,
            "tp1_price": tp1_price,
        })

    print(
        f"  {strategy_tag}: {len(records)} trades kept, {skipped} skipped "
        f"(out-of-range / off-session / bad-side)",
        file=sys.stderr,
    )
    return records


def main() -> None:
    p = argparse.ArgumentParser(description="Generate Phase 3 parity baseline CSV from backtest reports")
    p.add_argument("--v14-report", default="", help="V14 (tokyo_meanrev) backtest JSON report")
    p.add_argument("--london-report", default="", help="London V2 backtest JSON report")
    p.add_argument("--v44-report", default="", help="V44 NY (session_momentum) backtest JSON report")
    p.add_argument("--start", required=True, help="Start date UTC YYYY-MM-DD (inclusive)")
    p.add_argument("--end", required=True, help="End date UTC YYYY-MM-DD (inclusive)")
    p.add_argument("--output", default="", help="Output CSV path (auto-derived if omitted)")
    args = p.parse_args()

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    all_records: list[dict[str, Any]] = []

    if args.v14_report:
        print(f"Loading V14 report: {args.v14_report}", file=sys.stderr)
        all_records.extend(_load_v14_trades(Path(args.v14_report), start, end))

    if args.london_report:
        print(f"Loading London V2 report: {args.london_report}", file=sys.stderr)
        all_records.extend(
            _load_session_momentum_trades(
                Path(args.london_report),
                start,
                end,
                session_filter=_LONDON_SESSIONS,
                strategy_tag="phase3:london_v2_arb",
                session_label="london",
            )
        )

    if args.v44_report:
        print(f"Loading V44 NY report: {args.v44_report}", file=sys.stderr)
        all_records.extend(
            _load_session_momentum_trades(
                Path(args.v44_report),
                start,
                end,
                session_filter=_NY_SESSIONS,
                strategy_tag="phase3:v44_ny",
                session_label="ny",
            )
        )

    if not all_records:
        print("WARNING: No trades loaded. Check report paths and date range.", file=sys.stderr)

    df = pd.DataFrame(all_records) if all_records else pd.DataFrame(
        columns=["bar_time", "placed", "side", "strategy_tag", "strategy", "session",
                 "entry_price", "sl_price", "tp1_price"]
    )

    # Sort by bar_time ascending
    if "bar_time" in df.columns and not df.empty:
        df["bar_time"] = pd.to_datetime(df["bar_time"], utc=True, errors="coerce")
        df = df.sort_values("bar_time").reset_index(drop=True)
        df["bar_time"] = df["bar_time"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")

    out_path = args.output
    if not out_path:
        out_path = f"research_out/phase3_baseline_{args.start}_{args.end}.csv"

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"Wrote {len(df)} baseline entries to {out}", file=sys.stderr)
    print(
        json.dumps({
            "output": str(out),
            "total_entries": len(df),
            "date_range": [args.start, args.end],
            "by_strategy": df.groupby("strategy_tag").size().to_dict() if not df.empty and "strategy_tag" in df.columns else {},
            "by_session": df.groupby("session").size().to_dict() if not df.empty and "session" in df.columns else {},
        }, indent=2)
    )


if __name__ == "__main__":
    main()
