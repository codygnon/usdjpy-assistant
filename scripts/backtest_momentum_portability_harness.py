#!/usr/bin/env python3
"""
Agent 3 / Track 3A: momentum portability harness.

Purpose:
  1. Run the native V44 engine on the current NY-only path.
  2. Run a latent/all-day V44 variant with identical entry logic but relaxed
     session window.
  3. Classify latent trades by analysis session + ownership cell.
  4. Build a narrow coupled shadow pilot by adding only off-session latent V44
     trades that occur in stable V44-owned momentum cells.

This is a research harness only. It does not modify live routing.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.ownership_table import cells_where_owner_is, cell_key, load_conservative_table
from core.regime_classifier import RegimeThresholds
from scripts import backtest_merged_integrated_tokyo_london_v2_ny as merged_engine
from scripts import backtest_variant_i_pbt_standdown as variant_i
from scripts import backtest_variant_k_london_cluster as variant_k
from scripts import validate_regime_classifier as regime_validation

OUT_DIR = ROOT / "research_out"
STARTING_EQUITY = 100_000.0
DATASETS = [
    str(OUT_DIR / "USDJPY_M1_OANDA_500k.csv"),
    str(OUT_DIR / "USDJPY_M1_OANDA_1000k.csv"),
]
V44_CFG_PATH = OUT_DIR / "session_momentum_v44_base_config.json"
JSON_OUT = OUT_DIR / "momentum_portability_harness.json"


def _dataset_key(path: str) -> str:
    name = Path(path).name
    if "500k" in name:
        return "500k"
    if "1000k" in name:
        return "1000k"
    raise ValueError(f"Unknown dataset: {name}")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, float) and np.isnan(x):
            return default
        return float(x)
    except Exception:
        return default


def _analysis_session(ts: pd.Timestamp, flat_cfg: dict[str, Any]) -> str:
    return merged_engine.v44_engine.classify_session(
        pd.Timestamp(ts).tz_convert("UTC"),
        float(flat_cfg.get("london_start", 8.5)),
        float(flat_cfg.get("london_end", 11.0)),
        float(flat_cfg.get("ny_start", 13.0)),
        float(flat_cfg.get("ny_end", 16.0)),
        tokyo_start=0.0,
        tokyo_end=3.0,
        tokyo_enabled=True,
    )


def _override_v44_portable(flat_cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(flat_cfg)
    out["ny_start"] = 0.0
    out["ny_end"] = 23.95
    out["v5_ny_start_delay_minutes"] = 0
    out["v5_session_entry_cutoff_minutes"] = 0
    out["v5_sessions"] = "ny_only"
    return out


def _run_v44_native(input_csv: str) -> tuple[list[merged_engine.TradeRow], dict[str, Any]]:
    results, embedded = merged_engine._run_v44_in_process(V44_CFG_PATH, input_csv)
    if isinstance(embedded, dict) and "v5_account_size" in embedded:
        base_eq = _safe_float(embedded.get("v5_account_size"), STARTING_EQUITY)
    else:
        base_eq = _safe_float((embedded or {}).get("v5", {}).get("account_size"), STARTING_EQUITY)
    trades = merged_engine._extract_v44_trades(results, default_entry_equity=base_eq)
    flat_cfg = merged_engine._convert_v44_embedded_to_flat(embedded, input_csv, "")
    flat_cfg["v5_sessions"] = "ny_only"
    return trades, flat_cfg


def _run_v44_portable(input_csv: str) -> tuple[list[merged_engine.TradeRow], dict[str, Any]]:
    raw = json.loads(V44_CFG_PATH.read_text(encoding="utf-8"))
    embedded = raw.get("config", raw) if isinstance(raw, dict) else {}
    with tempfile.TemporaryDirectory(prefix="momentum_portable_") as td:
        tmp_cfg = Path(td) / "v44_portable_flat.json"
        flat = merged_engine._convert_v44_embedded_to_flat(
            embedded,
            input_csv,
            str(Path(td) / "v44_portable_report.json"),
        )
        flat = _override_v44_portable(flat)
        tmp_cfg.write_text(json.dumps(flat, indent=2), encoding="utf-8")
        args = merged_engine.v44_engine.parse_args(["--config", str(tmp_cfg)])
        results = merged_engine.v44_engine.run_backtest_v5(args)
    if isinstance(embedded, dict) and "v5_account_size" in embedded:
        base_eq = _safe_float(embedded.get("v5_account_size"), STARTING_EQUITY)
    else:
        base_eq = _safe_float((embedded or {}).get("v5", {}).get("account_size"), STARTING_EQUITY)
    trades = merged_engine._extract_v44_trades(results, default_entry_equity=base_eq)
    native_semantics = merged_engine._convert_v44_embedded_to_flat(embedded, input_csv, "")
    native_semantics["v5_sessions"] = "ny_only"
    return trades, native_semantics


def _lookup_cell(
    trade: merged_engine.TradeRow,
    classified_dynamic: pd.DataFrame,
    dyn_time_idx: pd.DatetimeIndex,
) -> str:
    regime_info = variant_i._lookup_regime_with_dynamic(
        classified_dynamic,
        dyn_time_idx,
        trade.entry_time,
    )
    idx = dyn_time_idx.get_indexer([pd.Timestamp(trade.entry_time)], method="ffill")[0]
    full_row = classified_dynamic.iloc[idx]
    er = float(full_row.get("sf_er", 0.5))
    if np.isnan(er):
        er = 0.5
    return cell_key(
        regime_info["regime_label"],
        variant_k._er_bucket(er),
        variant_k._der_bucket(regime_info["delta_er"]),
    )


def _trade_stats(trades: list[merged_engine.TradeRow]) -> dict[str, Any]:
    if not trades:
        return {
            "count": 0,
            "wins": 0,
            "losses": 0,
            "win_rate_pct": 0.0,
            "avg_pips": 0.0,
            "net_pips": 0.0,
            "avg_usd": 0.0,
            "net_usd": 0.0,
            "profit_factor": 0.0,
        }
    wins = sum(1 for t in trades if t.pips > 0)
    gross_win = sum(t.usd for t in trades if t.usd > 0)
    gross_loss = -sum(t.usd for t in trades if t.usd < 0)
    pf = gross_win / gross_loss if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
    n = len(trades)
    return {
        "count": n,
        "wins": wins,
        "losses": n - wins,
        "win_rate_pct": round(wins / n * 100.0, 1),
        "avg_pips": round(sum(t.pips for t in trades) / n, 2),
        "net_pips": round(sum(t.pips for t in trades), 2),
        "avg_usd": round(sum(t.usd for t in trades) / n, 2),
        "net_usd": round(sum(t.usd for t in trades), 2),
        "profit_factor": round(float(pf), 4),
    }


def _sample_rows(trades: list[merged_engine.TradeRow], sessions: dict[pd.Timestamp, str], cells: dict[pd.Timestamp, str], limit: int = 30) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for t in trades[:limit]:
        ts = pd.Timestamp(t.entry_time)
        rows.append({
            "entry_time": ts.isoformat(),
            "exit_time": pd.Timestamp(t.exit_time).isoformat(),
            "analysis_session": sessions.get(ts, "unknown"),
            "cell": cells.get(ts, "unknown"),
            "side": t.side,
            "pips": round(float(t.pips), 2),
            "usd": round(float(t.usd), 2),
            "entry_profile": str((t.raw or {}).get("entry_profile", "")),
            "entry_signal_mode": str((t.raw or {}).get("entry_signal_mode", "")),
            "exit_reason": t.exit_reason,
        })
    return rows


def _breakdown_by(items: list[merged_engine.TradeRow], key_fn) -> list[dict[str, Any]]:
    groups: dict[str, list[merged_engine.TradeRow]] = defaultdict(list)
    for t in items:
        groups[str(key_fn(t))].append(t)
    rows = []
    for key in sorted(groups):
        stats = _trade_stats(groups[key])
        rows.append({"key": key, **stats})
    return rows


def _annotate(
    trades: list[merged_engine.TradeRow],
    classified_dynamic: pd.DataFrame,
    dyn_time_idx: pd.DatetimeIndex,
    flat_cfg: dict[str, Any],
) -> tuple[dict[pd.Timestamp, str], dict[pd.Timestamp, str]]:
    sessions: dict[pd.Timestamp, str] = {}
    cells: dict[pd.Timestamp, str] = {}
    for t in trades:
        ts = pd.Timestamp(t.entry_time)
        sessions[ts] = _analysis_session(ts, flat_cfg)
        cells[ts] = _lookup_cell(t, classified_dynamic, dyn_time_idx)
    return sessions, cells


def _pilot_result(
    baseline_pre: list[merged_engine.TradeRow],
    baseline_stats: dict[str, Any],
    additions: list[merged_engine.TradeRow],
    v14_max_units: int,
) -> dict[str, Any]:
    combined = sorted(baseline_pre + additions, key=lambda t: (t.exit_time, t.entry_time))
    coupled = merged_engine._apply_shared_equity_coupling(combined, STARTING_EQUITY, v14_max_units=v14_max_units)
    eq = merged_engine._build_equity_curve(coupled, STARTING_EQUITY)
    stats = merged_engine._stats(coupled, STARTING_EQUITY, eq)
    overlaps = 0
    baseline_times = Counter(pd.Timestamp(t.entry_time) for t in baseline_pre)
    for t in additions:
        if baseline_times[pd.Timestamp(t.entry_time)] > 0:
            overlaps += 1
    return {
        "added_trade_count": len(additions),
        "exact_timestamp_overlaps_with_baseline": overlaps,
        "summary": stats,
        "delta_vs_baseline": {
            "total_trades": int(stats["total_trades"] - baseline_stats["total_trades"]),
            "net_usd": round(float(stats["net_usd"] - baseline_stats["net_usd"]), 2),
            "profit_factor": round(float(stats["profit_factor"] - baseline_stats["profit_factor"]), 4),
            "max_drawdown_usd": round(float(stats["max_drawdown_usd"] - baseline_stats["max_drawdown_usd"]), 2),
        },
        "by_strategy": merged_engine._subset_breakdown(coupled, lambda t: t.strategy),
        "added_samples": [
            {
                "entry_time": pd.Timestamp(t.entry_time).isoformat(),
                "exit_time": pd.Timestamp(t.exit_time).isoformat(),
                "side": t.side,
                "pips": round(float(t.pips), 2),
                "usd": round(float(t.usd), 2),
                "raw_entry_profile": str((t.raw or {}).get("entry_profile", "")),
                "raw_entry_signal_mode": str((t.raw or {}).get("entry_signal_mode", "")),
            }
            for t in additions[:30]
        ],
    }


def _load_classified_fast(dataset: str) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Load classified data WITH dynamic sf_er and sf_delta_er features.

    Uses the full dynamic classification pipeline so that ER/DER bucketing
    produces correct ownership cells (not the degenerate er_mid/der_pos that
    the lightweight classify_all_bars path produces).
    """
    classified = variant_i._load_classified_with_dynamic(dataset)
    return classified, pd.DatetimeIndex(classified["time"])


def _analyze_dataset(dataset: str, candidate_cells: set[str], *, raw_only: bool) -> dict[str, Any]:
    dk = _dataset_key(dataset)
    print(f"\n{'=' * 60}\nMomentum portability harness: {dk}\n{'=' * 60}")

    baseline_meta = None
    kept_pre: list[merged_engine.TradeRow] = []
    baseline_stats: dict[str, Any] | None = None
    if raw_only:
        classified_dynamic, dyn_time_idx = _load_classified_fast(dataset)
        print("Baseline K: skipped (raw-only mode)")
    else:
        kept_pre, baseline_meta, classified_dynamic, dyn_time_idx, _, _ = variant_k.build_variant_k_pre_coupling_kept(dataset)
        baseline_coupled = merged_engine._apply_shared_equity_coupling(
            sorted(kept_pre, key=lambda t: (t.exit_time, t.entry_time)),
            STARTING_EQUITY,
            v14_max_units=baseline_meta["v14_max_units"],
        )
        baseline_eq = merged_engine._build_equity_curve(baseline_coupled, STARTING_EQUITY)
        baseline_stats = merged_engine._stats(baseline_coupled, STARTING_EQUITY, baseline_eq)
        print(
            f"Baseline K: {baseline_stats['total_trades']} trades, "
            f"${baseline_stats['net_usd']:,.2f}, PF={baseline_stats['profit_factor']:.3f}"
        )

    native_trades, native_cfg = _run_v44_native(dataset)
    latent_trades, latent_cfg = _run_v44_portable(dataset)
    print(f"Native V44 trades: {len(native_trades)}")
    print(f"Latent all-day V44 trades: {len(latent_trades)}")

    native_sessions, native_cells = _annotate(native_trades, classified_dynamic, dyn_time_idx, native_cfg)
    latent_sessions, latent_cells = _annotate(latent_trades, classified_dynamic, dyn_time_idx, latent_cfg)

    native_in_candidate_cells = [
        t for t in native_trades
        if native_cells.get(pd.Timestamp(t.entry_time)) in candidate_cells
    ]
    latent_offsession_candidates = [
        t for t in latent_trades
        if latent_sessions.get(pd.Timestamp(t.entry_time)) != "ny_overlap"
        and latent_cells.get(pd.Timestamp(t.entry_time)) in candidate_cells
    ]
    latent_offsession_nonowner = [
        t for t in latent_trades
        if latent_sessions.get(pd.Timestamp(t.entry_time)) != "ny_overlap"
        and latent_cells.get(pd.Timestamp(t.entry_time)) not in candidate_cells
    ]

    pilot: dict[str, Any] | None = None
    if not raw_only:
        added_trades = []
        for t in latent_offsession_candidates:
            added_trades.append(
                merged_engine.TradeRow(
                    strategy="v44_portable",
                    entry_time=t.entry_time,
                    exit_time=t.exit_time,
                    entry_session=latent_sessions.get(pd.Timestamp(t.entry_time), "dead"),
                    side=t.side,
                    pips=t.pips,
                    usd=t.usd,
                    exit_reason=t.exit_reason,
                    standalone_entry_equity=t.standalone_entry_equity,
                    raw=t.raw,
                    size_scale=t.size_scale,
                )
            )
        pilot = _pilot_result(kept_pre, baseline_stats, added_trades, baseline_meta["v14_max_units"])

    print(f"Off-session portability candidates in V44-owned momentum cells: {len(latent_offsession_candidates)}")
    if pilot is not None:
        print(
            f"Coupled shadow delta vs K: "
            f"${pilot['delta_vs_baseline']['net_usd']:+,.2f}, "
            f"PF {pilot['delta_vs_baseline']['profit_factor']:+.4f}, "
            f"DD {pilot['delta_vs_baseline']['max_drawdown_usd']:+,.2f}"
        )

    same_cell_native_vs_off = []
    all_cells = sorted({
        native_cells.get(pd.Timestamp(t.entry_time), "")
        for t in native_in_candidate_cells
    } | {
        latent_cells.get(pd.Timestamp(t.entry_time), "")
        for t in latent_offsession_candidates
    })
    for cell in all_cells:
        native_cell = [t for t in native_in_candidate_cells if native_cells.get(pd.Timestamp(t.entry_time)) == cell]
        off_cell = [t for t in latent_offsession_candidates if latent_cells.get(pd.Timestamp(t.entry_time)) == cell]
        same_cell_native_vs_off.append({
            "cell": cell,
            "native_ny": _trade_stats(native_cell),
            "offsession": _trade_stats(off_cell),
        })

    verdict = []
    if not latent_offsession_candidates:
        verdict.append("No off-session V44 trades fired in the priority ownership cells.")
        verdict.append("3A fails immediately on fill scarcity.")
    else:
        verdict.append(
            f"Off-session V44 produced {len(latent_offsession_candidates)} real fills in priority cells."
        )
        if pilot is None:
            verdict.append("Coupled shadow pilot was skipped in raw-only mode.")
        else:
            if pilot["delta_vs_baseline"]["net_usd"] > 0:
                verdict.append("Coupled shadow pilot improved net USD vs Variant K.")
            else:
                verdict.append("Coupled shadow pilot did not improve net USD vs Variant K.")
            if pilot["delta_vs_baseline"]["max_drawdown_usd"] > 0:
                verdict.append("Drawdown worsened in the coupled pilot.")
            else:
                verdict.append("Drawdown held flat or improved in the coupled pilot.")

    return {
        "baseline_variant_k": baseline_stats,
        "native_v44_reference": {
            "all_native_trades": _trade_stats(native_trades),
            "candidate_cell_native_trades": _trade_stats(native_in_candidate_cells),
            "by_cell": _breakdown_by(native_in_candidate_cells, lambda t: native_cells.get(pd.Timestamp(t.entry_time), "unknown")),
            "samples": _sample_rows(native_in_candidate_cells, native_sessions, native_cells),
        },
        "latent_v44_all_day": {
            "all_trades": _trade_stats(latent_trades),
            "by_analysis_session": _breakdown_by(latent_trades, lambda t: latent_sessions.get(pd.Timestamp(t.entry_time), "unknown")),
            "by_owner_cell_type": [
                {
                    "key": "priority_v44_owner_cell",
                    **_trade_stats([t for t in latent_trades if latent_cells.get(pd.Timestamp(t.entry_time)) in candidate_cells]),
                },
                {
                    "key": "other_cell",
                    **_trade_stats([t for t in latent_trades if latent_cells.get(pd.Timestamp(t.entry_time)) not in candidate_cells]),
                },
            ],
        },
        "offsession_portability_candidates": {
            "stats": _trade_stats(latent_offsession_candidates),
            "by_analysis_session": _breakdown_by(latent_offsession_candidates, lambda t: latent_sessions.get(pd.Timestamp(t.entry_time), "unknown")),
            "by_cell": _breakdown_by(latent_offsession_candidates, lambda t: latent_cells.get(pd.Timestamp(t.entry_time), "unknown")),
            "same_cell_native_vs_offsession": same_cell_native_vs_off,
            "samples": _sample_rows(latent_offsession_candidates, latent_sessions, latent_cells),
        },
        "offsession_non_owner_reference": {
            "stats": _trade_stats(latent_offsession_nonowner),
            "by_analysis_session": _breakdown_by(latent_offsession_nonowner, lambda t: latent_sessions.get(pd.Timestamp(t.entry_time), "unknown")),
        },
        "coupled_shadow_pilot": pilot,
        "verdict": " ".join(verdict),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Momentum portability harness")
    p.add_argument(
        "--dataset",
        action="append",
        help="Optional dataset path(s). Defaults to both 500k and 1000k research datasets.",
    )
    p.add_argument(
        "--output",
        default=str(JSON_OUT),
        help="Output JSON path. Defaults to research_out/momentum_portability_harness.json",
    )
    p.add_argument(
        "--raw-only",
        action="store_true",
        help="Skip the expensive Variant K baseline + coupled pilot and only build raw portability evidence.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    table = load_conservative_table(research_out=OUT_DIR)
    v44_owner_cells = cells_where_owner_is(table, "v44_ny")
    candidate_cells = sorted(c for c in v44_owner_cells if c.startswith("momentum/"))
    datasets = args.dataset or DATASETS
    out_path = Path(args.output)

    payload: dict[str, Any] = {
        "objective": (
            "Scaffold a real momentum portability harness: compare native NY-only V44 to "
            "latent all-day V44 and test a narrow coupled pilot using only off-session "
            "trades in stable V44-owned momentum cells."
        ),
        "inputs": {
            "datasets": datasets,
            "v44_config": str(V44_CFG_PATH),
            "raw_only": bool(args.raw_only),
            "candidate_cells": candidate_cells,
            "selection_note": (
                "Uses stable conservative ownership table cells where owner = v44_ny, "
                "restricted to momentum/* contexts per Agent 3 priority order."
            ),
        },
        "dataset_results": {},
    }

    for dataset in datasets:
        payload["dataset_results"][_dataset_key(dataset)] = _analyze_dataset(
            dataset,
            set(candidate_cells),
            raw_only=bool(args.raw_only),
        )

    out_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
