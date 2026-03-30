#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path("/Users/codygnon/Documents/usdjpy_assistant")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.offensive_slice_spec import OffensiveSliceSpec
from scripts import diagnostic_chart_authorization_loop as auth_loop
from scripts import diagnostic_london_setupd_trade_outcomes as london_outcomes
from scripts import backtest_offensive_setupd_additive as setupd_additive
from scripts.backtest_offensive_slice_additive import build_baseline_context, run_slice_additive
from core.ownership_table import cell_key, der_bucket, er_bucket

OUT_DIR = ROOT / "research_out"
OUTPUT_JSON = OUT_DIR / "offensive_slice_discovery_matrix.json"
OUTPUT_BOARD_JSON = OUT_DIR / "offensive_slice_board.json"
OUTPUT_BOARD_MD = OUT_DIR / "offensive_slice_board.md"
REFERENCE_ARTIFACT = OUT_DIR / "offensive_setupd_additive_mean_reversion_er_low_der_neg_100pct_first30.json"
LONDON_TRADE_CACHE_VERSION = "v1"
DATASETS = {
    "500k": str(OUT_DIR / "USDJPY_M1_OANDA_500k.csv"),
    "1000k": str(OUT_DIR / "USDJPY_M1_OANDA_1000k.csv"),
}

LONDON_ARGS = SimpleNamespace(
    spread_pips=0.3,
    sl_buffer_pips=3.0,
    sl_min_pips=5.0,
    sl_max_pips=20.0,
    tp1_r_multiple=1.0,
    tp2_r_multiple=2.0,
    tp1_close_fraction=0.5,
    be_offset_pips=1.0,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified offensive slice discovery runner")
    p.add_argument("--output-json", default=str(OUTPUT_JSON))
    p.add_argument("--output-board-json", default=str(OUTPUT_BOARD_JSON))
    p.add_argument("--output-board-md", default=str(OUTPUT_BOARD_MD))
    p.add_argument("--reference-only", action="store_true", help="Run only the known reference slice regression.")
    return p.parse_args()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return str(obj)


def _ts(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _dataset_cache_dir(dataset: str) -> Path:
    cache_dir = Path(dataset).resolve().parent / "shadow_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _timing_gate_for_minutes(minutes: float | None, trade: dict[str, Any]) -> str:
    if minutes is None or not np.isfinite(minutes):
        return "unknown"
    if trade["strategy"] == "london_v2" and 15.0 <= minutes <= 45.0:
        return "first_30min"
    if minutes < 60:
        return "early"
    if minutes < 180:
        return "mid"
    return "late"


def _session_open_minutes(entry_ts: pd.Timestamp, session: str) -> float | None:
    day = entry_ts.floor("D")
    if session == "london":
        from scripts import backtest_v2_multisetup_london as london_v2_engine
        open_hour = london_v2_engine.uk_london_open_utc(day)
    elif session == "tokyo":
        open_hour = 16.0
    elif session in {"ny", "ny_overlap"}:
        open_hour = 13.0
    else:
        return None
    session_open = day + pd.Timedelta(hours=open_hour)
    return (entry_ts - session_open).total_seconds() / 60.0


def _coverage_class(count: int) -> str:
    if count <= 2:
        return "whitespace"
    if count <= 5:
        return "thin"
    return "covered"


def _load_bar_frame(dataset: str) -> pd.DataFrame:
    bars = auth_loop._load_bar_frame(dataset).copy()
    bars["time"] = pd.to_datetime(bars["time"], utc=True, errors="coerce")
    bars = bars.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    bars["ownership_cell"] = [
        cell_key(
            str(row.get("regime_hysteresis", "ambiguous")),
            er_bucket(float(row.get("sf_er", 0.5))),
            der_bucket(float(row.get("delta_er", 0.0))),
        )
        for _, row in bars.iterrows()
    ]
    return bars[["time", "ownership_cell", "regime_hysteresis", "sf_er", "delta_er"]]


def _classify_times_to_cells(dataset: str, timestamps: list[pd.Timestamp]) -> list[dict[str, Any]]:
    bars = _load_bar_frame(dataset)
    ts_df = pd.DataFrame({"entry_time": sorted(_ts(t) for t in timestamps)})
    merged = pd.merge_asof(ts_df, bars.rename(columns={"time": "bar_time"}), left_on="entry_time", right_on="bar_time", direction="backward")
    out = []
    for row in merged.to_dict(orient="records"):
        out.append({
            "ownership_cell": row.get("ownership_cell", "unknown/er_mid/der_pos"),
            "regime_hysteresis": row.get("regime_hysteresis", "ambiguous"),
            "sf_er": float(row.get("sf_er", 0.5) or 0.5),
            "delta_er": float(row.get("delta_er", 0.0) or 0.0),
        })
    return out


def _load_variant_k_coverage(dataset: str) -> dict[str, dict[str, Any]]:
    dataset_path = Path(dataset).resolve()
    cache_path = _dataset_cache_dir(dataset) / f"offensive_slice_variant_k_coverage_{dataset_path.stem}_v1.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    dataset_key = "500k" if "500k" in dataset_path.name else "1000k"
    report_path = OUT_DIR / f"phase3_integrated_variant_k_{dataset_key}_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    closed = report.get("closed_trades", [])
    cells = _classify_times_to_cells(dataset, [_ts(t["entry_time"]) for t in closed])
    out: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "by_strategy": Counter(), "net_pips": 0.0})
    for trade, cell_info in zip(closed, cells):
        cell = str(cell_info["ownership_cell"])
        out[cell]["count"] += 1
        out[cell]["by_strategy"][str(trade["strategy"])] += 1
        out[cell]["net_pips"] += float(trade["pips"])
    finalized = {}
    for cell, info in out.items():
        finalized[cell] = {
            "variant_k_trade_count": int(info["count"]),
            "variant_k_by_strategy": dict(info["by_strategy"]),
            "variant_k_net_pips": round(float(info["net_pips"]), 2),
            "coverage_class": _coverage_class(int(info["count"])),
        }
    cache_path.write_text(json.dumps(finalized, indent=2, default=_json_default), encoding="utf-8")
    return finalized


def _normalize_london_trades(dataset: str) -> list[dict[str, Any]]:
    cache_path = _dataset_cache_dir(dataset) / f"offensive_slice_london_trades_{Path(dataset).stem}_{LONDON_TRADE_CACHE_VERSION}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    report = london_outcomes.run_dataset(dataset, LONDON_ARGS)
    london_cfg = setupd_additive._load_london_cfg()
    out = []
    for t in report["_all_trades"]:
        if not t.get("native_allowed", False):
            continue
        entry_ts = _ts(t["entry_time"])
        signal_ts = _ts(t.get("signal_time", t["entry_time"]))
        session = "london"
        minutes = _session_open_minutes(signal_ts, session)
        usd, units = setupd_additive._calc_setupd_usd(t, london_cfg, size_scale=1.0)
        out.append({
            "strategy": "london_v2",
            "source_mode": "london_stateful_trade_sim",
            "dataset_key": Path(dataset).name,
            "entry_time": entry_ts.isoformat(),
            "exit_time": _ts(t["exit_time"]).isoformat(),
            "signal_time": signal_ts.isoformat(),
            "entry_session": session,
            "side": "buy" if t["direction"] == "long" else "sell",
            "direction": str(t["direction"]),
            "pips": float(t["pnl_pips"]),
            "usd": float(usd),
            "exit_reason": str(t["exit_reason"]),
            "ownership_cell": str(t["ownership_cell"]),
            "setup_type": "D",
            "evaluator_mode": "setup_d_trade_outcome_sim",
            "timing_gate": _timing_gate_for_minutes(minutes, {"strategy": "london_v2"}),
            "timing_minutes": round(float(minutes), 2) if minutes is not None else None,
            "native_allowed": True,
            "standalone_entry_equity": 100000.0,
            "raw": {**dict(t), "position_units": int(units)},
            "size_scale": 1.0,
        })
    cache_path.write_text(json.dumps(out, indent=2, default=_json_default), encoding="utf-8")
    return out


def _normalize_v14_trades(dataset: str) -> list[dict[str, Any]]:
    dataset_key = "500k" if "500k" in Path(dataset).name else "1000k"
    candidate_paths = [
        OUT_DIR / f"tokyo_optimized_v14_{dataset_key}_trades.csv",
        OUT_DIR / f"phase1_v14_baseline_{dataset_key}_trades.csv",
    ]
    csv_path = next((p for p in candidate_paths if p.exists()), None)
    if csv_path is None:
        return []
    df = pd.read_csv(csv_path)
    entries = [_ts(x) for x in df["entry_datetime"]]
    cells = _classify_times_to_cells(dataset, entries)
    out = []
    for row, cell_info in zip(df.to_dict(orient="records"), cells):
        entry_ts = _ts(row["entry_datetime"])
        session = str(row.get("entry_session", "tokyo"))
        minutes = _session_open_minutes(entry_ts, session)
        direction = "long" if str(row["direction"]).lower() == "buy" else str(row["direction"]).lower()
        if direction not in {"long", "short"}:
            direction = "long" if str(row["direction"]).lower() == "long" else "short"
        out.append({
            "strategy": "v14",
            "source_mode": "v14_native_trade_csv",
            "dataset_key": Path(dataset).name,
            "entry_time": entry_ts.isoformat(),
            "exit_time": _ts(row["exit_datetime"]).isoformat(),
            "signal_time": _ts(row.get("signal_datetime", row["entry_datetime"])).isoformat(),
            "entry_session": session,
            "side": "buy" if direction == "long" else "sell",
            "direction": direction,
            "pips": float(row["pips"]),
            "usd": float(row.get("usd", row.get("pnl_usd", 0.0))),
            "exit_reason": str(row.get("exit_reason", "unknown")),
            "ownership_cell": str(cell_info["ownership_cell"]),
            "setup_type": None,
            "evaluator_mode": "tokyo_meanrev_native_trade",
            "timing_gate": _timing_gate_for_minutes(minutes, {"strategy": "v14"}),
            "timing_minutes": round(float(minutes), 2) if minutes is not None else None,
            "standalone_entry_equity": float(row.get("equity_before", 100000.0)),
            "raw": row,
            "confluence_score": int(row.get("confluence_score", 0)),
            "confluence_combo": str(row.get("confluence_combo", "")),
            "quality_label": str(row.get("quality_label", "")),
            "day_of_week": str(row.get("day_of_week", "")),
            "hour_utc": int(row.get("hour_utc", 0)) if not pd.isna(row.get("hour_utc", 0)) else None,
            "size_scale": float(row.get("size_mult_total", 1.0) or 1.0),
        })
    return out


def _normalize_v44_trades(dataset: str) -> list[dict[str, Any]]:
    dataset_key = "500k" if "500k" in Path(dataset).name else "1000k"
    report_path = OUT_DIR / f"phase1_v44_baseline_{dataset_key}_report.json"
    if not report_path.exists():
        return []
    data = json.loads(report_path.read_text(encoding="utf-8"))
    closed = data["results"].get("closed_trades", [])
    entries = [_ts(x["entry_time"]) for x in closed]
    cells = _classify_times_to_cells(dataset, entries)
    out = []
    for row, cell_info in zip(closed, cells):
        entry_ts = _ts(row["entry_time"])
        session = "ny" if str(row.get("entry_session")) == "ny_overlap" else str(row.get("entry_session", "ny"))
        minutes = _session_open_minutes(entry_ts, session)
        direction = "long" if str(row["side"]).lower() == "buy" else "short"
        out.append({
            "strategy": "v44_ny",
            "source_mode": "v44_native_trade_report",
            "dataset_key": Path(dataset).name,
            "entry_time": entry_ts.isoformat(),
            "exit_time": _ts(row["exit_time"]).isoformat(),
            "signal_time": entry_ts.isoformat(),
            "entry_session": session,
            "side": str(row["side"]),
            "direction": direction,
            "pips": float(row["pips"]),
            "usd": float(row["usd"]),
            "exit_reason": str(row.get("exit_reason", "unknown")),
            "ownership_cell": str(cell_info["ownership_cell"]),
            "setup_type": None,
            "evaluator_mode": str(row.get("entry_signal_mode", "native_trade")),
            "timing_gate": _timing_gate_for_minutes(minutes, {"strategy": "v44_ny"}),
            "timing_minutes": round(float(minutes), 2) if minutes is not None else None,
            "standalone_entry_equity": 100000.0,
            "raw": row,
            "entry_signal_mode": str(row.get("entry_signal_mode", "")),
            "entry_profile": str(row.get("entry_profile", "")),
            "entry_mode": int(row.get("entry_mode", 0)),
            "size_scale": float(row.get("conviction_scale", 1.0) or 1.0),
        })
    return out


def _load_all_normalized_trades(strategies: set[str] | None = None) -> dict[str, dict[str, list[dict[str, Any]]]]:
    wanted = strategies or {"london_v2", "v14", "v44_ny"}
    out: dict[str, dict[str, list[dict[str, Any]]]] = {"500k": {}, "1000k": {}}
    for dk, dataset in DATASETS.items():
        if "london_v2" in wanted:
            out[dk]["london_v2"] = _normalize_london_trades(dataset)
        if "v14" in wanted:
            out[dk]["v14"] = _normalize_v14_trades(dataset)
        if "v44_ny" in wanted:
            out[dk]["v44_ny"] = _normalize_v44_trades(dataset)
    return out


def _passes_filters(trade: dict[str, Any], spec: OffensiveSliceSpec) -> bool:
    if trade["strategy"] != spec.strategy:
        return False
    if spec.ownership_cells and trade.get("ownership_cell") not in spec.ownership_cells:
        return False
    if spec.direction and trade.get("direction") != spec.direction:
        return False
    if spec.setup_type and str(trade.get("setup_type") or "") != spec.setup_type:
        return False
    if spec.session_gate and str(trade.get("entry_session") or "") != spec.session_gate:
        return False
    if spec.timing_gate and str(trade.get("timing_gate") or "") != spec.timing_gate:
        return False
    fg = spec.feature_gates
    if "native_allowed" in fg and bool(trade.get("native_allowed", False)) != bool(fg["native_allowed"]):
        return False
    if "confluence_score_min" in fg and int(trade.get("confluence_score", 0)) < int(fg["confluence_score_min"]):
        return False
    if "quality_labels" in fg and str(trade.get("quality_label", "")) not in set(fg["quality_labels"]):
        return False
    if "confluence_combos" in fg and str(trade.get("confluence_combo", "")) not in set(fg["confluence_combos"]):
        return False
    if "entry_signal_modes" in fg and str(trade.get("entry_signal_mode", "")) not in set(fg["entry_signal_modes"]):
        return False
    if "entry_profiles" in fg and str(trade.get("entry_profile", "")) not in set(fg["entry_profiles"]):
        return False
    if "day_of_week_in" in fg and str(trade.get("day_of_week", "")) not in set(fg["day_of_week_in"]):
        return False
    return True


def _metrics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    if not trades:
        return {
            "trade_count": 0,
            "avg_pips": 0.0,
            "total_pips": 0.0,
            "avg_usd": 0.0,
            "total_usd": 0.0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
        }
    pips = [float(t["pips"]) for t in trades]
    usd = [float(t["usd"]) for t in trades]
    wins = [x for x in usd if x > 0]
    losses = [x for x in usd if x <= 0]
    gp = sum(wins) if wins else 0.0
    gl = abs(sum(losses)) if losses else 0.0
    pf = round(gp / gl, 4) if gl > 0 else (999.0 if gp > 0 else 0.0)
    return {
        "trade_count": len(trades),
        "avg_pips": round(statistics.mean(pips), 3),
        "total_pips": round(sum(pips), 2),
        "avg_usd": round(statistics.mean(usd), 2),
        "total_usd": round(sum(usd), 2),
        "win_rate_pct": round(100.0 * sum(1 for x in usd if x > 0) / len(usd), 2),
        "profit_factor": pf,
    }


def _reference_slice() -> OffensiveSliceSpec:
    return OffensiveSliceSpec(
        strategy="london_v2",
        setup_type="D",
        direction="long",
        ownership_cells=("mean_reversion/er_low/der_neg",),
        timing_gate="first_30min",
        session_gate="london",
        state_realism_mode="london_setupd_stateful_trade_outcome",
        feature_gates={"native_allowed": True},
        size_scale=1.0,
        source_mode="london_stateful_trade_sim",
        notes="Validated reference slice",
        slice_id="reference__london_v2__setup_D__long__first_30min__cells_mean_reversion_er_low_der_neg",
    )


def _generate_london_specs(trades_by_dataset: dict[str, list[dict[str, Any]]]) -> list[OffensiveSliceSpec]:
    cells = sorted({t["ownership_cell"] for trades in trades_by_dataset.values() for t in trades if t.get("native_allowed", False)})
    specs: list[OffensiveSliceSpec] = []
    for cell in cells:
        for timing in [None, "first_30min"]:
            spec = OffensiveSliceSpec(
                strategy="london_v2",
                setup_type="D",
                direction="long",
                ownership_cells=(cell,),
                timing_gate=timing,
                session_gate="london",
                state_realism_mode="london_setupd_stateful_trade_outcome",
                feature_gates={"native_allowed": True},
                source_mode="london_stateful_trade_sim",
                notes="London Setup D slice",
            )
            specs.append(spec)
    return specs


def _top_values(trades_by_dataset: dict[str, list[dict[str, Any]]], key: str, min_count: int = 4, top_n: int = 5) -> list[str]:
    counter = Counter()
    for trades in trades_by_dataset.values():
        for t in trades:
            val = str(t.get(key, ""))
            if val:
                counter[val] += 1
    return [v for v, c in counter.most_common() if c >= min_count][:top_n]


def _generate_v14_specs(trades_by_dataset: dict[str, list[dict[str, Any]]]) -> list[OffensiveSliceSpec]:
    cells = sorted({t["ownership_cell"] for trades in trades_by_dataset.values() for t in trades})
    combos = _top_values(trades_by_dataset, "confluence_combo", min_count=5, top_n=4)
    specs: list[OffensiveSliceSpec] = []
    for cell in cells:
        for direction in ["long", "short"]:
            specs.append(OffensiveSliceSpec(
                strategy="v14",
                direction=direction,
                ownership_cells=(cell,),
                session_gate="tokyo",
                state_realism_mode="native_trade_proxy",
                source_mode="v14_native_trade_csv",
                notes="V14 base cell slice",
            ))
            specs.append(OffensiveSliceSpec(
                strategy="v14",
                direction=direction,
                ownership_cells=(cell,),
                session_gate="tokyo",
                state_realism_mode="native_trade_proxy",
                source_mode="v14_native_trade_csv",
                feature_gates={"confluence_score_min": 5},
                notes="V14 high-confluence cell slice",
            ))
            specs.append(OffensiveSliceSpec(
                strategy="v14",
                direction=direction,
                ownership_cells=(cell,),
                session_gate="tokyo",
                state_realism_mode="native_trade_proxy",
                source_mode="v14_native_trade_csv",
                feature_gates={"quality_labels": ["high"]},
                notes="V14 high-quality cell slice",
            ))
            for combo in combos:
                specs.append(OffensiveSliceSpec(
                    strategy="v14",
                    direction=direction,
                    ownership_cells=(cell,),
                    session_gate="tokyo",
                    state_realism_mode="native_trade_proxy",
                    source_mode="v14_native_trade_csv",
                    feature_gates={"confluence_combos": [combo]},
                    notes="V14 combo slice",
                ))
    return specs


def _generate_v44_specs(trades_by_dataset: dict[str, list[dict[str, Any]]]) -> list[OffensiveSliceSpec]:
    cells = sorted({t["ownership_cell"] for trades in trades_by_dataset.values() for t in trades})
    modes = _top_values(trades_by_dataset, "entry_signal_mode", min_count=8, top_n=4)
    profiles = _top_values(trades_by_dataset, "entry_profile", min_count=8, top_n=3)
    specs: list[OffensiveSliceSpec] = []
    for cell in cells:
        for direction in ["long", "short"]:
            specs.append(OffensiveSliceSpec(
                strategy="v44_ny",
                direction=direction,
                ownership_cells=(cell,),
                state_realism_mode="native_trade_proxy",
                source_mode="v44_native_trade_report",
                notes="V44 base cell slice",
            ))
            for mode in modes:
                specs.append(OffensiveSliceSpec(
                    strategy="v44_ny",
                    direction=direction,
                    ownership_cells=(cell,),
                    state_realism_mode="native_trade_proxy",
                    source_mode="v44_native_trade_report",
                    feature_gates={"entry_signal_modes": [mode]},
                    notes="V44 signal-mode slice",
                ))
            for profile in profiles:
                specs.append(OffensiveSliceSpec(
                    strategy="v44_ny",
                    direction=direction,
                    ownership_cells=(cell,),
                    state_realism_mode="native_trade_proxy",
                    source_mode="v44_native_trade_report",
                    feature_gates={"entry_profiles": [profile]},
                    notes="V44 profile slice",
                ))
    return specs


def _slice_dataset_summary(spec: OffensiveSliceSpec, trades: list[dict[str, Any]], coverage: dict[str, dict[str, Any]], additive_result: dict[str, Any]) -> dict[str, Any]:
    cell = spec.ownership_cells[0] if spec.ownership_cells else "multiple"
    coverage_info = coverage.get(cell, {
        "variant_k_trade_count": 0,
        "variant_k_by_strategy": {},
        "variant_k_net_pips": 0.0,
        "coverage_class": "whitespace",
    })
    metrics = _metrics(trades)
    raw_count = len(trades)
    return {
        "slice_spec": spec.as_dict(),
        "ownership_cell": cell,
        "coverage": coverage_info,
        "raw_candidate_count": raw_count,
        "state_survived_count": raw_count,
        "standalone": metrics,
        "additive": additive_result,
        "stability_note": "positive" if metrics["avg_pips"] > 0 else ("flat" if abs(metrics["avg_pips"]) < 1e-9 else "negative"),
    }


def _classify_board_entry(per_dataset: dict[str, dict[str, Any]]) -> tuple[str, str]:
    ds500 = per_dataset.get("500k")
    ds1000 = per_dataset.get("1000k")
    if not ds500 or not ds1000:
        return "not_enough_data", "missing one dataset"
    cov_classes = {ds500["coverage"]["coverage_class"], ds1000["coverage"]["coverage_class"]}
    standalone_pos_both = ds500["standalone"]["avg_pips"] > 0 and ds1000["standalone"]["avg_pips"] > 0
    additive_pos_both = ds500["additive"]["delta_vs_baseline"]["net_usd"] > 0 and ds1000["additive"]["delta_vs_baseline"]["net_usd"] > 0
    additive_pf_ok_both = ds500["additive"]["delta_vs_baseline"]["profit_factor"] >= 0 and ds1000["additive"]["delta_vs_baseline"]["profit_factor"] >= 0
    dd_ok_both = ds500["additive"]["delta_vs_baseline"]["max_drawdown_usd"] <= 50 and ds1000["additive"]["delta_vs_baseline"]["max_drawdown_usd"] <= 50
    additive_count_total = ds500["additive"]["selection_counts"]["new_additive_trades_count"] + ds1000["additive"]["selection_counts"]["new_additive_trades_count"]
    standalone_count_total = ds500["standalone"]["trade_count"] + ds1000["standalone"]["trade_count"]
    redundant_heavy = cov_classes == {"covered"} and ds500["additive"]["delta_vs_baseline"]["profit_factor"] < 0 and ds1000["additive"]["delta_vs_baseline"]["profit_factor"] < 0
    if standalone_count_total < 3 or additive_count_total == 0:
        return "not_enough_data", "too few additive or standalone trades"
    if standalone_pos_both and additive_pos_both and additive_pf_ok_both and dd_ok_both:
        if cov_classes <= {"whitespace", "thin"}:
            return "pilot_candidate", "positive standalone and additive-safe in whitespace/thin coverage"
        return "research_followup", "positive standalone and additive-safe but not whitespace/thin"
    if redundant_heavy:
        return "redundant", "covered cell with additive-negative portfolio effect"
    if ds500["standalone"]["avg_pips"] <= 0 and ds1000["standalone"]["avg_pips"] <= 0:
        return "dead", "standalone edge is negative on both datasets"
    if standalone_pos_both:
        return "research_followup", "standalone-positive but additive gate not fully cleared"
    return "dead", "does not clear additive-safe evidence bar"


def _reference_regression(results_by_slice: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    ref = _reference_slice().resolved_slice_id()
    observed = results_by_slice.get(ref)
    expected = json.loads(REFERENCE_ARTIFACT.read_text(encoding="utf-8"))["results"]
    if not observed:
        return {"passed": False, "reason": "reference slice missing from discovery output"}
    checks = {}
    for dk in ["500k", "1000k"]:
        exp_var = expected[dk]["variants"]["100pct"]
        obs = observed[dk]
        checks[dk] = {
            "new_additive_matches": obs["additive"]["selection_counts"]["new_additive_trades_count"] == exp_var["selection_counts"]["new_additive_trades_count"],
            "exact_overlap_matches": obs["additive"]["selection_counts"]["exact_baseline_overlap_count"] == exp_var["selection_counts"]["exact_baseline_overlap_count"],
            "delta_net_usd_matches": abs(obs["additive"]["delta_vs_baseline"]["net_usd"] - exp_var["delta_vs_baseline"]["net_usd"]) < 0.01,
        }
    passed = all(all(v.values()) for v in checks.values())
    return {"passed": passed, "checks": checks}


def _build_board(results_by_slice: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    rows = []
    for slice_id, per_dataset in results_by_slice.items():
        classification, reason = _classify_board_entry(per_dataset)
        ds500 = per_dataset.get("500k", {})
        ds1000 = per_dataset.get("1000k", {})
        rows.append({
            "slice_id": slice_id,
            "slice_spec": (ds500.get("slice_spec") or ds1000.get("slice_spec")),
            "classification": classification,
            "reason": reason,
            "500k": ds500,
            "1000k": ds1000,
            "score": round(
                float(ds500.get("additive", {}).get("delta_vs_baseline", {}).get("net_usd", 0.0))
                + float(ds1000.get("additive", {}).get("delta_vs_baseline", {}).get("net_usd", 0.0))
                + 500.0 * max(0.0, float(ds500.get("additive", {}).get("delta_vs_baseline", {}).get("profit_factor", 0.0)))
                + 500.0 * max(0.0, float(ds1000.get("additive", {}).get("delta_vs_baseline", {}).get("profit_factor", 0.0))),
                2,
            ),
        })
    rows.sort(key=lambda r: (-r["score"], r["slice_id"]))
    return {
        "active_pilot_candidate": [r for r in rows if r["classification"] == "pilot_candidate"][:3],
        "next_additive_tests": [r for r in rows if r["classification"] == "research_followup"][:10],
        "high_quality_standalone_but_additive_unproven": [
            r for r in rows
            if r["classification"] == "not_enough_data" and (
                (r.get("500k", {}).get("standalone", {}).get("avg_pips", 0.0) > 0)
                or (r.get("1000k", {}).get("standalone", {}).get("avg_pips", 0.0) > 0)
            )
        ][:10],
        "redundant_dead_stop": [r for r in rows if r["classification"] in {"redundant", "dead"}][:20],
        "all_ranked": rows,
    }


def _render_board_md(board: dict[str, Any], regression: dict[str, Any]) -> str:
    lines = ["# Offensive Slice Discovery Board", "", "## Regression", ""]
    lines.append(f"- Reference slice regression passed: `{regression['passed']}`")
    if not regression.get("passed"):
        lines.append(f"- Detail: `{regression}``")
    sections = [
        ("Active pilot candidate", "active_pilot_candidate"),
        ("Next additive tests", "next_additive_tests"),
        ("High-quality standalone but additive-unproven", "high_quality_standalone_but_additive_unproven"),
        ("Redundant / dead / stop", "redundant_dead_stop"),
    ]
    for title, key in sections:
        lines.extend(["", f"## {title}", ""])
        rows = board.get(key, [])
        if not rows:
            lines.append("- None")
            continue
        for row in rows:
            spec = row["slice_spec"]
            lines.append(
                f"- `{row['slice_id']}`: `{row['classification']}` | `{spec['strategy']}` | cell `{spec['ownership_cells'][0] if spec['ownership_cells'] else 'multiple'}` | "
                f"500k delta USD `{row['500k']['additive']['delta_vs_baseline']['net_usd']}` PF `{row['500k']['additive']['delta_vs_baseline']['profit_factor']}` | "
                f"1000k delta USD `{row['1000k']['additive']['delta_vs_baseline']['net_usd']}` PF `{row['1000k']['additive']['delta_vs_baseline']['profit_factor']}` | reason: {row['reason']}"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    if args.reference_only:
        specs = [_reference_slice()]
        needed_strategies = {"london_v2"}
    else:
        needed_strategies = {"london_v2", "v14", "v44_ny"}
        normalized = _load_all_normalized_trades(needed_strategies)
        london_specs = _generate_london_specs({dk: normalized[dk].get("london_v2", []) for dk in DATASETS})
        v14_specs = _generate_v14_specs({dk: normalized[dk].get("v14", []) for dk in DATASETS})
        v44_specs = _generate_v44_specs({dk: normalized[dk].get("v44_ny", []) for dk in DATASETS})
        specs = [_reference_slice(), *london_specs, *v14_specs, *v44_specs]

    normalized = _load_all_normalized_trades(needed_strategies)
    baseline_contexts = {dk: build_baseline_context(path) for dk, path in DATASETS.items()}
    coverage = {dk: _load_variant_k_coverage(path) for dk, path in DATASETS.items()}

    dedup = {}
    for spec in specs:
        dedup[spec.resolved_slice_id()] = spec
    specs = list(dedup.values())

    matrix = {
        "title": "Offensive slice discovery matrix",
        "datasets": list(DATASETS.keys()),
        "assumptions": {
            "search_scope": "all archetypes",
            "evidence_bar": "additive_safe_only",
            "london_source_mode": "stateful_trade_outcome_sim",
            "v14_source_mode": "native_trade_csv_proxy",
            "v44_source_mode": "native_trade_report_proxy",
        },
        "results": {},
    }
    results_by_slice: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

    for spec in specs:
        slice_id = spec.resolved_slice_id()
        matrix["results"][slice_id] = {"slice_spec": spec.as_dict(), "per_dataset": {}}
        for dk in ["500k", "1000k"]:
            trades = [t for t in normalized[dk][spec.strategy] if _passes_filters(t, spec)]
            additive = run_slice_additive(
                baseline_ctx=baseline_contexts[dk],
                slice_spec=spec.as_dict(),
                selected_trades=trades,
                size_scale=spec.size_scale,
            )
            ds_summary = _slice_dataset_summary(spec, trades, coverage[dk], additive)
            matrix["results"][slice_id]["per_dataset"][dk] = ds_summary
            results_by_slice[slice_id][dk] = ds_summary

    regression = _reference_regression(results_by_slice)
    board = _build_board(results_by_slice)
    board_payload = {
        "title": "Offensive slice discovery board",
        "regression": regression,
        "board": board,
    }

    Path(args.output_json).write_text(json.dumps(matrix, indent=2, default=_json_default), encoding="utf-8")
    Path(args.output_board_json).write_text(json.dumps(board_payload, indent=2, default=_json_default), encoding="utf-8")
    Path(args.output_board_md).write_text(_render_board_md(board, regression), encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_board_json}")
    print(f"Wrote {args.output_board_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
