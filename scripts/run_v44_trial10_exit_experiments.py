#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

ROOT = Path("/Users/codygnon/Documents/usdjpy_assistant")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import backtest_session_momentum as sm

OUT_DIR = ROOT / "research_out"
BASE_CONFIG_PATH = OUT_DIR / "session_momentum_v44_base_config.json"
DEFAULT_DATASET = OUT_DIR / "USDJPY_M1_OANDA_1000k.csv"


def build_args(config: dict[str, Any]) -> SimpleNamespace:
    merged = dict(sm.DEFAULTS)
    merged.update(config)
    merged.setdefault("version", "v5")
    merged.setdefault("mode", "session")
    merged["inputs"] = [str(Path(p)) for p in config.get("inputs", [str(DEFAULT_DATASET)])]
    return SimpleNamespace(**merged)


def filter_v44_trades(results: dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame(results.get("closed_trades", []))
    if df.empty:
        return df
    return df[df["entry_session"].astype(str) == "ny_overlap"].copy().reset_index(drop=True)


def summarize_trades(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "net_usd": 0.0,
            "net_pips": 0.0,
            "profit_factor": None,
            "max_drawdown_usd": 0.0,
            "max_drawdown_pct": 0.0,
        }
    work = df.copy()
    work["usd"] = pd.to_numeric(work["usd"], errors="coerce").fillna(0.0)
    work["pips"] = pd.to_numeric(work["pips"], errors="coerce").fillna(0.0)
    work["entry_time"] = pd.to_datetime(work["entry_time"], utc=True, errors="coerce")
    work = work.sort_values("entry_time").reset_index(drop=True)
    work["cum_usd"] = work["usd"].cumsum()
    work["cum_peak"] = work["cum_usd"].cummax()
    work["dd_usd"] = work["cum_peak"] - work["cum_usd"]
    starting_equity = float(DEFAULT_BASE_CONFIG.get("v5_account_size", 100000.0))
    gross_win = float(work.loc[work["usd"] > 0, "usd"].sum())
    gross_loss = abs(float(work.loc[work["usd"] < 0, "usd"].sum()))
    max_dd = float(work["dd_usd"].max()) if len(work) else 0.0
    return {
        "trades": int(len(work)),
        "wins": int((work["usd"] > 0).sum()),
        "losses": int((work["usd"] <= 0).sum()),
        "win_rate": round(float((work["usd"] > 0).mean()) * 100.0, 3),
        "net_usd": round(float(work["usd"].sum()), 2),
        "net_pips": round(float(work["pips"].sum()), 2),
        "profit_factor": round(gross_win / gross_loss, 4) if gross_loss > 0 else None,
        "max_drawdown_usd": round(max_dd, 2),
        "max_drawdown_pct": round((max_dd / starting_equity) * 100.0, 3) if starting_equity > 0 else 0.0,
    }


def grouped_breakdown(df: pd.DataFrame, col: str, label: str) -> list[dict[str, Any]]:
    if df.empty or col not in df.columns:
        return []
    work = df.copy()
    work["usd"] = pd.to_numeric(work["usd"], errors="coerce").fillna(0.0)
    work["pips"] = pd.to_numeric(work["pips"], errors="coerce").fillna(0.0)
    out = (
        work.groupby(col, dropna=False)
        .agg(
            trades=("trade_id", "size"),
            wins=("usd", lambda s: int((s > 0).sum())),
            losses=("usd", lambda s: int((s <= 0).sum())),
            win_rate=("usd", lambda s: float((s > 0).mean()) * 100.0),
            net_usd=("usd", "sum"),
            net_pips=("pips", "sum"),
        )
        .reset_index()
        .rename(columns={col: label})
        .sort_values("net_usd", ascending=False)
    )
    return out.to_dict("records")


def trade_compare_key(row: pd.Series) -> str:
    return "|".join(
        [
            str(row.get("entry_time")),
            str(row.get("side")),
            str(row.get("entry_profile")),
            str(row.get("entry_signal_mode")),
            str(row.get("entry_regime")),
        ]
    )


def compare_trade_sets(baseline_df: pd.DataFrame, variant_df: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    b = baseline_df.copy()
    v = variant_df.copy()
    if not b.empty:
        b["trade_key"] = b.apply(trade_compare_key, axis=1)
    else:
        b["trade_key"] = pd.Series(dtype=str)
    if not v.empty:
        v["trade_key"] = v.apply(trade_compare_key, axis=1)
    else:
        v["trade_key"] = pd.Series(dtype=str)
    merged = pd.merge(
        b[["trade_key", "usd", "pips", "exit_reason", "entry_time", "side", "entry_profile", "entry_signal_mode"]].rename(
            columns={"usd": "baseline_usd", "pips": "baseline_pips", "exit_reason": "baseline_exit_reason"}
        ),
        v[["trade_key", "usd", "pips", "exit_reason"]].rename(
            columns={"usd": "variant_usd", "pips": "variant_pips", "exit_reason": "variant_exit_reason"}
        ),
        on="trade_key",
        how="outer",
    ).fillna({"baseline_usd": 0.0, "baseline_pips": 0.0, "variant_usd": 0.0, "variant_pips": 0.0})
    merged["usd_delta"] = pd.to_numeric(merged["variant_usd"], errors="coerce").fillna(0.0) - pd.to_numeric(merged["baseline_usd"], errors="coerce").fillna(0.0)
    merged["pips_delta"] = pd.to_numeric(merged["variant_pips"], errors="coerce").fillna(0.0) - pd.to_numeric(merged["baseline_pips"], errors="coerce").fillna(0.0)
    cols = [
        "entry_time",
        "side",
        "entry_profile",
        "entry_signal_mode",
        "baseline_usd",
        "variant_usd",
        "usd_delta",
        "baseline_pips",
        "variant_pips",
        "pips_delta",
        "baseline_exit_reason",
        "variant_exit_reason",
    ]
    improved = merged.sort_values("usd_delta", ascending=False).head(10)[cols].to_dict("records")
    degraded = merged.sort_values("usd_delta", ascending=True).head(10)[cols].to_dict("records")
    return improved, degraded


def make_variant_name(tp1_pips: float, be_extra: float, trail_buf: float, runner_mode: str) -> str:
    be_tag = "spread" if be_extra == 0.0 else f"spread_plus_{str(be_extra).replace('.', 'p')}"
    return f"v44_trial10_exit_tp1_{str(tp1_pips).replace('.', 'p')}_{be_tag}_trail_{str(trail_buf).replace('.', 'p')}_{runner_mode}"


def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
    args = build_args(config)
    return sm.run_backtest_v5(args)


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
parser.add_argument("--tp1-list", default="5.0,6.0")
parser.add_argument("--be-extra-list", default="0.0,0.5")
parser.add_argument("--trail-list", default="3.0,4.0")
parser.add_argument("--runner-modes", default="fixed_tp2_then_trail,trail_only_after_tp1")
parser.add_argument("--skip-existing", action="store_true")
parser.add_argument("--output-tag", default="")
args_cli = parser.parse_args()

DEFAULT_BASE_CONFIG = json.loads(BASE_CONFIG_PATH.read_text(encoding="utf-8"))
DEFAULT_BASE_CONFIG["inputs"] = [str(Path(args_cli.dataset))]
DEFAULT_BASE_CONFIG["version"] = "v5"
DEFAULT_BASE_CONFIG["mode"] = "session"
dataset_tag = str(args_cli.output_tag).strip() or Path(args_cli.dataset).stem

baseline_results = run_experiment(DEFAULT_BASE_CONFIG)
baseline_df = filter_v44_trades(baseline_results)
baseline_summary = summarize_trades(baseline_df)

tp1_values = [float(x.strip()) for x in str(args_cli.tp1_list).split(",") if x.strip()]
be_extra_values = [float(x.strip()) for x in str(args_cli.be_extra_list).split(",") if x.strip()]
trail_values = [float(x.strip()) for x in str(args_cli.trail_list).split(",") if x.strip()]
runner_modes = [x.strip() for x in str(args_cli.runner_modes).split(",") if x.strip()]

reports_index: list[dict[str, Any]] = []

for tp1 in tp1_values:
    for be_extra in be_extra_values:
        for trail in trail_values:
            for runner_mode in runner_modes:
                cfg = dict(DEFAULT_BASE_CONFIG)
                cfg.update(
                    {
                        "v5_trial10_exit_enabled": True,
                        "v5_trial10_exit_tp1_pips": tp1,
                        "v5_trial10_exit_be_extra_pips": be_extra,
                        "v5_trial10_exit_trail_buffer_pips": trail,
                        "v5_trial10_exit_runner_mode": runner_mode,
                        "v5_trial10_exit_apply_profiles": "Strong,Normal",
                        "v5_trial10_exit_skip_news": True,
                    }
                )
                variant_name = make_variant_name(tp1, be_extra, trail, runner_mode)
                file_stem = f"{variant_name}_{dataset_tag}"
                config_path = OUT_DIR / f"{file_stem}_config.json"
                report_path = OUT_DIR / f"{file_stem}_report.json"
                if args_cli.skip_existing and report_path.exists():
                    existing = json.loads(report_path.read_text(encoding="utf-8"))
                    reports_index.append(
                        {
                            "name": variant_name,
                            "config_path": str(config_path),
                            "report_path": str(report_path),
                            "summary": existing.get("variant_summary", {}),
                            "delta_vs_baseline": existing.get("delta_vs_baseline", {}),
                        }
                    )
                    continue
                config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
                variant_results = run_experiment(cfg)
                variant_df = filter_v44_trades(variant_results)
                variant_summary = summarize_trades(variant_df)
                improved, degraded = compare_trade_sets(baseline_df, variant_df)
                report = {
                    "name": variant_name,
                    "dataset": str(Path(args_cli.dataset)),
                    "baseline_summary": baseline_summary,
                    "variant_summary": variant_summary,
                    "delta_vs_baseline": {
                        "net_usd": round(float(variant_summary["net_usd"]) - float(baseline_summary["net_usd"]), 2),
                        "net_pips": round(float(variant_summary["net_pips"]) - float(baseline_summary["net_pips"]), 2),
                        "profit_factor": None
                        if baseline_summary["profit_factor"] is None or variant_summary["profit_factor"] is None
                        else round(float(variant_summary["profit_factor"]) - float(baseline_summary["profit_factor"]), 4),
                        "max_drawdown_usd": round(float(variant_summary["max_drawdown_usd"]) - float(baseline_summary["max_drawdown_usd"]), 2),
                        "max_drawdown_pct": round(float(variant_summary["max_drawdown_pct"]) - float(baseline_summary["max_drawdown_pct"]), 3),
                        "win_rate": round(float(variant_summary["win_rate"]) - float(baseline_summary["win_rate"]), 3),
                    },
                    "by_trend_strength": grouped_breakdown(variant_df, "entry_profile", "trend_strength"),
                    "by_exit_reason": grouped_breakdown(variant_df, "exit_reason", "exit_reason"),
                    "top_improved_trades": improved,
                    "top_degraded_trades": degraded,
                }
                report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
                reports_index.append(
                    {
                        "name": variant_name,
                        "config_path": str(config_path),
                        "report_path": str(report_path),
                        "summary": variant_summary,
                        "delta_vs_baseline": report["delta_vs_baseline"],
                    }
                )

summary_path = OUT_DIR / f"v44_trial10_exit_experiments_index_{dataset_tag}.json"
summary_path.write_text(
    json.dumps(
        {
            "dataset": str(Path(args_cli.dataset)),
            "baseline_summary": baseline_summary,
            "variants": reports_index,
        },
        indent=2,
        default=str,
    ),
    encoding="utf-8",
)

print(f"Wrote baseline/variant index to {summary_path}")
