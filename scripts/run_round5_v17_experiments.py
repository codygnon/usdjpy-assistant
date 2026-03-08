#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


ROOT = Path("/Users/codygnon/Documents/usdjpy_assistant")
ENGINE = ROOT / "scripts" / "backtest_tokyo_meanrev.py"
OUT = ROOT / "research_out"
TMP = Path("/tmp/round5_v17_cfgs")

BASELINE_CFG = OUT / "tokyo_optimized_v15_config.json"

DATA_250 = OUT / "USDJPY_M1_OANDA_250k.csv"
DATA_500 = OUT / "USDJPY_M1_OANDA_500k.csv"
DATA_1000 = OUT / "USDJPY_M1_OANDA_1000k.csv"
DATA_700 = OUT / "USDJPY_M1_OANDA_700k_split.csv"
DATA_300 = OUT / "USDJPY_M1_OANDA_300k_split.csv"

MONTHS_1000K = 32.465
MONTHS_500K = MONTHS_1000K * 0.5
MONTHS_250K = MONTHS_1000K * 0.25
MONTHS_700K = MONTHS_1000K * 0.7
MONTHS_300K = MONTHS_1000K * 0.3


@dataclass
class Variant:
    code: str
    label: str
    apply_fn: Callable[[dict], None]


def deep_clone(x: dict) -> dict:
    return copy.deepcopy(x)


def run_backtest(cfg: dict, tag: str, csv_path: Path, out_prefix: str) -> dict:
    TMP.mkdir(parents=True, exist_ok=True)
    run_cfg = deep_clone(cfg)
    run_cfg["run_sequence"] = [
        {
            "label": out_prefix,
            "input_csv": str(csv_path),
            "output_json": str(OUT / f"{out_prefix}_report.json"),
            "output_trades_csv": str(OUT / f"{out_prefix}_trades.csv"),
            "output_equity_csv": str(OUT / f"{out_prefix}_equity.csv"),
        }
    ]
    cfg_path = TMP / f"{tag}.json"
    cfg_path.write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")
    p = subprocess.run(
        ["python3", str(ENGINE), "--config", str(cfg_path)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if p.returncode != 0:
        raise RuntimeError(
            f"Backtest failed for {tag}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"
        )
    report_path = OUT / f"{out_prefix}_report.json"
    trades_path = OUT / f"{out_prefix}_trades.csv"
    equity_path = OUT / f"{out_prefix}_equity.csv"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    trades = pd.read_csv(trades_path)
    return {
        "config_used": run_cfg,
        "report": report,
        "trades_df": trades,
        "report_path": str(report_path),
        "trades_path": str(trades_path),
        "equity_path": str(equity_path),
    }


def compute_capture_ratio(trades_df: pd.DataFrame) -> float:
    if "mfe_pips" not in trades_df.columns or "pips" not in trades_df.columns:
        return 0.0
    w = trades_df[(trades_df["pips"] > 0) & (trades_df["mfe_pips"] > 0)].copy()
    if w.empty:
        return 0.0
    cap = (w["pips"] / w["mfe_pips"]).replace([np.inf, -np.inf], np.nan).dropna()
    return float(cap.mean()) if not cap.empty else 0.0


def summarize(run: dict, months: float) -> dict:
    rep = run["report"]
    s = rep["summary"]
    tdf = run["trades_df"]
    net = float(s["net_profit_usd"])
    maxdd = float(s["max_drawdown_usd"])
    trades = int(s["total_trades"])
    wins = tdf[tdf["pips"] > 0]
    losses = tdf[tdf["pips"] < 0]
    out = {
        "trades": trades,
        "wr": float(s["win_rate_pct"]),
        "pf": float(s["profit_factor"]),
        "net_usd": net,
        "max_dd_usd": maxdd,
        "max_dd_pct": float(s["max_drawdown_pct"]),
        "net_over_maxdd": (net / maxdd) if maxdd > 0 else 0.0,
        "usd_per_month": (net / months) if months > 0 else 0.0,
        "expectancy_per_trade": (net / trades) if trades > 0 else 0.0,
        "sharpe": float(s.get("sharpe_ratio", 0.0)),
        "calmar": float(s.get("calmar_ratio", 0.0)),
        "blocked_margin_cap": int(rep.get("diagnostics", {}).get("counts", {}).get("blocked_margin_cap", 0)),
        "avg_winner_pips": float(wins["pips"].mean()) if not wins.empty else 0.0,
        "avg_loser_pips": float(losses["pips"].mean()) if not losses.empty else 0.0,
        "capture_ratio": compute_capture_ratio(tdf),
        "exit_reason_breakdown": rep.get("breakdown", {}).get("exit_distribution", []),
        "report_path": run["report_path"],
        "trades_path": run["trades_path"],
        "equity_path": run["equity_path"],
    }
    return out


def pick_winner(rows: list[dict]) -> tuple[dict, str]:
    passers = [r for r in rows if float(r["dev_250k"]["pf"]) > 1.3]
    if passers:
        with_20 = [r for r in passers if int(r["dev_250k"]["trades"]) >= 20]
        pool = with_20 if with_20 else passers
        winner = max(
            pool,
            key=lambda r: (
                float(r["dev_250k"]["net_usd"]),
                float(r["dev_250k"]["pf"]),
                int(r["dev_250k"]["trades"]),
                -float(r["dev_250k"]["max_dd_usd"]),
            ),
        )
        reason = "250k hard gate PF>1.3 passed; ranked by Net USD, PF, trade count, lower MaxDD"
        return winner, reason

    winner = max(
        rows,
        key=lambda r: (
            float(r["dev_250k"]["pf"]),
            float(r["dev_250k"]["net_usd"]),
            int(r["dev_250k"]["trades"]),
            -float(r["dev_250k"]["max_dd_usd"]),
        ),
    )
    reason = "No variant passed 250k PF>1.3; fallback winner = highest PF, then Net USD, trade count, lower MaxDD"
    return winner, reason


def reset_entry_window(cfg: dict) -> None:
    sf = cfg.setdefault("session_filter", {})
    sf.pop("entry_start_utc", None)
    sf.pop("entry_end_utc", None)
    sf.pop("block_new_entries_minutes_before_end", None)


def reset_late_session(cfg: dict) -> None:
    lsm = cfg.setdefault("late_session_management", {})
    lsm["enabled"] = False
    lsm["minutes_before_end"] = int(lsm.get("minutes_before_end", 45))
    lsm["close_if_no_tp1_and_pips_below"] = float(lsm.get("close_if_no_tp1_and_pips_below", -2.0))
    lsm["tp1_hit_tighten_trail_pips"] = float(lsm.get("tp1_hit_tighten_trail_pips", 3.0))
    lsm["hard_close_all_minutes_before_end"] = 0
    lsm["be_or_close_minutes_before_end"] = 0
    lsm["be_min_profit_pips"] = 1.0
    lsm["be_offset_pips"] = 0.0
    lsm["profit_tighten_minutes_before_end"] = 0
    lsm["profit_tighten_trail_mult"] = 0.5


def reset_day_risk(cfg: dict) -> None:
    cfg.setdefault("position_sizing", {}).pop("day_risk_multipliers", None)
    cfg.setdefault("session_filter", {})["allowed_trading_days"] = ["Tuesday", "Friday"]
    adx = cfg.setdefault("adx_filter", {})
    adx.pop("day_max_by_day", None)
    adx.pop("day_overrides", None)


def varset_exp33() -> list[Variant]:
    def a(c: dict) -> None:
        reset_entry_window(c)

    def b(c: dict) -> None:
        reset_entry_window(c)
        c.setdefault("session_filter", {})["entry_start_utc"] = "18:00"

    def cc(c: dict) -> None:
        reset_entry_window(c)
        c.setdefault("session_filter", {})["entry_start_utc"] = "18:00"
        c.setdefault("session_filter", {})["entry_end_utc"] = "20:00"

    def d(c: dict) -> None:
        reset_entry_window(c)
        c.setdefault("session_filter", {})["entry_start_utc"] = "18:00"
        c.setdefault("session_filter", {})["entry_end_utc"] = "21:00"

    def e(c: dict) -> None:
        reset_entry_window(c)
        c.setdefault("session_filter", {})["block_new_entries_minutes_before_end"] = 60

    def f(c: dict) -> None:
        reset_entry_window(c)
        c.setdefault("session_filter", {})["entry_start_utc"] = "18:00"
        c.setdefault("session_filter", {})["block_new_entries_minutes_before_end"] = 60

    return [
        Variant("33A", "baseline", a),
        Variant("33B", "entry_start=18:00", b),
        Variant("33C", "entry 18:00-20:00", cc),
        Variant("33D", "entry 18:00-21:00", d),
        Variant("33E", "block final 60m", e),
        Variant("33F", "start18 + block final60", f),
    ]


def varset_exp34() -> list[Variant]:
    def a(c: dict) -> None:
        reset_late_session(c)

    def b(c: dict) -> None:
        reset_late_session(c)
        c.setdefault("late_session_management", {})["hard_close_all_minutes_before_end"] = 15

    def cc(c: dict) -> None:
        reset_late_session(c)
        c.setdefault("late_session_management", {})["hard_close_all_minutes_before_end"] = 30

    def d(c: dict) -> None:
        reset_late_session(c)
        l = c.setdefault("late_session_management", {})
        l["profit_tighten_minutes_before_end"] = 30
        l["profit_tighten_trail_mult"] = 0.5

    def e(c: dict) -> None:
        reset_late_session(c)
        l = c.setdefault("late_session_management", {})
        l["be_or_close_minutes_before_end"] = 30
        l["be_min_profit_pips"] = 1.0
        l["be_offset_pips"] = 0.0

    def f(c: dict) -> None:
        reset_late_session(c)
        l = c.setdefault("late_session_management", {})
        l["be_or_close_minutes_before_end"] = 30
        l["be_min_profit_pips"] = 1.0
        l["be_offset_pips"] = 0.0
        l["hard_close_all_minutes_before_end"] = 15

    return [
        Variant("34A", "baseline", a),
        Variant("34B", "hard close T-15", b),
        Variant("34C", "hard close T-30", cc),
        Variant("34D", "last30 tighten trail x0.5", d),
        Variant("34E", "T-30 BE-or-close", e),
        Variant("34F", "T-30 BE-or-close + T-15 hard close", f),
    ]


def varset_exp35() -> list[Variant]:
    def set_tp_trail(c: dict, tp_mult: float = 1.0, trail_mult: float = 1.0) -> None:
        tp = c.setdefault("exit_rules", {}).setdefault("take_profit", {})
        tr = c.setdefault("exit_rules", {}).setdefault("trailing_stop", {})
        tp["partial_tp_atr_mult"] = float(tp.get("partial_tp_atr_mult", 0.5)) * float(tp_mult)
        tr["trail_distance_pips"] = float(tr.get("trail_distance_pips", 8.0)) * float(trail_mult)

    return [
        Variant("35A", "baseline", lambda c: None),
        Variant("35B", "TP1 x1.25", lambda c: set_tp_trail(c, tp_mult=1.25, trail_mult=1.0)),
        Variant("35C", "trail x0.75", lambda c: set_tp_trail(c, tp_mult=1.0, trail_mult=0.75)),
        Variant("35D", "TP1 x1.25 + trail x0.75", lambda c: set_tp_trail(c, tp_mult=1.25, trail_mult=0.75)),
        Variant("35E", "TP1 x1.50", lambda c: set_tp_trail(c, tp_mult=1.50, trail_mult=1.0)),
        Variant("35F", "trail x0.50", lambda c: set_tp_trail(c, tp_mult=1.0, trail_mult=0.50)),
        Variant("35G", "TP1 x1.25 + trail x0.50", lambda c: set_tp_trail(c, tp_mult=1.25, trail_mult=0.50)),
    ]


def varset_exp36() -> list[Variant]:
    def a(c: dict) -> None:
        reset_day_risk(c)

    def b(c: dict) -> None:
        reset_day_risk(c)
        c.setdefault("session_filter", {})["allowed_trading_days"] = ["Tuesday", "Wednesday", "Friday"]
        c.setdefault("position_sizing", {})["day_risk_multipliers"] = {"Wednesday": 0.5}

    def cc(c: dict) -> None:
        reset_day_risk(c)
        c.setdefault("session_filter", {})["allowed_trading_days"] = ["Tuesday", "Wednesday", "Friday"]
        c.setdefault("position_sizing", {})["day_risk_multipliers"] = {"Wednesday": 0.33}

    def d(c: dict) -> None:
        reset_day_risk(c)
        c.setdefault("session_filter", {})["allowed_trading_days"] = ["Tuesday", "Wednesday", "Friday"]
        c.setdefault("position_sizing", {})["day_risk_multipliers"] = {"Tuesday": 1.33, "Wednesday": 0.5}

    def e(c: dict) -> None:
        reset_day_risk(c)
        c.setdefault("session_filter", {})["allowed_trading_days"] = ["Tuesday", "Friday"]
        c.setdefault("position_sizing", {})["day_risk_multipliers"] = {"Tuesday": 1.33}

    return [
        Variant("36A", "baseline Tue/Fri", a),
        Variant("36B", "add Wed 0.5x", b),
        Variant("36C", "add Wed 0.33x", cc),
        Variant("36D", "Tue 1.33x + Wed 0.5x", d),
        Variant("36E", "Tue 1.33x only", e),
    ]


EXPERIMENTS: list[tuple[int, list[Variant]]] = [
    (33, varset_exp33()),
    (34, varset_exp34()),
    (35, varset_exp35()),
    (36, varset_exp36()),
]


def run_experiment(exp_num: int, base_cfg: dict, variants: list[Variant]) -> tuple[dict, dict]:
    rows: list[dict] = []
    cfg_by_code: dict[str, dict] = {}

    for v in variants:
        cfg = deep_clone(base_cfg)
        v.apply_fn(cfg)
        dev_run = run_backtest(cfg, tag=f"round5_exp{exp_num}_{v.code}_250k", csv_path=DATA_250, out_prefix=f"round5_exp{exp_num}_{v.code}_250k")
        dev = summarize(dev_run, months=MONTHS_250K)
        row = {
            "variant": v.code,
            "label": v.label,
            "dev_250k": dev,
            "trade_count_lt15_warning": bool(dev["trades"] < 15),
            "hard_gate_pass": bool(dev["pf"] > 1.3),
        }
        rows.append(row)
        cfg_by_code[v.code] = cfg

    winner, reason = pick_winner(rows)
    winner_cfg = cfg_by_code[winner["variant"]]

    # Validate winner forward only.
    val500_run = run_backtest(winner_cfg, tag=f"round5_exp{exp_num}_{winner['variant']}_500k", csv_path=DATA_500, out_prefix=f"round5_exp{exp_num}_{winner['variant']}_500k")
    val1000_run = run_backtest(winner_cfg, tag=f"round5_exp{exp_num}_{winner['variant']}_1000k", csv_path=DATA_1000, out_prefix=f"round5_exp{exp_num}_{winner['variant']}_1000k")
    winner["val_500k"] = summarize(val500_run, months=MONTHS_500K)
    winner["val_1000k"] = summarize(val1000_run, months=MONTHS_1000K)

    payload = {
        "experiment": exp_num,
        "selection_rule": {
            "train_set_only": "250k",
            "hard_gate": "250k PF > 1.3",
            "ranking": ["250k net_usd", "250k pf", "250k trades (>=20 preferred)", "250k lower max_dd_usd"],
            "fallback": "if none pass hard gate, choose highest 250k PF",
        },
        "variants": rows,
        "winner": winner,
        "winner_selection_reason": reason,
    }
    (OUT / f"experiment_{exp_num}_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return winner_cfg, payload


def run_scaling(cfg: dict) -> dict:
    out = {}
    for label, data, months in [
        ("250k", DATA_250, MONTHS_250K),
        ("500k", DATA_500, MONTHS_500K),
        ("1000k", DATA_1000, MONTHS_1000K),
    ]:
        run = run_backtest(cfg, tag=f"tokyo_optimized_v17_{label}", csv_path=data, out_prefix=f"tokyo_optimized_v17_{label}")
        out[label] = summarize(run, months=months)
    out["pf_stddev"] = float(statistics.pstdev([out["250k"]["pf"], out["500k"]["pf"], out["1000k"]["pf"]]))
    return out


def run_walkforward(cfg: dict) -> dict:
    tr = run_backtest(cfg, tag="tokyo_optimized_v17_walkforward_700k", csv_path=DATA_700, out_prefix="tokyo_optimized_v17_walkforward_700k")
    te = run_backtest(cfg, tag="tokyo_optimized_v17_walkforward_300k", csv_path=DATA_300, out_prefix="tokyo_optimized_v17_walkforward_300k")
    tr_s = summarize(tr, months=MONTHS_700K)
    te_s = summarize(te, months=MONTHS_300K)
    retention = (te_s["pf"] / tr_s["pf"] * 100.0) if tr_s["pf"] > 0 else 0.0
    comp = {"train_700k": tr_s, "test_300k": te_s, "pf_retention_pct": retention}
    (OUT / "tokyo_optimized_v17_walkforward_comparison.json").write_text(json.dumps(comp, indent=2), encoding="utf-8")
    return comp


def deployment_scorecard(final_1000: dict, pf_std: float, wf_ret: float) -> dict:
    checks = [
        {"metric": "Trades > 80", "actual": final_1000["trades"], "pass": bool(final_1000["trades"] > 80)},
        {"metric": "PF > 1.5", "actual": final_1000["pf"], "pass": bool(final_1000["pf"] > 1.5)},
        {"metric": "WR > 55%", "actual": final_1000["wr"], "pass": bool(final_1000["wr"] > 55.0)},
        {"metric": "MaxDD < 8%", "actual": final_1000["max_dd_pct"], "pass": bool(final_1000["max_dd_pct"] < 8.0)},
        {"metric": "$/Month > $250", "actual": final_1000["usd_per_month"], "pass": bool(final_1000["usd_per_month"] > 250.0)},
        {"metric": "PF StdDev < 0.50", "actual": pf_std, "pass": bool(pf_std < 0.50)},
        {"metric": "WF PF retention > 60%", "actual": wf_ret, "pass": bool(wf_ret > 60.0)},
    ]
    return {"checks": checks, "all_pass": all(c["pass"] for c in checks)}


def extract_prestep_fields(cfg: dict) -> dict:
    sf = cfg.get("session_filter", {})
    tp = cfg.get("exit_rules", {}).get("take_profit", {})
    tr = cfg.get("exit_rules", {}).get("trailing_stop", {})
    tm = cfg.get("trade_management", {})
    return {
        "session_entry_window_fields": {
            "session_filter.session_start_utc": sf.get("session_start_utc"),
            "session_filter.session_end_utc": sf.get("session_end_utc"),
            "session_filter.entry_start_utc": sf.get("entry_start_utc", "<uses session_start_utc>"),
            "session_filter.entry_end_utc": sf.get("entry_end_utc", "<uses session_end_utc>"),
            "session_filter.block_new_entries_minutes_before_end": sf.get("block_new_entries_minutes_before_end", 0),
        },
        "take_profit_fields": {
            "exit_rules.take_profit.mode": tp.get("mode"),
            "exit_rules.take_profit.partial_tp_atr_mult": tp.get("partial_tp_atr_mult"),
            "exit_rules.take_profit.partial_tp_min_pips": tp.get("partial_tp_min_pips"),
            "exit_rules.take_profit.partial_tp_max_pips": tp.get("partial_tp_max_pips"),
            "exit_rules.take_profit.partial_close_pct": tp.get("partial_close_pct"),
            "exit_rules.take_profit.breakeven_offset_pips": tp.get("breakeven_offset_pips"),
        },
        "trailing_stop_fields": {
            "exit_rules.trailing_stop.enabled": tr.get("enabled"),
            "exit_rules.trailing_stop.activate_after_profit_pips": tr.get("activate_after_profit_pips"),
            "exit_rules.trailing_stop.trail_distance_pips": tr.get("trail_distance_pips"),
            "exit_rules.trailing_stop.requires_tp1_hit": tr.get("requires_tp1_hit"),
        },
        "day_filter_fields": {
            "session_filter.allowed_trading_days": sf.get("allowed_trading_days"),
            "position_sizing.day_risk_multipliers": cfg.get("position_sizing", {}).get("day_risk_multipliers", {}),
        },
        "session_close_fields": {
            "session_filter.force_close_at_session_end": sf.get("force_close_at_session_end"),
            "exit_rules.time_exit.force_close_at_tokyo_end": cfg.get("exit_rules", {}).get("time_exit", {}).get("force_close_at_tokyo_end"),
            "late_session_management": cfg.get("late_session_management", {}),
        },
        "sizing_and_risk_fields": {
            "position_sizing.risk_per_trade_pct": cfg.get("position_sizing", {}).get("risk_per_trade_pct"),
            "exit_rules.stop_loss.hard_max_sl_pips": cfg.get("exit_rules", {}).get("stop_loss", {}).get("hard_max_sl_pips"),
            "position_sizing.max_units": cfg.get("position_sizing", {}).get("max_units"),
            "margin_model.enabled": cfg.get("margin_model", {}).get("enabled"),
            "margin_model.leverage": cfg.get("margin_model", {}).get("leverage"),
            "execution_model.spread_mode": cfg.get("execution_model", {}).get("spread_mode"),
        },
    }


def load_prior_comparison() -> dict:
    v14 = {
        "trades": 141,
        "wr": 72.3404255319149,
        "pf": 1.7505228585053982,
        "net_usd": 13820.670952136574,
        "max_dd_usd": 3926.512873084095,
        "max_dd_pct": None,
        "usd_per_month": 425.70987069572067,
        "wf_retention_pct": 97.3,
    }
    v15 = {
        "trades": 97,
        "wr": 76.28865979381443,
        "pf": 2.354396938017809,
        "net_usd": 18869.976348193803,
        "max_dd_usd": 3320.7933628269675,
        "max_dd_pct": None,
        "usd_per_month": 581.2406082918159,
        "wf_retention_pct": 45.71242292651361,
    }
    v16 = {
        "trades": 141,
        "wr": 71.63120567375887,
        "pf": 1.7038318612884638,
        "net_usd": 11834.263244551426,
        "max_dd_usd": 3228.754947663532,
        "max_dd_pct": 2.9675873808066147,
        "usd_per_month": 364.523740783965,
        "wf_retention_pct": 81.82668564641344,
    }
    return {"v14": v14, "v15": v15, "v16": v16}


def main() -> None:
    if not BASELINE_CFG.exists():
        raise RuntimeError(f"Missing baseline config: {BASELINE_CFG}")
    cfg = json.loads(BASELINE_CFG.read_text(encoding="utf-8"))

    prestep = extract_prestep_fields(cfg)
    changelog = []
    experiment_payloads = []
    current_cfg = deep_clone(cfg)

    for exp_num, variants in EXPERIMENTS:
        current_cfg, payload = run_experiment(exp_num, current_cfg, variants)
        experiment_payloads.append(payload)
        w = payload["winner"]
        changelog.append(
            {
                "experiment": exp_num,
                "winner_variant": w["variant"],
                "winner_label": w["label"],
                "winner_250k": w["dev_250k"],
                "winner_500k": w["val_500k"],
                "winner_1000k": w["val_1000k"],
            }
        )

    final_cfg_path = OUT / "tokyo_optimized_v17_config.json"
    final_cfg_path.write_text(json.dumps(current_cfg, indent=2), encoding="utf-8")

    scaling = run_scaling(current_cfg)
    wf = run_walkforward(current_cfg)
    scorecard = deployment_scorecard(scaling["1000k"], float(scaling["pf_stddev"]), float(wf["pf_retention_pct"]))
    prior = load_prior_comparison()

    comparison = {
        "v14": prior["v14"],
        "v15": prior["v15"],
        "v16": prior["v16"],
        "v17": {
            "trades": scaling["1000k"]["trades"],
            "wr": scaling["1000k"]["wr"],
            "pf": scaling["1000k"]["pf"],
            "net_usd": scaling["1000k"]["net_usd"],
            "max_dd_usd": scaling["1000k"]["max_dd_usd"],
            "max_dd_pct": scaling["1000k"]["max_dd_pct"],
            "usd_per_month": scaling["1000k"]["usd_per_month"],
            "wf_retention_pct": wf["pf_retention_pct"],
        },
    }

    summary = {
        "preamble_field_inspection": prestep,
        "objective": "Round5 edge improvement with winner selection on 250k only",
        "experiments_33_to_36": experiment_payloads,
        "cumulative_changelog": changelog,
        "final_config_path": str(final_cfg_path),
        "scaling_analysis": scaling,
        "walkforward_700k_300k": wf,
        "deployment_scorecard": scorecard,
        "comparison_v14_v15_v16_v17": comparison,
    }
    (OUT / "round5_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "ok",
                "final_config": str(final_cfg_path),
                "summary": str(OUT / "round5_summary.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
