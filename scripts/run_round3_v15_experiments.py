#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


ROOT = Path("/Users/codygnon/Documents/usdjpy_assistant")
ENGINE = ROOT / "scripts" / "backtest_tokyo_meanrev.py"
OUT = ROOT / "research_out"
TMP = Path("/tmp/round3_v15_cfgs")

BASELINE_CFG = OUT / "tokyo_v14_realism_maxopen3_risk1p5_maxu1p5m_config.json"
DATA_250 = OUT / "USDJPY_M1_OANDA_250k.csv"
DATA_500 = OUT / "USDJPY_M1_OANDA_500k.csv"
DATA_1000 = OUT / "USDJPY_M1_OANDA_1000k.csv"
DATA_700 = OUT / "USDJPY_M1_OANDA_700k_split.csv"
DATA_300 = OUT / "USDJPY_M1_OANDA_300k_split.csv"
MONTHS_1000K = 32.465


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
    rep_path = OUT / f"{out_prefix}_report.json"
    rep = json.loads(rep_path.read_text(encoding="utf-8"))
    return {
        "config_used": run_cfg,
        "report_path": str(rep_path),
        "trades_path": str(OUT / f"{out_prefix}_trades.csv"),
        "equity_path": str(OUT / f"{out_prefix}_equity.csv"),
        "report": rep,
    }


def summarize_variant(code: str, label: str, run: dict, months: float = MONTHS_1000K) -> dict:
    rep = run["report"]
    s = rep["summary"]
    counts = rep.get("diagnostics", {}).get("counts", {})
    maxdd = float(s["max_drawdown_usd"])
    net = float(s["net_profit_usd"])
    trades = int(s["total_trades"])
    row = {
        "variant": code,
        "label": label,
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
        "blocked_margin_cap": int(counts.get("blocked_margin_cap", 0)),
        "exit_reason_breakdown": rep.get("breakdown", {}).get("exit_distribution", []),
        "report_path": run["report_path"],
        "trades_path": run["trades_path"],
        "equity_path": run["equity_path"],
    }
    return row


def choose_winner(rows: list[dict]) -> tuple[dict, str]:
    eligible = [r for r in rows if float(r["pf"]) > 1.5]
    if not eligible:
        raise RuntimeError("No eligible variant with PF > 1.5 (hard constraint violated).")
    winner = max(
        eligible,
        key=lambda r: (
            float(r["net_usd"]),
            -float(r["max_dd_usd"]),
            int(r["trades"]),
            float(r["wr"]),
            float(r["pf"]),
        ),
    )
    reason = "max Net USD (PF>1.5), tie-break: lower MaxDD, higher trades, higher WR, higher PF"
    return winner, reason


def clean_day_risk_and_adx_overrides(c: dict) -> None:
    c.setdefault("position_sizing", {}).pop("day_risk_multipliers", None)
    c.setdefault("adx_filter", {}).pop("day_max_by_day", None)


def variant_set_exp21() -> list[Variant]:
    def v_disabled(c: dict) -> None:
        c["early_exit"] = {"enabled": False}

    def v_enabled(c: dict, t: int, lp: float, mp: float) -> None:
        c["early_exit"] = {
            "enabled": True,
            "time_threshold_minutes": t,
            "loss_threshold_pips": lp,
            "max_profit_seen_pips": mp,
            "action": "close_at_market",
        }

    return [
        Variant("21A", "disabled", v_disabled),
        Variant("21B", "20m / 3p / 1.5p", lambda c: v_enabled(c, 20, 3.0, 1.5)),
        Variant("21C", "30m / 5p / 2.0p", lambda c: v_enabled(c, 30, 5.0, 2.0)),
        Variant("21D", "20m / 5p / 2.0p", lambda c: v_enabled(c, 20, 5.0, 2.0)),
        Variant("21E", "15m / 3p / 1.0p", lambda c: v_enabled(c, 15, 3.0, 1.0)),
        Variant("21F", "30m / 3p / 1.0p", lambda c: v_enabled(c, 30, 3.0, 1.0)),
    ]


def variant_set_exp22() -> list[Variant]:
    def set_late(c: dict, enabled: bool, mins: int = 45, no_tp1_below: float = -2.0, tighten: float = 3.0) -> None:
        c["late_session_management"] = {
            "enabled": bool(enabled),
            "minutes_before_end": int(mins),
            "close_if_no_tp1_and_pips_below": float(no_tp1_below),
            "tp1_hit_tighten_trail_pips": float(tighten),
        }

    return [
        Variant("22A", "disabled", lambda c: set_late(c, False)),
        Variant("22B", "45m / cut<-2 / tighten=3", lambda c: set_late(c, True, 45, -2.0, 3.0)),
        Variant("22C", "30m / cut<-1 / tighten=3", lambda c: set_late(c, True, 30, -1.0, 3.0)),
        Variant("22D", "60m / cut<-3 / tighten=4", lambda c: set_late(c, True, 60, -3.0, 4.0)),
        Variant("22E", "45m / cut<0 / tighten=2", lambda c: set_late(c, True, 45, 0.0, 2.0)),
    ]


def variant_set_exp23() -> list[Variant]:
    def v23a(c: dict) -> None:
        clean_day_risk_and_adx_overrides(c)
        c.setdefault("session_filter", {})["allowed_trading_days"] = ["Tuesday", "Wednesday", "Friday"]

    def v23b(c: dict) -> None:
        clean_day_risk_and_adx_overrides(c)
        c.setdefault("session_filter", {})["allowed_trading_days"] = ["Tuesday", "Friday"]

    def v23c(c: dict) -> None:
        clean_day_risk_and_adx_overrides(c)
        c.setdefault("position_sizing", {})["day_risk_multipliers"] = {"Wednesday": 0.5}
        c.setdefault("session_filter", {})["allowed_trading_days"] = ["Tuesday", "Wednesday", "Friday"]

    def v23d(c: dict) -> None:
        clean_day_risk_and_adx_overrides(c)
        c.setdefault("position_sizing", {})["day_risk_multipliers"] = {"Wednesday": 0.25}
        c.setdefault("session_filter", {})["allowed_trading_days"] = ["Tuesday", "Wednesday", "Friday"]

    def v23e(c: dict) -> None:
        clean_day_risk_and_adx_overrides(c)
        c.setdefault("adx_filter", {})["day_max_by_day"] = {"Wednesday": 25.0}
        c.setdefault("session_filter", {})["allowed_trading_days"] = ["Tuesday", "Wednesday", "Friday"]

    return [
        Variant("23A", "Wed full risk", v23a),
        Variant("23B", "Drop Wednesday (Tue,Fri)", v23b),
        Variant("23C", "Wed half risk", v23c),
        Variant("23D", "Wed quarter risk", v23d),
        Variant("23E", "Wed ADX<=25", v23e),
    ]


def variant_set_exp24() -> list[Variant]:
    def set_exit(c: dict, partial_pct: float, trail_act: float, trail_dist: float) -> None:
        c.setdefault("exit_rules", {}).setdefault("take_profit", {})["partial_close_pct"] = float(partial_pct)
        c.setdefault("exit_rules", {}).setdefault("trailing_stop", {})["activate_after_profit_pips"] = float(trail_act)
        c.setdefault("exit_rules", {}).setdefault("trailing_stop", {})["trail_distance_pips"] = float(trail_dist)

    return [
        Variant("24A", "50% / trail 8/5", lambda c: set_exit(c, 0.50, 8.0, 5.0)),
        Variant("24B", "65% / trail 8/4", lambda c: set_exit(c, 0.65, 8.0, 4.0)),
        Variant("24C", "70% / trail 6/3", lambda c: set_exit(c, 0.70, 6.0, 3.0)),
        Variant("24D", "60% / trail 6/4", lambda c: set_exit(c, 0.60, 6.0, 4.0)),
        Variant("24E", "50% / trail 6/3", lambda c: set_exit(c, 0.50, 6.0, 3.0)),
        Variant("24F", "75% / trail 10/5", lambda c: set_exit(c, 0.75, 10.0, 5.0)),
    ]


def variant_set_exp25() -> list[Variant]:
    def set_sl(c: dict, max_sl: float, min_sl: float | None = None) -> None:
        c.setdefault("exit_rules", {}).setdefault("stop_loss", {})["hard_max_sl_pips"] = float(max_sl)
        if min_sl is not None:
            c.setdefault("exit_rules", {}).setdefault("stop_loss", {})["minimum_sl_pips"] = float(min_sl)

    return [
        Variant("25A", "max35", lambda c: set_sl(c, 35.0)),
        Variant("25B", "max30", lambda c: set_sl(c, 30.0)),
        Variant("25C", "max25", lambda c: set_sl(c, 25.0)),
        Variant("25D", "max20", lambda c: set_sl(c, 20.0)),
        Variant("25E", "max28 min10", lambda c: set_sl(c, 28.0, 10.0)),
    ]


def variant_set_exp26() -> list[Variant]:
    def set_rej(c: dict, enabled: bool, lookback: int = 2, sl_buf: float = 2.0, ratio: float = 1.5) -> None:
        c["rejection_bonus"] = {
            "enabled": bool(enabled),
            "sl_improvement": True,
            "lookback_m5_bars": int(lookback),
            "sl_buffer_pips": float(sl_buf),
            "wick_to_body_ratio": float(ratio),
            "close_position_pct": 0.4,
            "min_candle_range_pips": 3.0,
            "doji_wick_pct": 0.60,
        }

    return [
        Variant("26A", "disabled", lambda c: set_rej(c, False)),
        Variant("26B", "lb2 buf2.0 ratio1.5", lambda c: set_rej(c, True, 2, 2.0, 1.5)),
        Variant("26C", "lb3 buf1.5 ratio1.2", lambda c: set_rej(c, True, 3, 1.5, 1.2)),
        Variant("26D", "lb2 buf3.0 ratio2.0", lambda c: set_rej(c, True, 2, 3.0, 2.0)),
    ]


EXPERIMENTS: list[tuple[int, list[Variant]]] = [
    (21, variant_set_exp21()),
    (22, variant_set_exp22()),
    (23, variant_set_exp23()),
    (24, variant_set_exp24()),
    (25, variant_set_exp25()),
    (26, variant_set_exp26()),
]


def run_experiment(exp_num: int, base_cfg: dict, variants: list[Variant]) -> tuple[dict, dict]:
    rows: list[dict] = []
    cfg_by_code: dict[str, dict] = {}
    for v in variants:
        cfg = deep_clone(base_cfg)
        v.apply_fn(cfg)
        out_prefix = f"round3_exp{exp_num}_{v.code}"
        run = run_backtest(cfg, tag=out_prefix, csv_path=DATA_1000, out_prefix=out_prefix)
        row = summarize_variant(v.code, v.label, run, months=MONTHS_1000K)
        rows.append(row)
        cfg_by_code[v.code] = cfg
    winner, rule = choose_winner(rows)
    winner_cfg = cfg_by_code[winner["variant"]]
    payload = {
        "experiment": exp_num,
        "selection_rule": rule,
        "baseline_path_used": str(BASELINE_CFG),
        "variants": rows,
        "winner": winner,
    }
    (OUT / f"experiment_{exp_num}_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return winner_cfg, payload


def run_scaling(cfg: dict) -> dict:
    out = {}
    for label, csv_path in [("250k", DATA_250), ("500k", DATA_500), ("1000k", DATA_1000)]:
        prefix = f"tokyo_optimized_v15_{label}"
        run = run_backtest(cfg, tag=prefix, csv_path=csv_path, out_prefix=prefix)
        out[label] = summarize_variant(label, label, run, months=MONTHS_1000K if label == "1000k" else MONTHS_1000K * (int(label.replace('k',''))/1000.0))
    pfs = [out[k]["pf"] for k in ["250k", "500k", "1000k"]]
    out["pf_stddev"] = float(statistics.pstdev(pfs))
    return out


def run_walkforward(cfg: dict) -> dict:
    tr = run_backtest(cfg, tag="tokyo_optimized_v15_walkforward_700k", csv_path=DATA_700, out_prefix="tokyo_optimized_v15_walkforward_700k")
    te = run_backtest(cfg, tag="tokyo_optimized_v15_walkforward_300k", csv_path=DATA_300, out_prefix="tokyo_optimized_v15_walkforward_300k")
    r_tr = summarize_variant("700k", "700k", tr, months=MONTHS_1000K * 0.7)
    r_te = summarize_variant("300k", "300k", te, months=MONTHS_1000K * 0.3)
    ret = (r_te["pf"] / r_tr["pf"] * 100.0) if r_tr["pf"] > 0 else 0.0
    comp = {
        "train": r_tr,
        "test": r_te,
        "pf_retention_pct": ret,
    }
    (OUT / "tokyo_optimized_v15_walkforward_comparison.json").write_text(json.dumps(comp, indent=2), encoding="utf-8")
    return comp


def deployment_scorecard(final_1000k: dict, wf: dict, pf_std: float) -> dict:
    checks = [
        {"metric": "Trades (1000k)", "target": "> 80", "actual": final_1000k["trades"], "pass": final_1000k["trades"] > 80},
        {"metric": "PF (1000k)", "target": "> 1.50", "actual": final_1000k["pf"], "pass": final_1000k["pf"] > 1.5},
        {"metric": "WR (1000k)", "target": "> 55%", "actual": final_1000k["wr"], "pass": final_1000k["wr"] > 55.0},
        {"metric": "MaxDD", "target": "< 8% equity", "actual": final_1000k["max_dd_pct"], "pass": final_1000k["max_dd_pct"] < 8.0},
        {"metric": "$/Month", "target": "> $250", "actual": final_1000k["usd_per_month"], "pass": final_1000k["usd_per_month"] > 250.0},
        {"metric": "WF PF retention", "target": "> 60%", "actual": wf["pf_retention_pct"], "pass": wf["pf_retention_pct"] > 60.0},
        {"metric": "PF StdDev (scaling)", "target": "< 0.50", "actual": pf_std, "pass": pf_std < 0.50},
    ]
    return {"checks": checks, "all_pass": all(c["pass"] for c in checks)}


def main() -> None:
    base_cfg = json.loads(BASELINE_CFG.read_text(encoding="utf-8"))

    baseline_run = run_backtest(
        base_cfg,
        tag="round3_baseline_v14",
        csv_path=DATA_1000,
        out_prefix="round3_baseline_v14_realism_1p5_maxu1p5m_1000k",
    )
    baseline_row = summarize_variant("baseline_v14", "baseline_v14", baseline_run, months=MONTHS_1000K)

    changelog = []
    experiment_payloads = []
    cur_cfg = deep_clone(base_cfg)
    for exp_num, variants in EXPERIMENTS:
        cur_cfg, payload = run_experiment(exp_num, cur_cfg, variants)
        experiment_payloads.append(payload)
        changelog.append(
            {
                "experiment": exp_num,
                "winner_variant": payload["winner"]["variant"],
                "winner_label": payload["winner"]["label"],
                "winner_net_usd": payload["winner"]["net_usd"],
                "winner_pf": payload["winner"]["pf"],
                "winner_max_dd_usd": payload["winner"]["max_dd_usd"],
            }
        )

    # Save final config.
    final_cfg_path = OUT / "tokyo_optimized_v15_config.json"
    final_cfg_path.write_text(json.dumps(cur_cfg, indent=2), encoding="utf-8")

    scaling = run_scaling(cur_cfg)
    walkforward = run_walkforward(cur_cfg)
    scorecard = deployment_scorecard(scaling["1000k"], walkforward, float(scaling["pf_stddev"]))

    side_by_side = {
        "baseline_v14_realism_1p5_maxu1p5m": baseline_row,
        "v15_final_1000k": scaling["1000k"],
        "delta": {
            "trades": scaling["1000k"]["trades"] - baseline_row["trades"],
            "wr": scaling["1000k"]["wr"] - baseline_row["wr"],
            "pf": scaling["1000k"]["pf"] - baseline_row["pf"],
            "net_usd": scaling["1000k"]["net_usd"] - baseline_row["net_usd"],
            "max_dd_usd": scaling["1000k"]["max_dd_usd"] - baseline_row["max_dd_usd"],
            "usd_per_month": scaling["1000k"]["usd_per_month"] - baseline_row["usd_per_month"],
        },
    }

    summary = {
        "objective_priority": [
            "max_net_usd",
            "min_max_dd",
            "max_trade_count",
            "max_wr",
            "max_pf",
        ],
        "hard_constraint": "PF > 1.5",
        "baseline": baseline_row,
        "experiments": experiment_payloads,
        "cumulative_changelog": changelog,
        "final_config_path": str(final_cfg_path),
        "scaling": scaling,
        "walkforward": walkforward,
        "deployment_scorecard": scorecard,
        "side_by_side_v14_vs_v15": side_by_side,
    }
    (OUT / "round3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "final_config": str(final_cfg_path), "summary": str(OUT / "round3_summary.json")}, indent=2))


if __name__ == "__main__":
    main()
