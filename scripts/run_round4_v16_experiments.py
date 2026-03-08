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
TMP = Path("/tmp/round4_v16_cfgs")

BASELINE_CFG = OUT / "tokyo_optimized_v15_config.json"
V14_CFG = OUT / "tokyo_v14_realism_maxopen3_risk1p5_maxu1p5m_config.json"
ROUND3_SUMMARY = OUT / "round3_summary.json"

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


def summarize_run(run: dict, months: float | None = None) -> dict:
    rep = run["report"]
    s = rep["summary"]
    counts = rep.get("diagnostics", {}).get("counts", {})
    maxdd = float(s["max_drawdown_usd"])
    net = float(s["net_profit_usd"])
    trades = int(s["total_trades"])
    out = {
        "trades": trades,
        "wr": float(s["win_rate_pct"]),
        "pf": float(s["profit_factor"]),
        "net_usd": net,
        "max_dd_usd": maxdd,
        "max_dd_pct": float(s["max_drawdown_pct"]),
        "net_over_maxdd": (net / maxdd) if maxdd > 0 else 0.0,
        "expectancy_per_trade": (net / trades) if trades > 0 else 0.0,
        "sharpe": float(s.get("sharpe_ratio", 0.0)),
        "calmar": float(s.get("calmar_ratio", 0.0)),
        "blocked_margin_cap": int(counts.get("blocked_margin_cap", 0)),
        "report_path": run["report_path"],
        "trades_path": run["trades_path"],
        "equity_path": run["equity_path"],
    }
    if months and months > 0:
        out["usd_per_month"] = net / months
    return out


def run_triplet(cfg: dict, prefix: str) -> dict:
    r1000 = run_backtest(cfg, tag=f"{prefix}_1000k", csv_path=DATA_1000, out_prefix=f"{prefix}_1000k")
    r700 = run_backtest(cfg, tag=f"{prefix}_700k", csv_path=DATA_700, out_prefix=f"{prefix}_700k")
    r300 = run_backtest(cfg, tag=f"{prefix}_300k", csv_path=DATA_300, out_prefix=f"{prefix}_300k")
    m1000 = summarize_run(r1000, months=MONTHS_1000K)
    m700 = summarize_run(r700, months=MONTHS_1000K * 0.7)
    m300 = summarize_run(r300, months=MONTHS_1000K * 0.3)
    retention = (m300["pf"] / m700["pf"] * 100.0) if m700["pf"] > 0 else 0.0
    m1000["exit_reason_breakdown"] = r1000["report"].get("breakdown", {}).get("exit_distribution", [])
    return {
        "full_1000k": m1000,
        "wf_train_700k": m700,
        "wf_test_300k": m300,
        "wf_pf_retention_pct": retention,
    }


def choose_winner(rows: list[dict]) -> tuple[dict, str]:
    eligible = []
    for r in rows:
        pf_1000 = float(r["full_1000k"]["pf"])
        pf_test = float(r["wf_test_300k"]["pf"])
        if pf_1000 > 1.5 and pf_test >= 1.3:
            eligible.append(r)
    if not eligible:
        raise RuntimeError("No eligible variant passed hard gates (1000k PF>1.5 and test PF>=1.3).")

    target = [r for r in eligible if float(r["wf_pf_retention_pct"]) >= 60.0]
    pool = target if target else eligible

    winner = max(
        pool,
        key=lambda r: (
            float(r["wf_pf_retention_pct"]),
            float(r["full_1000k"]["net_usd"]),
            -float(r["full_1000k"]["max_dd_usd"]),
            int(r["full_1000k"]["trades"]),
        ),
    )
    if target:
        reason = "maximize WF retention (>=60 pool), then 1000k Net USD, lower MaxDD, higher trades"
    else:
        reason = "no variant reached 60% retention; maximize WF retention among hard-gate passers, then Net USD, MaxDD, trades"
    return winner, reason


def reset_wednesday_tuning(c: dict) -> None:
    c.setdefault("position_sizing", {}).pop("day_risk_multipliers", None)
    adx = c.setdefault("adx_filter", {})
    adx.pop("day_max_by_day", None)
    adx.pop("day_overrides", None)


def variant_set_exp27() -> list[Variant]:
    def set_max_sl(c: dict, v: float) -> None:
        c.setdefault("exit_rules", {}).setdefault("stop_loss", {})["hard_max_sl_pips"] = float(v)

    return [
        Variant("27A", "max_sl=20", lambda c: set_max_sl(c, 20.0)),
        Variant("27B", "max_sl=22", lambda c: set_max_sl(c, 22.0)),
        Variant("27C", "max_sl=25", lambda c: set_max_sl(c, 25.0)),
        Variant("27D", "max_sl=28", lambda c: set_max_sl(c, 28.0)),
        Variant("27E", "max_sl=30", lambda c: set_max_sl(c, 30.0)),
        Variant("27F", "max_sl=23", lambda c: set_max_sl(c, 23.0)),
    ]


def variant_set_exp28() -> list[Variant]:
    def base_days(c: dict, days: list[str]) -> None:
        c.setdefault("session_filter", {})["allowed_trading_days"] = days

    def v28a(c: dict) -> None:
        reset_wednesday_tuning(c)
        base_days(c, ["Tuesday", "Friday"])

    def v28b(c: dict) -> None:
        reset_wednesday_tuning(c)
        base_days(c, ["Tuesday", "Wednesday", "Friday"])
        c.setdefault("position_sizing", {})["day_risk_multipliers"] = {"Wednesday": 0.5}

    def v28c(c: dict) -> None:
        reset_wednesday_tuning(c)
        base_days(c, ["Tuesday", "Wednesday", "Friday"])
        c.setdefault("position_sizing", {})["day_risk_multipliers"] = {"Wednesday": 0.25}

    def v28d(c: dict) -> None:
        reset_wednesday_tuning(c)
        base_days(c, ["Tuesday", "Wednesday", "Friday"])

    def v28e(c: dict) -> None:
        reset_wednesday_tuning(c)
        base_days(c, ["Tuesday", "Wednesday", "Friday"])
        c.setdefault("position_sizing", {})["day_risk_multipliers"] = {"Wednesday": 0.5}
        adx = c.setdefault("adx_filter", {})
        adx["day_max_by_day"] = {"Wednesday": 25.0}
        # Alias requested by prompt; engine supports both.
        adx["day_overrides"] = {"Wednesday": 25.0}

    return [
        Variant("28A", "No Wednesday", v28a),
        Variant("28B", "Wednesday 0.5x risk", v28b),
        Variant("28C", "Wednesday 0.25x risk", v28c),
        Variant("28D", "Wednesday full risk", v28d),
        Variant("28E", "Wednesday 0.5x + ADX<=25", v28e),
    ]


def variant_set_exp29() -> list[Variant]:
    def set_risk(c: dict, pct: float) -> None:
        c.setdefault("position_sizing", {})["risk_per_trade_pct"] = float(pct)

    return [
        Variant("29A", "risk=1.5", lambda c: set_risk(c, 1.5)),
        Variant("29B", "risk=1.25", lambda c: set_risk(c, 1.25)),
        Variant("29C", "risk=1.0", lambda c: set_risk(c, 1.0)),
        Variant("29D", "risk=1.75", lambda c: set_risk(c, 1.75)),
        Variant("29E", "risk=2.0", lambda c: set_risk(c, 2.0)),
    ]


def variant_set_exp30() -> list[Variant]:
    def set_cap(c: dict, units: int) -> None:
        c.setdefault("position_sizing", {})["max_units"] = int(units)

    return [
        Variant("30A", "max_units=1500000", lambda c: set_cap(c, 1_500_000)),
        Variant("30B", "max_units=1000000", lambda c: set_cap(c, 1_000_000)),
        Variant("30C", "max_units=750000", lambda c: set_cap(c, 750_000)),
        Variant("30D", "max_units=500000", lambda c: set_cap(c, 500_000)),
        Variant("30E", "max_units=2000000", lambda c: set_cap(c, 2_000_000)),
    ]


def variant_set_exp31() -> list[Variant]:
    def set_session_ctrl(c: dict, stop_pct: float, consec: int) -> None:
        tm = c.setdefault("trade_management", {})
        tm["session_loss_stop_pct"] = float(stop_pct)
        tm["stop_after_consecutive_losses"] = int(consec)

    return [
        Variant("31A", "stop_pct=1.5, consec=3", lambda c: set_session_ctrl(c, 1.5, 3)),
        Variant("31B", "stop_pct=1.0, consec=3", lambda c: set_session_ctrl(c, 1.0, 3)),
        Variant("31C", "stop_pct=1.5, consec=2", lambda c: set_session_ctrl(c, 1.5, 2)),
        Variant("31D", "stop_pct=1.0, consec=2", lambda c: set_session_ctrl(c, 1.0, 2)),
        Variant("31E", "stop_pct=2.0, consec=4", lambda c: set_session_ctrl(c, 2.0, 4)),
    ]


EXPERIMENTS: list[tuple[int, list[Variant]]] = [
    (27, variant_set_exp27()),
    (28, variant_set_exp28()),
    (29, variant_set_exp29()),
    (30, variant_set_exp30()),
    (31, variant_set_exp31()),
]


def run_experiment(exp_num: int, base_cfg: dict, variants: list[Variant]) -> tuple[dict, dict]:
    rows: list[dict] = []
    cfg_by_code: dict[str, dict] = {}
    for v in variants:
        cfg = deep_clone(base_cfg)
        v.apply_fn(cfg)
        tag = f"round4_exp{exp_num}_{v.code}"
        metrics = run_triplet(cfg, prefix=tag)
        row = {
            "variant": v.code,
            "label": v.label,
            **metrics,
        }
        pf_1000 = float(metrics["full_1000k"]["pf"])
        pf_test = float(metrics["wf_test_300k"]["pf"])
        row["hard_gate_pass"] = bool(pf_1000 > 1.5 and pf_test >= 1.3)
        rows.append(row)
        cfg_by_code[v.code] = cfg

    winner, rule = choose_winner(rows)
    winner_cfg = cfg_by_code[winner["variant"]]
    payload = {
        "experiment": exp_num,
        "selection_rule": rule,
        "hard_gates": {
            "full_1000k_pf_gt": 1.5,
            "wf_test_pf_gte": 1.3,
        },
        "objective_priority": [
            "maximize_wf_pf_retention",
            "maximize_1000k_net_usd",
            "minimize_1000k_maxdd_usd",
            "maximize_1000k_trades",
        ],
        "baseline_path_used": str(BASELINE_CFG),
        "variants": rows,
        "winner": winner,
    }
    (OUT / f"experiment_{exp_num}_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return winner_cfg, payload


def ensure_row_split(source_csv: Path, train_rows: int, train_csv: Path, test_csv: Path) -> dict:
    train_csv.parent.mkdir(parents=True, exist_ok=True)
    test_csv.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with source_csv.open("r", encoding="utf-8") as src:
        header = src.readline()
        if not header:
            raise RuntimeError(f"Empty CSV source: {source_csv}")
        with train_csv.open("w", encoding="utf-8") as tr, test_csv.open("w", encoding="utf-8") as te:
            tr.write(header)
            te.write(header)
            for line in src:
                if total < train_rows:
                    tr.write(line)
                else:
                    te.write(line)
                total += 1
    if total < train_rows:
        raise RuntimeError(f"Requested train_rows={train_rows}, but source has only {total} rows.")
    return {
        "source_rows": total,
        "train_rows": train_rows,
        "test_rows": total - train_rows,
        "train_csv": str(train_csv),
        "test_csv": str(test_csv),
    }


def run_experiment_32(cfg: dict) -> dict:
    split_defs = [
        ("32A", "700k/300k", DATA_700, DATA_300, None),
        (
            "32B",
            "600k/400k",
            OUT / "USDJPY_M1_OANDA_600k_split.csv",
            OUT / "USDJPY_M1_OANDA_400k_split.csv",
            600_000,
        ),
        (
            "32C",
            "500k/500k",
            OUT / "USDJPY_M1_OANDA_500k_train_from_1000k.csv",
            OUT / "USDJPY_M1_OANDA_500k_test_from_1000k.csv",
            500_000,
        ),
        (
            "32D",
            "800k/200k",
            OUT / "USDJPY_M1_OANDA_800k_split.csv",
            OUT / "USDJPY_M1_OANDA_200k_split.csv",
            800_000,
        ),
    ]
    rows = []
    split_meta = {}

    for code, label, train_csv, test_csv, train_rows in split_defs:
        if train_rows is not None:
            split_meta[code] = ensure_row_split(DATA_1000, train_rows, train_csv, test_csv)

        tr = run_backtest(cfg, tag=f"round4_exp32_{code}_train", csv_path=train_csv, out_prefix=f"round4_exp32_{code}_train")
        te = run_backtest(cfg, tag=f"round4_exp32_{code}_test", csv_path=test_csv, out_prefix=f"round4_exp32_{code}_test")
        tr_m = summarize_run(tr)
        te_m = summarize_run(te)
        retention = (te_m["pf"] / tr_m["pf"] * 100.0) if tr_m["pf"] > 0 else 0.0
        rows.append(
            {
                "split": code,
                "label": label,
                "train": tr_m,
                "test": te_m,
                "pf_retention_pct": retention,
                "retention_ge_60": retention >= 60.0,
                "split_files": {
                    "train_csv": str(train_csv),
                    "test_csv": str(test_csv),
                },
                "split_meta": split_meta.get(code, {}),
            }
        )

    rets = [float(r["pf_retention_pct"]) for r in rows]
    avg_ret = float(sum(rets) / len(rets)) if rets else 0.0
    ge60 = sum(1 for r in rows if r["retention_ge_60"])
    validation_pass = bool(avg_ret >= 55.0 and ge60 >= 2)
    payload = {
        "experiment": 32,
        "type": "multi_split_walkforward_validation",
        "splits": rows,
        "average_pf_retention_pct": avg_ret,
        "splits_ge_60_pct": ge60,
        "validation_rule": {
            "avg_retention_gte": 55.0,
            "min_splits_with_retention_ge_60": 2,
        },
        "validation_pass": validation_pass,
    }
    (OUT / "experiment_32_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_scaling(cfg: dict) -> dict:
    out = {}
    months = {
        "250k": MONTHS_1000K * 0.25,
        "500k": MONTHS_1000K * 0.50,
        "1000k": MONTHS_1000K,
    }
    for label, csv_path in [("250k", DATA_250), ("500k", DATA_500), ("1000k", DATA_1000)]:
        prefix = f"tokyo_optimized_v16_{label}"
        run = run_backtest(cfg, tag=prefix, csv_path=csv_path, out_prefix=prefix)
        row = summarize_run(run, months=months[label])
        row["exit_reason_breakdown"] = run["report"].get("breakdown", {}).get("exit_distribution", [])
        out[label] = row
    pfs = [out[k]["pf"] for k in ["250k", "500k", "1000k"]]
    out["pf_stddev"] = float(statistics.pstdev(pfs))
    return out


def run_walkforward_standard(cfg: dict) -> dict:
    tr = run_backtest(cfg, tag="tokyo_optimized_v16_walkforward_700k", csv_path=DATA_700, out_prefix="tokyo_optimized_v16_walkforward_700k")
    te = run_backtest(cfg, tag="tokyo_optimized_v16_walkforward_300k", csv_path=DATA_300, out_prefix="tokyo_optimized_v16_walkforward_300k")
    tr_m = summarize_run(tr)
    te_m = summarize_run(te)
    ret = (te_m["pf"] / tr_m["pf"] * 100.0) if tr_m["pf"] > 0 else 0.0
    comp = {"train": tr_m, "test": te_m, "pf_retention_pct": ret}
    (OUT / "tokyo_optimized_v16_walkforward_comparison.json").write_text(json.dumps(comp, indent=2), encoding="utf-8")
    return comp


def deployment_scorecard(final_1000k: dict, wf: dict, pf_std: float) -> dict:
    checks = [
        {"metric": "Trades (1000k)", "target": "> 80", "actual": final_1000k["trades"], "pass": final_1000k["trades"] > 80},
        {"metric": "PF (1000k)", "target": "> 1.50", "actual": final_1000k["pf"], "pass": final_1000k["pf"] > 1.5},
        {"metric": "WR (1000k)", "target": "> 55%", "actual": final_1000k["wr"], "pass": final_1000k["wr"] > 55.0},
        {"metric": "MaxDD", "target": "< 8% equity", "actual": final_1000k["max_dd_pct"], "pass": final_1000k["max_dd_pct"] < 8.0},
        {"metric": "$/Month", "target": "> $250", "actual": final_1000k["usd_per_month"], "pass": final_1000k["usd_per_month"] > 250.0},
        {"metric": "PF StdDev (scaling)", "target": "< 0.50", "actual": pf_std, "pass": pf_std < 0.50},
        {"metric": "WF PF retention", "target": "> 60%", "actual": wf["pf_retention_pct"], "pass": wf["pf_retention_pct"] > 60.0},
    ]
    return {"checks": checks, "all_pass": all(c["pass"] for c in checks)}


def load_v14_v15_refs(v15_baseline_row: dict) -> tuple[dict, dict]:
    if ROUND3_SUMMARY.exists():
        r3 = json.loads(ROUND3_SUMMARY.read_text(encoding="utf-8"))
        v14 = r3.get("side_by_side_v14_vs_v15", {}).get("baseline_v14_realism_1p5_maxu1p5m")
        v15 = r3.get("side_by_side_v14_vs_v15", {}).get("v15_final_1000k")
        if v14 and v15:
            return v14, v15
    # fallback if historical summary is missing
    return {}, v15_baseline_row


def main() -> None:
    if not BASELINE_CFG.exists():
        raise RuntimeError(f"Missing baseline config: {BASELINE_CFG}")

    base_cfg = json.loads(BASELINE_CFG.read_text(encoding="utf-8"))

    # Baseline V15 metrics on full 1000k.
    base_run_1000 = run_backtest(
        base_cfg,
        tag="round4_baseline_v15_1000k",
        csv_path=DATA_1000,
        out_prefix="round4_baseline_v15_1000k",
    )
    baseline_v15_row = summarize_run(base_run_1000, months=MONTHS_1000K)
    baseline_v15_row["exit_reason_breakdown"] = base_run_1000["report"].get("breakdown", {}).get("exit_distribution", [])

    # Sequential winner stacking for 27-31.
    cur_cfg = deep_clone(base_cfg)
    changelog = []
    experiment_payloads = []
    for exp_num, variants in EXPERIMENTS:
        cur_cfg, payload = run_experiment(exp_num, cur_cfg, variants)
        experiment_payloads.append(payload)
        changelog.append(
            {
                "experiment": exp_num,
                "winner_variant": payload["winner"]["variant"],
                "winner_label": payload["winner"]["label"],
                "winner_1000k_net_usd": payload["winner"]["full_1000k"]["net_usd"],
                "winner_1000k_pf": payload["winner"]["full_1000k"]["pf"],
                "winner_wf_retention_pct": payload["winner"]["wf_pf_retention_pct"],
            }
        )

    # Exp32 multi-split validation (no winner selection).
    exp32_payload = run_experiment_32(cur_cfg)

    final_cfg_path = OUT / "tokyo_optimized_v16_config.json"
    final_cfg_path.write_text(json.dumps(cur_cfg, indent=2), encoding="utf-8")

    scaling = run_scaling(cur_cfg)
    wf_std = run_walkforward_standard(cur_cfg)
    scorecard = deployment_scorecard(scaling["1000k"], wf_std, float(scaling["pf_stddev"]))
    v14_ref, v15_ref = load_v14_v15_refs(baseline_v15_row)

    side_by_side = {
        "v14_realism_1p5_maxu1p5m": v14_ref,
        "v15_baseline": v15_ref,
        "v16_final": scaling["1000k"],
        "delta_v15_to_v16": {
            "trades": scaling["1000k"]["trades"] - baseline_v15_row["trades"],
            "wr": scaling["1000k"]["wr"] - baseline_v15_row["wr"],
            "pf": scaling["1000k"]["pf"] - baseline_v15_row["pf"],
            "net_usd": scaling["1000k"]["net_usd"] - baseline_v15_row["net_usd"],
            "max_dd_usd": scaling["1000k"]["max_dd_usd"] - baseline_v15_row["max_dd_usd"],
            "usd_per_month": scaling["1000k"]["usd_per_month"] - baseline_v15_row["usd_per_month"],
        },
    }

    summary = {
        "objective_priority": [
            "hard_gate_1000k_pf_gt_1p5",
            "hard_gate_test_pf_gte_1p3",
            "maximize_wf_pf_retention",
            "maximize_1000k_net_usd",
            "minimize_1000k_maxdd_usd",
            "maximize_1000k_trade_count",
        ],
        "baseline_v15": baseline_v15_row,
        "experiments_27_to_31": experiment_payloads,
        "experiment_32_multi_split": exp32_payload,
        "cumulative_changelog": changelog,
        "final_config_path": str(final_cfg_path),
        "scaling": scaling,
        "walkforward_standard_700_300": wf_std,
        "deployment_scorecard": scorecard,
        "side_by_side_v14_v15_v16": side_by_side,
    }
    (OUT / "round4_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "ok",
                "final_config": str(final_cfg_path),
                "summary": str(OUT / "round4_summary.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
