#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd


ROOT = Path("/Users/codygnon/Documents/usdjpy_assistant")
ENGINE = ROOT / "scripts" / "backtest_tokyo_meanrev.py"
BASE_CFG = ROOT / "research_out" / "tokyo_optimized_v13_config.json"
DATA_250 = ROOT / "research_out" / "USDJPY_M1_OANDA_250k.csv"
DATA_500 = ROOT / "research_out" / "USDJPY_M1_OANDA_500k.csv"
DATA_1000 = ROOT / "research_out" / "USDJPY_M1_OANDA_1000k.csv"
OUT = ROOT / "research_out"
TMP = Path("/tmp/round2_v14_cfgs")


@dataclass
class Variant:
    code: str
    label: str
    apply_fn: Callable[[dict], None]


def deep_clone(x: dict) -> dict:
    return copy.deepcopy(x)


def dataset_months(csv_path: Path) -> float:
    df = pd.read_csv(csv_path, usecols=["time"])
    ts = pd.to_datetime(df["time"], utc=True, errors="coerce").dropna()
    if ts.empty:
        return 1.0
    months = (ts.iloc[-1] - ts.iloc[0]).total_seconds() / (86400.0 * 30.4375)
    return float(max(1e-6, months))


def run_one(tag: str, cfg: dict, csv_path: Path) -> dict:
    TMP.mkdir(parents=True, exist_ok=True)
    run_cfg = deep_clone(cfg)
    run_cfg["run_sequence"] = [
        {
            "label": "1000k",
            "input_csv": str(csv_path),
            "output_json": str(OUT / f"{tag}_report.json"),
            "output_trades_csv": str(OUT / f"{tag}_trades.csv"),
            "output_equity_csv": str(OUT / f"{tag}_equity.csv"),
        }
    ]
    cfg_path = TMP / f"{tag}.json"
    cfg_path.write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")
    p = subprocess.run(["python3", str(ENGINE), "--config", str(cfg_path)], cwd=str(ROOT), capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Run failed for {tag}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}")
    rep_path = OUT / f"{tag}_report.json"
    rep = json.loads(rep_path.read_text(encoding="utf-8"))
    s = rep["summary"]
    months = dataset_months(csv_path)
    usd_month = float(s["net_profit_usd"]) / months
    maxdd = float(s["max_drawdown_usd"])
    row = {
        "tag": tag,
        "report_path": str(rep_path),
        "trades_path": str(OUT / f"{tag}_trades.csv"),
        "equity_path": str(OUT / f"{tag}_equity.csv"),
        "months": months,
        "trades": int(s["total_trades"]),
        "wr": float(s["win_rate_pct"]),
        "pf": float(s["profit_factor"]),
        "net_usd": float(s["net_profit_usd"]),
        "max_dd_usd": maxdd,
        "max_dd_pct": float(s["max_drawdown_pct"]),
        "usd_per_month": usd_month,
        "net_over_maxdd": (float(s["net_profit_usd"]) / maxdd) if maxdd > 0 else 0.0,
        "avg_win_pips": float(s["average_win_pips"]),
        "avg_loss_pips": float(s["average_loss_pips"]),
        "expectancy_per_trade": (float(s["net_profit_usd"]) / int(s["total_trades"])) if int(s["total_trades"]) > 0 else 0.0,
    }
    return row


def run_variants(exp_num: int, base_cfg: dict, variants: list[Variant], csv_path: Path) -> list[dict]:
    jobs: list[tuple[Variant, dict]] = []
    for v in variants:
        c = deep_clone(base_cfg)
        v.apply_fn(c)
        jobs.append((v, c))
    rows = []
    with ThreadPoolExecutor(max_workers=min(4, len(jobs))) as ex:
        futs = {
            ex.submit(run_one, f"r2_exp{exp_num:02d}_{v.code}", c, csv_path): v
            for v, c in jobs
        }
        for fut in as_completed(futs):
            v = futs[fut]
            r = fut.result()
            r["variant"] = v.code
            r["label"] = v.label
            rows.append(r)
    order = {v.code: i for i, v in enumerate(variants)}
    rows.sort(key=lambda r: order[r["variant"]])
    return rows


def choose_by_round2_rule(rows: list[dict]) -> tuple[dict, str]:
    eligible = [r for r in rows if r["pf"] > 1.5]
    if eligible:
        # Soft preference for trades > 80 as tie-breaker.
        best = max(eligible, key=lambda r: (r["usd_per_month"], int(r["trades"] > 80), r["pf"]))
        return best, "max $/month under PF>1.5"
    pf_pool = [r for r in rows if r["trades"] > 60]
    if pf_pool:
        best = max(pf_pool, key=lambda r: (r["pf"], r["usd_per_month"]))
        return best, "no PF>1.5; max PF with trades>60"
    best = max(rows, key=lambda r: (r["pf"], r["usd_per_month"]))
    return best, "no PF>1.5 and no trades>60 fallback"


def choose_risk_rule(rows: list[dict]) -> tuple[dict, str]:
    ok = [r for r in rows if r["max_dd_pct"] < 8.0]
    if ok:
        best = max(ok, key=lambda r: (r["usd_per_month"], r["pf"]))
        return best, "max $/month with maxDD<8%"
    best = max(rows, key=lambda r: (r["usd_per_month"], r["pf"]))
    return best, "no variant under 8% DD fallback"


def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def kelly_fraction(wr_pct: float, avg_win_pips: float, avg_loss_pips: float) -> float:
    if avg_win_pips <= 0 or avg_loss_pips >= 0:
        return 0.0
    w = wr_pct / 100.0
    r = avg_win_pips / abs(avg_loss_pips)
    if r <= 0:
        return 0.0
    return float(w - ((1.0 - w) / r))


def apply_adx(enabled: bool, mx: float = 35.0) -> Callable[[dict], None]:
    def _f(c: dict) -> None:
        c.setdefault("adx_filter", {})["enabled"] = enabled
        c.setdefault("adx_filter", {})["max_adx_for_entry"] = mx
    return _f


def apply_risk(pct: float) -> Callable[[dict], None]:
    def _f(c: dict) -> None:
        c["position_sizing"]["risk_per_trade_pct"] = pct
    return _f


def apply_bb(period: int, stdv: float) -> Callable[[dict], None]:
    def _f(c: dict) -> None:
        c["indicators"]["bollinger_bands"]["period"] = period
        c["indicators"]["bollinger_bands"]["std_dev"] = stdv
    return _f


def apply_tol(pips: float) -> Callable[[dict], None]:
    def _f(c: dict) -> None:
        c["entry_rules"]["long"]["price_zone"]["tolerance_pips"] = pips
        c["entry_rules"]["short"]["price_zone"]["tolerance_pips"] = pips
    return _f


def apply_days(days: list[str]) -> Callable[[dict], None]:
    def _f(c: dict) -> None:
        c["session_filter"]["allowed_trading_days"] = days
    return _f


def apply_sl_trail(max_sl: float, min_sl: float, trail_enabled: bool, trail_act: float, trail_dist: float) -> Callable[[dict], None]:
    def _f(c: dict) -> None:
        c["exit_rules"]["stop_loss"]["hard_max_sl_pips"] = max_sl
        c["exit_rules"]["stop_loss"]["minimum_sl_pips"] = min_sl
        c["exit_rules"]["trailing_stop"]["enabled"] = trail_enabled
        c["exit_rules"]["trailing_stop"]["activate_after_profit_pips"] = trail_act
        c["exit_rules"]["trailing_stop"]["trail_distance_pips"] = trail_dist
    return _f


def apply_partial(atr_mult: float, tp_min: float, tp_max: float, pct_close: float) -> Callable[[dict], None]:
    def _f(c: dict) -> None:
        tp = c["exit_rules"]["take_profit"]
        tp["mode"] = "partial"
        tp["partial_tp_atr_mult"] = atr_mult
        tp["partial_tp_min_pips"] = tp_min
        tp["partial_tp_max_pips"] = tp_max
        tp["partial_close_pct"] = pct_close
        c["exit_rules"]["trailing_stop"]["enabled"] = True
        c["exit_rules"]["trailing_stop"]["requires_tp1_hit"] = True
    return _f


def apply_partial_fixed(tp_pips: float, pct_close: float) -> Callable[[dict], None]:
    def _f(c: dict) -> None:
        tp = c["exit_rules"]["take_profit"]
        tp["mode"] = "partial"
        tp["partial_tp_atr_mult"] = 0.0
        tp["partial_tp_min_pips"] = tp_pips
        tp["partial_tp_max_pips"] = tp_pips
        tp["partial_close_pct"] = pct_close
        c["exit_rules"]["trailing_stop"]["enabled"] = True
        c["exit_rules"]["trailing_stop"]["requires_tp1_hit"] = True
    return _f


def apply_single_pivot() -> Callable[[dict], None]:
    def _f(c: dict) -> None:
        tp = c["exit_rules"]["take_profit"]
        tp["mode"] = "single_pivot"
        tp["partial_close_pct"] = 1.0
        c["exit_rules"]["trailing_stop"]["enabled"] = False
    return _f


def apply_trail_only() -> Callable[[dict], None]:
    def _f(c: dict) -> None:
        tp = c["exit_rules"]["take_profit"]
        tp["mode"] = "trail_only"
        tp["partial_close_pct"] = 0.0
        c["exit_rules"]["trailing_stop"]["enabled"] = True
        c["exit_rules"]["trailing_stop"]["requires_tp1_hit"] = False
    return _f


def apply_confirmation(enabled: bool, window: int = 8) -> Callable[[dict], None]:
    def _f(c: dict) -> None:
        c["entry_confirmation"] = {"enabled": enabled, "type": "m1", "window_bars": window}
    return _f


def variants_for_experiment(exp_num: int) -> list[Variant]:
    if exp_num == 13:
        return [
            Variant("13A", "ADX<=20", apply_adx(True, 20)),
            Variant("13B", "ADX<=22", apply_adx(True, 22)),
            Variant("13C", "ADX<=25", apply_adx(True, 25)),
            Variant("13D", "ADX<=28", apply_adx(True, 28)),
            Variant("13E", "ADX<=30", apply_adx(True, 30)),
            Variant("13F", "ADX<=32", apply_adx(True, 32)),
            Variant("13G", "ADX<=35", apply_adx(True, 35)),
            Variant("13H", "ADX disabled", apply_adx(False, 35)),
        ]
    if exp_num == 14:
        return [
            Variant("14A", "risk 0.25%", apply_risk(0.25)),
            Variant("14B", "risk 0.50%", apply_risk(0.50)),
            Variant("14C", "risk 0.75%", apply_risk(0.75)),
            Variant("14D", "risk 1.00%", apply_risk(1.00)),
            Variant("14E", "risk 1.25%", apply_risk(1.25)),
            Variant("14F", "risk 1.50%", apply_risk(1.50)),
            Variant("14G", "risk 2.00%", apply_risk(2.00)),
        ]
    if exp_num == 15:
        return [
            Variant("15A", "BB(20,1.8)", apply_bb(20, 1.8)),
            Variant("15B", "BB(20,2.0)", apply_bb(20, 2.0)),
            Variant("15C", "BB(20,2.2)", apply_bb(20, 2.2)),
            Variant("15D", "BB(22,2.0)", apply_bb(22, 2.0)),
            Variant("15E", "BB(22,2.2)", apply_bb(22, 2.2)),
            Variant("15F", "BB(25,2.0)", apply_bb(25, 2.0)),
            Variant("15G", "BB(25,2.2)", apply_bb(25, 2.2)),
            Variant("15H", "BB(30,2.0)", apply_bb(30, 2.0)),
        ]
    if exp_num == 16:
        return [
            Variant("16A", "tol 5", apply_tol(5)),
            Variant("16B", "tol 8", apply_tol(8)),
            Variant("16C", "tol 10", apply_tol(10)),
            Variant("16D", "tol 12", apply_tol(12)),
            Variant("16E", "tol 15", apply_tol(15)),
            Variant("16F", "tol 20", apply_tol(20)),
            Variant("16G", "tol 25", apply_tol(25)),
        ]
    if exp_num == 17:
        return [
            Variant("17A", "Tue,Wed,Fri", apply_days(["Tuesday", "Wednesday", "Friday"])),
            Variant("17B", "Tue-Fri", apply_days(["Tuesday", "Wednesday", "Thursday", "Friday"])),
            Variant("17C", "Mon,Tue,Wed,Fri", apply_days(["Monday", "Tuesday", "Wednesday", "Friday"])),
            Variant("17D", "Mon-Fri", apply_days(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])),
            Variant("17E", "Tue,Wed,Thu", apply_days(["Tuesday", "Wednesday", "Thursday"])),
            Variant("17F", "Wed,Thu,Fri", apply_days(["Wednesday", "Thursday", "Friday"])),
        ]
    if exp_num == 18:
        return [
            Variant("18A", "SL25/8 TR6/4", apply_sl_trail(25, 8, True, 6, 4)),
            Variant("18B", "SL30/10 TR8/5", apply_sl_trail(30, 10, True, 8, 5)),
            Variant("18C", "SL35/12 TR8/5", apply_sl_trail(35, 12, True, 8, 5)),
            Variant("18D", "SL35/12 TR10/6", apply_sl_trail(35, 12, True, 10, 6)),
            Variant("18E", "SL35/12 TR12/8", apply_sl_trail(35, 12, True, 12, 8)),
            Variant("18F", "SL40/12 TR8/5", apply_sl_trail(40, 12, True, 8, 5)),
            Variant("18G", "SL30/10 TR6/4", apply_sl_trail(30, 10, True, 6, 4)),
            Variant("18H", "SL35/12 noTrail", apply_sl_trail(35, 12, False, 8, 5)),
        ]
    if exp_num == 19:
        return [
            Variant("19A", "40%@0.4ATR 5-10", apply_partial(0.4, 5, 10, 0.40)),
            Variant("19B", "50%@0.5ATR 6-12", apply_partial(0.5, 6, 12, 0.50)),
            Variant("19C", "60%@0.5ATR 6-12", apply_partial(0.5, 6, 12, 0.60)),
            Variant("19D", "70%@0.3ATR 4-8", apply_partial(0.3, 4, 8, 0.70)),
            Variant("19E", "50%@fixed8", apply_partial_fixed(8, 0.50)),
            Variant("19F", "50%@fixed5", apply_partial_fixed(5, 0.50)),
            Variant("19G", "single TP pivot", apply_single_pivot()),
            Variant("19H", "trail only", apply_trail_only()),
        ]
    if exp_num == 20:
        return [
            Variant("20A", "no confirm", apply_confirmation(False, 0)),
            Variant("20B", "m1<=3", apply_confirmation(True, 3)),
            Variant("20C", "m1<=5", apply_confirmation(True, 5)),
            Variant("20D", "m1<=8", apply_confirmation(True, 8)),
            Variant("20E", "m1<=12", apply_confirmation(True, 12)),
        ]
    raise ValueError(exp_num)


def day_breakdown(trades_csv: str) -> list[dict]:
    tdf = pd.read_csv(trades_csv)
    if tdf.empty:
        return []
    tdf["entry_ts"] = pd.to_datetime(tdf["entry_datetime"], utc=True, errors="coerce")
    out = (
        tdf.assign(day=tdf["entry_ts"].dt.day_name().str[:3])
        .groupby("day")
        .agg(
            trades=("usd", "count"),
            wr=("usd", lambda s: float((s > 0).mean() * 100.0)),
            gp=("usd", lambda s: float(s[s > 0].sum())),
            gl=("usd", lambda s: float(abs(s[s < 0].sum()))),
            net=("usd", "sum"),
            usd_per_trade=("usd", "mean"),
        )
        .reset_index()
    )
    out["pf"] = out.apply(lambda r: (r["gp"] / r["gl"]) if r["gl"] > 0 else (math.inf if r["gp"] > 0 else 0.0), axis=1)
    return out.to_dict(orient="records")


def run_final_sizes(cfg: dict) -> dict:
    outs = {}
    for label, csv in [("250k", DATA_250), ("500k", DATA_500), ("1000k", DATA_1000)]:
        tag = f"tokyo_optimized_v14_{label}"
        outs[label] = run_one(tag, cfg, csv)
    return outs


def run_walkforward(cfg: dict) -> dict:
    df = pd.read_csv(DATA_1000)
    train = df.iloc[:700000].copy()
    test = df.iloc[700000:].copy()
    train_csv = OUT / "tokyo_optimized_v14_train_700k.csv"
    test_csv = OUT / "tokyo_optimized_v14_test_300k.csv"
    train.to_csv(train_csv, index=False)
    test.to_csv(test_csv, index=False)

    train_res = run_one("tokyo_optimized_v14_walkforward_700k", cfg, train_csv)
    test_res = run_one("tokyo_optimized_v14_walkforward_300k", cfg, test_csv)
    ret = (test_res["pf"] / train_res["pf"] * 100.0) if train_res["pf"] > 0 else 0.0
    comp = {
        "training": train_res,
        "testing": test_res,
        "pf_retention_pct": ret,
    }
    save_json(OUT / "tokyo_optimized_v14_walkforward_comparison.json", comp)
    return comp


def main() -> int:
    if not BASE_CFG.exists():
        raise FileNotFoundError(BASE_CFG)
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)

    base_cfg = json.loads(BASE_CFG.read_text(encoding="utf-8"))
    base_cfg["starting_equity_usd"] = 100000

    baseline = run_one("round2_baseline_v13_1000k", base_cfg, DATA_1000)
    current_cfg = deep_clone(base_cfg)
    current_base = dict(baseline)

    summary_rows: list[dict] = []
    changelog: list[dict] = []

    for exp in range(13, 21):
        variants = variants_for_experiment(exp)
        print(f"\n=== Round2 Experiment {exp} ===")
        rows = run_variants(exp, current_cfg, variants, DATA_1000)

        if exp == 14:
            for r in rows:
                r["kelly_fraction"] = kelly_fraction(r["wr"], r["avg_win_pips"], r["avg_loss_pips"])
            chosen, rule = choose_risk_rule(rows)
        else:
            chosen, rule = choose_by_round2_rule(rows)

        chosen_variant = next(v for v in variants if v.code == chosen["variant"])
        new_cfg = deep_clone(current_cfg)
        chosen_variant.apply_fn(new_cfg)
        current_cfg = new_cfg
        current_base = dict(chosen)

        payload = {
            "experiment": exp,
            "objective": "maximize usd_per_month with PF>1.5",
            "rule_used": rule,
            "baseline_before_experiment": baseline if exp == 13 else None,
            "variant_results": rows,
            "winner": {
                "variant": chosen["variant"],
                "label": chosen["label"],
                "trades": chosen["trades"],
                "pf": chosen["pf"],
                "usd_per_month": chosen["usd_per_month"],
                "net_usd": chosen["net_usd"],
                "max_dd_usd": chosen["max_dd_usd"],
                "max_dd_pct": chosen["max_dd_pct"],
            },
        }
        if exp == 17:
            payload["pre_filter_day_breakdown"] = day_breakdown(rows[0]["trades_path"])
        save_json(OUT / f"experiment_{exp}_results.json", payload)

        summary_rows.append(
            {
                "experiment": exp,
                "winner": chosen["variant"],
                "winner_label": chosen["label"],
                "trades": chosen["trades"],
                "pf": chosen["pf"],
                "usd_per_month": chosen["usd_per_month"],
            }
        )
        changelog.append({"experiment": exp, "applied": chosen["label"], "rule": rule})
        print(
            f"EXP {exp} winner {chosen['variant']} {chosen['label']} | trades={chosen['trades']} "
            f"pf={chosen['pf']:.3f} $/mo={chosen['usd_per_month']:.2f}"
        )

    # Final config and runs.
    final_cfg_path = OUT / "tokyo_optimized_v14_config.json"
    final_cfg_path.write_text(json.dumps(current_cfg, indent=2), encoding="utf-8")

    scaling = run_final_sizes(current_cfg)
    wf = run_walkforward(current_cfg)

    pfs = [scaling[k]["pf"] for k in ["250k", "500k", "1000k"]]
    pf_std = float(pd.Series(pfs).std(ddof=0))

    scorecard = {
        "trades_1000k_pass": scaling["1000k"]["trades"] > 80,
        "pf_1000k_pass": scaling["1000k"]["pf"] > 1.5,
        "wr_1000k_pass": scaling["1000k"]["wr"] > 55.0,
        "maxdd_pass": scaling["1000k"]["max_dd_pct"] < 8.0,
        "usd_per_month_pass": scaling["1000k"]["usd_per_month"] > 250.0,
        "wf_retention_pass": wf["pf_retention_pct"] > 60.0,
        "pf_std_pass": pf_std < 0.5,
    }

    summary = {
        "baseline_round2_v13": baseline,
        "experiments_13_to_20": summary_rows,
        "changelog": changelog,
        "final_v14_scaling": scaling,
        "pf_stddev_scaling": pf_std,
        "walkforward": wf,
        "deployment_scorecard": scorecard,
    }
    save_json(OUT / "round2_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
