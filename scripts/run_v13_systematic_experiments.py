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
BASE_CFG_PATH = ROOT / "research_out" / "tokyo_mean_reversion_v7_config.json"
DATA_1000K = ROOT / "research_out" / "USDJPY_M1_OANDA_1000k.csv"
OUT_DIR = ROOT / "research_out"
TMP_CFG_DIR = Path("/tmp/systematic_v13_cfgs")


@dataclass
class Variant:
    name: str
    label: str
    apply_fn: Callable[[dict], None]
    notes: str = ""


def clone_cfg(cfg: dict) -> dict:
    return copy.deepcopy(cfg)


def normalize_base_config(cfg: dict) -> dict:
    out = clone_cfg(cfg)
    out["starting_equity_usd"] = 100000
    out.setdefault("entry_confirmation", {"enabled": True, "type": "m1", "window_bars": 5})
    out.setdefault("news_filter", {"enabled": False})
    out.setdefault("exit_rules", {}).setdefault("take_profit", {}).setdefault("mode", "partial")
    return out


def set_single_run(cfg: dict, tag: str) -> dict:
    c = clone_cfg(cfg)
    c["run_sequence"] = [
        {
            "label": "1000k",
            "input_csv": str(DATA_1000K),
            "output_json": str(OUT_DIR / f"{tag}_report.json"),
            "output_trades_csv": str(OUT_DIR / f"{tag}_trades.csv"),
            "output_equity_csv": str(OUT_DIR / f"{tag}_equity.csv"),
        }
    ]
    return c


def run_cfg_and_read(tag: str, cfg: dict) -> dict:
    TMP_CFG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg_path = TMP_CFG_DIR / f"{tag}.json"
    run_cfg = set_single_run(cfg, tag)
    cfg_path.write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")
    rep_path = OUT_DIR / f"{tag}_report.json"
    if not rep_path.exists():
        proc = subprocess.run(
            ["python3", str(ENGINE), "--config", str(cfg_path)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Backtest failed for {tag}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    rep = json.loads(rep_path.read_text(encoding="utf-8"))
    s = rep["summary"]
    return {
        "tag": tag,
        "report_path": str(rep_path),
        "trades_path": str(OUT_DIR / f"{tag}_trades.csv"),
        "equity_path": str(OUT_DIR / f"{tag}_equity.csv"),
        "trades": int(s["total_trades"]),
        "wr": float(s["win_rate_pct"]),
        "pf": float(s["profit_factor"]),
        "net_usd": float(s["net_profit_usd"]),
        "max_dd_usd": float(s["max_drawdown_usd"]),
        "max_dd_pct": float(s["max_drawdown_pct"]),
        "avg_win_pips": float(s["average_win_pips"]),
        "avg_loss_pips": float(s["average_loss_pips"]),
        "expectancy_per_trade": float(s["net_profit_usd"] / s["total_trades"]) if int(s["total_trades"]) > 0 else 0.0,
        "sharpe": float(s["sharpe_ratio"]),
        "net_over_maxdd": (float(s["net_profit_usd"]) / float(s["max_drawdown_usd"])) if float(s["max_drawdown_usd"]) > 0 else 0.0,
        "blocked_counts": rep.get("diagnostics", {}).get("counts", {}),
    }


def run_variants_parallel(exp_num: int, base_cfg: dict, variants: list[Variant]) -> list[dict]:
    jobs: list[tuple[str, dict]] = []
    for v in variants:
        c = clone_cfg(base_cfg)
        v.apply_fn(c)
        tag = f"exp{exp_num:02d}_{v.name}"
        jobs.append((tag, c))
    results = []
    with ThreadPoolExecutor(max_workers=min(4, len(jobs))) as ex:
        futs = {ex.submit(run_cfg_and_read, tag, c): tag for tag, c in jobs}
        for fut in as_completed(futs):
            results.append(fut.result())
    order = {f"exp{exp_num:02d}_{v.name}": i for i, v in enumerate(variants)}
    results.sort(key=lambda x: order[x["tag"]])
    return results


def print_exp_table(title: str, baseline_label: str, baseline_window: str, baseline: dict, rows: list[dict], window_key: str = "window") -> None:
    print("╔" + "═" * 67 + "╗")
    print(f"║  {title:<63}║")
    print("╠" + "═" * 67 + "╣")
    print("║  Variant   Setting          Trades  WR%    PF     Net$    MaxDD$ ║")
    print(
        f"║  {baseline_label:<8} {baseline_window:<15} {baseline['trades']:>6d} "
        f"{baseline['wr']:>6.1f} {baseline['pf']:>6.3f} {baseline['net_usd']:>8.0f} {baseline['max_dd_usd']:>8.0f} ║"
    )
    for r in rows:
        setting = str(r.get(window_key, ""))[:15]
        print(
            f"║  {r['name']:<8} {setting:<15} {r['trades']:>6d} "
            f"{r['wr']:>6.1f} {r['pf']:>6.3f} {r['net_usd']:>8.0f} {r['max_dd_usd']:>8.0f} ║"
        )
    print("╚" + "═" * 67 + "╝")


def choose_variant(baseline: dict, rows: list[dict], exp_name: str) -> dict:
    best = max(rows, key=lambda x: x["pf"])
    if best["pf"] > baseline["pf"] + 0.03:
        verdict = "KEEP"
        chosen = best
    elif abs(best["pf"] - baseline["pf"]) <= 0.03:
        verdict = "DISCARD"
        chosen = baseline
    else:
        verdict = "DISCARD"
        chosen = baseline
    return {
        "experiment": exp_name,
        "best_variant": best,
        "chosen": chosen,
        "verdict": verdict,
        "pf_delta_vs_baseline": float(best["pf"] - baseline["pf"]),
    }


def compute_blocked_trade_stats(reference_trades_csv: str, variant_trades_csv: str) -> dict:
    ref = pd.read_csv(reference_trades_csv)
    var = pd.read_csv(variant_trades_csv)
    if ref.empty:
        return {"blocked_trades": 0, "blocked_pf": 0.0, "blocked_net_usd": 0.0}

    def key_df(df: pd.DataFrame) -> pd.DataFrame:
        k = df.copy()
        k["key"] = (
            k["entry_datetime"].astype(str)
            + "|"
            + k["direction"].astype(str)
            + "|"
            + k["entry_price"].astype(float).round(5).astype(str)
        )
        return k

    refk = key_df(ref)
    vark = key_df(var) if not var.empty else var
    var_keys = set(vark["key"].tolist()) if not var.empty else set()
    blocked = refk[~refk["key"].isin(var_keys)].copy()
    if blocked.empty:
        return {"blocked_trades": 0, "blocked_pf": 0.0, "blocked_net_usd": 0.0}
    gp = float(blocked.loc[blocked["usd"] > 0, "usd"].sum())
    gl = float(abs(blocked.loc[blocked["usd"] < 0, "usd"].sum()))
    pf = (gp / gl) if gl > 0 else (math.inf if gp > 0 else 0.0)
    return {
        "blocked_trades": int(len(blocked)),
        "blocked_pf": float(pf),
        "blocked_net_usd": float(blocked["usd"].sum()),
    }


def apply_session_window(start: str, end: str) -> Callable[[dict], None]:
    def _f(c: dict) -> None:
        c["session_filter"]["session_start_utc"] = start
        c["session_filter"]["session_end_utc"] = end
    return _f


def apply_days(days: list[str]) -> Callable[[dict], None]:
    def _f(c: dict) -> None:
        c["session_filter"]["allowed_trading_days"] = days
    return _f


def apply_bb(period: int, stdv: float) -> Callable[[dict], None]:
    def _f(c: dict) -> None:
        c["indicators"]["bollinger_bands"]["period"] = period
        c["indicators"]["bollinger_bands"]["std_dev"] = stdv
    return _f


def apply_rsi(period: int, low: float, high: float) -> Callable[[dict], None]:
    def _f(c: dict) -> None:
        c["indicators"]["rsi"]["period"] = period
        c["entry_rules"]["long"]["rsi_soft_filter"]["entry_soft_threshold"] = low
        c["entry_rules"]["long"]["rsi_soft_filter"]["bonus_threshold"] = max(1.0, low - 5.0)
        c["entry_rules"]["short"]["rsi_soft_filter"]["entry_soft_threshold"] = high
        c["entry_rules"]["short"]["rsi_soft_filter"]["bonus_threshold"] = min(99.0, high + 5.0)
    return _f


def apply_tol(pips: float) -> Callable[[dict], None]:
    def _f(c: dict) -> None:
        c["entry_rules"]["long"]["price_zone"]["tolerance_pips"] = pips
        c["entry_rules"]["short"]["price_zone"]["tolerance_pips"] = pips
    return _f


def apply_sl(buffer_pips: float, max_sl: float, min_sl: float) -> Callable[[dict], None]:
    def _f(c: dict) -> None:
        c["exit_rules"]["stop_loss"]["buffer_pips"] = buffer_pips
        c["exit_rules"]["stop_loss"]["hard_max_sl_pips"] = max_sl
        c["exit_rules"]["stop_loss"]["minimum_sl_pips"] = min_sl
    return _f


def apply_partial(close_pct: float, atr_mult: float, tp_min: float, tp_max: float) -> Callable[[dict], None]:
    def _f(c: dict) -> None:
        tp = c["exit_rules"]["take_profit"]
        tp["mode"] = "partial"
        tp["partial_close_pct"] = close_pct
        tp["partial_tp_atr_mult"] = atr_mult
        tp["partial_tp_min_pips"] = tp_min
        tp["partial_tp_max_pips"] = tp_max
        c["exit_rules"]["trailing_stop"]["enabled"] = True
    return _f


def apply_single_tp(mode: str) -> Callable[[dict], None]:
    def _f(c: dict) -> None:
        tp = c["exit_rules"]["take_profit"]
        if mode == "pivot":
            tp["mode"] = "single_pivot"
        else:
            tp["mode"] = "single_atr"
            tp["single_tp_atr_mult"] = 1.0
            tp["single_tp_min_pips"] = 8.0
            tp["single_tp_max_pips"] = 40.0
        tp["partial_close_pct"] = 1.0
        c["exit_rules"]["trailing_stop"]["enabled"] = False
    return _f


def apply_trail(enabled: bool, activate: float = 12.0, dist: float = 8.0) -> Callable[[dict], None]:
    def _f(c: dict) -> None:
        c["exit_rules"]["trailing_stop"]["enabled"] = enabled
        c["exit_rules"]["trailing_stop"]["activate_after_profit_pips"] = activate
        c["exit_rules"]["trailing_stop"]["trail_distance_pips"] = dist
    return _f


def apply_risk(risk_pct: float) -> Callable[[dict], None]:
    def _f(c: dict) -> None:
        c["position_sizing"]["risk_per_trade_pct"] = risk_pct
    return _f


def apply_adx(enabled: bool, max_adx: float = 35.0) -> Callable[[dict], None]:
    def _f(c: dict) -> None:
        c["adx_filter"]["enabled"] = enabled
        c["adx_filter"]["max_adx_for_entry"] = max_adx
    return _f


def apply_confirmation(enabled: bool, typ: str = "m1", window: int = 5) -> Callable[[dict], None]:
    def _f(c: dict) -> None:
        c["entry_confirmation"] = {"enabled": enabled, "type": typ, "window_bars": window}
    return _f


def apply_news_disabled() -> Callable[[dict], None]:
    def _f(c: dict) -> None:
        c["news_filter"] = {"enabled": False}
    return _f


def apply_news_all(pre: int, post: int) -> Callable[[dict], None]:
    def _f(c: dict) -> None:
        c["news_filter"] = {
            "enabled": True,
            "calendar_path": str(ROOT / "research_out" / "v5_scheduled_events_utc.csv"),
            "mode": "all",
            "all_pre_block_minutes": pre,
            "all_post_block_minutes": post,
            "block_impacts": ["high", "medium", "low"],
        }
    return _f


def apply_news_tiered(block_impacts: list[str], high_pre: int, high_post: int, med_pre: int = 0, med_post: int = 0) -> Callable[[dict], None]:
    def _f(c: dict) -> None:
        c["news_filter"] = {
            "enabled": True,
            "calendar_path": str(ROOT / "research_out" / "v5_scheduled_events_utc.csv"),
            "mode": "tiered",
            "high_impact_pre_block_minutes": high_pre,
            "high_impact_post_block_minutes": high_post,
            "medium_impact_pre_block_minutes": med_pre,
            "medium_impact_post_block_minutes": med_post,
            "block_impacts": block_impacts,
        }
    return _f


def build_experiments() -> list[tuple[str, list[Variant]]]:
    return [
        ("Session Window", [
            Variant("1A", "15:00-22:00", apply_session_window("15:00", "22:00")),
            Variant("1B", "16:00-22:00", apply_session_window("16:00", "22:00")),
            Variant("1C", "16:30-22:00", apply_session_window("16:30", "22:00")),
            Variant("1D", "17:00-22:00", apply_session_window("17:00", "22:00")),
            Variant("1E", "16:30-21:00", apply_session_window("16:30", "21:00")),
            Variant("1F", "16:30-23:00", apply_session_window("16:30", "23:00")),
            Variant("1G", "17:00-21:00", apply_session_window("17:00", "21:00")),
            Variant("1H", "16:00-23:00", apply_session_window("16:00", "23:00")),
        ]),
        ("Day-of-Week", [
            Variant("2A", "Mon-Fri", apply_days(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])),
            Variant("2B", "Tue-Fri", apply_days(["Tuesday", "Wednesday", "Thursday", "Friday"])),
            Variant("2C", "Tue-Thu", apply_days(["Tuesday", "Wednesday", "Thursday"])),
            Variant("2D", "Wed-Fri", apply_days(["Wednesday", "Thursday", "Friday"])),
            Variant("2E", "Tue,Wed,Fri", apply_days(["Tuesday", "Wednesday", "Friday"])),
            Variant("2F", "Mon-Thu", apply_days(["Monday", "Tuesday", "Wednesday", "Thursday"])),
        ]),
        ("Bollinger", [
            Variant("3A", "BB(15,1.8)", apply_bb(15, 1.8)),
            Variant("3B", "BB(15,2.0)", apply_bb(15, 2.0)),
            Variant("3C", "BB(20,1.8)", apply_bb(20, 1.8)),
            Variant("3D", "BB(20,2.0)", apply_bb(20, 2.0)),
            Variant("3E", "BB(20,2.2)", apply_bb(20, 2.2)),
            Variant("3F", "BB(25,2.0)", apply_bb(25, 2.0)),
            Variant("3G", "BB(25,2.2)", apply_bb(25, 2.2)),
            Variant("3H", "BB(30,2.0)", apply_bb(30, 2.0)),
        ]),
        ("RSI", [
            Variant("4A", "RSI7 35/65", apply_rsi(7, 35, 65)),
            Variant("4B", "RSI7 40/60", apply_rsi(7, 40, 60)),
            Variant("4C", "RSI7 45/55", apply_rsi(7, 45, 55)),
            Variant("4D", "RSI14 30/70", apply_rsi(14, 30, 70)),
            Variant("4E", "RSI14 35/65", apply_rsi(14, 35, 65)),
            Variant("4F", "RSI14 40/60", apply_rsi(14, 40, 60)),
            Variant("4G", "RSI14 45/55", apply_rsi(14, 45, 55)),
            Variant("4H", "RSI21 40/60", apply_rsi(21, 40, 60)),
        ]),
        ("Pivot Tolerance", [
            Variant("5A", "tol 3", apply_tol(3)),
            Variant("5B", "tol 5", apply_tol(5)),
            Variant("5C", "tol 8", apply_tol(8)),
            Variant("5D", "tol 10", apply_tol(10)),
            Variant("5E", "tol 12", apply_tol(12)),
            Variant("5F", "tol 15", apply_tol(15)),
            Variant("5G", "tol 20", apply_tol(20)),
        ]),
        ("Stop Loss", [
            Variant("6A", "buf5 max20 min8", apply_sl(5, 20, 8)),
            Variant("6B", "buf8 max25 min8", apply_sl(8, 25, 8)),
            Variant("6C", "buf8 max28 min10", apply_sl(8, 28, 10)),
            Variant("6D", "buf10 max30 min10", apply_sl(10, 30, 10)),
            Variant("6E", "buf8 max35 min12", apply_sl(8, 35, 12)),
            Variant("6F", "buf12 max28 min10", apply_sl(12, 28, 10)),
        ]),
        ("Partial TP", [
            Variant("7A", "40%@0.4ATR", apply_partial(0.4, 0.4, 5, 10)),
            Variant("7B", "50%@0.5ATR", apply_partial(0.5, 0.5, 6, 12)),
            Variant("7C", "60%@0.5ATR", apply_partial(0.6, 0.5, 6, 12)),
            Variant("7D", "70%@0.4ATR", apply_partial(0.7, 0.4, 5, 10)),
            Variant("7E", "50%@0.6ATR", apply_partial(0.5, 0.6, 7, 15)),
            Variant("7F", "single pivot", apply_single_tp("pivot")),
            Variant("7G", "single 1.0ATR", apply_single_tp("atr")),
        ]),
        ("Trailing Stop", [
            Variant("8A", "act8 trail5", apply_trail(True, 8, 5)),
            Variant("8B", "act10 trail6", apply_trail(True, 10, 6)),
            Variant("8C", "act12 trail8", apply_trail(True, 12, 8)),
            Variant("8D", "act15 trail10", apply_trail(True, 15, 10)),
            Variant("8E", "act10 trail8", apply_trail(True, 10, 8)),
            Variant("8F", "act8 trail8", apply_trail(True, 8, 8)),
            Variant("8G", "no trailing", apply_trail(False, 12, 8)),
            Variant("8H", "act12 trail6", apply_trail(True, 12, 6)),
        ]),
        ("Risk Per Trade", [
            Variant("9A", "0.25%", apply_risk(0.25)),
            Variant("9B", "0.50%", apply_risk(0.50)),
            Variant("9C", "0.75%", apply_risk(0.75)),
            Variant("9D", "1.00%", apply_risk(1.00)),
            Variant("9E", "1.25%", apply_risk(1.25)),
            Variant("9F", "1.50%", apply_risk(1.50)),
            Variant("9G", "2.00%", apply_risk(2.00)),
        ]),
        ("ADX Filter", [
            Variant("10A", "disabled", apply_adx(False, 35)),
            Variant("10B", "max 20", apply_adx(True, 20)),
            Variant("10C", "max 25", apply_adx(True, 25)),
            Variant("10D", "max 30", apply_adx(True, 30)),
            Variant("10E", "max 35", apply_adx(True, 35)),
            Variant("10F", "max 40", apply_adx(True, 40)),
            Variant("10G", "max 45", apply_adx(True, 45)),
        ]),
        ("Confirmation Candle", [
            Variant("11A", "none", apply_confirmation(False, "m1", 0)),
            Variant("11B", "m1<=3", apply_confirmation(True, "m1", 3)),
            Variant("11C", "m1<=5", apply_confirmation(True, "m1", 5)),
            Variant("11D", "m1<=8", apply_confirmation(True, "m1", 8)),
            Variant("11E", "m1<=12", apply_confirmation(True, "m1", 12)),
            Variant("11F", "m5<=1", apply_confirmation(True, "m5", 1)),
            Variant("11G", "m5<=2", apply_confirmation(True, "m5", 2)),
        ]),
        ("News Filter", [
            Variant("12A", "off", apply_news_disabled()),
            Variant("12B", "all 15/15", apply_news_all(15, 15)),
            Variant("12C", "all 30/30", apply_news_all(30, 30)),
            Variant("12D", "high 30/60", apply_news_tiered(["high"], 30, 60)),
            Variant("12E", "high 60/60", apply_news_tiered(["high"], 60, 60)),
            Variant("12F", "high 60/90", apply_news_tiered(["high"], 60, 90)),
            Variant("12G", "high60/120 med15/15", apply_news_tiered(["high", "medium"], 60, 120, 15, 15)),
        ]),
    ]


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def run_walkforward(final_cfg: dict) -> dict:
    src = pd.read_csv(DATA_1000K)
    train = src.iloc[:700000].copy()
    test = src.iloc[700000:].copy()
    train_path = OUT_DIR / "tokyo_optimized_v13_train_700k.csv"
    test_path = OUT_DIR / "tokyo_optimized_v13_test_300k.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    c_train = clone_cfg(final_cfg)
    c_train["run_sequence"] = [
        {
            "label": "700k",
            "input_csv": str(train_path),
            "output_json": str(OUT_DIR / "tokyo_optimized_v13_walkforward_700k_report.json"),
            "output_trades_csv": str(OUT_DIR / "tokyo_optimized_v13_walkforward_700k_trades.csv"),
            "output_equity_csv": str(OUT_DIR / "tokyo_optimized_v13_walkforward_700k_equity.csv"),
        }
    ]
    cfg_train = TMP_CFG_DIR / "tokyo_optimized_v13_walk_train.json"
    cfg_train.write_text(json.dumps(c_train, indent=2), encoding="utf-8")
    p1 = subprocess.run(["python3", str(ENGINE), "--config", str(cfg_train)], cwd=str(ROOT), capture_output=True, text=True)
    if p1.returncode != 0:
        raise RuntimeError(f"WF train failed: {p1.stderr}\n{p1.stdout}")

    c_test = clone_cfg(final_cfg)
    c_test["run_sequence"] = [
        {
            "label": "300k",
            "input_csv": str(test_path),
            "output_json": str(OUT_DIR / "tokyo_optimized_v13_walkforward_300k_report.json"),
            "output_trades_csv": str(OUT_DIR / "tokyo_optimized_v13_walkforward_300k_trades.csv"),
            "output_equity_csv": str(OUT_DIR / "tokyo_optimized_v13_walkforward_300k_equity.csv"),
        }
    ]
    cfg_test = TMP_CFG_DIR / "tokyo_optimized_v13_walk_test.json"
    cfg_test.write_text(json.dumps(c_test, indent=2), encoding="utf-8")
    p2 = subprocess.run(["python3", str(ENGINE), "--config", str(cfg_test)], cwd=str(ROOT), capture_output=True, text=True)
    if p2.returncode != 0:
        raise RuntimeError(f"WF test failed: {p2.stderr}\n{p2.stdout}")

    train_rep = json.loads((OUT_DIR / "tokyo_optimized_v13_walkforward_700k_report.json").read_text(encoding="utf-8"))
    test_rep = json.loads((OUT_DIR / "tokyo_optimized_v13_walkforward_300k_report.json").read_text(encoding="utf-8"))
    tr = train_rep["summary"]
    te = test_rep["summary"]
    out = {
        "walk_forward_validation": {
            "training": {
                "candles": 700000,
                "trades": int(tr["total_trades"]),
                "wr": float(tr["win_rate_pct"]),
                "pf": float(tr["profit_factor"]),
                "net": float(tr["net_profit_usd"]),
                "maxdd": float(tr["max_drawdown_usd"]),
            },
            "testing": {
                "candles": 300000,
                "trades": int(te["total_trades"]),
                "wr": float(te["win_rate_pct"]),
                "pf": float(te["profit_factor"]),
                "net": float(te["net_profit_usd"]),
                "maxdd": float(te["max_drawdown_usd"]),
            },
        }
    }
    tr_pf = float(tr["profit_factor"])
    te_pf = float(te["profit_factor"])
    out["walk_forward_validation"]["degradation"] = {
        "pf_change": te_pf - tr_pf,
        "pf_change_pct": ((te_pf - tr_pf) / tr_pf * 100.0) if tr_pf != 0 else 0.0,
        "wr_change": float(te["win_rate_pct"] - tr["win_rate_pct"]),
        "pf_retention_pct": (te_pf / tr_pf * 100.0) if tr_pf > 0 else 0.0,
    }
    save_json(OUT_DIR / "tokyo_optimized_v13_walkforward_comparison.json", out)
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_CFG_DIR.mkdir(parents=True, exist_ok=True)
    base_cfg = normalize_base_config(json.loads(BASE_CFG_PATH.read_text(encoding="utf-8")))

    baseline = run_cfg_and_read("baseline_v7_100k_1000k", base_cfg)
    save_json(OUT_DIR / "baseline_v7_100k_1000k_metrics.json", {"baseline": baseline})

    if not (195 <= baseline["trades"] <= 215 and baseline["wr"] > 62.0 and baseline["pf"] > 1.10):
        raise RuntimeError(
            f"Baseline guardrail failed: trades={baseline['trades']} wr={baseline['wr']:.2f}% pf={baseline['pf']:.3f}"
        )

    experiments = build_experiments()
    current_cfg = clone_cfg(base_cfg)
    current_baseline = baseline
    changelog = []
    summary_rows = []

    # Baseline day-of-week table (from trade log).
    tdf_base = pd.read_csv(current_baseline["trades_path"])
    if not tdf_base.empty:
        tdf_base["entry_ts"] = pd.to_datetime(tdf_base["entry_datetime"], utc=True, errors="coerce")
        dow = (
            tdf_base.assign(day=tdf_base["entry_ts"].dt.day_name().str[:3])
            .groupby("day")
            .agg(
                trades=("usd", "count"),
                wr=("usd", lambda s: float((s > 0).mean() * 100.0)),
                gp=("usd", lambda s: float(s[s > 0].sum())),
                gl=("usd", lambda s: float(abs(s[s < 0].sum()))),
                net=("usd", "sum"),
            )
        )
        dow["pf"] = dow.apply(lambda r: (r["gp"] / r["gl"]) if r["gl"] > 0 else (math.inf if r["gp"] > 0 else 0.0), axis=1)
        save_json(OUT_DIR / "baseline_day_of_week_table.json", dow.reset_index().to_dict(orient="records"))

    for idx, (exp_name, variants) in enumerate(experiments, start=1):
        print(f"\n=== Running Experiment {idx}: {exp_name} ===")
        exp_baseline = dict(current_baseline)
        rows = run_variants_parallel(idx, current_cfg, variants)
        variant_rows = []
        for v, r in zip(variants, rows):
            rr = dict(r)
            rr["name"] = v.name
            rr["label"] = v.label
            variant_rows.append(rr)

        if idx == 12:
            ref = next((x for x in variant_rows if x["name"] == "12A"), None)
            if ref is not None:
                for rr in variant_rows:
                    stats = compute_blocked_trade_stats(ref["trades_path"], rr["trades_path"])
                    rr.update(stats)

        decision = choose_variant(exp_baseline, variant_rows, exp_name)
        chosen = decision["chosen"]
        best = decision["best_variant"]
        verdict = decision["verdict"]

        chosen_cfg = clone_cfg(current_cfg)
        if verdict == "KEEP":
            win_variant = next(v for v in variants if f"exp{idx:02d}_{v.name}" == chosen["tag"])
            win_variant.apply_fn(chosen_cfg)
            current_cfg = chosen_cfg
            current_baseline = chosen
            changelog.append(
                {
                    "experiment": idx,
                    "feature": exp_name,
                    "verdict": "KEEP",
                    "variant": win_variant.name,
                    "label": win_variant.label,
                    "pf_delta": float(best["pf"] - baseline["pf"]),
                }
            )
            chosen_name = win_variant.name
            chosen_label = win_variant.label
        else:
            changelog.append(
                {
                    "experiment": idx,
                    "feature": exp_name,
                    "verdict": "DISCARD",
                    "variant": "baseline",
                    "label": "baseline",
                    "pf_delta": float(best["pf"] - current_baseline["pf"]),
                }
            )
            chosen_name = "baseline"
            chosen_label = "baseline"

        save_json(
            OUT_DIR / f"experiment_{idx:02d}_results.json",
            {
                "experiment": idx,
                "feature": exp_name,
                "baseline": exp_baseline,
                "variant_results": variant_rows,
                "decision": {
                    "verdict": verdict,
                    "best_variant_name": best.get("name", ""),
                    "best_variant_tag": best.get("tag", ""),
                    "best_variant_pf": best.get("pf", 0.0),
                    "chosen": chosen_name,
                    "chosen_label": chosen_label,
                    "pf_delta_vs_baseline": decision["pf_delta_vs_baseline"],
                },
            },
        )
        summary_rows.append(
            {
                "experiment": idx,
                "feature": exp_name,
                "verdict": verdict,
                "best_param": chosen_label,
                "pf_impact": decision["pf_delta_vs_baseline"],
            }
        )
        print(f"Experiment {idx} verdict: {verdict} | best={best.get('name')} pf={best.get('pf'):.3f} | baseline_pf={exp_baseline.get('pf',0):.3f}")

    # Final full run with optimized config.
    final_cfg = clone_cfg(current_cfg)
    save_json(ROOT / "research_out" / "tokyo_optimized_v13_config.json", final_cfg)
    final_full = run_cfg_and_read("tokyo_optimized_v13_1000k", final_cfg)
    wf = run_walkforward(final_cfg)

    summary = {
        "baseline": baseline,
        "final_full_1000k": final_full,
        "experiment_summary_rows": summary_rows,
        "changelog": changelog,
        "walkforward": wf,
    }
    save_json(OUT_DIR / "experiment_summary.json", summary)
    # Also save at requested root filenames.
    save_json(ROOT / "research_out" / "experiment_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
