#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/Users/codygnon/Documents/usdjpy_assistant")
SCRIPT_V14 = ROOT / "scripts" / "backtest_tokyo_meanrev.py"
SCRIPT_V44 = ROOT / "scripts" / "backtest_session_momentum.py"


@dataclass
class TradeRow:
    strategy: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_session: str
    side: str
    pips: float
    usd: float
    exit_reason: str
    raw: dict[str, Any]
    standalone_entry_equity: float = 0.0
    size_scale: float = 1.0


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def _profit_factor(trades: list[TradeRow]) -> float:
    wins = sum(t.usd for t in trades if t.usd > 0.0)
    losses = sum(-t.usd for t in trades if t.usd < 0.0)
    if losses <= 0:
        return 0.0 if wins <= 0 else 999.0
    return wins / losses


def _max_drawdown_from_equity_curve(points: list[dict[str, Any]], start_equity: float) -> tuple[float, float]:
    if not points:
        return 0.0, 0.0
    peak = start_equity
    max_dd = 0.0
    for p in points:
        eq = _safe_float(p.get("equity_after"), start_equity)
        peak = max(peak, eq)
        dd = peak - eq
        if dd > max_dd:
            max_dd = dd
    max_dd_pct = (max_dd / peak * 100.0) if peak > 0 else 0.0
    return max_dd, max_dd_pct


def _sharpe_calmar(points: list[dict[str, Any]], start_equity: float) -> tuple[float, float]:
    if len(points) < 3:
        return 0.0, 0.0
    eq_series = [start_equity] + [_safe_float(p.get("equity_after"), start_equity) for p in points]
    rets = []
    for i in range(1, len(eq_series)):
        prev = eq_series[i - 1]
        cur = eq_series[i]
        if prev > 0:
            rets.append((cur - prev) / prev)
    if len(rets) < 2:
        return 0.0, 0.0
    r = np.array(rets, dtype=float)
    std = float(np.std(r, ddof=1))
    mean = float(np.mean(r))
    sharpe = (mean / std * math.sqrt(len(r))) if std > 1e-12 else 0.0
    total_return = (eq_series[-1] - start_equity) / start_equity if start_equity > 0 else 0.0
    max_dd, _ = _max_drawdown_from_equity_curve(points, start_equity)
    calmar = (total_return / (max_dd / start_equity)) if (start_equity > 0 and max_dd > 0) else 0.0
    return float(sharpe), float(calmar)


def _stats(trades: list[TradeRow], start_equity: float, equity_curve: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(trades)
    wins = [t for t in trades if t.usd > 0.0]
    losses = [t for t in trades if t.usd <= 0.0]
    win_rate = (len(wins) / n * 100.0) if n else 0.0
    net_usd = float(sum(t.usd for t in trades))
    net_pips = float(sum(t.pips for t in trades))
    max_dd_usd, max_dd_pct = _max_drawdown_from_equity_curve(equity_curve, start_equity)
    sharpe, calmar = _sharpe_calmar(equity_curve, start_equity)
    avg_win_pips = float(np.mean([t.pips for t in wins])) if wins else 0.0
    avg_loss_pips = float(np.mean([t.pips for t in losses])) if losses else 0.0
    avg_win_usd = float(np.mean([t.usd for t in wins])) if wins else 0.0
    avg_loss_usd = float(np.mean([t.usd for t in losses])) if losses else 0.0

    max_cw = max_cl = 0
    cur_w = cur_l = 0
    for t in sorted(trades, key=lambda x: x.exit_time):
        if t.usd > 0:
            cur_w += 1
            cur_l = 0
        else:
            cur_l += 1
            cur_w = 0
        max_cw = max(max_cw, cur_w)
        max_cl = max(max_cl, cur_l)

    return {
        "total_trades": int(n),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "win_rate_pct": float(win_rate),
        "profit_factor": float(_profit_factor(trades)),
        "net_usd": float(net_usd),
        "net_pips": float(net_pips),
        "max_drawdown_usd": float(max_dd_usd),
        "max_drawdown_pct": float(max_dd_pct),
        "sharpe_ratio": float(sharpe),
        "calmar_ratio": float(calmar),
        "avg_win_pips": float(avg_win_pips),
        "avg_loss_pips": float(avg_loss_pips),
        "avg_win_usd": float(avg_win_usd),
        "avg_loss_usd": float(avg_loss_usd),
        "max_consecutive_wins": int(max_cw),
        "max_consecutive_losses": int(max_cl),
    }


def _convert_v44_embedded_to_flat(embedded: dict[str, Any], input_csv: str, out_json: str) -> dict[str, Any]:
    cfg = dict(embedded)
    v5 = dict(cfg.pop("v5", {}))
    sessions_utc = dict(cfg.pop("sessions_utc", {}))
    trend = dict(cfg.pop("trend", {}))
    caps = dict(cfg.pop("caps", {}))
    risk = dict(cfg.pop("risk", {}))

    flat: dict[str, Any] = {
        "version": "v5",
        "mode": cfg.get("mode", "session"),
        "inputs": [input_csv],
        "out": out_json,
        "spread_mode": cfg.get("spread_mode", "realistic"),
        "spread_pips": cfg.get("spread_avg_target_pips", cfg.get("spread_pips", 2.0)),
        "spread_min_pips": cfg.get("spread_min_pips", 1.0),
        "spread_max_pips": cfg.get("spread_max_pips", 3.0),
        "max_entry_spread_pips": cfg.get("max_entry_spread_pips", 3.0),
        "london_start": _safe_float(sessions_utc.get("london", [8.5, 11.0])[0], 8.5),
        "london_end": _safe_float(sessions_utc.get("london", [8.5, 11.0])[1], 11.0),
        "ny_start": _safe_float(sessions_utc.get("ny_overlap", [13.0, 16.0])[0], 13.0),
        "ny_end": _safe_float(sessions_utc.get("ny_overlap", [13.0, 16.0])[1], 16.0),
        "h1_ema_fast": _safe_int(trend.get("h1_ema_fast", 20), 20),
        "h1_ema_slow": _safe_int(trend.get("h1_ema_slow", 50), 50),
        "h1_allow_slope_direction": bool(trend.get("h1_allow_slope_direction", True)),
        "h1_slope_bars": _safe_int(trend.get("h1_slope_bars", 6), 6),
        "max_open_positions": _safe_int(caps.get("max_open_positions", 1), 1),
        "max_entries_per_day": _safe_int(caps.get("max_entries_per_day", 3), 3),
        "loss_after_first_full_sl_lot_mult": _safe_float(caps.get("loss_after_first_full_sl_lot_mult", 1.0), 1.0),
    }
    for k, v in risk.items():
        flat[k] = v
    for k, v in v5.items():
        flat[f"v5_{k}"] = v
    return flat


def _run_v14(v14_config_path: Path, input_csv: str, workdir: Path) -> tuple[dict[str, Any], Path]:
    cfg = json.loads(v14_config_path.read_text(encoding="utf-8"))
    out_json = workdir / "v14_report.json"
    out_trades = workdir / "v14_trades.csv"
    out_equity = workdir / "v14_equity.csv"
    cfg["run_sequence"] = [
        {
            "label": "merged",
            "input_csv": str(input_csv),
            "output_json": str(out_json),
            "output_trades_csv": str(out_trades),
            "output_equity_csv": str(out_equity),
        }
    ]
    tmp_cfg = workdir / "v14_tmp_config.json"
    tmp_cfg.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    subprocess.run(
        ["python3", str(SCRIPT_V14), "--config", str(tmp_cfg)],
        check=True,
        cwd=str(ROOT),
    )
    return json.loads(out_json.read_text(encoding="utf-8")), out_json


def _run_v44(v44_config_path: Path, input_csv: str, workdir: Path) -> tuple[dict[str, Any], Path]:
    raw = json.loads(v44_config_path.read_text(encoding="utf-8"))
    if "config" in raw and isinstance(raw["config"], dict):
        embedded = raw["config"]
    else:
        embedded = raw
    out_json = workdir / "v44_report.json"
    flat_cfg = _convert_v44_embedded_to_flat(embedded, input_csv, str(out_json))
    tmp_cfg = workdir / "v44_tmp_config.json"
    tmp_cfg.write_text(json.dumps(flat_cfg, indent=2), encoding="utf-8")
    subprocess.run(
        ["python3", str(SCRIPT_V44), "--config", str(tmp_cfg)],
        check=True,
        cwd=str(ROOT),
    )
    return json.loads(out_json.read_text(encoding="utf-8")), out_json


def _load_v44_embedded_config(v44_config_path: Path) -> dict[str, Any]:
    raw = json.loads(v44_config_path.read_text(encoding="utf-8"))
    if "config" in raw and isinstance(raw["config"], dict):
        return dict(raw["config"])
    return dict(raw)


def _extract_v14_trades(report: dict[str, Any]) -> list[TradeRow]:
    out: list[TradeRow] = []
    for t in report.get("trades", []):
        entry_ts = pd.Timestamp(t.get("entry_datetime", t.get("entry_ts")))
        exit_ts = pd.Timestamp(t.get("exit_datetime", t.get("exit_ts")))
        if entry_ts.tzinfo is None:
            entry_ts = entry_ts.tz_localize("UTC")
        if exit_ts.tzinfo is None:
            exit_ts = exit_ts.tz_localize("UTC")
        out.append(
            TradeRow(
                strategy="v14",
                entry_time=entry_ts.tz_convert("UTC"),
                exit_time=exit_ts.tz_convert("UTC"),
                entry_session=str(t.get("entry_session", "tokyo")),
                side="buy" if str(t.get("direction", "long")).lower() in {"long", "buy"} else "sell",
                pips=_safe_float(t.get("pips", t.get("pnl_pips", 0.0))),
                usd=_safe_float(t.get("usd", t.get("pnl_usd", 0.0))),
                exit_reason=str(t.get("exit_reason", "")),
                raw=t,
                standalone_entry_equity=_safe_float(t.get("equity_before", 0.0)),
            )
        )
    return out


def _extract_v44_trades(report: dict[str, Any], baseline_account_equity: float) -> list[TradeRow]:
    out: list[TradeRow] = []
    for t in report.get("results", {}).get("closed_trades", []):
        entry_ts = pd.Timestamp(t.get("entry_time"))
        exit_ts = pd.Timestamp(t.get("exit_time"))
        if entry_ts.tzinfo is None:
            entry_ts = entry_ts.tz_localize("UTC")
        if exit_ts.tzinfo is None:
            exit_ts = exit_ts.tz_localize("UTC")
        out.append(
            TradeRow(
                strategy="v44",
                entry_time=entry_ts.tz_convert("UTC"),
                exit_time=exit_ts.tz_convert("UTC"),
                entry_session=str(t.get("entry_session", "")),
                side=str(t.get("side", "")),
                pips=_safe_float(t.get("pips", 0.0)),
                usd=_safe_float(t.get("usd", 0.0)),
                exit_reason=str(t.get("exit_reason", "")),
                raw=t,
                standalone_entry_equity=float(baseline_account_equity),
            )
        )
    return out


def _apply_shared_equity_coupling(
    trades: list[TradeRow],
    starting_equity: float,
    v14_max_units: int,
    v44_max_lot: float,
) -> list[TradeRow]:
    if not trades:
        return []
    # Copy rows so we do not mutate source objects.
    sim = [TradeRow(**{**t.__dict__}) for t in trades]
    by_idx = {i: t for i, t in enumerate(sim)}
    events: list[tuple[pd.Timestamp, int, int]] = []
    # Order: exits before entries on the same timestamp (engines manage exits first, then entries).
    for i, t in by_idx.items():
        events.append((pd.Timestamp(t.entry_time), 1, i))  # entry
        events.append((pd.Timestamp(t.exit_time), 0, i))   # exit
    events.sort(key=lambda x: (x[0], x[1]))

    equity = float(starting_equity)
    entry_scale: dict[int, float] = {}
    for _, evt_type, i in events:
        t = by_idx[i]
        if evt_type == 1:
            base_eq = float(t.standalone_entry_equity) if float(t.standalone_entry_equity) > 0 else float(starting_equity)
            scale = float(equity / base_eq) if base_eq > 0 else 1.0
            # Approximate cap behavior: if already capped at baseline size, do not upscale above 1x.
            if t.strategy == "v14":
                raw_units = _safe_float(t.raw.get("position_units", t.raw.get("position_size_units", 0.0)))
                if raw_units >= float(v14_max_units) - 1 and scale > 1.0:
                    scale = 1.0
            else:
                raw_lots = _safe_float(t.raw.get("lots", 0.0))
                if raw_lots >= float(v44_max_lot) - 1e-9 and scale > 1.0:
                    scale = 1.0
            # Guard rails.
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0
            entry_scale[i] = float(scale)
            t.size_scale = float(scale)
        else:
            sc = float(entry_scale.get(i, 1.0))
            t.usd = float(t.usd * sc)
            equity += float(t.usd)
    return sim


def _build_equity_curve(trades: list[TradeRow], starting_equity: float) -> list[dict[str, Any]]:
    equity = float(starting_equity)
    peak = equity
    rows = []
    for i, t in enumerate(sorted(trades, key=lambda x: (x.exit_time, x.entry_time)), start=1):
        equity += t.usd
        peak = max(peak, equity)
        dd = peak - equity
        rows.append(
            {
                "trade_number": i,
                "strategy": t.strategy,
                "entry_time": t.entry_time.isoformat(),
                "exit_time": t.exit_time.isoformat(),
                "pnl_usd": float(t.usd),
                "equity_after": float(equity),
                "drawdown_usd": float(dd),
            }
        )
    return rows


def _group_monthly(trades: list[TradeRow]) -> list[dict[str, Any]]:
    acc: dict[str, dict[str, Any]] = {}
    for t in trades:
        k = pd.Timestamp(t.exit_time).tz_convert("UTC").strftime("%Y-%m")
        if k not in acc:
            acc[k] = {"month": k, "trades": 0, "net_usd": 0.0, "net_pips": 0.0, "wins": 0, "losses": 0}
        acc[k]["trades"] += 1
        acc[k]["net_usd"] += float(t.usd)
        acc[k]["net_pips"] += float(t.pips)
        if t.usd > 0:
            acc[k]["wins"] += 1
        else:
            acc[k]["losses"] += 1
    rows = []
    for k in sorted(acc):
        r = acc[k]
        wr = (r["wins"] / r["trades"] * 100.0) if r["trades"] else 0.0
        rows.append(
            {
                "month": k,
                "trades": int(r["trades"]),
                "wins": int(r["wins"]),
                "losses": int(r["losses"]),
                "win_rate_pct": float(wr),
                "net_usd": float(r["net_usd"]),
                "net_pips": float(r["net_pips"]),
            }
        )
    return rows


def _subset_breakdown(trades: list[TradeRow], key_fn) -> list[dict[str, Any]]:
    groups: dict[str, list[TradeRow]] = defaultdict(list)
    for t in trades:
        groups[str(key_fn(t))].append(t)
    rows = []
    for k in sorted(groups):
        ts = groups[k]
        rows.append(
            {
                "key": k,
                "trades": len(ts),
                "win_rate_pct": (sum(1 for x in ts if x.usd > 0) / len(ts) * 100.0) if ts else 0.0,
                "pf": _profit_factor(ts),
                "net_usd": float(sum(x.usd for x in ts)),
                "net_pips": float(sum(x.pips for x in ts)),
            }
        )
    return rows


def _print_summary(v14: dict[str, Any], v44: dict[str, Any], combo: dict[str, Any]) -> None:
    def _row(name: str, s: dict[str, Any]) -> str:
        return (
            f"║ {name:<13} ║ {int(s['total_trades']):<8} ║ {s['win_rate_pct']:<6.1f} ║ "
            f"{s['profit_factor']:<5.2f} ║ {s['net_usd']:+9.0f} ║ {s['max_drawdown_usd']:<7.0f} ║"
        )

    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║           MERGED V14+V44 BACKTEST RESULTS                       ║")
    print("╠═══════════════╦══════════╦════════╦═══════╦═══════════╦═════════╣")
    print("║ Strategy      ║ Trades   ║ WR%    ║ PF    ║ Net USD   ║ Max DD  ║")
    print("╠═══════════════╬══════════╬════════╬═══════╬═══════════╬═════════╣")
    print(_row("V14 Tokyo", v14))
    print(_row("V44 London+NY", v44))
    print(_row("COMBINED", combo))
    print("╚═══════════════╩══════════╩════════╩═══════╩═══════════╩═════════╝")
    print(
        f"Combined Sharpe: {combo['sharpe_ratio']:.2f} | Calmar: {combo['calmar_ratio']:.2f} | "
        f"Return: {((combo['net_usd']/100000.0)*100.0):+.2f}%"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merged V14 + V44 backtest orchestrator")
    p.add_argument("--v14-config", required=True)
    p.add_argument("--v44-config", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--starting-equity", type=float, default=100000.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_csv = str(Path(args.input))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="merged_v14_v44_") as td:
        workdir = Path(td)
        v14_report, _ = _run_v14(Path(args.v14_config), input_csv, workdir)
        v44_report, _ = _run_v44(Path(args.v44_config), input_csv, workdir)

    v44_embedded_cfg = _load_v44_embedded_config(Path(args.v44_config))
    v14_cfg = json.loads(Path(args.v14_config).read_text(encoding="utf-8"))
    v14_max_units = _safe_int(v14_cfg.get("position_sizing", {}).get("max_units", 500000), 500000)
    v44_max_lot = _safe_float(v44_embedded_cfg.get("v5", {}).get("rp_max_lot", 20.0), 20.0)
    v44_base_account = _safe_float(v44_embedded_cfg.get("v5", {}).get("account_size", 100000.0), 100000.0)

    v14_trades_raw = _extract_v14_trades(v14_report)
    v44_trades_raw = _extract_v44_trades(v44_report, baseline_account_equity=v44_base_account)
    all_trades_raw = sorted(v14_trades_raw + v44_trades_raw, key=lambda x: (x.exit_time, x.entry_time))
    all_trades = _apply_shared_equity_coupling(
        all_trades_raw,
        starting_equity=float(args.starting_equity),
        v14_max_units=v14_max_units,
        v44_max_lot=v44_max_lot,
    )
    v14_trades = [t for t in all_trades if t.strategy == "v14"]
    v44_trades = [t for t in all_trades if t.strategy == "v44"]

    equity_curve = _build_equity_curve(all_trades, float(args.starting_equity))

    combined_stats = _stats(all_trades, float(args.starting_equity), equity_curve)
    v14_stats_calc = _stats(v14_trades, float(args.starting_equity), _build_equity_curve(v14_trades, float(args.starting_equity)))
    v44_stats_calc = _stats(v44_trades, float(args.starting_equity), _build_equity_curve(v44_trades, float(args.starting_equity)))

    v14_src = v14_report.get("summary", {})
    v44_src = v44_report.get("results", {}).get("summary", {})
    v14_stats = {
        **v14_stats_calc,
        "total_trades": _safe_int(v14_src.get("total_trades", v14_stats_calc["total_trades"])),
        "wins": _safe_int(v14_stats_calc.get("wins")),
        "losses": _safe_int(v14_stats_calc.get("losses")),
        "win_rate_pct": _safe_float(v14_src.get("win_rate_pct", v14_stats_calc["win_rate_pct"])),
        "profit_factor": _safe_float(v14_src.get("profit_factor", v14_stats_calc["profit_factor"])),
        "net_usd": _safe_float(v14_src.get("net_profit_usd", v14_stats_calc["net_usd"])),
        "net_pips": _safe_float(v14_src.get("net_profit_pips", v14_stats_calc["net_pips"])),
        "max_drawdown_usd": _safe_float(v14_src.get("max_drawdown_usd", v14_stats_calc["max_drawdown_usd"])),
        "max_drawdown_pct": _safe_float(v14_src.get("max_drawdown_pct", v14_stats_calc["max_drawdown_pct"])),
        "avg_win_pips": _safe_float(v14_src.get("average_win_pips", v14_stats_calc["avg_win_pips"])),
        "avg_loss_pips": _safe_float(v14_src.get("average_loss_pips", v14_stats_calc["avg_loss_pips"])),
        "avg_win_usd": _safe_float(v14_src.get("average_win_usd", v14_stats_calc["avg_win_usd"])),
        "avg_loss_usd": _safe_float(v14_src.get("average_loss_usd", v14_stats_calc["avg_loss_usd"])),
        "sharpe_ratio": _safe_float(v14_src.get("sharpe_ratio", v14_stats_calc["sharpe_ratio"])),
        "calmar_ratio": _safe_float(v14_src.get("calmar_ratio", v14_stats_calc["calmar_ratio"])),
    }
    v44_stats = {
        **v44_stats_calc,
        "total_trades": _safe_int(v44_src.get("trades", v44_stats_calc["total_trades"])),
        "wins": _safe_int(v44_src.get("wins", v44_stats_calc["wins"])),
        "losses": _safe_int(v44_src.get("losses", v44_stats_calc["losses"])),
        "win_rate_pct": _safe_float(v44_src.get("win_rate", v44_stats_calc["win_rate_pct"])),
        "profit_factor": _safe_float(v44_src.get("profit_factor", v44_stats_calc["profit_factor"])),
        "net_usd": _safe_float(v44_src.get("net_usd", v44_stats_calc["net_usd"])),
        "net_pips": _safe_float(v44_src.get("net_pips", v44_stats_calc["net_pips"])),
        "max_drawdown_usd": _safe_float(v44_src.get("max_drawdown_usd", v44_stats_calc["max_drawdown_usd"])),
        "avg_win_pips": _safe_float(v44_src.get("avg_win_pips", v44_stats_calc["avg_win_pips"])),
        "avg_loss_pips": _safe_float(v44_src.get("avg_loss_pips", v44_stats_calc["avg_loss_pips"])),
    }

    m1 = pd.read_csv(input_csv)
    m1["time"] = pd.to_datetime(m1["time"], utc=True)
    date_range = {
        "start_utc": pd.Timestamp(m1["time"].iloc[0]).isoformat(),
        "end_utc": pd.Timestamp(m1["time"].iloc[-1]).isoformat(),
    }

    v14_by_dow = _subset_breakdown(v14_trades, lambda t: pd.Timestamp(t.entry_time).day_name())
    v14_by_exit = _subset_breakdown(v14_trades, lambda t: t.exit_reason)

    v44_by_session = _subset_breakdown(v44_trades, lambda t: t.entry_session)
    v44_by_signal = _subset_breakdown(v44_trades, lambda t: t.raw.get("entry_signal_mode", ""))
    v44_by_exit = _subset_breakdown(v44_trades, lambda t: t.exit_reason)
    v44_by_strength = _subset_breakdown(v44_trades, lambda t: t.raw.get("entry_profile", ""))

    # Remap keys to expected names.
    for row in v14_by_dow:
        row["day_of_week"] = row.pop("key")
    for row in v14_by_exit:
        row["exit_reason"] = row.pop("key")
    for row in v44_by_session:
        row["session"] = row.pop("key")
    for row in v44_by_signal:
        row["entry_signal_mode"] = row.pop("key")
    for row in v44_by_exit:
        row["exit_reason"] = row.pop("key")
    for row in v44_by_strength:
        row["trend_strength"] = row.pop("key")

    by_session = [
        {
            "session": "tokyo",
            "strategy": "v14",
            "trades": len(v14_trades),
            "win_rate": v14_stats["win_rate_pct"],
            "net_usd": v14_stats["net_usd"],
            "pf": v14_stats["profit_factor"],
        },
        {
            "session": "london",
            "strategy": "v44",
            "trades": sum(1 for t in v44_trades if t.entry_session == "london"),
            "win_rate": _subset_breakdown([t for t in v44_trades if t.entry_session == "london"], lambda x: "x")[0]["win_rate_pct"]
            if any(t.entry_session == "london" for t in v44_trades)
            else 0.0,
            "net_usd": sum(t.usd for t in v44_trades if t.entry_session == "london"),
            "pf": _profit_factor([t for t in v44_trades if t.entry_session == "london"]),
        },
        {
            "session": "ny_overlap",
            "strategy": "v44",
            "trades": sum(1 for t in v44_trades if t.entry_session == "ny_overlap"),
            "win_rate": _subset_breakdown([t for t in v44_trades if t.entry_session == "ny_overlap"], lambda x: "x")[0]["win_rate_pct"]
            if any(t.entry_session == "ny_overlap" for t in v44_trades)
            else 0.0,
            "net_usd": sum(t.usd for t in v44_trades if t.entry_session == "ny_overlap"),
            "pf": _profit_factor([t for t in v44_trades if t.entry_session == "ny_overlap"]),
        },
    ]

    closed_trades = []
    for t in all_trades:
        closed_trades.append(
            {
                "strategy": t.strategy,
                "entry_time": t.entry_time.isoformat(),
                "exit_time": t.exit_time.isoformat(),
                "entry_session": t.entry_session,
                "side": t.side,
                "pips": t.pips,
                "usd": t.usd,
                "exit_reason": t.exit_reason,
                "entry_signal_mode": t.raw.get("entry_signal_mode"),
                "trend_strength": t.raw.get("entry_profile"),
                "size_scale": t.size_scale,
            }
        )

    output = {
        "engine": "merged_v14_v44",
        "dataset": Path(input_csv).name,
        "date_range": date_range,
        "starting_equity": float(args.starting_equity),
        "ending_equity": float(args.starting_equity + combined_stats["net_usd"]),
        "combined": combined_stats,
        "v14_subset": {
            **v14_stats,
            "by_day_of_week": v14_by_dow,
            "by_exit_reason": v14_by_exit,
        },
        "v44_subset": {
            **v44_stats,
            "by_session": v44_by_session,
            "by_entry_signal_mode": v44_by_signal,
            "by_exit_reason": v44_by_exit,
            "by_trend_strength": v44_by_strength,
            "sizing_stats": v44_report.get("results", {}).get("sizing_stats", {}),
        },
        "by_session": by_session,
        "by_month": _group_monthly(all_trades),
        "equity_curve": equity_curve,
        "closed_trades": closed_trades,
        "diagnostics": {
            "v14": {
                "signals_generated": _safe_int(v14_report.get("entry_confirmation_stats", {}).get("signals_generated", 0)),
                "signals_confirmed": _safe_int(v14_report.get("entry_confirmation_stats", {}).get("signals_confirmed", 0)),
                "signals_expired": _safe_int(v14_report.get("entry_confirmation_stats", {}).get("signals_expired", 0)),
                "entries_blocked_by": v14_report.get("diagnostics", {}),
            },
            "v44": {
                "entries_attempted": _safe_int(v44_report.get("results", {}).get("summary", {}).get("trades", 0)),
                "entries_blocked_by": v44_report.get("results", {}).get("blocked_reasons", {}),
                "news_trend_entries": _safe_int(
                    sum(
                        1
                        for t in v44_report.get("results", {}).get("closed_trades", [])
                        if str(t.get("entry_signal_mode", "")).lower() == "news_trend"
                    )
                ),
                "sl_cap_skipped": _safe_int(v44_report.get("results", {}).get("sl_cap_skipped_count", 0)),
            },
        },
    }

    out_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    _print_summary(output["v14_subset"], output["v44_subset"], output["combined"])
    print(f"Wrote merged report -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
