#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path("/Users/codygnon/Documents/usdjpy_assistant")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import backtest_session_momentum as v44_engine
from scripts import backtest_tokyo_meanrev as v14_engine
from scripts import backtest_v2_multisetup_london as v2_engine


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
    standalone_entry_equity: float
    raw: dict[str, Any]
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
        return 999.0 if wins > 0 else 0.0
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
    return float(max_dd), float(max_dd_pct)


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
    }


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


def _convert_v44_embedded_to_flat(embedded: dict[str, Any], input_csv: str, out_json: str) -> dict[str, Any]:
    # Newer research configs are already flat v5 configs. Preserve them as-is
    # instead of trying to reinterpret them as nested "embedded" configs.
    if (
        isinstance(embedded, dict)
        and str(embedded.get("version", "")).lower() == "v5"
        and "ny_start" in embedded
        and any(str(k).startswith("v5_") for k in embedded.keys())
    ):
        flat = dict(embedded)
        flat["inputs"] = [input_csv]
        flat["out"] = out_json
        return flat

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
    # Enforce NY-only routing for integrated engine.
    flat["v5_sessions"] = "ny_only"
    return flat


def _load_m1(input_csv: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    need = {"time", "open", "high", "low", "close"}
    if not need.issubset(df.columns):
        raise ValueError(f"{input_csv} missing required columns: {sorted(need)}")
    df = df[["time", "open", "high", "low", "close"]].copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().sort_values("time").reset_index(drop=True)
    return df


def _run_v14_in_process(v14_config_path: Path, input_csv: str) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = json.loads(v14_config_path.read_text(encoding="utf-8"))
    run_cfg = {
        "label": "integrated",
        "input_csv": input_csv,
        "output_json": "",
        "output_trades_csv": "",
        "output_equity_csv": "",
    }
    report = v14_engine.run_one(cfg, run_cfg)
    return report, cfg


def _run_v2_in_process(v2_config_path: Path, m1: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    user_cfg = json.loads(v2_config_path.read_text(encoding="utf-8"))
    cfg = v2_engine.merge_config(user_cfg)
    trades_df, _, diagnostics = v2_engine.run_backtest(m1, cfg)
    return trades_df, diagnostics, cfg


def _run_v44_in_process(v44_config_path: Path, input_csv: str) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = json.loads(v44_config_path.read_text(encoding="utf-8"))
    embedded = raw.get("config", raw) if isinstance(raw, dict) else {}
    with tempfile.TemporaryDirectory(prefix="phase3_v44_") as td:
        tmp_cfg = Path(td) / "v44_flat.json"
        flat = _convert_v44_embedded_to_flat(embedded, input_csv, str(Path(td) / "v44_out.json"))
        # Preserve official Phase 3 behavior for flat V5 configs too.  The
        # early-return in _convert_v44_embedded_to_flat() can otherwise leave
        # session mode as "both", which inflates London-session V44 entries.
        flat["v5_sessions"] = "ny_only"
        tmp_cfg.write_text(json.dumps(flat, indent=2), encoding="utf-8")
        args = v44_engine.parse_args(["--config", str(tmp_cfg)])
        results = v44_engine.run_backtest_v5(args)
    return results, embedded


def _build_variant_k_trades(
    input_csv: str,
    starting_equity: float,
) -> tuple[list[TradeRow], dict[str, Any]]:
    # Reuse the researched Variant K pre-coupling builder, but couple and report
    # on the exact official integrated generator path.
    from scripts import backtest_variant_k_london_cluster as variant_k

    kept, baseline, _, _, blocked_cluster, blocked_global = variant_k.build_variant_k_pre_coupling_kept(input_csv)
    all_raw = sorted(kept, key=lambda x: (x.exit_time, x.entry_time))
    all_trades = _apply_shared_equity_coupling(all_raw, float(starting_equity), v14_max_units=baseline["v14_max_units"])
    meta = {
        "variant": "K",
        "blocked_cluster": blocked_cluster,
        "blocked_global": blocked_global,
        "baseline_v14_max_units": baseline["v14_max_units"],
    }
    return all_trades, meta


def _extract_v14_trades(report: dict[str, Any], default_entry_equity: float) -> list[TradeRow]:
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
                entry_session="tokyo",
                side="buy" if str(t.get("direction", "long")).lower() in {"long", "buy"} else "sell",
                pips=_safe_float(t.get("pips", 0.0)),
                usd=_safe_float(t.get("usd", 0.0)),
                exit_reason=str(t.get("exit_reason", "")),
                standalone_entry_equity=_safe_float(t.get("equity_before"), default_entry_equity),
                raw=t,
            )
        )
    return out


def _extract_v2_trades(trades_df: pd.DataFrame, default_entry_equity: float) -> list[TradeRow]:
    out: list[TradeRow] = []
    if trades_df is None or trades_df.empty:
        return out
    for _, t in trades_df.iterrows():
        entry_ts = pd.Timestamp(t["entry_time_utc"])
        exit_ts = pd.Timestamp(t["exit_time_utc"])
        if entry_ts.tzinfo is None:
            entry_ts = entry_ts.tz_localize("UTC")
        if exit_ts.tzinfo is None:
            exit_ts = exit_ts.tz_localize("UTC")
        risk_pct = _safe_float(t.get("risk_pct"), 0.0)
        sl_pips = _safe_float(t.get("sl_pips"), 0.0)
        units = _safe_float(t.get("position_units"), 0.0)
        entry_price = _safe_float(t.get("entry_price"), 0.0)
        inferred_eq = default_entry_equity
        if risk_pct > 0 and sl_pips > 0 and units > 0 and entry_price > 0:
            # units = (equity*risk_pct) / (sl_pips*(0.01/price))
            inferred_eq = units * sl_pips * (0.01 / entry_price) / risk_pct
        side = str(t.get("direction", "long")).lower()
        out.append(
            TradeRow(
                strategy="london_v2",
                entry_time=entry_ts.tz_convert("UTC"),
                exit_time=exit_ts.tz_convert("UTC"),
                entry_session="london",
                side="buy" if side == "long" else "sell",
                pips=_safe_float(t.get("pnl_pips"), 0.0),
                usd=_safe_float(t.get("pnl_usd"), 0.0),
                exit_reason=str(t.get("exit_reason", "")),
                standalone_entry_equity=float(inferred_eq),
                raw={k: (v.item() if hasattr(v, "item") else v) for k, v in t.to_dict().items()},
            )
        )
    return out


def _extract_v44_trades(results: dict[str, Any], default_entry_equity: float) -> list[TradeRow]:
    out: list[TradeRow] = []
    for t in results.get("closed_trades", []):
        entry_ts = pd.Timestamp(t.get("entry_time"))
        exit_ts = pd.Timestamp(t.get("exit_time"))
        if entry_ts.tzinfo is None:
            entry_ts = entry_ts.tz_localize("UTC")
        if exit_ts.tzinfo is None:
            exit_ts = exit_ts.tz_localize("UTC")
        out.append(
            TradeRow(
                strategy="v44_ny",
                entry_time=entry_ts.tz_convert("UTC"),
                exit_time=exit_ts.tz_convert("UTC"),
                entry_session="ny",
                side=str(t.get("side", "")).lower(),
                pips=_safe_float(t.get("pips", 0.0)),
                usd=_safe_float(t.get("usd", 0.0)),
                exit_reason=str(t.get("exit_reason", "")),
                standalone_entry_equity=float(default_entry_equity),
                raw=t,
            )
        )
    return out


def _apply_shared_equity_coupling(
    trades: list[TradeRow],
    starting_equity: float,
    v14_max_units: int,
) -> list[TradeRow]:
    if not trades:
        return []
    sim = [TradeRow(**{**t.__dict__}) for t in trades]
    by_idx = {i: t for i, t in enumerate(sim)}
    events: list[tuple[pd.Timestamp, int, int]] = []
    # exits first at the same timestamp
    for i, t in by_idx.items():
        events.append((pd.Timestamp(t.entry_time), 1, i))
        events.append((pd.Timestamp(t.exit_time), 0, i))
    events.sort(key=lambda x: (x[0], x[1]))

    equity = float(starting_equity)
    entry_scale: dict[int, float] = {}
    for _, evt_type, i in events:
        t = by_idx[i]
        if evt_type == 1:
            base_eq = float(t.standalone_entry_equity) if float(t.standalone_entry_equity) > 0 else float(starting_equity)
            scale = float(equity / base_eq) if base_eq > 0 else 1.0
            if t.strategy == "v14":
                raw_units = _safe_float(t.raw.get("position_size_units", t.raw.get("position_units", 0.0)))
                if raw_units >= float(v14_max_units) - 1 and scale > 1.0:
                    scale = 1.0
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
        rows.append(
            {
                "trade_number": i,
                "strategy": t.strategy,
                "entry_time": t.entry_time.isoformat(),
                "exit_time": t.exit_time.isoformat(),
                "pnl_usd": float(t.usd),
                "equity_after": float(equity),
                "drawdown_usd": float(peak - equity),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Integrated shared-equity merged backtest: V14 + London v2 + V44(NY)")
    p.add_argument("--v14-config", required=True)
    p.add_argument("--london-v2-config", required=True)
    p.add_argument("--v44-config", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--starting-equity", type=float, default=100000.0)
    p.add_argument("--variant", choices=["baseline", "k"], default="baseline")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_csv = str(Path(args.input))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    m1 = _load_m1(input_csv)
    date_range = {
        "start_utc": pd.Timestamp(m1["time"].iloc[0]).isoformat(),
        "end_utc": pd.Timestamp(m1["time"].iloc[-1]).isoformat(),
    }

    v14_report, v14_cfg = _run_v14_in_process(Path(args.v14_config), input_csv)
    v2_trades_df, v2_diag, v2_cfg = _run_v2_in_process(Path(args.london_v2_config), m1)
    v44_results, v44_embedded = _run_v44_in_process(Path(args.v44_config), input_csv)

    if args.variant == "baseline":
        v44_base_eq = _safe_float(v44_embedded.get("v5", {}).get("account_size", args.starting_equity), args.starting_equity)
        v14_trades_raw = _extract_v14_trades(v14_report, default_entry_equity=float(args.starting_equity))
        v2_trades_raw = _extract_v2_trades(v2_trades_df, default_entry_equity=float(args.starting_equity))
        v44_trades_raw = _extract_v44_trades(v44_results, default_entry_equity=float(v44_base_eq))

        all_raw = sorted(v14_trades_raw + v2_trades_raw + v44_trades_raw, key=lambda x: (x.exit_time, x.entry_time))
        v14_max_units = _safe_int(v14_cfg.get("position_sizing", {}).get("max_units", 500000), 500000)
        all_trades = _apply_shared_equity_coupling(all_raw, float(args.starting_equity), v14_max_units=v14_max_units)
        variant_notes: dict[str, Any] = {"variant": "baseline"}
    else:
        all_trades, variant_notes = _build_variant_k_trades(input_csv, float(args.starting_equity))

    v14_trades = [t for t in all_trades if t.strategy == "v14"]
    v2_trades = [t for t in all_trades if t.strategy == "london_v2"]
    v44_trades = [t for t in all_trades if t.strategy == "v44_ny"]

    eq_curve = _build_equity_curve(all_trades, float(args.starting_equity))
    combined_stats = _stats(all_trades, float(args.starting_equity), eq_curve)
    v14_stats = _stats(v14_trades, float(args.starting_equity), _build_equity_curve(v14_trades, float(args.starting_equity)))
    v2_stats = _stats(v2_trades, float(args.starting_equity), _build_equity_curve(v2_trades, float(args.starting_equity)))
    v44_stats = _stats(v44_trades, float(args.starting_equity), _build_equity_curve(v44_trades, float(args.starting_equity)))

    by_session = [
        {"session": "tokyo", "strategy": "v14", "trades": len(v14_trades), "win_rate": v14_stats["win_rate_pct"], "pf": v14_stats["profit_factor"], "net_usd": v14_stats["net_usd"]},
        {"session": "london", "strategy": "london_v2", "trades": len(v2_trades), "win_rate": v2_stats["win_rate_pct"], "pf": v2_stats["profit_factor"], "net_usd": v2_stats["net_usd"]},
        {"session": "ny", "strategy": "v44_ny", "trades": len(v44_trades), "win_rate": v44_stats["win_rate_pct"], "pf": v44_stats["profit_factor"], "net_usd": v44_stats["net_usd"]},
    ]

    by_strategy_breakdown = _subset_breakdown(all_trades, lambda t: t.strategy)
    for row in by_strategy_breakdown:
        row["strategy"] = row.pop("key")

    closed_trades = [
        {
            "strategy": t.strategy,
            "entry_time": t.entry_time.isoformat(),
            "exit_time": t.exit_time.isoformat(),
            "entry_session": t.entry_session,
            "side": t.side,
            "pips": float(t.pips),
            "usd": float(t.usd),
            "exit_reason": t.exit_reason,
            "size_scale": float(t.size_scale),
        }
        for t in sorted(all_trades, key=lambda x: (x.exit_time, x.entry_time))
    ]

    output = {
        "engine": "merged_tokyo_london_v2_ny_integrated",
        "dataset": Path(input_csv).name,
        "date_range": date_range,
        "starting_equity": float(args.starting_equity),
        "ending_equity": float(args.starting_equity + combined_stats["net_usd"]),
        "variant": args.variant,
        "combined": combined_stats,
        "v14_subset": {
            **v14_stats,
            "by_day_of_week": _subset_breakdown(v14_trades, lambda t: pd.Timestamp(t.entry_time).day_name()),
            "by_exit_reason": _subset_breakdown(v14_trades, lambda t: t.exit_reason),
        },
        "london_v2_subset": {
            **v2_stats,
            "by_day_of_week": _subset_breakdown(v2_trades, lambda t: pd.Timestamp(t.entry_time).day_name()),
            "by_exit_reason": _subset_breakdown(v2_trades, lambda t: t.exit_reason),
        },
        "v44_ny_subset": {
            **v44_stats,
            "by_entry_signal_mode": _subset_breakdown(v44_trades, lambda t: t.raw.get("entry_signal_mode", "")),
            "by_exit_reason": _subset_breakdown(v44_trades, lambda t: t.exit_reason),
            "by_trend_strength": _subset_breakdown(v44_trades, lambda t: t.raw.get("entry_profile", "")),
        },
        "by_session": by_session,
        "by_strategy": by_strategy_breakdown,
        "by_month": _group_monthly(all_trades),
        "equity_curve": eq_curve,
        "closed_trades": closed_trades,
        "notes": {
            "london_v2_baseline_config": str(Path(args.london_v2_config)),
            "v44_mode_applied": "ny_only",
            "implementation": "single orchestrator process with shared-equity trade coupling",
            "v2_diagnostics": v2_diag,
            "v2_config_active_days": v2_cfg.get("session", {}).get("active_days_utc", []),
            "v14_active_days": v14_cfg.get("session_filter", {}).get("active_days_utc", []),
            "v44_sessions_utc": v44_embedded.get("sessions_utc", {}),
            "variant_notes": variant_notes,
        },
    }

    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        "engine": output["engine"],
        "dataset": output["dataset"],
        "combined": {
            "trades": output["combined"]["total_trades"],
            "wr_pct": round(output["combined"]["win_rate_pct"], 2),
            "pf": round(output["combined"]["profit_factor"], 3),
            "net_usd": round(output["combined"]["net_usd"], 2),
            "max_dd_usd": round(output["combined"]["max_drawdown_usd"], 2),
        },
        "subsets": {
            "v14": {"trades": output["v14_subset"]["total_trades"], "net_usd": round(output["v14_subset"]["net_usd"], 2)},
            "london_v2": {"trades": output["london_v2_subset"]["total_trades"], "net_usd": round(output["london_v2_subset"]["net_usd"], 2)},
            "v44_ny": {"trades": output["v44_ny_subset"]["total_trades"], "net_usd": round(output["v44_ny_subset"]["net_usd"], 2)},
        },
        "output": str(out_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
