#!/usr/bin/env python3
"""
Conservative V44 routing layer backtest — aligned to official Phase 3 baseline.

Keeps session ownership unchanged for all strategies.  Applies a negative
routing overlay on V44 trades only:

  Variant A – Block V44 in breakout / post_breakout_trend regimes.
  Variant B – Block V44 in breakout / post_breakout_trend AND hard-skip V44
              when ER is low + trend decay is high (exhaustion skip-gate).
  Variant C – ER + decay hard-skip gate only (no regime block).
  Variant D – Block V44 in breakout / post_breakout_trend AND soft-penalize
              V44 when ER is low + trend decay is high (half lot size).
  Variant E – V44 momentum-only authorization (block all non-momentum regimes).

IMPORTANT: Baseline alignment fix (2026-03-26)
    The original router baseline ran V44 with v5_sessions="both" because
    _convert_v44_embedded_to_flat() has an early-return that bypasses the
    v5_sessions="ny_only" enforcement for flat V5 configs.  This produced
    ~150 extra London-session V44 trades not present in the official
    Phase 3 integrated baseline.  Fixed by forcing ny_only after conversion.

Usage:
    python scripts/backtest_v44_conservative_router.py \
        --dataset research_out/USDJPY_M1_OANDA_500k.csv \
        --output-prefix research_out/v44_conservative_router_aligned_500k
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path("/Users/codygnon/Documents/usdjpy_assistant")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.regime_classifier import RegimeThresholds
from core.regime_features import compute_efficiency_ratio, compute_trend_decay_rate
from scripts import backtest_merged_integrated_tokyo_london_v2_ny as merged_engine
from scripts import backtest_v2_multisetup_london as v2_engine
from scripts import validate_regime_classifier as regime_validation

OUT_DIR = ROOT / "research_out"
PIP = v2_engine.PIP_SIZE
BE_OFFSET_PIPS = 1.0
TRAIL_ACTIVATE_PIPS = 8.0
TRAIL_DISTANCE_PIPS = 8.0
NY_END_HOUR = 16

# Import V44 engine for direct control of session mode
v44_engine = merged_engine.v44_engine


def _safe_float(x: Any, default: float = 0.0) -> float:
    return merged_engine._safe_float(x, default)


def _safe_int(x: Any, default: int = 0) -> int:
    return merged_engine._safe_int(x, default)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return str(obj)


# ── Load & classify bars ──────────────────────────────────────────────

def _load_classified_bars(input_csv: str) -> pd.DataFrame:
    m1 = regime_validation._load_m1(input_csv)
    featured = regime_validation.compute_features(m1)
    return regime_validation.classify_all_bars(featured, RegimeThresholds())


def _lookup_regime(classified: pd.DataFrame, ts: pd.Timestamp) -> dict[str, Any]:
    time_idx = pd.DatetimeIndex(classified["time"])
    idx = time_idx.get_indexer([pd.Timestamp(ts)], method="ffill")[0]
    if idx < 0:
        idx = time_idx.get_indexer([pd.Timestamp(ts)], method="bfill")[0]
    if idx < 0:
        return {"regime_label": "ambiguous", "regime_margin": 0.0, "regime_scores": {}}
    row = classified.iloc[idx]
    return {
        "regime_label": str(row.get("regime_hysteresis", "ambiguous")),
        "regime_margin": float(row.get("score_margin", 0.0)),
        "regime_scores": {
            "momentum": float(row.get("score_momentum", 0.0)),
            "mean_reversion": float(row.get("score_mean_reversion", 0.0)),
            "breakout": float(row.get("score_breakout", 0.0)),
            "post_breakout_trend": float(row.get("score_post_breakout_trend", 0.0)),
        },
    }


# ── Build M5 data for ER/decay computation ────────────────────────────

def _build_m5(input_csv: str) -> pd.DataFrame:
    from scripts.regime_threshold_analysis import _load_m1, _resample
    m1 = _load_m1(input_csv)
    return _resample(m1, "5min")


def _compute_er_decay_at(m5: pd.DataFrame, ts: pd.Timestamp,
                         er_lookback: int = 12, decay_lookback: int = 12) -> dict[str, float]:
    """Compute ER and trend decay at a given timestamp using M5 bars up to that point."""
    ts_val = pd.Timestamp(ts)
    m5_tz = m5["time"].dt.tz
    if m5_tz is not None and ts_val.tzinfo is None:
        ts_val = ts_val.tz_localize(m5_tz)
    elif m5_tz is None and ts_val.tzinfo is not None:
        ts_val = ts_val.tz_localize(None)
    mask = m5["time"] <= ts_val
    available = m5.loc[mask]
    if len(available) < max(er_lookback, decay_lookback) + 30:
        return {"efficiency_ratio": 0.5, "trend_decay_rate": 0.0}
    er = compute_efficiency_ratio(available, lookback=er_lookback)
    decay = compute_trend_decay_rate(available, lookback=decay_lookback)
    return {"efficiency_ratio": er, "trend_decay_rate": decay}


# ── V44 with forced NY-only session (baseline alignment) ──────────────

def _run_v44_ny_only(v44_config_path: Path, input_csv: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run V44 backtest with v5_sessions forced to 'ny_only'.

    This matches the official Phase 3 integrated baseline behavior where
    _convert_v44_embedded_to_flat() enforces ny_only before the early-return
    bypass was added.
    """
    raw = json.loads(v44_config_path.read_text(encoding="utf-8"))
    embedded = raw.get("config", raw) if isinstance(raw, dict) else {}
    with tempfile.TemporaryDirectory(prefix="phase3_v44_aligned_") as td:
        tmp_cfg = Path(td) / "v44_flat.json"
        flat = merged_engine._convert_v44_embedded_to_flat(
            embedded, input_csv, str(Path(td) / "v44_out.json"),
        )
        # Force NY-only to match official integrated baseline
        flat["v5_sessions"] = "ny_only"
        tmp_cfg.write_text(json.dumps(flat, indent=2), encoding="utf-8")
        args = v44_engine.parse_args(["--config", str(tmp_cfg)])
        results = v44_engine.run_backtest_v5(args)
    return results, embedded


# ── Baseline: run all strategies matching official Phase 3 setup ──────

def _build_baseline(
    input_csv: str,
    v14_config: Path,
    london_v2_config: Path,
    v44_config: Path,
    starting_equity: float,
) -> dict[str, Any]:
    m1 = merged_engine._load_m1(input_csv)
    v14_report, v14_cfg = merged_engine._run_v14_in_process(v14_config, input_csv)
    v2_trades_df, v2_diag, v2_cfg = merged_engine._run_v2_in_process(london_v2_config, m1)
    # Use NY-only V44 to match official baseline
    v44_results, v44_embedded = _run_v44_ny_only(v44_config, input_csv)

    if isinstance(v44_embedded, dict) and "v5_account_size" in v44_embedded:
        v44_base_eq = _safe_float(v44_embedded.get("v5_account_size", starting_equity), starting_equity)
    else:
        v44_base_eq = _safe_float(v44_embedded.get("v5", {}).get("account_size", starting_equity), starting_equity)

    all_trades = sorted(
        merged_engine._extract_v14_trades(v14_report, default_entry_equity=starting_equity)
        + merged_engine._extract_v2_trades(v2_trades_df, default_entry_equity=starting_equity)
        + merged_engine._extract_v44_trades(v44_results, default_entry_equity=v44_base_eq),
        key=lambda x: (x.exit_time, x.entry_time),
    )
    v14_max_units = _safe_int(v14_cfg.get("position_sizing", {}).get("max_units", 500000), 500000)
    scaled = merged_engine._apply_shared_equity_coupling(all_trades, starting_equity, v14_max_units=v14_max_units)
    eq_curve = merged_engine._build_equity_curve(scaled, starting_equity)
    summary = merged_engine._stats(scaled, starting_equity, eq_curve)

    return {
        "trades": scaled,
        "summary": summary,
        "v14_max_units": v14_max_units,
        "by_strategy": merged_engine._subset_breakdown(scaled, lambda t: t.strategy),
        "by_session": merged_engine._subset_breakdown(scaled, lambda t: t.entry_session),
        "by_month": merged_engine._group_monthly(scaled),
    }


# ── V44 routing filters ──────────────────────────────────────────────

@dataclass
class V44FilterResult:
    blocked: bool
    reason: str
    regime_label: str
    regime_margin: float
    er: float
    decay: float
    size_scale: float  # 1.0 = full size, <1.0 = soft penalty


def _filter_v44_trade(
    trade: merged_engine.TradeRow,
    classified: pd.DataFrame,
    m5: pd.DataFrame,
    *,
    block_breakout: bool,
    block_post_breakout: bool,
    block_ambiguous_non_momentum: bool,
    momentum_only: bool,
    exhaustion_gate: bool,      # hard skip
    soft_exhaustion: bool,      # soft penalty (half size)
    er_threshold: float,
    decay_threshold: float,
) -> V44FilterResult:
    """Decide whether a V44 trade should be blocked or penalized."""
    regime = _lookup_regime(classified, trade.entry_time)
    label = regime["regime_label"]
    margin = regime["regime_margin"]

    er_decay = _compute_er_decay_at(m5, trade.entry_time)
    er = er_decay["efficiency_ratio"]
    decay = er_decay["trend_decay_rate"]

    if momentum_only and label != "momentum":
        return V44FilterResult(True, f"blocked_non_momentum_{label}", label, margin, er, decay, 0.0)

    if block_ambiguous_non_momentum and label == "ambiguous":
        scores = regime["regime_scores"]
        top_regime = max(scores, key=scores.get) if scores else "ambiguous"
        if top_regime != "momentum":
            return V44FilterResult(True, f"blocked_ambiguous_top_{top_regime}", label, margin, er, decay, 0.0)

    # Check regime block
    if block_breakout and label == "breakout":
        return V44FilterResult(True, "blocked_breakout", label, margin, er, decay, 0.0)
    if block_post_breakout and label == "post_breakout_trend":
        return V44FilterResult(True, "blocked_post_breakout", label, margin, er, decay, 0.0)

    # Check exhaustion: hard skip vs soft penalty
    exhaustion_triggered = er < er_threshold and decay > decay_threshold
    if exhaustion_triggered:
        if exhaustion_gate:
            return V44FilterResult(True, f"blocked_exhaustion_er{er:.2f}_decay{decay:.2f}",
                                   label, margin, er, decay, 0.0)
        if soft_exhaustion:
            return V44FilterResult(False, f"soft_penalty_er{er:.2f}_decay{decay:.2f}",
                                   label, margin, er, decay, 0.5)

    return V44FilterResult(False, "passed", label, margin, er, decay, 1.0)


# ── Run a variant ─────────────────────────────────────────────────────

@dataclass
class VariantConfig:
    name: str
    block_breakout: bool
    block_post_breakout: bool
    exhaustion_gate: bool       # hard skip
    block_ambiguous_non_momentum: bool = False
    momentum_only: bool = False
    soft_exhaustion: bool = False  # soft penalty (half size)
    london_holdthrough: bool = False
    er_threshold: float = 0.35
    decay_threshold: float = 0.40


def _run_variant(
    baseline_trades: list[merged_engine.TradeRow],
    classified: pd.DataFrame,
    m5: pd.DataFrame,
    variant: VariantConfig,
    starting_equity: float,
    v14_max_units: int,
) -> tuple[dict[str, Any], list[merged_engine.TradeRow]]:
    kept: list[merged_engine.TradeRow] = []
    blocked_log: list[dict[str, Any]] = []
    passed_log: list[dict[str, Any]] = []
    soft_penalized_log: list[dict[str, Any]] = []

    for trade in baseline_trades:
        if trade.strategy != "v44_ny":
            kept.append(trade)
            continue

        result = _filter_v44_trade(
            trade, classified, m5,
            block_breakout=variant.block_breakout,
            block_post_breakout=variant.block_post_breakout,
            block_ambiguous_non_momentum=variant.block_ambiguous_non_momentum,
            momentum_only=variant.momentum_only,
            exhaustion_gate=variant.exhaustion_gate,
            soft_exhaustion=variant.soft_exhaustion,
            er_threshold=variant.er_threshold,
            decay_threshold=variant.decay_threshold,
        )

        entry = {
            "entry_time": trade.entry_time.isoformat(),
            "side": trade.side,
            "pips": float(trade.pips),
            "usd": float(trade.usd),
            "regime_label": result.regime_label,
            "regime_margin": result.regime_margin,
            "er": round(result.er, 4),
            "decay": round(result.decay, 4),
            "reason": result.reason,
            "exit_reason": trade.exit_reason,
            "raw_profile": str((trade.raw or {}).get("entry_profile", "")),
            "size_scale": result.size_scale,
        }

        if result.blocked:
            blocked_log.append(entry)
        else:
            # Apply soft penalty by scaling the trade's USD
            if result.size_scale < 1.0:
                scaled_trade = merged_engine.TradeRow(
                    strategy=trade.strategy,
                    entry_time=trade.entry_time,
                    exit_time=trade.exit_time,
                    entry_session=trade.entry_session,
                    side=trade.side,
                    pips=trade.pips,
                    usd=trade.usd * result.size_scale,
                    exit_reason=trade.exit_reason,
                    standalone_entry_equity=trade.standalone_entry_equity,
                    raw=trade.raw,
                    size_scale=trade.size_scale * result.size_scale,
                )
                kept.append(scaled_trade)
                soft_penalized_log.append(entry)
            else:
                kept.append(trade)
            passed_log.append(entry)

    # Re-sort and re-apply equity coupling
    kept_sorted = sorted(kept, key=lambda t: (t.exit_time, t.entry_time))
    scaled = merged_engine._apply_shared_equity_coupling(kept_sorted, starting_equity, v14_max_units=v14_max_units)
    eq_curve = merged_engine._build_equity_curve(scaled, starting_equity)
    summary = merged_engine._stats(scaled, starting_equity, eq_curve)

    # Analyze blocked trades
    blocked_winners = [b for b in blocked_log if b["pips"] > 0]
    blocked_losers = [b for b in blocked_log if b["pips"] <= 0]
    blocked_by_regime = defaultdict(list)
    for b in blocked_log:
        blocked_by_regime[b["regime_label"]].append(b)
    blocked_by_reason = defaultdict(list)
    for b in blocked_log:
        blocked_by_reason[b["reason"]].append(b)

    # Cross-strategy interaction: per-strategy breakdown
    by_strategy_detail = {}
    for strat_key in ["v14", "london_v2", "v44_ny"]:
        strat_trades = [t for t in scaled if t.strategy == strat_key]
        if strat_trades:
            strat_eq = merged_engine._build_equity_curve(strat_trades, 0.0)
            by_strategy_detail[strat_key] = {
                "trades": len(strat_trades),
                "net_usd": round(sum(t.usd for t in strat_trades), 2),
                "winners": sum(1 for t in strat_trades if t.usd > 0),
                "losers": sum(1 for t in strat_trades if t.usd <= 0),
                "win_rate_pct": round(100.0 * sum(1 for t in strat_trades if t.usd > 0) / len(strat_trades), 1),
            }
        else:
            by_strategy_detail[strat_key] = {"trades": 0, "net_usd": 0.0, "winners": 0, "losers": 0, "win_rate_pct": 0.0}

    # V44 regime distribution of passed trades
    passed_by_regime = defaultdict(lambda: {"count": 0, "winners": 0, "losers": 0, "net_pips": 0.0})
    for p in passed_log:
        r = p["regime_label"]
        passed_by_regime[r]["count"] += 1
        if p["pips"] > 0:
            passed_by_regime[r]["winners"] += 1
        else:
            passed_by_regime[r]["losers"] += 1
        passed_by_regime[r]["net_pips"] += p["pips"]

    return {
        "name": variant.name,
        "config": {
            "block_breakout": variant.block_breakout,
            "block_post_breakout": variant.block_post_breakout,
            "block_ambiguous_non_momentum": variant.block_ambiguous_non_momentum,
            "momentum_only": variant.momentum_only,
            "exhaustion_gate": variant.exhaustion_gate,
            "soft_exhaustion": variant.soft_exhaustion,
            "london_holdthrough": variant.london_holdthrough,
            "er_threshold": variant.er_threshold,
            "decay_threshold": variant.decay_threshold,
        },
        "summary": summary,
        "by_strategy": merged_engine._subset_breakdown(scaled, lambda t: t.strategy),
        "by_strategy_detail": by_strategy_detail,
        "by_session": merged_engine._subset_breakdown(scaled, lambda t: t.entry_session),
        "by_month": merged_engine._group_monthly(scaled),
        "v44_filter_stats": {
            "total_v44_baseline": len(blocked_log) + len(passed_log),
            "v44_blocked": len(blocked_log),
            "v44_passed": len(passed_log),
            "v44_soft_penalized": len(soft_penalized_log),
            "blocked_winners": len(blocked_winners),
            "blocked_losers": len(blocked_losers),
            "blocked_winner_pips": round(sum(b["pips"] for b in blocked_winners), 1),
            "blocked_loser_pips": round(sum(abs(b["pips"]) for b in blocked_losers), 1),
            "blocked_net_pips": round(sum(b["pips"] for b in blocked_log), 1),
            "blocked_net_usd": round(sum(b["usd"] for b in blocked_log), 2),
            "blocked_by_reason": {
                reason: {
                    "count": len(trades),
                    "net_pips": round(sum(t["pips"] for t in trades), 1),
                    "net_usd": round(sum(t["usd"] for t in trades), 2),
                    "winners": sum(1 for t in trades if t["pips"] > 0),
                    "losers": sum(1 for t in trades if t["pips"] <= 0),
                }
                for reason, trades in sorted(blocked_by_reason.items())
            },
            "blocked_by_regime": {
                regime: {
                    "count": len(trades),
                    "net_pips": round(sum(t["pips"] for t in trades), 1),
                    "winners": sum(1 for t in trades if t["pips"] > 0),
                    "losers": sum(1 for t in trades if t["pips"] <= 0),
                }
                for regime, trades in sorted(blocked_by_regime.items())
            },
            "passed_by_regime": {
                regime: {
                    "count": d["count"],
                    "net_pips": round(d["net_pips"], 1),
                    "winners": d["winners"],
                    "losers": d["losers"],
                }
                for regime, d in sorted(passed_by_regime.items())
            },
        },
        "blocked_trades": blocked_log[:100],
        "soft_penalized_trades": soft_penalized_log[:50],
        "passed_v44_er_decay_stats": {
            "mean_er": round(float(np.mean([p["er"] for p in passed_log])), 4) if passed_log else 0.0,
            "mean_decay": round(float(np.mean([p["decay"] for p in passed_log])), 4) if passed_log else 0.0,
            "median_er": round(float(np.median([p["er"] for p in passed_log])), 4) if passed_log else 0.0,
            "median_decay": round(float(np.median([p["decay"] for p in passed_log])), 4) if passed_log else 0.0,
        },
    }, kept_sorted


def _simulate_london_holdthrough(trade: merged_engine.TradeRow, m1: pd.DataFrame, m1_idx: pd.DatetimeIndex) -> dict[str, Any] | None:
    raw = trade.raw or {}
    exit_reason = str(raw.get("exit_reason", trade.exit_reason))
    if exit_reason not in {"HARD_CLOSE", "TP1_ONLY_HARD_CLOSE"}:
        return None

    direction = str(raw.get("direction", "long")).lower()
    entry_price = float(raw.get("entry_price"))
    hardclose_price = float(raw.get("exit_price", 0.0))
    hardclose_time = pd.Timestamp(raw.get("exit_time_utc", trade.exit_time))
    initial_units = int(raw.get("position_units", 0))
    if initial_units <= 0:
        return None

    deadline = hardclose_time.replace(hour=NY_END_HOUR, minute=0, second=0, microsecond=0)
    if deadline <= hardclose_time:
        return None
    forward = m1.loc[(m1_idx > hardclose_time) & (m1_idx <= deadline)]
    if forward.empty:
        return None

    if direction == "long":
        runner_pips_at_hc = (hardclose_price - entry_price) / PIP
    else:
        runner_pips_at_hc = (entry_price - hardclose_price) / PIP

    close_fraction = float(raw.get("tp1_close_fraction", 0.5))
    tp1_units = 0
    runner_units = initial_units
    locked_pips_sum = 0.0
    locked_usd = 0.0
    if exit_reason == "TP1_ONLY_HARD_CLOSE":
        tp1_units = int(np.floor(initial_units * close_fraction / v2_engine.ROUND_UNITS) * v2_engine.ROUND_UNITS)
        tp1_units = max(0, min(initial_units, tp1_units))
        runner_units = initial_units - tp1_units
        tp1_price = float(raw.get("tp1_price"))
        tp1_pips, tp1_usd = v2_engine.calc_leg_pnl(direction, entry_price, tp1_price, tp1_units)
        locked_pips_sum += tp1_pips * tp1_units
        locked_usd += tp1_usd
    if runner_units <= 0:
        return None

    best_price = hardclose_price
    if runner_pips_at_hc >= TRAIL_ACTIVATE_PIPS:
        trailing_active = True
        if direction == "long":
            stop_price = max(best_price - TRAIL_DISTANCE_PIPS * PIP, entry_price + BE_OFFSET_PIPS * PIP)
        else:
            stop_price = min(best_price + TRAIL_DISTANCE_PIPS * PIP, entry_price - BE_OFFSET_PIPS * PIP)
    elif runner_pips_at_hc > 0:
        trailing_active = False
        stop_price = entry_price + BE_OFFSET_PIPS * PIP if direction == "long" else entry_price - BE_OFFSET_PIPS * PIP
    else:
        trailing_active = False
        stop_price = entry_price

    sim_exit_price = None
    sim_exit_time = None
    sim_exit_type = None
    for _, bar in forward.iterrows():
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])
        bar_close = float(bar["close"])
        bar_time = pd.Timestamp(bar["time"])
        if direction == "long":
            if bar_high > best_price:
                best_price = bar_high
            if not trailing_active and (best_price - entry_price) / PIP >= TRAIL_ACTIVATE_PIPS:
                trailing_active = True
            if trailing_active:
                candidate = max(best_price - TRAIL_DISTANCE_PIPS * PIP, entry_price + BE_OFFSET_PIPS * PIP)
                stop_price = max(stop_price, candidate)
            if bar_low <= stop_price:
                sim_exit_price = stop_price
                sim_exit_time = bar_time
                sim_exit_type = "trail_stop" if trailing_active else "be_stop"
                break
        else:
            if bar_low < best_price:
                best_price = bar_low
            if not trailing_active and (entry_price - best_price) / PIP >= TRAIL_ACTIVATE_PIPS:
                trailing_active = True
            if trailing_active:
                candidate = min(best_price + TRAIL_DISTANCE_PIPS * PIP, entry_price - BE_OFFSET_PIPS * PIP)
                stop_price = min(stop_price, candidate)
            if bar_high >= stop_price:
                sim_exit_price = stop_price
                sim_exit_time = bar_time
                sim_exit_type = "trail_stop" if trailing_active else "be_stop"
                break
    if sim_exit_price is None:
        last_bar = forward.iloc[-1]
        sim_exit_price = float(last_bar["close"])
        sim_exit_time = pd.Timestamp(last_bar["time"])
        sim_exit_type = "ny_session_close"

    runner_pips_new, runner_usd_new = v2_engine.calc_leg_pnl(direction, entry_price, sim_exit_price, runner_units)
    runner_pips_old, runner_usd_old = v2_engine.calc_leg_pnl(direction, entry_price, hardclose_price, runner_units)
    total_pips_sum_new = locked_pips_sum + runner_pips_new * runner_units
    total_usd_new = locked_usd + runner_usd_new
    total_pips_new = total_pips_sum_new / initial_units if initial_units > 0 else 0.0
    total_pips_sum_old = locked_pips_sum + runner_pips_old * runner_units
    total_usd_old = locked_usd + runner_usd_old
    total_pips_old = total_pips_sum_old / initial_units if initial_units > 0 else 0.0
    return {
        "new_exit_time": sim_exit_time,
        "new_exit_reason": f"HT_{sim_exit_type.upper()}",
        "new_pips": float(total_pips_new),
        "new_usd": float(total_usd_new),
        "delta_pips": float(total_pips_new - total_pips_old),
        "delta_usd": float(total_usd_new - total_usd_old),
        "exit_type": sim_exit_type,
        "improved": total_usd_new > total_usd_old,
        "worsened": total_usd_new < total_usd_old,
    }


def _run_holdthrough_variant(
    f_trades_pre_coupling: list[merged_engine.TradeRow],
    m1: pd.DataFrame,
    starting_equity: float,
    v14_max_units: int,
) -> dict[str, Any]:
    m1_idx = pd.DatetimeIndex(m1["time"])
    modified: list[merged_engine.TradeRow] = []
    held_stats = {
        "trades_held": 0,
        "held_improved": 0,
        "held_worsened": 0,
        "additional_pips": 0.0,
        "additional_usd": 0.0,
        "exit_distribution": Counter(),
    }
    for trade in f_trades_pre_coupling:
        if trade.strategy != "london_v2":
            modified.append(trade)
            continue
        sim = _simulate_london_holdthrough(trade, m1, m1_idx)
        if sim is None:
            modified.append(trade)
            continue
        held_stats["trades_held"] += 1
        held_stats["held_improved"] += int(sim["improved"])
        held_stats["held_worsened"] += int(sim["worsened"])
        held_stats["additional_pips"] += sim["delta_pips"]
        held_stats["additional_usd"] += sim["delta_usd"]
        held_stats["exit_distribution"][sim["exit_type"]] += 1
        new_raw = dict(trade.raw or {})
        new_raw["exit_time_utc"] = sim["new_exit_time"]
        new_raw["exit_reason"] = sim["new_exit_reason"]
        new_raw["pnl_pips"] = sim["new_pips"]
        new_raw["pnl_usd"] = sim["new_usd"]
        modified.append(
            merged_engine.TradeRow(
                strategy=trade.strategy,
                entry_time=trade.entry_time,
                exit_time=pd.Timestamp(sim["new_exit_time"]).tz_convert("UTC"),
                entry_session=trade.entry_session,
                side=trade.side,
                pips=sim["new_pips"],
                usd=sim["new_usd"],
                exit_reason=sim["new_exit_reason"],
                standalone_entry_equity=trade.standalone_entry_equity,
                raw=new_raw,
                size_scale=trade.size_scale,
            )
        )

    modified_sorted = sorted(modified, key=lambda t: (t.exit_time, t.entry_time))
    scaled = merged_engine._apply_shared_equity_coupling(modified_sorted, starting_equity, v14_max_units=v14_max_units)
    eq_curve = merged_engine._build_equity_curve(scaled, starting_equity)
    summary = merged_engine._stats(scaled, starting_equity, eq_curve)
    by_strategy_detail = {}
    for strat_key in ["v14", "london_v2", "v44_ny"]:
        strat_trades = [t for t in scaled if t.strategy == strat_key]
        by_strategy_detail[strat_key] = {
            "trades": len(strat_trades),
            "net_usd": round(sum(t.usd for t in strat_trades), 2),
            "winners": sum(1 for t in strat_trades if t.usd > 0),
            "losers": sum(1 for t in strat_trades if t.usd <= 0),
            "win_rate_pct": round(100.0 * sum(1 for t in strat_trades if t.usd > 0) / max(1, len(strat_trades)), 1),
        }
    return {
        "name": "G_london_holdthrough",
        "config": {
            "london_holdthrough": True,
            "trail_activate_pips": TRAIL_ACTIVATE_PIPS,
            "trail_distance_pips": TRAIL_DISTANCE_PIPS,
            "be_offset_pips": BE_OFFSET_PIPS,
            "deadline_hour_utc": NY_END_HOUR,
        },
        "summary": summary,
        "by_strategy": merged_engine._subset_breakdown(scaled, lambda t: t.strategy),
        "by_strategy_detail": by_strategy_detail,
        "by_session": merged_engine._subset_breakdown(scaled, lambda t: t.entry_session),
        "by_month": merged_engine._group_monthly(scaled),
        "holdthrough_stats": {
            "trades_held": int(held_stats["trades_held"]),
            "held_improved": int(held_stats["held_improved"]),
            "held_worsened": int(held_stats["held_worsened"]),
            "additional_pips": round(float(held_stats["additional_pips"]), 1),
            "additional_usd": round(float(held_stats["additional_usd"]), 2),
            "exit_distribution": {k: int(v) for k, v in sorted(held_stats["exit_distribution"].items())},
        },
    }


# ── Main ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Conservative V44 routing layer backtest (aligned)")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output-prefix", required=True)
    p.add_argument("--v14-config", default=str(OUT_DIR / "tokyo_optimized_v14_config.json"))
    p.add_argument("--london-v2-config", default=str(OUT_DIR / "v2_exp4_winner_baseline_config.json"))
    p.add_argument("--v44-config", default=str(OUT_DIR / "session_momentum_v44_base_config.json"))
    p.add_argument("--starting-equity", type=float, default=100000.0)
    p.add_argument("--er-threshold", type=float, default=0.35)
    p.add_argument("--decay-threshold", type=float, default=0.40)
    p.add_argument("--skip-soft", action="store_true", help="Skip soft-penalty variant D")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    dataset = str(Path(args.dataset).resolve())
    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    starting_equity = float(args.starting_equity)

    print(f"[1/5] Building aligned baseline (V44 NY-only, matching official Phase 3) ...")
    baseline = _build_baseline(
        dataset,
        Path(args.v14_config),
        Path(args.london_v2_config),
        Path(args.v44_config),
        starting_equity,
    )
    bs = baseline["summary"]
    print(f"       Baseline: {bs['total_trades']} trades, "
          f"${bs['net_usd']:.2f} net, PF={bs['profit_factor']:.3f}, "
          f"DD=${bs['max_drawdown_usd']:.2f}")

    # Show per-strategy breakdown
    for entry in baseline["by_strategy"]:
        print(f"         {entry['key']}: {entry['trades']} trades, ${entry['net_usd']:.2f}")

    print(f"[2/5] Classifying bars ...")
    classified = _load_classified_bars(dataset)

    print(f"[3/5] Building M5 for ER/decay ...")
    m5 = _build_m5(dataset)

    print(f"[4/5] Running routing variants ...")

    variants = [
        VariantConfig("A_breakout_block", block_breakout=True, block_post_breakout=True,
                       exhaustion_gate=False),
        VariantConfig("B_block_plus_exhaustion_skip", block_breakout=True, block_post_breakout=True,
                       exhaustion_gate=True, er_threshold=args.er_threshold, decay_threshold=args.decay_threshold),
        VariantConfig("C_exhaustion_skip_only", block_breakout=False, block_post_breakout=False,
                       exhaustion_gate=True, er_threshold=args.er_threshold, decay_threshold=args.decay_threshold),
        VariantConfig("E_momentum_only", block_breakout=False, block_post_breakout=False,
                       momentum_only=True, exhaustion_gate=False),
        VariantConfig("F_block_plus_ambiguous_non_momentum", block_breakout=True, block_post_breakout=True,
                       exhaustion_gate=False, block_ambiguous_non_momentum=True),
    ]
    if not args.skip_soft:
        variants.append(
            VariantConfig("D_block_plus_soft_exhaustion", block_breakout=True, block_post_breakout=True,
                           exhaustion_gate=False, soft_exhaustion=True,
                           er_threshold=args.er_threshold, decay_threshold=args.decay_threshold),
        )

    variant_results = {}
    variant_pre_coupling_trades: dict[str, list[merged_engine.TradeRow]] = {}
    for v in variants:
        print(f"       Running {v.name} ...")
        result, pre_coupling_trades = _run_variant(
            baseline["trades"], classified, m5, v, starting_equity, baseline["v14_max_units"],
        )
        variant_results[v.name] = result
        variant_pre_coupling_trades[v.name] = pre_coupling_trades
        s = result["summary"]
        fs = result["v44_filter_stats"]
        delta_usd = s["net_usd"] - bs["net_usd"]
        delta_pf = s["profit_factor"] - bs["profit_factor"]
        print(f"         {s['total_trades']} trades, ${s['net_usd']:.2f} net (Δ${delta_usd:+.2f}), "
              f"PF={s['profit_factor']:.3f} (Δ{delta_pf:+.4f}), DD=${s['max_drawdown_usd']:.2f}")
        print(f"         V44: {fs['v44_blocked']} blocked ({fs['blocked_losers']}L/{fs['blocked_winners']}W), "
              f"blocked net: {fs['blocked_net_pips']:.1f} pips / ${fs['blocked_net_usd']:.2f}")
        if fs["v44_soft_penalized"] > 0:
            print(f"         V44 soft-penalized: {fs['v44_soft_penalized']} trades (half size)")

    print(f"       Running G_london_holdthrough ...")
    m1 = merged_engine._load_m1(dataset)
    g_result = _run_holdthrough_variant(
        variant_pre_coupling_trades["F_block_plus_ambiguous_non_momentum"],
        m1,
        starting_equity,
        baseline["v14_max_units"],
    )
    variant_results[g_result["name"]] = g_result
    g_summary = g_result["summary"]
    g_delta_usd = g_summary["net_usd"] - bs["net_usd"]
    g_delta_pf = g_summary["profit_factor"] - bs["profit_factor"]
    print(f"         {g_summary['total_trades']} trades, ${g_summary['net_usd']:.2f} net (Δ${g_delta_usd:+.2f}), "
          f"PF={g_summary['profit_factor']:.3f} (Δ{g_delta_pf:+.4f}), DD=${g_summary['max_drawdown_usd']:.2f}")
    hs = g_result["holdthrough_stats"]
    print(f"         London HT: {hs['trades_held']} held, {hs['held_improved']} improved / {hs['held_worsened']} worsened, "
          f"add {hs['additional_pips']:+.1f} pips / ${hs['additional_usd']:+.2f}")

    print(f"[5/5] Writing outputs ...")

    # ── Baseline output ───────────────────────────────────────────────

    # Per-strategy detail for baseline
    baseline_strat_detail = {}
    for strat_key in ["v14", "london_v2", "v44_ny"]:
        strat_trades = [t for t in baseline["trades"] if t.strategy == strat_key]
        baseline_strat_detail[strat_key] = {
            "trades": len(strat_trades),
            "net_usd": round(sum(t.usd for t in strat_trades), 2),
            "winners": sum(1 for t in strat_trades if t.usd > 0),
            "losers": sum(1 for t in strat_trades if t.usd <= 0),
            "win_rate_pct": round(100.0 * sum(1 for t in strat_trades if t.usd > 0) / max(1, len(strat_trades)), 1),
        }

    baseline_out = {
        "dataset": Path(dataset).name,
        "alignment_note": "V44 forced to v5_sessions=ny_only to match official Phase 3 baseline",
        "summary": bs,
        "by_strategy": baseline["by_strategy"],
        "by_strategy_detail": baseline_strat_detail,
        "by_session": baseline["by_session"],
        "by_month": baseline["by_month"],
        "closed_trades": [
            {
                "strategy": t.strategy,
                "entry_time": t.entry_time.isoformat(),
                "exit_time": t.exit_time.isoformat(),
                "entry_session": t.entry_session,
                "side": t.side,
                "pips": float(t.pips),
                "usd": float(t.usd),
                "exit_reason": t.exit_reason,
            }
            for t in baseline["trades"]
        ],
    }

    # ── Comparison output ─────────────────────────────────────────────

    comparison = {
        "dataset": Path(dataset).name,
        "baseline": bs,
        "baseline_by_strategy": baseline_strat_detail,
        "variants": {},
    }
    for name, result in variant_results.items():
        delta = {
            "net_usd": round(result["summary"]["net_usd"] - bs["net_usd"], 2),
            "profit_factor": round(result["summary"]["profit_factor"] - bs["profit_factor"], 4),
            "max_drawdown_usd": round(result["summary"]["max_drawdown_usd"] - bs["max_drawdown_usd"], 2),
            "total_trades": result["summary"]["total_trades"] - bs["total_trades"],
            "win_rate_pct": round(result["summary"]["win_rate_pct"] - bs["win_rate_pct"], 3),
        }
        # Cross-strategy interaction
        cross_strat = {}
        for strat_key in ["v14", "london_v2", "v44_ny"]:
            bl = baseline_strat_detail[strat_key]
            vr = result["by_strategy_detail"][strat_key]
            cross_strat[strat_key] = {
                "baseline_trades": bl["trades"],
                "variant_trades": vr["trades"],
                "delta_trades": vr["trades"] - bl["trades"],
                "baseline_net_usd": bl["net_usd"],
                "variant_net_usd": vr["net_usd"],
                "delta_net_usd": round(vr["net_usd"] - bl["net_usd"], 2),
            }

        comparison["variants"][name] = {
            "summary": result["summary"],
            "delta_vs_baseline": delta,
            "v44_filter_stats": result.get("v44_filter_stats", {}),
            "cross_strategy_interaction": cross_strat,
            "passed_v44_er_decay_stats": result.get("passed_v44_er_decay_stats", {}),
        }
        if "holdthrough_stats" in result:
            comparison["variants"][name]["holdthrough_stats"] = result["holdthrough_stats"]

    # ── Write files ───────────────────────────────────────────────────

    baseline_path = out_prefix.with_name(out_prefix.name + "_baseline.json")
    comparison_path = out_prefix.with_name(out_prefix.name + "_comparison.json")
    baseline_path.write_text(json.dumps(baseline_out, indent=2, default=_json_default), encoding="utf-8")
    comparison_path.write_text(json.dumps(comparison, indent=2, default=_json_default), encoding="utf-8")

    for name, result in variant_results.items():
        variant_path = out_prefix.with_name(f"{out_prefix.name}_{name}.json")
        variant_path.write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")

    # ── Print comparison table ────────────────────────────────────────

    print("\n" + "=" * 100)
    print("COMPARISON SUMMARY")
    print("=" * 100)
    print(f"{'Variant':<42s} {'Trades':>6s} {'Net USD':>11s} {'ΔNet':>10s} {'PF':>6s} {'ΔPF':>7s} {'DD':>10s} {'ΔDD':>10s}")
    print("-" * 100)
    print(f"{'BASELINE':<42s} {bs['total_trades']:>6d} {bs['net_usd']:>11.2f} {'':>10s} {bs['profit_factor']:>6.3f} {'':>7s} {bs['max_drawdown_usd']:>10.2f} {'':>10s}")
    for name, result in variant_results.items():
        s = result["summary"]
        d = comparison["variants"][name]["delta_vs_baseline"]
        print(f"{name:<42s} {s['total_trades']:>6d} {s['net_usd']:>11.2f} {d['net_usd']:>+10.2f} {s['profit_factor']:>6.3f} {d['profit_factor']:>+7.4f} {s['max_drawdown_usd']:>10.2f} {d['max_drawdown_usd']:>+10.2f}")

    # Cross-strategy interaction table
    print("\nCROSS-STRATEGY INTERACTION (trade count / net USD)")
    print("-" * 100)
    print(f"{'Variant':<42s} {'V14 Δt':>6s} {'V14 ΔUSD':>10s} {'Ldn Δt':>6s} {'Ldn ΔUSD':>10s} {'V44 Δt':>6s} {'V44 ΔUSD':>10s}")
    print("-" * 100)
    for name in variant_results:
        cs = comparison["variants"][name]["cross_strategy_interaction"]
        print(f"{name:<42s} "
              f"{cs['v14']['delta_trades']:>+6d} {cs['v14']['delta_net_usd']:>+10.2f} "
              f"{cs['london_v2']['delta_trades']:>+6d} {cs['london_v2']['delta_net_usd']:>+10.2f} "
              f"{cs['v44_ny']['delta_trades']:>+6d} {cs['v44_ny']['delta_net_usd']:>+10.2f}")

    print(f"\nOutputs: {comparison_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
