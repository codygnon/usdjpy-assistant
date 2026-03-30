#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path("/Users/codygnon/Documents/usdjpy_assistant")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.phase3_integrated_engine import classify_session, load_phase3_sizing_config
from core.regime_classifier import RegimeThresholds
from scripts import backtest_merged_integrated_tokyo_london_v2_ny as merged_engine
from scripts import validate_regime_classifier as regime_validation

OUT_DIR = ROOT / "research_out"


@dataclass
class RouterCandidate:
    strategy: str
    side: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    sl_pips: float
    tp_pips: float
    raw_quality: float
    normalized_quality: float | None
    regime_compatible: bool | None
    source_session_owner: bool
    metadata: dict[str, Any] = field(default_factory=dict)
    trade: merged_engine.TradeRow | None = None


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


def _safe_float(x: Any, default: float = 0.0) -> float:
    return merged_engine._safe_float(x, default)


def _safe_int(x: Any, default: int = 0) -> int:
    return merged_engine._safe_int(x, default)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _all_weekdays() -> list[str]:
    return ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def _override_v14_latent(cfg: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(cfg))
    sf = out.setdefault("session_filter", {})
    sf["enabled"] = True
    sf["session_start_utc"] = "00:00"
    sf["session_end_utc"] = "23:59"
    sf["allowed_trading_days"] = _all_weekdays()
    sf["trade_only_inside_session"] = True
    sf["force_close_at_session_end"] = False
    return out


def _convert_v44_source_to_flat(raw_cfg: dict[str, Any], input_csv: str, out_json: str) -> dict[str, Any]:
    embedded = raw_cfg.get("config", raw_cfg) if isinstance(raw_cfg, dict) else {}
    return merged_engine._convert_v44_embedded_to_flat(embedded, input_csv, out_json)


def _override_v44_latent(flat_cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(flat_cfg)
    out["ny_start"] = 0.0
    # The V44 engine does not behave correctly with ny_end=24.0 in all-day mode.
    # Keep the NY session effectively all-day while staying inside its expected range.
    out["ny_end"] = 23.95
    out["v5_ny_start_delay_minutes"] = 0
    out["v5_session_entry_cutoff_minutes"] = 0
    out["v5_sessions"] = "ny_only"
    return out


def _load_classified_bars(input_csv: str) -> pd.DataFrame:
    m1 = regime_validation._load_m1(input_csv)
    featured = regime_validation.compute_features(m1)
    classified = regime_validation.classify_all_bars(featured, RegimeThresholds())
    return classified


def _lookup_bar_state(classified: pd.DataFrame, ts: pd.Timestamp) -> dict[str, Any]:
    time_idx = pd.DatetimeIndex(classified["time"])
    idx = time_idx.get_indexer([pd.Timestamp(ts)], method="ffill")[0]
    if idx < 0:
        idx = time_idx.get_indexer([pd.Timestamp(ts)], method="bfill")[0]
    if idx < 0:
        return {
            "regime_label": "ambiguous",
            "regime_margin": 0.0,
            "regime_scores": {},
        }
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


def _compute_v14_raw_quality(trade: merged_engine.TradeRow) -> float:
    raw = trade.raw or {}
    confluence = _safe_float(raw.get("confluence_score", raw.get("confluence", 0.0)), 0.0)
    strength = _safe_float(raw.get("signal_strength_score", 0.0), 0.0)
    quality_label = str(raw.get("quality_label", "")).lower()
    label_bonus = {"low": 0.0, "medium": 0.5, "high": 1.0}.get(quality_label, 0.0)
    return confluence * 10.0 + strength + label_bonus


def _compute_london_raw_quality(trade: merged_engine.TradeRow) -> float:
    raw = trade.raw or {}
    setup = str(raw.get("setup_type", "")).upper()
    risk_pct = _safe_float(raw.get("risk_pct", 0.0), 0.0)
    asian_range = _safe_float(raw.get("asian_range_pips", 0.0), 0.0)
    lor_range = _safe_float(raw.get("lor_range_pips", 0.0), 0.0)
    is_reentry = bool(raw.get("is_reentry", False))
    setup_base = {"A": 3.2, "D": 3.0, "C": 2.4, "B": 2.2}.get(setup, 2.0)
    asian_bonus = max(0.0, 1.5 - abs(asian_range - 45.0) / 15.0) if asian_range > 0 else 0.0
    lor_bonus = max(0.0, 1.2 - abs(lor_range - 10.0) / 8.0) if lor_range > 0 else 0.0
    reentry_penalty = 0.5 if is_reentry else 0.0
    return setup_base * 10.0 + risk_pct * 100.0 + asian_bonus + lor_bonus - reentry_penalty


def _compute_v44_raw_quality(trade: merged_engine.TradeRow) -> float:
    raw = trade.raw or {}
    profile = str(raw.get("entry_profile", "")).lower()
    signal_mode = str(raw.get("entry_signal_mode", "")).lower()
    profile_score = {
        "strong": 85.0,
        "normal": 65.0,
        "weak": 45.0,
        "news_trend": 95.0,
        "news_yield": 78.0,
        "bb_reversion": 52.0,
        "ema_cross": 54.0,
    }.get(profile, 55.0)
    mode_bonus = {
        "news_trend_confirm": 10.0,
        "breakout": 6.0,
        "orb_breakout": 5.0,
        "pullback": 4.0,
        "v5_pullback_bounce": 5.0,
        "range_fade": -2.0,
    }.get(signal_mode, 0.0)
    return profile_score + mode_bonus


def _compute_raw_quality(trade: merged_engine.TradeRow) -> float:
    if trade.strategy == "v14":
        return _compute_v14_raw_quality(trade)
    if trade.strategy == "london_v2":
        return _compute_london_raw_quality(trade)
    return _compute_v44_raw_quality(trade)


def _build_candidates(
    trades: list[merged_engine.TradeRow],
    classified: pd.DataFrame,
    effective_cfg: dict[str, Any],
) -> list[RouterCandidate]:
    candidates: list[RouterCandidate] = []
    for trade in trades:
        bar_state = _lookup_bar_state(classified, trade.entry_time)
        tp_pips = _safe_float((trade.raw or {}).get("tp2_pips", (trade.raw or {}).get("partial_tp_pips", 0.0)), 0.0)
        session_owner = classify_session(pd.Timestamp(trade.entry_time).to_pydatetime(), effective_cfg) == trade.entry_session
        cand = RouterCandidate(
            strategy=str(trade.strategy),
            side=str(trade.side),
            entry_time=pd.Timestamp(trade.entry_time),
            exit_time=pd.Timestamp(trade.exit_time),
            entry_price=_safe_float((trade.raw or {}).get("entry_price", 0.0), 0.0),
            sl_pips=_safe_float((trade.raw or {}).get("sl_pips", (trade.raw or {}).get("initial_sl_pips", 0.0)), 0.0),
            tp_pips=tp_pips,
            raw_quality=_compute_raw_quality(trade),
            normalized_quality=None,
            regime_compatible=None,
            source_session_owner=bool(session_owner),
            metadata={
                "entry_session": trade.entry_session,
                "exit_reason": trade.exit_reason,
                "regime_label": bar_state["regime_label"],
                "regime_margin": bar_state["regime_margin"],
                "regime_scores": bar_state["regime_scores"],
                "raw": trade.raw,
            },
            trade=trade,
        )
        candidates.append(cand)
    return candidates


def _normalize_candidates(candidates: list[RouterCandidate]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    by_strategy: dict[str, list[RouterCandidate]] = defaultdict(list)
    for c in candidates:
        by_strategy[c.strategy].append(c)
    for strategy, group in by_strategy.items():
        values = np.array([c.raw_quality for c in group], dtype=float)
        order = values.argsort(kind="mergesort")
        ranks = np.empty(len(values), dtype=float)
        ranks[order] = np.arange(1, len(values) + 1, dtype=float)
        pct = ranks / max(1.0, float(len(values)))
        for c, p in zip(group, pct):
            c.normalized_quality = float(min(1.0, max(0.0, p)))
        summary[strategy] = {
            "count": len(group),
            "raw_quality_min": float(values.min()) if len(values) else 0.0,
            "raw_quality_p50": float(np.median(values)) if len(values) else 0.0,
            "raw_quality_p90": float(np.quantile(values, 0.90)) if len(values) else 0.0,
            "raw_quality_max": float(values.max()) if len(values) else 0.0,
        }
    return summary


def _parse_hhmm_to_minutes(value: str) -> int:
    hh_s, mm_s = [x.strip() for x in str(value).split(":", 1)]
    hh = max(0, min(23, int(hh_s)))
    mm = max(0, min(59, int(mm_s)))
    return hh * 60 + mm


def _in_fixed_utc_window(ts: pd.Timestamp, start_hhmm: str, end_hhmm: str) -> bool:
    ts_utc = pd.Timestamp(ts).tz_convert("UTC") if pd.Timestamp(ts).tzinfo is not None else pd.Timestamp(ts).tz_localize("UTC")
    minute_of_day = int(ts_utc.hour) * 60 + int(ts_utc.minute)
    start_m = _parse_hhmm_to_minutes(start_hhmm)
    end_m = _parse_hhmm_to_minutes(end_hhmm)
    if start_m <= end_m:
        return start_m <= minute_of_day <= end_m
    return minute_of_day >= start_m or minute_of_day <= end_m


def _filter_candidates_by_window(candidates: list[RouterCandidate], start_hhmm: str, end_hhmm: str) -> list[RouterCandidate]:
    return [c for c in candidates if _in_fixed_utc_window(c.entry_time, start_hhmm, end_hhmm)]


def _copy_trade(trade: merged_engine.TradeRow) -> merged_engine.TradeRow:
    return merged_engine.TradeRow(**{**trade.__dict__})


def _global_max_open(effective_cfg: dict[str, Any]) -> int:
    v14_max = _safe_int((effective_cfg.get("v14", {}) or {}).get("max_concurrent_positions", 3), 3)
    london_max = _safe_int((effective_cfg.get("london_v2", {}) or {}).get("max_open_positions", 2), 2)
    v44_max = _safe_int((effective_cfg.get("v44_ny", {}) or {}).get("max_open", 3), 3)
    return max(1, max(v14_max, london_max, v44_max))


def _is_v44_blocked(regime_label: str, candidate: RouterCandidate) -> tuple[bool, str | None]:
    if candidate.strategy != "v44_ny":
        return False, None
    if regime_label == "breakout":
        return True, "blocked_v44_breakout"
    if regime_label == "post_breakout_trend":
        return True, "blocked_v44_post_breakout_trend"
    return False, None


def _authorize_candidates(
    candidates: list[RouterCandidate],
    *,
    regime_bonus: float,
    quality_margin_epsilon: float,
    high_confidence_regime_margin: float,
    apply_v44_preference: bool,
    effective_cfg: dict[str, Any],
) -> dict[str, Any]:
    by_bar: dict[pd.Timestamp, list[RouterCandidate]] = defaultdict(list)
    for c in candidates:
        by_bar[pd.Timestamp(c.entry_time).floor("min")].append(c)

    max_open_positions = _global_max_open(effective_cfg)
    selected: list[merged_engine.TradeRow] = []
    log: list[dict[str, Any]] = []
    open_trades: list[merged_engine.TradeRow] = []

    for bar_time in sorted(by_bar):
        group = sorted(by_bar[bar_time], key=lambda c: (c.strategy, c.side))

        still_open: list[merged_engine.TradeRow] = []
        for ot in open_trades:
            if pd.Timestamp(ot.exit_time) > bar_time:
                still_open.append(ot)
        open_trades = still_open

        blocked: list[dict[str, Any]] = []
        survivors: list[RouterCandidate] = []
        regime_label = str(group[0].metadata.get("regime_label", "ambiguous")) if group else "ambiguous"
        regime_margin = _safe_float(group[0].metadata.get("regime_margin", 0.0), 0.0) if group else 0.0

        for c in group:
            is_blocked, reason = _is_v44_blocked(regime_label, c)
            c.regime_compatible = not is_blocked
            if is_blocked:
                blocked.append({"strategy": c.strategy, "side": c.side, "reason": reason})
            else:
                survivors.append(c)

        winner: RouterCandidate | None = None
        decision_reason = "no_candidates"

        if len(open_trades) >= max_open_positions:
            decision_reason = f"global_max_open_{len(open_trades)}/{max_open_positions}"
            survivors = []

        if survivors:
            if len(survivors) == 1:
                winner = survivors[0]
                decision_reason = "single_survivor"
            else:
                scored: list[tuple[float, RouterCandidate]] = []
                for c in survivors:
                    adj = float(c.normalized_quality or 0.0)
                    if (
                        apply_v44_preference
                        and regime_label == "momentum"
                        and regime_margin >= high_confidence_regime_margin
                        and c.strategy == "v44_ny"
                    ):
                        adj += regime_bonus
                    scored.append((adj, c))
                scored.sort(key=lambda x: x[0], reverse=True)
                if len(scored) == 1:
                    winner = scored[0][1]
                    decision_reason = "normalized_quality_single"
                else:
                    top_adj, top_c = scored[0]
                    second_adj = scored[1][0]
                    if (top_adj - second_adj) < quality_margin_epsilon:
                        decision_reason = "quality_tie_no_trade"
                    else:
                        winner = top_c
                        if winner.strategy == "v44_ny" and regime_label == "momentum" and regime_margin >= high_confidence_regime_margin:
                            decision_reason = "v44_momentum_preferred"
                        else:
                            decision_reason = "normalized_quality_winner"

        if winner is not None and winner.trade is not None:
            chosen = _copy_trade(winner.trade)
            selected.append(chosen)
            open_trades.append(chosen)

        log.append(
            {
                "bar_time": bar_time.isoformat(),
                "regime_label": regime_label,
                "regime_margin": float(regime_margin),
                "blocked": blocked,
                "winner_strategy": winner.strategy if winner else None,
                "winner_side": winner.side if winner else None,
                "winner_normalized_quality": float(winner.normalized_quality or 0.0) if winner else None,
                "reason": decision_reason,
                "candidate_count": len(group),
                "survivor_count": len(survivors),
                "candidates": [
                    {
                        "strategy": c.strategy,
                        "side": c.side,
                        "normalized_quality": float(c.normalized_quality or 0.0),
                        "raw_quality": float(c.raw_quality),
                        "source_session_owner": bool(c.source_session_owner),
                    }
                    for c in group
                ],
            }
        )

    return {"selected_trades": selected, "arbitration_log": log, "global_max_open_positions": max_open_positions}


def _authorize_quality_only(
    candidates: list[RouterCandidate],
    *,
    quality_margin_epsilon: float,
    effective_cfg: dict[str, Any],
) -> dict[str, Any]:
    by_bar: dict[pd.Timestamp, list[RouterCandidate]] = defaultdict(list)
    for c in candidates:
        by_bar[pd.Timestamp(c.entry_time).floor("min")].append(c)

    max_open_positions = _global_max_open(effective_cfg)
    selected: list[merged_engine.TradeRow] = []
    log: list[dict[str, Any]] = []
    open_trades: list[merged_engine.TradeRow] = []

    for bar_time in sorted(by_bar):
        group = sorted(by_bar[bar_time], key=lambda c: (-(float(c.normalized_quality or 0.0)), c.strategy, c.side))
        open_trades = [ot for ot in open_trades if pd.Timestamp(ot.exit_time) > bar_time]

        winner: RouterCandidate | None = None
        reason = "no_candidates"
        if len(open_trades) >= max_open_positions:
            reason = f"global_max_open_{len(open_trades)}/{max_open_positions}"
        elif len(group) == 1:
            winner = group[0]
            reason = "single_candidate"
        elif len(group) > 1:
            top = float(group[0].normalized_quality or 0.0)
            second = float(group[1].normalized_quality or 0.0)
            if (top - second) < quality_margin_epsilon:
                reason = "quality_tie_no_trade"
            else:
                winner = group[0]
                reason = "normalized_quality_winner"

        if winner is not None and winner.trade is not None:
            chosen = _copy_trade(winner.trade)
            selected.append(chosen)
            open_trades.append(chosen)

        log.append(
            {
                "bar_time": bar_time.isoformat(),
                "winner_strategy": winner.strategy if winner else None,
                "winner_side": winner.side if winner else None,
                "winner_normalized_quality": float(winner.normalized_quality or 0.0) if winner else None,
                "reason": reason,
                "candidate_count": len(group),
                "candidates": [
                    {
                        "strategy": c.strategy,
                        "side": c.side,
                        "normalized_quality": float(c.normalized_quality or 0.0),
                        "raw_quality": float(c.raw_quality),
                        "source_session_owner": bool(c.source_session_owner),
                    }
                    for c in group
                ],
            }
        )

    return {"selected_trades": selected, "arbitration_log": log, "global_max_open_positions": max_open_positions}


def _build_baseline_report(
    input_csv: str,
    *,
    v14_config_path: Path,
    london_v2_config_path: Path,
    v44_config_path: Path,
    starting_equity: float,
    window_start_hhmm: str | None = None,
    window_end_hhmm: str | None = None,
) -> dict[str, Any]:
    m1 = merged_engine._load_m1(input_csv)
    v14_report, v14_cfg = merged_engine._run_v14_in_process(v14_config_path, input_csv)
    v2_trades_df, v2_diag, v2_cfg = merged_engine._run_v2_in_process(london_v2_config_path, m1)
    v44_results, v44_embedded = merged_engine._run_v44_in_process(v44_config_path, input_csv)
    if isinstance(v44_embedded, dict) and "v5_account_size" in v44_embedded:
        v44_base_eq = _safe_float(v44_embedded.get("v5_account_size", starting_equity), starting_equity)
    else:
        v44_base_eq = _safe_float(v44_embedded.get("v5", {}).get("account_size", starting_equity), starting_equity)
    all_raw = sorted(
        merged_engine._extract_v14_trades(v14_report, default_entry_equity=float(starting_equity))
        + merged_engine._extract_v2_trades(v2_trades_df, default_entry_equity=float(starting_equity))
        + merged_engine._extract_v44_trades(v44_results, default_entry_equity=float(v44_base_eq)),
        key=lambda x: (x.exit_time, x.entry_time),
    )
    if window_start_hhmm and window_end_hhmm:
        all_raw = [
            t
            for t in all_raw
            if _in_fixed_utc_window(t.entry_time, window_start_hhmm, window_end_hhmm)
        ]
    v14_max_units = _safe_int(v14_cfg.get("position_sizing", {}).get("max_units", 500000), 500000)
    scaled = merged_engine._apply_shared_equity_coupling(all_raw, float(starting_equity), v14_max_units=v14_max_units)
    eq_curve = merged_engine._build_equity_curve(scaled, float(starting_equity))
    return {
        "summary": merged_engine._stats(scaled, float(starting_equity), eq_curve),
        "by_strategy": merged_engine._subset_breakdown(scaled, lambda t: t.strategy),
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
                "size_scale": float(t.size_scale),
            }
            for t in scaled
        ],
        "notes": {"v2_diag": v2_diag, "v2_active_days": v2_cfg.get("session", {}).get("active_days_utc", [])},
    }


def _run_v14_latent(v14_config_path: Path, input_csv: str) -> tuple[list[merged_engine.TradeRow], dict[str, Any]]:
    cfg = _override_v14_latent(_read_json(v14_config_path))
    run_cfg = {
        "label": "latent_router_v1",
        "input_csv": input_csv,
        "output_json": "",
        "output_trades_csv": "",
        "output_equity_csv": "",
    }
    report = merged_engine.v14_engine.run_one(cfg, run_cfg)
    trades = merged_engine._extract_v14_trades(report, default_entry_equity=100000.0)
    return trades, cfg


def _run_london_latent(london_v2_config_path: Path, input_csv: str) -> tuple[list[merged_engine.TradeRow], dict[str, Any], dict[str, Any]]:
    m1 = merged_engine._load_m1(input_csv)
    trades_df, diag, cfg = merged_engine._run_v2_in_process(london_v2_config_path, m1)
    trades = merged_engine._extract_v2_trades(trades_df, default_entry_equity=100000.0)
    return trades, cfg, diag


def _run_v44_latent(v44_config_path: Path, input_csv: str, starting_equity: float) -> tuple[list[merged_engine.TradeRow], dict[str, Any]]:
    raw = _read_json(v44_config_path)
    with tempfile.TemporaryDirectory(prefix="phase3_latent_v44_") as td:
        tmp_cfg = Path(td) / "v44_latent_flat.json"
        flat = _override_v44_latent(_convert_v44_source_to_flat(raw, input_csv, str(Path(td) / "v44_latent_report.json")))
        tmp_cfg.write_text(json.dumps(flat, indent=2), encoding="utf-8")
        args = merged_engine.v44_engine.parse_args(["--config", str(tmp_cfg)])
        results = merged_engine.v44_engine.run_backtest_v5(args)
    embedded = raw.get("config", raw) if isinstance(raw, dict) else {}
    if isinstance(embedded, dict) and "v5_account_size" in embedded:
        base_eq = _safe_float(embedded.get("v5_account_size", starting_equity), starting_equity)
    else:
        base_eq = _safe_float(embedded.get("v5", {}).get("account_size", starting_equity), starting_equity)
    trades = merged_engine._extract_v44_trades(results, default_entry_equity=float(base_eq))
    return trades, embedded


def _count_by(items: list[RouterCandidate], key_fn) -> list[dict[str, Any]]:
    counts: dict[str, int] = defaultdict(int)
    for item in items:
        counts[str(key_fn(item))] += 1
    return [{"key": k, "count": counts[k]} for k in sorted(counts)]


def _bars_with_candidate_count(candidates: list[RouterCandidate], classified: pd.DataFrame) -> dict[str, Any]:
    total_bars = int(len(classified))
    per_bar: dict[pd.Timestamp, set[str]] = defaultdict(set)
    for c in candidates:
        per_bar[pd.Timestamp(c.entry_time).floor("min")].add(c.strategy)
    counts = Counter(len(v) for v in per_bar.values())
    out = {}
    for n in range(0, 4):
        bars = counts.get(n, 0) if n > 0 else total_bars - sum(counts.values())
        out[str(n)] = {"bars": int(bars), "pct": round(100.0 * bars / max(1, total_bars), 2)}
    return out


def _daily_large_move_no_trade(input_csv: str, selected_trades: list[merged_engine.TradeRow], threshold_pips: float = 100.0) -> dict[str, Any]:
    m1 = merged_engine._load_m1(input_csv)
    m1["date"] = m1["time"].dt.strftime("%Y-%m-%d")
    daily = (
        m1.groupby("date", as_index=False)
        .agg(high=("high", "max"), low=("low", "min"))
    )
    daily["range_pips"] = (daily["high"] - daily["low"]) / 0.01
    trade_days = Counter(pd.Timestamp(t.entry_time).strftime("%Y-%m-%d") for t in selected_trades)
    large = daily[daily["range_pips"] >= threshold_pips].copy()
    large["trade_count"] = large["date"].map(lambda d: int(trade_days.get(d, 0)))
    large["no_trade"] = large["trade_count"] == 0
    return {
        "threshold_pips": float(threshold_pips),
        "large_move_days": int(len(large)),
        "large_move_no_trade_days": int(large["no_trade"].sum()) if not large.empty else 0,
        "sample_days": large.sort_values("range_pips", ascending=False).head(20).to_dict("records"),
    }


def _report_for_selected(
    name: str,
    selected: list[merged_engine.TradeRow],
    *,
    starting_equity: float,
    v14_max_units: int,
    arbitration_log: list[dict[str, Any]],
    baseline_report: dict[str, Any],
    input_csv: str,
) -> dict[str, Any]:
    scaled = merged_engine._apply_shared_equity_coupling(sorted(selected, key=lambda t: (t.exit_time, t.entry_time)), starting_equity, v14_max_units=v14_max_units)
    eq_curve = merged_engine._build_equity_curve(scaled, starting_equity)
    summary = merged_engine._stats(scaled, starting_equity, eq_curve)
    by_strategy = merged_engine._subset_breakdown(scaled, lambda t: t.strategy)
    by_session = merged_engine._subset_breakdown(scaled, lambda t: t.entry_session)
    by_regime: dict[str, int] = Counter()
    for row in arbitration_log:
        if row.get("winner_strategy"):
            by_regime[str(row.get("regime_label", "ambiguous"))] += 1

    baseline_keys = {
        (r["strategy"], r["entry_time"], r["side"])
        for r in baseline_report.get("closed_trades", [])
    }
    selected_rows = [
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
        for t in scaled
    ]
    selected_keys = {(r["strategy"], r["entry_time"], r["side"]) for r in selected_rows}
    removed = [r for r in baseline_report.get("closed_trades", []) if (r["strategy"], r["entry_time"], r["side"]) not in selected_keys][:50]
    added = [r for r in selected_rows if (r["strategy"], r["entry_time"], r["side"]) not in baseline_keys][:50]

    return {
        "name": name,
        "summary": summary,
        "delta_vs_baseline": {
            "net_usd": round(summary["net_usd"] - _safe_float(baseline_report["summary"]["net_usd"]), 2),
            "profit_factor": round(summary["profit_factor"] - _safe_float(baseline_report["summary"]["profit_factor"]), 4),
            "max_drawdown_usd": round(summary["max_drawdown_usd"] - _safe_float(baseline_report["summary"]["max_drawdown_usd"]), 2),
            "total_trades": int(summary["total_trades"] - _safe_int(baseline_report["summary"]["total_trades"], 0)),
        },
        "by_strategy": by_strategy,
        "by_session": by_session,
        "by_day_of_week": merged_engine._subset_breakdown(
            scaled,
            lambda t: pd.Timestamp(t.entry_time).tz_convert("UTC").day_name(),
        ),
        "by_hour_utc": merged_engine._subset_breakdown(
            scaled,
            lambda t: pd.Timestamp(t.entry_time).tz_convert("UTC").strftime("%H"),
        ),
        "by_month": merged_engine._group_monthly(scaled),
        "authorized_by_regime": [{"regime": k, "trades": int(v)} for k, v in sorted(by_regime.items())],
        "replaced_vs_baseline": {
            "baseline_removed_sample": removed,
            "router_added_sample": added,
        },
        "large_move_no_trade": _daily_large_move_no_trade(input_csv, scaled),
        "closed_trades": selected_rows,
        "arbitration_log": arbitration_log,
    }


def _candidate_census_report(candidates: list[RouterCandidate], classified: pd.DataFrame) -> dict[str, Any]:
    return {
        "summary": {
            "total_candidates": int(len(candidates)),
            "bars_with_candidate_count": _bars_with_candidate_count(candidates, classified),
        },
        "by_strategy": _count_by(candidates, lambda c: c.strategy),
        "by_hour_utc": _count_by(candidates, lambda c: pd.Timestamp(c.entry_time).hour),
        "by_side": _count_by(candidates, lambda c: c.side),
        "by_source_session_owner": _count_by(candidates, lambda c: "home_session" if c.source_session_owner else "cross_session"),
        "by_strategy_cross_session": [
            {
                "strategy": strategy,
                "home_session": int(sum(1 for c in candidates if c.strategy == strategy and c.source_session_owner)),
                "cross_session": int(sum(1 for c in candidates if c.strategy == strategy and not c.source_session_owner)),
            }
            for strategy in sorted({c.strategy for c in candidates})
        ],
        "quality_distribution": {
            strategy: {
                "count": int(len(vals)),
                "raw_quality_p50": float(np.median(vals)) if vals else 0.0,
                "raw_quality_p90": float(np.quantile(vals, 0.90)) if len(vals) > 1 else (float(vals[0]) if vals else 0.0),
                "normalized_quality_p50": float(np.median(nvals)) if nvals else 0.0,
                "normalized_quality_p90": float(np.quantile(nvals, 0.90)) if len(nvals) > 1 else (float(nvals[0]) if nvals else 0.0),
            }
            for strategy in sorted({c.strategy for c in candidates})
            for vals, nvals in [(
                [c.raw_quality for c in candidates if c.strategy == strategy],
                [float(c.normalized_quality or 0.0) for c in candidates if c.strategy == strategy],
            )]
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 3 full-latent regime router backtest")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output-prefix", required=True)
    p.add_argument("--v14-config", default=str(OUT_DIR / "tokyo_optimized_v14_config.json"))
    p.add_argument("--london-v2-config", default=str(OUT_DIR / "v2_exp4_winner_baseline_config.json"))
    p.add_argument("--v44-config", default=str(OUT_DIR / "session_momentum_v44_base_config.json"))
    p.add_argument("--starting-equity", type=float, default=100000.0)
    p.add_argument("--regime-bonus", type=float, default=0.10)
    p.add_argument("--quality-margin-epsilon", type=float, default=0.03)
    p.add_argument("--high-confidence-regime-margin", type=float, default=1.0)
    p.add_argument("--router-mode", choices=["regime_v1", "quality_only"], default="regime_v1")
    p.add_argument("--window-start-utc", default="07:55")
    p.add_argument("--window-end-utc", default="22:00")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    dataset = str(Path(args.dataset).resolve())
    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    effective_cfg = load_phase3_sizing_config()
    classified = _load_classified_bars(dataset)

    baseline = _build_baseline_report(
        dataset,
        v14_config_path=Path(args.v14_config),
        london_v2_config_path=Path(args.london_v2_config),
        v44_config_path=Path(args.v44_config),
        starting_equity=float(args.starting_equity),
        window_start_hhmm=str(args.window_start_utc),
        window_end_hhmm=str(args.window_end_utc),
    )

    v14_trades, _ = _run_v14_latent(Path(args.v14_config), dataset)
    london_trades, _, london_diag = _run_london_latent(Path(args.london_v2_config), dataset)
    v44_trades, _ = _run_v44_latent(Path(args.v44_config), dataset, float(args.starting_equity))

    candidates = _build_candidates(v14_trades + london_trades + v44_trades, classified, effective_cfg)
    candidates = _filter_candidates_by_window(candidates, str(args.window_start_utc), str(args.window_end_utc))
    normalizer_summary = _normalize_candidates(candidates)

    v14_max_units = _safe_int((effective_cfg.get("v14", {}) or {}).get("max_units", 500000), 500000)

    if str(args.router_mode) == "quality_only":
        exp2 = _authorize_quality_only(
            candidates,
            quality_margin_epsilon=float(args.quality_margin_epsilon),
            effective_cfg=effective_cfg,
        )
        exp3 = exp2
    else:
        exp2 = _authorize_candidates(
            candidates,
            regime_bonus=float(args.regime_bonus),
            quality_margin_epsilon=float(args.quality_margin_epsilon),
            high_confidence_regime_margin=float(args.high_confidence_regime_margin),
            apply_v44_preference=False,
            effective_cfg=effective_cfg,
        )
        exp3 = _authorize_candidates(
            candidates,
            regime_bonus=float(args.regime_bonus),
            quality_margin_epsilon=float(args.quality_margin_epsilon),
            high_confidence_regime_margin=float(args.high_confidence_regime_margin),
            apply_v44_preference=True,
            effective_cfg=effective_cfg,
        )

    census_report = {
        "dataset": Path(dataset).name,
        "notes": {
            "type": "latent_candidate_census",
            "router_mode": str(args.router_mode),
            "window_utc": {"start": str(args.window_start_utc), "end": str(args.window_end_utc)},
            "london_diagnostics": london_diag,
            "normalizer_summary": normalizer_summary,
        },
        **_candidate_census_report(candidates, classified),
    }
    exp2_report = _report_for_selected(
        "negative_gating_router",
        exp2["selected_trades"],
        starting_equity=float(args.starting_equity),
        v14_max_units=v14_max_units,
        arbitration_log=exp2["arbitration_log"],
        baseline_report=baseline,
        input_csv=dataset,
    )
    exp3_report = _report_for_selected(
        "full_latent_regime_priority_router",
        exp3["selected_trades"],
        starting_equity=float(args.starting_equity),
        v14_max_units=v14_max_units,
        arbitration_log=exp3["arbitration_log"],
        baseline_report=baseline,
        input_csv=dataset,
    )
    baseline_path = out_prefix.with_name(out_prefix.name + "_baseline.json")
    census_path = out_prefix.with_name(out_prefix.name + "_experiment1_census.json")
    exp2_path = out_prefix.with_name(out_prefix.name + "_experiment2_negative_gating.json")
    exp3_path = out_prefix.with_name(out_prefix.name + "_experiment3_full_router.json")
    index_path = out_prefix.with_name(out_prefix.name + "_index.json")

    baseline_path.write_text(json.dumps(baseline, indent=2, default=_json_default), encoding="utf-8")
    census_path.write_text(json.dumps(census_report, indent=2, default=_json_default), encoding="utf-8")
    exp2_path.write_text(json.dumps(exp2_report, indent=2, default=_json_default), encoding="utf-8")
    exp3_path.write_text(json.dumps(exp3_report, indent=2, default=_json_default), encoding="utf-8")
    index_path.write_text(
        json.dumps(
            {
                "dataset": Path(dataset).name,
                "baseline": str(baseline_path),
                "experiment1": str(census_path),
                "experiment2": str(exp2_path),
                "experiment3": str(exp3_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "dataset": Path(dataset).name,
                "baseline": baseline["summary"],
                "experiment2": exp2_report["summary"],
                "experiment3": exp3_report["summary"],
                "output_index": str(index_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
