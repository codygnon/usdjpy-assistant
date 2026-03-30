"""
Shadow invocation layer for strategy entry engines.

This module bridges from:
  chart authorization + shadow entry eligibility
to:
  whether the strategy's real native entry path would emit a candidate on
  the evaluated bar.

Current source modes:
  - ``v44_ny``: native-entry timestamp registry from the real standalone run
  - ``v14``: direct calls into ``core.v14_entry_evaluator``
  - ``london_v2``: direct calls into ``core.london_v2_entry_evaluator``

Important limitation:
  This is still a shadow layer. It does not execute trades, manage positions,
  or run coupling. V44 uses ``native_trade_entry_registry`` (standalone run
  trade times) until a shared per-bar evaluator exists; see
  ``scripts/diagnostic_v44_shadow_registry_parity.py`` and
  ``research_out/track3_v44_shadow_registry_parity.json`` for the formal
  protocol and integrity checks. V14/London use extracted evaluators and are
  per-bar invocable in-process.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.london_v2_entry_evaluator import evaluate_london_v2_entry_signal
from core.v14_entry_evaluator import evaluate_v14_entry_signal
from scripts import backtest_tokyo_meanrev as v14_engine
from scripts import backtest_v2_multisetup_london as london_v2_engine


CACHE_VERSION = "mixed_v2"


@dataclass(frozen=True)
class InvocationCandidate:
    strategy: str
    bar_time: str
    entry_time: str | None
    side: str | None
    trigger_type: str | None
    raw: dict[str, Any]
    exit_time: str | None = None
    pips: float | None = None
    usd: float | None = None
    exit_reason: str | None = None


@dataclass(frozen=True)
class InvocationDecision:
    strategy: str
    attempted: bool
    invoked: bool
    candidate_found: bool
    reason: str
    candidate: InvocationCandidate | None = None

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        if self.candidate is not None:
            out["candidate"] = asdict(self.candidate)
        return out


@dataclass(frozen=True)
class InvocationRegistry:
    dataset: str
    counts_by_strategy: dict[str, int]
    source_mode_by_strategy: dict[str, str]
    entries_by_strategy: dict[str, dict[str, list[InvocationCandidate]]]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _cache_path_for(dataset: str) -> Path:
    stem = Path(dataset).stem
    cache_dir = Path(dataset).resolve().parent / "shadow_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"invocation_registry_{stem}_{CACHE_VERSION}.pkl"


def _trade_to_candidate(trade) -> InvocationCandidate:
    raw = trade.raw if isinstance(trade.raw, dict) else {}
    return InvocationCandidate(
        strategy=str(trade.strategy),
        bar_time=pd.Timestamp(trade.entry_time).isoformat(),
        entry_time=pd.Timestamp(trade.entry_time).isoformat(),
        side=str(trade.side),
        trigger_type=str(raw.get("setup_type") or raw.get("entry_reason") or "native_trade"),
        exit_time=pd.Timestamp(trade.exit_time).isoformat(),
        pips=float(trade.pips),
        usd=float(trade.usd),
        exit_reason=str(trade.exit_reason),
        raw=raw,
    )


def _extract_v14_cfg_params(cfg: dict[str, Any]) -> tuple[dict[str, Any], int]:
    PIP_SIZE = 0.01
    scoring_model = str(cfg.get("entry_rules", {}).get("scoring_model", "")).strip().lower()
    tokyo_v2_scoring = scoring_model in {"tokyo_v2", "v2", "tokyo_actual_v2"}
    tol_pips = float(cfg["entry_rules"]["long"].get("price_zone", {}).get("tolerance_pips", 10.0))
    combo_filter_cfg = cfg.get("confluence_combo_filter", {})
    ss_cfg = cfg.get("signal_strength_tracking", {})
    ss_filter_cfg = cfg.get("signal_strength_filter", {})
    regime_gate = cfg.get("regime_gate", {})
    cq = cfg.get("confluence_quality", {})
    return {
        "tokyo_v2_scoring": tokyo_v2_scoring,
        "confluence_min_long": int(cfg["entry_rules"]["long"].get("confluence_scoring", {}).get("minimum_score", 2)),
        "confluence_min_short": int(cfg["entry_rules"]["short"].get("confluence_scoring", {}).get("minimum_score", 2)),
        "long_rsi_soft_entry": float(cfg["entry_rules"]["long"].get("rsi_soft_filter", {}).get("entry_soft_threshold", 35.0)),
        "long_rsi_bonus": float(cfg["entry_rules"]["long"].get("rsi_soft_filter", {}).get("bonus_threshold", 30.0)),
        "short_rsi_soft_entry": float(cfg["entry_rules"]["short"].get("rsi_soft_filter", {}).get("entry_soft_threshold", 65.0)),
        "short_rsi_bonus": float(cfg["entry_rules"]["short"].get("rsi_soft_filter", {}).get("bonus_threshold", 70.0)),
        "tol": tol_pips * PIP_SIZE,
        "atr_max": float(cfg["indicators"]["atr"]["max_threshold_price_units"]),
        "core_gate_use_zone": bool(cfg.get("entry_rules", {}).get("core_gate", {}).get("use_zone", True)),
        "core_gate_use_bb": bool(cfg.get("entry_rules", {}).get("core_gate", {}).get("use_bb_touch", True)),
        "core_gate_use_sar": bool(cfg.get("entry_rules", {}).get("core_gate", {}).get("use_sar_flip", True)),
        "core_gate_use_rsi": bool(cfg.get("entry_rules", {}).get("core_gate", {}).get("use_rsi_soft", True)),
        "core_gate_required": int(cfg.get("entry_rules", {}).get("core_gate", {}).get("required_count", 4)),
        "regime_enabled": bool(regime_gate.get("enabled", False)),
        "atr_ratio_trend": float(regime_gate.get("atr_ratio_trending_threshold", 1.3)),
        "atr_ratio_calm": float(regime_gate.get("atr_ratio_calm_threshold", 0.8)),
        "adx_trend": float(regime_gate.get("adx_trending_threshold", 25.0)),
        "adx_range": float(regime_gate.get("adx_ranging_threshold", 20.0)),
        "favorable_min_score": int(regime_gate.get("favorable_min_score", 1)),
        "neutral_min_score": int(regime_gate.get("neutral_min_score", 0)),
        "neutral_size_mult": float(regime_gate.get("neutral_size_multiplier", 0.5)),
        "ss_enabled": bool(ss_cfg.get("enabled", False)),
        "ss_comp": ss_cfg.get("components", {}),
        "combo_filter_enabled": bool(combo_filter_cfg.get("enabled", False)),
        "combo_filter_mode": str(combo_filter_cfg.get("mode", "allowlist")).strip().lower(),
        "combo_allow": set(combo_filter_cfg.get("allowed_combos", [])),
        "combo_block": set(combo_filter_cfg.get("blocked_combos", [])),
        "ss_filter_enabled": bool(ss_filter_cfg.get("enabled", False)),
        "ss_filter_min_score": int(ss_filter_cfg.get("min_score", 0)),
        "cq_enabled": bool(cq.get("enabled", False)),
        "top_combos": set(cq.get("top_combos", [])),
        "bottom_combos": set(cq.get("bottom_combos", [])),
        "high_quality_mult": float(cq.get("high_quality_size_mult", 1.0)),
        "medium_quality_mult": float(cq.get("medium_quality_size_mult", 0.75)),
        "low_quality_skip": bool(cq.get("low_quality_skip", True)),
        "rejection_bonus_enabled": bool(cfg.get("rejection_bonus", {}).get("enabled", False)),
        "div_track_enabled": bool(cfg.get("rsi_divergence_tracking", {}).get("enabled", False)),
        "session_env_enabled": bool(cfg.get("session_envelope", {}).get("enabled", False)),
        "session_env_log_ir_pos": bool(cfg.get("session_envelope", {}).get("log_ir_position", True)),
    }, int(cfg.get("session_envelope", {}).get("warmup_minutes", 30))


def _update_v14_session_state(state_map: dict[str, dict[str, Any]], row: dict[str, Any], warmup_minutes: int) -> dict[str, Any]:
    sday = str(row["session_day_jst"])
    ts = pd.Timestamp(row["time"])
    mid_high = float(row["high"])
    mid_low = float(row["low"])
    if sday not in state_map:
        state_map[sday] = {
            "session_high": mid_high,
            "session_low": mid_low,
            "ir_ready": False,
            "ir_high": mid_high,
            "ir_low": mid_low,
            "warmup_end_ts": ts + pd.Timedelta(minutes=warmup_minutes),
        }
    sst = state_map[sday]
    sst["session_high"] = max(float(sst.get("session_high", mid_high)), mid_high)
    sst["session_low"] = min(float(sst.get("session_low", mid_low)), mid_low)
    if not bool(sst.get("ir_ready", False)):
        if ts <= pd.Timestamp(sst.get("warmup_end_ts")):
            sst["ir_high"] = max(float(sst.get("ir_high", mid_high)), mid_high)
            sst["ir_low"] = min(float(sst.get("ir_low", mid_low)), mid_low)
        else:
            sst["ir_ready"] = True
    return sst


def _build_v14_candidate_map(*, dataset: str, config_path: Path) -> tuple[dict[str, list[InvocationCandidate]], int]:
    cfg = _load_json(config_path)
    m1 = v14_engine.load_m1(dataset)
    df = v14_engine.add_indicators(m1, cfg)
    df["time_utc"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df["time_jst"] = df["time_utc"].dt.tz_convert("Asia/Tokyo")
    df["session_day_jst"] = df["time_jst"].dt.date.astype(str)
    df["utc_day_name"] = df["time_utc"].dt.day_name()
    df["hour_utc"] = df["time_utc"].dt.hour
    df["minute_utc"] = df["time_utc"].dt.minute
    df["minute_of_day_utc"] = df["hour_utc"] * 60 + df["minute_utc"]
    cfg_params, warmup_minutes = _extract_v14_cfg_params(cfg)

    session_cfg = cfg.get("session_filter", {})
    start_h, start_m = [int(x) for x in str(session_cfg.get("session_start_utc", "16:00")).split(":")]
    end_h, end_m = [int(x) for x in str(session_cfg.get("session_end_utc", "22:00")).split(":")]
    start_min = start_h * 60 + start_m
    end_min = end_h * 60 + end_m
    if start_min < end_min:
        in_tokyo = (df["minute_of_day_utc"] >= start_min) & (df["minute_of_day_utc"] < end_min)
    else:
        in_tokyo = (df["minute_of_day_utc"] >= start_min) | (df["minute_of_day_utc"] < end_min)
    allowed_days = set(session_cfg.get("allowed_trading_days", []))
    if allowed_days:
        df = df[in_tokyo & df["utc_day_name"].isin(allowed_days)].copy()
    else:
        df = df[in_tokyo].copy()

    by_ts: dict[str, list[InvocationCandidate]] = {}
    session_state: dict[str, dict[str, Any]] = {}
    required_cols = [
        "pivot_P", "pivot_R1", "pivot_R2", "pivot_R3", "pivot_S1", "pivot_S2", "pivot_S3",
        "bb_upper", "bb_lower", "bb_mid", "rsi_m5", "atr_m15",
    ]
    for row in df.itertuples(index=False, name="V14ShadowRow"):
        rowd = row._asdict()
        sst = _update_v14_session_state(session_state, rowd, warmup_minutes)
        if any(pd.isna(rowd.get(col)) for col in required_cols):
            continue
        candidate = evaluate_v14_entry_signal(
            row=rowd,
            mid_close=float(rowd["close"]),
            mid_open=float(rowd["open"]),
            mid_high=float(rowd["high"]),
            mid_low=float(rowd["low"]),
            pivot_levels={
                "P": float(rowd["pivot_P"]),
                "R1": float(rowd["pivot_R1"]),
                "R2": float(rowd["pivot_R2"]),
                "R3": float(rowd["pivot_R3"]),
                "S1": float(rowd["pivot_S1"]),
                "S2": float(rowd["pivot_S2"]),
                "S3": float(rowd["pivot_S3"]),
            },
            cfg_params=cfg_params,
            sst=sst,
        )
        if not candidate or candidate.get("_blocked_reason"):
            continue
        ts_key = pd.Timestamp(rowd["time"]).isoformat()
        by_ts.setdefault(ts_key, []).append(
            InvocationCandidate(
                strategy="v14",
                bar_time=ts_key,
                entry_time=ts_key,
                side="buy" if candidate.get("direction") == "long" else "sell",
                trigger_type="v14_confluence",
                raw=candidate,
            )
        )
    return by_ts, sum(len(v) for v in by_ts.values())


def _build_london_candidate_map(*, dataset: str, config_path: Path, merged_engine) -> tuple[dict[str, list[InvocationCandidate]], int]:
    cfg = _load_json(config_path)
    df = merged_engine._load_m1(dataset).copy()
    df["day_utc"] = df["time"].dt.floor("D")
    grouped = {k: g.copy().reset_index(drop=True) for k, g in df.groupby("day_utc")}
    asian_min = float(cfg["levels"]["asian_range_min_pips"])
    asian_max = float(cfg["levels"]["asian_range_max_pips"])
    lor_min = float(cfg["levels"]["lor_range_min_pips"])
    lor_max = float(cfg["levels"]["lor_range_max_pips"])

    by_ts: dict[str, list[InvocationCandidate]] = {}

    for day in sorted(grouped.keys()):
        day_df = grouped[day]
        # Keep the shadow registry aligned with the London Setup D research
        # path. The additive/state/outcome diagnostics intentionally scan all
        # London days and let downstream gating decide what survives, so
        # filtering here by config active_days would hide valid candidates and
        # break parity with the pilot reference artifacts.
        if day_df.empty:
            continue
        day_start = day
        london_h = london_v2_engine.uk_london_open_utc(day)
        ny_h = london_v2_engine.us_ny_open_utc(day)
        london_open = day_start + pd.Timedelta(hours=london_h)
        ny_open = day_start + pd.Timedelta(hours=ny_h)
        day_df = day_df[day_df["time"] < ny_open].copy().reset_index(drop=True)
        if len(day_df) < 2:
            continue

        asian = day_df[(day_df["time"] >= day_start) & (day_df["time"] < london_open)]
        if asian.empty:
            continue
        asian_high = float(asian["high"].max())
        asian_low = float(asian["low"].min())
        asian_range_pips = (asian_high - asian_low) / london_v2_engine.PIP_SIZE
        asian_valid = asian_min <= asian_range_pips <= asian_max

        lor_end = london_open + pd.Timedelta(minutes=15)
        lor = day_df[(day_df["time"] >= london_open) & (day_df["time"] < lor_end)]
        lor_high = float(lor["high"].max()) if not lor.empty else np.nan
        lor_low = float(lor["low"].min()) if not lor.empty else np.nan
        lor_range_pips = (lor_high - lor_low) / london_v2_engine.PIP_SIZE if not lor.empty else np.nan
        lor_valid = (not lor.empty) and (lor_min <= lor_range_pips <= lor_max)

        def _window_for_setup(setup_key: str) -> tuple[pd.Timestamp, pd.Timestamp]:
            s = cfg["setups"][setup_key]
            start = london_open + pd.Timedelta(minutes=int(s["entry_start_min_after_london"]))
            if s.get("entry_end_min_before_ny") is not None:
                end = ny_open - pd.Timedelta(minutes=int(s["entry_end_min_before_ny"]))
            else:
                end = london_open + pd.Timedelta(minutes=int(s["entry_end_min_after_london"]))
            if end > ny_open:
                end = ny_open
            return start, end

        windows = {
            "A": _window_for_setup("A"),
            "B": _window_for_setup("B"),
            "C": _window_for_setup("C"),
            "D": _window_for_setup("D"),
        }
        channels: dict[tuple[str, str], dict[str, Any]] = {
            (setup, direction): {"state": "ARMED", "entries": 0}
            for setup in ["A", "B", "C", "D"]
            for direction in ["long", "short"]
        }
        b_candidates = {"long": [], "short": []}

        rows = day_df.to_dict("records")
        for i in range(len(rows) - 1):
            row = rows[i]
            ts = pd.Timestamp(row["time"])
            nxt_ts = pd.Timestamp(rows[i + 1]["time"])
            entries, b_candidates = evaluate_london_v2_entry_signal(
                row=row,
                cfg=cfg,
                asian_high=asian_high,
                asian_low=asian_low,
                asian_range_pips=asian_range_pips,
                asian_valid=asian_valid,
                lor_high=lor_high,
                lor_low=lor_low,
                lor_range_pips=float(lor_range_pips) if np.isfinite(lor_range_pips) else 0.0,
                lor_valid=bool(lor_valid),
                ts=ts,
                nxt_ts=nxt_ts,
                bar_index=i,
                windows=windows,
                channels=channels,
                b_candidates=b_candidates,
            )
            if not entries:
                continue
            first = entries[0]
            ts_key = ts.isoformat()
            by_ts.setdefault(ts_key, []).append(
                InvocationCandidate(
                    strategy="london_v2",
                    bar_time=ts_key,
                    entry_time=pd.Timestamp(first.get("execute_time")).isoformat() if first.get("execute_time") is not None else None,
                    side="buy" if first.get("direction") == "long" else "sell",
                    trigger_type=f"setup_{first.get('setup_type')}",
                    raw=first,
                )
            )
    return by_ts, sum(len(v) for v in by_ts.values())


def build_native_invocation_registry(
    *,
    dataset: str,
    merged_engine,
    v14_config_path: Path,
    london_v2_config_path: Path,
    v44_config_path: Path,
    starting_equity: float = 100000.0,
) -> InvocationRegistry:
    """
    Build a mixed-mode shadow invocation registry.

    - V44 uses native standalone trade entries as the source of truth.
    - V14/London use extracted evaluators to build per-bar candidate maps.
    """
    cache_path = _cache_path_for(dataset)
    if cache_path.exists():
        with cache_path.open("rb") as fh:
            cached = pickle.load(fh)
        if isinstance(cached, InvocationRegistry):
            return cached

    v44_results, embedded = merged_engine._run_v44_in_process(v44_config_path, dataset)
    if isinstance(embedded, dict) and "v5_account_size" in embedded:
        v44_eq = float(embedded.get("v5_account_size", starting_equity))
    else:
        v44_eq = float((embedded or {}).get("v5", {}).get("account_size", starting_equity))
    v44_trades = merged_engine._extract_v44_trades(v44_results, default_entry_equity=v44_eq)
    v44_by_ts: dict[str, list[InvocationCandidate]] = {}
    for t in v44_trades:
        key = pd.Timestamp(t.entry_time).isoformat()
        v44_by_ts.setdefault(key, []).append(_trade_to_candidate(t))

    v14_by_ts, v14_count = _build_v14_candidate_map(dataset=dataset, config_path=v14_config_path)
    london_by_ts, london_count = _build_london_candidate_map(
        dataset=dataset,
        config_path=london_v2_config_path,
        merged_engine=merged_engine,
    )

    registry = InvocationRegistry(
        dataset=str(dataset),
        counts_by_strategy={
            "v14": v14_count,
            "london_v2": london_count,
            "v44_ny": len(v44_trades),
        },
        source_mode_by_strategy={
            "v14": "extracted_v14_evaluator",
            "london_v2": "extracted_london_v2_evaluator_armed_channels",
            "v44_ny": "native_trade_entry_registry",
        },
        entries_by_strategy={
            "v14": v14_by_ts,
            "london_v2": london_by_ts,
            "v44_ny": v44_by_ts,
        },
    )
    with cache_path.open("wb") as fh:
        pickle.dump(registry, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return registry


def shadow_invoke_authorized_bar(
    *,
    strategy: str | None,
    ts: pd.Timestamp,
    registry: InvocationRegistry,
) -> InvocationDecision:
    if not strategy:
        return InvocationDecision(
            strategy="none",
            attempted=False,
            invoked=False,
            candidate_found=False,
            reason="no_authorized_strategy",
        )

    strategy_entries = registry.entries_by_strategy.get(strategy)
    if strategy_entries is None:
        return InvocationDecision(
            strategy=strategy,
            attempted=True,
            invoked=False,
            candidate_found=False,
            reason="engine_registry_missing",
        )

    key = pd.Timestamp(ts).isoformat()
    matches = strategy_entries.get(key, [])
    if not matches:
        return InvocationDecision(
            strategy=strategy,
            attempted=True,
            invoked=True,
            candidate_found=False,
            reason="invoked_no_candidate",
        )

    return InvocationDecision(
        strategy=strategy,
        attempted=True,
        invoked=True,
        candidate_found=True,
        reason="candidate_found",
        candidate=matches[0],
    )


def registry_as_dict(registry: InvocationRegistry) -> dict[str, Any]:
    return {
        "dataset": registry.dataset,
        "counts_by_strategy": registry.counts_by_strategy,
        "source_mode_by_strategy": registry.source_mode_by_strategy,
        "entries_by_strategy": {
            k: {
                ts: [asdict(t) for t in trades]
                for ts, trades in by_ts.items()
            }
            for k, by_ts in registry.entries_by_strategy.items()
        },
    }
