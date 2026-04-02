from __future__ import annotations

import json
import math
from bisect import bisect_right, insort
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from core.london_v2_entry_evaluator import evaluate_london_v2_entry_signal
from core.ownership_table import cell_key, der_bucket, er_bucket
from core.regime_classifier import RegimeThresholds
from core.regime_features import compute_delta_efficiency_ratio, compute_efficiency_ratio
from core.v14_entry_evaluator import evaluate_v14_entry_signal
from scripts import backtest_tokyo_meanrev as tokyo_engine
from scripts import backtest_v2_multisetup_london as london_engine
from scripts import validate_regime_classifier as regime_validation

PIP_SIZE = 0.01
ROUND_UNITS = 100
ROOT = Path("/Users/codygnon/Documents/usdjpy_assistant")


@dataclass
class UnifiedExecutionPlan:
    sl_price: float
    tp1_price: Optional[float]
    tp2_price: Optional[float]
    tp1_close_fraction: float
    be_offset_pips: float
    trail_activate_pips: float
    trail_distance_pips: float
    trail_requires_tp1: bool
    session_close_time: Optional[pd.Timestamp]
    partial_enabled: bool = True
    stale_bars: Optional[int] = None
    stale_exit_underwater_pct: Optional[float] = None
    time_decay_minutes: Optional[float] = None
    time_decay_profit_cap_pips: Optional[float] = None
    trail_reference: str = "price_extreme"
    trail_ema_field: Optional[str] = None


@dataclass
class RegimeOwnedCandidate:
    strategy: str
    session_owner: str
    side: str
    signal_time: pd.Timestamp
    execute_time: pd.Timestamp
    entry_price_basis: str
    sl_pips: float
    raw_quality: float
    quality_normalized: float
    regime_label: str
    regime_margin: float
    ownership_cell: str
    reason: str
    source_features: dict[str, Any]
    execution_plan: UnifiedExecutionPlan
    risk_pct: float
    max_spread_pips: float
    source_family: str
    source_setup: Optional[str] = None


@dataclass
class RegimeOwnedDecision:
    bar_time: pd.Timestamp
    regime_label: str
    regime_margin: float
    ownership_cell: str
    candidate_count: int
    candidates: list[dict[str, Any]]
    winner_strategy: Optional[str]
    winner_side: Optional[str]
    winner_reason: str
    no_trade_reason: Optional[str] = None


@dataclass
class UnifiedPositionState:
    trade_id: int
    strategy: str
    source_family: str
    session_owner: str
    source_setup: Optional[str]
    side: str
    entry_time: pd.Timestamp
    entry_price: float
    initial_units: int
    remaining_units: int
    planned_risk_usd: float
    margin_required_usd: float
    execution_plan: UnifiedExecutionPlan
    raw: dict[str, Any] = field(default_factory=dict)
    tp1_hit: bool = False
    moved_to_be: bool = False
    trail_stop_price: Optional[float] = None
    realized_usd: float = 0.0
    weighted_pips_sum: float = 0.0
    mfe_pips: float = 0.0
    mae_pips: float = 0.0
    exit_reason: Optional[str] = None
    exit_time: Optional[pd.Timestamp] = None
    exit_price_last: Optional[float] = None
    bars_open: int = 0


@dataclass
class PortfolioBacktestState:
    equity: float
    peak_equity: float
    trade_id_seq: int = 0
    open_positions: list[UnifiedPositionState] = field(default_factory=list)
    closed_trades: list[dict[str, Any]] = field(default_factory=list)
    equity_curve: list[dict[str, Any]] = field(default_factory=list)
    decision_log: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    pending_orders: list[RegimeOwnedCandidate] = field(default_factory=list)
    strategy_quality_history: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    strategy_quality_sorted: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    tokyo_session_state: dict[str, Any] = field(default_factory=dict)
    london_day_state: dict[str, Any] = field(default_factory=dict)
    v44_session_state: dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestBarRecord:
    time: pd.Timestamp
    regime_label: str
    regime_margin: float
    ownership_cell: str
    candidate_count: int
    winner_strategy: Optional[str]
    winner_side: Optional[str]
    no_trade_reason: Optional[str]


@dataclass
class BacktestResult:
    summary: dict[str, Any]
    closed_trades: pd.DataFrame
    equity_curve: pd.DataFrame
    decision_log: pd.DataFrame
    diagnostics: dict[str, Any]


class IndexedRow:
    __slots__ = ("values", "index_map")

    def __init__(self, values: Any, index_map: dict[str, int]) -> None:
        self.values = values
        self.index_map = index_map

    def __getitem__(self, key: str) -> Any:
        return self.values[self.index_map[key]]

    def get(self, key: str, default: Any = None) -> Any:
        idx = self.index_map.get(key)
        if idx is None:
            return default
        val = self.values[idx]
        return default if val is None else val


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return str(obj)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


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


def _pip_value_per_unit(price: float) -> float:
    return PIP_SIZE / max(1e-9, float(price))


def _to_bid_ask(mid: float, spread_pips: float) -> tuple[float, float]:
    half = float(spread_pips) * PIP_SIZE / 2.0
    return float(mid) - half, float(mid) + half


def _calc_leg_pnl(direction: str, entry: float, exit_px: float, units: int) -> tuple[float, float]:
    if direction == "buy":
        pips = (float(exit_px) - float(entry)) / PIP_SIZE
    else:
        pips = (float(entry) - float(exit_px)) / PIP_SIZE
    usd = pips * int(units) * _pip_value_per_unit(exit_px)
    return float(pips), float(usd)


def _compute_max_drawdown(equity_curve: pd.DataFrame, start_equity: float) -> tuple[float, float]:
    if equity_curve.empty:
        return 0.0, 0.0
    peak = start_equity
    max_dd = 0.0
    for _, row in equity_curve.iterrows():
        eq = float(row["equity"])
        peak = max(peak, eq)
        max_dd = max(max_dd, peak - eq)
    return float(max_dd), float((max_dd / peak * 100.0) if peak > 0 else 0.0)


def _profit_factor(pnls: list[float]) -> float:
    wins = sum(x for x in pnls if x > 0)
    losses = sum(-x for x in pnls if x < 0)
    if losses <= 0:
        return 999.0 if wins > 0 else 0.0
    return float(wins / losses)


def _build_summary(
    *,
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    decision_df: pd.DataFrame,
    diagnostics: dict[str, Any],
    start_equity: float,
) -> dict[str, Any]:
    pnls = trades_df["pnl_usd"].tolist() if not trades_df.empty else []
    dd_usd, dd_pct = _compute_max_drawdown(equity_df, start_equity)
    wins = trades_df[trades_df["pnl_usd"] > 0] if not trades_df.empty else pd.DataFrame()
    losses = trades_df[trades_df["pnl_usd"] <= 0] if not trades_df.empty else pd.DataFrame()
    per_strategy = []
    if not trades_df.empty:
        for key, g in trades_df.groupby("strategy"):
            per_strategy.append(
                {
                    "strategy": str(key),
                    "trades": int(len(g)),
                    "net_usd": float(g["pnl_usd"].sum()),
                    "profit_factor": float(_profit_factor(g["pnl_usd"].tolist())),
                    "win_rate_pct": float((g["pnl_usd"] > 0).mean() * 100.0),
                }
            )
    per_regime = []
    if not trades_df.empty and "regime_label" in trades_df.columns:
        for key, g in trades_df.groupby("regime_label"):
            per_regime.append(
                {
                    "regime_label": str(key),
                    "trades": int(len(g)),
                    "net_usd": float(g["pnl_usd"].sum()),
                    "win_rate_pct": float((g["pnl_usd"] > 0).mean() * 100.0),
                }
            )
    by_hour = []
    if not trades_df.empty:
        tmp = trades_df.copy()
        tmp["entry_hour_utc"] = pd.to_datetime(tmp["entry_time_utc"], utc=True).dt.hour
        for key, g in tmp.groupby("entry_hour_utc"):
            by_hour.append({"hour_utc": int(key), "trades": int(len(g)), "net_usd": float(g["pnl_usd"].sum())})
    by_day = []
    if not trades_df.empty:
        tmp = trades_df.copy()
        tmp["entry_day"] = pd.to_datetime(tmp["entry_time_utc"], utc=True).dt.day_name()
        for key, g in tmp.groupby("entry_day"):
            by_day.append({"day": str(key), "trades": int(len(g)), "net_usd": float(g["pnl_usd"].sum())})
    no_trade_reasons = Counter()
    if not decision_df.empty and "no_trade_reason" in decision_df.columns:
        for val in decision_df["no_trade_reason"].dropna():
            no_trade_reasons[str(val)] += 1
    return {
        "starting_equity_usd": float(start_equity),
        "ending_equity_usd": float(equity_df.iloc[-1]["equity"]) if not equity_df.empty else float(start_equity),
        "net_usd": float(sum(pnls)),
        "profit_factor": float(_profit_factor(pnls)),
        "max_drawdown_usd": float(dd_usd),
        "max_drawdown_pct": float(dd_pct),
        "total_trades": int(len(trades_df)),
        "win_rate_pct": float((trades_df["pnl_usd"] > 0).mean() * 100.0) if not trades_df.empty else 0.0,
        "expectancy_usd": float(trades_df["pnl_usd"].mean()) if not trades_df.empty else 0.0,
        "avg_winner_usd": float(wins["pnl_usd"].mean()) if not wins.empty else 0.0,
        "avg_loser_usd": float(losses["pnl_usd"].mean()) if not losses.empty else 0.0,
        "max_concurrent_positions": int(diagnostics.get("max_concurrent_positions", 0)),
        "margin_rejects": int(diagnostics.get("margin_rejects", 0)),
        "per_strategy": per_strategy,
        "per_regime": per_regime,
        "by_hour": by_hour,
        "by_day": by_day,
        "no_trade_reasons": dict(no_trade_reasons),
        "diagnostics": diagnostics,
    }


def _load_defended_overlays() -> dict[str, Any]:
    path = ROOT / "research_out" / "paper_candidate_v7_defended.json"
    data = _read_json(path)
    return dict(data.get("overrides") or {})


def _build_tokyo_cfg_params(cfg: dict[str, Any]) -> dict[str, Any]:
    entry_rules = cfg.get("entry_rules", {})
    long_cfg = entry_rules.get("long", {})
    short_cfg = entry_rules.get("short", {})
    regime_gate = cfg.get("regime_gate", {})
    combo_filter_cfg = cfg.get("confluence_combo_filter", {})
    cq = cfg.get("confluence_quality", {})
    ss_cfg = cfg.get("signal_strength_tracking", {})
    ss_filter_cfg = cfg.get("signal_strength_filter", {})
    core_gate_cfg = entry_rules.get("core_gate", {})
    return {
        "tokyo_v2_scoring": True,
        "confluence_min_long": int(long_cfg.get("confluence_scoring", {}).get("minimum_score", 2)),
        "confluence_min_short": int(short_cfg.get("confluence_scoring", {}).get("minimum_score", 2)),
        "long_rsi_soft_entry": float(long_cfg.get("rsi_soft_filter", {}).get("entry_soft_threshold", 35.0)),
        "long_rsi_bonus": float(long_cfg.get("rsi_soft_filter", {}).get("bonus_threshold", 30.0)),
        "short_rsi_soft_entry": float(short_cfg.get("rsi_soft_filter", {}).get("entry_soft_threshold", 65.0)),
        "short_rsi_bonus": float(short_cfg.get("rsi_soft_filter", {}).get("bonus_threshold", 70.0)),
        "tol": float(long_cfg.get("price_zone", {}).get("tolerance_pips", 20.0)) * PIP_SIZE,
        "atr_max": float(cfg.get("indicators", {}).get("atr", {}).get("max_threshold_price_units", 0.3)),
        "core_gate_use_zone": bool(core_gate_cfg.get("use_zone", True)),
        "core_gate_use_bb": bool(core_gate_cfg.get("use_bb_touch", True)),
        "core_gate_use_sar": bool(core_gate_cfg.get("use_sar_flip", True)),
        "core_gate_use_rsi": bool(core_gate_cfg.get("use_rsi_soft", True)),
        "core_gate_required": int(core_gate_cfg.get("required_count", 4)),
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
        "combo_filter_mode": str(combo_filter_cfg.get("mode", "blocklist")),
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
    }


def _build_tokyo_frame(input_csv: str, cfg: dict[str, Any]) -> pd.DataFrame:
    df = tokyo_engine.add_indicators(tokyo_engine.load_m1(input_csv), cfg)
    df["time_utc"] = df["time"].dt.tz_convert("UTC")
    df["time_jst"] = df["time"].dt.tz_convert(tokyo_engine.TOKYO_TZ)
    df["session_day_jst"] = df["time_jst"].dt.date.astype(str)
    df["weekday_jst"] = df["time_jst"].dt.dayofweek
    df["utc_day_name"] = df["time_utc"].dt.day_name()
    df["hour_utc"] = df["time_utc"].dt.hour
    df["minute_utc"] = df["time_utc"].dt.minute
    df["minute_of_day_utc"] = df["hour_utc"] * 60 + df["minute_utc"]
    sf = cfg.get("session_filter", {})
    start = str(sf.get("session_start_utc", "16:00"))
    end = str(sf.get("session_end_utc", "22:00"))
    allowed_days = set(sf.get("allowed_trading_days", ["Tuesday", "Wednesday", "Friday"]))

    def hhmm_to_minutes(s: str) -> int:
        hh, mm = s.strip().split(":")
        return int(hh) * 60 + int(mm)

    start_min = hhmm_to_minutes(start)
    end_min = hhmm_to_minutes(end)
    mins = df["minute_of_day_utc"].astype(int)
    if start_min < end_min:
        df["in_tokyo_session"] = (mins >= start_min) & (mins < end_min)
    else:
        df["in_tokyo_session"] = (mins >= start_min) | (mins < end_min)
    df["allowed_trading_day"] = df["utc_day_name"].isin(allowed_days)
    return df


def _resample_m1(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return tokyo_engine.resample_ohlc_continuous(df, rule)


def _build_regime_frame(m1_df: pd.DataFrame) -> pd.DataFrame:
    featured = regime_validation.compute_features(m1_df)
    classified = regime_validation.classify_all_bars(featured, RegimeThresholds())
    m5 = regime_validation._resample(m1_df, "5min")
    close = m5["close"].astype(float)
    net_disp = (close - close.shift(12)).abs()
    total_path = close.diff().abs().rolling(12, min_periods=12).sum()
    er = net_disp / total_path.replace(0.0, np.nan)
    delta_er = er - er.shift(3)
    aux = m5[["time"]].copy()
    aux["er_m5"] = er
    aux["delta_er_m5"] = delta_er
    classified = pd.merge_asof(
        classified.sort_values("time"),
        aux.sort_values("time"),
        on="time",
        direction="backward",
    )
    return classified


def _build_v44_feature_frame(m1_df: pd.DataFrame, v44_cfg: dict[str, Any]) -> pd.DataFrame:
    m5 = regime_validation._resample(m1_df, "5min")
    h1 = regime_validation._resample(m1_df, "1h")

    m5_close = m5["close"].astype(float)
    m5_open = m5["open"].astype(float)
    m5["v44_ema_fast"] = m5_close.ewm(span=int(v44_cfg.get("v5_m5_ema_fast", 9)), adjust=False).mean()
    m5["v44_ema_slow"] = m5_close.ewm(span=int(v44_cfg.get("v5_m5_ema_slow", 21)), adjust=False).mean()
    m5["v44_trail_ema_9"] = m5_close.ewm(span=int(v44_cfg.get("v5_normal_trail_ema", 9)), adjust=False).mean()
    m5["v44_trail_ema_21"] = m5_close.ewm(span=int(v44_cfg.get("v5_strong_trail_ema", 21)), adjust=False).mean()
    slope_bars = max(1, int(v44_cfg.get("v5_slope_bars", 4)))
    m5["v44_slope"] = (m5["v44_ema_fast"] - m5["v44_ema_fast"].shift(slope_bars)) / PIP_SIZE / slope_bars
    m5["v44_body_pips"] = (m5_close - m5_open).abs() / PIP_SIZE
    tr = pd.concat(
        [
            (m5["high"].astype(float) - m5["low"].astype(float)).abs(),
            (m5["high"].astype(float) - m5_close.shift()).abs(),
            (m5["low"].astype(float) - m5_close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    m5["v44_atr14"] = tr.rolling(14, min_periods=14).mean()
    lookback = max(20, int(v44_cfg.get("v5_atr_pct_lookback", 200)))
    cap = float(v44_cfg.get("v5_atr_pct_cap", 0.67))
    atr_vals = m5["v44_atr14"].to_numpy(dtype=float)
    atr_ok = np.ones(len(m5), dtype=bool)
    atr_window: deque[float] = deque()
    atr_sorted: list[float] = []
    for idx, aval in enumerate(atr_vals):
        if np.isfinite(aval):
            aval_f = float(aval)
            atr_window.append(aval_f)
            insort(atr_sorted, aval_f)
            if len(atr_window) > lookback:
                old = float(atr_window.popleft())
                rm_idx = bisect_right(atr_sorted, old) - 1
                if 0 <= rm_idx < len(atr_sorted):
                    atr_sorted.pop(rm_idx)
            if len(atr_sorted) >= 20:
                rank = bisect_right(atr_sorted, aval_f) / len(atr_sorted)
                atr_ok[idx] = rank <= cap
    m5["v44_atr_ok"] = atr_ok
    m5["v44_sl_low_6"] = m5["low"].astype(float).rolling(6, min_periods=1).min()
    m5["v44_sl_high_6"] = m5["high"].astype(float).rolling(6, min_periods=1).max()

    h1_close = h1["close"].astype(float)
    h1_fast = h1_close.ewm(span=int(v44_cfg.get("h1_ema_fast", 20)), adjust=False).mean()
    h1_slow = h1_close.ewm(span=int(v44_cfg.get("h1_ema_slow", 50)), adjust=False).mean()
    h1["v44_h1_trend"] = np.where(h1_fast > h1_slow, "up", np.where(h1_fast < h1_slow, "down", None))

    out = pd.merge_asof(
        m1_df[["time", "open", "high", "low", "close"]].sort_values("time"),
        m5[["time", "v44_ema_fast", "v44_ema_slow", "v44_slope", "v44_body_pips", "v44_atr_ok", "open", "close"]]
        .assign(
            v44_trail_ema_9=m5["v44_trail_ema_9"],
            v44_trail_ema_21=m5["v44_trail_ema_21"],
            v44_sl_low_6=m5["v44_sl_low_6"],
            v44_sl_high_6=m5["v44_sl_high_6"],
        )
        .rename(columns={"open": "v44_m5_open", "close": "v44_m5_close"})
        .sort_values("time"),
        on="time",
        direction="backward",
    )
    out = pd.merge_asof(
        out.sort_values("time"),
        h1[["time", "v44_h1_trend"]].sort_values("time"),
        on="time",
        direction="backward",
    )
    return out


def _session_hour(ts: pd.Timestamp) -> float:
    return float(ts.hour) + float(ts.minute) / 60.0


def _session_spread_pips(ts: pd.Timestamp, family: str, bar_index: int, london_cfg: dict[str, Any], v44_cfg: dict[str, Any], tokyo_cfg: dict[str, Any]) -> float:
    if family == "tokyo":
        exec_cfg = tokyo_cfg.get("execution_model", {})
        return float(tokyo_engine.compute_spread_pips(bar_index, ts, str(exec_cfg.get("spread_mode", "fixed")), float(exec_cfg.get("spread_pips", 1.5)), float(exec_cfg.get("spread_min_pips", 1.0)), float(exec_cfg.get("spread_max_pips", 3.0))))
    if family == "london":
        return float(london_engine.compute_spread_pips(bar_index, ts, london_cfg))
    base = float(v44_cfg.get("spread_pips", 2.0))
    mn = float(v44_cfg.get("spread_min_pips", 1.0))
    mx = float(v44_cfg.get("spread_max_pips", 3.0))
    h = _session_hour(ts)
    if 13.0 <= h < 16.0:
        base += 0.3
    wiggle = 0.12 * math.sin(bar_index * 0.017) + 0.05 * math.sin(bar_index * 0.071)
    return max(mn, min(mx, base + wiggle))


def _slippage_pips(candidate: RegimeOwnedCandidate) -> float:
    if candidate.source_family == "tokyo":
        return 0.15
    if candidate.source_family == "london":
        return 0.25
    if candidate.source_family == "v44" and "news" in candidate.reason.lower():
        return 0.5
    return 0.35


def _entry_price_from_bar(side: str, open_mid: float, spread_pips: float, slippage_pips: float) -> float:
    bid, ask = _to_bid_ask(open_mid, spread_pips)
    slip = float(slippage_pips) * PIP_SIZE
    if side == "buy":
        return ask + slip
    return bid - slip


def _regime_owner_bonus(strategy: str, regime_label: str, regime_margin: float) -> float:
    if strategy == "v14":
        return 0.18 if regime_label == "mean_reversion" else 0.0
    if strategy == "london_v2":
        return 0.18 if regime_label in {"breakout", "post_breakout_trend"} else 0.0
    if strategy == "v44_ny":
        return 0.18 if regime_label == "momentum" else 0.0
    return 0.0


def _compute_raw_quality_tokyo(candidate: dict[str, Any]) -> float:
    confluence = _safe_float(candidate.get("confluence_score", 0.0))
    strength = _safe_float(candidate.get("signal_strength_score", 0.0))
    label_bonus = {"low": 0.0, "medium": 0.5, "high": 1.0}.get(str(candidate.get("quality_label", "")).lower(), 0.0)
    return confluence * 10.0 + strength + label_bonus


def _compute_raw_quality_london(pending: dict[str, Any], setup_cfg: dict[str, Any]) -> float:
    setup = str(pending.get("setup_type", "")).upper()
    risk_pct = float(setup_cfg.get("risk_per_trade_pct", 0.01))
    asian_range = _safe_float(pending.get("asian_range_pips", 0.0), 0.0)
    lor_range = _safe_float(pending.get("lor_range_pips", 0.0), 0.0)
    is_reentry = bool(pending.get("is_reentry", False))
    setup_base = {"A": 3.2, "D": 3.0, "C": 2.4, "B": 2.2}.get(setup, 2.0)
    asian_bonus = max(0.0, 1.5 - abs(asian_range - 45.0) / 15.0) if asian_range > 0 else 0.0
    lor_bonus = max(0.0, 1.2 - abs(lor_range - 10.0) / 8.0) if lor_range > 0 else 0.0
    reentry_penalty = 0.5 if is_reentry else 0.0
    return setup_base * 10.0 + risk_pct * 100.0 + asian_bonus + lor_bonus - reentry_penalty


def _compute_raw_quality_v44(side: str, strength: str, reason: str, session: str) -> float:
    base = {"strong": 85.0, "normal": 65.0, "weak": 45.0}.get(strength, 55.0)
    if session == "london":
        base -= 5.0
    if "news" in reason.lower():
        base += 10.0
    if side == "buy" or side == "sell":
        base += 0.0
    return base


def _normalize_quality(state: PortfolioBacktestState, strategy: str, raw_quality: float) -> float:
    hist = state.strategy_quality_history[strategy]
    sorted_hist = state.strategy_quality_sorted[strategy]
    if len(hist) < 20:
        hist.append(float(raw_quality))
        insort(sorted_hist, float(raw_quality))
        if len(hist) == 1:
            return 0.5
        mn = min(hist)
        mx = max(hist)
        return float((raw_quality - mn) / (mx - mn)) if mx > mn else 0.5
    pos = bisect_right(sorted_hist, float(raw_quality))
    score = float(pos / max(1, len(sorted_hist)))
    hist.append(float(raw_quality))
    insort(sorted_hist, float(raw_quality))
    if len(hist) > 5000:
        old = hist.pop(0)
        sorted_hist.pop(bisect_right(sorted_hist, old) - 1)
    return score


def _regime_snapshot_row(classified: pd.DataFrame, idx: int) -> tuple[str, float, str, dict[str, Any]]:
    row = classified.iloc[idx]
    regime_label = str(row.get("regime_hysteresis", "ambiguous"))
    regime_margin = float(row.get("score_margin", 0.0)) if pd.notna(row.get("score_margin")) else 0.0
    m5_time = pd.Timestamp(row["time"])
    er_val = float(row.get("er_m5", row.get("dir_efficiency", 0.5))) if pd.notna(row.get("er_m5", np.nan)) or pd.notna(row.get("dir_efficiency", np.nan)) else 0.5
    der_val = float(row.get("delta_er_m5", 0.0)) if pd.notna(row.get("delta_er_m5", np.nan)) else 0.0
    ownership_cell = cell_key(regime_label, er_bucket(er_val), der_bucket(der_val))
    meta = {
        "score_momentum": float(row.get("score_momentum", 0.0)) if pd.notna(row.get("score_momentum")) else 0.0,
        "score_mean_reversion": float(row.get("score_mean_reversion", 0.0)) if pd.notna(row.get("score_mean_reversion")) else 0.0,
        "score_breakout": float(row.get("score_breakout", 0.0)) if pd.notna(row.get("score_breakout")) else 0.0,
        "score_post_breakout_trend": float(row.get("score_post_breakout_trend", 0.0)) if pd.notna(row.get("score_post_breakout_trend")) else 0.0,
        "m5_time": m5_time,
    }
    return regime_label, regime_margin, ownership_cell, meta


def _ownership_cell_from_row(classified: pd.DataFrame, idx: int, regime_label: str) -> str:
    row = classified.iloc[idx]
    er_val = float(row.get("er_m5", row.get("dir_efficiency", 0.5))) if pd.notna(row.get("er_m5", np.nan)) or pd.notna(row.get("dir_efficiency", np.nan)) else 0.5
    der_val = float(row.get("delta_er_m5", 0.0)) if pd.notna(row.get("delta_er_m5", np.nan)) else 0.0
    return cell_key(regime_label, er_bucket(er_val), der_bucket(der_val))


def _position_strategy_exit_family(position: UnifiedPositionState) -> str:
    return position.source_family


def _close_position_leg(position: UnifiedPositionState, exit_px: float, units: int) -> tuple[float, float]:
    pips, usd = _calc_leg_pnl(position.side, position.entry_price, exit_px, units)
    position.realized_usd += usd
    position.weighted_pips_sum += pips * units
    position.remaining_units -= units
    return pips, usd


def _finalize_closed_position(state: PortfolioBacktestState, position: UnifiedPositionState, regime_label: str, ownership_cell: str) -> None:
    state.equity += position.realized_usd
    state.peak_equity = max(state.peak_equity, state.equity)
    if position.strategy == "v14":
        session_day = str(position.raw.get("tokyo_session_day") or "")
        sst = state.tokyo_session_state.get(session_day)
        if sst is not None:
            if position.realized_usd < 0:
                sst["consecutive_losses"] = int(sst.get("consecutive_losses", 0)) + 1
                cooldown_minutes = int(position.raw.get("cooldown_minutes", 0) or 0)
                if cooldown_minutes > 0 and position.exit_time is not None:
                    sst["cooldown_until"] = (pd.Timestamp(position.exit_time) + pd.Timedelta(minutes=cooldown_minutes)).isoformat()
                same_dir_minutes = int(position.raw.get("same_direction_stop_minutes", 0) or 0)
                if same_dir_minutes > 0 and position.exit_time is not None:
                    sst["same_direction_stop_until"] = (pd.Timestamp(position.exit_time) + pd.Timedelta(minutes=same_dir_minutes)).isoformat()
                    sst["same_direction_stop_side"] = str(position.side)
            else:
                sst["consecutive_losses"] = 0
    if position.strategy == "v44_ny" and state.v44_session_state:
        sess = state.v44_session_state.get(position.session_owner) or {}
        if position.realized_usd < 0:
            sess["consecutive_losses"] = int(sess.get("consecutive_losses", 0)) + 1
            sess["cooldown_until"] = (pd.Timestamp(position.exit_time) + pd.Timedelta(minutes=1)).isoformat() if position.exit_time is not None else None
        else:
            sess["consecutive_losses"] = 0
    state.closed_trades.append(
        {
            "trade_id": position.trade_id,
            "strategy": position.strategy,
            "source_family": position.source_family,
            "session_owner": position.session_owner,
            "source_setup": position.source_setup,
            "side": position.side,
            "entry_time_utc": position.entry_time,
            "exit_time_utc": position.exit_time,
            "entry_price": position.entry_price,
            "exit_price": position.exit_price_last,
            "sl_price": position.execution_plan.sl_price,
            "tp1_price": position.execution_plan.tp1_price,
            "tp2_price": position.execution_plan.tp2_price,
            "pnl_pips": (position.weighted_pips_sum / position.initial_units) if position.initial_units > 0 else 0.0,
            "pnl_usd": position.realized_usd,
            "exit_reason": position.exit_reason,
            "position_units": position.initial_units,
            "mfe_pips": position.mfe_pips,
            "mae_pips": position.mae_pips,
            "regime_label": regime_label,
            "ownership_cell": ownership_cell,
        }
    )
    state.equity_curve.append({"time": position.exit_time, "equity": state.equity})


def _manage_open_positions(
    *,
    state: PortfolioBacktestState,
    open_mid: float,
    high_mid: float,
    low_mid: float,
    close_mid: float,
    bar_index: int,
    ts: pd.Timestamp,
    regime_label: str,
    ownership_cell: str,
    london_cfg: dict[str, Any],
    v44_cfg: dict[str, Any],
    tokyo_cfg: dict[str, Any],
    v44_trail_ema_9: Optional[float] = None,
    v44_trail_ema_21: Optional[float] = None,
) -> None:
    survivors: list[UnifiedPositionState] = []
    for p in state.open_positions:
        family = _position_strategy_exit_family(p)
        spread_pips = _session_spread_pips(ts, family, bar_index, london_cfg, v44_cfg, tokyo_cfg)
        bid_o, ask_o = _to_bid_ask(float(open_mid), spread_pips)
        bid_h, ask_h = _to_bid_ask(float(high_mid), spread_pips)
        bid_l, ask_l = _to_bid_ask(float(low_mid), spread_pips)
        if p.side == "buy":
            px_open, px_high, px_low = bid_o, bid_h, bid_l
            p.mfe_pips = max(p.mfe_pips, (float(high_mid) - p.entry_price) / PIP_SIZE)
            p.mae_pips = max(p.mae_pips, (p.entry_price - float(low_mid)) / PIP_SIZE)
        else:
            px_open, px_high, px_low = ask_o, ask_h, ask_l
            p.mfe_pips = max(p.mfe_pips, (p.entry_price - float(low_mid)) / PIP_SIZE)
            p.mae_pips = max(p.mae_pips, (float(high_mid) - p.entry_price) / PIP_SIZE)
        p.bars_open += 1

        if p.execution_plan.session_close_time is not None and ts >= p.execution_plan.session_close_time:
            exit_px = px_open
            _close_position_leg(p, exit_px, p.remaining_units)
            p.exit_reason = "SESSION_CLOSE"
            p.exit_time = ts
            p.exit_price_last = exit_px
            _finalize_closed_position(state, p, regime_label, ownership_cell)
            continue

        active_stop = p.trail_stop_price if p.trail_stop_price is not None else p.execution_plan.sl_price
        tp1 = p.execution_plan.tp1_price
        tp2 = p.execution_plan.tp2_price
        hit_sl = (px_low <= active_stop) if p.side == "buy" else (px_high >= active_stop)
        hit_tp1 = False
        if tp1 is not None and not p.tp1_hit:
            hit_tp1 = (px_high >= tp1) if p.side == "buy" else (px_low <= tp1)
        hit_tp2 = False
        if tp2 is not None and p.tp1_hit:
            hit_tp2 = (px_high >= tp2) if p.side == "buy" else (px_low <= tp2)

        def _first(level_a: float, level_b: float) -> str:
            return "a" if abs(px_open - level_a) <= abs(px_open - level_b) else "b"

        if hit_tp1 and hit_sl:
            if _first(tp1 or active_stop, active_stop) == "a":
                hit_sl = False
            else:
                hit_tp1 = False
        if hit_tp2 and hit_sl:
            if _first(tp2 or active_stop, active_stop) == "a":
                hit_sl = False
            else:
                hit_tp2 = False

        if hit_tp1 and not p.tp1_hit and tp1 is not None:
            units_to_close = int(max(0, min(p.remaining_units, math.floor(p.initial_units * p.execution_plan.tp1_close_fraction / ROUND_UNITS) * ROUND_UNITS)))
            if units_to_close <= 0:
                units_to_close = int(p.remaining_units)
            _close_position_leg(p, tp1, units_to_close)
            p.tp1_hit = True
            p.execution_plan.sl_price = p.entry_price + (p.execution_plan.be_offset_pips * PIP_SIZE if p.side == "buy" else -p.execution_plan.be_offset_pips * PIP_SIZE)
            p.moved_to_be = True
            active_stop = p.execution_plan.sl_price
            if p.remaining_units <= 0:
                p.exit_reason = "TP1_FULL"
                p.exit_time = ts
                p.exit_price_last = tp1
                _finalize_closed_position(state, p, regime_label, ownership_cell)
                continue

        if hit_tp2 and tp2 is not None:
            _close_position_leg(p, tp2, p.remaining_units)
            p.exit_reason = "TP2"
            p.exit_time = ts
            p.exit_price_last = tp2
            _finalize_closed_position(state, p, regime_label, ownership_cell)
            continue

        if hit_sl:
            _close_position_leg(p, active_stop, p.remaining_units)
            p.exit_reason = "STOP_LOSS" if not p.tp1_hit else "RUNNER_STOP"
            p.exit_time = ts
            p.exit_price_last = active_stop
            _finalize_closed_position(state, p, regime_label, ownership_cell)
            continue

        if (
            p.tp1_hit
            and p.execution_plan.time_decay_minutes
            and p.execution_plan.time_decay_profit_cap_pips is not None
        ):
            held_minutes = float((ts - pd.Timestamp(p.entry_time)).total_seconds() / 60.0)
            close_ref = bid_o if p.side == "buy" else ask_o
            current_pips = ((close_ref - p.entry_price) / PIP_SIZE) if p.side == "buy" else ((p.entry_price - close_ref) / PIP_SIZE)
            if held_minutes >= float(p.execution_plan.time_decay_minutes) and 0.0 <= current_pips < float(p.execution_plan.time_decay_profit_cap_pips):
                _close_position_leg(p, close_ref, p.remaining_units)
                p.exit_reason = "TIME_DECAY"
                p.exit_time = ts
                p.exit_price_last = close_ref
                _finalize_closed_position(state, p, regime_label, ownership_cell)
                continue

        trail_ok = (not p.execution_plan.trail_requires_tp1) or p.tp1_hit
        if trail_ok and p.mfe_pips >= p.execution_plan.trail_activate_pips:
            if p.execution_plan.trail_reference == "ema":
                trail_field = str(p.execution_plan.trail_ema_field or "")
                ema_val = None
                if trail_field == "v44_trail_ema_21":
                    ema_val = v44_trail_ema_21
                elif trail_field == "v44_trail_ema_9":
                    ema_val = v44_trail_ema_9
                if ema_val is not None and np.isfinite(float(ema_val)):
                    if p.side == "buy":
                        new_stop = float(ema_val) - p.execution_plan.trail_distance_pips * PIP_SIZE
                        if new_stop > p.execution_plan.sl_price:
                            p.trail_stop_price = new_stop
                    else:
                        new_stop = float(ema_val) + p.execution_plan.trail_distance_pips * PIP_SIZE
                        if new_stop < p.execution_plan.sl_price:
                            p.trail_stop_price = new_stop
            elif p.side == "buy":
                new_stop = px_high - p.execution_plan.trail_distance_pips * PIP_SIZE
                if new_stop > p.execution_plan.sl_price:
                    p.trail_stop_price = new_stop
            else:
                new_stop = px_low + p.execution_plan.trail_distance_pips * PIP_SIZE
                if new_stop < p.execution_plan.sl_price:
                    p.trail_stop_price = new_stop

        survivors.append(p)

    state.open_positions = survivors


def _next_day_reset_v44(state: PortfolioBacktestState, day_key: str) -> None:
    if state.v44_session_state.get("day_key") == day_key:
        return
    state.v44_session_state = {
        "day_key": day_key,
        "london": {"trade_count": 0, "consecutive_losses": 0, "cooldown_until": None},
        "ny": {"trade_count": 0, "consecutive_losses": 0, "cooldown_until": None},
    }


def _generate_tokyo_candidate(
    *,
    state: PortfolioBacktestState,
    row: pd.Series,
    next_ts: pd.Timestamp,
    regime_label: str,
    regime_margin: float,
    ownership_cell: str,
    tokyo_cfg: dict[str, Any],
    tokyo_cfg_params: dict[str, Any],
    bar_index: int,
) -> Optional[RegimeOwnedCandidate]:
    if not bool(row.get("in_tokyo_session", False)) or not bool(row.get("allowed_trading_day", False)):
        return None
    session_day = str(row.get("session_day_jst"))
    sst = state.tokyo_session_state.get(session_day)
    if sst is None:
        sst = {
            "session_open_price": float(row["open"]),
            "session_high": float(row["high"]),
            "session_low": float(row["low"]),
            "ir_ready": False,
            "ir_high": float(row["high"]),
            "ir_low": float(row["low"]),
            "trades": 0,
            "last_entry_time": None,
            "consecutive_losses": 0,
            "cooldown_until": None,
        }
        state.tokyo_session_state[session_day] = sst
    sst["session_high"] = max(float(sst["session_high"]), float(row["high"]))
    sst["session_low"] = min(float(sst["session_low"]), float(row["low"]))
    if not sst["ir_ready"]:
        sst["ir_high"] = max(float(sst["ir_high"]), float(row["high"]))
        sst["ir_low"] = min(float(sst["ir_low"]), float(row["low"]))
        minutes_from_start = int((_safe_int(row["minute_of_day_utc"]) - (_safe_int(pd.Timestamp(next_ts).hour) * 0)))
        if bool(tokyo_cfg.get("session_envelope", {}).get("enabled", False)):
            warmup = int(tokyo_cfg.get("session_envelope", {}).get("warmup_minutes", 30))
            session_start = tokyo_cfg.get("session_filter", {}).get("session_start_utc", "16:00")
            sh, sm = [int(x) for x in str(session_start).split(":")]
            now_m = int(row["hour_utc"]) * 60 + int(row["minute_utc"])
            start_m = sh * 60 + sm
            if now_m - start_m >= warmup:
                sst["ir_ready"] = True
    if sst.get("cooldown_until") is not None and pd.Timestamp(next_ts) < pd.Timestamp(sst["cooldown_until"]):
        return None
    stop_after_losses = int(tokyo_cfg.get("trade_management", {}).get("stop_after_consecutive_losses", 0))
    if stop_after_losses > 0 and int(sst.get("consecutive_losses", 0)) >= stop_after_losses:
        return None
    min_gap = int(tokyo_cfg.get("trade_management", {}).get("min_time_between_entries_minutes", 0))
    last_entry_time = sst.get("last_entry_time")
    if min_gap > 0 and last_entry_time is not None and (pd.Timestamp(next_ts) - pd.Timestamp(last_entry_time)).total_seconds() / 60.0 < min_gap:
        return None
    same_dir_until = sst.get("same_direction_stop_until")
    same_dir_side = sst.get("same_direction_stop_side")
    max_trades = int(tokyo_cfg.get("trade_management", {}).get("max_trades_per_session", 4))
    if int(sst.get("trades", 0)) >= max_trades:
        return None
    mid_close = float(row["close"])
    mid_open = float(row["open"])
    mid_high = float(row["high"])
    mid_low = float(row["low"])
    pivot_levels = {
        "P": float(row.get("pivot_P", np.nan)),
        "R1": float(row.get("pivot_R1", np.nan)),
        "R2": float(row.get("pivot_R2", np.nan)),
        "R3": float(row.get("pivot_R3", np.nan)),
        "S1": float(row.get("pivot_S1", np.nan)),
        "S2": float(row.get("pivot_S2", np.nan)),
        "S3": float(row.get("pivot_S3", np.nan)),
    }
    if any(not np.isfinite(v) for v in pivot_levels.values()):
        return None
    candidate = evaluate_v14_entry_signal(
        row=row,
        mid_close=mid_close,
        mid_open=mid_open,
        mid_high=mid_high,
        mid_low=mid_low,
        pivot_levels=pivot_levels,
        cfg_params=tokyo_cfg_params,
        sst=sst,
    )
    if candidate is None or candidate.get("_blocked_reason"):
        return None
    direction = "buy" if str(candidate.get("direction")) == "long" else "sell"
    if same_dir_until is not None and same_dir_side == direction and pd.Timestamp(next_ts) < pd.Timestamp(same_dir_until):
        return None
    spread_pips = _session_spread_pips(pd.Timestamp(next_ts), "tokyo", bar_index, {}, {}, tokyo_cfg)
    entry_mid = float(row["close"])
    entry_price = _entry_price_from_bar(direction, entry_mid, spread_pips, _slippage_pips_placeholder("tokyo"))
    sl_buffer = float(tokyo_cfg.get("exit_rules", {}).get("stop_loss", {}).get("buffer_pips", 8.0))
    min_sl_pips = float(tokyo_cfg.get("exit_rules", {}).get("stop_loss", {}).get("minimum_sl_pips", 12.0))
    max_sl_pips = float(tokyo_cfg.get("exit_rules", {}).get("stop_loss", {}).get("hard_max_sl_pips", 35.0))
    if direction == "buy":
        pivot_ref = min(float(pivot_levels["S1"]), float(pivot_levels["S2"]))
        raw_sl = pivot_ref - sl_buffer * PIP_SIZE
        sl_pips = min(max((entry_price - raw_sl) / PIP_SIZE, min_sl_pips), max_sl_pips)
        sl_price = entry_price - sl_pips * PIP_SIZE
        tp2_price = float(pivot_levels["P"])
    else:
        pivot_ref = max(float(pivot_levels["R1"]), float(pivot_levels["R2"]))
        raw_sl = pivot_ref + sl_buffer * PIP_SIZE
        sl_pips = min(max((raw_sl - entry_price) / PIP_SIZE, min_sl_pips), max_sl_pips)
        sl_price = entry_price + sl_pips * PIP_SIZE
        tp2_price = float(pivot_levels["P"])
    atr_pips = float(row.get("atr_m15", 0.2)) / PIP_SIZE
    tp1_pips = min(float(tokyo_cfg.get("exit_rules", {}).get("take_profit", {}).get("partial_tp_max_pips", 12.0)), max(float(tokyo_cfg.get("exit_rules", {}).get("take_profit", {}).get("partial_tp_min_pips", 6.0)), float(tokyo_cfg.get("exit_rules", {}).get("take_profit", {}).get("partial_tp_atr_mult", 0.5)) * atr_pips))
    tp1_price = entry_price + tp1_pips * PIP_SIZE if direction == "buy" else entry_price - tp1_pips * PIP_SIZE
    raw_quality = _compute_raw_quality_tokyo(candidate)
    normalized = _normalize_quality(state, "v14", raw_quality)
    session_end = tokyo_cfg.get("session_filter", {}).get("session_end_utc", "22:00")
    eh, em = [int(x) for x in str(session_end).split(":")]
    close_day = pd.Timestamp(next_ts).normalize()
    session_close = close_day + pd.Timedelta(hours=eh, minutes=em)
    return RegimeOwnedCandidate(
        strategy="v14",
        session_owner="tokyo",
        side=direction,
        signal_time=pd.Timestamp(row["time"]),
        execute_time=pd.Timestamp(next_ts),
        entry_price_basis="next_bar_open",
        sl_pips=float(sl_pips),
        raw_quality=float(raw_quality),
        quality_normalized=float(normalized),
        regime_label=regime_label,
        regime_margin=float(regime_margin),
        ownership_cell=ownership_cell,
        reason=str(candidate.get("combo") or candidate.get("signal_strength_tier") or "v14 signal"),
        source_features={
            "confluence_score": int(candidate.get("confluence_score", 0)),
            "signal_strength_score": int(candidate.get("signal_strength_score", 0)),
            "quality_label": str(candidate.get("quality_label", "medium")),
            "tp1_offset_pips": float(tp1_pips),
            "tp2_offset_pips": float(abs(tp2_price - entry_price) / PIP_SIZE),
        },
        execution_plan=UnifiedExecutionPlan(
            sl_price=float(sl_price),
            tp1_price=float(tp1_price),
            tp2_price=float(tp2_price),
            tp1_close_fraction=float(tokyo_cfg.get("exit_rules", {}).get("take_profit", {}).get("partial_close_pct", 0.5)),
            be_offset_pips=float(tokyo_cfg.get("exit_rules", {}).get("take_profit", {}).get("breakeven_offset_pips", 2.0)),
            trail_activate_pips=float(tokyo_cfg.get("exit_rules", {}).get("trailing_stop", {}).get("activate_after_profit_pips", 8.0)),
            trail_distance_pips=float(tokyo_cfg.get("exit_rules", {}).get("trailing_stop", {}).get("trail_distance_pips", 5.0)),
            trail_requires_tp1=bool(tokyo_cfg.get("exit_rules", {}).get("trailing_stop", {}).get("requires_tp1_hit", True)),
            session_close_time=session_close,
            time_decay_minutes=float(tokyo_cfg.get("exit_rules", {}).get("time_exit", {}).get("time_decay_minutes", 0.0) or 0.0),
            time_decay_profit_cap_pips=float(tokyo_cfg.get("exit_rules", {}).get("time_exit", {}).get("time_decay_profit_cap_pips", 0.0) or 0.0),
        ),
        risk_pct=float(tokyo_cfg.get("position_sizing", {}).get("risk_per_trade_pct", 2.0)) / 100.0,
        max_spread_pips=float(tokyo_cfg.get("execution_model", {}).get("spread_pips", 1.5)) + 1.0,
        source_family="tokyo",
    )


# helper used above

def _slippage_pips_placeholder(family: str) -> float:
    if family == "tokyo":
        return 0.15
    if family == "london":
        return 0.25
    return 0.35


def _ensure_london_day_state(state: PortfolioBacktestState, day: pd.Timestamp, london_cfg: dict[str, Any], day_df: pd.DataFrame) -> dict[str, Any]:
    day_key = str(day.date())
    if day_key in state.london_day_state:
        return state.london_day_state[day_key]
    london_open = day + pd.Timedelta(hours=london_engine.uk_london_open_utc(day))
    ny_open = day + pd.Timedelta(hours=london_engine.us_ny_open_utc(day))
    asian = day_df[(day_df["time"] >= day) & (day_df["time"] < london_open)]
    asian_high = float(asian["high"].max()) if not asian.empty else np.nan
    asian_low = float(asian["low"].min()) if not asian.empty else np.nan
    asian_range_pips = (asian_high - asian_low) / PIP_SIZE if not asian.empty else np.nan
    asian_valid = (not asian.empty) and float(london_cfg["levels"]["asian_range_min_pips"]) <= asian_range_pips <= float(london_cfg["levels"]["asian_range_max_pips"])
    lor_end = london_open + pd.Timedelta(minutes=15)
    lor = day_df[(day_df["time"] >= london_open) & (day_df["time"] < lor_end)]
    lor_high = float(lor["high"].max()) if not lor.empty else np.nan
    lor_low = float(lor["low"].min()) if not lor.empty else np.nan
    lor_range_pips = (lor_high - lor_low) / PIP_SIZE if not lor.empty else np.nan
    lor_valid = (not lor.empty) and float(london_cfg["levels"]["lor_range_min_pips"]) <= lor_range_pips <= float(london_cfg["levels"]["lor_range_max_pips"])
    windows = {}
    for setup in ["A", "B", "C", "D"]:
        s = london_cfg["setups"][setup]
        start = london_open + pd.Timedelta(minutes=int(s["entry_start_min_after_london"]))
        if s.get("entry_end_min_before_ny") is not None:
            end = ny_open - pd.Timedelta(minutes=int(s["entry_end_min_before_ny"]))
        else:
            end = london_open + pd.Timedelta(minutes=int(s["entry_end_min_after_london"]))
        if end > ny_open:
            end = ny_open
        windows[setup] = (start, end)
    channels: dict[tuple[str, str], dict[str, Any]] = {}
    for setup in ["A", "B", "C", "D"]:
        for d in ["long", "short"]:
            channels[(setup, d)] = {"state": "ARMED", "cooldown_until": None, "entries": 0, "resets": 0}
    st = {
        "day_df": day_df,
        "day": day,
        "london_open": london_open,
        "ny_open": ny_open,
        "asian_high": asian_high,
        "asian_low": asian_low,
        "asian_range_pips": asian_range_pips,
        "asian_valid": asian_valid,
        "lor_high": lor_high,
        "lor_low": lor_low,
        "lor_range_pips": lor_range_pips,
        "lor_valid": lor_valid,
        "windows": windows,
        "channels": channels,
        "b_candidates": {"long": [], "short": []},
        "entries_total": 0,
        "entries_setup": {"A": 0, "B": 0, "C": 0, "D": 0},
        "entries_setup_dir": {(s, d): 0 for s in ["A", "B", "C", "D"] for d in ["long", "short"]},
    }
    state.london_day_state[day_key] = st
    return st


def _generate_london_candidates(
    *,
    state: PortfolioBacktestState,
    row: pd.Series,
    next_ts: pd.Timestamp,
    day_state: dict[str, Any],
    regime_label: str,
    regime_margin: float,
    ownership_cell: str,
    london_cfg: dict[str, Any],
    bar_index: int,
    defended_overlays: Optional[dict[str, Any]] = None,
) -> list[RegimeOwnedCandidate]:
    ts = pd.Timestamp(row["time"])
    if ts.day_name() not in set(london_cfg.get("session", {}).get("active_days_utc", [])):
        return []
    max_total = london_cfg.get("entry_limits", {}).get("max_trades_per_day_total")
    if max_total is not None and int(day_state.get("entries_total", 0)) >= int(max_total):
        return []
    entries, day_state["b_candidates"] = evaluate_london_v2_entry_signal(
        row=row,
        cfg=london_cfg,
        asian_high=day_state["asian_high"],
        asian_low=day_state["asian_low"],
        asian_range_pips=day_state["asian_range_pips"],
        asian_valid=day_state["asian_valid"],
        lor_high=day_state["lor_high"],
        lor_low=day_state["lor_low"],
        lor_range_pips=day_state["lor_range_pips"],
        lor_valid=day_state["lor_valid"],
        ts=ts,
        nxt_ts=next_ts,
        bar_index=bar_index,
        windows=day_state["windows"],
        channels=day_state["channels"],
        b_candidates=day_state["b_candidates"],
    )
    out: list[RegimeOwnedCandidate] = []
    l1_disabled_days = set(str(x) for x in ((defended_overlays or {}).get("l1_weekday_disable") or []))
    l1_exit_override = dict((defended_overlays or {}).get("l1_exit_override") or {})
    for pe in entries:
        setup = str(pe.get("setup_type"))
        if setup == "D" and ts.day_name() in l1_disabled_days:
            continue
        direction = "buy" if str(pe.get("direction")) == "long" else "sell"
        s_cfg = london_cfg["setups"][setup]
        open_mid = float(row["close"])
        entry_spread = _session_spread_pips(next_ts, "london", bar_index, london_cfg, {}, {})
        entry_price = _entry_price_from_bar(direction, open_mid, entry_spread, _slippage_pips_placeholder("london"))
        sl_price, sl_pips = london_engine.clamp_sl(
            entry_price,
            float(pe["raw_sl"]),
            "long" if direction == "buy" else "short",
            float(s_cfg["sl_min_pips"]),
            float(s_cfg["sl_max_pips"]),
        )
        sign = 1.0 if direction == "buy" else -1.0
        tp1_r_multiple = float(l1_exit_override.get("tp1_r_multiple", s_cfg["tp1_r_multiple"])) if setup == "D" and l1_exit_override else float(s_cfg["tp1_r_multiple"])
        tp2_r_multiple = float(l1_exit_override.get("tp2_r_multiple", s_cfg["tp2_r_multiple"])) if setup == "D" and l1_exit_override else float(s_cfg["tp2_r_multiple"])
        be_offset_pips = float(l1_exit_override.get("be_offset_pips", s_cfg["be_offset_pips"])) if setup == "D" and l1_exit_override else float(s_cfg["be_offset_pips"])
        tp1 = entry_price + sign * tp1_r_multiple * sl_pips * PIP_SIZE
        tp2 = entry_price + sign * tp2_r_multiple * sl_pips * PIP_SIZE
        raw_quality = _compute_raw_quality_london(pe, s_cfg)
        normalized = _normalize_quality(state, "london_v2", raw_quality)
        session_close = day_state["ny_open"] if bool(london_cfg.get("session", {}).get("hard_close_at_ny_open", True)) else None
        out.append(
            RegimeOwnedCandidate(
                strategy="london_v2",
                session_owner="london",
                side=direction,
                signal_time=ts,
                execute_time=next_ts,
                entry_price_basis="next_bar_open",
                sl_pips=float(sl_pips),
                raw_quality=float(raw_quality),
                quality_normalized=float(normalized),
                regime_label=regime_label,
                regime_margin=float(regime_margin),
                ownership_cell=ownership_cell,
                reason=f"setup_{setup}",
                source_features={
                    "setup_type": setup,
                    "asian_range_pips": pe.get("asian_range_pips"),
                    "lor_range_pips": pe.get("lor_range_pips"),
                    "is_reentry": bool(pe.get("is_reentry", False)),
                    "l1_defended_overlay": bool(setup == "D" and l1_exit_override),
                    "tp1_offset_pips": float(abs(tp1 - entry_price) / PIP_SIZE),
                    "tp2_offset_pips": float(abs(tp2 - entry_price) / PIP_SIZE),
                },
                execution_plan=UnifiedExecutionPlan(
                    sl_price=float(sl_price),
                    tp1_price=float(tp1),
                    tp2_price=float(tp2),
                    tp1_close_fraction=float(s_cfg["tp1_close_fraction"]),
                    be_offset_pips=float(be_offset_pips),
                    trail_activate_pips=float(sl_pips),
                    trail_distance_pips=float(max(2.0, sl_pips * 0.6)),
                    trail_requires_tp1=True,
                    session_close_time=session_close,
                ),
                risk_pct=float(s_cfg.get("risk_per_trade_pct", london_cfg.get("risk", {}).get("risk_per_trade_pct", 0.01))),
                max_spread_pips=float(london_cfg.get("execution_model", {}).get("spread_max_pips", 3.5)),
                source_family="london",
                source_setup=setup,
            )
        )
    return out


def _generate_v44_candidate(
    *,
    state: PortfolioBacktestState,
    ts: pd.Timestamp,
    next_ts: pd.Timestamp,
    v44_row: pd.Series,
    regime_label: str,
    regime_margin: float,
    ownership_cell: str,
    v44_cfg: dict[str, Any],
    defended_overlays: Optional[dict[str, Any]] = None,
) -> Optional[RegimeOwnedCandidate]:
    day_key = str(ts.date())
    _next_day_reset_v44(state, day_key)
    h = _session_hour(ts)
    session = None
    if float(v44_cfg.get("london_start", 8.5)) <= h < float(v44_cfg.get("london_end", 11.0)):
        session = "london"
    elif float(v44_cfg.get("ny_start", 13.0)) <= h < float(v44_cfg.get("ny_end", 16.0)):
        session = "ny"
    if session is None:
        return None
    session_state = state.v44_session_state[session]
    if int(v44_cfg.get("v5_max_entries_day", 7)) > 0 and int(session_state.get("trade_count", 0)) >= int(v44_cfg.get("v5_max_entries_day", 7)):
        return None
    if int(session_state.get("consecutive_losses", 0)) >= int(v44_cfg.get("v5_session_stop_losses", 3)):
        return None
    cooldown_until = session_state.get("cooldown_until")
    if cooldown_until and ts < pd.Timestamp(cooldown_until):
        return None
    if not bool(v44_row.get("v44_atr_ok", True)):
        return None
    trend = str(v44_row.get("v44_h1_trend") or "")
    ema_fast = _safe_float(v44_row.get("v44_ema_fast"), np.nan)
    ema_slow = _safe_float(v44_row.get("v44_ema_slow"), np.nan)
    slope = _safe_float(v44_row.get("v44_slope"), 0.0)
    body_pips = _safe_float(v44_row.get("v44_body_pips"), 0.0)
    if not np.isfinite(ema_fast) or not np.isfinite(ema_slow):
        return None
    strong_threshold = float(v44_cfg.get("v5_strong_slope", 0.5))
    weak_threshold = float(v44_cfg.get("v5_weak_slope", 0.2))
    strength = "strong" if abs(slope) > strong_threshold else "normal" if abs(slope) > weak_threshold else "weak"
    min_body = float(v44_cfg.get("v5_entry_min_body_pips", 1.5))
    bullish_bar = _safe_float(v44_row.get("v44_m5_close")) > _safe_float(v44_row.get("v44_m5_open")) and body_pips >= min_body
    bearish_bar = _safe_float(v44_row.get("v44_m5_close")) < _safe_float(v44_row.get("v44_m5_open")) and body_pips >= min_body
    side = None
    reason = "v44: directional conditions not met"
    if trend == "up" and ema_fast > ema_slow and bullish_bar and slope > 0:
        side = "buy"
        reason = "v44: H1 up + M5 strong bullish momentum"
    elif trend == "down" and ema_fast < ema_slow and bearish_bar and slope < 0:
        side = "sell"
        reason = "v44: H1 down + M5 strong bearish momentum"
    if side is None:
        return None
    defensive_veto = dict((defended_overlays or {}).get("defensive_veto") or {})
    if (
        defensive_veto
        and str(defensive_veto.get("strategy")) == "v44_ny"
        and str(defensive_veto.get("mode", "block")) == "block"
        and ownership_cell == str(defensive_veto.get("ownership_cell"))
    ):
        return None
    if regime_label in {"breakout", "post_breakout_trend"} and session == "ny":
        return None
    direction = str(side)
    open_mid = _safe_float(v44_row.get("v44_m5_close"), _safe_float(v44_row.get("close")))
    spread_pips = float(v44_cfg.get("spread_pips", 2.0))
    entry_price = _entry_price_from_bar(direction, open_mid, spread_pips, _slippage_pips_placeholder("v44"))
    if direction == "buy":
        raw_sl = _safe_float(v44_row.get("v44_sl_low_6"), np.nan) - 1.5 * PIP_SIZE
        raw_sl_pips = (entry_price - raw_sl) / PIP_SIZE if np.isfinite(raw_sl) else 7.0
    else:
        raw_sl = _safe_float(v44_row.get("v44_sl_high_6"), np.nan) + 1.5 * PIP_SIZE
        raw_sl_pips = (raw_sl - entry_price) / PIP_SIZE if np.isfinite(raw_sl) else 7.0
    sl_pips = max(float(v44_cfg.get("v5_sl_floor_pips", 7.0)), min(float(v44_cfg.get("v5_sl_cap_pips", 9.0)), raw_sl_pips))
    sl_price = entry_price - sl_pips * PIP_SIZE if direction == "buy" else entry_price + sl_pips * PIP_SIZE
    if strength == "strong":
        tp1_r = float(v44_cfg.get("v5_strong_tp1", 2.0))
        tp2_pips = float(v44_cfg.get("v5_strong_tp2", 5.0))
        tp1_close = float(v44_cfg.get("v5_strong_tp1_close_pct", 0.3))
        trail = float(v44_cfg.get("v5_strong_trail_buffer_pips", v44_cfg.get("v5_strong_trail_buffer", 4.0)))
        trail_ema_field = "v44_trail_ema_21"
    elif strength == "normal":
        tp1_r = float(v44_cfg.get("v5_normal_tp1", 1.75))
        tp2_pips = float(v44_cfg.get("v5_normal_tp2", 3.0))
        tp1_close = float(v44_cfg.get("v5_normal_tp1_close_pct", 0.5))
        trail = float(v44_cfg.get("v5_normal_trail_buffer", 3.0))
        trail_ema_field = "v44_trail_ema_9"
    else:
        tp1_r = float(v44_cfg.get("v5_weak_tp1", 1.2))
        tp2_pips = float(v44_cfg.get("v5_weak_tp2", 2.0))
        tp1_close = float(v44_cfg.get("v5_weak_tp1_close_pct", 0.6))
        trail = float(v44_cfg.get("v5_weak_trail_buffer", 2.0))
        trail_ema_field = "v44_trail_ema_9"
    tp1_price = entry_price + tp1_r * sl_pips * PIP_SIZE if direction == "buy" else entry_price - tp1_r * sl_pips * PIP_SIZE
    tp2_price = entry_price + tp2_pips * PIP_SIZE if direction == "buy" else entry_price - tp2_pips * PIP_SIZE
    raw_quality = _compute_raw_quality_v44(direction, strength, reason, session)
    normalized = _normalize_quality(state, "v44_ny", raw_quality)
    close_hour = float(v44_cfg.get("ny_end", 16.0) if session == "ny" else v44_cfg.get("london_end", 11.0))
    ch = int(close_hour)
    cm = int(round((close_hour - ch) * 60.0))
    session_close = pd.Timestamp(next_ts).normalize() + pd.Timedelta(hours=ch, minutes=cm)
    return RegimeOwnedCandidate(
        strategy="v44_ny",
        session_owner=session,
        side=direction,
        signal_time=ts,
        execute_time=next_ts,
        entry_price_basis="next_bar_open",
        sl_pips=float(sl_pips),
        raw_quality=float(raw_quality),
        quality_normalized=float(normalized),
        regime_label=regime_label,
        regime_margin=float(regime_margin),
        ownership_cell=ownership_cell,
        reason=reason,
        source_features={"strength": strength, "session": session, "tp1_offset_pips": float(abs(tp1_price - entry_price) / PIP_SIZE), "tp2_offset_pips": float(abs(tp2_price - entry_price) / PIP_SIZE)},
        execution_plan=UnifiedExecutionPlan(
            sl_price=float(sl_price),
            tp1_price=float(tp1_price),
            tp2_price=float(tp2_price),
            tp1_close_fraction=float(tp1_close),
            be_offset_pips=float(v44_cfg.get("v5_be_offset", 0.5)),
            trail_activate_pips=float(tp1_r * sl_pips + float(v44_cfg.get("v5_trail_start_after_tp1_mult", 0.5)) * sl_pips),
            trail_distance_pips=float(trail),
            trail_requires_tp1=True,
            session_close_time=session_close,
            stale_bars=int(v44_cfg.get("v5_stale_exit_bars", 0) or 0),
            stale_exit_underwater_pct=float(v44_cfg.get("v5_stale_exit_underwater_pct", 0.0) or 0.0),
            trail_reference="ema",
            trail_ema_field=trail_ema_field,
        ),
        risk_pct=float(v44_cfg.get("v5_news_trend_risk_pct", v44_cfg.get("v5_risk_per_trade_pct", 0.5))) / 100.0,
        max_spread_pips=float(v44_cfg.get("v5_max_entry_spread_pips", v44_cfg.get("max_entry_spread_pips", 3.0))),
        source_family="v44",
    )


def _route_candidates(
    *,
    ts: pd.Timestamp,
    regime_label: str,
    regime_margin: float,
    ownership_cell: str,
    candidates: list[RegimeOwnedCandidate],
    quality_margin_epsilon: float = 0.05,
    high_confidence_regime_margin: float = 0.75,
) -> RegimeOwnedDecision:
    if not candidates:
        return RegimeOwnedDecision(
            bar_time=ts,
            regime_label=regime_label,
            regime_margin=regime_margin,
            ownership_cell=ownership_cell,
            candidate_count=0,
            candidates=[],
            winner_strategy=None,
            winner_side=None,
            winner_reason="no_candidates",
            no_trade_reason="no_candidates",
        )
    scored: list[tuple[float, RegimeOwnedCandidate]] = []
    for c in candidates:
        adj = float(c.quality_normalized) + _regime_owner_bonus(c.strategy, regime_label, regime_margin)
        if c.strategy == "v44_ny" and regime_label == "momentum" and regime_margin >= high_confidence_regime_margin:
            adj += 0.05
        scored.append((adj, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    top_score, top = scored[0]
    if len(scored) > 1 and (top_score - scored[1][0]) < quality_margin_epsilon:
        return RegimeOwnedDecision(
            bar_time=ts,
            regime_label=regime_label,
            regime_margin=regime_margin,
            ownership_cell=ownership_cell,
            candidate_count=len(candidates),
            candidates=[
                {
                    "strategy": c.strategy,
                    "side": c.side,
                    "quality_normalized": c.quality_normalized,
                    "raw_quality": c.raw_quality,
                    "reason": c.reason,
                    "session_owner": c.session_owner,
                    "ownership_cell": c.ownership_cell,
                }
                for _, c in scored
            ],
            winner_strategy=None,
            winner_side=None,
            winner_reason="quality_tie_no_trade",
            no_trade_reason="quality_tie_no_trade",
        )
    return RegimeOwnedDecision(
        bar_time=ts,
        regime_label=regime_label,
        regime_margin=regime_margin,
        ownership_cell=ownership_cell,
        candidate_count=len(candidates),
        candidates=[
            {
                "strategy": c.strategy,
                "side": c.side,
                "quality_normalized": c.quality_normalized,
                "raw_quality": c.raw_quality,
                "reason": c.reason,
                "session_owner": c.session_owner,
                "ownership_cell": c.ownership_cell,
            }
            for _, c in scored
        ],
        winner_strategy=top.strategy,
        winner_side=top.side,
        winner_reason="router_winner",
        no_trade_reason=None,
    )


def _admit_and_place_candidate(
    *,
    state: PortfolioBacktestState,
    candidate: RegimeOwnedCandidate,
    open_mid: float,
    ts: pd.Timestamp,
    bar_index: int,
    tokyo_cfg: dict[str, Any],
    london_cfg: dict[str, Any],
    v44_cfg: dict[str, Any],
) -> tuple[bool, str]:
    spread_pips = _session_spread_pips(ts, candidate.source_family, bar_index, london_cfg, v44_cfg, tokyo_cfg)
    if spread_pips > candidate.max_spread_pips:
        return False, "spread_block"
    slippage = _slippage_pips(candidate)
    entry_price = _entry_price_from_bar(candidate.side, open_mid, spread_pips, slippage)
    risk_usd = state.equity * float(candidate.risk_pct)
    units = int(math.floor(risk_usd / max(1e-9, candidate.sl_pips * _pip_value_per_unit(entry_price)) / ROUND_UNITS) * ROUND_UNITS)
    if units <= 0:
        return False, "zero_units"
    leverage = 33.3
    req_margin = (units * float(entry_price) * PIP_SIZE) / leverage
    used_margin = float(sum(p.margin_required_usd for p in state.open_positions))
    free_margin = max(0.0, state.equity - used_margin)
    if req_margin > 0.5 * free_margin:
        state.diagnostics["margin_rejects"] = int(state.diagnostics.get("margin_rejects", 0)) + 1
        return False, "margin_reject"
    state.trade_id_seq += 1
    plan = UnifiedExecutionPlan(**asdict(candidate.execution_plan))
    # re-anchor stop/tp to realized entry when needed
    tp1_offset_pips = _safe_float(candidate.source_features.get("tp1_offset_pips"), 0.0)
    tp2_offset_pips = _safe_float(candidate.source_features.get("tp2_offset_pips"), 0.0)
    if candidate.side == "buy":
        plan.sl_price = entry_price - candidate.sl_pips * PIP_SIZE
        if plan.tp1_price is not None and tp1_offset_pips > 0:
            plan.tp1_price = entry_price + tp1_offset_pips * PIP_SIZE
        if plan.tp2_price is not None and tp2_offset_pips > 0:
            plan.tp2_price = entry_price + tp2_offset_pips * PIP_SIZE
    else:
        plan.sl_price = entry_price + candidate.sl_pips * PIP_SIZE
        if plan.tp1_price is not None and tp1_offset_pips > 0:
            plan.tp1_price = entry_price - tp1_offset_pips * PIP_SIZE
        if plan.tp2_price is not None and tp2_offset_pips > 0:
            plan.tp2_price = entry_price - tp2_offset_pips * PIP_SIZE
    state.open_positions.append(
        UnifiedPositionState(
            trade_id=state.trade_id_seq,
            strategy=candidate.strategy,
            source_family=candidate.source_family,
            session_owner=candidate.session_owner,
            source_setup=candidate.source_setup,
            side=candidate.side,
            entry_time=ts,
            entry_price=float(entry_price),
            initial_units=units,
            remaining_units=units,
            planned_risk_usd=float(risk_usd),
            margin_required_usd=float(req_margin),
            execution_plan=plan,
            raw={
                "reason": candidate.reason,
                "ownership_cell": candidate.ownership_cell,
                "regime_label": candidate.regime_label,
                "tokyo_session_day": str(candidate.signal_time.tz_convert(tokyo_engine.TOKYO_TZ).date()) if candidate.strategy == "v14" else None,
                "cooldown_minutes": int(tokyo_cfg.get("trade_management", {}).get("cooldown_minutes", 0)) if candidate.strategy == "v14" else None,
                "same_direction_stop_minutes": int(tokyo_cfg.get("trade_management", {}).get("no_reentry_same_direction_after_stop_minutes", 0)) if candidate.strategy == "v14" else None,
                **candidate.source_features,
            },
        )
    )
    state.diagnostics["max_concurrent_positions"] = max(int(state.diagnostics.get("max_concurrent_positions", 0)), len(state.open_positions))
    if candidate.strategy == "v14":
        sday = str(candidate.signal_time.tz_convert(tokyo_engine.TOKYO_TZ).date())
        if sday in state.tokyo_session_state:
            state.tokyo_session_state[sday]["trades"] += 1
    elif candidate.strategy == "v44_ny":
        if state.v44_session_state:
            state.v44_session_state[candidate.session_owner]["trade_count"] += 1
    elif candidate.strategy == "london_v2":
        day_key = str(pd.Timestamp(candidate.signal_time).floor("D").date())
        dst = state.london_day_state.get(day_key)
        if dst is not None:
            dst["entries_total"] = int(dst.get("entries_total", 0)) + 1
            if candidate.source_setup is not None:
                dst["entries_setup"][candidate.source_setup] = int(dst["entries_setup"].get(candidate.source_setup, 0)) + 1
    return True, "placed"


def run_regime_owned_backtest(
    *,
    input_csv: str,
    tokyo_config_path: str,
    london_config_path: str,
    v44_config_path: str,
    start_equity: float = 100000.0,
    defended_overlays: Optional[dict[str, Any]] = None,
    variant_name: Optional[str] = None,
) -> BacktestResult:
    tokyo_cfg = _read_json(tokyo_config_path)
    london_cfg = london_engine.merge_config(_read_json(london_config_path))
    v44_cfg = _read_json(v44_config_path)
    base_m1 = tokyo_engine.load_m1(input_csv)
    tokyo_df = _build_tokyo_frame(input_csv, tokyo_cfg)
    classified = _build_regime_frame(base_m1)
    v44_frame = _build_v44_feature_frame(base_m1, v44_cfg)
    tokyo_cfg_params = _build_tokyo_cfg_params(tokyo_cfg)
    state = PortfolioBacktestState(
        equity=float(start_equity),
        peak_equity=float(start_equity),
        diagnostics={"margin_rejects": 0, "max_concurrent_positions": 0, "bars_processed": 0},
    )

    day_groups = {k: g.copy().reset_index(drop=True) for k, g in base_m1.assign(day_utc=base_m1["time"].dt.floor("D")).groupby("day_utc")}
    base_time = base_m1["time"].tolist()
    base_open = base_m1["open"].to_numpy(dtype=float)
    base_high = base_m1["high"].to_numpy(dtype=float)
    base_low = base_m1["low"].to_numpy(dtype=float)
    base_close = base_m1["close"].to_numpy(dtype=float)
    regime_label_arr = classified["regime_hysteresis"].fillna("ambiguous").astype(str).tolist()
    regime_margin_arr = classified["score_margin"].fillna(0.0).to_numpy(dtype=float)
    er_arr = classified["er_m5"].where(classified["er_m5"].notna(), classified["dir_efficiency"]).fillna(0.5).to_numpy(dtype=float)
    der_arr = classified["delta_er_m5"].fillna(0.0).to_numpy(dtype=float)
    ownership_cells = [cell_key(lbl, er_bucket(float(er)), der_bucket(float(der))) for lbl, er, der in zip(regime_label_arr, er_arr, der_arr)]
    v44_trail_ema9 = v44_frame["v44_trail_ema_9"].to_numpy(dtype=float)
    v44_trail_ema21 = v44_frame["v44_trail_ema_21"].to_numpy(dtype=float)
    tokyo_matrix = tokyo_df.to_numpy(copy=False)
    tokyo_index = {str(col): idx for idx, col in enumerate(tokyo_df.columns)}
    v44_matrix = v44_frame.to_numpy(copy=False)
    v44_index = {str(col): idx for idx, col in enumerate(v44_frame.columns)}

    for i in range(len(base_m1) - 1):
        ts = pd.Timestamp(base_time[i])
        next_ts = pd.Timestamp(base_time[i + 1])
        state.diagnostics["bars_processed"] += 1
        regime_label = regime_label_arr[i]
        regime_margin = float(regime_margin_arr[i])
        ownership_cell = ownership_cells[i]
        base_row = {"time": ts, "open": base_open[i], "high": base_high[i], "low": base_low[i], "close": base_close[i]}
        v44_row = IndexedRow(v44_matrix[i], v44_index)

        # execute orders scheduled for this bar open
        due = [c for c in state.pending_orders if pd.Timestamp(c.execute_time) == ts]
        state.pending_orders = [c for c in state.pending_orders if pd.Timestamp(c.execute_time) != ts]
        for cand in due:
            ok, why = _admit_and_place_candidate(
                state=state,
                candidate=cand,
                open_mid=float(base_open[i]),
                ts=ts,
                bar_index=i,
                tokyo_cfg=tokyo_cfg,
                london_cfg=london_cfg,
                v44_cfg=v44_cfg,
            )
            if not ok:
                state.diagnostics[f"admission_{why}"] = int(state.diagnostics.get(f"admission_{why}", 0)) + 1

        _manage_open_positions(
            state=state,
            open_mid=float(base_open[i]),
            high_mid=float(base_high[i]),
            low_mid=float(base_low[i]),
            close_mid=float(base_close[i]),
            bar_index=i,
            ts=ts,
            regime_label=regime_label,
            ownership_cell=ownership_cell,
            london_cfg=london_cfg,
            v44_cfg=v44_cfg,
            tokyo_cfg=tokyo_cfg,
            v44_trail_ema_9=float(v44_trail_ema9[i]) if np.isfinite(v44_trail_ema9[i]) else None,
            v44_trail_ema_21=float(v44_trail_ema21[i]) if np.isfinite(v44_trail_ema21[i]) else None,
        )

        candidates: list[RegimeOwnedCandidate] = []
        tokyo_cand = _generate_tokyo_candidate(
            state=state,
            row=IndexedRow(tokyo_matrix[i], tokyo_index),
            next_ts=next_ts,
            regime_label=regime_label,
            regime_margin=regime_margin,
            ownership_cell=ownership_cell,
            tokyo_cfg=tokyo_cfg,
            tokyo_cfg_params=tokyo_cfg_params,
            bar_index=i,
        )
        if tokyo_cand is not None:
            candidates.append(tokyo_cand)

        day = ts.floor("D")
        day_df = day_groups.get(day)
        if day_df is not None:
            london_day = _ensure_london_day_state(state, day, london_cfg, day_df)
            london_candidates = _generate_london_candidates(
                state=state,
                row=base_row,
                next_ts=next_ts,
                day_state=london_day,
                regime_label=regime_label,
                regime_margin=regime_margin,
                ownership_cell=ownership_cell,
                london_cfg=london_cfg,
                bar_index=i,
                defended_overlays=defended_overlays,
            )
            candidates.extend(london_candidates)

        v44_cand = _generate_v44_candidate(
            state=state,
            ts=ts,
            next_ts=next_ts,
            v44_row=v44_row,
            regime_label=regime_label,
            regime_margin=regime_margin,
            ownership_cell=ownership_cell,
            v44_cfg=v44_cfg,
            defended_overlays=defended_overlays,
        )
        if v44_cand is not None:
            candidates.append(v44_cand)

        decision = _route_candidates(
            ts=ts,
            regime_label=regime_label,
            regime_margin=regime_margin,
            ownership_cell=ownership_cell,
            candidates=candidates,
        )
        winner = None
        if decision.winner_strategy is not None:
            for cand in candidates:
                if cand.strategy == decision.winner_strategy and cand.side == decision.winner_side:
                    winner = cand
                    break
        if winner is not None:
            state.pending_orders.append(winner)
        state.decision_log.append(
            {
                "bar_time": ts,
                "regime_label": regime_label,
                "regime_margin": regime_margin,
                "ownership_cell": ownership_cell,
                "candidate_count": len(candidates),
                "winner_strategy": decision.winner_strategy,
                "winner_side": decision.winner_side,
                "winner_reason": decision.winner_reason,
                "no_trade_reason": decision.no_trade_reason,
                "candidate_summary": json.dumps(decision.candidates, default=_json_default, separators=(",", ":")),
            }
        )

    # final forced close on last bar
    if state.open_positions:
        last_ts = pd.Timestamp(base_time[-1])
        regime_label = regime_label_arr[-1]
        ownership_cell = ownership_cells[-1]
        for p in state.open_positions:
            spread_pips = _session_spread_pips(last_ts, p.source_family, len(base_m1) - 1, london_cfg, v44_cfg, tokyo_cfg)
            bid_o, ask_o = _to_bid_ask(float(base_close[-1]), spread_pips)
            exit_px = bid_o if p.side == "buy" else ask_o
            _close_position_leg(p, exit_px, p.remaining_units)
            p.exit_reason = "END_OF_TEST"
            p.exit_time = last_ts
            p.exit_price_last = exit_px
            _finalize_closed_position(state, p, regime_label, ownership_cell)
        state.open_positions = []

    closed_df = pd.DataFrame(state.closed_trades)
    equity_df = pd.DataFrame(state.equity_curve)
    decision_df = pd.DataFrame(state.decision_log)
    diagnostics = dict(state.diagnostics)
    diagnostics["ending_equity"] = state.equity
    summary = _build_summary(
        trades_df=closed_df,
        equity_df=equity_df,
        decision_df=decision_df,
        diagnostics=diagnostics,
        start_equity=float(start_equity),
    )
    if variant_name:
        summary["variant_name"] = str(variant_name)
    if defended_overlays:
        summary["defended_overlays"] = defended_overlays
    return BacktestResult(
        summary=summary,
        closed_trades=closed_df,
        equity_curve=equity_df,
        decision_log=decision_df,
        diagnostics=diagnostics,
    )


def write_backtest_package(result: BacktestResult, out_prefix: str | Path) -> dict[str, str]:
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = out_prefix.with_suffix(".summary.json")
    trades_path = out_prefix.with_suffix(".closed_trades.csv")
    equity_path = out_prefix.with_suffix(".equity.csv")
    decisions_path = out_prefix.with_suffix(".decision_log.csv")
    diagnostics_path = out_prefix.with_suffix(".diagnostics.json")
    summary_path.write_text(json.dumps(result.summary, indent=2, default=_json_default), encoding="utf-8")
    diagnostics_path.write_text(json.dumps(result.diagnostics, indent=2, default=_json_default), encoding="utf-8")
    result.closed_trades.to_csv(trades_path, index=False)
    result.equity_curve.to_csv(equity_path, index=False)
    result.decision_log.to_csv(decisions_path, index=False)
    return {
        "summary": str(summary_path),
        "closed_trades": str(trades_path),
        "equity": str(equity_path),
        "decision_log": str(decisions_path),
        "diagnostics": str(diagnostics_path),
    }
