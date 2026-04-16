"""Autonomous Fillmore — lets the AI place trades on its own, gated by a code-side
signal filter so we don't spam the LLM.

Architecture (three layers):

    tick →  L1 hard filters  → L2 signal gate (mode-dependent) → L3 adaptive throttle
                 │                       │                           │
                 └── block early ────────┴── decides whether to wake up the LLM ──┘

Only when all three layers pass do we call OpenAI via the existing
`ai_suggest_trade` endpoint logic, and if the model returns confidence>=min we
auto-place the limit through `place_limit_order_endpoint`.

All config and the rolling decisions log live in `runtime_state.json` under the
`autonomous_fillmore` key so the run loop and the HTTP API share one source of
truth without needing a new DB migration.

See MEMORY.md ("Autonomous Fillmore" planning notes) for the design discussion.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Literal, Optional

import pandas as pd

# -----------------------------------------------------------------------------
# Config defaults + types
# -----------------------------------------------------------------------------

AutonomousMode = Literal["off", "shadow", "paper", "live"]
Aggressiveness = Literal["conservative", "balanced", "aggressive", "very_aggressive"]
OrderType = Literal["market", "limit"]

# Rough cost per 1M tokens (input/output) for the default model. Used to estimate
# budget spend from tokens — not authoritative, just directional.
# gpt-5.4-mini is priced similarly to gpt-5-mini / gpt-4o-mini tier.
_MODEL_COST_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-5.4-mini": (0.15, 0.60),
    "gpt-5.4":      (2.50, 10.00),
    "gpt-5-mini":   (0.15, 0.60),
    "gpt-4o-mini":  (0.15, 0.60),
    "gpt-4o":       (2.50, 10.00),
}

# Estimated prompt/response sizes for the autonomous call. These are
# pessimistic-ish so the budget stats trend slightly conservative.
_ASSUMED_INPUT_TOKENS = 4000
_ASSUMED_OUTPUT_TOKENS = 350

DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": False,
    "mode": "off",                      # off|shadow|paper|live — paper = real OANDA practice orders (not in-app sim)
    "aggressiveness": "balanced",
    "order_type": "market",             # market|limit — market = fill NOW at bid/ask
    "daily_budget_usd": 2.00,
    "min_llm_cooldown_sec": 60,
    "trading_hours": {
        "tokyo": True,
        "london": True,
        "ny": True,
    },
    "max_lots_per_trade": 15.0,          # matches manual Fillmore (1-15 lot guidance)
    "max_open_ai_trades": 2,
    "max_daily_loss_usd": 50.0,
    "max_consecutive_errors": 5,
    "min_confidence": "medium",         # low|medium|high — below this we skip placing
    "model": "gpt-5.4-mini",
    # adaptive throttle
    "throttle_no_trade_streak": 5,       # after N "no trade" LLM replies, pause
    "throttle_no_trade_cooldown_sec": 300,
    "throttle_loss_streak": 2,
    "throttle_loss_cooldown_sec": 900,
}

# Per-mode gate thresholds. Lower = easier to pass = more LLM calls.
# These are the *signal* filters applied after hard filters pass.
GATE_THRESHOLDS: dict[str, dict[str, Any]] = {
    "conservative": {
        "description": "M3 trend + M1 EMA stack + (pullback OR zone entry) + daily H/L buffer",
        "require_m3_trend": True,
        "require_m1_stack": True,
        "require_pullback_or_zone": True,
        "require_daily_hl_buffer_pips": 5.0,
        "expected_pass_rate_pct": 1.5,
    },
    "balanced": {
        "description": "M3 trend aligned + M1 EMA stack",
        "require_m3_trend": True,
        "require_m1_stack": True,
        "require_pullback_or_zone": False,
        "require_daily_hl_buffer_pips": 0.0,
        "expected_pass_rate_pct": 4.0,
    },
    "aggressive": {
        "description": "M1 EMA stack OR M3 trend signal",
        "require_m3_trend": False,
        "require_m1_stack": False,
        "require_pullback_or_zone": False,
        "require_daily_hl_buffer_pips": 0.0,
        "require_any_trend_signal": True,
        "expected_pass_rate_pct": 10.0,
    },
    "very_aggressive": {
        "description": "Hard filters only (spread/cooldown/budget)",
        "require_m3_trend": False,
        "require_m1_stack": False,
        "require_pullback_or_zone": False,
        "require_daily_hl_buffer_pips": 0.0,
        "require_any_trend_signal": False,
        "expected_pass_rate_pct": 27.0,
    },
}


@dataclass
class GateDecision:
    """One gate evaluation. Logged to the rolling decisions list."""
    timestamp_utc: str
    result: Literal["pass", "block"]
    layer: Literal["hard", "signal", "throttle", "pass"]
    reason: str
    mode: str
    aggressiveness: str
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "t": self.timestamp_utc,
            "r": self.result,
            "l": self.layer,
            "why": self.reason,
            "mode": self.mode,
            "agg": self.aggressiveness,
            **({"x": self.extras} if self.extras else {}),
        }


# -----------------------------------------------------------------------------
# Config load / save (merged with runtime_state.json)
# -----------------------------------------------------------------------------

def _load_state(state_path: Path) -> dict[str, Any]:
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _save_state(state_path: Path, data: dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def get_config(state_path: Path) -> dict[str, Any]:
    state = _load_state(state_path)
    cfg = dict(DEFAULT_CONFIG)
    saved = state.get("autonomous_fillmore") or {}
    saved_cfg = saved.get("config") or {}
    for k, v in saved_cfg.items():
        if k in cfg:
            # Shallow-merge trading_hours so partial updates keep siblings.
            if k == "trading_hours" and isinstance(v, dict) and isinstance(cfg[k], dict):
                merged = dict(cfg[k])
                merged.update(v)
                cfg[k] = merged
            else:
                cfg[k] = v
    return cfg


def set_config(state_path: Path, patch: dict[str, Any]) -> dict[str, Any]:
    """Merge-update the autonomous config. Returns the new effective config."""
    state = _load_state(state_path)
    auto = state.get("autonomous_fillmore") or {}
    cur = auto.get("config") or {}
    # Validate the patch against known keys (silently drop unknowns).
    valid_keys = set(DEFAULT_CONFIG.keys())
    clean: dict[str, Any] = {}
    for k, v in patch.items():
        if k not in valid_keys:
            continue
        clean[k] = v
    # Type coerce a few fields
    if "aggressiveness" in clean:
        agg = str(clean["aggressiveness"]).lower()
        if agg not in GATE_THRESHOLDS:
            agg = "balanced"
        clean["aggressiveness"] = agg
    if "mode" in clean:
        m = str(clean["mode"]).lower()
        if m not in ("off", "shadow", "paper", "live"):
            m = "off"
        clean["mode"] = m
    if "min_confidence" in clean:
        mc = str(clean["min_confidence"]).lower()
        if mc not in ("low", "medium", "high"):
            mc = "medium"
        clean["min_confidence"] = mc
    if "order_type" in clean:
        ot = str(clean["order_type"]).lower()
        if ot not in ("market", "limit"):
            ot = "market"
        clean["order_type"] = ot
    for numkey in (
        "daily_budget_usd", "min_llm_cooldown_sec", "max_lots_per_trade",
        "max_open_ai_trades", "max_daily_loss_usd", "max_consecutive_errors",
        "throttle_no_trade_streak", "throttle_no_trade_cooldown_sec",
        "throttle_loss_streak", "throttle_loss_cooldown_sec",
    ):
        if numkey in clean:
            try:
                clean[numkey] = float(clean[numkey]) if "." in str(clean[numkey]) or numkey.endswith("usd") or numkey.endswith("lots_per_trade") else int(clean[numkey])
            except (TypeError, ValueError):
                clean.pop(numkey, None)
    merged = {**cur, **clean}
    auto["config"] = merged
    state["autonomous_fillmore"] = auto
    _save_state(state_path, state)
    return get_config(state_path)


# -----------------------------------------------------------------------------
# Decisions log + runtime counters
# -----------------------------------------------------------------------------

_DECISIONS_MAX = 500  # rolling cap


def _runtime_block(state: dict[str, Any]) -> dict[str, Any]:
    auto = state.setdefault("autonomous_fillmore", {})
    runtime = auto.setdefault("runtime", {
        "decisions": [],             # rolling GateDecision dicts
        "last_llm_call_utc": None,
        "llm_calls_today": 0,
        "llm_spend_today_usd": 0.0,
        "spend_day_utc": None,
        "trades_placed_today": 0,
        "consecutive_errors": 0,
        "consecutive_no_trade_replies": 0,
        "consecutive_losses": 0,
        "throttle_until_utc": None,
        "throttle_reason": None,
        "last_suggestion_id": None,
        "last_placed_order_id": None,
        "daily_pnl_usd": 0.0,
        "pnl_day_utc": None,
    })
    return runtime


def log_decision(state_path: Path, decision: GateDecision) -> None:
    state = _load_state(state_path)
    rt = _runtime_block(state)
    lst = list(rt.get("decisions") or [])
    lst.append(decision.to_dict())
    if len(lst) > _DECISIONS_MAX:
        lst = lst[-_DECISIONS_MAX:]
    rt["decisions"] = lst
    _save_state(state_path, state)


def _today_utc_key() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _rollover_daily_counters(rt: dict[str, Any]) -> None:
    today = _today_utc_key()
    if rt.get("spend_day_utc") != today:
        rt["spend_day_utc"] = today
        rt["llm_spend_today_usd"] = 0.0
        rt["llm_calls_today"] = 0
        rt["trades_placed_today"] = 0
    if rt.get("pnl_day_utc") != today:
        rt["pnl_day_utc"] = today
        rt["daily_pnl_usd"] = 0.0


def _estimated_call_cost_usd(model: str) -> float:
    inp_per_1m, out_per_1m = _MODEL_COST_PER_1M.get(model, (0.15, 0.60))
    return (inp_per_1m * _ASSUMED_INPUT_TOKENS + out_per_1m * _ASSUMED_OUTPUT_TOKENS) / 1_000_000.0


def record_llm_invocation(state_path: Path, model: str) -> None:
    state = _load_state(state_path)
    rt = _runtime_block(state)
    _rollover_daily_counters(rt)
    rt["last_llm_call_utc"] = datetime.now(timezone.utc).isoformat()
    rt["llm_calls_today"] = int(rt.get("llm_calls_today") or 0) + 1
    rt["llm_spend_today_usd"] = float(rt.get("llm_spend_today_usd") or 0.0) + _estimated_call_cost_usd(model)
    _save_state(state_path, state)


def record_trade_placed(state_path: Path, order_id: Any, suggestion_id: Optional[str]) -> None:
    state = _load_state(state_path)
    rt = _runtime_block(state)
    _rollover_daily_counters(rt)
    rt["trades_placed_today"] = int(rt.get("trades_placed_today") or 0) + 1
    rt["last_placed_order_id"] = str(order_id) if order_id is not None else None
    rt["last_suggestion_id"] = suggestion_id
    rt["consecutive_errors"] = 0
    rt["consecutive_no_trade_replies"] = 0
    _save_state(state_path, state)


def record_no_trade_reply(state_path: Path, cfg: dict[str, Any]) -> None:
    state = _load_state(state_path)
    rt = _runtime_block(state)
    rt["consecutive_no_trade_replies"] = int(rt.get("consecutive_no_trade_replies") or 0) + 1
    streak = int(rt["consecutive_no_trade_replies"])
    if streak >= int(cfg.get("throttle_no_trade_streak") or 5):
        until = datetime.now(timezone.utc) + timedelta(seconds=int(cfg.get("throttle_no_trade_cooldown_sec") or 300))
        rt["throttle_until_utc"] = until.isoformat()
        rt["throttle_reason"] = f"no_trade_streak={streak}"
    _save_state(state_path, state)


def record_error(state_path: Path, cfg: dict[str, Any], err_msg: str) -> None:
    state = _load_state(state_path)
    rt = _runtime_block(state)
    rt["consecutive_errors"] = int(rt.get("consecutive_errors") or 0) + 1
    if rt["consecutive_errors"] >= int(cfg.get("max_consecutive_errors") or 5):
        # Kill switch — flip autonomous off.
        auto = state.setdefault("autonomous_fillmore", {})
        cur_cfg = auto.setdefault("config", {})
        cur_cfg["mode"] = "off"
        cur_cfg["enabled"] = False
        rt["throttle_reason"] = f"error_kill_switch: {err_msg[:120]}"
    _save_state(state_path, state)


def record_trade_outcome(state_path: Path, pnl_usd: float, cfg: dict[str, Any]) -> None:
    """Called from trade-close path when an ai_manual trade closes."""
    state = _load_state(state_path)
    rt = _runtime_block(state)
    _rollover_daily_counters(rt)
    rt["daily_pnl_usd"] = float(rt.get("daily_pnl_usd") or 0.0) + float(pnl_usd or 0.0)
    if pnl_usd is not None and float(pnl_usd) < 0:
        rt["consecutive_losses"] = int(rt.get("consecutive_losses") or 0) + 1
        streak = int(rt["consecutive_losses"])
        if streak >= int(cfg.get("throttle_loss_streak") or 2):
            until = datetime.now(timezone.utc) + timedelta(seconds=int(cfg.get("throttle_loss_cooldown_sec") or 900))
            rt["throttle_until_utc"] = until.isoformat()
            rt["throttle_reason"] = f"loss_streak={streak}"
    elif pnl_usd is not None and float(pnl_usd) > 0:
        rt["consecutive_losses"] = 0
    _save_state(state_path, state)


# -----------------------------------------------------------------------------
# Gate logic
# -----------------------------------------------------------------------------

def _session_flag_now(trading_hours: dict[str, Any]) -> tuple[bool, str]:
    """Return (in_allowed_session, label). Uses UTC hour bands.

    Tokyo: 00-09 UTC.  London: 07-16 UTC.  NY: 12-21 UTC.
    If *any* enabled session overlaps the current hour, we allow.
    """
    now_h = datetime.now(timezone.utc).hour
    in_tokyo = 0 <= now_h < 9
    in_london = 7 <= now_h < 16
    in_ny = 12 <= now_h < 21
    allowed = False
    labels: list[str] = []
    if in_tokyo:
        labels.append("tokyo")
        if trading_hours.get("tokyo", True):
            allowed = True
    if in_london:
        labels.append("london")
        if trading_hours.get("london", True):
            allowed = True
    if in_ny:
        labels.append("ny")
        if trading_hours.get("ny", True):
            allowed = True
    return allowed, "/".join(labels) if labels else "off-hours"


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _m3_trend(data_by_tf: dict[str, pd.DataFrame]) -> Optional[str]:
    """Return 'bull'|'bear'|None based on M3 EMA stack (close > EMA9 > EMA21)."""
    df = data_by_tf.get("M3") if data_by_tf else None
    if df is None or len(df) < 25:
        return None
    close = df["close"].astype(float)
    e9 = _ema(close, 9).iloc[-1]
    e21 = _ema(close, 21).iloc[-1]
    c = float(close.iloc[-1])
    if c > e9 > e21:
        return "bull"
    if c < e9 < e21:
        return "bear"
    return None


def _m1_stack(data_by_tf: dict[str, pd.DataFrame]) -> Optional[str]:
    """M1 EMA5/9/21 stack alignment."""
    df = data_by_tf.get("M1") if data_by_tf else None
    if df is None or len(df) < 25:
        return None
    close = df["close"].astype(float)
    e5 = _ema(close, 5).iloc[-1]
    e9 = _ema(close, 9).iloc[-1]
    e21 = _ema(close, 21).iloc[-1]
    if e5 > e9 > e21:
        return "bull"
    if e5 < e9 < e21:
        return "bear"
    return None


def _m1_pullback_or_zone(data_by_tf: dict[str, pd.DataFrame], trend: Optional[str]) -> bool:
    """True if price is at/near EMA9 (zone) or touched EMA13-17 (pullback) aligned with trend."""
    if trend is None:
        return False
    df = data_by_tf.get("M1") if data_by_tf else None
    if df is None or len(df) < 30:
        return False
    close = df["close"].astype(float)
    last = float(close.iloc[-1])
    e9 = float(_ema(close, 9).iloc[-1])
    e13 = float(_ema(close, 13).iloc[-1])
    e17 = float(_ema(close, 17).iloc[-1])
    # Zone: within 1.5 pips of EMA9
    if abs(last - e9) < 0.015:
        return True
    # Pullback: last bar low <= EMA13..17 for bull (or high >= for bear)
    last_low = float(df["low"].astype(float).iloc[-1])
    last_high = float(df["high"].astype(float).iloc[-1])
    if trend == "bull" and (last_low <= e13 or last_low <= e17):
        return True
    if trend == "bear" and (last_high >= e13 or last_high >= e17):
        return True
    return False


def _near_daily_hl(tick_mid: float, data_by_tf: dict[str, pd.DataFrame], buffer_pips: float) -> bool:
    """True if current price is within buffer_pips of today's H or L."""
    df = data_by_tf.get("D") if data_by_tf else None
    if df is None or len(df) == 0 or buffer_pips <= 0:
        return False
    try:
        today = df.iloc[-1]
        hi = float(today["high"])
        lo = float(today["low"])
    except Exception:
        return False
    pip = 0.01
    if abs(tick_mid - hi) / pip <= buffer_pips:
        return True
    if abs(tick_mid - lo) / pip <= buffer_pips:
        return True
    return False


@dataclass
class GateInputs:
    spread_pips: float
    tick_mid: float
    open_ai_trade_count: int
    data_by_tf: dict[str, pd.DataFrame]
    ntz_active: bool = False


def evaluate_gate(
    cfg: dict[str, Any],
    rt: dict[str, Any],
    inputs: GateInputs,
    now_utc: Optional[datetime] = None,
) -> GateDecision:
    """Run all three gate layers. Returns the decision (does NOT mutate state)."""
    now_utc = now_utc or datetime.now(timezone.utc)
    mode = str(cfg.get("mode") or "off")
    agg = str(cfg.get("aggressiveness") or "balanced")

    def _block(layer: str, reason: str, extras: Optional[dict[str, Any]] = None) -> GateDecision:
        return GateDecision(
            timestamp_utc=now_utc.isoformat(),
            result="block", layer=layer, reason=reason,  # type: ignore[arg-type]
            mode=mode, aggressiveness=agg,
            extras=extras or {},
        )

    # ---- Layer 1: hard filters ----
    if not cfg.get("enabled"):
        return _block("hard", "autonomous_disabled")
    if mode == "off":
        return _block("hard", "mode_off")
    if inputs.spread_pips is None or inputs.spread_pips > 3.0:
        return _block("hard", f"spread_too_wide:{inputs.spread_pips}")
    if inputs.ntz_active:
        return _block("hard", "ntz_active")
    if inputs.open_ai_trade_count >= int(cfg.get("max_open_ai_trades") or 2):
        return _block("hard", f"max_open_ai_trades:{inputs.open_ai_trade_count}")
    # Daily loss cap
    if float(rt.get("daily_pnl_usd") or 0.0) <= -float(cfg.get("max_daily_loss_usd") or 50.0):
        return _block("hard", f"daily_loss_cap:{rt.get('daily_pnl_usd')}")
    # Budget cap
    if float(rt.get("llm_spend_today_usd") or 0.0) >= float(cfg.get("daily_budget_usd") or 2.0):
        return _block("hard", f"budget_cap:${rt.get('llm_spend_today_usd'):.3f}")
    # Trading hours
    allowed, session_label = _session_flag_now(cfg.get("trading_hours") or {})
    if not allowed:
        return _block("hard", f"out_of_session:{session_label}")
    # Min LLM cooldown
    last_call = rt.get("last_llm_call_utc")
    if last_call:
        try:
            last = datetime.fromisoformat(last_call)
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            elapsed = (now_utc - last).total_seconds()
            if elapsed < float(cfg.get("min_llm_cooldown_sec") or 60):
                return _block("hard", f"llm_cooldown:{elapsed:.0f}s")
        except Exception:
            pass

    # ---- Layer 3: adaptive throttle (checked before signal gate so we
    #     don't pay the signal-compute cost during a cooldown) ----
    throttle_until = rt.get("throttle_until_utc")
    if throttle_until:
        try:
            tu = datetime.fromisoformat(throttle_until)
            if tu.tzinfo is None:
                tu = tu.replace(tzinfo=timezone.utc)
            if tu > now_utc:
                return _block("throttle", f"{rt.get('throttle_reason') or 'adaptive'}_until_{tu.isoformat()}")
        except Exception:
            pass

    # ---- Layer 2: signal gate (mode-dependent) ----
    thresholds = GATE_THRESHOLDS.get(agg) or GATE_THRESHOLDS["balanced"]

    m3 = _m3_trend(inputs.data_by_tf)
    m1 = _m1_stack(inputs.data_by_tf)
    extras: dict[str, Any] = {"m3": m3, "m1": m1, "spread": inputs.spread_pips, "session": session_label}

    if thresholds.get("require_m3_trend") and m3 is None:
        return _block("signal", "no_m3_trend", extras)
    if thresholds.get("require_m1_stack") and m1 is None:
        return _block("signal", "no_m1_stack", extras)
    if thresholds.get("require_m3_trend") and thresholds.get("require_m1_stack"):
        # Require alignment between M3 and M1 to avoid fighting the tape.
        if m3 is not None and m1 is not None and m3 != m1:
            return _block("signal", f"m3_m1_mismatch:{m3}/{m1}", extras)

    if thresholds.get("require_pullback_or_zone"):
        trend = m1 or m3
        if not _m1_pullback_or_zone(inputs.data_by_tf, trend):
            return _block("signal", "no_pullback_or_zone", extras)

    buf = float(thresholds.get("require_daily_hl_buffer_pips") or 0.0)
    if buf > 0 and _near_daily_hl(inputs.tick_mid, inputs.data_by_tf, buf):
        return _block("signal", f"near_daily_hl_within_{buf:.1f}p", extras)

    # "aggressive" mode only requires *some* trend signal — either M3 or M1.
    if thresholds.get("require_any_trend_signal"):
        if m3 is None and m1 is None:
            return _block("signal", "no_trend_signal", extras)

    return GateDecision(
        timestamp_utc=now_utc.isoformat(),
        result="pass", layer="pass", reason="ok",
        mode=mode, aggressiveness=agg, extras=extras,
    )


# -----------------------------------------------------------------------------
# Stats for the UI
# -----------------------------------------------------------------------------

def build_stats(state_path: Path, cfg: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    cfg = cfg or get_config(state_path)
    state = _load_state(state_path)
    rt = (state.get("autonomous_fillmore") or {}).get("runtime") or {}
    _rollover_daily_counters(rt)  # in-memory only, we don't persist here

    decisions = list(rt.get("decisions") or [])
    # Break down by result + layer over last N.
    total = len(decisions)
    passes = sum(1 for d in decisions if d.get("r") == "pass")
    blocks = total - passes
    layer_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for d in decisions:
        if d.get("r") == "block":
            layer_counts[str(d.get("l") or "?")] = layer_counts.get(str(d.get("l") or "?"), 0) + 1
            reason_counts[str(d.get("why") or "?")] = reason_counts.get(str(d.get("why") or "?"), 0) + 1

    thresholds = GATE_THRESHOLDS.get(cfg.get("aggressiveness") or "balanced") or {}
    est_cost_per_call = _estimated_call_cost_usd(cfg.get("model") or "gpt-5.4-mini")

    throttle_until = rt.get("throttle_until_utc")
    throttled = False
    if throttle_until:
        try:
            tu = datetime.fromisoformat(throttle_until)
            if tu.tzinfo is None:
                tu = tu.replace(tzinfo=timezone.utc)
            throttled = tu > datetime.now(timezone.utc)
        except Exception:
            pass

    return {
        "config": cfg,
        "gate_description": thresholds.get("description"),
        "expected_pass_rate_pct": thresholds.get("expected_pass_rate_pct"),
        "est_cost_per_llm_call_usd": round(est_cost_per_call, 6),
        "window": {
            "total": total,
            "passes": passes,
            "blocks": blocks,
            "pass_rate_pct": round((passes / total * 100.0), 2) if total > 0 else 0.0,
            "top_block_layers": dict(sorted(layer_counts.items(), key=lambda kv: -kv[1])[:5]),
            "top_block_reasons": dict(sorted(reason_counts.items(), key=lambda kv: -kv[1])[:8]),
        },
        "today": {
            "llm_calls": int(rt.get("llm_calls_today") or 0),
            "spend_usd": round(float(rt.get("llm_spend_today_usd") or 0.0), 4),
            "budget_usd": float(cfg.get("daily_budget_usd") or 0.0),
            "budget_used_pct": round(
                (float(rt.get("llm_spend_today_usd") or 0.0) / max(float(cfg.get("daily_budget_usd") or 0.0001), 1e-6)) * 100.0, 1
            ),
            "trades_placed": int(rt.get("trades_placed_today") or 0),
            "pnl_usd": round(float(rt.get("daily_pnl_usd") or 0.0), 2),
        },
        "throttle": {
            "active": throttled,
            "until_utc": throttle_until if throttled else None,
            "reason": rt.get("throttle_reason") if throttled else None,
            "consecutive_no_trade_replies": int(rt.get("consecutive_no_trade_replies") or 0),
            "consecutive_losses": int(rt.get("consecutive_losses") or 0),
            "consecutive_errors": int(rt.get("consecutive_errors") or 0),
        },
        "last_llm_call_utc": rt.get("last_llm_call_utc"),
        "last_placed_order_id": rt.get("last_placed_order_id"),
        "last_suggestion_id": rt.get("last_suggestion_id"),
        "recent_decisions": decisions[-30:][::-1],  # most recent first
    }


# -----------------------------------------------------------------------------
# Engine — called from run_loop each tick
# -----------------------------------------------------------------------------

def _confidence_rank(label: str) -> int:
    return {"low": 1, "medium": 2, "high": 3}.get(str(label).lower(), 0)


def _count_open_ai_trades(store, profile_name: str) -> int:
    try:
        rows = store.list_open_trades(profile_name)
    except Exception:
        return 0
    n = 0
    for r in rows:
        rd = dict(r)
        if str(rd.get("entry_type") or "").lower() == "ai_manual":
            n += 1
    return n


def _refresh_daily_pnl(store, profile_name: str, rt: dict[str, Any]) -> None:
    """Query closed ai_manual trades for today and update rt['daily_pnl_usd'].

    Runs best-effort — any error leaves rt['daily_pnl_usd'] unchanged. Called
    each tick so the daily loss cap and loss-streak throttle see real outcomes
    without us needing to hook into every close path.
    """
    today = _today_utc_key()
    try:
        rows = store.get_trades_df(profile_name) if hasattr(store, "get_trades_df") else None
        if rows is None or getattr(rows, "empty", True):
            return
    except Exception:
        return
    try:
        df = rows.copy()
        # Filter to ai_manual entries that closed today.
        if "entry_type" in df.columns:
            df = df[df["entry_type"].astype(str).str.lower() == "ai_manual"]
        if "close_time" not in df.columns and "exit_time_utc" in df.columns:
            df = df.rename(columns={"exit_time_utc": "close_time"})
        if "close_time" not in df.columns and "exit_timestamp_utc" in df.columns:
            df = df.rename(columns={"exit_timestamp_utc": "close_time"})
        if "close_time" in df.columns:
            df = df[df["close_time"].astype(str).str.startswith(today)]
        if df.empty:
            return
        pnl_col = None
        for cand in ("pnl_usd", "pnl", "net_pnl", "profit_usd", "profit"):
            if cand in df.columns:
                pnl_col = cand
                break
        if pnl_col is None:
            return
        daily = float(df[pnl_col].fillna(0).astype(float).sum())
        # Compute loss streak from closed ai_manual trades today (chronological).
        if "close_time" in df.columns:
            df = df.sort_values("close_time")
        losses_in_streak = 0
        for v in df[pnl_col].fillna(0).astype(float).tolist():
            if v < 0:
                losses_in_streak += 1
            elif v > 0:
                losses_in_streak = 0
            # v == 0: leave streak as-is
        rt["daily_pnl_usd"] = daily
        rt["consecutive_losses"] = losses_in_streak
    except Exception:
        return


def tick_autonomous_fillmore(
    profile,
    profile_name: str,
    state_path: Path,
    store,
    tick,
    data_by_tf: dict[str, pd.DataFrame],
    ntz_active: bool = False,
    **_unused: Any,
) -> None:
    """Called from the run loop each iteration after trade management.

    Fast-paths out when autonomous is disabled. When the gate passes and mode is
    paper/live, it invokes the suggest endpoint and places the limit.
    """
    cfg = get_config(state_path)
    if not cfg.get("enabled"):
        return  # fast path — don't even log when fully off.

    # Gather inputs.
    pip = float(getattr(profile, "pip_size", 0.01) or 0.01)
    mid = (float(tick.bid) + float(tick.ask)) / 2.0
    spread_pips = (float(tick.ask) - float(tick.bid)) / pip if pip > 0 else None

    inputs = GateInputs(
        spread_pips=float(spread_pips) if spread_pips is not None else 999.0,
        tick_mid=mid,
        open_ai_trade_count=_count_open_ai_trades(store, profile_name),
        data_by_tf=data_by_tf or {},
        ntz_active=bool(ntz_active),
    )

    state = _load_state(state_path)
    rt = _runtime_block(state)
    _rollover_daily_counters(rt)
    # Refresh daily P&L + loss streak from the trades store so the daily loss
    # cap + loss-streak throttle stay honest without needing a close-path hook.
    _refresh_daily_pnl(store, profile_name, rt)
    # Apply loss-streak throttle if needed (fresh computation each tick).
    streak = int(rt.get("consecutive_losses") or 0)
    loss_thr = int(cfg.get("throttle_loss_streak") or 2)
    if streak >= loss_thr:
        # Only set if not already set to a future time.
        existing = rt.get("throttle_until_utc")
        existing_future = False
        if existing:
            try:
                tu = datetime.fromisoformat(existing)
                if tu.tzinfo is None:
                    tu = tu.replace(tzinfo=timezone.utc)
                existing_future = tu > datetime.now(timezone.utc)
            except Exception:
                pass
        if not existing_future:
            until = datetime.now(timezone.utc) + timedelta(seconds=int(cfg.get("throttle_loss_cooldown_sec") or 900))
            rt["throttle_until_utc"] = until.isoformat()
            rt["throttle_reason"] = f"loss_streak={streak}"
    state["autonomous_fillmore"]["runtime"] = rt
    _save_state(state_path, state)

    decision = evaluate_gate(cfg, rt, inputs)
    log_decision(state_path, decision)

    if decision.result != "pass":
        return

    # ---- Gate passed: invoke LLM via existing suggest plumbing ----
    mode = str(cfg.get("mode") or "off")
    print(f"[{profile_name}] autonomous Fillmore: gate passed ({cfg.get('aggressiveness')}, {mode}); invoking LLM")

    try:
        suggestion = _invoke_suggest(profile, profile_name, cfg)
    except Exception as e:
        print(f"[{profile_name}] autonomous Fillmore: LLM error: {e}")
        record_error(state_path, cfg, str(e))
        return

    # Record the call & cost regardless of trade-or-not.
    record_llm_invocation(state_path, str(cfg.get("model") or "gpt-5.4-mini"))

    conf = str(suggestion.get("confidence") or "low").lower()
    min_conf = str(cfg.get("min_confidence") or "medium").lower()
    if _confidence_rank(conf) < _confidence_rank(min_conf):
        print(f"[{profile_name}] autonomous Fillmore: suggestion confidence {conf} < min {min_conf}; skipping")
        record_no_trade_reply(state_path, cfg)
        return

    # Shadow mode logs the would-have-traded decision but stops here.
    if mode == "shadow":
        print(f"[{profile_name}] autonomous Fillmore: SHADOW — would have placed {suggestion.get('side')} @ {suggestion.get('price')}")
        # Successful decision in shadow mode; clear no-trade streak.
        state = _load_state(state_path)
        rt = _runtime_block(state)
        rt["consecutive_no_trade_replies"] = 0
        _save_state(state_path, state)
        return

    # Safety cap — LLM is told the ceiling in the prompt, but backstop here.
    max_lots = float(cfg.get("max_lots_per_trade") or 15.0)
    if float(suggestion.get("lots") or 0) > max_lots:
        print(f"[{profile_name}] autonomous Fillmore: clipping lots {suggestion.get('lots')} -> {max_lots}")
        suggestion["lots"] = max_lots
    if float(suggestion.get("lots") or 0) <= 0:
        print(f"[{profile_name}] autonomous Fillmore: invalid lots {suggestion.get('lots')}; skipping")
        return

    # ---- Place the order (market or limit per config) ----
    order_type = str(cfg.get("order_type") or "market").lower()
    if mode == "paper":
        bt = str(getattr(profile, "broker_type", None) or "mt5").lower()
        if bt == "oanda":
            env = str(getattr(profile, "oanda_environment", None) or "practice").lower()
            if env != "practice":
                print(
                    f"[{profile_name}] autonomous Fillmore: PAPER mode requires "
                    f"profile oanda_environment='practice' (found {env!r}); skipping place."
                )
                return
    try:
        placed = _place_from_suggestion(
            profile, profile_name, state_path, suggestion, order_type, store,
        )
        record_trade_placed(state_path, placed.get("order_id"), suggestion.get("suggestion_id"))
        kind = "MKT" if order_type == "market" else "LMT"
        print(
            f"[{profile_name}] autonomous Fillmore: placed {kind} {suggestion.get('side')} "
            f"{suggestion.get('lots')} @ {suggestion.get('price')} (order {placed.get('order_id')})"
        )
    except Exception as e:
        print(f"[{profile_name}] autonomous Fillmore: place error: {e}")
        record_error(state_path, cfg, str(e))


def _invoke_suggest(profile, profile_name: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Call the same suggestion pipeline the HTTP endpoint uses.

    We import lazily to avoid a circular dependency (api.main imports us).
    """
    from api.ai_trading_chat import (
        build_trade_suggestion_news_block,
        build_trading_context,
        resolve_ai_suggest_model,
        system_prompt_from_context,
    )
    from api import suggestion_tracker
    from api.ai_exit_strategies import (
        AI_EXIT_STRATEGIES, DEFAULT_AI_EXIT_STRATEGY,
        exit_strategies_prompt_block, merge_exit_params, normalize_exit_strategy,
    )
    import openai
    import json as _json

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not configured")

    ctx = build_trading_context(profile, profile_name)
    model = resolve_ai_suggest_model(cfg.get("model"))
    system = system_prompt_from_context(ctx, model)
    # Best-effort news enrichment; swallow errors.
    try:
        news_block = build_trade_suggestion_news_block(
            symbol=getattr(profile, "symbol", "USD_JPY"),
            rss_headline_count=8,
            web_result_count=3,
            parallel_timeout_sec=10.0,
        )
        system = f"{system}\n\n{news_block}"
    except Exception:
        pass
    # Learning memory block.
    try:
        from api.main import _suggestions_db_path
        learning = suggestion_tracker.build_learning_prompt_block(
            _suggestions_db_path(profile_name), days_back=180, max_recent_examples=8, current_ctx=ctx,
        )
        system = f"{system}\n\n{learning}"
    except Exception:
        pass

    # Autonomous header: tell the LLM it's running unattended + current mode + order type.
    agg = cfg.get("aggressiveness") or "balanced"
    order_type = cfg.get("order_type") or "market"
    max_lots = float(cfg.get("max_lots_per_trade") or 15.0)
    mode = str(cfg.get("mode") or "off")
    if mode == "paper":
        order_context = (
            "PAPER (OANDA practice): MARKET orders are sent to your OANDA **practice** API — real fills on demo "
            "margin, same as LIVE but only when profile oanda_environment is 'practice'."
            if str(order_type).lower() == "market" else
            "PAPER (OANDA practice): LIMIT orders are sent to OANDA practice — real working orders on demo funds."
        )
    else:
        order_context = (
            "These trades execute as MARKET orders — the 'price' field is informational only; "
            "the fill happens at current bid/ask the moment the gate passes. Size and side "
            "are what matter. SL/TP are the real constraints."
            if str(order_type).lower() == "market" else
            "These trades execute as LIMIT orders at the 'price' you set. They may never fill "
            "if the market doesn't come to your level. Default GTD 1 day unless you specify otherwise."
        )
    paper_note = (
        "\nPAPER MODE: Autonomous uses real broker order placement on the OANDA **practice** server only. "
        "Ensure this profile has oanda_environment='practice' and a practice API token. "
        "Fills, SL/TP, and managed exits run through the same paths as LIVE (including learning logs).\n"
        if mode == "paper" else ""
    )
    system = (
        f"{system}\n\n=== AUTONOMOUS MODE ===\n"
        f"You are running unattended in gate mode '{agg}' (execution mode: {mode}). The signal gate woke you up because it "
        f"saw a setup worth checking, but the trader is NOT reviewing each suggestion. "
        f"Be MORE selective than usual — only return confidence='high' when the setup is genuinely strong. "
        f"Return confidence='low' freely when the setup is weak; the system will skip placing.\n\n"
        f"ORDER EXECUTION: {order_context}\n"
        f"LOT CEILING: {max_lots:.0f} lots maximum. Size within your normal 1-{int(max_lots)} range."
        f"{paper_note}"
    )

    _strategy_ids = ", ".join(f'"{sid}"' for sid in AI_EXIT_STRATEGIES.keys())
    _exit_catalog_text = exit_strategies_prompt_block()
    order_label = "market" if order_type == "market" else "limit"
    price_instr = (
        'The "price" field is informational — put your expected fill reference (e.g., current mid).'
        if order_type == "market" else
        'Limit price must be AWAY from current price (BUY below current bid, SELL above current ask).'
    )
    suggest_prompt = (
        f"Based on LIVE TRADING CONTEXT and the prefetched EXTERNAL MARKET NEWS section (RSS headlines + web search), "
        f"suggest ONE specific USDJPY {order_label} order trade right now OR return low confidence. "
        "You MUST respond with ONLY a valid JSON object — no markdown, no code fences, no explanation outside the JSON. "
        "Use this exact JSON schema:\n"
        '{\n'
        '  "side": "buy" or "sell",\n'
        f'  "price": <{order_label} entry price as number>,\n'
        '  "sl": <stop loss price as number>,\n'
        '  "tp": <take profit price as number>,\n'
        f'  "lots": <position size as number, 1-{int(max_lots)}>,\n'
        '  "time_in_force": "GTC" or "GTD",\n'
        '  "gtd_time_utc": <ISO datetime string if GTD, else null>,\n'
        '  "exit_strategy": one of [' + _strategy_ids + '],\n'
        '  "exit_params": {<optional numeric overrides>} or null,\n'
        '  "rationale": "<2-4 sentence SETUP + EXIT choice explanation>",\n'
        '  "confidence": "low" or "medium" or "high"\n'
        '}\n\n'
        + _exit_catalog_text
        + "\n\n=== DECISION FRAMEWORK (guidelines — bypass with reason when the tape warrants) ===\n"
        "\nDIRECTION:\n"
        "- Cross-check TECHNICAL SNAPSHOT CONSENSUS. Prefer setups where 3+ timeframes align. "
        "When consensus is 'mixed' or 'leaning', demand a better entry or downgrade confidence.\n"
        "- Name any DIVERGENCES in the rationale (e.g., 'M1 bear against H1/M15/M5 bull — pullback setup').\n"
        "- Use JPY CROSS BIAS to disambiguate. Higher BUY conviction when crosses confirm JPY weakness.\n"
        "- RSI extremes across 2+ TFs = conviction dampener.\n"
        "\nENTRY PRICE:\n"
        f"- {price_instr}\n"
        "- Anchor on PRICE STRUCTURE — PDH/PDL, PWH/PWL, WH/WL, round levels (xx.00 / xx.50).\n"
        "- Cross-reference ORDER BOOK nearest S/R for confirmation.\n"
        "- Favor bar-pattern context (engulfing, hammer, etc.) when naming the setup.\n"
        "\nSTOP AND TARGET (let ATR breathe):\n"
        "- Baseline: SL 10-15 pips, TP 4-10 pips (scalper's style). Flex with M5/M15 ATR — "
        "wider stops on high ATR, tighter on compressed ranges. State the ATR you saw in rationale.\n"
        "- Never stick the SL 1-2 pips past a known level (it gets swept). Give it room.\n"
        f"\nSIZING (1-{int(max_lots)} lots, proportional to conviction):\n"
        f"- Range: 1 lot (low conviction / probe) up to {int(max_lots)} lots (elite setup — TF consensus + JPY crosses + "
        "strong structure + no catalyst risk).\n"
        f"- Typical medium-conviction: 3-7 lots. High conviction: 8-12. Reserve 13-{int(max_lots)} for A+ setups.\n"
        "- Downgrade lots if SESSION PERFORMANCE is negative over last 14d.\n"
        "- Downgrade lots if OPEN P&L is red — don't pyramid into drawdown.\n"
        "\nEXIT STRATEGY:\n"
        f'- Default "{DEFAULT_AI_EXIT_STRATEGY}" unless setup favors another (or "none").\n'
        "- If EXIT STRATEGY PERFORMANCE is present with meaningful N, prefer strategies with positive net P&L.\n"
        "- Rationale MUST justify the exit_strategy choice.\n"
        "\nTIMING VETO:\n"
        "- If IMMINENT EVENT banner is active (<30m to high-impact release), strongly consider confidence='low' "
        "unless rationale names a specific post-event level worth fading/riding.\n"
        "\nMECHANICS:\n"
        "- If you genuinely see no good setup, return confidence='low' and explain.\n"
        "- If market is closed or data is stale, still provide a suggestion based on last known levels.\n"
    )

    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": suggest_prompt},
        ],
    )
    raw = (resp.choices[0].message.content or "").strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
    if raw.endswith("```"):
        raw = raw[: raw.rfind("```")]
    raw = raw.strip()
    suggestion = _json.loads(raw)

    # Normalize like the HTTP path does.
    suggestion["side"] = str(suggestion.get("side") or "buy").lower()
    suggestion["price"] = float(suggestion.get("price") or 0)
    suggestion["sl"] = float(suggestion.get("sl") or 0)
    suggestion["tp"] = float(suggestion.get("tp") or 0)
    suggestion["lots"] = float(suggestion.get("lots") or 0)
    tif = str(suggestion.get("time_in_force") or "GTD").upper()
    if tif not in ("GTC", "GTD"):
        tif = "GTD"
    suggestion["time_in_force"] = tif
    if tif == "GTD" and not suggestion.get("gtd_time_utc"):
        suggestion["gtd_time_utc"] = (
            (datetime.now(timezone.utc) + timedelta(days=1))
            .replace(microsecond=0).isoformat().replace("+00:00", "Z")
        )
    raw_exit = suggestion.get("exit_strategy")
    raw_exit_s = str(raw_exit).strip().lower() if raw_exit is not None else ""
    if raw_exit_s == "none":
        suggestion["exit_strategy"] = "none"
        suggestion["exit_params"] = {}
    else:
        chosen = normalize_exit_strategy(raw_exit_s) if raw_exit_s in AI_EXIT_STRATEGIES else DEFAULT_AI_EXIT_STRATEGY
        suggestion["exit_strategy"] = chosen
        raw_params = suggestion.get("exit_params") if isinstance(suggestion.get("exit_params"), dict) else None
        suggestion["exit_params"] = merge_exit_params(chosen, raw_params)
    suggestion["model_used"] = model

    # Persist the suggestion row (so learning memory + stats include it).
    try:
        from api.main import _suggestions_db_path
        sid = suggestion_tracker.log_generated(
            _suggestions_db_path(profile_name),
            profile=profile_name,
            model=model,
            suggestion=suggestion,
        )
        suggestion["suggestion_id"] = sid
    except Exception as e:
        print(f"[{profile_name}] autonomous Fillmore: log_generated failed: {e}")
    return suggestion


def _place_from_suggestion(
    profile, profile_name: str, state_path: Path, suggestion: dict[str, Any],
    order_type: str, store,
) -> dict[str, Any]:
    """Place the order (market or limit) + register managed exit for the run loop.

    Used for both Autonomous ``live`` and ``paper`` modes: orders always go through
    the broker adapter (OANDA practice when profile ``oanda_environment`` is
    ``practice``). ``paper`` mode is a product toggle only; it does not bypass the broker.
    """
    from adapters.broker import get_adapter
    from api.ai_exit_strategies import (
        merge_exit_params, normalize_exit_strategy, trail_mode_for_strategy,
    )
    from api import suggestion_tracker
    import json as _json

    side = str(suggestion.get("side") or "buy").lower()
    price = float(suggestion.get("price") or 0)
    lots = float(suggestion.get("lots") or 0)
    sl = suggestion.get("sl")
    tp = suggestion.get("tp")
    tif = str(suggestion.get("time_in_force") or "GTD").upper()
    gtd = suggestion.get("gtd_time_utc")

    exit_strat = (suggestion.get("exit_strategy") or "").strip().lower()
    managed_strategy: Optional[str] = None
    managed_params: dict[str, Any] = {}
    if exit_strat and exit_strat != "none":
        managed_strategy = normalize_exit_strategy(exit_strat)
        managed_params = merge_exit_params(managed_strategy, suggestion.get("exit_params"))

    adapter = get_adapter(profile)
    adapter.initialize()
    try:
        if order_type == "market":
            # ----- Market path -----
            result = adapter.order_send_market(
                symbol=profile.symbol,
                side=side,
                volume_lots=lots,
                sl=float(sl) if sl is not None else None,
                tp=float(tp) if tp is not None else None,
                comment=f"autonomous_fillmore:{profile_name}",
            )
            if result.retcode != 0:
                raise RuntimeError(f"market order rejected: {result.comment}")

            fill_price = float(getattr(result, "fill_price", None) or price or 0.0)
            position_id = getattr(result, "deal", None)  # OANDA tradeID = position id
            order_id = getattr(result, "order", None)

            # Insert trade row directly so _manage_ai_manual_trades picks it up.
            if managed_strategy and position_id is not None:
                try:
                    trail_mode = trail_mode_for_strategy(managed_strategy)
                    trade_id = f"ai_manual:{order_id or position_id}:{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d%H%M%S')}"
                    row: dict[str, Any] = {
                        "trade_id": trade_id,
                        "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                        "profile": profile.profile_name,
                        "symbol": profile.symbol,
                        "side": side,
                        "policy_type": "ai_manual",
                        "config_json": _json.dumps({
                            "source": "autonomous_fillmore",
                            "order_id": order_id,
                            "exit_strategy": managed_strategy,
                            "exit_params": managed_params,
                            "order_type": "market",
                        }),
                        "entry_price": fill_price,
                        "stop_price": float(sl) if sl is not None else None,
                        "target_price": float(tp) if tp is not None else None,
                        "size_lots": lots,
                        "notes": f"autonomous_fillmore:{managed_strategy}:order_{order_id}",
                        "snapshot_id": None,
                        "mt5_order_id": int(order_id) if order_id is not None else None,
                        "mt5_deal_id": None,
                        "mt5_retcode": 0,
                        "mt5_position_id": int(position_id),
                        "opened_by": "ai_manual",
                        "preset_name": getattr(profile, "active_preset_name", None) or "Autonomous Fillmore",
                        "entry_type": "ai_manual",
                        "breakeven_applied": 0,
                        "tp1_partial_done": 0,
                        "managed_trail_mode": trail_mode or "none",
                    }
                    if managed_params.get("tp1_pips") is not None:
                        row["managed_tp1_pips"] = float(managed_params["tp1_pips"])
                    if managed_params.get("tp1_close_pct") is not None:
                        row["managed_tp1_close_pct"] = float(managed_params["tp1_close_pct"])
                    if managed_params.get("be_plus_pips") is not None:
                        row["managed_be_plus_pips"] = float(managed_params["be_plus_pips"])
                    store.insert_trade(row)
                    print(
                        f"[{profile_name}] autonomous Fillmore: inserted market trade {trade_id} "
                        f"(pos {position_id}, strategy={managed_strategy}, trail={trail_mode})"
                    )
                except Exception as e:
                    print(f"[{profile_name}] autonomous Fillmore: insert_trade error: {e}")

            # Stamp the suggestion as placed + filled.
            sid = suggestion.get("suggestion_id")
            if sid:
                try:
                    from api.main import _suggestions_db_path
                    suggestion_tracker.log_action(
                        _suggestions_db_path(profile_name),
                        suggestion_id=sid,
                        action="placed",
                        edited_fields={},
                        placed_order={
                            "side": side, "price": fill_price, "lots": lots,
                            "sl": float(sl) if sl is not None else None,
                            "tp": float(tp) if tp is not None else None,
                            "time_in_force": "MARKET", "gtd_time_utc": None,
                            "exit_strategy": managed_strategy or "none",
                            "exit_params": managed_params if managed_strategy else {},
                            "autonomous": True,
                            "order_type": "market",
                        },
                        oanda_order_id=str(order_id) if order_id is not None else None,
                    )
                    suggestion_tracker.mark_filled(
                        _suggestions_db_path(profile_name),
                        oanda_order_id=str(order_id) if order_id is not None else None,
                        fill_price=fill_price,
                        filled_at=pd.Timestamp.now(tz="UTC").isoformat(),
                        trade_id=f"ai_manual:{order_id or position_id}",
                    )
                except Exception as e:
                    print(f"[{profile_name}] autonomous Fillmore: tracker stamp failed: {e}")
            return {"order_id": order_id, "status": "filled", "fill_price": fill_price}

        # ----- Limit path (existing behavior) -----
        result = adapter.order_send_pending_limit(
            symbol=profile.symbol,
            side=side,
            price=price,
            volume_lots=lots,
            sl=float(sl) if sl is not None else None,
            tp=float(tp) if tp is not None else None,
            time_in_force=tif,
            gtd_time_utc=gtd,
            comment=f"autonomous_fillmore:{profile_name}",
        )
        if result.retcode != 0:
            raise RuntimeError(f"limit order rejected: {result.comment}")

        if managed_strategy and result.order is not None:
            state = _load_state(state_path)
            pending = list(state.get("managed_pending_orders") or [])
            pending = [p for p in pending if str(p.get("order_id")) != str(result.order)]
            pending.append({
                "order_id": int(result.order),
                "side": side,
                "price": price,
                "lots": lots,
                "sl": float(sl) if sl is not None else None,
                "tp": float(tp) if tp is not None else None,
                "exit_strategy": managed_strategy,
                "trail_mode": trail_mode_for_strategy(managed_strategy),
                "exit_params": managed_params,
                "source": "ai_manual",  # reuse watchdog path
                "suggestion_id": suggestion.get("suggestion_id"),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "autonomous": True,
            })
            state["managed_pending_orders"] = pending
            _save_state(state_path, state)

        sid = suggestion.get("suggestion_id")
        if sid:
            try:
                from api.main import _suggestions_db_path
                suggestion_tracker.log_action(
                    _suggestions_db_path(profile_name),
                    suggestion_id=sid,
                    action="placed",
                    edited_fields={},
                    placed_order={
                        "side": side, "price": price, "lots": lots,
                        "sl": float(sl) if sl is not None else None,
                        "tp": float(tp) if tp is not None else None,
                        "time_in_force": tif, "gtd_time_utc": gtd,
                        "exit_strategy": managed_strategy or "none",
                        "exit_params": managed_params if managed_strategy else {},
                        "autonomous": True,
                        "order_type": "limit",
                    },
                    oanda_order_id=str(result.order) if result.order is not None else None,
                )
            except Exception as e:
                print(f"[{profile_name}] autonomous Fillmore: log_action failed: {e}")
        return {"order_id": result.order, "status": "placed"}
    finally:
        try:
            adapter.shutdown()
        except Exception:
            pass
