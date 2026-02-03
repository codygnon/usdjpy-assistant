from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

# Broker adapter (MT5 or OANDA) is passed into execute_* as adapter=
from adapters.mt5_adapter import Tick
from core.context_engine import compute_tf_context
from core.indicators import bollinger_bands as bollinger_bands_fn
from core.indicators import ema as ema_fn
from core.indicators import macd as macd_fn
from core.indicators import rsi as rsi_fn
from core.indicators import vwap as vwap_fn
from core.models import MarketContext, TradeCandidate
from core.profile import (
    ExecutionPolicyBollingerBands,
    ExecutionPolicyBreakout,
    ExecutionPolicyEmaPullback,
    ExecutionPolicyIndicator,
    ExecutionPolicyPriceLevelTrend,
    ExecutionPolicySessionMomentum,
    ExecutionPolicyVWAP,
    ProfileV1,
    get_effective_risk,
)
from core.risk_engine import evaluate_trade
from core.signal_engine import Signal, drop_incomplete_last_bar
from core.timeframes import Timeframe
from storage.sqlite_store import SqliteStore


@dataclass(frozen=True)
class ExecutionDecision:
    attempted: bool
    placed: bool
    reason: str
    order_retcode: Optional[int] = None
    order_id: Optional[int] = None
    deal_id: Optional[int] = None
    side: Optional[str] = None


def _store(log_dir: Path) -> SqliteStore:
    store = SqliteStore(log_dir / "assistant.db")
    store.init_db()
    return store


def build_default_candidate_from_signal(profile: ProfileV1, signal: Signal) -> TradeCandidate:
    """Convert a confirmed signal into an executable market order candidate.

    v1 defaults:
    - entry at market (best effort) using mt5 tick at execution time.
    - stop uses risk.min_stop_pips away from entry (acts as default stop).
    - target uses trade_management.target (fixed pips by default).
    - size defaults to risk.max_lots (or can be set later by sizing).
    """
    r = get_effective_risk(profile)
    entry = float(signal.entry_price_hint)

    # default stop: min_stop_pips away
    stop = None
    if r.require_stop:
        dp = float(r.min_stop_pips) * float(profile.pip_size)
        stop = entry - dp if signal.side == "buy" else entry + dp

    target = None
    tcfg = profile.trade_management.target
    if tcfg.mode == "fixed_pips":
        tp = float(tcfg.pips_default) * float(profile.pip_size)
        target = entry + tp if signal.side == "buy" else entry - tp

    return TradeCandidate(
        symbol=profile.symbol,
        side=signal.side,
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        size_lots=float(r.max_lots),
    )


def _rsi_zone(value: float, oversold: float, overbought: float) -> str:
    if value <= oversold:
        return "oversold"
    if value >= overbought:
        return "overbought"
    return "neutral"


def evaluate_indicator_policy(
    profile: ProfileV1,
    policy: ExecutionPolicyIndicator,
    data_by_tf: dict[Timeframe, pd.DataFrame],
) -> tuple[bool, list[str]]:
    """Check regime matches policy.regime and RSI is in policy.rsi_zone; optionally require MACD on correct side."""
    reasons: list[str] = []
    passed: list[str] = []
    policy_id = f"{policy.type}:{policy.id}"
    
    df = data_by_tf.get(policy.timeframe)
    if df is None or df.empty:
        return False, [f"REJECTED: {policy_id} | Filter: data | TF: {policy.timeframe} | Actual: no data | Expected: bars available"]

    df = df.copy().sort_values("time")
    if len(df) < policy.rsi_period + 1:
        return False, [f"REJECTED: {policy_id} | Filter: bar_count | TF: {policy.timeframe} | Actual: {len(df)} bars | Expected: >= {policy.rsi_period + 1}"]

    close = df["close"].astype(float)
    rsi_series = rsi_fn(close, policy.rsi_period)
    rsi_val = float(rsi_series.iloc[-1]) if not rsi_series.empty and pd.notna(rsi_series.iloc[-1]) else None
    if rsi_val is None:
        return False, [f"REJECTED: {policy_id} | Filter: RSI | Actual: could not compute | Expected: valid RSI value"]

    zone = _rsi_zone(rsi_val, policy.rsi_oversold, policy.rsi_overbought)
    if zone != policy.rsi_zone:
        zone_range = f"<{policy.rsi_oversold}" if policy.rsi_zone == "oversold" else f">{policy.rsi_overbought}" if policy.rsi_zone == "overbought" else f"{policy.rsi_oversold}-{policy.rsi_overbought}"
        reasons.append(f"Filter: RSI_zone | Actual: {rsi_val:.1f} ({zone}) | Expected: {policy.rsi_zone} ({zone_range})")
    else:
        passed.append(f"RSI={rsi_val:.1f} ({zone})")

    ctx = compute_tf_context(profile, policy.timeframe, df)
    if ctx.regime != policy.regime:
        reasons.append(f"Filter: regime | TF: {policy.timeframe} | Actual: {ctx.regime} | Expected: {policy.regime}")
    else:
        passed.append(f"regime={ctx.regime}")

    if policy.use_macd_cross:
        macd_line, signal_line, hist = macd_fn(
            close, policy.macd_fast, policy.macd_slow, policy.macd_signal
        )
        if hist.empty or pd.isna(hist.iloc[-1]):
            reasons.append(f"Filter: MACD | Actual: histogram N/A | Expected: valid histogram")
        else:
            hist_last = float(hist.iloc[-1])
            if policy.side == "buy" and hist_last <= 0:
                reasons.append(f"Filter: MACD_histogram | Actual: {hist_last:.4f} | Expected: > 0 for {policy.side}")
            elif policy.side == "sell" and hist_last >= 0:
                reasons.append(f"Filter: MACD_histogram | Actual: {hist_last:.4f} | Expected: < 0 for {policy.side}")
            else:
                passed.append(f"MACD hist={hist_last:.4f}")

    if reasons:
        return False, [f"REJECTED: {policy_id} | side={policy.side} | " + "; ".join(reasons)]
    return True, [f"PASSED: {policy_id} | side={policy.side} | " + ", ".join(passed)]


def _indicator_candidate(
    profile: ProfileV1,
    policy: ExecutionPolicyIndicator,
    entry_price: float,
) -> TradeCandidate:
    pip = float(profile.pip_size)
    tp_pips = policy.tp_pips
    sl_pips = policy.sl_pips
    if policy.side == "buy":
        target = entry_price + tp_pips * pip if tp_pips else None
        stop = (entry_price - sl_pips * pip) if sl_pips is not None else None
    else:
        target = entry_price - tp_pips * pip if tp_pips else None
        stop = (entry_price + sl_pips * pip) if sl_pips is not None else None
    return TradeCandidate(
        symbol=profile.symbol,
        side=policy.side,
        entry_price=entry_price,
        stop_price=stop,
        target_price=target,
        size_lots=float(get_effective_risk(profile).max_lots),
    )


def execute_indicator_policy_demo_only(
    *,
    adapter,
    profile: ProfileV1,
    log_dir: Path,
    policy: ExecutionPolicyIndicator,
    context: MarketContext,
    data_by_tf: dict[Timeframe, pd.DataFrame],
    tick: Tick,
    trades_df: Optional[pd.DataFrame],
    mode: str,
    bar_time_utc: str,
) -> ExecutionDecision:
    """Evaluate indicator policy and optionally place a market order (ARMED_AUTO_DEMO). Idempotent per bar via rule_id."""
    if mode not in ("ARMED_MANUAL_CONFIRM", "ARMED_AUTO_DEMO"):
        return ExecutionDecision(attempted=False, placed=False, reason=f"mode={mode} not armed")

    if not adapter.is_demo_account():
        return ExecutionDecision(attempted=False, placed=False, reason="not a demo account (execution disabled)")

    store = _store(log_dir)
    rule_id = f"indicator:{policy.id}:{policy.timeframe}:{bar_time_utc}"
    bar_minutes = {"M1": 2, "M15": 16, "H4": 250}
    within = bar_minutes.get(policy.timeframe, 60)
    if store.has_recent_price_level_placement(profile.profile_name, rule_id, within):
        return ExecutionDecision(attempted=False, placed=False, reason="indicator_based: recent placement (idempotent)")

    ok, eval_reasons = evaluate_indicator_policy(profile, policy, data_by_tf)
    if not ok:
        store.insert_execution(
            {
                "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "profile": profile.profile_name,
                "symbol": profile.symbol,
                "signal_id": f"{rule_id}:{pd.Timestamp.now(tz='UTC').isoformat()}",
                "rule_id": rule_id,
                "mode": mode,
                "attempted": 1,
                "placed": 0,
                "reason": "; ".join(eval_reasons),
                "mt5_retcode": None,
                "mt5_order_id": None,
                "mt5_deal_id": None,
            }
        )
        return ExecutionDecision(attempted=True, placed=False, reason="; ".join(eval_reasons))

    entry_price = (tick.bid + tick.ask) / 2.0
    candidate = _indicator_candidate(profile, policy, entry_price)
    decision = evaluate_trade(profile=profile, candidate=candidate, context=context, trades_df=trades_df)
    if not decision.allow:
        sig_id = f"{rule_id}:{pd.Timestamp.now(tz='UTC').isoformat()}"
        store.insert_execution(
            {
                "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "profile": profile.profile_name,
                "symbol": profile.symbol,
                "signal_id": sig_id,
                "rule_id": rule_id,
                "mode": mode,
                "attempted": 1,
                "placed": 0,
                "reason": "risk_reject: " + "; ".join(decision.hard_reasons),
                "mt5_retcode": None,
                "mt5_order_id": None,
                "mt5_deal_id": None,
            }
        )
        return ExecutionDecision(attempted=True, placed=False, reason="risk rejected: " + "; ".join(decision.hard_reasons))

    if mode == "ARMED_MANUAL_CONFIRM":
        sig_id = f"{rule_id}:{pd.Timestamp.now(tz='UTC').isoformat()}"
        store.insert_execution(
            {
                "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "profile": profile.profile_name,
                "symbol": profile.symbol,
                "signal_id": sig_id,
                "rule_id": rule_id,
                "mode": mode,
                "attempted": 1,
                "placed": 0,
                "reason": "manual_confirm_required",
                "mt5_retcode": None,
                "mt5_order_id": None,
                "mt5_deal_id": None,
            }
        )
        return ExecutionDecision(attempted=True, placed=False, reason="manual confirm required")

    res = adapter.order_send_market(
        symbol=profile.symbol,
        side=policy.side,
        volume_lots=float(candidate.size_lots or get_effective_risk(profile).max_lots),
        sl=candidate.stop_price,
        tp=candidate.target_price,
        comment=f"ind:{policy.id}",
    )
    placed = res.retcode in (0, 10008, 10009)  # 0=OANDA success; 10008/10009=MT5
    reason = "order_sent" if placed else f"order_failed:{res.retcode}:{res.comment}"
    sig_id = f"{rule_id}:{pd.Timestamp.now(tz='UTC').isoformat()}"
    store.insert_execution(
        {
            "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "profile": profile.profile_name,
            "symbol": profile.symbol,
            "signal_id": sig_id,
            "rule_id": rule_id,
            "mode": mode,
            "attempted": 1,
            "placed": 1 if placed else 0,
            "reason": reason,
            "mt5_retcode": res.retcode,
            "mt5_order_id": res.order,
            "mt5_deal_id": res.deal,
        }
    )
    return ExecutionDecision(
        attempted=True,
        placed=placed,
        reason=reason,
        order_retcode=res.retcode,
        order_id=res.order,
        deal_id=res.deal,
    )


def evaluate_price_level_trend(
    profile: ProfileV1,
    policy: ExecutionPolicyPriceLevelTrend,
    data_by_tf: dict[Timeframe, pd.DataFrame],
) -> tuple[bool, list[str]]:
    """Check that all trend_timeframes match trend_direction (bearish -> bear, bullish -> bull)."""
    reasons: list[str] = []
    passed: list[str] = []
    want = "bear" if policy.trend_direction == "bearish" else "bull"
    for tf in policy.trend_timeframes:
        df = data_by_tf.get(tf)
        if df is None or df.empty:
            return False, [f"REJECTED: {policy.type}:{policy.id} | Filter: data | TF: {tf} | Actual: no data | Expected: bars available"]
        ctx = compute_tf_context(profile, tf, df)
        if ctx.regime != want:
            reasons.append(f"Filter: trend_direction | TF: {tf} | Actual: {ctx.regime} | Expected: {want}")
        else:
            passed.append(f"TF: {tf} regime={ctx.regime}")
    if reasons:
        return False, [f"REJECTED: {policy.type}:{policy.id} | " + "; ".join(reasons)]
    return True, [f"PASSED: {policy.type}:{policy.id} | trend OK: " + ", ".join(passed)]


def price_level_reached(tick: Tick, policy: ExecutionPolicyPriceLevelTrend) -> bool:
    """True if price has reached the level for market execution (poll-and-market fallback)."""
    x = policy.price_level
    if policy.side == "buy":
        return tick.bid <= x
    return tick.ask >= x


def _price_level_candidate(
    profile: ProfileV1,
    policy: ExecutionPolicyPriceLevelTrend,
) -> TradeCandidate:
    """Build TradeCandidate for a price_level_trend policy (entry at price_level, TP/SL from pips)."""
    pip = float(profile.pip_size)
    entry = policy.price_level
    tp_pips = policy.tp_pips
    sl_pips = policy.sl_pips

    if policy.side == "buy":
        target = entry + tp_pips * pip if tp_pips else None
        stop = (entry - sl_pips * pip) if sl_pips is not None else None
    else:
        target = entry - tp_pips * pip if tp_pips else None
        stop = (entry + sl_pips * pip) if sl_pips is not None else None

    return TradeCandidate(
        symbol=profile.symbol,
        side=policy.side,
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        size_lots=float(get_effective_risk(profile).max_lots),
    )


def execute_price_level_policy_demo_only(
    *,
    adapter,
    profile: ProfileV1,
    log_dir: Path,
    policy: ExecutionPolicyPriceLevelTrend,
    context: MarketContext,
    data_by_tf: dict[Timeframe, pd.DataFrame],
    tick: Tick,
    trades_df: Optional[pd.DataFrame],
    mode: str,
) -> ExecutionDecision:
    """Evaluate and optionally execute a price_level_trend policy (pending or market)."""
    if mode not in ("ARMED_MANUAL_CONFIRM", "ARMED_AUTO_DEMO"):
        return ExecutionDecision(attempted=False, placed=False, reason=f"mode={mode} not armed")

    if not adapter.is_demo_account():
        return ExecutionDecision(attempted=False, placed=False, reason="not a demo account (execution disabled)")

    store = _store(log_dir)
    rule_id = f"price_level:{policy.id}:{policy.price_level}:{policy.side}"
    within = policy.max_wait_minutes if policy.max_wait_minutes is not None else 60
    if store.has_recent_price_level_placement(profile.profile_name, rule_id, within):
        return ExecutionDecision(attempted=False, placed=False, reason="price_level: recent placement (idempotent)")

    ok, trend_reasons = evaluate_price_level_trend(profile, policy, data_by_tf)
    if not ok:
        # trend_reasons already includes the prefix, so don't add it again
        reason_str = "; ".join(trend_reasons)
        store.insert_execution(
            {
                "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "profile": profile.profile_name,
                "symbol": profile.symbol,
                "signal_id": f"{rule_id}:{pd.Timestamp.now(tz='UTC').isoformat()}",
                "rule_id": rule_id,
                "mode": mode,
                "attempted": 1,
                "placed": 0,
                "reason": reason_str,
                "mt5_retcode": None,
                "mt5_order_id": None,
                "mt5_deal_id": None,
            }
        )
        return ExecutionDecision(attempted=True, placed=False, reason=reason_str)

    candidate = _price_level_candidate(profile, policy)
    decision = evaluate_trade(profile=profile, candidate=candidate, context=context, trades_df=trades_df)
    if not decision.allow:
        sig_id = f"{rule_id}:{pd.Timestamp.now(tz='UTC').isoformat()}"
        store.insert_execution(
            {
                "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "profile": profile.profile_name,
                "symbol": profile.symbol,
                "signal_id": sig_id,
                "rule_id": rule_id,
                "mode": mode,
                "attempted": 1,
                "placed": 0,
                "reason": "risk_reject: " + "; ".join(decision.hard_reasons),
                "mt5_retcode": None,
                "mt5_order_id": None,
                "mt5_deal_id": None,
            }
        )
        return ExecutionDecision(attempted=True, placed=False, reason="risk rejected: " + "; ".join(decision.hard_reasons))

    if mode == "ARMED_MANUAL_CONFIRM":
        sig_id = f"{rule_id}:{pd.Timestamp.now(tz='UTC').isoformat()}"
        store.insert_execution(
            {
                "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "profile": profile.profile_name,
                "symbol": profile.symbol,
                "signal_id": sig_id,
                "rule_id": rule_id,
                "mode": mode,
                "attempted": 1,
                "placed": 0,
                "reason": "manual_confirm_required",
                "mt5_retcode": None,
                "mt5_order_id": None,
                "mt5_deal_id": None,
            }
        )
        return ExecutionDecision(attempted=True, placed=False, reason="manual confirm required")

    pip = float(profile.pip_size)
    entry = policy.price_level
    tp_pips = policy.tp_pips
    sl_pips = policy.sl_pips
    if policy.side == "buy":
        tp = entry + tp_pips * pip if tp_pips else None
        sl = (entry - sl_pips * pip) if sl_pips is not None else None
    else:
        tp = entry - tp_pips * pip if tp_pips else None
        sl = (entry + sl_pips * pip) if sl_pips is not None else None

    if policy.use_pending_order:
        # Place pending limit at X. Buy limit: X < bid; sell limit: X > ask.
        if policy.side == "buy" and tick.bid <= policy.price_level:
            return ExecutionDecision(attempted=False, placed=False, reason="price_level: buy limit X already at or below bid")
        if policy.side == "sell" and tick.ask >= policy.price_level:
            return ExecutionDecision(attempted=False, placed=False, reason="price_level: sell limit X already at or above ask")
        res = adapter.order_send_pending_limit(
            symbol=profile.symbol,
            side=policy.side,
            price=policy.price_level,
            volume_lots=float(candidate.size_lots or get_effective_risk(profile).max_lots),
            sl=sl,
            tp=tp,
            comment=f"pl:{policy.id}",
        )
    else:
        if not price_level_reached(tick, policy):
            return ExecutionDecision(attempted=False, placed=False, reason="price_level: level not reached (poll)")
        res = adapter.order_send_market(
            symbol=profile.symbol,
            side=policy.side,
            volume_lots=float(candidate.size_lots or get_effective_risk(profile).max_lots),
            sl=sl,
            tp=tp,
            comment=f"pl:{policy.id}",
        )

    placed = res.retcode in (0, 10008, 10009)  # 0=OANDA success; 10008/10009=MT5
    reason = "order_sent" if placed else f"order_failed:{res.retcode}:{res.comment}"
    sig_id = f"{rule_id}:{pd.Timestamp.now(tz='UTC').isoformat()}"
    store.insert_execution(
        {
            "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "profile": profile.profile_name,
            "symbol": profile.symbol,
            "signal_id": sig_id,
            "rule_id": rule_id,
            "mode": mode,
            "attempted": 1,
            "placed": 1 if placed else 0,
            "reason": reason,
            "mt5_retcode": res.retcode,
            "mt5_order_id": res.order,
            "mt5_deal_id": res.deal,
        }
    )
    return ExecutionDecision(
        attempted=True,
        placed=placed,
        reason=reason,
        order_retcode=res.retcode,
        order_id=res.order,
        deal_id=res.deal,
        side=policy.side,
    )


# ---------------------------------------------------------------------------
# Breakout Range Policy
# ---------------------------------------------------------------------------


def evaluate_breakout_conditions(
    profile: ProfileV1,
    policy: ExecutionPolicyBreakout,
    data_by_tf: dict[Timeframe, pd.DataFrame],
) -> tuple[bool, str | None, list[str]]:
    """Check if we're in consolidation and if a breakout occurred.
    
    Returns: (breakout_detected, side, reasons)
    """
    from core.indicators import atr as atr_fn
    
    reasons: list[str] = []
    df = data_by_tf.get(policy.timeframe)
    if df is None or df.empty:
        return False, None, [f"breakout: no data for {policy.timeframe}"]
    
    df = df.copy().sort_values("time")
    if len(df) < policy.lookback_bars + policy.atr_period:
        return False, None, [f"breakout: not enough bars ({len(df)} < {policy.lookback_bars + policy.atr_period})"]
    
    # Calculate ATR
    atr_series = atr_fn(df, period=policy.atr_period)
    if atr_series.empty or pd.isna(atr_series.iloc[-1]):
        return False, None, ["breakout: ATR could not be computed"]
    
    current_atr = float(atr_series.iloc[-1])
    atr_mean = float(atr_series.dropna().tail(50).mean())
    
    # Check if we're in consolidation (ATR below threshold)
    if atr_mean == 0:
        return False, None, ["breakout: ATR mean is zero"]
    
    atr_ratio = current_atr / atr_mean
    is_consolidation = atr_ratio <= policy.atr_threshold_ratio
    
    if not is_consolidation:
        reasons.append(f"not in consolidation: ATR ratio={atr_ratio:.2f} (need <={policy.atr_threshold_ratio})")
        return False, None, ["breakout: " + "; ".join(reasons)]
    
    # Calculate recent range
    lookback_df = df.tail(policy.lookback_bars)
    range_high = float(lookback_df["high"].max())
    range_low = float(lookback_df["low"].min())
    current_close = float(df["close"].iloc[-1])
    pip_size = float(profile.pip_size)
    buffer = policy.breakout_buffer_pips * pip_size
    
    # Check for breakout
    if current_close > range_high + buffer:
        return True, "buy", [f"breakout: bullish breakout above {range_high:.3f} + buffer"]
    elif current_close < range_low - buffer:
        return True, "sell", [f"breakout: bearish breakout below {range_low:.3f} - buffer"]
    
    return False, None, ["breakout: price within range, no breakout yet"]


def _breakout_candidate(
    profile: ProfileV1,
    policy: ExecutionPolicyBreakout,
    entry_price: float,
    side: str,
) -> TradeCandidate:
    pip = float(profile.pip_size)
    if side == "buy":
        target = entry_price + policy.tp_pips * pip if policy.tp_pips else None
        stop = (entry_price - policy.sl_pips * pip) if policy.sl_pips else None
    else:
        target = entry_price - policy.tp_pips * pip if policy.tp_pips else None
        stop = (entry_price + policy.sl_pips * pip) if policy.sl_pips else None
    
    return TradeCandidate(
        symbol=profile.symbol,
        side=side,
        entry_price=entry_price,
        stop_price=stop,
        target_price=target,
        size_lots=float(get_effective_risk(profile).max_lots),
    )


def execute_breakout_policy_demo_only(
    *,
    adapter,
    profile: ProfileV1,
    log_dir: Path,
    policy: ExecutionPolicyBreakout,
    context: MarketContext,
    data_by_tf: dict[Timeframe, pd.DataFrame],
    tick: Tick,
    trades_df: Optional[pd.DataFrame],
    mode: str,
    bar_time_utc: str,
) -> ExecutionDecision:
    """Evaluate breakout policy and optionally place a market order."""
    if mode not in ("ARMED_MANUAL_CONFIRM", "ARMED_AUTO_DEMO"):
        return ExecutionDecision(attempted=False, placed=False, reason=f"mode={mode} not armed")

    if not adapter.is_demo_account():
        return ExecutionDecision(attempted=False, placed=False, reason="not a demo account (execution disabled)")

    store = _store(log_dir)
    rule_id = f"breakout:{policy.id}:{policy.timeframe}:{bar_time_utc}"
    bar_minutes = {"M1": 2, "M15": 16, "H4": 250}
    within = bar_minutes.get(policy.timeframe, 60)
    if store.has_recent_price_level_placement(profile.profile_name, rule_id, within):
        return ExecutionDecision(attempted=False, placed=False, reason="breakout: recent placement (idempotent)")

    breakout_detected, side, eval_reasons = evaluate_breakout_conditions(profile, policy, data_by_tf)
    if not breakout_detected or side is None:
        store.insert_execution(
            {
                "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "profile": profile.profile_name,
                "symbol": profile.symbol,
                "signal_id": f"{rule_id}:{pd.Timestamp.now(tz='UTC').isoformat()}",
                "rule_id": rule_id,
                "mode": mode,
                "attempted": 1,
                "placed": 0,
                "reason": "; ".join(eval_reasons),
                "mt5_retcode": None,
                "mt5_order_id": None,
                "mt5_deal_id": None,
            }
        )
        return ExecutionDecision(attempted=True, placed=False, reason="; ".join(eval_reasons))

    entry_price = (tick.bid + tick.ask) / 2.0
    candidate = _breakout_candidate(profile, policy, entry_price, side)
    decision = evaluate_trade(profile=profile, candidate=candidate, context=context, trades_df=trades_df)
    if not decision.allow:
        sig_id = f"{rule_id}:{pd.Timestamp.now(tz='UTC').isoformat()}"
        store.insert_execution(
            {
                "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "profile": profile.profile_name,
                "symbol": profile.symbol,
                "signal_id": sig_id,
                "rule_id": rule_id,
                "mode": mode,
                "attempted": 1,
                "placed": 0,
                "reason": "risk_reject: " + "; ".join(decision.hard_reasons),
                "mt5_retcode": None,
                "mt5_order_id": None,
                "mt5_deal_id": None,
            }
        )
        return ExecutionDecision(attempted=True, placed=False, reason="risk rejected: " + "; ".join(decision.hard_reasons))

    if mode == "ARMED_MANUAL_CONFIRM":
        sig_id = f"{rule_id}:{pd.Timestamp.now(tz='UTC').isoformat()}"
        store.insert_execution(
            {
                "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "profile": profile.profile_name,
                "symbol": profile.symbol,
                "signal_id": sig_id,
                "rule_id": rule_id,
                "mode": mode,
                "attempted": 1,
                "placed": 0,
                "reason": "manual_confirm_required",
                "mt5_retcode": None,
                "mt5_order_id": None,
                "mt5_deal_id": None,
            }
        )
        return ExecutionDecision(attempted=True, placed=False, reason="manual confirm required")

    res = adapter.order_send_market(
        symbol=profile.symbol,
        side=side,
        volume_lots=float(candidate.size_lots or get_effective_risk(profile).max_lots),
        sl=candidate.stop_price,
        tp=candidate.target_price,
        comment=f"brk:{policy.id}",
    )
    placed = res.retcode in (0, 10008, 10009)  # 0=OANDA success; 10008/10009=MT5
    reason = "order_sent" if placed else f"order_failed:{res.retcode}:{res.comment}"
    sig_id = f"{rule_id}:{pd.Timestamp.now(tz='UTC').isoformat()}"
    store.insert_execution(
        {
            "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "profile": profile.profile_name,
            "symbol": profile.symbol,
            "signal_id": sig_id,
            "rule_id": rule_id,
            "mode": mode,
            "attempted": 1,
            "placed": 1 if placed else 0,
            "reason": reason,
            "mt5_retcode": res.retcode,
            "mt5_order_id": res.order,
            "mt5_deal_id": res.deal,
        }
    )
    return ExecutionDecision(
        attempted=True,
        placed=placed,
        reason=reason,
        order_retcode=res.retcode,
        order_id=res.order,
        deal_id=res.deal,
        side=side,
    )


# ---------------------------------------------------------------------------
# Bollinger Bands Policy
# ---------------------------------------------------------------------------


def evaluate_bollinger_conditions(
    profile: ProfileV1,
    policy: ExecutionPolicyBollingerBands,
    data_by_tf: dict[Timeframe, pd.DataFrame],
) -> tuple[bool, str | None, list[str]]:
    """Check if price is at lower band (buy) or upper band (sell) and regime matches.
    Returns: (passed, side, reasons)
    """
    reasons: list[str] = []
    df = data_by_tf.get(policy.timeframe)
    if df is None or df.empty:
        return False, None, [f"bollinger: no data for {policy.timeframe}"]
    df = df.copy().sort_values("time")
    if len(df) < policy.period + 1:
        return False, None, [f"bollinger: not enough bars ({len(df)} < {policy.period + 1})"]
    close = df["close"].astype(float)
    upper, middle, lower = bollinger_bands_fn(close, period=policy.period, std_dev=policy.std_dev)
    price = float(close.iloc[-1])
    u = float(upper.iloc[-1]) if not pd.isna(upper.iloc[-1]) else None
    m = float(middle.iloc[-1]) if not pd.isna(middle.iloc[-1]) else None
    l_ = float(lower.iloc[-1]) if not pd.isna(lower.iloc[-1]) else None
    if u is None or m is None or l_ is None:
        return False, None, ["bollinger: could not compute bands"]
    ctx = compute_tf_context(profile, policy.timeframe, df)
    if policy.trigger == "lower_band_buy":
        if ctx.regime != policy.regime:
            reasons.append(f"regime {ctx.regime} != {policy.regime}")
        if price > l_:
            reasons.append(f"price {price:.3f} above lower band {l_:.3f}")
        if reasons:
            return False, None, ["bollinger: " + "; ".join(reasons)]
        return True, "buy", [f"bollinger: price at/below lower band ({price:.3f} <= {l_:.3f}), regime={ctx.regime}"]
    else:
        if ctx.regime != policy.regime:
            reasons.append(f"regime {ctx.regime} != {policy.regime}")
        if price < u:
            reasons.append(f"price {price:.3f} below upper band {u:.3f}")
        if reasons:
            return False, None, ["bollinger: " + "; ".join(reasons)]
        return True, "sell", [f"bollinger: price at/above upper band ({price:.3f} >= {u:.3f}), regime={ctx.regime}"]


def _bollinger_candidate(
    profile: ProfileV1,
    policy: ExecutionPolicyBollingerBands,
    entry_price: float,
    side: str,
) -> TradeCandidate:
    pip = float(profile.pip_size)
    if side == "buy":
        target = entry_price + policy.tp_pips * pip if policy.tp_pips else None
        stop = (entry_price - policy.sl_pips * pip) if policy.sl_pips else None
    else:
        target = entry_price - policy.tp_pips * pip if policy.tp_pips else None
        stop = (entry_price + policy.sl_pips * pip) if policy.sl_pips else None
    return TradeCandidate(
        symbol=profile.symbol,
        side=side,
        entry_price=entry_price,
        stop_price=stop,
        target_price=target,
        size_lots=float(get_effective_risk(profile).max_lots),
    )


def execute_bollinger_policy_demo_only(
    *,
    profile: ProfileV1,
    log_dir: Path,
    policy: ExecutionPolicyBollingerBands,
    context: MarketContext,
    data_by_tf: dict[Timeframe, pd.DataFrame],
    tick: Tick,
    trades_df: Optional[pd.DataFrame],
    mode: str,
    bar_time_utc: str,
) -> ExecutionDecision:
    """Evaluate Bollinger Bands policy and optionally place a market order."""
    if mode not in ("ARMED_MANUAL_CONFIRM", "ARMED_AUTO_DEMO"):
        return ExecutionDecision(attempted=False, placed=False, reason=f"mode={mode} not armed")
    if not adapter.is_demo_account():
        return ExecutionDecision(attempted=False, placed=False, reason="not a demo account (execution disabled)")
    store = _store(log_dir)
    rule_id = f"bollinger:{policy.id}:{policy.timeframe}:{bar_time_utc}"
    bar_minutes = {"M1": 2, "M15": 16, "H4": 250}
    within = bar_minutes.get(policy.timeframe, 60)
    if store.has_recent_price_level_placement(profile.profile_name, rule_id, within):
        return ExecutionDecision(attempted=False, placed=False, reason="bollinger_bands: recent placement (idempotent)")
    passed, side, eval_reasons = evaluate_bollinger_conditions(profile, policy, data_by_tf)
    if not passed or side is None:
        store.insert_execution(
            {
                "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "profile": profile.profile_name,
                "symbol": profile.symbol,
                "signal_id": f"{rule_id}:{pd.Timestamp.now(tz='UTC').isoformat()}",
                "rule_id": rule_id,
                "mode": mode,
                "attempted": 1,
                "placed": 0,
                "reason": "; ".join(eval_reasons),
                "mt5_retcode": None,
                "mt5_order_id": None,
                "mt5_deal_id": None,
            }
        )
        return ExecutionDecision(attempted=True, placed=False, reason="; ".join(eval_reasons))
    entry_price = (tick.bid + tick.ask) / 2.0
    candidate = _bollinger_candidate(profile, policy, entry_price, side)
    decision = evaluate_trade(profile=profile, candidate=candidate, context=context, trades_df=trades_df)
    if not decision.allow:
        sig_id = f"{rule_id}:{pd.Timestamp.now(tz='UTC').isoformat()}"
        store.insert_execution(
            {
                "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "profile": profile.profile_name,
                "symbol": profile.symbol,
                "signal_id": sig_id,
                "rule_id": rule_id,
                "mode": mode,
                "attempted": 1,
                "placed": 0,
                "reason": "risk_reject: " + "; ".join(decision.hard_reasons),
                "mt5_retcode": None,
                "mt5_order_id": None,
                "mt5_deal_id": None,
            }
        )
        return ExecutionDecision(attempted=True, placed=False, reason="risk rejected: " + "; ".join(decision.hard_reasons))
    if mode == "ARMED_MANUAL_CONFIRM":
        sig_id = f"{rule_id}:{pd.Timestamp.now(tz='UTC').isoformat()}"
        store.insert_execution(
            {
                "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "profile": profile.profile_name,
                "symbol": profile.symbol,
                "signal_id": sig_id,
                "rule_id": rule_id,
                "mode": mode,
                "attempted": 1,
                "placed": 0,
                "reason": "manual_confirm_required",
                "mt5_retcode": None,
                "mt5_order_id": None,
                "mt5_deal_id": None,
            }
        )
        return ExecutionDecision(attempted=True, placed=False, reason="manual confirm required")
    res = adapter.order_send_market(
        symbol=profile.symbol,
        side=side,
        volume_lots=float(candidate.size_lots or get_effective_risk(profile).max_lots),
        sl=candidate.stop_price,
        tp=candidate.target_price,
        comment=f"bb:{policy.id}",
    )
    placed = res.retcode in (0, 10008, 10009)  # 0=OANDA success; 10008/10009=MT5
    reason = "order_sent" if placed else f"order_failed:{res.retcode}:{res.comment}"
    sig_id = f"{rule_id}:{pd.Timestamp.now(tz='UTC').isoformat()}"
    store.insert_execution(
        {
            "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "profile": profile.profile_name,
            "symbol": profile.symbol,
            "signal_id": sig_id,
            "rule_id": rule_id,
            "mode": mode,
            "attempted": 1,
            "placed": 1 if placed else 0,
            "reason": reason,
            "mt5_retcode": res.retcode,
            "mt5_order_id": res.order,
            "mt5_deal_id": res.deal,
        }
    )
    return ExecutionDecision(
        attempted=True,
        placed=placed,
        reason=reason,
        order_retcode=res.retcode,
        order_id=res.order,
        deal_id=res.deal,
        side=side,
    )


# ---------------------------------------------------------------------------
# VWAP Policy
# ---------------------------------------------------------------------------


def evaluate_vwap_conditions(
    profile: ProfileV1,
    policy: ExecutionPolicyVWAP,
    data_by_tf: dict[Timeframe, pd.DataFrame],
) -> tuple[bool, str | None, list[str]]:
    """Check if price vs VWAP matches policy trigger; then apply no-trade zone, session filter, slope filter. Returns: (passed, side, reasons)."""
    reasons: list[str] = []
    df = data_by_tf.get(policy.timeframe)
    if df is None or df.empty:
        return False, None, [f"vwap: no data for {policy.timeframe}"]
    df = df.copy().sort_values("time")
    if len(df) < 2:
        return False, None, ["vwap: need at least 2 bars"]
    vwap_series = vwap_fn(df)
    if vwap_series.empty or pd.isna(vwap_series.iloc[-1]):
        return False, None, ["vwap: could not compute VWAP"]
    vwap_now = float(vwap_series.iloc[-1])
    vwap_prev = float(vwap_series.iloc[-2]) if len(vwap_series) >= 2 else vwap_now
    close_now = float(df["close"].iloc[-1])
    close_prev = float(df["close"].iloc[-2])
    # --- Trigger logic: determine if signal is valid and side ---
    passed, side = False, None
    if policy.trigger == "cross_above":
        if close_prev <= vwap_prev and close_now > vwap_now:
            passed, side = True, "buy"
            reasons.append(f"vwap: price crossed above VWAP ({close_prev:.3f} -> {close_now:.3f}, VWAP {vwap_now:.3f})")
        else:
            reasons.append(f"no cross above: close_prev={close_prev:.3f} close_now={close_now:.3f} vwap={vwap_now:.3f}")
            return False, None, ["vwap: " + "; ".join(reasons)]
    elif policy.trigger == "cross_below":
        if close_prev >= vwap_prev and close_now < vwap_now:
            passed, side = True, "sell"
            reasons.append(f"vwap: price crossed below VWAP ({close_prev:.3f} -> {close_now:.3f}, VWAP {vwap_now:.3f})")
        else:
            reasons.append(f"no cross below: close_prev={close_prev:.3f} close_now={close_now:.3f} vwap={vwap_now:.3f}")
            return False, None, ["vwap: " + "; ".join(reasons)]
    elif policy.trigger == "above_buy":
        if close_now > vwap_now:
            passed, side = True, "buy"
            reasons.append(f"vwap: price above VWAP ({close_now:.3f} > {vwap_now:.3f})")
        else:
            reasons.append(f"price not above VWAP: close={close_now:.3f} vwap={vwap_now:.3f}")
            return False, None, ["vwap: " + "; ".join(reasons)]
    elif policy.trigger == "below_sell":
        if close_now < vwap_now:
            passed, side = True, "sell"
            reasons.append(f"vwap: price below VWAP ({close_now:.3f} < {vwap_now:.3f})")
        else:
            reasons.append(f"price not below VWAP: close={close_now:.3f} vwap={vwap_now:.3f}")
            return False, None, ["vwap: " + "; ".join(reasons)]
    else:
        return False, None, ["vwap: unknown trigger"]
    if not passed or side is None:
        return False, None, ["vwap: " + "; ".join(reasons)]
    # --- Post-trigger filters: no-trade zone, session, slope ---
    pip_size = float(profile.pip_size)
    if policy.no_trade_zone_pips > 0:
        zone_dist = abs(close_now - vwap_now)
        required = policy.no_trade_zone_pips * pip_size
        if zone_dist < required:
            reasons.append(f"inside no-trade zone: |close-vwap|={zone_dist:.5f} < {required:.5f}")
            return False, None, ["vwap: " + "; ".join(reasons)]
    if policy.session_filter_enabled:
        hour_utc = datetime.now(timezone.utc).hour
        in_london = 8 <= hour_utc < 16
        in_ny = 13 <= hour_utc < 21
        if not (in_london or in_ny):
            reasons.append(f"outside session: UTC hour={hour_utc} (London 8-16, NY 13-21)")
            return False, None, ["vwap: " + "; ".join(reasons)]
    if policy.use_slope_filter:
        lb = policy.vwap_slope_lookback_bars
        if len(vwap_series) <= lb:
            reasons.append(f"slope filter: not enough bars ({len(vwap_series)} <= {lb})")
            return False, None, ["vwap: " + "; ".join(reasons)]
        vwap_old = float(vwap_series.iloc[-1 - lb])
        slope = (vwap_now - vwap_old) / lb
        if side == "buy" and slope <= 0:
            reasons.append(f"VWAP slope not positive: slope={slope:.6f}")
            return False, None, ["vwap: " + "; ".join(reasons)]
        if side == "sell" and slope >= 0:
            reasons.append(f"VWAP slope not negative: slope={slope:.6f}")
            return False, None, ["vwap: " + "; ".join(reasons)]
    return True, side, reasons


def _vwap_candidate(
    profile: ProfileV1,
    policy: ExecutionPolicyVWAP,
    entry_price: float,
    side: str,
) -> TradeCandidate:
    pip = float(profile.pip_size)
    if side == "buy":
        target = entry_price + policy.tp_pips * pip if policy.tp_pips else None
        stop = (entry_price - policy.sl_pips * pip) if policy.sl_pips else None
    else:
        target = entry_price - policy.tp_pips * pip if policy.tp_pips else None
        stop = (entry_price + policy.sl_pips * pip) if policy.sl_pips else None
    return TradeCandidate(
        symbol=profile.symbol,
        side=side,
        entry_price=entry_price,
        stop_price=stop,
        target_price=target,
        size_lots=float(get_effective_risk(profile).max_lots),
    )


def execute_vwap_policy_demo_only(
    *,
    adapter,
    profile: ProfileV1,
    log_dir: Path,
    policy: ExecutionPolicyVWAP,
    context: MarketContext,
    data_by_tf: dict[Timeframe, pd.DataFrame],
    tick: Tick,
    trades_df: Optional[pd.DataFrame],
    mode: str,
    bar_time_utc: str,
) -> ExecutionDecision:
    """Evaluate VWAP policy and optionally place a market order."""
    if mode not in ("ARMED_MANUAL_CONFIRM", "ARMED_AUTO_DEMO"):
        return ExecutionDecision(attempted=False, placed=False, reason=f"mode={mode} not armed")
    if not adapter.is_demo_account():
        return ExecutionDecision(attempted=False, placed=False, reason="not a demo account (execution disabled)")
    store = _store(log_dir)
    rule_id = f"vwap:{policy.id}:{policy.timeframe}:{bar_time_utc}"
    bar_minutes = {"M1": 2, "M15": 16, "H4": 250}
    within = bar_minutes.get(policy.timeframe, 60)
    if store.has_recent_price_level_placement(profile.profile_name, rule_id, within):
        return ExecutionDecision(attempted=False, placed=False, reason="vwap: recent placement (idempotent)")
    passed, side, eval_reasons = evaluate_vwap_conditions(profile, policy, data_by_tf)
    if not passed or side is None:
        store.insert_execution(
            {
                "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "profile": profile.profile_name,
                "symbol": profile.symbol,
                "signal_id": f"{rule_id}:{pd.Timestamp.now(tz='UTC').isoformat()}",
                "rule_id": rule_id,
                "mode": mode,
                "attempted": 1,
                "placed": 0,
                "reason": "; ".join(eval_reasons),
                "mt5_retcode": None,
                "mt5_order_id": None,
                "mt5_deal_id": None,
            }
        )
        return ExecutionDecision(attempted=True, placed=False, reason="; ".join(eval_reasons))
    entry_price = (tick.bid + tick.ask) / 2.0
    candidate = _vwap_candidate(profile, policy, entry_price, side)
    decision = evaluate_trade(profile=profile, candidate=candidate, context=context, trades_df=trades_df)
    if not decision.allow:
        sig_id = f"{rule_id}:{pd.Timestamp.now(tz='UTC').isoformat()}"
        store.insert_execution(
            {
                "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "profile": profile.profile_name,
                "symbol": profile.symbol,
                "signal_id": sig_id,
                "rule_id": rule_id,
                "mode": mode,
                "attempted": 1,
                "placed": 0,
                "reason": "risk_reject: " + "; ".join(decision.hard_reasons),
                "mt5_retcode": None,
                "mt5_order_id": None,
                "mt5_deal_id": None,
            }
        )
        return ExecutionDecision(attempted=True, placed=False, reason="risk rejected: " + "; ".join(decision.hard_reasons))
    if mode == "ARMED_MANUAL_CONFIRM":
        sig_id = f"{rule_id}:{pd.Timestamp.now(tz='UTC').isoformat()}"
        store.insert_execution(
            {
                "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "profile": profile.profile_name,
                "symbol": profile.symbol,
                "signal_id": sig_id,
                "rule_id": rule_id,
                "mode": mode,
                "attempted": 1,
                "placed": 0,
                "reason": "manual_confirm_required",
                "mt5_retcode": None,
                "mt5_order_id": None,
                "mt5_deal_id": None,
            }
        )
        return ExecutionDecision(attempted=True, placed=False, reason="manual confirm required")
    res = adapter.order_send_market(
        symbol=profile.symbol,
        side=side,
        volume_lots=float(candidate.size_lots or get_effective_risk(profile).max_lots),
        sl=candidate.stop_price,
        tp=candidate.target_price,
        comment=f"vwap:{policy.id}",
    )
    placed = res.retcode in (0, 10008, 10009)  # 0=OANDA success; 10008/10009=MT5
    reason = "order_sent" if placed else f"order_failed:{res.retcode}:{res.comment}"
    sig_id = f"{rule_id}:{pd.Timestamp.now(tz='UTC').isoformat()}"
    store.insert_execution(
        {
            "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "profile": profile.profile_name,
            "symbol": profile.symbol,
            "signal_id": sig_id,
            "rule_id": rule_id,
            "mode": mode,
            "attempted": 1,
            "placed": 1 if placed else 0,
            "reason": reason,
            "mt5_retcode": res.retcode,
            "mt5_order_id": res.order,
            "mt5_deal_id": res.deal,
        }
    )
    return ExecutionDecision(
        attempted=True,
        placed=placed,
        reason=reason,
        order_retcode=res.retcode,
        order_id=res.order,
        deal_id=res.deal,
        side=side,
    )


# ---------------------------------------------------------------------------
# EMA Pullback Policy (M5-M15 momentum pullback)
# ---------------------------------------------------------------------------


def evaluate_ema_pullback_conditions(
    profile: ProfileV1,
    policy: ExecutionPolicyEmaPullback,
    data_by_tf: dict[Timeframe, pd.DataFrame],
) -> tuple[bool, Optional[str], list[str]]:
    """Trend from EMA 50/200 on trend_timeframe; entry when price in EMA 20-50 zone on entry_timeframe.
    Returns (passed, side, reasons)."""
    reasons: list[str] = []
    trend_df = data_by_tf.get(policy.trend_timeframe)
    entry_df = data_by_tf.get(policy.entry_timeframe)
    if trend_df is None or trend_df.empty:
        return False, None, [f"ema_pullback: no {policy.trend_timeframe} data"]
    if entry_df is None or entry_df.empty:
        return False, None, [f"ema_pullback: no {policy.entry_timeframe} data"]
    trend_df = drop_incomplete_last_bar(trend_df.copy(), policy.trend_timeframe)
    entry_df = drop_incomplete_last_bar(entry_df.copy(), policy.entry_timeframe)
    if len(trend_df) < policy.ema_trend_slow:
        return False, None, [f"ema_pullback: trend TF needs at least {policy.ema_trend_slow} bars"]
    if len(entry_df) < policy.ema_zone_high:
        return False, None, [f"ema_pullback: entry TF needs at least {policy.ema_zone_high} bars"]
    close_trend = trend_df["close"]
    ema50_t = ema_fn(close_trend, policy.ema_trend_fast)
    ema200_t = ema_fn(close_trend, policy.ema_trend_slow)
    bull = float(ema50_t.iloc[-1]) > float(ema200_t.iloc[-1])
    close_entry = entry_df["close"]
    ema_zone_low = ema_fn(close_entry, policy.ema_zone_low)
    ema_zone_high = ema_fn(close_entry, policy.ema_zone_high)
    zone_lo = min(float(ema_zone_low.iloc[-1]), float(ema_zone_high.iloc[-1]))
    zone_hi = max(float(ema_zone_low.iloc[-1]), float(ema_zone_high.iloc[-1]))
    price = float(close_entry.iloc[-1])
    pip = float(profile.pip_size)
    tolerance = pip * 2.0
    in_zone = (zone_lo - tolerance) <= price <= (zone_hi + tolerance)
    if not in_zone:
        return False, None, [f"ema_pullback: price {price:.5f} not in zone [{zone_lo:.5f}, {zone_hi:.5f}]"]
    if bull:
        return True, "buy", [f"ema_pullback: bull trend (EMA50>EMA200), price in EMA{policy.ema_zone_low}-{policy.ema_zone_high} zone"]
    else:
        return True, "sell", [f"ema_pullback: bear trend (EMA50<EMA200), price in EMA{policy.ema_zone_low}-{policy.ema_zone_high} zone"]


def _ema_pullback_candidate(
    profile: ProfileV1,
    policy: ExecutionPolicyEmaPullback,
    entry_price: float,
    side: str,
) -> TradeCandidate:
    pip = float(profile.pip_size)
    sl_pips = policy.sl_pips if policy.sl_pips is not None else float(get_effective_risk(profile).min_stop_pips)
    if side == "buy":
        target = entry_price + policy.tp_pips * pip if policy.tp_pips else None
        stop = entry_price - sl_pips * pip
    else:
        target = entry_price - policy.tp_pips * pip if policy.tp_pips else None
        stop = entry_price + sl_pips * pip
    return TradeCandidate(
        symbol=profile.symbol,
        side=side,
        entry_price=entry_price,
        stop_price=stop,
        target_price=target,
        size_lots=float(get_effective_risk(profile).max_lots),
    )


def execute_ema_pullback_policy_demo_only(
    *,
    adapter,
    profile: ProfileV1,
    log_dir: Path,
    policy: ExecutionPolicyEmaPullback,
    context: MarketContext,
    data_by_tf: dict[Timeframe, pd.DataFrame],
    tick: Tick,
    trades_df: Optional[pd.DataFrame],
    mode: str,
    bar_time_utc: str,
) -> ExecutionDecision:
    """Evaluate EMA pullback policy and optionally place a market order."""
    if mode not in ("ARMED_MANUAL_CONFIRM", "ARMED_AUTO_DEMO"):
        return ExecutionDecision(attempted=False, placed=False, reason=f"mode={mode} not armed")
    if not adapter.is_demo_account():
        return ExecutionDecision(attempted=False, placed=False, reason="not a demo account (execution disabled)")
    store = _store(log_dir)
    rule_id = f"ema_pullback:{policy.id}:{policy.entry_timeframe}:{bar_time_utc}"
    bar_minutes = {"M5": 6, "M15": 16}
    within = bar_minutes.get(policy.entry_timeframe, 10)
    if store.has_recent_price_level_placement(profile.profile_name, rule_id, within):
        return ExecutionDecision(attempted=False, placed=False, reason="ema_pullback: recent placement (idempotent)")
    passed, side, eval_reasons = evaluate_ema_pullback_conditions(profile, policy, data_by_tf)
    if not passed or side is None:
        store.insert_execution(
            {
                "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "profile": profile.profile_name,
                "symbol": profile.symbol,
                "signal_id": f"{rule_id}:{pd.Timestamp.now(tz='UTC').isoformat()}",
                "rule_id": rule_id,
                "mode": mode,
                "attempted": 1,
                "placed": 0,
                "reason": "; ".join(eval_reasons),
                "mt5_retcode": None,
                "mt5_order_id": None,
                "mt5_deal_id": None,
            }
        )
        return ExecutionDecision(attempted=True, placed=False, reason="; ".join(eval_reasons))
    entry_price = (tick.bid + tick.ask) / 2.0
    candidate = _ema_pullback_candidate(profile, policy, entry_price, side)
    decision = evaluate_trade(profile=profile, candidate=candidate, context=context, trades_df=trades_df)
    if not decision.allow:
        sig_id = f"{rule_id}:{pd.Timestamp.now(tz='UTC').isoformat()}"
        store.insert_execution(
            {
                "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "profile": profile.profile_name,
                "symbol": profile.symbol,
                "signal_id": sig_id,
                "rule_id": rule_id,
                "mode": mode,
                "attempted": 1,
                "placed": 0,
                "reason": "risk_reject: " + "; ".join(decision.hard_reasons),
                "mt5_retcode": None,
                "mt5_order_id": None,
                "mt5_deal_id": None,
            }
        )
        return ExecutionDecision(attempted=True, placed=False, reason="risk rejected: " + "; ".join(decision.hard_reasons))
    if mode == "ARMED_MANUAL_CONFIRM":
        sig_id = f"{rule_id}:{pd.Timestamp.now(tz='UTC').isoformat()}"
        store.insert_execution(
            {
                "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "profile": profile.profile_name,
                "symbol": profile.symbol,
                "signal_id": sig_id,
                "rule_id": rule_id,
                "mode": mode,
                "attempted": 1,
                "placed": 0,
                "reason": "manual_confirm_required",
                "mt5_retcode": None,
                "mt5_order_id": None,
                "mt5_deal_id": None,
            }
        )
        return ExecutionDecision(attempted=True, placed=False, reason="manual confirm required")
    res = adapter.order_send_market(
        symbol=profile.symbol,
        side=side,
        volume_lots=float(candidate.size_lots or get_effective_risk(profile).max_lots),
        sl=candidate.stop_price,
        tp=candidate.target_price,
        comment=f"ep:{policy.id}",
    )
    placed = res.retcode in (0, 10008, 10009)
    reason = "order_sent" if placed else f"order_failed:{res.retcode}:{res.comment}"
    sig_id = f"{rule_id}:{pd.Timestamp.now(tz='UTC').isoformat()}"
    store.insert_execution(
        {
            "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "profile": profile.profile_name,
            "symbol": profile.symbol,
            "signal_id": sig_id,
            "rule_id": rule_id,
            "mode": mode,
            "attempted": 1,
            "placed": 1 if placed else 0,
            "reason": reason,
            "mt5_retcode": res.retcode,
            "mt5_order_id": res.order,
            "mt5_deal_id": res.deal,
        }
    )
    return ExecutionDecision(
        attempted=True,
        placed=placed,
        reason=reason,
        order_retcode=res.retcode,
        order_id=res.order,
        deal_id=res.deal,
        side=side,
    )


# ---------------------------------------------------------------------------
# Session Momentum Policy
# ---------------------------------------------------------------------------


def _get_session_times_utc(session: str) -> tuple[int, int]:
    """Get session open hour in UTC. Returns (open_hour, close_hour)."""
    # Approximate session times in UTC
    sessions = {
        "tokyo": (0, 9),      # Tokyo: 00:00 - 09:00 UTC
        "london": (8, 16),    # London: 08:00 - 16:00 UTC
        "newyork": (13, 21),  # New York: 13:00 - 21:00 UTC
    }
    return sessions.get(session, (0, 24))


def evaluate_session_momentum(
    profile: ProfileV1,
    policy: ExecutionPolicySessionMomentum,
    data_by_tf: dict[Timeframe, pd.DataFrame],
) -> tuple[bool, str | None, list[str]]:
    """Check if session momentum is established and get direction.
    
    Returns: (momentum_established, side, reasons)
    """
    from datetime import datetime, timezone
    
    reasons: list[str] = []
    
    # Get M1 data for precise timing
    df = data_by_tf.get(Timeframe.M1)
    if df is None or df.empty:
        return False, None, ["session_momentum: no M1 data"]
    
    df = df.copy().sort_values("time")
    now_utc = datetime.now(timezone.utc)
    session_open_hour, session_close_hour = _get_session_times_utc(policy.session)
    
    # Check if we're in the session window
    current_hour = now_utc.hour
    if not (session_open_hour <= current_hour < session_close_hour):
        return False, None, [f"session_momentum: outside {policy.session} session hours"]
    
    # Check if we're past the setup period
    minutes_since_open = (current_hour - session_open_hour) * 60 + now_utc.minute
    if minutes_since_open < policy.setup_minutes:
        return False, None, [f"session_momentum: still in setup period ({minutes_since_open} < {policy.setup_minutes} min)"]
    
    # Get session open price (first bar of the session)
    # Filter for bars from today's session
    today = now_utc.date()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    session_start = pd.Timestamp(today, tz="UTC").replace(hour=session_open_hour, minute=0)
    session_df = df[df["time"] >= session_start]
    
    if session_df.empty or len(session_df) < policy.setup_minutes:
        return False, None, ["session_momentum: not enough session data"]
    
    session_open_price = float(session_df.iloc[0]["open"])
    # Get price after setup period
    setup_idx = min(policy.setup_minutes, len(session_df) - 1)
    setup_end_price = float(session_df.iloc[setup_idx]["close"])
    
    pip_size = float(profile.pip_size)
    move_pips = (setup_end_price - session_open_price) / pip_size
    
    if abs(move_pips) < policy.momentum_threshold_pips:
        return False, None, [f"session_momentum: move too small ({abs(move_pips):.1f} < {policy.momentum_threshold_pips} pips)"]
    
    if move_pips > 0:
        return True, "buy", [f"session_momentum: bullish bias established (+{move_pips:.1f} pips)"]
    else:
        return True, "sell", [f"session_momentum: bearish bias established ({move_pips:.1f} pips)"]


def _session_momentum_candidate(
    profile: ProfileV1,
    policy: ExecutionPolicySessionMomentum,
    entry_price: float,
    side: str,
) -> TradeCandidate:
    pip = float(profile.pip_size)
    if side == "buy":
        target = entry_price + policy.tp_pips * pip if policy.tp_pips else None
        stop = (entry_price - policy.sl_pips * pip) if policy.sl_pips else None
    else:
        target = entry_price - policy.tp_pips * pip if policy.tp_pips else None
        stop = (entry_price + policy.sl_pips * pip) if policy.sl_pips else None
    
    return TradeCandidate(
        symbol=profile.symbol,
        side=side,
        entry_price=entry_price,
        stop_price=stop,
        target_price=target,
        size_lots=float(get_effective_risk(profile).max_lots),
    )


def execute_session_momentum_policy_demo_only(
    *,
    adapter,
    profile: ProfileV1,
    log_dir: Path,
    policy: ExecutionPolicySessionMomentum,
    context: MarketContext,
    data_by_tf: dict[Timeframe, pd.DataFrame],
    tick: Tick,
    trades_df: Optional[pd.DataFrame],
    mode: str,
) -> ExecutionDecision:
    """Evaluate session momentum policy and optionally place a market order."""
    from datetime import datetime, timezone
    
    if mode not in ("ARMED_MANUAL_CONFIRM", "ARMED_AUTO_DEMO"):
        return ExecutionDecision(attempted=False, placed=False, reason=f"mode={mode} not armed")

    if not adapter.is_demo_account():
        return ExecutionDecision(attempted=False, placed=False, reason="not a demo account (execution disabled)")

    store = _store(log_dir)
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    rule_id = f"session:{policy.id}:{policy.session}:{today_str}"
    
    # Check idempotency - only one trade per session per day
    within = 60 * 24  # 24 hours in minutes
    if store.has_recent_price_level_placement(profile.profile_name, rule_id, within):
        return ExecutionDecision(attempted=False, placed=False, reason="session_momentum: already traded this session today")

    momentum_established, side, eval_reasons = evaluate_session_momentum(profile, policy, data_by_tf)
    if not momentum_established or side is None:
        store.insert_execution(
            {
                "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "profile": profile.profile_name,
                "symbol": profile.symbol,
                "signal_id": f"{rule_id}:{pd.Timestamp.now(tz='UTC').isoformat()}",
                "rule_id": rule_id,
                "mode": mode,
                "attempted": 1,
                "placed": 0,
                "reason": "; ".join(eval_reasons),
                "mt5_retcode": None,
                "mt5_order_id": None,
                "mt5_deal_id": None,
            }
        )
        return ExecutionDecision(attempted=True, placed=False, reason="; ".join(eval_reasons))

    entry_price = (tick.bid + tick.ask) / 2.0
    candidate = _session_momentum_candidate(profile, policy, entry_price, side)
    decision = evaluate_trade(profile=profile, candidate=candidate, context=context, trades_df=trades_df)
    if not decision.allow:
        sig_id = f"{rule_id}:{pd.Timestamp.now(tz='UTC').isoformat()}"
        store.insert_execution(
            {
                "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "profile": profile.profile_name,
                "symbol": profile.symbol,
                "signal_id": sig_id,
                "rule_id": rule_id,
                "mode": mode,
                "attempted": 1,
                "placed": 0,
                "reason": "risk_reject: " + "; ".join(decision.hard_reasons),
                "mt5_retcode": None,
                "mt5_order_id": None,
                "mt5_deal_id": None,
            }
        )
        return ExecutionDecision(attempted=True, placed=False, reason="risk rejected: " + "; ".join(decision.hard_reasons))

    if mode == "ARMED_MANUAL_CONFIRM":
        sig_id = f"{rule_id}:{pd.Timestamp.now(tz='UTC').isoformat()}"
        store.insert_execution(
            {
                "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "profile": profile.profile_name,
                "symbol": profile.symbol,
                "signal_id": sig_id,
                "rule_id": rule_id,
                "mode": mode,
                "attempted": 1,
                "placed": 0,
                "reason": "manual_confirm_required",
                "mt5_retcode": None,
                "mt5_order_id": None,
                "mt5_deal_id": None,
            }
        )
        return ExecutionDecision(attempted=True, placed=False, reason="manual confirm required")

    res = adapter.order_send_market(
        symbol=profile.symbol,
        side=side,
        volume_lots=float(candidate.size_lots or get_effective_risk(profile).max_lots),
        sl=candidate.stop_price,
        tp=candidate.target_price,
        comment=f"ses:{policy.id}",
    )
    placed = res.retcode in (0, 10008, 10009)  # 0=OANDA success; 10008/10009=MT5
    reason = "order_sent" if placed else f"order_failed:{res.retcode}:{res.comment}"
    sig_id = f"{rule_id}:{pd.Timestamp.now(tz='UTC').isoformat()}"
    store.insert_execution(
        {
            "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "profile": profile.profile_name,
            "symbol": profile.symbol,
            "signal_id": sig_id,
            "rule_id": rule_id,
            "mode": mode,
            "attempted": 1,
            "placed": 1 if placed else 0,
            "reason": reason,
            "mt5_retcode": res.retcode,
            "mt5_order_id": res.order,
            "mt5_deal_id": res.deal,
        }
    )
    return ExecutionDecision(
        attempted=True,
        placed=placed,
        reason=reason,
        order_retcode=res.retcode,
        order_id=res.order,
        deal_id=res.deal,
        side=side,
    )


def execute_signal_demo_only(
    *,
    adapter,
    profile: ProfileV1,
    log_dir: Path,
    signal: Signal,
    context: MarketContext,
    trades_df: Optional[pd.DataFrame],
    mode: str,
) -> ExecutionDecision:
    """Execute a signal with strong safety checks.

    - demo-only guard enforced
    - idempotency by signal_id
    - uses risk engine to block trades
    """
    if mode not in ("ARMED_MANUAL_CONFIRM", "ARMED_AUTO_DEMO"):
        return ExecutionDecision(attempted=False, placed=False, reason=f"mode={mode} not armed")

    if not adapter.is_demo_account():
        return ExecutionDecision(attempted=False, placed=False, reason="not a demo account (execution disabled)")

    store = _store(log_dir)
    if signal.signal_id in store.executed_signal_ids(profile.profile_name):
        return ExecutionDecision(attempted=False, placed=False, reason="signal already executed (idempotent)")

    candidate = build_default_candidate_from_signal(profile, signal)

    # Risk check
    decision = evaluate_trade(profile=profile, candidate=candidate, context=context, trades_df=trades_df)
    if not decision.allow:
        store.insert_execution(
            {
                "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "profile": profile.profile_name,
                "symbol": profile.symbol,
                "signal_id": signal.signal_id,
                "mode": mode,
                "attempted": 1,
                "placed": 0,
                "reason": "risk_reject: " + "; ".join(decision.hard_reasons),
                "mt5_retcode": None,
                "mt5_order_id": None,
                "mt5_deal_id": None,
            }
        )
        return ExecutionDecision(attempted=True, placed=False, reason="risk rejected: " + "; ".join(decision.hard_reasons))

    if mode == "ARMED_MANUAL_CONFIRM":
        # In manual confirm mode, do not place order automatically.
        store.insert_execution(
            {
                "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "profile": profile.profile_name,
                "symbol": profile.symbol,
                "signal_id": signal.signal_id,
                "mode": mode,
                "attempted": 1,
                "placed": 0,
                "reason": "manual_confirm_required",
                "mt5_retcode": None,
                "mt5_order_id": None,
                "mt5_deal_id": None,
            }
        )
        return ExecutionDecision(attempted=True, placed=False, reason="manual confirm required")

    # Auto demo mode: place order
    res = adapter.order_send_market(
        symbol=profile.symbol,
        side=signal.side,
        volume_lots=float(candidate.size_lots or get_effective_risk(profile).max_lots),
        sl=candidate.stop_price,
        tp=candidate.target_price,
        comment=f"sig:{signal.signal_id}",
    )

    placed = res.retcode in (0, 10008, 10009)  # 0=OANDA; 10008/10009=MT5 TRADE_RETCODE_DONE
    reason = "order_sent" if placed else f"order_failed:{res.retcode}:{res.comment}"

    # Lightweight reconciliation: verify a position exists after a 'done' retcode.
    if placed:
        pos = adapter.get_open_positions(profile.symbol)
        if pos is None or len(pos) == 0:
            placed = False
            reason = "reconcile_failed:no_position_after_order"

    store.insert_execution(
        {
            "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "profile": profile.profile_name,
            "symbol": profile.symbol,
            "signal_id": signal.signal_id,
            "mode": mode,
            "attempted": 1,
            "placed": 1 if placed else 0,
            "reason": reason,
            "mt5_retcode": res.retcode,
            "mt5_order_id": res.order,
            "mt5_deal_id": res.deal,
        }
    )

    return ExecutionDecision(
        attempted=True,
        placed=placed,
        reason=reason,
        order_retcode=res.retcode,
        order_id=res.order,
        deal_id=res.deal,
        side=signal.side,
    )

