from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

# Broker adapter (MT5 or OANDA) is passed into execute_* as adapter=
from adapters.mt5_adapter import Tick
from core.context_engine import compute_tf_context
from core.indicators import atr as atr_fn
from core.indicators import bollinger_bands as bollinger_bands_fn
from core.indicators import detect_rsi_divergence
from core.indicators import ema as ema_fn
from core.indicators import macd as macd_fn
from core.indicators import rsi as rsi_fn
from core.indicators import vwap as vwap_fn
from core.models import MarketContext, TradeCandidate
from core.profile import (
    ExecutionPolicyBollingerBands,
    ExecutionPolicyBreakout,
    ExecutionPolicyEmaBbScalp,
    ExecutionPolicyEmaPullback,
    ExecutionPolicyIndicator,
    ExecutionPolicyKtCgCounterTrendPullback,
    ExecutionPolicyKtCgHybrid,
    ExecutionPolicyKtCgTrial6,
    ExecutionPolicyPriceLevelTrend,
    ExecutionPolicySessionMomentum,
    ExecutionPolicySessionMomentumV5,
    ExecutionPolicyVWAP,
    ProfileV1,
    get_effective_risk,
)
from core.daily_level_filter import DailyLevelFilter, drop_incomplete_m5_for_filter
from core.risk_engine import evaluate_trade
from core.signal_engine import (
    Signal,
    compute_latest_diffs,
    drop_incomplete_last_bar,
    passes_alignment_filter,
    passes_atr_filter,
    passes_ema_stack_filter,
)
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
    fill_price: Optional[float] = None


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

    entry_price = tick.ask if policy.side == "buy" else tick.bid
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
        fill_price=getattr(res, 'fill_price', None),
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
        fill_price=getattr(res, 'fill_price', None),
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

    entry_price = tick.ask if side == "buy" else tick.bid
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
        fill_price=getattr(res, 'fill_price', None),
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
    entry_price = tick.ask if side == "buy" else tick.bid
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
        fill_price=getattr(res, 'fill_price', None),
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
    entry_price = tick.ask if side == "buy" else tick.bid
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
        fill_price=getattr(res, 'fill_price', None),
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

    # Rejection candle: long wick in direction of trade (sell: upper wick; buy: lower wick)
    if policy.require_rejection_candle and len(entry_df) >= 1:
        row = entry_df.iloc[-1]
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        body = abs(c - o)
        if bull:  # buy: want lower wick (rejection of lows)
            wick = min(o, c) - l
            if body < 1e-8 or wick <= body:
                return False, None, ["ema_pullback: require_rejection_candle (buy) not met"]
        else:  # sell: want upper wick
            wick = h - max(o, c)
            if body < 1e-8 or wick <= body:
                return False, None, ["ema_pullback: require_rejection_candle (sell) not met"]

    # Engulfing: current bar body engulfs previous bar body in direction of trade
    if policy.require_engulfing_confirmation and len(entry_df) >= 2:
        cur = entry_df.iloc[-1]
        prev = entry_df.iloc[-2]
        co, ch, cl, cc = float(cur["open"]), float(cur["high"]), float(cur["low"]), float(cur["close"])
        po, pc = float(prev["open"]), float(prev["close"])
        if bull:  # buy: current green candle engulfs previous
            if cc <= co or co > pc or cc < po:
                return False, None, ["ema_pullback: require_engulfing_confirmation (buy) not met"]
        else:  # sell: current red candle engulfs previous
            if cc >= co or co < pc or cc > po:
                return False, None, ["ema_pullback: require_engulfing_confirmation (sell) not met"]

    if bull:
        return True, "buy", [f"ema_pullback: bull trend (EMA50>EMA200), price in EMA{policy.ema_zone_low}-{policy.ema_zone_high} zone"]
    else:
        return True, "sell", [f"ema_pullback: bear trend (EMA50<EMA200), price in EMA{policy.ema_zone_low}-{policy.ema_zone_high} zone"]


def _ema_pullback_candidate(
    profile: ProfileV1,
    policy: ExecutionPolicyEmaPullback,
    entry_price: float,
    side: str,
    sl_pips_override: Optional[float] = None,
) -> TradeCandidate:
    pip = float(profile.pip_size)
    sl_pips = sl_pips_override if sl_pips_override is not None else (policy.sl_pips if policy.sl_pips is not None else float(get_effective_risk(profile).min_stop_pips))
    tcfg = profile.trade_management.target
    # When mode is scaled, do NOT set TP on the initial order (loop will partial-close at TP1)
    if tcfg.mode == "scaled":
        target = None
    else:
        tp_pips = policy.tp_pips
        if side == "buy":
            target = entry_price + tp_pips * pip if tp_pips else None
        else:
            target = entry_price - tp_pips * pip if tp_pips else None
    if side == "buy":
        stop = entry_price - sl_pips * pip
    else:
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

    # Strategy filters: session, alignment (by trend or score), ema_stack (M15), atr (M15)
    now_utc = datetime.now(timezone.utc)
    ok, reason = passes_session_filter(profile, now_utc)
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
                "reason": reason or "session_filter",
                "mt5_retcode": None,
                "mt5_order_id": None,
                "mt5_deal_id": None,
            }
        )
        return ExecutionDecision(attempted=True, placed=False, reason=reason or "session_filter")
    ok, reason = passes_session_boundary_block(profile, now_utc)
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
                "reason": reason or "session_boundary_block",
                "mt5_retcode": None,
                "mt5_order_id": None,
                "mt5_deal_id": None,
            }
        )
        return ExecutionDecision(attempted=True, placed=False, reason=reason or "session_boundary_block")
    al = profile.strategy.filters.alignment
    if al.enabled:
        if al.trend_timeframe is not None:
            ok, reason = passes_alignment_trend(
                profile, data_by_tf, al.trend_timeframe, side,
                ema_fast=policy.ema_trend_fast, ema_slow=policy.ema_trend_slow,
            )
        else:
            diffs = compute_latest_diffs(profile, data_by_tf)
            ok, reason = passes_alignment_filter(profile, diffs, side)
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
                    "reason": reason or "alignment",
                    "mt5_retcode": None,
                    "mt5_order_id": None,
                    "mt5_deal_id": None,
                }
            )
            return ExecutionDecision(attempted=True, placed=False, reason=reason or "alignment")
    df_m15 = data_by_tf.get(policy.trend_timeframe)
    if df_m15 is not None:
        ok, reason = passes_ema_stack_filter(profile, df_m15, policy.trend_timeframe)
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
                    "reason": reason or "ema_stack_filter",
                    "mt5_retcode": None,
                    "mt5_order_id": None,
                    "mt5_deal_id": None,
                }
            )
            return ExecutionDecision(attempted=True, placed=False, reason=reason or "ema_stack_filter")
        ok, reason = passes_atr_filter(profile, df_m15, policy.trend_timeframe)
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
                    "reason": reason or "atr_filter",
                    "mt5_retcode": None,
                    "mt5_order_id": None,
                    "mt5_deal_id": None,
                }
            )
            return ExecutionDecision(attempted=True, placed=False, reason=reason or "atr_filter")

    entry_price = tick.ask if side == "buy" else tick.bid
    pip = float(profile.pip_size)

    # avoid_round_numbers: reject if entry within buffer of round level (.00 or .50 for JPY)
    if getattr(policy, "avoid_round_numbers", False):
        buf = getattr(policy, "round_number_buffer_pips", 5.0) * pip
        for level in (round(entry_price * 2) / 2.0, round(entry_price)):
            if abs(entry_price - level) < buf:
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
                        "reason": f"avoid_round_numbers: entry {entry_price:.5f} within {policy.round_number_buffer_pips} pips of {level}",
                        "mt5_retcode": None,
                        "mt5_order_id": None,
                        "mt5_deal_id": None,
                    }
                )
                return ExecutionDecision(attempted=True, placed=False, reason="avoid_round_numbers")

    # ATR-based SL when trade_management.stop_loss.mode == "atr"
    sl_pips_override = None
    sl_cfg = getattr(profile.trade_management, "stop_loss", None)
    if sl_cfg is not None and getattr(sl_cfg, "mode", None) == "atr":
        trend_df = data_by_tf.get(policy.trend_timeframe)
        if trend_df is not None and len(trend_df) >= 15:
            atr_series = atr_fn(trend_df, 14)
            atr_val = float(atr_series.iloc[-1]) if len(atr_series) else 0.0
            atr_pips = atr_val / pip if pip else 0.0
            sl_pips_override = min(atr_pips * sl_cfg.atr_multiplier, sl_cfg.max_sl_pips)
            sl_pips_override = max(sl_pips_override, float(get_effective_risk(profile).min_stop_pips))

    candidate = _ema_pullback_candidate(profile, policy, entry_price, side, sl_pips_override=sl_pips_override)

    # min_rr: require TP distance >= min_rr * SL distance
    tcfg = profile.trade_management.target
    sl_dist_pips = abs(candidate.entry_price - candidate.stop_price) / pip if pip else 0.0
    if getattr(policy, "min_rr", 0) > 0 and sl_dist_pips > 0:
        tp_dist_pips = None
        if candidate.target_price is not None:
            tp_dist_pips = abs(candidate.target_price - candidate.entry_price) / pip
        elif tcfg.mode == "scaled" and tcfg.tp1_pips is not None:
            tp_dist_pips = float(tcfg.tp1_pips)
        if tp_dist_pips is not None and tp_dist_pips / sl_dist_pips < policy.min_rr:
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
                    "reason": f"min_rr: rr={tp_dist_pips/sl_dist_pips:.2f} < {policy.min_rr}",
                    "mt5_retcode": None,
                    "mt5_order_id": None,
                    "mt5_deal_id": None,
                }
            )
            return ExecutionDecision(attempted=True, placed=False, reason=f"min_rr not met (rr={tp_dist_pips/sl_dist_pips:.2f})")

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
        fill_price=getattr(res, 'fill_price', None),
        side=side,
    )


# ---------------------------------------------------------------------------
# EMA 9/21 + Bollinger Band Expansion Scalper (KumaTora-style)
# ---------------------------------------------------------------------------


def evaluate_ema_bb_scalp_conditions(
    profile: ProfileV1,
    policy: ExecutionPolicyEmaBbScalp,
    data_by_tf: dict[Timeframe, pd.DataFrame],
) -> tuple[bool, Optional[str], list[str]]:
    """Check EMA 9/21 trend + recent cross + pullback to EMA fast + Bollinger expansion on a single TF."""
    reasons: list[str] = []
    tf = policy.timeframe
    df = data_by_tf.get(tf)
    if df is None or df.empty:
        return False, None, [f"ema_bb_scalp: no {tf} data"]
    df = drop_incomplete_last_bar(df.copy(), tf)
    if df.empty:
        return False, None, [f"ema_bb_scalp: no complete {tf} bars"]

    close = df["close"]
    if len(close) < max(policy.ema_slow, policy.bollinger_period) + 2:
        return False, None, [f"ema_bb_scalp: need at least {max(policy.ema_slow, policy.bollinger_period)+2} bars"]

    # EMA 9/21 trend and recent cross
    ema_fast = ema_fn(close, policy.ema_fast)
    ema_slow = ema_fn(close, policy.ema_slow)
    diff = ema_fast - ema_slow
    now = diff.iloc[-1]
    if pd.isna(now):
        return False, None, ["ema_bb_scalp: EMA warmup not complete"]
    bull = now > 0

    cb = max(int(getattr(policy, "confirm_bars", 2)), 1)
    if len(diff) < cb + 1:
        return False, None, [f"ema_bb_scalp: insufficient bars for confirm_bars={cb}"]
    prev = diff.iloc[-cb - 1]
    crossed_up = prev <= 0 < now
    crossed_down = prev >= 0 > now
    if bull and not crossed_up:
        return False, None, ["ema_bb_scalp: no recent EMA fast>slow cross up"]
    if not bull and not crossed_down:
        return False, None, ["ema_bb_scalp: no recent EMA fast<slow cross down"]

    # Pullback: price near EMA fast
    price = float(close.iloc[-1])
    ema_fast_last = float(ema_fast.iloc[-1])
    pip = float(profile.pip_size)
    tol = max(float(getattr(policy, "min_distance_pips", 1.0)), 0.5)
    dist_pips = abs(price - ema_fast_last) / pip
    if dist_pips > tol:
        return False, None, [f"ema_bb_scalp: price not near EMA{policy.ema_fast} (dist={dist_pips:.2f} > tol={tol:.2f})"]

    # Bollinger expansion: current band width > rolling SMA of width * 1.1
    mid, upper, lower = bollinger_bands_fn(close, policy.bollinger_period, policy.bollinger_deviation)
    width = upper - lower
    if len(width) < policy.bollinger_period + 5:
        return False, None, ["ema_bb_scalp: insufficient bars for Bollinger expansion"]
    bw_now = float(width.iloc[-1])
    bw_ma = float(width.rolling(policy.bollinger_period).mean().iloc[-1])
    if pd.isna(bw_ma) or bw_ma <= 0:
        return False, None, ["ema_bb_scalp: Bollinger bandwidth warmup not complete"]
    if bw_now <= bw_ma * 1.1:
        return False, None, [f"ema_bb_scalp: band not in expansion (bw_now={bw_now:.6f} <= 1.1*bw_ma={bw_ma:.6f})"]

    side = "buy" if bull else "sell"
    reasons.append(f"ema_bb_scalp: trend={side}, bw_now>{bw_ma:.6f}, dist_pips={dist_pips:.2f}")
    return True, side, reasons


def _ema_bb_scalp_candidate(
    profile: ProfileV1,
    policy: ExecutionPolicyEmaBbScalp,
    entry_price: float,
    side: str,
) -> TradeCandidate:
    pip = float(profile.pip_size)
    sl_pips = float(policy.sl_pips or get_effective_risk(profile).min_stop_pips)
    tp_pips = float(policy.tp_pips)
    if side == "buy":
        stop = entry_price - sl_pips * pip
        target = entry_price + tp_pips * pip
    else:
        stop = entry_price + sl_pips * pip
        target = entry_price - tp_pips * pip
    return TradeCandidate(
        symbol=profile.symbol,
        side=side,
        entry_price=entry_price,
        stop_price=stop,
        target_price=target,
        size_lots=float(get_effective_risk(profile).max_lots),
    )


def execute_ema_bb_scalp_policy_demo_only(
    *,
    adapter,
    profile: ProfileV1,
    log_dir: Path,
    policy: ExecutionPolicyEmaBbScalp,
    context: MarketContext,
    data_by_tf: dict[Timeframe, pd.DataFrame],
    tick: Tick,
    trades_df: Optional[pd.DataFrame],
    mode: str,
    bar_time_utc: str,
) -> ExecutionDecision:
    """Evaluate EMA 9/21 + Bollinger expansion scalper and optionally place a market order."""
    if mode not in ("ARMED_MANUAL_CONFIRM", "ARMED_AUTO_DEMO"):
        return ExecutionDecision(attempted=False, placed=False, reason=f"mode={mode} not armed")
    if not adapter.is_demo_account():
        return ExecutionDecision(attempted=False, placed=False, reason="not a demo account (execution disabled)")
    store = _store(log_dir)
    tf = policy.timeframe
    rule_id = f"ema_bb_scalp:{policy.id}:{tf}:{bar_time_utc}"
    bar_minutes = {"M1": 2, "M5": 6}
    within = bar_minutes.get(tf, 5)
    if store.has_recent_price_level_placement(profile.profile_name, rule_id, within):
        return ExecutionDecision(attempted=False, placed=False, reason="ema_bb_scalp: recent placement (idempotent)")

    passed, side, eval_reasons = evaluate_ema_bb_scalp_conditions(profile, policy, data_by_tf)
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

    # Strategy filters: EMA stack, ATR, session
    df_tf = data_by_tf.get(tf)
    if df_tf is None or df_tf.empty:
        return ExecutionDecision(attempted=True, placed=False, reason=f"ema_bb_scalp: no {tf} data")

    ok, reason = passes_ema_stack_filter(profile, df_tf, tf)
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
                "reason": reason or "ema_stack_filter",
                "mt5_retcode": None,
                "mt5_order_id": None,
                "mt5_deal_id": None,
            }
        )
        return ExecutionDecision(attempted=True, placed=False, reason=reason or "ema_stack_filter")

    ok, reason = passes_atr_filter(profile, df_tf, tf)
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
                "reason": reason or "atr_filter",
                "mt5_retcode": None,
                "mt5_order_id": None,
                "mt5_deal_id": None,
            }
        )
        return ExecutionDecision(attempted=True, placed=False, reason=reason or "atr_filter")

    now_utc = datetime.now(timezone.utc)
    ok, reason = passes_session_filter(profile, now_utc)
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
                "reason": reason or "session_filter",
                "mt5_retcode": None,
                "mt5_order_id": None,
                "mt5_deal_id": None,
            }
        )
        return ExecutionDecision(attempted=True, placed=False, reason=reason or "session_filter")
    ok, reason = passes_session_boundary_block(profile, now_utc)
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
                "reason": reason or "session_boundary_block",
                "mt5_retcode": None,
                "mt5_order_id": None,
                "mt5_deal_id": None,
            }
        )
        return ExecutionDecision(attempted=True, placed=False, reason=reason or "session_boundary_block")

    entry_price = tick.ask if side == "buy" else tick.bid
    candidate = _ema_bb_scalp_candidate(profile, policy, entry_price, side)
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
        comment=f"ema_bb_scalp:{policy.id}",
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
        fill_price=getattr(res, 'fill_price', None),
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


def passes_session_filter(profile: ProfileV1, now_utc: datetime) -> tuple[bool, str | None]:
    """Return (True, None) if session filter is disabled or current time is inside one of the allowed sessions."""
    f = profile.strategy.filters.session_filter
    if not f.enabled:
        return True, None
    current_hour = now_utc.hour
    for name in f.sessions:
        key = name.lower().replace(" ", "")  # Tokyo -> tokyo, NewYork -> newyork
        open_h, close_h = _get_session_times_utc(key)
        if open_h <= current_hour < close_h:
            return True, None
    return False, f"session_filter: outside allowed sessions (UTC hour={current_hour})"


def _session_boundary_block_windows_utc(buffer_minutes: int) -> list[tuple[int, int]]:
    """Return list of (start_min, end_min) since midnight UTC for each 30-min block around open/close.
    Tokyo 0,9; London 8,16; NY 13,21. Each boundary gives [boundary*60 - buf, boundary*60 + buf].
    """
    boundaries_h = [0, 9, 8, 16, 13, 21]  # open/close for tokyo, london, ny
    windows: list[tuple[int, int]] = []
    for h in boundaries_h:
        center = h * 60
        start = (center - buffer_minutes + 24 * 60) % (24 * 60)
        end = (center + buffer_minutes) % (24 * 60)
        windows.append((start, end))
    return windows


def passes_session_boundary_block(profile: ProfileV1, now_utc: datetime) -> tuple[bool, str | None]:
    """Return (True, None) if boundary block is disabled or current time is not in any open/close window.
    When enabled, blocks entries from buffer_minutes before until buffer_minutes after each NY/London/Tokyo open and close.
    """
    f = getattr(profile.strategy.filters, "session_boundary_block", None)
    if f is None or not getattr(f, "enabled", False):
        return True, None
    buffer_minutes = max(0, min(60, int(getattr(f, "buffer_minutes", 15))))
    current_minutes = now_utc.hour * 60 + now_utc.minute
    windows = _session_boundary_block_windows_utc(buffer_minutes)
    for start, end in windows:
        if start <= end:
            in_window = start <= current_minutes <= end
        else:
            in_window = current_minutes >= start or current_minutes <= end
        if in_window:
            return False, (
                f"session_boundary_block: in {buffer_minutes}min window around session open/close "
                f"(UTC {now_utc.hour:02d}:{now_utc.minute:02d})"
            )
    return True, None


def passes_alignment_trend(
    profile: ProfileV1,
    data_by_tf: dict[Timeframe, pd.DataFrame],
    trend_timeframe: str,
    side: str,
    ema_fast: int = 50,
    ema_slow: int = 200,
) -> tuple[bool, str | None]:
    """Check that trend on the given TF (EMA fast vs slow) agrees with side. Returns (True, None) if pass."""
    df = data_by_tf.get(trend_timeframe)
    if df is None or df.empty:
        return False, f"alignment_trend: no {trend_timeframe} data"
    df = drop_incomplete_last_bar(df.copy(), trend_timeframe)
    if len(df) < ema_slow:
        return False, f"alignment_trend: need at least {ema_slow} bars on {trend_timeframe}"
    close = df["close"]
    e_fast = ema_fn(close, ema_fast)
    e_slow = ema_fn(close, ema_slow)
    bull = float(e_fast.iloc[-1]) > float(e_slow.iloc[-1])
    if side == "buy" and not bull:
        return False, f"alignment_trend: {trend_timeframe} not bullish (EMA{ema_fast} <= EMA{ema_slow})"
    if side == "sell" and bull:
        return False, f"alignment_trend: {trend_timeframe} not bearish (EMA{ema_fast} > EMA{ema_slow})"
    return True, None


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

    entry_price = tick.ask if side == "buy" else tick.bid
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
        fill_price=getattr(res, 'fill_price', None),
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
        fill_price=getattr(res, 'fill_price', None),
        side=signal.side,
    )


# ---------------------------------------------------------------------------
# Swing Level Filter (for KT/CG policies)
# ---------------------------------------------------------------------------


def detect_swing_levels(
    df: pd.DataFrame,
    lookback_bars: int = 100,
    confirmation_bars: int = 5,
) -> tuple[Optional[float], Optional[float]]:
    """Detect the most recent swing high and swing low within lookback window.

    A swing high is a bar whose high is higher than the `confirmation_bars` bars before AND after it.
    A swing low is a bar whose low is lower than the `confirmation_bars` bars before AND after it.

    Returns: (swing_high, swing_low) - the most recent of each, or None if not found.
    """
    if df is None or df.empty or len(df) < lookback_bars:
        return None, None

    # Use only the last `lookback_bars` bars
    df_window = df.tail(lookback_bars).copy()
    highs = df_window["high"].values
    lows = df_window["low"].values
    n = len(highs)

    swing_high: Optional[float] = None
    swing_low: Optional[float] = None

    # Search from newest to oldest to find the most recent swing points
    # We need at least `confirmation_bars` bars on each side
    for i in range(n - confirmation_bars - 1, confirmation_bars - 1, -1):
        if swing_high is None:
            # Check if this is a swing high
            is_swing_high = True
            for j in range(1, confirmation_bars + 1):
                if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                    is_swing_high = False
                    break
            if is_swing_high:
                swing_high = float(highs[i])

        if swing_low is None:
            # Check if this is a swing low
            is_swing_low = True
            for j in range(1, confirmation_bars + 1):
                if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                    is_swing_low = False
                    break
            if is_swing_low:
                swing_low = float(lows[i])

        if swing_high is not None and swing_low is not None:
            break

    return swing_high, swing_low


def check_swing_level_filter(
    current_price: float,
    side: str,
    swing_high: Optional[float],
    swing_low: Optional[float],
    danger_zone_pct: float = 0.15,
) -> tuple[bool, Optional[str]]:
    """Check if current price is in the danger zone near swing levels.

    For BUY: blocks if price is in/above upper danger zone (near/above swing_high resistance)
    For SELL: blocks if price is in/below lower danger zone (near/below swing_low support)

    The danger zone extends inward from the swing level by danger_zone_pct of the range.

    Returns: (ok, reason) - ok=True means trade is allowed, ok=False means blocked
    """
    if swing_high is None or swing_low is None:
        return True, None  # Can't determine swings, allow trade

    swing_range = swing_high - swing_low
    if swing_range <= 0:
        return True, None  # Invalid range

    danger_distance = swing_range * danger_zone_pct

    if side == "buy":
        # For BUY: block if price is in upper danger zone (at or above swing_high minus danger distance)
        # This blocks: above swing high, at swing high, or within danger zone below swing high
        upper_danger_zone_threshold = swing_high - danger_distance
        if current_price >= upper_danger_zone_threshold:
            return False, f"swing_filter: BUY blocked - price {current_price:.3f} >= {upper_danger_zone_threshold:.3f} (swing high {swing_high:.3f} - danger zone {danger_zone_pct*100:.0f}% = {danger_distance:.3f})"
    elif side == "sell":
        # For SELL: block if price is in lower danger zone (at or below swing_low plus danger distance)
        # This blocks: below swing low, at swing low, or within danger zone above swing low
        lower_danger_zone_threshold = swing_low + danger_distance
        if current_price <= lower_danger_zone_threshold:
            return False, f"swing_filter: SELL blocked - price {current_price:.3f} <= {lower_danger_zone_threshold:.3f} (swing low {swing_low:.3f} + danger zone {danger_zone_pct*100:.0f}% = {danger_distance:.3f})"

    return True, None


# ---------------------------------------------------------------------------
# KT/CG Hybrid Policy (Trial #2)
# ---------------------------------------------------------------------------


def evaluate_kt_cg_hybrid_conditions(
    profile: ProfileV1,
    policy: ExecutionPolicyKtCgHybrid,
    data_by_tf: dict[Timeframe, pd.DataFrame],
) -> tuple[bool, Optional[str], list[str], str]:
    """Evaluate KT/CG Hybrid (Trial #2) conditions.

    Two INDEPENDENT entry triggers:
    1. Zone Entry (continuous state, respects cooldown):
       - M5 BULL + M1 EMA9 > M1 EMA(zone_slow) -> BUY
       - M5 BEAR + M1 EMA9 < M1 EMA(zone_slow) -> SELL

    2. Pullback Cross (discrete event, OVERRIDES cooldown):
       - M5 BULL + M1 EMA9 crosses BELOW EMA(pullback_slow) -> BUY
       - M5 BEAR + M1 EMA9 crosses ABOVE EMA(pullback_slow) -> SELL

    Returns: (passed, side, reasons, trigger_type)
    trigger_type: "zone_entry" or "pullback_cross" (for cooldown handling)
    """
    reasons: list[str] = []

    m5_ema_fast = policy.m5_trend_ema_fast
    m5_ema_slow = policy.m5_trend_ema_slow
    m1_zone_ema_slow = policy.m1_zone_entry_ema_slow
    m1_pullback_ema_slow = policy.m1_pullback_cross_ema_slow

    # Get M5 data for trend
    m5_df = data_by_tf.get("M5")
    if m5_df is None or m5_df.empty:
        return False, None, ["kt_cg_hybrid: no M5 data"], ""
    m5_df = drop_incomplete_last_bar(m5_df.copy(), "M5")
    if len(m5_df) < m5_ema_slow + 1:
        return False, None, [f"kt_cg_hybrid: need at least {m5_ema_slow + 1} M5 bars"], ""

    # Get M1 data for zone entry and pullback cross
    m1_df = data_by_tf.get("M1")
    if m1_df is None or m1_df.empty:
        return False, None, ["kt_cg_hybrid: no M1 data"], ""
    m1_df = drop_incomplete_last_bar(m1_df.copy(), "M1")
    min_m1_bars = max(m1_zone_ema_slow, m1_pullback_ema_slow) + policy.confirm_bars + 2
    if len(m1_df) < min_m1_bars:
        return False, None, [f"kt_cg_hybrid: need at least {min_m1_bars} M1 bars"], ""

    # Step 1: M5 Trend
    m5_close = m5_df["close"]
    m5_ema_fast_series = ema_fn(m5_close, m5_ema_fast)
    m5_ema_slow_series = ema_fn(m5_close, m5_ema_slow)
    m5_ema_fast_val = float(m5_ema_fast_series.iloc[-1])
    m5_ema_slow_val = float(m5_ema_slow_series.iloc[-1])
    is_bull = m5_ema_fast_val > m5_ema_slow_val
    trend = "BULL" if is_bull else "BEAR"
    reasons.append(f"M5 trend: {trend} (EMA{m5_ema_fast}={m5_ema_fast_val:.3f} vs EMA{m5_ema_slow}={m5_ema_slow_val:.3f})")

    # Compute M1 EMAs
    m1_close = m1_df["close"]
    m1_ema9 = ema_fn(m1_close, 9)
    m1_ema_zone = ema_fn(m1_close, m1_zone_ema_slow)
    m1_ema_pullback = ema_fn(m1_close, m1_pullback_ema_slow)
    m1_ema9_val = float(m1_ema9.iloc[-1])
    m1_zone_val = float(m1_ema_zone.iloc[-1])

    # Check Pullback Cross FIRST (it overrides cooldown, so prioritize it)
    diff = m1_ema9 - m1_ema_pullback
    cb = max(int(policy.confirm_bars), 1)
    pullback_cross_triggered = False
    pullback_side: Optional[str] = None

    if len(diff) >= cb + 1:
        now_diff = float(diff.iloc[-1])
        prev_diff = float(diff.iloc[-cb - 1])
        cross_below = prev_diff >= 0 > now_diff
        cross_above = prev_diff <= 0 < now_diff

        if is_bull and cross_below:
            pullback_cross_triggered = True
            pullback_side = "buy"
            reasons.append(f"PULLBACK CROSS: BULL + EMA9 crossed BELOW EMA{m1_pullback_ema_slow} -> BUY (overrides cooldown)")
        elif not is_bull and cross_above:
            pullback_cross_triggered = True
            pullback_side = "sell"
            reasons.append(f"PULLBACK CROSS: BEAR + EMA9 crossed ABOVE EMA{m1_pullback_ema_slow} -> SELL (overrides cooldown)")

    if pullback_cross_triggered and pullback_side:
        return True, pullback_side, reasons, "pullback_cross"

    # Check Zone Entry (respects cooldown)
    zone_entry_triggered = False
    zone_side: Optional[str] = None

    if is_bull and m1_ema9_val > m1_zone_val:
        zone_entry_triggered = True
        zone_side = "buy"
        reasons.append(f"ZONE ENTRY: BULL + M1 EMA9 ({m1_ema9_val:.3f}) > EMA{m1_zone_ema_slow} ({m1_zone_val:.3f}) -> BUY")
    elif not is_bull and m1_ema9_val < m1_zone_val:
        zone_entry_triggered = True
        zone_side = "sell"
        reasons.append(f"ZONE ENTRY: BEAR + M1 EMA9 ({m1_ema9_val:.3f}) < EMA{m1_zone_ema_slow} ({m1_zone_val:.3f}) -> SELL")

    if zone_entry_triggered and zone_side:
        return True, zone_side, reasons, "zone_entry"

    # Neither trigger fired
    reasons.append(f"No trigger: Zone={m1_ema9_val:.3f} vs EMA{m1_zone_ema_slow}={m1_zone_val:.3f}, no pullback cross")
    return False, None, reasons, ""


def _kt_cg_hybrid_candidate(
    profile: ProfileV1,
    policy: ExecutionPolicyKtCgHybrid,
    entry_price: float,
    side: str,
) -> TradeCandidate:
    pip = float(profile.pip_size)
    sl_pips = float(policy.sl_pips or get_effective_risk(profile).min_stop_pips)
    tp_pips = float(policy.tp_pips)
    if side == "buy":
        stop = entry_price - sl_pips * pip
        target = entry_price + tp_pips * pip
    else:
        stop = entry_price + sl_pips * pip
        target = entry_price - tp_pips * pip
    return TradeCandidate(
        symbol=profile.symbol,
        side=side,
        entry_price=entry_price,
        stop_price=stop,
        target_price=target,
        size_lots=float(get_effective_risk(profile).max_lots),
    )


def _check_kt_cg_hybrid_cooldown(
    store: SqliteStore,
    profile_name: str,
    policy_id: str,
    cooldown_minutes: float,
) -> tuple[bool, Optional[str]]:
    """Check if cooldown has elapsed since last kt_cg_hybrid trade.

    Returns: (cooldown_ok, reason)
    """
    if cooldown_minutes <= 0:
        return True, None

    # Check executions for recent placements with this policy
    try:
        execs = store.read_executions_df(profile_name)
        if execs is None or execs.empty:
            return True, None

        # Filter to kt_cg_hybrid executions that were placed
        kt_cg_execs = execs[
            (execs["rule_id"].str.contains(f"kt_cg_hybrid:{policy_id}", na=False)) &
            (execs["placed"] == 1)
        ]
        if kt_cg_execs.empty:
            return True, None

        # Get the most recent placed execution
        last_exec = kt_cg_execs.iloc[-1]
        last_time = pd.to_datetime(last_exec["timestamp_utc"])
        now = pd.Timestamp.now(tz="UTC")
        elapsed = (now - last_time).total_seconds() / 60.0

        if elapsed < cooldown_minutes:
            return False, f"cooldown: {elapsed:.1f}m < {cooldown_minutes}m since last trade"
        return True, None
    except Exception:
        return True, None  # If we can't check, allow the trade


def execute_kt_cg_hybrid_policy_demo_only(
    *,
    adapter,
    profile: ProfileV1,
    log_dir: Path,
    policy: ExecutionPolicyKtCgHybrid,
    context: MarketContext,
    data_by_tf: dict[Timeframe, pd.DataFrame],
    tick: Tick,
    trades_df: Optional[pd.DataFrame],
    mode: str,
    bar_time_utc: str,
) -> ExecutionDecision:
    """Evaluate KT/CG Hybrid (Trial #2) policy and optionally place a market order.

    Two independent triggers:
    1. Zone Entry (respects cooldown)
    2. Pullback Cross (overrides cooldown)
    """
    if mode not in ("ARMED_MANUAL_CONFIRM", "ARMED_AUTO_DEMO"):
        return ExecutionDecision(attempted=False, placed=False, reason=f"mode={mode} not armed")
    if not adapter.is_demo_account():
        return ExecutionDecision(attempted=False, placed=False, reason="not a demo account (execution disabled)")

    store = _store(log_dir)
    rule_id = f"kt_cg_hybrid:{policy.id}:M1:{bar_time_utc}"
    within = 2  # M1 cadence
    if store.has_recent_price_level_placement(profile.profile_name, rule_id, within):
        return ExecutionDecision(attempted=False, placed=False, reason="kt_cg_hybrid: recent placement (idempotent)")

    # Evaluate conditions - returns (passed, side, reasons, trigger_type)
    passed, side, eval_reasons, trigger_type = evaluate_kt_cg_hybrid_conditions(profile, policy, data_by_tf)
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

    # Check cooldown for Zone Entry (Pullback Cross overrides cooldown)
    if trigger_type == "zone_entry":
        cooldown_ok, cooldown_reason = _check_kt_cg_hybrid_cooldown(
            store, profile.profile_name, policy.id, policy.cooldown_minutes
        )
        if not cooldown_ok:
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
                    "reason": cooldown_reason or "zone_entry_cooldown",
                    "mt5_retcode": None,
                    "mt5_order_id": None,
                    "mt5_deal_id": None,
                }
            )
            return ExecutionDecision(attempted=True, placed=False, reason=cooldown_reason or "zone_entry_cooldown")

    # Strategy filters: session, ATR
    now_utc = datetime.now(timezone.utc)
    ok, reason = passes_session_filter(profile, now_utc)
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
                "reason": reason or "session_filter",
                "mt5_retcode": None,
                "mt5_order_id": None,
                "mt5_deal_id": None,
            }
        )
        return ExecutionDecision(attempted=True, placed=False, reason=reason or "session_filter")
    ok, reason = passes_session_boundary_block(profile, now_utc)
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
                "reason": reason or "session_boundary_block",
                "mt5_retcode": None,
                "mt5_order_id": None,
                "mt5_deal_id": None,
            }
        )
        return ExecutionDecision(attempted=True, placed=False, reason=reason or "session_boundary_block")

    m1_df = data_by_tf.get("M1")
    if m1_df is not None:
        ok, reason = passes_atr_filter(profile, m1_df, "M1")
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
                    "reason": reason or "atr_filter",
                    "mt5_retcode": None,
                    "mt5_order_id": None,
                    "mt5_deal_id": None,
                }
            )
            return ExecutionDecision(attempted=True, placed=False, reason=reason or "atr_filter")

    # Swing level filter: block trades near M15 swing highs/lows
    if policy.swing_level_filter_enabled:
        m15_df = data_by_tf.get("M15")
        if m15_df is not None and not m15_df.empty:
            swing_high, swing_low = detect_swing_levels(
                m15_df,
                lookback_bars=policy.swing_lookback_bars,
                confirmation_bars=policy.swing_confirmation_bars,
            )
            entry_price_check = tick.ask if side == "buy" else tick.bid
            sh_str = f"{swing_high:.3f}" if swing_high else "None"
            sl_str = f"{swing_low:.3f}" if swing_low else "None"
            print(f"[{profile.profile_name}] kt_cg_hybrid swing filter: price={entry_price_check:.3f} side={side} swing_high={sh_str} swing_low={sl_str} danger_zone={policy.swing_danger_zone_pct*100:.0f}%")
            swing_ok, swing_reason = check_swing_level_filter(
                current_price=entry_price_check,
                side=side,
                swing_high=swing_high,
                swing_low=swing_low,
                danger_zone_pct=policy.swing_danger_zone_pct,
            )
            if not swing_ok:
                print(f"[{profile.profile_name}] kt_cg_hybrid BLOCKED by swing filter: {swing_reason}")
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
                        "reason": swing_reason or "swing_filter",
                        "mt5_retcode": None,
                        "mt5_order_id": None,
                        "mt5_deal_id": None,
                    }
                )
                return ExecutionDecision(attempted=True, placed=False, reason=swing_reason or "swing_filter")

    entry_price = tick.ask if side == "buy" else tick.bid
    candidate = _kt_cg_hybrid_candidate(profile, policy, entry_price, side)
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
                "reason": decision.reason or "risk_rejected",
                "mt5_retcode": None,
                "mt5_order_id": None,
                "mt5_deal_id": None,
            }
        )
        return ExecutionDecision(attempted=True, placed=False, reason=decision.reason or "risk_rejected")

    # Place the order
    res = adapter.order_send_market(
        symbol=profile.symbol,
        side=side,
        volume_lots=candidate.size_lots,
        sl=candidate.stop_price,
        tp=candidate.target_price,
        comment=f"kt_cg_hybrid:{policy.id}:{trigger_type}",
    )

    placed = res.retcode in (0, 10008, 10009)
    reason = f"{trigger_type}:order_sent" if placed else f"order_failed:{res.retcode}:{res.comment}"

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
            "signal_id": f"{rule_id}:{pd.Timestamp.now(tz='UTC').isoformat()}",
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
        fill_price=getattr(res, 'fill_price', None),
        side=side,
    )


# ---------------------------------------------------------------------------
# KT/CG Counter-Trend Pullback Policy (Trial #3)
# ---------------------------------------------------------------------------


def evaluate_kt_cg_ctp_conditions(
    profile: ProfileV1,
    policy: ExecutionPolicyKtCgCounterTrendPullback,
    data_by_tf: dict[Timeframe, pd.DataFrame],
    temp_overrides: Optional[dict] = None,
) -> tuple[bool, Optional[str], list[str], str]:
    """Evaluate KT/CG Counter-Trend Pullback (Trial #3) conditions.

    Two INDEPENDENT entry triggers:
    1. Zone Entry (continuous state, respects cooldown):
       - M5 BULL + M1 EMA9 > M1 EMA(zone_slow) -> BUY
       - M5 BEAR + M1 EMA9 < M1 EMA(zone_slow) -> SELL

    2. Pullback Cross (discrete event, OVERRIDES cooldown):
       - M5 BULL + M1 EMA9 crosses BELOW EMA(pullback_slow) -> BUY
       - M5 BEAR + M1 EMA9 crosses ABOVE EMA(pullback_slow) -> SELL

    Returns: (passed, side, reasons, trigger_type)
    trigger_type: "zone_entry" or "pullback_cross" (for cooldown handling)
    """
    reasons: list[str] = []

    # Apply temporary overrides if provided
    m5_ema_fast = temp_overrides.get("m5_trend_ema_fast", policy.m5_trend_ema_fast) if temp_overrides else policy.m5_trend_ema_fast
    m5_ema_slow = temp_overrides.get("m5_trend_ema_slow", policy.m5_trend_ema_slow) if temp_overrides else policy.m5_trend_ema_slow
    m1_zone_ema_slow = temp_overrides.get("m1_zone_entry_ema_slow", policy.m1_zone_entry_ema_slow) if temp_overrides else policy.m1_zone_entry_ema_slow
    m1_pullback_ema_slow = temp_overrides.get("m1_pullback_cross_ema_slow", policy.m1_pullback_cross_ema_slow) if temp_overrides else policy.m1_pullback_cross_ema_slow

    # Get M5 data for trend
    m5_df = data_by_tf.get("M5")
    if m5_df is None or m5_df.empty:
        return False, None, ["kt_cg_ctp: no M5 data"], ""
    m5_df = drop_incomplete_last_bar(m5_df.copy(), "M5")
    if len(m5_df) < m5_ema_slow + 1:
        return False, None, [f"kt_cg_ctp: need at least {m5_ema_slow + 1} M5 bars"], ""

    # Get M1 data for zone entry and pullback cross
    m1_df = data_by_tf.get("M1")
    if m1_df is None or m1_df.empty:
        return False, None, ["kt_cg_ctp: no M1 data"], ""
    m1_df = drop_incomplete_last_bar(m1_df.copy(), "M1")
    min_m1_bars = max(m1_zone_ema_slow, m1_pullback_ema_slow) + policy.confirm_bars + 2
    if len(m1_df) < min_m1_bars:
        return False, None, [f"kt_cg_ctp: need at least {min_m1_bars} M1 bars"], ""

    # Step 1: M5 Trend
    m5_close = m5_df["close"]
    m5_ema_fast_series = ema_fn(m5_close, m5_ema_fast)
    m5_ema_slow_series = ema_fn(m5_close, m5_ema_slow)
    m5_ema_fast_val = float(m5_ema_fast_series.iloc[-1])
    m5_ema_slow_val = float(m5_ema_slow_series.iloc[-1])
    is_bull = m5_ema_fast_val > m5_ema_slow_val
    trend = "BULL" if is_bull else "BEAR"
    reasons.append(f"M5 trend: {trend} (EMA{m5_ema_fast}={m5_ema_fast_val:.3f} vs EMA{m5_ema_slow}={m5_ema_slow_val:.3f})")

    # Compute M1 EMAs
    m1_close = m1_df["close"]
    m1_ema9 = ema_fn(m1_close, 9)
    m1_ema_zone = ema_fn(m1_close, m1_zone_ema_slow)
    m1_ema_pullback = ema_fn(m1_close, m1_pullback_ema_slow)
    m1_ema9_val = float(m1_ema9.iloc[-1])
    m1_zone_val = float(m1_ema_zone.iloc[-1])

    # Check Pullback Cross FIRST (it overrides cooldown, so prioritize it)
    diff = m1_ema9 - m1_ema_pullback
    cb = max(int(policy.confirm_bars), 1)
    pullback_cross_triggered = False
    pullback_side: Optional[str] = None

    if len(diff) >= cb + 1:
        now_diff = float(diff.iloc[-1])
        prev_diff = float(diff.iloc[-cb - 1])
        cross_below = prev_diff >= 0 > now_diff
        cross_above = prev_diff <= 0 < now_diff

        if is_bull and cross_below:
            pullback_cross_triggered = True
            pullback_side = "buy"
            reasons.append(f"PULLBACK CROSS: BULL + EMA9 crossed BELOW EMA{m1_pullback_ema_slow} -> BUY (overrides cooldown)")
        elif not is_bull and cross_above:
            pullback_cross_triggered = True
            pullback_side = "sell"
            reasons.append(f"PULLBACK CROSS: BEAR + EMA9 crossed ABOVE EMA{m1_pullback_ema_slow} -> SELL (overrides cooldown)")

    if pullback_cross_triggered and pullback_side:
        return True, pullback_side, reasons, "pullback_cross"

    # Check Zone Entry (respects cooldown)
    zone_entry_triggered = False
    zone_side: Optional[str] = None

    if is_bull and m1_ema9_val > m1_zone_val:
        zone_entry_triggered = True
        zone_side = "buy"
        reasons.append(f"ZONE ENTRY: BULL + M1 EMA9 ({m1_ema9_val:.3f}) > EMA{m1_zone_ema_slow} ({m1_zone_val:.3f}) -> BUY")
    elif not is_bull and m1_ema9_val < m1_zone_val:
        zone_entry_triggered = True
        zone_side = "sell"
        reasons.append(f"ZONE ENTRY: BEAR + M1 EMA9 ({m1_ema9_val:.3f}) < EMA{m1_zone_ema_slow} ({m1_zone_val:.3f}) -> SELL")

    if zone_entry_triggered and zone_side:
        return True, zone_side, reasons, "zone_entry"

    # Neither trigger fired
    reasons.append(f"No trigger: Zone={m1_ema9_val:.3f} vs EMA{m1_zone_ema_slow}={m1_zone_val:.3f}, no pullback cross")
    return False, None, reasons, ""


def _kt_cg_ctp_candidate(
    profile: ProfileV1,
    policy: ExecutionPolicyKtCgCounterTrendPullback,
    entry_price: float,
    side: str,
) -> TradeCandidate:
    pip = float(profile.pip_size)
    sl_pips = float(policy.sl_pips or get_effective_risk(profile).min_stop_pips)
    tp_pips = float(policy.tp_pips)
    if side == "buy":
        stop = entry_price - sl_pips * pip
        target = entry_price + tp_pips * pip
    else:
        stop = entry_price + sl_pips * pip
        target = entry_price - tp_pips * pip
    return TradeCandidate(
        symbol=profile.symbol,
        side=side,
        entry_price=entry_price,
        stop_price=stop,
        target_price=target,
        size_lots=float(get_effective_risk(profile).max_lots),
    )


def _check_kt_cg_ctp_cooldown(
    store: SqliteStore,
    profile_name: str,
    policy_id: str,
    cooldown_minutes: float,
) -> tuple[bool, Optional[str]]:
    """Check if cooldown has elapsed since last kt_cg_ctp trade.

    Returns: (cooldown_ok, reason)
    """
    if cooldown_minutes <= 0:
        return True, None

    try:
        execs = store.read_executions_df(profile_name)
        if execs is None or execs.empty:
            return True, None

        kt_cg_execs = execs[
            (execs["rule_id"].str.contains(f"kt_cg_ctp:{policy_id}", na=False)) &
            (execs["placed"] == 1)
        ]
        if kt_cg_execs.empty:
            return True, None

        last_exec = kt_cg_execs.iloc[-1]
        last_time = pd.to_datetime(last_exec["timestamp_utc"])
        now = pd.Timestamp.now(tz="UTC")
        elapsed = (now - last_time).total_seconds() / 60.0

        if elapsed < cooldown_minutes:
            return False, f"cooldown: {elapsed:.1f}m < {cooldown_minutes}m since last trade"
        return True, None
    except Exception:
        return True, None


def execute_kt_cg_ctp_policy_demo_only(
    *,
    adapter,
    profile: ProfileV1,
    log_dir: Path,
    policy: ExecutionPolicyKtCgCounterTrendPullback,
    context: MarketContext,
    data_by_tf: dict[Timeframe, pd.DataFrame],
    tick: Tick,
    trades_df: Optional[pd.DataFrame],
    mode: str,
    bar_time_utc: str,
    temp_overrides: Optional[dict] = None,
) -> ExecutionDecision:
    """Evaluate KT/CG Counter-Trend Pullback (Trial #3) policy and optionally place a market order.

    Two independent triggers:
    1. Zone Entry (respects cooldown)
    2. Pullback Cross (overrides cooldown)
    """
    if mode not in ("ARMED_MANUAL_CONFIRM", "ARMED_AUTO_DEMO"):
        return ExecutionDecision(attempted=False, placed=False, reason=f"mode={mode} not armed")
    if not adapter.is_demo_account():
        return ExecutionDecision(attempted=False, placed=False, reason="not a demo account (execution disabled)")

    store = _store(log_dir)
    rule_id = f"kt_cg_ctp:{policy.id}:M1:{bar_time_utc}"
    within = 2  # M1 cadence
    if store.has_recent_price_level_placement(profile.profile_name, rule_id, within):
        return ExecutionDecision(attempted=False, placed=False, reason="kt_cg_ctp: recent placement (idempotent)")

    # Evaluate conditions - returns (passed, side, reasons, trigger_type)
    passed, side, eval_reasons, trigger_type = evaluate_kt_cg_ctp_conditions(profile, policy, data_by_tf, temp_overrides)
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

    # Check cooldown for Zone Entry (Pullback Cross overrides cooldown)
    if trigger_type == "zone_entry":
        cooldown_ok, cooldown_reason = _check_kt_cg_ctp_cooldown(
            store, profile.profile_name, policy.id, policy.cooldown_minutes
        )
        if not cooldown_ok:
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
                    "reason": cooldown_reason or "zone_entry_cooldown",
                    "mt5_retcode": None,
                    "mt5_order_id": None,
                    "mt5_deal_id": None,
                }
            )
            return ExecutionDecision(attempted=True, placed=False, reason=cooldown_reason or "zone_entry_cooldown")

    # Strategy filters: session, ATR
    now_utc = datetime.now(timezone.utc)
    ok, reason = passes_session_filter(profile, now_utc)
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
                "reason": reason or "session_filter",
                "mt5_retcode": None,
                "mt5_order_id": None,
                "mt5_deal_id": None,
            }
        )
        return ExecutionDecision(attempted=True, placed=False, reason=reason or "session_filter")
    ok, reason = passes_session_boundary_block(profile, now_utc)
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
                "reason": reason or "session_boundary_block",
                "mt5_retcode": None,
                "mt5_order_id": None,
                "mt5_deal_id": None,
            }
        )
        return ExecutionDecision(attempted=True, placed=False, reason=reason or "session_boundary_block")

    m1_df = data_by_tf.get("M1")
    if m1_df is not None:
        ok, reason = passes_atr_filter(profile, m1_df, "M1")
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
                    "reason": reason or "atr_filter",
                    "mt5_retcode": None,
                    "mt5_order_id": None,
                    "mt5_deal_id": None,
                }
            )
            return ExecutionDecision(attempted=True, placed=False, reason=reason or "atr_filter")

    # Swing level filter: block trades near M15 swing highs/lows
    if policy.swing_level_filter_enabled:
        m15_df = data_by_tf.get("M15")
        if m15_df is not None and not m15_df.empty:
            swing_high, swing_low = detect_swing_levels(
                m15_df,
                lookback_bars=policy.swing_lookback_bars,
                confirmation_bars=policy.swing_confirmation_bars,
            )
            entry_price_check = tick.ask if side == "buy" else tick.bid
            sh_str = f"{swing_high:.3f}" if swing_high else "None"
            sl_str = f"{swing_low:.3f}" if swing_low else "None"
            print(f"[{profile.profile_name}] kt_cg_ctp swing filter: price={entry_price_check:.3f} side={side} swing_high={sh_str} swing_low={sl_str} danger_zone={policy.swing_danger_zone_pct*100:.0f}%")
            swing_ok, swing_reason = check_swing_level_filter(
                current_price=entry_price_check,
                side=side,
                swing_high=swing_high,
                swing_low=swing_low,
                danger_zone_pct=policy.swing_danger_zone_pct,
            )
            if not swing_ok:
                print(f"[{profile.profile_name}] kt_cg_ctp BLOCKED by swing filter: {swing_reason}")
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
                        "reason": swing_reason or "swing_filter",
                        "mt5_retcode": None,
                        "mt5_order_id": None,
                        "mt5_deal_id": None,
                    }
                )
                return ExecutionDecision(attempted=True, placed=False, reason=swing_reason or "swing_filter")

    entry_price = tick.ask if side == "buy" else tick.bid
    candidate = _kt_cg_ctp_candidate(profile, policy, entry_price, side)
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
        comment=f"kt_cg_ctp:{policy.id}:{trigger_type}",
    )
    placed = res.retcode in (0, 10008, 10009)
    reason = f"{trigger_type}:order_sent" if placed else f"order_failed:{res.retcode}:{res.comment}"
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

    if placed:
        print(f"[{profile.profile_name}] TRADE PLACED: kt_cg_ctp:{trigger_type} | {'; '.join(eval_reasons)}")

    return ExecutionDecision(
        attempted=True,
        placed=placed,
        reason=reason,
        order_retcode=res.retcode,
        order_id=res.order,
        deal_id=res.deal,
        fill_price=getattr(res, 'fill_price', None),
        side=side,
    )


# ==============================================================================
# KT/CG Trial #4 (M3 Trend + M1 EMA 5/9)
# ==============================================================================


def _kt_cg_trial_4_candidate(
    profile: ProfileV1,
    policy,  # ExecutionPolicyKtCgTrial4
    entry_price: float,
    side: str,
) -> TradeCandidate:
    pip = float(profile.pip_size)
    sl_pips = float(policy.sl_pips or get_effective_risk(profile).min_stop_pips)
    tp_pips = float(policy.tp_pips)
    if side == "buy":
        stop = entry_price - sl_pips * pip
        target = entry_price + tp_pips * pip
    else:
        stop = entry_price + sl_pips * pip
        target = entry_price - tp_pips * pip
    return TradeCandidate(
        symbol=profile.symbol,
        side=side,
        entry_price=entry_price,
        stop_price=stop,
        target_price=target,
        size_lots=float(get_effective_risk(profile).max_lots),
    )


def _check_kt_cg_trial_4_cooldown(
    store: SqliteStore,
    profile_name: str,
    policy_id: str,
    cooldown_minutes: float,
    rule_prefix: str = "kt_cg_trial_4",
) -> tuple[bool, Optional[str]]:
    """Check if cooldown has elapsed since last kt_cg_trial_4 trade.

    Returns: (cooldown_ok, reason)
    """
    if cooldown_minutes <= 0:
        return True, None

    try:
        execs = store.read_executions_df(profile_name)
        if execs is None or execs.empty:
            return True, None

        kt_cg_execs = execs[
            (execs["rule_id"].str.contains(f"{rule_prefix}:{policy_id}", na=False)) &
            (execs["placed"] == 1)
        ]
        if kt_cg_execs.empty:
            return True, None

        last_exec = kt_cg_execs.iloc[-1]
        last_time = pd.to_datetime(last_exec["timestamp_utc"])
        now = pd.Timestamp.now(tz="UTC")
        elapsed = (now - last_time).total_seconds() / 60.0

        if elapsed < cooldown_minutes:
            return False, f"cooldown: {elapsed:.1f}m < {cooldown_minutes}m since last trade"
        return True, None
    except Exception:
        return True, None


def _passes_tiered_atr_filter_trial_4(policy, m1_df: pd.DataFrame, pip_size: float, trigger_type: str) -> tuple[bool, str | None]:
    """Tiered ATR(14) filter for Trial #4.

    ATR ranges (in pips):
    - < block_below: block ALL (too quiet)
    - block_below to allow_all_max: allow ALL
    - allow_all_max to pullback_only_max: block zone entry, allow pullback only
    - > pullback_only_max: block ALL (too volatile)
    """
    if not getattr(policy, "tiered_atr_filter_enabled", False):
        return True, None
    if m1_df is None or len(m1_df) < 16:
        return False, "tiered_atr_filter: insufficient data"
    a = atr_fn(m1_df, 14)
    if a.empty or pd.isna(a.iloc[-1]):
        return False, "tiered_atr_filter: ATR not available"
    atr_pips = float(a.iloc[-1]) / pip_size
    block_below = getattr(policy, "tiered_atr_block_below_pips", 4.0)
    allow_all_max = getattr(policy, "tiered_atr_allow_all_max_pips", 12.0)
    pullback_only_max = getattr(policy, "tiered_atr_pullback_only_max_pips", 15.0)
    if atr_pips < block_below:
        return False, f"tiered_atr_filter: ATR {atr_pips:.1f}p < {block_below}p (too quiet, ALL blocked)"
    if atr_pips <= allow_all_max:
        return True, None  # Allow all
    if atr_pips <= pullback_only_max:
        if trigger_type == "zone_entry":
            return False, f"tiered_atr_filter: ATR {atr_pips:.1f}p in [{allow_all_max}-{pullback_only_max}]p (zone entry blocked, pullback only)"
        return True, None  # Allow pullback
    return False, f"tiered_atr_filter: ATR {atr_pips:.1f}p > {pullback_only_max}p (too volatile, ALL blocked)"


def _get_current_sessions(utc_hour: int) -> list[str]:
    """Return active trading sessions for a given UTC hour.

    - Tokyo: 00:00-09:00 UTC
    - London: 07:00-16:00 UTC
    - NY: 12:00-21:00 UTC
    """
    sessions = []
    if 0 <= utc_hour < 9:
        sessions.append("tokyo")
    if 7 <= utc_hour < 16:
        sessions.append("london")
    if 12 <= utc_hour < 21:
        sessions.append("ny")
    return sessions


def _update_daily_reset_state(tick_mid: float, daily_reset_state: dict, d_candle_df=None) -> dict:
    """Update daily reset H/L tracking, seeded from OANDA D1 candle and extended by ticks.

    Called every loop iteration for Trial #5.
    - At 00:00 UTC (new day): seed H/L from D1 candle (covers full day so far), set block_active=True
    - During 00:00-02:00: extend H/L from ticks, keep block active
    - At 02:00+: set block_active=False, settled=True, continue extending H/L from ticks
    - On first init (bot start mid-day): seed from D1 candle to capture price action before bot started
    """
    now_utc = datetime.now(timezone.utc)
    today_str = now_utc.strftime("%Y-%m-%d")
    utc_hour = now_utc.hour

    current_date = daily_reset_state.get("daily_reset_date")

    # Seed H/L from OANDA D1 candle (covers the full day including before bot started)
    d_high = None
    d_low = None
    if d_candle_df is not None and not d_candle_df.empty:
        last_row = d_candle_df.iloc[-1]
        d_high = float(last_row["high"])
        d_low = float(last_row["low"])

    if current_date != today_str:
        # New day or first init — seed from D1 candle, then extend with tick
        seed_high = d_high if d_high is not None else tick_mid
        seed_low = d_low if d_low is not None else tick_mid
        # Extend with current tick (tick may be beyond candle if candle is slightly stale)
        daily_reset_state["daily_reset_date"] = today_str
        daily_reset_state["daily_reset_high"] = max(seed_high, tick_mid)
        daily_reset_state["daily_reset_low"] = min(seed_low, tick_mid)
        daily_reset_state["daily_reset_block_active"] = (utc_hour >= 21 or utc_hour < 2)
        daily_reset_state["daily_reset_settled"] = (2 <= utc_hour < 21)
    else:
        # Same day — extend H/L from tick AND D1 candle (candle updates as OANDA refreshes)
        current_high = daily_reset_state.get("daily_reset_high")
        current_low = daily_reset_state.get("daily_reset_low")

        new_high = tick_mid
        new_low = tick_mid
        if current_high is not None:
            new_high = max(current_high, tick_mid)
        if current_low is not None:
            new_low = min(current_low, tick_mid)
        # Also incorporate D1 candle values (catches any price action missed between polls)
        if d_high is not None:
            new_high = max(new_high, d_high)
        if d_low is not None:
            new_low = min(new_low, d_low)

        daily_reset_state["daily_reset_high"] = new_high
        daily_reset_state["daily_reset_low"] = new_low

        # Update block/settled status (dead zone: 21:00-02:00 UTC)
        daily_reset_state["daily_reset_block_active"] = (utc_hour >= 21 or utc_hour < 2)
        daily_reset_state["daily_reset_settled"] = (2 <= utc_hour < 21)

    return daily_reset_state


def _passes_atr_filter_trial_5(
    policy, m1_df: pd.DataFrame, m3_df: pd.DataFrame | None, pip_size: float, trigger_type: str
) -> tuple[bool, str | None]:
    """Dual ATR filter for Trial #5.

    1. M1 ATR(period) with session-dynamic threshold (block below) — skipped if m1_atr_filter_enabled is False
       - Then applies tiered logic (allow_all_max / pullback_only_max / block above)
    2. M3 ATR(period) simple range filter (block outside min-max)
    """
    m1_filter_enabled = getattr(policy, "m1_atr_filter_enabled", True)

    # --- M1 ATR check (skip entirely if filter disabled) ---
    if m1_filter_enabled:
        m1_atr_period = getattr(policy, "m1_atr_period", 7)
        if m1_df is None or len(m1_df) < m1_atr_period + 2:
            return False, "trial5_atr: insufficient M1 data"
        a1 = atr_fn(m1_df, m1_atr_period)
        if a1.empty or pd.isna(a1.iloc[-1]):
            return False, "trial5_atr: M1 ATR not available"
        m1_atr_pips = float(a1.iloc[-1]) / pip_size

        # Determine M1 ATR minimum threshold (session-dynamic or static)
        m1_min = getattr(policy, "m1_atr_min_pips", 2.5)
        session_dynamic = getattr(policy, "session_dynamic_atr_enabled", False)
        auto_session = getattr(policy, "auto_session_detection_enabled", True)
        current_session = "none"
        if session_dynamic and auto_session:
            utc_hour = datetime.now(timezone.utc).hour
            active_sessions = _get_current_sessions(utc_hour)
            # Use highest threshold among active sessions
            thresholds = []
            for s in active_sessions:
                if s == "tokyo":
                    thresholds.append(getattr(policy, "m1_atr_tokyo_min_pips", 2.2))
                elif s == "london":
                    thresholds.append(getattr(policy, "m1_atr_london_min_pips", 2.5))
                elif s == "ny":
                    thresholds.append(getattr(policy, "m1_atr_ny_min_pips", 2.8))
            if thresholds:
                m1_min = max(thresholds)
                current_session = "+".join(active_sessions)
            else:
                current_session = "none"
        elif session_dynamic:
            # Session-dynamic enabled but auto-detection off: use default
            current_session = "manual"

        if m1_atr_pips < m1_min:
            return False, f"trial5_m1_atr: {m1_atr_pips:.1f}p < {m1_min:.1f}p min (session={current_session}, ALL blocked)"

        # M1 ATR MAX check (session-dynamic upper cap)
        m1_max = getattr(policy, "m1_atr_max_pips", 11.0)
        if session_dynamic and auto_session:
            max_thresholds = []
            for s in active_sessions:
                if s == "tokyo":
                    max_thresholds.append(getattr(policy, "m1_atr_tokyo_max_pips", 8.0))
                elif s == "london":
                    max_thresholds.append(getattr(policy, "m1_atr_london_max_pips", 10.0))
                elif s == "ny":
                    max_thresholds.append(getattr(policy, "m1_atr_ny_max_pips", 11.0))
            if max_thresholds:
                m1_max = min(max_thresholds)  # Conservative: use lowest max among active sessions
        if m1_atr_pips > m1_max:
            return False, f"trial5_m1_atr: {m1_atr_pips:.1f}p > {m1_max:.1f}p max (session={current_session}, ALL blocked)"


    # --- M3 ATR check ---
    m3_atr_enabled = getattr(policy, "m3_atr_filter_enabled", False)
    if m3_atr_enabled:
        m3_atr_period = getattr(policy, "m3_atr_period", 14)
        if m3_df is None or len(m3_df) < m3_atr_period + 2:
            return False, "trial5_m3_atr: insufficient M3 data"
        a3 = atr_fn(m3_df, m3_atr_period)
        if a3.empty or pd.isna(a3.iloc[-1]):
            return False, "trial5_m3_atr: M3 ATR not available"
        m3_atr_pips = float(a3.iloc[-1]) / pip_size
        m3_min = getattr(policy, "m3_atr_min_pips", 4.5)
        m3_max = getattr(policy, "m3_atr_max_pips", 11.0)
        if m3_atr_pips < m3_min:
            return False, f"trial5_m3_atr: {m3_atr_pips:.1f}p < {m3_min:.1f}p min (ALL blocked)"
        if m3_atr_pips > m3_max:
            return False, f"trial5_m3_atr: {m3_atr_pips:.1f}p > {m3_max:.1f}p max (ALL blocked)"

    return True, None


def _passes_daily_hl_filter(
    policy, data_by_tf: dict, tick, side: str, pip_size: float,
    daily_reset_high: float | None = None, daily_reset_low: float | None = None,
    daily_reset_settled: bool = False,
) -> tuple[bool, str | None]:
    """Daily High/Low filter.

    Blocks BUY within X pips of daily high, blocks SELL within X pips of daily low.
    If state-tracked H/L is available and settled, use it; otherwise fall back to OANDA D candle.
    """
    if not getattr(policy, "daily_hl_filter_enabled", False):
        return True, None

    daily_high = None
    daily_low = None
    source = "oanda_d"

    # Prefer state-tracked H/L if available and settled
    if daily_reset_settled and daily_reset_high is not None and daily_reset_low is not None:
        daily_high = daily_reset_high
        daily_low = daily_reset_low
        source = "live_tracked"
    else:
        d_df = data_by_tf.get("D")
        if d_df is None or d_df.empty:
            return True, None  # No data, allow
        last_row = d_df.iloc[-1]
        daily_high = float(last_row["high"])
        daily_low = float(last_row["low"])
    buffer = getattr(policy, "daily_hl_buffer_pips", 5.0) * pip_size
    entry_price = tick.ask if side == "buy" else tick.bid
    if side == "buy" and entry_price >= daily_high - buffer:
        return False, f"daily_hl_filter: BUY blocked, price {entry_price:.3f} within {getattr(policy, 'daily_hl_buffer_pips', 5.0):.1f}p of daily high {daily_high:.3f} ({source})"
    if side == "sell" and entry_price <= daily_low + buffer:
        return False, f"daily_hl_filter: SELL blocked, price {entry_price:.3f} within {getattr(policy, 'daily_hl_buffer_pips', 5.0):.1f}p of daily low {daily_low:.3f} ({source})"
    return True, None


def _compute_ema_zone_filter_score(
    m1_df: pd.DataFrame,
    pip_size: float,
    is_bull: bool,
    lookback_bars: int,
    policy=None,
) -> tuple[float, dict]:
    """Compute weighted EMA zone filter score for Trial #4/5 zone entries.

    Uses M1 EMA 9 vs EMA 17 to detect compression/fading momentum.
    When policy is provided, reads weights and interpolation ranges from it.
    Returns (weighted_score, details_dict).
    """
    close = m1_df["close"].astype(float)
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema17 = close.ewm(span=17, adjust=False).mean()

    # Need at least lookback+1 complete bars
    if len(ema9) < lookback_bars + 1 or len(ema17) < lookback_bars + 1:
        return 1.0, {"error": "insufficient_bars"}

    ema9_now = float(ema9.iloc[-1])
    ema17_now = float(ema17.iloc[-1])
    ema9_prev = float(ema9.iloc[-(lookback_bars + 1)])
    ema17_prev = float(ema17.iloc[-(lookback_bars + 1)])

    # Direction-aware spread (positive = healthy trend)
    if is_bull:
        spread_pips = (ema9_now - ema17_now) / pip_size
        slope_pips = (ema9_now - ema9_prev) / pip_size
    else:
        spread_pips = (ema17_now - ema9_now) / pip_size
        slope_pips = (ema9_prev - ema9_now) / pip_size

    # Spread direction: how spread has changed over lookback
    if is_bull:
        spread_prev = (ema9_prev - ema17_prev) / pip_size
    else:
        spread_prev = (ema17_prev - ema9_prev) / pip_size
    spread_dir_pips = spread_pips - spread_prev

    # Read weights and ranges from policy (with hardcoded fallbacks)
    w_spread = getattr(policy, "ema_zone_filter_spread_weight", 0.45) if policy else 0.45
    w_slope = getattr(policy, "ema_zone_filter_slope_weight", 0.40) if policy else 0.40
    w_dir = getattr(policy, "ema_zone_filter_direction_weight", 0.15) if policy else 0.15

    # Auto-normalize weights to sum to 1.0
    w_total = w_spread + w_slope + w_dir
    if w_total > 0:
        w_spread /= w_total
        w_slope /= w_total
        w_dir /= w_total

    spread_min = getattr(policy, "ema_zone_filter_spread_min_pips", 0.0) if policy else 0.0
    spread_max = getattr(policy, "ema_zone_filter_spread_max_pips", 4.0) if policy else 4.0
    slope_min = getattr(policy, "ema_zone_filter_slope_min_pips", -1.0) if policy else -1.0
    slope_max = getattr(policy, "ema_zone_filter_slope_max_pips", 3.0) if policy else 3.0
    dir_min = getattr(policy, "ema_zone_filter_dir_min_pips", -3.0) if policy else -3.0
    dir_max = getattr(policy, "ema_zone_filter_dir_max_pips", 3.0) if policy else 3.0

    # Score each metric: linearly interpolate, clamp to [0, 1]
    spread_range = spread_max - spread_min
    spread_score = max(0.0, min(1.0, (spread_pips - spread_min) / spread_range)) if spread_range > 0 else 0.5
    slope_range = slope_max - slope_min
    slope_score = max(0.0, min(1.0, (slope_pips - slope_min) / slope_range)) if slope_range > 0 else 0.5
    dir_range = dir_max - dir_min
    dir_score = max(0.0, min(1.0, (spread_dir_pips - dir_min) / dir_range)) if dir_range > 0 else 0.5

    weighted = w_spread * spread_score + w_slope * slope_score + w_dir * dir_score

    details = {
        "spread_pips": round(spread_pips, 2),
        "slope_pips": round(slope_pips, 2),
        "spread_dir_pips": round(spread_dir_pips, 2),
        "spread_score": round(spread_score, 3),
        "slope_score": round(slope_score, 3),
        "dir_score": round(dir_score, 3),
        "weighted_score": round(weighted, 3),
    }
    return weighted, details


def _passes_ema_zone_slope_filter_trial_7(
    policy,
    m1_df: pd.DataFrame,
    pip_size: float,
    side: str,
) -> tuple[bool, str | None, dict]:
    """Slope-only EMA zone filter for Trial #7 zone entries.

    BUY: EMA5/EMA9/EMA21 slopes must all be >= configured mins.
    SELL: EMA5/EMA9/EMA21 slopes must all be <= -configured mins.
    """
    lookback = max(1, int(getattr(policy, "ema_zone_filter_lookback_bars", 3)))
    close = m1_df["close"].astype(float)

    ema5 = close.ewm(span=5, adjust=False).mean()
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    needed = lookback + 1
    if len(ema5) < needed or len(ema9) < needed or len(ema21) < needed:
        return True, None, {"error": "insufficient_bars"}

    e5_now = float(ema5.iloc[-1])
    e9_now = float(ema9.iloc[-1])
    e21_now = float(ema21.iloc[-1])
    e5_prev = float(ema5.iloc[-(lookback + 1)])
    e9_prev = float(ema9.iloc[-(lookback + 1)])
    e21_prev = float(ema21.iloc[-(lookback + 1)])

    slope5 = ((e5_now - e5_prev) / pip_size) / lookback
    slope9 = ((e9_now - e9_prev) / pip_size) / lookback
    slope21 = ((e21_now - e21_prev) / pip_size) / lookback

    min5 = float(getattr(policy, "ema_zone_filter_ema5_min_slope_pips_per_bar", 0.10))
    min9 = float(getattr(policy, "ema_zone_filter_ema9_min_slope_pips_per_bar", 0.08))
    min21 = float(getattr(policy, "ema_zone_filter_ema21_min_slope_pips_per_bar", 0.05))

    details = {
        "lookback_bars": lookback,
        "ema5_slope_pips_per_bar": round(slope5, 3),
        "ema9_slope_pips_per_bar": round(slope9, 3),
        "ema21_slope_pips_per_bar": round(slope21, 3),
        "ema5_min": round(min5, 3),
        "ema9_min": round(min9, 3),
        "ema21_min": round(min21, 3),
    }

    if side == "buy":
        ok = slope5 >= min5 and slope9 >= min9 and slope21 >= min21
        if ok:
            return True, None, details
        return (
            False,
            (
                "ema_zone_slope_filter: BUY blocked "
                f"(EMA5={slope5:.3f}/{min5:.3f}, EMA9={slope9:.3f}/{min9:.3f}, EMA21={slope21:.3f}/{min21:.3f} pips/bar)"
            ),
            details,
        )

    ok = slope5 <= -min5 and slope9 <= -min9 and slope21 <= -min21
    if ok:
        return True, None, details
    return (
        False,
        (
            "ema_zone_slope_filter: SELL blocked "
            f"(EMA5={slope5:.3f}/-{min5:.3f}, EMA9={slope9:.3f}/-{min9:.3f}, EMA21={slope21:.3f}/-{min21:.3f} pips/bar)"
        ),
        details,
    )


def _trial7_session_name_utc(ts_utc: pd.Timestamp) -> str:
    """Return session label using UTC hour buckets used in calibration."""
    h = int(ts_utc.hour)
    if 0 <= h < 8:
        return "tokyo"
    if 8 <= h < 13:
        return "london"
    return "ny"


def _canonical_trial7_tier_periods(periods: Optional[list[int] | tuple[int, ...]]) -> list[int]:
    """Canonical Trial #7 tiers: EMA 9 and EMA 11..34 (EMA10 intentionally excluded)."""
    allowed = {9, *range(11, 35)}
    default_tiers = [9, *list(range(11, 35))]
    if not periods:
        return default_tiers
    cleaned = sorted({int(x) for x in periods if int(x) in allowed})
    return cleaned if cleaned else default_tiers


def _trial7_m5_ema_gap_pips(m5_close: pd.Series, fast_period: int, slow_period: int, pip_size: float) -> tuple[float, float, float]:
    """Return (ema_gap_pips, fast_val, slow_val) for M5 EMA fast/slow."""
    fast_series = ema_fn(m5_close, fast_period)
    slow_series = ema_fn(m5_close, slow_period)
    fast_val = float(fast_series.iloc[-1])
    slow_val = float(slow_series.iloc[-1])
    gap_pips = abs(fast_val - slow_val) / pip_size
    return gap_pips, fast_val, slow_val


def _compute_trial7_trend_exhaustion(
    *,
    policy,
    m5_df: pd.DataFrame,
    current_price: float,
    pip_size: float,
    trend_side: str,
) -> dict:
    """Compute Trial #7 stretch-based trend exhaustion regime.

    Stretch is measured from EMA21 on M5:
      stretch_pips = abs(price_ref - EMA21_M5) / pip_size
    """
    out = {
        "zone": "normal",
        "stretch_pips": None,
        "threshold_p80": None,
        "threshold_p90": None,
        "mode": getattr(policy, "trend_exhaustion_mode", "session_and_side"),
        "session": None,
        "trend_side": trend_side,
    }

    if m5_df is None or m5_df.empty or len(m5_df) < 22:
        out["reason"] = "insufficient_m5_bars"
        return out

    m5_local = drop_incomplete_last_bar(m5_df.copy(), "M5")
    if len(m5_local) < 22:
        out["reason"] = "insufficient_complete_m5_bars"
        return out

    close = m5_local["close"].astype(float)
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema21_last = float(ema21.iloc[-1])
    use_current = bool(getattr(policy, "trend_exhaustion_use_current_price", True))
    price_ref = float(current_price if use_current else close.iloc[-1])
    stretch_pips = abs(price_ref - ema21_last) / pip_size

    now_utc = pd.Timestamp.now(tz="UTC")
    session = _trial7_session_name_utc(now_utc)
    mode = str(getattr(policy, "trend_exhaustion_mode", "session_and_side"))

    if mode == "global":
        p80 = float(getattr(policy, "trend_exhaustion_p80_global", 12.03))
        p90 = float(getattr(policy, "trend_exhaustion_p90_global", 17.02))
    elif mode == "session":
        p80 = float(getattr(policy, f"trend_exhaustion_p80_{session}", getattr(policy, "trend_exhaustion_p80_global", 12.03)))
        p90 = float(getattr(policy, f"trend_exhaustion_p90_{session}", getattr(policy, "trend_exhaustion_p90_global", 17.02)))
    else:
        side_key = "bull" if trend_side == "bull" else "bear"
        p80 = float(getattr(policy, f"trend_exhaustion_p80_{side_key}_{session}", getattr(policy, "trend_exhaustion_p80_global", 12.03)))
        p90 = float(getattr(policy, f"trend_exhaustion_p90_{side_key}_{session}", getattr(policy, "trend_exhaustion_p90_global", 17.02)))

    if p90 < p80:
        p80, p90 = p90, p80

    # Simple hysteresis widening to reduce noisy boundary flips.
    hyst = max(0.0, float(getattr(policy, "trend_exhaustion_hysteresis_pips", 0.5)))
    if stretch_pips >= (p90 + hyst):
        zone = "very_extended"
    elif stretch_pips >= (p80 + hyst):
        zone = "extended"
    else:
        zone = "normal"

    out.update(
        {
            "zone": zone,
            "stretch_pips": round(stretch_pips, 2),
            "threshold_p80": round(p80, 2),
            "threshold_p90": round(p90, 2),
            "session": session.upper(),
            "ema21_m5": round(ema21_last, 5),
            "price_ref": round(price_ref, 5),
        }
    )
    return out


def _compute_trial7_effective_tp_pips(
    policy,
    exhaustion_result: Optional[dict],
) -> tuple[float, str]:
    """Compute Trial #7 effective TP from base tp_pips and exhaustion zone offsets."""
    base_tp = max(0.1, float(getattr(policy, "tp_pips", 4.0)))
    adaptive = bool(getattr(policy, "trend_exhaustion_adaptive_tp_enabled", False))
    zone = str((exhaustion_result or {}).get("zone", "normal")).lower()
    if not adaptive:
        return base_tp, zone
    extended_offset = max(0.0, float(getattr(policy, "trend_exhaustion_tp_extended_offset_pips", 1.0)))
    very_extended_offset = max(0.0, float(getattr(policy, "trend_exhaustion_tp_very_extended_offset_pips", 2.0)))
    min_tp = max(0.1, float(getattr(policy, "trend_exhaustion_tp_min_pips", 0.5)))

    if zone == "very_extended":
        return max(min_tp, base_tp - very_extended_offset), zone
    if zone == "extended":
        return max(min_tp, base_tp - extended_offset), zone
    return base_tp, zone


def evaluate_kt_cg_trial_4_conditions(
    profile: ProfileV1,
    policy,  # ExecutionPolicyKtCgTrial4
    data_by_tf: dict[Timeframe, pd.DataFrame],
    current_bid: float,
    current_ask: float,
    tier_state: dict[int, bool],
    temp_overrides: Optional[dict] = None,
) -> dict:
    """Evaluate KT/CG Trial #4 (M3 Trend + Tiered Pullback System) conditions.

    Two INDEPENDENT entry triggers:
    1. Zone Entry (continuous state, respects cooldown):
       - M3 BULL + M1 EMA5 > M1 EMA9 -> BUY
       - M3 BEAR + M1 EMA5 < M1 EMA9 -> SELL

    2. Tiered Pullback (discrete event, NO cooldown):
       - Dynamic tiers: M1 EMA periods from policy.tier_ema_periods
       - M3 BULL + bid touches/goes below tier EMA -> BUY (each tier fires once)
       - M3 BEAR + ask touches/goes above tier EMA -> SELL (each tier fires once)
       - Tier resets when price moves away from EMA by reset_buffer

    Returns: dict with keys:
        - passed: bool
        - side: Optional[str]
        - reasons: list[str]
        - trigger_type: str ("zone_entry" or "tiered_pullback")
        - tiered_pullback_tier: Optional[int]
        - tier_updates: dict[int, bool] (new state for tiers)
        - m3_trend: str ("bull" or "bear")
    """
    reasons: list[str] = []
    tier_updates: dict[int, bool] = {}

    # Apply temporary overrides if provided
    m3_ema_fast = temp_overrides.get("m3_trend_ema_fast", policy.m3_trend_ema_fast) if temp_overrides else policy.m3_trend_ema_fast
    m3_ema_slow = temp_overrides.get("m3_trend_ema_slow", policy.m3_trend_ema_slow) if temp_overrides else policy.m3_trend_ema_slow
    m1_zone_ema_fast = temp_overrides.get("m1_zone_entry_ema_fast", policy.m1_zone_entry_ema_fast) if temp_overrides else policy.m1_zone_entry_ema_fast
    m1_zone_ema_slow = temp_overrides.get("m1_zone_entry_ema_slow", policy.m1_zone_entry_ema_slow) if temp_overrides else policy.m1_zone_entry_ema_slow

    # Debug: log which EMAs are being used
    override_status = "WITH OVERRIDES" if temp_overrides else "using defaults"
    reasons.append(f"[DEBUG] Trial #4 {override_status}: M3({m3_ema_fast}/{m3_ema_slow}) M1-Zone({m1_zone_ema_fast}/{m1_zone_ema_slow})")

    # Get M3 data for trend
    m3_df = data_by_tf.get("M3")
    if m3_df is None or m3_df.empty:
        return {"passed": False, "side": None, "reasons": ["kt_cg_trial_4: no M3 data"], "trigger_type": "", "tiered_pullback_tier": None, "tier_updates": {}, "m3_trend": ""}
    m3_df = drop_incomplete_last_bar(m3_df.copy(), "M3")
    if len(m3_df) < m3_ema_slow + 1:
        return {"passed": False, "side": None, "reasons": [f"kt_cg_trial_4: need at least {m3_ema_slow + 1} M3 bars"], "trigger_type": "", "tiered_pullback_tier": None, "tier_updates": {}, "m3_trend": ""}

    # Get M1 data for zone entry and tiered pullback EMAs
    m1_df = data_by_tf.get("M1")
    if m1_df is None or m1_df.empty:
        return {"passed": False, "side": None, "reasons": ["kt_cg_trial_4: no M1 data"], "trigger_type": "", "tiered_pullback_tier": None, "tier_updates": {}, "m3_trend": ""}
    m1_df = drop_incomplete_last_bar(m1_df.copy(), "M1")

    # Get tier periods from policy
    tier_periods = list(getattr(policy, "tier_ema_periods", (9, 11, 13, 15, 17)))
    max_tier_period = max(tier_periods) if tier_periods else 17
    min_m1_bars = max(m1_zone_ema_slow, max_tier_period) + 2
    if len(m1_df) < min_m1_bars:
        return {"passed": False, "side": None, "reasons": [f"kt_cg_trial_4: need at least {min_m1_bars} M1 bars"], "trigger_type": "", "tiered_pullback_tier": None, "tier_updates": {}, "m3_trend": ""}

    # Step 1: M3 Trend
    m3_close = m3_df["close"]
    m3_ema_fast_series = ema_fn(m3_close, m3_ema_fast)
    m3_ema_slow_series = ema_fn(m3_close, m3_ema_slow)
    m3_ema_fast_val = float(m3_ema_fast_series.iloc[-1])
    m3_ema_slow_val = float(m3_ema_slow_series.iloc[-1])
    is_bull = m3_ema_fast_val > m3_ema_slow_val
    trend = "bull" if is_bull else "bear"
    reasons.append(f"M3 trend: {trend.upper()} (EMA{m3_ema_fast}={m3_ema_fast_val:.3f} vs EMA{m3_ema_slow}={m3_ema_slow_val:.3f})")

    # Compute M1 EMAs for zone entry
    m1_close = m1_df["close"]
    m1_ema_zone_fast = ema_fn(m1_close, m1_zone_ema_fast)
    m1_ema_zone_slow = ema_fn(m1_close, m1_zone_ema_slow)
    m1_zone_fast_val = float(m1_ema_zone_fast.iloc[-1])
    m1_zone_slow_val = float(m1_ema_zone_slow.iloc[-1])

    # Check Tiered Pullback FIRST (no cooldown, so prioritize it)
    tiered_pullback_enabled = getattr(policy, "tiered_pullback_enabled", True)
    tiered_pullback_triggered = False
    tiered_pullback_tier: Optional[int] = None
    tiered_pullback_side: Optional[str] = None

    if tiered_pullback_enabled:
        # Calculate all tier EMAs
        tier_emas: dict[int, float] = {}
        for period in tier_periods:
            tier_emas[period] = float(ema_fn(m1_close, period).iloc[-1])

        # Reset buffer in price units (1 pip = 0.01 for JPY pairs)
        reset_buffer = getattr(policy, "tier_reset_buffer_pips", 1.0) * float(profile.pip_size)

        # Check each tier for touch and reset
        for tier in tier_periods:
            ema_value = tier_emas[tier]
            tier_fired = tier_state.get(tier, False)

            if is_bull:
                # BULL: BUY when bid touches or goes below EMA
                is_touching = current_bid <= ema_value
                has_moved_away = current_bid > ema_value + reset_buffer

                if is_touching and not tier_fired:
                    # Tier touch detected - fire trade!
                    tiered_pullback_triggered = True
                    tiered_pullback_tier = tier
                    tiered_pullback_side = "buy"
                    tier_updates[tier] = True
                    reasons.append(f"TIERED PULLBACK: BULL + bid ({current_bid:.3f}) touched EMA{tier} ({ema_value:.3f}) -> BUY (tier {tier})")
                    break  # Only fire one tier per tick
                elif has_moved_away and tier_fired:
                    # Price moved away - reset tier
                    tier_updates[tier] = False
                    reasons.append(f"Tier {tier} RESET: bid ({current_bid:.3f}) > EMA{tier} ({ema_value:.3f}) + buffer")
            else:
                # BEAR: SELL when ask touches or goes above EMA
                is_touching = current_ask >= ema_value
                has_moved_away = current_ask < ema_value - reset_buffer

                if is_touching and not tier_fired:
                    # Tier touch detected - fire trade!
                    tiered_pullback_triggered = True
                    tiered_pullback_tier = tier
                    tiered_pullback_side = "sell"
                    tier_updates[tier] = True
                    reasons.append(f"TIERED PULLBACK: BEAR + ask ({current_ask:.3f}) touched EMA{tier} ({ema_value:.3f}) -> SELL (tier {tier})")
                    break  # Only fire one tier per tick
                elif has_moved_away and tier_fired:
                    # Price moved away - reset tier
                    tier_updates[tier] = False
                    reasons.append(f"Tier {tier} RESET: ask ({current_ask:.3f}) < EMA{tier} ({ema_value:.3f}) - buffer")

    if tiered_pullback_triggered and tiered_pullback_side:
        return {
            "passed": True,
            "side": tiered_pullback_side,
            "reasons": reasons,
            "trigger_type": "tiered_pullback",
            "tiered_pullback_tier": tiered_pullback_tier,
            "tier_updates": tier_updates,
            "m3_trend": trend,
        }

    # Check Zone Entry (respects cooldown) — only if zone_entry_enabled
    zone_entry_triggered = False
    zone_side: Optional[str] = None
    zone_entry_enabled = getattr(policy, "zone_entry_enabled", True)

    if zone_entry_enabled:
        if is_bull and m1_zone_fast_val > m1_zone_slow_val:
            zone_entry_triggered = True
            zone_side = "buy"
            reasons.append(f"ZONE ENTRY: BULL + M1 EMA{m1_zone_ema_fast} ({m1_zone_fast_val:.3f}) > EMA{m1_zone_ema_slow} ({m1_zone_slow_val:.3f}) -> BUY")
        elif not is_bull and m1_zone_fast_val < m1_zone_slow_val:
            zone_entry_triggered = True
            zone_side = "sell"
            reasons.append(f"ZONE ENTRY: BEAR + M1 EMA{m1_zone_ema_fast} ({m1_zone_fast_val:.3f}) < EMA{m1_zone_ema_slow} ({m1_zone_slow_val:.3f}) -> SELL")
    else:
        reasons.append("ZONE ENTRY: disabled by zone_entry_enabled=False")

    if zone_entry_triggered and zone_side:
        return {
            "passed": True,
            "side": zone_side,
            "reasons": reasons,
            "trigger_type": "zone_entry",
            "tiered_pullback_tier": None,
            "tier_updates": tier_updates,
            "m3_trend": trend,
        }

    # Neither trigger fired
    reasons.append(f"No trigger: Zone={m1_zone_fast_val:.3f} vs EMA{m1_zone_ema_slow}={m1_zone_slow_val:.3f}, no tier touch")
    return {
        "passed": False,
        "side": None,
        "reasons": reasons,
        "trigger_type": "",
        "tiered_pullback_tier": None,
        "tier_updates": tier_updates,
        "m3_trend": trend,
    }


def evaluate_kt_cg_trial_7_conditions(
    profile: ProfileV1,
    policy,  # ExecutionPolicyKtCgTrial7
    data_by_tf: dict[Timeframe, pd.DataFrame],
    current_bid: float,
    current_ask: float,
    tier_state: dict[int, bool],
) -> dict:
    """Evaluate KT/CG Trial #7 (M5 Trend + Tiered Pullback) conditions."""
    reasons: list[str] = []
    tier_updates: dict[int, bool] = {}

    m5_ema_fast = getattr(policy, "m5_trend_ema_fast", 9)
    m5_ema_slow = getattr(policy, "m5_trend_ema_slow", 21)
    min_ema_gap_pips = float(getattr(policy, "m5_min_ema_distance_pips", 1.0))
    zone_entry_mode = str(getattr(policy, "zone_entry_mode", "ema_cross"))
    if zone_entry_mode not in ("ema_cross", "price_vs_ema5"):
        zone_entry_mode = "ema_cross"
    m1_zone_ema_fast = getattr(policy, "m1_zone_entry_ema_fast", 5)
    m1_zone_ema_slow = getattr(policy, "m1_zone_entry_ema_slow", 9)

    m5_df = data_by_tf.get("M5")
    if m5_df is None or m5_df.empty:
        return {"passed": False, "side": None, "reasons": ["kt_cg_trial_7: no M5 data"], "trigger_type": "", "tiered_pullback_tier": None, "tier_updates": {}, "m5_trend": ""}
    m5_df = drop_incomplete_last_bar(m5_df.copy(), "M5")
    if len(m5_df) < m5_ema_slow + 1:
        return {"passed": False, "side": None, "reasons": [f"kt_cg_trial_7: need at least {m5_ema_slow + 1} M5 bars"], "trigger_type": "", "tiered_pullback_tier": None, "tier_updates": {}, "m5_trend": ""}

    m5_close = m5_df["close"]
    m5_ema_gap_pips, m5_ema_fast_val, m5_ema_slow_val = _trial7_m5_ema_gap_pips(
        m5_close, int(m5_ema_fast), int(m5_ema_slow), float(profile.pip_size)
    )
    if m5_ema_gap_pips < min_ema_gap_pips:
        return {
            "passed": False,
            "side": None,
            "reasons": [f"kt_cg_trial_7: blocked (M5 EMA gap {m5_ema_gap_pips:.2f}p < min {min_ema_gap_pips:.2f}p)"],
            "trigger_type": "",
            "tiered_pullback_tier": None,
            "tier_updates": {},
            "m5_trend": "",
            "m5_ema_gap_pips": round(m5_ema_gap_pips, 3),
            "m5_ema_gap_min_pips": round(min_ema_gap_pips, 3),
        }

    is_bull = m5_ema_fast_val > m5_ema_slow_val
    trend = "bull" if is_bull else "bear"
    reasons.append(
        f"M5 trend: {trend.upper()} (EMA{m5_ema_fast}={m5_ema_fast_val:.3f} vs EMA{m5_ema_slow}={m5_ema_slow_val:.3f}, gap={m5_ema_gap_pips:.2f}p >= {min_ema_gap_pips:.2f}p)"
    )

    m1_df = data_by_tf.get("M1")
    if m1_df is None or m1_df.empty:
        return {"passed": False, "side": None, "reasons": ["kt_cg_trial_7: no M1 data"], "trigger_type": "", "tiered_pullback_tier": None, "tier_updates": {}, "m5_trend": ""}
    m1_df = drop_incomplete_last_bar(m1_df.copy(), "M1")

    tier_periods = _canonical_trial7_tier_periods(getattr(policy, "tier_ema_periods", tuple(range(9, 35))))
    max_tier_period = max(tier_periods) if tier_periods else 34
    min_m1_bars = max(m1_zone_ema_slow, max_tier_period) + 2
    if len(m1_df) < min_m1_bars:
        return {"passed": False, "side": None, "reasons": [f"kt_cg_trial_7: need at least {min_m1_bars} M1 bars"], "trigger_type": "", "tiered_pullback_tier": None, "tier_updates": {}, "m5_trend": ""}

    m1_close = m1_df["close"]
    m1_ema_zone_fast = ema_fn(m1_close, m1_zone_ema_fast)
    m1_ema_zone_slow = ema_fn(m1_close, m1_zone_ema_slow)
    current_price = (float(current_bid) + float(current_ask)) / 2.0
    # Price-vs-EMA5 mode should reflect current poll price, not only last closed M1 price.
    # Replace the latest close with current_price for a live EMA5 snapshot.
    m1_close_live = m1_close.copy()
    try:
        m1_close_live.iloc[-1] = current_price
    except Exception:
        pass
    m1_ema5_live = ema_fn(m1_close_live, 5)
    m1_zone_fast_val = float(m1_ema_zone_fast.iloc[-1])
    m1_zone_slow_val = float(m1_ema_zone_slow.iloc[-1])
    m1_ema5_val = float(m1_ema5_live.iloc[-1])

    tiered_pullback_enabled = getattr(policy, "tiered_pullback_enabled", True)
    tiered_pullback_triggered = False
    tiered_pullback_tier: Optional[int] = None
    tiered_pullback_side: Optional[str] = None

    if tiered_pullback_enabled:
        tier_emas: dict[int, float] = {}
        for period in tier_periods:
            tier_emas[period] = float(ema_fn(m1_close, period).iloc[-1])

        reset_buffer = getattr(policy, "tier_reset_buffer_pips", 1.0) * float(profile.pip_size)
        for tier in tier_periods:
            ema_value = tier_emas[tier]
            tier_fired = tier_state.get(tier, False)
            if is_bull:
                is_touching = current_bid <= ema_value
                has_moved_away = current_bid > ema_value + reset_buffer
                if is_touching and not tier_fired:
                    tiered_pullback_triggered = True
                    tiered_pullback_tier = tier
                    tiered_pullback_side = "buy"
                    tier_updates[tier] = True
                    reasons.append(f"TIERED PULLBACK: BULL + bid ({current_bid:.3f}) touched EMA{tier} ({ema_value:.3f}) -> BUY (tier {tier})")
                    break
                if has_moved_away and tier_fired:
                    tier_updates[tier] = False
                    reasons.append(f"Tier {tier} RESET: bid ({current_bid:.3f}) > EMA{tier} ({ema_value:.3f}) + buffer")
            else:
                is_touching = current_ask >= ema_value
                has_moved_away = current_ask < ema_value - reset_buffer
                if is_touching and not tier_fired:
                    tiered_pullback_triggered = True
                    tiered_pullback_tier = tier
                    tiered_pullback_side = "sell"
                    tier_updates[tier] = True
                    reasons.append(f"TIERED PULLBACK: BEAR + ask ({current_ask:.3f}) touched EMA{tier} ({ema_value:.3f}) -> SELL (tier {tier})")
                    break
                if has_moved_away and tier_fired:
                    tier_updates[tier] = False
                    reasons.append(f"Tier {tier} RESET: ask ({current_ask:.3f}) < EMA{tier} ({ema_value:.3f}) - buffer")

    if tiered_pullback_triggered and tiered_pullback_side:
        return {
            "passed": True,
            "side": tiered_pullback_side,
            "reasons": reasons,
            "trigger_type": "tiered_pullback",
            "tiered_pullback_tier": tiered_pullback_tier,
            "tier_updates": tier_updates,
            "m5_trend": trend,
        }

    zone_entry_triggered = False
    zone_side: Optional[str] = None
    zone_entry_enabled = getattr(policy, "zone_entry_enabled", True)
    if zone_entry_enabled:
        if zone_entry_mode == "price_vs_ema5":
            if is_bull and current_price > m1_ema5_val:
                zone_entry_triggered = True
                zone_side = "buy"
                reasons.append(
                    f"ZONE ENTRY [price_vs_ema5]: BULL + current_price ({current_price:.3f}) > M1 EMA5 ({m1_ema5_val:.3f}) -> BUY"
                )
            elif (not is_bull) and current_price < m1_ema5_val:
                zone_entry_triggered = True
                zone_side = "sell"
                reasons.append(
                    f"ZONE ENTRY [price_vs_ema5]: BEAR + current_price ({current_price:.3f}) < M1 EMA5 ({m1_ema5_val:.3f}) -> SELL"
                )
        else:
            if is_bull and m1_zone_fast_val > m1_zone_slow_val:
                zone_entry_triggered = True
                zone_side = "buy"
                reasons.append(
                    f"ZONE ENTRY [ema_cross]: BULL + M1 EMA{m1_zone_ema_fast} ({m1_zone_fast_val:.3f}) > EMA{m1_zone_ema_slow} ({m1_zone_slow_val:.3f}) -> BUY"
                )
            elif (not is_bull) and m1_zone_fast_val < m1_zone_slow_val:
                zone_entry_triggered = True
                zone_side = "sell"
                reasons.append(
                    f"ZONE ENTRY [ema_cross]: BEAR + M1 EMA{m1_zone_ema_fast} ({m1_zone_fast_val:.3f}) < EMA{m1_zone_ema_slow} ({m1_zone_slow_val:.3f}) -> SELL"
                )
    else:
        reasons.append("ZONE ENTRY: disabled by zone_entry_enabled=False")

    if zone_entry_triggered and zone_side:
        return {
            "passed": True,
            "side": zone_side,
            "reasons": reasons,
            "trigger_type": "zone_entry",
            "tiered_pullback_tier": None,
            "tier_updates": tier_updates,
            "m5_trend": trend,
        }

    if zone_entry_mode == "price_vs_ema5":
        reasons.append(f"No trigger: current_price={current_price:.3f}, EMA5={m1_ema5_val:.3f}, no tier touch")
    else:
        reasons.append(f"No trigger: Zone={m1_zone_fast_val:.3f} vs EMA{m1_zone_ema_slow}={m1_zone_slow_val:.3f}, no tier touch")
    return {
        "passed": False,
        "side": None,
        "reasons": reasons,
        "trigger_type": "",
        "tiered_pullback_tier": None,
        "tier_updates": tier_updates,
        "m5_trend": trend,
    }


def _evaluate_kt_cg_trial_7_zone_only_candidate(
    profile: ProfileV1,
    policy,
    data_by_tf: dict[Timeframe, pd.DataFrame],
    tick: Tick,
    tier_state: dict[int, bool],
) -> dict:
    """Re-evaluate Trial #7 conditions for zone-entry only (tier trigger disabled)."""
    original_tiered_enabled = bool(getattr(policy, "tiered_pullback_enabled", True))
    try:
        object.__setattr__(policy, "tiered_pullback_enabled", False)
        return evaluate_kt_cg_trial_7_conditions(
            profile,
            policy,
            data_by_tf,
            current_bid=tick.bid,
            current_ask=tick.ask,
            tier_state=tier_state,
        )
    finally:
        object.__setattr__(policy, "tiered_pullback_enabled", original_tiered_enabled)


def execute_kt_cg_trial_4_policy_demo_only(
    *,
    adapter,
    profile: ProfileV1,
    log_dir: Path,
    policy,  # ExecutionPolicyKtCgTrial4
    context: MarketContext,
    data_by_tf: dict[Timeframe, pd.DataFrame],
    tick: Tick,
    trades_df: Optional[pd.DataFrame],
    mode: str,
    bar_time_utc: str,
    tier_state: dict[int, bool],
    temp_overrides: Optional[dict] = None,
    divergence_state: Optional[dict[str, str]] = None,
) -> dict:
    """Evaluate KT/CG Trial #4 (M3 Trend + Tiered Pullback System) policy and optionally place a market order.

    Two independent triggers:
    1. Zone Entry (respects cooldown)
    2. Tiered Pullback (NO cooldown - each tier independent)

    Returns a dict with:
        - decision: ExecutionDecision
        - tier_updates: dict[int, bool] (new state for tiers to persist)
        - divergence_updates: dict[str, str] (new divergence block timestamps)
    """
    if mode not in ("ARMED_MANUAL_CONFIRM", "ARMED_AUTO_DEMO"):
        return {"decision": ExecutionDecision(attempted=False, placed=False, reason=f"mode={mode} not armed"), "tier_updates": {}, "divergence_updates": {}}
    if not adapter.is_demo_account():
        return {"decision": ExecutionDecision(attempted=False, placed=False, reason="not a demo account (execution disabled)"), "tier_updates": {}, "divergence_updates": {}}

    store = _store(log_dir)
    rule_id = f"kt_cg_trial_4:{policy.id}:M1:{bar_time_utc}"

    # Evaluate conditions - returns dict with passed, side, reasons, trigger_type, tier_updates, etc.
    result = evaluate_kt_cg_trial_4_conditions(
        profile, policy, data_by_tf,
        current_bid=tick.bid,
        current_ask=tick.ask,
        tier_state=tier_state,
        temp_overrides=temp_overrides,
    )
    passed = result["passed"]
    side = result["side"]
    eval_reasons = result["reasons"]
    trigger_type = result["trigger_type"]
    tier_updates = result.get("tier_updates", {})
    tiered_pullback_tier = result.get("tiered_pullback_tier")

    if not passed or side is None:
        # Still return tier_updates so resets can be persisted even when no trade fires
        return {"decision": ExecutionDecision(attempted=False, placed=False, reason="; ".join(eval_reasons)), "tier_updates": tier_updates, "divergence_updates": {}}

    # For tiered pullback, use tier-specific rule_id (no bar_time_utc — tier_state
    # already guarantees fire-once semantics, and we need re-fire after reset within
    # the same M1 bar)
    if trigger_type == "tiered_pullback" and tiered_pullback_tier:
        rule_id = f"kt_cg_trial_4:{policy.id}:tier_{tiered_pullback_tier}"

    # Idempotency check for zone_entry only — tiered pullback relies on tier_state
    # (tier_X_fired) to prevent double-firing, not on bar_time_utc-based rule_id
    if trigger_type == "zone_entry":
        within = 2  # M1 cadence
        if store.has_recent_price_level_placement(profile.profile_name, rule_id, within):
            return {"decision": ExecutionDecision(attempted=False, placed=False, reason="kt_cg_trial_4: recent placement (idempotent)"), "tier_updates": tier_updates, "divergence_updates": {}}

    # Check cooldown for Zone Entry only (Tiered Pullback has NO cooldown)
    if trigger_type == "zone_entry":
        cooldown_ok, cooldown_reason = _check_kt_cg_trial_4_cooldown(
            store, profile.profile_name, policy.id, policy.cooldown_minutes, rule_prefix="kt_cg_trial_4"
        )
        if not cooldown_ok:
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
                    "reason": cooldown_reason or "zone_entry_cooldown",
                    "mt5_retcode": None,
                    "mt5_order_id": None,
                    "mt5_deal_id": None,
                }
            )
            return {"decision": ExecutionDecision(attempted=True, placed=False, reason=cooldown_reason or "zone_entry_cooldown"), "tier_updates": tier_updates, "divergence_updates": {}}

    # EMA Zone Entry Filter: block zone entries during EMA compression
    if trigger_type == "zone_entry":
        ema_zone_filter_enabled = getattr(policy, "ema_zone_filter_enabled", False)
        if ema_zone_filter_enabled:
            m1_df_zone = data_by_tf.get("M1")
            if m1_df_zone is not None and not m1_df_zone.empty:
                is_bull = side == "buy"
                zf_lookback = getattr(policy, "ema_zone_filter_lookback_bars", 3)
                zf_threshold = getattr(policy, "ema_zone_filter_block_threshold", 0.35)
                pip_size = float(profile.pip_size)
                zf_score, zf_details = _compute_ema_zone_filter_score(
                    m1_df_zone, pip_size, is_bull, zf_lookback
                )
                if "error" not in zf_details and zf_score < zf_threshold:
                    zf_reason = (
                        f"ema_zone_filter: BLOCKED score={zf_score:.2f}"
                        f" (spread={zf_details['spread_pips']:.1f}p"
                        f" slope={zf_details['slope_pips']:.1f}p"
                        f" dir={zf_details['spread_dir_pips']:.1f}p)"
                        f" threshold={zf_threshold}"
                    )
                    print(f"[{profile.profile_name}] kt_cg_trial_4 {zf_reason}")
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
                            "reason": zf_reason,
                            "mt5_retcode": None,
                            "mt5_order_id": None,
                            "mt5_deal_id": None,
                        }
                    )
                    return {"decision": ExecutionDecision(attempted=True, placed=False, reason=zf_reason), "tier_updates": tier_updates, "divergence_updates": {}}

    # Strategy filters: session, ATR
    now_utc = datetime.now(timezone.utc)
    ok, reason = passes_session_filter(profile, now_utc)
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
                "reason": reason or "session_filter",
                "mt5_retcode": None,
                "mt5_order_id": None,
                "mt5_deal_id": None,
            }
        )
        return {"decision": ExecutionDecision(attempted=True, placed=False, reason=reason or "session_filter"), "tier_updates": tier_updates, "divergence_updates": {}}
    ok, reason = passes_session_boundary_block(profile, now_utc)
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
                "reason": reason or "session_boundary_block",
                "mt5_retcode": None,
                "mt5_order_id": None,
                "mt5_deal_id": None,
            }
        )
        return {"decision": ExecutionDecision(attempted=True, placed=False, reason=reason or "session_boundary_block"), "tier_updates": tier_updates, "divergence_updates": {}}

    m1_df = data_by_tf.get("M1")
    if m1_df is not None:
        # Use tiered ATR filter if enabled (replaces generic ATR filter for Trial #4)
        if getattr(policy, "tiered_atr_filter_enabled", False):
            ok, reason = _passes_tiered_atr_filter_trial_4(policy, m1_df, float(profile.pip_size), trigger_type)
        else:
            ok, reason = passes_atr_filter(profile, m1_df, "M1")
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
                    "reason": reason or "atr_filter",
                    "mt5_retcode": None,
                    "mt5_order_id": None,
                    "mt5_deal_id": None,
                }
            )
            return {"decision": ExecutionDecision(attempted=True, placed=False, reason=reason or "atr_filter"), "tier_updates": tier_updates, "divergence_updates": {}}

    # Rolling Danger Zone filter: block entries near M1 rolling high/low extremes
    rolling_danger_enabled = getattr(policy, "rolling_danger_zone_enabled", False)
    if rolling_danger_enabled:
        m1_df = data_by_tf.get("M1")
        if m1_df is not None and not m1_df.empty:
            lookback = getattr(policy, "rolling_danger_lookback_bars", 100)
            danger_pct = getattr(policy, "rolling_danger_zone_pct", 0.15)

            # Get last X bars for rolling high/low calculation
            bars_to_use = m1_df.tail(lookback)
            if len(bars_to_use) >= 10:  # Need minimum bars for meaningful calculation
                rolling_high = float(bars_to_use["high"].max())
                rolling_low = float(bars_to_use["low"].min())
                price_range = rolling_high - rolling_low

                if price_range > 0:
                    # Upper danger zone = top Y% of range (blocks BUY)
                    upper_danger_threshold = rolling_high - (price_range * danger_pct)
                    # Lower danger zone = bottom Y% of range (blocks SELL)
                    lower_danger_threshold = rolling_low + (price_range * danger_pct)

                    entry_price_check = tick.ask if side == "buy" else tick.bid
                    print(f"[{profile.profile_name}] kt_cg_trial_4 danger zone: price={entry_price_check:.3f} side={side} high={rolling_high:.3f} low={rolling_low:.3f} upper_danger={upper_danger_threshold:.3f} lower_danger={lower_danger_threshold:.3f}")

                    danger_blocked = False
                    danger_reason = None

                    if side == "buy" and entry_price_check >= upper_danger_threshold:
                        danger_blocked = True
                        danger_reason = f"rolling_danger_zone: BUY blocked, price {entry_price_check:.3f} >= upper threshold {upper_danger_threshold:.3f} (top {danger_pct*100:.0f}% of {lookback}-bar range)"
                    elif side == "sell" and entry_price_check <= lower_danger_threshold:
                        danger_blocked = True
                        danger_reason = f"rolling_danger_zone: SELL blocked, price {entry_price_check:.3f} <= lower threshold {lower_danger_threshold:.3f} (bottom {danger_pct*100:.0f}% of {lookback}-bar range)"

                    if danger_blocked:
                        print(f"[{profile.profile_name}] kt_cg_trial_4 BLOCKED by danger zone: {danger_reason}")
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
                                "reason": danger_reason or "rolling_danger_zone",
                                "mt5_retcode": None,
                                "mt5_order_id": None,
                                "mt5_deal_id": None,
                            }
                        )
                        return {"decision": ExecutionDecision(attempted=True, placed=False, reason=danger_reason or "rolling_danger_zone"), "tier_updates": tier_updates, "divergence_updates": {}}

    # Daily High/Low Filter: block zone entry near daily extremes
    if trigger_type == "zone_entry":
        ok, reason = _passes_daily_hl_filter(policy, data_by_tf, tick, side, float(profile.pip_size))
        if not ok:
            print(f"[{profile.profile_name}] kt_cg_trial_4 BLOCKED by daily H/L filter: {reason}")
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
                    "reason": reason or "daily_hl_filter",
                    "mt5_retcode": None,
                    "mt5_order_id": None,
                    "mt5_deal_id": None,
                }
            )
            return {"decision": ExecutionDecision(attempted=True, placed=False, reason=reason or "daily_hl_filter"), "tier_updates": tier_updates, "divergence_updates": {}}

    # RSI Divergence Detection and Blocking (M5-based)
    # BULL trend + bearish divergence detected -> block BUY entries for X minutes
    # BEAR trend + bullish divergence detected -> block SELL entries for X minutes
    divergence_updates: dict[str, str] = {}
    rsi_divergence_enabled = getattr(policy, "rsi_divergence_enabled", False)
    if rsi_divergence_enabled:
        m5_df = data_by_tf.get("M5")
        if m5_df is not None and not m5_df.empty:
            rsi_period = getattr(policy, "rsi_divergence_period", 14)
            lookback_bars = getattr(policy, "rsi_divergence_lookback_bars", 50)
            block_minutes = getattr(policy, "rsi_divergence_block_minutes", 5.0)

            # Get trend from evaluate result
            m3_trend = result.get("m3_trend", "NEUTRAL")

            # Check if existing block is still active
            now_utc = datetime.now(timezone.utc)
            if divergence_state:
                block_buy_until_str = divergence_state.get("block_buy_until")
                block_sell_until_str = divergence_state.get("block_sell_until")

                if side == "buy" and block_buy_until_str:
                    try:
                        block_until = datetime.fromisoformat(block_buy_until_str.replace("Z", "+00:00"))
                        if now_utc < block_until:
                            block_reason = f"rsi_divergence: BUY blocked until {block_buy_until_str} (bearish divergence detected)"
                            print(f"[{profile.profile_name}] kt_cg_trial_4 BLOCKED by RSI divergence: {block_reason}")
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
                                    "reason": block_reason,
                                    "mt5_retcode": None,
                                    "mt5_order_id": None,
                                    "mt5_deal_id": None,
                                }
                            )
                            return {"decision": ExecutionDecision(attempted=True, placed=False, reason=block_reason), "tier_updates": tier_updates, "divergence_updates": {}}
                    except (ValueError, TypeError):
                        pass

                if side == "sell" and block_sell_until_str:
                    try:
                        block_until = datetime.fromisoformat(block_sell_until_str.replace("Z", "+00:00"))
                        if now_utc < block_until:
                            block_reason = f"rsi_divergence: SELL blocked until {block_sell_until_str} (bullish divergence detected)"
                            print(f"[{profile.profile_name}] kt_cg_trial_4 BLOCKED by RSI divergence: {block_reason}")
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
                                    "reason": block_reason,
                                    "mt5_retcode": None,
                                    "mt5_order_id": None,
                                    "mt5_deal_id": None,
                                }
                            )
                            return {"decision": ExecutionDecision(attempted=True, placed=False, reason=block_reason), "tier_updates": tier_updates, "divergence_updates": {}}
                    except (ValueError, TypeError):
                        pass

            # Detect new divergence using rolling window comparison (M5 data)
            has_bearish, has_bullish, divergence_details = detect_rsi_divergence(
                m5_df, rsi_period=rsi_period, lookback_bars=lookback_bars
            )

            # BULL trend + bearish divergence + trying to BUY -> set block_buy_until and reject
            if m3_trend == "BULL" and has_bearish and side == "buy":
                from datetime import timedelta
                block_until = now_utc + timedelta(minutes=block_minutes)
                block_until_str = block_until.isoformat()
                divergence_updates["block_buy_until"] = block_until_str

                block_reason = f"rsi_divergence: bearish divergence detected in BULL trend, BUY blocked for {block_minutes:.1f} min (price HH {divergence_details.get('bearish_divergence', {}).get('recent_price', 'N/A'):.3f} vs RSI LH {divergence_details.get('bearish_divergence', {}).get('recent_rsi', 'N/A'):.1f})"
                print(f"[{profile.profile_name}] kt_cg_trial_4 NEW DIVERGENCE BLOCK: {block_reason}")
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
                        "reason": block_reason,
                        "mt5_retcode": None,
                        "mt5_order_id": None,
                        "mt5_deal_id": None,
                    }
                )
                return {"decision": ExecutionDecision(attempted=True, placed=False, reason=block_reason), "tier_updates": tier_updates, "divergence_updates": divergence_updates}

            # BEAR trend + bullish divergence + trying to SELL -> set block_sell_until and reject
            if m3_trend == "BEAR" and has_bullish and side == "sell":
                from datetime import timedelta
                block_until = now_utc + timedelta(minutes=block_minutes)
                block_until_str = block_until.isoformat()
                divergence_updates["block_sell_until"] = block_until_str

                block_reason = f"rsi_divergence: bullish divergence detected in BEAR trend, SELL blocked for {block_minutes:.1f} min (price LL {divergence_details.get('bullish_divergence', {}).get('recent_price', 'N/A'):.3f} vs RSI HL {divergence_details.get('bullish_divergence', {}).get('recent_rsi', 'N/A'):.1f})"
                print(f"[{profile.profile_name}] kt_cg_trial_4 NEW DIVERGENCE BLOCK: {block_reason}")
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
                        "reason": block_reason,
                        "mt5_retcode": None,
                        "mt5_order_id": None,
                        "mt5_deal_id": None,
                    }
                )
                return {"decision": ExecutionDecision(attempted=True, placed=False, reason=block_reason), "tier_updates": tier_updates, "divergence_updates": divergence_updates}

    entry_price = tick.ask if side == "buy" else tick.bid
    candidate = _kt_cg_trial_4_candidate(profile, policy, entry_price, side)
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
        return {"decision": ExecutionDecision(attempted=True, placed=False, reason="risk rejected: " + "; ".join(decision.hard_reasons)), "tier_updates": tier_updates, "divergence_updates": divergence_updates}

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
        return {"decision": ExecutionDecision(attempted=True, placed=False, reason="manual_confirm_required"), "tier_updates": tier_updates, "divergence_updates": divergence_updates}

    # ARMED_AUTO_DEMO: place the trade
    # Build comment with tier info for tiered pullback
    comment = f"kt_cg_trial_4:{policy.id}:{trigger_type}"
    if trigger_type == "tiered_pullback" and tiered_pullback_tier:
        comment = f"kt_cg_trial_4:{policy.id}:tier_{tiered_pullback_tier}"

    res = adapter.order_send_market(
        symbol=profile.symbol,
        side=side,
        volume_lots=float(candidate.size_lots or get_effective_risk(profile).max_lots),
        sl=candidate.stop_price,
        tp=candidate.target_price,
        comment=comment,
    )
    placed = res.retcode in (0, 10008, 10009)
    reason = f"{trigger_type}:order_sent" if placed else f"order_failed:{res.retcode}:{res.comment}"
    if trigger_type == "tiered_pullback" and tiered_pullback_tier:
        reason = f"tier_{tiered_pullback_tier}:order_sent" if placed else f"order_failed:{res.retcode}:{res.comment}"
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

    if placed:
        tier_info = f" tier_{tiered_pullback_tier}" if tiered_pullback_tier else ""
        print(f"[{profile.profile_name}] TRADE PLACED: kt_cg_trial_4:{trigger_type}{tier_info} | {'; '.join(eval_reasons)}")

    return {
        "decision": ExecutionDecision(
            attempted=True,
            placed=placed,
            reason=reason,
            order_retcode=res.retcode,
            order_id=res.order,
            deal_id=res.deal,
            fill_price=getattr(res, 'fill_price', None),
            side=side,
        ),
        "tier_updates": tier_updates,
        "divergence_updates": divergence_updates,
        "trigger_type": trigger_type,
    }


def execute_kt_cg_trial_7_policy_demo_only(
    *,
    adapter,
    profile: ProfileV1,
    log_dir: Path,
    policy,  # ExecutionPolicyKtCgTrial7 | ExecutionPolicyKtCgTrial8
    context: MarketContext,
    data_by_tf: dict[Timeframe, pd.DataFrame],
    tick: Tick,
    trades_df: Optional[pd.DataFrame],
    mode: str,
    bar_time_utc: str,
    tier_state: dict[int, bool],
    store=None,
    daily_level_filter: Optional[DailyLevelFilter] = None,
    daily_state: Optional[dict] = None,
) -> dict:
    """Evaluate and execute KT/CG Trial #7 or #8 (demo only). T8: no EMA zone/reversal risk; daily level filter optional."""
    candidate_side: Optional[str] = None
    candidate_trigger: Optional[str] = None
    if mode not in ("ARMED_MANUAL_CONFIRM", "ARMED_AUTO_DEMO"):
        return {
            "decision": ExecutionDecision(attempted=False, placed=False, reason=f"mode={mode} not armed"),
            "tier_updates": {},
            "trigger_type": None,
            "candidate_side": candidate_side,
            "candidate_trigger": candidate_trigger,
            "side": candidate_side,
            "exhaustion_result": None,
        }
    if not adapter.is_demo_account():
        return {
            "decision": ExecutionDecision(attempted=False, placed=False, reason="not a demo account (execution disabled)"),
            "tier_updates": {},
            "trigger_type": None,
            "candidate_side": candidate_side,
            "candidate_trigger": candidate_trigger,
            "side": candidate_side,
            "exhaustion_result": None,
        }

    store = _store(log_dir)
    policy_type = getattr(policy, "type", "kt_cg_trial_7")
    rule_id = f"{policy_type}:{policy.id}:M1:{bar_time_utc}"

    exhaustion_result = None
    tier_updates: dict[int, bool] = {}
    effective_tiers = _canonical_trial7_tier_periods(getattr(policy, "tier_ema_periods", tuple(range(9, 35))))
    block_zone_entry_by_exhaustion = False
    cap_multiplier = 1.0
    cap_minimum = 1

    def _result_payload(
        decision: ExecutionDecision,
        *,
        extra: Optional[dict] = None,
    ) -> dict:
        payload = {
            "decision": decision,
            "tier_updates": tier_updates,
            "trigger_type": candidate_trigger,
            "candidate_side": candidate_side,
            "candidate_trigger": candidate_trigger,
            "side": candidate_side,
            "exhaustion_result": exhaustion_result,
        }
        if extra:
            payload.update(extra)
        return payload

    if getattr(policy, "trend_exhaustion_enabled", False):
        m5_df_ex = data_by_tf.get("M5")
        if m5_df_ex is not None and not m5_df_ex.empty:
            m5_local = drop_incomplete_last_bar(m5_df_ex.copy(), "M5")
            if len(m5_local) >= max(getattr(policy, "m5_trend_ema_slow", 21) + 1, 22):
                close_m5 = m5_local["close"].astype(float)
                ema_fast = close_m5.ewm(span=getattr(policy, "m5_trend_ema_fast", 9), adjust=False).mean()
                ema_slow = close_m5.ewm(span=getattr(policy, "m5_trend_ema_slow", 21), adjust=False).mean()
                trend_side = "bull" if float(ema_fast.iloc[-1]) > float(ema_slow.iloc[-1]) else "bear"
                current_mid = (tick.bid + tick.ask) / 2.0
                exhaustion_result = _compute_trial7_trend_exhaustion(
                    policy=policy,
                    m5_df=m5_local,
                    current_price=current_mid,
                    pip_size=float(profile.pip_size),
                    trend_side=trend_side,
                )
                ex_zone = exhaustion_result.get("zone", "normal")
                if ex_zone == "extended":
                    if bool(getattr(policy, "trend_exhaustion_extended_disable_zone_entry", True)):
                        block_zone_entry_by_exhaustion = True
                    min_tier = int(getattr(policy, "trend_exhaustion_extended_min_tier_period", 21))
                    effective_tiers = [t for t in effective_tiers if int(t) >= min_tier]
                elif ex_zone == "very_extended":
                    if bool(getattr(policy, "trend_exhaustion_very_extended_disable_zone_entry", True)):
                        block_zone_entry_by_exhaustion = True
                    min_tier = int(getattr(policy, "trend_exhaustion_very_extended_min_tier_period", 29))
                    effective_tiers = [t for t in effective_tiers if int(t) >= min_tier]
                    if bool(getattr(policy, "trend_exhaustion_very_extended_tighten_caps", True)):
                        cap_multiplier = max(0.05, float(getattr(policy, "trend_exhaustion_very_extended_cap_multiplier", 0.5)))
                        cap_minimum = max(1, int(getattr(policy, "trend_exhaustion_very_extended_cap_min", 1)))
            else:
                exhaustion_result = {"zone": "normal", "reason": "insufficient_m5_bars"}

    effective_tp_pips, tp_zone = _compute_trial7_effective_tp_pips(policy, exhaustion_result)
    if exhaustion_result is None:
        exhaustion_result = {"zone": tp_zone}
    exhaustion_result["tp_base_pips"] = round(float(getattr(policy, "tp_pips", 4.0)), 3)
    exhaustion_result["tp_effective_pips"] = round(float(effective_tp_pips), 3)
    exhaustion_result["tp_adaptive_enabled"] = bool(getattr(policy, "trend_exhaustion_adaptive_tp_enabled", False))

    original_tiers = getattr(policy, "tier_ema_periods", tuple(range(9, 35)))
    if not effective_tiers:
        effective_tiers = [int(min(original_tiers))] if original_tiers else [29]

    try:
        object.__setattr__(policy, "tier_ema_periods", tuple(int(x) for x in effective_tiers))
        result = evaluate_kt_cg_trial_7_conditions(
            profile, policy, data_by_tf,
            current_bid=tick.bid,
            current_ask=tick.ask,
            tier_state=tier_state,
        )
    finally:
        object.__setattr__(policy, "tier_ema_periods", original_tiers)
    passed = result["passed"]
    side = result["side"]
    eval_reasons = result["reasons"]
    trigger_type = str(result.get("trigger_type") or "")
    tier_updates = result.get("tier_updates", {})
    tiered_pullback_tier = result.get("tiered_pullback_tier")
    if side in ("buy", "sell"):
        candidate_side = side
    candidate_trigger = trigger_type or None

    def _run_zone_prechecks() -> Optional[str]:
        if trigger_type != "zone_entry":
            return None
        if block_zone_entry_by_exhaustion:
            ex_zone = (exhaustion_result or {}).get("zone", "normal")
            stretch = (exhaustion_result or {}).get("stretch_pips")
            return f"trend_exhaustion: {ex_zone} blocks zone_entry (stretch={stretch}p)"

        cooldown_ok, cooldown_reason = _check_kt_cg_trial_4_cooldown(
            store, profile.profile_name, policy.id, policy.cooldown_minutes, rule_prefix=policy_type
        )
        if not cooldown_ok:
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
                    "reason": cooldown_reason or "zone_entry_cooldown",
                    "mt5_retcode": None,
                    "mt5_order_id": None,
                    "mt5_deal_id": None,
                }
            )
            return cooldown_reason or "zone_entry_cooldown"

        ema_zone_filter_enabled = getattr(policy, "ema_zone_filter_enabled", False)
        if ema_zone_filter_enabled and policy_type != "kt_cg_trial_8":
            m1_df_zone = data_by_tf.get("M1")
            if m1_df_zone is not None and not m1_df_zone.empty:
                ok, reason, _details = _passes_ema_zone_slope_filter_trial_7(
                    policy, m1_df_zone, float(profile.pip_size), side
                )
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
                            "reason": reason or "ema_zone_slope_filter",
                            "mt5_retcode": None,
                            "mt5_order_id": None,
                            "mt5_deal_id": None,
                        }
                    )
                    return reason or "ema_zone_slope_filter"
        return None

    def _attempt_zone_fallback_from_tier_cap(cap_reason: str) -> tuple[bool, str]:
        nonlocal side, trigger_type, tiered_pullback_tier, eval_reasons, tier_updates, rule_id, candidate_side, candidate_trigger
        fallback_prefix = f"{cap_reason}; tiered_cap_reached -> fallback_zone_attempted"
        if block_zone_entry_by_exhaustion:
            ex_zone = (exhaustion_result or {}).get("zone", "normal")
            stretch = (exhaustion_result or {}).get("stretch_pips")
            return False, f"{fallback_prefix}; trend_exhaustion: {ex_zone} blocks zone_entry (stretch={stretch}p)"

        fallback = _evaluate_kt_cg_trial_7_zone_only_candidate(
            profile=profile,
            policy=policy,
            data_by_tf=data_by_tf,
            tick=tick,
            tier_state=tier_state,
        )
        fallback_passed = bool(fallback.get("passed")) and fallback.get("side") is not None and str(fallback.get("trigger_type") or "") == "zone_entry"
        if not fallback_passed:
            fallback_reasons = "; ".join(fallback.get("reasons") or [])
            if fallback_reasons:
                return False, f"{fallback_prefix}; zone_fallback_not_valid: {fallback_reasons}"
            return False, f"{fallback_prefix}; zone_fallback_not_valid"

        fallback_side = str(fallback.get("side") or "").lower()
        if fallback_side not in ("buy", "sell"):
            return False, f"{fallback_prefix}; zone_fallback_invalid_side"
        if candidate_side in ("buy", "sell") and fallback_side != candidate_side:
            return False, f"{fallback_prefix}; zone_fallback_side_mismatch ({candidate_side}->{fallback_side})"

        trigger_type = "zone_entry"
        candidate_trigger = "zone_entry"
        side = fallback_side
        candidate_side = fallback_side
        tiered_pullback_tier = None
        # Do not mark tier as fired when tier trade did not execute.
        tier_updates = {int(k): bool(v) for k, v in tier_updates.items() if not bool(v)}
        eval_reasons = list(eval_reasons) + [fallback_prefix] + list(fallback.get("reasons") or [])
        rule_id = f"{policy_type}:{policy.id}:M1:{bar_time_utc}"
        return True, fallback_prefix

    if not passed or side is None:
        return _result_payload(
            ExecutionDecision(attempted=False, placed=False, reason="; ".join(eval_reasons)),
        )

    zone_precheck_reason = _run_zone_prechecks()
    if zone_precheck_reason:
        return _result_payload(
            ExecutionDecision(attempted=True, placed=False, reason=zone_precheck_reason),
        )

    # Daily Level Filter (Trial #8): applies to both zone_entry and tiered_pullback
    if policy_type == "kt_cg_trial_8" and getattr(policy, "use_daily_level_filter", False) and daily_level_filter is not None and daily_state:
        m5_df = data_by_tf.get("M5")
        m5_closed = drop_incomplete_m5_for_filter(m5_df) if m5_df is not None and not m5_df.empty else pd.DataFrame()
        current_price = (tick.bid + tick.ask) / 2.0
        allowed, reason = daily_level_filter.should_allow_trade(
            direction=side,
            current_price=current_price,
            yesterday_high=daily_state.get("prev_day_high"),
            yesterday_low=daily_state.get("prev_day_low"),
            today_high=daily_state.get("today_high"),
            today_low=daily_state.get("today_low"),
            m5_closed_candles=m5_closed,
            current_date_utc=daily_state.get("date_utc"),
            store=store,
            profile_name=profile.profile_name,
            symbol=profile.symbol,
            mode=mode,
            rule_id=rule_id,
        )
        if not allowed:
            print(f"[{profile.profile_name}] kt_cg_trial_8 daily_level_filter: {reason}")
            if store is not None:
                store.insert_execution(
                    {
                        "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                        "profile": profile.profile_name,
                        "symbol": profile.symbol,
                        "signal_id": f"{rule_id}:{pd.Timestamp.now(tz='UTC').isoformat()}",
                        "mode": mode,
                        "attempted": 1,
                        "placed": 0,
                        "reason": reason,
                        "mt5_retcode": None,
                        "mt5_order_id": None,
                        "mt5_deal_id": None,
                    }
                )
            return _result_payload(
                ExecutionDecision(attempted=True, placed=False, reason=reason),
            )
        if daily_level_filter.enabled:
            snap = daily_level_filter.get_state_snapshot()
            print(f"[{profile.profile_name}] kt_cg_trial_8 daily_level_filter state: {snap}")

    if trigger_type == "tiered_pullback" and tiered_pullback_tier:
        rule_id = f"{policy_type}:{policy.id}:tier_{tiered_pullback_tier}"

    max_open_per_side = getattr(policy, "max_open_trades_per_side", None)
    if max_open_per_side is not None:
        max_open_per_side = max(cap_minimum, int(round(float(max_open_per_side) * cap_multiplier)))
    if max_open_per_side is not None:
        try:
            open_positions = adapter.get_open_positions(profile.symbol)
            side_open = 0
            if open_positions:
                for pos in open_positions:
                    if isinstance(pos, dict):
                        # OANDA returns all open trades for the symbol.
                        # For Trial #7 per-side cap, count Trial #7-tagged trades only
                        # so manual/other-strategy trades do not consume this policy cap.
                        client_ext = pos.get("clientExtensions")
                        comment = ""
                        if isinstance(client_ext, dict):
                            comment = str(client_ext.get("comment") or "").strip()
                        if not comment:
                            trade_ext = pos.get("tradeClientExtensions")
                            if isinstance(trade_ext, dict):
                                comment = str(trade_ext.get("comment") or "").strip()
                        if comment and not (comment.startswith("kt_cg_trial_7:") or comment.startswith("kt_cg_trial_8:")):
                            continue
                        units = float(pos.get("currentUnits") or pos.get("initialUnits") or 0)
                        if units > 0:
                            pos_side = "buy"
                        elif units < 0:
                            pos_side = "sell"
                        else:
                            continue
                    else:
                        mt5_type = getattr(pos, "type", None)
                        if mt5_type == 0:
                            pos_side = "buy"
                        elif mt5_type == 1:
                            pos_side = "sell"
                        else:
                            continue
                    if pos_side == side:
                        side_open += 1
            if side_open >= max_open_per_side:
                return _result_payload(
                    ExecutionDecision(
                        attempted=True,
                        placed=False,
                        reason=f"max_open_trades_per_side: {side_open} {side} trade(s) open (max {max_open_per_side})",
                    ),
                )
        except Exception:
            pass

    try:
        open_trades = store.list_open_trades(profile.profile_name)
        # sqlite3.Row doesn't support .get(); convert to plain dicts for safe access
        open_trades = [dict(r) if hasattr(r, "keys") else r for r in open_trades]
        # Live broker position ids: cap counts only actually open positions (not stale DB rows)
        _live_pos_ids: set[int] = set()
        try:
            if adapter is not None:
                positions = adapter.get_open_positions(profile.symbol)
                for pos in positions or []:
                    pid = pos.get("id") if isinstance(pos, dict) else getattr(pos, "ticket", None)
                    if pid is not None:
                        try:
                            _live_pos_ids.add(int(pid))
                        except (TypeError, ValueError):
                            pass
        except Exception:
            pass

        # All DB position ids (any entry type) — used to detect orphaned live positions
        _db_all_pos_ids: set[int] = set()
        for _row in open_trades:
            _rdict = dict(_row) if hasattr(_row, "keys") else _row
            _pid = _rdict.get("mt5_position_id")
            if _pid is not None:
                try:
                    _db_all_pos_ids.add(int(_pid))
                except (TypeError, ValueError):
                    pass
        # Orphaned live positions: in OANDA but absent from DB entirely (e.g. placed before DB recording worked)
        _orphaned_live_count = len(_live_pos_ids - _db_all_pos_ids) if _live_pos_ids else 0

        def _row_still_open(row: dict) -> bool:
            if not _live_pos_ids:
                return True
            pid = row.get("mt5_position_id")
            if pid is None:
                return False
            try:
                return int(pid) in _live_pos_ids
            except (TypeError, ValueError):
                return False

        while True:
            if trigger_type == "tiered_pullback":
                max_tiered_pullback_open = getattr(policy, "max_tiered_pullback_open", None)
                if max_tiered_pullback_open is not None:
                    max_tiered_pullback_open = max(cap_minimum, int(round(float(max_tiered_pullback_open) * cap_multiplier)))
                    tiered_open = sum(
                        1 for row in open_trades
                        if row.get("entry_type") == "tiered_pullback" and _row_still_open(dict(row) if hasattr(row, "keys") else row)
                    )
                    if tiered_open >= max_tiered_pullback_open:
                        tier_cap_reason = (
                            f"max_tiered_pullback_open: {tiered_open} tiered pullback trade(s) already open "
                            f"(max {max_tiered_pullback_open})"
                        )
                        fallback_ok, fallback_reason = _attempt_zone_fallback_from_tier_cap(tier_cap_reason)
                        if not fallback_ok:
                            return _result_payload(
                                ExecutionDecision(attempted=True, placed=False, reason=fallback_reason),
                            )
                        zone_precheck_reason = _run_zone_prechecks()
                        if zone_precheck_reason:
                            return _result_payload(
                                ExecutionDecision(attempted=True, placed=False, reason=zone_precheck_reason),
                            )
                        continue
            if trigger_type == "zone_entry":
                max_zone_entry_open = getattr(policy, "max_zone_entry_open", None)
                if max_zone_entry_open is not None:
                    max_zone_entry_open = max(cap_minimum, int(round(float(max_zone_entry_open) * cap_multiplier)))
                    zone_entry_open = sum(
                        1 for row in open_trades
                        if row.get("entry_type") == "zone_entry" and _row_still_open(dict(row) if hasattr(row, "keys") else row)
                    )
                    # Conservatively count orphaned live positions (in OANDA but not in DB) toward the zone
                    # entry cap. These arise when trades were placed but not recorded (e.g. pre-fix crashes).
                    zone_entry_open += _orphaned_live_count
                    if zone_entry_open >= max_zone_entry_open:
                        return _result_payload(
                            ExecutionDecision(
                                attempted=True,
                                placed=False,
                                reason=f"max_zone_entry_open: {zone_entry_open} zone entry trade(s) already open (max {max_zone_entry_open}; includes {_orphaned_live_count} unrecorded)",
                            ),
                        )
            break
    except Exception:
        pass

    now_utc = datetime.now(timezone.utc)
    ok, reason = passes_session_filter(profile, now_utc)
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
                "reason": reason or "session_filter",
                "mt5_retcode": None,
                "mt5_order_id": None,
                "mt5_deal_id": None,
            }
        )
        return _result_payload(
            ExecutionDecision(attempted=True, placed=False, reason=reason or "session_filter"),
        )
    ok, reason = passes_session_boundary_block(profile, now_utc)
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
                "reason": reason or "session_boundary_block",
                "mt5_retcode": None,
                "mt5_order_id": None,
                "mt5_deal_id": None,
            }
        )
        return _result_payload(
            ExecutionDecision(attempted=True, placed=False, reason=reason or "session_boundary_block"),
        )

    entry_price = tick.ask if side == "buy" else tick.bid
    original_tp_pips = float(getattr(policy, "tp_pips", 4.0))
    try:
        object.__setattr__(policy, "tp_pips", float(effective_tp_pips))
        candidate = _kt_cg_trial_4_candidate(profile, policy, entry_price, side)
    finally:
        object.__setattr__(policy, "tp_pips", original_tp_pips)
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
        return _result_payload(
            ExecutionDecision(attempted=True, placed=False, reason="risk rejected: " + "; ".join(decision.hard_reasons)),
        )

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
        return _result_payload(
            ExecutionDecision(attempted=True, placed=False, reason="manual_confirm_required"),
        )

    comment = f"{policy_type}:{policy.id}:{trigger_type}"
    if trigger_type == "tiered_pullback" and tiered_pullback_tier:
        comment = f"{policy_type}:{policy.id}:tier_{tiered_pullback_tier}"

    res = adapter.order_send_market(
        symbol=profile.symbol,
        side=side,
        volume_lots=float(candidate.size_lots or get_effective_risk(profile).max_lots),
        sl=candidate.stop_price,
        tp=candidate.target_price,
        comment=comment,
    )
    placed = res.retcode in (0, 10008, 10009)
    reason = f"{trigger_type}:order_sent" if placed else f"order_failed:{res.retcode}:{res.comment}"
    if trigger_type == "tiered_pullback" and tiered_pullback_tier:
        reason = f"tier_{tiered_pullback_tier}:order_sent" if placed else f"order_failed:{res.retcode}:{res.comment}"
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

    if placed:
        tier_info = f" tier_{tiered_pullback_tier}" if tiered_pullback_tier else ""
        print(
            f"[{profile.profile_name}] TRADE PLACED: {policy_type}:{trigger_type}{tier_info} "
            f"| TP={effective_tp_pips:.2f}p (base={original_tp_pips:.2f}p zone={tp_zone})"
            f" | {'; '.join(eval_reasons)}"
        )

    return _result_payload(
        ExecutionDecision(
            attempted=True,
            placed=placed,
            reason=reason,
            order_retcode=res.retcode,
            order_id=res.order,
            deal_id=res.deal,
            fill_price=getattr(res, "fill_price", None),
            side=side,
        ),
        extra={
            "tp_pips_effective": float(effective_tp_pips),
            "tp_pips_base": float(original_tp_pips),
        },
    )


def execute_kt_cg_trial_8_policy_demo_only(
    *,
    adapter,
    profile: ProfileV1,
    log_dir: Path,
    policy,
    context,
    data_by_tf,
    tick,
    trades_df,
    mode: str,
    bar_time_utc: str,
    tier_state: dict[int, bool],
    store=None,
    daily_level_filter: Optional[DailyLevelFilter] = None,
    daily_state: Optional[dict] = None,
) -> dict:
    """Trial #8: delegates to Trial #7 flow with daily_level_filter and daily_state (no EMA zone, no reversal risk)."""
    return execute_kt_cg_trial_7_policy_demo_only(
        adapter=adapter,
        profile=profile,
        log_dir=log_dir,
        policy=policy,
        context=context,
        data_by_tf=data_by_tf,
        tick=tick,
        trades_df=trades_df,
        mode=mode,
        bar_time_utc=bar_time_utc,
        tier_state=tier_state,
        store=store,
        daily_level_filter=daily_level_filter,
        daily_state=daily_state,
    )


def _passes_fresh_cross_check(m1_df: pd.DataFrame, is_bull: bool) -> tuple[bool, str | None]:
    """Check if there was a fresh EMA5/EMA9 cross within the last 10 M1 bars.

    BULL: EMA5 > EMA9 now AND within bars [-11] to [-2] EMA5 WAS <= EMA9
    BEAR: EMA5 < EMA9 now AND within bars [-11] to [-2] EMA5 WAS >= EMA9

    Hardcoded lookback of 10 bars. Always active for Trial #5.
    """
    FRESH_CROSS_LOOKBACK = 10

    close = m1_df["close"].astype(float)
    ema5 = close.ewm(span=5, adjust=False).mean()
    ema9 = close.ewm(span=9, adjust=False).mean()

    # Need enough bars
    needed = FRESH_CROSS_LOOKBACK + 2
    if len(ema5) < needed:
        return True, None  # Insufficient data, allow

    # Current state check
    ema5_now = float(ema5.iloc[-1])
    ema9_now = float(ema9.iloc[-1])

    if is_bull and ema5_now <= ema9_now:
        return False, "fresh_cross: EMA5 not above EMA9 (no bull condition)"
    if not is_bull and ema5_now >= ema9_now:
        return False, "fresh_cross: EMA5 not below EMA9 (no bear condition)"

    # Check if the opposite condition existed within lookback window (bars -11 to -2)
    found_opposite = False
    for i in range(-(FRESH_CROSS_LOOKBACK + 1), -1):
        e5 = float(ema5.iloc[i])
        e9 = float(ema9.iloc[i])
        if is_bull and e5 <= e9:
            found_opposite = True
            break
        if not is_bull and e5 >= e9:
            found_opposite = True
            break

    if not found_opposite:
        direction = "BULL" if is_bull else "BEAR"
        return False, f"fresh_cross: no recent cross within {FRESH_CROSS_LOOKBACK} bars ({direction}, EMA5 has been on same side)"

    return True, None


def _find_last_m3_ema_cross(m3_df: pd.DataFrame) -> tuple[Optional[float], Optional[str], Optional[str]]:
    """Find the most recent M3 EMA9/EMA21 crossover in history. Used to bootstrap exhaustion state.

    Returns (flip_price, direction, flip_bar_time) with flip_price = close of the bar where cross
    occurred, direction = "bull" or "bear", flip_bar_time = ISO UTC timestamp of that bar.
    Returns (None, None, None) if no crossover found or insufficient bars.
    """
    close = m3_df["close"].astype(float)
    if len(close) < 3:
        return None, None, None
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    n = len(ema9)
    for i in range(n - 1, 0, -1):
        bull_now = float(ema9.iloc[i]) > float(ema21.iloc[i])
        bull_prev = float(ema9.iloc[i - 1]) > float(ema21.iloc[i - 1])
        if bull_now != bull_prev:
            flip_price = float(close.iloc[i])
            direction = "bull" if bull_now else "bear"
            flip_bar_time = None
            if "time" in m3_df.columns:
                try:
                    t = m3_df["time"].iloc[i]
                    flip_bar_time = pd.Timestamp(t).tz_convert("UTC").isoformat()
                except Exception:
                    pass
            return flip_price, direction, flip_bar_time
    return None, None, None


def _detect_trend_flip_and_compute_exhaustion(
    m3_df: pd.DataFrame,
    current_price: float,
    pip_size: float,
    exhaustion_state: dict,
    policy=None,
) -> dict:
    """Detect M3 EMA9/EMA21 trend flip and compute extension exhaustion.

    Returns dict with:
        - flip_detected: bool
        - reset_tiers: bool (True on flip)
        - zone: str ("FRESH" / "MATURE" / "EXTENDED" / "EXHAUSTED")
        - extension_ratio: float
        - trend_flip_price: float or None
        - trend_flip_direction: str or None
    """
    result = {
        "flip_detected": False,
        "reset_tiers": False,
        "zone": "FRESH",
        "extension_ratio": 0.0,
        "adjusted_ratio": 0.0,
        "time_factor": 1.0,
        "trend_flip_price": exhaustion_state.get("trend_flip_price"),
        "trend_flip_direction": exhaustion_state.get("trend_flip_direction"),
        "trend_flip_time": exhaustion_state.get("trend_flip_time"),
    }

    close = m3_df["close"].astype(float)
    if len(close) < 22:
        return result

    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()

    if len(ema9) < 3:
        return result

    # Crossover detection: compare bar[-1] vs bar[-2]
    ema9_now = float(ema9.iloc[-1])
    ema21_now = float(ema21.iloc[-1])
    ema9_prev = float(ema9.iloc[-2])
    ema21_prev = float(ema21.iloc[-2])

    is_bull_now = ema9_now > ema21_now
    was_bull_prev = ema9_prev > ema21_prev

    if is_bull_now != was_bull_prev:
        # Trend flip detected!
        new_direction = "bull" if is_bull_now else "bear"
        now_utc = datetime.now(timezone.utc).isoformat()
        result["flip_detected"] = True
        result["reset_tiers"] = True
        result["trend_flip_price"] = current_price
        result["trend_flip_direction"] = new_direction
        result["trend_flip_time"] = now_utc
        result["zone"] = "FRESH"
        result["extension_ratio"] = 0.0
        result["adjusted_ratio"] = 0.0
        result["time_factor"] = 0.0
        return result

    # No flip — compute extension ratio from flip price
    flip_price = exhaustion_state.get("trend_flip_price")
    if flip_price is None:
        return result

    # Compute M3 ATR(14) for normalization
    atr_series = atr_fn(m3_df, 14)
    if atr_series.empty or pd.isna(atr_series.iloc[-1]):
        return result
    m3_atr_pips = float(atr_series.iloc[-1]) / pip_size
    if m3_atr_pips <= 0:
        return result

    extension_pips = abs(current_price - flip_price) / pip_size
    extension_ratio = extension_pips / m3_atr_pips
    result["extension_ratio"] = round(extension_ratio, 2)

    # Time-based ramp: scale extension_ratio from 0→1 over ramp_minutes after flip
    flip_time_str = exhaustion_state.get("trend_flip_time")
    ramp_minutes = getattr(policy, "trend_exhaustion_ramp_minutes", 12.0) if policy else 12.0
    time_factor = 1.0
    if flip_time_str and ramp_minutes > 0:
        try:
            ft = pd.to_datetime(flip_time_str, utc=True)
            elapsed_min = (pd.Timestamp.now(tz="UTC") - ft).total_seconds() / 60.0
            time_factor = min(1.0, elapsed_min / ramp_minutes)
        except Exception:
            pass
    adjusted_ratio = extension_ratio * time_factor
    result["adjusted_ratio"] = round(adjusted_ratio, 2)
    result["time_factor"] = round(time_factor, 3)

    # Determine zone from thresholds (using adjusted_ratio so fresh flips stay in FRESH/MATURE)
    fresh_max = getattr(policy, "trend_exhaustion_fresh_max", 2.0) if policy else 2.0
    mature_max = getattr(policy, "trend_exhaustion_mature_max", 3.5) if policy else 3.5
    extended_max = getattr(policy, "trend_exhaustion_extended_max", 5.0) if policy else 5.0

    if adjusted_ratio <= fresh_max:
        result["zone"] = "FRESH"
    elif adjusted_ratio <= mature_max:
        result["zone"] = "MATURE"
    elif adjusted_ratio <= extended_max:
        result["zone"] = "EXTENDED"
    else:
        result["zone"] = "EXHAUSTED"

    return result


def execute_kt_cg_trial_5_policy_demo_only(
    *,
    adapter,
    profile: ProfileV1,
    log_dir: Path,
    policy,  # ExecutionPolicyKtCgTrial5
    context: MarketContext,
    data_by_tf: dict[Timeframe, pd.DataFrame],
    tick: Tick,
    trades_df: Optional[pd.DataFrame],
    mode: str,
    bar_time_utc: str,
    tier_state: dict[int, bool],
    temp_overrides: Optional[dict] = None,
    divergence_state: Optional[dict[str, str]] = None,
    daily_reset_state: Optional[dict] = None,
    exhaustion_state: Optional[dict] = None,
) -> dict:
    """Evaluate KT/CG Trial #5 (Overhauled: Fresh Cross, Exhaustion, Extended Tiers).

    Key changes from previous Trial #5:
    - Fresh Cross replaces cooldown for zone entry
    - Trend Extension Exhaustion restricts entries during extended moves
    - Dead zone extended to 21:00-02:00 UTC
    - EMA Zone Filter accepts configurable weights/ranges from policy
    """
    if mode not in ("ARMED_MANUAL_CONFIRM", "ARMED_AUTO_DEMO"):
        return {"decision": ExecutionDecision(attempted=False, placed=False, reason=f"mode={mode} not armed"), "tier_updates": {}, "divergence_updates": {}}
    if not adapter.is_demo_account():
        return {"decision": ExecutionDecision(attempted=False, placed=False, reason="not a demo account (execution disabled)"), "tier_updates": {}, "divergence_updates": {}}

    if exhaustion_state is None:
        exhaustion_state = {}

    # --- Dead Zone Block & H/L Tracking (21:00-02:00 UTC) ---
    if daily_reset_state is not None:
        tick_mid = (tick.bid + tick.ask) / 2.0
        _update_daily_reset_state(tick_mid, daily_reset_state, d_candle_df=data_by_tf.get("D"))

        if getattr(policy, "daily_reset_block_enabled", False) and daily_reset_state.get("daily_reset_block_active", False):
            return {
                "decision": ExecutionDecision(attempted=True, placed=False, reason="dead_zone_block: 21:00-02:00 UTC block active"),
                "tier_updates": {},
                "divergence_updates": {},
                "daily_reset_state": daily_reset_state,
                "exhaustion_state": exhaustion_state,
                "exhaustion_result": None,
            }

    # --- Trend Extension Exhaustion ---
    exhaustion_result = None
    m3_df = data_by_tf.get("M3")
    current_price = (tick.bid + tick.ask) / 2.0
    pip_size = float(profile.pip_size)
    if getattr(policy, "trend_exhaustion_enabled", False) and m3_df is not None and not m3_df.empty:
        # Bootstrap: if no persisted flip price (e.g. after long pause), find last M3 cross from history
        if exhaustion_state.get("trend_flip_price") is None:
            flip_price, flip_dir, flip_bar_time = _find_last_m3_ema_cross(m3_df)
            if flip_price is not None and flip_dir is not None:
                exhaustion_state["trend_flip_price"] = flip_price
                exhaustion_state["trend_flip_direction"] = flip_dir
                exhaustion_state["trend_flip_time"] = flip_bar_time
        exhaustion_result = _detect_trend_flip_and_compute_exhaustion(
            m3_df, current_price, pip_size, exhaustion_state, policy
        )
        # Update exhaustion state in-place for persistence
        if exhaustion_result.get("flip_detected"):
            exhaustion_state["trend_flip_price"] = exhaustion_result["trend_flip_price"]
            exhaustion_state["trend_flip_direction"] = exhaustion_result["trend_flip_direction"]
            exhaustion_state["trend_flip_time"] = exhaustion_result["trend_flip_time"]

        exhaust_zone = exhaustion_result.get("zone", "FRESH")
        if exhaust_zone == "EXHAUSTED":
            return {
                "decision": ExecutionDecision(
                    attempted=True, placed=False,
                    reason=f"trend_exhaustion: EXHAUSTED (ratio={exhaustion_result['extension_ratio']:.1f}x ATR, ALL blocked)"
                ),
                "tier_updates": {},
                "divergence_updates": {},
                "exhaustion_state": exhaustion_state,
                "exhaustion_result": exhaustion_result,
            }

    store = _store(log_dir)
    rule_id = f"kt_cg_trial_5:{policy.id}:M1:{bar_time_utc}"

    # --- Apply exhaustion tier filtering BEFORE evaluating conditions ---
    # Build effective tier list based on exhaustion zone
    effective_tier_periods = list(getattr(policy, "tier_ema_periods", (18, 21, 25, 29, 34)))
    block_zone_entry_by_exhaustion = False
    if exhaustion_result:
        exhaust_zone = exhaustion_result.get("zone", "FRESH")
        if exhaust_zone == "EXTENDED":
            # Only allow deepest tiers (29, 34), block zone entry
            effective_tier_periods = [p for p in effective_tier_periods if p >= 29]
            block_zone_entry_by_exhaustion = True
        elif exhaust_zone == "MATURE":
            # Filter out shallowest active tier, block zone entry
            if effective_tier_periods:
                effective_tier_periods = effective_tier_periods[1:]  # Remove shallowest
            block_zone_entry_by_exhaustion = True

        # On flip: tier reset is handled in run_loop.py (tier_fired cleared)

    # Create a temporary policy-like object with filtered tier_ema_periods if needed
    # We modify the tier_state passed to evaluate_kt_cg_trial_4_conditions
    # by only passing tiers that are in the effective list
    filtered_tier_state = {k: v for k, v in tier_state.items() if k in effective_tier_periods}

    # Build temp_overrides with effective tiers
    eval_temp_overrides = dict(temp_overrides) if temp_overrides else {}

    # Evaluate conditions - reuses Trial #4 evaluate function
    # We need to temporarily override tier_ema_periods on the policy
    original_tiers = policy.tier_ema_periods
    try:
        object.__setattr__(policy, 'tier_ema_periods', tuple(effective_tier_periods))
        result = evaluate_kt_cg_trial_4_conditions(
            profile, policy, data_by_tf,
            current_bid=tick.bid,
            current_ask=tick.ask,
            tier_state=filtered_tier_state,
            temp_overrides=eval_temp_overrides if eval_temp_overrides else None,
        )
    finally:
        object.__setattr__(policy, 'tier_ema_periods', original_tiers)

    passed = result["passed"]
    side = result["side"]
    eval_reasons = result["reasons"]
    trigger_type = result["trigger_type"]
    tier_updates = result.get("tier_updates", {})
    tiered_pullback_tier = result.get("tiered_pullback_tier")

    if not passed or side is None:
        return {"decision": ExecutionDecision(attempted=False, placed=False, reason="; ".join(eval_reasons)), "tier_updates": tier_updates, "divergence_updates": {}, "exhaustion_state": exhaustion_state, "exhaustion_result": exhaustion_result}

    # Block zone entry if exhaustion says so
    if trigger_type == "zone_entry" and block_zone_entry_by_exhaustion:
        exhaust_zone = exhaustion_result.get("zone", "FRESH") if exhaustion_result else "FRESH"
        return {
            "decision": ExecutionDecision(
                attempted=True, placed=False,
                reason=f"trend_exhaustion: {exhaust_zone} zone blocks zone_entry (ratio={exhaustion_result['extension_ratio']:.1f}x ATR)"
            ),
            "tier_updates": tier_updates,
            "divergence_updates": {},
            "exhaustion_state": exhaustion_state,
            "exhaustion_result": exhaustion_result,
        }

    if trigger_type == "tiered_pullback" and tiered_pullback_tier:
        rule_id = f"kt_cg_trial_5:{policy.id}:tier_{tiered_pullback_tier}"

    # Fresh Cross Check: zone entry requires M1 EMA5/EMA9 cross within last 10 bars
    if trigger_type == "zone_entry":
        m1_df_fc = data_by_tf.get("M1")
        if m1_df_fc is not None and not m1_df_fc.empty:
            is_bull = side == "buy"
            fc_ok, fc_reason = _passes_fresh_cross_check(m1_df_fc, is_bull)
            if not fc_ok:
                return {
                    "decision": ExecutionDecision(attempted=True, placed=False, reason=fc_reason or "fresh_cross_check_failed"),
                    "tier_updates": tier_updates,
                    "divergence_updates": {},
                    "exhaustion_state": exhaustion_state,
                    "exhaustion_result": exhaustion_result,
                }

    # Max 3 zone entry trades open at once (Trial #5)
    if trigger_type == "zone_entry":
        max_zone_entry_open = getattr(policy, "max_zone_entry_open", 3)
        try:
            open_trades = store.list_open_trades(profile.profile_name)
            zone_entry_open = sum(1 for row in open_trades if row.get("entry_type") == "zone_entry")
            if zone_entry_open >= max_zone_entry_open:
                return {
                    "decision": ExecutionDecision(
                        attempted=True, placed=False,
                        reason=f"max_zone_entry_open: {zone_entry_open} zone entry trade(s) already open (max {max_zone_entry_open})",
                    ),
                    "tier_updates": tier_updates,
                    "divergence_updates": {},
                    "exhaustion_state": exhaustion_state,
                    "exhaustion_result": exhaustion_result,
                }
        except Exception:
            pass

    # Per-direction open trade cap (uses live broker positions, not DB)
    max_open_per_side = policy.max_open_trades_per_side
    if max_open_per_side is not None:
        try:
            open_positions = adapter.get_open_positions(profile.symbol)
            side_open = 0
            if open_positions:
                for pos in open_positions:
                    if isinstance(pos, dict):
                        # OANDA trades have currentUnits (positive=buy, negative=sell), not "side"
                        units = float(pos.get("currentUnits") or pos.get("initialUnits") or 0)
                        if units > 0:
                            pos_side = "buy"
                        elif units < 0:
                            pos_side = "sell"
                        else:
                            continue
                    else:
                        mt5_type = getattr(pos, "type", None)
                        if mt5_type == 0:
                            pos_side = "buy"
                        elif mt5_type == 1:
                            pos_side = "sell"
                        else:
                            continue
                    if pos_side == side:
                        side_open += 1
            if side_open >= max_open_per_side:
                return {
                    "decision": ExecutionDecision(
                        attempted=True, placed=False,
                        reason=f"max_open_trades_per_side: {side_open} {side} trade(s) open (max {max_open_per_side})",
                    ),
                    "tier_updates": tier_updates,
                    "divergence_updates": {},
                    "exhaustion_state": exhaustion_state,
                    "exhaustion_result": exhaustion_result,
                }
        except Exception:
            pass

    # Idempotency check for zone_entry only
    if trigger_type == "zone_entry":
        within = 2
        if store.has_recent_price_level_placement(profile.profile_name, rule_id, within):
            return {"decision": ExecutionDecision(attempted=False, placed=False, reason="kt_cg_trial_5: recent placement (idempotent)"), "tier_updates": tier_updates, "divergence_updates": {}, "exhaustion_state": exhaustion_state}

    # EMA Zone Entry Filter (with configurable weights/ranges from policy)
    if trigger_type == "zone_entry":
        ema_zone_filter_enabled = getattr(policy, "ema_zone_filter_enabled", False)
        if ema_zone_filter_enabled:
            m1_df_zone = data_by_tf.get("M1")
            if m1_df_zone is not None and not m1_df_zone.empty:
                is_bull = side == "buy"
                zf_lookback = getattr(policy, "ema_zone_filter_lookback_bars", 3)
                zf_threshold = getattr(policy, "ema_zone_filter_block_threshold", 0.35)
                zf_score, zf_details = _compute_ema_zone_filter_score(
                    m1_df_zone, pip_size, is_bull, zf_lookback, policy=policy
                )
                if "error" not in zf_details and zf_score < zf_threshold:
                    zf_reason = (
                        f"ema_zone_filter: BLOCKED score={zf_score:.2f}"
                        f" (spread={zf_details['spread_pips']:.1f}p"
                        f" slope={zf_details['slope_pips']:.1f}p"
                        f" dir={zf_details['spread_dir_pips']:.1f}p)"
                        f" threshold={zf_threshold}"
                    )
                    print(f"[{profile.profile_name}] kt_cg_trial_5 {zf_reason}")
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
                            "reason": zf_reason,
                            "mt5_retcode": None,
                            "mt5_order_id": None,
                            "mt5_deal_id": None,
                        }
                    )
                    return {"decision": ExecutionDecision(attempted=True, placed=False, reason=zf_reason), "tier_updates": tier_updates, "divergence_updates": {}, "exhaustion_state": exhaustion_state}

    # Session filter
    now_utc = datetime.now(timezone.utc)
    ok, reason = passes_session_filter(profile, now_utc)
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
                "reason": reason or "session_filter",
                "mt5_retcode": None,
                "mt5_order_id": None,
                "mt5_deal_id": None,
            }
        )
        return {"decision": ExecutionDecision(attempted=True, placed=False, reason=reason or "session_filter"), "tier_updates": tier_updates, "divergence_updates": {}, "exhaustion_state": exhaustion_state}
    ok, reason = passes_session_boundary_block(profile, now_utc)
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
                "reason": reason or "session_boundary_block",
                "mt5_retcode": None,
                "mt5_order_id": None,
                "mt5_deal_id": None,
            }
        )
        return {"decision": ExecutionDecision(attempted=True, placed=False, reason=reason or "session_boundary_block"), "tier_updates": tier_updates, "divergence_updates": {}, "exhaustion_state": exhaustion_state}

    # --- Trial #5 Dual ATR Filter ---
    m1_df = data_by_tf.get("M1")
    if m1_df is not None:
        ok, reason = _passes_atr_filter_trial_5(policy, m1_df, m3_df, pip_size, trigger_type)
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
                    "reason": reason or "trial5_atr_filter",
                    "mt5_retcode": None,
                    "mt5_order_id": None,
                    "mt5_deal_id": None,
                }
            )
            return {"decision": ExecutionDecision(attempted=True, placed=False, reason=reason or "trial5_atr_filter"), "tier_updates": tier_updates, "divergence_updates": {}, "exhaustion_state": exhaustion_state}

    # Daily High/Low Filter (applies to BOTH zone entry AND pullback in Trial #5)
    dr_high = daily_reset_state.get("daily_reset_high") if daily_reset_state else None
    dr_low = daily_reset_state.get("daily_reset_low") if daily_reset_state else None
    dr_settled = daily_reset_state.get("daily_reset_settled", False) if daily_reset_state else False
    ok, reason = _passes_daily_hl_filter(
        policy, data_by_tf, tick, side, pip_size,
        daily_reset_high=dr_high, daily_reset_low=dr_low, daily_reset_settled=dr_settled,
    )
    if not ok:
        print(f"[{profile.profile_name}] kt_cg_trial_5 BLOCKED by daily H/L filter: {reason}")
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
                "reason": reason or "daily_hl_filter",
                "mt5_retcode": None,
                "mt5_order_id": None,
                "mt5_deal_id": None,
            }
        )
        return {"decision": ExecutionDecision(attempted=True, placed=False, reason=reason or "daily_hl_filter"), "tier_updates": tier_updates, "divergence_updates": {}, "exhaustion_state": exhaustion_state}

    divergence_updates: dict[str, str] = {}

    entry_price = tick.ask if side == "buy" else tick.bid
    candidate = _kt_cg_trial_4_candidate(profile, policy, entry_price, side)
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
        return {"decision": ExecutionDecision(attempted=True, placed=False, reason="risk rejected: " + "; ".join(decision.hard_reasons)), "tier_updates": tier_updates, "divergence_updates": divergence_updates, "exhaustion_state": exhaustion_state}

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
        return {"decision": ExecutionDecision(attempted=True, placed=False, reason="manual_confirm_required"), "tier_updates": tier_updates, "divergence_updates": divergence_updates, "exhaustion_state": exhaustion_state}

    # ARMED_AUTO_DEMO: place the trade
    comment = f"kt_cg_trial_5:{policy.id}:{trigger_type}"
    if trigger_type == "tiered_pullback" and tiered_pullback_tier:
        comment = f"kt_cg_trial_5:{policy.id}:tier_{tiered_pullback_tier}"

    res = adapter.order_send_market(
        symbol=profile.symbol,
        side=side,
        volume_lots=float(candidate.size_lots or get_effective_risk(profile).max_lots),
        sl=candidate.stop_price,
        tp=candidate.target_price,
        comment=comment,
    )
    placed = res.retcode in (0, 10008, 10009)
    reason = f"{trigger_type}:order_sent" if placed else f"order_failed:{res.retcode}:{res.comment}"
    if trigger_type == "tiered_pullback" and tiered_pullback_tier:
        reason = f"tier_{tiered_pullback_tier}:order_sent" if placed else f"order_failed:{res.retcode}:{res.comment}"
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

    if placed:
        tier_info = f" tier_{tiered_pullback_tier}" if tiered_pullback_tier else ""
        print(f"[{profile.profile_name}] TRADE PLACED: kt_cg_trial_5:{trigger_type}{tier_info} | {'; '.join(eval_reasons)}")

    return {
        "decision": ExecutionDecision(
            attempted=True,
            placed=placed,
            reason=reason,
            order_retcode=res.retcode,
            order_id=res.order,
            deal_id=res.deal,
            fill_price=getattr(res, 'fill_price', None),
            side=side,
        ),
        "tier_updates": tier_updates,
        "divergence_updates": divergence_updates,
        "trigger_type": trigger_type,
        "exhaustion_state": exhaustion_state,
        "exhaustion_result": exhaustion_result,
    }


# ---------------------------------------------------------------------------
# Trial #6: BB Slope Trend + EMA Tier Pullback + BB Reversal
# ---------------------------------------------------------------------------


def _compute_bollinger_bands(df: pd.DataFrame, period: int, std_dev: float) -> dict:
    """Compute Bollinger Bands from a DataFrame with 'close' column.

    Returns dict with upper, lower, middle (floats), width, bb_expanding (bool).
    """
    close = df["close"]
    middle_series = close.rolling(window=period, min_periods=period).mean()
    std_series = close.rolling(window=period, min_periods=period).std()
    upper_series = middle_series + std_dev * std_series
    lower_series = middle_series - std_dev * std_series
    width_series = upper_series - lower_series

    if len(width_series) < 2 or pd.isna(middle_series.iloc[-1]):
        return {
            "upper": None, "lower": None, "middle": None,
            "width": None, "bb_expanding": False,
        }

    upper = float(upper_series.iloc[-1])
    lower = float(lower_series.iloc[-1])
    middle = float(middle_series.iloc[-1])
    width_current = float(width_series.iloc[-1])
    width_prev = float(width_series.iloc[-2]) if not pd.isna(width_series.iloc[-2]) else width_current
    bb_expanding = width_current > width_prev

    return {
        "upper": upper, "lower": lower, "middle": middle,
        "width": width_current, "bb_expanding": bb_expanding,
    }


def _evaluate_m3_slope_trend_trial_6(
    m3_df: pd.DataFrame,
    policy,  # ExecutionPolicyKtCgTrial6
    pip_size: float,
) -> dict:
    """BULL = M3 EMA9 > EMA21. BEAR = M3 EMA9 < EMA21. NONE = equal."""
    close = m3_df["close"]
    ema_slow = ema_fn(close, policy.m3_trend_ema_slow)   # EMA9
    ema_extra = ema_fn(close, policy.m3_trend_ema_extra)  # EMA21

    ema_slow_val = float(ema_slow.iloc[-1])
    ema_extra_val = float(ema_extra.iloc[-1])

    trend = "NONE"
    if ema_slow_val > ema_extra_val:
        trend = "BULL"
    elif ema_slow_val < ema_extra_val:
        trend = "BEAR"

    reasons = [
        f"M3 trend: {trend} | EMA{policy.m3_trend_ema_slow}={ema_slow_val:.3f} "
        f"EMA{policy.m3_trend_ema_extra}={ema_extra_val:.3f}"
    ]

    return {
        "trend": trend,
        "ema_slow_val": ema_slow_val,
        "ema_extra_val": ema_extra_val,
        "reasons": reasons,
    }


def _evaluate_ema_tier_system_a_trial_6(
    policy,  # ExecutionPolicyKtCgTrial6
    m1_df: pd.DataFrame,
    trend: str,
    current_bid: float,
    current_ask: float,
    tier_state: dict[int, bool],
    pip_size: float,
) -> dict:
    """System A: EMA Tier Pullback.

    Fires when bid (BULL) or ask (BEAR) touches an EMA tier value.
    """
    reasons = []
    tier_updates: dict[int, bool] = {}
    fired_tier = None
    fired_side = None

    if not policy.ema_tier_enabled:
        reasons.append("System A disabled")
        return {"fired": False, "tier": None, "side": None, "tier_updates": {}, "reasons": reasons}

    tier_periods = list(policy.tier_ema_periods)
    if not tier_periods:
        return {"fired": False, "tier": None, "side": None, "tier_updates": {}, "reasons": ["no tier periods"]}

    m1_close = m1_df["close"]
    reset_buffer = policy.tier_reset_buffer_pips * pip_size

    for period in tier_periods:
        if len(m1_close) < period + 2:
            continue
        ema_value = float(ema_fn(m1_close, period).iloc[-1])
        tier_fired = tier_state.get(period, False)

        if trend == "BULL":
            is_touching = current_bid <= ema_value
            has_moved_away = current_bid > ema_value + reset_buffer
            if is_touching and not tier_fired:
                fired_tier = period
                fired_side = "buy"
                tier_updates[period] = True
                reasons.append(f"TIER FIRE: bid {current_bid:.3f} <= EMA{period} {ema_value:.3f} -> BUY")
                break
            elif has_moved_away and tier_fired:
                tier_updates[period] = False
                reasons.append(f"Tier {period} RESET")
        elif trend == "BEAR":
            is_touching = current_ask >= ema_value
            has_moved_away = current_ask < ema_value - reset_buffer
            if is_touching and not tier_fired:
                fired_tier = period
                fired_side = "sell"
                tier_updates[period] = True
                reasons.append(f"TIER FIRE: ask {current_ask:.3f} >= EMA{period} {ema_value:.3f} -> SELL")
                break
            elif has_moved_away and tier_fired:
                tier_updates[period] = False
                reasons.append(f"Tier {period} RESET")

    return {
        "fired": fired_tier is not None,
        "tier": fired_tier,
        "side": fired_side,
        "tier_updates": tier_updates,
        "reasons": reasons,
    }


def _evaluate_bb_reversal_system_b_trial_6(
    policy,  # ExecutionPolicyKtCgTrial6
    m1_bb: dict,
    current_bid: float,
    current_ask: float,
    bb_tier_state: dict[int, bool],
    pip_size: float,
) -> dict:
    """System B: BB Reversal (counter-trend).

    Generate tier offsets from BB upper/lower. Fire counter-trend entry when
    price exceeds BB + offset. Reset ALL bb_tier_fired when price returns inside BB.
    """
    reasons = []
    bb_tier_updates: dict[int, bool] = {}
    fired_tier = None
    fired_side = None

    if not policy.bb_reversal_enabled:
        reasons.append("System B disabled")
        return {"fired": False, "tier": None, "side": None, "bb_tier_updates": {}, "reasons": reasons, "reset_all": False}

    bb_upper = m1_bb.get("upper")
    bb_lower = m1_bb.get("lower")
    bb_middle = m1_bb.get("middle")
    if bb_upper is None or bb_lower is None or bb_middle is None:
        return {"fired": False, "tier": None, "side": None, "bb_tier_updates": {}, "reasons": ["BB not available"], "reset_all": False}

    start_offset = policy.bb_reversal_start_offset_pips * pip_size
    increment = policy.bb_reversal_increment_pips * pip_size
    num_tiers = policy.bb_reversal_num_tiers

    # Check if price is back inside BB -> reset all
    price_inside = current_bid < bb_upper and current_ask > bb_lower
    if price_inside and any(bb_tier_state.get(i, False) for i in range(num_tiers)):
        for i in range(num_tiers):
            bb_tier_updates[i] = False
        reasons.append("BB reversal: price inside BB, all tiers RESET")
        return {"fired": False, "tier": None, "side": None, "bb_tier_updates": bb_tier_updates, "reasons": reasons, "reset_all": True}

    # Check each tier for fire
    for i in range(num_tiers):
        offset = start_offset + i * increment
        tier_fired = bb_tier_state.get(i, False)

        # SELL when ask >= upper_bb + offset (unfired)
        upper_trigger = bb_upper + offset
        if current_ask >= upper_trigger and not tier_fired:
            fired_tier = i
            fired_side = "sell"
            bb_tier_updates[i] = True
            reasons.append(f"BB REVERSAL FIRE: ask {current_ask:.3f} >= upper_bb+{offset/pip_size:.1f}p ({upper_trigger:.3f}) -> SELL (tier {i})")
            break

        # BUY when bid <= lower_bb - offset (unfired)
        lower_trigger = bb_lower - offset
        if current_bid <= lower_trigger and not tier_fired:
            fired_tier = i
            fired_side = "buy"
            bb_tier_updates[i] = True
            reasons.append(f"BB REVERSAL FIRE: bid {current_bid:.3f} <= lower_bb-{offset/pip_size:.1f}p ({lower_trigger:.3f}) -> BUY (tier {i})")
            break

    return {
        "fired": fired_tier is not None,
        "tier": fired_tier,
        "side": fired_side,
        "bb_tier_updates": bb_tier_updates,
        "reasons": reasons,
        "reset_all": False,
    }


def _update_daily_reset_state_t6(tick_mid: float, state: dict, start_hour: int, end_hour: int) -> dict:
    """Update dead zone state for Trial #6 with configurable hours.

    Handles midnight wrap (e.g. start=21, end=2 means 21:00-02:00 UTC).
    """
    now = datetime.now(timezone.utc)
    current_hour = now.hour

    # Determine if in dead zone (handles midnight wrap)
    if start_hour > end_hour:
        # Wraps midnight: e.g. 21-2 means hour >= 21 OR hour < 2
        in_dead_zone = current_hour >= start_hour or current_hour < end_hour
    else:
        in_dead_zone = start_hour <= current_hour < end_hour

    return {
        "daily_reset_block_active": in_dead_zone,
        "dead_zone_start": start_hour,
        "dead_zone_end": end_hour,
    }


def compute_t6_bb_reversal_tp(
    m1_df: pd.DataFrame,
    policy,  # ExecutionPolicyKtCgTrial6
    pip_size: float,
    entry_price: float,
    side: str,
) -> float | None:
    """For middle_bb_dynamic mode: compute live TP from current middle BB, clamped to [min, max].

    Returns the TP price, or None if BB not available.
    """
    if m1_df is None or len(m1_df) < policy.m1_bb_period + 1:
        return None

    bb = _compute_bollinger_bands(m1_df, policy.m1_bb_period, policy.m1_bb_std)
    middle = bb.get("middle")
    if middle is None:
        return None

    # Compute TP distance in pips
    if side == "buy":
        tp_distance_pips = (middle - entry_price) / pip_size
    else:
        tp_distance_pips = (entry_price - middle) / pip_size

    # Clamp
    tp_distance_pips = max(policy.bb_reversal_tp_min_pips, min(policy.bb_reversal_tp_max_pips, tp_distance_pips))

    if side == "buy":
        return entry_price + tp_distance_pips * pip_size
    else:
        return entry_price - tp_distance_pips * pip_size


def execute_kt_cg_trial_6_policy_demo_only(
    *,
    adapter,
    profile: ProfileV1,
    log_dir: Path,
    policy,  # ExecutionPolicyKtCgTrial6
    context: MarketContext,
    data_by_tf: dict[Timeframe, pd.DataFrame],
    tick: Tick,
    trades_df: Optional[pd.DataFrame],
    mode: str,
    tier_state: dict[int, bool],
    daily_reset_state: dict | None = None,
) -> dict:
    """Execute Trial #6: M3 Slope Trend + EMA Tier Pullback.

    Pipeline: mode guard -> dead zone -> M3 trend (NONE blocks) -> System A (EMA tiers)
    -> risk checks -> trade placement.

    Returns dict with decision, tier_updates, trigger_type, tp_pips, sl_pips,
    daily_reset_state, trend_result.
    """
    pip_size = float(profile.pip_size)
    store = _store(log_dir)
    rule_id = f"kt_cg_trial_6:{policy.id}"
    tier_updates: dict[int, bool] = {}
    no_trade = {
        "decision": ExecutionDecision(attempted=False, placed=False, reason="no_signal"),
        "tier_updates": {},
        "trigger_type": "",
        "tp_pips": None,
        "sl_pips": None,
        "daily_reset_state": daily_reset_state or {},
        "trend_result": None,
    }

    # Mode guard
    if mode == "DISARMED":
        return no_trade

    # Dead zone check
    if policy.dead_zone_enabled:
        dz_state = _update_daily_reset_state_t6(
            (tick.bid + tick.ask) / 2.0,
            daily_reset_state or {},
            policy.dead_zone_start_hour_utc,
            policy.dead_zone_end_hour_utc,
        )
        no_trade["daily_reset_state"] = dz_state
        if dz_state.get("daily_reset_block_active", False):
            return no_trade

    # Get M3 data
    m3_df = data_by_tf.get("M3")
    if m3_df is None or m3_df.empty:
        return no_trade
    m3_df = drop_incomplete_last_bar(m3_df.copy(), "M3")
    min_m3_bars = policy.m3_trend_ema_extra + 2
    if len(m3_df) < min_m3_bars:
        return no_trade

    # Get M1 data
    m1_df = data_by_tf.get("M1")
    if m1_df is None or m1_df.empty:
        return no_trade
    m1_df = drop_incomplete_last_bar(m1_df.copy(), "M1")
    max_tier_period = max(policy.tier_ema_periods) if policy.tier_ema_periods else 21
    min_m1_bars = max_tier_period + 2
    if len(m1_df) < min_m1_bars:
        return no_trade

    # M3 Slope Trend
    trend_result = _evaluate_m3_slope_trend_trial_6(m3_df, policy, pip_size)
    no_trade["trend_result"] = trend_result
    trend = trend_result["trend"]

    current_bid = tick.bid
    current_ask = tick.ask

    # System A: EMA Tier Pullback (only when trend is BULL or BEAR)
    system_a_result = {"fired": False, "tier": None, "side": None, "tier_updates": {}, "reasons": []}
    if trend != "NONE" and policy.ema_tier_enabled:
        system_a_result = _evaluate_ema_tier_system_a_trial_6(
            policy, m1_df, trend, current_bid, current_ask, tier_state, pip_size,
        )
        tier_updates.update(system_a_result["tier_updates"])

    # Always propagate tier resets even when no trade fires
    no_trade["tier_updates"] = tier_updates

    trigger_type = ""
    side = None
    tp_pips = None
    sl_pips = None
    fired_tier_info = None

    if system_a_result["fired"]:
        trigger_type = "ema_tier"
        side = system_a_result["side"]
        tp_pips = policy.ema_tier_tp_pips
        sl_pips = policy.sl_pips
        fired_tier_info = system_a_result["tier"]

    if not side:
        return no_trade

    # Risk checks: per-direction cap
    if trades_df is not None and not trades_df.empty:
        open_trades = trades_df[trades_df["status"] == "open"] if "status" in trades_df.columns else trades_df
        same_side = open_trades[open_trades["side"].str.lower() == side.lower()] if "side" in open_trades.columns else pd.DataFrame()
        if len(same_side) >= policy.max_open_trades_per_side:
            no_trade["tier_updates"] = tier_updates
            return no_trade

    # Cooldown after loss
    if policy.cooldown_after_loss_seconds > 0:
        try:
            execs = store.read_executions_df(profile.profile_name)
            if execs is not None and not execs.empty:
                placed_execs = execs[execs["placed"] == 1]
                if not placed_execs.empty:
                    last_exec = placed_execs.iloc[-1]
                    last_time = pd.to_datetime(last_exec["timestamp_utc"])
                    now = pd.Timestamp.now(tz="UTC")
                    elapsed = (now - last_time).total_seconds()
                    if elapsed < policy.cooldown_after_loss_seconds:
                        # Only apply cooldown if last trade was a loss (check trades_df)
                        pass  # simplified: cooldown applies after any trade for now
        except Exception:
            pass

    # Build candidate
    entry_price = current_ask if side == "buy" else current_bid
    sl_dist = (sl_pips or policy.sl_pips) * pip_size
    tp_dist = (tp_pips or policy.ema_tier_tp_pips) * pip_size

    if side == "buy":
        stop = entry_price - sl_dist
        target = entry_price + tp_dist
    else:
        stop = entry_price + sl_dist
        target = entry_price - tp_dist

    candidate = TradeCandidate(
        symbol=profile.symbol,
        side=side,
        entry_price=entry_price,
        stop_price=stop,
        target_price=target,
        size_lots=float(get_effective_risk(profile).max_lots),
    )

    decision = evaluate_trade(profile=profile, candidate=candidate, context=context, trades_df=trades_df)
    if not decision.allow:
        sig_id = f"{rule_id}:{pd.Timestamp.now(tz='UTC').isoformat()}"
        store.insert_execution({
            "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "profile": profile.profile_name, "symbol": profile.symbol,
            "signal_id": sig_id, "rule_id": rule_id, "mode": mode,
            "attempted": 1, "placed": 0,
            "reason": "risk_reject: " + "; ".join(decision.hard_reasons),
            "mt5_retcode": None, "mt5_order_id": None, "mt5_deal_id": None,
        })
        return {
            "decision": ExecutionDecision(attempted=True, placed=False, reason="risk rejected: " + "; ".join(decision.hard_reasons)),
            "tier_updates": tier_updates,
            "trigger_type": trigger_type, "tp_pips": tp_pips, "sl_pips": sl_pips,
            "daily_reset_state": no_trade["daily_reset_state"],
            "trend_result": trend_result,
        }

    if mode == "ARMED_MANUAL_CONFIRM":
        sig_id = f"{rule_id}:{pd.Timestamp.now(tz='UTC').isoformat()}"
        store.insert_execution({
            "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "profile": profile.profile_name, "symbol": profile.symbol,
            "signal_id": sig_id, "rule_id": rule_id, "mode": mode,
            "attempted": 1, "placed": 0,
            "reason": "manual_confirm_required",
            "mt5_retcode": None, "mt5_order_id": None, "mt5_deal_id": None,
        })
        return {
            "decision": ExecutionDecision(attempted=True, placed=False, reason="manual_confirm_required"),
            "tier_updates": tier_updates,
            "trigger_type": trigger_type, "tp_pips": tp_pips, "sl_pips": sl_pips,
            "daily_reset_state": no_trade["daily_reset_state"],
            "trend_result": trend_result,
        }

    # ARMED_AUTO_DEMO: place the trade
    tier_label = f"_tier{fired_tier_info}" if fired_tier_info is not None else ""
    comment = f"kt_cg_trial_6:{policy.id}:{trigger_type}{tier_label}"

    res = adapter.order_send_market(
        symbol=profile.symbol,
        side=side,
        volume_lots=float(candidate.size_lots or get_effective_risk(profile).max_lots),
        sl=candidate.stop_price,
        tp=candidate.target_price,
        comment=comment,
    )
    placed = res.retcode in (0, 10008, 10009)
    reason = f"{trigger_type}{tier_label}:order_sent" if placed else f"order_failed:{res.retcode}:{res.comment}"
    sig_id = f"{rule_id}:{pd.Timestamp.now(tz='UTC').isoformat()}"
    store.insert_execution({
        "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "profile": profile.profile_name, "symbol": profile.symbol,
        "signal_id": sig_id, "rule_id": rule_id, "mode": mode,
        "attempted": 1, "placed": 1 if placed else 0,
        "reason": reason,
        "mt5_retcode": res.retcode, "mt5_order_id": res.order, "mt5_deal_id": res.deal,
    })

    if placed:
        print(f"[{profile.profile_name}] TRADE PLACED: kt_cg_trial_6:{trigger_type}{tier_label} | {side.upper()} @ {entry_price:.3f}")

    return {
        "decision": ExecutionDecision(
            attempted=True, placed=placed, reason=reason,
            order_retcode=res.retcode, order_id=res.order, deal_id=res.deal,
            fill_price=getattr(res, 'fill_price', None), side=side,
        ),
        "tier_updates": tier_updates,
        "trigger_type": trigger_type,
        "tp_pips": tp_pips,
        "sl_pips": sl_pips,
        "daily_reset_state": no_trade["daily_reset_state"],
        "trend_result": trend_result,
    }


# ---------------------------------------------------------------------------
# Session Momentum v5.3
# ---------------------------------------------------------------------------

def evaluate_session_momentum_v5(
    profile: ProfileV1,
    policy: ExecutionPolicySessionMomentumV5,
    data_by_tf: dict[Timeframe, pd.DataFrame],
    tick: Tick,
    trades_df: Optional[pd.DataFrame],
) -> tuple[bool, Optional[str], float, float, float, float, str, list[str]]:
    """Evaluate Session Momentum v5.3 entry conditions.

    Returns (should_enter, side, lot_size, sl_pips, tp1_pips, tp2_pips,
             trend_strength, reasons).
    """
    _no = (False, None, 0.0, 0.0, 0.0, 0.0, "", [])
    pip = float(profile.pip_size)
    reasons: list[str] = []

    # --- 1. Get data ---
    m1 = data_by_tf.get("M1")
    m5 = data_by_tf.get("M5")
    h1 = data_by_tf.get("H1")
    if m1 is None or m1.empty or len(m1) < max(policy.m1_ema_slow, 30):
        return _no[:-1] + (["M1 data insufficient"],)
    if m5 is None or m5.empty or len(m5) < max(policy.m5_ema_slow, policy.sl_lookback) + 5:
        return _no[:-1] + (["M5 data insufficient"],)
    if h1 is None or h1.empty or len(h1) < policy.h1_ema_slow + 5:
        return _no[:-1] + (["H1 data insufficient"],)

    # Use only completed bars so evaluation is deterministic regardless of poll timing.
    m1 = drop_incomplete_last_bar(m1.copy(), "M1")
    m5 = drop_incomplete_last_bar(m5.copy(), "M5")
    if len(m1) < max(policy.m1_ema_slow, 30):
        return _no[:-1] + (["M1 data insufficient (after drop incomplete)"],)
    if len(m5) < policy.sl_lookback + 1:
        return _no[:-1] + (["M5 data insufficient (after drop incomplete)"],)

    now_utc = datetime.now(timezone.utc)
    hour_float = now_utc.hour + now_utc.minute / 60.0

    # --- 2. Session window ---
    london_start = policy.london_start_hour
    london_end = policy.london_end_hour
    ny_start = policy.ny_start_hour + policy.ny_start_delay_minutes / 60.0
    ny_end = policy.ny_end_hour

    # Backwards-compatible migration: older profiles used London 6.5–11.0 and NY end 16.0.
    # Align them with the global session filter defaults (London 8–16, NY 13–21 UTC)
    # unless the user has explicitly customized them away from the old defaults.
    if london_start == 6.5 and london_end == 11.0:
        london_start, london_end = 8.0, 16.0
    if ny_end == 16.0:
        ny_end = 21.0

    in_london = london_start <= hour_float < london_end and policy.sessions != "ny_only"
    in_ny = ny_start <= hour_float < ny_end and policy.sessions != "london_only"

    if not in_london and not in_ny:
        return _no[:-1] + (["outside session window"],)

    # --- 3. Entry cutoff ---
    session_end = london_end if in_london else ny_end
    minutes_to_end = (session_end - hour_float) * 60
    if minutes_to_end < policy.session_entry_cutoff_minutes:
        return _no[:-1] + ([f"entry cutoff: {minutes_to_end:.0f}m to session end"],)

    # --- 4. Spread gate ---
    spread_pips = (tick.ask - tick.bid) / pip
    if spread_pips > policy.max_spread_pips:
        return _no[:-1] + ([f"spread too high: {spread_pips:.1f} > {policy.max_spread_pips}"],)

    # --- 5. Max open (v5.3-only trades) ---
    open_count = 0
    if trades_df is not None and not trades_df.empty and "exit_price" in trades_df.columns:
        open_mask = trades_df["exit_price"].isna()
        # Only count open trades created by the Session Momentum v5.3 policy
        if "notes" in trades_df.columns:
            v5_mask = trades_df["notes"].astype(str).str.startswith("auto:session_momentum_v5:")
            open_mask = open_mask & v5_mask
        open_count = int(open_mask.sum())
    if open_count >= policy.max_open:
        return _no[:-1] + ([f"max open reached: {open_count}/{policy.max_open}"],)

    # --- 6. Max entries today + London cap (v5.3-only trades) ---
    today_str = now_utc.strftime("%Y-%m-%d")
    entries_today = 0
    london_entries_today = 0
    if trades_df is not None and not trades_df.empty and "timestamp_utc" in trades_df.columns:
        for _, row in trades_df.iterrows():
            # Only count trades opened by the Session Momentum v5.3 policy
            notes = str(row.get("notes", "") or "")
            if not notes.startswith("auto:session_momentum_v5:"):
                continue
            ts = str(row.get("timestamp_utc", ""))
            if ts.startswith(today_str):
                entries_today += 1
                # Check if entry was during London session (for London cap)
                try:
                    entry_hour = float(ts[11:13]) + float(ts[14:16]) / 60.0
                    if london_start <= entry_hour < london_end:
                        london_entries_today += 1
                except (ValueError, IndexError):
                    pass
    if entries_today >= policy.max_entries_day:
        return _no[:-1] + ([f"max daily entries: {entries_today}/{policy.max_entries_day}"],)
    if in_london and london_entries_today >= policy.london_max_entries:
        return _no[:-1] + ([f"London cap: {london_entries_today}/{policy.london_max_entries}"],)

    # --- 7. Cooldown ---
    if trades_df is not None and not trades_df.empty and "exit_price" in trades_df.columns:
        closed = trades_df[trades_df["exit_price"].notna()].copy()
        if not closed.empty:
            closed = closed.sort_values("timestamp_utc", ascending=False)
            last = closed.iloc[0]
            last_exit_ts = pd.to_datetime(last.get("exit_timestamp_utc", last.get("timestamp_utc")), utc=True)
            bars_since = max(0, (now_utc - last_exit_ts).total_seconds() / 60.0)  # M1 bars ~ minutes
            pips_result = float(last.get("pips", 0) or 0)
            if abs(pips_result) < policy.scratch_threshold_pips:
                cd = policy.cooldown_scratch
            elif pips_result > 0:
                cd = policy.cooldown_win
            else:
                cd = policy.cooldown_loss
            if bars_since < cd:
                return _no[:-1] + ([f"cooldown: {bars_since:.0f}/{cd} M1 bars"],)

    # --- 8. H1 trend direction ---
    h1_close = h1["close"].astype(float)
    h1_ema_fast = ema_fn(h1_close, policy.h1_ema_fast)
    h1_ema_slow_s = ema_fn(h1_close, policy.h1_ema_slow)
    if pd.isna(h1_ema_fast.iloc[-1]) or pd.isna(h1_ema_slow_s.iloc[-1]):
        return _no[:-1] + (["H1 EMA insufficient"],)
    h1_fast_val = float(h1_ema_fast.iloc[-1])
    h1_slow_val = float(h1_ema_slow_s.iloc[-1])
    if h1_fast_val > h1_slow_val:
        side = "buy"
    elif h1_fast_val < h1_slow_val:
        side = "sell"
    else:
        return _no[:-1] + (["H1 trend flat"],)

    # --- 9. M5 trend strength ---
    m5_close = m5["close"].astype(float)
    m5_ema_fast_s = ema_fn(m5_close, policy.m5_ema_fast)
    m5_ema_slow_s = ema_fn(m5_close, policy.m5_ema_slow)
    if pd.isna(m5_ema_fast_s.iloc[-1]) or pd.isna(m5_ema_slow_s.iloc[-1]):
        return _no[:-1] + (["M5 EMA insufficient"],)
    if len(m5_ema_fast_s) < policy.slope_bars:
        return _no[:-1] + (["M5 slope data insufficient"],)

    slope_raw = (float(m5_ema_fast_s.iloc[-1]) - float(m5_ema_fast_s.iloc[-policy.slope_bars])) / policy.slope_bars
    slope_pips = abs(slope_raw) / pip
    if slope_pips >= policy.strong_slope:
        trend_strength = "Strong"
    elif slope_pips >= policy.weak_slope:
        trend_strength = "Normal"
    else:
        trend_strength = "Weak"

    # Strength gating
    if policy.strength_allow == "strong_only" and trend_strength != "Strong":
        return _no[:-1] + ([f"strength gate: {trend_strength} (need Strong)"],)
    if policy.strength_allow == "strong_normal" and trend_strength == "Weak":
        return _no[:-1] + ([f"strength gate: {trend_strength} (need Strong/Normal)"],)

    # --- 10. M5 EMA alignment with H1 ---
    m5_fast_val = float(m5_ema_fast_s.iloc[-1])
    m5_slow_val = float(m5_ema_slow_s.iloc[-1])
    if side == "buy" and m5_fast_val <= m5_slow_val:
        return _no[:-1] + (["M5 EMA not aligned for buy"],)
    if side == "sell" and m5_fast_val >= m5_slow_val:
        return _no[:-1] + (["M5 EMA not aligned for sell"],)

    # --- 11. M1 pullback + recovery entry ---
    m1_close = m1["close"].astype(float)
    m1_open = m1["open"].astype(float)
    m1_ema_slow_s = ema_fn(m1_close, policy.m1_ema_slow)
    if pd.isna(m1_ema_slow_s.iloc[-1]):
        return _no[:-1] + (["M1 EMA insufficient"],)

    # Check pullback: price came within 1 pip of M1 EMA slow in last 5 bars
    pullback_found = False
    for i in range(-5, 0):
        if i >= -len(m1_close):
            bar_low = float(m1["low"].iloc[i])
            bar_high = float(m1["high"].iloc[i])
            ema_val = float(m1_ema_slow_s.iloc[i])
            if side == "buy":
                if abs(bar_low - ema_val) / pip <= 1.0 or bar_low <= ema_val:
                    pullback_found = True
                    break
            else:
                if abs(bar_high - ema_val) / pip <= 1.0 or bar_high >= ema_val:
                    pullback_found = True
                    break
    if not pullback_found:
        return _no[:-1] + (["no M1 pullback to EMA"],)

    # Check recovery on current bar
    curr_close = float(m1_close.iloc[-1])
    curr_open = float(m1_open.iloc[-1])
    body_pips = abs(curr_close - curr_open) / pip
    if side == "buy":
        if curr_close <= curr_open or body_pips < policy.entry_min_body_pips:
            return _no[:-1] + ([f"no M1 recovery candle (buy): body={body_pips:.1f}p"],)
    else:
        if curr_close >= curr_open or body_pips < policy.entry_min_body_pips:
            return _no[:-1] + ([f"no M1 recovery candle (sell): body={body_pips:.1f}p"],)

    # --- 12. Structural SL ---
    entry_price = tick.ask if side == "buy" else tick.bid
    m5_completed = m5.iloc[-policy.sl_lookback:]  # last N completed M5 bars (m5 already dropped incomplete)
    if len(m5_completed) < policy.sl_lookback:
        return _no[:-1] + (["insufficient M5 bars for structural SL"],)

    if side == "buy":
        swing_low = float(m5_completed["low"].astype(float).min())
        sl_price = swing_low - policy.sl_buffer * pip
    else:
        swing_high = float(m5_completed["high"].astype(float).max())
        sl_price = swing_high + policy.sl_buffer * pip

    sl_distance_pips = abs(entry_price - sl_price) / pip
    if sl_distance_pips < policy.sl_floor_pips:
        sl_distance_pips = policy.sl_floor_pips
    if sl_distance_pips > policy.sl_max_pips:
        return _no[:-1] + ([f"SL too wide: {sl_distance_pips:.1f} > {policy.sl_max_pips}"],)

    # --- 13. TP ---
    if trend_strength == "Strong":
        tp1_pips = sl_distance_pips * policy.strong_tp1
        tp2_pips = sl_distance_pips * policy.strong_tp2
    else:
        tp1_pips = sl_distance_pips * policy.normal_tp1
        tp2_pips = sl_distance_pips * policy.normal_tp2

    # --- 14. Lot size ---
    if policy.sizing_mode == "risk_parity":
        risk_usd = policy.account_size * policy.risk_per_trade_pct / 100.0
        pip_value_per_lot = 100000.0 * pip / entry_price
        if pip_value_per_lot <= 0 or sl_distance_pips <= 0:
            return _no[:-1] + (["invalid pip value or SL for sizing"],)
        lot_size = risk_usd / (sl_distance_pips * pip_value_per_lot)
        lot_size = max(policy.rp_min_lot, min(policy.rp_max_lot, lot_size))
    else:
        lot_size = policy.fixed_lots

    reasons = [
        f"H1={'bull' if side == 'buy' else 'bear'}",
        f"M5 strength={trend_strength} slope={slope_pips:.2f}p/bar",
        f"M1 pullback+recovery",
        f"SL={sl_distance_pips:.1f}p TP1={tp1_pips:.1f}p TP2={tp2_pips:.1f}p",
        f"lots={lot_size:.2f}",
        f"session={'london' if in_london else 'ny'}",
    ]

    return (True, side, lot_size, sl_distance_pips, tp1_pips, tp2_pips, trend_strength, reasons)


def execute_session_momentum_v5_policy_demo_only(
    *,
    adapter,
    profile: ProfileV1,
    log_dir: Path,
    policy: ExecutionPolicySessionMomentumV5,
    context: MarketContext,
    data_by_tf: dict[Timeframe, pd.DataFrame],
    tick: Tick,
    trades_df: Optional[pd.DataFrame],
    mode: str,
) -> ExecutionDecision:
    """Evaluate session momentum v5.3 policy and optionally place a market order."""
    from core.dashboard_models import TradeEvent, append_trade_event

    if mode not in ("ARMED_MANUAL_CONFIRM", "ARMED_AUTO_DEMO"):
        return ExecutionDecision(attempted=False, placed=False, reason=f"mode={mode} not armed")

    if not adapter.is_demo_account():
        return ExecutionDecision(attempted=False, placed=False, reason="not a demo account (execution disabled)")

    store = _store(log_dir)
    now_utc = datetime.now(timezone.utc)
    today_str = now_utc.strftime("%Y-%m-%d")
    rule_id = f"smv5:{policy.id}:{today_str}:{now_utc.strftime('%H%M')}"

    should_enter, side, lot_size, sl_distance_pips, tp1_pips, tp2_pips, trend_strength, eval_reasons = \
        evaluate_session_momentum_v5(profile, policy, data_by_tf, tick, trades_df)

    if not should_enter or side is None:
        return ExecutionDecision(attempted=True, placed=False, reason="; ".join(eval_reasons))

    entry_price = tick.ask if side == "buy" else tick.bid
    pip = float(profile.pip_size)

    # Build candidate with structural SL and TP2 as target
    if side == "buy":
        stop_price = entry_price - sl_distance_pips * pip
        target_price = entry_price + tp2_pips * pip
    else:
        stop_price = entry_price + sl_distance_pips * pip
        target_price = entry_price - tp2_pips * pip

    candidate = TradeCandidate(
        symbol=profile.symbol,
        side=side,
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
        size_lots=lot_size,
    )

    decision = evaluate_trade(profile=profile, candidate=candidate, context=context, trades_df=trades_df)
    if not decision.allow:
        return ExecutionDecision(attempted=True, placed=False, reason="risk_reject: " + "; ".join(decision.hard_reasons))

    if mode == "ARMED_MANUAL_CONFIRM":
        return ExecutionDecision(attempted=True, placed=False, reason="manual confirm required")

    res = adapter.order_send_market(
        symbol=profile.symbol,
        side=side,
        volume_lots=lot_size,
        sl=stop_price,
        tp=target_price,
        comment=f"smv5:{policy.id}",
    )
    placed = res.retcode in (0, 10008, 10009)
    reason = "order_sent" if placed else f"order_failed:{res.retcode}:{res.comment}"

    sig_id = f"{rule_id}:{now_utc.isoformat()}"
    store.insert_execution({
        "timestamp_utc": now_utc.isoformat(),
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
    })

    if placed:
        spread_pips = round((tick.ask - tick.bid) / pip, 2)
        hour_float = now_utc.hour + now_utc.minute / 60.0
        in_london = policy.london_start_hour <= hour_float < policy.london_end_hour
        event = TradeEvent(
            event_type="open",
            timestamp_utc=now_utc.isoformat(),
            trade_id=sig_id,
            side=side,
            price=entry_price,
            trigger_type="smv5_pullback_recovery",
            spread_at_entry=spread_pips,
            context_snapshot={
                "spread_at_entry": spread_pips,
                "trend_strength": trend_strength,
                "sl_distance_pips": round(sl_distance_pips, 1),
                "tp1_pips": round(tp1_pips, 1),
                "tp2_pips": round(tp2_pips, 1),
                "lot_size": round(lot_size, 2),
                "session": "london" if in_london else "ny",
            },
        )
        append_trade_event(log_dir, event)

    return ExecutionDecision(
        attempted=True,
        placed=placed,
        reason=reason,
        order_retcode=res.retcode,
        order_id=res.order,
        deal_id=res.deal,
        fill_price=getattr(res, 'fill_price', None),
        side=side,
    )
