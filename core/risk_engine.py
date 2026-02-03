from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from .models import MarketContext, RiskDecision, RiskSizing, TradeCandidate
from .profile import ProfileV1, get_effective_risk


def _parse_ts_utc(s: str) -> Optional[datetime]:
    try:
        ts = pd.to_datetime(s, utc=True)
        if pd.isna(ts):
            return None
        return ts.to_pydatetime()
    except Exception:
        return None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _pip_distance(entry: float, other: float, pip_size: float) -> float:
    return abs(entry - other) / pip_size


def evaluate_trade(
    *,
    profile: ProfileV1,
    candidate: TradeCandidate,
    context: MarketContext,
    trades_df: Optional[pd.DataFrame] = None,
    account_equity: Optional[float] = None,
    pip_value_per_lot: Optional[float] = None,
) -> RiskDecision:
    """Evaluate candidate trade against hard risk rules.

    - trades_df: existing trades log for this profile (can be CSV/SQLite output)
    - account_equity: optional; used only if risk_per_trade_pct sizing is enabled
    - pip_value_per_lot: optional; used for size suggestion based on stop distance
    """
    hard: list[str] = []
    warn: list[str] = []

    r = get_effective_risk(profile)

    # --- Spread gate ---
    if context.spread_pips is None:
        warn.append("spread_pips missing; cannot enforce max_spread_pips")
    else:
        if context.spread_pips > r.max_spread_pips:
            hard.append(f"spread too wide: {context.spread_pips:.3f} > max {r.max_spread_pips:.3f} pips")

    # --- Max trades/day gate ---
    trades_today = 0
    if trades_df is not None and not trades_df.empty and "timestamp_utc" in trades_df.columns:
        ts = pd.to_datetime(trades_df["timestamp_utc"], utc=True, errors="coerce")
        today = pd.Timestamp.now(tz="UTC").date()
        trades_today = int((ts.dt.date == today).sum())
    if trades_today >= r.max_trades_per_day:
        hard.append(f"max_trades_per_day reached: {trades_today} >= {r.max_trades_per_day}")

    # --- Max open trades gate ---
    if trades_df is not None and not trades_df.empty and "exit_price" in trades_df.columns:
        exit_price = pd.to_numeric(trades_df["exit_price"], errors="coerce")
        open_trades = int(exit_price.isna().sum())
        if open_trades >= r.max_open_trades:
            hard.append(f"max_open_trades reached: {open_trades} >= {r.max_open_trades}")

    # --- Cooldown after loss gate ---
    if r.cooldown_minutes_after_loss > 0 and trades_df is not None and not trades_df.empty:
        if "exit_timestamp_utc" in trades_df.columns and "pips" in trades_df.columns:
            closed = trades_df[pd.to_numeric(trades_df["exit_price"], errors="coerce").notna()].copy() if "exit_price" in trades_df.columns else trades_df.copy()
            if not closed.empty:
                closed["exit_ts"] = pd.to_datetime(closed["exit_timestamp_utc"], utc=True, errors="coerce")
                closed["pips_num"] = pd.to_numeric(closed["pips"], errors="coerce")
                closed = closed.dropna(subset=["exit_ts", "pips_num"]).sort_values("exit_ts")
                if not closed.empty:
                    last = closed.iloc[-1]
                    if float(last["pips_num"]) < 0:
                        last_exit: pd.Timestamp = last["exit_ts"]
                        # Ensure we compare against a sane, non-negative delta to avoid
                        # confusing messages like "cooldown active after loss: -119m < 5m"
                        now_utc = pd.Timestamp.now(tz="UTC")
                        delta = now_utc - last_exit
                        # If timestamps are skewed and delta is negative, treat as zero
                        if delta < pd.Timedelta(0):
                            delta = pd.Timedelta(0)
                        if delta < pd.Timedelta(minutes=r.cooldown_minutes_after_loss):
                            minutes = int(delta.total_seconds() / 60)
                            if minutes < 0:
                                minutes = 0
                            hard.append(f"cooldown active after loss: {minutes}m < {r.cooldown_minutes_after_loss}m")

    # --- Stop rules ---
    pip_size = float(profile.pip_size)
    if r.require_stop and candidate.stop_price is None:
        hard.append("stop is required")
    if candidate.stop_price is not None:
        stop_pips = _pip_distance(candidate.entry_price, candidate.stop_price, pip_size)
        if stop_pips < r.min_stop_pips:
            hard.append(f"stop too tight: {stop_pips:.1f} < min_stop_pips {r.min_stop_pips:.1f}")

        # sanity: stop must be on correct side
        if candidate.side == "buy" and candidate.stop_price >= candidate.entry_price:
            hard.append("invalid stop: BUY stop must be below entry")
        if candidate.side == "sell" and candidate.stop_price <= candidate.entry_price:
            hard.append("invalid stop: SELL stop must be above entry")

    # --- Size cap ---
    if candidate.size_lots is not None and candidate.size_lots > r.max_lots:
        hard.append(f"size too large: {candidate.size_lots} > max_lots {r.max_lots}")

    # --- Optional: risk-per-trade sizing suggestion ---
    sizing = RiskSizing()
    if r.risk_per_trade_pct is not None:
        if candidate.stop_price is None:
            warn.append("risk_per_trade_pct set, but stop is missing; cannot size")
        elif account_equity is None:
            warn.append("risk_per_trade_pct set, but account_equity not provided; cannot size")
        elif pip_value_per_lot is None:
            warn.append("risk_per_trade_pct set, but pip_value_per_lot not provided; cannot size")
        else:
            risk_amount = float(account_equity) * (float(r.risk_per_trade_pct) / 100.0)
            risk_pips = _pip_distance(candidate.entry_price, candidate.stop_price, pip_size)
            if risk_pips <= 0:
                warn.append("risk_per_trade sizing: invalid risk_pips")
            else:
                suggested = risk_amount / (risk_pips * float(pip_value_per_lot))
                sizing = RiskSizing(
                    suggested_size_lots=float(suggested),
                    risk_per_trade_pct=float(r.risk_per_trade_pct),
                    risk_amount=risk_amount,
                    risk_pips=float(risk_pips),
                    pip_value_per_lot=float(pip_value_per_lot),
                )

    allow = len(hard) == 0
    return RiskDecision(allow=allow, hard_reasons=hard, warnings=warn, sizing=sizing)

