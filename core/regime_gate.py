"""Regime gate for Trial #10: operator-style activity filter.

Encodes the "when should I even be active?" layer that a human operator
applies naturally. Three components:

1. **London-sell veto** – hard block on sells during London hours (ET).
2. **Time/side multiplier** – operator-style sizing tiers by ET hour:
   - buy boost (1.35x): buys during best windows
   - buy base (0.65x): all other buys
   - sell base (0.35x): all non-veto sells
3. **Chop auto-pause** – pauses a side for N minutes after clustered
   stops/sloppy conditions, resuming only when the timer expires AND M5
   is not weak.
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RegimeGateResult:
    allowed: bool = True
    multiplier: float = 1.0
    label: str = "BUY_BASE_HOUR"
    reason: str = ""
    side: str = ""
    hour_et: Optional[int] = None
    m5_bucket: str = "normal"
    chop_pause_active: bool = False


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_regime_gate(
    *,
    hour_et: int,
    side: str,
    m5_bucket: str,
    enabled: bool = True,
    # London sell veto
    london_sell_veto: bool = True,
    london_start_hour_et: int = 3,
    london_end_hour_et: int = 12,
    # Multiplier tiers
    boost_hours_et: tuple[int, ...] = (6, 7, 12, 13, 14, 15),
    boost_multiplier: float = 1.35,
    buy_base_multiplier: float = 0.65,
    sell_base_multiplier: float = 0.35,
    worst_hours_et: tuple[int, ...] = (3, 4, 8, 9, 10, 11),
    worst_multiplier: float = 0.35,
    weak_regime_multiplier: float = 0.5,
    # Chop pause (pre-evaluated flag)
    chop_paused: bool = False,
    chop_pause_reason: str = "",
) -> RegimeGateResult:
    """Evaluate regime gate and return allow/block + multiplier."""
    side = str(side or "").lower()
    m5_bucket = str(m5_bucket or "normal").lower()
    if not enabled:
        return RegimeGateResult(
            allowed=True, multiplier=1.0,
            label="DISABLED", reason="Regime gate disabled",
            side=side, hour_et=hour_et, m5_bucket=m5_bucket,
        )

    # --- Hard veto: London sells ---
    if london_sell_veto and side == "sell":
        if london_start_hour_et <= hour_et < london_end_hour_et:
            return RegimeGateResult(
                allowed=False, multiplier=0.0,
                label="VETO_LONDON_SELL",
                reason=f"London sell veto: sells blocked {london_start_hour_et:02d}-{london_end_hour_et:02d} ET",
                side=side, hour_et=hour_et, m5_bucket=m5_bucket,
            )

    # --- Chop auto-pause ---
    if chop_paused:
        return RegimeGateResult(
            allowed=False, multiplier=0.0,
            label="CHOP_PAUSE",
            reason=chop_pause_reason or "Chop pause active",
            side=side,
            hour_et=hour_et,
            m5_bucket=m5_bucket,
            chop_pause_active=True,
        )

    # --- Operator sizing tiers ---
    if hour_et in boost_hours_et and side == "buy":
        return RegimeGateResult(
            allowed=True, multiplier=boost_multiplier,
            label="BUY_BOOST_HOUR",
            reason=f"Boost window: buy at {hour_et:02d} ET ({boost_multiplier}x)",
            side=side, hour_et=hour_et, m5_bucket=m5_bucket,
        )

    if side == "buy":
        return RegimeGateResult(
            allowed=True,
            multiplier=buy_base_multiplier,
            label="BUY_BASE_HOUR",
            reason=f"Buy base hour: {hour_et:02d} ET ({buy_base_multiplier}x)",
            side=side, hour_et=hour_et, m5_bucket=m5_bucket,
        )

    return RegimeGateResult(
        allowed=True,
        multiplier=sell_base_multiplier,
        label="SELL_BASE_HOUR",
        reason=f"Sell base hour: {hour_et:02d} ET ({sell_base_multiplier}x)",
        side=side, hour_et=hour_et, m5_bucket=m5_bucket,
    )


# ---------------------------------------------------------------------------
# Chop auto-pause logic
# ---------------------------------------------------------------------------

@dataclass
class ChopPauseState:
    buy_paused: bool = False
    buy_pause_start_utc: Optional[datetime.datetime] = None
    buy_pause_reason: str = ""
    sell_paused: bool = False
    sell_pause_start_utc: Optional[datetime.datetime] = None
    sell_pause_reason: str = ""


def check_chop_pause(
    *,
    side: str,
    recent_trades: list[dict],
    now_utc: datetime.datetime,
    lookback_trades: int = 5,
    stop_rate_threshold: float = 0.6,
    pause_minutes: int = 45,
    current_pause_start: Optional[datetime.datetime] = None,
    m5_bucket: str = "normal",
) -> tuple[bool, str]:
    """Check whether *side* should be paused due to chop.

    Uses a rolling stop-rate approach: if >= ``stop_rate_threshold`` of the
    last ``lookback_trades`` closed trades (both sides) were initial stops,
    pause the requested side.  Default 3/5 = 60%.

    Parameters
    ----------
    recent_trades : list[dict]
        Each dict should have at minimum:
        - ``exit_reason``: e.g. ``"initial_stop"``
        - ``close_time_utc``: ``datetime.datetime`` or ISO string

    Returns
    -------
    (is_paused, reason)
    """
    side = str(side).lower()

    # Check if existing pause should be lifted
    if current_pause_start is not None:
        elapsed = (now_utc - current_pause_start).total_seconds() / 60.0
        if elapsed >= pause_minutes and m5_bucket != "weak":
            return False, ""
        remaining = max(0, pause_minutes - elapsed)
        if elapsed < pause_minutes:
            return True, f"Chop pause: {remaining:.0f}m remaining"
        return True, f"Chop pause: timer done but M5 still weak"

    # Sort recent trades by close time (newest first) and take last N
    sorted_trades: list[dict] = []
    for t in recent_trades:
        close_time = t.get("close_time_utc")
        if close_time is None:
            continue
        if isinstance(close_time, str):
            try:
                close_time = datetime.datetime.fromisoformat(close_time)
            except (ValueError, TypeError):
                continue
        if close_time.tzinfo is None:
            close_time = close_time.replace(tzinfo=datetime.timezone.utc)
        sorted_trades.append({**t, "_close_dt": close_time})

    sorted_trades.sort(key=lambda x: x["_close_dt"], reverse=True)
    last_n = sorted_trades[:lookback_trades]

    if len(last_n) < lookback_trades:
        return False, ""

    stop_count = sum(1 for t in last_n if t.get("exit_reason") == "initial_stop")
    stop_rate = stop_count / len(last_n)

    if stop_rate > stop_rate_threshold:
        return True, f"Chop pause triggered: {stop_count}/{len(last_n)} stops ({stop_rate:.0%})"

    return False, ""


def regime_gate_snapshot(result: RegimeGateResult) -> dict:
    """Serialise a RegimeGateResult for dashboard / logging."""
    return {
        "allowed": result.allowed,
        "multiplier": result.multiplier,
        "label": result.label,
        "reason": result.reason,
        "side": result.side,
        "hour_et": result.hour_et,
        "m5_bucket": result.m5_bucket,
        "chop_pause_active": result.chop_pause_active,
    }
