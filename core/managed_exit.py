from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _tier_rank(tier: str) -> int:
    t = str(tier or "").lower()
    if t == "medium":
        return 1
    if t == "high":
        return 2
    if t == "critical":
        return 3
    return 0


def _tier_at_or_above(current_tier: str, threshold_tier: str) -> bool:
    return _tier_rank(current_tier) >= _tier_rank(threshold_tier)


def _position_lots_from_live_pos(pos: Any) -> float:
    try:
        if isinstance(pos, dict):
            units = float(pos.get("currentUnits") or pos.get("initialUnits") or 0)
            return abs(units) / 100_000.0
        return float(getattr(pos, "volume", 0.0) or 0.0)
    except Exception:
        return 0.0


def apply_trial7_managed_exit_for_position(
    *,
    adapter,
    profile,
    store,
    policy,
    position_id: int,
    live_position: Any,
    trade_row: dict,
    tick,
    pip_size: float,
) -> None:
    """Apply managed-exit controls for Trial #7 trades in medium/high/critical tiers."""
    try:
        if not bool(getattr(policy, "use_reversal_risk_score", False)):
            return

        current_tier = str(trade_row.get("reversal_risk_tier") or "").lower()
        if current_tier not in ("medium", "high", "critical"):
            return

        threshold = str(getattr(policy, "rr_use_managed_exit_at", "high")).lower()
        if threshold not in ("medium", "high", "critical"):
            threshold = "high"
        if not _tier_at_or_above(current_tier, threshold):
            return

        side = str(trade_row.get("side") or "").lower()
        if side not in ("buy", "sell"):
            return
        entry = float(trade_row.get("entry_price") or 0.0)
        if entry <= 0:
            return

        trade_id = str(trade_row.get("trade_id") or "")
        if not trade_id:
            return

        # PnL at executable side of spread
        check_price = float(tick.bid) if side == "buy" else float(tick.ask)
        current_pips = (check_price - entry) / float(pip_size) if side == "buy" else (entry - check_price) / float(pip_size)

        now_utc = datetime.now(timezone.utc)
        age_min = 0.0
        ts = trade_row.get("timestamp_utc")
        if ts:
            try:
                t0 = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                age_min = (now_utc - t0).total_seconds() / 60.0
            except Exception:
                age_min = 0.0

        hard_sl_pips = max(0.1, float(getattr(policy, "rr_managed_exit_hard_sl_pips", 72.0)))
        max_hold_underwater = max(1.0, float(getattr(policy, "rr_managed_exit_max_hold_underwater_min", 30.0)))
        trail_activation = max(0.1, float(getattr(policy, "rr_managed_exit_trail_activation_pips", 4.0)))
        trail_distance = max(0.1, float(getattr(policy, "rr_managed_exit_trail_distance_pips", 2.5)))

        # 1) Disaster hard-stop close
        if current_pips <= -hard_sl_pips:
            lots = _position_lots_from_live_pos(live_position)
            if lots > 0:
                position_type = 1 if side == "sell" else 0
                adapter.close_position(
                    ticket=int(position_id),
                    symbol=profile.symbol,
                    volume=float(lots),
                    position_type=position_type,
                )
                store.update_trade(trade_id, {
                    "notes": f"{trade_row.get('notes') or ''} | managed_exit_hard_sl@{hard_sl_pips:.1f}p".strip(" |"),
                    "managed_exit_last_action": f"hard_sl_close:{hard_sl_pips:.1f}p",
                })
                print(
                    f"[{profile.profile_name}] managed_exit close: trade={trade_id} tier={current_tier} "
                    f"reason=hard_sl current={current_pips:.1f}p limit=-{hard_sl_pips:.1f}p"
                )
            return

        # 2) Time-based underwater close
        if age_min >= max_hold_underwater and current_pips < 0:
            lots = _position_lots_from_live_pos(live_position)
            if lots > 0:
                position_type = 1 if side == "sell" else 0
                adapter.close_position(
                    ticket=int(position_id),
                    symbol=profile.symbol,
                    volume=float(lots),
                    position_type=position_type,
                )
                store.update_trade(trade_id, {
                    "notes": f"{trade_row.get('notes') or ''} | managed_exit_time_close@{age_min:.1f}m".strip(" |"),
                    "managed_exit_last_action": f"underwater_time_close:{age_min:.1f}m",
                })
                print(
                    f"[{profile.profile_name}] managed_exit close: trade={trade_id} tier={current_tier} "
                    f"reason=underwater_time age={age_min:.1f}m pips={current_pips:.1f}"
                )
            return

        # 3) Trailing SL activation
        if current_pips >= trail_activation:
            if side == "buy":
                new_sl = float(tick.bid) - trail_distance * float(pip_size)
            else:
                new_sl = float(tick.ask) + trail_distance * float(pip_size)

            prev = trade_row.get("managed_exit_sl_price")
            if prev is None:
                prev = trade_row.get("breakeven_sl_price")
            if prev is None:
                prev = trade_row.get("stop_price")
            try:
                prev_f = float(prev) if prev is not None else None
            except Exception:
                prev_f = None

            should_update = False
            if prev_f is None:
                should_update = True
            elif side == "buy" and new_sl > prev_f + (0.1 * float(pip_size)):
                should_update = True
            elif side == "sell" and new_sl < prev_f - (0.1 * float(pip_size)):
                should_update = True

            if should_update:
                adapter.update_position_stop_loss(int(position_id), profile.symbol, round(float(new_sl), 3))
                store.update_trade(trade_id, {
                    "managed_exit_sl_price": round(float(new_sl), 5),
                    "managed_exit_active": 1,
                    "managed_exit_last_action": f"trail_sl:{round(float(new_sl), 5)}",
                })
                print(
                    f"[{profile.profile_name}] managed_exit trail: trade={trade_id} tier={current_tier} "
                    f"pips={current_pips:.1f} sl->{new_sl:.3f}"
                )
    except Exception as e:
        print(f"[{profile.profile_name}] managed_exit error trade={trade_row.get('trade_id')}: {e}")
