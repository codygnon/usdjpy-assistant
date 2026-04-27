from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from api.fillmore_llm_guard import FillmoreLLMCircuitOpenError, run_guarded_fillmore_llm_call


THESIS_MONITOR_MODEL = "gpt-5.4-mini"
REFLECTION_MODEL = "gpt-5.4-mini"
THESIS_MONITOR_MIN_OPEN_AGE_SEC = 120
THESIS_MONITOR_INTERVAL_SEC = 180


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_iso(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _current_lots_from_position(position: Any) -> float:
    if isinstance(position, dict):
        units = _safe_float(position.get("currentUnits"))
        if units is not None:
            return abs(units) / 100_000.0
        volume = _safe_float(position.get("volume"))
        if volume is not None:
            return abs(volume)
        long_units = _safe_float(position.get("long", {}).get("units")) if isinstance(position.get("long"), dict) else None
        short_units = _safe_float(position.get("short", {}).get("units")) if isinstance(position.get("short"), dict) else None
        if long_units is not None and long_units != 0:
            return abs(long_units) / 100_000.0
        if short_units is not None and short_units != 0:
            return abs(short_units) / 100_000.0
        return 0.0
    volume = _safe_float(getattr(position, "volume", None))
    return abs(volume) if volume is not None else 0.0


def _current_stop_from_position(position: Any) -> Optional[float]:
    if isinstance(position, dict):
        stop_keys = (
            position.get("stop_loss"),
            position.get("stopLoss"),
            position.get("sl"),
        )
        for value in stop_keys:
            stop = _safe_float(value)
            if stop is not None and stop > 0:
                return stop
        slo = position.get("stopLossOrder")
        if isinstance(slo, dict):
            stop = _safe_float(slo.get("price"))
            if stop is not None and stop > 0:
                return stop
        trailing = position.get("trailingStopLossOrder")
        if isinstance(trailing, dict):
            stop = _safe_float(trailing.get("price"))
            if stop is not None and stop > 0:
                return stop
        return None
    for attr in ("stop_loss", "sl", "stopLoss"):
        stop = _safe_float(getattr(position, attr, None))
        if stop is not None and stop > 0:
            return stop
    return None


def _current_unrealized_usd(position: Any) -> Optional[float]:
    if isinstance(position, dict):
        for key in ("unrealizedPL", "unrealized_pl", "profit", "pl"):
            pnl = _safe_float(position.get(key))
            if pnl is not None:
                return pnl
        return None
    for attr in ("unrealized_pl", "profit", "pl"):
        pnl = _safe_float(getattr(position, attr, None))
        if pnl is not None:
            return pnl
    return None


def _normalize_reflection_tags(raw: dict[str, Any], suggestion_row: dict[str, Any]) -> dict[str, Any]:
    def _match(value: Any, valid: set[str], default: str | None = None) -> str | None:
        text = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
        if not text:
            return default
        if text in valid:
            return text
        for item in valid:
            if item in text or text in item:
                return item
        return default

    snap = suggestion_row.get("market_snapshot") or {}
    session = snap.get("session") or {}
    session_label = str(session.get("overlap") or "").strip().lower()
    if "tokyo" in session_label:
        inferred_session = "asian"
    elif "london" in session_label and ("new york" in session_label or "ny" in session_label):
        inferred_session = "overlap"
    elif "london" in session_label:
        inferred_session = "london"
    elif "new york" in session_label or "ny" in session_label:
        inferred_session = "ny"
    else:
        active = ",".join(str(x).lower() for x in (session.get("active_sessions") or []))
        if "tokyo" in active:
            inferred_session = "asian"
        elif "london" in active:
            inferred_session = "london"
        elif "new york" in active or "ny" in active:
            inferred_session = "ny"
        else:
            inferred_session = "unknown"

    tech = snap.get("technicals") or {}
    h1 = tech.get("H1") or {}
    regime_guess = _match(h1.get("regime"), {"trending", "ranging", "mixed"}, None)

    return {
        "primary_error_category": _match(
            raw.get("primary_error_category"),
            {"timing", "direction", "sizing", "exit_management", "regime_mismatch", "event_risk", "none"},
            None,
        ),
        "primary_strength_category": _match(
            raw.get("primary_strength_category"),
            {"entry_precision", "thesis_accuracy", "exit_discipline", "risk_management", "none"},
            None,
        ),
        "regime_at_entry": _match(raw.get("regime_at_entry"), {"trending", "ranging", "mixed"}, regime_guess),
        "session_at_entry": _match(raw.get("session_at_entry"), {"asian", "london", "ny", "overlap", "unknown"}, inferred_session),
        "lesson": str(raw.get("lesson") or "").strip()[:500] or None,
    }


def _approx_usd_profit_usdjpy(*, pips: float, lots: float, mid: float, pip_size: float) -> float:
    if mid <= 0 or lots <= 0:
        return 0.0
    return float(pips * pip_size * 100_000.0 * lots / mid)


def _effective_stop(trade_row: dict[str, Any], position: Any) -> Optional[float]:
    for candidate in (
        _current_stop_from_position(position),
        _safe_float(trade_row.get("breakeven_sl_price")),
        _safe_float(trade_row.get("stop_price")),
    ):
        if candidate is not None and candidate > 0:
            return candidate
    return None


def _recent_checks_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No prior thesis-monitor checks for this trade."
    lines: list[str] = []
    for row in rows[:2]:
        action = str(row.get("action") or "").strip()
        reason = str(row.get("reason") or "").strip()
        conf = str(row.get("confidence") or "").strip()
        created = str(row.get("created_utc") or "")[:16]
        detail = f"{created} {action}"
        if conf:
            detail += f" ({conf})"
        if reason:
            detail += f": {reason}"
        lines.append(detail)
    return "\n".join(lines)


def _market_snapshot_text(suggestion_row: dict[str, Any]) -> str:
    snap = suggestion_row.get("market_snapshot") or {}
    session = snap.get("session") or {}
    macro = snap.get("macro_bias") or {}
    vol = snap.get("volatility") or {}
    return (
        f"entry_session={session.get('overlap') or session.get('active_sessions')}, "
        f"entry_bias={macro.get('combined_bias')}, entry_vol={vol.get('label')}"
    )


def evaluate_trade_thesis(
    *,
    profile: Any,
    profile_name: str,
    trade_row: dict[str, Any],
    position: Any,
    tick: Any,
    db_path: Path,
    check_reason: str | None = None,
    custom_exit: bool = False,
) -> Optional[dict[str, Any]]:
    """Ask Fillmore whether an open trade's thesis is still intact."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None

    from api import suggestion_tracker
    from api.ai_trading_chat import build_trading_context, resolve_ai_suggest_model
    from api.autonomous_fillmore import _extract_json_object
    from api.prompt_builder import PromptBuilder
    import json as _json
    import openai

    trade_id = str(trade_row.get("trade_id") or "").strip()
    order_id = str(trade_row.get("mt5_order_id") or "").strip()
    suggestion_row = suggestion_tracker.resolve_suggestion_for_trade(
        db_path,
        trade_id=trade_id or None,
        oanda_order_id=order_id or None,
    )
    if not suggestion_row:
        return None

    rationale = str(suggestion_row.get("rationale") or "").strip()
    if not rationale:
        return None
    exit_plan = str(suggestion_row.get("exit_plan") or "default").strip()
    custom_exit_plan = suggestion_row.get("custom_exit_plan")
    if not isinstance(custom_exit_plan, dict):
        try:
            raw_cfg = trade_row.get("config_json")
            cfg = _json.loads(str(raw_cfg)) if raw_cfg else {}
            custom_exit_plan = cfg.get("custom_exit_plan") if isinstance(cfg, dict) else {}
        except Exception:
            custom_exit_plan = {}
    if not isinstance(custom_exit_plan, dict):
        custom_exit_plan = {}

    ctx = build_trading_context(profile, profile_name)
    model = resolve_ai_suggest_model(THESIS_MONITOR_MODEL)
    assembly_builder = PromptBuilder.for_thesis_monitor(
        profile=profile,
        profile_name=profile_name,
        ctx=ctx,
        model=model,
    )

    side = str(trade_row.get("side") or suggestion_row.get("side") or "buy").lower()
    entry = _safe_float(trade_row.get("entry_price")) or _safe_float(suggestion_row.get("fill_price")) or 0.0
    pip_size = float(getattr(profile, "pip_size", 0.01) or 0.01)
    mid = (float(tick.bid) + float(tick.ask)) / 2.0
    current_lots = _current_lots_from_position(position)
    effective_stop = _effective_stop(trade_row, position)
    target_price = _safe_float(trade_row.get("target_price")) or _safe_float(suggestion_row.get("tp"))
    tp1_done = bool(trade_row.get("tp1_partial_done") or 0)
    be_applied = bool(trade_row.get("breakeven_applied") or 0)
    current_pips = ((mid - entry) / pip_size) if side == "buy" else ((entry - mid) / pip_size)
    current_pnl = _current_unrealized_usd(position)
    if current_pnl is None:
        current_pnl = _approx_usd_profit_usdjpy(
            pips=float(current_pips),
            lots=float(current_lots or _safe_float(trade_row.get("size_lots")) or 0.0),
            mid=mid,
            pip_size=pip_size,
        )
    opened_at = _parse_iso(trade_row.get("timestamp_utc"))
    open_age_sec = max(0.0, (_now_utc() - opened_at).total_seconds()) if opened_at else None
    recent_checks = suggestion_tracker.list_thesis_checks(db_path, trade_id=trade_id, limit=2)
    spot = ctx.get("spot_price") if isinstance(ctx, dict) else {}
    spread_pips = _safe_float((spot or {}).get("spread_pips")) if isinstance(spot, dict) else None
    if spread_pips is not None and spread_pips > 0:
        spread_noise_note = (
            f"Execution-friction note: live spread is about {spread_pips:.1f}p. "
            "A trade being red by roughly the spread, or a few pips in noisy conditions, is not by itself thesis failure. "
            "Treat small negative P&L as normal entry friction unless price has structurally accepted beyond the original "
            "invalidation level or the setup's key reclaim/rejection has clearly failed.\n"
        )
    else:
        spread_noise_note = (
            "Execution-friction note: do not treat small negative P&L by itself as thesis failure. "
            "Exit because the level/thesis is structurally invalidated, not merely because the trade is red.\n"
        )

    mode_note = (
        "This trade uses llm_custom_exit. You are the active exit manager, but checks are call-budgeted. "
        "Only request broker action when it is actually useful; otherwise hold with a precise next watch condition.\n"
        if custom_exit else
        "This trade uses a template or broker-side exit. Use the original exit plan as guidance and intervene only when the thesis has materially changed.\n"
    )

    prompt = (
        "You are reviewing an ALREADY-OPEN Fillmore trade.\n"
        "Your job is NOT to find a new setup. Your only job is to decide whether the original thesis is still alive.\n"
        "You may only reduce risk. Never widen risk, add size, reverse the trade, or suggest a new order.\n"
        f"{mode_note}"
        "\n"
        "Allowed actions:\n"
        "- hold\n"
        "- tighten_sl\n"
        "- scale_out\n"
        "- exit_now\n"
        "\n"
        "Return JSON only:\n"
        "{\n"
        '  "action": "hold" | "tighten_sl" | "scale_out" | "exit_now",\n'
        '  "new_sl": number | null,\n'
        '  "scale_out_pct": 25 | 33 | 50 | null,\n'
        '  "exit_state": "thesis_intact" | "watching_invalidation" | "profit_capture" | "thesis_broken",\n'
        '  "invalidation_status": "none" | "threatened" | "confirmed",\n'
        '  "management_intent": "let_work" | "protect_profit" | "reduce_risk" | "kill_trade",\n'
        '  "updated_exit_plan": "short update or null",\n'
        '  "next_watch_condition": "what would trigger the next action",\n'
        '  "reason": "short explanation",\n'
        '  "confidence": "low" | "medium" | "high"\n'
        "}\n"
        "\n"
        f"Original suggestion: side={suggestion_row.get('side')} entry={suggestion_row.get('limit_price')} "
        f"sl={suggestion_row.get('sl')} tp={suggestion_row.get('tp')} lots={suggestion_row.get('lots')} "
        f"exit={suggestion_row.get('exit_strategy')}\n"
        f"Original rationale:\n{rationale}\n"
        f"Original exit plan: {exit_plan}\n"
        f"Structured custom exit plan: {_json.dumps(custom_exit_plan, sort_keys=True) if custom_exit_plan else '{}'}\n"
        "(The exit plan describes the trader's intended mid-trade management logic. Use it as your "
        "primary guide for hold/tighten/scale/exit — follow the plan unless the tape has changed "
        "materially enough to override it. If the plan says 'exit if M3 flips bear' and M3 is now "
        "bearish, that's an exit_now.)\n"
        f"Entry snapshot: {_market_snapshot_text(suggestion_row)}\n"
        f"Current trade state: trade_id={trade_id}, side={side}, entry={entry:.3f}, "
        f"current_mid={mid:.3f}, effective_sl={effective_stop}, target={target_price}, "
        f"remaining_lots={current_lots:.2f}, tp1_done={int(tp1_done)}, be_applied={int(be_applied)}, "
        f"open_age_sec={open_age_sec}, current_pips={current_pips:+.1f}, current_usd={float(current_pnl):+.2f}\n"
        f"Monitor trigger: {check_reason or 'scheduled'}\n"
        "Recent thesis checks:\n"
        + _recent_checks_text(recent_checks)
        + "\n"
        + spread_noise_note
        + "Decision rule: use hold unless the thesis is materially weaker than the original rationale implied. "
        "If you choose tighten_sl, the stop must be closer to price in a risk-reducing direction only. "
        "If you choose scale_out, only use 25, 33, or 50. If the trade is clearly broken, use exit_now."
    )

    assembly = assembly_builder.build(
        user=prompt,
        prompt_version="thesis_monitor_v1",
    )
    client = openai.OpenAI()
    try:
        resp = run_guarded_fillmore_llm_call(
            "thesis_monitor",
            lambda: client.chat.completions.create(
                model=assembly.model,
                messages=[
                    {"role": "system", "content": assembly.system},
                    {"role": "user", "content": assembly.user},
                ],
            ),
        )
    except FillmoreLLMCircuitOpenError:
        return None
    raw = (resp.choices[0].message.content or "").strip()
    json_str, _ = _extract_json_object(raw)
    decision = _json.loads(json_str)

    action = str(decision.get("action") or "hold").strip().lower()
    if action not in {"hold", "tighten_sl", "scale_out", "exit_now"}:
        action = "hold"
    confidence = str(decision.get("confidence") or "low").strip().lower()
    if confidence not in {"low", "medium", "high"}:
        confidence = "low"

    new_sl = _safe_float(decision.get("new_sl"))
    scale_out_pct = _safe_float(decision.get("scale_out_pct"))
    if scale_out_pct not in {25.0, 33.0, 50.0}:
        scale_out_pct = None

    exit_state = str(decision.get("exit_state") or "").strip().lower()
    if exit_state not in {"thesis_intact", "watching_invalidation", "profit_capture", "thesis_broken"}:
        exit_state = "thesis_intact" if action == "hold" else "watching_invalidation"
    invalidation_status = str(decision.get("invalidation_status") or "").strip().lower()
    if invalidation_status not in {"none", "threatened", "confirmed"}:
        invalidation_status = "none"
    management_intent = str(decision.get("management_intent") or "").strip().lower()
    if management_intent not in {"let_work", "protect_profit", "reduce_risk", "kill_trade"}:
        management_intent = "let_work" if action == "hold" else "reduce_risk"

    return {
        "action": action,
        "new_sl": new_sl,
        "scale_out_pct": int(scale_out_pct) if scale_out_pct is not None else None,
        "exit_state": exit_state,
        "invalidation_status": invalidation_status,
        "management_intent": management_intent,
        "updated_exit_plan": str(decision.get("updated_exit_plan") or "").strip() or None,
        "next_watch_condition": str(decision.get("next_watch_condition") or "").strip() or None,
        "reason": str(decision.get("reason") or "").strip(),
        "confidence": confidence,
        "model_used": model,
        "suggestion_id": suggestion_row.get("suggestion_id"),
        "trade_id": trade_id,
    }


def maybe_generate_trade_reflection(
    *,
    profile_name: str,
    trade_row: dict[str, Any],
    db_path: Path,
) -> bool:
    """Create one compact post-trade reflection for a closed Fillmore trade."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return False

    from api import suggestion_tracker
    from api.ai_trading_chat import resolve_ai_suggest_model
    from api.autonomous_fillmore import _extract_json_object
    import json as _json
    import openai

    trade_id = str(trade_row.get("trade_id") or "").strip()
    order_id = str(trade_row.get("mt5_order_id") or "").strip()
    suggestion_row = suggestion_tracker.resolve_suggestion_for_trade(
        db_path,
        trade_id=trade_id or None,
        oanda_order_id=order_id or None,
    )
    if not suggestion_row:
        return False

    suggestion_id = str(suggestion_row.get("suggestion_id") or "").strip()
    rationale = str(suggestion_row.get("rationale") or "").strip()
    if not suggestion_id or not trade_id or not rationale:
        return False
    if suggestion_tracker.reflection_exists(
        db_path,
        profile=profile_name,
        suggestion_id=suggestion_id,
        trade_id=trade_id,
    ):
        return False

    side = str(trade_row.get("side") or suggestion_row.get("side") or "buy").upper()
    entry = _safe_float(trade_row.get("entry_price")) or _safe_float(suggestion_row.get("fill_price")) or _safe_float(suggestion_row.get("limit_price")) or 0.0
    exit_price = _safe_float(trade_row.get("exit_price")) or _safe_float(suggestion_row.get("exit_price")) or 0.0
    pnl = _safe_float(trade_row.get("profit"))
    if pnl is None:
        pnl = _safe_float(trade_row.get("pnl")) or _safe_float(suggestion_row.get("pnl")) or 0.0
    pips = _safe_float(trade_row.get("pips"))
    if pips is None:
        pips = _safe_float(suggestion_row.get("pips")) or 0.0
    exit_reason = str(trade_row.get("exit_reason") or "").strip()
    hold_minutes = None
    opened_at = _parse_iso(trade_row.get("timestamp_utc"))
    closed_at = _parse_iso(trade_row.get("exit_timestamp_utc") or trade_row.get("closed_at"))
    if opened_at and closed_at:
        hold_minutes = max(0.0, (closed_at - opened_at).total_seconds() / 60.0)
    thesis_checks = suggestion_tracker.list_thesis_checks(db_path, trade_id=trade_id, limit=4)
    thesis_text = _recent_checks_text(thesis_checks)
    model = resolve_ai_suggest_model(REFLECTION_MODEL)

    prompt = (
        "Write a terse self-postmortem for this completed Fillmore trade.\n"
        "Return JSON only:\n"
        "{\n"
        '  "what_read_right": "one concise line",\n'
        '  "what_missed": "one concise line",\n'
        '  "primary_error_category": "timing" | "direction" | "sizing" | "exit_management" | "regime_mismatch" | "event_risk" | "none",\n'
        '  "primary_strength_category": "entry_precision" | "thesis_accuracy" | "exit_discipline" | "risk_management" | "none",\n'
        '  "regime_at_entry": "trending" | "ranging" | "mixed",\n'
        '  "session_at_entry": "asian" | "london" | "ny" | "overlap" | "unknown",\n'
        '  "lesson": "one concise line"\n'
        "}\n"
        "\n"
        f"Original suggestion: side={suggestion_row.get('side')} entry={suggestion_row.get('limit_price')} "
        f"sl={suggestion_row.get('sl')} tp={suggestion_row.get('tp')} lots={suggestion_row.get('lots')} exit={suggestion_row.get('exit_strategy')}\n"
        f"Original rationale:\n{rationale}\n"
        f"Entry snapshot: {_market_snapshot_text(suggestion_row)}\n"
        f"Final result: side={side} entry={entry:.3f} exit={exit_price:.3f} pips={float(pips):+.1f} "
        f"usd={float(pnl):+.2f} exit_reason={exit_reason} hold_minutes={hold_minutes}\n"
        f"Recent thesis-monitor checks:\n{thesis_text}\n"
        "Be specific. Focus on what the model read correctly versus what it failed to notice."
    )

    client = openai.OpenAI()
    try:
        resp = run_guarded_fillmore_llm_call(
            "trade_reflection",
            lambda: client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are Fillmore writing a compact post-trade self-review for future autonomous suggestions. "
                            "Be honest, concrete, and concise."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            ),
        )
    except FillmoreLLMCircuitOpenError:
        return False
    raw = (resp.choices[0].message.content or "").strip()
    json_str, _ = _extract_json_object(raw)
    parsed = _json.loads(json_str)
    right = str(parsed.get("what_read_right") or "").strip()
    missed = str(parsed.get("what_missed") or "").strip()
    if not right or not missed:
        return False
    tags = _normalize_reflection_tags(parsed, suggestion_row)

    outcome = "win" if float(pnl) > 0 else "loss" if float(pnl) < 0 else "breakeven"
    summary_text = (
        f"{side} {outcome} ({float(pips):+.1f}p, ${float(pnl):+.2f}, exit={exit_reason or 'closed'})"
    )
    return suggestion_tracker.log_reflection(
        db_path,
        profile=profile_name,
        suggestion_id=suggestion_id,
        trade_id=trade_id,
        model=model,
        what_read_right=right,
        what_missed=missed,
        summary_text=summary_text,
        primary_error_category=tags.get("primary_error_category"),
        primary_strength_category=tags.get("primary_strength_category"),
        regime_at_entry=tags.get("regime_at_entry"),
        session_at_entry=tags.get("session_at_entry"),
        lesson=tags.get("lesson"),
        autonomous=suggestion_tracker.is_autonomous_suggestion_row(suggestion_row),
    )
