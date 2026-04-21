from __future__ import annotations

import json
import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from api import suggestion_tracker


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _parse_iso(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_avg(values: list[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _count_win_rate(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        label = str(row.get(key) or "").strip().lower()
        if not label:
            continue
        grouped[label].append(row)
    out: dict[str, float] = {}
    for label, items in grouped.items():
        closed = [r for r in items if r.get("closed_at")]
        if not closed:
            continue
        wins = sum(1 for r in closed if str(r.get("win_loss") or "") == "win")
        out[label] = wins / max(len(closed), 1)
    return out


def _autonomous_marker_sql() -> str:
    return '%"autonomous": true%'


def load_autonomous_suggestions(db_path: Path) -> list[dict[str, Any]]:
    suggestion_tracker.init_db(db_path)
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            SELECT *
            FROM ai_suggestions
            WHERE placed_order_json LIKE ?
            ORDER BY created_utc DESC
            """,
            (_autonomous_marker_sql(),),
        )
        return [suggestion_tracker._deserialize_row(dict(r)) for r in cur.fetchall()]


def load_autonomous_closed_outcomes(
    db_path: Path,
    *,
    scratch_threshold_pips: float = 1.0,
) -> list[dict[str, Any]]:
    rows = load_autonomous_suggestions(db_path)
    closed = [row for row in rows if row.get("closed_at") and row.get("pips") is not None]
    closed.sort(key=lambda row: str(row.get("closed_at") or row.get("created_utc") or ""))
    for row in closed:
        pips = _safe_float(row.get("pips")) or 0.0
        if pips >= scratch_threshold_pips:
            streak_outcome = "win"
        elif pips <= -scratch_threshold_pips:
            streak_outcome = "loss"
        else:
            streak_outcome = "scratch"
        row["streak_outcome"] = streak_outcome
    return closed


def get_last_terminal_event_utc(db_path: Path) -> str | None:
    suggestion_tracker.init_db(db_path)
    latest: datetime | None = None
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            SELECT created_utc, action_utc, filled_at, closed_at
            FROM ai_suggestions
            WHERE placed_order_json LIKE ?
            ORDER BY created_utc DESC
            """,
            (_autonomous_marker_sql(),),
        )
        for row in cur.fetchall():
            for key in ("created_utc", "action_utc", "filled_at", "closed_at"):
                ts = _parse_iso(row[key])
                if ts and (latest is None or ts > latest):
                    latest = ts
        cur = conn.execute(
            """
            SELECT created_utc
            FROM ai_reflections
            WHERE autonomous = 1
            ORDER BY created_utc DESC
            LIMIT 50
            """
        )
        for row in cur.fetchall():
            ts = _parse_iso(row["created_utc"])
            if ts and (latest is None or ts > latest):
                latest = ts
    return latest.isoformat() if latest else None


def _load_trade_map(assistant_db_path: Path, trade_ids: list[str]) -> dict[str, dict[str, Any]]:
    trade_ids = [str(tid) for tid in trade_ids if tid]
    if not trade_ids or not assistant_db_path.exists():
        return {}
    placeholders = ",".join(["?"] * len(trade_ids))
    with _connect(assistant_db_path) as conn:
        cur = conn.execute(
            f"""
            SELECT trade_id, timestamp_utc, exit_timestamp_utc, entry_price, exit_price, side,
                   profit, pips, max_adverse_pips, max_favorable_pips, mae_mfe_estimated
            FROM trades
            WHERE trade_id IN ({placeholders})
            """,
            trade_ids,
        )
        return {str(r["trade_id"]): dict(r) for r in cur.fetchall()}


def reconcile_closed_outcomes(
    *,
    suggestions_db_path: Path,
    assistant_db_path: Path,
) -> int:
    """Backfill missing closed outcomes on autonomous suggestion rows from assistant.db."""
    rows = load_autonomous_suggestions(suggestions_db_path)
    pending = [
        row for row in rows
        if row.get("trade_id") and row.get("filled_at") and not row.get("closed_at")
    ]
    if not pending:
        return 0

    trade_map = _load_trade_map(
        assistant_db_path,
        [str(row.get("trade_id") or "") for row in pending],
    )
    updated = 0
    for row in pending:
        trade = trade_map.get(str(row.get("trade_id") or ""))
        if not trade:
            continue
        closed_at = trade.get("exit_timestamp_utc")
        exit_price = _safe_float(trade.get("exit_price"))
        pnl = _safe_float(trade.get("profit"))
        pips = _safe_float(trade.get("pips"))
        if not closed_at or exit_price is None or pnl is None or pips is None:
            continue
        if suggestion_tracker.mark_closed_by_suggestion_id(
            suggestions_db_path,
            suggestion_id=str(row.get("suggestion_id") or ""),
            exit_price=float(exit_price),
            pnl=float(pnl),
            pips=float(pips),
            closed_at=str(closed_at),
        ):
            updated += 1
    return updated


def _hold_minutes(row: dict[str, Any], trade: dict[str, Any] | None) -> float | None:
    if trade:
        opened = _parse_iso(trade.get("timestamp_utc"))
        closed = _parse_iso(trade.get("exit_timestamp_utc"))
        if opened and closed:
            return max(0.0, (closed - opened).total_seconds() / 60.0)
    opened = _parse_iso(row.get("filled_at"))
    closed = _parse_iso(row.get("closed_at"))
    if opened and closed:
        return max(0.0, (closed - opened).total_seconds() / 60.0)
    return None


def _time_to_fill_seconds(row: dict[str, Any]) -> float | None:
    placed = _parse_iso(row.get("action_utc"))
    filled = _parse_iso(row.get("filled_at"))
    if not placed or not filled:
        return None
    return max(0.0, (filled - placed).total_seconds())


def _fill_vs_requested_pips(row: dict[str, Any]) -> float | None:
    placed = row.get("placed_order") or {}
    if str(placed.get("order_type") or "").lower() != "market":
        return None
    req = _safe_float(row.get("requested_price"))
    fill = _safe_float(row.get("fill_price"))
    side = str(row.get("side") or "").lower()
    if req is None or fill is None or side not in {"buy", "sell"}:
        return None
    if side == "buy":
        return (req - fill) / 0.01
    return (fill - req) / 0.01


def _window_rows(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    if key == "rolling_20":
        return rows[:20]
    if key == "rolling_50":
        return rows[:50]
    now = datetime.now(timezone.utc)
    if key == "today":
        today = now.strftime("%Y-%m-%d")
        return [row for row in rows if str(row.get("closed_at") or "").startswith(today)]
    if key == "week":
        cutoff = now - timedelta(days=7)
        return [row for row in rows if (_parse_iso(row.get("closed_at")) or now) >= cutoff]
    return rows


def _compute_window_stats(
    rows: list[dict[str, Any]],
    thesis_checks_by_trade: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    placed_rows = [row for row in rows if str(row.get("action") or "") == "placed"]
    limit_rows = [
        row for row in placed_rows
        if str((row.get("placed_order") or {}).get("order_type") or "").lower() == "limit"
    ]
    market_rows = [
        row for row in placed_rows
        if str((row.get("placed_order") or {}).get("order_type") or "").lower() == "market"
    ]
    closed_rows = [row for row in rows if row.get("closed_at")]
    wins = [row for row in closed_rows if str(row.get("win_loss") or "") == "win"]
    losses = [row for row in closed_rows if str(row.get("win_loss") or "") == "loss"]
    gross_win = sum(max(_safe_float(row.get("pnl")) or 0.0, 0.0) for row in closed_rows)
    gross_loss = abs(sum(min(_safe_float(row.get("pnl")) or 0.0, 0.0) for row in closed_rows))
    intervention_checks = 0
    total_checks = 0
    for row in rows:
        checks = thesis_checks_by_trade.get(str(row.get("trade_id") or ""), [])
        total_checks += len(checks)
        intervention_checks += sum(1 for chk in checks if str(chk.get("action") or "") != "hold")
    return {
        "trade_count": len(rows),
        "closed_count": len(closed_rows),
        "win_rate": (len(wins) / len(closed_rows)) if closed_rows else None,
        "avg_win_pips": _safe_avg([_safe_float(row.get("pips")) for row in wins]),
        "avg_loss_pips": _safe_avg([_safe_float(row.get("pips")) for row in losses]),
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else (None if gross_win <= 0 else 999.0),
        "avg_hold_minutes": _safe_avg([_safe_float(row.get("hold_minutes")) for row in closed_rows]),
        "fill_rate_limits": (
            len([row for row in limit_rows if str(row.get("outcome_status") or "") == "filled"]) / len(limit_rows)
            if limit_rows else None
        ),
        "avg_time_to_fill_sec": _safe_avg([_time_to_fill_seconds(row) for row in limit_rows]),
        "avg_fill_vs_requested_pips": _safe_avg([_fill_vs_requested_pips(row) for row in market_rows]),
        "thesis_intervention_rate": (intervention_checks / total_checks) if total_checks else None,
        "avg_mae_pips": _safe_avg([_safe_float(row.get("max_adverse_pips")) for row in closed_rows]),
        "avg_mfe_pips": _safe_avg([_safe_float(row.get("max_favorable_pips")) for row in closed_rows]),
        "win_rate_by_confidence_json": json.dumps(_count_win_rate(closed_rows, "confidence")),
        "win_rate_by_side_json": json.dumps(_count_win_rate(closed_rows, "side")),
        "win_rate_by_session_json": json.dumps(_count_win_rate(closed_rows, "session_label")),
        "prompt_version_breakdown_json": json.dumps(_prompt_version_breakdown(rows)),
    }


def _prompt_version_breakdown(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0.0, "wins": 0.0})
    for row in rows:
        version = str(row.get("prompt_version") or "unknown").strip() or "unknown"
        grouped[version]["count"] += 1.0
        if str(row.get("win_loss") or "") == "win":
            grouped[version]["wins"] += 1.0
    out: dict[str, Any] = {}
    for version, payload in grouped.items():
        count = int(payload["count"])
        wins = int(payload["wins"])
        out[version] = {"count": count, "win_rate": (wins / count) if count else None}
    return out


def recompute_performance_stats(
    *,
    profile: str,
    suggestions_db_path: Path,
    assistant_db_path: Path,
) -> dict[str, dict[str, Any]]:
    rows = load_autonomous_suggestions(suggestions_db_path)
    if not rows:
        for key in ("rolling_20", "rolling_50", "today", "week"):
            suggestion_tracker.upsert_performance_stats(
                suggestions_db_path,
                profile=profile,
                stats_key=key,
                values={"updated_utc": _now_iso()},
            )
        return {}

    trade_map = _load_trade_map(
        assistant_db_path,
        [str(row.get("trade_id") or "") for row in rows if row.get("trade_id")],
    )
    thesis_checks = suggestion_tracker.get_recent_thesis_checks(suggestions_db_path, limit=500)
    thesis_checks_by_trade: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for check in thesis_checks:
        trade_id = str(check.get("trade_id") or "")
        if trade_id:
            thesis_checks_by_trade[trade_id].append(check)

    merged: list[dict[str, Any]] = []
    for row in rows:
        trade = trade_map.get(str(row.get("trade_id") or ""))
        snap = row.get("market_snapshot") or {}
        session = snap.get("session") or {}
        row_out = dict(row)
        row_out["session_label"] = (
            str(session.get("overlap") or "").strip().lower()
            or ",".join(str(x).strip().lower() for x in (session.get("active_sessions") or []) if x)
        )
        row_out["hold_minutes"] = _hold_minutes(row, trade)
        row_out["max_adverse_pips"] = _safe_float((trade or {}).get("max_adverse_pips"))
        row_out["max_favorable_pips"] = _safe_float((trade or {}).get("max_favorable_pips"))
        row_out["mae_mfe_estimated"] = int((trade or {}).get("mae_mfe_estimated") or 0) if trade else None
        merged.append(row_out)

    stats_rows: dict[str, dict[str, Any]] = {}
    for key in ("rolling_20", "rolling_50", "today", "week"):
        win_rows = _window_rows(merged, key)
        values = _compute_window_stats(win_rows, thesis_checks_by_trade)
        values["updated_utc"] = _now_iso()
        suggestion_tracker.upsert_performance_stats(
            suggestions_db_path,
            profile=profile,
            stats_key=key,
            values=values,
        )
        stats_rows[key] = values
    return stats_rows


def get_materialized_stats(db_path: Path) -> dict[str, dict[str, Any]]:
    return suggestion_tracker.get_performance_stats(db_path)


def build_performance_memory_block(
    db_path: Path,
    *,
    risk_regime: dict[str, Any] | None = None,
) -> str:
    rows = get_materialized_stats(db_path)
    stats = rows.get("rolling_20") or {}
    closed_count = int(stats.get("closed_count") or 0)
    if closed_count < 5:
        return ""

    lines = ["=== STRUCTURED PERFORMANCE MEMORY ==="]
    win_rate = stats.get("win_rate")
    pf = stats.get("profit_factor")
    avg_win = stats.get("avg_win_pips")
    avg_loss = stats.get("avg_loss_pips")
    avg_hold = stats.get("avg_hold_minutes")
    pf_text = f"{pf:.2f}" if isinstance(pf, (int, float)) else "n/a"
    lines.append(
        f"Last {closed_count} autonomous closes: "
        f"{(win_rate or 0.0) * 100:.0f}% WR | PF {pf_text}"
    )
    core_tail = []
    if isinstance(avg_win, (int, float)):
        core_tail.append(f"avg win {avg_win:+.1f}p")
    if isinstance(avg_loss, (int, float)):
        core_tail.append(f"avg loss {avg_loss:+.1f}p")
    if isinstance(avg_hold, (int, float)):
        core_tail.append(f"avg hold {avg_hold:.0f}m")
    if core_tail:
        lines[-1] = lines[-1] + " | " + " | ".join(core_tail)
    fill_rate_limits = stats.get("fill_rate_limits")
    if isinstance(fill_rate_limits, (int, float)):
        lines.append(f"Limit fill rate: {fill_rate_limits * 100:.0f}%")
    fill_vs_req = stats.get("avg_fill_vs_requested_pips")
    if isinstance(fill_vs_req, (int, float)):
        lines.append(f"Avg fill vs requested (market only): {fill_vs_req:+.2f}p")
    conf_json = suggestion_tracker._json_obj(stats.get("win_rate_by_confidence_json"))
    if conf_json:
        parts = [f"{k}={(float(v) * 100):.0f}%" for k, v in sorted(conf_json.items())]
        lines.append("WR by confidence: " + ", ".join(parts))
    session_json = suggestion_tracker._json_obj(stats.get("win_rate_by_session_json"))
    if session_json:
        pairs = sorted(((k, float(v)) for k, v in session_json.items()), key=lambda item: item[1], reverse=True)
        if len(pairs) >= 2 and pairs[0][1] - pairs[-1][1] >= 0.15:
            lines.append(
                f"Session edge: best={pairs[0][0]} ({pairs[0][1] * 100:.0f}%), "
                f"worst={pairs[-1][0]} ({pairs[-1][1] * 100:.0f}%)"
            )
    mae = stats.get("avg_mae_pips")
    mfe = stats.get("avg_mfe_pips")
    if isinstance(mae, (int, float)) and isinstance(mfe, (int, float)):
        lines.append(f"Avg MAE {mae:.1f}p | Avg MFE {mfe:.1f}p")
    if risk_regime and str(risk_regime.get("label") or "normal") != "normal":
        lines.append(
            f"ACTIVE RISK REGIME: {str(risk_regime.get('label') or '').upper()} | "
            f"lots {float(risk_regime.get('risk_multiplier') or 1.0):.2f}x | "
            f"min confidence {risk_regime.get('effective_min_confidence')}"
        )

    # Keep this block small: cap roughly around 180 words and drop lower-priority lines first.
    priority = lines[:]
    while len(" ".join(priority).split()) > 180 and len(priority) > 3:
        for idx, marker in enumerate(priority):
            if marker.startswith("Session edge:"):
                priority.pop(idx)
                break
        else:
            for idx, marker in enumerate(priority):
                if marker.startswith("WR by confidence:"):
                    priority.pop(idx)
                    break
            else:
                for idx, marker in enumerate(priority):
                    if marker.startswith("Avg MAE"):
                        priority.pop(idx)
                        break
                else:
                    priority.pop()
    return "\n".join(priority)
