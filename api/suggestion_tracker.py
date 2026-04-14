"""AI trade-suggestion tracker.

Persists every generated trade suggestion with a market snapshot, the user's
terminal action (placed / rejected) plus any edit diff, and the eventual broker
outcome (filled, closed, pnl). Scoped per-profile — each profile gets its own
`logs/<profile_name>/ai_suggestions.sqlite` file alongside `assistant.db`.

The collection layer runs now; the stats dashboard comes later once we've accrued
enough data to compare models. The primary query this schema is built for is:

    SELECT model, COUNT(*), AVG(pnl), SUM(CASE WHEN win_loss='win' THEN 1 ELSE 0 END)
    FROM ai_suggestions
    WHERE action='placed' AND closed_at IS NOT NULL
    GROUP BY model;
"""
from __future__ import annotations

import json
import math
import sqlite3
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS ai_suggestions (
    suggestion_id TEXT PRIMARY KEY,
    created_utc TEXT NOT NULL,
    profile TEXT NOT NULL,
    model TEXT NOT NULL,

    -- Suggestion output (what the model returned)
    side TEXT NOT NULL,
    limit_price REAL NOT NULL,
    sl REAL,
    tp REAL,
    lots REAL NOT NULL,
    time_in_force TEXT,
    gtd_time_utc TEXT,
    confidence TEXT,
    rationale TEXT,
    exit_strategy TEXT,
    exit_params_json TEXT,

    -- Market snapshot at generation time (selected fields as JSON)
    market_snapshot_json TEXT,

    -- Terminal user action (NULL until Place or Reject fires)
    action TEXT,                -- 'placed' | 'rejected' | NULL
    action_utc TEXT,
    edited_fields_json TEXT,    -- diff vs original: {field: {before, after}}
    placed_order_json TEXT,     -- actual order payload sent when action='placed'
    oanda_order_id TEXT,        -- set on 'placed'
    trade_id TEXT,              -- linked ai_manual trade row once the limit fills

    -- Broker outcome (populated by run_loop watchdogs)
    outcome_status TEXT,        -- 'filled' | 'cancelled' | 'expired' | NULL
    filled_at TEXT,
    fill_price REAL,
    closed_at TEXT,
    exit_price REAL,
    pnl REAL,
    pips REAL,
    win_loss TEXT               -- 'win' | 'loss' | 'breakeven' | NULL
);

CREATE INDEX IF NOT EXISTS idx_ai_suggestions_model
    ON ai_suggestions(model);
CREATE INDEX IF NOT EXISTS idx_ai_suggestions_action
    ON ai_suggestions(action);
CREATE INDEX IF NOT EXISTS idx_ai_suggestions_order
    ON ai_suggestions(oanda_order_id);
CREATE INDEX IF NOT EXISTS idx_ai_suggestions_created
    ON ai_suggestions(created_utc);
"""


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path) -> None:
    """Create schema if missing. Idempotent."""
    with _connect(db_path) as conn:
        conn.executescript(SCHEMA_SQL)
        _ensure_column(conn, "ai_suggestions", "placed_order_json", "TEXT")
        _ensure_column(conn, "ai_suggestions", "trade_id", "TEXT")
        conn.commit()


def _ensure_column(conn: sqlite3.Connection, table: str, col: str, col_type: str) -> None:
    cols = {str(r[1]) for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if col in cols:
        return
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")


def _extract_market_snapshot(ctx: dict[str, Any]) -> dict[str, Any]:
    """Pull the specified fields out of the live trading context.

    Keep this flat and compact — no candle arrays, no preamble text. The point
    is to capture enough state to explain why the model said what it said when
    we later compare performance by regime.
    """
    snap: dict[str, Any] = {}

    spot = ctx.get("spot_price") or {}
    if spot:
        snap["live_price_mid"] = spot.get("mid")
        snap["live_price_bid"] = spot.get("bid")
        snap["live_price_ask"] = spot.get("ask")
        snap["spread_pips"] = spot.get("spread_pips")

    session = ctx.get("session") or {}
    if session:
        snap["session"] = {
            "active_sessions": session.get("active_sessions"),
            "overlap": session.get("overlap"),
            "warnings": session.get("warnings"),
        }

    acct = ctx.get("account") or {}
    if acct:
        snap["account_balance"] = acct.get("balance")
        snap["account_equity"] = acct.get("equity")
        snap["margin_used"] = acct.get("margin_used")

    bias = ctx.get("cross_asset_bias") or {}
    if bias:
        snap["macro_bias"] = {
            "combined_bias": bias.get("combined_bias"),
            "confidence": bias.get("confidence"),
            "usdjpy_implication": bias.get("usdjpy_implication"),
            "oil_direction": (bias.get("oil") or {}).get("direction"),
            "dxy_direction": (bias.get("dxy") or {}).get("direction"),
        }

    ta = ctx.get("ta_snapshot") or {}
    if ta:
        tech: dict[str, Any] = {}
        for tf in ("H1", "M5", "M1"):
            t = ta.get(tf)
            if not t:
                continue
            tech[tf] = {
                "regime": t.get("regime"),
                "rsi": t.get("rsi_value"),
                "rsi_zone": t.get("rsi_zone"),
                "macd_direction": t.get("macd_direction"),
                "atr_pips": t.get("atr_pips"),
                "atr_state": t.get("atr_state"),
                "adx": t.get("adx"),
                "adxr": t.get("adxr"),
            }
        if tech:
            snap["technicals"] = tech

    ob = ctx.get("order_book") or {}
    if ob:
        snap["order_book"] = {
            "nearest_support": ob.get("nearest_support"),
            "nearest_support_distance_pips": ob.get("nearest_support_distance_pips"),
            "nearest_resistance": ob.get("nearest_resistance"),
            "nearest_resistance_distance_pips": ob.get("nearest_resistance_distance_pips"),
            "buy_clusters": ob.get("buy_clusters"),
            "sell_clusters": ob.get("sell_clusters"),
        }

    cross = ctx.get("cross_assets") or {}
    if cross:
        snap["cross_assets"] = {
            "dxy": cross.get("dxy"),
            "eurusd": cross.get("eurusd"),
            "us10y_yield": cross.get("us10y_yield"),
            "wti_usd": cross.get("wti_usd"),
            "bco_usd": cross.get("bco_usd"),
            "xau_usd": cross.get("xau_usd"),
        }

    vol = ctx.get("volatility") or {}
    if vol:
        snap["volatility"] = {
            "label": vol.get("label"),
            "ratio": vol.get("ratio"),
            "recent_avg_pips": vol.get("recent_avg_pips"),
        }

    return snap


def log_generated(
    db_path: Path,
    *,
    profile: str,
    model: str,
    suggestion: dict[str, Any],
    ctx: dict[str, Any],
) -> str:
    """Insert a new suggestion row. Returns the generated suggestion_id."""
    init_db(db_path)

    suggestion_id = uuid.uuid4().hex
    snapshot = _extract_market_snapshot(ctx)

    def _fnum(key: str) -> Optional[float]:
        v = suggestion.get(key)
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    row = {
        "suggestion_id": suggestion_id,
        "created_utc": _now_utc_iso(),
        "profile": profile,
        "model": model,
        "side": str(suggestion.get("side") or "").lower(),
        "limit_price": _fnum("price") or 0.0,
        "sl": _fnum("sl"),
        "tp": _fnum("tp"),
        "lots": _fnum("lots") or 0.0,
        "time_in_force": suggestion.get("time_in_force"),
        "gtd_time_utc": suggestion.get("gtd_time_utc"),
        "confidence": suggestion.get("confidence"),
        "rationale": suggestion.get("rationale"),
        "exit_strategy": suggestion.get("exit_strategy"),
        "exit_params_json": json.dumps(suggestion.get("exit_params") or {}),
        "market_snapshot_json": json.dumps(snapshot),
    }

    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO ai_suggestions (
                suggestion_id, created_utc, profile, model,
                side, limit_price, sl, tp, lots, time_in_force, gtd_time_utc,
                confidence, rationale, exit_strategy, exit_params_json,
                market_snapshot_json
            ) VALUES (
                :suggestion_id, :created_utc, :profile, :model,
                :side, :limit_price, :sl, :tp, :lots, :time_in_force, :gtd_time_utc,
                :confidence, :rationale, :exit_strategy, :exit_params_json,
                :market_snapshot_json
            )
            """,
            row,
        )
        conn.commit()

    return suggestion_id


def log_action(
    db_path: Path,
    *,
    suggestion_id: str,
    action: str,
    edited_fields: Optional[dict[str, Any]] = None,
    placed_order: Optional[dict[str, Any]] = None,
    oanda_order_id: Optional[str] = None,
) -> bool:
    """Record the user's terminal action (placed or rejected).

    `edited_fields` is a diff: {field_name: {"before": X, "after": Y}}. Pass
    None or {} if the user accepted the suggestion verbatim.
    """
    action = (action or "").strip().lower()
    if action not in ("placed", "rejected"):
        raise ValueError(f"action must be 'placed' or 'rejected', got {action!r}")

    init_db(db_path)
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            UPDATE ai_suggestions
            SET action = ?,
                action_utc = ?,
                edited_fields_json = ?,
                placed_order_json = COALESCE(?, placed_order_json),
                oanda_order_id = COALESCE(?, oanda_order_id)
            WHERE suggestion_id = ?
            """,
            (
                action,
                _now_utc_iso(),
                json.dumps(edited_fields or {}),
                json.dumps(placed_order) if placed_order is not None else None,
                oanda_order_id,
                suggestion_id,
            ),
        )
        conn.commit()
        return cur.rowcount > 0


def mark_filled(
    db_path: Path,
    *,
    oanda_order_id: str,
    fill_price: float,
    filled_at: Optional[str] = None,
    trade_id: Optional[str] = None,
) -> bool:
    """Flag a previously-placed suggestion as filled by the broker.

    Called by run_loop when the pending-order watchdog detects a fill. We look
    up the row by oanda_order_id (written in log_action on 'placed').
    """
    init_db(db_path)
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            UPDATE ai_suggestions
            SET outcome_status = 'filled',
                filled_at = ?,
                fill_price = ?,
                trade_id = COALESCE(?, trade_id)
            WHERE oanda_order_id = ?
              AND outcome_status IS NULL
            """,
            (filled_at or _now_utc_iso(), float(fill_price), trade_id, str(oanda_order_id)),
        )
        conn.commit()
        return cur.rowcount > 0


def mark_closed(
    db_path: Path,
    *,
    oanda_order_id: str,
    exit_price: float,
    pnl: float,
    pips: float,
    closed_at: Optional[str] = None,
) -> bool:
    """Flag a filled suggestion-trade as closed and stamp the P&L outcome."""
    init_db(db_path)
    if pnl > 0:
        outcome = "win"
    elif pnl < 0:
        outcome = "loss"
    else:
        outcome = "breakeven"
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            UPDATE ai_suggestions
            SET closed_at = ?,
                exit_price = ?,
                pnl = ?,
                pips = ?,
                win_loss = ?
            WHERE oanda_order_id = ?
              AND closed_at IS NULL
            """,
            (
                closed_at or _now_utc_iso(),
                float(exit_price),
                float(pnl),
                float(pips),
                outcome,
                str(oanda_order_id),
            ),
        )
        conn.commit()
        return cur.rowcount > 0


def mark_cancelled_or_expired(
    db_path: Path,
    *,
    oanda_order_id: str,
    status: str,
    at: Optional[str] = None,
) -> bool:
    """Flag a placed-but-unfilled order as cancelled or expired."""
    status = (status or "").strip().lower()
    if status not in ("cancelled", "expired"):
        raise ValueError("status must be 'cancelled' or 'expired'")
    init_db(db_path)
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            UPDATE ai_suggestions
            SET outcome_status = ?,
                closed_at = ?
            WHERE oanda_order_id = ?
              AND outcome_status IS NULL
            """,
            (status, at or _now_utc_iso(), str(oanda_order_id)),
        )
        conn.commit()
        return cur.rowcount > 0


def get_by_order_id(db_path: Path, oanda_order_id: str) -> Optional[dict[str, Any]]:
    """Look up a suggestion row by its linked OANDA order ID."""
    init_db(db_path)
    with _connect(db_path) as conn:
        cur = conn.execute(
            "SELECT * FROM ai_suggestions WHERE oanda_order_id = ? LIMIT 1",
            (str(oanda_order_id),),
        )
        row = cur.fetchone()
        return _deserialize_row(dict(row)) if row else None


def get_history(
    db_path: Path,
    *,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    """Return row-level suggestion history ordered newest-first."""
    init_db(db_path)
    limit = max(1, min(int(limit), 500))
    offset = max(0, int(offset))

    with _connect(db_path) as conn:
        total = int(conn.execute("SELECT COUNT(*) FROM ai_suggestions").fetchone()[0] or 0)
        cur = conn.execute(
            """
            SELECT *
            FROM ai_suggestions
            ORDER BY created_utc DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        )
        items = [_deserialize_row(dict(row)) for row in cur.fetchall()]

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": items,
    }


def get_stats(db_path: Path, days_back: int = 30) -> dict[str, Any]:
    """Return summary stats grouped by model — the headline query.

    Shape of the return is designed to feed directly into the eventual UI card.
    Call shape is stable so the frontend can be built against it today even
    though we don't render anything yet.
    """
    init_db(db_path)
    cutoff_iso = (
        datetime.now(timezone.utc) - _timedelta_days(days_back)
    ).isoformat()

    rows = _load_rows_since(db_path, cutoff_iso)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("model") or "unknown")].append(row)
    by_model = [
        {"model": model, **_summarize_rows(model_rows)}
        for model, model_rows in sorted(grouped.items(), key=lambda item: len(item[1]), reverse=True)
    ]

    return {
        "days_back": days_back,
        "since_utc": cutoff_iso,
        "overall": _summarize_rows(rows),
        "by_model": by_model,
    }


def build_learning_prompt_block(
    db_path: Path,
    *,
    days_back: int = 180,
    max_recent_examples: int = 8,
) -> str:
    """Compact performance-memory block for the AI suggestion prompt."""
    init_db(db_path)
    cutoff_iso = (datetime.now(timezone.utc) - _timedelta_days(days_back)).isoformat()
    rows = _load_rows_since(db_path, cutoff_iso)
    if not rows:
        return (
            "=== FILLMORE LEARNING MEMORY (recent AI limit-order outcomes) ===\n"
            "No prior AI suggestion history is available yet. Do not force a setup; prioritize selectivity.\n"
        )

    summary = _summarize_rows(rows)
    recent_examples = _format_recent_examples(rows, max_items=max_recent_examples)
    top_edit_fields = _top_edited_fields(rows, limit=4)
    bias_summary = _label_summary(rows, key="macro_bias")
    session_summary = _label_summary(rows, key="session")
    vol_summary = _label_summary(rows, key="vol_label")

    lines = [
        "=== FILLMORE LEARNING MEMORY (recent AI limit-order outcomes) ===",
        "Use this as behavioral feedback from prior AI suggestions. Treat LIVE TRADING CONTEXT above as authoritative for the current market.",
        (
            f"Recent sample: {summary['generated']} generated, {summary['placed']} placed, "
            f"{summary['rejected']} rejected, {summary['edited_before_placed']} edited-before-place."
        ),
        (
            f"Limit-order behavior: fill rate after placement {summary['fill_rate_after_placement_pct']:.1f}%, "
            f"cancel/expire rate {summary['cancel_or_expire_rate_after_placement_pct']:.1f}%, "
            f"avg time to fill {summary['avg_time_to_fill_minutes']:.1f}m."
        ),
        (
            f"Closed outcomes: {summary['closed']} closed, win rate {summary['win_rate_closed_pct']:.1f}%, "
            f"avg pnl {summary['avg_pnl_closed']:.2f}, avg pips {summary['avg_pips_closed']:.2f}, "
            f"avg hold {summary['avg_hold_minutes']:.1f}m."
        ),
    ]
    if top_edit_fields:
        lines.append(f"Most-edited fields before placement/rejection: {', '.join(top_edit_fields)}.")
    if bias_summary:
        lines.append(f"Macro-bias outcomes seen: {bias_summary}.")
    if session_summary:
        lines.append(f"Session outcomes seen: {session_summary}.")
    if vol_summary:
        lines.append(f"Volatility outcomes seen: {vol_summary}.")
    if recent_examples:
        lines.append("Recent examples:")
        lines.extend(f"  - {line}" for line in recent_examples)
    lines.append(
        "Learning rule: prefer setups that align with what has been filling and closing well; be cautious with patterns that were often edited, rejected, cancelled, or closed poorly."
    )
    return "\n".join(lines)


def _timedelta_days(days: int):
    from datetime import timedelta

    return timedelta(days=max(1, int(days)))


def _load_rows_since(db_path: Path, cutoff_iso: str) -> list[dict[str, Any]]:
    init_db(db_path)
    with _connect(db_path) as conn:
        cur = conn.execute(
            "SELECT * FROM ai_suggestions WHERE created_utc >= ? ORDER BY created_utc DESC",
            (cutoff_iso,),
        )
        return [dict(r) for r in cur.fetchall()]


def _deserialize_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["market_snapshot"] = _json_obj(out.pop("market_snapshot_json", None))
    out["edited_fields"] = _json_obj(out.pop("edited_fields_json", None))
    out["placed_order"] = _json_obj(out.pop("placed_order_json", None))
    out["exit_params"] = _json_obj(out.pop("exit_params_json", None))
    return out


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    generated = len(rows)
    placed_rows = [row for row in rows if str(row.get("action") or "") == "placed"]
    rejected_rows = [row for row in rows if str(row.get("action") or "") == "rejected"]
    filled_rows = [row for row in placed_rows if str(row.get("outcome_status") or "") == "filled"]
    closed_rows = [row for row in rows if row.get("closed_at") and row.get("pnl") is not None]
    cancelled_rows = [row for row in placed_rows if str(row.get("outcome_status") or "") == "cancelled"]
    expired_rows = [row for row in placed_rows if str(row.get("outcome_status") or "") == "expired"]
    edited_placed_rows = [row for row in placed_rows if _row_was_edited(row)]
    edited_rejected_rows = [row for row in rejected_rows if _row_was_edited(row)]

    wins = [row for row in closed_rows if str(row.get("win_loss") or "") == "win"]
    losses = [row for row in closed_rows if str(row.get("win_loss") or "") == "loss"]
    breakeven_rows = [row for row in closed_rows if str(row.get("win_loss") or "") == "breakeven"]

    pnl_values = [_to_float(row.get("pnl")) for row in closed_rows]
    pips_values = [_to_float(row.get("pips")) for row in closed_rows]
    fill_minutes = [_minutes_between(row.get("action_utc"), row.get("filled_at")) for row in filled_rows]
    hold_minutes = [_minutes_between(row.get("filled_at"), row.get("closed_at")) for row in closed_rows]

    return {
        "generated": generated,
        "placed": len(placed_rows),
        "rejected": len(rejected_rows),
        "edited_before_placed": len(edited_placed_rows),
        "edited_before_rejected": len(edited_rejected_rows),
        "filled": len(filled_rows),
        "cancelled": len(cancelled_rows),
        "expired": len(expired_rows),
        "closed": len(closed_rows),
        "wins": len(wins),
        "losses": len(losses),
        "breakeven": len(breakeven_rows),
        "total_pnl_closed": round(sum(pnl_values), 4),
        "total_pips_closed": round(sum(pips_values), 4),
        "avg_pnl_closed": round(_safe_avg(pnl_values), 4),
        "avg_pips_closed": round(_safe_avg(pips_values), 4),
        "fill_rate_after_placement_pct": round(_safe_rate(len(filled_rows), len(placed_rows)), 2),
        "cancel_or_expire_rate_after_placement_pct": round(
            _safe_rate(len(cancelled_rows) + len(expired_rows), len(placed_rows)), 2
        ),
        "win_rate_closed_pct": round(_safe_rate(len(wins), len(closed_rows)), 2),
        "avg_time_to_fill_minutes": round(_safe_avg(fill_minutes), 2),
        "avg_hold_minutes": round(_safe_avg(hold_minutes), 2),
    }


def _format_recent_examples(rows: list[dict[str, Any]], max_items: int = 8) -> list[str]:
    out: list[str] = []
    for row in rows[: max(1, max_items)]:
        snap = _snapshot_context(row)
        side = str(row.get("side") or "?").upper()
        action = str(row.get("action") or "none")
        edited = "edited" if _row_was_edited(row) else "verbatim"
        outcome = str(row.get("win_loss") or row.get("outcome_status") or "open")
        pnl = row.get("pnl")
        pips = row.get("pips")
        detail = f"{side} {action}/{edited} -> {outcome}"
        if pnl is not None and pips is not None:
            detail += f" ({_to_float(pips):+.1f}p, pnl {_to_float(pnl):+.2f})"
        elif pips is not None:
            detail += f" ({_to_float(pips):+.1f}p)"
        extras = [
            f"session={snap.get('session')}" if snap.get("session") else None,
            f"bias={snap.get('macro_bias')}" if snap.get("macro_bias") else None,
            f"vol={snap.get('vol_label')}" if snap.get("vol_label") else None,
            f"exit={_placed_exit_strategy(row)}" if _placed_exit_strategy(row) else None,
        ]
        extras = [item for item in extras if item]
        if extras:
            detail += " | " + ", ".join(extras)
        out.append(detail)
    return out


def _top_edited_fields(rows: list[dict[str, Any]], limit: int = 4) -> list[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        for key in _edited_fields(row).keys():
            counts[str(key)] += 1
    return [field for field, _ in counts.most_common(max(1, limit))]


def _label_summary(rows: list[dict[str, Any]], *, key: str) -> str:
    counts: Counter[str] = Counter()
    wins: Counter[str] = Counter()
    for row in rows:
        snap = _snapshot_context(row)
        label = str(snap.get(key) or "").strip().lower()
        if not label:
            continue
        counts[label] += 1
        if str(row.get("win_loss") or "") == "win":
            wins[label] += 1
    if not counts:
        return ""
    parts: list[str] = []
    for label, total in counts.most_common(3):
        win_rate = _safe_rate(wins[label], total)
        parts.append(f"{label} ({total} seen, {win_rate:.0f}% wins)")
    return ", ".join(parts)


def _snapshot_context(row: dict[str, Any]) -> dict[str, Any]:
    raw = _json_obj(row.get("market_snapshot_json"))
    session = _json_obj(raw.get("session"))
    macro_bias = _json_obj(raw.get("macro_bias"))
    volatility = _json_obj(raw.get("volatility"))
    return {
        "session": _normalize_label(session.get("overlap")) or _normalize_label_list(session.get("active_sessions")),
        "macro_bias": _normalize_label(macro_bias.get("combined_bias")),
        "vol_label": _normalize_label(volatility.get("label")),
    }


def _placed_exit_strategy(row: dict[str, Any]) -> str:
    placed = _json_obj(row.get("placed_order_json"))
    if placed.get("exit_strategy") is not None:
        return str(placed.get("exit_strategy") or "").strip()
    return str(row.get("exit_strategy") or "").strip()


def _edited_fields(row: dict[str, Any]) -> dict[str, Any]:
    data = _json_obj(row.get("edited_fields_json"))
    return data if isinstance(data, dict) else {}


def _row_was_edited(row: dict[str, Any]) -> bool:
    return len(_edited_fields(row)) > 0


def _json_obj(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(str(raw))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _normalize_label(value: Any) -> str:
    text = str(value or "").strip()
    return text.lower() if text else ""


def _normalize_label_list(values: Any) -> str:
    if not isinstance(values, list):
        return ""
    cleaned = [str(v).strip().lower() for v in values if str(v).strip()]
    return "+".join(cleaned[:2])


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_avg(values: list[float | None]) -> float:
    cleaned = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not cleaned:
        return 0.0
    return sum(cleaned) / len(cleaned)


def _safe_rate(part: int, whole: int) -> float:
    if whole <= 0:
        return 0.0
    return (float(part) / float(whole)) * 100.0


def _parse_iso(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _minutes_between(start: Any, end: Any) -> Optional[float]:
    start_dt = _parse_iso(start)
    end_dt = _parse_iso(end)
    if start_dt is None or end_dt is None:
        return None
    return max(0.0, (end_dt - start_dt).total_seconds() / 60.0)
