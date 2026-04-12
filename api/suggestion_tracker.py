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
import sqlite3
import uuid
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
    oanda_order_id TEXT,        -- set on 'placed'

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
        conn.commit()


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
                oanda_order_id = COALESCE(?, oanda_order_id)
            WHERE suggestion_id = ?
            """,
            (
                action,
                _now_utc_iso(),
                json.dumps(edited_fields or {}),
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
                fill_price = ?
            WHERE oanda_order_id = ?
              AND outcome_status IS NULL
            """,
            (filled_at or _now_utc_iso(), float(fill_price), str(oanda_order_id)),
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
        return dict(row) if row else None


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

    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            SELECT
                model,
                COUNT(*) AS total,
                SUM(CASE WHEN action = 'placed' THEN 1 ELSE 0 END) AS placed,
                SUM(CASE WHEN action = 'rejected' THEN 1 ELSE 0 END) AS rejected,
                SUM(CASE WHEN outcome_status = 'filled' THEN 1 ELSE 0 END) AS filled,
                SUM(CASE WHEN win_loss = 'win' THEN 1 ELSE 0 END) AS wins,
                SUM(CASE WHEN win_loss = 'loss' THEN 1 ELSE 0 END) AS losses,
                SUM(CASE WHEN closed_at IS NOT NULL THEN pnl ELSE 0 END) AS total_pnl,
                SUM(CASE WHEN closed_at IS NOT NULL THEN pips ELSE 0 END) AS total_pips
            FROM ai_suggestions
            WHERE created_utc >= ?
            GROUP BY model
            ORDER BY total DESC
            """,
            (cutoff_iso,),
        )
        by_model = [dict(r) for r in cur.fetchall()]

    return {
        "days_back": days_back,
        "since_utc": cutoff_iso,
        "by_model": by_model,
    }


def _timedelta_days(days: int):
    from datetime import timedelta

    return timedelta(days=max(1, int(days)))
