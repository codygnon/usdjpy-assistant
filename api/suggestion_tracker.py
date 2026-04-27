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

from api.suggestion_features import compute_suggestion_features

ENTRY_TYPE_FILLMORE_MANUAL = "ai_manual"
ENTRY_TYPE_FILLMORE_AUTONOMOUS = "ai_autonomous"
_FILLMORE_ENTRY_TYPES = {ENTRY_TYPE_FILLMORE_MANUAL, ENTRY_TYPE_FILLMORE_AUTONOMOUS}
_AUTONOMOUS_SOURCES = {"autonomous_fillmore", "autonomous_fillmore_paper"}


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
    requested_price REAL,
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
    exit_plan TEXT,
    prompt_version TEXT,
    prompt_hash TEXT,
    snap_distance_pips REAL,
    entry_type TEXT,
    trigger_family TEXT,
    trigger_reason TEXT,
    thesis_fingerprint TEXT,
    decision TEXT,
    conviction_rung TEXT,
    skip_reason TEXT,
    trade_thesis TEXT,
    whats_different TEXT,
    why_not_stop TEXT,
    zone_memory_read TEXT,
    repeat_trade_case TEXT,
    planned_rr_estimate REAL,
    low_rr_edge TEXT,
    timeframe_alignment TEXT,
    countertrend_edge TEXT,
    trigger_fit TEXT,
    why_trade_despite_weakness TEXT,
    custom_exit_plan_json TEXT,
    features_json TEXT,
    max_adverse_pips REAL,
    max_favorable_pips REAL,
    mae_mfe_estimated INTEGER,

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
    price_at_expiry REAL,
    distance_at_expiry_pips REAL,
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

CREATE TABLE IF NOT EXISTS ai_thesis_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile TEXT NOT NULL,
    suggestion_id TEXT,
    trade_id TEXT,
    position_id TEXT,
    created_utc TEXT NOT NULL,
    model TEXT NOT NULL,
    action TEXT NOT NULL,
    reason TEXT,
    requested_new_sl REAL,
    requested_scale_out_pct REAL,
    confidence TEXT,
    check_reason TEXT,
    current_pips REAL,
    current_mae_pips REAL,
    current_mfe_pips REAL,
    exit_state TEXT,
    invalidation_status TEXT,
    management_intent TEXT,
    updated_exit_plan TEXT,
    next_watch_condition TEXT,
    custom_exit INTEGER NOT NULL DEFAULT 0,
    execution_succeeded INTEGER NOT NULL DEFAULT 0,
    execution_note TEXT
);

CREATE INDEX IF NOT EXISTS idx_ai_thesis_checks_trade
    ON ai_thesis_checks(trade_id, created_utc DESC);
CREATE INDEX IF NOT EXISTS idx_ai_thesis_checks_suggestion
    ON ai_thesis_checks(suggestion_id, created_utc DESC);

CREATE TABLE IF NOT EXISTS ai_reflections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile TEXT NOT NULL,
    suggestion_id TEXT NOT NULL,
    trade_id TEXT NOT NULL,
    created_utc TEXT NOT NULL,
    model TEXT NOT NULL,
    what_read_right TEXT NOT NULL,
    what_missed TEXT NOT NULL,
    summary_text TEXT NOT NULL,
    primary_error_category TEXT,
    primary_strength_category TEXT,
    regime_at_entry TEXT,
    session_at_entry TEXT,
    lesson TEXT,
    autonomous INTEGER NOT NULL DEFAULT 0,
    UNIQUE(profile, suggestion_id, trade_id)
);

CREATE INDEX IF NOT EXISTS idx_ai_reflections_created
    ON ai_reflections(created_utc DESC);
CREATE INDEX IF NOT EXISTS idx_ai_reflections_autonomous
    ON ai_reflections(autonomous, created_utc DESC);

CREATE TABLE IF NOT EXISTS ai_performance_stats (
    profile TEXT NOT NULL,
    stats_key TEXT NOT NULL,
    trade_count INTEGER DEFAULT 0,
    closed_count INTEGER DEFAULT 0,
    win_rate REAL,
    avg_win_pips REAL,
    avg_loss_pips REAL,
    net_pnl REAL,
    profit_factor REAL,
    avg_hold_minutes REAL,
    fill_rate_limits REAL,
    avg_time_to_fill_sec REAL,
    avg_fill_vs_requested_pips REAL,
    thesis_intervention_rate REAL,
    avg_mae_pips REAL,
    avg_mfe_pips REAL,
    win_rate_by_confidence_json TEXT,
    win_rate_by_side_json TEXT,
    win_rate_by_session_json TEXT,
    prompt_version_breakdown_json TEXT,
    updated_utc TEXT NOT NULL,
    PRIMARY KEY (profile, stats_key)
);
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
        _ensure_column(conn, "ai_suggestions", "exit_plan", "TEXT")
        _ensure_column(conn, "ai_suggestions", "requested_price", "REAL")
        _ensure_column(conn, "ai_suggestions", "prompt_version", "TEXT")
        _ensure_column(conn, "ai_suggestions", "prompt_hash", "TEXT")
        _ensure_column(conn, "ai_suggestions", "snap_distance_pips", "REAL")
        _ensure_column(conn, "ai_suggestions", "entry_type", "TEXT")
        _ensure_column(conn, "ai_suggestions", "trigger_family", "TEXT")
        _ensure_column(conn, "ai_suggestions", "trigger_reason", "TEXT")
        _ensure_column(conn, "ai_suggestions", "thesis_fingerprint", "TEXT")
        _ensure_column(conn, "ai_suggestions", "decision", "TEXT")
        _ensure_column(conn, "ai_suggestions", "conviction_rung", "TEXT")
        _ensure_column(conn, "ai_suggestions", "skip_reason", "TEXT")
        _ensure_column(conn, "ai_suggestions", "trade_thesis", "TEXT")
        _ensure_column(conn, "ai_suggestions", "whats_different", "TEXT")
        _ensure_column(conn, "ai_suggestions", "why_not_stop", "TEXT")
        _ensure_column(conn, "ai_suggestions", "zone_memory_read", "TEXT")
        _ensure_column(conn, "ai_suggestions", "repeat_trade_case", "TEXT")
        _ensure_column(conn, "ai_suggestions", "planned_rr_estimate", "REAL")
        _ensure_column(conn, "ai_suggestions", "low_rr_edge", "TEXT")
        _ensure_column(conn, "ai_suggestions", "timeframe_alignment", "TEXT")
        _ensure_column(conn, "ai_suggestions", "countertrend_edge", "TEXT")
        _ensure_column(conn, "ai_suggestions", "trigger_fit", "TEXT")
        _ensure_column(conn, "ai_suggestions", "why_trade_despite_weakness", "TEXT")
        _ensure_column(conn, "ai_suggestions", "custom_exit_plan_json", "TEXT")
        _ensure_column(conn, "ai_suggestions", "features_json", "TEXT")
        _ensure_column(conn, "ai_suggestions", "max_adverse_pips", "REAL")
        _ensure_column(conn, "ai_suggestions", "max_favorable_pips", "REAL")
        _ensure_column(conn, "ai_suggestions", "mae_mfe_estimated", "INTEGER")
        _ensure_column(conn, "ai_suggestions", "price_at_expiry", "REAL")
        _ensure_column(conn, "ai_suggestions", "distance_at_expiry_pips", "REAL")
        _ensure_column(conn, "ai_thesis_checks", "check_reason", "TEXT")
        _ensure_column(conn, "ai_thesis_checks", "current_pips", "REAL")
        _ensure_column(conn, "ai_thesis_checks", "current_mae_pips", "REAL")
        _ensure_column(conn, "ai_thesis_checks", "current_mfe_pips", "REAL")
        _ensure_column(conn, "ai_thesis_checks", "exit_state", "TEXT")
        _ensure_column(conn, "ai_thesis_checks", "invalidation_status", "TEXT")
        _ensure_column(conn, "ai_thesis_checks", "management_intent", "TEXT")
        _ensure_column(conn, "ai_thesis_checks", "updated_exit_plan", "TEXT")
        _ensure_column(conn, "ai_thesis_checks", "next_watch_condition", "TEXT")
        _ensure_column(conn, "ai_thesis_checks", "custom_exit", "INTEGER")
        _ensure_column(conn, "ai_performance_stats", "net_pnl", "REAL")
        _ensure_column(conn, "ai_reflections", "primary_error_category", "TEXT")
        _ensure_column(conn, "ai_reflections", "primary_strength_category", "TEXT")
        _ensure_column(conn, "ai_reflections", "regime_at_entry", "TEXT")
        _ensure_column(conn, "ai_reflections", "session_at_entry", "TEXT")
        _ensure_column(conn, "ai_reflections", "lesson", "TEXT")
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
    features = compute_suggestion_features(suggestion, ctx)

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
        "requested_price": _fnum("requested_price") if _fnum("requested_price") is not None else (_fnum("price") or 0.0),
        "limit_price": _fnum("price") or 0.0,
        "sl": _fnum("sl"),
        "tp": _fnum("tp"),
        "lots": _fnum("lots") or 0.0,
        "time_in_force": suggestion.get("time_in_force"),
        "gtd_time_utc": suggestion.get("gtd_time_utc"),
        "confidence": suggestion.get("quality") or suggestion.get("confidence"),
        "rationale": suggestion.get("rationale"),
        "exit_strategy": suggestion.get("exit_strategy"),
        "exit_params_json": json.dumps(suggestion.get("exit_params") or {}),
        "exit_plan": suggestion.get("exit_plan"),
        "prompt_version": suggestion.get("prompt_version"),
        "prompt_hash": suggestion.get("prompt_hash"),
        "snap_distance_pips": _fnum("snap_distance_pips"),
        "entry_type": str(suggestion.get("entry_type") or ENTRY_TYPE_FILLMORE_MANUAL).strip().lower(),
        "trigger_family": suggestion.get("trigger_family"),
        "trigger_reason": suggestion.get("trigger_reason"),
        "thesis_fingerprint": suggestion.get("thesis_fingerprint"),
        "decision": suggestion.get("decision"),
        "conviction_rung": suggestion.get("conviction_rung"),
        "skip_reason": suggestion.get("skip_reason"),
        "trade_thesis": suggestion.get("trade_thesis"),
        "whats_different": suggestion.get("whats_different"),
        "why_not_stop": suggestion.get("why_not_stop"),
        "zone_memory_read": suggestion.get("zone_memory_read"),
        "repeat_trade_case": suggestion.get("repeat_trade_case"),
        "planned_rr_estimate": _fnum("planned_rr_estimate"),
        "low_rr_edge": suggestion.get("low_rr_edge"),
        "timeframe_alignment": suggestion.get("timeframe_alignment"),
        "countertrend_edge": suggestion.get("countertrend_edge"),
        "trigger_fit": suggestion.get("trigger_fit"),
        "why_trade_despite_weakness": suggestion.get("why_trade_despite_weakness"),
        "custom_exit_plan_json": json.dumps(suggestion.get("custom_exit_plan") or {}),
        "features_json": json.dumps(features),
        "market_snapshot_json": json.dumps(snapshot),
    }

    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO ai_suggestions (
                suggestion_id, created_utc, profile, model,
                side, requested_price, limit_price, sl, tp, lots, time_in_force, gtd_time_utc,
                confidence, rationale, exit_strategy, exit_params_json,
                exit_plan, prompt_version, prompt_hash, snap_distance_pips,
                entry_type, trigger_family, trigger_reason,
                thesis_fingerprint, decision, conviction_rung, skip_reason,
                trade_thesis, whats_different, why_not_stop,
                zone_memory_read, repeat_trade_case, planned_rr_estimate,
                low_rr_edge, timeframe_alignment, countertrend_edge, trigger_fit,
                why_trade_despite_weakness, custom_exit_plan_json,
                features_json, market_snapshot_json
            ) VALUES (
                :suggestion_id, :created_utc, :profile, :model,
                :side, :requested_price, :limit_price, :sl, :tp, :lots, :time_in_force, :gtd_time_utc,
                :confidence, :rationale, :exit_strategy, :exit_params_json,
                :exit_plan, :prompt_version, :prompt_hash, :snap_distance_pips,
                :entry_type, :trigger_family, :trigger_reason,
                :thesis_fingerprint, :decision, :conviction_rung, :skip_reason,
                :trade_thesis, :whats_different, :why_not_stop,
                :zone_memory_read, :repeat_trade_case, :planned_rr_estimate,
                :low_rr_edge, :timeframe_alignment, :countertrend_edge, :trigger_fit,
                :why_trade_despite_weakness, :custom_exit_plan_json,
                :features_json, :market_snapshot_json
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
    max_adverse_pips: float | None = None,
    max_favorable_pips: float | None = None,
    mae_mfe_estimated: int | None = None,
) -> bool:
    """Flag a filled suggestion-trade as closed and stamp the P&L outcome."""
    init_db(db_path)
    outcome = _win_loss_from_pnl_or_pips(pnl=pnl, pips=pips)
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            UPDATE ai_suggestions
            SET closed_at = ?,
                exit_price = ?,
                pnl = ?,
                pips = ?,
                win_loss = ?,
                max_adverse_pips = COALESCE(?, max_adverse_pips),
                max_favorable_pips = COALESCE(?, max_favorable_pips),
                mae_mfe_estimated = COALESCE(?, mae_mfe_estimated)
            WHERE oanda_order_id = ?
              AND closed_at IS NULL
            """,
            (
                closed_at or _now_utc_iso(),
                float(exit_price),
                float(pnl),
                float(pips),
                outcome,
                float(max_adverse_pips) if max_adverse_pips is not None else None,
                float(max_favorable_pips) if max_favorable_pips is not None else None,
                int(mae_mfe_estimated) if mae_mfe_estimated is not None else None,
                str(oanda_order_id),
            ),
        )
        conn.commit()
        return cur.rowcount > 0


def mark_closed_by_suggestion_id(
    db_path: Path,
    *,
    suggestion_id: str,
    exit_price: float,
    pnl: float,
    pips: float,
    closed_at: Optional[str] = None,
    max_adverse_pips: float | None = None,
    max_favorable_pips: float | None = None,
    mae_mfe_estimated: int | None = None,
) -> bool:
    """Close outcome keyed by suggestion_id (paper trades use stable synthetic order ids)."""
    init_db(db_path)
    outcome = _win_loss_from_pnl_or_pips(pnl=pnl, pips=pips)
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            UPDATE ai_suggestions
            SET closed_at = ?,
                exit_price = ?,
                pnl = ?,
                pips = ?,
                win_loss = ?,
                max_adverse_pips = COALESCE(?, max_adverse_pips),
                max_favorable_pips = COALESCE(?, max_favorable_pips),
                mae_mfe_estimated = COALESCE(?, mae_mfe_estimated)
            WHERE suggestion_id = ?
              AND closed_at IS NULL
            """,
            (
                closed_at or _now_utc_iso(),
                float(exit_price),
                float(pnl),
                float(pips),
                outcome,
                float(max_adverse_pips) if max_adverse_pips is not None else None,
                float(max_favorable_pips) if max_favorable_pips is not None else None,
                int(mae_mfe_estimated) if mae_mfe_estimated is not None else None,
                str(suggestion_id),
            ),
        )
        conn.commit()
        return cur.rowcount > 0


def update_excursion(
    db_path: Path,
    *,
    suggestion_id: str | None = None,
    oanda_order_id: str | None = None,
    trade_id: str | None = None,
    max_adverse_pips: float | None = None,
    max_favorable_pips: float | None = None,
    mae_mfe_estimated: int | None = None,
) -> bool:
    init_db(db_path)
    clauses: list[str] = []
    params: list[Any] = []
    if suggestion_id:
        clauses.append("suggestion_id = ?")
        params.append(str(suggestion_id))
    if oanda_order_id:
        clauses.append("oanda_order_id = ?")
        params.append(str(oanda_order_id))
    if trade_id:
        clauses.append("trade_id = ?")
        params.append(str(trade_id))
    if not clauses:
        return False
    with _connect(db_path) as conn:
        cur = conn.execute(
            f"""
            UPDATE ai_suggestions
            SET max_adverse_pips = COALESCE(?, max_adverse_pips),
                max_favorable_pips = COALESCE(?, max_favorable_pips),
                mae_mfe_estimated = COALESCE(?, mae_mfe_estimated)
            WHERE {" OR ".join(clauses)}
            """,
            (
                float(max_adverse_pips) if max_adverse_pips is not None else None,
                float(max_favorable_pips) if max_favorable_pips is not None else None,
                int(mae_mfe_estimated) if mae_mfe_estimated is not None else None,
                *params,
            ),
        )
        conn.commit()
        return cur.rowcount > 0


def backfill_null_win_loss(db_path: Path) -> dict[str, int]:
    init_db(db_path)
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            SELECT suggestion_id, pnl, pips
            FROM ai_suggestions
            WHERE closed_at IS NOT NULL
              AND (win_loss IS NULL OR TRIM(win_loss) = '')
            """
        )
        rows = cur.fetchall()
        updated = 0
        skipped = 0
        for row in rows:
            win_loss = _win_loss_from_pnl_or_pips(pnl=row["pnl"], pips=row["pips"])
            if not win_loss:
                skipped += 1
                continue
            conn.execute(
                """
                UPDATE ai_suggestions
                SET win_loss = COALESCE(win_loss, ?),
                    outcome_status = COALESCE(outcome_status, 'filled')
                WHERE suggestion_id = ?
                """,
                (win_loss, str(row["suggestion_id"])),
            )
            updated += 1
        conn.commit()
    return {"updated": updated, "skipped": skipped, "total": len(rows)}


def mark_cancelled_or_expired(
    db_path: Path,
    *,
    oanda_order_id: str,
    status: str,
    at: Optional[str] = None,
    price_at_expiry: float | None = None,
    distance_at_expiry_pips: float | None = None,
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
                closed_at = ?,
                price_at_expiry = COALESCE(?, price_at_expiry),
                distance_at_expiry_pips = COALESCE(?, distance_at_expiry_pips)
            WHERE oanda_order_id = ?
              AND outcome_status IS NULL
            """,
            (
                status,
                at or _now_utc_iso(),
                float(price_at_expiry) if price_at_expiry is not None else None,
                float(distance_at_expiry_pips) if distance_at_expiry_pips is not None else None,
                str(oanda_order_id),
            ),
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


def get_by_trade_id(db_path: Path, trade_id: str) -> Optional[dict[str, Any]]:
    """Look up a suggestion row by linked trade_id."""
    init_db(db_path)
    with _connect(db_path) as conn:
        cur = conn.execute(
            "SELECT * FROM ai_suggestions WHERE trade_id = ? LIMIT 1",
            (str(trade_id),),
        )
        row = cur.fetchone()
        return _deserialize_row(dict(row)) if row else None


def resolve_suggestion_for_trade(
    db_path: Path,
    *,
    trade_id: str | None = None,
    oanda_order_id: str | None = None,
) -> Optional[dict[str, Any]]:
    """Resolve the best matching suggestion row for a Fillmore trade."""
    if trade_id:
        row = get_by_trade_id(db_path, trade_id)
        if row is not None:
            return row
    if oanda_order_id:
        return get_by_order_id(db_path, oanda_order_id)
    return None


def is_autonomous_suggestion_row(row: dict[str, Any] | None) -> bool:
    if not isinstance(row, dict):
        return False
    entry_type = str(row.get("entry_type") or "").strip().lower()
    if entry_type == ENTRY_TYPE_FILLMORE_AUTONOMOUS:
        return True
    placed = _json_obj(row.get("placed_order"))
    if placed:
        if str(placed.get("entry_type") or "").strip().lower() == ENTRY_TYPE_FILLMORE_AUTONOMOUS:
            return True
        return bool(placed.get("autonomous"))
    placed_json = _json_obj(row.get("placed_order_json"))
    if placed_json:
        if str(placed_json.get("entry_type") or "").strip().lower() == ENTRY_TYPE_FILLMORE_AUTONOMOUS:
            return True
        return bool(placed_json.get("autonomous"))
    return False


def is_fillmore_trade_row(row: dict[str, Any] | None) -> bool:
    if not isinstance(row, dict):
        return False
    entry_type = str(row.get("entry_type") or "").strip().lower()
    policy_type = str(row.get("policy_type") or "").strip().lower()
    opened_by = str(row.get("opened_by") or "").strip().lower()
    trade_id = str(row.get("trade_id") or "").strip().lower()
    notes = str(row.get("notes") or "").strip().lower()
    if entry_type in _FILLMORE_ENTRY_TYPES:
        return True
    if policy_type in _FILLMORE_ENTRY_TYPES:
        return True
    if opened_by in _FILLMORE_ENTRY_TYPES:
        return True
    if trade_id.startswith(f"{ENTRY_TYPE_FILLMORE_MANUAL}:") or trade_id.startswith(f"{ENTRY_TYPE_FILLMORE_AUTONOMOUS}:"):
        return True
    if notes.startswith("ai_manual:") or notes.startswith("autonomous_fillmore:") or notes.startswith("autonomous_fillmore_paper:"):
        return True
    try:
        raw_cfg = row.get("config_json")
        cfg = json.loads(raw_cfg) if isinstance(raw_cfg, str) and raw_cfg.strip() else (raw_cfg if isinstance(raw_cfg, dict) else {})
    except Exception:
        cfg = {}
    source = str((cfg or {}).get("source") or "").strip().lower()
    return source in _AUTONOMOUS_SOURCES or source == ENTRY_TYPE_FILLMORE_MANUAL


def is_autonomous_trade_row(row: dict[str, Any] | None) -> bool:
    if not isinstance(row, dict):
        return False
    entry_type = str(row.get("entry_type") or "").strip().lower()
    if entry_type == ENTRY_TYPE_FILLMORE_AUTONOMOUS:
        return True
    try:
        raw_cfg = row.get("config_json")
        cfg = json.loads(raw_cfg) if isinstance(raw_cfg, str) and raw_cfg.strip() else (raw_cfg if isinstance(raw_cfg, dict) else {})
    except Exception:
        cfg = {}
    source = str((cfg or {}).get("source") or "").strip().lower()
    notes = str(row.get("notes") or "").strip().lower()
    opened_by = str(row.get("opened_by") or "").strip().lower()
    policy_type = str(row.get("policy_type") or "").strip().lower()
    trade_id = str(row.get("trade_id") or "").strip().lower()
    return (
        source in _AUTONOMOUS_SOURCES
        or notes.startswith("autonomous_fillmore:")
        or notes.startswith("autonomous_fillmore_paper:")
        or opened_by == ENTRY_TYPE_FILLMORE_AUTONOMOUS
        or policy_type == ENTRY_TYPE_FILLMORE_AUTONOMOUS
        or trade_id.startswith(f"{ENTRY_TYPE_FILLMORE_AUTONOMOUS}:")
    )


def log_thesis_check(
    db_path: Path,
    *,
    profile: str,
    suggestion_id: str | None,
    trade_id: str | None,
    position_id: str | int | None,
    model: str,
    action: str,
    reason: str,
    requested_new_sl: float | None = None,
    requested_scale_out_pct: float | None = None,
    confidence: str | None = None,
    check_reason: str | None = None,
    current_pips: float | None = None,
    current_mae_pips: float | None = None,
    current_mfe_pips: float | None = None,
    exit_state: str | None = None,
    invalidation_status: str | None = None,
    management_intent: str | None = None,
    updated_exit_plan: str | None = None,
    next_watch_condition: str | None = None,
    custom_exit: bool = False,
    execution_succeeded: bool = False,
    execution_note: str | None = None,
    created_utc: str | None = None,
) -> int:
    """Persist one thesis-monitor check row."""
    init_db(db_path)
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO ai_thesis_checks (
                profile, suggestion_id, trade_id, position_id, created_utc,
                model, action, reason, requested_new_sl, requested_scale_out_pct,
                confidence, check_reason, current_pips, current_mae_pips, current_mfe_pips,
                exit_state, invalidation_status, management_intent, updated_exit_plan,
                next_watch_condition, custom_exit, execution_succeeded, execution_note
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(profile),
                str(suggestion_id or "") or None,
                str(trade_id or "") or None,
                str(position_id) if position_id not in (None, "") else None,
                created_utc or _now_utc_iso(),
                str(model or ""),
                str(action or ""),
                str(reason or ""),
                float(requested_new_sl) if requested_new_sl is not None else None,
                float(requested_scale_out_pct) if requested_scale_out_pct is not None else None,
                str(confidence or "") or None,
                str(check_reason or "") or None,
                float(current_pips) if current_pips is not None else None,
                float(current_mae_pips) if current_mae_pips is not None else None,
                float(current_mfe_pips) if current_mfe_pips is not None else None,
                str(exit_state or "") or None,
                str(invalidation_status or "") or None,
                str(management_intent or "") or None,
                str(updated_exit_plan or "") or None,
                str(next_watch_condition or "") or None,
                1 if custom_exit else 0,
                1 if execution_succeeded else 0,
                str(execution_note or "") or None,
            ),
        )
        conn.commit()
        return int(cur.lastrowid or 0)


def list_thesis_checks(
    db_path: Path,
    *,
    trade_id: str | None = None,
    suggestion_id: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Return thesis-monitor checks newest-first for a trade or suggestion."""
    init_db(db_path)
    limit = max(1, min(int(limit), 100))
    clauses: list[str] = []
    params: list[Any] = []
    if trade_id:
        clauses.append("trade_id = ?")
        params.append(str(trade_id))
    if suggestion_id:
        clauses.append("suggestion_id = ?")
        params.append(str(suggestion_id))
    if not clauses:
        return []
    where = " OR ".join(clauses)
    with _connect(db_path) as conn:
        cur = conn.execute(
            f"""
            SELECT *
            FROM ai_thesis_checks
            WHERE {where}
            ORDER BY created_utc DESC, id DESC
            LIMIT ?
            """,
            (*params, limit),
        )
        return [_deserialize_thesis_check(dict(row)) for row in cur.fetchall()]


def reflection_exists(
    db_path: Path,
    *,
    profile: str,
    suggestion_id: str,
    trade_id: str,
) -> bool:
    init_db(db_path)
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            SELECT 1
            FROM ai_reflections
            WHERE profile = ? AND suggestion_id = ? AND trade_id = ?
            LIMIT 1
            """,
            (str(profile), str(suggestion_id), str(trade_id)),
        )
        return cur.fetchone() is not None


def log_reflection(
    db_path: Path,
    *,
    profile: str,
    suggestion_id: str,
    trade_id: str,
    model: str,
    what_read_right: str,
    what_missed: str,
    summary_text: str,
    autonomous: bool,
    created_utc: str | None = None,
    primary_error_category: str | None = None,
    primary_strength_category: str | None = None,
    regime_at_entry: str | None = None,
    session_at_entry: str | None = None,
    lesson: str | None = None,
) -> bool:
    """Persist one post-trade Fillmore reflection. Idempotent per trade."""
    init_db(db_path)
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO ai_reflections (
                profile, suggestion_id, trade_id, created_utc, model,
                what_read_right, what_missed, summary_text,
                primary_error_category, primary_strength_category,
                regime_at_entry, session_at_entry, lesson, autonomous
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(profile),
                str(suggestion_id),
                str(trade_id),
                created_utc or _now_utc_iso(),
                str(model or ""),
                str(what_read_right or "").strip(),
                str(what_missed or "").strip(),
                str(summary_text or "").strip(),
                str(primary_error_category or "").strip() or None,
                str(primary_strength_category or "").strip() or None,
                str(regime_at_entry or "").strip() or None,
                str(session_at_entry or "").strip() or None,
                str(lesson or "").strip() or None,
                1 if autonomous else 0,
            ),
        )
        conn.commit()
        return cur.rowcount > 0


def get_reflections(
    db_path: Path,
    *,
    limit: int = 10,
    autonomous_only: bool = False,
) -> list[dict[str, Any]]:
    init_db(db_path)
    limit = max(1, min(int(limit), 100))
    where = "WHERE autonomous = 1" if autonomous_only else ""
    with _connect(db_path) as conn:
        cur = conn.execute(
            f"""
            SELECT *
            FROM ai_reflections
            {where}
            ORDER BY created_utc DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [_deserialize_reflection(dict(row)) for row in cur.fetchall()]


def build_autonomous_reflection_prompt_block(
    db_path: Path,
    *,
    limit: int = 10,
    autonomous_only: bool = True,
) -> str:
    """Compact recent self-reflection block for autonomous prompts."""
    rows = get_reflections(db_path, limit=limit, autonomous_only=autonomous_only)
    header = "=== FILLMORE SELF-REFLECTION MEMORY (recent closed-trade postmortems) ==="
    if not rows:
        return (
            f"{header}\n"
            "No prior autonomous postmortems yet. Stay selective and write a clean rationale worth reviewing later."
        )
    lines = [header]
    for row in rows:
        created = str(row.get("created_utc") or "")[:16]
        summary = str(row.get("summary_text") or "").strip()
        right = str(row.get("what_read_right") or "").strip()
        missed = str(row.get("what_missed") or "").strip()
        if summary:
            lines.append(f"- {created} {summary}")
        if right:
            lines.append(f"  Right: {right}")
        if missed:
            lines.append(f"  Missed: {missed}")
    return "\n".join(lines)


def get_recent_thesis_checks(
    db_path: Path,
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return the N most recent thesis checks across all trades, newest first."""
    init_db(db_path)
    limit = max(1, min(int(limit), 100))
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            SELECT *
            FROM ai_thesis_checks
            ORDER BY created_utc DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [_deserialize_thesis_check(dict(row)) for row in cur.fetchall()]


def get_confidence_distribution(
    db_path: Path,
    *,
    days: int = 1,
) -> dict[str, Any]:
    """Count generated suggestions by confidence level over the last N days."""
    init_db(db_path)
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            SELECT
                LOWER(COALESCE(confidence, 'unknown')) AS conf,
                COUNT(*) AS cnt,
                SUM(CASE WHEN action = 'placed' THEN 1 ELSE 0 END) AS placed
            FROM ai_suggestions
            WHERE created_utc >= ?
            GROUP BY conf
            ORDER BY cnt DESC
            """,
            (cutoff,),
        )
        dist: dict[str, dict[str, int]] = {}
        total = 0
        total_placed = 0
        for row in cur.fetchall():
            conf = str(row["conf"] or "unknown").strip() or "unknown"
            cnt = int(row["cnt"])
            placed = int(row["placed"])
            dist[conf] = {"generated": cnt, "placed": placed}
            total += cnt
            total_placed += placed
    return {
        "days": days,
        "total_generated": total,
        "total_placed": total_placed,
        "placement_rate_pct": round(total_placed / max(total, 1) * 100, 1),
        "by_confidence": dist,
    }


def get_reasoning_feed(
    db_path: Path,
    *,
    suggestions_limit: int = 8,
    thesis_checks_limit: int = 10,
    reflections_limit: int = 8,
) -> dict[str, Any]:
    """Combined reasoning feed for the frontend — suggestions with rationale,
    thesis monitor actions, and self-reflections, all ordered newest-first.
    Zero extra LLM calls; purely reads stored data.
    """
    init_db(db_path)
    history = get_history(db_path, limit=suggestions_limit, offset=0)
    suggestions = []
    for row in history.get("items", []):
        suggestions.append({
            "suggestion_id": row.get("suggestion_id"),
            "created_utc": row.get("created_utc"),
            "side": row.get("side"),
            "trigger_family": row.get("trigger_family") or ((row.get("placed_order") or {}).get("trigger_family")),
            "trigger_reason": row.get("trigger_reason") or ((row.get("placed_order") or {}).get("trigger_reason")),
            "thesis_fingerprint": row.get("thesis_fingerprint") or ((row.get("placed_order") or {}).get("thesis_fingerprint")),
            "decision": row.get("decision"),
            "conviction_rung": row.get("conviction_rung"),
            "skip_reason": row.get("skip_reason"),
            "trade_thesis": row.get("trade_thesis"),
            "whats_different": row.get("whats_different"),
            "why_not_stop": row.get("why_not_stop"),
            "zone_memory_read": row.get("zone_memory_read"),
            "repeat_trade_case": row.get("repeat_trade_case"),
            "planned_rr_estimate": row.get("planned_rr_estimate"),
            "low_rr_edge": row.get("low_rr_edge"),
            "timeframe_alignment": row.get("timeframe_alignment"),
            "countertrend_edge": row.get("countertrend_edge"),
            "trigger_fit": row.get("trigger_fit"),
            "why_trade_despite_weakness": row.get("why_trade_despite_weakness"),
            "custom_exit_plan": row.get("custom_exit_plan"),
            "requested_price": row.get("requested_price"),
            "price": row.get("limit_price"),
            "lots": row.get("lots"),
            "quality": row.get("confidence"),
            "rationale": row.get("rationale"),
            "exit_plan": row.get("exit_plan"),
            "exit_strategy": row.get("exit_strategy"),
            "action": row.get("action"),
            "outcome_status": row.get("outcome_status"),
            "win_loss": row.get("win_loss"),
            "pips": row.get("pips"),
            "pnl": row.get("pnl"),
            "features": row.get("features"),
            "max_adverse_pips": row.get("max_adverse_pips"),
            "max_favorable_pips": row.get("max_favorable_pips"),
            "mae_mfe_estimated": row.get("mae_mfe_estimated"),
        })

    thesis_checks = get_recent_thesis_checks(db_path, limit=thesis_checks_limit)
    reflections = get_reflections(db_path, limit=reflections_limit)
    return {
        "suggestions": suggestions,
        "thesis_checks": thesis_checks,
        "reflections": reflections,
    }


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
    current_ctx: Optional[dict[str, Any]] = None,
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
    top_edit_fields = _top_edited_fields(rows, limit=4)
    current_labels = _context_labels_from_live_ctx(current_ctx) if current_ctx else {}
    matched_rows, matched_keys = _select_matching_rows(rows, current_labels)
    matched_summary = _summarize_rows(matched_rows) if matched_rows else None
    recent_examples = _format_recent_examples(matched_rows or rows, max_items=max_recent_examples)
    exit_summary = _exit_strategy_summary(matched_rows or rows, limit=3)
    directional_summary = _directional_outcome_summary(matched_rows or rows)
    autonomous_summary = _autonomous_outcome_summary(matched_rows or rows)

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
    if matched_rows and matched_summary:
        lines.append(
            "Matched analog cohort: "
            f"{_format_match_description(current_labels, matched_keys)} "
            f"({len(matched_rows)} similar suggestions in sample)."
        )
        lines.append(
            "Matched analog outcomes: "
            f"{matched_summary['placed']} placed, {matched_summary['closed']} closed, "
            f"win rate {matched_summary['win_rate_closed_pct']:.1f}%, "
            f"avg pnl {matched_summary['avg_pnl_closed']:.2f}, avg pips {matched_summary['avg_pips_closed']:.2f}, "
            f"fill rate {matched_summary['fill_rate_after_placement_pct']:.1f}%."
        )
    elif current_labels:
        lines.append(
            "Matched analog cohort: no close condition match found for the current context; "
            "falling back to broad AI history."
        )
    if directional_summary:
        lines.append(f"Directional edge in this cohort: {directional_summary}.")
    if top_edit_fields:
        lines.append(f"Most-edited fields before placement/rejection: {', '.join(top_edit_fields)}.")
    if exit_summary:
        lines.append(f"Exit strategy results in this cohort: {exit_summary}.")
    if autonomous_summary:
        lines.append(f"Autonomous-only outcomes in this cohort: {autonomous_summary}.")
    if recent_examples:
        lines.append("Recent matched examples:" if matched_rows else "Recent examples:")
        lines.extend(f"  - {line}" for line in recent_examples)
    lines.append(
        "Learning rule: prefer setups that align with what has been filling and closing well; be cautious with patterns that were often edited, rejected, cancelled, or closed poorly."
    )
    return "\n".join(lines)


def get_open_filled_positions(db_path: Path) -> list[dict[str, Any]]:
    """Suggestions that were placed and filled but haven't closed yet.

    Used by the autonomous correlation veto to avoid stacking same-side trades
    near the same level. Includes both autonomous and manually-placed Fillmore
    suggestions (they share the DB) — both should count as "in book."
    """
    init_db(db_path)
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            SELECT * FROM ai_suggestions
            WHERE action = 'placed'
              AND outcome_status = 'filled'
              AND closed_at IS NULL
            ORDER BY filled_at DESC
            """
        )
        return [_deserialize_row(dict(r)) for r in cur.fetchall()]


def get_recent_suggestions_since(db_path: Path, since_iso: str) -> list[dict[str, Any]]:
    """All suggestion rows created since `since_iso`, newest first.

    Used by the autonomous same-setup dedupe gate to detect repeat-fires.
    """
    return _load_rows_since(db_path, since_iso)


def build_autonomous_today_block(
    db_path: Path,
    *,
    max_items: int = 10,
    pip_size: float = 0.01,
) -> str:
    """One-shot 'what your autonomous self has done today' block for the prompt.

    Distinct from the learning memory block — this is *today only*, ordered
    chronologically, and surfaces side+price+outcome explicitly so the LLM can
    notice repeat-fade patterns that the cohort-summary view masks.
    """
    init_db(db_path)
    today_utc_midnight = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    rows = _load_rows_since(db_path, today_utc_midnight.isoformat())
    rows = [_deserialize_row(r) if "market_snapshot" not in r else r for r in rows]

    header = "=== AUTONOMOUS RUN TODAY (your own recent suggestions) ==="
    if not rows:
        return f"{header}\nNo prior autonomous suggestions today. Tape is fresh — be selective."

    # Show oldest -> newest in the printed log so the LLM can read it as a story.
    rows = list(reversed(rows[: max(1, max_items)]))
    pip = pip_size or 0.01
    lines = [header]
    pnl_today = 0.0
    closed_today = 0
    for r in rows:
        ts_raw = str(r.get("created_utc") or "")
        ts = ts_raw[11:16] if len(ts_raw) >= 16 else ts_raw
        side = str(r.get("side") or "?").upper()
        price = r.get("limit_price")
        lots = r.get("lots") or 0
        action = str(r.get("action") or "none")
        outcome = str(r.get("outcome_status") or "")
        win_loss = str(r.get("win_loss") or "")
        pips = r.get("pips")
        pnl = r.get("pnl")

        if r.get("closed_at") and pnl is not None:
            closed_today += 1
            pnl_today += _to_float(pnl)

        if action == "rejected":
            tail = "REJECTED by user"
        elif action != "placed":
            tail = f"NOT PLACED (lots={_to_float(lots):.0f})"
        elif win_loss in ("win", "loss", "breakeven"):
            tail = f"CLOSED {win_loss} ({_to_float(pips):+.1f}p, ${_to_float(pnl):+.2f})"
        elif outcome == "filled":
            tail = "FILLED — still open"
        elif outcome in ("cancelled", "expired"):
            tail = f"{outcome.upper()} (limit never filled)"
        elif outcome == "":
            tail = f"placed (waiting for fill, {conf})"
        else:
            tail = f"{outcome} ({conf})"

        price_str = f"@{_to_float(price):.3f}" if price is not None else "@?"
        lines.append(f"  {ts}Z  {side:4s} {price_str}  -> {tail}")

    if closed_today > 0:
        lines.append(f"Today closed P&L (autonomous + manual AI): ${pnl_today:+.2f} across {closed_today} closed trade(s).")
    lines.append(
        "Self-coaching rule: same-zone continuation is valid when the zone is working. "
        "If the same side+level keeps firing without working, the tape is rejecting that thesis — pass instead of stacking."
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
    out["features"] = _json_obj(out.pop("features_json", None))
    out["edited_fields"] = _json_obj(out.pop("edited_fields_json", None))
    out["placed_order"] = _json_obj(out.pop("placed_order_json", None))
    out["exit_params"] = _json_obj(out.pop("exit_params_json", None))
    out["custom_exit_plan"] = _json_obj(out.pop("custom_exit_plan_json", None))
    return out


def _win_loss_from_pnl_or_pips(*, pnl: Any, pips: Any) -> str | None:
    pnl_f = _to_float(pnl)
    if pnl_f is not None:
        if pnl_f > 0:
            return "win"
        if pnl_f < 0:
            return "loss"
        return "breakeven"
    pips_f = _to_float(pips)
    if pips_f is not None:
        if pips_f > 0:
            return "win"
        if pips_f < 0:
            return "loss"
        return "breakeven"
    return None


def _deserialize_thesis_check(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["execution_succeeded"] = bool(int(out.get("execution_succeeded") or 0))
    out["custom_exit"] = bool(int(out.get("custom_exit") or 0))
    return out


def _deserialize_reflection(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["autonomous"] = bool(int(out.get("autonomous") or 0))
    return out


def upsert_performance_stats(
    db_path: Path,
    *,
    profile: str,
    stats_key: str,
    values: dict[str, Any],
) -> None:
    init_db(db_path)
    payload = {
        "profile": str(profile),
        "stats_key": str(stats_key),
        "trade_count": int(values.get("trade_count") or 0),
        "closed_count": int(values.get("closed_count") or 0),
        "win_rate": values.get("win_rate"),
        "avg_win_pips": values.get("avg_win_pips"),
        "avg_loss_pips": values.get("avg_loss_pips"),
        "net_pnl": values.get("net_pnl"),
        "profit_factor": values.get("profit_factor"),
        "avg_hold_minutes": values.get("avg_hold_minutes"),
        "fill_rate_limits": values.get("fill_rate_limits"),
        "avg_time_to_fill_sec": values.get("avg_time_to_fill_sec"),
        "avg_fill_vs_requested_pips": values.get("avg_fill_vs_requested_pips"),
        "thesis_intervention_rate": values.get("thesis_intervention_rate"),
        "avg_mae_pips": values.get("avg_mae_pips"),
        "avg_mfe_pips": values.get("avg_mfe_pips"),
        "win_rate_by_confidence_json": values.get("win_rate_by_confidence_json"),
        "win_rate_by_side_json": values.get("win_rate_by_side_json"),
        "win_rate_by_session_json": values.get("win_rate_by_session_json"),
        "prompt_version_breakdown_json": values.get("prompt_version_breakdown_json"),
        "updated_utc": str(values.get("updated_utc") or _now_utc_iso()),
    }
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO ai_performance_stats (
                profile, stats_key, trade_count, closed_count, win_rate,
                avg_win_pips, avg_loss_pips, net_pnl, profit_factor, avg_hold_minutes,
                fill_rate_limits, avg_time_to_fill_sec, avg_fill_vs_requested_pips,
                thesis_intervention_rate, avg_mae_pips, avg_mfe_pips,
                win_rate_by_confidence_json, win_rate_by_side_json, win_rate_by_session_json,
                prompt_version_breakdown_json, updated_utc
            ) VALUES (
                :profile, :stats_key, :trade_count, :closed_count, :win_rate,
                :avg_win_pips, :avg_loss_pips, :net_pnl, :profit_factor, :avg_hold_minutes,
                :fill_rate_limits, :avg_time_to_fill_sec, :avg_fill_vs_requested_pips,
                :thesis_intervention_rate, :avg_mae_pips, :avg_mfe_pips,
                :win_rate_by_confidence_json, :win_rate_by_side_json, :win_rate_by_session_json,
                :prompt_version_breakdown_json, :updated_utc
            )
            ON CONFLICT(profile, stats_key) DO UPDATE SET
                trade_count=excluded.trade_count,
                closed_count=excluded.closed_count,
                win_rate=excluded.win_rate,
                avg_win_pips=excluded.avg_win_pips,
                avg_loss_pips=excluded.avg_loss_pips,
                net_pnl=excluded.net_pnl,
                profit_factor=excluded.profit_factor,
                avg_hold_minutes=excluded.avg_hold_minutes,
                fill_rate_limits=excluded.fill_rate_limits,
                avg_time_to_fill_sec=excluded.avg_time_to_fill_sec,
                avg_fill_vs_requested_pips=excluded.avg_fill_vs_requested_pips,
                thesis_intervention_rate=excluded.thesis_intervention_rate,
                avg_mae_pips=excluded.avg_mae_pips,
                avg_mfe_pips=excluded.avg_mfe_pips,
                win_rate_by_confidence_json=excluded.win_rate_by_confidence_json,
                win_rate_by_side_json=excluded.win_rate_by_side_json,
                win_rate_by_session_json=excluded.win_rate_by_session_json,
                prompt_version_breakdown_json=excluded.prompt_version_breakdown_json,
                updated_utc=excluded.updated_utc
            """,
            payload,
        )
        conn.commit()


def get_performance_stats(db_path: Path) -> dict[str, dict[str, Any]]:
    init_db(db_path)
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            SELECT *
            FROM ai_performance_stats
            ORDER BY stats_key
            """
        )
        return {str(row["stats_key"]): dict(row) for row in cur.fetchall()}


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


def _context_labels_from_live_ctx(ctx: dict[str, Any] | None) -> dict[str, str]:
    if not isinstance(ctx, dict):
        return {}
    snap = _extract_market_snapshot(ctx)
    return _labels_from_snapshot(snap)


def _select_matching_rows(
    rows: list[dict[str, Any]],
    current_labels: dict[str, str],
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    if not rows or not current_labels:
        return ([], ())

    ordered_keys = (
        "session",
        "macro_bias",
        "vol_label",
        "h1_regime",
        "m5_regime",
        "structure_context",
    )
    available_keys = tuple(key for key in ordered_keys if current_labels.get(key))
    if not available_keys:
        return ([], ())

    specs: list[tuple[str, ...]] = [
        tuple(key for key in ("session", "macro_bias", "vol_label", "h1_regime", "m5_regime", "structure_context") if current_labels.get(key)),
        tuple(key for key in ("session", "macro_bias", "vol_label", "h1_regime", "m5_regime") if current_labels.get(key)),
        tuple(key for key in ("session", "macro_bias", "vol_label", "h1_regime") if current_labels.get(key)),
        tuple(key for key in ("session", "macro_bias", "vol_label", "structure_context") if current_labels.get(key)),
        tuple(key for key in ("session", "macro_bias", "vol_label") if current_labels.get(key)),
        tuple(key for key in ("session", "macro_bias", "h1_regime") if current_labels.get(key)),
        tuple(key for key in ("session", "macro_bias") if current_labels.get(key)),
        tuple(key for key in ("session", "vol_label") if current_labels.get(key)),
        tuple(key for key in ("macro_bias", "vol_label") if current_labels.get(key)),
        tuple(key for key in ("session",) if current_labels.get(key)),
        tuple(key for key in ("macro_bias",) if current_labels.get(key)),
        tuple(key for key in ("vol_label",) if current_labels.get(key)),
    ]
    specs = [spec for spec in specs if spec]

    best_nonempty: tuple[list[dict[str, Any]], tuple[str, ...]] | None = None
    for spec in specs:
        matched = []
        for row in rows:
            labels = _snapshot_context(row)
            if all(labels.get(key) == current_labels.get(key) for key in spec):
                matched.append(row)
        if matched and best_nonempty is None:
            best_nonempty = (matched, spec)
        if len(matched) >= 3:
            return (matched, spec)
    if best_nonempty is not None:
        return best_nonempty
    return ([], ())


def _format_match_description(current_labels: dict[str, str], keys: tuple[str, ...]) -> str:
    if not keys:
        return "broad history fallback"
    labels = {
        "session": "session",
        "macro_bias": "bias",
        "vol_label": "vol",
        "h1_regime": "H1",
        "m5_regime": "M5",
        "structure_context": "structure",
    }
    return ", ".join(f"{labels[key]}={current_labels.get(key)}" for key in keys if current_labels.get(key))


def _exit_strategy_summary(rows: list[dict[str, Any]], limit: int = 3) -> str:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strat = _placed_exit_strategy(row)
        if strat:
            grouped[strat].append(row)
    ranked: list[tuple[str, list[dict[str, Any]]]] = []
    for strat, strat_rows in grouped.items():
        closed_rows = [row for row in strat_rows if row.get("closed_at") and row.get("pnl") is not None]
        ranked.append((strat, closed_rows if closed_rows else strat_rows))
    if not ranked:
        return ""
    ranked.sort(
        key=lambda item: (
            len([row for row in item[1] if row.get("closed_at") and row.get("pnl") is not None]),
            _safe_avg([_to_float(row.get("pips")) for row in item[1] if row.get("pips") is not None]),
            len(item[1]),
        ),
        reverse=True,
    )
    parts: list[str] = []
    for strat, strat_rows in ranked[: max(1, limit)]:
        closed_rows = [row for row in strat_rows if row.get("closed_at") and row.get("pnl") is not None]
        rows_for_metrics = closed_rows if closed_rows else strat_rows
        avg_pips = _safe_avg([_to_float(row.get("pips")) for row in rows_for_metrics if row.get("pips") is not None])
        wins = len([row for row in closed_rows if str(row.get("win_loss") or "") == "win"])
        descriptor = (
            f"{strat} ({len(closed_rows)} closed, { _safe_rate(wins, len(closed_rows)):.0f}% wins, avg {avg_pips:+.1f}p)"
            if closed_rows
            else f"{strat} ({len(rows_for_metrics)} seen, no closed outcomes yet)"
        )
        parts.append(descriptor)
    return ", ".join(parts)


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
    return _labels_from_snapshot(raw)


def _labels_from_snapshot(raw: dict[str, Any]) -> dict[str, Any]:
    session = _json_obj(raw.get("session"))
    macro_bias = _json_obj(raw.get("macro_bias"))
    volatility = _json_obj(raw.get("volatility"))
    technicals = _json_obj(raw.get("technicals"))
    h1 = _json_obj(technicals.get("H1"))
    m5 = _json_obj(technicals.get("M5"))
    order_book = _json_obj(raw.get("order_book"))
    return {
        "session": _normalize_label(session.get("overlap")) or _normalize_label_list(session.get("active_sessions")),
        "macro_bias": _normalize_label(macro_bias.get("combined_bias")),
        "vol_label": _normalize_label(volatility.get("label")),
        "h1_regime": _normalize_label(h1.get("regime")),
        "m5_regime": _normalize_label(m5.get("regime")),
        "structure_context": _structure_context_from_order_book(order_book),
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


def _is_autonomous_row(row: dict[str, Any]) -> bool:
    return is_autonomous_suggestion_row(row)


def _directional_outcome_summary(rows: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for side in ("buy", "sell"):
        side_rows = [row for row in rows if str(row.get("side") or "").lower() == side]
        closed_rows = [row for row in side_rows if row.get("closed_at") and row.get("pnl") is not None]
        if not closed_rows:
            continue
        avg_pips = _safe_avg([_to_float(row.get("pips")) for row in closed_rows if row.get("pips") is not None])
        wins = len([row for row in closed_rows if str(row.get("win_loss") or "") == "win"])
        parts.append(
            f"{side.upper()} {len(closed_rows)} closed, { _safe_rate(wins, len(closed_rows)):.0f}% wins, avg {avg_pips:+.1f}p"
        )
    return "; ".join(parts)


def _autonomous_outcome_summary(rows: list[dict[str, Any]]) -> str:
    autonomous_rows = [row for row in rows if _is_autonomous_row(row)]
    if not autonomous_rows:
        return ""
    summary = _summarize_rows(autonomous_rows)
    return (
        f"{summary['placed']} placed, {summary['closed']} closed, "
        f"win rate {summary['win_rate_closed_pct']:.1f}%, avg {summary['avg_pips_closed']:+.2f}p, "
        f"fill rate {summary['fill_rate_after_placement_pct']:.1f}%"
    )


def _structure_context_from_order_book(order_book: dict[str, Any]) -> str:
    support_pips = _positive_float(order_book.get("nearest_support_distance_pips"))
    resistance_pips = _positive_float(order_book.get("nearest_resistance_distance_pips"))
    threshold_pips = 12.0
    near_support = support_pips is not None and support_pips <= threshold_pips
    near_resistance = resistance_pips is not None and resistance_pips <= threshold_pips
    if near_support and near_resistance:
        return "between_levels"
    if near_support:
        return "near_support"
    if near_resistance:
        return "near_resistance"
    if support_pips is not None or resistance_pips is not None:
        return "midrange"
    return ""


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


def _positive_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out >= 0 else None


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
