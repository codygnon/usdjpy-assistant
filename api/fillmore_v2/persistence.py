"""v2 persistence — additive schema migration on ai_suggestions.

Per Step 1 design decision (#2): add columns to the existing ai_suggestions
table rather than create a new table. New columns are NULL for pre-cutover
rows; do not backfill. v1 and v2 rows are distinguishable by `engine_version`
alone with no joins.

Schema migration is idempotent. Insert function writes a single v2 row per
LLM call (or per blocked gate event when no LLM call is made).
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from . import ENGINE_VERSION
from .snapshot import Snapshot

# Columns this module owns. Each is added if missing on init.
# Listed with type so _ensure_column can be called idempotently.
V2_COLUMNS: tuple[tuple[str, str], ...] = (
    # Identity / versioning
    ("engine_version", "TEXT"),
    ("snapshot_version", "TEXT"),
    ("snapshot_schema_hash", "TEXT"),
    # Blocking telemetry
    ("rendered_prompt", "TEXT"),
    ("rendered_context_json", "TEXT"),
    ("open_lots_buy", "REAL"),
    ("open_lots_sell", "REAL"),
    ("unrealized_pnl_buy", "REAL"),
    ("unrealized_pnl_sell", "REAL"),
    ("pip_value_per_lot", "REAL"),
    ("risk_after_fill_usd", "REAL"),
    ("rolling_20_trade_pnl", "REAL"),
    ("rolling_20_lot_weighted_pnl", "REAL"),
    # Level packet (CLR-required)
    ("level_packet_json", "TEXT"),
    ("level_age_metadata_json", "TEXT"),
    # Gate transparency
    ("gate_candidates_json", "TEXT"),
    # Path telemetry
    ("path_time_mae_mfe_json", "TEXT"),
    # Skip outcomes (additional to existing price_at_expiry/distance_at_expiry_pips)
    ("skip_post_gate_mae_pips", "REAL"),
    ("skip_post_gate_mfe_pips", "REAL"),
    # Volatility regime
    ("volatility_regime", "TEXT"),
    # Pre-decision veto inputs (Step 3 / v2.snap.2)
    ("timeframe_alignment", "TEXT"),
    ("macro_bias", "TEXT"),
    ("catalyst_category", "TEXT"),
    ("active_sessions_json", "TEXT"),
    ("session_overlap", "TEXT"),
    # Halt diagnostics
    ("halt_reason", "TEXT"),
    # Validator audit (Step 2 will populate; column added now to avoid later migration)
    ("validator_overrides_json", "TEXT"),
    # Veto audit (Step 3)
    ("pre_veto_fired_json", "TEXT"),
    # Sizing audit (Step 4)
    ("sizing_inputs_json", "TEXT"),
    ("deterministic_lots", "REAL"),
    # Exit audit (Step 7)
    ("exit_replay_json", "TEXT"),
)


def _ensure_column(conn: sqlite3.Connection, table: str, col: str, col_type: str) -> None:
    cols = {str(r[1]) for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if col in cols:
        return
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")


def init_v2_schema(db_path: Path) -> None:
    """Idempotently add v2 columns to ai_suggestions.

    Caller is expected to have already run api.suggestion_tracker.init_db() so
    the base table exists. Pre-cutover rows keep NULL in v2 columns; v1 rows
    written after this migration will also have NULL because v1 doesn't write
    them. The discriminator is `engine_version`.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        # Verify base table exists; do not create here (suggestion_tracker owns it).
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='ai_suggestions'"
        ).fetchall()
        if not rows:
            raise RuntimeError(
                "ai_suggestions table missing — call api.suggestion_tracker.init_db first"
            )
        for col, col_type in V2_COLUMNS:
            _ensure_column(conn, "ai_suggestions", col, col_type)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ai_suggestions_engine "
            "ON ai_suggestions(engine_version, created_utc)"
        )
        conn.commit()


def _json_or_none(obj: Any) -> Optional[str]:
    if obj is None:
        return None
    if isinstance(obj, str):
        return obj
    return json.dumps(asdict(obj) if hasattr(obj, "__dataclass_fields__") else obj, default=str)


def insert_v2_row(
    db_path: Path,
    *,
    suggestion_id: str,
    profile: str,
    model: str,
    snapshot: Snapshot,
    side: str,
    lots: float,
    limit_price: float,
    decision: str,
    rationale: Optional[str] = None,
    skip_reason: Optional[str] = None,
    halt_reason: Optional[str] = None,
    validator_overrides: Optional[list[dict[str, Any]]] = None,
    pre_vetoes_fired: Optional[list[dict[str, Any]]] = None,
    sizing_inputs: Optional[dict[str, Any]] = None,
    deterministic_lots: Optional[float] = None,
    sl: Optional[float] = None,
    tp: Optional[float] = None,
    trigger_family: Optional[str] = None,
    entry_type: str = "ai_autonomous",
) -> None:
    """Insert one v2 ai_suggestions row.

    Always sets engine_version='v2' so v1/v2 rows are distinguishable without
    joins. Snapshot fields are flattened into typed columns; nested objects
    (level packet, gate candidates) are JSON-encoded into *_json columns.
    """
    gate_candidates_json = (
        json.dumps([asdict(c) for c in snapshot.all_gate_candidates], default=str)
        if snapshot.all_gate_candidates
        else None
    )
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """
            INSERT INTO ai_suggestions (
                suggestion_id, created_utc, profile, model,
                side, limit_price, sl, tp, lots,
                rationale, decision, skip_reason,
                trigger_family, entry_type,
                engine_version, snapshot_version, snapshot_schema_hash, prompt_version,
                rendered_prompt, rendered_context_json,
                open_lots_buy, open_lots_sell,
                unrealized_pnl_buy, unrealized_pnl_sell,
                pip_value_per_lot, risk_after_fill_usd,
                rolling_20_trade_pnl, rolling_20_lot_weighted_pnl,
                level_packet_json, level_age_metadata_json,
                gate_candidates_json, path_time_mae_mfe_json,
                volatility_regime,
                timeframe_alignment, macro_bias, catalyst_category,
                active_sessions_json, session_overlap,
                halt_reason,
                validator_overrides_json, pre_veto_fired_json,
                sizing_inputs_json, deterministic_lots
            ) VALUES (
                :suggestion_id, :created_utc, :profile, :model,
                :side, :limit_price, :sl, :tp, :lots,
                :rationale, :decision, :skip_reason,
                :trigger_family, :entry_type,
                :engine_version, :snapshot_version, :snapshot_schema_hash, :prompt_version,
                :rendered_prompt, :rendered_context_json,
                :open_lots_buy, :open_lots_sell,
                :unrealized_pnl_buy, :unrealized_pnl_sell,
                :pip_value_per_lot, :risk_after_fill_usd,
                :rolling_20_trade_pnl, :rolling_20_lot_weighted_pnl,
                :level_packet_json, :level_age_metadata_json,
                :gate_candidates_json, :path_time_mae_mfe_json,
                :volatility_regime,
                :timeframe_alignment, :macro_bias, :catalyst_category,
                :active_sessions_json, :session_overlap,
                :halt_reason,
                :validator_overrides_json, :pre_veto_fired_json,
                :sizing_inputs_json, :deterministic_lots
            )
            """,
            {
                "suggestion_id": suggestion_id,
                "created_utc": snapshot.created_utc,
                "profile": profile,
                "model": model,
                "side": side,
                "limit_price": limit_price,
                "sl": sl,
                "tp": tp,
                "lots": lots,
                "rationale": rationale,
                "decision": decision,
                "skip_reason": skip_reason,
                "trigger_family": trigger_family,
                "entry_type": entry_type,
                "engine_version": ENGINE_VERSION,
                "snapshot_version": snapshot.snapshot_version,
                "snapshot_schema_hash": snapshot.snapshot_schema_hash,
                "prompt_version": snapshot.prompt_version,
                "rendered_prompt": snapshot.rendered_prompt,
                "rendered_context_json": snapshot.rendered_context_json,
                "open_lots_buy": snapshot.open_lots_buy,
                "open_lots_sell": snapshot.open_lots_sell,
                "unrealized_pnl_buy": snapshot.unrealized_pnl_buy,
                "unrealized_pnl_sell": snapshot.unrealized_pnl_sell,
                "pip_value_per_lot": snapshot.pip_value_per_lot,
                "risk_after_fill_usd": snapshot.risk_after_fill_usd,
                "rolling_20_trade_pnl": snapshot.rolling_20_trade_pnl,
                "rolling_20_lot_weighted_pnl": snapshot.rolling_20_lot_weighted_pnl,
                "level_packet_json": _json_or_none(snapshot.level_packet),
                "level_age_metadata_json": _json_or_none(snapshot.level_age_metadata),
                "gate_candidates_json": gate_candidates_json,
                "path_time_mae_mfe_json": _json_or_none(snapshot.path_time_mae_mfe),
                "volatility_regime": snapshot.volatility_regime,
                "timeframe_alignment": snapshot.timeframe_alignment,
                "macro_bias": snapshot.macro_bias,
                "catalyst_category": snapshot.catalyst_category,
                "active_sessions_json": (
                    json.dumps(snapshot.active_sessions, default=str) if snapshot.active_sessions else None
                ),
                "session_overlap": snapshot.session_overlap,
                "halt_reason": halt_reason,
                "validator_overrides_json": (
                    json.dumps(validator_overrides, default=str) if validator_overrides else None
                ),
                "pre_veto_fired_json": (
                    json.dumps(pre_vetoes_fired, default=str) if pre_vetoes_fired else None
                ),
                "sizing_inputs_json": (
                    json.dumps(sizing_inputs, default=str) if sizing_inputs else None
                ),
                "deterministic_lots": deterministic_lots,
            },
        )
        conn.commit()


def fetch_v2_row(db_path: Path, suggestion_id: str) -> Optional[dict[str, Any]]:
    """Read one v2 row back. Used by the acceptance test for replay-from-logs."""
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM ai_suggestions WHERE suggestion_id = ?", (suggestion_id,)
        ).fetchone()
        return dict(row) if row else None
