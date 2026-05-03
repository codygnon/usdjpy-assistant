"""Telemetry & Snapshot Layer — Phase 9 Step 1 (PHASE9.8).

The Phase 8 audit was compromised by missing telemetry. The v2 rebuild builds
this layer FIRST so every subsequent layer logs into structures the next audit
can replay from logs alone.

Field-set authority lives in `Snapshot`. Any change to the captured field set
MUST bump SNAPSHOT_VERSION in api.fillmore_v2.__init__ and is verified by the
schema hash recorded on every row.

Blocking fields (per PHASE9.8): items 1, 2, 5, 10. Without them, autonomous
placement halts. CLR additionally requires items 6 and 7.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from . import ENGINE_VERSION, PROMPT_VERSION, SNAPSHOT_VERSION
from . import state as v2_state

# --- Blocking-field policy ----------------------------------------------------
# Strike definition: one strike is one completed gate poll where a Snapshot was
# successfully built but one or more blocking fields are absent/empty. Snapshot
# construction exceptions are not strikes in this counter; callers must fail
# closed for that poll and record the exception through operational telemetry.
# Per PHASE9.8, missing any blocking field halts placement after 3 consecutive
# strikes. A clean snapshot resets the counter to zero.
BLOCKING_FIELDS = (
    "open_lots_buy",
    "open_lots_sell",
    "risk_after_fill_usd",
    "pip_value_per_lot",
    "rendered_prompt",
    "snapshot_version",
)
CLR_REQUIRED_FIELDS = ("level_packet", "level_age_metadata")
HALT_STRIKE_THRESHOLD = 3
_HALT_STATE_KEY = "snapshot_blocking_strikes"
_HALT_REASON_KEY = "snapshot_halt_reason"


@dataclass
class LevelPacket:
    """Side-normalized level packet — PHASE9.8 item 6.

    Fixes the Phase 8 ambiguity: v1 stored raw support/resistance with no
    side orientation. The LLM had to infer side relevance. v2 normalizes
    against the proposed trade side.
    """
    side: str  # 'buy_support' | 'sell_resistance'
    level_price: float
    level_quality_score: float  # 0-100; thresholds: buy>=70, sell>=85
    distance_pips: float  # signed: + means level is in trade direction
    profit_path_blocker_distance_pips: Optional[float] = None
    structural_origin: Optional[str] = None  # e.g. 'pdh', 'h1_swing_low'


@dataclass
class LevelAgeMetadata:
    """PHASE9.8 item 7. Directly addresses CLR fast failures."""
    level_age_minutes: Optional[float] = None
    touch_count: Optional[int] = None
    broken_then_reclaimed: bool = False
    last_touch_utc: Optional[str] = None


@dataclass
class GateCandidate:
    """Single candidate from the gate layer. PHASE9.8 item 11.

    v1 only logged the selected gate. v2 logs every candidate with its score
    so multi-gate conflicts are auditable.
    """
    gate_id: str
    score: float
    eligible: bool
    veto_reason: Optional[str] = None


@dataclass
class PathTimeMaeMfe:
    """PHASE9.8 item 8. Best-effort initially; required by 50 forward trades."""
    mae_1m: Optional[float] = None
    mfe_1m: Optional[float] = None
    mae_3m: Optional[float] = None
    mfe_3m: Optional[float] = None
    mae_5m: Optional[float] = None
    mfe_5m: Optional[float] = None
    mae_15m: Optional[float] = None
    mfe_15m: Optional[float] = None


@dataclass
class Snapshot:
    """Canonical v2 snapshot. One per gate event.

    Every field captured here is replayable from logs. Adding/removing a field
    requires bumping SNAPSHOT_VERSION; the schema hash on every row catches
    silent drift.
    """
    # --- Identity / versioning (PHASE9.8 item 10) ---
    snapshot_id: str
    created_utc: str
    engine_version: str = ENGINE_VERSION
    snapshot_version: str = SNAPSHOT_VERSION
    prompt_version: str = PROMPT_VERSION
    snapshot_schema_hash: str = ""  # filled by __post_init__

    # --- Market state ---
    tick_mid: Optional[float] = None
    tick_bid: Optional[float] = None
    tick_ask: Optional[float] = None
    spread_pips: Optional[float] = None

    # --- Account & exposure (PHASE9.8 items 1, 4) ---
    account_equity: Optional[float] = None
    open_lots_buy: Optional[float] = None
    open_lots_sell: Optional[float] = None
    unrealized_pnl_buy: Optional[float] = None
    unrealized_pnl_sell: Optional[float] = None

    # --- Sizing inputs (PHASE9.8 items 2, 5) ---
    pip_value_per_lot: Optional[float] = None
    risk_after_fill_usd: Optional[float] = None  # filled post-LLM/pre-order

    # --- Drawdown awareness (PHASE9.8 item 3) ---
    rolling_20_trade_pnl: Optional[float] = None
    rolling_20_lot_weighted_pnl: Optional[float] = None

    # --- Level packet (PHASE9.8 items 6, 7; CLR-required) ---
    level_packet: Optional[LevelPacket] = None
    level_age_metadata: Optional[LevelAgeMetadata] = None

    # --- Gate transparency (PHASE9.8 item 11) ---
    all_gate_candidates: list[GateCandidate] = field(default_factory=list)
    selected_gate_id: Optional[str] = None

    # --- LLM call context (PHASE9.8 item 10) ---
    rendered_prompt: Optional[str] = None  # full final prompt text
    rendered_context_json: Optional[str] = None  # canonical user-message context

    # --- Path telemetry (PHASE9.8 items 8, 12) ---
    path_time_mae_mfe: Optional[PathTimeMaeMfe] = None

    # --- Skip outcomes (PHASE9.8 item 9) ---
    # Populated post-expiry by the feedback layer; absent at capture time.
    skip_price_at_expiry: Optional[float] = None
    skip_distance_pips_at_expiry: Optional[float] = None
    skip_post_gate_mae_pips: Optional[float] = None
    skip_post_gate_mfe_pips: Optional[float] = None

    # --- Volatility regime (sizing input) ---
    volatility_regime: Optional[str] = None  # 'normal' | 'elevated'

    # --- Trade side under consideration ---
    proposed_side: Optional[str] = None  # 'buy' | 'sell'
    sl_pips: Optional[float] = None
    tp_pips: Optional[float] = None

    # --- Pre-decision veto inputs (Step 3 / PHASE9.3) ---
    # Added at v2.snap.2 so V1/V2 vetoes have the inputs they need without
    # reaching into v1's gate primitives. None means "unknown"; a veto that
    # depends on an unknown field cannot fire (fail-open at gate-time).
    timeframe_alignment: Optional[str] = None  # 'aligned_buy'|'aligned_sell'|'mixed'|'neutral'
    macro_bias: Optional[str] = None  # 'bullish'|'bearish'|'neutral'
    catalyst_category: Optional[str] = None  # 'material'|'structure_only'
    active_sessions: list[str] = field(default_factory=list)  # ['tokyo','london','ny']
    session_overlap: Optional[str] = None  # 'tokyo_london'|'london_ny'|None

    def __post_init__(self) -> None:
        if not self.snapshot_schema_hash:
            self.snapshot_schema_hash = compute_schema_hash()

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str, sort_keys=True)


def compute_schema_hash() -> str:
    """Hash of (snapshot field set + prompt version).

    Catches silent prompt drift even when SNAPSHOT_VERSION isn't bumped.
    Stable across runs as long as the dataclass field set and PROMPT_VERSION
    haven't changed.
    """
    fields = sorted(Snapshot.__dataclass_fields__.keys())
    payload = json.dumps(
        {
            "snapshot_version": SNAPSHOT_VERSION,
            "prompt_version": PROMPT_VERSION,
            "fields": fields,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- Blocking-field check & halt logic ---------------------------------------

def check_blocking_fields(snapshot: Snapshot, *, require_clr: bool = False) -> list[str]:
    """Return list of missing blocking field names. Empty = clear to place."""
    missing: list[str] = []
    for f in BLOCKING_FIELDS:
        if getattr(snapshot, f, None) in (None, ""):
            missing.append(f)
    if require_clr:
        for f in CLR_REQUIRED_FIELDS:
            if getattr(snapshot, f, None) is None:
                missing.append(f)
    return missing


def register_blocking_result(
    profile_dir: Path,
    missing: list[str],
) -> tuple[bool, Optional[str]]:
    """Track consecutive missing-blocking-field polls.

    `missing` must come from `check_blocking_fields()` after Snapshot build
    succeeds. Do not pass snapshot-build exceptions here; those are immediate
    fail-closed events, not accumulated missing-field strikes.
    Returns (halt_active, reason).
    """
    state = v2_state.load_state(profile_dir)
    strikes = int(state.get(_HALT_STATE_KEY, 0))
    if missing:
        strikes += 1
        reason = f"missing blocking fields: {','.join(missing)}"
    else:
        strikes = 0
        reason = None
    state[_HALT_STATE_KEY] = strikes
    state[_HALT_REASON_KEY] = reason
    v2_state.save_state(profile_dir, state)
    halt_active = strikes >= HALT_STRIKE_THRESHOLD
    return halt_active, reason if halt_active else None


def reset_blocking_strikes(profile_dir: Path) -> None:
    state = v2_state.load_state(profile_dir)
    state[_HALT_STATE_KEY] = 0
    state[_HALT_REASON_KEY] = None
    v2_state.save_state(profile_dir, state)


def is_halted(profile_dir: Path) -> tuple[bool, Optional[str]]:
    state = v2_state.load_state(profile_dir)
    strikes = int(state.get(_HALT_STATE_KEY, 0))
    if strikes >= HALT_STRIKE_THRESHOLD:
        return True, state.get(_HALT_REASON_KEY)
    return False, None


# --- Constructors -------------------------------------------------------------

def new_snapshot_id() -> str:
    import uuid
    return f"v2-{uuid.uuid4().hex[:16]}"
