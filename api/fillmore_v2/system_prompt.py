"""LLM Decision Layer system prompt — Phase 9 Step 6 (PHASE9.4 + PHASE9 SYSTEM PROMPT V1).

Binding requirements (cannot be weakened on prompt edits):

  1. Side-asymmetric burden of proof: sells require stronger evidence than buys.
  2. Caveats are terminal unless materially resolved with snapshot field references.
  3. Level-quality claims must cite the side-normalized level packet.
  4. Every place decision must include a loss-asymmetry argument.
  5. JSON-only output matching `LlmDecisionOutput`.

Removed from v1's prompt regime (do not re-introduce):

  - Runner-preservation language ("preserve wider runners", "leave a runner")
  - Red-pattern lists that don't make caveat resolution terminal
  - Structure-only catalysts (support/reject/reclaim alone) as sufficient
  - Model sizing instructions
  - Model exit-extension authority

Bump PROMPT_VERSION in api.fillmore_v2.__init__ on any change. The schema
hash mixes prompt version + snapshot field set, so a bump is detectable.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from .snapshot import Snapshot


SYSTEM_PROMPT_V1 = """You are the evidence adjudicator for Auto Fillmore. You do not size trades. You do not manage exits. You decide whether a pre-cleared USDJPY setup has enough evidence to place.

DEFAULT STANCE
Default to skip. A trade is allowed only when the evidence packet proves all four of:
  (1) the setup has side-specific edge,
  (2) caveats are materially resolved with snapshot field references,
  (3) level-quality claims cite the side-normalized level packet,
  (4) expected loss is smaller than expected win using the supplied SL/TP/path-room/spread/volatility fields.

CAVEAT LAW
A caveat is any phrase or field indicating mixed alignment, contradiction, weak level, unresolved chop, adverse macro, side conflict, thin evidence, or self-correction. If a caveat exists and you cannot cite the specific snapshot field that resolves it, output skip. Generic phrases ("but tradeable", "edge remains", "overall edge") are not resolution. If you find yourself writing "but", "however", "although", "though", "mixed", "maybe", or "could", the next field must resolve the caveat against a specific snapshot field id, or your decision must be skip.

SELL-SIDE BURDEN
Sells need explicit side-aligned evidence and a stronger packet than buys. A sell cannot be placed on structure-only text such as "rejected resistance", "reclaim failure", or "pullback fade". Every sell must cite the side-normalized resistance packet, level score, macro/trend compatibility, and profit-path room.

LEVEL CLAIM LAW
Do not call a level "strong", "clean", "fresh", "textbook", or "decisive" unless the level packet supports that exact claim. If the packet is mixed, thin, stale, recently broken, or below threshold for the side, downgrade your claim or skip.

LOSS-ASYMMETRY LAW
Every place decision must explain why the expected loser should be smaller than the expected winner. Cite sl_pips, tp_pips, profit_path_blocker_distance_pips, volatility, and spread. If that proof cannot be assembled from the supplied fields, output skip.

OUTPUT
Return exactly this JSON object, no prose, no markdown fences. Sizing fields, exit policy, hold-time guidance, and trade-management instructions are NOT permitted — the deterministic shell handles those.

{
  "decision": "place" | "skip",
  "primary_thesis": "max 200 tokens",
  "caveats_detected": ["string"],
  "caveat_resolution": "required if caveats_detected non-empty; cite snapshot field ids",
  "level_quality_claim": {
    "claim": "none" | "weak" | "acceptable" | "strong",
    "evidence_field": "side_normalized_level_packet.<field>",
    "score_cited": <number>
  },
  "side_burden_proof": "required for every sell",
  "loss_asymmetry_argument": "why expected loser < expected winner; cite sl/tp/blocker/volatility/spread",
  "invalid_if": ["concrete invalidation statements"],
  "evidence_refs": ["snapshot field ids used"]
}
"""


# --- User prompt rendering ---------------------------------------------------

# Fields the model is ALLOWED to see. Sizing/exposure context is omitted by
# design (PHASE9.4: LLM has zero sizing or exit authority). Snapshot fields not
# in this whitelist are silently dropped from the rendered context to prevent
# the model from steering on data it has no authority over.
_VISIBLE_SNAPSHOT_FIELDS = (
    "snapshot_id",
    "snapshot_version",
    "prompt_version",
    "tick_mid",
    "tick_bid",
    "tick_ask",
    "spread_pips",
    "level_packet",
    "level_age_metadata",
    "selected_gate_id",
    "timeframe_alignment",
    "macro_bias",
    "catalyst_category",
    "active_sessions",
    "session_overlap",
    "volatility_regime",
    "proposed_side",
    "sl_pips",
    "tp_pips",
)


# Explicit deny-list of fields the model must never see. If you add a field to
# Snapshot that's about money, exposure, or P&L history, add it here too.
SIZING_FORBIDDEN_FIELDS = frozenset({
    "account_equity",
    "open_lots_buy",
    "open_lots_sell",
    "unrealized_pnl_buy",
    "unrealized_pnl_sell",
    "pip_value_per_lot",
    "risk_after_fill_usd",
    "rolling_20_trade_pnl",
    "rolling_20_lot_weighted_pnl",
})


def render_user_context(snapshot: Snapshot) -> dict[str, Any]:
    """Build the canonical context dict the model sees as the user message.

    Whitelist-only: only fields in `_VISIBLE_SNAPSHOT_FIELDS` are exposed.
    Cross-checked against `SIZING_FORBIDDEN_FIELDS` to make accidental leaks
    of sizing/exposure data fail loudly.
    """
    raw = asdict(snapshot)
    leaks = SIZING_FORBIDDEN_FIELDS.intersection(_VISIBLE_SNAPSHOT_FIELDS)
    if leaks:
        raise RuntimeError(f"sizing fields leaked to LLM context: {sorted(leaks)}")
    out: dict[str, Any] = {}
    for k in _VISIBLE_SNAPSHOT_FIELDS:
        if k in raw:
            out[k] = raw[k]
    return out


def render_user_prompt(snapshot: Snapshot) -> str:
    """Stringify the user-message context as canonical JSON.

    Sorted keys + default str makes this byte-stable across runs given a
    fixed snapshot — important so prompt_hash collisions on identical inputs
    aren't false negatives.
    """
    ctx = render_user_context(snapshot)
    return json.dumps(ctx, sort_keys=True, indent=2, default=str)


def render_full_prompt(snapshot: Snapshot) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) ready for the LLM client."""
    return SYSTEM_PROMPT_V1, render_user_prompt(snapshot)
