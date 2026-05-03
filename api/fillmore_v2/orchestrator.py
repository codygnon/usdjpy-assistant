"""v2 Forward-Path Orchestrator — Phase 9 Step 6.

Wires Steps 1-5 into a single callable pipeline. Pure orchestration: every
decision is delegated to the layer module that owns it. The orchestrator's
job is sequencing, audit-record assembly, and persistence.

NOT YET WIRED INTO PRODUCTION. v1 (`api/autonomous_fillmore.py`) remains
the live engine. Step 9 implements the engine flag and live wiring. This
module is callable from tests and the smoke-test script only.

Sequence (each layer is a separate guarded step; first failure short-circuits):

  1. Halt check (Step 1) — if v2 is halted from prior strikes, refuse.
  2. Blocking-field check (Step 1) — register strike or reset; halt at 3.
     The rendered prompt is checked later because this orchestrator renders it
     after gate/pre-veto context is known.
  3. Gate eligibility (Step 5) — pre-pre-veto pass to confirm a primary
     gate could even fire on this snapshot. Records all candidates.
  4. Pre-decision vetoes (Step 3) — V1, V2 with protected bypass.
  5. Gate eligibility (Step 5, take 2) — re-evaluate with pre_veto_summary
     so the audit log shows the gate+veto interaction.
  6. Deterministic sizing (Step 4) — compute lots BEFORE LLM call so
     `risk_after_fill_usd` is known.
  7. Render prompt + context (Step 6 system_prompt).
  8. LLM call (Step 6 llm_client).
  9. Strict JSON parse (Step 2 llm_output_schema). Parse failure → skip.
 10. Post-decision validators (Step 2). Any override → final = skip.
 11. Persist v2 row (Step 1 persistence).
 12. Return decision summary.
"""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from . import ENGINE_VERSION, PROMPT_VERSION, SNAPSHOT_VERSION
from .gates import GateRunSummary, evaluate_all_gates
from .llm_client import LlmCallResult, LlmClient
from .llm_output_schema import LlmDecisionOutput, LlmOutputParseError, parse
from .persistence import insert_v2_row
from .pre_decision_vetoes import PreVetoRunSummary, run_pre_vetoes
from .snapshot import (
    GateCandidate,
    Snapshot,
    check_blocking_fields,
    is_halted,
    register_blocking_result,
    reset_blocking_strikes,
)
from .system_prompt import render_full_prompt
from .telemetry import pip_value_per_lot, risk_after_fill_usd
from .validators import ValidatorRunSummary, run_all
from core.fillmore_v2_sizing import (
    SizingContext,
    SizingResult,
    Stage,
    compute_autonomous_lots,
)


@dataclass
class OrchestrationResult:
    """Audit-grade output capturing every layer's decision."""
    suggestion_id: str
    final_decision: str  # 'place' | 'skip' | 'halt' | 'no_gate' | 'parse_failure'
    reason: str
    snapshot: Snapshot
    halt_active: bool = False
    gates_initial: Optional[GateRunSummary] = None
    pre_veto_summary: Optional[PreVetoRunSummary] = None
    gates_final: Optional[GateRunSummary] = None
    sizing: Optional[SizingResult] = None
    rendered_system_prompt: Optional[str] = None
    rendered_user_prompt: Optional[str] = None
    llm_call: Optional[LlmCallResult] = None
    llm_output: Optional[LlmDecisionOutput] = None
    validator_summary: Optional[ValidatorRunSummary] = None
    deterministic_lots: float = 0.0
    persisted: bool = False

    def to_audit_records(self) -> dict[str, Any]:
        return {
            "suggestion_id": self.suggestion_id,
            "final_decision": self.final_decision,
            "reason": self.reason,
            "halt_active": self.halt_active,
            "gates": (self.gates_final or self.gates_initial).to_audit_records() if (self.gates_final or self.gates_initial) else [],
            "pre_vetoes": self.pre_veto_summary.to_audit_records() if self.pre_veto_summary else [],
            "validators": self.validator_summary.to_audit_records() if self.validator_summary else [],
            "deterministic_lots": self.deterministic_lots,
            "llm_error": (self.llm_call.error if self.llm_call else None),
        }


# --- Helpers -----------------------------------------------------------------

def _new_suggestion_id() -> str:
    return f"v2-{uuid.uuid4().hex[:16]}"


def _build_sizing_context(snapshot: Snapshot, *, stage: Stage) -> SizingContext:
    """Project a Snapshot into a SizingContext.

    Defaults safe values when blocking telemetry is absent (the blocking-field
    check upstream should have caught this; defaults exist so a degenerate
    sizing call returns lots=0 cleanly rather than crashing).
    """
    pkt = snapshot.level_packet
    is_protected_buy_clr = (
        snapshot.proposed_side == "buy"
        and pkt is not None
        and pkt.side == "buy_support"
        and pkt.level_quality_score >= 70.0
    )
    return SizingContext(
        account_equity=snapshot.account_equity or 0.0,
        sl_pips=snapshot.sl_pips or 0.0,
        pip_value_per_lot=snapshot.pip_value_per_lot or 0.0,
        proposed_side=snapshot.proposed_side or "buy",  # type: ignore[arg-type]
        open_lots_buy=snapshot.open_lots_buy or 0.0,
        open_lots_sell=snapshot.open_lots_sell or 0.0,
        rolling_20_trade_pnl=snapshot.rolling_20_trade_pnl or 0.0,
        rolling_20_lot_weighted_pnl=snapshot.rolling_20_lot_weighted_pnl or 0.0,
        risk_after_fill_usd=snapshot.risk_after_fill_usd or 0.0,
        volatility_regime=snapshot.volatility_regime or "unknown",  # type: ignore[arg-type]
        stage=stage,
        forward_100_trade_profit_factor=0.0,  # populated by Stage 4 once telemetry available
        net_pips_100=0.0,
        protected_buy_clr_packet=is_protected_buy_clr,
    )


def _apply_gate_context(snapshot: Snapshot, gate_summary: GateRunSummary) -> None:
    """Copy gate results onto the Snapshot before prompt render/persistence."""
    snapshot.all_gate_candidates = [
        GateCandidate(
            gate_id=c.gate_id,
            score=c.score,
            eligible=c.eligible,
            veto_reason=c.reason_code,
        )
        for c in gate_summary.candidates
    ]
    snapshot.selected_gate_id = (
        gate_summary.selected_gate.gate_id if gate_summary.selected_gate else None
    )


def _sl_tp_prices(snapshot: Snapshot) -> tuple[Optional[float], Optional[float]]:
    if not snapshot.tick_mid or not snapshot.sl_pips or not snapshot.tp_pips:
        return None, None
    pip = 0.01
    if snapshot.proposed_side == "buy":
        return (
            snapshot.tick_mid - snapshot.sl_pips * pip,
            snapshot.tick_mid + snapshot.tp_pips * pip,
        )
    if snapshot.proposed_side == "sell":
        return (
            snapshot.tick_mid + snapshot.sl_pips * pip,
            snapshot.tick_mid - snapshot.tp_pips * pip,
        )
    return None, None


def _persist(
    db_path: Optional[Path],
    *,
    profile: str,
    model: str,
    suggestion_id: str,
    snapshot: Snapshot,
    final_decision: str,
    reason: str,
    sizing: Optional[SizingResult],
    pre_veto_summary: Optional[PreVetoRunSummary],
    validator_summary: Optional[ValidatorRunSummary],
    rationale: Optional[str],
    halt_reason: Optional[str],
) -> bool:
    if db_path is None:
        return False
    sl_price, tp_price = _sl_tp_prices(snapshot)
    insert_v2_row(
        db_path,
        suggestion_id=suggestion_id,
        profile=profile,
        model=model,
        snapshot=snapshot,
        side=snapshot.proposed_side or "buy",
        lots=(sizing.lots if sizing and final_decision == "place" else 0.0),
        limit_price=snapshot.tick_mid or 0.0,
        decision=final_decision,
        rationale=rationale,
        skip_reason=(reason if final_decision != "place" else None),
        halt_reason=halt_reason,
        validator_overrides=(validator_summary.to_audit_records() if validator_summary else None),
        pre_vetoes_fired=(pre_veto_summary.to_audit_records() if pre_veto_summary else None),
        sizing_inputs=(asdict(sizing) if sizing else None),
        deterministic_lots=(sizing.lots if sizing else None),
        sl=sl_price,
        tp=tp_price,
        trigger_family=(validator_summary and None) or (snapshot.selected_gate_id),
    )
    return True


# --- Main entrypoint ---------------------------------------------------------

def run_decision(
    snapshot: Snapshot,
    *,
    llm_client: LlmClient,
    profile_dir: Path,
    db_path: Optional[Path] = None,
    profile: str = "default",
    model: str = "gpt-5.4-mini",
    stage: Stage = "paper",
) -> OrchestrationResult:
    """Run the full v2 pipeline. Pure-ish: side-effects are halt-state writes
    and (if `db_path` provided) a single persistence row.

    Always returns an `OrchestrationResult`. Never raises on LLM failures
    or parse failures — those become `final_decision='skip'` with a logged
    reason. Raises only on programmer error (invalid inputs to the layers).
    """
    suggestion_id = _new_suggestion_id()

    # 1. Halt check
    halted, halt_reason = is_halted(profile_dir)
    if halted:
        return OrchestrationResult(
            suggestion_id=suggestion_id, final_decision="halt",
            reason=f"v2 halted: {halt_reason}", snapshot=snapshot, halt_active=True,
        )

    # 2. Blocking-field check + strike accounting.
    # `rendered_prompt` is a blocking field for persisted call rows, but it
    # cannot exist yet: the prompt includes selected-gate context and is
    # rendered only after gate/pre-veto decisions. Enforce it after rendering
    # instead of counting a false strike here.
    require_clr = (snapshot.selected_gate_id or "").endswith("clr")
    missing = [
        f for f in check_blocking_fields(snapshot, require_clr=require_clr)
        if f not in ("rendered_prompt", "risk_after_fill_usd")
    ]
    halt_now, halt_now_reason = register_blocking_result(profile_dir, missing)
    if missing:
        result = OrchestrationResult(
            suggestion_id=suggestion_id,
            final_decision=("halt" if halt_now else "skip"),
            reason=f"missing_blocking_fields={missing}",
            snapshot=snapshot, halt_active=halt_now,
        )
        _persist(
            db_path, profile=profile, model=model, suggestion_id=suggestion_id,
            snapshot=snapshot, final_decision=result.final_decision, reason=result.reason,
            sizing=None, pre_veto_summary=None, validator_summary=None,
            rationale=None,
            halt_reason=(halt_now_reason if halt_now else f"strike: {missing}"),
        )
        result.persisted = db_path is not None
        return result

    # 3. Gate eligibility (initial pass — no pre-veto context yet)
    gates_initial = evaluate_all_gates(snapshot)
    if not gates_initial.any_eligible:
        _apply_gate_context(snapshot, gates_initial)
        return _finalize_skip_no_gate(
            suggestion_id, snapshot, gates_initial,
            db_path=db_path, profile=profile, model=model,
        )

    # 4. Pre-decision vetoes
    pre_veto_summary = run_pre_vetoes(snapshot, deterministic_lots=None)

    # 5. Gate re-evaluation with pre-veto context (per gates.py module docstring)
    gates_final = evaluate_all_gates(snapshot, pre_veto_summary=pre_veto_summary)
    _apply_gate_context(snapshot, gates_final)

    # CLR-specific blocking fields can only be known after a CLR gate is selected.
    if snapshot.selected_gate_id in ("buy_clr", "sell_clr"):
        clr_missing = [
            f for f in check_blocking_fields(snapshot, require_clr=True)
            if f not in ("rendered_prompt", "risk_after_fill_usd")
        ]
        if clr_missing:
            halt_now, halt_now_reason = register_blocking_result(profile_dir, clr_missing)
            result = OrchestrationResult(
                suggestion_id=suggestion_id,
                final_decision=("halt" if halt_now else "skip"),
                reason=f"missing_clr_blocking_fields={clr_missing}",
                snapshot=snapshot,
                halt_active=halt_now,
                gates_initial=gates_initial,
                pre_veto_summary=pre_veto_summary,
                gates_final=gates_final,
            )
            _persist(
                db_path, profile=profile, model=model, suggestion_id=suggestion_id,
                snapshot=snapshot, final_decision=result.final_decision, reason=result.reason,
                sizing=None, pre_veto_summary=pre_veto_summary, validator_summary=None,
                rationale=None,
                halt_reason=(halt_now_reason if halt_now else f"strike: {clr_missing}"),
            )
            result.persisted = db_path is not None
            return result

    if pre_veto_summary.skip_before_call or not gates_final.any_eligible:
        reason = (
            f"pre_veto_fired={[f.veto_id for f in pre_veto_summary.fires]}"
            if pre_veto_summary.skip_before_call
            else "no_eligible_gate_after_pre_veto"
        )
        return _finalize_skip(
            suggestion_id, snapshot, gates_initial, pre_veto_summary, gates_final,
            reason=reason, db_path=db_path, profile=profile, model=model,
        )

    # 6. Deterministic sizing
    sizing_ctx = _build_sizing_context(snapshot, stage=stage)
    sizing = compute_autonomous_lots(sizing_ctx)
    selected = gates_final.selected_gate
    if selected and selected.sizing_cap_lots is not None:
        sizing = SizingResult(
            **{**asdict(sizing), "lots": min(sizing.lots, selected.sizing_cap_lots)}
        )
    try:
        snapshot.risk_after_fill_usd = risk_after_fill_usd(
            proposed_lots=sizing.lots,
            sl_pips=snapshot.sl_pips or 0.0,
            pip_value_per_lot_usd=snapshot.pip_value_per_lot or 0.0,
        )
    except Exception as e:  # noqa: BLE001 — missing risk means fail closed.
        halt_now, halt_now_reason = register_blocking_result(profile_dir, ["risk_after_fill_usd"])
        result = OrchestrationResult(
            suggestion_id=suggestion_id,
            final_decision=("halt" if halt_now else "skip"),
            reason=f"missing_blocking_fields=['risk_after_fill_usd']: {type(e).__name__}: {e}",
            snapshot=snapshot,
            halt_active=halt_now,
            gates_initial=gates_initial,
            pre_veto_summary=pre_veto_summary,
            gates_final=gates_final,
            sizing=sizing,
            deterministic_lots=sizing.lots,
        )
        _persist(
            db_path, profile=profile, model=model, suggestion_id=suggestion_id,
            snapshot=snapshot, final_decision=result.final_decision, reason=result.reason,
            sizing=sizing, pre_veto_summary=pre_veto_summary, validator_summary=None,
            rationale=None,
            halt_reason=(halt_now_reason if halt_now else "strike: ['risk_after_fill_usd']"),
        )
        result.persisted = db_path is not None
        return result
    if sizing.lots <= 0:
        return _finalize_skip(
            suggestion_id, snapshot, gates_initial, pre_veto_summary, gates_final,
            reason=f"sizing_zero: notes={list(sizing.notes)}",
            sizing=sizing, db_path=db_path, profile=profile, model=model,
        )

    # 7. Render prompt + context
    system_prompt, user_prompt = render_full_prompt(snapshot)
    if not system_prompt.strip() or not user_prompt.strip():
        return _finalize_skip(
            suggestion_id, snapshot, gates_initial, pre_veto_summary, gates_final,
            reason="rendered_prompt_missing",
            sizing=sizing,
            system_prompt=system_prompt, user_prompt=user_prompt,
            db_path=db_path, profile=profile, model=model,
        )

    # 8. LLM call
    llm_call = llm_client.complete_json(
        system_prompt=system_prompt, user_prompt=user_prompt, model=model,
    )
    if llm_call.error:
        return _finalize_skip(
            suggestion_id, snapshot, gates_initial, pre_veto_summary, gates_final,
            reason=f"llm_transport_error: {llm_call.error}",
            sizing=sizing, llm_call=llm_call,
            system_prompt=system_prompt, user_prompt=user_prompt,
            db_path=db_path, profile=profile, model=model,
        )

    # 9. Strict JSON parse
    try:
        llm_output = parse(llm_call.raw_text)
    except LlmOutputParseError as e:
        return _finalize_skip(
            suggestion_id, snapshot, gates_initial, pre_veto_summary, gates_final,
            reason=f"parse_failure: {e}",
            sizing=sizing, llm_call=llm_call,
            system_prompt=system_prompt, user_prompt=user_prompt,
            db_path=db_path, profile=profile, model=model,
            final_decision_override="parse_failure",
        )

    # 10. Post-decision validators
    validator_summary = run_all(llm_output, snapshot)

    # 11. Decide final + persist
    final_decision = validator_summary.final_decision
    reason = (
        "place"
        if final_decision == "place"
        else f"validator_overrides={[o.validator_id for o in validator_summary.overrides]}"
        if validator_summary.any_fired
        else "model_skip"
    )

    result = OrchestrationResult(
        suggestion_id=suggestion_id,
        final_decision=final_decision, reason=reason, snapshot=snapshot,
        gates_initial=gates_initial, pre_veto_summary=pre_veto_summary,
        gates_final=gates_final, sizing=sizing,
        rendered_system_prompt=system_prompt, rendered_user_prompt=user_prompt,
        llm_call=llm_call, llm_output=llm_output, validator_summary=validator_summary,
        deterministic_lots=sizing.lots,
    )

    # Mutate the snapshot's prompt fields just before persist so the row
    # captures exactly what was sent to the model (PHASE9.8 item 10).
    snapshot.rendered_prompt = system_prompt + "\n\n" + user_prompt
    snapshot.rendered_context_json = user_prompt

    _persist(
        db_path, profile=profile, model=model, suggestion_id=suggestion_id,
        snapshot=snapshot, final_decision=final_decision, reason=reason,
        sizing=sizing, pre_veto_summary=pre_veto_summary,
        validator_summary=validator_summary, rationale=llm_call.raw_text,
        halt_reason=None,
    )
    result.persisted = db_path is not None

    # Successful place clears the strike counter
    if final_decision == "place":
        reset_blocking_strikes(profile_dir)

    return result


# --- Internal finalizers (DRY — these call _persist with the right args) ----

def _finalize_skip_no_gate(
    suggestion_id: str, snapshot: Snapshot, gates_initial: GateRunSummary,
    *, db_path: Optional[Path], profile: str, model: str,
) -> OrchestrationResult:
    reason = "no_eligible_gate"
    res = OrchestrationResult(
        suggestion_id=suggestion_id, final_decision="no_gate",
        reason=reason, snapshot=snapshot, gates_initial=gates_initial,
    )
    _persist(
        db_path, profile=profile, model=model, suggestion_id=suggestion_id,
        snapshot=snapshot, final_decision="skip", reason=reason,
        sizing=None, pre_veto_summary=None, validator_summary=None,
        rationale=None, halt_reason=None,
    )
    res.persisted = db_path is not None
    return res


def _finalize_skip(
    suggestion_id: str, snapshot: Snapshot,
    gates_initial: Optional[GateRunSummary],
    pre_veto_summary: Optional[PreVetoRunSummary],
    gates_final: Optional[GateRunSummary],
    *, reason: str,
    sizing: Optional[SizingResult] = None,
    llm_call: Optional[LlmCallResult] = None,
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
    db_path: Optional[Path] = None, profile: str = "default", model: str = "gpt-5.4-mini",
    final_decision_override: Optional[str] = None,
) -> OrchestrationResult:
    final = final_decision_override or "skip"
    res = OrchestrationResult(
        suggestion_id=suggestion_id, final_decision=final, reason=reason,
        snapshot=snapshot, gates_initial=gates_initial,
        pre_veto_summary=pre_veto_summary, gates_final=gates_final,
        sizing=sizing, rendered_system_prompt=system_prompt,
        rendered_user_prompt=user_prompt, llm_call=llm_call,
    )
    if system_prompt and user_prompt:
        snapshot.rendered_prompt = system_prompt + "\n\n" + user_prompt
        snapshot.rendered_context_json = user_prompt
    _persist(
        db_path, profile=profile, model=model, suggestion_id=suggestion_id,
        snapshot=snapshot, final_decision="skip", reason=reason,
        sizing=sizing, pre_veto_summary=pre_veto_summary, validator_summary=None,
        rationale=(llm_call.raw_text if llm_call else None),
        halt_reason=None,
    )
    res.persisted = db_path is not None
    return res
