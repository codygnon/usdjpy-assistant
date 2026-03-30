from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

from core.phase3_ownership_core import compute_phase3_ownership_audit_for_data
from core.phase3_overlay_resolver import build_phase3_overlay_state
from core.phase3_package_spec import (
    PHASE3_DEFENDED_PRESET_ID,
    load_phase3_package_spec,
    phase3_package_spec_to_dict,
)


@dataclass(frozen=True)
class Phase3ExitPolicy:
    label: str
    tp1_r: float | None = None
    be_offset_pips: float | None = None
    tp2_r: float | None = None


@dataclass(frozen=True)
class Phase3DecisionEnvelope:
    generated_at_utc: str
    package_id: str
    preset_id: str | None
    session: str | None
    strategy_tag: str | None
    strategy_family: str | None
    ownership_cell: str | None
    attempted: bool
    placed: bool
    blocking_filter_ids: list[str] = field(default_factory=list)
    reason: str | None = None
    side: str | None = None
    size_units: int | None = None
    entry_price: float | None = None
    sl_price: float | None = None
    tp1_price: float | None = None
    exit_policy: Phase3ExitPolicy | None = None
    attribution: dict[str, Any] = field(default_factory=dict)
    raw_decision: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Phase3ParityDiff:
    matches: bool
    mismatches: list[str] = field(default_factory=list)


class ReplayAdapter:
    def __init__(self, *, equity: float = 100000.0, balance: float | None = None) -> None:
        self._next_order_id = 1
        self._account = SimpleNamespace(balance=float(balance if balance is not None else equity), equity=float(equity), margin_used=0.0)

    def is_demo(self) -> bool:
        return True

    def get_account_info(self):
        return self._account

    def place_order(self, *, symbol, side, lots, stop_price, target_price, comment):
        fill_price = None
        self._next_order_id += 1
        return SimpleNamespace(
            order_id=self._next_order_id,
            deal_id=self._next_order_id,
            order_retcode=0,
            fill_price=fill_price,
            symbol=symbol,
            side=side,
            lots=lots,
            stop_price=stop_price,
            target_price=target_price,
            comment=comment,
        )

    def get_position_id_from_order(self, order_id: int) -> int:
        return int(order_id)


class ReplayStore:
    def get_trades_for_date(self, profile_name: str, date_str: str):
        return []

    def list_open_trades(self, profile_name: str):
        return []


def _parse_strategy(strategy_tag: str | None) -> tuple[str | None, str | None]:
    tag = str(strategy_tag or "")
    if "@" in tag:
        base, _, cell = tag.partition("@")
    else:
        base, cell = tag, None
    family = None
    if base.startswith("phase3:v14"):
        family = "v14"
    elif base.startswith("phase3:london_v2_d"):
        family = "london_v2_d"
    elif base.startswith("phase3:london_v2_arb"):
        family = "london_v2_arb"
    elif base.startswith("phase3:v44_ny"):
        family = "v44_ny"
    return family, cell


def _exit_policy_for_strategy(strategy_tag: str | None, sizing_cfg: dict[str, Any]) -> Phase3ExitPolicy | None:
    family, _ = _parse_strategy(strategy_tag)
    if family == "london_v2_d":
        cfg = dict((sizing_cfg or {}).get("london_v2") or {})
        return Phase3ExitPolicy(
            label="L1 defended exit override",
            tp1_r=float(cfg.get("d_tp1_r", 0.0)),
            be_offset_pips=float(cfg.get("d_be_offset_pips", 0.0)),
            tp2_r=float(cfg.get("d_tp2_r", 0.0)),
        )
    if family == "london_v2_arb":
        cfg = dict((sizing_cfg or {}).get("london_v2") or {})
        return Phase3ExitPolicy(
            label="London ARB exit",
            tp1_r=float(cfg.get("arb_tp1_r", 0.0)),
            be_offset_pips=float(cfg.get("arb_be_offset_pips", 0.0)),
            tp2_r=float(cfg.get("arb_tp2_r", 0.0)),
        )
    if family == "v44_ny":
        return Phase3ExitPolicy(label="V44 session exit")
    if family == "v14":
        return Phase3ExitPolicy(label="V14 mean-reversion exit")
    return None


def _blocking_filters_from_reason(reason: str | None) -> list[str]:
    text = str(reason or "")
    if "blocks=" not in text:
        return []
    blocks = text.split("blocks=", 1)[1]
    for stop in (" | ", "\n", "\t"):
        if stop in blocks:
            blocks = blocks.split(stop, 1)[0]
    return [part.strip() for part in blocks.split(",") if part.strip()]


def normalize_phase3_decision_envelope(
    *,
    exec_result: dict[str, Any],
    policy,
    preset_id: str | None,
    sizing_config: dict[str, Any],
    ownership_audit: dict[str, Any] | None = None,
) -> Phase3DecisionEnvelope:
    decision = exec_result.get("decision")
    attempted = bool(getattr(decision, "attempted", False))
    placed = bool(getattr(decision, "placed", False))
    reason = str(getattr(decision, "reason", "") or "") or None
    side = getattr(decision, "side", None)
    strategy_tag = exec_result.get("strategy_tag")
    strategy_family, ownership_cell = _parse_strategy(strategy_tag)
    session = None
    if strategy_family == "v14":
        session = "tokyo"
    elif strategy_family and strategy_family.startswith("london_v2"):
        session = "london"
    elif strategy_family == "v44_ny":
        session = "ny"
    if not ownership_cell and ownership_audit:
        ownership_cell = ownership_audit.get("ownership_cell")

    spec = load_phase3_package_spec(preset_id=preset_id)
    return Phase3DecisionEnvelope(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        package_id=spec.package_id,
        preset_id=preset_id,
        session=session,
        strategy_tag=strategy_tag,
        strategy_family=strategy_family,
        ownership_cell=ownership_cell,
        attempted=attempted,
        placed=placed,
        blocking_filter_ids=_blocking_filters_from_reason(reason),
        reason=reason,
        side=side,
        size_units=exec_result.get("units"),
        entry_price=exec_result.get("entry_price"),
        sl_price=exec_result.get("sl_price"),
        tp1_price=exec_result.get("tp1_price"),
        exit_policy=_exit_policy_for_strategy(strategy_tag, sizing_config),
        attribution={
            "ownership_audit": ownership_audit or {},
            "package_spec": phase3_package_spec_to_dict(spec),
        },
        raw_decision={
            "reason": reason,
            "attempted": attempted,
            "placed": placed,
            "strategy_tag": strategy_tag,
        },
    )


def evaluate_phase3_bar(
    *,
    adapter,
    profile,
    log_dir,
    policy,
    context,
    data_by_tf: dict[str, Any],
    tick,
    mode: str,
    phase3_state: dict[str, Any],
    store=None,
    sizing_config: dict[str, Any] | None = None,
    is_new_m1: bool = True,
    preset_id: str | None = None,
) -> dict[str, Any]:
    from core.phase3_integrated_engine import (
        execute_phase3_integrated_policy_demo_only,
        load_phase3_sizing_config,
    )

    spec = load_phase3_package_spec(preset_id=preset_id)
    if sizing_config is not None:
        effective_cfg = sizing_config
    else:
        try:
            effective_cfg = load_phase3_sizing_config(preset_id=preset_id)
        except TypeError:
            effective_cfg = load_phase3_sizing_config()
    # Prefer the canonical spec projection when present so both live and replay
    # resolve the defended package the same way.
    if spec.runtime_overrides:
        effective_cfg = _deep_merge(effective_cfg, spec.runtime_overrides)

    pip_size = float(getattr(profile, "pip_size", 0.01) or 0.01)
    ownership_audit = compute_phase3_ownership_audit_for_data(data_by_tf, pip_size)
    overlay_state = build_phase3_overlay_state(effective_cfg)

    exec_result = execute_phase3_integrated_policy_demo_only(
        adapter=adapter,
        profile=profile,
        log_dir=log_dir,
        policy=policy,
        context=context,
        data_by_tf=data_by_tf,
        tick=tick,
        mode=mode,
        phase3_state=phase3_state,
        store=store,
        sizing_config=effective_cfg,
        is_new_m1=is_new_m1,
        ownership_audit=ownership_audit,
        overlay_state=overlay_state,
    )
    exec_result["phase3_ownership_audit"] = ownership_audit
    exec_result["phase3_overlay_state"] = overlay_state
    exec_result["decision_envelope"] = asdict(
        normalize_phase3_decision_envelope(
            exec_result=exec_result,
            policy=policy,
            preset_id=preset_id,
            sizing_config=effective_cfg,
            ownership_audit=ownership_audit,
        )
    )
    exec_result["phase3_package_spec"] = phase3_package_spec_to_dict(spec)
    return exec_result


def evaluate_phase3_bar_replay(
    *,
    profile,
    policy,
    data_by_tf: dict[str, Any],
    tick,
    phase3_state: dict[str, Any],
    sizing_config: dict[str, Any] | None = None,
    mode: str = "ARMED_AUTO_DEMO",
    preset_id: str | None = PHASE3_DEFENDED_PRESET_ID,
    equity: float = 100000.0,
    adapter=None,
    store=None,
) -> dict[str, Any]:
    adapter = adapter or ReplayAdapter(equity=equity)
    return evaluate_phase3_bar(
        adapter=adapter,
        profile=profile,
        log_dir=None,
        policy=policy,
        context={},
        data_by_tf=data_by_tf,
        tick=tick,
        mode=mode,
        phase3_state=phase3_state,
        store=store or ReplayStore(),
        sizing_config=sizing_config,
        is_new_m1=True,
        preset_id=preset_id,
    )


def compare_phase3_envelopes(left: dict[str, Any], right: dict[str, Any]) -> Phase3ParityDiff:
    mismatch: list[str] = []
    keys = [
        "session",
        "strategy_tag",
        "strategy_family",
        "ownership_cell",
        "attempted",
        "placed",
        "blocking_filter_ids",
        "size_units",
        "entry_price",
        "sl_price",
        "tp1_price",
    ]
    for key in keys:
        if left.get(key) != right.get(key):
            mismatch.append(f"{key}: {left.get(key)!r} != {right.get(key)!r}")
    if left.get("reason") != right.get("reason"):
        mismatch.append(f"reason: {left.get('reason')!r} != {right.get('reason')!r}")
    if left.get("attribution") != right.get("attribution"):
        mismatch.append("attribution differs")
    left_exit = left.get("exit_policy") or {}
    right_exit = right.get("exit_policy") or {}
    for key in ("label", "tp1_r", "be_offset_pips", "tp2_r"):
        if left_exit.get(key) != right_exit.get(key):
            mismatch.append(f"exit_policy.{key}: {left_exit.get(key)!r} != {right_exit.get(key)!r}")
    return Phase3ParityDiff(matches=not mismatch, mismatches=mismatch)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out
