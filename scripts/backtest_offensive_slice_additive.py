#!/usr/bin/env python3
from __future__ import annotations

import json
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path("/Users/codygnon/Documents/usdjpy_assistant")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import backtest_merged_integrated_tokyo_london_v2_ny as merged_engine
from scripts import backtest_variant_k_london_cluster as variant_k

STARTING_EQUITY = 100000.0
BASELINE_CACHE_VERSION = "v1"


@dataclass
class BaselineContext:
    dataset: str
    baseline_kept: list[merged_engine.TradeRow]
    baseline_coupled: list[merged_engine.TradeRow]
    baseline_summary: dict[str, Any]
    baseline_meta: dict[str, Any]


@dataclass(frozen=True)
class ConflictPolicy:
    name: str = "default"
    hedging_enabled: bool = False
    allow_internal_overlap: bool = True
    allow_opposite_side_overlap: bool = False
    max_open_offensive: int | None = None
    max_entries_per_day: int | None = None
    margin_model_enabled: bool = False
    margin_leverage: float = 33.3
    margin_buffer_pct: float = 0.0
    max_lot_per_trade: float | None = None


def build_baseline_context(dataset: str) -> BaselineContext:
    dataset_path = Path(dataset).resolve()
    cache_dir = dataset_path.parent / "shadow_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"offensive_slice_baseline_ctx_{dataset_path.stem}_{BASELINE_CACHE_VERSION}.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as fh:
            cached = pickle.load(fh)
        if isinstance(cached, BaselineContext):
            return cached

    baseline_kept, baseline_meta, *_ = variant_k.build_variant_k_pre_coupling_kept(dataset)
    baseline_coupled = merged_engine._apply_shared_equity_coupling(
        sorted(baseline_kept, key=lambda t: (t.exit_time, t.entry_time)),
        STARTING_EQUITY,
        v14_max_units=baseline_meta["v14_max_units"],
    )
    baseline_eq = merged_engine._build_equity_curve(baseline_coupled, STARTING_EQUITY)
    baseline_summary = merged_engine._stats(baseline_coupled, STARTING_EQUITY, baseline_eq)
    ctx = BaselineContext(
        dataset=str(Path(dataset).resolve()),
        baseline_kept=baseline_kept,
        baseline_coupled=baseline_coupled,
        baseline_summary=baseline_summary,
        baseline_meta=baseline_meta,
    )
    with cache_path.open("wb") as fh:
        pickle.dump(ctx, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return ctx


def _ts(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def trade_dict_to_trade_row(trade: dict[str, Any], *, size_scale: float = 1.0) -> merged_engine.TradeRow:
    raw = dict(trade.get("raw") or {})
    raw.update({
        "ownership_cell": trade.get("ownership_cell"),
        "setup_type": trade.get("setup_type"),
        "evaluator_mode": trade.get("evaluator_mode"),
        "timing_gate": trade.get("timing_gate"),
        "slice_id": trade.get("slice_id"),
    })
    return merged_engine.TradeRow(
        strategy=str(trade["strategy"]),
        entry_time=_ts(trade["entry_time"]),
        exit_time=_ts(trade["exit_time"]),
        entry_session=str(trade.get("entry_session") or trade.get("session") or "unknown"),
        side=str(trade["side"]),
        pips=float(trade["pips"]),
        usd=float(trade["usd"]) * float(size_scale),
        exit_reason=str(trade.get("exit_reason") or "unknown"),
        standalone_entry_equity=float(trade.get("standalone_entry_equity", STARTING_EQUITY)),
        raw=raw,
        size_scale=float(trade.get("size_scale", 1.0)) * float(size_scale),
    )


def _trade_identity_from_trade_dict(trade: dict[str, Any]) -> tuple[str, str, str, str]:
    raw = dict(trade.get("raw") or {})
    trade_id = raw.get("trade_id")
    if trade_id is not None:
        return (str(trade.get("strategy", "")), "trade_id", str(trade_id), str(trade.get("side", "")))
    return (
        str(trade.get("strategy", "")),
        str(trade.get("entry_time", "")),
        str(trade.get("exit_time", "")),
        str(trade.get("side", "")),
    )


def _trade_identity_from_trade_row(trade: merged_engine.TradeRow) -> tuple[str, str, str, str]:
    raw = dict(getattr(trade, "raw", {}) or {})
    trade_id = raw.get("trade_id")
    if trade_id is not None:
        return (str(trade.strategy), "trade_id", str(trade_id), str(trade.side))
    return (
        str(trade.strategy),
        pd.Timestamp(trade.entry_time).isoformat(),
        pd.Timestamp(trade.exit_time).isoformat(),
        str(trade.side),
    )


def _has_exact_baseline_match(trade: dict[str, Any], baseline_trades: list[merged_engine.TradeRow]) -> bool:
    entry_time = _ts(trade["entry_time"])
    side = str(trade["side"])
    for t in baseline_trades:
        if _ts(t.entry_time) == entry_time and str(t.side) == side:
            return True
    return False


def _has_exact_baseline_match_by_identity(
    trade: dict[str, Any],
    baseline_identities: set[tuple[str, str, str, str]],
) -> bool:
    return _trade_identity_from_trade_dict(trade) in baseline_identities


def _intervals_overlap(a0: pd.Timestamp, a1: pd.Timestamp, b0: pd.Timestamp, b1: pd.Timestamp) -> bool:
    return max(a0, b0) < min(a1, b1)


def _count_internal_overlap_pairs(trades: list[dict[str, Any]]) -> tuple[int, int]:
    overlap_pairs = 0
    opposite_side_pairs = 0
    ordered = sorted(trades, key=lambda t: (_ts(t["entry_time"]), _ts(t["exit_time"])))
    for i, a in enumerate(ordered):
        a0 = _ts(a["entry_time"])
        a1 = _ts(a["exit_time"])
        for b in ordered[i + 1:]:
            b0 = _ts(b["entry_time"])
            b1 = _ts(b["exit_time"])
            if _intervals_overlap(a0, a1, b0, b1):
                overlap_pairs += 1
                if str(a.get("side", "")) != str(b.get("side", "")):
                    opposite_side_pairs += 1
    return overlap_pairs, opposite_side_pairs


def _infer_trade_lots_from_dict(trade: dict[str, Any]) -> float:
    raw = dict(trade.get("raw") or {})
    for key in ("size_lots", "lot_size", "lots_initial", "lots_remaining"):
        value = raw.get(key)
        if value is not None:
            try:
                lots = float(value)
                if lots > 0:
                    return lots * float(trade.get("size_scale", 1.0))
            except Exception:
                pass
    pips = abs(float(trade.get("pips", 0.0) or 0.0))
    usd = abs(float(trade.get("usd", 0.0) or 0.0))
    entry_price = float(trade.get("entry_price") or raw.get("entry_price") or 0.0)
    if pips <= 1e-9 or usd <= 1e-9 or entry_price <= 1e-9:
        return 0.0
    pip_value_per_lot = 1000.0 / entry_price
    return max(0.0, usd / max(1e-9, pips * pip_value_per_lot))


def _infer_trade_lots_from_row(trade: merged_engine.TradeRow) -> float:
    raw = dict(getattr(trade, "raw", {}) or {})
    for key in ("size_lots", "lot_size", "lots_initial", "lots_remaining"):
        value = raw.get(key)
        if value is not None:
            try:
                lots = float(value)
                if lots > 0:
                    return lots * float(getattr(trade, "size_scale", 1.0) or 1.0)
            except Exception:
                pass
    pips = abs(float(getattr(trade, "pips", 0.0) or 0.0))
    usd = abs(float(getattr(trade, "usd", 0.0) or 0.0))
    entry_price = float(raw.get("entry_price") or 0.0)
    if pips <= 1e-9 or usd <= 1e-9 or entry_price <= 1e-9:
        return 0.0
    pip_value_per_lot = 1000.0 / entry_price
    return max(0.0, usd / max(1e-9, pips * pip_value_per_lot))


def _apply_conflict_policy_to_selected_trades(
    selected_trades: list[dict[str, Any]],
    policy: ConflictPolicy,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    ordered = sorted(selected_trades, key=lambda t: (_ts(t["entry_time"]), _ts(t["exit_time"])))
    kept: list[dict[str, Any]] = []
    active: list[dict[str, Any]] = []
    entries_by_day: dict[str, int] = {}
    seen_ids: set[tuple[str, str, str, str]] = set()
    stats = {
        "exact_duplicate_blocked": 0,
        "max_open_blocked": 0,
        "max_entries_day_blocked": 0,
        "overlap_blocked": 0,
        "opposite_side_blocked": 0,
    }

    for trade in ordered:
        ident = _trade_identity_from_trade_dict(trade)
        if ident in seen_ids:
            stats["exact_duplicate_blocked"] += 1
            continue
        seen_ids.add(ident)

        entry_ts = _ts(trade["entry_time"])
        exit_ts = _ts(trade["exit_time"])
        day_key = entry_ts.strftime("%Y-%m-%d")
        active = [row for row in active if _ts(row["exit_time"]) > entry_ts]

        if policy.max_entries_per_day is not None and entries_by_day.get(day_key, 0) >= policy.max_entries_per_day:
            stats["max_entries_day_blocked"] += 1
            continue
        if policy.max_open_offensive is not None and len(active) >= policy.max_open_offensive:
            stats["max_open_blocked"] += 1
            continue

        if not policy.allow_internal_overlap and active:
            stats["overlap_blocked"] += 1
            continue

        if not policy.allow_opposite_side_overlap and any(str(row.get("side", "")) != str(trade.get("side", "")) for row in active):
            stats["opposite_side_blocked"] += 1
            continue

        kept.append(trade)
        active.append(trade)
        entries_by_day[day_key] = entries_by_day.get(day_key, 0) + 1

    return kept, stats


def _mark_price_for_ts(entry_price: float) -> float:
    return max(1e-6, float(entry_price))


def _unrealized_pnl_usd(open_positions: list[dict[str, Any]], mark_price: float) -> float:
    total = 0.0
    for pos in open_positions:
        lots = max(0.0, float(pos.get("lots", 0.0)))
        if lots <= 0:
            continue
        entry_price = float(pos["entry_price"])
        side = str(pos["side"])
        if side == "buy":
            pips_live = (mark_price - entry_price) / 0.01
        else:
            pips_live = (entry_price - mark_price) / 0.01
        pip_value = 1000.0 / max(1e-6, mark_price)
        total += float(pips_live) * pip_value * lots
    return float(total)


def _used_margin_usd(open_positions: list[dict[str, Any]], leverage: float) -> float:
    total = 0.0
    for pos in open_positions:
        lots = max(0.0, float(pos.get("lots", 0.0)))
        if lots <= 0:
            continue
        total += lots * 100000.0 / max(1.0, float(leverage))
    return float(total)


def _apply_margin_policy_to_candidates(
    *,
    additive_candidates: list[dict[str, Any]],
    baseline_trades: list[merged_engine.TradeRow],
    starting_equity: float,
    policy: ConflictPolicy,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not policy.margin_model_enabled and policy.max_lot_per_trade is None:
        return list(additive_candidates), {
            "lot_cap_blocked": 0,
            "margin_rejected": 0,
            "margin_call_state_blocks": 0,
        }

    baseline_events: list[tuple[pd.Timestamp, int, dict[str, Any]]] = []
    for trade in baseline_trades:
        raw = dict(getattr(trade, "raw", {}) or {})
        entry_price = float(raw.get("entry_price") or 0.0)
        if entry_price <= 0:
            continue
        lots = _infer_trade_lots_from_row(trade)
        ident = _trade_identity_from_trade_row(trade)
        payload = {
            "id": ident,
            "source": "baseline",
            "side": str(trade.side),
            "entry_price": entry_price,
            "lots": lots,
            "usd": float(trade.usd),
        }
        baseline_events.append((_ts(trade.entry_time), 1, payload))
        baseline_events.append((_ts(trade.exit_time), 0, payload))

    candidate_entries = sorted(additive_candidates, key=lambda t: (_ts(t["entry_time"]), _ts(t["exit_time"])))
    candidate_map = {}
    for trade in candidate_entries:
        ident = _trade_identity_from_trade_dict(trade)
        raw = dict(trade.get("raw") or {})
        candidate_map[ident] = {
            "id": ident,
            "source": "offensive",
            "side": str(trade["side"]),
            "entry_price": float(trade.get("entry_price") or raw.get("entry_price") or 0.0),
            "lots": _infer_trade_lots_from_dict(trade),
            "usd": float(trade.get("usd", 0.0) or 0.0),
            "trade": trade,
        }

    candidate_events: list[tuple[pd.Timestamp, int, dict[str, Any]]] = []
    for trade in candidate_entries:
        ident = _trade_identity_from_trade_dict(trade)
        payload = candidate_map[ident]
        candidate_events.append((_ts(trade["entry_time"]), 1, payload))
        candidate_events.append((_ts(trade["exit_time"]), 0, payload))

    events = baseline_events + candidate_events
    events.sort(key=lambda x: (x[0], x[1]))

    open_positions: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    accepted: set[tuple[str, str, str, str]] = set()
    realized_equity = float(starting_equity)
    stats = {
        "lot_cap_blocked": 0,
        "margin_rejected": 0,
        "margin_call_state_blocks": 0,
    }

    for ts, evt_type, payload in events:
        ident = payload["id"]
        if evt_type == 0:
            pos = open_positions.pop(ident, None)
            if pos is not None:
                realized_equity += float(pos.get("usd", 0.0))
            continue

        if payload["source"] == "baseline":
            open_positions[ident] = payload
            continue

        lots = max(0.0, float(payload.get("lots", 0.0)))
        if policy.max_lot_per_trade is not None and lots > float(policy.max_lot_per_trade):
            stats["lot_cap_blocked"] += 1
            continue

        mark_price = _mark_price_for_ts(float(payload.get("entry_price", 0.0) or 0.0))
        equity = float(realized_equity + _unrealized_pnl_usd(list(open_positions.values()), mark_price))
        used_margin = _used_margin_usd(list(open_positions.values()), policy.margin_leverage)
        buffer_usd = equity * (float(policy.margin_buffer_pct) / 100.0)
        free_margin = equity - used_margin - buffer_usd
        required_margin = lots * 100000.0 / max(1.0, float(policy.margin_leverage))

        if free_margin < 0:
            stats["margin_call_state_blocks"] += 1
            continue
        if policy.margin_model_enabled and required_margin > free_margin:
            stats["margin_rejected"] += 1
            continue

        accepted.add(ident)
        open_positions[ident] = payload

    kept = [trade for trade in candidate_entries if _trade_identity_from_trade_dict(trade) in accepted]
    return kept, stats


def run_slice_additive(
    *,
    baseline_ctx: BaselineContext,
    slice_spec: dict[str, Any],
    selected_trades: list[dict[str, Any]],
    size_scale: float = 1.0,
) -> dict[str, Any]:
    exact_overlap = [t for t in selected_trades if _has_exact_baseline_match(t, baseline_ctx.baseline_coupled)]
    additive_candidates = [t for t in selected_trades if not _has_exact_baseline_match(t, baseline_ctx.baseline_coupled)]
    offensive_rows = [trade_dict_to_trade_row(t, size_scale=size_scale) for t in additive_candidates]

    displaced_baseline: list[merged_engine.TradeRow] = []
    for bt in baseline_ctx.baseline_coupled:
        if any(
            _intervals_overlap(_ts(bt.entry_time), _ts(bt.exit_time), _ts(ot.entry_time), _ts(ot.exit_time))
            for ot in offensive_rows
        ):
            displaced_baseline.append(bt)

    variant_coupled = merged_engine._apply_shared_equity_coupling(
        sorted(baseline_ctx.baseline_kept + offensive_rows, key=lambda t: (t.exit_time, t.entry_time)),
        STARTING_EQUITY,
        v14_max_units=baseline_ctx.baseline_meta["v14_max_units"],
    )
    variant_eq = merged_engine._build_equity_curve(variant_coupled, STARTING_EQUITY)
    variant_summary = merged_engine._stats(variant_coupled, STARTING_EQUITY, variant_eq)
    baseline_summary = baseline_ctx.baseline_summary

    return {
        "slice_spec": slice_spec,
        "selection_counts": {
            "selected_trade_count": len(selected_trades),
            "exact_baseline_overlap_count": len(exact_overlap),
            "new_additive_trades_count": len(offensive_rows),
            "displaced_trades_count": len({(t.strategy, t.entry_time.isoformat(), t.exit_time.isoformat(), t.side) for t in displaced_baseline}),
        },
        "delta_vs_baseline": {
            "total_trades": int(variant_summary["total_trades"] - baseline_summary["total_trades"]),
            "net_usd": round(variant_summary["net_usd"] - baseline_summary["net_usd"], 2),
            "profit_factor": round(variant_summary["profit_factor"] - baseline_summary["profit_factor"], 4),
            "max_drawdown_usd": round(variant_summary["max_drawdown_usd"] - baseline_summary["max_drawdown_usd"], 2),
        },
        "variant_summary": variant_summary,
        "samples": {
            "exact_overlap": exact_overlap[:20],
            "new_additive": [asdict(t) for t in offensive_rows[:20]],
            "displaced_baseline": [asdict(t) for t in displaced_baseline[:20]],
        },
    }


def run_slice_additive_with_policy(
    *,
    baseline_ctx: BaselineContext,
    slice_spec: dict[str, Any],
    selected_trades: list[dict[str, Any]],
    conflict_policy: ConflictPolicy,
    size_scale: float = 1.0,
) -> dict[str, Any]:
    policy_selected, policy_stats = _apply_conflict_policy_to_selected_trades(selected_trades, conflict_policy)
    baseline_identities = {_trade_identity_from_trade_row(t) for t in baseline_ctx.baseline_coupled}
    exact_overlap = [t for t in policy_selected if _has_exact_baseline_match_by_identity(t, baseline_identities)]
    additive_candidates = [t for t in policy_selected if not _has_exact_baseline_match_by_identity(t, baseline_identities)]
    margin_selected, margin_stats = _apply_margin_policy_to_candidates(
        additive_candidates=additive_candidates,
        baseline_trades=baseline_ctx.baseline_coupled,
        starting_equity=STARTING_EQUITY,
        policy=conflict_policy,
    )
    offensive_rows = [trade_dict_to_trade_row(t, size_scale=size_scale) for t in margin_selected]

    overlapping_baseline: list[merged_engine.TradeRow] = []
    for bt in baseline_ctx.baseline_coupled:
        if any(
            _intervals_overlap(_ts(bt.entry_time), _ts(bt.exit_time), _ts(ot.entry_time), _ts(ot.exit_time))
            for ot in offensive_rows
        ):
            overlapping_baseline.append(bt)

    variant_coupled = merged_engine._apply_shared_equity_coupling(
        sorted(baseline_ctx.baseline_kept + offensive_rows, key=lambda t: (t.exit_time, t.entry_time)),
        STARTING_EQUITY,
        v14_max_units=baseline_ctx.baseline_meta["v14_max_units"],
    )
    variant_eq = merged_engine._build_equity_curve(variant_coupled, STARTING_EQUITY)
    variant_summary = merged_engine._stats(variant_coupled, STARTING_EQUITY, variant_eq)
    baseline_summary = baseline_ctx.baseline_summary
    internal_overlap_pairs, opposite_side_pairs = _count_internal_overlap_pairs(policy_selected)

    return {
        "slice_spec": slice_spec,
        "conflict_policy": asdict(conflict_policy),
        "policy_stats": {**policy_stats, **margin_stats},
        "selection_counts": {
            "raw_selected_trade_count": len(selected_trades),
            "policy_selected_trade_count": len(policy_selected),
            "exact_baseline_overlap_count": len(exact_overlap),
            "margin_selected_trade_count": len(margin_selected),
            "new_additive_trades_count": len(offensive_rows),
            "overlapping_baseline_trades_count": len({(t.strategy, t.entry_time.isoformat(), t.exit_time.isoformat(), t.side) for t in overlapping_baseline}),
            "internal_overlap_pairs": internal_overlap_pairs,
            "internal_opposite_side_overlap_pairs": opposite_side_pairs,
        },
        "delta_vs_baseline": {
            "total_trades": int(variant_summary["total_trades"] - baseline_summary["total_trades"]),
            "net_usd": round(variant_summary["net_usd"] - baseline_summary["net_usd"], 2),
            "profit_factor": round(variant_summary["profit_factor"] - baseline_summary["profit_factor"], 4),
            "max_drawdown_usd": round(variant_summary["max_drawdown_usd"] - baseline_summary["max_drawdown_usd"], 2),
        },
        "variant_summary": variant_summary,
        "samples": {
            "exact_overlap": exact_overlap[:20],
            "new_additive": [asdict(t) for t in offensive_rows[:20]],
            "overlapping_baseline": [asdict(t) for t in overlapping_baseline[:20]],
        },
    }


def main() -> int:
    raise SystemExit("Use this module via import from the discovery runner.")


if __name__ == "__main__":
    main()
