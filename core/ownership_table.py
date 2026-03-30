"""
Conservative ownership table loader for chart-first / Phase A–B diagnostics.

Single source of truth for:
  - Cell string format (must match diagnostic_strategy_ownership.json grid keys)
  - ER / ΔER bucketing (must stay aligned with variant_k._er_bucket / _der_bucket)
  - Stable cross-dataset cells from diagnostic_ownership_stability.json

Does not import scripts.* to avoid core → scripts cycles. Bucket thresholds are
duplicated here intentionally; keep them in sync with
``scripts.backtest_variant_k_london_cluster``.)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

STRATEGY_KEYS: tuple[str, ...] = ("v14", "london_v2", "v44_ny")


def _repo_research_out() -> Path:
    return Path(__file__).resolve().parent.parent / "research_out"


def er_bucket(er: float) -> str:
    """Aligned with ``variant_k._er_bucket``."""
    if er < 0.35:
        return "er_low"
    if er < 0.55:
        return "er_mid"
    return "er_high"


def der_bucket(delta_er: float) -> str:
    """Aligned with ``variant_k._der_bucket``."""
    return "der_neg" if delta_er < 0 else "der_pos"


def cell_key(regime_label: str, er_bucket_str: str, der_bucket_str: str) -> str:
    """Grid cell id as used in diagnostic_strategy_ownership.json."""
    return f"{regime_label}/{er_bucket_str}/{der_bucket_str}"


def parse_cell_key(cell: str) -> tuple[str, str, str]:
    parts = cell.split("/")
    if len(parts) != 3:
        raise ValueError(f"Invalid cell key (expected regime/er/der): {cell!r}")
    return parts[0], parts[1], parts[2]


def cell_key_from_floats(regime_label: str, er: float, delta_er: float) -> str:
    """Convenience: bucket floats then build cell key."""
    return cell_key(regime_label, er_bucket(er), der_bucket(delta_er))


def load_stability_json(*, research_out: Path | None = None) -> dict[str, Any]:
    base = research_out if research_out is not None else _repo_research_out()
    path = base / "diagnostic_ownership_stability.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_strategy_ownership_json(*, research_out: Path | None = None) -> dict[str, Any]:
    base = research_out if research_out is not None else _repo_research_out()
    path = base / "diagnostic_strategy_ownership.json"
    return json.loads(path.read_text(encoding="utf-8"))


def build_conservative_table(
    stability: dict[str, Any],
    ownership: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """
    Merge stable_same_owner + stable_no_trade into one lookup.

    Values:
      - stable_owner: recommended_strategy, owner_avg_500k/1000k, owner_count_500k/1000k
      - stable_no_trade: recommended_strategy == \"NO-TRADE\"
    """
    stable_cells = set(stability.get("stable_same_owner", []))
    stable_no_trade = set(stability.get("stable_no_trade", []))

    table: dict[str, dict[str, Any]] = {}

    for cell in stable_cells:
        cell_data_500k = ownership.get("500k", {}).get("grid", {}).get(cell)
        cell_data_1000k = ownership.get("1000k", {}).get("grid", {}).get(cell)
        if cell_data_500k is None or cell_data_1000k is None:
            continue

        owner = cell_data_500k.get("owner", "unknown")
        if cell_data_1000k.get("owner", "unknown") != owner:
            continue

        strategies_500k = cell_data_500k.get("strategies", {})
        strategies_1000k = cell_data_1000k.get("strategies", {})

        table[cell] = {
            "recommended_strategy": owner,
            "type": "stable_owner",
            "owner_avg_500k": strategies_500k.get(owner, {}).get("avg_pips", 0),
            "owner_avg_1000k": strategies_1000k.get(owner, {}).get("avg_pips", 0),
            "owner_count_500k": strategies_500k.get(owner, {}).get("count", 0),
            "owner_count_1000k": strategies_1000k.get(owner, {}).get("count", 0),
        }

    for cell in stable_no_trade:
        table[cell] = {
            "recommended_strategy": "NO-TRADE",
            "type": "stable_no_trade",
        }

    return table


def load_conservative_table(*, research_out: Path | None = None) -> dict[str, dict[str, Any]]:
    """Load JSON diagnostics and return the conservative ownership table."""
    st = load_stability_json(research_out=research_out)
    ow = load_strategy_ownership_json(research_out=research_out)
    return build_conservative_table(st, ow)


def stable_owner_cells(table: dict[str, dict[str, Any]]) -> list[str]:
    """Cells with a stable strategy owner (excludes stable no-trade)."""
    return sorted(c for c, v in table.items() if v.get("type") == "stable_owner")


def stable_no_trade_cells(table: dict[str, dict[str, Any]]) -> list[str]:
    return sorted(c for c, v in table.items() if v.get("type") == "stable_no_trade")


def cells_where_owner_is(table: dict[str, dict[str, Any]], owner: str) -> list[str]:
    """Stable owner cells whose recommended_strategy equals ``owner`` (e.g. ``\"v14\"``)."""
    return sorted(
        c
        for c, v in table.items()
        if v.get("type") == "stable_owner" and v.get("recommended_strategy") == owner
    )
