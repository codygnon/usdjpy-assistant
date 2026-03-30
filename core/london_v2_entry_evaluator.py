"""
Extracted London V2 (multi-setup) entry signal evaluator.

This module contains the exact setup evaluation logic from
``scripts/backtest_v2_multisetup_london.py``, extracted into callable
functions so that both the native backtest loop and the shadow invocation
layer can use the same source of truth.

**No thresholds or conditions were changed during extraction.**

Setup priority order (native): A -> B -> C -> D
Each setup evaluates independently on the same bar; multiple setups can
fire on the same bar (they are not mutually exclusive).

State caveat -- Setup B:
    Setup B uses a candidate buffer (b_candidates) that tracks recent wicks.
    This buffer is an explicit input/output parameter so the caller owns it.
"""
from __future__ import annotations

from typing import Any

PIP_SIZE = 0.01


# ---------------------------------------------------------------------------
# Setup A: Asian range breakout
# ---------------------------------------------------------------------------
def evaluate_london_setup_a(
    *,
    row: Any,
    cfg_a: dict[str, Any],
    asian_high: float,
    asian_low: float,
    asian_range_pips: float,
    lor_range_pips: float | None,
    lor_valid: bool,
    ts: Any,
    nxt_ts: Any,
    a_start: Any,
    a_end: Any,
    asian_valid: bool,
    channel_long_state: str,
    channel_short_state: str,
    channel_long_entries: int,
    channel_short_entries: int,
) -> list[dict[str, Any]]:
    """
    Evaluate Setup A (breakout) on the current closed bar.

    Returns a list of 0-2 pending entry dicts (one per direction).
    Each dict contains: setup_type, direction, execute_time, raw_sl,
    asian_range_pips, lor_range_pips, is_reentry.
    """
    if not cfg_a.get("enabled", True):
        return []
    if not asian_valid:
        return []
    if not (a_start <= ts < a_end):
        return []

    results = []
    breakout_buffer = float(cfg_a["breakout_buffer_pips"]) * PIP_SIZE
    sl_buffer = float(cfg_a["sl_buffer_pips"]) * PIP_SIZE
    close = float(row["close"])

    long_break = close > asian_high + breakout_buffer
    short_break = close < asian_low - breakout_buffer

    if long_break and bool(cfg_a.get("allow_long", True)) and channel_long_state == "ARMED":
        results.append({
            "setup_type": "A",
            "direction": "long",
            "execute_time": nxt_ts,
            "raw_sl": asian_low - sl_buffer,
            "asian_range_pips": asian_range_pips,
            "lor_range_pips": lor_range_pips if lor_valid else None,
            "is_reentry": channel_long_entries > 0,
        })

    if short_break and bool(cfg_a.get("allow_short", True)) and channel_short_state == "ARMED":
        results.append({
            "setup_type": "A",
            "direction": "short",
            "execute_time": nxt_ts,
            "raw_sl": asian_high + sl_buffer,
            "asian_range_pips": asian_range_pips,
            "lor_range_pips": lor_range_pips if lor_valid else None,
            "is_reentry": channel_short_entries > 0,
        })

    return results


# ---------------------------------------------------------------------------
# Setup B: False breakout reversal
# ---------------------------------------------------------------------------
def evaluate_london_setup_b(
    *,
    row: Any,
    cfg_b: dict[str, Any],
    asian_high: float,
    asian_low: float,
    asian_range_pips: float,
    lor_range_pips: float | None,
    lor_valid: bool,
    ts: Any,
    nxt_ts: Any,
    bar_index: int,
    b_start: Any,
    b_end: Any,
    asian_valid: bool,
    channel_long_state: str,
    channel_short_state: str,
    channel_long_entries: int,
    channel_short_entries: int,
    b_candidates: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """
    Evaluate Setup B (false breakout reversal) on the current closed bar.

    b_candidates is the shared state buffer tracking wick candidates.
    It is passed in and a (possibly modified) copy is returned so the caller
    can maintain it across bars.

    Returns (pending_entries, updated_b_candidates).
    """
    if not cfg_b.get("enabled", True):
        return [], b_candidates
    if not asian_valid:
        return [], b_candidates
    if not (b_start <= ts < b_end):
        return [], b_candidates

    results = []
    close_back_inside = float(cfg_b["close_back_inside_pips"]) * PIP_SIZE
    sl_wick_buffer = float(cfg_b["sl_wick_buffer_pips"]) * PIP_SIZE
    confirm_window = int(cfg_b.get("confirm_window_candles", 3))
    i = bar_index

    # Short candidate: upside break then close back inside
    if float(row["high"]) > asian_high:
        b_candidates["short"].append({"expire_i": i + confirm_window - 1, "wick_high": float(row["high"])})
    for cand in b_candidates["short"]:
        cand["wick_high"] = max(float(cand["wick_high"]), float(row["high"]))
    short_confirm = float(row["close"]) < asian_high - close_back_inside
    if short_confirm and bool(cfg_b.get("allow_short", True)) and channel_short_state == "ARMED":
        active = [c for c in b_candidates["short"] if c["expire_i"] >= i]
        if active:
            wick_high = max(float(c["wick_high"]) for c in active)
            results.append({
                "setup_type": "B",
                "direction": "short",
                "execute_time": nxt_ts,
                "raw_sl": wick_high + sl_wick_buffer,
                "asian_range_pips": asian_range_pips,
                "lor_range_pips": lor_range_pips if lor_valid else None,
                "is_reentry": channel_short_entries > 0,
            })
            b_candidates["short"] = []
    b_candidates["short"] = [c for c in b_candidates["short"] if c["expire_i"] >= i]

    # Long candidate: downside break then close back inside
    if float(row["low"]) < asian_low:
        b_candidates["long"].append({"expire_i": i + confirm_window - 1, "wick_low": float(row["low"])})
    for cand in b_candidates["long"]:
        cand["wick_low"] = min(float(cand["wick_low"]), float(row["low"]))
    long_confirm = float(row["close"]) > asian_low + close_back_inside
    if long_confirm and bool(cfg_b.get("allow_long", True)) and channel_long_state == "ARMED":
        active = [c for c in b_candidates["long"] if c["expire_i"] >= i]
        if active:
            wick_low = min(float(c["wick_low"]) for c in active)
            results.append({
                "setup_type": "B",
                "direction": "long",
                "execute_time": nxt_ts,
                "raw_sl": wick_low - sl_wick_buffer,
                "asian_range_pips": asian_range_pips,
                "lor_range_pips": lor_range_pips if lor_valid else None,
                "is_reentry": channel_long_entries > 0,
            })
            b_candidates["long"] = []
    b_candidates["long"] = [c for c in b_candidates["long"] if c["expire_i"] >= i]

    return results, b_candidates


# ---------------------------------------------------------------------------
# Setup C: Level bounce
# ---------------------------------------------------------------------------
def evaluate_london_setup_c(
    *,
    row: Any,
    cfg_c: dict[str, Any],
    asian_high: float,
    asian_low: float,
    asian_range_pips: float,
    lor_range_pips: float | None,
    lor_valid: bool,
    ts: Any,
    nxt_ts: Any,
    c_start: Any,
    c_end: Any,
    asian_valid: bool,
    channel_long_state: str,
    channel_short_state: str,
    channel_long_entries: int,
    channel_short_entries: int,
) -> list[dict[str, Any]]:
    """
    Evaluate Setup C (level bounce) on the current closed bar.

    Returns a list of 0-2 pending entry dicts.
    """
    if not cfg_c.get("enabled", True):
        return []
    if not asian_valid:
        return []
    if not (c_start <= ts < c_end):
        return []

    results = []
    bounce_zone = float(cfg_c["bounce_zone_pips"]) * PIP_SIZE
    min_close_offset = float(cfg_c["min_close_offset_pips"]) * PIP_SIZE
    min_body = float(cfg_c["min_body_pips"]) * PIP_SIZE
    sl_buffer = float(cfg_c["sl_buffer_pips"]) * PIP_SIZE

    body = abs(float(row["close"]) - float(row["open"]))

    long_cond = (
        float(row["low"]) <= asian_low + bounce_zone
        and float(row["low"]) >= asian_low
        and float(row["close"]) > float(row["open"])
        and float(row["close"]) >= asian_low + min_close_offset
        and body >= min_body
    )
    short_cond = (
        float(row["high"]) >= asian_high - bounce_zone
        and float(row["high"]) <= asian_high
        and float(row["close"]) < float(row["open"])
        and float(row["close"]) <= asian_high - min_close_offset
        and body >= min_body
    )

    if long_cond and bool(cfg_c.get("allow_long", True)) and channel_long_state == "ARMED":
        results.append({
            "setup_type": "C",
            "direction": "long",
            "execute_time": nxt_ts,
            "raw_sl": asian_low - sl_buffer,
            "asian_range_pips": asian_range_pips,
            "lor_range_pips": lor_range_pips if lor_valid else None,
            "is_reentry": channel_long_entries > 0,
        })
    if short_cond and bool(cfg_c.get("allow_short", True)) and channel_short_state == "ARMED":
        results.append({
            "setup_type": "C",
            "direction": "short",
            "execute_time": nxt_ts,
            "raw_sl": asian_high + sl_buffer,
            "asian_range_pips": asian_range_pips,
            "lor_range_pips": lor_range_pips if lor_valid else None,
            "is_reentry": channel_short_entries > 0,
        })

    return results


# ---------------------------------------------------------------------------
# Setup D: LOR breakout
# ---------------------------------------------------------------------------
def evaluate_london_setup_d(
    *,
    row: Any,
    cfg_d: dict[str, Any],
    lor_high: float,
    lor_low: float,
    asian_range_pips: float | None,
    lor_range_pips: float,
    lor_valid: bool,
    asian_valid: bool,
    ts: Any,
    nxt_ts: Any,
    d_start: Any,
    d_end: Any,
    channel_long_state: str,
    channel_short_state: str,
    channel_long_entries: int,
    channel_short_entries: int,
) -> list[dict[str, Any]]:
    """
    Evaluate Setup D (LOR breakout) on the current closed bar.

    Returns a list of 0-2 pending entry dicts.
    """
    if not cfg_d.get("enabled", True):
        return []
    if not lor_valid:
        return []
    if not (d_start <= ts < d_end):
        return []

    results = []
    breakout_buffer = float(cfg_d["breakout_buffer_pips"]) * PIP_SIZE
    sl_buffer = float(cfg_d["sl_buffer_pips"]) * PIP_SIZE
    close = float(row["close"])

    long_break = close > lor_high + breakout_buffer
    short_break = close < lor_low - breakout_buffer

    if long_break and bool(cfg_d.get("allow_long", True)) and channel_long_state == "ARMED":
        results.append({
            "setup_type": "D",
            "direction": "long",
            "execute_time": nxt_ts,
            "raw_sl": lor_low - sl_buffer,
            "asian_range_pips": asian_range_pips if asian_valid else None,
            "lor_range_pips": lor_range_pips,
            "is_reentry": channel_long_entries > 0,
        })
    if short_break and bool(cfg_d.get("allow_short", True)) and channel_short_state == "ARMED":
        results.append({
            "setup_type": "D",
            "direction": "short",
            "execute_time": nxt_ts,
            "raw_sl": lor_high + sl_buffer,
            "asian_range_pips": asian_range_pips if asian_valid else None,
            "lor_range_pips": lor_range_pips,
            "is_reentry": channel_short_entries > 0,
        })

    return results


# ---------------------------------------------------------------------------
# Orchestrator: evaluate all setups in native priority order
# ---------------------------------------------------------------------------
def evaluate_london_v2_entry_signal(
    *,
    row: Any,
    cfg: dict[str, Any],
    asian_high: float,
    asian_low: float,
    asian_range_pips: float,
    asian_valid: bool,
    lor_high: float,
    lor_low: float,
    lor_range_pips: float,
    lor_valid: bool,
    ts: Any,
    nxt_ts: Any,
    bar_index: int,
    windows: dict[str, tuple[Any, Any]],
    channels: dict[tuple[str, str], dict[str, Any]],
    b_candidates: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """
    Run all four setups in native priority order (A, B, C, D) on one bar.

    Parameters
    ----------
    channels : dict[(setup, direction)] -> {state, entries, ...}
        The channel state dict. Read-only here; the caller manages transitions.
    b_candidates : dict
        Setup B candidate buffer. Updated and returned.

    Returns
    -------
    (pending_entries, updated_b_candidates)
        pending_entries: list of entry dicts generated by all setups.
        Each entry includes setup_type, direction, execute_time, raw_sl, etc.
    """
    all_entries: list[dict[str, Any]] = []

    # Setup A
    a_start, a_end = windows["A"]
    a_results = evaluate_london_setup_a(
        row=row,
        cfg_a=cfg["setups"]["A"],
        asian_high=asian_high,
        asian_low=asian_low,
        asian_range_pips=asian_range_pips,
        lor_range_pips=lor_range_pips if lor_valid else None,
        lor_valid=lor_valid,
        ts=ts,
        nxt_ts=nxt_ts,
        a_start=a_start,
        a_end=a_end,
        asian_valid=asian_valid,
        channel_long_state=channels[("A", "long")]["state"],
        channel_short_state=channels[("A", "short")]["state"],
        channel_long_entries=channels[("A", "long")]["entries"],
        channel_short_entries=channels[("A", "short")]["entries"],
    )
    all_entries.extend(a_results)

    # Setup B
    b_start, b_end = windows["B"]
    b_results, b_candidates = evaluate_london_setup_b(
        row=row,
        cfg_b=cfg["setups"]["B"],
        asian_high=asian_high,
        asian_low=asian_low,
        asian_range_pips=asian_range_pips,
        lor_range_pips=lor_range_pips if lor_valid else None,
        lor_valid=lor_valid,
        ts=ts,
        nxt_ts=nxt_ts,
        bar_index=bar_index,
        b_start=b_start,
        b_end=b_end,
        asian_valid=asian_valid,
        channel_long_state=channels[("B", "long")]["state"],
        channel_short_state=channels[("B", "short")]["state"],
        channel_long_entries=channels[("B", "long")]["entries"],
        channel_short_entries=channels[("B", "short")]["entries"],
        b_candidates=b_candidates,
    )
    all_entries.extend(b_results)

    # Setup C
    c_start, c_end = windows["C"]
    c_results = evaluate_london_setup_c(
        row=row,
        cfg_c=cfg["setups"]["C"],
        asian_high=asian_high,
        asian_low=asian_low,
        asian_range_pips=asian_range_pips,
        lor_range_pips=lor_range_pips if lor_valid else None,
        lor_valid=lor_valid,
        ts=ts,
        nxt_ts=nxt_ts,
        c_start=c_start,
        c_end=c_end,
        asian_valid=asian_valid,
        channel_long_state=channels[("C", "long")]["state"],
        channel_short_state=channels[("C", "short")]["state"],
        channel_long_entries=channels[("C", "long")]["entries"],
        channel_short_entries=channels[("C", "short")]["entries"],
    )
    all_entries.extend(c_results)

    # Setup D
    d_start, d_end = windows["D"]
    d_results = evaluate_london_setup_d(
        row=row,
        cfg_d=cfg["setups"]["D"],
        lor_high=lor_high,
        lor_low=lor_low,
        asian_range_pips=asian_range_pips,
        lor_range_pips=lor_range_pips,
        lor_valid=lor_valid,
        asian_valid=asian_valid,
        ts=ts,
        nxt_ts=nxt_ts,
        d_start=d_start,
        d_end=d_end,
        channel_long_state=channels[("D", "long")]["state"],
        channel_short_state=channels[("D", "short")]["state"],
        channel_long_entries=channels[("D", "long")]["entries"],
        channel_short_entries=channels[("D", "short")]["entries"],
    )
    all_entries.extend(d_results)

    return all_entries, b_candidates
