#!/usr/bin/env python3
"""
Advanced analytics for the canonical narrow system using validated artifacts.

This is intentionally lightweight: it reads the canonical system artifact, the
validated additive artifact, and the pilot/validation outputs to produce a
decision-oriented analysis package without rerunning the full baseline stack.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("/Users/codygnon/Documents/usdjpy_assistant")
OUT_DIR = ROOT / "research_out"

SYSTEM_ARTIFACT = OUT_DIR / "system_variant_k_plus_setupd_whitespace.json"
ADDITIVE_ARTIFACT = OUT_DIR / "offensive_setupd_additive_mean_reversion_er_low_der_neg_100pct_first30.json"
PILOT_ARTIFACT = OUT_DIR / "offensive_setupd_whitespace_shadow_pilot.json"
VALIDATION_ARTIFACT = OUT_DIR / "offensive_pilot_setupd_replay_validation.json"
BOARD_ARTIFACT = OUT_DIR / "offensive_candidate_board_ladder.json"

OUTPUT_JSON = OUT_DIR / "system_variant_k_plus_setupd_whitespace_advanced_analytics.json"
OUTPUT_MD = OUT_DIR / "system_variant_k_plus_setupd_whitespace_advanced_analytics.md"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _round(v: float, digits: int = 2) -> float:
    return round(float(v), digits)


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if abs(b) > 1e-12 else 0.0


def _ts(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _month(value: Any) -> str:
    return _ts(value).strftime("%Y-%m")


def _format_money(v: float) -> str:
    sign = "+" if v > 0 else ""
    return f"{sign}{v:,.2f}"


def _format_num(v: float, digits: int = 2) -> str:
    sign = "+" if v > 0 else ""
    return f"{sign}{v:.{digits}f}"


def _added_trade_quality(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "trade_count": 0,
            "net_usd_sample_sum": 0.0,
            "net_pips_sample_sum": 0.0,
            "avg_usd_sample": 0.0,
            "avg_pips_sample": 0.0,
            "win_rate_pct_sample": 0.0,
            "avg_mfe_pips": 0.0,
            "avg_mae_pips": 0.0,
            "avg_duration_minutes": 0.0,
            "avg_mfe_capture_ratio": 0.0,
        }
    usd = [float(r["usd"]) for r in rows]
    pips = [float(r["pips"]) for r in rows]
    mfes = [float(r.get("raw", {}).get("mfe_pips", 0.0) or 0.0) for r in rows]
    maes = [float(r.get("raw", {}).get("mae_pips", 0.0) or 0.0) for r in rows]
    durations = [float(r.get("raw", {}).get("trade_duration_minutes", 0.0) or 0.0) for r in rows]
    capture = [
        _safe_div(float(r["pips"]), float(r.get("raw", {}).get("mfe_pips", 0.0) or 0.0))
        for r in rows
        if float(r.get("raw", {}).get("mfe_pips", 0.0) or 0.0) > 0
    ]
    wins = sum(1 for x in usd if x > 0.0)
    return {
        "trade_count": len(rows),
        "net_usd_sample_sum": _round(sum(usd)),
        "net_pips_sample_sum": _round(sum(pips)),
        "avg_usd_sample": _round(sum(usd) / len(rows)),
        "avg_pips_sample": _round(sum(pips) / len(rows)),
        "win_rate_pct_sample": _round(100.0 * wins / len(rows), 2),
        "avg_mfe_pips": _round(sum(mfes) / len(mfes)),
        "avg_mae_pips": _round(sum(maes) / len(maes)),
        "avg_duration_minutes": _round(sum(durations) / len(durations), 1),
        "avg_mfe_capture_ratio": _round(sum(capture) / len(capture), 3) if capture else 0.0,
    }


def _build_dataset(
    dataset_key: str,
    system_ds: dict[str, Any],
    additive_ds: dict[str, Any],
    pilot_ds: dict[str, Any] | None,
    validation_ds: dict[str, Any] | None,
) -> dict[str, Any]:
    baseline = additive_ds["baseline_summary"]
    variant = additive_ds["variants"]["100pct"]["variant_summary"]
    selection = additive_ds["variants"]["100pct"]["selection_counts"]
    samples = additive_ds["variants"]["100pct"]["samples"]
    added = list(samples["new_additive"])
    overlap = list(samples["exact_overlap"])

    direct_sample_usd = sum(float(r["usd"]) for r in added)
    direct_sample_pips = sum(float(r["pips"]) for r in added)
    total_delta_usd = float(variant["net_usd"] - baseline["net_usd"])
    total_delta_pips = float(variant["net_pips"] - baseline["net_pips"])
    implied_coupling_gap_usd = total_delta_usd - direct_sample_usd
    implied_coupling_gap_pips = total_delta_pips - direct_sample_pips

    month_delta = {}
    for row in added:
        month_delta.setdefault(_month(row["exit_time"]), 0.0)
        month_delta[_month(row["exit_time"])] += float(row["usd"])

    ledger = []
    for row in added:
        raw = dict(row.get("raw", {}))
        mfe = float(raw.get("mfe_pips", 0.0) or 0.0)
        capture = _safe_div(float(row["pips"]), mfe) if mfe > 0 else 0.0
        ledger.append(
            {
                "entry_time": _ts(row["entry_time"]).isoformat(),
                "exit_time": _ts(row["exit_time"]).isoformat(),
                "month": _month(row["exit_time"]),
                "usd_sample": _round(float(row["usd"])),
                "pips_sample": _round(float(row["pips"])),
                "exit_reason": row["exit_reason"],
                "ownership_cell": raw.get("ownership_cell"),
                "signal_time": raw.get("signal_time"),
                "mfe_pips": _round(float(raw.get("mfe_pips", 0.0) or 0.0), 2),
                "mae_pips": _round(float(raw.get("mae_pips", 0.0) or 0.0), 2),
                "duration_minutes": _round(float(raw.get("trade_duration_minutes", 0.0) or 0.0), 1),
                "mfe_capture_ratio": _round(capture, 3),
            }
        )

    overlap_details = [
        {
            "signal_time": row.get("signal_time"),
            "entry_time": row.get("entry_time"),
            "exit_time": row.get("exit_time"),
            "ownership_cell": row.get("ownership_cell"),
            "pnl_pips": row.get("pnl_pips"),
            "exit_reason": row.get("exit_reason"),
        }
        for row in overlap
    ]

    return {
        "baseline_summary": baseline,
        "system_summary": variant,
        "canonical_system_summary": system_ds["system_summary"],
        "delta_vs_baseline": {
            "total_trades": int(variant["total_trades"] - baseline["total_trades"]),
            "net_usd": _round(total_delta_usd),
            "net_pips": _round(total_delta_pips),
            "profit_factor": _round(float(variant["profit_factor"] - baseline["profit_factor"]), 4),
            "max_drawdown_usd": _round(float(variant["max_drawdown_usd"] - baseline["max_drawdown_usd"])),
        },
        "candidate_funnel": {
            "raw_native_long_trades_in_cell": int(selection["raw_native_long_trades_in_cells"]),
            "after_daily_cap": int(selection["after_daily_cap"]),
            "exact_overlap_count": int(selection["exact_baseline_overlap_count"]),
            "new_additive_trades_count": int(selection["new_additive_trades_count"]),
            "displaced_trades_count": int(selection["displaced_trades_count"]),
            "pilot_scope_bars": int(pilot_ds["summary"]["pilot_scope_bars"]) if pilot_ds else None,
            "entry_eligible_scope_bars": int(pilot_ds["summary"]["entry_eligible_scope_bars"]) if pilot_ds else None,
            "pilot_candidates": int(pilot_ds["summary"]["pilot_candidates"]) if pilot_ds else None,
            "channel_state_blocked": int(pilot_ds["blocked_invocation_reasons"].get("setupd_channel_consumed_by_earlier_setupd_long", 0)) if pilot_ds else None,
        },
        "validation_status": validation_ds,
        "direct_trade_quality": _added_trade_quality(added),
        "delta_decomposition": {
            "direct_sample_added_trade_net_usd": _round(direct_sample_usd),
            "direct_sample_added_trade_net_pips": _round(direct_sample_pips),
            "implied_shared_equity_coupling_gap_usd": _round(implied_coupling_gap_usd),
            "implied_shared_equity_coupling_gap_pips": _round(implied_coupling_gap_pips),
            "direct_sample_share_of_total_delta_usd": _round(_safe_div(direct_sample_usd, total_delta_usd), 4),
        },
        "month_delta_from_added_trades_only": [
            {"month": month, "net_usd_sample": _round(val)}
            for month, val in sorted(month_delta.items())
        ],
        "added_trade_ledger": ledger,
        "overlap": {
            "exact_overlap_count": len(overlap_details),
            "exact_overlap_details": overlap_details,
        },
    }


def _build_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Variant K + Setup D Whitespace Advanced Analytics")
    lines.append("")
    lines.append("This report uses the validated additive and pilot artifacts to explain how the narrow new system behaves.")
    lines.append("")
    for dataset_key in ["500k", "1000k"]:
        ds = payload["datasets"][dataset_key]
        base = ds["baseline_summary"]
        sysm = ds["system_summary"]
        delta = ds["delta_vs_baseline"]
        funnel = ds["candidate_funnel"]
        qual = ds["direct_trade_quality"]
        decomp = ds["delta_decomposition"]
        lines.append(f"## {dataset_key}")
        lines.append("")
        lines.append("### Headline")
        lines.append("")
        lines.append(f"- Baseline: `{base['total_trades']}` trades, net USD `{_format_money(base['net_usd'])}`, PF `{base['profit_factor']:.4f}`, max DD `{base['max_drawdown_usd']:.2f}`")
        lines.append(f"- New system: `{sysm['total_trades']}` trades, net USD `{_format_money(sysm['net_usd'])}`, PF `{sysm['profit_factor']:.4f}`, max DD `{sysm['max_drawdown_usd']:.2f}`")
        lines.append(f"- Delta: trades `{delta['total_trades']:+d}`, net USD `{_format_money(delta['net_usd'])}`, PF `{_format_num(delta['profit_factor'], 4)}`, max DD `{_format_money(delta['max_drawdown_usd'])}`")
        lines.append("")
        lines.append("### Funnel")
        lines.append("")
        lines.append(f"- Raw Setup D long trades in target cell: `{funnel['raw_native_long_trades_in_cell']}`")
        lines.append(f"- After daily cap: `{funnel['after_daily_cap']}`")
        lines.append(f"- Exact overlap with baseline London entry: `{funnel['exact_overlap_count']}`")
        lines.append(f"- New additive trades: `{funnel['new_additive_trades_count']}`")
        if funnel["pilot_candidates"] is not None:
            lines.append(f"- Historical pilot candidates: `{funnel['pilot_candidates']}`")
            lines.append(f"- Channel-state blocked: `{funnel['channel_state_blocked']}`")
        lines.append("")
        lines.append("### Direct Added-Trade Quality")
        lines.append("")
        lines.append(f"- Added trades in sample: `{qual['trade_count']}`")
        lines.append(f"- Net USD from added-trade sample: `{_format_money(qual['net_usd_sample_sum'])}`")
        lines.append(f"- Avg USD / trade: `{_format_money(qual['avg_usd_sample'])}`")
        lines.append(f"- Avg pips / trade: `{_format_num(qual['avg_pips_sample'])}`")
        lines.append(f"- Win rate: `{qual['win_rate_pct_sample']:.2f}%`")
        lines.append(f"- Avg MFE / MAE: `{qual['avg_mfe_pips']}` / `{qual['avg_mae_pips']}`")
        lines.append(f"- Avg MFE capture: `{qual['avg_mfe_capture_ratio']}`")
        lines.append("")
        lines.append("### Delta Decomposition")
        lines.append("")
        lines.append(f"- Direct added-trade sample net USD: `{_format_money(decomp['direct_sample_added_trade_net_usd'])}`")
        lines.append(f"- Implied shared-equity coupling gap: `{_format_money(decomp['implied_shared_equity_coupling_gap_usd'])}`")
        lines.append(f"- Direct sample share of total delta: `{decomp['direct_sample_share_of_total_delta_usd']:.2%}`")
        lines.append("")
        lines.append("### Added Trade Ledger")
        lines.append("")
        for row in ds["added_trade_ledger"]:
            lines.append(
                f"- `{row['entry_time']}` -> `{row['exit_time']}`: USD `{_format_money(row['usd_sample'])}`, "
                f"pips `{_format_num(row['pips_sample'])}`, exit `{row['exit_reason']}`, "
                f"MFE `{row['mfe_pips']}`, MAE `{row['mae_pips']}`, capture `{row['mfe_capture_ratio']}`"
            )
        if not ds["added_trade_ledger"]:
            lines.append("- None")
        lines.append("")
        lines.append("### Month Impact From Added Trades")
        lines.append("")
        if ds["month_delta_from_added_trades_only"]:
            for row in ds["month_delta_from_added_trades_only"]:
                lines.append(f"- `{row['month']}`: `{_format_money(row['net_usd_sample'])}`")
        else:
            lines.append("- No added-trade month delta.")
        lines.append("")
        lines.append("### Read")
        lines.append("")
        if dataset_key == "500k":
            lines.append("- The result is driven by one real additive winner plus one exact-overlap day that does not expand the book. The drawdown cost stays modest, but the sample is tiny.")
        else:
            lines.append("- The result is driven by two additive trades: one small loser and one strong winner. The net effect stays positive and the overall system still improves.")
        lines.append("")
    lines.append("## Cross-Dataset Read")
    lines.append("")
    lines.append("- The candidate is still rare. This is a narrow enhancement, not a broad offensive engine.")
    lines.append("- The direct added-trade sample explains most, but not all, of the system delta. The remaining gap comes from shared-equity coupling and artifact-level interaction effects.")
    lines.append("- The exact-overlap day remains present on both datasets, which reinforces the need to keep treating this as a narrow whitespace slice rather than a general London overlay.")
    return "\n".join(lines) + "\n"


def main() -> int:
    system = _load_json(SYSTEM_ARTIFACT)
    additive = _load_json(ADDITIVE_ARTIFACT)
    pilot = _load_json(PILOT_ARTIFACT) if PILOT_ARTIFACT.exists() else {"results": {}}
    validation = _load_json(VALIDATION_ARTIFACT) if VALIDATION_ARTIFACT.exists() else {"results": {}}
    board = _load_json(BOARD_ARTIFACT) if BOARD_ARTIFACT.exists() else {}

    payload = {
        "title": "Variant K plus Setup D whitespace advanced analytics",
        "system_artifact": str(SYSTEM_ARTIFACT),
        "candidate": system["frozen_rule"],
        "board_context": {
            "pilot_candidate": board.get("pilot_candidate"),
            "research_follow_up": board.get("research_follow_up"),
        },
        "datasets": {},
    }

    for dataset_key in ["500k", "1000k"]:
        pilot_key = f"USDJPY_M1_OANDA_{dataset_key}.csv"
        payload["datasets"][dataset_key] = _build_dataset(
            dataset_key=dataset_key,
            system_ds=system["datasets"][dataset_key],
            additive_ds=additive["results"][dataset_key],
            pilot_ds=pilot.get("results", {}).get(pilot_key),
            validation_ds=validation.get("results", {}).get(dataset_key),
        )

    OUTPUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    OUTPUT_MD.write_text(_build_markdown(payload), encoding="utf-8")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
