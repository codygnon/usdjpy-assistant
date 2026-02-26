#!/usr/bin/env python3
"""
Recalibration script: recompute optimal Trial #7 thresholds from actual trade data.

What it does:
  1. Pulls last N days of closed trades from SQLite (default 30 days)
  2. Recomputes optimal reversal risk tier thresholds from score vs outcome data
  3. Recomputes optimal TP by entry type × risk tier using MFE simulation
  4. Outputs recommended config changes as JSON
  5. Optionally applies them to the active profile (--apply)

Usage:
  python3 scripts/recalibrate.py --profile default
  python3 scripts/recalibrate.py --profile default --apply
  python3 scripts/recalibrate.py --profile /abs/path/to/profile.json --days 60
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup — allow running from repo root or scripts/
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

_data_base_env = os.environ.get("RAILWAY_VOLUME_MOUNT_PATH") or os.environ.get("USDJPY_DATA_DIR")
DATA_BASE = Path(_data_base_env) if _data_base_env else REPO_ROOT
PROFILES_DIR = DATA_BASE / "profiles"
LOGS_DIR = DATA_BASE / "logs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_profile_path(name_or_path: str) -> Path:
    p = Path(name_or_path)
    if p.is_absolute() and p.exists():
        return p
    # Try exact name under PROFILES_DIR
    candidate = PROFILES_DIR / f"{name_or_path}.json"
    if candidate.exists():
        return candidate
    # Fallback: rglob search
    matches = list(PROFILES_DIR.rglob(f"{name_or_path}.json"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Profile not found: {name_or_path!r}  (looked in {PROFILES_DIR})")


def _db_path_for_profile(profile_name: str) -> Path:
    """Mimic the logic in api/main.py: logs/<profile_name>/trades.db"""
    return LOGS_DIR / profile_name / "trades.db"


def _load_trades(db_path: Path, days: int) -> list[dict[str, Any]]:
    """Load closed trades from the last `days` calendar days."""
    import sqlite3

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    cutoff_ts = (datetime.now(timezone.utc) - timedelta(days=days)).timestamp()

    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        cur = con.execute(
            """
            SELECT
                id, side, entry_price, close_price, pips, profit,
                max_favorable_pips, risk_pips, sl_price,
                entry_type, reversal_risk_score, reversal_risk_tier,
                tier_number, open_time, close_time
            FROM trades
            WHERE close_time IS NOT NULL
              AND close_time > ?
            ORDER BY close_time ASC
            """,
            (cutoff_ts,),
        )
        rows = [dict(r) for r in cur.fetchall()]
    finally:
        con.close()
    return rows


# ---------------------------------------------------------------------------
# Analysis: reversal risk tier thresholds
# ---------------------------------------------------------------------------

def _analyze_tier_thresholds(
    trades: list[dict[str, Any]],
    score_field: str = "reversal_risk_score",
    outcome_field: str = "pips",
    win_threshold: float = 0.0,
) -> dict[str, Any]:
    """
    Sweep score thresholds 35–85.
    For each T, compute win_rate of trades where score >= T.
    Find where win_rate crosses reference levels:
      medium:   first T where win_rate >= 50%
      high:     first T where win_rate >= 57%  (stricter)
      critical: first T where win_rate >= 64%
    Returns recommended thresholds and supporting data.
    """
    scored = [
        t for t in trades
        if t.get(score_field) is not None and t.get(outcome_field) is not None
    ]
    if not scored:
        return {"error": f"No trades with {score_field!r} data", "scored_count": 0}

    sweep: list[dict[str, Any]] = []
    for threshold in range(35, 86):
        group = [t for t in scored if float(t[score_field]) >= threshold]
        if not group:
            sweep.append({"threshold": threshold, "count": 0, "win_rate": None})
            continue
        wins = sum(1 for t in group if float(t[outcome_field]) > win_threshold)
        sweep.append({
            "threshold": threshold,
            "count": len(group),
            "win_rate": wins / len(group),
        })

    # Reference levels
    TARGET_MEDIUM   = 0.50
    TARGET_HIGH     = 0.57
    TARGET_CRITICAL = 0.64

    def _find_first_crossing(target: float) -> int | None:
        for row in sweep:
            if row["win_rate"] is not None and row["win_rate"] >= target and row["count"] >= 5:
                return row["threshold"]
        return None

    rec_medium   = _find_first_crossing(TARGET_MEDIUM)
    rec_high     = _find_first_crossing(TARGET_HIGH)
    rec_critical = _find_first_crossing(TARGET_CRITICAL)

    return {
        "scored_count": len(scored),
        "sweep": sweep,
        "recommended_medium":   rec_medium,
        "recommended_high":     rec_high,
        "recommended_critical": rec_critical,
        "targets": {
            "medium":   TARGET_MEDIUM,
            "high":     TARGET_HIGH,
            "critical": TARGET_CRITICAL,
        },
    }


# ---------------------------------------------------------------------------
# Analysis: TP optimisation per (entry_type × risk_tier)
# ---------------------------------------------------------------------------

def _optimal_tp_for_group(
    trades: list[dict[str, Any]],
    tp_candidates: list[float],
    avg_sl_pips: float,
) -> dict[str, Any]:
    """
    For each candidate TP, compute EV = win_rate(MFE >= TP) * TP - (1 - win_rate) * avg_sl_pips.
    Return the TP with highest EV.
    """
    with_mfe = [
        t for t in trades
        if t.get("max_favorable_pips") is not None and t.get("risk_pips") is not None
    ]
    if len(with_mfe) < 5:
        return {"error": "insufficient_data", "count": len(with_mfe)}

    sl_values = [float(t["risk_pips"]) for t in with_mfe if float(t["risk_pips"]) > 0]
    effective_sl = sum(sl_values) / len(sl_values) if sl_values else avg_sl_pips

    results: list[dict[str, Any]] = []
    for tp in tp_candidates:
        mfe_vals = [float(t["max_favorable_pips"]) for t in with_mfe]
        wins = sum(1 for m in mfe_vals if m >= tp)
        wr = wins / len(with_mfe)
        ev = wr * tp - (1 - wr) * effective_sl
        results.append({"tp": tp, "win_rate": round(wr, 4), "ev": round(ev, 4), "count": len(with_mfe)})

    best = max(results, key=lambda r: r["ev"])
    return {
        "count": len(with_mfe),
        "avg_sl_pips": round(effective_sl, 2),
        "best_tp": best["tp"],
        "best_ev": best["ev"],
        "best_win_rate": best["win_rate"],
        "sweep": results,
    }


def _analyze_tp_by_group(
    trades: list[dict[str, Any]],
    tp_candidates: list[float] | None = None,
) -> dict[str, Any]:
    """Group by (entry_type × risk_tier) and optimise TP for each group."""
    if tp_candidates is None:
        tp_candidates = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0]

    groups: dict[str, list[dict[str, Any]]] = {}
    for t in trades:
        et = t.get("entry_type") or "unknown"
        rt = t.get("reversal_risk_tier") or "unknown"
        key = f"{et}:{rt}"
        groups.setdefault(key, []).append(t)

    # Also analyse overall
    groups["all:all"] = trades

    results: dict[str, Any] = {}
    for key, group in sorted(groups.items()):
        results[key] = _optimal_tp_for_group(group, tp_candidates, avg_sl_pips=10.0)

    return results


# ---------------------------------------------------------------------------
# Build recommendations
# ---------------------------------------------------------------------------

def _build_recommendations(
    profile_data: dict[str, Any],
    tier_analysis: dict[str, Any],
    tp_analysis: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Compare current profile values to recommendations; return list of changes.
    Each change: {field, current, recommended, delta, confidence, source}
    """
    recs: list[dict[str, Any]] = []

    # Find the Trial #7 execution policy
    policies = profile_data.get("execution_policies", [])
    t7_policy: dict[str, Any] | None = None
    for pol in policies:
        if pol.get("policy_type") == "kt_cg_trial_7":
            t7_policy = pol
            break

    if t7_policy is None:
        return [{"warning": "No kt_cg_trial_7 policy found in profile — skipping threshold recommendations"}]

    # --- Tier threshold recommendations ---
    for tier_name, rec_key in [
        ("medium",   "recommended_medium"),
        ("high",     "recommended_high"),
        ("critical", "recommended_critical"),
    ]:
        field = f"rr_tier_{tier_name}"
        current = t7_policy.get(field)
        recommended = tier_analysis.get(rec_key)
        if recommended is None:
            recs.append({
                "field": field,
                "current": current,
                "recommended": None,
                "delta": None,
                "confidence": "low",
                "reason": f"No trades found with sufficient win_rate for {tier_name} threshold",
            })
            continue
        delta = recommended - (current or 0)
        count_at_rec = next(
            (row["count"] for row in tier_analysis.get("sweep", []) if row["threshold"] == recommended),
            0,
        )
        confidence = "high" if count_at_rec >= 20 else ("medium" if count_at_rec >= 10 else "low")
        recs.append({
            "field": field,
            "current": current,
            "recommended": float(recommended),
            "delta": delta,
            "confidence": confidence,
            "sample_size_at_threshold": count_at_rec,
            "reason": f"win_rate >= {tier_analysis['targets'][tier_name]:.0%} first met at score {recommended}",
        })

    # --- TP recommendations ---
    # Overall best TP
    overall = tp_analysis.get("all:all", {})
    if "best_tp" in overall:
        current_tp = t7_policy.get("tp_pips")
        rec_tp = overall["best_tp"]
        delta = rec_tp - (current_tp or 0)
        confidence = "high" if overall.get("count", 0) >= 30 else ("medium" if overall.get("count", 0) >= 15 else "low")
        recs.append({
            "field": "tp_pips",
            "current": current_tp,
            "recommended": rec_tp,
            "delta": delta,
            "confidence": confidence,
            "sample_size": overall.get("count"),
            "best_ev": overall.get("best_ev"),
            "best_win_rate": overall.get("best_win_rate"),
            "reason": f"EV-optimal TP across all groups (EV={overall.get('best_ev'):.2f} pips)",
        })

    # Per-group TP (informational — shown in report but not applied individually)
    tp_by_group: list[dict[str, Any]] = []
    for key, result in tp_analysis.items():
        if key == "all:all" or "error" in result:
            continue
        tp_by_group.append({
            "group": key,
            "best_tp": result.get("best_tp"),
            "best_ev": result.get("best_ev"),
            "count": result.get("count"),
            "confidence": "high" if result.get("count", 0) >= 20 else "low",
        })
    if tp_by_group:
        recs.append({"informational_tp_by_group": tp_by_group})

    return recs


# ---------------------------------------------------------------------------
# Apply recommendations to profile
# ---------------------------------------------------------------------------

def _apply_recommendations(
    profile_path: Path,
    profile_data: dict[str, Any],
    recommendations: list[dict[str, Any]],
) -> list[str]:
    """Patch profile JSON in-place; return list of applied changes."""
    applied: list[str] = []

    policies = profile_data.get("execution_policies", [])
    t7_idx: int | None = None
    for i, pol in enumerate(policies):
        if pol.get("policy_type") == "kt_cg_trial_7":
            t7_idx = i
            break

    if t7_idx is None:
        return ["ERROR: No kt_cg_trial_7 policy found — nothing applied"]

    for rec in recommendations:
        if "informational_tp_by_group" in rec:
            continue
        if "warning" in rec:
            continue
        field = rec.get("field")
        recommended = rec.get("recommended")
        confidence = rec.get("confidence", "low")
        if field is None or recommended is None:
            continue
        if confidence == "low":
            applied.append(f"SKIP {field}: confidence=low (insufficient data)")
            continue

        current = policies[t7_idx].get(field)
        policies[t7_idx][field] = recommended
        applied.append(f"SET {field}: {current} -> {recommended}  (confidence={confidence})")

    # Write back
    profile_data["execution_policies"] = policies
    profile_path.write_text(json.dumps(profile_data, indent=2) + "\n", encoding="utf-8")
    return applied


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Recalibrate Trial #7 thresholds from actual trade data."
    )
    p.add_argument("--profile", required=True, help="Profile name (e.g. 'default') or full path to .json")
    p.add_argument("--days", type=int, default=30, help="Look-back window in calendar days (default: 30)")
    p.add_argument("--apply", action="store_true", help="Apply medium/high confidence recommendations to profile")
    p.add_argument(
        "--tp-candidates",
        default="2,3,4,5,6,7,8,10,12",
        help="Comma-separated TP values in pips to evaluate (default: 2,3,4,5,6,7,8,10,12)",
    )
    p.add_argument("--out", default=None, help="Write JSON report to this path (default: stdout only)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # --- Resolve profile ---
    profile_path = _resolve_profile_path(args.profile)
    print(f"Profile: {profile_path}")

    with open(profile_path, encoding="utf-8") as fh:
        profile_data = json.load(fh)

    profile_name = profile_data.get("profile_name", profile_path.stem)

    # --- Find database ---
    db_path = _db_path_for_profile(profile_name)
    print(f"Database: {db_path}")

    # --- Load trades ---
    print(f"Loading trades from last {args.days} days...")
    trades = _load_trades(db_path, args.days)
    print(f"  Found {len(trades)} closed trades")

    if not trades:
        print("No trades found — nothing to recalibrate. Exiting.")
        sys.exit(0)

    # --- Analyse ---
    tp_candidates = [float(x.strip()) for x in args.tp_candidates.split(",")]

    print("Analysing reversal risk tier thresholds...")
    tier_analysis = _analyze_tier_thresholds(trades)

    print("Analysing TP optimisation by entry type × risk tier...")
    tp_analysis = _analyze_tp_by_group(trades, tp_candidates)

    # --- Build recommendations ---
    recommendations = _build_recommendations(profile_data, tier_analysis, tp_analysis)

    # --- Assemble report ---
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "profile": str(profile_path),
        "profile_name": profile_name,
        "lookback_days": args.days,
        "total_trades_analysed": len(trades),
        "tier_threshold_analysis": {
            "scored_count": tier_analysis.get("scored_count", 0),
            "recommended_medium":   tier_analysis.get("recommended_medium"),
            "recommended_high":     tier_analysis.get("recommended_high"),
            "recommended_critical": tier_analysis.get("recommended_critical"),
            "targets": tier_analysis.get("targets"),
        },
        "tp_analysis_summary": {
            key: {k: v for k, v in val.items() if k != "sweep"}
            for key, val in tp_analysis.items()
        },
        "recommendations": recommendations,
    }

    # --- Print human-readable summary ---
    print()
    print("=" * 60)
    print("RECALIBRATION REPORT")
    print("=" * 60)
    print(f"  Trades analysed : {len(trades)}")
    print(f"  Look-back       : {args.days} days")
    print()

    ta = report["tier_threshold_analysis"]
    print("Reversal Risk Tier Thresholds:")
    for tier in ("medium", "high", "critical"):
        rec = ta.get(f"recommended_{tier}")
        label = str(rec) if rec is not None else "N/A (insufficient data)"
        print(f"  {tier:10s}: {label}")

    print()
    print("TP Optimisation:")
    overall_tp = tp_analysis.get("all:all", {})
    if "best_tp" in overall_tp:
        print(f"  Overall best TP : {overall_tp['best_tp']} pips  "
              f"(EV={overall_tp['best_ev']:.2f}, WR={overall_tp['best_win_rate']:.1%}, n={overall_tp['count']})")
    else:
        print(f"  Overall: {overall_tp.get('error', 'N/A')}")

    print()
    print("Recommendations:")
    for rec in recommendations:
        if "informational_tp_by_group" in rec:
            print("  Per-group TP breakdown:")
            for g in rec["informational_tp_by_group"]:
                print(f"    {g['group']:30s}  best_tp={g['best_tp']}  ev={g['best_ev']}  n={g['count']}")
            continue
        if "warning" in rec:
            print(f"  WARN: {rec['warning']}")
            continue
        field = rec.get("field", "?")
        cur = rec.get("current")
        new = rec.get("recommended")
        conf = rec.get("confidence", "?")
        delta = rec.get("delta")
        delta_str = f"Δ{delta:+.1f}" if delta is not None else ""
        print(f"  {field:30s} {cur} -> {new}  {delta_str}  [{conf}]")

    print()

    # --- Apply ---
    if args.apply:
        print("Applying recommendations (medium/high confidence only)...")
        applied = _apply_recommendations(profile_path, profile_data, recommendations)
        for line in applied:
            print(f"  {line}")
        report["applied"] = applied
        print()
        print(f"Profile saved: {profile_path}")
    else:
        print("Dry-run mode — use --apply to write changes to profile.")

    # --- Output JSON report ---
    report_json = json.dumps(report, indent=2)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report_json + "\n", encoding="utf-8")
        print(f"Report written to: {out_path}")
    else:
        print()
        print("--- JSON REPORT ---")
        print(report_json)


if __name__ == "__main__":
    main()
