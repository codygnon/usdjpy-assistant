#!/usr/bin/env python3
"""
Agent 1 helper: summarize Phase 3 minute diagnostics by ownership_cell/regime.

Reads run_loop's phase3_minute_diagnostics.log lines and aggregates the most
frequent blocking/decision patterns with ownership fields:
  - ownership_cell
  - regime
  - defensive_flags

Usage:
  python3 scripts/diagnostic_phase3_ownership_pain_log.py --log "/path/to/phase3_minute_diagnostics.log"
  python3 scripts/diagnostic_phase3_ownership_pain_log.py --log ... --output-json research_out/pain_summary.json --filter-cell ambiguous/er_low/der_neg
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _parse_line(line: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in line.strip().split("\t"):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def summarize_pain_log(
    path: Path,
    *,
    top: int = 20,
    filter_cell: str | None = None,
) -> dict[str, Any]:
    by_cell_reason: Counter[tuple[str, str]] = Counter()
    by_regime_flags: Counter[tuple[str, str]] = Counter()
    reason_counts: Counter[str] = Counter()
    by_cell_samples: defaultdict[str, list[str]] = defaultdict(list)
    total = 0
    placed = 0

    for line in path.read_text(encoding="utf-8").splitlines():
        rec = _parse_line(line)
        if not rec:
            continue
        total += 1
        if rec.get("placed", "0") == "1":
            placed += 1
        cell = rec.get("ownership_cell", "")
        regime = rec.get("regime", "")
        flags = rec.get("defensive_flags", "")
        reason = rec.get("reason", "")
        if reason:
            reason_counts[reason] += 1
        if cell or reason:
            by_cell_reason[(cell, reason)] += 1
            if len(by_cell_samples[cell]) < 3 and reason:
                by_cell_samples[cell].append(reason)
        if regime or flags:
            by_regime_flags[(regime, flags)] += 1

    out: dict[str, Any] = {
        "log_path": str(path.resolve()),
        "rows_total": total,
        "rows_placed": placed,
        "rows_no_trade": total - placed,
        "top_reasons": [{"reason": r, "count": n} for r, n in reason_counts.most_common(top)],
        "top_regime_defensive_flags": [
            {"regime": reg, "defensive_flags": flg, "count": n}
            for (reg, flg), n in by_regime_flags.most_common(top)
        ],
        "top_ownership_cell_reason": [
            {"ownership_cell": cell, "reason": reason, "count": n}
            for (cell, reason), n in by_cell_reason.most_common(top)
        ],
        "cell_samples": {c: v for c, v in by_cell_samples.items() if c},
    }

    if filter_cell:
        fc = filter_cell.strip()
        cell_reasons = Counter()
        cell_regime_flags = Counter()
        cell_rows = 0
        for (cell, reason), n in by_cell_reason.items():
            if cell == fc:
                cell_rows += n
                if reason:
                    cell_reasons[reason] += n
        for (reg, flg), n in by_regime_flags.items():
            # approximate: count rows where we need cell match — re-scan
            pass
        # Second pass for filter_cell regime/flags distribution
        cell_regime_flags_ctr: Counter[tuple[str, str]] = Counter()
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = _parse_line(line)
            if not rec:
                continue
            if rec.get("ownership_cell", "") != fc:
                continue
            cell_regime_flags_ctr[(rec.get("regime", ""), rec.get("defensive_flags", ""))] += 1

        out["filter_cell"] = fc
        out["target_cell"] = {
            "rows_with_cell": cell_rows,
            "top_reasons": [{"reason": r, "count": n} for r, n in cell_reasons.most_common(top)],
            "top_regime_defensive_flags": [
                {"regime": reg, "defensive_flags": flg, "count": n}
                for (reg, flg), n in cell_regime_flags_ctr.most_common(top)
            ],
        }

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Path to phase3_minute_diagnostics.log")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument(
        "--output-json",
        default="",
        help="Write full summary JSON to this path (in addition to stdout).",
    )
    ap.add_argument(
        "--filter-cell",
        default="",
        help="If set, include per-cell breakdown for this ownership_cell (e.g. ambiguous/er_low/der_neg).",
    )
    args = ap.parse_args()

    path = Path(args.log)
    if not path.exists():
        print(f"Missing log: {path}")
        return 1

    filter_cell = args.filter_cell.strip() or None
    summary = summarize_pain_log(path, top=args.top, filter_cell=filter_cell)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")

    total = summary["rows_total"]
    placed = summary["rows_placed"]
    print(f"Rows={total} placed={placed} no_trade={total - placed}")
    print("\nTop reasons:")
    for row in summary["top_reasons"][: args.top]:
        print(f"  {row['count']:5d}  {row['reason']}")

    print("\nTop regime + defensive_flags:")
    for row in summary["top_regime_defensive_flags"][: args.top]:
        print(f"  {row['count']:5d}  regime={row['regime'] or '-'} flags={row['defensive_flags'] or '-'}")

    print("\nTop ownership_cell + reason:")
    for row in summary["top_ownership_cell_reason"][: args.top]:
        print(f"  {row['count']:5d}  cell={row['ownership_cell'] or '-'}  reason={row['reason']}")

    print("\nCell samples:")
    for cell, samples in list(summary["cell_samples"].items())[: args.top]:
        if not cell:
            continue
        print(f"  {cell}: {samples}")

    if filter_cell and "target_cell" in summary:
        tc = summary["target_cell"]
        print(f"\n--- Filter cell: {filter_cell} (rows={tc['rows_with_cell']}) ---")
        print("Top reasons:")
        for row in tc.get("top_reasons", [])[: args.top]:
            print(f"  {row['count']:5d}  {row['reason']}")
        print("Top regime + defensive_flags:")
        for row in tc.get("top_regime_defensive_flags", [])[: args.top]:
            print(f"  {row['count']:5d}  regime={row['regime'] or '-'} flags={row['defensive_flags'] or '-'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
