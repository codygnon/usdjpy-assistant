#!/usr/bin/env python3
"""
Run post–Variant K research artifacts in dependency order.

Covers: regime validation, ownership diagnostic, chart authorization loop,
Phase A chart-first routing, offensive shadow ledger, Agent 3 program,
full latent regime router (500k/1000k).

Usage:
  python3 scripts/run_post_variant_k_full_stack.py
  python3 scripts/run_post_variant_k_full_stack.py --skip-chart-auth
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "research_out"


def _run(label: str, cmd: list[str]) -> dict:
    t0 = time.perf_counter()
    print(f"\n{'='*72}\n[{label}]\n  {' '.join(cmd)}\n{'='*72}", flush=True)
    p = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.perf_counter() - t0
    row = {"step": label, "cmd": cmd, "returncode": p.returncode, "seconds": round(elapsed, 2)}
    print(f"[{label}] exit={p.returncode} in {elapsed:.1f}s", flush=True)
    if p.returncode != 0:
        raise RuntimeError(f"Step failed: {label} (exit {p.returncode})")
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-chart-auth", action="store_true")
    ap.add_argument("--skip-regime-validation", action="store_true")
    ap.add_argument("--skip-latent-router", action="store_true")
    args = ap.parse_args()

    py = sys.executable
    rows: list[dict] = []
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    index_path = OUT / f"post_variant_k_stack_index_{stamp}.json"

    def _write_partial() -> None:
        partial = {"generated_utc": stamp, "status": "partial_or_complete", "steps": rows}
        index_path.write_text(json.dumps(partial, indent=2), encoding="utf-8")

    def run(label: str, cmd: list[str]) -> None:
        rows.append(_run(label, cmd))
        _write_partial()

    # 1–4: ownership cells, Phase A, ledger, Agent 3 (downstream of Phase A + phaseb JSON)
    run(
        "diagnostic_strategy_ownership",
        [py, str(ROOT / "scripts" / "diagnostic_strategy_ownership.py")],
    )
    run(
        "diagnostic_chart_first_routing",
        [py, str(ROOT / "scripts" / "diagnostic_chart_first_routing.py")],
    )
    run(
        "offensive_shadow_ledger",
        [py, str(ROOT / "scripts" / "offensive_shadow_ledger.py")],
    )
    run(
        "agent3_portability_program",
        [py, str(ROOT / "scripts" / "agent3_portability_program.py")],
    )

    ds500 = str(OUT / "USDJPY_M1_OANDA_500k.csv")
    ds1m = str(OUT / "USDJPY_M1_OANDA_1000k.csv")

    if not args.skip_regime_validation:
        run(
            "validate_regime_classifier_500k",
            [
                py,
                str(ROOT / "scripts" / "validate_regime_classifier.py"),
                "--input",
                ds500,
                "--output",
                str(OUT / f"post_variant_k_stack_regime_validation_500k_{stamp}.json"),
                "--experiment-name",
                f"post_variant_k_stack_500k_{stamp}",
            ],
        )
        run(
            "validate_regime_classifier_1000k",
            [
                py,
                str(ROOT / "scripts" / "validate_regime_classifier.py"),
                "--input",
                ds1m,
                "--output",
                str(OUT / f"post_variant_k_stack_regime_validation_1000k_{stamp}.json"),
                "--experiment-name",
                f"post_variant_k_stack_1000k_{stamp}",
            ],
        )

    if not args.skip_latent_router:
        prefix500 = str(OUT / f"post_variant_k_stack_latent_500k_{stamp}")
        prefix1m = str(OUT / f"post_variant_k_stack_latent_1000k_{stamp}")
        run(
            "backtest_phase3_full_latent_router_500k",
            [
                py,
                str(ROOT / "scripts" / "backtest_phase3_full_latent_router.py"),
                "--dataset",
                ds500,
                "--output-prefix",
                prefix500,
            ],
        )
        run(
            "backtest_phase3_full_latent_router_1000k",
            [
                py,
                str(ROOT / "scripts" / "backtest_phase3_full_latent_router.py"),
                "--dataset",
                ds1m,
                "--output-prefix",
                prefix1m,
            ],
        )

    if not args.skip_chart_auth:
        run(
            "diagnostic_chart_authorization_loop_500k",
            [
                py,
                str(ROOT / "scripts" / "diagnostic_chart_authorization_loop.py"),
                "--dataset",
                ds500,
                "--output",
                str(OUT / f"post_variant_k_stack_chart_authorization_500k_{stamp}.json"),
            ],
        )
        run(
            "diagnostic_chart_authorization_loop_1000k",
            [
                py,
                str(ROOT / "scripts" / "diagnostic_chart_authorization_loop.py"),
                "--dataset",
                ds1m,
                "--output",
                str(OUT / f"post_variant_k_stack_chart_authorization_1000k_{stamp}.json"),
            ],
        )

    index = {
        "generated_utc": stamp,
        "status": "complete",
        "steps": rows,
        "notes": {
            "variant_k_baseline_reports": [
                str(OUT / "phase3_integrated_variant_k_500k_report.json"),
                str(OUT / "phase3_integrated_variant_k_1000k_report.json"),
            ],
            "engine_code": [
                "core/regime_classifier.py",
                "core/ownership_table.py",
                "core/chart_authorization.py",
                "core/chart_shadow_contracts.py",
                "core/phase3_integrated_engine.py",
            ],
        },
    }
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"\nWrote {index_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
