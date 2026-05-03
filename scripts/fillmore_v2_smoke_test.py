#!/usr/bin/env python3
"""Manual smoke test: run a real LLM call through the v2 orchestrator.

NOT a pytest. NOT wired into CI. Run this locally when you want to verify
the live OpenAI client path end-to-end. Spends a few hundred tokens per run.

Usage:
    OPENAI_API_KEY=... .venv/bin/python scripts/fillmore_v2_smoke_test.py
    OPENAI_API_KEY=... .venv/bin/python scripts/fillmore_v2_smoke_test.py --side sell --score 88

Prints:
    - The rendered system prompt (truncated)
    - The rendered user context (full)
    - The model's raw JSON output
    - Whether the parser accepted it
    - Which validators (if any) overrode the decision
    - Final decision

No DB writes. No autonomous_fillmore wiring. Pure stdout.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.fillmore_v2.llm_client import OpenAILlmClient
from api.fillmore_v2.orchestrator import run_decision
from api.fillmore_v2.snapshot import (
    LevelAgeMetadata,
    LevelPacket,
    Snapshot,
    new_snapshot_id,
    now_utc_iso,
    reset_blocking_strikes,
)
from api.fillmore_v2.telemetry import pip_value_per_lot, risk_after_fill_usd


def _build_demo_snapshot(*, side: str, score: float) -> Snapshot:
    pip_v = pip_value_per_lot(150.0)
    risk = risk_after_fill_usd(proposed_lots=2.0, sl_pips=8.0, pip_value_per_lot_usd=pip_v)
    return Snapshot(
        snapshot_id=new_snapshot_id(),
        created_utc=now_utc_iso(),
        tick_mid=150.0, tick_bid=149.99, tick_ask=150.01, spread_pips=1.0,
        account_equity=100_000.0,
        open_lots_buy=0.0, open_lots_sell=0.0,
        unrealized_pnl_buy=0.0, unrealized_pnl_sell=0.0,
        pip_value_per_lot=pip_v,
        risk_after_fill_usd=risk,
        rolling_20_trade_pnl=0.0, rolling_20_lot_weighted_pnl=0.0,
        level_packet=LevelPacket(
            side="buy_support" if side == "buy" else "sell_resistance",
            level_price=149.50 if side == "buy" else 150.30,
            level_quality_score=score,
            distance_pips=-50.0 if side == "buy" else -30.0,
            profit_path_blocker_distance_pips=30.0,
            structural_origin="h1_swing_low" if side == "buy" else "pdh",
        ),
        level_age_metadata=LevelAgeMetadata(touch_count=3, broken_then_reclaimed=False),
        rendered_prompt="placeholder",
        rendered_context_json="{}",
        proposed_side=side, sl_pips=8.0, tp_pips=16.0,
        timeframe_alignment="aligned_buy" if side == "buy" else "aligned_sell",
        macro_bias="neutral" if side == "buy" else "bearish",
        catalyst_category="material",
        active_sessions=["ny"],
        session_overlap=None,
        volatility_regime="normal",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=["buy", "sell"], default="buy")
    parser.add_argument("--score", type=float, default=78.0,
                        help="level_quality_score (buy >=70 needed; sell >=85)")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--no-call", action="store_true",
                        help="Print rendered prompt and exit without calling LLM")
    args = parser.parse_args()

    if not args.no_call and not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. Use --no-call to print prompt only.",
              file=sys.stderr)
        return 2

    snap = _build_demo_snapshot(side=args.side, score=args.score)

    from api.fillmore_v2.system_prompt import render_full_prompt
    sys_p, user_p = render_full_prompt(snap)
    print("=== SYSTEM PROMPT (first 800 chars) ===")
    print(sys_p[:800])
    print("...")
    print("=== USER CONTEXT ===")
    print(user_p)

    if args.no_call:
        return 0

    print(f"\n=== Calling {args.model} ===")
    with tempfile.TemporaryDirectory(prefix="fillmore_v2_smoke_") as td:
        td_path = Path(td)
        reset_blocking_strikes(td_path)
        client = OpenAILlmClient()
        result = run_decision(
            snap, llm_client=client, profile_dir=td_path,
            db_path=None, model=args.model,
        )

    print(f"\n=== RAW LLM OUTPUT ===")
    if result.llm_call:
        print(result.llm_call.raw_text)
        if result.llm_call.error:
            print(f"!! transport error: {result.llm_call.error}")
    print(f"\n=== ORCHESTRATOR RESULT ===")
    print(f"final_decision   = {result.final_decision}")
    print(f"reason           = {result.reason}")
    print(f"deterministic_lots = {result.deterministic_lots}")
    if result.gates_final:
        for c in result.gates_final.candidates:
            print(f"  gate {c.gate_id:25s} eligible={c.eligible} score={c.score:.2f} reason={c.reason_code}")
    if result.pre_veto_summary:
        for f in result.pre_veto_summary.fires:
            print(f"  pre-veto FIRED: {f.veto_id} ({f.reason_code})")
    if result.validator_summary:
        for o in result.validator_summary.overrides:
            print(f"  validator OVERRIDE: {o.validator_id} ({o.reason_code})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
