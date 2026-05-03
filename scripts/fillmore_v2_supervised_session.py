#!/usr/bin/env python3
"""Run one local supervised v2 paper session without broker order placement.

This is a launch-readiness harness, not production code. It flips a temporary
runtime_state.json to engine='v2', dispatches one v2 paper tick through the
real v1->v2 bridge with a fake adapter and fake LLM, and prints the DB path for
`scripts/fillmore_v2_first_tick_check.py`.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api import suggestion_tracker
from api.fillmore_v2 import persistence
from api.fillmore_v2.engine_flag import read_engine_flag, set_engine_flag
from api.fillmore_v2.llm_client import FakeLlmClient
from api.fillmore_v2.v1_bridge import dispatch_v2_tick


class FakeProfile:
    pip_size = 0.01
    symbol = "USD_JPY"


class FakeTick:
    bid = 149.99
    ask = 150.01


class FakeAccount:
    equity = 100_000.0


class FakeAdapter:
    def get_account_info(self):
        return FakeAccount()

    def get_open_positions(self, symbol):
        return []


def _bars() -> dict[str, list[dict[str, float | str]]]:
    rows = []
    for i in range(24):
        close = 149.70 + i * 0.012
        rows.append({
            "time": f"2026-05-02T00:{i:02d}:00+00:00",
            "open": close - 0.006,
            "high": 150.30,
            "low": close - 0.035,
            "close": close,
        })
    return {"M1": rows, "M5": rows, "H1": rows}


def _place_json() -> str:
    return json.dumps({
        "decision": "place",
        "primary_thesis": "Momentum continuation has side-aligned path room before the next blocker.",
        "caveats_detected": [],
        "level_quality_claim": {
            "claim": "acceptable",
            "evidence_field": "side_normalized_level_packet.profit_path_blocker_distance_pips",
            "score_cited": 65,
        },
        "loss_asymmetry_argument": "SL 14p, TP 17p, blocker 20p, spread 2p, volatility normal.",
        "invalid_if": ["price loses the momentum shelf"],
        "evidence_refs": [
            "side_normalized_level_packet.profit_path_blocker_distance_pips",
            "sl_pips",
            "tp_pips",
            "spread_pips",
            "volatility_regime",
        ],
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=None, help="Optional directory for temp state/DB")
    args = parser.parse_args()

    root = args.workdir or Path(tempfile.mkdtemp(prefix="fillmore_v2_supervised_"))
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "runtime_state.json"
    db_path = root / "ai_suggestions.sqlite"
    suggestion_tracker.init_db(db_path)
    persistence.init_v2_schema(db_path)

    set_engine_flag(state_path, "v2")
    result = dispatch_v2_tick(
        profile=FakeProfile(),
        profile_name="supervised",
        state_path=state_path,
        tick=FakeTick(),
        adapter=FakeAdapter(),
        db_path=db_path,
        data_by_tf=_bars(),
        llm_client=FakeLlmClient(static_response=_place_json()),
        stage="paper",
    )
    print(f"engine={read_engine_flag(state_path)}")
    print(f"final_decision={result.final_decision}")
    print(f"reason={result.reason}")
    print(f"selected_gate={result.snapshot.selected_gate_id}")
    print(f"deterministic_lots={result.deterministic_lots}")
    print(f"db_path={db_path}")
    print(f"state_path={state_path}")
    return 0 if result.final_decision in {"place", "skip", "no_gate"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
