from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.ai_exit_strategies import compute_post_tp1_trail_sl
from api import paper_fillmore


class _PaperStore:
    def __init__(self, open_rows: list[dict]) -> None:
        self._open_rows = list(open_rows)
        self.updated: list[tuple[str, dict]] = []

    def list_open_trades(self, profile_name: str) -> list[dict]:
        return list(self._open_rows)

    def update_trade(self, trade_id: str, updates: dict) -> None:
        self.updated.append((trade_id, dict(updates)))
        for row in self._open_rows:
            if str(row.get("trade_id")) == trade_id:
                row.update(updates)


def test_compute_post_tp1_trail_sl_buy_basic() -> None:
    trail_sl = compute_post_tp1_trail_sl(
        trade_side="buy",
        entry_price=160.000,
        tp1_pips=6.0,
        pip_size=0.01,
        high_water_mark=160.080,
        lock_sl=160.017,
    )
    assert trail_sl == 160.056


def test_compute_post_tp1_trail_sl_sell_basic() -> None:
    trail_sl = compute_post_tp1_trail_sl(
        trade_side="sell",
        entry_price=160.000,
        tp1_pips=6.0,
        pip_size=0.01,
        high_water_mark=159.920,
        lock_sl=159.983,
    )
    assert trail_sl == 159.944


def test_compute_post_tp1_trail_sl_floored_at_lock() -> None:
    trail_sl = compute_post_tp1_trail_sl(
        trade_side="buy",
        entry_price=160.000,
        tp1_pips=6.0,
        pip_size=0.01,
        high_water_mark=160.030,
        lock_sl=160.017,
    )
    assert trail_sl == 160.017


def test_compute_post_tp1_trail_sl_tightens_on_extension() -> None:
    trail_sl = compute_post_tp1_trail_sl(
        trade_side="buy",
        entry_price=160.000,
        tp1_pips=6.0,
        pip_size=0.01,
        high_water_mark=160.120,
        lock_sl=160.017,
        extended_threshold=1.5,
        extended_trail_fraction=0.30,
    )
    assert trail_sl == 160.102


def test_compute_post_tp1_trail_sl_is_monotonic_for_buy_hwm() -> None:
    lock_sl = 160.017
    peaks = [160.060, 160.080, 160.100, 160.120, 160.140]
    values = [
        compute_post_tp1_trail_sl(
            trade_side="buy",
            entry_price=160.000,
            tp1_pips=6.0,
            pip_size=0.01,
            high_water_mark=peak,
            lock_sl=lock_sl,
        )
        for peak in peaks
    ]
    assert values == sorted(values)


def test_paper_hwm_trail_uses_proportional_stop() -> None:
    profile = SimpleNamespace(profile_name="kumatora2", symbol="USDJPY", pip_size=0.01)
    store = _PaperStore(
        [
            {
                "trade_id": "ai_autonomous:paper:1",
                "side": "buy",
                "entry_price": 160.000,
                "size_lots": 0.1,
                "stop_price": 160.017,
                "target_price": 160.200,
                "managed_trail_mode": "hwm",
                "managed_tp1_pips": 6.0,
                "managed_tp1_close_pct": 70.0,
                "managed_be_plus_pips": 0.5,
                "tp1_partial_done": 1,
                "tp1_triggered": 1,
                "breakeven_sl_price": 160.017,
                "peak_price": None,
                "config_json": json.dumps(
                    {
                        "source": "autonomous_fillmore",
                        "paper": True,
                        "exit_strategy": "tp1_be_hwm_trail",
                    }
                ),
            }
        ]
    )

    paper_fillmore._tick_one_paper_trade(
        profile=profile,
        profile_name="kumatora2",
        state_path=Path("/tmp/runtime_state.json"),
        store=store,
        trade_row=store.list_open_trades("kumatora2")[0],
        bid=160.120,
        ask=160.140,
        mid=160.130,
        spread=0.02,
        pip=0.01,
        m1_df=None,
        m5_df=None,
    )

    assert ("ai_autonomous:paper:1", {"peak_price": 160.13, "stop_price": 160.112}) in store.updated
