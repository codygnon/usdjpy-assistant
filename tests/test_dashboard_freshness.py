from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import api.main as api_main
import core.dashboard_models as dashboard_models
import core.dashboard_builder as dashboard_builder
import run_loop


def test_get_dashboard_reports_stale_even_while_loop_running(monkeypatch) -> None:
    stale_ts = (datetime.now(timezone.utc) - timedelta(seconds=api_main._DASHBOARD_FILE_FRESHNESS + 30)).isoformat()

    monkeypatch.setattr(api_main, "LEAN_UI_MODE", True)
    monkeypatch.setattr(api_main, "_pick_best_dashboard_log_dir", lambda *_args, **_kwargs: Path("/tmp/dashboard-test"))
    monkeypatch.setattr(dashboard_models, "read_dashboard_state", lambda _log_dir: {"timestamp_utc": stale_ts, "filters": [], "context": []})
    monkeypatch.setattr(api_main, "_is_loop_running", lambda _profile_name: True)
    monkeypatch.setattr(api_main, "_fetch_live_positions", lambda *_args, **_kwargs: [])

    result = api_main.get_dashboard("demo-profile")

    assert result["loop_running"] is True
    assert result["stale"] is True
    assert result["stale_age_seconds"] is not None
    assert result["stale_age_seconds"] >= api_main._DASHBOARD_FILE_FRESHNESS


def test_build_dashboard_uses_position_snapshot_without_refetch(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class DummyStore:
        def list_open_trades(self, _profile_name: str):
            return []

        def get_trades_for_date(self, _profile_name: str, _date_str: str):
            return []

    class DummyAdapter:
        def get_open_positions(self, _symbol: str):
            raise AssertionError("expected open_positions_snapshot to be reused")

    def _capture_dashboard_state(_log_dir: Path, state) -> None:
        captured["state"] = state

    monkeypatch.setattr(dashboard_builder, "build_dashboard_filters", lambda **_kwargs: [])
    monkeypatch.setattr(dashboard_builder, "effective_policy_for_dashboard", lambda policy, _temp_overrides: policy)
    monkeypatch.setattr(run_loop, "write_dashboard_state", _capture_dashboard_state)

    profile = SimpleNamespace(
        pip_size=0.01,
        profile_name="demo-profile",
        active_preset_name="Demo",
        symbol="USDJPY",
    )
    tick = SimpleNamespace(bid=150.00, ask=150.02)
    snapshot = [{
        "id": "12345",
        "currentUnits": "1000",
        "price": "150.01",
        "openTime": datetime.now(timezone.utc).isoformat(),
    }]

    run_loop._build_and_write_dashboard(
        profile=profile,
        store=DummyStore(),
        log_dir=tmp_path,
        tick=tick,
        data_by_tf={"M1": pd.DataFrame()},
        mode="ARMED_AUTO",
        adapter=DummyAdapter(),
        open_positions_snapshot=snapshot,
    )

    state = captured["state"]
    assert state.positions
    assert state.positions[0].trade_id == "12345"
