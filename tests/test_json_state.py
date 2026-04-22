from __future__ import annotations

import json

from core.execution_state import RuntimeState, load_state, save_state
from core.json_state import load_json_state, save_json_state


def test_save_json_state_writes_backup(tmp_path):
    path = tmp_path / "runtime_state.json"

    save_json_state(path, {"mode": "DISARMED"}, indent=2, trailing_newline=True)

    assert json.loads(path.read_text(encoding="utf-8"))["mode"] == "DISARMED"
    assert json.loads(path.with_suffix(".json.bak").read_text(encoding="utf-8"))["mode"] == "DISARMED"


def test_load_json_state_falls_back_to_backup_when_main_is_corrupt(tmp_path):
    path = tmp_path / "runtime_state.json"
    backup = path.with_suffix(".json.bak")

    backup.write_text('{"mode":"ARMED_MANUAL_CONFIRM"}', encoding="utf-8")
    path.write_text("{bad json", encoding="utf-8")

    loaded = load_json_state(path, default={})

    assert loaded["mode"] == "ARMED_MANUAL_CONFIRM"


def test_execution_state_roundtrip_uses_json_state_storage(tmp_path):
    path = tmp_path / "runtime_state.json"
    state = RuntimeState(mode="ARMED_MANUAL_CONFIRM", kill_switch=True)

    save_state(path, state)
    loaded = load_state(path)

    assert loaded.mode == "ARMED_MANUAL_CONFIRM"
    assert loaded.kill_switch is True
    assert path.with_suffix(".json.bak").exists()
