"""Separate runtime state file for v2.

v1 reads/writes runtime_state.json via core.json_state. v2 uses its own file
so concurrent or cross-restart writes cannot corrupt v1 state, and so any v1
post-mortem can read its history without v2 noise. Process-level engine flag
guarantees only one engine is active at a time, so this file has a single
writer at any moment.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()
STATE_FILE_NAME = "runtime_state_fillmore_v2.json"


def state_path(profile_dir: Path) -> Path:
    return Path(profile_dir) / STATE_FILE_NAME


def load_state(profile_dir: Path) -> dict[str, Any]:
    p = state_path(profile_dir)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def save_state(profile_dir: Path, state: dict[str, Any]) -> None:
    p = state_path(profile_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with _LOCK:
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True, default=str))
        tmp.replace(p)


def update_state(profile_dir: Path, **fields: Any) -> dict[str, Any]:
    s = load_state(profile_dir)
    s.update(fields)
    save_state(profile_dir, s)
    return s
