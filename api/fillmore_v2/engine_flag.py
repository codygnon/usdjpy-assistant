"""Process-level engine flag — Phase 9 Step 9 (PHASE9.10).

Selects which Auto Fillmore engine handles the autonomous tick:
  - "v1" (default): the legacy `api/autonomous_fillmore.py` pipeline
  - "v2": the v2 orchestrator (api/fillmore_v2/orchestrator.run_decision)

Read from `runtime_state.json` under key `autonomous_fillmore.engine`.
Default is "v1" — flipping to "v2" is a deliberate operator action.

Per Step 1 design decision: the flag is checked at process startup AND
on every tick (cheap dict lookup). No mid-session swaps in the same
tick — once tick processing begins, the engine for that tick is locked.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

EngineId = Literal["v1", "v2"]
DEFAULT_ENGINE: EngineId = "v1"
VALID_ENGINES: tuple[EngineId, ...] = ("v1", "v2")
_CONFIG_KEY_NESTED = ("autonomous_fillmore", "engine")


def read_engine_flag(state_path: Path) -> EngineId:
    """Return 'v1' or 'v2'. Unknown / missing values fall back to 'v1'.

    Defensive: any read error or invalid value silently returns 'v1' so a
    corrupted state file cannot accidentally enable v2. Operators flip
    via `set_engine_flag` (or hand-edit + restart).
    """
    try:
        if not Path(state_path).exists():
            return DEFAULT_ENGINE
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
        section = state
        for k in _CONFIG_KEY_NESTED:
            if not isinstance(section, dict):
                return DEFAULT_ENGINE
            section = section.get(k)
            if section is None:
                return DEFAULT_ENGINE
        if section in VALID_ENGINES:
            return section  # type: ignore[return-value]
        return DEFAULT_ENGINE
    except (OSError, json.JSONDecodeError):
        return DEFAULT_ENGINE


def set_engine_flag(state_path: Path, engine: EngineId) -> None:
    """Persist the engine selection. Raises on invalid value."""
    if engine not in VALID_ENGINES:
        raise ValueError(f"engine must be one of {VALID_ENGINES}, got {engine!r}")
    state_path = Path(state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = {}
    if state_path.exists():
        try:
            with open(state_path, encoding="utf-8") as f:
                state = json.load(f)
        except (OSError, json.JSONDecodeError):
            state = {}
    section = state.setdefault("autonomous_fillmore", {})
    if not isinstance(section, dict):
        raise RuntimeError("runtime_state.json: autonomous_fillmore is not an object")
    section["engine"] = engine
    tmp = state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(state_path)
