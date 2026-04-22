from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


def load_json_state(path: str | Path, *, default: dict[str, Any] | None = None) -> dict[str, Any]:
    p = Path(path)
    fallback = dict(default or {})
    if not p.exists():
        backup = _backup_path(p)
        if backup.exists():
            try:
                return json.loads(backup.read_text(encoding="utf-8")) or dict(fallback)
            except Exception:
                return dict(fallback)
        return dict(fallback)
    try:
        return json.loads(p.read_text(encoding="utf-8")) or dict(fallback)
    except Exception:
        backup = _backup_path(p)
        if backup.exists():
            try:
                return json.loads(backup.read_text(encoding="utf-8")) or dict(fallback)
            except Exception:
                return dict(fallback)
        return dict(fallback)


def save_json_state(path: str | Path, data: dict[str, Any], *, indent: int = 2, trailing_newline: bool = True) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=indent, default=str)
    if trailing_newline:
        text += "\n"

    fd, tmp_name = tempfile.mkstemp(dir=str(p.parent), prefix=f".{p.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, p)
        try:
            shutil.copyfile(p, _backup_path(p))
        except Exception:
            pass
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        raise


def _backup_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".bak")
