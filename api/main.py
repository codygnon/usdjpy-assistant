"""FastAPI backend for USDJPY Assistant.

Run with: uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Paths (use persistent volume on Railway when RAILWAY_VOLUME_MOUNT_PATH is set)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
_data_base_env = os.environ.get("RAILWAY_VOLUME_MOUNT_PATH") or os.environ.get("USDJPY_DATA_DIR")
DATA_BASE = Path(_data_base_env) if _data_base_env else BASE_DIR
PROFILES_DIR = DATA_BASE / "profiles"
LOGS_DIR = DATA_BASE / "logs"
FRONTEND_DIR = BASE_DIR / "frontend" / "dist"

# Ensure persistent data dirs exist when using a volume
if DATA_BASE != BASE_DIR:
    (DATA_BASE / "profiles").mkdir(parents=True, exist_ok=True)
    (DATA_BASE / "logs").mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Imports from existing modules
# ---------------------------------------------------------------------------
sys.path.insert(0, str(BASE_DIR))

from core.execution_state import RuntimeState, load_state, save_state
from core.phase3_operator import (
    build_phase3_acceptance_payload,
    build_phase3_defensive_monitor_payload,
    build_phase3_provenance_payload,
    enrich_phase3_attribution,
    extract_phase3_reason_text,
    infer_phase3_session_from_signal_id,
    parse_phase3_blocking_filter_ids,
)
from core.presets import PresetId, apply_preset, get_preset_patch, list_presets
from core.profile import (
    ProfileV1,
    ProfileV1AllowExtra,
    default_profile_for_name,
    get_effective_risk,
    load_profile_v1,
    save_profile_v1,
)
from storage.sqlite_store import SqliteStore

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(title="USDJPY Assistant API", version="1.0.0")

# CORS for local development (frontend on different port)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1024)

# Track running loop processes: {profile_name: subprocess.Popen}
_loop_processes: dict[str, subprocess.Popen] = {}

# ---------------------------------------------------------------------------
# Module-level caches for performance
# ---------------------------------------------------------------------------
import time as _time

# Candle cache: {(symbol, tf, count): (timestamp, dataframe)}
_api_candle_cache: dict[tuple[str, str, int], tuple[float, Any]] = {}
_CANDLE_TTL: dict[str, float] = {
    "M1": 55, "M3": 175, "M5": 300, "M15": 300, "M30": 300,
    "H1": 600, "H4": 600, "D": 300,
}

def _get_bars_cached(adapter: Any, symbol: str, tf: str, count: int = 700) -> Any:
    """Fetch bars with module-level TTL cache."""
    key = (symbol, tf, count)
    now = _time.monotonic()
    cached = _api_candle_cache.get(key)
    ttl = _CANDLE_TTL.get(tf, 300)
    if cached and (now - cached[0]) < ttl:
        return cached[1]
    df = adapter.get_bars(symbol, tf, count=count)
    _api_candle_cache[key] = (now, df)
    return df

# Backfill MAE/MFE throttle: {profile_name: last_run_timestamp}
_backfill_last_run: dict[str, float] = {}
_BACKFILL_INTERVAL = 60.0  # seconds

# Lean API mode: reduce broker-backed read amplification for UI pages.
LEAN_UI_MODE = os.environ.get("LEAN_UI_MODE", "true").strip().lower() in {"1", "true", "yes", "on"}

# Verbose stdout logging (Railway log volume / CPU); default off.
API_VERBOSE_LOGS = os.environ.get("API_VERBOSE_LOGS", "").strip().lower() in {"1", "true", "yes", "on"}


def _run_in_threadpool_with_timeout(
    fn: Callable[..., Any],
    timeout_seconds: float,
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run *fn* in a worker thread and return its result (or propagate its exception).

    Uses ``shutdown(wait=False)`` so a timed-out ``fut.result`` does **not** block until a
    stuck broker/HTTP call finishes. The default ``with ThreadPoolExecutor`` path calls
    ``shutdown(wait=True)`` on exit, which defeats timeouts and can trigger Railway/proxy
    **502 Application failed to respond** under slow OANDA responses.
    """
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        fut = executor.submit(fn, *args, **kwargs)
        return fut.result(timeout=timeout_seconds)
    finally:
        executor.shutdown(wait=False)

# Short-lived endpoint response cache for expensive read endpoints.
_endpoint_response_cache: dict[str, tuple[float, Any]] = {}
_ENDPOINT_CACHE_TTL_SECONDS: dict[str, float] = {
    "trade_history_detail": 10.0,
    "advanced_analytics": 10.0,
    "stats_by_preset": 10.0,
    "mt5_report": 10.0,
    "phase3_decisions": 5.0,
    "phase3_provenance": 5.0,
    "phase3_blockers": 8.0,
}


def _cache_compose_key(endpoint: str, *parts: Any) -> str:
    safe_parts = [str(p) for p in parts]
    return f"{endpoint}|" + "|".join(safe_parts)


def _cache_get(endpoint: str, key: str) -> Any | None:
    ttl = _ENDPOINT_CACHE_TTL_SECONDS.get(endpoint)
    if ttl is None:
        return None
    cached = _endpoint_response_cache.get(key)
    if not cached:
        return None
    ts, payload = cached
    if (_time.monotonic() - ts) >= ttl:
        _endpoint_response_cache.pop(key, None)
        return None
    return payload


def _cache_set(endpoint: str, key: str, payload: Any) -> None:
    if endpoint in _ENDPOINT_CACHE_TTL_SECONDS:
        _endpoint_response_cache[key] = (_time.monotonic(), payload)


def _cache_invalidate_profile(profile_name: str) -> None:
    token = f"|{profile_name}|"
    stale_keys = [k for k in _endpoint_response_cache.keys() if token in k]
    for k in stale_keys:
        _endpoint_response_cache.pop(k, None)
    _dashboard_live_cache.pop(profile_name, None)
    lean_rm = [k for k in list(_lean_dashboard_cache.keys()) if k.startswith(f"{profile_name}\x1f")]
    for k in lean_rm:
        _lean_dashboard_cache.pop(k, None)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _list_profile_paths() -> list[Path]:
    if not PROFILES_DIR.exists():
        return []
    return sorted([p for p in PROFILES_DIR.rglob("*.json") if p.is_file()])


def _dedupe_profile_paths_by_stem(paths: list[Path]) -> list[Path]:
    """Keep one profile per stem to avoid runtime/log collisions in the UI."""
    best_by_stem: dict[str, Path] = {}

    def _rank(path: Path) -> tuple[int, str]:
        try:
            rel = path.relative_to(PROFILES_DIR)
        except Exception:
            rel = path
        # Prefer shallower paths like profiles/foo.json over profiles/v1/foo.json.
        return (len(rel.parts), str(rel))

    for path in paths:
        stem = path.stem
        current = best_by_stem.get(stem)
        if current is None or _rank(path) < _rank(current):
            best_by_stem[stem] = path
    return sorted(best_by_stem.values(), key=lambda p: str(p))


_oanda_cleanup_done: set[str] = set()

def _store_for(profile_name: str, log_dir: Optional[Path] = None) -> SqliteStore:
    log_dir = log_dir or (LOGS_DIR / profile_name)
    log_dir.mkdir(parents=True, exist_ok=True)
    store = SqliteStore(log_dir / "assistant.db")
    store.init_db()
    # One-time cleanup of duplicate oanda_ imports (once per profile per process)
    _cleanup_key = str(log_dir)
    if _cleanup_key not in _oanda_cleanup_done:
        _oanda_cleanup_done.add(_cleanup_key)
        try:
            _active = log_dir.name
            deleted = store.delete_duplicate_oanda_imports(_active)
            if deleted > 0:
                print(f"[api] cleaned up {deleted} duplicate oanda_ import(s) for '{_active}'")
        except Exception:
            pass
    return store


def _runtime_state_path(profile_name: str) -> Path:
    return LOGS_DIR / profile_name / "runtime_state.json"


def _loop_pid_path(profile_name: str) -> Path:
    return LOGS_DIR / profile_name / "loop.pid"


def _research_out_path(filename: str) -> Path:
    return BASE_DIR / "research_out" / filename


def _resolve_phase3_preset_id_api(profile_name: str | None = None, profile_path: str | None = None) -> str | None:
    try:
        from core.phase3_package_spec import PHASE3_DEFENDED_PRESET_ID

        defended = str(PHASE3_DEFENDED_PRESET_ID).strip().lower()
        if profile_path:
            p = _resolve_profile_path(profile_path)
            if p.exists():
                prof = load_profile_v1(p)
                active = str(getattr(prof, "active_preset_name", "") or "").strip().lower()
                if active == defended:
                    return PHASE3_DEFENDED_PRESET_ID
                for pol in list(getattr(getattr(prof, "execution", None), "policies", []) or []):
                    if not getattr(pol, "enabled", True):
                        continue
                    if str(getattr(pol, "type", "") or "") != "phase3_integrated":
                        continue
                    if str(getattr(pol, "id", "") or "").strip().lower() == defended:
                        return PHASE3_DEFENDED_PRESET_ID
                return getattr(prof, "active_preset_name", None)
        return None
    except Exception:
        return None


def _phase3_policy_id_from_trade_row_api(trade_row: dict[str, Any]) -> str | None:
    trade_id = str(trade_row.get("trade_id") or "")
    if trade_id.startswith("phase3_integrated:"):
        parts = trade_id.split(":")
        if len(parts) >= 2 and str(parts[1] or "").strip():
            return str(parts[1]).strip()
    notes = str(trade_row.get("notes") or "")
    if notes.startswith("auto:phase3_integrated:"):
        parts = notes.split(":")
        if len(parts) >= 3 and str(parts[2] or "").strip():
            return str(parts[2]).strip()
    return None


def _load_phase3_sizing_cfg_api(profile_name: str | None = None, profile_path: str | None = None, preset_id: str | None = None) -> dict[str, Any]:
    try:
        from core.phase3_integrated_engine import load_phase3_sizing_config

        effective_preset_id = preset_id if preset_id is not None else _resolve_phase3_preset_id_api(profile_name, profile_path)
        return load_phase3_sizing_config(preset_id=effective_preset_id) or {}
    except Exception:
        return {}


def _enrich_phase3_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [enrich_phase3_attribution(row) for row in rows]


def _apply_phase3_sync_close_state_update(
    *,
    profile_name: str,
    trade_row: dict[str, Any],
    updates: dict[str, Any],
) -> None:
    entry_type = str(trade_row.get("entry_type") or "")
    if not entry_type.startswith("phase3:"):
        return
    try:
        from core.phase3_integrated_engine import (
            load_phase3_sizing_config,
            infer_phase3_entry_session,
            phase3_trade_key_date,
            apply_phase3_session_outcome,
        )

        state_path = _runtime_state_path(profile_name)
        runtime_data: dict[str, Any] = {}
        if state_path.exists():
            try:
                runtime_data = json.loads(state_path.read_text(encoding="utf-8"))
            except Exception:
                runtime_data = {}
        phase3_state = dict(runtime_data.get("phase3_state", {}) or {})

        exit_ts = updates.get("exit_timestamp_utc") or pd.Timestamp.now(tz="UTC").isoformat()
        now_utc = pd.Timestamp(exit_ts)
        if now_utc.tzinfo is None:
            now_utc = now_utc.tz_localize("UTC")
        else:
            now_utc = now_utc.tz_convert("UTC")
        pips_val = updates.get("pips")
        try:
            pips_float = float(pips_val)
        except Exception:
            pips_float = 0.0
        action = str(updates.get("exit_reason") or "")
        is_loss = True if action == "hard_sl" else (pips_float < 0.0)
        entry_session = infer_phase3_entry_session(entry_type, trade_row.get("entry_session"))
        if entry_session not in {"tokyo", "london", "ny"}:
            return
        key_date = phase3_trade_key_date(trade_row.get("timestamp_utc"), now_utc)
        phase3_policy_id = _phase3_policy_id_from_trade_row_api(trade_row)
        phase3_preset_id = None
        try:
            from core.phase3_package_spec import PHASE3_DEFENDED_PRESET_ID

            if str(phase3_policy_id or "").strip().lower() == str(PHASE3_DEFENDED_PRESET_ID).strip().lower():
                phase3_preset_id = PHASE3_DEFENDED_PRESET_ID
        except Exception:
            phase3_preset_id = None
        phase3_sizing_cfg = load_phase3_sizing_config(preset_id=phase3_preset_id) or {}
        sd = apply_phase3_session_outcome(
            phase3_state=phase3_state,
            phase3_sizing_cfg=phase3_sizing_cfg,
            entry_session=entry_session,
            entry_type=entry_type,
            is_loss=is_loss,
            action=action,
            side=str(trade_row.get("side") or "").lower(),
            key_date=key_date,
            now_utc=now_utc,
        )
        runtime_data["phase3_state"] = phase3_state
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(runtime_data, indent=2), encoding="utf-8")
        if entry_session == "ny":
            print(
                f"[api] phase3 sync state update: session=ny loss={1 if is_loss else 0} "
                f"consec={int(sd.get('consecutive_losses', 0))} trade_id={trade_row.get('trade_id')}"
            )
    except Exception as e:
        print(f"[api] phase3 sync state update failed for {trade_row.get('trade_id')}: {e}")


def _candidate_log_dirs(profile_name: str, profile_path: Optional[str] = None) -> list[Path]:
    """Return possible log dirs for this profile, handling profile_name mismatches."""
    dirs: list[Path] = []
    seen: set[str] = set()

    def _add(name: str) -> None:
        if not name:
            return
        key = str(name)
        if key in seen:
            return
        seen.add(key)
        dirs.append(LOGS_DIR / name)

    _add(profile_name)
    if profile_path:
        try:
            rp = _resolve_profile_path(profile_path)
            _add(rp.stem)
            if rp.exists():
                prof = load_profile_v1(rp)
                _add(getattr(prof, "profile_name", ""))
        except Exception:
            pass
    return dirs


def _pick_best_dashboard_log_dir(profile_name: str, profile_path: Optional[str] = None) -> Path:
    """Pick log dir with freshest dashboard_state.json among candidate dirs."""
    from datetime import datetime, timezone

    best_dir = LOGS_DIR / profile_name
    best_score = -1.0
    for d in _candidate_log_dirs(profile_name, profile_path):
        p = d / "dashboard_state.json"
        if not p.exists():
            continue
        score = 0.0
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            ts = data.get("timestamp_utc")
            if ts:
                dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                score = dt.timestamp()
            else:
                score = p.stat().st_mtime
        except Exception:
            try:
                score = p.stat().st_mtime
            except Exception:
                score = 0.0
        if score > best_score:
            best_score = score
            best_dir = d
    return best_dir


def _pick_best_trade_events_log_dir(profile_name: str, profile_path: Optional[str] = None) -> Path:
    """Pick log dir with newest trade_events.json among candidate dirs."""
    best_dir = LOGS_DIR / profile_name
    best_mtime = -1.0
    for d in _candidate_log_dirs(profile_name, profile_path):
        p = d / "trade_events.json"
        if not p.exists():
            continue
        try:
            mt = p.stat().st_mtime
        except Exception:
            mt = 0.0
        if mt > best_mtime:
            best_mtime = mt
            best_dir = d
    return best_dir


def _strip_wrapped_quotes(value: str | None) -> str:
    text = str(value or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    return text


def _parse_phase3_diag_filters(raw_filters: str | None) -> list[str]:
    text = str(raw_filters or "").strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if not text:
        return []
    return [part.strip() for part in text.split(";") if part.strip()]


def _tail_text_lines(path: Path, max_lines: int) -> list[str]:
    if max_lines <= 0 or not path.exists():
        return []
    chunk_size = 8192
    remaining = max_lines
    data = bytearray()
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        pos = handle.tell()
        while pos > 0 and data.count(b"\n") <= max_lines:
            read_size = min(chunk_size, pos)
            pos -= read_size
            handle.seek(pos)
            data[:0] = handle.read(read_size)
    text = data.decode("utf-8", errors="ignore")
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return lines
    return lines[-max_lines:]


def _load_phase3_decision_rows_from_diagnostics(
    log_dir: Path | None,
    *,
    days: int,
    limit: int,
) -> list[dict[str, Any]]:
    if log_dir is None:
        return []
    diag_path = log_dir / "phase3_minute_diagnostics.log"
    if not diag_path.exists():
        return []

    from datetime import timedelta

    since_dt = datetime.now(timezone.utc) - timedelta(days=days)
    try:
        recent_lines = [
            line for line in _tail_text_lines(diag_path, max(limit * 4, 500))
            if "bar=" in line and "reason=" in line
        ]
    except Exception:
        return []

    rows: list[dict[str, Any]] = []
    for raw_line in recent_lines:
        parts = raw_line.split("\t")
        if not parts:
            continue
        ts_raw = str(parts[0] or "").strip()
        try:
            ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts < since_dt:
                continue
        except Exception:
            continue

        fields: dict[str, str] = {}
        for part in parts[1:]:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            fields[key.strip()] = value.strip()

        bar_time = str(fields.get("bar") or "").strip()
        session = str(fields.get("session") or "").strip() or None
        reason_text = _strip_wrapped_quotes(fields.get("reason"))
        placed = str(fields.get("placed") or "0").strip() == "1"
        blocking_ids = _parse_phase3_diag_filters(fields.get("filters"))
        rows.append(
            {
                "timestamp_utc": ts.isoformat(),
                "signal_id": f"eval:phase3_integrated:diag:{bar_time}" if bar_time else None,
                "mode": None,
                "attempted": 1 if placed or reason_text else 0,
                "placed": 1 if placed else 0,
                "reason": reason_text,
                "reason_text": reason_text,
                "blocking_filter_ids": blocking_ids,
                "phase3_session": session,
            }
        )
    return rows


def _get_display_currency(profile: ProfileV1) -> tuple[str, float]:
    """Return (currency_code, rate). Rate multiplies USD amounts to get display value.
    For USD, rate=1. For JPY, rate=usdjpy bid (1 USD = rate JPY)."""
    curr = getattr(profile, "display_currency", None) or "USD"
    if curr != "JPY":
        return "USD", 1.0
    try:
        from adapters.broker import get_adapter
        adapter = get_adapter(profile)
        adapter.initialize()
        tick = adapter.get_tick(profile.symbol)
        adapter.shutdown()
        if tick and tick.bid > 0:
            return "JPY", float(tick.bid)
    except Exception:
        pass
    return "USD", 1.0


def _convert_amount(amount: float | None, rate: float) -> float | None:
    if amount is None:
        return None
    return round(amount * rate, 2)


def _row_get(row: Any, key: str) -> Any:
    if isinstance(row, dict):
        return row.get(key)
    try:
        return row.get(key)  # pandas Series / dict-like
    except Exception:
        pass
    try:
        return row[key]
    except Exception:
        return None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_symbol_code(symbol: str | None) -> str:
    s = re.sub(r"[^A-Za-z]", "", str(symbol or "").upper())
    return s[:6] if len(s) >= 6 else s


def _estimate_profit_usd_from_row(row: Any, symbol_hint: str | None = None) -> float | None:
    """Best-effort USD P/L estimate from entry/exit/size/side."""
    entry = _float_or_none(_row_get(row, "entry_price"))
    exit_ = _float_or_none(_row_get(row, "exit_price"))
    lots = _float_or_none(_row_get(row, "size_lots"))
    if lots is None:
        lots = _float_or_none(_row_get(row, "volume"))
    side = str(_row_get(row, "side") or "").lower()
    if entry is None or exit_ is None or lots is None or lots == 0 or side not in ("buy", "sell"):
        return None

    diff = (exit_ - entry) if side == "buy" else (entry - exit_)
    units = lots * 100_000.0
    sym = _normalize_symbol_code(str(_row_get(row, "symbol") or symbol_hint or ""))
    if not sym:
        return None

    if sym.endswith("JPY"):
        if exit_ <= 0:
            return None
        return round(diff * units / exit_, 2)
    if len(sym) == 6 and sym[3:] == "USD":
        return round(diff * units, 2)
    if len(sym) == 6 and sym[:3] == "USD":
        if exit_ <= 0:
            return None
        return round(diff * units / exit_, 2)
    return None


def _normalized_trade_profit_usd(row: Any, symbol_hint: str | None = None) -> float | None:
    """Prefer stored profit, but repair obvious outliers using price-based estimate."""
    raw = _float_or_none(_row_get(row, "profit"))
    est = _estimate_profit_usd_from_row(row, symbol_hint=symbol_hint)
    if raw is None:
        return est
    if est is None:
        return raw
    if abs(raw) <= 0.01 and abs(est) >= 0.5:
        return est

    # Only override raw with est when they AGREE in sign.  The raw value
    # comes from the broker's realizedPL (authoritative) while est is a
    # rough price-based approximation that can be wrong for partially-closed
    # trades (where exit_price is a weighted average or last-fill price).
    abs_raw = abs(raw)
    abs_est = abs(est)
    if (raw * est) >= 0:
        # Same sign — trust raw unless the magnitude ratio is extreme (>6×),
        # which signals a unit/currency mismatch in stored data.
        if abs_raw >= 250 and abs_est >= 5:
            ratio = abs_raw / abs_est if abs_est > 0 else float("inf")
            if ratio >= 6.0:
                return est
    return raw


def _normalized_trade_pips(row: Any, pip_size: float) -> float | None:
    """Prefer stored pips, but repair missing/zero values from entry/exit prices."""
    raw = _float_or_none(_row_get(row, "pips"))
    entry = _float_or_none(_row_get(row, "entry_price"))
    exit_ = _float_or_none(_row_get(row, "exit_price"))
    side = str(_row_get(row, "side") or "").lower()

    est = None
    if entry is not None and exit_ is not None and pip_size > 0 and side in ("buy", "sell"):
        diff = (exit_ - entry) if side == "buy" else (entry - exit_)
        est = round(diff / pip_size, 1)

    if raw is None:
        return est
    if est is None:
        return raw
    if abs(raw) <= 0.05 and abs(est) >= 0.2:
        return est
    return raw


def _pid_looks_like_run_loop(pid: int) -> bool:
    """Best-effort check that PID is our run_loop process."""
    # Preferred path: psutil if installed.
    try:
        import psutil  # type: ignore

        proc = psutil.Process(int(pid))
        cmdline = " ".join(proc.cmdline()).lower()
        if "run_loop.py" in cmdline or "run_loop" in cmdline:
            return True
    except Exception:
        pass

    # Fallback path: inspect /proc on Linux-like systems.
    try:
        cmdline_path = Path(f"/proc/{int(pid)}/cmdline")
        if cmdline_path.exists():
            raw = cmdline_path.read_bytes().decode("utf-8", errors="ignore").replace("\x00", " ").lower()
            if "run_loop.py" in raw or "run_loop" in raw:
                return True
    except Exception:
        pass

    return False


def _is_loop_running(profile_name: str) -> bool:
    proc = _loop_processes.get(profile_name)
    if proc is not None:
        if proc.poll() is None:
            return True
        # Process finished; remove from tracking
        del _loop_processes[profile_name]

    pid_path = _loop_pid_path(profile_name)
    if not pid_path.exists():
        return False
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
        # Verify it's actually our run_loop process (guard recycled PIDs).
        if not _pid_looks_like_run_loop(pid):
            raise ProcessLookupError("stale PID")
        return True
    except Exception:
        try:
            pid_path.unlink(missing_ok=True)
        except Exception:
            pass
        return False


# ---------------------------------------------------------------------------
# Pydantic models for API
# ---------------------------------------------------------------------------


class ProfileInfo(BaseModel):
    path: str
    name: str


class RuntimeStateUpdate(BaseModel):
    mode: str
    kill_switch: bool
    exit_system_only: bool = False


class TempEmaSettingsUpdate(BaseModel):
    m5_trend_ema_fast: Optional[int] = None
    m5_trend_ema_slow: Optional[int] = None
    m5_trend_source: Optional[str] = None
    m1_zone_entry_ema_slow: Optional[int] = None
    m1_pullback_cross_ema_slow: Optional[int] = None
    # Trial #4 fields (Zone Entry only - Tiered Pullback uses fixed tiers)
    m3_trend_ema_fast: Optional[int] = None
    m3_trend_ema_slow: Optional[int] = None
    m1_t4_zone_entry_ema_fast: Optional[int] = None
    m1_t4_zone_entry_ema_slow: Optional[int] = None
    # Uncle Parsh H1 Breakout fields
    up_m5_ema_fast: Optional[int] = None
    up_m5_ema_slow: Optional[int] = None
    # Uncle Parsh H1 Breakout: H1 Detection
    up_major_extremes_only: Optional[bool] = None
    up_h1_lookback_hours: Optional[int] = None
    up_h1_swing_strength: Optional[int] = None
    up_h1_cluster_tolerance_pips: Optional[float] = None
    up_h1_min_touches_for_major: Optional[int] = None
    # Uncle Parsh H1 Breakout: M5 Catalyst
    up_power_close_body_pct: Optional[float] = None
    up_velocity_pips: Optional[float] = None
    # Uncle Parsh H1 Breakout: Exit Strategy
    up_initial_sl_spread_plus_pips: Optional[float] = None
    up_tp1_pips: Optional[float] = None
    up_tp1_close_pct: Optional[float] = None
    up_be_spread_plus_pips: Optional[float] = None
    up_trail_ema_period: Optional[int] = None
    # Uncle Parsh H1 Breakout: Discipline
    up_max_spread_pips: Optional[float] = None

    # Trial #8 exit strategy
    t8_exit_strategy: Optional[str] = None
    t8_tp1_pips: Optional[float] = None
    t8_tp1_close_pct: Optional[float] = None
    t8_be_spread_plus_pips: Optional[float] = None
    t8_trail_ema_period: Optional[int] = None
    t8_m1_exit_ema_fast: Optional[int] = None
    t8_m1_exit_ema_slow: Optional[int] = None
    t8_scale_out_pct: Optional[float] = None
    t8_initial_sl_spread_plus_pips: Optional[float] = None
    # Trial #9 exit strategy
    t9_exit_strategy: Optional[str] = None
    t9_hwm_trail_pips: Optional[float] = None
    t9_tp1_pips: Optional[float] = None
    t9_tp1_close_pct: Optional[float] = None
    t9_be_spread_plus_pips: Optional[float] = None
    t9_trail_ema_period: Optional[int] = None
    t9_trail_m5_ema_period: Optional[int] = None
    # Trial #10 regime / execution fields
    t10_regime_gate_enabled: Optional[bool] = None
    t10_regime_london_sell_veto: Optional[bool] = None
    t10_regime_london_start_hour_et: Optional[int] = None
    t10_regime_london_end_hour_et: Optional[int] = None
    t10_regime_boost_multiplier: Optional[float] = None
    t10_regime_buy_base_multiplier: Optional[float] = None
    t10_regime_sell_base_multiplier: Optional[float] = None
    t10_regime_chop_pause_enabled: Optional[bool] = None
    t10_regime_chop_pause_minutes: Optional[int] = None
    t10_regime_chop_pause_lookback_trades: Optional[int] = None
    t10_regime_chop_pause_stop_rate: Optional[float] = None
    t10_tier17_nonboost_multiplier: Optional[float] = None
    t10_bucketed_exit_enabled: Optional[bool] = None
    t10_quick_tp1_pips: Optional[float] = None
    t10_quick_tp1_close_pct: Optional[float] = None
    t10_quick_be_spread_plus_pips: Optional[float] = None
    t10_runner_tp1_pips: Optional[float] = None
    t10_runner_tp1_close_pct: Optional[float] = None
    t10_runner_be_spread_plus_pips: Optional[float] = None
    t10_trail_escalation_enabled: Optional[bool] = None
    t10_trail_escalation_tier1_pips: Optional[float] = None
    t10_trail_escalation_tier2_pips: Optional[float] = None
    t10_trail_escalation_m15_ema_period: Optional[int] = None
    t10_trail_escalation_m15_buffer_pips: Optional[float] = None
    t10_runner_score_sizing_enabled: Optional[bool] = None
    t10_runner_base_lots: Optional[float] = None
    t10_runner_min_lots: Optional[float] = None
    t10_runner_max_lots: Optional[float] = None
    t10_atr_stop_enabled: Optional[bool] = None
    t10_atr_stop_multiplier: Optional[float] = None
    t10_atr_stop_max_pips: Optional[float] = None



class ApplyPresetRequest(BaseModel):
    preset_id: str
    options: Optional[dict[str, Any]] = None  # e.g. vwap_session_filter_enabled for vwap_trend


class ProfileUpdateRequest(BaseModel):
    profile_data: dict[str, Any]


class CreateProfileRequest(BaseModel):
    name: str


class AuthLoginRequest(BaseModel):
    profile_path: str
    password: str


class AuthSetPasswordRequest(BaseModel):
    profile_path: str
    current_password: str | None = None
    new_password: str


class AuthRemovePasswordRequest(BaseModel):
    profile_path: str
    password: str


def _sanitize_profile_name(name: str) -> str:
    """Lowercase, replace spaces with underscores, restrict to alphanumeric and underscore."""
    s = name.strip().lower().replace(" ", "_")
    return "".join(c for c in s if c.isalnum() or c == "_") or "default"


def _resolve_profile_path(profile_path: str) -> Path:
    """Resolve profile_path to absolute Path; relative paths are under PROFILES_DIR."""
    p = Path(profile_path)
    return p.resolve() if p.is_absolute() else (PROFILES_DIR / p).resolve()


def _profile_path_safe(path: Path) -> bool:
    """True if path is under PROFILES_DIR (resolve both) and not the dir itself."""
    try:
        r = path.resolve()
        base = PROFILES_DIR.resolve()
        r.relative_to(base)
        return r != base and r.is_file()
    except (ValueError, OSError):
        return False


# ---------------------------------------------------------------------------
# Endpoints: Profiles
# ---------------------------------------------------------------------------


@app.get("/api/profiles")
def list_profiles() -> list[ProfileInfo]:
    """List all profile JSON files."""
    paths = _dedupe_profile_paths_by_stem(_list_profile_paths())
    return [
        ProfileInfo(path=str(p.resolve()), name=p.stem)
        for p in paths
    ]


@app.get("/api/profiles/{profile_path:path}")
def get_profile(profile_path: str) -> dict[str, Any]:
    """Load and return a profile as JSON."""
    path = _resolve_profile_path(profile_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Profile not found")
    try:
        profile = load_profile_v1(path)
        return profile.model_dump()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.put("/api/profiles/{profile_path:path}")
def save_profile(profile_path: str, req: ProfileUpdateRequest) -> dict[str, str]:
    """Save updated profile data to disk."""
    from pydantic import ValidationError
    path = _resolve_profile_path(profile_path)
    try:
        profile = ProfileV1.model_validate(req.profile_data)
        save_profile_v1(profile, path)
        return {"status": "saved", "path": str(path)}
    except ValidationError as e:
        if "extra_forbidden" in str(e) or "Extra inputs" in str(e):
            try:
                profile = ProfileV1AllowExtra.model_validate(req.profile_data)
                save_profile_v1(profile, path)
                return {"status": "saved", "path": str(path)}
            except Exception as fallback_e:
                raise HTTPException(status_code=400, detail=str(fallback_e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/profiles")
def create_profile(req: CreateProfileRequest) -> ProfileInfo:
    """Create a new account/profile with default settings. Name is sanitized to a filename."""
    name = _sanitize_profile_name(req.name)
    if not name:
        raise HTTPException(status_code=400, detail="Invalid profile name")
    path = PROFILES_DIR / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise HTTPException(status_code=409, detail=f"Profile already exists: {name}")
    profile = default_profile_for_name(name)
    save_profile_v1(profile, path)
    return ProfileInfo(path=str(path.resolve()), name=path.stem)


@app.delete("/api/profiles")
def delete_profile(path: str) -> dict[str, str]:
    """Delete an account/profile. Path must be under profiles dir. Logs and DB are not removed."""
    p = Path(path)
    if not _profile_path_safe(p):
        raise HTTPException(status_code=400, detail="Invalid or forbidden profile path")
    try:
        p.unlink()
        return {"status": "deleted", "path": str(p)}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Profile not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Endpoints: Authentication
# ---------------------------------------------------------------------------


@app.get("/api/auth/check")
def check_auth(profile_path: str) -> dict[str, bool]:
    """Check if a profile has a password set."""
    from core.auth import has_password
    
    profile_name = Path(profile_path).stem
    return {"has_password": has_password(profile_name)}


@app.post("/api/auth/login")
def auth_login(req: AuthLoginRequest) -> dict[str, bool]:
    """Verify password for a profile. Returns success: true if password is correct."""
    from core.auth import has_password, verify_password
    
    profile_name = Path(req.profile_path).stem
    
    # If no password set, login is always successful
    if not has_password(profile_name):
        return {"success": True}
    
    return {"success": verify_password(profile_name, req.password)}


@app.post("/api/auth/set-password")
def auth_set_password(req: AuthSetPasswordRequest) -> dict[str, Any]:
    """Set or update password for a profile.
    
    If profile already has a password, current_password must be provided.
    """
    from core.auth import set_password
    
    profile_name = Path(req.profile_path).stem
    success, error = set_password(profile_name, req.new_password, req.current_password)
    
    if success:
        return {"success": True}
    raise HTTPException(status_code=400, detail=error)


@app.post("/api/auth/remove-password")
def auth_remove_password(req: AuthRemovePasswordRequest) -> dict[str, Any]:
    """Remove password from a profile. Requires current password to verify."""
    from core.auth import remove_password
    
    profile_name = Path(req.profile_path).stem
    success, error = remove_password(profile_name, req.password)
    
    if success:
        return {"success": True}
    raise HTTPException(status_code=400, detail=error)


# ---------------------------------------------------------------------------
# Endpoints: Presets
# ---------------------------------------------------------------------------


@app.get("/api/presets")
def get_presets() -> list[dict[str, Any]]:
    """List available presets with id, name, description."""
    return list_presets()


@app.get("/api/presets/{preset_id}/preview")
def preview_preset(preset_id: str, profile_path: str) -> dict[str, Any]:
    """Preview what fields a preset would change on a profile."""
    path = Path(profile_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Profile not found")
    try:
        preset = PresetId(preset_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown preset: {preset_id}")
    patch = get_preset_patch(preset)
    return {"preset_id": preset_id, "changes": patch}


@app.post("/api/presets/{preset_id}/apply")
def apply_preset_to_profile(preset_id: str, req: ApplyPresetRequest, profile_path: str) -> dict[str, Any]:
    """Apply a preset to a profile and save it."""
    path = _resolve_profile_path(profile_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Profile not found")
    try:
        preset = PresetId(preset_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown preset: {preset_id}")
    try:
        profile = load_profile_v1(path)
        patch_overrides = None
        if preset_id == "vwap_trend" and isinstance(req.options, dict) and "vwap_session_filter_enabled" in req.options:
            patch_overrides = {
                "execution": {
                    "policies_overlay_for_vwap": {"session_filter_enabled": bool(req.options["vwap_session_filter_enabled"])},
                },
            }
        new_profile = apply_preset(profile, preset, patch_overrides)
        save_profile_v1(new_profile, path)
        return {"status": "applied", "preset_id": preset_id, "profile": new_profile.model_dump()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# Endpoints: Runtime state
# ---------------------------------------------------------------------------


@app.get("/api/runtime/{profile_name}")
def get_runtime_state(profile_name: str) -> dict[str, Any]:
    """Get runtime state (mode, kill_switch) for a profile."""
    state_path = _runtime_state_path(profile_name)
    state = load_state(state_path)
    return {
        "mode": state.mode,
        "kill_switch": state.kill_switch,
        "exit_system_only": state.exit_system_only,
        "last_processed_bar_time_utc": state.last_processed_bar_time_utc,
        "loop_running": _is_loop_running(profile_name),
        "daily_reset_date": state.daily_reset_date,
        "daily_reset_high": state.daily_reset_high,
        "daily_reset_low": state.daily_reset_low,
        "daily_reset_block_active": state.daily_reset_block_active,
        "daily_reset_settled": state.daily_reset_settled,
        "trend_flip_price": state.trend_flip_price,
        "trend_flip_direction": state.trend_flip_direction,
    }


@app.put("/api/runtime/{profile_name}")
def update_runtime_state(profile_name: str, req: RuntimeStateUpdate) -> dict[str, str]:
    """Update runtime state for a profile."""
    state_path = _runtime_state_path(profile_name)
    old = load_state(state_path)
    new_data = dict(old.__dict__)
    new_data["mode"] = req.mode
    new_data["kill_switch"] = req.kill_switch
    new_data["exit_system_only"] = req.exit_system_only
    new_state = RuntimeState(**new_data)  # type: ignore[arg-type]
    save_state(state_path, new_state)
    return {"status": "saved"}


@app.get("/api/runtime/{profile_name}/temp-settings")
def get_temp_settings(profile_name: str) -> dict[str, Any]:
    """Get temporary EMA settings for Apply Temporary Settings menu."""
    state_path = _runtime_state_path(profile_name)
    state = load_state(state_path)
    return {
        "m5_trend_ema_fast": state.temp_m5_trend_ema_fast,
        "m5_trend_ema_slow": state.temp_m5_trend_ema_slow,
        "m5_trend_source": state.temp_m5_trend_source,
        "m1_zone_entry_ema_slow": state.temp_m1_zone_entry_ema_slow,
        "m1_pullback_cross_ema_slow": state.temp_m1_pullback_cross_ema_slow,
        "m3_trend_ema_fast": state.temp_m3_trend_ema_fast,
        "m3_trend_ema_slow": state.temp_m3_trend_ema_slow,
        "m1_t4_zone_entry_ema_fast": state.temp_m1_t4_zone_entry_ema_fast,
        "m1_t4_zone_entry_ema_slow": state.temp_m1_t4_zone_entry_ema_slow,
        "up_m5_ema_fast": state.temp_up_m5_ema_fast,
        "up_m5_ema_slow": state.temp_up_m5_ema_slow,
        "up_major_extremes_only": state.temp_up_major_extremes_only,
        "up_h1_lookback_hours": state.temp_up_h1_lookback_hours,
        "up_h1_swing_strength": state.temp_up_h1_swing_strength,
        "up_h1_cluster_tolerance_pips": state.temp_up_h1_cluster_tolerance_pips,
        "up_h1_min_touches_for_major": state.temp_up_h1_min_touches_for_major,
        "up_h1_min_distance_between_levels_pips": getattr(state, "temp_up_h1_min_distance_between_levels_pips", None),
        "up_power_close_body_pct": state.temp_up_power_close_body_pct,
        "up_velocity_pips": state.temp_up_velocity_pips,
        "up_initial_sl_spread_plus_pips": state.temp_up_initial_sl_spread_plus_pips,
        "up_tp1_pips": state.temp_up_tp1_pips,
        "up_tp1_close_pct": state.temp_up_tp1_close_pct,
        "up_be_spread_plus_pips": state.temp_up_be_spread_plus_pips,
        "up_trail_ema_period": state.temp_up_trail_ema_period,
        "up_max_spread_pips": state.temp_up_max_spread_pips,
        "t8_exit_strategy": state.temp_t8_exit_strategy,
        "t8_tp1_pips": state.temp_t8_tp1_pips,
        "t8_tp1_close_pct": state.temp_t8_tp1_close_pct,
        "t8_be_spread_plus_pips": state.temp_t8_be_spread_plus_pips,
        "t8_trail_ema_period": state.temp_t8_trail_ema_period,
        "t8_m1_exit_ema_fast": state.temp_t8_m1_exit_ema_fast,
        "t8_m1_exit_ema_slow": state.temp_t8_m1_exit_ema_slow,
        "t8_scale_out_pct": state.temp_t8_scale_out_pct,
        "t8_initial_sl_spread_plus_pips": state.temp_t8_initial_sl_spread_plus_pips,
        "t9_exit_strategy": state.temp_t9_exit_strategy,
        "t9_hwm_trail_pips": state.temp_t9_hwm_trail_pips,
        "t9_tp1_pips": state.temp_t9_tp1_pips,
        "t9_tp1_close_pct": state.temp_t9_tp1_close_pct,
        "t9_be_spread_plus_pips": state.temp_t9_be_spread_plus_pips,
        "t9_trail_ema_period": state.temp_t9_trail_ema_period,
        "t9_trail_m5_ema_period": state.temp_t9_trail_m5_ema_period,
        "t10_regime_gate_enabled": state.temp_t10_regime_gate_enabled,
        "t10_regime_london_sell_veto": state.temp_t10_regime_london_sell_veto,
        "t10_regime_london_start_hour_et": state.temp_t10_regime_london_start_hour_et,
        "t10_regime_london_end_hour_et": state.temp_t10_regime_london_end_hour_et,
        "t10_regime_boost_multiplier": state.temp_t10_regime_boost_multiplier,
        "t10_regime_buy_base_multiplier": state.temp_t10_regime_buy_base_multiplier,
        "t10_regime_sell_base_multiplier": state.temp_t10_regime_sell_base_multiplier,
        "t10_regime_chop_pause_enabled": state.temp_t10_regime_chop_pause_enabled,
        "t10_regime_chop_pause_minutes": state.temp_t10_regime_chop_pause_minutes,
        "t10_regime_chop_pause_lookback_trades": state.temp_t10_regime_chop_pause_lookback_trades,
        "t10_regime_chop_pause_stop_rate": state.temp_t10_regime_chop_pause_stop_rate,
        "t10_tier17_nonboost_multiplier": state.temp_t10_tier17_nonboost_multiplier,
        "t10_bucketed_exit_enabled": state.temp_t10_bucketed_exit_enabled,
        "t10_quick_tp1_pips": state.temp_t10_quick_tp1_pips,
        "t10_quick_tp1_close_pct": state.temp_t10_quick_tp1_close_pct,
        "t10_quick_be_spread_plus_pips": state.temp_t10_quick_be_spread_plus_pips,
        "t10_runner_tp1_pips": state.temp_t10_runner_tp1_pips,
        "t10_runner_tp1_close_pct": state.temp_t10_runner_tp1_close_pct,
        "t10_runner_be_spread_plus_pips": state.temp_t10_runner_be_spread_plus_pips,
        "t10_trail_escalation_enabled": state.temp_t10_trail_escalation_enabled,
        "t10_trail_escalation_tier1_pips": state.temp_t10_trail_escalation_tier1_pips,
        "t10_trail_escalation_tier2_pips": state.temp_t10_trail_escalation_tier2_pips,
        "t10_trail_escalation_m15_ema_period": state.temp_t10_trail_escalation_m15_ema_period,
        "t10_trail_escalation_m15_buffer_pips": state.temp_t10_trail_escalation_m15_buffer_pips,
        "t10_runner_score_sizing_enabled": state.temp_t10_runner_score_sizing_enabled,
        "t10_runner_base_lots": state.temp_t10_runner_base_lots,
        "t10_runner_min_lots": state.temp_t10_runner_min_lots,
        "t10_runner_max_lots": state.temp_t10_runner_max_lots,
        "t10_atr_stop_enabled": state.temp_t10_atr_stop_enabled,
        "t10_atr_stop_multiplier": state.temp_t10_atr_stop_multiplier,
        "t10_atr_stop_max_pips": state.temp_t10_atr_stop_max_pips,
    }


@app.put("/api/runtime/{profile_name}/temp-settings")
def update_temp_settings(profile_name: str, req: TempEmaSettingsUpdate) -> dict[str, str]:
    """Update temporary EMA settings for Apply Temporary Settings menu."""
    state_path = _runtime_state_path(profile_name)
    old = load_state(state_path)
    new_data = dict(old.__dict__)
    new_data.update(
        {
            "temp_m5_trend_ema_fast": req.m5_trend_ema_fast,
            "temp_m5_trend_ema_slow": req.m5_trend_ema_slow,
            "temp_m5_trend_source": req.m5_trend_source,
            "temp_m1_zone_entry_ema_slow": req.m1_zone_entry_ema_slow,
            "temp_m1_pullback_cross_ema_slow": req.m1_pullback_cross_ema_slow,
            "temp_m3_trend_ema_fast": req.m3_trend_ema_fast,
            "temp_m3_trend_ema_slow": req.m3_trend_ema_slow,
            "temp_m1_t4_zone_entry_ema_fast": req.m1_t4_zone_entry_ema_fast,
            "temp_m1_t4_zone_entry_ema_slow": req.m1_t4_zone_entry_ema_slow,
            "temp_up_m5_ema_fast": req.up_m5_ema_fast,
            "temp_up_m5_ema_slow": req.up_m5_ema_slow,
            "temp_up_major_extremes_only": req.up_major_extremes_only,
            "temp_up_h1_lookback_hours": req.up_h1_lookback_hours,
            "temp_up_h1_swing_strength": req.up_h1_swing_strength,
            "temp_up_h1_cluster_tolerance_pips": req.up_h1_cluster_tolerance_pips,
            "temp_up_h1_min_touches_for_major": req.up_h1_min_touches_for_major,
            "temp_up_power_close_body_pct": req.up_power_close_body_pct,
            "temp_up_velocity_pips": req.up_velocity_pips,
            "temp_up_initial_sl_spread_plus_pips": req.up_initial_sl_spread_plus_pips,
            "temp_up_tp1_pips": req.up_tp1_pips,
            "temp_up_tp1_close_pct": req.up_tp1_close_pct,
            "temp_up_be_spread_plus_pips": req.up_be_spread_plus_pips,
            "temp_up_trail_ema_period": req.up_trail_ema_period,
            "temp_up_max_spread_pips": req.up_max_spread_pips,
            "temp_t8_exit_strategy": req.t8_exit_strategy,
            "temp_t8_tp1_pips": req.t8_tp1_pips,
            "temp_t8_tp1_close_pct": req.t8_tp1_close_pct,
            "temp_t8_be_spread_plus_pips": req.t8_be_spread_plus_pips,
            "temp_t8_trail_ema_period": req.t8_trail_ema_period,
            "temp_t8_m1_exit_ema_fast": req.t8_m1_exit_ema_fast,
            "temp_t8_m1_exit_ema_slow": req.t8_m1_exit_ema_slow,
            "temp_t8_scale_out_pct": req.t8_scale_out_pct,
            "temp_t8_initial_sl_spread_plus_pips": req.t8_initial_sl_spread_plus_pips,
            "temp_t9_exit_strategy": req.t9_exit_strategy,
            "temp_t9_hwm_trail_pips": req.t9_hwm_trail_pips,
            "temp_t9_tp1_pips": req.t9_tp1_pips,
            "temp_t9_tp1_close_pct": req.t9_tp1_close_pct,
            "temp_t9_be_spread_plus_pips": req.t9_be_spread_plus_pips,
            "temp_t9_trail_ema_period": req.t9_trail_ema_period,
            "temp_t9_trail_m5_ema_period": req.t9_trail_m5_ema_period,
            "temp_t10_regime_gate_enabled": req.t10_regime_gate_enabled,
            "temp_t10_regime_london_sell_veto": req.t10_regime_london_sell_veto,
            "temp_t10_regime_london_start_hour_et": req.t10_regime_london_start_hour_et,
            "temp_t10_regime_london_end_hour_et": req.t10_regime_london_end_hour_et,
            "temp_t10_regime_boost_multiplier": req.t10_regime_boost_multiplier,
            "temp_t10_regime_buy_base_multiplier": req.t10_regime_buy_base_multiplier,
            "temp_t10_regime_sell_base_multiplier": req.t10_regime_sell_base_multiplier,
            "temp_t10_regime_chop_pause_enabled": req.t10_regime_chop_pause_enabled,
            "temp_t10_regime_chop_pause_minutes": req.t10_regime_chop_pause_minutes,
            "temp_t10_regime_chop_pause_lookback_trades": req.t10_regime_chop_pause_lookback_trades,
            "temp_t10_regime_chop_pause_stop_rate": req.t10_regime_chop_pause_stop_rate,
            "temp_t10_tier17_nonboost_multiplier": req.t10_tier17_nonboost_multiplier,
            "temp_t10_bucketed_exit_enabled": req.t10_bucketed_exit_enabled,
            "temp_t10_quick_tp1_pips": req.t10_quick_tp1_pips,
            "temp_t10_quick_tp1_close_pct": req.t10_quick_tp1_close_pct,
            "temp_t10_quick_be_spread_plus_pips": req.t10_quick_be_spread_plus_pips,
            "temp_t10_runner_tp1_pips": req.t10_runner_tp1_pips,
            "temp_t10_runner_tp1_close_pct": req.t10_runner_tp1_close_pct,
            "temp_t10_runner_be_spread_plus_pips": req.t10_runner_be_spread_plus_pips,
            "temp_t10_trail_escalation_enabled": req.t10_trail_escalation_enabled,
            "temp_t10_trail_escalation_tier1_pips": req.t10_trail_escalation_tier1_pips,
            "temp_t10_trail_escalation_tier2_pips": req.t10_trail_escalation_tier2_pips,
            "temp_t10_trail_escalation_m15_ema_period": req.t10_trail_escalation_m15_ema_period,
            "temp_t10_trail_escalation_m15_buffer_pips": req.t10_trail_escalation_m15_buffer_pips,
            "temp_t10_runner_score_sizing_enabled": req.t10_runner_score_sizing_enabled,
            "temp_t10_runner_base_lots": req.t10_runner_base_lots,
            "temp_t10_runner_min_lots": req.t10_runner_min_lots,
            "temp_t10_runner_max_lots": req.t10_runner_max_lots,
            "temp_t10_atr_stop_enabled": req.t10_atr_stop_enabled,
            "temp_t10_atr_stop_multiplier": req.t10_atr_stop_multiplier,
            "temp_t10_atr_stop_max_pips": req.t10_atr_stop_max_pips,
        }
    )
    new_state = RuntimeState(**new_data)  # type: ignore[arg-type]
    save_state(state_path, new_state)
    return {"status": "saved"}


# ---------------------------------------------------------------------------
# Endpoints: Loop control
# ---------------------------------------------------------------------------


@app.post("/api/loop/{profile_name}/start")
def start_loop(profile_name: str, profile_path: str) -> dict[str, Any]:
    """Start the trading loop for a profile."""
    if _is_loop_running(profile_name):
        return {"status": "already_running"}
    
    path = _resolve_profile_path(profile_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Profile not found")
    
    log_dir = LOGS_DIR / profile_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "loop.log"
    
    log_file = None
    try:
        log_file = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, "-u", str(BASE_DIR / "run_loop.py"), "--profile", str(path)],
            cwd=str(BASE_DIR),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        log_file.close()
        log_file = None
        _loop_processes[profile_name] = proc
        pid_path = _loop_pid_path(profile_name)
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        pid_path.write_text(str(proc.pid), encoding="utf-8")
        _time.sleep(0.75)
        if proc.poll() is not None:
            _loop_processes.pop(profile_name, None)
            pid_path.unlink(missing_ok=True)
            detail = f"Loop exited immediately with code {proc.returncode}."
            try:
                if log_path.exists():
                    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                    tail = "\n".join(lines[-12:]).strip()
                    if tail:
                        detail = f"{detail}\n{tail}"
            except Exception:
                pass
            raise HTTPException(status_code=500, detail=detail)
        return {"status": "started", "pid": proc.pid}
    except Exception as e:
        if log_file is not None:
            try:
                log_file.close()
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/loop/{profile_name}/stop")
def stop_loop(profile_name: str) -> dict[str, str]:
    """Stop the trading loop for a profile."""
    proc = _loop_processes.get(profile_name)
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        _loop_processes.pop(profile_name, None)
        try:
            _loop_pid_path(profile_name).unlink(missing_ok=True)
        except Exception:
            pass
        return {"status": "stopped"}

    # Fallback: stop by persisted pid (covers API restarts)
    pid_path = _loop_pid_path(profile_name)
    if not pid_path.exists():
        return {"status": "not_running"}
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except Exception:
        pid_path.unlink(missing_ok=True)
        return {"status": "not_running"}

    # Never signal an unrelated process if PID got recycled.
    try:
        if not _pid_looks_like_run_loop(pid):
            pid_path.unlink(missing_ok=True)
            return {"status": "not_running"}
    except Exception:
        pid_path.unlink(missing_ok=True)
        return {"status": "not_running"}

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pid_path.unlink(missing_ok=True)
        return {"status": "not_running"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop loop pid {pid}: {e}")

    deadline = _time.monotonic() + 10.0
    while _time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
            _time.sleep(0.1)
        except ProcessLookupError:
            pid_path.unlink(missing_ok=True)
            _loop_processes.pop(profile_name, None)
            return {"status": "stopped"}
        except Exception:
            break

    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to force-stop loop pid {pid}: {e}")

    pid_path.unlink(missing_ok=True)
    _loop_processes.pop(profile_name, None)
    return {"status": "stopped"}


@app.get("/api/loop/{profile_name}/log")
def get_loop_log(profile_name: str, lines: int = 200) -> dict[str, Any]:
    """Get the tail of the loop log."""
    log_path = LOGS_DIR / profile_name / "loop.log"
    if not log_path.exists():
        return {"exists": False, "content": ""}
    
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
        tail = all_lines[-lines:] if len(all_lines) > lines else all_lines
        return {"exists": True, "content": "".join(tail), "total_lines": len(all_lines)}
    except Exception as e:
        return {"exists": True, "content": f"Error reading log: {e}", "total_lines": 0}


# ---------------------------------------------------------------------------
# Endpoints: Database (snapshots, trades, executions)
# ---------------------------------------------------------------------------


@app.get("/api/data/{profile_name}/snapshots")
def get_snapshots(profile_name: str, limit: int = 20) -> list[dict[str, Any]]:
    """Get recent snapshots."""
    store = _store_for(profile_name)
    df = store.read_snapshots_df(profile_name).tail(limit)
    return df.to_dict(orient="records")


@app.get("/api/data/{profile_name}/trades")
def get_trades(
    profile_name: str,
    limit: int = 250,
    profile_path: Optional[str] = None,
    sync: bool = False,
) -> list[dict[str, Any]] | dict[str, Any]:
    """Get recent trades. If profile_path is provided, returns an object with
    trades (each including profit_display in display currency) and display_currency.

    Also syncs with broker first to detect any trades closed externally.
    """
    store = _store_for(profile_name)

    resolved_profile_path = _resolve_profile_path(profile_path) if profile_path else None

    # Sync with broker only when explicitly requested.
    if sync and resolved_profile_path and resolved_profile_path.exists():
        try:
            profile = load_profile_v1(resolved_profile_path)
            _sync_open_trades_with_broker(profile, store)
        except Exception as e:
            print(f"[api] sync error in get_trades: {e}")

    df = store.read_trades_df(profile_name).tail(limit)
    # Convert NaN to None for JSON
    df = df.where(pd.notna(df), None)
    records = _enrich_phase3_rows(df.to_dict(orient="records"))

    if not resolved_profile_path or not resolved_profile_path.exists():
        return records

    try:
        profile = load_profile_v1(resolved_profile_path)
    except Exception:
        return records

    curr, rate = _get_display_currency(profile)

    for row in records:
        profit_value = _normalized_trade_profit_usd(
            row,
            symbol_hint=str(getattr(profile, "symbol", "") or ""),
        )

        if profit_value is not None:
            row["profit_display"] = _convert_amount(profit_value, rate)
        else:
            row["profit_display"] = None
    return {"trades": records, "display_currency": curr}


@app.get("/api/data/{profile_name}/trade-history")
def get_trade_history(
    profile_name: str,
    profile_path: Optional[str] = None,
    days_back: int = 90,
) -> dict[str, Any]:
    """Get closed trade history for equity curve chart.

    Returns daily aggregated data: date, daily_profit, cumulative_profit, trade_count.
    Prefers local DB (complete data) over broker (may be paginated/incomplete).
    """
    from adapters.broker import get_adapter

    profile = None
    if profile_path and _resolve_profile_path(profile_path).exists():
        try:
            profile = load_profile_v1(_resolve_profile_path(profile_path))
        except Exception:
            pass

    curr, rate = ("USD", 1.0)
    if profile:
        curr, rate = _get_display_currency(profile)

    # Prefer broker history for OANDA so the chart matches the live broker view.
    if profile and getattr(profile, "broker_type", None) == "oanda":
        try:
            adapter = get_adapter(profile)
            adapter.initialize()
            try:
                closed_trades = adapter.get_closed_trade_summaries(
                    days_back=days_back,
                    symbol=profile.symbol,
                    pip_size=profile.pip_size,
                ) if hasattr(adapter, "get_closed_trade_summaries") else []
            finally:
                try:
                    adapter.shutdown()
                except Exception:
                    pass
            if closed_trades:
                by_date: dict[str, dict[str, Any]] = {}
                for t in closed_trades:
                    close_time = str(t.get("close_time") or "")
                    date_str = close_time[:10] if len(close_time) >= 10 else ""
                    if not date_str:
                        continue
                    profit_display = _convert_amount(float(t.get("profit", 0.0) or 0.0), rate) or 0.0
                    if date_str in by_date:
                        by_date[date_str]["daily_profit"] += profit_display
                        by_date[date_str]["trade_count"] += 1
                    else:
                        by_date[date_str] = {
                            "date": date_str,
                            "daily_profit": profit_display,
                            "trade_count": 1,
                        }
                sorted_days = sorted(by_date.values(), key=lambda d: d["date"])
                cum = 0.0
                for day in sorted_days:
                    day["daily_profit"] = round(day["daily_profit"], 2)
                    cum += day["daily_profit"]
                    day["cum_profit"] = round(cum, 2)
                return {"days": sorted_days, "display_currency": curr, "source": "broker"}
        except Exception as e:
            print(f"[api] trade-history oanda broker error: {e}")

    # Primary: local DB (has complete trade history)
    store = _store_for(profile_name)
    df = store.read_trades_df(profile_name)
    has_db_closed = False
    if not df.empty and "exit_price" in df.columns:
        closed_df = df[pd.to_numeric(df["exit_price"], errors="coerce").notna()].copy()
        if not closed_df.empty and "exit_timestamp_utc" in closed_df.columns:
            has_db_closed = True
            by_date_db: dict[str, dict[str, Any]] = {}
            for _, row in closed_df.iterrows():
                exit_ts = str(row.get("exit_timestamp_utc") or "")
                date_str = exit_ts[:10] if len(exit_ts) >= 10 else ""
                if not date_str:
                    continue
                profit_usd = _normalized_trade_profit_usd(
                    row,
                    symbol_hint=str(getattr(profile, "symbol", "") or ""),
                )
                profit_val = _convert_amount(profit_usd, rate) or 0.0
                if date_str in by_date_db:
                    by_date_db[date_str]["daily_profit"] += profit_val
                    by_date_db[date_str]["trade_count"] += 1
                else:
                    by_date_db[date_str] = {
                        "date": date_str,
                        "daily_profit": profit_val,
                        "trade_count": 1,
                    }
            sorted_days_db = sorted(by_date_db.values(), key=lambda d: d["date"])
            cum_db = 0.0
            for day in sorted_days_db:
                day["daily_profit"] = round(day["daily_profit"], 2)
                cum_db += day["daily_profit"]
                day["cum_profit"] = round(cum_db, 2)
            return {"days": sorted_days_db, "display_currency": curr, "source": "database"}

    # Fallback: broker history (only when local DB has no closed trades)
    if profile:
        try:
            adapter = get_adapter(profile)
            adapter.initialize()
            try:
                closed = adapter.get_closed_positions_from_history(
                    days_back=days_back,
                    symbol=profile.symbol,
                    pip_size=profile.pip_size,
                )
            finally:
                try:
                    adapter.shutdown()
                except Exception:
                    pass
            if closed:
                by_date: dict[str, dict[str, Any]] = {}
                for pos in closed:
                    exit_time = getattr(pos, "exit_time_utc", "") or ""
                    date_str = exit_time[:10] if len(exit_time) >= 10 else ""
                    if not date_str:
                        continue
                    profit = getattr(pos, "profit", 0.0) or 0.0
                    profit_display = _convert_amount(profit, rate) or 0.0
                    if date_str in by_date:
                        by_date[date_str]["daily_profit"] += profit_display
                        by_date[date_str]["trade_count"] += 1
                    else:
                        by_date[date_str] = {
                            "date": date_str,
                            "daily_profit": profit_display,
                            "trade_count": 1,
                        }
                sorted_days = sorted(by_date.values(), key=lambda d: d["date"])
                cum = 0.0
                for day in sorted_days:
                    day["daily_profit"] = round(day["daily_profit"], 2)
                    cum += day["daily_profit"]
                    day["cum_profit"] = round(cum, 2)
                return {"days": sorted_days, "display_currency": curr, "source": "broker"}
        except Exception as e:
            print(f"[api] trade-history broker error: {e}")

    return {"days": [], "display_currency": curr, "source": "database"}


@app.get("/api/data/{profile_name}/trade-history-detail")
def get_trade_history_detail(
    profile_name: str,
    profile_path: Optional[str] = None,
    days_back: int = 90,
) -> dict[str, Any]:
    """Get individual closed trade records for analytics.

    Returns per-trade data (not daily aggregates) for session, long/short,
    and spread analysis on the frontend.
    Prefers local DB (complete data) over broker (may be paginated/incomplete).
    """
    from adapters.broker import get_adapter

    cache_key = _cache_compose_key("trade_history_detail", profile_name, profile_path or "", days_back)
    cached_payload = _cache_get("trade_history_detail", cache_key)
    if cached_payload is not None:
        return cached_payload

    profile = None
    if profile_path and _resolve_profile_path(profile_path).exists():
        try:
            profile = load_profile_v1(_resolve_profile_path(profile_path))
        except Exception:
            pass

    curr, rate = ("USD", 1.0)
    if profile:
        curr, rate = _get_display_currency(profile)

    # Primary: local DB (has complete trade history)
    store = _store_for(profile_name)
    df = store.read_trades_df(profile_name)
    has_db_closed = False
    if not df.empty and "exit_price" in df.columns:
        closed_df = df[pd.to_numeric(df["exit_price"], errors="coerce").notna()].copy()
        if not closed_df.empty:
            has_db_closed = True

            # Join with snapshots for spread_pips where snapshot_id is available
            snap_spreads: dict[int, float] = {}
            if "snapshot_id" in closed_df.columns:
                snap_ids = closed_df["snapshot_id"].dropna().astype(int).unique().tolist()
                if snap_ids:
                    snap_df = store.read_snapshots_df(profile_name)
                    if not snap_df.empty and "spread_pips" in snap_df.columns:
                        for _, srow in snap_df[snap_df["id"].isin(snap_ids)].iterrows():
                            if pd.notna(srow.get("spread_pips")):
                                snap_spreads[int(srow["id"])] = float(srow["spread_pips"])

            pip_size = float(profile.pip_size) if profile else 0.01
            trades_list = []
            for _, row in closed_df.iterrows():
                side = str(row.get("side") or "").lower()
                entry_price = float(row["entry_price"]) if pd.notna(row.get("entry_price")) else 0
                exit_price = float(row["exit_price"]) if pd.notna(row.get("exit_price")) else 0
                pips_val = float(row["pips"]) if pd.notna(row.get("pips")) else None
                if pips_val is None and entry_price and exit_price and pip_size:
                    if side == "buy":
                        pips_val = round((exit_price - entry_price) / pip_size, 1)
                    elif side == "sell":
                        pips_val = round((entry_price - exit_price) / pip_size, 1)
                profit_raw = _normalized_trade_profit_usd(
                    row,
                    symbol_hint=str(getattr(profile, "symbol", "") or ""),
                )

                spread = None
                snap_id = row.get("snapshot_id")
                if snap_id is not None and pd.notna(snap_id):
                    spread = snap_spreads.get(int(snap_id))

                trades_list.append({
                    "side": side,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "entry_time_utc": str(row.get("timestamp_utc") or ""),
                    "exit_time_utc": str(row.get("exit_timestamp_utc") or ""),
                    "profit": _convert_amount(profit_raw, rate) if profit_raw is not None else None,
                    "pips": pips_val,
                    "volume": float(row.get("size_lots") or 0),
                    "spread_pips": spread,
                })
            payload = {"trades": trades_list, "display_currency": curr, "source": "database"}
            _cache_set("trade_history_detail", cache_key, payload)
            return payload

    # Fallback: broker history (only when local DB has no closed trades)
    if profile:
        try:
            adapter = get_adapter(profile)
            adapter.initialize()
            try:
                closed = adapter.get_closed_positions_from_history(
                    days_back=days_back,
                    symbol=profile.symbol,
                    pip_size=profile.pip_size,
                )
            finally:
                try:
                    adapter.shutdown()
                except Exception:
                    pass
            if closed:
                trades_list_broker = []
                for pos in closed:
                    trades_list_broker.append({
                        "side": getattr(pos, "side", ""),
                        "entry_price": getattr(pos, "entry_price", 0),
                        "exit_price": getattr(pos, "exit_price", 0),
                        "entry_time_utc": getattr(pos, "entry_time_utc", ""),
                        "exit_time_utc": getattr(pos, "exit_time_utc", ""),
                        "profit": _convert_amount(getattr(pos, "profit", 0), rate),
                        "pips": getattr(pos, "pips", None),
                        "volume": getattr(pos, "volume", 0),
                        "spread_pips": None,
                    })
                payload = {"trades": trades_list_broker, "display_currency": curr, "source": "broker"}
                _cache_set("trade_history_detail", cache_key, payload)
                return payload
        except Exception as e:
            print(f"[api] trade-history-detail broker error: {e}")

    payload = {"trades": [], "display_currency": curr, "source": "database"}
    _cache_set("trade_history_detail", cache_key, payload)
    return payload


@app.get("/api/data/{profile_name}/executions")
def get_executions(profile_name: str, limit: int = 50) -> list[dict[str, Any]]:
    """Get recent executions."""
    store = _store_for(profile_name)
    df = store.read_executions_df(profile_name).tail(limit)
    df = df.where(pd.notna(df), None)
    return df.to_dict(orient="records")


@app.get("/api/data/{profile_name}/rejection-breakdown")
def get_rejection_breakdown(profile_name: str, limit: int = 200) -> dict[str, int]:
    """Get rejection breakdown from recent executions."""
    store = _store_for(profile_name)
    df = store.read_executions_df(profile_name).tail(limit)
    
    if df.empty or "reason" not in df.columns:
        return {}
    
    def reason_key(r: Any) -> str:
        if pd.isna(r) or not r:
            return "unknown"
        s = str(r).strip()
        if ":" in s:
            return s.split(":")[0].strip()
        if " " in s:
            return s.split()[0].strip()
        return s[:40]
    
    df = df.copy()
    df["reason_group"] = df["reason"].map(reason_key)
    breakdown = df.groupby("reason_group", dropna=False).size().sort_values(ascending=False)
    return breakdown.to_dict()


@app.get("/api/data/{profile_name}/phase3-decisions")
def get_phase3_decisions(profile_name: str, days: int = 3, limit: int = 5000, profile_path: Optional[str] = None) -> list[dict[str, Any]]:
    """Return Phase 3 per-closed-M1 decision rows (stored in executions).

    Rows are written by run_loop as signal_id like:
      eval:phase3_integrated:<policy_id>:<m1_bar_time>
    """
    from datetime import datetime, timezone, timedelta

    days = max(1, min(int(days), 60))
    limit = max(100, min(int(limit), 20000))

    p3_key = _cache_compose_key("phase3_decisions", profile_name, profile_path or "", str(days), str(limit))
    cached_rows = _cache_get("phase3_decisions", p3_key)
    if cached_rows is not None:
        return cached_rows

    log_dir = _pick_best_dashboard_log_dir(profile_name, profile_path)
    active_profile_name = log_dir.name if log_dir is not None else profile_name
    store = _store_for(profile_name, log_dir=log_dir)
    tail_n = min(150_000, max(50_000, int(limit) * 50, days * 2500))
    df = store.read_executions_tail_df(active_profile_name, limit=tail_n)
    enriched: list[dict[str, Any]] = []
    if not df.empty and "signal_id" in df.columns:
        df = df.where(pd.notna(df), None)
        since_dt = datetime.now(timezone.utc) - timedelta(days=days)
        ts_col = "timestamp_utc" if "timestamp_utc" in df.columns else ("timestamp" if "timestamp" in df.columns else None)
        if ts_col is not None:
            ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
            df = df[ts >= since_dt]
        df = df[df["signal_id"].astype(str).str.startswith("eval:phase3_integrated:")]
        df = df.tail(limit)
        cfg = _load_phase3_sizing_cfg_api(profile_name=profile_name, profile_path=profile_path)
        rows = df.to_dict(orient="records")
        for row in rows:
            out = dict(row)
            out["phase3_session"] = infer_phase3_session_from_signal_id(out.get("signal_id"), cfg)
            out["blocking_filter_ids"] = parse_phase3_blocking_filter_ids(out.get("reason"))
            out["reason_text"] = extract_phase3_reason_text(out.get("reason"))
            enriched.append(out)
    if enriched:
        _cache_set("phase3_decisions", p3_key, enriched)
        return enriched
    diag = _load_phase3_decision_rows_from_diagnostics(log_dir, days=days, limit=limit)
    _cache_set("phase3_decisions", p3_key, diag)
    return diag


@app.get("/api/data/{profile_name}/phase3-blockers-breakdown")
def get_phase3_blockers_breakdown(profile_name: str, days: int = 7, limit: int = 20000, profile_path: Optional[str] = None) -> dict[str, int]:
    """Aggregate Phase 3 blocking filters from stored execution reasons.

    run_loop encodes blockers as:  "... | blocks=filter_id_a,filter_id_b"
    """
    from datetime import datetime, timezone, timedelta

    days = max(1, min(int(days), 60))
    limit = max(100, min(int(limit), 50000))

    b_key = _cache_compose_key("phase3_blockers", profile_name, profile_path or "", str(days), str(limit))
    cached_b = _cache_get("phase3_blockers", b_key)
    if cached_b is not None:
        return cached_b

    log_dir = _pick_best_dashboard_log_dir(profile_name, profile_path)
    active_profile_name = log_dir.name if log_dir is not None else profile_name
    store = _store_for(profile_name, log_dir=log_dir)
    tail_n = min(150_000, max(60_000, int(limit) * 4, days * 2500))
    df = store.read_executions_tail_df(active_profile_name, limit=tail_n)
    counts: dict[str, int] = {}
    if not df.empty and "reason" in df.columns and "signal_id" in df.columns:
        since_dt = datetime.now(timezone.utc) - timedelta(days=days)
        ts_col = "timestamp_utc" if "timestamp_utc" in df.columns else ("timestamp" if "timestamp" in df.columns else None)
        if ts_col is not None:
            ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
            df = df[ts >= since_dt]
        df = df[df["signal_id"].astype(str).str.startswith("eval:phase3_integrated:")].tail(limit)
        for r in df["reason"].tolist():
            if r is None:
                continue
            s = str(r)
            if "blocks=" not in s:
                continue
            try:
                blocks_part = s.split("blocks=", 1)[1].strip()
                for stop in (" | ", "\t", "\n"):
                    if stop in blocks_part:
                        blocks_part = blocks_part.split(stop, 1)[0]
                for bid in [b.strip() for b in blocks_part.split(",") if b.strip()]:
                    counts[bid] = int(counts.get(bid, 0)) + 1
            except Exception:
                continue
    if counts:
        out = dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))
        _cache_set("phase3_blockers", b_key, out)
        return out

    for row in _load_phase3_decision_rows_from_diagnostics(log_dir, days=days, limit=limit):
        for bid in list(row.get("blocking_filter_ids") or []):
            counts[bid] = int(counts.get(bid, 0)) + 1

    out = dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))
    _cache_set("phase3_blockers", b_key, out)
    return out


@app.get("/api/data/{profile_name}/phase3-provenance")
def get_phase3_provenance(profile_name: str, profile_path: Optional[str] = None) -> dict[str, Any]:
    pv_key = _cache_compose_key("phase3_provenance", profile_name, profile_path or "")
    hit_pv = _cache_get("phase3_provenance", pv_key)
    if hit_pv is not None:
        return hit_pv

    def _waiting_payload() -> dict[str, Any]:
        return {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "package_id": None,
            "preset_name": "",
            "session": None,
            "strategy_tag": None,
            "strategy_family": None,
            "window_label": None,
            "ownership_cell": None,
            "regime_label": None,
            "defensive_flags": [],
            "attempted": False,
            "placed": False,
            "outcome": "waiting",
            "reason": None,
            "blocking_filter_ids": [],
            "last_block_reason": None,
            "exit_policy": None,
            "frozen_modifiers": [],
            "data_freshness": {
                "dashboard_timestamp_utc": None,
                "decision_timestamp_utc": None,
            },
        }
    try:
        dashboard = _get_dashboard_impl(profile_name, profile_path)
        if "error" in dashboard:
            return _waiting_payload()

        latest_decision = None
        decisions = get_phase3_decisions(profile_name, days=7, limit=2000, profile_path=profile_path)
        if decisions:
            latest_decision = decisions[-1]
        payload = build_phase3_provenance_payload(
            preset_name=str(dashboard.get("preset_name") or ""),
            context_items=list(dashboard.get("context") or []),
            filters=list(dashboard.get("filters") or []),
            latest_decision=latest_decision,
            sizing_cfg=_load_phase3_sizing_cfg_api(profile_name=profile_name, profile_path=profile_path),
            dashboard_timestamp_utc=dashboard.get("timestamp_utc"),
            last_block_reason=dashboard.get("last_block_reason"),
        )
        _cache_set("phase3_provenance", pv_key, payload)
        return payload
    except Exception:
        return _waiting_payload()


@app.get("/api/system/phase3-paper-acceptance")
def get_phase3_paper_acceptance() -> dict[str, Any]:
    return build_phase3_acceptance_payload(_research_out_path("paper_acceptance_validation.json"))


@app.get("/api/system/phase3-defensive-monitor")
def get_phase3_defensive_monitor() -> dict[str, Any]:
    return build_phase3_defensive_monitor_payload(
        _research_out_path("defensive_paper_guardrail_profile.json"),
        _research_out_path("defensive_paper_pain_report.json"),
    )


@app.get("/api/data/{profile_name}/stats")
def get_quick_stats(profile_name: str, profile_path: Optional[str] = None, sync: bool = False) -> dict[str, Any]:
    """Get quick stats (win rate, avg pips, total profit).
    
    Prefers broker (MT5/OANDA) deal history as source of truth when available.
    Falls back to local database when broker report data is unavailable.
    """
    from adapters.broker import get_adapter

    # Resolve profile for symbol/pip_size
    profile = None
    if profile_path and _resolve_profile_path(profile_path).exists():
        try:
            profile = load_profile_v1(_resolve_profile_path(profile_path))
        except Exception:
            pass
    if profile is None:
        for p in _list_profile_paths():
            if p.stem == profile_name:
                try:
                    profile = load_profile_v1(p)
                    break
                except Exception:
                    pass

    # Prefer broker report stats for OANDA, or when explicitly requested.
    if profile and (sync or getattr(profile, "broker_type", None) == "oanda"):
        try:
            adapter = get_adapter(profile)
            adapter.initialize()
            try:
                mt5_stats = adapter.get_mt5_report_stats(
                    symbol=profile.symbol,
                    pip_size=profile.pip_size,
                    days_back=90,
                )
            finally:
                adapter.shutdown()
        except Exception:
            mt5_stats = None
        if mt5_stats is not None:
            curr, rate = _get_display_currency(profile)
            source = "oanda" if getattr(profile, "broker_type", None) == "oanda" else "mt5"
            return {
                "source": source,
                "display_currency": curr,
                "closed_trades": mt5_stats.closed_trades,
                "win_rate": mt5_stats.win_rate,
                "avg_pips": mt5_stats.avg_pips,
                "total_profit": _convert_amount(mt5_stats.total_profit, rate),
                "total_commission": _convert_amount(mt5_stats.total_commission, rate),
                "total_swap": _convert_amount(mt5_stats.total_swap, rate),
                "wins": mt5_stats.wins,
                "losses": mt5_stats.losses,
                "trades_with_profit": mt5_stats.closed_trades,
                "trades_without_profit": 0,
                "trades_with_position_id": mt5_stats.closed_trades,
            }

    # Fallback: local database
    store = _store_for(profile_name)
    df = store.read_trades_df(profile_name)

    curr, rate = ("USD", 1.0)
    if profile:
        curr, rate = _get_display_currency(profile)

    if df.empty:
        return {
            "source": "database",
            "display_currency": curr,
            "closed_trades": 0,
            "win_rate": None,
            "avg_pips": None,
            "total_profit": None,
            "trades_with_profit": 0,
            "trades_without_profit": 0,
            "trades_with_position_id": 0,
        }

    closed = df[pd.to_numeric(df.get("exit_price"), errors="coerce").notna()].copy() if "exit_price" in df.columns else pd.DataFrame()

    if closed.empty:
        return {
            "source": "database",
            "display_currency": curr,
            "closed_trades": 0,
            "win_rate": None,
            "avg_pips": None,
            "total_profit": None,
            "trades_with_profit": 0,
            "trades_without_profit": 0,
            "trades_with_position_id": 0,
        }

    pips = pd.to_numeric(closed.get("pips"), errors="coerce")
    symbol_hint = str(getattr(profile, "symbol", "") or "")
    profit_col = pd.to_numeric(
        closed.apply(lambda r: _normalized_trade_profit_usd(r, symbol_hint=symbol_hint), axis=1),
        errors="coerce",
    )

    use_profit = profit_col.notna()
    use_pips_fallback = ~use_profit & pips.notna()
    can_classify = use_profit | use_pips_fallback
    is_win_profit = use_profit & (profit_col > 0)
    is_win_pips = use_pips_fallback & (pips > 0)
    is_win = is_win_profit | is_win_pips

    total = int(can_classify.sum())
    wins_profit = int(is_win_profit.sum())
    avg_pips_val = float(pips.mean()) if pips.notna().any() else 0
    profit_used_count = int(use_profit.sum())
    if total > 0 and profit_used_count == total and wins_profit == total and avg_pips_val < -0.5:
        wins = int(((pips > 0) & pips.notna()).sum())
    else:
        wins = int(is_win.sum())
    avg_pips = round(float(pips.mean()), 3) if pips.notna().any() else None
    win_rate = round(wins / total, 3) if total > 0 else None
    total_profit = round(float(profit_col.sum()), 2) if profit_col.notna().any() else None

    profit_col_diag = profit_col
    pos_id_col = closed.get("mt5_position_id")
    trades_with_profit = int(profit_col_diag.notna().sum())
    trades_without_profit = int(profit_col_diag.isna().sum())
    trades_with_position_id = int(pd.to_numeric(pos_id_col, errors="coerce").notna().sum()) if pos_id_col is not None else 0

    return {
        "source": "database",
        "display_currency": curr,
        "closed_trades": total,
        "win_rate": win_rate,
        "avg_pips": avg_pips,
        "total_profit": _convert_amount(total_profit, rate),
        "trades_with_profit": trades_with_profit,
        "trades_without_profit": trades_without_profit,
        "trades_with_position_id": trades_with_position_id,
    }


@app.get("/api/data/{profile_name}/mt5-report")
def get_mt5_report(profile_name: str, profile_path: Optional[str] = None) -> dict[str, Any]:
    """Get full MT5 report (Summary, Closed P/L, Long/Short). Same data as View -> Reports.
    Returns {source: null} when MT5 is unavailable."""
    cache_key = _cache_compose_key("mt5_report", profile_name, profile_path or "")
    cached_payload = _cache_get("mt5_report", cache_key)
    if cached_payload is not None:
        return cached_payload

    profile = None
    if profile_path and _resolve_profile_path(profile_path).exists():
        try:
            profile = load_profile_v1(_resolve_profile_path(profile_path))
        except Exception:
            pass
    if profile is None:
        for p in _list_profile_paths():
            if p.stem == profile_name:
                try:
                    profile = load_profile_v1(p)
                    break
                except Exception:
                    pass
    if not profile:
        payload = {"source": None}
        _cache_set("mt5_report", cache_key, payload)
        return payload
    if getattr(profile, "broker_type", None) == "oanda":
        try:
            from adapters.broker import get_adapter
            adapter = get_adapter(profile)
            adapter.initialize()
            try:
                acct = adapter.get_account_info()
            finally:
                adapter.shutdown()
        except Exception:
            payload = {"source": None}
            _cache_set("mt5_report", cache_key, payload)
            return payload
        result = {
            "source": "oanda",
            "summary": {
                "balance": round(float(getattr(acct, "balance", 0.0) or 0.0), 2),
                "equity": round(float(getattr(acct, "equity", getattr(acct, "balance", 0.0)) or 0.0), 2),
                "margin": round(float(getattr(acct, "margin", 0.0) or 0.0), 2),
                "free_margin": round(float(getattr(acct, "margin_free", 0.0) or 0.0), 2),
            },
            "closed_pl": {
                "closed_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "total_profit": 0.0,
                "total_commission": 0.0,
                "total_swap": 0.0,
                "gross_profit": 0.0,
                "gross_loss": 0.0,
                "profit_factor": 1.0,
                "largest_profit_trade": 0.0,
                "largest_loss_trade": 0.0,
                "expected_payoff": 0.0,
                "avg_pips": None,
                "total_pips": 0.0,
            },
            "long_short": {
                "long_trades": 0,
                "long_wins": 0,
                "long_win_pct": 0.0,
                "short_trades": 0,
                "short_wins": 0,
                "short_win_pct": 0.0,
            },
        }
        curr, rate = _get_display_currency(profile)
        result["display_currency"] = curr
        s = result["summary"]
        for k in ("balance", "equity", "margin", "free_margin"):
            if k in s and s[k] is not None:
                s[k] = round(float(s[k]) * rate, 2)
        _cache_set("mt5_report", cache_key, result)
        return result
    try:
        from adapters.broker import get_adapter
        adapter = get_adapter(profile)
        adapter.initialize()
        try:
            result = adapter.get_mt5_full_report(symbol=profile.symbol, pip_size=profile.pip_size, days_back=90)
        finally:
            adapter.shutdown()
    except Exception:
        result = None
    if result is None:
        payload = {"source": None}
        _cache_set("mt5_report", cache_key, payload)
        return payload
    curr, rate = _get_display_currency(profile)
    result["display_currency"] = curr
    if "summary" in result and result["summary"]:
        s = result["summary"]
        for k in ("balance", "equity", "margin", "free_margin"):
            if k in s and s[k] is not None:
                s[k] = round(float(s[k]) * rate, 2)
    if "closed_pl" in result and result["closed_pl"]:
        pl = result["closed_pl"]
        for k in ("total_profit", "total_commission", "total_swap", "gross_profit", "gross_loss",
                  "largest_profit_trade", "largest_loss_trade", "expected_payoff"):
            if k in pl and pl[k] is not None:
                pl[k] = round(float(pl[k]) * rate, 2)
    _cache_set("mt5_report", cache_key, result)
    return result


@app.get("/api/data/{profile_name}/stats-by-preset")
def get_stats_by_preset(profile_name: str, profile_path: Optional[str] = None) -> dict[str, Any]:
    """Get comprehensive statistics grouped by preset name.

    Preset comes from local DB. Win/loss and profit/commission use the same authoritative
    MT5 history as Quick Stats: we fetch all closed positions from MT5 once via
    get_closed_positions_from_history, then match DB trades by mt5_position_id. Trades
    not found in MT5 (e.g. old or different symbol) fall back to DB profit or pips.
    Run "Sync from MT5" so DB trades have mt5_position_id for best matching.
    """
    from adapters.broker import get_adapter

    cache_key = _cache_compose_key("stats_by_preset", profile_name, profile_path or "")
    cached_payload = _cache_get("stats_by_preset", cache_key)
    if cached_payload is not None:
        return cached_payload

    store = _store_for(profile_name)
    df = store.read_trades_df(profile_name)

    if df.empty:
        payload = {"presets": {}, "source": "database"}
        _cache_set("stats_by_preset", cache_key, payload)
        return payload

    closed = df[pd.to_numeric(df.get("exit_price"), errors="coerce").notna()].copy() if "exit_price" in df.columns else pd.DataFrame()

    if closed.empty:
        payload = {"presets": {}, "source": "database"}
        _cache_set("stats_by_preset", cache_key, payload)
        return payload

    if "preset_name" not in closed.columns:
        closed["preset_name"] = "Unknown"
    closed["preset_name"] = closed["preset_name"].fillna("Unknown")

    closed["pips"] = pd.to_numeric(closed.get("pips"), errors="coerce")
    closed["r_multiple"] = pd.to_numeric(closed.get("r_multiple"), errors="coerce")
    closed["profit"] = pd.to_numeric(closed.get("profit"), errors="coerce")

    mt5_financials: dict[int, dict] = {}
    profile = None
    if profile_path and _resolve_profile_path(profile_path).exists():
        try:
            profile = load_profile_v1(_resolve_profile_path(profile_path))
        except Exception:
            pass
    if profile is None:
        for p in _list_profile_paths():
            if p.stem == profile_name:
                try:
                    profile = load_profile_v1(p)
                    break
                except Exception:
                    pass

    if profile:
        try:
            adapter = get_adapter(profile)
            adapter.initialize()
            try:
                closed_positions = adapter.get_closed_positions_from_history(
                    days_back=90,
                    symbol=profile.symbol,
                    pip_size=profile.pip_size,
                )
                for pos in closed_positions:
                    mt5_financials[pos.position_id] = {
                        "profit": pos.profit,
                        "commission": pos.commission,
                        "swap": pos.swap,
                    }
            finally:
                adapter.shutdown()
        except Exception:
            pass

    def _get_profit(row: pd.Series) -> float | None:
        pos_id = row.get("mt5_position_id")
        if pos_id is not None and pd.notna(pos_id):
            pid = int(pos_id)
            if pid in mt5_financials:
                return mt5_financials[pid]["profit"]
        return _normalized_trade_profit_usd(
            row,
            symbol_hint=str(getattr(profile, "symbol", "") or ""),
        )

    def _get_commission(row: pd.Series) -> float:
        pos_id = row.get("mt5_position_id")
        if pos_id is not None and pd.notna(pos_id):
            pid = int(pos_id)
            if pid in mt5_financials:
                return mt5_financials[pid]["commission"]
        return 0.0

    closed["_profit_use"] = closed.apply(_get_profit, axis=1)
    closed["_commission"] = closed.apply(_get_commission, axis=1)

    def _is_win(row: pd.Series) -> bool:
        p = row.get("_profit_use")
        if p is not None:
            return float(p) > 0
        if pd.notna(row.get("pips")):
            return float(row["pips"]) > 0
        return False

    def _is_loss(row: pd.Series) -> bool:
        p = row.get("_profit_use")
        if p is not None:
            return float(p) < 0
        if pd.notna(row.get("pips")):
            return float(row["pips"]) < 0
        return False

    source = ("oanda" if getattr(profile, "broker_type", None) == "oanda" else "mt5") if mt5_financials else "database"
    curr, rate = _get_display_currency(profile) if profile else ("USD", 1.0)
    presets_data: dict[str, Any] = {}

    def _stats_for_group(group: pd.DataFrame) -> dict[str, Any]:
        pips = group["pips"]
        r_mult = group["r_multiple"]
        profit_use = group["_profit_use"]
        commission_col = group["_commission"]
        total_trades = len(group)
        is_win = group.apply(_is_win, axis=1)
        is_loss = group.apply(_is_loss, axis=1)
        wins = int(is_win.sum())
        losses = int(is_loss.sum())
        win_rate = round(wins / total_trades, 3) if total_trades > 0 else None
        total_pips = round(float(pips.sum()), 2) if pips.notna().any() else 0
        avg_pips = round(float(pips.mean()), 2) if pips.notna().any() else None
        avg_rr = round(float(r_mult.mean()), 2) if r_mult.notna().any() else None
        total_profit = round(float(profit_use.sum()), 2) if profit_use.notna().any() else None
        total_commission = round(float(commission_col.sum()), 2)
        best_trade = round(float(pips.max()), 2) if pips.notna().any() else None
        worst_trade = round(float(pips.min()), 2) if pips.notna().any() else None
        win_streak, loss_streak = 0, 0
        current_win_streak, current_loss_streak = 0, 0
        for _, row in group.iterrows():
            w, l = _is_win(row), _is_loss(row)
            if w:
                current_win_streak += 1
                current_loss_streak = 0
                win_streak = max(win_streak, current_win_streak)
            elif l:
                current_loss_streak += 1
                current_win_streak = 0
                loss_streak = max(loss_streak, current_loss_streak)
            else:
                current_win_streak = current_loss_streak = 0
        if profit_use.notna().any():
            gross_profit = float(profit_use[profit_use > 0].sum()) if (profit_use > 0).any() else 0
            gross_loss = abs(float(profit_use[profit_use < 0].sum())) if (profit_use < 0).any() else 0
        else:
            gross_profit = float(pips[pips > 0].sum()) if (pips > 0).any() else 0
            gross_loss = abs(float(pips[pips < 0].sum())) if (pips < 0).any() else 0
        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else None
        cumulative = pips.cumsum()
        running_max = cumulative.cummax()
        drawdown = running_max - cumulative
        max_drawdown = round(float(drawdown.max()), 2) if drawdown.notna().any() else 0
        return {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "total_pips": total_pips,
            "total_profit": _convert_amount(total_profit, rate),
            "total_commission": _convert_amount(total_commission, rate),
            "avg_pips": avg_pips,
            "avg_rr": avg_rr,
            "best_trade": best_trade,
            "worst_trade": worst_trade,
            "win_streak": win_streak,
            "loss_streak": loss_streak,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
        }

    for preset_name, group in closed.groupby("preset_name"):
        presets_data[str(preset_name)] = _stats_for_group(group)

    # Phase 3 session breakdown: match by policy_type since preset_name varies by profile
    _phase3_policy_col = closed.get("policy_type")
    _phase3_has_sessions = "entry_session" in closed.columns and _phase3_policy_col is not None
    if _phase3_has_sessions:
        phase3_closed = closed[_phase3_policy_col == "phase3_integrated"]
        # Find the preset_name key these trades are grouped under
        _phase3_preset_key = None
        if not phase3_closed.empty:
            _phase3_preset_key = str(phase3_closed["preset_name"].iloc[0])
        if _phase3_preset_key and _phase3_preset_key in presets_data:
            session_trades = phase3_closed[phase3_closed["entry_session"].isin(["tokyo", "london", "ny"])]
            if not session_trades.empty:
                by_session: dict[str, Any] = {}
                for sess in ("tokyo", "london", "ny"):
                    grp = session_trades[session_trades["entry_session"] == sess]
                    if grp.empty:
                        continue
                    by_session[sess] = _stats_for_group(grp)
                presets_data[_phase3_preset_key]["by_session"] = by_session

    payload = {"presets": presets_data, "source": source, "display_currency": curr}
    _cache_set("stats_by_preset", cache_key, payload)
    return payload


def _compute_technical_analysis_payload(path: Path) -> dict[str, Any]:
    """Broker-heavy TA build; run under a wall-clock timeout in the request handler."""
    from core.indicators import ema
    from core.ta_analysis import compute_ta_for_tf
    from core.timeframes import Timeframe
    from core.book_cache import get_book_cache
    from core.scalp_score import ScalpScore

    profile = load_profile_v1(path)
    from adapters.broker import get_adapter

    adapter = get_adapter(profile)
    try:
        try:
            adapter.initialize()
        except RuntimeError as init_err:
            msg = str(init_err)
            if "MetaTrader5 is not installed" in msg or "MT5" in msg:
                raise HTTPException(
                    status_code=503,
                    detail="Broker data unavailable: MT5 is not installed on this server. Set Broker to OANDA in Profile Editor and add your API key to view technical analysis here.",
                ) from init_err
            raise
        adapter.ensure_symbol(profile.symbol)
        timeframes: list[Timeframe] = ["H4", "H1", "M30", "M15", "M5", "M3", "M1"]
        result: dict[str, Any] = {"timeframes": {}}

        book_cache = get_book_cache()
        try:
            book_cache.poll_books(adapter, profile.symbol)
        except Exception:
            pass

        scalp_tick = None
        try:
            scalp_tick = adapter.get_tick(profile.symbol)
        except Exception:
            pass

        for tf in timeframes:
            try:
                df = _get_bars_cached(adapter, profile.symbol, tf, count=700)
                if df is None or df.empty:
                    result["timeframes"][tf] = {
                        "error": f"No data available for {tf}",
                        "regime": "unknown",
                        "rsi": {"value": None, "zone": "unknown", "period": 14},
                        "macd": {"line": None, "signal": None, "histogram": None, "direction": "neutral"},
                        "atr": {"value": None, "value_pips": None, "state": "unknown"},
                        "price": {"current": None, "recent_high": None, "recent_low": None},
                        "bollinger": {"upper": None, "middle": None, "lower": None},
                        "vwap": None,
                        "summary": f"{tf}: No data available.",
                        "ohlc": [],
                        "all_emas": {},
                        "bollinger_series": {"upper": [], "middle": [], "lower": []},
                    }
                    continue
                ta = compute_ta_for_tf(profile, tf, df)
                macd_direction = "neutral"
                if ta.macd_hist is not None:
                    if ta.macd_hist > 0:
                        macd_direction = "positive"
                    elif ta.macd_hist < 0:
                        macd_direction = "negative"
                df_tail = df.tail(500)
                timestamps = df_tail["time"].apply(lambda t: int(t.timestamp())).values
                ohlc_data = [
                    {"time": int(ts), "open": round(float(o), 3), "high": round(float(h), 3), "low": round(float(l), 3), "close": round(float(c), 3)}
                    for ts, o, h, l, c in zip(timestamps, df_tail["open"].values, df_tail["high"].values, df_tail["low"].values, df_tail["close"].values)
                ]
                close = df["close"]
                tail_idx = df_tail.index
                all_emas_arrs: dict[str, list[dict[str, Any]]] = {}
                for p in [5, 7, 9, 11, 13, 15, 17, 21, 34, 50, 200]:
                    s = ema(close, p)
                    s_tail = s.reindex(tail_idx).dropna()
                    ts_arr = df_tail.loc[s_tail.index, "time"].apply(lambda t: int(t.timestamp()))
                    all_emas_arrs[f"ema{p}"] = [{"time": int(t), "value": round(float(v), 3)} for t, v in zip(ts_arr.values, s_tail.values)]
                from core.indicators import bollinger_bands

                bb_upper, bb_middle, bb_lower = bollinger_bands(close, 20, 2.0)
                bb_u_tail = bb_upper.reindex(tail_idx).dropna()
                bb_m_tail = bb_middle.reindex(tail_idx).dropna()
                bb_l_tail = bb_lower.reindex(tail_idx).dropna()
                common_bb_idx = bb_u_tail.index
                bb_ts = df_tail.loc[common_bb_idx, "time"].apply(lambda t: int(t.timestamp())).values
                bb_series: dict[str, list[dict[str, Any]]] = {
                    "upper": [{"time": int(t), "value": round(float(v), 3)} for t, v in zip(bb_ts, bb_u_tail.values)],
                    "middle": [{"time": int(t), "value": round(float(v), 3)} for t, v in zip(bb_ts, bb_m_tail.values)],
                    "lower": [{"time": int(t), "value": round(float(v), 3)} for t, v in zip(bb_ts, bb_l_tail.values)],
                }
                scalp_score_data = None
                if tf in ("M1", "M3", "M5") and scalp_tick and len(df) >= 20:
                    try:
                        from datetime import datetime, timezone as tz

                        ob_snaps = [s.data for s in book_cache.get_order_books(profile.symbol)]
                        pb_snaps = [s.data for s in book_cache.get_position_books(profile.symbol)]
                        ss = ScalpScore.calculate(
                            df=df,
                            tick_bid=scalp_tick.bid,
                            tick_ask=scalp_tick.ask,
                            order_book_snapshots=ob_snaps,
                            position_book_snapshots=pb_snaps,
                            pip_size=profile.pip_size,
                            timeframe=tf,
                            timestamp_iso=datetime.now(tz.utc).isoformat(),
                        )
                        scalp_score_data = {
                            "finalScore": ss.final_score,
                            "direction": ss.direction,
                            "confidence": ss.confidence,
                            "killSwitch": ss.kill_switch,
                            "killReason": ss.kill_reason,
                            "layers": ss.layers,
                            "timestamp": ss.timestamp,
                        }
                    except Exception:
                        pass

                result["timeframes"][tf] = {
                    "scalp_score": scalp_score_data,
                    "regime": ta.regime,
                    "rsi": {
                        "value": round(ta.rsi_value, 2) if ta.rsi_value is not None else None,
                        "zone": ta.rsi_zone,
                        "period": 14,
                    },
                    "macd": {
                        "line": round(ta.macd_value, 5) if ta.macd_value is not None else None,
                        "signal": round(ta.macd_signal, 5) if ta.macd_signal is not None else None,
                        "histogram": round(ta.macd_hist, 5) if ta.macd_hist is not None else None,
                        "direction": macd_direction,
                    },
                    "atr": {
                        "value": round(ta.atr_value, 5) if ta.atr_value is not None else None,
                        "value_pips": round(ta.atr_value / profile.pip_size, 1) if ta.atr_value is not None else None,
                        "state": ta.atr_state,
                    },
                    "price": {
                        "current": round(ta.price, 3) if ta.price is not None else None,
                        "recent_high": round(ta.recent_high, 3) if ta.recent_high is not None else None,
                        "recent_low": round(ta.recent_low, 3) if ta.recent_low is not None else None,
                    },
                    "bollinger": {
                        "upper": round(ta.bollinger_upper, 5) if ta.bollinger_upper is not None else None,
                        "middle": round(ta.bollinger_middle, 5) if ta.bollinger_middle is not None else None,
                        "lower": round(ta.bollinger_lower, 5) if ta.bollinger_lower is not None else None,
                    },
                    "vwap": round(ta.vwap_value, 5) if ta.vwap_value is not None else None,
                    "summary": ta.summary,
                    "ohlc": ohlc_data,
                    "all_emas": all_emas_arrs,
                    "bollinger_series": bb_series,
                }
            except Exception as tf_error:
                result["timeframes"][tf] = {
                    "error": str(tf_error),
                    "regime": "unknown",
                    "rsi": {"value": None, "zone": "unknown", "period": 14},
                    "macd": {"line": None, "signal": None, "histogram": None, "direction": "neutral"},
                    "atr": {"value": None, "value_pips": None, "state": "unknown"},
                    "price": {"current": None, "recent_high": None, "recent_low": None},
                    "bollinger": {"upper": None, "middle": None, "lower": None},
                    "vwap": None,
                    "summary": f"{tf}: Error fetching data.",
                    "ohlc": [],
                    "all_emas": {},
                    "bollinger_series": {"upper": [], "middle": [], "lower": []},
                }

        tick = scalp_tick
        if tick:
            spread_pips = (tick.ask - tick.bid) / profile.pip_size
            result["current_tick"] = {
                "bid": round(tick.bid, 3),
                "ask": round(tick.ask, 3),
                "spread_pips": round(spread_pips, 1),
            }

        return result
    finally:
        try:
            adapter.shutdown()
        except Exception:
            pass


@app.get("/api/data/{profile_name}/technical-analysis")
def get_technical_analysis(profile_name: str, profile_path: str) -> dict[str, Any]:
    """Get real-time technical analysis for USDJPY across all timeframes (H4, M15, M1).

    Returns per-timeframe: regime, RSI value/zone, MACD line/signal/histogram,
    ATR value/state, price info, and a plain-English summary.
    """
    path = _resolve_profile_path(profile_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Profile not found")

    try:
        ta_timeout = float(os.environ.get("API_TA_TIMEOUT_SEC", "12"))
    except ValueError:
        ta_timeout = 12.0

    try:
        return _run_in_threadpool_with_timeout(_compute_technical_analysis_payload, ta_timeout, path)
    except FuturesTimeoutError:
        return {
            "timeframes": {},
            "error": "compute_timeout",
            "detail": "Technical analysis exceeded the server time limit (slow broker or network). Retry in a moment.",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TA error: {str(e)}") from e


# ---------------------------------------------------------------------------
# Endpoints: Advanced Analytics
# ---------------------------------------------------------------------------


def _backfill_mae_mfe(profile, store: SqliteStore, profile_name: str) -> int:
    """Lazily estimate MAE/MFE from M1 candle data for closed trades that have NULL values.

    Returns count of trades backfilled. Writes estimated values with mae_mfe_estimated=1.
    """
    from adapters.broker import get_adapter

    df = store.read_trades_df(profile_name)
    if df.empty or "exit_price" not in df.columns:
        return 0
    closed = df[pd.to_numeric(df["exit_price"], errors="coerce").notna()].copy()
    if closed.empty:
        return 0

    # Find trades missing MAE/MFE
    needs_backfill = closed[
        pd.to_numeric(closed.get("max_adverse_pips"), errors="coerce").isna()
        & pd.to_numeric(closed.get("max_favorable_pips"), errors="coerce").isna()
    ]
    if needs_backfill.empty:
        return 0

    # Fetch M1 candles from broker
    try:
        adapter = get_adapter(profile)
        adapter.initialize()
        adapter.ensure_symbol(profile.symbol)
        m1_df = adapter.get_bars(profile.symbol, "M1", 3000)
        adapter.shutdown()
    except Exception:
        return 0

    if m1_df is None or m1_df.empty:
        return 0

    pip_size = float(profile.pip_size)
    m1_df["time_utc"] = pd.to_datetime(m1_df["time"], utc=True)
    backfilled = 0

    for _, row in needs_backfill.iterrows():
        try:
            entry_ts = pd.to_datetime(row.get("timestamp_utc"), utc=True)
            exit_ts = pd.to_datetime(row.get("exit_timestamp_utc"), utc=True)
            entry_price = float(row["entry_price"])
            side = str(row.get("side") or "").lower()
            if not side or pd.isna(entry_ts) or pd.isna(exit_ts):
                continue

            # Filter M1 bars within trade lifetime
            mask = (m1_df["time_utc"] >= entry_ts) & (m1_df["time_utc"] <= exit_ts)
            trade_bars = m1_df[mask]
            if trade_bars.empty:
                continue

            min_low = float(trade_bars["low"].min())
            max_high = float(trade_bars["high"].max())

            if side == "buy":
                mae = round((min_low - entry_price) / pip_size, 2)
                mfe = round((max_high - entry_price) / pip_size, 2)
            else:
                mae = round((entry_price - max_high) / pip_size, 2)
                mfe = round((entry_price - min_low) / pip_size, 2)

            store.update_trade(str(row["trade_id"]), {
                "max_adverse_pips": mae,
                "max_favorable_pips": mfe,
                "mae_mfe_estimated": 1,
            })
            backfilled += 1
        except Exception:
            continue

    return backfilled


@app.get("/api/data/{profile_name}/advanced-analytics")
def get_advanced_analytics(
    profile_name: str,
    profile_path: Optional[str] = None,
    days_back: int = 365,
) -> dict[str, Any]:
    """Get trade-level data for advanced analytics (MAE/MFE, rolling metrics,
    R-distribution, drawdown, duration). All computation happens on the frontend."""
    cache_key = _cache_compose_key("advanced_analytics", profile_name, profile_path or "", days_back)
    cached_payload = _cache_get("advanced_analytics", cache_key)
    if cached_payload is not None:
        return cached_payload

    profile = None
    if profile_path and _resolve_profile_path(profile_path).exists():
        try:
            profile = load_profile_v1(_resolve_profile_path(profile_path))
        except Exception:
            pass

    curr, rate = ("USD", 1.0)
    if profile:
        curr, rate = _get_display_currency(profile)

    store = _store_for(profile_name)

    # Lazily backfill MAE/MFE from candle data (throttled to once per 60s per profile)
    if profile:
        now_mono = _time.monotonic()
        last_run = _backfill_last_run.get(profile_name, 0)
        if now_mono - last_run >= _BACKFILL_INTERVAL:
            try:
                _backfill_mae_mfe(profile, store, profile_name)
                _backfill_last_run[profile_name] = now_mono
            except Exception:
                pass

    df = store.read_trades_df(profile_name)
    if df.empty or "exit_price" not in df.columns:
        payload = {"trades": [], "display_currency": curr, "source": "database", "starting_balance": None, "total_profit_currency": None}
        _cache_set("advanced_analytics", cache_key, payload)
        return payload

    closed = df[pd.to_numeric(df["exit_price"], errors="coerce").notna()].copy()
    if closed.empty:
        payload = {"trades": [], "display_currency": curr, "source": "database", "starting_balance": None, "total_profit_currency": None}
        _cache_set("advanced_analytics", cache_key, payload)
        return payload

    # Filter by days_back
    if "exit_timestamp_utc" in closed.columns:
        closed["_exit_dt"] = pd.to_datetime(closed["exit_timestamp_utc"], errors="coerce", utc=True)
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days_back)
        closed = closed[closed["_exit_dt"] >= cutoff]

    pip_size = float(profile.pip_size) if profile else 0.01
    starting_balance = None
    if profile and getattr(profile, "deposit_amount", None) is not None and float(profile.deposit_amount) > 0:
        starting_balance = round(float(profile.deposit_amount), 2)

    trades_list = []
    for _, row in closed.iterrows():
        side = str(row.get("side") or "").lower()
        entry_price = float(row["entry_price"]) if pd.notna(row.get("entry_price")) else 0
        exit_price = float(row["exit_price"]) if pd.notna(row.get("exit_price")) else 0
        stop_price = float(row["stop_price"]) if pd.notna(row.get("stop_price")) else None

        # Recompute pips from entry/exit (canonical formula)
        pips_val = None
        if entry_price and exit_price and pip_size:
            if side == "buy":
                pips_val = round((exit_price - entry_price) / pip_size, 2)
            elif side == "sell":
                pips_val = round((entry_price - exit_price) / pip_size, 2)

        # Recompute risk_pips and r_multiple when stop_price present (repairs corrupted DB values)
        risk_pips = None
        r_multiple = None
        if stop_price is not None and entry_price and pip_size > 0:
            risk_pips = abs(entry_price - stop_price) / pip_size
            if risk_pips > 0 and pips_val is not None:
                r_multiple = round(pips_val / risk_pips, 2)
                risk_pips = round(risk_pips, 2)
            elif risk_pips > 0:
                risk_pips = round(risk_pips, 2)

        # Fallback to stored values only when we could not recompute
        if pips_val is None:
            pips_val = float(row["pips"]) if pd.notna(row.get("pips")) else None
        if risk_pips is None:
            risk_pips = float(row["risk_pips"]) if pd.notna(row.get("risk_pips")) else None
        if r_multiple is None:
            r_multiple = float(row["r_multiple"]) if pd.notna(row.get("r_multiple")) else None

        profit_raw = _normalized_trade_profit_usd(
            row,
            symbol_hint=str(getattr(profile, "symbol", "") or ""),
        )
        duration = float(row["duration_minutes"]) if pd.notna(row.get("duration_minutes")) else None

        # Compute duration from timestamps if not stored
        if duration is None:
            try:
                t0 = pd.to_datetime(row.get("timestamp_utc"), utc=True)
                t1 = pd.to_datetime(row.get("exit_timestamp_utc"), utc=True)
                if pd.notna(t0) and pd.notna(t1):
                    duration = round((t1 - t0).total_seconds() / 60.0, 1)
            except Exception:
                pass

        mae = float(row["max_adverse_pips"]) if pd.notna(row.get("max_adverse_pips")) else None
        mfe = float(row["max_favorable_pips"]) if pd.notna(row.get("max_favorable_pips")) else None
        recovery = float(row["post_sl_recovery_pips"]) if pd.notna(row.get("post_sl_recovery_pips")) else None

        trades_list.append({
            "trade_id": str(row.get("trade_id") or ""),
            "side": side,
            "entry_time_utc": str(row.get("timestamp_utc") or ""),
            "exit_time_utc": str(row.get("exit_timestamp_utc") or ""),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pips": pips_val,
            "r_multiple": r_multiple,
            "risk_pips": risk_pips,
            "profit": _convert_amount(profit_raw, rate) if profit_raw is not None else None,
            "duration_minutes": duration,
            "max_adverse_pips": mae,
            "max_favorable_pips": mfe,
            "post_sl_recovery_pips": recovery,
            "preset_name": str(row.get("preset_name") or ""),
            "exit_reason": str(row.get("exit_reason") or ""),
            "entry_type": str(row.get("entry_type") or ""),
            "reversal_risk_tier": str(row.get("reversal_risk_tier") or ""),
            "tier_number": None if pd.isna(row.get("tier_number")) else int(row.get("tier_number")),
        })

    total_profit_currency = None
    profits = [t.get("profit") for t in trades_list if t.get("profit") is not None]
    if profits:
        total_profit_currency = round(sum(profits), 2)

    payload = {
        "trades": _enrich_phase3_rows(trades_list),
        "display_currency": curr,
        "source": "database",
        "starting_balance": starting_balance,
        "total_profit_currency": total_profit_currency,
    }
    _cache_set("advanced_analytics", cache_key, payload)
    return payload


# ---------------------------------------------------------------------------
# Endpoints: Trade management (close, sync)
# ---------------------------------------------------------------------------


def _open_trades_sync_and_broker_live(loaded_profile: ProfileV1, store: SqliteStore) -> dict[int, dict[str, Any]]:
    """DB sync + live unrealized/financing from broker; may block on OANDA."""
    _sync_open_trades_with_broker(loaded_profile, store)
    broker_live: dict[int, dict[str, Any]] = {}
    from adapters.broker import get_adapter

    adapter = get_adapter(loaded_profile)
    try:
        adapter.initialize()
        live_positions = adapter.get_open_positions(loaded_profile.symbol)
        for pos in live_positions:
            if isinstance(pos, dict):
                pos_id = pos.get("id")
                if pos_id is not None:
                    try:
                        broker_live[int(pos_id)] = {
                            "unrealized_pl": float(pos.get("unrealizedPL") or 0),
                            "financing": float(pos.get("financing") or 0),
                        }
                    except (TypeError, ValueError):
                        pass
    finally:
        try:
            adapter.shutdown()
        except Exception:
            pass
    return broker_live


@app.get("/api/data/{profile_name}/open-trades")
def get_open_trades(profile_name: str, profile_path: Optional[str] = None, sync: bool = True) -> list[dict[str, Any]]:
    """Get open trades (trades without exit_price).

    Syncs with broker first (default) to detect any trades closed externally (e.g. on OANDA).
    Ensures displayed open positions and max_open_trades logic match the broker.
    Also includes live unrealized_pl and financing from broker.
    """
    store = _store_for(profile_name)

    broker_live: dict[int, dict] = {}

    if profile_path and sync:
        loaded_profile = None
        try:
            loaded_profile = load_profile_v1(profile_path)
        except Exception as e:
            print(f"[api] load profile error in open-trades: {e}")

        if loaded_profile is not None:
            try:
                try:
                    sync_timeout = float(os.environ.get("API_OPEN_TRADES_SYNC_TIMEOUT_SEC", "8"))
                except ValueError:
                    sync_timeout = 8.0
                try:
                    broker_live = _run_in_threadpool_with_timeout(
                        _open_trades_sync_and_broker_live, sync_timeout, loaded_profile, store
                    )
                except FuturesTimeoutError:
                    print(f"[api] open-trades broker sync timed out after {sync_timeout}s")
                    broker_live = {}
            except Exception as e:
                print(f"[api] sync error in open-trades: {e}")

    rows = store.list_open_trades(profile_name)
    result = []
    for row in rows:
        d = dict(row)
        pos_id = d.get("mt5_position_id")
        if pos_id is not None and broker_live:
            try:
                live = broker_live.get(int(pos_id))
                if live:
                    d.update(live)
            except (TypeError, ValueError):
                pass
        result.append(d)

    # When we have live broker data, return only trades that are still open on the broker.
    # This keeps open position count and max_open_trades in sync with OANDA even if DB sync failed.
    if broker_live:
        broker_ids = set(broker_live.keys())
        filtered = []
        for r in result:
            try:
                pid = r.get("mt5_position_id")
                if pid is not None and int(pid) in broker_ids:
                    filtered.append(r)
            except (TypeError, ValueError):
                pass
        result = filtered
    return _enrich_phase3_rows(result)


def _sync_open_trades_with_broker(profile: ProfileV1, store: SqliteStore) -> int:
    """Quick sync: check if any DB open trades are closed on broker."""
    from adapters.broker import get_adapter

    open_trades = store.list_open_trades(profile.profile_name)
    if not open_trades:
        return 0

    try:
        adapter = get_adapter(profile)
        adapter.initialize()
        adapter.ensure_symbol(profile.symbol)
    except Exception as e:
        print(f"[api] broker init failed for sync: {e}")
        return 0

    try:
        broker_positions = adapter.get_open_positions(profile.symbol)
    except Exception as e:
        print(f"[api] get_open_positions failed: {e}")
        try:
            adapter.shutdown()
        except Exception:
            pass
        return 0

    # Build set of broker's open position IDs
    broker_open_ids = set()
    for pos in broker_positions:
        pos_id = pos.get("id") if isinstance(pos, dict) else getattr(pos, "ticket", None)
        if pos_id is not None:
            try:
                broker_open_ids.add(int(pos_id))
            except (TypeError, ValueError):
                pass

    pip_size = float(profile.pip_size)
    synced = 0

    # Log for debugging
    print(f"[api] _sync_open_trades_with_broker: {len(open_trades)} DB trades, {len(broker_open_ids)} broker positions")
    print(f"[api] broker position IDs: {broker_open_ids}")

    for trade_row in open_trades:
        trade_row = dict(trade_row)
        mt5_position_id = trade_row.get("mt5_position_id")
        trade_id = str(trade_row["trade_id"])

        # Check if this trade is on broker
        on_broker = False
        if mt5_position_id is not None:
            try:
                position_id = int(mt5_position_id)
                if position_id in broker_open_ids:
                    on_broker = True
            except (TypeError, ValueError):
                pass

        if on_broker:
            # Still open on broker
            continue

        # Trade is NOT on broker (or has no position ID but broker has fewer positions)
        # If trade has no position ID and we have more DB trades than broker positions, close it
        if mt5_position_id is None:
            if len(open_trades) <= len(broker_open_ids):
                # Can't determine if it's closed without position ID
                print(f"[api] sync: skipping trade {trade_id} - no position ID")
                continue
            print(f"[api] sync: closing trade {trade_id} - no position ID and DB > broker count")

        # Trade was closed on broker - update our DB
        entry_price = float(trade_row["entry_price"])
        side = str(trade_row["side"]).lower()
        target_price = trade_row.get("target_price")
        stop_price = trade_row.get("stop_price")

        # Try to get close info from broker if we have position ID
        exit_price = None
        exit_time = None
        profit = None
        if mt5_position_id is not None:
            try:
                close_info = adapter.get_position_close_info(int(mt5_position_id))
                if close_info:
                    exit_price = close_info.exit_price
                    exit_time = close_info.exit_time_utc
                    profit = close_info.profit
            except Exception:
                pass

        if exit_price is None:
            # Use approximate values
            exit_price = entry_price  # Will be corrected on next full sync
            exit_time = pd.Timestamp.now(tz="UTC").isoformat()

        # Calculate pips
        if side == "buy":
            pips = (exit_price - entry_price) / pip_size
        else:
            pips = (entry_price - exit_price) / pip_size

        # Determine exit reason (BE before original SL)
        be_sl = trade_row.get("breakeven_sl_price")
        be_applied = int(trade_row.get("breakeven_applied") or 0)
        exit_reason = "broker_closed"
        if target_price and abs(exit_price - float(target_price)) <= pip_size * 2:
            exit_reason = "hit_take_profit"
        elif be_applied and be_sl and abs(exit_price - float(be_sl)) <= pip_size * 3:
            exit_reason = "hit_breakeven"
        elif stop_price and abs(exit_price - float(stop_price)) <= pip_size * 2:
            exit_reason = "hit_stop_loss"

        # Calculate R-multiple
        risk_pips = None
        r_multiple = None
        if stop_price:
            risk_pips = abs(entry_price - float(stop_price)) / pip_size
            if risk_pips > 0:
                r_multiple = pips / risk_pips

        updates = {
            "exit_price": exit_price,
            "exit_timestamp_utc": exit_time,
            "exit_reason": exit_reason,
            "pips": pips,
            "risk_pips": risk_pips,
            "r_multiple": r_multiple,
        }
        if profit is not None:
            updates["profit"] = profit

        # Compute post-SL recovery pips for SL/BE closes
        if exit_reason in ("hit_stop_loss", "hit_breakeven") and exit_time:
            try:
                from core.trade_sync import compute_post_sl_recovery_pips
                recovery = compute_post_sl_recovery_pips(
                    adapter, profile.symbol, side, exit_price, exit_time, pip_size
                )
                if recovery is not None:
                    updates["post_sl_recovery_pips"] = recovery
            except Exception as _e:
                print(f"[api] post_sl_recovery failed for {trade_id}: {_e}")

        store.close_trade(trade_id=trade_id, updates=updates)
        _apply_phase3_sync_close_state_update(
            profile_name=profile.profile_name,
            trade_row=trade_row,
            updates=updates,
        )
        print(f"[api] synced closed trade {trade_id}: {exit_reason}, pips={pips:.2f}")
        synced += 1

    try:
        adapter.shutdown()
    except Exception:
        pass

    return synced


@app.post("/api/trades/{profile_name}/{trade_id}/close")
def close_trade_endpoint(profile_name: str, trade_id: str, profile_path: str) -> dict[str, Any]:
    """Close an open trade via broker (MT5/OANDA) and update the database.
    
    This endpoint:
    1. Looks up the trade in the database to get position ticket, symbol, side, volume
    2. Calls broker to close the position
    3. Updates the database with exit details
    """
    from datetime import datetime, timezone
    
    store = _store_for(profile_name)
    
    # Find the trade in database
    df = store.read_trades_df(profile_name)
    if df.empty:
        raise HTTPException(status_code=404, detail="No trades found")
    
    trade_row = df[df["trade_id"] == trade_id]
    if trade_row.empty:
        raise HTTPException(status_code=404, detail=f"Trade {trade_id} not found")
    
    trade = trade_row.iloc[0]
    
    # Check if already closed
    if pd.notna(trade.get("exit_price")):
        raise HTTPException(status_code=400, detail="Trade is already closed")
    
    # Get trade details
    mt5_position_id = trade.get("mt5_position_id")
    mt5_order_id = trade.get("mt5_order_id")
    mt5_deal_id = trade.get("mt5_deal_id")
    symbol = trade.get("symbol")
    side = str(trade.get("side", "")).lower()
    size_lots = trade.get("size_lots")
    entry_price = trade.get("entry_price")
    stop_price = trade.get("stop_price")
    
    # Load profile for pip_size
    path = _resolve_profile_path(profile_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Profile not found")
    profile = load_profile_v1(path)
    pip_size = profile.pip_size
    
    try:
        from adapters.broker import get_adapter
        adapter = get_adapter(profile)
        adapter.initialize()
        adapter.ensure_symbol(symbol)
        
        # Resolve position ticket - prefer position_id, fallback to lookup
        position_ticket = None
        
        if mt5_position_id and pd.notna(mt5_position_id):
            position_ticket = int(mt5_position_id)
        elif mt5_deal_id and pd.notna(mt5_deal_id):
            position_ticket = adapter.get_position_id_from_deal(int(mt5_deal_id))
        elif mt5_order_id and pd.notna(mt5_order_id):
            position_ticket = adapter.get_position_id_from_order(int(mt5_order_id))
        
        if not position_ticket:
            raise HTTPException(status_code=400, detail="Trade has no valid position ID - cannot close via broker")
        
        # Update position_id in DB if we just resolved it
        if not mt5_position_id or pd.isna(mt5_position_id):
            store.update_trade(trade_id, {"mt5_position_id": position_ticket})
        
        # Get current position info
        position = adapter.get_position_by_ticket(position_ticket)
        profit_value = None
        if position is None:
            # Position might already be closed - try to get exit info from history
            close_info = adapter.get_position_close_info(position_ticket)
            if close_info:
                # Update database with historical close info
                exit_price = close_info.exit_price
                exit_ts = close_info.exit_time_utc
                profit_value = close_info.profit
            else:
                raise HTTPException(status_code=400, detail="Position not found - may already be closed")
        else:
            # Close the position
            position_type = int(getattr(position, "type", 0))
            volume = float(getattr(position, "volume", size_lots or 0.1))
            
            result = adapter.close_position(
                ticket=position_ticket,
                symbol=symbol,
                volume=volume,
                position_type=position_type,
                comment=f"UI_close_{trade_id}",
            )
            
            # MT5: 10009 = TRADE_RETCODE_DONE; OANDA: 0 = success
            if result.retcode not in (10009, 0):
                raise HTTPException(
                    status_code=500,
                    detail=f"Close failed: {result.retcode} - {result.comment}"
                )
            
            # Get exit price and profit from broker
            tick = adapter.get_tick(symbol)
            exit_price = tick.bid if side == "buy" else tick.ask
            exit_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
            if result.deal:
                profit_value = adapter.get_deal_profit(result.deal)
        
        # Calculate pips and R-multiple
        if side == "buy":
            pips = (exit_price - entry_price) / pip_size
        else:
            pips = (entry_price - exit_price) / pip_size
        
        risk_pips = None
        r_multiple = None
        if stop_price and pd.notna(stop_price):
            risk_pips = abs(entry_price - stop_price) / pip_size
            if risk_pips > 0:
                r_multiple = pips / risk_pips
        
        # Calculate duration
        entry_ts = trade.get("timestamp_utc")
        duration_minutes = None
        if entry_ts:
            try:
                t0 = pd.to_datetime(entry_ts, utc=True)
                t1 = pd.to_datetime(exit_ts, utc=True)
                duration_minutes = float((t1 - t0).total_seconds() / 60.0)
            except Exception:
                pass
        
        # Update database
        updates = {
            "exit_price": float(exit_price),
            "exit_timestamp_utc": exit_ts,
            "exit_reason": "UI_manual_close",
            "pips": float(pips),
            "risk_pips": float(risk_pips) if risk_pips else None,
            "r_multiple": float(r_multiple) if r_multiple else None,
            "duration_minutes": float(duration_minutes) if duration_minutes else None,
        }
        if profit_value is not None:
            updates["profit"] = float(profit_value)
        store.close_trade(trade_id=trade_id, updates=updates)
        _apply_phase3_sync_close_state_update(
            profile_name=profile.profile_name,
            trade_row=trade.to_dict(),
            updates=updates,
        )
        _cache_invalidate_profile(profile_name)
        
        return {
            "status": "closed",
            "trade_id": trade_id,
            "exit_price": exit_price,
            "pips": round(pips, 2),
            "r_multiple": round(r_multiple, 2) if r_multiple else None,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error closing trade: {str(e)}")


@app.post("/api/data/{profile_name}/sync-trades")
def sync_trades_endpoint(
    profile_name: str,
    profile_path: str,
    force_profit_refresh: bool = True,
) -> dict[str, Any]:
    """Sync trades with MT5 - detect externally closed trades and update database.

    Also backfills position_ids and imports manual trades from MT5 history.
    Set force_profit_refresh=true to recompute profit from MT5 for all trades (fixes
    incorrect win rate when profit was from partial close only).
    """
    from core.trade_sync import sync_closed_trades, import_mt5_history, backfill_position_ids, backfill_profit

    path = _resolve_profile_path(profile_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Profile not found")

    try:
        profile = load_profile_v1(path)
        store = _store_for(profile_name)

        # First: aggressive sync - close any DB trades not on broker
        aggressive_synced = _aggressive_sync_with_broker(profile, store)

        # First backfill position_ids for existing trades
        backfilled_count = backfill_position_ids(profile, store)

        # Sync closed trades (detect externally closed)
        synced_count = sync_closed_trades(profile, store)

        # Import manual trades from broker history (MT5 or OANDA activity/transactions)
        imported_count = import_mt5_history(profile, store, days_back=90)

        # Backfill profit (force_refresh=True recomputes all to fix partial-close bug)
        profit_backfilled = backfill_profit(profile, store, force_refresh=force_profit_refresh)
        _cache_invalidate_profile(profile_name)

        return {
            "status": "synced",
            "trades_updated": synced_count + aggressive_synced,
            "trades_imported": imported_count,
            "position_ids_backfilled": backfilled_count,
            "profit_backfilled": profit_backfilled,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sync error: {str(e)}")


def _aggressive_sync_with_broker(profile: ProfileV1, store: SqliteStore) -> int:
    """Aggressively sync: close ANY DB open trade that is not on broker.

    This handles cases where mt5_position_id doesn't match or is missing.
    It compares by checking if there are MORE open trades in DB than on broker,
    and closes the excess ones that have no matching position.
    """
    from adapters.broker import get_adapter

    open_trades = list(store.list_open_trades(profile.profile_name))
    if not open_trades:
        return 0

    try:
        adapter = get_adapter(profile)
        adapter.initialize()
        adapter.ensure_symbol(profile.symbol)
        broker_positions = adapter.get_open_positions(profile.symbol)
    except Exception as e:
        print(f"[api] aggressive sync - broker error: {e}")
        return 0

    # Get all broker position IDs
    broker_ids = set()
    for pos in broker_positions:
        pos_id = pos.get("id") if isinstance(pos, dict) else getattr(pos, "ticket", None)
        if pos_id is not None:
            try:
                broker_ids.add(int(pos_id))
            except (TypeError, ValueError):
                pass

    print(f"[api] aggressive sync: DB has {len(open_trades)} open trades, broker has {len(broker_ids)} positions")
    print(f"[api] aggressive sync: broker position IDs: {broker_ids}")

    pip_size = float(profile.pip_size)
    synced = 0

    for trade_row in open_trades:
        trade_row = dict(trade_row)
        trade_id = str(trade_row["trade_id"])
        mt5_position_id = trade_row.get("mt5_position_id")

        # Check if this trade's position is on broker
        position_on_broker = False
        if mt5_position_id is not None:
            try:
                if int(mt5_position_id) in broker_ids:
                    position_on_broker = True
            except (TypeError, ValueError):
                pass

        if position_on_broker:
            print(f"[api] aggressive sync: trade {trade_id} (pos {mt5_position_id}) still on broker")
            continue

        # Trade is NOT on broker - close it in DB
        print(f"[api] aggressive sync: closing trade {trade_id} (pos {mt5_position_id}) - not on broker")

        entry_price = float(trade_row["entry_price"])
        side = str(trade_row["side"]).lower()
        target_price = trade_row.get("target_price")
        stop_price = trade_row.get("stop_price")

        # Try to get close info from broker if we have position ID
        exit_price = None
        exit_time = None
        profit = None
        if mt5_position_id is not None:
            try:
                close_info = adapter.get_position_close_info(int(mt5_position_id))
                if close_info:
                    exit_price = close_info.exit_price
                    exit_time = close_info.exit_time_utc
                    profit = close_info.profit
                    print(f"[api] aggressive sync: got close info for {trade_id}: exit={exit_price}, profit={profit}")
            except Exception as e:
                print(f"[api] aggressive sync: no close info for {trade_id}: {e}")

        if exit_price is None:
            # Use entry price as fallback (unknown exit)
            exit_price = entry_price
            exit_time = pd.Timestamp.now(tz="UTC").isoformat()

        # Calculate pips
        if side == "buy":
            pips = (exit_price - entry_price) / pip_size
        else:
            pips = (entry_price - exit_price) / pip_size

        # Determine exit reason (BE before original SL)
        be_sl = trade_row.get("breakeven_sl_price")
        be_applied = int(trade_row.get("breakeven_applied") or 0)
        exit_reason = "broker_closed_sync"
        if target_price and abs(exit_price - float(target_price)) <= pip_size * 2:
            exit_reason = "hit_take_profit"
        elif be_applied and be_sl and abs(exit_price - float(be_sl)) <= pip_size * 3:
            exit_reason = "hit_breakeven"
        elif stop_price and abs(exit_price - float(stop_price)) <= pip_size * 2:
            exit_reason = "hit_stop_loss"

        # Calculate R-multiple
        risk_pips = None
        r_multiple = None
        if stop_price:
            risk_pips = abs(entry_price - float(stop_price)) / pip_size
            if risk_pips > 0:
                r_multiple = pips / risk_pips

        updates: dict[str, Any] = {
            "exit_price": exit_price,
            "exit_timestamp_utc": exit_time,
            "exit_reason": exit_reason,
            "pips": pips,
            "risk_pips": risk_pips,
            "r_multiple": r_multiple,
        }
        if profit is not None:
            updates["profit"] = profit

        # Compute post-SL recovery pips for SL/BE closes
        if exit_reason in ("hit_stop_loss", "hit_breakeven") and exit_time:
            try:
                from core.trade_sync import compute_post_sl_recovery_pips
                recovery = compute_post_sl_recovery_pips(
                    adapter, profile.symbol, side, exit_price, exit_time, pip_size
                )
                if recovery is not None:
                    updates["post_sl_recovery_pips"] = recovery
            except Exception as _e:
                print(f"[api] post_sl_recovery failed for {trade_id}: {_e}")

        store.close_trade(trade_id=trade_id, updates=updates)
        _apply_phase3_sync_close_state_update(
            profile_name=profile.profile_name,
            trade_row=trade_row,
            updates=updates,
        )
        print(f"[api] aggressive sync: closed {trade_id} with exit_reason={exit_reason}, pips={pips:.2f}")
        synced += 1

    try:
        adapter.shutdown()
    except Exception:
        pass

    return synced


# ---------------------------------------------------------------------------
# Backfill exit analytics
# ---------------------------------------------------------------------------


@app.post("/api/data/{profile_name}/backfill-exit-analytics")
def backfill_exit_analytics(profile_name: str, profile_path: str) -> dict[str, Any]:
    """Backfill exit_reason and post_sl_recovery_pips for historical trades.

    Pass 1: Fix exit_reason for trades where breakeven was applied but reason
            was recorded as broker_closed / broker_closed_sync / hit_stop_loss.
    Pass 2: Compute post_sl_recovery_pips for SL/BE closes that are missing it.
    """
    from core.trade_sync import compute_post_sl_recovery_pips

    path = _resolve_profile_path(profile_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Profile not found")
    profile = load_profile_v1(path)
    store = _store_for(profile_name)
    pip_size = float(profile.pip_size)

    be_fixed = 0
    recovery_computed = 0
    recovery_failed = 0

    with store.connect() as conn:
        # --- Pass 1: fix exit_reason for BE trades ---
        rows = conn.execute(
            """SELECT trade_id, exit_price, stop_price, target_price,
                      side, breakeven_sl_price, breakeven_applied
               FROM trades
               WHERE exit_reason IN ('broker_closed','broker_closed_sync','hit_stop_loss')
                 AND breakeven_applied = 1
                 AND breakeven_sl_price IS NOT NULL
                 AND exit_price IS NOT NULL
                 AND profile = ?""",
            [profile_name],
        ).fetchall()

        for row in rows:
            exit_price = float(row["exit_price"])
            be_sl = float(row["breakeven_sl_price"])
            if abs(exit_price - be_sl) <= pip_size * 3:
                conn.execute(
                    "UPDATE trades SET exit_reason='hit_breakeven' WHERE trade_id=?",
                    [row["trade_id"]],
                )
                be_fixed += 1

        conn.commit()

        # --- Pass 2: compute post-SL recovery ---
        sl_rows = conn.execute(
            """SELECT trade_id, side, exit_price, exit_timestamp_utc
               FROM trades
               WHERE exit_reason IN ('hit_stop_loss','hit_breakeven')
                 AND post_sl_recovery_pips IS NULL
                 AND exit_timestamp_utc IS NOT NULL
                 AND exit_price IS NOT NULL
                 AND profile = ?""",
            [profile_name],
        ).fetchall()

    if sl_rows:
        try:
            from adapters.broker import get_adapter
            adapter = get_adapter(profile)
            adapter.initialize()
            adapter.ensure_symbol(profile.symbol)

            for row in sl_rows:
                recovery = compute_post_sl_recovery_pips(
                    adapter,
                    profile.symbol,
                    str(row["side"]),
                    float(row["exit_price"]),
                    str(row["exit_timestamp_utc"]),
                    pip_size,
                )
                if recovery is not None:
                    store.update_trade(row["trade_id"], {"post_sl_recovery_pips": recovery})
                    recovery_computed += 1
                else:
                    recovery_failed += 1

            try:
                adapter.shutdown()
            except Exception:
                pass
        except Exception as e:
            print(f"[api] backfill-exit-analytics recovery pass failed: {e}")
            recovery_failed += len(sl_rows) - recovery_computed

    print(f"[api] backfill-exit-analytics: be_fixed={be_fixed}, recovery_computed={recovery_computed}, recovery_failed={recovery_failed}")
    return {"be_fixed": be_fixed, "recovery_computed": recovery_computed, "recovery_failed": recovery_failed}


# ---------------------------------------------------------------------------
# Serve frontend (if built)
# ---------------------------------------------------------------------------


@app.get("/")
def serve_frontend_root():
    """Serve frontend index.html."""
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    # Fallback: show a simple HTML page with links
    return {
        "message": "USDJPY Assistant API is running!",
        "frontend": "Not built. Run BUILD_FRONTEND.bat or use the Streamlit UI (RUN_UI.bat).",
        "api_docs": "http://127.0.0.1:8000/docs",
        "endpoints": {
            "profiles": "/api/profiles",
            "presets": "/api/presets",
            "health": "/api/health",
        },
    }


# Mount static files if frontend exists
if FRONTEND_DIR.exists():
    assets_dir = FRONTEND_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Dashboard endpoints
# ---------------------------------------------------------------------------


_dashboard_live_cache: dict[str, tuple[float, dict]] = {}
_DASHBOARD_LIVE_TTL = 2.0  # seconds

# Lean-mode dashboard: full response cache (legacy mode already used _dashboard_live_cache).
_lean_dashboard_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_LEAN_DASHBOARD_CACHE_TTL = float(os.environ.get("LEAN_DASHBOARD_CACHE_TTL", "4.0"))

# Cached adapters for dashboard use — avoids init/shutdown every poll
_dashboard_adapters: dict[str, tuple[Any, float]] = {}  # {key: (adapter, init_time)}
_DASHBOARD_ADAPTER_TTL = 300.0  # re-init every 5 min


def _get_dashboard_adapter(profile):
    """Get or create a cached adapter for dashboard use."""
    key = getattr(profile, "profile_name", "") or str(id(profile))
    now = _time.monotonic()
    cached = _dashboard_adapters.get(key)
    if cached:
        adapter, init_time = cached
        if now - init_time < _DASHBOARD_ADAPTER_TTL:
            return adapter
        try:
            adapter.shutdown()
        except Exception:
            pass
    from adapters.broker import get_adapter
    adapter = get_adapter(profile)
    adapter.initialize()
    _dashboard_adapters[key] = (adapter, now)
    return adapter


def _fetch_live_positions(profile_name: str, profile_path: Optional[str] = None) -> list[dict[str, Any]]:
    """Fetch open positions from the broker for dashboard display. Returns [] on any failure."""
    resolved_path: Optional[Path] = None
    if profile_path:
        try:
            p = _resolve_profile_path(profile_path)
            if p.exists():
                resolved_path = p
        except Exception:
            pass
    if resolved_path is None:
        for p in _list_profile_paths():
            if p.stem == profile_name:
                resolved_path = p
                break
    if resolved_path is None:
        return []
    try:
        profile = load_profile_v1(str(resolved_path))
    except Exception:
        return []
    from datetime import datetime, timezone
    import pandas as _pd_dash
    now_utc = datetime.now(timezone.utc)
    pip_size = float(profile.pip_size)
    positions: list[dict] = []
    try:
        adapter = _get_dashboard_adapter(profile)
        tick = adapter.get_tick(profile.symbol)
        mid = (tick.bid + tick.ask) / 2.0
        for t in adapter.get_open_positions(profile.symbol):
            units = float(t.get("currentUnits", 0) or 0)
            s = "buy" if units > 0 else "sell"
            size_lots = abs(units) / 100_000.0 if units else 0.0
            entry = float(t.get("price", 0) or 0)
            unrealized = (mid - entry) / pip_size if s == "buy" else (entry - mid) / pip_size
            age = 0.0
            try:
                t0 = _pd_dash.to_datetime(t.get("openTime"), utc=True)
                age = (now_utc - t0.to_pydatetime(warn=False)).total_seconds() / 60.0
            except Exception:
                pass
            sl = tp = None
            try:
                if t.get("stopLossOrder"):
                    sl = float(t["stopLossOrder"]["price"])
            except Exception:
                pass
            try:
                if t.get("takeProfitOrder"):
                    tp = float(t["takeProfitOrder"]["price"])
            except Exception:
                pass
            positions.append({
                "trade_id": str(t.get("id", "")),
                "side": s,
                "entry_price": entry,
                "size_lots": round(size_lots, 4),
                "entry_type": None,
                "current_price": mid,
                "unrealized_pips": round(unrealized, 1),
                "age_minutes": round(age, 1),
                "stop_price": sl,
                "target_price": tp,
                "breakeven_applied": False,
            })
    except Exception as e:
        print(f"[api] _fetch_live_positions failed for '{profile_name}': {e}")
        return []
    return positions


def _build_live_dashboard_state(profile_name: str, profile_path: Optional[str] = None, log_dir: Optional[Path] = None) -> dict[str, Any]:
    """Build a live dashboard state directly from the broker (no run-loop file required)."""
    from datetime import datetime, timezone

    # Find the profile JSON
    resolved_path: Optional[Path] = None
    if profile_path:
        try:
            p = _resolve_profile_path(profile_path)
            if p.exists():
                resolved_path = p
        except Exception:
            pass
    if resolved_path is None:
        for p in _list_profile_paths():
            if p.stem == profile_name:
                resolved_path = p
                break
    if resolved_path is None:
        return {"error": "no_dashboard_data", "timestamp_utc": None}

    try:
        profile = load_profile_v1(str(resolved_path))
    except Exception as e:
        print(f"[api] dashboard: failed to load profile '{profile_name}': {e}")
        return {"error": "no_dashboard_data", "timestamp_utc": None}

    active_profile_name = log_dir.name if log_dir is not None else str(getattr(profile, "profile_name", profile_name) or profile_name)

    runtime = load_state(_runtime_state_path(profile_name))
    now_utc = datetime.now(timezone.utc)
    pip_size = float(profile.pip_size)

    bid = ask = spread_pips = 0.0
    positions: list[dict] = []
    filters: list[dict] = []
    context: list[dict] = []
    tick = None

    try:
        import pandas as _pd_dash
        adapter = _get_dashboard_adapter(profile)
        tick = adapter.get_tick(profile.symbol)
        bid, ask = tick.bid, tick.ask
        spread_pips = (ask - bid) / pip_size
        mid = (bid + ask) / 2.0

        for t in adapter.get_open_positions(profile.symbol):
            units = float(t.get("currentUnits", 0) or 0)
            s = "buy" if units > 0 else "sell"
            size_lots = abs(units) / 100_000.0 if units else 0.0
            entry = float(t.get("price", 0) or 0)
            unrealized = (mid - entry) / pip_size if s == "buy" else (entry - mid) / pip_size
            age = 0.0
            try:
                t0 = _pd_dash.to_datetime(t.get("openTime"), utc=True)
                age = (now_utc - t0.to_pydatetime(warn=False)).total_seconds() / 60.0
            except Exception:
                pass
            sl = tp = None
            try:
                if t.get("stopLossOrder"):
                    sl = float(t["stopLossOrder"]["price"])
            except Exception:
                pass
            try:
                if t.get("takeProfitOrder"):
                    tp = float(t["takeProfitOrder"]["price"])
            except Exception:
                pass
            positions.append({
                "trade_id": str(t.get("id", "")),
                "side": s,
                "entry_price": entry,
                "size_lots": round(size_lots, 4),
                "entry_type": None,
                "current_price": mid,
                "unrealized_pips": round(unrealized, 1),
                "age_minutes": round(age, 1),
                "stop_price": sl,
                "target_price": tp,
                "breakeven_applied": False,
            })
    except Exception as e:
        print(f"[api] dashboard build error for '{profile_name}': {e}")

    # --- Filters: same logic as run loop (shared build_dashboard_filters) ---
    try:
        from dataclasses import asdict
        from core.dashboard_builder import build_dashboard_filters

        # Need tick for filter build (may already have from block above)
        _tick = tick
        if _tick is None:
            try:
                _adapter = _get_dashboard_adapter(profile)
                _tick = _adapter.get_tick(profile.symbol)
            except Exception:
                _tick = None
        if _tick is not None:
            # First enabled KT/CG Trial policy for rich filters
            _policy = None
            _policy_type = ""
            enabled_policies = [
                pol for pol in (getattr(profile.execution, "policies", []) or [])
                if getattr(pol, "enabled", True)
            ]
            preferred_order = (
                "phase3_integrated",
                "kt_cg_trial_10",
                "kt_cg_trial_9",
                "kt_cg_trial_8",
                "kt_cg_trial_7",
                "kt_cg_trial_6",
                "kt_cg_trial_5",
                "kt_cg_trial_4",
            )
            for preferred_type in preferred_order:
                match = next((pol for pol in enabled_policies if (getattr(pol, "type", "") or "") == preferred_type), None)
                if match is not None:
                    _policy = match
                    _policy_type = preferred_type
                    break
            data_by_tf: dict = {}
            daily_reset_state: Optional[dict] = None
            exhaustion_state: Optional[dict] = None
            divergence_state: Optional[dict] = None
            exhaustion_result: Optional[dict] = None
            temp_overrides_api: Optional[dict] = None
            phase3_state_for_filters: Optional[dict] = None
            ntz_filter_snapshot: Optional[dict] = None
            if _policy is not None:
                try:
                    _adapter = _get_dashboard_adapter(profile)
                    data_by_tf["M1"] = _get_bars_cached(_adapter, profile.symbol, "M1", 3000)
                    if _policy_type in ("kt_cg_trial_4", "kt_cg_trial_5", "kt_cg_trial_6"):
                        data_by_tf["M3"] = _get_bars_cached(_adapter, profile.symbol, "M3", 3000)
                    if _policy_type in ("kt_cg_trial_4", "kt_cg_trial_5", "kt_cg_trial_8", "kt_cg_trial_9", "kt_cg_trial_10"):
                        data_by_tf["D"] = _get_bars_cached(_adapter, profile.symbol, "D", 3)
                    if _policy_type in ("kt_cg_trial_7", "kt_cg_trial_8", "kt_cg_trial_9", "kt_cg_trial_10", "phase3_integrated"):
                        data_by_tf["M5"] = _get_bars_cached(_adapter, profile.symbol, "M5", 2000)
                    if _policy_type == "phase3_integrated":
                        data_by_tf["M15"] = _get_bars_cached(_adapter, profile.symbol, "M15", 2000)
                        data_by_tf["D"] = _get_bars_cached(_adapter, profile.symbol, "D", 5)
                        data_by_tf["H1"] = _get_bars_cached(_adapter, profile.symbol, "H1", 200)
                    if _policy_type in ("kt_cg_trial_9", "kt_cg_trial_10"):
                        data_by_tf["M15"] = _get_bars_cached(_adapter, profile.symbol, "M15", 2000)
                        data_by_tf["W"] = _get_bars_cached(_adapter, profile.symbol, "W", 2)
                        data_by_tf["MN"] = _get_bars_cached(_adapter, profile.symbol, "MN", 2)
                except Exception:
                    pass
                try:
                    _state = load_state(_runtime_state_path(profile_name))
                    daily_reset_state = {
                        "daily_reset_date": _state.daily_reset_date,
                        "daily_reset_high": _state.daily_reset_high,
                        "daily_reset_low": _state.daily_reset_low,
                        "daily_reset_block_active": _state.daily_reset_block_active,
                        "daily_reset_settled": _state.daily_reset_settled,
                    }
                    exhaustion_state = {
                        "trend_flip_price": _state.trend_flip_price,
                        "trend_flip_direction": _state.trend_flip_direction,
                        "trend_flip_time": _state.trend_flip_time,
                    }
                    divergence_state = {}
                    if _state.divergence_block_buy_until:
                        divergence_state["block_buy_until"] = _state.divergence_block_buy_until
                    if _state.divergence_block_sell_until:
                        divergence_state["block_sell_until"] = _state.divergence_block_sell_until
                    # Apply Temporary Settings: same mapping as run loop (runtime_state temp_* -> policy attr names)
                    temp_overrides_api = {}
                    if _state.temp_m3_trend_ema_fast is not None:
                        temp_overrides_api["m3_trend_ema_fast"] = _state.temp_m3_trend_ema_fast
                    if _state.temp_m3_trend_ema_slow is not None:
                        temp_overrides_api["m3_trend_ema_slow"] = _state.temp_m3_trend_ema_slow
                    if _state.temp_m1_t4_zone_entry_ema_fast is not None:
                        temp_overrides_api["m1_zone_entry_ema_fast"] = _state.temp_m1_t4_zone_entry_ema_fast
                    if _state.temp_m1_t4_zone_entry_ema_slow is not None:
                        temp_overrides_api["m1_zone_entry_ema_slow"] = _state.temp_m1_t4_zone_entry_ema_slow
                    if _state.temp_m5_trend_ema_fast is not None:
                        temp_overrides_api["m5_trend_ema_fast"] = _state.temp_m5_trend_ema_fast
                    if _state.temp_m5_trend_ema_slow is not None:
                        temp_overrides_api["m5_trend_ema_slow"] = _state.temp_m5_trend_ema_slow
                    if _state.temp_m5_trend_source is not None:
                        temp_overrides_api["m5_trend_source"] = _state.temp_m5_trend_source
                    if _state.temp_m1_zone_entry_ema_slow is not None:
                        temp_overrides_api["m1_zone_entry_ema_slow"] = _state.temp_m1_zone_entry_ema_slow
                    if _state.temp_m1_pullback_cross_ema_slow is not None:
                        temp_overrides_api["m1_pullback_cross_ema_slow"] = _state.temp_m1_pullback_cross_ema_slow
                    if _policy_type == "kt_cg_trial_9":
                        if _state.temp_t9_exit_strategy is not None:
                            temp_overrides_api["exit_strategy"] = _state.temp_t9_exit_strategy
                        if _state.temp_t9_hwm_trail_pips is not None:
                            temp_overrides_api["hwm_trail_pips"] = _state.temp_t9_hwm_trail_pips
                        if _state.temp_t9_tp1_pips is not None:
                            temp_overrides_api["tp1_pips"] = _state.temp_t9_tp1_pips
                        if _state.temp_t9_tp1_close_pct is not None:
                            temp_overrides_api["tp1_close_pct"] = _state.temp_t9_tp1_close_pct
                        if _state.temp_t9_be_spread_plus_pips is not None:
                            temp_overrides_api["be_spread_plus_pips"] = _state.temp_t9_be_spread_plus_pips
                        if _state.temp_t9_trail_ema_period is not None:
                            temp_overrides_api["trail_ema_period"] = _state.temp_t9_trail_ema_period
                        if _state.temp_t9_trail_m5_ema_period is not None:
                            temp_overrides_api["trail_m5_ema_period"] = _state.temp_t9_trail_m5_ema_period
                    if _state.temp_t10_regime_gate_enabled is not None:
                        temp_overrides_api["regime_gate_enabled"] = _state.temp_t10_regime_gate_enabled
                    if _state.temp_t10_regime_london_sell_veto is not None:
                        temp_overrides_api["regime_london_sell_veto"] = _state.temp_t10_regime_london_sell_veto
                    if _state.temp_t10_regime_london_start_hour_et is not None:
                        temp_overrides_api["regime_london_start_hour_et"] = _state.temp_t10_regime_london_start_hour_et
                    if _state.temp_t10_regime_london_end_hour_et is not None:
                        temp_overrides_api["regime_london_end_hour_et"] = _state.temp_t10_regime_london_end_hour_et
                    if _state.temp_t10_regime_boost_multiplier is not None:
                        temp_overrides_api["regime_boost_multiplier"] = _state.temp_t10_regime_boost_multiplier
                    if _state.temp_t10_regime_buy_base_multiplier is not None:
                        temp_overrides_api["regime_buy_base_multiplier"] = _state.temp_t10_regime_buy_base_multiplier
                    if _state.temp_t10_regime_sell_base_multiplier is not None:
                        temp_overrides_api["regime_sell_base_multiplier"] = _state.temp_t10_regime_sell_base_multiplier
                    if _state.temp_t10_regime_chop_pause_enabled is not None:
                        temp_overrides_api["regime_chop_pause_enabled"] = _state.temp_t10_regime_chop_pause_enabled
                    if _state.temp_t10_regime_chop_pause_minutes is not None:
                        temp_overrides_api["regime_chop_pause_minutes"] = _state.temp_t10_regime_chop_pause_minutes
                    if _state.temp_t10_regime_chop_pause_lookback_trades is not None:
                        temp_overrides_api["regime_chop_pause_lookback_trades"] = _state.temp_t10_regime_chop_pause_lookback_trades
                    if _state.temp_t10_regime_chop_pause_stop_rate is not None:
                        temp_overrides_api["regime_chop_pause_stop_rate"] = _state.temp_t10_regime_chop_pause_stop_rate
                    if _state.temp_t10_tier17_nonboost_multiplier is not None:
                        temp_overrides_api["tier17_nonboost_multiplier"] = _state.temp_t10_tier17_nonboost_multiplier
                    if _state.temp_t10_bucketed_exit_enabled is not None:
                        temp_overrides_api["bucketed_exit_enabled"] = _state.temp_t10_bucketed_exit_enabled
                    if _state.temp_t10_quick_tp1_pips is not None:
                        temp_overrides_api["quick_tp1_pips"] = _state.temp_t10_quick_tp1_pips
                    if _state.temp_t10_quick_tp1_close_pct is not None:
                        temp_overrides_api["quick_tp1_close_pct"] = _state.temp_t10_quick_tp1_close_pct
                    if _state.temp_t10_quick_be_spread_plus_pips is not None:
                        temp_overrides_api["quick_be_spread_plus_pips"] = _state.temp_t10_quick_be_spread_plus_pips
                    if _state.temp_t10_runner_tp1_pips is not None:
                        temp_overrides_api["runner_tp1_pips"] = _state.temp_t10_runner_tp1_pips
                    if _state.temp_t10_runner_tp1_close_pct is not None:
                        temp_overrides_api["runner_tp1_close_pct"] = _state.temp_t10_runner_tp1_close_pct
                    if _state.temp_t10_runner_be_spread_plus_pips is not None:
                        temp_overrides_api["runner_be_spread_plus_pips"] = _state.temp_t10_runner_be_spread_plus_pips
                    if _state.temp_t10_trail_escalation_enabled is not None:
                        temp_overrides_api["trail_escalation_enabled"] = _state.temp_t10_trail_escalation_enabled
                    if _state.temp_t10_trail_escalation_tier1_pips is not None:
                        temp_overrides_api["trail_escalation_tier1_pips"] = _state.temp_t10_trail_escalation_tier1_pips
                    if _state.temp_t10_trail_escalation_tier2_pips is not None:
                        temp_overrides_api["trail_escalation_tier2_pips"] = _state.temp_t10_trail_escalation_tier2_pips
                    if _state.temp_t10_trail_escalation_m15_ema_period is not None:
                        temp_overrides_api["trail_escalation_m15_ema_period"] = _state.temp_t10_trail_escalation_m15_ema_period
                    if _state.temp_t10_trail_escalation_m15_buffer_pips is not None:
                        temp_overrides_api["trail_escalation_m15_buffer_pips"] = _state.temp_t10_trail_escalation_m15_buffer_pips
                    if _state.temp_t10_runner_score_sizing_enabled is not None:
                        temp_overrides_api["runner_score_sizing_enabled"] = _state.temp_t10_runner_score_sizing_enabled
                    if _state.temp_t10_runner_base_lots is not None:
                        temp_overrides_api["runner_base_lots"] = _state.temp_t10_runner_base_lots
                    if _state.temp_t10_runner_min_lots is not None:
                        temp_overrides_api["runner_min_lots"] = _state.temp_t10_runner_min_lots
                    if _state.temp_t10_runner_max_lots is not None:
                        temp_overrides_api["runner_max_lots"] = _state.temp_t10_runner_max_lots
                    if _state.temp_t10_atr_stop_enabled is not None:
                        temp_overrides_api["atr_stop_enabled"] = _state.temp_t10_atr_stop_enabled
                    if _state.temp_t10_atr_stop_multiplier is not None:
                        temp_overrides_api["atr_stop_multiplier"] = _state.temp_t10_atr_stop_multiplier
                    if _state.temp_t10_atr_stop_max_pips is not None:
                        temp_overrides_api["atr_stop_max_pips"] = _state.temp_t10_atr_stop_max_pips
                    if not temp_overrides_api:
                        temp_overrides_api = None
                    if _policy_type == "phase3_integrated":
                        try:
                            _raw_state_path = _runtime_state_path(profile_name)
                            _raw_state = json.loads(_raw_state_path.read_text(encoding="utf-8")) if _raw_state_path.exists() else {}
                            _raw_phase3 = _raw_state.get("phase3_state")
                            if isinstance(_raw_phase3, dict):
                                phase3_state_for_filters = _raw_phase3
                        except Exception:
                            phase3_state_for_filters = None
                    if _policy_type == "kt_cg_trial_5" and getattr(_policy, "trend_exhaustion_enabled", False):
                        m3_df = data_by_tf.get("M3")
                        if m3_df is not None and not m3_df.empty:
                            from core.execution_engine import _detect_trend_flip_and_compute_exhaustion
                            mid = (_tick.bid + _tick.ask) / 2.0
                            exhaustion_result = _detect_trend_flip_and_compute_exhaustion(
                                m3_df, mid, pip_size, exhaustion_state or {}, _policy
                            )
                    if _policy_type in ("kt_cg_trial_7", "kt_cg_trial_8", "kt_cg_trial_9", "kt_cg_trial_10") and getattr(_policy, "trend_exhaustion_enabled", False):
                        m5_df = data_by_tf.get("M5")
                        if m5_df is not None and not m5_df.empty:
                            from core.execution_engine import _compute_trial7_trend_exhaustion
                            from core.indicators import ema as ema_fn
                            m5_local = m5_df
                            close_m5 = m5_local["close"].astype(float)
                            fast = ema_fn(close_m5, int(getattr(_policy, "m5_trend_ema_fast", 9)))
                            slow = ema_fn(close_m5, int(getattr(_policy, "m5_trend_ema_slow", 21)))
                            trend_side = "bull" if float(fast.iloc[-1]) > float(slow.iloc[-1]) else "bear"
                            mid = (_tick.bid + _tick.ask) / 2.0
                            exhaustion_result = _compute_trial7_trend_exhaustion(
                                policy=_policy,
                                m5_df=m5_local,
                                current_price=mid,
                                pip_size=pip_size,
                                trend_side=trend_side,
                            )
                except Exception:
                    temp_overrides_api = None
                    pass
            intraday_fib_corridor_snapshot = None
            if _policy_type in ("kt_cg_trial_9", "kt_cg_trial_10"):
                try:
                    from core.fib_pivots import compute_daily_fib_pivots

                    def _prev_d_row(d_df: pd.DataFrame | None) -> dict[str, Any] | None:
                        if d_df is None or d_df.empty:
                            return None
                        d_local = d_df.copy()
                        d_local["time"] = pd.to_datetime(d_local["time"], utc=True, errors="coerce")
                        d_local = d_local.dropna(subset=["time"]).sort_values("time")
                        if d_local.empty:
                            return None
                        now_date = now_utc.date().isoformat()
                        if str(d_local.iloc[-1]["time"].date().isoformat()) == now_date:
                            # Last bar is today's forming candle; take the previous completed day
                            if len(d_local) < 2:
                                return None
                            row = d_local.iloc[-2]
                        else:
                            # Last bar is a completed previous day (e.g. only completed bars returned)
                            row = d_local.iloc[-1]
                        return {k: row.get(k) for k in ("high", "low", "close")}

                    def _prev_candle_hl(df: pd.DataFrame | None) -> tuple[float | None, float | None]:
                        if df is None or df.empty:
                            return None, None
                        local = df.copy()
                        local["time"] = pd.to_datetime(local["time"], utc=True, errors="coerce")
                        local = local.dropna(subset=["time"]).sort_values("time")
                        if local.empty:
                            return None, None
                        row = local.iloc[-1]
                        try:
                            return float(row["high"]), float(row["low"])
                        except Exception:
                            return None, None

                    ntz_levels: dict[str, float] = {}
                    prev_daily = _prev_d_row(data_by_tf.get("D"))
                    if getattr(_policy, "ntz_use_prev_day_hl", True) and prev_daily is not None:
                        try:
                            ntz_levels["PDH"] = float(prev_daily["high"])
                            ntz_levels["PDL"] = float(prev_daily["low"])
                        except Exception:
                            pass
                    if getattr(_policy, "ntz_use_weekly_hl", True):
                        wh, wl = _prev_candle_hl(data_by_tf.get("W"))
                        if wh is not None and wl is not None:
                            ntz_levels["WH"] = wh
                            ntz_levels["WL"] = wl
                    if getattr(_policy, "ntz_use_monthly_hl", True):
                        mh, ml = _prev_candle_hl(data_by_tf.get("MN"))
                        if mh is not None and ml is not None:
                            ntz_levels["MH"] = mh
                            ntz_levels["ML"] = ml

                    fib_levels: dict[str, float] = {}
                    _d_df_debug = data_by_tf.get("D")
                    _d_len = len(_d_df_debug) if _d_df_debug is not None else "None"
                    _fib_enabled = bool(getattr(_policy, "ntz_use_fib_pivots", False))
                    print(f"[NTZ-API-DEBUG] fib_enabled={_fib_enabled} prev_daily={prev_daily} d_df_len={_d_len}")
                    if _fib_enabled and prev_daily is not None:
                        try:
                            fib_raw = compute_daily_fib_pivots(
                                float(prev_daily["high"]),
                                float(prev_daily["low"]),
                                float(prev_daily["close"]),
                            )
                            fib_toggle_map = {
                                "Fib-PP": bool(getattr(_policy, "ntz_use_fib_pp", True)),
                                "Fib-R1": bool(getattr(_policy, "ntz_use_fib_r1", True)),
                                "Fib-R2": bool(getattr(_policy, "ntz_use_fib_r2", True)),
                                "Fib-R3": bool(getattr(_policy, "ntz_use_fib_r3", True)),
                                "Fib-S1": bool(getattr(_policy, "ntz_use_fib_s1", True)),
                                "Fib-S2": bool(getattr(_policy, "ntz_use_fib_s2", True)),
                                "Fib-S3": bool(getattr(_policy, "ntz_use_fib_s3", True)),
                            }
                            fib_value_map = {
                                "Fib-PP": float(fib_raw["P"]),
                                "Fib-R1": float(fib_raw["R1"]),
                                "Fib-R2": float(fib_raw["R2"]),
                                "Fib-R3": float(fib_raw["R3"]),
                                "Fib-S1": float(fib_raw["S1"]),
                                "Fib-S2": float(fib_raw["S2"]),
                                "Fib-S3": float(fib_raw["S3"]),
                            }
                            fib_levels = {
                                label: value
                                for label, value in fib_value_map.items()
                                if fib_toggle_map.get(label, False)
                            }
                            ntz_levels.update(fib_levels)
                            print(f"[NTZ-API-DEBUG] fib computed: {list(fib_levels.keys())}")
                        except Exception as _fib_ex:
                            print(f"[NTZ-API-DEBUG] fib exception: {_fib_ex}")
                            fib_levels = {}

                    ntz_filter_snapshot = {
                        "enabled": bool(getattr(_policy, "ntz_enabled", False)),
                        "buffer_pips": float(getattr(_policy, "ntz_buffer_pips", 10.0)),
                        "levels": ntz_levels,
                        "fib_pivots_enabled": _fib_enabled,
                        "fib_levels": fib_levels,
                    }
                except Exception as _ntz_ex:
                    print(f"[NTZ-API-DEBUG] outer exception: {_ntz_ex}")
                    ntz_filter_snapshot = None
                try:
                    from core.fib_pivots import (
                        compute_rolling_intraday_fib_levels,
                        is_fixed_intraday_fib_timeframe,
                        resample_intraday_ohlc,
                    )
                    from core.no_trade_zone import IntradayFibCorridorFilter

                    ifib_enabled = bool(getattr(_policy, "intraday_fib_enabled", False))
                    ifib_timeframe = str(getattr(_policy, "intraday_fib_timeframe", "M15"))
                    if ifib_timeframe not in ("M15", "M5", "H1", "H2", "H3"):
                        ifib_timeframe = "M15"
                    ifib_lookback = max(1, int(getattr(_policy, "intraday_fib_lookback_bars", 16)))
                    ifib_filter = IntradayFibCorridorFilter(
                        enabled=ifib_enabled,
                        lower_level=str(getattr(_policy, "intraday_fib_lower_level", "S1")),
                        upper_level=str(getattr(_policy, "intraday_fib_upper_level", "R1")),
                        timeframe=ifib_timeframe,
                        lookback_bars=ifib_lookback,
                        boundary_buffer_pips=float(getattr(_policy, "intraday_fib_boundary_buffer_pips", 1.0)),
                        hysteresis_pips=float(getattr(_policy, "intraday_fib_hysteresis_pips", 1.0)),
                        pip_size=pip_size,
                    )
                    if ifib_enabled:
                        if is_fixed_intraday_fib_timeframe(ifib_timeframe):
                            # Resample M1 → H1/H2/H3, use rolling window (same
                            # as M15/M5) so corridor covers multiple candles.
                            m1_df = data_by_tf.get("M1")
                            source_df = resample_intraday_ohlc(m1_df, ifib_timeframe) if m1_df is not None and not m1_df.empty else None
                            if source_df is not None and not source_df.empty:
                                ifib_levels = compute_rolling_intraday_fib_levels(source_df, lookback_bars=ifib_lookback)
                                if ifib_levels is not None:
                                    completed = source_df.iloc[:-1] if len(source_df) > 1 else source_df
                                    window = completed.iloc[-ifib_lookback:]
                                    r_high = float(window["high"].max())
                                    r_low = float(window["low"].min())
                                    ifib_filter.update_levels(
                                        ifib_levels,
                                        rolling_high=r_high,
                                        rolling_low=r_low,
                                        calculation_mode="rolling_window",
                                    )
                                    ifib_filter.check_corridor((_tick.bid + _tick.ask) / 2.0)
                                else:
                                    ifib_filter.update_levels(None, calculation_mode="rolling_window")
                            else:
                                ifib_filter.update_levels(None, calculation_mode="rolling_window")
                        else:
                            ifib_df = data_by_tf.get(ifib_timeframe)
                            if ifib_df is not None and not ifib_df.empty:
                                ifib_levels = compute_rolling_intraday_fib_levels(ifib_df, lookback_bars=ifib_lookback)
                                if ifib_levels is not None:
                                    completed = ifib_df.iloc[:-1] if len(ifib_df) > 1 else ifib_df
                                    window = completed.iloc[-ifib_lookback:]
                                    r_high = float(window["high"].max())
                                    r_low = float(window["low"].min())
                                    ifib_filter.update_levels(
                                        ifib_levels,
                                        rolling_high=r_high,
                                        rolling_low=r_low,
                                        calculation_mode="rolling_window",
                                    )
                                    ifib_filter.check_corridor((_tick.bid + _tick.ask) / 2.0)
                                else:
                                    ifib_filter.update_levels(None, calculation_mode="rolling_window")
                            else:
                                ifib_filter.update_levels(None, calculation_mode="rolling_window")
                    intraday_fib_corridor_snapshot = ifib_filter.get_snapshot()
                except Exception as _ifib_ex:
                    print(f"[api] intraday fib corridor snapshot error for '{profile_name}': {_ifib_ex}")
                    intraday_fib_corridor_snapshot = None
            # Compute policy-specific eval snapshot for filter display
            t6_eval_result = None
            t10_eval_result = None
            if _policy_type == "kt_cg_trial_6" and _policy is not None:
                try:
                    m3_df = data_by_tf.get("M3")
                    if m3_df is not None and not m3_df.empty:
                        from core.execution_engine import _evaluate_m3_slope_trend_trial_6
                        trend_result = _evaluate_m3_slope_trend_trial_6(m3_df, _policy, pip_size)
                        t6_eval_result = {"trend_result": trend_result}
                except Exception:
                    pass
            store = _store_for(profile_name, log_dir=log_dir)
            from core.dashboard_builder import effective_policy_for_dashboard
            _policy_for_snapshot = effective_policy_for_dashboard(_policy, temp_overrides_api) if _policy is not None else None
            if _policy_type == "kt_cg_trial_10" and _policy_for_snapshot is not None:
                try:
                    from core.execution_engine import evaluate_trial10_advisory_state
                    _t10_tier_state = {}
                    _tier_fired_raw = getattr(_state, "tier_fired", None)
                    if isinstance(_tier_fired_raw, dict):
                        _t10_tier_state = {int(k): bool(v) for k, v in _tier_fired_raw.items()}
                    t10_eval_result = evaluate_trial10_advisory_state(
                        profile,
                        _policy_for_snapshot,
                        data_by_tf,
                        float(_tick.bid),
                        float(_tick.ask),
                        _t10_tier_state,
                        temp_overrides=temp_overrides_api,
                    )
                except Exception as _t10_eval_err:
                    print(f"[api] trial10 eval snapshot error: {_t10_eval_err}")
            # Conviction sizing snapshot for dashboard
            _conviction_snap_api = None
            if _policy_type in ("kt_cg_trial_9", "kt_cg_trial_10") and _policy_for_snapshot is not None:
                try:
                    if not (
                        _policy_type == "kt_cg_trial_10"
                        and bool(getattr(_policy_for_snapshot, "runner_score_sizing_enabled", False))
                    ):
                        from core.conviction_sizing import compute_conviction as _compute_conv_api
                        from core.conviction_sizing import conviction_snapshot as _conv_snap_fn_api
                        from core.signal_engine import drop_incomplete_last_bar as _dilb_api
                        _conv_enabled_api = bool(getattr(_policy_for_snapshot, "conviction_sizing_enabled", False))
                        _m5_api = data_by_tf.get("M5")
                        _m1_api = data_by_tf.get("M1")
                        _is_bull_api = False
                        if _m5_api is not None and len(_m5_api) > 21:
                            _m5c_api = _dilb_api(_m5_api.copy(), "M5")
                            _m5_close_api = _m5c_api["close"].astype(float)
                            _e9_api = _m5_close_api.ewm(span=9, adjust=False).mean()
                            _e21_api = _m5_close_api.ewm(span=21, adjust=False).mean()
                            _is_bull_api = float(_e9_api.iloc[-1]) > float(_e21_api.iloc[-1])
                        _conv_r_api = _compute_conv_api(
                            m5_df=_dilb_api(_m5_api.copy(), "M5") if _m5_api is not None else None,
                            m1_df=_dilb_api(_m1_api.copy(), "M1") if _m1_api is not None else None,
                            pip_size=pip_size,
                            is_bull=_is_bull_api,
                            enabled=_conv_enabled_api,
                            base_lots=float(getattr(_policy_for_snapshot, "conviction_base_lots", 0.05)),
                            min_lots=float(getattr(_policy_for_snapshot, "conviction_min_lots", 0.01)),
                            max_lots=float(get_effective_risk(profile).max_lots),
                        )
                        _conviction_snap_api = _conv_snap_fn_api(_conv_r_api)
                except Exception as _conv_api_err:
                    print(f"[api] conviction sizing error: {_conv_api_err}")
            # --- Regime gate snapshot (Trial 10) ---
            _regime_snap_api: Optional[dict] = None
            if _policy_type == "kt_cg_trial_10" and _policy_for_snapshot is not None:
                try:
                    from core.execution_engine import _resolve_trial10_m5_bucket
                    from core.regime_gate import evaluate_regime_gate, regime_gate_snapshot, check_chop_pause
                    from core.signal_engine import drop_incomplete_last_bar as _dilb_rg_api
                    _hour_et_api = datetime.now(ZoneInfo("America/New_York")).hour
                    # Derive likely side from M5 trend
                    _rg_side_api = "buy"
                    _rg_m5_bucket_api = "normal"
                    _m5_rg = data_by_tf.get("M5")
                    if _m5_rg is not None and len(_m5_rg) > 21:
                        _m5c_rg = _dilb_rg_api(_m5_rg.copy(), "M5")
                        _m5_close_rg = _m5c_rg["close"].astype(float)
                        _e9_rg = _m5_close_rg.ewm(span=9, adjust=False).mean()
                        _e21_rg = _m5_close_rg.ewm(span=21, adjust=False).mean()
                        _rg_side_api = "buy" if float(_e9_rg.iloc[-1]) > float(_e21_rg.iloc[-1]) else "sell"
                        _rg_m5_bucket_api, _, _ = _resolve_trial10_m5_bucket(
                            _m5_close_rg,
                            float(profile.pip_size),
                            _rg_side_api == "buy",
                        )
                    _chop_start = _state.chop_pause_buy_start_utc if _rg_side_api == "buy" else _state.chop_pause_sell_start_utc
                    _chop_reason = _state.chop_pause_buy_reason if _rg_side_api == "buy" else _state.chop_pause_sell_reason
                    _chop_start_dt = pd.Timestamp(_chop_start).to_pydatetime() if _chop_start else None
                    if _chop_start_dt is not None and _chop_start_dt.tzinfo is None:
                        _chop_start_dt = _chop_start_dt.replace(tzinfo=timezone.utc)
                    _closed_events = []
                    if trades_df is not None and not trades_df.empty and "exit_timestamp_utc" in trades_df.columns:
                        _closed_df = trades_df.copy()
                        _closed_df["exit_dt"] = pd.to_datetime(_closed_df["exit_timestamp_utc"], utc=True, errors="coerce")
                        if "policy_type" in _closed_df.columns:
                            _closed_df = _closed_df[
                                (_closed_df["policy_type"].astype(str) == "kt_cg_trial_10")
                                & _closed_df["exit_dt"].notna()
                            ]
                        else:
                            _closed_df = _closed_df[_closed_df["exit_dt"].notna()]
                        for _, _row in _closed_df.tail(100).iterrows():
                            _tier = _row.get("tier_number")
                            _closed_events.append(
                                {
                                    "side": str(_row.get("side") or "").lower(),
                                    "exit_reason": str(_row.get("exit_reason") or ""),
                                    "close_time_utc": _row["exit_dt"].to_pydatetime(),
                                    "pullback_label": None if pd.isna(_row.get("pullback_quality_label")) else str(_row.get("pullback_quality_label")).lower(),
                                    "tier": None if pd.isna(_tier) else int(_tier),
                                }
                            )
                    _chop_paused_api, _chop_reason_live = check_chop_pause(
                        side=_rg_side_api,
                        recent_trades=_closed_events,
                        now_utc=datetime.now(timezone.utc),
                        lookback_trades=int(getattr(_policy_for_snapshot, "regime_chop_pause_lookback_trades", 5)),
                        stop_rate_threshold=float(getattr(_policy_for_snapshot, "regime_chop_pause_stop_rate", 0.6)),
                        pause_minutes=int(getattr(_policy_for_snapshot, "regime_chop_pause_minutes", 45)),
                        current_pause_start=_chop_start_dt,
                        m5_bucket=_rg_m5_bucket_api,
                    )
                    _rg_result_api = evaluate_regime_gate(
                        hour_et=_hour_et_api,
                        side=_rg_side_api,
                        m5_bucket=_rg_m5_bucket_api,
                        enabled=getattr(_policy_for_snapshot, "regime_gate_enabled", True),
                        london_sell_veto=getattr(_policy_for_snapshot, "regime_london_sell_veto", True),
                        london_start_hour_et=getattr(_policy_for_snapshot, "regime_london_start_hour_et", 3),
                        london_end_hour_et=getattr(_policy_for_snapshot, "regime_london_end_hour_et", 12),
                        boost_hours_et=getattr(_policy_for_snapshot, "regime_boost_hours_et", (6, 7, 12, 13, 14, 15)),
                        boost_multiplier=getattr(_policy_for_snapshot, "regime_boost_multiplier", 1.35),
                        buy_base_multiplier=getattr(_policy_for_snapshot, "regime_buy_base_multiplier", 0.65),
                        sell_base_multiplier=getattr(_policy_for_snapshot, "regime_sell_base_multiplier", 0.35),
                        chop_paused=_chop_paused_api,
                        chop_pause_reason=_chop_reason_live or _chop_reason or "",
                    )
                    _regime_snap_api = regime_gate_snapshot(_rg_result_api)
                    _regime_snap_api["chop_pause_reason"] = _chop_reason_live or _chop_reason or ""
                except Exception as _rg_err:
                    print(f"[api] regime gate snapshot error: {_rg_err}")

            # --- Runner Score (Trial #10 only, for dashboard) ---
            _runner_snap_api: Optional[dict] = None
            if _policy_type == "kt_cg_trial_10" and _policy_for_snapshot is not None:
                try:
                    from core.runner_score import compute_runner_score, compute_freshness, runner_score_snapshot
                    from core.execution_engine import _resolve_trial10_stop_pips
                    _rs_atr_api = _resolve_trial10_stop_pips(profile, _policy_for_snapshot, data_by_tf)
                    _rs_regime_api = str(getattr(_rg_result_api, "label", "") if "_rg_result_api" in dir() and _rg_result_api is not None else "")
                    _rs_m5_api = str((t10_eval_result or {}).get("trial10_m5_bucket") or (_rg_m5_bucket_api if "_rg_m5_bucket_api" in dir() else "normal"))
                    _rs_pq_api = dict((t10_eval_result or {}).get("trial10_pullback_quality") or {})
                    _rs_sr_api = _rs_pq_api.get("structure_ratio")
                    _rs_bars_cross_api = None
                    _rs_prior_ent_api = None
                    _m1_rs_api = data_by_tf.get("M1")
                    _m5_rs_api = data_by_tf.get("M5")
                    _rs_side_api = str((t10_eval_result or {}).get("side") or _rg_side_api if "_rg_side_api" in locals() else "buy").lower()
                    if _m1_rs_api is not None and _m5_rs_api is not None and not _m1_rs_api.empty and not _m5_rs_api.empty:
                        from core.signal_engine import drop_incomplete_last_bar as _dilb_rs_api
                        _m1c_rs_api = _dilb_rs_api(_m1_rs_api.copy(), "M1")["close"].astype(float)
                        _m5c_rs_api = _dilb_rs_api(_m5_rs_api.copy(), "M5")["close"].astype(float)
                        _rs_bars_cross_api, _rs_prior_ent_api = compute_freshness(
                            m1_close=_m1c_rs_api,
                            m5_close=_m5c_rs_api,
                            side=_rs_side_api,
                            trades_df=trades_df,
                            policy_type="kt_cg_trial_10",
                        )
                    _rs_result_api = compute_runner_score(
                        atr_stop_pips=_rs_atr_api,
                        regime_label=_rs_regime_api,
                        m5_bucket=_rs_m5_api,
                        structure_ratio=float(_rs_sr_api) if _rs_sr_api is not None else None,
                        bars_since_cross=_rs_bars_cross_api,
                        prior_entries=_rs_prior_ent_api,
                        freshness_mode="strict",
                    )
                    _runner_snap_api = runner_score_snapshot(_rs_result_api)
                except Exception as _rs_err_api:
                    print(f"[api] runner score snapshot error: {_rs_err_api}")

            filter_reports = build_dashboard_filters(
                profile=profile,
                tick=_tick,
                data_by_tf=data_by_tf,
                policy=_policy,
                policy_type=_policy_type,
                eval_result=t10_eval_result if _policy_type == "kt_cg_trial_10" else t6_eval_result,
                divergence_state=divergence_state,
                daily_reset_state=daily_reset_state,
                exhaustion_result=exhaustion_result,
                store=store,
                adapter=_adapter if '_adapter' in locals() else None,
                temp_overrides=temp_overrides_api,
                ntz_filter_snapshot=ntz_filter_snapshot,
                intraday_fib_corridor_snapshot=intraday_fib_corridor_snapshot,
                conviction_snapshot=_conviction_snap_api,
                regime_snapshot=_regime_snap_api,
                runner_snapshot=_runner_snap_api,
                phase3_state=phase3_state_for_filters,
            )
            filters.extend(asdict(f) for f in filter_reports)
            try:
                from core.dashboard_reporters import (
                    collect_trial_4_context,
                    collect_trial_5_context,
                    collect_trial_6_context,
                    collect_trial_7_context,
                    collect_trial_9_context,
                )

                policy_for_context = _policy_for_snapshot
                context_items = []
                if _policy_type == "kt_cg_trial_4" and policy_for_context is not None:
                    context_items = collect_trial_4_context(policy_for_context, data_by_tf, _tick, {}, None, pip_size)
                elif _policy_type == "kt_cg_trial_5" and policy_for_context is not None:
                    context_items = collect_trial_5_context(
                        policy_for_context, data_by_tf, _tick, {}, None, pip_size,
                        exhaustion_result=exhaustion_result, daily_reset_state=daily_reset_state,
                    )
                elif _policy_type == "kt_cg_trial_6" and policy_for_context is not None:
                    context_items = collect_trial_6_context(policy_for_context, data_by_tf, _tick, {}, t6_eval_result, pip_size)
                elif _policy_type == "kt_cg_trial_7" and policy_for_context is not None:
                    context_items = collect_trial_7_context(
                        policy_for_context, data_by_tf, _tick, {}, None, pip_size,
                        exhaustion_result=exhaustion_result,
                    )
                elif _policy_type == "kt_cg_trial_8" and policy_for_context is not None:
                    context_items = collect_trial_7_context(
                        policy_for_context, data_by_tf, _tick, {}, None, pip_size,
                        exhaustion_result=exhaustion_result,
                    )
                elif _policy_type in ("kt_cg_trial_9", "kt_cg_trial_10") and policy_for_context is not None:
                    context_items = collect_trial_9_context(
                        policy_for_context, data_by_tf, _tick, {}, None, pip_size,
                        exhaustion_result=exhaustion_result,
                        ntz_snapshot=ntz_filter_snapshot,
                        intraday_fib_snapshot=intraday_fib_corridor_snapshot,
                        conviction_snapshot=_conviction_snap_api,
                    )
                elif _policy_type == "phase3_integrated":
                    from core.dashboard_reporters import collect_phase3_context
                    context_items = collect_phase3_context(
                        policy_for_context or _policy,
                        data_by_tf,
                        _tick,
                        {},
                        phase3_state_for_filters or {},
                        pip_size,
                        active_preset_name=getattr(profile, "active_preset_name", None),
                    )
                context = [asdict(item) for item in context_items]
            except Exception as e:
                print(f"[api] dashboard context error for '{profile_name}': {e}")
    except Exception as e:
        print(f"[api] dashboard filters error for '{profile_name}': {e}")

    # --- Daily summary from store ---
    daily_summary = None
    try:
        store = _store_for(profile_name, log_dir=log_dir)
        date_str = now_utc.strftime("%Y-%m-%d")
        closed_today = store.get_trades_for_date(active_profile_name, date_str)
        trades_today = len(closed_today)
        wins = losses = 0
        total_pips = total_profit = 0.0
        pip_size = float(profile.pip_size) if profile else 0.01
        for row in closed_today:
            d = dict(row)
            pips = _normalized_trade_pips(d, pip_size)
            profit = _normalized_trade_profit_usd(
                d,
                symbol_hint=str(getattr(profile, "symbol", "") or ""),
            )
            raw_profit = d.get("profit")
            raw_pips = d.get("pips")
            if API_VERBOSE_LOGS:
                print(f"[api] daily_summary trade: id={d.get('trade_id')}, side={d.get('side')}, "
                      f"entry={d.get('entry_price')}, exit={d.get('exit_price')}, "
                      f"raw_profit={raw_profit}, norm_profit={profit}, raw_pips={raw_pips}, norm_pips={pips}")
            if pips is not None:
                total_pips += float(pips)
            if profit is not None:
                total_profit += float(profit)
            if profit is not None and abs(float(profit)) > 0.01:
                if float(profit) > 0:
                    wins += 1
                else:
                    losses += 1
            elif pips is not None and abs(float(pips)) > 0.05:
                if float(pips) > 0:
                    wins += 1
                else:
                    losses += 1
        if API_VERBOSE_LOGS:
            print(f"[api] daily_summary result: date={date_str}, profile={active_profile_name}, "
                  f"trades={trades_today}, wins={wins}, losses={losses}, "
                  f"total_pips={total_pips:.1f}, total_profit={total_profit:.2f}")
        win_rate = round(wins / trades_today * 100, 1) if trades_today > 0 else 0.0
        daily_summary = {
            "trades_today": trades_today,
            "wins": wins,
            "losses": losses,
            "total_pips": round(total_pips, 1),
            "total_profit": round(total_profit, 2),
            "win_rate": win_rate,
        }
    except Exception as e:
        print(f"[api] dashboard daily summary error for '{profile_name}': {e}")

    return {
        "timestamp_utc": now_utc.isoformat(),
        "preset_name": getattr(profile, "active_preset_name", "") or "",
        "mode": runtime.mode,
        "loop_running": _is_loop_running(profile_name),
        "entry_candidate_side": None,
        "entry_candidate_trigger": None,
        "filters": filters,
        "context": context,
        "positions": positions,
        "daily_summary": daily_summary,
        "bid": bid,
        "ask": ask,
        "spread_pips": round(spread_pips, 1),
    }


def _build_live_dashboard_state_with_timeout(
    profile_name: str,
    profile_path: Optional[str] = None,
    log_dir: Optional[Path] = None,
    timeout_seconds: Optional[float] = None,
) -> dict[str, Any]:
    """Bound live dashboard latency; fall back when broker calls stall."""
    if timeout_seconds is None:
        try:
            timeout_seconds = float(os.environ.get("LIVE_DASHBOARD_TIMEOUT_SEC", "2.5"))
        except ValueError:
            timeout_seconds = 2.5
    try:
        return _run_in_threadpool_with_timeout(
            _build_live_dashboard_state, timeout_seconds, profile_name, profile_path, log_dir
        )
    except FuturesTimeoutError:
        return {"error": "live_dashboard_timeout", "timestamp_utc": None}
    except Exception as e:
        return {"error": f"live_dashboard_exception:{e}", "timestamp_utc": None}


def _strip_trial10_directional_cap_filter(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove the retired Trial 10 directional-cap row from dashboard responses."""
    filters = payload.get("filters")
    if not isinstance(filters, list):
        return payload
    payload["filters"] = [
        row for row in filters
        if not (isinstance(row, dict) and str(row.get("display_name", "")).strip() == "Trial #10 Directional Cap")
    ]
    return payload


_DASHBOARD_FILE_FRESHNESS = 90.0  # seconds — file state considered fresh if < 90s old (avoids stale flash during slow loop iterations)


@app.get("/api/data/{profile_name}/filter-config")
def get_filter_config(profile_name: str, profile_path: Optional[str] = None) -> dict[str, Any]:
    """Return the active preset's filter configuration from the profile JSON."""
    resolved_path: Optional[Path] = None
    if profile_path:
        try:
            p = _resolve_profile_path(profile_path)
            if p.exists():
                resolved_path = p
        except Exception:
            pass
    if resolved_path is None:
        for p in _list_profile_paths():
            if p.stem == profile_name:
                resolved_path = p
                break
    if resolved_path is None:
        raise HTTPException(status_code=404, detail="Profile not found")

    profile = load_profile_v1(str(resolved_path))
    risk = profile.effective_risk or profile.risk

    # Find the active kt_cg policy
    policy = None
    for pol in profile.execution.policies:
        if getattr(pol, "enabled", False) and getattr(pol, "type", "").startswith("kt_cg"):
            policy = pol
            break

    if policy is None:
        return {"preset_name": profile.active_preset_name or "", "filters": {}}

    pol_type = getattr(policy, "type", "")
    is_trial_5 = pol_type == "kt_cg_trial_5"
    is_trial_4 = pol_type == "kt_cg_trial_4"
    is_trial_7 = pol_type == "kt_cg_trial_7"
    is_trial_8 = pol_type == "kt_cg_trial_8"
    is_trial_9 = pol_type in ("kt_cg_trial_9", "kt_cg_trial_10")

    filters: dict[str, Any] = {}

    # Spread (from risk config)
    filters["spread"] = {"enabled": True, "max_pips": risk.max_spread_pips}

    # Session filter (from strategy filters)
    sf = profile.strategy.filters.session_filter
    filters["session_filter"] = {"enabled": sf.enabled, "sessions": sf.sessions}
    sbb = getattr(profile.strategy.filters, "session_boundary_block", None)
    if sbb is not None:
        filters["session_boundary_block"] = {"enabled": getattr(sbb, "enabled", False), "buffer_minutes": getattr(sbb, "buffer_minutes", 15)}

    # EMA Zone Filter (T7 only; T8 has no EMA zone filter)
    if hasattr(policy, "ema_zone_filter_enabled"):
        if is_trial_7:
            zone_entry_mode = getattr(policy, "zone_entry_mode", "ema_cross")
            if zone_entry_mode not in ("ema_cross", "price_vs_ema5"):
                zone_entry_mode = "ema_cross"
            filters["ema_zone_filter"] = {
                "enabled": getattr(policy, "ema_zone_filter_enabled", False),
                "lookback": getattr(policy, "ema_zone_filter_lookback_bars", 3),
                "ema5_min_slope": getattr(policy, "ema_zone_filter_ema5_min_slope_pips_per_bar", 0.10),
                "ema9_min_slope": getattr(policy, "ema_zone_filter_ema9_min_slope_pips_per_bar", 0.08),
                "ema21_min_slope": getattr(policy, "ema_zone_filter_ema21_min_slope_pips_per_bar", 0.05),
            }
            filters["zone_entry"] = {
                "enabled": getattr(policy, "zone_entry_enabled", True),
                "mode": zone_entry_mode,
                "mode_description": (
                    "M5 trend + M1 EMA fast/slow cross"
                    if zone_entry_mode == "ema_cross"
                    else "M5 trend + live price vs M1 EMA5"
                ),
            }
            filters["m5_ema_distance_gate"] = {
                "enabled": True,
                "min_gap_pips": getattr(policy, "m5_min_ema_distance_pips", 1.0),
            }
        elif is_trial_8:
            zone_entry_mode = getattr(policy, "zone_entry_mode", "price_vs_ema5")
            if zone_entry_mode not in ("ema_cross", "price_vs_ema5"):
                zone_entry_mode = "price_vs_ema5"
            price_ema_period = int(getattr(policy, "m1_zone_entry_price_ema_period", 5))
            mode_description = (
                "M5 trend + M1 EMA fast/slow cross"
                if zone_entry_mode == "ema_cross"
                else f"M5 trend + live price vs M1 EMA{price_ema_period}"
            )
            filters["zone_entry"] = {
                "enabled": getattr(policy, "zone_entry_enabled", True),
                "mode": zone_entry_mode,
                "mode_description": mode_description,
            }
            filters["m5_ema_distance_gate"] = {
                "enabled": True,
                "min_gap_pips": getattr(policy, "m5_min_ema_distance_pips", 1.0),
            }
            filters["daily_level_filter"] = {
                "enabled": getattr(policy, "use_daily_level_filter", False),
                "buffer_pips": getattr(policy, "daily_level_buffer_pips", 3.0),
                "breakout_candles_required": getattr(policy, "daily_level_breakout_candles_required", 2),
            }
        elif is_trial_9:
            zone_entry_mode = getattr(policy, "zone_entry_mode", "price_vs_ema5")
            if zone_entry_mode not in ("ema_cross", "price_vs_ema5"):
                zone_entry_mode = "price_vs_ema5"
            price_ema_period = int(getattr(policy, "m1_zone_entry_price_ema_period", 5))
            mode_description = (
                "M5 trend + M1 EMA fast/slow cross"
                if zone_entry_mode == "ema_cross"
                else f"M5 trend + live price vs M1 EMA{price_ema_period}"
            )
            filters["zone_entry"] = {
                "enabled": getattr(policy, "zone_entry_enabled", True),
                "mode": zone_entry_mode,
                "mode_description": mode_description,
            }
            filters["m5_ema_distance_gate"] = {
                "enabled": True,
                "min_gap_pips": getattr(policy, "m5_min_ema_distance_pips", 1.0),
            }
            filters["ntz"] = {
                "enabled": getattr(policy, "ntz_enabled", True),
                "buffer_pips": getattr(policy, "ntz_buffer_pips", 10.0),
                "use_prev_day_hl": getattr(policy, "ntz_use_prev_day_hl", True),
                "use_weekly_hl": getattr(policy, "ntz_use_weekly_hl", True),
                "use_monthly_hl": getattr(policy, "ntz_use_monthly_hl", True),
                "use_fib_pivots": getattr(policy, "ntz_use_fib_pivots", False),
                "use_fib_pp": getattr(policy, "ntz_use_fib_pp", True),
                "use_fib_r1": getattr(policy, "ntz_use_fib_r1", True),
                "use_fib_r2": getattr(policy, "ntz_use_fib_r2", True),
                "use_fib_r3": getattr(policy, "ntz_use_fib_r3", True),
                "use_fib_s1": getattr(policy, "ntz_use_fib_s1", True),
                "use_fib_s2": getattr(policy, "ntz_use_fib_s2", True),
                "use_fib_s3": getattr(policy, "ntz_use_fib_s3", True),
            }
            filters["intraday_fib_corridor"] = {
                "enabled": getattr(policy, "intraday_fib_enabled", False),
                "timeframe": getattr(policy, "intraday_fib_timeframe", "M15"),
                "lookback_bars": getattr(policy, "intraday_fib_lookback_bars", 16),
                "lower_level": getattr(policy, "intraday_fib_lower_level", "S1"),
                "upper_level": getattr(policy, "intraday_fib_upper_level", "R1"),
                "boundary_buffer_pips": getattr(policy, "intraday_fib_boundary_buffer_pips", 1.0),
                "hysteresis_pips": getattr(policy, "intraday_fib_hysteresis_pips", 1.0),
            }
            filters["kill_switch"] = {
                "enabled": getattr(policy, "kill_switch_enabled", True),
                "zone_entry_action": getattr(policy, "kill_switch_zone_entry_action", "kill"),
            }
        else:
            filters["ema_zone_filter"] = {
                "enabled": getattr(policy, "ema_zone_filter_enabled", False),
                "threshold": getattr(policy, "ema_zone_filter_block_threshold", 0.35),
                "lookback": getattr(policy, "ema_zone_filter_lookback_bars", 3),
            }

    # Rolling Danger Zone (Trial #4 only)
    if hasattr(policy, "rolling_danger_zone_enabled"):
        filters["rolling_danger_zone"] = {
            "enabled": getattr(policy, "rolling_danger_zone_enabled", False),
            "lookback": getattr(policy, "rolling_danger_lookback_bars", 100),
            "pct": getattr(policy, "rolling_danger_zone_pct", 0.15),
        }

    # RSI Divergence (Trial #4 only)
    if hasattr(policy, "rsi_divergence_enabled"):
        filters["rsi_divergence"] = {
            "enabled": getattr(policy, "rsi_divergence_enabled", False),
        }

    # Tiered ATR (Trial #4)
    if is_trial_4 and hasattr(policy, "tiered_atr_filter_enabled"):
        filters["tiered_atr"] = {
            "enabled": getattr(policy, "tiered_atr_filter_enabled", False),
            "block_below": getattr(policy, "tiered_atr_block_below_pips", 4.0),
            "allow_all_max": getattr(policy, "tiered_atr_allow_all_max_pips", 12.0),
            "pullback_max": getattr(policy, "tiered_atr_pullback_only_max_pips", 15.0),
        }

    # M1 ATR (Trial #5)
    if is_trial_5 and hasattr(policy, "m1_atr_filter_enabled"):
        filters["m1_atr"] = {
            "enabled": getattr(policy, "m1_atr_filter_enabled", False),
            "tokyo_min": getattr(policy, "m1_atr_tokyo_min_pips", 3.0),
            "london_min": getattr(policy, "m1_atr_london_min_pips", 3.0),
            "ny_min": getattr(policy, "m1_atr_ny_min_pips", 3.5),
            "tokyo_max": getattr(policy, "m1_atr_tokyo_max_pips", 12.0),
            "london_max": getattr(policy, "m1_atr_london_max_pips", 14.0),
            "ny_max": getattr(policy, "m1_atr_ny_max_pips", 16.0),
        }

    # M3 ATR (Trial #5)
    if is_trial_5 and hasattr(policy, "m3_atr_filter_enabled"):
        filters["m3_atr"] = {
            "enabled": getattr(policy, "m3_atr_filter_enabled", False),
            "min": getattr(policy, "m3_atr_min_pips", 5.0),
            "max": getattr(policy, "m3_atr_max_pips", 16.0),
        }

    # Daily H/L Filter
    if hasattr(policy, "daily_hl_filter_enabled"):
        filters["daily_hl"] = {
            "enabled": getattr(policy, "daily_hl_filter_enabled", False),
            "buffer": getattr(policy, "daily_hl_buffer_pips", 5.0),
        }

    # Trend Exhaustion
    if hasattr(policy, "trend_exhaustion_enabled"):
        if is_trial_7 or is_trial_8 or is_trial_9:
            filters["trend_exhaustion"] = {
                "enabled": getattr(policy, "trend_exhaustion_enabled", False),
                "mode": getattr(policy, "trend_exhaustion_mode", "session_and_side"),
                "use_current_price": getattr(policy, "trend_exhaustion_use_current_price", True),
                "hysteresis_pips": getattr(policy, "trend_exhaustion_hysteresis_pips", 0.5),
                "p80_global": getattr(policy, "trend_exhaustion_p80_global", 12.03),
                "p90_global": getattr(policy, "trend_exhaustion_p90_global", 17.02),
                "p80_tokyo": getattr(policy, "trend_exhaustion_p80_tokyo", 12.67),
                "p90_tokyo": getattr(policy, "trend_exhaustion_p90_tokyo", 17.63),
                "p80_london": getattr(policy, "trend_exhaustion_p80_london", 11.06),
                "p90_london": getattr(policy, "trend_exhaustion_p90_london", 14.41),
                "p80_ny": getattr(policy, "trend_exhaustion_p80_ny", 12.66),
                "p90_ny": getattr(policy, "trend_exhaustion_p90_ny", 18.83),
                "p80_bull_tokyo": getattr(policy, "trend_exhaustion_p80_bull_tokyo", 11.85),
                "p90_bull_tokyo": getattr(policy, "trend_exhaustion_p90_bull_tokyo", 15.52),
                "p80_bull_london": getattr(policy, "trend_exhaustion_p80_bull_london", 10.21),
                "p90_bull_london": getattr(policy, "trend_exhaustion_p90_bull_london", 12.97),
                "p80_bull_ny": getattr(policy, "trend_exhaustion_p80_bull_ny", 11.21),
                "p90_bull_ny": getattr(policy, "trend_exhaustion_p90_bull_ny", 15.84),
                "p80_bear_tokyo": getattr(policy, "trend_exhaustion_p80_bear_tokyo", 13.44),
                "p90_bear_tokyo": getattr(policy, "trend_exhaustion_p90_bear_tokyo", 19.73),
                "p80_bear_london": getattr(policy, "trend_exhaustion_p80_bear_london", 12.01),
                "p90_bear_london": getattr(policy, "trend_exhaustion_p90_bear_london", 17.44),
                "p80_bear_ny": getattr(policy, "trend_exhaustion_p80_bear_ny", 13.97),
                "p90_bear_ny": getattr(policy, "trend_exhaustion_p90_bear_ny", 21.51),
                "extended_disable_zone_entry": getattr(policy, "trend_exhaustion_extended_disable_zone_entry", True),
                "very_extended_disable_zone_entry": getattr(policy, "trend_exhaustion_very_extended_disable_zone_entry", True),
                "extended_min_tier_period": getattr(policy, "trend_exhaustion_extended_min_tier_period", 21),
                "very_extended_min_tier_period": getattr(policy, "trend_exhaustion_very_extended_min_tier_period", 29),
                "very_extended_tighten_caps": getattr(policy, "trend_exhaustion_very_extended_tighten_caps", True),
                "very_extended_cap_multiplier": getattr(policy, "trend_exhaustion_very_extended_cap_multiplier", 0.5),
                "very_extended_cap_min": getattr(policy, "trend_exhaustion_very_extended_cap_min", 1),
                "adaptive_tp_enabled": getattr(policy, "trend_exhaustion_adaptive_tp_enabled", False),
                "tp_extended_offset_pips": getattr(policy, "trend_exhaustion_tp_extended_offset_pips", 1.0),
                "tp_very_extended_offset_pips": getattr(policy, "trend_exhaustion_tp_very_extended_offset_pips", 2.0),
                "tp_min_pips": getattr(policy, "trend_exhaustion_tp_min_pips", 0.5),
            }
        else:
            filters["trend_exhaustion"] = {
                "enabled": getattr(policy, "trend_exhaustion_enabled", False),
                "fresh_max": getattr(policy, "trend_exhaustion_fresh_max", 2.0),
                "mature_max": getattr(policy, "trend_exhaustion_mature_max", 3.5),
            }

    # Dead Zone (Trial #5)
    if hasattr(policy, "daily_reset_block_enabled"):
        filters["dead_zone"] = {
            "enabled": getattr(policy, "daily_reset_block_enabled", False),
        }

    # Dead Zone (Trial #6 — configurable hours)
    if hasattr(policy, "dead_zone_enabled"):
        filters["dead_zone"] = {
            "enabled": getattr(policy, "dead_zone_enabled", False),
            "start_hour": getattr(policy, "dead_zone_start_hour_utc", 21),
            "end_hour": getattr(policy, "dead_zone_end_hour_utc", 2),
        }

    # BB Reversal (Trial #6)
    if hasattr(policy, "bb_reversal_enabled"):
        filters["bb_reversal"] = {
            "enabled": getattr(policy, "bb_reversal_enabled", False),
            "num_tiers": getattr(policy, "bb_reversal_num_tiers", 10),
            "max_positions": getattr(policy, "max_bb_reversal_positions", 3),
            "tp_mode": getattr(policy, "bb_reversal_tp_mode", "middle_bb_entry"),
        }

    # Max trades per side
    if hasattr(policy, "max_open_trades_per_side"):
        val = getattr(policy, "max_open_trades_per_side", None)
        filters["max_trades"] = {
            "enabled": val is not None,
            "per_side": val,
        }
    if hasattr(policy, "max_zone_entry_open"):
        filters["max_zone_entry_open"] = {
            "enabled": getattr(policy, "max_zone_entry_open", None) is not None,
            "max": getattr(policy, "max_zone_entry_open", None),
        }
    if hasattr(policy, "max_tiered_pullback_open"):
        filters["max_tiered_pullback_open"] = {
            "enabled": getattr(policy, "max_tiered_pullback_open", None) is not None,
            "max": getattr(policy, "max_tiered_pullback_open", None),
        }

    return {"preset_name": profile.active_preset_name or "", "filters": filters}


def _compute_daily_summary_from_broker(
    profile_name: str,
    profile_path: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Compute daily summary by querying OANDA directly for today's closed trades.

    This is the single source of truth — no DB normalization, no stale file state.
    Uses OANDA's realizedPL for profit and averageClosePrice for pips.
    """
    resolved_path: Optional[Path] = None
    if profile_path:
        try:
            p = _resolve_profile_path(profile_path)
            if p.exists():
                resolved_path = p
        except Exception:
            pass
    if resolved_path is None:
        for p in _list_profile_paths():
            if p.stem == profile_name:
                resolved_path = p
                break
    if resolved_path is None:
        return None

    try:
        profile = load_profile_v1(str(resolved_path))
    except Exception:
        return None

    try:
        adapter = _get_dashboard_adapter(profile)
    except Exception:
        return None

    pip_size = float(profile.pip_size) if profile else 0.01

    try:
        closed_today = adapter.get_closed_trades_today(symbol=profile.symbol)
    except Exception as e:
        print(f"[api] daily summary broker query failed: {e}")
        return None

    trades_today = len(closed_today)
    wins = losses = 0
    total_pips = total_profit = 0.0
    for t in closed_today:
        profit = float(t.get("profit") or 0.0)
        entry = float(t.get("entry_price") or 0.0)
        exit_ = float(t.get("exit_price") or 0.0)
        side = str(t.get("side") or "").lower()
        # Pips from actual OANDA prices
        pips = 0.0
        if entry > 0 and exit_ > 0 and pip_size > 0 and side in ("buy", "sell"):
            pips = ((exit_ - entry) / pip_size) if side == "buy" else ((entry - exit_) / pip_size)
        total_pips += pips
        total_profit += profit
        # Use OANDA realizedPL directly — no normalization needed
        if abs(profit) > 0.01:
            if profit > 0:
                wins += 1
            else:
                losses += 1
    win_rate = round(wins / trades_today * 100, 1) if trades_today > 0 else 0.0
    return {
        "trades_today": trades_today,
        "wins": wins,
        "losses": losses,
        "total_pips": round(total_pips, 1),
        "total_profit": round(total_profit, 2),
        "win_rate": win_rate,
    }


@app.get("/api/data/{profile_name}/debug-daily-trades")
def debug_daily_trades(profile_name: str, profile_path: Optional[str] = None) -> dict[str, Any]:
    """Debug endpoint: show OANDA's actual closed trades for today."""
    from datetime import datetime, timezone

    resolved_path: Optional[Path] = None
    if profile_path:
        try:
            p = _resolve_profile_path(profile_path)
            if p.exists():
                resolved_path = p
        except Exception:
            pass
    if resolved_path is None:
        for p in _list_profile_paths():
            if p.stem == profile_name:
                resolved_path = p
                break
    if resolved_path is None:
        return {"error": "profile not found"}

    try:
        profile = load_profile_v1(str(resolved_path))
    except Exception as e:
        return {"error": f"load profile: {e}"}

    now_utc = datetime.now(timezone.utc)
    date_str = now_utc.strftime("%Y-%m-%d")

    # Query OANDA directly
    oanda_trades: list[dict] = []
    oanda_error: str | None = None
    try:
        adapter = _get_dashboard_adapter(profile)
        oanda_trades = adapter.get_closed_trades_today(symbol=profile.symbol)
    except Exception as e:
        oanda_error = str(e)

    # Also compute the summary
    broker_summary = _compute_daily_summary_from_broker(profile_name, profile_path=profile_path)

    return {
        "date_utc": date_str,
        "oanda_closed_today": oanda_trades,
        "oanda_count": len(oanda_trades),
        "oanda_error": oanda_error,
        "computed_summary": broker_summary,
    }


@app.get("/api/data/{profile_name}/dashboard")
def get_dashboard(profile_name: str, profile_path: Optional[str] = None) -> dict[str, Any]:
    """Return dashboard state with lean run-loop-file-first behavior."""
    try:
        return _get_dashboard_impl(profile_name, profile_path)
    except Exception as e:
        import traceback
        print(f"[api] DASHBOARD CRASH for '{profile_name}': {e}\n{traceback.format_exc()}")
        return {
            "timestamp_utc": None,
            "preset_name": "",
            "mode": "DISARMED",
            "loop_running": False,
            "entry_candidate_side": None,
            "entry_candidate_trigger": None,
            "filters": [],
            "context": [],
            "positions": [],
            "daily_summary": None,
            "bid": 0.0,
            "ask": 0.0,
            "spread_pips": 0.0,
            "stale": True,
            "stale_age_seconds": None,
            "data_source": "error",
            "error": str(e),
        }


def _dashboard_state_is_phase3(state: Optional[dict[str, Any]]) -> bool:
    if not isinstance(state, dict):
        return False
    preset_name = str(state.get("preset_name") or "").lower()
    if "phase3_integrated" in preset_name or "phase 3" in preset_name:
        return True
    for item in list(state.get("context") or []):
        if not isinstance(item, dict):
            continue
        category = str(item.get("category") or "").lower()
        key = str(item.get("key") or "").lower()
        if category in {"runtime", "decision", "frozen", "v14", "london", "v44"}:
            return True
        if key in {
            "active session",
            "strategy tag",
            "ownership cell",
            "regime label",
            "defensive flags",
            "frozen package",
        }:
            return True
    for flt in list(state.get("filters") or []):
        if not isinstance(flt, dict):
            continue
        filter_id = str(flt.get("filter_id") or "").lower()
        if (
            filter_id.startswith("phase3_")
            or filter_id.startswith("tokyo")
            or filter_id.startswith("london")
            or filter_id.startswith("ny_")
        ):
            return True
    return False


def _lean_dashboard_cache_key(profile_name: str, profile_path: Optional[str], log_dir: Path) -> str:
    try:
        ld = str(log_dir.resolve())
    except Exception:
        ld = str(log_dir)
    return f"{profile_name}\x1f{profile_path or ''}\x1f{ld}"


def _get_dashboard_impl(profile_name: str, profile_path: Optional[str] = None) -> dict[str, Any]:
    """Internal dashboard implementation."""
    from core.dashboard_models import read_dashboard_state

    if LEAN_UI_MODE:
        from datetime import datetime, timezone

        log_dir = _pick_best_dashboard_log_dir(profile_name, profile_path)
        lk = _lean_dashboard_cache_key(profile_name, profile_path, log_dir)
        now_mono = _time.monotonic()
        lean_hit = _lean_dashboard_cache.get(lk)
        if lean_hit and (now_mono - lean_hit[0]) < _LEAN_DASHBOARD_CACHE_TTL:
            out = dict(lean_hit[1])
            out["loop_running"] = _is_loop_running(profile_name)
            return out

        file_state = read_dashboard_state(log_dir)
        if file_state is not None:
            stale_age_seconds: float | None = None
            stale = True
            ts = file_state.get("timestamp_utc")
            if ts:
                try:
                    parsed = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                    stale_age_seconds = max(0.0, (datetime.now(timezone.utc) - parsed).total_seconds())
                    stale = stale_age_seconds >= _DASHBOARD_FILE_FRESHNESS
                except Exception:
                    stale_age_seconds = None
            loop_running = _is_loop_running(profile_name)
            result = dict(file_state)
            result["loop_running"] = loop_running
            result.setdefault("entry_candidate_side", None)
            result.setdefault("entry_candidate_trigger", None)
            result["stale"] = stale
            result["stale_age_seconds"] = stale_age_seconds
            result["data_source"] = "run_loop_file"
            try:
                _loop_log_path = log_dir / "loop_log.json"
                if _loop_log_path.exists():
                    result["loop_log"] = json.loads(_loop_log_path.read_text(encoding="utf-8"))
                else:
                    result["loop_log"] = []
            except Exception:
                result["loop_log"] = []
            # Only build live state when file is moderately stale (< 10 min).
            # Beyond that the loop is clearly not running — just return the file state
            # to avoid expensive OANDA fetches that cause 502 timeouts.
            prefer_live_phase3 = _dashboard_state_is_phase3(file_state)
            _moderately_stale = stale and (stale_age_seconds is not None and stale_age_seconds < 600)
            if prefer_live_phase3 and _moderately_stale:
                live = _build_live_dashboard_state_with_timeout(profile_name, profile_path, log_dir=log_dir)
                if "error" not in live:
                    result = dict(live)
                    result["loop_running"] = loop_running
                    result["stale"] = False
                    result["stale_age_seconds"] = 0.0
                    result["data_source"] = "live_phase3" if prefer_live_phase3 else "live_fallback"
                    result["loop_log"] = file_state.get("loop_log", result.get("loop_log", []))
                else:
                    result["loop_log"] = file_state.get("loop_log", result.get("loop_log", []))
            result["positions"] = _enrich_phase3_rows(list(result.get("positions") or []))
            out = _strip_trial10_directional_cap_filter(result)
            _lean_dashboard_cache[lk] = (_time.monotonic(), dict(out))
            return out

        # No file state — try live build (first load / no loop has ever run)
        live = _build_live_dashboard_state_with_timeout(profile_name, profile_path, log_dir=log_dir)
        if "error" not in live:
            live["stale"] = True
            live["stale_age_seconds"] = None
            live["data_source"] = "live_fallback"
            try:
                _loop_log_path = log_dir / "loop_log.json"
                if _loop_log_path.exists():
                    live["loop_log"] = json.loads(_loop_log_path.read_text(encoding="utf-8"))
                else:
                    live["loop_log"] = []
            except Exception:
                live["loop_log"] = []
            live["positions"] = _enrich_phase3_rows(list(live.get("positions") or []))
            out = _strip_trial10_directional_cap_filter(live)
            _lean_dashboard_cache[lk] = (_time.monotonic(), dict(out))
            return out

        out = _strip_trial10_directional_cap_filter({
            "timestamp_utc": None,
            "preset_name": "",
            "mode": "DISARMED",
            "loop_running": _is_loop_running(profile_name),
            "entry_candidate_side": None,
            "entry_candidate_trigger": None,
            "filters": [],
            "context": [],
            "positions": [],
            "daily_summary": None,
            "bid": 0.0,
            "ask": 0.0,
            "spread_pips": 0.0,
            "stale": True,
            "stale_age_seconds": None,
            "data_source": "none",
        })
        _lean_dashboard_cache[lk] = (_time.monotonic(), dict(out))
        return out

    # Legacy behavior (opt-in): live + file merge.
    now = _time.monotonic()
    cached = _dashboard_live_cache.get(profile_name)
    if cached and (now - cached[0]) < _DASHBOARD_LIVE_TTL:
        return cached[1]

    log_dir = _pick_best_dashboard_log_dir(profile_name, profile_path)
    live = _build_live_dashboard_state_with_timeout(profile_name, profile_path, log_dir=log_dir)
    if "error" in live:
        return _strip_trial10_directional_cap_filter(live)

    file_state = read_dashboard_state(log_dir)
    prefer_live_phase3 = _dashboard_state_is_phase3(file_state) or _dashboard_state_is_phase3(live)

    if file_state is not None:
        # Check freshness
        file_ts = file_state.get("timestamp_utc")
        is_fresh = False
        if file_ts:
            try:
                from datetime import datetime, timezone
                ft = datetime.fromisoformat(file_ts.replace("Z", "+00:00"))
                age = (datetime.now(timezone.utc) - ft).total_seconds()
                is_fresh = age < _DASHBOARD_FILE_FRESHNESS
            except Exception:
                pass

        if is_fresh and not prefer_live_phase3:
            # Use file's rich filters/context (trial-specific), but override
            # positions, prices, and daily summary with live broker data
            file_state["positions"] = live["positions"]
            file_state["bid"] = live["bid"]
            file_state["ask"] = live["ask"]
            file_state["spread_pips"] = live["spread_pips"]
            if live["daily_summary"] is not None:
                file_state["daily_summary"] = live["daily_summary"]
            result = file_state
        else:
            # Stale file — use pure live state
            result = live
    else:
        result = live

    result["stale"] = False
    result["stale_age_seconds"] = 0.0
    result["data_source"] = "live_phase3" if prefer_live_phase3 else "run_loop_file"
    result.setdefault("entry_candidate_side", None)
    result.setdefault("entry_candidate_trigger", None)
    result["positions"] = _enrich_phase3_rows(list(result.get("positions") or []))
    result = _strip_trial10_directional_cap_filter(result)
    _dashboard_live_cache[profile_name] = (now, result)
    return result


@app.get("/api/data/{profile_name}/trade-events")
def get_trade_events(profile_name: str, limit: int = 50, profile_path: Optional[str] = None) -> list[dict[str, Any]]:
    """Returns trade_events.json contents (append-only trade log)."""
    from core.dashboard_models import read_trade_events

    log_dir = _pick_best_trade_events_log_dir(profile_name, profile_path)
    _backfill_trade_event_tier_labels(profile_name, log_dir)
    events = read_trade_events(log_dir, limit=limit)
    events = _hydrate_trade_event_close_financials(profile_name, events, log_dir=log_dir)
    return _enrich_phase3_rows(events)


def _hydrate_trade_event_close_financials(profile_name: str, events: list[dict[str, Any]], log_dir: Optional[Path] = None) -> list[dict[str, Any]]:
    """Prefer DB close-trade pips/profit in trade-events for corrected historical display."""
    if not events:
        return events
    close_ids = {
        str(ev.get("trade_id") or "")
        for ev in events
        if isinstance(ev, dict) and ev.get("event_type") == "close" and ev.get("trade_id")
    }
    if not close_ids:
        return events
    try:
        active_profile_name = log_dir.name if log_dir is not None else profile_name
        store = _store_for(profile_name, log_dir=log_dir)
        trades_df = store.read_trades_df(active_profile_name)
        if trades_df is None or trades_df.empty:
            return events
        if "trade_id" not in trades_df.columns:
            return events
        matched = trades_df[trades_df["trade_id"].astype(str).isin(close_ids)]
        if matched.empty:
            return events
        financials: dict[str, dict[str, Any]] = {}
        for _, row in matched.iterrows():
            tid = str(row.get("trade_id") or "")
            if not tid:
                continue
            pips = row.get("pips")
            profit = _normalized_trade_profit_usd(row)
            if pd.isna(pips):
                pips = None
            financials[tid] = {
                "pips": float(pips) if pips is not None else None,
                "profit": float(profit) if profit is not None else None,
            }
        if not financials:
            return events
        hydrated: list[dict[str, Any]] = []
        for ev in events:
            if not isinstance(ev, dict):
                hydrated.append(ev)
                continue
            if ev.get("event_type") != "close":
                hydrated.append(ev)
                continue
            tid = str(ev.get("trade_id") or "")
            fin = financials.get(tid)
            if not fin:
                hydrated.append(ev)
                continue
            ev2 = dict(ev)
            ev2["pips"] = fin.get("pips")
            ev2["profit"] = fin.get("profit")
            hydrated.append(ev2)
        return hydrated
    except Exception:
        return events


def _backfill_trade_event_tier_labels(profile_name: str, log_dir: Path) -> None:
    """Backfill legacy tiered_pullback trade-events with explicit EMA tier labels.

    Converts trigger_type from "tiered_pullback" -> "tiered_pullback_emaXX" by
    matching open events against placed executions with rule_id "...:tier_XX".
    """
    events_path = log_dir / "trade_events.json"
    if not events_path.exists():
        return
    try:
        events = json.loads(events_path.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(events, list) or not events:
        return

    missing_idxs: list[int] = []
    for i, ev in enumerate(events):
        if not isinstance(ev, dict):
            continue
        if ev.get("event_type") != "open":
            continue
        if str(ev.get("trigger_type") or "") != "tiered_pullback":
            continue
        missing_idxs.append(i)
    if not missing_idxs:
        return

    try:
        active_profile_name = log_dir.name if log_dir is not None else profile_name
        store = _store_for(profile_name, log_dir=log_dir)
        execs = store.read_executions_df(active_profile_name)
    except Exception:
        return
    if execs is None or execs.empty:
        return

    tier_execs: list[dict[str, Any]] = []
    for _, row in execs.iterrows():
        try:
            if int(row.get("placed") or 0) != 1:
                continue
        except Exception:
            continue
        rule_id = str(row.get("rule_id") or "")
        m = re.search(r":tier_(\d+)", rule_id)
        if not m:
            continue
        try:
            ts = pd.to_datetime(row.get("timestamp_utc"), utc=True)
        except Exception:
            continue
        tier_execs.append(
            {
                "timestamp": ts,
                "tier": int(m.group(1)),
                "rule_id": rule_id,
            }
        )
    if not tier_execs:
        return

    changed = False
    for idx in missing_idxs:
        ev = events[idx]
        ev_ts_raw = ev.get("timestamp_utc")
        if not ev_ts_raw:
            continue
        try:
            ev_ts = pd.to_datetime(ev_ts_raw, utc=True)
        except Exception:
            continue

        trade_id = str(ev.get("trade_id") or "")
        policy_prefix = ""
        if trade_id.startswith("kt_cg_trial_"):
            parts = trade_id.split(":")
            if parts:
                policy_prefix = parts[0] + ":"

        best_match = None
        best_diff = None
        for ex in tier_execs:
            if policy_prefix and not str(ex["rule_id"]).startswith(policy_prefix):
                continue
            diff = abs((ex["timestamp"] - ev_ts).total_seconds())
            if diff > 8.0:
                continue
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_match = ex

        if best_match is None:
            continue
        ev["trigger_type"] = f"tiered_pullback_ema{best_match['tier']}"
        changed = True

    if changed:
        try:
            events_path.write_text(json.dumps(events, default=str) + "\n", encoding="utf-8")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# AI Trading Chat (SSE streaming)
# ---------------------------------------------------------------------------

class _AiChatMessage(BaseModel):
    role: str
    content: str

class AiChatRequest(BaseModel):
    message: str
    history: list[_AiChatMessage] = []
    chat_model: Optional[str] = None


class PlaceLimitOrderRequest(BaseModel):
    side: str  # "buy" or "sell"
    price: float
    lots: float
    sl: Optional[float] = None
    tp: Optional[float] = None
    time_in_force: str = "GTC"  # "GTC" or "GTD"
    gtd_time_utc: Optional[str] = None  # ISO datetime for GTD expiration
    comment: str = ""
    # Managed-exit strategy picked by the AI (or the operator after editing).
    # None / "none" -> broker SL/TP only (no runtime management).
    exit_strategy: Optional[str] = None
    exit_params: Optional[dict[str, Any]] = None


class AiSuggestTradeRequest(BaseModel):
    # Back-compat: older clients send chat_model. Newer clients should send
    # suggest_model. If both are provided, suggest_model wins.
    chat_model: Optional[str] = None
    suggest_model: Optional[str] = None


@app.post("/api/data/{profile_name}/place-limit-order")
def place_limit_order_endpoint(
    profile_name: str, profile_path: str, req: PlaceLimitOrderRequest,
) -> dict[str, Any]:
    """Place a pending limit order via OANDA."""
    path = _resolve_profile_path(profile_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Profile not found")
    try:
        profile = load_profile_v1(path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Profile load error: {e}")

    if getattr(profile, "broker_type", None) != "oanda":
        raise HTTPException(status_code=400, detail="Limit orders only supported for OANDA profiles")

    side = req.side.lower().strip()
    if side not in ("buy", "sell"):
        raise HTTPException(status_code=400, detail=f"Invalid side: {req.side!r}. Must be 'buy' or 'sell'.")
    if req.lots <= 0 or req.lots > 10:
        raise HTTPException(status_code=400, detail=f"Invalid lot size: {req.lots}. Must be 0 < lots <= 10.")
    if req.price <= 0:
        raise HTTPException(status_code=400, detail="Price must be positive.")
    tif = req.time_in_force.upper()
    if tif not in ("GTC", "GTD"):
        raise HTTPException(status_code=400, detail=f"Invalid time_in_force: {req.time_in_force!r}")
    if tif == "GTD" and not req.gtd_time_utc:
        raise HTTPException(status_code=400, detail="gtd_time_utc required when time_in_force is GTD")

    from adapters.broker import get_adapter
    from api.ai_exit_strategies import (
        AI_EXIT_STRATEGIES,
        DEFAULT_AI_EXIT_STRATEGY,
        merge_exit_params,
        normalize_exit_strategy,
        trail_mode_for_strategy,
    )

    # Resolve exit strategy + params (None or "none" means broker-only).
    exit_strategy_raw = (req.exit_strategy or "").strip().lower()
    managed_exit_strategy: Optional[str] = None
    managed_exit_params: dict[str, Any] = {}
    if exit_strategy_raw and exit_strategy_raw != "none":
        managed_exit_strategy = normalize_exit_strategy(exit_strategy_raw)
        managed_exit_params = merge_exit_params(managed_exit_strategy, req.exit_params)

    adapter = get_adapter(profile)
    try:
        adapter.initialize()
        result = adapter.order_send_pending_limit(
            symbol=profile.symbol,
            side=side,
            price=req.price,
            volume_lots=req.lots,
            sl=req.sl,
            tp=req.tp,
            time_in_force=tif,
            gtd_time_utc=req.gtd_time_utc,
            comment=req.comment or f"ai_assist:{profile_name}",
        )
        if result.retcode != 0:
            raise HTTPException(status_code=400, detail=f"Order rejected: {result.comment}")

        # If the operator chose a managed exit strategy, register the pending order
        # so the run loop's watchdog can promote it to a managed trade once it fills.
        if managed_exit_strategy and result.order is not None:
            try:
                _state_dir = LOGS_DIR / profile_name
                _state_dir.mkdir(parents=True, exist_ok=True)
                _state_path = _state_dir / "runtime_state.json"
                _state_data: dict[str, Any] = {}
                if _state_path.exists():
                    try:
                        _state_data = json.loads(_state_path.read_text(encoding="utf-8")) or {}
                    except Exception:
                        _state_data = {}
                _pending_list = list(_state_data.get("managed_pending_orders") or [])
                _pending_list = [p for p in _pending_list if str(p.get("order_id")) != str(result.order)]
                _pending_list.append({
                    "order_id": int(result.order),
                    "side": side,
                    "price": float(req.price),
                    "lots": float(req.lots),
                    "sl": float(req.sl) if req.sl is not None else None,
                    "tp": float(req.tp) if req.tp is not None else None,
                    "exit_strategy": managed_exit_strategy,
                    "trail_mode": trail_mode_for_strategy(managed_exit_strategy),
                    "exit_params": managed_exit_params,
                    "source": "ai_manual",
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                })
                _state_data["managed_pending_orders"] = _pending_list
                _state_path.write_text(json.dumps(_state_data, indent=2) + "\n", encoding="utf-8")
            except Exception as _reg_exc:
                # Non-fatal — the order is placed; runtime management is best-effort.
                print(f"[ai_suggest_trade] failed to register managed pending order: {_reg_exc}")

        return {
            "status": "placed",
            "order_id": result.order,
            "side": side,
            "price": req.price,
            "lots": req.lots,
            "sl": req.sl,
            "tp": req.tp,
            "time_in_force": tif,
            "gtd_time_utc": req.gtd_time_utc,
            "exit_strategy": managed_exit_strategy,
            "exit_params": managed_exit_params if managed_exit_strategy else None,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error placing order: {e}")
    finally:
        try:
            adapter.shutdown()
        except Exception:
            pass


@app.post("/api/data/{profile_name}/ai-suggest-trade")
def ai_suggest_trade(
    profile_name: str, profile_path: str, req: AiSuggestTradeRequest,
) -> dict[str, Any]:
    """Ask the AI for a limit order suggestion based on current market context."""
    path = _resolve_profile_path(profile_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Profile not found")

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not configured")

    try:
        profile = load_profile_v1(path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Profile load error: {e}")

    from api.ai_trading_chat import (
        build_trading_context,
        resolve_ai_suggest_model,
        system_prompt_from_context,
    )

    try:
        ctx_timeout = float(os.environ.get("API_AI_CHAT_CTX_TIMEOUT_SEC", "20"))
    except ValueError:
        ctx_timeout = 20.0

    import json as _json
    import traceback as _tb

    try:
        ctx = _run_in_threadpool_with_timeout(build_trading_context, ctx_timeout, profile, profile_name)
    except FuturesTimeoutError:
        raise HTTPException(status_code=502, detail="Context build timed out")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Context build error: {e}")

    # Suggest endpoint uses its own (higher-reasoning) model pool, independent of
    # the chat model. `suggest_model` wins when provided; fall back to legacy
    # `chat_model` field for backward compat, else server default (gpt-4o).
    try:
        requested_suggest = req.suggest_model or req.chat_model
        effective_model = resolve_ai_suggest_model(requested_suggest)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    try:
        system = system_prompt_from_context(ctx, effective_model)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"System prompt build error: {e}")

    from api.ai_exit_strategies import (
        AI_EXIT_STRATEGIES,
        DEFAULT_AI_EXIT_STRATEGY,
        exit_strategies_prompt_block,
        merge_exit_params,
        normalize_exit_strategy,
    )

    _strategy_ids = ", ".join(f'"{sid}"' for sid in AI_EXIT_STRATEGIES.keys())
    _exit_catalog_text = exit_strategies_prompt_block()

    suggest_prompt = (
        "Based on the current market context (technicals, macro bias, levels, session, volatility, and any recent news/events), "
        "suggest ONE specific USDJPY limit order trade right now. "
        "You MUST respond with ONLY a valid JSON object — no markdown, no code fences, no explanation outside the JSON. "
        "Use this exact JSON schema:\n"
        '{\n'
        '  "side": "buy" or "sell",\n'
        '  "price": <limit entry price as number>,\n'
        '  "sl": <stop loss price as number>,\n'
        '  "tp": <take profit price as number>,\n'
        '  "lots": <position size as number, e.g. 0.05>,\n'
        '  "time_in_force": "GTC" or "GTD",\n'
        '  "gtd_time_utc": <ISO datetime string if GTD, else null>,\n'
        '  "exit_strategy": one of [' + _strategy_ids + '],\n'
        '  "exit_params": {<optional numeric overrides, e.g. "tp1_pips": 5.0, "hwm_trail_pips": 4.0>} or null,\n'
        '  "rationale": "<2-4 sentence explanation of the SETUP and the EXIT STRATEGY choice>",\n'
        '  "confidence": "low" or "medium" or "high"\n'
        '}\n'
        "\n"
        + _exit_catalog_text
        + "\n\n"
        "Rules:\n"
        "- The limit price must be AWAY from current price (buy below current, sell above current).\n"
        "- Use the nearest support/resistance levels and current technicals to set price, SL, and TP.\n"
        "- SL should be 10-15 pips from entry, TP 4-10 pips from entry, consistent with a scalper's style.\n"
        "- Lot size should be proportional to confidence (0.01-0.10 typical range).\n"
        "- Default to GTC unless there is a clear time-based reason (event risk, session close).\n"
        f'- For exit_strategy, default to "{DEFAULT_AI_EXIT_STRATEGY}" UNLESS the setup clearly favors another option (or none). '
        "Your rationale MUST explicitly justify the exit_strategy choice (why the default fits, or why you picked an alternative, "
        'or why "none" is better here).\n'
        "- If you genuinely see no good setup, return the JSON with confidence 'low' and explain in rationale.\n"
        "- If market is closed or data is stale, still provide a suggestion based on last known levels and context.\n"
    )

    import openai

    raw = ""
    try:
        client = openai.OpenAI()
        resp = client.chat.completions.create(
            model=effective_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": suggest_prompt},
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[: raw.rfind("```")]
        raw = raw.strip()

        suggestion = _json.loads(raw)
        # Validate required fields
        required = ("side", "price", "sl", "tp", "lots", "rationale", "confidence")
        missing = [f for f in required if f not in suggestion]
        if missing:
            raise HTTPException(status_code=502, detail=f"AI response missing fields: {missing}")
        # Normalize
        suggestion["side"] = str(suggestion["side"]).lower()
        suggestion["price"] = float(suggestion["price"])
        suggestion["sl"] = float(suggestion["sl"])
        suggestion["tp"] = float(suggestion["tp"])
        suggestion["lots"] = float(suggestion["lots"])
        suggestion.setdefault("time_in_force", "GTC")
        suggestion.setdefault("gtd_time_utc", None)
        # Normalize exit strategy + params (fall back to default if missing/invalid).
        raw_exit = suggestion.get("exit_strategy")
        if raw_exit is None or str(raw_exit).strip().lower() == "none":
            suggestion["exit_strategy"] = "none"
            suggestion["exit_params"] = {}
        else:
            suggestion["exit_strategy"] = normalize_exit_strategy(str(raw_exit))
            raw_params = suggestion.get("exit_params") if isinstance(suggestion.get("exit_params"), dict) else None
            suggestion["exit_params"] = merge_exit_params(suggestion["exit_strategy"], raw_params)
        # Echo the chosen model so the UI can display which brain picked the trade.
        suggestion["model_used"] = effective_model
        # Expose catalog so the UI can render dropdown options + descriptions.
        suggestion["available_exit_strategies"] = {
            sid: {
                "id": cfg["id"],
                "label": cfg["label"],
                "description": cfg["description"],
                "defaults": cfg.get("defaults") or {},
                "trail_mode": cfg.get("trail_mode"),
            }
            for sid, cfg in AI_EXIT_STRATEGIES.items()
        }
        return suggestion
    except HTTPException:
        raise
    except _json.JSONDecodeError:
        raise HTTPException(status_code=502, detail=f"AI returned invalid JSON: {raw[:500]}")
    except Exception as e:
        detail = f"AI suggestion error: {e}\n{_tb.format_exc()}"
        raise HTTPException(status_code=502, detail=detail)


@app.get("/api/ai-chat/models")
def get_ai_chat_models_list() -> dict[str, Any]:
    """Models the assistant UI may select (allowlist; extend via AI_CHAT_ALLOWED_MODELS)."""
    from api.ai_trading_chat import (
        allowed_ai_chat_models,
        allowed_ai_suggest_models,
        default_ai_chat_model,
        default_ai_suggest_model,
    )

    return {
        "models": allowed_ai_chat_models(),
        "default_model": default_ai_chat_model(),
        "suggest_models": allowed_ai_suggest_models(),
        "default_suggest_model": default_ai_suggest_model(),
    }


@app.get("/api/data/{profile_name}/ai-rail")
def get_ai_assistant_rail(profile_name: str, profile_path: str, days_ahead: int = 7) -> dict[str, Any]:
    """Compact right-rail payload for AI assistant UI tiles."""
    path = _resolve_profile_path(profile_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Profile not found")
    try:
        profile = load_profile_v1(path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"API fetch error (profile): {e}")

    from api.ai_trading_chat import build_trading_context, get_economic_calendar_events

    try:
        rail_timeout = float(os.environ.get("API_AI_RAIL_TIMEOUT_SEC", "12"))
    except ValueError:
        rail_timeout = 12.0

    try:
        ctx = _run_in_threadpool_with_timeout(build_trading_context, rail_timeout, profile, profile_name)
    except FuturesTimeoutError:
        raise HTTPException(status_code=502, detail="API fetch error (rail context timed out)")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"API fetch error (rail context): {e}")

    spot = ctx.get("spot_price") or {}
    mid = float(spot.get("mid", 0) or 0)
    ob = ctx.get("order_book") or {}
    supports = list(ob.get("buy_clusters") or [])
    resistances = list(ob.get("sell_clusters") or [])

    def _level_rows(rows: list[dict[str, Any]], direction: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for r in rows[:3]:
            try:
                price = float(r.get("price", 0) or 0)
                pct = float(r.get("pct", 0) or 0)
            except Exception:
                continue
            if price <= 0:
                continue
            dist = None
            if mid > 0:
                dist = round(((price - mid) / 0.01), 1)
            out.append({
                "price": round(price, 3),
                "weight_pct": round(pct * 100, 2),
                "distance_pips": dist,
                "direction": direction,
            })
        return out

    bias = ctx.get("cross_asset_bias") or {}
    cross = ctx.get("cross_assets") or {}
    oil_bias = bias.get("oil") or {}
    dxy_bias = bias.get("dxy") or {}
    gold_bias = bias.get("gold") or {}
    us10y_data = cross.get("us10y_data") or {}
    macro = {
        "combined_bias": str(bias.get("combined_bias", "neutral")),
        "confidence": str(bias.get("confidence", "low")),
        "implication": str(bias.get("usdjpy_implication", "")),
        "dxy": {
            "value": cross.get("dxy") or dxy_bias.get("value"),
            "one_day": dxy_bias.get("1d_return"),
            "five_day": dxy_bias.get("5d_return"),
        },
        "us10y": {
            "value": us10y_data.get("value") or cross.get("us10y_yield"),
            "one_day": us10y_data.get("1d_change"),
            "five_day": us10y_data.get("5d_change"),
        },
        "oil": {
            "value": cross.get("bco_usd") or oil_bias.get("price"),
            "one_day": oil_bias.get("1d_return"),
            "five_day": oil_bias.get("5d_return"),
        },
        "gold": {
            "value": cross.get("xau_usd") or gold_bias.get("price"),
            "one_day": gold_bias.get("1d_return"),
            "five_day": gold_bias.get("5d_return"),
        },
    }

    session = ctx.get("session") or {}
    vol = ctx.get("volatility") or {}
    events = get_economic_calendar_events(days_ahead=days_ahead, limit=3)

    return {
        "as_of": ctx.get("as_of"),
        "macro": macro,
        "events": events,
        "levels": {
            "mid": round(mid, 3) if mid > 0 else None,
            "supports": _level_rows(supports, "support"),
            "resistances": _level_rows(resistances, "resistance"),
        },
        "session_vol": {
            "active_sessions": list(session.get("active_sessions") or []),
            "overlap": session.get("overlap"),
            "next_close": session.get("next_close"),
            "warnings": list(session.get("warnings") or []),
            "spread_pips": spot.get("spread_pips"),
            "vol_label": vol.get("label"),
            "vol_ratio": vol.get("ratio"),
            "recent_avg_pips": vol.get("recent_avg_pips"),
        },
    }


@app.post("/api/data/{profile_name}/ai-chat")
def ai_chat(profile_name: str, profile_path: str, req: AiChatRequest):
    """Streaming AI chat endpoint — SSE delta/done contract."""
    # --- Validate API key ---
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not configured on the server")

    # --- Validate message ---
    user_message = req.message.strip() if req.message else ""
    if not user_message:
        raise HTTPException(status_code=400, detail="message must be a non-empty string")

    # --- Validate history roles ---
    allowed_roles = {"user", "assistant"}
    for entry in req.history:
        if entry.role not in allowed_roles:
            raise HTTPException(status_code=400, detail=f"Invalid role in history: {entry.role!r}")

    # Cap history to last 10 pairs (20 messages)
    history_dicts = [{"role": h.role, "content": h.content} for h in req.history[-20:]]

    # --- Resolve profile ---
    path = _resolve_profile_path(profile_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Profile not found")

    try:
        profile = load_profile_v1(path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"API fetch error (profile): {e}")

    # --- Build trading context (blocking broker call in thread pool) ---
    from api.ai_trading_chat import (
        build_trading_context,
        resolve_ai_chat_model,
        system_prompt_from_context,
        stream_openai_chat,
    )

    try:
        ctx_timeout = float(os.environ.get("API_AI_CHAT_CTX_TIMEOUT_SEC", "20"))
    except ValueError:
        ctx_timeout = 20.0

    try:
        ctx = _run_in_threadpool_with_timeout(build_trading_context, ctx_timeout, profile, profile_name)
    except FuturesTimeoutError:
        raise HTTPException(status_code=502, detail="API fetch error (broker context timed out)")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"API fetch error (broker context): {e}")

    try:
        effective_model = resolve_ai_chat_model(req.chat_model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    system = system_prompt_from_context(ctx, effective_model)

    # --- Stream OpenAI response ---
    def event_stream():
        try:
            yield from stream_openai_chat(
                system=system,
                user_message=user_message,
                history=history_dicts,
                model=effective_model,
                profile=profile,
                profile_name=profile_name,
            )
        except Exception as e:
            import json as _json
            yield f"data: {_json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


# Catch-all for SPA routing (must be last so API routes match first)
@app.get("/{full_path:path}")
def serve_spa(full_path: str):
    """Serve index.html for SPA client-side routing."""
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not found")
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    raise HTTPException(status_code=404, detail="Frontend not built")
