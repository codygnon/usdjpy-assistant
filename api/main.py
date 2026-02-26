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
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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
from core.presets import PresetId, apply_preset, get_preset_patch, list_presets
from core.profile import ProfileV1, ProfileV1AllowExtra, default_profile_for_name, load_profile_v1, save_profile_v1
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

# Short-lived endpoint response cache for expensive read endpoints.
_endpoint_response_cache: dict[str, tuple[float, Any]] = {}
_ENDPOINT_CACHE_TTL_SECONDS: dict[str, float] = {
    "trade_history_detail": 10.0,
    "advanced_analytics": 10.0,
    "stats_by_preset": 10.0,
    "mt5_report": 10.0,
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _list_profile_paths() -> list[Path]:
    if not PROFILES_DIR.exists():
        return []
    return sorted([p for p in PROFILES_DIR.rglob("*.json") if p.is_file()])


def _store_for(profile_name: str) -> SqliteStore:
    log_dir = LOGS_DIR / profile_name
    log_dir.mkdir(parents=True, exist_ok=True)
    store = SqliteStore(log_dir / "assistant.db")
    store.init_db()
    return store


def _runtime_state_path(profile_name: str) -> Path:
    return LOGS_DIR / profile_name / "runtime_state.json"


def _loop_pid_path(profile_name: str) -> Path:
    return LOGS_DIR / profile_name / "loop.pid"


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

    abs_raw = abs(raw)
    abs_est = abs(est)
    if abs_raw >= 250 and abs_est >= 5:
        ratio = abs_raw / abs_est if abs_est > 0 else float("inf")
        if ratio >= 6.0:
            return est
    if abs_raw >= 250 and abs_est >= 20 and (raw * est) < 0:
        return est
    return raw


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


class TempEmaSettingsUpdate(BaseModel):
    m5_trend_ema_fast: Optional[int] = None
    m5_trend_ema_slow: Optional[int] = None
    m1_zone_entry_ema_slow: Optional[int] = None
    m1_pullback_cross_ema_slow: Optional[int] = None
    # Trial #4 fields (Zone Entry only - Tiered Pullback uses fixed tiers)
    m3_trend_ema_fast: Optional[int] = None
    m3_trend_ema_slow: Optional[int] = None
    m1_t4_zone_entry_ema_fast: Optional[int] = None
    m1_t4_zone_entry_ema_slow: Optional[int] = None


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
    paths = _list_profile_paths()
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
    new_state = RuntimeState(
        mode=req.mode,  # type: ignore
        kill_switch=req.kill_switch,
        last_processed_bar_time_utc=old.last_processed_bar_time_utc,
        temp_m5_trend_ema_fast=old.temp_m5_trend_ema_fast,
        temp_m5_trend_ema_slow=old.temp_m5_trend_ema_slow,
        temp_m1_zone_entry_ema_slow=old.temp_m1_zone_entry_ema_slow,
        temp_m1_pullback_cross_ema_slow=old.temp_m1_pullback_cross_ema_slow,
        tier_fired=old.tier_fired,
        divergence_block_buy_until=old.divergence_block_buy_until,
        divergence_block_sell_until=old.divergence_block_sell_until,
        daily_reset_date=old.daily_reset_date,
        daily_reset_high=old.daily_reset_high,
        daily_reset_low=old.daily_reset_low,
        daily_reset_block_active=old.daily_reset_block_active,
        daily_reset_settled=old.daily_reset_settled,
        trend_flip_price=old.trend_flip_price,
        trend_flip_direction=old.trend_flip_direction,

    )
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
        "m1_zone_entry_ema_slow": state.temp_m1_zone_entry_ema_slow,
        "m1_pullback_cross_ema_slow": state.temp_m1_pullback_cross_ema_slow,
        "m3_trend_ema_fast": state.temp_m3_trend_ema_fast,
        "m3_trend_ema_slow": state.temp_m3_trend_ema_slow,
        "m1_t4_zone_entry_ema_fast": state.temp_m1_t4_zone_entry_ema_fast,
        "m1_t4_zone_entry_ema_slow": state.temp_m1_t4_zone_entry_ema_slow,
    }


@app.put("/api/runtime/{profile_name}/temp-settings")
def update_temp_settings(profile_name: str, req: TempEmaSettingsUpdate) -> dict[str, str]:
    """Update temporary EMA settings for Apply Temporary Settings menu."""
    state_path = _runtime_state_path(profile_name)
    old = load_state(state_path)
    new_state = RuntimeState(
        mode=old.mode,
        kill_switch=old.kill_switch,
        last_processed_bar_time_utc=old.last_processed_bar_time_utc,
        temp_m5_trend_ema_fast=req.m5_trend_ema_fast,
        temp_m5_trend_ema_slow=req.m5_trend_ema_slow,
        temp_m1_zone_entry_ema_slow=req.m1_zone_entry_ema_slow,
        temp_m1_pullback_cross_ema_slow=req.m1_pullback_cross_ema_slow,
        temp_m3_trend_ema_fast=req.m3_trend_ema_fast,
        temp_m3_trend_ema_slow=req.m3_trend_ema_slow,
        temp_m1_t4_zone_entry_ema_fast=req.m1_t4_zone_entry_ema_fast,
        temp_m1_t4_zone_entry_ema_slow=req.m1_t4_zone_entry_ema_slow,
        # Preserve tier state (not modified through temp settings API)
        tier_fired=old.tier_fired,
        # Preserve divergence block state (not modified through temp settings API)
        divergence_block_buy_until=old.divergence_block_buy_until,
        divergence_block_sell_until=old.divergence_block_sell_until,
        # Preserve daily reset state (not modified through temp settings API)
        daily_reset_date=old.daily_reset_date,
        daily_reset_high=old.daily_reset_high,
        daily_reset_low=old.daily_reset_low,
        daily_reset_block_active=old.daily_reset_block_active,
        daily_reset_settled=old.daily_reset_settled,
        # Preserve exhaustion state (not modified through temp settings API)
        trend_flip_price=old.trend_flip_price,
        trend_flip_direction=old.trend_flip_direction,
        trend_flip_time=old.trend_flip_time,
        # Preserve BB tier state (Trial #6)

    )
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
    
    try:
        log_file = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, "-u", str(BASE_DIR / "run_loop.py"), "--profile", str(path)],
            cwd=str(BASE_DIR),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        _loop_processes[profile_name] = proc
        pid_path = _loop_pid_path(profile_name)
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        pid_path.write_text(str(proc.pid), encoding="utf-8")
        return {"status": "started", "pid": proc.pid}
    except Exception as e:
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
    records = df.to_dict(orient="records")

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


@app.get("/api/data/{profile_name}/stats")
def get_quick_stats(profile_name: str, profile_path: Optional[str] = None, sync: bool = False) -> dict[str, Any]:
    """Get quick stats (win rate, avg pips, total profit).
    
    Prefers broker (MT5/OANDA) deal history as source of truth when available.
    Falls back to local database when broker is not running or OANDA (no report stats).
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

    # Try broker report stats only when explicitly requested.
    if sync and profile:
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
            return {
                "source": "mt5",
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

    source = "mt5" if mt5_financials else "database"
    curr, rate = _get_display_currency(profile) if profile else ("USD", 1.0)
    presets_data: dict[str, Any] = {}

    for preset_name, group in closed.groupby("preset_name"):
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

        total_profit = None
        if profit_use.notna().any():
            total_profit = round(float(profit_use.sum()), 2)
        total_commission = round(float(commission_col.sum()), 2)

        best_trade = round(float(pips.max()), 2) if pips.notna().any() else None
        worst_trade = round(float(pips.min()), 2) if pips.notna().any() else None

        win_streak, loss_streak = 0, 0
        current_win_streak, current_loss_streak = 0, 0
        for _, row in group.iterrows():
            w = _is_win(row)
            l = _is_loss(row)
            if w:
                current_win_streak += 1
                current_loss_streak = 0
                win_streak = max(win_streak, current_win_streak)
            elif l:
                current_loss_streak += 1
                current_win_streak = 0
                loss_streak = max(loss_streak, current_loss_streak)
            else:
                current_win_streak = 0
                current_loss_streak = 0

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

        presets_data[str(preset_name)] = {
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

    payload = {"presets": presets_data, "source": source, "display_currency": curr}
    _cache_set("stats_by_preset", cache_key, payload)
    return payload


@app.get("/api/data/{profile_name}/technical-analysis")
def get_technical_analysis(profile_name: str, profile_path: str) -> dict[str, Any]:
    """Get real-time technical analysis for USDJPY across all timeframes (H4, M15, M1).
    
    Returns per-timeframe: regime, RSI value/zone, MACD line/signal/histogram, 
    ATR value/state, price info, and a plain-English summary.
    """
    from core.indicators import ema
    from core.ta_analysis import compute_ta_for_tf
    from core.timeframes import Timeframe
    from core.book_cache import get_book_cache
    from core.scalp_score import ScalpScore

    path = _resolve_profile_path(profile_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Profile not found")

    try:
        profile = load_profile_v1(path)
        from adapters.broker import get_adapter
        adapter = get_adapter(profile)
        try:
            adapter.initialize()
        except RuntimeError as init_err:
            msg = str(init_err)
            if "MetaTrader5 is not installed" in msg or "MT5" in msg:
                raise HTTPException(
                    status_code=503,
                    detail="Broker data unavailable: MT5 is not installed on this server. Set Broker to OANDA in Profile Editor and add your API key to view technical analysis here."
                ) from init_err
            raise
        try:
            adapter.ensure_symbol(profile.symbol)
            # Timeframe is a Literal type, not an Enum - use string values directly
            timeframes: list[Timeframe] = ["H4", "H1", "M30", "M15", "M5", "M3", "M1"]
            result: dict[str, Any] = {"timeframes": {}}

            # Poll order/position books for scalp score (deduplicates via cache)
            book_cache = get_book_cache()
            try:
                book_cache.poll_books(adapter, profile.symbol)
            except Exception:
                pass  # Books are optional; score degrades gracefully

            # Get tick once for scalp score + spread info
            scalp_tick = None
            try:
                scalp_tick = adapter.get_tick(profile.symbol)
            except Exception:
                pass
            
            for tf in timeframes:
                try:
                    # Fetch OHLC data from broker (cached)
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
                    # Compute TA for this timeframe
                    ta = compute_ta_for_tf(profile, tf, df)
                    # Format MACD direction
                    macd_direction = "neutral"
                    if ta.macd_hist is not None:
                        if ta.macd_hist > 0:
                            macd_direction = "positive"
                        elif ta.macd_hist < 0:
                            macd_direction = "negative"
                    # Build OHLC array for chart (last 500 bars) - vectorized
                    df_tail = df.tail(500)
                    timestamps = df_tail["time"].apply(lambda t: int(t.timestamp())).values
                    ohlc_data = [
                        {"time": int(ts), "open": round(float(o), 3), "high": round(float(h), 3), "low": round(float(l), 3), "close": round(float(c), 3)}
                        for ts, o, h, l, c in zip(timestamps, df_tail["open"].values, df_tail["high"].values, df_tail["low"].values, df_tail["close"].values)
                    ]
                    # Build all 10 EMA series for chart overlay - vectorized
                    close = df["close"]
                    tail_idx = df_tail.index
                    all_emas_arrs: dict[str, list[dict[str, Any]]] = {}
                    for p in [5, 7, 9, 11, 13, 15, 17, 21, 34, 50, 200]:
                        s = ema(close, p)
                        s_tail = s.reindex(tail_idx).dropna()
                        ts_arr = df_tail.loc[s_tail.index, "time"].apply(lambda t: int(t.timestamp()))
                        all_emas_arrs[f"ema{p}"] = [{"time": int(t), "value": round(float(v), 3)} for t, v in zip(ts_arr.values, s_tail.values)]
                    # Build Bollinger Bands series for chart overlay - vectorized
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
                    # Compute scalp score for M1/M3/M5
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
                    # Handle errors for individual timeframes gracefully
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

            # Reuse tick from scalp score (no second API call)
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
            adapter.shutdown()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TA error: {str(e)}")


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
        })

    total_profit_currency = None
    profits = [t.get("profit") for t in trades_list if t.get("profit") is not None]
    if profits:
        total_profit_currency = round(sum(profits), 2)

    payload = {
        "trades": trades_list,
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


@app.get("/api/data/{profile_name}/open-trades")
def get_open_trades(profile_name: str, profile_path: Optional[str] = None, sync: bool = False) -> list[dict[str, Any]]:
    """Get open trades (trades without exit_price).

    Syncs with broker first to detect any trades closed externally.
    Also includes live unrealized_pl and financing from broker.
    """
    store = _store_for(profile_name)

    broker_live: dict[int, dict] = {}

    if profile_path and sync:
        loaded_profile = None
        try:
            loaded_profile = load_profile_v1(profile_path)
            _sync_open_trades_with_broker(loaded_profile, store)
        except Exception as e:
            print(f"[api] sync error in open-trades: {e}")

        if loaded_profile is not None:
            try:
                from adapters.broker import get_adapter
                adapter = get_adapter(loaded_profile)
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
                try:
                    adapter.shutdown()
                except Exception:
                    pass
            except Exception as e:
                print(f"[api] live data fetch error in open-trades: {e}")

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
    return result


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


def _build_live_dashboard_state(profile_name: str, profile_path: Optional[str] = None) -> dict[str, Any]:
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

    runtime = load_state(_runtime_state_path(profile_name))
    now_utc = datetime.now(timezone.utc)
    pip_size = float(profile.pip_size)

    bid = ask = spread_pips = 0.0
    positions: list[dict] = []
    filters: list[dict] = []
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
                age = (now_utc - t0.to_pydatetime()).total_seconds() / 60.0
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
            for pol in getattr(profile.execution, "policies", []) or []:
                if not getattr(pol, "enabled", True):
                    continue
                pt = getattr(pol, "type", "") or ""
                if pt in ("kt_cg_trial_4", "kt_cg_trial_5", "kt_cg_trial_6", "kt_cg_trial_7"):
                    _policy = pol
                    _policy_type = pt
                    break
            data_by_tf: dict = {}
            daily_reset_state: Optional[dict] = None
            exhaustion_state: Optional[dict] = None
            divergence_state: Optional[dict] = None
            exhaustion_result: Optional[dict] = None
            temp_overrides_api: Optional[dict] = None
            if _policy is not None:
                try:
                    _adapter = _get_dashboard_adapter(profile)
                    data_by_tf["M1"] = _get_bars_cached(_adapter, profile.symbol, "M1", 3000)
                    if _policy_type in ("kt_cg_trial_4", "kt_cg_trial_5", "kt_cg_trial_6"):
                        data_by_tf["M3"] = _get_bars_cached(_adapter, profile.symbol, "M3", 3000)
                    if _policy_type in ("kt_cg_trial_4", "kt_cg_trial_5"):
                        data_by_tf["D"] = _get_bars_cached(_adapter, profile.symbol, "D", 2)
                    if _policy_type == "kt_cg_trial_7":
                        data_by_tf["M5"] = _get_bars_cached(_adapter, profile.symbol, "M5", 2000)
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
                    if _state.temp_m1_zone_entry_ema_slow is not None:
                        temp_overrides_api["m1_zone_entry_ema_slow"] = _state.temp_m1_zone_entry_ema_slow
                    if _state.temp_m1_pullback_cross_ema_slow is not None:
                        temp_overrides_api["m1_pullback_cross_ema_slow"] = _state.temp_m1_pullback_cross_ema_slow
                    if not temp_overrides_api:
                        temp_overrides_api = None
                    if _policy_type == "kt_cg_trial_5" and getattr(_policy, "trend_exhaustion_enabled", False):
                        m3_df = data_by_tf.get("M3")
                        if m3_df is not None and not m3_df.empty:
                            from core.execution_engine import _detect_trend_flip_and_compute_exhaustion
                            mid = (_tick.bid + _tick.ask) / 2.0
                            exhaustion_result = _detect_trend_flip_and_compute_exhaustion(
                                m3_df, mid, pip_size, exhaustion_state or {}, _policy
                            )
                    if _policy_type == "kt_cg_trial_7" and getattr(_policy, "trend_exhaustion_enabled", False):
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
            # Compute Trial #6 M3 trend for filter display
            t6_eval_result = None
            if _policy_type == "kt_cg_trial_6" and _policy is not None:
                try:
                    m3_df = data_by_tf.get("M3")
                    if m3_df is not None and not m3_df.empty:
                        from core.execution_engine import _evaluate_m3_slope_trend_trial_6
                        trend_result = _evaluate_m3_slope_trend_trial_6(m3_df, _policy, pip_size)
                        t6_eval_result = {"trend_result": trend_result}
                except Exception:
                    pass
            store = _store_for(profile_name)
            filter_reports = build_dashboard_filters(
                profile=profile,
                tick=_tick,
                data_by_tf=data_by_tf,
                policy=_policy,
                policy_type=_policy_type,
                eval_result=t6_eval_result,
                divergence_state=divergence_state,
                daily_reset_state=daily_reset_state,
                exhaustion_result=exhaustion_result,
                store=store,
                adapter=_adapter if '_adapter' in locals() else None,
                temp_overrides=temp_overrides_api,
            )
            filters.extend(asdict(f) for f in filter_reports)
    except Exception as e:
        print(f"[api] dashboard filters error for '{profile_name}': {e}")

    # --- Daily summary from store ---
    daily_summary = None
    try:
        store = _store_for(profile_name)
        date_str = now_utc.strftime("%Y-%m-%d")
        closed_today = store.get_trades_for_date(profile_name, date_str)
        trades_today = len(closed_today)
        wins = losses = 0
        total_pips = total_profit = 0.0
        for row in closed_today:
            d = dict(row)
            pips = d.get("pips")
            profit = _normalized_trade_profit_usd(
                d,
                symbol_hint=str(getattr(profile, "symbol", "") or ""),
            )
            if pips is not None:
                total_pips += float(pips)
                if float(pips) > 0:
                    wins += 1
                else:
                    losses += 1
            if profit is not None:
                total_profit += float(profit)
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
        "context": [],
        "positions": positions,
        "daily_summary": daily_summary,
        "bid": bid,
        "ask": ask,
        "spread_pips": round(spread_pips, 1),
    }


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

    filters: dict[str, Any] = {}

    # Spread (from risk config)
    filters["spread"] = {"enabled": True, "max_pips": risk.max_spread_pips}

    # Session filter (from strategy filters)
    sf = profile.strategy.filters.session_filter
    filters["session_filter"] = {"enabled": sf.enabled, "sessions": sf.sessions}
    sbb = getattr(profile.strategy.filters, "session_boundary_block", None)
    if sbb is not None:
        filters["session_boundary_block"] = {"enabled": getattr(sbb, "enabled", False), "buffer_minutes": getattr(sbb, "buffer_minutes", 15)}

    # EMA Zone Filter
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
        if is_trial_7:
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


@app.get("/api/data/{profile_name}/dashboard")
def get_dashboard(profile_name: str, profile_path: Optional[str] = None) -> dict[str, Any]:
    """Return dashboard state with lean run-loop-file-first behavior."""
    from core.dashboard_models import read_dashboard_state

    if LEAN_UI_MODE:
        from datetime import datetime, timezone

        log_dir = _pick_best_dashboard_log_dir(profile_name, profile_path)
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
            result = dict(file_state)
            loop_running = _is_loop_running(profile_name)
            result["loop_running"] = loop_running
            result.setdefault("entry_candidate_side", None)
            result.setdefault("entry_candidate_trigger", None)
            # While loop is running, don't show stale — file may lag during slow iterations
            result["stale"] = False if loop_running else stale
            result["stale_age_seconds"] = 0.0 if loop_running else stale_age_seconds
            result["data_source"] = "run_loop_file"
            return result

        return {
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
        }

    # Legacy behavior (opt-in): live + file merge.
    now = _time.monotonic()
    cached = _dashboard_live_cache.get(profile_name)
    if cached and (now - cached[0]) < _DASHBOARD_LIVE_TTL:
        return cached[1]

    live = _build_live_dashboard_state(profile_name, profile_path)
    if "error" in live:
        return live

    log_dir = LOGS_DIR / profile_name
    file_state = read_dashboard_state(log_dir)

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

        if is_fresh:
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
    result["data_source"] = "run_loop_file"
    result.setdefault("entry_candidate_side", None)
    result.setdefault("entry_candidate_trigger", None)
    _dashboard_live_cache[profile_name] = (now, result)
    return result


@app.get("/api/data/{profile_name}/trade-events")
def get_trade_events(profile_name: str, limit: int = 50, profile_path: Optional[str] = None) -> list[dict[str, Any]]:
    """Returns trade_events.json contents (append-only trade log)."""
    from core.dashboard_models import read_trade_events

    log_dir = _pick_best_trade_events_log_dir(profile_name, profile_path)
    _backfill_trade_event_tier_labels(profile_name, log_dir)
    events = read_trade_events(log_dir, limit=limit)
    return _hydrate_trade_event_close_financials(profile_name, events)


def _hydrate_trade_event_close_financials(profile_name: str, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
        store = _store_for(profile_name)
        trades_df = store.read_trades_df(profile_name)
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
        store = _store_for(profile_name)
        execs = store.read_executions_df(profile_name)
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
