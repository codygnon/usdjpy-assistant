"""FastAPI backend for USDJPY Assistant.

Run with: uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

# Track running loop processes: {profile_name: subprocess.Popen}
_loop_processes: dict[str, subprocess.Popen] = {}

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


def _is_loop_running(profile_name: str) -> bool:
    proc = _loop_processes.get(profile_name)
    if proc is None:
        return False
    if proc.poll() is None:
        return True
    # Process finished; remove from tracking
    del _loop_processes[profile_name]
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
        return {"status": "started", "pid": proc.pid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/loop/{profile_name}/stop")
def stop_loop(profile_name: str) -> dict[str, str]:
    """Stop the trading loop for a profile."""
    proc = _loop_processes.get(profile_name)
    if proc is None or proc.poll() is not None:
        return {"status": "not_running"}
    
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    
    del _loop_processes[profile_name]
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
) -> list[dict[str, Any]] | dict[str, Any]:
    """Get recent trades. If profile_path is provided, returns an object with
    trades (each including profit_display in display currency) and display_currency.

    Also syncs with broker first to detect any trades closed externally.
    """
    store = _store_for(profile_name)

    # Sync with broker to detect closed trades
    if profile_path and _resolve_profile_path(profile_path).exists():
        try:
            profile = load_profile_v1(_resolve_profile_path(profile_path))
            _sync_open_trades_with_broker(profile, store)
        except Exception as e:
            print(f"[api] sync error in get_trades: {e}")

    df = store.read_trades_df(profile_name).tail(limit)
    # Convert NaN to None for JSON
    df = df.where(pd.notna(df), None)
    records = df.to_dict(orient="records")

    if not profile_path or not _resolve_profile_path(profile_path).exists():
        return records

    try:
        profile = load_profile_v1(_resolve_profile_path(profile_path))
    except Exception:
        return records

    curr, rate = _get_display_currency(profile)
    symbol = str(getattr(profile, "symbol", "")).upper()

    def _compute_profit_usd(row: dict[str, Any]) -> float | None:
        """Best-effort per-trade profit estimate in USD from entry/exit/size."""
        try:
            entry_price = row.get("entry_price")
            exit_price = row.get("exit_price")
            size_lots = row.get("size_lots") or row.get("volume")
            side = (row.get("side") or "").lower()
            if entry_price is None or exit_price is None or size_lots is None or not side:
                return None
            entry = float(entry_price)
            exit_ = float(exit_price)
            lots = float(size_lots)
            if lots == 0:
                return None
            # Price difference in favor of the trade
            if side == "buy":
                diff = exit_ - entry
            elif side == "sell":
                diff = entry - exit_
            else:
                return None
            # For JPY-quoted pairs like USDJPY, approximate USD profit from price move
            if symbol.endswith("JPY") and exit_ > 0:
                # 1 standard lot = 100,000 base; convert JPY P/L back to USD
                profit_usd = diff * lots * 100_000.0 / exit_
                return round(profit_usd, 2)
        except Exception:
            return None
        return None

    for row in records:
        profit_raw = row.get("profit")
        profit_value: float | None = None
        if profit_raw is not None and not (isinstance(profit_raw, float) and pd.isna(profit_raw)):
            # Use stored profit when available (already in account currency, assumed USD)
            try:
                profit_value = float(profit_raw)
            except (TypeError, ValueError):
                profit_value = None
        if profit_value is None:
            profit_value = _compute_profit_usd(row)

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
                profit_raw = row.get("profit")
                profit_val = 0.0
                if profit_raw is not None and not (isinstance(profit_raw, float) and pd.isna(profit_raw)):
                    try:
                        profit_val = _convert_amount(float(profit_raw), rate) or 0.0
                    except (TypeError, ValueError):
                        pass
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
                profit_raw = float(row["profit"]) if pd.notna(row.get("profit")) else None

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
            return {"trades": trades_list, "display_currency": curr, "source": "database"}

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
                return {"trades": trades_list_broker, "display_currency": curr, "source": "broker"}
        except Exception as e:
            print(f"[api] trade-history-detail broker error: {e}")

    return {"trades": [], "display_currency": curr, "source": "database"}


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
def get_quick_stats(profile_name: str, profile_path: Optional[str] = None) -> dict[str, Any]:
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

    # Try broker report stats first (MT5 has them; OANDA returns None)
    if profile:
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
    profit_col = pd.to_numeric(closed.get("profit"), errors="coerce")

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

    profit_col_diag = pd.to_numeric(closed.get("profit"), errors="coerce")
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
        return {"source": None}
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
        return {"source": None}
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

    store = _store_for(profile_name)
    df = store.read_trades_df(profile_name)

    if df.empty:
        return {"presets": {}, "source": "database"}

    closed = df[pd.to_numeric(df.get("exit_price"), errors="coerce").notna()].copy() if "exit_price" in df.columns else pd.DataFrame()

    if closed.empty:
        return {"presets": {}, "source": "database"}

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
        p = row.get("profit")
        return float(p) if pd.notna(p) else None

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

    return {"presets": presets_data, "source": source, "display_currency": curr}


@app.get("/api/data/{profile_name}/technical-analysis")
def get_technical_analysis(profile_name: str, profile_path: str) -> dict[str, Any]:
    """Get real-time technical analysis for USDJPY across all timeframes (H4, M15, M1).
    
    Returns per-timeframe: regime, RSI value/zone, MACD line/signal/histogram, 
    ATR value/state, price info, and a plain-English summary.
    """
    from core.indicators import ema
    from core.ta_analysis import compute_ta_for_tf
    from core.timeframes import Timeframe
    
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
            
            for tf in timeframes:
                try:
                    # Fetch OHLC data from broker (MT5 or OANDA)
                    df = adapter.get_bars(profile.symbol, tf, count=700)
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
                    # Build OHLC array for chart (last 500 bars)
                    df_tail = df.tail(500)
                    ohlc_data = []
                    for _, row in df_tail.iterrows():
                        ohlc_data.append({
                            "time": int(row["time"].timestamp()),
                            "open": round(float(row["open"]), 3),
                            "high": round(float(row["high"]), 3),
                            "low": round(float(row["low"]), 3),
                            "close": round(float(row["close"]), 3),
                        })
                    # Build all 10 EMA series for chart overlay
                    close = df["close"]
                    all_emas_arrs: dict[str, list[dict[str, Any]]] = {}
                    for p in [5, 7, 9, 11, 13, 15, 17, 21, 50, 200]:
                        s = ema(close, p)
                        all_emas_arrs[f"ema{p}"] = [{"time": int(row["time"].timestamp()), "value": round(float(s.loc[row.name]), 3)} for _, row in df_tail.iterrows() if row.name in s.index and not pd.isna(s.loc[row.name])]
                    # Build Bollinger Bands series for chart overlay
                    from core.indicators import bollinger_bands
                    bb_upper, bb_middle, bb_lower = bollinger_bands(close, 20, 2.0)
                    bb_series: dict[str, list[dict[str, Any]]] = {"upper": [], "middle": [], "lower": []}
                    for _, row in df_tail.iterrows():
                        ts = int(row["time"].timestamp())
                        idx = row.name
                        if idx in bb_upper.index and not pd.isna(bb_upper.loc[idx]):
                            bb_series["upper"].append({"time": ts, "value": round(float(bb_upper.loc[idx]), 3)})
                            bb_series["middle"].append({"time": ts, "value": round(float(bb_middle.loc[idx]), 3)})
                            bb_series["lower"].append({"time": ts, "value": round(float(bb_lower.loc[idx]), 3)})
                    result["timeframes"][tf] = {
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

            # Get current tick for spread info
            tick = adapter.get_tick(profile.symbol)
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

    # Lazily backfill MAE/MFE from candle data for historical trades
    if profile:
        try:
            _backfill_mae_mfe(profile, store, profile_name)
        except Exception:
            pass

    df = store.read_trades_df(profile_name)
    if df.empty or "exit_price" not in df.columns:
        return {"trades": [], "display_currency": curr, "source": "database", "starting_balance": None, "total_profit_currency": None}

    closed = df[pd.to_numeric(df["exit_price"], errors="coerce").notna()].copy()
    if closed.empty:
        return {"trades": [], "display_currency": curr, "source": "database", "starting_balance": None, "total_profit_currency": None}

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

        profit_raw = float(row["profit"]) if pd.notna(row.get("profit")) else None
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
            "preset_name": str(row.get("preset_name") or ""),
            "exit_reason": str(row.get("exit_reason") or ""),
        })

    total_profit_currency = None
    profits = [t.get("profit") for t in trades_list if t.get("profit") is not None]
    if profits:
        total_profit_currency = round(sum(profits), 2)

    return {
        "trades": trades_list,
        "display_currency": curr,
        "source": "database",
        "starting_balance": starting_balance,
        "total_profit_currency": total_profit_currency,
    }


# ---------------------------------------------------------------------------
# Endpoints: Trade management (close, sync)
# ---------------------------------------------------------------------------


@app.get("/api/data/{profile_name}/open-trades")
def get_open_trades(profile_name: str, profile_path: Optional[str] = None) -> list[dict[str, Any]]:
    """Get open trades (trades without exit_price).

    Syncs with broker first to detect any trades closed externally.
    """
    store = _store_for(profile_name)

    # Sync with broker to detect closed trades
    if profile_path:
        try:
            profile = load_profile_v1(profile_path)
            _sync_open_trades_with_broker(profile, store)
        except Exception as e:
            print(f"[api] sync error in open-trades: {e}")

    rows = store.list_open_trades(profile_name)
    return [dict(row) for row in rows]


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

        # Determine exit reason
        exit_reason = "broker_closed"
        if target_price and abs(exit_price - float(target_price)) <= pip_size * 2:
            exit_reason = "hit_take_profit"
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

        # Determine exit reason
        exit_reason = "broker_closed_sync"
        if target_price and abs(exit_price - float(target_price)) <= pip_size * 2:
            exit_reason = "hit_take_profit"
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

        store.close_trade(trade_id=trade_id, updates=updates)
        print(f"[api] aggressive sync: closed {trade_id} with exit_reason={exit_reason}, pips={pips:.2f}")
        synced += 1

    try:
        adapter.shutdown()
    except Exception:
        pass

    return synced


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


# Catch-all for SPA routing (must be last)
@app.get("/{full_path:path}")
def serve_spa(full_path: str):
    """Serve index.html for SPA client-side routing."""
    # Don't intercept API routes
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not found")
    
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    raise HTTPException(status_code=404, detail="Frontend not built")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
