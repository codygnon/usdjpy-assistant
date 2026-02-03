"""Profile authentication module.

Provides optional password protection for profiles. Passwords are stored as bcrypt
hashes in logs/<profile_name>/auth.json. Profiles without auth.json are "non-secure"
and can be accessed without a password.
"""
from __future__ import annotations

import hashlib
import json
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path configuration - matches api/main.py (use same volume when set)
# ---------------------------------------------------------------------------
import os as _os
BASE_DIR = Path(__file__).resolve().parent.parent
_data_base_env = _os.environ.get("RAILWAY_VOLUME_MOUNT_PATH") or _os.environ.get("USDJPY_DATA_DIR")
LOGS_DIR = Path(_data_base_env) / "logs" if _data_base_env else BASE_DIR / "logs"


def _auth_path(profile_name: str) -> Path:
    """Return path to auth.json for a profile."""
    return LOGS_DIR / profile_name / "auth.json"


def _load_auth(profile_name: str) -> dict[str, Any] | None:
    """Load auth.json for a profile. Returns None if file doesn't exist."""
    path = _auth_path(profile_name)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _save_auth(profile_name: str, data: dict[str, Any]) -> bool:
    """Save auth.json for a profile. Creates logs/<profile_name>/ if needed."""
    path = _auth_path(profile_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except OSError:
        return False


def has_password(profile_name: str) -> bool:
    """Check if a profile has a password set.
    
    Returns True if auth.json exists and contains a password_hash.
    """
    auth_data = _load_auth(profile_name)
    if auth_data is None:
        return False
    return bool(auth_data.get("password_hash"))


def _hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    """Hash a password using PBKDF2-SHA256.
    
    Returns (hash, salt). If salt is provided, uses that salt.
    """
    if salt is None:
        salt = secrets.token_hex(16)
    # Use PBKDF2 with 100,000 iterations
    dk = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
    return dk.hex(), salt


def _verify_hash(password: str, stored_hash: str, salt: str) -> bool:
    """Verify a password against stored hash and salt."""
    computed_hash, _ = _hash_password(password, salt)
    return secrets.compare_digest(computed_hash, stored_hash)


def verify_password(profile_name: str, password: str) -> bool:
    """Verify a password against the stored hash.
    
    Returns False if no password is set or password doesn't match.
    """
    auth_data = _load_auth(profile_name)
    if auth_data is None:
        return False
    stored_hash = auth_data.get("password_hash")
    salt = auth_data.get("salt")
    if not stored_hash or not salt:
        return False
    try:
        return _verify_hash(password, stored_hash, salt)
    except Exception:
        return False


def set_password(
    profile_name: str,
    new_password: str,
    current_password: str | None = None,
) -> tuple[bool, str]:
    """Set or update a password for a profile.
    
    If the profile already has a password, current_password must be provided
    and must verify against the existing hash.
    
    Returns (success, error_message). error_message is empty on success.
    """
    if not new_password or len(new_password) < 4:
        return False, "Password must be at least 4 characters"
    
    # Check if profile already has a password
    if has_password(profile_name):
        if current_password is None:
            return False, "Current password required"
        if not verify_password(profile_name, current_password):
            return False, "Current password is incorrect"
    
    # Hash the new password and save
    password_hash, salt = _hash_password(new_password)
    auth_data = {
        "password_hash": password_hash,
        "salt": salt,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    
    if _save_auth(profile_name, auth_data):
        return True, ""
    return False, "Failed to save password"


def remove_password(profile_name: str, password: str) -> tuple[bool, str]:
    """Remove password protection from a profile.
    
    Requires the current password to verify before removing.
    Returns (success, error_message).
    """
    if not has_password(profile_name):
        return False, "Profile has no password"
    
    if not verify_password(profile_name, password):
        return False, "Incorrect password"
    
    # Delete the auth.json file
    path = _auth_path(profile_name)
    try:
        path.unlink()
        return True, ""
    except OSError as e:
        return False, f"Failed to remove password: {e}"
