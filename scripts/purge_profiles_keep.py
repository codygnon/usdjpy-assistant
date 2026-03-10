#!/usr/bin/env python3
"""
Purge profile JSON files and log/DB directories, keeping only named profiles.

Sidebar "Remove account" only deletes the profile JSON; logs/<name>/ (assistant.db,
loop logs, etc.) remain. This script removes:
  - Any profiles/*.json whose stem is not in the keep list
  - Any logs/<name>/ directory whose name is not in the keep list

Uses the same data root as the API:
  RAILWAY_VOLUME_MOUNT_PATH or USDJPY_DATA_DIR, else repo base.

Usage (dry-run — lists what would be removed):
  python scripts/purge_profiles_keep.py

Actually delete:
  python scripts/purge_profiles_keep.py --execute

On Railway (from project root, with volume mounted):
  railway run python scripts/purge_profiles_keep.py --execute

Or open a shell on the service and run the same command there.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# Match api/main.py data root
BASE_DIR = Path(__file__).resolve().parent.parent
_data_base_env = os.environ.get("RAILWAY_VOLUME_MOUNT_PATH") or os.environ.get("USDJPY_DATA_DIR")
DATA_BASE = Path(_data_base_env) if _data_base_env else BASE_DIR
PROFILES_DIR = DATA_BASE / "profiles"
LOGS_DIR = DATA_BASE / "logs"

# Profiles to keep (lowercase stems / directory names)
DEFAULT_KEEP = frozenset({"kumatora2", "newera8"})


def main() -> int:
    parser = argparse.ArgumentParser(description="Purge profiles except named keep set.")
    parser.add_argument(
        "--keep",
        nargs="*",
        default=sorted(DEFAULT_KEEP),
        help=f"Profile names to keep (default: {sorted(DEFAULT_KEEP)})",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete; without this, only prints what would be removed.",
    )
    args = parser.parse_args()
    keep = frozenset(n.strip().lower() for n in args.keep if n.strip())

    print(f"DATA_BASE:  {DATA_BASE}")
    print(f"PROFILES:   {PROFILES_DIR}")
    print(f"LOGS_DIR:   {LOGS_DIR}")
    print(f"Keep only:  {sorted(keep)}")
    print(f"Mode:       {'DELETE' if args.execute else 'DRY-RUN (no changes)'}")
    print()

    to_delete_profiles: list[Path] = []
    if PROFILES_DIR.exists():
        for p in sorted(PROFILES_DIR.rglob("*.json")):
            if not p.is_file():
                continue
            stem = p.stem.lower()
            if stem not in keep:
                to_delete_profiles.append(p)

    to_delete_log_dirs: list[Path] = []
    if LOGS_DIR.exists():
        for p in sorted(LOGS_DIR.iterdir()):
            if not p.is_dir():
                continue
            name = p.name.lower()
            if name not in keep:
                to_delete_log_dirs.append(p)

    if not to_delete_profiles and not to_delete_log_dirs:
        print("Nothing to remove — no extra profiles or log dirs found.")
        return 0

    if to_delete_profiles:
        print("Profile JSON files to remove:")
        for p in to_delete_profiles:
            print(f"  FILE  {p}")
    if to_delete_log_dirs:
        print("Log/DB directories to remove (assistant.db, loop.log, etc.):")
        for p in to_delete_log_dirs:
            print(f"  DIR   {p}")

    if not args.execute:
        print()
        print("Re-run with --execute to delete the above.")
        return 0

    for p in to_delete_profiles:
        try:
            p.unlink()
            print(f"Deleted file: {p}")
        except OSError as e:
            print(f"Failed {p}: {e}", file=sys.stderr)

    for p in to_delete_log_dirs:
        try:
            shutil.rmtree(p)
            print(f"Deleted dir:  {p}")
        except OSError as e:
            print(f"Failed {p}: {e}", file=sys.stderr)

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
