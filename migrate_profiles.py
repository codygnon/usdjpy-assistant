from __future__ import annotations

import argparse
from pathlib import Path

from core.profile import load_profile_v1, save_profile_v1


def main() -> None:
    ap = argparse.ArgumentParser(description="Migrate flat profiles to v1 schema safely.")
    ap.add_argument("--profiles-dir", default=str(Path(__file__).resolve().parent / "profiles"))
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "profiles" / "v1"))
    args = ap.parse_args()

    profiles_dir = Path(args.profiles_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for p in sorted(profiles_dir.glob("*.json")):
        try:
            prof = load_profile_v1(p)
        except Exception as e:
            print(f"❌ Failed to migrate {p.name}: {e}")
            continue

        out_path = out_dir / p.name
        save_profile_v1(prof, out_path)
        print(f"✅ Migrated {p.name} -> {out_path}")
        count += 1

    if count == 0:
        print("No profiles migrated.")


if __name__ == "__main__":
    main()

