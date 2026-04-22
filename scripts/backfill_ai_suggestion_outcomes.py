from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api import suggestion_tracker

LOGS_DIR = ROOT / "logs"


def main() -> int:
    db_paths = sorted(LOGS_DIR.glob("*/ai_suggestions.sqlite"))
    if not db_paths:
        print("No ai_suggestions.sqlite files found under logs/.")
        return 0

    grand_total = 0
    grand_updated = 0
    grand_skipped = 0
    for db_path in db_paths:
        result = suggestion_tracker.backfill_null_win_loss(db_path)
        grand_total += int(result.get("total") or 0)
        grand_updated += int(result.get("updated") or 0)
        grand_skipped += int(result.get("skipped") or 0)
        print(
            f"{db_path.parent.name}: total={result.get('total', 0)} "
            f"updated={result.get('updated', 0)} skipped={result.get('skipped', 0)}"
        )
    print(
        f"Backfill complete: total={grand_total} updated={grand_updated} skipped={grand_skipped}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
