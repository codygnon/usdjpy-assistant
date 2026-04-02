from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import RunManifest


def freeze_manifest(manifest: RunManifest, path: Path) -> dict[str, Any]:
    payload = manifest.model_dump()
    payload["committed_at_utc"] = datetime.now(timezone.utc).isoformat()
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def evaluate_summary_against_manifest(summary: dict[str, Any], manifest: RunManifest) -> dict[str, Any]:
    trade_count = int(summary.get("trade_count") or 0)
    profit_factor = float(summary.get("profit_factor") or 0.0)
    win_rate = float(summary.get("win_rate") or 0.0)
    max_drawdown_usd = float(summary.get("max_drawdown_usd") or 0.0)
    max_drawdown_pct = float(summary.get("max_drawdown_pct") or 0.0)

    checks = {
        "minimum_trade_count": {
            "expected": int(manifest.minimum_trade_count),
            "actual": trade_count,
            "passed": trade_count >= manifest.minimum_trade_count,
        },
        "minimum_profit_factor": {
            "expected": float(manifest.minimum_profit_factor),
            "actual": profit_factor,
            "passed": profit_factor >= manifest.minimum_profit_factor,
        },
        "expected_win_rate_range": {
            "expected": [float(manifest.expected_win_rate_min), float(manifest.expected_win_rate_max)],
            "actual": win_rate,
            "passed": manifest.expected_win_rate_min <= win_rate <= manifest.expected_win_rate_max,
        },
    }

    if manifest.maximum_drawdown_usd is not None:
        checks["maximum_drawdown_usd"] = {
            "expected": float(manifest.maximum_drawdown_usd),
            "actual": max_drawdown_usd,
            "passed": max_drawdown_usd <= manifest.maximum_drawdown_usd,
        }
    if manifest.maximum_drawdown_pct is not None:
        checks["maximum_drawdown_pct"] = {
            "expected": float(manifest.maximum_drawdown_pct),
            "actual": max_drawdown_pct,
            "passed": max_drawdown_pct <= manifest.maximum_drawdown_pct,
        }

    return {
        "hypothesis": manifest.hypothesis,
        "overall_pass": all(check["passed"] for check in checks.values()),
        "checks": checks,
    }
