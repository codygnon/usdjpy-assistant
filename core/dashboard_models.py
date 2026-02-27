"""Data structures for the trading dashboard.

Provides serializable dataclasses for filter reports, context items, positions,
trade events, and the composite DashboardState that the run loop writes each poll.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


@dataclass
class FilterReport:
    filter_id: str
    display_name: str
    enabled: bool
    is_clear: bool
    current_value: str = ""
    threshold: str = ""
    block_reason: Optional[str] = None
    sub_filters: list["FilterReport"] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextItem:
    key: str
    value: str
    category: str = "general"


@dataclass
class PositionInfo:
    trade_id: str
    side: str
    entry_price: float
    size_lots: Optional[float] = None
    entry_type: Optional[str] = None
    current_price: float = 0.0
    unrealized_pips: float = 0.0
    age_minutes: float = 0.0
    stop_price: Optional[float] = None
    target_price: Optional[float] = None
    breakeven_applied: bool = False


@dataclass
class TradeEvent:
    event_type: str  # "open" or "close"
    timestamp_utc: str
    trade_id: str
    side: str
    entry_type: Optional[str] = None
    price: float = 0.0
    trigger_type: Optional[str] = None
    pips: Optional[float] = None
    profit: Optional[float] = None
    exit_reason: Optional[str] = None
    context_snapshot: dict[str, Any] = field(default_factory=dict)
    spread_at_entry: Optional[float] = None


@dataclass
class DailySummary:
    trades_today: int = 0
    wins: int = 0
    losses: int = 0
    total_pips: float = 0.0
    total_profit: float = 0.0
    win_rate: float = 0.0


@dataclass
class DashboardState:
    timestamp_utc: str
    preset_name: str = ""
    mode: str = ""
    loop_running: bool = True
    entry_candidate_side: Optional[str] = None
    entry_candidate_trigger: Optional[str] = None
    filters: list[FilterReport] = field(default_factory=list)
    context: list[ContextItem] = field(default_factory=list)
    positions: list[PositionInfo] = field(default_factory=list)
    daily_summary: Optional[DailySummary] = None
    bid: float = 0.0
    ask: float = 0.0
    spread_pips: float = 0.0


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _atomic_write(path: Path, data: str) -> None:
    """Write data atomically using tmp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    closed = False
    try:
        os.write(fd, data.encode("utf-8"))
        os.close(fd)
        closed = True
        os.replace(tmp, str(path))
    except Exception:
        if not closed:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_dashboard_state(log_dir: Path, state: DashboardState) -> None:
    """Atomic write of dashboard state to JSON."""
    _atomic_write(log_dir / "dashboard_state.json", json.dumps(asdict(state), default=str) + "\n")


def append_trade_event(log_dir: Path, event: TradeEvent, max_events: int = 200) -> None:
    """Prepend a trade event to the events file, capping at max_events."""
    events_path = log_dir / "trade_events.json"
    existing: list[dict] = []
    if events_path.exists():
        try:
            existing = json.loads(events_path.read_text(encoding="utf-8"))
        except Exception:
            existing = []
    existing.insert(0, asdict(event))
    existing = existing[:max_events]
    _atomic_write(events_path, json.dumps(existing, default=str) + "\n")


def read_dashboard_state(log_dir: Path) -> Optional[dict]:
    """Read dashboard state JSON, returning None if not available."""
    p = log_dir / "dashboard_state.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_trade_events(log_dir: Path, limit: int = 50) -> list[dict]:
    """Read trade events JSON, returning up to limit entries."""
    p = log_dir / "trade_events.json"
    if not p.exists():
        return []
    try:
        events = json.loads(p.read_text(encoding="utf-8"))
        return events[:limit]
    except Exception:
        return []
