from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


@dataclass(frozen=True)
class SessionStatus:
    tokyo_active: bool
    london_active: bool
    ny_active: bool
    active_sessions: list[str]
    overlap: Optional[str]
    minutes_to_next_close: int
    next_close_session: str
    current_utc: datetime


class SessionManager:
    """Tracks forex session times and provides alerts."""

    def __init__(self, config):
        self._config = config

    def get_status(self, now: Optional[datetime] = None) -> SessionStatus:
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        else:
            now = now.astimezone(timezone.utc)

        hour = int(now.hour)
        minute = int(now.minute)
        tokyo = self._config.tokyo_start <= hour < self._config.tokyo_end
        london = self._config.london_start <= hour < self._config.london_end
        ny = self._config.ny_start <= hour < self._config.ny_end

        active: list[str] = []
        if tokyo:
            active.append("Tokyo")
        if london:
            active.append("London")
        if ny:
            active.append("New York")

        overlap = None
        if london and ny:
            overlap = "London-NY"
        elif tokyo and london:
            overlap = "Tokyo-London"

        closes: list[tuple[str, int]] = []
        if tokyo:
            closes.append(("Tokyo", self._config.tokyo_end))
        if london:
            closes.append(("London", self._config.london_end))
        if ny:
            closes.append(("New York", self._config.ny_end))

        if closes:
            nearest_session, nearest_hour = min(closes, key=lambda item: (((item[1] - hour) % 24) * 60 - minute))
            minutes_to_close = ((nearest_hour - hour) % 24) * 60 - minute
        else:
            nearest_session = "None"
            minutes_to_close = 999

        return SessionStatus(
            tokyo_active=tokyo,
            london_active=london,
            ny_active=ny,
            active_sessions=active,
            overlap=overlap,
            minutes_to_next_close=minutes_to_close,
            next_close_session=nearest_session,
            current_utc=now,
        )

    def format_status(self, status: SessionStatus) -> str:
        def icon(active: bool) -> str:
            return "🟢" if active else "⚫"

        overlap_text = f"  ⚡ OVERLAP: {status.overlap}" if status.overlap else ""
        close_warning = ""
        if status.minutes_to_next_close <= self._config.warn_before_close_minutes:
            close_warning = f"\n  ⚠️  {status.next_close_session} closing in {status.minutes_to_next_close} minutes!"
        return (
            "\n"
            "──────────────────────────────────────────────────\n"
            f"  SESSIONS  ({status.current_utc.strftime('%H:%M UTC')})\n"
            f"  {icon(status.tokyo_active)} Tokyo    {icon(status.london_active)} London    {icon(status.ny_active)} New York"
            f"{overlap_text}{close_warning}\n"
            "──────────────────────────────────────────────────\n"
        )
