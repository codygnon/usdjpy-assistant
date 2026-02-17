"""Centralized OANDA order/position book snapshot cache.

Stores the last 3 snapshots per instrument with timestamps.
Designed to be called inline from the TA endpoint — polls if stale (>20s).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BookSnapshot:
    timestamp: float
    data: dict


@dataclass
class BookCache:
    """Per-instrument cache for order and position book snapshots."""

    _order_books: dict[str, list[BookSnapshot]] = field(default_factory=dict)
    _position_books: dict[str, list[BookSnapshot]] = field(default_factory=dict)
    _max_snapshots: int = 3
    _stale_seconds: float = 20.0

    def _append(self, store: dict[str, list[BookSnapshot]], instrument: str, data: dict) -> None:
        if instrument not in store:
            store[instrument] = []
        store[instrument].append(BookSnapshot(timestamp=time.time(), data=data))
        # Keep only last N
        if len(store[instrument]) > self._max_snapshots:
            store[instrument] = store[instrument][-self._max_snapshots:]

    def _is_stale(self, store: dict[str, list[BookSnapshot]], instrument: str) -> bool:
        snaps = store.get(instrument, [])
        if not snaps:
            return True
        return (time.time() - snaps[-1].timestamp) > self._stale_seconds

    def poll_books(self, adapter: Any, instrument: str) -> None:
        """Poll order and position books if stale. Safe to call frequently."""
        if self._is_stale(self._order_books, instrument) or self._is_stale(self._position_books, instrument):
            try:
                ob = adapter.get_order_book(instrument)
                self._append(self._order_books, instrument, ob)
            except Exception:
                pass
            try:
                pb = adapter.get_position_book(instrument)
                self._append(self._position_books, instrument, pb)
            except Exception:
                pass

    def get_order_books(self, instrument: str) -> list[BookSnapshot]:
        return self._order_books.get(instrument, [])

    def get_position_books(self, instrument: str) -> list[BookSnapshot]:
        return self._position_books.get(instrument, [])


# Module-level singleton
_cache = BookCache()


def get_book_cache() -> BookCache:
    return _cache
