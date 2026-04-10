"""Regression: timed-out threadpool work must not block the HTTP handler."""

from __future__ import annotations

import threading
import time

import pytest

from api.main import _run_in_threadpool_with_timeout
from concurrent.futures import TimeoutError as FuturesTimeoutError


def test_threadpool_timeout_returns_without_waiting_for_stuck_worker() -> None:
    done = threading.Event()

    def stuck_until_released() -> str:
        done.wait(timeout=120.0)
        return "finished"

    t0 = time.monotonic()
    with pytest.raises(FuturesTimeoutError):
        _run_in_threadpool_with_timeout(stuck_until_released, 0.15)
    elapsed = time.monotonic() - t0
    done.set()
    assert elapsed < 1.5, "handler should return soon after timeout, not when worker finishes"
