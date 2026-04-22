from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, TypeVar


T = TypeVar("T")


class FillmoreLLMCircuitOpenError(RuntimeError):
    """Raised when the Fillmore LLM circuit breaker is open and a call is skipped."""


class FillmoreLLMCircuitBreaker:
    def __init__(self, *, failure_threshold: int = 3, reset_timeout_sec: float = 300.0) -> None:
        self._failure_threshold = max(1, int(failure_threshold))
        self._reset_timeout_sec = max(1.0, float(reset_timeout_sec))
        self._lock = threading.Lock()
        self._state: str = "closed"
        self._consecutive_failures = 0
        self._last_failure_epoch: float | None = None
        self._last_failure_utc: str | None = None
        self._last_success_utc: str | None = None
        self._last_error: str | None = None
        self._last_callsite: str | None = None
        self._half_open_in_flight = False
        self._total_calls = 0
        self._total_failures = 0
        self._total_skipped = 0

    def before_call(self, callsite: str) -> None:
        now = time.time()
        with self._lock:
            if self._state == "open":
                if self._last_failure_epoch is not None and (now - self._last_failure_epoch) >= self._reset_timeout_sec:
                    self._state = "half_open"
                    self._half_open_in_flight = False
                else:
                    self._total_skipped += 1
                    raise FillmoreLLMCircuitOpenError(
                        f"Fillmore LLM circuit breaker open; skipping {callsite}"
                    )
            if self._state == "half_open":
                if self._half_open_in_flight:
                    self._total_skipped += 1
                    raise FillmoreLLMCircuitOpenError(
                        f"Fillmore LLM circuit breaker probing; skipping {callsite}"
                    )
                self._half_open_in_flight = True
            self._total_calls += 1
            self._last_callsite = callsite

    def record_success(self, callsite: str) -> None:
        now_utc = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._state = "closed"
            self._consecutive_failures = 0
            self._last_success_utc = now_utc
            self._last_callsite = callsite
            self._half_open_in_flight = False

    def record_failure(self, callsite: str, err: BaseException) -> None:
        now = time.time()
        now_utc = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._consecutive_failures += 1
            self._total_failures += 1
            self._last_failure_epoch = now
            self._last_failure_utc = now_utc
            self._last_error = str(err)[:300]
            self._last_callsite = callsite
            self._half_open_in_flight = False
            if self._consecutive_failures >= self._failure_threshold:
                self._state = "open"
            elif self._state == "half_open":
                self._state = "open"
            else:
                self._state = "closed"

    def snapshot(self) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            cooldown_remaining = None
            if self._state == "open" and self._last_failure_epoch is not None:
                cooldown_remaining = max(0.0, self._reset_timeout_sec - (now - self._last_failure_epoch))
            return {
                "state": self._state,
                "failure_threshold": self._failure_threshold,
                "reset_timeout_sec": self._reset_timeout_sec,
                "consecutive_failures": self._consecutive_failures,
                "last_failure_utc": self._last_failure_utc,
                "last_success_utc": self._last_success_utc,
                "last_error": self._last_error,
                "last_callsite": self._last_callsite,
                "cooldown_remaining_sec": round(cooldown_remaining, 1) if cooldown_remaining is not None else None,
                "probe_in_flight": self._half_open_in_flight,
                "total_calls": self._total_calls,
                "total_failures": self._total_failures,
                "total_skipped": self._total_skipped,
            }

    def reset(self) -> None:
        with self._lock:
            self._state = "closed"
            self._consecutive_failures = 0
            self._last_failure_epoch = None
            self._last_failure_utc = None
            self._last_success_utc = None
            self._last_error = None
            self._last_callsite = None
            self._half_open_in_flight = False
            self._total_calls = 0
            self._total_failures = 0
            self._total_skipped = 0


_FILLMORE_BREAKER = FillmoreLLMCircuitBreaker()


def run_guarded_fillmore_llm_call(callsite: str, fn: Callable[[], T]) -> T:
    _FILLMORE_BREAKER.before_call(callsite)
    try:
        result = fn()
    except FillmoreLLMCircuitOpenError:
        raise
    except Exception as exc:
        _FILLMORE_BREAKER.record_failure(callsite, exc)
        raise
    _FILLMORE_BREAKER.record_success(callsite)
    return result


def get_fillmore_llm_health() -> dict[str, Any]:
    return _FILLMORE_BREAKER.snapshot()


def reset_fillmore_llm_health() -> None:
    _FILLMORE_BREAKER.reset()
