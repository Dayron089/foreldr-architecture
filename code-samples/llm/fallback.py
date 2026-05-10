"""
Circuit breaker, retry-with-jitter, and a typed decorator for guarded async calls.

WHY THIS EXISTS
---------------
A naive ``try/except`` fallback hammers a degraded provider with retries during an
outage and produces unbounded user-visible latency. Production-grade fallback needs
three independent pieces:

* A **circuit breaker** that short-circuits calls to a known-failing dependency until
  it has had time to recover.
* A **bounded, jittered retry** policy that gives transient failures one chance to
  resolve without amplifying load.
* **Deadline propagation** so a slow chain of fallbacks cannot blow the user's
  request budget.

This module implements all three and exposes them as a decorator that any async
callable can adopt without restructuring.

DESIGN NOTES
------------
* The breaker is a small state machine: ``closed`` → ``open`` → ``half-open``.
* The decorator is fully typed using ``ParamSpec`` and ``TypeVar`` so the wrapped
  callable's signature is preserved.
* Exception classification is configurable: only ``transient_exceptions`` trigger
  retry/breaker logic. A 4xx-class error should not flap the breaker.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, Iterable, ParamSpec, TypeVar

log = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class CircuitOpen(Exception):
    """Raised by the breaker when the wrapped callable should be skipped."""


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


@dataclass
class CircuitBreaker:
    """
    Per-resource breaker. Trip after ``failure_threshold`` failures within the
    measurement window. After ``cooldown`` elapses, allow one probe (half-open);
    success closes, failure re-opens.
    """

    failure_threshold: int = 5
    cooldown: timedelta = timedelta(seconds=30)

    _failures: int = field(default=0, init=False)
    _opened_at: datetime | None = field(default=None, init=False)
    _half_open_in_flight: bool = field(default=False, init=False)

    def allow(self) -> bool:
        """Return True if the next call should be attempted."""
        if self._opened_at is None:
            return True
        if datetime.now(timezone.utc) - self._opened_at >= self.cooldown:
            # Move to half-open: one probe at a time.
            if not self._half_open_in_flight:
                self._half_open_in_flight = True
                return True
            return False
        return False

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None
        self._half_open_in_flight = False

    def record_failure(self) -> None:
        self._half_open_in_flight = False
        self._failures += 1
        if self._failures >= self.failure_threshold and self._opened_at is None:
            self._opened_at = datetime.now(timezone.utc)
            log.warning("circuit opened failures=%d", self._failures)

    @property
    def is_open(self) -> bool:
        return self._opened_at is not None and (
            datetime.now(timezone.utc) - self._opened_at < self.cooldown
        )


# ---------------------------------------------------------------------------
# Retry / decorator
# ---------------------------------------------------------------------------


def _backoff(attempt: int, base_ms: int) -> float:
    """Exponential with full jitter."""
    base = base_ms / 1000.0 * (2 ** (attempt - 1))
    return base * (0.5 + random.random())


def guarded(
    *,
    breaker: CircuitBreaker,
    transient_exceptions: Iterable[type[BaseException]] = (TimeoutError, ConnectionError),
    max_attempts: int = 3,
    base_backoff_ms: int = 100,
    deadline_kwarg: str = "deadline",
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """
    Wrap an async callable with circuit-breaking, jittered retry, and deadline
    propagation. The wrapped callable's signature is preserved.

    ``deadline_kwarg`` names the optional keyword argument carrying a ``datetime``
    deadline. If present, it is honoured for the *whole* set of attempts.
    """
    transient: tuple[type[BaseException], ...] = tuple(transient_exceptions)

    def decorator(fn: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @functools.wraps(fn)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if not breaker.allow():
                raise CircuitOpen(f"breaker open for {fn.__name__}")

            deadline: datetime | None = kwargs.get(deadline_kwarg)  # type: ignore[assignment]

            last_error: BaseException | None = None
            for attempt in range(1, max_attempts + 1):
                remaining = _seconds_until(deadline)
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("deadline exceeded")

                try:
                    if remaining is None:
                        result = await fn(*args, **kwargs)
                    else:
                        result = await asyncio.wait_for(
                            fn(*args, **kwargs), timeout=remaining
                        )
                    breaker.record_success()
                    return result
                except transient as exc:
                    last_error = exc
                    log.info(
                        "guarded transient fn=%s attempt=%d/%d error=%s",
                        fn.__name__,
                        attempt,
                        max_attempts,
                        exc,
                    )
                    if attempt < max_attempts:
                        await asyncio.sleep(_backoff(attempt, base_backoff_ms))
                except Exception as exc:
                    # Non-transient: don't retry, don't flap the breaker.
                    raise

            breaker.record_failure()
            raise last_error or RuntimeError("retry exhausted with no error captured")

        return wrapper

    return decorator


def _seconds_until(deadline: datetime | None) -> float | None:
    if deadline is None:
        return None
    return (deadline - datetime.now(timezone.utc)).total_seconds()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


async def _demo() -> None:
    # A callable that fails the first two attempts, then succeeds.
    state = {"calls": 0}

    breaker = CircuitBreaker(failure_threshold=10, cooldown=timedelta(seconds=10))

    @guarded(
        breaker=breaker,
        transient_exceptions=(ConnectionError,),
        max_attempts=3,
        base_backoff_ms=20,
    )
    async def flaky_call(x: int, *, deadline: datetime | None = None) -> int:
        state["calls"] += 1
        if state["calls"] < 3:
            raise ConnectionError("transient")
        return x * 2

    result = await flaky_call(7, deadline=datetime.now(timezone.utc) + timedelta(seconds=2))
    print(f"flaky_call result={result} calls={state['calls']}")

    # Now demonstrate breaker-open behavior.
    open_breaker = CircuitBreaker(failure_threshold=2, cooldown=timedelta(seconds=10))

    @guarded(breaker=open_breaker, transient_exceptions=(ConnectionError,), max_attempts=1)
    async def always_fails(*, deadline: datetime | None = None) -> int:
        raise ConnectionError("upstream down")

    for i in range(3):
        try:
            await always_fails(deadline=None)
        except CircuitOpen:
            print(f"attempt {i + 1}: short-circuited by breaker")
        except ConnectionError:
            print(f"attempt {i + 1}: passed through to upstream and failed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    asyncio.run(_demo())
