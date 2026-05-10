"""
Session Manager — lifecycle for a stateful agent's conversational sessions.

WHY THIS EXISTS
---------------
A persona-driven companion does not have one infinite conversation. It has a sequence of
*sessions* — bounded chunks of dialogue separated by quiet periods. Memory extraction,
summarization, and retention metrics all key off session boundaries, not individual
messages. Doing those things per-message wastes LLM spend and produces low-quality
outputs (an in-progress conversation has no narrative arc to summarize).

This module owns:

1. Opening a session on the first message after a quiet period.
2. Appending messages to the active session.
3. Closing a session — either explicitly, or by an inactivity predicate the cron loop
   evaluates against an injected clock.
4. Firing post-close work (summarization, memory extraction) as background tasks.

DESIGN NOTES
------------
- Idempotent close. The cron loop and the chat path can both call close concurrently;
  exactly one set of post-close tasks must run per session.
- Time-based close uses a pure predicate, ``should_auto_close``, so the cron loop and
  tests use the same logic.
- Background tasks are injected as async callables; this module does not know about
  the LLM, the embedder, or any other subsystem.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, Protocol

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------


class SessionMessage(BaseModel):
    role: str  # "user" | "assistant"
    text: str
    created_at: datetime


class Session(BaseModel):
    id: str
    user_id: str
    character_id: str
    started_at: datetime
    last_message_at: datetime
    is_active: bool = True
    messages: list[SessionMessage] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Storage protocol
# ---------------------------------------------------------------------------


class SessionStore(Protocol):
    async def get_active(self, user_id: str, character_id: str) -> Session | None: ...
    async def insert(self, session: Session) -> None: ...
    async def update(self, session: Session) -> None: ...


# ---------------------------------------------------------------------------
# Post-close hooks
# ---------------------------------------------------------------------------


PostCloseHook = Callable[[Session], Awaitable[None]]


# ---------------------------------------------------------------------------
# Predicate for time-based auto-close
# ---------------------------------------------------------------------------


def should_auto_close(
    session: Session,
    *,
    now: datetime,
    inactivity_threshold: timedelta,
) -> bool:
    """
    Pure predicate. The cron loop iterates active sessions and applies this; tests
    exercise the same function.
    """
    if not session.is_active:
        return False
    return now - session.last_message_at >= inactivity_threshold


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class SessionManager:
    """
    Owns the open/append/close lifecycle. Post-close work is fire-and-forget so the
    response path is never gated on summarization or memory extraction.
    """

    def __init__(
        self,
        store: SessionStore,
        *,
        post_close_hooks: list[PostCloseHook] | None = None,
        inactivity_threshold: timedelta = timedelta(minutes=30),
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._store = store
        self._hooks: list[PostCloseHook] = list(post_close_hooks or [])
        self._inactivity = inactivity_threshold
        self._clock = clock

        # Track in-flight close work so tests and shutdown can await it. Keeping
        # strong references here is intentional — fire-and-forget tasks otherwise risk
        # being garbage-collected mid-execution.
        self._in_flight: set[asyncio.Task[None]] = set()

        # Idempotency: track sessions whose close has already been initiated.
        self._closed_ids: set[str] = set()
        self._lock = asyncio.Lock()

    # -- lifecycle ----------------------------------------------------------

    async def open_or_resume(self, user_id: str, character_id: str) -> Session:
        """
        Returns the active session for the (user, character) pair, opening one if
        none exists.
        """
        existing = await self._store.get_active(user_id, character_id)
        if existing is not None and existing.is_active:
            return existing

        now = self._clock()
        session = Session(
            id=str(uuid.uuid4()),
            user_id=user_id,
            character_id=character_id,
            started_at=now,
            last_message_at=now,
            is_active=True,
        )
        await self._store.insert(session)
        log.info("session opened user=%s character=%s id=%s", user_id, character_id, session.id)
        return session

    async def append(self, session: Session, message: SessionMessage) -> Session:
        """Append a message to the session and persist."""
        session.messages.append(message)
        session.last_message_at = message.created_at
        await self._store.update(session)
        return session

    async def close(self, session: Session, *, reason: str = "explicit") -> bool:
        """
        Close a session and schedule post-close work. Returns True if this call
        actually closed the session, False if it was already closed.

        Idempotent: safe to call from both the cron loop and the chat path concurrently.
        """
        async with self._lock:
            if session.id in self._closed_ids:
                return False
            self._closed_ids.add(session.id)

        session.is_active = False
        await self._store.update(session)
        log.info("session closed id=%s reason=%s", session.id, reason)

        for hook in self._hooks:
            self._launch_background(hook, session)

        return True

    async def maybe_auto_close(self, session: Session) -> bool:
        """
        Convenience for the cron loop: closes a session if it has been idle for the
        configured inactivity threshold.
        """
        if should_auto_close(session, now=self._clock(), inactivity_threshold=self._inactivity):
            return await self.close(session, reason="inactivity")
        return False

    async def shutdown(self) -> None:
        """Wait for any in-flight post-close tasks. Useful in tests and graceful shutdowns."""
        if not self._in_flight:
            return
        await asyncio.gather(*self._in_flight, return_exceptions=True)

    # -- internals ----------------------------------------------------------

    def _launch_background(self, hook: PostCloseHook, session: Session) -> None:
        """Spawn a hook as a background task, capturing failures so they don't propagate."""

        async def _wrapper() -> None:
            try:
                await hook(session)
            except Exception as exc:  # noqa: BLE001 — hooks must not break shutdown
                log.warning("post_close_hook session=%s error=%s", session.id, exc)

        task = asyncio.create_task(_wrapper())
        self._in_flight.add(task)
        task.add_done_callback(self._in_flight.discard)


# ---------------------------------------------------------------------------
# Demo with an in-memory store
# ---------------------------------------------------------------------------


class _MemoryStore:
    def __init__(self) -> None:
        self._by_active_key: dict[tuple[str, str], Session] = {}
        self._all: dict[str, Session] = {}

    async def get_active(self, user_id: str, character_id: str) -> Session | None:
        key = (user_id, character_id)
        session = self._by_active_key.get(key)
        if session and session.is_active:
            return session
        return None

    async def insert(self, session: Session) -> None:
        self._by_active_key[(session.user_id, session.character_id)] = session
        self._all[session.id] = session

    async def update(self, session: Session) -> None:
        self._all[session.id] = session
        if not session.is_active:
            self._by_active_key.pop((session.user_id, session.character_id), None)


async def _summarize_hook(session: Session) -> None:
    log.info("summarize id=%s msgs=%d", session.id, len(session.messages))
    await asyncio.sleep(0.02)


async def _extract_memory_hook(session: Session) -> None:
    log.info("extract_memory id=%s", session.id)
    await asyncio.sleep(0.02)


async def _demo() -> None:
    store = _MemoryStore()
    mgr = SessionManager(
        store,
        post_close_hooks=[_summarize_hook, _extract_memory_hook],
        inactivity_threshold=timedelta(minutes=30),
    )

    s = await mgr.open_or_resume("u_1", "c_1")
    await mgr.append(
        s,
        SessionMessage(role="user", text="hey", created_at=datetime.now(timezone.utc)),
    )
    await mgr.append(
        s,
        SessionMessage(
            role="assistant", text="hi there", created_at=datetime.now(timezone.utc)
        ),
    )

    # Idempotent close: two concurrent close calls produce one set of hooks.
    closed_a, closed_b = await asyncio.gather(mgr.close(s), mgr.close(s))
    print(f"closed_a={closed_a}  closed_b={closed_b}")

    await mgr.shutdown()

    # Auto-close demonstration with a fast-forwarded clock.
    s2 = await mgr.open_or_resume("u_2", "c_2")
    await mgr.append(
        s2,
        SessionMessage(role="user", text="hi", created_at=datetime.now(timezone.utc)),
    )
    s2.last_message_at = datetime.now(timezone.utc) - timedelta(hours=1)
    await store.update(s2)
    closed = await mgr.maybe_auto_close(s2)
    print(f"auto_closed={closed}")
    await mgr.shutdown()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    asyncio.run(_demo())
