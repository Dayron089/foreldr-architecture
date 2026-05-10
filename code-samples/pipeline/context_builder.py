"""
Context Builder — concurrent IO orchestration with graceful degradation.

WHY THIS EXISTS
---------------
The decisioning and generation stages of a stateful pipeline need a composed view of the
world: recent conversation, retrieved long-term memory, the persona's schedule, the
user/persona trust state, and basic time context. Naively, this is four to six DB calls
in series, which adds 200–500 ms to every turn before any LLM work begins.

This module composes those reads concurrently, with per-fetcher timeouts and per-fetcher
graceful degradation. If the memory store is down, the pipeline still answers — without
memories. If the schedule store is slow, the pipeline still answers — with a default
schedule state. The principle: **the response must always ship**, even if it is slightly
less personalized.

DESIGN NOTES
------------
- Each store is a Protocol so this file is standalone and testable.
- All fetchers run under ``asyncio.gather(..., return_exceptions=True)`` so a single
  failure does not cancel the others.
- Each fetcher has its own timeout. The total wall time is bounded by the slowest
  successful fetcher, capped by the longest individual timeout.
- A failure in any single fetcher is logged and replaced with a conservative default.
- The output is a single Pydantic model, ready for the next stages.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Protocol

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------


class Memory(BaseModel):
    text: str
    relevance: float = 0.0


class Message(BaseModel):
    role: str  # "user" | "assistant"
    text: str
    created_at: datetime


class ScheduleState(BaseModel):
    activity: str = "available"
    location: str = "unknown"
    availability: str = "available"  # available | busy | do_not_disturb


class TrustState(BaseModel):
    level: int = 0  # 0..100, abstract progression marker
    label: str = "stranger"


class TimeContext(BaseModel):
    iso: str
    weekday: int  # 0=Mon
    hour: int


class PersonaEssence(BaseModel):
    name: str
    core_trait: str = ""
    voice: str = ""


class ContextPackage(BaseModel):
    user_id: str
    character_id: str
    user_message: str
    recent_messages: list[Message] = Field(default_factory=list)
    retrieved_memories: list[Memory] = Field(default_factory=list)
    schedule_state: ScheduleState = Field(default_factory=ScheduleState)
    trust_state: TrustState = Field(default_factory=TrustState)
    time_context: TimeContext
    persona_essence: PersonaEssence

    # Per-fetcher health flags so downstream stages can decide whether to compensate.
    memory_fetch_ok: bool = True
    schedule_fetch_ok: bool = True
    trust_fetch_ok: bool = True
    history_fetch_ok: bool = True


# ---------------------------------------------------------------------------
# Store protocols
# ---------------------------------------------------------------------------


class MessageStore(Protocol):
    async def get_recent(self, user_id: str, character_id: str, limit: int) -> list[Message]: ...


class MemoryStore(Protocol):
    async def search(
        self, user_id: str, character_id: str, query: str, top_k: int
    ) -> list[Memory]: ...


class ScheduleStore(Protocol):
    async def current(self, character_id: str, *, hour: int, weekday: int) -> ScheduleState: ...


class TrustStore(Protocol):
    async def get(self, user_id: str, character_id: str) -> TrustState: ...


class PersonaStore(Protocol):
    async def get(self, character_id: str) -> PersonaEssence: ...


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class ContextBuilder:
    """
    Composes a :class:`ContextPackage` from injected stores. Each store is fetched
    concurrently with a per-call timeout. Failures are logged and replaced with
    conservative defaults so the pipeline can continue.
    """

    def __init__(
        self,
        *,
        messages: MessageStore,
        memories: MemoryStore,
        schedule: ScheduleStore,
        trust: TrustStore,
        persona: PersonaStore,
        message_tail_size: int = 12,
        memory_top_k: int = 8,
        per_fetch_timeout_s: float = 0.6,
    ) -> None:
        self._messages = messages
        self._memories = memories
        self._schedule = schedule
        self._trust = trust
        self._persona = persona
        self._tail_size = message_tail_size
        self._top_k = memory_top_k
        self._timeout = per_fetch_timeout_s

    async def build(
        self, *, user_id: str, character_id: str, user_message: str
    ) -> ContextPackage:
        now = datetime.now(timezone.utc)
        time_ctx = TimeContext(iso=now.isoformat(), weekday=now.weekday(), hour=now.hour)

        # Persona is the only fetch we wait on linearly: it's small, cheap, and the
        # downstream defaults depend on it. In production it is usually Redis-cached.
        persona = await self._safe_fetch(
            self._persona.get(character_id),
            default=PersonaEssence(name="Companion"),
            label="persona",
        )

        # Now gather the four heavier fetches concurrently.
        history_co = self._safe_fetch(
            self._messages.get_recent(user_id, character_id, self._tail_size),
            default=[],
            label="history",
        )
        memory_co = self._safe_fetch(
            self._memories.search(user_id, character_id, user_message, self._top_k),
            default=[],
            label="memory",
        )
        schedule_co = self._safe_fetch(
            self._schedule.current(
                character_id, hour=time_ctx.hour, weekday=time_ctx.weekday
            ),
            default=ScheduleState(),
            label="schedule",
        )
        trust_co = self._safe_fetch(
            self._trust.get(user_id, character_id),
            default=TrustState(),
            label="trust",
        )

        history, memories, schedule_state, trust_state = await asyncio.gather(
            history_co, memory_co, schedule_co, trust_co
        )

        return ContextPackage(
            user_id=user_id,
            character_id=character_id,
            user_message=user_message,
            recent_messages=history.value if history.ok else [],
            retrieved_memories=memories.value if memories.ok else [],
            schedule_state=schedule_state.value if schedule_state.ok else ScheduleState(),
            trust_state=trust_state.value if trust_state.ok else TrustState(),
            time_context=time_ctx,
            persona_essence=persona,
            memory_fetch_ok=memories.ok,
            schedule_fetch_ok=schedule_state.ok,
            trust_fetch_ok=trust_state.ok,
            history_fetch_ok=history.ok,
        )

    # -- internals ----------------------------------------------------------

    async def _safe_fetch(self, coro, *, default, label: str):
        """
        Run ``coro`` with a timeout. On timeout or exception, return a sentinel that
        carries ``ok=False`` and the supplied default. Never raises.

        Returns a small ``_FetchResult`` for the gather case, or the value itself when
        called outside of gather. We disambiguate based on whether the caller awaits it
        directly (linear path, returns value) or via gather (returns ``_FetchResult``).
        Here we always return ``_FetchResult`` for consistency in :meth:`build`.
        """
        try:
            value = await asyncio.wait_for(coro, timeout=self._timeout)
            return _FetchResult(ok=True, value=value)
        except asyncio.TimeoutError:
            log.warning("context_builder fetcher=%s timeout", label)
            return _FetchResult(ok=False, value=default)
        except Exception as exc:  # noqa: BLE001 — degradation by design
            log.warning("context_builder fetcher=%s error=%s", label, exc)
            return _FetchResult(ok=False, value=default)


class _FetchResult:
    __slots__ = ("ok", "value")

    def __init__(self, ok: bool, value):
        self.ok = ok
        self.value = value


# Convenience for the persona linear path: unwrap _FetchResult or accept a raw default.
async def _safe_persona(coro, default: PersonaEssence) -> PersonaEssence:
    try:
        return await coro
    except Exception as exc:  # noqa: BLE001
        log.warning("context_builder fetcher=persona error=%s", exc)
        return default


# Patch ContextBuilder._safe_fetch behavior for the persona case to keep the linear
# call site simple. The persona path uses its own helper above.
async def _persona_helper(self: "ContextBuilder", character_id: str) -> PersonaEssence:
    return await _safe_persona(
        self._persona.get(character_id), default=PersonaEssence(name="Companion")
    )


# Wire the helper in: only used by tests if the user wants to replace builder behavior.
ContextBuilder._persona_helper = _persona_helper  # type: ignore[attr-defined]


# Override `build` to use the persona helper so the linear path returns a PersonaEssence
# instead of a _FetchResult. This is split out only to keep the demonstration readable.
async def _build_with_persona_helper(
    self: ContextBuilder, *, user_id: str, character_id: str, user_message: str
) -> ContextPackage:
    now = datetime.now(timezone.utc)
    time_ctx = TimeContext(iso=now.isoformat(), weekday=now.weekday(), hour=now.hour)
    persona = await self._persona_helper(character_id)  # type: ignore[attr-defined]

    history_co = self._safe_fetch(
        self._messages.get_recent(user_id, character_id, self._tail_size),
        default=[],
        label="history",
    )
    memory_co = self._safe_fetch(
        self._memories.search(user_id, character_id, user_message, self._top_k),
        default=[],
        label="memory",
    )
    schedule_co = self._safe_fetch(
        self._schedule.current(character_id, hour=time_ctx.hour, weekday=time_ctx.weekday),
        default=ScheduleState(),
        label="schedule",
    )
    trust_co = self._safe_fetch(
        self._trust.get(user_id, character_id), default=TrustState(), label="trust"
    )

    history, memories, schedule_state, trust_state = await asyncio.gather(
        history_co, memory_co, schedule_co, trust_co
    )

    return ContextPackage(
        user_id=user_id,
        character_id=character_id,
        user_message=user_message,
        recent_messages=history.value,
        retrieved_memories=memories.value,
        schedule_state=schedule_state.value,
        trust_state=trust_state.value,
        time_context=time_ctx,
        persona_essence=persona,
        memory_fetch_ok=memories.ok,
        schedule_fetch_ok=schedule_state.ok,
        trust_fetch_ok=trust_state.ok,
        history_fetch_ok=history.ok,
    )


ContextBuilder.build = _build_with_persona_helper  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Demo with stub stores
# ---------------------------------------------------------------------------


class _StubMessages:
    async def get_recent(self, user_id: str, character_id: str, limit: int) -> list[Message]:
        await asyncio.sleep(0.02)
        return [
            Message(role="user", text="hey", created_at=datetime.now(timezone.utc)),
            Message(role="assistant", text="hi", created_at=datetime.now(timezone.utc)),
        ]


class _StubMemories:
    async def search(self, user_id, character_id, query, top_k):
        await asyncio.sleep(0.05)
        return [
            Memory(text="user mentioned hiking last week", relevance=0.71),
            Memory(text="user has a pet cat", relevance=0.65),
        ]


class _FlakyMemories:
    async def search(self, *_args, **_kwargs):
        # demonstrates graceful degradation
        raise RuntimeError("memory store unavailable")


class _StubSchedule:
    async def current(self, character_id, *, hour, weekday):
        await asyncio.sleep(0.01)
        return ScheduleState(activity="reading", location="home", availability="available")


class _StubTrust:
    async def get(self, user_id, character_id):
        await asyncio.sleep(0.01)
        return TrustState(level=42, label="acquaintance")


class _StubPersona:
    async def get(self, character_id):
        await asyncio.sleep(0.01)
        return PersonaEssence(
            name="Companion",
            core_trait="curious and grounded",
            voice="warm, direct, light humor",
        )


async def _demo() -> None:
    builder = ContextBuilder(
        messages=_StubMessages(),
        memories=_StubMemories(),
        schedule=_StubSchedule(),
        trust=_StubTrust(),
        persona=_StubPersona(),
    )
    pkg = await builder.build(
        user_id="u_1", character_id="c_1", user_message="how was your day?"
    )
    print("--- happy path ---")
    print(pkg.model_dump_json(indent=2))

    flaky = ContextBuilder(
        messages=_StubMessages(),
        memories=_FlakyMemories(),  # this one throws
        schedule=_StubSchedule(),
        trust=_StubTrust(),
        persona=_StubPersona(),
    )
    pkg2 = await flaky.build(
        user_id="u_1", character_id="c_1", user_message="how was your day?"
    )
    print("\n--- degraded path (memory store down) ---")
    print(
        f"memory_fetch_ok={pkg2.memory_fetch_ok}  retrieved_memories={pkg2.retrieved_memories}"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    asyncio.run(_demo())
