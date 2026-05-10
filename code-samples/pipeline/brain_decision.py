"""
Brain Decision Engine — single-call safety + intent + decision for a stateful agent pipeline.

WHY THIS EXISTS
---------------
A naive agent runs three separate LLM calls per turn: one for safety screening, one for
intent classification, one for action selection. Each call repeats most of the same context,
so the prompt cost roughly triples and so does the latency. Worse, the three decisions can
disagree — the safety call says "fine," the intent call says "media request," the action
call refuses anyway because it sees a different slice of the prompt.

This module collapses those three decisions into one structured-output LLM call. The model
returns a single JSON object with `safety`, `intent`, `action`, `mood`, and a few hint flags
that downstream stages use to pick prompt sections. The whole turn's decisioning becomes
one round trip instead of three.

The cost of the unification: the JSON schema is load-bearing. A malformed response now
breaks three responsibilities at once. The mitigation is rigorous validation, one retry on
malformed output, and a deterministic safe fallback if the model is fundamentally broken.
The conversation never fails because of this stage — it can only become slightly less
personalized.

DESIGN NOTES
------------
- The prompt is short and stable so it can be cached by providers that support implicit
  prompt caching. Variable content (the user message, recent tail) is at the bottom.
- The output schema is small enough that a fast-tier model produces it reliably.
- All persona-specific content is generic in this sample. The real engine pulls from a
  persona registry; here we accept a `PersonaEssence` object so the file is standalone.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Literal, Protocol

from pydantic import BaseModel, Field, ValidationError

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------


class PersonaEssence(BaseModel):
    """Immutable traits the model needs to reason about a persona's voice."""

    name: str
    core_trait: str
    voice: str
    never: list[str] = Field(default_factory=list)


class Memory(BaseModel):
    """A retrieved long-term memory the model may reference."""

    text: str
    relevance: float


class BrainContext(BaseModel):
    """Everything the engine needs to decide on a single turn."""

    user_message: str
    recent_tail: list[str] = Field(default_factory=list)
    memories: list[Memory] = Field(default_factory=list)
    persona: PersonaEssence
    trust_level: int = 0  # 0..100, abstract progression
    local_hour: int = 12
    is_first_message: bool = False


class SafetyFlags(BaseModel):
    nsfw: bool = False
    manipulation: bool = False
    aggression: bool = False
    extreme_risk: bool = False


class Intent(str, Enum):
    CHAT = "chat"
    QUESTION = "question"
    MEDIA_REQUEST = "media_request"
    EMOTIONAL = "emotional"
    FLIRT = "flirt"


class Mood(str, Enum):
    WARM = "warm"
    PLAYFUL = "playful"
    CASUAL = "casual"
    EMPATHETIC = "empathetic"


Action = Literal["respond", "deflect", "refuse"]


class Decision(BaseModel):
    """The structured output the LLM produces."""

    action: Action = "respond"
    intent: Intent = Intent.CHAT
    mood: Mood = Mood.WARM
    media_requested: bool = False
    requested_media_type: str | None = None
    needs_memory_context: bool = False
    needs_schedule_context: bool = False
    thought: str = ""
    confidence: float = 0.9


class BrainOutput(BaseModel):
    """The full output of the engine, including derived fields."""

    safety: SafetyFlags = Field(default_factory=SafetyFlags)
    decision: Decision = Field(default_factory=Decision)
    used_fallback: bool = False
    raw_latency_ms: int = 0


# ---------------------------------------------------------------------------
# LLM provider Protocol — keeps this file standalone
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMResponse(BaseModel):
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    latency_ms: int = 0


class LLMProvider(Protocol):
    async def generate(
        self,
        messages: list[ChatMessage],
        model: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.2,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse: ...


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT_TEMPLATE = """\
You are the decision unit for a persona-driven companion. Your job is to read the latest
user message, the conversation tail, retrieved memories, and the persona's voice, and
produce a single structured decision object.

OUTPUT
Return ONLY a JSON object matching this exact schema. No prose, no code fences:

{{
  "safety": {{
    "nsfw": false,
    "manipulation": false,
    "aggression": false,
    "extreme_risk": false
  }},
  "decision": {{
    "action": "respond" | "deflect" | "refuse",
    "intent": "chat" | "question" | "media_request" | "emotional" | "flirt",
    "mood": "warm" | "playful" | "casual" | "empathetic",
    "media_requested": false,
    "requested_media_type": null,
    "needs_memory_context": false,
    "needs_schedule_context": false,
    "thought": "<one short sentence rationale>",
    "confidence": 0.0
  }}
}}

DECISION RULES
- Use "refuse" only for a clear safety flag (nsfw, manipulation, aggression, extreme_risk).
- Use "deflect" when the user is pushing toward content the persona would not engage with
  but no hard safety boundary is hit.
- Use "respond" otherwise. The persona may still decline within natural conversation.
- "needs_memory_context" hints the writer stage to include retrieved memories.
- "needs_schedule_context" hints the writer stage to include the persona's day plan.
- Keep "thought" short and grounded — it is read by the writer stage as a guardrail.

PERSONA
name: {persona_name}
core_trait: {core_trait}
voice: {voice}
never: {never}
"""


_USER_TEMPLATE = """\
[trust_level={trust}/100  local_hour={hour}  first_message={first}]

RECENT TAIL
{tail}

RETRIEVED MEMORIES
{memories}

USER MESSAGE
{message}
"""


def _format_tail(tail: list[str]) -> str:
    if not tail:
        return "(empty)"
    # last 6 turns are usually enough for decisioning
    return "\n".join(f"- {t}" for t in tail[-6:])


def _format_memories(memories: list[Memory]) -> str:
    if not memories:
        return "(none)"
    top = sorted(memories, key=lambda m: m.relevance, reverse=True)[:5]
    return "\n".join(f"- ({m.relevance:.2f}) {m.text}" for m in top)


def _build_messages(ctx: BrainContext) -> list[ChatMessage]:
    system = _SYSTEM_PROMPT_TEMPLATE.format(
        persona_name=ctx.persona.name,
        core_trait=ctx.persona.core_trait,
        voice=ctx.persona.voice,
        never=", ".join(ctx.persona.never) if ctx.persona.never else "(none)",
    )
    user = _USER_TEMPLATE.format(
        trust=ctx.trust_level,
        hour=ctx.local_hour,
        first=str(ctx.is_first_message).lower(),
        tail=_format_tail(ctx.recent_tail),
        memories=_format_memories(ctx.memories),
        message=ctx.user_message.strip(),
    )
    return [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=user),
    ]


# ---------------------------------------------------------------------------
# Robust JSON parsing
# ---------------------------------------------------------------------------


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_blob(text: str) -> str:
    """
    Models occasionally wrap JSON in code fences or prefix it with a sentence,
    even when explicitly told not to. Pull out the largest balanced object.
    """
    text = text.strip()
    if text.startswith("```"):
        # strip code fence
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:-3]
    match = _JSON_OBJECT_RE.search(text)
    if not match:
        raise ValueError("no JSON object found in model output")
    return match.group(0)


def _parse_brain_output(raw: str) -> BrainOutput:
    blob = _extract_json_blob(raw)
    data = json.loads(blob)
    safety = SafetyFlags.model_validate(data.get("safety", {}))
    decision = Decision.model_validate(data.get("decision", {}))
    return BrainOutput(safety=safety, decision=decision)


def _safe_default() -> BrainOutput:
    """Used when the model is fundamentally broken — keeps the conversation alive."""
    return BrainOutput(
        safety=SafetyFlags(),
        decision=Decision(
            action="respond",
            intent=Intent.CHAT,
            mood=Mood.WARM,
            media_requested=False,
            needs_memory_context=False,
            needs_schedule_context=False,
            thought="fallback: keep the turn alive with a benign reply",
            confidence=0.3,
        ),
        used_fallback=True,
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class BrainDecisionEngine:
    """
    Single-call decisioning. One LLM round trip per user turn, one structured response,
    one retry on malformed output, one deterministic fallback.

    The engine is intentionally stateless. Construct it once with a provider; call
    :meth:`decide` per turn.
    """

    def __init__(
        self,
        provider: LLMProvider,
        *,
        model: str = "small-fast-v1",
        max_attempts: int = 2,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> None:
        self._provider = provider
        self._model = model
        self._max_attempts = max_attempts
        self._max_tokens = max_tokens
        self._temperature = temperature

    async def decide(self, ctx: BrainContext) -> BrainOutput:
        messages = _build_messages(ctx)
        last_error: Exception | None = None

        for attempt in range(1, self._max_attempts + 1):
            started = datetime.now(timezone.utc)
            try:
                response = await self._provider.generate(
                    messages=messages,
                    model=self._model,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                    response_format={"type": "json_object"},
                )
                parsed = _parse_brain_output(response.text)
                parsed.raw_latency_ms = response.latency_ms
                return parsed
            except (ValueError, json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                log.warning(
                    "brain_decision parse_failure attempt=%d error=%s",
                    attempt,
                    exc,
                )
                # On retry, nudge the model toward stricter output
                if attempt < self._max_attempts:
                    messages = self._stricter_messages(messages)
            except Exception as exc:  # provider-side failure
                last_error = exc
                log.warning(
                    "brain_decision provider_failure attempt=%d error=%s",
                    attempt,
                    exc,
                )
            elapsed_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)
            log.debug("brain_decision attempt=%d elapsed_ms=%d", attempt, elapsed_ms)

        log.error("brain_decision exhausted attempts, falling back: %s", last_error)
        return _safe_default()

    @staticmethod
    def _stricter_messages(messages: list[ChatMessage]) -> list[ChatMessage]:
        return list(messages) + [
            ChatMessage(
                role="user",
                content=(
                    "Your previous output could not be parsed. "
                    "Return ONLY a single JSON object matching the schema, no prose."
                ),
            )
        ]


# ---------------------------------------------------------------------------
# Demo with a stub provider
# ---------------------------------------------------------------------------


class _StubProvider:
    """Echoes a canned valid JSON decision. For the demo only."""

    name = "stub"

    async def generate(
        self,
        messages: list[ChatMessage],
        model: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.2,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        # In a real provider, the request would hit the network here.
        # For the demo, we return a valid decision blob regardless of input.
        await asyncio.sleep(0)  # yield to the event loop
        body = {
            "safety": {
                "nsfw": False,
                "manipulation": False,
                "aggression": False,
                "extreme_risk": False,
            },
            "decision": {
                "action": "respond",
                "intent": "question",
                "mood": "warm",
                "media_requested": False,
                "requested_media_type": None,
                "needs_memory_context": True,
                "needs_schedule_context": False,
                "thought": "user is asking a low-stakes question; reply warmly",
                "confidence": 0.92,
            },
        }
        return LLMResponse(
            text=json.dumps(body),
            prompt_tokens=420,
            completion_tokens=80,
            cached_tokens=380,
            latency_ms=412,
        )


async def _demo() -> None:
    engine = BrainDecisionEngine(_StubProvider())
    ctx = BrainContext(
        user_message="hey, how's your day going?",
        recent_tail=["earlier you said you were heading to a class", "good luck!"],
        memories=[
            Memory(text="user works as a software engineer", relevance=0.71),
            Memory(text="persona is studying for finals this week", relevance=0.83),
        ],
        persona=PersonaEssence(
            name="Companion",
            core_trait="curious and grounded",
            voice="warm, direct, light humor",
            never=["pretend to have physical form", "claim real-time information"],
        ),
        trust_level=42,
        local_hour=20,
    )
    output = await engine.decide(ctx)
    print(output.model_dump_json(indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    asyncio.run(_demo())
