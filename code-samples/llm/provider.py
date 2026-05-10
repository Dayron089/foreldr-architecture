"""
LLM provider abstraction — Protocol, two concrete adapters, fallback chain, cost layer.

WHY THIS EXISTS
---------------
A production agent has multiple LLM workloads (decisioning, generation, embeddings,
background batch). Each wants different latency/cost/quality trade-offs. Hardcoding a
single provider couples the entire system to one vendor's outages, rate limits, and
pricing changes.

This module defines:

* ``LLMProvider`` — a Protocol every provider implements.
* ``LLMResponse`` — a normalized response shape including cost and cache metadata.
* ``DirectProvider`` and ``AggregatorProvider`` — two concrete adapters over an
  HTTP API. Real implementations would use ``httpx``; here the network layer is
  stubbed so the file runs standalone.
* ``ProviderChain`` — a small fallback wrapper that composes a primary and one or
  more fallbacks with circuit breaking, jittered retry, and deadline propagation.
* A cost computation function driven by a per-provider/model rate table.

DESIGN NOTES
------------
* Cache metadata extraction is provider-specific but normalized into the
  ``cached_tokens`` field. Downstream code never branches on provider.
* Retries are bounded and jittered. The fallback chain skips a provider whose
  circuit breaker is open.
* All public functions carry type hints. The Protocol is structural, so any object
  with the right shape is usable.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Protocol

from pydantic import BaseModel

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Wire models
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMResponse(BaseModel):
    text: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int = 0
    cost_usd: float = 0.0
    provider: str
    model: str
    latency_ms: int
    finish_reason: str = "stop"


# ---------------------------------------------------------------------------
# Provider Protocol
# ---------------------------------------------------------------------------


class LLMProvider(Protocol):
    name: str

    async def generate(
        self,
        messages: list[ChatMessage],
        model: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        response_format: dict[str, Any] | None = None,
        deadline: datetime | None = None,
    ) -> LLMResponse: ...


# ---------------------------------------------------------------------------
# Cost computation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rate:
    """USD per 1M tokens. ``cached`` is None when the provider does not support caching."""

    prompt: float
    completion: float
    cached: float | None = None


# Illustrative rates only. A real deployment would read these from configuration so
# pricing changes don't require a code deploy.
RATE_TABLE: dict[tuple[str, str], Rate] = {
    ("direct-A", "small-fast-v1"): Rate(prompt=0.20, completion=0.50, cached=0.05),
    ("direct-A", "gen-mid-v3"): Rate(prompt=0.28, completion=0.42, cached=0.028),
    ("aggregator-B", "small-fast-v1"): Rate(prompt=0.22, completion=0.55),
    ("aggregator-B", "gen-mid-v3"): Rate(prompt=0.30, completion=0.45),
}


def compute_cost(
    provider: str, model: str, prompt_tokens: int, completion_tokens: int, cached_tokens: int
) -> float:
    rate = RATE_TABLE.get((provider, model))
    if rate is None:
        return 0.0
    uncached = max(0, prompt_tokens - cached_tokens)
    cached_cost = (
        (cached_tokens * rate.cached / 1_000_000) if rate.cached is not None else 0.0
    )
    if rate.cached is None:
        # Fall back to uncached rate if the provider doesn't expose a cache tier.
        cached_cost = cached_tokens * rate.prompt / 1_000_000
    return (
        uncached * rate.prompt / 1_000_000
        + cached_cost
        + completion_tokens * rate.completion / 1_000_000
    )


# ---------------------------------------------------------------------------
# Concrete adapters
# ---------------------------------------------------------------------------


class _SimulatedHTTPError(Exception):
    """Raised by stubbed adapters to demonstrate retry and fallback paths."""


class DirectProvider:
    """
    First-party provider with native prompt caching support.

    Real implementation would build an httpx.AsyncClient against the vendor's base
    URL, sign requests, and parse the response body. Here the network layer is
    replaced with a deterministic stub so the file is runnable.
    """

    def __init__(
        self,
        *,
        name: str = "direct-A",
        fail_rate: float = 0.0,
        latency_ms: int = 350,
        cache_hit_rate: float = 0.95,
    ) -> None:
        self.name = name
        self._fail_rate = fail_rate
        self._latency_ms = latency_ms
        self._cache_hit_rate = cache_hit_rate

    async def generate(
        self,
        messages: list[ChatMessage],
        model: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        response_format: dict[str, Any] | None = None,
        deadline: datetime | None = None,
    ) -> LLMResponse:
        await asyncio.sleep(self._latency_ms / 1000.0)
        if random.random() < self._fail_rate:
            raise _SimulatedHTTPError(f"{self.name} simulated 503")

        prompt_tokens = sum(_estimate_tokens(m.content) for m in messages)
        completion_tokens = min(max_tokens, 90)
        cached = int(prompt_tokens * self._cache_hit_rate)
        cost = compute_cost(self.name, model, prompt_tokens, completion_tokens, cached)

        return LLMResponse(
            text="(simulated direct response)",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached,
            cost_usd=cost,
            provider=self.name,
            model=model,
            latency_ms=self._latency_ms,
        )


class AggregatorProvider:
    """
    Multi-vendor aggregator. Higher latency, no native caching, but resilient
    against any one upstream vendor going down.
    """

    def __init__(
        self,
        *,
        name: str = "aggregator-B",
        fail_rate: float = 0.0,
        latency_ms: int = 700,
    ) -> None:
        self.name = name
        self._fail_rate = fail_rate
        self._latency_ms = latency_ms

    async def generate(
        self,
        messages: list[ChatMessage],
        model: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        response_format: dict[str, Any] | None = None,
        deadline: datetime | None = None,
    ) -> LLMResponse:
        await asyncio.sleep(self._latency_ms / 1000.0)
        if random.random() < self._fail_rate:
            raise _SimulatedHTTPError(f"{self.name} simulated 503")

        prompt_tokens = sum(_estimate_tokens(m.content) for m in messages)
        completion_tokens = min(max_tokens, 90)
        cost = compute_cost(self.name, model, prompt_tokens, completion_tokens, 0)

        return LLMResponse(
            text="(simulated aggregator response)",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=0,
            cost_usd=cost,
            provider=self.name,
            model=model,
            latency_ms=self._latency_ms,
        )


def _estimate_tokens(text: str) -> int:
    """Rough char/4 estimate. Real code uses a tokenizer."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Fallback chain
# ---------------------------------------------------------------------------


class CircuitOpen(Exception):
    """Raised internally to skip a provider whose breaker is open."""


@dataclass
class _BreakerState:
    failures: int = 0
    opened_at: datetime | None = None


class ProviderChain:
    """
    Composes providers with per-provider circuit breakers, jittered exponential
    backoff retries, and deadline propagation.

    The first provider in the list is the primary; subsequent providers are
    fallbacks tried in order if the primary fails or is circuit-broken.
    """

    def __init__(
        self,
        providers: list[LLMProvider],
        *,
        failure_threshold: int = 3,
        cooldown: timedelta = timedelta(seconds=30),
        retry_attempts: int = 2,
        base_backoff_ms: int = 100,
    ) -> None:
        if not providers:
            raise ValueError("ProviderChain requires at least one provider")
        self._providers = providers
        self._breakers: dict[str, _BreakerState] = {p.name: _BreakerState() for p in providers}
        self._failure_threshold = failure_threshold
        self._cooldown = cooldown
        self._retry_attempts = retry_attempts
        self._base_backoff_ms = base_backoff_ms

    async def generate(
        self,
        messages: list[ChatMessage],
        model: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        response_format: dict[str, Any] | None = None,
        deadline: datetime | None = None,
    ) -> LLMResponse:
        last_error: Exception | None = None

        for provider in self._providers:
            if self._is_open(provider.name):
                log.info("chain skip provider=%s (circuit open)", provider.name)
                continue
            try:
                return await self._call_with_retry(
                    provider,
                    messages=messages,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_format=response_format,
                    deadline=deadline,
                )
            except Exception as exc:  # noqa: BLE001 — chain absorbs all and rethrows once
                last_error = exc
                self._record_failure(provider.name)
                log.warning("chain failover provider=%s error=%s", provider.name, exc)

        raise RuntimeError(f"all providers in chain failed: {last_error}") from last_error

    # -- internals ----------------------------------------------------------

    async def _call_with_retry(
        self,
        provider: LLMProvider,
        *,
        messages: list[ChatMessage],
        model: str,
        max_tokens: int,
        temperature: float,
        response_format: dict[str, Any] | None,
        deadline: datetime | None,
    ) -> LLMResponse:
        last_error: Exception | None = None
        for attempt in range(1, self._retry_attempts + 1):
            remaining = self._remaining_seconds(deadline)
            if remaining is not None and remaining <= 0:
                raise TimeoutError("deadline exceeded before attempt")

            try:
                if remaining is None:
                    response = await provider.generate(
                        messages=messages,
                        model=model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        response_format=response_format,
                    )
                else:
                    response = await asyncio.wait_for(
                        provider.generate(
                            messages=messages,
                            model=model,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            response_format=response_format,
                        ),
                        timeout=remaining,
                    )
                self._record_success(provider.name)
                return response
            except (_SimulatedHTTPError, asyncio.TimeoutError, ConnectionError) as exc:
                last_error = exc
                if attempt < self._retry_attempts:
                    delay = self._jittered_backoff(attempt)
                    await asyncio.sleep(delay)
        raise last_error or RuntimeError("retry exhausted with no error captured")

    def _jittered_backoff(self, attempt: int) -> float:
        base = self._base_backoff_ms / 1000.0 * (2 ** (attempt - 1))
        return base * (0.5 + random.random())

    def _remaining_seconds(self, deadline: datetime | None) -> float | None:
        if deadline is None:
            return None
        return (deadline - datetime.now(timezone.utc)).total_seconds()

    def _is_open(self, name: str) -> bool:
        state = self._breakers[name]
        if state.opened_at is None:
            return False
        if datetime.now(timezone.utc) - state.opened_at >= self._cooldown:
            # Half-open: allow a probe and decide based on the outcome.
            state.opened_at = None
            state.failures = 0
            return False
        return True

    def _record_failure(self, name: str) -> None:
        state = self._breakers[name]
        state.failures += 1
        if state.failures >= self._failure_threshold and state.opened_at is None:
            state.opened_at = datetime.now(timezone.utc)
            log.warning("circuit opened provider=%s", name)

    def _record_success(self, name: str) -> None:
        state = self._breakers[name]
        state.failures = 0
        state.opened_at = None


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


async def _demo() -> None:
    # Healthy primary, healthy fallback.
    healthy_chain = ProviderChain(
        [
            DirectProvider(name="direct-A", fail_rate=0.0, latency_ms=300),
            AggregatorProvider(name="aggregator-B", fail_rate=0.0, latency_ms=700),
        ],
        retry_attempts=2,
    )

    # Failing primary; chain should fall over to aggregator.
    flaky_chain = ProviderChain(
        [
            DirectProvider(name="direct-A", fail_rate=1.0, latency_ms=120),
            AggregatorProvider(name="aggregator-B", fail_rate=0.0, latency_ms=600),
        ],
        retry_attempts=2,
        failure_threshold=2,
    )

    msgs = [
        ChatMessage(role="system", content="You are a helpful companion."),
        ChatMessage(role="user", content="hello"),
    ]

    healthy = await healthy_chain.generate(msgs, "small-fast-v1", deadline=None)
    print("--- healthy ---")
    print(healthy.model_dump_json(indent=2))

    fallback = await flaky_chain.generate(msgs, "small-fast-v1", deadline=None)
    print("\n--- fallback (primary failing, aggregator served) ---")
    print(fallback.model_dump_json(indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    asyncio.run(_demo())
