"""
Prompt caching utilities — metadata extraction, hit-rate tracking, response cache.

WHY THIS EXISTS
---------------
Several major providers offer prompt caching at substantially reduced rates for tokens
that match a previously-seen prefix within a TTL. Treating caching as a black box gives
up the largest available cost lever in modern LLM ops.

This module owns three pieces of caching mechanics:

1. ``extract_cache_metadata`` — normalize different providers' usage objects into a
   uniform ``CacheMetadata`` shape. Downstream code should never branch on provider.
2. ``PromptCacheTracker`` — accumulate per-provider/model hit-rate, savings, and call
   counts so an operator can see in real time whether caching is actually working.
3. ``ResponseCache`` — a small application-level cache for *identical* calls (same
   model, same messages, same parameters). Useful for embeddings, deterministic
   classifications, and fixture replay during tests.

Pricing is illustrative; production code should source rates from configuration.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache metadata
# ---------------------------------------------------------------------------


class CacheMetadata(BaseModel):
    cached_input_tokens: int = 0
    total_input_tokens: int = 0
    cache_hit_ratio: float = 0.0
    savings_usd: float = 0.0


def extract_cache_metadata(
    response_usage: dict[str, Any],
    *,
    provider: str,
    uncached_rate_per_million: float,
    cached_rate_per_million: float,
) -> CacheMetadata:
    """
    Normalize provider-specific usage shapes into ``CacheMetadata``.

    Different providers expose caching differently:

    * Some embed it in ``prompt_tokens_details.cached_tokens``.
    * Some use ``cache_read_input_tokens``.
    * Some report only the discounted total without separating cached vs uncached.

    This function takes a dict (the parsed JSON usage block) and the provider name,
    and returns a uniform metadata object.
    """
    total_input = int(
        response_usage.get("prompt_tokens")
        or response_usage.get("input_tokens")
        or 0
    )
    cached_input = _read_cached_tokens(response_usage, provider)
    if total_input <= 0:
        return CacheMetadata(cached_input_tokens=cached_input, total_input_tokens=0)

    cached_input = min(cached_input, total_input)
    ratio = cached_input / total_input if total_input else 0.0
    savings = (
        cached_input * (uncached_rate_per_million - cached_rate_per_million) / 1_000_000
    )
    return CacheMetadata(
        cached_input_tokens=cached_input,
        total_input_tokens=total_input,
        cache_hit_ratio=ratio,
        savings_usd=max(0.0, savings),
    )


def _read_cached_tokens(usage: dict[str, Any], provider: str) -> int:
    """
    Provider-specific extraction. Add a clause here when a new provider is wired up.
    Each branch reads only the fields that provider documents.
    """
    if provider in {"direct-A"}:
        details = usage.get("prompt_tokens_details") or {}
        return int(details.get("cached_tokens") or 0)
    if provider in {"direct-B"}:
        # Some providers expose cache reads at the top level.
        return int(usage.get("cache_read_input_tokens") or 0)
    if provider in {"direct-C"}:
        # Disk-cache providers: separate prompt_cache_hit_tokens.
        return int(usage.get("prompt_cache_hit_tokens") or 0)
    return 0


# ---------------------------------------------------------------------------
# Hit-rate tracker
# ---------------------------------------------------------------------------


@dataclass
class _Bucket:
    calls: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0
    savings_usd: float = 0.0


class PromptCacheTracker:
    """
    Accumulates per-(provider, model) cache statistics. Pure utility; no IO.

    Plug into your usage tracking layer and call :meth:`record` for every LLM
    response. :meth:`snapshot` returns a serializable view suitable for a metrics
    endpoint or dashboard.
    """

    def __init__(self) -> None:
        self._buckets: dict[tuple[str, str], _Bucket] = defaultdict(_Bucket)

    def record(self, *, provider: str, model: str, metadata: CacheMetadata) -> None:
        bucket = self._buckets[(provider, model)]
        bucket.calls += 1
        bucket.cached_tokens += metadata.cached_input_tokens
        bucket.total_tokens += metadata.total_input_tokens
        bucket.savings_usd += metadata.savings_usd

    def snapshot(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for (provider, model), bucket in sorted(self._buckets.items()):
            ratio = (
                bucket.cached_tokens / bucket.total_tokens if bucket.total_tokens else 0.0
            )
            rows.append(
                {
                    "provider": provider,
                    "model": model,
                    "calls": bucket.calls,
                    "cache_hit_ratio": round(ratio, 4),
                    "cached_tokens": bucket.cached_tokens,
                    "total_input_tokens": bucket.total_tokens,
                    "savings_usd": round(bucket.savings_usd, 6),
                }
            )
        return rows


# ---------------------------------------------------------------------------
# Response cache (application-level)
# ---------------------------------------------------------------------------


def _cache_key(model: str, messages: list[dict[str, str]], params: dict[str, Any]) -> str:
    payload = json.dumps(
        {"model": model, "messages": messages, "params": params}, sort_keys=True
    )
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class _Entry:
    value: Any
    expires_at: float


class ResponseCache:
    """
    In-memory keyed cache with TTL. For production use, swap the dict for Redis;
    the interface is the same.

    Suitable for: embeddings, deterministic classifications, fixture replay.
    Not suitable for: chat generation with temperature > 0.
    """

    def __init__(self, *, default_ttl_s: float = 3600.0) -> None:
        self._store: dict[str, _Entry] = {}
        self._default_ttl = default_ttl_s
        self._lock = asyncio.Lock()
        self.hits = 0
        self.misses = 0

    async def get(
        self, model: str, messages: list[dict[str, str]], params: dict[str, Any]
    ) -> Any | None:
        key = _cache_key(model, messages, params)
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self.misses += 1
                return None
            if entry.expires_at < time.time():
                self._store.pop(key, None)
                self.misses += 1
                return None
            self.hits += 1
            return entry.value

    async def set(
        self,
        model: str,
        messages: list[dict[str, str]],
        params: dict[str, Any],
        value: Any,
        *,
        ttl_s: float | None = None,
    ) -> None:
        key = _cache_key(model, messages, params)
        ttl = ttl_s if ttl_s is not None else self._default_ttl
        async with self._lock:
            self._store[key] = _Entry(value=value, expires_at=time.time() + ttl)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


async def _demo() -> None:
    tracker = PromptCacheTracker()

    # Simulate 5 calls to a provider that hits its cache 90% of the time.
    for _ in range(5):
        usage = {
            "prompt_tokens": 1500,
            "prompt_tokens_details": {"cached_tokens": 1350},
            "completion_tokens": 90,
        }
        meta = extract_cache_metadata(
            usage,
            provider="direct-A",
            uncached_rate_per_million=0.20,
            cached_rate_per_million=0.05,
        )
        tracker.record(provider="direct-A", model="small-fast-v1", metadata=meta)

    # And a few calls to an aggregator with no caching.
    for _ in range(3):
        meta = extract_cache_metadata(
            {"prompt_tokens": 1500, "completion_tokens": 90},
            provider="aggregator-B",
            uncached_rate_per_million=0.22,
            cached_rate_per_million=0.22,
        )
        tracker.record(provider="aggregator-B", model="small-fast-v1", metadata=meta)

    print("Cache snapshot:")
    for row in tracker.snapshot():
        print(row)

    # Application-level response cache demo.
    cache = ResponseCache(default_ttl_s=60)
    msgs = [{"role": "user", "content": "embed this"}]
    miss = await cache.get("embed-v1", msgs, {})
    await cache.set("embed-v1", msgs, {}, value=[0.1, 0.2, 0.3])
    hit = await cache.get("embed-v1", msgs, {})
    print(f"\nresponse_cache miss={miss}  hit={hit}  hits={cache.hits}  misses={cache.misses}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    asyncio.run(_demo())
