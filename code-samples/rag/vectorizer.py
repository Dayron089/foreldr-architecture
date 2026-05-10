"""
Embedding production for short, episodic memory facts.

Two design choices distinguish this vectorizer from a default
"embed each row" loop:

1. Enrichment.
   A short fact ("Loves long walks at sunset.") embeds poorly. The
   discriminative information lives in the surrounding episodic
   context: when it was said, what the conversation was about. We
   therefore embed (fact, context_excerpt) jointly.

   Why concatenation instead of fancier fusion (CLS-token average,
   learned projection, weighted mean of separate embeddings):

   - The combined string is short (sub-300 tokens). The encoder's
     own attention handles context-fact interaction correctly when
     they're tokenized together as a single passage.
   - Fancier fusion needs a fine-tuned encoder. The marginal recall
     gain is small relative to the operational cost.
   - Concatenation is reversible at debug time. Searches against
     the raw fact still work; the enrichment is purely additive.

2. Idempotent batching with content-hash dedup.
   Embeddings are the most cost-sensitive call in the system.
   Dedup-by-content-hash before the API call eliminates redundant
   re-embeds across retries, schema migrations, and reprocessing
   jobs. Combined with exponential backoff and a small batch size,
   throughput stays bounded and cost stays predictable.

Backend is a Protocol so the same Vectorizer wraps any embedding
model — a remote API, a local sentence-transformer, or a self-hosted
Matryoshka model. Swap the backend, keep the rest.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
from typing import Optional, Protocol, Sequence

from pydantic import BaseModel, Field, ConfigDict


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class EmbeddingResult(BaseModel):
    """The result of vectorizing a single piece of text."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    text: str
    vector: list[float]
    dim: int
    content_hash: str
    cached: bool = False


class VectorizerConfig(BaseModel):
    """Tunable parameters for the Vectorizer."""

    target_dim: int = 1536
    max_chars: int = 8000
    min_chars: int = 5

    # Retry / batching
    max_retries: int = 3
    base_backoff_seconds: float = 1.0
    timeout_seconds: float = 10.0
    batch_size: int = 16

    # Normalization
    l2_normalize: bool = True

    # Enrichment
    fact_context_separator: str = " "  # Single space; no special token.

    # Embedding cache TTL is handled by an external cache; the vectorizer
    # only does in-memory dedup within a single batch.


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


class EmbeddingProvider(Protocol):
    """Anything that can map a list of strings to a list of vectors."""

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def enrich_for_embedding(fact: str, context_excerpt: str = "", separator: str = " ") -> str:
    """
    Combine a fact with its episodic context into the string that will
    be embedded.

    Rationale:
      - The fact alone is too short to carry enough discriminative signal
        for short-text embedding models, especially for the retrieval-time
        case where the user types a question, not a statement.
      - The context excerpt anchors the fact in the conversation that
        produced it: tokens the user is likely to say again live there.
      - Concatenation with a plain separator (single space) keeps the
        encoder's tokenization regime intact. Inserting a sentinel
        separator like "[SEP]" or "\\n\\n" can shift the encoder's
        positional bias away from the fact and is rarely worth the risk.

    Falls back to the fact alone if context is empty or whitespace.
    """
    fact = fact.strip()
    ctx = context_excerpt.strip() if context_excerpt else ""
    if not ctx:
        return fact
    return f"{fact}{separator}{ctx}"


def content_hash(text: str) -> str:
    """Stable hash for cache and dedup keys."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def l2_normalize(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return list(vec)
    return [x / norm for x in vec]


# ---------------------------------------------------------------------------
# Vectorizer
# ---------------------------------------------------------------------------


class EmbeddingError(RuntimeError):
    """Raised when embedding fails after all retries."""


class Vectorizer:
    """
    Batched, retrying, dedup-aware embedding producer.

    Usage:
        vec = Vectorizer(provider=my_provider)
        results = await vec.embed_facts([
            ("User has a corgi named Biscuit.",
             "Mentioned while describing weekend routines."),
            ("Loves crunchy peanut butter.", "Brief food preference."),
        ])
    """

    def __init__(
        self,
        *,
        provider: EmbeddingProvider,
        config: Optional[VectorizerConfig] = None,
    ) -> None:
        self.provider = provider
        self.config = config or VectorizerConfig()

    # ----- Single-fact path ----------------------------------------------

    async def embed_fact(
        self, fact: str, context_excerpt: str = ""
    ) -> EmbeddingResult:
        """Embed a single (fact, context) pair. Convenience over embed_facts."""
        results = await self.embed_facts([(fact, context_excerpt)])
        return results[0]

    # ----- Batch path ----------------------------------------------------

    async def embed_facts(
        self, pairs: Sequence[tuple[str, str]]
    ) -> list[EmbeddingResult]:
        """
        Embed many (fact, context) pairs.

        Within a batch:
          - Inputs are validated and truncated to max_chars.
          - Identical enriched texts are deduplicated; the underlying
            provider sees each unique string at most once.
          - Original ordering is preserved in the returned list.
          - Vectors are L2-normalized if configured.
        """
        cfg = self.config

        # Build the per-input enriched text and validate.
        enriched: list[str] = []
        for fact, ctx in pairs:
            text = enrich_for_embedding(fact, ctx, cfg.fact_context_separator)
            if len(text) < cfg.min_chars:
                raise ValueError(
                    f"Enriched text below minimum length ({cfg.min_chars}): {text!r}"
                )
            if len(text) > cfg.max_chars:
                text = text[: cfg.max_chars]
            enriched.append(text)

        # Dedup by content hash; remember which original indices map where.
        unique_texts: list[str] = []
        text_to_index: dict[str, int] = {}
        original_to_unique: list[int] = []
        for text in enriched:
            h = content_hash(text)
            if h not in text_to_index:
                text_to_index[h] = len(unique_texts)
                unique_texts.append(text)
            original_to_unique.append(text_to_index[h])

        # Embed unique texts in fixed-size batches.
        unique_vectors: list[list[float]] = []
        for start in range(0, len(unique_texts), cfg.batch_size):
            chunk = unique_texts[start : start + cfg.batch_size]
            vectors = await self._embed_with_retry(chunk)
            unique_vectors.extend(vectors)

        # Optional normalization.
        if cfg.l2_normalize:
            unique_vectors = [l2_normalize(v) for v in unique_vectors]

        # Reconstruct results in original order.
        results: list[EmbeddingResult] = []
        for orig_text, unique_idx in zip(enriched, original_to_unique):
            vec = unique_vectors[unique_idx]
            if len(vec) != cfg.target_dim:
                raise EmbeddingError(
                    f"Vector dim mismatch: got {len(vec)}, expected {cfg.target_dim}"
                )
            results.append(
                EmbeddingResult(
                    text=orig_text,
                    vector=vec,
                    dim=len(vec),
                    content_hash=content_hash(orig_text),
                    cached=False,
                )
            )
        return results

    # ----- Retry wrapper -------------------------------------------------

    async def _embed_with_retry(self, texts: Sequence[str]) -> list[list[float]]:
        cfg = self.config
        last_exc: Optional[BaseException] = None
        for attempt in range(cfg.max_retries):
            try:
                return await asyncio.wait_for(
                    self.provider.embed_batch(texts),
                    timeout=cfg.timeout_seconds,
                )
            except (asyncio.TimeoutError, Exception) as exc:
                last_exc = exc
                if attempt + 1 == cfg.max_retries:
                    break
                # Exponential backoff: 1s, 2s, 4s, ...
                await asyncio.sleep(cfg.base_backoff_seconds * (2 ** attempt))
        raise EmbeddingError(
            f"Embedding failed after {cfg.max_retries} attempts: {last_exc!r}"
        )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


class _StubProvider:
    """
    Deterministic 'embedding' for demo purposes only. Maps each unique
    string to a fixed-dim float vector seeded by its hash. Not even
    remotely a real embedding model.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim

    async def embed_batch(self, texts):
        out: list[list[float]] = []
        for t in texts:
            seed = int(hashlib.sha256(t.encode("utf-8")).hexdigest()[:8], 16)
            rng = _SeededRng(seed)
            out.append([rng.gauss() for _ in range(self.dim)])
        return out


class _SeededRng:
    def __init__(self, seed: int) -> None:
        self.state = seed or 1

    def random(self) -> float:
        # xorshift32; not cryptographically secure, fine for fixtures.
        x = self.state & 0xFFFFFFFF
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17) & 0xFFFFFFFF
        x ^= (x << 5) & 0xFFFFFFFF
        self.state = x & 0xFFFFFFFF
        return self.state / 0xFFFFFFFF

    def gauss(self) -> float:
        # Box-Muller, single sample.
        u1 = max(self.random(), 1e-9)
        u2 = self.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


async def _demo() -> None:
    cfg = VectorizerConfig(target_dim=64, batch_size=4)
    vec = Vectorizer(provider=_StubProvider(cfg.target_dim), config=cfg)

    pairs = [
        (
            "User has a corgi named Biscuit.",
            "Mentioned while describing a weekend routine.",
        ),
        (
            "Loves long walks at sunset.",
            "Said during a quiet Sunday-evening conversation.",
        ),
        # Intentional duplicate of the first one to exercise dedup.
        (
            "User has a corgi named Biscuit.",
            "Mentioned while describing a weekend routine.",
        ),
        ("Listens to ambient music while coding.", ""),
    ]

    results = await vec.embed_facts(pairs)
    for i, r in enumerate(results):
        print(f"[{i}] dim={r.dim}  hash={r.content_hash[:10]}  {r.text[:60]}...")
    # Confirm dedup: indices 0 and 2 share hash and (post-normalization) vector.
    assert results[0].content_hash == results[2].content_hash
    assert results[0].vector == results[2].vector


if __name__ == "__main__":
    asyncio.run(_demo())
