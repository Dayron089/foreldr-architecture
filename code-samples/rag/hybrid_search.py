"""
Hybrid retrieval: dense vectors + sparse full-text, fused with RRF.

Why hybrid at all:

A pure dense retriever (cosine over embeddings) misses exact-token
matches that the user typed verbatim - proper nouns, slang, idiomatic
phrases. A pure sparse retriever (BM25 / Postgres FTS) misses paraphrase
and any cross-language semantic equivalence. The two systems fail on
disjoint queries; fusion captures both.

Why RRF instead of weighted-sum fusion:

The standard alternative is
    final_score = alpha * cosine_score + (1 - alpha) * bm25_score
This requires per-deployment tuning of alpha, because cosine sits in
[-1, 1] while BM25 is unbounded and varies with corpus statistics.
The "right" alpha drifts as the corpus grows.

Reciprocal Rank Fusion uses only ranks:
    RRF(d) = sum_r  1 / (k + rank_r(d))
It is scale-free, parameter-stable (k = 60 works almost everywhere),
and works the moment you have two ranked lists. We rerank the fused
candidate set with a domain-specific scoring function downstream;
RRF here is purely the fusion step.

Why HyDE before search:

Embedding a question and then matching it against embeddings of
declarative facts produces a wide cosine gap. A small LLM rewrites
the question as a hypothetical answer in the style of stored facts
("Visits a therapist biweekly..."). The hypothetical and the real
fact live in the same region of embedding space. Recall@k climbs
sharply.

This file is self-contained: backends are protocols, the LLM is an
injected callable, and there are no real DB drivers. The shape mirrors
a production implementation closely enough to read as documentation.
"""

from __future__ import annotations

from typing import Callable, Optional, Protocol, Sequence

from pydantic import BaseModel, Field, ConfigDict


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class Hit(BaseModel):
    """A single candidate document returned from a retrieval backend."""

    model_config = ConfigDict(extra="allow")

    id: str
    score: float = 0.0  # Backend-native score; not directly comparable across backends.
    text: str = ""
    metadata: dict = Field(default_factory=dict)


class FusedHit(Hit):
    """A hit augmented with per-backend ranks and the fused RRF score."""

    rrf_score: float = 0.0
    vector_rank: Optional[int] = None
    fulltext_rank: Optional[int] = None


# ---------------------------------------------------------------------------
# Backend protocols
# ---------------------------------------------------------------------------


class VectorBackend(Protocol):
    """Anything that can return ranked hits given a query embedding."""

    def search(
        self,
        embedding: Sequence[float],
        *,
        k: int,
        filters: Optional[dict] = None,
    ) -> list[Hit]:
        ...


class FullTextBackend(Protocol):
    """Anything that can return ranked hits given a text query."""

    def search(
        self,
        query: str,
        *,
        k: int,
        filters: Optional[dict] = None,
    ) -> list[Hit]:
        ...


# Filter predicates: composable callables that accept a Hit and return bool.
HitPredicate = Callable[[Hit], bool]


def filter_by(field: str, expected: object) -> HitPredicate:
    """Build a predicate that requires hit.metadata[field] == expected."""
    def pred(hit: Hit) -> bool:
        return hit.metadata.get(field) == expected
    return pred


def filter_in(field: str, allowed: set) -> HitPredicate:
    """Build a predicate that requires hit.metadata[field] in allowed."""
    def pred(hit: Hit) -> bool:
        return hit.metadata.get(field) in allowed
    return pred


def compose_filters(*predicates: HitPredicate) -> HitPredicate:
    """AND-compose predicates."""
    def pred(hit: Hit) -> bool:
        return all(p(hit) for p in predicates)
    return pred


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    *ranked_lists: Sequence[Hit],
    k: int = 60,
) -> list[FusedHit]:
    """
    Fuse multiple ranked lists with Reciprocal Rank Fusion.

    The fused score for document d is
        sum over lists r of  1 / (k + rank_r(d))
    where rank_r(d) is 1-indexed position in list r (missing = excluded).

    The constant k is the smoothing parameter; 60 is the commonly cited
    default and works across a wide range of corpora. Lower k makes top
    ranks count more relative to lower ranks.
    """
    fused: dict[str, FusedHit] = {}

    for list_idx, hits in enumerate(ranked_lists):
        for rank, hit in enumerate(hits, start=1):
            existing = fused.get(hit.id)
            if existing is None:
                existing = FusedHit(
                    id=hit.id,
                    score=hit.score,
                    text=hit.text,
                    metadata=dict(hit.metadata),
                )
                fused[hit.id] = existing
            existing.rrf_score += 1.0 / (k + rank)
            if list_idx == 0:
                existing.vector_rank = rank
            elif list_idx == 1:
                existing.fulltext_rank = rank

    return sorted(fused.values(), key=lambda h: h.rrf_score, reverse=True)


# ---------------------------------------------------------------------------
# HyDE
# ---------------------------------------------------------------------------


LLMFn = Callable[[str], str]


def expand_query_with_hyde(
    query: str,
    character_context: str,
    llm_fn: LLMFn,
) -> str:
    """
    Rewrite a user query into a hypothetical declarative fact, in the
    style of stored memories, anchored to the current character.

    The point is to embed the rewrite (not the original query) so that
    retrieval matches statement-shaped vectors with a statement-shaped
    probe.

    `character_context` is a compact hint (name, occupation, core trait).
    Without it, the LLM produces generic facts and HyDE's gain disappears.

    `llm_fn` is injected so the caller controls model, temperature, and
    timeout. A failed call should raise; the surrounding pipeline is
    expected to fall back to a smart query (see module docstring).
    """
    prompt = (
        "Rewrite the user's question as a single declarative fact, in the "
        "style of a stored long-term memory. Keep it under 25 words. "
        "Do not answer the question; describe what the answer would look "
        "like as a fact. Anchor the fact to the character described below.\n\n"
        f"Character: {character_context}\n"
        f"User question: {query}\n\n"
        "Hypothetical fact:"
    )
    return llm_fn(prompt).strip()


# ---------------------------------------------------------------------------
# Searcher
# ---------------------------------------------------------------------------


class HybridSearcher:
    """
    Orchestrates dense + sparse retrieval with RRF fusion.

    Filters are applied post-fusion. Backends do their own pre-filtering
    via the `filters` dict; this is faster (fewer rows leave the DB) but
    not always sufficient (some predicates are easier to express in
    Python than in SQL/RPC).
    """

    def __init__(
        self,
        *,
        vector_backend: VectorBackend,
        fulltext_backend: FullTextBackend,
        rrf_k: int = 60,
        candidate_multiplier: int = 3,
    ) -> None:
        self.vector_backend = vector_backend
        self.fulltext_backend = fulltext_backend
        self.rrf_k = rrf_k
        # Each backend pulls k * multiplier candidates so RRF has overlap.
        # Without this, two backends with completely disjoint top-k lists
        # produce a fused list of length 2k where every doc is ranked once.
        self.candidate_multiplier = candidate_multiplier

    def search(
        self,
        *,
        text_query: str,
        embedding: Sequence[float],
        k: int = 20,
        backend_filters: Optional[dict] = None,
        post_filter: Optional[HitPredicate] = None,
    ) -> list[FusedHit]:
        per_backend_k = k * self.candidate_multiplier

        vector_hits = self.vector_backend.search(
            embedding, k=per_backend_k, filters=backend_filters
        )
        text_hits = self.fulltext_backend.search(
            text_query, k=per_backend_k, filters=backend_filters
        )

        fused = reciprocal_rank_fusion(vector_hits, text_hits, k=self.rrf_k)

        if post_filter is not None:
            fused = [h for h in fused if post_filter(h)]

        return fused[:k]


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


class _StubVectorBackend:
    """In-memory cosine over precomputed vectors. For demo only."""

    def __init__(self, docs: list[Hit], vectors: dict[str, list[float]]) -> None:
        self.docs = docs
        self.vectors = vectors

    def search(self, embedding, *, k, filters=None):
        import math

        def cos(a, b):
            num = sum(x * y for x, y in zip(a, b))
            da = math.sqrt(sum(x * x for x in a))
            db = math.sqrt(sum(y * y for y in b))
            return num / (da * db) if da and db else 0.0

        scored = []
        for d in self.docs:
            if filters and any(d.metadata.get(k_) != v for k_, v in filters.items()):
                continue
            v = self.vectors.get(d.id)
            if v is None:
                continue
            scored.append(Hit(id=d.id, score=cos(embedding, v), text=d.text, metadata=d.metadata))
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:k]


class _StubFullTextBackend:
    """Naive token-overlap scorer. For demo only."""

    def __init__(self, docs: list[Hit]) -> None:
        self.docs = docs

    def search(self, query: str, *, k, filters=None):
        q_tokens = set(query.lower().split())
        scored = []
        for d in self.docs:
            if filters and any(d.metadata.get(k_) != v for k_, v in filters.items()):
                continue
            d_tokens = set(d.text.lower().split())
            overlap = len(q_tokens & d_tokens)
            if overlap == 0:
                continue
            scored.append(
                Hit(id=d.id, score=float(overlap), text=d.text, metadata=d.metadata)
            )
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:k]


if __name__ == "__main__":
    docs = [
        Hit(id="m1", text="Visits a therapist biweekly via video call.", metadata={"user_id": "u1"}),
        Hit(id="m2", text="Plays guitar in the evenings to unwind.", metadata={"user_id": "u1"}),
        Hit(id="m3", text="Has a corgi named Biscuit.", metadata={"user_id": "u1"}),
        Hit(id="m4", text="Has been considering therapy lately.", metadata={"user_id": "u1"}),
        Hit(id="m5", text="Loves long walks at sunset.", metadata={"user_id": "u2"}),
    ]

    # Synthetic vectors: m1, m4 are both close to the therapy query.
    fake_vectors = {
        "m1": [0.9, 0.1, 0.1],
        "m2": [0.1, 0.9, 0.1],
        "m3": [0.1, 0.1, 0.9],
        "m4": [0.85, 0.15, 0.1],
        "m5": [0.2, 0.7, 0.3],
    }
    query_vec = [0.95, 0.05, 0.05]

    searcher = HybridSearcher(
        vector_backend=_StubVectorBackend(docs, fake_vectors),
        fulltext_backend=_StubFullTextBackend(docs),
    )

    # HyDE: caller injects whatever LLM they want. Stubbed here.
    def stub_llm(prompt: str) -> str:
        return "Visits a therapist biweekly via video call to manage stress."

    expanded = expand_query_with_hyde(
        "do you go to therapy?",
        character_context="Companion: 28, illustrator, calm and observant.",
        llm_fn=stub_llm,
    )
    print(f"HyDE expansion: {expanded}\n")

    results = searcher.search(
        text_query=expanded,
        embedding=query_vec,
        k=5,
        backend_filters={"user_id": "u1"},
        post_filter=compose_filters(filter_by("user_id", "u1")),
    )

    for h in results:
        print(
            f"{h.id}  rrf={h.rrf_score:.4f}  "
            f"vec_rank={h.vector_rank}  ft_rank={h.fulltext_rank}  {h.text}"
        )
