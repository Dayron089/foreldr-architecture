"""
Multiplicative memory scoring for long-running conversational agents.

Why multiplicative, not weighted-sum:

A naive weighted sum
    score = w1*sim + w2*recency + w3*importance + w4*type_bonus
forces every weight to be a compromise. It also has the wrong topology
for this domain: a high `recency` term will lift a poor semantic match
above a great one whenever the weights are set carelessly. The cross-term
the system actually needs - "this is relevant AND temporally alive AND
has been useful before" - does not exist in a linear sum.

The multiplicative formula

    score = relevance * decay * activation * core_boost * category_boost

has three properties a linear sum lacks:

1. Veto. similarity = 0 collapses the entire score to zero. No amount
   of recency or activation can resurrect an irrelevant fact.

2. AND-semantics. A fact that excels on every axis dominates a fact
   that excels on only one. This matches the desired retrieval policy.

3. Bounded interaction. Each factor is normalized into roughly [0, ~1.5];
   no single axis can run away.

The three axes are intentionally orthogonal:
- relevance is purely semantic.
- decay is purely temporal (age and idle time).
- activation is purely behavioral (how often this fact has been useful).

This file is illustrative. Constants are sensible defaults, not
production values; tune to your dataset.
"""

from __future__ import annotations

from datetime import datetime, timezone
from math import log
from typing import Optional, Sequence

import numpy as np
from pydantic import BaseModel, Field, ConfigDict


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class Memory(BaseModel):
    """A single retrievable memory unit."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    fact: str
    context_excerpt: str = ""
    fact_type: str = "user_fact"
    importance: int = Field(default=5, ge=1, le=10)
    is_core: bool = False
    category: Optional[str] = None

    created_at: datetime
    last_accessed: Optional[datetime] = None
    access_count: int = 0

    # Embedding is optional so the model is usable in tests / demos
    # before a vectorizer has been wired up.
    embedding: Optional[list[float]] = None


class ScoringWeights(BaseModel):
    """Tunable scoring parameters.

    Defaults below are illustrative. The spirit of each parameter:

    age_decay_rate
        Half-life slope for age-based decay. Lower = facts age more slowly.

    use_decay_rate
        Slope for use-based decay (how quickly an unused fact fades).
        Multiplied by an importance dampener so high-importance facts
        decay much more slowly on this axis.

    activation_log_coeff
        Slope of the logarithmic activation boost. Higher = more aggressive
        promotion of frequently-recalled facts.

    exploration_bonus
        Score multiplier for facts with access_count == 0. A small bonus
        prevents brand-new facts from being permanently shadowed by older
        ones that happen to have been accessed once or twice.

    activation_cap
        Upper bound on activation. Stops runaway scores for very popular
        facts.

    decay_floor
        Lower bound on the combined decay product. Prevents very old,
        unused facts from collapsing to zero entirely.

    importance_threshold
        Cutoff for "high importance"; identity-level facts above this
        threshold get extra protection from decay.

    backstory_floor_high / backstory_floor_low
        Similarity-conditional floors for backstory facts. Without them,
        old backstory always loses to fresh conversation; with them
        unconditionally, irrelevant backstory always wins.

    core_boost
        Multiplier for facts marked is_core (starred / identity facts).

    category_boost
        Multiplier for facts whose category matches the query's
        classifier-derived categories. Soft signal, not a filter.
    """

    age_decay_rate: float = 0.02
    use_decay_rate: float = 0.05
    activation_log_coeff: float = 0.15
    exploration_bonus: float = 1.05
    activation_cap: float = 1.60
    decay_floor: float = 0.05

    importance_threshold: int = 7

    backstory_floor_high: float = 0.70  # applies when sim >= 0.50
    backstory_floor_low: float = 0.35   # applies when 0.40 <= sim < 0.50

    core_boost: float = 1.25
    category_boost: float = 1.15


# Per-type relevance multipliers. Identity-bearing types lift; noisier
# types attenuate. These are starting points, not gospel.
TYPE_FACTOR: dict[str, float] = {
    "lora_fact": 1.20,        # static character backstory
    "shared_moment": 1.15,
    "user_fact": 1.10,
    "persona_insight": 1.00,
    "relationship_event": 1.00,
    "persona_event": 0.90,
    "activity_change": 0.85,
}


# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity, clamped to [0, 1] for use as a relevance signal."""
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0.0 or nb == 0.0:
        return 0.0
    raw = float(np.dot(va, vb) / (na * nb))
    # Clamp negatives to 0; they're not useful as a signal here.
    return max(0.0, min(1.0, raw))


# ---------------------------------------------------------------------------
# Scoring components
# ---------------------------------------------------------------------------


def _days_between(later: datetime, earlier: datetime) -> float:
    """Non-negative float days between two timestamps (UTC-aware tolerant)."""
    if later.tzinfo is None:
        later = later.replace(tzinfo=timezone.utc)
    if earlier.tzinfo is None:
        earlier = earlier.replace(tzinfo=timezone.utc)
    delta = later - earlier
    return max(0.0, delta.total_seconds() / 86400.0)


def _type_factor(fact_type: str) -> float:
    return TYPE_FACTOR.get(fact_type, 1.0)


def _activation(access_count: int, w: ScoringWeights) -> float:
    """Logarithmic activation with an exploration bonus and a cap."""
    if access_count <= 0:
        return w.exploration_bonus
    if access_count == 1:
        return 1.0
    return min(w.activation_cap, 1.0 + log(access_count) * w.activation_log_coeff)


def _decay(
    days_old: float,
    days_idle: float,
    importance: int,
    fact_type: str,
    similarity: float,
    w: ScoringWeights,
) -> float:
    """
    Combined decay along two axes.

    Age decay is a soft hyperbolic decline. Use decay is the same shape but
    dampened by importance: a fact with importance=10 barely decays from
    idleness; a fact with importance=1 decays at the full rate. The product
    is floored to prevent total collapse, and backstory facts get an
    additional similarity-conditional floor.
    """
    age_decay = 1.0 / (1.0 + days_old * w.age_decay_rate)

    # Importance dampener for use-decay only.
    # Curve: imp=1 -> ~1.0, imp=5 -> ~0.6, imp=8 -> ~0.25, imp=10 -> ~0.15.
    imp_norm = importance / 10.0
    dampener = max(0.15, 1.0 - ((imp_norm - 0.1) / 0.9) ** 1.5)
    effective_use_rate = w.use_decay_rate * dampener
    use_decay = 1.0 / (1.0 + days_idle * effective_use_rate)

    combined = age_decay * use_decay

    # Backstory: similarity-conditional floor.
    # If we floor unconditionally, irrelevant backstory always wins.
    if fact_type == "lora_fact":
        if similarity >= 0.50:
            combined = max(w.backstory_floor_high, combined)
        elif similarity >= 0.40:
            combined = max(w.backstory_floor_low, combined)

    return max(w.decay_floor, combined)


def compute_score(
    memory: Memory,
    query_embedding: Sequence[float],
    now: Optional[datetime] = None,
    *,
    weights: Optional[ScoringWeights] = None,
    query_categories: Optional[Sequence[str]] = None,
) -> float:
    """
    Score a single memory against a query embedding.

    Multiplicative formula:
        score = relevance * decay * activation * core_boost * category_boost

    where
        relevance = cos(memory.embedding, query_embedding)
                    * (importance / 10)
                    * type_factor

    Returns 0.0 for memories without an embedding (cannot be ranked).
    """
    if memory.embedding is None:
        return 0.0

    now = now or datetime.now(timezone.utc)
    w = weights or ScoringWeights()

    # 1. Relevance
    sim = cosine_similarity(memory.embedding, query_embedding)
    importance_norm = max(0.1, memory.importance / 10.0)
    relevance = sim * importance_norm * _type_factor(memory.fact_type)

    # 2. Decay
    days_old = _days_between(now, memory.created_at)
    last = memory.last_accessed or memory.created_at
    days_idle = _days_between(now, last)
    decay = _decay(days_old, days_idle, memory.importance, memory.fact_type, sim, w)

    # 3. Activation
    activation = _activation(memory.access_count, w)

    # 4. Soft boosts
    core_boost = w.core_boost if memory.is_core else 1.0
    category_boost = (
        w.category_boost
        if query_categories and memory.category and memory.category in query_categories
        else 1.0
    )

    return relevance * decay * activation * core_boost * category_boost


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import random

    random.seed(7)

    def fake_vec(seed: int, dim: int = 32) -> list[float]:
        rng = random.Random(seed)
        return [rng.gauss(0, 1) for _ in range(dim)]

    query_vec = fake_vec(42)

    # Build vectors that are deliberately close/far from the query
    def near(seed: int, blend: float) -> list[float]:
        v = fake_vec(seed)
        return [blend * q + (1 - blend) * x for q, x in zip(query_vec, v)]

    now = datetime.now(timezone.utc)
    from datetime import timedelta

    memories = [
        Memory(
            id="A",
            fact="User has a corgi named Biscuit.",
            context_excerpt="Mentioned the dog while describing weekend routines.",
            fact_type="user_fact",
            importance=8,
            is_core=True,
            category="pets",
            created_at=now - timedelta(days=45),
            last_accessed=now - timedelta(days=12),
            access_count=6,
            embedding=near(1, blend=0.85),
        ),
        Memory(
            id="B",
            fact="User mentioned 'the dog' last week.",
            context_excerpt="Brief offhand reference during small talk.",
            fact_type="user_fact",
            importance=4,
            category="pets",
            created_at=now - timedelta(days=6),
            last_accessed=now - timedelta(days=6),
            access_count=0,
            embedding=near(2, blend=0.65),
        ),
        Memory(
            id="C",
            fact="Character read a book on pet psychology.",
            context_excerpt="Discussed it idly during a quiet evening.",
            fact_type="persona_event",
            importance=5,
            category="hobby",
            created_at=now - timedelta(days=30),
            last_accessed=now - timedelta(days=30),
            access_count=1,
            embedding=near(3, blend=0.40),
        ),
        Memory(
            id="D",
            fact="User enjoys cooking pasta on Sundays.",
            context_excerpt="Casual dinner-plan conversation.",
            fact_type="user_fact",
            importance=3,
            category="lifestyle",
            created_at=now - timedelta(days=12),
            last_accessed=now - timedelta(days=12),
            access_count=0,
            embedding=near(4, blend=0.20),
        ),
    ]

    ranked = sorted(
        memories,
        key=lambda m: compute_score(m, query_vec, now, query_categories=["pets"]),
        reverse=True,
    )

    print(f"{'id':>3}  {'score':>7}  {'sim':>5}  {'imp':>3}  {'age':>4}  {'idle':>4}  {'ac':>2}  fact")
    print("-" * 100)
    for m in ranked:
        s = compute_score(m, query_vec, now, query_categories=["pets"])
        sim = cosine_similarity(m.embedding, query_vec)
        age = _days_between(now, m.created_at)
        idle = _days_between(now, m.last_accessed or m.created_at)
        print(
            f"{m.id:>3}  {s:7.3f}  {sim:5.2f}  {m.importance:3d}  "
            f"{age:4.0f}  {idle:4.0f}  {m.access_count:2d}  {m.fact}"
        )
