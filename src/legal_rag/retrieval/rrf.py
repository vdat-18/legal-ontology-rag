"""Generic Reciprocal Rank Fusion for ranked retrieval results."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class FusedResult(Generic[T]):
    """An item ranked by its fused score and stable first-seen order."""

    item: T
    score: float
    first_seen_order: int


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[T]],
    *,
    key: Callable[[T], Hashable],
    rrf_k: int,
) -> list[FusedResult[T]]:
    """Fuse ranked lists, counting each keyed item at most once per list."""
    if rrf_k < 1:
        msg = "rrf_k must be at least 1"
        raise ValueError(msg)

    scores: dict[Hashable, float] = {}
    representatives: dict[Hashable, T] = {}
    first_seen_orders: dict[Hashable, int] = {}

    for ranking in rankings:
        seen_in_ranking: set[Hashable] = set()
        unique_rank = 0
        for item in ranking:
            item_key = key(item)
            if item_key in seen_in_ranking:
                continue
            seen_in_ranking.add(item_key)
            unique_rank += 1

            if item_key not in representatives:
                representatives[item_key] = item
                first_seen_orders[item_key] = len(first_seen_orders)
            scores[item_key] = scores.get(item_key, 0.0) + 1.0 / (rrf_k + unique_rank)

    ranked_keys = sorted(scores, key=lambda item_key: (-scores[item_key], first_seen_orders[item_key]))
    return [
        FusedResult(
            item=representatives[item_key],
            score=scores[item_key],
            first_seen_order=first_seen_orders[item_key],
        )
        for item_key in ranked_keys
    ]
