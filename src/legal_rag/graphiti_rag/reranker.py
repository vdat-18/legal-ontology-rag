"""Cross-encoder reranking for graph search, backed by a pinned local model."""

from __future__ import annotations

import asyncio

from graphiti_core.cross_encoder.client import CrossEncoderClient
from sentence_transformers import CrossEncoder


class LocalCrossEncoderReranker(CrossEncoderClient):
    """Score each candidate fact against the query with a local cross-encoder.

    Graphiti's own OpenAI reranker asks the language model for a single token and reads its
    logprobs. That cannot work here: the configured provider spends its completion on reasoning
    before emitting anything, so the one token never arrives and no score comes back. It would
    also cost one provider call per candidate on every query.

    A cross-encoder reads the query and the passage together and returns a relevance score
    directly, which is what reranking actually needs. Running it locally keeps a query to a single
    forward pass over the candidates, with no provider calls and no per-query cost.
    """

    def __init__(self, model_id: str, revision: str) -> None:
        """Load the pinned reranking model.

        The revision is pinned for the same reason the embedding profiles pin theirs: a silently
        updated model changes retrieval results and makes measurements incomparable.
        """
        self._model = CrossEncoder(model_id, revision=revision)

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        """Return the passages ordered by relevance to *query*, each with its score."""
        if not passages:
            return []

        scores = await asyncio.to_thread(self._model.predict, [[query, passage] for passage in passages])
        ranked = [(passage, float(score)) for passage, score in zip(passages, scores, strict=True)]
        ranked.sort(key=lambda scored: scored[1], reverse=True)
        return ranked
