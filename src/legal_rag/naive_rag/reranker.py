"""Cross-encoder reranking for Naive RAG hybrid candidates."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Protocol

from legal_rag.naive_rag.models import RetrievedChunk


class ChunkReranker(Protocol):
    """Rank retrieved chunks against one query."""

    def rerank(self, query: str, candidates: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
        """Return candidates ordered by descending relevance."""


class _CrossEncoderModel(Protocol):
    def predict(self, pairs: list[list[str]], *, batch_size: int) -> Sequence[float]:
        """Return one relevance score per query/passage pair."""


class CrossEncoderReranker:
    """Rerank a bounded candidate list with a pinned local Cross-Encoder model."""

    def __init__(self, model_id: str, revision: str, batch_size: int) -> None:
        """Load the configured model once for reuse across queries."""
        try:
            self._model = _load_cross_encoder(model_id, revision)
        except Exception as exc:
            msg = f"Failed to load reranker model {model_id!r} at revision {revision!r}"
            raise RuntimeError(msg) from exc
        self._model_id = model_id
        self._revision = revision
        self._batch_size = batch_size

    def rerank(self, query: str, candidates: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
        """Score query/chunk pairs and preserve RRF order when scores tie."""
        if not candidates:
            return []

        pairs = [[query, candidate.text] for candidate in candidates]
        try:
            raw_scores = self._model.predict(pairs, batch_size=self._batch_size)
            scores = [float(score) for score in raw_scores]
        except Exception as exc:
            msg = f"Reranker inference failed for {self._model_id!r} at revision {self._revision!r}"
            raise RuntimeError(msg) from exc

        if len(scores) != len(candidates):
            msg = f"Reranker returned {len(scores)} scores for {len(candidates)} candidates"
            raise RuntimeError(msg)

        scored = [
            (index, replace(candidate, score=score))
            for index, (candidate, score) in enumerate(zip(candidates, scores, strict=True))
        ]
        scored.sort(key=lambda item: (-item[1].score, item[0]))
        return [candidate for _, candidate in scored]


def _load_cross_encoder(model_id: str, revision: str) -> _CrossEncoderModel:
    """Import the heavy runtime only when the reranked mode is first used."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_id, revision=revision)
