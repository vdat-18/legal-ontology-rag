"""Tests for local Cross-Encoder reranking."""

from __future__ import annotations

from typing import ClassVar

import pytest

from legal_rag.naive_rag import reranker as reranker_module
from legal_rag.naive_rag.models import RetrievedChunk


def _chunk(chunk_id: str, text: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(chunk_id, text, "source.docx", "paragraph_group", score)


class _FakeCrossEncoder:
    """Return configured scores without loading a model."""

    scores: ClassVar[list[float]] = []
    calls: ClassVar[list[tuple[list[list[str]], int]]] = []

    def predict(self, pairs: list[list[str]], *, batch_size: int) -> list[float]:
        self.calls.append((pairs, batch_size))
        return self.scores


def test_cross_encoder_reranks_candidates_and_preserves_ties(monkeypatch: pytest.MonkeyPatch) -> None:
    """Higher scores win while equal scores retain the incoming RRF order."""
    monkeypatch.setattr(reranker_module, "_load_cross_encoder", lambda _model, _revision: _FakeCrossEncoder())
    _FakeCrossEncoder.scores = [0.5, 0.9, 0.9]
    _FakeCrossEncoder.calls = []
    reranker = reranker_module.CrossEncoderReranker("test/model", "test-revision", 8)

    ranked = reranker.rerank(
        "query",
        [_chunk("a", "text a", 0.03), _chunk("b", "text b", 0.02), _chunk("c", "text c", 0.01)],
    )

    assert [chunk.chunk_id for chunk in ranked] == ["b", "c", "a"]
    assert [chunk.score for chunk in ranked] == [0.9, 0.9, 0.5]
    assert _FakeCrossEncoder.calls == [([["query", "text a"], ["query", "text b"], ["query", "text c"]], 8)]


def test_cross_encoder_skips_inference_for_empty_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty candidate pool returns immediately."""
    monkeypatch.setattr(reranker_module, "_load_cross_encoder", lambda _model, _revision: _FakeCrossEncoder())
    _FakeCrossEncoder.calls = []
    reranker = reranker_module.CrossEncoderReranker("test/model", "test-revision", 8)

    assert reranker.rerank("query", []) == []
    assert _FakeCrossEncoder.calls == []


def test_cross_encoder_rejects_an_invalid_score_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """A model output must contain exactly one scalar score per candidate."""
    monkeypatch.setattr(reranker_module, "_load_cross_encoder", lambda _model, _revision: _FakeCrossEncoder())
    _FakeCrossEncoder.scores = []
    reranker = reranker_module.CrossEncoderReranker("test/model", "test-revision", 8)

    with pytest.raises(RuntimeError, match="0 scores for 1 candidates"):
        reranker.rerank("query", [_chunk("a", "text a", 0.03)])
