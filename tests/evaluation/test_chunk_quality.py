"""Tests for the chunk-quality filtering gate."""

from __future__ import annotations

import pytest

from legal_rag.chunker.models import Chunk
from legal_rag.evaluation import chunk_quality
from legal_rag.evaluation.chunk_quality import BamlChunkQualityClient, ChunkQualityScore, filter_chunks_by_quality
from legal_rag.settings import EvaluationSettings, LlmProvider, Settings


def _chunk(chunk_id: str, text: str) -> Chunk:
    """Build one minimal chunk for chunk-quality tests."""
    return Chunk(
        chunk_id=chunk_id,
        content_sha256="content-hash",
        chunker_config_sha256="config-hash",
        scope_id="scope-id",
        chunk_index=0,
        kind="paragraph_group",
        source_name="source.docx",
        source_sha256="source-hash",
        record_scope="root",
        parent_attachment_id=None,
        attachment_member_path=None,
        attachment_member_sha256=None,
        unit_ids=("unit-0",),
        source_refs=("word/document.xml#/body/0",),
        sequence_start=0,
        sequence_end=0,
        text=text,
        context_text="",
        embedding_text=text,
        text_token_count=5,
        embedding_token_count=5,
        previous_chunk_id=None,
        next_chunk_id=None,
    )


class _FakeQualityClient:
    """Return a pre-programmed score per chunk text."""

    def __init__(self, scores_by_text: dict[str, ChunkQualityScore]) -> None:
        self._scores_by_text = scores_by_text

    def evaluate(self, chunk_text: str) -> ChunkQualityScore:
        """Return the score configured for this exact chunk text."""
        return self._scores_by_text[chunk_text]


def test_chunk_quality_score_average_is_unweighted_mean() -> None:
    """The average property is the plain mean of the four criteria."""
    score = ChunkQualityScore(clarity=1.0, depth=0.5, structure=0.5, relevance=1.0)

    assert score.average == 0.75


def test_filter_chunks_by_quality_splits_accepted_and_excluded() -> None:
    """Chunks at or above threshold are accepted; the rest are excluded with a reason."""
    high = _chunk("chunk-high", "Điều 6 khoản 9 điểm b quy định mức phạt tiền.")
    low = _chunk("chunk-low", "Mục lục")
    client = _FakeQualityClient(
        {
            high.text: ChunkQualityScore(clarity=0.9, depth=0.9, structure=0.9, relevance=0.9),
            low.text: ChunkQualityScore(clarity=0.1, depth=0.0, structure=0.0, relevance=0.1),
        }
    )

    accepted, excluded = filter_chunks_by_quality([high, low], client, threshold=0.6)

    assert accepted == (high,)
    assert len(excluded) == 1
    assert excluded[0].chunk_id == "chunk-low"
    assert "below threshold 0.60" in excluded[0].reason


def test_filter_chunks_by_quality_accepts_score_exactly_at_threshold() -> None:
    """A chunk scoring exactly at the threshold is accepted, not excluded."""
    chunk = _chunk("chunk-boundary", "Điều 7.")
    client = _FakeQualityClient({chunk.text: ChunkQualityScore(0.6, 0.6, 0.6, 0.6)})

    accepted, excluded = filter_chunks_by_quality([chunk], client, threshold=0.6)

    assert accepted == (chunk,)
    assert excluded == ()


def _settings() -> Settings:
    """Build runtime settings for an offline BAML request."""
    return Settings(
        llm_base_url="http://localhost:20128/v1",
        llm_api_key="provider-test-key",
        llm_model="provider/test-model",
        llm_provider=LlmProvider.OPENAI,
        llm_timeout_seconds=30,
    )


def _evaluation_settings() -> EvaluationSettings:
    """Build ground-truth synthesis settings for an offline BAML request."""
    return EvaluationSettings(
        embedding_profile="qwen3_0_6b",
        chunk_quality_threshold=0.6,
        span_match_threshold=0.85,
        bundle_overlap_threshold=0.5,
        dedup_cosine_threshold=0.9,
        max_quality_retries=2,
        max_simulator_turns=8,
    )


def test_baml_chunk_quality_client_parses_typed_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """The generated non-stream client returns the typed BAML contract."""
    client = BamlChunkQualityClient(_settings(), _evaluation_settings())
    monkeypatch.setattr(
        chunk_quality,
        "execute_baml_request",
        lambda *args, **kwargs: '{"clarity":0.9,"depth":0.8,"structure":0.7,"relevance":0.6}',
    )

    score = client.evaluate("Điều 6 khoản 9 điểm b quy định mức phạt tiền.")

    assert score == ChunkQualityScore(clarity=0.9, depth=0.8, structure=0.7, relevance=0.6)
