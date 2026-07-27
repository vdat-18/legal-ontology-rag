"""Tests for the chunk-screening gate."""

from __future__ import annotations

import pytest

from legal_rag.baml_client.types import ChunkScreeningResult
from legal_rag.benchmark_synthesis import chunk_screening
from legal_rag.benchmark_synthesis.chunk_screening import BamlChunkScreeningClient, screen_chunks
from legal_rag.chunker.models import Chunk
from legal_rag.settings import EvaluationSettings, Settings


def _chunk(chunk_id: str, text: str) -> Chunk:
    """Build one minimal chunk for chunk-screening tests."""
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


class _FakeScreeningClient:
    """Return a pre-programmed result per chunk text."""

    def __init__(self, results_by_text: dict[str, ChunkScreeningResult]) -> None:
        self._results_by_text = results_by_text

    def screen(self, chunk_text: str) -> ChunkScreeningResult:
        """Return the result configured for this exact chunk text."""
        return self._results_by_text[chunk_text]


def test_screen_chunks_splits_accepted_and_excluded() -> None:
    """Chunks with penalty content are accepted; the rest are excluded with the model's reason."""
    high = _chunk("chunk-high", "Điều 6 khoản 9 điểm b quy định mức phạt tiền.")
    low = _chunk("chunk-low", "Mục lục")
    client = _FakeScreeningClient(
        {
            high.text: ChunkScreeningResult(has_penalty_content=True, reason="Nêu rõ mức phạt tiền cụ thể."),
            low.text: ChunkScreeningResult(has_penalty_content=False, reason="Đây là mục lục, không có chế tài."),
        }
    )

    accepted, excluded = screen_chunks([high, low], client)

    assert accepted == (high,)
    assert len(excluded) == 1
    assert excluded[0].chunk_id == "chunk-low"
    assert excluded[0].reason == "Đây là mục lục, không có chế tài."


def _settings() -> Settings:
    """Build runtime settings for an offline BAML request."""
    return Settings(
        llm_base_url="http://localhost:20128/v1",
        llm_api_key="provider-test-key",
        llm_model="provider/test-model",
        llm_timeout_seconds=30,
    )


def _evaluation_settings() -> EvaluationSettings:
    """Build ground-truth synthesis settings for an offline BAML request."""
    return EvaluationSettings(
        embedding_profile="qwen3_0_6b",
        span_match_threshold=0.85,
        bundle_overlap_threshold=0.5,
        dedup_cosine_threshold=0.9,
        max_quality_retries=2,
        max_simulator_turns=8,
    )


def test_baml_chunk_screening_client_parses_typed_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """The generated non-stream client returns the typed BAML contract."""
    client = BamlChunkScreeningClient(_settings(), _evaluation_settings())
    monkeypatch.setattr(
        chunk_screening,
        "execute_baml_request",
        lambda *args, **kwargs: '{"has_penalty_content":true,"reason":"Nêu mức phạt tiền cụ thể."}',
    )

    result = client.screen("Điều 6 khoản 9 điểm b quy định mức phạt tiền.")

    assert result.has_penalty_content is True
    assert result.reason == "Nêu mức phạt tiền cụ thể."
