"""Tests for the chunk-screening gate."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from legal_rag.baml_client.types import ChunkScreeningItem, ChunkScreeningResult
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
    """Return pre-programmed results keyed by chunk_id, ignoring batch grouping."""

    def __init__(self, results_by_chunk_id: dict[str, ChunkScreeningResult]) -> None:
        self._results_by_chunk_id = results_by_chunk_id

    def screen(self, chunks: Sequence[ChunkScreeningItem]) -> Sequence[ChunkScreeningResult]:
        """Return the configured result for each chunk_id in this batch."""
        return [self._results_by_chunk_id[chunk.chunk_id] for chunk in chunks]


def test_screen_chunks_returns_one_record_per_chunk_in_order() -> None:
    """Every chunk gets a verdict record, in input order, with the model's reason."""
    high = _chunk("chunk-high", "Điều 6 khoản 9 điểm b quy định mức phạt tiền.")
    low = _chunk("chunk-low", "Mục lục")
    client = _FakeScreeningClient(
        {
            "chunk-high": ChunkScreeningResult(
                chunk_id="chunk-high", has_penalty_content=True, reason="Nêu rõ mức phạt tiền cụ thể."
            ),
            "chunk-low": ChunkScreeningResult(
                chunk_id="chunk-low", has_penalty_content=False, reason="Đây là mục lục, không có chế tài."
            ),
        }
    )

    records = screen_chunks([high, low], client)

    assert [record.chunk_id for record in records] == ["chunk-high", "chunk-low"]
    assert records[0].has_penalty_content is True
    assert records[1].has_penalty_content is False
    assert records[1].reason == "Đây là mục lục, không có chế tài."


def test_screen_chunks_batches_calls_by_batch_size() -> None:
    """More chunks than batch_size are screened across multiple calls, preserving order."""
    chunks = [_chunk(f"chunk-{i}", f"Nội dung {i}") for i in range(5)]
    calls: list[int] = []
    results_by_chunk_id = {
        chunk.chunk_id: ChunkScreeningResult(chunk_id=chunk.chunk_id, has_penalty_content=True, reason="")
        for chunk in chunks
    }

    class _CountingClient:
        def screen(self, batch: Sequence[ChunkScreeningItem]) -> Sequence[ChunkScreeningResult]:
            calls.append(len(batch))
            return [results_by_chunk_id[item.chunk_id] for item in batch]

    records = screen_chunks(chunks, _CountingClient(), batch_size=2)

    assert calls == [2, 2, 1]
    assert [record.chunk_id for record in records] == [chunk.chunk_id for chunk in chunks]
    assert all(record.has_penalty_content for record in records)


def test_screen_chunks_retries_a_failing_batch_then_succeeds() -> None:
    """A batch that raises is retried, and a later successful attempt still produces a verdict."""
    chunk = _chunk("chunk-1", "Điều 6 khoản 9 điểm b quy định mức phạt tiền.")
    attempts = 0

    class _FlakyClient:
        def screen(self, batch: Sequence[ChunkScreeningItem]) -> Sequence[ChunkScreeningResult]:
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                msg = "provider timed out"
                raise TimeoutError(msg)
            return [ChunkScreeningResult(chunk_id="chunk-1", has_penalty_content=True, reason="Nêu rõ mức phạt.")]

    records = screen_chunks([chunk], _FlakyClient())

    assert attempts == 2
    assert len(records) == 1
    assert records[0].chunk_id == "chunk-1"
    assert records[0].has_penalty_content is True


def test_screen_chunks_skips_a_batch_that_fails_every_attempt() -> None:
    """A batch that keeps raising is skipped, but other batches are still screened."""
    failing = _chunk("chunk-fail", "Nội dung gây lỗi mạng.")
    ok_one = _chunk("chunk-ok-1", "Điều 6 khoản 9 điểm b quy định mức phạt tiền.")
    ok_two = _chunk("chunk-ok-2", "Mục lục")

    class _FailsOnlyForOneChunkClient:
        def screen(self, batch: Sequence[ChunkScreeningItem]) -> Sequence[ChunkScreeningResult]:
            if batch[0].chunk_id == "chunk-fail":
                msg = "provider timed out"
                raise TimeoutError(msg)
            return [
                ChunkScreeningResult(chunk_id=item.chunk_id, has_penalty_content=True, reason="ok") for item in batch
            ]

    records = screen_chunks([failing, ok_one, ok_two], _FailsOnlyForOneChunkClient(), batch_size=1)

    assert [record.chunk_id for record in records] == ["chunk-ok-1", "chunk-ok-2"]


def test_screen_chunks_respects_max_retries_override() -> None:
    """max_retries=0 means a single attempt; a failure skips the batch immediately."""
    chunk = _chunk("chunk-1", "Điều 6 khoản 9 điểm b quy định mức phạt tiền.")
    attempts = 0

    class _AlwaysFailsClient:
        def screen(self, batch: Sequence[ChunkScreeningItem]) -> Sequence[ChunkScreeningResult]:
            nonlocal attempts
            attempts += 1
            msg = "provider timed out"
            raise TimeoutError(msg)

    records = screen_chunks([chunk], _AlwaysFailsClient(), max_retries=0)

    assert attempts == 1
    assert records == ()


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
        max_quality_retries=2,
    )


def test_baml_chunk_screening_client_parses_typed_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """The generated non-stream client returns the typed BAML contract for a batch."""
    client = BamlChunkScreeningClient(_settings(), _evaluation_settings())
    monkeypatch.setattr(
        chunk_screening,
        "execute_baml_request",
        lambda *args, **kwargs: (
            '[{"chunk_id":"chunk-1","has_penalty_content":true,"reason":"Nêu mức phạt tiền cụ thể."}]'
        ),
    )

    results = client.screen([ChunkScreeningItem(chunk_id="chunk-1", text="Điều 6 khoản 9 điểm b quy định mức phạt.")])

    assert results[0].chunk_id == "chunk-1"
    assert results[0].has_penalty_content is True
    assert results[0].reason == "Nêu mức phạt tiền cụ thể."
