"""Tests for the end-to-end chunk-screening command."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from legal_rag import chunk_screening_cli
from legal_rag.baml_client.types import ChunkScreeningItem, ChunkScreeningResult
from legal_rag.benchmark_synthesis import chunk_screening
from legal_rag.benchmark_synthesis.chunk_screening import ChunkScreeningRecord
from legal_rag.benchmark_synthesis.writer import read_chunk_screening, write_chunk_screening
from legal_rag.chunker.models import Chunk
from legal_rag.chunker.writer import write_chunks


def _chunk(chunk_id: str, text: str) -> Chunk:
    """Build one final chunk as chunk-screening CLI input."""
    return Chunk(
        chunk_id=chunk_id,
        content_sha256="content-hash",
        chunker_config_sha256="config-hash",
        scope_id="scope-id",
        chunk_index=0,
        kind="paragraph_group",
        source_name="168_2024_ND-CP.docx",
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


class _AlternatingScreeningClient:
    """Accept only the first chunk of every batch, ignoring settings."""

    def __init__(self, settings: object, evaluation_settings: object) -> None:
        return None

    def screen(self, chunks: Sequence[ChunkScreeningItem]) -> Sequence[ChunkScreeningResult]:
        """Return has_penalty_content=True only for the batch's first chunk."""
        return [
            ChunkScreeningResult(
                chunk_id=chunk.chunk_id,
                has_penalty_content=index == 0,
                reason="Contains penalty content." if index == 0 else "Table of contents.",
            )
            for index, chunk in enumerate(chunks)
        ]


def _configure_cli(monkeypatch: pytest.MonkeyPatch, chunks_path: Path, screening_output_path: Path) -> None:
    """Set deterministic CLI arguments, environment, and a fake LLM client."""
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:20128/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "provider/test-model")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("EMBEDDING_PROFILE", "qwen3_0_6b")
    monkeypatch.setenv("GROUNDTRUTH_SPAN_MATCH_THRESHOLD", "0.85")
    monkeypatch.setenv("GROUNDTRUTH_BUNDLE_OVERLAP_THRESHOLD", "0.5")
    monkeypatch.setenv("GROUNDTRUTH_MAX_QUALITY_RETRIES", "2")
    monkeypatch.setattr(chunk_screening, "BamlChunkScreeningClient", _AlternatingScreeningClient)
    monkeypatch.setattr(sys, "argv", ["legal-screen-chunks", str(chunks_path), str(screening_output_path)])


def test_chunk_screening_cli_writes_one_verdict_per_chunk(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The command reads chunks, screens every one, and writes a verdict record for each."""
    chunks_path = tmp_path / "168_2024_ND-CP.chunks.jsonl"
    screening_path = tmp_path / "chunk_screening.jsonl"
    write_chunks(
        chunks_path,
        (
            _chunk("chunk-1", "Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng."),
            _chunk("chunk-2", "MỤC LỤC"),
        ),
    )
    _configure_cli(monkeypatch, chunks_path, screening_path)

    chunk_screening_cli.main()

    records = read_chunk_screening(screening_path)
    assert [record.chunk_id for record in records] == ["chunk-1", "chunk-2"]
    assert records[0].has_penalty_content is True
    assert records[1].has_penalty_content is False
    assert records[1].reason == "Table of contents."


def test_chunk_screening_cli_resumes_by_skipping_already_screened_chunks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A chunk with a prior verdict is kept unchanged and is not screened again."""
    chunks_path = tmp_path / "168_2024_ND-CP.chunks.jsonl"
    screening_path = tmp_path / "chunk_screening.jsonl"
    write_chunks(
        chunks_path,
        (
            _chunk("chunk-1", "Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng."),
            _chunk("chunk-2", "MỤC LỤC"),
        ),
    )
    write_chunk_screening(
        screening_path,
        [ChunkScreeningRecord(chunk_id="chunk-1", has_penalty_content=True, reason="Đã sàng lọc trước đó.")],
    )
    screened_chunk_ids: list[str] = []

    class _ResumingClient:
        """Screen only the chunks still passed to it, recording their ids."""

        def __init__(self, settings: object, evaluation_settings: object) -> None:
            return None

        def screen(self, chunks: Sequence[ChunkScreeningItem]) -> Sequence[ChunkScreeningResult]:
            """Record screened chunk ids and return a fixed verdict for each."""
            screened_chunk_ids.extend(chunk.chunk_id for chunk in chunks)
            return [
                ChunkScreeningResult(chunk_id=chunk.chunk_id, has_penalty_content=False, reason="Table of contents.")
                for chunk in chunks
            ]

    _configure_cli(monkeypatch, chunks_path, screening_path)
    monkeypatch.setattr(chunk_screening, "BamlChunkScreeningClient", _ResumingClient)

    chunk_screening_cli.main()

    assert screened_chunk_ids == ["chunk-2"]
    records = read_chunk_screening(screening_path)
    assert [record.chunk_id for record in records] == ["chunk-1", "chunk-2"]
    assert records[0].reason == "Đã sàng lọc trước đó."
    assert records[1].has_penalty_content is False


def test_chunk_screening_cli_writes_partial_output_when_a_batch_keeps_failing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A chunk whose batch keeps failing is left out of the output instead of crashing the run."""
    chunks_path = tmp_path / "168_2024_ND-CP.chunks.jsonl"
    screening_path = tmp_path / "chunk_screening.jsonl"
    first_batch = tuple(_chunk(f"chunk-{i}", f"Phạt tiền {i}.") for i in range(10))
    failing_chunk = _chunk("chunk-fail", "Nội dung gây lỗi mạng.")
    write_chunks(chunks_path, (*first_batch, failing_chunk))

    class _FailsForSecondBatchClient:
        """Succeed on the first batch, raise forever on the batch containing chunk-fail."""

        def __init__(self, settings: object, evaluation_settings: object) -> None:
            return None

        def screen(self, chunks: Sequence[ChunkScreeningItem]) -> Sequence[ChunkScreeningResult]:
            """Raise for the batch containing chunk-fail, otherwise return accepted verdicts."""
            if any(chunk.chunk_id == "chunk-fail" for chunk in chunks):
                msg = "provider timed out"
                raise TimeoutError(msg)
            return [
                ChunkScreeningResult(chunk_id=chunk.chunk_id, has_penalty_content=True, reason="Contains penalty.")
                for chunk in chunks
            ]

    _configure_cli(monkeypatch, chunks_path, screening_path)
    monkeypatch.setattr(chunk_screening, "BamlChunkScreeningClient", _FailsForSecondBatchClient)

    chunk_screening_cli.main()

    records = read_chunk_screening(screening_path)
    assert [record.chunk_id for record in records] == [chunk.chunk_id for chunk in first_batch]
    assert "chunk-fail" not in {record.chunk_id for record in records}


def test_chunk_screening_cli_skips_the_llm_entirely_when_fully_resumed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When every chunk already has a verdict, the LLM client is never constructed."""
    chunks_path = tmp_path / "168_2024_ND-CP.chunks.jsonl"
    screening_path = tmp_path / "chunk_screening.jsonl"
    write_chunks(chunks_path, (_chunk("chunk-1", "Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng."),))
    write_chunk_screening(
        screening_path,
        [ChunkScreeningRecord(chunk_id="chunk-1", has_penalty_content=True, reason="Đã sàng lọc trước đó.")],
    )

    def _fail(*args: object, **kwargs: object) -> None:
        msg = "BamlChunkScreeningClient must not be constructed when every chunk is resumed"
        raise AssertionError(msg)

    _configure_cli(monkeypatch, chunks_path, screening_path)
    monkeypatch.setattr(chunk_screening, "BamlChunkScreeningClient", _fail)

    chunk_screening_cli.main()

    records = read_chunk_screening(screening_path)
    assert [record.chunk_id for record in records] == ["chunk-1"]
    assert records[0].reason == "Đã sàng lọc trước đó."
