"""Tests for the combined document-preparation command."""

from __future__ import annotations

import sys
import threading
from collections.abc import Sequence
from pathlib import Path

import pytest

from legal_rag import document_preparation_cli
from legal_rag.baml_client.types import ChunkScreeningItem, ChunkScreeningResult
from legal_rag.baml_client.types import DocumentMetadata as BamlDocumentMetadata
from legal_rag.benchmark_synthesis import chunk_screening, document_metadata, document_preparation
from legal_rag.benchmark_synthesis.stage_cache import stage_cache_path
from legal_rag.benchmark_synthesis.writer import read_chunk_screening, read_document_metadata
from legal_rag.chunker.models import Chunk
from legal_rag.chunker.writer import write_chunks

_SYNTHESIS_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "config" / "groundtruth" / "synthesis_v1.json"
)


def _chunk(chunk_id: str, text: str) -> Chunk:
    """Build one final chunk as document-preparation CLI input."""
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


def test_document_preparation_cli_runs_concurrently_and_reuses_completed_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The combined CLI runs both calls concurrently and resumes without reconstructing either client."""
    chunks_path = tmp_path / "168_2024_ND-CP.chunks.jsonl"
    screening_path = tmp_path / "chunk_screening.jsonl"
    metadata_path = tmp_path / "document_metadata.jsonl"
    barrier = threading.Barrier(2)
    calls: list[str] = []
    log_messages: list[str] = []

    class _RecordingLogger:
        def info(self, message: str, *args: object) -> None:
            log_messages.append(message.format(*args))

    monkeypatch.setattr(document_preparation, "logger", _RecordingLogger())
    write_chunks(
        chunks_path,
        (
            _chunk("chunk-1", "CHÍNH PHỦ Số: 168/2024/NĐ-CP"),
            _chunk("chunk-2", "Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng."),
        ),
    )

    class _ConcurrentScreeningClient:
        def __init__(self, settings: object, synthesis_settings: object) -> None:
            return None

        def screen(self, chunks: Sequence[ChunkScreeningItem]) -> Sequence[ChunkScreeningResult]:
            calls.append("screening")
            barrier.wait(timeout=2)
            return tuple(
                ChunkScreeningResult(
                    chunk_id=chunk.chunk_id,
                    has_penalty_content=chunk.chunk_id == "chunk-2",
                    reason="Contains penalty." if chunk.chunk_id == "chunk-2" else "Opening metadata.",
                )
                for chunk in chunks
            )

    class _ConcurrentMetadataClient:
        def __init__(self, settings: object, synthesis_settings: object) -> None:
            return None

        def resolve(self, opening_sections: object, closing_sections: object) -> BamlDocumentMetadata:
            calls.append("metadata")
            barrier.wait(timeout=2)
            return BamlDocumentMetadata(document_number="168/2024/NĐ-CP", valid_from="2025-01-01", valid_to=None)

    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:20128/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "provider/test-model")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "30")
    monkeypatch.setattr(chunk_screening, "BamlChunkScreeningClient", _ConcurrentScreeningClient)
    monkeypatch.setattr(document_metadata, "BamlDocumentMetadataClient", _ConcurrentMetadataClient)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "legal-prepare-groundtruth-document",
            str(chunks_path),
            str(screening_path),
            str(metadata_path),
            "--synthesis-config",
            str(_SYNTHESIS_CONFIG_PATH),
        ],
    )

    document_preparation_cli.main()

    assert set(calls) == {"screening", "metadata"}
    assert [record.chunk_id for record in read_chunk_screening(screening_path)] == ["chunk-1", "chunk-2"]
    assert read_document_metadata(metadata_path).document_number == "168/2024/NĐ-CP"
    assert stage_cache_path(screening_path).exists()
    assert stage_cache_path(metadata_path).exists()
    assert "Step 1 started for 168_2024_ND-CP.docx: screening and metadata run concurrently" in log_messages
    assert "Step 1 work units completed for 168_2024_ND-CP.docx" in log_messages

    def _fail(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Completed document-preparation outputs must bypass both LLM clients")

    monkeypatch.setattr(chunk_screening, "BamlChunkScreeningClient", _fail)
    monkeypatch.setattr(document_metadata, "BamlDocumentMetadataClient", _fail)

    document_preparation_cli.main()

    assert "Chunk screening resumed: 2/2 chunks already completed" in log_messages
    assert f"Document metadata reused from {metadata_path}" in log_messages
