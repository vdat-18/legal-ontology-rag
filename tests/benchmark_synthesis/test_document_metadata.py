"""Tests for resolving document-level metadata from model-selected chunks."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pytest

from legal_rag.baml_client.types import DocumentMetadata as BamlDocumentMetadata
from legal_rag.benchmark_synthesis import document_metadata
from legal_rag.benchmark_synthesis.chunk_screening import ChunkScreeningRecord
from legal_rag.benchmark_synthesis.config import SynthesisSettings
from legal_rag.benchmark_synthesis.document_metadata import (
    BamlDocumentMetadataClient,
    resolve_document_metadata,
)
from legal_rag.benchmark_synthesis.llm_execution import LlmTaskExhaustedError
from legal_rag.chunker.models import Chunk, RecordScope
from legal_rag.settings import Settings

_SYNTHESIS_CONFIG_PATH = Path(__file__).resolve().parents[2] / "data" / "config" / "groundtruth" / "synthesis_v1.json"


def _chunk(chunk_id: str, text: str, *, chunk_index: int = 0, record_scope: RecordScope = "root") -> Chunk:
    """Build one minimal chunk for document-metadata tests."""
    return Chunk(
        chunk_id=chunk_id,
        content_sha256="content-hash",
        chunker_config_sha256="config-hash",
        scope_id="scope-id",
        chunk_index=chunk_index,
        kind="paragraph_group",
        source_name="168_2024_ND-CP.docx",
        source_sha256="source-hash",
        record_scope=record_scope,
        parent_attachment_id=None,
        attachment_member_path=None,
        attachment_member_sha256=None,
        unit_ids=(f"unit-{chunk_id}",),
        source_refs=(f"word/document.xml#/body/{chunk_id}",),
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


class _RecordingMetadataClient:
    """Record the sections it was given and return a pre-programmed metadata result."""

    def __init__(self, result: BamlDocumentMetadata) -> None:
        self._result = result
        self.metadata_sections: tuple[str, ...] = ()

    def resolve(self, metadata_sections: Sequence[str]) -> BamlDocumentMetadata:
        """Record selected metadata sections and return the configured result."""
        self.metadata_sections = tuple(metadata_sections)
        return self._result


def _baml_metadata(**overrides: object) -> BamlDocumentMetadata:
    """Build one generated BAML document-metadata result."""
    defaults: dict[str, object] = {
        "document_number": "168/2024/NĐ-CP",
        "valid_from": "2025-01-01",
        "valid_to": None,
    }
    defaults.update(overrides)
    return BamlDocumentMetadata(**defaults)


def _screening(chunks: Sequence[Chunk], *metadata_chunk_ids: str) -> tuple[ChunkScreeningRecord, ...]:
    """Build screening records that route selected chunks into metadata resolution."""
    selected = set(metadata_chunk_ids)
    return tuple(
        ChunkScreeningRecord(
            chunk_id=chunk.chunk_id,
            has_penalty_content=False,
            has_document_metadata=chunk.chunk_id in selected,
            reason="Metadata routing fixture.",
        )
        for chunk in chunks
    )


def test_resolve_document_metadata_parses_dates_into_typed_record() -> None:
    """ISO date strings from the model become typed dates, and an absent date stays None."""
    chunks = [_chunk("c1", "Số: 168/2024/NĐ-CP"), _chunk("c2", "Điều 55. Hiệu lực thi hành.")]
    client = _RecordingMetadataClient(_baml_metadata())

    record = resolve_document_metadata(
        chunks,
        _screening(chunks, "c1", "c2"),
        client,
        max_retries=2,
        retry_delay_seconds=0,
    )

    assert record.document_number == "168/2024/NĐ-CP"
    assert record.valid_from == date(2025, 1, 1)
    assert record.valid_to is None


def test_resolve_document_metadata_reads_only_selected_chunks_in_document_order() -> None:
    """Only marked root chunks reach the resolver, regardless of input order."""
    chunks = [_chunk(f"c{index}", f"Đoạn {index}", chunk_index=index) for index in range(10)]
    client = _RecordingMetadataClient(_baml_metadata())

    resolve_document_metadata(
        [chunks[8], chunks[1], chunks[6], chunks[3], chunks[0], chunks[9], chunks[4], chunks[2], chunks[7], chunks[5]],
        _screening(chunks, "c7", "c2", "c4"),
        client,
        max_retries=0,
        retry_delay_seconds=0,
    )

    assert client.metadata_sections == ("Đoạn 2", "Đoạn 4", "Đoạn 7")


def test_resolve_document_metadata_logs_selected_count_and_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Operators can see the selected metadata count and completed document identity."""
    chunks = [_chunk(f"c{index}", f"Đoạn {index}") for index in range(3)]
    client = _RecordingMetadataClient(_baml_metadata())
    messages: list[str] = []

    class _RecordingLogger:
        def info(self, message: str, *args: object) -> None:
            messages.append(message.format(*args))

    monkeypatch.setattr(document_metadata, "logger", _RecordingLogger())

    resolve_document_metadata(
        chunks,
        _screening(chunks, "c1"),
        client,
        max_retries=0,
        retry_delay_seconds=0,
    )

    assert messages == [
        "Document metadata started: metadata_chunks=1",
        "Document metadata completed: 168/2024/NĐ-CP",
    ]


def test_resolve_document_metadata_ignores_attachment_chunks() -> None:
    """Marked attachment chunks do not enter root-document metadata resolution."""
    body = [_chunk("c1", "Số: 168/2024/NĐ-CP"), _chunk("c2", "Điều 55. Hiệu lực thi hành.")]
    attachments = [_chunk(f"a{index}", f"Phụ lục {index}", record_scope="attachment_child") for index in range(5)]
    client = _RecordingMetadataClient(_baml_metadata())

    resolve_document_metadata(
        [*body, *attachments],
        _screening([*body, *attachments], "c2", "a0"),
        client,
        max_retries=2,
        retry_delay_seconds=0,
    )

    assert client.metadata_sections == ("Điều 55. Hiệu lực thi hành.",)


def test_resolve_document_metadata_raises_on_a_malformed_date() -> None:
    """An unparsable date from the model raises instead of being silently dropped."""
    chunks = [_chunk("c1", "Số: 168/2024/NĐ-CP")]
    client = _RecordingMetadataClient(_baml_metadata(valid_from="ngày 01 tháng 01 năm 2025"))

    with pytest.raises(LlmTaskExhaustedError, match="document_metadata:source-hash") as exc_info:
        resolve_document_metadata(
            chunks,
            _screening(chunks),
            client,
            max_retries=2,
            retry_delay_seconds=0,
        )

    assert isinstance(exc_info.value.__cause__, ValueError)


def test_baml_document_metadata_client_parses_typed_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """The generated non-stream client returns one typed document-metadata result."""
    client = BamlDocumentMetadataClient(
        Settings(
            llm_base_url="http://localhost:20128/v1",
            llm_api_key="provider-test-key",
            llm_model="provider/test-model",
            llm_timeout_seconds=30,
        ),
        SynthesisSettings.from_path(_SYNTHESIS_CONFIG_PATH),
    )
    monkeypatch.setattr(
        document_metadata,
        "execute_baml_request",
        lambda *args, **kwargs: '{"document_number":"336/2025/NĐ-CP","valid_from":"2025-08-01","valid_to":null}',
    )

    resolved = client.resolve(["Số: 336/2025/NĐ-CP", "Điều 20. Hiệu lực thi hành."])

    assert resolved.document_number == "336/2025/NĐ-CP"
    assert resolved.valid_from == "2025-08-01"
