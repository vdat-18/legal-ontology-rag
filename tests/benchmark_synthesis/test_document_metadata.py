"""Tests for resolving document-level metadata from a document's opening and closing chunks."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pytest

from legal_rag.baml_client.types import DocumentMetadata as BamlDocumentMetadata
from legal_rag.benchmark_synthesis import document_metadata
from legal_rag.benchmark_synthesis.config import SynthesisSettings
from legal_rag.benchmark_synthesis.document_metadata import (
    BamlDocumentMetadataClient,
    resolve_document_metadata,
)
from legal_rag.benchmark_synthesis.llm_execution import LlmTaskExhaustedError
from legal_rag.chunker.models import Chunk, RecordScope
from legal_rag.settings import Settings

_SYNTHESIS_CONFIG_PATH = Path(__file__).resolve().parents[2] / "data" / "config" / "groundtruth" / "synthesis_v1.json"


def _chunk(chunk_id: str, text: str, *, record_scope: RecordScope = "root") -> Chunk:
    """Build one minimal chunk for document-metadata tests."""
    return Chunk(
        chunk_id=chunk_id,
        content_sha256="content-hash",
        chunker_config_sha256="config-hash",
        scope_id="scope-id",
        chunk_index=0,
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
        self.opening_sections: tuple[str, ...] = ()
        self.closing_sections: tuple[str, ...] = ()

    def resolve(self, opening_sections: Sequence[str], closing_sections: Sequence[str]) -> BamlDocumentMetadata:
        """Record both section lists and return the configured result."""
        self.opening_sections = tuple(opening_sections)
        self.closing_sections = tuple(closing_sections)
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


def test_resolve_document_metadata_parses_dates_into_typed_record() -> None:
    """ISO date strings from the model become typed dates, and an absent date stays None."""
    chunks = [_chunk("c1", "Số: 168/2024/NĐ-CP"), _chunk("c2", "Điều 55. Hiệu lực thi hành.")]
    client = _RecordingMetadataClient(_baml_metadata())

    record = resolve_document_metadata(chunks, client, opening_chunk_count=3, closing_chunk_count=5, max_retries=2)

    assert record.document_number == "168/2024/NĐ-CP"
    assert record.valid_from == date(2025, 1, 1)
    assert record.valid_to is None


def test_resolve_document_metadata_reads_the_opening_and_closing_of_the_document() -> None:
    """The opening sections come from the first chunks and the closing sections from the last."""
    chunks = [_chunk(f"c{index}", f"Đoạn {index}") for index in range(10)]
    client = _RecordingMetadataClient(_baml_metadata())

    resolve_document_metadata(chunks, client, opening_chunk_count=2, closing_chunk_count=3, max_retries=2)

    assert client.opening_sections == ("Đoạn 0", "Đoạn 1")
    assert client.closing_sections == ("Đoạn 7", "Đoạn 8", "Đoạn 9")


def test_resolve_document_metadata_ignores_attachment_chunks() -> None:
    """Attachment chunks are excluded so the closing sections stay inside the document body."""
    body = [_chunk("c1", "Số: 168/2024/NĐ-CP"), _chunk("c2", "Điều 55. Hiệu lực thi hành.")]
    attachments = [_chunk(f"a{index}", f"Phụ lục {index}", record_scope="attachment_child") for index in range(5)]
    client = _RecordingMetadataClient(_baml_metadata())

    resolve_document_metadata(
        [*body, *attachments], client, opening_chunk_count=1, closing_chunk_count=1, max_retries=2
    )

    assert client.opening_sections == ("Số: 168/2024/NĐ-CP",)
    assert client.closing_sections == ("Điều 55. Hiệu lực thi hành.",)


def test_resolve_document_metadata_raises_on_a_malformed_date() -> None:
    """An unparsable date from the model raises instead of being silently dropped."""
    chunks = [_chunk("c1", "Số: 168/2024/NĐ-CP")]
    client = _RecordingMetadataClient(_baml_metadata(valid_from="ngày 01 tháng 01 năm 2025"))

    with pytest.raises(LlmTaskExhaustedError, match="document_metadata:source-hash") as exc_info:
        resolve_document_metadata(chunks, client, opening_chunk_count=3, closing_chunk_count=5, max_retries=2)

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

    resolved = client.resolve(["Số: 336/2025/NĐ-CP"], ["Điều 20. Hiệu lực thi hành."])

    assert resolved.document_number == "336/2025/NĐ-CP"
    assert resolved.valid_from == "2025-08-01"
