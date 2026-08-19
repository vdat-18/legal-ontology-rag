"""Tests for the end-to-end document-metadata command."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

from legal_rag import document_metadata_cli
from legal_rag.baml_client.types import DocumentMetadata as BamlDocumentMetadata
from legal_rag.benchmark_synthesis import document_metadata
from legal_rag.benchmark_synthesis.stage_cache import stage_cache_path
from legal_rag.benchmark_synthesis.writer import read_document_metadata
from legal_rag.chunker.models import Chunk
from legal_rag.chunker.writer import write_chunks

_SYNTHESIS_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "config" / "groundtruth" / "synthesis_v1.json"
)


def _chunk(chunk_id: str, text: str) -> Chunk:
    """Build one final chunk as document-metadata CLI input."""
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


class _FakeMetadataClient:
    """Return one pre-programmed document-metadata result."""

    def __init__(self, settings: object, synthesis_settings: object) -> None:
        return None

    def resolve(self, opening_sections: object, closing_sections: object) -> BamlDocumentMetadata:
        """Return the document number and effective date of the sampled document."""
        return BamlDocumentMetadata(document_number="168/2024/NĐ-CP", valid_from="2025-01-01", valid_to=None)


def test_document_metadata_cli_resolves_and_writes_one_record(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The command reads a document's chunks and writes exactly one typed metadata record."""
    chunks_path = tmp_path / "168_2024_ND-CP.chunks.jsonl"
    metadata_path = tmp_path / "document_metadata.jsonl"
    write_chunks(
        chunks_path,
        (
            _chunk("chunk-1", "CHÍNH PHỦ Số: 168/2024/NĐ-CP"),
            _chunk("chunk-2", "Điều 55. Hiệu lực thi hành."),
        ),
    )
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:20128/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "provider/test-model")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "30")
    monkeypatch.setattr(document_metadata, "BamlDocumentMetadataClient", _FakeMetadataClient)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "legal-resolve-document-metadata",
            str(chunks_path),
            str(metadata_path),
            "--synthesis-config",
            str(_SYNTHESIS_CONFIG_PATH),
        ],
    )

    document_metadata_cli.main()

    record = read_document_metadata(metadata_path)
    assert record.document_number == "168/2024/NĐ-CP"
    assert record.valid_from == date(2025, 1, 1)
    assert record.valid_to is None
    assert stage_cache_path(metadata_path).exists()
