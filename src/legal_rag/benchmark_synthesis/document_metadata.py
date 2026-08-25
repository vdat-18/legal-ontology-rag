"""Resolve the document-level metadata that a single chunk's reading window cannot supply."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Protocol

from baml_py.logging import set_log_level
from loguru import logger

from legal_rag.baml_client.sync_client import b
from legal_rag.baml_client.types import DocumentMetadata as BamlDocumentMetadata
from legal_rag.baml_transport import build_client_registry, execute_baml_request
from legal_rag.benchmark_synthesis.chunk_screening import ChunkScreeningRecord
from legal_rag.benchmark_synthesis.config import SynthesisSettings
from legal_rag.benchmark_synthesis.llm_execution import execute_llm_task
from legal_rag.chunker.models import Chunk
from legal_rag.settings import Settings

DOCUMENT_METADATA_SCHEMA_VERSION = "1.0.0"
DOCUMENT_METADATA_PROMPT_VERSION = "5.1.0"


@dataclass(frozen=True, slots=True)
class DocumentMetadataRecord:
    """One document's number and effective dates, stamped onto every fact extracted from it."""

    document_number: str
    valid_from: date | None
    valid_to: date | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable document-metadata record."""
        return {
            "schema_version": DOCUMENT_METADATA_SCHEMA_VERSION,
            "document_number": self.document_number,
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_to": self.valid_to.isoformat() if self.valid_to else None,
        }


class DocumentMetadataClient(Protocol):
    """Boundary implemented by a document-metadata language-model client."""

    def resolve(self, metadata_sections: Sequence[str]) -> BamlDocumentMetadata:
        """Return one document's number and effective dates from selected source text."""


class BamlDocumentMetadataClient:
    """Typed BAML client backed by a runtime OpenAI-compatible provider."""

    def __init__(self, settings: Settings, synthesis_settings: SynthesisSettings) -> None:
        """Create a runtime BAML client from typed, secret-safe configuration."""
        set_log_level("OFF")
        self._registry = build_client_registry(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            max_output_tokens=synthesis_settings.document_metadata.max_output_tokens,
            temperature=synthesis_settings.document_metadata.temperature,
            timeout_seconds=settings.llm_timeout_seconds,
            reasoning_effort=synthesis_settings.document_metadata.reasoning_effort,
        )
        self._timeout_seconds = settings.llm_timeout_seconds
        self._max_response_bytes = synthesis_settings.max_response_bytes

    def resolve(self, metadata_sections: Sequence[str]) -> BamlDocumentMetadata:
        """Run one bounded BAML call and return the typed document metadata."""
        request = b.request.ResolveDocumentMetadata(
            list(metadata_sections), baml_options={"client_registry": self._registry}
        )
        raw_output = execute_baml_request(request, self._timeout_seconds, self._max_response_bytes)
        return b.parse.ResolveDocumentMetadata(raw_output, baml_options={"client_registry": self._registry})


def resolve_document_metadata(
    chunks: Sequence[Chunk],
    screening_records: Sequence[ChunkScreeningRecord],
    client: DocumentMetadataClient,
    *,
    max_retries: int,
    retry_delay_seconds: float,
) -> DocumentMetadataRecord:
    """Resolve metadata only from model-selected root chunks."""
    body_chunks = sorted(
        (chunk for chunk in chunks if chunk.record_scope == "root"),
        key=lambda chunk: (chunk.chunk_index, chunk.chunk_id),
    )
    source_key = chunks[0].source_sha256 if chunks else "empty-document"
    metadata_chunk_ids = {record.chunk_id for record in screening_records if record.has_document_metadata}
    metadata_sections = [chunk.text for chunk in body_chunks if chunk.chunk_id in metadata_chunk_ids]
    logger.info("Document metadata started: metadata_chunks={}", len(metadata_sections))

    def _resolve() -> DocumentMetadataRecord:
        resolved = client.resolve(metadata_sections)
        return DocumentMetadataRecord(
            document_number=resolved.document_number,
            valid_from=date.fromisoformat(resolved.valid_from) if resolved.valid_from else None,
            valid_to=date.fromisoformat(resolved.valid_to) if resolved.valid_to else None,
        )

    metadata = execute_llm_task(
        f"document_metadata:{source_key}",
        _resolve,
        max_retries=max_retries,
        retry_delay_seconds=retry_delay_seconds,
    )
    logger.info("Document metadata completed: {}", metadata.document_number)
    return metadata
