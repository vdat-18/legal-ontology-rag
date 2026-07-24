"""Typed input and output models for the chunker pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

SOURCE_SCHEMA_VERSION = "2.0.0"
CHUNK_SCHEMA_VERSION = "1.0.0"

SourceKind = Literal["paragraph", "table", "image", "spreadsheet_row"]
RecordScope = Literal["root", "attachment_child"]
ChunkKind = Literal["paragraph_group", "table", "image", "spreadsheet_group"]

SUPPORTED_SOURCE_KINDS = frozenset(("paragraph", "table", "image", "spreadsheet_row"))


@dataclass(frozen=True, slots=True)
class SourceRecord:
    """One parser record with normalized scope and ordering fields."""

    unit_id: str
    source_name: str
    source_sha256: str
    source_ref: str
    kind: SourceKind
    record_scope: RecordScope
    order_index: int
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SourceScope:
    """One independently ordered root or attachment-child record stream."""

    scope_id: str
    source_name: str
    source_sha256: str
    record_scope: RecordScope
    root_source_name: str
    root_source_sha256: str
    parent_attachment_id: str | None
    attachment_member_path: str | None
    attachment_member_sha256: str | None
    records: tuple[SourceRecord, ...]


@dataclass(frozen=True, slots=True)
class NormalizedUnit:
    """One searchable source unit before token-aware splitting."""

    scope_id: str
    kind: SourceKind
    sequence_index: int
    unit_ids: tuple[str, ...]
    source_refs: tuple[str, ...]
    text: str
    source_payload: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ChunkFragment:
    """One token-bounded fragment ready for chunk packing."""

    scope_id: str
    kind: ChunkKind
    sequence_start: int
    sequence_end: int
    unit_ids: tuple[str, ...]
    source_refs: tuple[str, ...]
    text: str
    token_count: int


@dataclass(frozen=True, slots=True)
class Chunk:
    """One persisted retrieval chunk with source provenance and neighbor links."""

    chunk_id: str
    content_sha256: str
    chunker_config_sha256: str
    scope_id: str
    chunk_index: int
    kind: ChunkKind
    source_name: str
    source_sha256: str
    record_scope: RecordScope
    parent_attachment_id: str | None
    attachment_member_path: str | None
    attachment_member_sha256: str | None
    unit_ids: tuple[str, ...]
    source_refs: tuple[str, ...]
    sequence_start: int
    sequence_end: int
    text: str
    context_text: str
    embedding_text: str
    text_token_count: int
    embedding_token_count: int
    previous_chunk_id: str | None
    next_chunk_id: str | None

    def to_dict(self) -> dict[str, object]:
        """Return the stable JSON-serializable chunk schema."""
        return {
            "schema_version": CHUNK_SCHEMA_VERSION,
            "chunk_id": self.chunk_id,
            "content_sha256": self.content_sha256,
            "chunker_config_sha256": self.chunker_config_sha256,
            "scope_id": self.scope_id,
            "chunk_index": self.chunk_index,
            "kind": self.kind,
            "source_name": self.source_name,
            "source_sha256": self.source_sha256,
            "record_scope": self.record_scope,
            "parent_attachment_id": self.parent_attachment_id,
            "attachment_member_path": self.attachment_member_path,
            "attachment_member_sha256": self.attachment_member_sha256,
            "unit_ids": list(self.unit_ids),
            "source_refs": list(self.source_refs),
            "sequence_start": self.sequence_start,
            "sequence_end": self.sequence_end,
            "text": self.text,
            "context_text": self.context_text,
            "embedding_text": self.embedding_text,
            "text_token_count": self.text_token_count,
            "embedding_token_count": self.embedding_token_count,
            "previous_chunk_id": self.previous_chunk_id,
            "next_chunk_id": self.next_chunk_id,
        }
