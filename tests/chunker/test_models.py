"""Tests for chunker data models."""

from __future__ import annotations

from legal_rag.chunker.models import CHUNK_SCHEMA_VERSION, Chunk


def test_chunk_to_dict_serializes_stable_output_schema() -> None:
    """Chunk tuples and nullable provenance serialize to JSON-compatible values."""
    chunk = Chunk(
        chunk_id="chunk-id",
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
        text="Nội dung pháp luật.",
        context_text="",
        embedding_text="Nội dung pháp luật.",
        text_token_count=5,
        embedding_token_count=5,
        previous_chunk_id=None,
        next_chunk_id=None,
    )

    record = chunk.to_dict()

    assert record["schema_version"] == CHUNK_SCHEMA_VERSION
    assert record["unit_ids"] == ["unit-0"]
    assert record["source_refs"] == ["word/document.xml#/body/0"]
    assert record["parent_attachment_id"] is None
    assert record["attachment_member_sha256"] is None
