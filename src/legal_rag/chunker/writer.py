"""Atomic persistence for final retrieval chunks."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from legal_rag.chunker.models import CHUNK_SCHEMA_VERSION, Chunk
from legal_rag.io_utils import write_jsonl_atomic


def write_chunks(output_path: Path, chunks: Sequence[Chunk]) -> None:
    """Atomically replace one UTF-8 JSONL output with deterministic chunk records."""
    write_jsonl_atomic(output_path, (chunk.to_dict() for chunk in chunks))


def read_chunks(input_path: Path) -> tuple[Chunk, ...]:
    """Read a final chunk JSONL file back into typed records."""
    chunks: list[Chunk] = []
    with input_path.open(encoding="utf-8") as source_file:
        for line in source_file:
            payload = json.loads(line)
            if payload["schema_version"] != CHUNK_SCHEMA_VERSION:
                msg = f"Unsupported chunk schema: {payload['schema_version']}"
                raise ValueError(msg)
            chunks.append(
                Chunk(
                    chunk_id=payload["chunk_id"],
                    content_sha256=payload["content_sha256"],
                    chunker_config_sha256=payload["chunker_config_sha256"],
                    scope_id=payload["scope_id"],
                    chunk_index=payload["chunk_index"],
                    kind=payload["kind"],
                    source_name=payload["source_name"],
                    source_sha256=payload["source_sha256"],
                    record_scope=payload["record_scope"],
                    parent_attachment_id=payload["parent_attachment_id"],
                    attachment_member_path=payload["attachment_member_path"],
                    attachment_member_sha256=payload["attachment_member_sha256"],
                    unit_ids=tuple(payload["unit_ids"]),
                    source_refs=tuple(payload["source_refs"]),
                    sequence_start=payload["sequence_start"],
                    sequence_end=payload["sequence_end"],
                    text=payload["text"],
                    context_text=payload["context_text"],
                    embedding_text=payload["embedding_text"],
                    text_token_count=payload["text_token_count"],
                    embedding_token_count=payload["embedding_token_count"],
                    previous_chunk_id=payload["previous_chunk_id"],
                    next_chunk_id=payload["next_chunk_id"],
                )
            )
    return tuple(chunks)
