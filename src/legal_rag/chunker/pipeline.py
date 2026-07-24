"""Pack bounded source fragments into deterministic retrieval chunks."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from legal_rag.chunker.models import CHUNK_SCHEMA_VERSION, Chunk, ChunkFragment, SourceScope
from legal_rag.chunker.reader import read_source_scopes
from legal_rag.chunker.rendering import SPREADSHEET_RENDERING_VERSION, TABLE_RENDERING_VERSION, render_source_scope
from legal_rag.chunker.splitting import split_units
from legal_rag.embedding.tokenizer import EmbeddingTokenizer
from legal_rag.parser.hashing import sha256_json
from legal_rag.settings import ChunkerSettings

CHUNKING_ALGORITHM_VERSION = "1.0.0"


def chunk_source_file(
    input_path: Path,
    tokenizer: EmbeddingTokenizer,
    settings: ChunkerSettings,
) -> tuple[Chunk, ...]:
    """Run the complete in-memory chunking pipeline for one parser JSONL file."""
    chunks: list[Chunk] = []
    for scope in read_source_scopes(input_path):
        units = render_source_scope(scope)
        fragments = split_units(
            units,
            tokenizer,
            max_tokens=settings.max_tokens,
            overlap_tokens=settings.split_overlap_tokens,
        )
        chunks.extend(
            pack_scope_fragments(
                scope,
                fragments,
                tokenizer,
                max_tokens=settings.max_tokens,
                split_overlap_tokens=settings.split_overlap_tokens,
            )
        )
    return tuple(chunks)


def pack_scope_fragments(
    scope: SourceScope,
    fragments: Sequence[ChunkFragment],
    tokenizer: EmbeddingTokenizer,
    *,
    max_tokens: int,
    split_overlap_tokens: int,
) -> tuple[Chunk, ...]:
    """Greedily pack one isolated source scope and assign deterministic links."""
    tokenizer.validate_window_settings(
        max_tokens=max_tokens,
        overlap_tokens=split_overlap_tokens,
    )
    config_sha256 = sha256_json(
        {
            "algorithm_version": CHUNKING_ALGORITHM_VERSION,
            "chunk_schema_version": CHUNK_SCHEMA_VERSION,
            "max_tokens": max_tokens,
            "split_overlap_tokens": split_overlap_tokens,
            "spreadsheet_rendering_version": SPREADSHEET_RENDERING_VERSION,
            "table_rendering_version": TABLE_RENDERING_VERSION,
            "tokenizer_id": tokenizer.profile.tokenizer_id,
            "tokenizer_revision": tokenizer.profile.tokenizer_revision,
        }
    )

    groups: list[list[ChunkFragment]] = []
    current: list[ChunkFragment] = []
    for fragment in fragments:
        if current and _can_pack(current[-1], fragment):
            candidate_text = "\n\n".join(item.text for item in (*current, fragment))
            if tokenizer.fits_within(candidate_text, max_tokens=max_tokens):
                current.append(fragment)
                continue
        if current:
            groups.append(current)
        current = [fragment]
    if current:
        groups.append(current)

    chunks: list[Chunk] = []
    for chunk_index, group in enumerate(groups):
        text = "\n\n".join(fragment.text for fragment in group)
        content_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        unit_ids = tuple(dict.fromkeys(unit_id for fragment in group for unit_id in fragment.unit_ids))
        source_refs = tuple(dict.fromkeys(source_ref for fragment in group for source_ref in fragment.source_refs))
        chunk_id = sha256_json(
            {
                "chunker_config_sha256": config_sha256,
                "content_sha256": content_sha256,
                "kind": group[0].kind,
                "scope_id": scope.scope_id,
                "sequence_end": group[-1].sequence_end,
                "sequence_start": group[0].sequence_start,
                "source_refs": source_refs,
                "unit_ids": unit_ids,
            }
        )
        token_count = tokenizer.count_tokens(text)
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                content_sha256=content_sha256,
                chunker_config_sha256=config_sha256,
                scope_id=scope.scope_id,
                chunk_index=chunk_index,
                kind=group[0].kind,
                source_name=scope.source_name,
                source_sha256=scope.source_sha256,
                record_scope=scope.record_scope,
                parent_attachment_id=scope.parent_attachment_id,
                attachment_member_path=scope.attachment_member_path,
                attachment_member_sha256=scope.attachment_member_sha256,
                unit_ids=unit_ids,
                source_refs=source_refs,
                sequence_start=group[0].sequence_start,
                sequence_end=group[-1].sequence_end,
                text=text,
                context_text="",
                embedding_text=text,
                text_token_count=token_count,
                embedding_token_count=token_count,
                previous_chunk_id=None,
                next_chunk_id=None,
            )
        )

    return tuple(
        replace(
            chunk,
            previous_chunk_id=chunks[index - 1].chunk_id if index else None,
            next_chunk_id=chunks[index + 1].chunk_id if index + 1 < len(chunks) else None,
        )
        for index, chunk in enumerate(chunks)
    )


def _can_pack(previous: ChunkFragment, current: ChunkFragment) -> bool:
    """Return whether adjacent fragments share a supported packing boundary."""
    if previous.kind != current.kind or current.kind == "image":
        return False
    if current.kind == "paragraph_group":
        return True
    return previous.packing_key == current.packing_key
