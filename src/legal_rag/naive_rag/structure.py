"""Bounded article-level context expansion for Naive RAG."""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from legal_rag.chunker.models import Chunk
from legal_rag.chunker.writer import read_chunks
from legal_rag.naive_rag.models import RetrievedChunk


@dataclass(frozen=True, slots=True)
class StructuralUnit:
    """One article-level retrieval unit backed by persisted source chunks."""

    unit_id: str
    text: str
    source_name: str
    evidence_chunk_ids: tuple[str, ...]


class ChunkCatalog:
    """Immutable lookup catalog over parser-owned persisted chunks."""

    def __init__(self, chunks: tuple[Chunk, ...]) -> None:
        """Index unique chunks and their unambiguous article labels."""
        chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        if len(chunks_by_id) != len(chunks):
            msg = "Chunk catalog contains duplicate chunk_id values"
            raise ValueError(msg)
        self._chunks_by_id = chunks_by_id
        self._article_keys = {
            chunk.chunk_id: article_key for chunk in chunks if (article_key := _article_key(chunk)) is not None
        }

    @classmethod
    def from_directory(cls, chunks_dir: Path) -> ChunkCatalog:
        """Load every chunk JSONL file from one directory in stable order."""
        if not chunks_dir.is_dir():
            msg = f"Naive RAG chunks directory does not exist or is not a directory: {chunks_dir}"
            raise ValueError(msg)
        chunk_files = sorted(chunks_dir.glob("*.chunks.jsonl"))
        if not chunk_files:
            msg = f"No *.chunks.jsonl files found in {chunks_dir}"
            raise ValueError(msg)
        chunks = tuple(chunk for path in chunk_files for chunk in read_chunks(path))
        if not chunks:
            msg = f"Chunk catalog is empty: {chunks_dir}"
            raise ValueError(msg)
        logger.info("Loaded hierarchical chunk catalog with {} chunks from {}", len(chunks), chunks_dir)
        return cls(chunks)

    def build_structural_unit(
        self,
        seed: RetrievedChunk,
        *,
        max_chunks: int,
        max_tokens: int,
    ) -> StructuralUnit | None:
        """Expand one seed within its unambiguous contiguous article."""
        source_seed = self._validated_source_seed(seed)
        article_key = self._article_keys.get(source_seed.chunk_id)
        if article_key is None or source_seed.text_token_count > max_tokens:
            return None

        selected = [source_seed]
        selected_ids = {source_seed.chunk_id}
        token_count = source_seed.text_token_count
        previous_id = source_seed.previous_chunk_id
        next_id = source_seed.next_chunk_id
        previous_open = True
        next_open = True

        while len(selected) < max_chunks and (previous_open or next_open):
            if previous_open and len(selected) < max_chunks:
                previous, previous_id = self._next_article_neighbor(
                    previous_id, article_key, selected_ids, previous=True
                )
                if previous is None:
                    previous_open = False
                elif token_count + previous.text_token_count <= max_tokens:
                    selected.append(previous)
                    selected_ids.add(previous.chunk_id)
                    token_count += previous.text_token_count
                else:
                    previous_open = False

            if next_open and len(selected) < max_chunks:
                following, next_id = self._next_article_neighbor(next_id, article_key, selected_ids, previous=False)
                if following is None:
                    next_open = False
                elif token_count + following.text_token_count <= max_tokens:
                    selected.append(following)
                    selected_ids.add(following.chunk_id)
                    token_count += following.text_token_count
                else:
                    next_open = False

        if len(selected) == 1:
            return None

        ordered = tuple(sorted(selected, key=lambda chunk: chunk.chunk_index))
        return StructuralUnit(
            unit_id=_article_unit_id(article_key),
            text=_render_structural_text(source_seed.context_text, ordered),
            source_name=source_seed.source_name,
            evidence_chunk_ids=tuple(chunk.chunk_id for chunk in ordered),
        )

    def _validated_source_seed(self, seed: RetrievedChunk) -> Chunk:
        source_seed = self._chunks_by_id.get(seed.chunk_id)
        if source_seed is None:
            msg = f"Milvus seed chunk {seed.chunk_id!r} is missing from the hierarchical chunk catalog"
            raise ValueError(msg)
        expected = {
            "content_sha256": source_seed.content_sha256,
            "chunker_config_sha256": source_seed.chunker_config_sha256,
            "source_name": source_seed.source_name,
        }
        actual = {
            "content_sha256": seed.content_sha256,
            "chunker_config_sha256": seed.chunker_config_sha256,
            "source_name": seed.source_name,
        }
        mismatches = [name for name, value in expected.items() if actual[name] != value]
        if mismatches:
            fields = ", ".join(mismatches)
            msg = (
                f"Milvus seed chunk {seed.chunk_id!r} does not match the hierarchical chunk catalog "
                f"({fields}); rebuild or select matching corpus data"
            )
            raise ValueError(msg)
        return source_seed

    def _next_article_neighbor(
        self,
        chunk_id: str | None,
        article_key: tuple[str, str],
        selected_ids: set[str],
        *,
        previous: bool,
    ) -> tuple[Chunk | None, str | None]:
        if chunk_id is None or chunk_id in selected_ids:
            return None, None
        chunk = self._chunks_by_id.get(chunk_id)
        if chunk is None or self._article_keys.get(chunk.chunk_id) != article_key:
            return None, None
        following_id = chunk.previous_chunk_id if previous else chunk.next_chunk_id
        return chunk, following_id


def _article_key(chunk: Chunk) -> tuple[str, str] | None:
    context_text = chunk.context_text.strip()
    if not context_text or context_text.startswith("[LEGAL_STRUCTURE_VARIANTS "):
        return None
    article_labels = tuple(
        dict.fromkeys(line.strip() for line in context_text.splitlines() if line.strip().startswith("Điều "))
    )
    if len(article_labels) != 1:
        return None
    normalized = unicodedata.normalize("NFKC", " ".join(article_labels[0].split())).casefold()
    return chunk.scope_id, normalized


def _article_unit_id(article_key: tuple[str, str]) -> str:
    identity = "\0".join(article_key).encode("utf-8")
    return f"article:{hashlib.sha256(identity).hexdigest()}"


def _render_structural_text(context_text: str, chunks: tuple[Chunk, ...]) -> str:
    evidence_texts = tuple(chunk.text.strip() for chunk in chunks if chunk.text.strip())
    heading_lines = tuple(
        line.strip()
        for line in context_text.splitlines()
        if line.strip()
        and not line.strip().startswith("[")
        and not any(line.strip() in evidence_text for evidence_text in evidence_texts)
    )
    return "\n\n".join((*heading_lines, *evidence_texts))
