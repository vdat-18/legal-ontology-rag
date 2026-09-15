"""Tests for bounded article-level context expansion."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from legal_rag.chunker.models import Chunk
from legal_rag.chunker.writer import write_chunks
from legal_rag.naive_rag.models import RetrievedChunk
from legal_rag.naive_rag.structure import ChunkCatalog


def _chunk(
    chunk_id: str,
    index: int,
    *,
    scope_id: str = "scope",
    article: str | None = "Điều 7. Xử phạt xe mô tô",
    tokens: int = 100,
    previous: str | None = None,
    following: str | None = None,
    content_hash: str | None = None,
) -> Chunk:
    context = f"[LEGAL_STRUCTURE 1.0.0]\nChương I\n{article}\n[/LEGAL_STRUCTURE]" if article else ""
    return Chunk(
        chunk_id=chunk_id,
        content_sha256=content_hash or f"hash-{chunk_id}",
        chunker_config_sha256="config-hash",
        scope_id=scope_id,
        chunk_index=index,
        kind="paragraph_group",
        source_name="168.docx",
        source_sha256="source-hash",
        record_scope="root",
        parent_attachment_id=None,
        attachment_member_path=None,
        attachment_member_sha256=None,
        unit_ids=(f"unit-{index}",),
        source_refs=(f"ref-{index}",),
        sequence_start=index,
        sequence_end=index,
        text=f"Nội dung {chunk_id}",
        context_text=context,
        embedding_text=f"Nội dung {chunk_id}",
        text_token_count=tokens,
        embedding_token_count=tokens,
        previous_chunk_id=previous,
        next_chunk_id=following,
    )


def _seed(chunk: Chunk) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk.chunk_id,
        text=chunk.text,
        source_name=chunk.source_name,
        kind=chunk.kind,
        score=0.5,
        content_sha256=chunk.content_sha256,
        chunker_config_sha256=chunk.chunker_config_sha256,
    )


def test_catalog_groups_contiguous_chunks_in_the_same_article() -> None:
    """A seed expands both directions and renders evidence in document order."""
    chunks = (
        _chunk("c1", 1, following="c2"),
        _chunk("c2", 2, previous="c1", following="c3"),
        _chunk("c3", 3, previous="c2"),
    )

    unit = ChunkCatalog(chunks).build_structural_unit(_seed(chunks[1]), max_chunks=5, max_tokens=500)

    assert unit is not None
    assert unit.unit_id.startswith("article:")
    assert unit.evidence_chunk_ids == ("c1", "c2", "c3")
    assert unit.text.index("Nội dung c1") < unit.text.index("Nội dung c2") < unit.text.index("Nội dung c3")
    assert unit.text.count("Điều 7. Xử phạt xe mô tô") == 1


def test_catalog_does_not_cross_article_or_scope_boundaries() -> None:
    """Expansion stops before a different article or document scope."""
    chunks = (
        _chunk("other-scope", 0, scope_id="other", following="seed"),
        _chunk("seed", 1, previous="other-scope", following="other-article"),
        _chunk("other-article", 2, article="Điều 8. Xử phạt xe chuyên dùng", previous="seed"),
    )

    unit = ChunkCatalog(chunks).build_structural_unit(_seed(chunks[1]), max_chunks=5, max_tokens=500)

    assert unit is None


def test_catalog_keeps_missing_or_ambiguous_structure_as_a_leaf() -> None:
    """Only one parser-owned article context is safe to expand."""
    missing = _chunk("missing", 0, article=None)
    ambiguous = _chunk("ambiguous", 1)
    ambiguous = replace(
        ambiguous,
        context_text=(
            "[LEGAL_STRUCTURE_VARIANTS 1.0.0]\n[VARIANT 1]\nĐiều 7. A\n[/VARIANT]\n"
            "[VARIANT 2]\nĐiều 8. B\n[/VARIANT]\n[/LEGAL_STRUCTURE_VARIANTS]"
        ),
    )
    catalog = ChunkCatalog((missing, ambiguous))

    assert catalog.build_structural_unit(_seed(missing), max_chunks=5, max_tokens=500) is None
    assert catalog.build_structural_unit(_seed(ambiguous), max_chunks=5, max_tokens=500) is None


def test_catalog_enforces_chunk_and_token_budgets() -> None:
    """Expansion never exceeds either configured evidence budget."""
    chunks = (
        _chunk("c1", 1, tokens=100, following="c2"),
        _chunk("c2", 2, tokens=100, previous="c1", following="c3"),
        _chunk("c3", 3, tokens=100, previous="c2"),
    )
    catalog = ChunkCatalog(chunks)

    by_chunks = catalog.build_structural_unit(_seed(chunks[1]), max_chunks=2, max_tokens=500)
    by_tokens = catalog.build_structural_unit(_seed(chunks[1]), max_chunks=5, max_tokens=199)

    assert by_chunks is not None
    assert by_chunks.evidence_chunk_ids == ("c1", "c2")
    assert by_tokens is None


def test_catalog_rejects_stale_milvus_metadata() -> None:
    """A seed from a different indexed corpus cannot use the local catalog."""
    chunk = _chunk("seed", 0)
    stale_seed = RetrievedChunk(
        chunk_id=chunk.chunk_id,
        text=chunk.text,
        source_name=chunk.source_name,
        kind=chunk.kind,
        score=0.5,
        content_sha256="stale",
        chunker_config_sha256=chunk.chunker_config_sha256,
    )

    with pytest.raises(ValueError, match="content_sha256"):
        ChunkCatalog((chunk,)).build_structural_unit(stale_seed, max_chunks=5, max_tokens=500)


def test_catalog_loads_chunk_files_and_rejects_invalid_directories(tmp_path: Path) -> None:
    """Directory loading requires at least one non-empty chunk file."""
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    chunk = _chunk("seed", 0)
    write_chunks(chunks_dir / "document.chunks.jsonl", (chunk,))

    catalog = ChunkCatalog.from_directory(chunks_dir)

    assert catalog.build_structural_unit(_seed(chunk), max_chunks=5, max_tokens=500) is None
    with pytest.raises(ValueError, match="does not exist"):
        ChunkCatalog.from_directory(tmp_path / "missing")
    with pytest.raises(ValueError, match=r"No \*\.chunks\.jsonl"):
        ChunkCatalog.from_directory(tmp_path)


def test_catalog_rejects_duplicate_ids_and_stops_neighbor_cycles() -> None:
    """Catalog identity is unique and corrupt neighbor links cannot loop."""
    duplicate = _chunk("duplicate", 0)
    with pytest.raises(ValueError, match="duplicate chunk_id"):
        ChunkCatalog((duplicate, duplicate))

    chunks = (
        _chunk("c1", 1, previous="c2", following="c2"),
        _chunk("c2", 2, previous="c1", following="c1"),
    )
    unit = ChunkCatalog(chunks).build_structural_unit(_seed(chunks[0]), max_chunks=5, max_tokens=500)

    assert unit is not None
    assert unit.evidence_chunk_ids == ("c1", "c2")
