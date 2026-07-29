"""Tests for embedding-and-indexing chunked JSONL data into Milvus."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from pymilvus import MilvusClient

from legal_rag.chunker.models import Chunk
from legal_rag.embedding.encoder import EmbeddingEncoder
from legal_rag.embedding.profiles import EmbeddingProfile
from legal_rag.naive_rag.indexer import _truncate_varchar, index_chunks
from legal_rag.settings import NaiveRagSettings


class _FakeEncoderBackend:
    """Return one deterministic 2-dim vector per input text."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, sentences: list[str], *, convert_to_numpy: bool, normalize_embeddings: bool) -> np.ndarray:
        """Return a fixed-size vector per sentence."""
        self.calls.append(list(sentences))
        return np.array([[float(len(text)), 0.0] for text in sentences])


def _profile(*, dimensions: int = 2, model_revision: str = "abc123") -> EmbeddingProfile:
    """Build one embedding profile for indexer tests."""
    return EmbeddingProfile(
        profile_id="test-profile",
        model_id="test/model",
        model_revision=model_revision,
        tokenizer_id="test/model",
        tokenizer_revision="abc123",
        dimensions=dimensions,
        max_input_tokens=512,
        query_instruction=None,
    )


def _chunk(
    chunk_id: str,
    text: str,
    *,
    content_sha256: str = "content-hash",
    chunker_config_sha256: str = "config-hash",
) -> Chunk:
    """Build one final chunk for indexer tests."""
    return Chunk(
        chunk_id=chunk_id,
        content_sha256=content_sha256,
        chunker_config_sha256=chunker_config_sha256,
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


def _settings(tmp_path: Path) -> NaiveRagSettings:
    """Build naive RAG settings pointed at a temporary Milvus Lite database."""
    return NaiveRagSettings(
        embedding_profile="test-profile",
        top_k=2,
        milvus_uri=str(tmp_path / "naive_rag.db"),
        milvus_nlist=1,
        milvus_flush_after_insert=False,
    )


def test_truncate_varchar_keeps_short_text_unchanged() -> None:
    """Text within the byte limit passes through unmodified."""
    assert _truncate_varchar("short text", max_bytes=1000) == "short text"


def test_truncate_varchar_shrinks_text_over_the_byte_limit() -> None:
    """Text over the byte limit is shortened to fit within it."""
    text = "a" * 200

    truncated = _truncate_varchar(text, max_bytes=100)

    assert len(truncated.encode("utf-8")) <= 100


def test_index_chunks_inserts_one_row_per_chunk(tmp_path: Path) -> None:
    """Indexing a fresh collection inserts exactly one row per input chunk."""
    chunks = (_chunk("chunk-1", "First chunk."), _chunk("chunk-2", "Second chunk."))
    encoder = EmbeddingEncoder(_profile(), _FakeEncoderBackend())
    settings = _settings(tmp_path)
    client = MilvusClient(uri=settings.milvus_uri)

    total = index_chunks(chunks, encoder, client, settings)

    assert total == 2
    assert client.get_collection_stats(settings.milvus_collection_name)["row_count"] == 2


def test_index_chunks_is_a_no_op_when_the_matching_corpus_is_complete(tmp_path: Path) -> None:
    """A second indexing call against the same complete corpus does not embed again."""
    chunks = (_chunk("chunk-1", "First chunk."),)
    backend = _FakeEncoderBackend()
    encoder = EmbeddingEncoder(_profile(), backend)
    settings = _settings(tmp_path)
    client = MilvusClient(uri=settings.milvus_uri)
    index_chunks(chunks, encoder, client, settings)
    backend.calls.clear()

    total = index_chunks(chunks, encoder, client, settings)

    assert total == 1
    assert client.get_collection_stats(settings.milvus_collection_name)["row_count"] == 1
    assert backend.calls == []


def test_index_chunks_resumes_a_partially_indexed_corpus(tmp_path: Path) -> None:
    """A later run embeds and inserts only chunk IDs missing from a valid partial collection."""
    chunks = (_chunk("chunk-1", "Chunk one."), _chunk("chunk-2", "Chunk two."))
    backend = _FakeEncoderBackend()
    encoder = EmbeddingEncoder(_profile(), backend)
    settings = _settings(tmp_path)
    client = MilvusClient(uri=settings.milvus_uri)
    index_chunks(chunks[:1], encoder, client, settings)
    backend.calls.clear()

    total = index_chunks(chunks, encoder, client, settings)

    assert total == 2
    assert client.get_collection_stats(settings.milvus_collection_name)["row_count"] == 2
    assert backend.calls == [["Chunk two."]]


def test_index_chunks_rejects_rows_outside_the_current_corpus(tmp_path: Path) -> None:
    """A smaller or different corpus cannot silently reuse a collection containing stale rows."""
    chunks = (_chunk("chunk-1", "Chunk one."), _chunk("chunk-2", "Chunk two."))
    encoder = EmbeddingEncoder(_profile(), _FakeEncoderBackend())
    settings = _settings(tmp_path)
    client = MilvusClient(uri=settings.milvus_uri)
    index_chunks(chunks, encoder, client, settings)

    with pytest.raises(ValueError, match="outside the current corpus"):
        index_chunks(chunks[:1], encoder, client, settings)


@pytest.mark.parametrize(
    "changed_chunk",
    [
        _chunk("chunk-1", "Changed.", content_sha256="changed-content"),
        _chunk("chunk-1", "Same text.", chunker_config_sha256="changed-config"),
    ],
)
def test_index_chunks_rejects_stale_chunk_metadata(tmp_path: Path, changed_chunk: Chunk) -> None:
    """A reused chunk ID cannot hide changed content or chunker configuration."""
    encoder = EmbeddingEncoder(_profile(), _FakeEncoderBackend())
    settings = _settings(tmp_path)
    client = MilvusClient(uri=settings.milvus_uri)
    index_chunks((_chunk("chunk-1", "Same text."),), encoder, client, settings)

    with pytest.raises(ValueError, match="stale metadata"):
        index_chunks((changed_chunk,), encoder, client, settings)


def test_index_chunks_rejects_a_changed_embedding_model_revision(tmp_path: Path) -> None:
    """Rows embedded by another model revision cannot be mixed into the collection."""
    settings = _settings(tmp_path)
    client = MilvusClient(uri=settings.milvus_uri)
    index_chunks(
        (_chunk("chunk-1", "Same text."),),
        EmbeddingEncoder(_profile(), _FakeEncoderBackend()),
        client,
        settings,
    )

    with pytest.raises(ValueError, match="embedding_model_revision"):
        index_chunks(
            (_chunk("chunk-1", "Same text."),),
            EmbeddingEncoder(_profile(model_revision="def456"), _FakeEncoderBackend()),
            client,
            settings,
        )


def test_index_chunks_rejects_an_incompatible_vector_dimension(tmp_path: Path) -> None:
    """An existing collection must match the configured embedding vector dimension."""
    settings = _settings(tmp_path)
    client = MilvusClient(uri=settings.milvus_uri)
    index_chunks(
        (_chunk("chunk-1", "Same text."),),
        EmbeddingEncoder(_profile(), _FakeEncoderBackend()),
        client,
        settings,
    )

    with pytest.raises(ValueError, match="vector dimension"):
        index_chunks(
            (_chunk("chunk-1", "Same text."),),
            EmbeddingEncoder(_profile(dimensions=3), _FakeEncoderBackend()),
            client,
            settings,
        )


def test_index_chunks_rejects_an_incompatible_metric(tmp_path: Path) -> None:
    """An existing vector index must use the retrieval metric in typed settings."""
    settings = _settings(tmp_path)
    client = MilvusClient(uri=settings.milvus_uri)
    encoder = EmbeddingEncoder(_profile(), _FakeEncoderBackend())
    chunks = (_chunk("chunk-1", "Same text."),)
    index_chunks(chunks, encoder, client, settings)

    with pytest.raises(ValueError, match="uses metric"):
        index_chunks(chunks, encoder, client, replace(settings, milvus_metric_type="L2"))


def test_index_chunks_rejects_duplicate_input_ids(tmp_path: Path) -> None:
    """Duplicate primary keys are rejected before collection creation or embedding."""
    duplicate_chunks = (_chunk("chunk-1", "One."), _chunk("chunk-1", "Two."))
    encoder = EmbeddingEncoder(_profile(), _FakeEncoderBackend())
    settings = _settings(tmp_path)
    client = MilvusClient(uri=settings.milvus_uri)

    with pytest.raises(ValueError, match="duplicate chunk_id"):
        index_chunks(duplicate_chunks, encoder, client, settings)


def test_index_chunks_rejects_an_empty_corpus(tmp_path: Path) -> None:
    """An empty input directory cannot be reported as a successful index."""
    encoder = EmbeddingEncoder(_profile(), _FakeEncoderBackend())
    settings = _settings(tmp_path)
    client = MilvusClient(uri=settings.milvus_uri)

    with pytest.raises(ValueError, match="no chunks"):
        index_chunks((), encoder, client, settings)
