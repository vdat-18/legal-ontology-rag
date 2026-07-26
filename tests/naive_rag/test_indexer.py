"""Tests for embedding-and-indexing chunked JSONL data into Milvus."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pymilvus import MilvusClient

from legal_rag.chunker.models import Chunk
from legal_rag.embedding.encoder import EmbeddingEncoder
from legal_rag.embedding.profiles import EmbeddingProfile
from legal_rag.naive_rag.indexer import _truncate_varchar, index_chunks
from legal_rag.settings import NaiveRagSettings


class _FakeEncoderBackend:
    """Return one deterministic 2-dim vector per input text."""

    def encode(self, sentences: list[str], *, convert_to_numpy: bool, normalize_embeddings: bool) -> np.ndarray:
        """Return a fixed-size vector per sentence."""
        return np.array([[float(len(text)), 0.0] for text in sentences])


def _profile() -> EmbeddingProfile:
    """Build one 2-dim embedding profile for indexer tests."""
    return EmbeddingProfile(
        profile_id="test-profile",
        model_id="test/model",
        model_revision="abc123",
        tokenizer_id="test/model",
        tokenizer_revision="abc123",
        dimensions=2,
        max_input_tokens=512,
        query_instruction=None,
    )


def _chunk(chunk_id: str, text: str) -> Chunk:
    """Build one final chunk for indexer tests."""
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


def _settings(tmp_path: Path) -> NaiveRagSettings:
    """Build naive RAG settings pointed at a temporary Milvus Lite database."""
    return NaiveRagSettings(
        embedding_profile="test-profile",
        top_k=2,
        milvus_uri=str(tmp_path / "naive_rag.db"),
        milvus_nlist=1,
    )


def test_truncate_varchar_keeps_short_text_unchanged() -> None:
    """Text within the byte limit passes through unmodified."""
    assert _truncate_varchar("Vượt đèn đỏ", max_bytes=1000) == "Vượt đèn đỏ"


def test_truncate_varchar_shrinks_text_over_the_byte_limit() -> None:
    """Text over the byte limit is shortened to fit within it."""
    text = "a" * 200

    truncated = _truncate_varchar(text, max_bytes=100)

    assert len(truncated.encode("utf-8")) <= 100


def test_index_chunks_inserts_one_row_per_chunk(tmp_path: Path) -> None:
    """Indexing a fresh collection inserts exactly one row per input chunk."""
    chunks = (_chunk("chunk-1", "Phạt tiền từ 4 đến 6 triệu."), _chunk("chunk-2", "Phạt tiền từ 1 đến 2 triệu."))
    encoder = EmbeddingEncoder(_profile(), _FakeEncoderBackend())
    settings = _settings(tmp_path)
    client = MilvusClient(uri=settings.milvus_uri)

    total = index_chunks(chunks, encoder, client, settings)

    assert total == 2
    assert client.get_collection_stats(settings.milvus_collection_name)["row_count"] == 2


def test_index_chunks_skips_when_collection_already_populated(tmp_path: Path) -> None:
    """A second indexing call against an already-populated collection is a no-op."""
    chunks = (_chunk("chunk-1", "Phạt tiền từ 4 đến 6 triệu."),)
    encoder = EmbeddingEncoder(_profile(), _FakeEncoderBackend())
    settings = _settings(tmp_path)
    client = MilvusClient(uri=settings.milvus_uri)
    index_chunks(chunks, encoder, client, settings)

    total = index_chunks((_chunk("chunk-2", "Một chunk khác."),), encoder, client, settings)

    assert total == 1
    assert client.get_collection_stats(settings.milvus_collection_name)["row_count"] == 1
