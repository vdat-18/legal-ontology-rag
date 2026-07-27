"""Tests for scoping a Milvus client to its configured database."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pymilvus import MilvusClient

from legal_rag.chunker.models import Chunk
from legal_rag.embedding.encoder import EmbeddingEncoder
from legal_rag.embedding.profiles import EmbeddingProfile
from legal_rag.naive_rag.connection import build_milvus_client
from legal_rag.naive_rag.indexer import index_chunks
from legal_rag.settings import NaiveRagSettings


class _FakeEncoderBackend:
    """Return one deterministic 2-dim vector per input text."""

    def encode(self, sentences: list[str], *, convert_to_numpy: bool, normalize_embeddings: bool) -> np.ndarray:
        """Return a fixed-size vector per sentence."""
        return np.array([[float(len(text)), 0.0] for text in sentences])


def _profile() -> EmbeddingProfile:
    """Build one 2-dim embedding profile for connection tests."""
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
    """Build one final chunk for connection tests."""
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
    """Build naive RAG settings pointed at a temporary Milvus database."""
    return NaiveRagSettings(
        embedding_profile="bge_m3",
        top_k=5,
        milvus_uri=str(tmp_path / "naive_rag.db"),
        milvus_database="test_db",
        milvus_nlist=1,
    )


def test_build_milvus_client_creates_and_isolates_the_configured_database(tmp_path: Path) -> None:
    """Data indexed through the scoped client is invisible from the default database."""
    settings = _settings(tmp_path)
    encoder = EmbeddingEncoder(_profile(), _FakeEncoderBackend())

    scoped_client = build_milvus_client(settings)
    index_chunks((_chunk("chunk-1", "Phạt tiền từ 4 đến 6 triệu."),), encoder, scoped_client, settings)

    default_client = MilvusClient(uri=settings.milvus_uri)
    assert "test_db" in default_client.list_databases()
    assert default_client.list_collections() == []


def test_build_milvus_client_reuses_an_already_created_database(tmp_path: Path) -> None:
    """Calling build_milvus_client twice against the same database sees the prior data."""
    settings = _settings(tmp_path)
    encoder = EmbeddingEncoder(_profile(), _FakeEncoderBackend())
    first_client = build_milvus_client(settings)
    index_chunks((_chunk("chunk-1", "Phạt tiền từ 4 đến 6 triệu."),), encoder, first_client, settings)

    second_client = build_milvus_client(settings)

    assert second_client.get_collection_stats(settings.milvus_collection_name)["row_count"] == 1
