"""Tests for the naive RAG baseline indexing command."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from legal_rag import naive_rag_index_cli
from legal_rag.chunker.models import Chunk
from legal_rag.chunker.writer import write_chunks
from legal_rag.embedding.encoder import EmbeddingEncoder
from legal_rag.embedding.profiles import EmbeddingProfile
from legal_rag.naive_rag.connection import build_milvus_client
from legal_rag.settings import NaiveRagSettings


class _FakeEncoderBackend:
    """Return one deterministic 2-dim vector per input text."""

    def encode(self, sentences: list[str], *, convert_to_numpy: bool, normalize_embeddings: bool) -> np.ndarray:
        """Return a fixed-size vector per sentence."""
        return np.array([[float(len(text)), 0.0] for text in sentences])


def _profile() -> EmbeddingProfile:
    """Build one 2-dim embedding profile for index CLI tests."""
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
    """Build one final chunk for index CLI tests."""
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


def test_index_cli_indexes_every_chunk_file_in_the_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Running the index CLI embeds and inserts every chunk from every *.chunks.jsonl file."""
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    write_chunks(chunks_dir / "168_2024_ND-CP.chunks.jsonl", [_chunk("chunk-1", "Phạt tiền từ 4 đến 6 triệu.")])
    milvus_uri = str(tmp_path / "naive_rag.db")

    monkeypatch.setattr(
        EmbeddingEncoder,
        "from_profile",
        classmethod(lambda cls, profile: EmbeddingEncoder(_profile(), _FakeEncoderBackend())),
    )
    monkeypatch.setenv("EMBEDDING_PROFILE", "bge_m3")
    monkeypatch.setenv("NAIVE_RAG_TOP_K", "5")
    monkeypatch.setenv("NAIVE_RAG_MILVUS_FLUSH_AFTER_INSERT", "false")
    monkeypatch.setenv("NAIVE_RAG_MILVUS_URI", milvus_uri)
    monkeypatch.setattr(sys, "argv", ["legal-naive-rag-index", str(chunks_dir)])

    naive_rag_index_cli.main()

    client = build_milvus_client(NaiveRagSettings(embedding_profile="bge_m3", top_k=5, milvus_uri=milvus_uri))
    assert client.get_collection_stats("naive_rag_chunks")["row_count"] == 1


def test_index_cli_second_run_skips_reindexing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Running the index CLI twice against the same database leaves row count unchanged."""
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    write_chunks(chunks_dir / "168_2024_ND-CP.chunks.jsonl", [_chunk("chunk-1", "Phạt tiền từ 4 đến 6 triệu.")])
    milvus_uri = str(tmp_path / "naive_rag.db")

    monkeypatch.setattr(
        EmbeddingEncoder,
        "from_profile",
        classmethod(lambda cls, profile: EmbeddingEncoder(_profile(), _FakeEncoderBackend())),
    )
    monkeypatch.setenv("EMBEDDING_PROFILE", "bge_m3")
    monkeypatch.setenv("NAIVE_RAG_TOP_K", "5")
    monkeypatch.setenv("NAIVE_RAG_MILVUS_FLUSH_AFTER_INSERT", "false")
    monkeypatch.setenv("NAIVE_RAG_MILVUS_URI", milvus_uri)
    monkeypatch.setattr(sys, "argv", ["legal-naive-rag-index", str(chunks_dir)])
    naive_rag_index_cli.main()

    naive_rag_index_cli.main()

    client = build_milvus_client(NaiveRagSettings(embedding_profile="bge_m3", top_k=5, milvus_uri=milvus_uri))
    assert client.get_collection_stats("naive_rag_chunks")["row_count"] == 1
