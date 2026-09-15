"""Tests for vector-similarity retrieval against a Milvus Lite collection."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from pymilvus import MilvusClient

from legal_rag.chunker.models import Chunk
from legal_rag.chunker.writer import write_chunks
from legal_rag.embedding.encoder import EmbeddingEncoder
from legal_rag.embedding.profiles import EmbeddingProfile
from legal_rag.naive_rag.indexer import index_chunks
from legal_rag.naive_rag.retriever import Retriever, SearchMode
from legal_rag.settings import NaiveRagSettings


class _FakeEncoderBackend:
    """Record encode calls and return a deterministic vector per input text."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, sentences: list[str], *, convert_to_numpy: bool, normalize_embeddings: bool) -> np.ndarray:
        """Return a fixed-size vector per sentence, recording the call."""
        self.calls.append(list(sentences))
        return np.array([[float(len(text)), 0.0] for text in sentences])


def _profile() -> EmbeddingProfile:
    """Build one 2-dim embedding profile with a distinct query instruction."""
    return EmbeddingProfile(
        profile_id="test-profile",
        model_id="test/model",
        model_revision="abc123",
        tokenizer_id="test/model",
        tokenizer_revision="abc123",
        dimensions=2,
        max_input_tokens=512,
        query_instruction="QUERY: ",
    )


def _chunk(chunk_id: str, text: str) -> Chunk:
    """Build one final chunk for retriever tests."""
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


def _settings() -> NaiveRagSettings:
    """Build naive RAG retrieval settings."""
    return NaiveRagSettings(
        embedding_profile="test-profile",
        top_k=2,
        milvus_nlist=1,
        milvus_flush_after_insert=False,
    )


def test_retrieve_returns_top_k_chunks_with_populated_fields(tmp_path: Path) -> None:
    """The retriever returns the requested number of chunks with populated fields."""
    chunks = (_chunk("chunk-1", "Phạt tiền từ 4 đến 6 triệu."), _chunk("chunk-2", "Phạt tiền từ 1 đến 2 triệu."))
    encoder = EmbeddingEncoder(_profile(), _FakeEncoderBackend())
    settings = _settings()
    client = MilvusClient(uri=str(tmp_path / "naive_rag.db"))
    index_chunks(chunks, encoder, client, settings)
    retriever = Retriever(encoder, client, settings)

    hits = retriever.retrieve("Mức phạt vượt đèn đỏ", top_k=1)

    assert len(hits) == 1
    assert hits[0].chunk_id in {"chunk-1", "chunk-2"}
    assert hits[0].source_name == "168_2024_ND-CP.docx"
    assert hits[0].content_sha256 == "content-hash"
    assert hits[0].chunker_config_sha256 == "config-hash"


def test_retrieve_encodes_the_query_with_query_side_instruction(tmp_path: Path) -> None:
    """The query is embedded via encode_query, applying the profile's query instruction."""
    chunks = (_chunk("chunk-1", "Phạt tiền từ 4 đến 6 triệu."),)
    backend = _FakeEncoderBackend()
    encoder = EmbeddingEncoder(_profile(), backend)
    settings = _settings()
    client = MilvusClient(uri=str(tmp_path / "naive_rag.db"))
    index_chunks(chunks, encoder, client, settings)
    retriever = Retriever(encoder, client, settings)
    backend.calls.clear()

    retriever.retrieve("Mức phạt vượt đèn đỏ")

    assert backend.calls == [["QUERY: Mức phạt vượt đèn đỏ"]]


def test_hierarchical_retrieval_expands_a_milvus_seed_from_local_chunks(tmp_path: Path) -> None:
    """Milvus identity metadata safely connects one seed to its local article context."""
    first = _chunk("chunk-1", "Điều 7. Nội dung thứ nhất.")
    second = _chunk("chunk-2", "Nội dung thứ hai.")
    context = "[LEGAL_STRUCTURE 1.0.0]\nĐiều 7. Xử phạt xe mô tô\n[/LEGAL_STRUCTURE]"
    first = replace(first, chunk_index=0, context_text=context, next_chunk_id="chunk-2")
    second = replace(second, chunk_index=1, context_text=context, previous_chunk_id="chunk-1")
    chunks = (first, second)
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    write_chunks(chunks_dir / "document.chunks.jsonl", chunks)
    encoder = EmbeddingEncoder(_profile(), _FakeEncoderBackend())
    settings = NaiveRagSettings(
        embedding_profile="test-profile",
        top_k=1,
        milvus_nlist=1,
        milvus_flush_after_insert=False,
        chunks_dir=chunks_dir,
    )
    client = MilvusClient(uri=str(tmp_path / "naive_rag.db"))
    index_chunks(chunks, encoder, client, settings)
    retriever = Retriever(encoder, client, settings)

    [hit] = retriever.retrieve("Nội dung thứ hai", top_k=1, mode=SearchMode.HIERARCHICAL)

    assert hit.unit_kind == "article"
    assert hit.effective_evidence_chunk_ids == ("chunk-1", "chunk-2")
