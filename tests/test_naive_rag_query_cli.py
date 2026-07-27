"""Tests for the naive RAG baseline query command."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from loguru import logger

from legal_rag import naive_rag_query_cli
from legal_rag.chunker.models import Chunk
from legal_rag.embedding.encoder import EmbeddingEncoder
from legal_rag.embedding.profiles import EmbeddingProfile
from legal_rag.naive_rag import pipeline as pipeline_module
from legal_rag.naive_rag.connection import build_milvus_client
from legal_rag.naive_rag.indexer import index_chunks
from legal_rag.settings import NaiveRagSettings, Settings


class _FakeEncoderBackend:
    """Return one deterministic 2-dim vector per input text."""

    def encode(self, sentences: list[str], *, convert_to_numpy: bool, normalize_embeddings: bool) -> np.ndarray:
        """Return a fixed-size vector per sentence."""
        return np.array([[float(len(text)), 0.0] for text in sentences])


class _FakeGenerationClient:
    """Return a fixed answer, ignoring the retrieved context."""

    def __init__(self, settings: Settings, naive_rag_settings: NaiveRagSettings) -> None:
        return None

    def generate(self, query_vi: str, chunks: object) -> str:
        """Return a fixed answer for any query."""
        return "Phạt tiền từ 4 đến 6 triệu đồng."


def _profile() -> EmbeddingProfile:
    """Build one 2-dim embedding profile for query CLI tests."""
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
    """Build one final chunk for query CLI tests."""
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


def test_query_cli_logs_the_generated_answer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Running the query CLI retrieves the indexed chunk and logs the generated answer."""
    milvus_uri = str(tmp_path / "naive_rag.db")
    naive_rag_settings = NaiveRagSettings(embedding_profile="bge_m3", top_k=1, milvus_uri=milvus_uri, milvus_nlist=1)
    seed_encoder = EmbeddingEncoder(_profile(), _FakeEncoderBackend())
    seed_client = build_milvus_client(naive_rag_settings)
    index_chunks((_chunk("chunk-1", "Phạt tiền từ 4 đến 6 triệu."),), seed_encoder, seed_client, naive_rag_settings)

    monkeypatch.setattr(
        EmbeddingEncoder,
        "from_profile",
        classmethod(lambda cls, profile: EmbeddingEncoder(_profile(), _FakeEncoderBackend())),
    )
    monkeypatch.setattr(pipeline_module, "BamlGenerationClient", _FakeGenerationClient)
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:20128/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("EMBEDDING_PROFILE", "bge_m3")
    monkeypatch.setenv("NAIVE_RAG_TOP_K", "1")
    monkeypatch.setenv("MILVUS_URI", milvus_uri)
    monkeypatch.setattr(sys, "argv", ["legal-naive-rag-query", "Mức phạt vượt đèn đỏ?"])

    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(str(message)), level="INFO")
    try:
        naive_rag_query_cli.main()
    finally:
        logger.remove(sink_id)

    assert any("Phạt tiền từ 4 đến 6 triệu đồng." in message for message in messages)
    assert any("chunk-1" in message for message in messages)
