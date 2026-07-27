"""End-to-end tests for the naive RAG pipeline: retrieve then generate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from legal_rag.chunker.models import Chunk
from legal_rag.embedding.encoder import EmbeddingEncoder
from legal_rag.embedding.profiles import EmbeddingProfile
from legal_rag.naive_rag import pipeline as pipeline_module
from legal_rag.naive_rag.connection import build_milvus_client
from legal_rag.naive_rag.indexer import index_chunks
from legal_rag.naive_rag.pipeline import NaiveRAGPipeline
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
    """Build one 2-dim embedding profile for pipeline tests."""
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
    """Build one final chunk for pipeline tests."""
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


def test_pipeline_run_retrieves_and_generates_an_answer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Running the pipeline retrieves the indexed chunk and returns the generator's answer."""
    naive_rag_settings = NaiveRagSettings(
        embedding_profile="bge_m3",
        top_k=1,
        milvus_uri=str(tmp_path / "naive_rag.db"),
        milvus_nlist=1,
    )
    settings = Settings(
        llm_base_url="http://localhost:20128/v1", llm_api_key="test-key", llm_model="test-model", llm_timeout_seconds=30
    )
    monkeypatch.setattr(
        EmbeddingEncoder,
        "from_profile",
        classmethod(lambda cls, profile: EmbeddingEncoder(_profile(), _FakeEncoderBackend())),
    )
    monkeypatch.setattr(pipeline_module, "BamlGenerationClient", _FakeGenerationClient)

    seed_encoder = EmbeddingEncoder(_profile(), _FakeEncoderBackend())
    seed_client = build_milvus_client(naive_rag_settings)
    index_chunks((_chunk("chunk-1", "Phạt tiền từ 4 đến 6 triệu."),), seed_encoder, seed_client, naive_rag_settings)

    pipeline = NaiveRAGPipeline(settings, naive_rag_settings)
    result = pipeline.run("Mức phạt vượt đèn đỏ?")

    assert result.query == "Mức phạt vượt đèn đỏ?"
    assert result.answer == "Phạt tiền từ 4 đến 6 triệu đồng."
    assert len(result.sources) == 1
    assert result.sources[0].chunk_id == "chunk-1"
