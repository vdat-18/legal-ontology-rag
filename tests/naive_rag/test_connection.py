"""Tests for scoping a Milvus client to its configured shared database."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pymilvus import MilvusClient

from legal_rag.chunker.models import Chunk
from legal_rag.datastores import milvus as milvus_module
from legal_rag.datastores.milvus import build_milvus_client
from legal_rag.embedding.encoder import EmbeddingEncoder
from legal_rag.embedding.profiles import EmbeddingProfile
from legal_rag.naive_rag.indexer import index_chunks
from legal_rag.settings import MilvusSettings, NaiveRagSettings


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
        source_name="document.docx",
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


def _settings(tmp_path: Path) -> tuple[MilvusSettings, NaiveRagSettings]:
    """Build shared Milvus and naive RAG settings for connection tests."""
    return (
        MilvusSettings(uri=str(tmp_path / "milvus.db"), database="test_db"),
        NaiveRagSettings(
            embedding_profile="test-profile",
            top_k=5,
            milvus_nlist=1,
            milvus_flush_after_insert=False,
        ),
    )


def test_build_milvus_client_creates_and_isolates_the_configured_database(tmp_path: Path) -> None:
    """Data indexed through the scoped client is invisible from the default database."""
    milvus_settings, naive_rag_settings = _settings(tmp_path)
    encoder = EmbeddingEncoder(_profile(), _FakeEncoderBackend())

    scoped_client = build_milvus_client(milvus_settings)
    index_chunks((_chunk("chunk-1", "First chunk."),), encoder, scoped_client, naive_rag_settings)

    default_client = MilvusClient(uri=milvus_settings.uri)
    assert "test_db" in default_client.list_databases()
    assert default_client.list_collections() == []


def test_build_milvus_client_reuses_an_already_created_database(tmp_path: Path) -> None:
    """Calling build_milvus_client twice against the same database sees prior data."""
    milvus_settings, naive_rag_settings = _settings(tmp_path)
    encoder = EmbeddingEncoder(_profile(), _FakeEncoderBackend())
    first_client = build_milvus_client(milvus_settings)
    index_chunks((_chunk("chunk-1", "First chunk."),), encoder, first_client, naive_rag_settings)

    second_client = build_milvus_client(milvus_settings)

    assert second_client.get_collection_stats(naive_rag_settings.milvus_collection_name)["row_count"] == 1


@pytest.mark.parametrize(
    ("token", "expected_options"),
    [
        (None, {"uri": "http://milvus:19530"}),
        ("root:example", {"uri": "http://milvus:19530", "token": "root:example"}),
    ],
)
def test_build_milvus_client_passes_authentication_only_when_configured(
    monkeypatch: pytest.MonkeyPatch,
    token: str | None,
    expected_options: dict[str, str],
) -> None:
    """The optional Milvus token is forwarded without fallback credentials."""
    captured_options: dict[str, str] = {}

    class _FakeMilvusClient:
        """Record connection options and expose an existing database."""

        def list_databases(self) -> list[str]:
            """Return the configured database as already present."""
            return ["legal_rag"]

        def use_database(self, database_name: str) -> None:
            """Accept the configured database selection."""
            assert database_name == "legal_rag"

    def fake_client(**options: str) -> _FakeMilvusClient:
        captured_options.update(options)
        return _FakeMilvusClient()

    monkeypatch.setattr(milvus_module, "MilvusClient", fake_client)
    settings = MilvusSettings(uri="http://milvus:19530", database="legal_rag", token=token)

    milvus_module.build_milvus_client(settings)

    assert captured_options == expected_options
