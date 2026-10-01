"""Tests for Corrective RAG Milvus schema and metadata compatibility checks."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from legal_rag.chunker.models import Chunk
from legal_rag.chunker.writer import write_chunks
from legal_rag.corrective_rag.retrieval import database as database_module
from legal_rag.corrective_rag.retrieval.database import LegalVectorDB


class _FakeEmbedder:
    """Return deterministic vectors while exposing pinned embedding metadata."""

    def __init__(self, *, dimensions: int = 2, model_revision: str = "abc123") -> None:
        self.profile_id = "test-profile"
        self.model_id = "test/model"
        self.model_revision = model_revision
        self.dimensions = dimensions
        self.document_calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one fixed-size vector per document text."""
        self.document_calls.append(list(texts))
        return [[float(len(text)), *([0.0] * (self.dimensions - 1))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Return one fixed-size query vector."""
        return [float(len(text)), *([0.0] * (self.dimensions - 1))]


def test_constructor_closes_client_when_collection_setup_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A client created before collection setup fails is closed."""
    setup_error = RuntimeError("collection setup failed")

    class _FakeMilvusClient:
        """Expose successful database selection and record cleanup."""

        closed = False

        def list_databases(self) -> list[str]:
            """Return the configured database as already present."""
            return ["default"]

        def use_database(self, database_name: str) -> None:
            """Accept selection of the configured database."""
            assert database_name == "default"

        def close(self) -> None:
            """Record client cleanup."""
            self.closed = True

    client = _FakeMilvusClient()

    def fail_collection_setup(_database: LegalVectorDB) -> None:
        raise setup_error

    monkeypatch.setattr(database_module, "MilvusClient", lambda **_options: client)
    monkeypatch.setattr(LegalVectorDB, "_ensure_collection", fail_collection_setup)

    with pytest.raises(RuntimeError, match="collection setup failed") as captured:
        LegalVectorDB(_config(tmp_path), _FakeEmbedder())

    assert captured.value is setup_error
    assert client.closed


def test_constructor_preserves_setup_error_when_client_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A cleanup failure is attached to the original constructor failure."""
    setup_error = KeyboardInterrupt("collection setup interrupted")

    class _FailingCloseMilvusClient:
        """Expose successful database selection and fail client cleanup."""

        closed = False

        def list_databases(self) -> list[str]:
            """Return the configured database as already present."""
            return ["default"]

        def use_database(self, database_name: str) -> None:
            """Accept selection of the configured database."""
            assert database_name == "default"

        def close(self) -> None:
            """Record cleanup before raising a secondary error."""
            self.closed = True
            msg = "client cleanup failed"
            raise RuntimeError(msg)

    client = _FailingCloseMilvusClient()

    def fail_collection_setup(_database: LegalVectorDB) -> None:
        raise setup_error

    monkeypatch.setattr(database_module, "MilvusClient", lambda **_options: client)
    monkeypatch.setattr(LegalVectorDB, "_ensure_collection", fail_collection_setup)

    with pytest.raises(KeyboardInterrupt, match="collection setup interrupted") as captured:
        LegalVectorDB(_config(tmp_path), _FakeEmbedder())

    assert captured.value is setup_error
    assert client.closed
    assert setup_error.__notes__ == [
        "Failed to close Milvus client after Corrective RAG setup failed: RuntimeError('client cleanup failed')"
    ]


def test_index_processed_data_resumes_missing_chunks_without_reembedding_existing(tmp_path: Path) -> None:
    """A compatible partial CRAG collection embeds only missing chunks."""
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    write_chunks(chunks_dir / "a.chunks.jsonl", [_chunk("chunk-1", "One.")])
    config = _config(tmp_path)
    embedder = _FakeEmbedder()
    db = LegalVectorDB(config, embedder)
    try:
        assert db.index_processed_data(str(chunks_dir)) == 1
        embedder.document_calls.clear()
        write_chunks(
            chunks_dir / "a.chunks.jsonl",
            [_chunk("chunk-1", "One."), _chunk("chunk-2", "Two.", content_sha256="content-hash-2")],
        )

        assert db.index_processed_data(str(chunks_dir)) == 2

        assert embedder.document_calls == [["Two."]]
    finally:
        db.close()


@pytest.mark.parametrize(
    ("content_sha256", "chunker_config_sha256"),
    [
        ("changed-content", "config-hash"),
        ("content-hash", "changed-config"),
    ],
)
def test_index_processed_data_rejects_stale_chunk_metadata(
    tmp_path: Path,
    content_sha256: str,
    chunker_config_sha256: str,
) -> None:
    """A reused chunk ID cannot hide changed content or chunker configuration."""
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    chunks_path = chunks_dir / "a.chunks.jsonl"
    write_chunks(chunks_path, [_chunk("chunk-1", "Same text.")])
    db = LegalVectorDB(_config(tmp_path), _FakeEmbedder())
    try:
        db.index_processed_data(str(chunks_dir))
        changed_chunk = _chunk(
            "chunk-1",
            "Changed.",
            content_sha256=content_sha256,
            chunker_config_sha256=chunker_config_sha256,
        )
        write_chunks(chunks_path, [changed_chunk])

        with pytest.raises(ValueError, match="stale metadata"):
            db.index_processed_data(str(chunks_dir))
    finally:
        db.close()


def test_index_processed_data_rejects_rows_outside_current_corpus(tmp_path: Path) -> None:
    """A collection containing chunks outside the current corpus must be rebuilt."""
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    chunks_path = chunks_dir / "a.chunks.jsonl"
    write_chunks(chunks_path, [_chunk("chunk-1", "One."), _chunk("chunk-2", "Two.", content_sha256="content-hash-2")])
    db = LegalVectorDB(_config(tmp_path), _FakeEmbedder())
    try:
        db.index_processed_data(str(chunks_dir))
        write_chunks(chunks_path, [_chunk("chunk-1", "One.")])

        with pytest.raises(ValueError, match="outside the current corpus"):
            db.index_processed_data(str(chunks_dir))
    finally:
        db.close()


def test_index_processed_data_rejects_changed_embedding_model_revision(tmp_path: Path) -> None:
    """Rows embedded by another model revision cannot be mixed into the collection."""
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    write_chunks(chunks_dir / "a.chunks.jsonl", [_chunk("chunk-1", "Same text.")])
    config = _config(tmp_path)
    db = LegalVectorDB(config, _FakeEmbedder())
    try:
        db.index_processed_data(str(chunks_dir))
    finally:
        db.close()

    changed_db = LegalVectorDB(config, _FakeEmbedder(model_revision="def456"))
    try:
        with pytest.raises(ValueError, match="embedding_model_revision"):
            changed_db.index_processed_data(str(chunks_dir))
    finally:
        changed_db.close()


def test_collection_validation_rejects_incompatible_vector_dimension(tmp_path: Path) -> None:
    """An existing CRAG collection must match the configured embedding dimension."""
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    write_chunks(chunks_dir / "a.chunks.jsonl", [_chunk("chunk-1", "Same text.")])
    config = _config(tmp_path)
    db = LegalVectorDB(config, _FakeEmbedder())
    try:
        db.index_processed_data(str(chunks_dir))
    finally:
        db.close()

    with pytest.raises(ValueError, match="vector dimension"):
        LegalVectorDB(config, _FakeEmbedder(dimensions=3))


def test_collection_validation_rejects_incompatible_metric(tmp_path: Path) -> None:
    """An existing CRAG collection must use the configured retrieval metric."""
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    write_chunks(chunks_dir / "a.chunks.jsonl", [_chunk("chunk-1", "Same text.")])
    config = _config(tmp_path)
    db = LegalVectorDB(config, _FakeEmbedder())
    try:
        db.index_processed_data(str(chunks_dir))
    finally:
        db.close()

    with pytest.raises(ValueError, match="uses metric"):
        LegalVectorDB(_config(tmp_path, metric_type="L2"), _FakeEmbedder())


def test_preflight_corpus_fingerprints_the_verified_active_collection(tmp_path: Path) -> None:
    """CRAG exposes a manifest fingerprint only after exact active-row validation."""
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    chunks = (_chunk("chunk-2", "Two.", content_sha256="second-hash"), _chunk("chunk-1", "One."))
    write_chunks(chunks_dir / "corpus.chunks.jsonl", chunks)
    db = LegalVectorDB(_config(tmp_path), _FakeEmbedder())
    try:
        db.index_chunks(chunks)

        fingerprint = db.preflight_corpus(chunks_dir)

        assert fingerprint.collection_name == "legal_chunks"
        assert fingerprint.row_count == 2
        assert fingerprint.embedding_profile_id == "test-profile"
        assert fingerprint.embedding_model_id == "test/model"
        assert fingerprint.embedding_model_revision == "abc123"
        assert len(fingerprint.corpus_sha256) == 64
    finally:
        db.close()


def test_preflight_corpus_rejects_an_empty_local_corpus(tmp_path: Path) -> None:
    """An empty directory cannot validate an unrelated or empty active collection."""
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    db = LegalVectorDB(_config(tmp_path), _FakeEmbedder())
    try:
        with pytest.raises(ValueError, match=r"No \*\.chunks\.jsonl files found"):
            db.preflight_corpus(chunks_dir)
    finally:
        db.close()


@pytest.mark.parametrize(
    ("field_name", "changed_value"),
    [
        ("embedding_text", "changed embedding text"),
        ("source_name", "different.docx"),
    ],
)
def test_preflight_corpus_rejects_retrieval_fields_not_covered_by_metadata(
    tmp_path: Path,
    field_name: str,
    changed_value: str,
) -> None:
    """Exact CRAG rows catch stale embedding input and provenance with unchanged declared hashes."""
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    changed_chunk = replace(_chunk("chunk-1", "Original text."), **{field_name: changed_value})
    write_chunks(chunks_dir / "corpus.chunks.jsonl", (changed_chunk,))
    db = LegalVectorDB(_config(tmp_path), _FakeEmbedder())
    try:
        db.index_chunks((_chunk("chunk-1", "Original text."),))

        with pytest.raises(ValueError, match="does not match the configured corpus"):
            db.preflight_corpus(chunks_dir)
    finally:
        db.close()


def _config(tmp_path: Path, *, metric_type: str = "COSINE") -> SimpleNamespace:
    """Build CRAG vector DB settings for one isolated Milvus Lite database."""
    return SimpleNamespace(
        milvus_uri=str(tmp_path / "corrective_rag.db"),
        milvus_token=None,
        milvus_database="default",
        milvus_collection_name="legal_chunks",
        milvus_index_type="IVF_FLAT",
        milvus_metric_type=metric_type,
        milvus_nlist=1,
        milvus_nprobe=1,
        milvus_flush_after_insert=False,
        index_batch_size=2,
    )


def _chunk(
    chunk_id: str,
    text: str,
    *,
    content_sha256: str = "content-hash",
    chunker_config_sha256: str = "config-hash",
) -> Chunk:
    """Build one final chunk for CRAG indexing tests."""
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
