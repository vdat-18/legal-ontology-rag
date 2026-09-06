"""Tests for keyword, semantic, and hybrid search modes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from legal_rag.naive_rag.retriever import RetrievedChunk, Retriever, SearchMode


def _fake_settings(**overrides):
    """Build a minimal NaiveRagSettings-like object."""
    defaults = {
        "milvus_collection_name": "test",
        "milvus_metric_type": "COSINE",
        "milvus_nprobe": 16,
        "milvus_insert_batch_size": 200,
        "search_mode": "semantic",
        "hybrid_rrf_k": 60,
        "hybrid_pool_multiplier": 3,
    }
    defaults.update(overrides)
    return type("S", (), defaults)()


def _make_retriever():
    """Build a Retriever with mocked encoder and Milvus client."""
    encoder = MagicMock()
    encoder.encode_query.return_value = MagicMock(__getitem__=lambda s, i: MagicMock(tolist=lambda: [0.1] * 10))
    client = MagicMock()
    client.search.return_value = [
        [
            {
                "entity": {"chunk_id": f"s-{i}", "text": f"text {i}", "source_name": "doc.docx", "kind": "p"},
                "distance": 0.9 - i * 0.1,
            }
            for i in range(3)
        ]
    ]
    chunk_rows = [
        {"chunk_id": f"k-{i}", "text": f"keyword text {i}", "source_name": "doc.docx", "kind": "p"} for i in range(5)
    ]
    iterator = MagicMock()
    iterator.next.side_effect = [chunk_rows, []]
    client.query_iterator.return_value = iterator
    return Retriever(encoder, client, _fake_settings())


def test_semantic_search_returns_chunks():
    """Semantic mode returns chunks from Milvus vector search."""
    retriever = _make_retriever()
    hits = retriever.retrieve("test query", top_k=3, mode=SearchMode.SEMANTIC)
    assert len(hits) == 3
    assert all(isinstance(h, RetrievedChunk) for h in hits)
    assert hits[0].chunk_id == "s-0"


def test_keyword_search_returns_chunks():
    """Keyword mode returns BM25-ranked chunks."""
    retriever = _make_retriever()
    hits = retriever.retrieve("keyword text", top_k=3, mode=SearchMode.KEYWORD)
    assert len(hits) > 0
    assert all(isinstance(h, RetrievedChunk) for h in hits)


def test_hybrid_search_returns_chunks():
    """Hybrid mode fuses semantic and keyword results via RRF."""
    retriever = _make_retriever()
    hits = retriever.retrieve("test query", top_k=3, mode=SearchMode.HYBRID)
    assert len(hits) > 0
    assert all(isinstance(h, RetrievedChunk) for h in hits)


def test_default_mode_is_semantic():
    """Without explicit mode, retriever uses settings.search_mode."""
    retriever = _make_retriever()
    hits = retriever.retrieve("test", top_k=3)
    assert len(hits) == 3
    retriever.client.search.assert_called_once()


def test_keyword_mode_does_not_call_milvus_search():
    """Keyword mode uses BM25 only, not Milvus vector search."""
    retriever = _make_retriever()
    retriever.retrieve("test", top_k=3, mode=SearchMode.KEYWORD)
    retriever.client.search.assert_not_called()


def test_invalid_mode_raises():
    """Invalid search mode string raises ValueError."""
    with pytest.raises(ValueError, match="not a valid SearchMode"):
        SearchMode("invalid")


def test_pipeline_passes_search_mode():
    """Pipeline.run passes search_mode to retriever."""
    from legal_rag.naive_rag.retriever import SearchMode

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []
    mock_gen_client = MagicMock()
    mock_gen_client.generate.return_value = "answer"

    from legal_rag.naive_rag.pipeline import NaiveRAGPipeline

    pipeline = NaiveRAGPipeline.__new__(NaiveRAGPipeline)
    pipeline.settings = _fake_settings(top_k=5)
    pipeline._retriever = mock_retriever
    pipeline._generation_client = mock_gen_client

    pipeline.run("test", search_mode="keyword")
    mock_retriever.retrieve.assert_called_once_with("test", top_k=5, mode=SearchMode.KEYWORD)


def test_api_rejects_invalid_search_mode(monkeypatch: pytest.MonkeyPatch):
    """API rejects search_mode values outside the allowed set."""
    from fastapi.testclient import TestClient

    env = {
        "LLM_BASE_URL": "http://localhost/v1",
        "LLM_API_KEY": "k",
        "LLM_MODEL": "m",
        "LLM_TIMEOUT_SECONDS": "10",
        "EMBEDDING_PROFILE": "qwen3_0_6b",
        "MILVUS_URI": "http://localhost:19530",
        "MILVUS_DATABASE": "legal_rag",
        "REDIS_URL": "redis://localhost:6379/0",
        "RAG_API_KEY": "test-key",
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    mock_pipeline = MagicMock()
    mock_redis = MagicMock()
    mock_redis.lrange.return_value = []

    with (
        patch("legal_rag.naive_rag.pipeline.NaiveRAGPipeline", return_value=mock_pipeline),
        patch("legal_rag.naive_rag_api.redis") as mock_redis_mod,
    ):
        mock_redis_mod.from_url.return_value = mock_redis
        from legal_rag.naive_rag_api import app

        with TestClient(app) as tc:
            resp = tc.post(
                "/query", json={"query": "test", "search_mode": "invalid"}, headers={"X-Api-Key": "test-key"}
            )
            assert resp.status_code == 422
