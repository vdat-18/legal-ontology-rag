"""Tests for keyword, semantic, and hybrid search modes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from legal_rag.naive_rag.retriever import RetrievedChunk, Retriever, SearchMode
from legal_rag.naive_rag.structure import StructuralUnit


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
        "reranker_model_id": "test/model",
        "reranker_model_revision": "test-revision",
        "reranker_batch_size": 8,
        "reranker_pool_multiplier": 5,
        "chunks_dir": "unused",
        "hierarchical_max_chunks_per_unit": 5,
        "hierarchical_max_tokens_per_unit": 2500,
        "bm25_k1": 1.5,
        "bm25_b": 0.75,
        "bm25_delta": 0.5,
        "keyword_retrieval_pool_size": 100,
        "keyword_max_candidates_per_fact": 10,
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


def test_hybrid_search_preserves_rrf_order_scores_and_semantic_representative():
    """Hybrid fusion keeps the established scores, tie order, and semantic payload."""
    retriever = _make_retriever()
    semantic = [
        RetrievedChunk("shared", "semantic shared", "semantic.docx", "p", 0.9),
        RetrievedChunk("semantic-only", "semantic only", "semantic.docx", "p", 0.8),
        RetrievedChunk("semantic-tie", "semantic tie", "semantic.docx", "p", 0.7),
    ]
    keyword = [
        RetrievedChunk("shared", "keyword shared", "keyword.docx", "p", 3.0),
        RetrievedChunk("keyword-only", "keyword only", "keyword.docx", "p", 2.0),
        RetrievedChunk("keyword-tie", "keyword tie", "keyword.docx", "p", 1.0),
    ]
    retriever._semantic_search = MagicMock(return_value=semantic)
    retriever._keyword_search = MagicMock(return_value=keyword)

    hits = retriever.retrieve("test query", top_k=5, mode=SearchMode.HYBRID)

    assert [hit.chunk_id for hit in hits] == [
        "shared",
        "semantic-only",
        "keyword-only",
        "semantic-tie",
        "keyword-tie",
    ]
    assert [hit.score for hit in hits] == pytest.approx([2 / 61, 1 / 62, 1 / 62, 1 / 63, 1 / 63])
    assert hits[0].text == "semantic shared"
    assert hits[0].source_name == "semantic.docx"


def test_hybrid_rerank_fuses_candidates_before_reranking():
    """The reranker receives one deduplicated RRF list and controls final ordering."""

    class FakeReranker:
        def __init__(self) -> None:
            self.candidates: list[RetrievedChunk] = []

        def rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
            assert query == "test query"
            self.candidates = list(candidates)
            return list(reversed(candidates))

    reranker = FakeReranker()
    retriever = _make_retriever()
    retriever._reranker = reranker
    semantic = [
        RetrievedChunk("shared", "semantic", "doc.docx", "p", 0.9),
        RetrievedChunk("semantic", "semantic", "doc.docx", "p", 0.8),
    ]
    keyword = [
        RetrievedChunk("shared", "keyword", "doc.docx", "p", 3.0),
        RetrievedChunk("keyword", "keyword", "doc.docx", "p", 2.0),
    ]
    retriever._semantic_search = MagicMock(return_value=semantic)
    retriever._keyword_search = MagicMock(return_value=keyword)

    hits = retriever.retrieve("test query", top_k=2, mode=SearchMode.HYBRID_RERANK)

    assert len({chunk.chunk_id for chunk in reranker.candidates}) == 3
    assert [chunk.chunk_id for chunk in hits] == ["keyword", "semantic"]
    retriever._semantic_search.assert_called_once_with("test query", 10)
    retriever._keyword_search.assert_called_once_with("test query", 10)


def test_baseline_modes_do_not_load_the_reranker():
    """Existing keyword, semantic, and hybrid modes remain independent of the model."""
    retriever = _make_retriever()
    retriever._get_reranker = MagicMock(side_effect=AssertionError("reranker loaded"))

    retriever.retrieve("test", top_k=2, mode=SearchMode.SEMANTIC)
    retriever.retrieve("keyword", top_k=2, mode=SearchMode.KEYWORD)
    retriever.retrieve("test", top_k=2, mode=SearchMode.HYBRID)

    retriever._get_reranker.assert_not_called()


def test_hierarchical_search_expands_and_deduplicates_article_units():
    """Hierarchical mode expands RRF seeds and keeps one result per article unit."""

    class FakeCatalog:
        def build_structural_unit(self, seed, *, max_chunks, max_tokens):
            assert max_chunks == 5
            assert max_tokens == 2500
            if seed.chunk_id in {"seed-1", "seed-2"}:
                return StructuralUnit("article:1", "expanded article", "doc.docx", ("seed-1", "evidence-2"))
            return None

    retriever = _make_retriever()
    retriever._chunk_catalog = FakeCatalog()
    seeds = [
        RetrievedChunk("seed-1", "one", "doc.docx", "p", 0.3),
        RetrievedChunk("seed-2", "two", "doc.docx", "p", 0.2),
        RetrievedChunk("leaf", "leaf", "doc.docx", "p", 0.1),
    ]
    retriever._hybrid_candidates = MagicMock(return_value=seeds)

    hits = retriever.retrieve("query", top_k=2, mode=SearchMode.HIERARCHICAL)

    assert [hit.effective_unit_id for hit in hits] == ["article:1", "leaf"]
    assert hits[0].text == "expanded article"
    assert hits[0].effective_evidence_chunk_ids == ("seed-1", "evidence-2")
    retriever._hybrid_candidates.assert_called_once_with("query", 6, 6)


def test_baseline_modes_do_not_load_the_chunk_catalog():
    """Existing modes remain independent of local structural chunk files."""
    retriever = _make_retriever()
    retriever._get_chunk_catalog = MagicMock(side_effect=AssertionError("catalog loaded"))
    retriever._reranker = MagicMock(rerank=lambda _query, candidates: candidates)

    retriever.retrieve("test", top_k=2, mode=SearchMode.SEMANTIC)
    retriever.retrieve("keyword", top_k=2, mode=SearchMode.KEYWORD)
    retriever.retrieve("test", top_k=2, mode=SearchMode.HYBRID)
    retriever.retrieve("test", top_k=2, mode=SearchMode.HYBRID_RERANK)

    retriever._get_chunk_catalog.assert_not_called()


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
