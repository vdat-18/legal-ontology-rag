"""Tests for naive RAG baseline environment settings."""

from __future__ import annotations

from legal_rag.settings import NaiveRagSettings


def test_naive_rag_settings_load_from_environment() -> None:
    """Naive RAG settings parse from process environment variables, with defaults for the rest."""
    settings = NaiveRagSettings.from_environment(
        {
            "EMBEDDING_PROFILE": "bge_m3",
            "NAIVE_RAG_TOP_K": "5",
            "NAIVE_RAG_MILVUS_URI": "./naive_rag_test.db",
        }
    )

    assert settings.embedding_profile == "bge_m3"
    assert settings.top_k == 5
    assert settings.milvus_uri == "./naive_rag_test.db"
    assert settings.milvus_collection_name == "naive_rag_chunks"
    assert settings.milvus_metric_type == "COSINE"
