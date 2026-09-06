"""Tests for naive RAG baseline environment settings."""

from __future__ import annotations

import pytest

from legal_rag.settings import _NAIVE_RAG_TUNABLES, NaiveRagSettings


def test_naive_rag_settings_load_from_environment() -> None:
    """Naive RAG settings parse from process environment variables, with defaults for the rest."""
    environment = {"EMBEDDING_PROFILE": "bge_m3"}
    settings = NaiveRagSettings.from_environment(environment)

    assert settings.embedding_profile == "bge_m3"
    assert settings.top_k == 10
    assert settings.milvus_collection_name == "naive_rag_chunks"
    assert settings.milvus_metric_type == "COSINE"


def test_every_declared_tunable_is_overridable_from_the_environment() -> None:
    """No declared tunable can be silently ignored: each maps to a NAIVE_RAG_<FIELD_NAME> variable."""
    base = {"EMBEDDING_PROFILE": "bge_m3"}
    raw_values = {
        "top_k": "11",
        "milvus_collection_name": "custom_collection",
        "milvus_index_type": "HNSW",
        "milvus_metric_type": "L2",
        "milvus_nlist": "256",
        "milvus_nprobe": "32",
        "milvus_insert_batch_size": "50",
        "milvus_flush_after_insert": "false",
        "answer_max_output_tokens": "2048",
        "answer_temperature": "0.5",
        "max_response_bytes": "2097152",
        "search_mode": "keyword",
        "hybrid_rrf_k": "30",
        "hybrid_pool_multiplier": "2",
        "bm25_k1": "1.2",
        "bm25_b": "0.8",
        "bm25_delta": "0.3",
        "keyword_retrieval_pool_size": "50",
        "keyword_max_candidates_per_fact": "5",
    }

    assert raw_values.keys() == _NAIVE_RAG_TUNABLES.keys()
    for name, raw_value in raw_values.items():
        settings = NaiveRagSettings.from_environment({**base, f"NAIVE_RAG_{name.upper()}": raw_value})

        assert getattr(settings, name) == _NAIVE_RAG_TUNABLES[name](raw_value), name


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("top_k", 0),
        ("milvus_nlist", 0),
        ("milvus_nprobe", -1),
        ("milvus_insert_batch_size", 0),
        ("answer_max_output_tokens", 0),
        ("max_response_bytes", 0),
        ("answer_temperature", 2.1),
    ],
)
def test_naive_rag_settings_reject_invalid_numeric_values(field_name: str, value: float | int) -> None:
    """Numeric values are validated before they reach Milvus or the LLM provider."""
    with pytest.raises(ValueError, match=field_name):
        NaiveRagSettings(embedding_profile="bge_m3", **{field_name: value})
