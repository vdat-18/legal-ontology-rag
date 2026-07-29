"""Tests for naive RAG baseline environment settings."""

from __future__ import annotations

from legal_rag.settings import _NAIVE_RAG_TUNABLES, NaiveRagSettings


def test_naive_rag_settings_load_from_environment() -> None:
    """Naive RAG settings parse from process environment variables, with defaults for the rest."""
    settings = NaiveRagSettings.from_environment(
        {
            "EMBEDDING_PROFILE": "bge_m3",
            "MILVUS_URI": "./naive_rag_test.db",
        }
    )

    assert settings.embedding_profile == "bge_m3"
    assert settings.milvus_uri == "./naive_rag_test.db"
    assert settings.top_k == 10
    assert settings.milvus_collection_name == "naive_rag_chunks"
    assert settings.milvus_metric_type == "COSINE"


def test_every_declared_tunable_is_overridable_from_the_environment() -> None:
    """No declared tunable can be silently ignored: each maps to a NAIVE_RAG_<FIELD_NAME> variable."""
    base = {"EMBEDDING_PROFILE": "bge_m3", "MILVUS_URI": "./naive_rag_test.db"}
    defaults = NaiveRagSettings.from_environment(base)

    for name, converter in _NAIVE_RAG_TUNABLES.items():
        default = getattr(defaults, name)
        overridden = converter(str(default + 1))
        settings = NaiveRagSettings.from_environment({**base, f"NAIVE_RAG_{name.upper()}": str(overridden)})

        assert getattr(settings, name) == overridden, name
