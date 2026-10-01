"""Tests for Corrective RAG configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from legal_rag.corrective_rag.generate.config import CorrectiveRetrievalConfig, RAGConfig


def test_retrieval_config_does_not_require_llm_provider_settings() -> None:
    """Retrieval-only evaluation loads embedding and Milvus controls without LLM credentials."""
    config = CorrectiveRetrievalConfig.from_env(
        {
            "EMBEDDING_PROFILE": "bge_m3",
            "MILVUS_URI": "http://milvus.test:19530",
            "MILVUS_DATABASE": "legal_rag",
            "CRAG_MILVUS_COLLECTION_NAME": "legal_chunks_test",
        }
    )

    assert config.embedding_profile == "bge_m3"
    assert config.milvus_uri == "http://milvus.test:19530"
    assert config.milvus_database == "legal_rag"
    assert config.milvus_collection_name == "legal_chunks_test"


def test_rag_config_requires_llm_provider_settings() -> None:
    """CRAG config must not use hardcoded LLM provider defaults."""
    with pytest.raises(KeyError, match="LLM_BASE_URL"):
        RAGConfig.from_env({})


def test_rag_config_preserves_configured_llm_provider() -> None:
    """Configured OpenAI-compatible provider values pass through unchanged."""
    config = RAGConfig.from_env(
        {
            "LLM_BASE_URL": "https://provider.test/v1/",
            "LLM_API_KEY": "provider-key",
            "LLM_MODEL": "provider/model",
            "LLM_TIMEOUT_SECONDS": "12",
            "LLM_EXTRA_REQUEST_BODY": '{"thinking":{"type":"disabled"}}',
            "EMBEDDING_PROFILE": "bge_m3",
            "CRAG_ANSWER_MAX_OUTPUT_TOKENS": "2048",
            "CRAG_QUERY_REFORMULATION_MAX_OUTPUT_TOKENS": "256",
            "CRAG_QUERY_REWRITE_MAX_OUTPUT_TOKENS": "512",
        }
    )

    assert config.llm_base_url == "https://provider.test/v1"
    assert config.llm_api_key == "provider-key"
    assert config.llm_model == "provider/model"
    assert config.llm_timeout_seconds == 12.0
    assert config.llm_extra_request_body == {"thinking": {"type": "disabled"}}
    assert config.answer_max_output_tokens == 2048
    assert config.query_reformulation_max_output_tokens == 256
    assert config.query_rewrite_max_output_tokens == 512


def test_rag_config_loading_does_not_create_a_data_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reading configuration has no filesystem side effects in the caller's working directory."""
    monkeypatch.chdir(tmp_path)

    RAGConfig.from_env(
        {
            "LLM_BASE_URL": "https://provider.test/v1",
            "LLM_API_KEY": "provider-key",
            "LLM_MODEL": "provider/model",
            "LLM_TIMEOUT_SECONDS": "12",
            "EMBEDDING_PROFILE": "bge_m3",
        }
    )

    assert not (tmp_path / "data").exists()


def test_rag_config_invalid_numeric_value_raises() -> None:
    """Invalid numeric CRAG configuration must not silently fall back to defaults."""
    environment = {
        "LLM_BASE_URL": "https://provider.test/v1",
        "LLM_API_KEY": "provider-key",
        "LLM_MODEL": "provider/model",
        "LLM_TIMEOUT_SECONDS": "12",
        "EMBEDDING_PROFILE": "bge_m3",
        "CRAG_EVALUATOR_UPPER_THRESHOLD": "not-a-float",
    }

    with pytest.raises(ValueError, match="not-a-float"):
        RAGConfig.from_env(environment)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("CRAG_ANSWER_MAX_OUTPUT_TOKENS", "0", "answer_max_output_tokens must be positive"),
        ("CRAG_QUERY_REFORMULATION_MAX_OUTPUT_TOKENS", "0", "query_reformulation_max_output_tokens must be positive"),
        ("CRAG_QUERY_REWRITE_MAX_OUTPUT_TOKENS", "0", "query_rewrite_max_output_tokens must be positive"),
        ("CRAG_MAX_RESPONSE_BYTES", "0", "max_response_bytes must be positive"),
        ("CRAG_ANSWER_TEMPERATURE", "3", "answer_temperature must be between"),
    ],
)
def test_rag_config_rejects_invalid_llm_bounds(name: str, value: str, message: str) -> None:
    """CRAG provider calls must have positive response bounds and valid temperatures."""
    environment = {
        "LLM_BASE_URL": "https://provider.test/v1",
        "LLM_API_KEY": "provider-key",
        "LLM_MODEL": "provider/model",
        "LLM_TIMEOUT_SECONDS": "12",
        "EMBEDDING_PROFILE": "bge_m3",
        name: value,
    }

    with pytest.raises(ValueError, match=message):
        RAGConfig.from_env(environment)
