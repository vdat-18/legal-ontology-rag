"""Tests for shared application settings compatibility."""

from __future__ import annotations

from pathlib import Path

import pytest

from legal_rag.settings import Settings, SynthesisEmbeddingSettings, SynthesisVnCoreNlpSettings


def test_shared_settings_load_from_openai_compatible_environment() -> None:
    """An OpenAI-compatible endpoint needs only URL, key, model, and timeout settings."""
    settings = Settings.from_environment(
        {
            "LLM_BASE_URL": "http://localhost:20128/v1",
            "LLM_API_KEY": "test-key",
            "LLM_MODEL": "provider/test-model",
            "LLM_TIMEOUT_SECONDS": "30",
        }
    )

    assert settings.llm_base_url == "http://localhost:20128/v1"
    assert settings.llm_api_key == "test-key"
    assert settings.llm_model == "provider/test-model"
    assert settings.llm_timeout_seconds == 30


def test_synthesis_embedding_settings_are_isolated_from_shared_llm_environment() -> None:
    """Synthesis embeddings use dedicated provider variables without changing shared LLM settings."""
    settings = SynthesisEmbeddingSettings.from_environment(
        {
            "SYNTHESIS_EMBEDDING_BASE_URL": "https://embedding.test/v1/",
            "SYNTHESIS_EMBEDDING_API_KEY": "embedding-key",
            "SYNTHESIS_EMBEDDING_MODEL": "provider/bge-m3",
            "SYNTHESIS_EMBEDDING_TIMEOUT_SECONDS": "120",
        }
    )

    assert settings.base_url == "https://embedding.test/v1"
    assert settings.api_key == "embedding-key"
    assert settings.model == "provider/bge-m3"
    assert settings.timeout_seconds == 120


def test_synthesis_embedding_settings_reject_non_http_provider_urls() -> None:
    """Embedding requests cannot be redirected to local files or custom URL schemes."""
    with pytest.raises(ValueError, match="must use HTTP or HTTPS"):
        SynthesisEmbeddingSettings(
            base_url="file:///tmp/embeddings",
            api_key="embedding-key",
            model="provider/bge-m3",
            timeout_seconds=120,
        )


def test_synthesis_vncorenlp_settings_load_local_runtime_paths() -> None:
    """Synthesis word segmentation uses an explicit model directory and optional Java executable."""
    settings = SynthesisVnCoreNlpSettings.from_environment(
        {
            "SYNTHESIS_VNCORENLP_DIR": "C:/vncorenlp-1.2",
            "SYNTHESIS_VNCORENLP_JAVA_PATH": "C:/java/bin/java.exe",
        }
    )

    assert settings.model_dir == Path("C:/vncorenlp-1.2")
    assert settings.java_path == "C:/java/bin/java.exe"
