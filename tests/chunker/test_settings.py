"""Tests for chunker environment settings."""

from __future__ import annotations

from legal_rag.settings import ChunkerSettings


def test_chunker_settings_load_without_llm_environment() -> None:
    """Chunking configuration remains independent from image-caption credentials."""
    settings = ChunkerSettings.from_environment(
        {
            "EMBEDDING_PROFILE": "qwen3_0_6b",
            "CHUNK_MAX_TOKENS": "512",
            "CHUNK_SPLIT_OVERLAP_TOKENS": "64",
        }
    )

    assert settings.embedding_profile == "qwen3_0_6b"
    assert settings.max_tokens == 512
    assert settings.split_overlap_tokens == 64
