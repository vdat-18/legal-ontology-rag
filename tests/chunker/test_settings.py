"""Tests for chunker environment settings."""

from __future__ import annotations

from legal_rag.settings import _CHUNKER_TUNABLES, ChunkerSettings


def test_chunker_settings_load_without_llm_environment() -> None:
    """Chunking configuration remains independent from image-caption credentials, defaulted otherwise."""
    settings = ChunkerSettings.from_environment({"EMBEDDING_PROFILE": "qwen3_0_6b"})

    assert settings.embedding_profile == "qwen3_0_6b"
    assert settings.max_tokens == 512
    assert settings.split_overlap_tokens == 64


def test_every_declared_tunable_is_overridable_from_the_environment() -> None:
    """No declared tunable can be silently ignored: each maps to a CHUNK_<FIELD_NAME> variable."""
    base = {"EMBEDDING_PROFILE": "qwen3_0_6b"}
    defaults = ChunkerSettings.from_environment(base)

    for name, converter in _CHUNKER_TUNABLES.items():
        default = getattr(defaults, name)
        overridden = converter(str(default + 1))
        settings = ChunkerSettings.from_environment({**base, f"CHUNK_{name.upper()}": str(overridden)})

        assert getattr(settings, name) == overridden, name
