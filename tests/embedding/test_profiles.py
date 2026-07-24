"""Tests for versioned embedding model profiles."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from legal_rag.embedding.profiles import load_configured_embedding_profile, load_embedding_profile


def test_load_embedding_profile_reads_versioned_model_configuration() -> None:
    """The selected profile exposes pinned model and tokenizer metadata."""
    profile = load_configured_embedding_profile("qwen3_0_6b")

    assert profile.model_id == "Qwen/Qwen3-Embedding-0.6B"
    assert profile.tokenizer_id == profile.model_id
    assert profile.model_revision == profile.tokenizer_revision
    assert profile.dimensions == 1024
    assert profile.max_input_tokens == 32768
    assert profile.query_instruction is not None


def test_load_embedding_profile_raises_for_unknown_profile() -> None:
    """An unknown profile preserves the native mapping error."""
    with pytest.raises(KeyError):
        load_configured_embedding_profile("missing")


def test_load_embedding_profile_rejects_unsupported_schema(tmp_path: Path) -> None:
    """A profile file with an unknown schema fails at the input boundary."""
    config_path = tmp_path / "profiles.json"
    config_path.write_text(json.dumps({"schema_version": "unknown", "profiles": {}}), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported embedding profile schema"):
        load_embedding_profile(config_path, "qwen3_0_6b")
