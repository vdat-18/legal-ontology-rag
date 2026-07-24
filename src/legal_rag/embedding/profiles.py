"""Load versioned embedding model profiles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path

EMBEDDING_PROFILE_SCHEMA_VERSION = "1.0.0"
EMBEDDING_PROFILES_RESOURCE = "embedding_models.json"


@dataclass(frozen=True, slots=True)
class EmbeddingProfile:
    """Immutable model and tokenizer settings for one embedding profile."""

    profile_id: str
    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    dimensions: int
    max_input_tokens: int
    query_instruction: str | None


def load_configured_embedding_profile(profile_id: str) -> EmbeddingProfile:
    """Load one profile from the versioned configuration bundled with the package."""
    config_path = files("legal_rag.embedding").joinpath(EMBEDDING_PROFILES_RESOURCE)
    return load_embedding_profile(config_path, profile_id)


def load_embedding_profile(config_path: Path | Traversable, profile_id: str) -> EmbeddingProfile:
    """Load one embedding profile from a versioned JSON configuration file."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["schema_version"] != EMBEDDING_PROFILE_SCHEMA_VERSION:
        msg = f"Unsupported embedding profile schema: {config['schema_version']}"
        raise ValueError(msg)

    profile = config["profiles"][profile_id]
    return EmbeddingProfile(
        profile_id=profile_id,
        model_id=profile["model_id"],
        model_revision=profile["model_revision"],
        tokenizer_id=profile["tokenizer_id"],
        tokenizer_revision=profile["tokenizer_revision"],
        dimensions=profile["dimensions"],
        max_input_tokens=profile["max_input_tokens"],
        query_instruction=profile["query_instruction"],
    )
