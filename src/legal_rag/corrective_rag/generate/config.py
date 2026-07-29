"""Configuration loader for the Corrective RAG pipeline.

Loads LLM, embedding, and vector database settings from environment variables.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from legal_rag.settings import ChunkerSettings, Settings

_LLM_ENV_DEFAULTS = {
    "LLM_BASE_URL": "https://api.openai.com/v1",
    "LLM_API_KEY": "dummy-key",
    "LLM_MODEL": "gpt-3.5-turbo",
    "LLM_TIMEOUT_SECONDS": "60.0",
}
_CHUNKER_ENV_DEFAULTS = {
    "EMBEDDING_PROFILE": "vietnamese_embedding",
    "CHUNK_MAX_TOKENS": "512",
    "CHUNK_SPLIT_OVERLAP_TOKENS": "64",
}
_DEFAULT_MILVUS_URI = "http://localhost:19530"


@dataclass(frozen=True, slots=True)
class RAGConfig:
    """Configuration settings for the Corrective RAG system."""

    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_timeout_seconds: float
    embedding_profile: str
    milvus_uri: str
    index_batch_size: int
    evaluator_upper_threshold: float
    evaluator_lower_threshold: float
    refinement_max_strips_per_chunk: int
    refinement_strip_max_chars: int
    refinement_top_internal_strips: int
    refinement_top_external_strips: int
    refinement_decompose_mode: str
    web_search_enabled: bool
    web_search_url: str
    web_search_max_results: int
    web_search_timeout_seconds: float
    web_search_allowed_domains: tuple[str, ...]

    @classmethod
    def from_env(cls, environment: Mapping[str, str] | None = None) -> RAGConfig:
        """Load configuration from environment variables.

        Raises:
            ValueError: If required configuration environment variables are missing.
        """
        if environment is None:
            load_dotenv()
        values = os.environ if environment is None else environment
        llm_settings = Settings.from_environment(_with_defaults(values, _LLM_ENV_DEFAULTS))
        chunker_settings = ChunkerSettings.from_environment(_with_defaults(values, _CHUNKER_ENV_DEFAULTS))

        # Ensure data folder exists
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)

        return cls(
            llm_base_url=llm_settings.llm_base_url,
            llm_api_key=llm_settings.llm_api_key,
            llm_model=llm_settings.llm_model,
            llm_timeout_seconds=llm_settings.llm_timeout_seconds,
            embedding_profile=chunker_settings.embedding_profile,
            milvus_uri=values.get("MILVUS_URI", _DEFAULT_MILVUS_URI),
            index_batch_size=_read_int(values, "MILVUS_INDEX_BATCH_SIZE", 32),
            evaluator_upper_threshold=_read_float(values, "CRAG_EVALUATOR_UPPER_THRESHOLD", 0.70),
            evaluator_lower_threshold=_read_float(values, "CRAG_EVALUATOR_LOWER_THRESHOLD", 0.35),
            refinement_max_strips_per_chunk=_read_int(values, "CRAG_REFINEMENT_MAX_STRIPS_PER_CHUNK", 8),
            refinement_strip_max_chars=_read_int(values, "CRAG_REFINEMENT_STRIP_MAX_CHARS", 900),
            refinement_top_internal_strips=_read_int(values, "CRAG_REFINEMENT_TOP_INTERNAL_STRIPS", 6),
            refinement_top_external_strips=_read_int(values, "CRAG_REFINEMENT_TOP_EXTERNAL_STRIPS", 5),
            refinement_decompose_mode=values.get("CRAG_REFINEMENT_DECOMPOSE_MODE", "excerption").strip(),
            web_search_enabled=_read_bool(values, "CRAG_WEB_SEARCH_ENABLED", False),
            web_search_url=values.get("CRAG_WEB_SEARCH_URL", "https://duckduckgo.com/html/").strip(),
            web_search_max_results=_read_int(values, "CRAG_WEB_SEARCH_MAX_RESULTS", 3),
            web_search_timeout_seconds=_read_float(values, "CRAG_WEB_SEARCH_TIMEOUT_SECONDS", 10.0),
            web_search_allowed_domains=_read_csv(values, "CRAG_WEB_SEARCH_ALLOWED_DOMAINS"),
        )


def _with_defaults(values: Mapping[str, str], defaults: Mapping[str, str]) -> Mapping[str, str]:
    """Return environment values with defaults for optional CRAG commands."""
    merged = dict(defaults)
    merged.update(values)
    return merged


def _read_float(values: Mapping[str, str], name: str, default: float) -> float:
    """Read a float environment variable."""
    try:
        return float(values.get(name, str(default)))
    except ValueError:
        return default


def _read_int(values: Mapping[str, str], name: str, default: int) -> int:
    """Read an integer environment variable."""
    try:
        return int(values.get(name, str(default)))
    except ValueError:
        return default


def _read_bool(values: Mapping[str, str], name: str, default: bool) -> bool:
    """Read a boolean environment variable."""
    value = values.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _read_csv(values: Mapping[str, str], name: str) -> tuple[str, ...]:
    """Read a comma-separated environment variable."""
    value = values.get(name, "")
    return tuple(item.strip().lower() for item in value.split(",") if item.strip())
