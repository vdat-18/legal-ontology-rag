"""Application settings loaded from environment variables."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from urllib.parse import urlsplit

_MAX_LLM_TEMPERATURE = 2.0


@dataclass(frozen=True, slots=True)
class Settings:
    """LLM connection and bounded parser execution settings.

    ``llm_extra_request_body`` holds JSON merged into every chat-completion request, for provider
    fields outside the OpenAI-compatible schema. It exists because DeepSeek's reasoning mode can
    spend a whole completion on ``reasoning_content`` and return an empty ``content``, which no
    caller can parse; ``{"thinking": {"type": "disabled"}}`` turns that off. It is configuration
    rather than a model-name check, so a field one provider understands is never sent to another.
    """

    llm_base_url: str
    llm_api_key: str = field(repr=False)
    llm_model: str
    llm_timeout_seconds: float
    llm_extra_request_body: Mapping[str, object] = field(default_factory=dict)
    image_caption_max_concurrency: int = 4
    image_caption_max_output_tokens: int = 2048
    image_caption_max_context_characters: int = 4000
    image_caption_temperature: float = 0.0
    image_caption_surrounding_context_units: int = 3
    image_caption_max_rate_limit_retries: int = 3
    image_caption_rate_limit_retry_delay_seconds: float = 60.0
    image_caption_max_response_bytes: int = 1_048_576
    attachment_archive_max_members: int = 100
    attachment_archive_max_member_bytes: int = 67_108_864
    attachment_archive_max_total_bytes: int = 268_435_456
    office_conversion_timeout_seconds: float = 120.0

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> Settings:
        """Load application settings from process environment variables."""
        values = os.environ if environment is None else environment
        return cls(
            llm_base_url=values["LLM_BASE_URL"].strip().rstrip("/"),
            llm_api_key=values["LLM_API_KEY"].strip(),
            llm_model=values["LLM_MODEL"].strip(),
            llm_timeout_seconds=float(values["LLM_TIMEOUT_SECONDS"]),
            llm_extra_request_body=json.loads(values.get("LLM_EXTRA_REQUEST_BODY", "") or "{}"),
        )


@dataclass(frozen=True, slots=True)
class SynthesisEmbeddingSettings:
    """OpenAI-compatible embedding settings used only by benchmark synthesis."""

    base_url: str
    api_key: str = field(repr=False)
    model: str
    timeout_seconds: float

    def __post_init__(self) -> None:
        """Reject unusable or non-HTTP synthesis embedding provider settings."""
        if urlsplit(self.base_url).scheme not in {"http", "https"}:
            msg = "Synthesis embedding base_url must use HTTP or HTTPS"
            raise ValueError(msg)
        if not self.api_key or not self.model:
            msg = "Synthesis embedding api_key and model must not be empty"
            raise ValueError(msg)
        if not isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            msg = f"Synthesis embedding timeout_seconds must be positive, got {self.timeout_seconds}"
            raise ValueError(msg)

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> SynthesisEmbeddingSettings:
        """Load synthesis-only embedding settings from process environment variables."""
        values = os.environ if environment is None else environment
        return cls(
            base_url=values["SYNTHESIS_EMBEDDING_BASE_URL"].strip().rstrip("/"),
            api_key=values["SYNTHESIS_EMBEDDING_API_KEY"].strip(),
            model=values["SYNTHESIS_EMBEDDING_MODEL"].strip(),
            timeout_seconds=float(values["SYNTHESIS_EMBEDDING_TIMEOUT_SECONDS"]),
        )


@dataclass(frozen=True, slots=True)
class SynthesisVnCoreNlpSettings:
    """Local VnCoreNLP runtime paths used only by benchmark synthesis."""

    model_dir: Path
    java_path: str = "java"

    def __post_init__(self) -> None:
        """Reject empty executable configuration before starting VnCoreNLP."""
        if not self.java_path:
            msg = "Synthesis VnCoreNLP java_path must not be empty"
            raise ValueError(msg)

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> SynthesisVnCoreNlpSettings:
        """Load synthesis-only VnCoreNLP paths from environment variables."""
        values = os.environ if environment is None else environment
        return cls(
            model_dir=Path(values["SYNTHESIS_VNCORENLP_DIR"]).expanduser(),
            java_path=values.get("SYNTHESIS_VNCORENLP_JAVA_PATH", "java").strip(),
        )


@dataclass(frozen=True, slots=True)
class MilvusSettings:
    """Shared Milvus connection and database settings."""

    uri: str
    database: str
    token: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Reject empty Milvus connection identifiers."""
        for name, value in {"uri": self.uri, "database": self.database}.items():
            if not value:
                msg = f"Milvus {name} must not be empty"
                raise ValueError(msg)

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> MilvusSettings:
        """Load shared Milvus settings from process environment variables."""
        values = os.environ if environment is None else environment
        return cls(
            uri=values["MILVUS_URI"].strip(),
            database=values["MILVUS_DATABASE"].strip(),
            token=values.get("MILVUS_TOKEN", "").strip() or None,
        )


@dataclass(frozen=True, slots=True)
class Neo4jSettings:
    """Shared Neo4j connection settings."""

    uri: str
    user: str
    password: str = field(repr=False)

    def __post_init__(self) -> None:
        """Reject empty Neo4j connection values."""
        for name, value in {"uri": self.uri, "user": self.user, "password": self.password}.items():
            if not value:
                msg = f"Neo4j {name} must not be empty"
                raise ValueError(msg)

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> Neo4jSettings:
        """Load shared Neo4j settings from process environment variables."""
        values = os.environ if environment is None else environment
        return cls(
            uri=values["NEO4J_URI"].strip(),
            user=values["NEO4J_USER"].strip(),
            password=values["NEO4J_PASSWORD"].strip(),
        )


_CHUNKER_TUNABLES: dict[str, Callable[[str], int]] = {
    "max_tokens": int,
    "split_overlap_tokens": int,
}


@dataclass(frozen=True, slots=True)
class ChunkerSettings:
    """Embedding profile selection and token budgets for chunking."""

    embedding_profile: str
    max_tokens: int = 512
    split_overlap_tokens: int = 64

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> ChunkerSettings:
        """Load chunker settings from process environment variables.

        Every tunable above is overridable as ``CHUNK_<FIELD_NAME>`` and keeps its declared
        default otherwise, so no declared value can be silently ignored by the loader.
        """
        values = os.environ if environment is None else environment
        overrides = {
            name: converter(values[f"CHUNK_{name.upper()}"])
            for name, converter in _CHUNKER_TUNABLES.items()
            if f"CHUNK_{name.upper()}" in values
        }
        return cls(**overrides, embedding_profile=values["EMBEDDING_PROFILE"].strip())


def _parse_boolean(value: str) -> bool:
    """Parse one JSON boolean from an environment variable."""
    parsed = json.loads(value.lower())
    if not isinstance(parsed, bool):
        msg = f"Expected a boolean, got {value}"
        raise TypeError(msg)
    return parsed


_NAIVE_RAG_TUNABLES: dict[str, Callable[[str], str | bool | float | int]] = {
    "top_k": int,
    "milvus_collection_name": str.strip,
    "milvus_index_type": str.strip,
    "milvus_metric_type": str.strip,
    "milvus_nlist": int,
    "milvus_nprobe": int,
    "milvus_insert_batch_size": int,
    "milvus_flush_after_insert": _parse_boolean,
    "answer_max_output_tokens": int,
    "answer_temperature": float,
    "max_response_bytes": int,
}


@dataclass(frozen=True, slots=True)
class NaiveRagSettings:
    """Retrieval, indexing, and generation settings for the naive RAG baseline."""

    embedding_profile: str
    top_k: int = 10
    milvus_collection_name: str = "naive_rag_chunks"
    milvus_index_type: str = "IVF_FLAT"
    milvus_metric_type: str = "COSINE"
    milvus_nlist: int = 128
    milvus_nprobe: int = 16
    milvus_insert_batch_size: int = 200
    milvus_flush_after_insert: bool = True
    # DeepSeek spends output tokens on reasoning before the answer, and one context is sent per
    # supporting chunk, so a grounded answer over ten multi-source facts runs close to 3k tokens.
    # Too small a budget truncates the response into an empty string that cannot be parsed.
    answer_max_output_tokens: int = 8192
    answer_temperature: float = 0.3
    max_response_bytes: int = 1_048_576

    def __post_init__(self) -> None:
        """Reject invalid external-service configuration before connecting to Milvus."""
        required_strings = {
            "embedding_profile": self.embedding_profile,
            "milvus_collection_name": self.milvus_collection_name,
            "milvus_index_type": self.milvus_index_type,
            "milvus_metric_type": self.milvus_metric_type,
        }
        for name, value in required_strings.items():
            if not value:
                msg = f"{name} must not be empty"
                raise ValueError(msg)

        positive_values = {
            "top_k": self.top_k,
            "milvus_nlist": self.milvus_nlist,
            "milvus_nprobe": self.milvus_nprobe,
            "milvus_insert_batch_size": self.milvus_insert_batch_size,
            "answer_max_output_tokens": self.answer_max_output_tokens,
            "max_response_bytes": self.max_response_bytes,
        }
        for name, value in positive_values.items():
            if value < 1:
                msg = f"{name} must be positive, got {value}"
                raise ValueError(msg)

        if not isinstance(self.milvus_flush_after_insert, bool):
            msg = "milvus_flush_after_insert must be a boolean"
            raise TypeError(msg)

        if not isfinite(self.answer_temperature) or not 0 <= self.answer_temperature <= _MAX_LLM_TEMPERATURE:
            msg = f"answer_temperature must be between 0 and {_MAX_LLM_TEMPERATURE}, got {self.answer_temperature}"
            raise ValueError(msg)

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> NaiveRagSettings:
        """Load naive RAG baseline settings from process environment variables.

        Every tunable above is overridable as ``NAIVE_RAG_<FIELD_NAME>`` and keeps its declared
        default otherwise, so no declared value can be silently ignored by the loader.
        """
        values = os.environ if environment is None else environment
        overrides = {
            name: converter(values[f"NAIVE_RAG_{name.upper()}"])
            for name, converter in _NAIVE_RAG_TUNABLES.items()
            if f"NAIVE_RAG_{name.upper()}" in values
        }
        return cls(
            **overrides,
            embedding_profile=values["EMBEDDING_PROFILE"].strip(),
        )


_GRAPHITI_TUNABLES: dict[str, Callable[[str], int | float]] = {
    "search_num_results": int,
    "max_context_facts": int,
    "effective_date_window_token_budget": int,
    "answer_max_output_tokens": int,
    "answer_temperature": float,
    "query_rewrite_max_output_tokens": int,
    "effective_date_max_output_tokens": int,
    "citation_max_output_tokens": int,
    "follow_up_max_output_tokens": int,
    "follow_up_temperature": float,
    "extraction_temperature": float,
    "expand_around_top_fact": _parse_boolean,
    "rerank_with_cross_encoder": _parse_boolean,
    "max_response_bytes": int,
}


@dataclass(frozen=True, slots=True)
class GraphitiSettings:
    """Graphiti retrieval, extraction, and generation settings."""

    embedding_profile: str
    search_num_results: int = 30
    max_context_facts: int = 10
    effective_date_window_token_budget: int = 40_000
    # DeepSeek spends output tokens on reasoning before the answer, and one context is sent per
    # supporting chunk, so a grounded answer over ten multi-source facts runs close to 3k tokens.
    # Too small a budget truncates the response into an empty string that cannot be parsed.
    answer_max_output_tokens: int = 8192
    answer_temperature: float = 0.3
    query_rewrite_max_output_tokens: int = 2048
    effective_date_max_output_tokens: int = 2048
    citation_max_output_tokens: int = 2048
    follow_up_max_output_tokens: int = 2048
    follow_up_temperature: float = 0.4
    extraction_temperature: float = 0.0
    expand_around_top_fact: bool = True
    rerank_with_cross_encoder: bool = False
    max_response_bytes: int = 1_048_576

    def __post_init__(self) -> None:
        """Reject invalid external-service limits before they reach Graphiti or an LLM provider."""
        positive_values = {
            "search_num_results": self.search_num_results,
            "max_context_facts": self.max_context_facts,
            "effective_date_window_token_budget": self.effective_date_window_token_budget,
            "answer_max_output_tokens": self.answer_max_output_tokens,
            "query_rewrite_max_output_tokens": self.query_rewrite_max_output_tokens,
            "effective_date_max_output_tokens": self.effective_date_max_output_tokens,
            "citation_max_output_tokens": self.citation_max_output_tokens,
            "max_response_bytes": self.max_response_bytes,
        }
        for name, value in positive_values.items():
            if value < 1:
                msg = f"{name} must be positive, got {value}"
                raise ValueError(msg)

        temperatures = {
            "answer_temperature": self.answer_temperature,
            "extraction_temperature": self.extraction_temperature,
        }
        for name, value in temperatures.items():
            if not isfinite(value) or not 0 <= value <= _MAX_LLM_TEMPERATURE:
                msg = f"{name} must be between 0 and {_MAX_LLM_TEMPERATURE}, got {value}"
                raise ValueError(msg)

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> GraphitiSettings:
        """Load Graphiti knowledge-graph settings from process environment variables.

        Every tunable above is overridable as ``GRAPHITI_<FIELD_NAME>`` and keeps its declared
        default otherwise, so no declared value can be silently ignored by the loader.
        """
        values = os.environ if environment is None else environment
        overrides = {
            name: converter(values[f"GRAPHITI_{name.upper()}"])
            for name, converter in _GRAPHITI_TUNABLES.items()
            if f"GRAPHITI_{name.upper()}" in values
        }
        return cls(
            **overrides,
            embedding_profile=values["EMBEDDING_PROFILE"].strip(),
        )


# Imports already present at the top of settings.py: os, Mapping, dataclass, field, isfinite, Callable


_REDIS_TUNABLES: dict[str, Callable[[str], str | float | int]] = {
    "password": str.strip,
    "socket_timeout_seconds": float,
    "max_turns": int,
    "ttl_seconds": int,
}


@dataclass(frozen=True, slots=True)
class RedisSettings:
    """Redis connection and conversation history settings."""

    url: str
    password: str = field(default="", repr=False)
    socket_timeout_seconds: float = 5.0
    max_turns: int = 10
    ttl_seconds: int = 3600

    def __post_init__(self) -> None:
        """Reject invalid Redis configuration."""
        if not self.url:
            msg = "Redis url must not be empty"
            raise ValueError(msg)
        if self.max_turns < 1:
            msg = f"max_turns must be positive, got {self.max_turns}"
            raise ValueError(msg)
        if self.ttl_seconds < 1:
            msg = f"ttl_seconds must be positive, got {self.ttl_seconds}"
            raise ValueError(msg)
        if not isfinite(self.socket_timeout_seconds) or self.socket_timeout_seconds <= 0:
            msg = f"socket_timeout_seconds must be positive, got {self.socket_timeout_seconds}"
            raise ValueError(msg)

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> RedisSettings:
        """Load Redis settings from process environment variables.

        Every tunable above is overridable as ``REDIS_<FIELD_NAME>`` and keeps its declared
        default otherwise, so no declared value can be silently ignored by the loader.
        """
        values = os.environ if environment is None else environment
        overrides = {
            name: converter(values[f"REDIS_{name.upper()}"])
            for name, converter in _REDIS_TUNABLES.items()
            if f"REDIS_{name.upper()}" in values
        }
        return cls(url=values["REDIS_URL"].strip(), **overrides)
