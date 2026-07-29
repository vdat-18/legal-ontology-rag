"""Application settings loaded from environment variables."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from math import isfinite

_MAX_LLM_TEMPERATURE = 2.0


@dataclass(frozen=True, slots=True)
class Settings:
    """LLM connection and bounded parser execution settings."""

    llm_base_url: str
    llm_api_key: str = field(repr=False)
    llm_model: str
    llm_timeout_seconds: float
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


_EVALUATION_TUNABLES: dict[str, Callable[[str], float | int]] = {
    "span_match_threshold": float,
    "bundle_overlap_threshold": float,
    "max_quality_retries": int,
}


@dataclass(frozen=True, slots=True)
class EvaluationSettings:
    """Threshold and batch configuration for ground-truth synthesis."""

    embedding_profile: str
    span_match_threshold: float = 0.85
    bundle_overlap_threshold: float = 0.5
    max_quality_retries: int = 2
    chunk_screening_max_output_tokens: int = 2048
    chunk_screening_temperature: float = 0.0
    fact_extraction_max_output_tokens: int = 4096
    fact_extraction_temperature: float = 0.0
    candidate_generation_max_output_tokens: int = 2048
    candidate_generation_temperature: float = 0.0
    verification_max_output_tokens: int = 1024
    verification_temperature: float = 0.0
    max_response_bytes: int = 1_048_576

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> EvaluationSettings:
        """Load ground-truth synthesis settings from process environment variables.

        Every tunable above is overridable as ``GROUNDTRUTH_<FIELD_NAME>`` and keeps its declared
        default otherwise, so no declared value can be silently ignored by the loader.
        """
        values = os.environ if environment is None else environment
        overrides = {
            name: converter(values[f"GROUNDTRUTH_{name.upper()}"])
            for name, converter in _EVALUATION_TUNABLES.items()
            if f"GROUNDTRUTH_{name.upper()}" in values
        }
        return cls(**overrides, embedding_profile=values["EMBEDDING_PROFILE"].strip())


_NAIVE_RAG_TUNABLES: dict[str, Callable[[str], int]] = {
    "top_k": int,
}


@dataclass(frozen=True, slots=True)
class NaiveRagSettings:
    """Retrieval, indexing, and generation settings for the naive RAG baseline."""

    embedding_profile: str
    milvus_uri: str
    top_k: int = 10
    milvus_database: str = "naive_rag_b0"
    milvus_collection_name: str = "naive_rag_chunks"
    milvus_index_type: str = "IVF_FLAT"
    milvus_metric_type: str = "COSINE"
    milvus_nlist: int = 128
    milvus_nprobe: int = 16
    milvus_insert_batch_size: int = 200
    answer_max_output_tokens: int = 4096
    answer_temperature: float = 0.3
    max_response_bytes: int = 1_048_576

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
            milvus_uri=values["MILVUS_URI"].strip(),
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
    "extraction_temperature": float,
    "max_response_bytes": int,
}


@dataclass(frozen=True, slots=True)
class GraphitiSettings:
    """Embedding profile selection and Neo4j connection settings for the Graphiti knowledge graph."""

    embedding_profile: str
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str = field(repr=False)
    search_num_results: int = 30
    max_context_facts: int = 10
    effective_date_window_token_budget: int = 40_000
    answer_max_output_tokens: int = 4096
    answer_temperature: float = 0.3
    query_rewrite_max_output_tokens: int = 2048
    effective_date_max_output_tokens: int = 2048
    citation_max_output_tokens: int = 2048
    extraction_temperature: float = 0.0
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
            neo4j_uri=values["NEO4J_URI"].strip(),
            neo4j_user=values["NEO4J_USER"].strip(),
            neo4j_password=values["NEO4J_PASSWORD"].strip(),
        )
