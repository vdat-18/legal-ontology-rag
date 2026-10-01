"""Configuration loader for the Corrective RAG pipeline.

Loads LLM, embedding, and vector database settings from environment variables.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite

from dotenv import load_dotenv

from legal_rag.settings import ChunkerSettings, Settings

_CHUNKER_ENV_DEFAULTS = {
    "EMBEDDING_PROFILE": "vietnamese_embedding",
    "CHUNK_MAX_TOKENS": "512",
    "CHUNK_SPLIT_OVERLAP_TOKENS": "64",
}
_DEFAULT_MILVUS_URI = "http://localhost:19530"
_MAX_LLM_TEMPERATURE = 2.0


@dataclass(frozen=True, slots=True)
class CorrectiveRetrievalConfig:
    """Embedding and Milvus settings needed by Corrective retrieval-only workflows."""

    embedding_profile: str
    milvus_uri: str
    milvus_token: str | None
    milvus_database: str
    milvus_collection_name: str
    milvus_index_type: str
    milvus_metric_type: str
    milvus_nlist: int
    milvus_nprobe: int
    milvus_flush_after_insert: bool
    index_batch_size: int

    @classmethod
    def from_env(cls, environment: Mapping[str, str] | None = None) -> CorrectiveRetrievalConfig:
        """Load retrieval settings without requiring answer-model credentials."""
        if environment is None:
            load_dotenv()
        values = os.environ if environment is None else environment
        return _corrective_retrieval_config(values)


@dataclass(frozen=True, slots=True)
class RAGConfig:
    """Configuration settings for the Corrective RAG system."""

    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_timeout_seconds: float
    llm_extra_request_body: Mapping[str, object]
    embedding_profile: str
    milvus_uri: str
    milvus_token: str | None
    milvus_database: str
    milvus_collection_name: str
    milvus_index_type: str
    milvus_metric_type: str
    milvus_nlist: int
    milvus_nprobe: int
    milvus_flush_after_insert: bool
    index_batch_size: int
    evaluator_max_output_tokens: int
    evaluator_temperature: float
    answer_max_output_tokens: int
    answer_temperature: float
    query_reformulation_max_output_tokens: int
    query_reformulation_temperature: float
    query_rewrite_max_output_tokens: int
    query_rewrite_temperature: float
    evaluator_upper_threshold: float
    evaluator_lower_threshold: float
    max_response_bytes: int
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

    def __post_init__(self) -> None:
        """Reject invalid LLM response bounds before creating provider clients."""
        for name, value in {
            "evaluator_max_output_tokens": self.evaluator_max_output_tokens,
            "answer_max_output_tokens": self.answer_max_output_tokens,
            "query_reformulation_max_output_tokens": self.query_reformulation_max_output_tokens,
            "query_rewrite_max_output_tokens": self.query_rewrite_max_output_tokens,
            "max_response_bytes": self.max_response_bytes,
        }.items():
            if value < 1:
                msg = f"{name} must be positive, got {value}"
                raise ValueError(msg)
        for name, value in {
            "evaluator_temperature": self.evaluator_temperature,
            "answer_temperature": self.answer_temperature,
            "query_reformulation_temperature": self.query_reformulation_temperature,
            "query_rewrite_temperature": self.query_rewrite_temperature,
        }.items():
            if not isfinite(value) or not 0 <= value <= _MAX_LLM_TEMPERATURE:
                msg = f"{name} must be between 0 and {_MAX_LLM_TEMPERATURE}, got {value}"
                raise ValueError(msg)

    @classmethod
    def from_env(cls, environment: Mapping[str, str] | None = None) -> RAGConfig:
        """Load configuration from environment variables.

        Raises:
            ValueError: If required configuration environment variables are missing.
        """
        if environment is None:
            load_dotenv()
        values = os.environ if environment is None else environment
        llm_settings = Settings.from_environment(values)
        retrieval_settings = _corrective_retrieval_config(values)

        return cls(
            llm_base_url=llm_settings.llm_base_url,
            llm_api_key=llm_settings.llm_api_key,
            llm_model=llm_settings.llm_model,
            llm_timeout_seconds=llm_settings.llm_timeout_seconds,
            llm_extra_request_body=dict(llm_settings.llm_extra_request_body),
            embedding_profile=retrieval_settings.embedding_profile,
            milvus_uri=retrieval_settings.milvus_uri,
            milvus_token=retrieval_settings.milvus_token,
            milvus_database=retrieval_settings.milvus_database,
            milvus_collection_name=retrieval_settings.milvus_collection_name,
            milvus_index_type=retrieval_settings.milvus_index_type,
            milvus_metric_type=retrieval_settings.milvus_metric_type,
            milvus_nlist=retrieval_settings.milvus_nlist,
            milvus_nprobe=retrieval_settings.milvus_nprobe,
            milvus_flush_after_insert=retrieval_settings.milvus_flush_after_insert,
            index_batch_size=retrieval_settings.index_batch_size,
            evaluator_max_output_tokens=_read_int(values, "CRAG_EVALUATOR_MAX_OUTPUT_TOKENS", 1024),
            evaluator_temperature=_read_float(values, "CRAG_EVALUATOR_TEMPERATURE", 0.0),
            answer_max_output_tokens=_read_int(values, "CRAG_ANSWER_MAX_OUTPUT_TOKENS", 16384),
            answer_temperature=_read_float(values, "CRAG_ANSWER_TEMPERATURE", 0.0),
            query_reformulation_max_output_tokens=_read_int(values, "CRAG_QUERY_REFORMULATION_MAX_OUTPUT_TOKENS", 1024),
            query_reformulation_temperature=_read_float(values, "CRAG_QUERY_REFORMULATION_TEMPERATURE", 0.3),
            query_rewrite_max_output_tokens=_read_int(values, "CRAG_QUERY_REWRITE_MAX_OUTPUT_TOKENS", 4096),
            query_rewrite_temperature=_read_float(values, "CRAG_QUERY_REWRITE_TEMPERATURE", 0.0),
            evaluator_upper_threshold=_read_float(values, "CRAG_EVALUATOR_UPPER_THRESHOLD", 0.70),
            evaluator_lower_threshold=_read_float(values, "CRAG_EVALUATOR_LOWER_THRESHOLD", 0.35),
            max_response_bytes=_read_int(values, "CRAG_MAX_RESPONSE_BYTES", 1_048_576),
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


def _corrective_retrieval_config(values: Mapping[str, str]) -> CorrectiveRetrievalConfig:
    """Resolve the shared Corrective embedding and Milvus controls."""
    chunker_settings = ChunkerSettings.from_environment(_with_defaults(values, _CHUNKER_ENV_DEFAULTS))
    return CorrectiveRetrievalConfig(
        embedding_profile=chunker_settings.embedding_profile,
        milvus_uri=values.get("MILVUS_URI", _DEFAULT_MILVUS_URI),
        milvus_token=values.get("MILVUS_TOKEN", "").strip() or None,
        milvus_database=values.get("MILVUS_DATABASE", "default").strip() or "default",
        milvus_collection_name=values.get("CRAG_MILVUS_COLLECTION_NAME", "legal_chunks").strip() or "legal_chunks",
        milvus_index_type=values.get("CRAG_MILVUS_INDEX_TYPE", "IVF_FLAT").strip() or "IVF_FLAT",
        milvus_metric_type=values.get("CRAG_MILVUS_METRIC_TYPE", "COSINE").strip() or "COSINE",
        milvus_nlist=_read_int(values, "CRAG_MILVUS_NLIST", 128),
        milvus_nprobe=_read_int(values, "CRAG_MILVUS_NPROBE", 16),
        milvus_flush_after_insert=_read_bool(values, "CRAG_MILVUS_FLUSH_AFTER_INSERT", True),
        index_batch_size=_read_int(values, "MILVUS_INDEX_BATCH_SIZE", 32),
    )


def _with_defaults(values: Mapping[str, str], defaults: Mapping[str, str]) -> Mapping[str, str]:
    """Return environment values with defaults for optional CRAG commands."""
    merged = dict(defaults)
    merged.update(values)
    return merged


def _read_float(values: Mapping[str, str], name: str, default: float) -> float:
    """Read a float environment variable."""
    return float(values.get(name, str(default)))


def _read_int(values: Mapping[str, str], name: str, default: int) -> int:
    """Read an integer environment variable."""
    return int(values.get(name, str(default)))


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
