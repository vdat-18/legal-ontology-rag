"""Application settings loaded from environment variables."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field


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


@dataclass(frozen=True, slots=True)
class ChunkerSettings:
    """Embedding profile selection and token budgets for chunking."""

    embedding_profile: str
    max_tokens: int
    split_overlap_tokens: int

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> ChunkerSettings:
        """Load chunker settings from process environment variables."""
        values = os.environ if environment is None else environment
        return cls(
            embedding_profile=values["EMBEDDING_PROFILE"].strip(),
            max_tokens=int(values["CHUNK_MAX_TOKENS"]),
            split_overlap_tokens=int(values["CHUNK_SPLIT_OVERLAP_TOKENS"]),
        )


@dataclass(frozen=True, slots=True)
class EvaluationSettings:
    """Threshold and batch configuration for ground-truth synthesis."""

    embedding_profile: str
    chunk_quality_threshold: float
    span_match_threshold: float
    bundle_overlap_threshold: float
    dedup_cosine_threshold: float
    max_quality_retries: int
    max_simulator_turns: int
    chunk_quality_max_output_tokens: int = 256
    chunk_quality_temperature: float = 0.0
    fact_extraction_max_output_tokens: int = 4096
    fact_extraction_temperature: float = 0.0
    candidate_generation_max_output_tokens: int = 2048
    candidate_generation_temperature: float = 0.0
    verification_max_output_tokens: int = 1024
    verification_temperature: float = 0.0
    max_response_bytes: int = 1_048_576

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> EvaluationSettings:
        """Load ground-truth synthesis settings from process environment variables."""
        values = os.environ if environment is None else environment
        return cls(
            embedding_profile=values["EMBEDDING_PROFILE"].strip(),
            chunk_quality_threshold=float(values["GROUNDTRUTH_CHUNK_QUALITY_THRESHOLD"]),
            span_match_threshold=float(values["GROUNDTRUTH_SPAN_MATCH_THRESHOLD"]),
            bundle_overlap_threshold=float(values["GROUNDTRUTH_BUNDLE_OVERLAP_THRESHOLD"]),
            dedup_cosine_threshold=float(values["GROUNDTRUTH_DEDUP_COSINE_THRESHOLD"]),
            max_quality_retries=int(values["GROUNDTRUTH_MAX_QUALITY_RETRIES"]),
            max_simulator_turns=int(values["GROUNDTRUTH_MAX_SIMULATOR_TURNS"]),
        )


@dataclass(frozen=True, slots=True)
class NaiveRagSettings:
    """Retrieval, indexing, and generation settings for the naive RAG baseline."""

    embedding_profile: str
    top_k: int
    milvus_uri: str
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
        """Load naive RAG baseline settings from process environment variables."""
        values = os.environ if environment is None else environment
        return cls(
            embedding_profile=values["EMBEDDING_PROFILE"].strip(),
            top_k=int(values["NAIVE_RAG_TOP_K"]),
            milvus_uri=values["MILVUS_URI"].strip(),
        )
