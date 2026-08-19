"""Typed, versioned configuration for ground-truth synthesis."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

SYNTHESIS_CONFIG_SCHEMA_VERSION = "1.6.0"
_MAX_LLM_TEMPERATURE = 2.0


@dataclass(frozen=True, slots=True)
class LlmStageSettings:
    """Generation bounds for one synthesis LLM operation."""

    max_output_tokens: int
    temperature: float

    def __post_init__(self) -> None:
        """Reject invalid output limits before provider construction."""
        if self.max_output_tokens < 1:
            msg = f"max_output_tokens must be positive, got {self.max_output_tokens}"
            raise ValueError(msg)
        if not isfinite(self.temperature) or not 0 <= self.temperature <= _MAX_LLM_TEMPERATURE:
            msg = f"temperature must be between 0 and {_MAX_LLM_TEMPERATURE}, got {self.temperature}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class EvidenceRetrievalSettings:
    """Hybrid candidate-retrieval settings for evidence relation confirmation."""

    max_candidates_per_fact: int
    min_similarity: float
    pool_multiplier: int
    rrf_k: int
    bm25_k1: float
    bm25_b: float
    bm25_delta: float
    dense_weight: float
    lexical_core_weight: float
    lexical_conditions_weight: float
    lexical_consequences_weight: float
    lexical_authority_weight: float

    def __post_init__(self) -> None:
        """Reject invalid ranking parameters before candidate preparation."""
        for name, value in {
            "max_candidates_per_fact": self.max_candidates_per_fact,
            "pool_multiplier": self.pool_multiplier,
            "rrf_k": self.rrf_k,
        }.items():
            if value < 1:
                msg = f"{name} must be positive, got {value}"
                raise ValueError(msg)
        if not isfinite(self.min_similarity) or not 0 <= self.min_similarity <= 1:
            msg = f"min_similarity must be between 0 and 1, got {self.min_similarity}"
            raise ValueError(msg)
        if not isfinite(self.bm25_k1) or self.bm25_k1 <= 0:
            msg = f"bm25_k1 must be positive, got {self.bm25_k1}"
            raise ValueError(msg)
        if not isfinite(self.bm25_b) or not 0 <= self.bm25_b <= 1:
            msg = f"bm25_b must be between 0 and 1, got {self.bm25_b}"
            raise ValueError(msg)
        non_negative_values = {
            "bm25_delta": self.bm25_delta,
            "dense_weight": self.dense_weight,
            "lexical_core_weight": self.lexical_core_weight,
            "lexical_conditions_weight": self.lexical_conditions_weight,
            "lexical_consequences_weight": self.lexical_consequences_weight,
            "lexical_authority_weight": self.lexical_authority_weight,
        }
        for name, value in non_negative_values.items():
            if not isfinite(value) or value < 0:
                msg = f"{name} must not be negative, got {value}"
                raise ValueError(msg)
        ranking_weights = {name: value for name, value in non_negative_values.items() if name != "bm25_delta"}
        if not any(value > 0 for value in ranking_weights.values()):
            msg = "At least one evidence-retrieval ranking weight must be positive"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class SynthesisSettings:
    """All reproducibility-sensitive settings for ground-truth synthesis."""

    embedding_profile: str
    embedding_batch_size: int
    embedding_max_response_bytes: int
    vncorenlp_max_heap_megabytes: int
    vncorenlp_timeout_seconds: float
    llm_max_retries: int
    llm_retry_delay_seconds: float
    max_response_bytes: int
    span_match_threshold: float
    bundle_overlap_threshold: float
    max_quality_retries: int
    chunk_screening_batch_size: int
    chunk_screening_max_concurrency: int
    document_metadata_opening_chunk_count: int
    document_metadata_closing_chunk_count: int
    fact_extraction_batch_size: int
    fact_extraction_max_concurrency: int
    fact_extraction_max_preceding_chunks: int
    fact_extraction_max_following_chunks: int
    evidence_relation_batch_size: int
    evidence_relation_max_concurrency: int
    candidate_generation_batch_size: int
    candidate_generation_max_concurrency: int
    evidence_retrieval: EvidenceRetrievalSettings
    chunk_screening: LlmStageSettings
    document_metadata: LlmStageSettings
    fact_extraction: LlmStageSettings
    evidence_relation: LlmStageSettings
    candidate_generation: LlmStageSettings
    candidate_critique: LlmStageSettings
    candidate_evolution: LlmStageSettings
    verification: LlmStageSettings

    def __post_init__(self) -> None:
        """Reject invalid synthesis limits before a run starts."""
        if not self.embedding_profile:
            msg = "embedding_profile must not be empty"
            raise ValueError(msg)
        for name, value in {
            "llm_max_retries": self.llm_max_retries,
            "llm_retry_delay_seconds": self.llm_retry_delay_seconds,
            "max_quality_retries": self.max_quality_retries,
            "vncorenlp_timeout_seconds": self.vncorenlp_timeout_seconds,
        }.items():
            if not isfinite(value) or value < 0:
                msg = f"{name} must not be negative, got {value}"
                raise ValueError(msg)
        positive_limits = {
            "max_response_bytes": self.max_response_bytes,
            "embedding_batch_size": self.embedding_batch_size,
            "embedding_max_response_bytes": self.embedding_max_response_bytes,
            "vncorenlp_max_heap_megabytes": self.vncorenlp_max_heap_megabytes,
            "vncorenlp_timeout_seconds": self.vncorenlp_timeout_seconds,
            "chunk_screening_batch_size": self.chunk_screening_batch_size,
            "chunk_screening_max_concurrency": self.chunk_screening_max_concurrency,
            "document_metadata_opening_chunk_count": self.document_metadata_opening_chunk_count,
            "document_metadata_closing_chunk_count": self.document_metadata_closing_chunk_count,
            "fact_extraction_batch_size": self.fact_extraction_batch_size,
            "fact_extraction_max_concurrency": self.fact_extraction_max_concurrency,
            "evidence_relation_batch_size": self.evidence_relation_batch_size,
            "evidence_relation_max_concurrency": self.evidence_relation_max_concurrency,
            "candidate_generation_batch_size": self.candidate_generation_batch_size,
            "candidate_generation_max_concurrency": self.candidate_generation_max_concurrency,
        }
        for name, value in positive_limits.items():
            if value < 1:
                msg = f"{name} must be positive, got {value}"
                raise ValueError(msg)
        if self.fact_extraction_max_preceding_chunks < 0 or self.fact_extraction_max_following_chunks < 0:
            msg = "fact-extraction context sizes must not be negative"
            raise ValueError(msg)
        for name, value in {
            "span_match_threshold": self.span_match_threshold,
            "bundle_overlap_threshold": self.bundle_overlap_threshold,
        }.items():
            if not isfinite(value) or not 0 <= value <= 1:
                msg = f"{name} must be between 0 and 1, got {value}"
                raise ValueError(msg)

    @classmethod
    def from_path(cls, config_path: Path) -> SynthesisSettings:
        """Load one versioned synthesis configuration from JSON."""
        config: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
        if config["schema_version"] != SYNTHESIS_CONFIG_SCHEMA_VERSION:
            msg = f"Unsupported synthesis config schema: {config['schema_version']}"
            raise ValueError(msg)
        execution = config["llm_execution"]
        embedding = config["embedding"]
        lexical_retrieval = config["lexical_retrieval"]
        quality = config["quality"]
        stages = config["stages"]
        screening = stages["chunk_screening"]
        metadata = stages["document_metadata"]
        extraction = stages["fact_extraction"]
        relation = stages["evidence_relation"]
        retrieval = relation["retrieval"]
        return cls(
            embedding_profile=config["embedding_profile"],
            embedding_batch_size=embedding["batch_size"],
            embedding_max_response_bytes=embedding["max_response_bytes"],
            vncorenlp_max_heap_megabytes=lexical_retrieval["vncorenlp_max_heap_megabytes"],
            vncorenlp_timeout_seconds=lexical_retrieval["vncorenlp_timeout_seconds"],
            llm_max_retries=execution["max_retries"],
            llm_retry_delay_seconds=execution["retry_delay_seconds"],
            max_response_bytes=execution["max_response_bytes"],
            span_match_threshold=quality["span_match_threshold"],
            bundle_overlap_threshold=quality["bundle_overlap_threshold"],
            max_quality_retries=quality["max_retries"],
            chunk_screening_batch_size=screening["batch_size"],
            chunk_screening_max_concurrency=screening["max_concurrency"],
            document_metadata_opening_chunk_count=metadata["opening_chunk_count"],
            document_metadata_closing_chunk_count=metadata["closing_chunk_count"],
            fact_extraction_batch_size=extraction["batch_size"],
            fact_extraction_max_concurrency=extraction["max_concurrency"],
            fact_extraction_max_preceding_chunks=extraction["max_preceding_chunks"],
            fact_extraction_max_following_chunks=extraction["max_following_chunks"],
            evidence_relation_batch_size=relation["batch_size"],
            evidence_relation_max_concurrency=relation["max_concurrency"],
            candidate_generation_batch_size=stages["candidate_generation"]["batch_size"],
            candidate_generation_max_concurrency=stages["candidate_generation"]["max_concurrency"],
            evidence_retrieval=_evidence_retrieval_settings(retrieval),
            chunk_screening=_stage_settings(screening),
            document_metadata=_stage_settings(metadata),
            fact_extraction=_stage_settings(extraction),
            evidence_relation=_stage_settings(relation),
            candidate_generation=_stage_settings(stages["candidate_generation"]),
            candidate_critique=_stage_settings(stages["candidate_critique"]),
            candidate_evolution=_stage_settings(stages["candidate_evolution"]),
            verification=_stage_settings(stages["verification"]),
        )


def _stage_settings(config: dict[str, Any]) -> LlmStageSettings:
    """Build one stage's typed generation bounds."""
    return LlmStageSettings(max_output_tokens=config["max_output_tokens"], temperature=config["temperature"])


def _evidence_retrieval_settings(config: dict[str, Any]) -> EvidenceRetrievalSettings:
    """Build typed hybrid-retrieval settings from versioned config."""
    bm25 = config["bm25"]
    weights = config["weights"]
    return EvidenceRetrievalSettings(
        max_candidates_per_fact=config["max_candidates_per_fact"],
        min_similarity=config["min_similarity"],
        pool_multiplier=config["pool_multiplier"],
        rrf_k=config["rrf_k"],
        bm25_k1=bm25["k1"],
        bm25_b=bm25["b"],
        bm25_delta=bm25["delta"],
        dense_weight=weights["dense"],
        lexical_core_weight=weights["core"],
        lexical_conditions_weight=weights["conditions"],
        lexical_consequences_weight=weights["consequences"],
        lexical_authority_weight=weights["authority"],
    )
