"""Typed, versioned configuration for ground-truth synthesis."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

SYNTHESIS_CONFIG_SCHEMA_VERSION = "1.0.0"
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
class SynthesisSettings:
    """All reproducibility-sensitive settings for ground-truth synthesis."""

    embedding_profile: str
    llm_max_retries: int
    llm_retry_delay_seconds: float
    max_response_bytes: int
    span_match_threshold: float
    bundle_overlap_threshold: float
    max_quality_retries: int
    chunk_screening_batch_size: int
    document_metadata_opening_chunk_count: int
    document_metadata_closing_chunk_count: int
    fact_extraction_batch_size: int
    fact_extraction_max_preceding_chunks: int
    fact_extraction_max_following_chunks: int
    relation_max_candidates_per_fact: int
    relation_min_similarity: float
    relation_pool_multiplier: int
    relation_rrf_k: int
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
        }.items():
            if not isfinite(value) or value < 0:
                msg = f"{name} must not be negative, got {value}"
                raise ValueError(msg)
        positive_limits = {
            "max_response_bytes": self.max_response_bytes,
            "chunk_screening_batch_size": self.chunk_screening_batch_size,
            "document_metadata_opening_chunk_count": self.document_metadata_opening_chunk_count,
            "document_metadata_closing_chunk_count": self.document_metadata_closing_chunk_count,
            "fact_extraction_batch_size": self.fact_extraction_batch_size,
            "relation_max_candidates_per_fact": self.relation_max_candidates_per_fact,
            "relation_pool_multiplier": self.relation_pool_multiplier,
            "relation_rrf_k": self.relation_rrf_k,
        }
        for name, value in positive_limits.items():
            if value < 1:
                msg = f"{name} must be positive, got {value}"
                raise ValueError(msg)
        if self.fact_extraction_max_preceding_chunks < 0 or self.fact_extraction_max_following_chunks < 0:
            msg = "fact-extraction context sizes must not be negative"
            raise ValueError(msg)
        if not isfinite(self.relation_min_similarity) or not 0 <= self.relation_min_similarity <= 1:
            msg = f"relation_min_similarity must be between 0 and 1, got {self.relation_min_similarity}"
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
        quality = config["quality"]
        stages = config["stages"]
        screening = stages["chunk_screening"]
        metadata = stages["document_metadata"]
        extraction = stages["fact_extraction"]
        relation = stages["evidence_relation"]
        return cls(
            embedding_profile=config["embedding_profile"],
            llm_max_retries=execution["max_retries"],
            llm_retry_delay_seconds=execution["retry_delay_seconds"],
            max_response_bytes=execution["max_response_bytes"],
            span_match_threshold=quality["span_match_threshold"],
            bundle_overlap_threshold=quality["bundle_overlap_threshold"],
            max_quality_retries=quality["max_retries"],
            chunk_screening_batch_size=screening["batch_size"],
            document_metadata_opening_chunk_count=metadata["opening_chunk_count"],
            document_metadata_closing_chunk_count=metadata["closing_chunk_count"],
            fact_extraction_batch_size=extraction["batch_size"],
            fact_extraction_max_preceding_chunks=extraction["max_preceding_chunks"],
            fact_extraction_max_following_chunks=extraction["max_following_chunks"],
            relation_max_candidates_per_fact=relation["max_candidates_per_fact"],
            relation_min_similarity=relation["min_similarity"],
            relation_pool_multiplier=relation["pool_multiplier"],
            relation_rrf_k=relation["rrf_k"],
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
