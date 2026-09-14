"""Typed, versioned configuration for ground-truth synthesis."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

from legal_rag.baml_transport import ReasoningEffort

SYNTHESIS_CONFIG_SCHEMA_VERSION = "5.6.0"
_MAX_LLM_TEMPERATURE = 2.0


@dataclass(frozen=True, slots=True)
class LlmStageSettings:
    """Generation bounds for one synthesis LLM operation."""

    max_output_tokens: int
    temperature: float
    reasoning_effort: ReasoningEffort

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

    retrieval_pool_size: int
    max_candidates_per_fact: int
    rrf_k: int
    bm25_k1: float
    bm25_b: float
    bm25_delta: float

    def __post_init__(self) -> None:
        """Reject invalid ranking parameters before candidate preparation."""
        for name, value in {
            "retrieval_pool_size": self.retrieval_pool_size,
            "max_candidates_per_fact": self.max_candidates_per_fact,
            "rrf_k": self.rrf_k,
        }.items():
            if value < 1:
                msg = f"{name} must be positive, got {value}"
                raise ValueError(msg)
        if not isfinite(self.bm25_k1) or self.bm25_k1 <= 0:
            msg = f"bm25_k1 must be positive, got {self.bm25_k1}"
            raise ValueError(msg)
        if not isfinite(self.bm25_b) or not 0 <= self.bm25_b <= 1:
            msg = f"bm25_b must be between 0 and 1, got {self.bm25_b}"
            raise ValueError(msg)
        if not isfinite(self.bm25_delta) or self.bm25_delta < 0:
            msg = f"bm25_delta must not be negative, got {self.bm25_delta}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class SynthesisSettings:
    """All reproducibility-sensitive settings for ground-truth synthesis."""

    embedding_profile: str
    embedding_batch_size: int
    embedding_max_concurrency: int
    embedding_max_retries: int
    embedding_retry_delay_seconds: float
    embedding_max_response_bytes: int
    vncorenlp_max_heap_megabytes: int
    vncorenlp_timeout_seconds: float
    llm_max_retries: int
    llm_retry_delay_seconds: float
    max_response_bytes: int
    bundle_overlap_threshold: float
    chunk_screening_batch_size: int
    chunk_screening_max_concurrency: int
    fact_extraction_batch_size: int
    fact_extraction_max_concurrency: int
    fact_extraction_max_preceding_chunks: int
    fact_extraction_max_following_chunks: int
    evidence_relation_batch_size: int
    evidence_relation_max_concurrency: int
    multi_hop_3_candidate_limit: int
    multi_hop_3_batch_size: int
    multi_hop_3_max_concurrency: int
    multi_hop_4_candidate_limit: int
    multi_hop_4_batch_size: int
    multi_hop_4_max_concurrency: int
    multi_hop_5_candidate_limit: int
    multi_hop_5_batch_size: int
    multi_hop_5_max_concurrency: int
    bundle_scope_batch_size: int
    bundle_scope_max_concurrency: int
    candidate_generation_batch_size: int
    candidate_generation_max_concurrency: int
    multi_turn_generation_batch_size: int
    multi_turn_generation_max_concurrency: int
    evidence_retrieval: EvidenceRetrievalSettings
    chunk_screening: LlmStageSettings
    document_metadata: LlmStageSettings
    fact_extraction: LlmStageSettings
    evidence_relation: LlmStageSettings
    multi_hop_3: LlmStageSettings
    multi_hop_4: LlmStageSettings
    multi_hop_5: LlmStageSettings
    bundle_scope: LlmStageSettings
    candidate_generation: LlmStageSettings
    multi_turn_generation: LlmStageSettings

    def __post_init__(self) -> None:
        """Reject invalid synthesis limits before a run starts."""
        if not self.embedding_profile:
            msg = "embedding_profile must not be empty"
            raise ValueError(msg)
        for name, value in {
            "embedding_max_retries": self.embedding_max_retries,
            "embedding_retry_delay_seconds": self.embedding_retry_delay_seconds,
            "llm_max_retries": self.llm_max_retries,
            "llm_retry_delay_seconds": self.llm_retry_delay_seconds,
            "vncorenlp_timeout_seconds": self.vncorenlp_timeout_seconds,
        }.items():
            if not isfinite(value) or value < 0:
                msg = f"{name} must not be negative, got {value}"
                raise ValueError(msg)
        positive_limits = {
            "max_response_bytes": self.max_response_bytes,
            "embedding_batch_size": self.embedding_batch_size,
            "embedding_max_concurrency": self.embedding_max_concurrency,
            "embedding_max_response_bytes": self.embedding_max_response_bytes,
            "vncorenlp_max_heap_megabytes": self.vncorenlp_max_heap_megabytes,
            "vncorenlp_timeout_seconds": self.vncorenlp_timeout_seconds,
            "chunk_screening_batch_size": self.chunk_screening_batch_size,
            "chunk_screening_max_concurrency": self.chunk_screening_max_concurrency,
            "fact_extraction_batch_size": self.fact_extraction_batch_size,
            "fact_extraction_max_concurrency": self.fact_extraction_max_concurrency,
            "evidence_relation_batch_size": self.evidence_relation_batch_size,
            "evidence_relation_max_concurrency": self.evidence_relation_max_concurrency,
            "multi_hop_3_candidate_limit": self.multi_hop_3_candidate_limit,
            "multi_hop_3_batch_size": self.multi_hop_3_batch_size,
            "multi_hop_3_max_concurrency": self.multi_hop_3_max_concurrency,
            "multi_hop_4_candidate_limit": self.multi_hop_4_candidate_limit,
            "multi_hop_4_batch_size": self.multi_hop_4_batch_size,
            "multi_hop_4_max_concurrency": self.multi_hop_4_max_concurrency,
            "multi_hop_5_candidate_limit": self.multi_hop_5_candidate_limit,
            "multi_hop_5_batch_size": self.multi_hop_5_batch_size,
            "multi_hop_5_max_concurrency": self.multi_hop_5_max_concurrency,
            "bundle_scope_batch_size": self.bundle_scope_batch_size,
            "bundle_scope_max_concurrency": self.bundle_scope_max_concurrency,
            "candidate_generation_batch_size": self.candidate_generation_batch_size,
            "candidate_generation_max_concurrency": self.candidate_generation_max_concurrency,
            "multi_turn_generation_batch_size": self.multi_turn_generation_batch_size,
            "multi_turn_generation_max_concurrency": self.multi_turn_generation_max_concurrency,
        }
        for name, value in positive_limits.items():
            if value < 1:
                msg = f"{name} must be positive, got {value}"
                raise ValueError(msg)
        if self.fact_extraction_max_preceding_chunks < 0 or self.fact_extraction_max_following_chunks < 0:
            msg = "fact-extraction context sizes must not be negative"
            raise ValueError(msg)
        for name, value in {"bundle_overlap_threshold": self.bundle_overlap_threshold}.items():
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
        multi_hop_3 = stages["multi_hop_3"]
        multi_hop_4 = stages["multi_hop_4"]
        multi_hop_5 = stages["multi_hop_5"]
        bundle_scope = stages["bundle_scope"]
        candidate_generation = stages["candidate_generation"]
        multi_turn_generation = stages["multi_turn_generation"]
        retrieval = relation["retrieval"]
        return cls(
            embedding_profile=config["embedding_profile"],
            embedding_batch_size=embedding["batch_size"],
            embedding_max_concurrency=embedding["max_concurrency"],
            embedding_max_retries=embedding["max_retries"],
            embedding_retry_delay_seconds=embedding["retry_delay_seconds"],
            embedding_max_response_bytes=embedding["max_response_bytes"],
            vncorenlp_max_heap_megabytes=lexical_retrieval["vncorenlp_max_heap_megabytes"],
            vncorenlp_timeout_seconds=lexical_retrieval["vncorenlp_timeout_seconds"],
            llm_max_retries=execution["max_retries"],
            llm_retry_delay_seconds=execution["retry_delay_seconds"],
            max_response_bytes=execution["max_response_bytes"],
            bundle_overlap_threshold=quality["bundle_overlap_threshold"],
            chunk_screening_batch_size=screening["batch_size"],
            chunk_screening_max_concurrency=screening["max_concurrency"],
            fact_extraction_batch_size=extraction["batch_size"],
            fact_extraction_max_concurrency=extraction["max_concurrency"],
            fact_extraction_max_preceding_chunks=extraction["max_preceding_chunks"],
            fact_extraction_max_following_chunks=extraction["max_following_chunks"],
            evidence_relation_batch_size=relation["batch_size"],
            evidence_relation_max_concurrency=relation["max_concurrency"],
            multi_hop_3_candidate_limit=multi_hop_3["candidate_limit"],
            multi_hop_3_batch_size=multi_hop_3["batch_size"],
            multi_hop_3_max_concurrency=multi_hop_3["max_concurrency"],
            multi_hop_4_candidate_limit=multi_hop_4["candidate_limit"],
            multi_hop_4_batch_size=multi_hop_4["batch_size"],
            multi_hop_4_max_concurrency=multi_hop_4["max_concurrency"],
            multi_hop_5_candidate_limit=multi_hop_5["candidate_limit"],
            multi_hop_5_batch_size=multi_hop_5["batch_size"],
            multi_hop_5_max_concurrency=multi_hop_5["max_concurrency"],
            bundle_scope_batch_size=bundle_scope["batch_size"],
            bundle_scope_max_concurrency=bundle_scope["max_concurrency"],
            candidate_generation_batch_size=candidate_generation["batch_size"],
            candidate_generation_max_concurrency=candidate_generation["max_concurrency"],
            multi_turn_generation_batch_size=multi_turn_generation["batch_size"],
            multi_turn_generation_max_concurrency=multi_turn_generation["max_concurrency"],
            evidence_retrieval=_evidence_retrieval_settings(retrieval),
            chunk_screening=_stage_settings(screening),
            document_metadata=_stage_settings(metadata),
            fact_extraction=_stage_settings(extraction),
            evidence_relation=_stage_settings(relation),
            multi_hop_3=_stage_settings(multi_hop_3),
            multi_hop_4=_stage_settings(multi_hop_4),
            multi_hop_5=_stage_settings(multi_hop_5),
            bundle_scope=_stage_settings(bundle_scope),
            candidate_generation=_stage_settings(candidate_generation),
            multi_turn_generation=_stage_settings(multi_turn_generation),
        )


def _stage_settings(config: dict[str, Any]) -> LlmStageSettings:
    """Build one stage's typed generation bounds."""
    return LlmStageSettings(
        max_output_tokens=config["max_output_tokens"],
        temperature=config["temperature"],
        reasoning_effort=config["reasoning_effort"],
    )


def _evidence_retrieval_settings(config: dict[str, Any]) -> EvidenceRetrievalSettings:
    """Build typed hybrid-retrieval settings from versioned config."""
    bm25 = config["bm25"]
    return EvidenceRetrievalSettings(
        retrieval_pool_size=config["retrieval_pool_size"],
        max_candidates_per_fact=config["max_candidates_per_fact"],
        rrf_k=config["rrf_k"],
        bm25_k1=bm25["k1"],
        bm25_b=bm25["b"],
        bm25_delta=bm25["delta"],
    )
