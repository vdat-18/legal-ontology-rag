"""Tests for versioned ground-truth synthesis configuration."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from legal_rag.benchmark_synthesis.config import SynthesisSettings

_SYNTHESIS_CONFIG_PATH = Path(__file__).resolve().parents[2] / "data" / "config" / "groundtruth" / "synthesis_v1.json"


def test_synthesis_settings_load_every_pipeline_stage() -> None:
    """The checked-in config owns batch, retry, retrieval, and LLM bounds."""
    settings = SynthesisSettings.from_path(_SYNTHESIS_CONFIG_PATH)

    assert settings.embedding_profile == "bge_m3"
    assert settings.embedding_batch_size == 64
    assert settings.embedding_max_response_bytes == 16_777_216
    assert settings.vncorenlp_max_heap_megabytes == 512
    assert settings.vncorenlp_timeout_seconds == 120
    assert settings.llm_max_retries == 2
    assert settings.llm_retry_delay_seconds == 120
    assert settings.chunk_screening_batch_size == 50
    assert settings.chunk_screening_max_concurrency == 4
    assert settings.chunk_screening.max_output_tokens == 16_384
    assert settings.chunk_screening.reasoning_effort == "medium"
    assert settings.document_metadata_opening_chunk_count == 10
    assert settings.document_metadata_closing_chunk_count == 10
    assert settings.document_metadata.reasoning_effort == "medium"
    assert settings.fact_extraction_batch_size == 1
    assert settings.fact_extraction_max_concurrency == 4
    assert settings.fact_extraction_max_preceding_chunks == 2
    assert settings.fact_extraction_max_following_chunks == 1
    assert settings.fact_extraction.max_output_tokens == 32_768
    assert settings.fact_extraction.reasoning_effort == "medium"
    assert settings.evidence_relation_batch_size == 10
    assert settings.evidence_relation_max_concurrency == 10
    assert settings.candidate_generation_batch_size == 5
    assert settings.candidate_generation_max_concurrency == 10
    assert settings.evidence_retrieval.retrieval_pool_size == 30
    assert settings.evidence_retrieval.max_candidates_per_fact == 10
    assert settings.evidence_retrieval.rrf_k == 60
    assert settings.evidence_retrieval.bm25_k1 == 1.2
    assert settings.evidence_retrieval.bm25_b == 0.4
    assert settings.evidence_retrieval.bm25_delta == 0.5
    assert settings.evidence_relation.max_output_tokens == 4096
    assert settings.evidence_relation.reasoning_effort == "medium"
    assert settings.candidate_generation.max_output_tokens == 8192
    assert settings.candidate_generation.reasoning_effort == "medium"


def test_synthesis_settings_reject_unknown_schema(tmp_path: Path) -> None:
    """A config written for another schema cannot silently start a run."""
    config_path = tmp_path / "synthesis.json"
    config_path.write_text(json.dumps({"schema_version": "99.0.0"}), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported synthesis config schema"):
        SynthesisSettings.from_path(config_path)


def test_synthesis_settings_reject_negative_retry_delay() -> None:
    """A negative delay cannot start a synthesis run."""
    settings = SynthesisSettings.from_path(_SYNTHESIS_CONFIG_PATH)

    with pytest.raises(ValueError, match="llm_retry_delay_seconds must not be negative"):
        replace(settings, llm_retry_delay_seconds=-1)


def test_synthesis_settings_reject_non_positive_fact_extraction_batch_size() -> None:
    """Fact extraction cannot start with an empty batch size."""
    settings = SynthesisSettings.from_path(_SYNTHESIS_CONFIG_PATH)

    with pytest.raises(ValueError, match="fact_extraction_batch_size must be positive"):
        replace(settings, fact_extraction_batch_size=0)


def test_synthesis_settings_reject_non_positive_screening_max_concurrency() -> None:
    """Chunk screening cannot start with a non-positive worker pool size."""
    settings = SynthesisSettings.from_path(_SYNTHESIS_CONFIG_PATH)

    with pytest.raises(ValueError, match="chunk_screening_max_concurrency must be positive"):
        replace(settings, chunk_screening_max_concurrency=0)


def test_synthesis_settings_reject_non_positive_fact_extraction_max_concurrency() -> None:
    """Fact extraction cannot start with a non-positive worker pool size."""
    settings = SynthesisSettings.from_path(_SYNTHESIS_CONFIG_PATH)

    with pytest.raises(ValueError, match="fact_extraction_max_concurrency must be positive"):
        replace(settings, fact_extraction_max_concurrency=0)


def test_synthesis_settings_reject_negative_fact_extraction_context_size() -> None:
    """Fact extraction context limits cannot be negative."""
    settings = SynthesisSettings.from_path(_SYNTHESIS_CONFIG_PATH)

    with pytest.raises(ValueError, match="fact-extraction context sizes must not be negative"):
        replace(settings, fact_extraction_max_preceding_chunks=-1)


def test_synthesis_settings_reject_non_positive_evidence_relation_max_concurrency() -> None:
    """Evidence relation confirmation cannot start with a non-positive worker pool size."""
    settings = SynthesisSettings.from_path(_SYNTHESIS_CONFIG_PATH)

    with pytest.raises(ValueError, match="evidence_relation_max_concurrency must be positive"):
        replace(settings, evidence_relation_max_concurrency=0)


def test_synthesis_settings_reject_non_positive_candidate_generation_max_concurrency() -> None:
    """Candidate generation cannot start with a non-positive worker pool size."""
    settings = SynthesisSettings.from_path(_SYNTHESIS_CONFIG_PATH)

    with pytest.raises(ValueError, match="candidate_generation_max_concurrency must be positive"):
        replace(settings, candidate_generation_max_concurrency=0)


def test_synthesis_settings_reject_non_positive_candidate_generation_batch_size() -> None:
    """Candidate generation cannot start with an empty batch size."""
    settings = SynthesisSettings.from_path(_SYNTHESIS_CONFIG_PATH)

    with pytest.raises(ValueError, match="candidate_generation_batch_size must be positive"):
        replace(settings, candidate_generation_batch_size=0)
