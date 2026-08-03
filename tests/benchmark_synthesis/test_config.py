"""Tests for versioned ground-truth synthesis configuration."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from legal_rag.benchmark_synthesis.config import SynthesisSettings

_SYNTHESIS_CONFIG_PATH = Path(__file__).resolve().parents[2] / "data" / "config" / "groundtruth" / "synthesis_v1.json"


def test_synthesis_settings_load_every_pipeline_stage() -> None:
    """The checked-in config owns batch, retry, quality, and per-operation LLM bounds."""
    settings = SynthesisSettings.from_path(_SYNTHESIS_CONFIG_PATH)

    assert settings.embedding_profile == "bge_m3"
    assert settings.embedding_batch_size == 64
    assert settings.embedding_max_response_bytes == 16_777_216
    assert settings.llm_max_retries == 2
    assert settings.llm_retry_delay_seconds == 120
    assert settings.chunk_screening_batch_size == 50
    assert settings.chunk_screening.max_output_tokens == 16_384
    assert settings.document_metadata_opening_chunk_count == 3
    assert settings.document_metadata_closing_chunk_count == 5
    assert settings.fact_extraction_batch_size == 1
    assert settings.fact_extraction_max_preceding_chunks == 1
    assert settings.fact_extraction_max_following_chunks == 1
    assert settings.fact_extraction.max_output_tokens == 32_768
    assert settings.relation_max_candidates_per_fact == 5
    assert settings.relation_min_similarity == 0.5
    assert settings.relation_pool_multiplier == 2
    assert settings.relation_rrf_k == 60
    assert settings.evidence_relation.max_output_tokens == 4096
    assert settings.candidate_critique.max_output_tokens == 2048
    assert settings.candidate_evolution.max_output_tokens == 2048
    assert settings.verification.max_output_tokens == 1024


def test_synthesis_settings_reject_unknown_schema(tmp_path: Path) -> None:
    """A config written for another schema cannot silently start a run."""
    config_path = tmp_path / "synthesis.json"
    config_path.write_text(json.dumps({"schema_version": "2.0.0"}), encoding="utf-8")

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
