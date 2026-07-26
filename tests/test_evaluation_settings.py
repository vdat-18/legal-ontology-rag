"""Tests for ground-truth synthesis environment settings."""

from __future__ import annotations

from legal_rag.settings import EvaluationSettings


def test_evaluation_settings_load_from_environment() -> None:
    """Ground-truth synthesis thresholds parse from process environment variables."""
    settings = EvaluationSettings.from_environment(
        {
            "EMBEDDING_PROFILE": "qwen3_0_6b",
            "GROUNDTRUTH_CHUNK_QUALITY_THRESHOLD": "0.6",
            "GROUNDTRUTH_SPAN_MATCH_THRESHOLD": "0.85",
            "GROUNDTRUTH_BUNDLE_OVERLAP_THRESHOLD": "0.5",
            "GROUNDTRUTH_DEDUP_COSINE_THRESHOLD": "0.9",
            "GROUNDTRUTH_MAX_QUALITY_RETRIES": "2",
            "GROUNDTRUTH_MAX_SIMULATOR_TURNS": "8",
        }
    )

    assert settings.embedding_profile == "qwen3_0_6b"
    assert settings.chunk_quality_threshold == 0.6
    assert settings.span_match_threshold == 0.85
    assert settings.bundle_overlap_threshold == 0.5
    assert settings.dedup_cosine_threshold == 0.9
    assert settings.max_quality_retries == 2
    assert settings.max_simulator_turns == 8
