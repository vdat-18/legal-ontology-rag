"""Configuration tests for environment-driven settings."""

from __future__ import annotations

from pathlib import Path

from legal_rag.config import Settings


def test_groundtruth_defaults_point_to_versioned_files(monkeypatch) -> None:
    """Ground-truth settings default to the committed v1 corpus and candidates path."""
    for name in (
        "GROUNDTRUTH_CONFIG",
        "RAGAS_LLM_PROVIDER",
        "RAGAS_LLM_MODEL",
        "RAGAS_EMBEDDING_PROVIDER",
        "RAGAS_EMBEDDING_MODEL",
        "RAGAS_CANDIDATES_OUTPUT",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings.from_env()

    assert settings.groundtruth_config.name == "groundtruth_corpus_v1.json"
    assert settings.ragas_candidates_output.as_posix().endswith("data/evaluation/candidates.jsonl")
    assert settings.ragas_llm_provider is None
    assert settings.ragas_llm_model is None
    assert settings.ragas_embedding_provider is None
    assert settings.ragas_embedding_model is None
    assert settings.openai_api_key is None


def test_groundtruth_settings_support_environment_overrides(monkeypatch, tmp_path: Path) -> None:
    """Ragas and ground-truth paths are configurable without source edits."""
    config_path = tmp_path / "selection.json"
    output_path = tmp_path / "candidates.jsonl"
    monkeypatch.setenv("GROUNDTRUTH_CONFIG", str(config_path))
    monkeypatch.setenv("RAGAS_LLM_PROVIDER", "openai")
    monkeypatch.setenv("RAGAS_LLM_MODEL", "example-llm")
    monkeypatch.setenv("RAGAS_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("RAGAS_EMBEDDING_MODEL", "example-embedding")
    monkeypatch.setenv("RAGAS_CANDIDATES_OUTPUT", str(output_path))
    monkeypatch.setenv("OPENAI_API_KEY", "example-key")

    settings = Settings.from_env()

    assert settings.groundtruth_config == config_path
    assert settings.ragas_llm_provider == "openai"
    assert settings.ragas_llm_model == "example-llm"
    assert settings.ragas_embedding_provider == "openai"
    assert settings.ragas_embedding_model == "example-embedding"
    assert settings.ragas_candidates_output == output_path
    assert settings.openai_api_key == "example-key"
