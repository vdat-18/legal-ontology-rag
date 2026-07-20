"""Offline tests for the ground-truth generation service API."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import cast

import pytest
from ragas.dataset_schema import SingleTurnSample
from ragas.testset import TestsetGenerator as RagasTestsetGenerator
from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
from ragas.testset.synthesizers.testset_schema import Testset as RagasTestset
from ragas.testset.synthesizers.testset_schema import TestsetSample as RagasTestsetSample

from legal_rag.config import Settings
from legal_rag.errors import InvalidRagasComponentError, MissingCredentialError, UnsupportedRagasProviderError
from legal_rag.evaluation.groundtruth import build_ragas_generation_components, generate_groundtruth_candidates
from legal_rag.ingestion import LegalEpisode

_TEST_NEO4J_VALUE = "not-sensitive-test-value"


def _settings(tmp_path: Path) -> Settings:
    """Build settings for offline ground-truth tests."""
    return Settings(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password=_TEST_NEO4J_VALUE,
        anthropic_api_key=None,
        openai_api_key=None,
        llm_model="graphiti-llm",
        small_llm_model="graphiti-small-llm",
        embedder="openai",
        corpus_dir=tmp_path / "raw",
        corpus_zip=tmp_path / "raw.zip",
        penalty_config=tmp_path / "penalty.json",
        groundtruth_config=tmp_path / "groundtruth.json",
        ragas_llm_provider=None,
        ragas_llm_model=None,
        ragas_embedding_provider=None,
        ragas_embedding_model=None,
        ragas_candidates_output=tmp_path / "evaluation" / "candidates.jsonl",
        group_id="traffic_penalty_vn",
    )


def _episode() -> LegalEpisode:
    """Build one deterministic legal source episode."""
    return LegalEpisode(
        name="168/2024/ND-CP - Article 10",
        body="Article 10 source text",
        source_description="Decree 168/2024/ND-CP",
        reference_time=datetime(2025, 1, 1, tzinfo=UTC),
    )


def _testset() -> RagasTestset:
    """Build one deterministic Ragas testset."""
    return RagasTestset(
        samples=[
            RagasTestsetSample(
                eval_sample=SingleTurnSample(
                    user_input="What penalty applies?",
                    reference="The source defines the applicable penalty.",
                    reference_contexts=["Article 10 source text"],
                ),
                synthesizer_name="single_hop_specific_query_synthesizer",
            )
        ]
    )


def test_generate_groundtruth_candidates_writes_pending_jsonl(monkeypatch, tmp_path: Path) -> None:
    """The service reads the configured corpus and writes generated review candidates."""
    settings = _settings(tmp_path)
    captured: dict[str, object] = {}

    def fake_iter_episodes_from_config(
        settings_arg: Settings,
        selection_path: Path,
        *,
        archive_only: bool,
        skip_missing: bool,
    ):
        """Capture source-loading options without reading real DOCX files."""
        captured["settings"] = settings_arg
        captured["selection_path"] = selection_path
        captured["archive_only"] = archive_only
        captured["skip_missing"] = skip_missing
        return iter([_episode()])

    class FakeGenerator:
        """Return a fixed testset without calling any model provider."""

        def generate_with_chunks(self, documents, *, testset_size, transforms, query_distribution):
            """Capture explicit Ragas inputs and return deterministic output."""
            captured["documents"] = documents
            captured["testset_size"] = testset_size
            captured["query_distribution"] = query_distribution
            return _testset()

    monkeypatch.setattr(
        "legal_rag.evaluation.groundtruth.iter_episodes_from_config",
        fake_iter_episodes_from_config,
    )
    query_synthesizer = cast(SingleHopSpecificQuerySynthesizer, object())

    candidates = generate_groundtruth_candidates(
        settings,
        testset_size=1,
        as_of_date=date(2025, 2, 1),
        generator=cast(RagasTestsetGenerator, FakeGenerator()),
        query_synthesizer=query_synthesizer,
    )

    assert len(candidates) == 1
    assert candidates[0].approval_status == "pending"
    assert candidates[0].as_of_date == date(2025, 2, 1)
    assert settings.ragas_candidates_output.exists()
    assert captured["selection_path"] == settings.groundtruth_config
    assert captured["archive_only"] is True
    assert captured["skip_missing"] is False
    assert captured["testset_size"] == 1
    assert captured["query_distribution"] == [(query_synthesizer, 1.0)]


def test_generate_groundtruth_candidates_requires_component_pair(tmp_path: Path) -> None:
    """Injected Ragas components must be provided together."""
    with pytest.raises(InvalidRagasComponentError):
        generate_groundtruth_candidates(
            _settings(tmp_path),
            testset_size=1,
            generator=cast(RagasTestsetGenerator, object()),
        )


def test_build_ragas_generation_components_requires_provider(tmp_path: Path) -> None:
    """Ragas provider settings must be explicit before live generation."""
    with pytest.raises(MissingCredentialError):
        build_ragas_generation_components(_settings(tmp_path))


def test_build_ragas_generation_components_rejects_unknown_provider(tmp_path: Path) -> None:
    """Only configured provider integrations are accepted."""
    settings = replace(
        _settings(tmp_path),
        openai_api_key="test-key",
        ragas_llm_provider="unknown",
        ragas_llm_model="example-llm",
        ragas_embedding_provider="openai",
        ragas_embedding_model="example-embedding",
    )

    with pytest.raises(UnsupportedRagasProviderError):
        build_ragas_generation_components(settings)
