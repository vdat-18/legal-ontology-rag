"""Tests for the end-to-end evidence-graph and scenario-sampling command."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from legal_rag import evidence_graph_cli
from legal_rag.benchmark_synthesis import evidence_graph
from legal_rag.benchmark_synthesis.models import EvidenceReference, LegalFact
from legal_rag.benchmark_synthesis.writer import read_evidence_relations, read_scenarios, write_legal_facts

_REPO_CONFIG_DIR = Path(__file__).resolve().parent.parent / "data" / "config" / "groundtruth"


def _fact(fact_id: str, conduct: str) -> LegalFact:
    """Build one legal fact for CLI tests."""
    return LegalFact(
        fact_id=fact_id,
        fact_content_sha256=f"content-{fact_id}",
        conduct=conduct,
        subject="Người điều khiển ô tô",
        conditions=(),
        monetary_penalty="4.000.000 đồng",
        license_points_deducted=None,
        additional_penalties=(),
        remedial_measures=(),
        valid_from=None,
        valid_to=None,
        document_number="168/2024/NĐ-CP",
        provision_path="Điều 6",
        evidence=(EvidenceReference(chunk_id=f"chunk-{fact_id}", quoted_span=""),),
    )


class _AlwaysRelatedClient:
    """Confirm every candidate pair as a fixed relation type, ignoring settings."""

    def __init__(self, settings: object, evaluation_settings: object) -> None:
        return None

    def confirm(self, fact_a: LegalFact, fact_b: LegalFact) -> evidence_graph.EvidenceRelation:
        """Return one confirmed relation for any pair."""
        return evidence_graph.EvidenceRelation(
            relation_id=f"relation-{fact_a.fact_id}-{fact_b.fact_id}",
            relation_type="conduct_to_penalty",
            from_fact_id=fact_a.fact_id,
            to_fact_id=fact_b.fact_id,
            bridge=("Điều 6",),
            confirmed_by_model="provider/test-model",
        )


class _FakeEncoderBackend:
    """Return one fixed unit vector per input text."""

    def encode(self, sentences: list[str], *, convert_to_numpy: bool, normalize_embeddings: bool) -> np.ndarray:
        """Return a fixed unit vector for every input sentence."""
        return np.array([[1.0, 0.0] for _ in sentences])


def _configure_cli(
    monkeypatch: pytest.MonkeyPatch,
    facts_path: Path,
    relations_output_path: Path,
    scenarios_output_path: Path,
    *,
    total_samples: int,
) -> None:
    """Set deterministic CLI arguments, environment, and fake LLM/embedding clients."""
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:20128/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "provider/test-model")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("EMBEDDING_PROFILE", "qwen3_0_6b")
    monkeypatch.setenv("GROUNDTRUTH_SPAN_MATCH_THRESHOLD", "0.85")
    monkeypatch.setenv("GROUNDTRUTH_BUNDLE_OVERLAP_THRESHOLD", "0.5")
    monkeypatch.setenv("GROUNDTRUTH_DEDUP_COSINE_THRESHOLD", "0.9")
    monkeypatch.setenv("GROUNDTRUTH_MAX_QUALITY_RETRIES", "2")
    monkeypatch.setenv("GROUNDTRUTH_MAX_SIMULATOR_TURNS", "8")
    monkeypatch.setattr(evidence_graph, "BamlEvidenceRelationClient", _AlwaysRelatedClient)
    from legal_rag.embedding.encoder import EmbeddingEncoder

    monkeypatch.setattr(
        EmbeddingEncoder, "from_profile", lambda profile: EmbeddingEncoder(profile, _FakeEncoderBackend())
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "legal-build-evidence-graph",
            str(facts_path),
            str(relations_output_path),
            str(scenarios_output_path),
            "--query-taxonomy",
            str(_REPO_CONFIG_DIR / "query_taxonomy.json"),
            "--personas",
            str(_REPO_CONFIG_DIR / "personas.json"),
            "--scenario-distribution",
            str(_REPO_CONFIG_DIR / "scenario_distribution.json"),
            "--total-samples",
            str(total_samples),
        ],
    )


def test_evidence_graph_cli_confirms_relations_and_samples_scenarios(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The command confirms relations between facts and samples scenarios from them."""
    facts_path = tmp_path / "legal_facts.jsonl"
    relations_path = tmp_path / "evidence_relations.jsonl"
    scenarios_path = tmp_path / "scenarios.jsonl"
    fact_a = _fact("fact-a", "Vượt đèn đỏ")
    fact_b = _fact("fact-b", "Không chấp hành hiệu lệnh đèn tín hiệu")
    write_legal_facts(facts_path, [fact_a, fact_b])
    _configure_cli(monkeypatch, facts_path, relations_path, scenarios_path, total_samples=2)

    evidence_graph_cli.main()

    relations = read_evidence_relations(relations_path)
    scenarios = read_scenarios(scenarios_path)
    assert len(relations) == 1
    assert relations[0].from_fact_id == "fact-a"
    assert relations[0].to_fact_id == "fact-b"
    assert len(scenarios) > 0
    assert all(scenario.fact_ids for scenario in scenarios)
