"""Tests for the end-to-end evidence-graph and scenario-sampling command."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from legal_rag import evidence_graph_cli
from legal_rag.benchmark_synthesis import evidence_graph
from legal_rag.benchmark_synthesis.evidence_graph import EvidenceRelationProgressRecord
from legal_rag.benchmark_synthesis.models import EvidenceReference, LegalFact
from legal_rag.benchmark_synthesis.writer import (
    read_evidence_relation_progress,
    read_evidence_relations,
    read_scenarios,
    write_evidence_relation_progress,
    write_legal_facts,
)

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
) -> Path:
    """Set deterministic CLI arguments, environment, and fake LLM/embedding clients; return the progress path."""
    relations_progress_path = relations_output_path.parent / "evidence_relation_progress.jsonl"
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:20128/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "provider/test-model")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("EMBEDDING_PROFILE", "qwen3_0_6b")
    monkeypatch.setenv("GROUNDTRUTH_SPAN_MATCH_THRESHOLD", "0.85")
    monkeypatch.setenv("GROUNDTRUTH_BUNDLE_OVERLAP_THRESHOLD", "0.5")
    monkeypatch.setenv("GROUNDTRUTH_MAX_QUALITY_RETRIES", "2")
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
            str(relations_progress_path),
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
    return relations_progress_path


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


def test_evidence_graph_cli_does_not_recheck_a_pair_already_processed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A candidate pair recorded in the progress file is not sent to the LLM again."""
    facts_path = tmp_path / "legal_facts.jsonl"
    relations_path = tmp_path / "evidence_relations.jsonl"
    scenarios_path = tmp_path / "scenarios.jsonl"
    fact_a = _fact("fact-a", "Vượt đèn đỏ")
    fact_b = _fact("fact-b", "Không chấp hành hiệu lệnh đèn tín hiệu")
    fact_c = _fact("fact-c", "Không đội mũ bảo hiểm")
    write_legal_facts(facts_path, [fact_a, fact_b, fact_c])
    progress_path = _configure_cli(monkeypatch, facts_path, relations_path, scenarios_path, total_samples=2)
    write_evidence_relation_progress(
        progress_path, [EvidenceRelationProgressRecord(fact_id_a="fact-a", fact_id_b="fact-b")]
    )
    checked_pairs: list[tuple[str, str]] = []

    class _RecordingClient:
        """Confirm any pair as related, recording every pair it was asked about."""

        def __init__(self, settings: object, evaluation_settings: object) -> None:
            return None

        def confirm(self, fact_x: LegalFact, fact_y: LegalFact) -> evidence_graph.EvidenceRelation:
            """Record the pair and return one confirmed relation for it."""
            checked_pairs.append(tuple(sorted((fact_x.fact_id, fact_y.fact_id))))
            return evidence_graph.EvidenceRelation(
                relation_id=f"relation-{fact_x.fact_id}-{fact_y.fact_id}",
                relation_type="conduct_to_penalty",
                from_fact_id=fact_x.fact_id,
                to_fact_id=fact_y.fact_id,
                bridge=(),
                confirmed_by_model="provider/test-model",
            )

    monkeypatch.setattr(evidence_graph, "BamlEvidenceRelationClient", _RecordingClient)

    evidence_graph_cli.main()

    assert ("fact-a", "fact-b") not in checked_pairs
    assert set(checked_pairs) == {("fact-a", "fact-c"), ("fact-b", "fact-c")}


def test_evidence_graph_cli_persists_progress_if_interrupted_mid_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A pair already confirmed is durably written even if the process dies before the run finishes."""
    facts_path = tmp_path / "legal_facts.jsonl"
    relations_path = tmp_path / "evidence_relations.jsonl"
    scenarios_path = tmp_path / "scenarios.jsonl"
    fact_a = _fact("fact-a", "Vượt đèn đỏ")
    fact_b = _fact("fact-b", "Không chấp hành hiệu lệnh đèn tín hiệu")
    fact_c = _fact("fact-c", "Không đội mũ bảo hiểm")
    write_legal_facts(facts_path, [fact_a, fact_b, fact_c])
    progress_path = _configure_cli(monkeypatch, facts_path, relations_path, scenarios_path, total_samples=2)

    class _CrashesOnSecondPairClient:
        """Succeed on the first candidate pair, then simulate the whole process being killed."""

        def __init__(self, settings: object, evaluation_settings: object) -> None:
            self._calls = 0

        def confirm(self, fact_x: LegalFact, fact_y: LegalFact) -> evidence_graph.EvidenceRelation:
            """Return one confirmed relation on the first call, then raise an uncaught fatal error."""
            self._calls += 1
            if self._calls > 1:
                raise SystemExit
            return evidence_graph.EvidenceRelation(
                relation_id=f"relation-{fact_x.fact_id}-{fact_y.fact_id}",
                relation_type="conduct_to_penalty",
                from_fact_id=fact_x.fact_id,
                to_fact_id=fact_y.fact_id,
                bridge=(),
                confirmed_by_model="provider/test-model",
            )

    monkeypatch.setattr(evidence_graph, "BamlEvidenceRelationClient", _CrashesOnSecondPairClient)

    with pytest.raises(SystemExit):
        evidence_graph_cli.main()

    assert len(read_evidence_relation_progress(progress_path)) == 1
    assert len(read_evidence_relations(relations_path)) == 1


def test_evidence_graph_cli_does_not_recheck_a_pair_confirmed_unrelated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A pair the model found unrelated produces no relation, but is still recorded as checked and never resent."""
    facts_path = tmp_path / "legal_facts.jsonl"
    relations_path = tmp_path / "evidence_relations.jsonl"
    scenarios_path = tmp_path / "scenarios.jsonl"
    fact_a = _fact("fact-a", "Vượt đèn đỏ")
    fact_b = _fact("fact-b", "Không chấp hành hiệu lệnh đèn tín hiệu")
    write_legal_facts(facts_path, [fact_a, fact_b])
    progress_path = _configure_cli(monkeypatch, facts_path, relations_path, scenarios_path, total_samples=1)

    class _UnrelatedClient:
        """Confirm that no candidate pair is related."""

        def __init__(self, settings: object, evaluation_settings: object) -> None:
            return None

        def confirm(self, fact_x: LegalFact, fact_y: LegalFact) -> None:
            """Return None for every pair."""

    monkeypatch.setattr(evidence_graph, "BamlEvidenceRelationClient", _UnrelatedClient)

    evidence_graph_cli.main()

    assert read_evidence_relations(relations_path) == ()
    assert read_evidence_relation_progress(progress_path) == (
        EvidenceRelationProgressRecord(fact_id_a="fact-a", fact_id_b="fact-b"),
    )

    def _fail(*args: object, **kwargs: object) -> None:
        msg = "BamlEvidenceRelationClient must not be constructed when every pair is resumed"
        raise AssertionError(msg)

    monkeypatch.setattr(evidence_graph, "BamlEvidenceRelationClient", _fail)

    evidence_graph_cli.main()
