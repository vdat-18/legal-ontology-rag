"""End-to-end tests for the concept-resolution operator workflow."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest
from rdflib import Graph

from legal_rag import (
    ontology_concept_candidates_cli,
    ontology_concept_finalize_cli,
    ontology_concept_proposals_cli,
)
from legal_rag.baml_client.types import (
    ConceptResolutionInventoryInput,
    ConceptResolutionInventoryResult,
)
from legal_rag.ontology_change.corpus import ChangeCorpus
from legal_rag.ontology_change.writer import write_legal_change_events
from legal_rag.ontology_concept import proposals as concept_proposals
from legal_rag.ontology_concept.writer import (
    read_canonical_concepts,
    read_concept_candidates,
    read_concept_proposals,
)
from legal_rag.ontology_exception.writer import write_legal_exception_records

from .conftest import write_exception_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _CliConceptClient:
    """Return empty typed inventories and track resumable LLM calls."""

    calls = 0
    model = "test-model"

    def __init__(self, *_: object) -> None:
        """Accept the production client's constructor arguments."""

    def propose_batch(
        self,
        inventories: Sequence[ConceptResolutionInventoryInput],
    ) -> Sequence[ConceptResolutionInventoryResult]:
        """Complete each inventory without fabricating a semantic relation."""
        type(self).calls += 1
        return tuple(
            ConceptResolutionInventoryResult(
                candidate_id=inventory.candidate_id,
                proposals=[],
            )
            for inventory in inventories
        )


class _EmptyConceptClient:
    """Return completed concept-pair results with no relation proposals."""

    model = "test-model"

    def __init__(self, *_: object) -> None:
        """Accept production constructor arguments."""

    def propose_batch(
        self,
        inventories: Sequence[ConceptResolutionInventoryInput],
    ) -> Sequence[ConceptResolutionInventoryResult]:
        """Return an explicit no-proposal result for every replayed inventory."""
        return tuple(
            ConceptResolutionInventoryResult(candidate_id=inventory.candidate_id, proposals=[])
            for inventory in inventories
        )


def test_cli_runs_resumable_proposal_and_offline_finalization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    exception_corpus: ChangeCorpus,
) -> None:
    """The three commands call the fake model once per inventory and resume completed work."""
    manifest_path = tmp_path / "corpus.json"
    canonical_path = write_exception_manifest(manifest_path, exception_corpus)
    change_events_path = tmp_path / "change-events.jsonl"
    exception_records_path = tmp_path / "exceptions.jsonl"
    candidates_path = tmp_path / "candidates.json"
    proposals_path = tmp_path / "proposals.jsonl"
    progress_path = tmp_path / "progress.jsonl"
    concepts_path = tmp_path / "concepts.jsonl"
    relations_path = tmp_path / "relations.jsonl"
    graph_path = tmp_path / "canonical.ttl"
    write_legal_change_events(change_events_path, ())
    write_legal_exception_records(exception_records_path, ())

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "legal-build-concept-candidates",
            str(canonical_path),
            str(PROJECT_ROOT / "ontology" / "concept_resolution_contract.yaml"),
            str(candidates_path),
        ],
    )
    ontology_concept_candidates_cli.main()

    _CliConceptClient.calls = 0
    monkeypatch.setattr(concept_proposals, "BamlConceptProposalClient", _CliConceptClient)
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "30")
    proposal_argv = [
        "legal-propose-concept-relations",
        str(manifest_path),
        str(candidates_path),
        str(PROJECT_ROOT / "ontology" / "concept_resolution_contract.yaml"),
        str(proposals_path),
        str(progress_path),
        "--max-concurrency",
        "2",
    ]
    monkeypatch.setattr(sys, "argv", proposal_argv)
    ontology_concept_proposals_cli.main()
    ontology_concept_proposals_cli.main()

    assert read_concept_proposals(proposals_path) == ()
    assert _CliConceptClient.calls == 2
    finalize_argv = [
        "legal-finalize-concept-resolution",
        str(manifest_path),
        str(canonical_path),
        str(change_events_path),
        str(exception_records_path),
        str(candidates_path),
        str(proposals_path),
        str(PROJECT_ROOT / "ontology" / "population_contract.yaml"),
        str(PROJECT_ROOT / "ontology" / "concept_resolution_contract.yaml"),
        str(concepts_path),
        str(relations_path),
        str(graph_path),
    ]
    monkeypatch.setattr(sys, "argv", finalize_argv)
    ontology_concept_finalize_cli.main()

    assert len(read_canonical_concepts(concepts_path)) == 2
    assert len(Graph().parse(graph_path, format="turtle")) > 0

    selection_path = tmp_path / "concept-rerun.json"
    selected_id = read_concept_candidates(candidates_path).inventories[0].candidate_id
    selection_path.write_text(
        json.dumps({"schema_version": "0.1.0", "candidate_ids": [selected_id]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [*proposal_argv, "--rerun-candidate-ids-path", str(selection_path)],
    )
    ontology_concept_proposals_cli.main()

    assert _CliConceptClient.calls == 3

    progress_path.unlink()
    monkeypatch.setattr(concept_proposals, "BamlConceptProposalClient", _EmptyConceptClient)
    monkeypatch.setattr(sys, "argv", proposal_argv)
    ontology_concept_proposals_cli.main()

    assert read_concept_proposals(proposals_path) == ()
