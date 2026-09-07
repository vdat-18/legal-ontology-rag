"""Tests for legal-change finalization and RDF materialization."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from legal_rag.ontology_change.corpus import ChangeCorpus
from legal_rag.ontology_change.finalization import finalize_legal_changes
from legal_rag.ontology_change.materializer import materialize_legal_change_graph
from legal_rag.ontology_change.models import LegalChangeProposal
from legal_rag.ontology_change.writer import (
    read_change_proposals,
    read_legal_change_events,
    write_change_proposals,
    write_legal_change_events,
)
from legal_rag.ontology_population.config import PopulationSettings
from legal_rag.ontology_population.materializer import LORSP, LORTC, validate_population_graph
from legal_rag.ontology_population.models import PopulationEvidence
from legal_rag.ontology_validation.paths import OntologyPaths

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _proposal(change_evidence_text: str) -> LegalChangeProposal:
    """Return one explicit amendment proposal for the two fixture rules."""
    return LegalChangeProposal(
        proposal_id="proposal-1",
        candidate_id="candidate-1",
        earlier_population_id="earlier-population",
        later_population_id="later-population",
        change_type="Amendment",
        effective_on=date(2026, 8, 15),
        change_document_number="238/2026/NĐ-CP",
        change_provision_path="Điều 3",
        evidence=(PopulationEvidence("later-chunk", change_evidence_text),),
        rationale_vi="Điều 3 trực tiếp sửa đổi quy tắc trước.",
        run_id="run-1",
    )


def test_validated_change_updates_rule_versions_and_passes_temporal_shacl(
    tmp_path: Path,
    change_corpus: ChangeCorpus,
    change_evidence_text: str,
) -> None:
    """A valid change closes and opens validity and materializes one change event."""
    records = tuple(record for document in change_corpus.documents for record in document.records)
    proposal = _proposal(change_evidence_text)
    canonical, events = finalize_legal_changes(records, (proposal,))
    population_settings = PopulationSettings.from_path(PROJECT_ROOT / "ontology" / "population_contract.yaml")
    graph = materialize_legal_change_graph(
        change_corpus,
        canonical,
        events,
        population_settings.instance_namespace,
    )
    proposals_path = tmp_path / "proposals.jsonl"
    events_path = tmp_path / "events.jsonl"
    write_change_proposals(proposals_path, (proposal,))
    write_legal_change_events(events_path, events)

    assert canonical[0].valid_to == date(2026, 8, 15)
    assert canonical[1].valid_from == date(2026, 8, 15)
    assert events[0].earlier_population_id == canonical[0].population_id
    assert events[0].later_population_id == canonical[1].population_id
    assert validate_population_graph(graph, OntologyPaths(PROJECT_ROOT))
    change = next(graph.subjects(predicate=LORTC.changeType))
    assert tuple(graph.objects(change, LORSP.sourcedFromChunk))
    assert read_change_proposals(proposals_path) == (proposal,)
    assert read_legal_change_events(events_path) == events


def test_no_change_preserves_all_rule_versions(change_corpus: ChangeCorpus) -> None:
    """An empty proposal set changes neither validity nor canonical identity."""
    records = tuple(record for document in change_corpus.documents for record in document.records)
    assert finalize_legal_changes(records, ()) == (records, ())


def test_duplicate_proposal_ids_are_rejected(
    change_corpus: ChangeCorpus,
    change_evidence_text: str,
) -> None:
    """Proposal identifiers remain unique for deterministic persistence."""
    records = tuple(record for document in change_corpus.documents for record in document.records)
    proposal = _proposal(change_evidence_text)

    with pytest.raises(ValueError, match="identifiers must be unique"):
        finalize_legal_changes(records, (proposal, proposal))


def test_unknown_rule_members_are_rejected(
    change_corpus: ChangeCorpus,
    change_evidence_text: str,
) -> None:
    """A change cannot target rule versions outside the current corpus."""
    records = tuple(record for document in change_corpus.documents for record in document.records)
    proposal = replace(_proposal(change_evidence_text), earlier_population_id="unknown")

    with pytest.raises(ValueError, match="current canonical population records"):
        finalize_legal_changes(records, (proposal,))


def test_one_rule_can_participate_in_multiple_change_links(
    change_corpus: ChangeCorpus,
    change_evidence_text: str,
) -> None:
    """Rule splits and merges can create multiple links while sharing validity."""
    records = tuple(record for document in change_corpus.documents for record in document.records)
    first = _proposal(change_evidence_text)
    second = replace(first, proposal_id="proposal-2", later_population_id=None, change_type="Amendment")

    canonical, events = finalize_legal_changes(records, (first, second))

    assert canonical[0].valid_to == date(2026, 8, 15)
    assert canonical[1].valid_from == date(2026, 8, 15)
    assert len(events) == 2
    assert {event.earlier_population_id for event in events} == {canonical[0].population_id}
