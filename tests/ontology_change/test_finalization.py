"""Tests for reviewed legal-change finalization and RDF materialization."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from legal_rag.ontology_change.config import ChangeResolutionSettings
from legal_rag.ontology_change.corpus import ChangeCorpus
from legal_rag.ontology_change.finalization import finalize_legal_changes
from legal_rag.ontology_change.materializer import materialize_legal_change_graph
from legal_rag.ontology_change.models import ChangeReview, LegalChangeProposal
from legal_rag.ontology_change.writer import (
    pending_change_review_template,
    read_change_proposals,
    read_change_reviews,
    read_legal_change_events,
    write_change_proposals,
    write_change_reviews,
    write_legal_change_events,
)
from legal_rag.ontology_population.config import PopulationSettings
from legal_rag.ontology_population.materializer import validate_population_graph
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
        rationale_vi="Điều 3 trực tiếp sửa đổi rule trước.",
        proposed_by_model="test-model",
        prompt_version="0.1.0",
        run_id="run-1",
    )


def _review(proposal_id: str, decision: str = "accepted") -> ChangeReview:
    """Return one attributable resolved legal-change review."""
    return ChangeReview(
        proposal_id=proposal_id,
        decision=decision,
        reviewer="legal-reviewer",
        reviewed_on=date(2026, 8, 18),
        note="Đã đối chiếu điều khoản sửa đổi và ngày hiệu lực.",
    )


def test_accepted_change_updates_rule_versions_and_passes_temporal_shacl(
    tmp_path: Path,
    change_corpus: ChangeCorpus,
    change_evidence_text: str,
) -> None:
    """Acceptance closes/opens validity and materializes one fully evidenced LegalChange."""
    settings = ChangeResolutionSettings.from_path(PROJECT_ROOT / "ontology" / "change_resolution_contract.yaml")
    records = tuple(record for document in change_corpus.documents for record in document.records)
    proposal = _proposal(change_evidence_text)
    canonical, events = finalize_legal_changes(records, (proposal,), (_review(proposal.proposal_id),), settings)
    population_settings = PopulationSettings.from_path(PROJECT_ROOT / "ontology" / "population_contract.yaml")
    graph = materialize_legal_change_graph(
        change_corpus,
        canonical,
        events,
        population_settings.instance_namespace,
    )
    proposals_path = tmp_path / "proposals.jsonl"
    reviews_path = tmp_path / "reviews.jsonl"
    events_path = tmp_path / "events.jsonl"
    write_change_proposals(proposals_path, (proposal,))
    write_change_reviews(reviews_path, (_review(proposal.proposal_id),))
    write_legal_change_events(events_path, events)

    assert canonical[0].valid_to == date(2026, 8, 15)
    assert canonical[1].valid_from == date(2026, 8, 15)
    assert canonical[0].population_id != "earlier-population"
    assert canonical[1].population_id != "later-population"
    assert events[0].earlier_population_id == canonical[0].population_id
    assert events[0].later_population_id == canonical[1].population_id
    assert validate_population_graph(graph, OntologyPaths(PROJECT_ROOT))
    assert (
        len(
            list(
                graph.query(
                    (PROJECT_ROOT / "ontology/tests/cq_queries/cq_temporal_002_change_projection.rq").read_text()
                )
            )
        )
        == 1
    )
    assert read_change_proposals(proposals_path) == (proposal,)
    assert read_change_reviews(reviews_path) == (_review(proposal.proposal_id),)
    assert read_legal_change_events(events_path) == events


def test_rejected_change_preserves_all_rule_versions(
    change_corpus: ChangeCorpus,
    change_evidence_text: str,
) -> None:
    """A rejected model proposal changes neither validity nor canonical identity."""
    settings = ChangeResolutionSettings.from_path(PROJECT_ROOT / "ontology" / "change_resolution_contract.yaml")
    records = tuple(record for document in change_corpus.documents for record in document.records)
    proposal = _proposal(change_evidence_text)

    canonical, events = finalize_legal_changes(
        records,
        (proposal,),
        (_review(proposal.proposal_id, "rejected"),),
        settings,
    )

    assert canonical == records
    assert events == ()


@pytest.mark.parametrize(
    ("reviews", "message"),
    [
        ((), "requires a human review"),
        ((ChangeReview("proposal-1", "pending", "", None, ""),), "Pending legal-change reviews"),
        (
            (ChangeReview("proposal-1", "accepted", "", date(2026, 8, 18), ""),),
            "require reviewer identity",
        ),
        (
            (ChangeReview("unknown", "accepted", "reviewer", date(2026, 8, 18), ""),),
            "must reference known proposals",
        ),
    ],
)
def test_incomplete_or_invalid_review_blocks_finalization(
    reviews: tuple[ChangeReview, ...],
    message: str,
    change_corpus: ChangeCorpus,
    change_evidence_text: str,
) -> None:
    """No canonical temporal conclusion is written without complete human decisions."""
    settings = ChangeResolutionSettings.from_path(PROJECT_ROOT / "ontology" / "change_resolution_contract.yaml")
    records = tuple(record for document in change_corpus.documents for record in document.records)

    with pytest.raises(ValueError, match=message):
        finalize_legal_changes(records, (_proposal(change_evidence_text),), reviews, settings)


def test_overlapping_accepted_changes_are_rejected(
    change_corpus: ChangeCorpus,
    change_evidence_text: str,
) -> None:
    """One earlier rule version cannot receive multiple accepted terminal changes."""
    settings = ChangeResolutionSettings.from_path(PROJECT_ROOT / "ontology" / "change_resolution_contract.yaml")
    records = tuple(record for document in change_corpus.documents for record in document.records)
    first = _proposal(change_evidence_text)
    second = replace(first, proposal_id="proposal-2", later_population_id=None, change_type="RepealOrSupersession")

    with pytest.raises(ValueError, match="must not change one earlier rule more than once"):
        finalize_legal_changes(
            records,
            (first, second),
            (_review(first.proposal_id), _review(second.proposal_id)),
            settings,
        )


def test_pending_template_never_preapproves_legal_change(change_evidence_text: str) -> None:
    """Generated review rows require an explicit attributable human decision."""
    review = pending_change_review_template((_proposal(change_evidence_text),))[0]

    assert review.decision == "pending"
    assert review.reviewer == ""
    assert review.reviewed_on is None
