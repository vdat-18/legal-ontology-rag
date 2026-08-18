"""Tests for reviewed exception finalization and RDF materialization."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from legal_rag.ontology_change.corpus import ChangeCorpus
from legal_rag.ontology_exception.config import ExceptionResolutionSettings
from legal_rag.ontology_exception.finalization import finalize_legal_exceptions
from legal_rag.ontology_exception.materializer import materialize_legal_exception_graph
from legal_rag.ontology_exception.models import (
    ExceptionCitationResolution,
    ExceptionReview,
    LegalExceptionProposal,
)
from legal_rag.ontology_exception.writer import (
    pending_exception_review_template,
    read_exception_proposals,
    read_exception_reviews,
    read_legal_exception_records,
    write_exception_proposals,
    write_exception_reviews,
    write_legal_exception_records,
)
from legal_rag.ontology_population.config import PopulationSettings
from legal_rag.ontology_population.materializer import validate_population_graph
from legal_rag.ontology_population.models import PopulationEvidence
from legal_rag.ontology_validation.paths import OntologyPaths

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _proposal(exception_text: str, proposal_id: str = "proposal-1") -> LegalExceptionProposal:
    """Return one exact-evidenced exception proposal."""
    return LegalExceptionProposal(
        proposal_id=proposal_id,
        candidate_id="candidate-1",
        general_population_id="general-population",
        source_document_number="168/2024/NĐ-CP",
        exception_provision_path="Điều 6 khoản 1 điểm a",
        evidence=(PopulationEvidence("general-chunk", exception_text),),
        citations=(
            ExceptionCitationResolution(
                citation_text="Điều 3",
                source_chunk_id="general-chunk",
                target_population_ids=("target-population",),
            ),
        ),
        rationale_vi="Source trực tiếp loại trừ và dẫn tới Điều 3.",
        proposed_by_model="test-model",
        prompt_version="0.1.0",
        run_id="run-1",
    )


def _review(proposal_id: str, decision: str = "accepted") -> ExceptionReview:
    """Return one attributable resolved review."""
    return ExceptionReview(
        proposal_id=proposal_id,
        decision=decision,
        reviewer="legal-reviewer",
        reviewed_on=date(2026, 8, 18),
        note="Đã đối chiếu exception, citation và target rule version.",
    )


def test_accepted_exception_materializes_grouped_citation_and_passes_shacl(
    tmp_path: Path,
    exception_corpus: ChangeCorpus,
    exception_text: str,
) -> None:
    """Acceptance creates bidirectional exception links, citation grouping, redirects, and valid RDF."""
    settings = ExceptionResolutionSettings.from_path(PROJECT_ROOT / "ontology" / "exception_resolution_contract.yaml")
    population_records = exception_corpus.documents[0].records
    proposal = _proposal(exception_text)
    exceptions = finalize_legal_exceptions(
        population_records,
        (proposal,),
        (_review(proposal.proposal_id),),
        settings,
    )
    population_settings = PopulationSettings.from_path(PROJECT_ROOT / "ontology" / "population_contract.yaml")
    graph = materialize_legal_exception_graph(
        exception_corpus,
        population_records,
        (),
        exceptions,
        population_settings.instance_namespace,
    )
    proposals_path = tmp_path / "proposals.jsonl"
    reviews_path = tmp_path / "reviews.jsonl"
    exceptions_path = tmp_path / "exceptions.jsonl"
    write_exception_proposals(proposals_path, (proposal,))
    write_exception_reviews(reviews_path, (_review(proposal.proposal_id),))
    write_legal_exception_records(exceptions_path, exceptions)

    query = (PROJECT_ROOT / "ontology/tests/cq_queries/cq_exception_001.rq").read_text(encoding="utf-8")
    rows = list(graph.query(query))
    assert len(exceptions) == 1
    assert exceptions[0].citations[0].citation_text == "Điều 3"
    assert exceptions[0].citations[0].target_population_ids == ("target-population",)
    assert validate_population_graph(graph, OntologyPaths(PROJECT_ROOT))
    assert len(rows) == 1
    assert str(rows[0].citationText) == "Điều 3"
    assert read_exception_proposals(proposals_path) == (proposal,)
    assert read_exception_reviews(reviews_path) == (_review(proposal.proposal_id),)
    assert read_legal_exception_records(exceptions_path) == exceptions


def test_accepted_fragments_merge_without_losing_citation_grouping(
    exception_corpus: ChangeCorpus,
    exception_text: str,
) -> None:
    """Fragments from exhaustive target batches merge into one canonical exception."""
    settings = ExceptionResolutionSettings.from_path(PROJECT_ROOT / "ontology" / "exception_resolution_contract.yaml")
    first = _proposal(exception_text)
    second = replace(first, proposal_id="proposal-2", candidate_id="candidate-2")
    exceptions = finalize_legal_exceptions(
        exception_corpus.documents[0].records,
        (first, second),
        (_review(first.proposal_id), _review(second.proposal_id)),
        settings,
    )

    assert len(exceptions) == 1
    assert exceptions[0].accepted_proposal_ids == ("proposal-1", "proposal-2")
    assert len(exceptions[0].citations) == 1


def test_rejected_exception_creates_no_canonical_record(
    exception_corpus: ChangeCorpus,
    exception_text: str,
) -> None:
    """A rejected model proposal never changes the canonical ontology."""
    settings = ExceptionResolutionSettings.from_path(PROJECT_ROOT / "ontology" / "exception_resolution_contract.yaml")
    proposal = _proposal(exception_text)

    assert (
        finalize_legal_exceptions(
            exception_corpus.documents[0].records,
            (proposal,),
            (_review(proposal.proposal_id, "rejected"),),
            settings,
        )
        == ()
    )


@pytest.mark.parametrize(
    ("reviews", "message"),
    [
        ((), "requires a human review"),
        ((ExceptionReview("proposal-1", "pending", "", None, ""),), "Pending exception reviews"),
        (
            (ExceptionReview("proposal-1", "accepted", "", date(2026, 8, 18), ""),),
            "require reviewer identity",
        ),
        (
            (ExceptionReview("unknown", "accepted", "reviewer", date(2026, 8, 18), ""),),
            "must reference known proposals",
        ),
    ],
)
def test_incomplete_or_invalid_review_blocks_finalization(
    reviews: tuple[ExceptionReview, ...],
    message: str,
    exception_corpus: ChangeCorpus,
    exception_text: str,
) -> None:
    """No exception semantics enter the graph without complete attributable review."""
    settings = ExceptionResolutionSettings.from_path(PROJECT_ROOT / "ontology" / "exception_resolution_contract.yaml")

    with pytest.raises(ValueError, match=message):
        finalize_legal_exceptions(
            exception_corpus.documents[0].records,
            (_proposal(exception_text),),
            reviews,
            settings,
        )


def test_pending_template_never_preapproves_exception(exception_text: str) -> None:
    """Generated review rows remain pending and unattributed."""
    review = pending_exception_review_template((_proposal(exception_text),))[0]

    assert review.decision == "pending"
    assert review.reviewer == ""
    assert review.reviewed_on is None
