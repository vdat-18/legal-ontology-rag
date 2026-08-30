"""Tests for exception finalization and RDF materialization."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from legal_rag.ontology_change.corpus import ChangeCorpus
from legal_rag.ontology_exception.finalization import finalize_legal_exceptions
from legal_rag.ontology_exception.materializer import materialize_legal_exception_graph
from legal_rag.ontology_exception.models import ExceptionCitationResolution, LegalExceptionProposal
from legal_rag.ontology_exception.writer import (
    read_exception_proposals,
    read_legal_exception_records,
    write_exception_proposals,
    write_legal_exception_records,
)
from legal_rag.ontology_population.config import PopulationSettings
from legal_rag.ontology_population.materializer import LORNR, LORSP, validate_population_graph
from legal_rag.ontology_population.models import PopulationEvidence
from legal_rag.ontology_validation.paths import OntologyPaths

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _proposal(exception_text: str, proposal_id: str = "proposal-1") -> LegalExceptionProposal:
    """Return one source-grounded exception proposal."""
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
        rationale_vi="Nguồn trực tiếp loại trừ và dẫn tới Điều 3.",
        run_id="run-1",
    )


def test_validated_exception_materializes_grouped_citation_and_passes_shacl(
    tmp_path: Path,
    exception_corpus: ChangeCorpus,
    exception_text: str,
) -> None:
    """A valid proposal creates exception links, citation grouping and source provenance."""
    population_records = exception_corpus.documents[0].records
    proposal = _proposal(exception_text)
    exceptions = finalize_legal_exceptions(population_records, (proposal,))
    population_settings = PopulationSettings.from_path(PROJECT_ROOT / "ontology" / "population_contract.yaml")
    graph = materialize_legal_exception_graph(
        exception_corpus,
        population_records,
        (),
        exceptions,
        population_settings.instance_namespace,
    )
    proposals_path = tmp_path / "proposals.jsonl"
    exceptions_path = tmp_path / "exceptions.jsonl"
    write_exception_proposals(proposals_path, (proposal,))
    write_legal_exception_records(exceptions_path, exceptions)

    query = (PROJECT_ROOT / "ontology/tests/cq_queries/cq_exception_001.rq").read_text(encoding="utf-8")
    rows = list(graph.query(query))
    exception_iri = next(graph.subjects(predicate=LORNR.exceptionTo))
    assert len(exceptions) == 1
    assert exceptions[0].citations[0].target_population_ids == ("target-population",)
    assert tuple(graph.objects(exception_iri, LORSP.sourcedFromChunk))
    assert validate_population_graph(graph, OntologyPaths(PROJECT_ROOT))
    assert len(rows) == 1
    assert str(rows[0].citationText) == "Điều 3"
    assert read_exception_proposals(proposals_path) == (proposal,)
    assert read_legal_exception_records(exceptions_path) == exceptions


def test_fragments_merge_without_losing_citation_grouping(
    exception_corpus: ChangeCorpus,
    exception_text: str,
) -> None:
    """Fragments from exhaustive target batches merge into one canonical exception."""
    first = _proposal(exception_text)
    second = replace(first, proposal_id="proposal-2", candidate_id="candidate-2")
    exceptions = finalize_legal_exceptions(exception_corpus.documents[0].records, (first, second))

    assert len(exceptions) == 1
    assert len(exceptions[0].citations) == 1


def test_no_proposal_creates_no_exception_record(exception_corpus: ChangeCorpus) -> None:
    """An empty proposal set leaves the canonical ontology unchanged."""
    assert finalize_legal_exceptions(exception_corpus.documents[0].records, ()) == ()


def test_duplicate_proposal_ids_are_rejected(
    exception_corpus: ChangeCorpus,
    exception_text: str,
) -> None:
    """Proposal identifiers remain unique for deterministic persistence."""
    proposal = _proposal(exception_text)

    with pytest.raises(ValueError, match="identifiers must be unique"):
        finalize_legal_exceptions(exception_corpus.documents[0].records, (proposal, proposal))


def test_unknown_redirect_target_is_rejected(
    exception_corpus: ChangeCorpus,
    exception_text: str,
) -> None:
    """An exception cannot redirect to a rule outside the canonical corpus."""
    proposal = _proposal(exception_text)
    citation = replace(proposal.citations[0], target_population_ids=("unknown",))

    with pytest.raises(ValueError, match="canonical target rule versions"):
        finalize_legal_exceptions(
            exception_corpus.documents[0].records,
            (replace(proposal, citations=(citation,)),),
        )
