"""Tests for mandatory human review and canonical assembly finalization."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from legal_rag.benchmark_synthesis.document_metadata import DocumentMetadataRecord
from legal_rag.chunker.models import Chunk
from legal_rag.ontology_population.assembly_config import AssemblySettings
from legal_rag.ontology_population.assembly_finalization import finalize_assemblies
from legal_rag.ontology_population.assembly_models import AssemblyProposal, AssemblyReview
from legal_rag.ontology_population.assembly_writer import (
    pending_review_template,
    read_assembly_proposals,
    read_assembly_reviews,
    write_assembly_proposals,
    write_assembly_reviews,
)
from legal_rag.ontology_population.materializer import materialize_population_graph, validate_population_graph
from legal_rag.ontology_population.models import OntologyPopulationRecord
from legal_rag.ontology_validation.paths import OntologyPaths

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _proposal(records: tuple[OntologyPopulationRecord, ...]) -> AssemblyProposal:
    """Return one deterministic proposal replacing both source records."""
    assembled = replace(
        records[0],
        population_id="population-assembled",
        consequences=records[0].consequences + records[1].consequences,
        evidence=records[0].evidence + records[1].evidence,
    )
    return AssemblyProposal(
        proposal_id="proposal-1",
        candidate_id="candidate-1",
        member_population_ids=tuple(record.population_id for record in records),
        assembled_record=assembled,
        rationale_vi="Cùng một rule.",
        proposed_by_model="test-model",
        prompt_version="0.1.0",
        run_id="run-1",
    )


def _review(proposal_id: str, decision: str = "accepted") -> AssemblyReview:
    """Return one resolved test review."""
    return AssemblyReview(
        proposal_id=proposal_id,
        decision=decision,
        reviewer="legal-reviewer",
        reviewed_on=date(2026, 8, 18),
        note="Đã đối chiếu exact evidence.",
    )


def test_accepted_assembly_replaces_members_and_passes_shacl(
    tmp_path: Path,
    assembly_chunks: tuple[Chunk, Chunk],
    assembly_records: tuple[OntologyPopulationRecord, ...],
    assembly_metadata: DocumentMetadataRecord,
) -> None:
    """Human acceptance yields one persisted and SHACL-valid canonical rule."""
    settings = AssemblySettings.from_path(PROJECT_ROOT / "ontology" / "assembly_contract.yaml")
    proposal = _proposal(assembly_records)
    canonical = finalize_assemblies(assembly_records, (proposal,), (_review(proposal.proposal_id),), settings)
    graph = materialize_population_graph(
        assembly_chunks,
        assembly_metadata,
        canonical,
        "https://w3id.org/legal-ontology-rag/data/",
    )
    proposals_path = tmp_path / "proposals.jsonl"
    reviews_path = tmp_path / "reviews.jsonl"
    write_assembly_proposals(proposals_path, (proposal,))
    write_assembly_reviews(reviews_path, (_review(proposal.proposal_id),))

    assert [record.population_id for record in canonical] == ["population-assembled"]
    assert validate_population_graph(graph, OntologyPaths(PROJECT_ROOT))
    assert read_assembly_proposals(proposals_path) == (proposal,)
    assert read_assembly_reviews(reviews_path) == (_review(proposal.proposal_id),)


def test_rejected_assembly_preserves_original_records(
    assembly_records: tuple[OntologyPopulationRecord, ...],
) -> None:
    """A rejected proposal never changes canonical population records."""
    settings = AssemblySettings.from_path(PROJECT_ROOT / "ontology" / "assembly_contract.yaml")
    proposal = _proposal(assembly_records)

    assert (
        finalize_assemblies(assembly_records, (proposal,), (_review(proposal.proposal_id, "rejected"),), settings)
        == assembly_records
    )


@pytest.mark.parametrize(
    ("reviews", "message"),
    [
        ((), "requires a human review"),
        ((AssemblyReview("proposal-1", "pending", "", None, ""),), "Pending assembly reviews"),
        (
            (AssemblyReview("proposal-1", "accepted", "", date(2026, 8, 18), ""),),
            "require reviewer identity",
        ),
        (
            (AssemblyReview("unknown", "accepted", "reviewer", date(2026, 8, 18), ""),),
            "must reference known proposals",
        ),
    ],
)
def test_incomplete_or_invalid_review_blocks_finalization(
    reviews: tuple[AssemblyReview, ...],
    message: str,
    assembly_records: tuple[OntologyPopulationRecord, ...],
) -> None:
    """Canonical output requires a complete attributable decision for every proposal."""
    settings = AssemblySettings.from_path(PROJECT_ROOT / "ontology" / "assembly_contract.yaml")

    with pytest.raises(ValueError, match=message):
        finalize_assemblies(assembly_records, (_proposal(assembly_records),), reviews, settings)


def test_overlapping_accepted_proposals_are_rejected(
    assembly_records: tuple[OntologyPopulationRecord, ...],
) -> None:
    """One source population record cannot enter two accepted canonical assemblies."""
    settings = AssemblySettings.from_path(PROJECT_ROOT / "ontology" / "assembly_contract.yaml")
    first = _proposal(assembly_records)
    second = replace(first, proposal_id="proposal-2", member_population_ids=(assembly_records[0].population_id,))

    with pytest.raises(ValueError, match="must not overlap"):
        finalize_assemblies(
            assembly_records,
            (first, second),
            (_review(first.proposal_id), _review(second.proposal_id)),
            settings,
        )


def test_pending_review_template_never_preapproves_model_output(
    assembly_records: tuple[OntologyPopulationRecord, ...],
) -> None:
    """Generated review rows remain pending and unattributed until a human edits them."""
    reviews = pending_review_template((_proposal(assembly_records),))

    assert reviews[0].decision == "pending"
    assert reviews[0].reviewer == ""
    assert reviews[0].reviewed_on is None
