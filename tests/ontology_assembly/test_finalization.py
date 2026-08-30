"""Tests for deterministic assembly finalization."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from legal_rag.benchmark_synthesis.document_metadata import DocumentMetadataRecord
from legal_rag.chunker.models import Chunk
from legal_rag.ontology_population.assembly_finalization import finalize_assemblies
from legal_rag.ontology_population.assembly_models import AssemblyProposal
from legal_rag.ontology_population.assembly_writer import read_assembly_proposals, write_assembly_proposals
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
        rationale_vi="Cùng một quy tắc.",
        run_id="run-1",
    )


def test_validated_assembly_replaces_members_and_passes_shacl(
    tmp_path: Path,
    assembly_chunks: tuple[Chunk, Chunk],
    assembly_records: tuple[OntologyPopulationRecord, ...],
    assembly_metadata: DocumentMetadataRecord,
) -> None:
    """A valid proposal yields one persisted and SHACL-valid canonical rule."""
    proposal = _proposal(assembly_records)
    canonical = finalize_assemblies(assembly_records, (proposal,))
    graph = materialize_population_graph(
        assembly_chunks,
        assembly_metadata,
        canonical,
        "https://w3id.org/legal-ontology-rag/data/",
    )
    proposals_path = tmp_path / "proposals.jsonl"
    write_assembly_proposals(proposals_path, (proposal,))

    assert [record.population_id for record in canonical] == ["population-assembled"]
    assert validate_population_graph(graph, OntologyPaths(PROJECT_ROOT))
    assert read_assembly_proposals(proposals_path) == (proposal,)


def test_no_proposal_preserves_original_records(
    assembly_records: tuple[OntologyPopulationRecord, ...],
) -> None:
    """An empty proposal set leaves canonical population records unchanged."""
    assert finalize_assemblies(assembly_records, ()) == assembly_records


def test_duplicate_proposal_ids_are_rejected(
    assembly_records: tuple[OntologyPopulationRecord, ...],
) -> None:
    """Proposal identifiers remain unique for deterministic persistence."""
    proposal = _proposal(assembly_records)

    with pytest.raises(ValueError, match="identifiers must be unique"):
        finalize_assemblies(assembly_records, (proposal, proposal))


def test_unknown_members_are_rejected(
    assembly_records: tuple[OntologyPopulationRecord, ...],
) -> None:
    """A proposal cannot replace a population record outside current input."""
    proposal = replace(_proposal(assembly_records), member_population_ids=("unknown",))

    with pytest.raises(ValueError, match="current population records"):
        finalize_assemblies(assembly_records, (proposal,))


def test_overlapping_proposals_are_rejected(
    assembly_records: tuple[OntologyPopulationRecord, ...],
) -> None:
    """One source population record cannot enter two canonical assemblies."""
    first = _proposal(assembly_records)
    second = replace(first, proposal_id="proposal-2", member_population_ids=(assembly_records[0].population_id,))

    with pytest.raises(ValueError, match="must not overlap"):
        finalize_assemblies(assembly_records, (first, second))
