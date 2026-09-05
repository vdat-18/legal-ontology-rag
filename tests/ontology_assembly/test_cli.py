"""End-to-end tests for the assembly operator workflow."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import pytest
from rdflib import Graph

from legal_rag import (
    ontology_assembly_candidates_cli,
    ontology_assembly_finalize_cli,
    ontology_assembly_proposals_cli,
)
from legal_rag.baml_client.types import (
    OntologyConsequenceEntry,
    OntologyConsequenceKind,
    OntologyCurrency,
    OntologyEvidenceReference,
    OntologyRuleEntry,
    OntologySubjectKind,
    OntologyValidityMode,
    RuleAssemblyProposal,
    RuleAssemblyWindowInput,
    RuleAssemblyWindowResult,
)
from legal_rag.benchmark_synthesis.document_metadata import DocumentMetadataRecord
from legal_rag.benchmark_synthesis.writer import write_document_metadata
from legal_rag.chunker.models import Chunk
from legal_rag.chunker.writer import write_chunks
from legal_rag.ontology_population import assembly_proposals
from legal_rag.ontology_population.assembly_writer import read_assembly_proposals
from legal_rag.ontology_population.models import OntologyPopulationRecord
from legal_rag.ontology_population.writer import read_population_records, write_population_records

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _CliAssemblyClient:
    """Return one typed proposal and record how often the resumable CLI calls it."""

    calls = 0
    model = "test-model"

    def __init__(self, *_: object) -> None:
        """Accept production constructor arguments."""

    def propose_batch(self, windows: Sequence[RuleAssemblyWindowInput]) -> Sequence[RuleAssemblyWindowResult]:
        """Assemble the two fixture records using their exact summarized evidence."""
        type(self).calls += 1
        results = []
        for window in windows:
            first, second = window.records
            results.append(
                RuleAssemblyWindowResult(
                    candidate_id=window.candidate_id,
                    proposals=[
                        RuleAssemblyProposal(
                            member_population_ids=[first.population_id, second.population_id],
                            assembled_rule=OntologyRuleEntry(
                                conduct_label_vi=first.conduct_label_vi,
                                subject_kind=OntologySubjectKind.NaturalPerson,
                                role_labels_vi=first.role_labels_vi,
                                vehicle_category_labels_vi=first.vehicle_category_labels_vi,
                                condition_root_id=None,
                                conditions=[],
                                consequences=[
                                    OntologyConsequenceEntry(
                                        consequence_id=item.consequence_id,
                                        kind=OntologyConsequenceKind(item.kind),
                                        minimum_amount=item.minimum_amount,
                                        maximum_amount=item.maximum_amount,
                                        currency=OntologyCurrency(item.currency) if item.currency else None,
                                        point_value=item.point_value,
                                        target_label_vi=item.target_label_vi,
                                        responsible_role_label_vi=item.responsible_role_label_vi,
                                        evidence=[
                                            OntologyEvidenceReference(evidence_id=source.evidence_id)
                                            for source in item.evidence
                                        ],
                                    )
                                    for record in window.records
                                    for item in record.consequences
                                ],
                                validity_mode=OntologyValidityMode.ExplicitInterval,
                                valid_from=first.valid_from,
                                valid_to=first.valid_to,
                                provision_path=first.provision_path,
                                evidence=[
                                    OntologyEvidenceReference(evidence_id=source.evidence_id)
                                    for record in window.records
                                    for source in record.evidence
                                ],
                            ),
                            rationale_vi="Cùng hành vi và điều khoản; chunk sau tiếp tục hậu quả pháp lý.",
                        )
                    ],
                )
            )
        return tuple(results)


class _EmptyAssemblyClient:
    """Return a completed candidate result with no assembly proposal."""

    model = "test-model"

    def __init__(self, *_: object) -> None:
        """Accept production constructor arguments."""

    def propose_batch(self, windows: Sequence[RuleAssemblyWindowInput]) -> Sequence[RuleAssemblyWindowResult]:
        """Return an explicit no-proposal result for every replayed window."""
        return tuple(RuleAssemblyWindowResult(candidate_id=window.candidate_id, proposals=[]) for window in windows)


def test_cli_runs_candidate_proposal_and_offline_finalization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    assembly_chunks: tuple[Chunk, Chunk],
    assembly_records: tuple[OntologyPopulationRecord, ...],
    assembly_metadata: DocumentMetadataRecord,
) -> None:
    """The workflow resumes LLM work and deterministically materializes RDF."""
    chunks_path = tmp_path / "chunks.jsonl"
    metadata_path = tmp_path / "metadata.jsonl"
    records_path = tmp_path / "population.jsonl"
    candidates_path = tmp_path / "candidates.json"
    proposals_path = tmp_path / "proposals.jsonl"
    progress_path = tmp_path / "proposal-progress.jsonl"
    canonical_path = tmp_path / "canonical.jsonl"
    graph_path = tmp_path / "canonical.ttl"
    write_chunks(chunks_path, assembly_chunks)
    write_document_metadata(metadata_path, assembly_metadata)
    write_population_records(records_path, assembly_records)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "legal-build-assembly-candidates",
            str(chunks_path),
            str(records_path),
            str(PROJECT_ROOT / "ontology" / "assembly_contract.yaml"),
            str(candidates_path),
        ],
    )
    ontology_assembly_candidates_cli.main()

    _CliAssemblyClient.calls = 0
    monkeypatch.setattr(assembly_proposals, "BamlAssemblyProposalClient", _CliAssemblyClient)
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "30")
    proposal_argv = [
        "legal-propose-rule-assemblies",
        str(chunks_path),
        str(metadata_path),
        str(records_path),
        str(candidates_path),
        str(PROJECT_ROOT / "ontology" / "assembly_contract.yaml"),
        str(proposals_path),
        str(progress_path),
        "--max-concurrency",
        "2",
    ]
    monkeypatch.setattr(sys, "argv", proposal_argv)
    ontology_assembly_proposals_cli.main()
    ontology_assembly_proposals_cli.main()

    assert len(read_assembly_proposals(proposals_path)) == 1
    assert _CliAssemblyClient.calls == 1
    finalize_argv = [
        "legal-finalize-rule-assemblies",
        str(chunks_path),
        str(metadata_path),
        str(records_path),
        str(proposals_path),
        str(PROJECT_ROOT / "ontology" / "population_contract.yaml"),
        str(canonical_path),
        str(graph_path),
    ]
    monkeypatch.setattr(sys, "argv", finalize_argv)
    ontology_assembly_finalize_cli.main()

    assert len(read_population_records(canonical_path)) == 1
    assert len(Graph().parse(graph_path, format="turtle")) > 0

    progress_path.unlink()
    monkeypatch.setattr(assembly_proposals, "BamlAssemblyProposalClient", _EmptyAssemblyClient)
    monkeypatch.setattr(sys, "argv", proposal_argv)
    ontology_assembly_proposals_cli.main()

    assert read_assembly_proposals(proposals_path) == ()
