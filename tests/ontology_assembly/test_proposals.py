"""Tests for typed LLM assembly proposals and source binding."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from legal_rag.baml_client.types import (
    OntologyConsequenceEntry,
    OntologyConsequenceKind,
    OntologyCurrency,
    OntologyEvidenceReference,
    OntologyRuleEntry,
    OntologySubjectKind,
    RuleAssemblyProposal,
    RuleAssemblyWindowInput,
    RuleAssemblyWindowResult,
)
from legal_rag.benchmark_synthesis.document_metadata import DocumentMetadataRecord
from legal_rag.benchmark_synthesis.llm_execution import LlmTaskExhaustedError
from legal_rag.chunker.models import Chunk
from legal_rag.ontology_population.assembly_candidates import build_assembly_candidates
from legal_rag.ontology_population.assembly_config import AssemblySettings
from legal_rag.ontology_population.assembly_proposals import propose_assemblies, summarize_population_record
from legal_rag.ontology_population.models import OntologyPopulationRecord

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _AssemblyClient:
    """Return configurable typed assembly results without an external model."""

    model = "test-model"

    def __init__(self, member_ids: Sequence[str], *, empty: bool = False) -> None:
        self._member_ids = list(member_ids)
        self._empty = empty
        self.inputs: tuple[RuleAssemblyWindowInput, ...] = ()

    def propose_batch(self, windows: Sequence[RuleAssemblyWindowInput]) -> Sequence[RuleAssemblyWindowResult]:
        """Capture complete summaries and return one typed result per window."""
        self.inputs = tuple(windows)
        return tuple(
            RuleAssemblyWindowResult(
                candidate_id=window.candidate_id,
                proposals=[]
                if self._empty
                else [
                    RuleAssemblyProposal(
                        member_population_ids=self._member_ids,
                        assembled_rule=_assembled_rule(window),
                        rationale_vi="Hai chunk nêu hậu quả của cùng một hành vi và cùng điều khoản.",
                    )
                ],
            )
            for window in windows
        )


def _assembled_rule(window: RuleAssemblyWindowInput) -> OntologyRuleEntry:
    """Return one complete typed rule grounded in both supplied records."""
    first_source = window.records[0].evidence[0]
    second_source = window.records[1].evidence[0]
    first = OntologyEvidenceReference(evidence_id=first_source.evidence_id)
    second = OntologyEvidenceReference(evidence_id=second_source.evidence_id)
    return OntologyRuleEntry(
        conduct_label_vi="Điều khiển xe không gắn biển số",
        subject_kind=OntologySubjectKind.NaturalPerson,
        role_labels_vi=["Người điều khiển"],
        vehicle_category_labels_vi=["Xe mô tô"],
        condition_root_id=None,
        conditions=[],
        consequences=[
            OntologyConsequenceEntry(
                consequence_id="money",
                kind=OntologyConsequenceKind.MonetaryPenalty,
                minimum_amount="4000000",
                maximum_amount="6000000",
                currency=OntologyCurrency.VND,
                point_value=None,
                target_label_vi=None,
                responsible_role_label_vi=None,
                evidence=[first],
            ),
            OntologyConsequenceEntry(
                consequence_id="points",
                kind=OntologyConsequenceKind.PointDeduction,
                minimum_amount=None,
                maximum_amount=None,
                currency=None,
                point_value=6,
                target_label_vi="Giấy phép lái xe",
                responsible_role_label_vi=None,
                evidence=[second],
            ),
        ],
        valid_from=None,
        valid_to=None,
        provision_path="Điều 13 khoản 9 điểm b",
        evidence=[
            first,
            second,
        ],
    )


def _propose(
    assembly_records: tuple[OntologyPopulationRecord, ...],
    assembly_chunks: tuple[Chunk, Chunk],
    assembly_metadata: DocumentMetadataRecord,
    client: _AssemblyClient,
) -> tuple:
    """Run proposal generation with retries disabled for tests."""
    settings = replace(
        AssemblySettings.from_path(PROJECT_ROOT / "ontology" / "assembly_contract.yaml"),
        max_retries=0,
        retry_delay_seconds=0,
    )
    candidates = build_assembly_candidates(assembly_records, assembly_chunks, settings)
    return propose_assemblies(
        candidates.windows,
        assembly_records,
        assembly_chunks,
        assembly_metadata,
        client,
        settings,
        run_id="proposal-run",
    )


def test_proposal_preserves_complete_summaries_and_exact_evidence(
    assembly_chunks: tuple[Chunk, Chunk],
    assembly_records: tuple[OntologyPopulationRecord, ...],
    assembly_metadata: DocumentMetadataRecord,
) -> None:
    """The model sees full records and accepted output is rebound to exact source text."""
    client = _AssemblyClient([record.population_id for record in assembly_records])
    first = _propose(assembly_records, assembly_chunks, assembly_metadata, client)
    second = _propose(assembly_records, assembly_chunks, assembly_metadata, client)

    assert first == second
    assert first[0].assembled_record.source_chunk_id == assembly_chunks[0].chunk_id
    assert first[0].assembled_record.ontology_version == "0.2.0"
    assert first[0].assembled_record.ontology_semantic_sha256 == "ontology-semantic-sha256"
    assert first[0].assembled_record.population_run_id == "proposal-run"
    assert [item.exact_text for item in first[0].assembled_record.evidence] == [
        assembly_chunks[0].text,
        assembly_chunks[1].text,
    ]
    assert client.inputs[0].records[0].consequences[0].minimum_amount == "4000000"
    assert client.inputs[0].records[1].consequences[0].point_value == 6


def test_population_summary_preserves_additional_sanction_duration(
    assembly_records: tuple[OntologyPopulationRecord, ...],
) -> None:
    """Assembly input retains the V0.9 sanction label and normalized duration."""
    source = assembly_records[0]
    consequence = replace(
        source.consequences[0],
        kind="AdditionalSanction",
        minimum_amount=None,
        maximum_amount=None,
        currency_code=None,
        consequence_label_vi="Tước quyền sử dụng",
        minimum_duration_months=10,
        maximum_duration_months=12,
        target_label_vi="Giấy phép lái xe",
    )

    summary = summarize_population_record(replace(source, consequences=(consequence,)))

    assert summary.consequences[0].consequence_label_vi == "Tước quyền sử dụng"
    assert summary.consequences[0].minimum_duration_months == 10
    assert summary.consequences[0].maximum_duration_months == 12


@pytest.mark.parametrize("member_ids", [["population-money", "unknown"], ["population-money"]])
def test_proposal_rejects_invalid_candidate_members(
    member_ids: list[str],
    assembly_chunks: tuple[Chunk, Chunk],
    assembly_records: tuple[OntologyPopulationRecord, ...],
    assembly_metadata: DocumentMetadataRecord,
) -> None:
    """Only two or more unique members of the current candidate may be assembled."""
    with pytest.raises(LlmTaskExhaustedError):
        _propose(assembly_records, assembly_chunks, assembly_metadata, _AssemblyClient(member_ids))


def test_empty_proposal_result_is_valid_and_checkpointed(
    assembly_chunks: tuple[Chunk, Chunk],
    assembly_records: tuple[OntologyPopulationRecord, ...],
    assembly_metadata: DocumentMetadataRecord,
) -> None:
    """A model may correctly decline assembly while the candidate still completes."""
    processed: list[object] = []
    settings = replace(
        AssemblySettings.from_path(PROJECT_ROOT / "ontology" / "assembly_contract.yaml"),
        max_retries=0,
        retry_delay_seconds=0,
    )
    candidates = build_assembly_candidates(assembly_records, assembly_chunks, settings)
    proposals = propose_assemblies(
        candidates.windows,
        assembly_records,
        assembly_chunks,
        assembly_metadata,
        _AssemblyClient([], empty=True),
        settings,
        run_id="proposal-run",
        on_batch_processed=processed.extend,
    )

    assert proposals == ()
    assert processed == [(candidates.windows[0].candidate_id, ())]
