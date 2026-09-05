"""Tests for global ontology assembly conflict resolution."""

from __future__ import annotations

import json
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
    OntologyValidityMode,
    RuleAssemblyConflictInput,
    RuleAssemblyConflictResult,
    RuleAssemblyProposal,
)
from legal_rag.benchmark_synthesis.document_metadata import DocumentMetadataRecord
from legal_rag.chunker.models import Chunk
from legal_rag.ontology_population.assembly_config import AssemblySettings
from legal_rag.ontology_population.assembly_finalization import finalize_assemblies
from legal_rag.ontology_population.assembly_models import AssemblyProposal, AssemblyResolutionProgress
from legal_rag.ontology_population.assembly_resolution import (
    build_assembly_proposal_components,
    build_assembly_resolution_rerun_id,
    build_assembly_resolution_run_id,
    resolve_assembly_conflicts,
)
from legal_rag.ontology_population.assembly_resolution_selection import (
    AssemblyResolutionSelection,
    load_assembly_resolution_selection,
    select_assembly_resolution_rerun_components,
)
from legal_rag.ontology_population.models import OntologyPopulationRecord
from legal_rag.settings import Settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _ResolutionClient:
    """Return one globally consistent assembly for each conflict component."""

    def __init__(self) -> None:
        self.inputs: tuple[RuleAssemblyConflictInput, ...] = ()

    def resolve_batch(self, components: Sequence[RuleAssemblyConflictInput]) -> Sequence[RuleAssemblyConflictResult]:
        """Resolve each fixture component into one proposal."""
        self.inputs = tuple(components)
        return tuple(
            RuleAssemblyConflictResult(
                component_id=component.component_id,
                proposals=[
                    RuleAssemblyProposal(
                        member_population_ids=[record.population_id for record in component.source_records],
                        assembled_rule=_assembled_rule(component),
                        rationale_vi="Các đề xuất chồng lấn cùng mô tả một quy tắc xử phạt.",
                    )
                ],
            )
            for component in components
        )


def _proposal(records: tuple[OntologyPopulationRecord, ...], proposal_id: str, candidate_id: str) -> AssemblyProposal:
    """Build one local proposal over the fixture records."""
    return AssemblyProposal(
        proposal_id=proposal_id,
        candidate_id=candidate_id,
        member_population_ids=tuple(record.population_id for record in records),
        assembled_record=replace(records[0], population_id=f"assembled-{proposal_id}"),
        rationale_vi="Đề xuất cục bộ.",
        run_id="local-run",
    )


def _assembled_rule(component: RuleAssemblyConflictInput) -> OntologyRuleEntry:
    """Build one typed rule from all source records in a conflict component."""
    first = component.source_records[0]
    return OntologyRuleEntry(
        conduct_label_vi=first.conduct_label_vi,
        subject_kind=OntologySubjectKind(first.subject_kind) if first.subject_kind else None,
        role_labels_vi=first.role_labels_vi,
        vehicle_category_labels_vi=first.vehicle_category_labels_vi,
        condition_root_id=None,
        conditions=[],
        consequences=[
            OntologyConsequenceEntry(
                consequence_id=consequence.consequence_id,
                kind=OntologyConsequenceKind(consequence.kind),
                minimum_amount=consequence.minimum_amount,
                maximum_amount=consequence.maximum_amount,
                currency=OntologyCurrency(consequence.currency) if consequence.currency else None,
                point_value=consequence.point_value,
                consequence_label_vi=consequence.consequence_label_vi,
                minimum_duration_months=consequence.minimum_duration_months,
                maximum_duration_months=consequence.maximum_duration_months,
                target_label_vi=consequence.target_label_vi,
                responsible_role_label_vi=consequence.responsible_role_label_vi,
                evidence=[OntologyEvidenceReference(evidence_id=item.evidence_id) for item in consequence.evidence],
            )
            for record in component.source_records
            for consequence in record.consequences
        ],
        validity_mode=OntologyValidityMode.ExplicitInterval,
        valid_from=first.valid_from,
        valid_to=first.valid_to,
        provision_path=first.provision_path,
        evidence=[
            OntologyEvidenceReference(evidence_id=item.evidence_id)
            for record in component.source_records
            for item in record.evidence
        ],
    )


def test_components_connect_transitively_over_shared_members(
    assembly_records: tuple[OntologyPopulationRecord, ...],
) -> None:
    """Transitive overlap becomes one global decision rather than pairwise choices."""
    first, second = assembly_records
    third = replace(second, population_id="population-third")
    fourth = replace(second, population_id="population-fourth")
    proposals = (
        _proposal((first, second), "proposal-1", "candidate-1"),
        _proposal((second, third), "proposal-2", "candidate-2"),
        _proposal((third, fourth), "proposal-3", "candidate-3"),
    )

    components = build_assembly_proposal_components(proposals)

    assert len(components) == 1
    assert set(components[0].proposal_ids) == {"proposal-1", "proposal-2", "proposal-3"}
    assert set(components[0].member_population_ids) == {
        first.population_id,
        second.population_id,
        third.population_id,
        fourth.population_id,
    }


def test_resolution_run_id_ignores_batch_size(
    assembly_records: tuple[OntologyPopulationRecord, ...],
) -> None:
    """Resolution checkpoints depend on semantics rather than request grouping."""
    proposal = _proposal(assembly_records, "proposal-1", "candidate-1")
    assembly_settings = AssemblySettings.from_path(PROJECT_ROOT / "ontology" / "assembly_contract.yaml")
    provider_settings = Settings(
        llm_base_url="https://example.test/v1",
        llm_api_key="test-key",
        llm_model="test-model",
        llm_timeout_seconds=30,
    )

    assert build_assembly_resolution_run_id((proposal,), assembly_settings, provider_settings) == (
        build_assembly_resolution_run_id(
            (proposal,),
            replace(assembly_settings, batch_size=5),
            provider_settings,
        )
    )


def test_resolution_rerun_id_is_stable_under_component_order() -> None:
    """A selective replay identity depends on the selected set, not JSON ordering."""
    assert build_assembly_resolution_rerun_id("base-run", ("component-b", "component-a")) == (
        build_assembly_resolution_rerun_id("base-run", ("component-a", "component-b"))
    )


def test_resolution_selection_loads_versioned_component_ids(tmp_path: Path) -> None:
    """The selective-rerun boundary accepts only a typed string-array artifact."""
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps({"schema_version": "0.1.0", "component_ids": ["component-1"]}),
        encoding="utf-8",
    )

    assert load_assembly_resolution_selection(selection_path).component_ids == ("component-1",)


@pytest.mark.parametrize(
    ("selection", "progress_ids", "message"),
    [
        (AssemblyResolutionSelection(("component-1", "component-1")), ("component-1",), "must be unique"),
        (AssemblyResolutionSelection(("unknown",)), ("component-1",), "unknown component"),
        (AssemblyResolutionSelection(("component-1",)), (), "prior progress for every component"),
    ],
)
def test_resolution_selection_rejects_unsafe_replay_state(
    assembly_records: tuple[OntologyPopulationRecord, ...],
    selection: AssemblyResolutionSelection,
    progress_ids: tuple[str, ...],
    message: str,
) -> None:
    """Selective replay cannot discard unknown or unfinished prior decisions."""
    components = build_assembly_proposal_components((_proposal(assembly_records, "proposal-1", "candidate-1"),))
    component_id = components[0].component_id
    normalized_selection = AssemblyResolutionSelection(
        tuple(component_id if item == "component-1" else item for item in selection.component_ids)
    )
    progress = tuple(
        AssemblyResolutionProgress(
            component_id=component_id if item == "component-1" else item,
            run_id="old-run",
        )
        for item in progress_ids
    )

    with pytest.raises(ValueError, match=message):
        select_assembly_resolution_rerun_components(components, progress, normalized_selection)


def test_resolution_replaces_competing_proposals_with_one_global_decision(
    assembly_chunks: tuple[Chunk, Chunk],
    assembly_records: tuple[OntologyPopulationRecord, ...],
    assembly_metadata: DocumentMetadataRecord,
) -> None:
    """The resolver sees competing proposals and emits a disjoint final proposal."""
    proposals = (
        _proposal(assembly_records, "proposal-1", "candidate-1"),
        _proposal(assembly_records, "proposal-2", "candidate-2"),
    )
    components = build_assembly_proposal_components(proposals)
    client = _ResolutionClient()
    processed: list[object] = []
    settings = replace(
        AssemblySettings.from_path(PROJECT_ROOT / "ontology" / "assembly_contract.yaml"),
        max_retries=0,
        retry_delay_seconds=0,
    )

    resolved = resolve_assembly_conflicts(
        components,
        proposals,
        assembly_records,
        assembly_chunks,
        assembly_metadata,
        client,
        settings,
        run_id="resolution-run",
        on_batch_processed=processed.extend,
    )

    assert len(resolved) == 1
    assert resolved[0].candidate_id == components[0].component_id
    assert resolved[0].run_id == "resolution-run"
    assert set(resolved[0].member_population_ids) == {record.population_id for record in assembly_records}
    assert len(client.inputs[0].competing_proposals) == 2
    assert processed == [(components[0].component_id, resolved)]
    assert len(finalize_assemblies(assembly_records, resolved)) == 1
