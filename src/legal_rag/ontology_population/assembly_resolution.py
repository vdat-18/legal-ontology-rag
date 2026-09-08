"""Resolve globally overlapping ontology assembly proposals."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Protocol

from legal_rag.baml_client.sync_client import b
from legal_rag.baml_client.types import (
    RuleAssemblyConflictInput,
    RuleAssemblyConflictProposalInput,
    RuleAssemblyConflictResult,
)
from legal_rag.baml_transport import BamlCaller
from legal_rag.benchmark_synthesis.document_metadata import DocumentMetadataRecord
from legal_rag.chunker.models import Chunk
from legal_rag.ontology_population.assembly_config import AssemblySettings
from legal_rag.ontology_population.assembly_models import AssemblyConflictComponent, AssemblyProposal
from legal_rag.ontology_population.assembly_proposals import build_assembly_proposal, summarize_population_record
from legal_rag.ontology_population.models import OntologyPopulationRecord, population_identity
from legal_rag.ontology_proposal_execution import execute_proposal_batches
from legal_rag.settings import Settings

ONTOLOGY_ASSEMBLY_RESOLUTION_PROMPT_VERSION = "0.6.0"


class AssemblyResolutionClient(Protocol):
    """Boundary implemented by an assembly-conflict language-model client."""

    def resolve_batch(self, components: Sequence[RuleAssemblyConflictInput]) -> Sequence[RuleAssemblyConflictResult]:
        """Return globally consistent proposals for conflict components."""


class BamlAssemblyResolutionClient:
    """Typed BAML conflict resolver backed by an OpenAI-compatible provider."""

    def __init__(self, settings: Settings, assembly_settings: AssemblySettings) -> None:
        """Create a bounded runtime BAML client."""
        self._caller = BamlCaller.from_settings(
            settings,
            max_output_tokens=assembly_settings.max_output_tokens,
            temperature=assembly_settings.temperature,
            max_response_bytes=assembly_settings.max_response_bytes,
        )

    def resolve_batch(self, components: Sequence[RuleAssemblyConflictInput]) -> Sequence[RuleAssemblyConflictResult]:
        """Run one bounded BAML call over conflict components."""
        request = b.request.ResolveOntologyRuleAssemblyConflicts(list(components), baml_options=self._caller.options)
        raw_output = self._caller.execute(request)
        return b.parse.ResolveOntologyRuleAssemblyConflicts(raw_output, baml_options=self._caller.options)


def build_assembly_proposal_components(
    proposals: Sequence[AssemblyProposal],
) -> tuple[AssemblyConflictComponent, ...]:
    """Group proposals connected by any shared population member."""
    proposals_by_id = {proposal.proposal_id: proposal for proposal in proposals}
    if len(proposals_by_id) != len(proposals):
        msg = "Assembly proposal identifiers must be unique"
        raise ValueError(msg)

    owners_by_member: dict[str, list[str]] = defaultdict(list)
    for proposal in proposals:
        for member_id in proposal.member_population_ids:
            owners_by_member[member_id].append(proposal.proposal_id)

    neighbors = {proposal_id: set() for proposal_id in proposals_by_id}
    for owners in owners_by_member.values():
        for proposal_id in owners:
            neighbors[proposal_id].update(owners)

    components = []
    remaining = set(proposals_by_id)
    while remaining:
        pending = [min(remaining)]
        proposal_ids: set[str] = set()
        while pending:
            proposal_id = pending.pop()
            if proposal_id in proposal_ids:
                continue
            proposal_ids.add(proposal_id)
            pending.extend(neighbors[proposal_id] - proposal_ids)
        remaining.difference_update(proposal_ids)
        ordered_proposal_ids = tuple(sorted(proposal_ids))
        member_ids = tuple(
            sorted(
                {
                    member_id
                    for proposal_id in ordered_proposal_ids
                    for member_id in proposals_by_id[proposal_id].member_population_ids
                }
            )
        )
        components.append(
            AssemblyConflictComponent(
                component_id=population_identity({"proposal_ids": ordered_proposal_ids}),
                proposal_ids=ordered_proposal_ids,
                member_population_ids=member_ids,
            )
        )
    return tuple(sorted(components, key=lambda component: component.component_id))


def build_assembly_resolution_run_id(
    proposals: Sequence[AssemblyProposal],
    assembly_settings: AssemblySettings,
    settings: Settings,
) -> str:
    """Fingerprint inputs that can change global assembly resolution."""
    return population_identity(
        {
            "proposals": [proposal.to_dict() for proposal in sorted(proposals, key=lambda item: item.proposal_id)],
            "prompt_version": ONTOLOGY_ASSEMBLY_RESOLUTION_PROMPT_VERSION,
            "max_output_tokens": assembly_settings.max_output_tokens,
            "temperature": assembly_settings.temperature,
            "llm_base_url": settings.llm_base_url,
            "llm_model": settings.llm_model,
            "llm_extra_request_body": dict(settings.llm_extra_request_body),
        }
    )


def build_assembly_resolution_rerun_id(base_run_id: str, component_ids: Sequence[str]) -> str:
    """Fingerprint one explicit subset replay without invalidating retained decisions."""
    return population_identity(
        {
            "stage": "assembly-resolution-selective-rerun",
            "base_run_id": base_run_id,
            "component_ids": sorted(component_ids),
        }
    )


def resolve_assembly_conflicts(
    components: Sequence[AssemblyConflictComponent],
    proposals: Sequence[AssemblyProposal],
    records: Sequence[OntologyPopulationRecord],
    chunks: Sequence[Chunk],
    metadata: DocumentMetadataRecord,
    client: AssemblyResolutionClient,
    assembly_settings: AssemblySettings,
    *,
    run_id: str,
    max_concurrency: int = 1,
    on_batch_processed: Callable[[Sequence[tuple[str, tuple[AssemblyProposal, ...]]]], None] | None = None,
) -> tuple[AssemblyProposal, ...]:
    """Resolve conflicting components in bounded concurrent batches."""
    proposals_by_id = {proposal.proposal_id: proposal for proposal in proposals}
    records_by_id = {record.population_id: record for record in records}
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    tasks = []
    for batch_start in range(0, len(components), assembly_settings.batch_size):
        batch = components[batch_start : batch_start + assembly_settings.batch_size]
        inputs = tuple(_component_input(component, proposals_by_id, records_by_id) for component in batch)
        tasks.append(
            (
                _batch_task_key(inputs, run_id),
                partial(
                    _build_batch_resolutions,
                    client,
                    inputs,
                    batch,
                    records_by_id,
                    chunks_by_id,
                    metadata,
                    run_id,
                ),
            )
        )
    batch_results = execute_proposal_batches(
        tasks,
        stage="ontology rule assembly conflict resolution",
        max_concurrency=max_concurrency,
        max_retries=assembly_settings.max_retries,
        retry_delay_seconds=assembly_settings.retry_delay_seconds,
        on_batch_processed=on_batch_processed,
    )
    return tuple(
        proposal
        for batch_result in batch_results
        for _, component_proposals in batch_result
        for proposal in component_proposals
    )


def _component_input(
    component: AssemblyConflictComponent,
    proposals_by_id: Mapping[str, AssemblyProposal],
    records_by_id: Mapping[str, OntologyPopulationRecord],
) -> RuleAssemblyConflictInput:
    """Build one complete conflict-resolution input."""
    return RuleAssemblyConflictInput(
        component_id=component.component_id,
        source_records=[summarize_population_record(records_by_id[item]) for item in component.member_population_ids],
        competing_proposals=[
            RuleAssemblyConflictProposalInput(
                proposal_id=proposal_id,
                member_population_ids=list(proposals_by_id[proposal_id].member_population_ids),
                assembled_rule=summarize_population_record(proposals_by_id[proposal_id].assembled_record),
                rationale_vi=proposals_by_id[proposal_id].rationale_vi,
            )
            for proposal_id in component.proposal_ids
        ],
    )


def _build_batch_resolutions(
    client: AssemblyResolutionClient,
    inputs: Sequence[RuleAssemblyConflictInput],
    components: Sequence[AssemblyConflictComponent],
    records_by_id: Mapping[str, OntologyPopulationRecord],
    chunks_by_id: Mapping[str, Chunk],
    metadata: DocumentMetadataRecord,
    run_id: str,
) -> tuple[tuple[str, tuple[AssemblyProposal, ...]], ...]:
    """Translate conflict decisions with minimal identity checks."""
    results = client.resolve_batch(inputs)
    requested_ids = [item.component_id for item in inputs]
    result_ids = [result.component_id for result in results]
    if len(result_ids) != len(set(result_ids)) or set(result_ids) != set(requested_ids):
        msg = "Assembly resolution must return exactly one result per conflict component"
        raise ValueError(msg)
    results_by_id = {result.component_id: result for result in results}
    components_by_id = {component.component_id: component for component in components}
    translated = []
    for component_id in requested_ids:
        allowed_members = set(components_by_id[component_id].member_population_ids)
        used_members: set[str] = set()
        component_proposals = []
        for raw in results_by_id[component_id].proposals:
            members = tuple(raw.member_population_ids)
            if len(members) < 2 or len(members) != len(set(members)) or not set(members) <= allowed_members:
                msg = "Resolved assembly members must be unique source population ids"
                raise ValueError(msg)
            if used_members & set(members):
                msg = "Resolved assemblies within one conflict component must not overlap"
                raise ValueError(msg)
            used_members.update(members)
            component_proposals.append(
                build_assembly_proposal(raw, component_id, records_by_id, chunks_by_id, metadata, run_id)
            )
        translated.append((component_id, tuple(component_proposals)))
    return tuple(translated)


def _batch_task_key(inputs: Sequence[RuleAssemblyConflictInput], run_id: str) -> str:
    """Return a deterministic retry key for one resolution batch."""
    component_ids = "\0".join(item.component_id for item in inputs)
    digest = hashlib.sha256(f"{run_id}\0{component_ids}".encode()).hexdigest()
    return f"ontology-assembly-resolution:{digest}"
